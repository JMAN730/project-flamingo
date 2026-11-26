//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

use victory_gpu::resource::{PageMapOp, PageMapping};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_PRIVATE, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQuery,
    VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (the
/// title's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    if at.is_null() {
        let error = std::io::Error::last_os_error();
        if kind & MEM_COMMIT != 0 {
            commit_refused(size, "virtual memory", error);
        } else {
            tracing::warn!(
                size,
                kind = format_args!("{kind:#x}"),
                flags = format_args!("{flags:#x}"),
                %error,
                "XMemVirtualAlloc refused"
            );
        }
    }
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (the title's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (the title uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        tracing::warn!(size, "XMemAllocDefault: the host heap is out of memory");
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            VirtualFree(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;
/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16. The title stores `number >> 4` in 21 bits.
const FRAMES: usize = PAGE >> 12;
const PAGE_NUMBER_LIMIT: usize = (1 << 21) * FRAMES;
const FIRST_PAGE_NUMBER: usize = 0x10_0000;

struct Section {
    handle: HANDLE,
    pages: usize,
    /// A bit per 64 KiB page: committed, as far as any view asked.
    committed: Box<[AtomicU64]>,
}

impl Section {
    fn new(handle: HANDLE, pages: usize) -> Section {
        Section {
            handle,
            pages,
            committed: (0..pages.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Commits the pages from `index` on that a view at `base` shows, unless
    /// all of them are ([`lean`]): committed for one view is for all.
    fn commit_shown(&self, index: usize, base: usize, len: usize) -> bool {
        self.commit_pages(index, base, len, true)
    }

    fn commit_pages(&self, index: usize, base: usize, len: usize, count: bool) -> bool {
        let pages = index..(index + len.div_ceil(PAGE)).min(self.pages);
        let bit = |p: usize| (&self.committed[p / 64], 1u64 << (p % 64));
        let done = || {
            pages.clone().all(|p| {
                let (word, bit) = bit(p);
                word.load(Ordering::Acquire) & bit != 0
            })
        };
        if lean() && done() {
            return true;
        }
        let ok = if count {
            counted(Call::Commit, || commit(base, len))
        } else {
            commit(base, len)
        };
        if !ok {
            return false;
        }
        for p in pages {
            let (word, bit) = bit(p);
            word.fetch_or(bit, Ordering::AcqRel);
        }
        true
    }
}

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

impl Slice {
    /// The key in [`Shown::pages`] of the page `k` pages after this one.
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.section) as usize, self.index + k)
    }
}

#[derive(Default)]
struct Physical {
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Placeholders that are some view's own, left by its unmapping: address
    /// and length. The title may have joined one with others since.
    holes: HashMap<usize, usize>,
}

/// The title's physical pages by number.
#[derive(Default)]
struct Pages {
    next_page: usize,
    free_pages: Vec<usize>,
    pages: HashMap<usize, Slice>,
}

impl Pages {
    fn allocate(&mut self, section: Arc<Section>, numbers: &mut [MaybeUninit<usize>]) -> bool {
        let fresh = numbers.len().saturating_sub(self.free_pages.len());
        if fresh > (PAGE_NUMBER_LIMIT - self.next_page) / FRAMES {
            return false;
        }
        for (index, out) in numbers.iter_mut().enumerate() {
            let number = self.free_pages.pop().unwrap_or_else(|| {
                let number = self.next_page;
                self.next_page += FRAMES;
                number
            });
            self.pages.insert(
                number,
                Slice {
                    section: section.clone(),
                    index,
                },
            );
            out.write(number);
        }
        true
    }

    fn free(&mut self, number: usize) -> Option<Slice> {
        let removed = self.pages.remove(&number);
        if removed.is_some() {
            self.free_pages.push(number);
        }
        removed
    }
}

/// Where the title sees its pages: [`Physical::views`] the other way.
/// Locked apart from [`PHYSICAL`], and only for a moment.
#[derive(Default)]
struct Shown {
    /// The addresses that show a page, by the page's section and index.
    /// Views of more than [`SHORT`] pages are in `long`.
    pages: HashMap<(usize, usize), Vec<usize>>,
    /// Address, bytes, section, first page.
    long: Vec<(usize, usize, usize, usize)>,
}

/// Pages of a view noted page by page in [`Shown::pages`].
const SHORT: usize = 64;

/// Taken after [`PHYSICAL`] where both are.
static SHOWN: LazyLock<RwLock<Shown>> = LazyLock::new(Default::default);

fn shown() -> RwLockReadGuard<'static, Shown> {
    SHOWN.read().unwrap_or_else(|e| e.into_inner())
}

fn shown_mut() -> RwLockWriteGuard<'static, Shown> {
    SHOWN.write().unwrap_or_else(|e| e.into_inner())
}

impl Shown {
    fn note(&mut self, base: usize, len: usize, slice: &Slice) {
        let section = Arc::as_ptr(&slice.section) as usize;
        if len / PAGE > SHORT {
            self.long.push((base, len, section, slice.index));
            return;
        }
        for k in 0..len.div_ceil(PAGE) {
            self.pages
                .entry((section, slice.index + k))
                .or_default()
                .push(base + k * PAGE);
        }
    }

    fn forget(&mut self, base: usize, len: usize, slice: &Slice) {
        if len / PAGE > SHORT {
            self.long.retain(|view| view.0 != base);
            return;
        }
        let section = Arc::as_ptr(&slice.section) as usize;
        for k in 0..len.div_ceil(PAGE) {
            let key = (section, slice.index + k);
            if let Some(at) = self.pages.get_mut(&key) {
                at.retain(|&a| a != base + k * PAGE);
                if at.is_empty() {
                    self.pages.remove(&key);
                }
            }
        }
    }

    /// Guest views touched by a write through a worker's temporary section
    /// view. The live section identifies the bytes even after page-number reuse.
    fn written_views(&self, slice: &Slice, within: usize, bytes: usize) -> Vec<usize> {
        let section = Arc::as_ptr(&slice.section) as usize;
        let first = slice.index + within / PAGE;
        let end = if bytes == 0 {
            first
        } else {
            slice.index + (within + bytes).div_ceil(PAGE)
        };
        let mut out = Vec::new();
        for index in first..end {
            let start_len = out.len();
            if let Some(addresses) = self.pages.get(&(section, index)) {
                out.extend_from_slice(addresses);
            }
            for &(base, len, of, first_page) in &self.long {
                if of == section && (first_page..first_page + len.div_ceil(PAGE)).contains(&index) {
                    out.push(base + (index - first_page) * PAGE);
                }
            }
            out[start_len..].sort_unstable();
        }
        out
    }
}

/// The views, changed by the title's mapping calls; held for as long as
/// those take (the host maps, commits and unmaps under it).
static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(Default::default);

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The title's pages. Apart from the views: DirectStorage looks pages up at
/// every request, from many workers, without waiting for the title's mapping
/// calls. Written only where the title allocates or frees pages. Taken after
/// [`PHYSICAL`] where both are.
static PAGES: LazyLock<RwLock<Pages>> = LazyLock::new(|| {
    RwLock::new(Pages {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: FIRST_PAGE_NUMBER,
        free_pages: Vec::new(),
        pages: HashMap::new(),
    })
});

fn page_table() -> RwLockReadGuard<'static, Pages> {
    PAGES.read().unwrap_or_else(|e| e.into_inner())
}

fn page_table_mut() -> RwLockWriteGuard<'static, Pages> {
    PAGES.write().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    if n > (PAGE_NUMBER_LIMIT - FIRST_PAGE_NUMBER) / FRAMES {
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    // A pagefile-backed section of `size` bytes, its pages committed when
    // first mapped or written (`commit`): the title asks for 6.1
    // GiB at once, which a busy host often cannot commit (error 1450).
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            // Under Wine a committed section is a sparse file, free until
            // touched, and committing a reserved one page by page costs a
            // wineserver request each that scans every range committed so
            // far: commit it whole there.
            PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    let section = Arc::new(Section::new(handle, n));
    let mut local: Vec<MaybeUninit<usize>> = vec![MaybeUninit::uninit(); n];
    let allocated = {
        let mut table = page_table_mut();
        table.allocate(section, &mut local)
    };
    if !allocated {
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    // Title memory is touched without our locks: a fault there may wait for
    // a mapping call, which takes them.
    for (index, number) in local.into_iter().enumerate() {
        // SAFETY: `table.allocate` initialized every slot; `pages` holds `n` entries.
        unsafe { pages.add(index).write(number.assume_init()) };
    }
    1
}

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    // SAFETY: the title passes `n` page numbers; copy before taking any lock.
    let numbers = unsafe { std::slice::from_raw_parts(pages, n) }.to_vec();
    let mut table = page_table_mut();
    // Dropped without the lock: the last page of a section closes its handle.
    let gone: Vec<Slice> = numbers
        .iter()
        .filter_map(|&number| table.free(number))
        .collect();
    drop(table);
    drop(gone);
    1
}

/// A view of our own on consecutive slices of one section, committed and
/// writable, unmapped when dropped.
struct View {
    at: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl View {
    /// Maps `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                slice.section.handle,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View { at, len };
        slice
            .section
            .commit_pages(slice.index, at.Value as usize, len, false)
            .then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(self.at) };
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let table = page_table();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = table.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

#[cfg(test)]
fn pages_known(pages: &[usize], offset: usize, len: usize) -> bool {
    let Some(end) = offset.checked_add(len) else {
        return false;
    };
    let table = page_table();
    pages
        .get(offset / PAGE..end.div_ceil(PAGE))
        .is_some_and(|pages| pages.iter().all(|page| table.pages.contains_key(page)))
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// One view per run of consecutive pages, not one per page: each view
/// costs a map, a commit and an unmap (which flushes every core's TLB).
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = View::map(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        let addresses = shown().written_views(slice, within, k);
        victory_gpu::mapped_pages_written(addresses.into_iter());
        (within, done) = (0, done + k);
    }
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    slice: Slice,
    within: usize,
    len: usize,
}

impl PageWindow {
    pub(crate) fn fill<E>(
        mut self,
        write: impl FnOnce(&mut [u8]) -> Result<usize, E>,
    ) -> Result<(), E> {
        let out = &mut self.view.bytes()[self.within..self.within + self.len];
        let result = write(out);
        // A failed read or decode can leave a partial prefix of unknown size.
        let bytes = result.as_ref().copied().unwrap_or(self.len).min(self.len);
        let addresses = shown().written_views(&self.slice, self.within, bytes);
        victory_gpu::mapped_pages_written(addresses.into_iter());
        result.map(drop)
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    Some(PageWindow {
        view: View::map(slice, *n)?,
        slice: slice.clone(),
        within: offset % PAGE,
        len,
    })
}

/// What the title's mapping calls cost since the last asking, for the
/// streaming log, counted only while that log is on: calls, pages,
/// milliseconds in all, the longest one's and waiting for the lock; the same
/// of the GPU's page-mapping copies ([`map_alias`]; calls and milliseconds);
/// and by system call (unmap, split, map, commit) how many and their
/// milliseconds.
static MAPPING: [AtomicU64; 15] = [const { AtomicU64::new(0) }; 15];

pub(crate) fn map_costs() -> Option<String> {
    let n: [u64; 15] = std::array::from_fn(|i| MAPPING[i].swap(0, Ordering::Relaxed));
    let time = |i: usize| matches!(i, 2 | 3 | 4 | 6 | 8 | 10 | 12 | 14);
    (n[0] + n[5] > 0).then(|| {
        let all: Vec<String> = n
            .iter()
            .enumerate()
            .map(|(i, &n)| if time(i) { n / 1_000_000 } else { n })
            .map(|n| n.to_string())
            .collect();
        all.join("\t")
    })
}

/// The system calls that change a mapping, as [`MAPPING`] counts them.
#[derive(Clone, Copy)]
enum Call {
    Unmap = 7,
    Split = 9,
    Map = 11,
    Commit = 13,
}

/// Runs `f`, a system call, and counts it while the streaming log is on.
fn counted<T>(call: Call, f: impl FnOnce() -> T) -> T {
    if !super::dstorage::stream_log::on() {
        return f();
    }
    let started = std::time::Instant::now();
    let done = f();
    MAPPING[call as usize].fetch_add(1, Ordering::Relaxed);
    MAPPING[call as usize + 1].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    done
}

/// Counts a mapping call when dropped.
struct Costed {
    started: std::time::Instant,
    /// `None`: a copy of mappings.
    pages: Option<usize>,
    waited: u64,
}

impl Costed {
    fn begin(pages: Option<usize>) -> Option<Costed> {
        super::dstorage::stream_log::on().then(|| Costed {
            started: std::time::Instant::now(),
            pages,
            waited: 0,
        })
    }
}

impl Drop for Costed {
    fn drop(&mut self) {
        let ns = self.started.elapsed().as_nanos() as u64;
        let Some(pages) = self.pages else {
            MAPPING[5].fetch_add(1, Ordering::Relaxed);
            MAPPING[6].fetch_add(ns, Ordering::Relaxed);
            return;
        };
        for (i, n) in [1, pages as u64, ns].into_iter().enumerate() {
            MAPPING[i].fetch_add(n, Ordering::Relaxed);
        }
        MAPPING[3].fetch_max(ns, Ordering::Relaxed);
        MAPPING[4].fetch_add(self.waited, Ordering::Relaxed);
    }
}

/// Whether a mapping change asks the host only for what it needs: no split
/// of a placeholder that is already the view's own, no commit of section
/// pages that are committed (`VICTORY_MAP_LEAN=0`: both every time, as
/// before).
fn lean() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("VICTORY_MAP_LEAN").is_none_or(|v| v != "0"));
    *ON
}

/// The title's streaming thread runs below normal priority, as do shader
/// workers when pipelines are made; on the console nothing else wants its
/// core. Called where that thread maps: lifts a thread that is below normal
/// to above it (`VICTORY_STREAM_PRIORITY=0`: left as the title set it).
pub(crate) fn lift_thread() {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, GetThreadPriority, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
    };
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("VICTORY_STREAM_PRIORITY").is_none_or(|v| v != "0"));
    if !*ON {
        return;
    }
    // SAFETY: plain calls on this thread.
    unsafe {
        let thread = GetCurrentThread();
        if matches!(GetThreadPriority(thread), -2..0) {
            SetThreadPriority(thread, THREAD_PRIORITY_ABOVE_NORMAL);
        }
    }
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    let base = address as usize;
    let len = n.saturating_mul(PAGE);
    // Read before any lock is taken: a fault in title memory may wait for
    // the mapping call that holds them.
    // SAFETY: the title passes `n` page numbers, or null.
    let pages =
        (!pages.is_null()).then(|| unsafe { std::slice::from_raw_parts(pages, n) }.to_vec());
    let mut costed = Costed::begin(Some(n));
    let mut physical = physical();
    if let Some(costed) = &mut costed {
        costed.waited = costed.started.elapsed().as_nanos() as u64;
    }
    let (span_at, span_len) = touched_span(&physical, base, len);
    let _changing = victory_gpu::title_memory_remapping(span_at, span_len);
    let slices = pages.as_deref().map(|pages| {
        let table = page_table();
        pages
            .iter()
            .map(|&page| (page, table.pages.get(&page).cloned()))
            .collect::<Vec<_>>()
    });
    if map_pages(&mut physical, base, n, slices.as_deref()) {
        address
    } else {
        std::ptr::null_mut()
    }
}

/// [`XMemMapPhysicalPages`] with the physical state and remapping lock
/// already held: unmaps `[base, base + n pages)`, then maps `slices` there
/// (none: only unmaps). False if a page or the host fails.
fn map_pages(
    physical: &mut Physical,
    base: usize,
    n: usize,
    slices: Option<&[(usize, Option<Slice>)]>,
) -> bool {
    unmap(physical, base, n.saturating_mul(PAGE));
    let Some(slices) = slices else {
        return true;
    };
    let mut i = 0;
    while i < n {
        let Some(first) = slices[i].1.clone() else {
            tracing::warn!(page = slices[i].0, "XMemMapPhysicalPages: unknown page");
            return false;
        };
        // One mapping per run of consecutive slices of one section.
        let run = 1 + slices[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&((_, slice), k)| {
                slice.as_ref().is_some_and(|s| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
                })
            })
            .count();
        if !map(
            physical,
            base + i * PAGE,
            run * PAGE,
            first,
            PageCommitment::Required,
        ) {
            return false;
        }
        i += run;
    }
    true
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (the title passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    lift_thread();
    if n == 0 {
        return 1;
    }
    // Read before any lock is taken: a fault in title memory may wait for
    // the mapping call that holds them.
    // SAFETY: the title passes `n` addresses.
    let addresses = unsafe { std::slice::from_raw_parts(addresses, n) }.to_vec();
    let pages = (!pages.is_null()).then(|| {
        // SAFETY: checked non-null; the title passes `n` page numbers.
        unsafe { std::slice::from_raw_parts(pages, n) }.to_vec()
    });
    let mut costed = Costed::begin(Some(n));
    let mut physical = physical();
    if let Some(costed) = &mut costed {
        costed.waited = costed.started.elapsed().as_nanos() as u64;
    }
    let (span_at, span_len) = touched_scatter_span(&physical, &addresses);
    let _changing = victory_gpu::title_memory_remapping(span_at, span_len);
    let slices = pages.as_deref().map(|pages| {
        let table = page_table();
        pages
            .iter()
            .map(|&page| {
                let slice = (page != 0)
                    .then(|| table.pages.get(&page).cloned())
                    .flatten();
                (page, slice)
            })
            .collect::<Vec<_>>()
    });
    // Keep each mapped scatter address in its own host view on every host
    // (Windows and Wine alike): replacing or unmapping one page of a larger
    // view in `unmap()` calls `UnmapViewOfFile2` on the whole view before
    // remapping its neighbors, briefly unmapping pages other title threads may
    // be reading concurrently. Consecutive unmaps (null `pages` or page 0)
    // still batch into one call over these independent views.
    let mut i = 0;
    while i < n {
        let address = addresses[i] as usize;
        let zero = pages.as_ref().is_some_and(|p| p[i] == 0);
        let mut run = 1;
        let batch = zero || pages.is_none();
        while batch && i + run < n {
            let next = addresses[i + run] as usize;
            let next_zero = pages.as_ref().is_some_and(|p| p[i + run] == 0);
            if next != address + run * PAGE || next_zero != zero {
                break;
            }
            run += 1;
        }
        let ok = if zero || pages.is_none() {
            map_pages(&mut physical, address, run, None)
        } else {
            let slice = slices.as_deref().map(|s| &s[i..i + run]);
            map_pages(&mut physical, address, run, slice)
        };
        #[cfg(feature = "diagnostics")]
        {
            use victory_gpu::stats::{C, add};
            add(C::Scatter, 1);
            add(C::ScatterPages, run as u64);
            add(C::ScatterMaps, 1);
        }
        if !ok {
            if run == 1 {
                return 0;
            }
            for k in i..i + run {
                let addr = addresses[k] as usize;
                let entry = if zero || pages.is_none() {
                    None
                } else {
                    slices.as_deref().map(|s| &s[k..=k])
                };
                if !map_pages(&mut physical, addr, 1, entry) {
                    return 0;
                }
                #[cfg(feature = "diagnostics")]
                {
                    use victory_gpu::stats::{C, add};
                    add(C::ScatterMaps, 1);
                }
            }
        }
        i += run;
    }
    1
}

/// Whether batched `CopyPageMappingsBatchX` coalescing and redundant-remap
/// skipping are enabled (`VICTORY_BATCH_PAGE_MAP=0` restores legacy per-range
/// `map_alias` calls).
fn batch_page_map_enabled() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("VICTORY_BATCH_PAGE_MAP").is_none_or(|v| v != "0"));
    *ON
}

fn lookup_slice(physical: &Physical, at: usize) -> Option<Slice> {
    physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _))| at < *base + *size)
        .map(|(base, (_, s))| Slice {
            index: s.index + (at - *base) / PAGE,
            section: s.section.clone(),
        })
}

fn page_maps_slice(physical: &Physical, at: usize, slice: &Slice) -> bool {
    physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _))| at < *base + *size)
        .is_some_and(|(base, (_, s))| {
            Arc::ptr_eq(&s.section, &slice.section) && s.index + (at - *base) / PAGE == slice.index
        })
}

fn has_mapped_views(physical: &Physical, base: usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }
    let end = base.saturating_add(len);
    physical
        .views
        .range(..=base)
        .next_back()
        .is_some_and(|(at, (size, _))| *at + *size > base)
        || physical.views.range(base + 1..end).next().is_some()
}

fn touched_span(physical: &Physical, base: usize, len: usize) -> (usize, usize) {
    if len == 0 {
        return (0, 0);
    }
    let end = base.saturating_add(len);
    let mut from = base;
    let mut to = end;
    if let Some((&at, &(size, _))) = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + *size > base)
    {
        from = from.min(at);
        to = to.max(at.saturating_add(size));
    }
    if let Some((&at, &(size, _))) = physical.views.range(base + 1..end).next_back() {
        to = to.max(at.saturating_add(size));
    }
    (from, to.saturating_sub(from))
}

fn touched_scatter_span(physical: &Physical, addresses: &[*mut c_void]) -> (usize, usize) {
    let mut from = usize::MAX;
    let mut to = 0usize;
    for &address in addresses {
        let (start, span_len) = touched_span(physical, address as usize, PAGE);
        if span_len > 0 {
            from = from.min(start);
            to = to.max(start.saturating_add(span_len));
        }
    }
    if from < to { (from, to - from) } else { (0, 0) }
}

fn touched_ops_span(physical: &Physical, ops: &[PageMapOp]) -> (usize, usize) {
    let mut from = usize::MAX;
    let mut to = 0usize;
    for op in ops {
        let len = op.len as usize;
        if len == 0 {
            continue;
        }
        let (start, span_len) = touched_span(physical, op.to as usize, len);
        if span_len > 0 {
            from = from.min(start);
            to = to.max(start.saturating_add(span_len));
        }
    }
    if from < to { (from, to - from) } else { (0, 0) }
}

fn ensure_remapping(
    physical: &Physical,
    ops: &[PageMapOp],
    changing: &mut Option<victory_gpu::Remapping>,
) {
    if changing.is_none() {
        let (span_at, span_len) = touched_ops_span(physical, ops);
        *changing = Some(victory_gpu::title_memory_remapping(span_at, span_len));
    }
}

/// Maps the physical pages at `from` at `to` as well, in place of what
/// `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`): the
/// title maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64, mapping: PageMapping) -> bool {
    if !batch_page_map_enabled() {
        return map_alias_legacy(to, from, len, mapping);
    }
    map_alias_batch(&[PageMapOp {
        to,
        from,
        len,
        mapping,
    }])
}

fn map_alias_legacy(to: u64, from: u64, len: u64, mapping: PageMapping) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    if len == 0 {
        return true;
    }
    let _costed = Costed::begin(None);
    let mut physical = physical();
    let (span_at, span_len) = touched_span(&physical, to, len);
    let _changing = victory_gpu::title_memory_remapping(span_at, span_len);
    map_alias_one(&mut physical, to, from, len, mapping)
}

fn map_alias_one(
    physical: &mut Physical,
    to: usize,
    from: usize,
    len: usize,
    mapping: PageMapping,
) -> bool {
    let repeated = matches!(mapping, PageMapping::Repeated);
    if !repeated {
        unmap(physical, to, len);
        if from == 0 {
            return true;
        }
    }
    for k in (0..len).step_by(PAGE) {
        let at = if repeated {
            // Keep the page order, including when the source lies inside
            // the destination or a later page fails.
            unmap(physical, to + k, PAGE);
            if from == 0 {
                continue;
            }
            from
        } else {
            from + k
        };
        let Some(slice) = lookup_slice(physical, at) else {
            return false;
        };
        // The source view already committed this page of the same section.
        if !map(physical, to + k, PAGE, slice, PageCommitment::Existing) {
            return false;
        }
    }
    true
}

/// Applies all ranges of a `CopyPageMappingsBatchX` call under a single
/// `PHYSICAL` lock and at most one `title_memory_remapping()` lock acquisition,
/// skipping no-op unmaps and destination pages already mapped to the exact
/// same physical section slice.
pub fn map_alias_batch(ops: &[PageMapOp]) -> bool {
    if !batch_page_map_enabled() {
        let mut ok = true;
        for op in ops {
            ok &= map_alias_legacy(op.to, op.from, op.len, op.mapping);
        }
        return ok;
    }
    if ops.iter().all(|op| op.len == 0) {
        return true;
    }
    let _costed = Costed::begin(None);
    let mut physical = physical();
    let mut changing: Option<victory_gpu::Remapping> = None;
    let mut ok = true;
    for (op_idx, &op) in ops.iter().enumerate() {
        let remaining = &ops[op_idx..];
        let (to, from, len) = (op.to as usize, op.from as usize, op.len as usize);
        if len == 0 {
            continue;
        }
        if from == 0 {
            if has_mapped_views(&physical, to, len) {
                ensure_remapping(&physical, remaining, &mut changing);
                unmap(&mut physical, to, len);
            }
            continue;
        }
        if !len.is_multiple_of(PAGE) {
            ensure_remapping(&physical, remaining, &mut changing);
            ok &= map_alias_one(&mut physical, to, from, len, op.mapping);
            continue;
        }
        match op.mapping {
            PageMapping::Repeated => {
                let overlaps = from >= to && from < to.saturating_add(len);
                if overlaps {
                    ensure_remapping(&physical, remaining, &mut changing);
                    ok &= map_alias_one(&mut physical, to, from, len, PageMapping::Repeated);
                } else {
                    ok &= map_repeated_nonoverlapping(
                        &mut physical,
                        remaining,
                        &mut changing,
                        to,
                        from,
                        len,
                    );
                }
            }
            PageMapping::Consecutive => {
                let overlaps = from < to.saturating_add(len) && to < from.saturating_add(len);
                if overlaps {
                    ensure_remapping(&physical, remaining, &mut changing);
                    ok &= map_alias_one(&mut physical, to, from, len, PageMapping::Consecutive);
                } else {
                    ok &= map_consecutive_nonoverlapping(
                        &mut physical,
                        remaining,
                        &mut changing,
                        to,
                        from,
                        len,
                    );
                }
            }
        }
    }
    ok
}

fn map_repeated_nonoverlapping(
    physical: &mut Physical,
    remaining: &[PageMapOp],
    changing: &mut Option<victory_gpu::Remapping>,
    to: usize,
    from: usize,
    len: usize,
) -> bool {
    let Some(slice) = lookup_slice(physical, from) else {
        if has_mapped_views(physical, to, PAGE) {
            ensure_remapping(physical, remaining, changing);
            unmap(physical, to, PAGE);
        }
        return false;
    };
    let pages = len / PAGE;
    let mut i = 0;
    while i < pages {
        let same = page_maps_slice(physical, to + i * PAGE, &slice);
        let start = i;
        i += 1;
        while i < pages && page_maps_slice(physical, to + i * PAGE, &slice) == same {
            i += 1;
        }
        let run_start = to + start * PAGE;
        let run_len = (i - start) * PAGE;
        if same {
            victory_gpu::title_memory_alias_rebound(run_start, run_len);
            continue;
        }
        ensure_remapping(physical, remaining, changing);
        if has_mapped_views(physical, run_start, run_len) {
            unmap_inner(physical, run_start, run_len, false);
        }
        let mut mapped_len = 0;
        let mut failed = false;
        for offset in (0..run_len).step_by(PAGE) {
            let dst = run_start + offset;
            if !map_view_raw(physical, dst, PAGE, &slice, PageCommitment::Existing) {
                failed = true;
                break;
            }
            mapped_len += PAGE;
        }
        // Noted before the GPU side is told so concurrent storage writes see
        // the newly mapped addresses immediately.
        if mapped_len > 0 {
            let mut shown = shown_mut();
            let entry = shown.pages.entry(slice.key(0)).or_default();
            entry.reserve(mapped_len / PAGE);
            for offset in (0..mapped_len).step_by(PAGE) {
                entry.push(run_start + offset);
            }
        }
        victory_gpu::title_memory_remapped(run_start, run_len);
        for offset in (0..mapped_len).step_by(PAGE) {
            let dst = run_start + offset;
            physical.views.insert(dst, (PAGE, slice.clone()));
        }
        if failed {
            return false;
        }
    }
    true
}

fn map_consecutive_nonoverlapping(
    physical: &mut Physical,
    remaining: &[PageMapOp],
    changing: &mut Option<victory_gpu::Remapping>,
    to: usize,
    from: usize,
    len: usize,
) -> bool {
    let pages = len / PAGE;
    let mut slices = Vec::with_capacity(pages);
    for i in 0..pages {
        let Some(s) = lookup_slice(physical, from + i * PAGE) else {
            ensure_remapping(physical, remaining, changing);
            return map_alias_one(physical, to, from, len, PageMapping::Consecutive);
        };
        slices.push(s);
    }
    let mut i = 0;
    while i < pages {
        let same = page_maps_slice(physical, to + i * PAGE, &slices[i]);
        let start = i;
        i += 1;
        while i < pages && page_maps_slice(physical, to + i * PAGE, &slices[i]) == same {
            i += 1;
        }
        let run_start = to + start * PAGE;
        let run_len = (i - start) * PAGE;
        if same {
            victory_gpu::title_memory_alias_rebound(run_start, run_len);
            continue;
        }
        ensure_remapping(physical, remaining, changing);
        if has_mapped_views(physical, run_start, run_len) {
            unmap_inner(physical, run_start, run_len, false);
        }
        let mut mapped_end = start;
        for (offset, slice) in slices[start..i].iter().enumerate() {
            let dst = to + (start + offset) * PAGE;
            if !map_view_raw(physical, dst, PAGE, slice, PageCommitment::Existing) {
                break;
            }
            mapped_end += 1;
        }
        if mapped_end > start {
            let mut shown = shown_mut();
            for (offset, slice) in slices[start..mapped_end].iter().enumerate() {
                shown.note(to + (start + offset) * PAGE, PAGE, slice);
            }
        }
        victory_gpu::title_memory_remapped(run_start, run_len);
        for (offset, slice) in slices[start..mapped_end].iter().enumerate() {
            let dst = to + (start + offset) * PAGE;
            physical.views.insert(dst, (PAGE, slice.clone()));
        }
        if mapped_end < i {
            return false;
        }
    }
    true
}

#[derive(Clone, Copy)]
enum PageCommitment {
    Required,
    Existing,
}

fn map_view_raw(
    physical: &mut Physical,
    base: usize,
    len: usize,
    slice: &Slice,
    commitment: PageCommitment,
) -> bool {
    let own = physical.holes.remove(&base) == Some(len) && lean();
    if own && try_map_view(base, len, slice, commitment) {
        return true;
    }
    let end = base + len;
    let first = placeholder(base);
    let split_prefix =
        first.is_some_and(|(start, stop)| start == base && end < stop && split_reserved(base, len));
    if !split_prefix {
        split_at(base, first);
        split_at(end, placeholder(end));
    }
    map_view(base, len, slice, commitment)
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must be placeholders. A view replaces exactly one placeholder, and a
/// reservation is often several by now (each mapping splits it at its
/// ends, unmapping does not join the pieces, titles split their own), so
/// the range is split at its two ends and each placeholder in it gets a
/// view of its part. Placeholders are never joined: that would also join
/// separate reservations, or pieces the title split, for good. All or
/// nothing: if a view fails, those made before it are placeholders again.
fn map(
    physical: &mut Physical,
    base: usize,
    len: usize,
    slice: Slice,
    commitment: PageCommitment,
) -> bool {
    // Fast path: an exact-sized placeholder we just unmapped needs no
    // `VirtualQuery` or `VirtualFree` split unless the title coalesced it.
    if physical.holes.remove(&base) == Some(len)
        && lean()
        && try_map_view(base, len, &slice, commitment)
    {
        shown_mut().note(base, len, &slice);
        victory_gpu::title_memory_remapped(base, len);
        physical.views.insert(base, (len, slice));
        return true;
    }
    let end = base + len;
    let first = placeholder(base);
    // From the allocation base to a point inside the same reserved
    // region, a successful split gives exact bounds without another query.
    let split_prefix =
        first.is_some_and(|(start, stop)| start == base && end < stop && split_reserved(base, len));
    if !split_prefix {
        split_at(base, first);
        split_at(end, placeholder(end));
    }
    let mut made: Vec<(usize, usize, Slice)> = Vec::new();
    let mut at = base;
    while at < end {
        // What is not a placeholder fails in `map_view`.
        let next = if split_prefix {
            end
        } else {
            placeholder(at).map_or(end, |(_, stop)| stop.min(end))
        };
        let offset = at - base;
        let part = Slice {
            index: slice.index + offset / PAGE,
            section: slice.section.clone(),
        };
        // A view must start on a page of the run to show its slice.
        if !offset.is_multiple_of(PAGE) || !map_view(at, next - at, &part, commitment) {
            for &(view, _, _) in &made {
                let view = MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut c_void,
                };
                // SAFETY: a view made above, back to its placeholder.
                unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
            }
            return false;
        }
        made.push((at, next - at, part));
        at = next;
    }
    {
        let mut shown = shown_mut();
        for (at, len, part) in &made {
            shown.note(*at, *len, part);
        }
    }
    for (at, len, part) in made {
        // New contents, read-write: the GPU side's copies no longer guard it.
        victory_gpu::title_memory_remapped(at, len);
        physical.views.insert(at, (len, part));
    }
    true
}

/// Replaces the placeholder `[base, base + len)` with a view of `slice` and
/// the slices after it without logging on miss (used for the `holes` fast path).
fn try_map_view(base: usize, len: usize, slice: &Slice, commitment: PageCommitment) -> bool {
    // SAFETY: `GetCurrentProcess` returns a pseudo-handle.
    let process = unsafe { GetCurrentProcess() };
    // SAFETY: replaces the placeholder `[base, base + len)` with a view of
    // the section; the host refuses anything that is not that placeholder.
    let view = counted(Call::Map, || unsafe {
        MapViewOfFile3(
            slice.section.handle,
            process,
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    });
    if view.Value as usize != base {
        return false;
    }
    if matches!(commitment, PageCommitment::Required)
        && !slice.section.commit_shown(slice.index, base, len)
    {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(process, view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    true
}

/// Replaces the placeholder `[base, base + len)` with a view of `slice` and
/// the slices after it, committed when `commitment` requires it.
fn map_view(base: usize, len: usize, slice: &Slice, commitment: PageCommitment) -> bool {
    if try_map_view(base, len, slice, commitment) {
        return true;
    }
    tracing::warn!(
        address = format_args!("{base:#x}"),
        len,
        error = %std::io::Error::last_os_error(),
        "mapping physical pages failed (not a placeholder?)"
    );
    false
}

#[cfg(test)]
thread_local! {
    static PLACEHOLDER_QUERIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static COMMIT_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The start and end of the placeholder holding `at`, if it may be one
/// (other reserved memory looks the same, but the host refuses to split it
/// or map over it).
fn placeholder(at: usize) -> Option<(usize, usize)> {
    #[cfg(test)]
    PLACEHOLDER_QUERIES.with(|queries| queries.set(queries.get() + 1));
    // SAFETY: plain data; zero is a valid value.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: a query of our own address space into `info`.
    let found = unsafe { VirtualQuery(at as *const c_void, &mut info, size) } == size;
    (found && info.State == MEM_RESERVE && info.Type == MEM_PRIVATE).then(|| {
        (
            info.AllocationBase as usize,
            info.BaseAddress as usize + info.RegionSize,
        )
    })
}

/// Splits the placeholder holding `at` in two there, if it starts below.
/// The same call frees reserved memory that replaced a placeholder back to
/// one, committed pages and all, so everything below `at` in it must be
/// reserved: nothing committed is ever freed here.
fn split_at(at: usize, containing: Option<(usize, usize)>) {
    if let Some((start, _)) = containing
        && start < at
        && placeholder(start).is_some_and(|(_, stop)| stop >= at)
    {
        split_reserved(start, at - start);
    }
}

/// The caller has checked that the whole range, starting at the allocation
/// base, is reserved. A committed allocation must never be freed here.
fn split_reserved(start: usize, len: usize) -> bool {
    // SAFETY: the checked prefix holds no committed data. A successful
    // call makes exactly that prefix a placeholder.
    counted(Call::Split, || unsafe {
        VirtualFree(
            start as *mut c_void,
            len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        ) != 0
    })
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    #[cfg(test)]
    COMMIT_CALLS.with(|calls| calls.set(calls.get() + 1));
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    commit_refused(len, "physical pages", std::io::Error::last_os_error());
    false
}

/// Reports memory the host would not give: at the Windows commit limit to
/// `crashes.log`, since the title usually faults soon after on the null it
/// gets; otherwise as a warning.
fn commit_refused(len: usize, what: &str, error: std::io::Error) {
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(
            len,
            what,
            &error,
            queried.then_some(&info),
        ));
    } else {
        tracing::warn!(len, %error, "committing {what} failed");
    }
}

fn commit_limit_message(
    len: usize,
    what: &str,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of {what}: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    unmap_inner(physical, base, len, true);
}

fn unmap_inner(physical: &mut Physical, base: usize, len: usize, notify_remapped: bool) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + *size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    if hit.is_empty() {
        return;
    }
    let mut removed: Vec<(usize, usize, Slice)> = Vec::with_capacity(hit.len());
    let (mut min_at, mut max_end) = (usize::MAX, 0usize);
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        min_at = min_at.min(at);
        max_end = max_end.max(at + size);
        removed.push((at, size, slice));
    }
    if removed.is_empty() {
        return;
    }
    {
        let mut shown = shown_mut();
        for (at, size, slice) in &removed {
            shown.forget(*at, *size, slice);
        }
    }
    if notify_remapped || min_at < base || max_end > end {
        victory_gpu::title_memory_remapped(min_at, max_end - min_at);
    }
    // SAFETY: pseudo-handle for current process.
    let process = unsafe { GetCurrentProcess() };
    for (at, size, slice) in removed {
        // SAFETY: a view we mapped; it becomes a placeholder again.
        let gone = counted(Call::Unmap, || unsafe {
            UnmapViewOfFile2(
                process,
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        });
        if gone != 0 && at >= base && at + size <= end {
            physical.holes.insert(at, size);
        }
        if at < base {
            map(
                physical,
                at,
                base - at,
                slice.clone(),
                PageCommitment::Existing,
            );
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(
                physical,
                end,
                at + size - end,
                Slice { index, ..slice },
                PageCommitment::Existing,
            );
        }
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, "physical pages", &error, Some(&info));
        assert!(report.contains("4194304 bytes of physical pages"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, "virtual memory", &error, None);
        assert!(fallback.contains("4194304 bytes of virtual memory"));
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        let commits_before_alias = COMMIT_CALLS.with(std::cell::Cell::get);
        assert!(map_alias(
            b as u64,
            at(a, 2) as u64,
            PAGE as u64,
            PageMapping::Consecutive,
        ));
        assert_eq!(
            COMMIT_CALLS.with(std::cell::Cell::get),
            commits_before_alias
        );
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(
            b as u64,
            0,
            PAGE as u64,
            PageMapping::Consecutive
        ));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(
            b as u64,
            at(a, 1) as u64,
            PAGE as u64,
            PageMapping::Consecutive,
        ));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// A scatter of pages at consecutive addresses maps each one:
    /// each address shows its own page, pages asked for in a different order
    /// still land where they are asked for, a zero page in the middle unmaps
    /// only its own address, and an unmap run goes in one call.
    #[test]
    fn scatter_maps_runs_of_consecutive_addresses() {
        let pages = allocate(3);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        let at = |k: usize| a.wrapping_byte_add(k * PAGE);
        for k in 0..3 {
            assert!(write_physical(&pages[k..], 0, &[b'0' + k as u8]));
        }

        // Three pages at consecutive addresses.
        let addresses = [at(0), at(1), at(2)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 3, pages.as_ptr()),
            1
        );
        for k in 0..3 {
            // A later replacement of a neighbor can unmap only its own host view.
            assert_eq!(query(at(k)).host.AllocationBase, at(k));
            assert_eq!(
                // SAFETY: mapped read-write.
                unsafe { at(k).cast::<u8>().read() },
                b'0' + k as u8,
                "page {k} at its address"
            );
        }

        // Ask for them out of order: each still lands where it is named.
        let addresses = [at(2), at(0)];
        let swapped = [pages[0], pages[2]];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 2, swapped.as_ptr()),
            1
        );
        assert_eq!(
            // SAFETY: mapped read-write.
            unsafe { at(2).cast::<u8>().read() },
            b'0'
        );
        assert_eq!(
            // SAFETY: mapped read-write.
            unsafe { at(0).cast::<u8>().read() },
            b'2'
        );

        // A zero page ends the run it is in: its address is unmapped, the
        // page after it still maps.
        let addresses = [at(1), at(2)];
        let mixed = [0usize, pages[1]];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 2, mixed.as_ptr()),
            1
        );
        assert_eq!(query(at(1)).host.State, MEM_RESERVE);
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(2).cast::<u8>().read() }, b'1');

        // Three consecutive addresses without pages unmap as one run.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 2, std::ptr::null()),
            1
        );
        assert_eq!(query(at(1)).host.State, MEM_RESERVE);
        assert_eq!(query(at(2)).host.State, MEM_RESERVE);
        assert_eq!(
            XMemMapPhysicalPagesScatter([at(0)].as_ptr(), 1, std::ptr::null()),
            1
        );
        assert_eq!(query(at(0)).host.State, MEM_RESERVE);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    /// Unmapping or remapping one scatter-mapped page must not briefly unmap
    /// an adjacent scatter-mapped page while another thread reads it.
    #[test]
    fn scatter_neighbor_unmap_does_not_fault_concurrent_reader() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let pages = allocate(4);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        let at = |k: usize| a.wrapping_byte_add(k * PAGE);
        let addresses = [at(0), at(1), at(2), at(3)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 4, pages.as_ptr()),
            1
        );

        let sentinel: u64 = 0x1122_3344_5566_7788;
        let read_addr = at(2).wrapping_byte_add(PAGE - 0x110) as usize;
        // SAFETY: page 2 is mapped read-write and aligned for `u64`.
        unsafe { (read_addr as *mut u64).write(sentinel) };

        let stop = Arc::new(AtomicBool::new(false));
        let reader_stop = stop.clone();
        let reader = std::thread::spawn(move || {
            let mut reads = 0u64;
            while !reader_stop.load(Ordering::Relaxed) {
                // SAFETY: page 2 stays mapped for the lifetime of this thread.
                let value = unsafe { (read_addr as *const u64).read_volatile() };
                assert_eq!(value, sentinel);
                reads += 1;
            }
            reads
        });

        let neighbor = [at(3)];
        let unmap_page = [0usize];
        let remap_page = [pages[3]];
        for _ in 0..256 {
            assert_eq!(
                XMemMapPhysicalPagesScatter(neighbor.as_ptr(), 1, unmap_page.as_ptr()),
                1
            );
            assert_eq!(
                XMemMapPhysicalPagesScatter(neighbor.as_ptr(), 1, remap_page.as_ptr()),
                1
            );
        }

        stop.store(true, Ordering::Relaxed);
        let reads = reader.join().expect("reader thread should not fault");
        assert!(reads > 0);

        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 4, std::ptr::null()),
            1
        );
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0, PageMapping::Consecutive));
            assert!(map_alias(at as u64, 0, 0, PageMapping::Consecutive));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = page_table().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn freed_page_numbers_can_be_reused_at_the_limit() {
        let section = || {
            let numbers = allocate(1);
            let section = page_table().pages[&numbers[0]].section.clone();
            assert_eq!(XMemFreePhysicalPages(1, numbers.as_ptr()), 1);
            section
        };
        let mut table = Pages {
            next_page: PAGE_NUMBER_LIMIT - FRAMES,
            ..Pages::default()
        };
        let mut number = [MaybeUninit::new(0)];
        assert!(table.allocate(section(), &mut number));
        // SAFETY: the initialized slot was filled by allocate.
        let last = unsafe { number[0].assume_init() };
        assert_eq!(last >> 4, (1 << 21) - 1);
        assert!(!table.allocate(section(), &mut number));

        let mut old_view = View::map(&table.pages[&last], 1).unwrap();
        old_view.bytes()[0] = 7;
        table.free(last);
        table.free(last);
        // A second free must not grant another use of the same number.
        let mut two = [MaybeUninit::new(0); 2];
        assert!(!table.allocate(section(), &mut two));
        assert!(table.allocate(section(), &mut number));
        // SAFETY: the initialized slot was filled by allocate.
        assert_eq!(unsafe { number[0].assume_init() }, last);
        let mut new_view = View::map(&table.pages[&last], 1).unwrap();
        assert_eq!(new_view.bytes()[0], 0);
        assert_eq!(old_view.bytes()[0], 7);
    }

    /// Streaming allocate/free must reuse page numbers. Without a free list,
    /// `next_page` would climb by `n * FRAMES` each round and eventually pack
    /// past the title's 21-bit `number >> 4` field. Uses a private `Physical`
    /// so parallel suite tests cannot steal freed numbers or bump `next_page`.
    #[test]
    fn allocate_reuses_freed_page_numbers_without_growing_next_page() {
        let section = || {
            let size = 2 * PAGE;
            // SAFETY: plain pagefile-backed section, same as the public allocate path.
            let handle = unsafe {
                CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
                    0,
                    size as u32,
                    std::ptr::null(),
                )
            };
            assert!(!handle.is_null());
            Arc::new(Section::new(handle, 2))
        };
        let mut table = Pages {
            next_page: FIRST_PAGE_NUMBER,
            ..Pages::default()
        };
        let mut slots = [MaybeUninit::new(0); 2];
        assert!(table.allocate(section(), &mut slots));
        // SAFETY: both slots were filled by allocate.
        let pages = unsafe { [slots[0].assume_init(), slots[1].assume_init()] };
        let ceiling = table.next_page;
        for &p in &pages {
            table.free(p);
        }
        for _ in 0..1024 {
            assert!(table.allocate(section(), &mut slots));
            assert_eq!(table.next_page, ceiling);
            // SAFETY: both slots were filled by allocate.
            let again = unsafe { [slots[0].assume_init(), slots[1].assume_init()] };
            for &p in &again {
                assert!(p >> 4 < 1 << 21, "{p:#x}");
                assert!(p > 0);
            }
            // Same numbers we freed (LIFO may reverse them within the batch).
            let mut got = again;
            got.sort_unstable();
            let mut want = pages;
            want.sort_unstable();
            assert_eq!(got, want);
            for &p in &again {
                table.free(p);
            }
        }
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        page_window(&a, PAGE + 3, PAGE)
            .expect("one run")
            .fill(|out| {
                assert_eq!(out.len(), PAGE);
                out.fill(0x5a);
                Ok::<_, ()>(out.len())
            })
            .unwrap();
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// `n` physical pages (consecutive slices of one section) and a
    /// reservation of `n` 64 KiB pages for them.
    fn pages_and_reservation(n: usize) -> (Vec<usize>, *mut c_void) {
        let mut pages = vec![0usize; n];
        let mut count = n;
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut count, pages.as_mut_ptr()),
            1
        );
        let reservation = XMemVirtualAlloc(
            std::ptr::null_mut(),
            n * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!reservation.is_null());
        (pages, reservation)
    }

    #[test]
    fn storage_aliases_follow_partial_remaps() {
        let (pages, a) = pages_and_reservation(3);
        let (other, b) = pages_and_reservation(1);
        let middle = a.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        assert_eq!(XMemMapPhysicalPages(b, 1, pages[1..].as_ptr()), b);
        let slice = page_table().pages[&pages[0]].clone();
        let addresses = |within, bytes| shown().written_views(&slice, within, bytes);
        let mut both = [middle as usize, b as usize];
        both.sort_unstable();
        assert_eq!(addresses(PAGE + 17, 1), both);
        assert!(addresses(PAGE + 17, 0).is_empty());
        assert_eq!(XMemMapPhysicalPages(b, 1, other.as_ptr()), b);
        assert_eq!(addresses(PAGE - 1, 2), [a as usize, middle as usize]);
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        assert!(addresses(PAGE, 1).is_empty());
        assert_eq!(addresses(2 * PAGE, 1), [a as usize + 2 * PAGE]);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemMapPhysicalPages(b, 1, std::ptr::null()), b);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, other.as_ptr()), 1);
    }

    /// Unmapping part of a run leaves the range as several host
    /// placeholders; mapping a run over all of them again works, and the
    /// pages keep what was written through the first mapping.
    #[test]
    fn a_run_maps_again_after_a_partial_unmap() {
        let (pages, a) = pages_and_reservation(3);
        let at = |page: usize| a.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe {
            for page in 0..3 {
                at(page).cast::<u8>().write(10 + page as u8);
            }
        }
        assert_eq!(XMemMapPhysicalPages(at(1), 1, std::ptr::null()), at(1));
        assert_eq!(query(at(1)).host.State, MEM_RESERVE);

        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        for page in 0..3 {
            assert_eq!(query(at(page)).host.State, MEM_COMMIT, "page {page}");
            // SAFETY: mapped read-write.
            assert_eq!(unsafe { at(page).cast::<u8>().read() }, 10 + page as u8);
        }
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(query(a).host.State, MEM_RESERVE);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    /// A run that starts inside one host placeholder and ends inside the
    /// next (left by an earlier, shorter run) maps, and the pages around it
    /// stay unmapped.
    #[test]
    fn a_run_maps_across_two_placeholders() {
        let (pages, a) = pages_and_reservation(4);
        let at = |page: usize| a.wrapping_byte_add(page * PAGE);
        // Mapping and unmapping pages 0-1 splits the reservation in two.
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        assert_eq!(XMemMapPhysicalPages(a, 2, std::ptr::null()), a);

        assert_eq!(XMemMapPhysicalPages(at(1), 2, pages[1..].as_ptr()), at(1));
        assert_eq!(query(a).host.State, MEM_RESERVE);
        assert_eq!(query(at(1)).host.State, MEM_COMMIT);
        assert_eq!(query(at(2)).host.State, MEM_COMMIT);
        assert_eq!(query(at(3)).host.State, MEM_RESERVE);
        // Page 3, split off by the run, still takes a mapping: the run's
        // last page shows there too.
        // SAFETY: mapped read-write.
        unsafe {
            at(2).cast::<u8>().write(42);
            assert_eq!(XMemMapPhysicalPages(at(3), 1, pages[2..].as_ptr()), at(3));
            assert_eq!(at(3).cast::<u8>().read(), 42);
        }
        assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    /// Placeholders a run spans are never joined: a piece the title split
    /// off itself is still its own after the run is unmapped, so the title
    /// can release it.
    #[test]
    fn a_run_keeps_the_placeholders_it_spans() {
        let (pages, a) = pages_and_reservation(4);
        let at = |page: usize| a.wrapping_byte_add(page * PAGE);
        // SAFETY: splits the reservation's placeholder, as a title does.
        let split = unsafe { VirtualFree(a, 2 * PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) };
        assert_ne!(split, 0);
        assert_eq!(XMemMapPhysicalPages(a, 4, pages.as_ptr()), a);
        assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        let info = query(at(2)).host;
        assert_eq!((info.AllocationBase, info.RegionSize), (at(2), 2 * PAGE));
        // SAFETY: the title's own piece, unmapped.
        assert_ne!(unsafe { VirtualFree(at(2), 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    /// A run whose second part is no placeholder (memory committed in place
    /// of one) fails: its first part is unmapped again, and the committed
    /// memory, which the run ends inside, keeps its data.
    #[test]
    fn a_failed_run_leaves_no_view() {
        let (pages, a) = pages_and_reservation(3);
        let at = |page: usize| a.wrapping_byte_add(page * PAGE);
        // SAFETY: splits the placeholder, makes pages 1-2 an ordinary
        // allocation in place and commits page 1 of it.
        unsafe {
            assert_ne!(
                VirtualFree(a, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER),
                0
            );
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                at(1),
                2 * PAGE,
                MEM_RESERVE | MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(p, at(1));
            assert!(!VirtualAlloc(at(1), PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            at(1).cast::<u8>().write(99);
        }
        assert!(XMemMapPhysicalPages(a, 2, pages.as_ptr()).is_null());
        assert_eq!(query(a).host.State, MEM_RESERVE);
        assert_eq!(query(at(1)).host.State, MEM_COMMIT);
        // SAFETY: still committed.
        assert_eq!(unsafe { at(1).cast::<u8>().read() }, 99);
        // Page 0 is a placeholder again: it takes a mapping.
        assert_eq!(XMemMapPhysicalPages(a, 1, pages.as_ptr()), a);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn a_reserved_prefix_uses_one_placeholder_query() {
        let (pages, base) = pages_and_reservation(2);
        PLACEHOLDER_QUERIES.with(|queries| queries.set(0));
        assert_eq!(XMemMapPhysicalPages(base, 1, pages.as_ptr()), base);
        assert_eq!(PLACEHOLDER_QUERIES.with(std::cell::Cell::get), 1);

        assert_eq!(XMemMapPhysicalPages(base, 1, std::ptr::null()), base);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
        // SAFETY: release the test's two placeholders after unmapping.
        unsafe {
            assert_ne!(VirtualFree(base, 0, MEM_RELEASE), 0);
            assert_ne!(VirtualFree(base.wrapping_byte_add(PAGE), 0, MEM_RELEASE), 0);
        }
    }

    /// Splitting a reserved prefix must leave its committed suffix intact,
    /// even if the host accepts the split on a replacement allocation.
    #[test]
    fn a_reserved_prefix_keeps_the_committed_suffix() {
        let (pages, base) = pages_and_reservation(3);
        let suffix = base.wrapping_byte_add(2 * PAGE);
        // SAFETY: replace our placeholder with a private allocation and
        // commit its last page.
        unsafe {
            assert_eq!(
                VirtualAlloc2(
                    GetCurrentProcess(),
                    base,
                    3 * PAGE,
                    MEM_RESERVE | MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                ),
                base
            );
            assert_eq!(
                VirtualAlloc(suffix, PAGE, MEM_COMMIT, PAGE_READWRITE),
                suffix
            );
            suffix.cast::<u8>().write(99);
        }

        let mapped = XMemMapPhysicalPages(base, 1, pages.as_ptr());
        if !mapped.is_null() {
            assert_eq!(mapped, base);
            assert_eq!(XMemMapPhysicalPages(base, 1, std::ptr::null()), base);
        }
        assert_eq!(query(base).host.State, MEM_RESERVE);
        let suffix_info = query(suffix).host;
        assert_eq!(suffix_info.State, MEM_COMMIT);
        // SAFETY: the suffix is still committed.
        assert_eq!(unsafe { suffix.cast::<u8>().read() }, 99);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        // SAFETY: release the test's allocation, which may now be split.
        unsafe {
            if suffix_info.AllocationBase != base {
                assert_ne!(VirtualFree(suffix_info.AllocationBase, 0, MEM_RELEASE), 0);
            }
            assert_ne!(VirtualFree(base, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn batched_page_mappings_skip_noop_unmaps_and_identical_aliases_without_blocking_readers() {
        let pages = allocate(2);
        let pool = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        let heap = XMemVirtualAlloc(
            std::ptr::null_mut(),
            8 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!pool.is_null() && !heap.is_null());
        assert_eq!(XMemMapPhysicalPages(pool, 2, pages.as_ptr()), pool);
        // SAFETY: pool pages are mapped read-write.
        unsafe {
            pool.cast::<u8>().write(0x11);
            pool.wrapping_byte_add(PAGE).cast::<u8>().write(0x22);
        }

        // Initial batch maps `heap[0..4]` repeatedly to `pool[0]` and leaves `heap[4..8]` unmapped.
        let initial = [
            PageMapOp {
                to: heap as u64,
                from: pool as u64,
                len: (4 * PAGE) as u64,
                mapping: PageMapping::Repeated,
            },
            PageMapOp {
                to: (heap as usize + 4 * PAGE) as u64,
                from: 0,
                len: (4 * PAGE) as u64,
                mapping: PageMapping::Consecutive,
            },
        ];
        assert!(map_alias_batch(&initial));
        for k in 0..4 {
            // SAFETY: mapped read-write above.
            assert_eq!(unsafe { heap.cast::<u8>().add(k * PAGE).read() }, 0x11);
        }
        assert_eq!(
            query(heap.wrapping_byte_add(4 * PAGE)).host.State,
            MEM_RESERVE
        );

        // Re-applying identical repeated aliases and unmapping already-unmapped pages
        // must not acquire `title_memory_remapping()` when batched page mapping is
        // enabled (holding `title_memory_reading()` on the same thread would deadlock).
        if batch_page_map_enabled() {
            let _reading = victory_gpu::title_memory_reading(heap as usize, 8 * PAGE);
            assert!(map_alias_batch(&initial));
        } else {
            assert!(map_alias_batch(&initial));
        }

        // Now remap `heap[1..3]` to `pool[1]` and unmap `heap[3]`.
        let update = [
            PageMapOp {
                to: (heap as usize + PAGE) as u64,
                from: (pool as usize + PAGE) as u64,
                len: (2 * PAGE) as u64,
                mapping: PageMapping::Repeated,
            },
            PageMapOp {
                to: (heap as usize + 3 * PAGE) as u64,
                from: 0,
                len: PAGE as u64,
                mapping: PageMapping::Consecutive,
            },
        ];
        assert!(map_alias_batch(&update));
        // SAFETY: mapped read-write above.
        unsafe {
            assert_eq!(heap.cast::<u8>().read(), 0x11);
            assert_eq!(heap.cast::<u8>().add(PAGE).read(), 0x22);
            assert_eq!(heap.cast::<u8>().add(2 * PAGE).read(), 0x22);
        }
        assert_eq!(
            query(heap.wrapping_byte_add(3 * PAGE)).host.State,
            MEM_RESERVE
        );

        // Verify `SHOWN` tracks the pool view and the two repeated heap aliases.
        let slice1 = page_table().pages[&pages[1]].clone();
        let mut views = shown().written_views(&slice1, 0, PAGE);
        views.sort_unstable();
        let mut expected = vec![
            pool as usize + PAGE,
            heap as usize + PAGE,
            heap as usize + 2 * PAGE,
        ];
        expected.sort_unstable();
        assert_eq!(views, expected);

        assert!(map_alias_batch(&[PageMapOp {
            to: heap as u64,
            from: 0,
            len: (8 * PAGE) as u64,
            mapping: PageMapping::Consecutive,
        }]));
        assert_eq!(XMemMapPhysicalPages(pool, 2, std::ptr::null()), pool);
        for p in [pool, heap] {
            for k in 0..(if p == pool { 2 } else { 8 }) {
                // SAFETY: placeholders unmapped above.
                unsafe { VirtualFree(p.wrapping_byte_add(k * PAGE), 0, MEM_RELEASE) };
            }
        }
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// Pages keep what they hold however often they are mapped, moved and
    /// unmapped without a commit or a split of their own.
    #[test]
    fn pages_mapped_again_hold_what_they_held() {
        let pages = allocate(3);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            8 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        let at = |k: usize| a.wrapping_byte_add(k * PAGE);
        let mut commits_after_round0 = 0;
        for round in 0..6_usize {
            for (k, page) in pages.iter().enumerate() {
                let to = at((round + 2 * k) % 8);
                assert_eq!(XMemMapPhysicalPages(to, 1, page), to);
                let first = to.cast::<u64>();
                // SAFETY: mapped read-write; the far end of the page too.
                unsafe {
                    if round > 0 {
                        assert_eq!(first.read(), (round * 16 + k - 16) as u64);
                        assert_eq!(first.add(PAGE / 8 - 1).read(), k as u64);
                    }
                    first.write((round * 16 + k) as u64);
                    first.add(PAGE / 8 - 1).write(k as u64);
                }
            }
            if round == 0 {
                commits_after_round0 = COMMIT_CALLS.with(std::cell::Cell::get);
            } else if lean() {
                assert_eq!(
                    COMMIT_CALLS.with(std::cell::Cell::get),
                    commits_after_round0,
                    "remapping committed pages should skip VirtualAlloc(MEM_COMMIT)"
                );
            }
            for k in 0..pages.len() {
                let to = at((round + 2 * k) % 8);
                assert_eq!(XMemMapPhysicalPages(to, 1, std::ptr::null()), to);
                assert_eq!(query(to).host.State, MEM_RESERVE);
            }
            // The title joins what it unmapped: the next view splits it anew.
            if round == 3 {
                // SAFETY: placeholders of ours, nothing mapped in them.
                unsafe {
                    VirtualFree(
                        a,
                        8 * PAGE,
                        MEM_RELEASE | 1, /* MEM_COALESCE_PLACEHOLDERS */
                    )
                };
            }
        }
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn lift_thread_raises_below_normal_priority() {
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, GetThreadPriority, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
            THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_NORMAL,
        };
        std::thread::spawn(|| {
            // SAFETY: plain thread priority queries and updates on the current thread.
            unsafe {
                let thread = GetCurrentThread();
                assert_ne!(SetThreadPriority(thread, THREAD_PRIORITY_BELOW_NORMAL), 0);
                lift_thread();
                assert_eq!(GetThreadPriority(thread), THREAD_PRIORITY_ABOVE_NORMAL);
                assert_ne!(SetThreadPriority(thread, THREAD_PRIORITY_NORMAL), 0);
                lift_thread();
                assert_eq!(GetThreadPriority(thread), THREAD_PRIORITY_NORMAL);
            }
        })
        .join()
        .unwrap();
    }

    #[test]
    fn pages_known_checks_allocated_pages_under_read_lock() {
        let pages = allocate(2);
        assert!(pages_known(&pages, 0, 2 * PAGE));
        assert!(pages_known(&pages, PAGE - 1, 2));
        assert!(!pages_known(&pages, 0, 3 * PAGE));
        assert!(!pages_known(&[pages[0], 0x10], 0, 2 * PAGE));
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }
}
