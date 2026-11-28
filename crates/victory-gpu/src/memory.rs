//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy), plus a raw SRV of the directory and a raw UAV of the faults
//!   buffer for one-view root descriptor tables;
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. Before a dispatch the host makes ready everything it can
//! see the dispatch name: each root view, each pair of root constants that
//! is a title address and each root constant that is one in 256-byte units
//! readies the whole committed range of the title allocation around it
//! ([`ready_span`]: a structure the title builds in its heap is longer than
//! the window a pointer to it lands in), and the 4 MiB after it. A page a
//! shader still found empty readies its own allocation the same way, so a
//! fault costs one frame's accesses once per allocation, not per page.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (4 MiB windows mostly: 4 GiB at
//! most). Every dispatch of such a shader brings all numbered copies up to
//! date and, if it can store, declares all of them written (the bytes are
//! compared on the CPU afterwards, as for any unordered access view): a
//! dirty-page mask written by the shader would bound both.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots holds the null view,
/// `COPIES` copy views, the directory SRV, and the faults UAV).
const COPIES: u32 = 1023;
/// Total persistent dynamic slots reserved per view heap.
const HEAP_SLOTS: u32 = COPIES + 3;
/// Offset from `base` to the directory raw SRV slot.
const DIRECTORY_SLOT_OFFSET: u32 = COPIES + 1;
/// Offset from `base` to the faults raw UAV slot.
const FAULTS_SLOT_OFFSET: u32 = COPIES + 2;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;
/// The most of one allocation that one address readies whole.
const REGION_CAP: u64 = 256 << 20;
/// A region of an allocation: start, length, committed.
type Region = (u64, u64, bool);
/// Past the cap: the aligned block around the address and the next one.
const BLOCK: u64 = 64 << 20;

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
    /// Ranges [`Memory::name`] has readied whole (kept for the run, as the
    /// copies are).
    readied: Vec<(u64, u64)>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    /// GPU virtual address of the directory buffer (when root SRVs are
    /// unbounded root descriptors).
    pub(crate) directory: u64,
    /// Descriptor-table handle of the directory raw SRV (when root SRVs are
    /// bounded one-view tables).
    pub(crate) directory_table: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// Descriptor-table handle of the faults raw UAV.
    pub(crate) faults_table: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

/// Returns the directory page index for `address`, or `None` if it falls at
/// or past the reserved sentinel entry (`MEMORY_PAGES - 1`).
fn page_index(address: u64) -> Option<u64> {
    let page = address / PAGE;
    (page < u64::from(MEMORY_PAGES) - 1).then_some(page)
}

/// The directory pages a copy of the `len` bytes at title address `first`
/// holds whole, each with its byte offset in the copy; none at or past the
/// sentinel entry.
fn fill_pages(first: u64, len: u64) -> impl Iterator<Item = (u64, u32)> {
    let end = first.saturating_add(len);
    let start = first.checked_next_multiple_of(PAGE).unwrap_or(end);
    (start..end)
        .step_by(PAGE as usize)
        .take_while(move |&at| at.checked_add(PAGE).is_some_and(|e| e <= end))
        .take_while(|&at| page_index(at).is_some())
        .map(move |at| (at / PAGE, (at - first) as u32))
}

/// The next 4 MiB cell boundary after `address`.
fn next_cell(address: u64) -> u64 {
    (address & !(CELL - 1)).wrapping_add(CELL)
}

/// Candidate title addresses encoded in root constants: a 40-bit address in
/// two adjacent constants `(low, high)` or a 256-byte-aligned address in one
/// constant (`c << 8`).
fn constant_addresses(constants: &[u32]) -> Vec<u64> {
    let mut out = Vec::new();
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            out.push(u64::from(c) | u64::from(high) << 32);
        }
        if c >= 0x0100_0000 {
            out.push(u64::from(c) << 8);
        }
    }
    out
}

/// What naming `address` readies, from the regions of the allocation that
/// holds it (`(start, len, committed)`, in address order): the committed
/// run of contiguous regions around it, or past [`REGION_CAP`] the
/// [`BLOCK`]-aligned block around it and the next one, inside that run.
/// None when `address` lies in no committed region listed.
fn ready_span(regions: &[Region], address: u64) -> Option<(u64, u64)> {
    let at = regions
        .iter()
        .position(|&(s, l, _)| s <= address && address - s < l)?;
    if !regions[at].2 {
        return None;
    }
    let joined = |a: Region, b: Region| a.2 && b.2 && a.0 + a.1 == b.0;
    let mut first = at;
    while first > 0 && joined(regions[first - 1], regions[first]) {
        first -= 1;
    }
    let mut last = at;
    while last + 1 < regions.len() && joined(regions[last], regions[last + 1]) {
        last += 1;
    }
    let (start, end) = (
        regions[first].0,
        regions[last].0.saturating_add(regions[last].1),
    );
    if end - start <= REGION_CAP {
        return Some((start, end));
    }
    let block = address & !(BLOCK - 1);
    Some((block.max(start), block.saturating_add(2 * BLOCK).min(end)))
}

/// Page numbers as sorted runs of title addresses `(first byte, end)`.
fn page_runs(pages: &[u32]) -> Vec<(u64, u64)> {
    let mut sorted = pages.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for page in sorted {
        let at = u64::from(page) * PAGE;
        match runs.last_mut() {
            Some(run) if run.1 == at => run.1 = at + PAGE,
            _ => runs.push((at, at + PAGE)),
        }
    }
    runs
}

/// The regions of the allocation around `address` (see [`ready_span`])
/// within [`REGION_CAP`] of it, and the allocation's base and type.
fn allocation_regions(address: u64) -> (Vec<Region>, Option<(u64, u32)>) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let query = |at: u64| {
        // SAFETY: plain data, all zeros is a valid value.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a query of an address of this process into a local of
        // the size given.
        let n = unsafe {
            VirtualQuery(
                at as usize as *const core::ffi::c_void,
                &mut info,
                size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        (n != 0).then_some(info)
    };
    let Some(here) = query(address) else {
        return (Vec::new(), None);
    };
    let base = here.AllocationBase as u64;
    if base == 0 {
        return (Vec::new(), None);
    }
    let mut regions = Vec::new();
    let mut at = base.max(address.saturating_sub(REGION_CAP));
    let stop = address.saturating_add(REGION_CAP);
    while at < stop && regions.len() < 4096 {
        let Some(info) = query(at) else { break };
        if info.AllocationBase as u64 != base || info.RegionSize == 0 {
            break;
        }
        let start = info.BaseAddress as u64;
        regions.push((start, info.RegionSize as u64, info.State == MEM_COMMIT));
        at = start + info.RegionSize as u64;
    }
    (regions, Some((base, here.Type)))
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(u64::from(MEMORY_FAULTS), "faults")?,
            copies: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
            readied: Vec::new(),
        })
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's.
    fn map(&mut self, address: u64) {
        let Some(page) = page_index(address) else {
            return;
        };
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        if unsafe { entry.read() } != 0 {
            return;
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let Some(shadow) = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))
        else {
            return;
        };
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return;
        }
        let n = match self.copies.iter().position(|c| Arc::ptr_eq(c, &shadow)) {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                self.copies.push(shadow.clone());
                let n = self.copies.len() - 1;
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                for (page, offset) in fill_pages(first, len) {
                    // SAFETY: inside the directory's mapping
                    // ([`fill_pages`] stops before the sentinel).
                    let other = unsafe { self.directory.1.add(page as usize * 2) };
                    // SAFETY: as above; the offset first, as below.
                    unsafe {
                        if other.read() == 0 {
                            other.add(1).write_volatile(offset);
                            other.write_volatile(n as u32 + 1);
                        }
                    }
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros")
                });
                return;
            }
        };
        let offset = shadow.offset_in(start).unwrap_or(0);
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping.
        unsafe {
            entry.add(1).write_volatile(offset as u32);
            entry.write_volatile(n as u32 + 1);
        }
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                // The runs, each with the allocation it lies in: a run in a
                // range a dispatch named should not be here.
                let runs = page_runs(&pages);
                let mut text = String::new();
                for &(start, end) in runs.iter().take(8) {
                    let (regions, alloc) = allocation_regions(start);
                    let near: u64 = regions.iter().map(|r| r.1).sum();
                    let named = self.readied.iter().any(|&(s, e)| s <= start && start < e);
                    text.push_str(&match alloc {
                        Some((base, kind)) => format!(
                            " {start:#x}..{end:#x} (allocation {base:#x}, {near:#x} bytes near, type {kind:#x}{})",
                            if named { ", named before" } else { "" }
                        ),
                        None => format!(" {start:#x}..{end:#x} (no allocation)"),
                    });
                }
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    runs = runs.len(),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now:{text}"
                );
            }
        }
        for page in pages {
            self.name(u64::from(page) * PAGE);
        }
    }

    /// What a dispatch names: the committed range of the allocation around
    /// `address` ([`ready_span`]), at least the copy around it, and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        if !self
            .readied
            .iter()
            .any(|&(s, e)| s <= address && address < e)
        {
            let (regions, _) = allocation_regions(address);
            if let Some((start, end)) = ready_span(&regions, address) {
                let mut at = start & !(PAGE - 1);
                while at < end {
                    self.map(at.max(start));
                    at += PAGE;
                }
                self.readied.push((start, end));
            }
        }
        self.map(address);
        self.map(next_cell(address));
    }
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.take_faults();
    }
    let numbered = m.copies.len();
    for &address in addresses {
        m.name(address);
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for address in constant_addresses(constants) {
        m.name(address);
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(HEAP_SLOTS)?;
            heap.null_raw_uav_at(base);
            heap.raw_resource_at(
                base + DIRECTORY_SLOT_OFFSET,
                &m.directory.0,
                0,
                u64::from(MEMORY_PAGES) * 8,
            );
            heap.raw_uav_at(
                base + FAULTS_SLOT_OFFSET,
                &m.faults.0,
                0,
                u64::from(MEMORY_FAULTS) * 4,
            );
            (base, 0)
        }
    };
    for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
        let (_, len) = shadow.span();
        heap.raw_shadow_uav_at(base + 1 + n as u32, shadow, 0, len);
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    // Once per list, and for copies numbered since: the builder alone
    // dispatches hundreds of times a frame.
    // shortcut: a copy another list numbers meanwhile is brought up to date
    // by that list's submission, which may run after this one's.
    const READ: usize = usize::MAX - 0x6d;
    const WRITE: usize = usize::MAX - 0x6e;
    let first_read = list.first_use(READ);
    let first_write = writes && list.first_use(WRITE);
    let refresh_from = if first_read { 0 } else { numbered };
    if refresh_from < m.copies.len() {
        list.defer_refresh_batch(
            m.copies.len() - refresh_from,
            m.copies[refresh_from..].iter().map(|shadow| {
                let (start, len) = shadow.span();
                (shadow, start, len, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            }),
        );
    }
    if writes {
        let write_from = if first_write { 0 } else { numbered };
        for shadow in &m.copies[write_from..] {
            let (start, len) = shadow.span();
            shadow.gpu_writes(list, start, len);
        }
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        directory_table: heap.gpu_handle(base + DIRECTORY_SLOT_OFFSET),
        faults_table: heap.gpu_handle(base + FAULTS_SLOT_OFFSET),
        base,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_index_maps_64k_pages_and_reserves_sentinel() {
        assert_eq!(page_index(0), Some(0));
        assert_eq!(page_index(0xFFFF), Some(0));
        assert_eq!(page_index(0x1_0000), Some(1));
        assert_eq!(page_index(0x12_3456_7890), Some(0x12_3456));
        let last_valid = (u64::from(MEMORY_PAGES) - 2) * PAGE;
        assert_eq!(page_index(last_valid), Some(u64::from(MEMORY_PAGES) - 2));
        let sentinel = (u64::from(MEMORY_PAGES) - 1) * PAGE;
        assert_eq!(page_index(sentinel), None);
        assert_eq!(page_index(u64::MAX), None);
    }

    #[test]
    fn fill_pages_lists_the_whole_pages_a_copy_holds() {
        // Aligned: every page, offsets from the copy's start.
        let got: Vec<_> = fill_pages(0x20_0000, 3 * PAGE).collect();
        assert_eq!(got, [(0x20, 0), (0x21, 0x1_0000), (0x22, 0x2_0000)]);
        // Unaligned: the partial pages at both ends are left out.
        let got: Vec<_> = fill_pages(0x20_0100, 3 * PAGE).collect();
        assert_eq!(got, [(0x21, 0xFF00), (0x22, 0x1_FF00)]);
        // Smaller than a page: none.
        assert_eq!(fill_pages(0x20_0100, PAGE).count(), 0);
        // Never the sentinel entry, nor past the directory.
        let last = (u64::from(MEMORY_PAGES) - 2) * PAGE;
        let got: Vec<_> = fill_pages(last, 4 * PAGE).collect();
        assert_eq!(got, [(u64::from(MEMORY_PAGES) - 2, 0)]);
        assert_eq!(fill_pages(u64::MAX - PAGE, PAGE).count(), 0);
    }

    #[test]
    fn next_cell_advances_to_following_4mib_window() {
        assert_eq!(next_cell(0), CELL);
        assert_eq!(next_cell(0x1234), CELL);
        assert_eq!(next_cell(CELL - 1), CELL);
        assert_eq!(next_cell(CELL), 2 * CELL);
        assert_eq!(next_cell(0x12_3456_7800), 0x12_3480_0000);
    }

    #[test]
    fn constant_addresses_extracts_pairs_and_shifted_bases() {
        // Low + high pair (0x12_3456_7800) plus shifted 256-byte base (0x3456_7800 << 8).
        let got = constant_addresses(&[0x3456_7800, 0x12, 0x0000_0005, 0x0200_0000]);
        assert_eq!(got, vec![0x12_3456_7800, 0x34_5678_0000, 0x02_0000_0000]);
        // Small constants are ignored.
        assert!(constant_addresses(&[0, 1, 0xFFFF, 0x100]).is_empty());
    }

    const M: u64 = 1 << 20;

    #[test]
    fn ready_span_is_the_committed_run_around_the_address() {
        let regions = [
            (0x1000_0000, 4 * M, true),
            (0x1000_0000 + 4 * M, 12 * M, true),
            (0x1000_0000 + 16 * M, M, false),
            (0x1000_0000 + 17 * M, 2 * M, true),
        ];
        // A 13 MiB structure from 0x10300000 is readied whole with its run,
        // not only the 8 MiB the address's window and the next one hold.
        assert_eq!(
            ready_span(&regions, 0x1030_0000),
            Some((0x1000_0000, 0x1000_0000 + 16 * M))
        );
        // Past the reserved gap: its own run only.
        assert_eq!(
            ready_span(&regions, 0x1000_0000 + 18 * M),
            Some((0x1000_0000 + 17 * M, 0x1000_0000 + 19 * M))
        );
        // Reserved, or outside every region: nothing.
        assert_eq!(ready_span(&regions, 0x1000_0000 + 16 * M), None);
        assert_eq!(ready_span(&regions, 0x2000_0000), None);
        assert_eq!(ready_span(&[], 0x1000), None);
    }

    #[test]
    fn ready_span_past_the_cap_is_two_blocks_from_the_address() {
        let regions = [(0, 1 << 30, true)];
        assert_eq!(
            ready_span(&regions, 0x1234_5678),
            Some((0x1000_0000, 0x1000_0000 + 2 * BLOCK))
        );
        // At the run's end the second block is cut there.
        assert_eq!(
            ready_span(&regions, (1 << 30) - 1),
            Some(((1 << 30) - BLOCK, 1 << 30))
        );
    }

    /// An allocation of 24 MiB with its first 16 committed: an address
    /// 13 MiB in readies those 16, and none of the reserved rest.
    #[test]
    fn an_address_readies_the_committed_part_of_its_allocation() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        // SAFETY: a fresh reservation, committed in part, released below.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                (24 * M) as usize,
                MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        assert!(!base.is_null());
        // SAFETY: inside the reservation.
        let done = unsafe { VirtualAlloc(base, (16 * M) as usize, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!done.is_null());
        let start = base as u64;
        let (regions, alloc) = allocation_regions(start + 13 * M);
        assert_eq!(alloc.map(|a| a.0), Some(start));
        assert_eq!(
            ready_span(&regions, start + 13 * M),
            Some((start, start + 16 * M))
        );
        assert_eq!(ready_span(&regions, start + 20 * M), None);
        // SAFETY: the reservation made above.
        unsafe { VirtualFree(base, 0, MEM_RELEASE) };
    }

    #[test]
    fn page_runs_join_neighbouring_pages() {
        assert_eq!(
            page_runs(&[0x52fc5, 0x52fc4, 0x52fc6, 0x2, 0x52fc4]),
            [(0x2 * PAGE, 0x3 * PAGE), (0x52fc4 * PAGE, 0x52fc7 * PAGE)]
        );
        assert!(page_runs(&[]).is_empty());
    }

    #[test]
    fn heap_slot_layout_fits_null_copies_directory_and_faults() {
        assert_eq!(DIRECTORY_SLOT_OFFSET, COPIES + 1);
        assert_eq!(FAULTS_SLOT_OFFSET, COPIES + 2);
        assert_eq!(HEAP_SLOTS, FAULTS_SLOT_OFFSET + 1);
    }

    #[test]
    fn defer_refresh_batch_refreshes_and_deduplicates_multiple_shadows() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let mut buf_a = vec![0x11u8; 4096];
        let mut buf_b = vec![0x22u8; 4096];
        let addr_a = buf_a.as_ptr() as u64;
        let addr_b = buf_b.as_ptr() as u64;
        crate::shadow::register(addr_a as usize, addr_a, buf_a.len() as u64);
        crate::shadow::register(addr_b as usize, addr_b, buf_b.len() as u64);

        let shadow_a = crate::shadow::find(addr_a, 256).expect("shadow A");
        let shadow_b = crate::shadow::find(addr_b, 256).expect("shadow B");

        buf_a[..64].fill(0xA5);
        buf_b[..64].fill(0x5A);

        let list = host.new_list().expect("host list");
        // Include duplicate range and mixed SRV/UAV state transition on shadow A.
        list.defer_refresh_batch(
            3,
            [
                (&shadow_a, addr_a, 64, D3D12_RESOURCE_STATE_GENERIC_READ),
                (&shadow_a, addr_a, 64, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                (&shadow_b, addr_b, 64, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ],
        );
        list.close();
        host.execute(&[&list]);

        assert_eq!(shadow_a.bytes(addr_a, 64).expect("mapped A"), &[0xA5; 64]);
        assert_eq!(shadow_b.bytes(addr_b, 64).expect("mapped B"), &[0x5A; 64]);
        // SAFETY: plain device health query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        drop(shadow_a);
        drop(shadow_b);
        crate::shadow::unregister(addr_a as usize, addr_a);
        crate::shadow::unregister(addr_b as usize, addr_b);
    }
}
