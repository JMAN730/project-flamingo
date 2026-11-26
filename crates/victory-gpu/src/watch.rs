//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;
/// The 4 KiB parts of a 64 KiB page.
const PARTS: usize = PAGE / KERNEL_PAGE;
/// All 16 parts of a 64 KiB page.
const WHOLE: u16 = u16::MAX;

/// Whether the 4 KiB parts of a page that `GetWriteWatch` names are kept so
/// refreshes copy only the written 4 KiB sub-pages (`VICTORY_PAGE_PARTS=0`:
/// copy the whole 64 KiB page, as before).
fn parts_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PAGE_PARTS").is_none_or(|v| v != "0"))
}

/// Whether `Tracked::take` narrows previously-taken pages to their written
/// 4 KiB sub-pages (`VICTORY_TAKE_PARTS=0`: whole 64 KiB pages).
#[allow(dead_code)]
fn take_parts_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TAKE_PARTS").is_none_or(|v| v != "0"))
}

/// Whether a write fault opens only the view that faulted and leaves other
/// alias views read-only (marked unclean) (`VICTORY_OPEN_ONE_VIEW=0`: opens
/// every alias view).
fn open_one_view() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_OPEN_ONE_VIEW").is_none_or(|v| v != "0"))
}

/// How long a view block just opened by a CPU write fault rests writable
/// before refreshes guard it again (`VICTORY_GUARD_REST=<microseconds>`,
/// 3,000 unless given; `0`: no rest).
fn rest() -> std::time::Duration {
    #[cfg(test)]
    {
        REST_IN_TEST.get()
    }
    #[cfg(not(test))]
    {
        static US: OnceLock<u64> = OnceLock::new();
        let us = *US.get_or_init(|| match std::env::var("VICTORY_GUARD_REST").as_deref() {
            Ok("0") => 0,
            Ok(v) => v
                .trim()
                .parse::<u64>()
                .ok()
                .filter(|&us| us > 1)
                .unwrap_or(3000),
            Err(_) => 3000,
        });
        std::time::Duration::from_micros(us)
    }
}

#[cfg(test)]
thread_local! {
    /// [`rest`], for what this test thread asks (0 by default so existing tests
    /// see immediate re-guarding unless opted in).
    static REST_IN_TEST: std::cell::Cell<std::time::Duration> = const { std::cell::Cell::new(std::time::Duration::ZERO) };
}

/// Of pages previously copied (page number, epoch taken): the 4 KiB parts
/// written since, a bit per 4 KiB, in the same order.
pub(crate) fn parts_written(pages: &[(usize, u64)]) -> Vec<u16> {
    if !parts_on() {
        return vec![WHOLE; pages.len()];
    }
    pages
        .iter()
        .map(|&(page, taken)| DIR.parts_since(page, taken))
        .collect()
}

/// Master switch for Milestone 4 watch redesign (lock-free `PageDir`,
/// flat cold-page `PAGE_READONLY` promotion, and contiguous `VirtualProtect`
/// coalescing). Set `VICTORY_WATCH_V2=0` to opt out cleanly.
fn v2_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WATCH_V2").is_none_or(|v| v != "0"))
}

/// Whether contiguous `VirtualProtect` adaptive coalescing is enabled
/// (`VICTORY_WATCH_COALESCE=0` or `VICTORY_WATCH_V2=0` disables it).
fn coalesce_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    v2_enabled()
        && *ON.get_or_init(|| std::env::var_os("VICTORY_WATCH_COALESCE").is_none_or(|v| v != "0"))
}

/// Whether lock-free `PageDir` fast paths are enabled
/// (`VICTORY_WATCH_LOCKFREE=0` or `VICTORY_WATCH_V2=0` disables them).
fn lockfree_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    v2_enabled()
        && *ON.get_or_init(|| std::env::var_os("VICTORY_WATCH_LOCKFREE").is_none_or(|v| v != "0"))
}

/// Default distinct clean frames before a write-watched 64 KiB page promotes
/// to `PAGE_READONLY` under `VICTORY_WATCH_V2=1`.
const DEFAULT_COLD_FRAMES: u32 = 6;

/// `VICTORY_WATCH_COLD=<n>`: distinct clean frames before a write-watched
/// 64 KiB page promotes to `PAGE_READONLY` VEH guard (`6` by default under
/// M4; `0` disables cold promotion, and `VICTORY_WATCH_V2=0` defaults to `0`).
fn cold_after() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        if !v2_enabled() {
            return std::env::var("VICTORY_WATCH_COLD")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0);
        }
        std::env::var("VICTORY_WATCH_COLD")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(DEFAULT_COLD_FRAMES)
    })
}

#[inline]
fn current_frame32() -> u32 {
    (crate::frame::current() as u32).wrapping_add(1).max(1)
}

/// A title-memory range whose changed pages another subsystem needs to read.
#[allow(dead_code)]
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
#[allow(dead_code)]
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

// ---------------------------------------------------------------------------
// 3-Level Lock-Free Radix Page Directory (256 TiB 48-bit VA space)
// ---------------------------------------------------------------------------

const CHUNK_BITS: usize = 10; // 1024 x 64 KiB pages = 64 MiB per leaf chunk
const CHUNK_PAGES: usize = 1 << CHUNK_BITS;
const CHUNK_WORDS: usize = CHUNK_PAGES / 64; // 16 u64 words
const MID_BITS: usize = 10; // 1024 leaf chunks = 64 GiB per mid node
const MID_LEN: usize = 1 << MID_BITS;
const TOP_BITS: usize = 12; // 4096 mid nodes = 256 TiB total VA space
const TOP_LEN: usize = 1 << TOP_BITS;
const MAX_RADIX_PAGES: usize = TOP_LEN * MID_LEN * CHUNK_PAGES;

#[derive(Clone, Copy)]
enum BitField {
    Watched,
    ColdRo,
    GuardProtected,
    GuardClean,
    GuardAbsent,
    GuardAliased,
    GuardViewed,
    LazyArmed,
}

struct LeafChunk {
    written_bits: [AtomicU64; CHUNK_WORDS],
    written_epoch: [AtomicU64; CHUNK_PAGES],
    /// Latest epoch at which the whole 64 KiB page was marked dirty.
    whole_epoch: [AtomicU64; CHUNK_PAGES],
    /// Lazily allocated per-4 KiB sub-page write epochs (only for chunks that
    /// receive partial `GetWriteWatch` writes).
    parts_epoch: AtomicPtr<[[AtomicU64; PARTS]; CHUNK_PAGES]>,
    watched_bits: [AtomicU64; CHUNK_WORDS],
    cold_ro_bits: [AtomicU64; CHUNK_WORDS],
    cold_streak: [AtomicU8; CHUNK_PAGES],
    cold_last_frame: [AtomicU32; CHUNK_PAGES],
    guard_protected_bits: [AtomicU64; CHUNK_WORDS],
    guard_clean_bits: [AtomicU64; CHUNK_WORDS],
    guard_absent_bits: [AtomicU64; CHUNK_WORDS],
    guard_aliased_bits: [AtomicU64; CHUNK_WORDS],
    guard_viewed_bits: [AtomicU64; CHUNK_WORDS],
    lazy_armed_bits: [AtomicU64; CHUNK_WORDS],
}

impl LeafChunk {
    fn new() -> Self {
        Self {
            written_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
            written_epoch: [const { AtomicU64::new(0) }; CHUNK_PAGES],
            whole_epoch: [const { AtomicU64::new(0) }; CHUNK_PAGES],
            parts_epoch: AtomicPtr::new(std::ptr::null_mut()),
            watched_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
            cold_ro_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
            cold_streak: [const { AtomicU8::new(0) }; CHUNK_PAGES],
            cold_last_frame: [const { AtomicU32::new(0) }; CHUNK_PAGES],
            guard_protected_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
            guard_clean_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
            guard_absent_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
            guard_aliased_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
            guard_viewed_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
            lazy_armed_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
        }
    }

    fn get_or_alloc_parts(&self) -> &[[AtomicU64; PARTS]; CHUNK_PAGES] {
        let mut ptr = self.parts_epoch.load(Ordering::Acquire);
        if ptr.is_null() {
            let mut v: Vec<[AtomicU64; PARTS]> = Vec::with_capacity(CHUNK_PAGES);
            v.resize_with(CHUNK_PAGES, || [const { AtomicU64::new(0) }; PARTS]);
            let boxed: Box<[[AtomicU64; PARTS]; CHUNK_PAGES]> = v
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!("exact chunk length"));
            let fresh = Box::into_raw(boxed);
            match self.parts_epoch.compare_exchange(
                std::ptr::null_mut(),
                fresh,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => ptr = fresh,
                Err(existing) => {
                    // SAFETY: `fresh` was never published to any other thread.
                    unsafe { drop(Box::from_raw(fresh)) };
                    ptr = existing;
                }
            }
        }
        // SAFETY: `ptr` is non-null and permanently live once published.
        unsafe { &*ptr }
    }

    #[inline]
    fn bits(&self, field: BitField) -> &[AtomicU64; CHUNK_WORDS] {
        match field {
            BitField::Watched => &self.watched_bits,
            BitField::ColdRo => &self.cold_ro_bits,
            BitField::GuardProtected => &self.guard_protected_bits,
            BitField::GuardClean => &self.guard_clean_bits,
            BitField::GuardAbsent => &self.guard_absent_bits,
            BitField::GuardAliased => &self.guard_aliased_bits,
            BitField::GuardViewed => &self.guard_viewed_bits,
            BitField::LazyArmed => &self.lazy_armed_bits,
        }
    }
}

struct MidNode {
    leaves: [AtomicPtr<LeafChunk>; MID_LEN],
}

impl MidNode {
    fn new() -> Self {
        Self {
            leaves: [const { AtomicPtr::new(std::ptr::null_mut()) }; MID_LEN],
        }
    }
}

struct PageDir {
    top: [AtomicPtr<MidNode>; TOP_LEN],
}

static DIR: PageDir = PageDir {
    top: [const { AtomicPtr::new(std::ptr::null_mut()) }; TOP_LEN],
};

#[inline]
fn word_mask(lo: usize, hi: usize) -> u64 {
    let start_mask = !0u64 << lo;
    let end_mask = if hi == 64 { !0u64 } else { (1u64 << hi) - 1 };
    start_mask & end_mask
}

#[inline]
fn for_each_word_in_chunk(lo_idx: usize, hi_idx: usize, mut f: impl FnMut(usize, u64) -> bool) {
    let w0 = lo_idx / 64;
    let w1 = (hi_idx - 1) / 64;
    for w in w0..=w1 {
        let lo_bit = if w == w0 { lo_idx % 64 } else { 0 };
        let hi_bit = if w == w1 { ((hi_idx - 1) % 64) + 1 } else { 64 };
        if !f(w, word_mask(lo_bit, hi_bit)) {
            break;
        }
    }
}

impl PageDir {
    #[inline]
    fn split_page(page: usize) -> Option<(usize, usize, usize)> {
        if page >= MAX_RADIX_PAGES {
            return None;
        }
        let idx = page & (CHUNK_PAGES - 1);
        let mid = (page >> CHUNK_BITS) & (MID_LEN - 1);
        let top = page >> (CHUNK_BITS + MID_BITS);
        Some((top, mid, idx))
    }

    #[inline]
    fn get(&self, page: usize) -> Option<(&LeafChunk, usize)> {
        let (top_i, mid_i, idx) = Self::split_page(page)?;
        let mid_ptr = self.top[top_i].load(Ordering::Acquire);
        if mid_ptr.is_null() {
            return None;
        }
        // SAFETY: `MidNode` is never freed once published into `self.top`.
        let leaf_ptr = unsafe { (*mid_ptr).leaves[mid_i].load(Ordering::Acquire) };
        if leaf_ptr.is_null() {
            return None;
        }
        // SAFETY: `LeafChunk` is never freed once published into `MidNode`.
        Some((unsafe { &*leaf_ptr }, idx))
    }

    fn get_or_alloc(&self, page: usize) -> Option<(&LeafChunk, usize)> {
        let (top_i, mid_i, idx) = Self::split_page(page)?;
        let mut mid_ptr = self.top[top_i].load(Ordering::Acquire);
        if mid_ptr.is_null() {
            let fresh = Box::into_raw(Box::new(MidNode::new()));
            match self.top[top_i].compare_exchange(
                std::ptr::null_mut(),
                fresh,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => mid_ptr = fresh,
                Err(existing) => {
                    // SAFETY: `fresh` was never published to any other thread.
                    unsafe { drop(Box::from_raw(fresh)) };
                    mid_ptr = existing;
                }
            }
        }
        // SAFETY: `mid_ptr` is non-null and permanently live.
        let leaf_slot = unsafe { &(*mid_ptr).leaves[mid_i] };
        let mut leaf_ptr = leaf_slot.load(Ordering::Acquire);
        if leaf_ptr.is_null() {
            let fresh = Box::into_raw(Box::new(LeafChunk::new()));
            match leaf_slot.compare_exchange(
                std::ptr::null_mut(),
                fresh,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => leaf_ptr = fresh,
                Err(existing) => {
                    // SAFETY: `fresh` was never published to any other thread.
                    unsafe { drop(Box::from_raw(fresh)) };
                    leaf_ptr = existing;
                }
            }
        }
        // SAFETY: `leaf_ptr` is non-null and permanently live.
        Some((unsafe { &*leaf_ptr }, idx))
    }

    fn for_each_chunk(
        &self,
        first: usize,
        last: usize,
        create: bool,
        mut f: impl FnMut(usize, &LeafChunk, usize, usize) -> bool,
    ) {
        let mut cur = first.min(MAX_RADIX_PAGES);
        let end = last.min(MAX_RADIX_PAGES);
        while cur < end {
            let chunk_page = cur & !(CHUNK_PAGES - 1);
            let next_chunk = (chunk_page + CHUNK_PAGES).min(end);
            let lo_idx = cur - chunk_page;
            let hi_idx = next_chunk - chunk_page;
            let leaf = if create {
                self.get_or_alloc(cur).map(|(l, _)| l)
            } else {
                self.get(cur).map(|(l, _)| l)
            };
            if let Some(leaf) = leaf {
                if !f(chunk_page, leaf, lo_idx, hi_idx) {
                    break;
                }
            } else {
                let top_i = cur >> (MID_BITS + CHUNK_BITS);
                if top_i < TOP_LEN && self.top[top_i].load(Ordering::Acquire).is_null() {
                    cur = ((top_i + 1) << (MID_BITS + CHUNK_BITS)).min(end);
                    continue;
                }
            }
            cur = next_chunk;
        }
    }

    #[inline]
    fn get_bit(&self, page: usize, field: BitField, order: Ordering) -> bool {
        let Some((leaf, idx)) = self.get(page) else {
            return false;
        };
        let (w, bit) = (idx / 64, 1u64 << (idx % 64));
        leaf.bits(field)[w].load(order) & bit != 0
    }

    #[inline]
    fn set_bit(&self, page: usize, field: BitField, on: bool, order: Ordering) {
        let Some((leaf, idx)) = (if on {
            self.get_or_alloc(page)
        } else {
            self.get(page)
        }) else {
            return;
        };
        let (w, bit) = (idx / 64, 1u64 << (idx % 64));
        if on {
            leaf.bits(field)[w].fetch_or(bit, order);
        } else {
            leaf.bits(field)[w].fetch_and(!bit, order);
        }
    }

    #[inline]
    fn clear_bit_was_set(&self, page: usize, field: BitField, order: Ordering) -> bool {
        let Some((leaf, idx)) = self.get(page) else {
            return false;
        };
        let (w, bit) = (idx / 64, 1u64 << (idx % 64));
        leaf.bits(field)[w].fetch_and(!bit, order) & bit != 0
    }

    fn set_bits_range(
        &self,
        first: usize,
        last: usize,
        field: BitField,
        on: bool,
        order: Ordering,
    ) {
        if first >= last {
            return;
        }
        self.for_each_chunk(first, last, on, |_, leaf, lo_idx, hi_idx| {
            let words = leaf.bits(field);
            for_each_word_in_chunk(lo_idx, hi_idx, |w, mask| {
                if on {
                    words[w].fetch_or(mask, order);
                } else {
                    words[w].fetch_and(!mask, order);
                }
                true
            });
            true
        });
    }

    #[inline]
    fn record_written(&self, page: usize, epoch: u64) {
        self.record_written_parts(page, WHOLE, epoch);
    }

    #[inline]
    fn record_written_parts(&self, page: usize, parts: u16, epoch: u64) {
        if epoch == 0 || parts == 0 {
            return;
        }
        if let Some((leaf, idx)) = self.get_or_alloc(page) {
            if parts == WHOLE || !parts_on() {
                leaf.whole_epoch[idx].fetch_max(epoch, Ordering::AcqRel);
            } else {
                let table = leaf.get_or_alloc_parts();
                let mut mask = parts;
                while mask != 0 {
                    let k = mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    table[idx][k].fetch_max(epoch, Ordering::AcqRel);
                }
            }
            leaf.written_epoch[idx].fetch_max(epoch, Ordering::AcqRel);
            leaf.written_bits[idx / 64].fetch_or(1u64 << (idx % 64), Ordering::Release);
        }
    }

    #[inline]
    fn record_written_prev(&self, page: usize, epoch: u64) -> u64 {
        let Some((leaf, idx)) = self.get_or_alloc(page) else {
            return 0;
        };
        leaf.whole_epoch[idx].fetch_max(epoch, Ordering::AcqRel);
        let prev = leaf.written_epoch[idx].fetch_max(epoch, Ordering::AcqRel);
        leaf.written_bits[idx / 64].fetch_or(1u64 << (idx % 64), Ordering::Release);
        prev
    }

    /// The 4 KiB parts of `page` written at an epoch strictly after `taken`.
    fn parts_since(&self, page: usize, taken: u64) -> u16 {
        if !parts_on() || taken == 0 {
            return WHOLE;
        }
        let Some((leaf, idx)) = self.get(page) else {
            return WHOLE;
        };
        if leaf.whole_epoch[idx].load(Ordering::Acquire) > taken {
            return WHOLE;
        }
        let ptr = leaf.parts_epoch.load(Ordering::Acquire);
        if ptr.is_null() {
            return if leaf.written_epoch[idx].load(Ordering::Acquire) > taken {
                WHOLE
            } else {
                0
            };
        }
        // SAFETY: `ptr` is non-null and permanently live once published.
        let slots = unsafe { &(*ptr)[idx] };
        let mut mask = 0u16;
        for (k, at) in slots.iter().enumerate() {
            if at.load(Ordering::Acquire) > taken {
                mask |= 1u16 << k;
            }
        }
        mask
    }

    fn collect_written(&self, first: usize, last: usize) -> (Vec<(usize, u64)>, u64) {
        let mut out = Vec::new();
        let mut max_epoch = 0u64;
        if first >= last {
            return (out, max_epoch);
        }
        self.for_each_chunk(first, last, false, |chunk_page, leaf, lo_idx, hi_idx| {
            for_each_word_in_chunk(lo_idx, hi_idx, |w, mask| {
                let mut bits = leaf.written_bits[w].load(Ordering::Acquire) & mask;
                while bits != 0 {
                    let tz = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let idx = w * 64 + tz;
                    let epoch = leaf.written_epoch[idx].load(Ordering::Acquire);
                    if epoch != 0 {
                        max_epoch = max_epoch.max(epoch);
                        out.push((chunk_page + idx, epoch));
                    }
                }
                true
            });
            true
        });
        (out, max_epoch)
    }

    fn publish_written(&self, pages: &[usize]) -> u64 {
        let _w = DirWriteGuard::enter();
        let now = next_epoch();
        for &page in pages {
            self.record_written(page, now);
        }
        now
    }

    fn publish_written_parts(&self, pages: &[(usize, u16)]) -> u64 {
        let _w = DirWriteGuard::enter();
        let now = next_epoch();
        for &(page, parts) in pages {
            self.record_written_parts(page, parts, now);
        }
        now
    }

    fn publish_written_range(&self, first: usize, last: usize) -> u64 {
        let _w = DirWriteGuard::enter();
        let now = next_epoch();
        for page in first..last {
            self.record_written(page, now);
        }
        now
    }

    fn collect_written_consistent(
        &self,
        first: usize,
        last: usize,
        min_epoch: u64,
    ) -> (u64, Vec<(usize, u64)>) {
        if first >= last {
            return (min_epoch, Vec::new());
        }
        let mut spins = 0u32;
        loop {
            let end_before = DIR_WRITE_END.load(Ordering::SeqCst);
            let begin_before = DIR_WRITE_BEGIN.load(Ordering::SeqCst);
            if begin_before != end_before {
                spins = spins.saturating_add(1);
                if spins < 64 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
                continue;
            }
            let (written, max_w) = self.collect_written(first, last);
            let begin_after = DIR_WRITE_BEGIN.load(Ordering::SeqCst);
            if begin_after == begin_before {
                return (min_epoch.max(max_w), written);
            }
            spins = spins.saturating_add(1);
            if spins < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    #[inline]
    #[allow(dead_code)]
    fn written_epoch(&self, page: usize) -> u64 {
        self.get(page)
            .map(|(leaf, idx)| leaf.written_epoch[idx].load(Ordering::Acquire))
            .unwrap_or(0)
    }

    #[inline]
    fn is_cold_ro(&self, block: usize) -> bool {
        self.get_bit(block, BitField::ColdRo, Ordering::SeqCst)
    }

    #[inline]
    fn is_guard_clean(&self, block: usize) -> bool {
        self.get_bit(block, BitField::GuardClean, Ordering::SeqCst)
    }

    #[inline]
    fn is_lazy_armed(&self, block: usize) -> bool {
        self.get_bit(block, BitField::LazyArmed, Ordering::SeqCst)
    }

    #[inline]
    fn is_guard_protected(&self, block: usize) -> bool {
        self.get_bit(block, BitField::GuardProtected, Ordering::SeqCst)
    }

    #[inline]
    fn is_guard_absent(&self, block: usize) -> bool {
        self.get_bit(block, BitField::GuardAbsent, Ordering::SeqCst)
    }

    /// Whether `block` is committed view memory the guards already hold
    /// read-only (or lazy-armed / explicitly `VICTORY_VIEW_KNOWN`).
    /// `GuardAbsent` blocks are NOT treated as known views so a reserved
    /// placeholder later committed as private memory is still re-checked by
    /// [`has_mapped`].
    #[inline]
    fn is_known_view(&self, block: usize, vk: bool) -> bool {
        let Some((leaf, idx)) = self.get(block) else {
            return false;
        };
        let (w, bit) = (idx / 64, 1u64 << (idx % 64));
        (leaf.guard_protected_bits[w].load(Ordering::Acquire) & bit != 0)
            || (leaf.lazy_armed_bits[w].load(Ordering::Acquire) & bit != 0)
            || (vk && (leaf.guard_viewed_bits[w].load(Ordering::Acquire) & bit != 0))
    }

    #[inline]
    fn any_guard_flag(&self, block: usize) -> bool {
        let Some((leaf, idx)) = self.get(block) else {
            return false;
        };
        let (w, bit) = (idx / 64, 1u64 << (idx % 64));
        let combined = leaf.guard_protected_bits[w].load(Ordering::Acquire)
            | leaf.guard_absent_bits[w].load(Ordering::Acquire)
            | leaf.guard_aliased_bits[w].load(Ordering::Acquire)
            | leaf.guard_viewed_bits[w].load(Ordering::Acquire);
        combined & bit != 0
    }

    fn all_words_set(
        &self,
        first: usize,
        last: usize,
        mut word_val: impl FnMut(&LeafChunk, usize) -> u64,
    ) -> bool {
        if first >= last || last > MAX_RADIX_PAGES {
            return false;
        }
        let mut cur = first;
        while cur < last {
            let chunk_page = cur & !(CHUNK_PAGES - 1);
            let next_chunk = (chunk_page + CHUNK_PAGES).min(last);
            let Some((leaf, _)) = self.get(cur) else {
                return false;
            };
            let lo_idx = cur - chunk_page;
            let hi_idx = next_chunk - chunk_page;
            let mut all = true;
            for_each_word_in_chunk(lo_idx, hi_idx, |w, mask| {
                if word_val(leaf, w) & mask != mask {
                    all = false;
                    return false;
                }
                true
            });
            if !all {
                return false;
            }
            cur = next_chunk;
        }
        true
    }

    fn all_guard_clean_or_lazy(&self, first: usize, last: usize) -> bool {
        self.all_words_set(first, last, |leaf, w| {
            leaf.guard_clean_bits[w].load(Ordering::SeqCst)
                | leaf.lazy_armed_bits[w].load(Ordering::SeqCst)
        })
    }

    fn any_bits_in_range(&self, first: usize, last: usize, fields: &[BitField]) -> bool {
        if first >= last {
            return false;
        }
        let mut found = false;
        self.for_each_chunk(first, last, false, |_, leaf, lo_idx, hi_idx| {
            for_each_word_in_chunk(lo_idx, hi_idx, |w, mask| {
                let mut combined = 0u64;
                for &field in fields {
                    combined |= leaf.bits(field)[w].load(Ordering::SeqCst);
                }
                if combined & mask != 0 {
                    found = true;
                    return false;
                }
                true
            });
            !found
        });
        found
    }

    fn count_bits_in_range(&self, first: usize, last: usize, field: BitField) -> u64 {
        if first >= last {
            return 0;
        }
        let mut count = 0u64;
        self.for_each_chunk(first, last, false, |_, leaf, lo_idx, hi_idx| {
            let words = leaf.bits(field);
            for_each_word_in_chunk(lo_idx, hi_idx, |w, mask| {
                count += (words[w].load(Ordering::Acquire) & mask).count_ones() as u64;
                true
            });
            true
        });
        count
    }

    fn collect_set_bits(&self, first: usize, last: usize, field: BitField) -> Vec<usize> {
        let mut out = Vec::new();
        if first >= last {
            return out;
        }
        self.for_each_chunk(first, last, false, |chunk_page, leaf, lo_idx, hi_idx| {
            let words = leaf.bits(field);
            for_each_word_in_chunk(lo_idx, hi_idx, |w, mask| {
                let mut bits = words[w].load(Ordering::Acquire) & mask;
                while bits != 0 {
                    let tz = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    out.push(chunk_page + w * 64 + tz);
                }
                true
            });
            true
        });
        out
    }

    /// Collects contiguous sub-runs of `[first, last)` where `word_val` bits are 0.
    fn collect_clear_runs_with(
        &self,
        first: usize,
        last: usize,
        mut word_val: impl FnMut(&LeafChunk, usize) -> u64,
    ) -> Vec<(usize, usize)> {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        if first >= last {
            return runs;
        }
        let mut cur = first;
        let end = last.min(MAX_RADIX_PAGES);
        let push_run = |runs: &mut Vec<(usize, usize)>, from: usize, to: usize| {
            if from >= to {
                return;
            }
            match runs.last_mut() {
                Some(r) if r.1 == from => r.1 = to,
                _ => runs.push((from, to)),
            }
        };
        while cur < end {
            let chunk_page = cur & !(CHUNK_PAGES - 1);
            let next_chunk = (chunk_page + CHUNK_PAGES).min(end);
            let Some((leaf, _)) = self.get(cur) else {
                let top_i = cur >> (MID_BITS + CHUNK_BITS);
                let skip_to =
                    if top_i < TOP_LEN && self.top[top_i].load(Ordering::Acquire).is_null() {
                        ((top_i + 1) << (MID_BITS + CHUNK_BITS)).min(end)
                    } else {
                        next_chunk
                    };
                push_run(&mut runs, cur, skip_to);
                cur = skip_to;
                continue;
            };
            let lo_idx = cur - chunk_page;
            let hi_idx = next_chunk - chunk_page;
            for_each_word_in_chunk(lo_idx, hi_idx, |w, mask| {
                let val = word_val(leaf, w);
                let mut clear = (!val) & mask;
                while clear != 0 {
                    let tz = clear.trailing_zeros() as usize;
                    let shifted = clear >> tz;
                    let ones = shifted.trailing_ones() as usize;
                    let r_start = chunk_page + w * 64 + tz;
                    let r_end = r_start + ones;
                    push_run(&mut runs, r_start, r_end);
                    if tz + ones >= 64 {
                        break;
                    }
                    clear &= !(((1u64 << ones) - 1) << tz);
                }
                true
            });
            cur = next_chunk;
        }
        runs
    }

    /// Collects contiguous sub-runs of `[first, last)` where `field` bits are 0.
    fn collect_clear_runs(
        &self,
        first: usize,
        last: usize,
        field: BitField,
    ) -> Vec<(usize, usize)> {
        self.collect_clear_runs_with(first, last, |leaf, w| {
            leaf.bits(field)[w].load(Ordering::SeqCst)
        })
    }

    /// Collects contiguous sub-runs of `[first, last)` where neither `GuardClean` nor `LazyArmed` is set.
    fn collect_non_clean_non_lazy_runs(&self, first: usize, last: usize) -> Vec<(usize, usize)> {
        self.collect_clear_runs_with(first, last, |leaf, w| {
            leaf.guard_clean_bits[w].load(Ordering::Acquire)
                | leaf.lazy_armed_bits[w].load(Ordering::Acquire)
        })
    }

    #[inline]
    fn reset_cold(&self, page: usize, frame32: u32) {
        if let Some((leaf, idx)) = self.get(page) {
            leaf.cold_streak[idx].store(0, Ordering::Release);
            leaf.cold_last_frame[idx].store(frame32, Ordering::Release);
        }
    }

    fn reset_cold_range(&self, first: usize, last: usize, frame32: u32) {
        if first >= last {
            return;
        }
        self.for_each_chunk(first, last, false, |_, leaf, lo_idx, hi_idx| {
            for idx in lo_idx..hi_idx {
                leaf.cold_streak[idx].store(0, Ordering::Release);
                leaf.cold_last_frame[idx].store(frame32, Ordering::Release);
            }
            true
        });
    }

    #[inline]
    fn cold_streak(&self, page: usize) -> u8 {
        self.get(page).map_or(0, |(leaf, idx)| {
            leaf.cold_streak[idx].load(Ordering::Acquire)
        })
    }

    #[cfg(test)]
    fn clear_written_range(&self, first: usize, last: usize) {
        if first >= last {
            return;
        }
        self.for_each_chunk(first, last, false, |_, leaf, lo_idx, hi_idx| {
            for_each_word_in_chunk(lo_idx, hi_idx, |w, mask| {
                leaf.written_bits[w].fetch_and(!mask, Ordering::SeqCst);
                true
            });
            let parts_ptr = leaf.parts_epoch.load(Ordering::Acquire);
            for idx in lo_idx..hi_idx {
                leaf.written_epoch[idx].store(0, Ordering::SeqCst);
                leaf.whole_epoch[idx].store(0, Ordering::SeqCst);
                if !parts_ptr.is_null() {
                    // SAFETY: `parts_ptr` is permanently live once published.
                    for slot in unsafe { &(*parts_ptr)[idx] } {
                        slot.store(0, Ordering::SeqCst);
                    }
                }
            }
            true
        });
    }

    fn note_clean_for_cold(
        &self,
        first: usize,
        last: usize,
        dirty_sorted: &[(usize, u16)],
        frame32: u32,
        threshold: u8,
        out_promote: &mut Vec<usize>,
    ) {
        if first >= last {
            return;
        }
        let mut dirty_idx = dirty_sorted.partition_point(|&(p, _)| p < first);
        self.for_each_chunk(first, last, true, |chunk_page, leaf, lo_idx, hi_idx| {
            // Mark pages confirmed write-watched.
            for_each_word_in_chunk(lo_idx, hi_idx, |w, mask| {
                leaf.watched_bits[w].fetch_or(mask, Ordering::Release);
                true
            });
            for idx in lo_idx..hi_idx {
                let page = chunk_page + idx;
                while dirty_idx < dirty_sorted.len() && dirty_sorted[dirty_idx].0 < page {
                    dirty_idx += 1;
                }
                if dirty_idx < dirty_sorted.len() && dirty_sorted[dirty_idx].0 == page {
                    leaf.cold_streak[idx].store(0, Ordering::Release);
                    leaf.cold_last_frame[idx].store(frame32, Ordering::Release);
                    continue;
                }
                if threshold == 0 {
                    continue;
                }
                let (w, bit) = (idx / 64, 1u64 << (idx % 64));
                if leaf.cold_ro_bits[w].load(Ordering::Acquire) & bit != 0 {
                    continue;
                }
                if leaf.cold_last_frame[idx].swap(frame32, Ordering::AcqRel) != frame32 {
                    let prev = leaf.cold_streak[idx].load(Ordering::Acquire);
                    let next = prev.saturating_add(1).min(threshold);
                    leaf.cold_streak[idx].store(next, Ordering::Release);
                    if next >= threshold {
                        out_promote.push(page);
                    }
                }
            }
            true
        });
    }

    #[cfg(feature = "diagnostics")]
    fn guard_flags(&self, block: usize) -> (bool, bool, bool, bool, bool) {
        let Some((leaf, idx)) = self.get(block) else {
            return (false, false, false, false, false);
        };
        let (w, bit) = (idx / 64, 1u64 << (idx % 64));
        (
            leaf.guard_protected_bits[w].load(Ordering::Relaxed) & bit != 0,
            leaf.guard_clean_bits[w].load(Ordering::Relaxed) & bit != 0,
            leaf.guard_absent_bits[w].load(Ordering::Relaxed) & bit != 0,
            leaf.guard_viewed_bits[w].load(Ordering::Relaxed) & bit != 0,
            leaf.guard_aliased_bits[w].load(Ordering::Relaxed) & bit != 0,
        )
    }
}

// ---------------------------------------------------------------------------
// Journal Epoch & Ahead Batch State
// ---------------------------------------------------------------------------

/// Monotonic write-watch asking epoch; 0 means "never".
static ASK_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Lock-free multi-writer publication counters for [`PageDir`] written-page
/// records. Every writer that records written pages into [`DIR`] MUST:
/// 1. Enter [`DirWriteGuard`] (incrementing `DIR_WRITE_BEGIN` with `SeqCst`),
/// 2. Allocate its epoch via [`next_epoch`] while the guard is held,
/// 3. Write all pages of its batch via [`PageDir::record_written`], and
/// 4. Drop [`DirWriteGuard`] (incrementing `DIR_WRITE_END` with `SeqCst`).
///
/// Readers use [`PageDir::collect_written_consistent`] so a scan and its
/// returned watermark always come from one consistent publication without
/// skipping any page from a concurrent multi-word write batch.
static DIR_WRITE_BEGIN: AtomicU64 = AtomicU64::new(0);
static DIR_WRITE_END: AtomicU64 = AtomicU64::new(0);

struct DirWriteGuard;

impl DirWriteGuard {
    #[inline]
    fn enter() -> Self {
        DIR_WRITE_BEGIN.fetch_add(1, Ordering::SeqCst);
        Self
    }
}

impl Drop for DirWriteGuard {
    #[inline]
    fn drop(&mut self) {
        DIR_WRITE_END.fetch_add(1, Ordering::SeqCst);
    }
}

#[inline]
fn next_epoch() -> u64 {
    ASK_EPOCH.fetch_add(1, Ordering::AcqRel) + 1
}

struct AheadState {
    ranges: Vec<(usize, usize)>,
    epoch: u64,
    kept: bool,
}

impl AheadState {
    /// Drops `[first, last)` from any standing write-watch ask-ahead coverage
    /// so subsequent refreshes re-query kernel writes over that range.
    fn forget_ahead(&mut self, (first, last): (usize, usize)) {
        const _: () = assert!(BLOCK == PAGE);
        if self.ranges.is_empty() || first >= last {
            return;
        }
        let mut out = Vec::with_capacity(self.ranges.len() + 1);
        for &(a, b) in &self.ranges {
            if b <= first || a >= last {
                out.push((a, b));
            } else {
                if a < first {
                    out.push((a, first));
                }
                if b > last {
                    out.push((last, b));
                }
            }
        }
        self.ranges = out;
        let active = !self.ranges.is_empty();
        if !active {
            self.kept = false;
        }
        set_ahead_flags(self.kept, active);
    }
}

static AHEAD: RwLock<AheadState> = RwLock::new(AheadState {
    ranges: Vec::new(),
    epoch: 0,
    kept: false,
});

/// Serializes `GetWriteWatch(WRITE_WATCH_FLAG_RESET)` calls + `PageDir`
/// recording so concurrent non-ahead callers on overlapping ranges never miss
/// a reset page.
///
/// Global lock order (strictly acyclic):
/// `WATCH_SERIAL` -> `GUARD_SHARDS_TABLE[0..GUARD_SHARDS]` (strictly ascending
/// shard index) -> `AHEAD` (leaf lock).
static WATCH_SERIAL: Mutex<()> = Mutex::new(());

/// Fast lock-free mirror of `!AHEAD.ranges.is_empty()` so callers outside
/// `ask_once`/`ask_ahead` skip `AHEAD.read()` completely.
static AHEAD_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Mirrors `AheadState::kept` so `Batch::flush` / `ask_ahead` / `ask_ahead_done`
/// can skip range collection, `in_view` lock lookups, sorting, and exclusive
/// `AHEAD.write()` locks while a submission-wide `ask_once` is active.
static ASK_KEPT: AtomicBool = AtomicBool::new(false);

#[inline]
fn set_ahead_flags(kept: bool, active: bool) {
    AHEAD_ACTIVE.store(active, Ordering::Release);
    ASK_KEPT.store(kept, Ordering::Release);
}

/// Whether a submission-wide write-watch asking (`ask_once`) currently stands.
#[inline]
pub(crate) fn ask_kept() -> bool {
    ASK_KEPT.load(Ordering::Acquire)
}

fn merge_dirty_parts(dirty: &mut Vec<(usize, u16)>) {
    if dirty.len() <= 1 {
        return;
    }
    dirty.sort_unstable_by_key(|&(p, _)| p);
    let mut out = 0usize;
    for i in 0..dirty.len() {
        let (p, parts) = dirty[i];
        if out > 0 && dirty[out - 1].0 == p {
            dirty[out - 1].1 |= parts;
        } else {
            dirty[out] = (p, parts);
            out += 1;
        }
    }
    dirty.truncate(out);
}

#[inline]
fn ahead_covers(ranges: &[(usize, usize)], first: usize, last: usize) -> bool {
    let idx = ranges.partition_point(|&(a, _)| a <= first);
    idx > 0 && ranges[idx - 1].1 >= last
}

fn push_merged_interval(out: &mut Vec<(usize, usize)>, first: usize, last: usize) {
    if first >= last {
        return;
    }
    match out.last_mut() {
        Some(prev) if first <= prev.1 => prev.1 = prev.1.max(last),
        _ => out.push((first, last)),
    }
}

/// Thread-local scratch buffer for `GetWriteWatch` output, eliminating global
/// lock contention on buffer resizing. Moves the `Vec` out during `f` so any
/// re-entrant VEH call on the same thread never panics on `RefCell`.
fn with_watch_scratch<R>(count: usize, f: impl FnOnce(&mut Vec<usize>) -> R) -> R {
    thread_local! {
        static SCRATCH: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    }
    let mut buf = SCRATCH.with(|cell| {
        cell.try_borrow_mut()
            .map(|mut b| std::mem::take(&mut *b))
            .unwrap_or_default()
    });
    if buf.len() < count {
        buf.resize(count, 0);
    }
    let out = f(&mut buf);
    SCRATCH.with(|cell| {
        if let Ok(mut saved) = cell.try_borrow_mut()
            && buf.capacity() > saved.capacity()
        {
            *saved = buf;
        }
    });
    out
}

/// Experiment (`VICTORY_WATCH_STATS=1`): the kernel askings by where they
/// come from and how they end: ahead ok, ahead failed, `writes` ok,
/// `writes` failed (then the guards), the pages asked in each, and the
/// microseconds of each.
static ASKS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
/// Of the askings that were answered: those that found no written page
/// (count, pages, microseconds), the pages found written in the others,
/// and askings spared by cold promotion (count, pages).
static CLEAN: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

fn asks_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WATCH_STATS").is_some_and(|v| v == "1"))
}

fn asked_kernel(kind: usize, pages: usize, started: std::time::Instant) {
    use Ordering::Relaxed;
    ASKS[kind].fetch_add(1, Relaxed);
    ASKS[4 + kind].fetch_add(pages as u64, Relaxed);
    ASKS[8 + kind].fetch_add(started.elapsed().as_nanos() as u64 / 1000, Relaxed);
}

/// The counts since the last call, for the frame line's neighbour.
pub(crate) fn asks_say(frames: f64) {
    if !asks_on() || frames <= 0.0 {
        return;
    }
    let v: Vec<f64> = ASKS
        .iter()
        .map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames)
        .collect();
    let c: Vec<f64> = CLEAN
        .iter()
        .map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames)
        .collect();
    tracing::info!(
        "write-watch askings per frame, answered: {:.0} found nothing ({:.0} MB, {:.1} ms); the others found {:.0} pages written; spared as cold {:.0} ({:.0} MB)",
        c[0],
        c[1] * PAGE as f64 / 1048576.0,
        c[2] / 1000.0,
        c[3],
        c[4],
        c[5] * PAGE as f64 / 1048576.0
    );
    tracing::info!(
        "write-watch askings per frame (VICTORY_WATCH_STATS=1): ahead ok {:.0} ({:.0} MB, {:.1} ms), ahead failed {:.0} ({:.0} MB, {:.1} ms), single ok {:.0} ({:.0} MB, {:.1} ms), single failed {:.0} ({:.0} MB, {:.1} ms)",
        v[0],
        v[4] * PAGE as f64 / 1048576.0,
        v[8] / 1000.0,
        v[1],
        v[5] * PAGE as f64 / 1048576.0,
        v[9] / 1000.0,
        v[2],
        v[6] * PAGE as f64 / 1048576.0,
        v[10] / 1000.0,
        v[3],
        v[7] * PAGE as f64 / 1048576.0,
        v[11] / 1000.0,
    );
}

/// Whether pages `[first, last)` begin in a view of the title's physical
/// pages (`crate::inplace`'s record of what the runtime mapped). View
/// memory is not write-watched: the kernel refuses the asking, and the
/// guards answer instead. `VICTORY_WATCH_VIEWS=0` asks the kernel all the
/// same, as before.
fn in_view(first: usize, last: usize) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_WATCH_VIEWS").is_none_or(|v| v != "0")) {
        return false;
    }
    let _ = last;
    crate::inplace::cover(first * PAGE, 1) != crate::inplace::Cover::None
}

/// Calls `GetWriteWatch(WRITE_WATCH_FLAG_RESET)` on `[first, last)` and
/// returns `Some(dirty_pages_sorted_deduped_with_4k_parts)` on success or
/// `None` on failure.
fn call_get_write_watch(
    first: usize,
    last: usize,
    stats_kind_base: usize,
) -> Option<Vec<(usize, u16)>> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    if first >= last {
        return Some(Vec::new());
    }
    let needed = (last - first) * PAGE / KERNEL_PAGE;
    with_watch_scratch(needed, |buf| {
        let mut count = needed;
        let mut granularity = 0;
        let started = std::time::Instant::now();
        // SAFETY: `buf` holds at least `count` entries.
        let failed = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                buf.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        if asks_on() {
            asked_kernel(stats_kind_base + usize::from(failed), last - first, started);
        }
        if failed {
            return None;
        }
        if asks_on() {
            use Ordering::Relaxed;
            if count == 0 {
                CLEAN[0].fetch_add(1, Relaxed);
                CLEAN[1].fetch_add((last - first) as u64, Relaxed);
                CLEAN[2].fetch_add(started.elapsed().as_nanos() as u64 / 1000, Relaxed);
            } else {
                CLEAN[3].fetch_add(count as u64, Relaxed);
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
            slow(started.elapsed().as_micros() as u64);
        }
        let mut dirty: Vec<(usize, u16)> = Vec::new();
        let mut in_order = true;
        for &at in &buf[..count] {
            let page = at / PAGE;
            let part = 1u16 << ((at % PAGE) / KERNEL_PAGE);
            match dirty.last_mut() {
                Some((last, parts)) if *last == page => *parts |= part,
                Some((last, _)) => {
                    if page < *last {
                        in_order = false;
                    }
                    dirty.push((page, part));
                }
                None => dirty.push((page, part)),
            }
        }
        if !in_order {
            dirty.sort_unstable_by_key(|&(page, _)| page);
            let mut merged: Vec<(usize, u16)> = Vec::with_capacity(dirty.len());
            for (page, part) in dirty {
                match merged.last_mut() {
                    Some((last, parts)) if *last == page => *parts |= part,
                    _ => merged.push((page, part)),
                }
            }
            dirty = merged;
        }
        Some(dirty)
    })
}

/// Invokes `f(from, to)` for each maximal contiguous `[from, to)` run in
/// sorted, deduplicated `blocks`.
fn for_each_contiguous_run(blocks: &[usize], mut f: impl FnMut(usize, usize)) {
    let mut i = 0;
    while i < blocks.len() {
        let from = blocks[i];
        let mut to = from + 1;
        i += 1;
        while i < blocks.len() && blocks[i] == to {
            to += 1;
            i += 1;
        }
        f(from, to);
    }
}

/// Whether every 4 KiB page in the 64 KiB `block` is committed `PAGE_READWRITE`
/// (exact protection match, excluding `PAGE_READONLY`, `PAGE_NOACCESS`,
/// `PAGE_GUARD`, `PAGE_EXECUTE_*`, etc.) so promoting to `PAGE_READONLY` and
/// restoring `PAGE_READWRITE` on fault never drops a caller-installed
/// protection. Uses `cache = (region_start, region_end, is_rw)` to amortize
/// `VirtualQuery` across contiguous blocks.
fn block_is_all_readwrite(block: usize, cache: &mut (usize, usize, bool)) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    let start = block * PAGE;
    let end = start.saturating_add(PAGE);
    if start >= cache.0 && end <= cache.1 {
        return cache.2;
    }
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: queries virtual memory metadata without dereferencing `start`.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.RegionSize == 0 {
        *cache = (0, 0, false);
        return false;
    }
    let region_start = info.BaseAddress as usize;
    let region_end = region_start.saturating_add(info.RegionSize);
    let is_rw = info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE;
    *cache = (region_start, region_end, is_rw);
    is_rw && start >= region_start && end <= region_end
}

/// Promotes contiguous runs of newly-cold write-watched 64 KiB blocks to
/// `PAGE_READONLY` VEH guards, then re-verifies `GetWriteWatch(RESET)` while
/// `PAGE_READONLY` is active so any write sneaking in right before
/// `VirtualProtect` is caught and never lost.
fn promote_cold_pages(candidates: &[usize]) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    let threshold = cold_after();
    if threshold == 0 || candidates.is_empty() {
        return;
    }
    if !install() {
        return;
    }
    let threshold_u8 = threshold.min(u8::MAX as u32) as u8;
    let mut sorted = candidates.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let frame32 = current_frame32();

    // Lock order: `WATCH_SERIAL` -> `GUARD_SHARDS` (ascending). Holding both
    // across `VirtualProtect` + `GetWriteWatch` verification ensures `ColdRo`
    // is never visible until verified clean, and `on_write` (which acquires
    // only the faulted block's shard without `WATCH_SERIAL`) serializes cleanly
    // if a title thread faults before the shard lock drops.
    let _serial = WATCH_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut locked = GuardLockSet::lock_blocks(&sorted);
    locked.prune_pins();

    let lazy = crate::lazy_back::active();
    let mut eligible = Vec::with_capacity(sorted.len());
    let mut rw_cache = (0usize, 0usize, false);
    for block in sorted {
        if DIR.is_cold_ro(block)
            || DIR.any_guard_flag(block)
            || DIR.cold_streak(block) < threshold_u8
            || locked.pinned(block)
            || (lazy && crate::lazy_back::has_block(block))
            || in_view(block, block + 1)
        {
            continue;
        }
        if !block_is_all_readwrite(block, &mut rw_cache) {
            DIR.reset_cold(block, frame32);
            continue;
        }
        eligible.push(block);
    }
    if eligible.is_empty() {
        return;
    }

    let mut promoted_runs: Vec<(usize, usize)> = Vec::new();
    for_each_contiguous_run(&eligible, |from, to| {
        virtual_protect_adaptive(
            from,
            to,
            PAGE_READONLY,
            false,
            &mut |ok_from, ok_to| promoted_runs.push((ok_from, ok_to)),
            &mut |_, _| {},
            &mut |bad| DIR.reset_cold(bad, frame32),
        );
    });
    if promoted_runs.is_empty() {
        return;
    }

    // Re-check `GetWriteWatch(RESET)` while `PAGE_READONLY` is in effect and
    // before setting `BitField::ColdRo = true`, closing any race with writes
    // that landed right before `VirtualProtect`.
    fn verify_and_mark_cold(from: usize, to: usize, frame32: u32) {
        if from >= to {
            return;
        }
        let mut old = 0u32;
        if let Some(dirty) = call_get_write_watch(from, to, 2) {
            if dirty.is_empty() {
                DIR.set_bits_range(from, to, BitField::ColdRo, true, Ordering::SeqCst);
                return;
            }
            let _now = DIR.publish_written_parts(&dirty);
            let mut d_idx = dirty.partition_point(|&(b, _)| b < from);
            for block in from..to {
                while d_idx < dirty.len() && dirty[d_idx].0 < block {
                    d_idx += 1;
                }
                if d_idx < dirty.len() && dirty[d_idx].0 == block {
                    DIR.reset_cold(block, frame32);
                    // SAFETY: restoring `PAGE_READWRITE` on a block whose write snuck in before promotion.
                    unsafe {
                        VirtualProtect((block * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old);
                    }
                } else {
                    DIR.set_bit(block, BitField::ColdRo, true, Ordering::SeqCst);
                }
            }
            return;
        }
        if to - from > 1 {
            let mid = from + (to - from) / 2;
            verify_and_mark_cold(from, mid, frame32);
            verify_and_mark_cold(mid, to, frame32);
            return;
        }
        // Single block failed `GetWriteWatch` verification: restore RW and do not mark cold.
        DIR.reset_cold(from, frame32);
        // SAFETY: restoring `PAGE_READWRITE` on unverified block.
        unsafe { VirtualProtect((from * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old) };
    }

    for (from, to) in promoted_runs {
        verify_and_mark_cold(from, to, frame32);
    }
}

/// Demotes any cold-promoted `PAGE_READONLY` blocks in `first..last` back to
/// `PAGE_READWRITE` and resets their cold streak while their guard shards are held.
fn demote_cold_range(locked: &impl ShardAccess, first: usize, last: usize) {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    if first >= last {
        return;
    }
    debug_assert_range_locked(locked, first, last);
    let frame32 = current_frame32();
    let cold_blocks = DIR.collect_set_bits(first, last, BitField::ColdRo);
    if cold_blocks.is_empty() {
        return;
    }
    DIR.set_bits_range(first, last, BitField::ColdRo, false, Ordering::SeqCst);
    for &b in &cold_blocks {
        DIR.reset_cold(b, frame32);
    }
    protect_block_runs(&cold_blocks, PAGE_READWRITE);
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// [`AHEAD_GAP`], or `VICTORY_WATCH_GAP_MB`.
fn ahead_gap() -> usize {
    static GAP: OnceLock<usize> = OnceLock::new();
    *GAP.get_or_init(|| {
        std::env::var("VICTORY_WATCH_GAP_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(AHEAD_GAP, |mb| (mb << 20) / PAGE)
    })
}

/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Processes one non-cold `[first, last)` interval with `GetWriteWatch(RESET)`.
/// When the whole-range query fails, walks allocation boundaries via
/// `VirtualQuery` to skip free, mapped, image, and non-write-watched regions
/// immediately while still querying any private `MEM_WRITE_WATCH` portions of a
/// mixed range. Appends dirty pages to `batch_dirty` and pushes every succeeded
/// `[a, b)` page interval into `covered`.
fn ask_non_cold_range(
    first: usize,
    last: usize,
    frame32: u32,
    threshold: u8,
    batch_dirty: &mut Vec<(usize, u16)>,
    covered: &mut Vec<(usize, usize)>,
    to_promote: &mut Vec<usize>,
) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    if first >= last {
        return;
    }
    if let Some(dirty) = call_get_write_watch(first, last, 0) {
        batch_dirty.extend_from_slice(&dirty);
        DIR.note_clean_for_cold(first, last, &dirty, frame32, threshold, to_promote);
        push_merged_interval(covered, first, last);
        return;
    }

    let mut at = first * PAGE;
    let end = last * PAGE;
    while at < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: queries virtual memory metadata without dereferencing `at`.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0
            || info.RegionSize == 0
        {
            break;
        }
        let Some(mut to) = (info.BaseAddress as usize)
            .checked_add(info.RegionSize)
            .map(|e| e.min(end))
        else {
            break;
        };
        if to <= at {
            break;
        }
        // Free regions, mapped views (`MEM_MAPPED`), and images (`MEM_IMAGE`)
        // can never be `MEM_WRITE_WATCH`; skip their entire allocation without
        // issuing failing `GetWriteWatch` syscalls while holding `WATCH_SERIAL`.
        if info.State == MEM_FREE || info.Type != MEM_PRIVATE {
            let base = info.AllocationBase as usize;
            while info.State != MEM_FREE && to < end {
                let mut next = MEMORY_BASIC_INFORMATION::default();
                // SAFETY: metadata query only.
                if unsafe { VirtualQuery(to as *const _, &mut next, size_of_val(&next)) } == 0
                    || next.RegionSize == 0
                    || next.State == MEM_FREE
                    || next.AllocationBase as usize != base
                {
                    break;
                }
                let Some(next_to) = (next.BaseAddress as usize)
                    .checked_add(next.RegionSize)
                    .map(|e| e.min(end))
                else {
                    break;
                };
                if next_to <= to {
                    break;
                }
                to = next_to;
            }
            at = to;
            continue;
        }

        // Private allocation: gather all contiguous regions sharing `AllocationBase`.
        let base = info.AllocationBase as usize;
        let mut committed = info.State == MEM_COMMIT;
        let mut committed_subranges: Vec<(usize, usize)> = Vec::new();
        if info.State == MEM_COMMIT {
            committed_subranges.push((at, to));
        }
        while to < end {
            let mut next = MEMORY_BASIC_INFORMATION::default();
            // SAFETY: metadata query only.
            if unsafe { VirtualQuery(to as *const _, &mut next, size_of_val(&next)) } == 0
                || next.RegionSize == 0
                || next.State == MEM_FREE
                || next.AllocationBase as usize != base
            {
                break;
            }
            let Some(next_to) = (next.BaseAddress as usize)
                .checked_add(next.RegionSize)
                .map(|e| e.min(end))
            else {
                break;
            };
            if next_to <= to {
                break;
            }
            if next.State == MEM_COMMIT {
                committed = true;
                committed_subranges.push((to, next_to));
            }
            to = next_to;
        }

        if committed {
            let p_first = at.div_ceil(PAGE);
            let p_last = to / PAGE;
            if p_first < p_last {
                if (p_first != first || p_last != last)
                    && let Some(dirty) = call_get_write_watch(p_first, p_last, 0)
                {
                    batch_dirty.extend_from_slice(&dirty);
                    DIR.note_clean_for_cold(
                        p_first, p_last, &dirty, frame32, threshold, to_promote,
                    );
                    push_merged_interval(covered, p_first, p_last);
                } else if committed_subranges.len() > 1
                    || committed_subranges
                        .first()
                        .is_some_and(|&(s, e)| s != at || e != to)
                {
                    // Mixed committed+reserved private allocation where the full
                    // span failed: probe committed page-aligned subranges and
                    // bail on the first failure (non-MEM_WRITE_WATCH allocation).
                    for (s, e) in committed_subranges {
                        let sp0 = s.div_ceil(PAGE);
                        let sp1 = e / PAGE;
                        if sp0 >= sp1 {
                            continue;
                        }
                        if let Some(dirty) = call_get_write_watch(sp0, sp1, 0) {
                            batch_dirty.extend_from_slice(&dirty);
                            DIR.note_clean_for_cold(
                                sp0, sp1, &dirty, frame32, threshold, to_promote,
                            );
                            push_merged_interval(covered, sp0, sp1);
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        at = to;
    }
}

/// Queries `GetWriteWatch` on a merged `[first, last)` interval while skipping
/// any `cold_ro` sub-runs (which are already protected via `PAGE_READONLY` VEH).
fn ask_merged_interval(
    first: usize,
    last: usize,
    frame32: u32,
    threshold: u8,
    batch_dirty: &mut Vec<(usize, u16)>,
    covered: &mut Vec<(usize, usize)>,
    to_promote: &mut Vec<usize>,
) {
    if first >= last {
        return;
    }
    if v2_enabled() && threshold > 0 {
        let non_cold = DIR.collect_clear_runs(first, last, BitField::ColdRo);
        if non_cold.is_empty() {
            if asks_on() {
                CLEAN[4].fetch_add(1, Ordering::Relaxed);
                CLEAN[5].fetch_add((last - first) as u64, Ordering::Relaxed);
            }
            push_merged_interval(covered, first, last);
            return;
        }
        let non_cold_pages: usize = non_cold.iter().map(|&(a, b)| b - a).sum();
        if asks_on() && non_cold_pages < last - first {
            CLEAN[4].fetch_add(1, Ordering::Relaxed);
            CLEAN[5].fetch_add(((last - first) - non_cold_pages) as u64, Ordering::Relaxed);
        }
        let mut sub_covered = Vec::with_capacity(non_cold.len());
        for (r0, r1) in non_cold {
            ask_non_cold_range(
                r0,
                r1,
                frame32,
                threshold,
                batch_dirty,
                &mut sub_covered,
                to_promote,
            );
        }
        // Only mark the whole interval covered (including cold pages) if all
        // non-cold sub-runs succeeded; otherwise keep only the succeeded pieces
        // plus any cold sub-runs inside succeeded spans.
        let got_pages: usize = sub_covered.iter().map(|&(a, b)| b - a).sum();
        if got_pages == non_cold_pages {
            push_merged_interval(covered, first, last);
        } else {
            for (a, b) in sub_covered {
                push_merged_interval(covered, a, b);
            }
        }
        return;
    }
    ask_non_cold_range(
        first,
        last,
        frame32,
        threshold,
        batch_dirty,
        covered,
        to_promote,
    );
}

/// Prepares sorted, merged, non-view intervals and queries `GetWriteWatch(RESET)`
/// across them, coalescing small non-cold gaps without crossing cold or guard regions.
/// Reserves and publishes the batch epoch while holding `WATCH_SERIAL` so every
/// reset carries an epoch strictly greater than any earlier watermark.
fn collect_and_ask_batch(
    ranges: &[(usize, usize)],
    allow_singleton: bool,
) -> (u64, Vec<(usize, usize)>) {
    let vk = view_known();
    let skip_known_views = !view_ask_disabled();
    let mut pieces: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    #[cfg(feature = "diagnostics")]
    let mut skipped_views = 0u64;
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if skip_known_views && DIR.is_known_view(first, vk) {
            #[cfg(feature = "diagnostics")]
            {
                skipped_views += 1;
            }
            continue;
        }
        if in_view(first, last) {
            continue;
        }
        pieces.push((first, last));
    }
    #[cfg(feature = "diagnostics")]
    AHEAD_VIEWS.fetch_add(skipped_views, Ordering::Relaxed);

    if pieces.is_empty() || (!allow_singleton && pieces.len() < 2) {
        return (0, Vec::new());
    }
    pieces.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(pieces.len());
    for (a, b) in pieces {
        push_merged_interval(&mut merged, a, b);
    }

    let gap_limit = ahead_gap();
    let frame32 = current_frame32();
    let threshold = cold_after().min(u8::MAX as u32) as u8;
    let mut covered = Vec::with_capacity(merged.len());
    let mut to_promote = Vec::new();
    let mut batch_dirty = Vec::new();

    let now = {
        let _serial = WATCH_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut i = 0;
        while i < merged.len() {
            let group_first = merged[i].0;
            let mut group_last = merged[i].1;
            let mut k = i + 1;
            while k < merged.len() {
                let next = merged[k];
                let gap = next.0.saturating_sub(group_last);
                if gap > gap_limit || next.1 - group_first > AHEAD_MAX {
                    break;
                }
                if gap > 0
                    && v2_enabled()
                    && DIR.any_bits_in_range(
                        group_last,
                        next.0,
                        &[
                            BitField::ColdRo,
                            BitField::GuardProtected,
                            BitField::GuardAbsent,
                            BitField::GuardViewed,
                            BitField::GuardAliased,
                            BitField::LazyArmed,
                        ],
                    )
                {
                    break;
                }
                group_last = group_last.max(next.1);
                k += 1;
            }

            if k == i + 1 {
                ask_merged_interval(
                    group_first,
                    group_last,
                    frame32,
                    threshold,
                    &mut batch_dirty,
                    &mut covered,
                    &mut to_promote,
                );
            } else if v2_enabled()
                && threshold > 0
                && DIR.any_bits_in_range(group_first, group_last, &[BitField::ColdRo])
            {
                // Group contains cold pages inside one of the ranges: ask each
                // interval individually so cold pages are never walked by `GetWriteWatch`.
                for &(a, b) in &merged[i..k] {
                    ask_merged_interval(
                        a,
                        b,
                        frame32,
                        threshold,
                        &mut batch_dirty,
                        &mut covered,
                        &mut to_promote,
                    );
                }
            } else if let Some(dirty) = call_get_write_watch(group_first, group_last, 0) {
                batch_dirty.extend_from_slice(&dirty);
                for &(a, b) in &merged[i..k] {
                    DIR.note_clean_for_cold(a, b, &dirty, frame32, threshold, &mut to_promote);
                    push_merged_interval(&mut covered, a, b);
                }
            } else {
                // Bridged group crossed a boundary: drop bridge and ask each
                // merged interval directly without bisecting across the gaps.
                for &(a, b) in &merged[i..k] {
                    ask_merged_interval(
                        a,
                        b,
                        frame32,
                        threshold,
                        &mut batch_dirty,
                        &mut covered,
                        &mut to_promote,
                    );
                }
            }
            i = k;
        }
        merge_dirty_parts(&mut batch_dirty);
        if batch_dirty.is_empty() {
            next_epoch()
        } else {
            DIR.publish_written_parts(&batch_dirty)
        }
    };

    if !to_promote.is_empty() && threshold > 0 {
        promote_cold_pages(&to_promote);
    }
    (now, covered)
}

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`].
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    if ask_kept() {
        return;
    }
    let (now, covered) = collect_and_ask_batch(ranges, false);
    if covered.is_empty() {
        return;
    }
    let mut ahead = AHEAD.write().unwrap_or_else(|e| e.into_inner());
    if ahead.kept {
        return;
    }
    ahead.epoch = now;
    ahead.ranges = covered;
    set_ahead_flags(false, !ahead.ranges.is_empty());
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's.
#[cfg(feature = "diagnostics")]
static SLOW: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// Since the last frame line: ranges of view memory the guards hold that a
/// submission's askings ahead ([`ask_ahead`], [`ask_once`]) left out.
#[cfg(feature = "diagnostics")]
pub(crate) static AHEAD_VIEWS: AtomicU64 = AtomicU64::new(0);

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
#[allow(dead_code)]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    if ask_kept() {
        return;
    }
    let mut ahead = AHEAD.write().unwrap_or_else(|e| e.into_inner());
    if !ahead.kept {
        ahead.ranges.clear();
        set_ahead_flags(false, false);
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; on by default).
pub(crate) fn once() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    ask_once_end();
    if ranges.is_empty() {
        return;
    }
    // Snapshot active kernel-read announcements and the monotonic sequence
    // before querying the kernel so any read already in flight or announced
    // concurrently is never hidden behind a stale ahead interval.
    let seq_before = ANNOUNCED_SEQ.load(Ordering::SeqCst);
    let pending_before = snapshot_announced();
    let (now, mut covered) = collect_and_ask_batch(ranges, true);
    for &(p_first, p_last) in &pending_before {
        covered.retain(|&(a, b)| b <= p_first || p_last <= a);
    }
    let mut ahead = AHEAD.write().unwrap_or_else(|e| e.into_inner());
    if ANNOUNCED_SEQ.load(Ordering::SeqCst) != seq_before {
        // A `kernel_will_write` or `title_memory_remapped` landed concurrently
        // while `collect_and_ask_batch` ran; drop the batch cache for this
        // submission so no kernel write or remap can be hidden behind a stale
        // ahead interval.
        covered.clear();
    }
    ahead.epoch = now;
    ahead.ranges = covered;
    let kept = !ahead.ranges.is_empty();
    ahead.kept = kept;
    set_ahead_flags(kept, kept);
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    if !ask_kept() {
        return;
    }
    let mut ahead = AHEAD.write().unwrap_or_else(|e| e.into_inner());
    if ahead.kept {
        ahead.kept = false;
        ahead.ranges.clear();
        set_ahead_flags(false, false);
    }
}

/// Title memory was written in a way the journal was not told: no asking stands.
pub(crate) fn ask_once_void() {
    if !ask_kept() {
        return;
    }
    let mut ahead = AHEAD.write().unwrap_or_else(|e| e.into_inner());
    ahead.kept = false;
    ahead.ranges.clear();
    set_ahead_flags(false, false);
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    if len == 0 || !ask_kept() {
        return;
    }
    let mut ahead = AHEAD.write().unwrap_or_else(|e| e.into_inner());
    if !ahead.kept {
        return;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    let frame32 = current_frame32();
    let now = DIR.publish_written_range(first, last);
    ahead.epoch = ahead.epoch.max(now);
    DIR.reset_cold_range(first, last, frame32);
}

/// We wrote the 64 KiB page numbered `page` of title memory through a
/// view of our own (`crate::inplace::write_own`): no guard was opened and
/// no kernel write-watch bit set, so the journal is told here.
pub(crate) fn wrote_own(page: usize, others: &[usize]) -> (u64, u64) {
    let (before, now) = {
        let _w = DirWriteGuard::enter();
        let now = next_epoch();
        let before = DIR.record_written_prev(page, now);
        for &other in others {
            if other != page {
                DIR.record_written(other, now);
            }
        }
        (before, now)
    };
    if AHEAD_ACTIVE.load(Ordering::Acquire) {
        let mut ahead = AHEAD.write().unwrap_or_else(|e| e.into_inner());
        if ahead.kept || !ahead.ranges.is_empty() {
            ahead.epoch = ahead.epoch.max(now);
        }
    }
    (before, now)
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]).
/// `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    if start >= end || !AHEAD_ACTIVE.load(Ordering::Acquire) {
        return None;
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let ahead_epoch = {
        let ahead = AHEAD.read().unwrap_or_else(|e| e.into_inner());
        if !ahead_covers(&ahead.ranges, first, last) {
            return None;
        }
        ahead.epoch
    };
    Some(DIR.collect_written_consistent(first, last, ahead_epoch))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    if start >= end {
        let now = ASK_EPOCH.load(Ordering::Acquire);
        return Some((now, Vec::new()));
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));

    // Fast path 1: covered by `ask_once` / `ask_ahead` — lock-free `PageDir` read.
    if AHEAD_ACTIVE.load(Ordering::Acquire) {
        let ahead_epoch = {
            let ahead = AHEAD.read().unwrap_or_else(|e| e.into_inner());
            if ahead_covers(&ahead.ranges, first, last) {
                Some(ahead.epoch)
            } else {
                None
            }
        };
        if let Some(epoch) = ahead_epoch {
            return Some(DIR.collect_written_consistent(first, last, epoch));
        }
    }

    // Fast path 2: known view memory (`in_view` or `known_view`) goes straight
    // to guards without incurring a failing `GetWriteWatch` syscall.
    if guarded && (in_view(first, last) || known_view(first * PAGE)) {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let (published, _) = guard_and_publish(first * PAGE, last * PAGE)?;
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
        }
        let now = if published != 0 {
            published
        } else {
            next_epoch()
        };
        return Some(DIR.collect_written_consistent(first, last, now));
    }

    // Path 3: non-view write-watched memory (skipping `cold_ro` sub-runs).
    // Evaluate `known_view` / `in_view` before taking `WATCH_SERIAL` so
    // `WATCH_SERIAL` never nests `GUARDS` even under `VICTORY_WATCH_LOCKFREE=0`.
    let kv = guarded && known_view(first * PAGE);
    let view = kv || in_view(first, last);
    let threshold = cold_after().min(u8::MAX as u32) as u8;
    let frame32 = current_frame32();
    let mut to_promote = Vec::new();
    let result: Option<u64> = if view {
        None
    } else {
        let _serial = WATCH_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let non_cold = if v2_enabled() && threshold > 0 {
            DIR.collect_clear_runs(first, last, BitField::ColdRo)
        } else {
            vec![(first, last)]
        };
        if non_cold.is_empty() {
            if asks_on() {
                CLEAN[4].fetch_add(1, Ordering::Relaxed);
                CLEAN[5].fetch_add((last - first) as u64, Ordering::Relaxed);
            }
            Some(next_epoch())
        } else {
            let mut last_epoch = 0u64;
            let mut failed_from: Option<usize> = None;
            for (idx, &(r0, r1)) in non_cold.iter().enumerate() {
                if let Some(dirty) = call_get_write_watch(r0, r1, 2) {
                    // Record each successful reset in `DIR` immediately under
                    // `WATCH_SERIAL` before querying the next run so a later
                    // failure can never drop an already-reset write bit.
                    if !dirty.is_empty() {
                        last_epoch = DIR.publish_written_parts(&dirty);
                    }
                    DIR.note_clean_for_cold(r0, r1, &dirty, frame32, threshold, &mut to_promote);
                } else {
                    failed_from = Some(idx);
                    break;
                }
            }
            if let Some(start_idx) = failed_from {
                let mut all_ok = true;
                for &(r0, r1) in &non_cold[start_idx..] {
                    let (ok, pages) = by_allocation(r0 * PAGE, r1 * PAGE);
                    if !pages.is_empty() {
                        last_epoch = DIR.publish_written_parts(&pages);
                    }
                    if !ok {
                        all_ok = false;
                        break;
                    }
                }
                if all_ok {
                    Some(if last_epoch != 0 {
                        last_epoch
                    } else {
                        next_epoch()
                    })
                } else {
                    #[cfg(feature = "diagnostics")]
                    if guarded {
                        failed_asking(first * PAGE, last * PAGE);
                    }
                    None
                }
            } else {
                if asks_on() {
                    let non_cold_pages: usize = non_cold.iter().map(|&(a, b)| b - a).sum();
                    if non_cold_pages < last - first {
                        CLEAN[4].fetch_add(1, Ordering::Relaxed);
                        CLEAN[5]
                            .fetch_add(((last - first) - non_cold_pages) as u64, Ordering::Relaxed);
                    }
                }
                Some(if last_epoch != 0 {
                    last_epoch
                } else {
                    next_epoch()
                })
            }
        }
    };

    if !to_promote.is_empty() && threshold > 0 {
        promote_cold_pages(&to_promote);
    }

    if let Some(now) = result {
        return Some(DIR.collect_written_consistent(first, last, now));
    }

    if !guarded {
        return None;
    }

    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let (published, _) = guard_and_publish(first * PAGE, last * PAGE)?;
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    let now = if published != 0 {
        published
    } else {
        next_epoch()
    };
    Some(DIR.collect_written_consistent(first, last, now))
}

/// The pages (and 4 KiB part masks) of `[start, end)` written, asked allocation
/// by allocation: the kernel refuses a range over several (a title buffer spans
/// as many allocations as the runtime made for it). Parts never committed
/// cannot have been written, so a range none of which is committed (a reserved
/// allocation the title has not committed yet) has no writes. Returns
/// `(all_watched, reset_dirty_pages)` so even when a later allocation is not
/// write-watched (`all_watched == false`), any dirty pages already reset by
/// earlier allocations are returned and recorded rather than lost.
fn by_allocation(start: usize, end: usize) -> (bool, Vec<(usize, u16)>) {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_FREE, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let query = |at: usize| {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query about an address; nothing is dereferenced.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        (n != 0).then_some(info)
    };
    let (mut pages, mut at, mut allocations, mut any) = (Vec::new(), start, 0, false);
    while at < end {
        let Some(info) = query(at) else {
            merge_dirty_parts(&mut pages);
            return (false, pages);
        };
        // A committed section view cannot belong to a MEM_WRITE_WATCH
        // reservation. Its other protection regions need no further queries.
        if info.State == MEM_COMMIT && info.Type == MEM_MAPPED {
            merge_dirty_parts(&mut pages);
            return (false, pages);
        }
        let base = info.AllocationBase as usize;
        let mut committed = info.State == MEM_COMMIT;
        let mut to = (info.BaseAddress as usize)
            .saturating_add(info.RegionSize)
            .min(end);
        // The rest of the allocation's regions in the range.
        while info.State != MEM_FREE && to < end {
            let Some(next) = query(to) else {
                merge_dirty_parts(&mut pages);
                return (false, pages);
            };
            if next.State == MEM_FREE || next.AllocationBase as usize != base {
                break;
            }
            committed |= next.State == MEM_COMMIT;
            to = (next.BaseAddress as usize)
                .saturating_add(next.RegionSize)
                .min(end);
        }
        allocations += 1;
        any |= committed;
        if committed {
            let needed = (to - at) / KERNEL_PAGE;
            let ok = with_watch_scratch(needed, |found| {
                let mut count = needed;
                let mut granularity = 0;
                // SAFETY: the answer buffer holds `count` entries.
                let failed = unsafe {
                    GetWriteWatch(
                        WRITE_WATCH_FLAG_RESET,
                        at as *const _,
                        to - at,
                        found.as_mut_ptr().cast(),
                        &mut count,
                        &mut granularity,
                    )
                } != 0;
                if failed {
                    return false;
                }
                for &a in &found[..count] {
                    let page = a / PAGE;
                    let part = 1u16 << ((a % PAGE) / KERNEL_PAGE);
                    match pages.last_mut() {
                        Some((last, parts)) if *last == page => *parts |= part,
                        _ => pages.push((page, part)),
                    }
                }
                true
            });
            if !ok {
                merge_dirty_parts(&mut pages);
                return (false, pages);
            }
        }
        at = to;
    }
    merge_dirty_parts(&mut pages);
    (allocations > 1 || !any, pages)
}

/// Diagnostics: one in 2,048 of the askings the kernel's write watch
/// refused for memory the guards did not know.
#[cfg(feature = "diagnostics")]
fn failed_asking(start: usize, end: usize) {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    static N: AtomicU64 = AtomicU64::new(0);
    if !N.fetch_add(1, Ordering::Relaxed).is_multiple_of(2048) {
        return;
    }
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    let b = start / BLOCK;
    let (protected, clean, absent, viewed, aliased) = DIR.guard_flags(b);
    tracing::info!(
        "a failed write-watch asking (one in 2048): {start:#x}+{:#x}; Windows says (answered {n}): base {:#x} region {:#x} state {:#x} type {:#x} protect {:#x} allocation protect {:#x}; \
         the guards of its first block: protected {protected} clean {clean} absent {absent} viewed {viewed} aliased {aliased}; is a view: {}",
        end - start,
        info.BaseAddress as usize,
        info.RegionSize,
        info.State,
        info.Type,
        info.Protect,
        info.AllocationProtect,
        has_mapped(start, end)
    );
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    #[allow(dead_code)]
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut new: Vec<(usize, u64)> = Vec::new();
        for (page, write) in pages {
            let last = seen.get(&page).copied();
            if last.is_some_and(|last| last >= write)
                || (page * PAGE).max(from) >= ((page + 1) * PAGE).min(to)
            {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from && ((page + 1) * PAGE).min(self.end) <= to {
                seen.insert(page, write);
            }
            new.push((page, last.unwrap_or(0)));
        }
        // Of a page taken before, take only the 4 KiB parts written since
        // (`VICTORY_TAKE_PARTS=0`: whole 64 KiB pages).
        let parts = if take_parts_on() {
            parts_written(&new)
        } else {
            vec![WHOLE; new.len()]
        };
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (&(page, last), &parts) in new.iter().zip(&parts) {
            let mut parts = if last == 0 { WHOLE } else { parts };
            while parts != 0 {
                let first = parts.trailing_zeros() as usize;
                let past = first + (!(parts >> first)).trailing_zeros() as usize;
                parts &= if past >= PARTS { 0 } else { u16::MAX << past };
                let (start, end) = (
                    (page * PAGE + first * KERNEL_PAGE).max(from),
                    (page * PAGE + past.min(PARTS) * KERNEL_PAGE).min(to),
                );
                if start >= end {
                    continue;
                }
                match ranges.last_mut() {
                    Some((_, last)) if *last == start => *last = end,
                    _ => ranges.push((start, end)),
                }
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Whether skipping kernel write-watch calls for known view blocks is disabled
/// (`VICTORY_VIEW_ASK=1` or `VICTORY_GUARD_ASK=1`).
fn view_ask_disabled() -> bool {
    static ASK: OnceLock<bool> = OnceLock::new();
    *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    })
}

/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, or noted absent).
fn known_view(start: usize) -> bool {
    if view_ask_disabled() {
        return false;
    }
    let first = start / BLOCK;
    if lockfree_enabled() {
        return DIR.is_known_view(first, view_known());
    }
    let _g = guard_shard_read(shard_of(first));
    DIR.is_known_view(first, view_known())
}

/// Number of contiguous 64 KiB blocks mapped to one guard-shard stripe
/// (`1 << 4` = 16 blocks = 1 MiB) so contiguous runs stay in one shard while
/// disjoint buffers/textures land on independent shards.
const SHARD_BLOCK_SHIFT: usize = 4;
const SHARD_BLOCKS: usize = 1 << SHARD_BLOCK_SHIFT;
/// Power-of-two number of independent guard shards (128 MiB wrap-around period,
/// avoiding collisions between 64 MiB-aligned pools while keeping `ShardMask`
/// in a single `u128`).
const GUARD_SHARDS: usize = 128;

#[inline]
const fn shard_of(block: usize) -> usize {
    (block >> SHARD_BLOCK_SHIFT) & (GUARD_SHARDS - 1)
}

/// Splits `[first, last)` into maximal sub-runs where every block belongs to
/// the same guard shard, invoking `f(shard, sub_first, sub_last)` in order.
#[inline]
fn for_each_shard_run(first: usize, last: usize, mut f: impl FnMut(usize, usize, usize)) {
    let mut cur = first;
    while cur < last {
        let stripe_end = (cur & !(SHARD_BLOCKS - 1)).saturating_add(SHARD_BLOCKS);
        let next = stripe_end.min(last);
        f(shard_of(cur), cur, next);
        cur = next;
    }
}

/// Bitset of guard shard indices `0..GUARD_SHARDS`. Iterating set bits via
/// `trailing_zeros()` always yields shard indices in strictly ascending order.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct ShardMask(u128);

impl ShardMask {
    const EMPTY: Self = Self(0);

    #[inline]
    const fn single(shard: usize) -> Self {
        Self(1u128 << shard)
    }

    #[inline]
    fn insert(&mut self, shard: usize) {
        self.0 |= 1u128 << shard;
    }

    #[inline]
    fn insert_block(&mut self, block: usize) {
        self.insert(shard_of(block));
    }

    #[inline]
    fn insert_range(&mut self, first: usize, last: usize) {
        if first >= last {
            return;
        }
        let s0 = first >> SHARD_BLOCK_SHIFT;
        let s1 = (last - 1) >> SHARD_BLOCK_SHIFT;
        if s1 - s0 + 1 >= GUARD_SHARDS {
            self.0 = u128::MAX;
            return;
        }
        for s in s0..=s1 {
            self.insert(s & (GUARD_SHARDS - 1));
        }
    }

    #[inline]
    fn is_subset_of(self, other: Self) -> bool {
        (self.0 & !other.0) == 0
    }

    #[inline]
    fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    fn count(self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    fn for_each(self, mut f: impl FnMut(usize)) {
        let mut rem = self.0;
        while rem != 0 {
            let s = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            f(s);
        }
    }
}

struct GuardShard {
    /// Timed pins for blocks in this shard the kernel is about to write
    /// ([`kernel_writing`]), keyed by disjoint `[first, last)` block intervals
    /// clipped to stripes belonging to this shard, with their expiry.
    pins: BTreeMap<usize, (usize, std::time::Instant)>,
    /// Ref-counted kernel-read announcements ([`kernel_will_write`] /
    /// [`kernel_wrote`]) overlapping this shard, stored as block ranges
    /// `[first, end)` with open count so saturated lengths (`usize::MAX`) cost
    /// O(touched shards <= 128) and overlapping announcements never prematurely
    /// drop on the first `kernel_wrote`.
    announced: BTreeMap<(usize, usize), u32>,
    /// View blocks in this shard opened by a CPU write fault, and when ([`rest`]).
    opened: Option<std::collections::HashMap<usize, std::time::Instant>>,
}

/// Cache-line padded shard slot to prevent false sharing between adjacent shards.
#[repr(align(64))]
struct GuardShardSlot(RwLock<GuardShard>);

/// Whether a block once guarded stays known as view memory while its
/// guard is open (`VICTORY_VIEW_KNOWN=1` or `alt`; off by default).
fn view_known() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_VIEW_KNOWN").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            let on = (EPOCH
                .get_or_init(std::time::Instant::now)
                .elapsed()
                .as_secs()
                / 20)
                % 2
                == 1;
            static LAST: AtomicU8 = AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "view memory stays known while its guard is open from now (VICTORY_VIEW_KNOWN=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// Sharded guard metadata table.
///
/// Global lock order (strictly acyclic):
/// `WATCH_SERIAL` -> `GUARD_SHARDS_TABLE[0..GUARD_SHARDS]` (strictly ascending
/// shard index) -> `inplace::VIEWS` / `lazy_back::LAZY_STATE` (leaf locks).
/// `descriptor::host::REMAPPED` is never acquired while any guard shard is held.
static GUARD_SHARDS_TABLE: [GuardShardSlot; GUARD_SHARDS] = [const {
    GuardShardSlot(RwLock::new(GuardShard {
        pins: BTreeMap::new(),
        announced: BTreeMap::new(),
        opened: None,
    }))
}; GUARD_SHARDS];

/// 128-bit atomic summary of which guard shards currently have non-empty
/// `announced` maps, allowing [`ask_once`] to skip shard reads completely when
/// no kernel read is in flight. Updated while holding the corresponding shard's
/// write lock.
static ANNOUNCED_SHARD_MASK: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Monotonic counter incremented on every [`kernel_will_write`] and
/// [`title_memory_remapped`] so [`ask_once`] can detect a concurrent
/// announcement or remap that landed while `GetWriteWatch` was running.
static ANNOUNCED_SEQ: AtomicU64 = AtomicU64::new(0);

#[inline]
fn set_shard_announced_bit(shard: usize, non_empty: bool) {
    let word = shard / 64;
    let bit = 1u64 << (shard % 64);
    if non_empty {
        ANNOUNCED_SHARD_MASK[word].fetch_or(bit, Ordering::SeqCst);
    } else {
        ANNOUNCED_SHARD_MASK[word].fetch_and(!bit, Ordering::SeqCst);
    }
}

/// Snapshots all currently active `announced` block ranges without locking any
/// shard when no kernel read is active.
fn snapshot_announced() -> Vec<(usize, usize)> {
    let lo = ANNOUNCED_SHARD_MASK[0].load(Ordering::SeqCst);
    let hi = ANNOUNCED_SHARD_MASK[1].load(Ordering::SeqCst);
    if lo == 0 && hi == 0 {
        return Vec::new();
    }
    let mask = ShardMask((lo as u128) | ((hi as u128) << 64));
    let mut out = Vec::new();
    mask.for_each(|s| {
        let g = guard_shard_read(s);
        out.extend(g.announced.keys().copied());
    });
    out.sort_unstable();
    out.dedup();
    out
}

/// How long blocks stay writable after [`kernel_writing`].
#[allow(dead_code)]
const PIN: std::time::Duration = std::time::Duration::from_secs(2);

impl GuardShard {
    /// Whether an announced kernel write still to land covers `block`.
    fn announced(&self, block: usize) -> bool {
        self.announced
            .keys()
            .any(|&(first, end)| (first..end).contains(&block))
    }

    /// Whether `block` is resting writable after a recent CPU write fault.
    fn rests(&self, block: usize, rest: std::time::Duration) -> bool {
        !rest.is_zero()
            && self
                .opened
                .as_ref()
                .and_then(|m| m.get(&block))
                .is_some_and(|at| at.elapsed() < rest)
    }

    /// Records that a CPU write fault just opened `block`.
    fn opened(&mut self, block: usize) {
        const MANY: usize = 1 << 10;
        let map = self.opened.get_or_insert_with(Default::default);
        if map.len() >= MANY {
            map.retain(|_, at| at.elapsed().as_millis() < 50);
            if map.len() >= MANY {
                map.clear();
            }
        }
        map.insert(block, std::time::Instant::now());
    }

    /// Forgets fault-rest timestamps for remapped/unmapped blocks `[first, last)`.
    fn forget_opened(&mut self, first: usize, last: usize) {
        if let Some(map) = self.opened.as_mut()
            && !map.is_empty()
        {
            map.retain(|&b, _| b < first || b >= last);
        }
    }

    /// Whether `block` is left writable for a kernel write (either timed
    /// [`kernel_writing`] pin or active [`kernel_will_write`] announcement).
    fn pinned(&self, block: usize) -> bool {
        if self.announced(block) {
            return true;
        }
        if self.pins.is_empty() {
            return false;
        }
        self.pins
            .range(..=block)
            .next_back()
            .is_some_and(|(_, &(end, _))| block < end)
    }

    /// Marks `[first, last)` pinned until at least `until`, coalescing any
    /// overlapping or adjacent intervals so `pins` stays compact and disjoint.
    /// Caller must clip `[first, last)` to blocks belonging to this shard.
    #[allow(dead_code)]
    fn pin(&mut self, mut first: usize, mut last: usize, mut until: std::time::Instant) {
        if first >= last {
            return;
        }
        let scan_from = self
            .pins
            .range(..=first)
            .next_back()
            .filter(|&(_, &(p1, _))| p1 >= first)
            .map_or(first, |(&k, _)| k);
        let merged: Vec<(usize, usize, std::time::Instant)> = self
            .pins
            .range(scan_from..=last)
            .map(|(&p0, &(p1, u))| (p0, p1, u))
            .collect();
        for (p0, p1, u) in merged {
            self.pins.remove(&p0);
            first = first.min(p0);
            last = last.max(p1);
            if u > until {
                until = u;
            }
        }
        self.pins.insert(first, (last, until));
    }

    fn prune_pins(&mut self) {
        if !self.pins.is_empty() {
            let now = std::time::Instant::now();
            self.pins.retain(|_, &mut (_, until)| until > now);
        }
    }

    /// Removes `[first, last)` from active pins (the mapping there is gone or
    /// replaced, so a reused address must not inherit an old view's grace).
    fn unpin(&mut self, first: usize, last: usize) {
        if self.pins.is_empty() || first >= last {
            return;
        }
        let scan_from = self
            .pins
            .range(..first)
            .next_back()
            .filter(|&(_, &(p1, _))| p1 > first)
            .map_or(first, |(&k, _)| k);
        let hits: Vec<(usize, usize, std::time::Instant)> = self
            .pins
            .range(scan_from..last)
            .map(|(&p0, &(p1, u))| (p0, p1, u))
            .collect();
        for (p0, p1, until) in hits {
            self.pins.remove(&p0);
            if p0 < first {
                self.pins.insert(p0, (first, until));
            }
            if last < p1 {
                self.pins.insert(last, (p1, until));
            }
        }
    }
}

/// Lock-free `GuardClean` bitmap fast path; `VICTORY_GUARD_BITS=0` takes the
/// shard lock on every guard check, as before.
fn guard_bits_fast_enabled() -> bool {
    static BITS: OnceLock<bool> = OnceLock::new();
    lockfree_enabled()
        && *BITS.get_or_init(|| std::env::var_os("VICTORY_GUARD_BITS").is_none_or(|v| v != "0"))
}

#[inline]
fn guard_shard(shard: usize) -> std::sync::RwLockWriteGuard<'static, GuardShard> {
    GUARD_SHARDS_TABLE[shard]
        .0
        .write()
        .unwrap_or_else(|e| e.into_inner())
}

#[inline]
fn guard_shard_read(shard: usize) -> std::sync::RwLockReadGuard<'static, GuardShard> {
    GUARD_SHARDS_TABLE[shard]
        .0
        .read()
        .unwrap_or_else(|e| e.into_inner())
}

/// Proof token that the required guard shard write locks are currently held.
trait ShardAccess {
    fn pinned(&self, block: usize) -> bool;
    fn rests(&self, block: usize, rest: std::time::Duration) -> bool;
    fn debug_assert_locked(&self, block: usize);
}

#[inline]
fn debug_assert_range_locked(access: &impl ShardAccess, first: usize, last: usize) {
    #[cfg(debug_assertions)]
    {
        if first >= last {
            return;
        }
        let mut mask = ShardMask::EMPTY;
        mask.insert_range(first, last);
        mask.for_each(|s| access.debug_assert_locked(s << SHARD_BLOCK_SHIFT));
    }
    #[cfg(not(debug_assertions))]
    let _ = (access, first, last);
}

struct SingleShard<'a> {
    shard: usize,
    guard: &'a GuardShard,
}

impl ShardAccess for SingleShard<'_> {
    #[inline]
    fn pinned(&self, block: usize) -> bool {
        debug_assert_eq!(shard_of(block), self.shard);
        self.guard.pinned(block)
    }

    #[inline]
    fn rests(&self, block: usize, rest: std::time::Duration) -> bool {
        debug_assert_eq!(shard_of(block), self.shard);
        self.guard.rests(block, rest)
    }

    #[inline]
    fn debug_assert_locked(&self, block: usize) {
        debug_assert_eq!(shard_of(block), self.shard);
    }
}

enum GuardLockInner {
    Empty,
    Single(usize, std::sync::RwLockWriteGuard<'static, GuardShard>),
    Multi(Vec<(usize, std::sync::RwLockWriteGuard<'static, GuardShard>)>),
}

/// Holds write locks for a set of guard shards, ALWAYS acquired in strictly
/// ascending shard index order (`0..GUARD_SHARDS`) for deadlock freedom.
struct GuardLockSet {
    mask: ShardMask,
    inner: GuardLockInner,
}

impl GuardLockSet {
    fn lock_mask(mask: ShardMask) -> Self {
        let count = mask.count();
        let inner = match count {
            0 => GuardLockInner::Empty,
            1 => {
                let s = mask.0.trailing_zeros() as usize;
                GuardLockInner::Single(s, guard_shard(s))
            }
            _ => {
                let mut guards = Vec::with_capacity(count as usize);
                mask.for_each(|s| guards.push((s, guard_shard(s))));
                GuardLockInner::Multi(guards)
            }
        };
        Self { mask, inner }
    }

    fn lock_range(first: usize, last: usize) -> Self {
        let mut mask = ShardMask::EMPTY;
        mask.insert_range(first, last);
        Self::lock_mask(mask)
    }

    fn lock_blocks(blocks: &[usize]) -> Self {
        let mut mask = ShardMask::EMPTY;
        for &b in blocks {
            mask.insert_block(b);
        }
        Self::lock_mask(mask)
    }

    #[inline]
    fn shard(&self, s: usize) -> &GuardShard {
        match &self.inner {
            GuardLockInner::Single(idx, g) if *idx == s => g,
            GuardLockInner::Multi(v) => {
                let pos = v
                    .binary_search_by_key(&s, |&(idx, _)| idx)
                    .expect("guard shard must be locked");
                &v[pos].1
            }
            _ => panic!("guard shard {s} not locked"),
        }
    }

    #[inline]
    fn shard_mut(&mut self, s: usize) -> &mut GuardShard {
        match &mut self.inner {
            GuardLockInner::Single(idx, g) if *idx == s => g,
            GuardLockInner::Multi(v) => {
                let pos = v
                    .binary_search_by_key(&s, |&(idx, _)| idx)
                    .expect("guard shard must be locked");
                &mut v[pos].1
            }
            _ => panic!("guard shard {s} not locked"),
        }
    }

    fn for_each_mut(&mut self, mut f: impl FnMut(usize, &mut GuardShard)) {
        match &mut self.inner {
            GuardLockInner::Empty => {}
            GuardLockInner::Single(s, g) => f(*s, g),
            GuardLockInner::Multi(v) => {
                for (s, g) in v {
                    f(*s, g);
                }
            }
        }
    }

    fn prune_pins(&mut self) {
        self.for_each_mut(|_, g| g.prune_pins());
    }

    #[allow(dead_code)]
    fn pin_range(&mut self, first: usize, last: usize, until: std::time::Instant) {
        for_each_shard_run(first, last, |s, lo, hi| {
            self.shard_mut(s).pin(lo, hi, until);
        });
    }

    fn unpin_range(&mut self, first: usize, last: usize) {
        self.for_each_mut(|_, g| g.unpin(first, last));
    }

    fn forget_opened_range(&mut self, first: usize, last: usize) {
        self.for_each_mut(|_, g| g.forget_opened(first, last));
    }

    fn announce_range(&mut self, range: (usize, usize)) {
        self.for_each_mut(|s, g| {
            *g.announced.entry(range).or_default() += 1;
            set_shard_announced_bit(s, true);
        });
    }

    fn unannounce_range(&mut self, range: (usize, usize)) {
        self.for_each_mut(|s, g| {
            if let Some(n) = g.announced.get_mut(&range) {
                *n -= 1;
                if *n == 0 {
                    g.announced.remove(&range);
                }
            }
            set_shard_announced_bit(s, !g.announced.is_empty());
        });
    }

    fn clear_announced_inside(&mut self, first: usize, end: usize) {
        self.for_each_mut(|s, g| {
            g.announced
                .retain(|&(from, to), _| !(first <= from && to <= end));
            set_shard_announced_bit(s, !g.announced.is_empty());
        });
    }
}

impl ShardAccess for GuardLockSet {
    #[inline]
    fn pinned(&self, block: usize) -> bool {
        self.shard(shard_of(block)).pinned(block)
    }

    #[inline]
    fn rests(&self, block: usize, rest: std::time::Duration) -> bool {
        self.shard(shard_of(block)).rests(block, rest)
    }

    #[inline]
    fn debug_assert_locked(&self, block: usize) {
        debug_assert!(self.mask.0 & (1u128 << shard_of(block)) != 0);
    }
}

/// Acquires read locks for all guard shards covering `[first, last)` in
/// strictly ascending shard order (only used when `VICTORY_WATCH_LOCKFREE=0`).
fn lock_shards_read_range(
    first: usize,
    last: usize,
) -> Vec<std::sync::RwLockReadGuard<'static, GuardShard>> {
    let mut mask = ShardMask::EMPTY;
    mask.insert_range(first, last);
    let mut guards = Vec::with_capacity(mask.count() as usize);
    mask.for_each(|s| guards.push(guard_shard_read(s)));
    guards
}

/// Collects `(block, alias_blocks)` for every block in `[from, to)` that has
/// `2..=MANY_VIEWS` views in `inplace`, holding `inplace::VIEWS` as a leaf lock.
fn collect_aliases(from: usize, to: usize) -> Vec<(usize, Vec<usize>)> {
    let mut entries: Vec<(usize, Vec<usize>)> = Vec::new();
    if !view_guard() || from >= to {
        return entries;
    }
    crate::inplace::for_each_alias_in(
        from.saturating_mul(BLOCK),
        to.saturating_mul(BLOCK),
        |block, shown| {
            entries.push((block, shown.iter().map(|&a| a / BLOCK).collect()));
        },
    );
    entries
}

/// Locks `base_mask` plus every guard shard touched by aliases of `[from, to)`
/// in strictly ascending shard order. Retries (dropping all locks first) only
/// if a concurrent view mapping added an alias in a shard outside `mask`.
fn lock_with_aliases(
    mut mask: ShardMask,
    from: usize,
    to: usize,
) -> (GuardLockSet, Vec<(usize, Vec<usize>)>) {
    loop {
        let locked = GuardLockSet::lock_mask(mask);
        let entries = collect_aliases(from, to);
        let mut needed = mask;
        for (_, aliases) in &entries {
            for &a in aliases {
                needed.insert_block(a);
            }
        }
        if needed.is_subset_of(mask) {
            return (locked, entries);
        }
        drop(locked);
        mask = needed;
    }
}

/// Locks all guard shards covering `[first, last)` plus every alias block
/// (both current `inplace` aliases and any `GuardAliased` blocks) in strictly
/// ascending shard order.
#[allow(dead_code)]
fn lock_range_and_all_aliases(first: usize, last: usize) -> (GuardLockSet, Vec<usize>) {
    let mut mask = ShardMask::EMPTY;
    mask.insert_range(first, last);
    loop {
        let locked = GuardLockSet::lock_mask(mask);
        let mut alias_blocks = Vec::new();
        if view_guard() {
            crate::inplace::for_each_alias_in(
                first.saturating_mul(BLOCK),
                last.saturating_mul(BLOCK),
                |block, shown| {
                    for &alias in shown {
                        let alias_block = alias / BLOCK;
                        if alias_block != block {
                            alias_blocks.push(alias_block);
                        }
                    }
                },
            );
        }
        for block in DIR.collect_set_bits(first, last, BitField::GuardAliased) {
            for alias in crate::inplace::pages_of(block * BLOCK) {
                let ab = alias / BLOCK;
                if ab < first || ab >= last {
                    alias_blocks.push(ab);
                }
            }
        }
        alias_blocks.sort_unstable();
        alias_blocks.dedup();
        let mut needed = mask;
        for &ab in &alias_blocks {
            needed.insert_block(ab);
        }
        if needed.is_subset_of(mask) {
            return (locked, alias_blocks);
        }
        drop(locked);
        mask = needed;
    }
}

/// Locks the guard shards needed by [`open_under_locked`] for `[first, last)`.
/// Under [`open_one_view`] (default), only `[first, last)`'s shards are locked;
/// otherwise every alias shard of any `GuardAliased` block in `[first, last)`
/// is also locked in strictly ascending order.
fn lock_range_for_open(first: usize, last: usize) -> GuardLockSet {
    let mut mask = ShardMask::EMPTY;
    mask.insert_range(first, last);
    if open_one_view() {
        return GuardLockSet::lock_mask(mask);
    }
    loop {
        let locked = GuardLockSet::lock_mask(mask);
        let mut needed = mask;
        for block in DIR.collect_set_bits(first, last, BitField::GuardAliased) {
            for alias in crate::inplace::pages_of(block * BLOCK) {
                needed.insert_block(alias / BLOCK);
            }
        }
        if needed.is_subset_of(mask) {
            return locked;
        }
        drop(locked);
        mask = needed;
    }
}

/// Cold VEH fallback for `VICTORY_OPEN_ONE_VIEW=0`: locks `block`'s shard and
/// all shards of its `GuardAliased` views in strictly ascending order.
#[cold]
#[inline(never)]
fn lock_block_with_aliases(block: usize) -> GuardLockSet {
    let mut mask = ShardMask::single(shard_of(block));
    loop {
        let locked = GuardLockSet::lock_mask(mask);
        let mut needed = mask;
        if DIR.get_bit(block, BitField::GuardAliased, Ordering::SeqCst) {
            for alias in crate::inplace::pages_of(block * BLOCK) {
                needed.insert_block(alias / BLOCK);
            }
        }
        if needed.is_subset_of(mask) {
            return locked;
        }
        drop(locked);
        mask = needed;
    }
}

/// Whether the block `page` is read-only by our hand and unwritten since a
/// copy took it (diagnostics of `VICTORY_BACK_OWN_CHECK`).
#[allow(dead_code)]
pub(crate) fn clean(page: usize) -> bool {
    if lockfree_enabled() {
        return DIR.is_guard_clean(page);
    }
    let _g = guard_shard_read(shard_of(page));
    DIR.is_guard_clean(page)
}

pub(crate) fn ensure_handler() {
    install();
}

/// Returns `true` if `block` is known view memory in `PageDir` without locking `GUARDS`.
#[inline]
pub(crate) fn is_known_view_block(block: usize) -> bool {
    DIR.is_known_view(block, view_known())
}

/// Records a fresh write epoch and sets `BitField::LazyArmed` for view pages
/// armed with `PAGE_NOACCESS` (lock-free; safe under `LAZY_STATE`).
pub(crate) fn note_lazy_view_armed(armed_vas: &[usize]) {
    let now = next_epoch();
    for &va in armed_vas {
        let b = va / PAGE;
        DIR.set_bit(b, BitField::GuardProtected, false, Ordering::SeqCst);
        DIR.set_bit(b, BitField::GuardClean, false, Ordering::SeqCst);
        DIR.record_written(b, now);
        DIR.set_bit(b, BitField::LazyArmed, true, Ordering::SeqCst);
    }
}

/// Advances the write epoch when an already-armed lazy view entry is reused for
/// another GPU flight (lock-free; safe under `LAZY_STATE`).
pub(crate) fn note_lazy_view_reused(armed_vas: &[usize]) {
    let now = next_epoch();
    for &va in armed_vas {
        DIR.record_written(va / PAGE, now);
    }
}

/// Clears `BitField::LazyArmed` for all pages in `armed_vas` (lock-free).
pub(crate) fn clear_lazy_view_armed(armed_vas: &[usize]) {
    for &va in armed_vas {
        DIR.set_bit(va / PAGE, BitField::LazyArmed, false, Ordering::SeqCst);
    }
}

/// Clears `BitField::LazyArmed` for a single 64 KiB `block` (lock-free).
#[inline]
pub(crate) fn clear_lazy_view_armed_block(block: usize) {
    DIR.set_bit(block, BitField::LazyArmed, false, Ordering::SeqCst);
}

/// Atomically checks pin state, clears guard/cold metadata for `blocks`, and
/// invokes `arm(cpu_clean_after)` under all affected guard shards
/// (`GUARD_SHARDS_TABLE (ascending) -> LAZY_STATE` lock order) so no concurrent
/// `guard()`, `promote_cold_pages()`, or `kernel_writing()` can re-arm or pin
/// `blocks` between disarming and `LAZY_STATE` insertion.
pub(crate) fn try_arm_lazy_under_guards(
    blocks: &[usize],
    arm: impl FnOnce(Option<u64>) -> bool,
) -> bool {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    if blocks.is_empty() {
        return false;
    }
    let mut locked = GuardLockSet::lock_blocks(blocks);
    locked.prune_pins();
    if blocks.iter().any(|&b| locked.pinned(b)) {
        return false;
    }
    let primary_block = blocks[0];
    let was_clean_view = DIR.is_guard_protected(primary_block) && DIR.is_guard_clean(primary_block);
    let cpu_clean_after = was_clean_view.then(|| DIR.written_epoch(primary_block));
    let mut sorted = blocks.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    // Clear all guard/cold metadata across every block BEFORE restoring
    // `PAGE_READWRITE` so no concurrent thread observes stale flags while
    // another alias is already writable.
    let frame32 = current_frame32();
    let mut to_restore = Vec::new();
    for_each_contiguous_run(&sorted, |from, to| {
        to_restore.extend(DIR.collect_set_bits(from, to, BitField::GuardProtected));
        let cold = DIR.collect_set_bits(from, to, BitField::ColdRo);
        if !cold.is_empty() {
            for &b in &cold {
                DIR.reset_cold(b, frame32);
            }
            to_restore.extend(cold);
        }
        DIR.set_bits_range(from, to, BitField::GuardProtected, false, Ordering::SeqCst);
        DIR.set_bits_range(from, to, BitField::GuardClean, false, Ordering::SeqCst);
        DIR.set_bits_range(from, to, BitField::GuardAbsent, false, Ordering::SeqCst);
        DIR.set_bits_range(from, to, BitField::GuardAliased, false, Ordering::SeqCst);
        DIR.set_bits_range(from, to, BitField::ColdRo, false, Ordering::SeqCst);
    });
    if !to_restore.is_empty() {
        to_restore.sort_unstable();
        to_restore.dedup();
        protect_block_runs(&to_restore, PAGE_READWRITE);
    }
    arm(cpu_clean_after)
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them, and publishing their dirty epochs into
/// [`DIR`] strictly before any block becomes visible as clean. Returns
/// `(published_epoch, written_pages)` where `published_epoch == 0` when every
/// block was already clean, or `None` if the range contains no mapped view (or
/// contains committed non-view memory, or VEH installation failed).
fn guard_and_publish(start: usize, end: usize) -> Option<(u64, Vec<usize>)> {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // Every block in `GuardClean` is also in `GuardProtected` (previously
    // verified as mapped view memory and still read-only). When the entire
    // range is clean or lazy-armed, skip both the kernel query and shard locks.
    if guard_bits_fast_enabled() && DIR.all_guard_clean_or_lazy(first, last) {
        #[cfg(feature = "diagnostics")]
        crate::faultbook::note(crate::faultbook::What::Known, first, last, 0);
        return Some((0, Vec::new()));
    }
    // `VICTORY_GUARD_ASK=1` asks Windows (`has_mapped`) on every guard check even
    // when the block is already known to `PageDir` or `inplace`.
    static ASK: OnceLock<bool> = OnceLock::new();
    let known = !*ASK
        .get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"))
        && (DIR.is_known_view(first, view_known())
            || (v2_enabled()
                && crate::inplace::cover(start, end.saturating_sub(start))
                    != crate::inplace::Cover::None));
    if !known && !has_mapped(start, end) {
        return None;
    }
    if !install() {
        return None;
    }
    let rest = rest();
    let mut written = Vec::new();
    let mut max_epoch = 0u64;
    let mut total_publishes = 0usize;
    if guard_bits_fast_enabled() {
        let runs = DIR.collect_non_clean_non_lazy_runs(first, last);
        if runs.is_empty() {
            #[cfg(feature = "diagnostics")]
            crate::faultbook::note(crate::faultbook::What::Asked, first, last, 0);
            return Some((0, written));
        }
        // Split each run at guard-shard boundaries so each pass locks only 1
        // shard and unrelated workers / title write faults never contend.
        for (from, to) in runs {
            for_each_shard_run(from, to, |shard, s_from, s_to| {
                let (ep, pubs) = guard_run_shard(shard, s_from, s_to, &mut written, rest);
                if ep != 0 {
                    max_epoch = max_epoch.max(ep);
                    total_publishes += pubs;
                }
            });
        }
    } else {
        for_each_shard_run(first, last, |shard, s_from, s_to| {
            let (ep, pubs) = guard_run_shard(shard, s_from, s_to, &mut written, rest);
            if ep != 0 {
                max_epoch = max_epoch.max(ep);
                total_publishes += pubs;
            }
        });
    }
    if total_publishes > 1 {
        // Each dirty run already published its pages before marking them
        // clean, so concurrent readers could never miss a write; unify all
        // pages reported by this call under one asking epoch.
        max_epoch = DIR.publish_written(&written);
    }
    #[cfg(feature = "diagnostics")]
    crate::faultbook::note(
        crate::faultbook::What::Asked,
        first,
        last,
        written.len() as u64,
    );
    Some((max_epoch, written))
}

/// Test-facing wrapper around [`guard_and_publish`] returning only the dirty
/// pages of `[start, end)`.
#[cfg(test)]
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    guard_and_publish(start, end).map(|(_, written)| written)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GuardBlockKind {
    End,
    Skip,
    LeaveOpen,
    Reclean,
    Protect,
}

/// [`guard_and_publish`] for a shard-local sub-run `first..last` where every
/// block maps to `shard`. Locks only `shard` on the common path, publishes
/// dirty records into [`DIR`] strictly before making blocks visible as
/// clean/protected/absent, and if a block has an alias in another shard,
/// drops `shard` first and finishes that sub-run under an ascending-order
/// [`GuardLockSet`] before resuming. Returns `(max_epoch, publish_count)`.
fn guard_run_shard(
    shard: usize,
    first: usize,
    last: usize,
    written: &mut Vec<usize>,
    rest: std::time::Duration,
) -> (u64, usize) {
    if first >= last {
        return (0, 0);
    }
    let lazy = crate::lazy_back::active();
    let mut cur = first;
    let mut uncommitted_until = first;
    let mut shard_written: Vec<usize> = Vec::new();
    let mut published_idx = 0usize;
    let mut max_epoch = 0u64;
    let mut publishes = 0usize;
    while cur < last {
        let mut deferred: Option<(usize, usize, ShardMask)> = None;
        let mut next_cur = last;
        {
            let mut g = guard_shard(shard);
            g.prune_pins();
            let mut run: Option<usize> = None;
            let mut reclean: Option<usize> = None;
            for block in cur..=last {
                let kind = if block == last {
                    GuardBlockKind::End
                } else if DIR.is_guard_clean(block) || DIR.is_lazy_armed(block) {
                    GuardBlockKind::Skip
                } else {
                    let mut open = true;
                    if DIR.is_guard_absent(block) {
                        if block >= uncommitted_until {
                            uncommitted_until = uncommitted_end(block);
                        }
                        open = block >= uncommitted_until;
                        if open {
                            DIR.set_bit(block, BitField::GuardAbsent, false, Ordering::SeqCst);
                        }
                    }
                    if !open {
                        GuardBlockKind::Skip
                    } else if g.pinned(block)
                        || (lazy && crate::lazy_back::has_block(block))
                        || g.rests(block, rest)
                    {
                        GuardBlockKind::LeaveOpen
                    } else if DIR.is_guard_protected(block) {
                        GuardBlockKind::Reclean
                    } else {
                        GuardBlockKind::Protect
                    }
                };

                if kind != GuardBlockKind::Protect
                    && let Some(from) = run.take()
                {
                    if published_idx < shard_written.len() {
                        max_epoch =
                            max_epoch.max(DIR.publish_written(&shard_written[published_idx..]));
                        published_idx = shard_written.len();
                        publishes += 1;
                    }
                    if let Err(mask) = try_protect_single_shard(&g, shard, from, block, lazy, rest)
                    {
                        deferred = Some((from, block, mask));
                        next_cur = block;
                        break;
                    }
                }
                if kind != GuardBlockKind::Reclean
                    && let Some(from) = reclean.take()
                {
                    if published_idx < shard_written.len() {
                        max_epoch =
                            max_epoch.max(DIR.publish_written(&shard_written[published_idx..]));
                        published_idx = shard_written.len();
                        publishes += 1;
                    }
                    if let Err(mask) =
                        try_mark_protected_clean_single_shard(&g, shard, from, block, lazy, rest)
                    {
                        deferred = Some((from, block, mask));
                        next_cur = block;
                        break;
                    }
                }

                match kind {
                    GuardBlockKind::End | GuardBlockKind::Skip => {}
                    GuardBlockKind::LeaveOpen => {
                        // Left open (pinned for kernel write, lazy copy-back armed, or
                        // resting after a recent CPU write fault): reported written at
                        // every check until it is eligible to be guarded again.
                        shard_written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
                    }
                    GuardBlockKind::Reclean => {
                        shard_written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
                        reclean.get_or_insert(block);
                    }
                    GuardBlockKind::Protect => {
                        shard_written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
                        run.get_or_insert(block);
                    }
                }
            }
        }
        if let Some((from, to, alias_mask)) = deferred {
            finish_guard_range_multi_shard(shard, alias_mask, from, to, lazy, rest);
        }
        cur = next_cur;
    }
    if published_idx < shard_written.len() {
        max_epoch = max_epoch.max(DIR.publish_written(&shard_written[published_idx..]));
        publishes += 1;
    }
    written.extend_from_slice(&shard_written);
    (max_epoch, publishes)
}

/// Completes guarding/recleaning `[from, to)` (all blocks in `primary_shard`)
/// when at least one block has an alias in another shard, holding all primary
/// and alias shards simultaneously in strictly ascending shard order.
fn finish_guard_range_multi_shard(
    primary_shard: usize,
    initial_alias_mask: ShardMask,
    from: usize,
    to: usize,
    lazy: bool,
    rest: std::time::Duration,
) {
    let base_mask = initial_alias_mask.union(ShardMask::single(primary_shard));
    let (mut locked, entries) = lock_with_aliases(base_mask, from, to);
    locked.prune_pins();

    // Re-evaluate eligibility under the full lock set because `primary_shard`
    // was briefly released during the lock upgrade.
    let mut protect_runs: Vec<(usize, usize)> = Vec::new();
    let mut reclean_runs: Vec<(usize, usize)> = Vec::new();
    for block in from..to {
        if DIR.is_guard_clean(block)
            || DIR.is_lazy_armed(block)
            || DIR.is_guard_absent(block)
            || locked.pinned(block)
            || (lazy && crate::lazy_back::has_block(block))
            || locked.rests(block, rest)
        {
            continue;
        }
        if DIR.is_guard_protected(block) {
            push_merged_interval(&mut reclean_runs, block, block + 1);
        } else {
            push_merged_interval(&mut protect_runs, block, block + 1);
        }
    }

    for (r_from, r_to) in protect_runs {
        let sub_entries: Vec<(usize, Vec<usize>)> = entries
            .iter()
            .filter(|&&(b, _)| b >= r_from && b < r_to)
            .cloned()
            .collect();
        protect_with_access(&locked, sub_entries, r_from, r_to, lazy, rest);
    }
    for (r_from, r_to) in reclean_runs {
        let sub_entries: Vec<(usize, Vec<usize>)> = entries
            .iter()
            .filter(|&&(b, _)| b >= r_from && b < r_to)
            .cloned()
            .collect();
        mark_protected_clean_with_access(&locked, sub_entries, r_from, r_to, lazy, rest);
    }
}

/// Whether `[start, end)` contains a mapped view and no committed non-view
/// memory. Checks `start` first via [`mapped`] (fast path); when `start` is
/// not a view (e.g. an unmapped placeholder/hole inside a view pool), walks
/// `VirtualQuery` across `[start, end)` (unless `VICTORY_GUARD_HOLES=0`) and
/// rejects the range if any committed non-view region is present so
/// private/image memory is never read-protected.
fn has_mapped(start: usize, end: usize) -> bool {
    static HOLES: OnceLock<bool> = OnceLock::new();
    let holes =
        *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    if v2_enabled()
        && crate::inplace::cover(start, end.saturating_sub(start)) != crate::inplace::Cover::None
    {
        return true;
    }
    if mapped(start) {
        return true;
    }
    if !holes {
        return false;
    }
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let mut at = start;
    let mut found_mapped = false;
    while at < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: queries virtual memory metadata without dereferencing `at`.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0
            || info.RegionSize == 0
        {
            break;
        }
        if info.Type == MEM_MAPPED {
            found_mapped = true;
        } else if info.State == MEM_COMMIT {
            return false;
        }
        let Some(next) = (info.BaseAddress as usize).checked_add(info.RegionSize) else {
            break;
        };
        if next <= at {
            break;
        }
        at = next;
    }
    found_mapped
}

/// Whether the current allocation at `at` is a mapped view. Asking only for
/// the allocation type via `QueryVirtualMemoryInformation(MemoryRegionInfo)`
/// avoids enumerating pages with identical protection, which can span an entire
/// large view after its blocks were guarded.
fn mapped(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_MAPPED, MEMORY_BASIC_INFORMATION, MemoryRegionInfo, QueryVirtualMemoryInformation,
        VirtualQuery, WIN32_MEMORY_REGION_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut region = WIN32_MEMORY_REGION_INFORMATION::default();
    // SAFETY: both queries describe an address without dereferencing it. The
    // output buffers match their information classes.
    let queried = unsafe {
        QueryVirtualMemoryInformation(
            GetCurrentProcess(),
            at as *const _,
            MemoryRegionInfo,
            (&raw mut region).cast(),
            size_of_val(&region),
            std::ptr::null_mut(),
        )
    };
    if queried != 0 {
        // These are the Windows SDK's named bitfields in declaration order.
        // windows-sys exposes only the union's raw bits.
        const PRIVATE: u32 = 1 << 0;
        const MAPPED_DATA_FILE: u32 = 1 << 1;
        const MAPPED_IMAGE: u32 = 1 << 2;
        const MAPPED_PAGE_FILE: u32 = 1 << 3;
        const MAPPED_PHYSICAL: u32 = 1 << 4;
        // SAFETY: this union was filled by the successful query above.
        match unsafe { region.Anonymous.Flags } {
            MAPPED_DATA_FILE | MAPPED_PAGE_FILE | MAPPED_PHYSICAL => return true,
            PRIVATE | MAPPED_IMAGE => return false,
            _ => {}
        }
    }
    // Unknown flag combinations and failed allocation queries keep the
    // VirtualQuery type check, including for a free address.
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: the output buffer has the required layout and size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.Type == MEM_MAPPED
}

/// Shared adaptive `VirtualProtect` coalescer over `[from, to)` blocks.
/// When `coalesce_enabled()` is false, changes protection block by block;
/// otherwise calls `VirtualProtect` on the whole range first and on failure
/// skips uncommitted placeholder runs (when `check_absent` is true) and
/// bisects.
fn virtual_protect_adaptive(
    from: usize,
    to: usize,
    prot: u32,
    check_absent: bool,
    on_ok: &mut impl FnMut(usize, usize),
    on_absent: &mut impl FnMut(usize, usize),
    on_err: &mut impl FnMut(usize),
) {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    if from >= to {
        return;
    }
    let mut old = 0u32;
    if !coalesce_enabled() {
        for block in from..to {
            // SAFETY: single block protection change.
            if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, prot, &mut old) } != 0 {
                on_ok(block, block + 1);
            } else if check_absent && block < uncommitted_end(block) {
                on_absent(block, block + 1);
            } else {
                on_err(block);
            }
        }
        return;
    }
    // SAFETY: title memory block range; only protection changes.
    if unsafe {
        VirtualProtect(
            (from * BLOCK) as *const _,
            (to - from) * BLOCK,
            prot,
            &mut old,
        )
    } != 0
    {
        on_ok(from, to);
        return;
    }
    if to - from == 1 {
        if check_absent && from < uncommitted_end(from) {
            on_absent(from, from + 1);
        } else {
            on_err(from);
        }
        return;
    }
    if check_absent {
        let u_end = uncommitted_end(from);
        if from < u_end {
            let skip_to = u_end.min(to);
            on_absent(from, skip_to);
            virtual_protect_adaptive(skip_to, to, prot, true, on_ok, on_absent, on_err);
            return;
        }
    }
    let mid = from + (to - from) / 2;
    virtual_protect_adaptive(from, mid, prot, check_absent, on_ok, on_absent, on_err);
    virtual_protect_adaptive(mid, to, prot, check_absent, on_ok, on_absent, on_err);
}

/// Applies `VirtualProtect(..., prot)` over sorted block numbers, coalescing
/// contiguous runs into single syscalls (adaptive bisect if a run spans views).
fn protect_block_runs(blocks: &[usize], prot: u32) {
    for_each_contiguous_run(blocks, |from, to| {
        virtual_protect_adaptive(
            from,
            to,
            prot,
            false,
            &mut |_, _| {},
            &mut |_, _| {},
            &mut |_| {},
        );
    });
}

/// Guards the other mappings of blocks in `entries` while their guard shards
/// are held via `access`, returning:
/// - `(block, shown_blocks)` entries whose other mappings are all guarded;
/// - the set of blocks that still have an unguardable mapping (pinned,
///   lazy-armed, resting, or `VirtualProtect` failed) and must stay changed.
fn guard_aliases_with(
    access: &impl ShardAccess,
    mut entries: Vec<(usize, Vec<usize>)>,
    lazy: bool,
    rest: std::time::Duration,
) -> (Vec<(usize, Vec<usize>)>, std::collections::BTreeSet<usize>) {
    use std::collections::BTreeSet;
    if entries.is_empty() {
        return (Vec::new(), BTreeSet::new());
    }
    let mut unguarded = BTreeSet::new();
    let mut alias_targets = Vec::new();
    let mut pending_pairs: Vec<(usize, usize)> = Vec::new();
    for (block, aliases) in &entries {
        let mut blocked = false;
        for &alias in aliases {
            if alias == *block {
                continue;
            }
            if access.pinned(alias)
                || DIR.is_lazy_armed(alias)
                || (lazy && crate::lazy_back::has_block(alias))
                || (!DIR.is_guard_protected(alias) && access.rests(alias, rest))
            {
                blocked = true;
                break;
            }
        }
        if blocked {
            unguarded.insert(*block);
            continue;
        }
        for &alias in aliases {
            if alias != *block && !DIR.is_guard_protected(alias) && !DIR.is_cold_ro(alias) {
                access.debug_assert_locked(alias);
                alias_targets.push(alias);
                pending_pairs.push((*block, alias));
            }
        }
    }
    protect_alias_blocks(alias_targets);
    for (block, alias) in pending_pairs {
        if !DIR.is_guard_protected(alias) && !DIR.is_cold_ro(alias) {
            unguarded.insert(block);
        }
    }
    entries.retain(|(block, _)| !unguarded.contains(block));
    (entries, unguarded)
}

/// Computes the shard mask required by every alias in `entries`.
fn alias_entries_mask(entries: &[(usize, Vec<usize>)]) -> ShardMask {
    let mut mask = ShardMask::EMPTY;
    for (_, aliases) in entries {
        for &a in aliases {
            mask.insert_block(a);
        }
    }
    mask
}

/// Attempts [`mark_protected_clean_with_access`] while holding only `shard`.
/// Returns `Err(alias_mask)` without mutating anything if any alias lies in
/// another shard.
fn try_mark_protected_clean_single_shard(
    g: &GuardShard,
    shard: usize,
    from: usize,
    to: usize,
    lazy: bool,
    rest: std::time::Duration,
) -> Result<(), ShardMask> {
    let entries = collect_aliases(from, to);
    let alias_mask = alias_entries_mask(&entries);
    if !alias_mask.is_subset_of(ShardMask::single(shard)) {
        return Err(alias_mask);
    }
    let access = SingleShard { shard, guard: g };
    mark_protected_clean_with_access(&access, entries, from, to, lazy, rest);
    Ok(())
}

/// Attempts [`protect_with_access`] while holding only `shard`.
/// Returns `Err(alias_mask)` without mutating anything if any alias lies in
/// another shard.
fn try_protect_single_shard(
    g: &GuardShard,
    shard: usize,
    from: usize,
    to: usize,
    lazy: bool,
    rest: std::time::Duration,
) -> Result<(), ShardMask> {
    let entries = collect_aliases(from, to);
    let alias_mask = alias_entries_mask(&entries);
    if !alias_mask.is_subset_of(ShardMask::single(shard)) {
        return Err(alias_mask);
    }
    let access = SingleShard { shard, guard: g };
    protect_with_access(&access, entries, from, to, lazy, rest);
    Ok(())
}

/// Marks already-`GuardProtected` blocks `[from, to)` clean after ensuring
/// every alias view of each block is guarded; any block with an unguardable
/// alias stays unclean (`GuardClean = false`).
fn mark_protected_clean_with_access(
    access: &impl ShardAccess,
    entries: Vec<(usize, Vec<usize>)>,
    from: usize,
    to: usize,
    lazy: bool,
    rest: std::time::Duration,
) {
    debug_assert_range_locked(access, from, to);
    let (aliased, unguarded) = guard_aliases_with(access, entries, lazy, rest);
    for (block, aliases) in aliased {
        if !DIR.is_guard_protected(block) {
            continue;
        }
        for alias in aliases {
            DIR.set_bit(alias, BitField::GuardAliased, true, Ordering::SeqCst);
        }
    }
    if unguarded.is_empty() {
        DIR.set_bits_range(from, to, BitField::GuardClean, true, Ordering::SeqCst);
        return;
    }
    let mut cur = from;
    for bad in unguarded {
        if cur < bad {
            DIR.set_bits_range(cur, bad, BitField::GuardClean, true, Ordering::SeqCst);
        }
        cur = bad + 1;
    }
    if cur < to {
        DIR.set_bits_range(cur, to, BitField::GuardClean, true, Ordering::SeqCst);
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean,
/// also guarding all alias views of the same physical pages. Any block
/// whose alias cannot be guarded is left out (stays changed).
fn protect_with_access(
    access: &impl ShardAccess,
    entries: Vec<(usize, Vec<usize>)>,
    from: usize,
    to: usize,
    lazy: bool,
    rest: std::time::Duration,
) {
    debug_assert_range_locked(access, from, to);
    #[cfg(feature = "diagnostics")]
    crate::faultbook::note(crate::faultbook::What::Taken, from, to, 0);
    let (aliased, unguarded) = guard_aliases_with(access, entries, lazy, rest);
    if unguarded.is_empty() {
        protect_here(access, from, to);
    } else {
        let mut cur = from;
        for bad in unguarded {
            if cur < bad {
                protect_here(access, cur, bad);
            }
            cur = bad + 1;
        }
        if cur < to {
            protect_here(access, cur, to);
        }
    }
    for (block, aliases) in aliased {
        if !DIR.is_guard_protected(block) {
            continue;
        }
        #[cfg(feature = "diagnostics")]
        ALIASES[0].fetch_add(1, Ordering::Relaxed);
        for alias in aliases {
            DIR.set_bit(alias, BitField::GuardAliased, true, Ordering::SeqCst);
        }
    }
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Makes alias blocks read-only via coalesced `VirtualProtect` runs without
/// marking them clean (`guard` must still report them written on first check).
fn protect_alias_blocks(mut aliases: Vec<usize>) {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;
    if aliases.is_empty() {
        return;
    }
    aliases.sort_unstable();
    aliases.dedup();
    let vk = view_known();
    for_each_contiguous_run(&aliases, |from, to| {
        virtual_protect_adaptive(
            from,
            to,
            PAGE_READONLY,
            false,
            &mut |ok_from, ok_to| {
                DIR.set_bits_range(
                    ok_from,
                    ok_to,
                    BitField::GuardProtected,
                    true,
                    Ordering::SeqCst,
                );
                if vk {
                    DIR.set_bits_range(
                        ok_from,
                        ok_to,
                        BitField::GuardViewed,
                        true,
                        Ordering::SeqCst,
                    );
                }
                #[cfg(feature = "diagnostics")]
                for block in ok_from..ok_to {
                    crate::faultbook::note(crate::faultbook::What::Alias, block, 0, 0);
                }
            },
            &mut |_, _| {},
            &mut |_| {},
        );
    });
}

/// [`protect_with_access`] for one range, without the other views of its pages.
fn protect_here(access: &impl ShardAccess, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;
    debug_assert_range_locked(access, from, to);
    let vk = view_known();
    virtual_protect_adaptive(
        from,
        to,
        PAGE_READONLY,
        true,
        &mut |ok_from, ok_to| {
            DIR.set_bits_range(
                ok_from,
                ok_to,
                BitField::GuardProtected,
                true,
                Ordering::SeqCst,
            );
            DIR.set_bits_range(ok_from, ok_to, BitField::GuardClean, true, Ordering::SeqCst);
            if vk {
                DIR.set_bits_range(
                    ok_from,
                    ok_to,
                    BitField::GuardViewed,
                    true,
                    Ordering::SeqCst,
                );
            }
        },
        &mut |abs_from, abs_to| {
            DIR.set_bits_range(
                abs_from,
                abs_to,
                BitField::GuardAbsent,
                true,
                Ordering::SeqCst,
            );
        },
        &mut |bad_block| {
            #[cfg(feature = "diagnostics")]
            {
                let error = std::io::Error::last_os_error();
                static N: AtomicU64 = AtomicU64::new(0);
                if N.fetch_add(1, Ordering::Relaxed).is_multiple_of(1024) {
                    tracing::info!(
                        "a block of title memory could not be made read-only (one in 1024): {:#x}: {error}",
                        bad_block * BLOCK
                    );
                }
            }
            #[cfg(not(feature = "diagnostics"))]
            let _ = bad_block;
        },
    );
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and marks
/// all their alias views unclean; returns how many primary blocks were guarded.
/// Under [`open_one_view`] (`VICTORY_OPEN_ONE_VIEW`, on by default), alias views
/// outside `[start, end)` stay read-only (`GuardProtected`) and are only marked
/// unclean (`GuardClean = false`) so later writes through them still fault
/// while refreshes immediately observe the change.
/// Marks every primary and alias block dirty/unprotected in `PageDir` strictly
/// BEFORE any `VirtualProtect(..., PAGE_READWRITE)` opens a page.
fn open_under_locked(
    access: &impl ShardAccess,
    start: usize,
    end: usize,
    mut dirtied: impl FnMut(usize),
) -> usize {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    if start >= end {
        return 0;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    debug_assert_range_locked(access, first, last);
    let primary = DIR.collect_set_bits(first, last, BitField::GuardProtected);
    let aliased = DIR.collect_set_bits(first, last, BitField::GuardAliased);
    let one = open_one_view();
    let mut alias_blocks = Vec::new();
    if !aliased.is_empty() {
        for block in aliased {
            DIR.set_bit(block, BitField::GuardAliased, false, Ordering::SeqCst);
            for alias in crate::inplace::pages_of(block * BLOCK) {
                let ab = alias / BLOCK;
                if one {
                    if (ab < first || ab >= last)
                        && DIR.clear_bit_was_set(ab, BitField::GuardClean, Ordering::SeqCst)
                    {
                        dirtied(ab * BLOCK);
                    }
                    continue;
                }
                access.debug_assert_locked(ab);
                DIR.set_bit(ab, BitField::GuardAliased, false, Ordering::SeqCst);
                if ab < first || ab >= last {
                    alias_blocks.push(ab);
                }
            }
        }
        if !alias_blocks.is_empty() {
            alias_blocks.sort_unstable();
            alias_blocks.dedup();
            alias_blocks.retain(|&ab| DIR.is_guard_protected(ab));
        }
    }
    // Every primary and alias block is marked dirty/unprotected BEFORE any
    // page is opened: once a page is writable, a write by another thread lands
    // without a fault, and a concurrent lock-free `PageDir` reader on another
    // view must not still observe `GuardClean == true`.
    for &block in &primary {
        DIR.set_bit(block, BitField::GuardProtected, false, Ordering::SeqCst);
        if DIR.clear_bit_was_set(block, BitField::GuardClean, Ordering::SeqCst) {
            dirtied(block * BLOCK);
        }
    }
    for &block in &alias_blocks {
        DIR.set_bit(block, BitField::GuardProtected, false, Ordering::SeqCst);
        if DIR.clear_bit_was_set(block, BitField::GuardClean, Ordering::SeqCst) {
            dirtied(block * BLOCK);
        }
    }
    #[cfg(feature = "diagnostics")]
    if !alias_blocks.is_empty() {
        ALIASES[1].fetch_add(alias_blocks.len() as u64, Ordering::Relaxed);
    }
    if !primary.is_empty() {
        protect_block_runs(&primary, PAGE_READWRITE);
    }
    if !alias_blocks.is_empty() {
        protect_block_runs(&alias_blocks, PAGE_READWRITE);
    }
    primary.len()
}

/// Allocation-free single-block fast path for the VEH write fault handler when
/// the faulted block has no guarded alias views. Clears `PageDir` guard bits
/// strictly before `VirtualProtect(..., PAGE_READWRITE)`.
fn open_single_block(access: &impl ShardAccess, block: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    access.debug_assert_locked(block);
    if !DIR.is_guard_protected(block) {
        return false;
    }
    DIR.set_bit(block, BitField::GuardProtected, false, Ordering::SeqCst);
    DIR.set_bit(block, BitField::GuardClean, false, Ordering::SeqCst);
    let mut old = 0u32;
    // SAFETY: a block we made read-only.
    unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    true
}

/// Before our own synchronous user-mode write into title memory
/// `[at, at + len)` (GPU copy-back, `Step::Copy`).
pub(crate) fn writing(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let (first, last) = (at / BLOCK, end.div_ceil(BLOCK));
    let needs_open = if lockfree_enabled() {
        DIR.any_bits_in_range(
            first,
            last,
            &[
                BitField::GuardProtected,
                BitField::GuardAliased,
                BitField::ColdRo,
            ],
        )
    } else {
        let _g = lock_shards_read_range(first, last);
        DIR.any_bits_in_range(
            first,
            last,
            &[
                BitField::GuardProtected,
                BitField::GuardAliased,
                BitField::ColdRo,
            ],
        )
    };
    if needs_open {
        let locked = lock_range_for_open(first, last);
        let opened = open_under_locked(&locked, at, end, |_| {});
        demote_cold_range(&locked, first, last);
        drop(locked);
        #[cfg(feature = "diagnostics")]
        crate::faultbook::note(crate::faultbook::What::Ours, first, last, opened as u64);
        let _ = opened;
    }
}

/// Before a write into title memory `[at, at + len)` through the kernel (which
/// does not fault into our handler): opens its guarded/cold blocks and leaves them
/// writable for [`PIN`] across every view of their pages.
#[allow(dead_code)]
pub fn kernel_writing(at: usize, len: usize) {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let (first, last) = (at / BLOCK, end.div_ceil(BLOCK));
    // Pin + clear guard/cold metadata across the range and all aliases under
    // their guard shards (ascending order) strictly BEFORE any
    // `VirtualProtect(..., PAGE_READWRITE)` and BEFORE
    // `lazy_back::materialize_range`, so neither a concurrent
    // `try_arm_lazy_under_guards` nor a lock-free `PageDir` reader can race.
    {
        let (mut locked, alias_blocks) = lock_range_and_all_aliases(first, last);
        locked.prune_pins();
        let until = std::time::Instant::now() + PIN;
        locked.pin_range(first, last, until);
        for_each_contiguous_run(&alias_blocks, |r_from, r_to| {
            locked.pin_range(r_from, r_to, until);
        });

        let frame32 = current_frame32();
        let mut to_restore = DIR.collect_set_bits(first, last, BitField::GuardProtected);
        let cold_primary = DIR.collect_set_bits(first, last, BitField::ColdRo);
        if !cold_primary.is_empty() {
            for &b in &cold_primary {
                DIR.reset_cold(b, frame32);
            }
            to_restore.extend(cold_primary);
        }
        DIR.set_bits_range(
            first,
            last,
            BitField::GuardProtected,
            false,
            Ordering::SeqCst,
        );
        DIR.set_bits_range(first, last, BitField::GuardClean, false, Ordering::SeqCst);
        DIR.set_bits_range(first, last, BitField::GuardAliased, false, Ordering::SeqCst);
        DIR.set_bits_range(first, last, BitField::ColdRo, false, Ordering::SeqCst);

        for &ab in &alias_blocks {
            let was_prot = DIR.clear_bit_was_set(ab, BitField::GuardProtected, Ordering::SeqCst);
            DIR.set_bit(ab, BitField::GuardClean, false, Ordering::SeqCst);
            DIR.set_bit(ab, BitField::GuardAliased, false, Ordering::SeqCst);
            let was_cold = DIR.clear_bit_was_set(ab, BitField::ColdRo, Ordering::SeqCst);
            if was_cold {
                DIR.reset_cold(ab, frame32);
            }
            if was_prot || was_cold {
                to_restore.push(ab);
            }
        }
        if !to_restore.is_empty() {
            to_restore.sort_unstable();
            to_restore.dedup();
            protect_block_runs(&to_restore, PAGE_READWRITE);
        }
    }
    crate::lazy_back::materialize_range(at, len);
    let range = (first, last);
    let mut ahead = AHEAD.write().unwrap_or_else(|e| e.into_inner());
    ahead.forget_ahead(range);
}

/// Before the title's kernel read into its memory `[at, at + len)`: opens the
/// guarded and cold-promoted blocks there and keeps them open until
/// [`kernel_wrote`]. Stored as a ref-counted range per touched shard so
/// saturated lengths (`usize::MAX`) cost O(touched shards <= 128) and
/// overlapping reads stay open until every reader lands.
pub fn kernel_will_write(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let range = (at / BLOCK, end.div_ceil(BLOCK));
    // Lock order: `GUARD_SHARDS_TABLE` (ascending) -> `AHEAD`.
    {
        let mut locked = lock_range_for_open(range.0, range.1);
        open_under_locked(&locked, at, end, |_| {});
        demote_cold_range(&locked, range.0, range.1);
        locked.announce_range(range);
        ANNOUNCED_SEQ.fetch_add(1, Ordering::SeqCst);
        let mut ahead = AHEAD.write().unwrap_or_else(|e| e.into_inner());
        ahead.forget_ahead(range);
    }
    crate::lazy_back::materialize_range(at, len);
}

/// The kernel write [`kernel_will_write`] announced for `[at, at + len)` has
/// landed (or failed); decrements its open announcement count so its blocks
/// can be guarded or cold-promoted again once no announcement remains.
pub fn kernel_wrote(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let range = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    let mut locked = GuardLockSet::lock_range(range.0, range.1);
    locked.unannounce_range(range);
}

/// A storage worker finished writing these mapped guest pages through another
/// view of the same section. Their address protections are still in effect.
pub fn mapped_pages_written(addresses: impl Iterator<Item = usize>) {
    let blocks: Vec<usize> = addresses.map(|at| at / PAGE).collect();
    if blocks.is_empty() {
        return;
    }
    let _locked = GuardLockSet::lock_blocks(&blocks);
    for block in blocks {
        DIR.set_bit(block, BitField::GuardClean, false, Ordering::SeqCst);
    }
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]).
pub(crate) fn page_writes() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("VICTORY_PAGE_WRITES")
            .as_deref()
            .map(str::trim)
        {
            Ok("0") => 0,
            Ok("1") => 3,
            Ok("slots") => 2,
            Ok("all") => 7,
            _ => 1,
        }
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped.
pub(crate) static WRITTEN: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    let (first, last) = (at / BLOCK, end.div_ceil(BLOCK));
    {
        let locked = if mode & 1 != 0 {
            lock_range_for_open(first, last)
        } else {
            GuardLockSet::lock_range(first, last)
        };
        let guarded = DIR.count_bits_in_range(first, last, BitField::GuardProtected);
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 {
            if guarded != 0 {
                open_under_locked(&locked, at, end, |_| {});
            }
            if DIR.any_bits_in_range(first, last, &[BitField::ColdRo]) {
                demote_cold_range(&locked, first, last);
            }
        }
    }
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    if crate::lazy_back::active() {
        crate::lazy_back::materialize_ranges(&ranges);
    }
    let mut mask = ShardMask::EMPTY;
    for &(at, n) in &ranges {
        if n == 0 {
            continue;
        }
        let end = at.saturating_add(n);
        mask.insert_range(at / BLOCK, end.div_ceil(BLOCK));
    }
    if mask.is_empty() {
        return;
    }
    let _locked = GuardLockSet::lock_mask(mask);
    let mut to_open = Vec::new();
    #[cfg(feature = "diagnostics")]
    let mut opened_ranges = 0u64;
    for &(at, n) in &ranges {
        if n == 0 {
            continue;
        }
        let end = at.saturating_add(n);
        let (first, last) = (at / BLOCK, end.div_ceil(BLOCK));
        let blocks = DIR.collect_set_bits(first, last, BitField::GuardProtected);
        if !blocks.is_empty() {
            #[cfg(feature = "diagnostics")]
            {
                opened_ranges += 1;
            }
            to_open.extend(blocks);
        }
    }
    if to_open.is_empty() {
        return;
    }
    to_open.sort_unstable();
    to_open.dedup();
    for &block in &to_open {
        DIR.set_bit(block, BitField::GuardProtected, false, Ordering::SeqCst);
        DIR.set_bit(block, BitField::GuardClean, false, Ordering::SeqCst);
        DIR.set_bit(block, BitField::GuardAliased, false, Ordering::SeqCst);
    }
    #[cfg(feature = "diagnostics")]
    ALIASES[2].fetch_add(opened_ranges, Ordering::Relaxed);
    protect_block_runs(&to_open, PAGE_READWRITE);
}

/// Coordinates in-place CPU reads of title memory with mapping changes.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
///
/// Only used when `VICTORY_MAPPING_RANGES=0` ([`by_range`]); otherwise reads
/// and changes hold only the bytes they touch ([`SPANS`]).
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Whether a read in place and a change of mapping keep apart only when
/// their bytes meet ([`SPANS`]), not whatever memory each touches
/// ([`MAPPING`]); `VICTORY_MAPPING_RANGES=0` holds all of title memory for
/// each. On by default.
pub(crate) fn by_range() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MAPPING_RANGES").is_none_or(|v| v != "0"))
}

/// Reads of title memory in place under way, and changes of its mapping,
/// by the bytes they touch ([`by_range`]).
struct Spans {
    /// To tell them apart.
    next: u64,
    /// Reads under way: number, start and end.
    reads: Vec<(u64, usize, usize)>,
    /// Changes: number, start, end, and whether under way (else waiting for
    /// the reads over its bytes to end; no read begins over them meanwhile,
    /// so that one read after another cannot keep a change waiting).
    changes: Vec<(u64, usize, usize, bool)>,
    /// Threads waiting on [`SPANS_LEFT`].
    waiting: usize,
}

static SPANS: Mutex<Spans> = Mutex::new(Spans {
    next: 0,
    reads: Vec::new(),
    changes: Vec::new(),
    waiting: 0,
});

/// Told when a read or a change ends while a thread waits.
static SPANS_LEFT: std::sync::Condvar = std::sync::Condvar::new();

fn spans() -> std::sync::MutexGuard<'static, Spans> {
    SPANS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Waits until a read or a change ends.
fn spans_wait(
    mut s: std::sync::MutexGuard<'static, Spans>,
) -> std::sync::MutexGuard<'static, Spans> {
    s.waiting += 1;
    let mut s = SPANS_LEFT.wait(s).unwrap_or_else(|e| e.into_inner());
    s.waiting -= 1;
    s
}

/// Whether `[a, b)` and `[c, d)` share a byte.
fn meet(a: usize, b: usize, c: usize, d: usize) -> bool {
    a < b && c < d && a < d && c < b
}

impl Spans {
    fn read(&mut self, at: usize, end: usize) -> Span {
        self.next += 1;
        self.reads.push((self.next, at, end));
        Span(self.next, false)
    }
}

/// A read's or a change's place in [`SPANS`] (its number, and whether a
/// change), given up when dropped.
struct Span(u64, bool);

impl Drop for Span {
    fn drop(&mut self) {
        let mut s = spans();
        if self.1 {
            s.changes.retain(|c| c.0 != self.0);
        } else if let Some(i) = s.reads.iter().position(|r| r.0 == self.0) {
            s.reads.swap_remove(i);
        }
        if s.waiting > 0 {
            SPANS_LEFT.notify_all();
        }
    }
}

/// What a read or a change of mapping holds: its bytes ([`by_range`]) or
/// the lock over all of title memory.
enum Held<G> {
    All(#[allow(dead_code)] G),
    Bytes(#[allow(dead_code)] Span),
}

/// Since the last frame-rate line: changes of title memory's mapping, the
/// microseconds they waited for reads in place, and those they were under
/// way.
pub(crate) static REMAPS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping {
    _held: Held<std::sync::RwLockWriteGuard<'static, ()>>,
    since: std::time::Instant,
}

impl Drop for Remapping {
    fn drop(&mut self) {
        REMAPS[2].fetch_add(
            self.since.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// The runtime is about to map, unmap or move physical pages of title
/// memory `[at, at + len)`: until what it returns is dropped, no read of
/// those bytes in place is under way and none begins
/// ([`title_memory_reading`]). Taken around the whole change, the unmapping
/// and the mapping again, over all the bytes whose mapping it changes.
pub fn title_memory_remapping(at: usize, len: usize) -> Remapping {
    use std::sync::atomic::Ordering::Relaxed;
    crate::lazy_back::materialize_range(at, len);
    let started = std::time::Instant::now();
    let held = if by_range() {
        let end = at.saturating_add(len);
        let mut s = spans();
        s.next += 1;
        let id = s.next;
        s.changes.push((id, at, end, false));
        while s.reads.iter().any(|r| meet(r.1, r.2, at, end))
            || s.changes.iter().any(|c| c.3 && meet(c.1, c.2, at, end))
        {
            s = spans_wait(s);
        }
        if let Some(c) = s.changes.iter_mut().find(|c| c.0 == id) {
            c.3 = true;
        }
        Held::Bytes(Span(id, true))
    } else {
        Held::All(MAPPING.write().unwrap_or_else(|e| e.into_inner()))
    };
    REMAPS[0].fetch_add(1, Relaxed);
    REMAPS[1].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    Remapping {
        _held: held,
        since: std::time::Instant::now(),
    }
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] Held<std::sync::RwLockReadGuard<'static, ()>>);

/// For reading title memory `[at, at + len)` in place: until what it
/// returns is dropped, the runtime changes no mapping of those bytes, so
/// memory found mapped stays mapped. Not to be taken twice by one thread,
/// nor held across anything that maps title memory.
pub fn title_memory_reading(at: usize, len: usize) -> Reading {
    if !by_range() {
        return Reading(Held::All(MAPPING.read().unwrap_or_else(|e| e.into_inner())));
    }
    let end = at.saturating_add(len);
    let mut s = spans();
    while s.changes.iter().any(|c| meet(c.1, c.2, at, end)) {
        s = spans_wait(s);
    }
    Reading(Held::Bytes(s.read(at, end)))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping of those bytes (or waits to), for a caller
/// that has another way to read and would rather not wait.
pub fn title_memory_reading_now(at: usize, len: usize) -> Option<Reading> {
    if !by_range() {
        return match MAPPING.try_read() {
            Ok(guard) => Some(Reading(Held::All(guard))),
            Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(Held::All(e.into_inner()))),
            Err(std::sync::TryLockError::WouldBlock) => None,
        };
    }
    let end = at.saturating_add(len);
    let mut s = spans();
    if s.changes.iter().any(|c| meet(c.1, c.2, at, end)) {
        return None;
    }
    Some(Reading(Held::Bytes(s.read(at, end))))
}

/// Returns when the change of a mapping overlapping `[at, at + len)` that is
/// under way, if one is, is over (or has been going on for 2 ms): pages it
/// took away for its moment are back. Another may have begun.
#[allow(dead_code)]
pub(crate) fn change_over(at: usize, len: usize) {
    if !by_range() {
        if !matches!(MAPPING.try_read(), Err(std::sync::TryLockError::WouldBlock)) {
            return;
        }
        let started = std::time::Instant::now();
        while matches!(MAPPING.try_read(), Err(std::sync::TryLockError::WouldBlock))
            && started.elapsed().as_millis() < 2
        {
            for _ in 0..50 {
                std::hint::spin_loop();
            }
        }
        return;
    }
    let end = at.saturating_add(len);
    let mut s = spans();
    if !s.changes.iter().any(|c| meet(c.1, c.2, at, end)) {
        return;
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(2);
    while s.changes.iter().any(|c| meet(c.1, c.2, at, end)) {
        let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) else {
            break;
        };
        s.waiting += 1;
        let (next, timeout) = SPANS_LEFT
            .wait_timeout(s, left)
            .unwrap_or_else(|e| e.into_inner());
        s = next;
        s.waiting -= 1;
        if timeout.timed_out() {
            break;
        }
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    crate::lazy_back::discard_range(at, len);
    crate::texback::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let range = at / BLOCK..end.div_ceil(BLOCK);
    if range.start >= range.end {
        return;
    }
    let mut locked = GuardLockSet::lock_range(range.start, range.end);
    locked.unpin_range(range.start, range.end);
    locked.clear_announced_inside(range.start, range.end);
    locked.forget_opened_range(range.start, range.end);
    DIR.set_bits_range(
        range.start,
        range.end,
        BitField::GuardAbsent,
        false,
        Ordering::SeqCst,
    );
    DIR.set_bits_range(
        range.start,
        range.end,
        BitField::GuardAliased,
        false,
        Ordering::SeqCst,
    );
    DIR.set_bits_range(
        range.start,
        range.end,
        BitField::Watched,
        false,
        Ordering::Relaxed,
    );

    let mut to_restore = DIR.collect_set_bits(range.start, range.end, BitField::GuardProtected);
    let cold_ro = DIR.collect_set_bits(range.start, range.end, BitField::ColdRo);
    if !cold_ro.is_empty() {
        to_restore.extend(cold_ro);
        to_restore.sort_unstable();
        to_restore.dedup();
    }
    DIR.set_bits_range(
        range.start,
        range.end,
        BitField::GuardProtected,
        false,
        Ordering::SeqCst,
    );
    DIR.set_bits_range(
        range.start,
        range.end,
        BitField::GuardClean,
        false,
        Ordering::SeqCst,
    );
    DIR.set_bits_range(
        range.start,
        range.end,
        BitField::ColdRo,
        false,
        Ordering::SeqCst,
    );
    DIR.set_bits_range(
        range.start,
        range.end,
        BitField::LazyArmed,
        false,
        Ordering::SeqCst,
    );
    DIR.reset_cold_range(range.start, range.end, current_frame32());

    if !to_restore.is_empty() {
        // Restore protection in coalesced contiguous runs; adaptive bisect
        // ensures no block across multiple views is left read-only.
        protect_block_runs(&to_restore, PAGE_READWRITE);
    }

    if view_known() {
        DIR.set_bits_range(
            range.start,
            range.end,
            BitField::GuardViewed,
            true,
            Ordering::SeqCst,
        );
    } else {
        DIR.set_bits_range(
            range.start,
            range.end,
            BitField::GuardViewed,
            false,
            Ordering::SeqCst,
        );
    }
    // Invalidate any `ask_once` ahead coverage overlapping the remapped range
    // and bump `ANNOUNCED_SEQ` so a concurrent `ask_once` batch discards stale
    // pre-remap coverage. Lock order: `GUARD_SHARDS_TABLE` -> `AHEAD` (leaf).
    ANNOUNCED_SEQ.fetch_add(1, Ordering::SeqCst);
    let mut ahead = AHEAD.write().unwrap_or_else(|e| e.into_inner());
    ahead.forget_ahead((range.start, range.end));
}

/// The runtime left `[at, at + len)` mapped to the exact same physical pages
/// during a page-mapping batch (`CopyPageMappingsBatchX`), skipping redundant
/// unmap/remap syscalls. Placed textures and guarded buffer copies over the
/// range still treat its contents as potentially updated via another alias,
/// and any newly mapped alias of a guarded block becomes guarded too.
pub fn title_memory_alias_rebound(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    crate::lazy_back::materialize_range(at, len);
    wrote(at, len);
    let lazy = crate::lazy_back::active();
    let rest = rest();
    let end = at.saturating_add(len);
    let range = at / BLOCK..end.div_ceil(BLOCK);
    if !view_guard() {
        let _locked = GuardLockSet::lock_range(range.start, range.end);
        DIR.set_bits_range(
            range.start,
            range.end,
            BitField::GuardClean,
            false,
            Ordering::SeqCst,
        );
        return;
    }
    let mut base_mask = ShardMask::EMPTY;
    base_mask.insert_range(range.start, range.end);
    let (mut locked, entries) = lock_with_aliases(base_mask, range.start, range.end);
    locked.prune_pins();
    DIR.set_bits_range(
        range.start,
        range.end,
        BitField::GuardClean,
        false,
        Ordering::SeqCst,
    );
    if entries.is_empty() {
        return;
    }
    let mut alias_targets = Vec::new();
    let mut protected_pairs: Vec<(usize, usize)> = Vec::new();
    for (block, aliases) in &entries {
        let block_protected = DIR.is_guard_protected(*block);
        let any_alias_protected = aliases
            .iter()
            .any(|&a| a != *block && DIR.is_guard_protected(a));
        if block_protected {
            for &alias in aliases {
                if alias == *block {
                    continue;
                }
                if !DIR.is_guard_protected(alias) && !DIR.is_cold_ro(alias) {
                    if !locked.pinned(alias)
                        && !locked.rests(alias, rest)
                        && !DIR.is_lazy_armed(alias)
                        && !(lazy && crate::lazy_back::has_block(alias))
                    {
                        alias_targets.push(alias);
                        protected_pairs.push((*block, alias));
                    }
                } else {
                    DIR.set_bit(*block, BitField::GuardAliased, true, Ordering::SeqCst);
                    DIR.set_bit(alias, BitField::GuardAliased, true, Ordering::SeqCst);
                }
            }
        } else if any_alias_protected {
            // Another view of this rebound page is guarded: mark it changed
            // (its bytes may already reflect a write through `block`) and guard
            // `block` if possible; if `block` cannot be guarded, the guarded
            // alias stays changed on subsequent checks too.
            for &alias in aliases {
                if alias != *block {
                    DIR.set_bit(alias, BitField::GuardClean, false, Ordering::SeqCst);
                }
            }
            if !locked.pinned(*block)
                && !locked.rests(*block, rest)
                && !DIR.is_cold_ro(*block)
                && !DIR.is_lazy_armed(*block)
                && !(lazy && crate::lazy_back::has_block(*block))
            {
                alias_targets.push(*block);
                for &alias in aliases {
                    if alias != *block && DIR.is_guard_protected(alias) {
                        protected_pairs.push((alias, *block));
                    }
                }
            }
        }
    }
    protect_alias_blocks(alias_targets);
    for (guarded, target) in protected_pairs {
        if DIR.is_guard_protected(target) || DIR.is_cold_ro(target) {
            DIR.set_bit(guarded, BitField::GuardAliased, true, Ordering::SeqCst);
            DIR.set_bit(target, BitField::GuardAliased, true, Ordering::SeqCst);
        } else {
            DIR.set_bit(guarded, BitField::GuardClean, false, Ordering::SeqCst);
        }
    }
}

/// Restores `PAGE_READWRITE` for a cold-promoted write-watched 64 KiB block
/// upon CPU write fault while its guard shard is held, recording the page dirty
/// strictly before opening write access.
fn open_cold_ro(access: &impl ShardAccess, at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let block = at / BLOCK;
    access.debug_assert_locked(block);
    if !DIR.is_cold_ro(block) {
        return false;
    }
    let _now = DIR.publish_written(&[block]);
    DIR.set_bit(block, BitField::ColdRo, false, Ordering::SeqCst);
    DIR.reset_cold(block, current_frame32());
    let mut old = 0u32;
    // SAFETY: `block` was cold-promoted to `PAGE_READONLY` by us; restoring `PAGE_READWRITE`.
    unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) != 0 }
}

/// Cold VEH path when `VICTORY_OPEN_ONE_VIEW=0` and the faulted block has
/// cross-shard aliases that must all be opened together under ascending locks.
#[cold]
#[inline(never)]
fn handle_cross_shard_write_fault(
    at: usize,
    block: usize,
    shard: usize,
    #[cfg(feature = "diagnostics")] began: std::time::Instant,
) -> Option<bool> {
    let mut locked = lock_block_with_aliases(block);
    #[cfg(feature = "diagnostics")]
    let waited = began.elapsed();
    let opened = open_under_locked(&locked, at, at + 1, |_| {});
    if opened != 0 {
        locked.shard_mut(shard).opened(block);
        #[cfg(feature = "diagnostics")]
        {
            let ns = waited.as_nanos().min(u128::from(u32::MAX)) as usize;
            let held = (began.elapsed() - waited).as_nanos() as u64;
            crate::faultbook::note(
                crate::faultbook::What::Fault,
                block,
                (opened << 32) | ns,
                held,
            );
        }
        Some(true)
    } else if open_cold_ro(&locked, at) || writable(at) {
        Some(false)
    } else {
        None
    }
}

/// Registers the handler that takes writes to guarded and cold-promoted blocks,
/// once, ahead of the crash reporter's. Returns `true` if handler registration
/// succeeded; callers must not make pages `PAGE_READONLY` if this returns `false`.
fn install() -> bool {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const READ: usize = 0;
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: `ContextRecord` is valid when Windows dispatches an exception.
        let rip = unsafe { (*(*info).ContextRecord).Rip };
        if crate::host::is_fast_copy_rip(rip) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let access = r.ExceptionInformation[0];
        let at = r.ExceptionInformation[1];
        if (access == READ || access == WRITE)
            && crate::lazy_back::handle_fault(at, access == WRITE)
        {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if access != WRITE {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let block = at / BLOCK;
        // Another thread may have opened the block since this one faulted:
        // avoid waiting on the shard lock if `PageDir` and Windows already show it RW.
        if lockfree_enabled()
            && !DIR.is_guard_protected(block)
            && !DIR.is_cold_ro(block)
            && writable(at)
        {
            #[cfg(feature = "diagnostics")]
            crate::faultbook::note(crate::faultbook::What::Open, block, 0, 0);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // Held across both checks when opening returns 0: another thread may
        // have opened the block since this one faulted, and a copy may have
        // guarded it again since.
        #[cfg(feature = "diagnostics")]
        let began = std::time::Instant::now();
        let shard = shard_of(block);
        let handled = {
            let mut g = guard_shard(shard);
            #[cfg(feature = "diagnostics")]
            let waited = began.elapsed();
            let aliased = DIR.get_bit(block, BitField::GuardAliased, Ordering::SeqCst);
            let needs_cross_shard = !open_one_view()
                && aliased
                && crate::inplace::pages_of(block * BLOCK)
                    .iter()
                    .any(|&a| shard_of(a / BLOCK) != shard);
            if needs_cross_shard {
                drop(g);
                handle_cross_shard_write_fault(
                    at,
                    block,
                    shard,
                    #[cfg(feature = "diagnostics")]
                    began,
                )
            } else {
                let single = SingleShard { shard, guard: &g };
                // Allocation-free fast path for the common non-aliased write fault;
                // only fall back to `open_under_locked` when the block has guarded aliases.
                let opened = if !aliased {
                    usize::from(open_single_block(&single, block))
                } else {
                    open_under_locked(&single, at, at + 1, |_| {})
                };
                if opened != 0 {
                    g.opened(block);
                    #[cfg(feature = "diagnostics")]
                    {
                        let ns = waited.as_nanos().min(u128::from(u32::MAX)) as usize;
                        let held = (began.elapsed() - waited).as_nanos() as u64;
                        crate::faultbook::note(
                            crate::faultbook::What::Fault,
                            block,
                            (opened << 32) | ns,
                            held,
                        );
                    }
                    Some(true)
                } else if open_cold_ro(&single, at) || writable(at) {
                    Some(false)
                } else {
                    None
                }
            }
        };
        match handled {
            Some(true) => {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::GuardFaults, 1);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            Some(false) => return EXCEPTION_CONTINUE_EXECUTION,
            None => {}
        }
        // Read-only, in a view we were told of, and not noted guarded: only we
        // make title views read-only, so it is ours to give back.
        if view_guard() && crate::inplace::has_view_at(at) {
            let mut old = 0;
            let block = at - at % BLOCK;
            // SAFETY: only the protection of title view memory changes.
            let opened = unsafe {
                windows_sys::Win32::System::Memory::VirtualProtect(
                    block as *const _,
                    BLOCK,
                    windows_sys::Win32::System::Memory::PAGE_READWRITE,
                    &mut old,
                )
            } != 0;
            if opened {
                tracing::warn!(
                    at = format_args!("{at:#x}"),
                    "a read-only view block that no guard knew was opened"
                );
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    static INSTALLED: OnceLock<bool> = OnceLock::new();
    *INSTALLED.get_or_init(|| {
        // SAFETY: the handler only touches our guard state and protections.
        unsafe { !AddVectoredExceptionHandler(1, Some(on_write)).is_null() }
    })
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    static JOURNAL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII guard that ensures `ask_once_end()` always runs on scope exit.
    struct AskOnceGuard;
    impl Drop for AskOnceGuard {
        fn drop(&mut self) {
            ask_once_end();
        }
    }

    /// RAII guard for a private `MEM_WRITE_WATCH` test allocation that clears
    /// its `PageDir` / guard metadata and releases the allocation on drop.
    struct WatchedAllocGuard {
        base: usize,
        bytes: usize,
    }

    impl WatchedAllocGuard {
        fn new(pages: usize) -> Self {
            let bytes = pages * PAGE;
            // SAFETY: fresh write-watched allocation owned by this guard.
            let base = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    bytes,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            } as usize;
            assert_ne!(base, 0);
            // Isolate from prior tests whose `VirtualAlloc`+`VirtualFree` may have reused this VA.
            DIR.clear_written_range(base / PAGE, base / PAGE + pages);
            Self { base, bytes }
        }
    }

    impl Drop for WatchedAllocGuard {
        fn drop(&mut self) {
            title_memory_remapped(self.base, self.bytes);
            DIR.clear_written_range(self.base / PAGE, (self.base + self.bytes) / PAGE);
            // SAFETY: allocation owned by this guard.
            unsafe { VirtualFree(self.base as *mut _, 0, MEM_RELEASE) };
        }
    }

    fn clear_all_pins() {
        for s in 0..GUARD_SHARDS {
            guard_shard(s).pins.clear();
        }
    }

    fn pin_blocks_for_test(first: usize, last: usize, until: std::time::Instant) {
        let mut locked = GuardLockSet::lock_range(first, last);
        locked.pin_range(first, last, until);
    }

    fn open_for_test(start: usize, end: usize, dirtied: impl FnMut(usize)) -> usize {
        if start >= end {
            return 0;
        }
        let locked = lock_range_for_open(start / BLOCK, end.div_ceil(BLOCK));
        open_under_locked(&locked, start, end, dirtied)
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        let _journal_lock = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        // Isolate from prior tests whose `VirtualAlloc`+`VirtualFree` may have reused this VA.
        DIR.clear_written_range(p / PAGE, p / PAGE + 4);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        DIR.clear_written_range(p / PAGE, p / PAGE + 4);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped yet: an empty placeholder has no mapped view.
        assert!(!has_mapped(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(has_mapped(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!has_mapped(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(
            guard(guarded, guarded + SIZE),
            Some(vec![pages(0), pages(1)])
        );
        assert_eq!(
            guard(guarded, guarded + SIZE),
            Some(Vec::new()),
            "nothing written yet"
        );
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(
            guard(guarded, guarded + SIZE),
            Some(vec![pages(0), pages(1)])
        );
        assert_eq!(
            guard(guarded, guarded + SIZE),
            Some(Vec::new()),
            "nothing written yet"
        );
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(
            guard(guarded, guarded + SIZE),
            Some(vec![pages(0), pages(1)])
        );
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages:
    /// - synchronous user-mode [`writing`] opens the other views and immediately
    ///   re-guards on the next check without leaving a 2 s pin;
    /// - async [`kernel_writing`] keeps all views writable and reported written
    ///   while pinned, then re-guards cleanly once the pin expires.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(
            guard(guarded, guarded + SIZE),
            Some(vec![pages(0), pages(1)])
        );
        assert_eq!(
            guard(guarded, guarded + SIZE),
            Some(Vec::new()),
            "nothing written yet"
        );
        // Synchronous write (GPU copy-back): opens without pinning, so the next
        // check reports it once and re-guards immediately.
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Async kernel write: pinned writable for 2 s across all views.
        kernel_writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert!(writable(guarded + BLOCK));
        assert!(writable(other + BLOCK));
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        // Once the pin expires, the block re-guards cleanly.
        clear_all_pins();
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        kernel_writing(other + BLOCK / 2, BLOCK);
        assert_eq!(
            guard(guarded, guarded + SIZE),
            Some(vec![pages(0), pages(1)])
        );
        clear_all_pins();
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view block left read-only without being tracked in `Guards` (e.g.
    /// raced with a remap) is recovered by the vectored write handler when
    /// `inplace` knows it is a view.
    #[test]
    fn orphaned_readonly_view_block_is_opened_on_write_fault() {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
        let (section, one, two) = two_views(BLOCK);
        // Ensure the vectored handler is installed, then leave `one` read-only
        // without noting it in `Guards`.
        install();
        let mut old = 0;
        // SAFETY: our mapped view from `two_views`.
        let prot = unsafe { VirtualProtect(one as *const _, BLOCK, PAGE_READONLY, &mut old) };
        assert_ne!(prot, 0);
        // SAFETY: handler restores PAGE_READWRITE because `inplace` knows `one`.
        unsafe { ((one + 16) as *mut u8).write_volatile(0x5a) };
        assert!(writable(one));
        // SAFETY: readable byte we just wrote.
        assert_eq!(unsafe { ((one + 16) as *const u8).read_volatile() }, 0x5a);
        drop_views(section, &[one, two], BLOCK);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(
            guard(one, one + SIZE),
            Some(vec![pages(one, 0), pages(one, 1)])
        );
        assert_eq!(
            guard(two, two + SIZE),
            Some(vec![pages(two, 0), pages(two, 1)]),
            "the second view's first check"
        );
        assert_eq!(
            guard(one, one + SIZE),
            Some(Vec::new()),
            "nothing written yet"
        );
        assert_eq!(
            guard(two, two + SIZE),
            Some(Vec::new()),
            "nothing written yet"
        );
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(
            guard(two, two + SIZE),
            Some(vec![pages(two, 1)]),
            "written through this view, checked second"
        );
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(
            guard(one, one + SIZE),
            Some(vec![pages(one, 0)]),
            "written through this view, checked second"
        );
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    #[test]
    fn alias_rebound_marks_guarded_block_unclean_and_guards_new_aliases() {
        let (section, a, b) = two_views(BLOCK);
        // Pin `b` initially: guarding `a` MUST leave `a` unclean while `b` is
        // pinned and writable (cannot be guarded yet).
        let until = std::time::Instant::now() + PIN;
        pin_blocks_for_test(b / BLOCK, b / BLOCK + 1, until);
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert_eq!(
            guard(a, a + BLOCK),
            Some(vec![a / PAGE]),
            "a page with a pinned alias stays changed"
        );
        assert!(writable(b));

        // Clear the pin, write through `b`, then guard `a` cleanly, then re-open
        // `b` alone via `title_memory_alias_rebound` on `a`.
        clear_all_pins();
        // SAFETY: `b` is writable committed view memory.
        unsafe { ((b + 4) as *mut u8).write_volatile(0x42) };
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert_eq!(guard(a, a + BLOCK), Some(vec![]));
        assert!(!writable(b));

        // Simulate `b` having lost its read-only guard while `a` remained
        // `GuardProtected + GuardClean`, then notify an alias rebind on `a`.
        {
            let mut old = 0u32;
            DIR.set_bit(b / BLOCK, BitField::GuardProtected, false, Ordering::SeqCst);
            // SAFETY: `b` is our committed view from `two_views`.
            let prot = unsafe {
                windows_sys::Win32::System::Memory::VirtualProtect(
                    b as *const _,
                    BLOCK,
                    PAGE_READWRITE,
                    &mut old,
                )
            };
            assert_ne!(prot, 0);
        }
        assert!(writable(b));
        title_memory_alias_rebound(a, BLOCK);
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert_eq!(guard(a, a + BLOCK), Some(vec![]));

        // `b` must also be guarded now so subsequent CPU writes through `b` open `a`.
        assert!(!writable(b));
        // SAFETY: writing through guarded `b` faults into the VEH and opens `a` too.
        unsafe { ((b + 12) as *mut u8).write_volatile(0x99) };
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));

        drop_views(section, &[a, b], BLOCK);
    }

    /// A page whose alias cannot be guarded (pinned for a kernel write,
    /// `LazyArmed`, or failing `VirtualProtect`) stays changed across `guard`
    /// and `Tracked::take`; aliasing a busy mapping to a guarded page marks
    /// the guarded page changed.
    #[test]
    fn a_busy_or_unguardable_alias_leaves_the_page_changed() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualProtect, VirtualQuery,
        };
        let _journal_lock = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (section, a, b) = two_views(BLOCK);
        let t = track(a, PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(
            t.take(a, a + PAGE).is_empty(),
            "clean once both views are guarded"
        );

        // Pin alias `b` via `kernel_writing`: `a` is opened and stays changed
        // on every `take` while `b` remains pinned.
        kernel_writing(b, 16);
        assert!(writable(b));
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(
            t.take(a, a + PAGE),
            [(a, a + PAGE)],
            "take must leave the page changed while alias `b` is pinned"
        );
        // SAFETY: `b` is open and writable under `kernel_writing`.
        unsafe { (b as *mut u32).write_volatile(0x1234) };
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);

        // Once `b`'s pin expires, `take` reports once more, guards both views,
        // and becomes clean.
        clear_all_pins();
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(
            t.take(a, a + PAGE).is_empty(),
            "taken cleanly once the busy alias pin ended"
        );
        assert!(!writable(a));
        assert!(!writable(b));

        // Aliasing a pinned mapping `c` to guarded `a` via `title_memory_alias_rebound`
        // marks `a` changed without disturbing `c`'s pin.
        let (sec2, c, d) = two_views(BLOCK);
        // Make `c` alias `a`'s section in `inplace` while `c` is pinned.
        title_memory_remapped(c, BLOCK);
        kernel_writing(c, 16);
        crate::inplace::title_memory_view(c, BLOCK, section, 0);
        // `views_mapped` opened `a`; re-verify `take` stays changed while `c` is pinned.
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        // Clear `c`'s pin -> `a` becomes clean again, then pin `c` and exercise
        // `title_memory_alias_rebound(c, BLOCK)` directly.
        clear_all_pins();
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + PAGE).is_empty());
        {
            let until = std::time::Instant::now() + PIN;
            pin_blocks_for_test(c / BLOCK, c / BLOCK + 1, until);
            DIR.set_bit(c / BLOCK, BitField::GuardProtected, false, Ordering::SeqCst);
        }
        title_memory_alias_rebound(c, BLOCK);
        assert!(
            !DIR.is_guard_clean(a / BLOCK),
            "rebounding a pinned alias must mark the guarded view unclean"
        );
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        clear_all_pins();

        // A `LazyArmed` (`PAGE_NOACCESS`) alias `b` must NEVER have its
        // `PAGE_NOACCESS` overwritten with `PAGE_READONLY`, and must leave `a`
        // changed until `b` is no longer lazy-armed.
        title_memory_remapped(c, BLOCK);
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + PAGE).is_empty());
        writing(a, BLOCK);
        let mut old = 0u32;
        // SAFETY: `b` is our committed view; simulate `LazyArmed` (`PAGE_NOACCESS`).
        let prot = unsafe { VirtualProtect(b as *const _, PAGE, PAGE_NOACCESS, &mut old) };
        assert_ne!(prot, 0);
        note_lazy_view_armed(&[b]);
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(
            t.take(a, a + PAGE),
            [(a, a + PAGE)],
            "a page with a LazyArmed alias must stay changed"
        );
        // SAFETY: `MEMORY_BASIC_INFORMATION` is plain old data.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: plain `VirtualQuery` on committed `b`.
        let queried = unsafe { VirtualQuery(b as *const _, &mut info, size_of_val(&info)) };
        assert_ne!(queried, 0);
        assert_eq!(info.State, MEM_COMMIT);
        assert_eq!(
            info.Protect, PAGE_NOACCESS,
            "guarding view `a` must not clobber LazyArmed PAGE_NOACCESS on alias `b`"
        );
        clear_lazy_view_armed(&[b]);
        // SAFETY: restoring RW on `b` before cleanup.
        let prot_rw = unsafe { VirtualProtect(b as *const _, PAGE, PAGE_READWRITE, &mut old) };
        assert_ne!(prot_rw, 0);
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + PAGE).is_empty());

        drop_views(section, &[a, b], BLOCK);
        drop_views(sec2, &[c, d], BLOCK);
    }

    /// A VEH write fault on a guarded alias view opens the clean view of the
    /// same physical page and calls `descriptor::host::remapped` after dropping
    /// `GUARDS`, refreshing indexed textures without risking lock inversion.
    #[test]
    fn alias_write_fault_triggers_remapped_without_deadlock() {
        let (section, a, b) = two_views(BLOCK);
        // Guard `a`: `a` is protected+clean, `b` is protected as an alias.
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert_eq!(guard(a, a + BLOCK), Some(Vec::new()));

        // Direct `open` check: opening alias `b` reports clean view `a`.
        let mut dirty = Vec::new();
        assert_ne!(open_for_test(b + 8, b + 9, |block| dirty.push(block)), 0);
        assert_eq!(dirty, [a]);

        // Re-guard `a` so a real CPU write through `b` enters the VEH handler.
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert_eq!(guard(a, a + BLOCK), Some(Vec::new()));

        // SAFETY: writing to guarded alias `b` faults into `on_write` and marks `a` dirty.
        unsafe { ((b + 16) as *mut u8).write_volatile(0x7c) };
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        drop_views(section, &[a, b], BLOCK);
    }

    /// Guarded view memory is left out of `ask_ahead` / `ask_once` (even when
    /// not registered in `inplace`, so `in_view` is false and only `known_view`
    /// identifies it), while non-view write-watched memory is still queried via
    /// `GetWriteWatch` and its writes are reported.
    #[test]
    fn ask_once_skips_known_guarded_views_and_watches_private_memory() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        struct AskOnceGuard;
        impl Drop for AskOnceGuard {
            fn drop(&mut self) {
                ask_once_end();
            }
        }
        struct CleanupGuard {
            priv_p: usize,
            section: HANDLE,
            view_p: usize,
        }
        impl Drop for CleanupGuard {
            fn drop(&mut self) {
                title_memory_remapped(self.view_p, PAGE);
                DIR.clear_written_range(self.priv_p / PAGE, self.priv_p / PAGE + 1);
                // SAFETY: clean up view, section, and private allocation owned by this test.
                unsafe {
                    UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: self.view_p as *mut _,
                    });
                    CloseHandle(self.section);
                    VirtualFree(self.priv_p as *mut _, 0, MEM_RELEASE);
                }
            }
        }

        let _journal_lock = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: fresh write-watched private allocation and pagefile view.
        let (priv_p, section, view_p) = unsafe {
            let priv_p = VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize;
            assert_ne!(priv_p, 0);
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view_p = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE).Value as usize;
            assert_ne!(view_p, 0);
            (priv_p, section, view_p)
        };
        let _cleanup = CleanupGuard {
            priv_p,
            section,
            view_p,
        };
        // Baseline: clear private watch bit and guard the view so `known_view` is true.
        assert!(writes(priv_p, priv_p + PAGE, false).is_some());
        assert!(guard(view_p, view_p + PAGE).is_some());
        assert!(known_view(view_p));
        assert!(!known_view(priv_p));

        // Write a byte to private memory, then call `ask_once` with both ranges.
        // SAFETY: committed private write-watched page.
        unsafe { ((priv_p + 16) as *mut u8).write_volatile(0x7a) };
        let ask_guard = AskOnceGuard;
        ask_once(&[(priv_p, priv_p + PAGE), (view_p, view_p + PAGE)]);
        let (now, priv_written) = asked(priv_p, priv_p + PAGE).expect("private range asked ahead");
        assert!(
            priv_written
                .iter()
                .any(|&(p, w)| p == priv_p / PAGE && w == now),
            "private write must still be observed via GetWriteWatch"
        );
        assert!(
            asked(view_p, view_p + PAGE).is_none(),
            "known guarded view must be skipped by ask_once"
        );
        drop(ask_guard);

        // `wrote_own` records both the page and its aliases in the journal.
        let (before, own_now) = wrote_own(view_p / PAGE, &[view_p / PAGE + 1]);
        assert!(own_now > before);
    }

    #[test]
    fn lock_free_page_dir_dirty_fast_path_and_ask_once_coherence() {
        let _journal_lock = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        struct AllocGuard(usize);
        impl Drop for AllocGuard {
            fn drop(&mut self) {
                title_memory_remapped(self.0, 6 * PAGE);
                DIR.clear_written_range(self.0 / PAGE, self.0 / PAGE + 6);
                // SAFETY: owned by this test.
                unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE) };
            }
        }
        struct AskOnceGuard;
        impl Drop for AskOnceGuard {
            fn drop(&mut self) {
                ask_once_end();
            }
        }

        // SAFETY: 6 pages of private write-watched memory.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                6 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        DIR.clear_written_range(base / PAGE, base / PAGE + 6);
        let _alloc = AllocGuard(base);

        // Baseline reset.
        assert!(writes(base, base + 6 * PAGE, false).is_some());

        // Write page 1 and page 4, then batch via `ask_once` with two ranges
        // separated by a 1-page gap (page 3).
        // SAFETY: committed pages inside `base`.
        unsafe {
            ((base + PAGE + 4) as *mut u8).write_volatile(0x11);
            ((base + 4 * PAGE + 12) as *mut u8).write_volatile(0x22);
        }
        let _ask = AskOnceGuard;
        ask_once(&[(base, base + 3 * PAGE), (base + 4 * PAGE, base + 6 * PAGE)]);
        let (e0, w0) = asked(base, base + 3 * PAGE).expect("first sub-range covered");
        let (e1, w1) = asked(base + 4 * PAGE, base + 6 * PAGE).expect("second sub-range covered");
        assert_eq!(e0, e1);
        assert!(w0.iter().any(|&(p, w)| p == base / PAGE + 1 && w == e0));
        assert!(w1.iter().any(|&(p, w)| p == base / PAGE + 4 && w == e1));
        assert!(
            asked(base + 3 * PAGE, base + 4 * PAGE).is_none(),
            "gap between asked ranges is not claimed in ahead"
        );

        // `writes` hits the lock-free `PageDir` fast path while `ask_once` stands.
        let (wf0, ww0) = writes(base, base + 3 * PAGE, true).expect("fast path");
        assert_eq!(wf0, e0);
        assert_eq!(ww0, w0);

        // `wrote` during `ask_once` advances the epoch and marks page 2 dirty.
        wrote(base + 2 * PAGE + 8, 16);
        let (e2, w2) = writes(base, base + 3 * PAGE, true).expect("after wrote");
        assert!(e2 > e0);
        assert!(w2.iter().any(|&(p, w)| p == base / PAGE + 2 && w == e2));

        // Uncovered sub-range past `base + 6 * PAGE` is not reported in `asked`.
        assert!(asked(base + 6 * PAGE, base + 7 * PAGE).is_none());
    }

    #[test]
    fn cold_pages_promote_to_readonly_and_fault_back_on_write() {
        let _journal_lock = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        struct AllocGuard(usize, usize);
        impl Drop for AllocGuard {
            fn drop(&mut self) {
                title_memory_remapped(self.0, self.1);
                DIR.clear_written_range(self.0 / PAGE, (self.0 + self.1) / PAGE);
                // SAFETY: owned by this test.
                unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE) };
            }
        }

        let threshold = cold_after();
        if threshold == 0 {
            return;
        }
        const PAGES: usize = 4;
        // SAFETY: 4 pages of private write-watched memory.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGES * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        DIR.clear_written_range(base / PAGE, base / PAGE + PAGES);
        let _alloc = AllocGuard(base, PAGES * PAGE);

        // Initial clean sweep to clear initial state.
        assert!(writes(base, base + PAGES * PAGE, false).is_some());

        // Advance `threshold` distinct frames with zero writes so pages promote to `PAGE_READONLY`.
        let mut token = 0u64;
        for _ in 0..threshold {
            let _ = crate::frame::wait(std::ptr::null(), 0, 0, std::ptr::null(), 2, &mut token);
            assert!(writes(base, base + PAGES * PAGE, false).is_some());
        }

        for i in 0..PAGES {
            assert!(
                DIR.is_cold_ro(base / PAGE + i),
                "page {i} should promote to cold_ro after {threshold} clean frames"
            );
            assert!(
                !writable(base + i * PAGE),
                "cold-promoted page {i} must be PAGE_READONLY"
            );
        }

        // Subsequent `writes` skips `GetWriteWatch` on cold pages and returns clean.
        let (cold_epoch, _) = writes(base, base + PAGES * PAGE, false).expect("all cold");
        assert!(cold_epoch > 0);

        // CPU write to page 1 faults through VEH `on_write`, restores `PAGE_READWRITE`,
        // clears `cold_ro`, resets streak, and records page 1 dirty.
        // SAFETY: inside committed allocation; VEH restores write access.
        unsafe { ((base + PAGE + 32) as *mut u8).write_volatile(0xab) };
        assert!(!DIR.is_cold_ro(base / PAGE + 1));
        assert!(writable(base + PAGE));
        assert_eq!(DIR.cold_streak(base / PAGE + 1), 0);
        // SAFETY: written byte is readable.
        let byte = unsafe { ((base + PAGE + 32) as *const u8).read_volatile() };
        assert_eq!(byte, 0xab);

        let (after_fault, list) = writes(base, base + PAGES * PAGE, false).expect("post-fault");
        assert!(
            list.iter()
                .any(|&(p, w)| p == base / PAGE + 1 && w > cold_epoch && w <= after_fault),
            "faulted cold page must appear dirty with epoch > cold_epoch"
        );

        // `writing` and `kernel_writing` demote remaining cold pages cleanly.
        assert!(DIR.is_cold_ro(base / PAGE + 2));
        writing(base + 2 * PAGE, PAGE);
        assert!(!DIR.is_cold_ro(base / PAGE + 2));
        assert!(writable(base + 2 * PAGE));

        assert!(DIR.is_cold_ro(base / PAGE + 3));
        kernel_writing(base + 3 * PAGE, PAGE);
        assert!(!DIR.is_cold_ro(base / PAGE + 3));
        assert!(writable(base + 3 * PAGE));
        clear_all_pins();
    }

    #[test]
    fn coalesced_virtual_protect_handles_mixed_views_and_holes() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        const TOTAL: usize = 6;
        // SAFETY: 6-block placeholder with views at blocks 0..2 and 4..6, leaving blocks 2..4 unmapped.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                TOTAL * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (TOTAL * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map_span = |block: usize, count: usize| {
            // SAFETY: splits placeholder for `count` blocks and maps section slice.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    count * BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    count * BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };

        let v0 = map_span(0, 2);
        let v1 = map_span(4, 2);
        title_memory_remapped(p, TOTAL * BLOCK);

        // Guard across all 6 blocks: committed blocks 0,1,4,5 become protected+clean,
        // placeholder hole blocks 2,3 are marked absent in one uncommitted skip.
        let first = guard(p, p + TOTAL * BLOCK).expect("mixed views+holes is a view");
        assert_eq!(first.len(), TOTAL);
        for b in [0, 1, 4, 5] {
            let blk = p / BLOCK + b;
            assert!(DIR.is_guard_protected(blk), "block {b} must be protected");
            assert!(DIR.is_guard_clean(blk), "block {b} must be clean");
            assert!(!writable(p + b * BLOCK), "block {b} must be PAGE_READONLY");
        }
        for b in [2, 3] {
            let blk = p / BLOCK + b;
            assert!(DIR.is_guard_absent(blk), "hole block {b} must be absent");
        }

        // Second guard check hits lock-free fast path cleanly? Wait: holes are absent (not clean),
        // so `guard` reports empty written vec because absent blocks stay absent!
        assert_eq!(guard(p, p + TOTAL * BLOCK), Some(Vec::new()));

        // Remapping restores `PAGE_READWRITE` across coalesced spans cleanly.
        title_memory_remapped(p, TOTAL * BLOCK);
        for b in [0, 1, 4, 5] {
            assert!(writable(p + b * BLOCK), "block {b} must be restored RW");
        }

        // SAFETY: unmap views and free remaining placeholders.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), v0, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), v1, MEM_PRESERVE_PLACEHOLDER);
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
            VirtualFree((p + 2 * BLOCK) as *mut _, 0, MEM_RELEASE);
            VirtualFree((p + 4 * BLOCK) as *mut _, 0, MEM_RELEASE);
            CloseHandle(section);
        }
        title_memory_remapped(p, TOTAL * BLOCK);
    }

    #[test]
    fn concurrent_watch_writes_faults_and_remaps_are_safe() {
        let _journal_lock = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        const SIZE: usize = 4 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        // SAFETY: private write-watched buffer for concurrent `writes` + `asked`.
        let priv_p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                SIZE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(priv_p, 0);

        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_w = stop.clone();
        let stop_r = stop.clone();
        let stop_m = stop.clone();

        // Thread 1: repeatedly writes through `one` and `priv_p` (triggering VEH faults on `one`).
        let writer = std::thread::spawn(move || {
            let mut iter = 0u8;
            while !stop_w.load(std::sync::atomic::Ordering::Relaxed) {
                for b in 0..4 {
                    // SAFETY: `one` and `priv_p` are mapped `SIZE` bytes; VEH opens `one` on write.
                    unsafe {
                        ((one + b * BLOCK + 16) as *mut u8).write_volatile(iter);
                        ((priv_p + b * PAGE + 16) as *mut u8).write_volatile(iter);
                    }
                }
                iter = iter.wrapping_add(1);
                std::thread::yield_now();
            }
        });

        // Thread 2: repeatedly guards `two` and checks `writes` on `priv_p`.
        let reader = std::thread::spawn(move || {
            while !stop_r.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = writes(two, two + SIZE, true);
                let _ = writes(priv_p, priv_p + SIZE, false);
                let _r = title_memory_reading(two, SIZE);
                std::thread::yield_now();
            }
        });

        // Thread 3: repeatedly exercises `writing`, `title_memory_remapping`, and `title_memory_alias_rebound`.
        let remapper = std::thread::spawn(move || {
            while !stop_m.load(std::sync::atomic::Ordering::Relaxed) {
                writing(two, BLOCK);
                {
                    let _changing = title_memory_remapping(two + BLOCK, BLOCK);
                    title_memory_alias_rebound(two + BLOCK, BLOCK);
                }
                std::thread::yield_now();
            }
        });

        std::thread::sleep(std::time::Duration::from_millis(120));
        stop.store(true, std::sync::atomic::Ordering::SeqCst);
        writer.join().unwrap();
        reader.join().unwrap();
        remapper.join().unwrap();

        drop_views(section, &[one, two], SIZE);
        DIR.clear_written_range(priv_p / PAGE, (priv_p + SIZE) / PAGE);
        // SAFETY: private allocation owned by this test.
        unsafe { VirtualFree(priv_p as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn try_arm_lazy_under_guards_respects_pins_and_clears_guard_and_cold_state() {
        let _journal_lock = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (section, one, two) = two_views(2 * BLOCK);
        let b0 = one / BLOCK;
        let b1 = b0 + 1;
        let b2 = two / BLOCK;

        // Guard `one` so `b0`, `b1` are protected/clean and `b2` is aliased/protected.
        assert!(guard(one, one + 2 * BLOCK).is_some());
        assert!(DIR.is_guard_protected(b0));
        assert!(DIR.is_guard_clean(b0));
        assert!(DIR.is_guard_protected(b2));

        // Pin `b1`: `try_arm_lazy_under_guards(&[b0, b1])` must refuse without
        // invoking `arm` or clearing `b0`'s guard metadata.
        pin_blocks_for_test(b1, b1 + 1, std::time::Instant::now() + PIN);
        let mut armed = false;
        assert!(!try_arm_lazy_under_guards(&[b0, b1, b2], |_| {
            armed = true;
            true
        }));
        assert!(!armed, "arm closure must not run when any block is pinned");
        assert!(DIR.is_guard_protected(b0));

        // Clear pin, simulate a cold bit on `b1`, and arm under guard shards.
        clear_all_pins();
        DIR.set_bit(b1, BitField::ColdRo, true, Ordering::SeqCst);
        let blocks = [b0, b1, b2];
        let ok = try_arm_lazy_under_guards(&blocks, |cpu_clean_after| {
            assert!(cpu_clean_after.is_some());
            for &b in &blocks {
                assert!(!DIR.is_guard_protected(b));
                assert!(!DIR.is_guard_clean(b));
                assert!(!DIR.get_bit(b, BitField::GuardAliased, Ordering::SeqCst));
                assert!(!DIR.is_cold_ro(b));
                assert!(writable(b * BLOCK));
            }
            true
        });
        assert!(ok);
        drop_views(section, &[one, two], 2 * BLOCK);
    }

    #[test]
    fn wrote_own_epoch_is_monotonic_against_concurrent_higher_epoch() {
        let _journal_lock = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let page = 0x7df0_0000_0000usize / PAGE;
        DIR.clear_written_range(page, page + 2);

        let high = ASK_EPOCH.fetch_add(10, Ordering::AcqRel) + 10;
        DIR.record_written(page, high);
        // Even if `record_written_prev` is called with a lower epoch, `written_epoch`
        // must never roll backwards below `high`.
        let prev = DIR.record_written_prev(page, high - 5);
        assert_eq!(prev, high);
        let (entries, max_e) = DIR.collect_written(page, page + 1);
        assert_eq!(entries, vec![(page, high)]);
        assert_eq!(max_e, high);

        let (_, now) = wrote_own(page, &[page + 1]);
        assert!(now > high);
        let (entries2, max_e2) = DIR.collect_written(page, page + 2);
        assert_eq!(entries2, vec![(page, now), (page + 1, now)]);
        assert_eq!(max_e2, now);
        DIR.clear_written_range(page, page + 2);
    }

    #[test]
    fn watch_lazy_armed_view_is_lock_free_and_preserves_noaccess() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READWRITE, VirtualProtect,
            VirtualQuery,
        };
        let _journal_lock = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (section, va, two) = two_views(BLOCK);
        let block = va / PAGE;
        DIR.clear_written_range(block, block + 1);

        let mut old = 0u32;
        // SAFETY: `va` is a committed section view mapped by `two_views`.
        let prot_na = unsafe { VirtualProtect(va as *const _, PAGE, PAGE_NOACCESS, &mut old) };
        assert_ne!(prot_na, 0);
        note_lazy_view_armed(&[va]);
        assert!(DIR.is_lazy_armed(block));
        let defer_epoch = DIR.written_epoch(block);
        assert!(defer_epoch > 0);

        let (now, written) = writes(va, va + PAGE, true).expect("view writes");
        assert!(now >= defer_epoch);
        assert_eq!(written, vec![(block, defer_epoch)]);

        // SAFETY: plain `VirtualQuery` into a zeroed `MEMORY_BASIC_INFORMATION`.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `va` is a valid committed address in this process.
        let queried = unsafe { VirtualQuery(va as *const _, &mut info, size_of_val(&info)) };
        assert_ne!(queried, 0);
        assert_eq!(info.State, MEM_COMMIT);
        assert_eq!(info.Protect, PAGE_NOACCESS);

        clear_lazy_view_armed(&[va]);
        assert!(!DIR.is_lazy_armed(block));
        // SAFETY: restoring RW protection before resuming normal `PAGE_READONLY` guarding.
        let prot_rw = unsafe { VirtualProtect(va as *const _, PAGE, PAGE_READWRITE, &mut old) };
        assert_ne!(prot_rw, 0);
        let (now2, written2) = writes(va, va + PAGE, true).expect("guard resumed");
        assert!(now2 > defer_epoch);
        assert_eq!(written2, vec![(block, now2)]);
        assert!(DIR.is_guard_protected(block));
        assert!(DIR.is_guard_clean(block));
        assert!(!writable(va));

        DIR.clear_written_range(block, block + 1);
        drop_views(section, &[va, two], BLOCK);
    }

    #[test]
    fn the_parts_of_a_page_written_since_it_was_taken() {
        let _journal_lock = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let p7 = 0x7df1_0000_0000usize / PAGE;
        let p9 = p7 + 2;
        DIR.clear_written_range(p7, p9 + 1);

        assert_eq!(DIR.parts_since(p7, 0), WHOLE);
        DIR.record_written_parts(p7, WHOLE, 3);
        // Parts 0 and 5 at 4, part 5 again at 6: at 2 the whole page is due.
        DIR.record_written_parts(p7, 0b10_0001, 4);
        DIR.record_written_parts(p7, 0b10_0000, 6);
        assert_eq!(DIR.parts_since(p7, 2), WHOLE);
        if parts_on() {
            assert_eq!(DIR.parts_since(p7, 3), 0b10_0001);
            assert_eq!(DIR.parts_since(p7, 4), 0b10_0000);
            assert_eq!(DIR.parts_since(p7, 5), 0b10_0000);
            assert_eq!(DIR.parts_since(p7, 6), 0);
        } else {
            assert_eq!(DIR.parts_since(p7, 3), WHOLE);
        }
        // Written whole: all of it, whoever took it before.
        DIR.record_written_parts(p7, WHOLE, 8);
        assert_eq!(DIR.parts_since(p7, 7), WHOLE);
        // A page never written whole before: only the recorded part is due
        // (or whole page when `VICTORY_PAGE_PARTS=0`).
        DIR.record_written_parts(p9, 0b100, 9);
        assert_eq!(
            DIR.parts_since(p9, 5),
            if parts_on() { 0b100 } else { WHOLE }
        );
        DIR.clear_written_range(p7, p9 + 1);
    }

    /// Of a page taken before, what is taken again is the 4 KiB pages written
    /// since (or the full 64 KiB page when `VICTORY_PAGE_PARTS=0`); a page
    /// never taken before is taken whole.
    #[test]
    fn what_is_taken_again_is_what_was_written() {
        let _journal_lock = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        DIR.clear_written_range(p / PAGE, p / PAGE + 4);
        let write = |at: usize| {
            // SAFETY: committed pages of ours.
            unsafe { ((p + at) as *mut u8).write_volatile(1) }
        };
        let tracked = track(p, 4 * PAGE, false).expect("watched");
        write(5);
        write(PAGE + 5000);
        assert_eq!(
            tracked.take(p, p + 4 * PAGE),
            [(p, p + 2 * PAGE)],
            "never taken"
        );
        assert!(tracked.take(p, p + 4 * PAGE).is_empty());
        write(PAGE + 5000);
        write(PAGE + 8192);
        write(PAGE + 65535);
        write(2 * PAGE);
        if parts_on() && take_parts_on() {
            assert_eq!(
                tracked.take(p, p + 4 * PAGE),
                [
                    (p + PAGE + 4096, p + PAGE + 3 * 4096),
                    (p + 2 * PAGE - 4096, p + 3 * PAGE),
                ],
                "page 2 was never taken"
            );
        } else {
            assert_eq!(
                tracked.take(p, p + 4 * PAGE),
                [(p + PAGE, p + 3 * PAGE)],
                "whole 64 KiB pages when VICTORY_PAGE_PARTS=0"
            );
        }
        write(2 * PAGE + 70);
        // Two askings' writes are both included when taken after both.
        assert!(writes(p, p + 4 * PAGE, false).is_some());
        write(2 * PAGE + 9000);
        if parts_on() && take_parts_on() {
            assert_eq!(
                tracked.take(p, p + 4 * PAGE),
                [
                    (p + 2 * PAGE, p + 2 * PAGE + 4096),
                    (p + 2 * PAGE + 8192, p + 2 * PAGE + 12288),
                ]
            );
        } else {
            assert_eq!(
                tracked.take(p, p + 4 * PAGE),
                [(p + 2 * PAGE, p + 3 * PAGE)]
            );
        }
        DIR.clear_written_range(p / PAGE, p / PAGE + 4);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A write opens the view it goes through and no other (`VICTORY_OPEN_ONE_VIEW`):
    /// the others stay read-only, are reported written, and take writes of their own.
    #[test]
    fn a_write_opens_the_view_it_goes_through() {
        let (section, one, two) = two_views(BLOCK);
        let page = |view: usize| view / PAGE;
        for view in [one, two] {
            assert_eq!(guard(view, view + BLOCK), Some(vec![page(view)]));
        }
        // SAFETY: `two` is a mapped view, opened by the VEH handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert!(writable(two) && !writable(one));
        // SAFETY: `one` is a mapped view, opened by the VEH handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert!(writable(two) && writable(one));
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        // Written at the view left read-only after the other faulted.
        // SAFETY: `two` is a mapped view, opened by the VEH handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert_eq!(
            guard(one, one + BLOCK),
            Some(vec![page(one)]),
            "not clean, though never opened"
        );
        assert!(!writable(two) && !writable(one));
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        // A synchronous write to the still-guarded view opens it.
        // SAFETY: `two` is a mapped view, opened by the VEH handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert!(!writable(one));
        writing(one, 8);
        assert!(writable(one));
        drop_views(section, &[one, two], BLOCK);
    }

    /// A block just opened by a CPU write fault rests writable (`VICTORY_GUARD_REST`)
    /// and is reported written at every view of its physical page until the rest ends.
    #[test]
    fn a_block_just_written_rests() {
        let (section, one, two) = two_views(BLOCK);
        let page = one / PAGE;
        let rests =
            |on: bool| REST_IN_TEST.set(std::time::Duration::from_secs(if on { 3600 } else { 0 }));
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        rests(true);
        assert_eq!(
            guard(one, one + BLOCK),
            Some(Vec::new()),
            "nothing written: as it was"
        );
        // SAFETY: `two` is a mapped view, opened by the VEH handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        for _ in 0..3 {
            assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
            assert!(writable(two), "no fault for the next write");
            assert!(writes(two, two + BLOCK, true).is_some());
            assert!(!clean(two / PAGE) && !clean(one / PAGE));
            // SAFETY: `two` is writable during the guard rest window.
            unsafe { ((two + 8) as *mut u8).write(1) };
        }
        rests(false);
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert!(!writable(two) && !writable(one));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        drop_views(section, &[one, two], BLOCK);
    }

    /// Exercises sharded `GUARD_SHARDS_TABLE` across single-shard, cross-shard,
    /// and multi-view alias boundaries under concurrent `guard()`,
    /// `try_arm_lazy_under_guards()`, `kernel_writing()`, and `on_write()` VEH
    /// faults, verifying pin/rest semantics and deadlock freedom.
    #[test]
    fn sharded_guards_concurrent_ops_and_cross_shard_boundaries() {
        let _journal_lock = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // `SHARD_BLOCKS + 4` blocks (1.25 MiB) guarantees every view spans at
        // least two distinct guard shards.
        const BLOCKS: usize = SHARD_BLOCKS + 4;
        const SIZE: usize = BLOCKS * BLOCK;
        let (sec_a, one, two) = two_views(SIZE);
        let (sec_b, three, four) = two_views(SIZE);

        let b_one = one / BLOCK;
        let b_two = two / BLOCK;
        let b_three = three / BLOCK;
        assert_ne!(
            shard_of(b_one),
            shard_of(b_one + BLOCKS - 1),
            "test view must cross a guard shard boundary"
        );
        // Find the first index inside `0..BLOCKS` where the shard changes.
        let boundary = (1..BLOCKS)
            .find(|&i| shard_of(b_one + i) != shard_of(b_one + i - 1))
            .expect("shard boundary inside multi-stripe view");

        // 1. Cross-shard guarding & pin semantics across both alias views.
        let initial = guard(one, one + SIZE).expect("view guarded");
        assert_eq!(initial.len(), BLOCKS);
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        for i in 0..BLOCKS {
            assert!(DIR.is_guard_protected(b_one + i));
            assert!(DIR.is_guard_clean(b_one + i));
            assert!(!writable(one + i * BLOCK));
            assert!(!writable(two + i * BLOCK));
        }

        // Pin two blocks straddling the shard boundary via `kernel_writing`.
        let pin_lo = boundary - 1;
        let pin_hi = boundary + 1;
        kernel_writing(one + pin_lo * BLOCK, (pin_hi - pin_lo) * BLOCK);
        for i in pin_lo..pin_hi {
            assert!(
                writable(one + i * BLOCK),
                "pinned primary block {i} must be writable"
            );
            assert!(
                writable(two + i * BLOCK),
                "pinned alias block {i} must be writable"
            );
        }
        let expected_pinned: Vec<usize> = (pin_lo..pin_hi).map(|i| b_one + i).collect();
        assert_eq!(guard(one, one + SIZE), Some(expected_pinned.clone()));
        assert_eq!(
            guard(one, one + SIZE),
            Some(expected_pinned),
            "blocks straddling shard boundary stay open while pinned"
        );

        // `try_arm_lazy_under_guards` across multiple shards must refuse if ANY
        // block in ANY shard is pinned, leaving unpinned blocks untouched.
        let unpinned_idx = (boundary + 2).min(BLOCKS - 1);
        assert!(DIR.is_guard_protected(b_one + unpinned_idx));
        let mut arm_ran = false;
        assert!(!try_arm_lazy_under_guards(
            &[b_one + unpinned_idx, b_one + pin_lo, b_two + pin_hi - 1],
            |_| {
                arm_ran = true;
                true
            },
        ));
        assert!(
            !arm_ran,
            "arm must not run when a cross-shard block is pinned"
        );
        assert!(DIR.is_guard_protected(b_one + unpinned_idx));

        // Clearing pins allows immediate clean re-guarding across the boundary.
        clear_all_pins();
        assert_eq!(
            guard(one, one + SIZE),
            Some((pin_lo..pin_hi).map(|i| b_one + i).collect())
        );
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));

        // 2. Cross-shard alias CPU fault + `VICTORY_GUARD_REST` semantics.
        REST_IN_TEST.set(std::time::Duration::from_secs(3600));
        // SAFETY: `two` is our committed view; VEH `on_write` opens `two` and
        // marks `one` unclean across shards.
        unsafe { ((two + boundary * BLOCK + 16) as *mut u8).write_volatile(0x5a) };
        for _ in 0..2 {
            assert_eq!(guard(one, one + SIZE), Some(vec![b_one + boundary]));
            assert!(writable(two + boundary * BLOCK));
            assert!(!clean(b_one + boundary));
        }
        REST_IN_TEST.set(std::time::Duration::ZERO);
        assert_eq!(guard(one, one + SIZE), Some(vec![b_one + boundary]));
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));

        // 3. Concurrent multi-threaded stress across overlapping & disjoint
        //    shards (guard in opposing alias directions, VEH faults,
        //    kernel_writing across boundaries, try_arm_lazy_under_guards).
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut handles = Vec::new();

        // Thread 1: guard `one` and `three` across shard boundaries.
        {
            let stop = stop.clone();
            handles.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    assert!(guard(one, one + SIZE).is_some());
                    assert!(guard(three, three + SIZE).is_some());
                }
            }));
        }
        // Thread 2: guard `two` and `four` (opposing alias order vs Thread 1).
        {
            let stop = stop.clone();
            handles.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    assert!(guard(two, two + SIZE).is_some());
                    assert!(guard(four, four + SIZE).is_some());
                }
            }));
        }
        // Thread 3: trigger VEH `on_write` faults across shard boundaries on all 4 views.
        {
            let stop = stop.clone();
            handles.push(std::thread::spawn(move || {
                let mut v = 0u8;
                while !stop.load(Ordering::Relaxed) {
                    for &base in &[one, two, three, four] {
                        for idx in [0, boundary - 1, boundary, BLOCKS - 1] {
                            // SAFETY: mapped committed views; VEH restores RW on fault.
                            unsafe { ((base + idx * BLOCK + 8) as *mut u8).write_volatile(v) };
                        }
                    }
                    v = v.wrapping_add(1);
                    std::thread::yield_now();
                }
            }));
        }
        // Thread 4: `kernel_writing` straddling shard boundaries + pin cleanup.
        {
            let stop = stop.clone();
            handles.push(std::thread::spawn(move || {
                let mut iter = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    kernel_writing(one + (boundary - 1) * BLOCK, 3 * BLOCK);
                    writing(three + (boundary - 1) * BLOCK, 3 * BLOCK);
                    iter += 1;
                    if iter.is_multiple_of(8) {
                        clear_all_pins();
                    }
                    std::thread::yield_now();
                }
            }));
        }
        // Thread 5: `try_arm_lazy_under_guards` across multiple shards in unsorted order.
        {
            let stop = stop.clone();
            handles.push(std::thread::spawn(move || {
                let slice_a = [b_two + boundary, b_one + boundary - 1, b_one + boundary];
                let slice_b = [b_three + BLOCKS - 1, b_three, b_three + boundary];
                while !stop.load(Ordering::Relaxed) {
                    let _ = try_arm_lazy_under_guards(&slice_a, |_| {
                        for &b in &slice_a {
                            assert!(!DIR.is_guard_protected(b));
                            assert!(!DIR.is_guard_clean(b));
                            assert!(!DIR.is_cold_ro(b));
                        }
                        false
                    });
                    let _ = try_arm_lazy_under_guards(&slice_b, |_| {
                        for &b in &slice_b {
                            assert!(!DIR.is_guard_protected(b));
                            assert!(!DIR.is_guard_clean(b));
                            assert!(!DIR.is_cold_ro(b));
                        }
                        false
                    });
                    std::thread::yield_now();
                }
            }));
        }

        std::thread::sleep(std::time::Duration::from_millis(150));
        stop.store(true, Ordering::SeqCst);
        for h in handles {
            h.join().expect("worker thread must not panic or deadlock");
        }

        // Post-stress cleanup & sanity: clearing pins lets every view re-guard cleanly.
        clear_all_pins();
        assert!(guard(one, one + SIZE).is_some());
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert!(guard(three, three + SIZE).is_some());
        assert_eq!(guard(three, three + SIZE), Some(Vec::new()));

        drop_views(sec_a, &[one, two], SIZE);
        drop_views(sec_b, &[three, four], SIZE);
    }

    #[test]
    fn allocation_query_recognizes_file_views_and_rejects_nonviews() {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, MapViewOfFile, PAGE_READONLY, UnmapViewOfFile,
        };
        let file = tempfile::tempfile().unwrap();
        file.set_len(PAGE as u64).unwrap();
        // SAFETY: a valid, open file and a read-only mapping of its contents.
        let section = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: an open section containing at least one page.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_READ, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        assert!(mapped(view.Value as usize + 13));
        // SAFETY: our successful mapping and its section handle.
        unsafe {
            UnmapViewOfFile(view);
            CloseHandle(section);
        }
        assert!(!mapped((&raw const file) as usize)); // private stack allocation
        assert!(!mapped(mapped as *const () as usize)); // executable image
        assert!(!mapped(0)); // no allocation to query
    }

    /// Two write-watched allocations side by side (the runtime reserves
    /// title memory allocation by allocation), asked about as one range, as
    /// a buffer spanning them is: the pages written in each are reported.
    #[test]
    fn a_range_over_several_watched_allocations_is_watched() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (a, b) = loop {
            // SAFETY: a probe reservation, released at once to find room for two.
            let span =
                unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) };
            assert!(!span.is_null());
            // SAFETY: the probe above.
            unsafe { VirtualFree(span, 0, MEM_RELEASE) };
            let at = |p: usize| {
                // SAFETY: a fixed address in the room just released.
                let got = unsafe {
                    VirtualAlloc(
                        p as *const _,
                        2 * PAGE,
                        MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                        PAGE_READWRITE,
                    )
                };
                got as usize
            };
            let a = at(span as usize);
            let b = at(span as usize + 2 * PAGE);
            if a != 0 && b != 0 {
                break (a, b);
            }
            for p in [a, b].into_iter().filter(|&p| p != 0) {
                // SAFETY: ours.
                unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
            }
        };
        DIR.clear_written_range(a / PAGE, (b + 2 * PAGE) / PAGE);
        let _ = writes(a, b + 2 * PAGE, false);
        // SAFETY: committed pages of ours.
        unsafe {
            ((a + 8) as *mut u8).write(1);
            ((b + PAGE + 8) as *mut u8).write(1);
        }
        let (now, found) = writes(a, b + 2 * PAGE, false).expect("watched");
        assert_eq!(found, [(a / PAGE, now), (b / PAGE + 1, now)]);
        DIR.clear_written_range(a / PAGE, (b + 2 * PAGE) / PAGE);
        // SAFETY: ours.
        unsafe {
            VirtualFree(a as *mut _, 0, MEM_RELEASE);
            VirtualFree(b as *mut _, 0, MEM_RELEASE);
        }
    }

    /// A range in an allocation reserved but not committed yet (as the
    /// title's are before it uses them) has no writes, and is asked again
    /// once committed.
    #[test]
    fn a_range_never_committed_has_no_writes() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a fresh write-watched reservation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        // A previous test may have used and released these virtual addresses.
        DIR.clear_written_range(p / PAGE, p / PAGE + 2);
        let (_, none) = writes(p, p + 2 * PAGE, false).expect("nothing can be written there");
        assert!(none.is_empty());
        // SAFETY: committing and writing a page of our reservation.
        unsafe {
            assert!(!VirtualAlloc(p as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            ((p + 8) as *mut u8).write(1);
        }
        let (now, found) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert_eq!(found, [(p / PAGE, now)]);
        DIR.clear_written_range(p / PAGE, p / PAGE + 2);
        // SAFETY: ours.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A range that begins on an unmapped reserved hole before a mapped view
    /// is still guarded, whereas a range starting in committed private memory
    /// before a mapped view is rejected so private pages are never
    /// read-protected.
    #[test]
    fn guard_handles_unmapped_prefix_and_rejects_committed_private_prefix() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFileEx,
            PAGE_NOACCESS, UnmapViewOfFile,
        };
        // SAFETY: create a 1-block pagefile section.
        let section = unsafe {
            CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                BLOCK as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());

        let mut pair = None;
        for _ in 0..16 {
            // SAFETY: probe for a free 2-block-aligned address range.
            let probe =
                unsafe { VirtualAlloc(std::ptr::null(), 2 * BLOCK, MEM_RESERVE, PAGE_NOACCESS) }
                    as usize;
            assert_ne!(probe, 0);
            // SAFETY: freeing the probe we just reserved.
            assert_ne!(unsafe { VirtualFree(probe as *mut _, 0, MEM_RELEASE) }, 0);
            // SAFETY: reserve block 0 as an unmapped placeholder.
            let prefix =
                unsafe { VirtualAlloc(probe as *const _, BLOCK, MEM_RESERVE, PAGE_NOACCESS) }
                    as usize;
            if prefix != probe {
                if prefix != 0 {
                    // SAFETY: freeing our reservation.
                    unsafe { VirtualFree(prefix as *mut _, 0, MEM_RELEASE) };
                }
                continue;
            }
            // SAFETY: map the 1-block section immediately after the reserved prefix.
            let view = unsafe {
                MapViewOfFileEx(
                    section,
                    FILE_MAP_WRITE,
                    0,
                    0,
                    BLOCK,
                    (probe + BLOCK) as *const _,
                )
                .Value as usize
            };
            if view == probe + BLOCK {
                pair = Some((prefix, view));
                break;
            }
            if view != 0 {
                // SAFETY: unmapping our view.
                unsafe {
                    UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: view as *mut _,
                    });
                }
            }
            // SAFETY: freeing our prefix reservation.
            unsafe { VirtualFree(prefix as *mut _, 0, MEM_RELEASE) };
        }
        let (prefix, view) = pair.expect("reserve prefix block immediately before mapped view");

        // 1. Unmapped reserved prefix + mapped view: must stay guarded.
        assert!(!mapped(prefix));
        assert!(
            guard(prefix, view + BLOCK).is_some(),
            "range spanning an unmapped reserved prefix into a mapped view must stay guarded"
        );
        // SAFETY: writing inside our mapped read-write view.
        unsafe { (view as *mut u8).write(42) };
        let written = guard(prefix, view + BLOCK).expect("still guarded");
        assert!(written.contains(&(view / PAGE)));

        // 2. Committed private prefix + mapped view: must NOT read-protect private memory.
        // SAFETY: committing our reserved prefix block as private read-write memory.
        let committed =
            unsafe { VirtualAlloc(prefix as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!committed.is_null());
        assert!(
            guard(prefix, view + BLOCK).is_none(),
            "range containing committed private memory must not be read-protected by guard"
        );
        // SAFETY: writing to the committed private block must still succeed without faulting.
        unsafe { (prefix as *mut u8).write(99) };

        title_memory_remapped(prefix, 2 * BLOCK);
        // SAFETY: clean up our view, section, and prefix reservation.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
            CloseHandle(section);
            VirtualFree(prefix as *mut _, 0, MEM_RELEASE);
        }
    }

    /// `fread`'s size times count saturates to `usize::MAX`: announcing and
    /// landing that kernel read is one range per shard (not per block of the
    /// whole address space), and landing it leaves nothing announced.
    #[test]
    fn a_saturated_kernel_read_is_announced_in_one_step() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Kernel half of the address space: no user allocation of another
        // test running at the same time can fall in the announced range.
        let at = 0xFFFF_8000_0000_0000;
        let range = (at / BLOCK, usize::MAX.div_ceil(BLOCK));
        let probe_block = at / BLOCK + 12345;
        kernel_will_write(at, usize::MAX);
        assert!(guard_shard_read(shard_of(probe_block)).announced(probe_block));
        kernel_wrote(at, usize::MAX);
        assert!(
            !guard_shard_read(shard_of(probe_block))
                .announced
                .contains_key(&range)
        );
    }

    /// A range announced again and again without landing (overlapped reads
    /// into one buffer) is one counted entry, and a remap over it drops it.
    #[test]
    fn repeated_announcements_are_counted_and_a_remap_drops_them() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let at = 0xFFFF_9000_0000_0000;
        let range = (at / BLOCK, (at + BLOCK).div_ceil(BLOCK));
        let s = shard_of(at / BLOCK);
        for _ in 0..1000 {
            kernel_will_write(at, BLOCK);
        }
        assert_eq!(guard_shard_read(s).announced.get(&range), Some(&1000));
        kernel_wrote(at, BLOCK);
        assert_eq!(guard_shard_read(s).announced.get(&range), Some(&999));
        title_memory_remapped(at, BLOCK);
        assert!(!guard_shard_read(s).announced.contains_key(&range));
    }

    /// A kernel read announced with `kernel_will_write` survives a refresh
    /// on another thread before it lands; once `kernel_wrote` says it has,
    /// the next refresh guards its block again.
    #[test]
    fn an_announced_kernel_read_survives_a_refresh_before_it() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let v = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize;
            CloseHandle(section);
            v
        };
        assert_ne!(view, 0);
        let end = view + 2 * BLOCK;
        let (first, initial) = writes(view, end, true).expect("new mapped view");
        assert_eq!(initial, [(view / PAGE, first), (view / PAGE + 1, first)]);
        let (next, clean) = writes(view + 8, end - 1, true).expect("same protected pages");
        assert!(next > first);
        assert_eq!(clean, initial);
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        kernel_will_write(view + 100, 13); // the title's thread, in `ReadFile`
        let _ = guard(view, view + 2 * BLOCK); // a refresh on another thread
        assert!(
            read(&mut file),
            "the refresh guarded the block again under the read"
        );
        kernel_wrote(view + 100, 13);
        let (changed, pages) = writes(view, end, true).expect("kernel write visible");
        assert_eq!(pages, [(view / PAGE, changed), (view / PAGE + 1, first)]);
        assert_eq!(writes(view, end, true).unwrap().1, pages);
        assert!(!read(&mut file), "guarded again once the read is done");
        title_memory_remapped(view, 2 * BLOCK);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
        }
    }

    #[test]
    fn ask_once_coalesces_ranges_and_surfaces_intervening_wrote() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let alloc = WatchedAllocGuard::new(6);
        let p = alloc.base;
        // Baseline clear.
        assert!(writes(p, p + 6 * PAGE, false).is_some());

        // Write page 1 and page 4, then coalesce two disjoint ranges inside the allocation.
        // SAFETY: committed write-watched pages of ours.
        unsafe {
            ((p + PAGE + 4) as *mut u8).write(7);
            ((p + 4 * PAGE + 4) as *mut u8).write(9);
        }
        let ask_guard = AskOnceGuard;
        ask_once(&[(p, p + 2 * PAGE), (p + 4 * PAGE, p + 5 * PAGE)]);
        let (now1, found1) = writes(p, p + 2 * PAGE, false).expect("covered by ask_once");
        let (now2, found2) =
            writes(p + 4 * PAGE, p + 5 * PAGE, false).expect("covered by ask_once");
        assert_eq!(found1.len(), 1);
        assert_eq!(found2.len(), 1);
        assert_eq!(found1[0].0, p / PAGE + 1);
        assert_eq!(found2[0].0, p / PAGE + 4);
        let coalesced_gen = found1[0].1;
        assert_eq!(
            found2[0].1, coalesced_gen,
            "coalesced asks share one asking number"
        );
        assert!(coalesced_gen <= now1 && coalesced_gen <= now2);

        // An intervening submission copy into page 0 updates the journal and advances `now`.
        wrote(p + 16, 64);
        let (now3, found3) = writes(p, p + 2 * PAGE, false).expect("still covered");
        assert!(now3 > coalesced_gen);
        assert_eq!(found3.len(), 2);
        assert_eq!(found3[0].0, p / PAGE);
        assert!(found3[0].1 > coalesced_gen && found3[0].1 <= now3);
        assert_eq!(found3[1], (p / PAGE + 1, coalesced_gen));

        drop(ask_guard);
        // After `ask_once_end`, a fresh CPU write is seen by the kernel again.
        // SAFETY: committed write-watched page of ours.
        unsafe { ((p + 5 * PAGE + 8) as *mut u8).write(11) };
        let (now4, found4) = writes(p + 5 * PAGE, p + 6 * PAGE, false).expect("watched");
        assert!(now4 > now3);
        assert_eq!(found4, [(p / PAGE + 5, now4)]);
    }

    #[test]
    fn a_file_read_after_the_submission_query_reaches_every_reader() {
        use std::io::{Read, Seek, Write};
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let alloc = WatchedAllocGuard::new(1);
        let at = alloc.base;
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"texel").unwrap();
        for already_announced in [false, true] {
            let _ask = AskOnceGuard;
            let (before, _) = writes(at, at + PAGE, false).unwrap();
            if already_announced {
                kernel_will_write(at, 5);
            }
            ask_once(&[(at, at + PAGE)]);
            if !already_announced {
                kernel_will_write(at, 5);
            }
            file.rewind().unwrap();
            // SAFETY: these five bytes belong to the committed allocation.
            let dest = unsafe { std::slice::from_raw_parts_mut(at as *mut u8, 5) };
            file.read_exact(dest).unwrap();
            assert_eq!(dest, b"texel");
            // An overlapped read can still have an open announcement when a
            // refresh runs. Another reader can ask after completion.
            for finished in [false, true] {
                if finished {
                    kernel_wrote(at, 5);
                }
                let (_, pages) = writes(at, at + PAGE, false).unwrap();
                assert!(
                    pages.iter().any(|&(p, w)| p == at / PAGE && w > before),
                    "already_announced={already_announced}, finished={finished}"
                );
            }
        }
    }

    #[test]
    fn ask_once_handles_mixed_watched_and_unwatched_ranges() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let alloc = WatchedAllocGuard::new(2);
        let watched = alloc.base;
        let mut plain = vec![0u8; PAGE];
        let plain_start = plain.as_mut_ptr() as usize;

        // SAFETY: committed write-watched page of ours.
        unsafe { ((watched + 4) as *mut u8).write(3) };
        let _ask = AskOnceGuard;
        ask_once(&[
            (watched, watched + 2 * PAGE),
            (plain_start, plain_start + PAGE),
        ]);
        let (now, found) = writes(watched, watched + 2 * PAGE, false).expect("watched range");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, watched / PAGE);
        assert!(found[0].1 > 0 && found[0].1 <= now);
        assert!(writes(plain_start, plain_start + PAGE, false).is_none());
    }

    #[test]
    fn guard_clean_fast_path_returns_empty_and_still_detects_faults() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        // SAFETY: a 4-block pagefile section mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (4 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let v = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 4 * BLOCK).Value as usize;
            CloseHandle(section);
            v
        };
        assert_ne!(view, 0);
        let initial = guard(view, view + 4 * BLOCK).expect("guarded");
        assert_eq!(initial.len(), 4);
        // All blocks clean -> fast path returns empty without re-protecting.
        assert_eq!(guard(view, view + 4 * BLOCK), Some(Vec::new()));
        // Write block 2 via fault handler -> only block 2 reported dirty.
        // SAFETY: mapped view memory; our VEH opens the block on write.
        unsafe { ((view + 2 * BLOCK + 16) as *mut u8).write(0x55) };
        assert_eq!(guard(view, view + 4 * BLOCK), Some(vec![view / PAGE + 2]));
        // Clean again immediately after re-guarding.
        assert_eq!(guard(view, view + 4 * BLOCK), Some(Vec::new()));
        title_memory_remapped(view, 4 * BLOCK);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
        }
    }

    /// Re-entrant calls to `with_watch_scratch` on the same thread (e.g. a VEH
    /// fault landing while an outer watch query is active) must never panic on
    /// `RefCell already borrowed`.
    #[test]
    fn with_watch_scratch_is_reentrant_safe() {
        let outer = with_watch_scratch(32, |outer_buf| {
            outer_buf[0] = 0x1111;
            let inner = with_watch_scratch(128, |inner_buf| {
                inner_buf[127] = 0x2222;
                inner_buf.len()
            });
            assert_eq!(outer_buf[0], 0x1111);
            (outer_buf.len(), inner)
        });
        assert!(outer.0 >= 32);
        assert!(outer.1 >= 128);
    }

    #[test]
    fn cold_promotion_skips_readonly_and_mixed_protection_blocks() {
        use windows_sys::Win32::System::Memory::{
            MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READONLY, VirtualProtect, VirtualQuery,
        };
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let threshold = cold_after();
        if threshold == 0 {
            return;
        }
        let alloc = WatchedAllocGuard::new(3);
        let base = alloc.base;
        let mut old = 0u32;

        // Block 0: all PAGE_READONLY; Block 1: mixed PAGE_READWRITE + PAGE_NOACCESS;
        // Block 2: ordinary all-PAGE_READWRITE control block.
        // SAFETY: committed pages owned by `alloc`.
        unsafe {
            assert_ne!(
                VirtualProtect(base as *const _, PAGE, PAGE_READONLY, &mut old),
                0
            );
            assert_ne!(
                VirtualProtect(
                    (base + PAGE + KERNEL_PAGE) as *const _,
                    KERNEL_PAGE,
                    PAGE_NOACCESS,
                    &mut old,
                ),
                0
            );
        }

        // Baseline reset and `threshold` clean frames.
        assert!(writes(base, base + 3 * PAGE, false).is_some());
        let mut token = 0u64;
        for _ in 0..threshold {
            let _ = crate::frame::wait(std::ptr::null(), 0, 0, std::ptr::null(), 2, &mut token);
            assert!(writes(base, base + 3 * PAGE, false).is_some());
        }

        assert!(
            !DIR.is_cold_ro(base / PAGE),
            "read-only watched block must not be cold-promoted"
        );
        assert!(
            !DIR.is_cold_ro(base / PAGE + 1),
            "mixed-protection watched block must not be cold-promoted"
        );
        assert!(
            DIR.is_cold_ro(base / PAGE + 2),
            "plain PAGE_READWRITE block should still promote"
        );

        let query_prot = |at: usize| {
            let mut info = MEMORY_BASIC_INFORMATION::default();
            // SAFETY: metadata query on `alloc`.
            let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
            assert_ne!(n, 0);
            info.Protect
        };
        assert_eq!(query_prot(base), PAGE_READONLY);
        assert_eq!(query_prot(base + PAGE), PAGE_READWRITE);
        assert_eq!(query_prot(base + PAGE + KERNEL_PAGE), PAGE_NOACCESS);
    }

    #[test]
    fn title_memory_remapped_invalidates_ahead_coverage() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let alloc = WatchedAllocGuard::new(2);
        let base = alloc.base;
        assert!(writes(base, base + 2 * PAGE, false).is_some());

        let _ask = AskOnceGuard;
        ask_once(&[(base, base + 2 * PAGE)]);
        let (before, _) = asked(base, base + PAGE).expect("covered by ask_once");

        title_memory_remapped(base, PAGE);
        assert!(
            asked(base, base + PAGE).is_none(),
            "remap must drop overlapping ahead coverage"
        );

        // SAFETY: committed write-watched page owned by `alloc`.
        unsafe { ((base + 16) as *mut u8).write_volatile(0x33) };
        let (after, found) = writes(base, base + PAGE, false).expect("watched");
        assert!(after > before);
        assert!(
            found.iter().any(|&(p, w)| p == base / PAGE && w > before),
            "post-remap write must be observed immediately"
        );
    }

    #[test]
    fn writes_preserves_reset_dirty_watched_block_when_later_run_fails() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFileEx,
            PAGE_NOACCESS, UnmapViewOfFile,
        };
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Build adjacent [2-page MEM_WRITE_WATCH | 1-page mapped view].
        // SAFETY: 1-page pagefile section.
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());

        let mut pair = None;
        for _ in 0..32 {
            // SAFETY: probe for a free 3-page region.
            let probe =
                unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) }
                    as usize;
            assert_ne!(probe, 0);
            // SAFETY: release probe immediately.
            assert_ne!(unsafe { VirtualFree(probe as *mut _, 0, MEM_RELEASE) }, 0);
            // SAFETY: allocate 2 write-watched pages at `probe`.
            let watched = unsafe {
                VirtualAlloc(
                    probe as *const _,
                    2 * PAGE,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            } as usize;
            if watched != probe {
                if watched != 0 {
                    // SAFETY: release our allocation.
                    unsafe { VirtualFree(watched as *mut _, 0, MEM_RELEASE) };
                }
                continue;
            }
            // SAFETY: map 1-page section immediately after `watched`.
            let view = unsafe {
                MapViewOfFileEx(
                    section,
                    FILE_MAP_WRITE,
                    0,
                    0,
                    PAGE,
                    (probe + 2 * PAGE) as *const _,
                )
                .Value as usize
            };
            if view == probe + 2 * PAGE {
                pair = Some((watched, view));
                break;
            }
            if view != 0 {
                // SAFETY: unmap our view.
                unsafe {
                    UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: view as *mut _,
                    });
                }
            }
            // SAFETY: release our allocation.
            unsafe { VirtualFree(watched as *mut _, 0, MEM_RELEASE) };
        }
        let (watched, view) = pair.expect("allocate adjacent watched pages + mapped view");
        DIR.clear_written_range(watched / PAGE, watched / PAGE + 3);

        // Baseline clean; if cold promotion is active, promote block 1 to ColdRo
        // so `[watched..view+PAGE)` splits into two non-cold runs around it.
        assert!(writes(watched, watched + 2 * PAGE, false).is_some());
        let threshold = cold_after();
        if threshold > 0 {
            let mut token = 0u64;
            for _ in 0..threshold {
                let _ = crate::frame::wait(std::ptr::null(), 0, 0, std::ptr::null(), 2, &mut token);
                assert!(writes(watched + PAGE, watched + 2 * PAGE, false).is_some());
            }
            assert!(DIR.is_cold_ro(watched / PAGE + 1));
        }

        let (before, _) = writes(watched, watched + PAGE, false).expect("watched baseline");
        // Dirty block 0 (non-cold watched), then query across `[watched, view + PAGE)`
        // which resets block 0 and then fails on the mapped tail.
        // SAFETY: committed write-watched page owned by this test.
        unsafe { ((watched + 8) as *mut u8).write_volatile(0x5a) };
        assert!(
            writes(watched, view + PAGE, false).is_none(),
            "range ending in mapped view is not write-watched"
        );

        // A follow-up query for just the watched block MUST still see the write
        // whose kernel bit was already reset above.
        let (after, found) = writes(watched, watched + PAGE, false).expect("watched");
        assert!(after > before);
        assert!(
            found
                .iter()
                .any(|&(p, w)| p == watched / PAGE && w > before),
            "dirty watched block reset before a failing run must not be lost"
        );

        // Also verify `ask_once` over a spanning `[watched | unguarded mapped]`
        // range skips the mapped view cleanly and still covers the private part.
        // SAFETY: committed write-watched page owned by this test.
        unsafe { ((watched + 12) as *mut u8).write_volatile(0x6b) };
        let _ask = AskOnceGuard;
        ask_once(&[(watched, view + PAGE)]);
        let (ask_now, ask_found) =
            asked(watched, watched + PAGE).expect("private prefix covered by ask_once");
        assert!(
            ask_found
                .iter()
                .any(|&(p, w)| p == watched / PAGE && w == ask_now),
            "private write must be reported even when spanning an unguarded mapped view"
        );
        assert!(
            asked(view, view + PAGE).is_none(),
            "unguarded mapped view must not be claimed as ahead-covered"
        );
        drop(_ask);

        title_memory_remapped(watched, 3 * PAGE);
        DIR.clear_written_range(watched / PAGE, watched / PAGE + 3);
        // SAFETY: clean up our view, section, and watched allocation.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
            CloseHandle(section);
            VirtualFree(watched as *mut _, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn mapped_pages_written_concurrent_readers_never_skip_alias_write() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        const SIZE: usize = 2 * BLOCK;
        // SAFETY: 2-block pagefile section mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let v = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize;
            CloseHandle(section);
            v
        };
        assert_ne!(view, 0);
        DIR.clear_written_range(view / PAGE, view / PAGE + 2);
        let _ = writes(view, view + SIZE, true).expect("guard view");

        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let mut readers = Vec::new();
        for _ in 0..2 {
            let stop = stop.clone();
            readers.push(std::thread::spawn(move || {
                let mut last_watch = 0u64;
                while !stop.load(Ordering::Acquire) {
                    let (now, list) = writes(view, view + SIZE, true).expect("guarded");
                    for &(_, w) in &list {
                        assert!(w <= now, "returned watermark must cover all reported pages");
                    }
                    last_watch = now;
                }
                last_watch
            }));
        }

        for _ in 0..500 {
            mapped_pages_written([view, view + BLOCK].into_iter());
            std::thread::yield_now();
        }
        stop.store(true, Ordering::Release);
        let saved: Vec<u64> = readers.into_iter().map(|h| h.join().unwrap()).collect();

        // Publish one final alias write after all concurrent readers stopped,
        // and verify every reader's saved watermark is strictly below it.
        mapped_pages_written(std::iter::once(view + BLOCK));
        let (final_now, final_list) = writes(view, view + SIZE, true).expect("final guarded");
        let page1_epoch = final_list
            .iter()
            .find(|&&(p, _)| p == view / PAGE + 1)
            .map(|&(_, w)| w)
            .expect("page 1 written");
        for prev in saved {
            assert!(
                page1_epoch > prev && page1_epoch <= final_now,
                "saved watermark {prev} must not skip subsequent alias write {page1_epoch}"
            );
        }

        title_memory_remapped(view, SIZE);
        DIR.clear_written_range(view / PAGE, view / PAGE + 2);
        // SAFETY: unmap view owned by this test.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
        }
    }

    #[test]
    fn alias_rebound_marks_guarded_blocks_unclean_without_unprotecting() {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_ALL_ACCESS, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };
        // SAFETY: isolated 2-block pagefile-backed view.
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: maps our section.
        let view =
            unsafe { MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, 2 * BLOCK) }.Value as usize;
        assert_ne!(view, 0);

        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some(vec![view / PAGE, view / PAGE + 1])
        );
        assert_eq!(guard(view, view + 2 * BLOCK), Some(Vec::new()));

        // Identical alias rebind on block 0 marks only block 0 dirty while
        // keeping its protection intact.
        title_memory_alias_rebound(view, BLOCK);
        assert_eq!(guard(view, view + 2 * BLOCK), Some(vec![view / PAGE]));
        assert_eq!(guard(view, view + 2 * BLOCK), Some(Vec::new()));

        // SAFETY: clean up view and section.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(view, 2 * BLOCK);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{
        PAGE, by_range, title_memory_reading, title_memory_reading_now, title_memory_remapping,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// Addresses only these tests hold (nothing is read there).
    const HERE: usize = 0x7e00_0000_0000;

    /// A change of mapping waits for a read in place of its bytes that is
    /// under way, and such a read waits for the change.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading(HERE + PAGE / 2, PAGE);
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping(HERE + PAGE, PAGE);
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !changed.load(Ordering::SeqCst),
            "the mapping changed under a read in place"
        );
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping(HERE, 2 * PAGE);
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading(HERE + PAGE + 16, 16);
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !read.load(Ordering::SeqCst),
            "a read in place began during a change of mapping"
        );
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }

    /// Reads and changes whose bytes do not meet do not wait for each other,
    /// and a change waiting for a read keeps new reads of its bytes out (but
    /// not of others).
    #[test]
    fn reads_and_remappings_of_other_bytes_go_on() {
        if !by_range() {
            return; // VICTORY_MAPPING_RANGES=0: one lock over all of it
        }
        let at = HERE + 16 * PAGE;
        let reading = title_memory_reading(at, PAGE);
        // Another page changes while the read is under way, and another
        // read goes on while that change is.
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            let changing = title_memory_remapping(at + PAGE, PAGE);
            let (read, was_read) = mpsc::channel();
            std::thread::spawn(move || {
                let _reading = title_memory_reading(at + 2 * PAGE, PAGE);
                read.send(()).unwrap();
            });
            let other = was_read.recv_timeout(Duration::from_secs(10)).is_ok();
            drop(changing);
            done.send(other).unwrap();
        });
        assert_eq!(
            finished.recv_timeout(Duration::from_secs(10)),
            Ok(true),
            "kept apart though their bytes do not meet"
        );
        // A change of the read's page waits, and keeps a new read of its
        // bytes out meanwhile; a read elsewhere is not kept out.
        let changed = Arc::new(AtomicBool::new(false));
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                let _changing = title_memory_remapping(at, PAGE);
                changed.store(true, Ordering::SeqCst);
            })
        };
        while !super::spans().changes.iter().any(|c| c.1 == at) {
            std::thread::yield_now();
        }
        assert!(
            title_memory_reading_now(at + 8, 8).is_none(),
            "a read began over a change that waits"
        );
        assert!(title_memory_reading_now(at + 3 * PAGE, 8).is_some());
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !changed.load(Ordering::SeqCst),
            "the mapping changed under a read in place"
        );
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
    }

    /// Zero-length spans never overlap anything and ranges past `usize::MAX`
    /// saturate without overflowing.
    #[test]
    fn empty_and_saturating_ranges_are_safe() {
        assert!(!super::meet(HERE, HERE + PAGE, HERE + 8, HERE + 8));
        assert!(!super::meet(HERE + 8, HERE + 8, HERE, HERE + PAGE));
        assert!(super::meet(0, usize::MAX, HERE, HERE + PAGE));
        if !by_range() {
            return;
        }
        let at = HERE + 32 * PAGE;
        let changing = title_memory_remapping(at, PAGE);
        assert!(title_memory_reading_now(at + 8, 0).is_some());
        drop(changing);

        let empty = title_memory_remapping(at + 8, 0);
        assert!(title_memory_reading_now(at, PAGE).is_some());
        drop(empty);

        let top = title_memory_remapping(usize::MAX - 8, PAGE);
        assert!(title_memory_reading_now(usize::MAX - 4, 16).is_none());
        drop(top);
    }
}
