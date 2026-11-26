//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a lock-free
//! 3-level radix page directory ([`PageDir`]) recording each page's last
//! write, numbered by a monotonic asking epoch ([`ASK_EPOCH`]).
//!
//! Write-watched pages that remain clean across consecutive frames promote to
//! `PAGE_READONLY` (`VICTORY_WATCH_COLD`, default 6 under `VICTORY_WATCH_V2=1`),
//! so repeated sweeps skip `GetWriteWatch` completely until a CPU write faults
//! into the vectored handler and demotes the page back to `PAGE_READWRITE`.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. Guard metadata and `VirtualProtect` transitions are
//! sharded across 128 cache-line-aligned [`GUARD_SHARDS_TABLE`] stripes so
//! disjoint buffers and title write faults never contend. The runtime tells us
//! when it maps or unmaps views ([`title_memory_remapped`]), which resets
//! their protection.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are reported through
//! [`mapped_pages_written`] or go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! unless announced via [`kernel_will_write`].

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

/// Master switch for the watch v2 redesign (lock-free `PageDir` fast paths,
/// flat cold-page `PAGE_READONLY` promotion, and adaptive contiguous
/// `VirtualProtect` coalescing). Set `VICTORY_WATCH_V2=0` to opt out cleanly.
fn v2_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WATCH_V2").is_none_or(|v| v != "0"))
}

/// Whether adaptive contiguous `VirtualProtect` coalescing is enabled
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
/// `VICTORY_WATCH_V2=1`; `0` disables cold promotion, and `VICTORY_WATCH_V2=0`
/// defaults to `0`).
fn cold_after() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        let default = if v2_enabled() { DEFAULT_COLD_FRAMES } else { 0 };
        std::env::var("VICTORY_WATCH_COLD")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(default)
    })
}

#[inline]
fn current_frame32() -> u32 {
    (crate::frame::current() as u32).wrapping_add(1).max(1)
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
}

struct LeafChunk {
    written_bits: [AtomicU64; CHUNK_WORDS],
    written_epoch: [AtomicU64; CHUNK_PAGES],
    watched_bits: [AtomicU64; CHUNK_WORDS],
    cold_ro_bits: [AtomicU64; CHUNK_WORDS],
    cold_streak: [AtomicU8; CHUNK_PAGES],
    cold_last_frame: [AtomicU32; CHUNK_PAGES],
    guard_protected_bits: [AtomicU64; CHUNK_WORDS],
    guard_clean_bits: [AtomicU64; CHUNK_WORDS],
    guard_absent_bits: [AtomicU64; CHUNK_WORDS],
}

impl LeafChunk {
    fn new() -> Self {
        Self {
            written_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
            written_epoch: [const { AtomicU64::new(0) }; CHUNK_PAGES],
            watched_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
            cold_ro_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
            cold_streak: [const { AtomicU8::new(0) }; CHUNK_PAGES],
            cold_last_frame: [const { AtomicU32::new(0) }; CHUNK_PAGES],
            guard_protected_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
            guard_clean_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
            guard_absent_bits: [const { AtomicU64::new(0) }; CHUNK_WORDS],
        }
    }

    #[inline]
    fn bits(&self, field: BitField) -> &[AtomicU64; CHUNK_WORDS] {
        match field {
            BitField::Watched => &self.watched_bits,
            BitField::ColdRo => &self.cold_ro_bits,
            BitField::GuardProtected => &self.guard_protected_bits,
            BitField::GuardClean => &self.guard_clean_bits,
            BitField::GuardAbsent => &self.guard_absent_bits,
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
        if epoch == 0 {
            return;
        }
        if let Some((leaf, idx)) = self.get_or_alloc(page) {
            leaf.written_epoch[idx].fetch_max(epoch, Ordering::AcqRel);
            leaf.written_bits[idx / 64].fetch_or(1u64 << (idx % 64), Ordering::Release);
        }
    }

    #[allow(dead_code)]
    #[inline]
    fn record_written_prev(&self, page: usize, epoch: u64) -> u64 {
        let Some((leaf, idx)) = self.get_or_alloc(page) else {
            return 0;
        };
        let prev = leaf.written_epoch[idx].fetch_max(epoch, Ordering::AcqRel);
        leaf.written_bits[idx / 64].fetch_or(1u64 << (idx % 64), Ordering::Release);
        prev
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
    fn is_cold_ro(&self, block: usize) -> bool {
        self.get_bit(block, BitField::ColdRo, Ordering::SeqCst)
    }

    #[inline]
    fn is_guard_clean(&self, block: usize) -> bool {
        self.get_bit(block, BitField::GuardClean, Ordering::SeqCst)
    }

    #[inline]
    fn is_guard_protected(&self, block: usize) -> bool {
        self.get_bit(block, BitField::GuardProtected, Ordering::SeqCst)
    }

    #[inline]
    fn is_guard_absent(&self, block: usize) -> bool {
        self.get_bit(block, BitField::GuardAbsent, Ordering::SeqCst)
    }

    #[inline]
    fn any_guard_flag(&self, block: usize) -> bool {
        let Some((leaf, idx)) = self.get(block) else {
            return false;
        };
        let (w, bit) = (idx / 64, 1u64 << (idx % 64));
        let combined = leaf.guard_protected_bits[w].load(Ordering::Acquire)
            | leaf.guard_absent_bits[w].load(Ordering::Acquire);
        combined & bit != 0
    }

    fn all_guard_clean(&self, first: usize, last: usize) -> bool {
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
                if leaf.guard_clean_bits[w].load(Ordering::SeqCst) & mask != mask {
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

    /// Collects contiguous sub-runs of `[first, last)` where `field` bits are 0.
    fn collect_clear_runs(
        &self,
        first: usize,
        last: usize,
        field: BitField,
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
            let words = leaf.bits(field);
            let lo_idx = cur - chunk_page;
            let hi_idx = next_chunk - chunk_page;
            for_each_word_in_chunk(lo_idx, hi_idx, |w, mask| {
                let val = words[w].load(Ordering::SeqCst);
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
            for idx in lo_idx..hi_idx {
                leaf.written_epoch[idx].store(0, Ordering::SeqCst);
            }
            true
        });
    }

    fn note_clean_for_cold(
        &self,
        first: usize,
        last: usize,
        dirty_sorted: &[usize],
        frame32: u32,
        threshold: u8,
        out_promote: &mut Vec<usize>,
    ) {
        if first >= last {
            return;
        }
        let mut dirty_idx = dirty_sorted.partition_point(|&p| p < first);
        self.for_each_chunk(first, last, true, |chunk_page, leaf, lo_idx, hi_idx| {
            for_each_word_in_chunk(lo_idx, hi_idx, |w, mask| {
                leaf.watched_bits[w].fetch_or(mask, Ordering::Release);
                true
            });
            for idx in lo_idx..hi_idx {
                let page = chunk_page + idx;
                while dirty_idx < dirty_sorted.len() && dirty_sorted[dirty_idx] < page {
                    dirty_idx += 1;
                }
                if dirty_idx < dirty_sorted.len() && dirty_sorted[dirty_idx] == page {
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
    /// Sorted, merged page ranges `[first, last)` asked ahead for the current
    /// submission ([`ask_once`]).
    ranges: Vec<(usize, usize)>,
    epoch: u64,
    kept: bool,
}

impl AheadState {
    /// Refreshes overlapping an announced file read must ask the kernel
    /// again, even when a previous query covered their pages.
    fn forget_ahead(&mut self, (first, last): (usize, usize)) {
        const _: () = assert!(BLOCK == PAGE);
        self.ranges.retain(|&(a, b)| b <= first || last <= a);
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
/// `ask_once` skip `AHEAD.read()` completely.
static AHEAD_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Fast lock-free mirror of `AheadState::kept`.
static ASK_KEPT: AtomicBool = AtomicBool::new(false);

/// Whether a submission-wide write-watch asking (`ask_once`) currently stands.
#[inline]
pub(crate) fn ask_kept() -> bool {
    ASK_KEPT.load(Ordering::Acquire)
}

#[inline]
fn set_ahead_flags(kept: bool, active: bool) {
    AHEAD_ACTIVE.store(active, Ordering::Release);
    ASK_KEPT.store(kept, Ordering::Release);
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

/// Calls `GetWriteWatch(WRITE_WATCH_FLAG_RESET)` on `[first, last)` and
/// returns `Some(dirty_pages_sorted_deduped)` on success or `None` on failure.
fn call_get_write_watch(first: usize, last: usize) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    if first >= last {
        return Some(Vec::new());
    }
    let needed = (last - first) * PAGE / KERNEL_PAGE;
    with_watch_scratch(needed, |buf| {
        let mut count = needed;
        let mut granularity = 0;
        #[cfg(feature = "diagnostics")]
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
        if failed {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
        }
        let mut dirty: Vec<usize> = buf[..count].iter().map(|&at| at / PAGE).collect();
        dirty.sort_unstable();
        dirty.dedup();
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

    // Lock order: `WATCH_SERIAL` -> `GUARD_SHARDS_TABLE` (ascending). Holding
    // both across `VirtualProtect` + `GetWriteWatch` verification ensures
    // `ColdRo` is never visible until verified clean, and `on_write` (which
    // acquires only the faulted block's shard without `WATCH_SERIAL`)
    // serializes cleanly if a title thread faults before the shard lock drops.
    let _serial = WATCH_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let locked = GuardLockSet::lock_blocks(&sorted);

    let mut eligible = Vec::with_capacity(sorted.len());
    let mut rw_cache = (0usize, 0usize, false);
    for block in sorted {
        if DIR.is_cold_ro(block)
            || DIR.any_guard_flag(block)
            || DIR.cold_streak(block) < threshold_u8
            || locked.announced(block)
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
        if let Some(dirty) = call_get_write_watch(from, to) {
            if dirty.is_empty() {
                DIR.set_bits_range(from, to, BitField::ColdRo, true, Ordering::SeqCst);
                return;
            }
            let _now = DIR.publish_written(&dirty);
            let mut d_idx = dirty.partition_point(|&b| b < from);
            for block in from..to {
                while d_idx < dirty.len() && dirty[d_idx] < block {
                    d_idx += 1;
                }
                if d_idx < dirty.len() && dirty[d_idx] == block {
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
/// walking the pages between them costs less than another `GetWriteWatch`
/// reset call.
const AHEAD_GAP: usize = (8 << 20) / PAGE;

/// [`AHEAD_GAP`], or `VICTORY_WATCH_GAP_MB=<mb>`.
fn ahead_gap() -> usize {
    static GAP: OnceLock<usize> = OnceLock::new();
    *GAP.get_or_init(|| {
        std::env::var("VICTORY_WATCH_GAP_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(AHEAD_GAP, |mb| (mb << 20) / PAGE)
    })
}

/// Maximum span of a single coalesced `GetWriteWatch` asking (1 GiB).
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
    batch_dirty: &mut Vec<usize>,
    covered: &mut Vec<(usize, usize)>,
    to_promote: &mut Vec<usize>,
) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    if first >= last {
        return;
    }
    if let Some(dirty) = call_get_write_watch(first, last) {
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
                    && let Some(dirty) = call_get_write_watch(p_first, p_last)
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
                        if let Some(dirty) = call_get_write_watch(sp0, sp1) {
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
/// any `ColdRo` sub-runs (which are already protected via `PAGE_READONLY` VEH).
fn ask_merged_interval(
    first: usize,
    last: usize,
    frame32: u32,
    threshold: u8,
    batch_dirty: &mut Vec<usize>,
    covered: &mut Vec<(usize, usize)>,
    to_promote: &mut Vec<usize>,
) {
    if first >= last {
        return;
    }
    if v2_enabled() && threshold > 0 {
        let non_cold = DIR.collect_clear_runs(first, last, BitField::ColdRo);
        if non_cold.is_empty() {
            push_merged_interval(covered, first, last);
            return;
        }
        let non_cold_pages: usize = non_cold.iter().map(|&(a, b)| b - a).sum();
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
fn collect_and_ask_batch(ranges: &[(usize, usize)]) -> (u64, Vec<(usize, usize)>) {
    let mut pieces: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if known_view(start) {
            continue;
        }
        pieces.push((first, last));
    }
    if pieces.is_empty() {
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
            } else if let Some(dirty) = call_get_write_watch(group_first, group_last) {
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
        batch_dirty.sort_unstable();
        batch_dirty.dedup();
        if batch_dirty.is_empty() {
            next_epoch()
        } else {
            DIR.publish_written(&batch_dirty)
        }
    };

    if !to_promote.is_empty() && threshold > 0 {
        promote_cold_pages(&to_promote);
    }
    (now, covered)
}

/// Whether per-submission write-watch coalescing (`ask_once`) is enabled
/// (`VICTORY_ASK_ONCE=0` disables it; on by default).
fn once() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// Coalesces write-watch queries for all `(start, end)` ranges of a submission
/// into a small number of `GetWriteWatch` calls so repeated flushes within the
/// submission answer from the lock-free [`PageDir`] until [`ask_once_end`].
/// Ranges that are not single-allocation write-watched memory are left to
/// [`writes`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() || ranges.is_empty() {
        return;
    }
    ask_once_end();
    // Snapshot active announcements and the monotonic announcement sequence
    // before querying the kernel. Any read already in flight may land while
    // `GetWriteWatch` runs, and any newly announced read increments
    // `ANNOUNCED_SEQ` before updating `AHEAD`.
    let seq_before = ANNOUNCED_SEQ.load(Ordering::SeqCst);
    let pending_before = snapshot_announced();

    let (now, mut covered) = collect_and_ask_batch(ranges);
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

/// Ends the active per-submission coalesced asking so later [`writes`] calls
/// query the kernel again.
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

/// Notes that our submission wrote title memory `[at, at + len)` while a
/// coalesced asking stands, so refreshes recorded after the copy see those
/// pages as written without ending coalescing for the rest of the submission.
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

/// Records a monotonic write epoch for `page` and any alias `others`, returning
/// `(previous_epoch_of_page, new_epoch)`. Kept as a clean extension point for
/// own-view writes that bypass `GetWriteWatch` and page guards.
#[allow(dead_code)]
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

/// What [`writes`] would say of `[start, end)` from [`PageDir`] alone, if
/// the range lies inside one asked ahead ([`ask_once`]).
#[allow(dead_code)]
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

    // Fast path 1: covered by `ask_once` — lock-free `PageDir` read.
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

    // Fast path 2: known guarded view memory goes straight to guards without
    // incurring a failing `GetWriteWatch` syscall.
    if guarded && known_view(first * PAGE) {
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

    // Path 3: write-watched memory (skipping `ColdRo` sub-runs when v2 cold
    // promotion is enabled), falling back to `by_allocation` for multi-allocation
    // or uncommitted reservations, then to `guard` for mapped views.
    let threshold = cold_after().min(u8::MAX as u32) as u8;
    let frame32 = current_frame32();
    let mut to_promote = Vec::new();
    let result: Option<u64> = {
        let _serial = WATCH_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let non_cold = if v2_enabled() && threshold > 0 {
            DIR.collect_clear_runs(first, last, BitField::ColdRo)
        } else {
            vec![(first, last)]
        };
        if non_cold.is_empty() {
            Some(next_epoch())
        } else {
            let mut last_epoch = 0u64;
            let mut failed_from: Option<usize> = None;
            for (idx, &(r0, r1)) in non_cold.iter().enumerate() {
                if let Some(dirty) = call_get_write_watch(r0, r1) {
                    // Record each successful reset in `DIR` immediately under
                    // `WATCH_SERIAL` before querying the next run so a later
                    // failure can never drop an already-reset write bit.
                    if !dirty.is_empty() {
                        last_epoch = DIR.publish_written(&dirty);
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
                        last_epoch = DIR.publish_written(&pages);
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
                    None
                }
            } else {
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

/// The pages of `[start, end)` written, asked allocation by allocation:
/// the kernel refuses a range over several (a title buffer spans as many
/// allocations as the runtime made for it). Parts never committed cannot
/// have been written, so a range none of which is committed (a reserved
/// allocation the title has not committed yet) has no writes. Returns
/// `(all_watched, reset_dirty_pages)` so even when a later allocation is not
/// write-watched (`all_watched == false`), any dirty pages already reset by
/// earlier allocations are returned and recorded rather than lost.
fn by_allocation(start: usize, end: usize) -> (bool, Vec<usize>) {
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
            pages.sort_unstable();
            pages.dedup();
            return (false, pages);
        };
        // A committed section view cannot belong to a MEM_WRITE_WATCH
        // reservation. Its other protection regions need no further queries.
        if info.State == MEM_COMMIT && info.Type == MEM_MAPPED {
            pages.sort_unstable();
            pages.dedup();
            return (false, pages);
        }
        let base = info.AllocationBase as usize;
        let mut committed = info.State == MEM_COMMIT;
        let mut to = (info.BaseAddress as usize + info.RegionSize).min(end);
        // The rest of the allocation's regions in the range.
        while info.State != MEM_FREE && to < end {
            let Some(next) = query(to) else {
                pages.sort_unstable();
                pages.dedup();
                return (false, pages);
            };
            if next.State == MEM_FREE || next.AllocationBase as usize != base {
                break;
            }
            committed |= next.State == MEM_COMMIT;
            to = (next.BaseAddress as usize + next.RegionSize).min(end);
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
                pages.extend(found[..count].iter().map(|&a| a / PAGE));
                true
            });
            if !ok {
                pages.sort_unstable();
                pages.dedup();
                return (false, pages);
            }
        }
        at = to;
    }
    pages.sort_unstable();
    pages.dedup();
    (allocations > 1 || !any, pages)
}

// ---------------------------------------------------------------------------
// 128-Shard Cache-Line-Aligned Guard Table
// ---------------------------------------------------------------------------

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

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
    /// Announced kernel writes still to land overlapping this shard, as block
    /// ranges `[first, end)` with how many announcements of that range are
    /// open: their blocks are left open until they have landed, however many
    /// refreshes come first. Kept as ranges, not per block, so a title-supplied
    /// length (saturated by `fread`'s size times count) costs O(touched shards)
    /// (at most 128) however large; counted, so a range announced again and
    /// again stays one entry per touched shard.
    announced: BTreeMap<(usize, usize), u32>,
}

impl GuardShard {
    /// Whether an announced kernel write still to land covers `block`.
    fn announced(&self, block: usize) -> bool {
        self.announced
            .keys()
            .any(|&(first, end)| (first..end).contains(&block))
    }
}

/// Cache-line padded shard slot to prevent false sharing between adjacent shards.
#[repr(align(64))]
struct GuardShardSlot(RwLock<GuardShard>);

/// Sharded guard metadata table.
///
/// Global lock order (strictly acyclic):
/// `WATCH_SERIAL` -> `GUARD_SHARDS_TABLE[0..GUARD_SHARDS]` (strictly ascending
/// shard index) -> `AHEAD` (leaf lock).
static GUARD_SHARDS_TABLE: [GuardShardSlot; GUARD_SHARDS] = [const {
    GuardShardSlot(RwLock::new(GuardShard {
        announced: BTreeMap::new(),
    }))
}; GUARD_SHARDS];

/// 128-bit atomic summary of which guard shards currently have non-empty
/// `announced` maps, allowing `ask_once` to skip shard reads completely when
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
    fn announced(&self, block: usize) -> bool;
    fn debug_assert_locked(&self, block: usize);
}

#[inline]
fn debug_assert_range_locked(access: &impl ShardAccess, first: usize, last: usize) {
    #[cfg(debug_assertions)]
    for_each_shard_run(first, last, |_, lo, _| access.debug_assert_locked(lo));
    #[cfg(not(debug_assertions))]
    let _ = (access, first, last);
}

struct SingleShard<'a> {
    shard: usize,
    guard: &'a GuardShard,
}

impl ShardAccess for SingleShard<'_> {
    #[inline]
    fn announced(&self, block: usize) -> bool {
        debug_assert_eq!(shard_of(block), self.shard);
        self.guard.announced(block)
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
    fn announced(&self, block: usize) -> bool {
        self.shard(shard_of(block)).announced(block)
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

/// Lock-free `GuardClean` bitmap fast path; `VICTORY_GUARD_BITS=0` (or
/// `VICTORY_WATCH_LOCKFREE=0` / `VICTORY_WATCH_V2=0`) takes the shard lock on
/// every guard check.
fn guard_bits_fast_enabled() -> bool {
    static BITS: OnceLock<bool> = OnceLock::new();
    lockfree_enabled()
        && *BITS.get_or_init(|| std::env::var_os("VICTORY_GUARD_BITS").is_none_or(|v| v != "0"))
}

/// Whether skipping kernel write-watch / `has_mapped` checks for already
/// guarded views is disabled (`VICTORY_GUARD_ASK=1`).
fn guard_ask_forced() -> bool {
    static ASK: OnceLock<bool> = OnceLock::new();
    *ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"))
}

/// Whether the block holding `start` is committed view memory the guards
/// already hold read-only. (`GuardAbsent` blocks are NOT treated as known
/// views so a reserved placeholder later committed as private memory is still
/// re-checked by [`has_mapped`].)
fn known_view(start: usize) -> bool {
    if guard_ask_forced() {
        return false;
    }
    let first = start / BLOCK;
    if lockfree_enabled() {
        return DIR.is_guard_protected(first);
    }
    let _g = guard_shard_read(shard_of(first));
    DIR.is_guard_protected(first)
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
    // range is clean, skip both the kernel query and shard locks.
    if guard_bits_fast_enabled() && DIR.all_guard_clean(first, last) {
        return Some((0, Vec::new()));
    }
    if !known_view(start) && !has_mapped(start, end) {
        return None;
    }
    if !install() {
        return None;
    }
    let mut written = Vec::new();
    let mut max_epoch = 0u64;
    let mut dirty_shards = 0usize;
    if guard_bits_fast_enabled() {
        let runs = DIR.collect_clear_runs(first, last, BitField::GuardClean);
        if runs.is_empty() {
            return Some((0, written));
        }
        // Split each run at guard-shard boundaries so each pass locks only 1
        // shard and unrelated workers / title write faults never contend.
        for (from, to) in runs {
            for_each_shard_run(from, to, |shard, s_from, s_to| {
                let ep = guard_run_shard(shard, s_from, s_to, &mut written);
                if ep != 0 {
                    max_epoch = max_epoch.max(ep);
                    dirty_shards += 1;
                }
            });
        }
    } else {
        for_each_shard_run(first, last, |shard, s_from, s_to| {
            let ep = guard_run_shard(shard, s_from, s_to, &mut written);
            if ep != 0 {
                max_epoch = max_epoch.max(ep);
                dirty_shards += 1;
            }
        });
    }
    if dirty_shards > 1 {
        // Each dirty shard already published its pages before marking them
        // clean, so concurrent readers could never miss a write; unify all
        // pages reported by this call under one asking epoch.
        max_epoch = DIR.publish_written(&written);
    }
    Some((max_epoch, written))
}

/// Test-facing wrapper around [`guard_and_publish`] returning only the dirty
/// pages of `[start, end)`.
#[cfg(test)]
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    guard_and_publish(start, end).map(|(_, written)| written)
}

/// [`guard_and_publish`] for a shard-local sub-run `first..last` where every
/// block maps to `shard`. Locks only `shard`, publishes dirty records into
/// [`DIR`] strictly before making blocks visible as clean/protected/absent,
/// and returns the published epoch (or `0` when all blocks were already clean).
fn guard_run_shard(shard: usize, first: usize, last: usize, written: &mut Vec<usize>) -> u64 {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;
    if first >= last {
        return 0;
    }
    let g = guard_shard(shard);
    let access = SingleShard { shard, guard: &g };
    let mut run: Option<usize> = None;
    let mut uncommitted_until = first;
    let mut shard_written: Vec<usize> = Vec::new();
    let mut alias_clean: Vec<usize> = Vec::new();
    let mut newly_protected: Vec<(usize, usize)> = Vec::new();
    let mut newly_absent: Vec<(usize, usize)> = Vec::new();
    for block in first..=last {
        let is_clean = block < last && DIR.is_guard_clean(block);
        let mut open = block < last && !is_clean;
        if open && DIR.is_guard_absent(block) {
            if block >= uncommitted_until {
                uncommitted_until = uncommitted_end(block);
            }
            open = block >= uncommitted_until;
            if open {
                DIR.set_bit(block, BitField::GuardAbsent, false, Ordering::SeqCst);
            }
        }
        if open {
            shard_written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.announced(block) {
                // Announced kernel write still to land: leave open, reported
                // written at every check until `kernel_wrote`.
            } else if DIR.is_guard_protected(block) {
                // Written through the runtime (`mapped_pages_written`), still read-only.
                // Defer marking `GuardClean` until after dirty epochs are published.
                alias_clean.push(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            debug_assert_range_locked(&access, from, block);
            virtual_protect_adaptive(
                from,
                block,
                PAGE_READONLY,
                true,
                &mut |ok_from, ok_to| newly_protected.push((ok_from, ok_to)),
                &mut |abs_from, abs_to| newly_absent.push((abs_from, abs_to)),
                &mut |_| {},
            );
        }
    }
    if shard_written.is_empty() {
        return 0;
    }
    // Publish dirty records strictly before making any block visible as
    // `GuardClean`, `GuardProtected`, or `GuardAbsent` so lock-free readers
    // never observe a clean block ahead of its write record.
    let epoch = DIR.publish_written(&shard_written);
    for block in alias_clean {
        DIR.set_bit(block, BitField::GuardClean, true, Ordering::SeqCst);
    }
    for (ok_from, ok_to) in newly_protected {
        DIR.set_bits_range(
            ok_from,
            ok_to,
            BitField::GuardProtected,
            true,
            Ordering::SeqCst,
        );
        DIR.set_bits_range(ok_from, ok_to, BitField::GuardClean, true, Ordering::SeqCst);
    }
    for (abs_from, abs_to) in newly_absent {
        DIR.set_bits_range(
            abs_from,
            abs_to,
            BitField::GuardAbsent,
            true,
            Ordering::SeqCst,
        );
    }
    written.extend_from_slice(&shard_written);
    epoch
}

/// Whether `[start, end)` contains a mapped view and no committed non-view
/// memory. Checks `start` first via [`mapped`] (fast path); when `start` is
/// not a view (e.g. an unmapped placeholder/hole inside a view pool), walks
/// `VirtualQuery` across `[start, end)` and rejects the range if any
/// committed non-view region is present so private/image memory is never
/// read-protected.
fn has_mapped(start: usize, end: usize) -> bool {
    if mapped(start) {
        return true;
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
/// the allocation type avoids enumerating pages with identical protection,
/// which can span an entire large view after its blocks were guarded.
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

/// Gives write access back to the guarded blocks of `[start, end)` while their
/// guard shards are held; clears `PageDir` guard bits strictly before
/// `VirtualProtect(..., PAGE_READWRITE)` so lock-free readers immediately see
/// the block open. Returns `true` if any block was guarded.
fn open_under_locked(access: &impl ShardAccess, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    if start >= end {
        return false;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    debug_assert_range_locked(access, first, last);
    let blocks = DIR.collect_set_bits(first, last, BitField::GuardProtected);
    if blocks.is_empty() {
        return false;
    }
    for &block in &blocks {
        DIR.set_bit(block, BitField::GuardProtected, false, Ordering::SeqCst);
        DIR.set_bit(block, BitField::GuardClean, false, Ordering::SeqCst);
    }
    protect_block_runs(&blocks, PAGE_READWRITE);
    true
}

/// Allocation-free single-block fast path for the VEH write fault handler.
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

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded and
/// cold-promoted blocks. For the title's kernel reads, which other threads'
/// refreshes can overtake, the runtime calls [`kernel_will_write`] instead.
pub(crate) fn writing(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    // The length is the title's (`fread`'s size times count, saturated).
    let end = at.saturating_add(len);
    let (first, last) = (at / BLOCK, end.div_ceil(BLOCK));
    let needs_open = if lockfree_enabled() {
        DIR.any_bits_in_range(first, last, &[BitField::GuardProtected, BitField::ColdRo])
    } else {
        let _g = lock_shards_read_range(first, last);
        DIR.any_bits_in_range(first, last, &[BitField::GuardProtected, BitField::ColdRo])
    };
    if needs_open {
        let locked = GuardLockSet::lock_range(first, last);
        open_under_locked(&locked, at, end);
        demote_cold_range(&locked, first, last);
    }
}

/// Before the title's kernel read into its memory `[at, at + len)`: opens
/// the guarded and cold-promoted blocks there and keeps them open until
/// [`kernel_wrote`]. A refresh on another thread in between would otherwise
/// guard or cold-promote them again, and the read would fail.
pub fn kernel_will_write(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let range = (at / BLOCK, end.div_ceil(BLOCK));
    // Lock order: `GUARD_SHARDS_TABLE` (ascending) -> `AHEAD`.
    let mut locked = GuardLockSet::lock_range(range.0, range.1);
    open_under_locked(&locked, at, end);
    demote_cold_range(&locked, range.0, range.1);
    locked.announce_range(range);
    ANNOUNCED_SEQ.fetch_add(1, Ordering::SeqCst);
    let mut ahead = AHEAD.write().unwrap_or_else(|e| e.into_inner());
    ahead.forget_ahead(range);
}

/// The kernel write [`kernel_will_write`] announced for `[at, at + len)`
/// has landed (or failed); its blocks can be guarded or cold-promoted again.
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

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped), and kernel writes announced wholly inside it are moot (an
/// overlapped read that never said it landed no longer keeps those blocks
/// open).
pub fn title_memory_remapped(at: usize, len: usize) {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    crate::texback::remapped(at, len);
    if len == 0 {
        return;
    }
    let (first, end) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    if first >= end {
        return;
    }
    let mut locked = GuardLockSet::lock_range(first, end);
    locked.clear_announced_inside(first, end);
    DIR.set_bits_range(first, end, BitField::GuardAbsent, false, Ordering::SeqCst);
    DIR.set_bits_range(first, end, BitField::Watched, false, Ordering::Relaxed);

    let mut to_restore = DIR.collect_set_bits(first, end, BitField::GuardProtected);
    let cold_ro = DIR.collect_set_bits(first, end, BitField::ColdRo);
    if !cold_ro.is_empty() {
        to_restore.extend(cold_ro);
        to_restore.sort_unstable();
        to_restore.dedup();
    }
    DIR.set_bits_range(
        first,
        end,
        BitField::GuardProtected,
        false,
        Ordering::SeqCst,
    );
    DIR.set_bits_range(first, end, BitField::GuardClean, false, Ordering::SeqCst);
    DIR.set_bits_range(first, end, BitField::ColdRo, false, Ordering::SeqCst);
    DIR.reset_cold_range(first, end, current_frame32());

    if !to_restore.is_empty() {
        protect_block_runs(&to_restore, PAGE_READWRITE);
    }
    // Invalidate any `ask_once` ahead coverage overlapping the remapped range
    // and bump `ANNOUNCED_SEQ` so a concurrent `ask_once` batch discards stale
    // pre-remap coverage. Lock order: `GUARD_SHARDS_TABLE` -> `AHEAD` (leaf).
    ANNOUNCED_SEQ.fetch_add(1, Ordering::SeqCst);
    let mut ahead = AHEAD.write().unwrap_or_else(|e| e.into_inner());
    ahead.forget_ahead((first, end));
}

/// The runtime left `[at, at + len)` mapped to the exact same physical pages
/// during a page-mapping batch (`CopyPageMappingsBatchX`), skipping redundant
/// unmap/remap syscalls. Guarded copies over the range still treat its
/// contents as potentially updated via another alias.
pub fn title_memory_alias_rebound(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (first, end) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    if first >= end {
        return;
    }
    let _locked = GuardLockSet::lock_range(first, end);
    DIR.set_bits_range(first, end, BitField::GuardClean, false, Ordering::SeqCst);
}

/// Coordinates in-place CPU reads of title memory with mapping changes.
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
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: `ContextRecord` is valid when Windows dispatches an exception.
        let rip = unsafe { (*(*info).ContextRecord).Rip };
        if crate::host::is_fast_copy_rip(rip) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        if r.ExceptionInformation[0] != WRITE {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        let block = at / BLOCK;
        // Another thread may have opened the block since this one faulted:
        // avoid waiting on the shard lock if `PageDir` and Windows already
        // show it read-write.
        if lockfree_enabled()
            && !DIR.is_guard_protected(block)
            && !DIR.is_cold_ro(block)
            && writable(at)
        {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // Held across all three checks: another thread may have opened the
        // block since this one faulted, and a copy may have guarded or
        // cold-promoted it again since.
        let shard = shard_of(block);
        let g = guard_shard(shard);
        let single = SingleShard { shard, guard: &g };
        if open_single_block(&single, block) {
            drop(g);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if open_cold_ro(&single, at) || writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
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

    /// Serializes tests that drive the process-wide `ask_once` / `PageDir`
    /// journal state so concurrent tests cannot observe each other's `AHEAD`.
    static JOURNAL_TEST_LOCK: Mutex<()> = Mutex::new(());

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

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let alloc = WatchedAllocGuard::new(4);
        let p = alloc.base;
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

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
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
        mapped_pages_written(std::iter::once(view + BLOCK));
        let (alias, updated) = writes(view, end, true).expect("alias write");
        assert_eq!(updated, [(view / PAGE, first), (view / PAGE + 1, alias)]);
        assert_eq!(writes(view, end, true).unwrap().1, updated);
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
        assert_eq!(
            // SAFETY: the bytes just read.
            unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) },
            b"from the disk"
        );
        let (changed, pages) = writes(view, end, true).expect("kernel write visible");
        assert_eq!(pages, [(view / PAGE, changed), (view / PAGE + 1, alias)]);
        assert_eq!(writes(view, end, true).unwrap().1, pages);
        title_memory_remapped(view, 2 * BLOCK);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
        }
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

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
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

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile section of two.
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
            // SAFETY: splits the placeholder at `block` and maps the section slice there.
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
        title_memory_remapped(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            CloseHandle(section);
        }
    }

    /// Guarded view memory is left out of `ask_once`, while non-view
    /// write-watched memory is still queried via `GetWriteWatch` and its writes
    /// are reported.
    #[test]
    fn ask_once_skips_known_guarded_views_and_watches_private_memory() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        struct ViewGuard {
            section: HANDLE,
            view_p: usize,
            pages: usize,
        }
        impl Drop for ViewGuard {
            fn drop(&mut self) {
                title_memory_remapped(self.view_p, self.pages * PAGE);
                DIR.clear_written_range(self.view_p / PAGE, self.view_p / PAGE + self.pages);
                // SAFETY: clean up view and section owned by this test.
                unsafe {
                    UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: self.view_p as *mut _,
                    });
                    CloseHandle(self.section);
                }
            }
        }

        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let alloc = WatchedAllocGuard::new(1);
        let priv_p = alloc.base;
        // Reserve and map two pages so both the primary page and the alias page
        // tested via `wrote_own` belong to this fixture.
        // SAFETY: fresh 2-page pagefile section view.
        let view_guard = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * PAGE) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view_p = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * PAGE).Value as usize;
            assert_ne!(view_p, 0);
            DIR.clear_written_range(view_p / PAGE, view_p / PAGE + 2);
            ViewGuard {
                section,
                view_p,
                pages: 2,
            }
        };
        let view_p = view_guard.view_p;

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

        // `wrote_own` records both the page and its fixture-owned alias in `PageDir`.
        let (before, own_now) = wrote_own(view_p / PAGE, &[view_p / PAGE + 1]);
        assert!(own_now > before);
        let (entries, _) = DIR.collect_written(view_p / PAGE, view_p / PAGE + 2);
        assert_eq!(
            entries,
            vec![(view_p / PAGE, own_now), (view_p / PAGE + 1, own_now)]
        );
    }

    #[test]
    fn lock_free_page_dir_dirty_fast_path_and_ask_once_coherence() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let alloc = WatchedAllocGuard::new(6);
        let base = alloc.base;

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
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let threshold = cold_after();
        if threshold == 0 {
            return;
        }
        const PAGES: usize = 4;
        let alloc = WatchedAllocGuard::new(PAGES);
        let base = alloc.base;

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
        // clears `ColdRo`, resets streak, and records page 1 dirty.
        // SAFETY: inside committed allocation; VEH restores write access.
        unsafe { ((base + PAGE + 32) as *mut u8).write_volatile(0xab) };
        assert!(!DIR.is_cold_ro(base / PAGE + 1));
        assert!(writable(base + PAGE));
        assert_eq!(DIR.cold_streak(base / PAGE + 1), 0);
        assert_eq!(
            // SAFETY: written byte is readable.
            unsafe { ((base + PAGE + 32) as *const u8).read_volatile() },
            0xab
        );

        let (after_fault, list) = writes(base, base + PAGES * PAGE, false).expect("post-fault");
        assert!(
            list.iter()
                .any(|&(p, w)| p == base / PAGE + 1 && w > cold_epoch && w <= after_fault),
            "faulted cold page must appear dirty with epoch > cold_epoch"
        );

        // `writing` and `kernel_will_write` demote remaining cold pages cleanly.
        assert!(DIR.is_cold_ro(base / PAGE + 2));
        writing(base + 2 * PAGE, PAGE);
        assert!(!DIR.is_cold_ro(base / PAGE + 2));
        assert!(writable(base + 2 * PAGE));

        assert!(DIR.is_cold_ro(base / PAGE + 3));
        kernel_will_write(base + 3 * PAGE, PAGE);
        assert!(!DIR.is_cold_ro(base / PAGE + 3));
        assert!(writable(base + 3 * PAGE));
        kernel_wrote(base + 3 * PAGE, PAGE);
    }

    #[test]
    fn coalesced_virtual_protect_handles_mixed_views_and_holes() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

        // Second guard check skips absent blocks and returns empty written vec.
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
    fn wrote_own_epoch_is_monotonic_against_concurrent_higher_epoch() {
        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

    /// Exercises 128-shard `GUARD_SHARDS_TABLE` across single-shard and
    /// cross-shard boundaries under concurrent `guard()`, `kernel_will_write()` /
    /// `kernel_wrote()`, `writing()`, `mapped_pages_written()`, and VEH
    /// `on_write()` faults, verifying announcement semantics and deadlock freedom.
    #[test]
    fn sharded_guards_concurrent_ops_and_cross_shard_boundaries() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };

        let _serial = JOURNAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // `SHARD_BLOCKS + 4` blocks (1.25 MiB) guarantees every view spans at
        // least two distinct guard shards.
        const BLOCKS: usize = SHARD_BLOCKS + 4;
        const SIZE: usize = BLOCKS * BLOCK;

        struct MappedView(usize);
        impl MappedView {
            fn new() -> Self {
                // SAFETY: fresh pagefile section mapped read-write.
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
                Self(view)
            }
        }
        impl Drop for MappedView {
            fn drop(&mut self) {
                title_memory_remapped(self.0, SIZE);
                // SAFETY: mapped view owned by this guard.
                unsafe {
                    UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: self.0 as *mut _,
                    });
                }
            }
        }

        let view_a = MappedView::new();
        let view_b = MappedView::new();
        let one = view_a.0;
        let two = view_b.0;
        let b_one = one / BLOCK;
        assert_ne!(
            shard_of(b_one),
            shard_of(b_one + BLOCKS - 1),
            "test view must cross a guard shard boundary"
        );
        let boundary = (1..BLOCKS)
            .find(|&i| shard_of(b_one + i) != shard_of(b_one + i - 1))
            .expect("shard boundary inside multi-stripe view");

        // 1. Cross-shard guarding & announcement semantics straddling a boundary.
        let initial = guard(one, one + SIZE).expect("view guarded");
        assert_eq!(initial.len(), BLOCKS);
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        for i in 0..BLOCKS {
            assert!(DIR.is_guard_protected(b_one + i));
            assert!(DIR.is_guard_clean(b_one + i));
            assert!(!writable(one + i * BLOCK));
        }

        let ann_lo = boundary - 1;
        let ann_hi = boundary + 1;
        kernel_will_write(one + ann_lo * BLOCK, (ann_hi - ann_lo) * BLOCK);
        for i in ann_lo..ann_hi {
            assert!(
                writable(one + i * BLOCK),
                "announced block {i} must be writable"
            );
        }
        let expected_announced: Vec<usize> = (ann_lo..ann_hi).map(|i| b_one + i).collect();
        assert_eq!(guard(one, one + SIZE), Some(expected_announced.clone()));
        assert_eq!(
            guard(one, one + SIZE),
            Some(expected_announced),
            "blocks straddling shard boundary stay open while announced"
        );

        kernel_wrote(one + ann_lo * BLOCK, (ann_hi - ann_lo) * BLOCK);
        assert_eq!(
            guard(one, one + SIZE),
            Some((ann_lo..ann_hi).map(|i| b_one + i).collect())
        );
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));

        // 2. `mapped_pages_written` across shard boundary marks only those blocks unclean.
        mapped_pages_written([one + ann_lo * BLOCK, one + boundary * BLOCK].into_iter());
        assert_eq!(
            guard(one, one + SIZE),
            Some(vec![b_one + ann_lo, b_one + boundary])
        );
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));

        // 3. Concurrent multi-threaded stress across overlapping & disjoint shards.
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();

        // Thread 1: guard `one` and `two` across shard boundaries.
        {
            let stop = stop.clone();
            handles.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    assert!(guard(one, one + SIZE).is_some());
                    assert!(guard(two, two + SIZE).is_some());
                }
            }));
        }
        // Thread 2: guard in opposite order + exercise `writes` and `title_memory_reading`.
        {
            let stop = stop.clone();
            handles.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    assert!(writes(two, two + SIZE, true).is_some());
                    assert!(writes(one, one + SIZE, true).is_some());
                    let _r = title_memory_reading(one, SIZE);
                }
            }));
        }
        // Thread 3: trigger VEH `on_write` faults across shard boundaries on both views.
        {
            let stop = stop.clone();
            handles.push(std::thread::spawn(move || {
                let mut v = 0u8;
                while !stop.load(Ordering::Relaxed) {
                    for &base in &[one, two] {
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
        // Thread 4: `kernel_will_write`/`kernel_wrote`, `writing`, and `mapped_pages_written`
        // straddling shard boundaries.
        {
            let stop = stop.clone();
            handles.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    kernel_will_write(one + (boundary - 1) * BLOCK, 3 * BLOCK);
                    writing(two + (boundary - 1) * BLOCK, 3 * BLOCK);
                    mapped_pages_written(
                        [two + (boundary - 1) * BLOCK, two + boundary * BLOCK].into_iter(),
                    );
                    kernel_wrote(one + (boundary - 1) * BLOCK, 3 * BLOCK);
                    std::thread::yield_now();
                }
            }));
        }

        std::thread::sleep(std::time::Duration::from_millis(150));
        stop.store(true, Ordering::SeqCst);
        for h in handles {
            h.join().expect("worker thread must not panic or deadlock");
        }

        // Post-stress sanity: both views re-guard cleanly across all shards.
        assert!(guard(one, one + SIZE).is_some());
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert!(guard(two, two + SIZE).is_some());
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
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
        assert!(!changed.load(Ordering::SeqCst));
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
        assert!(!read.load(Ordering::SeqCst));
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
