//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (CPU-writable VRAM via Resizable BAR when supported with
//! headroom, or write-back system memory otherwise), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! Ranges draws are known to read are brought up to date (vertex and index
//! buffer views, constant buffers in tables, root descriptors, buffer views
//! in tables, and buffer / constant views reached through
//! `ResourceDescriptorHeap` via [`crate::descriptor::host::HostHeap::refresh_bindless_buffers`]).
//! shortcut: unordered buffer ranges reached only through the heap are
//! refreshed for reading but not copied back after the submission (a
//! full-heap UAV snapshot is too expensive).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATES, ID3D12Resource,
};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// Total VRAM bytes currently allocated to ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`)
/// registered shadow buffers across live [`Shadow`] instances.
static REBAR_SHADOW_BYTES: AtomicU64 = AtomicU64::new(0);

/// Total VRAM bytes currently allocated to ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`)
/// `LOOSE` window buffers across live [`Shadow`] instances.
static REBAR_LOOSE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Monotonic identifier for each live [`Shadow`] host copy (used by indirect
/// window and per-buffer gather gating to avoid pointer-reuse ABA).
static NEXT_SHADOW_ID: AtomicU64 = AtomicU64::new(1);

/// Current ReBAR VRAM bytes reserved by live registered [`Shadow`]s.
pub(crate) fn rebar_shadow_bytes() -> u64 {
    REBAR_SHADOW_BYTES.load(Ordering::Relaxed)
}

/// Current ReBAR VRAM bytes reserved by live `LOOSE` window [`Shadow`]s.
pub(crate) fn rebar_loose_bytes() -> u64 {
    REBAR_LOOSE_BYTES.load(Ordering::Relaxed)
}

/// Atomically reserves `size` bytes from `budget`, returning `false` if doing
/// so would exceed `budget`.
fn try_reserve_rebar(used: &AtomicU64, size: u64, budget: u64) -> bool {
    if budget == 0 || size == 0 || size > budget {
        return false;
    }
    used.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
        let next = cur.checked_add(size)?;
        (next <= budget).then_some(next)
    })
    .is_ok()
}

/// Maximum 64 KiB pieces hashed on the GPU for one ReBAR shadow
/// (`65535 * 64 KiB`, ~4 GiB): keeps 32-bit byte offsets in `rebar_hash.hlsl`
/// overflow-free and fits in a single 1D D3D12 `Dispatch(pieces, 1, 1)`.
pub(crate) const REBAR_HASH_MAX_PIECES: u32 = 65535;

/// Compact bitset of 64 KiB pieces whose VRAM contents were modified on the CPU
/// since their GPU hash was last computed.
struct RebarDirtyBits {
    pieces: u32,
    words: Vec<u64>,
    any: bool,
}

impl RebarDirtyBits {
    fn new(pieces: u32) -> Self {
        Self {
            pieces,
            words: vec![0; (pieces as usize).div_ceil(64)],
            any: false,
        }
    }

    /// Marks pieces `[first, first + count)` (clamped to `self.pieces`) dirty.
    fn mark_pieces(&mut self, first: u32, count: u32) {
        let start = first.min(self.pieces);
        let end = start.saturating_add(count).min(self.pieces);
        if start >= end {
            return;
        }
        self.any = true;
        let first_word = (start / 64) as usize;
        let last_word = ((end - 1) / 64) as usize;
        let start_bit = start % 64;
        let end_bit = ((end - 1) % 64) + 1;
        if first_word == last_word {
            let mask = if end_bit == 64 {
                !0u64 << start_bit
            } else {
                ((1u64 << (end_bit - start_bit)) - 1) << start_bit
            };
            self.words[first_word] |= mask;
        } else {
            self.words[first_word] |= !0u64 << start_bit;
            for w in &mut self.words[first_word + 1..last_word] {
                *w = !0u64;
            }
            let tail_mask = if end_bit == 64 {
                !0u64
            } else {
                (1u64 << end_bit) - 1
            };
            self.words[last_word] |= tail_mask;
        }
    }

    /// Marks every 64 KiB piece overlapping buffer byte range `[off, off + len)`.
    fn mark_byte_range(&mut self, off: usize, len: usize) {
        if len == 0 || self.pieces == 0 {
            return;
        }
        let end_byte = off.saturating_add(len);
        let first = (off / PAGE).min(self.pieces as usize) as u32;
        let end = end_byte.div_ceil(PAGE).min(self.pieces as usize) as u32;
        if first < end {
            self.mark_pieces(first, end - first);
        }
    }

    /// Clears all dirty bits and returns coalesced `(first_piece, count)` runs.
    fn take_runs(&mut self) -> Vec<(u32, u32)> {
        if !self.any {
            return Vec::new();
        }
        self.any = false;
        let mut runs: Vec<(u32, u32)> = Vec::new();
        let mut cur: Option<(u32, u32)> = None;
        for (wi, word) in self.words.iter_mut().enumerate() {
            let mut w = std::mem::replace(word, 0);
            let base = (wi as u32) * 64;
            while w != 0 {
                let tz = w.trailing_zeros();
                let shifted = w >> tz;
                let ones = shifted.trailing_ones();
                let run_start = base + tz;
                let run_end = (run_start + ones).min(self.pieces);
                if run_start < run_end {
                    match cur {
                        Some((s, e)) if e == run_start => cur = Some((s, run_end)),
                        Some((s, e)) => {
                            runs.push((s, e - s));
                            cur = Some((run_start, run_end));
                        }
                        None => cur = Some((run_start, run_end)),
                    }
                }
                if tz + ones >= 64 {
                    break;
                }
                w &= !(((1u64 << ones) - 1) << tz);
            }
        }
        if let Some((s, e)) = cur {
            runs.push((s, e - s));
        }
        runs
    }
}

/// Prepared GPU hash state for one ReBAR shadow before a submission.
pub(crate) struct PreparedRebarHashes {
    pub(crate) hashes_gpu: u64,
    pub(crate) aligned_len: u32,
    pub(crate) pre_runs: Vec<(u32, u32)>,
}

/// GPU-side 128-bit per-64-KiB-piece hashes for a UAV-written ReBAR shadow.
///
/// Invariant: for every piece whose bit in `dirty` is clear, `buffer[piece]`
/// holds the 128-bit hash of `Shadow::buffer`'s bytes in that piece.
struct RebarHashes {
    _buffer: ID3D12Resource,
    gpu: u64,
    dirty: RebarDirtyBits,
}

/// Extra state for a [`Shadow`] backed by CPU-writable VRAM
/// (`D3D12_HEAP_TYPE_GPU_UPLOAD`, Resizable BAR).
///
/// `Shadow::data` is a write-combining pointer into VRAM and MUST NEVER be read
/// on the CPU. Read-only buffers keep `mirror = None` and `hashes = None` (zero
/// duplicate RAM or hash VRAM). On the first UAV write, `mirror` is lazily
/// populated via a one-time GPU DMA and kept synced on subsequent
/// `Shadow::copy` / `copy_back_rebar` calls; `hashes` is lazily created as a
/// `DEFAULT` UAV buffer (`16` bytes per 64 KiB piece) so post-submission
/// copy-back only DMAs 64 KiB pieces whose GPU hash actually changed.
struct RebarShadow {
    bytes: u64,
    is_loose: bool,
    mirror: Mutex<Option<Vec<u8>>>,
    /// Leaf lock: never acquired while holding `mirror`, `pending`, or `copied`,
    /// and never held across D3D12 or kernel calls.
    hashes: Mutex<Option<RebarHashes>>,
}

impl RebarShadow {
    fn has_mirror(&self) -> bool {
        self.mirror
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    fn mark_dirty_range(&self, off: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut guard = self.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            h.dirty.mark_byte_range(off, len);
        }
    }

    fn mark_dirty_segs(&self, start: usize, segs: &[CopySeg]) {
        if segs.is_empty() {
            return;
        }
        let mut guard = self.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            for seg in segs {
                if seg.len > 0 {
                    h.dirty.mark_byte_range(seg.src - start, seg.len);
                }
            }
        }
    }
}

/// A title buffer's host copy.
pub(crate) struct Shadow {
    id: u64,
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Monotonic resource creation sequence for epoch-based retirement.
    born_seq: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Before>,
    /// Present when `buffer` lives in ReBAR `D3D12_HEAP_TYPE_GPU_UPLOAD` VRAM.
    rebar: Option<RebarShadow>,
    /// Monotonic tick of the last loose-cache access (0 for registered buffers).
    last_used: AtomicU64,
}

impl Drop for Shadow {
    fn drop(&mut self) {
        if let Some(rebar) = &self.rebar {
            if rebar.is_loose {
                REBAR_LOOSE_BYTES.fetch_sub(rebar.bytes, Ordering::Relaxed);
            } else {
                REBAR_SHADOW_BYTES.fetch_sub(rebar.bytes, Ordering::Relaxed);
            }
        }
    }
}

#[derive(Default)]
struct Before {
    snapshot: Option<(usize, Vec<u8>)>,
    unchanged: UnchangedPages,
}

/// Pages already compared in full against this submission's snapshot.
#[derive(Default)]
struct UnchangedPages {
    start: usize,
    end: usize,
    pages: Vec<bool>,
}

impl UnchangedPages {
    fn begin(&mut self, start: usize, end: usize) {
        if self.pages.is_empty() {
            self.start = start;
            self.end = end;
            self.pages.resize(end.div_ceil(PAGE) - start / PAGE, false);
        }
    }

    fn differs(&mut self, address: usize, current: &[u8], before: &[u8]) -> bool {
        let page = address / PAGE;
        let unchanged = &mut self.pages[page - self.start / PAGE];
        if *unchanged {
            return false;
        }
        if current != before {
            return true;
        }
        // A later range can touch bytes a partial comparison missed. At
        // the edges, only bytes covered by the snapshot can be compared.
        *unchanged = address == (page * PAGE).max(self.start)
            && address + current.len() == ((page + 1) * PAGE).min(self.end);
        false
    }

    fn clear(&mut self) {
        self.pages.clear();
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// Unique identifier for this host copy, stable for its lifetime.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Where the copy starts in title memory and how long it is.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// `len` bytes at a title address from the host copy: the bytes the copy
    /// holds, as of its last refresh (None outside the buffer). Tests only.
    #[cfg(test)]
    pub(crate) fn bytes(&self, address: u64, len: u64) -> Option<&[u8]> {
        if address < self.start || len == 0 || len > self.left(address) {
            return None;
        }
        let at = usize::try_from(address - self.start).ok()?;
        let len = usize::try_from(len).ok()?;
        // SAFETY: inside the mapping, which is as long as the buffer.
        Some(unsafe { std::slice::from_raw_parts(self.data.add(at), len) })
    }

    /// Clamps `[address, address + len)` to this copy's title-address span,
    /// returning `None` if the intersection is empty.
    #[inline]
    pub(crate) fn clamp_range(&self, address: u64, len: u64) -> Option<(u64, u64)> {
        let from = address.max(self.start);
        let clamped = range_inside(
            from,
            len.saturating_sub(from.saturating_sub(address)),
            self.start,
            self.len,
        );
        (clamped > 0).then_some((from, clamped))
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset, without recording a
    /// resource-state transition on `list` itself (used when `list`'s prelude
    /// reads the buffer and restores it to `COMMON`).
    pub(crate) fn refresh_for_prelude(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        list.keep(&self.buffer);
        let from = address.max(self.start);
        let len = range_inside(
            from,
            len.saturating_sub(from.saturating_sub(address)),
            self.start,
            self.len,
        );
        if len == 0 {
            return;
        }
        list.defer_refresh(self, from as usize, len as usize);
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset. Cut to the copy.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        self.refresh_as(list, address, len, D3D12_RESOURCE_STATE_GENERIC_READ);
    }

    /// Refresh a range and prepare its host buffer for the specified GPU use.
    /// Clamped to the copy.
    pub(crate) fn refresh_as(
        self: &Arc<Self>,
        list: &HostList,
        address: u64,
        len: u64,
        access: D3D12_RESOURCE_STATES,
    ) {
        list.use_shadow(&self.buffer, access);
        let Some((from, len)) = self.clamp_range(address, len) else {
            return;
        };
        list.defer_refresh(self, from as usize, len as usize);
    }

    fn new(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        rebar_bytes: Option<u64>,
    ) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            id: NEXT_SHADOW_ID.fetch_add(1, Ordering::Relaxed),
            buffer,
            data,
            gpu,
            start,
            len,
            born_seq: crate::host::birth_seq(),
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
            rebar: rebar_bytes.map(|bytes| RebarShadow {
                bytes,
                is_loose: false,
                mirror: Mutex::new(None),
                hashes: Mutex::new(None),
            }),
            last_used: AtomicU64::new(0),
        }
    }

    fn new_loose(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        rebar_bytes: Option<u64>,
    ) -> Shadow {
        let mut shadow = Self::new(buffer, data, gpu, start, len, rebar_bytes);
        if let Some(rebar) = &mut shadow.rebar {
            rebar.is_loose = true;
        }
        shadow
    }

    /// Whether this copy lives in ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) VRAM.
    pub(crate) fn is_rebar(&self) -> bool {
        self.rebar.is_some()
    }

    /// Logical byte length of the title buffer.
    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    /// Base host GPU virtual address of the shadow buffer.
    pub(crate) fn base_gpu(&self) -> u64 {
        self.gpu
    }

    /// Lazily allocates the per-64-KiB-piece hash UAV buffer for a ReBAR shadow
    /// and returns [`PreparedRebarHashes`], where `pre_runs` are the
    /// `(first_piece, piece_count)` runs that must be hashed before the
    /// submission executes (all pieces on initial creation, or only
    /// CPU-dirtied pieces thereafter). Returns `None` if this shadow is not
    /// eligible for GPU hashing or allocation fails.
    pub(crate) fn prepare_rebar_hashes(
        &self,
        alloc: impl FnOnce(u64) -> Option<ID3D12Resource>,
    ) -> Option<PreparedRebarHashes> {
        let rebar = self.rebar.as_ref()?;
        if self.len == 0 {
            return None;
        }
        let pieces = u32::try_from(self.len.div_ceil(PAGE as u64)).ok()?;
        if pieces > REBAR_HASH_MAX_PIECES {
            return None;
        }
        let aligned_len = u32::try_from(self.len.next_multiple_of(16)).ok()?;
        if rebar
            .mirror
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_none()
        {
            return None;
        }
        {
            let mut guard = rebar.hashes.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(h) = guard.as_mut() {
                return Some(PreparedRebarHashes {
                    hashes_gpu: h.gpu,
                    aligned_len,
                    pre_runs: h.dirty.take_runs(),
                });
            }
        }
        // Allocate outside the leaf lock.
        let buffer = alloc(u64::from(pieces) * 16)?;
        crate::host::name(&buffer, || {
            format!("ReBAR hashes {:#x}+{:#x}", self.start, self.len)
        });
        // SAFETY: live D3D12 buffer resource.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let mut guard = rebar.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            return Some(PreparedRebarHashes {
                hashes_gpu: h.gpu,
                aligned_len,
                pre_runs: h.dirty.take_runs(),
            });
        }
        *guard = Some(RebarHashes {
            _buffer: buffer,
            gpu,
            dirty: RebarDirtyBits::new(pieces),
        });
        Some(PreparedRebarHashes {
            hashes_gpu: gpu,
            aligned_len,
            pre_runs: vec![(0, pieces)],
        })
    }

    /// Marks `(first_piece, piece_count)` runs dirty in this ReBAR shadow's
    /// hash state (e.g. if a pre-hash recording failed after draining dirty
    /// bits).
    pub(crate) fn mark_rebar_dirty_runs(&self, runs: &[(u32, u32)]) {
        let Some(rebar) = &self.rebar else {
            return;
        };
        let mut guard = rebar.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            for &(first, count) in runs {
                h.dirty.mark_pieces(first, count);
            }
        }
    }

    /// Marks all 64 KiB pieces overlapping buffer-offset `spans` dirty in this
    /// ReBAR shadow's hash state (used when a submission falls back to
    /// full-range readback without running the post-compare shader).
    pub(crate) fn mark_rebar_dirty_spans(&self, spans: &[(usize, usize)]) {
        let Some(rebar) = &self.rebar else {
            return;
        };
        let mut guard = rebar.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            for &(from, to) in spans {
                if from < to {
                    h.dirty.mark_byte_range(from, to - from);
                }
            }
        }
    }

    fn touch(&self) {
        self.last_used.store(
            LOOSE_TICK.fetch_add(1, Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    #[cfg(all(test, windows))]
    pub(crate) fn read_test_bytes(&self, address: u64, len: usize) -> Vec<u8> {
        let (start, end) = self.inside(address as usize, len);
        // SAFETY: `self.data` is persistently mapped for `self.len` bytes and
        // `inside` clamps `start..end` to `0..self.len`.
        unsafe { std::slice::from_raw_parts(self.data.add(start), end - start) }.to_vec()
    }

    #[cfg(all(test, windows))]
    pub(crate) fn new_rebar_for_test(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
    ) -> Arc<Self> {
        // Balance `Shadow::drop`'s `REBAR_SHADOW_BYTES` decrement even on unwind.
        REBAR_SHADOW_BYTES.fetch_add(len, Ordering::Relaxed);
        Arc::new(Self::new(buffer, data, gpu, start, len, Some(len)))
    }

    #[cfg(all(test, windows))]
    pub(crate) fn has_pending(&self) -> bool {
        !self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    pub(crate) fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (
            from - start,
            address.saturating_add(len).clamp(from, end) - start,
        )
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let end = self.start.saturating_add(self.len);
        let from = address.max(self.start);
        let to = address.saturating_add(len).min(end);
        let len = to.saturating_sub(from);
        if len == 0 {
            return;
        }
        list.defer_gpu_write(self, from as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame).
    ///
    /// Returns `false` if a ReBAR shadow's initial RAM mirror could not be read
    /// back; in that case `pending` stays recorded so refreshes cannot overwrite
    /// the range.
    pub(crate) fn writing(
        &self,
        address: usize,
        len: usize,
        rebar_init: impl FnOnce(&ID3D12Resource, u64) -> Option<Vec<u8>>,
    ) -> bool {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address.saturating_add(len)));
        if let Some(rebar) = &self.rebar {
            let mut mirror = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
            if mirror.is_none() {
                *mirror = rebar_init(&self.buffer, self.len);
            }
            return mirror.is_some();
        }
        let (from, to) = self.inside(address, len);
        // SAFETY: inside the mapping (write-back system memory; ReBAR buffers
        // returned early above so write-combining memory is never read).
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let before = &mut before.snapshot;
        let (start, end) = match &*before {
            Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
            None => (from, to),
        };
        if let Some((at, bytes)) = &*before
            && (*at, at + bytes.len()) == (start, end)
        {
            return true;
        }
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        match before.take() {
            Some((at, bytes)) => {
                grown.extend_from_slice(host(start, at));
                grown.extend_from_slice(&bytes);
                grown.extend_from_slice(host(at + bytes.len(), end));
                give_back(bytes);
            }
            None => grown.extend_from_slice(host(start, end)),
        }
        *before = Some((start, grown));
        true
    }

    /// Clears pending UAV-write ranges for a ReBAR shadow once GPU hash-compare
    /// proves no copy-back is needed (or declared spans are empty).
    pub(crate) fn clear_pending_rebar(&self) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// ReBAR counterpart to [`Shadow::copy_back`]: merges post-submission GPU
    /// bytes (`gpu_slices`, already DMAed into cached `READBACK` RAM) for
    /// sorted disjoint buffer-offset `spans` against the cached RAM mirror,
    /// without ever reading write-combining VRAM on the CPU.
    ///
    /// Returns `None` (without clearing `pending`) if the shadow is not
    /// ReBAR-backed, its pre-submission mirror is missing, or `spans` and
    /// `gpu_slices` do not match.
    pub(crate) fn copy_back_rebar(
        &self,
        spans: &[(usize, usize)],
        gpu_slices: &[&[u8]],
    ) -> Option<usize> {
        let rebar = self.rebar.as_ref()?;
        if spans.len() != gpu_slices.len() {
            return None;
        }
        let mut mirror_guard = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
        let mirror = mirror_guard.as_mut()?;
        for (&(from, to), &gpu_span) in spans.iter().zip(gpu_slices) {
            if from > to || to > mirror.len() || gpu_span.len() != to - from {
                return None;
            }
        }
        let mut written = 0;
        let mut vram_dirty: Vec<(usize, usize)> = Vec::new();
        let start = self.start as usize;
        let mut title = None;
        for (&(from, to), &gpu_span) in spans.iter().zip(gpu_slices) {
            if from == to {
                continue;
            }
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PAGE - (start + at) % PAGE).min(to - at);
                let gpu = &gpu_span[at - from..at - from + n];
                let old = &mut mirror[at..at + n];
                if gpu != old {
                    let title_buf = &mut title.get_or_insert_with(|| vec![0u8; PAGE])[..n];
                    if crate::host::read_into((start + at) as *const u8, title_buf) {
                        let mut changed = false;
                        let mut host_differs = false;
                        for ((t, &g), o) in title_buf.iter_mut().zip(gpu).zip(old.iter_mut()) {
                            if g != *o && *t != g {
                                *t = g;
                                changed = true;
                            }
                            if *t != g {
                                host_differs = true;
                            }
                            *o = *t;
                        }
                        if changed {
                            crate::watch::writing(start + at, n);
                            crate::host::write_mapped(
                                (start + at) as *mut u8,
                                title_buf.as_ptr(),
                                n,
                            );
                            written += n;
                        }
                        if host_differs {
                            // Only push back to WC VRAM when a concurrent CPU
                            // write altered bytes in the same page; record the
                            // range so its GPU piece hash is refreshed next.
                            // SAFETY: `at + n <= self.len`, write-only WC store.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    title_buf.as_ptr(),
                                    self.data.add(at),
                                    n,
                                );
                            }
                            vram_dirty.push((at, n));
                        }
                        synced.push((start + at) / PAGE);
                    } else {
                        // Unmapped in title memory: VRAM holds `gpu`, keep mirror in sync.
                        old.copy_from_slice(gpu);
                    }
                }
                at += n;
            }
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        drop(mirror_guard);
        for (at, n) in vram_dirty {
            rebar.mark_dirty_range(at, n);
        }
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        Some(written)
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Before {
            snapshot,
            unchanged,
        } = &mut *before;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = &*snapshot
            && *taken <= from
            && to <= taken + before.len()
            && from < to
        {
            unchanged.begin(start + taken, start + taken + before.len());
            let before = &before[from - taken..to - taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            // An unchanged page is never modified by copy-back, so later
            // overlapping ranges can skip it for the rest of the submission.
            let mut title = None;
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PAGE - (start + at) % PAGE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                if unchanged.differs(start + at, host, old)
                    && let title = &mut title.get_or_insert_with(|| vec![0u8; PAGE])[..n]
                    && crate::host::read_into((start + at) as *const u8, title)
                {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    let mut changed = false;
                    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                        if *h != *o && *t != *h {
                            *t = *h;
                            changed = true;
                        }
                        *h = *t;
                    }
                    if changed {
                        crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += n;
                    }
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address.saturating_add(len)));
        if pending.is_empty()
            && let Some((_, bytes)) = snapshot.take()
        {
            unchanged.clear();
            give_back(bytes);
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let plan = self.plan_update(address, len);
        if plan.runs.is_empty() {
            return 0;
        }
        self.copy_runs(&plan.runs)
    }

    /// Computes the disjoint dirty byte runs of `[address, address + len)`
    /// outside pending shader writes and advances `self.copied` for whole
    /// stale pages, releasing `self.copied`'s lock before any memory copy.
    fn plan_update(&self, address: usize, len: usize) -> UpdatePlan {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        if from >= to {
            return UpdatePlan::default();
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        let mut plan = UpdatePlan::default();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let mut at = from;
            for (a, e) in busy {
                if a > at {
                    self.plan_update_free(at, a, &mut plan);
                }
                at = at.max(e);
            }
            if at < to {
                self.plan_update_free(at, to, &mut plan);
            }
            return plan;
        }
        self.plan_update_free(from, to, &mut plan);
        plan
    }

    /// [`Shadow::plan_update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn plan_update_free(&self, from: usize, to: usize, plan: &mut UpdatePlan) {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let before = plan.runs.len();
                self.collect_stale(from, to, now, &written, &mut plan.runs);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    let copied: usize = plan.runs[before..].iter().map(|&(a, b)| b - a).sum();
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                let _ = before;
                return;
            }
            // Only mark the shadow permanently unwatched if it has never been
            // observed as watched. A sub-range that lands entirely in an
            // unmapped hole inside an already-watched view pool must not
            // disable write-watching for the rest of the pool.
            let _ =
                self.watched
                    .compare_exchange(0, NOT_WATCHED, Ordering::Relaxed, Ordering::Relaxed);
        }
        plan.runs.push((from, to));
    }

    /// Collects contiguous stale byte ranges `(lo, hi)` of `[from, to)`
    /// written after their last copy (`written`: last writes by page, in page
    /// order; `now`: this asking), advances `self.copied` for whole pages,
    /// and releases `self.copied`'s lock before any copy runs.
    fn collect_stale(
        &self,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
        out: &mut Vec<(usize, usize)>,
    ) {
        let base = self.start as usize / PAGE;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let stale = stale_pages_in(base, &copied, from, to, written);
        let mut run: Option<(usize, usize)> = None;
        for page in stale {
            // A page only partly in the range is copied in part: it
            // stays stale for the rest.
            if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                copied[page - base] = now;
            }
            match run {
                Some((first, prev)) if page == prev + 1 => run = Some((first, page)),
                Some((first, prev)) => {
                    let lo = (first * PAGE).max(from);
                    let hi = ((prev + 1) * PAGE).min(to);
                    if lo < hi {
                        out.push((lo, hi));
                    }
                    run = Some((page, page));
                }
                None => run = Some((page, page)),
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some((first, prev)) = run {
            let lo = (first * PAGE).max(from);
            let hi = ((prev + 1) * PAGE).min(to);
            if lo < hi {
                out.push((lo, hi));
            }
        }
    }

    /// Converts disjoint title-address `runs` (clamped to the buffer) into
    /// [`CopySeg`] entries targeting `self.data`, returning the total bytes.
    fn push_copy_segs(&self, runs: &[(usize, usize)], out: &mut Vec<CopySeg>) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let mut total = 0usize;
        for &(raw_from, raw_to) in runs {
            let (from, to) = (raw_from.max(start), raw_to.min(end));
            if from >= to {
                continue;
            }
            let off = from - start;
            let len = to - from;
            out.push(CopySeg {
                src: from,
                dst: (self.data as usize) + off,
                mirror: 0,
                len,
            });
            total += len;
        }
        total
    }

    /// Copies disjoint title-memory `runs`, cut to the buffer, into the copy
    /// (in parallel across worker threads when [`wide_copy`] is enabled and
    /// total bytes reach [`WIDE_COPY_MIN`]).
    fn copy_runs(&self, runs: &[(usize, usize)]) -> usize {
        if runs.is_empty() {
            return 0;
        }
        let start = self.start as usize;
        let mut segs = Vec::with_capacity(runs.len());
        let total = self.push_copy_segs(runs, &mut segs);
        if total == 0 {
            return 0;
        }
        if let Some(rebar) = &self.rebar {
            let mut mirror = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(m) = mirror.as_mut() {
                let m_ptr = m.as_mut_ptr() as usize;
                let m_len = m.len();
                if segs.iter().all(|s| (s.src - start) + s.len <= m_len) {
                    // Copy once from guarded title memory into cached RAM, then
                    // push straight to write-combining VRAM so both see identical bytes.
                    for s in &mut segs {
                        s.mirror = m_ptr + (s.src - start);
                    }
                    run_copy_segs(&segs);
                    drop(mirror);
                    rebar.mark_dirty_segs(start, &segs);
                    return total;
                }
            }
        }
        run_copy_segs(&segs);
        self.finish_copy_segs(start, &segs);
        total
    }

    /// Post-copy metadata invalidation for non-mirrored `segs`, acquiring each
    /// per-shadow lock at most once.
    fn finish_copy_segs(&self, start: usize, segs: &[CopySeg]) {
        if segs.is_empty() {
            return;
        }
        if let Some(rebar) = &self.rebar {
            rebar.mark_dirty_segs(start, segs);
        }
    }
}

type Job = Box<dyn FnOnce() + Send>;

struct Workers {
    jobs: Mutex<std::collections::VecDeque<Job>>,
    more: std::sync::Condvar,
}

std::thread_local! {
    /// Set while a thread is executing a job inside [`wide`] so nested callers
    /// never re-enter the shared worker pool and deadlock.
    static IN_WIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct InWideGuard(bool);

impl InWideGuard {
    fn enter() -> Self {
        Self(IN_WIDE.with(|flag| flag.replace(true)))
    }
}

impl Drop for InWideGuard {
    fn drop(&mut self) {
        IN_WIDE.with(|flag| flag.set(self.0));
    }
}

fn in_wide() -> bool {
    IN_WIDE.with(|flag| flag.get())
}

fn workers() -> &'static Workers {
    static WORKERS: OnceLock<Workers> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let workers = WORKERS.get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: std::sync::Condvar::new(),
    });
    STARTED.call_once(|| {
        let n = std::thread::available_parallelism()
            .map_or(4, |n| n.get() / 2)
            .clamp(2, 6);
        for i in 0..n {
            let _ = std::thread::Builder::new()
                .name(format!("victory-wide-{i}"))
                .spawn(move || {
                    let workers = WORKERS.get().expect("made above");
                    loop {
                        let job = {
                            let mut jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
                            loop {
                                if let Some(job) = jobs.pop_front() {
                                    break job;
                                }
                                jobs = workers.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                            }
                        };
                        let _guard = InWideGuard::enter();
                        job();
                    }
                });
        }
    });
    workers
}

fn wide(jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            let _guard = InWideGuard::enter();
            job();
        }
        return;
    }
    let done = Arc::new((Mutex::new(jobs.len()), std::sync::Condvar::new()));
    let workers = workers();
    {
        let mut queue = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            let done = done.clone();
            queue.push_back(Box::new(move || {
                struct Done(Arc<(Mutex<usize>, std::sync::Condvar)>);
                impl Drop for Done {
                    fn drop(&mut self) {
                        let mut left = self.0.0.lock().unwrap_or_else(|e| e.into_inner());
                        *left -= 1;
                        if *left == 0 {
                            self.0.1.notify_all();
                        }
                    }
                }
                let _done = Done(done);
                job();
            }));
        }
    }
    workers.more.notify_all();
    loop {
        let job = workers
            .jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front();
        match job {
            Some(job) => {
                let _guard = InWideGuard::enter();
                job();
            }
            None => break,
        }
    }
    let mut left = done.0.lock().unwrap_or_else(|e| e.into_inner());
    while *left > 0 {
        left = done.1.wait(left).unwrap_or_else(|e| e.into_inner());
    }
}

/// Whether large shadow refreshes, parallel batch planning, and staging fills
/// parallelize their dirty copy chunks across worker threads
/// (`VICTORY_WIDE_COPY=0` disables it; `=1` forces it; on by default on 4+
/// logical CPUs).
pub(crate) fn wide_copy() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        wide_copy_enabled(
            std::env::var("VICTORY_WIDE_COPY").ok().as_deref(),
            std::thread::available_parallelism().ok().map(|n| n.get()),
        )
    })
}

fn wide_copy_enabled(setting: Option<&str>, logical_cpus: Option<usize>) -> bool {
    match setting {
        Some("1") => true,
        Some("0") => false,
        _ => logical_cpus.is_some_and(|n| n >= 4),
    }
}

/// Minimum total dirty bytes to split across worker threads in [`run_copy_segs`].
const WIDE_COPY_MIN: usize = 1 << 20;
/// Maximum target byte size per parallel copy chunk.
const WIDE_COPY_CHUNK: usize = 512 << 10;

/// One contiguous byte range to copy from title memory into a shadow copy
/// (and optionally through a cached ReBAR RAM mirror into WC VRAM).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CopySeg {
    src: usize,
    dst: usize,
    /// Non-zero for mirrored ReBAR shadows: copy `src -> mirror`, then
    /// `mirror -> dst` (WC VRAM).
    mirror: usize,
    len: usize,
}

/// Slices `segs` into buckets of at most `chunk_bytes` each, splitting long
/// runs cleanly without losing or duplicating a single byte.
fn chunk_copy_segs(segs: &[CopySeg], chunk_bytes: usize) -> Vec<Vec<CopySeg>> {
    let chunk_bytes = chunk_bytes.max(1);
    let mut buckets: Vec<Vec<CopySeg>> = Vec::new();
    let mut cur: Vec<CopySeg> = Vec::new();
    let mut bytes = 0usize;
    for &seg in segs {
        let mut at = 0usize;
        while at < seg.len {
            let n = (chunk_bytes - bytes).min(seg.len - at);
            cur.push(CopySeg {
                src: seg.src + at,
                dst: seg.dst + at,
                mirror: if seg.mirror != 0 { seg.mirror + at } else { 0 },
                len: n,
            });
            at += n;
            bytes += n;
            if bytes >= chunk_bytes {
                buckets.push(std::mem::take(&mut cur));
                bytes = 0;
            }
        }
    }
    if !cur.is_empty() {
        buckets.push(cur);
    }
    buckets
}

#[inline]
fn execute_copy_seg(seg: CopySeg) {
    if seg.len == 0 {
        return;
    }
    if seg.mirror != 0 {
        crate::host::copy_mapped(seg.mirror as *mut u8, seg.src as *const u8, seg.len, false);
        // SAFETY: `seg` lies inside the shadow's mirror and WC VRAM mapping
        // (`off + len <= self.len`), and chunks are strictly disjoint.
        unsafe {
            std::ptr::copy_nonoverlapping(seg.mirror as *const u8, seg.dst as *mut u8, seg.len);
        }
    } else {
        crate::host::copy_mapped(seg.dst as *mut u8, seg.src as *const u8, seg.len, false);
    }
}

/// Executes copy jobs on the warm `workers()` pool when called from outside
/// [`wide`], or via a bounded `std::thread::scope` if called re-entrantly from
/// inside a `wide` worker so the pool can never deadlock.
fn run_copy_jobs(jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            job();
        }
        return;
    }
    if !in_wide() {
        wide(jobs);
        return;
    }
    // Re-entrant call from inside a `wide` worker: never push back into the
    // pool queue; fan out with a small scoped group instead.
    let threads = jobs.len().min(4);
    let queue = Mutex::new(std::collections::VecDeque::from(jobs));
    std::thread::scope(|s| {
        for _ in 1..threads {
            let queue = &queue;
            s.spawn(move || {
                while let Some(job) = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front() {
                    job();
                }
            });
        }
        while let Some(job) = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front() {
            job();
        }
    });
}

fn run_copy_segs(segs: &[CopySeg]) {
    let total: usize = segs.iter().map(|s| s.len).sum();
    if total == 0 {
        return;
    }
    if !wide_copy() || total < WIDE_COPY_MIN {
        for &seg in segs {
            execute_copy_seg(seg);
        }
        return;
    }
    let chunk_bytes = (total / 4).clamp(256 << 10, WIDE_COPY_CHUNK);
    let buckets = chunk_copy_segs(segs, chunk_bytes);
    if buckets.len() < 2 {
        for &seg in segs {
            execute_copy_seg(seg);
        }
        return;
    }
    let jobs: Vec<Job> = buckets
        .into_iter()
        .map(|bucket| {
            Box::new(move || {
                for seg in bucket {
                    execute_copy_seg(seg);
                }
            }) as Job
        })
        .collect();
    run_copy_jobs(jobs);
}

/// Copies `len` bytes from title memory `src` into writable host memory `dst`,
/// splitting large fills across worker threads when [`wide_copy`] is enabled.
pub(crate) fn copy_mapped_wide(dst: *mut u8, src: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    if !wide_copy() || len < WIDE_COPY_MIN {
        crate::host::copy_mapped(dst, src, len, false);
        return;
    }
    run_copy_segs(&[CopySeg {
        src: src as usize,
        dst: dst as usize,
        mirror: 0,
        len,
    }]);
}

/// Dirty byte ranges discovered by [`Shadow::plan_update`] before copying.
#[derive(Default)]
struct UpdatePlan {
    runs: Vec<(usize, usize)>,
}

/// Two-phase refresh for a [`crate::host::Batch`] flush:
/// 1. Discover stale byte runs for each shadow (in parallel across shadows when
///    `parallel_plan` is enabled and there are at least 4 shadows), releasing
///    each shadow's `copied` lock immediately after planning.
/// 2. Flatten all non-mirrored dirty runs across the entire batch into balanced
///    copy chunks and execute them in one flat [`wide`] pass, then apply
///    per-shadow metadata updates (ReBAR dirty bits) and copy any mirrored
///    ReBAR shadows.
///
/// This eliminates both straggler single-thread bottlenecks (one huge 50–174 MB
/// buffer stalling a multi-shadow flush) and nested `wide` re-entrancy.
pub(crate) fn refresh_batch(items: Vec<(Arc<Shadow>, usize, usize)>, parallel_plan: bool) -> usize {
    if items.is_empty() {
        return 0;
    }
    if items.len() == 1 {
        let (shadow, from, to) = items.into_iter().next().unwrap();
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let copied = shadow.update(from, to - from);
        #[cfg(feature = "diagnostics")]
        {
            if let Some(bytes) = skipped(to - from, copied) {
                crate::copystat::add(crate::copystat::C::Skipped, 1);
                crate::copystat::add(crate::copystat::C::SkippedBytes, bytes);
            }
            crate::copystat::range(from, to - from, started.elapsed().as_micros() as u64, 0);
        }
        return copied;
    }

    #[cfg(feature = "diagnostics")]
    let batch_started = std::time::Instant::now();

    type PlannedItem = (Arc<Shadow>, usize, usize, UpdatePlan);

    let count = items.len();
    let planned: Vec<PlannedItem> = if parallel_plan && count >= 4 && !in_wide() {
        let slots: Arc<Vec<Mutex<Option<PlannedItem>>>> =
            Arc::new((0..count).map(|_| Mutex::new(None)).collect());
        let jobs: Vec<Job> = items
            .into_iter()
            .enumerate()
            .map(|(i, (shadow, from, to))| {
                let slots = slots.clone();
                Box::new(move || {
                    let plan = shadow.plan_update(from, to - from);
                    *slots[i].lock().unwrap_or_else(|e| e.into_inner()) =
                        Some((shadow, from, to, plan));
                }) as Job
            })
            .collect();
        wide(jobs);
        slots
            .iter()
            .filter_map(|slot| slot.lock().unwrap_or_else(|e| e.into_inner()).take())
            .collect()
    } else {
        items
            .into_iter()
            .map(|(shadow, from, to)| {
                let plan = shadow.plan_update(from, to - from);
                (shadow, from, to, plan)
            })
            .collect()
    };

    let mut all_segs: Vec<CopySeg> = Vec::new();
    let mut non_mirrored: Vec<(usize, usize, usize)> = Vec::new();
    let mut mirrored: Vec<usize> = Vec::new();
    let mut item_copied = vec![0usize; planned.len()];
    let mut total_copied = 0usize;

    for (idx, (shadow, _, _, plan)) in planned.iter().enumerate() {
        if plan.runs.is_empty() {
            continue;
        }
        if shadow.rebar.as_ref().is_some_and(RebarShadow::has_mirror) {
            mirrored.push(idx);
            continue;
        }
        let seg_lo = all_segs.len();
        let bytes = shadow.push_copy_segs(&plan.runs, &mut all_segs);
        let seg_hi = all_segs.len();
        if bytes > 0 {
            non_mirrored.push((idx, seg_lo, seg_hi));
            item_copied[idx] = bytes;
            total_copied += bytes;
        }
    }

    if !all_segs.is_empty() {
        run_copy_segs(&all_segs);
        for &(idx, seg_lo, seg_hi) in &non_mirrored {
            let shadow = &planned[idx].0;
            shadow.finish_copy_segs(shadow.start as usize, &all_segs[seg_lo..seg_hi]);
        }
    }

    for idx in mirrored {
        let (shadow, _, _, plan) = &planned[idx];
        let bytes = shadow.copy_runs(&plan.runs);
        item_copied[idx] = bytes;
        total_copied += bytes;
    }

    #[cfg(feature = "diagnostics")]
    {
        let batch_us = batch_started.elapsed().as_micros() as u64;
        for ((_, from, to, _), copied) in planned.iter().zip(item_copied) {
            if let Some(bytes) = skipped(to.saturating_sub(*from), copied) {
                crate::copystat::add(crate::copystat::C::Skipped, 1);
                crate::copystat::add(crate::copystat::C::SkippedBytes, bytes);
            }
            crate::copystat::range(*from, to.saturating_sub(*from), batch_us, 0);
        }
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = item_copied;

    total_copied
}

/// Pages of `[a, e)` that have never been copied (`copied == 0`) or were
/// written by the title after their last copy (`write > copied`). Matches
/// [`Shadow::copy_stale`], which copies both cases even when a never-copied
/// page does not appear in the write-watch delta `written`.
fn stale_pages_in(
    base: usize,
    copied: &[u64],
    a: usize,
    e: usize,
    written: &[(usize, u64)],
) -> Vec<usize> {
    if a >= e {
        return Vec::new();
    }
    let mut writes = written.iter().peekable();
    let mut out = Vec::new();
    for page in a / PAGE..=(e - 1) / PAGE {
        while writes.peek().is_some_and(|&&(p, _)| p < page) {
            writes.next();
        }
        let write = match writes.peek() {
            Some(&&(p, w)) if p == page => {
                writes.next();
                w
            }
            _ => 0,
        };
        if page >= base
            && copied
                .get(page - base)
                .is_none_or(|&at| at == 0 || write > at)
        {
            out.push(page);
        }
    }
    out
}

/// A refresh that copies none of its range is skipped: the writes it was
/// asked about were already in the copy, the range is outside the buffer,
/// or shaders were still writing all of it (that part is left to a later
/// refresh). Counted for the frame-rate line as one skipped refresh of
/// `len` bytes, so `0 skipped` there means no refresh was answered
/// without a copy, not that nothing was checked. A range copied in part is
/// not skipped; of what it did not copy, `unchanged` counts only the parts
/// this refresh checked, not the sub-ranges shaders still write.
#[cfg(any(feature = "diagnostics", test))]
pub(crate) fn skipped(len: usize, copied: usize) -> Option<u64> {
    (len > 0 && copied == 0).then_some(len as u64)
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
/// `RwLock` allows parallel recording threads to look up existing shadows
/// concurrently without contending on a global mutex.
static BUFFERS: RwLock<BTreeMap<(u64, usize), Entry>> = RwLock::new(BTreeMap::new());

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    BUFFERS.write().unwrap_or_else(|e| e.into_inner()).insert(
        (start, resource),
        Entry {
            // A size from the title's description: one past the address
            // space ends at its top instead of wrapping below `start`.
            end: start.saturating_add(len),
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive,
/// and any active epoch keeps it until that work completes.
pub(crate) fn unregister(resource: usize, start: u64) {
    let shadow = {
        let mut map = BUFFERS.write().unwrap_or_else(|e| e.into_inner());
        map.remove(&(start, resource)).and_then(|e| e.shadow)
    };
    if let Some(shadow) = shadow {
        let born = shadow.born_seq;
        crate::host::retire_born(shadow, born);
    }
}

/// Reserves `size` bytes against `used` up to `budget` and allocates a ReBAR
/// buffer, rolling back the reservation if D3D12 allocation fails.
fn try_alloc_rebar_in(
    used: &AtomicU64,
    size: u64,
    budget: u64,
) -> Option<(ID3D12Resource, *mut u8, u64)> {
    let h = host()?;
    if try_reserve_rebar(used, size, budget) {
        if let Some((buffer, data)) = h.try_rebar_shadow_buffer(size) {
            return Some((buffer, data, size));
        }
        used.fetch_sub(size, Ordering::Relaxed);
    }
    None
}

/// Allocates a shadow backing buffer: prefers ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`)
/// VRAM within the adapter's budget, falling back cleanly to system-RAM
/// [`Host::shared_buffer`] when ReBAR is unavailable, disabled, or full.
fn alloc_shadow_buffer(size: u64) -> Option<(ID3D12Resource, *mut u8, Option<u64>)> {
    let h = host()?;
    if let Some((buffer, data, bytes)) =
        try_alloc_rebar_in(&REBAR_SHADOW_BYTES, size, h.rebar_shadow_budget())
    {
        return Some((buffer, data, Some(bytes)));
    }
    let (buffer, data) = h.shared_buffer(size)?;
    Some((buffer, data, None))
}

/// Allocates a ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) backing buffer for a
/// `LOOSE` window from the dedicated [`REBAR_LOOSE_BYTES`] budget so `LOOSE`
/// windows never starve registered `BUFFERS` out of [`REBAR_SHADOW_BYTES`].
fn alloc_loose_rebar(size: u64) -> Option<(ID3D12Resource, *mut u8, u64)> {
    let h = host()?;
    try_alloc_rebar_in(&REBAR_LOOSE_BYTES, size, h.rebar_loose_budget())
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    // Fast path: concurrent read lock with a single BTreeMap lookup.
    let missing_key = {
        let map = BUFFERS.read().unwrap_or_else(|e| e.into_inner());
        match map
            .range(..=(address, usize::MAX))
            .rev()
            .take(64)
            .find(|(_, e)| e.end >= end)
        {
            Some((_, entry)) if entry.shadow.is_some() => return entry.shadow.clone(),
            Some((&key, _)) => Some(key),
            None => None,
        }
    };
    let Some(key) = missing_key else {
        return loose(address, end);
    };
    // Slow path: first use of this buffer creates its host copy under write lock.
    let mut map = BUFFERS.write().unwrap_or_else(|e| e.into_inner());
    let key = if map.get(&key).is_some_and(|e| e.end >= end) {
        key
    } else {
        match map
            .range(..=(address, usize::MAX))
            .rev()
            .take(64)
            .find(|(_, e)| e.end >= end)
            .map(|(k, _)| *k)
        {
            Some(k) => k,
            None => {
                drop(map);
                return loose(address, end);
            }
        }
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let alloc_size = len.checked_next_multiple_of(1 << 16)?;
        let (buffer, data, rebar_bytes) = alloc_shadow_buffer(alloc_size)?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                rebar = rebar_bytes.is_some(),
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                rebar = rebar_bytes.is_some(),
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(
            buffer,
            data,
            gpu,
            key.0,
            len,
            rebar_bytes,
        )));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (from their
/// own allocators). Such memory gets copies of the aligned windows it touches,
/// bounded by a RAM-scaled LRU cache.
const WINDOW: u64 = 4 << 20;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// Monotonic access clock for LRU eviction of loose shadow windows.
static LOOSE_TICK: AtomicU64 = AtomicU64::new(1);

fn total_physical_ram_bytes() -> Option<u64> {
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        dwMemoryLoad: 0,
        ullTotalPhys: 0,
        ullAvailPhys: 0,
        ullTotalPageFile: 0,
        ullAvailPageFile: 0,
        ullTotalVirtual: 0,
        ullAvailVirtual: 0,
        ullAvailExtendedVirtual: 0,
    };
    // SAFETY: `status` is initialized with `dwLength` set to its byte size as
    // required by `GlobalMemoryStatusEx`.
    let ok = unsafe { GlobalMemoryStatusEx(&raw mut status) };
    (ok != 0 && status.ullTotalPhys > 0).then_some(status.ullTotalPhys)
}

fn loose_budget_for_ram(total_ram_bytes: Option<u64>) -> u64 {
    match total_ram_bytes {
        None => 256 * MIB,
        Some(ram) if ram <= 16 * GIB => 128 * MIB,
        Some(ram) if ram <= 32 * GIB => 256 * MIB,
        Some(_) => 512 * MIB,
    }
}

#[cfg(all(test, windows))]
static TEST_LOOSE_BUDGET: AtomicU64 = AtomicU64::new(0);

#[cfg(all(test, windows))]
static TEST_LOOSE_LOCK: Mutex<()> = Mutex::new(());

#[cfg(all(test, windows))]
pub(crate) struct LooseBudgetGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(all(test, windows))]
impl Drop for LooseBudgetGuard {
    fn drop(&mut self) {
        TEST_LOOSE_BUDGET.store(0, Ordering::SeqCst);
        let drained: Vec<Arc<Shadow>> = {
            let mut guard = LOOSE.write().unwrap_or_else(|e| e.into_inner());
            guard.total_bytes = 0;
            std::mem::take(&mut guard.entries).into_values().collect()
        };
        drop(drained);
    }
}

/// Test-only helper that clears the global loose shadow cache and forces an
/// explicit byte budget until the returned guard drops.
#[cfg(all(test, windows))]
pub(crate) fn isolated_loose_budget(budget: u64) -> LooseBudgetGuard {
    let lock = TEST_LOOSE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    {
        let mut guard = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        guard.total_bytes = 0;
        guard.entries.clear();
    }
    TEST_LOOSE_BUDGET.store(budget, Ordering::SeqCst);
    LooseBudgetGuard { _lock: lock }
}

fn loose_budget() -> u64 {
    #[cfg(all(test, windows))]
    {
        let override_budget = TEST_LOOSE_BUDGET.load(Ordering::SeqCst);
        if override_budget != 0 {
            return override_budget;
        }
    }
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let ram = total_physical_ram_bytes();
        let budget = loose_budget_for_ram(ram);
        tracing::info!(
            total_ram_mib = ram.map(|b| b / MIB),
            budget_mib = budget / MIB,
            "loose shadow cache budget"
        );
        budget
    })
}

struct LooseCache {
    entries: BTreeMap<(u64, u64), Arc<Shadow>>,
    total_bytes: u64,
}

impl LooseCache {
    const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            total_bytes: 0,
        }
    }

    /// Evicts least-recently-used idle windows while `total_bytes > budget`,
    /// never evicting `keep`, entries still held outside the cache
    /// (`Arc::strong_count > 1`), or entries with pending GPU copy-back ranges.
    fn evict_over_budget(&mut self, keep: (u64, u64), budget: u64) -> Vec<Arc<Shadow>> {
        if self.total_bytes <= budget {
            return Vec::new();
        }
        let mut candidates: Vec<(u64, (u64, u64))> = self
            .entries
            .iter()
            .filter_map(|(&key, shadow)| {
                if key == keep || Arc::strong_count(shadow) != 1 {
                    return None;
                }
                if !shadow
                    .pending
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty()
                {
                    return None;
                }
                Some((shadow.last_used.load(Ordering::Relaxed), key))
            })
            .collect();
        candidates.sort_unstable_by_key(|&(tick, _)| tick);
        let mut evicted = Vec::new();
        for (_, key) in candidates {
            if self.total_bytes <= budget {
                break;
            }
            if let Some(shadow) = self.entries.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(shadow.len);
                evicted.push(shadow);
            }
        }
        if !evicted.is_empty() {
            tracing::debug!(
                evicted = evicted.len(),
                total_bytes = self.total_bytes,
                budget,
                "evicted idle loose shadow windows"
            );
        }
        evicted
    }
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: RwLock<LooseCache> = RwLock::new(LooseCache::new());

/// Copies of just one view's bytes, by (start, length): for a view whose
/// start is not on an element of its host copy (`aligned`). The heap slot
/// holds the copy; this map only finds it again, so it holds weak references
/// and the copy goes when the slot is remade.
static ALIGNED: Mutex<BTreeMap<(u64, u64), std::sync::Weak<Shadow>>> = Mutex::new(BTreeMap::new());

/// A copy of `[address, address + len)` that starts at `address` itself, so a
/// structured view of it can have its elements where the title put them.
///
/// A title aligns a resource; the host copy of a buffer starts where the
/// buffer does, so a view of memory the title placed inside another object
/// (`View::Placed`) starts at an offset that need not be a multiple of the
/// view's stride. This copy starts at the view, and the slot that made it
/// keeps it alive.
///
/// Its own refresh copies the 64 KiB pages the title wrote since they were
/// last copied, straight from title memory: a submission's copy-back runs
/// before the next submission prepares (`Host::execute` waits for the GPU
/// before copying back), so by then title memory holds what the GPU wrote in
/// any copy, and what shaders wrote in this one has gone back there. Copying
/// from the buffer's copy instead would copy the view's whole range again on
/// every submission.
pub(crate) fn aligned(address: u64, len: u64) -> Option<Arc<Shadow>> {
    if len == 0 {
        return None;
    }
    let key = (address, len);
    if let Some(copy) = aligned_existing(address, len) {
        return Some(copy);
    }
    // Views hold hundreds of megabytes in the world; made as a buffer's copy
    // is, so that shaders may write it (unordered access).
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(64))?;
    crate::host::name(&buffer, || format!("aligned view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let copy = Arc::new(Shadow::new(buffer, data, gpu, address, len, None));
    // Populate now so a ResourceHeap-only first use (which never walks the
    // table refresh path) still sees title memory rather than zeros (#553).
    // SAFETY: `data` is the mapping of `buffer`; `address` is title memory
    // of `len` bytes the view was created for.
    crate::host::copy_mapped(data, address as *const u8, len as usize, false);
    {
        let mut copies = ALIGNED.lock().unwrap_or_else(|e| e.into_inner());
        // Drop expired weaks so distinct (address, len) churn cannot grow
        // the map without bound (docs/code-standards.md host-cache bound).
        copies.retain(|_, weak| weak.strong_count() > 0);
        // Another recorder may have published the same range while we allocated.
        if let Some(existing) = copies.get(&key).and_then(std::sync::Weak::upgrade) {
            return Some(existing);
        }
        copies.insert(key, Arc::downgrade(&copy));
    }
    tracing::debug!(
        address = format_args!("{address:#x}"),
        len,
        "aligned view copy made"
    );
    Some(copy)
}

/// The copy `aligned` made for this range, if one is still alive.
pub(crate) fn aligned_existing(address: u64, len: u64) -> Option<Arc<Shadow>> {
    ALIGNED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(address, len))
        .and_then(std::sync::Weak::upgrade)
}

/// How many bytes of `[address, address + len)` lie inside the copy that
/// starts at `start` of `size` bytes: what a refresh of that copy can bring up
/// to date. `0` when the range is not inside it at all.
fn range_inside(address: u64, len: u64, start: u64, size: u64) -> u64 {
    let end = start.saturating_add(size);
    let from = address.max(start);
    let to = address.saturating_add(len).min(end);
    to.saturating_sub(from)
}

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    loose_in(&LOOSE, address, end, loose_budget())
}

fn loose_in(
    cache: &RwLock<LooseCache>,
    address: u64,
    end: u64,
    budget: u64,
) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let key = (start, len);
    if let Some(shadow) = cache
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .entries
        .get(&key)
    {
        shadow.touch();
        return Some(shadow.clone());
    }
    let (shadow, evicted) = {
        let mut guard = cache.write().unwrap_or_else(|e| e.into_inner());
        if let Some(shadow) = guard.entries.get(&key) {
            shadow.touch();
            return Some(shadow.clone());
        }
        // Prefer ReBAR VRAM from the dedicated `LOOSE` budget (read-only
        // windows keep zero duplicate RAM mirror/hashes) and fall back to
        // system memory when ReBAR is unavailable, disabled, or full.
        let (buffer, data, rebar_bytes) = match alloc_loose_rebar(len) {
            Some((buffer, data, rebar_bytes)) => {
                crate::host::name(&buffer, || {
                    format!("title memory {start:#x}+{len:#x} (ReBAR loose)")
                });
                (buffer, data, Some(rebar_bytes))
            }
            None => {
                let (buffer, data) = host()?.shared_buffer(len)?;
                crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
                (buffer, data, None)
            }
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        tracing::debug!(
            start = format_args!("{start:#x}"),
            len,
            rebar = rebar_bytes.is_some(),
            "loose memory copied to the host"
        );
        let shadow = Arc::new(Shadow::new_loose(
            buffer,
            data,
            gpu,
            start,
            len,
            rebar_bytes,
        ));
        shadow.touch();
        guard.entries.insert(key, shadow.clone());
        guard.total_bytes = guard.total_bytes.saturating_add(len);
        let evicted = guard.evict_over_budget(key, budget);
        (shadow, evicted)
    };
    for old in evicted {
        crate::host::retire(old);
    }
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = BUFFERS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .entries
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    all.extend(
        ALIGNED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter_map(|w| w.upgrade())
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

/// Serializes ReBAR GPU tests that share the process-wide `Host` hash stats
/// and `REBAR_SHADOW_BYTES` / `REBAR_LOOSE_BYTES` counters.
#[cfg(all(test, windows))]
static REBAR_GPU_TEST: Mutex<()> = Mutex::new(());

#[cfg(all(test, windows))]
pub(crate) fn lock_rebar_gpu_test() -> std::sync::MutexGuard<'static, ()> {
    REBAR_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    #[test]
    fn a_buffer_sized_past_the_address_space_ends_at_its_top() {
        let start = u64::MAX - 0x0fff;
        register(usize::MAX, start, u64::MAX);
        let end = BUFFERS
            .read()
            .unwrap()
            .get(&(start, usize::MAX))
            .map(|e| e.end);
        unregister(usize::MAX, start);
        assert_eq!(end, Some(u64::MAX));
    }

    #[test]
    fn a_buffer_too_big_to_round_to_a_shadow_size_gets_no_copy() {
        // The end saturates at the top, so rounding its length up to 64 KiB
        // would overflow: no copy, before any host allocation.
        register(usize::MAX - 1, 0x1000, u64::MAX);
        let found = find(0x1000, 1);
        unregister(usize::MAX - 1, 0x1000);
        assert!(found.is_none());
    }

    #[test]
    fn concurrent_aligned_views_share_the_copy_refreshed_at_submission() {
        if crate::host::host().is_none() {
            return;
        }

        const THREADS: usize = 8;
        let values = [[19u32; 14]; 8];
        for value in &values {
            let address = value.as_ptr() as u64;
            let len = size_of_val(value) as u64;
            let gate = std::sync::Barrier::new(THREADS);
            let copies = std::thread::scope(|scope| {
                let jobs: Vec<_> = (0..THREADS)
                    .map(|_| {
                        let gate = &gate;
                        scope.spawn(move || {
                            gate.wait();
                            aligned(address, len).expect("aligned host buffer")
                        })
                    })
                    .collect();
                jobs.into_iter()
                    .map(|job| job.join().unwrap())
                    .collect::<Vec<_>>()
            });
            let refreshed = aligned_existing(address, len).expect("cached copy");
            refreshed.update(address as usize, len as usize);
            for copy in copies {
                assert!(
                    Arc::ptr_eq(&copy, &refreshed),
                    "slot retained a different copy"
                );
                // SAFETY: the mapped buffer has 56 bytes and refresh has completed.
                let first = unsafe { copy.data.cast::<u32>().read_unaligned() };
                assert_eq!(first, value[0]);
            }
        }
    }

    /// A refresh of the aligned copy brings up to date the bytes of the view
    /// that lie inside it, cut at both ends, and nothing when the range is
    /// outside it.
    #[test]
    fn a_refresh_covers_the_views_bytes_inside_its_copy() {
        // The hit-detection view: 3136 bytes of 56-byte elements, 48 bytes
        // into its buffer, in a copy that starts at the view.
        assert_eq!(range_inside(0x3000_0030, 3136, 0x3000_0030, 3136), 3136);
        // A view reaching past its copy is cut to it.
        assert_eq!(range_inside(0x4000_0100, 900, 0x4000_0100, 0x100), 0x100);
        // A view starting before its copy loses the part before it.
        assert_eq!(range_inside(0x0ff8, 0x40, 0x1000, 0x1000), 0x38);
        // Outside the copy at either end: nothing to refresh.
        assert_eq!(range_inside(0x1000, 0x10, 0x2000, 0x100), 0);
        assert_eq!(range_inside(0x20f0, 0x20, 0x1000, 0x100), 0);
        // A size that would overflow does not.
        assert_eq!(range_inside(u64::MAX - 8, 16, 0, 16), 0);
    }

    /// A refresh that copied nothing of its range is what the frame-rate
    /// line's `skipped` counts; one that copied any part of it is not, and
    /// an empty range is neither.
    #[test]
    fn a_refresh_that_copies_nothing_is_counted_as_skipped() {
        assert_eq!(skipped(0, 0), None);
        assert_eq!(skipped(4 << 20, 0), Some(4 << 20));
        assert_eq!(skipped(4 << 20, 1), None);
        assert_eq!(skipped(4 << 20, 4 << 20), None);
        // Nothing in the buffer: the range was outside it.
        assert_eq!(skipped(4096, 0), Some(4096));
    }

    #[test]
    fn partial_match_leaves_unchecked_bytes_visible() {
        let mut unchanged = UnchangedPages::default();
        let before = vec![0; PAGE];
        let mut host = before.clone();
        host[PAGE - 1] = 1;
        unchanged.begin(PAGE, 2 * PAGE);

        assert!(!unchanged.differs(PAGE, &host[..4], &before[..4]));
        assert!(!unchanged.pages[0]);
        assert!(unchanged.differs(PAGE, &host, &before));
        assert!(!unchanged.pages[0]);
    }

    #[test]
    fn snapshot_edges_are_cacheable_and_survive_later_ranges() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 3, &[1; 3], &[1; 3]));
        assert!(!unchanged.differs(PAGE, &[2; 4], &[2; 4]));
        assert_eq!(unchanged.pages, [true, true]);

        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 2, &[1; 2], &[1; 2]));
        assert!(!unchanged.differs(PAGE + 1, &[2; 2], &[2; 2]));
        assert_eq!(unchanged.pages, [true, true]);
    }

    #[test]
    fn next_submission_checks_the_same_page_again() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, 4);
        assert!(!unchanged.differs(0, &[0; 4], &[0; 4]));
        unchanged.clear();
        unchanged.begin(0, 4);
        assert!(unchanged.differs(0, &[0, 0, 0, 1], &[0; 4]));
    }

    // Both versions walk exactly the same declared ranges, in their
    // original order. Only the page comparison differs.
    fn scan(
        host: &[u8],
        before: &[u8],
        ranges: &[Range<usize>],
        mut differs: impl FnMut(usize, &[u8], &[u8]) -> bool,
    ) -> usize {
        let mut changed = 0;
        for range in ranges {
            let mut at = range.start;
            while at < range.end {
                let n = (PAGE - at % PAGE).min(range.end - at);
                changed += usize::from(differs(at, &host[at..at + n], &before[at..at + n]));
                at += n;
            }
        }
        changed
    }

    #[test]
    fn overlapping_ranges_match_uncached_comparisons() {
        let before = vec![0; 4 * PAGE];
        let mut host = before.clone();
        for at in [2, PAGE - 1, PAGE + 5, 3 * PAGE + 7] {
            host[at] = 1;
        }
        let ranges = [
            0..4,
            2..2 * PAGE + 3,
            PAGE - 2..4 * PAGE,
            2 * PAGE..3 * PAGE,
        ];
        let uncached = scan(&host, &before, &ranges, |_, host, before| host != before);
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, before.len());
        let cached = scan(&host, &before, &ranges, |at, host, before| {
            unchanged.differs(at, host, before)
        });
        assert_eq!(uncached, 6);
        assert_eq!(cached, uncached);
        assert_eq!(unchanged.pages, [false, false, true, false]);
    }

    // cargo test --release -p victory-gpu benchmark_overlapping_copy_back -- --ignored --nocapture
    #[test]
    #[ignore = "synthetic CPU release benchmark"]
    fn benchmark_overlapping_copy_back() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        const SIZE: usize = 128 << 20;
        let before = vec![0xA5; SIZE];
        let mut host = before.clone();
        for page in (0..SIZE / PAGE).step_by(64) {
            host[page * PAGE + PAGE / 2] ^= 0xFF;
        }
        let ranges: Vec<_> = (0..24).map(|i| i * PAGE + i..SIZE - i * 13).collect();
        let mut baseline = [Duration::ZERO; 5];
        let mut cached = [Duration::ZERO; 5];
        let mut unchanged = UnchangedPages::default();
        for (baseline, cached) in baseline.iter_mut().zip(&mut cached) {
            let started = Instant::now();
            let expected = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |_, host, before| host != before,
            );
            *baseline = started.elapsed();

            unchanged.clear();
            let started = Instant::now();
            unchanged.begin(0, before.len());
            let actual = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |at, host, before| unchanged.differs(at, host, before),
            );
            *cached = started.elapsed();
            assert_eq!(actual, expected);
        }
        baseline.sort_unstable();
        cached.sort_unstable();
        println!(
            "median of five runs, 24 overlapping ranges across 128 MiB: {:?} -> {:?} ({:.1}x)",
            baseline[2],
            cached[2],
            baseline[2].as_secs_f64() / cached[2].as_secs_f64(),
        );
    }

    #[cfg(windows)]
    #[test]
    fn unregister_drops_shadow_immediately_when_idle() {
        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0001_usize;
        register(resource, start, data.len() as u64);
        let weak = {
            let shadow = find(start, 64).expect("shadow");
            Arc::downgrade(&shadow)
        };
        assert_eq!(weak.strong_count(), 1);
        unregister(resource, start);
        assert_eq!(weak.strong_count(), 0);
    }

    #[cfg(windows)]
    #[test]
    fn bytes_reads_the_copy_inside_the_buffer_only() {
        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0002_usize;
        register(resource, start, data.len() as u64);
        let shadow = find(start, 64).expect("shadow");
        assert_eq!(shadow.bytes(start, 16).map(<[u8]>::len), Some(16));
        assert_eq!(
            shadow
                .bytes(start + data.len() as u64 - 16, 16)
                .map(<[u8]>::len),
            Some(16),
            "a range ending in the buffer's last bytes is inside it"
        );
        assert!(shadow.bytes(start - 4, 4).is_none(), "before the buffer");
        assert!(
            shadow.bytes(start + data.len() as u64 - 8, 16).is_none(),
            "past the buffer"
        );
        assert!(shadow.bytes(start, 0).is_none(), "nothing to read");
        unregister(resource, start);
    }

    #[cfg(windows)]
    #[test]
    fn unregister_retires_shadow_until_active_epoch_finishes() {
        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0002_usize;
        register(resource, start, data.len() as u64);
        let weak = {
            let shadow = find(start, 64).expect("shadow");
            Arc::downgrade(&shadow)
        };
        assert_eq!(weak.strong_count(), 1);

        let list = host.new_list().expect("host list");
        list.mark_work();

        unregister(resource, start);
        assert_eq!(
            weak.strong_count(),
            1,
            "unregistered shadow must wait for active epochs that recorded after its birth"
        );

        list.close();
        list.reset();
        assert_eq!(weak.strong_count(), 0);
    }

    #[cfg(windows)]
    #[test]
    fn unregister_drops_new_shadow_despite_older_pinned_list() {
        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let stale = host.new_list().expect("stale list");
        stale.mark_work();

        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0003_usize;
        register(resource, start, data.len() as u64);
        let weak = {
            let shadow = find(start, 64).expect("shadow");
            Arc::downgrade(&shadow)
        };
        assert_eq!(weak.strong_count(), 1);

        unregister(resource, start);
        assert_eq!(
            weak.strong_count(),
            0,
            "shadow created after stale list's last work must not leak behind it"
        );
        stale.close();
        stale.reset();
    }

    #[test]
    fn rebar_budget_reservation_enforces_cap_and_releases() {
        let used = AtomicU64::new(0);
        assert!(!try_reserve_rebar(&used, 64, 0));
        assert!(!try_reserve_rebar(&used, 0, 1024));
        assert!(!try_reserve_rebar(&used, 2048, 1024));
        assert!(try_reserve_rebar(&used, 600, 1024));
        assert_eq!(used.load(Ordering::Relaxed), 600);
        assert!(!try_reserve_rebar(&used, 500, 1024));
        assert!(try_reserve_rebar(&used, 424, 1024));
        assert_eq!(used.load(Ordering::Relaxed), 1024);
    }

    #[cfg(windows)]
    #[test]
    fn rebar_loose_budget_is_isolated_and_releases_on_drop() {
        let _rebar = REBAR_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let _retire = crate::host::isolated_retire_scope();
        let Some(h) = host() else {
            return;
        };
        let before_shadow = rebar_shadow_bytes();
        let before_loose = rebar_loose_bytes();

        // Verify drop accounting isolates `REBAR_LOOSE_BYTES` from `REBAR_SHADOW_BYTES`.
        let (buf_a, ptr_a) = h.shared_buffer(4096).expect("shared_buffer a");
        let (buf_b, ptr_b) = h.shared_buffer(4096).expect("shared_buffer b");
        // SAFETY: live D3D12 buffers.
        let gpu_a = unsafe { buf_a.GetGPUVirtualAddress() };
        // SAFETY: live D3D12 buffers.
        let gpu_b = unsafe { buf_b.GetGPUVirtualAddress() };

        assert!(try_reserve_rebar(
            &REBAR_LOOSE_BYTES,
            4096,
            before_loose + 4096
        ));
        assert_eq!(rebar_loose_bytes(), before_loose + 4096);
        assert_eq!(rebar_shadow_bytes(), before_shadow);

        let loose_shadow = Shadow::new_loose(buf_a, ptr_a, gpu_a, 0x2000_0000, 4096, Some(4096));
        assert!(loose_shadow.is_rebar());
        assert!(loose_shadow.rebar.as_ref().is_some_and(|r| r.is_loose));

        assert!(try_reserve_rebar(
            &REBAR_SHADOW_BYTES,
            8192,
            before_shadow + 8192
        ));
        let reg_shadow = Shadow::new(buf_b, ptr_b, gpu_b, 0x3000_0000, 8192, Some(8192));
        assert!(reg_shadow.is_rebar());
        assert!(reg_shadow.rebar.as_ref().is_some_and(|r| !r.is_loose));
        assert_eq!(rebar_loose_bytes(), before_loose + 4096);
        assert_eq!(rebar_shadow_bytes(), before_shadow + 8192);

        drop(loose_shadow);
        assert_eq!(rebar_loose_bytes(), before_loose);
        assert_eq!(rebar_shadow_bytes(), before_shadow + 8192);

        drop(reg_shadow);
        assert_eq!(rebar_loose_bytes(), before_loose);
        assert_eq!(rebar_shadow_bytes(), before_shadow);

        // Exercise live `loose_in` allocation and eviction under the host's ReBAR loose budget.
        // Because `try_rebar_shadow_buffer` can still fall back to shared memory under VRAM
        // pressure or D3D12 allocation failure, derive per-window counter expectations from
        // the backend each `loose_in` call actually returned.
        let check_window = |shadow: &Shadow| -> bool {
            let is_rebar = shadow.is_rebar();
            if is_rebar {
                let r = shadow.rebar.as_ref().expect("rebar state");
                assert!(r.is_loose);
                assert!(r.mirror.lock().unwrap().is_none());
                assert!(r.hashes.lock().unwrap().is_none());
                assert_eq!(rebar_loose_bytes(), before_loose + WINDOW);
            } else {
                assert!(shadow.rebar.is_none());
                assert_eq!(rebar_loose_bytes(), before_loose);
            }
            assert_eq!(rebar_shadow_bytes(), before_shadow);
            is_rebar
        };

        let cache = RwLock::new(LooseCache::new());
        let base = 0x2000_0000_0000_u64;
        let avail_loose = h.rebar_loose_budget().saturating_sub(before_loose);
        if avail_loose >= WINDOW {
            let s0 = loose_in(&cache, base, base + 64, WINDOW).expect("loose win 0");
            let s0_rebar = check_window(&s0);
            drop(s0);

            // Inserting a second window evicts the first idle window and releases its ReBAR bytes.
            // Because `loose_in` reserves ReBAR before eviction runs, `s1` cannot acquire ReBAR
            // when `s0` already consumed the only available window in the loose budget (e.g.
            // `VICTORY_REBAR_LOOSE_MB=4`); evicting `s0` returns `rebar_loose_bytes()` to
            // `before_loose`.
            let s1 =
                loose_in(&cache, base + WINDOW, base + WINDOW + 64, WINDOW).expect("loose win 1");
            if s0_rebar && avail_loose < 2 * WINDOW {
                assert!(
                    !s1.is_rebar(),
                    "single-window loose budget must reject concurrent second ReBAR reservation"
                );
            }
            let s1_rebar = check_window(&s1);
            drop(s1);
            if !s1_rebar {
                // Evicting `s0` freed any single-window ReBAR reservation, so a third window can
                // reserve ReBAR again when VRAM headroom permits.
                let s2 = loose_in(&cache, base + 2 * WINDOW, base + 2 * WINDOW + 64, WINDOW)
                    .expect("loose win 2");
                let _ = check_window(&s2);
                drop(s2);
            }
            drop(cache);
            assert_eq!(rebar_loose_bytes(), before_loose);
            assert_eq!(rebar_shadow_bytes(), before_shadow);
        } else {
            // Opt-out (`VICTORY_REBAR_LOOSE_MB=0`) or insufficient remaining loose budget:
            // `loose_in` cleanly falls back to system RAM with zero ReBAR loose accounting.
            let s0 = loose_in(&cache, base, base + 64, WINDOW).expect("loose win fallback");
            assert!(!s0.is_rebar());
            let _ = check_window(&s0);
            drop(s0);
            drop(cache);
            assert_eq!(rebar_loose_bytes(), before_loose);
            assert_eq!(rebar_shadow_bytes(), before_shadow);
        }
    }

    #[cfg(windows)]
    #[test]
    fn rebar_mark_dirty_spans_restores_failed_readback_pieces() {
        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        let Some(h) = host() else {
            return;
        };
        let len = (3 * PAGE) as u64;
        let (buf, ptr) = h.shared_buffer(len).expect("shared buffer");
        // SAFETY: live D3D12 buffer.
        let gpu = unsafe { buf.GetGPUVirtualAddress() };
        let shadow = Shadow::new_rebar_for_test(buf, ptr, gpu, 0x4000_0000, len);
        // Seed mirror so `prepare_rebar_hashes` initializes the hash UAV & dirty bitset.
        assert!(shadow.writing(0x4000_0000, PAGE, |_, n| Some(vec![0u8; n as usize])));
        let initial = shadow
            .prepare_rebar_hashes(|n| h.default_buffer(n))
            .expect("initial rebar hashes");
        assert_eq!(initial.pre_runs, vec![(0, 3)]);
        let clean = shadow
            .prepare_rebar_hashes(|n| h.default_buffer(n))
            .expect("clean rebar hashes");
        assert!(clean.pre_runs.is_empty());

        // Simulate readback failure restoring dirty spans across piece 0 and piece 2.
        shadow.mark_rebar_dirty_spans(&[(16, 64), (2 * PAGE + 8, 2 * PAGE + 32)]);
        let restored = shadow
            .prepare_rebar_hashes(|n| h.default_buffer(n))
            .expect("restored rebar hashes");
        assert_eq!(restored.pre_runs, vec![(0, 1), (2, 1)]);
        shadow.clear_pending_rebar();
        drop(shadow);
    }

    #[cfg(windows)]
    #[test]
    fn rebar_failed_copy_back_keeps_pending_and_preserves_gpu_bytes_on_refresh() {
        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        let Some(h) = host() else {
            return;
        };
        let title = vec![0x11u8; 4096];
        let start = title.as_ptr() as u64;
        let len = title.len() as u64;
        let (buf, ptr) = h.shared_buffer(len).expect("shared buffer");
        // SAFETY: live D3D12 buffer.
        let gpu = unsafe { buf.GetGPUVirtualAddress() };
        let shadow = Shadow::new_rebar_for_test(buf, ptr, gpu, start, len);
        // Simulate an unwatched buffer whose GPU bytes were written to `[64..128)`.
        shadow.watched.store(NOT_WATCHED, Ordering::Relaxed);
        // SAFETY: `ptr` is mapped for `len` bytes of shared RAM in this test.
        unsafe {
            std::ptr::write_bytes(ptr, 0x22, len as usize);
            std::ptr::write_bytes(ptr.add(64), 0xAA, 64);
        }

        // If initial mirror readback fails, `writing` returns `false`, `copy_back_rebar`
        // returns `None`, and `pending` stays intact so a later unwatched refresh does
        // not overwrite `[64..128)` with stale title bytes (`0x11`).
        assert!(!shadow.writing(start as usize + 64, 64, |_, _| None));
        assert!(shadow.has_pending());
        assert!(
            shadow
                .copy_back_rebar(&[(64, 128)], &[&[0xAA; 64]])
                .is_none()
        );
        assert!(shadow.has_pending());

        shadow.update(start as usize, len as usize);
        let bytes = shadow.read_test_bytes(start, len as usize);
        assert!(bytes[..64].iter().all(|&b| b == 0x11));
        assert!(
            bytes[64..128].iter().all(|&b| b == 0xAA),
            "pending GPU result must survive unwatched refresh when copy-back failed"
        );
        assert!(bytes[128..].iter().all(|&b| b == 0x11));
    }

    #[cfg(windows)]
    #[test]
    fn rebar_shadow_gpu_write_round_trips_and_preserves_concurrent_cpu_writes() {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_DEST;

        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        let Some(h) = host() else {
            return;
        };
        if h.rebar_shadow_budget() == 0 {
            // Requires a discrete GPU with D3D12 GPU upload heap (ReBAR) support.
            return;
        }
        let mut backing = vec![0x11u8; 4096];
        let start = backing.as_ptr() as u64;
        let resource = 0xdead_beef_0005_usize;
        register(resource, start, backing.len() as u64);

        let shadow = find(start, backing.len() as u64).expect("shadow");
        assert!(
            shadow.is_rebar(),
            "expected ReBAR-backed shadow on ReBAR adapter"
        );

        let list = h.new_list().expect("host list");
        // Seed initial title bytes, declare overlapping GPU writes, and overwrite
        // bytes 16..32 on the GPU via CopyBufferRegion while the CPU concurrently
        // writes byte 64 in the same page.
        shadow.refresh_as(
            &list,
            start,
            backing.len() as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        shadow.gpu_writes(&list, start, 128);
        shadow.gpu_writes(&list, start + 16, 128);

        let (src_buf, src_ptr) = h.upload_buffer(16).expect("upload");
        // SAFETY: `src_ptr` is mapped for 16 bytes.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0xAA, 16);
        }
        // SAFETY: live buffers and valid 16-byte copy inside both.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, 16, &src_buf, 0, 16);
        }
        // Simulate a concurrent CPU write to byte 64 while the GPU runs.
        let concurrent_addr = backing.as_mut_ptr() as usize;
        list.defer_host(move || {
            // SAFETY: `backing` stays alive for the duration of this test.
            unsafe {
                *(concurrent_addr as *mut u8).add(64) = 0x77;
            }
        });
        list.close();
        h.execute(&[&list]);

        assert!(backing[..16].iter().all(|&b| b == 0x11));
        assert!(backing[16..32].iter().all(|&b| b == 0xAA));
        assert_eq!(backing[64], 0x77);
        assert!(backing[65..128].iter().all(|&b| b == 0x11));
        // Also verify the initial 0x11 bytes ACTUALLY arrived in VRAM (not zeroed
        // by ReadProcessMemory failing on a PCIe BAR destination!).
        let rebar_mirror = shadow
            .rebar
            .as_ref()
            .unwrap()
            .mirror
            .lock()
            .unwrap()
            .clone()
            .expect("mirror populated after gpu_writes");
        assert!(
            rebar_mirror[..16].iter().all(|&b| b == 0x11),
            "initial title bytes did not reach ReBAR VRAM! got: {:02x?}",
            &rebar_mirror[..16]
        );

        list.reset();
        drop(shadow);
        unregister(resource, start);
    }

    #[test]
    fn rebar_dirty_bits_marks_and_coalesces_across_word_boundaries() {
        let mut bits = RebarDirtyBits::new(140);
        assert!(bits.take_runs().is_empty());

        // Single piece, cross-word range (62..67), and tail clamped at 140.
        bits.mark_pieces(3, 1);
        bits.mark_pieces(62, 5);
        bits.mark_pieces(138, 10);
        assert_eq!(bits.take_runs(), vec![(3, 1), (62, 5), (138, 2)]);
        assert!(bits.take_runs().is_empty());

        // Byte range straddling a 64 KiB boundary marks both pieces.
        bits.mark_byte_range(PAGE - 8, 16);
        bits.mark_byte_range(5 * PAGE, PAGE);
        assert_eq!(bits.take_runs(), vec![(0, 2), (5, 1)]);
        assert!(bits.take_runs().is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn rebar_gpu_hash_compare_skips_unchanged_pieces_and_merges_concurrent_writes() {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_DEST;

        let _rebar = REBAR_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let _retire = crate::host::isolated_retire_scope();
        let Some(h) = host() else {
            return; // no host GPU here
        };
        if h.rebar_shadow_budget() == 0 {
            // Requires a discrete GPU with D3D12 GPU upload heap (ReBAR) support.
            return;
        }
        let errors_before = h.d3d12_errors();
        let mut backing = vec![0x11u8; 4 * PAGE];
        let start = backing.as_ptr() as u64;
        let resource = 0xdead_beef_0004_usize;
        register(resource, start, backing.len() as u64);

        let shadow = find(start, backing.len() as u64).expect("shadow");
        assert!(
            shadow.is_rebar(),
            "expected ReBAR-backed shadow on ReBAR adapter"
        );

        let (src_aa, ptr_aa) = h.upload_buffer(16).expect("upload AA");
        let (src_bb, ptr_bb) = h.upload_buffer(16).expect("upload BB");
        let (src_11, ptr_11) = h.upload_buffer(16).expect("upload 11");
        // SAFETY: each upload buffer is mapped for 16 bytes.
        unsafe {
            std::ptr::write_bytes(ptr_aa, 0xAA, 16);
            std::ptr::write_bytes(ptr_bb, 0xBB, 16);
            std::ptr::write_bytes(ptr_11, 0x11, 16);
        }

        // Submission 1: declare the entire 256 KiB buffer as UAV-written, but
        // only modify piece 1 (+32..+48) and piece 3 (+64..+80) on the GPU
        // while the CPU concurrently writes piece 1 (+128) and piece 0 (+128).
        let stats0 = h.rebar_hash_stats();
        let list = h.new_list().expect("host list");
        shadow.refresh_as(
            &list,
            start,
            backing.len() as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        shadow.gpu_writes(&list, start, backing.len() as u64);
        // SAFETY: live buffers and in-bounds 16-byte copies.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, (PAGE + 32) as u64, &src_aa, 0, 16);
            list.list
                .CopyBufferRegion(&shadow.buffer, (3 * PAGE + 64) as u64, &src_bb, 0, 16);
        }
        let base_addr = backing.as_mut_ptr() as usize;
        list.defer_host(move || {
            // SAFETY: `backing` stays alive for the duration of this test.
            unsafe {
                *(base_addr as *mut u8).add(128) = 0x55;
                *(base_addr as *mut u8).add(PAGE + 128) = 0x77;
            }
        });
        list.close();
        h.execute(&[&list]);
        list.reset();

        let stats1 = h.rebar_hash_stats();
        assert_eq!(stats1.compared_pieces - stats0.compared_pieces, 4);
        assert_eq!(
            stats1.changed_pieces - stats0.changed_pieces,
            2,
            "pieces 0 and 2 were untouched by the GPU and must be skipped"
        );
        assert_eq!(stats1.post_readback_calls - stats0.post_readback_calls, 1);
        assert_eq!(
            stats1.post_readback_bytes - stats0.post_readback_bytes,
            (2 * PAGE) as u64,
            "only the 2 changed 64 KiB pieces should be DMAed back"
        );
        assert_eq!(backing[128], 0x55);
        assert!(backing[PAGE + 32..PAGE + 48].iter().all(|&b| b == 0xAA));
        assert_eq!(backing[PAGE + 128], 0x77);
        assert!(backing[2 * PAGE..3 * PAGE].iter().all(|&b| b == 0x11));
        assert!(
            backing[3 * PAGE + 64..3 * PAGE + 80]
                .iter()
                .all(|&b| b == 0xBB)
        );

        // Submission 2: speculative UAV over all 256 KiB with zero GPU writes.
        // Piece 1 was pushed back to VRAM at the end of submission 1 (due to
        // the concurrent 0x77 write), so pre-hash must re-hash piece 1 and
        // post-compare must report 0 changed pieces and 0 readback DMAs.
        shadow.refresh_as(
            &list,
            start,
            backing.len() as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        shadow.gpu_writes(&list, start, backing.len() as u64);
        list.close();
        h.execute(&[&list]);
        list.reset();

        let stats2 = h.rebar_hash_stats();
        assert_eq!(stats2.compared_pieces - stats1.compared_pieces, 4);
        assert_eq!(stats2.changed_pieces - stats1.changed_pieces, 0);
        assert_eq!(
            stats2.post_readback_calls - stats1.post_readback_calls,
            0,
            "0 changed pieces must trigger 0 readback DMAs"
        );
        assert_eq!(stats2.post_readback_bytes - stats1.post_readback_bytes, 0);
        assert_eq!(backing[128], 0x55);
        assert_eq!(backing[PAGE + 128], 0x77);

        // Submission 3: GPU overwrites byte PAGE + 128 back to 0x11 (replacing
        // the concurrent CPU 0x77 write). Only piece 1 must be flagged & DMAed.
        shadow.refresh_as(
            &list,
            start,
            backing.len() as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        shadow.gpu_writes(&list, start, backing.len() as u64);
        // SAFETY: live buffers and in-bounds 16-byte copy.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, (PAGE + 128) as u64, &src_11, 0, 16);
        }
        list.close();
        h.execute(&[&list]);
        list.reset();

        let stats3 = h.rebar_hash_stats();
        assert_eq!(stats3.compared_pieces - stats2.compared_pieces, 4);
        assert_eq!(stats3.changed_pieces - stats2.changed_pieces, 1);
        assert_eq!(
            stats3.post_readback_bytes - stats2.post_readback_bytes,
            PAGE as u64
        );
        assert!(backing[PAGE + 128..PAGE + 144].iter().all(|&b| b == 0x11));

        // Submission 4: CPU modifies piece 2 and refreshes it, then declares a
        // speculative UAV write with no GPU writes. Pre-hash must refresh piece
        // 2's hash so post-compare sees 0 changes and 0 readback DMAs.
        backing[2 * PAGE + 16..2 * PAGE + 32].fill(0xCC);
        shadow.refresh_as(
            &list,
            start,
            backing.len() as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        shadow.gpu_writes(&list, start, backing.len() as u64);
        list.close();
        h.execute(&[&list]);
        list.reset();

        let stats4 = h.rebar_hash_stats();
        assert_eq!(stats4.compared_pieces - stats3.compared_pieces, 4);
        assert_eq!(stats4.changed_pieces - stats3.changed_pieces, 0);
        assert_eq!(stats4.post_readback_calls - stats3.post_readback_calls, 0);
        assert_eq!(stats4.post_readback_bytes - stats3.post_readback_bytes, 0);
        assert!(
            backing[2 * PAGE + 16..2 * PAGE + 32]
                .iter()
                .all(|&b| b == 0xCC)
        );
        assert_eq!(
            h.d3d12_errors(),
            errors_before,
            "unexpected D3D12 debug layer error during ReBAR hash-compare test"
        );

        drop(shadow);
        unregister(resource, start);
    }

    #[cfg(windows)]
    #[test]
    fn concurrent_find_reads_share_the_same_shadow_under_rwlock() {
        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let data = vec![0u8; 8192];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0006_usize;
        register(resource, start, data.len() as u64);
        let first = find(start, 64).expect("initial shadow");
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let addr = start + (i * 256) as u64;
                std::thread::spawn(move || find(addr, 64).expect("concurrent shadow"))
            })
            .collect();
        for h in handles {
            let got = h.join().expect("thread");
            assert!(Arc::ptr_eq(&first, &got));
        }
        drop(first);
        unregister(resource, start);
    }

    /// Never-copied pages (`copied == 0`) inside a range must still be
    /// reported as stale even when absent from the write-watch journal,
    /// and leading entries before the range must not stall the scan.
    #[test]
    fn stale_pages_include_uncopied_pages_absent_from_write_watch() {
        let base = 0x100usize;
        // Pages 0..4: page 0 copied at t=5, page 1 never copied (0), page 2
        // copied at t=5, page 3 copied at t=10.
        let copied = [5u64, 0, 5, 10];
        let a = base * PAGE;
        let e = (base + 4) * PAGE;
        // Write-watch includes a leading out-of-range page plus pages 2 and 3.
        let written = [(base - 1, 99u64), (base + 2, 7u64), (base + 3, 8u64)];
        assert_eq!(
            stale_pages_in(base, &copied, a, e, &written),
            vec![base + 1, base + 2]
        );
        // Even with an empty write-watch delta, uncopied page 1 is still stale.
        assert_eq!(stale_pages_in(base, &copied, a, e, &[]), vec![base + 1]);
        // Once all pages are copied and clean, nothing is stale.
        assert!(stale_pages_in(base, &[10, 10, 10, 10], a, e, &written).is_empty());
    }

    #[test]
    fn loose_budget_scales_across_ram_tiers() {
        assert_eq!(loose_budget_for_ram(None), 256 * MIB);
        assert_eq!(loose_budget_for_ram(Some(8 * GIB)), 128 * MIB);
        assert_eq!(loose_budget_for_ram(Some(16 * GIB)), 128 * MIB);
        assert_eq!(loose_budget_for_ram(Some(16 * GIB + 1)), 256 * MIB);
        assert_eq!(loose_budget_for_ram(Some(32 * GIB)), 256 * MIB);
        assert_eq!(loose_budget_for_ram(Some(32 * GIB + 1)), 512 * MIB);
        assert_eq!(loose_budget_for_ram(Some(64 * GIB)), 512 * MIB);
    }

    #[cfg(windows)]
    #[test]
    fn loose_evicts_lru_idle_windows_over_budget_and_skips_in_use_or_pending() {
        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let cache = RwLock::new(LooseCache::new());
        let budget = 2 * WINDOW;
        let base = 0x1000_0000_0000_u64;
        let win = |i: u64| base + i * WINDOW;

        let weak0 = Arc::downgrade(&loose_in(&cache, win(0), win(0) + 64, budget).expect("win 0"));
        let weak1 = Arc::downgrade(&loose_in(&cache, win(1), win(1) + 64, budget).expect("win 1"));
        assert_eq!(weak0.strong_count(), 1);
        assert_eq!(weak1.strong_count(), 1);

        // Touch window 0 so window 1 becomes the least-recently-used entry.
        drop(loose_in(&cache, win(0), win(0) + 64, budget).expect("touch win 0"));

        // Inserting window 2 exceeds the 2-window budget and immediately drops
        // idle LRU window 1 through `retire`.
        let weak2 = Arc::downgrade(&loose_in(&cache, win(2), win(2) + 64, budget).expect("win 2"));
        assert_eq!(weak1.strong_count(), 0, "idle LRU window 1 must be evicted");
        assert_eq!(weak0.strong_count(), 1, "MRU window 0 must stay cached");
        assert_eq!(weak2.strong_count(), 1, "new window 2 must stay cached");
        {
            let guard = cache.read().unwrap();
            assert_eq!(guard.total_bytes, budget);
            assert_eq!(guard.entries.len(), 2);
        }

        // With an active epoch, evicted window 0 leaves the cache immediately
        // but stays alive in the retirement queue until the epoch completes.
        let list = host.new_list().expect("host list");
        list.mark_work();
        let weak3 = Arc::downgrade(&loose_in(&cache, win(3), win(3) + 64, budget).expect("win 3"));
        {
            let guard = cache.read().unwrap();
            assert!(!guard.entries.contains_key(&(win(0), WINDOW)));
            assert_eq!(guard.total_bytes, budget);
        }
        assert_eq!(
            weak0.strong_count(),
            1,
            "evicted loose window must wait for active epochs"
        );
        list.close();
        list.reset();
        assert_eq!(
            weak0.strong_count(),
            0,
            "evicted loose window must drop once active epochs finish"
        );

        // Pin window 2 via an external Arc (strong_count > 1) and window 3 via
        // a pending GPU UAV copy-back range; neither may be evicted.
        let held2 = loose_in(&cache, win(2), win(2) + 64, budget).expect("hold win 2");
        {
            let s3 = loose_in(&cache, win(3), win(3) + 64, budget).expect("mark win 3 pending");
            s3.pending
                .lock()
                .unwrap()
                .push((win(3) as usize, win(3) as usize + 64));
        }
        let weak4 = Arc::downgrade(&loose_in(&cache, win(4), win(4) + 64, budget).expect("win 4"));
        assert_eq!(
            weak2.strong_count(),
            2,
            "in-use window 2 must not be evicted"
        );
        assert_eq!(
            weak3.strong_count(),
            1,
            "window 3 with pending copy-back must not be evicted"
        );
        assert_eq!(weak4.strong_count(), 1);
        {
            let guard = cache.read().unwrap();
            assert_eq!(guard.entries.len(), 3);
            assert_eq!(guard.total_bytes, 3 * WINDOW);
        }

        // Once window 2 is released and window 3's pending copy-back clears,
        // the next insert drains both stale windows back down to budget.
        drop(held2);
        weak3
            .upgrade()
            .expect("win 3 still live")
            .pending
            .lock()
            .unwrap()
            .clear();
        let weak5 = Arc::downgrade(&loose_in(&cache, win(5), win(5) + 64, budget).expect("win 5"));
        assert_eq!(weak2.strong_count(), 0);
        assert_eq!(weak3.strong_count(), 0);
        assert_eq!(weak4.strong_count(), 1);
        assert_eq!(weak5.strong_count(), 1);
        {
            let guard = cache.read().unwrap();
            assert_eq!(guard.entries.len(), 2);
            assert_eq!(guard.total_bytes, budget);
        }

        // Re-accessing previously evicted window 1 cleanly recreates its shadow.
        let recreated =
            loose_in(&cache, win(1) + 128, win(1) + 256, budget).expect("recreate win 1");
        assert_eq!(recreated.start, win(1));
        assert_eq!(recreated.len, WINDOW);
        assert_ne!(recreated.gpu(win(1) + 128), 0);
        assert_eq!(weak4.strong_count(), 0);
        assert_eq!(weak5.strong_count(), 1);
        drop(recreated);
        {
            let guard = cache.read().unwrap();
            assert_eq!(guard.entries.len(), 2);
            assert_eq!(guard.total_bytes, budget);
            assert!(guard.entries.contains_key(&(win(1), WINDOW)));
        }
    }

    #[test]
    fn wide_copy_is_default_on_multicore_and_can_be_overridden() {
        assert!(wide_copy_enabled(None, Some(4)));
        assert!(wide_copy_enabled(None, Some(16)));
        assert!(!wide_copy_enabled(None, Some(2)));
        assert!(!wide_copy_enabled(None, None));
        assert!(wide_copy_enabled(Some("1"), Some(1)));
        assert!(!wide_copy_enabled(Some("0"), Some(16)));
        assert!(wide_copy_enabled(Some("auto"), Some(4)));
    }

    #[test]
    fn chunk_copy_segs_preserves_boundaries_and_mirror_offsets() {
        let segs = [
            CopySeg {
                src: 0x1000,
                dst: 0x2000,
                mirror: 0x3000,
                len: 700_000,
            },
            CopySeg {
                src: 0x1000 + 800_000,
                dst: 0x2000 + 800_000,
                mirror: 0x3000 + 800_000,
                len: 123,
            },
            CopySeg {
                src: 0x1000 + 900_000,
                dst: 0x2000 + 900_000,
                mirror: 0,
                len: 1_500_000,
            },
        ];
        let buckets = chunk_copy_segs(&segs, 256 << 10);
        assert!(buckets.len() >= 8);
        let mut flat: Vec<CopySeg> = Vec::new();
        for bucket in &buckets {
            let sum: usize = bucket.iter().map(|s| s.len).sum();
            assert!(sum > 0 && sum <= 256 << 10);
            flat.extend_from_slice(bucket);
        }
        let mut merged: Vec<CopySeg> = Vec::new();
        for seg in flat {
            match merged.last_mut() {
                Some(last)
                    if last.src + last.len == seg.src
                        && last.dst + last.len == seg.dst
                        && ((last.mirror == 0 && seg.mirror == 0)
                            || (last.mirror != 0 && last.mirror + last.len == seg.mirror)) =>
                {
                    last.len += seg.len;
                }
                _ => merged.push(seg),
            }
        }
        assert_eq!(merged, segs);
    }

    #[test]
    fn parallel_shadow_update_preserves_boundaries_unmapped_pages_and_pending_gaps() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        let Some(h) = host() else {
            return;
        };
        const MEM_TOP_DOWN: u32 = 0x10_0000;
        const LEN: usize = 4 << 20; // 4 MiB (> WIDE_COPY_MIN)
        // SAFETY: reserve a 4 MiB address range and commit parts around a hole.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                LEN,
                MEM_RESERVE | MEM_TOP_DOWN,
                PAGE_NOACCESS,
            )
            .cast::<u8>()
        };
        assert!(!base.is_null());
        let mut expected = vec![0u8; LEN];
        // Leave [1 MiB, 1 MiB + 128 KiB) unmapped (hole) and keep [2 MiB, 2 MiB + 128 KiB)
        // as a pending shader-write range that `update` must skip completely.
        let hole = (1 << 20, (1 << 20) + 2 * PAGE);
        let pending = (2 << 20, (2 << 20) + 2 * PAGE);
        for &(from, to) in &[(0usize, hole.0), (hole.1, LEN)] {
            // SAFETY: committing subranges inside our reserved allocation.
            let p = unsafe {
                VirtualAlloc(base.add(from).cast(), to - from, MEM_COMMIT, PAGE_READWRITE)
            };
            assert_eq!(p, base.wrapping_add(from).cast());
            for (off, dst_byte) in expected.iter_mut().enumerate().take(to).skip(from) {
                let v = ((off.wrapping_mul(31) ^ (off >> 12)) & 0xff) as u8;
                // SAFETY: committed above.
                unsafe { *base.add(off) = v };
                *dst_byte = v;
            }
        }

        let Some((buffer, data)) = h.shared_buffer(LEN as u64) else {
            // SAFETY: freeing our test allocation.
            unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
            return;
        };
        // Pre-fill shadow staging with a sentinel so untouched regions are verifiable.
        // SAFETY: `data` maps `LEN` bytes.
        unsafe { std::ptr::write_bytes(data, 0xA5, LEN) };
        // SAFETY: live D3D12 buffer.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu_va, base as u64, LEN as u64, None);
        shadow
            .pending
            .lock()
            .unwrap()
            .push((base as usize + pending.0, base as usize + pending.1));

        // Refresh [37, LEN - 41): tests unaligned ends, hole zeroing, pending gap,
        // and multi-chunk parallel copy.
        let req_lo = 37usize;
        let req_hi = LEN - 41;
        let copied = shadow.update(base as usize + req_lo, req_hi - req_lo);
        assert_eq!(copied, (pending.0 - req_lo) + (req_hi - pending.1));

        // SAFETY: `data` maps `LEN` bytes and `update` has finished all workers.
        let got = unsafe { std::slice::from_raw_parts(data, LEN) };
        assert!(
            got[..req_lo].iter().all(|&b| b == 0xA5),
            "bytes before refreshed start must stay untouched"
        );
        assert!(
            got[req_hi..].iter().all(|&b| b == 0xA5),
            "bytes after refreshed end must stay untouched"
        );
        assert!(
            got[pending.0..pending.1].iter().all(|&b| b == 0xA5),
            "pending shader-write range must stay untouched"
        );
        assert_eq!(got[req_lo..hole.0], expected[req_lo..hole.0]);
        assert!(
            got[hole.0..hole.1].iter().all(|&b| b == 0),
            "unmapped hole must be zeroed"
        );
        assert_eq!(got[hole.1..pending.0], expected[hole.1..pending.0]);
        assert_eq!(got[pending.1..req_hi], expected[pending.1..req_hi]);

        // SAFETY: freeing our test allocation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn parallel_shadow_update_rebar_mirror_syncs_mirror_and_vram() {
        let Some(h) = host() else {
            return;
        };
        const LEN: usize = 3 << 20; // 3 MiB (> WIDE_COPY_MIN)
        let title: Vec<u8> = (0..LEN)
            .map(|i| ((i.wrapping_mul(17) ^ (i >> 8)) & 0xff) as u8)
            .collect();
        let start = title.as_ptr() as u64;
        let Some((buffer, data)) = h.shared_buffer(LEN as u64) else {
            return;
        };
        // SAFETY: `data` maps `LEN` bytes.
        unsafe { std::ptr::write_bytes(data, 0x5A, LEN) };
        // SAFETY: live D3D12 buffer.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu_va, start, LEN as u64, Some(0));
        {
            let rebar = shadow.rebar.as_ref().expect("rebar state");
            *rebar.mirror.lock().unwrap() = Some(vec![0x5Au8; LEN]);
        }

        // Disjoint runs totaling > 2 MiB that straddle chunk cuts.
        let runs = [
            (start as usize + 19, start as usize + (1 << 20) + 123),
            (start as usize + (1 << 20) + PAGE, start as usize + LEN - 29),
        ];
        let copied = shadow.copy_runs(&runs);
        let expected_copied: usize = runs.iter().map(|&(a, b)| b - a).sum();
        assert_eq!(copied, expected_copied);

        let mirror = shadow
            .rebar
            .as_ref()
            .unwrap()
            .mirror
            .lock()
            .unwrap()
            .clone()
            .unwrap();
        // SAFETY: `data` maps `LEN` bytes.
        let vram = unsafe { std::slice::from_raw_parts(data, LEN) };
        let r0 = (runs[0].0 - start as usize, runs[0].1 - start as usize);
        let r1 = (runs[1].0 - start as usize, runs[1].1 - start as usize);
        assert_eq!(mirror[r0.0..r0.1], title[r0.0..r0.1]);
        assert_eq!(vram[r0.0..r0.1], title[r0.0..r0.1]);
        assert_eq!(mirror[r1.0..r1.1], title[r1.0..r1.1]);
        assert_eq!(vram[r1.0..r1.1], title[r1.0..r1.1]);
        assert!(mirror[..r0.0].iter().all(|&b| b == 0x5A));
        assert!(vram[..r0.0].iter().all(|&b| b == 0x5A));
        assert!(mirror[r0.1..r1.0].iter().all(|&b| b == 0x5A));
        assert!(vram[r0.1..r1.0].iter().all(|&b| b == 0x5A));
        assert!(mirror[r1.1..].iter().all(|&b| b == 0x5A));
        assert!(vram[r1.1..].iter().all(|&b| b == 0x5A));
    }

    #[test]
    fn wide_refresh_reentrancy_and_batch_do_not_deadlock() {
        let Some(h) = host() else {
            return;
        };
        const BIG: usize = 2 << 20;
        const SMALL: usize = 64 << 10;
        let mut backing: Vec<Vec<u8>> = Vec::new();
        let mut shadows: Vec<Arc<Shadow>> = Vec::new();
        for i in 0..8usize {
            let len = if i < 4 { BIG } else { SMALL };
            let buf: Vec<u8> = (0..len)
                .map(|j| (((i + 1) * 29 + j * 7 + (j >> 9)) & 0xff) as u8)
                .collect();
            let start = buf.as_ptr() as u64;
            let Some((res, data)) = h.shared_buffer(len as u64) else {
                return;
            };
            // SAFETY: live D3D12 buffer.
            let gpu_va = unsafe { res.GetGPUVirtualAddress() };
            let shadow = Arc::new(Shadow::new(res, data, gpu_va, start, len as u64, None));
            backing.push(buf);
            shadows.push(shadow);
        }

        // 1. Direct re-entrancy stress: invoke multi-MB `Shadow::update` from
        // inside `wide` workers concurrently. Must never deadlock `workers()`.
        let jobs: Vec<Job> = shadows
            .iter()
            .map(|s| {
                let s = s.clone();
                Box::new(move || {
                    let n = s.update(s.start as usize, s.len as usize);
                    assert_eq!(n, s.len as usize);
                }) as Job
            })
            .collect();
        wide(jobs);
        for (shadow, src) in shadows.iter().zip(&backing) {
            // SAFETY: `shadow.data` maps `shadow.len` bytes.
            let got = unsafe { std::slice::from_raw_parts(shadow.data, src.len()) };
            assert_eq!(got, src.as_slice());
        }

        // 2. Two-phase `refresh_batch` mixing big + small shadows (parallel plan
        // + single flat parallel copy pass).
        for shadow in &shadows {
            // SAFETY: clear staging before batch refresh.
            unsafe { std::ptr::write_bytes(shadow.data, 0, shadow.len as usize) };
        }
        let batch_items: Vec<(Arc<Shadow>, usize, usize)> = shadows
            .iter()
            .map(|s| (s.clone(), s.start as usize, (s.start + s.len) as usize))
            .collect();
        let expected_total: usize = backing.iter().map(Vec::len).sum();
        assert_eq!(refresh_batch(batch_items, true), expected_total);
        for (shadow, src) in shadows.iter().zip(&backing) {
            // SAFETY: `shadow.data` maps `shadow.len` bytes.
            let got = unsafe { std::slice::from_raw_parts(shadow.data, src.len()) };
            assert_eq!(got, src.as_slice());
        }

        // 3. `copy_mapped_wide` splits a multi-MB `Step::Fill` cleanly.
        let mut wide_dst = vec![0u8; BIG];
        copy_mapped_wide(wide_dst.as_mut_ptr(), backing[0].as_ptr(), BIG);
        assert_eq!(wide_dst, backing[0]);
    }
}
