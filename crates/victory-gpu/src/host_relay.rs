//! Asynchronous GPU submission (`execute_late`, `Flight`, `Line`), VRAM-backed
//! `Card` shadow compare/fetch passes (`CARD_COMPARE`, `CARD_FETCH`), dedicated
//! `victory-relay` copy-back worker (`VICTORY_RELAY`), unified `COPY`-queue
//! card+ReBAR DMA readback, `fill_ahead`, and async `land()` offload to
//! `victory-late-wait`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, Once, OnceLock};

use windows::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

use super::{
    COPY_BACK, Host, HostList, Prelude, SUBMISSION_US, Step, fail, host, last_notes, merge_ranges,
    name, read_backs, read_into_raw, transition,
};

/// Sentinel `mark_base` passed to [`CARD_COMPARE`] when view-mark filtering is
/// inactive.
const VIEW_MARK_NONE: u32 = u32::MAX;

/// Monotonic count of submissions started (`Host::submit`).
pub(super) static STARTED: AtomicU64 = AtomicU64::new(0);
/// Highest submission number that has completed on the GPU.
pub(super) static FINISHED: AtomicU64 = AtomicU64::new(0);

/// What the host keeps for copies in the card's own memory
/// ([`crate::shadow::cards`]): three lists of ours and the compare.
struct CardWork {
    /// Before a submission's lists: what changed on the copies' CPU side
    /// goes to their card side. Those not in use: a plan takes one and
    /// gives it back when its submission has run (`Host::card_done`), so
    /// that submissions sent ahead ([`late`]) each have their own.
    sync: Vec<Prelude>,
    /// After them, in the same submission: the compare of the declared
    /// ranges with their bytes from before. As `sync`.
    check: Vec<Prelude>,
    /// After it ran: the pieces that differ come down.
    fetch: Prelude,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// One dword per piece compared, set where the piece differs: PC
    /// memory, mapped (buffer, mapping, bytes).
    differs: (ID3D12Resource, usize, u64),
    /// Where the pieces that differ are fetched to: PC memory, mapped.
    fetched: Option<(ID3D12Resource, usize, u64)>,
    /// The dwords of `differs` taken by plans whose results were not read
    /// yet.
    taken: u32,
    /// Plans made and not done with (`Host::card_done`).
    out: u32,
    /// The fetch inside flights ([`inflight_fetch`]), if it is on and was
    /// made.
    inflight: Option<CardInflight>,
}

/// What the fetch inside flights needs ([`inflight_fetch`], [`CARD_FETCH`]).
struct CardInflight {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// PC memory, mapped (buffer, mapping, bytes): dword 0 the counter of
    /// tickets, then one dword per dword of `CardWork::differs`, the entry
    /// of that piece.
    table: (ID3D12Resource, usize, u64),
    /// PC memory, mapped (buffer, mapping, slots of [`INFLIGHT_SLOT`]
    /// bytes): where the pieces come down.
    fetched: (ID3D12Resource, usize, u32),
    /// Every ticket below it belongs to a landed flight: its slot may be
    /// written over. Raised by the thread that lands flights; read when a
    /// compare's list is recorded (an older value only gives fewer slots).
    landed: u32,
    /// What fetches the old way brought (copy, offsets, and the highest
    /// send whose lists were surely on the queue ahead of the fetch: its
    /// order of the queue less one).
    brought: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)>,
}

// SAFETY: D3D12 objects (free-threaded) and persistent mappings, used only
// under `CARD_WORK` and synchronized by GPU fences.
unsafe impl Send for CardWork {}

static CARD_WORK: Mutex<Option<CardWork>> = Mutex::new(None);

/// The compare of a declared range of a copy in the card's memory with
/// its bytes from before, one 64-thread group per 64 KiB piece (cut at
/// multiples of 64 KiB of title memory, as `Shadow::card_piece` cuts them):
/// each lane strides through the piece 1024 bytes at a time for coalesced
/// 16-byte loads, then the group reduces the 64 partial hashes in shared
/// memory. `differs` gets a dword per piece, 1 where a byte differs.
pub(crate) const CARD_COMPARE: &str = r#"
#define RS "RootConstants(num32BitConstants=7, b0), SRV(t0), UAV(u0), UAV(u1), UAV(u2)"
cbuffer Span : register(b0) {
    uint first;     // the first piece
    uint skew;      // where the copy starts in its piece 0
    uint size;      // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;    // the first piece's dword in `differs`, in bytes
    uint compare;   // 0: the hashes are stored; else compared with those stored (2: and then stored)
    uint mark_base; // base bit in `marks` (0xFFFFFFFFu = no view-mark filter)
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
RWByteAddressBuffer marks : register(u2);
groupshared uint4 shared_h[64];
uint4 mix_pair(uint4 a, uint4 b) {
    uint4 x = (a ^ (b * uint4(0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du, 0x27D4EB2Fu)))
            * uint4(0x01000193u, 0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u);
    return x.yzwx ^ (x >> 15);
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint idx = gid.x;
    if (idx >= pieces) return;
    uint lane = gtid.x;
    uint piece = first + idx;
    bool active = true;
    if (compare != 0u && mark_base != 0xFFFFFFFFu) {
        uint bit = min(mark_base + piece, 4194303u);
        uint word = marks.Load((bit >> 5u) << 2u);
        active = ((word & (1u << (bit & 31u))) != 0u);
    }
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu) ^ (lane * 0x9E3779B9u);
    if (active) {
        uint lo = piece * 65536u;
        lo = lo > skew ? lo - skew : 0u;
        uint hi = min((piece + 1u) * 65536u - skew, size);
        lo = lo & ~15u;
        hi = (hi + 15u) & ~15u;
        [loop]
        for (uint at = lo + lane * 16u; at < hi; at += 1024u) {
            h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
            h = h.yzwx ^ (h >> 15);
        }
    }
    shared_h[lane] = h;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 32u; stride > 1u; stride >>= 1u) {
        if (active && lane < stride) {
            shared_h[lane] = mix_pair(shared_h[lane], shared_h[lane + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        if (!active) {
            differs.Store(out_at + idx * 4u, 0u);
        } else {
            uint4 final_h = mix_pair(shared_h[0], shared_h[1]);
            if (compare != 0u) {
                differs.Store(out_at + idx * 4u, any(final_h != hashes.Load4(piece * 16u)) ? 1u : 0u);
                if (compare == 2u) hashes.Store4(piece * 16u, final_h);
            } else {
                hashes.Store4(piece * 16u, final_h);
            }
        }
    }
}
"#;

/// Dispatch [`CARD_COMPARE`] for `pieces` pieces (one thread group per piece),
/// chunking at D3D12's 65,535-group per-dimension limit if ever exceeded.
///
/// # Safety
/// The caller must have bound [`CARD_COMPARE`]'s root signature, pipeline,
/// SRV, and UAVs on `list`, and `list` must be open for recording.
#[allow(clippy::too_many_arguments)]
unsafe fn dispatch_card_compare(
    list: &ID3D12GraphicsCommandList,
    first: u32,
    skew: u32,
    size: u32,
    pieces: u32,
    out_at_bytes: u32,
    compare: u32,
    mark_base: u32,
) {
    const MAX_GROUPS: u32 = 65_535;
    let mut done = 0u32;
    while done < pieces {
        let chunk = (pieces - done).min(MAX_GROUPS);
        let constants = [
            first + done,
            skew,
            size,
            chunk,
            out_at_bytes + done * 4,
            compare,
            mark_base,
        ];
        // SAFETY: caller guarantees `list` is open with CARD_COMPARE bound.
        unsafe {
            list.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
            list.Dispatch(chunk, 1, 1);
        }
        done += chunk;
    }
}

/// Whether the pieces a flight's compare flags come down to the PC inside
/// the flight itself (`VICTORY_CARD_FETCH_INFLIGHT=1`; off by default, and
/// only with [`relay`]).
fn inflight_fetch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        relay() && std::env::var_os("VICTORY_CARD_FETCH_INFLIGHT").is_some_and(|v| v == "1")
    })
}

/// The bytes of the PC's memory pieces fetched inside flights go to
/// ([`inflight_fetch`]; `VICTORY_CARD_FETCH_INFLIGHT_MB`, default 64 MiB).
fn inflight_fetch_bytes() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_FETCH_INFLIGHT_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(64)
            .max(1)
            << 20
    })
}

/// A slot of [`CARD_FETCH`]: a 64 KiB piece as the compare cuts it, widened
/// to 16 bytes at both ends.
pub(crate) const INFLIGHT_SLOT: u64 = (64 << 10) + 16;

/// The slots for `bytes` asked of [`inflight_fetch_bytes`]: one per 64 KiB,
/// rounded down to a power of two (at least 1, at most 2^15).
pub(crate) fn inflight_slots(bytes: u64) -> u32 {
    let wanted = (bytes >> 16).clamp(1, 1 << 15);
    1u32 << (63 - wanted.leading_zeros())
}

/// [`CARD_FETCH`]'s table entry for a piece the compare did not flag.
pub(crate) const INFLIGHT_NONE: u32 = 0xffff_ffff;

/// Where piece `i` of `span` starts in its [`CARD_FETCH`] slot, as an
/// offset of the copy.
pub(crate) fn inflight_piece_from(span: &crate::shadow::CardSpan, i: u32) -> usize {
    let lo = (span.first + i) as usize * 65536;
    let lo = lo.saturating_sub(span.skew as usize);
    lo & !15
}

/// Whether the ticket `k` of [`CARD_FETCH`]'s counter gets a slot, with
/// every ticket below `floor` landed and `slots` slots.
pub(crate) fn inflight_gets_slot(k: u32, floor: u32, slots: u32) -> bool {
    k.wrapping_sub(floor) < slots
}

/// The fetch inside a flight ([`inflight_fetch`]), run in the compare's list
/// right behind the compare of a declared range (a UAV barrier between).
pub(crate) const CARD_FETCH: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), SRV(t0), UAV(u0), UAV(u1), UAV(u2)"
#define SLOT 65552u
#define NONE 0xffffffffu
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint landed;   // every ticket below it is landed
    uint slots;    // slots of `fetched`
    uint groups;   // groups dispatched
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer differs : register(u0);
RWByteAddressBuffer table : register(u1);
RWByteAddressBuffer fetched : register(u2);
groupshared uint ticket;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint t : SV_GroupIndex) {
    [loop]
    for (uint i = group.x; i < pieces; i += groups) {
        if (t == 0u) {
            uint entry = NONE;
            if (differs.Load(out_at + i * 4u) != 0u) {
                uint k;
                table.InterlockedAdd(0, 1u, k);
                entry = k;
            }
            table.Store(4u + out_at + i * 4u, entry);
            ticket = entry;
        }
        GroupMemoryBarrierWithGroupSync();
        uint k = ticket;
        GroupMemoryBarrierWithGroupSync();
        if (k != NONE && (k - landed) < slots) {
            uint piece = first + i;
            uint lo = piece * 65536u;
            lo = lo > skew ? lo - skew : 0u;
            uint hi = min((piece + 1u) * 65536u - skew, size);
            lo = lo & ~15u;
            hi = (hi + 15u) & ~15u;
            uint into = (k % slots) * SLOT;
            [loop]
            for (uint at = lo + t * 16u; at < hi; at += 1024u) {
                fetched.Store4(into + (at - lo), now.Load4(at));
            }
        }
    }
}
"#;

/// Whether the pieces hashed again after ranges were sent up to a copy in
/// the card's memory are hashed in coalesced piece runs
/// (on by default; `VICTORY_CARD_REHASH_RUNS=0` disables), and not in a pass
/// per range sent up.
pub(super) fn rehash_runs_enabled(setting: Option<&str>) -> bool {
    setting.is_none_or(|v| v != "0")
}

fn rehash_runs() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        rehash_runs_enabled(std::env::var("VICTORY_CARD_REHASH_RUNS").ok().as_deref())
    })
}

/// Since the last frame-rate line, of our own passes over copies in the
/// card's memory: hash passes after ranges were sent up, the pieces they
/// hashed, and compare passes after the lists.
pub(super) static CARD_PASSES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Whether what shaders wrote to copies in the card's memory is merged
/// into title memory on several threads (`VICTORY_WIDE_MERGE=0` disables;
/// `=1` forces; on by default on 4+ logical CPUs). Used by both synchronous
/// [`Host::card_back`] and the relay thread's [`Host::relay_fetch_back`] /
/// [`Host::relay_card_inflight`].
fn wide_merge() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        wide_merge_enabled(
            std::env::var("VICTORY_WIDE_MERGE").ok().as_deref(),
            std::thread::available_parallelism().ok().map(|n| n.get()),
        )
    })
}

pub(super) fn wide_merge_enabled(setting: Option<&str>, logical_cpus: Option<usize>) -> bool {
    match setting {
        Some("1") => true,
        Some("0") => false,
        _ => logical_cpus.is_some_and(|n| n >= 4),
    }
}

/// One fetched piece of a card-backed shadow ready to merge on the relay thread.
pub(super) struct RelayCardPiece {
    pub(super) shadow: std::sync::Arc<crate::shadow::Shadow>,
    pub(super) lo: usize,
    pub(super) hi: usize,
    /// Pointer to `hi - lo` readable fetched GPU bytes, valid for the duration
    /// of [`relay_merge_card_pieces`].
    pub(super) gpu_ptr: usize,
    pub(super) after: u64,
}

/// Merges fetched card-copy pieces into title memory via
/// [`crate::shadow::Shadow::card_merge_relay`], fanning out across
/// [`crate::shadow::wide_run`] when `parallel` is enabled and there are either
/// multiple distinct card shadows or at least 16 total pieces.
///
/// Any pieces whose title address ranges `[start + lo, start + hi)` overlap
/// (for example, two in-flight plans touching the same 64 KiB piece) are placed
/// in the same worker bucket in their original relative order so concurrent
/// workers only ever mutate strictly disjoint slices of `Shadow::data` and
/// title memory.
pub(super) fn relay_merge_card_pieces(pieces: Vec<RelayCardPiece>, parallel: bool) -> usize {
    if pieces.is_empty() {
        return 0;
    }
    let run_seq = |items: &[RelayCardPiece]| -> usize {
        let mut title = Vec::new();
        let mut written = 0usize;
        for p in items {
            // SAFETY: caller guarantees `p.gpu_ptr` points to `p.hi - p.lo`
            // readable bytes that stay live until this function returns.
            let gpu = unsafe { std::slice::from_raw_parts(p.gpu_ptr as *const u8, p.hi - p.lo) };
            written += p.shadow.card_merge_relay(p.lo, gpu, p.after, &mut title);
        }
        written
    };

    let first_shadow = std::sync::Arc::as_ptr(&pieces[0].shadow) as usize;
    let multi_shadow = pieces
        .iter()
        .skip(1)
        .any(|p| std::sync::Arc::as_ptr(&p.shadow) as usize != first_shadow);
    if !parallel || pieces.len() < 2 || (!multi_shadow && pieces.len() < 16) {
        return run_seq(&pieces);
    }

    // Group overlapping title-address intervals `[start + lo, start + hi)`
    // into disjoint units while preserving original order inside each unit.
    let mut order: Vec<usize> = (0..pieces.len()).collect();
    order.sort_unstable_by_key(|&idx| {
        let p = &pieces[idx];
        ((p.shadow.span().0 as usize).saturating_add(p.lo), idx)
    });
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut cur_end = 0usize;
    for idx in order {
        let p = &pieces[idx];
        let base = p.shadow.span().0 as usize;
        let t_lo = base.saturating_add(p.lo);
        let t_hi = base.saturating_add(p.hi);
        let bytes = p.hi - p.lo;
        match groups.last_mut() {
            Some((g_bytes, g_indices)) if t_lo < cur_end => {
                g_indices.push(idx);
                *g_bytes += bytes;
                cur_end = cur_end.max(t_hi);
            }
            _ => {
                groups.push((bytes, vec![idx]));
                cur_end = t_hi;
            }
        }
    }
    if groups.len() < 2 {
        return run_seq(&pieces);
    }
    for (_, g_indices) in &mut groups {
        if g_indices.len() > 1 {
            g_indices.sort_unstable();
        }
    }

    // Pack disjoint groups into up to 16 balanced buckets (LPT by bytes) so
    // large single-shadow megabuffer piece lists spread evenly across workers
    // with dynamic work stealing.
    groups.sort_unstable_by_key(|a| std::cmp::Reverse(a.0));
    let num_buckets = groups.len().min(16);
    let mut bucket_load = vec![(0usize, 0usize); num_buckets];
    let mut bucket_indices: Vec<Vec<usize>> = (0..num_buckets).map(|_| Vec::new()).collect();
    for (g_bytes, g_indices) in groups {
        let best = bucket_load
            .iter()
            .enumerate()
            .min_by_key(|&(_, &load)| load)
            .map_or(0, |(i, _)| i);
        bucket_load[best].0 += g_bytes;
        bucket_load[best].1 += g_indices.len();
        bucket_indices[best].extend(g_indices);
    }

    let mut slots: Vec<Option<RelayCardPiece>> = pieces.into_iter().map(Some).collect();
    let written = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let jobs: Vec<Box<dyn FnOnce() + Send>> = bucket_indices
        .into_iter()
        .filter(|b| !b.is_empty())
        .map(|indices| {
            let bucket: Vec<RelayCardPiece> = indices
                .into_iter()
                .filter_map(|idx| slots[idx].take())
                .collect();
            let written = written.clone();
            Box::new(move || {
                let mut title = Vec::new();
                let mut local = 0usize;
                for p in bucket {
                    // SAFETY: caller guarantees `p.gpu_ptr` points to `p.hi - p.lo`
                    // readable bytes that stay live until `wide_run` returns.
                    let gpu =
                        unsafe { std::slice::from_raw_parts(p.gpu_ptr as *const u8, p.hi - p.lo) };
                    local += p.shadow.card_merge_relay(p.lo, gpu, p.after, &mut title);
                }
                written.fetch_add(local, Ordering::Relaxed);
            }) as Box<dyn FnOnce() + Send>
        })
        .collect();
    crate::shadow::wide_run(jobs);
    written.load(Ordering::Relaxed)
}

/// One declared range of a copy in the card's memory, in a submission.
struct CardRange {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    declared: (usize, usize),
    span: crate::shadow::CardSpan,
    at: u32,
    compared: bool,
}

/// What a submission does for copies in the card's memory.
pub(crate) struct CardPlan {
    ranges: Vec<CardRange>,
    sent_as: u64,
    pub(super) check: Option<ID3D12CommandList>,
    _synced: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    lists: Vec<(bool, Prelude)>,
    order_after: u64,
    inflight: bool,
    inflight_landed: u32,
}

/// Whether `VICTORY_RELAY` itself requests relay mode (before checking
/// [`late_stage`] / `VICTORY_GPU_TIMING` / `VICTORY_LATE_WAIT=0`).
///
/// Default on in production (`VICTORY_RELAY=0` disables; `=2` enables stage 2);
/// off by default in unit tests unless `VICTORY_RELAY=1` or `=2` is set so
/// synchronous tests observe title memory immediately after submission.
fn relay_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_RELAY").map_or(!cfg!(test), |v| v == "1" || v == "2"))
}

/// Whether a queue's submissions go to the card without a wait for each
/// (`VICTORY_LATE_WAIT=1|2|3`, or automatically stage 3 when `VICTORY_RELAY`
/// is enabled; disabled under `VICTORY_GPU_TIMING` or `VICTORY_LATE_WAIT=0`).
pub(crate) fn late() -> bool {
    late_stage() > 0
}

/// `VICTORY_LATE_WAIT`: 0 (off), 1, 2 ([`late_exact`]), or 3 ([`late_free_present`]).
/// Defaults to 3 when [`relay_env`] (`VICTORY_RELAY`) is active.
fn late_stage() -> u8 {
    static STAGE: OnceLock<u8> = OnceLock::new();
    *STAGE.get_or_init(|| {
        if std::env::var_os("VICTORY_GPU_TIMING").is_some() {
            return 0;
        }
        match std::env::var("VICTORY_LATE_WAIT").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            Ok("3") => 3,
            Err(_) if relay_env() => 3,
            _ => 0,
        }
    })
}

/// Stage 2 of [`late`] (`VICTORY_LATE_WAIT=2`): merges title byte writes under
/// active flight shader writes instead of settling immediately.
fn late_exact() -> bool {
    late_stage() > 1
}

/// Stage 3 of [`late`] (`VICTORY_LATE_WAIT=3`).
#[allow(dead_code)]
pub(crate) fn late_free_present() -> bool {
    late_stage() > 2
}

/// Whether GPU shader write copy-back runs on a dedicated `"victory-relay"`
/// worker thread beside the submitting thread (`VICTORY_RELAY=1|2`; default on
/// in production, `VICTORY_RELAY=0` disables; off by default in unit tests
/// unless `VICTORY_RELAY=1|2`; also disabled under `VICTORY_LATE_WAIT=0` or
/// `VICTORY_GPU_TIMING`).
pub(crate) fn relay() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| late_stage() > 0 && relay_env())
}

/// Relay stage 2 (`VICTORY_RELAY=2`): waits only until flights blocking the
/// next submission's reads have landed, and at [`late_most`] waits until
/// `line.lists < late_most()`.
pub(super) fn relay_exact() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| relay() && std::env::var_os("VICTORY_RELAY").is_some_and(|v| v == "2"))
}

/// With [`relay`]: whether card-copy and ReBAR readbacks run on `Host::copy_queue`
/// (`VICTORY_RELAY_COPY_QUEUE=0` disables and puts them on the direct queue).
pub(super) fn relay_copy_queue() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RELAY_COPY_QUEUE").is_none_or(|v| v != "0"))
}

static RELAY_MOVED: Condvar = Condvar::new();
static LINE_LANDED: Condvar = Condvar::new();

/// Coordinated queue-order and fence-watermark timeline for relay card uploads.
///
/// Tracks `queue_order` (locked when submitting card sync lists or direct-queue
/// fallback fetches), its lock-free mirror `submitted_order` (updated strictly
/// under `queue_order` right after `ExecuteCommandLists`), and the latest
/// `(fence_value, submitted_order)` pair (`last_signalled`).
///
/// Lock order: `FETCHER` (`direct_submit_guard()`) -> `queue_order` -> `last_signalled`.
/// [`Self::note_signalled`] reads `submitted_order` lock-free so it never
/// acquires `queue_order` and unsubmitted candidate plans are never included in
/// a fence watermark before submission.
pub(crate) struct QueueOrderTimeline {
    queue_order: Mutex<u64>,
    submitted_order: AtomicU64,
    last_signalled: Mutex<(u64, u64)>,
}

impl QueueOrderTimeline {
    pub(crate) const fn new() -> Self {
        Self {
            queue_order: Mutex::new(0),
            submitted_order: AtomicU64::new(0),
            last_signalled: Mutex::new((0, 0)),
        }
    }

    pub(crate) fn lock_queue_order(&self) -> std::sync::MutexGuard<'_, u64> {
        self.queue_order.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[inline]
    pub(crate) fn submitted_order(&self) -> u64 {
        self.submitted_order.load(Ordering::Acquire)
    }

    /// Computes `(prev_order, candidate_sent_as)` for a pending card plan.
    #[inline]
    pub(crate) fn next_candidate(&self) -> (u64, u64) {
        let prev = self.submitted_order();
        (prev, prev + 1)
    }

    /// Advances `queue_order` and `submitted_order` after `ExecuteCommandLists`
    /// while holding `order_guard`, returning the committed submitted order.
    #[inline]
    pub(crate) fn commit_submitted_locked(
        &self,
        order_guard: &mut u64,
        sent_as: u64,
        has_uploads: bool,
    ) -> u64 {
        if has_uploads {
            *order_guard = (*order_guard).max(sent_as);
            self.submitted_order.store(*order_guard, Ordering::Release);
        }
        *order_guard
    }

    /// Records `(fence_val, current submitted_order)` without locking `queue_order`.
    pub(crate) fn note_signalled(&self, fence_val: u64) {
        let order = self.submitted_order();
        self.note_signalled_with_order(fence_val, order);
    }

    pub(crate) fn note_signalled_with_order(&self, fence_val: u64, order: u64) {
        let mut pair = self
            .last_signalled
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if fence_val > pair.0 {
            *pair = (fence_val, order);
        }
    }

    pub(crate) fn last_signalled(&self) -> (u64, u64) {
        *self
            .last_signalled
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Computes `(fence_to_wait, visible_watermark)` for an auxiliary `COPY` queue.
    pub(crate) fn sync_copy_plan(
        &self,
        completed_fence: u64,
        wait_direct: bool,
        relay_fetched: u64,
    ) -> (Option<u64>, u64) {
        let (v, order) = self.last_signalled();
        if v > 0 {
            if completed_fence >= v {
                return (None, order.max(relay_fetched));
            }
            if wait_direct {
                return (Some(v), order.max(relay_fetched));
            }
        }
        (None, relay_fetched)
    }
}

/// Process-wide queue-order and fence-watermark timeline for `Host::queue`.
pub(super) static QUEUE_TIMELINE: QueueOrderTimeline = QueueOrderTimeline::new();

static RELAY_STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static INFLIGHT_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static RELAY_REBAR_STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// After a fetch the old way taken with the queue's count at `after`, for
/// plans whose highest send is `landed`: up to which send the notes of
/// sends to card copies may go (`crate::shadow::relay_fetched`).
pub(crate) fn fallback_fetched_mark(inflight: bool, after: u64, landed: Option<u64>) -> u64 {
    match (inflight, landed) {
        (false, _) => after,
        (true, Some(landed)) => landed.min(after),
        (true, None) => 0,
    }
}

/// Computes the `relay_fetched` watermark when a relay landing has no eager
/// card pieces left to DMA (including when every changed card page was
/// deferred by `lazy_back`). Deferred pages never pin this watermark.
pub(crate) fn relay_no_card_fetch_watermark(
    inflight: bool,
    has_copy_queue: bool,
    max_sent_as: Option<u64>,
    completed_order: u64,
    queue_order: u64,
) -> Option<u64> {
    if inflight {
        max_sent_as
    } else if has_copy_queue {
        Some(completed_order)
    } else {
        Some(queue_order)
    }
}

/// The landed mark of [`CARD_FETCH`]'s tickets after ticket `k` landed.
pub(crate) fn inflight_raise(landed: u32, k: u32) -> u32 {
    let next = k.wrapping_add(1);
    if (next.wrapping_sub(landed) as i32) > 0 {
        next
    } else {
        landed
    }
}

fn inflight_tickets_landed(plans: &[&CardPlan]) {
    let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(work) = work.as_mut() else {
        return;
    };
    let dwords = (work.differs.2 / 4) as usize;
    let Some(inflight) = work.inflight.as_mut() else {
        return;
    };
    // SAFETY: the mapping of the tickets; these plans' fetches have run.
    let table =
        unsafe { std::slice::from_raw_parts((inflight.table.1 as *const u32).add(1), dwords) };
    for plan in plans.iter().filter(|p| p.inflight) {
        for r in plan.ranges.iter().filter(|r| r.compared) {
            for p in 0..r.span.pieces {
                if let Some(&k) = table.get((r.at + p) as usize)
                    && k != INFLIGHT_NONE
                {
                    inflight.landed = inflight_raise(inflight.landed, k);
                }
            }
        }
    }
    if let Some(last) = plans.iter().map(|p| p.sent_as).max() {
        inflight.brought.retain(|b| b.3 > last);
    }
}

pub(super) fn should_record_brought(after: u64, landed: Option<u64>) -> bool {
    after > 1 && landed.is_none_or(|last| after - 1 > last)
}

pub(super) fn landable(line: &[Option<u64>], completed: u64) -> usize {
    line.iter()
        .position(|f| f.is_some_and(|f| f > completed))
        .unwrap_or(line.len())
}

pub(super) fn flagged_pieces(
    flags: &[(usize, u32, bool)],
) -> std::collections::HashSet<(usize, u32)> {
    flags.iter().filter(|f| f.2).map(|f| (f.0, f.1)).collect()
}

/// The maximum number of host lists in flight before settling
/// (`VICTORY_LATE_WAIT_LISTS`, default 256).
fn late_most() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_LATE_WAIT_LISTS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(256)
            .max(1)
    })
}

/// Whether a submission with bottom-of-pipe writes may fly
/// (`VICTORY_LATE_WAIT_WRITES=0` settles immediately after submitting).
fn late_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LATE_WAIT_WRITES").is_none_or(|v| v != "0"))
}

/// Per host copy, the ranges a submission's shaders may write, as declared
/// `(address, end)`.
pub(super) type Declared = (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>);

/// What [`Host::submit`] hands back for a submission sent ahead (`late = true`).
pub(super) struct Sent {
    pub(super) number: u64,
    pub(super) card: Option<CardPlan>,
    pub(super) back: Vec<Declared>,
    pub(super) readback_ranges: Vec<(usize, usize)>,
    pub(super) writes: Vec<(usize, crate::queue::Written, usize)>,
    pub(super) exec_epoch: crate::retire::ExecEpochGuard,
}

/// A submission on the card that has not been waited for ([`late`]).
pub(super) struct Flight {
    pub(super) number: u64,
    pub(super) fence: u64,
    pub(super) lists: Vec<crate::queue::InFlight>,
    pub(super) card: Option<CardPlan>,
    pub(super) back: Vec<Declared>,
    pub(super) readback_ranges: Vec<(usize, usize)>,
    pub(super) _exec_epoch: crate::retire::ExecEpochGuard,
}

/// One item in the ordered flight line.
pub(super) enum Late {
    Run(Flight),
    Write(usize, crate::queue::Written, usize),
    Signal(crate::fence::Held, u64),
}

pub(super) struct Line {
    pub(super) line: Vec<Late>,
    /// Joined title-memory ranges that in-flight shaders or readbacks may write.
    pub(super) written: Vec<(usize, usize)>,
    /// Host lists currently in flight.
    pub(super) lists: usize,
    /// Last fence value signalled behind a queued flight.
    pub(super) target: u64,
    /// True while a settle or `relay_land` is actively draining flights.
    pub(super) settling: bool,
    /// Signals `victory-late-wait` to drain landed lists / woken timelines.
    pub(super) poke: bool,
    /// Finished flights' `InFlight` lists queued for async reset/recycle.
    pub(super) landed: Vec<Vec<crate::queue::InFlight>>,
    /// Bottom-of-pipe writes of flights currently being landed (`late_met`).
    pub(super) in_hand: Vec<(usize, crate::queue::Written, usize)>,
    /// With [`relay`]: `(shadow_ptr, from, to)` ranges declared in
    /// `flight.back` (or `(0, from, to)` for `flight.readback_ranges`) of
    /// flights currently being landed (`items` drained while `settling` is
    /// true), for range-scoped `WHY_STALE` and `WHY_READ` waits.
    pub(super) in_hand_back: Vec<(usize, usize, usize)>,
    /// Timelines woken by `Late::Write` / `Late::Signal`, drained outside
    /// `Host::submitted` and never on `victory-relay`.
    pub(super) pending_woken: Vec<crate::fence::Woken>,
}

#[cfg(test)]
impl Line {
    pub(super) fn new_for_test() -> Self {
        Self {
            line: Vec::new(),
            written: Vec::new(),
            lists: 0,
            target: 0,
            settling: false,
            poke: false,
            landed: Vec::new(),
            in_hand: Vec::new(),
            in_hand_back: Vec::new(),
            pending_woken: Vec::new(),
        }
    }
}

// SAFETY: D3D12 objects, `InFlight`, `Held`, and `Woken` are safe to transfer
// across threads under `LINE`.
unsafe impl Send for Line {}

static LINE: Mutex<Line> = Mutex::new(Line {
    line: Vec::new(),
    written: Vec::new(),
    lists: 0,
    target: 0,
    settling: false,
    poke: false,
    landed: Vec::new(),
    in_hand: Vec::new(),
    in_hand_back: Vec::new(),
    pending_woken: Vec::new(),
});

static LINE_MOVED: Condvar = Condvar::new();

fn line() -> std::sync::MutexGuard<'static, Line> {
    LINE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Why flights were settled (`LATE_STATS` indices 2..=10).
pub(super) const WHY_SYNC: usize = 2;
const WHY_CARD_DONE: usize = 3;
const WHY_VALUE: usize = 4;
pub(super) const WHY_STALE: usize = 5;
pub(super) const WHY_READ: usize = 6;
pub(super) const WHY_RUN: usize = 7;
pub(super) const WHY_MOST: usize = 8;
const WHY_WRITES: usize = 9;
const WHY_FENCE: usize = 10;

static LATE_STATS: [AtomicU64; 18] = [const { AtomicU64::new(0) }; 18];
static LATE_WAITED: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

fn touches(ranges: &[(usize, usize)], from: usize, to: usize) -> bool {
    if from >= to {
        return false;
    }
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges.get(at).is_some_and(|r| r.0 < to)
}

pub(super) fn join_ranges(
    ranges: &mut Vec<(usize, usize)>,
    more: impl IntoIterator<Item = (usize, usize)>,
) {
    ranges.extend(more.into_iter().filter(|r| r.0 < r.1));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges.iter() {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    *ranges = out;
}

/// Whether any flights are queued or currently being landed.
pub(super) fn flying() -> bool {
    if !late() {
        return false;
    }
    let line = line();
    line.settling || line.line.iter().any(|l| matches!(l, Late::Run(_)))
}

/// Result of checking `runs` against in-flight shader writes ([`late_blocked`]).
pub(super) enum LateBlock {
    /// A `Step::Run` step may read or write arbitrary title memory (`WHY_RUN`).
    Run,
    /// Joined title-memory ranges where copies/fills/host-reads/textures
    /// conflict with in-flight shader writes or readbacks (`WHY_READ`).
    Read(Vec<(usize, usize)>),
}

/// Checks whether `runs` reads or runs arbitrary code over title memory that
/// in-flight shaders or readbacks may write (`WHY_READ` or `WHY_RUN`).
pub(super) fn late_blocked(runs: &[&HostList]) -> Option<LateBlock> {
    let written = {
        let line = line();
        if line.written.is_empty() {
            return None;
        }
        line.written.clone()
    };
    blocked_ranges(&written, runs, true)
}

pub(super) fn blocked_ranges(
    written: &[(usize, usize)],
    runs: &[&HostList],
    count: bool,
) -> Option<LateBlock> {
    if written.is_empty() {
        return None;
    }
    let mut ranges = Vec::new();
    let mut first_stat: Option<usize> = None;
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                &Step::Copy(dst, src, len) => {
                    let hit_dst = touches(written, dst, dst + len);
                    let hit_src = touches(written, src, src + len);
                    if hit_dst || hit_src {
                        if !hit_dst && fill_ahead::on(src, len) {
                            if count {
                                fill_ahead::read(src, len);
                            }
                            continue;
                        }
                        if hit_dst {
                            ranges.push((dst, dst + len));
                            first_stat.get_or_insert(15);
                        }
                        if hit_src {
                            ranges.push((src, src + len));
                            first_stat.get_or_insert(15);
                        }
                    }
                }
                &Step::Fill(_, src, len) => {
                    if touches(written, src, src + len) {
                        if fill_ahead::on(src, len) {
                            if count {
                                fill_ahead::read(src, len);
                            }
                            continue;
                        }
                        ranges.push((src, src + len));
                        first_stat.get_or_insert(16);
                    }
                }
                &Step::HostRead(src, len, _) => {
                    if touches(written, src, src + len) {
                        ranges.push((src, src + len));
                        first_stat.get_or_insert(16);
                    }
                }
                Step::Texture(staging, _) => {
                    let (start, end) = staging.memory_range();
                    if touches(written, start, end) {
                        ranges.push((start, end));
                        first_stat.get_or_insert(17);
                    }
                }
                Step::Run(_) => return Some(LateBlock::Run),
                Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(_) => {}
            }
        }
    }
    if ranges.is_empty() {
        return None;
    }
    if count && let Some(slot) = first_stat {
        LATE_STATS[slot].fetch_add(1, Ordering::Relaxed);
    }
    join_ranges(&mut ranges, std::iter::empty());
    Some(LateBlock::Read(ranges))
}

/// Fills of upload memory that read title memory flights may write are not
/// waited for unless a landing actually modifies bytes a fill read ahead of it
/// (`VICTORY_FILL_WAIT=1` forces waiting from the start).
mod fill_ahead {
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    use std::sync::{Mutex, OnceLock};

    const PIECE: usize = 1 << 16;
    const KEEP: u64 = 16;
    const HELD_MAX: usize = 8;

    static WAIT: AtomicBool = AtomicBool::new(false);
    static ANY_HELD: AtomicBool = AtomicBool::new(false);
    static HELD: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
    static COUNT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

    fn print(src: usize, len: usize) -> u64 {
        crate::texture::fingerprint(src as u64, len as u64, &[])
    }

    fn print_raw(src: usize, len: usize) -> Option<u64> {
        crate::texture::fingerprint_raw(src as u64, len as u64, &[])
    }

    #[derive(Default)]
    struct Piece {
        pruned: u64,
        fills: Vec<(usize, usize, u64, u64, u64)>,
    }

    static LANDING: AtomicU64 = AtomicU64::new(0);

    pub(super) fn landing(first: u64) {
        LANDING.store(first, Relaxed);
    }

    static READ: Mutex<Option<HashMap<usize, Piece>>> = Mutex::new(None);

    pub(super) fn on(src: usize, len: usize) -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_FILL_WAIT").is_none_or(|v| v != "1"))
            || WAIT.load(Relaxed)
        {
            return false;
        }
        if len == 0 || !ANY_HELD.load(Relaxed) {
            return true;
        }
        let held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        let Some(held) = held.as_ref() else {
            return true;
        };
        !(src / PIECE..=(src + len - 1) / PIECE).any(|piece| held.contains(&piece))
    }

    fn hold(from: usize, to: usize) -> bool {
        let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        let held = held.get_or_insert_with(HashSet::new);
        let first = held.is_empty();
        held.extend(from / PIECE..=(to - 1) / PIECE);
        ANY_HELD.store(true, Relaxed);
        if held.len() > HELD_MAX && !WAIT.swap(true, Relaxed) {
            tracing::warn!(
                "fills of upload memory read ahead of flights that changed their bytes in {} pieces of 64 KiB: every fill \
                 waits for flights again from here on, as with VICTORY_FILL_WAIT=1",
                held.len()
            );
        }
        first
    }

    pub(crate) fn read(src: usize, len: usize) {
        if len == 0 {
            return;
        }
        COUNT[0].fetch_add(1, Relaxed);
        let frame = crate::frame::current();
        let print = print(src, len);
        let mut read = READ.lock().unwrap_or_else(|e| e.into_inner());
        let map = read.get_or_insert_with(HashMap::new);
        for piece in src / PIECE..=(src + len - 1) / PIECE {
            let p = map.entry(piece).or_default();
            if p.pruned != frame {
                p.pruned = frame;
                p.fills.retain(|f| f.2 + KEEP >= frame);
            }
            p.fills
                .push((src, src + len, frame, super::STARTED.load(Relaxed), print));
        }
    }

    pub(crate) fn wrote(at: usize, new: &[u8]) {
        if new.is_empty() || WAIT.load(Relaxed) {
            return;
        }
        let frame = crate::frame::current();
        let landing = LANDING.load(Relaxed);
        let mut hits: Vec<(usize, usize, u64, u64)> = {
            let read = READ.lock().unwrap_or_else(|e| e.into_inner());
            let Some(map) = read.as_ref() else { return };
            let end = at + new.len();
            (at / PIECE..=(end - 1) / PIECE)
                .filter_map(|piece| map.get(&piece))
                .flat_map(|p| p.fills.iter().copied())
                .filter(|f| {
                    f.0 < end && f.1 > at && f.2 + KEEP >= frame && (landing == 0 || landing <= f.3)
                })
                .map(|f| (f.0, f.1, f.2, f.4))
                .collect()
        };
        if hits.is_empty() {
            return;
        }
        hits.sort_unstable();
        hits.dedup();
        COUNT[1].fetch_add(1, Relaxed);
        let mut now = Vec::new();
        for (from, to, read_in, print_then) in hits {
            // `wrote` runs while callers hold per-`Shadow` locks (`rebar.mirror`
            // / `Shadow::before`), so reads use non-materializing `print_raw`
            // and `read_into_raw`.
            let Some(print_now) = print_raw(from, to - from) else {
                continue;
            };
            if print_now != print_then {
                COUNT[3].fetch_add(1, Relaxed);
                continue;
            }
            let (a, b) = (from.max(at), to.min(at + new.len()));
            now.clear();
            now.resize(b - a, 0);
            if !super::read_into_raw(a as *const u8, &mut now) || now[..] == new[a - at..b - at] {
                continue;
            }
            let n = COUNT[2].fetch_add(1, Relaxed) + 1;
            let first = now
                .iter()
                .zip(&new[a - at..b - at])
                .position(|(x, y)| x != y)
                .unwrap_or(0);
            if hold(from, to) {
                tracing::warn!(
                    "a fill of upload memory read {} bytes at {from:#x} in frame {read_in} ahead of a flight whose shaders changed \
                     them (first at {:#x}, landed in frame {frame}): fills reading its 64 KiB wait from here on ({n} misses)",
                    to - from,
                    a + first
                );
            }
        }
    }

    pub(super) fn line(frames: f64) {
        let [ahead, looked] = [0, 1].map(|i| COUNT[i].swap(0, Relaxed));
        if ahead == 0 && looked == 0 {
            return;
        }
        let kept = {
            let frame = crate::frame::current();
            let mut read = READ.lock().unwrap_or_else(|e| e.into_inner());
            read.as_mut().map_or(0, |map| {
                map.retain(|_, p| {
                    p.fills.retain(|f| f.2 + KEEP >= frame);
                    !p.fills.is_empty()
                });
                map.values().map(|p| p.fills.len()).sum::<usize>()
            })
        };
        tracing::info!(
            "fills ahead of flights (VICTORY_FILL_WAIT=1 waits): {:.1}/frame, {:.1} checked/frame, \
             {} misses ({} 64 KiB pieces held{}), {} rewritten, {kept} tracked",
            ahead as f64 / frames,
            looked as f64 / frames,
            COUNT[2].load(Relaxed),
            HELD.lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .map_or(0, HashSet::len),
            if WAIT.load(Relaxed) {
                ", every fill waits"
            } else {
                ""
            },
            COUNT[3].load(Relaxed)
        );
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn held(at: usize) -> bool {
            HELD.lock()
                .unwrap()
                .as_ref()
                .is_some_and(|h| h.contains(&(at / PIECE)))
        }

        #[test]
        fn only_real_misses_hold_fills_back() {
            const LEN: usize = 480;
            let mut memory = vec![0u8; 4 * PIECE];
            let start = memory.as_ptr() as usize;
            let base = start.next_multiple_of(PIECE) - start;
            let at = |i: usize| start + base + i * PIECE;
            let (rewritten, missed, same) = (at(0), at(1), at(2));
            for i in 0..3 {
                memory[base + i * PIECE..][..LEN].fill(1);
            }

            read(rewritten, LEN);
            memory[base..][..LEN].fill(2);
            wrote(rewritten, &[3; LEN]);
            assert!(!held(rewritten), "a rewrite by the title is not a miss");
            assert!(on(rewritten, LEN));

            read(missed, LEN);
            wrote(missed, &[3; LEN]);
            assert!(held(missed), "a real miss holds its piece back");
            assert!(!on(missed, LEN));
            assert!(on(same, LEN));
            assert!(
                !WAIT.load(Relaxed),
                "one piece does not make every fill wait"
            );

            read(same, LEN);
            wrote(same, &[1; LEN]);
            assert!(!held(same));
            assert!(on(same, LEN));
        }
    }
}

#[cfg(test)]
pub(crate) use fill_ahead::read as test_fill_ahead_read;
pub(crate) use fill_ahead::wrote as shaders_write;

/// Whether `WHY_STALE` and `WHY_READ` wait only on flights whose declared
/// write/readback ranges overlap the conflicting ranges
/// (`VICTORY_RANGE_SETTLE=0` falls back to full [`Host::settle`]).
pub(super) fn range_settle_enabled(setting: Option<&str>) -> bool {
    setting != Some("0")
}

pub(super) fn range_settle() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| range_settle_enabled(std::env::var("VICTORY_RANGE_SETTLE").ok().as_deref()))
}

/// Whether any queued flight in `line.line` or currently landing flight in
/// `line.in_hand_back` declares a `back` range overlapping any
/// `(shadow_ptr, from, to)` in `stale`.
pub(super) fn line_overlaps_stale(line: &Line, stale: &[(usize, usize, usize)]) -> bool {
    if stale.is_empty() {
        return false;
    }
    for &(s_ptr, a, b) in &line.in_hand_back {
        if stale
            .iter()
            .any(|&(ptr, from, to)| ptr == s_ptr && a < to && from < b)
        {
            return true;
        }
    }
    for item in &line.line {
        let Late::Run(flight) = item else {
            continue;
        };
        for (shadow, ranges) in &flight.back {
            let s_ptr = std::sync::Arc::as_ptr(shadow) as usize;
            if stale.iter().any(|&(ptr, from, to)| {
                ptr == s_ptr && ranges.iter().any(|&(a, b)| a < to && from < b)
            }) {
                return true;
            }
        }
    }
    false
}

/// Whether any queued flight in `line.line` or currently landing flight in
/// `line.in_hand_back` declares a `back` or `readback_ranges` range
/// overlapping any `(from, to)` in `ranges` (for range-scoped `WHY_READ`).
pub(super) fn line_overlaps_read(line: &Line, ranges: &[(usize, usize)]) -> bool {
    if ranges.is_empty() {
        return false;
    }
    for &(_, a, b) in &line.in_hand_back {
        if ranges.iter().any(|&(from, to)| a < to && from < b) {
            return true;
        }
    }
    for item in &line.line {
        let Late::Run(flight) = item else {
            continue;
        };
        if flight
            .back
            .iter()
            .flat_map(|(_, r)| r.iter())
            .chain(flight.readback_ranges.iter())
            .any(|&(a, b)| ranges.iter().any(|&(from, to)| a < to && from < b))
        {
            return true;
        }
    }
    false
}

/// With flights in the air, collects `(shadow_ptr, from, to)` ranges where
/// `runs` refreshes a range of a host copy that has pending shader writes and
/// dirty title pages. Under [`late_exact`] (`VICTORY_LATE_WAIT>=2`), merges
/// changed title bytes into the copy in place (up to `VICTORY_LATE_MERGE_RUNS`,
/// default 4096 runs per submission for card shadows; ReBAR and PC shadows are
/// uncapped) and returns only the subranges that still require settling.
pub(super) fn stale_under_flights(runs: &[&HostList]) -> Vec<(usize, usize, usize)> {
    use std::sync::Arc;
    // Per copy, the joined refreshed intervals (not a bounding box, so
    // disjoint refreshes do not pull in unrelated pending shader writes).
    let mut by_shadow: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut spans: Vec<Declared> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::Refresh(shadow, address, len) = step
                && *len > 0
                && shadow.writes_pending()
            {
                let key = Arc::as_ptr(shadow) as usize;
                let range = (*address, *address + *len);
                match by_shadow.get(&key) {
                    Some(&i) => spans[i].1.push(range),
                    None => {
                        by_shadow.insert(key, spans.len());
                        spans.push((shadow.clone(), vec![range]));
                    }
                }
            }
        }
    }
    if spans.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut per_shadow = Vec::new();
    // `VICTORY_LATE_MERGE_RUNS`: card `CopyBufferRegion` run budget per
    // submission (4096 avoids stalling on scattered constant writes while
    // keeping command-list recording negligible; ReBAR is uncapped).
    static RUNS: OnceLock<usize> = OnceLock::new();
    let mut runs_left = late_exact().then(|| {
        *RUNS.get_or_init(|| {
            std::env::var("VICTORY_LATE_MERGE_RUNS")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(4096)
        })
    });
    for (shadow, ranges) in &mut spans {
        join_ranges(ranges, std::iter::empty());
        per_shadow.clear();
        for &(from, to) in ranges.iter() {
            if let Some(left) = runs_left.as_mut() {
                shadow.merge_ranges_under_writes(from, to - from, left, &mut per_shadow);
            } else {
                shadow.stale_ranges_under_writes(from, to - from, &mut per_shadow);
            }
        }
        if !per_shadow.is_empty() {
            join_ranges(&mut per_shadow, std::iter::empty());
            let ptr = Arc::as_ptr(shadow) as usize;
            out.extend(per_shadow.iter().map(|&(a, b)| (ptr, a, b)));
        }
    }
    out
}

/// Queues a fence signal behind in-flight submissions when [`late`] is active
/// and flights are queued or settling. Returns `Ok(())` if queued, or
/// `Err(fence)` if the caller should signal `fence` immediately.
pub(crate) fn late_signal(fence: crate::fence::Held, value: u64) -> Result<(), crate::fence::Held> {
    if !late() {
        return Err(fence);
    }
    let mut line = line();
    if line.line.is_empty() && !line.settling {
        return Err(fence);
    }
    line.line.push(Late::Signal(fence, value));
    LATE_STATS[14].fetch_add(1, Ordering::Relaxed);
    drop(line);
    LINE_MOVED.notify_one();
    RELAY_MOVED.notify_one();
    Ok(())
}

/// Pure decision for [`late_met`]: `Some(true)` if the final queued write to
/// `[address, address + bytes)` satisfies `holds` with no conflicting shader,
/// timestamp, or partial write; `Some(false)` if definitely not met without
/// settling; `None` if an overlapping write requires settling first.
pub(super) fn eval_late_met(
    written: &[(usize, usize)],
    writes: impl IntoIterator<Item = (usize, crate::queue::Written, usize)>,
    address: usize,
    bytes: usize,
    holds: impl Fn(u64) -> bool,
) -> Option<bool> {
    let end = address.saturating_add(bytes);
    let mut unsure = touches(written, address, end);
    let mut last_exact: Option<bool> = None;
    for (at, what, n) in writes {
        let write_end = at.saturating_add(n);
        if at >= end || address >= write_end {
            continue;
        }
        match what {
            crate::queue::Written::Value(v) if at == address && n >= bytes => {
                last_exact = Some(holds(v));
            }
            _ => unsure = true,
        }
    }
    if unsure {
        None
    } else {
        Some(last_exact.unwrap_or(false))
    }
}

/// Checks whether a memory wait is satisfied by queued bottom-of-pipe writes
/// of in-flight submissions; settles flights if an overlapping shader,
/// timestamp, or partial write could satisfy it.
pub(crate) fn late_met(wait: &crate::timeline::MemoryWait) -> bool {
    let (address, bytes) = wait.place();
    let decided = {
        let line = line();
        if line.line.is_empty() && line.in_hand.is_empty() && !line.settling {
            return false;
        }
        let writes = line
            .in_hand
            .iter()
            .copied()
            .chain(line.line.iter().filter_map(|item| match item {
                Late::Write(at, what, n) => Some((*at, *what, *n)),
                _ => None,
            }));
        eval_late_met(&line.written, writes, address, bytes, |v| wait.holds_for(v))
    };
    match decided {
        Some(true) => {
            LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
            true
        }
        Some(false) => false,
        None => {
            settle_now(WHY_VALUE);
            wait.satisfied()
        }
    }
}

/// [`late_met`] without settling.
#[allow(dead_code)]
pub(crate) fn late_met_known(wait: &crate::timeline::MemoryWait) -> bool {
    let (address, bytes) = wait.place();
    let line = line();
    if line.line.is_empty() && line.in_hand.is_empty() && !line.settling {
        return false;
    }
    let writes = line
        .in_hand
        .iter()
        .copied()
        .chain(line.line.iter().filter_map(|item| match item {
            Late::Write(at, what, n) => Some((*at, *what, *n)),
            _ => None,
        }));
    if eval_late_met(&line.written, writes, address, bytes, |v| wait.holds_for(v)) == Some(true) {
        LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
        return true;
    }
    false
}

/// Settles all in-flight submissions for reason `why`. Must not be called while
/// holding `Host::submitted`.
pub(crate) fn settle_now(why: usize) {
    if !late() {
        return;
    }
    let Some(host) = host() else {
        return;
    };
    let woken = {
        let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        host.settle(&mut submitted, why)
    };
    woken.run();
    drain_woken();
}

/// Settles in-flight submissions before presenting.
pub(crate) fn settle_for_present() {
    settle_now(WHY_SYNC);
}

/// Settles in-flight submissions before a blocking fence wait.
pub(crate) fn settle_for_fence() {
    settle_now(WHY_FENCE);
}

/// Whether the `victory-late-wait` helper thread is running and draining `land()`.
static LATE_HELPER_ALIVE: AtomicBool = AtomicBool::new(false);

/// Recycles finished flights' `InFlight` host lists outside `Host::submitted`.
pub(super) fn land() {
    let landed = std::mem::take(&mut line().landed);
    let Some(host) = host() else {
        return;
    };
    for batch in landed {
        for entry in batch {
            let list = entry.take();
            if let Ok(orphan) = std::sync::Arc::try_unwrap(list) {
                host.recycle_list(orphan);
            }
        }
    }
}

/// Queues `woken` to be drained outside `Host::submitted` by `"victory-late-wait"`.
pub(super) fn enqueue_woken(woken: crate::fence::Woken) {
    let mut l = line();
    l.pending_woken.push(woken);
    l.poke = true;
    drop(l);
    LINE_MOVED.notify_one();
}

/// Drains and runs any `Woken` timelines queued by `relay_land` or `settle`.
/// Must NEVER be called while holding `Host::submitted` or `LINE`, or from
/// `victory-relay`.
pub(super) fn drain_woken() {
    let pending = std::mem::take(&mut line().pending_woken);
    for woken in pending {
        woken.run();
    }
}

/// Background thread that:
/// - In non-relay `VICTORY_LATE_WAIT=1..3` mode: waits for the latest flight's
///   GPU fence and settles the line when the GPU catches up.
/// - In all modes: asynchronously recycles landed host lists (`land()`) and
///   wakes timelines released by `Late::Write` / `Late::Signal` (`drain_woken()`).
fn late_helper() {
    let Some(host) = host() else {
        LATE_HELPER_ALIVE.store(false, Ordering::Release);
        return;
    };
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        LATE_HELPER_ALIVE.store(false, Ordering::Release);
        tracing::warn!("no event for victory-late-wait thread");
        return;
    };
    loop {
        let target = {
            let mut line = line();
            loop {
                if line.poke || !line.landed.is_empty() || !line.pending_woken.is_empty() {
                    line.poke = false;
                    break None;
                }
                if !relay() && !line.line.is_empty() && !line.settling {
                    break Some(line.target);
                }
                line = LINE_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
            }
        };
        let Some(target) = target else {
            land();
            drain_woken();
            continue;
        };
        // Wait for the GPU without holding `Host::submitted`.
        // SAFETY: our fence and event.
        unsafe {
            if host.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        let woken = {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            if line().target != target {
                continue;
            }
            host.settle(&mut submitted, WHY_CARD_DONE)
        };
        land();
        woken.run();
        drain_woken();
    }
}

/// Starts `"victory-late-wait"` (and `"victory-relay"` when [`relay`] is on) once.
fn late_threads() {
    static HELPER: Once = Once::new();
    HELPER.call_once(|| {
        match std::thread::Builder::new()
            .name("victory-late-wait".into())
            .spawn(late_helper)
        {
            Ok(_) => LATE_HELPER_ALIVE.store(true, Ordering::Release),
            Err(e) => tracing::warn!(
                error = %e,
                "no victory-late-wait thread: flights will settle synchronously on callers"
            ),
        }
        tracing::info!(
            stage = late_stage(),
            most = late_most(),
            writes = late_writes(),
            "asynchronous GPU submissions enabled (VICTORY_LATE_WAIT)"
        );
        if relay() {
            if let Err(e) = std::thread::Builder::new()
                .name("victory-relay".into())
                .spawn(relay_worker)
            {
                fail(format_args!(
                    "could not spawn victory-relay thread (VICTORY_RELAY): {e}"
                ));
            }
            tracing::info!(
                exact = relay_exact(),
                copy_queue = relay_copy_queue(),
                "dedicated relay copy-back worker enabled (VICTORY_RELAY)"
            );
        }
    });
}

/// Waits on `cv` under `line_lock` until `done(&line)` holds, returning
/// `Some(elapsed_us)` when it had to wait or `None` if `done` already held.
pub(super) fn wait_on_line(
    line_lock: &Mutex<Line>,
    cv: &Condvar,
    why: usize,
    record_stats: bool,
    done: impl Fn(&Line) -> bool,
) -> Option<u64> {
    let mut line = line_lock.lock().unwrap_or_else(|e| e.into_inner());
    if done(&line) {
        return None;
    }
    if record_stats {
        LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
        LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
    }
    let waiting = std::time::Instant::now();
    while !done(&line) {
        line = cv.wait(line).unwrap_or_else(|e| e.into_inner());
    }
    drop(line);
    let us = waiting.elapsed().as_micros() as u64;
    if record_stats {
        LATE_STATS[12].fetch_add(us, Ordering::Relaxed);
        LATE_WAITED[why.min(10)].fetch_add(us, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(us, Ordering::Relaxed);
    }
    Some(us)
}

/// Waits on `LINE_LANDED` until `done(&line)` holds. Safe while holding
/// `Host::submitted` because `victory-relay` never acquires `Host::submitted`.
pub(super) fn relay_wait(why: usize, done: impl Fn(&Line) -> bool) {
    wait_on_line(&LINE, &LINE_LANDED, why, true, done);
}

/// Settles in-flight relay submissions whose declared writes may overlap
/// `(shadow_ptr, from, to)` so `Shadow::update_busy` can upload a single
/// contiguous range instead of hundreds of thousands of byte-diff fragments.
///
/// Caller MUST drop `Shadow::before` and `list.deferred()` before calling so
/// `victory-relay` can finish `relay_merge_card_pieces`, `read_backs`, and
/// `copy_back_done` without lock-order inversion.
pub(crate) fn settle_card_busy(shadow_ptr: usize, from: usize, to: usize) -> bool {
    if !relay() || !late() {
        return false;
    }
    let waited = if relay_exact() && range_settle() {
        wait_on_line(&LINE, &LINE_LANDED, WHY_STALE, true, |line| {
            !line_overlaps_stale(line, &[(shadow_ptr, from, to)])
        })
        .is_some()
    } else if relay_exact() {
        wait_on_line(&LINE, &LINE_LANDED, WHY_STALE, true, |line| {
            !line.settling
                && !line.line.iter().any(|l| {
                    matches!(l, Late::Run(flight) if flight.back.iter().any(|(shadow, ranges)| {
                        std::sync::Arc::as_ptr(shadow) as usize == shadow_ptr
                            && ranges.iter().any(|&(a, b)| a < to && from < b)
                    }))
                })
        })
        .is_some()
    } else {
        wait_on_line(&LINE, &LINE_LANDED, WHY_STALE, true, |line| {
            !line.settling && !line.line.iter().any(|l| matches!(l, Late::Run(_)))
        })
        .is_some()
    };
    if waited {
        crate::watch::ask_once_void();
    }
    waited
}

/// Resources owned by the `"victory-relay"` worker thread for DMA readback of
/// card-copy and ReBAR shadow ranges.
pub(super) struct RelayFetch {
    list: Prelude,
    copy: Option<ID3D12CommandQueue>,
    fence: ID3D12Fence,
    event: HANDLE,
    sent: u64,
    buffer: Option<(ID3D12Resource, usize, u64)>,
}

fn relay_worker() {
    let Some(host) = host() else {
        return;
    };
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        fail(format_args!(
            "no event for victory-relay thread (VICTORY_RELAY)"
        ));
    };
    let mut fetch: Option<RelayFetch> = None;
    loop {
        let next = {
            let mut line = line();
            loop {
                if line.line.is_empty() {
                    line = RELAY_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                let fences: Vec<Option<u64>> = line
                    .line
                    .iter()
                    .map(|l| match l {
                        Late::Run(flight) => Some(flight.fence),
                        _ => None,
                    })
                    .collect();
                // SAFETY: a plain query.
                let completed = unsafe { host.fence.GetCompletedValue() };
                let n = landable(&fences, completed);
                if n == 0 {
                    break Err(fences[0].unwrap_or(0));
                }
                line.settling = true;
                let items: Vec<Late> = line.line.drain(..n).collect();
                line.in_hand = items
                    .iter()
                    .filter_map(|l| match l {
                        Late::Write(at, what, bytes) => Some((*at, *what, *bytes)),
                        _ => None,
                    })
                    .collect();
                line.in_hand_back.clear();
                for item in &items {
                    if let Late::Run(flight) = item {
                        for (shadow, ranges) in &flight.back {
                            let s_ptr = std::sync::Arc::as_ptr(shadow) as usize;
                            for &(from, to) in ranges {
                                if from < to {
                                    line.in_hand_back.push((s_ptr, from, to));
                                }
                            }
                        }
                        for &(from, to) in &flight.readback_ranges {
                            if from < to {
                                line.in_hand_back.push((0, from, to));
                            }
                        }
                    }
                }
                break Ok(items);
            }
        };
        match next {
            Ok(items) => host.relay_land(items, &mut fetch),
            Err(target) => {
                let waiting = std::time::Instant::now();
                // SAFETY: host's fence and this thread's event.
                let waited = unsafe {
                    host.fence.SetEventOnCompletion(target, event).map(|()| {
                        WaitForSingleObject(event, INFINITE);
                    })
                };
                if waited.is_err() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                RELAY_STATS[2].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }
    }
}

/// Copies back GPU writes for PC-RAM (non-card, non-ReBAR) shadows synchronously.
pub(super) fn hand_back(back: &[Declared]) {
    for (shadow, declared) in back {
        if shadow.carded() || shadow.is_rebar() {
            continue;
        }
        let mut spans = Vec::new();
        join_ranges(&mut spans, declared.iter().copied());
        for (from, to) in spans {
            let written = shadow.copy_back(from, to - from);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
    }
}

/// Copies back GPU writes for PC-RAM (non-card, non-ReBAR) shadows on the relay
/// thread without writing back into `Shadow::data`.
fn relay_hand_back(back: &[Declared]) {
    let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for (shadow, declared) in back {
        if shadow.carded() || shadow.is_rebar() {
            continue;
        }
        let mut spans = Vec::new();
        join_ranges(&mut spans, declared.iter().copied());
        work.extend(
            spans
                .into_iter()
                .map(|(from, to)| (shadow.clone(), from, to)),
        );
    }
    if work.is_empty() {
        return;
    }
    let found = crate::shadow::find_all(&work);
    let (mut now, mut title) = (Vec::new(), Vec::new());
    for ((shadow, _, _), found) in work.iter().zip(found) {
        let mut written = 0;
        for (at, n) in found {
            written += shadow.relay_merge(at, n, &mut now, &mut title);
        }
        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
    }
}

/// Logs async/relay/card/stretch statistics on the periodic frame-rate line.
pub(super) fn log_stats(frames: f64) {
    let card = crate::shadow::card_stats();
    if card[7] > 0 {
        let [syncs, hashed, compares] =
            [0, 1, 2].map(|i| CARD_PASSES[i].swap(0, Ordering::Relaxed) as f64 / frames);
        tracing::info!(
            "card shadows: {} alive ({} MiB, {} MiB hashes), up {:.2} MiB ({:.0} ranges)/frame, \
             {:.0} compared / {:.0} fetched ({:.2} MiB, {:.1} ms)/frame, passes: sync {syncs:.1} ({hashed:.0} pieces), check {compares:.1}",
            card[7],
            card[8] >> 20,
            card[9] >> 20,
            card[0] as f64 / (1 << 20) as f64 / frames,
            card[1] as f64 / frames,
            card[2] as f64 / frames,
            card[3] as f64 / frames,
            card[4] as f64 / (1 << 20) as f64 / frames,
            card[6] as f64 / 1000.0 / frames,
        );
    }
    let (stretches, stretch_bytes) = crate::shadow::stretch_stats();
    if stretches > 0 {
        tracing::info!(
            "in-place stretches (VICTORY_STRETCH=0 disables): {stretches} ({} MiB)",
            stretch_bytes >> 20
        );
    }
    if !late() {
        return;
    }
    let s: Vec<f64> = LATE_STATS
        .iter()
        .map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames)
        .collect();
    tracing::info!(
        "flights/frame: {:.1} in {:.1} settles ({:.1} flights/settle), {:.1} ms waiting; \
         reasons: sync {:.1}, done {:.1}, value {:.1}, stale {:.1}, read {:.1} (copy {:.1}, fill {:.1}, tex {:.1}), \
         run {:.1}, most({}) {:.1}, writes {:.1}, fence {:.1}; value-met {:.1}, late-sig {:.1}",
        s[0],
        s[1],
        if s[1] > 0.0 { s[11] / s[1] } else { 0.0 },
        s[12] / 1000.0,
        s[WHY_SYNC],
        s[WHY_CARD_DONE],
        s[WHY_VALUE],
        s[WHY_STALE],
        s[WHY_READ],
        s[15],
        s[16],
        s[17],
        s[WHY_RUN],
        late_most(),
        s[WHY_MOST],
        s[WHY_WRITES],
        s[WHY_FENCE],
        s[13],
        s[14],
    );
    if relay() {
        let r: Vec<f64> = RELAY_STATS
            .iter()
            .map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames)
            .collect();
        tracing::info!(
            "relay/frame: {:.1} flights in {:.1} landings, card-wait {:.1} ms, fetch-wait {:.1} ms, total {:.1} ms",
            r[1],
            r[0],
            r[2] / 1000.0,
            r[3] / 1000.0,
            r[4] / 1000.0,
        );
        if inflight_fetch() {
            let i: Vec<f64> = INFLIGHT_STATS
                .iter()
                .map(|i| i.swap(0, Ordering::Relaxed) as f64 / frames)
                .collect();
            tracing::info!(
                "inflight card fetch/frame: {:.1} pieces ({:.2} MiB), fallback {:.1} (overflow {:.1})",
                i[0],
                i[1] / (1 << 20) as f64,
                i[3],
                i[2],
            );
        }
        let rr: [f64; 3] =
            [0, 1, 2].map(|i| RELAY_REBAR_STATS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        if rr[0] > 0.0 {
            tracing::info!(
                "relay ReBAR fetches/frame ({}): {:.1} fetches, {:.2} MiB, {:.1} ms waiting",
                if host().is_some_and(|h| h.copy_queue.is_some()) {
                    "COPY queue"
                } else {
                    "direct queue"
                },
                rr[0],
                rr[1] / (1 << 20) as f64,
                rr[2] / 1000.0,
            );
        }
    }
    if late_exact() {
        let [pages, runs, bytes, unwatched, no_watch, runs_cap, no_ref] =
            crate::shadow::merged_stats().map(|n| n as f64 / frames);
        if pages > 0.0 || unwatched > 0.0 || no_watch > 0.0 || runs_cap > 0.0 || no_ref > 0.0 {
            tracing::info!(
                "merge under flights/frame: {pages:.0} pages, {runs:.0} runs, {:.2} MiB \
                 (fallbacks: unwatched {unwatched:.1}, no_watch {no_watch:.1}, runs_cap {runs_cap:.1}, no_ref {no_ref:.1})",
                bytes / (1 << 20) as f64
            );
        }
    }
    fill_ahead::line(frames);
}

impl Host {
    /// Records `(fence_val, current submitted_order)` after `self.queue.Signal`.
    /// Reads the lock-free `submitted_order` so only card uploads whose
    /// command lists were already submitted to `self.queue` are included and
    /// `queue_order` is never acquired inside `direct_submit_guard()`.
    pub(crate) fn note_signalled(&self, fence_val: u64) {
        QUEUE_TIMELINE.note_signalled(fence_val);
    }

    #[cfg(test)]
    pub(crate) fn last_signalled_for_test(&self) -> (u64, u64) {
        QUEUE_TIMELINE.last_signalled()
    }

    #[cfg(test)]
    pub(crate) fn card_before_for_test(&self) -> Option<(CardPlan, u64)> {
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let plan = self.card_before(Vec::new(), false, &mut submitted)?;
        Some((plan, *submitted))
    }

    #[cfg(test)]
    pub(crate) fn card_done_and_wait_for_test(&self, plan: CardPlan, fence_val: u64) {
        // SAFETY: `self.fence` is live; local wait event avoids contending on `self.event`.
        unsafe {
            if self.fence.GetCompletedValue() < fence_val {
                let ev = CreateEventW(None, false, false, None).expect("wait event");
                self.fence
                    .SetEventOnCompletion(fence_val, ev)
                    .expect("set fence event");
                WaitForSingleObject(ev, INFINITE);
                let _ = windows::Win32::Foundation::CloseHandle(ev);
            }
        }
        self.card_done(plan);
    }

    pub(super) fn note_signalled_with_order(&self, fence_val: u64, order: u64) {
        QUEUE_TIMELINE.note_signalled_with_order(fence_val, order);
    }

    /// Synchronizes an auxiliary `COPY` queue behind the latest `self.queue`
    /// signal and returns the `QUEUE_ORDER` watermark visible after the wait.
    pub(crate) fn sync_copy_queue(
        &self,
        copy_queue: &ID3D12CommandQueue,
    ) -> windows::core::Result<u64> {
        self.sync_copy_queue_for(copy_queue, true)
    }

    /// Synchronizes an auxiliary `COPY` queue behind `self.queue` only when
    /// `wait_direct` is true and `self.fence` has not already completed past
    /// `last_signalled`.
    pub(crate) fn sync_copy_queue_for(
        &self,
        copy_queue: &ID3D12CommandQueue,
        wait_direct: bool,
    ) -> windows::core::Result<u64> {
        // SAFETY: `self.fence` is live.
        let completed = unsafe { self.fence.GetCompletedValue() };
        let (wait_v, watermark) = QUEUE_TIMELINE.sync_copy_plan(
            completed,
            wait_direct,
            crate::shadow::relay_fetched_value(),
        );
        if let Some(v) = wait_v {
            // SAFETY: `self.fence` is live and `v` was already signalled on `self.queue`.
            unsafe { copy_queue.Wait(&self.fence, v)? };
        }
        Ok(watermark)
    }

    /// Whether the host has what copies in the card's memory need (lists and
    /// the compare shader); built on first ask.
    pub(crate) fn card_ready(&self) -> bool {
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| match self.make_card_work() {
            Ok(work) => {
                *CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()) = Some(work);
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no card shadows: compare pipeline could not be made");
                false
            }
        })
    }

    fn make_card_work(&self) -> windows::core::Result<CardWork> {
        let bytes = crate::hlsl::compile_compute("card_compare", CARD_COMPARE)?;
        let list = || self.make_prelude().ok_or_else(windows::core::Error::empty);
        const RESULTS: u64 = 1 << 20;
        let (differs, data) = self
            .shared_buffer(RESULTS)
            .ok_or_else(windows::core::Error::empty)?;
        name(&differs, || "card compare results".into());
        // SAFETY: compiled DXBC with embedded root signature.
        unsafe {
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, &bytes)?;
            let pso: ID3D12PipelineState =
                self.device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: bytes.as_ptr().cast(),
                            BytecodeLength: bytes.len(),
                        },
                        ..Default::default()
                    })?;
            Ok(CardWork {
                sync: vec![list()?],
                check: vec![list()?],
                fetch: list()?,
                root,
                pso,
                differs: (differs, data as usize, RESULTS),
                fetched: None,
                taken: 0,
                out: 0,
                inflight: if inflight_fetch() {
                    self.make_card_inflight(RESULTS)
                } else {
                    None
                },
            })
        }
    }

    fn make_card_inflight(&self, results: u64) -> Option<CardInflight> {
        let made = (|| -> windows::core::Result<CardInflight> {
            let bytes = crate::hlsl::compile_compute("card_fetch", CARD_FETCH)?;
            let slots = inflight_slots(inflight_fetch_bytes());
            let (table, table_data) = self
                .shared_buffer(4 + results)
                .ok_or_else(windows::core::Error::empty)?;
            name(&table, || "card fetch in flight: tickets".into());
            let fetched_bytes = u64::from(slots) * INFLIGHT_SLOT;
            let (fetched, fetched_data) = self
                .shared_buffer(fetched_bytes)
                .ok_or_else(windows::core::Error::empty)?;
            name(&fetched, || "card fetch in flight: pieces".into());
            // SAFETY: freshly mapped buffer; no GPU work references it yet.
            unsafe { table_data.cast::<u32>().write_volatile(0) };
            // SAFETY: compiled DXBC with embedded root signature.
            unsafe {
                let root: ID3D12RootSignature = self.device.CreateRootSignature(0, &bytes)?;
                let pso: ID3D12PipelineState =
                    self.device
                        .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                            pRootSignature: std::mem::transmute_copy(&root),
                            CS: D3D12_SHADER_BYTECODE {
                                pShaderBytecode: bytes.as_ptr().cast(),
                                BytecodeLength: bytes.len(),
                            },
                            ..Default::default()
                        })?;
                Ok(CardInflight {
                    root,
                    pso,
                    table: (table, table_data as usize, 4 + results),
                    fetched: (fetched, fetched_data as usize, slots),
                    landed: 0,
                    brought: Vec::new(),
                })
            }
        })();
        match made {
            Ok(inflight) => {
                tracing::info!(
                    slots = inflight.fetched.2,
                    "card copies' flagged pieces come down inside flights (VICTORY_CARD_FETCH_INFLIGHT=1)"
                );
                Some(inflight)
            }
            Err(e) => {
                tracing::warn!(error = %e, "no in-flight card fetch (VICTORY_CARD_FETCH_INFLIGHT=1)");
                None
            }
        }
    }

    /// Uploads dirty CPU ranges to card-backed shadows before the submission's
    /// command lists and records the post-submission [`CARD_COMPARE`] (and
    /// optional [`CARD_FETCH`]) command list.
    pub(super) fn card_before(
        &self,
        declared: Vec<Declared>,
        flown: bool,
        submitted: &mut u64,
    ) -> Option<CardPlan> {
        use std::sync::Arc;
        if !crate::shadow::cards() {
            return None;
        }
        let (prev_order, candidate_sent_as) = if relay() {
            QUEUE_TIMELINE.next_candidate()
        } else {
            (0, 0)
        };
        let mut dirty = crate::shadow::card_dirty(candidate_sent_as);
        let has_uploads = !dirty.is_empty();
        let sent_as = if has_uploads {
            candidate_sent_as
        } else {
            prev_order
        };
        if declared.is_empty() && dirty.is_empty() {
            return None;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let work = work.as_mut()?;
        let mut whole: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
        for (shadow, _) in &declared {
            if shadow.card_make_before(|size| {
                let before = self.uav_buffer(crate::shadow::card_hash_bytes(size))?;
                name(&before, || "title buffer on the card, hashes".into());
                Some(before)
            }) {
                whole.push(shadow.clone());
                if !dirty.iter().any(|(s, _)| Arc::ptr_eq(s, shadow)) {
                    dirty.push((shadow.clone(), Vec::new()));
                }
            }
        }
        let mut used: Vec<(bool, Prelude)> = Vec::new();
        if !dirty.is_empty() {
            let Some(sync) = work.sync.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!(
                    "no host list to bring card-backed shadows up to date"
                ));
            };
            let (allocator, list) = &sync;
            // SAFETY: recycled list is idle; buffers live through submission.
            let recorded = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    let mut compute_bound = false;
                    let joins = rehash_runs();
                    for (shadow, ranges) in &dirty {
                        let Some((staging, before, size)) = shadow.card_sides() else {
                            continue;
                        };
                        for &(a, b) in ranges {
                            list.CopyBufferRegion(
                                &shadow.buffer,
                                a as u64,
                                staging,
                                a as u64,
                                (b - a) as u64,
                            );
                        }
                        let bytes: u64 = ranges.iter().map(|&(a, b)| (b - a) as u64).sum();
                        crate::shadow::CARD_STATS[0].fetch_add(bytes, Ordering::Relaxed);
                        crate::shadow::CARD_STATS[1]
                            .fetch_add(ranges.len() as u64, Ordering::Relaxed);
                        let Some(before) = before else {
                            continue;
                        };
                        if !ranges.is_empty() {
                            list.ResourceBarrier(&[transition(
                                &shadow.buffer,
                                D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            )]);
                        }
                        if !compute_bound {
                            list.SetComputeRootSignature(&work.root);
                            list.SetPipelineState(&work.pso);
                            list.SetComputeRootUnorderedAccessView(
                                3,
                                work.differs.0.GetGPUVirtualAddress(),
                            );
                            list.SetComputeRootUnorderedAccessView(
                                4,
                                work.differs.0.GetGPUVirtualAddress(),
                            );
                            compute_bound = true;
                        }
                        list.SetComputeRootShaderResourceView(
                            1,
                            shadow.buffer.GetGPUVirtualAddress(),
                        );
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        let runs: Vec<crate::shadow::CardSpan> =
                            if whole.iter().any(|s| Arc::ptr_eq(s, shadow)) {
                                vec![shadow.card_run(0, usize::MAX)]
                            } else {
                                let mut runs: Vec<crate::shadow::CardSpan> = Vec::new();
                                for run in ranges.iter().map(|&(a, b)| shadow.card_run(a, b)) {
                                    match runs.last_mut() {
                                        Some(last)
                                            if joins && run.first <= last.first + last.pieces =>
                                        {
                                            last.pieces = last
                                                .pieces
                                                .max(run.first + run.pieces - last.first);
                                            last.to = last.to.max(run.to);
                                        }
                                        _ => runs.push(run),
                                    }
                                }
                                if joins && runs.len() > 1 {
                                    crate::shadow::joined_piece_runs(runs)
                                } else {
                                    runs
                                }
                            };
                        CARD_PASSES[0].fetch_add(runs.len() as u64, Ordering::Relaxed);
                        CARD_PASSES[1].fetch_add(
                            runs.iter().map(|r| u64::from(r.pieces)).sum::<u64>(),
                            Ordering::Relaxed,
                        );
                        for run in runs {
                            dispatch_card_compare(
                                list,
                                run.first,
                                run.skew,
                                size as u32,
                                run.pieces,
                                0,
                                0,
                                VIEW_MARK_NONE,
                            );
                        }
                    }
                    list.Close()?;
                    {
                        // Lock order: `FETCHER` (`direct_submit_guard()`) -> `queue_order` -> `last_signalled`.
                        let _lazy_order = crate::lazy_back::direct_submit_guard();
                        let mut order = relay().then(|| QUEUE_TIMELINE.lock_queue_order());
                        self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                        let submitted_order = if let Some(order) = order.as_mut() {
                            QUEUE_TIMELINE.commit_submitted_locked(order, sent_as, has_uploads)
                        } else {
                            0
                        };
                        if crate::lazy_back::can_arm() || crate::lazy_back::ever_active() {
                            *submitted += 1;
                            self.queue.Signal(&self.fence, *submitted)?;
                            self.note_signalled_with_order(*submitted, submitted_order);
                        }
                    }
                    Ok(())
                })()
            };
            if let Err(e) = recorded {
                fail(format_args!(
                    "card-backed shadows could not be updated ({e})"
                ));
            }
            used.push((false, sync));
        }
        let order_after = if relay() {
            QUEUE_TIMELINE.submitted_order()
        } else {
            0
        };
        let mut plan = CardPlan {
            ranges: Vec::new(),
            sent_as,
            check: None,
            _synced: dirty.into_iter().map(|(shadow, _)| shadow).collect(),
            lists: used,
            order_after,
            inflight: false,
            inflight_landed: 0,
        };
        let mut at = work.taken;
        for (shadow, ranges) in &declared {
            let mut sorted = ranges.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            let twin = shadow
                .card_sides()
                .is_some_and(|(_, before, _)| before.is_some());
            for (from, to) in spans {
                let Some(span) = shadow.card_span(from, to) else {
                    continue;
                };
                let compared = twin && u64::from(at + span.pieces) * 4 <= work.differs.2;
                plan.ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (from, to),
                    span,
                    at,
                    compared,
                });
                if compared {
                    at += span.pieces;
                }
            }
        }
        work.taken = at;
        if plan.ranges.iter().any(|r| r.compared) {
            let Some(check) = work.check.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list for card shadow compare"));
            };
            let (allocator, list) = &check;
            // SAFETY: recycled list is idle; buffers live through submission.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    list.ResourceBarrier(&[crate::indirect::uav_barrier()]);
                    list.SetComputeRootSignature(&work.root);
                    list.SetPipelineState(&work.pso);
                    list.SetComputeRootUnorderedAccessView(
                        3,
                        work.differs.0.GetGPUVirtualAddress(),
                    );
                    list.SetComputeRootUnorderedAccessView(
                        4,
                        work.differs.0.GetGPUVirtualAddress(),
                    );
                    for r in plan.ranges.iter().filter(|r| r.compared) {
                        let Some((_, Some(before), size)) = r.shadow.card_sides() else {
                            continue;
                        };
                        list.SetComputeRootShaderResourceView(
                            1,
                            r.shadow.buffer.GetGPUVirtualAddress(),
                        );
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        dispatch_card_compare(
                            list,
                            r.span.first,
                            r.span.skew,
                            size as u32,
                            r.span.pieces,
                            r.at * 4,
                            if flown { 2 } else { 1 },
                            VIEW_MARK_NONE,
                        );
                        CARD_PASSES[2].fetch_add(1, Ordering::Relaxed);
                    }
                    if flown && let Some(inflight) = &work.inflight {
                        list.ResourceBarrier(&[crate::indirect::uav_barrier()]);
                        list.SetComputeRootSignature(&inflight.root);
                        list.SetPipelineState(&inflight.pso);
                        list.SetComputeRootUnorderedAccessView(
                            2,
                            work.differs.0.GetGPUVirtualAddress(),
                        );
                        list.SetComputeRootUnorderedAccessView(
                            3,
                            inflight.table.0.GetGPUVirtualAddress(),
                        );
                        list.SetComputeRootUnorderedAccessView(
                            4,
                            inflight.fetched.0.GetGPUVirtualAddress(),
                        );
                        let mut every = true;
                        for r in plan.ranges.iter().filter(|r| r.compared) {
                            let Some((_, Some(_), size)) = r.shadow.card_sides() else {
                                every = false;
                                continue;
                            };
                            let groups = r.span.pieces.clamp(1, 65535);
                            let constants = [
                                r.span.first,
                                r.span.skew,
                                size as u32,
                                r.span.pieces,
                                r.at * 4,
                                inflight.landed,
                                inflight.fetched.2,
                                groups,
                            ];
                            list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(
                                1,
                                r.shadow.buffer.GetGPUVirtualAddress(),
                            );
                            list.Dispatch(groups, 1, 1);
                        }
                        plan.inflight = every;
                        plan.inflight_landed = inflight.landed;
                    }
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(list) => plan.check = Some(list),
                Err(e) => fail(format_args!("card shadow compare could not record ({e})")),
            }
            plan.lists.push((true, check));
        }
        work.out += 1;
        Some(plan)
    }

    pub(super) fn card_results_low(&self) -> bool {
        CARD_WORK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|work| u64::from(work.taken) * 8 > work.differs.2)
    }

    pub(super) fn card_done(&self, plan: CardPlan) {
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        for (check, list) in plan.lists {
            if check {
                work.check.push(list);
            } else {
                work.sync.push(list);
            }
        }
        work.out = work.out.saturating_sub(1);
        if !relay() || work.out == 0 {
            work.taken = 0;
            if let Some(inflight) = &mut work.inflight {
                inflight.brought.clear();
            }
        }
    }

    fn make_relay_fetch(&self) -> Option<RelayFetch> {
        // SAFETY: plain D3D12 object creation on our live device.
        let copy: Option<(ID3D12CommandQueue, Prelude)> = relay_copy_queue()
            .then_some(self.copy_queue.as_ref())
            .flatten()
            .and_then(|queue| unsafe {
                (|| -> windows::core::Result<(ID3D12CommandQueue, Prelude)> {
                    let allocator: ID3D12CommandAllocator = self
                        .device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
                    let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                        0,
                        D3D12_COMMAND_LIST_TYPE_COPY,
                        &allocator,
                        None,
                    )?;
                    list.Close()?;
                    Ok((queue.clone(), (allocator, list)))
                })()
                .map_err(|e| {
                    tracing::warn!(
                        error = %e,
                        "no copy-queue command list for relay fetches; falling back to direct queue"
                    )
                })
                .ok()
            });
        let (copy, list) = match copy {
            Some((queue, list)) => (Some(queue), list),
            None => (None, self.make_prelude()?),
        };
        // SAFETY: plain D3D12 fence and Win32 event creation.
        unsafe {
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(RelayFetch {
                list,
                copy,
                fence,
                event,
                sent: 0,
                buffer: None,
            })
        }
    }

    /// Unified relay readback on `Host::copy_queue` (falling back to `self.queue`
    /// when `VICTORY_RELAY_COPY_QUEUE=0` or unavailable): DMAs flagged card-copy
    /// pieces and ReBAR UAV spans in one submission, then merges GPU writes via
    /// `Shadow::card_merge_relay` and parallel SIMD `Shadow::relay_merge_rebar`.
    fn relay_fetch_back(
        &self,
        plans: &[&CardPlan],
        back: &[Declared],
        fetch: &mut Option<RelayFetch>,
    ) {
        struct RebarEntry {
            shadow: std::sync::Arc<crate::shadow::Shadow>,
            spans: Vec<(usize, usize)>,
            offsets: Vec<u64>,
        }

        let lazy = crate::lazy_back::enabled();
        let mut card_done_inflight = false;
        if inflight_fetch() {
            let landed = if lazy {
                false
            } else {
                self.relay_card_inflight(plans)
            };
            inflight_tickets_landed(plans);
            if landed {
                card_done_inflight = true;
            }
        }
        let mut pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)> =
            Vec::new();
        let mut card_total = 0u64;
        if !card_done_inflight {
            let differs = {
                let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
                work.as_ref()
                    .map(|work| (work.differs.1, (work.differs.2 / 4) as usize))
            };
            if let Some((differs, dwords)) = differs {
                // SAFETY: these plans' compares have finished on the GPU.
                let differs = unsafe { std::slice::from_raw_parts(differs as *const u32, dwords) };
                let mut seen = std::collections::HashSet::new();
                for plan in plans {
                    let flags: Vec<(usize, u32, bool)> = plan
                        .ranges
                        .iter()
                        .flat_map(|r| {
                            (0..r.span.pieces).map(move |p| {
                                let flagged = !r.compared
                                    || differs.get((r.at + p) as usize).is_none_or(|&d| d != 0);
                                (
                                    std::sync::Arc::as_ptr(&r.shadow) as usize,
                                    r.span.first + p,
                                    flagged,
                                )
                            })
                        })
                        .collect();
                    let fetched = flagged_pieces(&flags);
                    for r in &plan.ranges {
                        let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
                        for p in 0..r.span.pieces {
                            if !fetched.contains(&(copy, r.span.first + p)) {
                                continue;
                            }
                            let (lo, hi) = r.shadow.card_piece(&r.span, p);
                            if lo < hi && seen.insert((copy, lo, hi)) {
                                if lazy
                                    && crate::lazy_back::try_defer_page(
                                        &r.shadow,
                                        lo,
                                        hi - lo,
                                        plan.sent_as,
                                    )
                                {
                                    continue;
                                }
                                pieces.push((r.shadow.clone(), lo, hi, card_total));
                                card_total += (hi - lo) as u64;
                            }
                        }
                        crate::shadow::CARD_STATS[2]
                            .fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
                    }
                }
            }
        }

        let mut rebar_entries: Vec<RebarEntry> = Vec::new();
        let mut total = card_total;
        for (shadow, declared) in back {
            if !shadow.is_rebar() {
                continue;
            }
            let ranges: Vec<(usize, usize)> = declared
                .iter()
                .map(|&(from, to)| shadow.inside(from, to.saturating_sub(from)))
                .filter(|&(from, to)| from < to)
                .collect();
            let merged = merge_ranges(ranges);
            if merged.is_empty() {
                continue;
            }
            let spans = if lazy {
                let start = shadow.start() as usize;
                let page = crate::watch::PAGE;
                let mut kept: Vec<(usize, usize)> = Vec::new();
                for (from, to) in merged {
                    let mut at = from;
                    while at < to {
                        let va = start + at;
                        let step = (page - va % page).min(to - at);
                        let next = at + step;
                        if crate::lazy_back::try_defer_page(shadow, at, step, 0) {
                            shadow.mark_rebar_dirty_range(at, step);
                        } else {
                            match kept.last_mut() {
                                Some(last) if last.1 == at => last.1 = next,
                                _ => kept.push((at, next)),
                            }
                        }
                        at = next;
                    }
                }
                kept
            } else {
                merged
            };
            if spans.is_empty() {
                continue;
            }
            let mut offsets = Vec::with_capacity(spans.len());
            for &(from, to) in &spans {
                offsets.push(total);
                total += (to - from) as u64;
            }
            rebar_entries.push(RebarEntry {
                shadow: shadow.clone(),
                spans,
                offsets,
            });
        }
        let rebar_total = total - card_total;
        let has_card = !pieces.is_empty();
        let has_rebar = !rebar_entries.is_empty();

        let completed_order = plans.iter().map(|p| p.order_after).max().unwrap_or(0);
        if !card_done_inflight && !has_card {
            let max_sent_as = plans.iter().map(|p| p.sent_as).max();
            let queue_order = if !inflight_fetch() && self.copy_queue.is_none() {
                *QUEUE_TIMELINE.lock_queue_order()
            } else {
                0
            };
            if let Some(mark) = relay_no_card_fetch_watermark(
                inflight_fetch(),
                self.copy_queue.is_some(),
                max_sent_as,
                completed_order,
                queue_order,
            ) {
                crate::shadow::relay_fetched(mark);
            }
        }
        if !has_card && !has_rebar {
            return;
        }
        if has_card && inflight_fetch() {
            INFLIGHT_STATS[3].fetch_add(1, Ordering::Relaxed);
        }
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        let Some(f) = fetch.as_mut() else {
            fail(format_args!(
                "no list, fence or event for relay readback (VICTORY_RELAY)"
            ));
        };
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self
                .shared_buffer(size)
                .map(|(buffer, data)| (buffer, data as usize, size));
        }
        let Some((buffer, data, _)) = f.buffer.clone() else {
            fail(format_args!(
                "no memory for relay readback ({total} bytes, VICTORY_RELAY)"
            ));
        };
        f.sent += 1;
        let ((allocator, list), fence, event, sent) =
            (f.list.clone(), f.fence.clone(), f.event, f.sent);
        let copy = f.copy.clone();
        let landed = plans.iter().map(|p| p.sent_as).max().unwrap_or(0);
        let waiting = std::time::Instant::now();
        // SAFETY: `list` is idle; all source/dest buffers are live and in COMMON
        // state; `fence` and `event` are owned exclusively by the relay worker.
        let ran = unsafe {
            (|| -> windows::core::Result<u64> {
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                for (shadow, lo, hi, at) in &pieces {
                    list.CopyBufferRegion(
                        &buffer,
                        *at,
                        &shadow.buffer,
                        *lo as u64,
                        (*hi - *lo) as u64,
                    );
                }
                for entry in &rebar_entries {
                    for (&(from, to), &at) in entry.spans.iter().zip(&entry.offsets) {
                        list.CopyBufferRegion(
                            &buffer,
                            at,
                            &entry.shadow.buffer,
                            from as u64,
                            (to - from) as u64,
                        );
                    }
                }
                list.Close()?;
                let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                if let Some(queue) = &copy {
                    let after = self.sync_copy_queue(queue)?;
                    queue.ExecuteCommandLists(&lists);
                    queue.Signal(&fence, sent)?;
                    fence.SetEventOnCompletion(sent, event)?;
                    WaitForSingleObject(event, INFINITE);
                    return Ok(after.max(landed));
                }
                let order = QUEUE_TIMELINE.lock_queue_order();
                self.queue.ExecuteCommandLists(&lists);
                let after = *order;
                drop(order);
                self.queue.Signal(&fence, sent)?;
                fence.SetEventOnCompletion(sent, event)?;
                WaitForSingleObject(event, INFINITE);
                Ok(after)
            })()
        };
        let after = match ran {
            Ok(after) => after,
            Err(e) => fail(format_args!("relay readback failed ({e}, VICTORY_RELAY)")),
        };
        let us = waiting.elapsed().as_micros() as u64;
        RELAY_STATS[3].fetch_add(us, Ordering::Relaxed);
        if has_card {
            let landed_max = plans.iter().map(|p| p.sent_as).max();
            if inflight_fetch()
                && should_record_brought(after, landed_max)
                && let Some(inflight) = CARD_WORK
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_mut()
                    .and_then(|w| w.inflight.as_mut())
            {
                inflight.brought.extend(
                    pieces
                        .iter()
                        .map(|(shadow, lo, hi, _)| (shadow.clone(), *lo, *hi, after - 1)),
                );
            }
            {
                use crate::shadow::CARD_STATS;
                CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
                CARD_STATS[4].fetch_add(card_total, Ordering::Relaxed);
                CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
                CARD_STATS[6].fetch_add(us, Ordering::Relaxed);
            }
            let merge_pieces: Vec<RelayCardPiece> = pieces
                .into_iter()
                .map(|(shadow, lo, hi, at)| RelayCardPiece {
                    shadow,
                    lo,
                    hi,
                    gpu_ptr: data + at as usize,
                    after,
                })
                .collect();
            let written = relay_merge_card_pieces(merge_pieces, wide_merge());
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            let mark = fallback_fetched_mark(inflight_fetch(), after, landed_max);
            crate::shadow::relay_fetched(mark);
        }
        if has_rebar {
            RELAY_REBAR_STATS[0].fetch_add(1, Ordering::Relaxed);
            RELAY_REBAR_STATS[1].fetch_add(rebar_total, Ordering::Relaxed);
            RELAY_REBAR_STATS[2].fetch_add(us, Ordering::Relaxed);
            let mut written = 0usize;
            for entry in &rebar_entries {
                let slices: Vec<&[u8]> = entry
                    .spans
                    .iter()
                    .zip(&entry.offsets)
                    .map(|(&(from, to), &at)| {
                        // SAFETY: DMA fence completed and `data` covers `total` bytes.
                        unsafe {
                            std::slice::from_raw_parts((data + at as usize) as *const u8, to - from)
                        }
                    })
                    .collect();
                let Some(merged) = entry.shadow.relay_merge_rebar(&entry.spans, &slices) else {
                    fail(format_args!(
                        "relay ReBAR readback merge failed: title read or lazy materialization failed (VICTORY_RELAY)"
                    ));
                };
                written += merged;
                entry.shadow.mark_rebar_dirty_spans(&entry.spans);
            }
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    pub(super) fn relay_rebar_back(&self, back: &[Declared], fetch: &mut Option<RelayFetch>) {
        self.relay_fetch_back(&[], back, fetch);
    }

    fn relay_card_inflight(&self, plans: &[&CardPlan]) -> bool {
        use std::sync::Arc;
        let mut pieces: Vec<RelayCardPiece> = Vec::new();
        {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            let Some(work) = work.as_ref() else {
                return false;
            };
            let Some(inflight) = &work.inflight else {
                return false;
            };
            let dwords = (work.differs.2 / 4) as usize;
            // SAFETY: these plans' compares and fetches have completed on the GPU.
            let (differs, table) = unsafe {
                (
                    std::slice::from_raw_parts(work.differs.1 as *const u32, dwords),
                    std::slice::from_raw_parts((inflight.table.1 as *const u32).add(1), dwords),
                )
            };
            let slots = inflight.fetched.2;
            for plan in plans {
                if plan.ranges.is_empty() {
                    continue;
                }
                if !plan.inflight || plan.ranges.iter().any(|r| !r.compared) {
                    return false;
                }
                let mut tickets: std::collections::HashMap<(usize, u32), u32> =
                    std::collections::HashMap::new();
                let mut flags: Vec<(usize, u32, bool)> = Vec::new();
                for r in &plan.ranges {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let at = (r.at + p) as usize;
                        let (Some(&flag), Some(&entry)) = (differs.get(at), table.get(at)) else {
                            return false;
                        };
                        if flag != 0 {
                            if entry == INFLIGHT_NONE {
                                return false;
                            }
                            if !inflight_gets_slot(entry, plan.inflight_landed, slots) {
                                INFLIGHT_STATS[2].fetch_add(1, Ordering::Relaxed);
                                return false;
                            }
                            tickets.insert((copy, r.span.first + p), entry);
                        }
                        flags.push((copy, r.span.first + p, flag != 0));
                    }
                }
                let fetched = flagged_pieces(&flags);
                let mut seen = std::collections::HashSet::new();
                for r in &plan.ranges {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let piece = r.span.first + p;
                        if !fetched.contains(&(copy, piece)) {
                            continue;
                        }
                        let (lo, hi) = r.shadow.card_piece(&r.span, p);
                        if lo >= hi || !seen.insert((copy, lo, hi)) {
                            continue;
                        }
                        let from = inflight_piece_from(&r.span, p);
                        let Some(&k) = tickets.get(&(copy, piece)) else {
                            return false;
                        };
                        if lo < from || (hi - from) as u64 > INFLIGHT_SLOT {
                            return false;
                        }
                        let slot =
                            inflight.fetched.1 + (k % slots) as usize * INFLIGHT_SLOT as usize;
                        let newer: Vec<(usize, usize)> = inflight
                            .brought
                            .iter()
                            .filter(|b| {
                                b.3 >= plan.sent_as
                                    && Arc::ptr_eq(&b.0, &r.shadow)
                                    && b.1 < hi
                                    && b.2 > lo
                            })
                            .map(|b| (b.1, b.2))
                            .collect();
                        for (a, b) in crate::shadow::outside(lo, hi, newer) {
                            pieces.push(RelayCardPiece {
                                shadow: r.shadow.clone(),
                                lo: a,
                                hi: b,
                                gpu_ptr: slot + (a - from),
                                after: plan.sent_as,
                            });
                        }
                    }
                }
            }
        }
        // Outside `CARD_WORK`: slots remain pinned until `inflight_tickets_landed` runs after this.
        let piece_count = pieces.len() as u64;
        let bytes: u64 = pieces.iter().map(|p| (p.hi - p.lo) as u64).sum();
        let written = relay_merge_card_pieces(pieces, wide_merge());
        for plan in plans {
            for r in &plan.ranges {
                crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
            }
        }
        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        INFLIGHT_STATS[0].fetch_add(piece_count, Ordering::Relaxed);
        INFLIGHT_STATS[1].fetch_add(bytes, Ordering::Relaxed);
        if let Some(last) = plans.iter().map(|p| p.sent_as).max() {
            crate::shadow::relay_fetched(last);
        }
        true
    }

    fn relay_land(&self, items: Vec<Late>, fetch: &mut Option<RelayFetch>) {
        let landing = std::time::Instant::now();
        {
            let flights: Vec<&Flight> = items
                .iter()
                .filter_map(|l| match l {
                    Late::Run(flight) => Some(flight),
                    _ => None,
                })
                .collect();
            if !flights.is_empty() {
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = flights
                        .iter()
                        .flat_map(|f| f.lists.iter().map(crate::queue::InFlight::list))
                        .collect();
                    fail(format_args!(
                        "host GPU device removed after {} flights (VICTORY_RELAY) of {} lists{}{}",
                        flights.len(),
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                let plans: Vec<&CardPlan> =
                    flights.iter().filter_map(|f| f.card.as_ref()).collect();
                let mut back: Vec<Declared> = Vec::new();
                for flight in &flights {
                    for (shadow, ranges) in &flight.back {
                        match back
                            .iter_mut()
                            .find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow))
                        {
                            Some((_, all)) => {
                                for range in ranges {
                                    if !all.contains(range) {
                                        all.push(*range);
                                    }
                                }
                            }
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                fill_ahead::landing(flights.iter().map(|f| f.number).min().unwrap_or(0));
                self.relay_fetch_back(&plans, &back, fetch);
                relay_hand_back(&back);
                fill_ahead::landing(0);
                COPY_BACK[0].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
                RELAY_STATS[0].fetch_add(1, Ordering::Relaxed);
                RELAY_STATS[1].fetch_add(flights.len() as u64, Ordering::Relaxed);
            }
        }
        let mut last = 0;
        let mut landed = Vec::new();
        let mut backs: Vec<Declared> = Vec::new();
        let mut woken = crate::fence::Woken::default();
        for item in items {
            match item {
                Late::Run(flight) => {
                    last = flight.number;
                    for entry in &flight.lists {
                        if !read_backs(entry.list()) {
                            fail(format_args!(
                                "relay readback failed: lazy copy-back materialization failed (VICTORY_RELAY)"
                            ));
                        }
                    }
                    if let Some(card) = flight.card {
                        self.card_done(card);
                    }
                    landed.push(flight.lists);
                    for (shadow, ranges) in flight.back {
                        backs.push((shadow, ranges));
                    }
                }
                Late::Write(at, what, bytes) => {
                    {
                        let mut line = line();
                        if let Some(pos) = line
                            .in_hand
                            .iter()
                            .position(|&(a, _, b)| a == at && b == bytes)
                        {
                            line.in_hand.remove(pos);
                        }
                    }
                    woken.add(what.write(at as *mut u8, bytes));
                }
                Late::Signal(fence, value) => {
                    woken.add(fence.signal_deferred(value));
                }
            }
        }
        if last > 0 {
            FINISHED.store(last, Ordering::Release);
        }
        self.log_messages();
        RELAY_STATS[4].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
        // Release landed shadow write ranges BEFORE clearing `line.settling` and
        // notifying `LINE_LANDED`, so woken submits never see landed flights as
        // still pending in `shadow.pending`/`flights`.
        for (shadow, ranges) in &backs {
            shadow.copy_back_done(ranges);
        }
        {
            let mut line = line();
            let Line {
                line: items,
                written,
                lists,
                settling,
                poke,
                landed: kept,
                in_hand,
                in_hand_back,
                pending_woken,
                ..
            } = &mut *line;
            kept.extend(landed);
            pending_woken.push(woken);
            in_hand.clear();
            in_hand_back.clear();
            written.clear();
            *lists = 0;
            let mut ranges = Vec::new();
            for item in items.iter() {
                if let Late::Run(flight) = item {
                    *lists += flight.lists.len().max(1);
                    ranges.extend(flight.back.iter().flat_map(|(_, r)| r.iter().copied()));
                    ranges.extend(flight.readback_ranges.iter().copied());
                }
            }
            join_ranges(written, ranges);
            *settling = false;
            *poke = true;
        }
        LINE_MOVED.notify_all();
        LINE_LANDED.notify_all();
    }

    /// Synchronous card-copy readback after the submission's fence completed.
    pub(super) fn card_back(
        &self,
        plan: &CardPlan,
        submitted: &mut u64,
        seen: &mut std::collections::HashSet<(usize, usize, usize)>,
    ) {
        if plan.ranges.is_empty() {
            return;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        // SAFETY: the compare pass has completed on the GPU.
        let differs = unsafe {
            std::slice::from_raw_parts(work.differs.1 as *const u32, (work.differs.2 / 4) as usize)
        };
        let mut pieces: Vec<(usize, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        for (i, r) in plan.ranges.iter().enumerate() {
            for p in 0..r.span.pieces {
                if r.compared && differs[(r.at + p) as usize] == 0 {
                    continue;
                }
                let (lo, hi) = r.shadow.card_piece(&r.span, p);
                if lo < hi && seen.insert((std::sync::Arc::as_ptr(&r.shadow) as usize, lo, hi)) {
                    pieces.push((i, lo, hi, total));
                    total += (hi - lo) as u64;
                }
            }
            crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
        }
        let mut written = vec![0usize; plan.ranges.len()];
        if !pieces.is_empty() {
            if work.fetched.as_ref().is_none_or(|f| f.2 < total) {
                let size = total.next_multiple_of(16 << 20);
                work.fetched = self
                    .shared_buffer(size)
                    .map(|(buffer, data)| (buffer, data as usize, size));
            }
            let Some((fetched, data, _)) = &work.fetched else {
                fail(format_args!(
                    "no memory to fetch card shadow writes ({total} bytes)"
                ));
            };
            let waiting = std::time::Instant::now();
            let (allocator, list) = &work.fetch;
            // SAFETY: serialized under `submitted`; buffers are live and idle.
            let ran = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    for &(i, lo, hi, at) in &pieces {
                        list.CopyBufferRegion(
                            fetched,
                            at,
                            &plan.ranges[i].shadow.buffer,
                            lo as u64,
                            (hi - lo) as u64,
                        );
                    }
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    *submitted += 1;
                    self.queue.Signal(&self.fence, *submitted)?;
                    self.note_signalled(*submitted);
                    self.fence.SetEventOnCompletion(*submitted, self.event)?;
                    let waited = WaitForSingleObject(self.event, INFINITE);
                    if waited != WAIT_OBJECT_0 {
                        self.poll_fence(*submitted);
                    }
                    Ok(())
                })()
            };
            if let Err(e) = ran {
                fail(format_args!("card shadow readback failed ({e})"));
            }
            {
                use crate::shadow::CARD_STATS;
                CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
                CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
                CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
                CARD_STATS[6].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            if wide_merge() && pieces.len() >= 8 {
                let fetched = *data;
                let counts: std::sync::Arc<Vec<std::sync::atomic::AtomicUsize>> =
                    std::sync::Arc::new(
                        (0..plan.ranges.len())
                            .map(|_| std::sync::atomic::AtomicUsize::new(0))
                            .collect(),
                    );
                let each = pieces.len().div_ceil(8);
                let jobs: Vec<Box<dyn FnOnce() + Send>> = pieces
                    .chunks(each)
                    .map(|chunk| {
                        let chunk: Vec<(
                            std::sync::Arc<crate::shadow::Shadow>,
                            usize,
                            usize,
                            usize,
                            u64,
                        )> = chunk
                            .iter()
                            .map(|&(i, lo, hi, at)| (plan.ranges[i].shadow.clone(), i, lo, hi, at))
                            .collect();
                        let counts = counts.clone();
                        Box::new(move || {
                            let mut title = Vec::new();
                            for (shadow, i, lo, hi, at) in chunk {
                                // SAFETY: DMA completed; slice stays live until `wide_run` returns.
                                let gpu = unsafe {
                                    std::slice::from_raw_parts(
                                        (fetched + at as usize) as *const u8,
                                        hi - lo,
                                    )
                                };
                                let merged = shadow.card_merge(lo, gpu, &mut title);
                                counts[i].fetch_add(merged, Ordering::Relaxed);
                            }
                        }) as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                crate::shadow::wide_run(jobs);
                for (written, count) in written.iter_mut().zip(counts.iter()) {
                    *written += count.load(Ordering::Relaxed);
                }
            } else {
                let mut title = Vec::new();
                for &(i, lo, hi, at) in &pieces {
                    // SAFETY: DMA completed; slice covers `hi - lo` bytes.
                    let gpu = unsafe {
                        std::slice::from_raw_parts((*data + at as usize) as *const u8, hi - lo)
                    };
                    written[i] += plan.ranges[i].shadow.card_merge(lo, gpu, &mut title);
                }
            }
        }
        for (r, written) in plan.ranges.iter().zip(written) {
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            let _ = r.declared;
        }
    }

    /// Submits `in_flight` asynchronously without waiting for the GPU when
    /// [`late`] is active, offloading host-list recycling (`land()`) to
    /// `"victory-late-wait"` with a backpressure valve at 32 pending batches.
    pub(crate) fn execute_late(&self, in_flight: Vec<crate::queue::InFlight>) {
        if in_flight.is_empty() {
            return;
        }
        late_threads();
        let sent = {
            let runs: Vec<&HostList> = in_flight.iter().map(crate::queue::InFlight::list).collect();
            self.submit(&runs, true)
        };
        let Some((mut submitted, sent)) = sent else {
            // Synchronous fallback (e.g. duplicate HostList pointers in one batch).
            for entry in in_flight {
                let list = entry.take();
                if let Ok(orphan) = std::sync::Arc::try_unwrap(list) {
                    self.recycle_list(orphan);
                }
            }
            return;
        };
        let told = !sent.writes.is_empty();
        let most = {
            let mut line = line();
            let Line {
                line: items,
                written,
                lists: flying,
                target,
                ..
            } = &mut *line;
            join_ranges(
                written,
                sent.back
                    .iter()
                    .flat_map(|(_, ranges)| ranges.iter().copied())
                    .chain(sent.readback_ranges.iter().copied()),
            );
            *flying += in_flight.len();
            *target = *submitted;
            items.push(Late::Run(Flight {
                number: sent.number,
                fence: *target,
                lists: in_flight,
                card: sent.card,
                back: sent.back,
                readback_ranges: sent.readback_ranges,
                _exec_epoch: sent.exec_epoch,
            }));
            items.extend(
                sent.writes
                    .into_iter()
                    .map(|(at, what, bytes)| Late::Write(at, what, bytes)),
            );
            *flying >= late_most()
        };
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        RELAY_MOVED.notify_one();
        let mut woken = crate::fence::Woken::default();
        if told && !late_writes() {
            woken.add(self.settle(&mut submitted, WHY_WRITES));
        } else if most && relay_exact() {
            relay_wait(WHY_MOST, |line| line.lists < late_most());
        } else if most {
            woken.add(self.settle(&mut submitted, WHY_MOST));
        } else {
            LINE_MOVED.notify_one();
        }
        drop(submitted);
        woken.run();
        if !LATE_HELPER_ALIVE.load(Ordering::Acquire) {
            land();
            drain_woken();
        } else {
            let pending = {
                let mut l = line();
                let n = l.landed.len();
                if n > 0 && !l.poke {
                    l.poke = true;
                    LINE_MOVED.notify_one();
                }
                n
            };
            // Backpressure safety valve if `victory-late-wait` ever falls far behind.
            if pending >= 32 {
                land();
            }
        }
    }

    /// Settles only the flights in `LINE` whose `flight.back` overlaps `stale`
    /// when range-scoped settle is enabled under [`relay_exact`], falling back
    /// to a full [`Host::settle`] when `VICTORY_RANGE_SETTLE=0` or `!relay_exact()`.
    pub(super) fn settle_stale(
        &self,
        submitted: &mut u64,
        stale: &[(usize, usize, usize)],
    ) -> crate::fence::Woken {
        if !late() || stale.is_empty() {
            return crate::fence::Woken::default();
        }
        if range_settle() && relay_exact() {
            relay_wait(WHY_STALE, |line| !line_overlaps_stale(line, stale));
            crate::fence::Woken::default()
        } else {
            self.settle(submitted, WHY_STALE)
        }
    }

    /// Waits for all in-flight submissions to complete and copies back their
    /// shader writes, readbacks, bottom-of-pipe writes, and fence signals.
    /// Returns any [`crate::fence::Woken`] timelines to be run AFTER releasing
    /// `Host::submitted`.
    pub(super) fn settle(&self, submitted: &mut u64, why: usize) -> crate::fence::Woken {
        let mut total_woken = crate::fence::Woken::default();
        if !late() {
            return total_woken;
        }
        if relay() {
            relay_wait(why, |line| line.line.is_empty() && !line.settling);
            return total_woken;
        }
        let mut counted = false;
        loop {
            let items = {
                let mut line = line();
                if line.line.is_empty() {
                    line.settling = false;
                    break;
                }
                line.settling = true;
                line.written.clear();
                line.lists = 0;
                std::mem::take(&mut line.line)
            };
            let flights = items.iter().filter(|i| matches!(i, Late::Run(_))).count();
            if !counted {
                counted = true;
                LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
                LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
            }
            LATE_STATS[11].fetch_add(flights as u64, Ordering::Relaxed);
            if flights > 0 {
                let waiting = std::time::Instant::now();
                // SAFETY: fence and event are serialized under `submitted`.
                unsafe {
                    match self.fence.SetEventOnCompletion(*submitted, self.event) {
                        Ok(()) => {
                            let waited = WaitForSingleObject(self.event, INFINITE);
                            if waited != WAIT_OBJECT_0 {
                                self.poll_fence(*submitted);
                            }
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "flights wait failed; polling fence");
                            self.poll_fence(*submitted);
                        }
                    }
                }
                let gpu = waiting.elapsed();
                LATE_STATS[12].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                LATE_WAITED[why.min(10)].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                // SAFETY: plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = items
                        .iter()
                        .filter_map(|i| match i {
                            Late::Run(f) => Some(f),
                            _ => None,
                        })
                        .flat_map(|f| f.lists.iter().map(crate::queue::InFlight::list))
                        .collect();
                    fail(format_args!(
                        "host GPU device removed after {flights} flights (VICTORY_LATE_WAIT) of {} lists{}{}",
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                let copying = std::time::Instant::now();
                let mut seen = std::collections::HashSet::new();
                let mut back: Vec<Declared> = Vec::new();
                for item in &items {
                    let Late::Run(flight) = item else {
                        continue;
                    };
                    if let Some(card) = &flight.card {
                        self.card_back(card, submitted, &mut seen);
                    }
                    for (shadow, ranges) in &flight.back {
                        match back
                            .iter_mut()
                            .find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow))
                        {
                            Some((_, all)) => {
                                for range in ranges {
                                    if !all.contains(range) {
                                        all.push(*range);
                                    }
                                }
                            }
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                self.rebar_back(submitted, &back);
                hand_back(&back);
                for (shadow, ranges) in &back {
                    if shadow.carded() {
                        shadow.copy_back_done(ranges);
                    }
                }
                COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            let mut last = 0;
            let mut landed = Vec::new();
            for item in items {
                match item {
                    Late::Run(flight) => {
                        last = flight.number;
                        for entry in &flight.lists {
                            if !read_backs(entry.list()) {
                                fail(format_args!(
                                    "settle readback failed: lazy copy-back materialization failed"
                                ));
                            }
                        }
                        if let Some(card) = flight.card {
                            self.card_done(card);
                        }
                        landed.push(flight.lists);
                    }
                    Late::Write(at, what, bytes) => {
                        total_woken.add(what.write(at as *mut u8, bytes));
                    }
                    Late::Signal(fence, value) => {
                        total_woken.add(fence.signal_deferred(value));
                    }
                }
            }
            if last > 0 {
                FINISHED.store(last, Ordering::Release);
            }
            line().landed.extend(landed);
        }
        if counted {
            self.log_messages();
            if why != WHY_CARD_DONE {
                line().poke = true;
                LINE_MOVED.notify_one();
            }
        }
        total_woken
    }

    /// Synchronous batched DMA readback for ReBAR shadows in `back` (used by
    /// non-relay `Host::settle`).
    fn rebar_back(&self, submitted: &mut u64, back: &[Declared]) {
        struct RebarMerged {
            shadow: std::sync::Arc<crate::shadow::Shadow>,
            spans: Vec<(usize, usize)>,
            first: usize,
            end: usize,
        }
        let mut merged: Vec<RebarMerged> = Vec::new();
        let mut total_jobs = 0usize;
        for (shadow, declared) in back {
            if !shadow.is_rebar() {
                continue;
            }
            let ranges: Vec<(usize, usize)> = declared
                .iter()
                .map(|&(from, to)| shadow.inside(from, to.saturating_sub(from)))
                .filter(|&(from, to)| from < to)
                .collect();
            let spans = merge_ranges(ranges);
            let first = total_jobs;
            total_jobs += spans.len();
            merged.push(RebarMerged {
                shadow: shadow.clone(),
                spans,
                first,
                end: total_jobs,
            });
        }
        if merged.is_empty() {
            return;
        }
        if total_jobs == 0 {
            for entry in &merged {
                entry.shadow.clear_pending_rebar();
            }
            return;
        }
        let mut copy_jobs: Vec<(&ID3D12Resource, u64, u64)> = Vec::with_capacity(total_jobs);
        for entry in &merged {
            for &(from, to) in &entry.spans {
                copy_jobs.push((&entry.shadow.buffer, from as u64, (to - from) as u64));
            }
        }
        let ok = self.copy_rebar_readback(submitted, &copy_jobs, |slices| {
            for entry in &merged {
                if entry.first < entry.end {
                    match entry
                        .shadow
                        .copy_back_rebar(&entry.spans, &slices[entry.first..entry.end])
                    {
                        Some(written) => {
                            entry.shadow.mark_rebar_dirty_spans(&entry.spans);
                            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                        }
                        None => {
                            entry.shadow.mark_rebar_dirty_spans(&entry.spans);
                        }
                    }
                } else {
                    entry.shadow.clear_pending_rebar();
                }
            }
        });
        if !ok {
            for entry in &merged {
                if !entry.spans.is_empty() {
                    entry.shadow.mark_rebar_dirty_spans(&entry.spans);
                } else {
                    entry.shadow.clear_pending_rebar();
                }
            }
        }
    }
}
