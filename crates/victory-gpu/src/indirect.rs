//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution (or
//! gathered into a list's prelude via [`gather`], or hoisted into a preceding
//! list via `VICTORY_HOIST_CLAMPS`), by a compute shader of ours (HLSL
//! compiled by [`crate::hlsl`], with its root signature inside), leaving the
//! title's argument and count buffers untouched.
//!
//! Non-indexed rectangle-list draws follow the clamp with GPU-side
//! rectangle expansion in [`crate::rectangles`].
//!
//! When a count buffer is given, threads at or past `min(max, count)` return
//! immediately without charging the list's budget or reading uninitialized
//! tail slots. Each list has a budget (`2^31` vertex instances or threads,
//! groups counted as 64 threads, mesh groups as 128): active records past it
//! run empty, saturating without 32-bit wraparound (modelled and tested in
//! `indirect_budget`) so a list of thousands of garbage records cannot run
//! for seconds either.
//!
//! shortcut: the bounds (`2^24` vertices, `2^26` vertex instances, 65,535
//! groups per dimension and `2^20` per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::host::{HostList, host, transition};

const SOURCE: &str = include_str!("../shaders/argument_clamp.hlsl");
const MESH_SOURCE: &str = include_str!("../shaders/mesh_draws.hlsl");

/// A mesh dispatch's `kind` bit for [`bound`]: its records are clamped as rows,
/// each [`ROW_STRIDE`] bytes in the scratch buffer (two root constants, then
/// the groups).
pub(crate) const ROWS: u32 = 1 << 16;
/// See [`ROWS`].
pub(crate) const ROW_STRIDE: u32 = 20;

/// The scratch bytes of `records` clamped records `stride` bytes apart in the
/// title's buffer ([`ROWS`]: [`ROW_STRIDE`] apart in the scratch buffer).
fn scratch_bytes(kind: u32, stride: u32, records: u32) -> u64 {
    let step = if kind & ROWS != 0 { ROW_STRIDE } else { stride };
    u64::from(step) * u64::from(records)
}

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_for(device)
}

fn make_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    pass_for(device, "argument clamp", SOURCE)
}

/// A compute pass of ours from HLSL `source`, named `name` in messages.
fn pass_for(device: &ID3D12Device, name: &str, source: &str) -> windows::core::Result<Clamp> {
    // Wine's d3dcompiler_47 refuses it; `compile_compute` then uses DXC.
    let bytes = crate::hlsl::compile_compute(name, source)?;
    let bytes = &bytes[..];
    // SAFETY: plain object creation; the root signature is inside the shader.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Minimum record byte stride for a supported PC indirect argument `kind`.
fn min_stride(kind: u32) -> Option<u32> {
    match kind & 0xff {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Validates record alignment and returns the scratch byte size `stride * records`.
pub(crate) fn validate_layout(
    (kind, stride, records): (u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    let min = min_stride(kind)?;
    if stride < min || !stride.is_multiple_of(4) || !src_offset.is_multiple_of(4) {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let bytes = u64::from(stride).checked_mul(u64::from(records))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

/// Records, in `list`, the non-destructive clamp of `records` records of `kind`
/// (PC argument numbering) `stride` bytes apart from `src` into a list-owned
/// scratch buffer, leaving the scratch buffer in `UNORDERED_ACCESS` with a UAV
/// barrier after the write (for passes such as [`crate::rectangles`] that read
/// the clamped records in another compute shader). Callers return the scratch
/// buffer to `COMMON` via [`done_uav`] once done reading it. Replaces the
/// list's compute root signature and pipeline. `None` if the clamp or layout
/// is not available.
pub(crate) fn clamp_uav(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    validate_layout(spec, src.1, None)?;
    let bytes = scratch_bytes(spec.0, spec.1, spec.2);
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    let wait = list.take_buffer_writes() || every_wait();
    record_clamp_inner(
        c,
        &list.list,
        &dst,
        budget,
        src,
        None,
        spec,
        false,
        wait,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    )
    .then_some(dst)
}

/// Returns a scratch buffer left in `UNORDERED_ACCESS` by [`clamp_uav`] back
/// to `COMMON` for reuse by subsequent indirect commands on `list`.
pub(crate) fn done_uav(list: &HostList, scratch: &ID3D12Resource) {
    // SAFETY: an open list and a live scratch buffer in UNORDERED_ACCESS.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            scratch,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )]);
    }
}

/// Records, in `list`, the non-destructive clamp of up to `records` records of
/// `kind` (PC argument numbering) `stride` bytes apart from `src` (and bounded
/// by the 32-bit count in `count` when given) into a device-local scratch
/// buffer, leaving that scratch buffer in `INDIRECT_ARGUMENT` at offset `0`.
/// Replaces the list's compute root signature and pipeline (callers bind
/// theirs after). `None` if the clamp or alignment is not available.
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let bytes = scratch_bytes(spec.0, spec.1, spec.2);
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    let wait = list.take_buffer_writes() || every_wait();
    record_clamp_with(c, &list.list, &dst, budget, src, count, spec, false, wait).then_some(dst)
}

/// Ensures earlier compute writes to indirect argument/count buffers are
/// visible before a direct (unclamped or amplified) draw, without running the
/// clamp compute shader or invalidating compute pipeline state.
pub(crate) fn wait_direct(list: &HostList) {
    // SAFETY: an open command list; global UAV barrier has no resource hazards.
    unsafe { list.list.ResourceBarrier(&[uav_barrier()]) };
}

/// [`bound`] for a window of records clamped ahead of the draws that will
/// read them (`crate::draw`'s windows; no count buffer). Unfilled tail slots
/// that ask for more than can run are left empty and charge nothing.
pub(crate) fn bound_ahead(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    validate_layout(spec, src.1, None)?;
    let bytes = scratch_bytes(spec.0, spec.1, spec.2);
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    let wait = list.take_buffer_writes() || every_wait();
    record_clamp_with(c, &list.list, &dst, budget, src, None, spec, true, wait).then_some(dst)
}

/// `VICTORY_INDIRECT_WAITS=1` (opt-in; off by default): emit a global UAV
/// barrier both before and after every indirect argument clamp. By default,
/// the pre-clamp UAV barrier is emitted only when a shader that can store to
/// buffers was bound on the list since the last wait, and the post-clamp state
/// transition (`UNORDERED_ACCESS -> INDIRECT_ARGUMENT`) orders the indirect
/// draw without an extra global UAV barrier.
fn every_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_WAITS").is_some_and(|v| v == "1"))
}

#[allow(clippy::too_many_arguments)]
fn record_clamp_with(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    ahead: bool,
    wait: bool,
) -> bool {
    record_clamp_inner(
        c,
        l,
        dst,
        budget,
        src,
        count,
        spec,
        ahead,
        wait,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    )
}

#[allow(clippy::too_many_arguments)]
fn record_clamp_inner(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    ahead: bool,
    wait: bool,
    final_state: D3D12_RESOURCE_STATES,
) -> bool {
    let Some(bytes) = validate_layout(
        (kind, stride, records),
        src_offset,
        count.map(|(_, off)| off),
    ) else {
        return false;
    };
    // SAFETY: live D3D12 buffer resources.
    let (src_base, dst_gpu) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    if !src_base.is_multiple_of(4) || !dst_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
        return false;
    }
    let (src_gpu, args_offset) = match src_offset
        .checked_add(bytes)
        .filter(|&end| end <= u64::from(u32::MAX))
    {
        Some(_) => (src_base, src_offset as u32),
        None => {
            let Some(rebased) = src_base.checked_add(src_offset) else {
                return false;
            };
            (rebased, 0)
        }
    };
    let (count_gpu, has_count, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: a live D3D12 buffer resource.
            let base = unsafe { buf.GetGPUVirtualAddress() };
            if !base.is_multiple_of(4) {
                return false;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => {
                    let Some(rebased) = base.checked_add(off) else {
                        return false;
                    };
                    (rebased, 1u32, 0u32)
                }
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    // `kind` bits 8-15: host groups per console group along z; bit 16: mesh rows ([`ROWS`]).
    let (raw_kind, split, rows) = (kind & 0xff, (kind >> 8) & 0xff, (kind >> 16) & 1);
    let flags = has_count | (u32::from(ahead) << 1) | (split << 8) | (rows << 16);
    let constants = [
        raw_kind,
        stride,
        records,
        args_offset,
        flags,
        count_offset,
        0,
    ];
    let first = [
        uav_barrier(),
        transition(
            dst,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ),
    ];
    let skip = usize::from(!wait);
    // SAFETY: an open list; our root signature and pipeline; live source,
    // count, scratch destination, and budget buffers.
    unsafe {
        l.ResourceBarrier(&first[skip..]);
        if records > 0 {
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, budget);
            // One dispatch per 64 records, each after the last, so chunks
            // charge the budget deterministically in lane order.
            let mut offset: u32 = 0;
            while offset < records {
                let mut chunk = constants;
                chunk[6] = offset;
                l.SetComputeRoot32BitConstants(0, 7, chunk.as_ptr().cast(), 0);
                l.Dispatch(1, 1, 1);
                offset += 64;
                if offset < records {
                    l.ResourceBarrier(&[uav_barrier()]);
                }
            }
        }
        if final_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
            l.ResourceBarrier(&[uav_barrier()]);
        } else {
            let second = [
                uav_barrier(),
                transition(dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, final_state),
            ];
            let skip_after = usize::from(!every_wait());
            l.ResourceBarrier(&second[skip_after..]);
        }
    }
    true
}

/// Tracks which argument buffers may still be gathered into a list's prelude.
///
/// A command that writes to a known host copy (`src_id`) blocks future gathers
/// from that buffer alone, leaving unmodified argument buffers gatherable; an
/// unbounded or unknown write closes gathering globally for the rest of the
/// list. Records gathered before a write remain valid because the prelude runs
/// before the list.
#[derive(Default)]
pub(crate) struct GatherGate {
    closed: bool,
    written: Vec<u64>,
}

impl GatherGate {
    /// Maximum distinct per-buffer poisons tracked before promoting to a global close.
    const MAX_WRITTEN: usize = 64;

    /// Closes gathering globally for the rest of the list.
    pub(crate) fn close_all(&mut self) {
        self.closed = true;
        self.written.clear();
    }

    /// Blocks future gathering from host copy `src_id`.
    pub(crate) fn close_buffer(&mut self, src_id: u64) {
        if self.closed || self.written.contains(&src_id) {
            return;
        }
        if self.written.len() >= Self::MAX_WRITTEN {
            self.close_all();
        } else {
            self.written.push(src_id);
        }
    }

    /// Whether a draw at the current point in the list may gather from `src_id`.
    pub(crate) fn allows(&self, src_id: u64) -> bool {
        !self.closed && !self.written.contains(&src_id)
    }

    /// Re-opens gathering when the list is reset for a new recording.
    pub(crate) fn reset(&mut self) {
        self.closed = false;
        self.written.clear();
    }
}

/// The most records of one argument buffer gathered for a list ([`gather`]):
/// a multiple of the clamp's 64 threads a group, so that threads past the
/// last record read inside the table.
pub(crate) const GATHER_MOST: u32 = 4096;
/// The widest record gathered: its slot in the scratch buffer.
pub(crate) const GATHER_STRIDE: u32 = 32;

/// The records of one argument buffer that a list's draws read, to be
/// clamped in one pass before the list runs ([`gather`]): the buffer (a
/// host copy, and its number), the records' kind (including [`ROWS`] when
/// present) and stride, the scratch buffer whose slot `i` holds record `i`
/// clamped, and the table of the records' offsets in the buffer (upload
/// memory of the list).
pub(crate) struct GatherSet {
    pub(crate) src: ID3D12Resource,
    pub(crate) src_id: u64,
    pub(crate) kind: u32,
    pub(crate) stride: u32,
    pub(crate) scratch: ID3D12Resource,
    table_gpu: u64,
    table: usize,
    /// A record's offset in the buffer, and its slot.
    slots: std::collections::HashMap<u32, u32>,
}

// SAFETY: D3D12 objects are free-threaded; the table is the list's own
// upload memory, written under the list's lock before the list is submitted.
unsafe impl Send for GatherSet {}

impl GatherSet {
    pub(crate) fn new(
        src: ID3D12Resource,
        src_id: u64,
        (kind, stride): (u32, u32),
        scratch: ID3D12Resource,
        (table_gpu, table): (u64, *mut u8),
    ) -> Self {
        Self {
            src,
            src_id,
            kind,
            stride,
            scratch,
            table_gpu,
            table: table as usize,
            slots: Default::default(),
        }
    }

    /// The records gathered so far.
    pub(crate) fn records(&self) -> u32 {
        self.slots.len() as u32
    }

    /// The slot of the record at `offset`: the one it has, or the next
    /// free one. None when the set is full.
    pub(crate) fn slot(&mut self, offset: u32) -> Option<u32> {
        if let Some(&slot) = self.slots.get(&offset) {
            return Some(slot);
        }
        let slot = self.slots.len() as u32;
        if slot >= GATHER_MOST {
            return None;
        }
        // SAFETY: the table holds `GATHER_MOST` words.
        unsafe {
            (self.table as *mut u32)
                .add(slot as usize)
                .write_volatile(offset);
        }
        self.slots.insert(offset, slot);
        Some(slot)
    }

    /// The scratch buffer `src`'s set uses, to keep for the list's next recording.
    pub(crate) fn into_scratch(self) -> ID3D12Resource {
        self.scratch
    }
}

/// Whether [`gather`] can take a record of `kind` and `stride` at `offset`
/// of a buffer at `src_gpu`: what [`record_gather_as`] asks of its
/// arguments, checked when the record is taken, because a record once
/// given a slot is drawn from the slot.
pub(crate) fn gatherable(src_gpu: u64, offset: u64, (kind, stride): (u32, u32)) -> Option<u32> {
    let min = min_stride(kind)?;
    let dst_stride = if kind & ROWS != 0 { ROW_STRIDE } else { stride };
    if stride < min
        || stride > GATHER_STRIDE
        || dst_stride > GATHER_STRIDE
        || !stride.is_multiple_of(4)
        || !offset.is_multiple_of(4)
        || !src_gpu.is_multiple_of(4)
    {
        return None;
    }
    u32::try_from(offset)
        .ok()
        .filter(|o| o.checked_add(stride).is_some())
}

/// For a single record of `kind` and `stride` at `offset` of the host copy
/// `src` (its number `src_id`) that a draw of `list` is about to read: a
/// slot in the list's scratch buffer for that argument buffer, which the
/// list's prelude fills with the record clamped, in one pass with all the
/// other records the list's draws read (`HostList::prelude`,
/// [`record_gathers`]). Nothing is recorded in the list. Returns the scratch
/// buffer and the record's byte offset in it; `None` if the list no longer
/// gathers for `src_id`, the set is full, or the record does not fit.
pub(crate) fn gather(
    list: &HostList,
    (src, src_id): (&ID3D12Resource, u64),
    offset: u64,
    spec: (u32, u32),
) -> Option<(ID3D12Resource, u64)> {
    clamp()?;
    // SAFETY: a live D3D12 buffer resource.
    let Some(offset) = gatherable(unsafe { src.GetGPUVirtualAddress() }, offset, spec) else {
        GATHER_NONE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    };
    list.gather_slot(src, src_id, offset, spec)
}

/// Since the start: records that got no slot because they do not fit the
/// clamp, because their list had stopped gathering for that buffer, because
/// the list gathers from too many buffers, because their set was full, or
/// because a buffer could not be made.
pub(crate) static GATHER_NONE: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Records the clamp of `records` gathered records into `dst`, which is in
/// state `before`, and leaves `dst` in `INDIRECT_ARGUMENT`: record `i` is
/// read at the offset word `i` of the table at `table_gpu`, in the buffer
/// `src`, and stored clamped at `i * dst_stride` (`ROW_STRIDE` when [`ROWS`]
/// is set on `kind`, else `stride`). Charged to the budget word at `budget`
/// in 64-record lane order as records clamped one by one are.
fn record_gather_as(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    src: &ID3D12Resource,
    table_gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> bool {
    // SAFETY: live D3D12 buffer resources.
    let (src_gpu, dst_gpu) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    if gatherable(src_gpu, 0, (kind, stride)).is_none()
        || records == 0
        || records > GATHER_MOST
        || !dst_gpu.is_multiple_of(4)
        || !budget.is_multiple_of(4)
        || !table_gpu.is_multiple_of(4)
    {
        return false;
    }
    let (raw_kind, rows) = (kind & 0xff, (kind >> 16) & 1);
    let flags = 8u32 | (rows << 16);
    let constants = [raw_kind, stride, records, 0u32, flags, 0u32, 0u32];
    let first = [
        uav_barrier(),
        transition(dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    ];
    let second = [
        uav_barrier(),
        transition(
            dst,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ),
    ];
    // SAFETY: an open list; our root signature and pipeline; live source,
    // table, scratch destination and budget buffers.
    unsafe {
        l.ResourceBarrier(&first);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRootShaderResourceView(1, src_gpu);
        l.SetComputeRootShaderResourceView(2, table_gpu);
        l.SetComputeRootUnorderedAccessView(3, dst_gpu);
        l.SetComputeRootUnorderedAccessView(4, budget);
        let mut offset: u32 = 0;
        while offset < records {
            let mut chunk = constants;
            chunk[6] = offset;
            l.SetComputeRoot32BitConstants(0, 7, chunk.as_ptr().cast(), 0);
            l.Dispatch(1, 1, 1);
            offset += 64;
            if offset < records {
                l.ResourceBarrier(&[uav_barrier()]);
            }
        }
        l.ResourceBarrier(&second);
    }
    true
}

/// Records, into a list's prelude `l`, the passes that fill the scratch
/// buffers of `sets` ([`gather`]), charged to the words of `budget`, which
/// is zeroed first. Every source buffer is transitioned `COMMON ->
/// NON_PIXEL_SHADER_RESOURCE -> COMMON` so the main list that follows in the
/// same `ExecuteCommandLists` sees every shadow buffer in `COMMON`.
/// Returns `false` if anything could not be recorded.
pub(crate) fn record_gathers(
    l: &ID3D12GraphicsCommandList,
    sets: &[GatherSet],
    budget: &ID3D12Resource,
) -> bool {
    let (Some(c), Ok(l2)) = (clamp(), l.cast::<ID3D12GraphicsCommandList2>()) else {
        return false;
    };
    let mut srcs: Vec<&ID3D12Resource> = Vec::new();
    for set in sets.iter().filter(|s| s.records() > 0) {
        if !srcs.iter().any(|r| r.as_raw() == set.src.as_raw()) {
            srcs.push(&set.src);
        }
    }
    // SAFETY: a live buffer.
    let at = unsafe { budget.GetGPUVirtualAddress() };
    let zeros = [0u64, 4, 8].map(|offset| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
        Dest: at + offset,
        Value: 0,
    });
    let mut enter: Vec<D3D12_RESOURCE_BARRIER> = Vec::with_capacity(srcs.len() + 1);
    enter.push(transition(
        budget,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    ));
    for src in &srcs {
        enter.push(transition(
            src,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        ));
    }
    // SAFETY: an open list; the budget and source buffers are in COMMON at
    // prelude entry.
    unsafe {
        l.ResourceBarrier(&enter);
        l2.WriteBufferImmediate(zeros.len() as u32, zeros.as_ptr(), None);
        l.ResourceBarrier(&[transition(
            budget,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
    }
    let mut all = true;
    for set in sets.iter().filter(|s| s.records() > 0) {
        all &= record_gather_as(
            c,
            l,
            (&set.scratch, D3D12_RESOURCE_STATE_COMMON),
            at,
            &set.src,
            set.table_gpu,
            (set.kind, set.stride, set.records()),
        );
    }
    let mut exit: Vec<D3D12_RESOURCE_BARRIER> = Vec::with_capacity(srcs.len() + 1);
    exit.push(transition(
        budget,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    ));
    for src in &srcs {
        exit.push(transition(
            src,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        ));
    }
    // SAFETY: as above; restores budget and source buffers to COMMON.
    unsafe {
        l.ResourceBarrier(&exit);
    }
    all
}

/// How clamps of indirect draws are recorded (`VICTORY_HOIST_CLAMPS`, opt-in):
/// - `0` (default): off — each clamp runs inline before its draw (or via
///   `VICTORY_INDIRECT_AMPLIFY` / `VICTORY_INDIRECT_GATHER` / windowed clamping).
/// - `1`: in a list of their own before the list that draws, so that a run of
///   draws is not interleaved with our dispatches.
/// - `2`: as `1`, with a UAV barrier before each draw.
/// - `3`: as `1` for mesh dispatches only; `4`: for draws only; `5`: always a
///   new list for clamps; `6`: lists cut as for `1`, clamps inline.
/// - `7` (also `"on"` / `"true"`): as `1`, with clamps run one after another
///   (a UAV barrier between them) so mesh objects do not flicker when hitting
///   the list's budget.
pub(crate) fn hoist_mode() -> u8 {
    use std::sync::atomic::{AtomicU8, Ordering};
    static MODE: AtomicU8 = AtomicU8::new(u8::MAX);
    let mode = MODE.load(Ordering::Relaxed);
    if mode == u8::MAX {
        let set = parse_hoist_mode(std::env::var_os("VICTORY_HOIST_CLAMPS").as_deref());
        MODE.store(set, Ordering::Relaxed);
        return set;
    }
    mode
}

fn parse_hoist_mode(value: Option<&std::ffi::OsStr>) -> u8 {
    let Some(text) = value.and_then(|v| v.to_str()) else {
        return 0;
    };
    let trimmed = text.trim();
    if trimmed.eq_ignore_ascii_case("on") || trimmed.eq_ignore_ascii_case("true") {
        return 7;
    }
    trimmed.parse::<u8>().ok().filter(|&n| n < 8).unwrap_or(0)
}

/// Cuts made for hoisted clamps, and those skipped for open queries.
pub(crate) static CUTS: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Whether clamps of indirect draws go to a list before the one that draws.
pub(crate) fn hoisting() -> bool {
    hoist_mode() != 0
}

/// As [`bound`], recorded in `clamps`, a list that runs before the one that
/// executes: the buffer and the offset of the clamped records, readable as
/// indirect arguments once `clamps` ended ([`HostList::clamp_end`]).
pub(crate) fn bound_before(
    clamps: &HostList,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
) -> Option<(ID3D12Resource, u64)> {
    let bytes = validate_layout(
        (kind, stride, records),
        src_offset,
        count.map(|(_, off)| off),
    )?;
    if records == 0 {
        return None;
    }
    let (c, budget) = (clamp()?, clamps.budget()?);
    // SAFETY: live buffers.
    let (src_base, count_base) = unsafe {
        (
            src.GetGPUVirtualAddress(),
            count.map(|(b, _)| b.GetGPUVirtualAddress()),
        )
    };
    let fits = |offset: u64, len: u64| {
        offset
            .checked_add(len)
            .is_some_and(|end| end <= u64::from(u32::MAX))
    };
    let (src_gpu, args_offset) = if fits(src_offset, bytes) {
        (src_base, src_offset as u32)
    } else {
        (src_base.checked_add(src_offset)?, 0)
    };
    let (count_gpu, has_count, count_offset) = match (count, count_base) {
        (Some((_, off)), Some(base)) if fits(off, 4) => (base, 1u32, off as u32),
        (Some((_, off)), Some(base)) => (base.checked_add(off)?, 1, 0),
        _ => (src_gpu, 0, 0),
    };
    if !src_gpu.is_multiple_of(4) || !count_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
        return None;
    }
    let dst_bytes = scratch_bytes(kind, stride, records);
    let (dst, at) = clamps.clamp_room(dst_bytes)?;
    // SAFETY: a live buffer.
    let dst_gpu = unsafe { dst.GetGPUVirtualAddress() } + at;
    let (raw_kind, split, rows) = (kind & 0xff, (kind >> 8) & 0xff, (kind >> 16) & 1);
    let flags = has_count | (split << 8) | (rows << 16);
    let constants = [
        raw_kind,
        stride,
        records,
        args_offset,
        flags,
        count_offset,
        0,
    ];
    let l = &clamps.list;
    // SAFETY: an open list; our root signature and pipeline; live source,
    // count, destination (writable: `clamp_room`) and budget buffers.
    unsafe {
        if clamps.clamp_unbound() {
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRootUnorderedAccessView(4, budget);
        }
        l.SetComputeRootShaderResourceView(1, src_gpu);
        l.SetComputeRootShaderResourceView(2, count_gpu);
        l.SetComputeRootUnorderedAccessView(3, dst_gpu);
        let mut offset: u32 = 0;
        while offset < records {
            let mut chunk = constants;
            chunk[6] = offset;
            l.SetComputeRoot32BitConstants(0, 7, chunk.as_ptr().cast(), 0);
            l.Dispatch(1, 1, 1);
            offset += 64;
            if offset < records {
                l.ResourceBarrier(&[uav_barrier()]);
            }
        }
        // Mode 7: one clamp after another, so that the budget is charged in
        // the order of the draws as it is when each clamp precedes its draw.
        if hoist_mode() == 7 {
            l.ResourceBarrier(&[uav_barrier()]);
        }
    }
    Some((dst, at))
}

/// After indirect execution (or if preparation failed after [`bound`]):
/// returns the scratch buffer to the common state for the next call.
pub(crate) fn done(list: &HostList, scratch: &ID3D12Resource) {
    done_on(&list.list, scratch);
}

fn done_on(list: &ID3D12GraphicsCommandList, scratch: &ID3D12Resource) {
    // SAFETY: an open list and a live scratch buffer in INDIRECT_ARGUMENT.
    unsafe {
        list.ResourceBarrier(&[transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )]);
    }
}

fn mesh_pass() -> Option<&'static Clamp> {
    static PASS: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    PASS.get_or_init(|| {
        match host()
            .ok_or_else(windows::core::Error::empty)
            .and_then(|h| pass_for(&h.device, "mesh draw arguments", MESH_SOURCE))
        {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "no mesh draw arguments: indirect mesh dispatches dropped on this host"
                );
                None
            }
        }
    })
    .as_ref()
}

/// Bytes of a draw's and of a dispatch's arguments.
pub(crate) const DRAW_BYTES: u32 = 16;
pub(crate) const DISPATCH_BYTES: u32 = 12;

/// Records, in `list`, the pass that makes the arguments of an indirect
/// mesh dispatch on a host without mesh shaders: from the first clamped
/// record at host address `args` (in unordered-access state, as
/// [`clamp_uav`] leaves it) it writes, at host address `made`, `parts`
/// draws [`DRAW_BYTES`] apart and after them `parts` dispatches
/// [`DISPATCH_BYTES`] apart. Part k covers the record's thread groups from
/// `k * capacity`, at most `capacity` of them: its dispatch runs that many
/// groups and its draw has `vertices` per instance and one instance per
/// group. Replaces the list's compute root signature and pipeline. False if
/// the pass is not available.
pub(crate) fn mesh_draws(
    list: &HostList,
    args: u64,
    made: u64,
    (parts, capacity, vertices): (u32, u32, u32),
) -> bool {
    let Some(c) = mesh_pass() else {
        return false;
    };
    let l = &list.list;
    let constants = [parts, capacity, vertices, 0];
    // SAFETY: an open list; our root signature, pipeline and live buffers
    // the caller's list holds, both in unordered-access state.
    unsafe {
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, args);
        l.SetComputeRootUnorderedAccessView(2, made);
        l.Dispatch(1, 1, 1);
        l.ResourceBarrier(&[uav_barrier()]);
    }
    true
}

pub(crate) fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indirect_budget::BUDGET;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    struct WarpHarness {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        clamp: Clamp,
        worker: Clamp,
    }

    impl WarpHarness {
        fn new() -> Option<Self> {
            // SAFETY: plain DXGI WARP query and D3D12 test harness setup.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                    return None;
                }
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                let clamp = make_for(&device).expect("clamp shader must compile on WARP");
                let worker = compile_cs(&device, WORKER_CS).expect("worker shader must compile");
                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    clamp,
                    worker,
                })
            }
        }

        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and persistent CPU mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        fn default_uav(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(64 << 10),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain device-local buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: valid single-argument command signature description.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting on fence, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let bytes = crate::hlsl::compile_compute("indirect test worker", source)?;
        // SAFETY: valid compute shader blob with embedded root signature.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Clamp { root, pso })
        }
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn rejects_unaligned_or_undersized_layouts() {
        assert_eq!(validate_layout((0, 16, 4), 0, None), Some(64));
        assert_eq!(validate_layout((1, 20, 4), 64, Some(128)), Some(80));
        assert_eq!(validate_layout((2, 12, 0), 0, Some(0)), Some(0));
        assert_eq!(validate_layout((10, 16, 2), 8, None), Some(32));
        assert_eq!(validate_layout((0, 12, 1), 0, None), None);
        assert_eq!(validate_layout((1, 16, 1), 0, None), None);
        assert_eq!(validate_layout((2, 8, 1), 0, None), None);
        assert_eq!(validate_layout((10, 8, 1), 0, None), None);
        assert_eq!(validate_layout((3, 16, 1), 0, None), None);
        assert_eq!(validate_layout((0, 18, 1), 0, None), None);
        assert_eq!(validate_layout((0, 16, 1), 2, None), None);
        assert_eq!(validate_layout((0, 16, 1), 0, Some(6)), None);
        assert_eq!(validate_layout((1, 20, u32::MAX), 0, None), None);
    }

    #[test]
    fn parse_hoist_mode_defaults_off_and_handles_overrides() {
        use std::ffi::OsStr;
        assert_eq!(parse_hoist_mode(None), 0);
        assert_eq!(parse_hoist_mode(Some(OsStr::new("0"))), 0);
        assert_eq!(parse_hoist_mode(Some(OsStr::new("1"))), 1);
        assert_eq!(parse_hoist_mode(Some(OsStr::new("7"))), 7);
        assert_eq!(parse_hoist_mode(Some(OsStr::new("on"))), 7);
        assert_eq!(parse_hoist_mode(Some(OsStr::new("TRUE"))), 7);
        assert_eq!(parse_hoist_mode(Some(OsStr::new("8"))), 0);
        assert_eq!(parse_hoist_mode(Some(OsStr::new("invalid"))), 0);
    }

    #[test]
    fn gather_gate_tracks_per_buffer_and_global_close() {
        let mut gate = GatherGate::default();
        assert!(gate.allows(10));
        assert!(gate.allows(20));

        // Writing buffer A blocks future gathers from A while allowing unmodified buffer B.
        gate.close_buffer(10);
        assert!(!gate.allows(10));
        assert!(gate.allows(20));
        assert!(gate.allows(30));

        // Duplicate close_buffer is idempotent and does not consume extra slots.
        gate.close_buffer(10);
        assert!(!gate.allows(10));
        assert!(gate.allows(20));

        // Unbounded/unknown write closes gathering globally for every buffer.
        gate.close_all();
        assert!(!gate.allows(10));
        assert!(!gate.allows(20));
        assert!(!gate.allows(999));

        // Reset re-opens gathering for the next recording.
        gate.reset();
        assert!(gate.allows(10));
        assert!(gate.allows(20));

        // Overflowing MAX_WRITTEN distinct buffers promotes cleanly to global close.
        for id in 0..GatherGate::MAX_WRITTEN as u64 {
            gate.close_buffer(id);
            assert!(!gate.allows(id));
        }
        assert!(gate.allows(10_000));
        gate.close_buffer(10_000);
        assert!(!gate.allows(10_000));
        assert!(!gate.allows(20_000));
    }

    #[test]
    fn host_list_gather_slot_respects_per_buffer_and_global_close() {
        let Some(h) = host() else {
            return;
        };
        let Ok(list) = h.new_list() else {
            return;
        };
        let Some(buf_a) = h.default_buffer(4096) else {
            return;
        };
        let Some(buf_b) = h.default_buffer(4096) else {
            return;
        };
        const KIND_DRAW: u32 = 0;
        const STRIDE: u32 = 16;

        // Before any write: both buffer A (id 1) and buffer B (id 2) gather cleanly.
        assert!(gather(&list, (&buf_a, 1), 0, (KIND_DRAW, STRIDE)).is_some());
        assert!(gather(&list, (&buf_b, 2), 0, (KIND_DRAW, STRIDE)).is_some());

        // Write to buffer A: blocks A, while B continues to gather new slots.
        list.gather_close_buffer(1);
        assert!(gather(&list, (&buf_a, 1), 32, (KIND_DRAW, STRIDE)).is_none());
        assert!(gather(&list, (&buf_b, 2), 32, (KIND_DRAW, STRIDE)).is_some());

        // Unbounded write: closes globally for B as well.
        list.gather_close();
        assert!(gather(&list, (&buf_b, 2), 64, (KIND_DRAW, STRIDE)).is_none());

        list.close();
        h.execute(&[&list]);
    }

    #[test]
    fn gathered_records_must_fit_the_clamp() {
        assert_eq!(gatherable(0x1000, 0x6d71550, (10, 12)), Some(0x6d71550));
        assert_eq!(
            gatherable(0x1000, 0x6d71550, (10 | ROWS, 12)),
            Some(0x6d71550)
        );
        assert_eq!(gatherable(0x1000, 0, (1, 20)), Some(0));
        assert_eq!(gatherable(0x1000, 0x6d71552, (10, 12)), None);
        assert_eq!(gatherable(0x1002, 0x40, (10, 12)), None);
        assert_eq!(gatherable(0x1000, 0x40, (10, 8)), None);
        assert_eq!(gatherable(0x1000, 0x40, (10, GATHER_STRIDE + 4)), None);
        assert_eq!(gatherable(0x1000, 0x40, (7, 12)), None);
        assert_eq!(gatherable(0x1000, 1 << 32, (10, 12)), None);
        assert_eq!(gatherable(0x1000, u64::from(u32::MAX) - 3, (10, 12)), None);
    }

    #[test]
    fn warp_count_buffer_ignores_garbage_tail_and_preserves_source() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 24;
        const MAX: u32 = 4096;
        const ARGS_OFFSET: usize = 64;
        const COUNT_OFFSET_1: usize = 128;
        const COUNT_OFFSET_2: usize = 192;

        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(u64::from(STRIDE * 8));
        let dst_buf = warp.default_uav(u64::from(STRIDE) * u64::from(MAX));
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        let mut initial_src = vec![0xA5u8; src_bytes];
        let write_record = |buf: &mut [u8], idx: usize, words: [u32; 6]| {
            let base = ARGS_OFFSET + idx * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                buf[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        };
        write_record(
            &mut initial_src,
            0,
            [3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666],
        );
        write_record(
            &mut initial_src,
            1,
            [5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC],
        );
        write_record(
            &mut initial_src,
            2,
            [100_000, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567],
        );
        for slot in 3..MAX as usize {
            write_record(
                &mut initial_src,
                slot,
                [0xFFFF_FFFF, 16, 1, 0xDEAD_BEEF, 0xBAD0_C0DE, slot as u32],
            );
        }

        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8) as usize);
        }

        assert!(record_clamp_with(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_1 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            false,
            true,
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_1 as u64,
            );
        }
        done_on(&warp.list, &dst_buf);

        // SAFETY: open command list; `dst_buf` returned to `COMMON` by `done_on`.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, u64::from(STRIDE * 5));
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }

        assert!(record_clamp_with(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_2 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            false,
            true,
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_2 as u64,
            );
        }
        done_on(&warp.list, &dst_buf);
        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 5,
            )
            .to_vec();
            (hits, spent, after_src, after_count, words)
        };

        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(hits, expected_groups);
        assert_eq!(spent, expected_groups * 64);
        assert_eq!(after_src, initial_src);
        assert_eq!(after_count, initial_count);
        assert_eq!(
            &clamped_words[0..6],
            &[3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666]
        );
        assert_eq!(
            &clamped_words[6..12],
            &[5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC]
        );
        assert_eq!(
            &clamped_words[12..18],
            &[65_535, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567]
        );
        assert_ne!(clamped_words[21], 0xDEAD_BEEF);
    }

    #[test]
    fn warp_window_leaves_unfilled_slots_empty_and_uncharged() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 32;
        const RECORDS: u32 = 16;
        let filled = [5u32, 0, 519];
        let total = (STRIDE * RECORDS) as usize;
        let mut src = vec![0xffu8; total];
        for (i, groups) in filled.into_iter().enumerate() {
            let words = [groups, 1, 1, 0xAB00 + i as u32];
            for (w, word) in words.into_iter().enumerate() {
                let at = i * STRIDE as usize + w * 4;
                src[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        for ahead in [true, false] {
            let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
            let dst_buf = warp.default_uav(total as u64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, total);
            }
            assert!(record_clamp_with(
                &warp.clamp,
                &warp.list,
                &dst_buf,
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_MESH, STRIDE, RECORDS),
                ahead,
                true,
            ));
            done_on(&warp.list, &dst_buf);
            // SAFETY: open list; copy clamped scratch buffer for inspection.
            unsafe {
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                warp.list
                    .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            warp.submit_and_wait();
            // SAFETY: GPU work completed.
            let (spent, largest, out) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).add(1).read_volatile(),
                    std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4)
                        .to_vec(),
                )
            };
            for (i, groups) in filled.into_iter().enumerate() {
                let rec = &out[i * 8..i * 8 + 4];
                assert_eq!(rec, [groups, 1, 1, 0xAB00 + i as u32]);
            }
            let charged_filled = (5 + 519) * 128;
            if ahead {
                for i in filled.len()..RECORDS as usize {
                    assert_eq!(out[i * 8..i * 8 + 3], [0, 0, 0]);
                    assert_eq!(out[i * 8 + 3], u32::MAX);
                }
                assert_eq!(spent, charged_filled);
                assert_eq!(largest, 519);
            } else {
                assert_eq!(spent, charged_filled + 13 * 65535 * 128);
                assert_eq!(largest, u32::MAX);
            }
        }
    }

    #[test]
    fn warp_budget_saturates_without_wraparound_and_preserves_draw_fields() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 24;
        const RECORDS: u32 = 128;

        let total_bytes = (STRIDE * RECORDS) as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            let words: [u32; 6] = [
                u32::MAX,
                u32::MAX,
                1000 + i as u32,
                (-200i32 + i as i32) as u32,
                5000 + i as u32,
                0xCAFE_0000 | (i as u32),
            ];
            let base = i * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            (budget_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
        }

        assert!(record_clamp_with(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_INDEXED, STRIDE, RECORDS),
            false,
            true,
        ));
        done_on(&warp.list, &dst_buf);

        // SAFETY: open command list; copy clamped scratch buffer for verification.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                total_bytes / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(spent, BUDGET);
        assert_eq!(after_src, initial_src);

        let fit = (BUDGET >> 26) as usize;
        // Lane-order chunking guarantees records 0..fit are kept and fit..RECORDS are zeroed.
        for i in 0..RECORDS as usize {
            let rec = &out_words[i * 6..(i + 1) * 6];
            if i < fit {
                assert_eq!((rec[0], rec[1]), (1 << 24, 4), "record {i} must be kept");
            } else {
                assert_eq!((rec[0], rec[1]), (0, 0), "record {i} must be zeroed");
            }
            assert_eq!(rec[2], 1000 + i as u32);
            assert_eq!(rec[3], (-200i32 + i as i32) as u32);
            assert_eq!(rec[4], 5000 + i as u32);
            assert_eq!(rec[5], 0xCAFE_0000 | (i as u32));
        }
    }

    #[test]
    fn warp_gathered_records_come_out_as_if_clamped_one_by_one() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 12;
        // 70 records (>64 to exercise the 64-record `chunk[6] = offset` loop).
        let mut records: Vec<(u32, [u32; 3])> = vec![
            (0x3350, [5, 1, 1]),
            (0x0040, [519, 1, 1]),
            (0x0050, [0, 1, 1]),
            (0x7ff0, [70000, 1, 1]),
            (0x1230, [u32::MAX, u32::MAX, u32::MAX]),
            (0x0010, [9, 7, 3]),
            (0x2000, [1, 0, 0]),
        ];
        for i in 0..63u32 {
            records.push((0x4000 + i * 16, [i + 1, 1, 1]));
        }
        let total = 0x8000usize;
        let mut src = vec![0x5Au8; total];
        for &(at, words) in &records {
            for (w, word) in words.into_iter().enumerate() {
                let i = at as usize + w * 4;
                src[i..i + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        let slots = u64::from(GATHER_MOST * STRIDE);
        let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
        let (_table_buf, table_ptr, table_gpu) = warp.shared_buffer(u64::from(GATHER_MOST) * 4);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(slots);
        let dst_buf = warp.default_uav(slots);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
            std::ptr::write_bytes(table_ptr, 0, GATHER_MOST as usize * 4);
            for (i, &(at, _)) in records.iter().enumerate() {
                (table_ptr as *mut u32).add(i).write_volatile(at);
            }
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(readback_ptr, 0xEE, slots as usize);
        }
        assert!(record_gather_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            &src_buf,
            table_gpu,
            (KIND_MESH, STRIDE, records.len() as u32),
        ));
        // SAFETY: an open list; copy the clamped scratch buffer for inspection.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, slots);
        }
        warp.submit_and_wait();
        // SAFETY: GPU work completed.
        let (spent, out) = unsafe {
            (
                (budget_ptr as *const u32).read_volatile(),
                std::slice::from_raw_parts(
                    readback_ptr.cast::<u32>().cast_const(),
                    records.len() * 3,
                )
                .to_vec(),
            )
        };
        let want: Vec<[u32; 3]> = records
            .iter()
            .map(|(_, w)| [w[0].min(65535), w[1].min(1), w[2].min(1)])
            .collect();
        for (i, want) in want.iter().enumerate() {
            assert_eq!(
                &out[i * 3..i * 3 + 3],
                want,
                "slot {i} (the record at {:#x})",
                records[i].0
            );
        }
        let charged: u32 = want.iter().map(|w| w[0] * w[1] * w[2] * 128).sum();
        assert_eq!(spent, charged);
        // SAFETY: `src_ptr` is mapped for `total` bytes and the GPU is idle.
        let src_after = unsafe { std::slice::from_raw_parts(src_ptr.cast_const(), total) };
        assert_eq!(src_after, &src[..]);
    }

    /// The arguments of an indirect mesh dispatch without mesh shaders: the
    /// first record's thread groups are split over the parts, each with its
    /// draw and its dispatch, and the title's records stay as they are.
    #[test]
    fn mesh_draw_arguments_split_the_first_record() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        // (first record, what each of 3 parts of 1000 groups runs).
        let cases: [([u32; 3], [u32; 3]); 4] = [
            ([2500, 1, 1], [1000, 1000, 500]),
            ([1000, 1, 1], [1000, 0, 0]),
            ([5000, 1, 1], [1000, 1000, 1000]),
            ([700, 0, 1], [0, 0, 0]),
        ];
        for (first, want) in cases {
            let list = host.new_list().expect("host command list");
            let (args, args_at) = host.shared_buffer(256).expect("records");
            let (made, made_at) = host.shared_buffer(256).expect("made arguments");
            let records: [[u32; 3]; 3] = [first, [7, 1, 1], [9, 1, 1]];
            // SAFETY: freshly mapped host buffers of 256 bytes; live buffers.
            let (args_gpu, made_gpu) = unsafe {
                std::ptr::copy_nonoverlapping(
                    records.as_ptr().cast::<u8>(),
                    args_at,
                    size_of_val(&records),
                );
                std::ptr::write_bytes(made_at, 0xaa, 256);
                (args.GetGPUVirtualAddress(), made.GetGPUVirtualAddress())
            };
            // Three parts of 1000 groups, 768 vertices each.
            assert!(super::mesh_draws(&list, args_gpu, made_gpu, (3, 1000, 768)));
            list.close();
            host.execute(&[&list]);
            // SAFETY: execution finished; the buffers are mapped and readable.
            let (after, draws, dispatches) = unsafe {
                (
                    args_at.cast::<[[u32; 3]; 3]>().read_unaligned(),
                    made_at.cast::<[[u32; 4]; 3]>().read_unaligned(),
                    made_at.add(48).cast::<[[u32; 3]; 3]>().read_unaligned(),
                )
            };
            assert_eq!(after, records, "{first:?}: the title's records");
            assert_eq!(draws, want.map(|n| [768, n, 0, 0]), "{first:?}: draws");
            let empty_or = |n: u32| if n == 0 { [0; 3] } else { [n, 1, 1] };
            assert_eq!(dispatches, want.map(empty_or), "{first:?}: dispatches");
        }
    }
}
