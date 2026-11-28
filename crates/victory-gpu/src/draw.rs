//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_UNKNOWN,
};
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::{HostPipeline, MeshFallbackPipeline, Stage};

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    pub(crate) address: u64,
    pub(crate) size: u32,
    pub(crate) stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

const SHADER_RESOURCE: u32 = 0x40 | 0x80;
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;

#[derive(Clone, Copy)]
struct DepthTarget {
    resource: usize,
    state: u32,
}

/// A table entry whose raw buffer view the host binds as a structured view
/// of `stride`-byte elements: `entry` descriptors into root table `slot`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
struct StructuredBinding {
    slot: u8,
    entry: u32,
    stride: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum RawAccess {
    Read,
    ReadWrite,
}

/// The records of an indirect mesh dispatch of up to `max` that a host
/// without mesh shaders runs, and so clamps and charges to the list's
/// budget: the first alone (`dispatch_mesh_without_mesh_shaders`'s
/// one-record parts), none of `max == 0`.
// shortcut: as the fallback runs only the first record, a dispatch whose
// count asks for more draws only its first.
fn mesh_fallback_records(max: u32) -> u32 {
    max.min(1)
}

/// A raw UAV exposes whole dwords, and D3D12 stores the element count in a u32.
fn writable_root_bytes(remaining: u64) -> Option<u64> {
    u32::try_from(remaining / 4).ok().map(|n| u64::from(n) * 4)
}

/// A raw view describes bytes of a particular host resource. The descriptor
/// slot keeps that resource alive for the recorded commands.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RawBufferRange {
    resource: usize,
    offset: u64,
    len: u64,
}

/// What a render target or depth view of the host is made from: the host
/// texture (`0`: typed null view, `1`: fallback null target), and
/// [`crate::descriptor::RtvRecord::key`].
type TargetKey = (usize, (usize, [u32; 7]));

/// Render targets and depth currently bound on the host list
/// ([`State::host_targets`]), stored inline for the standard `<= 8` RTVs so
/// cache hits allocate nothing on the heap.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundTargets {
    rtvs: [TargetKey; 8],
    len: u8,
    extra: Vec<TargetKey>,
    dsv: Option<TargetKey>,
}

impl BoundTargets {
    fn new() -> Self {
        Self {
            rtvs: [(0, (0, [0; 7])); 8],
            len: 0,
            extra: Vec::new(),
            dsv: None,
        }
    }

    fn push(&mut self, key: TargetKey) {
        if self.extra.is_empty() && usize::from(self.len) < self.rtvs.len() {
            self.rtvs[usize::from(self.len)] = key;
            self.len += 1;
        } else {
            self.extra.push(key);
        }
    }
}

/// Whether [`State::set_targets`] skips redundant `OMSetRenderTargets` and
/// host RTV/DSV descriptor writes when the resolved targets match what is
/// already bound on the host list (`VICTORY_BIND_CACHE=0` binds every time).
fn bind_cache() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BIND_CACHE").is_none_or(|v| v != "0"))
}

/// Snapshot of a resolved render-target view captured at `OMSetRenderTargets`
/// bind time, so a mid-list wait split reapplies the original view even if the
/// title later overwrites the descriptor slot in place.
struct BoundRtv {
    host: Option<ID3D12Resource>,
    desc: Option<D3D12_RENDER_TARGET_VIEW_DESC>,
}

/// Snapshot of a resolved depth-stencil view captured at `OMSetRenderTargets`
/// bind time.
struct BoundDsv {
    host: ID3D12Resource,
    desc: Option<D3D12_DEPTH_STENCIL_VIEW_DESC>,
}

/// Root slots (bytes) as a set that lives on the stack: a draw collects them
/// without the heap, and they come out in order.
#[derive(Default)]
struct Slots([u64; 4]);

impl Slots {
    fn insert(&mut self, slot: u8) {
        self.0[usize::from(slot / 64)] |= 1 << (slot % 64);
    }

    fn contains(&self, slot: u8) -> bool {
        (self.0[usize::from(slot / 64)] >> (slot % 64)) & 1 != 0
    }

    fn iter(&self) -> impl Iterator<Item = u8> {
        let words = self.0;
        (0..4u8).flat_map(move |w| {
            let mut bits = words[usize::from(w)];
            std::iter::from_fn(move || {
                (bits != 0).then(|| {
                    let bit = bits.trailing_zeros() as u8;
                    bits &= bits - 1;
                    w * 64 + bit
                })
            })
        })
    }
}

/// A hash that is mixed already, used as it is.
#[derive(Default)]
struct Plain(u64);

impl std::hash::Hasher for Plain {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0 << 8 | u64::from(b);
        }
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = x;
    }
}

#[inline]
fn mix_hash(h: u64, x: u64) -> u64 {
    (h.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95)
}

/// A quick hash of a descriptor table's heap, copy generation, slot records,
/// liveness flags, and structured alias/stride overrides. Hits still verify
/// every exact `[u32; 3]` record word and override, so a collision costs a
/// copy, never wrong descriptors.
#[allow(clippy::too_many_arguments)]
fn table_hash(
    heap: usize,
    copy_gen: u64,
    raws: &[[u32; 3]],
    dead: &[bool],
    start_slot: u32,
    step_dwords: u32,
    aliases: &[StructuredBinding],
    strides: &[Option<(victory_shader::emit::StructuredBinding, u32)>],
) -> u64 {
    let mut h = mix_hash(heap as u64, copy_gen);
    h = mix_hash(h, raws.len() as u64);
    for (&[a, b, c], &is_dead) in raws.iter().zip(dead) {
        h = mix_hash(h, u64::from(a) | (u64::from(b) << 32));
        h = mix_hash(h, u64::from(c) | (u64::from(is_dead) << 32));
    }
    for alias in aliases {
        h = mix_hash(
            h,
            u64::from(alias.entry - start_slot) | (u64::from(alias.stride) << 32),
        );
    }
    for (binding, stride) in strides.iter().flatten() {
        let rel = binding.offset_dwords / step_dwords - start_slot;
        h = mix_hash(h, u64::from(rel) | (u64::from(binding.uav) << 32));
        h = mix_hash(h, u64::from(*stride));
    }
    h ^ (h >> 32)
}

/// The words of entry `k` of `table` (32-byte view slots or 16-byte sampler
/// slots, stepped by `step_bytes`).
#[cfg(test)]
fn record(table: u64, k: u32) -> [u32; 3] {
    // SAFETY: a descriptor slot in title memory.
    unsafe { ((table + u64::from(k) * 32) as *const [u32; 3]).read_unaligned() }
}

/// What a shader adds to a heap slot it found through a root table to get
/// the title's slot (`victory_shader::emit::USER_TABLE_SLOTS`): the title's
/// table is `title` bytes into its heap, and the shader was given `fresh`
/// instead when the list bound its own copy of the table
/// ([`State::tables_with_strides`]), whose slots the buffer view table has
/// no records for. Wrapping, as the shader's sum is.
fn table_slots_ahead(title: u32, fresh: Option<u32>) -> u32 {
    fresh.map_or(0, |fresh| (title / 32).wrapping_sub(fresh / 32))
}

struct FreshTableEntry {
    heap: usize,
    copy_gen: u64,
    slots: u32,
    records_at: usize,
    first: u32,
    /// Sorted offsets `k - start_slot` that held a tagged stale (released) view
    /// compiled as null, so unchanged stale slots reuse the rebuilt range.
    dead_slots: Vec<u32>,
    /// Relative `(entry - start_slot, stride)` overrides compiled into `first`.
    aliases: Vec<(u32, u32)>,
    /// Relative `(slot - start_slot, uav, stride)` overrides compiled into `first`.
    strides: Vec<(u32, bool, u32)>,
}

/// Fast-path and incremental-sync cache for the texture half of
/// [`State::bindless_textures`]. Buffer refreshes always run when a stage
/// indexes `ResourceHeap` and are not gated by this cache.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BindlessCache {
    heap: usize,
    tex_gen: u64,
    state_gen: u64,
    targets_hash: u64,
    uses_heap_srv: bool,
    uses_heap_uav: bool,
    /// Cursor into `HostList::dirty_resources` after the last bindless sync.
    dirty_cursor: usize,
    /// Active graphics RTs / depth target skipped on the last bindless sync so
    /// they are re-checked once unbound even without another intervening copy.
    skipped_targets: Vec<usize>,
    /// Whether the cached bindless pass also sampled a read-only depth target.
    depth_sampled: bool,
}

/// A window of consecutive single-record indirect draws clamped together into
/// the list's scratch buffer (`windows`).
struct Window {
    copy: u64,
    kind: u32,
    stride: u32,
    step: u32,
    base: u64,
    records: u32,
    scratch: ID3D12Resource,
}

/// The most records clamped in one pass ([`windows`]).
const WINDOW_MOST: u64 = 64;

/// Whether indirect draws that read consecutive records from one argument
/// buffer are clamped a window at a time (`VICTORY_INDIRECT_WINDOW`, on by
/// default; `=0` disables and clamps each record individually).
fn windows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_WINDOW").is_none_or(|v| v != "0"))
}

/// Parses `VICTORY_INDIRECT_GATHER`: `1` (enabled) by default; `"0"`, `"false"`,
/// `"no"`, or `"off"` disables (`0`); `"alt"` alternates twenty seconds on/off (`2`).
fn parse_gather_mode(value: Option<&std::ffi::OsStr>) -> u8 {
    let Some(text) = value.and_then(|v| v.to_str()) else {
        return 1;
    };
    let trimmed = text.trim();
    if trimmed == "0"
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no")
        || trimmed.eq_ignore_ascii_case("off")
    {
        0
    } else if trimmed.eq_ignore_ascii_case("alt") {
        2
    } else {
        1
    }
}

/// Whether a single record that misses its window is gathered: given a slot
/// that the list's prelude fills in one pass with the list's other such
/// records, in place of a clamp pass in front of its draw
/// (`crate::indirect::gather`; on by default, `VICTORY_INDIRECT_GATHER=0`
/// disables; `alt`: twenty seconds on, twenty off, logged on each flip).
/// Needs [`windows`].
fn gathers() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE
        .get_or_init(|| parse_gather_mode(std::env::var_os("VICTORY_INDIRECT_GATHER").as_deref()))
    {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH
                .get_or_init(std::time::Instant::now)
                .elapsed()
                .as_secs()
                / 20)
                % 2
                == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "single indirect records are gathered for their list's prelude from now (VICTORY_INDIRECT_GATHER=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// Records served from gathered slots vs asked and got none (`VICTORY_INDIRECT_GATHER`).
static GATHER_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Where the clamps of a run of indirect draws go: the list `clamps`, open,
/// that runs before the host list at `into`, as long as that one has `writes`
/// dispatches and copies (a later one may write the arguments).
pub(crate) struct Hoist {
    pub(crate) clamps: std::sync::Arc<HostList>,
    pub(crate) into: usize,
    pub(crate) writes: u64,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    /// The list before the host list recorded into that holds the clamps of
    /// its indirect draws (`crate::indirect::hoisting`), still open.
    hoist: Option<Hoist>,
    /// The indirect records clamped last, if later draws may still read them
    /// ([`windows`]).
    window: Option<Window>,
    /// A shader bound since the last amplified-draw wait may have written a
    /// buffer (`VICTORY_AMPLIFY_WAIT=auto`).
    wrote: bool,
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: Option<DepthTarget>,
    /// The current draw also samples its read-only depth target.
    depth_sampled: bool,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// Fixed-function state and resolved target snapshots, cached so a
    /// mid-list wait split can reapply them onto a fresh host list without
    /// re-reading mutable descriptor slots.
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<RECT>,
    blend_factor: Option<[f32; 4]>,
    stencil_ref: Option<u32>,
    bound_rtvs: Vec<BoundRtv>,
    bound_dsv: Option<BoundDsv>,
    /// The render target views and depth view currently bound on the host list
    /// ([`State::set_targets`]), cleared on list split or reset.
    host_targets: Option<BoundTargets>,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// [`descriptor::heap_bind_id`] for each of [`Self::host_heaps`] when
    /// last bound; catches a new heap reused at the same title address.
    host_heap_ids: [u64; 2],
    /// Tables made afresh in this list, keyed by quick hash ([`table_hash`])
    /// and verified by exact `[u32; 3]` record words stored in
    /// [`Self::fresh_records`], `heap`, `copy_gen`, liveness, and overrides.
    /// Partitioning by `copy_gen` preserves `Fill -> Copy -> Fill` ordering
    /// for CBV snapshots across intervening deferred buffer copies.
    fresh_tables:
        std::collections::HashMap<u64, FreshTableEntry, std::hash::BuildHasherDefault<Plain>>,
    /// The record words of [`FreshTableEntry`] tables, one after the other.
    fresh_records: Vec<[u32; 3]>,
    /// What [`State::tables_with_strides`] made for the stage being bound:
    /// root slot, byte offset of its fresh table. Kept to reuse its storage
    /// across draws.
    made: Vec<(u8, u32)>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// A fresh table of the stage being bound could not be made.
    table_failed: bool,
    /// One slot per view and access kind in each heap, including null views.
    root_view_slots: std::collections::HashMap<(usize, RawAccess, Option<RawBufferRange>), u32>,
    /// This recording, for [`crate::descriptor::host::HostHeap::make_listed`]
    /// (`0`: none assigned yet).
    recording: u64,
    /// Fast-path and incremental-sync cache for [`State::bindless_textures`].
    bindless_cache: Option<BindlessCache>,
    /// Reusable scratch buffers for single-pass descriptor table hashing and
    /// liveness checks in [`State::tables`].
    table_raws: Vec<[u32; 3]>,
    table_dead: Vec<bool>,
    /// Number of full-heap bindless walks performed on this list (for tests).
    #[cfg(test)]
    bindless_full_walks: u64,
}

impl Drop for State {
    fn drop(&mut self) {
        if self.recording != 0 {
            crate::descriptor::host::clear_list_views(self.recording);
        }
        crate::descriptor::release_dynamic_batch(&self.dynamic);
    }
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Mesh dispatches run as a compute pass and a draw (a host without mesh
/// shaders) since the last call: direct ones, then indirect ones.
pub(crate) fn take_mesh_without_mesh_shaders() -> [u64; 2] {
    [0, 1].map(|i| MESH_WITHOUT_MESH_SHADERS[i].swap(0, Ordering::Relaxed))
}
static MESH_WITHOUT_MESH_SHADERS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// The most thread groups a dispatch has in a dimension, on the host and in
/// a clamped indirect record.
const GROUPS_MOST: u32 = 65535;

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB,
/// other views as 1 MiB (both cut at their buffer's end).
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<u64> {
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow
        .left(address)
        .min(if srv { 1 << 20 } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        return Some(gpu);
    }
    list.fill_scratch(address as *const u8, len)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    /// Whether the clamps of indirect draws recorded in `list` now go to
    /// the list before it.
    pub(crate) fn hoists_into(&self, list: &HostList) -> bool {
        self.hoist.as_ref().is_some_and(|h| {
            h.into == std::ptr::from_ref(list) as usize && h.writes == list.writes()
        })
    }

    /// From here on the clamps of `into`'s indirect draws go to `clamps`.
    pub(crate) fn hoist(&mut self, clamps: std::sync::Arc<HostList>, into: &HostList) {
        let _ = self.end_hoist();
        self.hoist = Some(Hoist {
            clamps,
            into: std::ptr::from_ref(into) as usize,
            writes: into.writes(),
        });
    }

    /// Ends and closes the list of hoisted clamps, returning it so the caller
    /// can schedule it ahead of the draw list.
    pub(crate) fn end_hoist(&mut self) -> Option<std::sync::Arc<HostList>> {
        let h = self.hoist.take()?;
        h.clamps.clamp_end();
        h.clamps.close();
        Some(h.clamps)
    }

    #[cfg(test)]
    pub(crate) fn current_pipeline(&self) -> This {
        self.pipeline as This
    }

    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// Takes the dynamic descriptor slots held by this recording, so a
    /// pending host list can keep them alive until its submission finishes.
    pub(crate) fn take_dynamic(&mut self) -> Vec<(usize, u32, u32)> {
        // A following segment must acquire its own slots before recording a bind.
        self.fresh_tables.clear();
        self.fresh_records.clear();
        self.made.clear();
        self.root_view_slots.clear();
        if self.recording != 0 {
            crate::descriptor::host::clear_list_views(self.recording);
            self.recording = 0;
        }
        std::mem::take(&mut self.dynamic)
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_viewports(&mut self, list: &HostList, items: &[D3D12_VIEWPORT]) {
        self.viewports = items.to_vec();
        viewports(list, &self.viewports);
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.scissors = rects.to_vec();
        scissors(list, &self.scissors);
    }

    pub(crate) fn set_blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.blend_factor = factor.copied();
        blend_factor(list, factor);
    }

    pub(crate) fn set_stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.stencil_ref = Some(reference);
        stencil_ref(list, reference);
    }

    /// Reapplies fixed-function state and targets onto `new_list` after a
    /// mid-list wait split, moving dynamic descriptor slots onto `prev_list`
    /// so they stay alive until its submission finishes.
    pub(crate) fn reapply_host(&mut self, new_list: &HostList, prev_list: &HostList) {
        self.bindless_cache = None;
        self.window = None;
        prev_list.keep_dynamic(self.take_dynamic());
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_heaps = [0; 2];
        self.host_heap_ids = [0; 2];
        if self.topology != 0 {
            self.set_topology(new_list, self.topology);
        }
        if !self.viewports.is_empty() {
            viewports(new_list, &self.viewports);
        }
        if !self.scissors.is_empty() {
            scissors(new_list, &self.scissors);
        }
        if let Some(factor) = self.blend_factor {
            blend_factor(new_list, Some(&factor));
        }
        if let Some(reference) = self.stencil_ref {
            stencil_ref(new_list, reference);
        }
        self.host_targets = None;
        if !self.bound_rtvs.is_empty() || self.bound_dsv.is_some() {
            self.apply_targets(new_list, false);
        }
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self
            .vertex
            .iter_mut()
            .skip(start as usize)
            .take(count as usize)
        {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, resolved at bind time and set on the host
    /// list, unless the host list already has the exact same targets bound
    /// (`VICTORY_BIND_CACHE=0` binds every time).
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        self.depth_sampled = false;
        if crate::host::host().is_none() {
            self.targets.clear();
            self.bound_rtvs.clear();
            self.depth = None;
            self.bound_dsv = None;
            self.host_targets = None;
            return;
        }
        #[derive(Clone, Copy)]
        enum PendingRtv {
            Null(D3D12_RENDER_TARGET_VIEW_DESC),
            Live(&'static ID3D12Resource, crate::descriptor::RtvRecord),
        }
        let mut views: Few<Option<PendingRtv>, 8> = Few::default();
        let mut now = BoundTargets::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            if record.resource.is_null() {
                // A typed null RTV discards writes without changing the output slot.
                if let Some(desc) = record.desc() {
                    views.push(Some(PendingRtv::Null(desc)));
                    now.push((0, record.key()));
                }
                continue;
            }
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            if !crate::host::renders_to(texture) {
                once!(
                    resource = format_args!("{:#x}", record.resource as usize),
                    "render target the host cannot render to; bound as a null view"
                );
                views.push(Some(PendingRtv::Null(crate::host::null_target())));
                now.push((1, (0, [0; 7])));
                continue;
            }
            views.push(Some(PendingRtv::Live(texture, record)));
            now.push((texture.as_raw() as usize, record.key()));
        }
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            Some((texture, record))
        });
        now.dsv = dsv
            .as_ref()
            .map(|(texture, record)| (texture.as_raw() as usize, record.key()));
        if bind_cache() && self.host_targets.as_ref() == Some(&now) {
            return;
        }
        self.host_targets = Some(now);
        self.targets.clear();
        self.bound_rtvs.clear();
        for view in views.iter().flatten().copied() {
            match view {
                PendingRtv::Null(desc) => {
                    self.bound_rtvs.push(BoundRtv {
                        host: None,
                        desc: Some(desc),
                    });
                }
                PendingRtv::Live(texture, record) => {
                    self.targets.push(record.resource as usize);
                    crate::texture::gpu_target(record.resource);
                    #[cfg(feature = "diagnostics")]
                    crate::capture::note_target(record.resource);
                    self.bound_rtvs.push(BoundRtv {
                        host: Some(texture.clone()),
                        desc: record.desc(),
                    });
                }
            }
        }
        self.depth = None;
        self.bound_dsv = None;
        if let Some((texture, record)) = dsv {
            crate::texture::gpu_target(record.resource);
            let desc = record.depth_desc();
            self.depth = Some(DepthTarget {
                resource: record.resource as usize,
                state: depth_state(texture, desc.as_ref()),
            });
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            self.bound_dsv = Some(BoundDsv {
                host: texture.clone(),
                desc,
            });
        }
        self.apply_targets(list, true);
    }

    /// Binds the cached RTV/DSV snapshot onto `list` without re-reading the
    /// title's descriptor slots, keeping each bound host texture alive until
    /// `list` resets or drops. When `mark_work` is false (reapplying state onto
    /// a newly split trailing list after a mid-list wait), the list's COM
    /// references are retained without pinning a retirement epoch until actual
    /// GPU work is recorded.
    fn apply_targets(&self, list: &HostList, mark_work: bool) {
        let Some(host) = crate::host::host() else {
            return;
        };
        if mark_work && (!self.bound_rtvs.is_empty() || self.bound_dsv.is_some()) {
            list.mark_work();
        }
        let mut rtvs: Few<D3D12_CPU_DESCRIPTOR_HANDLE, 8> = Few::default();
        for rtv in &self.bound_rtvs {
            if let Some(tex) = &rtv.host {
                if mark_work {
                    list.keep(tex);
                } else {
                    list.keep_unmarked(tex);
                }
            }
            rtvs.push(host.rtv(rtv.host.as_ref(), rtv.desc.as_ref()));
        }
        let dsv = self.bound_dsv.as_ref().map(|dsv| {
            if mark_work {
                list.keep(&dsv.host);
            } else {
                list.keep_unmarked(&dsv.host);
            }
            host.dsv(&dsv.host, dsv.desc.as_ref())
        });
        // SAFETY: an open list; the views were just made from live host resources.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    fn bound_pipeline(&self, compute: bool) -> Option<std::sync::Arc<HostPipeline>> {
        crate::pipeline::host::bound_pipeline(self.pipeline as This, |binding| {
            let Root::Table(base) = *self.roots_ref(compute).get(usize::from(binding.slot))? else {
                return None;
            };
            let handle = base.checked_add(u64::from(binding.offset_dwords) * 4)?;
            let heap = self.heap_of(base)?;
            if heap.samplers() {
                return None;
            }
            heap.index(handle)?;
            let view = crate::descriptor::view(handle)?;
            crate::descriptor::host::structured_stride(&view, binding.uav)
        })
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = self.bound_pipeline(false) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(
                mesh,
                "a draw or mesh dispatch with another kind of pipeline"
            );
            return false;
        }
        if p.rects && self.topology != RECTANGLES {
            once!("a rectangle pipeline drawn with another topology");
            return false;
        }
        self.prepare_with(list, &p, indexed)
    }

    /// What [`State::prepare_for`] does with the graphics pipeline `p` it
    /// found: its bindings, the vertex and index buffers, the targets.
    fn prepare_with(&mut self, list: &HostList, p: &HostPipeline, indexed: bool) -> bool {
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if let Some(depth) = self.depth {
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(depth.resource as This, false, true);
        }
        self.ensure_depth_state(list);
        true
    }

    fn ensure_depth_state(&self, list: &HostList) {
        if let Some(depth) = self.depth {
            let state = depth.state
                | if self.depth_sampled {
                    SHADER_RESOURCE
                } else {
                    0
                };
            crate::list::ensure_state(list, depth.resource as This, state);
        }
    }

    fn read_only_depth(&self, compute: bool) -> Option<usize> {
        self.depth
            .filter(|depth| !compute && depth.state == DEPTH_READ)
            .map(|depth| depth.resource)
    }

    /// A user-data dword the host computes for a translated shader
    /// (`victory_shader::emit::USER_MEMORY_BASE`, `USER_VIEW_ADDRESSES`).
    fn host_word(&self, kind: u32, memory: Option<&crate::memory::Ready>) -> u32 {
        if kind == victory_shader::emit::USER_MEMORY_BASE {
            return memory.map_or(0, |m| m.base);
        }
        if kind == victory_shader::emit::USER_VIEW_ADDRESSES {
            // Where the bound view heap's buffer view table lists its
            // views' addresses; 0 (no list) without such a heap.
            let key = self.heaps[0];
            if key == 0 {
                return 0;
            }
            let heap = crate::descriptor::heap(key as This);
            return match heap.host() {
                Some(host) if !heap.samplers() => {
                    crate::descriptor::table::address_list(host.count())
                }
                _ => 0,
            };
        }
        once!(kind, "a shader reads a host value this build does not know");
        0
    }

    /// For a shader that reads title memory by address
    /// ([`crate::memory`]): its directory, fault notes and view slots,
    /// with the pages of the root views the list has set made ready.
    fn memory_ready(
        &self,
        list: &HostList,
        compute: bool,
        writes: bool,
    ) -> Option<crate::memory::Ready> {
        let heap = crate::descriptor::recording_heap(self.host_heaps[0])?;
        let (mut addresses, mut constants) = (Vec::new(), Vec::new());
        for root in self.roots_ref(compute) {
            match root {
                Root::View(address) => addresses.push(*address),
                Root::Constants(values) => constants.extend_from_slice(values),
                _ => {}
            }
        }
        crate::memory::ready(list, heap, &addresses, &constants, writes)
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a table it reads could
    /// not be made.
    fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        list.mark_work();
        // Refresh `list.work_seq` when `bind` returns (including early `false`
        // exits) so resources created lazily during descriptor binding (`tables` ->
        // `host.make_at`, `memory::ready`, root views) cannot get
        // `born_seq > list.work_seq`.
        struct MarkWorkGuard<'a>(&'a HostList);
        impl Drop for MarkWorkGuard<'_> {
            fn drop(&mut self) {
                self.0.mark_work();
            }
        }
        let _mark_after = MarkWorkGuard(list);
        self.depth_sampled = false;
        let l = &list.list;
        // A mesh program run as a compute pass binds its root views from
        // the same heap.
        let needs_root_heap = self.heaps[0] == 0
            && p.stages
                .iter()
                .chain(p.fallback.as_ref().map(|f| &f.stage))
                .any(|stage| {
                    !stage.root_uavs.is_empty()
                        || (!stage.root_srvs.is_empty()
                            && crate::pipeline::host::bounded_root_srvs())
                });
        if !self.bind_heaps(list, needs_root_heap) {
            return false;
        }
        // Mirror any descriptor slots the title copied directly in heap memory
        // before bindless texture tracking or root-table binding reads them.
        for key in self.heaps {
            if key != 0 {
                crate::descriptor::heap(key as This).sync_direct_copies(true);
            }
        }
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            // The title may release the pipeline before it submits the list.
            list.bind_pipeline(p);
        }
        self.bindless_textures(list, p, compute);
        for stage in &p.stages {
            if !self.bind_stage(list, stage, &p.structured_strides, (compute, compute)) {
                return false;
            }
        }
        // Depth textures it writes through unordered access: their colour
        // twins hold what they lack from here on.
        list.twins_written();
        if compute
            || p.stages
                .iter()
                .chain(p.fallback.as_ref().map(|f| &f.stage))
                .any(|stage| stage.translated().stores)
        {
            list.buffers_written();
            self.wrote = true;
        }
        true
    }

    /// The buffers `shader` reads or writes through heap entries it is given
    /// by index in its root constants (`ResourceDescriptorHeap[index]`, its
    /// [`victory_shader::emit::heap_reads`]), and the buffer such an index is
    /// read from: their ranges are brought up to date from title memory
    /// at submission, and those written go back to it after, as for buffers
    /// in tables. Without this a buffer only bindless shaders read keeps
    /// what its host copy held when something else last refreshed it: a
    /// heap index read from it then names another kind of descriptor
    /// (NVIDIA stops with a TEX exception).
    fn bindless_buffers(
        &self,
        list: &HostList,
        shader: &victory_shader::emit::HostShader,
        compute: bool,
    ) {
        let key = self.heaps[0];
        if key == 0 {
            return;
        }
        let reads = victory_shader::emit::heap_reads(&shader.dxil);
        if !reads.iter().any(|r| r.absolute && !r.sampler) {
            return;
        }
        let heap = crate::descriptor::heap(key as This);
        let mut done: Vec<(u32, bool)> = Vec::new();
        let mut refresh = |index: u32, uav: bool| {
            if done.contains(&(index, uav)) {
                return;
            }
            done.push((index, uav));
            // The console reads 27 bits of an index, as `indexed_buffer`
            // does.
            let index = index & (u32::MAX >> 5);
            // An index past the heap (a value the title left unset, say)
            // names no view.
            let Some(at) = heap
                .start
                .checked_add(u64::from(index) << 5)
                .filter(|&at| heap.index(at).is_some())
            else {
                return;
            };
            // The host slot made now, as a table entry's is at a draw:
            // buffer views are otherwise left null until a draw reads
            // them through a table, and a heap index alone never does.
            if let Some(host_heap) = heap.host() {
                #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
                let mirrored = host_heap.mirror(index, at, true);
                host_heap.refresh_slot(index, list);
                #[cfg(feature = "diagnostics")]
                if crate::descriptor::check::enabled() {
                    let kind = host_heap.checked_kind(index, 0);
                    if kind == Some(crate::descriptor::check::Kind::INITIAL_VIEW) {
                        static SEEN: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
                        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                        if !seen.contains(&index) && seen.len() < 64 {
                            seen.push(index);
                            tracing::warn!(
                                index,
                                mirrored = mirrored.is_some(),
                                view = describe(crate::descriptor::view(at).as_ref()),
                                "descriptor check: a heap-indexed slot is still the initial null after mirroring"
                            );
                        }
                    }
                }
            }
            let Some(view) = crate::descriptor::view(at) else {
                return;
            };
            let view = crate::descriptor::host::placed_texture(&view).unwrap_or(view);
            let Some((address, len)) = crate::descriptor::host::buffer_range(&view) else {
                return;
            };
            if len == 0 {
                return;
            }
            let Some(shadow) = crate::shadow::find(address, len) else {
                return;
            };
            if uav {
                shadow.refresh_as(list, address, len, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                shadow.gpu_writes(list, address, len);
            } else {
                shadow.refresh(list, address, len);
            }
        };
        for r in reads.iter().filter(|r| r.absolute && !r.sampler) {
            let uav = r.props.0 & 0x1000 != 0;
            let entry = r.offset_dwords / 8;
            let index = match (r.dynamic, r.index_from, r.index_load) {
                (false, _, _) => Some(entry),
                (true, Some((slot, dword)), _) => Some(
                    self.user_value(compute, slot, RootType::Constants, dword)
                        .wrapping_add(entry),
                ),
                (true, None, Some(l)) => {
                    // The buffer the index is read from first, then the
                    // entry it names now.
                    let at = self.user_value(compute, l.slot, RootType::Constants, l.dword);
                    refresh(at, false);
                    self.index_in_memory(compute, l)
                        .map(|i| i.wrapping_add(entry))
                }
                _ => None,
            };
            if let Some(index) = index {
                refresh(index, uav);
            }
        }
    }

    /// The heap index a shader reads from title memory ([`IndexLoad`]), as
    /// the memory of the title holds it now.
    ///
    /// [`IndexLoad`]: victory_shader::emit::IndexLoad
    fn index_in_memory(&self, compute: bool, l: victory_shader::emit::IndexLoad) -> Option<u32> {
        let key = self.heaps[0];
        if key == 0 {
            return None;
        }
        let heap = crate::descriptor::heap(key as This);

        // The console reads 27 bits of an index.
        let at = self.user_value(compute, l.slot, RootType::Constants, l.dword) & (u32::MAX >> 5);
        let slot = heap.start.checked_add(u64::from(at) << 5)?;
        heap.index(slot)?;
        let view = crate::descriptor::view(slot)?;
        let (address, len) = crate::descriptor::host::buffer_range(&view)?;
        let word = address.checked_add(u64::from(l.byte))?;
        if u64::from(l.byte) + 4 > len || !crate::texture::committed(word as *const u8, 4) {
            return None;
        }
        // SAFETY: four committed bytes of title memory.
        let v = unsafe { (word as *const u32).read_unaligned() };
        Some(v & l.mask)
    }

    /// The table entries `shader` reads as structured buffers where the title
    /// bound a raw buffer view, read from the compute or graphics root
    /// arguments: they get a structured alias, since a view of another kind
    /// than the shader declares is undefined (NVIDIA reads zeros, so a draw
    /// fed from it covers nothing).
    fn raw_structured_aliases(
        &self,
        shader: &victory_shader::emit::HostShader,
        compute: bool,
    ) -> Vec<StructuredBinding> {
        shader
            .structured_srvs
            .iter()
            .filter_map(|s| {
                let Root::Table(base) = *self.roots_ref(compute).get(usize::from(s.slot))? else {
                    return None;
                };
                let handle = base.checked_add(u64::from(s.offset_dwords) * 4)?;
                let heap = self.heap_of(base)?;
                if heap.samplers() {
                    return None;
                }
                heap.index(handle)?;
                let view = crate::descriptor::view(handle)?;
                crate::descriptor::host::raw_aliasable(&view, s.stride).then_some(
                    StructuredBinding {
                        slot: s.slot,
                        // Views are 8 dwords apart.
                        entry: s.offset_dwords / 8,
                        stride: s.stride,
                    },
                )
            })
            .collect()
    }

    /// One stage's user data, root views and the descriptors its tables
    /// hold: read from the title's compute or graphics root arguments
    /// (`from_compute`) and set on the host list's compute or graphics root
    /// signature (`to_compute`). The two differ for a mesh stage run as a
    /// compute shader, whose arguments are the title's graphics ones.
    /// False if the draw or dispatch must not run: a table it reads could
    /// not be made.
    fn bind_stage(
        &mut self,
        list: &HostList,
        stage: &Stage,
        structured_strides: &[(victory_shader::emit::StructuredBinding, u32)],
        (from_compute, to_compute): (bool, bool),
    ) -> bool {
        let l = &list.list;
        let t = stage.translated();
        #[cfg(feature = "diagnostics")]
        let _cbv_trace = crate::cbv_trace::enter(&stage.shader.hash);
        self.table_failed = false;
        self.bindless_buffers(list, &t.host, from_compute);
        let aliases = self.raw_structured_aliases(&t.host, from_compute);
        self.tables_with_strides(list, &t.used, from_compute, &aliases, structured_strides);
        if self.table_failed {
            static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
            let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::warn!(
                    dropped = n,
                    "a draw or dispatch dropped: no dynamic slots for its tables"
                );
            }
            return false;
        }
        #[cfg(feature = "diagnostics")]
        if crate::descriptor::check::enabled()
            && !self.check_descriptors(stage, &t.host, &self.made, from_compute)
        {
            return false;
        }
        // Title memory by address: what the shader finds it by.
        let memory = if t
            .host
            .root_srvs
            .contains(&victory_shader::emit::SLOT_MEMORY)
        {
            let Some(ready) = self.memory_ready(list, from_compute, false) else {
                once!(
                    "a shader that reads title memory by address without a view heap to hold its views: the draw is dropped"
                );
                return false;
            };
            Some(ready)
        } else {
            None
        };
        if let Some((param, dwords)) = stage.user_data {
            let mut stack_values = [0u32; 64];
            let mut heap_values = Vec::new();
            let values: &mut [u32] = if (dwords as usize) <= stack_values.len() {
                &mut stack_values[..dwords as usize]
            } else {
                heap_values.resize(dwords as usize, 0);
                &mut heap_values[..]
            };
            for u in &t.host.user_data {
                if u.root == RootType::Other(victory_shader::emit::USER_TABLE_SLOTS) {
                    let fresh = self.made.iter().find(|m| m.0 == u.slot).map(|m| m.1);
                    let title = self.user_value(from_compute, u.slot, RootType::Table, 0);
                    values[u.index as usize] = table_slots_ahead(title, fresh);
                    continue;
                }
                if let RootType::Other(kind) = u.root {
                    values[u.index as usize] = self.host_word(kind, memory.as_ref());
                    continue;
                }
                if u.indexed_buffer {
                    let selected =
                        self.user_value(from_compute, u.slot, RootType::Constants, u.dword);
                    let Some(binding) = self.indexed_buffer(list, selected, to_compute) else {
                        return false;
                    };
                    values[u.index as usize..][..3].copy_from_slice(&binding);
                    continue;
                }
                if u.root == RootType::ResourceHeap {
                    // The heap slot of the buffer view table.
                    let Some(slot) = self.buffer_table(list, to_compute) else {
                        once!("no buffer view table: the draw or dispatch is dropped");
                        return false;
                    };
                    values[u.index as usize] = slot;
                    continue;
                }
                let fresh = self
                    .made
                    .iter()
                    .find(|m| m.0 == u.slot && u.root == RootType::Table)
                    .copied();
                values[u.index as usize] = match fresh {
                    Some((_, offset)) => offset,
                    None => self.user_value(from_compute, u.slot, u.root, u.dword),
                };
            }
            // SAFETY: an open list; `values` covers the parameter.
            unsafe {
                if to_compute {
                    l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                } else {
                    l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                }
            }
        }
        for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
            for &(slot, param) in views {
                let tabled = srv && crate::pipeline::host::bounded_root_srvs();
                if srv && slot == victory_shader::emit::SLOT_MEMORY {
                    // Ours: the directory of title memory by address.
                    if let Some(m) = &memory {
                        // SAFETY: an open list; the directory and its heap slot live
                        // as long as the process and the list holds both.
                        unsafe {
                            match (to_compute, tabled) {
                                (true, true) => {
                                    l.SetComputeRootDescriptorTable(param, m.directory_table)
                                }
                                (false, true) => {
                                    l.SetGraphicsRootDescriptorTable(param, m.directory_table)
                                }
                                (true, false) => {
                                    l.SetComputeRootShaderResourceView(param, m.directory)
                                }
                                (false, false) => {
                                    l.SetGraphicsRootShaderResourceView(param, m.directory)
                                }
                            }
                        }
                    }
                    continue;
                }
                let Some(&Root::View(address)) =
                    self.roots_ref(from_compute).get(usize::from(slot))
                else {
                    once!(slot, "a shader reads a root view the title did not set");
                    // An unset table is undefined: a null view instead.
                    if tabled
                        && !self.root_buffer_table(list, None, param, to_compute, RawAccess::Read)
                    {
                        return false;
                    }
                    continue;
                };
                if tabled {
                    if !self.root_buffer_table(
                        list,
                        Some(address),
                        param,
                        to_compute,
                        RawAccess::Read,
                    ) {
                        return false;
                    }
                    continue;
                }
                let Some(gpu) = root_view(list, address, srv) else {
                    continue;
                };
                // SAFETY: an open list; the address is in live memory.
                unsafe {
                    match (to_compute, srv) {
                        (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                        (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                        (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                        (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                    }
                }
            }
        }
        for &(slot, param) in &stage.root_uavs {
            if slot == victory_shader::emit::SLOT_FAULTS {
                // Ours: where shaders note the pages they found empty.
                if let Some(m) = &memory {
                    // SAFETY: an open list; the buffer and its heap slot live as long
                    // as the process and the list holds both.
                    unsafe {
                        if to_compute {
                            l.SetComputeRootDescriptorTable(param, m.faults_table);
                        } else {
                            l.SetGraphicsRootDescriptorTable(param, m.faults_table);
                        }
                    }
                }
                continue;
            }
            let address = match self.roots_ref(from_compute).get(usize::from(slot)) {
                Some(&Root::View(address)) => Some(address),
                _ => {
                    once!(slot, "a shader writes a root view the title did not set");
                    None
                }
            };
            if !self.root_buffer_table(list, address, param, to_compute, RawAccess::ReadWrite) {
                return false;
            }
        }
        true
    }

    /// Compares the descriptors at the table entries `shader` reads (its
    /// [`victory_shader::emit::heap_reads`]) with the kinds it declares
    /// there (`VICTORY_CHECK_DESCRIPTORS`). Mismatches are logged; false
    /// (the draw or dispatch is dropped) for one NVIDIA does not survive:
    /// a buffer read as a texture, a texture of another dimension, a slot
    /// past the heap. `made`: the fresh tables of this stage.
    #[cfg(feature = "diagnostics")]
    fn check_descriptors(
        &self,
        stage: &Stage,
        shader: &victory_shader::emit::HostShader,
        made: &[(u8, u32)],
        compute: bool,
    ) -> bool {
        use crate::descriptor::check::{Kind, Mismatch};
        let reads = victory_shader::emit::heap_reads(&shader.dxil);
        if reads.is_empty() {
            return true;
        }
        static CHECKED: AtomicU64 = AtomicU64::new(0);
        CHECKED.fetch_add(1, Ordering::Relaxed);
        {
            static LISTED: std::sync::Mutex<Option<std::collections::HashSet<String>>> =
                std::sync::Mutex::new(None);
            let mut listed = LISTED.lock().unwrap_or_else(|e| e.into_inner());
            if listed
                .get_or_insert_with(Default::default)
                .insert(stage.shader.hash.clone())
                && reads.iter().any(|r| r.dynamic)
            {
                let text: Vec<String> = reads
                    .iter()
                    .map(|r| {
                        format!(
                            "s{}+{}{}{} {:#x},{:#x}",
                            r.slot,
                            r.offset_dwords,
                            if r.dynamic { " dyn" } else { "" },
                            if r.absolute { " abs" } else { "" },
                            r.props.0,
                            r.props.1
                        )
                    })
                    .collect();
                let heap_used: Vec<String> = stage
                    .translated()
                    .used
                    .iter()
                    .filter(|u| u.root != RootType::Table && u.root != RootType::Constants)
                    .map(|u| format!("{:?} {:?} reg {}", u.root, u.kind, u.lower_bound))
                    .collect();
                tracing::info!(hash = %stage.shader.hash, reads = ?text, ?heap_used, "descriptor check: heap reads");
            }
        }
        let read_by = crate::host::submitted_value() + 1;
        let mut ok = true;
        for r in reads.iter() {
            let Some(declared) = Kind::declared(r.props, r.sampler) else {
                continue;
            };
            let (heap, table) = if r.absolute {
                let key = self.heaps[usize::from(r.sampler)];
                if key == 0 {
                    continue;
                }
                let heap = crate::descriptor::heap(key as This);
                (heap, heap.start)
            } else {
                let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(r.slot))
                else {
                    continue;
                };
                let Some(heap) = self.heap_of(table) else {
                    continue;
                };
                (heap, table)
            };
            if heap.samplers() != r.sampler {
                continue;
            }
            let Some(host) = heap.host() else {
                continue;
            };
            let fresh = if r.absolute {
                None
            } else {
                made.iter().find(|m| m.0 == r.slot).map(|m| m.1)
            };
            let mut base = fresh.unwrap_or((table - heap.start) as u32);
            // A heap index passed in a root constant: the entry is known.
            let known = match r.index_from {
                Some((slot, dword)) => {
                    let v = self.user_value(compute, slot, RootType::Constants, dword);
                    base = base.wrapping_add(v << shift_of(r.sampler));
                    true
                }
                None => match r.index_load.and_then(|l| self.index_in_memory(compute, l)) {
                    Some(i) => {
                        base = base.wrapping_add(i << shift_of(r.sampler));
                        true
                    }
                    None => false,
                },
            };
            if r.dynamic && !known && r.absolute {
                // Nothing known about the entry: no check.
                continue;
            }
            let (shift, limit) = if r.sampler {
                (4, victory_shader::emit::SAMPLER_HEAP_LIMIT)
            } else {
                (5, victory_shader::emit::VIEW_HEAP_LIMIT)
            };
            let mut index = base.wrapping_add(r.offset_dwords * 4) >> shift;
            if index >= limit {
                index = 0;
            }
            let placed = host.checked_kind(index, read_by);
            let mismatch = placed.map_or(Mismatch::Hard, |k| k.mismatch(&declared));
            if known && r.absolute {
                // Which kinds heap-indexed reads meet, once per pair.
                static PAIRS: std::sync::Mutex<
                    Option<std::collections::HashSet<(String, String)>>,
                > = std::sync::Mutex::new(None);
                let pair = (
                    declared.text(),
                    placed.map_or_else(|| "past the heap".into(), |k| k.text()),
                );
                let mut pairs = PAIRS.lock().unwrap_or_else(|e| e.into_inner());
                if pairs
                    .get_or_insert_with(Default::default)
                    .insert(pair.clone())
                {
                    tracing::info!(
                        hash = %stage.shader.hash,
                        declared = pair.0,
                        placed = pair.1,
                        "descriptor check: a heap-indexed read meets this kind"
                    );
                }
            }
            if mismatch == Mismatch::None {
                continue;
            }
            // An indexed read checks only where its entries start.
            let dropped = mismatch == Mismatch::Hard && (!r.dynamic || known);
            ok &= !dropped;
            static SEEN: std::sync::Mutex<Option<std::collections::HashMap<String, u64>>> =
                std::sync::Mutex::new(None);
            let placed_text = placed.map_or_else(|| "past the heap".into(), |k| k.text());
            let key = format!(
                "{} {} slot {} +{}{}{} {} {}",
                stage.shader.stage,
                stage.shader.hash,
                r.slot,
                r.offset_dwords,
                if r.dynamic { " dynamic" } else { "" },
                if r.absolute { " absolute" } else { "" },
                declared.text(),
                placed_text
            );
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            let n = {
                let n = seen
                    .get_or_insert_with(Default::default)
                    .entry(key)
                    .or_insert(0);
                *n += 1;
                *n
            };
            drop(seen);
            if n == 1 || n.is_power_of_two() && n >= 64 {
                // The title's slot behind it (for a fresh table, the entry
                // the table names).
                let title = if fresh.is_some() {
                    table.checked_add(u64::from(r.offset_dwords) * 4)
                } else {
                    heap.start.checked_add(u64::from(index) << shift)
                };
                let view = title
                    .filter(|&at| heap.index(at).is_some())
                    .and_then(crate::descriptor::view);
                tracing::warn!(
                    stage = stage.shader.stage,
                    hash = %stage.shader.hash,
                    pipeline = format_args!("{:#x}", self.pipeline),
                    slot = r.slot,
                    offset_dwords = r.offset_dwords,
                    dynamic = r.dynamic,
                    absolute = r.absolute,
                    known,
                    index,
                    fresh = fresh.is_some(),
                    declared = declared.text(),
                    props = format_args!("{:#x},{:#x}", r.props.0, r.props.1),
                    placed = placed_text,
                    title = describe(view.as_ref()),
                    hard = mismatch == Mismatch::Hard,
                    dropped,
                    times = n,
                    checked = CHECKED.load(Ordering::Relaxed),
                    "descriptor check: a shader reads a descriptor of another kind; {}",
                    if dropped {
                        "its draw or dispatch is dropped"
                    } else {
                        "it runs"
                    }
                );
            }
        }
        ok
    }

    /// The host heap must precede the root signature, including when switching
    /// between the renderer's view heap and either of the title's heaps.
    fn bind_heaps(&mut self, list: &HostList, needs_root_heap: bool) -> bool {
        let mut keys = self.heaps;
        if keys[0] == 0 && needs_root_heap {
            keys[0] = crate::descriptor::ROOT_VIEW_HEAP;
        }
        let ids = [
            crate::descriptor::heap_bind_id(keys[0]),
            crate::descriptor::heap_bind_id(keys[1]),
        ];
        // Compare creation ids as well as pointers: a released heap's
        // address may be reused by a new heap (ABA), which still needs a
        // fresh `SetDescriptorHeaps`.
        if keys != self.host_heaps || ids != self.host_heap_ids {
            let mut heaps: [Option<ID3D12DescriptorHeap>; 2] = [None, None];
            let mut heap_len = 0usize;
            for (key, id) in keys.into_iter().zip(ids) {
                if key == 0 {
                    continue;
                }
                if key != crate::descriptor::ROOT_VIEW_HEAP && id == 0 {
                    once!("a bound descriptor heap was released");
                    return false;
                }
                let Some(heap) = crate::descriptor::recording_heap(key) else {
                    once!("a bound descriptor heap could not be created on the host");
                    return false;
                };
                list.keep(&heap.heap);
                heaps[heap_len] = Some(heap.heap.clone());
                heap_len += 1;
            }
            // SAFETY: an open list and shader-visible heaps retained for the recording.
            unsafe { list.list.SetDescriptorHeaps(&heaps[..heap_len]) };
            self.host_heaps = keys;
            self.host_heap_ids = ids;
            self.host_roots = [0; 2];
        }
        true
    }

    fn raw_slot(
        &mut self,
        host: &crate::descriptor::host::HostHeap,
        heap_id: usize,
        access: RawAccess,
        view: Option<(&std::sync::Arc<crate::shadow::Shadow>, u64, u64)>,
    ) -> Option<u32> {
        let range = view.map(|(shadow, offset, len)| RawBufferRange {
            resource: shadow.buffer.as_raw() as usize,
            offset,
            len,
        });
        // A title address may now resolve to a different buffer. Earlier
        // commands still reference the old slot, so allocate on a miss.
        let index = match self.root_view_slots.entry((heap_id, access, range)) {
            std::collections::hash_map::Entry::Occupied(slot) => *slot.get(),
            std::collections::hash_map::Entry::Vacant(slot) => {
                let index = host.dynamic(1)?;
                match (access, view) {
                    (RawAccess::Read, Some((shadow, offset, len))) => {
                        host.raw_at(index, shadow, offset, len)
                    }
                    (RawAccess::ReadWrite, Some((shadow, offset, len))) => {
                        host.raw_shadow_uav_at(index, shadow, offset, len)
                    }
                    (RawAccess::Read, None) => host.null_raw_at(index),
                    (RawAccess::ReadWrite, None) => host.null_raw_uav_at(index),
                }
                self.dynamic.push((heap_id, index, 1));
                *slot.insert(index)
            }
        };
        Some(index)
    }

    /// The slot of the bound view heap's buffer view table
    /// ([`crate::descriptor::table`]), up to date for `list`.
    fn buffer_table(&self, list: &HostList, compute: bool) -> Option<u32> {
        let key = self.heaps[0];
        if key == 0 {
            return None;
        }
        let heap = crate::descriptor::heap(key as This);
        if heap.samplers() {
            return None;
        }
        crate::descriptor::table::bind(heap.host()?, key, heap.start, list, !compute)
    }

    fn indexed_buffer(
        &mut self,
        list: &HostList,
        selected: u32,
        compute: bool,
    ) -> Option<[u32; 3]> {
        let key = self.heaps[0];
        if key == 0 {
            return None;
        }
        let heap = crate::descriptor::heap(key as This);
        let host = heap.host()?;
        // The console forms this byte offset with a 32-bit shift by five.
        let Some(read) = heap.selected_buffer(selected & (u32::MAX >> 5)) else {
            return Some([self.raw_slot(host, key, RawAccess::Read, None)?, 0, 0]);
        };
        let offset = read.shadow.offset(read.address);
        let bias = offset % 16;
        let index = self.raw_slot(
            host,
            key,
            RawAccess::Read,
            Some((&read.shadow, offset - bias, read.bytes + bias)),
        )?;
        let state = if compute {
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        } else {
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        };
        read.shadow
            .refresh_as(list, read.address, read.bytes, state);
        Some([index, read.stride, bias as u32])
    }

    /// Bind a raw view of the remaining host buffer. A writable view, its
    /// initial contents, and its copy-back all cover the same whole dwords.
    fn root_buffer_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        access: RawAccess,
    ) -> bool {
        let heap_id = self.host_heaps[0];
        let Some(host) = crate::descriptor::recording_heap(heap_id) else {
            once!("a raw root view without a host heap: the draw is dropped");
            return false;
        };
        let shadow = address
            .filter(|&a| a != 0)
            .and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)));
        let view = if let Some((a, s)) = &shadow {
            let offset = s.offset(*a);
            if offset.is_multiple_of(16) {
                let remaining = s.left(*a);
                let len = match access {
                    RawAccess::Read => remaining,
                    RawAccess::ReadWrite => {
                        let Some(len) = writable_root_bytes(remaining) else {
                            once!(
                                remaining,
                                "writable root buffer exceeds the D3D12 element count: the draw is dropped"
                            );
                            return false;
                        };
                        len
                    }
                };
                (len >= 4).then_some((s, offset, len))
            } else {
                once!(
                    address = format_args!("{a:#x}"),
                    "an unaligned raw root view is bound as null"
                );
                None
            }
        } else {
            None
        };
        let Some(index) = self.raw_slot(host, heap_id, access, view) else {
            return false;
        };
        if let (Some((a, s)), Some((_, _, len))) = (&shadow, view) {
            let wanted = match access {
                RawAccess::Read if compute => D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                RawAccess::Read => {
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                        | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                }
                RawAccess::ReadWrite => D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            };
            match access {
                RawAccess::Read => s.refresh_as(list, *a, len.min(1 << 20), wanted),
                RawAccess::ReadWrite => {
                    s.refresh_as(list, *a, len, wanted);
                    s.gpu_writes(list, *a, len);
                }
            }
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Refreshes bindless buffer views and transitions (and uploads
    /// title-written) live bindless textures in the bound CBV/SRV/UAV heap
    /// when any stage indexes `ResourceHeap`.
    fn bindless_textures(&mut self, list: &HostList, p: &HostPipeline, compute: bool) {
        let mut uses_heap = false;
        let mut uses_heap_srv = false;
        let mut uses_heap_uav = false;
        // With a mesh program run as a compute pass (a host without mesh
        // shaders), that pass reads the heap too, and may be the only
        // stage that does.
        let fallback = p.fallback.as_ref().map(|f| &f.stage);
        for stage in p.stages.iter().chain(fallback) {
            for u in &stage.translated().used {
                if u.root == RootType::ResourceHeap {
                    uses_heap = true;
                    match u.kind {
                        ResourceType::Srv => uses_heap_srv = true,
                        ResourceType::Uav => uses_heap_uav = true,
                        _ => {}
                    }
                }
            }
        }
        if !uses_heap || self.heaps[0] == 0 {
            return;
        }
        let Some(host) = crate::descriptor::heap(self.heaps[0] as This).host() else {
            return;
        };
        // Buffer / constant views through ResourceDescriptorHeap: title CPU
        // writes do not bump tex_gen or state_gen, so refresh every time a
        // stage indexes the heap — same cadence as table `readable()`.
        // Also covers off-element / aliased slots (#553).
        host.refresh_bindless_buffers(list);
        if !uses_heap_srv && !uses_heap_uav {
            return;
        }
        let targets_hash = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            compute.hash(&mut h);
            if !compute {
                self.targets.hash(&mut h);
                if let Some(d) = self.depth {
                    d.resource.hash(&mut h);
                    d.state.hash(&mut h);
                }
            }
            h.finish()
        };
        // Only check pending short-read notices on SRV passes, and only for
        // resources that actually have a live SRV in this heap. Notices are
        // cleared per-resource inside `texture::upload_from` once an eligible
        // pass schedules the upload, so UAV-only passes and unrelated heaps
        // never consume another texture's retry notice.
        let short_retry = if uses_heap_srv && crate::descriptor::host::has_short_reads() {
            let pending = crate::descriptor::host::pending_short_resources();
            host.matching_live_srvs(&pending)
        } else {
            Vec::new()
        };
        let cur_tex_gen = host.tex_gen();
        let cur_state_gen = list.state_gen();
        if short_retry.is_empty()
            && let Some(cache) = &self.bindless_cache
            && cache.heap == self.heaps[0]
            && cache.tex_gen == cur_tex_gen
            && cache.state_gen == cur_state_gen
            && cache.targets_hash == targets_hash
            && cache.uses_heap_srv == uses_heap_srv
            && cache.uses_heap_uav == uses_heap_uav
        {
            self.depth_sampled |= cache.depth_sampled;
            return;
        }
        let active_depth = (!compute).then_some(self.depth).flatten();
        // When the view heap and bindless SRV/UAV mode are unchanged, only
        // textures whose descriptor was newly written since `cache.tex_gen`,
        // textures transitioned on `list` since the last sync, targets skipped
        // on the previous draw, the active depth target, or pending short-read
        // textures in this heap can need transitions/uploads — O(|new| + |dirty|).
        let incremental = self
            .bindless_cache
            .as_ref()
            .filter(|cache| {
                cache.heap == self.heaps[0]
                    && cache.uses_heap_srv == uses_heap_srv
                    && cache.uses_heap_uav == uses_heap_uav
            })
            .and_then(|cache| {
                let (_, mut candidates) = list.dirty_since(cache.dirty_cursor);
                candidates.extend_from_slice(&cache.skipped_targets);
                candidates.extend_from_slice(&short_retry);
                if let Some(d) = active_depth {
                    candidates.push(d.resource);
                }
                host.pin_live_subset_since(cache.tex_gen, &candidates, uses_heap_srv, uses_heap_uav)
            });
        let (tex_gen, live_srvs, live_uavs) = match incremental {
            Some(subset) => subset,
            None => {
                #[cfg(test)]
                {
                    self.bindless_full_walks += 1;
                }
                host.live_textures(uses_heap_srv, uses_heap_uav)
            }
        };
        let is_active_rt = |r: usize| !compute && self.targets.contains(&r);
        // When a pipeline accesses both bindless SRVs and bindless UAVs, a
        // texture with both descriptor types resident in the heap is ambiguous:
        // preserve its SRV state only if the title explicitly transitioned it
        // for shader reading (not merely because an earlier bindless sweep put
        // it in SRV state).
        let srv_set: Option<std::collections::HashSet<usize>> =
            (uses_heap_srv && uses_heap_uav && live_srvs.len() > 64)
                .then(|| live_srvs.iter().map(|(h, _)| h.get() as usize).collect());
        let in_srvs = |r: usize| {
            uses_heap_srv
                && match &srv_set {
                    Some(set) => set.contains(&r),
                    None => live_srvs.iter().any(|(h, _)| h.get() as usize == r),
                }
        };
        let mut uav_transitioned: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        if uses_heap_uav {
            const UNORDERED_ACCESS: u32 = 0x8;
            for held in &live_uavs {
                let r = held.get() as usize;
                if is_active_rt(r) || active_depth.is_some_and(|d| d.resource == r) {
                    continue;
                }
                if in_srvs(r)
                    && list
                        .explicit_state(held.get())
                        .is_some_and(|s| s & SHADER_RESOURCE != 0)
                {
                    continue;
                }
                #[cfg(feature = "diagnostics")]
                {
                    crate::capture::note_target(held.get());
                    crate::capture::note_op_write(held.get());
                }
                crate::list::ensure_bindless_state(list, held.get(), UNORDERED_ACCESS);
                if uses_heap_srv {
                    uav_transitioned.insert(r);
                }
            }
        }
        let mut depth_sampled = false;
        if uses_heap_srv {
            for (held, first) in &live_srvs {
                let r = held.get() as usize;
                if is_active_rt(r) || uav_transitioned.contains(&r) {
                    continue;
                }
                if let Some(d) = active_depth
                    && d.resource == r
                {
                    if d.state == DEPTH_READ {
                        depth_sampled = true;
                        crate::list::ensure_bindless_state(
                            list,
                            held.get(),
                            DEPTH_READ | SHADER_RESOURCE,
                        );
                    }
                    continue;
                }
                if let Some(t) = crate::resource::host_resource(held.get()) {
                    if crate::texture::reads_title_memory(held.get(), t) {
                        crate::texture::PICKING.set(Some(u64::MAX));
                        crate::texture::upload_from(list, held.get(), t, *first);
                        crate::texture::PICKING.set(None);
                    }
                    crate::list::ensure_bindless_state(list, held.get(), SHADER_RESOURCE);
                }
            }
        }
        let mut skipped_targets = if compute {
            Vec::new()
        } else {
            self.targets.clone()
        };
        if let Some(d) = active_depth {
            skipped_targets.push(d.resource);
        }
        self.depth_sampled |= depth_sampled;
        self.bindless_cache = Some(BindlessCache {
            heap: self.heaps[0],
            tex_gen,
            state_gen: list.state_gen(),
            targets_hash,
            uses_heap_srv,
            uses_heap_uav,
            dirty_cursor: list.dirty_len(),
            skipped_targets,
            depth_sampled,
        });
    }

    /// An indirect mesh dispatch on a host without mesh shaders: as
    /// `dispatch_mesh_without_mesh_shaders`, with the thread group count
    /// known only to the GPU. A pass of ours splits the title's first
    /// record into parts the mesh buffer holds (`crate::indirect::mesh_draws`)
    /// and every part that count could have is recorded; those past it run
    /// empty. A part has at most `part` groups (tests make them small).
    /// False if it could not be recorded.
    fn execute_indirect_mesh(
        &mut self,
        list: &HostList,
        (p, f, part): (&HostPipeline, &MeshFallbackPipeline, u32),
        (args, arguments, stride, max): (&std::sync::Arc<crate::shadow::Shadow>, u64, u32, u32),
        counter: Option<(&std::sync::Arc<crate::shadow::Shadow>, u64)>,
    ) -> bool {
        use crate::indirect::{DISPATCH_BYTES, DRAW_BYTES};
        const DRAW: u32 = 0;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        // shortcut: the groups past this many parts are not drawn. With the
        // largest group records the mesh buffer holds some 1900 a part.
        const PARTS_MOST: u32 = 16;
        if max == 0 {
            return true;
        }
        let (Some(dispatches), Some(draws)) = (
            host_signature(DISPATCH, DISPATCH_BYTES),
            host_signature(DRAW, DRAW_BYTES),
        ) else {
            return false;
        };
        let most = u64::from(GROUPS_MOST);
        let Some((buffer, gpu, bytes)) = list.mesh_buffer(most * f.group_bytes) else {
            return false;
        };
        let capacity = (bytes / f.group_bytes.max(1)).min(most) as u32;
        let capacity = capacity.min(part);
        if capacity == 0 {
            return false;
        }
        let parts = GROUPS_MOST.div_ceil(capacity).min(PARTS_MOST);
        let made_bytes = u64::from(parts) * u64::from(DRAW_BYTES + DISPATCH_BYTES);
        let Some(made) = list.rect_args(made_bytes) else {
            return false;
        };
        let len = u64::from(stride) * u64::from(max);
        args.refresh(list, arguments, len);
        if let Some((s, at)) = counter {
            s.refresh_as(list, at, 4, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        }
        // SAFETY: a live buffer the list keeps.
        let made_gpu = unsafe { made.GetGPUVirtualAddress() };
        // Both passes bind compute pipelines of their own. The clamp leaves
        // the title's records as they are: its copy is what the pass reads.
        // Only the records this runs are clamped and charged to the list's
        // budget: what the title left in the others (stale records of an
        // earlier frame) would spend it, and later indirect work in the list
        // would run empty.
        let count = counter.map(|(s, at)| (&s.buffer, s.offset(at)));
        let Some(clamped) = crate::indirect::clamp_uav(
            list,
            (&args.buffer, args.offset(arguments)),
            count,
            (MESH, stride, mesh_fallback_records(max)),
        ) else {
            self.host_pipeline = 0;
            self.host_roots[1] = 0;
            return false;
        };
        // SAFETY: a live buffer the list keeps.
        let clamped_gpu = unsafe { clamped.GetGPUVirtualAddress() };
        let ready =
            crate::indirect::mesh_draws(list, clamped_gpu, made_gpu, (parts, capacity, f.vertices));
        crate::indirect::done_uav(list, &clamped);
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        if !ready || !self.prepare_with(list, p, false) {
            return false;
        }
        let l = &list.list;
        // What the pass made is read as arguments, then left for unordered
        // access again, as the list's argument scratch is between uses
        // (`crate::rectangles` shares it).
        let to_arguments = |back: bool| {
            let states = [
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ];
            let (before, after) = (states[usize::from(back)], states[usize::from(!back)]);
            crate::host::transition(&made, before, after)
        };
        let root = f.root.as_raw() as usize;
        // SAFETY: an open list, live root signatures and pipelines.
        unsafe {
            l.ResourceBarrier(&[to_arguments(false)]);
            if self.host_roots[1] != root {
                l.SetComputeRootSignature(&f.root);
                self.host_roots[1] = root;
                list.keep(&f.root);
            }
            l.SetPipelineState(&f.pso);
        }
        list.keep(&f.pso);
        if !self.bind_stage(list, &f.stage, &p.structured_strides, (false, true)) {
            return false;
        }
        // A depth texture the mesh program writes through unordered access:
        // its colour twin holds what it lacks from here on. (The pipeline's
        // own stages were bound, and their twins noted, before this one.)
        list.twins_written();
        self.ensure_depth_state(list);
        let (unordered, read) = (
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(
                format!(
                    "indirect mesh as compute and draw, max {max} at {arguments:#x}, \
                     {parts} parts of {capacity} groups"
                ),
                false,
            )
        });
        // SAFETY: an open list and the live mesh buffer.
        unsafe {
            l.SetComputeRootUnorderedAccessView(f.uav, gpu);
            l.SetGraphicsRootShaderResourceView(f.srv, gpu);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }
        // Binding may have moved the count's buffer back to a shader state.
        if let Some((s, _)) = counter {
            list.use_shadow(&s.buffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        }
        for part in 0..parts {
            list.use_shadow(&buffer, unordered);
            // SAFETY: as above; the arguments are ours, in a buffer the
            // list keeps alive, and bounded by the pass that made them: the
            // part's compute pass stays within the mesh buffer.
            unsafe {
                l.SetPipelineState(&f.pso);
                l.SetComputeRoot32BitConstant(f.first_group, part * capacity, 0);
                l.ExecuteIndirect(
                    &dispatches,
                    1,
                    &made,
                    u64::from(parts * DRAW_BYTES + part * DISPATCH_BYTES),
                    count.map(|c| c.0),
                    count.map_or(0, |c| c.1),
                );
            }
            list.use_shadow(&buffer, read);
            // SAFETY: as above; the part's draw reads what its compute pass
            // stored.
            unsafe {
                l.SetPipelineState(&p.pso);
                l.ExecuteIndirect(
                    &draws,
                    1,
                    &made,
                    u64::from(part * DRAW_BYTES),
                    count.map(|c| c.0),
                    count.map_or(0, |c| c.1),
                );
            }
        }
        // SAFETY: an open list and the live argument scratch.
        unsafe { l.ResourceBarrier(&[to_arguments(true)]) };
        self.set_topology(list, self.topology);
        true
    }

    /// A mesh dispatch of `x` thread groups on a host without mesh shaders:
    /// the pipeline's compute half runs the mesh program, which stores each
    /// group's vertices and primitives in the host's mesh buffer, and a
    /// draw of one instance per group reads them back. More groups than the
    /// buffer holds run in parts, each told its first group's number.
    /// A part has at most `part` groups (tests make them small).
    /// `prepare_for` has bound the draw's state. False if it could not be
    /// recorded.
    fn dispatch_mesh_without_mesh_shaders(
        &mut self,
        list: &HostList,
        p: &HostPipeline,
        f: &MeshFallbackPipeline,
        (x, part): (u32, u32),
    ) -> bool {
        if x == 0 {
            return true;
        }
        // No dispatch has more groups in a dimension; a count past that
        // (a value the title left unset) would otherwise record millions
        // of parts.
        let x = x.min(GROUPS_MOST);
        let bytes = u64::from(x) * f.group_bytes;
        let buffer = list.mesh_buffer(bytes);
        // A dispatch has at most `GROUPS_MOST` groups in a dimension.
        let most = u64::from(x.min(part).min(GROUPS_MOST));
        let capacity = buffer
            .as_ref()
            .map_or(0, |b| (b.2 / f.group_bytes.max(1)).min(most) as u32);
        let Some((buffer, gpu, _)) = buffer.filter(|_| capacity > 0) else {
            once!(
                groups = x,
                bytes,
                "no memory for a mesh dispatch without mesh shaders"
            );
            return false;
        };
        let l = &list.list;
        let root = f.root.as_raw() as usize;
        // SAFETY: an open list, live root signatures and pipelines.
        unsafe {
            if self.host_roots[1] != root {
                l.SetComputeRootSignature(&f.root);
                self.host_roots[1] = root;
                list.keep(&f.root);
            }
            l.SetPipelineState(&f.pso);
        }
        list.keep(&f.pso);
        // The mesh stage's arguments are the title's graphics ones.
        if !self.bind_stage(list, &f.stage, &p.structured_strides, (false, true)) {
            return false;
        }
        // A depth texture the mesh program writes through unordered access:
        // its colour twin holds what it lacks from here on. (The pipeline's
        // own stages were bound, and their twins noted, before this one.)
        list.twins_written();
        self.ensure_depth_state(list);
        let (unordered, read) = (
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        // SAFETY: an open list and the live mesh buffer.
        unsafe {
            l.SetComputeRootUnorderedAccessView(f.uav, gpu);
            l.SetGraphicsRootShaderResourceView(f.srv, gpu);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }
        let mut first = 0;
        while first < x {
            let groups = (x - first).min(capacity);
            list.use_shadow(&buffer, unordered);
            // SAFETY: as above; the part covers at most `capacity` thread
            // groups' records, which the buffer holds.
            unsafe {
                l.SetPipelineState(&f.pso);
                l.SetComputeRoot32BitConstant(f.first_group, first, 0);
                l.Dispatch(groups, 1, 1);
            }
            list.use_shadow(&buffer, read);
            // SAFETY: as above; the draw reads what the part stored.
            unsafe {
                l.SetPipelineState(&p.pso);
                l.DrawInstanced(f.vertices, groups, 0, 0);
            }
            first += groups;
        }
        // The title's own topology again, for its next draw.
        self.set_topology(list, self.topology);
        true
    }

    /// Records the indirect-argument gathering effect of executing `p` on
    /// `list` (`crate::indirect::gather`):
    /// - Pipelines whose stages cannot store to buffers leave gathering open.
    /// - Pipelines that may write arbitrary buffers (raw-address stores via
    ///   `SLOT_MEMORY`, heap-indexed UAVs, tables without static `used`
    ///   entries, or a storing stage whose written buffers cannot all be named
    ///   including any unresolved table UAV handle) close gathering globally
    ///   for the rest of the list (`gather_close`).
    /// - Pipelines whose storing stages write only to named root/table buffer
    ///   UAVs poison only those specific host copies (`gather_close_buffer`),
    ///   so later indirect draws reading unmodified argument buffers can still
    ///   gather into the list prelude.
    fn note_gather_writes(&self, list: &HostList, p: &HostPipeline, compute: bool) {
        let mut poisoned: Vec<u64> = Vec::new();
        // A mesh program run as a compute pass writes what it writes too.
        for stage in p.stages.iter().chain(p.fallback.as_ref().map(|f| &f.stage)) {
            let t = stage.translated();
            if !t.stores {
                continue;
            }
            if (t.address_stores
                && t.host
                    .root_srvs
                    .contains(&victory_shader::emit::SLOT_MEMORY))
                || t.used.iter().any(|u| {
                    u.kind == ResourceType::Uav
                        && !matches!(u.root, RootType::Table | RootType::Descriptor)
                })
                || t.host.user_data.iter().any(|u| {
                    u.root == RootType::Table
                        && !t
                            .used
                            .iter()
                            .any(|r| r.root == RootType::Table && r.slot == u.slot)
                })
            {
                list.gather_close();
                return;
            }
            let mut stage_buffers = 0usize;
            for &(slot, _) in &stage.root_uavs {
                if slot == victory_shader::emit::SLOT_FAULTS {
                    continue;
                }
                if let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    && address != 0
                {
                    match crate::shadow::find(address, 1) {
                        Some(s) => {
                            poisoned.push(s.id());
                            stage_buffers += 1;
                        }
                        None => {
                            list.gather_close();
                            return;
                        }
                    }
                }
            }
            for u in t
                .used
                .iter()
                .filter(|u| u.root == RootType::Table && u.kind == ResourceType::Uav)
            {
                let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot))
                else {
                    list.gather_close();
                    return;
                };
                let handle = table + u64::from(u.table_offset_dwords) * 4;
                let Some(view) = crate::descriptor::view(handle) else {
                    // Soundness: an unmirrored/unresolved UAV handle may write
                    // any buffer, so conservatively close gathering globally.
                    list.gather_close();
                    return;
                };
                let placed = crate::descriptor::host::placed_texture(&view);
                let view = placed.as_ref().unwrap_or(&view);
                if let Some((address, len)) = crate::descriptor::host::buffer_range(view) {
                    if len > 0 {
                        match crate::shadow::find(address, len) {
                            Some(shadow) => {
                                poisoned.push(shadow.id());
                                stage_buffers += 1;
                            }
                            None => {
                                list.gather_close();
                                return;
                            }
                        }
                    }
                } else if matches!(
                    view,
                    View::Unordered(r, _, None)
                        if crate::resource::buffer_bytes(*r as This).is_some()
                ) {
                    list.gather_close();
                    return;
                }
            }
            if stage_buffers == 0 {
                list.gather_close();
                return;
            }
        }
        for id in poisoned {
            list.gather_close_buffer(id);
        }
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            self.frame_op(&format!("dispatch {x}x{y}x{z}"), true, false);
            return;
        }
        let Some(p) = self.bound_pipeline(true) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            #[cfg(feature = "diagnostics")]
            self.frame_op(&format!("dispatch {x}x{y}x{z}"), true, false);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            #[cfg(feature = "diagnostics")]
            self.frame_op_why(
                &format!("dispatch {x}x{y}x{z}"),
                true,
                false,
                Some("a graphics pipeline"),
            );
            return;
        }
        if !self.bind(list, &p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        #[cfg(feature = "diagnostics")]
        self.frame_op(&format!("dispatch {x}x{y}x{z}"), true, true);
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        self.note_gather_writes(list, &p, true);
        list.count_write();
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// In frames listed for capture, at debug level: one line per draw or
    /// dispatch with its render targets and depth (named as in
    /// `VICTORY_DUMP_TARGETS` files) and whether it ran, else why it was
    /// dropped (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn frame_op(&self, kind: &str, compute: bool, ran: bool) {
        self.frame_op_why(kind, compute, ran, None);
    }

    /// [`State::frame_op`] with the reason a drop happened before a pipeline
    /// was looked at (else the pipeline's).
    #[cfg(feature = "diagnostics")]
    fn frame_op_why(&self, kind: &str, compute: bool, ran: bool, why: Option<&str>) {
        // Taken even when not logged, so one op's writes never reach the next.
        let writes = crate::capture::take_op_writes();
        if !tracing::enabled!(tracing::Level::DEBUG)
            || !crate::capture::listed(crate::queue::presented() + 1)
        {
            return;
        }
        let targets: Vec<String> = if compute {
            Vec::new()
        } else {
            self.targets.iter().map(|t| format!("{t:#x}")).collect()
        };
        let writes: Vec<String> = writes.iter().map(|t| format!("{t:#x}")).collect();
        if ran {
            self.dump_reads(compute);
        }
        let outcome = if ran {
            "ran".to_owned()
        } else {
            match why {
                Some(why) => format!("DROPPED: {why}"),
                None => format!(
                    "DROPPED: {}",
                    crate::pipeline::drop_reason(self.pipeline as This)
                ),
            }
        };
        let text = self.op_text(kind.to_owned(), compute);
        tracing::debug!(
            "frame {} op: pipeline {:#x} topology {} targets {targets:?} depth {:#x} writes {writes:?} -> {outcome} || {text}",
            crate::queue::presented() + 1,
            self.pipeline,
            self.topology,
            self.depth.map_or(0, |d| d.resource)
        );
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (`table` values with the view the table starts at; `const` values
    /// with the view at that heap index, if one is, since dynamic resources
    /// may index the heap with one, though most are plain numbers), the
    /// host addresses of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = self.bound_pipeline(compute) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                if u.root == RootType::ResourceHeap {
                    let _ = write!(text, " u{}=buffer view table", u.index);
                    continue;
                }
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let table = u.root == RootType::Table;
                let (kind, index) = if table {
                    ("table", v / 32)
                } else {
                    ("const", v)
                };
                let _ = write!(text, " u{}={v:#x} {kind}", u.index);
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let at = if table { "heap" } else { "as heap index" };
                    let _ = write!(text, " [{at} {index}: {slot}]");
                }
            }
            for &(slot, _) in stage
                .root_cbvs
                .iter()
                .chain(&stage.root_srvs)
                .chain(&stage.root_uavs)
            {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        // A host without mesh shaders runs the mesh program as a compute
        // pass and draws what it stored. From the pipeline the draw is
        // prepared with (`prepare_for`): one made for the strides bound now
        // has its compute half too, where the title pipeline has no host
        // pipeline for every stride.
        let bound = self.bound_pipeline(false);
        let fallback = bound
            .as_deref()
            .and_then(|p| p.fallback.as_ref().map(|f| (p, f)));
        let ran = self.prepare_for(list, false, true)
            && match fallback {
                Some((p, f)) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x} as compute and draw"), false));
                    let ran = self.dispatch_mesh_without_mesh_shaders(list, p, f, (x, u32::MAX));
                    if ran {
                        self.note_gather_writes(list, p, false);
                        DRAWN.fetch_add(1, Ordering::Relaxed);
                        MESH_WITHOUT_MESH_SHADERS[0].fetch_add(1, Ordering::Relaxed);
                    }
                    ran
                }
                None => match list.list.cast::<ID3D12GraphicsCommandList6>() {
                    Ok(l6) => {
                        #[cfg(feature = "diagnostics")]
                        list.note_op(|| self.op_text(format!("mesh {x}"), false));
                        // SAFETY: an open list with everything set.
                        unsafe { l6.DispatchMesh(x, 1, 1) };
                        if let Some(p) = self.bound_pipeline(false) {
                            self.note_gather_writes(list, &p, false);
                        }
                        DRAWN.fetch_add(1, Ordering::Relaxed);
                        true
                    }
                    Err(_) => false,
                },
            };
        #[cfg(feature = "diagnostics")]
        self.frame_op("mesh", false, ran);
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&mut self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true)
            && !host.refresh_slot(index, list)
        {
            self.depth_sampled |= readable(list, u, &view, self.read_only_depth(compute));
        }
    }

    /// Gives tables with shader resources or constants slots owned by the
    /// command list. Later title descriptor writes cannot change what a
    /// recorded draw samples. Constants also get 256-byte-aligned storage,
    /// filled at submission. Identical descriptors share slots within a list.
    #[cfg(test)]
    fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        compute: bool,
        aliases: &[StructuredBinding],
    ) -> Vec<(u8, u32)> {
        self.tables_with_strides(list, used, compute, aliases, &[]);
        self.made.clone()
    }

    fn tables_with_strides(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        compute: bool,
        aliases: &[StructuredBinding],
        structured_strides: &[(victory_shader::emit::StructuredBinding, u32)],
    ) {
        let read_only_depth = self.read_only_depth(compute);
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        // Taken out so `self` stays free to borrow; restored at the end.
        let mut made = std::mem::take(&mut self.made);
        made.clear();
        let mut fresh = Slots::default();
        for u in used
            .iter()
            .filter(tables)
            .filter(|u| matches!(u.kind, ResourceType::Cbv | ResourceType::Srv))
        {
            fresh.insert(u.slot);
        }
        for a in aliases {
            fresh.insert(a.slot);
        }
        for (binding, _) in structured_strides {
            fresh.insert(binding.slot);
        }
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA, 2,048 samplers anywhere)
        // are made afresh too: the title's slots there have no host slot of
        // their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.insert(u.slot);
            }
        }
        for u in used.iter().filter(tables) {
            if !fresh.contains(u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let list_views = crate::descriptor::host::list_views();
        let list_ptr = std::ptr::from_ref(list) as usize;
        for slot in fresh.iter() {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            // Their places among `used`, on the stack for the common <= 32 entries.
            let mut places: Few<u16, 32> = Few::default();
            for (i, u) in used.iter().enumerate() {
                if u.root == RootType::Table && u.slot == slot {
                    places.push(i as u16);
                }
            }
            let entries = || places.iter().map(|&i| &used[usize::from(i)]);
            let mut table_aliases: Few<StructuredBinding, 4> = Few::default();
            for &a in aliases {
                if a.slot == slot {
                    table_aliases.push(a);
                }
            }
            let mut table_strides: Few<Option<(victory_shader::emit::StructuredBinding, u32)>, 8> =
                Few::default();
            for &(binding, stride) in structured_strides {
                if binding.slot == slot {
                    table_strides.push(Some((binding, stride)));
                }
            }
            let Some(heap) = self.heap_of(table) else {
                self.table_failed |= !table_strides.is_empty();
                continue;
            };
            let Some(host) = heap.host() else {
                self.table_failed |= !table_strides.is_empty();
                continue;
            };
            let step_dwords: u32 = if heap.samplers() { 4 } else { 8 };
            let step_bytes: u64 = u64::from(step_dwords) * 4;
            let min_slot = entries()
                .map(|u| u.table_offset_dwords / step_dwords)
                .chain(table_aliases.iter().map(|a| a.entry))
                .chain(
                    table_strides
                        .iter()
                        .flatten()
                        .map(|(b, _)| b.offset_dwords / step_dwords),
                )
                .min()
                .unwrap_or(0);
            let max_slot = entries()
                .map(|u| u.table_offset_dwords / step_dwords)
                .chain(table_aliases.iter().map(|a| a.entry))
                .chain(
                    table_strides
                        .iter()
                        .flatten()
                        .map(|(b, _)| b.offset_dwords / step_dwords),
                )
                .max()
                .unwrap_or(0);
            // `UsedResource.table_offset_dwords` records the base of each static
            // descriptor or dynamic array range in this table; translated shaders
            // only add non-negative dynamic offsets (`lower.rs::heap_index`), so
            // they never index below `min_slot`. Guarantee `first >= host.count()
            // >= start_slot` so `first - start_slot` cannot underflow when a host
            // heap is capped below `min_slot`.
            let start_slot = if host.count() >= min_slot {
                min_slot
            } else {
                0
            };
            debug_assert!(
                entries().all(|u| u.table_offset_dwords / step_dwords >= start_slot),
                "table entry offset below start_slot"
            );
            let n = max_slot - start_slot + 1;
            let last = u64::from(max_slot)
                .checked_mul(step_bytes)
                .and_then(|bytes| table.checked_add(bytes));
            if last.and_then(|end| heap.index(end)).is_none() {
                self.table_failed |= !table_strides.is_empty();
                for u in entries() {
                    self.table_entry(list, u, compute);
                }
                continue;
            }
            // Reject impossible table sizes before hashing/scanning the range
            // (checks total dynamic capacity, not current free space, so valid
            // cache hits still succeed when the dynamic region is full).
            if n > host.dynamic_capacity() {
                self.table_failed = true;
                continue;
            }
            // Read the descriptor headers in `start_slot..=max_slot` once into
            // reusable scratch buffers, batch-check liveness with at most one
            // shared read lock (and zero `View` clones), then hash and verify
            // cache hits against saved `[u32; 3]` words in `self.fresh_records`.
            let heap_ptr = std::ptr::from_ref(heap) as usize;
            let copy_gen = list.copy_gen();
            self.table_raws.clear();
            self.table_raws.reserve(n as usize);
            for k in start_slot..=max_slot {
                let at = table + u64::from(k) * step_bytes;
                // SAFETY: views are 32 bytes and samplers are 16 bytes, so `[u32; 3]`
                // (12 bytes) fits safely inside every in-bounds slot in title memory.
                self.table_raws
                    .push(unsafe { (at as *const [u32; 3]).read_unaligned() });
            }
            crate::descriptor::check_dead_slots(&self.table_raws, &mut self.table_dead);
            let key = table_hash(
                heap_ptr,
                copy_gen,
                &self.table_raws,
                &self.table_dead,
                start_slot,
                step_dwords,
                &table_aliases,
                &table_strides,
            );
            let offset_for = |first: u32| {
                first
                    .checked_sub(start_slot)
                    .and_then(|base| base.checked_mul(step_bytes as u32))
            };
            let same =
                self.fresh_tables.get(&key).filter(|entry| {
                    if entry.heap != heap_ptr
                        || entry.copy_gen != copy_gen
                        || entry.slots != n
                        || self.fresh_records[entry.records_at..entry.records_at + n as usize]
                            != self.table_raws[..]
                    {
                        return false;
                    }
                    if entry.aliases.len() != table_aliases.len()
                        || !entry.aliases.iter().zip(table_aliases.iter()).all(
                            |(&(rel, stride), a)| rel == a.entry - start_slot && stride == a.stride,
                        )
                    {
                        return false;
                    }
                    if entry.strides.len() != table_strides.len()
                        || !entry
                            .strides
                            .iter()
                            .zip(table_strides.iter().flatten())
                            .all(|(&(rel, uav, stride), (b, s))| {
                                rel == b.offset_dwords / step_dwords - start_slot
                                    && uav == b.uav
                                    && stride == *s
                            })
                    {
                        return false;
                    }
                    // Compare view liveness against what was compiled into `first`:
                    // a newly released view misses and rebuilds a null slot, while
                    // an unchanged stale slot already compiled as null reuses it.
                    let mut dead_cursor = 0usize;
                    for (idx, &is_dead) in self.table_dead.iter().enumerate() {
                        let i = idx as u32;
                        let was_dead = if entry.dead_slots.get(dead_cursor) == Some(&i) {
                            dead_cursor += 1;
                            true
                        } else {
                            false
                        };
                        if is_dead != was_dead {
                            return false;
                        }
                    }
                    dead_cursor == entry.dead_slots.len()
                });
            if let Some(&FreshTableEntry { first, .. }) = same {
                let Some(byte_offset) = offset_for(first) else {
                    self.table_failed = true;
                    continue;
                };
                for alias in table_aliases.iter() {
                    if !entries().any(|u| {
                        u.kind != ResourceType::Cbv
                            && u.table_offset_dwords / step_dwords == alias.entry
                    }) {
                        let _ = host.refresh_slot(first + (alias.entry - start_slot), list);
                    }
                }
                for u in entries().filter(|u| u.kind != ResourceType::Cbv) {
                    let dyn_index = first + (u.table_offset_dwords / step_dwords - start_slot);
                    if !host.refresh_slot(dyn_index, list) {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        if let Some(view) = crate::descriptor::view(handle) {
                            self.depth_sampled |= readable(list, u, &view, read_only_depth);
                        }
                    }
                }
                made.push((slot, byte_offset));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // Without its fresh table the shader would read the title's
                // slots, which for constant buffer tables hold no constant
                // view: a texture or null descriptor read as a buffer gives
                // a wild address on AMD and the device is removed (#59).
                // The draw or dispatch is dropped instead.
                self.table_failed = true;
                continue;
            };
            let Some(byte_offset) = offset_for(first) else {
                host.release(first, n);
                self.table_failed = true;
                continue;
            };
            let use_list_views = list_views && !heap.samplers();
            if use_list_views && self.recording == 0 {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                self.recording = NEXT.fetch_add(1, Ordering::Relaxed);
            }
            let recording = self.recording;
            // Compile the whole compact range before publishing it in
            // `fresh_tables` / `dynamic`: a shader may index entries between
            // `start_slot` and `max_slot` not listed in `entries` (descriptor
            // arrays), and a transient `list.scratch` failure must not leave a
            // partial range cached for subsequent draws.
            let mut complete = true;
            let mut dead_slots = Vec::new();
            let mut listed_slots: Few<Option<([u32; 3], crate::descriptor::host::Listed)>, 32> =
                Few::default();
            for k in start_slot..=max_slot {
                let rel = k - start_slot;
                let idx = rel as usize;
                let index = first + rel;
                let at = table + u64::from(k) * step_bytes;
                let stride_req = table_strides
                    .iter()
                    .flatten()
                    .find(|(b, _)| b.offset_dwords / step_dwords == k);
                // A view this list has made before: copy its descriptor from
                // thread-local CPU-only staging via `CopyDescriptorsSimple`,
                // skipping repeated D3D12 view creation and repeated buffer-SRV
                // refreshes on the same host list (`VICTORY_LIST_VIEWS`).
                if use_list_views
                    && stride_req.is_none()
                    && !table_aliases.iter().any(|a| a.entry == k)
                    && !self.table_dead[idx]
                    && self.table_raws[idx][0] == crate::descriptor::VIEW_TAG
                    && let Some(listed) = host.make_listed(recording, self.table_raws[idx], index)
                {
                    listed_slots.push(Some((self.table_raws[idx], listed)));
                    continue;
                }
                listed_slots.push(None);
                let view = crate::descriptor::view(at);
                if let Some((binding, stride)) = stride_req {
                    // Check the same view that is copied to the list's heap.
                    // Failure to mirror it must not bind a null of another kind.
                    complete = view.is_some_and(|view| {
                        crate::descriptor::host::structured_stride(&view, binding.uav)
                            == Some(*stride)
                            && host.make_at(index, &view)
                    });
                    if !complete {
                        once!(
                            slot,
                            "a structured buffer view could not be bound at its required stride"
                        );
                        break;
                    }
                    continue;
                }
                match view {
                    Some(View::Constant(address, size)) if address != 0 => {
                        let Some(gpu) = list.fill_scratch(address as *const u8, u64::from(size))
                        else {
                            complete = false;
                            break;
                        };
                        host.constant_at(index, gpu, size);
                        #[cfg(feature = "diagnostics")]
                        if let Some(hash) = crate::cbv_trace::on() {
                            tracing::info!(
                                hash,
                                slot,
                                entry = k,
                                address = format_args!("{address:#x}"),
                                size,
                                scratch = format_args!("{gpu:#x}"),
                                host_index = index,
                                "cbv trace: fresh table constant view filled from title memory"
                            );
                        }
                    }
                    Some(view) => {
                        let aliased = table_aliases.iter().any(|alias| {
                            alias.entry == k && host.make_structured_at(index, &view, alias.stride)
                        });
                        if !aliased {
                            host.make_at(index, &view);
                        }
                    }
                    None => {
                        host.make_null(index);
                        if self.table_raws[idx][0] == crate::descriptor::VIEW_TAG {
                            dead_slots.push(rel);
                        }
                        if entries().any(|u| u.table_offset_dwords / step_dwords == k) {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            if !complete {
                host.release(first, n);
                self.table_failed = true;
                continue;
            }
            for k in start_slot..=max_slot {
                let rel = k - start_slot;
                let index = first + rel;
                if !host.refresh_slot(index, list) {
                    if let Some((words, mut listed)) = listed_slots[rel as usize] {
                        let was = listed.ready;
                        for u in entries().filter(|u| u.table_offset_dwords / step_dwords == k) {
                            let shadow_gen = list.shadow_state_gen();
                            if u.kind == ResourceType::Srv
                                && listed.buffer_only
                                && listed.ready == (list_ptr, copy_gen, shadow_gen)
                            {
                                continue;
                            }
                            if let Some(view) =
                                crate::descriptor::host::listed_view(recording, words)
                            {
                                self.depth_sampled |= readable(list, u, &view, read_only_depth);
                                if u.kind == ResourceType::Srv && listed.buffer_only {
                                    listed.ready = (list_ptr, copy_gen, list.shadow_state_gen());
                                }
                            }
                        }
                        if listed.ready != was {
                            crate::descriptor::host::note_listed(recording, words, listed);
                        }
                    } else {
                        let at = table + u64::from(k) * step_bytes;
                        if let Some(view) = crate::descriptor::view(at)
                            && !matches!(view, View::Constant(..))
                        {
                            for u in entries().filter(|u| u.table_offset_dwords / step_dwords == k)
                            {
                                self.depth_sampled |= readable(list, u, &view, read_only_depth);
                            }
                        }
                    }
                }
            }
            let records_at = self.fresh_records.len();
            self.fresh_records.extend_from_slice(&self.table_raws);
            let saved_aliases = table_aliases
                .iter()
                .map(|a| (a.entry - start_slot, a.stride))
                .collect();
            let saved_strides = table_strides
                .iter()
                .flatten()
                .map(|(b, s)| (b.offset_dwords / step_dwords - start_slot, b.uav, *s))
                .collect();
            self.fresh_tables.insert(
                key,
                FreshTableEntry {
                    heap: heap_ptr,
                    copy_gen,
                    slots: n,
                    records_at,
                    first,
                    dead_slots,
                    aliases: saved_aliases,
                    strides: saved_strides,
                },
            );
            self.dynamic.push((heap_ptr, first, n));
            made.push((slot, byte_offset));
        }
        self.made = made;
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        // Listed frames: `frame_op` logs every operation instead.
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = !listed && TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = self.bound_pipeline(compute) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    /// In frames listed for capture, with `VICTORY_DUMP_READS=<hash>,...`
    /// (diagnostics builds): for an op whose bound pipeline has one of those
    /// stage hashes, the title memory behind every buffer and constant view
    /// its tables give it, as the op is replayed, saved to the capture folder
    /// as `frame-<n>-<hash>-<k>-s<slot>+<offset>-<address>.bin` (`k` counts
    /// that program's ops in the frame; at most 4 MiB a view).
    #[cfg(feature = "diagnostics")]
    fn dump_reads(&self, compute: bool) {
        static HASHES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        static SEEN: std::sync::Mutex<(u64, Vec<(String, u32)>)> =
            std::sync::Mutex::new((0, Vec::new()));
        let hashes = HASHES.get_or_init(|| {
            std::env::var("VICTORY_DUMP_READS")
                .unwrap_or_default()
                .split(',')
                .map(|h| h.trim().to_owned())
                .filter(|h| !h.is_empty())
                .collect()
        });
        let frame = crate::queue::presented() + 1;
        if hashes.is_empty() || !crate::capture::listed(frame) {
            return;
        }
        let Some(folder) = crate::capture::folder() else {
            return;
        };
        let Some(p) = self.bound_pipeline(compute) else {
            return;
        };
        for stage in &p.stages {
            let hash = stage.shader.hash.to_string();
            if !hashes.contains(&hash) {
                continue;
            }
            let k = {
                let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                if seen.0 != frame {
                    *seen = (frame, Vec::new());
                }
                let k = seen.1.iter().filter(|(h, _)| *h == hash).count() as u32;
                seen.1.push((hash.clone(), k));
                k
            };
            let t = stage.translated();
            for u in &t.used {
                let (RootType::Table, Some(Root::Table(table))) =
                    (u.root, self.roots_ref(compute).get(usize::from(u.slot)))
                else {
                    continue;
                };
                let handle = table + u64::from(u.table_offset_dwords) * 4;
                let Some(view) = crate::descriptor::view(handle) else {
                    continue;
                };
                let range = match &view {
                    View::Constant(address, size) => Some((*address, u64::from(*size))),
                    v => crate::descriptor::host::buffer_range(v),
                };
                let Some((address, len)) = range else {
                    continue;
                };
                let mut bytes = vec![0u8; len.min(4 << 20) as usize];
                if !crate::host::read_into(address as *const u8, &mut bytes) {
                    tracing::warn!(
                        address = format_args!("{address:#x}"),
                        len,
                        "dump: unreadable"
                    );
                    continue;
                }
                let name = format!(
                    "frame-{frame}-{hash}-{k}-s{}+{}-{address:x}.bin",
                    u.slot, u.table_offset_dwords
                );
                if let Err(e) = std::fs::write(folder.join(&name), &bytes) {
                    tracing::warn!(%name, "dump not written: {e}");
                }
            }
        }
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let prepared = self.prepare(list, false);
        let rects = prepared && self.needs_rect_expansion();
        let ran = prepared && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers (their
            // count checked to fit by `rectangles`).
            let (count, from) = if rects {
                (expanded_vertices(vertices).unwrap_or(0), 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            if let Some(p) = self.bound_pipeline(false) {
                self.note_gather_writes(list, &p, false);
            }
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        self.frame_op("draw", false, ran);
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn needs_rect_expansion(&self) -> bool {
        self.topology == RECTANGLES && self.bound_pipeline(false).is_some_and(|p| !p.rects)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = self.bound_pipeline(false) else {
            return false;
        };
        // The draw runs `count / 3 * 6` vertices, a count the host takes in
        // 32 bits.
        if expanded_vertices(count).is_none() {
            once!(count, "too many rectangle vertices");
            return false;
        }
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let source = rect_source(first, count, v.stride, v.size)
                .filter(|_| v.address != 0)
                .map(|offset| v.address + offset)
                .filter(|&from| crate::shadow::find(from, rects * 3 * stride).is_some());
            let Some(from) = source else {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            };
            // The expanded vertices' view takes its size in 32 bits.
            let Ok(span) = u32::try_from(rects * 6 * stride) else {
                once!(slot, "rectangle vertices too many to expand");
                return false;
            };
            let Some((gpu, to)) = list.scratch(u64::from(span)) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host_read(from as *const u8, n * 3 * stride, move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded), the host read holds `title_memory_reading` over
                // `[from, from + n * 3 * stride)`, and the scratch memory is
                // the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: span,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        let prepared = self.prepare(list, true);
        let needs_expansion = prepared && self.needs_rect_expansion();
        if needs_expansion {
            once!("indexed rectangles not drawn");
        }
        let ran = prepared && !needs_expansion;
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            if let Some(p) = self.bound_pipeline(false) {
                self.note_gather_writes(list, &p, false);
            }
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        self.frame_op("draw_indexed", false, ran);
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host, with the reason in
    /// `dropped` (the caller logs it once per signature).
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
        dropped: &mut &'static str,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            *dropped = "skipped shader";
            crate::pipeline::note_drop(self.pipeline as This);
            self.frame_op(&format!("indirect kind {kind}"), kind == 2, false);
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        // Without mesh shaders a mesh dispatch is a compute pass and a draw
        // of one instance per thread group (`execute_indirect_mesh`).
        // From the pipeline bound for the strides in use now, as a direct
        // mesh dispatch takes it.
        let bound_mesh = (kind == MESH).then(|| self.bound_pipeline(false)).flatten();
        let without_mesh_shaders = bound_mesh
            .as_deref()
            .and_then(|p| p.fallback.as_ref().map(|f| (p, f)));
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            *dropped = "arguments outside title buffers";
            why("arguments outside title buffers");
            #[cfg(feature = "diagnostics")]
            self.frame_op_why(
                &format!("indirect kind {kind}"),
                kind == DISPATCH,
                false,
                Some("arguments outside title buffers"),
            );
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    *dropped = "count outside title buffers";
                    why("count outside title buffers");
                    #[cfg(feature = "diagnostics")]
                    self.frame_op_why(
                        &format!("indirect kind {kind}"),
                        kind == DISPATCH,
                        false,
                        Some("count outside title buffers"),
                    );
                    return false;
                }
            },
            None => None,
        };
        // Pipeline creation can still be in flight. Select the rectangle path
        // only from a finished pipeline, whose geometry stage is now known.
        let fallback = if matches!(kind, DRAW | INDEXED) && self.topology == RECTANGLES {
            let Some(p) = self.bound_pipeline(false) else {
                *dropped = "no host pipeline for the rectangle path";
                #[cfg(feature = "diagnostics")]
                crate::pipeline::note_drop(self.pipeline as This);
                return false;
            };
            (!p.rects).then_some(p)
        } else {
            None
        };
        if let Some(p) = fallback {
            if kind == INDEXED {
                once!(shaders = %p.name, "indirect indexed rectangles not drawn");
                *dropped = "indexed rectangles not supported";
                why("indexed rectangles not supported");
                return false;
            }
            if p.compute || p.mesh {
                *dropped = "pipeline mismatch";
                why(&format!(
                    "pipeline mismatch (compute {}, mesh {}, topology {})",
                    p.compute, p.mesh, self.topology
                ));
                return false;
            }
            let Some(signature) = host_signature(DRAW, 16) else {
                *dropped = "no host signature";
                why("no host signature");
                return false;
            };
            if max == 0 {
                DRAWN.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            let prepared = match crate::rectangles::expand(
                list,
                &p,
                &self.vertex,
                (&args, arguments, stride, max),
                counter.as_ref().map(|(s, at)| (s, *at)),
            ) {
                Ok(p) => p,
                Err(reason) => {
                    *dropped = "rectangle expansion failed";
                    why(reason);
                    return false;
                }
            };
            self.host_pipeline = 0;
            self.host_roots[1] = 0;
            if !self.prepare(list, false) {
                *dropped = "prepare failed for indirect rectangles";
                why("prepare failed for indirect rectangles");
                return false;
            }
            prepared.before_draw(list);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| {
                self.op_text(
                    format!("indirect rect draw max {max} at {arguments:#x}"),
                    false,
                )
            });
            // SAFETY: an open list with everything set; `prepared.args_buf` is
            // list-owned, `counter` is kept alive by refresh, signature cached.
            unsafe {
                list.list.ExecuteIndirect(
                    &signature,
                    max,
                    &prepared.args_buf,
                    0,
                    counter.as_ref().map(|(s, _)| &s.buffer),
                    counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
                )
            };
            self.window = None;
            self.note_gather_writes(list, &p, false);
            prepared.after_draw(list);
            DRAWN.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        if let Some((p, f)) = without_mesh_shaders {
            // `VICTORY_SKIP_INDIRECT_MESH=1`: drop indirect mesh dispatches on
            // hosts without mesh shaders instead of running the fallback.
            static SKIP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *SKIP.get_or_init(|| {
                std::env::var_os("VICTORY_SKIP_INDIRECT_MESH").is_some_and(|v| v != "0")
            }) {
                *dropped = "indirect mesh dispatch skipped (VICTORY_SKIP_INDIRECT_MESH)";
                return false;
            }
            once!(
                records = max,
                counted = counter.is_some(),
                "indirect mesh dispatches run as a compute pass and a draw (the first of each)"
            );
            let ran = self.execute_indirect_mesh(
                list,
                (p, f, u32::MAX),
                (&args, arguments, stride, max),
                counter.as_ref().map(|(s, at)| (s, *at)),
            );
            #[cfg(feature = "diagnostics")]
            self.frame_op("indirect mesh as compute and draw", false, ran);
            if ran {
                // The compute pass may have stored to an argument buffer a
                // later indirect draw reads through the window.
                self.window = None;
                self.note_gather_writes(list, p, false);
                DRAWN.fetch_add(1, Ordering::Relaxed);
                MESH_WITHOUT_MESH_SHADERS[1].fetch_add(1, Ordering::Relaxed);
            } else {
                why("indirect mesh dispatch without mesh shaders not recorded");
            }
            return ran;
        }
        let Some(p) = self.bound_pipeline(kind == DISPATCH) else {
            *dropped = "no host pipeline";
            #[cfg(feature = "diagnostics")]
            {
                self.frame_op("indirect", kind == DISPATCH, false);
                crate::pipeline::note_why(format!(
                    "indirect kind {kind}: {}",
                    crate::pipeline::drop_reason(self.pipeline as This)
                ));
            }
            return false;
        };
        if (kind == DISPATCH && !p.compute)
            || (kind != DISPATCH && (p.compute || p.mesh != (kind == MESH)))
        {
            *dropped = if kind == DISPATCH {
                "prepare/bind failed (compute)"
            } else {
                "prepare/bind failed"
            };
            #[cfg(feature = "diagnostics")]
            {
                self.frame_op("indirect", kind == DISPATCH, false);
                why(&format!(
                    "prepare failed (compute {}, mesh {})",
                    p.compute, p.mesh
                ));
            }
            return false;
        }
        // `VICTORY_INDIRECT_AMPLIFY` (on by default; `=0` disables): the record
        // is read, bounded and dispatched by an amplification stage of ours on
        // the GPU (`crate::pipeline::host::amplifier`), drawn directly via
        // `DispatchMesh(1, 1, 1)`.
        if kind == MESH
            && max == 1
            && count.is_none()
            && let Some((amplified, buffer_at, offset_at)) = &p.amplify
        {
            const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
            // SAFETY: a live buffer.
            let (flags, base) = unsafe {
                (
                    args.buffer.GetDesc().Flags,
                    args.buffer.GetGPUVirtualAddress(),
                )
            };
            let offset = args.offset(arguments);
            static AMPLIFIED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
            let fits = flags.0 & UNORDERED != 0
                && offset.is_multiple_of(4)
                && offset <= u64::from(u32::MAX);
            let n = AMPLIFIED[usize::from(!fits)].fetch_add(1, Ordering::Relaxed) + 1;
            if fits && n.is_power_of_two() && n >= 1 << 12 {
                tracing::info!(
                    amplified = n,
                    left = AMPLIFIED[1].load(Ordering::Relaxed),
                    "one-record indirect mesh draws through the amplification stage (VICTORY_INDIRECT_AMPLIFY=0 disables); left: a buffer shaders cannot write, or an odd offset"
                );
            }
            if fits {
                self.window = None;
                if !self.prepare_for(list, false, true) {
                    *dropped = "prepare/bind failed";
                    #[cfg(feature = "diagnostics")]
                    self.frame_op("indirect", false, false);
                    return false;
                }
                // The amplification stage reads `args.buffer` via a root UAV
                // (`RWByteAddressBuffer`), so track it in UNORDERED_ACCESS state.
                args.refresh_as(list, arguments, len, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                // `VICTORY_AMPLIFY_WAIT` (`1` default, `0` none, `auto` only
                // after a storing shader or on the list's first amplified draw).
                static WAIT: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
                let wait =
                    *WAIT.get_or_init(|| match std::env::var("VICTORY_AMPLIFY_WAIT").as_deref() {
                        Ok("0") => 0,
                        Ok("auto") => 2,
                        _ => 1,
                    });
                const WAIT_USE: usize = usize::MAX - 0x78;
                let first = list.first_use(WAIT_USE);
                if wait == 1 || (wait == 2 && (self.wrote || first)) {
                    crate::indirect::wait_direct(list);
                }
                self.wrote = false;
                #[cfg(feature = "diagnostics")]
                self.frame_op("indirect", false, true);
                #[cfg(feature = "diagnostics")]
                list.note_op(|| self.op_text(format!("amplified at {arguments:#x}"), false));
                if let Ok(l6) = list.list.cast::<ID3D12GraphicsCommandList6>() {
                    // SAFETY: an open list with the pipeline and root signature
                    // set by `prepare_for`; `args.buffer` is kept and in UAV state.
                    unsafe {
                        list.list.SetPipelineState(amplified);
                        self.host_pipeline = amplified.as_raw() as usize;
                        list.keep(amplified);
                        list.list
                            .SetGraphicsRootUnorderedAccessView(*buffer_at, base);
                        list.list
                            .SetGraphicsRoot32BitConstant(*offset_at, offset as u32, 0);
                        l6.DispatchMesh(1, 1, 1);
                    }
                    self.note_gather_writes(list, &p, false);
                }
                DRAWN.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            *dropped = "no host signature";
            why("no host signature");
            #[cfg(feature = "diagnostics")]
            self.frame_op_why(
                &format!("indirect kind {kind}"),
                kind == DISPATCH,
                false,
                Some("no host signature"),
            );
            return false;
        };
        let offset = args.offset(arguments);
        // Opt-in hoisted clamps (`VICTORY_HOIST_CLAMPS`): draws of a run are
        // clamped in the list before, which the run's first draw opened
        // (`crate::list::hoist`), so nothing of ours comes between the draws.
        // Off by default (`VICTORY_INDIRECT_AMPLIFY`, `VICTORY_INDIRECT_GATHER`,
        // and windowed clamping remain untouched).
        let hoist_allowed = kind != DISPATCH
            && match crate::indirect::hoist_mode() {
                0 | 6 => false,
                3 => kind == MESH,
                4 => kind != MESH,
                _ => true,
            }
            && !(matches!(kind, DRAW | INDEXED) && self.topology == RECTANGLES)
            && self.hoists_into(list);
        let before = if hoist_allowed {
            self.hoist.as_ref().and_then(|h| {
                args.refresh(&h.clamps, arguments, len);
                if let Some((s, at)) = &counter {
                    s.refresh(&h.clamps, *at, 4);
                }
                let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                #[cfg(feature = "diagnostics")]
                h.clamps.note_op(|| format!("hoisted clamp of kind {kind}"));
                let (bounded, bounded_at) = crate::indirect::bound_before(
                    &h.clamps,
                    (&args.buffer, offset),
                    count_arg,
                    (kind, stride, max),
                )?;
                list.keep_hoisted(&bounded);
                Some((bounded, bounded_at))
            })
        } else {
            None
        };
        let inline = before.is_none();
        // Whether `crate::indirect::done` must transition `bounded` back from
        // `INDIRECT_ARGUMENT` to `COMMON` after `ExecuteIndirect` (false for
        // hoisted clamps and for gathered scratch buffers, which `HostList::close`
        // transitions once at list close).
        let mut needs_done = inline;
        // Whether `bounded` is ALREADY in `INDIRECT_ARGUMENT` before
        // `ExecuteIndirect` runs (true only after an inline clamp; false for
        // window hits, whose scratch is in `COMMON` until `ExecuteIndirect`
        // implicitly promotes it, and for hoisted/gathered buffers).
        let mut in_indirect = inline;
        let (bounded, at) = match before {
            Some(hoisted) => {
                self.window = None;
                hoisted
            }
            None => {
                const WINDOW_USE: usize = usize::MAX - 0x77;
                let windowed = windows()
                    && kind != DISPATCH
                    && max == 1
                    && count.is_none()
                    && !p
                        .stages
                        .iter()
                        .chain(p.fallback.as_ref().map(|f| &f.stage))
                        .any(|stage| stage.translated().stores);
                if !windowed || list.first_use(WINDOW_USE) {
                    self.window = None;
                }
                let same = |w: &Window| w.copy == args.id() && w.kind == kind && w.stride == stride;
                let hit = self
                    .window
                    .as_ref()
                    .filter(|w| same(w) && offset >= w.base)
                    .filter(|w| {
                        (offset - w.base) % u64::from(w.step) == 0
                            && (offset - w.base) / u64::from(w.step) < u64::from(w.records)
                    })
                    .map(|w| {
                        let n = (offset - w.base) / u64::from(w.step);
                        (w.scratch.clone(), n * u64::from(w.step))
                    });
                // A record that misses its window: a slot the list's prelude
                // fills, if the list still gathers for this buffer ([`gathers`]).
                let gathered = if hit.is_none() && windowed && gathers() {
                    let slot = crate::indirect::gather(
                        list,
                        (&args.buffer, args.id()),
                        offset,
                        (kind, stride),
                    );
                    let n = GATHER_STATS[usize::from(slot.is_none())]
                        .fetch_add(1, Ordering::Relaxed)
                        + 1;
                    if slot.is_some() && n.is_power_of_two() && n >= 1 << 12 {
                        let g = crate::indirect::GATHER_NONE
                            .each_ref()
                            .map(|c| c.load(Ordering::Relaxed));
                        tracing::info!(
                            gathered = n,
                            missed = GATHER_STATS[1].load(Ordering::Relaxed),
                            no_fit = g[0],
                            closed = g[1],
                            too_many_buffers = g[2],
                            set_full = g[3],
                            no_buffer = g[4],
                            "scattered indirect records gathered for list preludes (VICTORY_INDIRECT_GATHER=0 disables)"
                        );
                    }
                    slot
                } else {
                    None
                };
                match (hit, gathered) {
                    (Some(hit), _) => {
                        in_indirect = false;
                        hit
                    }
                    (None, Some((scratch, at))) => {
                        // Refresh title bytes before the prelude runs without
                        // forcing a redundant state transition on the main list.
                        args.refresh_for_prelude(list, arguments, len);
                        needs_done = false;
                        in_indirect = false;
                        (scratch, at)
                    }
                    (None, None) => {
                        let (step, records) = match &self.window {
                            Some(w) if windowed && same(w) => {
                                let last = w.base + u64::from(w.records - 1) * u64::from(w.step);
                                let step = offset.saturating_sub(last);
                                let walks = step >= u64::from(stride)
                                    && step <= 256
                                    && step.is_multiple_of(4)
                                    && (w.records == 1 || step == u64::from(w.step));
                                if walks {
                                    let fit = (args.left(arguments) / step).min(WINDOW_MOST) as u32;
                                    (
                                        step as u32,
                                        (if w.records == 1 { 16 } else { w.records * 2 })
                                            .min(fit)
                                            .max(1),
                                    )
                                } else {
                                    (stride, max)
                                }
                            }
                            _ => (stride, max),
                        };
                        args.refresh(
                            list,
                            arguments,
                            len.max(u64::from(step) * u64::from(records)),
                        );
                        if let Some((s, at)) = &counter {
                            s.refresh(list, *at, 4);
                        }
                        let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                        let bounded = if windowed && records > 1 {
                            crate::indirect::bound_ahead(
                                list,
                                (&args.buffer, offset),
                                (kind, step, records),
                            )
                        } else {
                            crate::indirect::bound(
                                list,
                                (&args.buffer, offset),
                                count_arg,
                                (kind, step, records),
                            )
                        };
                        let Some(bounded) = bounded else {
                            self.window = None;
                            *dropped = "not bounded (no clamp, or arguments not 4-byte aligned)";
                            why("not bounded (no clamp, or arguments not 4-byte aligned)");
                            #[cfg(feature = "diagnostics")]
                            self.frame_op_why(
                                &format!("indirect kind {kind}"),
                                kind == DISPATCH,
                                false,
                                Some("not bounded (no clamp, or arguments not 4-byte aligned)"),
                            );
                            return false;
                        };
                        self.host_pipeline = 0;
                        self.host_roots[1] = 0;
                        self.window = windowed.then(|| Window {
                            copy: args.id(),
                            kind,
                            stride,
                            step,
                            base: offset,
                            records,
                            scratch: bounded.clone(),
                        });
                        (bounded, 0)
                    }
                }
            }
        };
        if !inline && let Some((s, at)) = &counter {
            s.refresh_as(list, *at, 4, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        }
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => self.bind(list, &p, true),
            _ => false,
        };
        #[cfg(feature = "diagnostics")]
        self.frame_op("indirect", kind == DISPATCH, ran);
        if !ran {
            if in_indirect {
                crate::indirect::done(list, &bounded);
            }
            *dropped = if kind == DISPATCH {
                "prepare/bind failed (compute)"
            } else {
                "prepare/bind failed"
            };
            #[cfg(feature = "diagnostics")]
            why(&format!(
                "prepare failed (compute {}, mesh {})",
                p.compute, p.mesh
            ));
            return false;
        }
        self.note_gather_writes(list, &p, kind == DISPATCH);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(
                format!("indirect kind {kind} max {max} at {arguments:#x}"),
                kind == DISPATCH,
            )
        });
        // Binding may have moved the count's buffer back to a shader state
        // (a pass that reads or writes the buffer that holds it);
        // ExecuteIndirect needs it as arguments.
        if let Some((s, _)) = &counter {
            list.use_shadow(&s.buffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        }
        if !inline && crate::indirect::hoist_mode() == 2 {
            // SAFETY: an open list; the barrier names no resource.
            unsafe { list.list.ResourceBarrier(&[crate::indirect::uav_barrier()]) };
        }
        // SAFETY: an open list with everything set; the scratch buffer and
        // host copies are kept alive by the list, and the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &bounded,
                at,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        if needs_done {
            crate::indirect::done(list, &bounded);
        }
        if kind == DISPATCH {
            list.count_write();
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// A few things, up to `N` of them without heap allocation: used while
/// walking a draw's descriptor tables and targets.
struct Few<T, const N: usize> {
    few: [T; N],
    len: usize,
    /// All of them, if they exceed `N`.
    many: Vec<T>,
}

impl<T: Copy + Default, const N: usize> Default for Few<T, N> {
    fn default() -> Self {
        Few {
            few: [T::default(); N],
            len: 0,
            many: Vec::new(),
        }
    }
}

impl<T: Copy, const N: usize> Few<T, N> {
    fn push(&mut self, what: T) {
        if self.many.is_empty() && self.len < N {
            self.few[self.len] = what;
            self.len += 1;
        } else {
            if self.many.is_empty() {
                self.many.extend_from_slice(&self.few);
            }
            self.many.push(what);
        }
    }
}

impl<T, const N: usize> std::ops::Deref for Few<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        if self.many.is_empty() {
            &self.few[..self.len]
        } else {
            &self.many
        }
    }
}

impl<T, const N: usize> std::ops::DerefMut for Few<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        if self.many.is_empty() {
            &mut self.few[..self.len]
        } else {
            &mut self.many
        }
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// The tracker uses whole resources, so both planes must be read-only
/// when the depth format carries stencil.
fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format == DXGI_FORMAT_UNKNOWN {
        // SAFETY: a live texture; a default format inherits its format.
        unsafe { texture.GetDesc() }.Format
    } else {
        desc.Format
    };
    let mut required = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if format != DXGI_FORMAT_D16_UNORM && format != DXGI_FORMAT_D32_FLOAT {
        required |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & required == required {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need. Returns whether the view
/// also reads the draw's bound depth target.
/// The title memory a buffer or constant view reads, brought up to date in
/// its host copy for a draw or dispatch ([`readable`]). A constant view at a
/// title slot (not one a fresh table gives scratch memory to) is made on
/// that host copy, so it is refreshed like a buffer view; without this its
/// host copy kept whatever an earlier refresh left (zeros for one nothing
/// else reads).
fn read_range(view: &View) -> Option<(u64, u64)> {
    match *view {
        View::Constant(0, _) => None,
        View::Constant(address, size) => Some((address, u64::from(size))),
        _ => crate::descriptor::host::buffer_range(view),
    }
}

fn readable(
    list: &HostList,
    u: &UsedResource,
    view: &View,
    read_only_depth: Option<usize>,
) -> bool {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let first_mip = crate::descriptor::host::placed_first_mip(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((address, len)) = read_range(view)
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        if u.kind == ResourceType::Uav {
            shadow.refresh_as(list, address, len, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            shadow.gpu_writes(list, address, len);
        } else {
            shadow.refresh(list, address, len);
        }
        // A shader view off its buffer copy's elements reads its own copy of
        // just its bytes: that one is a host copy like any other, so its own
        // refresh brings it up to date from title memory (which the last
        // submission's copy-back already wrote).
        if u.kind == ResourceType::Srv
            && let Some(aligned) = crate::shadow::aligned_existing(address, len)
        {
            aligned.refresh(list, address, len);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            // A view the title placed, in another format, in the memory of
            // a texture the host writes: that texture's texels, copied on
            // the host (title memory does not have them).
            let aliased = crate::resource::alias_of(*resource as This)
                .is_some_and(|source| crate::texture::alias(list, *resource as This, source));
            if !aliased
                && let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::reads_title_memory(*resource as This, t)
            {
                crate::texture::upload_from(list, *resource as This, t, first_mip);
            }
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(*resource as This, false, false);
            // Keep the same read bits the depth target will request at draw time.
            let samples_depth = read_only_depth == Some(*resource);
            let state = if samples_depth {
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            crate::list::ensure_state(list, *resource as This, state);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                // Trace, not debug: in the world it fires for most reads of
                // most draws and buries the per-draw lines a capture needs.
                if d.Flags.0 & 7 == 0 {
                    tracing::trace!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
            samples_depth
        }
        (View::Unordered(resource, _, desc), ResourceType::Uav) => {
            // A shader writes it, so its contents are on the host and not in
            // title memory (as for a copy destination): the upload a later
            // read brings textures the title fills up to date with would copy
            // the old title memory over what the shader wrote.
            if crate::resource::host_resource(*resource as This).is_some() {
                crate::texture::host_written(*resource as This);
            }
            #[cfg(feature = "diagnostics")]
            {
                crate::capture::note_target(*resource as This);
                crate::capture::note_op_write(*resource as This);
            }
            // Read by the title's CPU code: copied back into title memory.
            list.texture_written(*resource as This, desc.as_ref());
            // Unordered access to a depth texture is its colour twin's: the
            // write does not land in the depth texture, so it is not the
            // draw's depth target and is not marked as sampled.
            let twinned = list.twin_use(*resource as This, true, true);
            if !twinned {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            }
            false
        }
        _ => false,
    }
}

/// The shift from a heap's byte offset to its index (views are 32 bytes,
/// samplers 16).
#[cfg(feature = "diagnostics")]
fn shift_of(sampler: bool) -> u32 {
    if sampler { 4 } else { 5 }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!(
                "{}x{} f{} flags {:#x}",
                d.Width, d.Height, d.Format.0, d.Flags.0
            )
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || format!("uav buffer {:?}", crate::descriptor::host::buffer_range(v)),
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are. They
        // are read through the kernel, which fails on an unmapped page:
        // asking for the region first (`texture::committed`) walks its pages
        // and, once per draw, made world frames ten times slower.
        Some(View::Constant(address, size)) if *size >= 16 => {
            let mut bytes = [0u8; 16];
            if crate::host::read_into(*address as *const u8, &mut bytes) {
                let v: [f32; 4] =
                    std::array::from_fn(|i| f32::from_le_bytes(bytes.as_chunks::<4>().0[i]));
                format!("Constant({address:#x}, {size}) {v:?}")
            } else {
                format!("{:?}", View::Constant(*address, *size))
                    .chars()
                    .take(60)
                    .collect()
            }
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// The host vertices `vertices` rectangle-list vertices become (six per
/// rectangle), if the count fits in 32 bits.
fn expanded_vertices(vertices: u32) -> Option<u32> {
    (vertices / 3).checked_mul(6)
}

/// The byte offset of vertex `first` in a buffer of `size` bytes holding
/// `stride`-byte vertices, if vertices `first..first + count` all lie in
/// it. Computed wide: `first + count` and the byte offsets overflow 32
/// bits with the values a title may pass.
fn rect_source(first: u32, count: u32, stride: u32, size: u32) -> Option<u64> {
    let stride = u64::from(stride);
    let end = (u64::from(first) + u64::from(count)).checked_mul(stride)?;
    (stride != 0 && end <= u64::from(size)).then(|| u64::from(first) * stride)
}

#[cfg(test)]
mod tests {
    use super::{expanded_vertices, rect_source};

    /// A constant view read through a title slot names the title memory it
    /// reads, so its host copy is brought up to date; a null one names none.
    #[test]
    fn constant_views_name_the_memory_they_read() {
        use crate::descriptor::View;
        assert_eq!(
            super::read_range(&View::Constant(0x9_74df_0200, 832)),
            Some((0x9_74df_0200, 832))
        );
        assert_eq!(super::read_range(&View::Constant(0, 256)), None);
    }

    /// The pattern a run may have hit: an indirect mesh dispatch of many
    /// records on a host without mesh shaders, the records past the count
    /// holding an earlier frame's (65535, 1, 1). Charging all of them spends
    /// the list's budget and later indirect work runs empty; charging only
    /// the record the fallback runs leaves it.
    #[test]
    fn a_mesh_fallback_charges_only_the_record_it_runs() {
        use crate::indirect_budget::Budget;
        // A mesh group counts as 128 threads.
        let stale = 65_535 * 128;
        let later = 1 << 20;
        let charge = |records: u32| {
            let mut budget = Budget::default();
            for _ in 0..records {
                budget.admit(stale);
            }
            budget.admit(later)
        };
        assert!(!charge(1024), "every record charged: later work runs empty");
        assert_eq!(super::mesh_fallback_records(1024), 1);
        assert!(charge(super::mesh_fallback_records(1024)));
        assert_eq!(super::mesh_fallback_records(0), 0);
    }

    /// A slot found through a list's own copy of a root table is looked up
    /// in the buffer view table at the title's slot, whichever lies first.
    #[test]
    fn a_copied_tables_slots_map_back_to_the_titles() {
        use super::table_slots_ahead;
        // The title's own table: nothing to add.
        assert_eq!(table_slots_ahead(0x1240, None), 0);
        // What the shader computes: the slot of the byte it was given plus
        // the entry's own offset, then the correction.
        let slot = |given: u32, entry_bytes: u32, ahead: u32| {
            (given.wrapping_add(entry_bytes) >> 5).wrapping_add(ahead)
        };
        for (title, fresh) in [(0x1240u32, 0x0200_0000u32), (0x0200_0000, 0x40), (96, 96)] {
            let ahead = table_slots_ahead(title, Some(fresh));
            for entry_bytes in [0, 32, 40, 0x520] {
                assert_eq!(
                    slot(fresh, entry_bytes, ahead),
                    (title + entry_bytes) >> 5,
                    "{title:#x} {fresh:#x} {entry_bytes:#x}"
                );
            }
        }
    }

    /// A constant view's note shows its first four values when its memory
    /// is readable, and only the view when it is not (#357).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn constant_view_notes_read_their_values_or_fall_back() {
        use crate::descriptor::View;
        let values = [1.0f32, -0.0, 2.5, 4.0];
        let address = values.as_ptr() as u64;
        assert_eq!(
            super::describe(Some(&View::Constant(address, 16))),
            format!("Constant({address:#x}, 16) [1.0, -0.0, 2.5, 4.0]")
        );
        // Nothing is mapped at the first page.
        assert_eq!(
            super::describe(Some(&View::Constant(0x10, 16))),
            "Constant(16, 16)"
        );
    }

    #[test]
    fn writable_root_extent_covers_every_representable_whole_dword() {
        let max = u64::from(u32::MAX) * 4;
        assert_eq!(super::writable_root_bytes(3), Some(0));
        assert_eq!(
            super::writable_root_bytes((1 << 20) + 7),
            Some((1 << 20) + 4)
        );
        assert_eq!(super::writable_root_bytes(max + 3), Some(max));
        assert_eq!(super::writable_root_bytes(max + 4), None);
    }

    #[test]
    fn bounded_root_dispatch_writes_then_reads_without_a_title_heap() {
        use super::*;
        let Some(host) = crate::host::host() else {
            eprintln!("no host D3D12 device; skipped bounded root dispatch");
            return;
        };
        let pipeline = |source| {
            let bytes = crate::hlsl::compile_compute("raw root test", source).unwrap();
            // SAFETY: synthetic shader bytecode contains its matching root signature.
            unsafe {
                let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &bytes).unwrap();
                let pso: ID3D12PipelineState = host
                    .device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: bytes.as_ptr().cast(),
                            BytecodeLength: bytes.len(),
                        },
                        ..Default::default()
                    })
                    .unwrap();
                (root, pso)
            }
        };
        let first = pipeline(
            r#"RWByteAddressBuffer dst : register(u0);
                [RootSignature("DescriptorTable(UAV(u0))")]
                [numthreads(1,1,1)] void main() { dst.Store(1048580, dst.Load(1048580) + 22); }"#,
        );
        let second = pipeline(
            r#"RWByteAddressBuffer dst : register(u0);
                ByteAddressBuffer src : register(t0);
                [RootSignature("DescriptorTable(UAV(u0)), DescriptorTable(SRV(t0))")]
                [numthreads(1,1,1)] void main() { dst.Store(8, src.Load(1048580) + 5); }"#,
        );
        // One dword beyond the old one-MiB view.
        const WORD: usize = (1 << 20) / 4 + 1;
        struct Registration(u64);
        impl Drop for Registration {
            fn drop(&mut self) {
                crate::shadow::unregister(self.0 as usize, self.0);
            }
        }
        let register = |words: &mut [u32]| {
            let addr = words.as_mut_ptr() as u64;
            crate::shadow::register(addr as usize, addr, std::mem::size_of_val(words) as u64);
            Registration(addr)
        };
        let mut source = vec![7; WORD + 3];
        let mut destination = [0; 4];
        let src = register(&mut source);
        let dst = register(&mut destination);
        let list = host.new_list().unwrap();
        let mut state = State::default();
        assert!(state.bind_heaps(&list, true));
        assert_eq!(state.host_heaps, [crate::descriptor::ROOT_VIEW_HEAP, 0]);
        assert_eq!(state.host_heap_ids, [u64::MAX, 0]);
        assert_eq!(state.heaps, [0, 0]);
        // Same title keys and creation ids must not re-bind.
        let bound = (state.host_heaps, state.host_heap_ids);
        assert!(state.bind_heaps(&list, true));
        assert_eq!((state.host_heaps, state.host_heap_ids), bound);
        for ((root, pso), dst, src) in [(&first, src.0, None), (&second, dst.0, Some(src.0))] {
            // SAFETY: an open list and matching live root signature and pipeline.
            unsafe {
                list.list.SetComputeRootSignature(root);
                list.list.SetPipelineState(pso);
            }
            assert!(state.root_buffer_table(&list, Some(dst), 0, true, RawAccess::ReadWrite));
            if let Some(src) = src {
                assert!(state.root_buffer_table(&list, Some(src), 1, true, RawAccess::Read));
            }
            // SAFETY: a single workgroup with the required tables bound.
            unsafe { list.list.Dispatch(1, 1, 1) };
        }
        assert!(state.bind_heaps(&list, false));
        assert_eq!(state.host_heaps, [0, 0]);
        list.close();
        host.execute(&[&list]);
        assert_eq!(source[WORD], 29);
        assert!(
            source[..WORD]
                .iter()
                .chain(&source[WORD + 1..])
                .all(|&word| word == 7)
        );
        assert_eq!(destination, [0, 0, 34, 0]);
        eprintln!("bounded root UAV to SRV dispatch readback verified");
    }

    #[test]
    fn bounded_root_slots_belong_to_the_recording_that_allocated_them() {
        let mut state = super::State::default();
        let heap = 1;
        state.fresh_records.push([0; 3]);
        state.fresh_tables.insert(
            11,
            super::FreshTableEntry {
                heap,
                copy_gen: 0,
                slots: 1,
                records_at: 0,
                first: 4,
                dead_slots: Vec::new(),
                aliases: Vec::new(),
                strides: Vec::new(),
            },
        );
        state
            .root_view_slots
            .insert((heap, super::RawAccess::Read, None), 5);
        state
            .root_view_slots
            .insert((heap, super::RawAccess::ReadWrite, None), 6);
        state
            .dynamic
            .extend([(heap, 4, 1), (heap, 5, 1), (heap, 6, 1)]);

        assert_eq!(state.root_view_slots.len(), 2);
        assert_eq!(
            state.take_dynamic(),
            [(heap, 4, 1), (heap, 5, 1), (heap, 6, 1)]
        );
        assert!(state.fresh_tables.is_empty());
        assert!(state.fresh_records.is_empty());
        assert!(state.root_view_slots.is_empty());
        assert!(state.take_dynamic().is_empty());
    }

    #[test]
    fn slots_bitset_iterates_in_ascending_order_across_all_words() {
        let mut slots = super::Slots::default();
        for s in [255u8, 64, 0, 129, 63, 64, 192] {
            slots.insert(s);
        }
        assert!(slots.contains(0));
        assert!(slots.contains(63));
        assert!(slots.contains(64));
        assert!(slots.contains(129));
        assert!(slots.contains(192));
        assert!(slots.contains(255));
        assert!(!slots.contains(1));
        assert!(!slots.contains(128));
        assert_eq!(
            slots.iter().collect::<Vec<_>>(),
            vec![0, 63, 64, 129, 192, 255]
        );
    }

    #[test]
    fn table_hash_and_record_compare_detect_record_and_override_changes() {
        #[repr(C, align(32))]
        struct RawTable([[u32; 8]; 2]);

        let mut raw = RawTable([[0; 8]; 2]);
        raw.0[0][0..3].copy_from_slice(&[0x1111_2222, 0x3333_4444, 0x5555_6666]);
        raw.0[1][0..3].copy_from_slice(&[0x7777_8888, 0x9999_aaaa, 0xbbbb_cccc]);
        let table = std::ptr::from_ref(&raw) as u64;

        let r0 = super::record(table, 0);
        let r1 = super::record(table, 1);
        assert_eq!(r0, [0x1111_2222, 0x3333_4444, 0x5555_6666]);
        assert_eq!(r1, [0x7777_8888, 0x9999_aaaa, 0xbbbb_cccc]);

        let raws = [r0, r1];
        let alive = [false, false];
        let dead = [false, true];
        let alias = [super::StructuredBinding {
            slot: 1,
            entry: 1,
            stride: 64,
        }];
        let h_plain = super::table_hash(0x1000, 0, &raws, &alive, 0, 8, &[], &[]);
        let h_dead = super::table_hash(0x1000, 0, &raws, &dead, 0, 8, &[], &[]);
        let h_alias = super::table_hash(0x1000, 0, &raws, &alive, 0, 8, &alias, &[]);
        let h_gen = super::table_hash(0x1000, 1, &raws, &alive, 0, 8, &[], &[]);
        assert_ne!(h_plain, h_dead);
        assert_ne!(h_plain, h_alias);
        assert_ne!(h_plain, h_gen);

        // Mutating an entry word changes both the exact record and the quick hash.
        raw.0[1][2] ^= 1;
        let table = std::ptr::from_ref(&raw) as u64;
        let raws_mut = [super::record(table, 0), super::record(table, 1)];
        assert_ne!(raws_mut[1], r1);
        assert_ne!(
            super::table_hash(0x1000, 0, &raws_mut, &alive, 0, 8, &[], &[]),
            h_plain
        );
    }

    #[test]
    fn rectangle_vertices_inside_their_buffer() {
        assert_eq!(rect_source(0, 3, 16, 48), Some(0));
        assert_eq!(rect_source(3, 3, 16, 96), Some(48));
        assert_eq!(rect_source(3, 3, 16, 95), None);
        assert_eq!(rect_source(0, 3, 0, 48), None);
    }

    /// `first + count` past 32 bits wrapped to a small count before, which
    /// passed the bounds check and read far outside the buffer.
    #[test]
    fn rectangle_bounds_do_not_wrap() {
        assert_eq!(rect_source(u32::MAX, 3, 16, u32::MAX), None);
        assert_eq!(rect_source(u32::MAX - 1, 3, 1, u32::MAX), None);
        assert_eq!(rect_source(u32::MAX, u32::MAX, u32::MAX, u32::MAX), None);
    }

    #[test]
    fn expanded_counts_fit_32_bits() {
        assert_eq!(expanded_vertices(3), Some(6));
        assert_eq!(expanded_vertices(7), Some(12));
        assert_eq!(expanded_vertices(u32::MAX / 6 * 3), Some(u32::MAX / 6 * 6));
        assert_eq!(expanded_vertices(u32::MAX), None);
    }
}

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

/// Fully initialized `[u32; 10]` encoding of a 2D `D3D12_SHADER_RESOURCE_VIEW_DESC`,
/// avoiding uninitialized struct padding or union tail bytes when `create_srv`
/// reads the descriptor as `[u32; 10]`.
#[cfg(test)]
fn tex2d_srv_words(format: DXGI_FORMAT, mip_levels: u32) -> [u32; 10] {
    const {
        assert!(size_of::<D3D12_SHADER_RESOURCE_VIEW_DESC>() == size_of::<[u32; 10]>());
        assert!(std::mem::offset_of!(D3D12_SHADER_RESOURCE_VIEW_DESC, Format) == 0);
        assert!(std::mem::offset_of!(D3D12_SHADER_RESOURCE_VIEW_DESC, ViewDimension) == 4);
        assert!(
            std::mem::offset_of!(D3D12_SHADER_RESOURCE_VIEW_DESC, Shader4ComponentMapping) == 8
        );
        assert!(std::mem::offset_of!(D3D12_SHADER_RESOURCE_VIEW_DESC, Anonymous) == 16);
    }
    [
        format.0 as u32,
        D3D12_SRV_DIMENSION_TEXTURE2D.0 as u32,
        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        0,          // padding before 8-byte-aligned union
        0,          // Texture2D.MostDetailedMip
        mip_levels, // Texture2D.MipLevels
        0,          // Texture2D.PlaneSlice
        0,          // Texture2D.ResourceMinLODClamp (0.0f)
        0,          // unused union tail
        0,          // unused union tail
    ]
}

#[cfg(test)]
mod depth_tests {
    use super::*;

    #[test]
    fn shader_table_keeps_read_only_depth_state() {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_SAMPLE_DESC,
        };

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        };
        let resource_iid = ID3D12Resource::IID;
        let mut depth = null;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const heap).cast(),
                0,
                (&raw const desc).cast(),
                D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32,
                null.cast(),
                (&raw const resource_iid).cast(),
                &mut depth,
            ),
            crate::object::S_OK
        );
        #[repr(C)]
        struct ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC,
            placement: [u64; 2],
        }
        let desc = ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            placement: [0; 2],
        };
        let heap_iid = ID3D12DescriptorHeap::IID;
        let mut views = null;
        assert_eq!(
            crate::descriptor::create_heap(
                null,
                (&raw const desc).cast(),
                (&raw const heap_iid).cast(),
                &mut views,
            ),
            crate::object::S_OK
        );
        let srv_desc = tex2d_srv_words(DXGI_FORMAT_R32_FLOAT, 1);
        let srv = crate::descriptor::heap(views).start;
        crate::descriptor::create_srv(null, depth, &srv_desc, srv);
        let used = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };

        let shader_resource = (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
            .0 as u32;
        let combined = D3D12_RESOURCE_STATE_DEPTH_READ.0 as u32 | shader_resource;
        for (flags, unsampled_state, shader_state, draw_state) in [
            (
                D3D12_DSV_FLAG_READ_ONLY_DEPTH,
                DEPTH_READ,
                combined,
                combined,
            ),
            (
                D3D12_DSV_FLAG_NONE,
                DEPTH_WRITE,
                shader_resource,
                DEPTH_WRITE,
            ),
        ] {
            let list = host.new_list().expect("host list");
            let desc = [
                DXGI_FORMAT_D32_FLOAT.0 as u32,
                D3D12_DSV_DIMENSION_TEXTURE2D.0 as u32,
                flags.0 as u32,
                0,
                0,
                0,
            ];
            let mut slot = [0u64; 16];
            let dsv = slot.as_mut_ptr() as u64;
            crate::descriptor::create_dsv(null, depth, &desc, dsv);
            let mut state = State::default();
            state.set_targets(&list, &[], dsv);
            state.set_heaps(&[views]);
            state.set_table(false, 0, srv);
            crate::list::ensure_state(&list, depth, D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32);
            state.ensure_depth_state(&list);
            assert!(
                list.transition(depth, unsampled_state).is_none(),
                "unsampled depth state for {flags:?}"
            );
            state.tables(&list, &[used], false, &[]);
            assert!(
                list.transition(depth, shader_state).is_none(),
                "shader-table state for {flags:?}"
            );
            state.ensure_depth_state(&list);
            assert!(
                list.transition(depth, draw_state).is_none(),
                "depth state for {flags:?}"
            );
            list.close();
            host.execute(&[&list]);
        }
        let release = |object: This| {
            crate::object::slot::<extern "system" fn(This) -> u32>(object, 0x10)(object)
        };
        assert_eq!(release(views), 0);
        assert_eq!(release(depth), 0);
    }

    #[test]
    fn reapply_host_preserves_state_clears_fresh_tables_and_keeps_dynamic_on_prev() {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };

        let Some(host) = crate::host::host() else {
            return;
        };
        let prev_list = host.new_list().expect("prev list");
        let new_list = host.new_list().expect("new list");
        let null = std::ptr::null();
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let make_tex = |format, flags, initial_state| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 4,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: flags,
                ..Default::default()
            };
            let iid = ID3D12Resource::IID;
            let mut res = null;
            assert_eq!(
                crate::resource::create_committed(
                    null,
                    (&raw const heap).cast(),
                    0,
                    (&raw const desc).cast(),
                    initial_state,
                    null.cast(),
                    (&raw const iid).cast(),
                    &mut res,
                ),
                crate::object::S_OK
            );
            res
        };
        let rt = make_tex(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
        );
        let depth = make_tex(
            DXGI_FORMAT_D32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32,
        );

        let mut rtv_slot = [0u64; 16];
        let rtv_handle = rtv_slot.as_mut_ptr() as u64;
        crate::descriptor::create_rtv(null, rt, null.cast(), rtv_handle);
        let mut dsv_slot = [0u64; 16];
        let dsv_handle = dsv_slot.as_mut_ptr() as u64;
        crate::descriptor::create_dsv(null, depth, null.cast(), dsv_handle);

        // Create a sampler heap so dynamic slots are constrained and measurable.
        #[repr(C)]
        struct ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC,
            placement: [u64; 2],
        }
        let desc = ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                NumDescriptors: 4096,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            placement: [0; 2],
        };
        let heap_iid = ID3D12DescriptorHeap::IID;
        let mut sampler_heap = null;
        assert_eq!(
            crate::descriptor::create_heap(
                null,
                (&raw const desc).cast(),
                (&raw const heap_iid).cast(),
                &mut sampler_heap,
            ),
            crate::object::S_OK
        );
        let title_heap = crate::descriptor::heap(sampler_heap);
        let host_heap = title_heap.host().expect("host sampler heap");
        let dyn_slot = host_heap.dynamic(200).expect("dynamic sampler slots");

        let mut state = State::default();
        let vp = [D3D12_VIEWPORT {
            Width: 64.0,
            Height: 32.0,
            MaxDepth: 1.0,
            ..Default::default()
        }];
        let sc = [RECT {
            left: 1,
            top: 2,
            right: 64,
            bottom: 32,
        }];
        let blend = [0.25, 0.5, 0.75, 1.0];
        state.set_topology(&prev_list, 4);
        state.set_viewports(&prev_list, &vp);
        state.set_scissors(&prev_list, &sc);
        state.set_blend_factor(&prev_list, Some(&blend));
        state.set_stencil_ref(&prev_list, 9);
        state.set_targets(&prev_list, &[rtv_handle], dsv_handle);
        let host_rt = crate::resource::host_resource(rt).expect("host rt").clone();
        let host_depth = crate::resource::host_resource(depth)
            .expect("host depth")
            .clone();
        // Overwrite the title's RTV/DSV descriptor slots in place before the
        // mid-list wait split; `reapply_host` must restore the views captured
        // at bind time rather than re-reading the mutated slots.
        rtv_slot.fill(0);
        dsv_slot.fill(0);
        state.host_pipeline = 0x1234;
        state.host_roots = [0x11, 0x22];
        state.host_heaps = [0x33, 0x44];
        state.host_heap_ids = [0x55, 0x66];
        state.fresh_records.push([0; 3]);
        state.fresh_tables.insert(
            0xbeef,
            FreshTableEntry {
                heap: 0xdead,
                copy_gen: 0,
                slots: 1,
                records_at: 0,
                first: 1,
                dead_slots: Vec::new(),
                aliases: Vec::new(),
                strides: Vec::new(),
            },
        );
        state
            .dynamic
            .push((std::ptr::from_ref(title_heap) as usize, dyn_slot, 200));

        state.reapply_host(&new_list, &prev_list);

        assert!(state.fresh_tables.is_empty());
        assert!(state.dynamic.is_empty());
        assert_eq!(state.host_pipeline, 0);
        assert_eq!(state.host_roots, [0; 2]);
        assert_eq!(state.host_heaps, [0; 2]);
        assert_eq!(state.host_heap_ids, [0; 2]);
        assert_eq!(state.topology, 4);
        assert_eq!(state.viewports.len(), 1);
        assert_eq!(state.viewports[0].Width, 64.0);
        assert_eq!(state.scissors.len(), 1);
        assert_eq!(state.scissors[0].left, 1);
        assert_eq!(state.blend_factor, Some(blend));
        assert_eq!(state.stencil_ref, Some(9));
        assert_eq!(state.bound_rtvs.len(), 1);
        assert_eq!(state.bound_rtvs[0].host.as_ref(), Some(&host_rt));
        assert_eq!(state.bound_dsv.as_ref().map(|d| &d.host), Some(&host_depth));
        assert!(new_list.kept_contains(&host_rt));
        assert!(new_list.kept_contains(&host_depth));
        assert!(
            !new_list.has_epoch(),
            "reapply_host must not pin an epoch on an empty trailing list"
        );
        assert_eq!(state.targets, vec![rt as usize]);
        assert_eq!(state.depth.map(|d| d.resource), Some(depth as usize));

        // Dynamic slots moved onto `prev_list` stay reserved until `prev_list` drops.
        let rest = host_heap.dynamic(200);
        assert!(rest.is_none(), "kept dynamic slots should still be held");

        prev_list.close();
        new_list.close();
        host.execute(&[&prev_list, &new_list]);
        drop(prev_list);
        assert!(
            host_heap.dynamic(200).is_some(),
            "dropping prev_list must release kept dynamic slots"
        );

        let release = |object: This| {
            crate::object::slot::<extern "system" fn(This) -> u32>(object, 0x10)(object)
        };
        assert_eq!(release(sampler_heap), 0);
        assert_eq!(release(depth), 0);
        assert_eq!(release(rt), 0);
    }

    #[test]
    fn set_targets_keeps_released_rtv_and_dsv_textures_alive_until_list_reset() {
        let _retire = crate::host::isolated_retire_scope();
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };

        let Some(host) = crate::host::host() else {
            return;
        };
        let null = std::ptr::null();
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let resource_iid = ID3D12Resource::IID;
        let create_tex = |format, flags, initial_state| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 4,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: flags,
                ..Default::default()
            };
            let mut tex = null;
            assert_eq!(
                crate::resource::create_committed(
                    null,
                    (&raw const heap).cast(),
                    0,
                    (&raw const desc).cast(),
                    initial_state,
                    null.cast(),
                    (&raw const resource_iid).cast(),
                    &mut tex,
                ),
                crate::object::S_OK
            );
            tex
        };
        let release = |o: This| crate::object::slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);

        let rtv_tex = create_tex(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
        );
        let dsv_tex = create_tex(
            DXGI_FORMAT_D32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32,
        );

        let mut rtv_slot = [0u64; 16];
        let mut dsv_slot = [0u64; 16];
        let rtv_handle = rtv_slot.as_mut_ptr() as u64;
        let dsv_handle = dsv_slot.as_mut_ptr() as u64;
        crate::descriptor::create_rtv(null, rtv_tex, null.cast(), rtv_handle);
        crate::descriptor::create_dsv(null, dsv_tex, null.cast(), dsv_handle);

        let rtv_weak =
            std::sync::Arc::downgrade(&crate::resource::host_copy(rtv_tex).expect("rtv host copy"));
        let dsv_weak =
            std::sync::Arc::downgrade(&crate::resource::host_copy(dsv_tex).expect("dsv host copy"));

        let list = host.new_list().expect("host list");
        let mut state = State::default();
        let _ = host.log_messages();
        let errors_before = host.d3d12_errors();
        state.set_targets(&list, &[rtv_handle], dsv_handle);
        assert!(
            list.kept_contains(crate::resource::host_resource(rtv_tex).expect("rtv host resource")),
            "set_targets must retain the host RTV COM object directly on the list"
        );
        assert!(
            list.kept_contains(crate::resource::host_resource(dsv_tex).expect("dsv host resource")),
            "set_targets must retain the host DSV COM object directly on the list"
        );

        // Release both title textures before executing the recorded list.
        assert_eq!(release(rtv_tex), 0);
        assert_eq!(release(dsv_tex), 0);
        assert!(
            rtv_weak.strong_count() > 0,
            "set_targets must keep host RTV texture alive while list is open"
        );
        assert!(
            dsv_weak.strong_count() > 0,
            "set_targets must keep host DSV texture alive while list is open"
        );

        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device-removed query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(
            host.d3d12_errors(),
            errors_before,
            "D3D12 debug layer reported errors on set_targets execution"
        );
        assert!(
            rtv_weak.strong_count() > 0,
            "host RTV texture must stay alive until list resets"
        );
        assert!(
            dsv_weak.strong_count() > 0,
            "host DSV texture must stay alive until list resets"
        );

        list.reset();
        assert_eq!(rtv_weak.strong_count(), 0);
        assert_eq!(dsv_weak.strong_count(), 0);

        // Dropping a recorded list without resetting also releases the kept
        // target textures cleanly.
        let rtv_tex2 = create_tex(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
        );
        crate::descriptor::create_rtv(null, rtv_tex2, null.cast(), rtv_handle);
        let rtv_weak2 = std::sync::Arc::downgrade(
            &crate::resource::host_copy(rtv_tex2).expect("rtv2 host copy"),
        );
        let mut state2 = State::default();
        state2.set_targets(&list, &[rtv_handle], 0);
        assert_eq!(release(rtv_tex2), 0);
        assert!(rtv_weak2.strong_count() > 0);
        list.close();
        host.execute(&[&list]);
        assert_eq!(host.d3d12_errors(), errors_before);
        assert!(rtv_weak2.strong_count() > 0);
        drop(list);
        assert_eq!(rtv_weak2.strong_count(), 0);
    }
}

#[cfg(test)]
mod table_tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_FLOAT, DXGI_SAMPLE_DESC,
    };
    use windows::core::s;

    struct TitleObject(This);

    impl Drop for TitleObject {
        fn drop(&mut self) {
            let release = crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10);
            release(self.0);
        }
    }

    fn create_title_heap(kind: D3D12_DESCRIPTOR_HEAP_TYPE, count: u32) -> TitleObject {
        #[repr(C)]
        struct PlacedDesc {
            desc: D3D12_DESCRIPTOR_HEAP_DESC,
            placement: [u64; 2],
        }
        let desc = PlacedDesc {
            desc: D3D12_DESCRIPTOR_HEAP_DESC {
                Type: kind,
                NumDescriptors: count,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            placement: [0; 2],
        };
        let iid = ID3D12DescriptorHeap::IID;
        let mut out = std::ptr::null();
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                (&raw const desc).cast(),
                (&raw const iid).cast(),
                &mut out,
            ),
            crate::object::S_OK
        );
        TitleObject(out)
    }

    fn create_title_texture(
        width: u64,
        height: u32,
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        initial_state: D3D12_RESOURCE_STATES,
    ) -> TitleObject {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: width,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: flags,
            ..Default::default()
        };
        let iid = ID3D12Resource::IID;
        let mut out = std::ptr::null();
        assert_eq!(
            crate::resource::create_committed(
                std::ptr::null(),
                (&raw const heap).cast(),
                0,
                (&raw const desc).cast(),
                initial_state.0 as u32,
                std::ptr::null(),
                (&raw const iid).cast(),
                &mut out,
            ),
            crate::object::S_OK
        );
        TitleObject(out)
    }

    fn create_sampler_at(desc: &D3D12_SAMPLER_DESC, handle: u64) {
        const _: () = assert!(size_of::<D3D12_SAMPLER_DESC>() == size_of::<[u32; 13]>());
        // SAFETY: `D3D12_SAMPLER_DESC` is 13 32-bit words with no padding.
        let words = unsafe {
            std::ptr::from_ref(desc)
                .cast::<[u32; 13]>()
                .read_unaligned()
        };
        crate::descriptor::create_sampler(std::ptr::null(), &words, handle);
    }

    fn used_table(kind: ResourceType, slot: u8, table_slot: u32, step_dwords: u32) -> UsedResource {
        UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: table_slot * step_dwords,
            lower_bound: 0,
            stride: 0,
            dwords: step_dwords as u8,
            user_data: 0,
            slot,
        }
    }

    fn sample_via_host_heaps(
        host: &crate::host::Host,
        view_heap: &ID3D12DescriptorHeap,
        view_slot: u32,
        sampler_heap: &ID3D12DescriptorHeap,
        sampler_slot: u32,
        uv: [f32; 2],
    ) -> [f32; 4] {
        const SOURCE: &str = r#"
            #define RS "RootConstants(num32BitConstants=2, b0), DescriptorTable(SRV(t0, numDescriptors=1)), DescriptorTable(Sampler(s0, numDescriptors=1)), UAV(u0)"
            cbuffer Coords : register(b0) { float2 uv; };
            Texture2D<float4> tex : register(t0);
            SamplerState samp : register(s0);
            RWByteAddressBuffer out_buf : register(u0);
            [RootSignature(RS)]
            [numthreads(1, 1, 1)]
            void main() {
                float4 c = tex.SampleLevel(samp, uv, 0);
                out_buf.Store4(0, asuint(c));
            }
        "#;
        let mut blob: Option<ID3DBlob> = None;
        // SAFETY: source and entry/target names are live; `blob` is our output.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("table-probe"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut blob,
                None,
            )
            .expect("compile probe compute shader");
        }
        let blob = blob.expect("compiled shader blob");
        // SAFETY: `blob` owns its bytecode buffer.
        let code = unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        };
        // SAFETY: root signature and compute PSO creation on live device.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host
                .device
                .CreateRootSignature(0, code)
                .expect("compute root signature");
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            };
            let pso = host
                .device
                .CreateComputePipelineState(&desc)
                .expect("compute pipeline");
            (root, pso)
        };
        let make_buffer = |kind: D3D12_HEAP_TYPE, flags, state| -> ID3D12Resource {
            let heap_props = D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: plain committed buffer creation.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap_props,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut out,
                    )
                    .expect("create probe buffer");
            }
            out.expect("probe buffer")
        };
        let uav_buf = make_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let readback = make_buffer(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let list = host.new_list().expect("probe list");
        let l = &list.list;
        // SAFETY: live device, heaps, root signature, PSO, and buffers on an open list.
        unsafe {
            let view_step = host
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let sampler_step = host
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
            let mut view_gpu = view_heap.GetGPUDescriptorHandleForHeapStart();
            view_gpu.ptr += u64::from(view_slot) * u64::from(view_step);
            let mut sampler_gpu = sampler_heap.GetGPUDescriptorHandleForHeapStart();
            sampler_gpu.ptr += u64::from(sampler_slot) * u64::from(sampler_step);

            l.SetDescriptorHeaps(&[Some(view_heap.clone()), Some(sampler_heap.clone())]);
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            let uv_bits = [uv[0].to_bits(), uv[1].to_bits()];
            l.SetComputeRoot32BitConstants(0, 2, uv_bits.as_ptr().cast(), 0);
            l.SetComputeRootDescriptorTable(1, view_gpu);
            l.SetComputeRootDescriptorTable(2, sampler_gpu);
            l.SetComputeRootUnorderedAccessView(3, uav_buf.GetGPUVirtualAddress());
            l.Dispatch(1, 1, 1);
            l.ResourceBarrier(&[crate::host::transition(
                &uav_buf,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyResource(&readback, &uav_buf);
        }
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: `host.execute` completed; `readback` holds at least 16 bytes.
        unsafe {
            readback
                .Map(0, None, Some(&mut mapped))
                .expect("map probe readback");
            let val = mapped.cast::<[f32; 4]>().read();
            readback.Unmap(0, None);
            val
        }
    }

    #[test]
    fn compact_range_allocates_used_slots_only_and_mirrors_view() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let tex = create_title_texture(
            4,
            4,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let init_list = host.new_list().expect("init list");
        let host_tex = crate::resource::host_resource(tex.0).expect("host tex");
        crate::list::ensure_state(
            &init_list,
            tex.0,
            D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
        );
        // SAFETY: open list and live texture RTV.
        unsafe {
            init_list.list.ClearRenderTargetView(
                host.rtv(Some(host_tex), None),
                &[0.0, 1.0, 0.0, 1.0],
                None,
            );
        }
        init_list.close();
        host.execute(&[&init_list]);

        let views = create_title_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1024);
        let samplers = create_title_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 4);
        let view_heap = crate::descriptor::heap(views.0);
        let sampler_heap = crate::descriptor::heap(samplers.0);
        let view_host = view_heap.host().expect("host view heap");
        let sampler_host = sampler_heap.host().expect("host sampler heap");

        let table = view_heap.start;
        crate::descriptor::create_srv(std::ptr::null(), tex.0, std::ptr::null(), table + 500 * 32);
        let clamp_sampler = D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            ..Default::default()
        };
        create_sampler_at(&clamp_sampler, sampler_heap.start);
        assert!(sampler_host.mirror(0, sampler_heap.start, true).is_some());

        let list = host.new_list().expect("table list");
        let mut state = State::default();
        state.set_heaps(&[views.0, samplers.0]);
        state.set_table(false, 0, table);
        let used = used_table(ResourceType::Srv, 0, 500, 8);
        let made = state.tables(&list, &[used], false, &[]);
        assert!(!state.table_failed);
        assert_eq!(state.dynamic.len(), 1);
        let (dyn_heap, first, n) = state.dynamic[0];
        assert_eq!(dyn_heap, std::ptr::from_ref(view_heap) as usize);
        assert_eq!(n, 1, "must allocate only 1 compact slot for slot 500");
        assert!(first >= view_host.count());
        assert_eq!(made, vec![(0, (first - 500) * 32)]);

        // Confirm HostHeap handed out exactly 1 dynamic slot.
        let next_slot = view_host.dynamic(1).expect("next dynamic slot");
        assert_eq!(next_slot, first + 1);
        view_host.release(next_slot, 1);

        // Execute `list` so `tex` transitions to SHADER_RESOURCE, then sample via `first`.
        list.close();
        host.execute(&[&list]);
        let sampled = sample_via_host_heaps(
            host,
            &view_host.heap,
            made[0].1 / 32 + 500,
            &sampler_host.heap,
            0,
            [0.5, 0.5],
        );
        assert_eq!(sampled, [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn exact_content_dedupe_shares_slots_across_distinct_table_addresses() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let depth = create_title_texture(
            4,
            4,
            DXGI_FORMAT_D32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
        );
        let srv_desc = tex2d_srv_words(DXGI_FORMAT_R32_FLOAT, 1);
        let dsv_desc = [
            DXGI_FORMAT_D32_FLOAT.0 as u32,
            D3D12_DSV_DIMENSION_TEXTURE2D.0 as u32,
            D3D12_DSV_FLAG_READ_ONLY_DEPTH.0 as u32,
            0,
            0,
            0,
        ];
        let mut dsv_slot = [0u64; 16];
        let dsv = dsv_slot.as_mut_ptr() as u64;
        crate::descriptor::create_dsv(std::ptr::null(), depth.0, &dsv_desc, dsv);

        let views = create_title_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1024);
        let views_other = create_title_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 64);
        let view_heap = crate::descriptor::heap(views.0);
        let other_heap = crate::descriptor::heap(views_other.0);
        let view_host = view_heap.host().expect("host view heap");
        let other_host = other_heap.host().expect("other host view heap");

        let table1 = view_heap.start + 100 * 32;
        let table2 = view_heap.start + 300 * 32;
        let table_other = other_heap.start;
        crate::descriptor::create_srv(std::ptr::null(), depth.0, &srv_desc, table1 + 10 * 32);
        crate::descriptor::create_srv(std::ptr::null(), depth.0, &srv_desc, table2 + 25 * 32);
        crate::descriptor::create_srv(std::ptr::null(), depth.0, &srv_desc, table_other + 10 * 32);

        let list1 = host.new_list().expect("list1");
        let mut state = State::default();
        state.set_targets(&list1, &[], dsv);
        state.set_heaps(&[views.0]);
        state.set_table(false, 0, table1);

        let used1 = used_table(ResourceType::Srv, 0, 10, 8);
        let made1 = state.tables(&list1, &[used1], false, &[]);
        assert!(!state.table_failed);
        assert_eq!(state.dynamic.len(), 1);
        assert_eq!(state.fresh_tables.len(), 1);
        let (_, first, n) = state.dynamic[0];
        assert_eq!(n, 1);
        assert!(first >= view_host.count());
        assert_eq!(made1, vec![(0, (first - 10) * 32)]);
        assert!(state.depth_sampled);

        // Second table at a different title address with identical descriptor content.
        let list2 = host.new_list().expect("list2");
        state.depth_sampled = false;
        state.set_table(false, 0, table2);
        let used2 = used_table(ResourceType::Srv, 0, 25, 8);
        let made2 = state.tables(&list2, &[used2], false, &[]);
        assert!(!state.table_failed);
        assert_eq!(state.dynamic.len(), 1, "must reuse dynamic allocation");
        assert_eq!(
            state.fresh_tables.len(),
            1,
            "must dedupe across table addresses"
        );
        assert_eq!(made2, vec![(0, (first - 25) * 32)]);
        assert!(state.depth_sampled, "cache hit must still run readable");

        let shader_resource = (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
            .0 as u32;
        let combined = D3D12_RESOURCE_STATE_DEPTH_READ.0 as u32 | shader_resource;
        assert!(
            list2.transition(depth.0, combined).is_none(),
            "cache hit must record depth read + shader resource transition on list2"
        );

        // Identical descriptor content in a different heap must NOT share slots across heaps.
        state.set_heaps(&[views_other.0]);
        state.set_table(false, 0, table_other);
        let made_other = state.tables(&list2, &[used1], false, &[]);
        assert!(!state.table_failed);
        assert_eq!(state.dynamic.len(), 2);
        assert_eq!(state.fresh_tables.len(), 2);
        let (other_ptr, other_first, other_n) = state.dynamic[1];
        assert_eq!(other_ptr, std::ptr::from_ref(other_heap) as usize);
        assert_eq!(other_n, 1);
        assert!(other_first >= other_host.count());
        assert_eq!(made_other, vec![(0, (other_first - 10) * 32)]);

        list1.close();
        list2.close();
        host.execute(&[&list1, &list2]);
    }

    #[test]
    fn sampler_heap_rebuild_uses_16_byte_stride_for_adjacent_slots() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let tex = create_title_texture(
            2,
            1,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let init_list = host.new_list().expect("init list");
        let host_tex = crate::resource::host_resource(tex.0).expect("host tex");
        crate::list::ensure_state(
            &init_list,
            tex.0,
            D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
        );
        // Left pixel = Red, right pixel = Blue.
        let left = RECT {
            left: 0,
            top: 0,
            right: 1,
            bottom: 1,
        };
        let right = RECT {
            left: 1,
            top: 0,
            right: 2,
            bottom: 1,
        };
        // SAFETY: open list, live texture RTV, and valid sub-rectangles.
        unsafe {
            let rtv = host.rtv(Some(host_tex), None);
            init_list
                .list
                .ClearRenderTargetView(rtv, &[1.0, 0.0, 0.0, 1.0], Some(&[left]));
            init_list
                .list
                .ClearRenderTargetView(rtv, &[0.0, 0.0, 1.0, 1.0], Some(&[right]));
        }
        crate::list::ensure_state(&init_list, tex.0, SHADER_RESOURCE);
        init_list.close();
        host.execute(&[&init_list]);

        let views = create_title_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4);
        let samplers = create_title_heap(
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            4 * D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE,
        );
        let view_heap = crate::descriptor::heap(views.0);
        let sampler_heap = crate::descriptor::heap(samplers.0);
        let view_host = view_heap.host().expect("host view heap");
        let sampler_host = sampler_heap.host().expect("host sampler heap");
        assert!(sampler_host.count() < 4 * D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE);

        crate::descriptor::create_srv(std::ptr::null(), tex.0, std::ptr::null(), view_heap.start);
        assert!(view_host.mirror(0, view_heap.start, true).is_some());

        // Place table near the host sampler cap so slots 4 and 5 sit past `host.count()`.
        let table = sampler_heap.start + u64::from(sampler_host.count() - 2) * 16;
        let clamp_sampler = D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            ..Default::default()
        };
        let border_green_sampler = D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_BORDER,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_BORDER,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_BORDER,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            BorderColor: [0.0, 1.0, 0.0, 1.0],
            ..Default::default()
        };
        create_sampler_at(&clamp_sampler, table + 4 * 16);
        create_sampler_at(&border_green_sampler, table + 5 * 16);

        let list = host.new_list().expect("sampler table list");
        let mut state = State::default();
        state.set_heaps(&[views.0, samplers.0]);
        state.set_table(false, 1, table);
        let used = [
            used_table(ResourceType::Sampler, 1, 4, 4),
            used_table(ResourceType::Sampler, 1, 5, 4),
        ];
        let made = state.tables(&list, &used, false, &[]);
        assert!(!state.table_failed);
        assert_eq!(state.dynamic.len(), 1);
        let (_, first, n) = state.dynamic[0];
        assert_eq!(n, 2, "two adjacent sampler slots");
        assert!(first >= sampler_host.count());
        assert_eq!(made, vec![(1, (first - 4) * 16)]);

        list.close();
        host.execute(&[&list]);

        let sampled_clamp = sample_via_host_heaps(
            host,
            &view_host.heap,
            0,
            &sampler_host.heap,
            made[0].1 / 16 + 4,
            [1.25, 0.5],
        );
        assert_eq!(sampled_clamp, [0.0, 0.0, 1.0, 1.0]);

        let sampled_border = sample_via_host_heaps(
            host,
            &view_host.heap,
            0,
            &sampler_host.heap,
            made[0].1 / 16 + 5,
            [1.25, 0.5],
        );
        assert_eq!(sampled_border, [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn capped_heap_fallback_starts_at_zero_without_underflow() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let samplers = create_title_heap(
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            4 * D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE,
        );
        let sampler_heap = crate::descriptor::heap(samplers.0);
        let sampler_host = sampler_heap.host().expect("host sampler heap");
        let cap = sampler_host.count();
        let table = sampler_heap.start;
        let desc = D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            ..Default::default()
        };
        create_sampler_at(&desc, table + u64::from(cap) * 16);
        create_sampler_at(&desc, table + u64::from(cap + 1) * 16);

        // Boundary: `min_slot == host.count()` keeps `start_slot = min_slot` with `first >= cap`.
        {
            let list = host.new_list().expect("boundary list");
            let mut state = State::default();
            state.set_heaps(&[samplers.0]);
            state.set_table(false, 0, table);
            let used = used_table(ResourceType::Sampler, 0, cap, 4);
            let made = state.tables(&list, &[used], false, &[]);
            assert!(!state.table_failed);
            assert_eq!(state.dynamic.len(), 1);
            let (_, first, n) = state.dynamic[0];
            assert_eq!(n, 1);
            assert!(first >= cap);
            assert_eq!(made, vec![(0, (first - cap) * 16)]);
            list.close();
            host.execute(&[&list]);
        }

        // Capped fallback: `host.count() < min_slot` resets `start_slot = 0` instead of
        // underflowing `first - start_slot`; `n = cap + 2` exceeds `SAMPLER_DYNAMIC`, so
        // allocation cleanly fails with `table_failed`.
        {
            let list = host.new_list().expect("fallback list");
            let mut state = State::default();
            state.set_heaps(&[samplers.0]);
            state.set_table(false, 0, table);
            let used = used_table(ResourceType::Sampler, 0, cap + 1, 4);
            let made = state.tables(&list, &[used], false, &[]);
            assert!(state.table_failed);
            assert!(made.is_empty());
            assert!(state.dynamic.is_empty());
            list.close();
            host.execute(&[&list]);
        }
    }

    #[test]
    fn multiple_used_resources_at_same_slot_all_invoke_readable() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let uav_tex = create_title_texture(
            4,
            4,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let depth = create_title_texture(
            4,
            4,
            DXGI_FORMAT_D32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
        );
        let dsv_desc = [
            DXGI_FORMAT_D32_FLOAT.0 as u32,
            D3D12_DSV_DIMENSION_TEXTURE2D.0 as u32,
            D3D12_DSV_FLAG_READ_ONLY_DEPTH.0 as u32,
            0,
            0,
            0,
        ];
        let mut dsv_slot = [0u64; 16];
        let dsv = dsv_slot.as_mut_ptr() as u64;
        crate::descriptor::create_dsv(std::ptr::null(), depth.0, &dsv_desc, dsv);

        let depth_srv_desc = tex2d_srv_words(DXGI_FORMAT_R32_FLOAT, 1);

        let views = create_title_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 64);
        let view_heap = crate::descriptor::heap(views.0);
        let table0 = view_heap.start;
        let table1 = view_heap.start + 16 * 32;
        crate::descriptor::create_uav(
            std::ptr::null(),
            uav_tex.0,
            std::ptr::null(),
            std::ptr::null(),
            table0,
        );
        crate::descriptor::create_srv(std::ptr::null(), depth.0, &depth_srv_desc, table1);

        let list = host.new_list().expect("multi-used list");
        crate::list::ensure_state(&list, uav_tex.0, D3D12_RESOURCE_STATE_COMMON.0 as u32);
        let mut state = State::default();
        state.set_targets(&list, &[], dsv);
        state.set_heaps(&[views.0]);
        state.set_table(false, 0, table0);
        state.set_table(false, 1, table1);

        let used = [
            // Slot 0 has SRV before UAV: both must invoke `readable` so UAV transitions `uav_tex`.
            used_table(ResourceType::Srv, 0, 0, 8),
            used_table(ResourceType::Uav, 0, 0, 8),
            // Slot 1 has CBV before SRV: both must invoke `readable` so SRV sets `depth_sampled`.
            used_table(ResourceType::Cbv, 1, 0, 8),
            used_table(ResourceType::Srv, 1, 0, 8),
        ];
        let made = state.tables(&list, &used, false, &[]);
        assert!(!state.table_failed);
        assert_eq!(made.len(), 2);
        assert!(
            list.transition(uav_tex.0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32)
                .is_none(),
            "UAV entry sharing a table slot with an SRV entry must still transition the texture"
        );
        assert!(
            state.depth_sampled,
            "SRV entry sharing a table slot with a CBV entry must still mark depth_sampled"
        );

        list.close();
        host.execute(&[&list]);
    }

    #[test]
    fn released_view_invalidates_cached_fresh_table_and_rebuilds_fresh_range() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let tex = create_title_texture(
            4,
            4,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let views = create_title_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 64);
        let view_heap = crate::descriptor::heap(views.0);
        let table = view_heap.start;
        crate::descriptor::create_srv(std::ptr::null(), tex.0, std::ptr::null(), table);

        let list = host.new_list().expect("liveness list");
        let mut state = State::default();
        state.set_heaps(&[views.0]);
        state.set_table(false, 0, table);
        let used = [used_table(ResourceType::Srv, 0, 0, 8)];

        let made1 = state.tables(&list, &used, false, &[]);
        assert!(!state.table_failed);
        assert_eq!(state.dynamic.len(), 1);
        let first1 = state.dynamic[0].1;
        assert_eq!(made1, vec![(0, first1 * 32)]);

        // Release the title texture: its raw handle words stay in `table`, but
        // `descriptor::forget` bumped the view's generation so the cached range
        // must not be reused.
        drop(tex);
        let made2 = state.tables(&list, &used, false, &[]);
        assert!(!state.table_failed);
        assert_eq!(
            state.dynamic.len(),
            2,
            "stale released view must trigger a fresh allocation while keeping the earlier range"
        );
        let first2 = state.dynamic[1].1;
        assert_ne!(first2, first1);
        assert_eq!(made2, vec![(0, first2 * 32)]);

        // Subsequent identical draws with the same unchanged stale slot must
        // reuse the rebuilt null range rather than leaking new dynamic slots.
        let made3 = state.tables(&list, &used, false, &[]);
        assert!(!state.table_failed);
        assert_eq!(
            state.dynamic.len(),
            2,
            "unchanged stale slot must reuse its rebuilt null range"
        );
        assert_eq!(made3, made2);

        list.close();
        host.execute(&[&list]);
    }

    #[test]
    fn oversized_table_exceeding_dynamic_capacity_fails_before_allocating() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        // A capped sampler heap has 256 dynamic slots (`SAMPLER_DYNAMIC`) and
        // `host.count() == 1792`. A fresh table starting past `host.count()`
        // whose span exceeds `dynamic_capacity()` must be rejected cleanly
        // before hashing or allocating.
        let samplers = create_title_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 4096);
        let sampler_heap = crate::descriptor::heap(samplers.0);
        let sampler_host = sampler_heap.host().expect("host sampler heap");
        let cap = sampler_host.dynamic_capacity();
        assert!(cap > 0);
        let base_slot = sampler_host.count() + 8;
        let table = sampler_heap.start + u64::from(base_slot) * 16;

        let list = host.new_list().expect("oversized list");
        let mut state = State::default();
        state.set_heaps(&[std::ptr::null(), samplers.0]);
        state.set_table(false, 0, table);
        // Span `0..=cap` has `n = cap + 1 > dynamic_capacity()`.
        let used = [
            used_table(ResourceType::Sampler, 0, 0, 0),
            used_table(ResourceType::Sampler, 0, 1, cap * 4),
        ];
        let made = state.tables(&list, &used, false, &[]);
        assert!(state.table_failed);
        assert!(made.is_empty());
        assert!(state.fresh_tables.is_empty());
        assert!(state.dynamic.is_empty());
    }

    #[test]
    fn intervening_defer_copy_allocates_fresh_cbv_snapshot() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let views = create_title_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 64);
        let view_heap = crate::descriptor::heap(views.0);
        let mut cb_dst = [1u8; 256];
        let cb_src = [2u8; 256];
        let cb_addr = cb_dst.as_ptr() as u64;
        let desc = [cb_addr as u32, (cb_addr >> 32) as u32, 256, 0];

        let table1 = view_heap.start;
        let table2 = view_heap.start + 8 * 32;
        crate::descriptor::create_cbv(std::ptr::null(), &desc, table1);
        // Copy the exact CBV handle words into `table2` so both tables have
        // identical descriptor content.
        // SAFETY: both `table1` and `table2` are valid, distinct 32-byte slots in `views`.
        unsafe {
            std::ptr::copy_nonoverlapping(table1 as *const u8, table2 as *mut u8, 32);
        }

        let list = host.new_list().expect("cbv defer_copy list");
        let mut state = State::default();
        state.set_heaps(&[views.0]);
        state.set_table(false, 0, table1);
        let used = [used_table(ResourceType::Cbv, 0, 0, 0)];

        let made1 = state.tables(&list, &used, false, &[]);
        assert!(!state.table_failed);
        assert_eq!(state.dynamic.len(), 1);
        let first1 = state.dynamic[0].1;
        assert_eq!(made1, vec![(0, first1 * 32)]);

        // Without an intervening deferred copy, identical table content at
        // `table2` reuses `first1`.
        state.set_table(false, 0, table2);
        let made2 = state.tables(&list, &used, false, &[]);
        assert!(!state.table_failed);
        assert_eq!(state.dynamic.len(), 1);
        assert_eq!(made2, vec![(0, first1 * 32)]);

        // An intervening `defer_copy` (e.g. `CopyBufferRegion` into the CBV)
        // must partition the cache so the next draw allocates a fresh CBV
        // snapshot (`Fill -> Copy -> Fill`).
        list.defer_copy(cb_dst.as_mut_ptr(), cb_src.as_ptr(), cb_src.len());
        let made3 = state.tables(&list, &used, false, &[]);
        assert!(!state.table_failed);
        assert_eq!(state.dynamic.len(), 2);
        let first2 = state.dynamic[1].1;
        assert_ne!(first2, first1);
        assert_eq!(made3, vec![(0, first2 * 32)]);

        // Subsequent draws with no further copies reuse `first2`.
        state.set_table(false, 0, table1);
        let made4 = state.tables(&list, &used, false, &[]);
        assert!(!state.table_failed);
        assert_eq!(state.dynamic.len(), 2);
        assert_eq!(made4, vec![(0, first2 * 32)]);

        // An intervening `defer` (e.g. `CopyMemoryToMemoryX` writing title
        // memory) must also bump `copy_gen` and allocate a fresh CBV snapshot.
        list.defer(|| {});
        let made5 = state.tables(&list, &used, false, &[]);
        assert!(!state.table_failed);
        assert_eq!(state.dynamic.len(), 3);
        let first3 = state.dynamic[2].1;
        assert_ne!(first3, first2);
        assert_eq!(made5, vec![(0, first3 * 32)]);

        list.close();
        host.execute(&[&list]);
        assert_eq!(cb_dst, cb_src);
    }
}

#[cfg(test)]
mod bindless_tests {
    use super::*;
    use crate::pipeline::Shader;
    use crate::pipeline::host::{HostPipeline, Stage};
    use victory_shader::HwStage;
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
    };
    use windows::core::s;

    struct OwnedObject(This);

    impl Drop for OwnedObject {
        fn drop(&mut self) {
            if !self.0.is_null() {
                let release = crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10);
                release(self.0);
            }
        }
    }

    fn make_texture(format: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS) -> OwnedObject {
        let null = std::ptr::null();
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: flags,
            ..Default::default()
        };
        let iid = ID3D12Resource::IID;
        let mut out = null;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const heap).cast(),
                0,
                (&raw const desc).cast(),
                0,
                null.cast(),
                (&raw const iid).cast(),
                &mut out,
            ),
            crate::object::S_OK
        );
        OwnedObject(out)
    }

    fn make_view_heap(count: u32) -> OwnedObject {
        #[repr(C)]
        struct ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC,
            placement: [u64; 2],
        }
        let desc = ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: count,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            placement: [0; 2],
        };
        let iid = ID3D12DescriptorHeap::IID;
        let mut out = std::ptr::null();
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                (&raw const desc).cast(),
                (&raw const iid).cast(),
                &mut out,
            ),
            crate::object::S_OK
        );
        OwnedObject(out)
    }

    fn compile_hlsl(
        source: &str,
        entry: windows::core::PCSTR,
        target: windows::core::PCSTR,
    ) -> ID3DBlob {
        let mut code = None;
        // SAFETY: source and entry/target strings are live; code is our output.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("bindless-test"),
                None,
                None,
                entry,
                target,
                0,
                0,
                &mut code,
                None,
            )
            .expect("compile test shader");
        }
        code.expect("shader blob")
    }

    fn make_test_pipelines(device: &ID3D12Device) -> (HostPipeline, HostPipeline, HostPipeline) {
        let cs_src = r#"
            [RootSignature("RootFlags(0)")]
            [numthreads(1, 1, 1)]
            void main() {}
        "#;
        let gfx_src = r#"
            [RootSignature("RootFlags(0)")]
            float4 vs(uint id : SV_VertexID) : SV_Position { return 0; }
            float4 ps() : SV_Target0 { return 1; }
        "#;
        let cs = compile_hlsl(cs_src, s!("main"), s!("cs_5_1"));
        let vs = compile_hlsl(gfx_src, s!("vs"), s!("vs_5_1"));
        let ps = compile_hlsl(gfx_src, s!("ps"), s!("ps_5_1"));

        // SAFETY: compiled blobs contain embedded root signatures and valid DXBC.
        let (cs_root, cs_pso, gfx_root, gfx_pso) = unsafe {
            let bytes = |b: &ID3DBlob| {
                std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())
            };
            let bc = |b: &ID3DBlob| D3D12_SHADER_BYTECODE {
                pShaderBytecode: b.GetBufferPointer(),
                BytecodeLength: b.GetBufferSize(),
            };
            let cs_root: ID3D12RootSignature =
                device.CreateRootSignature(0, bytes(&cs)).expect("cs root");
            let cs_pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&cs_root),
                    CS: bc(&cs),
                    ..Default::default()
                })
                .expect("cs pso");
            let gfx_root: ID3D12RootSignature =
                device.CreateRootSignature(0, bytes(&vs)).expect("gfx root");
            let mut gfx_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&gfx_root),
                VS: bc(&vs),
                PS: bc(&ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            gfx_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            let gfx_pso: ID3D12PipelineState = device
                .CreateGraphicsPipelineState(&gfx_desc)
                .expect("gfx pso");
            (cs_root, cs_pso, gfx_root, gfx_pso)
        };

        let used_srv = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::ResourceHeap,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        let used_uav = UsedResource {
            kind: ResourceType::Uav,
            ..used_srv
        };

        let gfx_srv_uav = HostPipeline {
            structured_strides: Vec::new(),
            fallback: None,
            compute: false,
            mesh: false,
            rects: false,
            amplify: None,
            pso: gfx_pso.clone(),
            root: gfx_root.clone(),
            stages: vec![Stage {
                shader: Shader::for_test("pixel", HwStage::Ps, vec![used_srv, used_uav]),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                fallback: false,
            }],
            slots: 0,
            inputs: Vec::new(),
            name: "gfx_srv_uav".into(),
        };
        let gfx_srv_only = HostPipeline {
            structured_strides: Vec::new(),
            fallback: None,
            compute: false,
            mesh: false,
            rects: false,
            amplify: None,
            pso: gfx_pso,
            root: gfx_root,
            stages: vec![Stage {
                shader: Shader::for_test("pixel", HwStage::Ps, vec![used_srv]),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                fallback: false,
            }],
            slots: 0,
            inputs: Vec::new(),
            name: "gfx_srv_only".into(),
        };
        let compute_srv = HostPipeline {
            structured_strides: Vec::new(),
            fallback: None,
            compute: true,
            mesh: false,
            rects: false,
            amplify: None,
            pso: cs_pso,
            root: cs_root,
            stages: vec![Stage {
                shader: Shader::for_test("compute", HwStage::Cs, vec![used_srv]),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                fallback: false,
            }],
            slots: 0,
            inputs: Vec::new(),
            name: "compute_srv".into(),
        };
        (gfx_srv_uav, gfx_srv_only, compute_srv)
    }

    #[test]
    fn bindless_textures_transitions_uploads_skips_targets_and_caches() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let _ = host.log_messages();
        let (gfx_srv_uav, gfx_srv_only, compute_srv) = make_test_pipelines(&host.device);

        let rt = make_texture(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        );
        let depth = make_texture(
            DXGI_FORMAT_D32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        );
        let uav_tex = make_texture(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        );
        // Title-written texture (flags = 0) with non-zero bytes in title memory.
        let tw_tex = make_texture(DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
        let (tw_mem, tw_desc) = crate::resource::texture_memory(tw_tex.0).unwrap();
        let tw_size = crate::resource::texture_layout(&tw_desc).unwrap().size as usize;
        // SAFETY: title memory for `tw_tex` holds at least `tw_size` writable bytes.
        unsafe { std::slice::from_raw_parts_mut(tw_mem.cast_mut(), tw_size).fill(0x5a) };
        let srv_tex = make_texture(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        );

        let heap = make_view_heap(16);
        let start = crate::descriptor::heap(heap.0).start;
        let slot_at = |i: u64| start + i * 32;
        let null = std::ptr::null();
        crate::descriptor::create_srv(null, rt.0, null.cast(), slot_at(0));
        crate::descriptor::create_srv(null, depth.0, null.cast(), slot_at(1));
        // `uav_tex` is bound as BOTH SRV and UAV in the heap to test UAV precedence.
        crate::descriptor::create_srv(null, uav_tex.0, null.cast(), slot_at(2));
        crate::descriptor::create_uav(null, uav_tex.0, null, null.cast(), slot_at(3));
        crate::descriptor::create_srv(null, tw_tex.0, null.cast(), slot_at(4));
        crate::descriptor::create_srv(null, srv_tex.0, null.cast(), slot_at(5));
        crate::descriptor::create_uav(null, depth.0, null, null.cast(), slot_at(6));

        let list = host.new_list().expect("host list");
        let mut rtv_slot = [0u64; 16];
        let mut dsv_slot = [0u64; 16];
        crate::descriptor::create_rtv(null, rt.0, null.cast(), rtv_slot.as_mut_ptr() as u64);
        let dsv_write_desc = [
            DXGI_FORMAT_D32_FLOAT.0 as u32,
            D3D12_DSV_DIMENSION_TEXTURE2D.0 as u32,
            D3D12_DSV_FLAG_NONE.0 as u32,
            0,
            0,
            0,
        ];
        crate::descriptor::create_dsv(null, depth.0, &dsv_write_desc, dsv_slot.as_mut_ptr() as u64);

        const UNORDERED_ACCESS: u32 = 0x8;
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;

        // Put active RT in COPY_SOURCE and active depth in DEPTH_WRITE before bind.
        crate::list::ensure_state(&list, rt.0, COPY_SOURCE);
        crate::list::ensure_state(&list, depth.0, DEPTH_WRITE);

        let mut state = State::default();
        state.set_heaps(&[heap.0]);
        state.set_targets(&list, &[rtv_slot.as_ptr() as u64], dsv_slot.as_ptr() as u64);

        // 1. Graphics bind with ResourceHeap SRV + UAV and DEPTH_WRITE.
        assert!(state.bind(&list, &gfx_srv_uav, false));
        assert!(
            !state.depth_sampled,
            "DEPTH_WRITE must not mark depth_sampled"
        );
        assert!(
            list.transition(rt.0, COPY_SOURCE).is_none(),
            "active RT must be skipped"
        );
        assert!(
            list.transition(depth.0, DEPTH_WRITE).is_none(),
            "active DEPTH_WRITE target must be skipped"
        );
        assert!(
            list.transition(uav_tex.0, UNORDERED_ACCESS).is_none(),
            "UAV must take precedence over SRV when uses_heap_uav is true"
        );
        assert!(
            list.covers_texture(tw_tex.0 as usize, 0),
            "title_written SRV must be uploaded"
        );
        assert!(
            list.transition(tw_tex.0, SHADER_RESOURCE).is_none(),
            "title_written SRV must end in SHADER_RESOURCE"
        );
        assert!(list.transition(srv_tex.0, SHADER_RESOURCE).is_none());

        // 2. Consecutive identical bind hits bindless_cache without state changes.
        assert_eq!(state.bindless_full_walks, 1);
        let cached = state.bindless_cache.clone().expect("cache populated");
        let gen_before = list.state_gen();
        assert!(state.bind(&list, &gfx_srv_uav, false));
        assert_eq!(list.state_gen(), gen_before);
        assert_eq!(state.bindless_cache, Some(cached.clone()));
        assert_eq!(state.bindless_full_walks, 1);

        // 3. Intervening copy/clear bumps state_gen and uses the incremental
        // dirty-resource fast path (no full-heap walk).
        crate::list::ensure_state(&list, srv_tex.0, COPY_DEST);
        assert!(list.state_gen() > gen_before);
        assert!(state.bind(&list, &gfx_srv_uav, false));
        assert_eq!(
            state.bindless_full_walks, 1,
            "intervening state change with unchanged heap must use incremental dirty sync"
        );
        assert!(
            list.transition(srv_tex.0, SHADER_RESOURCE).is_none(),
            "incremental sync must re-transition dirtied srv_tex back to SHADER_RESOURCE"
        );
        assert_ne!(state.bindless_cache, Some(cached));

        // 3b. Writing a brand-new bindless SRV descriptor mid-list bumps
        // `tex_gen` and syncs only the newly added slot without another full walk.
        let late_srv = make_texture(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        );
        crate::descriptor::create_srv(null, late_srv.0, null.cast(), slot_at(6));
        assert!(state.bind(&list, &gfx_srv_uav, false));
        assert_eq!(
            state.bindless_full_walks, 1,
            "writing a new bindless descriptor mid-list must use incremental change-ring sync"
        );
        assert_eq!(list.tracked_state(late_srv.0), Some(SHADER_RESOURCE));

        // 4. Graphics bind with DEPTH_READ sets depth_sampled and keeps it on cache hit.
        let dsv_read_desc = [
            DXGI_FORMAT_D32_FLOAT.0 as u32,
            D3D12_DSV_DIMENSION_TEXTURE2D.0 as u32,
            D3D12_DSV_FLAG_READ_ONLY_DEPTH.0 as u32,
            0,
            0,
            0,
        ];
        crate::descriptor::create_dsv(null, depth.0, &dsv_read_desc, dsv_slot.as_mut_ptr() as u64);
        state.set_targets(&list, &[rtv_slot.as_ptr() as u64], dsv_slot.as_ptr() as u64);
        assert!(state.bind(&list, &gfx_srv_only, false));
        assert!(
            state.depth_sampled,
            "DEPTH_READ in live_srvs sets depth_sampled"
        );
        assert!(
            list.transition(depth.0, DEPTH_READ | SHADER_RESOURCE)
                .is_none()
        );
        assert!(
            list.transition(uav_tex.0, SHADER_RESOURCE).is_none(),
            "when uses_heap_uav is false, uav_tex transitions to SHADER_RESOURCE"
        );
        // Second identical bind hits cache and MUST preserve `depth_sampled = true`.
        let gen_read = list.state_gen();
        assert!(state.bind(&list, &gfx_srv_only, false));
        assert_eq!(list.state_gen(), gen_read);
        assert!(
            state.depth_sampled,
            "cache hit must preserve depth_sampled for DEPTH_READ"
        );

        // 5. Compute dispatch ignores graphics RTs/depth and transitions previously
        // skipped `rt` to SRV incrementally without another full-heap walk.
        let walks_before_compute = state.bindless_full_walks;
        assert!(state.bind(&list, &compute_srv, true));
        assert_eq!(state.bindless_full_walks, walks_before_compute);
        assert!(!state.depth_sampled);
        assert!(
            list.transition(rt.0, SHADER_RESOURCE).is_none(),
            "compute dispatch must transition graphics RT to SHADER_RESOURCE"
        );

        // 6. State::reset clears bindless_cache.
        state.reset(null);
        assert!(state.bindless_cache.is_none());

        let _ = host.log_messages();
        let errors_before = host.d3d12_errors();
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host.d3d12_errors(), errors_before);
    }

    #[test]
    fn short_read_notice_is_not_consumed_by_uav_only_or_other_heap_passes() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let (_, gfx_srv_only, compute_srv) = make_test_pipelines(&host.device);
        let used_uav = UsedResource {
            kind: ResourceType::Uav,
            root: RootType::ResourceHeap,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        let compute_uav_only = HostPipeline {
            structured_strides: Vec::new(),
            compute: true,
            mesh: false,
            rects: false,
            amplify: None,
            pso: compute_srv.pso.clone(),
            root: compute_srv.root.clone(),
            stages: vec![Stage {
                shader: Shader::for_test("compute", HwStage::Cs, vec![used_uav]),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                fallback: false,
            }],
            slots: 0,
            inputs: Vec::new(),
            name: "compute_uav_only".into(),
            fallback: None,
        };

        let tex_a = make_texture(DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
        let tex_b = make_texture(DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
        let heap_a = make_view_heap(4);
        let heap_b = make_view_heap(4);
        let null = std::ptr::null();
        crate::descriptor::create_srv(
            null,
            tex_a.0,
            null.cast(),
            crate::descriptor::heap(heap_a.0).start,
        );
        crate::descriptor::create_srv(
            null,
            tex_b.0,
            null.cast(),
            crate::descriptor::heap(heap_b.0).start,
        );

        let list = host.new_list().expect("host list");
        let mut state_a = State::default();
        state_a.set_heaps(&[heap_a.0]);
        assert!(state_a.bind(&list, &gfx_srv_only, false));
        let walks_a = state_a.bindless_full_walks;
        assert!(list.covers_texture(tex_a.0 as usize, 0));

        // Reset the list (keeping `state_a`'s bindless_cache warm, simulating a
        // cached State seeing a short-read notice for `tex_a`).
        let list2 = host.new_list().expect("host list 2");
        let (mem_a, _) = crate::resource::texture_memory(tex_a.0).unwrap();
        let start_a = mem_a as u64;
        crate::descriptor::host::read_all(start_a);
        crate::descriptor::host::read_short(start_a, tex_a.0 as usize);
        assert!(crate::descriptor::host::is_short_read_pending(
            tex_a.0 as usize
        ));

        // 1. A UAV-only bindless pass on heap A must NOT consume `tex_a`'s notice.
        let mut state_uav = State::default();
        state_uav.set_heaps(&[heap_a.0]);
        assert!(state_uav.bind(&list2, &compute_uav_only, true));
        assert!(
            crate::descriptor::host::is_short_read_pending(tex_a.0 as usize),
            "UAV-only pass must not consume an SRV short-read notice"
        );

        // 2. An SRV bindless pass on heap B (which does not contain `tex_a`)
        // must NOT consume `tex_a`'s notice.
        let mut state_b = State::default();
        state_b.set_heaps(&[heap_b.0]);
        assert!(state_b.bind(&list2, &gfx_srv_only, false));
        assert!(
            crate::descriptor::host::is_short_read_pending(tex_a.0 as usize),
            "SRV pass on an unrelated heap must not consume tex_a's notice"
        );

        // 3. The owning State on heap A (even with a warm bindless_cache) must
        // retry `tex_a` incrementally, schedule its upload on `list2`, and clear
        // the notice.
        assert!(!list2.covers_texture(tex_a.0 as usize, 0));
        assert!(state_a.bind(&list2, &gfx_srv_only, false));
        assert_eq!(
            state_a.bindless_full_walks, walks_a,
            "short-read retry on warm cache should use incremental subset sync"
        );
        assert!(
            list2.covers_texture(tex_a.0 as usize, 0),
            "owning heap SRV pass must upload the short-read texture"
        );
        assert!(
            !crate::descriptor::host::is_short_read_pending(tex_a.0 as usize),
            "short-read notice must clear once an eligible pass schedules the upload"
        );

        crate::descriptor::host::read_all(start_a);
        list.close();
        list.reset();
        list2.close();
        list2.reset();
    }

    #[test]
    fn bindless_textures_uses_hashset_when_over_64_uavs_are_live() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let _ = host.log_messages();
        let (gfx_srv_uav, _, _) = make_test_pipelines(&host.device);
        const UAVS: u32 = 70;
        const SRVS: u32 = 70;
        let heap = make_view_heap(UAVS + SRVS + 8);
        let start = crate::descriptor::heap(heap.0).start;
        let slot_at = |i: u32| start + u64::from(i) * 32;
        let null = std::ptr::null();

        let mut uav_textures = Vec::with_capacity(UAVS as usize);
        for i in 0..UAVS {
            let tex = make_texture(
                DXGI_FORMAT_R8G8B8A8_UNORM,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            );
            crate::descriptor::create_uav(null, tex.0, null, null.cast(), slot_at(i));
            uav_textures.push(tex);
        }
        // Populate > 64 live SRVs so `live_srvs.len() > 64` exercises the
        // `HashSet` lookup branch, including an overlapping UAV+SRV texture.
        crate::descriptor::create_srv(null, uav_textures[0].0, null.cast(), slot_at(UAVS));
        let mut srv_textures = Vec::with_capacity(SRVS as usize);
        for i in 0..SRVS {
            let tex = make_texture(
                DXGI_FORMAT_R8G8B8A8_UNORM,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            );
            crate::descriptor::create_srv(null, tex.0, null.cast(), slot_at(UAVS + 1 + i));
            srv_textures.push(tex);
        }

        let list = host.new_list().expect("host list");
        let mut state = State::default();
        state.set_heaps(&[heap.0]);
        assert!(state.bind(&list, &gfx_srv_uav, false));

        const UNORDERED_ACCESS: u32 = 0x8;
        for tex in &uav_textures {
            assert_eq!(list.tracked_state(tex.0), Some(UNORDERED_ACCESS));
        }
        for tex in &srv_textures {
            assert_eq!(list.tracked_state(tex.0), Some(SHADER_RESOURCE));
        }
        let _ = host.log_messages();
        let errors_before = host.d3d12_errors();
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host.d3d12_errors(), errors_before);
    }

    #[test]
    fn bindless_dual_view_texture_preserves_srv_state_when_already_readable() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let _ = host.log_messages();
        let (gfx_srv_uav, gfx_srv_only, _) = make_test_pipelines(&host.device);
        let heap = make_view_heap(16);
        let start = crate::descriptor::heap(heap.0).start;
        let slot_at = |i: u32| start + u64::from(i) * 32;
        let null = std::ptr::null();

        let tex_a = make_texture(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        );
        let tex_b = make_texture(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        );
        // Both textures have SRV + UAV views resident in the bindless heap.
        crate::descriptor::create_srv(null, tex_a.0, null.cast(), slot_at(0));
        crate::descriptor::create_uav(null, tex_a.0, null, null.cast(), slot_at(1));
        crate::descriptor::create_srv(null, tex_b.0, null.cast(), slot_at(2));
        crate::descriptor::create_uav(null, tex_b.0, null, null.cast(), slot_at(3));

        const UNORDERED_ACCESS: u32 = 0x8;
        let list = host.new_list().expect("host list");
        // Title transitions A for SRV sampling and B for UAV writing.
        crate::list::ensure_state(&list, tex_a.0, SHADER_RESOURCE);
        crate::list::ensure_state(&list, tex_b.0, UNORDERED_ACCESS);

        let mut state = State::default();
        state.set_heaps(&[heap.0]);
        // An intervening SRV-only bindless draw temporarily sweeps B to
        // SHADER_RESOURCE without overwriting its explicit UAV access intent.
        assert!(state.bind(&list, &gfx_srv_only, false));
        assert_eq!(list.tracked_state(tex_b.0), Some(SHADER_RESOURCE));

        // A subsequent dual SRV+UAV draw must keep A in SHADER_RESOURCE
        // (explicit SRV intent) and restore B to UNORDERED_ACCESS (explicit UAV
        // intent, not masked by the intervening bindless sweep).
        assert!(state.bind(&list, &gfx_srv_uav, false));
        assert_eq!(list.tracked_state(tex_a.0), Some(SHADER_RESOURCE));
        assert_eq!(list.tracked_state(tex_b.0), Some(UNORDERED_ACCESS));

        let _ = host.log_messages();
        let errors_before = host.d3d12_errors();
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host.d3d12_errors(), errors_before);
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod diagnostics_tests {
    use super::*;

    /// A mesh pipeline on a host without mesh shaders draws what its mesh
    /// program describes: the pipeline pair made for it, the compute pass
    /// into the host's mesh buffer and the draw that reads it back, for one
    /// thread group and then again in the same list (the buffer is reused).
    #[test]
    fn a_mesh_dispatch_draws_without_mesh_shaders() {
        mesh_dispatch_without_mesh_shaders(false, 1, u32::MAX);
    }

    /// The same with the thread-group count where only the GPU has it: the
    /// record is clamped, its count becomes the draws' instances, and the
    /// compute passes and the draws all run indirectly.
    #[test]
    fn an_indirect_mesh_dispatch_draws_without_mesh_shaders() {
        mesh_dispatch_without_mesh_shaders(true, 1, u32::MAX);
    }

    /// More thread groups than a part holds run in parts (three here, the
    /// last one not full), and the program counts its groups through all of
    /// them: the sample's groups each cover the target with their own
    /// value, so the last one's stays.
    #[test]
    fn a_mesh_dispatch_of_several_parts_draws_without_mesh_shaders() {
        mesh_dispatch_without_mesh_shaders(false, 2500, 1000);
    }

    /// The same for an indirect dispatch, whose parts the GPU fills in:
    /// three with groups, and those after them empty.
    #[test]
    fn an_indirect_mesh_dispatch_of_several_parts_draws_without_mesh_shaders() {
        mesh_dispatch_without_mesh_shaders(true, 12345, 5000);
    }

    fn mesh_dispatch_without_mesh_shaders(indirect: bool, groups: u32, part: u32) {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_SAMPLE_DESC,
        };

        use crate::pipeline::Shader;
        use crate::pipeline::host::GraphicsState;

        struct Texture(This);

        impl Drop for Texture {
            fn drop(&mut self) {
                let release = crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10);
                release(self.0);
            }
        }

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let device = &host.device;
        let format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        let sample = victory_shader::emit::sample::mesh();
        let ms = Shader::translated_as("mesh", sample.mesh, Some(sample.fallback));
        let ps = Shader::translated_as("pixel", sample.pixel, None);
        let mut g = GraphicsState::default();
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = format;
        let shaders = [ms.clone(), ps.clone()];
        let p = HostPipeline::without_mesh_shaders(&shaders, (&ms, Some(&ps)), Some(&g), &[])
            .expect("the compute and draw pipelines");
        let f = p.fallback.as_ref().expect("the compute half");
        assert_eq!((f.vertices, f.group_bytes), (3, 27 * 4));

        const SIDE: u32 = 16;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(SIDE),
            Height: SIDE,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut target = std::ptr::null();
        let null = std::ptr::null();
        let iid = ID3D12Resource::IID;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const heap).cast(),
                0,
                (&raw const desc).cast(),
                D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
                null.cast(),
                (&raw const iid).cast(),
                &mut target,
            ),
            crate::object::S_OK
        );
        let _target = Texture(target);
        let color = crate::resource::host_resource(target).expect("color target");
        let mut slot = [0u64; 16];
        let handle = slot.as_mut_ptr() as u64;
        crate::descriptor::create_rtv(null, target, null.cast(), handle);

        let list = host.new_list().expect("host command list");
        let l = &list.list;
        crate::list::ensure_state(&list, target, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32);
        // SAFETY: an open list and the live target.
        unsafe { l.ClearRenderTargetView(host.rtv(Some(color), None), &[9.0; 4], None) };
        viewports(
            &list,
            &[D3D12_VIEWPORT {
                Width: SIDE as f32,
                Height: SIDE as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }],
        );
        scissors(
            &list,
            &[RECT {
                right: SIDE as i32,
                bottom: SIDE as i32,
                ..Default::default()
            }],
        );
        let mut state = State::default();
        state.set_targets(&list, &[handle], 0);
        // The title buffer of the records, kept until the list has run (its
        // memory is read when the list is submitted), and its host copy, to
        // see what the passes left in it.
        let mut records_kept = None;
        let mut records_after = None;
        if indirect {
            // The thread-group count in title memory, where the GPU reads it.
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut records = std::ptr::null();
            assert_eq!(
                crate::resource::create_committed(
                    null,
                    (&raw const heap).cast(),
                    0,
                    (&raw const desc).cast(),
                    0,
                    null.cast(),
                    (&raw const iid).cast(),
                    &mut records,
                ),
                crate::object::S_OK
            );
            records_kept = Some(Texture(records));
            let (data, _) = crate::resource::buffer_bytes(records).expect("title memory");
            // One record of the thread groups, then one that must not run
            // (it would leave its own last group's value).
            // SAFETY: the buffer holds 256 bytes.
            unsafe {
                data.cast::<[[u32; 3]; 2]>()
                    .write_unaligned([[groups, 1, 1], [groups + 7, 1, 1]])
            };
            let address = data as u64;
            let args = crate::shadow::find(address, 24).expect("the records' host copy");
            assert!(state.execute_indirect_mesh(
                &list,
                (&p, f, part),
                (&args, address, 12, 2),
                None
            ));
            records_after = Some((args, address));
        } else {
            for _ in 0..2 {
                assert!(state.prepare_with(&list, &p, false));
                assert!(state.dispatch_mesh_without_mesh_shaders(&list, &p, f, (groups, part)));
            }
        }

        crate::list::ensure_state(&list, target, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
        let (readback, footprint) =
            crate::capture::read_back(device, l, color).expect("color readback");
        list.close();
        host.execute(&[&list]);
        drop(records_kept);
        if let Some((args, address)) = &records_after {
            // The passes read the records and leave them as the title's.
            let bytes = args.host_bytes(*address, 24);
            let words: Vec<u32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect();
            let title = [groups, 1, 1, groups + 7, 1, 1];
            assert_eq!(words, title, "the records after their passes");
        }
        let mut mapped = std::ptr::null_mut();
        let mut texels = Vec::new();
        // SAFETY: execution has completed and all texels lie in the mapped footprint.
        unsafe {
            readback
                .Map(0, None, Some(&mut mapped))
                .expect("map readback");
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let at = footprint.Offset as usize
                        + (y * footprint.Footprint.RowPitch + x * 16) as usize;
                    texels.push(
                        mapped
                            .cast::<u8>()
                            .add(at)
                            .cast::<[f32; 4]>()
                            .read_unaligned(),
                    );
                }
            }
            readback.Unmap(0, None);
        }
        let want = victory_shader::emit::sample::group_texels(SIDE, SIDE, groups - 1);
        for (n, (got, want)) in texels.iter().zip(&want).enumerate() {
            let close = got.iter().zip(want).all(|(a, b)| (a - b).abs() <= 1e-4);
            assert!(close, "texel {n} is {got:?}, not {want:?}");
        }
    }

    #[test]
    fn typed_null_rtv_keeps_the_next_color_slot() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        use windows::core::s;

        struct Texture(This);

        impl Drop for Texture {
            fn drop(&mut self) {
                let release = crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10);
                release(self.0);
            }
        }

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let device = &host.device;
        let list = host.new_list().expect("host command list");
        let l = &list.list;
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let source = r#"
            [RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)")]
            float4 vs(uint id : SV_VertexID) : SV_Position {
                return float4(id == 2 ? 3 : -1, id == 1 ? -3 : 1, 0, 1);
            }
            struct Colors { float4 red : SV_Target0; float4 green : SV_Target1; };
            Colors ps() {
                Colors c;
                c.red = float4(1, 0, 0, 1);
                c.green = float4(0, 1, 0, 1);
                return c;
            }
        "#;
        let compile = |entry, target| {
            let mut code = None;
            // SAFETY: source and shader names are live, and code is our output pointer.
            unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    s!("rtv-slot-test"),
                    None,
                    None,
                    entry,
                    target,
                    0,
                    0,
                    &mut code,
                    None,
                )
                .expect("compile fixture shader");
            }
            code.expect("fixture bytecode")
        };
        let vs = compile(s!("vs"), s!("vs_5_1"));
        let ps = compile(s!("ps"), s!("ps_5_1"));
        // SAFETY: compiled fixture shaders and their embedded root signature are live.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let bytecode = |blob: &ID3DBlob| D3D12_SHADER_BYTECODE {
                pShaderBytecode: blob.GetBufferPointer(),
                BytecodeLength: blob.GetBufferSize(),
            };
            let code = bytecode(&vs);
            let root = device
                .CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(code.pShaderBytecode.cast(), code.BytecodeLength),
                )
                .expect("graphics root signature");
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code,
                PS: bytecode(&ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 2,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            // D3D12 requires UNKNOWN in the PSO slot that binds a null RTV.
            desc.RTVFormats[1] = format;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
                D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let pso = device
                .CreateGraphicsPipelineState(&desc)
                .expect("graphics pipeline");
            (root, pso)
        };

        const SIDE: u32 = 4;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(SIDE),
            Height: SIDE,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut target = std::ptr::null();
        let null = std::ptr::null();
        let iid = ID3D12Resource::IID;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const heap).cast(),
                0,
                (&raw const desc).cast(),
                D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
                null.cast(),
                (&raw const iid).cast(),
                &mut target,
            ),
            crate::object::S_OK
        );
        let _target = Texture(target);
        let color = crate::resource::host_resource(target).expect("color target");
        let mut slots = [[0u64; 16]; 2];
        let handles = slots.each_mut().map(|slot| slot.as_mut_ptr() as u64);
        let null_desc = [
            format.0 as u32,
            D3D12_RTV_DIMENSION_TEXTURE2D.0 as u32,
            0,
            0,
            0,
            0,
        ];
        crate::descriptor::create_rtv(null, null, &null_desc, handles[0]);
        crate::descriptor::create_rtv(null, target, null.cast(), handles[1]);
        crate::list::ensure_state(&list, target, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32);
        // SAFETY: open list, live target and graphics pipeline.
        unsafe {
            l.ClearRenderTargetView(host.rtv(Some(color), None), &[0.0, 0.0, 1.0, 1.0], None);
            l.SetGraphicsRootSignature(&root);
            l.SetPipelineState(&pso);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }
        viewports(
            &list,
            &[D3D12_VIEWPORT {
                Width: SIDE as f32,
                Height: SIDE as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }],
        );
        scissors(
            &list,
            &[RECT {
                right: SIDE as i32,
                bottom: SIDE as i32,
                ..Default::default()
            }],
        );
        State::default().set_targets(&list, &handles, 0);
        // SAFETY: complete graphics state on the open list.
        unsafe { l.DrawInstanced(3, 1, 0, 0) };

        crate::list::ensure_state(&list, target, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
        let (readback, footprint) =
            crate::capture::read_back(device, l, color).expect("color readback");
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        let mut pixels = Vec::new();
        // SAFETY: execution has completed and all pixels lie in the mapped footprint.
        unsafe {
            readback
                .Map(0, None, Some(&mut mapped))
                .expect("map readback");
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let at = footprint.Offset as usize
                        + (y * footprint.Footprint.RowPitch + x * 4) as usize;
                    pixels.push(mapped.cast::<u8>().add(at).cast::<[u8; 4]>().read());
                }
            }
            readback.Unmap(0, None);
        }
        assert_eq!(pixels, [[0, 255, 0, 255]; (SIDE * SIDE) as usize]);
    }
}

#[cfg(test)]
mod structured_alias_tests {
    use super::*;

    struct TitleObject(This);

    impl Drop for TitleObject {
        fn drop(&mut self) {
            let release = crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10);
            release(self.0);
        }
    }

    struct ShadowGuard {
        resource: usize,
        address: u64,
    }

    impl Drop for ShadowGuard {
        fn drop(&mut self) {
            crate::shadow::unregister(self.resource, self.address);
        }
    }

    #[test]
    fn root_constant_selects_the_buffer_stride_and_bounded_range_at_dispatch() {
        use victory_shader::native::{ComputeIo, RootParam, StageIo};
        use victory_shader::{HwStage, NativeProgram};
        let Some(host) = crate::host::host() else {
            eprintln!("skipped indexed buffer readback: no D3D12 device");
            return;
        };
        // Select an SRV via s1, read the record in s2 at byte 4 and write it to UAV slot 0.
        let words = [
            0xbe84_0300u32,
            0xbe85_0380, // heap base in s[4:5]
            0x8f06_8501, // s6 = s1 << 5
            0xf408_0202,
            0x0c00_0000, // s[8:11] = selected descriptor
            0xf408_0302,
            0xfa00_0000, // s[12:15] = output descriptor
            0x7e00_0202, // v0 = s2
            0xe034_2004,
            0x8002_0100, // buffer_load_dwordx2 v[1:2], v0, s[8:11], idxen offset:4
            0xe074_0000,
            0x8003_0100, // buffer_store_dwordx2 v[1:2], off, s[12:15]
            0xbf81_0000,
        ];
        let root = |root, slot| RootParam {
            root,
            slot,
            user_data: slot,
            sgprs: 1,
            table_offset_dwords: 0,
            base_register: 0,
        };
        let shader = crate::pipeline::Shader::test_compute(NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![
                root(RootType::ResourceHeap, 0),
                root(RootType::Constants, 1),
                root(RootType::Constants, 2),
            ],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 4,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        });
        let pipeline = HostPipeline::new(&[shader], None, &[]).expect("indexed compute pipeline");
        let mut backing: Vec<u32> = (0..64).map(|i| 0x6200_0000 + i).collect();
        let base = backing.as_mut_ptr() as u64;
        crate::shadow::register(base as usize, base, 256);
        let _shadow = ShadowGuard {
            resource: base as usize,
            address: base,
        };
        #[repr(C)]
        struct HeapDesc {
            desc: D3D12_DESCRIPTOR_HEAP_DESC,
            placement: [u64; 2],
        }
        let desc = HeapDesc {
            desc: D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 4,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            placement: [0; 2],
        };
        let mut views = std::ptr::null();
        let iid = ID3D12DescriptorHeap::IID;
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                (&raw const desc).cast(),
                (&raw const iid).cast(),
                &mut views
            ),
            crate::object::S_OK
        );
        let _views = TitleObject(views);
        let heap = crate::descriptor::heap(views);
        let out = host.default_buffer(256).unwrap();
        heap.host().unwrap().raw_uav_at(0, &out, 0, 256);
        let mut resource = [0u32; 14];
        resource[0] = 1;
        resource[4] = 256;
        // The first view starts at byte 12 and has stride 12. The second starts
        // at byte 32 and has stride 32. The last contains only one record.
        for (slot, stride, count) in [(1, 12, 2), (2, 32, 2), (3, 16, 1)] {
            let mut words = [0u32; 16];
            words[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
            words[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            words[4] = 1; // FirstElement
            words[6] = count;
            words[7] = stride;
            words[10] = base as u32;
            words[11] = (base >> 32) as u32;
            crate::descriptor::create_placed_srv(
                std::ptr::null(),
                &resource,
                &words,
                heap.start + slot * 32,
            );
        }
        let (buffer, mapped) = host.readback_buffer(48).unwrap();
        let mut state = State::default();
        state.set_heaps(&[views]);
        let list = host.new_list().unwrap();
        // The last two addresses wrap during the multiplication or the offset addition.
        let cases = [
            (1, 1),
            (2, 1),
            (3, 1),
            (99, 1),
            (2, 0x0800_0001),
            (1, 0x1555_5555),
        ];
        for (i, (selected, record)) in cases.into_iter().enumerate() {
            state.set_constants(true, 2, &[record], 0);
            state.set_constants(true, 1, &[selected], 0);
            assert!(state.bind(&list, &pipeline, true));
            // SAFETY: the open list has live descriptors and the readback has two dwords per dispatch.
            unsafe {
                list.list.Dispatch(1, 1, 1);
                list.list.ResourceBarrier(&[crate::host::transition(
                    &out,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.list
                    .CopyBufferRegion(&buffer, 8 * i as u64, &out, 0, 8);
                list.list.ResourceBarrier(&[crate::host::transition(
                    &out,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: the submission completed, and the mapped buffer holds six pairs of dwords.
        let got = unsafe { mapped.cast::<[[u32; 2]; 6]>().read_unaligned() };
        assert_eq!(
            got,
            [
                [backing[7], backing[8]],
                [backing[17], backing[18]],
                [0; 2],
                [0; 2],
                [0; 2],
                [0; 2]
            ]
        );
        eprintln!("indexed buffer readback: {got:x?}");
    }

    #[test]
    fn state_tables_allocates_distinct_fresh_structured_table_and_refreshes_on_hit() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC};
        use windows::core::s;

        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };

        let mut backing = vec![0u8; 512];
        let base = backing.as_ptr() as u64;
        let _shadow = ShadowGuard {
            resource: base as usize,
            address: base,
        };
        crate::shadow::register(base as usize, base, backing.len() as u64);
        for i in 0..32u32 {
            backing[i as usize * 4..i as usize * 4 + 4]
                .copy_from_slice(&(0x3300_0000 | i).to_le_bytes());
        }

        #[repr(C)]
        struct ViewHeapDesc {
            desc: D3D12_DESCRIPTOR_HEAP_DESC,
            placement: [u64; 2],
        }
        let heap_desc = ViewHeapDesc {
            desc: D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 4,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            placement: [0; 2],
        };
        let heap_iid = ID3D12DescriptorHeap::IID;
        let mut views = std::ptr::null();
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                (&raw const heap_desc).cast(),
                (&raw const heap_iid).cast(),
                &mut views,
            ),
            crate::object::S_OK
        );
        let _views = TitleObject(views);

        let table_handle = crate::descriptor::heap(views).start;
        let mut res_words = [0u32; 14];
        res_words[0] = 1; // BUFFER
        res_words[4] = 128;
        let raw_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 32, // 32 raw dwords = 128 bytes = 2x 64-byte records
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: `D3D12_SHADER_RESOURCE_VIEW_DESC` is 40 bytes (`[u32; 10]`).
        let raw_words: [u32; 10] = unsafe {
            std::ptr::from_ref(&raw_srv)
                .cast::<[u32; 10]>()
                .read_unaligned()
        };
        let mut srv_words = [0u32; 16];
        srv_words[..10].copy_from_slice(&raw_words);
        srv_words[10] = base as u32;
        srv_words[11] = (base >> 32) as u32;
        crate::descriptor::create_placed_srv(
            std::ptr::null(),
            &res_words,
            &srv_words,
            table_handle,
        );

        let host_heap = crate::descriptor::heap(views).host().expect("host heap");
        let _ = host_heap.mirror(0, table_handle, true);

        let used_srv = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        let mut state = State::default();
        state.set_heaps(&[views]);
        state.set_table(false, 0, table_handle);

        // A shader declaring 64-byte records at table 0, entry 0, where the
        // title bound a raw view: it gets a structured alias. 48-byte records
        // do not divide the view's 128 bytes, so that one keeps the raw view.
        let shader = |stride| victory_shader::emit::HostShader {
            stage: victory_shader::HwStage::Vs,
            dxil: Vec::new(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            structured_srvs: vec![victory_shader::emit::StructuredSrv {
                slot: 0,
                offset_dwords: 0,
                stride,
            }],
        };
        let aliases = state.raw_structured_aliases(&shader(64), false);
        assert_eq!(
            aliases,
            [StructuredBinding {
                slot: 0,
                entry: 0,
                stride: 64,
            }]
        );
        assert!(state.raw_structured_aliases(&shader(48), false).is_empty());
        let alias = aliases[0];

        let list1 = host.new_list().expect("host list");
        let made_unaliased = state.tables(&list1, &[used_srv], false, &[]);
        let made_aliased = state.tables(&list1, &[used_srv], false, &[alias]);
        let made_aliased_hit = state.tables(&list1, &[], false, &[alias]);
        assert_eq!(made_unaliased.len(), 1);
        assert_eq!(made_aliased.len(), 1);
        assert_ne!(
            made_unaliased[0].1, made_aliased[0].1,
            "aliased and unaliased draws on the same title table must get distinct dynamic tables"
        );
        assert!(
            made_aliased[0].1 >= host_heap.count() * 32,
            "aliased table must live in dynamic slots, leaving canonical slot 0 untouched"
        );
        assert_eq!(made_aliased_hit, made_aliased);

        // Compile a shader verifying the fresh dynamic slot serves StructuredBuffer(64)
        // while canonical slot 0 still serves raw ByteAddressBuffer.
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=1)), DescriptorTable(SRV(t1, numDescriptors=1)), UAV(u0)"
struct Record { uint w[16]; };
StructuredBuffer<Record> aliased : register(t0);
ByteAddressBuffer canonical_raw : register(t1);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint count = 0;
    uint stride = 0;
    aliased.GetDimensions(count, stride);
    dst.Store(0, count);
    dst.Store(4, stride);
    dst.Store(8, aliased[1].w[3]);
    dst.Store(12, canonical_raw.Load(16 * 4 + 3 * 4));
}
"#;
        let mut blob: Option<ID3DBlob> = None;
        // SAFETY: shader source and entry point strings are static.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("draw_fresh_table_alias"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut blob,
                None,
            )
            .unwrap();
        }
        let blob = blob.unwrap();
        // SAFETY: `blob` holds valid DXBC bytes.
        let code = unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        };
        // SAFETY: `code` includes the root signature and compute bytecode.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root = host.device.CreateRootSignature(0, code).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            };
            let pso = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        };

        let make_buffer = |kind: D3D12_HEAP_TYPE, flags, state| -> ID3D12Resource {
            let mut b: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            // SAFETY: valid buffer creation parameters.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut b,
                    )
                    .unwrap();
            }
            b.unwrap()
        };
        let out = make_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let readback = make_buffer(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        // Mutate record 1, dword 3 and issue a cache-hit table lookup on `list2`
        // to confirm cache hits still schedule a fresh backing refresh.
        let updated = 0xDEAD_BEEFu32;
        backing[(16 + 3) * 4..(16 + 3) * 4 + 4].copy_from_slice(&updated.to_le_bytes());

        list1.close();
        host.execute(&[&list1]);

        let list2 = host.new_list().expect("host list");
        let cached = state.tables(&list2, &[], false, &[alias]);
        assert_eq!(cached, made_aliased);
        let aliased_gpu = host_heap.gpu(cached[0].1 / 32);
        let canonical_gpu = host_heap.gpu(0);
        // SAFETY: `list2` is open with live pipeline, heap, and output buffer.
        unsafe {
            let l = &list2.list;
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            l.SetDescriptorHeaps(&[Some(host_heap.heap.clone())]);
            l.SetComputeRootDescriptorTable(0, aliased_gpu);
            l.SetComputeRootDescriptorTable(1, canonical_gpu);
            l.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
            l.Dispatch(1, 1, 1);
            l.ResourceBarrier(&[crate::host::transition(
                &out,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyBufferRegion(&readback, 0, &out, 0, 16);
        }
        list2.close();
        host.execute(&[&list2]);

        let mut mapped = std::ptr::null_mut();
        let mut got = [0u32; 4];
        // SAFETY: `host.execute` completed and readback holds 16 bytes.
        unsafe {
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            readback.Map(0, Some(&range), Some(&mut mapped)).unwrap();
            std::ptr::copy_nonoverlapping(mapped.cast::<u32>(), got.as_mut_ptr(), 4);
            readback.Unmap(0, None);
        }
        assert_eq!(got, [2, 64, updated, updated]);
    }

    /// Direct byte copies into a shader-visible descriptor heap (bypassing
    /// `Create*View` / `CopyDescriptors*`) are mirrored on the host so a GPU
    /// dispatch reading that canonical slot sees the copied view rather than
    /// a null or stale descriptor.
    #[test]
    fn draw_sees_descriptor_copied_directly_in_heap_memory() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC};
        use windows::core::s;

        struct DropHeap(This);
        impl Drop for DropHeap {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
                }
            }
        }

        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };

        // Two distinct title buffers holding known 4-dword patterns.
        let mut buf_a = vec![0u8; 256];
        let mut buf_b = vec![0u8; 256];
        let pat_a = [0x1122_3344u32, 0x5566_7788, 0xAABB_CCDD, 0xCAFE_BABE];
        let pat_b = [0xDEAD_BEEFu32, 0x1337_C0DE, 0x0102_0304, 0xFEDC_BA98];
        for (i, w) in pat_a.iter().enumerate() {
            buf_a[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        for (i, w) in pat_b.iter().enumerate() {
            buf_b[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }

        let make_placed_raw_srv = |buf: &[u8], slot_addr: u64| {
            let base = buf.as_ptr() as u64;
            let mut res_words = [0u32; 14];
            res_words[0] = 1; // D3D12_RESOURCE_DIMENSION_BUFFER
            res_words[4] = buf.len() as u32;
            res_words[6] = 1; // Height
            res_words[7] = 1 | (1 << 16); // DepthOrArraySize=1, MipLevels=1
            res_words[9] = 1; // SampleDesc.Count=1
            res_words[11] = 1; // D3D12_TEXTURE_LAYOUT_ROW_MAJOR
            let raw_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: 0,
                        NumElements: 4,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            };
            // SAFETY: `D3D12_SHADER_RESOURCE_VIEW_DESC` is 40 bytes (`[u32; 10]`).
            let raw_words: [u32; 10] = unsafe {
                std::ptr::from_ref(&raw_srv)
                    .cast::<[u32; 10]>()
                    .read_unaligned()
            };
            let mut srv_words = [0u32; 16];
            srv_words[..10].copy_from_slice(&raw_words);
            srv_words[10] = base as u32;
            srv_words[11] = (base >> 32) as u32;
            crate::descriptor::create_placed_srv(
                std::ptr::null(),
                &res_words,
                &srv_words,
                slot_addr,
            );
        };

        // Write both views into a CPU-only staging buffer (not a shader-visible heap).
        let mut staging = [0u64; 2 * 4];
        let stage_a = staging.as_mut_ptr() as u64;
        let stage_b = stage_a + 32;
        make_placed_raw_srv(&buf_a, stage_a);
        make_placed_raw_srv(&buf_b, stage_b);

        // Create a shader-visible view heap and initialize its host mirror.
        #[repr(C)]
        struct ExtendedHeap {
            desc: [u32; 4],
            placement: [u64; 2],
        }
        let heap_desc = ExtendedHeap {
            desc: [0, 16, 1, 0],
            placement: [0; 2],
        };
        let mut raw_heap: This = std::ptr::null();
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                std::ptr::from_ref(&heap_desc).cast(),
                &crate::object::guid(
                    0x8efb471d,
                    0x616c,
                    0x4f49,
                    [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
                ),
                &mut raw_heap,
            ),
            crate::object::S_OK
        );
        let _guard = DropHeap(raw_heap);
        let title_heap = crate::descriptor::heap(raw_heap);
        let host_heap = title_heap.host().expect("host view heap");
        let target_slot = title_heap.start + 3 * 32;

        // Copy `stage_a` bytes directly into slot 3 without calling any descriptor API.
        // SAFETY: both pointers refer to live 32-byte descriptor slots.
        unsafe { std::ptr::copy_nonoverlapping(stage_a as *const u8, target_slot as *mut u8, 32) };

        // Mirror via the submission-time sync hook and refresh backing memory.
        crate::descriptor::sync_all_heaps();

        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=1)), UAV(u0)"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    dst.Store(0, src.Load(0));
    dst.Store(4, src.Load(4));
    dst.Store(8, src.Load(8));
    dst.Store(12, src.Load(12));
}
"#;
        let mut blob: Option<ID3DBlob> = None;
        // SAFETY: static shader source and entry point strings.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("draw_direct_heap_copy"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut blob,
                None,
            )
            .unwrap();
        }
        let blob = blob.unwrap();
        // SAFETY: `blob` holds valid DXBC bytes.
        let code = unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        };
        // SAFETY: `code` includes the root signature and compute bytecode.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root = host.device.CreateRootSignature(0, code).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            };
            let pso = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        };

        let make_buf = |kind: D3D12_HEAP_TYPE, flags, state| -> ID3D12Resource {
            let mut b: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            // SAFETY: valid committed buffer description.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut b,
                    )
                    .unwrap();
            }
            b.unwrap()
        };
        let out = make_buf(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let readback = make_buf(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        let run_and_read = |backing: &[u8]| -> [u32; 4] {
            let list = host.new_list().expect("host list");
            if let Some(shadow) = crate::shadow::find(backing.as_ptr() as u64, backing.len() as u64)
            {
                shadow.refresh(&list, backing.as_ptr() as u64, backing.len() as u64);
            }
            // SAFETY: open list with live pipeline, host view heap, and buffers.
            unsafe {
                let l = &list.list;
                l.SetComputeRootSignature(&root);
                l.SetPipelineState(&pso);
                l.SetDescriptorHeaps(&[Some(host_heap.heap.clone())]);
                l.SetComputeRootDescriptorTable(0, host_heap.gpu(3));
                l.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
                l.Dispatch(1, 1, 1);
                l.ResourceBarrier(&[crate::host::transition(
                    &out,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                l.CopyBufferRegion(&readback, 0, &out, 0, 16);
                l.ResourceBarrier(&[crate::host::transition(
                    &out,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            list.close();
            host.execute(&[&list]);
            let mut mapped = std::ptr::null_mut();
            let mut got = [0u32; 4];
            // SAFETY: execution completed and `readback` holds 16 bytes.
            unsafe {
                let range = D3D12_RANGE { Begin: 0, End: 16 };
                readback.Map(0, Some(&range), Some(&mut mapped)).unwrap();
                std::ptr::copy_nonoverlapping(mapped.cast::<u32>(), got.as_mut_ptr(), 4);
                readback.Unmap(0, None);
            }
            got
        };

        assert_eq!(run_and_read(&buf_a), pat_a);

        // Overwrite slot 3 directly with `stage_b` bytes and sync again.
        // SAFETY: both pointers refer to live 32-byte descriptor slots.
        unsafe { std::ptr::copy_nonoverlapping(stage_b as *const u8, target_slot as *mut u8, 32) };
        assert_eq!(title_heap.sync_direct_copies(true), 1);
        assert_eq!(run_and_read(&buf_b), pat_b);
    }

    #[test]
    fn few_keeps_items_in_order_inline_and_on_spill() {
        for n in 0..10u32 {
            let mut few: super::Few<u32, 4> = super::Few::default();
            (0..n).for_each(|i| few.push(n - i));
            assert_eq!(few.to_vec(), (0..n).map(|i| n - i).collect::<Vec<_>>());
            few.sort_unstable();
            assert_eq!(few.to_vec(), (1..=n).collect::<Vec<_>>());
        }
    }

    #[test]
    fn bound_targets_compares_inline_and_depth_keys() {
        let mut a = super::BoundTargets::new();
        let mut b = super::BoundTargets::new();
        assert_eq!(a, b);
        let k1 = (0x1000usize, (0x2000usize, [1, 28, 4, 0, 0, 0, 0]));
        let k2 = (0x3000usize, (0x4000usize, [1, 28, 4, 1, 0, 0, 0]));
        a.push(k1);
        assert_ne!(a, b);
        b.push(k1);
        assert_eq!(a, b);
        a.dsv = Some(k2);
        assert_ne!(a, b);
        b.dsv = Some(k2);
        assert_eq!(a, b);
        // Pushing past 8 RTVs spills cleanly into `extra`.
        for i in 0..9usize {
            let k = (0x1000 + i, (0x2000 + i, [0; 7]));
            a.push(k);
            b.push(k);
        }
        assert_eq!(a, b);
    }

    #[test]
    fn set_targets_bind_cache_skips_redundant_binds_and_clears_on_split() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let list1 = host.new_list().expect("host list 1");
        let list2 = host.new_list().expect("host list 2");
        let mut state = super::State::default();

        // Two typed null RTV descriptors with distinct formats.
        let mut rtv_a = [0u64; 16];
        let mut rtv_b = [0u64; 16];
        let h_a = rtv_a.as_mut_ptr() as u64;
        let h_b = rtv_b.as_mut_ptr() as u64;
        let desc_a = [28u32, 4, 0, 0, 0, 0];
        let desc_b = [10u32, 4, 0, 0, 0, 0];
        crate::descriptor::create_rtv(std::ptr::null(), std::ptr::null(), &desc_a, h_a);
        crate::descriptor::create_rtv(std::ptr::null(), std::ptr::null(), &desc_b, h_b);

        state.set_targets(&list1, &[h_a], 0);
        let first_key = state.host_targets.clone().expect("cached bound targets");
        assert_eq!(state.bound_rtvs.len(), 1);

        // Identical bind hits cache and preserves cached key without mutating state.
        state.set_targets(&list1, &[h_a], 0);
        assert_eq!(state.host_targets.as_ref(), Some(&first_key));

        // Changing target invalidates cache and binds the new target.
        state.set_targets(&list1, &[h_b], 0);
        assert_ne!(state.host_targets.as_ref(), Some(&first_key));

        // Mid-list split (`reapply_host`) clears `host_targets` so the new list re-binds.
        state.reapply_host(&list2, &list1);
        assert!(state.host_targets.is_none());
        state.set_targets(&list2, &[h_b], 0);
        assert!(state.host_targets.is_some());

        list1.close();
        list2.close();
    }

    /// Regression test for `VICTORY_LIST_VIEWS`: SRV -> UAV -> SRV across
    /// distinct fresh tables in the same recording must still emit the
    /// UAV -> SRV barrier when `shadow_state_gen` advances (even though
    /// `copy_gen` is unchanged).
    #[test]
    fn listed_buffer_srv_emits_uav_to_srv_barrier_across_fresh_tables() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC};
        use windows::core::s;

        struct DropHeap(This);
        impl Drop for DropHeap {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
                }
            }
        }
        struct UnregShadow(usize, u64);
        impl Drop for UnregShadow {
            fn drop(&mut self) {
                crate::shadow::unregister(self.0, self.1);
            }
        }

        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };

        let compile = |source: &str| -> (ID3D12RootSignature, ID3D12PipelineState) {
            let mut blob: Option<ID3DBlob> = None;
            let mut errors: Option<ID3DBlob> = None;
            // SAFETY: valid static HLSL source and entry point strings.
            unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    s!("listed_srv_uav_srv"),
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut blob,
                    Some(&mut errors),
                )
                .unwrap();
            }
            let blob = blob.unwrap();
            // SAFETY: compiled DXBC with embedded root signature.
            let code = unsafe {
                std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                )
            };
            // SAFETY: valid DXBC root signature and compute PSO creation.
            unsafe {
                let root: ID3D12RootSignature = host.device.CreateRootSignature(0, code).unwrap();
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                };
                let pso: ID3D12PipelineState =
                    host.device.CreateComputePipelineState(&desc).unwrap();
                (root, pso)
            }
        };

        let (read_root, read_pso) = compile(
            r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=1)), UAV(u0), RootConstants(num32BitConstants=1, b0)"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
cbuffer Params : register(b0) { uint out_off; };
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    dst.Store(out_off, src.Load(0));
}
"#,
        );
        let (write_root, write_pso) = compile(
            r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=1))"
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    dst.Store(0, 0xDEADBEEFu);
}
"#,
        );

        let make_buffer = |kind: D3D12_HEAP_TYPE, flags, state| -> ID3D12Resource {
            let mut b: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            // SAFETY: valid buffer creation parameters.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut b,
                    )
                    .unwrap();
            }
            b.unwrap()
        };
        let out = make_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let readback = make_buffer(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        let backing = vec![0x1111_1111u32; 1024];
        let base = backing.as_ptr() as u64;
        let _unreg = UnregShadow(base as usize, base);
        crate::shadow::register(base as usize, base, 4096);
        let shadow = crate::shadow::find(base, 64).expect("registered shadow");

        #[repr(C)]
        struct ExtendedHeap([u32; 4], [u64; 2]);
        let ext = ExtendedHeap([0, 8, 1, 0], [0, 0]);
        let mut heap_ptr: This = std::ptr::null();
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                (&raw const ext.0).cast(),
                &crate::object::guid(
                    0x8efb471d,
                    0x616c,
                    0x4f49,
                    [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
                ),
                &mut heap_ptr,
            ),
            crate::object::S_OK
        );
        let _drop_heap = DropHeap(heap_ptr);
        let title_heap = crate::descriptor::heap(heap_ptr);
        let host_heap = title_heap.host().expect("host view heap");
        let start = title_heap.start;

        let mut res_words = [0u32; 14];
        res_words[0] = 1; // BUFFER
        res_words[4] = 4096;

        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 16,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: 16,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        let mut srv_words = [0u32; 16];
        let mut uav_words = [0u32; 16];
        // SAFETY: both D3D12 view descriptions are 40 bytes (`[u32; 10]`).
        unsafe {
            srv_words[..10].copy_from_slice(
                &std::ptr::from_ref(&srv_desc)
                    .cast::<[u32; 10]>()
                    .read_unaligned(),
            );
            uav_words[..10].copy_from_slice(
                &std::ptr::from_ref(&uav_desc)
                    .cast::<[u32; 10]>()
                    .read_unaligned(),
            );
        }
        srv_words[10] = base as u32;
        srv_words[11] = (base >> 32) as u32;
        uav_words[10] = base as u32;
        uav_words[11] = (base >> 32) as u32;

        // Table A (slots 0..=1): raw SRV + CBV1
        crate::descriptor::create_placed_srv(std::ptr::null(), &res_words, &srv_words, start);
        crate::descriptor::create_cbv(
            std::ptr::null(),
            &[base as u32, (base >> 32) as u32, 256, 0],
            start + 32,
        );
        // Table B (slot 2): raw UAV of the same buffer
        crate::descriptor::create_placed_uav(
            std::ptr::null(),
            &res_words,
            &uav_words,
            start + 2 * 32,
        );
        // Table C (slots 3..=4): identical SRV record as slot 0 + distinct CBV2 so
        // `fresh_tables` misses and `make_listed` reuses the cached SRV entry.
        crate::descriptor::create_placed_srv(
            std::ptr::null(),
            &res_words,
            &srv_words,
            start + 3 * 32,
        );
        let cbv2_addr = base + 256;
        crate::descriptor::create_cbv(
            std::ptr::null(),
            &[cbv2_addr as u32, (cbv2_addr >> 32) as u32, 256, 0],
            start + 4 * 32,
        );

        let srv_table_used = [
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Cbv,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            },
        ];
        let uav_used = [UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }];

        let list = host.new_list().expect("host list");
        let mut state = super::State::default();
        state.set_heaps(&[heap_ptr]);

        // Pass 1 (SRV via fresh table A -> populates `ListViews` cache and sets GENERIC_READ).
        state.set_table(true, 0, start);
        let made_a = state.tables(&list, &srv_table_used, true, &[]);
        assert_eq!(made_a.len(), 1);
        assert_eq!(
            list.shadow_state_of(&shadow.buffer),
            Some(D3D12_RESOURCE_STATE_GENERIC_READ)
        );
        let gen_after_p1 = list.shadow_state_gen();
        assert!(gen_after_p1 > 0);
        let srv_a_gpu = host_heap.gpu(made_a[0].1 / 32);

        // SAFETY: `list` is open with live PSO, heap, and resources.
        unsafe {
            let l = &list.list;
            l.SetDescriptorHeaps(&[Some(host_heap.heap.clone())]);
            l.SetComputeRootSignature(&read_root);
            l.SetPipelineState(&read_pso);
            l.SetComputeRootDescriptorTable(0, srv_a_gpu);
            l.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            l.SetComputeRoot32BitConstant(2, 0, 0);
            l.Dispatch(1, 1, 1);
        }

        // Pass 2 (UAV on the same shadow -> transitions GENERIC_READ -> UNORDERED_ACCESS;
        // `copy_gen` stays 0 while `shadow_state_gen` advances).
        state.set_table(true, 0, start + 2 * 32);
        let made_uav = state.tables(&list, &uav_used, true, &[]);
        assert!(made_uav.is_empty());
        assert_eq!(list.copy_gen(), 0);
        assert_eq!(
            list.shadow_state_of(&shadow.buffer),
            Some(D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        );
        let gen_after_p2 = list.shadow_state_gen();
        assert!(gen_after_p2 > gen_after_p1);

        // SAFETY: `list` is open with live write PSO and canonical UAV slot 2.
        unsafe {
            let l = &list.list;
            l.SetComputeRootSignature(&write_root);
            l.SetPipelineState(&write_pso);
            l.SetComputeRootDescriptorTable(0, host_heap.gpu(2));
            l.Dispatch(1, 1, 1);
        }

        // Pass 3 (SRV via fresh table C with identical SRV record -> `make_listed` hits
        // `ListViews`, sees `shadow_state_gen` changed, re-runs `readable()`, and emits
        // the required UNORDERED_ACCESS -> GENERIC_READ barrier).
        state.set_table(true, 0, start + 3 * 32);
        let made_c = state.tables(&list, &srv_table_used, true, &[]);
        assert_eq!(made_c.len(), 1);
        assert_ne!(made_c[0].1, made_a[0].1, "table C must be a fresh table");
        assert_eq!(
            list.shadow_state_of(&shadow.buffer),
            Some(D3D12_RESOURCE_STATE_GENERIC_READ),
            "listed buffer SRV must transition shadow back to GENERIC_READ after UAV write"
        );
        assert!(list.shadow_state_gen() > gen_after_p2);
        let srv_c_gpu = host_heap.gpu(made_c[0].1 / 32);

        // SAFETY: `list` is open with live read PSO and output buffer.
        unsafe {
            let l = &list.list;
            l.SetComputeRootSignature(&read_root);
            l.SetPipelineState(&read_pso);
            l.SetComputeRootDescriptorTable(0, srv_c_gpu);
            l.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            l.SetComputeRoot32BitConstant(2, 4, 0);
            l.Dispatch(1, 1, 1);
            l.ResourceBarrier(&[crate::host::transition(
                &out,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyBufferRegion(&readback, 0, &out, 0, 8);
        }

        let iq = host.device.cast::<ID3D12InfoQueue>().ok();
        let before_msgs = iq
            .as_ref()
            // SAFETY: plain InfoQueue query.
            .map_or(0, |q| unsafe { q.GetNumStoredMessages() });

        list.close();
        host.execute(&[&list]);

        if let Some(iq) = iq {
            // SAFETY: inspects D3D12 debug messages added during execution.
            unsafe {
                let after = iq.GetNumStoredMessages();
                for idx in before_msgs.min(after)..after {
                    let mut size = 0usize;
                    if iq.GetMessage(idx, None, &mut size).is_err() || size == 0 {
                        continue;
                    }
                    let mut raw = vec![0u64; size.div_ceil(8)];
                    let msg = raw.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    if iq.GetMessage(idx, Some(msg), &mut size).is_ok() {
                        let m = &*msg;
                        let desc =
                            std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                        let text = String::from_utf8_lossy(desc);
                        assert!(
                            m.Severity.0 > D3D12_MESSAGE_SEVERITY_ERROR.0,
                            "unexpected D3D12 error {}: {text}",
                            m.ID.0
                        );
                    }
                }
            }
        }

        let mut mapped = std::ptr::null_mut();
        let mut got = [0u32; 2];
        // SAFETY: execution completed and readback holds 8 bytes.
        unsafe {
            let range = D3D12_RANGE { Begin: 0, End: 8 };
            readback.Map(0, Some(&range), Some(&mut mapped)).unwrap();
            std::ptr::copy_nonoverlapping(mapped.cast::<u32>(), got.as_mut_ptr(), 2);
            readback.Unmap(0, None);
        }
        assert_eq!(got, [0x1111_1111, 0xDEAD_BEEF]);

        list.reset();
        assert_eq!(list.shadow_state_gen(), 0);
        // SAFETY: plain device health query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn parse_gather_mode_defaults_on_and_handles_overrides() {
        use std::ffi::OsStr;
        assert_eq!(super::parse_gather_mode(None), 1);
        assert_eq!(super::parse_gather_mode(Some(OsStr::new("1"))), 1);
        assert_eq!(super::parse_gather_mode(Some(OsStr::new("on"))), 1);
        assert_eq!(super::parse_gather_mode(Some(OsStr::new("0"))), 0);
        assert_eq!(super::parse_gather_mode(Some(OsStr::new(" 0 "))), 0);
        assert_eq!(super::parse_gather_mode(Some(OsStr::new("false"))), 0);
        assert_eq!(super::parse_gather_mode(Some(OsStr::new("no"))), 0);
        assert_eq!(super::parse_gather_mode(Some(OsStr::new("off"))), 0);
        assert_eq!(super::parse_gather_mode(Some(OsStr::new("alt"))), 2);
        assert_eq!(super::parse_gather_mode(Some(OsStr::new(" ALT "))), 2);
    }

    /// Soundness regression test for `State::note_gather_writes`:
    /// 1. A storing stage writing only a resolved table buffer UAV poisons only
    ///    that buffer (`id_a`), leaving unmodified argument buffers (`id_b`)
    ///    gatherable.
    /// 2. When ANY table UAV handle fails to resolve (`descriptor::view(handle)`
    ///    is `None`), `note_gather_writes` MUST conservatively close gathering
    ///    globally — even when another UAV in the same stage did resolve — so
    ///    an unmirrored UAV binding can never bypass gather invalidation.
    #[test]
    fn note_gather_writes_closes_globally_on_unresolved_table_uav() {
        use crate::pipeline::Shader;
        use crate::pipeline::host::{HostPipeline, Stage};
        use victory_shader::HwStage;
        let Some(host) = crate::host::host() else {
            return;
        };
        let cs_src = r#"
            [RootSignature("RootFlags(0)")]
            [numthreads(1, 1, 1)]
            void main() {}
        "#;
        let cs_bytes = crate::hlsl::compile_compute("gather_gate_test_cs", cs_src)
            .expect("compile test compute shader");
        // SAFETY: valid compute shader blob with embedded root signature.
        let (cs_root, cs_pso) = unsafe {
            let root: ID3D12RootSignature = host
                .device
                .CreateRootSignature(0, &cs_bytes)
                .expect("test root signature");
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: cs_bytes.as_ptr().cast(),
                        BytecodeLength: cs_bytes.len(),
                    },
                    ..Default::default()
                })
                .expect("test compute pso");
            (root, pso)
        };

        #[repr(C)]
        struct ViewHeapDesc {
            desc: D3D12_DESCRIPTOR_HEAP_DESC,
            placement: [u64; 2],
        }
        let heap_desc = ViewHeapDesc {
            desc: D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 8,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            placement: [0; 2],
        };
        let mut heap_ptr = std::ptr::null();
        let iid = ID3D12DescriptorHeap::IID;
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                (&raw const heap_desc).cast(),
                (&raw const iid).cast(),
                &mut heap_ptr,
            ),
            crate::object::S_OK
        );
        let _heap = TitleObject(heap_ptr);
        let start = crate::descriptor::heap(heap_ptr).start;

        // Register two distinct title buffers (`buf_a` and `buf_b`) with shadows.
        let buf_a = vec![0u8; 4096];
        let buf_b = vec![0u8; 4096];
        let addr_a = buf_a.as_ptr() as u64;
        let addr_b = buf_b.as_ptr() as u64;
        let res_a = addr_a as usize;
        let res_b = addr_b as usize;
        crate::shadow::register(res_a, addr_a, buf_a.len() as u64);
        let _guard_a = ShadowGuard {
            resource: res_a,
            address: addr_a,
        };
        crate::shadow::register(res_b, addr_b, buf_b.len() as u64);
        let _guard_b = ShadowGuard {
            resource: res_b,
            address: addr_b,
        };
        let shadow_a = crate::shadow::find(addr_a, 256).expect("shadow A");
        let shadow_b = crate::shadow::find(addr_b, 256).expect("shadow B");
        let (id_a, id_b) = (shadow_a.id(), shadow_b.id());
        assert_ne!(id_a, id_b);

        // Slot 0 (dword offset 0) holds a valid placed raw buffer UAV for `buf_a`.
        // Slot 1 (dword offset 8, byte offset 32) is left unpopulated so
        // `crate::descriptor::view(start + 32)` returns `None`.
        let mut resource = [0u32; 14];
        resource[0] = 1; // D3D12_RESOURCE_DIMENSION_BUFFER
        resource[4] = 256;
        let mut words = [0u32; 16];
        words[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS.0 as u32;
        words[1] = D3D12_UAV_DIMENSION_BUFFER.0 as u32;
        words[4] = 64; // NumElements (64 dwords = 256 bytes)
        words[8] = D3D12_BUFFER_UAV_FLAG_RAW.0 as u32;
        words[10] = addr_a as u32;
        words[11] = (addr_a >> 32) as u32;
        crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &words, start);
        assert!(crate::descriptor::view(start).is_some());
        assert!(crate::descriptor::view(start + 32).is_none());

        let uav_at = |offset_dwords: u32| UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: offset_dwords,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        let make_cs_pipeline = |used: Vec<UsedResource>| HostPipeline {
            structured_strides: Vec::new(),
            compute: true,
            mesh: false,
            rects: false,
            amplify: None,
            pso: cs_pso.clone(),
            root: cs_root.clone(),
            stages: vec![Stage {
                shader: Shader::for_test("compute", HwStage::Cs, used),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                fallback: false,
            }],
            slots: 0,
            inputs: Vec::new(),
            name: "gather_gate_test".into(),
            fallback: None,
        };

        let mut state = State::default();
        state.set_heaps(&[heap_ptr]);
        state.set_table(true, 0, start);

        // Case 1: Only the resolved UAV (slot 0 -> `buf_a`) is used.
        // Must close `id_a` while leaving `id_b` gatherable.
        let p_resolved = make_cs_pipeline(vec![uav_at(0)]);
        let list = host.new_list().expect("list 1");
        assert!(list.gather_allows(id_a) && list.gather_allows(id_b));
        state.note_gather_writes(&list, &p_resolved, true);
        assert!(
            !list.gather_allows(id_a),
            "written buffer A must be blocked"
        );
        assert!(
            list.gather_allows(id_b),
            "unmodified buffer B must stay gatherable"
        );

        // Case 2: Only an unresolved UAV handle (slot 1 -> `descriptor::view` == None).
        // Must close gathering globally for both A and B.
        let p_unresolved = make_cs_pipeline(vec![uav_at(8)]);
        let list = host.new_list().expect("list 2");
        assert!(list.gather_allows(id_a) && list.gather_allows(id_b));
        state.note_gather_writes(&list, &p_unresolved, true);
        assert!(
            !list.gather_allows(id_a),
            "unresolved UAV must close gate globally (A)"
        );
        assert!(
            !list.gather_allows(id_b),
            "unresolved UAV must close gate globally (B)"
        );

        // Case 3: Mixed — slot 0 resolves to `buf_a`, slot 1 is unresolved (`None`).
        // Must NOT `continue` past slot 1 and only poison `buf_a`; must close globally!
        let p_mixed = make_cs_pipeline(vec![uav_at(0), uav_at(8)]);
        let list = host.new_list().expect("list 3");
        assert!(list.gather_allows(id_a) && list.gather_allows(id_b));
        state.note_gather_writes(&list, &p_mixed, true);
        assert!(
            !list.gather_allows(id_a),
            "mixed resolved+unresolved UAV must close gate globally (A)"
        );
        assert!(
            !list.gather_allows(id_b),
            "mixed resolved+unresolved UAV must close gate globally (B)"
        );
    }
}
