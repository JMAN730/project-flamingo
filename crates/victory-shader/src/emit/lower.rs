//! IR to DXIL, one machine instruction at a time (decision 0003).
//!
//! Every register dword is an `i32` (SCC an `i1`); float operations bitcast
//! their operands. Lane masks exist only as the per-lane booleans of
//! structurisation; branches and guards are built from them. A value the
//! emitter cannot express is `Missing` with a reason, and only fails the
//! program when something with an effect (an export, a branch) needs it.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::bitcode::{Attr, Bin, Cast, Module, Term, V, pred};
use super::container::SigElement;
use super::meshbuf::MeshBuffer;
use super::shader::{Shader, flags};
use super::{StructuredBinding, UNKNOWN_STRUCTURED_STRIDE, UserData, ViewStride};
use crate::ShaderError;
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::io::{Attribute, Barycentric, Export, Input, Interp, Io, Target};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo, Usage, UsedResource};
use crate::resource::{Address, CODE, MEMORY, Resources, Role, Site, Source};
use crate::structure::{Atom, Bool, LaneExit, Structured};

mod memory_ops;
mod split;
mod table;

/// The ray-tracing node test (`image_bvh_intersect_ray`), a child module so
/// its arithmetic sits apart from the general emitter.
#[path = "intersect.rs"]
pub(super) mod intersect;

/// Title-memory addressing, a child module for the same reason.
#[path = "lower_memory.rs"]
mod lower_memory;

type Why = &'static str;

/// What a barycentric register read as data is missing, until the last
/// chance reads the host's `SV_Barycentrics`.
const BARYCENTRICS_READ: Why = "barycentrics read directly";

/// What an indexed buffer access addresses with its index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexedStride {
    Raw,
    Structured(u32),
}

#[derive(Debug, Clone, Copy)]
enum Val {
    I32(V),
    Bit(V),
    Missing(Why),
}

pub(crate) struct Lowered {
    pub dxil: Vec<u8>,
    /// Bit n: immediate sampler n is used for comparisons.
    pub cmp_samplers: u64,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    pub root_uavs: Vec<u8>,
    /// Table entries read through structured buffer handles.
    pub structured_srvs: Vec<super::StructuredSrv>,
    /// Table entries read through heap handles, with their properties.
    pub heap_reads: Vec<super::HeapRead>,
    pub user_data: Vec<UserData>,
    pub samplers: Vec<[u32; 4]>,
    /// A mesh program emitted as a compute shader: the buffer it fills.
    pub mesh_buffer: Option<MeshBuffer>,
    #[cfg(test)]
    pub has_isnan_op: bool,
    #[cfg(test)]
    pub has_fcmp_uno: bool,
    #[cfg(test)]
    pub has_fcmp_oeq: bool,
    /// Whether the program emits `fcmp false` or `fcmp true`.
    #[cfg(test)]
    pub has_fcmp_constant: bool,
    /// Whether the program reads the pixel coverage mask.
    #[cfg(test)]
    pub has_coverage_op: bool,
    /// Whether a pixel helper lane's `v_readfirstlane` selects its own value.
    #[cfg(test)]
    pub has_helper_lane_op: bool,
    /// Whether the program ballots active lanes (`waveActiveBallot`).
    #[cfg(test)]
    pub has_wave_ballot: bool,
    /// Whether the program emits `dx.op.discard` (pixel survival).
    #[cfg(test)]
    pub has_discard_op: bool,
    /// Test builds: the `dx.op` calls before optimization, see
    /// [`Module::dx_op_calls`].
    #[cfg(test)]
    pub dx_ops: Vec<(String, Vec<Option<i32>>)>,
    /// Heap handles made with a non-uniform index.
    #[cfg(test)]
    pub nonuniform_handles: u32,
    /// Heap handles annotated as constant buffers.
    #[cfg(test)]
    pub cbuffer_handles: u32,
    /// Whether the program uses wave operations.
    #[cfg(test)]
    pub wave_ops: bool,
    /// Test builds: typed loads assembled from a dword-aligned pair.
    #[cfg(test)]
    pub unaligned_typed_loads: u32,
    /// Test builds: d16 format loads that kept the destination's high half.
    #[cfg(test)]
    pub d16_kept_high_halves: u32,
    /// Test builds: constant byte offsets of `s_buffer_load_dword*`.
    #[cfg(test)]
    pub smem_buffer_bytes: Vec<u32>,
    /// Test builds: indices into the address list of image instructions that
    /// were read.
    #[cfg(test)]
    pub image_address_reads: Vec<usize>,
    /// Test builds: whether a 16-bit float is widened (`legacyF16ToF32`).
    #[cfg(test)]
    pub widens_f16: bool,
    /// Test builds: whether i32 immediate `0x3F80_0000` (f32 1.0 bits) is present.
    #[cfg(test)]
    pub has_f32_one_imm: bool,
    /// Test builds: whether i32 immediate `0x3C00` (f16 1.0 bits) is present.
    #[cfg(test)]
    pub has_f16_one_imm: bool,
    /// Test builds: ballots of active masks taken after the loop that wrote
    /// them (issue #465), not at the write site inside the loop.
    #[cfg(test)]
    pub mask_ballots_after_loop: u32,
}

#[derive(Clone)]
pub(crate) struct Parts<'a> {
    pub native: &'a NativeProgram,
    pub p: &'a Program,
    pub st: &'a Structured,
    pub io: &'a Io,
    pub res: &'a Resources,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
    /// Mesh: emit a compute shader that stores the outputs in a buffer
    /// (`super::meshbuf`), for hosts without mesh shaders.
    pub mesh_buffer: bool,
}

/// Retry from the wave-uniform structure when neither half can run alone.
pub(crate) const CONSOLE_FLOW: &str = "a wave64 program whose halves exchange lanes (console flow)";

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum WaveMode {
    Original,
    Wave32,
    ConsoleFlow,
}

enum Stop {
    /// A phi whose incoming value has no DXIL form was read: redo with it missing.
    Retry(Value, Why),
    Fail(ShaderError),
}

const UNTRACKED_SELECT: Why = "a select on an untracked mask";

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    // As `emit` tries them, less the threading of edges: the last chance when
    // the others have refused.
    lower_with_wave32(parts, true, None, None, false)
        .or_else(|refusal| lower_with_wave32(parts, true, None, None, true).map_err(|_| refusal))
}

pub(crate) fn lower_with_wave32(
    parts: &Parts,
    allow_wave32: bool,
    pixel_inputs: Option<&[SigElement]>,
    view_stride: Option<ViewStride<'_>>,
    leftovers: bool,
) -> Result<Lowered, ShaderError> {
    lower_with_wave_mode(
        parts,
        allow_wave32,
        pixel_inputs,
        view_stride,
        WaveMode::Original,
        leftovers,
    )
}

pub(crate) fn lower_with_wave_mode(
    parts: &Parts,
    allow_wave32: bool,
    pixel_inputs: Option<&[SigElement]>,
    view_stride: Option<ViewStride<'_>>,
    wave_mode: WaveMode,
    leftovers: bool,
) -> Result<Lowered, ShaderError> {
    // A failed optional lowering must preserve the ordinary translation (and its
    // refusal reason, if any). First try the two-register-bank form only for a
    // single complete, straight-line 64-lane compute wave.
    if wave_mode != WaveMode::ConsoleFlow
        && allow_wave32
        && parts.native.wave64
        && matches!(parts.native.io, StageIo::Cs(cs) if cs.threads == [64, 1, 1])
        && double_lanes_enabled()
        && can_double_lane_reads(parts.p)
        && let Ok(shader) = lower_program(
            parts,
            false,
            false,
            leftovers,
            allow_wave32,
            true,
            pixel_inputs,
            view_stride,
            wave_mode,
        )
    {
        return Ok(shader);
    }
    let attempt = |second: bool| match lower_program(
        parts,
        false,
        second,
        leftovers,
        allow_wave32,
        false,
        pixel_inputs,
        view_stride,
        wave_mode,
    ) {
        Err(ShaderError::Unsupported {
            what: UNTRACKED_SELECT,
            ..
        }) if parts.native.stage == HwStage::Cs => lower_program(
            parts,
            true,
            second,
            leftovers,
            allow_wave32,
            false,
            pixel_inputs,
            view_stride,
            wave_mode,
        ),
        result => result,
    };
    // What translation does only for programs it would refuse otherwise (so
    // that every other program's code stays as it was) is a second chance.
    attempt(false).or_else(|refusal| {
        if crate::switch::no_second_pass() {
            return Err(refusal);
        }
        attempt(true).map_err(|_| refusal)
    })
}

#[allow(clippy::too_many_arguments)]
fn lower_program(
    parts: &Parts,
    numeric_selects: bool,
    second: bool,
    leftovers: bool,
    allow_wave32: bool,
    doubled_lane_reads: bool,
    pixel_inputs: Option<&[SigElement]>,
    view_stride: Option<ViewStride<'_>>,
    wave_mode: WaveMode,
) -> Result<Lowered, ShaderError> {
    let mut bad = HashMap::new();
    loop {
        match Lower::new(
            parts,
            &bad,
            numeric_selects,
            second,
            leftovers,
            allow_wave32,
            doubled_lane_reads,
            view_stride,
            wave_mode,
        )
        .run(pixel_inputs)
        {
            Ok(l) => return Ok(l),
            Err(Stop::Retry(v, why)) => {
                if bad.insert(v, why).is_some() {
                    return Err(ShaderError::Unsupported {
                        offset: 0,
                        what: "a phi that does not settle",
                    });
                }
            }
            Err(Stop::Fail(e)) => return Err(e),
        }
    }
}

const TWO_PI: f32 = std::f32::consts::TAU;
/// `s_sendmsg` id of a mesh or NGG stage's output allocation request.
const GS_ALLOC_REQ: u16 = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// A GFX10 buffer format (the unified format of typed buffer loads): the
/// channel widths from the lowest bits up, and how the channels convert.
/// Formats are numbered in families (`8`, `16`, `8_8`, ...: AMD names
/// list channels from the highest bits), each with its numeric formats in
/// the order unorm, snorm, uscaled, sscaled, uint, sint, float.
fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    // Numbers and order from LLVM's `UfmtGFX10` (`SIDefines.h`); the GFX10
    // buffer formats have no 5_9_9_9. Channel order from Mesa
    // (`ac_formats.c`): R11G11B10_FLOAT is 10_11_11, and 10_10_10_2 has
    // its 2-bit channel in the lowest bits.
    const FAMILIES: [(&[u32], &[Numeric]); 14] = [
        (&[8], NORM),
        (&[16], ALL),
        (&[8, 8], NORM),
        (&[32], WIDE),
        (&[16, 16], ALL),
        (&[11, 11, 10], ALL),     // 10_11_11
        (&[10, 11, 11], ALL),     // 11_11_10
        (&[2, 10, 10, 10], NORM), // 10_10_10_2
        (&[10, 10, 10, 2], NORM), // 2_10_10_10
        (&[8, 8, 8, 8], NORM),
        (&[32, 32], WIDE),
        (&[16, 16, 16, 16], ALL),
        (&[32, 32, 32], WIDE),
        (&[32, 32, 32, 32], WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

/// Whether handle properties `props` are a structured SRV's: kind 12 in
/// the low byte, whatever its alignment bits (bits 8 and up, as
/// `buffer_props` sets for strides other than 4); not a UAV's (bit 12).
fn structured_srv(props: (u32, u32)) -> bool {
    props.0 & 0xFF == 12 && props.0 & 0x1000 == 0
}

/// How far a float channel of `width` bits (16, or the unsigned 11- and
/// 10-bit floats of the packed formats) shifts up to be a half:
/// the small floats are a half's exponent (5 bits, biased 15) and the top
/// of its mantissa, without the sign. Non-NaN values widen as a half;
/// NaNs keep their payload via [`Lower::f16_widen_keep_snan`].
fn half_shift(width: u32) -> u32 {
    debug_assert!(matches!(width, 10 | 11 | 16), "{width}-bit float");
    if width == 16 { 0 } else { 15 - width }
}

/// f32 bits for an f16 NaN without quieting: `sign | 0x7F800000 |
/// (payload << 13)`. RDNA typed-buffer float formats preserve the
/// signalling bit; `legacyF16ToF32` would set it (VIC-BT-019 / #440).
#[cfg(test)]
fn f16_nan_widen_bits(h: u32) -> u32 {
    let h = h & 0xFFFF;
    debug_assert!(h & 0x7FFF > 0x7C00, "{h:#x} is not an f16 NaN");
    ((h & 0x8000) << 16) | 0x7F80_0000 | ((h & 0x3FF) << 13)
}

/// A cached descriptor handle: block, the load site, its first dword, whether
/// it is a sampler, and the resource properties.
type HandleKey = (usize, Site, u8, bool, (u32, u32));

/// Source element id, element-relative channel, value, and optional f16 half.
type ExportStore = (usize, u8, Value, Option<bool>);

struct Lower<'a> {
    s: Shader,
    x: &'a Parts<'a>,
    /// Original to retained element ids for mesh vertex and primitive exports.
    mesh_output_ids: Option<[Vec<Option<usize>>; 2]>,
    view_stride: Option<ViewStride<'a>>,
    bad: &'a HashMap<Value, Why>,
    lane_fail: HashMap<Value, Why>,
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    /// DXIL block per IR block (unreachable: none).
    block: Vec<Option<usize>>,
    cur: usize,
    bools: HashMap<(usize, Bool), V>,
    lane_bits: HashMap<Value, V>,
    /// IR phi location by value.
    phi_at: HashMap<Value, (usize, usize)>,
    mask_phis: HashMap<Value, V>,
    /// SCC values a lane-mask operation wrote (the mask is non-zero), by
    /// value: the block and the mask's lane bit. Read as data (a loop test
    /// over a whole wave), SCC is `WaveActiveAnyTrue` of the lane bits.
    scc_any: HashMap<Value, (usize, Bool)>,
    /// The instruction defining each value, for [`Lower::restored_lane`].
    spill_defs: std::cell::OnceCell<HashMap<Value, (usize, usize)>>,
    /// [`super::demanded`], made when a last chance first asks.
    demanded: std::cell::OnceCell<HashMap<Value, u32>>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    read_phis: HashSet<Value>,
    params: HashMap<Value, (usize, u32)>,
    user_data: Option<V>,
    user_reads: BTreeMap<u32, UserData>,
    samplers: Vec<[u32; 4]>,
    descriptors: HashMap<(Site, bool), &'a Source>,
    atomic_descriptor_loads: HashSet<Site>,
    scalar: HashMap<Site, Address>,
    exports: HashMap<Site, &'a Export>,
    interps: HashMap<Site, &'a Interp>,
    handles: HashMap<HandleKey, V>,
    /// Table entries read through structured buffer handles.
    structured_srvs: Vec<super::StructuredSrv>,
    /// Table entries read through heap handles, with their properties.
    heap_reads: Vec<super::HeapRead>,
    /// Blocks splitting the edge from one IR block to another.
    edges: HashMap<(usize, usize), usize>,
    /// Per IR block: a `done` export is behind every lane entering it.
    exported: Vec<bool>,
    /// Per IR block: a `vm` export is behind every lane entering it.
    vm_exported: Vec<bool>,
    /// Whether a Valid-Mask export lies on some path into each block.
    vm_reached: Vec<bool>,
    /// Per-lane Valid Mask SSA value while lowering a pixel block.
    vm_live: Option<V>,
    /// Phi of `vm_live` at the start of multi-predecessor pixel blocks.
    vm_live_phis: Vec<Option<V>>,
    /// `vm_live` leaving each IR block (pixel shaders).
    vm_live_out: Vec<Option<V>>,
    /// Reverse-postorder index per IR block.
    rpo: Vec<usize>,
    /// DXIL block each IR block ends in (guarded stores split blocks).
    tail: Vec<Option<usize>>,
    /// The program writes UAVs: barriers fence them too.
    writes_uavs: bool,
    /// src0 of the current instruction as DPP fetched it.
    src0: Option<V>,
    /// Compute code that reads other lanes: thread ids follow the host
    /// lanes, so lane n is console lane n.
    console_lanes: bool,
    doubled_lane_reads: bool,
    /// Retry a compute shader with numeric selects only when the normal
    /// translation needs one. Unused selects need no fixed wave size.
    numeric_selects: bool,
    /// A second try at a program the first refused: lowers what the first
    /// leaves out (SCC of a lane mask read as a number, lane reads after a
    /// whole-wave section or from a spill the code cannot follow).
    second: bool,
    /// A third try at a program the second refused too: reads the leftovers
    /// of registers the program reuses (a lane mask's dword where data is
    /// joined, a mesh program's counts of masks, a raw view's size).
    leftovers: bool,
    /// Heap indices are being computed for a lookup in the host's buffer
    /// view table, which lists the title's slots ([`Self::title_slot`]).
    title_slots: bool,
    /// A console wave64 compute program run as two 32-lane host waves
    /// (each lane's reads stay in its half; see `Lower::new`), under a
    /// [`WaveMode`] other than the original. Not [`Self::split`], which
    /// exchanges between the halves in the original mode.
    halves: bool,
    /// Split in halves, and run as the console runs it ([`CONSOLE_FLOW`]).
    wave_mode: WaveMode,
    console_cf: bool,
    /// Halves without [`Self::console_cf`] cannot run it, the console flow can.
    wants_console_flow: bool,
    /// Why a wave64 program that needs its whole wave cannot take the
    /// console flow on a host without 64-lane waves.
    no_console_flow: Option<Why>,
    /// The console thread's flat id in the group (relabelled; see
    /// [`Self::thread_id`]).
    console_flat: Option<V>,
    /// Group-shared slots the two halves exchange a dword through.
    xchg: Option<V>,
    /// Mesh: the thread index in the group (made in the entry block).
    flat: Option<V>,
    /// Mesh stored in a buffer: its layout, and the dword where this
    /// thread's group record starts (made in the entry block).
    buffer: Option<MeshBuffer>,
    buffer_record: Option<V>,
    /// Mesh stored in a buffer: the number of this dispatch's first thread
    /// group (made in the entry block).
    buffer_base: Option<V>,
    /// Mesh: IR blocks whose `s_barrier` is left out.
    no_barrier: HashSet<usize>,
    /// IR blocks whose `s_barrier` is control-dependent on a non-uniform
    /// (per-lane) branch: emitting `SYNC_THREAD_GROUP` (bit 0) there
    /// deadlocks 32-lane hosts when warps diverge, so only the memory fence
    /// bits are emitted.
    fence_only_barrier: HashSet<usize>,
    /// Vertex and mesh outputs holding 16-bit values: row, whether two per
    /// dword, and the row packed after it.
    half_outputs: HashMap<u8, (bool, Option<u8>)>,
    /// How lane masks become data.
    mask_data: MaskData,
    /// The instruction being lowered may read lane masks as data.
    mask_ok: bool,
    /// Ballots of masks read as data, by low dword: (low, high).
    ballots: HashMap<Value, (V, V)>,
    /// Masks whose ballot is being built: a mask whose lane bit needs its
    /// own value as a number (an opaque mask keyed by a dword it shares
    /// with another) would otherwise recurse until the stack overflows.
    balloting: HashSet<Value>,
    /// See [`Lowered::mask_ballots_after_loop`].
    #[cfg(test)]
    mask_ballots_after_loop: u32,
    /// Block writing each value.
    def_block: HashMap<Value, usize>,
    /// IR block being lowered.
    cur_ir: usize,
    /// Whole-wave sections by block and start index.
    sections: HashMap<usize, HashMap<usize, super::wave::Section>>,
    /// Values written in a section that code outside it reads.
    section_out: HashSet<Value>,
    /// Instructions after a section's restore that the section covers:
    /// block and index to the section's start.
    section_tail: HashMap<(usize, usize), usize>,
    /// The program's constant table (the code from the first byte read
    /// through `s_getpc_b64` on): global, first byte, dwords.
    code_table: Option<(V, u32, u32)>,
    /// Loop headers' iteration counters (see [`LOOP_LIMIT`]): the phi by
    /// IR block, and its incoming (value, DXIL block) pairs.
    counters: HashMap<usize, (V, Vec<(V, usize)>)>,
    /// The block a loop stopped by [`LOOP_LIMIT`] leaves the program from.
    abort: Option<usize>,
    /// A console wave64 program run as two 32-lane halves ([`split`]).
    split: Option<split::Split>,
    /// Why the program cannot run on this host at all.
    refuse: Option<Why>,
}

/// Iterations after which a loop is left (per entry into it), unless
/// [`super::loop_limit`] says otherwise. The console runs a loop as long as
/// its data says; data the host has not mirrored yet (a null view reads
/// zeros) can make that four billion iterations, which hangs the host GPU
/// until the driver resets it. A lane that reaches the limit leaves the
/// program as killed (a pixel discards).
// shortcut: a genuine loop of more iterations than this is cut short too.
pub(super) const LOOP_LIMIT: u32 = 1 << 20;

/// Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt`, scalar
/// compares, bit tests) are ballots of the lanes' booleans, taken where the
/// mask is written: structurisation's `active` masks hold only lanes that
/// reach that block, so the ballot misses none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskData {
    /// Compute: host lane n is console lane n (the program declares the
    /// console's wave size), so ballots are the console's masks bit for bit
    /// and any use is exact.
    Console,
    /// Pixel and vertex programs: ballots over the host's wave, whatever its
    /// size. Only uses that mean the same on any wave are allowed: counts,
    /// the first lane, the count below a lane, equality (the HLSL wave
    /// operations the console compiler expands). Lane numbers stay host
    /// lanes throughout (`v_mbcnt`, `v_readlane` at a computed lane).
    Host,
    /// Mesh programs: their lanes and waves (LDS per console wave) are the
    /// console's but no host wave size can be declared. Only counts and
    /// compares of masks are allowed (`s_bcnt1`, scalar compares: the HLSL
    /// wave operations the console compiler expands, which hold on any wave
    /// size); lane numbers and cross-lane operations stay refused, because
    /// host lanes do not follow the console's.
    // shortcut: a mesh program's count covers its host wave's lanes, which
    // may be fewer than its console wave's.
    Count,
    /// Other stages: masks are not read as numbers.
    None,
}

/// What a value holding a lane mask read as a number stops with.
const LANE_MASK: Why = "a lane mask read as data";

/// Granlund-Montgomery unsigned reciprocal `(magic, shift)` for dividing any
/// `x < 2^30` by odd `odd >= 3`: `((x * magic) >> (32 + shift)) == x / odd`.
fn structured_div_magic(odd: u32) -> (u32, u32) {
    debug_assert!(odd >= 3 && odd % 2 == 1 && odd < (1 << 30));
    let shift = 31 - odd.leading_zeros();
    let magic = (1u64 << (32 + shift)).div_ceil(u64::from(odd));
    (magic as u32, shift)
}

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
/// Samples whose level comes from derivatives across each quad of lanes.
/// Clamp and offset tokens (`_cl`, `_o`) keep that; explicit LOD, gradients,
/// and comparison forms do not.
fn implicit_derivatives(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("image_sample") else {
        return false;
    };
    !rest
        .split('_')
        .any(|t| matches!(t, "l" | "lz" | "d" | "cd" | "c" | "g16"))
}

fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// Whether `name` is a scalar logic operation whose SCC is "the result is
/// not zero" (so, for a lane mask, "some lane is set").
fn sets_scc_nonzero(name: &str) -> bool {
    let op = name
        .strip_prefix("s_")
        .and_then(|n| n.strip_suffix("_b32").or_else(|| n.strip_suffix("_b64")));
    matches!(
        op,
        Some("and" | "or" | "xor" | "andn2" | "orn2" | "nand" | "nor" | "xnor" | "not")
    )
}

/// The SCC values written by lane-mask operations (not EXEC writes): each
/// says whether its mask has a lane set.
fn mask_sccs(st: &Structured, p: &Program) -> HashSet<Value> {
    let mut out = HashSet::new();
    for i in p.blocks.iter().flat_map(|b| &b.insts) {
        if matches!(i.fields, Fields::Sop { .. })
            && sets_scc_nonzero(i.name())
            && i.write(Reg::S(EXEC)).is_none()
            && i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && st.masks.contains_key(&w))
            && let Some(scc) = i.write(Reg::Scc)
        {
            out.insert(scc);
        }
    }
    out
}

/// An instruction other than a branch reading the SCC of a lane-mask
/// operation as data (a branch on it is a per-lane condition already).
fn reads_mask_scc(sccs: &HashSet<Value>, i: &Inst) -> bool {
    !i.name().starts_with("s_cbranch_") && i.read(Reg::Scc).is_some_and(|v| sccs.contains(&v))
}

/// A select that needs the actual scalar mask words and console lane index.
fn numeric_select(st: &Structured, inst: &Inst) -> bool {
    let Fields::Valu(v) = &inst.fields else {
        return false;
    };
    let Some(&Operand::Scalar { code, .. }) = v.src.get(2) else {
        return false;
    };
    code != NULL
        && match inst.name() {
            "v_cndmask_b32" => inst
                .writes
                .first()
                .is_some_and(|&(_, v)| !st.select_masks.contains_key(&v)),
            "v_add_co_ci_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => inst
                .read(Reg::S(code))
                .is_none_or(|m| !st.masks.contains_key(&m)),
            _ => false,
        }
}

/// Where half `h` (dword h / 2, the high half when odd) of a parameter
/// packing 16-bit rows `a` and `b` two per dword comes from: a row and
/// channel of `elements`, none for padding. Channel c of `a` is half c;
/// `b` starts at the dword after `a`'s last channel.
fn packed_half(elements: &[SigElement], a: u8, b: Option<u8>, h: u8) -> Option<(u8, u8)> {
    let extent = |r: u8| {
        elements
            .iter()
            .filter(|e| e.row == Some(u32::from(r)))
            .map(|e| e.col + e.cols)
            .max()
    };
    let na = extent(a)?;
    let a_halves = na.div_ceil(2) * 2;
    if h < a_halves {
        return (h < na).then_some((a, h));
    }
    let b = b?;
    let c = h - a_halves;
    (c < extent(b)?).then_some((b, c))
}

/// Vertex or mesh outputs of 16-bit values (see [`Lower::half_outputs`]):
/// a packed row's partner is the next 16-bit row of the signature that no
/// output entry names, taken in row order.
fn half_outputs(io: &StageIo, outputs: &[SigElement]) -> HashMap<u8, (bool, Option<u8>)> {
    let entries = match io {
        StageIo::Vs { outputs, .. } => outputs,
        StageIo::Ms(ms) => &ms.outputs,
        _ => return HashMap::new(),
    };
    let per_vertex = || entries.iter().filter(|o| o.param & 0x80 == 0);
    let mut free: Vec<u8> = outputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !per_vertex().any(|o| o.row == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut halves: Vec<_> = per_vertex().filter(|o| o.fp16).collect();
    halves.sort_by_key(|o| o.row);
    halves
        .into_iter()
        .map(|o| {
            let b = o
                .pack2
                .then(|| free.iter().position(|&r| r > o.row))
                .flatten();
            (o.row, (o.pack2, b.map(|i| free.remove(i))))
        })
        .collect()
}

/// Loads whose eight SGPRs are used only as one image atomic's resource.
/// The atomic rebuilds the heap index from the load address.
fn atomic_descriptor_loads(p: &Program, res: &Resources) -> HashSet<Site> {
    let mut loads = HashSet::new();
    let mut readers = HashMap::new();
    for d in &res.descriptors {
        let Source::Loaded { load, first: 0 } = d.source else {
            continue;
        };
        let atomic = &p.blocks[d.site.0].insts[d.site.1];
        let Fields::Image {
            rsrc: Operand::Scalar { code, dwords: 8 },
            ..
        } = atomic.fields
        else {
            continue;
        };
        let inst = &p.blocks[load.0].insts[load.1];
        if !atomic.name().starts_with("image_atomic_")
            || inst.name() != "s_load_dwordx8"
            || code > crate::ir::VCC - 8
            || inst.writes.len() != 8
        {
            continue;
        }
        if inst.writes.iter().all(|&(reg, v)| {
            matches!(reg, Reg::S(n) if (code..code + 8).contains(&n)) && atomic.read(reg) == Some(v)
        }) {
            loads.insert(load);
            for &(_, v) in &inst.writes {
                readers.insert(v, (load, d.site));
            }
        }
    }
    if !loads.is_empty() {
        for (b, block) in p.blocks.iter().enumerate() {
            for v in block.phis.iter().flat_map(|phi| &phi.args) {
                if let Some((load, _)) = readers.get(v) {
                    loads.remove(load);
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for (_, v) in &inst.reads {
                    if let Some(&(load, reader)) = readers.get(v)
                        && reader != (b, i)
                    {
                        loads.remove(&load);
                    }
                }
            }
        }
    }
    loads
}

fn unsupported(offset: usize, what: Why) -> Stop {
    Stop::Fail(ShaderError::Unsupported { offset, what })
}

/// A full, unpredicated console wave that communicates only by indexed register
/// reads or dword LDS accesses. Scalar memory and descriptor operands stay
/// uniform. Each thread keeps both logical lanes in registers and can
/// broadcast either without reserving any of the program's LDS.
fn can_double_lane_reads(p: &Program) -> bool {
    let [block] = p.blocks.as_slice() else {
        return false;
    };
    if block.exit != Exit::End || !block.phis.is_empty() {
        return false;
    }
    let mut lane_read = false;
    for inst in &block.insts {
        let name = inst.name();
        match &inst.fields {
            Fields::Valu(v) => {
                if v.abs != 0
                    || v.neg != 0
                    || v.clamp
                    || v.omod != 0
                    || v.dpp.is_some()
                    || v.sdwa.is_some()
                    || !matches!(
                        name,
                        "v_readlane_b32"
                            | "v_mov_b32"
                            | "v_add_nc_u32"
                            | "v_lshlrev_b32"
                            | "v_lshrrev_b32"
                            | "v_and_b32"
                            | "v_or_b32"
                            | "v_xor_b32"
                    )
                {
                    return false;
                }
                if name == "v_readlane_b32" {
                    if !matches!(v.dst, Some(Operand::Scalar { code: 0..=105, .. }))
                        || !matches!(
                            v.src.get(1),
                            Some(
                                Operand::Scalar { code: 0..=105, .. }
                                    | Operand::Const(128..=248)
                                    | Operand::Literal(_)
                            )
                        )
                    {
                        return false;
                    }
                    lane_read = true;
                }
            }
            Fields::Sop {
                dst: Some(Operand::Scalar { code: 0..=105, .. }),
                ..
            } if name == "s_mov_b32" => {}
            Fields::Smem { .. }
                if name == "s_load_dwordx4"
                    && inst
                        .writes
                        .iter()
                        .all(|(reg, _)| matches!(reg, Reg::S(0..=105))) => {}
            Fields::Buffer { .. } if name == "buffer_store_dword" => {}
            Fields::Ds { gds: false, .. } if matches!(name, "ds_read_b32" | "ds_write_b32") => {}
            Fields::Sopp { .. } if matches!(name, "s_waitcnt" | "s_endpgm") => {}
            _ => return false,
        }
    }
    lane_read
}

#[cfg(test)]
thread_local! {
    static TEST_DOUBLE_LANES: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

fn double_lanes_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = TEST_DOUBLE_LANES.get() {
        return value;
    }
    super::wave64_double()
}

impl<'a> Lower<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: &'a Parts<'a>,
        bad: &'a HashMap<Value, Why>,
        numeric_selects: bool,
        second: bool,
        leftovers: bool,
        allow_wave32: bool,
        doubled_lane_reads: bool,
        view_stride: Option<ViewStride<'a>>,
        wave_mode: WaveMode,
    ) -> Self {
        let p = x.p;
        let mut phi_at = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phi_at.insert(phi.value, (b, i));
            }
        }
        let mut s = Shader::new(x.native.stage == HwStage::Ps);
        // A POPS pixel program waits for the earlier waves that cover its
        // pixels before its UAV accesses. The host orders a rasterizer-ordered
        // view's accesses between overlapping pixels itself.
        s.rov = matches!(&x.native.io, StageIo::Ps { rsrc2, .. }
            if rsrc2.collision_wave_id_sgpr().is_some());
        s.inputs = x.inputs.clone();
        s.outputs = x.outputs.clone();
        let insts = || p.blocks.iter().flat_map(|b| &b.insts);
        let writes_uavs = insts().any(|i| {
            matches!(i.fields, Fields::Image { .. } | Fields::Buffer { .. })
                && (i.name().contains("store") || i.name().contains("atomic"))
        });
        let mut console_lanes = false;
        let (mut split, mut refuse) = (None, None);
        let mut halves = false;
        let mut wants_console_flow = false;
        let mut no_console_flow = None;
        if let StageIo::Cs(cs) = x.native.io {
            s.threads = Some(cs.threads);
            s.lds_dwords = cs.lds_bytes() / 4;
            // Code reading other lanes needs the console's wave size and
            // lane order; so do lane masks read as numbers, and LDS shared
            // within one wave, where the console compiler leaves out barriers.
            let mask_sccs = if second {
                mask_sccs(x.st, p)
            } else {
                HashSet::new()
            };
            // Swizzles and DPP read lanes of the same 32 (rows of 16, groups
            // of 8, 32 or 4): the rest may read the other half of a wave64.
            let in_half = |i: &Inst| {
                i.name().starts_with("ds_swizzle")
                    || implicit_derivatives(i.name())
                    || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
            };
            let whole = |i: &Inst| {
                let n = i.name();
                n.contains("permute")
                    || n.starts_with("v_permlane")
                    || n.starts_with("v_mbcnt")
                    || matches!(n, "v_readlane_b32" | "v_writelane_b32")
                    || (wave_mode != WaveMode::Original && n == "v_readfirstlane_b32")
                    || reads_mask_as_data(x.st, i)
                    || reads_mask_scc(&mask_sccs, i)
                    || numeric_selects && numeric_select(x.st, i)
            };
            let reads_whole = insts().any(whole);
            let masks_held = {
                // Masks held as numbers or constant lane sets: the lane
                // tests its own bit by its console lane.
                let whens = x.st.exits.iter().filter_map(|e| match *e {
                    LaneExit::Branch { when, .. } => Some(when),
                    _ => None,
                });
                let mut all: Vec<Bool> =
                    x.st.guards
                        .iter()
                        .flatten()
                        .flatten()
                        .copied()
                        .chain(whens)
                        .collect();
                all.sort_unstable();
                all.dedup();
                all.into_iter().any(|g| {
                    x.st.bools
                        .support(g)
                        .iter()
                        .any(|a| matches!(a, Atom::Opaque(_) | Atom::Constant(_)))
                })
            };
            console_lanes = reads_whole || masks_held || insts().any(in_half);
            let wave = if x.native.wave64 { 64 } else { 32 };
            let threads = cs.threads.iter().product::<u32>();
            let one_wave = threads <= wave;
            // A wave64 compute group of exactly 32 threads has no upper-half
            // threads. When whole-wave Active reductions are its only
            // cross-lane requirement (no LDS, no held masks, and no
            // independent lane exchange outside those sections), lowering it
            // with WaveSize(32) matches the 64-lane reduction identically on
            // 32-lane hosts (`thread_id()` reads `s.wave_size = 32`).
            let one_wave32 = allow_wave32
                && x.native.wave64
                && threads == 32
                && s.lds_dwords == 0
                && !masks_held
                && super::wave::single_wave32_reductions(p);
            let needs_wave = console_lanes || one_wave && s.lds_dwords > 0;

            // On a host without 64-lane waves, a wave64 program whose lanes
            // read only their own half runs as two 32-lane waves: what each
            // lane reads is in its host wave, at its host lane's place in the
            // row. Not with LDS (the console compiler may leave out barriers
            // between the halves) nor whole-wave sections (their reductions
            // become wave operations over the host's wave).
            // (`v_readfirstlane` reads the console wave's first lane, which
            // the upper half does not hold: such programs take the console
            // flow below.)
            halves = console_lanes
                && x.native.wave64
                && !reads_whole
                && !insts().any(|i| i.name() == "v_readfirstlane_b32")
                && !masks_held
                && s.lds_dwords == 0
                && wave_mode != WaveMode::Original
                && (0..p.blocks.len()).all(|b| super::wave::sections(p, b).is_empty());
            // The rest of such programs need the halves to exchange values
            // (masks read whole, lanes of the other half) or to share LDS in
            // the console's instruction order: the console flow, for a group
            // of one console wave. What keeps a program out of it is the
            // reason it stops with.
            let needs_wave64 = (console_lanes || one_wave && s.lds_dwords > 0)
                && !halves
                && !one_wave32
                && !doubled_lane_reads
                && x.native.wave64;
            if needs_wave64 && wave_mode != WaveMode::Original {
                no_console_flow = if cs.threads.iter().product::<u32>() > 64 {
                    Some(
                        "console wave64 lanes in a group of more than 64 threads on a host without 64-lane waves",
                    )
                } else if insts().any(|i| i.name().contains("permute")) {
                    Some("ds_permute across the halves of a wave64 on a host without 64-lane waves")
                } else if !(0..p.blocks.len()).all(|b| super::wave::sections(p, b).is_empty()) {
                    Some("a whole-wave section of a wave64 on a host without 64-lane waves")
                } else {
                    None
                };
                wants_console_flow = no_console_flow.is_none();
            }
            if wave_mode == WaveMode::ConsoleFlow {
                console_lanes = true;
                // In halves only where the host cannot run the console wave.
                halves = x.native.wave64;
            }
            if doubled_lane_reads {
                s.threads = Some([32, 1, 1]);
                s.wave_size = Some(32);
            } else if one_wave32 || halves {
                s.wave_size = Some(32);
            } else if needs_wave
                && wave == 64
                && allow_wave32
                && wave_mode == WaveMode::Original
                && !super::host_wave64()
            {
                // The host has no 64-lane waves. A group of at most 32
                // threads has no upper half: the console's lanes 32-63 are
                // off, as a 32-lane host's are absent. Each whole console
                // wave runs as two 32-lane halves.
                s.wave_size = Some(32);
                if threads > 32 {
                    if threads % 64 == 0 {
                        let waves = super::probe::split_waves().unwrap_or(threads / 64);
                        let mut sp = split::Split::new(p, x.st, waves);
                        // Two dwords past the program's LDS take its
                        // accesses out of bounds (see `split_lds_index`).
                        if s.lds_dwords > 0 {
                            sp.lds_bound = s.lds_dwords;
                            s.lds_dwords += 2;
                        }
                        refuse = sp.refusal(s.lds_dwords);
                        split = Some(sp);
                        console_lanes = true;
                    } else {
                        // shortcut: a partial console wave above 32 threads
                        // has an upper half whose missing lanes the halves'
                        // exchanges would wait for; such programs stay
                        // refused on these hosts.
                        refuse = Some("a partial console wave64 group on a 32-lane host");
                    }
                }
            } else if needs_wave {
                s.wave_size = Some(wave);
            }
        }
        let mut no_barrier = HashSet::new();
        let mut buffer = None;
        if let StageIo::Ms(ms) = &x.native.io {
            s.prims = x.prims.clone();
            s.threads = Some([ms.threads, 1, 1]);
            s.lds_dwords = ms.lds_bytes.div_ceil(4);
            let mesh = super::shader::Mesh {
                max_vertices: ms.max_vertices,
                max_primitives: ms.max_primitives,
                topology: ms.topology,
            };
            s.mesh = Some(mesh);
            if x.mesh_buffer {
                s.mesh_buffer = true;
                buffer = Some(MeshBuffer::new(mesh, &x.outputs, &x.prims));
            }
            // Barriers order the lanes' LDS accesses; the console's
            // allocation request needs them too, but the host does not. One
            // lane sends that request, and what it dominates (the path for
            // counts out of range) runs in that lane alone, so its barrier
            // would hang the host.
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut index = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                index[b] = n;
            }
            let idom = crate::ir::idoms(&p.blocks, &order, &index);
            let alloc: Vec<usize> = (0..p.blocks.len())
                .filter(|&b| {
                    p.blocks[b].insts.iter().any(|i| {
                        i.name() == "s_sendmsg"
                            && matches!(i.fields, Fields::Sopp { imm } if imm & 0xF == GS_ALLOC_REQ)
                    })
                })
                .collect();
            for &b in &order {
                let mut d = b;
                let dominated = loop {
                    if alloc.contains(&d) {
                        break true;
                    }
                    if idom[d] == d || idom[d] == usize::MAX {
                        break false;
                    }
                    d = idom[d];
                };
                if ms.lds_bytes == 0 || dominated {
                    no_barrier.insert(b);
                }
            }
        }
        // Console branches dispatch the same edge to the whole thread group.
        let fence_only_barrier = if wave_mode == WaveMode::ConsoleFlow {
            HashSet::new()
        } else {
            divergent_barriers(p, x.st, &no_barrier)
        };
        Lower {
            s,
            x,
            mesh_output_ids: None,
            view_stride,
            bad,
            lane_fail: HashMap::new(),
            vals: HashMap::new(),
            floats: HashMap::new(),
            block: vec![None; p.blocks.len()],
            cur: 0,
            bools: HashMap::new(),
            lane_bits: HashMap::new(),
            phi_at,
            mask_phis: HashMap::new(),
            scc_any: HashMap::new(),
            spill_defs: std::cell::OnceCell::new(),
            demanded: std::cell::OnceCell::new(),
            pending_masks: Vec::new(),
            data_phis: Vec::new(),
            read_phis: HashSet::new(),
            params: x.res.params.iter().map(|&(v, p, d)| (v, (p, d))).collect(),
            user_data: None,
            user_reads: BTreeMap::new(),
            samplers: Vec::new(),
            atomic_descriptor_loads: atomic_descriptor_loads(p, x.res),
            descriptors: x
                .res
                .descriptors
                .iter()
                .map(|d| ((d.site, d.role == Role::Sampler), &d.source))
                .collect(),
            scalar: x.res.scalar.iter().copied().collect(),
            exports: x.io.exports.iter().map(|e| (e.site, e)).collect(),
            interps: x.io.interps.iter().map(|(s, i)| (*s, i)).collect(),
            handles: HashMap::new(),
            structured_srvs: Vec::new(),
            heap_reads: Vec::new(),
            edges: HashMap::new(),
            exported: vec![false; p.blocks.len()],
            vm_exported: vec![false; p.blocks.len()],
            vm_reached: vec![false; p.blocks.len()],
            vm_live: None,
            vm_live_phis: vec![None; p.blocks.len()],
            vm_live_out: vec![None; p.blocks.len()],
            rpo: vec![usize::MAX; p.blocks.len()],
            tail: vec![None; p.blocks.len()],
            writes_uavs,
            src0: None,
            console_lanes,
            doubled_lane_reads,
            numeric_selects,
            second,
            leftovers,
            title_slots: false,
            halves,
            wave_mode,
            console_cf: wave_mode == WaveMode::ConsoleFlow,
            wants_console_flow,
            no_console_flow,
            console_flat: None,
            xchg: None,
            flat: None,
            buffer,
            buffer_record: None,
            buffer_base: None,
            no_barrier,
            fence_only_barrier,
            half_outputs: half_outputs(&x.native.io, &x.outputs),
            mask_data: match x.native.stage {
                HwStage::Cs => MaskData::Console,
                HwStage::Ps | HwStage::Vs => MaskData::Host,
                // A third chance only (see `lower_with_wave32`): the first
                // two leave a program's code as it was.
                HwStage::Ms if leftovers => MaskData::Count,
                _ => MaskData::None,
            },
            mask_ok: false,
            ballots: HashMap::new(),
            balloting: HashSet::new(),
            #[cfg(test)]
            mask_ballots_after_loop: 0,
            def_block: p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| {
                    block
                        .insts
                        .iter()
                        .flat_map(move |i| i.writes.iter().map(move |w| (w.1, b)))
                        .chain(block.phis.iter().map(move |phi| (phi.value, b)))
                })
                .collect(),
            cur_ir: 0,
            sections: HashMap::new(),
            section_out: HashSet::new(),
            section_tail: HashMap::new(),
            code_table: None,
            counters: HashMap::new(),
            abort: None,
            split,
            refuse,
        }
    }

    /// The dword of the program's code at byte `byte` (a constant table
    /// the compiler appends to the code), as an internal constant array.
    // shortcut: an index outside the table reads its last dword (the
    // console would read whatever memory follows the code).
    fn code_dword(&mut self, byte: V) -> Result<V, Why> {
        let (g, base, len) = match self.code_table {
            Some(t) => t,
            None => {
                let joins = &self.x.res.code_joins;
                let joined = |a: &Address| {
                    a.dynamic
                        .is_some_and(|d| joins.iter().any(|j| j.address == d))
                };
                let first = self
                    .x
                    .res
                    .scalar
                    .iter()
                    .map(|s| s.1)
                    .chain(self.x.res.descriptors.iter().flat_map(|d| {
                        d.source.leaves().into_iter().filter_map(|l| match l {
                            Source::Built { address, .. } => Some(*address),
                            _ => None,
                        })
                    }))
                    .filter(|a| a.param == CODE && !joined(a))
                    .map(|a| a.offset)
                    .chain(joins.iter().map(|j| j.lowest))
                    .map(|offset| offset & !3)
                    .min()
                    .ok_or("a code table")?;
                let code = &self.x.native.code;
                let words: Vec<u32> = code
                    .get(first as usize..)
                    .ok_or("a code table past the code")?
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        u32::from_le_bytes(w)
                    })
                    .collect();
                if words.is_empty() {
                    return Err("a code table past the code");
                }
                let elems: Vec<V> = words.iter().map(|&w| self.s.m.i32(w)).collect();
                let arr = self.s.m.array(words.len() as u32, self.s.t.i32);
                let init = self.s.m.aggregate(arr, &elems);
                let g = self.s.m.constant_global("code", init);
                let t = (g, first, words.len() as u32);
                self.code_table = Some(t);
                t
            }
        };
        let c = self.cur;
        let k = self.s.m.i32(base);
        let rel = self.s.m.bin(c, Bin::Sub, byte, k);
        let two = self.s.m.i32(2);
        let index = self.s.m.bin(c, Bin::LShr, rel, two);
        let last = self.s.m.i32(len - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let at = self.s.m.element(c, g, index);
        Ok(self.s.m.load(c, at))
    }

    /// A global memory load (`global_load_*`): from the program's constant
    /// table, or from a root SRV at the address the title sets.
    fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if self.scalar.get(&(b, i)).is_some_and(|a| a.param == MEMORY) {
            return self.memory_flat(inst, b, i);
        }
        let rest = name
            .strip_prefix("global_load_")
            .ok_or("a global memory write")?;
        let (n, sub) = match rest {
            "dword" => (1, None),
            "dwordx2" => (2, None),
            "dwordx3" => (3, None),
            "dwordx4" => (4, None),
            "ubyte" => (1, Some((8, false))),
            "sbyte" => (1, Some((8, true))),
            "ushort" => (1, Some((16, false))),
            "sshort" => (1, Some((16, true))),
            _ => return Err(name),
        };
        let a = *self
            .scalar
            .get(&(b, i))
            .ok_or("an untracked global address")?;
        let guard = self.guard(b, i)?;
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        let c = self.cur;
        let mut off = self.s.m.i32(a.offset);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            off = self.s.m.bin(c, Bin::Add, off, d);
        }
        let h = if a.param == CODE {
            None
        } else {
            let src = Source::Built {
                address: a,
                bits: 0,
                words: [0, 0],
                records: None,
            };
            Some(self.handle((b, i), &src, false, (11, 0))?)
        };
        let dword = |l: &mut Self, byte: V| -> Result<V, Why> {
            match h {
                None => l.code_dword(byte),
                Some(h) => {
                    let c = l.cur;
                    let (u, mask, al) = (l.s.m.undef(l.s.t.i32), l.s.m.i8(1), l.s.m.i32(4));
                    let r = l.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        l.s.t.res_ret,
                        139,
                        &[h, byte, u, mask, al],
                    );
                    let x = l.s.m.extract(c, r, 0);
                    Ok(l.as_int(x))
                }
            }
        };
        for k in 0..n {
            let x = match sub {
                None => {
                    let four = self.s.m.i32(4 * k);
                    let at = self.s.m.bin(c, Bin::Add, off, four);
                    dword(self, at)?
                }
                Some((width, signed)) => {
                    // shortcut: the field is taken not to cross a dword.
                    let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
                    let at = self.s.m.bin(c, Bin::And, off, not3);
                    let w = dword(self, at)?;
                    let byte = self.s.m.bin(c, Bin::And, off, three);
                    let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                    let x = self.s.m.bin(c, Bin::LShr, w, sh);
                    let left = self.s.m.i32(32 - width);
                    let x = self.s.m.bin(c, Bin::Shl, x, left);
                    self.s
                        .m
                        .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left)
                }
            };
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// Finds the whole-wave sections and the values they leave.
    fn find_sections(&mut self) {
        let p = self.x.p;
        let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
        for b in 0..p.blocks.len() {
            let found = super::wave::sections_with(p, b, self.second);
            for (&start, sec) in &found {
                // The readlanes after the restore belong to the section.
                let limit = if sec.tail > 0 {
                    sec.end + sec.tail + 1
                } else {
                    sec.end
                };
                for i in start + 1..limit {
                    for &(_, w) in &p.blocks[b].insts[i].writes {
                        inside.insert(w, (b, start, limit));
                    }
                }
                for i in sec.end + 1..sec.end + 1 + sec.tail {
                    self.section_tail.insert((b, i), start);
                }
            }
            if !found.is_empty() {
                self.sections.insert(b, found);
            }
        }
        if inside.is_empty() {
            return;
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                self.section_out
                    .extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && !(sb == b && i > start && i < end)
                    {
                        self.section_out.insert(r);
                    }
                }
            }
        }
    }

    /// An instruction inside a whole-wave section: the values that leave
    /// the section are made from what the simulation found them to be.
    fn section_inst(&mut self, b: usize, i: usize, start: usize) {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        // Split: a lane mask the section computes from others (the lanes that
        // were active, for the reduction to read) keeps its exact bits.
        if self.split.is_some()
            && matches!(inst.fields, Fields::Sop { .. })
            && inst.write(Reg::S(EXEC)).is_none()
            && matches!(self.split_sop(inst), Some(Ok(())))
        {
            return;
        }
        for &(_, w) in &inst.writes {
            let val = if self.section_out.contains(&w) {
                let shape = self.sections[&b][&start].shapes.get(&w).copied();
                match shape.ok_or(UNKNOWN).and_then(|sh| self.shape(sh)) {
                    Ok(x) => Val::I32(x),
                    Err(why) => Val::Missing(why),
                }
            } else {
                Val::Missing(UNKNOWN)
            };
            self.vals.insert(w, val);
        }
    }

    /// A value a whole-wave section leaves, as DXIL wave operations over the
    /// host's active lanes: the console's result with console lanes; on
    /// another wave size, what the HLSL wave operation the console compiler
    /// expanded gives on that wave.
    fn shape(&mut self, shape: super::wave::Shape) -> Result<V, Why> {
        use super::wave::Shape;
        let (op, src, mask, ident, prefix) = match shape {
            Shape::K(k) => return Ok(self.s.m.i32(k)),
            Shape::Value(v) => return self.i32_of(v),
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Masked(src, mask, ident) => {
                let c = self.cur;
                let on = self.section_mask(mask)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                return Ok(self.s.m.select(c, on, x, id));
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
        };
        if matches!(self.mask_data, MaskData::None | MaskData::Count) {
            return Err("a wave reduction in a mesh shader");
        }
        if self.halves {
            return Err("a wave reduction in a wave64 program split in halves");
        }
        let c = self.cur;
        let on = self.section_mask(mask)?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);
        let r = self.host_wave_op(op, x, id, prefix)?;
        if self.split.is_none() {
            return Ok(r);
        }
        // Split: this half's result, with the other half's.
        Ok(self.split_reduce(op, x, r, prefix.is_some(), |l| {
            l.host_wave_op(op, x, id, None)
                .expect("a reduction of a prefix's operation")
        }))
    }

    /// The lane bit of a whole-wave section's mask.
    fn section_mask(&mut self, mask: Value) -> Result<V, Why> {
        let c = self.cur;
        if self.split.is_some() {
            return self.exact_pair(mask, self.pair_high(mask), c);
        }
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        self.boolean(m, c)
    }

    /// `op` (or its prefix) over the host wave's active lanes of `x`, whose
    /// identity is `id`.
    fn host_wave_op(
        &mut self,
        op: super::wave::Op,
        x: V,
        id: V,
        prefix: Option<bool>,
    ) -> Result<V, Why> {
        use super::wave::Op;
        let c = self.cur;
        self.s.flags |= flags::WAVE_OPS;
        if op == Op::AddH {
            // The console adds 16-bit floats in the order of its expansion,
            // rounding to f16 at each add; a host wave sum has an order of
            // its own (and an f32 sum rounds once), which can give another
            // result (2048 + 1 - 2048: 0 on the console, 1 summed in f32).
            return Err(if prefix.is_some() {
                "a wave prefix of 16-bit floats"
            } else {
                "a whole-wave sum of 16-bit floats"
            });
        }
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        if let (Op::AddF, Some(inclusive)) = (op, prefix) {
            let r = self.float_prefix_sum(x, id, inclusive);
            return Ok(self.as_int(r));
        }
        let arg = if float { self.as_float(x) } else { x };
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        let r = match (op, prefix) {
            (Op::AddU, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    "wavePrefixOp.i32",
                    Attr::NoUnwind,
                    i32t,
                    121,
                    &[arg, k0, k1],
                );
                if inclusive {
                    self.s.m.bin(c, Bin::Add, r, arg)
                } else {
                    r
                }
            }
            (Op::Or | Op::And | Op::Xor, Some(inclusive)) => {
                // DXIL has no plain bitwise prefix; WaveMultiPrefixOp over
                // one partition holding every active lane is one (exclusive,
                // the identity in the first lane). Inactive lanes already
                // hold the identity.
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let yes = self.s.m.bool(true);
                let all = self
                    .s
                    .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[yes]);
                let parts: Vec<V> = (0..4).map(|k| self.s.m.extract(c, all, k)).collect();
                let (kind, bin) = match op {
                    Op::And => (1, Bin::And),
                    Op::Or => (2, Bin::Or),
                    _ => (3, Bin::Xor),
                };
                let (k, u) = (self.s.m.i8(kind), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    "waveMultiPrefixOp.i32",
                    Attr::NoUnwind,
                    i32t,
                    166,
                    &[arg, parts[0], parts[1], parts[2], parts[3], k, u],
                );
                if inclusive {
                    self.s.m.bin(c, bin, r, arg)
                } else {
                    r
                }
            }
            (_, Some(_)) => return Err("a wave prefix of a minimum or maximum"),
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s
                    .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, k, u],
                )
            }
        };
        Ok(if float { self.as_int(r) } else { r })
    }

    /// `Op::AddF` prefix sum over active host lanes from cross-lane reads
    /// instead of `wavePrefixOp.f32`, which on AMD Radeon drivers corrupts a
    /// lane whose own term is NaN, infinity or large enough to round away
    /// earlier terms.
    fn float_prefix_sum(&mut self, x: V, id: V, inclusive: bool) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let me = self.lane_index();
        let t = self.s.m.bool(true);
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[t]);
        let (words, rounds) = match self.s.wave_size {
            Some(32) => (1, 5),
            Some(64) => (2, 6),
            _ => (4, 7),
        };
        let (z, one, k5, k31, none) = (
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(5),
            self.s.m.i32(31),
            self.s.m.i32(u32::MAX),
        );
        let word_idx = self.s.m.bin(c, Bin::LShr, me, k5);
        let bit_idx = self.s.m.bin(c, Bin::And, me, k31);
        let low_mask = self.s.m.bin(c, Bin::Shl, one, bit_idx);
        let low_mask = self.s.m.bin(c, Bin::Sub, low_mask, one);
        let mut prev = none;
        for j in 0..words {
            let w = self.s.m.extract(c, ballot, j);
            let m = if words == 1 {
                low_mask
            } else {
                let kj = self.s.m.i32(j);
                let below = self.s.m.cmp(c, pred::IULT, kj, word_idx);
                let same = self.s.m.cmp(c, pred::EQ, kj, word_idx);
                let m = self.s.m.select(c, same, low_mask, z);
                self.s.m.select(c, below, none, m)
            };
            let mw = self.s.m.bin(c, Bin::And, w, m);
            let clz = self
                .s
                .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[mw]);
            let mut cand = self.s.m.bin(c, Bin::Sub, k31, clz);
            if j > 0 {
                let base = self.s.m.i32(32 * j);
                cand = self.s.m.bin(c, Bin::Or, cand, base);
            }
            let has = self.s.m.cmp(c, pred::NE, mw, z);
            prev = self.s.m.select(c, has, cand, prev);
        }
        let (mut acc, mut p) = if inclusive {
            (self.as_float(x), prev)
        } else {
            let has = self.s.m.cmp(c, pred::NE, prev, none);
            let at = self.s.m.select(c, has, prev, me);
            let px = self.read_lane(x, at);
            let pp = self.read_lane(prev, at);
            let x0 = self.s.m.select(c, has, px, id);
            let has2 = self.s.m.cmp(c, pred::NE, pp, none);
            let p0 = self.s.m.select(c, has2, prev, none);
            (self.as_float(x0), p0)
        };
        for _ in 0..rounds {
            let has = self.s.m.cmp(c, pred::NE, p, none);
            let at = self.s.m.select(c, has, p, me);
            let ai = self.as_int(acc);
            let other = self.read_lane(ai, at);
            let other = self.as_float(other);
            let next = self.read_lane(p, at);
            let sum = self.s.m.bin(c, Bin::Add, other, acc);
            acc = self.s.m.select(c, has, sum, acc);
            p = self.s.m.select(c, has, next, none);
        }
        acc
    }

    fn run(mut self, pixel_inputs: Option<&[SigElement]>) -> Result<Lowered, Stop> {
        if self.wants_console_flow && !self.console_cf {
            return Err(unsupported(0, CONSOLE_FLOW));
        }
        if let Some(n) = self.s.wave_size
            && self.wave_mode != WaveMode::Original
            && n != 32
        {
            // shortcut: no form of these lanes on the host's wave size yet
            // (two 32-lane waves standing for one console wave64 need the
            // halves to exchange masks and lane values).
            return Err(unsupported(
                0,
                match (n, self.console_lanes) {
                    (64, _) if self.no_console_flow.is_some() => {
                        self.no_console_flow.expect("checked")
                    }
                    (64, true) => "console wave64 lanes on a host without 64-lane waves",
                    (64, false) => {
                        "a wave64 group sharing LDS without barriers on a host without 64-lane waves"
                    }
                    _ => "a console wave size the host does not run",
                },
            ));
        }
        let p = self.x.p;
        if let Some(why) = self.refuse {
            return Err(unsupported(0, why));
        }
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        for (n, &b) in order.iter().enumerate() {
            self.block[b] = Some(if n == 0 {
                self.s.entry
            } else {
                self.s.m.new_block()
            });
        }
        // Loop headers count iterations (`LOOP_LIMIT`).
        for &b in &order {
            let header = p.blocks[b]
                .preds
                .iter()
                .any(|&q| self.block[q].is_some() && self.rpo[b] <= self.rpo[q]);
            if header {
                let blk = self.block[b].expect("reachable");
                let phi = self.s.m.phi(blk, self.s.t.i32);
                self.counters.insert(b, (phi, Vec::new()));
            }
        }
        self.cur = self.s.entry;
        self.find_sections();
        self.inputs()?;
        if self.split.is_some() {
            self.split_inputs();
        }
        if let Some(mesh) = self.s.mesh {
            self.mesh_prologue(mesh).map_err(|w| unsupported(0, w))?;
            self.prune_mesh_outputs(mesh, pixel_inputs)?;
            self.block[order[0]] = Some(self.cur);
        }
        for &b in &order {
            self.cur = self.block[b].expect("reachable");
            self.cur_ir = b;
            let preds: Vec<usize> = p.blocks[b]
                .preds
                .iter()
                .copied()
                .filter(|&q| self.block[q].is_some())
                .collect();
            self.exported[b] = !preds.is_empty() && preds.iter().all(|&q| self.exported_out(q));
            self.vm_exported[b] =
                !preds.is_empty() && preds.iter().all(|&q| self.vm_exported_out(q));
            self.vm_reached[b] = preds
                .iter()
                .any(|&q| self.vm_reached[q] || self.has_vm_export(q));
            if self.s.pixel {
                self.vm_live = Some(if preds.is_empty() {
                    // Alive until the first Valid-Mask export.
                    self.s.m.bool(true)
                } else {
                    let phi = self.s.m.phi(self.cur, self.s.t.i1);
                    self.vm_live_phis[b] = Some(phi);
                    phi
                });
            }
            for (i, phi) in p.blocks[b].phis.iter().enumerate() {
                let val = match phi.reg {
                    Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                    _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                    Reg::Scc => {
                        let v = self.s.m.phi(self.cur, self.s.t.i1);
                        self.data_phis.push((v, b, i));
                        Val::Bit(v)
                    }
                    _ => {
                        let v = self.s.m.phi(self.cur, self.s.t.i32);
                        self.data_phis.push((v, b, i));
                        Val::I32(v)
                    }
                };
                self.vals.insert(phi.value, val);
                if self.split.is_some() {
                    self.split_phi(b, i);
                }
            }
            if self.split.is_some() {
                self.split_phi_numbers();
            }
            if self.doubled_lane_reads {
                self.doubled_reads_block(b)?;
            } else {
                let mut i = 0;
                while i < p.blocks[b].insts.len() {
                    if let Some(&start) = self.section_tail.get(&(b, i)) {
                        self.section_inst(b, i, start);
                        i += 1;
                        continue;
                    }
                    self.inst(b, i)?;
                    if let Some(end) = self.sections.get(&b).and_then(|s| s.get(&i)).map(|s| s.end)
                    {
                        for j in i + 1..end {
                            self.section_inst(b, j, i);
                        }
                        i = end;
                        continue;
                    }
                    i += 1;
                }
            }
            self.mask_ok = self.mask_data == MaskData::Console;
            self.tail[b] = Some(self.cur);
            if self.s.pixel {
                self.vm_live_out[b] = self.vm_live;
            }
            self.exit(b)?;
        }
        self.fill_phis()?;
        if self.split.is_some() {
            self.split_fill_phis().map_err(|w| unsupported(0, w))?;
        }
        if self.xchg.is_some() && self.s.lds_dwords > 8190 {
            return Err(unsupported(
                0,
                "no group-shared space for the cross-half exchange",
            ));
        }
        self.s.prune_pixel_inputs();
        let user_data = self.user_reads.values().copied().collect();
        self.s.samplers = self.samplers.len() as u32;
        let root_cbvs = self.s.root_cbvs.iter().map(|r| r.0).collect();
        let root_srvs = self.s.root_srvs.iter().map(|r| r.0).collect();
        let root_uavs = self.s.root_uavs.iter().map(|r| r.0).collect();
        let cmp_samplers = self.s.cmp_samplers;
        #[cfg(test)]
        let has_isnan_op = self.s.m.has_dx_op("dx.op.isSpecialFloat.f32", 8);
        #[cfg(test)]
        let has_fcmp_uno = self.s.m.has_cmp(pred::UNO);
        #[cfg(test)]
        let has_fcmp_oeq = self.s.m.has_cmp(pred::OEQ);
        #[cfg(test)]
        let has_fcmp_constant = self.s.m.has_cmp(pred::FALSE) || self.s.m.has_cmp(pred::TRUE);
        #[cfg(test)]
        let has_coverage_op = self.s.m.has_dx_op("dx.op.coverage.i32", 91);
        #[cfg(test)]
        let has_helper_lane_op = self.s.m.has_dx_op("dx.op.isHelperLane.i1", 221);
        #[cfg(test)]
        let has_wave_ballot = self.s.m.has_dx_op("dx.op.waveActiveBallot", 116);
        #[cfg(test)]
        let has_discard_op = self.s.m.has_dx_op("dx.op.discard", 82);
        #[cfg(test)]
        let dx_ops = self.s.m.dx_op_calls();
        Ok(Lowered {
            #[cfg(test)]
            unaligned_typed_loads: self.s.unaligned_typed_loads,
            #[cfg(test)]
            d16_kept_high_halves: self.s.d16_kept_high_halves,
            #[cfg(test)]
            smem_buffer_bytes: std::mem::take(&mut self.s.smem_buffer_bytes),
            #[cfg(test)]
            image_address_reads: std::mem::take(&mut self.s.image_address_reads),
            #[cfg(test)]
            widens_f16: self.s.m.has_dx_op("dx.op.legacyF16ToF32", 131),
            #[cfg(test)]
            has_f32_one_imm: self.s.m.has_i32_const(i64::from(0x3F80_0000u32 as i32)),
            #[cfg(test)]
            has_f16_one_imm: self.s.m.has_i32_const(0x3C00),
            #[cfg(test)]
            mask_ballots_after_loop: self.mask_ballots_after_loop,
            cmp_samplers,
            root_cbvs,
            root_srvs,
            root_uavs,
            structured_srvs: std::mem::take(&mut self.structured_srvs),
            heap_reads: std::mem::take(&mut self.heap_reads),
            #[cfg(test)]
            nonuniform_handles: self.s.nonuniform_handles,
            #[cfg(test)]
            cbuffer_handles: self.s.cbuffer_handles,
            #[cfg(test)]
            wave_ops: self.s.flags & flags::WAVE_OPS != 0,
            dxil: self.s.finish(),
            user_data,
            samplers: self.samplers,
            mesh_buffer: self.buffer,
            #[cfg(test)]
            has_isnan_op,
            #[cfg(test)]
            has_fcmp_uno,
            #[cfg(test)]
            has_fcmp_oeq,
            #[cfg(test)]
            has_fcmp_constant,
            #[cfg(test)]
            has_coverage_op,
            #[cfg(test)]
            has_helper_lane_op,
            #[cfg(test)]
            has_wave_ballot,
            #[cfg(test)]
            has_discard_op,
            #[cfg(test)]
            dx_ops,
        })
    }

    /// Lower the straight-line block admitted by `can_double_lane_reads`.
    /// Host lane i owns both console lanes i and i + 32. Scalar instructions
    /// execute once, vector instructions consume and write each register bank.
    fn doubled_reads_block(&mut self, b: usize) -> Result<(), Stop> {
        self.s.flags |= flags::WAVE_OPS;
        let mut upper = self.vals.clone();
        for &(v, input) in &self.x.io.inputs {
            if matches!(input, Input::ThreadIdInGroup(0)) {
                let lane = self.lane_index();
                let offset = self.s.m.i32(32);
                let high_id = self.s.m.bin(self.cur, Bin::Add, lane, offset);
                upper.insert(v, Val::I32(high_id));
            }
        }
        for (i, inst) in self.x.p.blocks[b].insts.iter().enumerate() {
            if inst.name() == "v_readlane_b32" {
                let offset = inst.machine.offset;
                let lo = self.src(inst, 0).map_err(|why| unsupported(offset, why))?;
                let lane = self.src(inst, 1).map_err(|why| unsupported(offset, why))?;
                std::mem::swap(&mut self.vals, &mut upper);
                let hi = self.src(inst, 0).map_err(|why| unsupported(offset, why))?;
                std::mem::swap(&mut self.vals, &mut upper);
                let mask = self.s.m.i32(31);
                let at = self.s.m.bin(self.cur, Bin::And, lane, mask);
                let lo = self.read_lane(lo, at);
                let hi = self.read_lane(hi, at);
                let bit = self.s.m.i32(32);
                let half = self.s.m.bin(self.cur, Bin::And, lane, bit);
                let zero = self.s.m.i32(0);
                let high = self.s.m.cmp(self.cur, pred::NE, half, zero);
                let value = self.s.m.select(self.cur, high, hi, lo);
                for &(_, dst) in &inst.writes {
                    self.vals.insert(dst, Val::I32(value));
                    upper.insert(dst, Val::I32(value));
                }
            } else if matches!(
                inst.fields,
                Fields::Valu(_) | Fields::Buffer { .. } | Fields::Ds { .. }
            ) {
                self.inst(b, i)?;
                std::mem::swap(&mut self.vals, &mut upper);
                self.inst(b, i)?;
                std::mem::swap(&mut self.vals, &mut upper);
                if matches!(inst.fields, Fields::Ds { .. }) {
                    // Both halves must finish this LDS instruction before the
                    // next can overwrite or consume another lane's result.
                    let mode = self.s.m.i32(9); // group sync and groupshared fence
                    self.s.op(
                        self.cur,
                        "barrier",
                        Attr::NoDuplicate,
                        self.s.t.void,
                        80,
                        &[mode],
                    );
                }
            } else {
                self.inst(b, i)?;
                for &(_, dst) in &inst.writes {
                    if let Some(&value) = self.vals.get(&dst) {
                        upper.insert(dst, value);
                    }
                }
            }
        }
        Ok(())
    }

    fn exported_out(&self, b: usize) -> bool {
        self.exported[b] || self.has_done_export(b)
    }

    fn has_done_export(&self, b: usize) -> bool {
        (0..self.x.p.blocks[b].insts.len())
            .any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.done))
    }

    fn vm_exported_out(&self, b: usize) -> bool {
        self.vm_exported[b] || self.has_vm_export(b)
    }

    fn has_vm_export(&self, b: usize) -> bool {
        (0..self.x.p.blocks[b].insts.len()).any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.vm))
    }

    /// Values of the program's input registers.
    fn inputs(&mut self) -> Result<(), Stop> {
        let io_inputs: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        let x = self.x;
        let written = |r: Reg| {
            x.p.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| i.writes.iter().any(|&(w, _)| w == r))
        };
        for &(reg, v) in &self.x.p.inputs {
            let val = match reg {
                Reg::S(_) if self.params.contains_key(&v) => continue, // loaded on first use
                Reg::S(n) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Above the SGPRs the hardware initializes, a register
                    // holds what the last wave left, so no program depends
                    // on it: a read the paths only seem to allow (a branch
                    // past the write, taken only with one past the read)
                    // reads zero, as VGPRs do. A register no instruction
                    // writes is no such path but more likely an input the
                    // decoding does not know about (a system SGPR in another
                    // position, say), so it stays refused.
                    None if n <= 105 && self.x.io.initialized_sgprs.is_some_and(|k| n >= k) => {
                        if written(reg) {
                            Val::I32(self.s.m.i32(0))
                        } else {
                            Val::Missing(
                                "an SGPR no instruction writes, above the initialized ones",
                            )
                        }
                    }
                    // A scalar parked in a VGPR lane (`ir::unspill`), read where
                    // the paths only seem to allow it before any save.
                    None if n >= crate::ir::PARKED => Val::I32(self.s.m.i32(0)),
                    None => Val::Missing("a hardware SGPR input"),
                },
                Reg::Scc => Val::Missing("SCC on entry"),
                Reg::V(_) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Registers no one set read as zero (the validator
                    // refuses reads of undef).
                    None => Val::I32(self.s.m.i32(0)),
                },
            };
            self.vals.insert(v, val);
        }
        Ok(())
    }

    fn input(&mut self, input: Input) -> Val {
        let (kind, col) = match input {
            Input::VertexId => (1, 0),
            Input::InstanceId => (2, 0),
            Input::Position(c) => (3, c),
            Input::Attribute { row, channel } => {
                return match self.find_input(row, channel) {
                    Some((id, col)) => self.load_input(id, col),
                    None => Val::Missing("a vertex input the signature lacks"),
                };
            }
            Input::Barycentric { .. } => return Val::Missing(BARYCENTRICS_READ),
            Input::PrimitiveId => return Val::Missing("the primitive id"),
            // The console code tests it against 0 (`v_cmp_ne_u32`), as the
            // host's value can be.
            Input::FrontFace => (13, 0),
            Input::ThreadIdInGroup(c) => return self.thread_id(c),
            Input::GroupId(c) => return self.compute_id("groupId.i32", 94, c),
            Input::GroupInfo => return Val::Missing("the thread-group info SGPR"),
            // M0 for `v_interp`, which takes the primitive's parameters from
            // the signature here, so no lowering reads the value. A phi can
            // still join it with the register's reuse on another path (an
            // address's high dword set to 0) and needs a value then.
            Input::PrimMask => return Val::I32(self.s.m.i32(0)),
            // Bit 31 clear: the wave overlaps no earlier one, so it does not
            // wait. Its UAV handles are rasterizer ordered instead.
            Input::CollisionWaveId => return Val::I32(self.s.m.i32(0)),
            Input::MeshGroup => {
                let g = self.mesh_group();
                let n = self.s.m.i32(self.x.mesh_stride);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Mul, g, n));
            }
            Input::MergedWaveInfo => {
                // shortcut: only the wave index (bits 24-27) is set; the
                // console's other bits (lane counts) are read by no program.
                let f = self.mesh_thread();
                let shift = self.s.m.i32(self.wave().trailing_zeros());
                let w = self.s.m.bin(self.s.entry, Bin::LShr, f, shift);
                let k24 = self.s.m.i32(24);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Shl, w, k24));
            }
            // DXIL has no signature element for the coverage mask of a pixel
            // input: it is the intrinsic's result.
            Input::SampleCoverage => {
                let entry = self.s.entry;
                return Val::I32(self.s.op(
                    entry,
                    "coverage.i32",
                    Attr::ReadNone,
                    self.s.t.i32,
                    91,
                    &[],
                ));
            }
            _ => return Val::Missing("a pixel system input"),
        };
        let free = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let id = match self.s.inputs.iter().position(|e| e.kind == kind) {
            Some(id) => {
                if kind != 3 {
                    self.s.inputs[id].row.get_or_insert(free);
                }
                id
            }
            None if kind >= 3 => return Val::Missing("a pixel system input the signature lacks"),
            None => {
                // The console signature leaves out vertex and instance ids; PC
                // gives them a row after the others.
                let row = self
                    .s
                    .inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .map(|r| r + 1)
                    .max()
                    .unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: ["", "SV_VertexID", "SV_InstanceID"][kind as usize].into(),
                    index: 0,
                    kind,
                    comp: 5,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: 1,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.load_input(id, col)
    }

    /// The thread group's number as a mesh program counts it, made in the
    /// entry block. Stored in a buffer, the host runs a dispatch in parts
    /// that each number their groups from 0, and says where the part starts
    /// (`Shader::mesh_base_handle`).
    fn mesh_group(&mut self) -> V {
        let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
            unreachable!()
        };
        if self.buffer.is_none() {
            return g;
        }
        let entry = self.s.entry;
        let base = match self.buffer_base {
            Some(b) => b,
            None => {
                let h = self.s.mesh_base_handle();
                let row = self.s.m.i32(0);
                let r = self.s.op(
                    entry,
                    "cbufferLoadLegacy.i32",
                    Attr::ReadOnly,
                    self.s.t.cbuf_ret,
                    59,
                    &[h, row],
                );
                let b = self.s.m.extract(entry, r, 0);
                self.buffer_base = Some(b);
                b
            }
        };
        self.s.m.bin(entry, Bin::Add, g, base)
    }

    /// A component of a compute thread or group id, read in the entry block.
    fn compute_id(&mut self, name: &str, opcode: u32, c: u8) -> Val {
        let (entry, c) = (self.s.entry, self.s.m.i32(u32::from(c)));
        Val::I32(
            self.s
                .op(entry, name, Attr::ReadNone, self.s.t.i32, opcode, &[c]),
        )
    }

    fn find_input(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.inputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            // `then`, not `then_some`: the channel is only past `col` when
            // the condition holds.
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    fn load_input(&mut self, id: usize, col: u8) -> Val {
        let x = self.load_input_bits(id, col);
        Val::I32(if self.s.inputs[id].half {
            // The fetch ABI gives each channel its own VGPR.
            self.input_half(id, x)
        } else {
            x
        })
    }

    /// An input channel as declared on the host, where 16-bit types are widened.
    fn load_input_bits(&mut self, id: usize, col: u8) -> V {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.undef(self.s.t.i32),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32)
        } else if int {
            v
        } else {
            self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32)
        }
    }

    /// Recover the console's 16-bit payload from a widened host channel.
    fn input_half(&mut self, id: usize, x: V) -> V {
        if self.s.inputs[id].comp == 9 {
            let f = self.as_float(x);
            self.f16_nearest(f)
        } else {
            let mask = self.s.m.i32(0xFFFF);
            self.s.m.bin(self.cur, Bin::And, x, mask)
        }
    }

    /// The DXIL form of an IR value.
    fn get(&mut self, v: Value) -> Result<Val, Why> {
        if self.phi_at.contains_key(&v) {
            self.read_phis.insert(v);
        }
        let is_mask = self.x.st.masks.contains_key(&v)
            || self.x.st.highs.contains_key(&v)
            || self.split.as_ref().is_some_and(|sp| sp.is_mask(v));
        match self.vals.get(&v) {
            Some(&Val::Missing(_)) | None if self.mask_ok && is_mask => {
                return self.mask_dword(v).map(Val::I32);
            }
            Some(&Val::Missing(w)) => {
                // SCC of a lane-mask operation, read in its block: some lane
                // of the wave has its bit set.
                if w == LANE_MASK
                    && let Some(&(at, m)) = self.scc_any.get(&v)
                    && at == self.cur
                {
                    let bit = self.boolean(m, at)?;
                    self.cross_lane(false)?;
                    let i1 = self.s.t.i1;
                    let any = self
                        .s
                        .op(at, "waveAnyTrue", Attr::NoUnwind, i1, 113, &[bit]);
                    return Ok(Val::Bit(any));
                }
                if w == BARYCENTRICS_READ && self.leftovers {
                    // A barycentric none of whose bits reach anything (the
                    // other half of a 64-bit shift that drops it) needs no
                    // value: hosts without `SV_Barycentrics` can run it.
                    if self.demanded_bits(v) == 0 {
                        let val = Val::I32(self.s.m.i32(0));
                        self.vals.insert(v, val);
                        return Ok(val);
                    }
                    if let Some(val) = self.barycentric_input(v) {
                        self.vals.insert(v, val);
                        return Ok(val);
                    }
                }
                return Err(w);
            }
            Some(&val) => return Ok(val),
            None => {}
        }
        if let Some(&(param, dword)) = self.params.get(&v) {
            let val = self.user(param, dword);
            self.vals.insert(v, val);
            return Ok(val);
        }
        Err("a value defined after its use")
    }

    fn i32_of(&mut self, v: Value) -> Result<V, Why> {
        match self.get(v)? {
            Val::I32(x) => Ok(x),
            Val::Bit(x) => Ok(self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32)),
            Val::Missing(w) => Err(w),
        }
    }

    /// A user-data dword: a root parameter's value.
    fn user(&mut self, param: usize, dword: u32) -> Val {
        let rp = self.x.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => return Val::I32(self.s.m.i32(0)),
            RootType::Constants | RootType::Table => {}
            _ => return Val::Missing("a root descriptor"),
        }
        let index = u32::from(rp.user_data) + dword;
        self.user_reads.insert(
            index,
            UserData {
                indexed_buffer: false,
                index,
                slot: rp.slot,
                root: rp.root,
                dword,
            },
        );
        Val::I32(self.user_word(index))
    }

    /// A user-data dword no register of the console's holds: one the host
    /// computes when it binds (`root` says what, an `Other` code of
    /// `shader.rs`), placed after the layout's own dwords.
    fn host_word(&mut self, root: RootType, slot: u8, dword: u32) -> V {
        let known = self
            .user_reads
            .values()
            .find(|u| u.root == root && u.slot == slot && u.dword == dword)
            .map(|u| u.index);
        let index = match known {
            Some(index) => index,
            None => {
                let mut index = self
                    .x
                    .native
                    .layout
                    .iter()
                    .map(|e| u32::from(e.user_data) + e.sgprs)
                    .max()
                    .unwrap_or(0);
                while self.user_reads.contains_key(&index) {
                    index += 1;
                }
                self.user_reads.insert(
                    index,
                    UserData {
                        indexed_buffer: false,
                        index,
                        slot,
                        root,
                        dword,
                    },
                );
                index
            }
        };
        self.user_word(index)
    }

    /// User-data dword `index`, loaded in the entry block.
    fn user_word(&mut self, index: u32) -> V {
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        self.s.m.extract(entry, r, index % 4)
    }

    /// A dword of a lane mask as a number: the ballot of the lanes' bits
    /// (see [`MaskData`]). Taken where the mask is written, except an active
    /// mask written inside a loop and read after that loop: balloting there
    /// sees only the lanes still in the loop, so each lane's own bit of an
    /// exit mask (issue #465); after the loop the wave has reconverged and
    /// the ballot is the wave's mask.
    fn mask_dword(&mut self, v: Value) -> Result<V, Why> {
        if self.split.is_some() {
            return self.split_mask_dword(v);
        }
        let (lo, high) = match self.x.st.highs.get(&v) {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        if self.console_cf {
            // Every lane runs every block: the mask's bits are the lanes'
            // own, both halves' words exchanged.
            if let Some(&xy) = self.ballots.get(&lo) {
                return Ok(if high { xy.1 } else { xy.0 });
            }
            let at = match self.def_block.get(&lo) {
                None => self.s.entry,
                Some(&b) if b == self.cur_ir => self.cur,
                Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
            };
            if !self.balloting.insert(lo) {
                return Err("a lane mask whose lanes depend on its own value");
            }
            let bit = self.boolean(m, at);
            self.balloting.remove(&lo);
            let xy = self.console_ballot(at, bit?);
            self.ballots.insert(lo, xy);
            return Ok(if high { xy.1 } else { xy.0 });
        }
        if self.halves {
            return Err("a lane mask read whole in a wave64 program split in halves");
        }
        // A mask also set in every lane that does not run where it is
        // written: the ballot leaves those lanes out, so they are added.
        // Compute only, where host lane n is console lane n.
        let outside = !self.x.st.active.contains(&lo)
            && self.x.st.outside.contains(&lo)
            && matches!(self.x.native.io, StageIo::Cs(_));
        if !self.x.st.active.contains(&lo) && !outside {
            return Err("a lane mask holding lanes elsewhere read as data");
        }
        let (x, y) = match self.ballots.get(&lo) {
            Some(&xy) => xy,
            None => {
                let at = match self.def_block.get(&lo) {
                    // Program inputs (EXEC on entry).
                    None => self.s.entry,
                    Some(&b) if b == self.cur_ir => self.cur,
                    // After a loop its lanes leave at different iterations:
                    // ballot at the reconverged post-loop block that dominates
                    // this read (not inside a later divergent branch).
                    Some(&b) if !outside => {
                        if let Some(at) = self.after_loop_ballot_site(b, self.cur_ir) {
                            #[cfg(test)]
                            {
                                self.mask_ballots_after_loop += 1;
                            }
                            at
                        } else {
                            self.tail[b].ok_or("a lane mask written after its use")?
                        }
                    }
                    Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
                };
                if !self.balloting.insert(lo) {
                    return Err("a lane mask whose lanes depend on its own value");
                }
                let bit = self.boolean(m, at);
                self.balloting.remove(&lo);
                let bit = bit?;
                let i32t = self.s.t.i32;
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let r = self
                    .s
                    .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
                self.s.flags |= flags::WAVE_OPS;
                let mut xy = (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1));
                if outside {
                    let t = self.s.m.bool(true);
                    let here = self
                        .s
                        .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[t]);
                    let ones = self.s.m.i32(u32::MAX);
                    let mut add = |word: V, n: u32| {
                        let present = self.s.m.extract(at, here, n);
                        let away = self.s.m.bin(at, Bin::Xor, present, ones);
                        self.s.m.bin(at, Bin::Or, word, away)
                    };
                    xy = (add(xy.0, 0), add(xy.1, 1));
                }
                self.ballots.insert(lo, xy);
                xy
            }
        };
        Ok(if high { y } else { x })
    }

    /// Successors of IR block `b`.
    fn block_successors(p: &Program, b: usize) -> impl Iterator<Item = usize> {
        match p.blocks[b].exit {
            Exit::Jump(t) => [Some(t), None],
            Exit::Branch { taken, not_taken } => [Some(taken), Some(not_taken)],
            Exit::End | Exit::Indirect => [None, None],
        }
        .into_iter()
        .flatten()
    }

    /// Whether there is a nonempty path from `from` to `to`.
    fn reaches(p: &Program, from: usize, to: usize) -> bool {
        let mut seen = vec![false; p.blocks.len()];
        let mut stack = vec![from];
        while let Some(b) = stack.pop() {
            if b == to {
                return true;
            }
            if seen[b] {
                continue;
            }
            seen[b] = true;
            stack.extend(Self::block_successors(p, b));
        }
        false
    }

    /// Whether `from` lies on a cycle (a nonempty path back to itself).
    fn in_cycle(p: &Program, from: usize) -> bool {
        let mut seen = vec![false; p.blocks.len()];
        let mut stack: Vec<usize> = Self::block_successors(p, from).collect();
        while let Some(b) = stack.pop() {
            if b == from {
                return true;
            }
            if seen[b] {
                continue;
            }
            seen[b] = true;
            stack.extend(Self::block_successors(p, b));
        }
        false
    }

    /// Blocks of the natural loop containing `def` (reachable from `def` and
    /// able to reach `def` again). Empty when `def` is not in a cycle.
    fn natural_loop(p: &Program, def: usize) -> Vec<bool> {
        let n = p.blocks.len();
        if !Self::in_cycle(p, def) {
            return vec![false; n];
        }
        (0..n)
            .map(|b| Self::reaches(p, def, b) && Self::reaches(p, b, def))
            .collect()
    }

    /// DXIL block for a ballot of a mask written in loop block `def` and
    /// first read at `use_b` after leaving that loop: the earliest (RPO)
    /// loop-exit target that reaches the use, so the ballot sees the
    /// reconverged wave, if it dominates the read (none otherwise).
    fn after_loop_ballot_site(&self, def: usize, use_b: usize) -> Option<usize> {
        let p = self.x.p;
        if !Self::in_cycle(p, def) || Self::reaches(p, use_b, def) {
            return None;
        }
        let in_loop = Self::natural_loop(p, def);
        let mut best: Option<usize> = None;
        for (b, &member) in in_loop.iter().enumerate() {
            if !member {
                continue;
            }
            for exit in Self::block_successors(p, b) {
                if in_loop[exit] {
                    continue;
                }
                if exit != use_b && !Self::reaches(p, exit, use_b) {
                    continue;
                }
                if best.is_none_or(|cur| self.rpo[exit] < self.rpo[cur]) {
                    best = Some(exit);
                }
            }
        }
        let exit = best?;
        // With several exits that reach the read, the earliest need not be
        // on every path to it: the ballot must dominate the read.
        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut index = vec![usize::MAX; p.blocks.len()];
        for (n, &b) in order.iter().enumerate() {
            index[b] = n;
        }
        let idom = crate::ir::idoms(&p.blocks, &order, &index);
        let mut d = use_b;
        while d != exit {
            if idom[d] == d || idom[d] == usize::MAX {
                return None;
            }
            d = idom[d];
        }
        if exit == self.cur_ir {
            Some(self.cur)
        } else {
            self.block[exit].or(self.tail[exit])
        }
    }

    // ---- operands ----

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok(self.s.m.i32(0)),
            Operand::Scalar { code, .. } => {
                let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Vector { reg, .. } => {
                let v = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Const(253) => {
                // SCC as a number: 0 or 1.
                let v = inst.read(Reg::Scc).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Const(c @ (251 | 252)) => {
                // vccz / execz: 1 when no lane of the mask is set.
                let base = if c == 251 { crate::ir::VCC } else { EXEC };
                let mut any = None;
                for i in 0..1 + u8::from(self.x.p.wave64) {
                    let v = inst.read(Reg::S(base + i)).ok_or("an unread operand")?;
                    let x = self.i32_of(v)?;
                    any = Some(match any {
                        Some(a) => self.s.m.bin(self.cur, Bin::Or, a, x),
                        None => x,
                    });
                }
                let zero = self.s.m.i32(0);
                let any = any.ok_or("an unread operand")?;
                let none = self.s.m.cmp(self.cur, pred::EQ, any, zero);
                Ok(self.s.m.cast(self.cur, Cast::ZExt, none, self.s.t.i32))
            }
            // `src_pops_exiting_wave_id`: only compared with the collision
            // wave id's fields, which a wave that overlaps none never reads.
            Operand::Const(239) if self.s.rov => Ok(self.s.m.i32(0)),
            Operand::Const(c) => crate::resource::inline(c)
                .map(|x| self.s.m.i32(x))
                .ok_or("a special operand (lds_direct)"),
            Operand::Literal(x) => Ok(self.s.m.i32(x)),
            Operand::Attr { .. } => Err("an attribute operand"),
        }
    }

    /// GFX10 puts an inline f16 constant in the low half of the operand.
    fn f16_operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Const(c @ 240..=248) => {
                let h = [
                    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                ][usize::from(c - 240)];
                Ok(self.s.m.i32(h))
            }
            _ => self.operand(inst, op),
        }
    }

    fn as_float(&mut self, x: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, x, self.s.t.f32)
    }

    fn as_int(&mut self, f: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, f, self.s.t.i32)
    }

    /// The SDWA operand selection of source `k`: a byte or word of the
    /// dword, zero- or sign-extended (none: the whole dword).
    fn sdwa_src(inst: &Inst, k: usize) -> Option<(u8, bool)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        let sel = *s.src_sel.get(k)?;
        let sext = s.sext >> k & 1 != 0;
        (sel < 6).then_some((sel, sext))
    }

    /// Bits `sel` (0-3 a byte, 4-5 a word) of `x`, extended.
    fn select_bits(&mut self, x: V, sel: u8, sext: bool) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8, 8 * sel)
        } else {
            (16, 16 * (sel - 4))
        };
        let left = self.s.m.i32(u32::from(32 - width - at));
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - u32::from(width));
        let op = if sext { Bin::AShr } else { Bin::LShr };
        self.s.m.bin(c, op, t, right)
    }

    /// A float source with its abs/neg modifiers.
    fn fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let selected = Self::sdwa_src(inst, k);
        let mut f = match op {
            _ if selected.is_some() => {
                let (sel, sext) = selected.expect("checked");
                let x = self.operand(inst, &op)?;
                let x = self.select_bits(x, sel, sext);
                self.as_float(x)
            }
            _ if k == 0 && self.src0.is_some() => {
                let x = self.src0.expect("checked");
                self.as_float(x)
            }
            Operand::Vector { reg, .. } | Operand::Scalar { code: reg, .. }
                if !matches!(op, Operand::Scalar { code: NULL, .. }) =>
            {
                let r = if matches!(op, Operand::Vector { .. }) {
                    Reg::V(reg)
                } else {
                    Reg::S(reg)
                };
                let val = inst.read(r).ok_or("an unread operand")?;
                match self.floats.get(&val) {
                    Some(&f) => f,
                    None => {
                        let x = self.i32_of(val)?;
                        self.as_float(x)
                    }
                }
            }
            _ => {
                let x = self.operand(inst, &op)?;
                self.as_float(x)
            }
        };
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    fn src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 {
            return Err("integer operand modifiers");
        }
        self.raw_src(inst, k)
    }

    /// Source `k`'s bits (with its SDWA selection); its abs/neg modifiers
    /// are left to the caller.
    fn raw_src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        match self.src0 {
            Some(x) if k == 0 => Ok(x),
            _ => {
                let x = self.operand(inst, &op)?;
                Ok(match Self::sdwa_src(inst, k) {
                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                    None => x,
                })
            }
        }
    }

    /// Operand `k` of a typeless `_b32` instruction: `abs` clears bit 31 and
    /// `neg` flips it, with no float operation, so denormals and NaN payloads
    /// pass through as on the hardware.
    fn bits_src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let (abs, neg) = (v.abs >> k & 1 != 0, v.neg >> k & 1 != 0);
        let mut x = self.raw_src(inst, k)?;
        if abs {
            let m = self.s.m.i32(0x7FFF_FFFF);
            x = self.s.m.bin(self.cur, Bin::And, x, m);
        }
        if neg {
            let m = self.s.m.i32(0x8000_0000);
            x = self.s.m.bin(self.cur, Bin::Xor, x, m);
        }
        Ok(x)
    }

    fn unary(&mut self, op: u32, f: V) -> V {
        self.s.op(
            self.cur,
            "unary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[f],
        )
    }

    fn binary(&mut self, op: u32, a: V, b: V) -> V {
        self.s.op(
            self.cur,
            "binary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[a, b],
        )
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but -0 orders below +0 (RDNA2 ISA, `v_max_f32`), where
    /// the compare and select DXIL specifies gives either zero.
    fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    fn fmad(&mut self, a: V, b: V, c: V) -> V {
        self.s.op(
            self.cur,
            "tertiary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            46,
            &[a, b, c],
        )
    }

    /// `fma(a, b, c)` in binary32 with a round-to-odd sticky bit when the
    /// fused result is not exact, so a following nearest-even narrow to
    /// binary16 matches a single round of the exact `a*b + c` (RDNA2
    /// `v_fma_mixlo_f16` / `v_fma_mixhi_f16`). Without this, an exact value
    /// just above a half-ulp midpoint can round to the midpoint in f32 and
    /// then tie-to-even the wrong way in f16. The sticky step picks the
    /// adjacent odd f32 toward the exact result (residual sign); always
    /// OR-ing 1 would push a just-below-midpoint value across the midpoint.
    fn fmad_round_to_odd(&mut self, a: V, b: V, c: V) -> V {
        let cur = self.cur;
        let r = self.fmad(a, b, c);
        // Residual of the f32 FMA: nonzero ⇒ the exact value was not `r`.
        let z = self.s.m.f32(0x8000_0000);
        let neg_r = self.s.m.bin(cur, Bin::Sub, z, r);
        let err = self.fmad(a, b, neg_r);
        let err = self.s.m.bin(cur, Bin::Add, err, c);
        let bits = self.as_int(r);
        let err_bits = self.as_int(err);
        let abs_mask = self.s.m.i32(0x7FFF_FFFF);
        let abs = self.s.m.bin(cur, Bin::And, bits, abs_mask);
        let err_abs = self.s.m.bin(cur, Bin::And, err_bits, abs_mask);
        let zero = self.s.m.i32(0);
        let sticky = self.s.m.cmp(cur, pred::NE, err_abs, zero);
        let inf = self.s.m.i32(0x7F80_0000);
        let finite = self.s.m.cmp(cur, pred::IULT, abs, inf);
        let adjust = self.s.m.bin(cur, Bin::And, sticky, finite);
        // Already odd: keep. Even: step one ulp toward the exact value.
        let one = self.s.m.i32(1);
        let lsb = self.s.m.bin(cur, Bin::And, bits, one);
        let even = self.s.m.cmp(cur, pred::EQ, lsb, zero);
        let do_step = self.s.m.bin(cur, Bin::And, adjust, even);
        let sign = self.s.m.i32(0x8000_0000);
        let err_sign = self.s.m.bin(cur, Bin::And, err_bits, sign);
        let err_pos = self.s.m.cmp(cur, pred::EQ, err_sign, zero);
        let abs_zero = self.s.m.cmp(cur, pred::EQ, abs, zero);
        let min_neg = self.s.m.i32(0x8000_0001);
        let stepped_z = self.s.m.select(cur, err_pos, one, min_neg);
        let xored = self.s.m.bin(cur, Bin::Xor, err_bits, bits);
        let xored_sign = self.s.m.bin(cur, Bin::And, xored, sign);
        let same_sign = self.s.m.cmp(cur, pred::EQ, xored_sign, zero);
        let up = self.s.m.bin(cur, Bin::Add, bits, one);
        let down = self.s.m.bin(cur, Bin::Sub, bits, one);
        let stepped_nz = self.s.m.select(cur, same_sign, up, down);
        let stepped = self.s.m.select(cur, abs_zero, stepped_z, stepped_nz);
        let out = self.s.m.select(cur, do_step, stepped, bits);
        self.as_float(out)
    }

    /// `f` as an f16 in the low bits, rounded to nearest even as the
    /// console does (MODE's default): a tie goes to the even half, a finite
    /// value from 65520 up becomes infinity, results below 2^-14 stay
    /// subnormal, NaN stays NaN (quiet, top payload bits kept). Worked on
    /// the bits: the host's f32-to-f16 conversion (DXIL op 130) rounds
    /// toward zero on NVIDIA and WARP alike (1 + 3/2048 gives 0x3C01, 1e6
    /// gives 0x7BFF).
    fn f16_nearest(&mut self, f: V) -> V {
        let c = self.cur;
        let x = self.as_int(f);
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k0, k1, k13, k16, k23) = (
            k(self, 0),
            k(self, 1),
            k(self, 13),
            k(self, 16),
            k(self, 23),
        );
        let sign = self.s.m.bin(c, Bin::LShr, x, k16);
        let sign_mask = k(self, 0x8000);
        let sign = self.s.m.bin(c, Bin::And, sign, sign_mask);
        let abs_mask = k(self, 0x7FFF_FFFF);
        let a = self.s.m.bin(c, Bin::And, x, abs_mask);
        // Normal halves: rebias the exponent (127 - 15) and round on bit 13.
        let odd = self.s.m.bin(c, Bin::LShr, a, k13);
        let odd = self.s.m.bin(c, Bin::And, odd, k1);
        let bias = k(self, 0xFFF_u32.wrapping_sub(112 << 23));
        let n = self.s.m.bin(c, Bin::Add, a, bias);
        let n = self.s.m.bin(c, Bin::Add, n, odd);
        let normal = self.s.m.bin(c, Bin::LShr, n, k13);
        // Subnormal halves: the 24-bit significand shifted down by
        // 126 - exponent (at least 14, at most 31), rounded to nearest even.
        let e = self.s.m.bin(c, Bin::LShr, a, k23);
        let man_mask = k(self, 0x7F_FFFF);
        let man = self.s.m.bin(c, Bin::And, a, man_mask);
        let implicit = k(self, 0x80_0000);
        let m = self.s.m.bin(c, Bin::Or, man, implicit);
        let k126 = k(self, 126);
        let sh = self.s.m.bin(c, Bin::Sub, k126, e);
        let k31 = k(self, 31);
        let wide = self.s.m.cmp(c, pred::IUGT, sh, k31);
        let sh = self.s.m.select(c, wide, k31, sh);
        let q = self.s.m.bin(c, Bin::LShr, m, sh);
        let one_s = self.s.m.bin(c, Bin::Shl, k1, sh);
        let low = self.s.m.bin(c, Bin::Sub, one_s, k1);
        let rem = self.s.m.bin(c, Bin::And, m, low);
        let half = self.s.m.bin(c, Bin::LShr, one_s, k1);
        let above = self.s.m.cmp(c, pred::IUGT, rem, half);
        let tie = self.s.m.cmp(c, pred::EQ, rem, half);
        let q_odd = self.s.m.bin(c, Bin::And, q, k1);
        let q_odd = self.s.m.cmp(c, pred::NE, q_odd, k0);
        let tie_up = self.s.m.bin(c, Bin::And, tie, q_odd);
        let up = self.s.m.bin(c, Bin::Or, above, tie_up);
        let inc = self.s.m.select(c, up, k1, k0);
        let sub = self.s.m.bin(c, Bin::Add, q, inc);
        let min_normal = k(self, 0x3880_0000);
        let is_sub = self.s.m.cmp(c, pred::IULT, a, min_normal);
        let r = self.s.m.select(c, is_sub, sub, normal);
        let overflow = k(self, 0x477F_F000);
        let big = self.s.m.cmp(c, pred::IUGE, a, overflow);
        let inf = k(self, 0x7C00);
        let r = self.s.m.select(c, big, inf, r);
        let f32_inf = k(self, 0x7F80_0000);
        let nan = self.s.m.cmp(c, pred::IUGT, a, f32_inf);
        let payload = self.s.m.bin(c, Bin::LShr, man, k13);
        let quiet = k(self, 0x7E00);
        let qnan = self.s.m.bin(c, Bin::Or, quiet, payload);
        let r = self.s.m.select(c, nan, qnan, r);
        self.s.m.bin(c, Bin::Or, r, sign)
    }

    /// `f` as an f16 in the low bits, rounded toward zero: the nearest-even
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let h = self.f16_nearest(f);
        let back = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's
    /// top bit when `signed`; widths from 32 on take every bit. A signed
    /// field that runs past bit 31 extends from bit 31, as the hardware
    /// does: the shift is arithmetic then.
    fn bfe(&mut self, a: V, off: V, width: V, signed: bool) -> V {
        let c = self.cur;
        let (k31, k32, zero, one, ones) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(u32::MAX),
        );
        let off = self.s.m.bin(c, Bin::And, off, k31);
        let shift = if signed { Bin::AShr } else { Bin::LShr };
        let shifted = self.s.m.bin(c, shift, a, off);
        let full = self.s.m.cmp(c, pred::IUGE, width, k32);
        let w = self.s.m.bin(c, Bin::And, width, k31);
        let mask = self.s.m.bin(c, Bin::Shl, one, w);
        let mask = self.s.m.bin(c, Bin::Sub, mask, one);
        let mask = self.s.m.select(c, full, ones, mask);
        let r = self.s.m.bin(c, Bin::And, shifted, mask);
        if !signed {
            return r;
        }
        // Sign-extend a field narrower than 32 bits (none for width 0).
        let left = self.s.m.bin(c, Bin::Sub, k32, w);
        let left = self.s.m.bin(c, Bin::And, left, k31);
        let t = self.s.m.bin(c, Bin::Shl, r, left);
        let t = self.s.m.bin(c, Bin::AShr, t, left);
        let is0 = self.s.m.cmp(c, pred::EQ, w, zero);
        let keep = self.s.m.bin(c, Bin::Or, full, is0);
        self.s.m.select(c, keep, r, t)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the three instead (the host's minimum skips NaNs, so that
    /// is the smallest number among them).
    ///
    /// Uses `dx.op.isSpecialFloat.f32` (`IsNaN`) instead of `fcmp uno`: some
    /// D3D12 drivers (observed on Intel Iris Xe) fold `fcmp uno x, x` to false
    /// under default shader fast-math, returning the wrong operand when the
    /// third input is NaN.
    fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let lo = self.binary(36, a, b);
        let hi = self.binary(35, a, b);
        let t = self.binary(36, hi, third);
        let median = self.binary(35, lo, t);
        let least = self.binary(36, lo, third);
        let is_nan = |l: &mut Self, x: V| {
            l.s.op(c, "isSpecialFloat.f32", Attr::ReadNone, l.s.t.i1, 8, &[x])
        };
        let nan_a = is_nan(self, a);
        let nan_b = is_nan(self, b);
        let nan_c = is_nan(self, third);
        let ab = self.s.m.bin(c, Bin::Or, nan_a, nan_b);
        let nan = self.s.m.bin(c, Bin::Or, ab, nan_c);
        self.s.m.select(c, nan, least, median)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        self.s.m.select(c, exact, zero, r)
    }

    /// The high dword of a 32 × 32-bit product.
    fn mul_hi(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let ext = if signed { Cast::SExt } else { Cast::ZExt };
        let a = self.s.m.cast(c, ext, a, i64t);
        let b = self.s.m.cast(c, ext, b, i64t);
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::LShr, m, k32);
        self.s.flags |= flags::INT64;
        self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32)
    }

    /// `0 × anything = 0`, as in D3D9.
    fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let zero = self.s.m.f32(0);
        let az = self.s.m.cmp(c, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(c, pred::OEQ, b, zero);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        self.s.m.select(c, z, zero, m)
    }

    /// Flush an f32 subnormal to a signed zero (sign bit kept). Zeros, normals,
    /// infinities and NaNs pass through. Used for `v_mad_f32` / `v_mac_f32`,
    /// which flush regardless of `MODE.denorm32`.
    fn flush_f32_subnormal(&mut self, f: V) -> V {
        let c = self.cur;
        let bits = self.as_int(f);
        let abs_mask = self.s.m.i32(0x7FFF_FFFF);
        let min_normal = self.s.m.i32(0x0080_0000);
        let sign_mask = self.s.m.i32(0x8000_0000);
        let abs = self.s.m.bin(c, Bin::And, bits, abs_mask);
        let is_sub = self.s.m.cmp(c, pred::IULT, abs, min_normal);
        let sign = self.s.m.bin(c, Bin::And, bits, sign_mask);
        let zero = self.as_float(sign);
        self.s.m.select(c, is_sub, zero, f)
    }

    /// `a × b + c` with subnormal inputs, product and sum flushed (RDNA
    /// `v_mad_f32`, independent of `MODE.denorm32`).
    fn mad_f32(&mut self, a: V, b: V, c: V) -> V {
        let a = self.flush_f32_subnormal(a);
        let b = self.flush_f32_subnormal(b);
        let c = self.flush_f32_subnormal(c);
        let m = self.s.m.bin(self.cur, Bin::Mul, a, b);
        let m = self.flush_f32_subnormal(m);
        let r = self.s.m.bin(self.cur, Bin::Add, m, c);
        self.flush_f32_subnormal(r)
    }

    /// `a × b + addend` fused, with `0 × anything = 0` as in D3D9
    /// (`v_fma_legacy_f32`).
    fn fma_legacy(&mut self, a: V, b: V, addend: V) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let az = self.s.m.cmp(c, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(c, pred::OEQ, b, zero);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        let product_zero = self.s.m.bin(c, Bin::Add, zero, addend);
        // shortcut: FMad lets the driver fuse, as for `v_fma_f32`.
        let fused = self.fmad(a, b, addend);
        self.s.m.select(c, z, product_zero, fused)
    }

    // ---- lane booleans ----

    fn boolean(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        if b == Bool::TRUE || b == Bool::FALSE {
            return Ok(self.s.m.bool(b == Bool::TRUE));
        }
        if self.split.is_some() {
            // Structurisation's booleans hold for lanes that reach a block;
            // the split lowering keeps exact masks instead.
            return Err("a per-lane condition the split lowering does not follow");
        }
        if let Some(&v) = self.bools.get(&(at, b)) {
            return Ok(v);
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let a = self.atom(atom, at)?;
        let m = &mut self.s.m;
        let v = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => {
                let t = m.bool(true);
                m.bin(at, Bin::Xor, a, t)
            }
            (Bool::FALSE, h) => {
                let h = self.boolean(h, at)?;
                self.s.m.bin(at, Bin::And, a, h)
            }
            (l, Bool::TRUE) => {
                let l = self.boolean(l, at)?;
                self.s.m.bin(at, Bin::Or, a, l)
            }
            (l, h) => {
                let l = self.boolean(l, at)?;
                let h = self.boolean(h, at)?;
                self.s.m.select(at, a, h, l)
            }
        };
        self.bools.insert((at, b), v);
        Ok(v)
    }

    fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        match atom {
            Atom::Lane(v) => self.lane_bits.get(&v).copied().ok_or_else(|| {
                self.lane_fail
                    .get(&v)
                    .copied()
                    .unwrap_or("a lane mask from a carry")
            }),
            Atom::Uniform(v) => match self.get(v)? {
                Val::Bit(x) => Ok(x),
                Val::I32(x) => {
                    let z = self.s.m.i32(0);
                    Ok(self.s.m.cmp(at, pred::NE, x, z))
                }
                Val::Missing(w) => Err(w),
            },
            Atom::Phi(id) => {
                if let Some(&v) = self.mask_phis.get(&id) {
                    return Ok(v);
                }
                let &(b, i) = self
                    .phi_at
                    .get(&id)
                    .ok_or("a lane-mask join without a phi")?;
                let blk = self.block[b].ok_or("an unreachable join")?;
                let v = self.s.m.phi(blk, self.s.t.i1);
                self.mask_phis.insert(id, v);
                self.pending_masks.push((v, b, i));
                Ok(v)
            }
            // shortcut: "some lane of the quad is set" is taken as true: the
            // lanes it adds only feed derivatives, and exports keep their own
            // masks. Wrong if such a section had a side effect.
            Atom::Quad(_) => Ok(self.s.m.bool(true)),
            // With console lanes, a mask held as a number is the console's:
            // the lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes => {
                let hi = match self.pair_high(v) {
                    Some(h) => Some(self.get(h)?),
                    None => None,
                };
                let lo = self.get(v)?;
                let word = |w: Val| match w {
                    Val::I32(x) => Ok(x),
                    _ => Err("an opaque lane mask"),
                };
                let lo = word(lo)?;
                let hi = hi.map(word).transpose()?;
                Ok(self.lane_bit(at, lo, hi))
            }
            Atom::Constant(0) => Ok(self.s.m.bool(false)),
            Atom::Constant(k)
                if k == u64::MAX || (!self.x.native.wave64 && k as u32 == u32::MAX) =>
            {
                Ok(self.s.m.bool(true))
            }
            Atom::Constant(k) if self.console_lanes => {
                let lo = self.s.m.i32(k as u32);
                let hi = self.s.m.i32((k >> 32) as u32);
                Ok(self.lane_bit(at, lo, Some(hi)))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
        }
    }

    /// This lane's bit of the mask `lo`, `hi` (wave64; host lane = console
    /// lane, so not in a program split in halves).
    fn lane_bit(&mut self, at: usize, lo: V, hi: Option<V>) -> V {
        debug_assert!(!self.halves || self.console_cf);
        if self.console_cf {
            // The console lane's bit: its half's word, its host lane's bit.
            let (lane, half) = self.console_lane();
            let (k31, one, z) = (self.s.m.i32(31), self.s.m.i32(1), self.s.m.i32(0));
            let word = match hi {
                Some(hi) => {
                    let low = self.s.m.cmp(at, pred::EQ, half, z);
                    self.s.m.select(at, low, lo, hi)
                }
                None => lo,
            };
            let sh = self.s.m.bin(at, Bin::And, lane, k31);
            let t = self.s.m.bin(at, Bin::LShr, word, sh);
            let t = self.s.m.bin(at, Bin::And, t, one);
            return self.s.m.cmp(at, pred::NE, t, z);
        }
        let lane = if self.split.is_some() {
            self.split_lanes().1
        } else {
            self.s.op(
                at,
                "waveGetLaneIndex",
                Attr::ReadOnly,
                self.s.t.i32,
                111,
                &[],
            )
        };
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = match hi {
            Some(hi) => {
                let low = self.s.m.cmp(at, pred::IULT, lane, k32);
                self.s.m.select(at, low, lo, hi)
            }
            None => lo,
        };
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    fn select_condition(&mut self, inst: &Inst, mask: Option<&Operand>) -> Result<V, Why> {
        if self.split.is_some() {
            return match mask {
                Some(&Operand::Scalar { code, .. }) if code == NULL => Ok(self.s.m.bool(false)),
                Some(&Operand::Scalar { code, .. }) => {
                    let lo = inst.read(Reg::S(code)).ok_or(UNTRACKED_SELECT)?;
                    let c = self.cur;
                    self.exact_pair(lo, inst.read(Reg::S(code + 1)), c)
                }
                _ => Err(UNTRACKED_SELECT),
            };
        }
        if let Some(&m) = inst
            .writes
            .first()
            .and_then(|&(_, v)| self.x.st.select_masks.get(&v))
        {
            return self.boolean(m, self.cur);
        }
        // Inline 0 and -1: no lane or every lane.
        match mask {
            Some(&Operand::Const(128)) => return Ok(self.s.m.bool(false)),
            Some(&Operand::Const(193)) => return Ok(self.s.m.bool(true)),
            _ => {}
        }
        let Some(&Operand::Scalar { code, .. }) = mask else {
            return Err(UNTRACKED_SELECT);
        };
        if code == NULL {
            return Ok(self.s.m.bool(false));
        }
        if !self.numeric_selects || !self.console_lanes {
            return Err(UNTRACKED_SELECT);
        }
        // A scalar operation can define either dword independently. The SSA
        // operands at this use give the actual pair, even across a join.
        let lo = self.i32_of(inst.read(Reg::S(code)).ok_or(UNTRACKED_SELECT)?)?;
        let hi = if self.x.p.wave64 {
            Some(self.i32_of(inst.read(Reg::S(code + 1)).ok_or(UNTRACKED_SELECT)?)?)
        } else {
            None
        };
        self.s.flags |= flags::WAVE_OPS;
        Ok(self.lane_bit(self.cur, lo, hi))
    }

    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        let b = *self.def_block.get(&lo)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
        inst.write(pick(reg)?)
    }

    fn guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        if self.console_cf {
            // Every lane is here: EXEC is the guard. Lane reads and writes
            // ignore EXEC on the console.
            let inst = &self.x.p.blocks[b].insts[i];
            let ignores = matches!(
                inst.name(),
                "v_readlane_b32" | "v_writelane_b32" | "v_readfirstlane_b32"
            ) || matches!(
                inst.fields,
                Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. }
            );
            return match self.x.st.execs[b].get(i).copied().flatten() {
                Some(e) if !ignores && e != Bool::TRUE => Ok(Some(self.boolean(e, self.cur)?)),
                _ => Ok(None),
            };
        }
        if self.split.is_some() {
            return self.split_guard(&self.x.p.blocks[b].insts[i]);
        }
        match self.x.st.guards[b][i] {
            None => Ok(None),
            Some(g) => Ok(Some(self.boolean(g, self.cur)?)),
        }
    }

    // ---- results ----

    fn set(&mut self, v: Option<Value>, val: Val) {
        if let Some(v) = v {
            self.vals.insert(v, val);
        }
    }

    /// Writes VGPR `reg` (or the SGPR of a lane read), under `guard`.
    fn write(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let x = match (Self::sdwa_dst(inst), reg) {
            (Some((sel, unused)), Reg::V(_)) => self.place_bits(inst, reg, x, sel, unused),
            _ => x,
        };
        self.write_placed(inst, reg, x, guard);
    }

    /// [`Self::write`] of a dword already placed (SDWA applied).
    fn write_placed(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let Some(new) = inst.write(reg) else { return };
        let x = match (guard, reg) {
            (Some(g), Reg::V(_)) => {
                let o = inst.read(reg);
                let read = o.is_some_and(|o| self.read_phis.contains(&o));
                let old = match o.map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    // Split: the lanes EXEC leaves out keep a value this
                    // lowering does not have; so does the register.
                    Some(Err(w)) if self.split.is_some() => {
                        self.vals.insert(new, Val::Missing(w));
                        return;
                    }
                    // The validator refuses reads of undef.
                    _ => self.s.m.i32(0),
                };
                // What lanes keep is zero where it has no DXIL form (as
                // above), so keeping a phi does not make it live.
                if let Some(o) = o.filter(|_| !read) {
                    self.read_phis.remove(&o);
                }
                self.s.m.select(self.cur, g, x, old)
            }
            _ => x,
        };
        self.vals.insert(new, Val::I32(x));
    }

    /// The SDWA destination selection: a byte or word, and what the other
    /// bits get (0 zeros, 1 the sign above the field, 2 the old bits).
    fn sdwa_dst(inst: &Inst) -> Option<(u8, u8)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        (s.dst_sel < 6).then_some((s.dst_sel, s.dst_unused))
    }

    /// The low byte or word of `x` placed at `sel` in VGPR `reg`.
    fn place_bits(&mut self, inst: &Inst, reg: Reg, x: V, sel: u8, unused: u8) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8u32, 8 * u32::from(sel))
        } else {
            (16, 16 * u32::from(sel - 4))
        };
        let left = self.s.m.i32(32 - width);
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - width - at);
        if unused == 1 {
            // Bits below the field shift in as zeros.
            return self.s.m.bin(c, Bin::AShr, t, right);
        }
        let field = self.s.m.bin(c, Bin::LShr, t, right);
        if unused != 2 {
            return field;
        }
        let old = match inst.read(reg).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let keep = self.s.m.i32(!((((1u64 << width) - 1) as u32) << at));
        let old = self.s.m.bin(c, Bin::And, old, keep);
        self.s.m.bin(c, Bin::Or, old, field)
    }

    fn write_f(&mut self, inst: &Inst, f: V, guard: Option<V>) -> Result<(), Why> {
        self.write_f_inner(inst, f, guard, false)
    }

    /// Like [`Self::write_f`], but flushes f32 subnormals to signed zero before
    /// and after `omod` (`v_mad_f32` / `v_mac_f32` on RDNA).
    fn write_f_flush_denorms(&mut self, inst: &Inst, f: V, guard: Option<V>) -> Result<(), Why> {
        self.write_f_inner(inst, f, guard, true)
    }

    fn write_f_inner(
        &mut self,
        inst: &Inst,
        f: V,
        guard: Option<V>,
        flush_denorms: bool,
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let mut f = if flush_denorms {
            self.flush_f32_subnormal(f)
        } else {
            f
        };
        match v.omod {
            0 => {}
            k => {
                let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                let c = self.cur;
                let scale = self.s.m.f32(factor.to_bits());
                f = self.s.m.bin(c, Bin::Mul, f, scale);
                if flush_denorms {
                    f = self.flush_f32_subnormal(f);
                }
                // RDNA2 omod (mul:2/mul:4/div:2) flushes -0.0 to +0.0;
                // IEEE FMul keeps the sign of zero.
                let zero = self.s.m.f32(0);
                let is_zero = self.s.m.cmp(c, pred::OEQ, f, zero);
                f = self.s.m.select(c, is_zero, zero, f);
            }
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        let x = self.as_int(f);
        self.write(inst, Reg::V(reg), x, guard);
        if guard.is_none()
            && Self::sdwa_dst(inst).is_none()
            && let Some(new) = inst.write(Reg::V(reg))
        {
            self.floats.insert(new, f);
        }
        Ok(())
    }

    fn write_i(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        // Integer `clamp` is unsigned saturation on add/sub; those ops apply it
        // before calling here. Other integer ops refuse the modifier.
        if v.omod != 0 || (v.clamp && !Self::integer_clamp_op(inst.name())) {
            return Err("integer output modifiers");
        }
        match v.dst {
            Some(Operand::Vector { reg, .. }) => self.write(inst, Reg::V(reg), x, guard),
            Some(Operand::Scalar { code, .. }) => self.write(inst, Reg::S(code), x, None),
            _ => return Err("a result without a register"),
        }
        Ok(())
    }

    /// VOP3 integer ops whose `clamp` means unsigned saturation via carry/borrow.
    fn integer_clamp_op(name: &str) -> bool {
        matches!(
            name,
            "v_add_co_u32"
                | "v_sub_co_u32"
                | "v_subrev_co_u32"
                | "v_add_co_ci_u32"
                | "v_sub_co_ci_u32"
                | "v_subrev_co_ci_u32"
                | "v_add_nc_u32"
                | "v_sub_nc_u32"
                | "v_subrev_nc_u32"
        )
    }

    /// Starts code only the lanes where `guard` holds run (a store under a
    /// lane guard): returns where it started and the join block.
    fn begin_guarded(&mut self, guard: Option<V>) -> Option<(usize, usize)> {
        let g = guard?;
        let (then, join) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(self.cur, Term::CondBr(g, then, join));
        let head = self.cur;
        self.cur = then;
        if let Some(sp) = &mut self.split {
            sp.guarded.insert(then);
        }
        Some((head, join))
    }

    /// Ends guarded code; each (value inside, value outside) becomes a phi.
    fn end_guarded(&mut self, at: Option<(usize, usize)>, values: &[(V, V)]) -> Vec<V> {
        let Some((head, join)) = at else {
            return values.iter().map(|v| v.0).collect();
        };
        let then = self.cur;
        self.s.m.terminate(then, Term::Br(join));
        self.cur = join;
        values
            .iter()
            .map(|&(inside, outside)| {
                let ty = self.s.m.type_of(inside);
                let phi = self.s.m.phi(join, ty);
                self.s.m.set_phi(phi, vec![(inside, then), (outside, head)]);
                phi
            })
            .collect()
    }

    /// Code that depends on the lane layout: wave operations, and in compute
    /// the console's wave size (lanes are thread ids in group order there).
    /// Pixel lanes match only within a quad.
    fn cross_lane(&mut self, quad: bool) -> Result<(), Why> {
        if self.s.mesh.is_some() {
            // shortcut: no wave size for mesh shaders in SM 6.6, so host
            // lanes cannot follow the console's; emulate through group-shared
            // memory if mesh programs turn out to need these.
            return Err("a cross-lane operation in a mesh shader");
        }
        if self.s.threads.is_none() && !(quad && self.s.pixel) {
            return Err("a cross-lane operation outside compute");
        }
        debug_assert!(self.s.threads.is_none() || self.console_lanes);
        self.s.flags |= flags::WAVE_OPS;
        Ok(())
    }

    /// Component `c` of the thread id in the group. With console lanes, the
    /// thread is relabelled so that its host lane is its console lane: the
    /// flat id is the wave's first lane's, rounded down to the wave size,
    /// plus the lane index. (A bijection while each host wave holds a
    /// wave-aligned run of flat ids, as the console's waves do.)
    fn thread_id(&mut self, c: u8) -> Val {
        if !self.console_lanes {
            return self.compute_id("threadIdInGroup.i32", 95, c);
        }
        let [x, y, _] = self.s.threads.expect("compute");
        let e = self.s.entry;
        let f = self.console_flat();
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        Val::I32(match c {
            0 => self.s.m.bin(e, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                self.s.m.bin(e, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(e, Bin::UDiv, f, xy),
        })
    }

    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    fn mesh_prologue(&mut self, mesh: super::shader::Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        // Stored in a buffer, there are no counts to declare: the vertex
        // shader reading it back draws every primitive slot.
        if self.buffer.is_none() {
            self.s
                .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        }
        let flat = self.mesh_thread();
        let z = self.s.m.i32(0);
        let passes = mesh.max_primitives.div_ceil(threads.max(1));
        for pass in 0..passes {
            let idx = if pass == 0 {
                flat
            } else {
                let off = self.s.m.i32(pass * threads);
                self.s.m.bin(self.cur, Bin::Add, flat, off)
            };
            let need_guard = (pass + 1) * threads > mesh.max_primitives;
            let guard = need_guard.then(|| self.s.m.cmp(self.cur, pred::IULT, idx, p));
            let at = self.begin_guarded(guard);
            if self.buffer.is_some() {
                self.buffer_store(true, idx, 0, z);
            } else {
                self.s.op(
                    self.cur,
                    "emitIndices",
                    Attr::NoUnwind,
                    void,
                    169,
                    &[idx, z, z, z],
                );
            }
            self.end_guarded(at, &[]);
        }
        Ok(())
    }

    /// Mesh stored in a buffer: stores `x` in dword `within` of vertex or
    /// primitive `slot` of this thread's group record
    /// (`super::meshbuf` has the layout).
    fn buffer_store(&mut self, primitive: bool, slot: V, within: u32, x: V) {
        let layout = self.buffer.as_ref().expect("a mesh stored in a buffer");
        let (each, first) = if primitive {
            (layout.primitive_dwords, layout.primitives_at())
        } else {
            (layout.vertex_dwords, 0)
        };
        let group = layout.group_dwords();
        let record = match self.buffer_record {
            Some(r) => r,
            None => {
                let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
                    unreachable!()
                };
                let n = self.s.m.i32(group);
                let r = self.s.m.bin(self.s.entry, Bin::Mul, g, n);
                self.buffer_record = Some(r);
                r
            }
        };
        let handle = self.s.mesh_buffer_handle(true);
        let c = self.cur;
        let each = self.s.m.i32(each);
        let dword = self.s.m.bin(c, Bin::Mul, slot, each);
        let k = self.s.m.i32(first + within);
        let dword = self.s.m.bin(c, Bin::Add, dword, k);
        let dword = self.s.m.bin(c, Bin::Add, dword, record);
        let two = self.s.m.i32(2);
        let byte = self.s.m.bin(c, Bin::Shl, dword, two);
        let iu = self.s.m.undef(self.s.t.i32);
        let args = [
            handle,
            byte,
            iu,
            x,
            iu,
            iu,
            iu,
            self.s.m.i8(1),
            self.s.m.i32(4),
        ];
        let void = self.s.t.void;
        self.s
            .op(c, "rawBufferStore.i32", Attr::NoUnwind, void, 140, &args);
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }

    /// The console wave size.
    fn wave(&self) -> u32 {
        if self.x.native.wave64 { 64 } else { 32 }
    }

    fn lane_index(&mut self) -> V {
        if self.split.is_some() {
            return self.split_lanes().1;
        }
        if self.s.mesh.is_some() {
            // The console's lane: mesh code reads no other lanes, so host
            // waves need not match the console's.
            let f = self.mesh_thread();
            let m = self.s.m.i32(self.wave() - 1);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        self.s.op(
            self.cur,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        )
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant. `pattern`, a quad permutation every quad
    /// uses (two bits per lane), lets a broadcast or an exchange take one
    /// quad operation instead of a read of each lane.
    fn quad_pick(&mut self, x: V, src: V, pattern: Option<u8>) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        match pattern {
            Some(0xE4) => return x,
            Some(p @ (0x00 | 0x55 | 0xAA | 0xFF)) => {
                let lane = self.s.m.i32(u32::from(p & 3));
                return self.s.op(
                    c,
                    "quadReadLaneAt.i32",
                    Attr::NoUnwind,
                    self.s.t.i32,
                    122,
                    &[x, lane],
                );
            }
            Some(p @ (0xB1 | 0x4E | 0x1B)) => {
                // Lane 0 reads lane 1, 2 or 3: across x, across y or the
                // diagonal (QuadOp 0, 1, 2), every lane its partner.
                let direction = self.s.m.i8((p & 3) - 1);
                return self.s.op(
                    c,
                    "quadOp.i32",
                    Attr::NoUnwind,
                    self.s.t.i32,
                    123,
                    &[x, direction],
                );
            }
            _ => {}
        }
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    /// `v_readfirstlane_b32`'s result `first` of `own` in a pixel shader:
    /// a helper lane takes its own value instead.
    ///
    /// On the console, whole-quad mode puts helper lanes in EXEC, so a
    /// waterfall loop (read the first lane's value, run the lanes with that
    /// value, take them out of EXEC) serves helper lanes too. On the host,
    /// wave operations leave helper lanes out (DXIL before
    /// `WaveOpsIncludeHelperLanes`): once only helper lanes are left in such
    /// a loop, `WaveReadLaneFirst` reads no lane, no lane matches, and the
    /// loop runs to [`LOOP_LIMIT`] in every such lane: draws of 0.3 to 1.3 s
    /// each, then DEVICE_HUNG. With its own value a helper lane matches in
    /// the first pass and leaves; what a helper lane computes only feeds
    /// derivatives, and its stores are dropped.
    fn helper_reads_own(&mut self, own: V, first: V) -> V {
        if !self.s.pixel {
            return first;
        }
        let c = self.cur;
        let helper = self
            .s
            .op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        self.s.m.select(c, helper, own, first)
    }

    /// Console flow: `x` of console lane `lane` (uniform), from whichever
    /// half holds it.
    fn console_read_lane(&mut self, x: V, lane: V) -> V {
        if !self.halves {
            // One host wave is the console wave, all of its lanes here.
            return self.read_lane(x, lane);
        }
        let c = self.cur;
        let (k31, k5, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(5),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let at = self.s.m.bin(c, Bin::And, lane, k31);
        let mine = self.read_lane(x, at);
        let (x0, x1) = self.exchange(c, mine);
        let h = self.s.m.bin(c, Bin::LShr, lane, k5);
        let h = self.s.m.bin(c, Bin::And, h, one);
        let low = self.s.m.cmp(c, pred::EQ, h, z);
        self.s.m.select(c, low, x0, x1)
    }

    fn read_lane(&mut self, x: V, lane: V) -> V {
        // Split: the lane is the console's, in this lane's half.
        let lane = if self.split.is_some() {
            let k31 = self.s.m.i32(31);
            self.s.m.bin(self.cur, Bin::And, lane, k31)
        } else {
            lane
        };
        self.read_host_lane(x, lane)
    }

    fn read_host_lane(&mut self, x: V, lane: V) -> V {
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// `value` when console lane `lane` is in EXEC, else 0. Cross-lane reads
    /// such as `ds_swizzle_b32` return 0 from an inactive source lane rather
    /// than the stale VGPR `waveReadLaneAt` would see.
    fn active_lane_or_zero(&mut self, value: V, lane: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let yes = self.s.m.bool(true);
        let ballot = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[yes]);
        self.s.flags |= flags::WAVE_OPS;
        let (k5, k31, one, z) = (
            self.s.m.i32(5),
            self.s.m.i32(31),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let words = match self.s.wave_size {
            Some(32) => 1,
            Some(64) => 2,
            _ => 4,
        };
        let word_idx = self.s.m.bin(c, Bin::LShr, lane, k5);
        let mut word = self.s.m.extract(c, ballot, 0);
        for j in 1..words {
            let kj = self.s.m.i32(j);
            let wj = self.s.m.extract(c, ballot, j);
            let is = self.s.m.cmp(c, pred::EQ, word_idx, kj);
            word = self.s.m.select(c, is, wj, word);
        }
        let sh = self.s.m.bin(c, Bin::And, lane, k31);
        let bit = self.s.m.bin(c, Bin::LShr, word, sh);
        let bit = self.s.m.bin(c, Bin::And, bit, one);
        let on = self.s.m.cmp(c, pred::NE, bit, z);
        self.s.m.select(c, on, value, z)
    }

    /// Coarse derivatives of `coords` across this lane's quad, as DXIL
    /// gradients (x for up to three coordinates, then y), each times
    /// `scale` if given.
    fn quad_gradients(&mut self, coords: &[V], scale: Option<V>) -> Vec<V> {
        let c = self.cur;
        let lane = self.lane_index();
        let top = self.s.m.i32(!3);
        let base = self.s.m.bin(c, Bin::And, lane, top);
        let lanes: Vec<V> = (1..3u32)
            .map(|k| {
                let k = self.s.m.i32(k);
                self.s.m.bin(c, Bin::Or, base, k)
            })
            .collect();
        let fu = self.s.m.undef(self.s.t.f32);
        let mut g = vec![fu; 6];
        for (k, &x) in coords.iter().enumerate().take(3) {
            let xi = self.as_int(x);
            let at = |l: &mut Self, lane: V| {
                let r = l.read_lane(xi, lane);
                l.as_float(r)
            };
            let origin = at(self, base);
            for (d, &lane) in lanes.iter().enumerate() {
                let there = at(self, lane);
                let mut diff = self.s.m.bin(c, Bin::Sub, there, origin);
                if let Some(s) = scale {
                    diff = self.s.m.bin(c, Bin::Mul, diff, s);
                }
                g[3 * d + k] = diff;
            }
        }
        g
    }

    /// The float value of a register value.
    fn float_of(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&f) = self.floats.get(&v) {
            return Ok(f);
        }
        let x = self.i32_of(v)?;
        Ok(self.as_float(x))
    }

    /// The values of VGPRs `reg..reg + n` as `inst` reads them.
    fn vgprs(&mut self, inst: &Inst, op: &Operand, n: u8) -> Result<Vec<V>, Why> {
        let Operand::Vector { reg, .. } = *op else {
            return Err("a vector operand");
        };
        (0..n)
            .map(|k| {
                let v = inst.read(Reg::V(reg + k)).ok_or("an unread operand")?;
                self.i32_of(v)
            })
            .collect()
    }

    /// LDS (thread-group shared memory) and `ds_swizzle`.
    fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *gds {
            return Err("GDS");
        }
        let name = inst.name();
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        if name == "ds_swizzle_b32" {
            // Bit 15 set: each lane of a quad picks a lane of it; else lanes
            // of each group of 32 combine their index with and/or/xor masks.
            let quad = offset & 0x8000 != 0
                || offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4;
            // Mesh: host quads are threads 4n..4n+3 of a 1D group, as the
            // console's, so quad-local swizzles read through quad operations.
            let mesh_quad = self.s.mesh.is_some() && quad;
            if !mesh_quad {
                self.cross_lane(quad)?;
            }
            // The source sits in the address field.
            let x = self.vgprs(inst, addr.as_ref().ok_or("a swizzle source")?, 1)?[0];
            let lane = self.lane_index();
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let src = if offset & 0x8000 != 0 {
                let (three, one) = (k(self, 3), k(self, 1));
                let sel = self.s.m.bin(c, Bin::And, lane, three);
                let sh = self.s.m.bin(c, Bin::Shl, sel, one);
                let pattern = k(self, offset & 0xFF);
                let pick = self.s.m.bin(c, Bin::LShr, pattern, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, three);
                let not3 = k(self, !3);
                let base = self.s.m.bin(c, Bin::And, lane, not3);
                self.s.m.bin(c, Bin::Or, base, pick)
            } else {
                let and = k(self, offset & 0x1F | !0x1F);
                let or = k(self, offset >> 5 & 0x1F);
                let xor = k(self, offset >> 10 & 0x1F);
                let t = self.s.m.bin(c, Bin::And, lane, and);
                let t = self.s.m.bin(c, Bin::Or, t, or);
                self.s.m.bin(c, Bin::Xor, t, xor)
            };
            let mut r = if mesh_quad {
                let pattern = (offset & 0x8000 != 0).then_some(offset as u8);
                self.quad_pick(x, src, pattern)
            } else {
                self.read_lane(x, src)
            };
            if self.split.is_some() {
                // Split: a source lane EXEC switches off gives 0.
                let e = self.split_exec(inst)?;
                let ei = self.s.m.cast(c, Cast::ZExt, e, self.s.t.i32);
                let there = self.read_lane(ei, src);
                let z = k(self, 0);
                let on = self.s.m.cmp(c, pred::NE, there, z);
                r = self.s.m.select(c, on, r, z);
            } else {
                // Inactive source lanes return 0, not the stale VGPR.
                // (Mesh: the ballot is of the host wave, so ask for the source's
                // host lane, which a quad shares with this lane's quad.)
                let probe = if mesh_quad {
                    let host = self.s.op(
                        c,
                        "waveGetLaneIndex",
                        Attr::ReadOnly,
                        self.s.t.i32,
                        111,
                        &[],
                    );
                    let (not3, three) = (k(self, !3), k(self, 3));
                    let base = self.s.m.bin(c, Bin::And, host, not3);
                    let low = self.s.m.bin(c, Bin::And, src, three);
                    self.s.m.bin(c, Bin::Or, base, low)
                } else {
                    src
                };
                r = if self.console_cf {
                    let active = self.src_lane_active(probe, guard);
                    let zero = self.s.m.i32(0);
                    self.s.m.select(c, active, r, zero)
                } else {
                    self.active_lane_or_zero(r, probe)
                };
            }
            let reg = dst_reg.ok_or("a swizzle result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if self.s.lds_dwords == 0 {
            return Err("LDS the program does not allocate");
        }
        if self.console_cf
            && self.halves
            && self
                .s
                .threads
                .is_some_and(|t| t.iter().product::<u32>() > 32)
        {
            // The console runs an instruction in all 64 lanes before the
            // next one. The halves are two host waves, both here (console
            // flow): a group barrier before each LDS access puts it after
            // every earlier one, whichever half made it.
            let (mode, void) = (self.s.m.i32(1 | 8), self.s.t.void);
            self.s
                .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        }
        let a = self.vgprs(inst, addr.as_ref().ok_or("an LDS address")?, 1)?[0];
        // shortcut: LDS addresses are taken as aligned and in bounds
        // (out-of-bounds console accesses read 0 and drop writes).
        let byte_addr = |l: &mut Self, bytes: u32| -> V {
            let c = l.cur;
            let o = l.s.m.i32(bytes);
            l.s.m.bin(c, Bin::Add, a, o)
        };
        let dword_at = |l: &mut Self, at: V| -> V {
            let c = l.cur;
            let two = l.s.m.i32(2);
            let mut index = l.s.m.bin(c, Bin::LShr, at, two);
            if let Some(bound) = l.split.as_ref().map(|sp| sp.lds_bound) {
                index = l.split_lds_index(index, bound, name.starts_with("ds_read"));
            }
            let lds = l.s.lds();
            l.s.m.element(c, lds, index)
        };
        let dword = |l: &mut Self, bytes: u32| -> V {
            let at = byte_addr(l, bytes);
            dword_at(l, at)
        };
        let width = |n: &str| match n.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            _ => 1,
        };
        // (byte offset, dwords) per element; read2/write2 scale their two
        // offsets by the element size (×64 for st64).
        let pair = |n: &str| -> Vec<(u32, u32)> {
            let w = width(n);
            let scale = 4 * w * if n.contains("st64") { 64 } else { 1 };
            vec![
                (u32::from(*offset0) * scale, w),
                (u32::from(*offset1) * scale, w),
            ]
        };
        // Bytes and halves: the dword holding them, and their bit position
        // in it. Halfwords at byte offset 3 straddle into byte 0 of the next
        // dword. `_d16` reads fill a register's low half (`_d16_hi` its high
        // half) and keep the other; `_d16_hi` writes take the data's high half.
        let base = name.trim_end_matches("_hi").trim_end_matches("_d16");
        let sub = match base {
            "ds_read_u8" | "ds_read_i8" | "ds_write_b8" => Some(8u32),
            "ds_read_u16" | "ds_read_i16" | "ds_write_b16" => Some(16),
            _ => None,
        };
        let sub = sub.filter(|_| {
            base == name
                || name.ends_with("_d16") && base.starts_with("ds_read")
                || name.ends_with("_d16_hi")
        });
        if let Some(bits) = sub {
            let at = byte_addr(self, offset);
            let p = dword_at(self, at);
            let three = self.s.m.i32(3);
            let byte = self.s.m.bin(c, Bin::And, at, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            let mask = self.s.m.i32((1u32 << bits) - 1);
            if name.starts_with("ds_read") {
                let reg = dst_reg.ok_or("an LDS read result")?;
                let w = self.s.m.load(c, p);
                let mut x = self.s.m.bin(c, Bin::LShr, w, sh);
                if bits == 16 {
                    // A halfword at byte offset 3 straddles into byte 0 of the
                    // next dword; for offsets 0..=2, `(at + 1) >> 2` is the
                    // same dword so the second load stays in bounds without a
                    // branch.
                    let one = self.s.m.i32(1);
                    let at_hi = self.s.m.bin(c, Bin::Add, at, one);
                    let p_hi = dword_at(self, at_hi);
                    let w_hi = self.s.m.load(c, p_hi);
                    let eight = self.s.m.i32(8);
                    let hi = self.s.m.bin(c, Bin::Shl, w_hi, eight);
                    let straddled = self.s.m.bin(c, Bin::Or, x, hi);
                    let is_straddle = self.s.m.cmp(c, pred::EQ, byte, three);
                    x = self.s.m.select(c, is_straddle, straddled, x);
                }
                let x = if base.contains("_i") {
                    let k = self.s.m.i32(32 - bits);
                    let x = self.s.m.bin(c, Bin::Shl, x, k);
                    self.s.m.bin(c, Bin::AShr, x, k)
                } else {
                    self.s.m.bin(c, Bin::And, x, mask)
                };
                let x = if base == name {
                    x
                } else if name.ends_with("_hi") {
                    self.high_half(inst, reg, x)
                } else {
                    self.low_half(inst, reg, x)
                };
                self.write(inst, Reg::V(reg), x, guard);
            } else {
                // Clear, then set: two atomics, so lanes writing other
                // bytes of the dword keep theirs.
                let mut val = self.vgprs(inst, data.first().ok_or("an LDS write")?, 1)?[0];
                if name != base {
                    // The data's high half (`_d16_hi`).
                    let k16 = self.s.m.i32(16);
                    val = self.s.m.bin(c, Bin::LShr, val, k16);
                }
                let val = self.s.m.bin(c, Bin::And, val, mask);
                let x = self.s.m.bin(c, Bin::Shl, val, sh);
                let m = self.s.m.bin(c, Bin::Shl, mask, sh);
                let ones = self.s.m.i32(u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, m, ones);
                use super::bitcode::Rmw;
                let at_g = self.begin_guarded(guard);
                self.s.m.atomic(self.cur, Rmw::And, p, keep);
                self.s.m.atomic(self.cur, Rmw::Or, p, x);
                if bits == 16 {
                    // A halfword at byte offset 3 writes its high byte into
                    // byte 0 of the next dword.
                    let is_straddle = self.s.m.cmp(self.cur, pred::EQ, byte, three);
                    let at_s = self.begin_guarded(Some(is_straddle));
                    let one = self.s.m.i32(1);
                    let at_hi = self.s.m.bin(self.cur, Bin::Add, at, one);
                    let p_hi = dword_at(self, at_hi);
                    let eight = self.s.m.i32(8);
                    let hi_byte = self.s.m.bin(self.cur, Bin::LShr, val, eight);
                    let keep_hi = self.s.m.i32(0xFFFF_FF00);
                    self.s.m.atomic(self.cur, Rmw::And, p_hi, keep_hi);
                    self.s.m.atomic(self.cur, Rmw::Or, p_hi, hi_byte);
                    self.end_guarded(at_s, &[]);
                }
                self.end_guarded(at_g, &[]);
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_read") {
            let reg = dst_reg.ok_or("an LDS read result")?;
            if !(rest.starts_with('2') || matches!(rest, "_b32" | "_b64" | "_b96" | "_b128")) {
                return Err("an unsupported LDS read");
            }
            let elements = if rest.starts_with('2') {
                pair(name)
            } else {
                vec![(offset, width(name))]
            };
            let mut n = 0;
            for (off, w) in elements {
                for k in 0..w {
                    let p = dword(self, off + 4 * k);
                    let x = self.s.m.load(self.cur, p);
                    self.write(inst, Reg::V(reg + n), x, guard);
                    n += 1;
                }
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_write") {
            if !(rest.starts_with('2') || matches!(rest, "_b32" | "_b64" | "_b96" | "_b128")) {
                return Err("an unsupported LDS write");
            }
            let elements = if rest.starts_with('2') {
                pair(name)
            } else {
                vec![(offset, width(name))]
            };
            let mut stores = Vec::new();
            for (e, (off, w)) in elements.into_iter().enumerate() {
                let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
                for (k, x) in vals.into_iter().enumerate() {
                    stores.push((off + 4 * k as u32, x));
                }
            }
            let at = self.begin_guarded(guard);
            for (off, x) in stores {
                let p = dword(self, off);
                self.s.m.store(self.cur, p, x);
            }
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Atomics: the old value returns with `_rtn`.
        let op = name
            .strip_prefix("ds_")
            .map(|n| n.replace("_rtn", ""))
            .ok_or("an LDS operation")?;
        use super::bitcode::Rmw;
        let rmw = match op.as_str() {
            "add_u32" => Some(Rmw::Add),
            "sub_u32" => Some(Rmw::Sub),
            "and_b32" => Some(Rmw::And),
            "or_b32" => Some(Rmw::Or),
            "xor_b32" => Some(Rmw::Xor),
            "min_i32" => Some(Rmw::Min),
            "max_i32" => Some(Rmw::Max),
            "min_u32" => Some(Rmw::UMin),
            "max_u32" => Some(Rmw::UMax),
            "wrxchg_b32" => Some(Rmw::Xchg),
            "cmpst_b32" => None,
            _ => return Err("an LDS operation"),
        };
        let v0 = self.vgprs(inst, data.first().ok_or("an LDS atomic")?, 1)?[0];
        let v1 = match data.get(1) {
            Some(d) => Some(self.vgprs(inst, d, 1)?[0]),
            None => None,
        };
        let old_reg = dst_reg.and_then(|r| inst.read(Reg::V(r)).map(|v| (r, v)));
        let outside = match old_reg {
            Some((_, v)) => self.i32_of(v).unwrap_or(self.s.m.i32(0)),
            None => self.s.m.i32(0),
        };
        let p = dword(self, offset);
        let at = self.begin_guarded(guard);
        let r = match rmw {
            Some(op) => self.s.m.atomic(self.cur, op, p, v0),
            // Compare with data0, store data1.
            None => self
                .s
                .m
                .cmpxchg(self.cur, p, v0, v1.ok_or("an LDS compare")?),
        };
        let r = self.end_guarded(at, &[(r, outside)])[0];
        if let Some(reg) = dst_reg {
            self.write(inst, Reg::V(reg), r, None);
        }
        Ok(())
    }

    // ---- instructions ----

    fn inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let offset = inst.machine.offset;
        let name = inst.name();
        self.mask_ok = match self.mask_data {
            MaskData::Console => true,
            MaskData::Host => {
                name.starts_with("s_bcnt1_i32_")
                    || name.starts_with("s_ff1_i32_")
                    || name.starts_with("v_mbcnt_")
                    || matches!(
                        name,
                        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
                    )
            }
            MaskData::Count => {
                name.starts_with("s_bcnt1_i32_")
                    || matches!(
                        name,
                        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
                    )
            }
            MaskData::None => false,
        };
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read")
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute"));
        let r = match &inst.fields {
            Fields::Sop { .. } => self.sop(inst),
            Fields::Sopp { .. } if name == "s_barrier" && self.no_barrier.contains(&b) => Ok(()),
            Fields::Sopp { .. } if name == "s_barrier" => {
                // Thread-group sync with a shared-memory fence, and a UAV
                // fence when the program writes UAVs (the waits before an
                // s_barrier make those writes visible on the console).
                // Omit SYNC_THREAD_GROUP (1) when control-dependent on a
                // per-lane branch, where a 32-lane host deadlocks if one
                // warp skips the barrier while another waits on it.
                let sync = if self.fence_only_barrier.contains(&b) && self.split.is_none() {
                    0
                } else {
                    1
                };
                let mode = sync | 8 | if self.writes_uavs { 2 } else { 0 };
                let mode = self.s.m.i32(mode);
                let void = self.s.t.void;
                self.s
                    .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
                Ok(())
            }
            Fields::Sopp { .. } => Ok(()),
            Fields::Smem { .. } => self.smem(inst, (b, i)),
            Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
            Fields::Valu(_) => self.valu(inst, b, i),
            Fields::Vintrp { .. } => self.interp(inst, (b, i)),
            // The ray and node test. A failure is a refusal at once: an
            // instruction without a side effect otherwise only stops the
            // program where its results are needed.
            Fields::Image { .. } if name.starts_with("image_bvh") => {
                // A query whose structure was traced is intersected for
                // real; one that was not lowers as a miss.
                if self.x.res.bvh.iter().any(|n| n.0 == (b, i)) {
                    return self.bvh(inst, b, i).map_err(|why| unsupported(offset, why));
                }
                return self
                    .image(inst, b, i)
                    .map_err(|why| unsupported(offset, why));
            }
            Fields::Image { .. } => self.image(inst, b, i),
            Fields::Buffer { .. } => self.buffer(inst, b, i),
            Fields::Exp { .. } => self.export(b, i),
            Fields::Ds { .. } => {
                // Split: the halves share LDS as one console wave's lanes do.
                if self.split.is_some()
                    && !name.starts_with("ds_swizzle")
                    && !name.contains("permute")
                {
                    let kind = if name.contains("addtid") {
                        None
                    } else if name.starts_with("ds_read") {
                        Some(true)
                    } else if name.starts_with("ds_write") {
                        Some(false)
                    } else {
                        None
                    };
                    self.split_lds_barrier(kind);
                }
                self.ds(inst, b, i)
            }
            Fields::Flat { .. } if name.starts_with("global_") => self.flat(inst, b, i),
            Fields::Flat { .. } => Err("flat memory"),
        };
        match r {
            Ok(()) => Ok(()),
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                for &(_, v) in &inst.writes {
                    self.vals.insert(v, Val::Missing(why));
                }
                Ok(())
            }
        }
    }

    /// What the SGPR spilled to the lane `inst` (a `v_readlane_b32`) reads
    /// holds, when its vector is built by `v_writelane_b32` alone.
    fn restored_lane(&self, inst: &Inst) -> Option<crate::resource::Moved> {
        let p = self.x.p;
        let defs = self.spill_defs.get_or_init(|| {
            let mut defs = HashMap::new();
            for (b, block) in p.blocks.iter().enumerate() {
                for (i, inst) in block.insts.iter().enumerate() {
                    for &(_, w) in &inst.writes {
                        defs.insert(w, (b, i));
                    }
                }
            }
            defs
        });
        crate::resource::restored_in(
            inst,
            &|v| self.phi_at.get(&v).map(|&(b, i)| &p.blocks[b].phis[i]),
            &|v| defs.get(&v).map(|&(b, i)| &p.blocks[b].insts[i]),
        )
    }

    fn is_mask(&self, v: Option<Value>) -> bool {
        v.is_some_and(|v| self.x.st.masks.contains_key(&v))
    }

    fn sop(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Sop { dst, src, imm } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let scc = inst.write(Reg::Scc);
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        // The program counter of a join of code-table addresses
        // ([`crate::resource::CodeJoin`]): the next instruction's byte
        // offset in the code, so that the join is the offset it reads. Any
        // other read of the program counter stays refused.
        // shortcut: a read of this program counter as data gets that offset
        // too, not the console's address.
        if name == "s_getpc_b64"
            && let Some(lo) = lo
            && self.x.res.code_joins.iter().any(|j| j.pc == lo)
        {
            let pc = self.s.m.i32(inst.machine.offset as u32 + 4);
            self.set(Some(lo), Val::I32(pc));
            let zero = self.s.m.i32(0);
            self.set(
                dst_code.and_then(|c| inst.write(Reg::S(c + 1))),
                Val::I32(zero),
            );
            return Ok(());
        }
        if self.split.is_some()
            && let Some(r) = self.split_sop(inst)
        {
            return r;
        }
        if inst.write(Reg::S(EXEC)).is_some() || self.is_mask(lo) {
            if inst.write(Reg::S(EXEC)).is_none()
                && self.second
                && self.split.is_none()
                && self.console_lanes
                && sets_scc_nonzero(name)
                && let (Some(scc), Some(lo)) = (scc, lo)
            {
                let hi = match (self.x.native.wave64, dst_code) {
                    (true, Some(code)) => inst.write(Reg::S(code + 1)),
                    _ => None,
                };
                if let Some(m) = self.x.st.mask_bits(lo, hi) {
                    self.scc_any.insert(scc, (self.cur, m));
                }
            }
            return Err(LANE_MASK);
        }
        if matches!(
            name,
            "s_setreg_imm32_b32" | "s_setreg_b32" | "s_swappc_b64" | "s_version"
        ) || name.starts_with("s_waitcnt")
        {
            // shortcut: the float MODE register is not modelled; s_swappc_b64 is the
            // vertex fetch call, whose results arrive as inputs.
            return Err("a register the emitter does not model");
        }
        let c = self.cur;
        let src32 = |l: &mut Self, k: usize| -> Result<V, Why> {
            // SOPK arithmetic reads its destination as the first source.
            if k == 0 && matches!(name, "s_addk_i32" | "s_mulk_i32") {
                let d = dst.as_ref().ok_or("a missing operand")?;
                return l.operand(inst, d);
            }
            let op = src.get(k).ok_or("a missing operand")?;
            l.operand(inst, op)
        };
        let sext16 = |l: &mut Self| l.s.m.i32(i32::from(imm.unwrap_or(0) as i16) as u32);
        // Dword `h` of 64-bit source `k` (inline constants sign-extend).
        let half = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                    l.i32_of(v)
                }
                Some(op) if h == 0 => l.operand(inst, op),
                Some(Operand::Const(k)) => {
                    let x = crate::resource::inline(*k).ok_or("a special operand")?;
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                // Of a signed operand, a literal sign-extends too.
                Some(&Operand::Literal(x)) if name.ends_with("_i64") => {
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                _ => Ok(l.s.m.i32(0)),
            }
        };
        if let Some(op) = name.strip_prefix("s_bitcmp") {
            // SCC is whether bit src1 (mod 32, or mod 64 for the 64-bit form)
            // of src0 is clear (bitcmp0) or set (bitcmp1).
            let (bit, k31) = (src32(self, 1)?, self.s.m.i32(31));
            let idx = self.s.m.bin(c, Bin::And, bit, k31);
            let word = if op.ends_with("_b64") {
                let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                let k32 = self.s.m.i32(32);
                let upper = self.s.m.bin(c, Bin::And, bit, k32);
                let z = self.s.m.i32(0);
                let upper = self.s.m.cmp(c, pred::NE, upper, z);
                self.s.m.select(c, upper, b, a)
            } else {
                src32(self, 0)?
            };
            let shifted = self.s.m.bin(c, Bin::LShr, word, idx);
            let one = self.s.m.i32(1);
            let tested = self.s.m.bin(c, Bin::And, shifted, one);
            let want = self.s.m.i32(u32::from(op.starts_with('1')));
            let r = self.s.m.cmp(c, pred::EQ, tested, want);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").filter(|o| o.ends_with("_u64")) {
            // eq and lg only: both dwords.
            let mut r = None;
            for h in 0..2 {
                let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                let e = self.s.m.cmp(c, pred::EQ, a, b);
                r = Some(match r {
                    Some(x) => self.s.m.bin(c, Bin::And, x, e),
                    None => e,
                });
            }
            let mut r = r.expect("two dwords");
            if op.starts_with("lg") {
                let t = self.s.m.bool(true);
                r = self.s.m.bin(c, Bin::Xor, r, t);
            }
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
            let (a, b) = if name.starts_with("s_cmpk_") {
                let b = if op.ends_with("_i32") {
                    sext16(self)
                } else {
                    self.s.m.i32(u32::from(imm.unwrap_or(0)))
                };
                (src32(self, 0)?, b)
            } else {
                if op.ends_with("64") {
                    return Err("a 64-bit scalar compare");
                }
                (src32(self, 0)?, src32(self, 1)?)
            };
            let signed = op.ends_with("_i32");
            let p = match &op[..2] {
                "eq" => pred::EQ,
                "lg" => pred::NE,
                "gt" if signed => pred::SGT,
                "ge" if signed => pred::SGE,
                "lt" if signed => pred::SLT,
                "le" if signed => pred::SLE,
                "gt" => pred::IUGT,
                "ge" => pred::IUGE,
                "lt" => pred::IULT,
                "le" => pred::IULE,
                _ => return Err("a scalar compare"),
            };
            let r = self.s.m.cmp(c, p, a, b);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if matches!(
            name,
            "s_bitcmp0_b32" | "s_bitcmp1_b32" | "s_bitcmp0_b64" | "s_bitcmp1_b64"
        ) {
            let want_one = name.starts_with("s_bitcmp1_");
            let wide = name.ends_with("_b64");
            let bit = src32(self, 1)?;
            let (k31, one, zero) = (self.s.m.i32(31), self.s.m.i32(1), self.s.m.i32(0));
            let word = if wide {
                let (lo_w, hi_w) = (half(self, 0, 0)?, half(self, 0, 1)?);
                let k32 = self.s.m.i32(32);
                let hi_bit = self.s.m.bin(c, Bin::And, bit, k32);
                let is_hi = self.s.m.cmp(c, pred::NE, hi_bit, zero);
                self.s.m.select(c, is_hi, hi_w, lo_w)
            } else {
                src32(self, 0)?
            };
            let shift = self.s.m.bin(c, Bin::And, bit, k31);
            let shifted = self.s.m.bin(c, Bin::LShr, word, shift);
            let masked = self.s.m.bin(c, Bin::And, shifted, one);
            let p = if want_one { pred::NE } else { pred::EQ };
            let r = self.s.m.cmp(c, p, masked, zero);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        if wide {
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            let nonzero = |l: &mut Self, a: V, b: V| {
                let o = l.s.m.bin(c, Bin::Or, a, b);
                let z = l.s.m.i32(0);
                l.s.m.cmp(c, pred::NE, o, z)
            };
            return match name {
                "s_quadmask_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let low = self.quad_mask(a);
                    let high = self.quad_mask(b);
                    let k8 = self.s.m.i32(8);
                    let high = self.s.m.bin(c, Bin::Shl, high, k8);
                    let mask = self.s.m.bin(c, Bin::Or, low, high);
                    let zero = self.s.m.i32(0);
                    self.set(lo, Val::I32(mask));
                    self.set(hi, Val::I32(zero));
                    let nonzero = self.s.m.cmp(c, pred::NE, mask, zero);
                    self.set(scc, Val::Bit(nonzero));
                    Ok(())
                }
                "s_bcnt1_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let na = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[a]);
                    let nb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[b]);
                    let r = self.s.m.bin(c, Bin::Add, na, nb);
                    self.set(lo, Val::I32(r));
                    let z = self.s.m.i32(0);
                    let s = self.s.m.cmp(c, pred::NE, r, z);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_ff1_i32_b64" => {
                    // The lowest set bit; -1 for none (as DXIL's first bit low).
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let fa = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[a]);
                    let fb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[b]);
                    let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                    let fb32 = self.s.m.bin(c, Bin::Add, fb, k32);
                    let none = self.s.m.cmp(c, pred::EQ, b, z);
                    let fb = self.s.m.select(c, none, fb, fb32);
                    let low = self.s.m.cmp(c, pred::NE, a, z);
                    let r = self.s.m.select(c, low, fa, fb);
                    self.set(lo, Val::I32(r));
                    Ok(())
                }
                "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
                | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" | "s_not_b64" => {
                    let ones = self.s.m.i32(u32::MAX);
                    let mut r = Vec::new();
                    for h in 0..2u8 {
                        let a = half(self, 0, h)?;
                        let x = if name == "s_not_b64" {
                            self.s.m.bin(c, Bin::Xor, a, ones)
                        } else {
                            let mut b = half(self, 1, h)?;
                            if name.contains("n2") {
                                b = self.s.m.bin(c, Bin::Xor, b, ones);
                            }
                            let op = match &name[2..5] {
                                "and" | "nan" => Bin::And,
                                "or_" | "orn" | "nor" => Bin::Or,
                                _ => Bin::Xor,
                            };
                            let x = self.s.m.bin(c, op, a, b);
                            if matches!(name, "s_nand_b64" | "s_nor_b64" | "s_xnor_b64") {
                                self.s.m.bin(c, Bin::Xor, x, ones)
                            } else {
                                x
                            }
                        };
                        r.push(x);
                    }
                    self.set(lo, Val::I32(r[0]));
                    self.set(hi, Val::I32(r[1]));
                    let s = nonzero(self, r[0], r[1]);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let n = src32(self, 1)?;
                    let i64t = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let a = self.s.m.cast(c, Cast::ZExt, a, i64t);
                    let b = self.s.m.cast(c, Cast::ZExt, b, i64t);
                    let k32 = self.s.m.int_const(i64t, 32);
                    let b = self.s.m.bin(c, Bin::Shl, b, k32);
                    let x = self.s.m.bin(c, Bin::Or, a, b);
                    let k63 = self.s.m.i32(63);
                    let n = self.s.m.bin(c, Bin::And, n, k63);
                    let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                    let op = match name {
                        "s_lshl_b64" => Bin::Shl,
                        "s_lshr_b64" => Bin::LShr,
                        _ => Bin::AShr,
                    };
                    let r = self.s.m.bin(c, op, x, n);
                    let rl = self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32);
                    let rh = self.s.m.bin(c, Bin::LShr, r, k32);
                    let rh = self.s.m.cast(c, Cast::Trunc, rh, self.s.t.i32);
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    let s = nonzero(self, rl, rh);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_mov_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    self.set(lo, Val::I32(a));
                    self.set(hi, Val::I32(b));
                    Ok(())
                }
                "s_cselect_b64" => {
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                        let r = self.s.m.select(c, cond, a, b);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                _ => Err(name),
            };
        }
        let m = &mut self.s.m;
        let _ = m;
        let (r, carry): (V, Option<V>) = match name {
            "s_mov_b32" => (src32(self, 0)?, None),
            "s_movk_i32" => (sext16(self), None),
            "s_quadmask_b32" => {
                let pixels = src32(self, 0)?;
                (self.quad_mask(pixels), None)
            }
            "s_cselect_b32" => {
                let cond = self.scc_in(inst)?;
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.s.m.select(c, cond, a, b), None)
            }
            // The destination keeps its value where SCC is clear.
            "s_cmov_b32" => {
                let cond = self.scc_in(inst)?;
                let new = src32(self, 0)?;
                let d = dst.as_ref().ok_or("a missing operand")?;
                let old = self.operand(inst, d)?;
                (self.s.m.select(c, cond, new, old), None)
            }
            "s_brev_b32" => {
                let a = src32(self, 0)?;
                let r = self
                    .s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a]);
                (r, None)
            }
            "s_sext_i32_i8" | "s_sext_i32_i16" => {
                let a = src32(self, 0)?;
                let shift = self.s.m.i32(if name == "s_sext_i32_i8" { 24 } else { 16 });
                let t = self.s.m.bin(c, Bin::Shl, a, shift);
                (self.s.m.bin(c, Bin::AShr, t, shift), None)
            }
            "s_not_b32" => {
                let a = src32(self, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                (self.s.m.bin(c, Bin::Xor, a, ones), None)
            }
            "s_add_u32" | "s_add_i32" | "s_addk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_addk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                let r = self.s.m.bin(c, Bin::Add, a, b);
                let carry = if name == "s_add_u32" {
                    Some(self.s.m.cmp(c, pred::IULT, r, a))
                } else {
                    None
                };
                (r, carry)
            }
            "s_sub_u32" | "s_sub_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let r = self.s.m.bin(c, Bin::Sub, a, b);
                let carry = (name == "s_sub_u32").then(|| self.s.m.cmp(c, pred::IULT, a, b));
                (r, carry)
            }
            "s_addc_u32" | "s_subb_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let cin = self.scc_in(inst)?;
                let (r, carry) = self.add_sub_carry(c, name == "s_addc_u32", a, b, Some(cin));
                (r, Some(carry))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_mulk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                (self.s.m.bin(c, Bin::Mul, a, b), None)
            }
            "s_mul_hi_u32" | "s_mul_hi_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.mul_hi(a, b, name.ends_with("i32")), None)
            }
            "s_flbit_i32_b32" | "s_flbit_i32" => {
                // The first bit from the top (differing from the sign for
                // the signed form), -1 for none: DXIL's first-bit-high.
                let a = src32(self, 0)?;
                let op = if name == "s_flbit_i32_b32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_ff1_i32_b32" | "s_bcnt1_i32_b32" => {
                // First set bit (-1 for zero, as DXIL's), or the bit count.
                let a = src32(self, 0)?;
                let op = if name == "s_ff1_i32_b32" { 32 } else { 31 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                // src1: offset in bits 0-4, width in bits 16-22.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let (k16, k127) = (self.s.m.i32(16), self.s.m.i32(0x7F));
                let w = self.s.m.bin(c, Bin::LShr, b, k16);
                let w = self.s.m.bin(c, Bin::And, w, k127);
                (self.bfe(a, b, w, name == "s_bfe_i32"), None)
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                let (a, mut b) = (src32(self, 0)?, src32(self, 1)?);
                let ones = self.s.m.i32(u32::MAX);
                if name.contains("n2") {
                    b = self.s.m.bin(c, Bin::Xor, b, ones);
                }
                let op = match &name[2..5] {
                    "and" | "nan" => Bin::And,
                    "or_" | "orn" | "nor" => Bin::Or,
                    _ => Bin::Xor,
                };
                let mut r = self.s.m.bin(c, op, a, b);
                if matches!(name, "s_nand_b32" | "s_nor_b32" | "s_xnor_b32") {
                    r = self.s.m.bin(c, Bin::Xor, r, ones);
                }
                (r, None)
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let op = match name {
                    "s_lshl_b32" => Bin::Shl,
                    "s_lshr_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                (self.s.m.bin(c, op, a, b), None)
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k = self.s.m.i32(u32::from(name.as_bytes()[6] - b'0'));
                let t = self.s.m.bin(c, Bin::Shl, a, k);
                (self.s.m.bin(c, Bin::Add, t, b), None)
            }
            "s_min_u32" | "s_max_u32" | "s_min_i32" | "s_max_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let p = match name {
                    "s_min_u32" => pred::IULT,
                    "s_max_u32" => pred::IUGT,
                    "s_min_i32" => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                (self.s.m.select(c, t, a, b), None)
            }
            "s_abs_i32" => {
                let a = src32(self, 0)?;
                let z = self.s.m.i32(0);
                let neg = self.s.m.cmp(c, pred::SLT, a, z);
                let n = self.s.m.bin(c, Bin::Sub, z, a);
                (self.s.m.select(c, neg, n, a), None)
            }
            "s_absdiff_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let d = self.s.m.bin(c, Bin::Sub, a, b);
                let z = self.s.m.i32(0);
                let neg = self.s.m.cmp(c, pred::SLT, d, z);
                let n = self.s.m.bin(c, Bin::Sub, z, d);
                (self.s.m.select(c, neg, n, d), None)
            }
            "s_pack_ll_b32_b16" | "s_pack_lh_b32_b16" | "s_pack_hh_b32_b16" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let (k16, lo_mask, hi_mask) = (
                    self.s.m.i32(16),
                    self.s.m.i32(0xFFFF),
                    self.s.m.i32(0xFFFF_0000),
                );
                let lo_half = if name == "s_pack_hh_b32_b16" {
                    self.s.m.bin(c, Bin::LShr, a, k16)
                } else {
                    self.s.m.bin(c, Bin::And, a, lo_mask)
                };
                let hi_half = if name == "s_pack_ll_b32_b16" {
                    self.s.m.bin(c, Bin::Shl, b, k16)
                } else {
                    self.s.m.bin(c, Bin::And, b, hi_mask)
                };
                (self.s.m.bin(c, Bin::Or, lo_half, hi_half), None)
            }
            "s_bitset0_b32" | "s_bitset1_b32" => {
                // Sets (clears) bit src[4:0] of the destination it reads.
                let old = self.operand(inst, dst.as_ref().ok_or("a missing operand")?)?;
                let a = src32(self, 0)?;
                let (k31, one) = (self.s.m.i32(31), self.s.m.i32(1));
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let bit = self.s.m.bin(c, Bin::Shl, one, a);
                if name == "s_bitset1_b32" {
                    (self.s.m.bin(c, Bin::Or, old, bit), None)
                } else {
                    let ones = self.s.m.i32(u32::MAX);
                    let nb = self.s.m.bin(c, Bin::Xor, bit, ones);
                    (self.s.m.bin(c, Bin::And, old, nb), None)
                }
            }
            "s_bfm_b32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let one = self.s.m.i32(1);
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                (self.s.m.bin(c, Bin::Shl, m, b), None)
            }
            _ => return Err(name),
        };
        self.set(lo, Val::I32(r));
        if let Some(scc) = scc {
            let val = match carry {
                Some(k) => Val::Bit(k),
                None if matches!(
                    name,
                    "s_mov_b32"
                        | "s_movk_i32"
                        | "s_cselect_b32"
                        | "s_cmov_b32"
                        | "s_brev_b32"
                        | "s_sext_i32_i8"
                        | "s_sext_i32_i16"
                        | "s_mul_i32"
                        | "s_mulk_i32"
                ) =>
                {
                    return Ok(());
                }
                // s_lshlN_add_u32 sets SCC to the add's carry, not to
                // "result != 0".
                None if name.starts_with("s_add")
                    || name.starts_with("s_sub")
                    || name.starts_with("s_lshl") && name.ends_with("_add_u32") =>
                {
                    Val::Missing("the carry of a scalar operation")
                }
                None => {
                    let z = self.s.m.i32(0);
                    Val::Bit(self.s.m.cmp(c, pred::NE, r, z))
                }
            };
            self.vals.insert(scc, val);
        }
        Ok(())
    }

    /// Packs whether each consecutive group of four pixels has a set bit.
    fn quad_mask(&mut self, pixels: V) -> V {
        let c = self.cur;
        let one = self.s.m.i32(1);
        let shifted = self.s.m.bin(c, Bin::LShr, pixels, one);
        let mut mask = self.s.m.bin(c, Bin::Or, pixels, shifted);
        // Fold each nibble into its bottom bit, then gather those bits into
        // the low byte in three stages without needing a 64-bit operation.
        for (distance, keep) in [
            (2, 0x1111_1111),
            (3, 0x0303_0303),
            (6, 0x000f_000f),
            (12, 0x0000_00ff),
        ] {
            let distance = self.s.m.i32(distance);
            let shifted = self.s.m.bin(c, Bin::LShr, mask, distance);
            let combined = self.s.m.bin(c, Bin::Or, mask, shifted);
            let keep = self.s.m.i32(keep);
            mask = self.s.m.bin(c, Bin::And, combined, keep);
        }
        mask
    }

    /// `a + b (+ cin)` or `a - b (- cin)` with the carry (borrow) out, as the
    /// scalar and vector add-with-carry instructions set it.
    fn add_sub_carry(&mut self, c: usize, add: bool, a: V, b: V, cin: Option<V>) -> (V, V) {
        let op = if add { Bin::Add } else { Bin::Sub };
        let mut r = self.s.m.bin(c, op, a, b);
        // Without a carry in: a + b wraps below a; a - b borrows when a < b.
        let mut carry = if add {
            self.s.m.cmp(c, pred::IULT, r, a)
        } else {
            self.s.m.cmp(c, pred::IULT, a, b)
        };
        if let Some(cin) = cin {
            let one = self.s.m.cast(c, Cast::ZExt, cin, self.s.t.i32);
            let t = r;
            r = self.s.m.bin(c, op, t, one);
            // The carry in carries on when the partial result is all ones
            // (add) or zero (sub).
            let edge = self.s.m.i32(if add { u32::MAX } else { 0 });
            let at_edge = self.s.m.cmp(c, pred::EQ, t, edge);
            let more = self.s.m.bin(c, Bin::And, cin, at_edge);
            carry = self.s.m.bin(c, Bin::Or, carry, more);
        }
        (r, carry)
    }

    /// VOP3 integer `clamp`: unsigned saturation from the carry (add →
    /// `0xFFFF_FFFF`) or borrow (sub → `0`) bit.
    fn clamp_u32(&mut self, c: usize, add: bool, r: V, carry: V) -> V {
        let sat = self.s.m.i32(if add { u32::MAX } else { 0 });
        self.s.m.select(c, carry, sat, r)
    }

    fn scc_in(&mut self, inst: &Inst) -> Result<V, Why> {
        let v = inst.read(Reg::Scc).ok_or("an unread SCC")?;
        match self.get(v)? {
            Val::Bit(x) => Ok(x),
            _ => Err("SCC without a value"),
        }
    }

    fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Fields::Smem {
            data,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, a);
        }
        if self.scalar.get(&site).is_some_and(|a| a.param == MEMORY) {
            return self.memory_smem(inst);
        }
        let Some(rest) = name.strip_prefix("s_buffer_load_dword") else {
            return self.descriptor_data(inst, site);
        };
        let count = match rest {
            "" => 1,
            "x2" => 2,
            "x4" => 4,
            "x8" => 8,
            "x16" => 16,
            _ => return Err("a scalar buffer load"),
        };
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar buffer load");
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked descriptor")?;
        if matches!(src, Source::Built { address, .. } if address.param == MEMORY) {
            return self.memory_sbuffer(inst, src);
        }
        let kind = self.used_kind(src);
        let c = self.cur;
        let (base, extra) = self.built_offset(src)?;
        // Hardware clears the low 2 bits of each *instruction* addend
        // (soffset, immediate offset) before summing, so soffset=3 and
        // offset:3 load byte 0, not byte 4 from carrying the low bits across.
        // The built descriptor address (constant `base` + computed `extra`)
        // is one operand already combined by `resource::combine_sum`: add
        // those parts first, then truncate their sum, so carries from
        // constructing the descriptor are kept.
        let align = |l: &mut Self, v: V| match l.s.m.int_value(v) {
            Some(n) => l.s.m.i32((n as u32) & !3),
            None => {
                let not3 = l.s.m.i32(!3);
                l.s.m.bin(c, Bin::And, v, not3)
            }
        };
        let add = |l: &mut Self, a: V, b: V| match (l.s.m.int_value(a), l.s.m.int_value(b)) {
            (Some(x), Some(y)) => l.s.m.i32((x as u32).wrapping_add(y as u32)),
            _ => l.s.m.bin(c, Bin::Add, a, b),
        };
        let mut built = self.s.m.i32(base);
        if let Some(x) = extra {
            built = add(self, built, x);
        }
        let built = align(self, built);
        let imm = self.s.m.i32((*offset as u32) & !3);
        let mut addr = add(self, built, imm);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            let so = align(self, so);
            addr = add(self, addr, so);
        }
        #[cfg(test)]
        if let Some(v) = self.s.m.int_value(addr) {
            self.s.smem_buffer_bytes.push(v as u32);
        }
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            // A descriptor built around a table in the code.
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.code_dword(a)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        // A descriptor of no listed kind (one fetched from the heap by
        // index) that vector buffer loads read, and none write, is an SRV in
        // the title's heap, not a constant buffer: reading its slot through
        // a constant buffer handle is a descriptor of the wrong type. So is
        // one fetched from the resource heap by index that nothing writes:
        // the title's heap-indexed buffers are views (a raw buffer that
        // only scalar loads read was found read through a constant buffer
        // handle, which NVIDIA reads as another address).
        if let Some(((stride, uav), heap_only)) = self.scalar_view(src, kind) {
            let h = self.handle(site, src, false, Self::buffer_props(stride, uav))?;
            // A view only scalar loads read skips heap index 0, as the
            // constant buffer reads below do (the title's null texture).
            let live = if heap_only && self.is_heap_direct(src) {
                let index = self
                    .uniform_heap_index(src, false)
                    .or_else(|_| self.lane_heap_index(src, false))?;
                let zero = self.s.m.i32(0);
                Some(self.s.m.cmp(self.cur, pred::NE, index, zero))
            } else {
                None
            };
            let zero = self.s.m.i32(0);
            for k in 0..count {
                let at = self.begin_guarded(live);
                let a = self.add_bytes(addr, 4 * k);
                let x = match stride {
                    Some(stride) => self.structured_load(h, a, stride, 4),
                    None => {
                        let iu = self.s.m.undef(self.s.t.i32);
                        self.buffer_dword(h, a, iu)
                    }
                };
                let x = self.end_guarded(at, &[(x, zero)])[0];
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let h = self.handle(site, src, false, (13, 0x1_0000))?;
        // shortcut: untyped heap-direct scalar CBV loads skip index 0. This
        // title parks a null *texture* SRV at heap slot 0 that shaders also
        // sample as an SRV; a single host descriptor cannot be both kinds,
        // and a raw-buffer handle would still mismatch that texture null
        // (#485). On the console a CBV read through that null returns zeros.
        // Listed ResourceHeap CBVs (used-resource metadata) keep a live load
        // at index 0. Table CBVs and heap slots that vector buffer loads
        // already classify as SRVs are unchanged (paths above).
        let live = if kind.is_none() && self.is_heap_direct(src) {
            let index = self
                .uniform_heap_index(src, false)
                .or_else(|_| self.lane_heap_index(src, false))?;
            let zero = self.s.m.i32(0);
            Some(self.s.m.cmp(self.cur, pred::NE, index, zero))
        } else {
            None
        };
        let constant = self.s.m.int_value(addr).map(|v| v as u32);
        let mut rows: HashMap<u32, V> = HashMap::new();
        let zero = self.s.m.i32(0);
        for k in 0..count {
            let at = self.begin_guarded(live);
            let x = match constant {
                Some(base) => {
                    // Shader byte arithmetic wraps; debug builds must not
                    // panic when a near-end constant address expands to
                    // dwordxN (e.g. soffset 0xfffffffc, dwordx2).
                    let byte = base.wrapping_add(4 * k);
                    // Row loads are emitted inside the guard so a null-slot
                    // index never executes cbufferLoadLegacy.
                    let ri = self.s.m.i32(byte >> 4);
                    let r = match rows.get(&(byte >> 4)) {
                        Some(&r) => r,
                        None => {
                            let r = self.s.op(
                                self.cur,
                                "cbufferLoadLegacy.i32",
                                Attr::ReadOnly,
                                self.s.t.cbuf_ret,
                                59,
                                &[h, ri],
                            );
                            if live.is_none() {
                                rows.insert(byte >> 4, r);
                            }
                            r
                        }
                    };
                    self.s.m.extract(self.cur, r, (byte >> 2) & 3)
                }
                None => {
                    let off = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(self.cur, Bin::Add, addr, off);
                    self.cbuffer_dword(h, a)
                }
            };
            let x = self.end_guarded(at, &[(x, zero)])[0];
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// Splits byte offset `addr` into `(element_index, element_offset)` for a
    /// structured SRV whose `stride` is a multiple of 4 (`stride >= 4`).
    ///
    /// Uses shifts/masks for power-of-two strides and 32-bit half-product
    /// reciprocal multiplication for non-power-of-two strides instead of
    /// `udiv`/`urem`: NVIDIA's DXIL compiler (`nvgpucomp64.dll`) hits a null
    /// dereference (`E_OUTOFMEMORY`) inside `CreatePipelineState` on graphics
    /// stages (`ms`/`vs`/`ps`) when a `waveReadLane*`-derived uniform offset
    /// passes through `udiv(addr, stride)` (or `dx.op.binaryWithTwoOutputs`
    /// `UMul`) into `rawBufferLoad` on a `StructuredBuffer` with that stride.
    /// Because `stride` is a multiple of 4, `addr >> tz < 2^30`, so the
    /// Granlund-Montgomery multiplier fits in `u32` with no carry step and is
    /// exact for every `u32` without requiring `INT64` or `WAVE_OPS`.
    fn structured_index_and_offset(&mut self, addr: V, stride: u32) -> (V, V) {
        debug_assert!(stride >= 4 && stride.is_multiple_of(4));
        if let Some(byte) = self.s.m.int_value(addr) {
            let byte = byte as u32;
            return (self.s.m.i32(byte / stride), self.s.m.i32(byte % stride));
        }
        let c = self.cur;
        let tz = stride.trailing_zeros();
        let odd = stride >> tz;
        if odd == 1 {
            let sh = self.s.m.i32(tz);
            let mask = self.s.m.i32(stride - 1);
            return (
                self.s.m.bin(c, Bin::LShr, addr, sh),
                self.s.m.bin(c, Bin::And, addr, mask),
            );
        }
        let (magic, shift) = structured_div_magic(odd);
        let tz_v = self.s.m.i32(tz);
        let shifted = self.s.m.bin(c, Bin::LShr, addr, tz_v);
        let k16 = self.s.m.i32(16);
        let mask16 = self.s.m.i32(0xffff);
        let a_lo = self.s.m.bin(c, Bin::And, shifted, mask16);
        let a_hi = self.s.m.bin(c, Bin::LShr, shifted, k16);
        let m_lo = self.s.m.i32(magic & 0xffff);
        let m_hi = self.s.m.i32(magic >> 16);
        let p0 = self.s.m.bin(c, Bin::Mul, a_lo, m_lo);
        let p1 = self.s.m.bin(c, Bin::Mul, a_lo, m_hi);
        let p2 = self.s.m.bin(c, Bin::Mul, a_hi, m_lo);
        let p3 = self.s.m.bin(c, Bin::Mul, a_hi, m_hi);
        let p0_hi = self.s.m.bin(c, Bin::LShr, p0, k16);
        let p1_lo = self.s.m.bin(c, Bin::And, p1, mask16);
        let p2_lo = self.s.m.bin(c, Bin::And, p2, mask16);
        let mid = self.s.m.bin(c, Bin::Add, p0_hi, p1_lo);
        let mid = self.s.m.bin(c, Bin::Add, mid, p2_lo);
        let mid_hi = self.s.m.bin(c, Bin::LShr, mid, k16);
        let p1_hi = self.s.m.bin(c, Bin::LShr, p1, k16);
        let p2_hi = self.s.m.bin(c, Bin::LShr, p2, k16);
        let hi = self.s.m.bin(c, Bin::Add, p3, p1_hi);
        let hi = self.s.m.bin(c, Bin::Add, hi, p2_hi);
        let hi = self.s.m.bin(c, Bin::Add, hi, mid_hi);
        let shift_v = self.s.m.i32(shift);
        let index = self.s.m.bin(c, Bin::LShr, hi, shift_v);
        let stride_v = self.s.m.i32(stride);
        let elem_base = self.s.m.bin(c, Bin::Mul, index, stride_v);
        let elem_off = self.s.m.bin(c, Bin::Sub, addr, elem_base);
        (index, elem_off)
    }

    /// Two scalar loads of the same dwords: same opcode and immediate
    /// offset, base read from the same values, and the same or an
    /// equivalent register offset.
    fn twin_loads(&self, a: Site, b: Site) -> bool {
        let (ia, ib): (&Inst, &Inst) = (
            &self.x.p.blocks[a.0].insts[a.1],
            &self.x.p.blocks[b.0].insts[b.1],
        );
        let (
            Fields::Smem {
                base: Operand::Scalar { code: ba, .. },
                offset: oa,
                soffset: sa,
                ..
            },
            Fields::Smem {
                base: Operand::Scalar { code: bb, .. },
                offset: ob,
                soffset: sb,
                ..
            },
        ) = (&ia.fields, &ib.fields)
        else {
            return false;
        };
        let same_base = (0..2).all(|k| {
            let va = ia.read(Reg::S(ba + k));
            va.is_some() && va == ib.read(Reg::S(bb + k))
        });
        let same_soffset = match (sa, sb) {
            (Operand::Scalar { code: NULL, .. }, Operand::Scalar { code: NULL, .. }) => true,
            (Operand::Scalar { code: ca, .. }, Operand::Scalar { code: cb, .. }) => {
                match (ia.read(Reg::S(*ca)), ib.read(Reg::S(*cb))) {
                    (Some(x), Some(y)) => self.same_scalar(x, y),
                    _ => false,
                }
            }
            _ => false,
        };
        ia.name() == ib.name() && oa == ob && same_base && same_soffset
    }

    /// The same value, or two `v_readfirstlane_b32` of the same VGPR value
    /// (each waterfall loop reads the index anew).
    fn same_scalar(&self, x: Value, y: Value) -> bool {
        if x == y {
            return true;
        }
        let first_lane_of = |v: Value| {
            self.x
                .p
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .find(|i| i.writes.iter().any(|w| w.1 == v))
                .filter(|i| i.name() == "v_readfirstlane_b32")
                .and_then(|i| {
                    i.reads
                        .iter()
                        .find(|r| matches!(r.0, Reg::V(_)))
                        .map(|r| r.1)
                })
        };
        matches!((first_lane_of(x), first_lane_of(y)), (Some(a), Some(b)) if a == b)
    }

    /// The element stride used by scalar data loads through this SRV.
    fn scalar_srv_stride(&self, src: &Source, srv: Option<&UsedResource>) -> Option<u32> {
        srv.filter(|u| u.root == RootType::Table && u.stride > 4 && u.stride.is_multiple_of(4))
            .map(|u| u.stride)
            .or_else(|| self.bound_structured_stride(src, false, false))
    }

    /// Whether every dword `inst` writes is read only as the descriptor of
    /// `image_bvh*` instructions (directly or through phis), which never
    /// look at it.
    fn read_only_by_bvh(&self, inst: &Inst) -> bool {
        let mut pending: Vec<Value> = inst.writes.iter().map(|w| w.1).collect();
        let mut seen = HashSet::new();
        let mut reads = 0;
        while let Some(v) = pending.pop() {
            if !seen.insert(v) {
                continue;
            }
            for block in &self.x.p.blocks {
                for phi in block.phis.iter().filter(|phi| phi.args.contains(&v)) {
                    pending.push(phi.value);
                }
                for reader in block
                    .insts
                    .iter()
                    .filter(|r| r.reads.iter().any(|r| r.1 == v))
                {
                    let rsrc = match &reader.fields {
                        Fields::Image {
                            rsrc: Operand::Scalar { code, dwords },
                            ..
                        } if reader.name().starts_with("image_bvh") => (*code, *dwords),
                        _ => return false,
                    };
                    let in_rsrc = reader.reads.iter().any(|r| {
                        r.1 == v
                            && matches!(r.0, Reg::S(n) if (rsrc.0..rsrc.0 + rsrc.1).contains(&n))
                    });
                    if !in_rsrc {
                        return false;
                    }
                    reads += 1;
                }
            }
        }
        reads > 0
    }

    /// A descriptor load (`s_load_dword*` from a table or heap) whose dwords
    /// the code also reads as numbers: the console compiler computes sizes
    /// from descriptor bits (an image's width, height, depth, levels; a
    /// buffer's record count) instead of querying them. Those fields are
    /// rebuilt from the host view's dimensions in the GFX10 layout (image
    /// dword 1 bits 30-31 and dword 2 bits 0-13 width - 1, dword 2 bits
    /// 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
    /// 28, dword 4 depth or layers - 1; buffer dword 2 the record count).
    /// A buffer descriptor no memory instruction takes is read for its base
    /// address too, from the host's view address list
    /// ([`Self::view_address_data`]).
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right) and the other
    // dwords (an image's address, formats, tiling) have no host value:
    // reading them stops the program.
    fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        if self.atomic_descriptor_loads.contains(&site) {
            return Ok(());
        }
        if self.read_only_by_bvh(inst) {
            return Ok(());
        }
        const OTHER: Why = "a descriptor's address or format read as data";
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data"),
        };
        // How the program uses the descriptor tells the view's kind: its
        // own uses, else those of a twin load of the same dwords (a
        // non-uniform index reloads the descriptor in each waterfall loop,
        // one for the size bits, another for the access).
        let uses = |twin: &dyn Fn(Site) -> bool| {
            self.x
                .res
                .descriptors
                .iter()
                .filter(|d| d.role == Role::Resource)
                .find_map(|d| {
                    d.source.leaves().into_iter().find_map(|l| match *l {
                        Source::Loaded { load, first } if twin(load) => Some((d.site, first)),
                        _ => None,
                    })
                })
        };
        let use_at =
            uses(&|load| load == site).or_else(|| uses(&|load| self.twin_loads(site, load)));
        let (first, src, props, image, uav) = if let Some((at, first)) = use_at {
            let src = Source::Loaded { load: site, first };
            let used = self.used_kind(&src);
            let ui: &Inst = &self.x.p.blocks[at.0].insts[at.1];
            let un = ui.name();
            let uav = un.contains("store")
                || un.contains("atomic")
                || used.is_some_and(|u| u.kind == ResourceType::Uav);
            let u = if uav { 0x1000 } else { 0 };
            let (props, image) = match &ui.fields {
                // A ray-tracing structure in a buffer of the host: read raw.
                Fields::Image { .. } if un.starts_with("image_bvh") => ((11, 0), None),
                Fields::Image { dim, .. } => {
                    let kind = match dim {
                        0 => 1,
                        1 => 2,
                        2 => 4,
                        4 => 6,
                        5 => 7,
                        _ => return Err("an image dimension (cube or MSAA)"),
                    };
                    ((kind | u, 0x409), Some(kind))
                }
                Fields::Buffer { idxen, format, .. } => {
                    let props = if format.is_none()
                        && (un.starts_with("buffer_load_format")
                            || un.starts_with("buffer_store_format"))
                    {
                        (10 | u, 0x409)
                    } else {
                        let uav = uav
                            || (*idxen
                                && format.is_none()
                                && used.is_none()
                                && self
                                    .descriptors
                                    .get(&(at, false))
                                    .is_some_and(|src| !self.buffer_loads_only(src)));
                        let stride = if *idxen && format.is_none() {
                            match self.indexed_buffer_stride(&src, uav) {
                                Ok(stride) => stride,
                                // The element count from the host's table,
                                // as the loads through it take the stride.
                                Err(why) if !uav => {
                                    return self
                                        .table_descriptor_data(inst, code, count, first, &src)
                                        .map_err(|_| why);
                                }
                                Err(why) => return Err(why),
                            }
                        } else {
                            self.table_buffer_stride(&src, used, uav)
                        };
                        Self::buffer_props(stride, uav)
                    };
                    if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                        return Err("a constant buffer's descriptor read as data");
                    }
                    (props, None)
                }
                Fields::Smem { .. } if un.starts_with("s_buffer_load_dword") => {
                    // A scalar buffer load consumes its descriptor without
                    // reading word 2 as a count. Keep that path free of a
                    // dimensions query unless another operand needs the count.
                    let k = u32::from(first) + 2;
                    if k >= count
                        || inst
                            .write(Reg::S(code + k as u8))
                            .is_none_or(|v| !self.scalar_record_count_used(v))
                    {
                        return Err("a scalar memory read as data");
                    }
                    // The view the loads themselves take (see `smem`).
                    match self.scalar_view(&src, used) {
                        Some(((stride, uav), _)) => (Self::buffer_props(stride, uav), None),
                        None if self.leftovers => ((11, 0), None),
                        None => return Err("a scalar memory read as data"),
                    }
                }
                _ => return Err("a scalar memory read as data"),
            };
            (first, src, props, image, uav)
        } else {
            let src = Source::Loaded {
                load: site,
                first: 0,
            };
            let used = self.used_kind(&src);
            let uav = used.is_some_and(|u| u.kind == ResourceType::Uav);
            if let Some(stride) = (count == 4)
                .then(|| self.table_buffer_stride(&src, used, uav))
                .flatten()
            {
                (0, src, Self::buffer_props(Some(stride), uav), None, uav)
            } else if count == 8 && used.is_none_or(|u| u.kind == ResourceType::Srv) {
                // shortcut: a size-only image descriptor is taken as a 2D texture.
                (0, src, (2, 0x409), Some(2), false)
            } else if count <= 4
                && self.leftovers
                // Not a sampler's descriptor, which a sample takes whole.
                && !self.x.res.descriptors.iter().any(|d| {
                    d.source
                        .leaves()
                        .into_iter()
                        .any(|l| matches!(*l, Source::Loaded { load, .. } if load == site))
                })
                && (0..count.min(3))
                    .filter_map(|k| inst.write(Reg::S(code + k as u8)))
                    .any(|v| self.demanded_bits(v) != 0)
            {
                // A buffer descriptor read for its base address or record
                // count. In the last chance only, and only when those words
                // reach something, so that every other program keeps its
                // user data as it was.
                return self.view_address_data(inst, code, count, &src);
            } else {
                return Err("a descriptor read only as data");
            }
        };
        let h = self.handle(site, &src, false, props)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = if image.is_some() && !uav {
            self.s.m.i32(0)
        } else {
            self.s.m.undef(i32t)
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
        let words: Vec<(u32, V)> = match image {
            Some(kind) => {
                let one = self.s.m.i32(1);
                let w = field(self, 0);
                let (height, depth) = match kind {
                    1 => (one, one),
                    2 => (field(self, 1), one),
                    6 => (one, field(self, 1)),
                    _ => (field(self, 1), field(self, 2)),
                };
                // The level count only feeds word 3. A level query of a 3D
                // image inside a loop crashes NVIDIA's Vulkan compiler (vkd3d-
                // proton turns it into OpImageQueryLevels), so query it only
                // when the program reads that word as data.
                let reads_word3 = (0..count)
                    .filter(|&k| k.wrapping_sub(u32::from(first)) == 3)
                    .filter_map(|k| inst.write(Reg::S(code + k as u8)))
                    .any(|v| self.read_as_data(v));
                let levels = if uav || !reads_word3 {
                    one
                } else {
                    field(self, 3)
                };
                let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
                let (w, height, depth, levels) = (
                    less(self, w),
                    less(self, height),
                    less(self, depth),
                    less(self, levels),
                );
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (k30, k2, k14, k16, mask) = (
                    k(self, 30),
                    k(self, 2),
                    k(self, 14),
                    k(self, 16),
                    k(self, 0x3FFF),
                );
                let d1 = self.s.m.bin(c, Bin::Shl, w, k30);
                let hi = self.s.m.bin(c, Bin::LShr, w, k2);
                let hi = self.s.m.bin(c, Bin::And, hi, mask);
                let ht = self.s.m.bin(c, Bin::Shl, height, k14);
                let d2 = self.s.m.bin(c, Bin::Or, hi, ht);
                let ty = [0, 8, 9, 0, 10, 0, 12, 13][kind as usize];
                let base = k(self, 0xFAC | ty << 28);
                let last = self.s.m.bin(c, Bin::Shl, levels, k16);
                let d3 = self.s.m.bin(c, Bin::Or, base, last);
                vec![(1, d1), (2, d2), (3, d3), (4, depth)]
            }
            None => vec![(2, field(self, 0))],
        };
        for k in 0..count {
            let rel = k.wrapping_sub(u32::from(first));
            match words.iter().find(|w| w.0 == rel) {
                Some(&(_, x)) => self.write(inst, Reg::S(code + k as u8), x, None),
                None => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(OTHER));
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether a phi, or an instruction other than an interpolation's second
    /// half (`v_interp_p2*`), reads `v`.
    fn read_past_interpolation(&self, v: Value) -> bool {
        // In the last chance a join passes the value on: it is read where
        // the join's value is (a join of leftovers nothing reads is dead).
        let mut seen = vec![v];
        let mut next = 0;
        while let Some(&v) = seen.get(next) {
            next += 1;
            for (bi, b) in self.x.p.blocks.iter().enumerate() {
                for p in b.phis.iter().filter(|p| p.args.contains(&v)) {
                    if !self.leftovers {
                        return true;
                    }
                    if !seen.contains(&p.value) {
                        seen.push(p.value);
                    }
                }
                for (ii, i) in b.insts.iter().enumerate() {
                    if !i.name().starts_with("v_interp_p2")
                        && i.reads
                            .iter()
                            .any(|r| r.1 == v && !self.overwritten_by(i, r.0, (bi, ii)))
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Whether the read of register `reg` by `inst` (at `site`) is only the
    /// old value every vector write reads (see [`crate::ir`]) and the
    /// instruction writes all 32 bits in every lane that reaches it (or
    /// half of them, as a pair of halves does), so that nothing of the old
    /// value is kept. Only in the last chance.
    // shortcut: a lane whose EXEC bit is clear at the write is taken to
    // reach none (a guarded write stays a read).
    fn overwritten_by(&self, inst: &Inst, reg: Reg, site: Site) -> bool {
        if !self.leftovers || !matches!(reg, Reg::V(_)) {
            return false;
        }
        let name = inst.name();
        let whole = match &inst.fields {
            Fields::Vintrp { .. } => name.starts_with("v_interp_p1"),
            Fields::Valu(v) => {
                v.sdwa.is_none()
                    && (name == "v_interp_p1ll_f16"
                        || name.starts_with("v_cvt_f32_")
                        // shortcut: the half these keep of the old value is
                        // the zero the first half stored, taken to be
                        // written by the pair's other half (mixhi, mixlo).
                        || name.ends_with("_mixlo_f16")
                        || name.ends_with("_mixhi_f16")
                        || ["_f32", "_u32", "_i32", "_b32"]
                            .iter()
                            .any(|t| name.ends_with(t)))
            }
            Fields::Buffer { .. } | Fields::Flat { .. } => {
                name.starts_with("buffer_load_dword") || name.starts_with("global_load_dword")
            }
            _ => false,
        };
        // The register is also an explicit source (`v_add_f32 v2, v2, v3`,
        // an address register loaded over): that read is data, not only the
        // old destination. So is the accumulator a MAC or `dot*c` adds to,
        // and the old value DPP keeps in lanes its masks leave out.
        let Reg::V(n) = reg else { return false };
        let names = |o: &Operand| {
            matches!(*o, Operand::Vector { reg, dwords }
                if (u32::from(reg)..u32::from(reg) + u32::from(dwords.max(1)))
                    .contains(&u32::from(n)))
        };
        let source = match &inst.fields {
            Fields::Vintrp { src, .. } => names(src),
            Fields::Valu(v) => {
                v.src.iter().any(names)
                    || v.dpp.is_some()
                    || name.contains("mac")
                    || name.starts_with("v_dot") && name.contains("c_")
            }
            Fields::Buffer { addr, .. } => names(addr),
            Fields::Flat { addr, .. } => addr.as_ref().is_some_and(names),
            _ => false,
        };
        whole
            && !source
            && inst.writes.iter().any(|w| w.0 == reg)
            && self
                .x
                .st
                .guards
                .get(site.0)
                .and_then(|g| g.get(site.1))
                .is_some_and(|g| g.is_none())
    }

    /// A scalar buffer load takes the whole descriptor by reference. Its
    /// byte offset can also name one of those SGPRs, this time as a value.
    fn scalar_record_count_used(&self, v: Value) -> bool {
        self.x.p.blocks.iter().any(|b| {
            b.phis.iter().any(|p| p.args.contains(&v))
                || b.insts.iter().any(|i| {
                    if let Fields::Smem { soffset, .. } = i.fields
                        && i.name().starts_with("s_buffer_load_dword")
                    {
                        return matches!(soffset, Operand::Scalar { code, .. }
                            if i.read(Reg::S(code)) == Some(v));
                    }
                    i.reads.iter().any(|r| r.1 == v)
                })
        })
    }

    /// Whether an instruction or phi reads `v` other than as a resource
    /// operand of a memory instruction (which takes the descriptor whole).
    fn read_as_data(&self, v: Value) -> bool {
        self.x.p.blocks.iter().any(|b| {
            b.phis.iter().any(|p| p.args.contains(&v))
                || b.insts.iter().any(|i| {
                    !matches!(
                        i.fields,
                        Fields::Image { .. } | Fields::Buffer { .. } | Fields::Smem { .. }
                    ) && i.reads.iter().any(|r| r.1 == v)
                })
        })
    }

    /// `s_load_dword*` from the program's constant table.
    fn code_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar load from the code"),
        };
        let c = self.cur;
        let mut addr = self.s.m.i32(a.offset);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            addr = self.s.m.bin(c, Bin::Add, addr, d);
        }
        for k in 0..count {
            let off = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, addr, off);
            let x = self.code_dword(at)?;
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// The dword at a computed byte address of a constant buffer.
    fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }

    /// Correctly rounded f32 bits of `value / (2^n - 1)`, for a signed or
    /// unsigned integer already clamped to the normalized range.
    fn norm_bits(&mut self, value: V, n: u32) -> V {
        let c = self.cur;
        // All normalized fields fit in signed i32 and convert to f32 exactly.
        let f = self.s.m.cast(c, Cast::SiToFp, value, self.s.t.f32);
        let bits = self.as_int(f);
        if n == 1 {
            return bits;
        }
        // Start at |value| / 2^n. In units of its last place, its magnitude
        // is the 24-bit significand M. Multiplying by 2^n / (2^n - 1)
        // adds M / (2^n - 1) ULPs. The denominator is odd, so rounding the
        // correction has no ties. Only a full-scale input reaches the next
        // exponent, and adding the correction to the bits carries into it.
        let mask = self.s.m.i32(0x7f_ffff);
        let implicit = self.s.m.i32(1 << 23);
        let fraction = self.s.m.bin(c, Bin::And, bits, mask);
        let significand = self.s.m.bin(c, Bin::Or, fraction, implicit);
        let max = (1 << n) - 1;
        let half = self.s.m.i32(max / 2);
        let divisor = self.s.m.i32(max);
        let biased = self.s.m.bin(c, Bin::Add, significand, half);
        let correction = self.s.m.bin(c, Bin::UDiv, biased, divisor);
        let scale = self.s.m.i32(n << 23);
        let base = self.s.m.bin(c, Bin::Sub, bits, scale);
        let rounded = self.s.m.bin(c, Bin::Add, base, correction);
        let zero = self.s.m.i32(0);
        let is_zero = self.s.m.cmp(c, pred::EQ, value, zero);
        self.s.m.select(c, is_zero, zero, rounded)
    }

    fn table_buffer_stride(
        &self,
        src: &Source,
        used: Option<&UsedResource>,
        uav: bool,
    ) -> Option<u32> {
        let listed = used.filter(|u| {
            u.root == RootType::Table && matches!(u.kind, ResourceType::Srv | ResourceType::Uav)
        });
        match listed.map(|u| u.stride) {
            None | Some(0) => self.bound_structured_stride(src, uav, false),
            // Raw table buffers can declare a stride of four. Only a bound
            // structured view establishes a four-byte stride for flat access.
            Some(stride) => Some(stride).filter(|s| *s > 4 && Self::valid_buffer_stride(s)),
        }
    }

    /// Whether a descriptor is read by SMEM through a buffer handle.
    fn scalar_buffer_view(
        &self,
        src: &Source,
        used: Option<&UsedResource>,
    ) -> Option<(Option<u32>, bool)> {
        match used.map(|u| u.kind) {
            Some(ResourceType::Srv) => Some((self.scalar_srv_stride(src, used), false)),
            Some(ResourceType::Uav) => {
                Some((Some(self.table_buffer_stride(src, used, true)?), true))
            }
            None if self.buffer_loads_only(src) => {
                Some((self.table_buffer_stride(src, used, false), false))
            }
            _ => None,
        }
    }

    /// [`Self::scalar_buffer_view`], else a descriptor of no listed kind
    /// fetched from the resource heap that nothing writes
    /// ([`Self::heap_view_read_only`]); and whether it is the latter.
    fn scalar_view(
        &self,
        src: &Source,
        used: Option<&UsedResource>,
    ) -> Option<((Option<u32>, bool), bool)> {
        if let Some(view) = self.scalar_buffer_view(src, used) {
            return Some((view, false));
        }
        // A constant buffer the console's resource list places in the heap
        // stays one.
        let listed_cbv = self
            .x
            .native
            .used
            .iter()
            .any(|u| u.root == RootType::ResourceHeap && u.kind == ResourceType::Cbv);
        (used.is_none() && !listed_cbv && self.heap_view_read_only(src))
            .then(|| ((self.table_buffer_stride(src, used, false), false), true))
    }

    fn buffer_props(stride: Option<u32>, uav: bool) -> (u32, u32) {
        let flag = if uav { 0x1000 } else { 0 };
        stride.map_or((11 | flag, 0), |stride| {
            // Match DXC's scalar dword and wider dword-record annotations.
            // Queries and accesses share the annotation for the same view.
            let alignment = if stride == 4 { 0 } else { 2 << 8 };
            (12 | alignment | flag, stride)
        })
    }

    fn add_bytes(&mut self, addr: V, bytes: u32) -> V {
        if bytes == 0 {
            addr
        } else if let Some(addr) = self.s.m.int_value(addr) {
            self.s.m.i32((addr as u32).wrapping_add(bytes))
        } else {
            let bytes = self.s.m.i32(bytes);
            self.s.m.bin(self.cur, Bin::Add, addr, bytes)
        }
    }

    fn structured_coords(&mut self, addr: V, stride: u32) -> (V, V) {
        self.structured_index_and_offset(addr, stride)
    }

    fn buffer_dword(&mut self, h: V, index: V, off: V) -> V {
        let c = self.cur;
        let mask = self.s.m.i8(1);
        let align = self.s.m.i32(4);
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[h, index, off, mask, align],
        );
        let x = self.s.m.extract(c, r, 0);
        self.as_int(x)
    }

    fn structured_dword(&mut self, h: V, addr: V, stride: u32) -> V {
        let (index, off) = self.structured_coords(addr, stride);
        let c = self.cur;
        let i32t = self.s.t.i32;
        let ret = self.s.m.named_struct("dx.types.ResRet.i32", vec![i32t; 5]);
        let mask = self.s.m.i8(1);
        let align = self.s.m.i32(4);
        let r = self.s.op(
            c,
            "rawBufferLoad.i32",
            Attr::ReadOnly,
            ret,
            139,
            &[h, index, off, mask, align],
        );
        self.s.m.extract(c, r, 0)
    }

    /// The aligned word address and the number of bits preceding the access.
    fn buffer_word(&mut self, addr: V) -> (V, V) {
        if let Some(addr) = self.s.m.int_value(addr) {
            let addr = addr as u32;
            return (self.s.m.i32(addr & !3), self.s.m.i32((addr & 3) * 8));
        }
        let c = self.cur;
        let aligned = self.s.m.i32(!3);
        let three = self.s.m.i32(3);
        let word = self.s.m.bin(c, Bin::And, addr, aligned);
        let byte = self.s.m.bin(c, Bin::And, addr, three);
        (word, self.s.m.bin(c, Bin::Shl, byte, three))
    }

    /// Aligned, single-dword reads stay inside their structured element. If
    /// the value straddles a dword, split the second address independently.
    fn structured_load(&mut self, h: V, addr: V, stride: u32, bytes: u32) -> V {
        let c = self.cur;
        let (first, shift) = self.buffer_word(addr);
        let lo = self.structured_dword(h, first, stride);
        let lo = if self.s.m.int_value(shift) == Some(0) {
            lo
        } else {
            self.s.m.bin(c, Bin::LShr, lo, shift)
        };
        if bytes == 1 {
            return lo;
        }
        let end = self.add_bytes(addr, bytes - 1);
        let (last, _) = self.buffer_word(end);
        if first == last {
            return lo;
        }
        let hi = self.structured_dword(h, last, stride);
        let zero = self.s.m.i32(0);
        let mask = self.s.m.i32(31);
        let reverse = self.s.m.bin(c, Bin::Sub, zero, shift);
        let reverse = self.s.m.bin(c, Bin::And, reverse, mask);
        let hi = self.s.m.bin(c, Bin::Shl, hi, reverse);
        let joined = self.s.m.bin(c, Bin::Or, lo, hi);
        let crosses = self.s.m.cmp(c, pred::NE, first, last);
        self.s.m.select(c, crosses, joined, lo)
    }

    fn structured_store_word(&mut self, h: V, addr: V, stride: u32, val: V) {
        let (index, off) = self.structured_coords(addr, stride);
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(1);
        let align = self.s.m.i32(4);
        self.s.op(
            self.cur,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            self.s.t.void,
            140,
            &[h, index, off, val, iu, iu, iu, mask, align],
        );
    }

    fn structured_store_masked(&mut self, h: V, addr: V, stride: u32, val: V, mask: V) {
        // Retry with the observed value so another lane's writes to the
        // other bytes survive. Keep the compare-exchange and its backedge
        // in a separate latch from the loop-carried value.
        let start = self.cur;
        let (index, off) = self.structured_coords(addr, stride);
        let retry = self.s.m.new_block();
        let latch = self.s.m.new_block();
        let done = self.s.m.new_block();
        let iu = self.s.m.undef(self.s.t.i32);
        let zero = self.s.m.i32(0);
        let ones = self.s.m.i32(u32::MAX);
        let keep = self.s.m.bin(start, Bin::Xor, mask, ones);
        self.s.m.terminate(start, Term::Br(retry));
        let old = self.s.m.phi(retry, self.s.t.i32);
        let rest = self.s.m.bin(retry, Bin::And, old, keep);
        let new = self.s.m.bin(retry, Bin::Or, rest, val);
        self.s.m.terminate(retry, Term::Br(latch));
        let seen = self.s.op(
            latch,
            "atomicCompareExchange.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            79,
            &[h, index, off, iu, old, new],
        );
        self.s.m.set_phi(old, vec![(zero, start), (seen, latch)]);
        let success = self.s.m.cmp(latch, pred::EQ, old, seen);
        self.s
            .m
            .terminate(latch, Term::CondBr(success, done, retry));
        self.cur = done;
    }

    fn structured_store(&mut self, h: V, addr: V, stride: u32, val: V, bytes: u32) {
        let (first, shift) = self.buffer_word(addr);
        if bytes == 4 && self.s.m.int_value(shift) == Some(0) {
            self.structured_store_word(h, first, stride, val);
            return;
        }
        // The aligned case is common even when the address is computed.
        let full = if bytes == 4 && self.s.m.int_value(shift).is_none() {
            let zero = self.s.m.i32(0);
            let aligned = self.s.m.cmp(self.cur, pred::EQ, shift, zero);
            let (full, partial) = (self.s.m.new_block(), self.s.m.new_block());
            self.s
                .m
                .terminate(self.cur, Term::CondBr(aligned, full, partial));
            self.cur = full;
            self.structured_store_word(h, first, stride, val);
            self.cur = partial;
            Some(full)
        } else {
            None
        };
        let c = self.cur;
        let mask = self.s.m.i32(u32::MAX >> (32 - 8 * bytes));
        let val = self.s.m.bin(c, Bin::And, val, mask);
        let lo = self.s.m.bin(c, Bin::Shl, val, shift);
        let lo_mask = self.s.m.bin(c, Bin::Shl, mask, shift);
        let end = self.add_bytes(addr, bytes - 1);
        let (last, _) = self.buffer_word(end);
        self.structured_store_masked(h, first, stride, lo, lo_mask);
        if bytes > 1 && first != last {
            let head = self.cur;
            let crosses = self.s.m.cmp(head, pred::NE, first, last);
            let c = self.s.m.new_block();
            self.cur = c;
            let zero = self.s.m.i32(0);
            let bits = self.s.m.i32(31);
            let reverse = self.s.m.bin(c, Bin::Sub, zero, shift);
            let reverse = self.s.m.bin(c, Bin::And, reverse, bits);
            let hi = self.s.m.bin(c, Bin::LShr, val, reverse);
            let hi_mask = self.s.m.bin(c, Bin::LShr, mask, reverse);
            self.structured_store_masked(h, last, stride, hi, hi_mask);
            let join = self.s.m.new_block();
            self.s.m.terminate(head, Term::CondBr(crosses, c, join));
            self.s.m.terminate(self.cur, Term::Br(join));
            self.cur = join;
        }
        if let Some(full) = full {
            let join = self.s.m.new_block();
            self.s.m.terminate(full, Term::Br(join));
            self.s.m.terminate(self.cur, Term::Br(join));
            self.cur = join;
        }
    }

    /// `channels` channels of the element at byte `off` of buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, the fourth too).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds). A dynamic raw
    // address for elements of 4 bytes or more is taken as dword-aligned.
    fn format_load(
        &mut self,
        h: V,
        off: V,
        format: u8,
        channels: u32,
        stride: Option<u32>,
    ) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            // A ResRet holds four data words; the fifth value is the status.
            debug_assert!((1..=4).contains(&n), "a raw load of {n} words");
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, at, u, mask, align],
            );
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if let Some(stride) = stride {
            (0..total.div_ceil(32))
                .map(|k| {
                    let at = self.add_bytes(off, 4 * k);
                    self.structured_load(h, at, stride, (total - 32 * k).min(32) / 8)
                })
                .collect()
        } else if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            let low = self.s.m.bin(c, Bin::LShr, w, sh);
            let word = if total == 16 && self.s.m.int_value(off).is_none_or(|v| v % 4 == 3) {
                // A halfword at byte 3 reads `(lo >> 24) | (hi << 8)` from
                // the dword pair around it; shifting only the first dword
                // would drop the high byte (VIC-BT-019 / #440). Other
                // alignments stay within the first dword, including when it
                // is the last word exposed by a root buffer.
                #[cfg(test)]
                if self.s.m.int_value(off).is_some_and(|v| v % 4 == 3) {
                    self.s.unaligned_typed_loads += 1;
                }
                let crosses = self.s.m.cmp(c, pred::EQ, byte, three);
                let four = self.s.m.i32(4);
                let next = self.s.m.bin(c, Bin::Add, at, four);
                let addr = self.s.m.select(c, crosses, next, at);
                let high = load(self, addr, 1)[0];
                let eight = self.s.m.i32(8);
                let high = self.s.m.bin(c, Bin::Shl, high, eight);
                let combined = self.s.m.bin(c, Bin::Or, low, high);
                self.s.m.select(c, crosses, combined, low)
            } else {
                low
            };
            vec![word]
        } else if let Some(at) = self.s.m.int_value(off) {
            if at % 4 == 0 {
                load(self, off, total / 32)
            } else {
                // A value of several dwords at a byte offset inside one: the
                // console reads the bytes the address names, so each word is
                // taken from the two dword-aligned words around it. Reading
                // at the byte address itself would promise the host an
                // alignment the console does not have, and a host that
                // believes it can hand back zeros (a position fetched at
                // an odd byte offset collapsed every vertex to the
                // origin).
                #[cfg(test)]
                {
                    self.s.unaligned_typed_loads += 1;
                }
                let n = (total / 32) as usize;
                let shift = (at % 4) as u32 * 8;
                let aligned = self.s.m.i32((at & !3) as u32);
                let sh = self.s.m.i32(shift);
                let up = self.s.m.i32(32 - shift);
                // n + 1 aligned words, at most four to a load.
                let base = (at & !3) as u32;
                let mut raw = Vec::with_capacity(n + 1);
                for first in (0..=n).step_by(4) {
                    let count = (n + 1 - first).min(4) as u32;
                    let from = if first == 0 {
                        aligned
                    } else {
                        self.s.m.i32(base + 4 * first as u32)
                    };
                    raw.extend(load(self, from, count));
                }
                (0..n)
                    .map(|k| {
                        let lo = self.s.m.bin(c, Bin::LShr, raw[k], sh);
                        let hi = self.s.m.bin(c, Bin::Shl, raw[k + 1], up);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    })
                    .collect()
            }
        } else {
            load(self, off, total / 32)
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            let x = match numeric {
                Numeric::Unorm => self.norm_bits(field, w),
                Numeric::Snorm => {
                    let s = signed(self);
                    let min = self.s.m.i32(1u32.wrapping_sub(1 << (w - 1)));
                    let below = self.s.m.cmp(c, pred::SLT, s, min);
                    let clamped = self.s.m.select(c, below, min, s);
                    self.norm_bits(clamped, w - 1)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let h = match half_shift(w) {
                        0 => field,
                        k => {
                            let sh = self.s.m.i32(k);
                            self.s.m.bin(c, Bin::Shl, field, sh)
                        }
                    };
                    let f = self.f16_widen_keep_snan(h);
                    float(self, f)
                }
            };
            out.push(x);
        }
        // Absent channels are 0; DST_SEL supplies 1 only when set to SEL_1
        // (VIC-BT-019 / #440).
        for _ in out.len() as u32..channels {
            out.push(self.s.m.i32(0));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    /// Widen an f16 (or a 10/11-bit float shifted into half form) to f32
    /// without quieting sNaNs: NaNs become `sign | 0x7F800000 |
    /// (payload << 13)`; other values use `legacyF16ToF32`.
    fn f16_widen_keep_snan(&mut self, h: V) -> V {
        let c = self.cur;
        let f32t = self.s.t.f32;
        let quieted = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        let quieted = self.as_int(quieted);
        let (mask_abs, nan_min, sign_bit, k16, mant, k13, exp_ones) = (
            self.s.m.i32(0x7FFF),
            self.s.m.i32(0x7C00),
            self.s.m.i32(0x8000),
            self.s.m.i32(16),
            self.s.m.i32(0x3FF),
            self.s.m.i32(13),
            self.s.m.i32(0x7F80_0000),
        );
        let abs = self.s.m.bin(c, Bin::And, h, mask_abs);
        let is_nan = self.s.m.cmp(c, pred::IUGT, abs, nan_min);
        let sign = self.s.m.bin(c, Bin::And, h, sign_bit);
        let sign = self.s.m.bin(c, Bin::Shl, sign, k16);
        let payload = self.s.m.bin(c, Bin::And, h, mant);
        let payload = self.s.m.bin(c, Bin::Shl, payload, k13);
        let kept = self.s.m.bin(c, Bin::Or, sign, exp_ones);
        let kept = self.s.m.bin(c, Bin::Or, kept, payload);
        let bits = self.s.m.select(c, is_nan, kept, quieted);
        self.as_float(bits)
    }

    /// Channels of buffer format `format`, as [`Self::format_load`] gives
    /// them, as d16 data: 16 bits each, two per dword, low half first.
    /// Integer formats saturate to 16 bits (UINT to 0xFFFF, SINT to
    /// -32768..32767), as RDNA2 does for a 32-bit channel; the others become
    /// f16 (nearest even). An odd last channel is a single half: the caller
    /// merges it with [`Self::low_half`].
    fn d16_channels(&mut self, format: u8, channels: Vec<V>) -> Result<Vec<V>, Why> {
        let (_, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let mut halves = Vec::new();
        for x in channels {
            halves.push(if matches!(numeric, Numeric::Uint) {
                let m = self.s.m.i32(0xFFFF);
                let lt = self.s.m.cmp(c, pred::IULT, x, m);
                self.s.m.select(c, lt, x, m)
            } else if matches!(numeric, Numeric::Sint) {
                let (lo, hi, m) = (
                    self.s.m.i32((-32768i32) as u32),
                    self.s.m.i32(32767),
                    self.s.m.i32(0xFFFF),
                );
                let below = self.s.m.cmp(c, pred::SLT, x, lo);
                let x = self.s.m.select(c, below, lo, x);
                let above = self.s.m.cmp(c, pred::SGT, x, hi);
                let x = self.s.m.select(c, above, hi, x);
                self.s.m.bin(c, Bin::And, x, m)
            } else {
                // Float / normalized: round to nearest even f16.
                let f = self.as_float(x);
                self.f16_nearest(f)
            });
        }
        let mut out = Vec::new();
        for pair in halves.chunks(2) {
            out.push(match *pair {
                [lo, hi] => {
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                }
                [lo] => lo,
                _ => unreachable!("chunks of one or two"),
            });
        }
        Ok(out)
    }

    /// The byte offset a buffer descriptor built around a root descriptor adds
    /// (constant and computed parts); none for other descriptors.
    fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<V>), Why> {
        match src {
            Source::Built { address, .. } => {
                let x = match address.dynamic {
                    Some(v) => Some(self.i32_of(v)?),
                    None => None,
                };
                Ok((address.offset, x))
            }
            _ => Ok((0, None)),
        }
    }

    /// The stride of a buffer descriptor the code builds around a root
    /// descriptor's address (dword 1 bits 16-29), unless swizzled.
    fn root_view_stride(&self, src: &Source) -> Option<u32> {
        let Source::Built { address, bits, .. } = src else {
            return None;
        };
        let root = self.x.native.layout.get(address.param)?.root;
        (root == RootType::Descriptor && bits >> 31 == 0).then_some(bits >> 16 & 0x3FFF)
    }

    /// Whether a vector buffer access other than a load goes through a
    /// descriptor built around root parameter `param` (which makes that root
    /// descriptor a UAV).
    fn root_view_written(&self, param: usize) -> bool {
        self.descriptors.iter().any(|(&((b, i), sampler), &src)| {
            !sampler
                && matches!(src, Source::Built { address, .. } if address.param == param)
                && self.x.p.blocks[b].insts.get(i).is_some_and(|inst| {
                    matches!(inst.fields, Fields::Buffer { .. })
                        && !inst.name().starts_with("buffer_load")
                        && !inst.name().starts_with("tbuffer_load")
                        && !inst.name().starts_with("buffer_gl")
                })
        })
    }

    /// Whether vector buffer loads (`buffer_load_*`) read descriptor `src`
    /// and no vector buffer access writes through it (which makes it a UAV).
    /// Whether `src` is a descriptor the code fetches from the resource
    /// heap (`ResourceDescriptorHeap`) and no vector memory instruction
    /// writes through or reads as an image.
    fn heap_view_read_only(&self, src: &Source) -> bool {
        let Source::Loaded { load, .. } = src else {
            return false;
        };
        let Some(address) = self.scalar.get(load) else {
            return false;
        };
        if self.x.native.layout.get(address.param).map(|p| p.root) != Some(RootType::ResourceHeap) {
            return false;
        }
        for (&((b, i), sampler), &other) in &self.descriptors {
            if sampler || other != src {
                continue;
            }
            let Some(inst) = self.x.p.blocks[b].insts.get(i) else {
                continue;
            };
            match inst.fields {
                Fields::Buffer { .. } => {
                    let name = inst.name();
                    if !(name.starts_with("buffer_load") || name.starts_with("tbuffer_load")) {
                        return false;
                    }
                }
                Fields::Smem { .. } => {}
                _ => return false,
            }
        }
        true
    }

    fn buffer_loads_only(&self, src: &Source) -> bool {
        let mut loads = false;
        for (&((b, i), sampler), &other) in &self.descriptors {
            if sampler || other != src {
                continue;
            }
            let Some(inst) = self.x.p.blocks[b].insts.get(i) else {
                continue;
            };
            if !matches!(inst.fields, Fields::Buffer { .. }) {
                continue;
            }
            let name = inst.name();
            if name.starts_with("buffer_load") || name.starts_with("tbuffer_load") {
                loads = true;
            } else {
                return false;
            }
        }
        loads
    }

    /// Whether a descriptor comes from the resource heap itself (root type
    /// 5, not a table) on some path.
    fn is_heap_direct(&self, src: &Source) -> bool {
        src.leaves().iter().any(|l| match l {
            Source::Loaded { load, .. } | Source::Fetched { load, .. } => self
                .scalar
                .get(load)
                .and_then(|a| self.x.native.layout.get(a.param))
                .is_some_and(|p| p.root == RootType::ResourceHeap),
            _ => false,
        })
    }

    /// The used-resource entry a descriptor loaded from a table falls in.
    fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
        if let Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } =
            src
        {
            return self.used_kind(args.first()?);
        }
        if let Source::LanePhi { args, .. } = src {
            return args.iter().flatten().find_map(|a| self.used_kind(a));
        }
        if let Source::Built { address, .. } = src {
            let rp = self.x.native.layout.get(address.param)?;
            return self
                .x
                .native
                .used
                .iter()
                .find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data);
        }
        let (Source::Loaded { load, first } | Source::Fetched { load, first }) = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table {
            return None;
        }
        let dword = (a.offset / 4) + u32::from(*first);
        self.x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)
    }

    /// How an indexed (`idxen`) access addresses the buffer. A stride the
    /// console metadata declares is used as given. Failing that, the bound
    /// view decides: a stride of exactly zero is a flat byte buffer, which
    /// is a raw access (the index contributes `index * 0`), not a structured
    /// buffer of unknown stride.
    fn indexed_stride(&self, src: &Source, uav: bool) -> Result<IndexedStride, Why> {
        if let Some(stride) = self
            .used_kind(src)
            .map(|u| u.stride)
            .filter(Self::valid_buffer_stride)
        {
            return Ok(IndexedStride::Structured(stride));
        }
        match self.bound_view_stride(src, uav, true) {
            Some(0) => Ok(IndexedStride::Raw),
            Some(stride) => Ok(IndexedStride::Structured(stride)),
            None => Err(UNKNOWN_STRUCTURED_STRIDE),
        }
    }

    /// The structured view of an indexed access: its index and stride, or
    /// `None` when the bound view is raw and the index drops out.
    fn structured_index(&self, src: &Source, uav: bool, index: V) -> Result<Option<(V, u32)>, Why> {
        Ok(match self.indexed_stride(src, uav)? {
            IndexedStride::Raw => None,
            IndexedStride::Structured(stride) => Some((index, stride)),
        })
    }

    /// The stride of an indexed access, `None` for a raw (zero-stride) view.
    fn indexed_buffer_stride(&self, src: &Source, uav: bool) -> Result<Option<u32>, Why> {
        Ok(match self.indexed_stride(src, uav)? {
            IndexedStride::Raw => None,
            IndexedStride::Structured(stride) => Some(stride),
        })
    }

    fn valid_buffer_stride(stride: &u32) -> bool {
        (4..=2048).contains(stride) && stride.is_multiple_of(4)
    }

    fn bound_structured_stride(&self, src: &Source, uav: bool, required: bool) -> Option<u32> {
        self.bound_view_stride(src, uav, required)
            .filter(|&stride| stride != 0)
    }

    /// The stride every possible view of `src` supplies: 0 for raw, else a
    /// valid structured stride.
    fn bound_view_stride(&self, src: &Source, uav: bool, required: bool) -> Option<u32> {
        let view_stride = self.view_stride?;
        let stride = |source: &Source| {
            // shortcut: a descriptor selected by a dynamic index has no
            // statically identifiable view to specialize this PSO with.
            let Source::Loaded { load, first } = source else {
                return None;
            };
            let address = self.scalar.get(load)?;
            let param = self.x.native.layout.get(address.param)?;
            let offset = address.offset.checked_add(4 * u32::from(*first))?;
            if param.root != RootType::Table || address.dynamic.is_some() || offset % 32 != 0 {
                return None;
            }
            view_stride(
                StructuredBinding {
                    slot: param.slot,
                    offset_dwords: offset / 4,
                    uav,
                },
                required,
            )
            .filter(|stride| *stride == 0 || Self::valid_buffer_stride(stride))
        };
        // A selected handle has one DXIL annotation. Every possible view must
        // supply the same stride for that instruction.
        let mut strides = src.leaves().into_iter().map(stride);
        let first = strides.next().flatten()?;
        strides.all(|other| other == Some(first)).then_some(first)
    }

    /// A fixed root-constant dword used as a view index: heap base plus
    /// that value shifted left by five (32 bytes per descriptor).
    fn buffer_selector(&self, src: &Source) -> Option<(u8, u32)> {
        let Source::Loaded { load, first: 0 } = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        if self.x.native.layout.get(a.param)?.root != RootType::ResourceHeap || a.offset != 0 {
            return None;
        }
        let value = a.dynamic?;
        let block = &self.x.p.blocks[*self.def_block.get(&value)?];
        let shift = block
            .insts
            .iter()
            .find(|i| i.writes.first().is_some_and(|w| w.1 == value))?;
        let Fields::Sop { src, .. } = &shift.fields else {
            return None;
        };
        let [Operand::Scalar { code, .. }, Operand::Const(133)] = src.as_slice() else {
            return None;
        };
        if shift.name() != "s_lshl_b32" {
            return None;
        }
        let &(param, dword) = self.params.get(&shift.read(Reg::S(*code))?)?;
        let root = self.x.native.layout[param];
        (root.root == RootType::Constants).then_some((root.slot, dword))
    }

    fn selected_buffer(&mut self, slot: u8, dword: u32) -> Result<[V; 3], Why> {
        let existing = self
            .user_reads
            .values()
            .find(|u| u.indexed_buffer && u.slot == slot && u.dword == dword);
        let index = if let Some(u) = existing {
            u.index
        } else {
            // Inputs that are used later must retain their original positions.
            let inputs = self
                .params
                .values()
                .filter_map(|&(p, word)| {
                    let rp = self.x.native.layout[p];
                    matches!(rp.root, RootType::Constants | RootType::Table)
                        .then_some(u32::from(rp.user_data) + word + 1)
                })
                .max()
                .unwrap_or(0);
            let index = inputs.max(self.s.user_data);
            if index + 3 > 64 {
                return Err("an indexed buffer beyond the root constant budget");
            }
            self.user_reads.insert(
                index,
                UserData {
                    indexed_buffer: true,
                    index,
                    slot,
                    root: RootType::Constants,
                    dword,
                },
            );
            index
        };
        Ok([
            self.user_word(index),
            self.user_word(index + 1),
            self.user_word(index + 2),
        ])
    }

    /// Records the table entries a structured SRV handle of `stride`-byte
    /// elements reads, for the host to alias a raw view bound there.
    /// shortcut: an entry chosen by a dynamic index is not recorded (no
    /// statically known entry), as for [`Lower::bound_structured_stride`].
    fn note_structured(&mut self, src: &Source, stride: u32) {
        for leaf in src.leaves() {
            let Source::Loaded { load, first } = *leaf else {
                continue;
            };
            let Some(address) = self.scalar.get(&load) else {
                continue;
            };
            let Some(param) = self.x.native.layout.get(address.param) else {
                continue;
            };
            let Some(offset) = address.offset.checked_add(4 * u32::from(first)) else {
                continue;
            };
            if param.root != RootType::Table || address.dynamic.is_some() || offset % 32 != 0 {
                continue;
            }
            let srv = super::StructuredSrv {
                slot: param.slot,
                offset_dwords: offset / 4,
                stride,
            };
            if !self.structured_srvs.contains(&srv) {
                self.structured_srvs.push(srv);
            }
        }
    }

    /// Records the statically known table entries `src` reads as `props`
    /// (for the host's descriptor kind check, [`super::HeapRead`]).
    fn note_heap_read(&mut self, src: &Source, sampler: bool, props: (u32, u32)) {
        for leaf in src.leaves() {
            let Source::Loaded { load, first } = *leaf else {
                continue;
            };
            let Some(address) = self.scalar.get(&load) else {
                continue;
            };
            let Some(param) = self.x.native.layout.get(address.param) else {
                continue;
            };
            let Some(offset) = address.offset.checked_add(4 * u32::from(first)) else {
                continue;
            };
            let step = if sampler { 16 } else { 32 };
            let absolute = match param.root {
                RootType::Table => false,
                RootType::ResourceHeap if !sampler => true,
                RootType::SamplerHeap if sampler => true,
                _ => continue,
            };
            if offset % step != 0 {
                continue;
            }
            let read = super::HeapRead {
                slot: param.slot,
                offset_dwords: offset / 4,
                sampler,
                props,
                dynamic: address.dynamic.is_some(),
                absolute,
                index_from: address
                    .dynamic
                    .and_then(|d| self.ud_index(d, if sampler { 4 } else { 5 })),
                index_load: address
                    .dynamic
                    .and_then(|d| self.loaded_index(d, if sampler { 4 } else { 5 })),
            };
            if !self.heap_reads.contains(&read) {
                if let Some(d) = address.dynamic
                    && std::env::var_os("VICTORY_SHADER_TRACE_DYNAMIC").is_some()
                {
                    let text = format!(
                        "props {:#x},{:#x} offset {offset}: {}",
                        props.0,
                        props.1,
                        self.value_chain(d, 6)
                    );
                    super::DYNAMIC_CHAINS.with(|c| c.borrow_mut().push(text));
                }
                self.heap_reads.push(read);
            }
        }
    }

    /// The instruction that writes `v`.
    fn def_of(&self, v: Value) -> Option<&'a Inst> {
        let p = self.x.p;
        let defs = self.spill_defs.get_or_init(|| {
            let mut defs = HashMap::new();
            for (b, block) in p.blocks.iter().enumerate() {
                for (i, inst) in block.insts.iter().enumerate() {
                    for &(_, w) in &inst.writes {
                        defs.insert(w, (b, i));
                    }
                }
            }
            defs
        });
        defs.get(&v).map(|&(b, i)| &p.blocks[b].insts[i])
    }

    /// The root constant (root parameter slot, dword) `v` is, shifted left
    /// by `shift`: a heap index the title passes in its root constants.
    fn ud_index(&self, v: Value, shift: u32) -> Option<(u8, u32)> {
        let inst = self.def_of(v)?;
        if inst.name() != "s_lshl_b32" {
            return None;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        let amount = match *src.get(1)? {
            Operand::Const(c) => crate::resource::inline(c)?,
            Operand::Literal(x) => x,
            _ => return None,
        };
        let Operand::Scalar { code, .. } = *src.first()? else {
            return None;
        };
        if amount != shift {
            return None;
        }
        let mut x = inst.read(Reg::S(code))?;
        for _ in 0..4 {
            if let Some(&(param, dword)) = self.params.get(&x) {
                let rp = self.x.native.layout.get(param)?;
                return (rp.root == RootType::Constants).then_some((rp.slot, dword));
            }
            let def = self.def_of(x)?;
            if def.name() != "s_mov_b32" {
                return None;
            }
            x = def.reads.first()?.1;
        }
        None
    }

    /// A heap index the code reads from a buffer: `v` is
    /// `(s_buffer_load(heap[root constant]) & mask) << shift`.
    fn loaded_index(&self, v: Value, shift: u32) -> Option<super::IndexLoad> {
        let operand = |inst: &Inst, k: usize| -> Option<Result<u32, Value>> {
            let Fields::Sop { src, .. } = &inst.fields else {
                return None;
            };
            Some(match *src.get(k)? {
                Operand::Const(c) => Ok(crate::resource::inline(c)?),
                Operand::Literal(x) => Ok(x),
                Operand::Scalar { code, .. } => Err(inst.read(Reg::S(code))?),
                _ => return None,
            })
        };
        let shl = self.def_of(v)?;
        if shl.name() != "s_lshl_b32" || operand(shl, 1)? != Ok(shift) {
            return None;
        }
        let Err(x) = operand(shl, 0)? else {
            return None;
        };
        let and = self.def_of(x)?;
        if and.name() != "s_and_b32" {
            return None;
        }
        let (mask, y) = match (operand(and, 0)?, operand(and, 1)?) {
            (Ok(m), Err(y)) | (Err(y), Ok(m)) => (m, y),
            _ => return None,
        };
        let load = self.def_of(y)?;
        if !load.name().starts_with("s_buffer_load_dword") {
            return None;
        }
        let word = load.writes.iter().position(|&(_, w)| w == y)? as u32;
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &load.fields
        else {
            return None;
        };
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) || *offset < 0 {
            return None;
        }
        let Operand::Scalar { code, .. } = *base else {
            return None;
        };
        // The buffer descriptor: a heap entry at a root constant index.
        let desc = self.def_of(load.read(Reg::S(code))?)?;
        if !desc.name().starts_with("s_load_dwordx") {
            return None;
        }
        let Fields::Smem {
            base: dbase,
            offset: doffset,
            soffset: dsoffset,
            ..
        } = &desc.fields
        else {
            return None;
        };
        let Operand::Scalar { code: dcode, .. } = *dbase else {
            return None;
        };
        let Operand::Scalar { code: scode, .. } = *dsoffset else {
            return None;
        };
        let mut b = desc.read(Reg::S(dcode))?;
        for _ in 0..4 {
            if self.params.contains_key(&b) {
                break;
            }
            let d = self.def_of(b)?;
            if d.name() != "s_mov_b32" {
                return None;
            }
            b = d.reads.first()?.1;
        }
        let &(param, _) = self.params.get(&b)?;
        if self.x.native.layout.get(param)?.root != RootType::ResourceHeap || *doffset != 0 {
            return None;
        }
        let (slot, dword) = self.ud_index(desc.read(Reg::S(scode))?, 5)?;
        Some(super::IndexLoad {
            slot,
            dword,
            byte: *offset as u32 + 4 * word,
            mask,
        })
    }

    /// How `v` is computed, `depth` definitions deep (diagnostics).
    fn value_chain(&self, v: Value, depth: u32) -> String {
        if let Some(&(param, dword)) = self.params.get(&v) {
            let rp = self.x.native.layout[param];
            return format!("ud(slot {} {:?} dword {dword})", rp.slot, rp.root);
        }
        if depth == 0 {
            return format!("v{}", v.0);
        }
        let Some(inst) = self.def_of(v) else {
            return format!("v{}(phi or input)", v.0);
        };
        let args: Vec<String> = inst
            .reads
            .iter()
            .filter(|(r, _)| !matches!(r, Reg::S(c) if *c == EXEC || *c == EXEC + 1))
            .take(4)
            .map(|&(_, a)| self.value_chain(a, depth - 1))
            .collect();
        format!("{}({})", inst.name(), args.join(", "))
    }

    /// The handle of the immediate sampler with `words`.
    fn static_sampler(&mut self, words: &[u32], props: (u32, u32)) -> Result<V, Why> {
        let words: [u32; 4] = words
            .get(..4)
            .and_then(|w| w.try_into().ok())
            .ok_or("a sampler constant")?;
        // Comparison and non-comparison samples need distinct static
        // sampler slots (D3D12 SamplerComparisonState vs SamplerState):
        // annotateHandle props must match dx.resources sampler kind.
        let cmp = props.0 & 0x8000 != 0;
        let n = match self
            .samplers
            .iter()
            .enumerate()
            .find(|(n, w)| **w == words && (self.s.cmp_samplers >> *n & 1 != 0) == cmp)
        {
            Some((n, _)) => n,
            None => {
                let n = self.samplers.len();
                if n >= 64 {
                    return Err("too many immediate samplers");
                }
                self.samplers.push(words);
                if cmp {
                    self.s.cmp_samplers |= 1 << n;
                }
                n
            }
        };
        Ok(self.s.sampler_handle(self.cur, n as u32))
    }

    /// Records the heap entries `src` reads for the split harness
    /// ([`super::probe`]).
    fn probe_handle(&self, src: &Source, sampler: bool, props: (u32, u32)) {
        for leaf in src.leaves() {
            match leaf {
                Source::Loaded { load, first } => match self.scalar.get(load) {
                    Some(a) => super::probe::note(super::probe::HeapUse {
                        param: a.param,
                        offset: a.offset.wrapping_add(4 * u32::from(*first)),
                        dynamic: a.dynamic.is_some(),
                        sampler,
                        props,
                    }),
                    None => super::probe::note_untracked(),
                },
                Source::Constant(_) if sampler => {}
                Source::Built { .. } => {}
                _ => super::probe::note_untracked(),
            }
        }
    }

    /// A handle for the descriptor `src` read at `site`.
    fn handle(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        // A structured SRV (kind 12, stride in the second word).
        if !sampler && structured_srv(props) {
            self.note_structured(src, props.1);
        }
        self.note_heap_read(src, sampler, props);
        if super::probe::recording() {
            self.probe_handle(src, sampler, props);
        }
        match src {
            Source::Loaded { load, first } => {
                let key = (self.cur, *load, *first, sampler, props);
                if let Some(&h) = self.handles.get(&key) {
                    return Ok(h);
                }
                let index = self.heap_index(*load, *first, sampler)?;
                // A waterfall loop's load at the first lane's offset reads
                // each lane's own entry (see `resource::Waterfall`).
                let per_lane = self.x.res.waterfall.loads.contains(load);
                let h = self
                    .s
                    .heap_handle(self.cur, index, sampler, per_lane, props);
                self.handles.insert(key, h);
                Ok(h)
            }
            Source::Constant(words) if sampler => self.static_sampler(words, props),
            Source::Constant(_) => Err("a constant resource descriptor"),
            Source::Built { address, .. } if address.param == MEMORY => {
                Err("a descriptor built around title memory chosen by control flow")
            }
            Source::Built { address, .. } => {
                let rp = *self
                    .x
                    .native
                    .layout
                    .get(address.param)
                    .ok_or("a descriptor built around a table in the code")?;
                let kind = self
                    .x
                    .native
                    .used
                    .iter()
                    .find(|u| u.user_data == rp.user_data)
                    .map(|u| u.kind);
                match (rp.root, kind, props.0) {
                    (RootType::Descriptor, Some(ResourceType::Cbv) | None, 13) => {
                        Ok(self.s.root_cbv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) => {
                        Ok(self.s.root_srv_handle(rp.slot))
                    }
                    // A raw buffer the program may write: a root UAV.
                    (RootType::Descriptor, Some(ResourceType::Uav), 0x100b) => {
                        Ok(self.s.root_uav_handle(rp.slot))
                    }
                    // One the program's resource list leaves out: a UAV if
                    // the program writes through it anywhere, so that each
                    // root parameter is one kind of view.
                    (RootType::Descriptor, None, 11 | 0x100b)
                        if super::fixes() & super::FIX_ROOT_INDEX != 0 =>
                    {
                        if props.0 & 0x1000 != 0 || self.root_view_written(address.param) {
                            Ok(self.s.root_uav_handle(rp.slot))
                        } else {
                            Ok(self.s.root_srv_handle(rp.slot))
                        }
                    }
                    _ => Err("a typed root view"),
                }
            }
            Source::Phi { .. } => {
                let _ = site;
                let index = self.uniform_heap_index(src, sampler)?;
                Ok(self.s.heap_handle(self.cur, index, sampler, false, props))
            }
            Source::Select { scc, args } => {
                // The heap index of each choice, picked by SCC (uniform).
                let [
                    Source::Loaded { load: a, first: fa },
                    Source::Loaded { load: b, first: fb },
                ] = args[..]
                else {
                    return Err("a descriptor chosen by a condition");
                };
                let cond = match self.get(*scc)? {
                    Val::Bit(x) => x,
                    _ => return Err("SCC without a value"),
                };
                let set = self.heap_index(a, fa, sampler)?;
                let clear = self.heap_index(b, fb, sampler)?;
                let index = self.s.m.select(self.cur, cond, set, clear);
                Ok(self.s.heap_handle(self.cur, index, sampler, false, props))
            }
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => {
                // A descriptor each lane has its own of (a waterfall loop
                // reads it from the first active lane): each lane's heap
                // index, non-uniform.
                let index = self.lane_heap_index(src, sampler)?;
                Ok(self.s.heap_handle(self.cur, index, sampler, true, props))
            }
        }
    }

    /// The sampler of a join of immediate samplers (and a heap one): the
    /// handle of the first choice, and for each other choice the condition
    /// that the path taken brought it and its handle. Sampling is repeated
    /// with each, and the result of the choice taken kept, so that every
    /// lane samples in the same (uniform) control flow. Only in the last
    /// chance: a sampler chosen by control flow was refused before.
    ///
    /// The handles of immediate samplers are one binding each, so a handle
    /// cannot be picked by an index.
    fn sampler_choices(
        &mut self,
        src: &Source,
        props: (u32, u32),
    ) -> Result<Option<SamplerChoices>, Why> {
        let Source::Phi { .. } = src else {
            return Ok(None);
        };
        if !self.leftovers {
            return Ok(None);
        }
        let mut constants: Vec<&Vec<u32>> = Vec::new();
        let mut heap = false;
        for leaf in src.leaves() {
            match leaf {
                Source::Constant(w) => {
                    if !constants.contains(&w) {
                        constants.push(w);
                    }
                }
                Source::Loaded { .. } => heap = true,
                _ => return Ok(None),
            }
        }
        if constants.is_empty() {
            return Ok(None);
        }
        // Each choice's number: 0 for the heap's, then the immediate ones.
        let first = usize::from(heap);
        let kind = self.sampler_kind(src, &constants, first)?;
        let c = self.cur;
        let mut handles = Vec::new();
        if heap {
            let index = self.heap_index_joined(src, true, true)?;
            handles.push(self.s.heap_handle(c, index, true, false, props));
        }
        for w in &constants {
            handles.push(self.static_sampler(w, props)?);
        }
        let mut choices = handles.into_iter().enumerate();
        let (_, primary) = choices.next().expect("a choice");
        let mut others = Vec::new();
        for (k, h) in choices {
            let n = self.s.m.i32(k as u32);
            let taken = self.s.m.cmp(c, pred::EQ, kind, n);
            others.push((taken, h));
        }
        Ok(Some((primary, others)))
    }

    /// The number of the choice a join of samplers takes, as a phi of the
    /// number each predecessor brings (`first` for the first of the
    /// `constants`, 0 for a heap sampler).
    fn sampler_kind(
        &mut self,
        src: &Source,
        constants: &[&Vec<u32>],
        first: usize,
    ) -> Result<V, Why> {
        match src {
            Source::Constant(w) => {
                let k = constants.iter().position(|c| *c == w).expect("listed");
                Ok(self.s.m.i32((first + k) as u32))
            }
            Source::Loaded { .. } => Ok(self.s.m.i32(0)),
            Source::Phi { block, args } => {
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail = self.tail[q].ok_or("a descriptor chosen by control flow")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let here = std::mem::replace(&mut self.cur, at);
                    let kind = self.sampler_kind(arg, constants, first);
                    self.cur = here;
                    incoming.push((kind?, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a descriptor chosen by control flow"),
        }
    }

    /// The heap index of a descriptor every lane shares, computed in the
    /// current block. One chosen by control flow is a phi at its join of the
    /// index each predecessor computes at its end; an edge may bring another
    /// join's (nested if/else).
    fn uniform_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        self.heap_index_joined(src, sampler, false)
    }

    /// [`Self::uniform_heap_index`]; with `constants` an immediate sampler
    /// among the choices gives index 0 (a sampler chosen among immediate ones
    /// and a heap one, see [`Self::sampler_choices`]).
    fn heap_index_joined(
        &mut self,
        src: &Source,
        sampler: bool,
        constants: bool,
    ) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Constant(_) if constants => Ok(self.s.m.i32(0)),
            Source::Phi { block, args } => {
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail = self.tail[q].ok_or("a descriptor chosen by control flow")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let here = std::mem::replace(&mut self.cur, at);
                    let index = self.heap_index_joined(arg, sampler, constants);
                    self.cur = here;
                    incoming.push((index?, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a descriptor chosen by control flow"),
        }
    }

    /// The heap index of a descriptor that may differ between lanes,
    /// computed in the current block (or, chosen by control flow, at the
    /// join).
    fn lane_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Fetched { load, first } => self.fetched_index(*load, *first, sampler),
            Source::Lanes { args, .. } | Source::Select { args, .. } => {
                let [a, b] = &args[..] else {
                    return Err("a descriptor chosen per lane");
                };
                let cond = match src {
                    // The exact pair: in wave64 either dword may have
                    // changed alone since the compare.
                    Source::Lanes { mask, high, .. } if self.split.is_some() => {
                        let c = self.cur;
                        self.exact_pair(*mask, *high, c)?
                    }
                    Source::Lanes { mask, high, .. } => {
                        let m = self
                            .x
                            .st
                            .mask_bits(*mask, *high)
                            .ok_or("a descriptor chosen per lane by an untracked mask")?;
                        self.boolean(m, self.cur)?
                    }
                    Source::Select { scc, .. } => match self.get(*scc)? {
                        Val::Bit(x) => x,
                        _ => return Err("SCC without a value"),
                    },
                    _ => unreachable!(),
                };
                let set = self.lane_heap_index(a, sampler)?;
                let clear = self.lane_heap_index(b, sampler)?;
                Ok(self.s.m.select(self.cur, cond, set, clear))
            }
            // Split: the wave arrives by one edge, its lanes by several.
            Source::LanePhi { .. } if self.split.is_some() => {
                Err("a per-lane descriptor chosen by control flow (split lowering)")
            }
            Source::LanePhi { block, args } => {
                // Each predecessor computes its heap index at its end (what
                // it reads is there: the dwords are, on that edge); the join
                // picks the one of the edge the lane came by. An edge no
                // lane reading it takes gives index 0.
                let join =
                    self.block[*block].ok_or("a per-lane descriptor chosen by control flow")?;
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail =
                        self.tail[q].ok_or("a per-lane descriptor joined from a later block")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let index = match arg {
                        Some(arg) => {
                            let here = std::mem::replace(&mut self.cur, at);
                            let index = self.lane_heap_index(arg, sampler);
                            self.cur = here;
                            index?
                        }
                        None => self.s.m.i32(0),
                    };
                    incoming.push((index, at));
                }
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a per-lane descriptor from neither a heap load nor a choice of them"),
        }
    }

    /// The heap index of the descriptor at dword `first` of what the global
    /// load at `load` reads, in each lane: its address (base plus offsets)
    /// over the descriptor size.
    fn fetched_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        const WHY: Why = "a per-lane descriptor load";
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            saddr,
            offset,
            ..
        } = li.fields
        else {
            return Err(WHY);
        };
        let c = self.cur;
        // The low dword of the address: SGPR base plus VGPR offset, or the
        // VGPR pair's low half (heap addresses fit in 32 bits).
        let mut at = self.i32_of(li.read(Reg::V(reg)).ok_or(WHY)?)?;
        if let Some(Operand::Scalar { code, .. }) = saddr
            && code != NULL
        {
            let base = self.i32_of(li.read(Reg::S(code)).ok_or(WHY)?)?;
            at = self.s.m.bin(c, Bin::Add, base, at);
        }
        let k = self
            .s
            .m
            .i32((offset as i32 as u32).wrapping_add(4 * u32::from(first)));
        at = self.s.m.bin(c, Bin::Add, at, k);
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        let index = self.s.m.bin(c, Bin::LShr, at, shift);
        Ok(self.title_slot(a.param, index))
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block.
    fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &li.fields
        else {
            return Err("a descriptor load");
        };
        let Operand::Scalar { code, .. } = *base else {
            return Err("a descriptor load");
        };
        let c = self.cur;
        let base = li.read(Reg::S(code)).ok_or("a descriptor load")?;
        let mut addr = self.i32_of(base)?;
        let k = self
            .s
            .m
            .i32((*offset as u32).wrapping_add(4 * u32::from(first)));
        addr = self.s.m.bin(c, Bin::Add, addr, k);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        let index = self.s.m.bin(c, Bin::LShr, addr, shift);
        Ok(self.title_slot(a.param, index))
    }

    /// `index`, a heap slot found through root parameter `param`, as the
    /// slot of the title's own heap where a record of the host's buffer view
    /// table is looked up ([`Self::title_slots`]): the host may bind its own
    /// copy of a root table in other slots, and the table lists the title's
    /// (see [`super::USER_TABLE_SLOTS`]).
    fn title_slot(&mut self, param: usize, index: V) -> V {
        let table = self
            .x
            .native
            .layout
            .get(param)
            .filter(|p| self.title_slots && p.root == RootType::Table)
            .map(|p| p.slot);
        let Some(slot) = table else {
            return index;
        };
        let ahead = self.host_word(RootType::Other(super::USER_TABLE_SLOTS), slot, 0);
        self.s.m.bin(self.cur, Bin::Add, index, ahead)
    }

    fn valu(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if self.x.res.waterfall.compares.contains(&(b, i)) {
            // The first lane's dwords of a waterfall loop are each lane's
            // own (see `resource::Waterfall`).
            self.lane(inst, Some(true));
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("v_cmp_").or(name.strip_prefix("v_cmpx_")) {
            let r = if v.dpp.is_some() {
                Err("a compare with DPP")
            } else {
                self.compare(inst, rest)
            };
            if let Err(why) = r {
                for &(_, w) in &inst.writes {
                    self.lane_fail.insert(w, why);
                }
            }
            return r;
        }
        // A scalar result (`v_readlane`, `v_readfirstlane`) is written
        // whatever EXEC holds.
        let mut guard = match v.dst {
            Some(Operand::Scalar { .. }) => None,
            _ => self.guard(b, i)?,
        };
        if let Some(d) = v.dpp {
            let (x, enable) = self.dpp(inst, d, guard)?;
            guard = match (guard, enable) {
                (Some(g), Some(e)) => Some(self.s.m.bin(self.cur, Bin::And, g, e)),
                (g, e) => g.or(e),
            };
            self.src0 = Some(x);
        }
        let r = self.valu_op(inst, guard);
        self.src0 = None;
        r
    }

    /// Whether lane `src` is active in the current EXEC (via `waveActiveBallot`).
    fn src_lane_active(&mut self, src: V, exec: Option<V>) -> V {
        let c = self.cur;
        if self.console_cf {
            // DPP and swizzles only address lanes within their own half.
            let active = exec.unwrap_or_else(|| self.s.m.bool(true));
            let word = self.ballot_word(c, active);
            let mask = self.s.m.i32(31);
            let one = self.s.m.i32(1);
            let zero = self.s.m.i32(0);
            let shift = self.s.m.bin(c, Bin::And, src, mask);
            let bit = self.s.m.bin(c, Bin::LShr, word, shift);
            let bit = self.s.m.bin(c, Bin::And, bit, one);
            return self.s.m.cmp(c, pred::NE, bit, zero);
        }
        let i32t = self.s.t.i32;
        let t = self.s.m.bool(true);
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[t]);
        self.s.flags |= flags::WAVE_OPS;
        let lo = self.s.m.extract(c, ballot, 0);
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = if self.wave() > 32 {
            let hi = self.s.m.extract(c, ballot, 1);
            let low = self.s.m.cmp(c, pred::IULT, src, k32);
            self.s.m.select(c, low, lo, hi)
        } else {
            lo
        };
        let sh = self.s.m.bin(c, Bin::And, src, k31);
        let bit = self.s.m.bin(c, Bin::LShr, word, sh);
        let bit = self.s.m.bin(c, Bin::And, bit, one);
        self.s.m.cmp(c, pred::NE, bit, z)
    }

    /// DPP: src0 as read from another lane, and which lanes write their
    /// result (none: all).
    ///
    /// When `fi` is clear, an inactive source lane is out of bounds the same
    /// way a row-bound miss is: `bound_ctrl` selects zero, else the lane
    /// keeps its old destination. DPP8 has no `bound_ctrl` bit and always
    /// zeros inactive sources when `fi` is clear.
    fn dpp(&mut self, inst: &Inst, dpp: Dpp, exec: Option<V>) -> Result<(V, Option<V>), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return Err("a DPP operand");
        };
        let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("a DPP operand")?)?;
        // Quad permutations, and DPP8 where every lane picks a lane of its
        // own quad, stay in groups the pixel layout shares.
        let quad = match dpp {
            Dpp::Row { ctrl, .. } => ctrl <= 0xFF,
            Dpp::Lanes { sel, .. } => (0..8).all(|i| ((sel >> (3 * i)) ^ i) & 4 == 0),
        };
        let fi = match dpp {
            Dpp::Lanes { fi, .. } | Dpp::Row { fi, .. } => fi,
        };
        let mesh_quad = self.s.mesh.is_some() && quad;
        if !mesh_quad {
            self.cross_lane(quad)?;
        } else if !fi {
            // Inactive-source checks need a wave ballot; mesh cannot.
            self.cross_lane(false)?;
        }
        let c = self.cur;
        let lane = self.lane_index();
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (sixteen_mask, low) = (k(self, !15), k(self, 15));
        let row = self.s.m.bin(c, Bin::And, lane, sixteen_mask);
        let pos = self.s.m.bin(c, Bin::And, lane, low);
        // Source lane, whether it lies in the row, and bound_ctrl (zero on
        // miss). DPP8 always zeros inactive sources when `fi` is clear.
        let (src, inside, masks, zero) = match dpp {
            Dpp::Lanes { sel, .. } => {
                let (seven, not7, three) = (k(self, 7), k(self, !7), k(self, 3));
                let p = self.s.m.bin(c, Bin::And, lane, seven);
                let sh = self.s.m.bin(c, Bin::Mul, p, three);
                let sel = k(self, sel);
                let pick = self.s.m.bin(c, Bin::LShr, sel, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, seven);
                let base = self.s.m.bin(c, Bin::And, lane, not7);
                (self.s.m.bin(c, Bin::Or, base, pick), None, None, true)
            }
            Dpp::Row {
                ctrl,
                row_mask,
                bank_mask,
                bound_ctrl,
                ..
            } => {
                let n = u32::from(ctrl & 15);
                let (src, inside) = match ctrl {
                    0..=0xFF => {
                        let (three, one, not3) = (k(self, 3), k(self, 1), k(self, !3));
                        let p = self.s.m.bin(c, Bin::And, lane, three);
                        let sh = self.s.m.bin(c, Bin::Shl, p, one);
                        let perm = k(self, u32::from(ctrl));
                        let pick = self.s.m.bin(c, Bin::LShr, perm, sh);
                        let pick = self.s.m.bin(c, Bin::And, pick, three);
                        let base = self.s.m.bin(c, Bin::And, lane, not3);
                        (self.s.m.bin(c, Bin::Or, base, pick), None)
                    }
                    0x101..=0x10F | 0x111..=0x11F => {
                        let nn = k(self, n);
                        let left = ctrl < 0x110;
                        let (op, p) = if left {
                            (Bin::Add, pred::IULT)
                        } else {
                            (Bin::Sub, pred::IUGE)
                        };
                        let s = self.s.m.bin(c, op, lane, nn);
                        // Left: pos + n < 16; right: pos >= n.
                        let bound = if left { k(self, 16 - n) } else { nn };
                        (s, Some(self.s.m.cmp(c, p, pos, bound)))
                    }
                    0x121..=0x12F => {
                        let nn = k(self, n);
                        let t = self.s.m.bin(c, Bin::Sub, pos, nn);
                        let t = self.s.m.bin(c, Bin::And, t, low);
                        (self.s.m.bin(c, Bin::Or, row, t), None)
                    }
                    0x140 | 0x141 => {
                        let m = k(self, if ctrl == 0x140 { 15 } else { 7 });
                        let t = self.s.m.bin(c, Bin::Xor, lane, m);
                        (t, None)
                    }
                    0x150..=0x15F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Or, row, nn), None)
                    }
                    0x160..=0x16F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Xor, lane, nn), None)
                    }
                    _ => return Err("a DPP lane pattern"),
                };
                let masks = (row_mask != 0xF || bank_mask != 0xF).then_some((row_mask, bank_mask));
                (src, inside, masks, bound_ctrl)
            }
        };
        let mut r = if mesh_quad {
            let pattern = match dpp {
                Dpp::Row { ctrl, .. } => Some(ctrl as u8),
                Dpp::Lanes { .. } => None,
            };
            self.quad_pick(x, src, pattern)
        } else {
            self.read_lane(x, src)
        };
        // Geometric bounds, and source EXEC when fetch-inactive is clear.
        let mut valid = inside;
        if !fi {
            let active = self.src_lane_active(src, exec);
            valid = Some(match valid {
                Some(v) => self.s.m.bin(c, Bin::And, v, active),
                None => active,
            });
        }
        let mut enable = None;
        if let Some(inside) = valid {
            if zero {
                let z = k(self, 0);
                r = self.s.m.select(c, inside, r, z);
            } else {
                enable = Some(inside);
            }
        }
        if let Some((rows, banks)) = masks {
            // Row (16 lanes) and bank (4 lanes of a row) write masks. Rows
            // are the console's: split in halves, the upper half's host rows
            // 0 and 1 are console rows 2 and 3.
            let (four, two, three) = (k(self, 4), k(self, 2), k(self, 3));
            let row_lane = if self.halves {
                self.console_lane().0
            } else {
                lane
            };
            let ri = self.s.m.bin(c, Bin::LShr, row_lane, four);
            let ri = self.s.m.bin(c, Bin::And, ri, three);
            let bi = self.s.m.bin(c, Bin::LShr, lane, two);
            let bi = self.s.m.bin(c, Bin::And, bi, three);
            let (rm, bm) = (k(self, u32::from(rows)), k(self, u32::from(banks)));
            let rb = self.s.m.bin(c, Bin::LShr, rm, ri);
            let bb = self.s.m.bin(c, Bin::LShr, bm, bi);
            let both = self.s.m.bin(c, Bin::And, rb, bb);
            let (one, z) = (k(self, 1), k(self, 0));
            let both = self.s.m.bin(c, Bin::And, both, one);
            let on = self.s.m.cmp(c, pred::NE, both, z);
            enable = Some(match enable {
                Some(e) => self.s.m.bin(c, Bin::And, e, on),
                None => on,
            });
        }
        let fi = match dpp {
            Dpp::Row { fi, .. } | Dpp::Lanes { fi, .. } => fi,
        };
        if self.split.is_some() && !fi {
            // Split: lanes EXEC switches off are there with stale values.
            // Without fetch-inactive, one as the source gives 0 (bound_ctrl,
            // DPP8) or leaves the destination alone.
            let e = self.split_exec(inst)?;
            let ei = self.s.m.cast(c, Cast::ZExt, e, self.s.t.i32);
            let there = self.read_lane(ei, src);
            let z = k(self, 0);
            let on = self.s.m.cmp(c, pred::NE, there, z);
            if zero || matches!(dpp, Dpp::Lanes { .. }) {
                r = self.s.m.select(c, on, r, z);
            } else {
                enable = Some(match enable {
                    Some(e) => self.s.m.bin(c, Bin::And, e, on),
                    None => on,
                });
            }
        }
        Ok((r, enable))
    }

    fn valu_op(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let c = self.cur;
        let f = |l: &mut Self, k| l.fsrc(inst, k);
        let s = |l: &mut Self, k| l.src(inst, k);
        let float = |l: &mut Self, r: V| l.write_f(inst, r, guard);
        let int = |l: &mut Self, r: V| l.write_i(inst, r, guard);
        if let Some(r) = self.valu16(inst, name, guard) {
            return r;
        }
        if let Some(op) = name.strip_prefix("v_pk_") {
            return self.packed16(inst, op, guard);
        }
        match name {
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
                let n = s(self, 0)?;
                let x = self.src64(inst, 1)?;
                let i64t = self.s.m.int(64);
                let k63 = self.s.m.i32(63);
                let n = self.s.m.bin(c, Bin::And, n, k63);
                let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                let op = match name {
                    "v_lshlrev_b64" => Bin::Shl,
                    "v_lshrrev_b64" => Bin::LShr,
                    _ => Bin::AShr,
                };
                let r = self.s.m.bin(c, op, x, n);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 64-bit result");
                };
                let (lo, hi) = self.split64(r);
                self.write(inst, Reg::V(reg), lo, guard);
                self.write(inst, Reg::V(reg + 1), hi, guard);
                Ok(())
            }
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                // Cube-map face coordinates (AMD's ISA): the major axis is z
                // when |z| >= |x| and |z| >= |y|, else y when |y| >= |x|,
                // else x.
                let (x, y, z) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (ax, ay, az) = (self.unary(6, x), self.unary(6, y), self.unary(6, z));
                let zx = self.s.m.cmp(c, pred::OGE, az, ax);
                let zy = self.s.m.cmp(c, pred::OGE, az, ay);
                let zmaj = self.s.m.bin(c, Bin::And, zx, zy);
                let ymaj = self.s.m.cmp(c, pred::OGE, ay, ax);
                let zero = self.s.m.f32(0);
                let neg = |l: &mut Self, v: V| l.s.m.cmp(c, pred::OLT, v, zero);
                let minus = |l: &mut Self, v: V| {
                    let z = l.s.m.f32(0x8000_0000);
                    l.s.m.bin(c, Bin::Sub, z, v)
                };
                let pick = |l: &mut Self, zv: V, yv: V, xv: V| {
                    let t = l.s.m.select(c, ymaj, yv, xv);
                    l.s.m.select(c, zmaj, zv, t)
                };
                let r = match name {
                    "v_cubeid_f32" => {
                        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
                        let (nz, ny, nx) = (neg(self, z), neg(self, y), neg(self, x));
                        let (f5, f4, f3, f2, f1, f0) = (
                            k(self, 5.0),
                            k(self, 4.0),
                            k(self, 3.0),
                            k(self, 2.0),
                            k(self, 1.0),
                            k(self, 0.0),
                        );
                        let zf = self.s.m.select(c, nz, f5, f4);
                        let yf = self.s.m.select(c, ny, f3, f2);
                        let xf = self.s.m.select(c, nx, f1, f0);
                        pick(self, zf, yf, xf)
                    }
                    "v_cubesc_f32" => {
                        let (nz, nx) = (neg(self, z), neg(self, x));
                        let (mx, mz) = (minus(self, x), minus(self, z));
                        let zf = self.s.m.select(c, nz, mx, x);
                        let xf = self.s.m.select(c, nx, z, mz);
                        pick(self, zf, x, xf)
                    }
                    "v_cubetc_f32" => {
                        let ny = neg(self, y);
                        let (my, mz) = (minus(self, y), minus(self, z));
                        let yf = self.s.m.select(c, ny, mz, z);
                        pick(self, my, yf, my)
                    }
                    _ => {
                        let t = pick(self, z, y, x);
                        let two = self.s.m.f32(2.0f32.to_bits());
                        self.s.m.bin(c, Bin::Mul, t, two)
                    }
                };
                float(self, r)
            }
            "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" => {
                // M0 indexes the source (movrels), the destination (movreld)
                // or both, from the encoded register up to the program's
                // VGPR count (the registers the instruction reads or writes).
                let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("an M0-relative move");
                };
                let indexed_src = name != "v_movreld_b32";
                let value = if indexed_src {
                    let mut r = self.s.m.i32(0);
                    for k in 0..=255u8.saturating_sub(sreg) {
                        let Some(x) = inst.read(Reg::V(sreg + k)) else {
                            break;
                        };
                        let x = self.i32_of(x)?;
                        let kk = self.s.m.i32(u32::from(k));
                        let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                        r = self.s.m.select(c, is, x, r);
                    }
                    r
                } else {
                    s(self, 0)?
                };
                if name == "v_movrels_b32" {
                    return int(self, value);
                }
                for k in 0..=255u8.saturating_sub(dreg) {
                    let Some(_) = inst.write(Reg::V(dreg + k)) else {
                        break;
                    };
                    let old = match inst.read(Reg::V(dreg + k)) {
                        Some(o) => self.i32_of(o)?,
                        None => self.s.m.i32(0),
                    };
                    let kk = self.s.m.i32(u32::from(k));
                    let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                    let x = self.s.m.select(c, is, value, old);
                    self.write_placed(inst, Reg::V(dreg + k), x, guard);
                }
                Ok(())
            }
            "v_swap_b32" => {
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("v_swap_b32");
                };
                let a = self.i32_of(inst.read(Reg::V(dreg)).ok_or("v_swap_b32")?)?;
                let b = self.i32_of(inst.read(Reg::V(sreg)).ok_or("v_swap_b32")?)?;
                self.write_placed(inst, Reg::V(dreg), b, guard);
                if dreg != sreg {
                    self.write_placed(inst, Reg::V(sreg), a, guard);
                }
                Ok(())
            }
            "v_ffbh_u32" | "v_ffbh_i32" => {
                // Counted from the top bit, -1 when none (DXIL's too).
                let a = s(self, 0)?;
                let op = if name == "v_ffbh_u32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                int(self, r)
            }
            "v_mov_b32" => {
                if v.clamp || v.omod != 0 {
                    let a = f(self, 0)?;
                    return float(self, a);
                }
                let a = self.bits_src(inst, 0)?;
                int(self, a)
            }
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
            | "v_max_f32" | "v_min_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let r = match name {
                    "v_add_f32" => self.s.m.bin(c, Bin::Add, a, bb),
                    "v_sub_f32" => self.s.m.bin(c, Bin::Sub, a, bb),
                    "v_subrev_f32" => self.s.m.bin(c, Bin::Sub, bb, a),
                    "v_mul_f32" => self.s.m.bin(c, Bin::Mul, a, bb),
                    "v_mul_legacy_f32" => self.legacy_mul(a, bb),
                    "v_max_f32" => self.max_min(true, a, bb),
                    _ => self.max_min(false, a, bb),
                };
                float(self, r)
            }
            "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" => {
                // shortcut: DXIL has no fused f32 multiply-add; FMad lets the driver fuse.
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fmad(a, bb, cc);
                float(self, r)
            }
            "v_fmac_f32" | "v_mac_f32" | "v_fmac_legacy_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
                let acc = match self.floats.get(&acc) {
                    Some(&x) => x,
                    None => {
                        let x = self.i32_of(acc)?;
                        self.as_float(x)
                    }
                };
                let r = match name {
                    "v_fmac_f32" => self.fmad(a, bb, acc),
                    "v_mac_f32" => self.mad_f32(a, bb, acc),
                    _ => self.fma_legacy(a, bb, acc),
                };
                // `v_mac_f32` flushes subnormals around omod; fused forms do not.
                if name == "v_mac_f32" {
                    self.write_f_flush_denorms(inst, r, guard)
                } else {
                    float(self, r)
                }
            }
            "v_fma_legacy_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fma_legacy(a, bb, cc);
                float(self, r)
            }
            "v_mad_f32" | "v_madak_f32" | "v_madmk_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.mad_f32(a, bb, cc);
                self.write_f_flush_denorms(inst, r, guard)
            }
            "v_max3_f32" | "v_min3_f32" | "v_med3_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = match name {
                    "v_max3_f32" => {
                        let t = self.max_min(true, a, bb);
                        self.max_min(true, t, cc)
                    }
                    "v_min3_f32" => {
                        let t = self.max_min(false, a, bb);
                        self.max_min(false, t, cc)
                    }
                    _ => self.med3(a, bb, cc),
                };
                float(self, r)
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                // shortcut: the hardware reciprocal is approximate (1 ULP); this divides.
                let a = f(self, 0)?;
                let one = self.s.m.f32(1.0f32.to_bits());
                let r = self.s.m.bin(c, Bin::SDiv, one, a);
                float(self, r)
            }
            "v_rsq_f32" | "v_sqrt_f32" | "v_log_f32" | "v_exp_f32" | "v_fract_f32"
            | "v_trunc_f32" | "v_floor_f32" | "v_ceil_f32" | "v_rndne_f32" | "v_sin_f32"
            | "v_cos_f32" => {
                let a = f(self, 0)?;
                let op = match name {
                    "v_rsq_f32" => 25,
                    "v_sqrt_f32" => 24,
                    "v_log_f32" => 23,
                    "v_exp_f32" => 21,
                    "v_fract_f32" => 22,
                    "v_trunc_f32" => 29,
                    "v_floor_f32" => 27,
                    "v_ceil_f32" => 28,
                    "v_rndne_f32" => 26,
                    "v_sin_f32" => 13,
                    _ => 12,
                };
                let r = match op {
                    12 | 13 => self.trig_turns(a, op == 12),
                    22 => self.fract(a, 1.0, f32::from_bits(0x3F7F_FFFF)),
                    _ => self.unary(op, a),
                };
                float(self, r)
            }
            "v_cvt_i32_f32" | "v_cvt_u32_f32" | "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => {
                let mut a = f(self, 0)?;
                if name == "v_cvt_rpi_i32_f32" {
                    let h = self.s.m.f32(0.5f32.to_bits());
                    a = self.s.m.bin(c, Bin::Add, a, h);
                }
                if name != "v_cvt_i32_f32" && name != "v_cvt_u32_f32" {
                    a = self.unary(27, a);
                }
                // D3D's conversions saturate and give 0 for NaN, as the
                // hardware does (checked on the RX 6700 XT and WARP).
                let op = if name == "v_cvt_u32_f32" {
                    Cast::FpToUi
                } else {
                    Cast::FpToSi
                };
                let r = self.s.m.cast(c, op, a, self.s.t.i32);
                int(self, r)
            }
            "v_cvt_f32_i32" | "v_cvt_f32_u32" | "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1"
            | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let mut a = s(self, 0)?;
                if let Some(n) = name.strip_prefix("v_cvt_f32_ubyte") {
                    let sh = self.s.m.i32(8 * u32::from(n.as_bytes()[0] - b'0'));
                    let ff = self.s.m.i32(0xff);
                    a = self.s.m.bin(c, Bin::LShr, a, sh);
                    a = self.s.m.bin(c, Bin::And, a, ff);
                }
                let op = if name == "v_cvt_f32_i32" {
                    Cast::SiToFp
                } else {
                    Cast::UiToFp
                };
                let r = self.s.m.cast(c, op, a, self.s.t.f32);
                float(self, r)
            }
            "v_cvt_off_f32_i4" => {
                // The low nibble as a signed 4-bit integer, over 16.
                let a = s(self, 0)?;
                let k28 = self.s.m.i32(28);
                let a = self.s.m.bin(c, Bin::Shl, a, k28);
                let a = self.s.m.bin(c, Bin::AShr, a, k28);
                let f = self.s.m.cast(c, Cast::SiToFp, a, self.s.t.f32);
                let k = self.s.m.f32(0.0625f32.to_bits());
                let r = self.s.m.bin(c, Bin::Mul, f, k);
                float(self, r)
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                // Each source is f32, or with op_sel_hi the low (op_sel: high)
                // f16 half; neg_hi is the absolute value here.
                let mut xs = Vec::new();
                for i in 0..3 {
                    let op = v.src.get(i).ok_or("a missing operand")?;
                    let x = self.operand(inst, op)?;
                    let mut x = if v.opsel_hi >> i & 1 != 0 {
                        let h = if v.opsel >> i & 1 != 0 {
                            let k16 = self.s.m.i32(16);
                            self.s.m.bin(c, Bin::LShr, x, k16)
                        } else {
                            x
                        };
                        self.s
                            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h])
                    } else {
                        self.as_float(x)
                    };
                    if v.neg_hi >> i & 1 != 0 {
                        x = self.unary(6, x);
                    }
                    if v.neg >> i & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        x = self.s.m.bin(c, Bin::Sub, z, x);
                    }
                    xs.push(x);
                }
                if name == "v_fma_mix_f32" {
                    let r = self.fmad(xs[0], xs[1], xs[2]);
                    return float(self, r);
                }
                // Single-round to f16: round-to-odd in f32, then nearest-even.
                let mut r = self.fmad_round_to_odd(xs[0], xs[1], xs[2]);
                if v.clamp {
                    r = self.unary(7, r);
                }
                // The f16 result goes to one half; the other half stays.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a mixed-precision result");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                let h = self.f16_nearest(r);
                let (keep, h) = if name == "v_fma_mixhi_f16" {
                    let k16 = self.s.m.i32(16);
                    (0xFFFF, self.s.m.bin(c, Bin::Shl, h, k16))
                } else {
                    (0xFFFF_0000, h)
                };
                let keep = self.s.m.i32(keep);
                let old = self.s.m.bin(c, Bin::And, old, keep);
                let x = self.s.m.bin(c, Bin::Or, old, h);
                self.write(inst, Reg::V(reg), x, guard);
                Ok(())
            }
            "v_cvt_pkrtz_f16_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let lo = self.f16_toward_zero(a);
                let hi = self.f16_toward_zero(bb);
                let sh = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, hi, sh);
                let r = self.s.m.bin(c, Bin::Or, lo, hi);
                int(self, r)
            }
            "v_pack_b32_f16" => {
                // Two f16 sources (op_sel: their high halves; abs and neg on
                // the sign bit) into the low and high half.
                let mut halves = Vec::new();
                for k in 0..2 {
                    let op = v.src.get(k).ok_or("a missing operand")?;
                    let mut x = self.operand(inst, op)?;
                    if v.opsel >> k & 1 != 0 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let mask = if v.abs >> k & 1 != 0 { 0x7FFF } else { 0xFFFF };
                    let mask = self.s.m.i32(mask);
                    x = self.s.m.bin(c, Bin::And, x, mask);
                    if v.neg >> k & 1 != 0 {
                        let sign = self.s.m.i32(0x8000);
                        x = self.s.m.bin(c, Bin::Xor, x, sign);
                    }
                    halves.push(x);
                }
                let k16 = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
                let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
                int(self, r)
            }
            "v_cndmask_b32" => {
                let cond = self.select_condition(inst, v.src.get(2))?;
                let (a, bb) = (self.bits_src(inst, 0)?, self.bits_src(inst, 1)?);
                let r = self.s.m.select(c, cond, bb, a);
                int(self, r)
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32"
            | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                // The carry (borrow) out is a lane mask: this lane's bit.
                let (mut a, mut bb) = (s(self, 0)?, s(self, 1)?);
                if name.contains("subrev") {
                    std::mem::swap(&mut a, &mut bb);
                }
                let cin = if name.ends_with("_ci_u32") && self.split.is_some() {
                    let Some(&Operand::Scalar { code, .. }) = v.src.get(2) else {
                        return Err("a carry-in from an untracked mask");
                    };
                    let lo = inst.read(Reg::S(code)).ok_or("a carry-in")?;
                    Some(self.exact_pair(lo, inst.read(Reg::S(code + 1)), c)?)
                } else if name.ends_with("_ci_u32") {
                    let m = match v.src.get(2) {
                        Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
                        _ => None,
                    };
                    Some(match m.and_then(|m| self.x.st.masks.get(&m).copied()) {
                        Some(m) => self.boolean(m, c)?,
                        // Not followed by structure: this lane's bit of the register.
                        None => self.select_condition(inst, v.src.get(2))?,
                    })
                } else {
                    None
                };
                let add = name.starts_with("v_add");
                let (mut r, carry) = self.add_sub_carry(c, add, a, bb, cin);
                if v.clamp {
                    r = self.clamp_u32(c, add, r, carry);
                }
                int(self, r)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    self.lane_bits.insert(w, carry);
                    if self.split.is_some() {
                        self.split_lanes_written(inst, code, carry);
                    }
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            "v_readlane_b32" => {
                let a = match s(self, 0) {
                    Ok(a) => a,
                    // The vector holds lanes the code cannot follow (a spill
                    // of an address, say), but this lane may be one it can:
                    // the SGPR spilled there is what the lane reads.
                    Err(why) => {
                        let Some(m) = self.restored_lane(inst).filter(|_| self.second) else {
                            return Err(why);
                        };
                        let x = match m {
                            crate::resource::Moved::Const(k) => self.s.m.i32(k),
                            crate::resource::Moved::Value(v) => self.i32_of(v)?,
                        };
                        return int(self, x);
                    }
                };
                let lane = s(self, 1)?;
                let lane = wave_lane(&mut self.s.m, c, lane, self.x.native.wave64);
                // A computed lane in a pixel or vertex program is a host lane
                // (from a mask or `v_mbcnt`); a constant one is the console's.
                if self.mask_data == MaskData::Host && !matches!(lane, V::Const(_)) {
                    self.s.flags |= flags::WAVE_OPS;
                } else {
                    self.cross_lane(false)?;
                }
                let r = if self.split.is_some() {
                    self.split_readlane(a, lane)
                } else if self.console_cf {
                    self.console_read_lane(a, lane)
                } else {
                    self.read_lane(a, lane)
                };
                int(self, r)
            }
            "v_writelane_b32" => {
                // The lane given takes the scalar; the others keep their value.
                // RDNA2 ignores EXEC for v_writelane_b32 so compilers can spill
                // SGPRs into inactive lanes.
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                let lane = wave_lane(&mut self.s.m, c, lane, self.x.native.wave64);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a lane write");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                self.cross_lane(false)?;
                let me = if self.console_cf {
                    self.console_lane().0
                } else {
                    self.lane_index()
                };
                let is = self.s.m.cmp(c, pred::EQ, me, lane);
                let r = self.s.m.select(c, is, a, old);
                self.write(inst, Reg::V(reg), r, None);
                Ok(())
            }
            "v_permlanex16_b32" => {
                // Each lane reads src0 of a lane of the *other* row of 16 in
                // its 32-lane half; lane i's pick is nibble i of the {src2,
                // src1} selector (v_permlane16_b32 reads the same row). Both
                // stay inside a 32-lane half, so the lane math is unchanged
                // when a console wave64 program runs as two wave32 halves.
                if v.opsel != 0 {
                    return Err("op_sel on v_permlanex16_b32");
                }
                let (a, lo, hi) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                self.cross_lane(false)?;
                let lane = self.lane_index();
                let (k2, k8, k15, k16, k31, n31) = (
                    self.s.m.i32(2),
                    self.s.m.i32(8),
                    self.s.m.i32(15),
                    self.s.m.i32(16),
                    self.s.m.i32(31),
                    self.s.m.i32(!31),
                );
                let nib = self.s.m.bin(c, Bin::And, lane, k15);
                let sh = self.s.m.bin(c, Bin::Shl, nib, k2);
                let sh = self.s.m.bin(c, Bin::And, sh, k31);
                let top = self.s.m.cmp(c, pred::IUGE, nib, k8);
                let word = self.s.m.select(c, top, hi, lo);
                let pick = self.s.m.bin(c, Bin::LShr, word, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, k15);
                let half = self.s.m.bin(c, Bin::And, lane, n31);
                let row = self.s.m.bin(c, Bin::And, lane, k16);
                let row = self.s.m.bin(c, Bin::Xor, row, k16);
                let base = self.s.m.bin(c, Bin::Or, half, row);
                let idx = self.s.m.bin(c, Bin::Or, base, pick);
                let val = self.read_lane(a, idx);
                int(self, val)
            }
            "v_readfirstlane_b32"
                if inst
                    .writes
                    .first()
                    .is_some_and(|w| self.x.res.waterfall.reads.contains(&w.1)) =>
            {
                // A waterfall loop's read of a descriptor dword: the lane's own.
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_readfirstlane_b32" if self.split.is_some() => {
                let a = s(self, 0)?;
                let r = self.split_readfirstlane(inst, a)?;
                int(self, r)
            }
            "v_readfirstlane_b32" if self.console_cf => {
                // The console's first lane in EXEC (lane 0 when none).
                let a = s(self, 0)?;
                let word = |l: &mut Self, r: u8| -> Result<V, Why> {
                    match l.get(inst.read(Reg::S(r)).ok_or("EXEC")?)? {
                        Val::I32(x) => Ok(x),
                        _ => Err(LANE_MASK),
                    }
                };
                let lo = word(self, EXEC)?;
                let hi = if self.x.native.wave64 {
                    word(self, EXEC + 1)?
                } else {
                    self.s.m.i32(0)
                };
                let i32t = self.s.t.i32;
                let flo = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
                let fhi = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
                let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                let fhi = self.s.m.bin(c, Bin::Add, fhi, k32);
                let none_hi = self.s.m.cmp(c, pred::EQ, hi, z);
                let fhi = self.s.m.select(c, none_hi, z, fhi);
                let some_lo = self.s.m.cmp(c, pred::NE, lo, z);
                let first = self.s.m.select(c, some_lo, flo, fhi);
                let r = self.console_read_lane(a, first);
                int(self, r)
            }
            "v_readfirstlane_b32" => {
                let a = s(self, 0)?;
                self.s.flags |= flags::WAVE_OPS;
                let r = self.s.op(
                    c,
                    "waveReadLaneFirst.i32",
                    Attr::NoUnwind,
                    self.s.t.i32,
                    118,
                    &[a],
                );
                let r = self.helper_reads_own(a, r);
                int(self, r)
            }
            _ => {
                let r = self.integer(inst, name)?;
                int(self, r)
            }
        }
    }

    /// A 64-bit source (a register pair; inline constants sign-extend).
    fn src64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<V, Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            Ok(l.join64(lo, hi))
        };
        match op {
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1)),
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))
            }
            Operand::Const(k @ 128..=208) => {
                let x = crate::resource::inline(k).ok_or("a special operand")?;
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            // A signed 64-bit operand sign-extends its 32-bit literal, as
            // the hardware does (LLVM's AMDGPU operand rules).
            Operand::Literal(x) if inst.name().ends_with("_i64") => {
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            _ => {
                // Literals (of unsigned and bit operands) and the null
                // register: zero-extended.
                let x = self.operand(inst, &op)?;
                Ok(self.s.m.cast(c, Cast::ZExt, x, i64t))
            }
        }
    }

    fn join64(&mut self, lo: V, hi: V) -> V {
        let c = self.cur;
        let ty = self.s.m.int(64);
        let lo = self.s.m.cast(c, Cast::ZExt, lo, ty);
        let hi = self.s.m.cast(c, Cast::ZExt, hi, ty);
        let shift = self.s.m.int_const(ty, 32);
        let hi = self.s.m.bin(c, Bin::Shl, hi, shift);
        self.s.m.bin(c, Bin::Or, lo, hi)
    }

    /// The low and high dwords of an i64.
    fn split64(&mut self, x: V) -> (V, V) {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let k32 = self.s.m.int_const(i64t, 32);
        let lo = self.s.m.cast(c, Cast::Trunc, x, self.s.t.i32);
        let hi = self.s.m.bin(c, Bin::LShr, x, k32);
        let hi = self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32);
        (lo, hi)
    }

    /// Source `k` of a 16-bit f16 operation in the low 16 bits (the rest 0):
    /// op_sel (or an SDWA word select) picks the high half, except that
    /// 16-bit compares other than `v_cmp_class` ignore op_sel; inline float
    /// constants are f16 values here.
    fn half_src_f16(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        self.half_src(inst, k, true)
    }

    /// Source `k` of a 16-bit integer (u16/i16/b16) operation in the low 16
    /// bits (the rest 0): same half selection as [`Self::half_src_f16`], but
    /// inline float constants keep their 32-bit f32 bit patterns (hardware
    /// uses the selected 16 of those).
    fn half_src_i16(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        self.half_src(inst, k, false)
    }

    fn half_src(&mut self, inst: &Inst, k: usize, float: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        // RDNA2 ignores VOP3 op_sel on 16-bit compares (v_cmp_class not measured).
        let name = inst.name();
        let compare = name.starts_with("v_cmp") && !name.contains("class");
        let mut high = !compare && v.opsel >> k & 1 != 0;
        if let Some(sd) = v.sdwa {
            match sd.src_sel.get(k).copied().unwrap_or(6) {
                4 => {}
                5 => high = true,
                6 => {}
                _ => return Err("a byte select on a 16-bit operation"),
            }
        }
        let x = match self.src0 {
            Some(x) if k == 0 => x,
            _ if float => self.f16_operand(inst, &op)?,
            _ => self.operand(inst, &op)?,
        };
        let c = self.cur;
        let x = if high {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// An f16 source as an f32 (exact), with its abs/neg modifiers.
    fn half_fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let h = self.half_src_f16(inst, k)?;
        let f32t = self.s.t.f32;
        let mut f = self
            .s
            .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    /// Writes a 16-bit result (low bits of `x`) into the destination's low
    /// half, or its high half with op_sel bit 3 (or an SDWA WORD_1 select);
    /// the other half is kept (GFX10).
    fn write_half(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a 16-bit result outside a VGPR");
        };
        let mut high = v.opsel >> 3 & 1 != 0;
        // SDWA: the word written, the other one kept (preserve) or zero
        // (pad), or the low word's sign bit extended above it (sext; above
        // the high word that adds nothing).
        let (mut keep, mut sext) = (true, false);
        if let Some(sd) = v.sdwa {
            match (sd.dst_sel, sd.dst_unused) {
                (4, 2) => {}
                (5, 2) => high = true,
                (4, 1) => (keep, sext) = (false, true),
                (4, 0) | (6, _) => keep = false,
                (5, 0 | 1) => (high, keep) = (true, false),
                _ => return Err("an SDWA destination of a 16-bit operation"),
            }
        }
        if sext {
            let c = self.cur;
            let k16 = self.s.m.i32(16);
            let x = self.s.m.bin(c, Bin::Shl, x, k16);
            let r = self.s.m.bin(c, Bin::AShr, x, k16);
            self.write_placed(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) if keep => o,
            _ => self.s.m.i32(0),
        };
        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
        let x = self.s.m.bin(c, Bin::And, x, low);
        let r = if high {
            let o = self.s.m.bin(c, Bin::And, old, low);
            let x = self.s.m.bin(c, Bin::Shl, x, k16);
            self.s.m.bin(c, Bin::Or, o, x)
        } else {
            let keep = self.s.m.i32(0xFFFF_0000);
            let o = self.s.m.bin(c, Bin::And, old, keep);
            self.s.m.bin(c, Bin::Or, o, x)
        };
        self.write_placed(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// 16-bit operations, computed in f32 or i32: f16 sums, differences and
    /// products are exact there and round once to f16 (f32 has the 2 x 11 +
    /// 2 bits that make double rounding harmless). Unfused multiply-adds
    /// (`v_mad_f16` / `v_mac_f16`) round the product to f16 before the add;
    /// fused ones (`v_fma_f16` / `v_fmac_f16`) keep a single f32 FMad.
    /// `None` if `name` is none.
    fn valu16(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Option<Result<(), Why>> {
        const FLOAT2: &[&str] = &[
            "v_add_f16",
            "v_sub_f16",
            "v_subrev_f16",
            "v_mul_f16",
            "v_max_f16",
            "v_min_f16",
        ];
        const FLOAT3: &[&str] = &[
            "v_fma_f16",
            "v_mad_f16",
            "v_fmac_f16",
            "v_mac_f16",
            "v_max3_f16",
            "v_min3_f16",
            "v_med3_f16",
        ];
        const FLOAT1: &[&str] = &[
            "v_rcp_f16",
            "v_sqrt_f16",
            "v_rsq_f16",
            "v_log_f16",
            "v_exp_f16",
            "v_fract_f16",
            "v_trunc_f16",
            "v_floor_f16",
            "v_ceil_f16",
            "v_rndne_f16",
            "v_sin_f16",
            "v_cos_f16",
            "v_cvt_f16_u16",
            "v_cvt_f16_i16",
        ];
        const INT: &[&str] = &[
            "v_add_nc_u16",
            "v_sub_nc_u16",
            "v_subrev_nc_u16",
            "v_mul_lo_u16",
            "v_lshlrev_b16",
            "v_lshrrev_b16",
            "v_ashrrev_i16",
            "v_max_u16",
            "v_max_i16",
            "v_min_u16",
            "v_min_i16",
            "v_cvt_u16_f16",
            "v_cvt_i16_f16",
            "v_cvt_norm_i16_f16",
            "v_cvt_norm_u16_f16",
        ];
        let all = [FLOAT1, FLOAT2, FLOAT3, INT];
        if !all.iter().any(|l| l.contains(&name))
            && !matches!(name, "v_cvt_f16_f32" | "v_cvt_f32_f16")
        {
            return None;
        }
        Some(self.valu16_op(inst, name, guard, FLOAT1, FLOAT2, FLOAT3))
    }

    #[allow(clippy::too_many_arguments)]
    fn valu16_op(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
        float1: &[&str],
        float2: &[&str],
        float3: &[&str],
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let i32t = self.s.t.i32;
        let f32t = self.s.t.f32;
        if name == "v_cvt_f32_f16" {
            let f = self.half_fsrc(inst, 0)?;
            return self.write_f(inst, f, guard);
        }
        let float_result = |l: &mut Self, mut f: V| -> Result<(), Why> {
            if v.omod != 0 {
                return Err("an output modifier on a 16-bit operation");
            }
            if v.clamp {
                f = l.unary(7, f);
            }
            let h = l.f16_nearest(f);
            l.write_half(inst, h, guard)
        };
        if name == "v_cvt_f16_f32" {
            let f = self.fsrc(inst, 0)?;
            return float_result(self, f);
        }
        if float2.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let r = match name {
                "v_add_f16" => self.s.m.bin(c, Bin::Add, a, b),
                "v_sub_f16" => self.s.m.bin(c, Bin::Sub, a, b),
                "v_subrev_f16" => self.s.m.bin(c, Bin::Sub, b, a),
                "v_mul_f16" => self.s.m.bin(c, Bin::Mul, a, b),
                "v_max_f16" => self.max_min(true, a, b),
                _ => self.max_min(false, a, b),
            };
            return float_result(self, r);
        }
        if float3.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let acc = if matches!(name, "v_fmac_f16" | "v_mac_f16") {
                // The accumulator is the destination's half that is written.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
                let x = if v.opsel >> 3 & 1 != 0 {
                    let k16 = self.s.m.i32(16);
                    self.s.m.bin(c, Bin::LShr, x, k16)
                } else {
                    x
                };
                let m = self.s.m.i32(0xFFFF);
                let x = self.s.m.bin(c, Bin::And, x, m);
                self.s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
            } else {
                self.half_fsrc(inst, 2)?
            };
            let r = match name {
                // Unfused: round a×b to f16 before adding (overflow of the
                // product must become ±inf, not feed a fused sum).
                "v_mad_f16" | "v_mac_f16" => {
                    let prod = self.s.m.bin(c, Bin::Mul, a, b);
                    let h = self.f16_nearest(prod);
                    let prod = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    self.s.m.bin(c, Bin::Add, prod, acc)
                }
                // shortcut: fused f16 multiply-add as FMad in f32 (the
                // product is exact; the sum then rounds to f16, so midpoints
                // can double-round vs a true binary16 FMA).
                "v_fma_f16" | "v_fmac_f16" => self.fmad(a, b, acc),
                "v_max3_f16" => {
                    let t = self.max_min(true, a, b);
                    self.max_min(true, t, acc)
                }
                "v_min3_f16" => {
                    let t = self.max_min(false, a, b);
                    self.max_min(false, t, acc)
                }
                _ => self.med3(a, b, acc),
            };
            return float_result(self, r);
        }
        if float1.contains(&name) {
            let r = match name {
                "v_cvt_f16_u16" | "v_cvt_f16_i16" => {
                    let mut x = self.half_src_i16(inst, 0)?;
                    let op = if name == "v_cvt_f16_i16" {
                        let k16 = self.s.m.i32(16);
                        let t = self.s.m.bin(c, Bin::Shl, x, k16);
                        x = self.s.m.bin(c, Bin::AShr, t, k16);
                        Cast::SiToFp
                    } else {
                        Cast::UiToFp
                    };
                    self.s.m.cast(c, op, x, f32t)
                }
                "v_rcp_f16" => {
                    let a = self.half_fsrc(inst, 0)?;
                    let one = self.s.m.f32(1.0f32.to_bits());
                    self.s.m.bin(c, Bin::SDiv, one, a)
                }
                _ => {
                    let a = self.half_fsrc(inst, 0)?;
                    let op = match name {
                        "v_sqrt_f16" => 24,
                        "v_rsq_f16" => 25,
                        "v_log_f16" => 23,
                        "v_exp_f16" => 21,
                        "v_fract_f16" => 22,
                        "v_trunc_f16" => 29,
                        "v_floor_f16" => 27,
                        "v_ceil_f16" => 28,
                        "v_rndne_f16" => 26,
                        "v_sin_f16" => 13,
                        _ => 12,
                    };
                    match op {
                        12 | 13 => self.trig_turns(a, op == 12),
                        // Any f16 in (-2^-11, 0) has fract >= 1 - 2^-12; clamp to 0x3bff (1 - 2^-11).
                        22 => self.fract(a, 1.0 - 1.0 / 4096.0, 1.0 - 1.0 / 2048.0),
                        _ => self.unary(op, a),
                    }
                }
            };
            return float_result(self, r);
        }
        if matches!(name, "v_cvt_norm_i16_f16" | "v_cvt_norm_u16_f16") {
            if v.clamp || v.omod != 0 {
                return Err("modifiers on a 16-bit integer operation");
            }
            let r = self.cvt_norm16(inst, name == "v_cvt_norm_i16_f16")?;
            return self.write_half(inst, r, guard);
        }
        // 16-bit integers: computed on the low halves, the low 16 bits kept.
        if v.clamp || v.omod != 0 || v.abs != 0 || v.neg != 0 {
            return Err("modifiers on a 16-bit integer operation");
        }
        let r = match name {
            "v_cvt_u16_f16" | "v_cvt_i16_f16" => {
                // Toward zero, saturating at the 16-bit range (infinities
                // included); NaN gives 0. Clamped in f32 first, where every
                // f16 is exact, so the conversion never leaves i32's range.
                let a = self.half_fsrc(inst, 0)?;
                let (lo, hi) = if name == "v_cvt_u16_f16" {
                    (0.0f32, 65535.0f32)
                } else {
                    (-32768.0, 32767.0)
                };
                let (lo, hi) = (self.s.m.f32(lo.to_bits()), self.s.m.f32(hi.to_bits()));
                let t = self.binary(35, a, lo);
                let t = self.binary(36, t, hi);
                let r = self.s.m.cast(c, Cast::FpToSi, t, i32t);
                let nan = self.s.m.cmp(c, pred::UNO, a, a);
                let z = self.s.m.i32(0);
                self.s.m.select(c, nan, z, r)
            }
            _ => {
                let (mut a, mut b) = (self.half_src_i16(inst, 0)?, self.half_src_i16(inst, 1)?);
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if name.ends_with("_i16") {
                    a = sext(self, a);
                    b = sext(self, b);
                }
                let k15 = self.s.m.i32(15);
                match name {
                    "v_add_nc_u16" => self.s.m.bin(c, Bin::Add, a, b),
                    "v_sub_nc_u16" => self.s.m.bin(c, Bin::Sub, a, b),
                    "v_subrev_nc_u16" => self.s.m.bin(c, Bin::Sub, b, a),
                    "v_mul_lo_u16" => self.s.m.bin(c, Bin::Mul, a, b),
                    "v_lshlrev_b16" | "v_lshrrev_b16" | "v_ashrrev_i16" => {
                        // src0 is the shift (4 bits), src1 the value.
                        let sh = self.s.m.bin(c, Bin::And, a, k15);
                        let x = if name == "v_ashrrev_i16" {
                            sext(self, b)
                        } else {
                            b
                        };
                        let op = match name {
                            "v_lshlrev_b16" => Bin::Shl,
                            "v_lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        self.s.m.bin(c, op, x, sh)
                    }
                    _ => {
                        let p = match name {
                            "v_max_u16" => pred::IUGT,
                            "v_min_u16" => pred::IULT,
                            "v_max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, a, b);
                        self.s.m.select(c, t, a, b)
                    }
                }
            }
        };
        self.write_half(inst, r, guard)
    }

    /// `v_cvt_norm_i16_f16` / `v_cvt_norm_u16_f16`: the f16 source (with its
    /// abs/neg modifiers) clamped to [-1, 1] or [0, 1], times 32767 or 65535,
    /// rounded to nearest even once from the exact product; NaN gives 0.
    /// Worked on the f16's bits in integers, because in f32 the product
    /// (up to 26 significant bits) would round before the rounding to an
    /// integer.
    fn cvt_norm16(&mut self, inst: &Inst, signed: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let mut h = self.half_src_f16(inst, 0)?;
        if v.abs & 1 != 0 {
            let m = k(self, 0x7FFF);
            h = self.s.m.bin(c, Bin::And, h, m);
        }
        if v.neg & 1 != 0 {
            let m = k(self, 0x8000);
            h = self.s.m.bin(c, Bin::Xor, h, m);
        }
        let scale = if signed { 0x7FFF } else { 0xFFFF };
        let (k0, k1, k10) = (k(self, 0), k(self, 1), k(self, 10));
        let mag_mask = k(self, 0x7FFF);
        let mag = self.s.m.bin(c, Bin::And, h, mag_mask);
        let k15 = k(self, 15);
        let neg = self.s.m.bin(c, Bin::LShr, h, k15);
        let neg = self.s.m.cmp(c, pred::NE, neg, k0);
        // |x| < 1: x = m * 2^-s with m < 2^11 and 11 <= s <= 24.
        let exp = self.s.m.bin(c, Bin::LShr, mag, k10);
        let man_mask = k(self, 0x3FF);
        let man = self.s.m.bin(c, Bin::And, mag, man_mask);
        let implicit = k(self, 0x400);
        let normal = self.s.m.bin(c, Bin::Or, man, implicit);
        let sub = self.s.m.cmp(c, pred::EQ, exp, k0);
        let m = self.s.m.select(c, sub, man, normal);
        let k25 = k(self, 25);
        let s_norm = self.s.m.bin(c, Bin::Sub, k25, exp);
        let k24 = k(self, 24);
        let sh = self.s.m.select(c, sub, k24, s_norm);
        let ks = k(self, scale);
        let p = self.s.m.bin(c, Bin::Mul, m, ks);
        let q = self.s.m.bin(c, Bin::LShr, p, sh);
        let one_s = self.s.m.bin(c, Bin::Shl, k1, sh);
        let low_mask = self.s.m.bin(c, Bin::Sub, one_s, k1);
        let rem = self.s.m.bin(c, Bin::And, p, low_mask);
        let half = self.s.m.bin(c, Bin::LShr, one_s, k1);
        let above = self.s.m.cmp(c, pred::IUGT, rem, half);
        let tie = self.s.m.cmp(c, pred::EQ, rem, half);
        let odd = self.s.m.bin(c, Bin::And, q, k1);
        let odd = self.s.m.cmp(c, pred::NE, odd, k0);
        let tie_up = self.s.m.bin(c, Bin::And, tie, odd);
        let up = self.s.m.bin(c, Bin::Or, above, tie_up);
        let r = self.s.m.select(c, up, k1, k0);
        let r = self.s.m.bin(c, Bin::Add, q, r);
        // |x| >= 1 (and infinity) saturates; NaN gives 0.
        let one_h = k(self, 0x3C00);
        let big = self.s.m.cmp(c, pred::IUGE, mag, one_h);
        let r = self.s.m.select(c, big, ks, r);
        let inf = k(self, 0x7C00);
        let nan = self.s.m.cmp(c, pred::IUGT, mag, inf);
        let r = self.s.m.select(c, nan, k0, r);
        let r = if signed {
            let n = self.s.m.bin(c, Bin::Sub, k0, r);
            self.s.m.select(c, neg, n, r)
        } else {
            self.s.m.select(c, neg, k0, r)
        };
        let m16 = k(self, 0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, r, m16))
    }

    /// Half `high` of source `k` of a packed (VOP3P) operation, in the low
    /// 16 bits: op_sel picks the half the low result reads, op_sel_hi the
    /// half the high result reads. Packed integer instructions see the f32
    /// bits of an inline float constant.
    fn packed_src(&mut self, inst: &Inst, k: usize, high: bool, float: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let sel = if high { v.opsel_hi } else { v.opsel } >> k & 1 != 0;
        let x = if float {
            self.f16_operand(inst, &op)?
        } else {
            self.operand(inst, &op)?
        };
        let c = self.cur;
        let x = if sel {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// `v_pk_fmac_f16` (VOP2): each half of the destination plus the
    /// product of the same halves of the sources, f16 through f32 as in
    /// [`Self::packed16`].
    fn packed_fmac(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.sdwa.is_some() || v.dpp.is_some() {
            return Err("v_pk_fmac_f16 with SDWA or DPP");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed result outside a VGPR");
        };
        let c = self.cur;
        let f32t = self.s.t.f32;
        let mut words = Vec::new();
        for k in 0..2 {
            let op = *v.src.get(k).ok_or("a missing operand")?;
            words.push(self.f16_operand(inst, &op)?);
        }
        words.push(self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?);
        let (k16, m) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
        let mut halves = Vec::new();
        for high in [false, true] {
            let mut xs = Vec::new();
            for &w in &words {
                let h = if high {
                    self.s.m.bin(c, Bin::LShr, w, k16)
                } else {
                    w
                };
                let h = self.s.m.bin(c, Bin::And, h, m);
                xs.push(
                    self.s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]),
                );
            }
            // shortcut: FMad in f32, as v_fma_f16.
            let f = self.fmad(xs[0], xs[1], xs[2]);
            let h = self.f16_nearest(f);
            halves.push(self.s.m.bin(c, Bin::And, h, m));
        }
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// Packed 16-bit operations (`v_pk_*`): each half computed as the
    /// 16-bit operation does, f16 through f32 as in [`Self::valu16`].
    fn packed16(&mut self, inst: &Inst, op: &str, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if op == "fmac_f16" {
            return self.packed_fmac(inst, guard);
        }
        let float = op.ends_with("_f16");
        let n = match op {
            "fma_f16" | "mad_u16" | "mad_i16" => 3,
            "add_f16" | "mul_f16" | "min_f16" | "max_f16" | "add_u16" | "sub_u16" | "add_i16"
            | "sub_i16" | "mul_lo_u16" | "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16"
            | "max_u16" | "min_u16" | "max_i16" | "min_i16" => 2,
            _ => return Err("a packed 16-bit operation"),
        };
        if v.omod != 0 || !float && (v.clamp || v.neg != 0 || v.neg_hi != 0) {
            return Err("modifiers on a packed integer operation");
        }
        let c = self.cur;
        let f32t = self.s.t.f32;
        let mut halves = Vec::new();
        for high in [false, true] {
            let mut xs = Vec::new();
            for k in 0..n {
                let h = self.packed_src(inst, k, high, float)?;
                if float {
                    let mut f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    let neg = if high { v.neg_hi } else { v.neg };
                    if neg >> k & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        f = self.s.m.bin(c, Bin::Sub, z, f);
                    }
                    xs.push(f);
                } else {
                    xs.push(h);
                }
            }
            let r = if float {
                let mut f = match op {
                    "add_f16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "mul_f16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "min_f16" => self.max_min(false, xs[0], xs[1]),
                    "max_f16" => self.max_min(true, xs[0], xs[1]),
                    // shortcut: FMad in f32, as v_fma_f16.
                    _ => self.fmad(xs[0], xs[1], xs[2]),
                };
                if v.clamp {
                    f = self.unary(7, f);
                }
                self.f16_nearest(f)
            } else {
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if op.ends_with("_i16") {
                    for x in &mut xs {
                        *x = sext(self, *x);
                    }
                }
                let k15 = self.s.m.i32(15);
                match op {
                    "add_u16" | "add_i16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "sub_u16" | "sub_i16" => self.s.m.bin(c, Bin::Sub, xs[0], xs[1]),
                    "mul_lo_u16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "mad_u16" | "mad_i16" => {
                        let m = self.s.m.bin(c, Bin::Mul, xs[0], xs[1]);
                        self.s.m.bin(c, Bin::Add, m, xs[2])
                    }
                    "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16" => {
                        let sh = self.s.m.bin(c, Bin::And, xs[0], k15);
                        let o = match op {
                            "lshlrev_b16" => Bin::Shl,
                            "lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        let x = if op == "ashrrev_i16" {
                            sext(self, xs[1])
                        } else {
                            xs[1]
                        };
                        self.s.m.bin(c, o, x, sh)
                    }
                    _ => {
                        let p = match op {
                            "max_u16" => pred::IUGT,
                            "min_u16" => pred::IULT,
                            "max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, xs[0], xs[1]);
                        self.s.m.select(c, t, xs[0], xs[1])
                    }
                }
            };
            let m = self.s.m.i32(0xFFFF);
            halves.push(self.s.m.bin(c, Bin::And, r, m));
        }
        let k16 = self.s.m.i32(16);
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed result outside a VGPR");
        };
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    fn integer(&mut self, inst: &Inst, name: &'static str) -> Result<V, Why> {
        let c = self.cur;
        let s = |l: &mut Self, k| l.src(inst, k);
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let m31 = |l: &mut Self, x: V| {
            let k31 = l.s.m.i32(31);
            l.s.m.bin(c, Bin::And, x, k31)
        };
        Ok(match name {
            "v_and_b32" | "v_or_b32" | "v_xor_b32" | "v_mul_lo_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (op, a, b) = match name {
                    "v_and_b32" => (Bin::And, a, b),
                    "v_or_b32" => (Bin::Or, a, b),
                    "v_xor_b32" => (Bin::Xor, a, b),
                    _ => (Bin::Mul, a, b),
                };
                self.s.m.bin(c, op, a, b)
            }
            "v_add_nc_u32" | "v_sub_nc_u32" | "v_subrev_nc_u32" => {
                let (mut a, mut b) = (s(self, 0)?, s(self, 1)?);
                if name == "v_subrev_nc_u32" {
                    std::mem::swap(&mut a, &mut b);
                }
                let add = name == "v_add_nc_u32";
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                if v.clamp {
                    let (r, carry) = self.add_sub_carry(c, add, a, b, None);
                    self.clamp_u32(c, add, r, carry)
                } else {
                    let op = if add { Bin::Add } else { Bin::Sub };
                    self.s.m.bin(c, op, a, b)
                }
            }
            "v_xnor_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let x = self.s.m.bin(c, Bin::Xor, a, b);
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, x, ones)
            }
            "v_not_b32" => {
                let a = s(self, 0)?;
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, a, ones)
            }
            "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                let (sh, a) = (s(self, 0)?, s(self, 1)?);
                let sh = m31(self, sh);
                let op = match name {
                    "v_lshlrev_b32" => Bin::Shl,
                    "v_lshrrev_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                self.s.m.bin(c, op, a, sh)
            }
            "v_lshl_add_u32" | "v_lshl_or_b32" | "v_add_lshl_u32" | "v_and_or_b32"
            | "v_or3_b32" | "v_xad_u32" | "v_add3_u32" | "v_xor3_b32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                match name {
                    "v_lshl_add_u32" | "v_lshl_or_b32" => {
                        let b = m31(self, b);
                        let t = self.s.m.bin(c, Bin::Shl, a, b);
                        let op = if name == "v_lshl_add_u32" {
                            Bin::Add
                        } else {
                            Bin::Or
                        };
                        self.s.m.bin(c, op, t, d)
                    }
                    "v_add_lshl_u32" => {
                        let t = self.s.m.bin(c, Bin::Add, a, b);
                        let d = m31(self, d);
                        self.s.m.bin(c, Bin::Shl, t, d)
                    }
                    _ => {
                        let (o1, o2) = match name {
                            "v_and_or_b32" => (Bin::And, Bin::Or),
                            "v_or3_b32" => (Bin::Or, Bin::Or),
                            "v_xad_u32" => (Bin::Xor, Bin::Add),
                            "v_add3_u32" => (Bin::Add, Bin::Add),
                            _ => (Bin::Xor, Bin::Xor),
                        };
                        let t = self.s.m.bin(c, o1, a, b);
                        self.s.m.bin(c, o2, t, d)
                    }
                }
            }
            "v_mul_u32_u24" | "v_mul_i32_i24" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mul_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                self.s.m.bin(c, Bin::Mul, a, b)
            }
            "v_mul_hi_u32_u24" | "v_mul_hi_i32_i24" => {
                // The high dword of the same 48-bit product: the low 24 bits
                // of each operand, zero- or sign-extended, multiplied.
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let signed = name == "v_mul_hi_i32_i24";
                let (sh, op) = (k(self, 8), if signed { Bin::AShr } else { Bin::LShr });
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                self.mul_hi(a, b, signed)
            }
            "v_bfe_u32" | "v_bfe_i32" => {
                let (a, off, w) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let w = m31(self, w);
                self.bfe(a, off, w, name == "v_bfe_i32")
            }
            "v_bfi_b32" => {
                let (m, a, b) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let ones = k(self, u32::MAX);
                let nm = self.s.m.bin(c, Bin::Xor, m, ones);
                let x = self.s.m.bin(c, Bin::And, m, a);
                let y = self.s.m.bin(c, Bin::And, nm, b);
                self.s.m.bin(c, Bin::Or, x, y)
            }
            "v_max_i32" | "v_min_i32" | "v_max_u32" | "v_min_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let p = match name {
                    "v_max_i32" => pred::SGT,
                    "v_min_i32" => pred::SLT,
                    "v_max_u32" => pred::IUGT,
                    _ => pred::IULT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                self.s.m.select(c, t, a, b)
            }
            "v_mul_hi_u32" | "v_mul_hi_i32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                self.mul_hi(a, b, name == "v_mul_hi_i32")
            }
            "v_alignbit_b32" | "v_alignbyte_b32" => {
                // The low dword of {a, b} >> shift.
                let (a, b, sh) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let sh = if name == "v_alignbyte_b32" {
                    let three = k(self, 3);
                    let t = self.s.m.bin(c, Bin::And, sh, three);
                    let k3 = k(self, 3);
                    self.s.m.bin(c, Bin::Shl, t, k3)
                } else {
                    m31(self, sh)
                };
                let k32 = k(self, 32);
                let lo = self.s.m.bin(c, Bin::LShr, b, sh);
                let left = self.s.m.bin(c, Bin::Sub, k32, sh);
                let left = m31(self, left);
                let hi = self.s.m.bin(c, Bin::Shl, a, left);
                let zero = k(self, 0);
                let none = self.s.m.cmp(c, pred::EQ, sh, zero);
                let hi = self.s.m.select(c, none, zero, hi);
                self.s.m.bin(c, Bin::Or, lo, hi)
            }
            "v_bfrev_b32" => {
                let a = s(self, 0)?;
                self.s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a])
            }
            "v_bcnt_u32_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[a]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            "v_dot4c_i32_i8" | "v_dot4_i32_i8" | "v_dot4_u32_u8" => {
                // The sum of the four byte products of src0 and src1 (signed
                // or not) plus the accumulator: the destination's old value
                // for the VOP2 form, src2 for the VOP3P ones.
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                if v.sdwa.is_some() || v.dpp.is_some() {
                    return Err("a dot product with SDWA or DPP");
                }
                if v.clamp {
                    return Err("a clamped dot product");
                }
                // shortcut: OPSEL and OPSEL_HI select 16-bit halves, which a
                // dot product of whole dwords has none of; the console
                // compiler sets them to a pattern (opsel 2, opsel_hi 5) that
                // the hardware must ignore. Only NEG is refused.
                if v.neg != 0 || v.neg_hi != 0 {
                    return Err("a dot product with a modifier");
                }
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut sum = if name == "v_dot4c_i32_i8" {
                    let Some(Operand::Vector { reg, .. }) = v.dst else {
                        return Err("an accumulator");
                    };
                    let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
                    self.i32_of(acc)?
                } else {
                    s(self, 2)?
                };
                let signed = name != "v_dot4_u32_u8";
                let k8 = self.s.m.i32(8);
                let (mut x, mut y) = (a, b);
                for _ in 0..4 {
                    let (px, py) = if signed {
                        let up = self.s.m.i32(24);
                        let lx = self.s.m.bin(c, Bin::Shl, x, up);
                        let ly = self.s.m.bin(c, Bin::Shl, y, up);
                        (
                            self.s.m.bin(c, Bin::AShr, lx, up),
                            self.s.m.bin(c, Bin::AShr, ly, up),
                        )
                    } else {
                        let mask = self.s.m.i32(0xFF);
                        (
                            self.s.m.bin(c, Bin::And, x, mask),
                            self.s.m.bin(c, Bin::And, y, mask),
                        )
                    };
                    let m = self.s.m.bin(c, Bin::Mul, px, py);
                    sum = self.s.m.bin(c, Bin::Add, sum, m);
                    x = self.s.m.bin(c, Bin::LShr, x, k8);
                    y = self.s.m.bin(c, Bin::LShr, y, k8);
                }
                sum
            }
            "v_mad_u32_u24" | "v_mad_i32_i24" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mad_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                let m = self.s.m.bin(c, Bin::Mul, a, b);
                self.s.m.bin(c, Bin::Add, m, d)
            }
            "v_sad_u8" | "v_sad_hi_u8" | "v_sad_u16" | "v_sad_u32" | "v_msad_u8" => {
                // src2 plus the sum of the unsigned differences |a - b| of
                // the operands' bytes, halves or whole dwords, wrapping at 32
                // bits. v_sad_hi_u8 adds the byte sum shifted to the high half;
                // v_msad_u8 leaves out a byte whose mask byte (src1) is 0.
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let (width, parts) = match name {
                    "v_sad_u16" => (16, 2),
                    "v_sad_u32" => (32, 1),
                    _ => (8, 4),
                };
                let zero = k(self, 0);
                let mut sum = zero;
                for i in 0..parts {
                    let (mut x, mut y) = (a, b);
                    if width < 32 {
                        let (sh, mask) = (k(self, width * i), k(self, (1 << width) - 1));
                        x = self.s.m.bin(c, Bin::LShr, x, sh);
                        x = self.s.m.bin(c, Bin::And, x, mask);
                        y = self.s.m.bin(c, Bin::LShr, y, sh);
                        y = self.s.m.bin(c, Bin::And, y, mask);
                    }
                    let above = self.s.m.cmp(c, pred::IUGT, x, y);
                    let hi = self.s.m.select(c, above, x, y);
                    let lo = self.s.m.select(c, above, y, x);
                    let mut diff = self.s.m.bin(c, Bin::Sub, hi, lo);
                    if name == "v_msad_u8" {
                        let masked = self.s.m.cmp(c, pred::NE, y, zero);
                        diff = self.s.m.select(c, masked, diff, zero);
                    }
                    sum = self.s.m.bin(c, Bin::Add, sum, diff);
                }
                if name == "v_sad_hi_u8" {
                    let k16 = k(self, 16);
                    sum = self.s.m.bin(c, Bin::Shl, sum, k16);
                }
                self.s.m.bin(c, Bin::Add, sum, d)
            }
            // VOP3P integer dot products: src0 and src1 hold packed fields of
            // 16, 8 or 4 bits, src2 is a 32-bit accumulator, and the sum of the
            // field products and the accumulator wraps at 32 bits (RDNA2 ISA).
            // shortcut: only the form assemblers write (op_sel 0, op_sel_hi
            // all set, no neg, register sources); `clamp` is refused by
            // `write_i`.
            // (The byte forms are above, with the console's OPSEL pattern.)
            "v_dot2_i32_i16" | "v_dot2_u32_u16" | "v_dot8_i32_i4" | "v_dot8_u32_u4" => {
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                if v.opsel != 0 || v.opsel_hi != 7 || v.neg_hi != 0 {
                    return Err("a dot product's operand modifiers");
                }
                if v.src
                    .iter()
                    .take(2)
                    .any(|o| !matches!(o, Operand::Vector { .. } | Operand::Scalar { .. }))
                {
                    return Err("a dot product of constants");
                }
                let width: u32 = match name {
                    "v_dot2_i32_i16" | "v_dot2_u32_u16" => 16,
                    _ => 4,
                };
                let signed = name.contains("_i32_");
                let (a, b, mut sum) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                for n in 0..32 / width {
                    let field = |l: &mut Self, x: V| {
                        if signed {
                            // Up to the top, then back with its sign.
                            let up = k(l, 32 - width * (n + 1));
                            let down = k(l, 32 - width);
                            let t = l.s.m.bin(c, Bin::Shl, x, up);
                            l.s.m.bin(c, Bin::AShr, t, down)
                        } else {
                            let (sh, mask) = (k(l, width * n), k(l, (1 << width) - 1));
                            let t = l.s.m.bin(c, Bin::LShr, x, sh);
                            l.s.m.bin(c, Bin::And, t, mask)
                        }
                    };
                    let (x, y) = (field(self, a), field(self, b));
                    let p = self.s.m.bin(c, Bin::Mul, x, y);
                    sum = self.s.m.bin(c, Bin::Add, sum, p);
                }
                sum
            }
            // 16×16→32 multiply-add: op_sel[0]/[1] pick high/low halves of
            // src0/src1; src2 is a full 32-bit addend. Clamp stays refused.
            "v_mad_u32_u16" | "v_mad_i32_i16" => {
                let (mut a, mut b, d) = (
                    self.half_src_i16(inst, 0)?,
                    self.half_src_i16(inst, 1)?,
                    s(self, 2)?,
                );
                if name == "v_mad_i32_i16" {
                    let sh = k(self, 16);
                    let sext = |l: &mut Self, x: V| {
                        let t = l.s.m.bin(c, Bin::Shl, x, sh);
                        l.s.m.bin(c, Bin::AShr, t, sh)
                    };
                    a = sext(self, a);
                    b = sext(self, b);
                }
                let m = self.s.m.bin(c, Bin::Mul, a, b);
                self.s.m.bin(c, Bin::Add, m, d)
            }
            // Packed 4×8-bit unsigned average; per-byte rounding bit from src2:
            // dst.u8[i] = (src0.u8[i] + src1.u8[i] + (src2.u8[i] & 1)) >> 1.
            "v_lerp_u8" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let (mff, one) = (k(self, 0xFF), k(self, 1));
                let mut out = k(self, 0);
                for i in 0..4u32 {
                    let (ai, bi, di) = if i == 0 {
                        (a, b, d)
                    } else {
                        let sh = k(self, i * 8);
                        (
                            self.s.m.bin(c, Bin::LShr, a, sh),
                            self.s.m.bin(c, Bin::LShr, b, sh),
                            self.s.m.bin(c, Bin::LShr, d, sh),
                        )
                    };
                    let ai = self.s.m.bin(c, Bin::And, ai, mff);
                    let bi = self.s.m.bin(c, Bin::And, bi, mff);
                    let di = self.s.m.bin(c, Bin::And, di, one);
                    let sum = self.s.m.bin(c, Bin::Add, ai, bi);
                    let sum = self.s.m.bin(c, Bin::Add, sum, di);
                    let avg = self.s.m.bin(c, Bin::LShr, sum, one);
                    let byte = if i == 0 {
                        avg
                    } else {
                        let sh = k(self, i * 8);
                        self.s.m.bin(c, Bin::Shl, avg, sh)
                    };
                    out = if i == 0 {
                        byte
                    } else {
                        self.s.m.bin(c, Bin::Or, out, byte)
                    };
                }
                out
            }
            "v_min3_i32" | "v_max3_i32" | "v_min3_u32" | "v_max3_u32" | "v_med3_i32"
            | "v_med3_u32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let signed = name.ends_with("i32");
                let (lt, gt) = if signed {
                    (pred::SLT, pred::SGT)
                } else {
                    (pred::IULT, pred::IUGT)
                };
                let pick = |l: &mut Self, p, x, y| {
                    let t = l.s.m.cmp(c, p, x, y);
                    l.s.m.select(c, t, x, y)
                };
                match &name[2..6] {
                    "min3" => {
                        let t = pick(self, lt, a, b);
                        pick(self, lt, t, d)
                    }
                    "max3" => {
                        let t = pick(self, gt, a, b);
                        pick(self, gt, t, d)
                    }
                    _ => {
                        let lo = pick(self, lt, a, b);
                        let hi = pick(self, gt, a, b);
                        let t = pick(self, lt, hi, d);
                        pick(self, gt, lo, t)
                    }
                }
            }
            "v_perm_b32" => {
                // Byte i of the result is byte S of {src0, src1} (src1 the
                // low dword) for selector byte S below 8; 8-11 repeat the
                // sign of bytes 1 and 3 of src1, then of src0; 12 is 0x00,
                // higher 0xFF.
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                let sel = match v.src.get(2) {
                    Some(&Operand::Literal(x)) => x,
                    Some(&Operand::Const(k)) => crate::resource::inline(k).ok_or(name)?,
                    _ => {
                        // A computed selector: each byte picked at run time.
                        let (a, b, sv) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                        let mut r = k(self, 0);
                        for i in 0..4u32 {
                            let (sh, ff) = (k(self, 8 * i), k(self, 0xFF));
                            let sel = self.s.m.bin(c, Bin::LShr, sv, sh);
                            let sel = self.s.m.bin(c, Bin::And, sel, ff);
                            // Select a dword and a byte within it.
                            // Later selects handle codes 8 and above.
                            let four = k(self, 4);
                            let zero = k(self, 0);
                            let high = self.s.m.bin(c, Bin::And, sel, four);
                            let high = self.s.m.cmp(c, pred::NE, high, zero);
                            let word = self.s.m.select(c, high, a, b);
                            let three = k(self, 3);
                            let index = self.s.m.bin(c, Bin::And, sel, three);
                            let bits = self.s.m.bin(c, Bin::Shl, index, three);
                            let x = self.s.m.bin(c, Bin::LShr, word, bits);
                            let mut byte = self.s.m.bin(c, Bin::And, x, ff);
                            // 8-11: the sign of bytes 1, 3 of src1, then of src0.
                            for (n, src, bit) in [(8, b, 15), (9, b, 31), (10, a, 15), (11, a, 31)]
                            {
                                let left = k(self, 31 - bit);
                                let t = self.s.m.bin(c, Bin::Shl, src, left);
                                let k31 = k(self, 31);
                                let t = self.s.m.bin(c, Bin::AShr, t, k31);
                                let t = self.s.m.bin(c, Bin::And, t, ff);
                                let kn = k(self, n);
                                let is = self.s.m.cmp(c, pred::EQ, sel, kn);
                                byte = self.s.m.select(c, is, t, byte);
                            }
                            let (k12, z) = (k(self, 12), k(self, 0));
                            let is12 = self.s.m.cmp(c, pred::EQ, sel, k12);
                            byte = self.s.m.select(c, is12, z, byte);
                            let above = self.s.m.cmp(c, pred::IUGT, sel, k12);
                            byte = self.s.m.select(c, above, ff, byte);
                            let at = k(self, 8 * i);
                            let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                            r = self.s.m.bin(c, Bin::Or, r, byte);
                        }
                        return Ok(r);
                    }
                };
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut r = k(self, 0);
                for i in 0..4 {
                    let pick = sel >> (8 * i) & 0xFF;
                    let byte = match pick {
                        0..=7 => {
                            let (src, n) = if pick < 4 { (b, pick) } else { (a, pick - 4) };
                            let sh = k(self, 8 * n);
                            let t = self.s.m.bin(c, Bin::LShr, src, sh);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        8..=11 => {
                            let src = if pick < 10 { b } else { a };
                            let bit = if pick % 2 == 0 { 15 } else { 31 };
                            let left = k(self, 31 - bit);
                            let t = self.s.m.bin(c, Bin::Shl, src, left);
                            let k31 = k(self, 31);
                            let t = self.s.m.bin(c, Bin::AShr, t, k31);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        12 => continue,
                        _ => k(self, 0xFF),
                    };
                    let at = k(self, 8 * i);
                    let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                    r = self.s.m.bin(c, Bin::Or, r, byte);
                }
                r
            }
            "v_bfm_b32" => {
                // ((1 << src0[4:0]) - 1) << src1[4:0]
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let one = k(self, 1);
                let (a, b) = (m31(self, a), m31(self, b));
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                self.s.m.bin(c, Bin::Shl, m, b)
            }
            "v_cvt_pk_u8_f32" => {
                // src0 rounded to nearest even and saturated to a byte, into
                // byte src1[1:0] of src2.
                let f = self.fsrc(inst, 0)?;
                let (b, d) = (s(self, 1)?, s(self, 2)?);
                let f = self.unary(26, f);
                let (zero, max) = (self.s.m.f32(0), self.s.m.f32(255.0f32.to_bits()));
                let f = self.binary(35, f, zero);
                let f = self.binary(36, f, max);
                let x = self.s.m.cast(c, Cast::FpToUi, f, self.s.t.i32);
                let (three, k3) = (k(self, 3), k(self, 3));
                let sh = self.s.m.bin(c, Bin::And, b, three);
                let sh = self.s.m.bin(c, Bin::Shl, sh, k3);
                let x = self.s.m.bin(c, Bin::Shl, x, sh);
                let ff = k(self, 0xFF);
                let hole = self.s.m.bin(c, Bin::Shl, ff, sh);
                let ones = k(self, u32::MAX);
                let hole = self.s.m.bin(c, Bin::Xor, hole, ones);
                let d = self.s.m.bin(c, Bin::And, d, hole);
                self.s.m.bin(c, Bin::Or, d, x)
            }
            "v_ffbl_b32" => {
                // -1 for zero, as DXIL's first-bit-low.
                let a = s(self, 0)?;
                self.s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 32, &[a])
            }
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                // Bits of the mask's low (high) half below this lane, plus src1:
                // with an all-ones mask, the lane index.
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                // Mesh lanes are the console's by construction; pixel and
                // vertex lanes are the host's, as their masks are.
                if self.mask_data == MaskData::Host {
                    // Host lanes: the mask must be one (a ballot) or all lanes.
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    let host = match v.src.first() {
                        Some(&Operand::Const(193)) => true,
                        Some(&Operand::Scalar { code, .. }) => {
                            inst.read(Reg::S(code)).is_some_and(|m| {
                                self.x.st.masks.contains_key(&m) || self.x.st.highs.contains_key(&m)
                            })
                        }
                        _ => false,
                    };
                    if !host {
                        return Err("v_mbcnt of a mask that is no ballot (console lane layout)");
                    }
                    self.s.flags |= flags::WAVE_OPS;
                } else if self.s.mesh.is_none() {
                    self.cross_lane(false)?;
                }
                let lane = if self.console_cf {
                    self.console_lane().0
                } else {
                    self.lane_index()
                };
                let below = if name == "v_mbcnt_lo_u32_b32" {
                    lane
                } else {
                    let (k32, z) = (k(self, 32), k(self, 0));
                    let t = self.s.m.bin(c, Bin::Sub, lane, k32);
                    let low = self.s.m.cmp(c, pred::IULT, lane, k32);
                    self.s.m.select(c, low, z, t)
                };
                // (1 << below) - 1, all ones from 32 on.
                let (one, k32, ones) = (k(self, 1), k(self, 32), k(self, u32::MAX));
                let sh = m31(self, below);
                let m = self.s.m.bin(c, Bin::Shl, one, sh);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                let full = self.s.m.cmp(c, pred::IUGE, below, k32);
                let m = self.s.m.select(c, full, ones, m);
                let x = self.s.m.bin(c, Bin::And, a, m);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[x]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            _ => return Err(name),
        })
    }

    fn compare(&mut self, inst: &Inst, rest: &str) -> Result<(), Why> {
        let (op, ty) = rest.rsplit_once('_').ok_or("a compare")?;
        let c = self.cur;
        let r = match ty {
            "f32" | "f16" if op == "class" => {
                // Bit k of src1 tests class k: signalling NaN, quiet NaN,
                // -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf.
                let f16 = ty == "f16";
                let x = if f16 {
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    let mut h = self.half_src_f16(inst, 0)?;
                    if v.abs & 1 != 0 {
                        let m = self.s.m.i32(0x7FFF);
                        h = self.s.m.bin(c, Bin::And, h, m);
                    }
                    if v.neg & 1 != 0 {
                        let s = self.s.m.i32(0x8000);
                        h = self.s.m.bin(c, Bin::Xor, h, s);
                    }
                    h
                } else {
                    // abs/neg only touch the sign bit: float ops would flush
                    // denormals and quiet signalling NaNs before the test.
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    let (abs, neg) = (v.abs & 1 != 0, v.neg & 1 != 0);
                    let mut x = self.raw_src(inst, 0)?;
                    if abs {
                        let m = self.s.m.i32(0x7FFF_FFFF);
                        x = self.s.m.bin(c, Bin::And, x, m);
                    }
                    if neg {
                        let s = self.s.m.i32(0x8000_0000);
                        x = self.s.m.bin(c, Bin::Xor, x, s);
                    }
                    x
                };
                let mask = self.src(inst, 1)?;
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (exp_sh, sign_sh, exp_mask, mant_mask, quiet_bit) = if f16 {
                    (10, 15, 0x1F, 0x03FF, 0x0200)
                } else {
                    (23, 31, 0xFF, 0x7F_FFFF, 0x40_0000)
                };
                let (k_exp, k_sign, k_emax, mant, quiet) = (
                    k(self, exp_sh),
                    k(self, sign_sh),
                    k(self, exp_mask),
                    k(self, mant_mask),
                    k(self, quiet_bit),
                );
                let e = self.s.m.bin(c, Bin::LShr, x, k_exp);
                let e = self.s.m.bin(c, Bin::And, e, k_emax);
                let m = self.s.m.bin(c, Bin::And, x, mant);
                let neg = self.s.m.bin(c, Bin::LShr, x, k_sign);
                let zero = k(self, 0);
                let (e0, emax) = (
                    self.s.m.cmp(c, pred::EQ, e, zero),
                    self.s.m.cmp(c, pred::EQ, e, k_emax),
                );
                let m0 = self.s.m.cmp(c, pred::EQ, m, zero);
                let q = self.s.m.bin(c, Bin::And, x, quiet);
                let q = self.s.m.cmp(c, pred::NE, q, zero);
                // Positive classes 6-9; negative ones are 11 minus those.
                let (normal, inf, zero_c, denorm) =
                    (k(self, 8), k(self, 9), k(self, 6), k(self, 7));
                let t = self.s.m.select(c, m0, zero_c, denorm);
                let t = self.s.m.select(c, e0, t, normal);
                let t = self.s.m.select(c, emax, inf, t);
                let eleven = k(self, 11);
                let mirrored = self.s.m.bin(c, Bin::Sub, eleven, t);
                let is_neg = self.s.m.cmp(c, pred::NE, neg, zero);
                let t = self.s.m.select(c, is_neg, mirrored, t);
                let (one, snan) = (k(self, 1), k(self, 0));
                let nan_class = self.s.m.select(c, q, one, snan);
                let yes = self.s.m.bool(true);
                let nan = self.s.m.bin(c, Bin::Xor, m0, yes);
                let nan = self.s.m.bin(c, Bin::And, nan, emax);
                let t = self.s.m.select(c, nan, nan_class, t);
                let bit = self.s.m.bin(c, Bin::LShr, mask, t);
                let bit = self.s.m.bin(c, Bin::And, bit, one);
                self.s.m.cmp(c, pred::NE, bit, zero)
            }
            "f32" | "f16" => {
                // False and true whatever the operands, NaNs included. As
                // constants, like the integer forms: `fcmp false`/`fcmp true`
                // (which DXC never emits) remove the WARP device at dispatch.
                if matches!(op, "f" | "tru") {
                    self.lane(inst, Some(op == "tru"));
                    return Ok(());
                }
                let (a, b) = if ty == "f16" {
                    (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?)
                } else {
                    (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?)
                };
                if op == "lg" {
                    // Ordered and not equal, false for a NaN, as `olt | ogt`.
                    // `fcmp one` means the same, but the AMD D3D12 driver
                    // runs it as unordered (true for a NaN) where NVIDIA
                    // and WARP do not; olt and ogt agree on all three.
                    let lt = self.s.m.cmp(c, pred::OLT, a, b);
                    let gt = self.s.m.cmp(c, pred::OGT, a, b);
                    self.s.m.bin(c, Bin::Or, lt, gt)
                } else {
                    let p = match op {
                        "lt" => pred::OLT,
                        "eq" => pred::OEQ,
                        "le" => pred::OLE,
                        "gt" => pred::OGT,
                        "ge" => pred::OGE,
                        "o" => pred::ORD,
                        "u" => pred::UNO,
                        "nge" => pred::ULT,
                        "nlg" => pred::UEQ,
                        "ngt" => pred::ULE,
                        "nle" => pred::UGT,
                        "neq" => pred::UNE,
                        "nlt" => pred::UGE,
                        _ => return Err(inst.name()),
                    };
                    self.s.m.cmp(c, p, a, b)
                }
            }
            "u64" | "i64" => {
                if matches!(op, "f" | "t") {
                    self.lane(inst, Some(op == "t"));
                    return Ok(());
                }
                let (a, b) = (self.src64(inst, 0)?, self.src64(inst, 1)?);
                let signed = ty == "i64";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    _ => return Err("a 64-bit compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            "u32" | "i32" | "u16" | "i16" => {
                let (a, b) = if ty.ends_with("16") {
                    let (mut a, mut b) = (self.half_src_i16(inst, 0)?, self.half_src_i16(inst, 1)?);
                    if ty == "i16" {
                        let k16 = self.s.m.i32(16);
                        for x in [&mut a, &mut b] {
                            let t = self.s.m.bin(c, Bin::Shl, *x, k16);
                            *x = self.s.m.bin(c, Bin::AShr, t, k16);
                        }
                    }
                    (a, b)
                } else {
                    (self.src(inst, 0)?, self.src(inst, 1)?)
                };
                let signed = ty == "i32" || ty == "i16";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    "f" | "t" => {
                        self.lane(inst, Some(op == "t"));
                        return Ok(());
                    }
                    _ => return Err("an integer compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            _ => return Err(inst.name()),
        };
        self.lane_bits_for(inst, r);
        Ok(())
    }

    fn lane(&mut self, inst: &Inst, value: Option<bool>) {
        let b = self.s.m.bool(value == Some(true));
        self.lane_bits_for(inst, b);
    }

    fn lane_bits_for(&mut self, inst: &Inst, r: V) {
        let Fields::Valu(v) = &inst.fields else {
            return;
        };
        let code = if inst.name().starts_with("v_cmpx") {
            Some(EXEC)
        } else {
            match v.dst {
                Some(Operand::Scalar { code, .. }) => Some(code),
                _ => None,
            }
        };
        if let Some(lo) = code.and_then(|c| inst.write(Reg::S(c))) {
            self.lane_bits.insert(lo, r);
            if let (Some(_), Some(c)) = (&self.split, code) {
                self.split_lanes_written(inst, c, r);
            }
        }
        for &(_, w) in &inst.writes {
            self.vals
                .entry(w)
                .or_insert(Val::Missing("a lane mask read as data"));
        }
    }

    fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Some(&interp) = self.interps.get(&site) else {
            if inst.name().starts_with("v_interp_p1") {
                let dst = match &inst.fields {
                    Fields::Vintrp { dst, .. } => *dst,
                    Fields::Valu(v) => v.dst.ok_or("an interpolation")?,
                    _ => unreachable!(),
                };
                // The second half (`v_interp_p2*`) evaluates the whole
                // attribute; it reads this register only as its accumulator.
                // Any other reader, or a join, would take the zero as data.
                if let Operand::Vector { reg, .. } = dst
                    && let Some(w) = inst.write(Reg::V(reg))
                {
                    if self.read_past_interpolation(w) {
                        return Err("the first half of an interpolation");
                    }
                    let z = self.s.m.i32(0);
                    self.write(inst, Reg::V(reg), z, None);
                }
                return Ok(());
            }
            return Err("the first half of an interpolation");
        };
        if let Interp::Computed { attribute, second } = interp {
            return self.computed_interp(inst, *attribute, *second);
        }
        let kind = match interp {
            Interp::Interpolated { barycentric, .. } => Some(*barycentric),
            _ => None,
        };
        let (attribute, mode) = match interp {
            Interp::Interpolated {
                attribute,
                barycentric,
                ..
            } => (
                attribute,
                match barycentric {
                    Barycentric::PerspectiveCenter => 2,
                    Barycentric::PerspectiveCentroid => 3,
                    Barycentric::PerspectiveSample => 6,
                    Barycentric::LinearCenter => 4,
                    Barycentric::LinearCentroid => 5,
                    Barycentric::LinearSample => 7,
                    Barycentric::PerspectivePullModel => return Err("pull-model interpolation"),
                },
            ),
            // P0, the provoking vertex: a flat (nointerpolation) input.
            Interp::Vertex {
                attribute,
                vertex: 2,
            } => (attribute, 1),
            Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
            Interp::Computed { .. } => unreachable!(),
        };
        let dst = match &inst.fields {
            Fields::Vintrp { dst, .. } => *dst,
            Fields::Valu(v) => v.dst.ok_or("an interpolation")?,
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else {
            return Err("an interpolation");
        };
        // A channel of an input element, interpolated as `mode` says.
        let channel = |l: &mut Self, id: usize, col: u8| -> Result<V, Why> {
            let e = &mut l.s.inputs[id];
            if e.interp != mode && e.usage != 0 {
                return Err("one input interpolated two ways");
            }
            if mode != 1 && e.comp != 9 {
                return Err("interpolation of an integer input");
            }
            e.interp = mode;
            Ok(l.load_input_bits(id, col))
        };
        let packing = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => inputs
                .iter()
                .find(|i| i.row == attribute.row && i.fp16)
                .map(|i| (i.pack2, (i.row2 != i.row).then_some(i.row2))),
            _ => None,
        };
        let f16 = inst.name().ends_with("_f16");
        let x = match packing {
            None if f16 || attribute.high => return Err("a 16-bit read of a 32-bit input"),
            None => {
                let (id, col) = self
                    .find_input(attribute.row, attribute.channel)
                    .ok_or("an input the signature lacks")?;
                match kind {
                    Some(kind) if self.per_vertex(id) => {
                        self.interpolate_vertices(id, col, kind)?
                    }
                    _ => channel(self, id, col)?,
                }
            }
            Some((pack2, b)) => {
                if !self
                    .s
                    .inputs
                    .iter()
                    .any(|e| e.row == Some(u32::from(attribute.row)))
                {
                    return Err("an input the signature lacks");
                }
                let half = |l: &mut Self, h: u8| -> Result<V, Why> {
                    let src = if pack2 {
                        packed_half(&l.s.inputs, attribute.row, b, h)
                    } else {
                        h.is_multiple_of(2).then_some((attribute.row, h / 2))
                    };
                    let Some((id, col)) = src.and_then(|(row, ch)| l.find_input(row, ch)) else {
                        // An unwritten half has no host input channel. Give it
                        // the same zero as an unwritten register.
                        return Ok(l.s.m.i32(0));
                    };
                    let x = channel(l, id, col)?;
                    Ok(l.input_half(id, x))
                };
                let h = 2 * attribute.channel;
                let c = self.cur;
                if f16 {
                    // VOP3: omod/clamp on the f32 value, then narrow; op_sel[3]
                    // selects the destination half via write_half (GFX10).
                    let half_idx = h + u8::from(attribute.high);
                    let src = if pack2 {
                        packed_half(&self.s.inputs, attribute.row, b, half_idx)
                    } else {
                        half_idx
                            .is_multiple_of(2)
                            .then_some((attribute.row, half_idx / 2))
                    };
                    let mut f = match src.and_then(|(row, ch)| self.find_input(row, ch)) {
                        Some((id, col)) => {
                            let x = channel(self, id, col)?;
                            self.as_float(x)
                        }
                        None => self.s.m.f32(0),
                    };
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!("v_interp_*_f16 is VOP3")
                    };
                    let (omod, clamp) = (v.omod, v.clamp);
                    match omod {
                        0 => {}
                        k => {
                            let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                            let c = self.s.m.f32(factor.to_bits());
                            f = self.s.m.bin(self.cur, Bin::Mul, f, c);
                        }
                    }
                    if clamp {
                        f = self.unary(7, f);
                    }
                    let h16 = self.f16_nearest(f);
                    return self.write_half(inst, h16, None);
                } else if mode == 1 {
                    // The whole dword of a flat input: both halves.
                    let lo = half(self, h)?;
                    let hi = half(self, h + 1)?;
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        // A pixel interpolates in every lane; guards only hide inactive ones.
        self.write(inst, Reg::V(reg), x, None);
        Ok(())
    }

    /// The bits of `v` that reach anything the program does
    /// ([`super::demanded`]).
    fn demanded_bits(&self, v: Value) -> u32 {
        let split = self.split.is_some();
        let st = self.x.st;
        self.demanded
            .get_or_init(|| {
                // Split: the guards are the halves' (every write may keep).
                let guarded = |b: usize, i: usize| split || st.guards[b][i].is_some();
                super::demanded::demanded(self.x.p, &guarded)
            })
            .get(&v)
            .copied()
            .unwrap_or(0)
    }

    /// The register `v`, a barycentric the hardware loads, as data: the host's
    /// `SV_Barycentrics`, whose channel 1 weighs the second vertex (the
    /// console's i, P10) and channel 2 the third (j, P20). Declared in the
    /// last chance only, so that every other program's signature stays as it
    /// was.
    /// shortcut: host vertex k stands for the console's P0, P10, P20 in the
    /// order of the draw's primitive; the console may rotate them to put its
    /// provoking vertex first.
    fn barycentric_input(&mut self, v: Value) -> Option<Val> {
        let &(_, Input::Barycentric { kind, index }) =
            self.x.io.inputs.iter().find(|i| i.0 == v)?
        else {
            return None;
        };
        self.host_barycentric(kind, index).map(Val::I32)
    }

    /// The bits of the host's weight of vertex `index + 1` (channel `index`
    /// of barycentric `kind`: 0 i, 1 j).
    fn host_barycentric(&mut self, kind: Barycentric, index: u8) -> Option<V> {
        // The DXIL interpolation modes of the kinds (see `interp`).
        let mode = match kind {
            Barycentric::PerspectiveCenter => 2,
            Barycentric::PerspectiveCentroid => 3,
            Barycentric::PerspectiveSample => 6,
            Barycentric::LinearCenter => 4,
            Barycentric::LinearCentroid => 5,
            Barycentric::LinearSample => 7,
            Barycentric::PerspectivePullModel => return None,
        };
        let col = match index {
            0 => 1,
            1 => 2,
            _ => return None,
        };
        if super::without_barycentrics() {
            // The weights the geometry shader of `bary.rs` gives each
            // vertex, interpolated in this mode: one input per mode.
            let name = super::bary::BARYCENTRICS;
            let id = self.fallback_input(name.into(), u32::from(mode), |row| SigElement {
                name: name.into(),
                index: u32::from(mode),
                kind: 0,
                comp: 9,
                interp: mode,
                row: Some(row),
                col: 0,
                cols: 3,
                usage: 0,
                half: false,
            });
            return Some(self.load_input_bits(id, col));
        }
        let id = match self.s.inputs.iter().position(|e| e.kind == 28) {
            // One element carries one mode.
            Some(id) if self.s.inputs[id].interp == mode => id,
            Some(_) => return None,
            None => {
                // Without a register, as the sample index has none.
                self.s.inputs.push(SigElement {
                    name: "SV_Barycentrics".into(),
                    index: 0,
                    kind: 28,
                    comp: 9,
                    interp: mode,
                    row: None,
                    col: 0,
                    cols: 3,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.s.flags |= flags::BARYCENTRICS;
        Some(self.load_input_bits(id, col))
    }

    /// The input `name`, `index` of a translation without barycentrics
    /// (`bary.rs`), declared with `element` in a row past the others the
    /// first time it is read.
    fn fallback_input(
        &mut self,
        name: String,
        index: u32,
        element: impl FnOnce(u32) -> SigElement,
    ) -> usize {
        if let Some(id) = self
            .s
            .inputs
            .iter()
            .position(|e| e.kind == 0 && e.name == name && e.index == index)
        {
            return id;
        }
        let row = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .max()
            .map_or(0, |r| r + 1);
        self.s.inputs.push(element(row));
        self.s.inputs.len() - 1
    }

    /// Whether input `id` is read per vertex: some interpolation of it has
    /// weights the program computes, and one element has one mode, so all
    /// its reads do (last chance only).
    fn per_vertex(&self, id: usize) -> bool {
        self.leftovers
            && self.x.io.interps.iter().any(|(_, i)| match i {
                Interp::Computed { attribute, .. } => self
                    .find_input(attribute.row, attribute.channel)
                    .is_some_and(|f| f.0 == id),
                _ => false,
            })
    }

    /// Input `id` declared constant, whose values the program reads at each
    /// vertex.
    fn vertex_element(&mut self, id: usize) -> Result<(), Why> {
        let e = &mut self.s.inputs[id];
        if e.interp != 1 && e.usage != 0 {
            return Err("one input interpolated two ways");
        }
        if e.comp != 9 {
            return Err("interpolation of an integer input");
        }
        e.interp = 1;
        Ok(())
    }

    /// Channel `col` of input `id` interpolated with the host's barycentrics
    /// of `kind` over the values at the three vertices: what the hardware
    /// computes from the same weights, for an input whose other reads have
    /// computed weights.
    fn interpolate_vertices(&mut self, id: usize, col: u8, kind: Barycentric) -> Result<V, Why> {
        self.vertex_element(id)?;
        let i = self
            .host_barycentric(kind, 0)
            .ok_or("barycentrics the host cannot give")?;
        let j = self
            .host_barycentric(kind, 1)
            .ok_or("barycentrics the host cannot give")?;
        let (i, j) = (self.as_float(i), self.as_float(j));
        let a0 = self.vertex_attribute(id, col, 0);
        let a1 = self.vertex_attribute(id, col, 1);
        let a2 = self.vertex_attribute(id, col, 2);
        let c = self.cur;
        let d1 = self.s.m.bin(c, Bin::Sub, a1, a0);
        let d2 = self.s.m.bin(c, Bin::Sub, a2, a0);
        let r = self.fmad(i, d1, a0);
        let r = self.fmad(j, d2, r);
        Ok(self.as_int(r))
    }

    /// One half of an interpolation with weights the program computes, from
    /// the attribute's value at each vertex: the first half is
    /// `P0 + i * (P1 - P0)`, the second adds `j * (P2 - P0)` to the old
    /// destination, as the hardware evaluates them. Only in the last chance.
    fn computed_interp(
        &mut self,
        inst: &Inst,
        attribute: Attribute,
        second: bool,
    ) -> Result<(), Why> {
        if !self.leftovers {
            return Err("an interpolation with computed barycentrics");
        }
        let Fields::Vintrp {
            dst: Operand::Vector { reg: d, .. },
            src: Operand::Vector { reg: w, .. },
            ..
        } = inst.fields
        else {
            return Err("a 16-bit interpolation with computed barycentrics");
        };
        let packed = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => inputs.iter().any(|i| i.row == attribute.row && i.fp16),
            _ => false,
        };
        if packed || attribute.high {
            return Err("computed barycentrics of a 16-bit input");
        }
        let (id, col) = self
            .find_input(attribute.row, attribute.channel)
            .ok_or("an input the signature lacks")?;
        // The values at the vertices are those of a constant input.
        self.vertex_element(id)?;
        let weight = inst.read(Reg::V(w)).ok_or("an interpolation")?;
        let weight = self.i32_of(weight)?;
        let weight = self.as_float(weight);
        let a0 = self.vertex_attribute(id, col, 0);
        let (base, to) = if second {
            let acc = inst.read(Reg::V(d)).ok_or("an interpolation")?;
            let acc = self.i32_of(acc)?;
            (self.as_float(acc), self.vertex_attribute(id, col, 2))
        } else {
            (a0, self.vertex_attribute(id, col, 1))
        };
        let c = self.cur;
        let delta = self.s.m.bin(c, Bin::Sub, to, a0);
        let r = self.fmad(weight, delta, base);
        let r = self.as_int(r);
        self.write(inst, Reg::V(d), r, None);
        Ok(())
    }

    /// Channel `col` of input `id` at one vertex of the primitive (SM 6.1
    /// `GetAttributeAtVertex`; the input is declared constant).
    fn vertex_attribute(&mut self, id: usize, col: u8, vertex: u8) -> V {
        if super::without_barycentrics() {
            // The copy of the input at that vertex that the geometry shader
            // of `bary.rs` gives every vertex: a constant input.
            let e = &self.s.inputs[id];
            let name = super::bary::vertex_copy_name(vertex, e.semantic_name());
            let copy = SigElement {
                name: name.clone(),
                index: e.index,
                kind: 0,
                comp: e.comp,
                interp: 1,
                row: None,
                col: e.col,
                cols: e.cols,
                usage: 0,
                half: false,
            };
            let copy = self.fallback_input(name, copy.index, |row| SigElement {
                row: Some(row),
                ..copy
            });
            let bits = self.load_input_bits(copy, col);
            return self.as_float(bits);
        }
        self.s.inputs[id].usage |= 1 << col;
        self.s.flags |= flags::BARYCENTRICS;
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.i8(vertex),
        ];
        self.s.op(
            entry,
            "attributeAtVertex.f32",
            Attr::ReadNone,
            self.s.t.f32,
            137,
            &args,
        )
    }

    /// Saturate a 32-bit unsigned channel to 16 bits (RDNA2 `image_load` d16).
    fn saturate_u16(&mut self, x: V) -> V {
        let c = self.cur;
        let m = self.s.m.i32(0xFFFF);
        let lt = self.s.m.cmp(c, pred::IULT, x, m);
        self.s.m.select(c, lt, x, m)
    }

    /// Saturate a 32-bit signed channel to 16 bits (RDNA2 `image_load` d16).
    fn saturate_i16(&mut self, x: V) -> V {
        let c = self.cur;
        let (lo, hi, m) = (
            self.s.m.i32((-32768i32) as u32),
            self.s.m.i32(32767),
            self.s.m.i32(0xFFFF),
        );
        let below = self.s.m.cmp(c, pred::SLT, x, lo);
        let x = self.s.m.select(c, below, lo, x);
        let above = self.s.m.cmp(c, pred::SGT, x, hi);
        let x = self.s.m.select(c, above, hi, x);
        self.s.m.bin(c, Bin::And, x, m)
    }

    /// Whether GFX10 `IMG_FORMAT` / buffer format `fmt` is UINT or SINT
    /// (the families in [`buffer_format`]).
    fn format_is_uint_sint(&mut self, fmt: V) -> (V, V) {
        let c = self.cur;
        // Numbers from `buffer_format`'s UINT / SINT slots (1..=77).
        const UINTS: &[u32] = &[5, 11, 18, 20, 27, 34, 41, 48, 54, 60, 62, 69, 72, 75];
        const SINTS: &[u32] = &[6, 12, 19, 21, 28, 35, 42, 49, 55, 61, 63, 70, 73, 76];
        let mut is_uint = self.s.m.int_const(self.s.t.i1, 0);
        for &n in UINTS {
            let lit = self.s.m.i32(n);
            let eq = self.s.m.cmp(c, pred::EQ, fmt, lit);
            is_uint = self.s.m.bin(c, Bin::Or, is_uint, eq);
        }
        let mut is_sint = self.s.m.int_const(self.s.t.i1, 0);
        for &n in SINTS {
            let lit = self.s.m.i32(n);
            let eq = self.s.m.cmp(c, pred::EQ, fmt, lit);
            is_sint = self.s.m.bin(c, Bin::Or, is_sint, eq);
        }
        (is_uint, is_sint)
    }

    /// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit
    /// results keep the register's high half (GFX10).
    fn low_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let old = self.s.m.bin(c, Bin::And, old, high);
        let x = self.s.m.bin(c, Bin::And, x, low);
        self.s.m.bin(c, Bin::Or, old, x)
    }

    /// `x` (low 16 bits) as the high half of VGPR `reg`, keeping its low half.
    fn high_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let (low, k16) = (self.s.m.i32(0xFFFF), self.s.m.i32(16));
        let old = self.s.m.bin(c, Bin::And, old, low);
        let x = self.s.m.bin(c, Bin::Shl, x, k16);
        self.s.m.bin(c, Bin::Or, old, x)
    }

    /// `image_bvh_intersect_ray` and `image_bvh64_intersect_ray`: every
    /// ray misses.
    ///
    /// A leaf node (low pointer bits 0..=3) answers with an infinite
    /// numerator over a denominator of one, so no distance test passes and
    /// no division gives NaN; a box node (4..=7) answers with four
    /// `INVALID_NODE` children, so a traversal loop pops its stack and ends.
    // shortcut: no ray is traced. Shadow rays find no occluder and
    // reflection rays no surface; real traversal needs the BVH read through
    // the title's memory.
    fn bvh_miss(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image { data, addr, .. } = &inst.fields else {
            unreachable!()
        };
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let Some(&Operand::Vector { reg: node, .. }) = addr.first() else {
            return Err("an image address");
        };
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        let node = inst.read(Reg::V(node)).ok_or("an image address")?;
        let node = self.i32_of(node)?;
        let (seven, four) = (self.s.m.i32(7), self.s.m.i32(4));
        let kind = self.s.m.bin(c, Bin::And, node, seven);
        let leaf = self.s.m.cmp(c, pred::IULT, kind, four);
        let invalid = self.s.m.i32(u32::MAX);
        let (inf, one, zero) = (
            self.s.m.i32(0x7F80_0000),
            self.s.m.i32(0x3F80_0000),
            self.s.m.i32(0),
        );
        for (k, (leaf_word, box_word)) in [
            (inf, invalid),
            (one, invalid),
            (zero, invalid),
            (zero, invalid),
        ]
        .into_iter()
        .enumerate()
        {
            let x = self.s.m.select(c, leaf, leaf_word, box_word);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    fn prune_mesh_outputs(
        &mut self,
        mesh: super::shader::Mesh,
        pixel_inputs: Option<&[SigElement]>,
    ) -> Result<(), Stop> {
        use super::shader::Mesh;

        if mesh.output_bytes(&self.s.outputs, &self.s.prims) <= Mesh::MAX_OUTPUT_BYTES {
            return Ok(());
        }

        let mut written = [
            vec![false; self.s.outputs.len()],
            vec![false; self.s.prims.len()],
        ];
        if pixel_inputs.is_none() {
            // Standalone emission lacks a consumer. Only declarations with no
            // corresponding export can go, and rasterizer system values stay.
            for export in &self.x.io.exports {
                let primitive = match export.target {
                    Target::Param { .. } => 0,
                    Target::PrimitiveParam { .. } => 1,
                    _ => continue,
                };
                for (id, _, _, _) in self.export_stores(export).map_err(|w| unsupported(0, w))? {
                    written[primitive][id] = true;
                }
            }
        }

        let retain = |elements: &mut Vec<SigElement>, written: &[bool]| {
            let mut ids = Vec::with_capacity(elements.len());
            let mut next = 0;
            elements.retain(|e| {
                let keep = e.kind != 0
                    || pixel_inputs.map_or(written[ids.len()], |inputs| {
                        inputs.iter().any(|input| {
                            input.kind == 0
                                && input.index == e.index
                                && input.name.eq_ignore_ascii_case(&e.name)
                        })
                    });
                ids.push(keep.then_some(next));
                next += usize::from(keep);
                keep
            });
            ids
        };

        self.mesh_output_ids = Some([
            retain(&mut self.s.outputs, &written[0]),
            retain(&mut self.s.prims, &written[1]),
        ]);
        if mesh.output_bytes(&self.s.outputs, &self.s.prims) > Mesh::MAX_OUTPUT_BYTES {
            return Err(unsupported(
                0,
                "a mesh output signature beyond the host's size limit",
            ));
        }
        Ok(())
    }

    fn export_stores(&self, e: &Export) -> Result<Vec<ExportStore>, Why> {
        let mesh = self.s.mesh.is_some();
        let param_row = |usage: Usage, row: u8| match usage {
            // The 2020 compiler leaves a mesh program's first entry zeroed
            // (usage "position"): its row is the one no other entry names.
            Usage::Other(0) if mesh => self.unnamed_row(),
            _ => Some(row),
        };
        // (output element, channel) per exported channel; per-primitive
        // outputs are elements of `prims`.
        let mut stores = Vec::new();
        let primitive = matches!(e.target, Target::PrimitiveParam { .. });
        let outputs = if primitive {
            &self.x.prims
        } else {
            &self.x.outputs
        };
        let by_kind = |kind: u8| outputs.iter().position(|o| o.kind == kind);
        // 16-bit vertex outputs: the halves each dword carries, as (output
        // element, channel, high half) per channel.
        let halves = match e.target {
            Target::Param { usage, row } => {
                param_row(usage, row).and_then(|r| self.half_outputs.get(&r).map(|&h| (r, h)))
            }
            _ => None,
        };
        if let Some((row, (pack2, b))) = halves {
            for ch in 0..4u8 {
                let Some(value) = e.values[usize::from(ch)] else {
                    continue;
                };
                for high in [false, true] {
                    let h = 2 * ch + u8::from(high);
                    let src = if pack2 {
                        packed_half(outputs, row, b, h)
                    } else {
                        (!high).then_some((row, ch))
                    };
                    let found = src.and_then(|(r, c)| {
                        outputs.iter().position(|o| {
                            o.row == Some(u32::from(r)) && c >= o.col && c < o.col + o.cols
                        })
                    });
                    if let Some(id) = found {
                        let col = src.expect("found").1 - outputs[id].col;
                        stores.push((id, col, value, Some(high)));
                    }
                }
            }
        }
        for ch in 0..4u8 {
            let Some(value) = e.values[usize::from(ch)] else {
                continue;
            };
            if halves.is_some() {
                break;
            }
            let found = match e.target {
                Target::Color(n) => outputs
                    .iter()
                    .position(|o| {
                        o.kind == 16
                            && o.index == u32::from(n)
                            && ch >= o.col
                            && ch < o.col + o.cols
                    })
                    .map(|id| (id, ch - outputs[id].col)),
                Target::Position(0) => by_kind(3).map(|id| (id, ch)),
                Target::Depth => match ch {
                    0 => by_kind(17)
                        .or(by_kind(18))
                        .or(by_kind(19))
                        .map(|id| (id, 0)),
                    1 => by_kind(20).map(|id| (id, 0)),
                    2 => by_kind(14).map(|id| (id, 0)),
                    _ => None,
                },
                Target::Param { usage, row } | Target::PrimitiveParam { usage, row } => {
                    let row = param_row(usage, row);
                    outputs
                        .iter()
                        .position(|o| {
                            row.is_some_and(|r| o.row == Some(u32::from(r)))
                                && ch >= o.col
                                && ch < o.col + o.cols
                        })
                        .map(|id| (id, ch - outputs[id].col))
                }
                Target::Null | Target::Unread => None,
                Target::Position(n @ 1..=2) => {
                    // shortcut: without the misc vector (point size, render
                    // target or viewport index) pos1 and pos2 carry the clip
                    // then cull distances, in semantic order.
                    if outputs.iter().any(|o| matches!(o.kind, 4 | 5)) {
                        return Err("render-target or viewport index exports");
                    }
                    let mut dists: Vec<usize> = (0..outputs.len())
                        .filter(|&i| matches!(outputs[i].kind, 6 | 7))
                        .collect();
                    dists.sort_by_key(|&i| (outputs[i].kind, outputs[i].index));
                    let flat: Vec<(usize, u8)> = dists
                        .iter()
                        .flat_map(|&i| (0..outputs[i].cols).map(move |c| (i, c)))
                        .collect();
                    // Channels past the distances the signature declares are
                    // ones the clipper does not enable (`PA_CL_VS_OUT_CNTL`).
                    let k = usize::from(n - 1) * 4 + usize::from(ch);
                    flat.get(k).copied()
                }
                Target::Position(_) => return Err("clip, cull or point-size exports"),
                Target::Primitive => return Err("a primitive export"),
            };
            if let Some((id, col)) = found {
                // Two f16 channels per dword: an odd channel the high half.
                let half = e.compressed.then_some(ch % 2 == 1);
                stores.push((id, col, value, half));
            }
        }
        Ok(stores)
    }

    fn export(&mut self, b: usize, i: usize) -> Result<(), Why> {
        let e = *self
            .exports
            .get(&(b, i))
            .ok_or("an export I/O raising did not see")?;
        // Two f16 channels per source dword: only the pixel color and
        // depth targets unpack them; the rest stay refused.
        if e.compressed && !matches!(e.target, Target::Color(_) | Target::Depth) {
            return Err("a compressed export");
        }
        let guard = self.guard(b, i)?;
        let mesh = self.s.mesh.is_some();
        // Pixel survival is the EXEC of the last `vm` export (not `done`).
        if self.s.pixel && e.vm {
            self.vm_live = Some(match guard {
                Some(g) => g,
                None => self.s.m.bool(true),
            });
        }
        if guard.is_some() && !mesh && !self.s.pixel {
            return Err("a conditional export");
        }
        if mesh && e.target == Target::Primitive {
            return self.emit_indices(e, guard);
        }
        let primitive = matches!(e.target, Target::PrimitiveParam { .. });
        let mut stores = self.export_stores(e)?;
        if let Some(ids) = &self.mesh_output_ids {
            stores.retain_mut(|store| {
                let Some(id) = ids[usize::from(primitive)][store.0] else {
                    return false;
                };
                store.0 = id;
                true
            });
        }
        if mesh && stores.is_empty() {
            return Ok(());
        }
        fn elements<'l>(l: &'l mut Lower, primitive: bool) -> &'l mut Vec<SigElement> {
            if primitive {
                &mut l.s.prims
            } else {
                &mut l.s.outputs
            }
        }
        if stores
            .iter()
            .any(|s| s.3.is_none() && elements(self, primitive)[s.0].half)
        {
            return Err("a 16-bit output the stage data does not mark");
        }
        // A mesh lane writes its own vertex or primitive, where it exports;
        // guard against threads past the declared vertex/primitive maximum.
        // Pixel colour / depth / stencil writes are last-write-masked by EXEC.
        let at = if mesh {
            let m = self.s.mesh.expect("mesh");
            let threads = self.s.threads.expect("mesh")[0];
            let limit = if primitive {
                m.max_primitives
            } else {
                m.max_vertices
            };
            let guard = if threads > limit {
                let flat = self.mesh_thread();
                let lim = self.s.m.i32(limit);
                let in_bounds = self.s.m.cmp(self.cur, pred::IULT, flat, lim);
                Some(match guard {
                    Some(g) => self.s.m.bin(self.cur, Bin::And, g, in_bounds),
                    None => in_bounds,
                })
            } else {
                guard
            };
            self.begin_guarded(guard)
        } else if self.s.pixel {
            self.begin_guarded(guard)
        } else {
            None
        };
        let (name, opcode) = match (mesh, primitive) {
            (false, _) => ("storeOutput", 5),
            (true, false) => ("storeVertexOutput", 171),
            (true, true) => ("storePrimitiveOutput", 172),
        };
        for (id, col, value, half) in stores {
            let mut x = self.i32_of(value)?;
            let comp = elements(self, primitive)[id].comp;
            if let Some(high) = half {
                let c = self.cur;
                if high {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                // The host links these channels at 32 bits.
                match comp {
                    4 if !e.compressed => {
                        let k16 = self.s.m.i32(16);
                        let low = self.s.m.bin(c, Bin::Shl, x, k16);
                        x = self.s.m.bin(c, Bin::AShr, low, k16);
                    }
                    5 if !e.compressed => {
                        let mask = self.s.m.i32(0xFFFF);
                        x = self.s.m.bin(c, Bin::And, x, mask);
                    }
                    _ => {
                        let f32t = self.s.t.f32;
                        let f = self
                            .s
                            .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                        x = self.as_int(f);
                    }
                }
            }
            if self.buffer.is_some() {
                // The dword as it is: the vertex shader reading it back
                // gives it the element's type.
                let (within, slot) = {
                    let layout = self.buffer.as_ref().expect("checked");
                    let within = if primitive {
                        layout.primitive_dword(id, col)
                    } else {
                        layout.vertex_dword(id, col)
                    };
                    (within, self.mesh_thread())
                };
                self.buffer_store(primitive, slot, within, x);
                continue;
            }
            let el = &mut elements(self, primitive)[id];
            el.usage |= 1 << col;
            let (suffix, x) = match comp {
                9 if half.is_some() => (".f32", self.as_float(x)),
                9 => {
                    let f = match self.floats.get(&value) {
                        Some(&f) => f,
                        None => self.as_float(x),
                    };
                    (".f32", f)
                }
                _ => (".i32", x),
            };
            let mut args = vec![
                self.s.m.i32(id as u32),
                self.s.m.i32(0),
                self.s.m.i8(col),
                x,
            ];
            if mesh {
                args.push(self.mesh_thread());
            }
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                &format!("{name}{suffix}"),
                Attr::NoUnwind,
                void,
                opcode,
                &args,
            );
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: `exp prim` as the lane's primitive. Bit 31 culls it (all
    /// indices 0: no area); any vertex index at or past `max_vertices` is
    /// also culled so out-of-range indices never fault the host rasterizer.
    fn emit_indices(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
        let mesh = self.s.mesh.expect("mesh");
        let threads = self.s.threads.expect("mesh")[0];
        let flat = self.mesh_thread();
        let guard = if threads > mesh.max_primitives {
            let mp = self.s.m.i32(mesh.max_primitives);
            let in_bounds = self.s.m.cmp(self.cur, pred::IULT, flat, mp);
            Some(match guard {
                Some(g) => self.s.m.bin(self.cur, Bin::And, g, in_bounds),
                None => in_bounds,
            })
        } else {
            guard
        };
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let z = self.s.m.i32(0);
        let mv = self.s.m.i32(mesh.max_vertices);
        let mut bad = self.s.m.cmp(c, pred::SLT, x, z);
        let mut idxs = [z; 3];
        for k in 0..3 {
            let (shift, mask) = (self.s.m.i32(10 * k), self.s.m.i32(0x3FF));
            let i = self.s.m.bin(c, Bin::LShr, x, shift);
            let i = self.s.m.bin(c, Bin::And, i, mask);
            let oob = self.s.m.cmp(c, pred::IUGE, i, mv);
            bad = self.s.m.bin(c, Bin::Or, bad, oob);
            idxs[k as usize] = i;
        }
        if self.buffer.is_some() {
            // The three indices in one dword, as the console packs them.
            let mut packed = idxs[0];
            for k in 1..3u32 {
                let shift = self.s.m.i32(10 * k);
                let hi = self.s.m.bin(c, Bin::Shl, idxs[k as usize], shift);
                packed = self.s.m.bin(c, Bin::Or, packed, hi);
            }
            let packed = self.s.m.select(c, bad, z, packed);
            self.buffer_store(true, flat, 0, packed);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let mut args = vec![flat];
        for i in idxs {
            args.push(self.s.m.select(c, bad, z, i));
        }
        let void = self.s.t.void;
        self.s
            .op(c, "emitIndices", Attr::NoUnwind, void, 169, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// The per-vertex arbitrary row no mesh output entry names but one
    /// (the 2020 compiler's zeroed entry).
    fn unnamed_row(&self) -> Option<u8> {
        let StageIo::Ms(ms) = &self.x.native.io else {
            return None;
        };
        let named = |r: u32| {
            ms.outputs
                .iter()
                .any(|o| o.param & 0x80 == 0 && o.usage != Usage::Other(0) && u32::from(o.row) == r)
        };
        let free: std::collections::BTreeSet<u32> = self
            .x
            .outputs
            .iter()
            .filter(|o| o.kind == 0)
            .filter_map(|o| o.row)
            .filter(|&r| !named(r))
            .collect();
        match free.into_iter().collect::<Vec<_>>()[..] {
            [r] => u8::try_from(r).ok(),
            _ => None,
        }
    }

    fn exit(&mut self, b: usize) -> Result<(), Stop> {
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        // Split: the console wave's own branches (see `split`).
        let lane_exit = match (&self.split, self.x.p.blocks[b].exit) {
            (None, _) => self.x.st.exits[b],
            (Some(_), Exit::Jump(t)) => LaneExit::Jump(t),
            (Some(_), Exit::Branch { taken, not_taken }) => LaneExit::Branch {
                when: Bool::TRUE,
                taken,
                not_taken,
            },
            (Some(_), Exit::End) => LaneExit::End,
            (Some(_), Exit::Indirect) => LaneExit::Indirect,
        };
        let term = match lane_exit {
            LaneExit::Jump(t) if self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b] => {
                let ok = self.count_iteration(t, pred::IULT);
                let abort = self.abort_block();
                Term::CondBr(ok, self.block[t].expect("reachable"), abort)
            }
            LaneExit::Jump(t) => {
                self.enter_loop(b, t, self.cur);
                Term::Br(self.block[t].expect("reachable"))
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } if taken == not_taken => {
                // Degenerate branch: both arms share one successor. Emitting
                // CondBr with two critical-edge blocks would insert the same
                // edges map key twice and drop the first edge from fill_phis
                // (VIC-BT-007). Treat it as an unconditional jump.
                let _ = when;
                if self.counters.contains_key(&taken) && self.rpo[taken] <= self.rpo[b] {
                    let ok = self.count_iteration(taken, pred::IULT);
                    let abort = self.abort_block();
                    Term::CondBr(ok, self.block[taken].expect("reachable"), abort)
                } else {
                    self.enter_loop(b, taken, self.cur);
                    Term::Br(self.block[taken].expect("reachable"))
                }
            }
            LaneExit::Branch {
                taken, not_taken, ..
            } if self.console_cf => {
                let cond = self.console_branch(b).map_err(|w| unsupported(offset, w))?;
                self.branch_term(b, cond, taken, not_taken)
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition branches the other way round.
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    _ if self.split.is_some() => {
                        self.split_branch(b).map_err(|w| unsupported(offset, w))?
                    }
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self
                            .atom(atom, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken)
                    }
                    _ => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                };
                self.branch_term(b, cond, taken, not_taken)
            }
            LaneExit::End => {
                if self.s.pixel {
                    // Last `vm` export's EXEC is the survival mask; paths that
                    // never hit a Valid-Mask or `done` export were killed.
                    if self.vm_exported[b] || self.has_vm_export(b) {
                        let live = self.vm_live.expect("pixel vm_live");
                        let t = self.s.m.bool(true);
                        let kill = self.s.m.bin(self.cur, Bin::Xor, live, t);
                        self.s.op(
                            self.cur,
                            "discard",
                            Attr::NoUnwind,
                            self.s.t.void,
                            82,
                            &[kill],
                        );
                    } else if !self.exported[b] && !self.has_done_export(b) {
                        let t = self.s.m.bool(true);
                        self.s
                            .op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                    } else if self.vm_reached[b] {
                        // A Valid-Mask export lies on some paths in only
                        // (the others bypass it): the phi of `vm_live`
                        // is still each lane's last guard, true where no
                        // such export ran.
                        let live = self.vm_live.expect("pixel vm_live");
                        let t = self.s.m.bool(true);
                        let kill = self.s.m.bin(self.cur, Bin::Xor, live, t);
                        self.s.op(
                            self.cur,
                            "discard",
                            Attr::NoUnwind,
                            self.s.t.void,
                            82,
                            &[kill],
                        );
                    }
                }
                Term::Ret
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    /// The terminator of block `b` branching to `taken` where `cond` holds.
    fn branch_term(&mut self, b: usize, cond: V, taken: usize, not_taken: usize) -> Term {
        // A back edge is taken only below the iteration limit.
        let back = |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
        // (No `xor` of a compare: a loop exit testing one hung the
        // AMD driver.)
        let cond = if back(self, taken) {
            let ok = self.count_iteration(taken, pred::IULT);
            self.s.m.bin(self.cur, Bin::And, cond, ok)
        } else if back(self, not_taken) {
            let stop = self.count_iteration(not_taken, pred::IUGE);
            self.s.m.bin(self.cur, Bin::Or, cond, stop)
        } else {
            cond
        };
        // Critical edges get a block of their own, so loops have
        // dedicated exits and preheaders as DXC emits them.
        let (t, f) = (self.edge(b, taken), self.edge(b, not_taken));
        for (to, via) in [(taken, t), (not_taken, f)] {
            if !back(self, to) {
                // From an edge block, or from here.
                let from = if Some(via) == self.block[to] {
                    self.cur
                } else {
                    via
                };
                self.enter_loop(b, to, from);
            }
        }
        Term::CondBr(cond, t, f)
    }

    /// Console flow: whether block `b`'s `s_cbranch_*` is taken, the same
    /// in both halves (wave-wide mask tests through [`Self::any_lane`]).
    fn console_branch(&mut self, b: usize) -> Result<V, Why> {
        let inst = self.x.p.blocks[b].insts.last().ok_or("a branch")?;
        let name = inst.name();
        let c = self.cur;
        let (any, set) = match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                let v = inst.read(Reg::Scc).ok_or("a branch's SCC")?;
                let scc = self.scc_bit(v)?;
                (scc, name == "s_cbranch_scc1")
            }
            "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let reg = if name.contains("vcc") {
                    crate::ir::VCC
                } else {
                    EXEC
                };
                let lo = inst.read(Reg::S(reg)).ok_or("a branch's mask")?;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&lo)
                    .or_else(|| self.x.st.numbers.get(&lo))
                    .ok_or("a branch on an untracked mask")?;
                (self.any_lane(m)?, name.ends_with("nz"))
            }
            _ => return Err("a branch the console flow does not model"),
        };
        Ok(if set {
            any
        } else {
            let t = self.s.m.bool(true);
            self.s.m.bin(c, Bin::Xor, any, t)
        })
    }

    /// Console flow: an SCC value as a bit. A mask operation's SCC (result
    /// not zero) is whether some lane of the console wave has its bit.
    fn scc_bit(&mut self, v: Value) -> Result<V, Why> {
        // A live join must either resolve every incoming SCC or refuse to
        // translate. An unused phi may be filled with a dummy false value.
        if self.phi_at.contains_key(&v) {
            self.read_phis.insert(v);
        }
        if let Some(&Val::Bit(x)) = self.vals.get(&v) {
            return Ok(x);
        }
        let b = *self
            .def_block
            .get(&v)
            .ok_or("an SCC from the program's start")?;
        let inst = self.x.p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))
            .ok_or("an SCC without its instruction")?;
        let mask = inst
            .writes
            .iter()
            .filter(|w| matches!(w.0, Reg::S(_)))
            .find_map(|w| self.x.st.masks.get(&w.1).copied())
            .ok_or("an SCC the emitter did not compute")?;
        self.any_lane(mask)
    }

    /// Console flow: whether `m` holds in some lane of the console wave.
    fn any_lane(&mut self, m: Bool) -> Result<V, Why> {
        let c = self.cur;
        let bit = self.boolean(m, c)?;
        let (lo, hi) = self.console_ballot(c, bit);
        let or = self.s.m.bin(c, Bin::Or, lo, hi);
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, pred::NE, or, z))
    }

    /// Console flow: the console wave's ballot of `bit`, low and high word.
    fn console_ballot(&mut self, at: usize, bit: V) -> (V, V) {
        if self.halves {
            let word = self.ballot_word(at, bit);
            return self.exchange(at, word);
        }
        // One host wave is the console wave (the high word is 0 in waves
        // of 32).
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1))
    }

    /// The host wave's ballot of `bit` (32 lanes: the first word).
    fn ballot_word(&mut self, at: usize, bit: V) -> V {
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        self.s.m.extract(at, r, 0)
    }

    /// The console thread's flat id in the group (see [`Self::thread_id`]).
    fn console_flat(&mut self) -> V {
        if let Some(f) = self.console_flat {
            return f;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = match self.split {
            // The console lane of this half's host lane.
            Some(_) => self.split_lanes().0,
            None => self.s.op(
                e,
                "waveGetLaneIndex",
                Attr::ReadOnly,
                self.s.t.i32,
                111,
                &[],
            ),
        };
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        self.console_flat = Some(f);
        f
    }

    /// This lane's console lane (0..63, 0..31 in waves of 32) and its
    /// half of a wave64 (0, 1).
    fn console_lane(&mut self) -> (V, V) {
        let f = self.console_flat();
        let e = self.s.entry;
        let (mask, k5) = (self.s.m.i32(self.wave() - 1), self.s.m.i32(5));
        let lane = self.s.m.bin(e, Bin::And, f, mask);
        let half = self.s.m.bin(e, Bin::LShr, lane, k5);
        (lane, half)
    }

    /// Console flow: each half gives `x` (the same in all its lanes) and
    /// both get (the lower half's, the upper half's): through two group-
    /// shared dwords between group barriers, which every thread of the
    /// group reaches (see [`CONSOLE_FLOW`]). A group of 32 threads or
    /// fewer has no upper half: its lanes count as inactive there (0).
    fn exchange(&mut self, at: usize, x: V) -> (V, V) {
        let threads: u32 = self.s.threads.expect("compute").iter().product();
        let z = self.s.m.i32(0);
        if threads <= 32 {
            return (x, z);
        }
        let (_, half) = self.console_lane();
        let slots = match self.xchg {
            Some(g) => g,
            None => {
                let arr = self.s.m.array(2, self.s.t.i32);
                let g = self.s.m.global("xchg", arr, 3);
                self.xchg = Some(g);
                g
            }
        };
        let void = self.s.t.void;
        // Thread-group sync with a group-shared memory fence.
        let mode = self.s.m.i32(1 | 8);
        let mine = self.s.m.element(at, slots, half);
        // Every lane of a half writes the same value. Atomic publication
        // keeps those concurrent writes to a shared dword defined.
        self.s.m.atomic(at, super::bitcode::Rmw::Xchg, mine, x);
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        let one = self.s.m.i32(1);
        let p0 = self.s.m.element(at, slots, z);
        let p1 = self.s.m.element(at, slots, one);
        let lo = self.s.m.load(at, p0);
        let hi = self.s.m.load(at, p1);
        // Both read before the next exchange writes.
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        (lo, hi)
    }

    /// A forward edge into loop header `to` (from DXIL block `via`) starts
    /// its count at 0.
    fn enter_loop(&mut self, _from: usize, to: usize, via: usize) {
        if let Some((_, incoming)) = self.counters.get_mut(&to) {
            let z = self.s.m.i32(0);
            incoming.push((z, via));
        }
    }

    /// A back edge to header `to` (a back edge keeps its block, no
    /// critical-edge block): the count goes up; the result compares it with
    /// [`LOOP_LIMIT`] by `p` (below: go on; at or above: stop).
    fn count_iteration(&mut self, to: usize, p: u8) -> V {
        let c = self.cur;
        let (phi, _) = self.counters[&to];
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, phi, one);
        let limit = super::probe::loop_limit().unwrap_or_else(super::loop_limit);
        let limit = self.s.m.i32(limit);
        self.counters
            .get_mut(&to)
            .expect("header")
            .1
            .push((next, c));
        self.s.m.cmp(c, p, next, limit)
    }

    /// The block a lane stopped by [`LOOP_LIMIT`] leaves through.
    fn abort_block(&mut self) -> usize {
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s
                .op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        self.s.m.terminate(a, Term::Ret);
        self.abort = Some(a);
        a
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    fn edge(&mut self, from: usize, to: usize) -> usize {
        let target = self.block[to].expect("reachable");
        let preds = self.x.p.blocks[to]
            .preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return target;
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        e
    }

    /// A dword a scalar load writes, where the load is a descriptor the
    /// program uses as a resource or sampler.
    fn descriptor_dword(&self, v: Value) -> bool {
        let Some(load) = self.x.p.blocks.iter().enumerate().find_map(|(b, block)| {
            block
                .insts
                .iter()
                .position(|i| i.writes.iter().any(|w| w.1 == v))
                .map(|i| (b, i))
        }) else {
            return false;
        };
        self.x.p.blocks[load.0].insts[load.1]
            .name()
            .starts_with("s_load_dword")
            && self.x.res.descriptors.iter().any(|d| {
                d.source
                    .leaves()
                    .iter()
                    .any(|l| matches!(l, Source::Loaded { load: at, .. } if *at == load))
            })
    }

    /// Whether `arg`, a phi's incoming value, is a leftover of a register the
    /// program reuses for data on another edge of `phi`: a dword of a lane
    /// mask or of a descriptor, or a join of such leftovers.
    fn mask_junk(&self, phi: &crate::ir::Phi, arg: Value) -> bool {
        let st = self.x.st;
        // Compute reads any mask as data, and refuses a join it cannot
        // follow rather than guess.
        self.mask_data != MaskData::Console
            && self.leftover(arg, 0)
            && phi
                .args
                .iter()
                .any(|a| matches!(self.vals.get(a), Some(Val::I32(_))) && !st.masks.contains_key(a))
    }

    fn leftover(&self, arg: Value, depth: usize) -> bool {
        let st = self.x.st;
        if st.masks.contains_key(&arg) || st.highs.contains_key(&arg) {
            return true;
        }
        if !matches!(self.vals.get(&arg), Some(Val::Missing(_))) {
            return false;
        }
        if self.descriptor_dword(arg) {
            return true;
        }
        let Some(&(b, i)) = self.phi_at.get(&arg) else {
            return false;
        };
        depth < 4
            && self.x.p.blocks[b].phis[i]
                .args
                .iter()
                .all(|&a| self.leftover(a, depth + 1))
    }

    /// The incoming edges (phi, predecessor index) over which a value can
    /// reach a read. A phi that only another read phi consumes is live too
    /// (else a value with no DXIL form reaching a read through two joins
    /// would be replaced by zero), but only over edges a lane can take on
    /// its way there: each step back adds the branch a lane took on that
    /// edge, and a contradiction (the lane that skipped an arm is not one
    /// that ran it) leaves the edge dead. A back edge forgets the branches
    /// (the loop redefines their conditions). Fewer conditions are weaker,
    /// so keeping only the last few stays sound.
    fn live_edges(&self) -> HashSet<(Value, usize)> {
        const MAX_KNOWN: usize = 4;
        let (p, st) = (self.x.p, &self.x.st);
        let mut work: Vec<(Value, Vec<(Bool, bool)>)> =
            self.read_phis.iter().map(|&v| (v, Vec::new())).collect();
        let mut live = HashSet::new();
        let mut seen = HashMap::<Value, Vec<Vec<(Bool, bool)>>>::new();
        while let Some((v, known)) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            let done = seen.entry(v).or_default();
            // Known of fewer lanes than before: nothing new is live.
            if done.iter().any(|d| d.iter().all(|c| known.contains(c))) {
                continue;
            }
            done.push(known.clone());
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                if self.tail[q].is_none() {
                    continue;
                }
                let mut k = Vec::new();
                // Forward edges out of a block that is no loop's latch (a
                // loop's exit is also taken at its iteration limit).
                if let LaneExit::Branch {
                    when,
                    taken,
                    not_taken,
                } = st.exits[q]
                    && taken != not_taken
                    && self.rpo[taken] > self.rpo[q]
                    && self.rpo[not_taken] > self.rpo[q]
                {
                    k.clone_from(&known);
                    k.push((when, taken == b));
                    k.drain(..k.len().saturating_sub(MAX_KNOWN));
                    if !st.bools.meet(&k) {
                        continue;
                    }
                } else if self.rpo[q] < self.rpo[b] {
                    k.clone_from(&known);
                }
                live.insert((v, n));
                let arg = p.blocks[b].phis[i].args[n];
                if self.phi_at.contains_key(&arg) {
                    work.push((arg, k));
                }
            }
        }
        live
    }

    fn fill_phis(&mut self) -> Result<(), Stop> {
        let p = self.x.p;
        let live = self.live_edges();
        for (v, b, i) in std::mem::take(&mut self.data_phis) {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                let read = live.contains(&(phi.value, n));
                // A barycentric register read through a join (last chance;
                // a join nothing reads takes the zero, as it did).
                if self.leftovers
                    && !bit
                    && read
                    && matches!(self.vals.get(&arg), Some(&Val::Missing(w)) if w == BARYCENTRICS_READ)
                {
                    // A join none of whose bits reach anything needs no
                    // value there (see `get`).
                    let val = if self.demanded_bits(phi.value) == 0 {
                        Some(Val::I32(self.s.m.i32(0)))
                    } else {
                        self.barycentric_input(arg)
                    };
                    if let Some(val) = val {
                        self.vals.insert(arg, val);
                    }
                }
                let x = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    (Some(Val::Bit(_)), false) if read => {
                        return Err(Stop::Retry(phi.value, "an SCC joined with SGPR values"));
                    }
                    // A descriptor's dwords reaching a join that also takes
                    // data (the registers are reused after the last access).
                    // shortcut: undefined on that edge, assuming the program
                    // reads the joined value only after the data edge.
                    (Some(Val::Missing(_)), _) if self.descriptor_dword(arg) => {
                        if bit {
                            self.s.m.undef(self.s.t.i1)
                        } else {
                            self.s.m.undef(self.s.t.i32)
                        }
                    }
                    // A lane mask's dword reaching a join that also takes
                    // data: the register was reused for data on the other
                    // edge, and the mask itself is dead there.
                    // shortcut: undefined on that edge, assuming the program
                    // reads the joined value only after a data edge (as for a
                    // descriptor's dwords above).
                    (Some(Val::Missing(_)), false)
                        if self.leftovers && read && self.mask_junk(phi, arg) =>
                    {
                        self.s.m.undef(self.s.t.i32)
                    }
                    (Some(Val::Missing(why)), _) if read => {
                        return Err(Stop::Retry(phi.value, why));
                    }
                    _ if read => {
                        return Err(Stop::Retry(phi.value, "a join of mismatched values"));
                    }
                    _ if bit => self.s.m.bool(false),
                    _ => self.s.m.i32(0),
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        for (b, phi) in self.vm_live_phis.iter().enumerate() {
            let Some(&phi) = phi.as_ref() else {
                continue;
            };
            let mut incoming = Vec::new();
            for &q in &p.blocks[b].preds {
                let Some(qb) = self.tail[q] else {
                    continue;
                };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let live = self.vm_live_out[q].unwrap_or_else(|| self.s.m.bool(true));
                incoming.push((live, qb));
            }
            self.s.m.set_phi(phi, incoming);
        }
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                let st = self.x.st;
                // A number joined in (a constant before a branch) only where
                // the join was checked to hold per lane.
                let checked = self.console_cf
                    || st.active.contains(&phi.value)
                    || st.outside.contains(&phi.value);
                let m = *st
                    .masks
                    .get(&arg)
                    .or_else(|| st.numbers.get(&arg).filter(|_| checked))
                    .ok_or_else(|| unsupported(offset, "a lane-mask join of an untracked mask"))?;
                let x = self.boolean(m, qb).map_err(|w| unsupported(offset, w))?;
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        Ok(())
    }
}

/// The first choice of a sampler chosen by control flow, and for each other
/// whether the path taken brought it and its handle.
type SamplerChoices = (V, Vec<(V, V)>);

/// IR blocks containing `s_barrier` (not already in `skip`) that are
/// control-dependent, directly or through other branches, on a non-uniform
/// (per-lane) branch: whether a thread reaches the barrier then depends on
/// its lane. Emitting `SYNC_THREAD_GROUP` in such a block deadlocks 32-lane
/// hosts whenever warps diverge. A per-lane branch that reconverges before
/// the barrier does not count: the barrier post-dominates both its sides
/// (or neither, past a later uniform branch that alone decides).
fn divergent_barriers(
    p: &crate::ir::Program,
    st: &crate::structure::Structured,
    skip: &HashSet<usize>,
) -> HashSet<usize> {
    let n = p.blocks.len();
    let barriers: Vec<usize> = (0..n)
        .filter(|b| !skip.contains(b) && p.blocks[*b].insts.iter().any(|i| i.name() == "s_barrier"))
        .collect();
    if barriers.is_empty() {
        return HashSet::new();
    }
    let divergent: Vec<bool> = (0..n)
        .map(|u| match st.exits[u] {
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                taken != not_taken
                    && st
                        .bools
                        .support(when)
                        .iter()
                        .any(|a| !matches!(a, Atom::Uniform(_)))
            }
            _ => false,
        })
        .collect();
    if !divergent.contains(&true) {
        return HashSet::new();
    }
    // The CFG with a virtual exit (`n`) after every block that ends the
    // program.
    let exit = n;
    let mut preds: Vec<Vec<usize>> = p.blocks.iter().map(|b| b.preds.clone()).collect();
    preds.push(
        (0..n)
            .filter(|&b| matches!(st.exits[b], LaneExit::End))
            .collect(),
    );
    let mut succs = vec![Vec::new(); n + 1];
    for (b, ps) in preds.iter().enumerate() {
        for &pred in ps {
            succs[pred].push(b);
        }
    }
    let ipdom = post_dominators(&succs, &preds, exit);
    // `controls[x]`: the branches `x` is directly control-dependent on, the
    // blocks `a` with a successor `x` post-dominates and `a` itself is not
    // strictly post-dominated by `x`.
    let mut controls = vec![Vec::new(); n + 1];
    for a in 0..n {
        let Some(stop) = ipdom[a] else {
            continue;
        };
        for &s in &succs[a] {
            let mut r = s;
            while r != stop {
                if !controls[r].contains(&a) {
                    controls[r].push(a);
                }
                match ipdom[r] {
                    Some(up) if up != r => r = up,
                    _ => break,
                }
            }
        }
    }
    let mut out = HashSet::new();
    for b in barriers {
        let mut seen = vec![false; n + 1];
        let mut stack = vec![b];
        seen[b] = true;
        'search: while let Some(cur) = stack.pop() {
            for &a in &controls[cur] {
                if divergent[a] {
                    out.insert(b);
                    break 'search;
                }
                if !seen[a] {
                    seen[a] = true;
                    stack.push(a);
                }
            }
        }
    }
    out
}

/// Immediate post-dominators over `succs`/`preds` with `exit` as root
/// (Cooper, Harvey and Kennedy's iteration on the reversed graph); `None`
/// for a block that never reaches `exit`.
fn post_dominators(succs: &[Vec<usize>], preds: &[Vec<usize>], exit: usize) -> Vec<Option<usize>> {
    let m = succs.len();
    // Postorder of the reversed graph from `exit`.
    let mut order = Vec::with_capacity(m);
    let mut seen = vec![false; m];
    let mut stack = vec![(exit, 0usize)];
    seen[exit] = true;
    while let Some(top) = stack.last_mut() {
        let (b, i) = *top;
        if let Some(&pred) = preds[b].get(i) {
            top.1 += 1;
            if !seen[pred] {
                seen[pred] = true;
                stack.push((pred, 0));
            }
        } else {
            order.push(b);
            stack.pop();
        }
    }
    let mut number = vec![usize::MAX; m];
    for (k, &b) in order.iter().enumerate() {
        number[b] = k;
    }
    let mut ipdom = vec![None; m];
    ipdom[exit] = Some(exit);
    let intersect = |ipdom: &[Option<usize>], mut a: usize, mut b: usize| {
        while a != b {
            while number[a] < number[b] {
                a = ipdom[a].expect("processed");
            }
            while number[b] < number[a] {
                b = ipdom[b].expect("processed");
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().rev() {
            if b == exit {
                continue;
            }
            let mut new = None;
            for &s in &succs[b] {
                if ipdom[s].is_some() {
                    new = Some(new.map_or(s, |x| intersect(&ipdom, x, s)));
                }
            }
            if new.is_some() && ipdom[b] != new {
                ipdom[b] = new;
                changed = true;
            }
        }
    }
    ipdom
}

/// The lane `v_readlane_b32` / `v_writelane_b32` names, wrapped to the wave
/// (`& 31`, `& 63` in wave64). A constant stays a constant: its value is the
/// lane, not its place in the constant pool (`V::Const` holds the place).
fn wave_lane(m: &mut Module, block: usize, lane: V, wave64: bool) -> V {
    let mask = if wave64 { 63 } else { 31 };
    match m.int_value(lane) {
        Some(k) => m.i32(k as u32 & mask),
        None => {
            let mask = m.i32(mask);
            m.bin(block, Bin::And, lane, mask)
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "select_tests.rs"]
mod select_tests;

#[cfg(test)]
mod wave_size_tests {
    use super::super::{emit, emit_with_wave32};
    use crate::native::{ComputeIo, RootParam};
    use crate::{Container, FourCC, HwStage, NativeProgram, ResourceType, RootType, UsedResource};

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// Assembles a wave64 compute program with one raw UAV table at `s0`,
    /// lowers it, validates DXIL when `dxil.dll` is present, and returns the
    /// lowered `HostShader` along with its `PSV0` `(min, max)` wave size.
    fn compile_wave64_cs(
        words: &[u32],
        threads: [u32; 3],
        rsrc2_extra: u32,
    ) -> (super::super::HostShader, (u32, u32)) {
        compile_wave64_cs_mode(words, threads, rsrc2_extra, true)
    }

    fn compile_wave64_cs_mode(
        words: &[u32],
        threads: [u32; 3],
        rsrc2_extra: u32,
        allow_wave32: bool,
    ) -> (super::super::HostShader, (u32, u32)) {
        compile_wave64_cs_threads(words, threads, rsrc2_extra, allow_wave32, threads)
    }

    fn compile_wave64_cs_threads(
        words: &[u32],
        threads: [u32; 3],
        rsrc2_extra: u32,
        allow_wave32: bool,
        expected_threads: [u32; 3],
    ) -> (super::super::HostShader, (u32, u32)) {
        let shader = try_compile_wave64_cs(words, threads, rsrc2_extra, allow_wave32).unwrap();
        check_wave64_cs(shader, expected_threads)
    }

    fn try_compile_wave64_cs(
        words: &[u32],
        threads: [u32; 3],
        rsrc2_extra: u32,
        allow_wave32: bool,
    ) -> Result<super::super::HostShader, crate::ShaderError> {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 8,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: crate::native::StageIo::Cs(ComputeIo {
                rsrc2: (2 << 1) | rsrc2_extra,
                threads,
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        let p = crate::ir::build(&native.code, true, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        if allow_wave32 {
            emit(&c, &native, &p, &st, &io, &res)
        } else {
            emit_with_wave32(&c, &native, &p, &st, &io, &res, false)
        }
    }

    fn check_wave64_cs(
        shader: super::super::HostShader,
        expected_threads: [u32; 3],
    ) -> (super::super::HostShader, (u32, u32)) {
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut shader.dxil.clone())
                .expect("DXIL validation failed");
        }
        let emitted = Container::parse(&shader.dxil).unwrap();
        let sfi0 = u64::from_le_bytes(emitted.part(FourCC::SFI0).unwrap().data.try_into().unwrap());
        assert_ne!(sfi0 & 0x4000, 0, "expected WAVE_OPS feature bit");
        let psv = emitted.part(FourCC::PSV0).unwrap().data;
        let min_wave = u32::from_le_bytes(psv[20..24].try_into().unwrap());
        let max_wave = u32::from_le_bytes(psv[24..28].try_into().unwrap());
        let psv_threads = [
            u32::from_le_bytes(psv[40..44].try_into().unwrap()),
            u32::from_le_bytes(psv[44..48].try_into().unwrap()),
            u32::from_le_bytes(psv[48..52].try_into().unwrap()),
        ];
        assert_eq!(psv_threads, expected_threads);
        (shader, (min_wave, max_wave))
    }

    /// Wave64 compute program that computes `v7` from `v0` (or `v1, v0`),
    /// runs `extra_before`, executes a whole-wave sum reduction of `v7` into
    /// `v4`, and stores `v4` to `out[flat_tid]`.
    fn wave64_sum_cs_words(setup_2d: bool, extra_before: &[u32], extra_after: &[u32]) -> Vec<u32> {
        let mut words = vec![
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF408_0200,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null
        ];
        if setup_2d {
            words.extend([
                0x340E_0283, // v_lshlrev_b32_e32 v7, 3, v1
                0x4A0E_0107, // v_add_nc_u32_e32 v7, v7, v0
                0x340C_0E82, // v_lshlrev_b32_e32 v6, 2, v7
                0x4A0E_0E81, // v_add_nc_u32_e32 v7, 1, v7
            ]);
        } else {
            words.extend([
                0x340C_0082, // v_lshlrev_b32_e32 v6, 2, v0
                0x4A0E_0081, // v_add_nc_u32_e32 v7, 1, v0
            ]);
        }
        words.extend_from_slice(extra_before);
        words.extend([
            0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
            0xD501_0001,
            0x0002_0E80, // v_cndmask_b32_e64 v1, 0, v7, s[0:1]
            0x4A02_02FA,
            0xFF01_6101, // v_add_nc_u32_dpp v1, v1, v1 row_xmask:1
            0x4A02_02FA,
            0xFF01_6201, // v_add_nc_u32_dpp v1, v1, v1 row_xmask:2
            0x4A02_02FA,
            0xFF01_6401, // v_add_nc_u32_dpp v1, v1, v1 row_xmask:4
            0x4A02_02FA,
            0xFF01_6801, // v_add_nc_u32_dpp v1, v1, v1 row_xmask:8
            0xD777_0002,
            0x0305_8301, // v_permlane16_b32 v2, v1, -1, -1
            0xD778_0003,
            0x0305_8302, // v_permlanex16_b32 v3, v2, -1, -1
            0x4A08_0702, // v_add_nc_u32_e32 v4, v2, v3
            0xD760_0002,
            0x0001_3F04, // v_readlane_b32 s2, v4, 31
            0xD760_0003,
            0x0001_7F04, // v_readlane_b32 s3, v4, 63
            0x7E0A_0203, // v_mov_b32_e32 v5, s3
            0x4A08_0A02, // v_add_nc_u32_e32 v4, s2, v5
            0xBEFE_0400, // s_mov_b64 exec, s[0:1]
            0xE070_1000,
            0x8002_0406, // buffer_store_dword v4, v6, s[8:11], 0 offen
        ]);
        words.extend_from_slice(extra_after);
        words.push(S_ENDPGM);
        words
    }

    #[cfg(windows)]
    fn run_cs_uav(shader: &super::super::HostShader, dwords: u32) -> Vec<(&'static str, Vec<u32>)> {
        use super::super::host::{Dispatch, Host, Root, View};
        use windows::Win32::Graphics::Direct3D12::*;
        let _one = super::super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                continue;
            };
            let mut opts1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: querying D3D12_FEATURE_D3D12_OPTIONS1 into a zeroed struct.
            let supported = unsafe {
                host.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS1,
                    (&raw mut opts1).cast(),
                    size_of_val(&opts1) as u32,
                )
            }
            .is_ok()
                && opts1.WaveOps.as_bool()
                && (opts1.WaveLaneCountMin..=opts1.WaveLaneCountMax).contains(&32);
            if !supported {
                // WARP reports WaveLaneCountMin/Max = 4, so fixed WaveSize(32)
                // pipelines are only executable on hardware Wave32 devices.
                eprintln!(
                    "{label}: skipped WaveSize(32) execution (device supports {}..{})",
                    opts1.WaveLaneCountMin, opts1.WaveLaneCountMax
                );
                continue;
            }
            let (root, pso) = host
                .compute_pipeline(shader)
                .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let res = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(dwords)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            out.push((label, res[0].clone()));
        }
        out
    }

    #[test]
    fn doubled_lane_reads_broadcast_registers_from_both_halves_without_scratch_lds() {
        // Each lane writes tid + 1 near the end of the 32 KiB LDS allocation
        // and reads its opposite half. Sum registers from lanes 31 and 63,
        // then add the cross-half LDS result. Every output tests both paths.
        let mut words = vec![
            0xBE81_0380,
            0xF408_0200,
            0xFA00_0000, // load UAV descriptor
            0x340C_0082, // v_lshlrev_b32 v6, 2, v0 (byte address)
            0x4A0E_0081, // v_add_nc_u32 v7, 1, v0
            0xD760_0002,
            0x0001_3F07, // v_readlane_b32 s2, v7, 31
            0xD760_0003,
            0x0001_7F07, // v_readlane_b32 s3, v7, 63
            0x7E0A_0203, // v_mov_b32 v5, s3
            0x4A08_0A02, // v_add_nc_u32 v4, s2, v5
            0xD834_7F00,
            0x0000_0706, // ds_write_b32 v6, v7 offset:32512
            0x3A00_00A0, // v_xor_b32 v0, 32, v0
            0x3400_0082, // v_lshlrev_b32 v0, 2, v0
            0xD8D8_7F00,
            0x0500_0000, // ds_read_b32 v5, v0 offset:32512
            0x4A08_0905, // v_add_nc_u32 v4, v5, v4
            0xE070_1000,
            0x8002_0406, // buffer_store_dword v4, v6
            S_ENDPGM,
        ];
        let full_lds = 64 << 15;
        let group = [64, 1, 1];
        super::TEST_DOUBLE_LANES.set(Some(false));
        let (_, off) = compile_wave64_cs_mode(&words, group, full_lds, true);
        assert_eq!(off, (64, 64));
        super::TEST_DOUBLE_LANES.set(Some(true));
        let (_, native) = compile_wave64_cs_mode(&words, group, full_lds, false);
        assert_eq!(native, (64, 64));
        let (shader, doubled) =
            compile_wave64_cs_threads(&words, group, full_lds, true, [32, 1, 1]);
        assert_eq!(doubled, (32, 32));
        #[cfg(windows)]
        for (host, result) in run_cs_uav(&shader, 64) {
            assert_eq!(
                result,
                (0..64).map(|lane| 97 + (lane ^ 32)).collect::<Vec<_>>(),
                "{host}"
            );
            eprintln!("{host}: read back all 64 logical lanes");
        }
        #[cfg(not(windows))]
        let _ = shader;
        // Inter-thread state is outside the straight-line indexed-read proof.
        words.insert(words.len() - 1, 0xBF8A_0000); // s_barrier
        let (_, with_barrier) = compile_wave64_cs_mode(&words, group, full_lds, true);
        assert_eq!(with_barrier, (64, 64));
        super::TEST_DOUBLE_LANES.set(None);
    }

    #[test]
    fn split_console_flow_preserves_lds_instruction_order() {
        // Each lane stores its id in LDS, reads from the opposite half,
        // then publishes the result through its own UAV element.
        let words = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF408_0200,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null
            0x3408_0082, // v_lshlrev_b32 v4, 2, v0
            0x3A0A_00A0, // v_xor_b32 v5, 32, v0
            0x340C_0A82, // v_lshlrev_b32 v6, 2, v5
            0xD834_0000,
            0x0000_0004, // ds_write_b32 v4, v0
            0xD8D8_0000,
            0x0200_0006, // ds_read_b32 v2, v6
            0xD760_0002,
            0x0001_5100, // v_readlane_b32 s2, v0, 40
            0x4A04_0402, // v_add_nc_u32 v2, s2, v2
            0xE070_1000,
            0x8002_0204, // buffer_store_dword v2, v4, s[8:11], 0 offen
            S_ENDPGM,
        ];
        let group = [64, 1, 1];
        let lds_512_bytes = 1 << 15;
        super::super::TEST_CONSOLE_FLOW.set(Some(false));
        let (_, native) = compile_wave64_cs(&words, group, lds_512_bytes);
        assert_eq!(native, (64, 64));

        super::super::TEST_CONSOLE_FLOW.set(Some(true));
        let (shader, split) = compile_wave64_cs(&words, group, lds_512_bytes);
        let _ = &shader;
        assert_eq!(split, (32, 32));
        #[cfg(windows)]
        for (host, output) in run_cs_uav(&shader, 64) {
            assert_eq!(
                output,
                (0..64).map(|lane| (lane ^ 32) + 40).collect::<Vec<_>>(),
                "{host}"
            );
        }

        let (_, native_only) = compile_wave64_cs_mode(&words, group, lds_512_bytes, false);
        assert_eq!(native_only, (64, 64));
        super::super::TEST_CONSOLE_FLOW.set(None);
    }

    #[test]
    fn console_flow_broadcasts_the_first_lane_to_both_halves() {
        let words = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF408_0200,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null
            0x3408_0082, // v_lshlrev_b32 v4, 2, v0
            0x7E04_0500, // v_readfirstlane_b32 s2, v0
            0x7E04_0281, // v_mov_b32 v2, 1
            0x4A04_0402, // v_add_nc_u32 v2, s2, v2
            0xE070_1000,
            0x8002_0204, // buffer_store_dword v2, v4, s[8:11], 0 offen
            S_ENDPGM,
        ];
        super::super::TEST_CONSOLE_FLOW.set(Some(true));
        let (shader, bounds) = compile_wave64_cs(&words, [64, 1, 1], 0);
        super::super::TEST_CONSOLE_FLOW.set(None);
        let _ = &shader;
        assert_eq!(bounds, (32, 32));
        #[cfg(windows)]
        for (host, values) in run_cs_uav(&shader, 64) {
            assert_eq!(values, vec![1; 64], "{host}");
        }
    }

    /// A wave64 program: a first-lane read forces the console flow, then
    /// `branch` runs between the stores of 1 and 2 into `out[tid]`.
    fn console_flow_branch_words(branch: &[u32]) -> Vec<u32> {
        let mut words = vec![
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF408_0200,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null
            0x3408_0082, // v_lshlrev_b32 v4, 2, v0
            0x7E04_0500, // v_readfirstlane_b32 s2, v0
            0x7E04_0281, // v_mov_b32 v2, 1
        ];
        words.extend_from_slice(branch);
        words.extend([
            0xE070_1000,
            0x8002_0204, // buffer_store_dword v2, v4, s[8:11], 0 offen
            S_ENDPGM,
        ]);
        words
    }

    #[test]
    fn console_flow_branches_on_a_numeric_vcc() {
        let words = console_flow_branch_words(&[
            0xBEEA_0480, // s_mov_b64 vcc, 0
            0xBF86_0001, // s_cbranch_vccz +1
            0x7E04_0282, // v_mov_b32 v2, 2
        ]);
        super::super::TEST_CONSOLE_FLOW.set(Some(true));
        let (_, bounds) = compile_wave64_cs(&words, [64, 1, 1], 0);
        super::super::TEST_CONSOLE_FLOW.set(None);
        assert_eq!(bounds, (32, 32));
    }

    #[test]
    fn console_flow_scc_join_of_mask_operations_is_resolved_or_refused() {
        let words = console_flow_branch_words(&[
            0x7D88_0088, // v_cmp_gt_u32_e32 vcc, 8, v0
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            0xBF85_0002, // s_cbranch_scc1 +2
            0x8782_7E6A, // s_and_b64 s[2:3], vcc, exec
            0xBF82_0001, // s_branch +1
            0x8784_7E6A, // s_and_b64 s[4:5], vcc, exec
            0xBF85_0001, // s_cbranch_scc1 +1
            0x7E04_0282, // v_mov_b32 v2, 2
        ]);
        super::super::TEST_CONSOLE_FLOW.set(Some(false));
        let off = try_compile_wave64_cs(&words, [64, 1, 1], 0, true);
        super::super::TEST_CONSOLE_FLOW.set(Some(true));
        let on = try_compile_wave64_cs(&words, [64, 1, 1], 0, true);
        super::super::TEST_CONSOLE_FLOW.set(None);
        eprintln!("off: {:?}", off.as_ref().err());
        eprintln!("on: {:?}", on.as_ref().err());
        assert!(on.is_err());
    }

    #[test]
    fn console_flow_exec_hi_only_write_terminates() {
        // A mask whose only change is its high dword is keyed by the
        // unchanged low dword: reading it must stop, not recurse.
        let words = console_flow_branch_words(&[
            0xBEFF_0380, // s_mov_b32 exec_hi, 0
            0xBF86_0001, // s_cbranch_vccz +1
            0x7E04_0282, // v_mov_b32 v2, 2
        ]);
        super::super::TEST_CONSOLE_FLOW.set(Some(true));
        let result = try_compile_wave64_cs(&words, [64, 1, 1], 0, true);
        super::super::TEST_CONSOLE_FLOW.set(None);
        let _ = result;
    }

    #[test]
    fn console_flow_keeps_the_native_whole_wave_reduction() {
        let words = wave64_sum_cs_words(false, &[], &[]);
        super::super::TEST_CONSOLE_FLOW.set(Some(true));
        let (_, wave_bounds) = compile_wave64_cs(&words, [64, 1, 1], 0);
        super::super::TEST_CONSOLE_FLOW.set(None);
        assert_eq!(wave_bounds, (64, 64));
    }

    #[test]
    fn small_wave64_whole_wave_reduction_emits_wave32_and_runs() {
        for (threads, setup_2d, rsrc2_extra, expected_sum) in [
            ([32, 1, 1], false, 0, 32 * 33 / 2),
            ([8, 4, 1], true, 1 << 11, 32 * 33 / 2),
        ] {
            let words = wave64_sum_cs_words(setup_2d, &[], &[]);
            let (shader, wave_bounds) = compile_wave64_cs(&words, threads, rsrc2_extra);
            let _ = (&shader, expected_sum);
            assert_eq!(wave_bounds, (32, 32), "threads={threads:?}");
            #[cfg(windows)]
            {
                let count = threads.iter().product::<u32>();
                for (label, data) in run_cs_uav(&shader, count) {
                    assert_eq!(
                        data,
                        vec![expected_sum; count as usize],
                        "{label} threads={threads:?}"
                    );
                }
            }
        }

        // Divergent control flow (`if (tid < 8)`) around the whole-wave
        // reduction still lowers with WaveSize(32) and computes 1+...+8 = 36.
        let body = wave64_sum_cs_words(false, &[], &[]);
        let sec_start = body.iter().position(|&w| w == 0xBE80_25C1).unwrap();
        let section_and_store = &body[sec_start..body.len() - 1];
        let mut divergent = body[..sec_start].to_vec();
        divergent.extend([
            0x7D88_0088,                                  // v_cmp_gt_u32_e32 vcc, 8, v0
            0xBE84_246A,                                  // s_and_saveexec_b64 s[4:5], vcc
            0xBF88_0000 | section_and_store.len() as u32, // s_cbranch_execz -> restore
        ]);
        divergent.extend_from_slice(section_and_store);
        divergent.extend([
            0xBEFE_0404, // s_mov_b64 exec, s[4:5]
            S_ENDPGM,
        ]);
        let (shader, wave_bounds) = compile_wave64_cs(&divergent, [32, 1, 1], 0);
        let _ = &shader;
        assert_eq!(wave_bounds, (32, 32));
        #[cfg(windows)]
        for (label, data) in run_cs_uav(&shader, 32) {
            let want: Vec<u32> = (0..32).map(|i| if i < 8 { 36 } else { 0 }).collect();
            assert_eq!(data, want, "{label} divergent");
        }
    }

    #[test]
    fn large_group_and_lds_and_disallowed_wave64_reductions_keep_wave64() {
        let base = wave64_sum_cs_words(false, &[], &[]);

        // Non-32-thread groups ([1, 1, 1], [16, 1, 1], [33, 1, 1], [64, 1, 1]) keep WaveSize(64).
        for threads in [[1, 1, 1], [16, 1, 1], [33, 1, 1], [64, 1, 1]] {
            let (_, bounds) = compile_wave64_cs(&base, threads, 0);
            assert_eq!(bounds, (64, 64), "threads={threads:?}");
        }

        // [32, 1, 1] with LDS keeps WaveSize(64).
        let (_, bounds) = compile_wave64_cs(&base, [32, 1, 1], 1 << 15);
        assert_eq!(bounds, (64, 64));

        // [32, 1, 1] with independent cross-lane exchange outside reduction
        // sections keeps WaveSize(64).
        for outside_op in [
            &[0x4A0E_0EFA, 0xFF01_6107][..], // v_add_nc_u32_dpp v7, v7, v7 row_xmask:1
            &[0xD778_0007, 0x0305_8307][..], // v_permlanex16_b32 v7, v7, -1, -1
            &[0xD760_0004, 0x0001_0107][..], // v_readlane_b32 s4, v7, 0
        ] {
            let words = wave64_sum_cs_words(false, outside_op, &[]);
            let (_, bounds) = compile_wave64_cs(&words, [32, 1, 1], 0);
            assert_eq!(bounds, (64, 64), "outside_op={outside_op:#x?}");
        }

        // [32, 1, 1] with a held constant mask (`masks_held`) keeps WaveSize(64).
        let held = wave64_sum_cs_words(
            false,
            &[0xBE86_24FF, 0x0000_1234], // s_and_saveexec_b64 s[6:7], 0x1234
            &[0xBEFE_0406],              // s_mov_b64 exec, s[6:7]
        );
        let (_, bounds) = compile_wave64_cs(&held, [32, 1, 1], 0);
        assert_eq!(bounds, (64, 64));

        // On adapters that only support 64-lane waves (`allow_wave32 = false`),
        // an otherwise eligible <=32-thread reduction keeps WaveSize(64).
        let (_, bounds) = compile_wave64_cs_mode(&base, [32, 1, 1], 0, false);
        assert_eq!(bounds, (64, 64));
    }

    #[test]
    fn sparse_mask_32_thread_wave64_float_reduction_emits_wave32_and_runs() {
        let words = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF408_0100,
            0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], null
            0x7E0A_02FF,
            0x3F80_0000, // v_mov_b32 v5, 1.0
            0x3606_0081, // v_and_b32 v3, 1, v0
            0x7D84_0680, // v_cmp_eq_u32 vcc, 0, v3
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBE8A_25C1, // s_or_saveexec_b64 s[10:11], -1
            0xD501_0001,
            0x002A_0A80, // v_cndmask_b32 v1, 0, v5, s[10:11]
            0x0602_02E9,
            0xDE54_C101,
            0x0602_02E9,
            0x9360_1201,
            0x0602_02E9,
            0x0009_2401,
            0x0602_02FA,
            0xFF09_4001,
            0xD778_0002,
            0x0201_0101,
            0x0602_0302,
            0xD760_0002,
            0x0001_4101, // v_readlane_b32 s2, v1, 32
            0x0602_0202,
            0xD760_0003,
            0x0001_0101, // v_readlane_b32 s3, v1, 0
            0xBEFE_040A, // s_mov_b64 exec, s[10:11]
            0x7E04_0203, // v_mov_b32 v2, s3
            0x3406_0082, // v_lshlrev_b32 v3, 2, v0
            0xE070_1000,
            0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
            S_ENDPGM,
        ];
        let (shader, wave_bounds) = compile_wave64_cs(&words, [32, 1, 1], 0);
        let _ = &shader;
        assert_eq!(wave_bounds, (32, 32));
        let (_, bounds_33) = compile_wave64_cs(&words, [33, 1, 1], 0);
        assert_eq!(
            bounds_33,
            (64, 64),
            "33-thread variant must stay out of the Wave32 reduction path"
        );
        #[cfg(windows)]
        for (label, data) in run_cs_uav(&shader, 32) {
            let expected: Vec<u32> = (0..32)
                .map(|i| if i % 2 == 0 { 16.0f32.to_bits() } else { 0 })
                .collect();
            assert_eq!(data, expected, "{label} sparse-mask float reduction");
        }
    }
}
