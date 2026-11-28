//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it.
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler used for comparison has a comparison filter there
//!   (reduction bits 7-8 set to 0x80); the same console words used both ways get two slots so
//!   annotateHandle matches `dx.resources`. A heap sampler used for
//!   comparison must be a comparison sampler on the host.
//! * **Root CBVs and root SRVs** (raw buffers) as root descriptors
//!   `b<i>` / `t<i>` in [`ROOT_CBV_SPACE`], `i` the title's root parameter
//!   index ([`HostShader::root_cbvs`], [`HostShader::root_srvs`]); the
//!   address the title sets is the buffer's.
//! * **Vertex fetch** (added 2026-11-23). The title's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the host binds the root SRV `t<slot>` above with the
//!   address the title sets, nothing more. The vertex index, the stride
//!   and the attribute offset are the shader's own arithmetic on data in
//!   that buffer (per-stream headers), not descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride or the bound view's stride when indexed; typed with four
//!   32-bit float channels for `buffer_load_format_*` and
//!   `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way.
//!
//! * **Texture atomics** use a scalar integer UAV. The host binds R32_UINT
//!   for 32-bit operations and R32G32_UINT for 64-bit operations. The latter
//!   require SM 6.6 atomic support on typed and descriptor-heap resources.
//! * **Barycentrics** (last chance only): a pixel shader that computes the
//!   weights it interpolates with, or reads the barycentric registers as
//!   data, gets `SV_Barycentrics` (SM 6.1; the host's weights of the second
//!   and third vertex are the console's i and j) and the attribute at each
//!   vertex (`GetAttributeAtVertex`, the input declared constant). The
//!   pipeline needs `D3D12_FEATURE_DATA_D3D12_OPTIONS3.BarycentricsSupported`.
//!   For a host without it, [`translate_without_barycentrics`] reads the
//!   values at each vertex and the weights from inputs that the geometry
//!   shader [`barycentric_gs`] adds (see `bary.rs`); a program that needs
//!   neither comes out the same either way.
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them). When the output signature exceeds the
//!   DXIL byte limit, the pipeline keeps only the varyings its pixel shader
//!   declares and the rasterizer system values.

mod bary;
#[cfg(all(test, windows))]
mod barycentric_tests;
mod bitcode;
#[cfg(test)]
mod bvh_tests;
#[cfg(all(test, windows))]
mod code_table_tests;
mod container;
mod demanded;
#[cfg(all(test, windows))]
mod diff_tests;
#[cfg(test)]
mod half_src_tests;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
#[cfg(all(test, windows))]
mod image_store_tests;
#[cfg(all(test, windows))]
mod image_tests;
#[cfg(all(test, windows))]
mod intersect_tests;
#[cfg(all(test, windows))]
mod lane_mask_tests;
#[cfg(test)]
mod lds_tests;
mod lower;
#[cfg(test)]
mod mad_flush_tests;
#[cfg(test)]
mod memory_tests;
mod meshbuf;
#[cfg(all(test, windows))]
mod meshbuf_tests;
#[cfg(test)]
mod mtbuf_tests;
#[cfg(all(test, windows))]
mod narrow_store_tests;
pub mod probe;
mod rect;
#[cfg(test)]
mod root_uav_tests;
#[cfg(test)]
mod root_view_tests;
#[cfg(any(test, feature = "samples"))]
pub mod sample;
mod sampler;
pub use bary::barycentric_gs;
pub use meshbuf::MeshBuffer;
pub use rect::rect_list_gs;
mod shader;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod split_diff;
#[cfg(all(test, windows))]
mod split_tests;
mod threading;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod valu_tests;
#[cfg(all(test, windows))]
mod waterfall_tests;
mod wave;
#[cfg(all(test, windows))]
mod wave_prefix_tests;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use sampler::StaticSampler;
pub use shader::{
    MEMORY_FAULTS, MEMORY_PAGES, MESH_BUFFER_SPACE, ROOT_CBV_SPACE, SAMPLER_SPACE, SLOT_FAULTS,
    SLOT_MEMORY, USER_DATA_SPACE, USER_MEMORY_BASE,
};

/// Resource heap indices at or past this go to slot 0 in translated
/// shaders (a negative or garbage index must not reach past the heap's
/// end). It is far above any real heap, and the host never makes a view
/// heap larger, so a valid index is never moved: 2^24.
pub const VIEW_HEAP_LIMIT: u32 = 1 << 24;

/// The same for sampler heap indices: the host's limit on every binding
/// tier.
pub const SAMPLER_HEAP_LIMIT: u32 = 2048;

/// The loop limit without an override: 2^20.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// The iteration count after which a translated loop is left, per lane and
/// per entry into the loop: [`DEFAULT_LOOP_LIMIT`].
///
/// With the `diagnostics` feature, `VICTORY_SHADER_LOOP_LIMIT` (decimal or
/// `0x` hex, at least 1) sets it instead. A loop that hits the limit in
/// every lane of a full-screen pass still runs for seconds (2^20 times a
/// 4K screen is about 10^13 iterations); a much lower limit shows whether
/// such loops are what makes a frame slow. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn loop_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        if cfg!(feature = "diagnostics") {
            parse_loop_limit(std::env::var("VICTORY_SHADER_LOOP_LIMIT").ok().as_deref())
        } else {
            lower::LOOP_LIMIT
        }
    })
}

/// Whether eligible wave64 compute programs may use one 32-lane host wave,
/// with each thread carrying the registers of two console lanes.
pub fn wave64_double() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("VICTORY_WAVE64_DOUBLE").as_deref() == Some("1".as_ref()))
}

#[cfg(test)]
thread_local! {
    static TEST_CONSOLE_FLOW: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Opt in to lowering console Wave64 compute for a Wave32 host.
pub fn wave64_console_flow() -> bool {
    #[cfg(test)]
    if let Some(value) = TEST_CONSOLE_FLOW.get() {
        return value;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("VICTORY_WAVE64_CONSOLE_FLOW").as_deref() == Some("1".as_ref())
    })
}

/// Verified water-shading compute hashes safe with Wave32 console-flow
/// lowering (includes `a10179ab7eda0dc3` =
/// `CS_TransmissionVolumetrics_VolumePortalsOnly` / Submerged from
/// fxceless/victory-fork#1248).
///
/// `5dc3b25986622949` and `6f225b641c5b0708` (`_Tiled` / `_Tiled_LocalLights`)
/// are intentionally excluded (caused black rectangles around the boat).
const WAVE_WIDE_ALLOWLIST: &[&str] = &[
    "6871d6e4b3ac9a10",
    "926c46bab3b86bd6",
    "a10179ab7eda0dc3",
    "369cbe906ea72764",
    "2cc4c197ee29e291",
];

thread_local! {
    /// Wave32 console-flow lowering for the program being translated on this
    /// thread ([`with_wave_wide`]), when `VICTORY_WAVE_WIDE` is unset.
    static WIDE_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn parse_wave_wide_only(value: Option<&str>) -> Vec<String> {
    value
        .unwrap_or_default()
        .split(',')
        .map(|h| h.trim().to_ascii_lowercase())
        .filter(|h| !h.is_empty())
        .collect()
}

fn matches_wave_wide_only(hash: &str, only: &[String]) -> bool {
    if only.is_empty() {
        return false;
    }
    let hash = hash.to_ascii_lowercase();
    only.iter().any(|o| hash.starts_with(o.as_str()))
}

/// Reads `VICTORY_WAVE_WIDE` (`0` disables Wave32 console-flow lowering even
/// for verified water-shading hashes; `1` enables for all eligible Wave64
/// compute shaders; unset uses the verified allowlist and
/// `VICTORY_WAVE_WIDE_ONLY`).
fn wave_wide_env() -> Option<&'static std::ffi::OsStr> {
    static VALUE: std::sync::OnceLock<Option<std::ffi::OsString>> = std::sync::OnceLock::new();
    VALUE
        .get_or_init(|| std::env::var_os("VICTORY_WAVE_WIDE"))
        .as_deref()
}

/// Reads `VICTORY_WAVE_WIDE_ONLY` (comma-separated 16-hex-digit hash prefixes
/// that override the default water-shading allowlist when non-empty).
fn wave_wide_only_list() -> &'static [String] {
    static ONLY: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    ONLY.get_or_init(|| {
        parse_wave_wide_only(std::env::var("VICTORY_WAVE_WIDE_ONLY").ok().as_deref())
    })
}

/// Whether `VICTORY_WAVE_WIDE_ONLY=<hash prefix>,...` names the container whose
/// 16-hex-digit BLAKE3 prefix is `hash`.
pub fn wave_wide_only(hash: &str) -> bool {
    matches_wave_wide_only(hash, wave_wide_only_list())
}

/// Runs `f` (a translation on this thread) with Wave32 console-flow lowering
/// enabled when `on` (unless opted out via `VICTORY_WAVE_WIDE=0`).
pub fn with_wave_wide<T>(on: bool, f: impl FnOnce() -> T) -> T {
    let before = WIDE_HERE.replace(on);
    let r = f();
    WIDE_HERE.set(before);
    r
}

/// Encodes `VICTORY_WAVE_WIDE` and `VICTORY_WAVE_WIDE_ONLY` settings for disk
/// shader-cache keying so toggling the opt-out never serves stale DXIL.
pub fn wave_wide_cache_key() -> Vec<u8> {
    let mut out = Vec::new();
    let mode = match wave_wide_env().and_then(std::ffi::OsStr::to_str) {
        Some("1") => 1u8,
        Some("0") => 2u8,
        _ => 0u8,
    };
    out.push(mode);
    for prefix in wave_wide_only_list() {
        out.extend_from_slice(prefix.as_bytes());
        out.push(b',');
    }
    out
}

fn wide_lowering_enabled(
    hash: &str,
    wide_env: Option<&std::ffi::OsStr>,
    console_flow_env: bool,
    wide_only: &[String],
    wide_here: bool,
) -> bool {
    if wide_env.is_some_and(|v| v == "1") {
        return true;
    }
    if wide_env.is_some_and(|v| v == "0") {
        return false;
    }
    if console_flow_env {
        return true;
    }
    if !wide_only.is_empty() {
        return wide_here || matches_wave_wide_only(hash, wide_only);
    }
    if wide_here {
        return true;
    }
    WAVE_WIDE_ALLOWLIST
        .iter()
        .any(|&allowed| hash.eq_ignore_ascii_case(allowed))
}

fn console_flow_for_hash(hash: &str) -> bool {
    #[cfg(test)]
    if let Some(value) = TEST_CONSOLE_FLOW.get() {
        return value;
    }
    wide_lowering_enabled(
        hash,
        wave_wide_env(),
        wave64_console_flow(),
        wave_wide_only_list(),
        WIDE_HERE.get(),
    )
}

/// The clean-up passes that run over a translated module before it is
/// written (`bitcode::opt`): a set of its pass bits. Every pass is exact, so
/// the switch is for telling a lowering fault from a clean-up fault and for
/// measuring what the clean-up is worth.
///
/// With the `diagnostics` feature, `VICTORY_SHADER_OPT` (decimal or `0x`
/// hex) chooses them: `0` none; 1 dead code, 2 constants and identities, 4
/// one instruction for one value, 8 what a branch taken says about its
/// condition, 16 one read for one place within a block. Unset or unreadable:
/// every pass. The value is part of the translation, so the shader cache
/// must key on it (`victory-gpu`'s does).
pub fn shader_opt() -> u32 {
    static PASSES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *PASSES.get_or_init(|| {
        if cfg!(feature = "diagnostics") {
            parse_shader_opt(std::env::var("VICTORY_SHADER_OPT").ok().as_deref())
        } else {
            bitcode::opt::ALL
        }
    })
}

/// [`fixes`] bit: spill-table loads read their own codes (`ir::spill_loads`).
pub const FIX_SPILL: u32 = 1;
/// [`fixes`] bit: indexed access through a root descriptor as a raw access,
/// and root descriptors the resource list leaves out as root views.
pub const FIX_ROOT_INDEX: u32 = 2;
/// [`fixes`] bit: byte and halfword buffer stores.
pub const FIX_NARROW_STORE: u32 = 4;

/// Translator changes that are off unless switched on, while they are being
/// bisected in the world: `VICTORY_SHADER_SPILL_FIX=1` ([`FIX_SPILL`]),
/// `VICTORY_SHADER_ROOT_INDEX=1` ([`FIX_ROOT_INDEX`]),
/// `VICTORY_SHADER_NARROW_STORE=1` ([`FIX_NARROW_STORE`]). Tests run with
/// all of them. The value is part of the translation, so the shader cache
/// must key on it (`victory-gpu`'s does).
pub fn fixes() -> u32 {
    static FIXES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    if cfg!(test) {
        return FIX_SPILL | FIX_ROOT_INDEX | FIX_NARROW_STORE;
    }
    *FIXES.get_or_init(|| {
        let on = |name: &str| std::env::var(name).is_ok_and(|v| v.trim() == "1");
        [
            ("VICTORY_SHADER_SPILL_FIX", FIX_SPILL),
            ("VICTORY_SHADER_ROOT_INDEX", FIX_ROOT_INDEX),
            ("VICTORY_SHADER_NARROW_STORE", FIX_NARROW_STORE),
        ]
        .into_iter()
        .filter(|&(name, _)| on(name))
        .fold(0, |bits, (_, bit)| bits | bit)
    })
}

/// `VICTORY_SHADER_OPT`'s value; every pass when it is unset or
/// unreadable. Pure, for testing.
fn parse_shader_opt(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    parsed.map_or(bitcode::opt::ALL, |n| n & bitcode::opt::ALL)
}

/// `VICTORY_SHADER_LOOP_LIMIT`'s value; the default when it is unset,
/// unreadable or 0. Pure, for testing.
fn parse_loop_limit(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => lower::LOOP_LIMIT,
    }
}

/// Parses `VICTORY_INDIRECT_AMPLIFY`: enabled by default; explicit `"0"`,
/// `"false"`, `"no"`, or `"off"` (ASCII case-insensitive) disables the
/// amplification stage for one-record indirect mesh draws.
pub fn parse_indirect_amplify(value: Option<&std::ffi::OsStr>) -> bool {
    let Some(value) = value else {
        return true;
    };
    let Some(text) = value.to_str() else {
        return true;
    };
    let trimmed = text.trim();
    !(trimmed == "0"
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no")
        || trimmed.eq_ignore_ascii_case("off"))
}

/// Whether one-record indirect mesh draws use our amplification stage
/// (`VICTORY_INDIRECT_AMPLIFY`, on by default; `=0` disables).
///
/// Part of the translation: `victory-gpu`'s shader cache keys on it.
pub fn indirect_amplify_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        parse_indirect_amplify(std::env::var_os("VICTORY_INDIRECT_AMPLIFY").as_deref())
    })
}

/// The payload bytes mesh shaders declare: 4 when
/// [`indirect_amplify_enabled`] (`VICTORY_INDIRECT_AMPLIFY=0` disables;
/// `victory-gpu` puts an amplification stage of its own before them, whose
/// payload is one word they do not read; the host wants the two sizes to
/// agree), else 0.
pub(crate) fn mesh_payload() -> u32 {
    if indirect_amplify_enabled() { 4 } else { 0 }
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the shader reads: `t<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_srvs: Vec<u8>,
    /// Root UAVs (raw buffers) the shader reads and writes: `u<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_uavs: Vec<u8>,
    /// The table entries the shader reads as structured buffers. A raw view
    /// the title binds at one of them needs a structured alias on the host:
    /// a view whose kind differs from the declaration is undefined (NVIDIA
    /// reads zeros).
    pub structured_srvs: Vec<StructuredSrv>,
}

/// A table entry a shader reads through a `StructuredBuffer` handle: root
/// parameter `slot`, `offset_dwords` into its table, `stride` bytes per
/// element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructuredSrv {
    pub slot: u8,
    pub offset_dwords: u32,
    pub stride: u32,
}

/// A table entry a translated shader reads through a descriptor heap
/// handle: root parameter `slot`, `offset_dwords` into its table, from the
/// view (or, `sampler`, the sampler) heap, as `props` (the DXIL resource
/// properties the handle is annotated with: kind in the low byte, `0x1000`
/// for a UAV). The host compares them with the descriptors it placed there
/// ([`heap_reads`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HeapRead {
    pub slot: u8,
    pub offset_dwords: u32,
    pub sampler: bool,
    pub props: (u32, u32),
    /// The code adds an index of its own: `offset_dwords` is where the
    /// entries it picks from start.
    pub dynamic: bool,
    /// The offset is from the heap's start (`ResourceDescriptorHeap`
    /// parameters), not from a table's.
    pub absolute: bool,
    /// The index the code adds is root constant (slot, dword): the entry
    /// read is known when the draw or dispatch is recorded.
    pub index_from: Option<(u8, u32)>,
    /// The index the code adds is read from a buffer ([`IndexLoad`]).
    pub index_load: Option<IndexLoad>,
}

/// A heap index read from memory: the dword at byte `byte` of the buffer
/// whose view is at the heap index in root constant (`slot`, `dword`),
/// AND `mask`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndexLoad {
    pub slot: u8,
    pub dword: u32,
    pub byte: u32,
    pub mask: u32,
}

/// The heap reads of translated shaders, by container digest. Kept beside
/// [`HostShader`] rather than in it: only the host's descriptor check reads
/// them.
static HEAP_READS: std::sync::Mutex<
    std::collections::BTreeMap<[u8; 16], std::sync::Arc<[HeapRead]>>,
> = std::sync::Mutex::new(std::collections::BTreeMap::new());

fn container_digest(dxil: &[u8]) -> Option<[u8; 16]> {
    dxil.get(4..20)?.try_into().ok()
}

thread_local! {
    /// How the dynamic heap indices of the shader being lowered are
    /// computed (`VICTORY_SHADER_TRACE_DYNAMIC`, diagnostics).
    pub(crate) static DYNAMIC_CHAINS: std::cell::RefCell<Vec<String>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Takes the dynamic index descriptions recorded on this thread.
pub fn take_dynamic_chains() -> Vec<String> {
    DYNAMIC_CHAINS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

/// Records the heap reads of the shader whose container is `dxil` (a
/// translation, or one read back from a cache).
pub fn note_heap_reads(dxil: &[u8], reads: Vec<HeapRead>) {
    let Some(key) = container_digest(dxil) else {
        return;
    };
    let mut map = HEAP_READS.lock().unwrap_or_else(|e| e.into_inner());
    if reads.is_empty() {
        map.remove(&key);
    } else {
        map.insert(key, reads.into());
    }
}

/// The heap reads recorded for the shader whose container is `dxil`; empty
/// when none were (or it reads none).
pub fn heap_reads(dxil: &[u8]) -> std::sync::Arc<[HeapRead]> {
    container_digest(dxil)
        .and_then(|key| {
            HEAP_READS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&key)
                .cloned()
        })
        .unwrap_or_else(|| std::sync::Arc::from(Vec::new()))
}

/// `SFI0` feature bits of a shader that indexes the resource or the sampler
/// descriptor heap directly.
pub(crate) const SFI0_RESOURCE_HEAP: u64 = 0x200_0000;
pub(crate) const SFI0_SAMPLER_HEAP: u64 = 0x400_0000;
/// `SFI0` feature bit of a shader that reads barycentrics or the attributes
/// at each vertex.
pub(crate) const SFI0_BARYCENTRICS: u64 = 0x2_0000;

impl HostShader {
    /// Whether the shader indexes a descriptor heap directly
    /// (`ResourceDescriptorHeap` / `SamplerDescriptorHeap`), which its root
    /// signature must then allow. A container without readable feature
    /// flags counts as one that does.
    pub fn indexes_heaps(&self) -> bool {
        let features = Container::parse(&self.dxil).ok().and_then(|c| {
            let data = c.part(FourCC::SFI0)?.data.get(..8)?;
            Some(u64::from_le_bytes(data.try_into().ok()?))
        });
        features.is_none_or(|f| f & (SFI0_RESOURCE_HEAP | SFI0_SAMPLER_HEAP) != 0)
    }

    /// Whether the shader needs the host's barycentrics (`SFI0`'s
    /// barycentrics bit: `SV_Barycentrics` or `GetAttributeAtVertex`), which
    /// a host without `BarycentricsSupported` refuses.
    pub fn needs_barycentrics(&self) -> bool {
        Container::parse(&self.dxil)
            .ok()
            .and_then(|c| {
                let data = c.part(FourCC::SFI0)?.data.get(..8)?;
                Some(u64::from_le_bytes(data.try_into().ok()?))
            })
            .is_some_and(|f| f & SFI0_BARYCENTRICS != 0)
    }

    /// The shader model and `SFI0` feature flags the DXIL declares, for the
    /// log line of a pipeline the host refused (`sm 6.6 features 0x..`).
    pub fn declared(&self) -> String {
        let c = Container::parse(&self.dxil).ok();
        let model = c
            .as_ref()
            .and_then(|c| c.part(FourCC::DXIL)?.data.get(..2).map(|b| b[0]))
            .map_or("?".to_owned(), |v| format!("{}.{}", v >> 4, v & 15));
        let features = c.as_ref().and_then(|c| {
            let data = c.part(FourCC::SFI0)?.data.get(..8)?;
            Some(u64::from_le_bytes(data.try_into().ok()?))
        });
        // PSV0: the info size, then the runtime info, whose bytes 16..24 are
        // the wave size range (0 and u32::MAX: any).
        let waves = c.as_ref().and_then(|c| {
            let d = c.part(FourCC::PSV0)?.data;
            Some((
                crate::dxbc::u32_at(d, 20).ok()?,
                crate::dxbc::u32_at(d, 24).ok()?,
            ))
        });
        let waves = match waves {
            Some((min, max)) if min == max => format!("WaveSize({min})"),
            Some(_) => "any wave size".to_owned(),
            None => "?".to_owned(),
        };
        format!(
            "sm {model} features {} {waves}",
            features.map_or("?".to_owned(), |f| format!("{f:#x}"))
        )
    }

    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + if u.indexed_buffer { 3 } else { 1 })
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }
}

/// A mesh program for a host without mesh shaders (see the notes in
/// `meshbuf.rs`): the program as a compute shader that stores its outputs in
/// a buffer, and the vertex shader that reads them back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshFallback {
    /// The program: user data, descriptors and root views as a mesh shader's,
    /// plus the buffer as the root UAV `u0` in [`MESH_BUFFER_SPACE`]. A
    /// program that reads its thread group's number adds the dword of the
    /// constant buffer `b0` in that space to the host's: the host may run a
    /// dispatch in parts, whose groups each count from 0.
    pub compute: HostShader,
    /// Reads the buffer, the root SRV `t0` in [`MESH_BUFFER_SPACE`], and
    /// nothing else.
    pub vertex: HostShader,
    pub buffer: MeshBuffer,
}

/// Bytes per record of the host's buffer view table (see [`UserData`]).
pub const BUFFER_TABLE_RECORD: u32 = 16;

/// One dword of user data and the root parameter that fills it.
///
/// With `root` [`RootType::ResourceHeap`] (and `indexed_buffer` false) the
/// host supplies at `index` the heap index of its buffer view table: a raw
/// SRV with one [`BUFFER_TABLE_RECORD`]-byte record per slot of the bound
/// view heap, for buffer loads whose element stride only the view knows (a
/// descriptor chosen at run time, such as each lane's own in a waterfall
/// loop). A record is four dwords: the heap index of a raw SRV holding the
/// view's bytes, the element stride in bytes (0 for a raw view, which an
/// index does not move through), the byte at which the view starts in that
/// SRV, and the view's element count (its bytes for a raw view; 0 when the
/// slot holds no buffer view the host can read this way).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// The root constant selects a heap buffer. At `index` the host supplies
    /// its bounded raw SRV slot, element stride in bytes and alignment bias.
    pub indexed_buffer: bool,
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants` or `Table` (see the module notes for the value).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

/// A statically addressed descriptor whose structured stride can come from
/// the bound view. Offsets are in the console's descriptor-table dwords.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StructuredBinding {
    pub slot: u8,
    pub offset_dwords: u32,
    pub uav: bool,
}

// A required lookup discovers bindings. An optional lookup recognizes a
// binding already being specialized when lowering another access to it.
type ViewStride<'a> = &'a dyn Fn(StructuredBinding, bool) -> Option<u32>;

const UNKNOWN_STRUCTURED_STRIDE: &str = "a structured buffer of unknown stride";

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container, lowering eligible <=32-thread Wave64 reductions to
/// `WaveSize(32)`.
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    translate_with_wave32(container, true)
}

/// Whether the host runs 64-lane compute waves (see [`set_host_wave64`]).
static HOST_WAVE64: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

thread_local! {
    /// A test's own answer to [`host_wave64`], over the process's.
    static HOST_WAVE64_HERE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tells translation whether the host's compute waves can hold 64 lanes
/// (`D3D12_FEATURE_DATA_D3D12_OPTIONS1::WaveLaneCountMax >= 64`; assumed
/// until told otherwise). Without them, a Wave64 compute program that needs
/// the console's lane layout runs each console wave as two 32-lane host
/// waves (`lower::split`) instead of declaring `WaveSize(64)`, which such a
/// host refuses. Part of what a cached translation depends on.
pub fn set_host_wave64(wave64: bool) {
    HOST_WAVE64.store(wave64, std::sync::atomic::Ordering::Relaxed);
}

/// Whether translation targets a host with 64-lane waves. A shader that
/// [`split_filter`] leaves out is translated as for one.
pub fn host_wave64() -> bool {
    HOST_WAVE64_HERE
        .with(std::cell::Cell::get)
        .unwrap_or_else(|| HOST_WAVE64.load(std::sync::atomic::Ordering::Relaxed))
        || SPLIT_LEFT_OUT.with(std::cell::Cell::get)
}

thread_local! {
    /// The shader being translated on this thread is not in [`split_filter`].
    static SPLIT_LEFT_OUT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `VICTORY_WAVE64_SPLIT_ONLY`, for bisecting the two-half lowering of
/// Wave64 programs (`lower::split`): a comma-separated list of shader
/// hashes (the 16 hex digits victory-gpu logs as `hash`) or prefixes of
/// them, and ranges of equal-length prefixes, `lo-hi` inclusive. Only the
/// shaders it matches are split; the others translate as for a 64-lane
/// host (`WaveSize(64)`, which a 32-lane host refuses). Unset or empty:
/// every shader. `none` matches none. E.g. `0-7` then `0-3` halves the
/// set by first digit; `5c09eb42b7a3c362,e958` names one shader and a prefix.
/// Part of what a cached translation depends on.
pub fn split_filter() -> &'static str {
    static FILTER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    FILTER.get_or_init(|| {
        std::env::var("VICTORY_WAVE64_SPLIT_ONLY")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
    })
}

/// Whether `filter` (see [`split_filter`]) takes the shader of hash `hash`.
fn split_filter_matches(filter: &str, hash: &str) -> bool {
    if filter.is_empty() {
        return true;
    }
    filter
        .split(',')
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .any(|e| match e.split_once('-') {
            Some((lo, hi)) if lo.len() == hi.len() && lo.len() <= hash.len() => {
                let h = &hash[..lo.len()];
                lo <= h && h <= hi
            }
            Some(_) => false,
            None => hash.starts_with(e),
        })
}

/// Wave64 programs of the console's ray-tracing BVH builder
/// (`XboxBvhBuilder.dll`), by the hash victory-gpu logs. Most of the
/// builder's passes do not translate (and three that do hang the host GPU),
/// so on a 32-lane host these four would run without the passes around
/// them: they write half-built structures the title then reads back as -1
/// and crashes on (runs 92-96). Left out, they stay refused like the rest
/// of the builder, as on hosts without the split.
// shortcut: the builder is kept off as a whole by naming its programs; a
// rule that drops every dispatch of a module whose chain is incomplete
// would cover other builds of the DLL too.
const SPLIT_LEFT_OUT_BVH: [&str; 4] = [
    "14394a322f6dd5c2",
    "28918b7150a7aad5",
    "e63d22c4def9ba6a",
    "f6eb6498409ca974",
];

/// Two 1024-thread classification programs of the title's tiled shadows,
/// by hash. Split, their output sends the tiled shadow passes that read it
/// to descriptors of the wrong kind, and the host GPU faults (NVIDIA "TEX
/// LAYOUT" exceptions, device hung) as loading starts (runs 99-124; with
/// these two skipped, runs 111-112 load).
// shortcut: refused as without the split, so the shadow passes they feed
// read nothing new. Which part of their lowering goes wrong (16 console
// waves, a wave leaving early, prefix sums in LDS) is not yet known.
const SPLIT_LEFT_OUT_TILES: [&str; 2] = ["b8f3f7979d6bbc50", "eae3498e6553afa8"];

/// More title programs whose split output is wrong, by hash, found by
/// bisecting runs with `VICTORY_WAVE64_SPLIT_ONLY`:
/// - `02384a60fda6f2d0`, the defragmenter of a VFX geometry heap (two
///   console waves walking the heap from both ends): split alone, the host
///   GPU faults as loading starts (run 138).
// shortcut: refused as without the split; the fault in their lowering is
// not yet found.
const SPLIT_LEFT_OUT_WRONG: &[&str] = &["02384a60fda6f2d0"];

/// Whether the two-half lowering takes the shader `container`: unless it
/// is in [`SPLIT_LEFT_OUT_BVH`], [`SPLIT_LEFT_OUT_TILES`] or
/// [`SPLIT_LEFT_OUT_WRONG`], when
/// [`split_filter`] is unset or matches it. What a cached translation of it
/// depends on, rather than the whole filter: bisecting then re-translates
/// nothing already seen.
pub fn split_takes(container: &[u8]) -> bool {
    let hash = blake3::hash(container).to_hex();
    split_takes_hash(&hash[..16])
}

/// [`split_takes`] for the shader whose container's BLAKE3 starts with the
/// 16 hex digits `hash`.
fn split_takes_hash(hash: &str) -> bool {
    !SPLIT_LEFT_OUT_BVH.contains(&hash)
        && !SPLIT_LEFT_OUT_TILES.contains(&hash)
        && !SPLIT_LEFT_OUT_WRONG.contains(&hash)
        && split_filter_matches(split_filter(), hash)
}

/// Runs `f` for the shader `container` with [`split_filter`] applied.
fn with_split_filter<T>(container: &[u8], f: impl FnOnce() -> T) -> T {
    let hash = blake3::hash(container).to_hex();
    with_split_selection(&hash[..16], f)
}

/// Runs `f` with the two-half selection translation applies to the shader
/// whose container's BLAKE3 starts with the 16 hex digits `hash`: one that
/// [`split_takes`] leaves out is lowered as for a 64-lane host
/// ([`host_wave64`] answers true inside `f`), as [`translate`] does. For
/// tools that call the emitter directly (`victory translate`).
pub fn with_split_selection<T>(hash: &str, f: impl FnOnce() -> T) -> T {
    let left_out = !probe::force_split() && !split_takes_hash(hash);
    let old = SPLIT_LEFT_OUT.with(|w| w.replace(left_out));
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SPLIT_LEFT_OUT.with(|w| w.set(self.0));
        }
    }
    let _restore = Restore(old);
    f()
}

/// Runs `f` with [`host_wave64`] answering `wave64` on this thread.
#[cfg(all(windows, any(test, feature = "host")))]
pub(crate) fn with_host_wave64<T>(wave64: bool, f: impl FnOnce() -> T) -> T {
    let old = HOST_WAVE64_HERE.with(|w| w.replace(Some(wave64)));
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            HOST_WAVE64_HERE.with(|w| w.set(self.0));
        }
    }
    let _restore = Restore(old);
    f()
}

thread_local! {
    /// The program being translated on this thread is for a host without
    /// barycentrics ([`translate_without_barycentrics`]).
    static WITHOUT_BARYCENTRICS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the program being translated on this thread reads barycentrics
/// and the values at each vertex from the inputs of `bary.rs` rather than
/// from the host's `SV_Barycentrics` and `GetAttributeAtVertex`.
pub(crate) fn without_barycentrics() -> bool {
    WITHOUT_BARYCENTRICS.with(std::cell::Cell::get)
}

/// Runs `f` with [`without_barycentrics`] answering true on this thread.
fn with_barycentric_fallback<T>(f: impl FnOnce() -> T) -> T {
    let old = WITHOUT_BARYCENTRICS.with(|w| w.replace(true));
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            WITHOUT_BARYCENTRICS.with(|w| w.set(self.0));
        }
    }
    let _restore = Restore(old);
    f()
}

/// As [`translate_with_wave32`], for a host without barycentrics: a pixel
/// program that would read `SV_Barycentrics` or `GetAttributeAtVertex`
/// reads the inputs [`barycentric_gs`] gives it instead (see `bary.rs`).
/// Every other program translates exactly as [`translate_with_wave32`]
/// translates it.
pub fn translate_without_barycentrics(
    container: &[u8],
    allow_wave32: bool,
) -> Result<HostShader, ShaderError> {
    with_barycentric_fallback(|| translate_with_wave32(container, allow_wave32))
}

/// As [`translate`], controlling whether eligible <=32-thread Wave64 compute
/// reductions may lower with `WaveSize(32)`. Pass `false` on adapters that
/// only support 64-lane waves (`WaveLaneCountMin > 32`).
pub fn translate_with_wave32(
    container: &[u8],
    allow_wave32: bool,
) -> Result<HostShader, ShaderError> {
    translate_with_wave32_for_hash(container, allow_wave32, "")
}

/// As [`translate_with_wave32`], providing the container's 16-hex-digit BLAKE3
/// prefix so allowlisted water-shading compute shaders (and
/// `VICTORY_WAVE_WIDE_ONLY`) can lower with Wave32 console flow.
pub fn translate_with_wave32_for_hash(
    container: &[u8],
    allow_wave32: bool,
    hash: &str,
) -> Result<HostShader, ShaderError> {
    translate_program(container, allow_wave32, hash, None).map(|(shader, _)| shader)
}

/// Translates using matching structured views for statically addressed table
/// entries with no usable stride in the shader. Returns the resource metadata
/// used for host binding alongside the translated shader. The resolver's
/// second argument is true for an access that requires a structured view.
pub fn translate_with_view_strides(
    container: &[u8],
    allow_wave32: bool,
    view_stride: ViewStride<'_>,
) -> Result<(HostShader, Vec<crate::UsedResource>), ShaderError> {
    translate_with_view_strides_for_hash(container, allow_wave32, "", view_stride)
}

/// As [`translate_with_view_strides`], providing the container's 16-hex-digit
/// BLAKE3 prefix for Wave32 allowlist selection.
pub fn translate_with_view_strides_for_hash(
    container: &[u8],
    allow_wave32: bool,
    hash: &str,
    view_stride: ViewStride<'_>,
) -> Result<(HostShader, Vec<crate::UsedResource>), ShaderError> {
    translate_program(container, allow_wave32, hash, Some(view_stride))
}

/// A precompiled compute module (`FSPX`): only GFX10 code, no DXIL to start from.
const FSPX_UNSUPPORTED: ShaderError = ShaderError::Unsupported {
    offset: 0,
    what: "FSPX compute module",
};

fn translate_program(
    container: &[u8],
    allow_wave32: bool,
    hash: &str,
    view_stride: Option<ViewStride<'_>>,
) -> Result<(HostShader, Vec<crate::UsedResource>), ShaderError> {
    with_split_filter(container, || {
        translate_filtered(container, allow_wave32, hash, view_stride)
    })
}

fn translate_filtered(
    container: &[u8],
    allow_wave32: bool,
    hash: &str,
    view_stride: Option<ViewStride<'_>>,
) -> Result<(HostShader, Vec<crate::UsedResource>), ShaderError> {
    if container.starts_with(crate::dxbc::Fspx::MAGIC) {
        return Err(FSPX_UNSUPPORTED);
    }
    let c = Container::parse(container)?;
    // A container with only a precompiled module: its programs when the
    // ACCC lowering is on (`crate::accc`), else refused as such.
    let native = match crate::container_programs(&c) {
        Err(ShaderError::Malformed("no DXIL part")) if c.fspx().is_some() => Err(FSPX_UNSUPPORTED),
        result => result,
    }?
    .into_iter()
    .find(|p| {
        matches!(
            p.stage,
            HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
        )
    })
    .ok_or(ShaderError::Unsupported {
        offset: 0,
        what: "a container without a vertex, pixel, compute or mesh program",
    })?;
    let shader = translate_native(&c, &native, allow_wave32, hash, None, view_stride)?;
    Ok((shader, native.used))
}

/// Emits an oversized mesh shader with only the varyings its pipeline's
/// translated pixel shader declares. With no pixel shader, keeps rasterizer
/// system values. Returns `None` if the original signature fits the budget.
pub fn translate_mesh_for_pixel(
    c: &Container<'_>,
    native: &NativeProgram,
    pixel_dxil: Option<&[u8]>,
) -> Result<Option<HostShader>, ShaderError> {
    let crate::native::StageIo::Ms(ms) = &native.io else {
        return Err(ShaderError::Malformed("no mesh program"));
    };
    let mesh = shader::Mesh {
        max_vertices: ms.max_vertices,
        max_primitives: ms.max_primitives,
        topology: ms.topology,
    };
    let (outputs, prims) = (signature_part(c, FourCC::OSG1)?, primitive_signature(c)?);
    if mesh.output_bytes(&outputs, &prims) <= shader::Mesh::MAX_OUTPUT_BYTES {
        return Ok(None);
    }
    let mut pixel_inputs = Vec::new();
    if let Some(dxil) = pixel_dxil {
        pixel_inputs = signature_part(&Container::parse(dxil)?, FourCC::ISG1)?;
    }
    translate_native(c, native, true, "", Some(&pixel_inputs), None).map(Some)
}

/// Translates the mesh program of a console shader container for a host
/// without mesh shaders.
pub fn translate_mesh_fallback(container: &[u8]) -> Result<MeshFallback, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| p.stage == HwStage::Ms)
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a mesh fallback of a container without a mesh program",
        })?;
    let (compute, buffer) = translate_stages(&c, &native, true, "", None, None, true)?;
    mesh_fallback(compute, buffer)
}

fn mesh_fallback(
    compute: HostShader,
    buffer: Option<MeshBuffer>,
) -> Result<MeshFallback, ShaderError> {
    let buffer = buffer.expect("a mesh program emitted into a buffer has its layout");
    Ok(MeshFallback {
        vertex: buffer.vertex_shader()?,
        compute,
        buffer,
    })
}

fn translate_native(
    c: &Container<'_>,
    native: &NativeProgram,
    allow_wave32: bool,
    hash: &str,
    pixel_inputs: Option<&[container::SigElement]>,
    view_stride: Option<ViewStride<'_>>,
) -> Result<HostShader, ShaderError> {
    translate_stages(
        c,
        native,
        allow_wave32,
        hash,
        pixel_inputs,
        view_stride,
        false,
    )
    .map(|e| e.0)
}

/// Runs the stages before emission and emits; `mesh_buffer` for a mesh
/// program on a host without mesh shaders (then with the buffer's layout).
fn translate_stages(
    c: &Container<'_>,
    native: &NativeProgram,
    allow_wave32: bool,
    hash: &str,
    pixel_inputs: Option<&[container::SigElement]>,
    view_stride: Option<ViewStride<'_>>,
    mesh_buffer: bool,
) -> Result<(HostShader, Option<MeshBuffer>), ShaderError> {
    // A compute program with root arguments past the 16 user SGPRs finds them
    // through the spill table `s[0:1]` points to.
    let p = ir::build_with(
        &native.patched_code(),
        native.wave64,
        native.vgprs,
        native.uses_spill_table(),
    )?;
    // Structurisation first: tracing per-lane descriptors needs it (its
    // failure is reported after the earlier stages').
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, native)?;
    let st = st?;
    let parts = Emission {
        c,
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        mesh_buffer,
    };
    emit_with_options(parts, allow_wave32, hash, pixel_inputs, view_stride)
}

/// Emits a program whose earlier stages already ran.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    emit_with_wave32(c, native, p, st, io, res, true)
}

/// As [`emit`], controlling whether eligible <=32-thread Wave64 compute
/// reductions may lower with `WaveSize(32)`.
pub fn emit_with_wave32(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    allow_wave32: bool,
) -> Result<HostShader, ShaderError> {
    emit_with_wave32_for_hash(c, native, p, st, io, res, allow_wave32, "")
}

/// As [`emit_with_wave32`], providing the container's 16-hex-digit BLAKE3
/// prefix for Wave32 console-flow allowlist selection.
#[allow(clippy::too_many_arguments)]
pub fn emit_with_wave32_for_hash(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    allow_wave32: bool,
    hash: &str,
) -> Result<HostShader, ShaderError> {
    let parts = Emission {
        c,
        native,
        p,
        st,
        io,
        res,
        mesh_buffer: false,
    };
    emit_with_options(parts, allow_wave32, hash, None, None).map(|e| e.0)
}

/// As [`emit`], for a host without barycentrics (see
/// [`translate_without_barycentrics`]).
pub fn emit_without_barycentrics(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    with_barycentric_fallback(|| emit(c, native, p, st, io, res))
}

/// Emits a mesh program whose earlier stages already ran, for a host
/// without mesh shaders. Nothing is pruned for a pixel shader: the buffer
/// has no output budget.
pub fn emit_mesh_fallback(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<MeshFallback, ShaderError> {
    if native.stage != HwStage::Ms {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "a mesh fallback of a program that is no mesh program",
        });
    }
    let parts = Emission {
        c,
        native,
        p,
        st,
        io,
        res,
        mesh_buffer: true,
    };
    let (compute, buffer) = emit_with_options(parts, true, "", None, None)?;
    mesh_fallback(compute, buffer)
}

#[derive(Clone, Copy)]
struct Emission<'a> {
    c: &'a Container<'a>,
    native: &'a NativeProgram,
    p: &'a ir::Program,
    st: &'a structure::Structured,
    io: &'a io::Io,
    res: &'a resource::Resources,
    /// A mesh program emitted for a host without mesh shaders.
    mesh_buffer: bool,
}

#[cfg(test)]
fn emit_with_view_strides(
    parts: Emission<'_>,
    allow_wave32: bool,
    view_stride: Option<ViewStride<'_>>,
) -> Result<HostShader, ShaderError> {
    emit_with_options(parts, allow_wave32, "", None, view_stride).map(|e| e.0)
}

fn emit_with_options(
    parts: Emission<'_>,
    allow_wave32: bool,
    hash: &str,
    pixel_inputs: Option<&[container::SigElement]>,
    view_stride: Option<ViewStride<'_>>,
) -> Result<(HostShader, Option<MeshBuffer>), ShaderError> {
    let result = emit_threaded(parts, allow_wave32, hash, pixel_inputs, view_stride);
    // What lowering does only for programs everything else refused (so that
    // every other program's code stays as it was) is a last chance. The
    // refusal reported is the first one.
    match result {
        Err(refusal @ ShaderError::Unsupported { .. }) if !crate::switch::no_last_chance() => {
            emit_program(parts, allow_wave32, hash, pixel_inputs, view_stride, true)
                .map_err(|_| refusal)
        }
        result => result,
    }
}

fn emit_threaded(
    parts: Emission<'_>,
    allow_wave32: bool,
    hash: &str,
    pixel_inputs: Option<&[container::SigElement]>,
    view_stride: Option<ViewStride<'_>>,
) -> Result<(HostShader, Option<MeshBuffer>), ShaderError> {
    let Emission {
        c,
        native,
        p,
        st,
        mesh_buffer,
        ..
    } = parts;
    let original = emit_program(parts, allow_wave32, hash, pixel_inputs, view_stride, false);
    // Retry unsupported programs only. Threading can separate an unknown
    // register value from its apparent use on a mutually exclusive path.
    if !matches!(original, Err(ShaderError::Unsupported { .. })) {
        return original;
    }
    let edges = threading::constant_edges(p, st);
    if edges.is_empty() {
        return original;
    }
    let p = ir::thread_edges(p, &edges)?;
    let st = structure::structure(&p)?;
    let io = io::raise(&p, native)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    emit_program(
        Emission {
            c,
            native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            mesh_buffer,
        },
        allow_wave32,
        hash,
        pixel_inputs,
        view_stride,
        false,
    )
}

/// `leftovers`: the last chance of [`emit_with_options`].
fn emit_program(
    parts: Emission<'_>,
    allow_wave32: bool,
    hash: &str,
    pixel_inputs: Option<&[container::SigElement]>,
    view_stride: Option<ViewStride<'_>>,
    leftovers: bool,
) -> Result<(HostShader, Option<MeshBuffer>), ShaderError> {
    let Emission {
        c,
        native,
        p,
        st,
        io,
        res,
        mesh_buffer,
    } = parts;
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = signature_part(c, FourCC::ISG1)?;
    let mut outputs = signature_part(c, FourCC::OSG1)?;
    if pixel {
        // Coverage inputs use intrinsics. Sample index is a signature
        // element without a register. Other rasterizer inputs are packed.
        inputs.retain(|e| !matches!(e.kind, 14 | 15));
        let mut next = inputs
            .iter()
            .filter_map(|e| e.row)
            .max()
            .map_or(0, |r| r + 1);
        for e in &mut inputs {
            if e.kind == 12 {
                e.row = None;
                e.col = 0;
            } else if matches!(e.kind, 3..=5 | 10 | 13) && e.row.is_none() {
                e.row = Some(next);
                e.col = 0;
                next += 1;
            }
        }
        for e in &mut outputs {
            match e.kind {
                14 | 17..=20 => {
                    e.row = None;
                    e.col = 0;
                }
                16 => e.row = Some(e.index),
                _ => {}
            }
        }
    }
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say.
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (4 | 5 | 10 | 13, _) if pixel => 1,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    if let Some(psv) = c.part(FourCC::PSV0) {
        match native.stage {
            HwStage::Vs => container::producer_interpolation(psv.data, 1, &mut outputs),
            HwStage::Ms => container::producer_interpolation(psv.data, 13, &mut outputs),
            _ => {}
        }
    }
    // With a conservative depth output (SV_DepthLessEqual, GreaterEqual) DXIL
    // wants the position centroid (or per sample), as DXC declares it then.
    if pixel && outputs.iter().any(|e| matches!(e.kind, 18 | 19)) {
        for e in inputs.iter_mut().filter(|e| e.kind == 3) {
            e.interp = 5;
        }
    }
    let prims = primitive_signature(c)?;
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (the title's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    let parts = lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs,
        outputs,
        prims,
        mesh_stride,
        mesh_buffer,
    };
    let lower_original =
        || lower::lower_with_wave32(&parts, allow_wave32, pixel_inputs, view_stride, leftovers);
    let l = if allow_wave32
        && console_flow_for_hash(hash)
        && native.stage == HwStage::Cs
        && native.wave64
    {
        use lower::WaveMode::{ConsoleFlow, Wave32};
        let lowered = match lower::lower_with_wave_mode(
            &parts,
            true,
            pixel_inputs,
            view_stride,
            Wave32,
            leftovers,
        ) {
            Err(ShaderError::Unsupported { what, .. }) if what == lower::CONSOLE_FLOW => {
                structure::structure_console(p).and_then(|console| {
                    let parts = lower::Parts {
                        st: &console,
                        ..parts.clone()
                    };
                    lower::lower_with_wave_mode(
                        &parts,
                        true,
                        pixel_inputs,
                        view_stride,
                        ConsoleFlow,
                        leftovers,
                    )
                })
            }
            result => result,
        };
        // The ordinary shader can still be usable on a host with Wave64.
        lowered.or_else(|reason| lower_original().map_err(|_| reason))?
    } else {
        lower_original()?
    };
    let shader = HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.make_comparison();
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        structured_srvs: l.structured_srvs,
    };
    note_heap_reads(&shader.dxil, l.heap_reads);
    Ok((shader, l.mesh_buffer))
}

fn signature_part(
    c: &Container<'_>,
    part: FourCC,
) -> Result<Vec<container::SigElement>, ShaderError> {
    c.part(part)
        .map_or_else(|| Ok(Vec::new()), |part| signature(part.data))
}

fn primitive_signature(c: &Container<'_>) -> Result<Vec<container::SigElement>, ShaderError> {
    // The console never exports a primitive id or cull flag as a
    // parameter (it culls with the primitive export).
    let mut prims = signature_part(c, FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    Ok(prims)
}

/// The rows of an `ISG1`/`OSG1`/`PSG1` part.
fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    if base > part.len() || count > (part.len() - base) / 32 {
        return Err(ShaderError::Malformed("signature records out of bounds"));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        if mask == 0 {
            continue;
        }
        let col = mask.trailing_zeros() as u8;
        let cols = mask.count_ones() as u8;
        if u16::from(mask) != ((1u16 << cols) - 1) << col || col + cols > 4 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a non-contiguous signature mask",
            });
        }
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            25 => 30, // cull primitive
            23 => 28, // barycentrics (a translated pixel shader's)
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col,
            cols,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    use super::bitcode::Attr;
    use super::lower::LOOP_LIMIT;
    use super::{parse_indirect_amplify, parse_loop_limit, parse_shader_opt};

    #[test]
    fn the_indirect_amplify_switch_defaults_on_and_disables_with_zero() {
        use std::ffi::OsStr;
        assert!(parse_indirect_amplify(None));
        assert!(parse_indirect_amplify(Some(OsStr::new("1"))));
        assert!(parse_indirect_amplify(Some(OsStr::new("on"))));
        assert!(parse_indirect_amplify(Some(OsStr::new(""))));
        assert!(!parse_indirect_amplify(Some(OsStr::new("0"))));
        assert!(!parse_indirect_amplify(Some(OsStr::new(" 0 "))));
        assert!(!parse_indirect_amplify(Some(OsStr::new("false"))));
        assert!(!parse_indirect_amplify(Some(OsStr::new("no"))));
        assert!(!parse_indirect_amplify(Some(OsStr::new("off"))));
    }

    #[test]
    fn fspx_modules_are_unsupported_not_missing_dxil() {
        let fspx = crate::dxbc::Fspx::assemble(&[0; 28], &[], &[0x55; 8], &[]);
        let wrapped = crate::dxbc::build(&[(crate::FourCC::ACCC, &fspx)]);
        let unsupported = crate::ShaderError::Unsupported {
            offset: 0,
            what: "FSPX compute module",
        };
        assert_eq!(super::translate(&fspx).unwrap_err(), unsupported);
        // In a container the ACCC lowering (`crate::accc`) reads it, unless
        // it is switched off; either way it is no missing DXIL part.
        let wrapped = super::translate(&wrapped).unwrap_err();
        assert_ne!(wrapped, crate::ShaderError::Malformed("no DXIL part"));
        if crate::switch::no_accc() {
            assert_eq!(wrapped, unsupported);
        }
        // Without an FSPX module it is still the missing DXIL part (when the
        // ACCC lowering is off; on, that lowering refuses it).
        let plain = crate::dxbc::build(&[(crate::FourCC::ACCC, b"other")]);
        let plain = super::translate(&plain).unwrap_err();
        assert_ne!(plain, unsupported);
        if crate::switch::no_accc() {
            assert_eq!(plain, crate::ShaderError::Malformed("no DXIL part"));
        }
    }

    #[test]
    fn the_shader_opt_switch_parses_or_falls_back() {
        let all = super::bitcode::opt::ALL;
        assert_eq!(parse_shader_opt(None), all);
        assert_eq!(parse_shader_opt(Some("0")), 0);
        assert_eq!(parse_shader_opt(Some(" 1 ")), 1);
        assert_eq!(parse_shader_opt(Some("0x3")), 3);
        assert_eq!(parse_shader_opt(Some("0xffffffff")), all);
        assert_eq!(parse_shader_opt(Some("on")), all);
    }

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
    }

    /// Heap indexing is read from the container's feature flags; without
    /// readable flags a shader is assumed to index heaps.
    #[test]
    fn heap_indexing_comes_from_the_feature_flags() {
        let shader = |parts: &[(FourCC, &[u8])]| HostShader {
            stage: HwStage::Ps,
            dxil: crate::dxbc::build(parts),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            structured_srvs: Vec::new(),
        };
        let with = |features: u64| shader(&[(FourCC::SFI0, &features.to_le_bytes())]);
        assert!(!with(0x4000).indexes_heaps());
        assert!(with(0x4000 | SFI0_RESOURCE_HEAP).indexes_heaps());
        assert!(with(SFI0_SAMPLER_HEAP).indexes_heaps());
        assert!(shader(&[]).indexes_heaps());
        assert!(shader(&[(FourCC::SFI0, &[0; 4])]).indexes_heaps());
    }
    use super::container::SigElement;
    #[cfg(windows)]
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    fn varying(name: &str, index: u32, row: u32) -> SigElement {
        SigElement {
            name: name.into(),
            index,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }
    }

    #[test]
    fn pixel_system_values_use_the_dxil_register_conventions() {
        let mut sample = varying("SV_SampleIndex", 0, 2);
        sample.kind = 12;
        sample.comp = 5;
        sample.cols = 1;
        sample.col = 2;
        let mut target = varying("SV_Target", 0, 3);
        target.kind = 16;
        let inputs = container::signature(&[sample], true);
        let outputs = container::signature(&[target], false);
        let source = crate::dxbc::build(&[(FourCC::ISG1, &inputs), (FourCC::OSG1, &outputs)]);
        let c = Container::parse(&source).unwrap();
        let words = [0x7E00_02F2u32, 0xF800_180F, 0, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 1,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                rsrc2: Default::default(),
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
        };
        let host = translate_native(&c, &native, true, "", None, None).unwrap();
        let c = Container::parse(&host.dxil).unwrap();
        let inputs = signature_part(&c, FourCC::ISG1).unwrap();
        let outputs = signature_part(&c, FourCC::OSG1).unwrap();
        assert_eq!((inputs[0].row, inputs[0].col), (None, 0));
        assert_eq!(outputs[0].row, Some(0));
        #[cfg(windows)]
        if let Some(mut v) = validator::Validator::for_test() {
            v.validate(&mut host.dxil.clone()).unwrap();
        }
    }

    /// `SPI_PS_INPUT_ADDR` bit 14 delivers the coverage mask in a VGPR. DXIL
    /// reads it with `dx.op.coverage.i32`, and `SV_Coverage` is not a pixel
    /// input there, so a console signature that lists it loses the element.
    #[test]
    fn pixel_sample_coverage_input_lowers_to_the_coverage_intrinsic() {
        let mut coverage = varying("SV_Coverage", 0, 0);
        coverage.kind = 14;
        coverage.comp = 5;
        coverage.row = None;
        coverage.cols = 1;
        let mut target = varying("SV_Target", 0, 3);
        target.kind = 16;
        let inputs = container::signature(&[coverage], true);
        let outputs = container::signature(&[target], false);
        let source = crate::dxbc::build(&[(FourCC::ISG1, &inputs), (FourCC::OSG1, &outputs)]);
        let c = Container::parse(&source).unwrap();
        // exp mrt0 v0, v0, v0, v0 (done); s_endpgm
        let words = [0xF800_180Fu32, 0, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 1,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 1 << 14,
                rsrc2: Default::default(),
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
        };
        let host = translate_native(&c, &native, true, "", None, None).unwrap();
        let c = Container::parse(&host.dxil).unwrap();
        let inputs = signature_part(&c, FourCC::ISG1).unwrap();
        assert!(inputs.iter().all(|e| e.kind != 14), "{inputs:?}");
        #[cfg(windows)]
        if let Some(mut v) = validator::Validator::for_test() {
            v.validate(&mut host.dxil.clone()).unwrap();
        }
    }

    #[test]
    fn oversized_mesh_keeps_only_exports_needed_by_its_pixel_shader() {
        use crate::native::{MeshIo, Usage, VsOutput};

        let mut position = varying("SV_Position", 0, 0);
        position.kind = 3;
        let mut outputs = vec![position];
        outputs.extend((1..=8).map(|n| varying("VALUE", n, n)));
        let prims = [varying("VALUE", 9, 0), varying("VALUE", 10, 1)];
        let osg1 = container::signature(&outputs, false);
        let psg1 = container::signature(&prims, false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1), (FourCC::PSG1, &psg1)]);
        let c = Container::parse(&bytes).unwrap();

        let mut words = vec![0x7E00_02F2u32, 0xF800_00CF, 0]; // v_mov, exp pos0
        for param in 0..10 {
            words.extend([0xF800_020F | (param << 4), 0]); // exp paramN
        }
        words.push(0xBF81_0000);
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64: false,
            vgprs: 1,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(MeshIo {
                outputs: (0..10)
                    .map(|param| VsOutput {
                        usage: Usage::Generic,
                        row: if param < 8 { param + 1 } else { param - 8 },
                        param: if param < 8 { param } else { param | 0x80 },
                        fp16: false,
                        pack2: false,
                    })
                    .collect(),
                clip_cntl: 0,
                threads: 32,
                max_vertices: 256,
                max_primitives: 32,
                topology: 2,
                lds_bytes: 0,
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        assert!(matches!(
            translate_native(&c, &native, true, "", None, None),
            Err(ShaderError::Unsupported {
                what: "a mesh output signature beyond the host's size limit",
                ..
            })
        ));
        // The pixel program declares all ten varyings but reads only the
        // eighth and tenth. The translated loads must follow their new IDs.
        let inputs: Vec<_> = (1..=10).map(|n| varying("value", n, n - 1)).collect();
        let mut target = varying("SV_Target", 0, 0);
        target.kind = 16;
        let isg1 = container::signature(&inputs, true);
        let target = container::signature(&[target], false);
        let pixel_bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &target)]);
        let pixel = NativeProgram {
            stage: HwStage::Ps,
            vgprs: 4,
            io: StageIo::Ps {
                inputs: [7, 9]
                    .into_iter()
                    .enumerate()
                    .map(|(attr, row)| crate::native::PsInput {
                        usage: Usage::Generic,
                        row,
                        mask: 15,
                        attr: attr as u8,
                        flat: true,
                        fp16: false,
                        pack2: false,
                        row2: row,
                    })
                    .collect(),
                input_addr: 0,
                rsrc2: Default::default(),
                db_shader_control: 0,
                patches: Vec::new(),
            },
            code: [
                0xC80A_0002u32, // v_interp_mov_f32 v2, p0, attr0.x
                0xC80E_0402,    // v_interp_mov_f32 v3, p0, attr1.x
                0xF800_180F,
                0x0302_0302, // exp mrt0, v2, v3, v2, v3
                0xBF81_0000,
            ]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
            ..native.clone()
        };
        let ps = translate_native(
            &Container::parse(&pixel_bytes).unwrap(),
            &pixel,
            true,
            "",
            None,
            None,
        )
        .unwrap()
        .dxil;
        let consumer = Container::parse(&ps).unwrap();
        assert_eq!(
            signature_part(&consumer, FourCC::ISG1)
                .unwrap()
                .iter()
                .map(|e| e.index)
                .collect::<Vec<_>>(),
            [8, 10]
        );
        let linked = translate_mesh_for_pixel(&c, &native, Some(&ps))
            .unwrap()
            .unwrap();
        let host = Container::parse(&linked.dxil).unwrap();
        for (part, expected) in [(FourCC::OSG1, &[0, 8][..]), (FourCC::PSG1, &[10][..])] {
            let rows = signature_part(&host, part).unwrap();
            assert_eq!(rows.iter().map(|e| e.index).collect::<Vec<_>>(), expected);
            let raw = host.part(part).unwrap().data;
            assert!((0..rows.len()).all(|i| raw[8 + 32 * i + 25] == 0)); // Written mask.
        }

        // Standalone emission can also save space when the extra
        // declarations have no corresponding native exports. With no
        // parameter table, exp param0 is the hardware's dummy export.
        let mut position_only = native;
        let StageIo::Ms(ms) = &mut position_only.io else {
            unreachable!();
        };
        ms.outputs.clear();
        position_only.code = words[..5]
            .iter()
            .chain(words.last())
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let standalone = translate_native(&c, &position_only, true, "", None, None).unwrap();
        let host = Container::parse(&standalone.dxil).unwrap();
        assert_eq!(signature_part(&host, FourCC::OSG1).unwrap().len(), 1);
        assert!(signature_part(&host, FourCC::PSG1).unwrap().is_empty());
        #[cfg(windows)]
        if let Some(mut v) = validator::Validator::for_test() {
            let mut ps = ps;
            v.validate(&mut ps).unwrap();
            v.validate(&mut linked.dxil.clone()).unwrap();
            v.validate(&mut standalone.dxil.clone()).unwrap();
        }
    }

    /// A pixel shader writing SV_StencilRef (channel y of `exp mrtz`).
    /// DXC sets ShaderFlags.StencilRef (bit 11) for a pixel shader whose
    /// output signature has SV_StencilRef, and the feature bit
    /// D3D_SHADER_FEATURE_STENCIL_REF (0x200) in SFI0; the validator checks
    /// both against the module.
    #[test]
    fn a_stencil_export_declares_the_stencil_ref_feature() {
        let words = [
            0x7E00_02F2u32, // v_mov_b32 v0, 1.0
            0x7E02_0285,    // v_mov_b32 v1, 5
            0xF800_1883,
            0x0000_0100, // exp mrtz v0, v1, off, off done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                rsrc2: Default::default(),
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, comp| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let osg1 =
            container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
        // dxil.dll, where there is one, checks the flags against the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        let host = Container::parse(&out.dxil).unwrap();
        let sfi0 = host.part(FourCC::SFI0).unwrap().data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_ne!(features & 0x200, 0, "SFI0 {features:#x}: no StencilRef");
    }

    /// Translates the pixel program `words` (no inputs or exports) with
    /// `rsrc2` and root `layout`, and returns the emitted DXIL.
    fn pixel_program(
        rsrc2: u32,
        layout: Vec<crate::native::RootParam>,
        used: Vec<crate::native::UsedResource>,
        words: &[u32],
    ) -> Result<Vec<u8>, ShaderError> {
        use crate::native::PsRsrc2;
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout,
            used,
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                rsrc2: PsRsrc2(rsrc2),
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let bytes = crate::dxbc::build(&[
            (FourCC::ISG1, &container::signature(&[], true)),
            (FourCC::OSG1, &container::signature(&[], false)),
        ]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native)?;
        Ok(emit(&c, &native, &p, &st, &io, &res)?.dxil)
    }

    fn table_param(user_data: u8) -> crate::native::RootParam {
        crate::native::RootParam {
            root: crate::native::RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data,
            slot: 0,
        }
    }

    /// `pixel_program` for a program with one user SGPR (s0, a descriptor
    /// table) that runs `body` and stores s4 to the table's buffer UAV, so
    /// the value of s4 is used.
    fn pixel_uav_store(rsrc2: u32, body: &[u32]) -> Result<Vec<u8>, ShaderError> {
        use crate::native::{ResourceType, RootType, UsedResource};
        let mut words = vec![
            0xBE86_0300, // s_mov_b32 s6, s0
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF408_0203,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        ];
        words.extend_from_slice(body);
        words.extend_from_slice(&[
            0x7E00_0204, // v_mov_b32 v0, s4
            0xE070_0000,
            0x8002_0000, // buffer_store_dword v0, off, s[8:11], 0
            0xBF81_0000, // s_endpgm
        ]);
        let used = vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }];
        pixel_program(rsrc2, vec![table_param(0)], used, &words)
    }

    fn sfi0_features(dxil: &[u8]) -> u64 {
        let host = Container::parse(dxil).unwrap();
        let sfi0 = host.part(FourCC::SFI0).unwrap().data;
        u64::from_le_bytes(sfi0.try_into().unwrap())
    }

    /// A POPS pixel program (`LOAD_COLLISION_WAVEID`) reads the collision
    /// wave id SGPR after the primitive mask and compares its fields with
    /// `src_pops_exiting_wave_id` to wait for the earlier waves that cover
    /// its pixels. Both translate as a wave that overlaps none, and the UAV
    /// is rasterizer ordered (D3D_SHADER_FEATURE_ROVS, 0x1000, in SFI0).
    #[test]
    fn a_pops_pixel_program_orders_its_uavs_on_the_host() {
        let body = [
            0x9003_9F02, // s_lshr_b32 s3, s2, 31 (the wave overlaps another)
            0x8104_03EF, // s_add_i32 s4, src_pops_exiting_wave_id, s3
            0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        ];
        let dxil = pixel_uav_store(1 << 1 | 1 << 25, &body).unwrap();
        let features = sfi0_features(&dxil);
        assert_ne!(features & 0x1000, 0, "SFI0 {features:#x}: no ROVs");
    }

    /// Without `LOAD_COLLISION_WAVEID` the same store is no ROV. s2 is then
    /// above the SGPRs the hardware initializes, and since no instruction
    /// writes it, reading it stays refused, as in compute programs.
    #[test]
    fn a_pixel_program_without_pops_keeps_plain_uavs() {
        let dxil = pixel_uav_store(1 << 1, &[0xBE84_0380]).unwrap(); // s_mov_b32 s4, 0
        let features = sfi0_features(&dxil);
        assert_eq!(features & 0x1000, 0, "SFI0 {features:#x}: ROVs");
        let body = [0x9004_9F02]; // s_lshr_b32 s4, s2, 31
        assert!(pixel_uav_store(1 << 1, &body).is_err());
        // src_pops_exiting_wave_id has no value there.
        let body = [0x8104_03EF]; // s_add_i32 s4, src_pops_exiting_wave_id, s3
        assert!(pixel_uav_store(1 << 1, &body).is_err());
    }

    /// The primitive mask (s1 with one user SGPR) arrives in every pixel
    /// program. A path that reuses its register and one that keeps it join
    /// in a phi, which the read after them takes only on the reusing path:
    /// the phi still needs a value for the mask.
    #[test]
    fn a_primitive_mask_joined_with_a_reused_register_translates() {
        let body = [
            0xBE84_0380, // s_mov_b32 s4, 0
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            0xBF85_0001, // s_cbranch_scc1 +1
            0xBE81_0387, // s_mov_b32 s1, 7
            0xBF85_0001, // s_cbranch_scc1 +1 (the same condition)
            0xBE84_0301, // s_mov_b32 s4, s1 (only where s1 is 7)
        ];
        pixel_uav_store(1 << 1, &body).unwrap();
    }

    /// A POPS pixel program (`LOAD_COLLISION_WAVEID`) reads the collision
    /// wave id SGPR after the primitive mask and compares its fields with
    /// `src_pops_exiting_wave_id`. Both translate as a wave that overlaps no
    /// earlier one.
    #[test]
    fn a_pops_pixel_program_translates() {
        let body = [
            0x9003_9F02, // s_lshr_b32 s3, s2, 31 (the wave overlaps another)
            0x8104_03EF, // s_add_i32 s4, src_pops_exiting_wave_id, s3
            0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        ];
        pixel_uav_store(1 << 1 | 1 << 25, &body).unwrap();
    }

    /// Without `LOAD_COLLISION_WAVEID` s2 is above the SGPRs the hardware
    /// initializes (one user SGPR and the primitive mask), and
    /// `src_pops_exiting_wave_id` has no value. No instruction writes s2, so
    /// reading it stays refused, as in compute programs.
    #[test]
    fn a_pixel_program_without_pops_refuses_its_inputs() {
        let plain = |body: &[u32]| pixel_uav_store(1 << 1, body);
        plain(&[0xBE84_0380]).unwrap(); // s_mov_b32 s4, 0
        assert!(plain(&[0x9004_9F02]).is_err()); // s_lshr_b32 s4, s2, 31
        assert!(plain(&[0x8104_03EF]).is_err()); // s_add_i32 s4, src_pops_exiting_wave_id, s3
    }

    /// The scratch wave offset follows the system SGPRs when `SCRATCH_EN`
    /// (bit 0) is set, so s2 is then among the initialized SGPRs: reading
    /// it is no longer the refusal of an SGPR above them, but of an input
    /// not modelled.
    #[test]
    fn the_scratch_wave_offset_is_an_initialized_pixel_sgpr() {
        let body = [0x9004_9F02]; // s_lshr_b32 s4, s2, 31
        let why = |rsrc2| format!("{:?}", pixel_uav_store(rsrc2, &body).unwrap_err());
        assert!(why(1 << 1).contains("above the initialized ones"));
        let scratch = why(1 << 1 | 1);
        assert!(!scratch.contains("above the initialized ones"), "{scratch}");
    }

    /// A pixel program loads a descriptor through the heap's base address,
    /// the pair s[2:3], with the heap stated at user-data dword 10 and the
    /// index in a root constant (s6). Without the pair belonging to the
    /// heap parameter, s2 and s3 were hardware SGPR inputs.
    #[test]
    fn a_pixel_program_reads_descriptors_through_the_heap_base_pair() {
        use crate::native::{RootParam, RootType};
        let param = |root, user_data, slot| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data,
            slot,
        };
        let layout = vec![
            param(RootType::Constants, 6, 0),
            param(RootType::ResourceHeap, 10, 1),
        ];
        let words = [
            0x8F08_8506, // s_lshl_b32 s8, s6, 5
            0xF408_0201,
            0x1000_0000, // s_load_dwordx4 s[8:11], s[2:3], s8
            0xE000_2000,
            0x8002_0400, // buffer_load_format_x v4, v0, s[8:11], 0 idxen
            0xBF81_0000, // s_endpgm
        ];
        pixel_program(1 << 1, layout, vec![], &words).unwrap();
    }

    /// Emits a pixel program that writes depth and stencil from `v0` and `v1`.
    fn emit_depth_stencil_ps(words: &[u32], wave64: bool) -> Result<(), crate::ShaderError> {
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                rsrc2: crate::native::PsRsrc2(0),
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, comp| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let osg1 =
            container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        // Only the Windows validator reads the DXIL.
        #[cfg_attr(not(windows), allow(unused_variables))]
        let out = emit(&c, &native, &p, &st, &io, &res)?;
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        Ok(())
    }

    /// A lane mask spilled into a VGPR lane (`v_writelane_b32`) and restored
    /// into another SGPR (`v_readlane_b32`, same constant lane) is the same
    /// mask: the select below followed it as an untracked one and refused.
    #[test]
    fn a_lane_mask_spilled_to_a_vgpr_lane_stays_a_mask() {
        let words = [
            0x7E00_02F2u32, // v_mov_b32 v0, 1.0
            0xD405_000A,    // v_cmp_lg_f32_e64 s10, v0, 0
            0x0001_0100,
            0xD761_0001, // v_writelane_b32 v1, s10, 0
            0x0001_000A,
            0xD760_000C, // v_readlane_b32 s12, v1, 0
            0x0001_0101,
            0xD501_0002, // v_cndmask_b32_e64 v2, 0, 1, s12
            0x0031_0280,
            0x7E02_0302, // v_mov_b32 v1, v2
            0xF800_1883, // exp mrtz v0, v1, off, off done vm
            0x0000_0100,
            0xBF81_0000, // s_endpgm
        ];
        emit_depth_stencil_ps(&words, false).unwrap();
    }

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask is
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A wave32 compute program with the whole-wave section the console
    /// compiler makes of an inclusive OR prefix (`WaveMultiPrefixBitOr`
    /// over every lane, plus the lane's own value): row_shr DPP steps
    /// inside each row of 16, then lane 15 joined into the upper row. Each
    /// lane stores its result at its own offset of a raw UAV.
    fn or_prefix_cs() -> NativeProgram {
        const fn sop1(op: u32, sdst: u32, src0: u32) -> u32 {
            0xBE80_0000 | sdst << 16 | op << 8 | src0
        }
        const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
            op << 25 | vdst << 17 | vsrc1 << 9 | src0
        }
        const fn vop3(op: u32, vdst: u32, src0: u32, src1: u32, src2: u32) -> [u32; 2] {
            [0x35 << 26 | op << 16 | vdst, src0 | src1 << 9 | src2 << 18]
        }
        /// v_or_b32_dpp v1, v1, v1 row_shr:n bound_ctrl:1.
        const fn or_shr(n: u32) -> [u32; 2] {
            [
                vop2(0x1C, 1, 0xFA, 1),
                1 | (0x110 | n) << 8 | 1 << 19 | 0xF << 24 | 0xF << 28,
            ]
        }
        let (exec_lo, v0, v1) = (126, 256, 257);
        let mut words = vec![
            sop1(0x03, 1, 0x80), // s_mov_b32 s1, 0
        ];
        // s_load_dwordx4 s[8:11], s[0:1], 0
        words.extend([0x3D << 26 | 0x02 << 18 | 8 << 6, 125 << 25]);
        words.push(sop1(0x3D, 2, 193)); // s_or_saveexec_b32 s2, -1
        words.extend(vop3(0x101, 1, 0x80, v0, 2)); // v_cndmask_b32 v1, 0, v0, s2
        for n in [1, 2, 4, 8] {
            words.extend(or_shr(n));
        }
        words.extend(vop3(0x360, 3, v1, 128 + 15, 0)); // v_readlane_b32 s3, v1, 15
        words.extend([sop1(0x03, exec_lo, 0xFF), 0xFFFF_0000]); // s_mov_b32 exec_lo, 0xffff0000
        words.push(vop2(0x1C, 1, 3, 1)); // v_or_b32 v1, s3, v1
        words.push(sop1(0x03, exec_lo, 2)); // s_mov_b32 exec_lo, s2
        words.push(vop2(0x1A, 2, 0x82, 0)); // v_lshlrev_b32 v2, 2, v0
        // buffer_store_dword v1, v2, s[8:11], 0 offen
        words.extend([
            0x38 << 26 | 0x1C << 18 | 1 << 12,
            0x80 << 24 | 2 << 16 | 1 << 8 | 2,
        ]);
        words.push(0xBF81_0000); // s_endpgm
        NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![crate::native::RootParam {
                root: crate::native::RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![crate::native::UsedResource {
                kind: crate::native::ResourceType::Uav,
                root: crate::native::RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 1, // one user SGPR: s0
                threads: [32, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    /// The OR prefix becomes `WaveMultiPrefixOp` (DXIL has no plain bitwise
    /// prefix) instead of failing the translation.
    #[test]
    fn a_bitwise_wave_prefix_is_a_multi_prefix() {
        let native = or_prefix_cs();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
        let shapes: Vec<_> = (0..p.blocks.len())
            .flat_map(|b| wave::sections(&p, b).into_values())
            .flat_map(|s| s.shapes.into_values())
            .collect();
        assert!(
            shapes
                .iter()
                .any(|s| matches!(s, wave::Shape::Prefix(wave::Op::Or, _, _, 0, true))),
            "{shapes:?}"
        );
        assert!(!out.dxil.is_empty());
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
    }

    /// A pixel shader writing a constant colour.
    #[cfg(windows)]
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            structured_srvs: Vec::new(),
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                structured_srvs: Vec::new(),
            })
        };
        let ps = file("VICTORY_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("VICTORY_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // vkd3d-proton does not enforce the DXIL container checksum. The
        // successful pipeline above is still exercised under Wine.
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: both names are NUL terminated, and the module is checked
        // before looking up its export. The export is never called.
        let under_wine = unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        };
        if under_wine {
            return;
        }
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// A pixel shader interpolating at sample positions (PERSP_SAMPLE
    /// barycentrics, `SPI_PS_INPUT_ADDR` bit 0) runs at sample frequency:
    /// DXC's PSV0 says so in PSInfo.SampleFrequency (runtime info byte 1)
    /// whenever an input is interpolated per sample.
    #[test]
    fn sample_interpolated_inputs_set_psv_sample_frequency() {
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let psv_of = |input_addr: u32| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr,
                    rsrc2: Default::default(),
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half: false,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let out = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
            // dxil.dll, where there is one, checks PSV0 against the module.
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = super::Container::parse(&out.dxil).unwrap();
            host.part(super::FourCC::PSV0).unwrap().data.to_vec()
        };
        // Runtime info starts after its size; PSInfo is {DepthOutput, SampleFrequency}.
        let center = psv_of(1 << 1);
        assert_eq!(center[4 + 1], 0, "center interpolation is per pixel");
        let sample = psv_of(1 << 0);
        assert_eq!(sample[4 + 1], 1, "PSInfo.SampleFrequency");
    }

    /// A `v_interp_p1*` that no second half completes is lowered only when
    /// nothing but second halves reads its register: a first half's result
    /// read as data (here exported) is refused, not exported as zero.
    #[test]
    fn a_first_interpolation_half_is_never_read_as_data() {
        let lower = |words: &[u32]| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    rsrc2: Default::default(),
                    input_addr: 1 << 1,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half: false,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            super::emit(&c, &native, &p, &st, &io, &res).map(|_| ())
        };
        // exp mrt0, v2, v2, v2, v2 done vm; s_endpgm
        let export = [0xF800_180F, 0x0202_0202, 0xBF81_0000];
        // An extra first half into v3 that nothing reads still lowers.
        let unread = [&[0xC808_0000, 0xC80C_0000, 0xC809_0001][..], &export].concat();
        assert_eq!(lower(&unread), Ok(()));
        // One a plain write overwrites before anything reads it lowers too, but
        // only as the last chance (the write reads the old value as well).
        let overwritten = [
            &[0xC808_0000, 0xC80C_0000, 0xC809_0001, 0x7E06_0280][..], // v_mov_b32 v3, 0
            &export,
        ]
        .concat();
        assert_eq!(lower(&overwritten), Ok(()));
        // A first half exported without its second half is refused.
        let lone = [&[0xC808_0000][..], &export].concat();
        let refused = lower(&lone).unwrap_err();
        assert!(
            refused
                .to_string()
                .contains("the first half of an interpolation"),
            "{refused}"
        );
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                rsrc2: Default::default(),
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res);
    }

    #[test]
    fn scalar_bitcmp_and_sext_execute_on_gpu() {
        let words: &[u32] = &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xBF0C_8000, // s_bitcmp0_b32 s0, 0
            0x850C_8081, // s_cselect_b32 s12, 1, 0
            0xBF0D_A100, // s_bitcmp1_b32 s0, 33
            0x850D_8081, // s_cselect_b32 s13, 1, 0
            0xBF0E_A200, // s_bitcmp0_b64 s[0:1], 34
            0x850E_8081, // s_cselect_b32 s14, 1, 0
            0xBF0F_BF00, // s_bitcmp1_b64 s[0:1], 63
            0x850F_8081, // s_cselect_b32 s15, 1, 0
            0xBF0F_BFC1, // s_bitcmp1_b64 -1, 63
            0x8510_8081, // s_cselect_b32 s16, 1, 0
            0xBE91_1902, // s_sext_i32_i8 s17, s2
            0xBE92_1A03, // s_sext_i32_i16 s18, s3
            0xBF0D_9F00, // s_bitcmp1_b32 s0, 31
            0xBE94_1903, // s_sext_i32_i8 s20, s3 (must preserve SCC)
            0x8513_8014, // s_cselect_b32 s19, s20, 0
            0x7E00_020C, // v_mov_b32 v0, s12
            0x7E02_020D, // v_mov_b32 v1, s13
            0x7E04_020E, // v_mov_b32 v2, s14
            0x7E06_020F, // v_mov_b32 v3, s15
            0x7E08_0280, // v_mov_b32 v4, 0
            0xE078_1000,
            0x8002_0004, // buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen
            0x7E00_0210, // v_mov_b32 v0, s16
            0x7E02_0211, // v_mov_b32 v1, s17
            0x7E04_0212, // v_mov_b32 v2, s18
            0x7E06_0213, // v_mov_b32 v3, s19
            0x7E08_0290, // v_mov_b32 v4, 16
            0xE078_1000,
            0x8002_0004, // buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        let native = crate::NativeProgram {
            stage: crate::HwStage::Cs,
            wave64: false,
            vgprs: 5,
            layout: vec![
                crate::RootParam {
                    root: crate::RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                crate::RootParam {
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![crate::UsedResource {
                kind: crate::ResourceType::Uav,
                root: crate::RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 4,
                slot: 1,
            }],
            io: crate::native::StageIo::Cs(crate::native::ComputeIo {
                rsrc2: 5 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let l = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            mesh_buffer: false,
        })
        .unwrap();
        #[cfg_attr(not(windows), allow(unused_mut, unused_variables))]
        let mut dxil = l.dxil;
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxil).unwrap();
        }
        #[cfg(windows)]
        if let Ok(host) = super::host::Host::new(false) {
            let cs = super::HostShader {
                stage: crate::HwStage::Cs,
                dxil,
                user_data: l.user_data,
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs,
                root_srvs: l.root_srvs,
                root_uavs: l.root_uavs,
                structured_srvs: Vec::new(),
            };
            let (root, pso) = host.compute_pipeline(&cs).unwrap();
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &super::host::Dispatch {
                        roots: vec![(
                            0,
                            super::host::Root::Constants(vec![
                                0x8000_0002,
                                0x8000_0004,
                                0x1234_5680,
                                0x1234_807F,
                                0,
                            ]),
                        )],
                        views: &[super::host::View::RawUav(64)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            assert_eq!(
                &out[0][..8],
                &[1, 1, 0, 1, 1, 0xFFFF_FF80, 0xFFFF_807F, 0x7F]
            );
        }
    }
}

#[cfg(test)]
mod atomic_tests {
    use super::*;
    use crate::native::{ComputeIo, RootParam, StageIo};

    #[test]
    fn heap_texture_atomic_widths() {
        // A synthetic compute shader loads a 2D UAV from a root table,
        // compares or adds at (0, 0), then uses the returned register(s).
        for (opcode, mask, wide) in [
            (16u32, 3, false),
            (16, 15, true),
            (17, 1, false),
            (17, 3, true),
        ] {
            let words = [
                0xBE81_0380, // s_mov_b32 s1, 0 (table address high)
                0xF40C_0200,
                0xFA00_0000, // s_load_dwordx8 s[8:15], s[0:1], 0
                0x7E00_0280,
                0x7E02_0280, // v0, v1 = coordinates
                0x7E04_0281,
                0x7E06_0282,
                0x7E08_0283,
                0x7E0A_0284, // v2..v5 = 1..4
                0xF000_2008 | opcode << 18 | mask << 8,
                0x0002_0200,
                // Use the old value in a second atomic at the same location.
                0xF044_0008 | (if wide { 3 } else { 1 }) << 8,
                0x0002_0200,
                0xBF81_0000,
            ];
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 6,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 2,
                    threads: [32, 1, 1],
                }),
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = ir::build(&native.code, false, native.vgprs).unwrap();
            let bytes = crate::dxbc::build(&[]);
            let shader = emit(
                &Container::parse(&bytes).unwrap(),
                &native,
                &p,
                &structure::structure(&p).unwrap(),
                &io::raise(&p, &native).unwrap(),
                &resource::track(
                    &p,
                    &native.layout,
                    native.stage,
                    structure::structure(&p).ok().as_ref(),
                )
                .unwrap(),
            )
            .unwrap();
            let container = Container::parse(&shader.dxil).unwrap();
            let features = u64::from_le_bytes(
                container
                    .part(FourCC::SFI0)
                    .unwrap()
                    .data
                    .try_into()
                    .unwrap(),
            );
            let required = 0x1040_8000; // heap atomics, typed atomics, Int64Ops
            assert_eq!(features & required, if wide { required } else { 0 });
            #[cfg(windows)]
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                v.validate(&mut shader.dxil.clone()).unwrap();
            }
        }
    }

    #[test]
    fn wide_lowering_allowlist_opt_out_and_exclusions() {
        use std::ffi::OsStr;

        let checked = [
            "6871d6e4b3ac9a10",
            "926c46bab3b86bd6",
            "a10179ab7eda0dc3",
            "369cbe906ea72764",
            "2cc4c197ee29e291",
        ];
        for hash in checked {
            assert!(
                wide_lowering_enabled(hash, None, false, &[], false),
                "{hash} must be enabled in default water allowlist"
            );
            assert!(
                !wide_lowering_enabled(hash, Some(OsStr::new("0")), false, &[], false),
                "{hash} must opt out with VICTORY_WAVE_WIDE=0"
            );
            assert!(
                !wide_lowering_enabled(hash, Some(OsStr::new("0")), true, &[], true),
                "{hash} must opt out with VICTORY_WAVE_WIDE=0 even with console flow / wide_here"
            );
            assert!(
                wide_lowering_enabled(hash, Some(OsStr::new("1")), false, &[], false),
                "{hash} with explicit VICTORY_WAVE_WIDE=1"
            );
        }

        // Explicitly excluded water hashes that caused black rectangles around the boat.
        for excluded in ["5dc3b25986622949", "6f225b641c5b0708"] {
            assert!(
                !wide_lowering_enabled(excluded, None, false, &[], false),
                "{excluded} must stay excluded from default water allowlist"
            );
        }

        for other in ["", "f133e35efed2a48a", "0123456789abcdef"] {
            assert!(
                !wide_lowering_enabled(other, None, false, &[], false),
                "{other} must stay off by default"
            );
            assert!(
                wide_lowering_enabled(other, None, true, &[], false),
                "{other} with VICTORY_WAVE64_CONSOLE_FLOW=1"
            );
            assert!(
                wide_lowering_enabled(other, Some(OsStr::new("1")), false, &[], false),
                "{other} with explicit VICTORY_WAVE_WIDE=1"
            );
        }

        let only = parse_wave_wide_only(Some(" A10179AB , deadbeef , "));
        assert_eq!(only, vec!["a10179ab".to_owned(), "deadbeef".to_owned()]);
        assert!(matches_wave_wide_only("a10179ab7eda0dc3", &only));
        assert!(matches_wave_wide_only("DEADBEEF00112233", &only));
        assert!(!matches_wave_wide_only("6871d6e4b3ac9a10", &only));
        assert!(!matches_wave_wide_only("", &only));

        // When VICTORY_WAVE_WIDE_ONLY is set, it overrides the default allowlist.
        assert!(wide_lowering_enabled(
            "a10179ab7eda0dc3",
            None,
            false,
            &only,
            false
        ));
        assert!(wide_lowering_enabled(
            "deadbeef12345678",
            None,
            false,
            &only,
            false
        ));
        assert!(!wide_lowering_enabled(
            "6871d6e4b3ac9a10",
            None,
            false,
            &only,
            false
        ));
        assert!(wide_lowering_enabled("", None, false, &only, true));
        assert!(!wide_lowering_enabled(
            "a10179ab7eda0dc3",
            Some(OsStr::new("0")),
            false,
            &only,
            true
        ));

        // `with_wave_wide` scopes and restores its previous state.
        assert!(!WIDE_HERE.get());
        with_wave_wide(true, || {
            assert!(WIDE_HERE.get());
            assert!(wide_lowering_enabled(
                "0123456789abcdef",
                None,
                false,
                &[],
                WIDE_HERE.get()
            ));
            with_wave_wide(false, || {
                assert!(!WIDE_HERE.get());
            });
            assert!(WIDE_HERE.get());
        });
        assert!(!WIDE_HERE.get());
    }

    #[test]
    fn wave32_allowlist_lowers_a10179ab_and_keeps_excluded_wave64() {
        // 64-thread Wave64 compute shader with cross-half `v_readlane_b32` and
        // LDS: stays WaveSize(64) under `WaveMode::Original`, and lowers to
        // WaveSize(32) via console flow when allowlisted (e.g. a10179ab7eda0dc3).
        let words = [
            0xBE81_0380u32, // s_mov_b32 s1, 0
            0xF408_0200,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null
            0x3408_0082, // v_lshlrev_b32 v4, 2, v0
            0xD834_0000,
            0x0000_0004, // ds_write_b32 v4, v0
            0x3A0A_00A0, // v_xor_b32 v5, 32, v0
            0x340C_0A82, // v_lshlrev_b32 v6, 2, v5
            0xD8D8_0000,
            0x0200_0006, // ds_read_b32 v2, v6
            0xD760_0002,
            0x0001_5100, // v_readlane_b32 s2, v0, 40
            0x4A04_0402, // v_add_nc_u32 v2, s2, v2
            0xE070_1000,
            0x8002_0204, // buffer_store_dword v2, v4, s[8:11], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 7,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![crate::UsedResource {
                kind: crate::ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: (1 << 1) | (1 << 15), // 512 bytes LDS
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let bytes = crate::dxbc::build(&[]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.code, true, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();

        let wave_bounds = |dxil: &[u8]| {
            let parsed = Container::parse(dxil).unwrap();
            let psv = parsed.part(FourCC::PSV0).unwrap().data;
            let min = u32::from_le_bytes(psv[20..24].try_into().unwrap());
            let max = u32::from_le_bytes(psv[24..28].try_into().unwrap());
            (min, max)
        };

        // Unless opted out via VICTORY_WAVE_WIDE=0, a10179ab7eda0dc3 lowers to WaveSize(32).
        let lowered =
            emit_with_wave32_for_hash(&c, &native, &p, &st, &io, &res, true, "a10179ab7eda0dc3")
                .unwrap();
        if wave_wide_env().is_some_and(|v| v == "0") {
            assert_eq!(
                wave_bounds(&lowered.dxil),
                (64, 64),
                "VICTORY_WAVE_WIDE=0 must opt out of Wave32 lowering for a10179ab7eda0dc3"
            );
        } else if wave_wide_only_list().is_empty() {
            assert_eq!(wave_bounds(&lowered.dxil), (32, 32));
        }

        // 64-lane-only adapter (`allow_wave32 = false`) keeps WaveSize(64).
        let wave64_only =
            emit_with_wave32_for_hash(&c, &native, &p, &st, &io, &res, false, "a10179ab7eda0dc3")
                .unwrap();
        assert_eq!(wave_bounds(&wave64_only.dxil), (64, 64));

        // Excluded water hashes stay WaveSize(64) when global env overrides are unset.
        if wave_wide_env().is_none() && !wave64_console_flow() && wave_wide_only_list().is_empty() {
            for excluded in ["5dc3b25986622949", "6f225b641c5b0708"] {
                let s = emit_with_wave32_for_hash(&c, &native, &p, &st, &io, &res, true, excluded)
                    .unwrap();
                assert_eq!(
                    wave_bounds(&s.dxil),
                    (64, 64),
                    "{excluded} must stay Wave64"
                );
            }
        }
    }
}
