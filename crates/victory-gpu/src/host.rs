//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{
    DXGI_STATUS_OCCLUDED, HANDLE, HWND, LPARAM, LRESULT, RECT, WAIT_OBJECT_0, WPARAM,
};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{HRESULT, Interface, PCWSTR, w};

use crate::present_compositor::PresentCompositor;

#[path = "host_prep.rs"]
mod prep;
#[path = "host_relay.rs"]
mod relay;
#[cfg(test)]
pub(crate) use relay::relay_no_card_fetch_watermark;
pub(crate) use relay::*;

#[cfg(all(test, feature = "diagnostics"))]
mod present_tests;

/// A reusable direct or copy command allocator and command list pair.
pub(super) type Prelude = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;
/// Default maximum number of detached host lists kept for reuse after pending
/// resets (overridable via `VICTORY_LIST_POOL_MOST`).
const LIST_POOL_CAP: usize = 512;

/// Maximum number of detached host lists kept for reuse
/// (`VICTORY_LIST_POOL_MOST`, default 512).
fn list_pool_cap() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_LIST_POOL_MOST")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(LIST_POOL_CAP)
    })
}

/// Maximum reusable readback buffer kept cached for ReBAR UAV copy-backs.
/// One-off readbacks larger than this are dropped right after use.
const REBAR_READBACK_KEEP_CAP: u64 = 256 << 20;

/// Minimum and maximum cached byte sizes for the ReBAR GPU hash-compare flags
/// (`DEFAULT` UAV + `READBACK`, 4 bytes per 64 KiB piece: 16 KiB = 4,096
/// pieces = 256 MiB; 4 MiB = 1,048,576 pieces = 64 GiB per submission).
const REBAR_FLAGS_MIN_BYTES: u64 = 16 << 10;
const REBAR_FLAGS_MAX_BYTES: u64 = 4 << 20;

const REBAR_HASH_SOURCE: &str = include_str!("../shaders/rebar_hash.hlsl");

struct RebarHashPipeline {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 root signatures and pipeline states are free-threaded.
unsafe impl Send for RebarHashPipeline {}
// SAFETY: as above.
unsafe impl Sync for RebarHashPipeline {}

fn rebar_hash_pipeline(device: &ID3D12Device) -> Option<&'static RebarHashPipeline> {
    static PIPELINE: OnceLock<Option<RebarHashPipeline>> = OnceLock::new();
    PIPELINE
        .get_or_init(|| match make_rebar_hash_pipeline(device) {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "no ReBAR hash-compare pipeline; falling back to full-range readbacks"
                );
                None
            }
        })
        .as_ref()
}

fn make_rebar_hash_pipeline(device: &ID3D12Device) -> windows::core::Result<RebarHashPipeline> {
    let bytes = crate::hlsl::compile_compute("rebar_hash", REBAR_HASH_SOURCE)?;
    let bytes = &bytes[..];
    // SAFETY: plain object creation; the root signature is embedded in the shader.
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
        Ok(RebarHashPipeline { root, pso })
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RebarHashStats {
    pub(crate) compared_pieces: u64,
    pub(crate) changed_pieces: u64,
    pub(crate) post_readback_calls: u64,
    pub(crate) post_readback_bytes: u64,
}

/// Reusable command lists and readback/hash-flag buffers for DMAing changed
/// GPU-written ranges of ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) shadow buffers
/// into cached system RAM.
#[derive(Default)]
struct RebarReadback {
    list: Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    buffer: Option<(ID3D12Resource, usize, u64)>,
    pre_hash_list: Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    post_compare_list: Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    flags_uav: Option<(ID3D12Resource, u64, u64)>,
    flags_readback: Option<(ID3D12Resource, usize, u64)>,
}

// SAFETY: D3D12 objects are free-threaded; mapped pointers are read only
// under `Host::rebar_readback` after the corresponding GPU fence has completed.
unsafe impl Send for RebarReadback {}

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    adapter: Option<IDXGIAdapter3>,
    rebar_scratch: bool,
    rebar_shadow_budget: u64,
    rebar_loose_budget: u64,
    rebar_readback: Mutex<RebarReadback>,
    queue: ID3D12CommandQueue,
    /// Optional dedicated `D3D12_COMMAND_LIST_TYPE_COPY` queue for relay DMA
    /// readbacks (`VICTORY_RELAY_COPY_QUEUE=0` disables).
    copy_queue: Option<ID3D12CommandQueue>,
    pub(crate) fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    /// Why no presenter was made (or resized), and when to try again: a run
    /// with no desktop to present on would otherwise ask DXGI at the frame
    /// rate.
    present_retry: Mutex<Retry>,
    present_compositor_retry: Mutex<Retry>,
    /// Detached host lists whose submissions finished, ready to reuse.
    free_lists: Mutex<Vec<HostList>>,
    /// GPU-writable memory for mesh programs run without mesh shaders.
    mesh_scratch: Mutex<MeshScratch>,
    #[cfg(test)]
    reused_lists: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    d3d12_errors: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    rebar_hash_compared_pieces: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    rebar_hash_changed_pieces: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    rebar_post_readback_calls: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    rebar_post_readback_bytes: std::sync::atomic::AtomicU64,
}

/// A presenter that could not be made, and when to try again. Without this,
/// a run behind a locked screen or a sleeping display retried every frame:
/// 1,654 `no swap chain` lines in 60 seconds, and one DXGI call per frame,
/// which buries the log and costs a submission's worth of work each time.
struct Retry {
    attempts: u32,
    /// The next attempt is due at (or after) this instant; `None` with
    /// `attempts == 0` means nothing has failed.
    next: Option<std::time::Instant>,
}

impl Retry {
    const fn new() -> Self {
        Retry {
            attempts: 0,
            next: None,
        }
    }

    /// Whether an attempt is due at `now`.
    fn due(&self, now: std::time::Instant) -> bool {
        self.next.is_none_or(|at| now >= at)
    }

    /// Records a failed attempt at `now` and when the next one may run.
    fn failed(&mut self, now: std::time::Instant) {
        self.attempts = self.attempts.saturating_add(1);
        self.next = Some(now + retry_delay(self.attempts));
    }

    /// Clears the failure state after an attempt worked, returning how many
    /// attempts had failed before it.
    fn reset(&mut self) -> u32 {
        let attempts = self.attempts;
        *self = Retry::new();
        attempts
    }
}

/// How long to wait before retry number `attempts`: 100 ms, doubling, up to
/// one second, so a state that clears still recovers within a second.
fn retry_delay(attempts: u32) -> std::time::Duration {
    const FIRST_MS: u64 = 100;
    const MOST_MS: u64 = 1_000;
    let doubling = FIRST_MS << attempts.saturating_sub(1).min(4);
    std::time::Duration::from_millis(doubling.min(MOST_MS))
}

/// Which step of making a presenter failed, so the log says whether the
/// window, the swap chain or its command list was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PresentStep {
    Window,
    Factory,
    SwapChain,
    List,
}

impl PresentStep {
    fn name(self) -> &'static str {
        match self {
            PresentStep::Window => "window",
            PresentStep::Factory => "DXGI factory",
            PresentStep::SwapChain => "swap chain",
            PresentStep::List => "copy list",
        }
    }
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

pub(crate) use crate::retire::*;

/// Milliseconds since the first call.
pub(crate) fn now_ms() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

/// Available host physical RAM in bytes reported by Windows, or `None` if the
/// query fails.
pub(crate) fn available_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data filled in by `GlobalMemoryStatusEx`.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: valid out pointer with `dwLength` matching `MEMORYSTATUSEX`.
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    ok.then_some(m.ullAvailPhys)
}

/// Effective idle timeout (in milliseconds) for RAM-backed staging pools,
/// scaled down under physical RAM pressure.
pub(crate) fn ram_idle_ms(avail_phys: Option<u64>) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    const NORMAL_MS: u64 = 10_000;
    const MODERATE_MS: u64 = 5_000;
    const HIGH_MS: u64 = 2_000;
    let Some(avail) = avail_phys else {
        return NORMAL_MS;
    };
    if avail < 2 * GIB {
        HIGH_MS
    } else if avail < 4 * GIB {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Detailed preparation counters for `host_prep` (`watch` ask-ahead us, fill
/// us, fill count, fill bytes).
pub(super) const PREP_WATCH_US: usize = 0;
pub(super) const PREP_FILL_US: usize = 1;
pub(super) const PREP_FILL_COUNT: usize = 2;
pub(super) const PREP_FILL_BYTES: usize = 3;
pub(super) static PREPARING_COPIES: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

pub(super) fn wide_refresh() -> bool {
    crate::shadow::wide_copy()
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

#[cfg(test)]
std::thread_local! {
    static FAIL_PANICS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) struct FailPanicsGuard(bool);

#[cfg(test)]
impl Drop for FailPanicsGuard {
    fn drop(&mut self) {
        FAIL_PANICS.with(|f| f.set(self.0));
    }
}

/// Test helper: causes [`fail`] on the current thread to panic (catchable via
/// `std::panic::catch_unwind`) instead of aborting the test process.
#[cfg(test)]
pub(crate) fn fail_panics_for_test() -> FailPanicsGuard {
    let prev = FAIL_PANICS.with(|f| f.replace(true));
    FailPanicsGuard(prev)
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => {
                text.push_str("; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command")
            }
            None => {}
        }
    }
    #[cfg(feature = "diagnostics")]
    if crumbs_on() {
        text.push_str(&crumb_report());
    }
    tracing::error!("{text}");
    #[cfg(test)]
    if FAIL_PANICS.with(std::cell::Cell::get) {
        panic!("{text}");
    }
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// Unless `VICTORY_FAST_WAKE=1`: a submission's completion
/// does not scan the GPU memory waits its copy-back may meet; the fence
/// watcher's poll finds them, as before that scan was added.
fn old_completion_scan() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        // Default: completions leave memory waits to the watcher (waking
        // them on the completing thread made world entry hang the GPU in 4
        // of 5 runs). `VICTORY_FAST_WAKE=1` wakes them at once.
        let fast = crate::switch::on("VICTORY_FAST_WAKE");
        if fast {
            tracing::info!("VICTORY_FAST_WAKE: completions wake satisfied memory waits at once");
        }
        !fast
    })
}

/// The host fence value of the last submission sent to the queue, read
/// without the submission lock (held while one is prepared): every signal
/// of the host fence notes it ([`Host::note_signalled`]) behind a leaf lock.
pub(crate) fn submitted_value() -> u64 {
    relay::QUEUE_TIMELINE.last_signalled().0
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        let mut chosen: Option<DXGI_ADAPTER_DESC1> = None;
        let mut adapter3: Option<IDXGIAdapter3> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            let filter = std::env::var("VICTORY_ADAPTER")
                .ok()
                .filter(|s| !s.trim().is_empty());
            match pick_hardware_adapter(filter.as_deref()) {
                Ok(picked) => {
                    device = Some(picked.device);
                    adapter3 = picked.adapter.cast::<IDXGIAdapter3>().ok();
                    chosen = Some(picked.desc);
                }
                Err(e) if filter.is_some() => return Err(e),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "hardware adapter selection failed; trying the system default adapter"
                    );
                    D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
                }
            }
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
        let uma = device
            .CheckFeatureSupport(
                D3D12_FEATURE_ARCHITECTURE1,
                (&raw mut arch).cast(),
                size_of::<D3D12_FEATURE_DATA_ARCHITECTURE1>() as u32,
            )
            .is_ok()
            && arch.UMA.as_bool();
        let mut opt16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
        let gpu_upload_supported = device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS16,
                (&raw mut opt16).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS16>() as u32,
            )
            .is_ok()
            && opt16.GPUUploadHeapSupported.as_bool();
        let local_vram_budget = adapter3
            .as_ref()
            .and_then(query_local_vram_budget)
            .or_else(|| chosen.map(|d| d.DedicatedVideoMemory as u64));
        // `VICTORY_REBAR` (diagnostics builds): `0` disables all ReBAR (`GPU_UPLOAD`) allocations.
        let allow_rebar = !cfg!(feature = "diagnostics")
            || std::env::var_os("VICTORY_REBAR").is_none_or(|v| v != "0");
        // `VICTORY_REBAR_MB` (MiB): overrides the `BUFFERS` ReBAR budget (`0` disables buffer ReBAR).
        let rebar_mb = std::env::var("VICTORY_REBAR_MB").ok();
        // `VICTORY_REBAR_LOOSE_MB` (MiB): overrides the `LOOSE` windows ReBAR budget (`0` disables loose ReBAR).
        let rebar_loose_mb = std::env::var("VICTORY_REBAR_LOOSE_MB").ok();
        let (rebar_scratch, rebar_shadow_budget, rebar_loose_budget) = resolve_rebar_budgets(
            gpu_upload_supported,
            uma,
            local_vram_budget,
            allow_rebar,
            rebar_mb.as_deref(),
            rebar_loose_mb.as_deref(),
        );
        #[cfg(feature = "diagnostics")]
        if debug
            && crate::switch::on("VICTORY_D3D12_GBV")
            && let Some(mode) =
                parse_eager_gbv(std::env::var_os("VICTORY_D3D12_GBV_EAGER").as_deref())
        {
            match configure_eager_gbv(&device, mode) {
                Ok(()) => tracing::info!(
                    guarded = matches!(mode, EagerGbv::Guarded),
                    "GPU validation pipeline patching front-loaded"
                ),
                Err(e) if matches!(mode, EagerGbv::Guarded) => return Err(e),
                Err(e) => tracing::warn!(error = %e, "could not front-load GPU validation"),
            }
        }
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let copy_queue: Option<ID3D12CommandQueue> = if relay_copy_queue() {
            match device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                ..Default::default()
            }) {
                Ok(q) => Some(q),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "no D3D12 COPY command queue; relay readbacks will share the direct queue"
                    );
                    None
                }
            }
        } else {
            None
        };
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        if let Some(desc) = chosen {
            let name = adapter_name(&desc.Description);
            tracing::info!(
                adapter = %name,
                vendor = format_args!("{:#06x}", desc.VendorId),
                device = format_args!("{:#06x}", desc.DeviceId),
                vram_mb = desc.DedicatedVideoMemory >> 20,
                gpu_upload_supported,
                uma,
                rebar_scratch,
                rebar_shadow_budget_mb = rebar_shadow_budget >> 20,
                rebar_loose_budget_mb = rebar_loose_budget >> 20,
                "host D3D12 device ready"
            );
        } else {
            tracing::info!(
                gpu_upload_supported,
                uma,
                rebar_scratch,
                rebar_shadow_budget_mb = rebar_shadow_budget >> 20,
                rebar_loose_budget_mb = rebar_loose_budget >> 20,
                "host D3D12 device ready"
            );
        }
        Ok(Host {
            device,
            adapter: adapter3,
            rebar_scratch,
            rebar_shadow_budget,
            rebar_loose_budget,
            rebar_readback: Mutex::default(),
            queue,
            copy_queue,
            fence,
            submitted: Mutex::new(0),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
            present_retry: Mutex::new(Retry::new()),
            present_compositor_retry: Mutex::new(Retry::new()),
            free_lists: Mutex::default(),
            mesh_scratch: Mutex::default(),
            #[cfg(test)]
            reused_lists: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            d3d12_errors: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            rebar_hash_compared_pieces: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            rebar_hash_changed_pieces: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            rebar_post_readback_calls: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            rebar_post_readback_bytes: std::sync::atomic::AtomicU64::new(0),
        })
    }
}

#[cfg(feature = "diagnostics")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EagerGbv {
    Current,
    Guarded,
}

#[cfg(feature = "diagnostics")]
fn parse_eager_gbv(value: Option<&std::ffi::OsStr>) -> Option<EagerGbv> {
    let value = value?;
    if value.is_empty() || value == "0" {
        return None;
    }
    if value == "guarded" {
        Some(EagerGbv::Guarded)
    } else {
        Some(EagerGbv::Current)
    }
}

#[cfg(feature = "diagnostics")]
fn front_load_flag_for(
    mode: D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE,
) -> windows::core::Result<D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAGS> {
    match mode {
        D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_UNGUARDED_VALIDATION => Ok(
            D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_FRONT_LOAD_CREATE_UNGUARDED_VALIDATION_SHADERS,
        ),
        D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_GUARDED_VALIDATION => Ok(
            D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_FRONT_LOAD_CREATE_GUARDED_VALIDATION_SHADERS,
        ),
        D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_STATE_TRACKING_ONLY => Ok(
            D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_FRONT_LOAD_CREATE_TRACKING_ONLY_SHADERS,
        ),
        _ => Err(windows::core::Error::from_hresult(
            windows::Win32::Foundation::E_INVALIDARG,
        )),
    }
}

/// Requests validation shader compilation at PSO creation rather than first
/// bind, preserving unrelated flags and verifying read-back settings before any
/// pipelines are built.
#[cfg(feature = "diagnostics")]
fn configure_eager_gbv(device: &ID3D12Device, mode: EagerGbv) -> windows::core::Result<()> {
    let debug = device.cast::<ID3D12DebugDevice1>()?;
    let kind = D3D12_DEBUG_DEVICE_PARAMETER_GPU_BASED_VALIDATION_SETTINGS;
    let size = size_of::<D3D12_DEBUG_DEVICE_GPU_BASED_VALIDATION_SETTINGS>() as u32;
    let mut settings = D3D12_DEBUG_DEVICE_GPU_BASED_VALIDATION_SETTINGS::default();
    // SAFETY: `kind` takes `D3D12_DEBUG_DEVICE_GPU_BASED_VALIDATION_SETTINGS` of exact `size`.
    unsafe { debug.GetDebugParameter(kind, (&raw mut settings).cast(), size)? };
    let prior_flags = settings.PipelineStateCreateFlags.0;
    if mode == EagerGbv::Guarded {
        settings.DefaultShaderPatchMode =
            D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_GUARDED_VALIDATION;
    }
    let flag = front_load_flag_for(settings.DefaultShaderPatchMode)?;
    settings.PipelineStateCreateFlags.0 |= flag.0;
    let mut observed = D3D12_DEBUG_DEVICE_GPU_BASED_VALIDATION_SETTINGS::default();
    // SAFETY: both structs are stack-owned for the synchronous COM calls.
    unsafe {
        debug.SetDebugParameter(kind, (&raw const settings).cast(), size)?;
        debug.GetDebugParameter(kind, (&raw mut observed).cast(), size)?;
    }
    if observed.DefaultShaderPatchMode != settings.DefaultShaderPatchMode
        || observed.PipelineStateCreateFlags.0 & flag.0 == 0
        || observed.PipelineStateCreateFlags.0 & prior_flags != prior_flags
    {
        return Err(windows::core::Error::from_hresult(
            windows::Win32::Foundation::E_FAIL,
        ));
    }
    Ok(())
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    /// The id occlusion query tracking knows this list's recordings by
    /// (`crate::query::next_recording`): it stays with the list when the
    /// list is moved out of its `Arc` to be recycled, which its address
    /// does not.
    recording: usize,
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices): keeps
    /// at most one warm chunk across resets and drops all chunks when parked in
    /// the pool.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// Actual D3D12 access state per host buffer for this recording.
    shadow_states: Mutex<std::collections::HashMap<usize, (ID3D12Resource, D3D12_RESOURCE_STATES)>>,
    /// Incremented whenever `shadow_states` inserts or transitions a shadow
    /// buffer's D3D12 access state; reset to 0 when the list is reset.
    shadow_state_gen: std::sync::atomic::AtomicU64,
    /// Incremented whenever `states` inserts a new texture or transitions a
    /// texture to a different state; reset to 0 when the list is reset.
    state_gen: std::sync::atomic::AtomicU64,
    /// Title resource pointers whose tracked state changed on this list since
    /// reset (append-only; cleared on reset). Lets `draw::State::bindless_textures`
    /// re-sync only the textures touched since the previous draw instead of
    /// walking every resident bindless texture on RT switches or copies.
    dirty_resources: Mutex<Vec<usize>>,
    /// The depth textures with colour twins the list uses
    /// ([`HostList::twin_use`]).
    twins: Mutex<std::collections::HashMap<usize, TwinUse>>,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// Memory handed out for the list's recording, kept until it is finished
    /// when the title reset or dropped the list while a submission of it was
    /// still to run.
    kept_uploads: Mutex<Vec<Vec<u8>>>,
    /// Dynamic descriptor slots kept alongside `kept_uploads` until the pending
    /// submission completes.
    kept_dynamic: Mutex<Vec<(usize, u32, u32)>>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// Dispatches and buffer copies recorded since the reset: what may have
    /// written indirect arguments (`crate::indirect` clamps hoisted before).
    writes: std::sync::atomic::AtomicU64,
    /// Whether a shader that can store to buffers was bound since the last
    /// clamp wait (starts true after reset so earlier lists' writes wait).
    buffer_writes: std::sync::atomic::AtomicBool,
    /// Buffers for clamped indirect arguments when this list holds the clamps
    /// of the list after it (`clamp_room`).
    clamp_chunks: Mutex<ClampChunks>,
    /// Hoisted clamped argument buffers read by this list's indirect draws,
    /// transitioned from `INDIRECT_ARGUMENT` back to `COMMON` at close.
    hoisted_buffers: Mutex<Vec<ID3D12Resource>>,
    /// The records its draws read from slots that its prelude fills
    /// (`crate::indirect::gather`).
    gather: Mutex<Gather>,
    /// GPU-writable scratch for indirect rectangle expansion (`rectangles`):
    /// expanded vertices, rewritten draw arguments and per-draw atomic
    /// counters, keeping at most one warm counter chunk across resets and
    /// releasing all buffers when parked in the pool.
    rect_scratch: Mutex<RectScratch>,
    /// Active retirement epoch pinned while this list has recorded work or is
    /// queued/executing (0 when idle).
    epoch: std::sync::atomic::AtomicU64,
    /// Highest [`RESOURCE_SEQ`] observed when this list recorded work; shared
    /// with [`RetireState`] while `epoch != 0` so newer resources are not
    /// blocked by stale lists.
    work_seq: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Queued or executing submissions currently pinning this list.
    in_flight: std::sync::atomic::AtomicUsize,
    /// Serializes `in_flight` transitions with `epoch` updates so a concurrent
    /// `end_submit` never clears an epoch pinned by a new `begin_submit`.
    epoch_lock: Mutex<()>,
    /// Test-only isolated retirement state captured when the list is created or
    /// acquired so parallel GPU tests do not pin each other's drops.
    #[cfg(test)]
    retire_scope: Option<std::sync::Arc<Mutex<RetireState>>>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
    /// With `VICTORY_BREADCRUMBS`: this recording's pair of words in the
    /// breadcrumb buffer ([`Crumbs`]), `u32::MAX` before its first note.
    #[cfg(feature = "diagnostics")]
    crumb: std::sync::atomic::AtomicU32,
}

impl Drop for HostList {
    fn drop(&mut self) {
        crate::query::forget_list(self.recording);
        self.release_kept();
        self.release_epoch();
    }
}

/// A list's buffers for hoisted clamped indirect arguments with their states,
/// how many are in use since the reset, the bytes taken of the last of those,
/// whether the list has the barrier that precedes the clamps, and whether the
/// clamp compute pipeline is already bound on this list.
#[derive(Default)]
struct ClampChunks {
    all: Vec<(ID3D12Resource, D3D12_RESOURCE_STATES)>,
    used: usize,
    at: u64,
    barrier: bool,
    bound: bool,
}

/// What a list gathers for its prelude to clamp (`crate::indirect::gather`).
#[derive(Default)]
struct Gather {
    /// Which argument buffers may still be gathered (per-buffer or global close).
    gate: crate::indirect::GatherGate,
    /// One set per argument buffer, kind and stride.
    sets: Vec<crate::indirect::GatherSet>,
    /// The budget words the gathered records are charged to.
    budget: Option<ID3D12Resource>,
    /// Scratch buffers of earlier recordings, kept for the next sets.
    spare: Vec<ID3D12Resource>,
}

/// The most argument buffers a list gathers from.
const GATHER_SETS: usize = 8;

// SAFETY: D3D12 lists and allocators may be used from any thread, one at
// a time: the title records a list on one thread at a time, and a
// submission runs it after its recording ended (a reset while one is to
// run moves recording to a new list). The rest is locked.
unsafe impl Send for HostList {}
// SAFETY: as above.
unsafe impl Sync for HostList {}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

/// A single first access can promote implicitly. Use an explicit state for
/// a combined read mask and retain existing read permissions on later reads.
fn shadow_transition(
    before: D3D12_RESOURCE_STATES,
    wanted: D3D12_RESOURCE_STATES,
) -> Option<D3D12_RESOURCE_STATES> {
    if before == D3D12_RESOURCE_STATE_COMMON {
        return (wanted.0.count_ones() != 1).then_some(wanted);
    }
    if before.contains(wanted) {
        return None;
    }
    let after = if before != D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        && wanted != D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    {
        before | wanted
    } else {
        wanted
    };
    Some(after)
}

type States = std::collections::HashMap<
    usize,
    (
        u32,
        u32,
        Option<u32>,
        std::sync::Arc<crate::resource::HostTexture>,
    ),
>;

/// What a list does with a depth texture that has a colour twin
/// ([`crate::resource::Twin`]). Lists are recorded on many threads and run
/// in submission order, so a list cannot know which side an earlier list
/// leaves behind: it notes which sides it uses as it finds them, and its
/// prelude copies one side into the other at submission if that side is
/// behind then. From its first use of a side on, the list knows.
struct TwinUse {
    /// The depth texture (the title resource's host copy) and its twin.
    depth: std::sync::Arc<crate::resource::HostTexture>,
    twin: std::sync::Arc<crate::resource::Twin>,
    /// The sides (depth texture, twin) the list uses as it finds them.
    needs: [bool; 2],
    /// Whether each side lacks what the other was written last, where the
    /// list is now; `None`: as the list found it.
    behind: [Option<bool>; 2],
    /// The draw or dispatch being prepared writes the twin.
    writing: bool,
    /// The depth plane's twin holds what the plane lacks although the list
    /// copied the twins since: the plane's pass waits for the list's close.
    late: bool,
}

/// Records into `list` the copy of a depth texture into its colour twin
/// (`colour`) or of the twin into the depth texture, both already in their
/// copy states.
fn record_twin_copy(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    twin: &crate::resource::Twin,
    colour: bool,
) {
    let (from, to) = if colour {
        (depth, &twin.colour.texture)
    } else {
        (&twin.colour.texture, depth)
    };
    crate::resource::TWINS[2 + usize::from(colour)].fetch_add(1, Ordering::Relaxed);
    // shortcut: the depth plane's twin is not given the plane (that is a
    // pass the other way: the plane read by a shader, the twin as its
    // target). The twin then lacks what draws and clears wrote to the
    // plane since shaders last wrote the twin. In a title whose draws and
    // clears only test the plane (192 draws, run 29), nothing writes it but
    // the twin's pass, so the twin lacks nothing. Counted.
    if colour && twin.plane.is_some() {
        crate::resource::PLANES[1].fetch_add(1, Ordering::Relaxed);
    }
    let Some(through) = &twin.through else {
        // SAFETY: an open list and two live textures.
        unsafe { list.CopyResource(to, from) };
        return;
    };
    // A stencil plane and its twin: each mip and slice of the plane goes
    // into the buffer, laid out as the host lays the plane out for copies,
    // and from the buffer into the other texture, whole subresources both
    // times. In the depth texture the stencil plane's subresources follow
    // the depth plane's; the twin has the one plane.
    let Some(host) = host() else { return };
    // SAFETY: a live texture.
    let of = unsafe { depth.GetDesc() };
    let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
    let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); planes as usize];
    // SAFETY: a plain query writing `planes` footprints.
    unsafe {
        host.device.GetCopyableFootprints(
            &of,
            planes,
            planes,
            0,
            Some(rows.as_mut_ptr()),
            None,
            None,
            None,
        );
    }
    let (first_from, first_to) = if colour { (planes, 0) } else { (0, planes) };
    let subresource = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index,
        },
    };
    let placed = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(through) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer was made as big
    // as these footprints, and rests in the common state between copies.
    unsafe {
        list.ResourceBarrier(&[transition(through, common, dest)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_from + i as u32;
            list.CopyTextureRegion(&placed(row), 0, 0, 0, &subresource(from, index), None);
        }
        list.ResourceBarrier(&[transition(through, dest, source)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_to + i as u32;
            list.CopyTextureRegion(&subresource(to, index), 0, 0, 0, &placed(row), None);
        }
        list.ResourceBarrier(&[transition(through, source, common)]);
    }
}

/// The pass that draws the twin of a depth plane into the plane: one
/// triangle over each mip of each slice, its pixel shader writing the
/// twin's texel as the depth, with the depth test "always" and no stencil
/// (the stencil plane is copied through a buffer and not touched here).
///
/// A 16-bit twin holds whole steps of 1/65535, and the depth written is
/// the step times the float nearest 1/65535 (0x37800080). That is how a
/// title's tiled lighting computes the depth of the rectangles it then
/// draws with the test "equal", one class of materials
/// a draw (all six of its vertex shaders multiply the class number by that
/// constant), so the pass and the draw give the host the same float, and
/// the host turns the same float into the same 24 bits. The step divided
/// by 65535 is another float for 512 of the 65,536 steps (none under 257),
/// and lands in another 24-bit step for 127 of them (none under 385).
const PLANE_CORNERS: &str = include_str!("../shaders/plane_corners.hlsl");
const PLANE_STEPS: &str = include_str!("../shaders/plane_steps.hlsl");
const PLANE_FLOATS: &str = include_str!("../shaders/plane_floats.hlsl");

/// Per kind of twin (16-bit steps or floats) and depth format the root signature
/// and pipeline, made when first needed (`None`: the host refused).
struct PlanePass {
    made: Mutex<Vec<(bool, i32, Option<PlanePipeline>)>>,
}

/// The pass's root signature and pipeline for one kind of twin and one
/// depth format.
type PlanePipeline = (ID3D12RootSignature, ID3D12PipelineState);

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for PlanePass {}
// SAFETY: as above.
unsafe impl Sync for PlanePass {}

impl PlanePass {
    /// The root signature and pipeline for a twin of 16-bit steps (`steps`)
    /// or of floats and the depth format `format`.
    fn pipeline(&self, steps: bool, format: DXGI_FORMAT) -> Option<PlanePipeline> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, pipeline)) = made.iter().find(|m| (m.0, m.1) == (steps, format.0)) {
            return pipeline.clone();
        }
        let pipeline = match Self::make(steps, format) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    steps,
                    format = format.0,
                    "no pipeline to draw a depth plane from its twin: such planes stay as they are"
                );
                None
            }
        };
        made.push((steps, format.0, pipeline.clone()));
        pipeline
    }

    fn make(steps: bool, format: DXGI_FORMAT) -> windows::core::Result<PlanePipeline> {
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        // Through the helpers' compiler: the system's first, then DXC. Wine's
        // `d3dcompiler_47` is the system compiler there, and what it refuses
        // (`RWByteAddressBuffer` loads; these shaders use none today, their
        // siblings do) DXC compiles.
        let vs =
            crate::hlsl::compile_stage("depth plane corners", PLANE_CORNERS, "corner", "vs_5_1")?;
        let source = if steps { PLANE_STEPS } else { PLANE_FLOATS };
        let ps = crate::hlsl::compile_stage("depth plane", source, "plane", "ps_5_1")?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: the root signature is inside the pixel shader; the
        // description points at live data and never releases its pointer.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Every texel is written; stencil stays off.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                DSVFormat: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }
}

/// Records into `list` the pass that draws `plane`, the twin of the depth
/// plane of `depth`, into that plane: `depth` in the depth-write state,
/// `plane` readable by pixel shaders. The pass sets the list's descriptor
/// heap, root signature, pipeline, target and viewport: it runs only in a
/// list that records nothing of the title's afterwards.
fn record_plane_pass(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
    plane_views: &OnceLock<Option<ID3D12DescriptorHeap>>,
) {
    static PASS: PlanePass = PlanePass {
        made: Mutex::new(Vec::new()),
    };
    let Some(host) = host() else { return };
    // SAFETY: live resources.
    let (of, twin) = unsafe { (depth.GetDesc(), plane.GetDesc()) };
    // An R16 typeless twin (53) holds steps, read as R16_UINT (57); an R32
    // typeless one floats, read as R32_FLOAT (41). The depth view names
    // the depth-stencil format of the texture's family (44: 45, 19: 20).
    let steps = twin.Format.0 == 53;
    let format = DXGI_FORMAT(match of.Format.0 {
        44 => 45,
        19 => 20,
        typed => typed,
    });
    let Some((root, pipeline)) = PASS.pipeline(steps, format) else {
        return;
    };
    let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    // SAFETY: an immutable property of our live device.
    let step = unsafe { host.device.GetDescriptorHandleIncrementSize(kind) };
    let mips = u32::from(of.MipLevels);
    let slices = u32::from(of.DepthOrArraySize);
    let Some(views) = plane_views.get_or_init(|| {
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: mips * slices,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        // SAFETY: desc lives through creation and its count covers the entire texture.
        let views: ID3D12DescriptorHeap = unsafe { host.device.CreateDescriptorHeap(&desc) }
            .inspect_err(
                |e| tracing::warn!(error = %e, "no descriptor heap for a depth plane twin"),
            )
            .ok()?;
        // SAFETY: live, initialized heap.
        let mut cpu = unsafe { views.GetCPUDescriptorHandleForHeapStart() };
        for slice in 0..slices {
            for mip in 0..mips {
                let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT(if steps { 57 } else { 41 }),
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                            MostDetailedMip: mip,
                            MipLevels: 1,
                            FirstArraySlice: slice,
                            ArraySize: 1,
                            ..Default::default()
                        },
                    },
                };
                // SAFETY: exactly one slot for every subresource; the heap is not published yet.
                unsafe {
                    host.device
                        .CreateShaderResourceView(plane, Some(&view), cpu)
                };
                cpu.ptr += step as usize;
            }
        }
        Some(views)
    }) else {
        return;
    };
    // SAFETY: this initialized, immutable heap remains live through the recording's execution.
    let mut gpu = unsafe { views.GetGPUDescriptorHandleForHeapStart() };
    for slice in 0..slices {
        for mip in 0..mips {
            let target = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                        MipSlice: mip,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                    },
                },
                ..Default::default()
            };
            let target = host.dsv(depth, Some(&target));
            let (width, height) = ((of.Width >> mip).max(1) as u32, (of.Height >> mip).max(1));
            // SAFETY: an open list; a slot of our heap; live resources and
            // objects (the pass's own live as long as the process).
            unsafe {
                list.SetDescriptorHeaps(&[Some(views.clone())]);
                list.SetGraphicsRootSignature(&root);
                list.SetPipelineState(&pipeline);
                list.SetGraphicsRootDescriptorTable(0, gpu);
                list.OMSetRenderTargets(0, None, false, Some(&target));
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(4),
                );
                list.DrawInstanced(3, 1, 0, 0);
            }
            gpu.ptr += u64::from(step);
            crate::resource::PLANES[0].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED"))
}

/// Whether our own breadcrumbs are on (`VICTORY_BREADCRUMBS`, diagnostics
/// builds, without device-removed data): each noted command of a list is
/// preceded by two immediate writes into system memory, read when the
/// device is removed to name the commands that had begun but not finished.
/// Lighter than device-removed data, which slows the GPU enough to hide
/// some faults.
#[cfg(feature = "diagnostics")]
pub(crate) fn crumbs_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !dred_on() && crate::switch::on("VICTORY_BREADCRUMBS"))
}

/// Recordings with breadcrumb words at once (reused round-robin).
#[cfg(feature = "diagnostics")]
const CRUMB_LISTS: u32 = 4096;

/// The breadcrumb buffer: per recording, the noted command done before
/// and the one begun up to.
#[cfg(feature = "diagnostics")]
struct Crumbs {
    _buffer: ID3D12Resource,
    gpu: u64,
    cpu: usize,
    next: std::sync::atomic::AtomicU32,
}

#[cfg(feature = "diagnostics")]
fn crumbs() -> Option<&'static Crumbs> {
    static CRUMBS: OnceLock<Option<Crumbs>> = OnceLock::new();
    CRUMBS
        .get_or_init(|| {
            let device = &host()?.device;
            let props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: u64::from(CRUMB_LISTS) * 8,
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
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain object creation; the mapping stays valid while
            // the buffer lives (for the process).
            unsafe {
                device
                    .CreateCommittedResource(
                        &props,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .ok()?;
                let buffer = buffer?;
                let mut cpu = std::ptr::null_mut();
                buffer.Map(0, None, Some(&mut cpu)).ok()?;
                std::ptr::write_bytes(cpu.cast::<u8>(), 0, CRUMB_LISTS as usize * 8);
                tracing::warn!("VICTORY_BREADCRUMBS: noted commands leave breadcrumbs");
                Some(Crumbs {
                    gpu: buffer.GetGPUVirtualAddress(),
                    cpu: cpu as usize,
                    _buffer: buffer,
                    next: std::sync::atomic::AtomicU32::new(0),
                })
            }
        })
        .as_ref()
}

/// The recordings of the last submissions, with breadcrumbs: host fence
/// value, then per list its breadcrumb words and noted commands (copied
/// from the list at each submission, since a closed list can be submitted
/// more than once and its notes are cleared only when it resets).
#[cfg(feature = "diagnostics")]
type CrumbRun = (u64, Vec<(u32, Vec<Note>)>);

#[cfg(feature = "diagnostics")]
static CRUMB_RUNS: Mutex<std::collections::VecDeque<CrumbRun>> =
    Mutex::new(std::collections::VecDeque::new());

/// A recording's noted commands for one submission's breadcrumb report.
/// The recording keeps its own: its breadcrumb commands run again when it
/// is submitted again, and that submission's report names them too.
#[cfg(feature = "diagnostics")]
fn submitted_notes(notes: &Mutex<Vec<Note>>) -> Vec<Note> {
    notes.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Keeps the noted commands of the lists of submission `sub` for the
/// breadcrumb report.
#[cfg(feature = "diagnostics")]
fn keep_crumbs(sub: u64, runs: &[&HostList]) {
    let lists = runs
        .iter()
        .map(|r| (r.crumb.load(Ordering::Relaxed), submitted_notes(&r.notes)))
        .collect();
    let mut kept = CRUMB_RUNS.lock().unwrap_or_else(|e| e.into_inner());
    kept.push_back((sub, lists));
    while kept.len() > 12 {
        kept.pop_front();
    }
}

/// For each list of the last submissions that did not finish: the noted
/// commands that had begun but not finished when the device was removed.
#[cfg(feature = "diagnostics")]
fn crumb_report() -> String {
    let Some(crumbs) = crumbs() else {
        return "; no breadcrumbs".into();
    };
    let mut text = String::new();
    let kept = CRUMB_RUNS.lock().unwrap_or_else(|e| e.into_inner());
    for (sub, lists) in kept.iter() {
        text.push_str(&format!("; submission {sub}"));
        for (i, (slot, notes)) in lists.iter().enumerate() {
            let slot = *slot;
            if slot == u32::MAX {
                text.push_str(&format!("; list {i}: no noted commands"));
                continue;
            }
            // SAFETY: the mapped breadcrumb words of this recording, written by the GPU.
            let (done, begun) = unsafe {
                let at = (crumbs.cpu as *const u32).add(slot as usize * 2);
                (at.read_volatile(), at.add(1).read_volatile())
            };
            let n = notes.len() as u32;
            if done >= n {
                text.push_str(&format!("; list {i}: {n} noted commands done"));
                continue;
            }
            text.push_str(&format!(
                "; list {i}: done before {done}, begun up to {begun} of {n} noted commands:"
            ));
            let last = begun.max(done + 1).min(n).min(done + 6);
            for k in done..last {
                let note: String = notes[k as usize].0.chars().take(600).collect();
                text.push_str(&format!(" [{k}] {note}"));
            }
        }
    }
    text
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;

/// The buffer for mesh programs run without mesh shaders
/// (`State::dispatch_mesh`), and its size: what their compute half stores
/// and their draw reads back. Every such dispatch of every list uses it
/// from the start: the host has one queue, which runs lists one after the
/// other and a list's commands in order, so one dispatch's draw has read it
/// before the next one's compute pass writes it.
#[derive(Default)]
struct MeshScratch {
    buffer: Option<(ID3D12Resource, u64)>,
}

/// The least the mesh buffer holds.
pub(crate) const MESH_BUFFER_MIN: u64 = 16 << 20;
/// The most the mesh buffer is grown to. A dispatch of more thread groups
/// than it holds runs in parts (`State::dispatch_mesh`), so this trades
/// memory against passes; an indirect dispatch, whose group count only the
/// GPU has, records the parts of the most groups it can have.
pub(crate) const MESH_BUFFER_MOST: u64 = 64 << 20;

/// Per-list GPU-writable buffers for indirect rectangle expansion
/// (`crate::rectangles`): one vertex scratch buffer (reused per draw after
/// its VB -> UAV barrier), one rewritten-arguments buffer (grown if a draw
/// has more records), and chunks of 4-byte atomic counters zeroed before
/// submission.
#[derive(Default)]
struct RectScratch {
    vertices: Option<ID3D12Resource>,
    args: Option<(ID3D12Resource, u64)>,
    counters: Vec<(ID3D12Resource, usize, u64)>,
    counter_chunk: usize,
    counter_used: u32,
}

// SAFETY: D3D12 resources are free-threaded; the counter pointers are
// persistent mappings written only before submission.
unsafe impl Send for RectScratch {}

/// Counters per chunk (4 KiB = 1024 words).
const RECT_COUNTER_CHUNK: u32 = 1024;
/// Initial rewritten-arguments buffer size (64 KiB = 4096 draw records).
const RECT_ARGS_MIN: u64 = 64 << 10;

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see. Runs with
    /// `title_dst = true` so unmapped title pages keep the commitment guard.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory. Runs with
    /// `title_dst = false`: the destination is host-mapped, so Wine/vkd3d
    /// `VirtualQuery` quirks must not skip it (#518).
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Work that writes host memory and reads a known title memory range.
    HostRead(usize, usize, Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>, u64),
}

/// A texture-to-buffer readback copied from a mapped readback buffer into
/// title memory once the GPU submission completes.
struct Readback {
    dst: usize,
    src: usize,
    dst_pitch: usize,
    src_pitch: usize,
    lines: usize,
    row_bytes: usize,
    total: usize,
    /// For occlusion query results: the resolve, whose spare query results
    /// follow the `total` bytes at `src` and are combined into them before
    /// the copy ([`crate::query::Resolve`]).
    query: Option<std::sync::Arc<crate::query::Resolve>>,
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload/readback buffers its GPU commands name, kept until reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    /// Post-submission readbacks from GPU textures into title buffers.
    readbacks: Vec<Readback>,
    /// Title textures of the kinds copied back ([`crate::texback`]) that
    /// the list's shaders write, copied when the list closes.
    back_wanted: Vec<crate::texback::Target>,
    /// Those copies, whose texels go to title memory once the list has run.
    texture_backs: Vec<crate::texback::Back>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// The host pipelines [`HostList::bind_pipeline`] kept (by address), so a
    /// pipeline the list binds again is not kept or queried again.
    bound: std::collections::HashSet<usize>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Upload memory filled from title memory since the last title write
    /// (source, bytes): its host address, which a second fill would find
    /// holding the same bytes.
    filled: std::collections::HashMap<(usize, usize), u64>,
    /// Buffer ranges declared written by shaders since the last title copy
    /// (host copy, address, bytes): declaring them again would change nothing.
    gpu_written: std::collections::HashSet<(usize, usize, usize)>,
    /// Most detailed mip this list has copied for each texture.
    textures: std::collections::HashMap<usize, u32>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Things the list's commands use that are given back when the last
    /// holder lets go ([`HostList::hold`]), by address.
    held: std::collections::HashMap<usize, std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    /// Monotonic count of deferred title-memory copies (`defer_copy`) recorded
    /// since reset, used to partition fresh-table CBV snapshot reuse across
    /// intervening buffer writes.
    copy_gen: u64,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            // Diagnostics builds: run mesh programs the way hosts without
            // mesh shaders do, on a host that has them (to compare the two).
            let unused =
                cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_NO_MESH_SHADERS");
            tracing::info!(mesh, unused, "host mesh shader support");
            mesh && !unused
        })
    }

    /// Whether the host gives pixel shaders barycentrics (`SV_Barycentrics`
    /// and `GetAttributeAtVertex`, `D3D12_FEATURE_DATA_D3D12_OPTIONS3
    /// .BarycentricsSupported`); logged once. Without them, pixel shaders
    /// that need them run behind the geometry shader of
    /// `victory_shader::emit::barycentric_gs`.
    pub(crate) fn barycentrics(&self) -> bool {
        static BARYCENTRICS: OnceLock<bool> = OnceLock::new();
        *BARYCENTRICS.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
            // SAFETY: the query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS3,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let barycentrics = ok && options.BarycentricsSupported.as_bool();
            // Diagnostics builds: run them the way hosts without
            // barycentrics do, on a host that has them (to compare the two).
            let unused =
                cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_NO_BARYCENTRICS");
            tracing::info!(barycentrics, unused, "host barycentric support");
            barycentrics && !unused
        })
    }

    /// Whether the host adapter accepts `WaveSize(32)` compute shaders; logged
    /// once. Adapters that only support 64-lane waves (`WaveLaneCountMin == 64`)
    /// return `false` so eligible <=32-thread Wave64 reductions keep `WaveSize(64)`.
    pub(crate) fn supports_wave32(&self) -> bool {
        static WAVE32: OnceLock<bool> = OnceLock::new();
        *WAVE32.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: query writes its own `D3D12_FEATURE_DATA_D3D12_OPTIONS1` struct.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS1,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            // `VICTORY_WAVE64_SPLIT=1`: on 32-lane hosts, wave64 programs run
            // as two 32-lane halves. shortcut: experimental; off, they are
            // refused as before (the title read -1 from GPU-written data
            // with it on).
            let split = std::env::var_os("VICTORY_WAVE64_SPLIT").is_some_and(|v| v != "0");
            victory_shader::emit::set_host_wave64(!split || ok && options.WaveLaneCountMax >= 64);
            let wave32 = ok
                && wave32_supported(
                    options.WaveOps.as_bool(),
                    options.WaveLaneCountMin,
                    options.WaveLaneCountMax,
                );
            tracing::info!(
                wave32,
                min = options.WaveLaneCountMin,
                max = options.WaveLaneCountMax,
                "host wave32 support"
            );
            wave32
        })
    }

    /// A recycled host list (reset and open), or a new one when the pool is
    /// empty.
    pub(crate) fn acquire_list(&self) -> windows::core::Result<HostList> {
        let pooled = self
            .free_lists
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop();
        #[cfg(test)]
        if let Some(mut list) = pooled {
            self.reused_lists.fetch_add(1, Ordering::Relaxed);
            list.reset();
            list.retire_scope = current_retire_scope();
            return Ok(list);
        }
        #[cfg(not(test))]
        if let Some(list) = pooled {
            list.reset();
            return Ok(list);
        }
        self.new_list()
    }

    #[cfg(test)]
    pub(crate) fn reused_lists(&self) -> u64 {
        self.reused_lists.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn d3d12_errors(&self) -> u64 {
        self.d3d12_errors.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn rebar_hash_stats(&self) -> RebarHashStats {
        RebarHashStats {
            compared_pieces: self.rebar_hash_compared_pieces.load(Ordering::Relaxed),
            changed_pieces: self.rebar_hash_changed_pieces.load(Ordering::Relaxed),
            post_readback_calls: self.rebar_post_readback_calls.load(Ordering::Relaxed),
            post_readback_bytes: self.rebar_post_readback_bytes.load(Ordering::Relaxed),
        }
    }

    /// Returns a finished, unreferenced host list to the pool up to
    /// [`list_pool_cap`], releasing all retained uploads, tracked resources,
    /// scratch buffers, and retirement pins immediately.
    pub(crate) fn recycle_list(&self, list: HostList) {
        list.release_for_pool();
        let mut pool = self.free_lists.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() < list_pool_cap() {
            pool.push(list);
        }
    }

    /// Optional dedicated `D3D12_COMMAND_LIST_TYPE_COPY` queue for relay DMA.
    #[allow(dead_code)]
    pub(crate) fn copy_queue(&self) -> Option<&ID3D12CommandQueue> {
        self.copy_queue.as_ref()
    }

    /// Creates a closed direct command allocator and command list pair for
    /// internal host passes (card sync/compare/fetch, relay fallback).
    pub(super) fn make_prelude(&self) -> Option<Prelude> {
        // SAFETY: plain D3D12 object creation on our live device.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .ok()?;
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .ok()?;
            list.Close().ok()?;
            Some((allocator, list))
        }
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            Ok(HostList {
                recording: crate::query::next_recording(),
                allocator,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                shadow_states: Mutex::default(),
                shadow_state_gen: std::sync::atomic::AtomicU64::new(0),
                state_gen: std::sync::atomic::AtomicU64::new(0),
                dirty_resources: Mutex::default(),
                twins: Mutex::default(),
                prelude: (prelude_allocator, prelude),
                kept_uploads: Mutex::default(),
                kept_dynamic: Mutex::default(),
                pipelines: Mutex::default(),
                budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                writes: std::sync::atomic::AtomicU64::new(0),
                buffer_writes: std::sync::atomic::AtomicBool::new(true),
                clamp_chunks: Mutex::default(),
                hoisted_buffers: Mutex::default(),
                gather: Mutex::default(),
                rect_scratch: Mutex::default(),
                epoch: std::sync::atomic::AtomicU64::new(0),
                work_seq: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
                in_flight: std::sync::atomic::AtomicUsize::new(0),
                epoch_lock: Mutex::default(),
                #[cfg(test)]
                retire_scope: current_retire_scope(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
                #[cfg(feature = "diagnostics")]
                crumb: std::sync::atomic::AtomicU32::new(u32::MAX),
            })
        }
    }

    /// Runs closed lists and waits for them (settling any in-flight async
    /// submissions first when [`late`] is active).
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        let _ = self.submit(runs, false);
    }

    /// Runs closed lists either synchronously (`late == false`) or returns
    /// `(submitted, Sent)` with the GPU running ahead (`late == true`).
    fn submit<'a>(
        &'a self,
        runs: &[&HostList],
        late: bool,
    ) -> Option<(std::sync::MutexGuard<'a, u64>, Sent)> {
        // D3D12 requires NumCommandLists >= 1; an empty submission is a no-op
        // (VIC-BT-035 / #440).
        if runs.is_empty() {
            return None;
        }
        let ptrs: Vec<usize> = runs
            .iter()
            .map(|list| std::ptr::from_ref(*list) as usize)
            .collect();
        let batches = unique_list_batches(&ptrs);
        // If the same HostList appears twice in one submission, fall back to
        // synchronous batch-by-batch execution so prelude() never resets an
        // in-flight allocator.
        let late = late && batches.len() == 1;

        // Safe pre-lock prep outside `self.submitted`: purely inspect closed
        // `runs` without touching `LINE` or mutable `Shadow` submission state.
        let watch_ranges = crate::watch::once().then(|| collect_ask_once_ranges(runs));
        let (back, mut rebar_entries, non_rebar_writes) = collect_submission_writes(runs, late);
        let card_declared = crate::shadow::cards().then(|| collect_card_declared(&back));

        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        // If another thread holds `self.submitted`, drain pending descriptor
        // heap syncs while waiting so descriptor prep overlaps with the holder.
        let mut submitted = match self.submitted.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::WouldBlock) => {
                crate::descriptor::sync_all_heaps();
                self.submitted.lock().unwrap_or_else(|e| e.into_inner())
            }
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        };
        let mut deferred_woken = crate::fence::Woken::default();
        if !late {
            deferred_woken.add(self.settle(&mut submitted, relay::WHY_SYNC));
        } else if let Some(block) = relay::late_blocked(runs) {
            match block {
                relay::LateBlock::Read(ranges) if relay::range_settle() && relay::relay_exact() => {
                    // Only until flights whose `back` / `readback_ranges`
                    // overlap the conflicting copy/fill/host-read/texture
                    // ranges have landed.
                    relay::relay_wait(relay::WHY_READ, |line| {
                        !relay::line_overlaps_read(line, &ranges)
                    });
                }
                relay::LateBlock::Read(_) => {
                    deferred_woken.add(self.settle(&mut submitted, relay::WHY_READ));
                }
                relay::LateBlock::Run => {
                    deferred_woken.add(self.settle(&mut submitted, relay::WHY_RUN));
                }
            }
        }
        let behind = late && relay::flying();
        if !behind {
            crate::inplace::tick();
        }
        // The order these lists reach the host queue in: occlusion query
        // resolves choose the parts of the runs submitted before them.
        for list in runs {
            crate::query::submitting(list.recording());
        }
        let this_submission = relay::STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        crate::shadow::preparing(this_submission);

        #[cfg(test)]
        let exec_scope = runs
            .iter()
            .find_map(|l| l.retire_scope.clone())
            .or_else(current_retire_scope);
        #[cfg(test)]
        let exec_state: &Mutex<RetireState> = exec_scope.as_deref().unwrap_or(&RETIRE_STATE);
        #[cfg(not(test))]
        let exec_state: &Mutex<RetireState> = &RETIRE_STATE;
        let exec_epoch = ExecEpochGuard {
            #[cfg(test)]
            scope: exec_scope.clone(),
            epoch: acquire_epoch_in(exec_state, exec_work_seq()),
        };
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        #[cfg(feature = "diagnostics")]
        if crumbs_on() {
            keep_crumbs(*submitted + 1, runs);
        }
        let mut batch = Batch::default();
        let mut settled = std::time::Duration::ZERO;

        struct AskOnceGuard;
        impl Drop for AskOnceGuard {
            fn drop(&mut self) {
                crate::watch::ask_once_end();
            }
        }
        if let Some(ranges) = &watch_ranges {
            crate::watch::ask_once(ranges);
        }
        let watch_guard = AskOnceGuard;
        // Behind flights: the title wrote memory that a flight's shaders may
        // write too, and this submission refreshes it. Only the flights whose
        // `back` overlaps the stale ranges are waited on (or all flights under
        // `VICTORY_RANGE_SETTLE=0`), and then the asking is voided and made
        // again so copy-back writes into private title memory are observed.
        if behind {
            let stale = relay::stale_under_flights(runs);
            if !stale.is_empty() {
                let settling = std::time::Instant::now();
                deferred_woken.add(self.settle_stale(&mut submitted, &stale));
                settled += settling.elapsed();
                // `settle_stale` may land GPU writes into private write-watched
                // title memory without updating `DIR`, so void any standing
                // `VICTORY_ASK_SHARED` asking before re-asking.
                crate::watch::ask_once_void();
                if let Some(ranges) = &watch_ranges {
                    crate::watch::ask_once(ranges);
                }
            }
        }
        if prep::ahead_of_writes() {
            batch.refresh_ahead(runs);
        }
        {
            #[cfg(test)]
            let _scope = enter_retire_scope(exec_scope.clone());
            for list in runs {
                list.run_deferred(&mut batch, self, &mut submitted);
            }
        }
        batch.fill();
        batch.flush();
        drop(watch_guard);
        if batch.rebar_init_failed {
            fail(format_args!(
                "ReBAR UAV mirror readback failed before a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        if batch.fill_failed {
            fail(format_args!(
                "title-to-upload fill failed after retrying lazy materialization before a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        if batch.refresh_failed {
            fail(format_args!(
                "shadow refresh failed after lazy materialization failure before a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        let flown = late && relay::relay();
        if flown && self.card_results_low() {
            let settling = std::time::Instant::now();
            deferred_woken.add(self.settle(&mut submitted, relay::WHY_MOST));
            settled += settling.elapsed();
        }
        let card = self.card_before(card_declared.unwrap_or_default(), flown, &mut submitted);
        let mut rebar_hash_plan = if rebar_entries.is_empty() {
            RebarHashPlan::default()
        } else {
            self.prepare_rebar_hash_plan(&mut rebar_entries)
        };
        let prepared = std::time::Instant::now();

        if late {
            let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len() + 2);
            {
                #[cfg(test)]
                let _scope = enter_retire_scope(exec_scope.clone());
                for list in runs {
                    if let Some(prelude) = list.prelude() {
                        lists.push(Some(prelude));
                    }
                    lists.push(list.list.cast().ok());
                }
            }
            if let Some(check) = card.as_ref().and_then(|c| c.check.clone()) {
                lists.push(Some(check));
            }
            *submitted += 1;
            let signalled = {
                let _lazy_order = crate::lazy_back::direct_submit_guard();
                // SAFETY: closed lists; the fence is ours under `submitted`.
                let signalled = unsafe {
                    let calling = std::time::Instant::now();
                    self.queue.ExecuteCommandLists(&lists);
                    crate::timeline::HELD_US[3]
                        .fetch_add(calling.elapsed().as_micros() as u64, Ordering::Relaxed);
                    self.queue.Signal(&self.fence, *submitted)
                };
                if signalled.is_ok() {
                    self.note_signalled(*submitted);
                }
                signalled
            };
            if let Err(e) = signalled {
                fail(format_args!(
                    "host GPU fence signal failed after an async submission of {} lists ({e}){}{}",
                    runs.len(),
                    self.removal_data(runs),
                    last_notes(runs)
                ));
            }
            let mut readback_ranges = Vec::new();
            for list in runs {
                for rb in &list.deferred().readbacks {
                    let span = if rb.dst_pitch == rb.src_pitch {
                        rb.total
                    } else {
                        rb.lines.saturating_mul(rb.dst_pitch)
                    };
                    readback_ranges.push((rb.dst, rb.dst.saturating_add(span)));
                }
            }
            let writes = runs
                .iter()
                .flat_map(|l| l.deferred().writes.clone())
                .collect();
            SUBMISSION_US[0].fetch_add(
                (prepared - started).saturating_sub(settled).as_micros() as u64,
                Ordering::Relaxed,
            );
            // Any woken timelines from an inline settle cannot run while
            // `submitted` is returned held; hand them to `victory-late-wait`.
            if !deferred_woken.is_empty() {
                relay::enqueue_woken(deferred_woken);
            }
            return Some((
                submitted,
                Sent {
                    number: this_submission,
                    card,
                    back,
                    readback_ranges,
                    writes,
                    exec_epoch,
                },
            ));
        }

        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let timed = timing();
        #[cfg(not(feature = "diagnostics"))]
        let timed = false;
        let mut pre_hash_ok = true;
        if timed && let Some(pre) = rebar_hash_plan.pre_list.take() {
            pre_hash_ok = self.submit_helper_list(&mut submitted, pre, "ReBAR pre-hash");
        }
        let last_batch = batches.len().saturating_sub(1);
        for (batch_idx, range) in batches.into_iter().enumerate() {
            let batch_runs = &runs[range];
            let mut lists: Vec<Option<ID3D12CommandList>> =
                Vec::with_capacity(2 * batch_runs.len() + 3);
            if !timed
                && batch_idx == 0
                && let Some(pre) = rebar_hash_plan.pre_list.take()
            {
                lists.push(Some(pre));
            }
            let mut firsts = Vec::with_capacity(batch_runs.len());
            {
                #[cfg(test)]
                let _scope = enter_retire_scope(exec_scope.clone());
                for list in batch_runs {
                    firsts.push(lists.len());
                    if let Some(prelude) = list.prelude() {
                        lists.push(Some(prelude));
                    }
                    lists.push(list.list.cast().ok());
                }
            }
            if batch_idx == last_batch
                && let Some(check) = card.as_ref().and_then(|c| c.check.clone())
            {
                lists.push(Some(check));
            }
            if !timed
                && batch_idx == last_batch
                && let Some(post) = rebar_hash_plan.post_list.take()
            {
                lists.push(Some(post));
            }
            let groups: Vec<&[Option<ID3D12CommandList>]> = if timed {
                firsts
                    .iter()
                    .zip(firsts.iter().skip(1).chain([&lists.len()]))
                    .map(|(&a, &b)| &lists[a..b])
                    .collect()
            } else {
                vec![&lists[..]]
            };
            for (i, group) in groups.iter().enumerate() {
                *submitted += 1;
                let started = std::time::Instant::now();
                // SAFETY: closed lists; the fence and event are ours. Prior
                // batches have been waited for, so any reused prelude
                // allocator is idle before the Reset inside prelude().
                unsafe {
                    let signalled = {
                        let _lazy_order = crate::lazy_back::direct_submit_guard();
                        #[cfg(test)]
                        if let Some(hook) =
                            AFTER_DIRECT_SUBMIT_GUARD_HOOK.with(|h| h.borrow_mut().take())
                        {
                            hook();
                        }
                        let calling = std::time::Instant::now();
                        self.queue.ExecuteCommandLists(group);
                        crate::timeline::HELD_US[3]
                            .fetch_add(calling.elapsed().as_micros() as u64, Ordering::Relaxed);
                        let res = self.queue.Signal(&self.fence, *submitted);
                        if res.is_ok() {
                            self.note_signalled(*submitted);
                        }
                        res
                    };
                    match signalled {
                        Ok(()) => match self.fence.SetEventOnCompletion(*submitted, self.event) {
                            Ok(()) => {
                                let waited = WaitForSingleObject(self.event, INFINITE);
                                if waited != WAIT_OBJECT_0 {
                                    tracing::warn!(
                                        wait = waited.0,
                                        "host fence event wait failed; polling fence"
                                    );
                                    self.poll_fence(*submitted);
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "host fence event registration failed; polling fence"
                                );
                                self.poll_fence(*submitted);
                            }
                        },
                        Err(e) => fail(format_args!(
                            "host GPU fence signal failed after a submission of {} lists ({e}){}{}",
                            runs.len(),
                            self.removal_data(runs),
                            last_notes(runs)
                        )),
                    }
                }
                #[cfg(feature = "diagnostics")]
                if groups.len() > 1 && started.elapsed() > std::time::Duration::from_millis(50) {
                    // SAFETY: a plain query.
                    let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                    let commands = batch_runs[i]
                        .notes
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .len();
                    tracing::warn!(
                        ms = started.elapsed().as_millis(),
                        commands,
                        "slow host list"
                    );
                    for (ms, op) in batch_runs[i].slowest(frequency, 6) {
                        tracing::warn!("slow list command, {ms:.1} ms: {op}");
                    }
                }
                let _ = (i, started);
            }
            let _ = &firsts;
        }
        let mut post_compare_ok = pre_hash_ok;
        if timed && let Some(post) = rebar_hash_plan.post_list {
            post_compare_ok =
                pre_hash_ok && self.submit_helper_list(&mut submitted, post, "ReBAR post-compare");
        }
        relay::FINISHED.store(this_submission, Ordering::Release);
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add(
            (prepared - started).saturating_sub(settled).as_micros() as u64,
            Ordering::Relaxed,
        );
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| {
                    l.pipelines
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone()
                })
                .collect();
            tracing::warn!(
                ms = gpu.as_millis(),
                lists = runs.len(),
                ?pipelines,
                "slow host submission"
            );
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        let copying = std::time::Instant::now();
        if let Some(card) = &card {
            self.card_back(card, &mut submitted, &mut std::collections::HashSet::new());
        }
        if let Some(card) = card {
            self.card_done(card);
        }
        for (shadow, ranges) in &back {
            if shadow.carded() {
                shadow.copy_back_done(ranges);
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::timeline::wait_stats::copied_back(
            *submitted,
            non_rebar_writes.iter().map(|w| (w.1, w.2)).collect(),
        );
        for (shadow, address, len) in non_rebar_writes {
            let written = shadow.copy_back(address, len);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
        if !rebar_entries.is_empty()
            && !self.finish_rebar_copy_back(
                &mut submitted,
                rebar_entries,
                if post_compare_ok {
                    rebar_hash_plan.total_flag_pieces
                } else {
                    0
                },
            )
        {
            fail(format_args!(
                "ReBAR UAV readback failed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        for list in runs {
            if !read_backs(list) {
                fail(format_args!(
                    "host readback failed: lazy copy-back materialization failed after a submission of {} lists{}{}",
                    runs.len(),
                    self.removal_data(runs),
                    last_notes(runs)
                ));
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        // Queues these writes release run their work once no lock is held.
        let mut woken = deferred_woken;
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                #[cfg(feature = "diagnostics")]
                let _scan = crate::timeline::wait_stats::scanning(
                    crate::timeline::wait_stats::Scan::DoneWrite {
                        sub: *submitted,
                        at,
                        bytes,
                    },
                );
                woken.add(what.write(at as *mut u8, bytes));
            }
        }
        // GPU memory waits that what shaders wrote (copied back above)
        // satisfies: woken now with `VICTORY_FAST_WAKE`, else at the fence
        // watcher's next poll.
        if !old_completion_scan() {
            #[cfg(feature = "diagnostics")]
            let scan = crate::timeline::wait_stats::scanning(
                crate::timeline::wait_stats::Scan::Completion { sub: *submitted },
            );
            woken.add(crate::fence::written(0, 0));
            #[cfg(feature = "diagnostics")]
            drop(scan);
        }
        self.log_messages();
        if relay::relay() {
            crate::shadow::relay_fetched(*relay::QUEUE_TIMELINE.lock_queue_order());
        }
        drop(submitted);
        drop(exec_epoch);
        woken.run();
        relay::drain_woken();
        None
    }
}

/// Submissions of lists whose indirect work budget ran out
/// ([`HostList::budget`]): their indirect records after that ran empty.
pub(crate) static BUDGET_SPENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Copies a finished list's texture-to-buffer and texture readbacks into title memory.
/// Returns `false` if lazy materialization failed for any readback destination;
/// rows that could not be materialized or written are never counted in `COPY_BACK[1]`.
pub(super) fn read_backs(list: &HostList) -> bool {
    list.note_budget_spent();
    let deferred = list.deferred();
    let mut ok = true;
    for rb in &deferred.readbacks {
        let combined = rb.query.as_ref().map(|resolve| {
            let spares = resolve.spares().1 as usize;
            // SAFETY: the list has run; its readback memory holds the
            // `total` bytes of results and 8 more for each spare query
            // (`defer_query_results`), mapped until the list is reset.
            let resolved =
                unsafe { std::slice::from_raw_parts(rb.src as *const u8, rb.total + 8 * spares) };
            resolve.combined(resolved)
        });
        let src = combined
            .as_ref()
            .map_or(rb.src as *const u8, |results| results.as_ptr());
        let written = if rb.dst_pitch == rb.src_pitch {
            crate::watch::writing(rb.dst, rb.total);
            match write_mapped(rb.dst as *mut u8, src, rb.total) {
                Some(n) => n,
                None => {
                    ok = false;
                    0
                }
            }
        } else {
            let mut n = 0;
            for line in 0..rb.lines {
                let to = rb.dst + line * rb.dst_pitch;
                let from = rb.src + line * rb.src_pitch;
                crate::watch::writing(to, rb.row_bytes);
                match write_mapped(to as *mut u8, from as *const u8, rb.row_bytes) {
                    Some(bytes) => n += bytes,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            n
        };
        if written > 0 {
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
    }
    if !ok {
        return false;
    }
    for back in &deferred.texture_backs {
        crate::texback::land(back);
    }
    true
}

/// Pre-lock inspection of `Step::GpuWrite` declarations in closed `runs`,
/// deduplicated per `(shadow_ptr, address, len)` to match [`Batch::gpu_write`]
/// and grouped in O(R) via [`collect_declared_writes`].
fn collect_submission_writes(
    runs: &[&HostList],
    late: bool,
) -> (
    Vec<relay::Declared>,
    Vec<RebarWriteEntry>,
    Vec<DeclaredWrite>,
) {
    let mut seen = std::collections::HashSet::new();
    let mut writes = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            let Step::GpuWrite(shadow, address, len) = step else {
                continue;
            };
            let (address, len) = (*address, *len);
            if seen.insert((std::sync::Arc::as_ptr(shadow) as usize, address, len)) {
                writes.push((shadow.clone(), address, len));
            }
        }
    }
    let (back, mut rebar_entries, non_rebar_writes) = collect_declared_writes(writes, late);
    for entry in &mut rebar_entries {
        entry.declared_spans = merge_ranges(std::mem::take(&mut entry.declared_spans));
        entry.piece_runs = spans_to_piece_runs(&entry.declared_spans);
    }
    (back, rebar_entries, non_rebar_writes)
}

/// Subset of `back` belonging to card-backed host copies.
fn collect_card_declared(back: &[relay::Declared]) -> Vec<relay::Declared> {
    back.iter()
        .filter(|(s, _)| s.carded())
        .map(|(s, r)| (s.clone(), r.clone()))
        .collect()
}

fn collect_ask_once_ranges(runs: &[&HostList]) -> Vec<(usize, usize)> {
    let mut watch_ranges = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                Step::Refresh(_, address, len) if *len > 0 => {
                    watch_ranges.push((*address, address.saturating_add(*len)));
                }
                Step::Texture(staging, _) => {
                    let (start, end) = staging.memory_range();
                    if start < end {
                        watch_ranges.push((start, end));
                    }
                }
                _ => {}
            }
        }
    }
    watch_ranges
}

/// Index ranges into a submission where each `HostList` pointer appears at
/// most once. Used so [`HostList::prelude`] is never reset while a prior
/// recording of the same list is still queued for `ExecuteCommandLists`.
fn unique_list_batches(ptrs: &[usize]) -> Vec<std::ops::Range<usize>> {
    let mut batches = Vec::new();
    let mut start = 0;
    let mut seen = std::collections::HashSet::new();
    for (i, &ptr) in ptrs.iter().enumerate() {
        if !seen.insert(ptr) {
            batches.push(start..i);
            seen.clear();
            seen.insert(ptr);
            start = i;
        }
    }
    if start < ptrs.len() {
        batches.push(start..ptrs.len());
    }
    batches
}

impl Host {
    /// Fallback completion wait when registering or waiting on `self.event`
    /// fails after `self.queue.Signal(&self.fence, target)` succeeded, so
    /// copy-back, writes, and epoch retirement never run while the GPU may
    /// still be executing.
    fn poll_fence(&self, target: u64) {
        let mut spins = 0u32;
        loop {
            // SAFETY: plain fence and device queries on our live host objects.
            let completed = unsafe { self.fence.GetCompletedValue() };
            if completed >= target || completed == u64::MAX {
                return;
            }
            // SAFETY: plain device-removed query.
            if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                return;
            }
            if spins < 64 {
                spins += 1;
                std::thread::yield_now();
            } else {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        // Without it the queries fail as unsupported, which reads like a
        // missing GPU feature.
        if !dred_on() {
            #[cfg(feature = "diagnostics")]
            if crumbs_on() {
                let _ = runs;
                return crumb_report();
            }
            return "; a diagnostics build with VICTORY_D3D12_DRED=1 records where the GPU stopped"
                .to_owned();
        }
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                        #[cfg(test)]
                        self.d3d12_errors.fetch_add(1, Ordering::Relaxed);
                        if first_error.is_none() {
                            first_error = Some(text.to_owned());
                        }
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource or a typed null view.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The id occlusion query tracking knows this list's recordings by.
    pub(crate) fn recording(&self) -> usize {
        self.recording
    }

    /// Keeps `blocks` (memory handed out for the recording) until the list
    /// finishes executing and is recycled or dropped.
    pub(crate) fn keep_uploads(&self, blocks: Vec<Vec<u8>>) {
        if !blocks.is_empty() {
            self.kept_uploads
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend(blocks);
        }
    }

    /// Holds `what` until the list is reset or dropped, which is after its
    /// last submission has run: what gives back a resource when its last
    /// holder lets go (a buffer view table's raw view slot) is not given
    /// back while this list may still use it. Holding the same thing again
    /// changes nothing.
    pub(crate) fn hold(&self, what: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        let at = std::sync::Arc::as_ptr(&what).cast::<()>() as usize;
        self.deferred().held.entry(at).or_insert(what);
    }

    /// Keeps dynamic descriptor slots alive until this host list finishes
    /// executing and is recycled or dropped.
    pub(crate) fn keep_dynamic(&self, slots: Vec<(usize, u32, u32)>) {
        if !slots.is_empty() {
            self.kept_dynamic
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend(slots);
        }
    }

    /// Frees kept upload blocks and returns kept dynamic descriptor slots.
    fn release_kept(&self) {
        self.kept_uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        let dynamic =
            std::mem::take(&mut *self.kept_dynamic.lock().unwrap_or_else(|e| e.into_inner()));
        crate::descriptor::release_dynamic_batch(&dynamic);
    }

    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
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
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC {
                                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                                Count: STAMPS,
                                NodeMask: 0,
                            },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: (count + 1) * 8,
        };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (
                    ticks as f64 * 1e3 / frequency.max(1) as f64,
                    notes[i].0.chars().take(300).collect(),
                )
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    /// Track resolved buffers by resource identity, including aliasing title addresses.
    pub(crate) fn use_shadow(&self, buffer: &ID3D12Resource, wanted: D3D12_RESOURCE_STATES) {
        self.mark_work();
        let mut states = self.shadow_states.lock().unwrap_or_else(|e| e.into_inner());
        let (_, state) = states
            .entry(buffer.as_raw() as usize)
            .or_insert_with(|| (buffer.clone(), D3D12_RESOURCE_STATE_COMMON));
        let before = *state;
        if let Some(after) = shadow_transition(before, wanted) {
            // SAFETY: this open list retains the resource and tracks its previous state.
            unsafe {
                self.list
                    .ResourceBarrier(&[transition(buffer, before, after)])
            };
            *state = after;
            self.shadow_state_gen.fetch_add(1, Ordering::Release);
        } else if before == D3D12_RESOURCE_STATE_COMMON {
            *state = wanted;
            self.shadow_state_gen.fetch_add(1, Ordering::Release);
        }
    }

    /// Monotonic count of shadow-buffer state transitions recorded on this
    /// list since reset (`VICTORY_LIST_VIEWS` buffer-SRV readiness key).
    pub(crate) fn shadow_state_gen(&self) -> u64 {
        self.shadow_state_gen.load(Ordering::Acquire)
    }

    /// Current tracked D3D12 access state for `buffer` on this list (tests).
    #[cfg(test)]
    pub(crate) fn shadow_state_of(&self, buffer: &ID3D12Resource) -> Option<D3D12_RESOURCE_STATES> {
        self.shadow_states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(buffer.as_raw() as usize))
            .map(|(_, state)| *state)
    }

    pub(crate) fn close(&self) {
        // The last noted command's breadcrumb: done once it finishes.
        #[cfg(feature = "diagnostics")]
        if crumbs_on() {
            let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
            if !notes.is_empty() {
                self.crumb(notes.len() as u32);
            }
        }
        // Lists can execute together, in a different order, or more than once.
        // Buffers do not decay between lists in the same ExecuteCommandLists.
        // Restore a known entry state for the next recording.
        let states = self.shadow_states.lock().unwrap_or_else(|e| e.into_inner());
        if !states.is_empty() {
            let barriers: Vec<_> = states
                .values()
                .map(|(buffer, state)| transition(buffer, *state, D3D12_RESOURCE_STATE_COMMON))
                .collect();
            // SAFETY: an open list and resources retained until reset.
            unsafe { self.list.ResourceBarrier(&barriers) };
        }
        drop(states);
        // The scratch buffers its prelude fills and leaves readable as
        // arguments, plus any hoisted clamp buffers read by this list:
        // returned to COMMON at list close.
        {
            let mut close_barriers = Vec::new();
            let gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
            for set in gather.sets.iter().filter(|s| s.records() > 0) {
                close_barriers.push(transition(
                    &set.scratch,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                ));
            }
            drop(gather);
            let hoisted = self
                .hoisted_buffers
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            for buf in hoisted.iter() {
                close_barriers.push(transition(
                    buf,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                ));
            }
            drop(hoisted);
            if !close_barriers.is_empty() {
                // SAFETY: an open list and live buffers left in INDIRECT_ARGUMENT.
                unsafe { self.list.ResourceBarrier(&close_barriers) };
            }
        }
        self.twins_to_depth();
        self.textures_back();
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count =
                (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list
                    .ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    fn retire_state(&self) -> &Mutex<RetireState> {
        #[cfg(test)]
        if let Some(scope) = self.retire_scope.as_deref() {
            return scope;
        }
        &RETIRE_STATE
    }

    fn ensure_epoch(&self, _guard: &std::sync::MutexGuard<'_, ()>) {
        if self.epoch.load(Ordering::Acquire) == 0 {
            let e = acquire_epoch_in(self.retire_state(), self.work_seq.clone());
            self.epoch.store(e, Ordering::Release);
        }
    }

    /// Pins a retirement epoch for this list once it records GPU work, and
    /// advances `work_seq` lock-free when new resources have been created
    /// since the last command on this list.
    pub(crate) fn mark_work(&self) {
        let seq = RESOURCE_SEQ.load(Ordering::SeqCst);
        if self.epoch.load(Ordering::Acquire) != 0 {
            if self.work_seq.load(Ordering::SeqCst) < seq {
                self.work_seq.fetch_max(seq, Ordering::SeqCst);
            }
            return;
        }
        let guard = self.epoch_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.work_seq.fetch_max(seq, Ordering::SeqCst);
        self.ensure_epoch(&guard);
    }

    /// Pins an epoch and increments the in-flight submission count for this
    /// list before queuing it on a command queue timeline.
    pub(crate) fn begin_submit(&self) {
        let guard = self.epoch_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        self.ensure_epoch(&guard);
    }

    /// Releases a queued submission pin and, if it was the last in-flight
    /// submission for this list, releases the list's retirement epoch.
    /// Keeps `work_seq` intact so a subsequent resubmission of the same closed
    /// list still protects resources recorded into it.
    pub(crate) fn end_submit(&self) {
        let released = {
            let _guard = self.epoch_lock.lock().unwrap_or_else(|e| e.into_inner());
            if self.in_flight.fetch_sub(1, Ordering::AcqRel) == 1 {
                self.epoch.swap(0, Ordering::AcqRel)
            } else {
                0
            }
        };
        if released != 0 {
            release_epoch_in(self.retire_state(), released);
        }
    }

    /// Releases a queued submission pin for a cancelled (unexecuted) submission
    /// without clearing the list's recording epoch: if the title still owns
    /// this closed list, its epoch stays pinned until it executes, resets, or
    /// drops; if the cancelled submission held the last reference, dropping
    /// the `HostList` immediately releases the epoch.
    pub(crate) fn cancel_submit(&self) {
        let _guard = self.epoch_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
    }

    pub(crate) fn has_epoch(&self) -> bool {
        self.epoch.load(Ordering::Acquire) != 0
    }

    fn release_epoch(&self) {
        self.release_epoch_inner(false);
    }

    fn release_epoch_inner(&self, clear_work_seq: bool) {
        let released = {
            let _guard = self.epoch_lock.lock().unwrap_or_else(|e| e.into_inner());
            if self.in_flight.load(Ordering::Acquire) == 0 {
                if clear_work_seq {
                    self.work_seq.store(0, Ordering::Release);
                }
                self.epoch.swap(0, Ordering::AcqRel)
            } else {
                0
            }
        };
        if released != 0 {
            release_epoch_in(self.retire_state(), released);
        }
    }

    /// Clears per-recording tracking state shared between [`Self::reset`] and
    /// [`Self::release_for_pool`].
    fn clear_tracking(&self) {
        // The recording is over: its occlusion query runs that no later
        // list resolves are let go.
        crate::query::forget_list(self.recording);
        self.release_kept();
        self.shadow_states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        *self.deferred() = Deferred::default();
        self.states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.shadow_state_gen.store(0, Ordering::Release);
        self.state_gen.store(0, Ordering::Relaxed);
        self.dirty_resources
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.pipelines
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.writes.store(0, Ordering::Relaxed);
        {
            let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
            for (_, state) in &mut chunks.all {
                *state = D3D12_RESOURCE_STATE_COMMON;
            }
            chunks.used = 0;
            chunks.at = 0;
            chunks.barrier = false;
            chunks.bound = false;
        }
        self.hoisted_buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        {
            let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
            let Gather {
                gate, sets, spare, ..
            } = &mut *gather;
            gate.reset();
            for set in sets.drain(..) {
                if spare.len() < GATHER_SETS {
                    spare.push(set.into_scratch());
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.crumb.store(u32::MAX, Ordering::Relaxed);
    }

    /// Releases all retained memory and retirement pins before parking an idle,
    /// finished list in `Host::free_lists`. D3D12 allocator/list resets still
    /// run in [`Self::reset`] when [`Host::acquire_list`] pops it for reuse.
    fn release_for_pool(&self) {
        self.clear_tracking();
        {
            *self.indirect_args.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
        {
            let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
            chunks.all.clear();
        }
        {
            let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
            gather.spare.clear();
        }
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            scratch.chunks.clear();
            (scratch.current, scratch.used) = (0, 0);
        }
        {
            let mut rect = self.rect_scratch.lock().unwrap_or_else(|e| e.into_inner());
            rect.vertices = None;
            rect.args = None;
            rect.counters.clear();
            (rect.counter_chunk, rect.counter_used) = (0, 0);
        }
        self.release_epoch();
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        self.buffer_writes.store(true, Ordering::Relaxed);
        self.clear_tracking();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            // Keep at most one warm 4 MiB upload chunk so steady-state active
            // lists avoid re-allocating while multi-chunk spikes are freed.
            scratch.chunks.truncate(1);
            (scratch.current, scratch.used) = (0, 0);
        }
        {
            let mut rect = self.rect_scratch.lock().unwrap_or_else(|e| e.into_inner());
            // Keep at most one warm rect counter chunk across resets.
            rect.counters.truncate(1);
            (rect.counter_chunk, rect.counter_used) = (0, 0);
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        self.release_epoch_inner(true);
    }

    /// The host state currently tracked for `resource` on this list, falling
    /// back to the resource's last submitted/initial host state if this list
    /// has not transitioned it yet.
    #[cfg(test)]
    pub(crate) fn tracked_state(&self, resource: crate::object::This) -> Option<u32> {
        let copy = crate::resource::host_copy(resource)?;
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&(_, last, _, _)) = states.get(&(std::sync::Arc::as_ptr(&copy) as usize)) {
            Some(last)
        } else {
            Some(copy.state.load(Ordering::Relaxed))
        }
    }

    /// The last state explicitly requested for `resource` on this list (via a
    /// title barrier, copy, clear, target bind, or non-bindless descriptor
    /// table), ignoring states chosen by bindless heap sweeps.
    pub(crate) fn explicit_state(&self, resource: crate::object::This) -> Option<u32> {
        let copy = crate::resource::host_copy_ref(resource)?;
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        states
            .get(&(std::sync::Arc::as_ptr(copy) as usize))
            .and_then(|&(_, _, explicit, _)| explicit)
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        self.transition_inner(resource, wanted, true)
    }

    /// Same as [`HostList::transition`], for bindless heap sweeps: updates the
    /// tracked host state without overwriting the title's explicit access
    /// intent recorded by barriers or explicit binds.
    pub(crate) fn transition_bindless(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        self.transition_inner(resource, wanted, false)
    }

    fn transition_inner(
        &self,
        resource: crate::object::This,
        wanted: u32,
        explicit: bool,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let copy = crate::resource::host_copy_ref(resource)?;
        let texture = &copy.texture;
        self.mark_work();
        let wanted = legal_state(texture, wanted);
        let (changed, result) = {
            let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
            match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert((wanted, wanted, explicit.then_some(wanted), copy.clone()));
                    (true, None)
                }
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    let entry = o.get_mut();
                    let before = std::mem::replace(&mut entry.1, wanted);
                    if explicit {
                        entry.2 = Some(wanted);
                    }
                    if before != wanted {
                        (true, Some((texture, before)))
                    } else {
                        (false, None)
                    }
                }
            }
        };
        if changed {
            self.dirty_resources
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(resource as usize);
            self.state_gen.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// The current state established by this list. A texture the list has
    /// never used can arrive in a different state at submission.
    pub(crate) fn tracked_texture_state(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
    ) -> Option<u32> {
        self.states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(std::sync::Arc::as_ptr(copy) as usize))
            .map(|&(_, last, _, _)| last)
    }

    /// Monotonic generation incremented whenever a texture is first tracked
    /// or changes state in this list.
    pub(crate) fn state_gen(&self) -> u64 {
        self.state_gen.load(Ordering::Relaxed)
    }

    /// Current cursor length of [`HostList::dirty_resources`].
    pub(crate) fn dirty_len(&self) -> usize {
        self.dirty_resources
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Returns `(new_cursor, resources)` of title texture pointers whose
    /// tracked state changed at or after `cursor`.
    pub(crate) fn dirty_since(&self, cursor: usize) -> (usize, Vec<usize>) {
        let dirty = self
            .dirty_resources
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let len = dirty.len();
        let slice = dirty.get(cursor..).unwrap_or(&[]);
        (len, slice.to_vec())
    }

    /// Moves `copy` to `wanted` (a host state) in this list, recording the
    /// barrier if the list had it elsewhere. Like [`Self::transition`] for a
    /// host texture the title has no resource for (a depth texture's twin).
    pub(crate) fn settle(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) {
        let wanted = legal_state(&copy.texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let before = match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, None, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        };
        drop(states);
        if let Some(before) = before {
            let barrier = transition(
                &copy.texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// A use of title texture `resource` about to be recorded, if it is a
    /// depth texture with a colour twin (false if it is not): of the twin
    /// (`colour`: unordered access) or of the depth texture (depth target,
    /// clear, shader read, copy, resolve), writing it or only reading. The
    /// side used is first brought up to date: by a copy recorded here if
    /// this list wrote the other side since, by the list's prelude if the
    /// list uses the side as it finds it. The twin is left in the
    /// unordered access state; the caller moves the depth texture to the
    /// state its use needs, as for any texture. A twin still ahead when the
    /// list closes is copied then (`twins_to_depth`).
    /// shortcut: every depth target bound for a draw counts as written
    /// (also when the draw only tests), and every unordered access view a
    /// shader names as written; views a shader picks from the heap by a
    /// computed index are not seen here at all.
    pub(crate) fn twin_use(
        &self,
        resource: crate::object::This,
        colour: bool,
        write: bool,
    ) -> bool {
        const UNORDERED_ACCESS: u32 = 0x8;
        // Asked for every texture a draw reads: most have no twin.
        let Some(twin) = crate::resource::twin(resource) else {
            return false;
        };
        let Some(depth) = crate::resource::host_copy(resource) else {
            return false;
        };
        let side = usize::from(colour);
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let u = twins
            .entry(std::sync::Arc::as_ptr(&twin) as usize)
            .or_insert_with(|| TwinUse {
                depth: depth.clone(),
                twin: twin.clone(),
                needs: [false; 2],
                behind: [None; 2],
                writing: false,
                late: false,
            });
        let copy = match u.behind[side] {
            Some(behind) => behind,
            None => {
                u.needs[side] = true;
                false
            }
        };
        u.behind[side] = Some(false);
        match (write, colour) {
            // Shaders write the twin when the draw or dispatch runs: what
            // the same one reads of the depth texture is prepared before
            // (`twins_written`).
            (true, true) => u.writing = true,
            (true, false) => u.behind[1] = Some(true),
            (false, _) => {}
        }
        // shortcut: the depth plane's twin is not drawn into the plane
        // here: the pass sets a pipeline and a target of ours, which a list
        // still recording the title's commands cannot take. This use sees
        // the plane as it was; the pass runs when the list closes
        // (`twins_to_depth`). Counted.
        if copy && !colour && twin.plane.is_some() {
            u.late = true;
            crate::resource::PLANES[2].fetch_add(1, Ordering::Relaxed);
        }
        drop(twins);
        if copy {
            self.twin_copy(&depth, &twin, colour);
        } else if colour {
            self.settle(&twin.colour, UNORDERED_ACCESS);
        }
        // The depth plane's twin is written through unordered access too.
        if colour && let Some(plane) = &twin.plane {
            self.settle(plane, UNORDERED_ACCESS);
        }
        true
    }

    /// Records the pass that draws the twin of the depth plane of `depth`
    /// into the plane, if `twin` has one, with the moves to the states the
    /// pass needs; the twin is left in the unordered access state. Only
    /// for a list that records nothing of the title's afterwards
    /// (`record_plane_pass`).
    fn pass_plane(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const DEPTH_WRITE: u32 = 0x10;
        const PIXEL_SHADER_RESOURCE: u32 = 0x80;
        let Some(plane) = &twin.plane else { return };
        self.settle(depth, DEPTH_WRITE);
        self.settle(plane, PIXEL_SHADER_RESOURCE);
        record_plane_pass(
            &self.list,
            &depth.texture,
            &plane.texture,
            &twin.plane_views,
        );
        self.settle(plane, UNORDERED_ACCESS);
    }

    /// Records the copy of a depth texture into its colour twin (`colour`)
    /// or of the twin into the depth texture, with the moves to the copy
    /// states. The twin is left in the unordered access state.
    fn twin_copy(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
        colour: bool,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let (from, to) = if colour {
            (depth, &twin.colour)
        } else {
            (&twin.colour, depth)
        };
        self.settle(from, COPY_SOURCE);
        self.settle(to, COPY_DEST);
        record_twin_copy(&self.list, &depth.texture, twin, colour);
        self.settle(&twin.colour, UNORDERED_ACCESS);
    }

    /// Before the list closes: the depth textures whose twins it wrote last
    /// are given what was written, and left readable by shaders. Shaders
    /// that pick a view from the heap by a computed index are not seen by
    /// `twin_use` and read the depth texture in lists that never name it.
    fn twins_to_depth(&self) {
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let written: Vec<_> = twins
            .values_mut()
            .filter(|u| u.behind[0] == Some(true) || u.late)
            .map(|u| {
                // Not copied again if the list copied since the last write
                // and only the depth plane's pass is left.
                let copy = u.behind[0] == Some(true);
                u.behind[0] = Some(false);
                u.late = false;
                (u.depth.clone(), u.twin.clone(), copy)
            })
            .collect();
        drop(twins);
        for (depth, twin, copy) in &written {
            if *copy {
                self.twin_copy(depth, twin, false);
            }
            // Nothing of the title's is recorded after this: the pass may
            // leave its own pipeline and target set.
            self.pass_plane(depth, twin);
            self.settle(depth, SHADER_RESOURCE);
        }
    }

    /// The draw or dispatch just prepared is recorded: the twins it writes
    /// now hold what their depth textures lack.
    pub(crate) fn twins_written(&self) {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values_mut().filter(|u| u.writing) {
            u.writing = false;
            u.behind[0] = Some(true);
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with. For depth textures with a
    /// colour twin it first copies the side the list uses as it finds it
    /// from the other, if a list submitted earlier left it behind.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let host_state = |state: u32| D3D12_RESOURCE_STATES(state as i32);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies (depth texture, twin, whether into the twin); the
        // sides are then as this list leaves them.
        let mut syncs = Vec::new();
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values() {
            let real = u.twin.behind.load(Ordering::Relaxed);
            let mut behind = [real & 1 != 0, real & 2 != 0];
            for side in [0, 1] {
                if u.needs[side] && behind[side] {
                    syncs.push((u.depth.clone(), u.twin.clone(), side == 1));
                    behind = [false; 2];
                }
            }
            let mut left = [0, 1].map(|side| u.behind[side].unwrap_or(behind[side]));
            if u.late {
                left[0] = true;
            }
            let left = u32::from(left[0]) | (u32::from(left[1]) << 1);
            u.twin.behind.store(left, Ordering::Relaxed);
        }
        drop(twins);
        // A copy runs with its textures in the copy states: each moves
        // there from its real state, and from there to the state the list
        // starts with (back to its real state if the list does not use it).
        let mut to_copy = Vec::new();
        let mut barriers = Vec::new();
        let mut copying = std::collections::HashMap::new();
        for (depth, twin, colour) in &syncs {
            let (from, to) = if *colour {
                (depth, &twin.colour)
            } else {
                (&twin.colour, depth)
            };
            for (texture, state) in [(from, COPY_SOURCE), (to, COPY_DEST)] {
                let key = std::sync::Arc::as_ptr(texture) as usize;
                let real = texture.state.load(Ordering::Relaxed);
                copying.insert(key, state);
                if real != state {
                    to_copy.push(transition(
                        &texture.texture,
                        host_state(real),
                        host_state(state),
                    ));
                    if !states.contains_key(&key) {
                        barriers.push(transition(
                            &texture.texture,
                            host_state(state),
                            host_state(real),
                        ));
                    }
                }
            }
        }
        barriers.extend(states.iter().filter_map(|(key, (first, last, _, copy))| {
            let real = copy.state.swap(*last, Ordering::Relaxed);
            let real = copying.get(key).copied().unwrap_or(real);
            (real != *first)
                .then(|| transition(&copy.texture, host_state(real), host_state(*first)))
        }));
        // The records the list's draws read from slots (`crate::indirect::gather`).
        let gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
        let gathered = gather.sets.iter().any(|s| s.records() > 0);
        if barriers.is_empty() && syncs.is_empty() && !gathered {
            return None;
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures and gathered buffers are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                if !to_copy.is_empty() {
                    list.ResourceBarrier(&to_copy);
                }
                for (depth, twin, colour) in &syncs {
                    record_twin_copy(list, &depth.texture, twin, *colour);
                }
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                if gathered {
                    // The slots must hold clamped records before a draw
                    // reads them: a prelude that cannot record the passes
                    // is no prelude.
                    let filled = match &gather.budget {
                        Some(budget) => crate::indirect::record_gathers(list, &gather.sets, budget),
                        None => false,
                    };
                    if !filled {
                        fail(format_args!(
                            "a list's gathered indirect records could not be clamped in its prelude"
                        ));
                    }
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if dred_on() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps()
                && at < STAMPS - 1
            {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list
                    .SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        } else if crumbs_on() {
            let mut notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
            self.crumb(notes.len() as u32);
            notes.push((what(), range));
        }
    }

    /// Writes the breadcrumbs before noted command `at`: begun up to it
    /// (when the front end of the GPU reaches it), done before it (once all
    /// earlier work has finished).
    #[cfg(feature = "diagnostics")]
    fn crumb(&self, at: u32) {
        let Some(crumbs) = crumbs() else { return };
        let Ok(list) = self.list.cast::<ID3D12GraphicsCommandList2>() else {
            return;
        };
        let mut slot = self.crumb.load(Ordering::Relaxed);
        if slot == u32::MAX {
            slot = crumbs.next.fetch_add(1, Ordering::Relaxed) % CRUMB_LISTS;
            self.crumb.store(slot, Ordering::Relaxed);
        }
        let done = crumbs.gpu + u64::from(slot) * 8;
        let params = [
            D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
                Dest: done + 4,
                Value: at + 1,
            },
            D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
                Dest: done,
                Value: at,
            },
        ];
        let modes = [
            D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN,
            D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT,
        ];
        // SAFETY: an open list; the destinations lie in our breadcrumb buffer.
        unsafe { list.WriteBufferImmediate(2, params.as_ptr(), Some(modes.as_ptr())) };
    }

    /// Notes a pipeline the list binds.
    #[cfg(test)]
    pub(crate) fn used_pipeline(&self, name: &str) {
        self.mark_work();
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    /// Notes a pipeline the list binds and keeps it and its root signature
    /// until the list is reset, the first time only: the list keeps the
    /// first, so its address names the same pipeline from then on.
    pub(crate) fn bind_pipeline(&self, p: &crate::pipeline::host::HostPipeline) {
        #[cfg(feature = "diagnostics")]
        crate::shadow::note_bound_pipeline(&p.name);
        self.mark_work();
        let first = {
            let mut deferred = self.deferred();
            let first = deferred.bound.insert(p.pso.as_raw() as usize);
            if first {
                deferred.kept.push(p.pso.clone().into());
                deferred.kept.push(p.root.clone().into());
            }
            first
        };
        if first {
            let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
            if !pipelines.contains(p.name.as_str()) {
                pipelines.insert(p.name.clone());
            }
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to title memory at `dst` when the list
    /// is submitted (`Step::Copy` -> `copy_mapped(..., title_dst = true)`).
    ///
    /// Title destinations keep the `VirtualQuery` commitment guard. Host-owned
    /// mapped upload memory must use [`Self::defer_fill`] instead: under
    /// Wine+vkd3d, `Map` pointers can report as `MEM_FREE` and the guard would
    /// skip the copy (#518).
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.mark_work();
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
        deferred.filled.clear();
        deferred.gpu_written.clear();
        deferred.copy_gen += 1;
    }

    /// Monotonic count of deferred title-memory copies recorded on this list
    /// since reset. Fresh-table dedupe partitions reuse by this generation so
    /// an intervening `CopyBufferRegion`/`CopyResource` creates a new CBV
    /// snapshot (`Fill -> Copy -> Fill`).
    pub(crate) fn copy_gen(&self) -> u64 {
        self.deferred().copy_gen
    }

    /// Copies `len` bytes of title memory at `src` into host-owned upload
    /// memory at `dst` when the list is submitted (`Step::Fill` ->
    /// `copy_mapped(..., title_dst = false)`).
    ///
    /// Buffer->texture staging in `copy_texture_region` uses this; buffer->buffer
    /// title copies stay on [`Self::defer_copy`].
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.mark_work();
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// The host GPU address of `len` bytes of upload memory holding the title
    /// memory at `src` as of the list's submission: one copy for every ask
    /// until a title copy or deferred run intervenes.
    pub(crate) fn fill_scratch(&self, src: *const u8, len: u64) -> Option<u64> {
        let key = (src as usize, len as usize);
        if let Some(&gpu) = self.deferred().filled.get(&key) {
            return Some(gpu);
        }
        let (gpu, to) = self.scratch(len)?;
        self.defer_fill(to, src, len as usize);
        self.deferred().filled.insert(key, gpu);
        Some(gpu)
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        self.mark_work();
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Batched [`crate::shadow::Shadow::refresh_as`] and [`HostList::defer_refresh`]:
    /// acquires `shadow_states` and `deferred` at most once for all
    /// `(shadow, address, len, access)` ranges and emits any resource state
    /// transitions in a single `ResourceBarrier`.
    pub(crate) fn defer_refresh_batch<'a>(
        &self,
        reserve: usize,
        items: impl IntoIterator<
            Item = (
                &'a std::sync::Arc<crate::shadow::Shadow>,
                u64,
                u64,
                D3D12_RESOURCE_STATES,
            ),
        >,
    ) {
        let mut staged: Vec<(&'a std::sync::Arc<crate::shadow::Shadow>, usize, usize)> =
            Vec::with_capacity(reserve);
        let mut any = false;
        {
            let mut states = self.shadow_states.lock().unwrap_or_else(|e| e.into_inner());
            let mut barriers: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
            let mut barrier_keys: Vec<usize> = Vec::new();
            let mut transitioned = false;
            for (shadow, address, len, wanted) in items {
                if !any {
                    self.mark_work();
                    any = true;
                }
                let buffer = &shadow.buffer;
                let raw_key = buffer.as_raw() as usize;
                let (_, state) = states
                    .entry(raw_key)
                    .or_insert_with(|| (buffer.clone(), D3D12_RESOURCE_STATE_COMMON));
                let before = *state;
                if let Some(after) = shadow_transition(before, wanted) {
                    if barrier_keys.contains(&raw_key) {
                        // SAFETY: open list; flush before a second transition on the same resource.
                        unsafe { self.list.ResourceBarrier(&barriers) };
                        barriers.clear();
                        barrier_keys.clear();
                    }
                    barriers.push(transition(buffer, before, after));
                    barrier_keys.push(raw_key);
                    *state = after;
                    transitioned = true;
                } else if before == D3D12_RESOURCE_STATE_COMMON {
                    *state = wanted;
                    transitioned = true;
                }
                if !shadow.is_placed()
                    && let Some((from, clamped)) = shadow.clamp_range(address, len)
                {
                    staged.push((shadow, from as usize, clamped as usize));
                }
            }
            if !barriers.is_empty() {
                // SAFETY: this open list retains every transitioned resource in `states`.
                unsafe { self.list.ResourceBarrier(&barriers) };
            }
            if transitioned {
                self.shadow_state_gen.fetch_add(1, Ordering::Release);
            }
        }
        if !any || staged.is_empty() {
            return;
        }
        let mut deferred = self.deferred();
        deferred.refreshed.reserve(staged.len());
        deferred.copies.reserve(staged.len());
        for (shadow, address, len) in staged {
            let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
            if deferred.refreshed.insert(key) {
                deferred
                    .copies
                    .push(Step::Refresh(shadow.clone(), address, len));
            }
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run
    /// (deduplicated per exact range until a title copy or run intervenes).
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        self.mark_work();
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.gpu_written.insert(key) {
            deferred
                .copies
                .push(Step::GpuWrite(shadow.clone(), address, len));
        }
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.mark_work();
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A draw or dispatch being prepared writes title texture `resource`
    /// through unordered access: if it is of a kind copied back
    /// ([`crate::texback`]), the list copies the exposed subresources when it
    /// closes (fxceless #1152).
    pub(crate) fn texture_written(&self, resource: crate::object::This, view: Option<&[u32; 10]>) {
        let Some(format) = crate::texback::wanted_format(resource) else {
            return;
        };
        let mut deferred = self.deferred();
        if let Some(target) = deferred
            .back_wanted
            .iter_mut()
            .find(|t| t.owner() == resource as usize)
        {
            target.include(view);
        } else if let Some(target) = crate::texback::Target::new(resource, format, view) {
            deferred.back_wanted.push(target);
        }
    }

    #[cfg(test)]
    pub(crate) fn push_texback_target(&self, target: crate::texback::Target) {
        self.deferred().back_wanted.push(target);
    }

    /// Before the list closes: the textures its shaders wrote that are copied
    /// back go into read-back buffers ([`crate::texback::record`]), restoring
    /// each texture's final recorded state afterward (fxceless #1152 + #1170).
    fn textures_back(&self) {
        const COPY_SOURCE: u32 = 0x800;
        let wanted = std::mem::take(&mut self.deferred().back_wanted)
            .into_iter()
            .filter(|target| target.can_tile() && target.is_valid());
        for target in wanted {
            let copy = target.copy.clone();
            let last = self.tracked_texture_state(&copy);
            let needs_barrier = last.is_none_or(|state| state & COPY_SOURCE == 0);
            if needs_barrier {
                self.settle(&copy, COPY_SOURCE);
            }
            let back = crate::texback::record(&self.list, target);
            if needs_barrier && let Some(last) = last {
                self.settle(&copy, last);
            }
            if let Some(back) = back {
                self.mark_work();
                self.deferred().texture_backs.push(back);
            }
        }
    }

    /// Copies a GPU texture readback from `src` into title memory at `dst`
    /// once the list has finished running on the host GPU.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn defer_readback(
        &self,
        dst: *mut u8,
        src: *const u8,
        dst_pitch: usize,
        src_pitch: usize,
        lines: usize,
        row_bytes: usize,
        total: usize,
    ) {
        self.mark_work();
        self.deferred().readbacks.push(Readback {
            dst: dst as usize,
            src: src as usize,
            dst_pitch,
            src_pitch,
            lines,
            row_bytes,
            total,
            query: None,
        });
    }

    /// Copies `count` occlusion query results from the readback memory at
    /// `src` into title memory at `dst` once the list has run, each
    /// combined with the results of its later parts as `resolve` chose them
    /// when the list was submitted: one further result follows the `count`
    /// for each spare query.
    pub(crate) fn defer_query_results(
        &self,
        dst: *mut u8,
        src: *const u8,
        count: usize,
        resolve: std::sync::Arc<crate::query::Resolve>,
    ) {
        let total = count * 8;
        self.mark_work();
        self.deferred().readbacks.push(Readback {
            dst: dst as usize,
            src: src as usize,
            dst_pitch: total,
            src_pitch: total,
            lines: 1,
            row_bytes: total,
            total,
            query: Some(resolve),
        });
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    #[cfg(test)]
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        self.mark_work();
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// A mapped readback buffer of `size` bytes, alive until the list is
    /// reset.
    pub(crate) fn readback(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *const u8)> {
        let (buffer, data) = host.readback_buffer(size)?;
        self.mark_work();
        name(&buffer, || format!("list readback {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `size` bytes of mapped upload memory, 512-byte aligned for a placed
    /// footprint, alive until the list is reset: the buffer, the offset in
    /// it and the CPU pointer.
    pub(crate) fn upload_scratch(&self, size: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        self.scratch_in(size, 512, |buffer, at, _, data| (buffer.clone(), at, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        self.scratch_in(len, 256, |_, _, gpu, data| (gpu, data))
    }

    /// `len` bytes of upload memory at a multiple of `align` until the list
    /// is reset, handed to `f` with the buffer, the offset in it, the host GPU
    /// address and the CPU pointer.
    fn scratch_in<R>(
        &self,
        len: u64,
        align: u64,
        f: impl FnOnce(&ID3D12Resource, u64, u64, *mut u8) -> R,
    ) -> Option<R> {
        let host = host()?;
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.scratch_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            let made = f(&buffer, 0, gpu, data);
            self.mark_work();
            self.deferred().uploads.push(buffer);
            return Some(made);
        }
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        while s.current < s.chunks.len() && s.used.next_multiple_of(align) + len > SCRATCH_CHUNK {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let (buffer, data) = host.scratch_buffer(SCRATCH_CHUNK)?;
            name(&buffer, || "list scratch".into());
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            s.chunks.push((buffer, data as usize, gpu));
            s.used = 0;
        }
        let (ref buffer, data, gpu) = s.chunks[s.current];
        let at = s.used.next_multiple_of(align);
        let made = f(buffer, at, gpu + at, (data + at as usize) as *mut u8);
        s.used = at + len;
        drop(s);
        self.mark_work();
        Some(made)
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        self.mark_work();
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred.copies.push(Step::Host(Box::new(move || unsafe {
                (at as *mut u32).write_volatile(0)
            })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// After the list ran: counts it in [`BUDGET_SPENT`] if its indirect
    /// work budget ([`Self::budget`]) ran out, so indirect records after that
    /// point ran empty, logging at powers of two.
    fn note_budget_spent(&self) {
        let Some(Some((_, data))) = self.budget.get() else {
            return;
        };
        // SAFETY: the list's own mapped word; the list has run.
        let used = unsafe { (*data as *const u32).read_volatile() };
        // `BUDGET` in `shaders/argument_clamp.hlsl` (`indirect_budget` models it).
        if used >= 1 << 31 {
            let n = BUDGET_SPENT.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::warn!(
                    lists = n,
                    "a list spent its indirect work budget: its later indirect draws and dispatches ran empty"
                );
            }
        }
    }

    /// A shader that can store to buffers is bound.
    pub(crate) fn buffers_written(&self) {
        self.buffer_writes.store(true, Ordering::Relaxed);
    }

    /// Whether a shader that can store to buffers was bound since the last
    /// call (the caller emits a UAV barrier before clamping).
    pub(crate) fn take_buffer_writes(&self) -> bool {
        self.buffer_writes.swap(false, Ordering::Relaxed)
    }

    /// Counts a dispatch or a buffer copy: what may write indirect arguments.
    pub(crate) fn count_write(&self) {
        self.writes.fetch_add(1, Ordering::Relaxed);
    }

    /// Dispatches and buffer copies recorded since the reset.
    pub(crate) fn writes(&self) -> u64 {
        self.writes.load(Ordering::Relaxed)
    }

    /// Whether this host list has no recorded work or tracked resource state yet.
    pub(crate) fn untouched(&self) -> bool {
        !self.has_epoch()
            && self
                .shadow_states
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
            && self
                .states
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
            && self
                .twins
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and resets of the
    /// list. Smaller buffers replaced mid-list stay alive until [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            let buf = buffer.clone();
            drop(slot);
            self.mark_work();
            return Some(buf);
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let buffer = host()?.default_buffer(alloc)?;
        name(&buffer, || format!("indirect arguments {alloc:#x}"));
        let old = slot.replace((buffer.clone(), alloc));
        drop(slot);
        self.mark_work();
        if let Some((old_buf, _)) = old {
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    /// Room for `bytes` of hoisted clamped indirect arguments in this list's
    /// buffers, writable by shaders from here on in the list: the buffer and
    /// the offset. `None` for more than a chunk holds.
    pub(crate) fn clamp_room(&self, bytes: u64) -> Option<(ID3D12Resource, u64)> {
        const CHUNK: u64 = 256 << 10;
        let bytes = bytes.next_multiple_of(16);
        if bytes > CHUNK {
            return None;
        }
        let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
        if chunks.used > 0 && chunks.at + bytes <= CHUNK {
            let at = chunks.at;
            chunks.at += bytes;
            let buf = chunks.all[chunks.used - 1].0.clone();
            drop(chunks);
            self.mark_work();
            return Some((buf, at));
        }
        if chunks.used == chunks.all.len() {
            let buffer = host()?.default_buffer(CHUNK)?;
            name(&buffer, || "hoisted indirect arguments".to_owned());
            chunks.all.push((buffer, D3D12_RESOURCE_STATE_COMMON));
        }
        let i = chunks.used;
        chunks.used += 1;
        chunks.at = bytes;
        let mut barriers = Vec::new();
        if !chunks.barrier {
            chunks.barrier = true;
            barriers.push(crate::indirect::uav_barrier());
        }
        let (buffer, state) = &mut chunks.all[i];
        if *state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
            barriers.push(transition(
                buffer,
                *state,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ));
            *state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        let buffer = buffer.clone();
        drop(chunks);
        self.mark_work();
        if !barriers.is_empty() {
            // SAFETY: an open list and live buffers.
            unsafe { self.list.ResourceBarrier(&barriers) };
        }
        Some((buffer, 0))
    }

    /// Whether the list's hoisted clamps still have to bind their root
    /// signature, pipeline and budget: the first after the reset does.
    pub(crate) fn clamp_unbound(&self) -> bool {
        let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
        !std::mem::replace(&mut chunks.bound, true)
    }

    /// Transitions hoisted clamped argument buffers written so far to
    /// `INDIRECT_ARGUMENT` for the draw list that follows.
    pub(crate) fn clamp_end(&self) {
        let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
        let used = chunks.used;
        let barriers: Vec<_> = chunks.all[..used]
            .iter_mut()
            .filter(|(_, state)| *state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            .map(|(buffer, state)| {
                *state = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
                transition(
                    buffer,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )
            })
            .collect();
        drop(chunks);
        if !barriers.is_empty() {
            // SAFETY: an open list and live buffers.
            unsafe { self.list.ResourceBarrier(&barriers) };
        }
    }

    /// Keeps a hoisted clamped argument buffer alive and schedules its return
    /// from `INDIRECT_ARGUMENT` to `COMMON` when this list closes.
    pub(crate) fn keep_hoisted(&self, buffer: &ID3D12Resource) {
        self.keep(buffer);
        let mut hoisted = self
            .hoisted_buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !hoisted.iter().any(|b| b.as_raw() == buffer.as_raw()) {
            hoisted.push(buffer.clone());
        }
    }

    /// From here on the list gathers no records (`crate::indirect::gather`):
    /// what is about to be recorded may write arbitrary arguments.
    pub(crate) fn gather_close(&self) {
        self.gather
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .gate
            .close_all();
    }

    /// From here on the list gathers no records from host copy `src_id`:
    /// unmodified argument buffers may still gather.
    pub(crate) fn gather_close_buffer(&self, src_id: u64) {
        self.gather
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .gate
            .close_buffer(src_id);
    }

    #[cfg(test)]
    pub(crate) fn gather_allows(&self, src_id: u64) -> bool {
        self.gather
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .gate
            .allows(src_id)
    }

    /// A slot for the record at `offset` of the host copy `src` in the
    /// scratch buffer the list's prelude fills (`crate::indirect::gather`,
    /// which checked the record): the buffer and the slot's byte offset.
    pub(crate) fn gather_slot(
        &self,
        src: &ID3D12Resource,
        src_id: u64,
        offset: u32,
        (kind, stride): (u32, u32),
    ) -> Option<(ID3D12Resource, u64)> {
        let none = |why: usize| {
            crate::indirect::GATHER_NONE[why].fetch_add(1, Ordering::Relaxed);
            None
        };
        let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
        if !gather.gate.allows(src_id) {
            return none(1);
        }
        let found = gather.sets.iter().position(|s| {
            s.src_id == src_id
                && s.kind == kind
                && s.stride == stride
                && s.src.as_raw() == src.as_raw()
        });
        let at = match found {
            Some(at) => at,
            None => {
                if gather.sets.len() >= GATHER_SETS {
                    return none(2);
                }
                // The prelude zeroes the budget with WriteBufferImmediate.
                if self.list.cast::<ID3D12GraphicsCommandList2>().is_err() {
                    return none(4);
                }
                let Some(host) = host() else {
                    return none(4);
                };
                if gather.budget.is_none() {
                    let Some(budget) = host.default_buffer(256) else {
                        return none(4);
                    };
                    name(&budget, || "indirect budget of gathered records".to_owned());
                    gather.budget = Some(budget);
                }
                let scratch = match gather.spare.pop() {
                    Some(scratch) => scratch,
                    None => {
                        let Some(scratch) = host.default_buffer(
                            u64::from(crate::indirect::GATHER_MOST)
                                * u64::from(crate::indirect::GATHER_STRIDE),
                        ) else {
                            return none(4);
                        };
                        name(&scratch, || "indirect arguments, gathered".to_owned());
                        scratch
                    }
                };
                let Some(table) = self.scratch(u64::from(crate::indirect::GATHER_MOST) * 4) else {
                    gather.spare.push(scratch);
                    return none(4);
                };
                gather.sets.push(crate::indirect::GatherSet::new(
                    src.clone(),
                    src_id,
                    (kind, stride),
                    scratch,
                    table,
                ));
                gather.sets.len() - 1
            }
        };
        let set = &mut gather.sets[at];
        let Some(slot) = set.slot(offset) else {
            return none(3);
        };
        let dst_stride = if kind & crate::indirect::ROWS != 0 {
            crate::indirect::ROW_STRIDE
        } else {
            stride
        };
        let out = (set.scratch.clone(), u64::from(slot) * u64::from(dst_stride));
        drop(gather);
        self.mark_work();
        Some(out)
    }

    /// The GPU-writable buffer for mesh programs run without mesh shaders,
    /// of at least `len` bytes where [`MESH_BUFFER_MOST`] allows (and of
    /// [`MESH_BUFFER_MIN`]): the buffer, its host GPU address and its size.
    /// The host shares one between its lists, and this list keeps it alive.
    /// Its state is tracked as a shadow buffer's is ([`HostList::use_shadow`]):
    /// common when a list starts and when it closes.
    pub(crate) fn mesh_buffer(&self, len: u64) -> Option<(ID3D12Resource, u64, u64)> {
        let want = len
            .clamp(MESH_BUFFER_MIN, MESH_BUFFER_MOST)
            .next_multiple_of(1 << 16);
        let host = host()?;
        let (buffer, have) = {
            let mut s = host.mesh_scratch.lock().unwrap_or_else(|e| e.into_inner());
            if s.buffer.as_ref().is_none_or(|(_, have)| *have < want) {
                let buffer = host.default_buffer(want)?;
                name(&buffer, || format!("mesh buffer {want:#x}"));
                // Lists that recorded the smaller one keep it alive.
                s.buffer = Some((buffer, want));
            }
            s.buffer.clone()?
        };
        self.keep(&buffer);
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        Some((buffer, gpu, have))
    }

    /// The list's GPU-writable scratch buffer for expanded indirect
    /// rectangle vertices (`capacity` bytes), created on first use.
    pub(crate) fn rect_vertices(&self, capacity: u64) -> Option<ID3D12Resource> {
        let mut rect = self.rect_scratch.lock().unwrap_or_else(|e| e.into_inner());
        if rect.vertices.is_none() {
            let buffer = host()?.default_buffer(capacity)?;
            name(&buffer, || format!("rect scratch {capacity:#x}"));
            rect.vertices = Some(buffer);
        }
        let buffer = rect.vertices.clone()?;
        drop(rect);
        self.mark_work();
        Some(buffer)
    }

    /// A GPU-writable buffer of at least `bytes` for rewritten indirect
    /// draw arguments, kept alive until the list is reset.
    pub(crate) fn rect_args(&self, bytes: u64) -> Option<ID3D12Resource> {
        let want = bytes.max(RECT_ARGS_MIN).next_multiple_of(256);
        let mut rect = self.rect_scratch.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buf, len)) = &rect.args
            && *len >= want
        {
            let buf = buf.clone();
            drop(rect);
            self.mark_work();
            return Some(buf);
        }
        let buffer = host()?.default_buffer(want)?;
        name(&buffer, || format!("rect args {want:#x}"));
        let old = rect.args.replace((buffer.clone(), want));
        drop(rect);
        self.mark_work();
        if let Some((old, _)) = old {
            self.keep(&old);
        }
        Some(buffer)
    }

    /// The host GPU address of a 4-byte atomic counter for one indirect
    /// rectangle draw, zeroed at each submission before the list runs.
    pub(crate) fn rect_counter(&self) -> Option<u64> {
        let mut rect = self.rect_scratch.lock().unwrap_or_else(|e| e.into_inner());
        if rect.counter_used == RECT_COUNTER_CHUNK {
            rect.counter_chunk += 1;
            rect.counter_used = 0;
        }
        if rect.counter_chunk == rect.counters.len() {
            let bytes = u64::from(RECT_COUNTER_CHUNK) * 4;
            let (buffer, data) = host()?.shared_buffer(bytes)?;
            name(&buffer, || "rect counters".into());
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            rect.counters.push((buffer, data as usize, gpu));
            rect.counter_used = 0;
        }
        let (_, data, gpu) = rect.counters[rect.counter_chunk];
        let slot = rect.counter_used;
        rect.counter_used += 1;
        drop(rect);
        self.mark_work();
        if slot == 0 {
            let bytes = RECT_COUNTER_CHUNK as usize * 4;
            // SAFETY: the list's own mapped chunk, zeroed before it runs.
            self.defer_host(move || unsafe {
                std::ptr::write_bytes(data as *mut u8, 0, bytes);
            });
        }
        Some(gpu + u64::from(slot) * 4)
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.mark_work();
            self.deferred().kept.push(object);
        }
    }

    /// Keeps `object` alive until the list is reset without pinning a
    /// retirement epoch (used when mirroring fixed-function target bindings
    /// onto a newly split trailing list before any GPU work is recorded).
    pub(crate) fn keep_unmarked(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    #[cfg(test)]
    pub(crate) fn kept_contains(&self, object: &impl Interface) -> bool {
        object.cast::<windows::core::IUnknown>().is_ok_and(|u| {
            let raw = u.as_raw();
            self.deferred().kept.iter().any(|k| k.as_raw() == raw)
        })
    }

    #[cfg(test)]
    fn defer_unmarked(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Run(Box::new(work)));
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a memory copy to bring up to date, keyed away from any
    /// texture's).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key, 0).is_none()
    }

    /// Whether this list already covers the requested mip and all smaller ones.
    pub(crate) fn covers_texture(&self, key: usize, first: u32) -> bool {
        self.deferred()
            .textures
            .get(&key)
            .is_some_and(|&seen| seen <= first)
    }

    /// Records coverage and reports whether the list gained any mip levels.
    pub(crate) fn use_texture(&self, key: usize, first: u32) -> bool {
        use std::collections::hash_map::Entry;
        match self.deferred().textures.entry(key) {
            Entry::Occupied(mut e) if first < *e.get() => {
                e.insert(first);
                true
            }
            Entry::Occupied(_) => false,
            Entry::Vacant(e) => {
                e.insert(first);
                true
            }
        }
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        self.mark_work();
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Run(Box::new(work)));
        deferred.refreshed.clear();
        deferred.filled.clear();
        deferred.gpu_written.clear();
        deferred.copy_gen += 1;
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.mark_work();
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// As [`HostList::defer_host`], for work that reads title memory `[src, src + len)`
    /// in place at submission (holding a byte-range mapping guard over that range).
    pub(crate) fn defer_host_read(
        &self,
        src: *const u8,
        len: usize,
        work: impl Fn() + Send + 'static,
    ) {
        self.mark_work();
        self.deferred()
            .copies
            .push(Step::HostRead(src as usize, len, Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset and carrying the
    /// mapped-byte baseline observed at recording time (`expected`).
    pub(crate) fn defer_texture(
        &self,
        staging: std::sync::Arc<crate::texture::Staging>,
        expected: u64,
    ) {
        self.mark_work();
        self.deferred()
            .copies
            .push(Step::Texture(staging, expected));
    }

    /// Raises the recorded mapped-byte baseline on any already-queued texture
    /// upload for `resource` covering `first` on this list.
    pub(crate) fn raise_texture_expected(&self, resource: usize, first: u32, expected: u64) {
        if expected == 0 {
            return;
        }
        let mut deferred = self.deferred();
        for step in deferred.copies.iter_mut().rev() {
            if let Step::Texture(staging, prev) = step
                && staging.resource() == resource
                && staging.first() <= first
            {
                if expected > *prev {
                    *prev = expected;
                }
                break;
            }
        }
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    ///
    /// Releases `self.deferred` before executing steps (restoring `copies` via
    /// a drop guard so closed lists remain resubmittable) so `batch.flush()` ->
    /// `Shadow::update_busy()` -> `settle_card_busy()` (`LINE` / `LINE_LANDED`)
    /// never holds `list.deferred()` across a relay wait (`Host::relay_land`
    /// acquires `list.deferred()` in `read_backs`, and `Host::submit` acquires
    /// `LINE -> list.deferred()` in `WHY_READ`).
    fn run_deferred(&self, batch: &mut Batch, host: &Host, submitted: &mut u64) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        struct RestoreCopies<'a> {
            list: &'a HostList,
            copies: Vec<Step>,
        }
        impl Drop for RestoreCopies<'_> {
            fn drop(&mut self) {
                let mut deferred = self.list.deferred();
                if deferred.copies.is_empty() {
                    deferred.copies = std::mem::take(&mut self.copies);
                }
            }
        }
        let guard = RestoreCopies {
            list: self,
            copies: std::mem::take(&mut self.deferred().copies),
        };
        for step in &guard.copies {
            let mut started = std::time::Instant::now();
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.title_written();
                    // The flush counted its own time; the copy starts here.
                    started = std::time::Instant::now();
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    crate::watch::wrote(dst, len);
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    if prep::wide_fills() {
                        batch.fills.push((dst, src, len));
                        continue;
                    }
                    if !crate::shadow::copy_mapped_wide_checked(
                        dst as *mut u8,
                        src as *const u8,
                        len,
                    ) {
                        batch.fill_failed = true;
                    }
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                        // `Step::Run` may write arbitrary title memory without
                        // calling `watch::wrote`, so void any standing asking
                        // (including `VICTORY_ASK_SHARED=1`) so subsequent
                        // refreshes and submissions query the kernel again.
                        crate::watch::ask_once_void();
                        #[cfg(feature = "diagnostics")]
                        {
                            started = std::time::Instant::now();
                        }
                    }
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::HostRead(src, len, work) => {
                    let _mapped = crate::watch::title_memory_reading(*src, *len);
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    batch.gpu_write(host, submitted, shadow, *address, *len);
                    continue;
                }
                Step::Texture(staging, expected) => {
                    if batch.texture_due(staging, *expected) {
                        staging.refresh(*expected);
                    }
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    declared_writes: std::collections::HashSet<(usize, usize, usize)>,
    /// Staging buffers (by address) refreshed since the last title write,
    /// mapped to the highest mapped-byte baseline refreshed so far.
    textures: std::collections::HashMap<usize, u64>,
    /// Deferred `Step::Fill` copies batched for parallel execution (`host_prep`).
    fills: Vec<(usize, usize, usize)>,
    /// Shadows already refreshed ahead of their first `Step::GpuWrite` (`host_prep`).
    ahead: std::collections::HashMap<usize, (usize, usize)>,
    /// Set if a ReBAR shadow's initial RAM mirror readback failed.
    rebar_init_failed: bool,
    /// Set if a `Step::Fill` copy from title memory failed after retrying lazy materialization.
    fill_failed: bool,
    /// Set if a shadow refresh (`Batch::flush` / `Batch::refresh_ahead`) failed
    /// to materialize or copy a dirty lazy-owned page.
    refresh_failed: bool,
}

impl Batch {
    fn new_gpu_write<T>(&mut self, shadow: &std::sync::Arc<T>, address: usize, len: usize) -> bool {
        self.declared_writes
            .insert((std::sync::Arc::as_ptr(shadow) as usize, address, len))
    }

    fn gpu_write(
        &mut self,
        host: &Host,
        submitted: &mut u64,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        if self.new_gpu_write(shadow, address, len) {
            self.ahead
                .remove(&(std::sync::Arc::as_ptr(shadow) as usize));
            // The first declaration needs the refreshed bytes for its snapshot.
            // Another declaration of this exact range is already pending until
            // the GPU submission finishes, so it needs no refresh boundary.
            self.flush();
            if !shadow.writing(address, len, |buffer, size| {
                host.readback_rebar_to_vec(submitted, buffer, size)
            }) {
                self.rebar_init_failed = true;
            }
        }
    }

    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address + len;
        if let Some(&(a, b)) = self.ahead.get(&(std::sync::Arc::as_ptr(shadow) as usize))
            && a <= address
            && end <= b
        {
            return;
        }
        match self
            .refreshes
            .iter_mut()
            .find(|(s, ..)| std::sync::Arc::ptr_eq(s, shadow))
        {
            Some((_, from, to)) => (*from, *to) = ((*from).min(address), (*to).max(end)),
            None => self.refreshes.push((shadow.clone(), address, end)),
        }
    }

    /// Whether `staging` is still to be refreshed in this submission (or needs
    /// a higher mapped-byte baseline than an earlier list in the same
    /// submission), recording the highest baseline refreshed.
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>, expected: u64) -> bool {
        let key = std::sync::Arc::as_ptr(staging) as *const () as usize;
        match self.textures.entry(key) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(expected);
                true
            }
            std::collections::hash_map::Entry::Occupied(mut e) if expected > *e.get() => {
                e.insert(expected);
                true
            }
            std::collections::hash_map::Entry::Occupied(_) => false,
        }
    }

    /// Before a step that writes title memory: pending fills and refreshes are
    /// done first, and pre-write refreshes and textures are cleared.
    fn title_written(&mut self) {
        self.fill();
        self.flush();
        self.ahead.clear();
        self.textures.clear();
    }

    fn flush(&mut self) {
        if self.refreshes.is_empty() {
            return;
        }
        let started = std::time::Instant::now();
        let items = std::mem::take(&mut self.refreshes);
        let (copied, ok) = crate::shadow::refresh_batch(items, crate::shadow::wide_copy());
        if !ok {
            self.refresh_failed = true;
        }
        let us = started.elapsed().as_micros() as u64;
        PREPARING[0].fetch_add(us, Ordering::Relaxed);
        PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped takes nothing from its
/// first unmapped page on.
///
/// Host-owned staging uploads pass `title_dst = false` (`Step::Fill` /
/// [`HostList::defer_fill`]): those destinations are known mapped, and under
/// Wine+vkd3d `VirtualQuery` can report them as `MEM_FREE`, so the commitment
/// guard must not apply (#518). Title buffer destinations keep `title_dst =
/// true` (`Step::Copy` / [`HostList::defer_copy`]).
///
/// For host-owned destinations (`title_dst = false`), title reads default to
/// in-process vectorized copies ([`victory_copy_try`]) with VEH fault recovery
/// into [`read_sparse`]/[`read_pieces`]; setting `VICTORY_FAST_COPY=0` restores
/// `ReadProcessMemory`. Title destinations always run through the kernel copy
/// so unmapped destination pages fail cleanly instead of faulting.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    if !copy_mapped_checked(dst, src, len, title_dst) {
        fail(format_args!(
            "copy_mapped failed after retrying lazy materialization (src={:#x}, dst={:#x}, len={len}, title_dst={title_dst})",
            src as usize, dst as usize
        ));
    }
}

/// [`copy_mapped`] that retries lazy materialization if a concurrent relay
/// re-defers the source (or destination for `title_dst`), returning `false`
/// instead of silently proceeding with unwritten destination bytes.
pub(crate) fn copy_mapped_checked(
    dst: *mut u8,
    src: *const u8,
    len: usize,
    title_dst: bool,
) -> bool {
    if len == 0 {
        return true;
    }
    const MAX_ATTEMPTS: usize = 8;
    for attempt in 0..MAX_ATTEMPTS {
        if crate::lazy_back::active() {
            let ok = if title_dst {
                // Title-to-title copies (`Step::Copy`) may read a deferred source
                // into a committed or deferred destination: materialize both ranges.
                crate::lazy_back::materialize_ranges(&[(src as usize, len), (dst as usize, len)])
            } else {
                // Host fills materialize only their title source.
                crate::lazy_back::materialize_range(src as usize, len)
            };
            if !ok {
                return false;
            }
        }
        if copy_mapped_raw_checked(dst, src, len, title_dst) {
            return true;
        }
        if attempt + 1 < MAX_ATTEMPTS {
            std::thread::yield_now();
        }
    }
    false
}

/// [`copy_mapped`] without triggering [`crate::lazy_back::materialize_range`],
/// returning `false` instead of zero-filling if a `PAGE_NOACCESS` source page
/// belongs to an active [`crate::lazy_back`] block. Callers holding a `Shadow`
/// lock MUST materialize `[src, src + len)` (or `[dst, dst + len)`) before
/// taking the lock and call this raw variant inside.
pub(crate) fn copy_mapped_raw_checked(
    dst: *mut u8,
    src: *const u8,
    len: usize,
    title_dst: bool,
) -> bool {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return true;
        }
        2 => return true,
        _ => {}
    }
    if len == 0 {
        return true;
    }
    #[cfg(test)]
    if let Some(hook) = BEFORE_COPY_MAPPED_RAW_HOOK.with(|h| h.borrow_mut().take()) {
        hook();
    }
    if !title_dst && fast_fill() {
        #[cfg(feature = "diagnostics")]
        let moved = std::time::Instant::now();
        // SAFETY: the destination is mapped writable host memory of `len` bytes;
        // a fault on the source is caught by `fast_fill`'s VEH and redirected to
        // `fail`, so `read_sparse_checked` / `read_pieces_checked` handles unmapped
        // holes directly without repeating a failing whole-range `ReadProcessMemory`
        // syscall.
        let copied = unsafe { victory_copy_try(dst, src, len) } != 0;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(
            crate::copystat::C::MoveUs,
            moved.elapsed().as_micros() as u64,
        );
        if copied {
            return true;
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Fallbacks, 1);
        return if len >= 64 << 10 {
            read_sparse_checked(src, dst, len)
        } else {
            read_pieces_checked(src, dst, len)
        };
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::MoveUs,
        moved.elapsed().as_micros() as u64,
    );
    if whole {
        return true;
    }
    // The copy stopped at an unmapped page. One of a title destination ends
    // it: the retries below write the destination themselves (zeros for
    // what cannot be read), and would fault there.
    if title_dst && !crate::texture::committed(dst, len) {
        return true;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    if !title_dst && len >= 64 << 10 {
        read_sparse_checked(src, dst, len)
    } else if !title_dst {
        read_pieces_checked(src, dst, len)
    } else {
        read_pieces_title_dst_checked(src, dst, len)
    }
}

// Non-overlapping vectorized copies that may fault on their source: both
// variants are pure Win64 ABI leaf functions (only volatile registers touched,
// stack untouched) so [`fast_fill`]'s vectored exception handler can redirect
// any faulting load inside `[start, end)` straight to the matching `fail` exit.
// Destination-aligned 128 B/iter AVX2 (or 64 B/iter SSE2) stores saturate both
// write-back staging RAM and write-combining ReBAR/upload VRAM far better than
// `rep movsb`.
std::arch::global_asm!(
    ".globl victory_copy_try_avx2",
    ".globl victory_copy_try_avx2_start",
    ".globl victory_copy_try_avx2_end",
    ".globl victory_copy_try_avx2_fail",
    "victory_copy_try_avx2:",
    "victory_copy_try_avx2_start:",
    "cmp r8, 128",
    "jb 2f",
    "vmovdqu ymm0, ymmword ptr [rdx]",
    "vmovdqu ymmword ptr [rcx], ymm0",
    "mov rax, rcx",
    "and rax, 31",
    "mov r9, 32",
    "sub r9, rax",
    "add rcx, r9",
    "add rdx, r9",
    "sub r8, r9",
    "cmp r8, 128",
    "jb 2f",
    ".p2align 4",
    "1:",
    "vmovdqu ymm0, ymmword ptr [rdx]",
    "vmovdqu ymm1, ymmword ptr [rdx + 32]",
    "vmovdqu ymm2, ymmword ptr [rdx + 64]",
    "vmovdqu ymm3, ymmword ptr [rdx + 96]",
    "vmovdqa ymmword ptr [rcx], ymm0",
    "vmovdqa ymmword ptr [rcx + 32], ymm1",
    "vmovdqa ymmword ptr [rcx + 64], ymm2",
    "vmovdqa ymmword ptr [rcx + 96], ymm3",
    "add rdx, 128",
    "add rcx, 128",
    "sub r8, 128",
    "cmp r8, 128",
    "jae 1b",
    "2:",
    "cmp r8, 32",
    "jb 4f",
    "3:",
    "vmovdqu ymm0, ymmword ptr [rdx]",
    "vmovdqu ymmword ptr [rcx], ymm0",
    "add rdx, 32",
    "add rcx, 32",
    "sub r8, 32",
    "cmp r8, 32",
    "jae 3b",
    "4:",
    "cmp r8, 16",
    "jb 5f",
    "vmovdqu xmm0, xmmword ptr [rdx]",
    "vmovdqu xmmword ptr [rcx], xmm0",
    "add rdx, 16",
    "add rcx, 16",
    "sub r8, 16",
    "5:",
    "cmp r8, 8",
    "jb 6f",
    "mov rax, qword ptr [rdx]",
    "mov qword ptr [rcx], rax",
    "add rdx, 8",
    "add rcx, 8",
    "sub r8, 8",
    "6:",
    "cmp r8, 4",
    "jb 7f",
    "mov eax, dword ptr [rdx]",
    "mov dword ptr [rcx], eax",
    "add rdx, 4",
    "add rcx, 4",
    "sub r8, 4",
    "7:",
    "test r8, r8",
    "jz 9f",
    "8:",
    "mov al, byte ptr [rdx]",
    "mov byte ptr [rcx], al",
    "inc rdx",
    "inc rcx",
    "dec r8",
    "jnz 8b",
    "9:",
    "victory_copy_try_avx2_end:",
    "vzeroupper",
    "mov eax, 1",
    "ret",
    "victory_copy_try_avx2_fail:",
    "vzeroupper",
    "xor eax, eax",
    "ret",
    ".globl victory_copy_try_sse2",
    ".globl victory_copy_try_sse2_start",
    ".globl victory_copy_try_sse2_end",
    ".globl victory_copy_try_sse2_fail",
    "victory_copy_try_sse2:",
    "victory_copy_try_sse2_start:",
    "cmp r8, 64",
    "jb 12f",
    "movdqu xmm0, xmmword ptr [rdx]",
    "movdqu xmmword ptr [rcx], xmm0",
    "mov rax, rcx",
    "and rax, 15",
    "mov r9, 16",
    "sub r9, rax",
    "add rcx, r9",
    "add rdx, r9",
    "sub r8, r9",
    "cmp r8, 64",
    "jb 12f",
    ".p2align 4",
    "11:",
    "movdqu xmm0, xmmword ptr [rdx]",
    "movdqu xmm1, xmmword ptr [rdx + 16]",
    "movdqu xmm2, xmmword ptr [rdx + 32]",
    "movdqu xmm3, xmmword ptr [rdx + 48]",
    "movdqa xmmword ptr [rcx], xmm0",
    "movdqa xmmword ptr [rcx + 16], xmm1",
    "movdqa xmmword ptr [rcx + 32], xmm2",
    "movdqa xmmword ptr [rcx + 48], xmm3",
    "add rdx, 64",
    "add rcx, 64",
    "sub r8, 64",
    "cmp r8, 64",
    "jae 11b",
    "12:",
    "cmp r8, 16",
    "jb 14f",
    "13:",
    "movdqu xmm0, xmmword ptr [rdx]",
    "movdqu xmmword ptr [rcx], xmm0",
    "add rdx, 16",
    "add rcx, 16",
    "sub r8, 16",
    "cmp r8, 16",
    "jae 13b",
    "14:",
    "cmp r8, 8",
    "jb 15f",
    "mov rax, qword ptr [rdx]",
    "mov qword ptr [rcx], rax",
    "add rdx, 8",
    "add rcx, 8",
    "sub r8, 8",
    "15:",
    "cmp r8, 4",
    "jb 16f",
    "mov eax, dword ptr [rdx]",
    "mov dword ptr [rcx], eax",
    "add rdx, 4",
    "add rcx, 4",
    "sub r8, 4",
    "16:",
    "test r8, r8",
    "jz 18f",
    "17:",
    "mov al, byte ptr [rdx]",
    "mov byte ptr [rcx], al",
    "inc rdx",
    "inc rcx",
    "dec r8",
    "jnz 17b",
    "18:",
    "victory_copy_try_sse2_end:",
    "mov eax, 1",
    "ret",
    "victory_copy_try_sse2_fail:",
    "xor eax, eax",
    "ret",
);

unsafe extern "C" {
    fn victory_copy_try_avx2(dst: *mut u8, src: *const u8, len: usize) -> u32;
    static victory_copy_try_avx2_start: u8;
    static victory_copy_try_avx2_end: u8;
    static victory_copy_try_avx2_fail: u8;

    fn victory_copy_try_sse2(dst: *mut u8, src: *const u8, len: usize) -> u32;
    static victory_copy_try_sse2_start: u8;
    static victory_copy_try_sse2_end: u8;
    static victory_copy_try_sse2_fail: u8;
}

/// Copies `len` non-overlapping bytes from `src` to `dst`; returns `0` if a
/// source page faulted (some bytes may have been written). Only valid with
/// [`fast_fill`]'s vectored handler installed.
#[inline]
unsafe fn victory_copy_try(dst: *mut u8, src: *const u8, len: usize) -> u32 {
    static USE_AVX2: OnceLock<bool> = OnceLock::new();
    if *USE_AVX2.get_or_init(|| std::is_x86_feature_detected!("avx2")) {
        // SAFETY: AVX2 availability verified once above.
        unsafe { victory_copy_try_avx2(dst, src, len) }
    } else {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { victory_copy_try_sse2(dst, src, len) }
    }
}

/// Whether title-to-host reads use vectorized in-process copies with VEH
/// fallback (`VICTORY_FAST_COPY=0` restores legacy `ReadProcessMemory`; on by
/// default).
fn fast_fill() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_FAST_COPY").is_none_or(|v| v != "0");
        if on {
            install_fast_copy_handler();
        }
        on
    })
}

fn fast_copy_fail_rip(rip: u64) -> Option<u64> {
    let (avx_s, avx_e) = (
        &raw const victory_copy_try_avx2_start as u64,
        &raw const victory_copy_try_avx2_end as u64,
    );
    if rip >= avx_s && rip < avx_e {
        return Some(&raw const victory_copy_try_avx2_fail as u64);
    }
    let (sse_s, sse_e) = (
        &raw const victory_copy_try_sse2_start as u64,
        &raw const victory_copy_try_sse2_end as u64,
    );
    if rip >= sse_s && rip < sse_e {
        return Some(&raw const victory_copy_try_sse2_fail as u64);
    }
    None
}

/// Whether `rip` lies inside `victory_copy_try_{avx2,sse2}`'s faulting copy
/// region so `watch::on_write` yields to `fast_fill`'s VEH.
pub(crate) fn is_fast_copy_rip(rip: u64) -> bool {
    fast_copy_fail_rip(rip).is_some()
}

fn install_fast_copy_handler() {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, EXCEPTION_GUARD_PAGE, NTSTATUS,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        unsafe {
            let r = &*(*info).ExceptionRecord;
            if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
                && r.ExceptionCode != EXCEPTION_GUARD_PAGE as NTSTATUS
            {
                return EXCEPTION_CONTINUE_SEARCH;
            }
            let context = &mut *(*info).ContextRecord;
            if let Some(fail_rip) = fast_copy_fail_rip(context.Rip) {
                context.Rip = fail_rip;
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        // SAFETY: the handler only inspects RIP within our leaf copy routines
        // and redirects a faulting source load to the matching failure exit.
        unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) };
    });
}

#[cfg(test)]
std::thread_local! {
    static AFTER_DIRECT_SUBMIT_GUARD_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static BEFORE_COPY_MAPPED_RAW_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static AFTER_VIRTUAL_QUERY_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static AFTER_READ_PIECES_FAIL_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_after_direct_submit_guard_hook(hook: Option<Box<dyn FnOnce()>>) {
    AFTER_DIRECT_SUBMIT_GUARD_HOOK.with(|h| *h.borrow_mut() = hook);
}

#[cfg(test)]
pub(crate) fn set_before_copy_mapped_raw_hook(hook: Option<Box<dyn FnOnce()>>) {
    BEFORE_COPY_MAPPED_RAW_HOOK.with(|h| *h.borrow_mut() = hook);
}

fn range_has_lazy_block(at: usize, len: usize) -> bool {
    crate::lazy_back::has_block_in_range(at, len)
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies: the title may remap the same address.
/// Returns `false` instead of zero-filling if any `PAGE_NOACCESS` span
/// belongs to an active [`crate::lazy_back`] block.
fn read_sparse_checked(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let fast = fast_fill();
    let mut offset = 0;
    let mut stale_retries = 0usize;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        // Sample `remove_generation()` strictly BEFORE `VirtualQuery` so if a
        // lazy block finishes materializing between `VirtualQuery` (seeing
        // `PAGE_NOACCESS`) and `range_has_lazy_block` (seeing it removed), the
        // generation mismatch prevents zero-filling and forces a re-query.
        let remove_gen = crate::lazy_back::remove_generation();
        // SAFETY: zero is valid for the fields of this Windows output structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: the query inspects an address and fills the local structure.
        let size = unsafe { VirtualQuery(at.cast(), &mut info, size_of_val(&info)) };
        #[cfg(test)]
        if let Some(hook) = AFTER_VIRTUAL_QUERY_HOOK.with(|h| h.borrow_mut().take()) {
            hook();
        }
        let n = (info.BaseAddress as usize)
            .saturating_add(info.RegionSize)
            .saturating_sub(at as usize)
            .min(remaining);
        if size != size_of_val(&info) || n == 0 {
            if !read_pieces_checked(at, to, remaining) {
                return false;
            }
            break;
        }
        if info.State != MEM_COMMIT || (info.Protect & PAGE_NOACCESS) != 0 {
            if (info.Protect & PAGE_NOACCESS) != 0 {
                if range_has_lazy_block(at as usize, n) {
                    return false;
                }
                if crate::lazy_back::remove_generation() != remove_gen {
                    stale_retries += 1;
                    if stale_retries > 8 {
                        return false;
                    }
                    continue;
                }
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if fast
            && (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0
            // SAFETY: destination is writable host memory; any concurrent
            // unmap on `at` faults into `on_fault` and falls through below.
            && unsafe { victory_copy_try(to, at, n) } != 0
        {
            // Fast userspace copy succeeded for this committed sub-region.
        } else if !read(at, to, n) {
            // A mapping may change between query and read.
            if !read_pieces_checked(at, to, n) {
                return false;
            }
        }
        stale_retries = 0;
        offset += n;
    }
    true
}

fn read_pieces_checked(src: *const u8, dst: *mut u8, len: usize) -> bool {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read (unless a `lazy_back` block
    // was concurrently armed or just materialized on the source).
    let mut ok = true;
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        if !ok {
            return;
        }
        let to = dst.wrapping_add(at - src as usize);
        let gen64 = crate::lazy_back::remove_generation();
        if !read(at as *const u8, to, n) {
            #[cfg(test)]
            if let Some(hook) = AFTER_READ_PIECES_FAIL_HOOK.with(|h| h.borrow_mut().take()) {
                hook();
            }
            if range_has_lazy_block(at, n) {
                ok = false;
                return;
            }
            if crate::lazy_back::remove_generation() != gen64 {
                if read(at as *const u8, to, n) {
                    return;
                }
                if range_has_lazy_block(at, n) {
                    ok = false;
                    return;
                }
            }
            for_pieces(at, n, 4 << 10, |at, n| {
                if !ok {
                    return;
                }
                let to = dst.wrapping_add(at - src as usize);
                let gen4 = crate::lazy_back::remove_generation();
                if !read(at as *const u8, to, n) {
                    if range_has_lazy_block(at, n) {
                        ok = false;
                        return;
                    }
                    if crate::lazy_back::remove_generation() != gen4 {
                        if read(at as *const u8, to, n) {
                            return;
                        }
                        ok = false;
                        return;
                    }
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
    ok
}

fn read_pieces_title_dst_checked(src: *const u8, dst: *mut u8, len: usize) -> bool {
    // Title destination fallback when a page is unmapped: copy 64 KiB / 4 KiB
    // pieces, zeroing unmapped source pieces unless the source piece belongs to
    // a deferred or concurrently materialized `lazy_back` block.
    read_pieces_checked(src, dst, len)
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    if !crate::lazy_back::materialize_range(src as usize, dst.len()) {
        return false;
    }
    read_into_raw(src, dst)
}

/// [`read_into`] without calling [`crate::lazy_back::materialize_range`].
/// Safe to call while holding `Shadow` locks after materializing outside.
pub(crate) fn read_into_raw(src: *const u8, dst: &mut [u8]) -> bool {
    if dst.is_empty() {
        return true;
    }
    if fast_fill() {
        // SAFETY: `dst` is valid writable memory; faults on `src` redirect to
        // `fail` via `fast_fill`'s VEH without invoking `lazy_back::handle_fault`.
        if unsafe { victory_copy_try(dst.as_mut_ptr(), src, dst.len()) } != 0 {
            return true;
        }
    }
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Probes readability via `ReadProcessMemory` without raising a VEH fault on
/// unmapped pages (used by `texture` residency probes after `K32QueryWorkingSetEx`).
pub(crate) fn read_into_probe(src: *const u8, dst: &mut [u8]) -> bool {
    if dst.is_empty() {
        return true;
    }
    if !crate::lazy_back::materialize_range(src as usize, dst.len()) {
        return false;
    }
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Reads title memory at `src` into `dst`, materializing deferred lazy blocks
/// and using [`victory_copy_try`] when `VICTORY_FAST_COPY` is active.
#[cfg(test)]
pub(crate) fn read_into_fast(src: *const u8, dst: &mut [u8]) -> bool {
    if dst.is_empty() {
        return true;
    }
    if !crate::lazy_back::materialize_range(src as usize, dst.len()) {
        return false;
    }
    if fast_fill() {
        // SAFETY: `dst` is mapped writable memory of `dst.len()` bytes; a
        // fault on `src` is caught by `fast_fill`'s vectored handler and
        // returns 0.
        if unsafe { victory_copy_try(dst.as_mut_ptr(), src, dst.len()) } != 0 {
            return true;
        }
        if !crate::lazy_back::materialize_range(src as usize, dst.len()) {
            return false;
        }
    }
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` in the kernel: false if a source
/// page is not readable or a destination page not writable (the bytes
/// before that page may have been copied).
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is `len` bytes the caller may write, host
    // memory or title memory; the kernel checks both ranges, and a page it
    // cannot read or write fails the call instead of faulting.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

/// Copies host-owned bytes into title memory without dereferencing a
/// potentially unmapped destination on a worker thread.
pub fn write_title_memory(destination: usize, bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }
    if !crate::watch::kernel_will_write(destination, bytes.len()) {
        crate::watch::kernel_wrote(destination, bytes.len());
        return false;
    }
    let written = read(bytes.as_ptr(), destination as *mut u8, bytes.len());
    crate::watch::kernel_wrote(destination, bytes.len());
    written
}

#[derive(Clone, Copy)]
enum CopyBack {
    WineRead,
    NativeWrite(NtWrite),
}

/// Select the checked same-process copy once. Proton's read path is much
/// cheaper for the small rows and pages written back after a submission.
fn copy_back_method() -> Option<CopyBack> {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    static METHOD: std::sync::OnceLock<Option<CopyBack>> = std::sync::OnceLock::new();
    *METHOD.get_or_init(|| {
        // SAFETY: ntdll is always loaded. Wine identifies itself by this export;
        // on native Windows the write routine has the NtWrite signature.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            if GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some() {
                return Some(CopyBack::WineRead);
            }
            let f = GetProcAddress(ntdll, c"NtWriteVirtualMemory".as_ptr().cast())?;
            Some(CopyBack::NativeWrite(std::mem::transmute::<
                unsafe extern "system" fn() -> isize,
                NtWrite,
            >(f)))
        }
    })
}

type NtWrite = unsafe extern "system" fn(
    process: isize,
    base: *mut core::ffi::c_void,
    buffer: *const core::ffi::c_void,
    len: usize,
    written: *mut usize,
) -> i32;

#[cfg(test)]
std::thread_local! {
    static BEFORE_WRITE_MAPPED_RAW_HOOK: std::cell::RefCell<Option<Box<dyn FnMut()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_before_write_mapped_raw_hook(hook: Option<Box<dyn FnMut()>>) {
    BEFORE_WRITE_MAPPED_RAW_HOOK.with(|h| *h.borrow_mut() = hook);
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
///
/// Wine uses the checked read into the destination. Native Windows uses
/// the NT write directly, avoiding the region query and temporary changes
/// to read-only page protections that WriteProcessMemory can perform.
///
/// # Lazy copy-back contract
/// - Materializes any deferred `lazy_back` blocks overlapping `[dst, dst + len)`
///   before writing. If materialization fails, returns `None` without calling
///   `write_mapped_raw`.
/// - If `write_mapped_raw` returns a short count (`written < len`) while a
///   concurrent lazy arm landed in `[dst, dst + len)` (observed via
///   `lazy_back::has_block_in_range` or a `lazy_back::remove_generation()`
///   change across the write), retries materialization and the raw write up to
///   `MAX_LAZY_RETRIES` times.
/// - If lazy-backed bytes still prevent a complete write after retries are
///   exhausted, returns `None` so callers such as [`read_backs`] fail closed
///   instead of reporting a short count as complete.
/// - Short writes caused solely by ordinary unmapped or read-only pages (no
///   overlapping lazy block or removal generation change) still return
///   `Some(written)` with the exact count of bytes written.
#[must_use]
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) -> Option<usize> {
    const MAX_LAZY_RETRIES: usize = 8;
    if len == 0 {
        return Some(0);
    }
    let addr = dst as usize;
    for _ in 0..=MAX_LAZY_RETRIES {
        if !crate::lazy_back::materialize_range(addr, len) {
            return None;
        }
        let gen_before = crate::lazy_back::remove_generation();
        #[cfg(test)]
        BEFORE_WRITE_MAPPED_RAW_HOOK.with(|h| {
            if let Some(cb) = h.borrow_mut().as_mut() {
                cb();
            }
        });
        let written = write_mapped_raw(dst, src, len);
        if written == len {
            return Some(written);
        }
        let lazy_interfered = crate::lazy_back::has_block_in_range(addr, len)
            || crate::lazy_back::remove_generation() != gen_before;
        if !lazy_interfered {
            return Some(written);
        }
    }
    None
}

/// Core piece-splitting implementation for [`write_mapped_raw_with`].
///
/// `write(at, n)` returns `(succeeded, bytes_transferred)`. Whenever a failed
/// call reports `bytes_transferred > 0`, that prefix is immediately credited to
/// `on_piece` and `written`, and fallback continues only on the remaining tail.
fn write_mapped_raw_impl(
    base: usize,
    len: usize,
    mut write: impl FnMut(usize, usize) -> (bool, usize),
    mut on_piece: impl FnMut(usize, usize),
) -> usize {
    if len == 0 {
        return 0;
    }
    let (ok, done) = write(base, len);
    if ok {
        on_piece(0, len);
        return len;
    }
    let mut written = 0usize;
    let mut rem_base = base;
    let mut rem_len = len;
    let prefix = done.min(rem_len);
    if prefix > 0 {
        on_piece(0, prefix);
        written += prefix;
        rem_base += prefix;
        rem_len -= prefix;
    }
    if rem_len == 0 {
        return written;
    }
    for_pieces(rem_base, rem_len, 64 << 10, |at, n| {
        let (ok, done) = write(at, n);
        if ok {
            on_piece(at - base, n);
            written += n;
            return;
        }
        let mut tail_at = at;
        let mut tail_n = n;
        let partial = done.min(tail_n);
        if partial > 0 {
            on_piece(tail_at - base, partial);
            written += partial;
            tail_at += partial;
            tail_n -= partial;
        }
        if tail_n > 0 {
            for_pieces(tail_at, tail_n, 4 << 10, |sub_at, sub_n| {
                let (sub_ok, sub_done) = write(sub_at, sub_n);
                let credited = if sub_ok { sub_n } else { sub_done.min(sub_n) };
                if credited > 0 {
                    on_piece(sub_at - base, credited);
                    written += credited;
                }
            });
        }
    });
    written
}

/// [`write_mapped`] without calling [`crate::lazy_back::materialize_range`],
/// invoking `on_piece(offset_from_dst, piece_len)` for every confirmed-written
/// sub-piece. Safe to call while holding `Shadow` locks after materializing
/// outside.
pub(crate) fn write_mapped_raw_with(
    dst: *mut u8,
    src: *const u8,
    len: usize,
    on_piece: impl FnMut(usize, usize),
) -> usize {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if len == 0 {
        return 0;
    }
    #[cfg(feature = "diagnostics")]
    // SAFETY: the caller's source holds `len` bytes.
    crate::shadow::high_ones(dst as usize, unsafe {
        std::slice::from_raw_parts(src, len)
    });
    let Some(method) = copy_back_method() else {
        return 0;
    };
    let base = dst as usize;
    write_mapped_raw_impl(
        base,
        len,
        |at, n| {
            let from = src.wrapping_add(at - base);
            let mut done = 0usize;
            match method {
                CopyBack::NativeWrite(call) => {
                    // SAFETY: the source is our mapped host copy; the kernel
                    // checks the destination and writes the exact transferred
                    // prefix count into `done` even on `STATUS_PARTIAL_COPY`.
                    let ok = unsafe {
                        call(
                            GetCurrentProcess() as isize,
                            at as _,
                            from.cast(),
                            n,
                            &mut done,
                        ) == 0
                    };
                    (ok, if ok { n } else { done.min(n) })
                }
                CopyBack::WineRead => {
                    // SAFETY: same-process checked copy; `ReadProcessMemory`
                    // reports any transferred prefix in `done` on partial copy.
                    let ok = unsafe {
                        ReadProcessMemory(GetCurrentProcess(), from.cast(), at as _, n, &mut done)
                            != 0
                    };
                    (ok, if ok { n } else { done.min(n) })
                }
            }
        },
        on_piece,
    )
}

/// [`write_mapped`] without calling [`crate::lazy_back::materialize_range`].
/// Safe to call while holding `Shadow` locks after materializing outside.
/// Returns the exact count of bytes successfully written into mapped pages.
pub(crate) fn write_mapped_raw(dst: *mut u8, src: *const u8, len: usize) -> usize {
    write_mapped_raw_with(dst, src, len, |_, _| {})
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
pub(crate) fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

#[cfg(test)]
mod title_copies {
    use super::{copy_mapped, write_mapped, write_title_memory};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    const BLOCK: usize = 64 << 10;

    /// Three reserved blocks, the first and the last committed and filled
    /// with `fill`: title memory with an unmapped block in it.
    fn holed(fill: u8) -> *mut u8 {
        // SAFETY: a fresh reservation; the committed blocks are ours to fill.
        unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            for block in [0, 2] {
                let at = base.add(block * BLOCK);
                assert!(!VirtualAlloc(at.cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.write_bytes(fill, BLOCK);
            }
            base
        }
    }

    fn block<'a>(base: *const u8, n: usize) -> &'a [u8] {
        // SAFETY: tests ask for committed blocks only.
        unsafe { std::slice::from_raw_parts(base.add(n * BLOCK), BLOCK) }
    }

    fn release(base: *mut u8) {
        // SAFETY: a reservation of `holed`.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }

    fn pattern() -> Vec<u8> {
        (0..3 * BLOCK).map(|i| (i % 251) as u8 + 1).collect()
    }

    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(
                    VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE),
                    at.cast()
                );
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(
                        VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old),
                        0
                    );
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(
                        VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE),
                        at.cast()
                    );
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn title_copies_reach_mapped_destinations_and_stop_at_unmapped_pages() {
        let from = pattern();
        // All mapped: copied, through a destination that was never asked.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), BLOCK, true);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        release(to);
        // An unmapped block in the destination: nothing faults, and nothing
        // arrives from that block on.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), 3 * BLOCK, true);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
        // An unmapped block in the source reads as zeros.
        let holes = holed(7);
        let mut to = vec![0xeeu8; 3 * BLOCK];
        copy_mapped(to.as_mut_ptr(), holes, 3 * BLOCK, true);
        assert!(to[..BLOCK].iter().all(|&b| b == 7));
        assert!(to[BLOCK..2 * BLOCK].iter().all(|&b| b == 0));
        assert!(to[2 * BLOCK..].iter().all(|&b| b == 7));
        release(holes);
    }

    /// Host-owned staging (`title_dst = false`, `defer_fill`) still copies;
    /// title destinations (`title_dst = true`, `defer_copy`) keep the
    /// commitment guard. Wine+vkd3d can report Map memory as MEM_FREE, which
    /// is why buffer->texture staging must not use the title path (#518).
    #[test]
    fn host_staging_fill_copies_while_title_dst_keeps_the_commitment_guard() {
        let from = pattern();
        // Fill path: a heap buffer stands in for mapped upload memory.
        let mut staging = vec![0u8; BLOCK];
        copy_mapped(staging.as_mut_ptr(), from.as_ptr(), BLOCK, false);
        assert_eq!(&staging[..], &from[..BLOCK]);
        // Copy path: an unmapped hole in the title destination stops the
        // write; the trailing committed block stays untouched.
        let title = holed(0xee);
        copy_mapped(title, from.as_ptr(), 3 * BLOCK, true);
        assert!(block(title, 2).iter().all(|&b| b == 0xee));
        release(title);
    }

    // Run on both revisions with --ignored --nocapture to measure the active path.
    #[test]
    #[ignore = "local copy-back timing"]
    fn time_small_copy_backs() {
        const ROW: usize = 4096;
        let source = vec![0xf3; ROW * 1024];
        let mut dest = vec![0; source.len()];
        let mut samples = Vec::new();
        for round in 0..10 {
            dest.fill(0);
            let start = std::time::Instant::now();
            for offset in (0..source.len()).step_by(ROW) {
                assert_eq!(
                    write_mapped(dest[offset..].as_mut_ptr(), source[offset..].as_ptr(), ROW),
                    Some(ROW)
                );
            }
            let elapsed = start.elapsed();
            assert_eq!(dest, source);
            if round >= 2 {
                samples.push(elapsed);
            }
        }
        samples.sort_unstable();
        eprintln!(
            "1024 x 4 KiB copy-backs, median {:?}",
            (samples[3] + samples[4]) / 2
        );
    }

    #[test]
    fn write_mapped_raw_credits_partial_prefix_before_inaccessible_page() {
        const OS_PAGE: usize = 4096;
        // 1. Live OS test: `[RW page 0, PAGE_NOACCESS page 1, RW page 2]` starting
        //    mid-page so the initial NT write transfers a partial writable prefix
        //    before failing with `STATUS_PARTIAL_COPY`.
        // SAFETY: fresh private 3-page region; page 1 is set to PAGE_NOACCESS.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                3 * OS_PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
            .cast::<u8>()
        };
        assert!(!base.is_null());
        // SAFETY: seeding committed pages and protecting middle page NOACCESS.
        unsafe {
            base.write_bytes(0xEE, 3 * OS_PAGE);
            let mut old = 0u32;
            assert_ne!(
                VirtualProtect(base.add(OS_PAGE).cast(), OS_PAGE, PAGE_NOACCESS, &mut old),
                0
            );
        }
        let start_off = 1234usize;
        let total_len = 2 * OS_PAGE; // crosses page 0 tail -> page 1 NOACCESS -> page 2 head
        let src: Vec<u8> = (0..total_len).map(|i| ((i % 251) + 1) as u8).collect();
        let mut pieces = Vec::new();
        // SAFETY: `base.add(start_off)` lies inside our 3-page reservation.
        let written = super::write_mapped_raw_with(
            unsafe { base.add(start_off) },
            src.as_ptr(),
            total_len,
            |off, n| pieces.push((off, n)),
        );
        let first_expected = OS_PAGE - start_off;
        let second_expected = total_len - first_expected - OS_PAGE;
        assert_eq!(written, first_expected + second_expected);
        assert_eq!(
            pieces,
            vec![
                (0, first_expected),
                (first_expected + OS_PAGE, second_expected)
            ],
            "on_piece must credit the writable prefix and trailing writable page without double-counting"
        );
        // SAFETY: page 0 and page 2 remain PAGE_READWRITE.
        unsafe {
            assert_eq!(
                std::slice::from_raw_parts(base, start_off),
                &vec![0xEE; start_off][..]
            );
            assert_eq!(
                std::slice::from_raw_parts(base.add(start_off), first_expected),
                &src[..first_expected]
            );
            assert_eq!(
                std::slice::from_raw_parts(base.add(2 * OS_PAGE), second_expected),
                &src[first_expected + OS_PAGE..]
            );
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }

        // 2. Simulated partial-copy failure at both 64 KiB and 4 KiB granularities
        //    (where retrying the same address returns 0): every reported prefix
        //    must be credited once and fallback must advance to the tail.
        let mut calls = Vec::new();
        let mut sim_pieces = Vec::new();
        let sim_base = 0x1000_0000usize;
        let sim_written = super::write_mapped_raw_impl(
            sim_base,
            2 * OS_PAGE,
            |at, n| {
                calls.push((at - sim_base, n));
                match calls.len() {
                    // Top-level write fails with 0 progress.
                    1 => (false, 0),
                    // 64 KiB piece writes 1024-byte prefix then fails.
                    2 => (false, 1024),
                    // First 4 KiB tail piece (`1024..4096`) writes 512 bytes then fails.
                    3 => (false, 512),
                    // Second 4 KiB piece (`4096..8192`) succeeds completely.
                    4 => (true, n),
                    _ => (false, 0),
                }
            },
            |off, n| sim_pieces.push((off, n)),
        );
        assert_eq!(sim_written, 1024 + 512 + OS_PAGE);
        assert_eq!(sim_pieces, vec![(0, 1024), (1024, 512), (OS_PAGE, OS_PAGE)]);
        assert_eq!(
            calls,
            vec![
                (0, 2 * OS_PAGE),
                (0, 2 * OS_PAGE),
                (1024, OS_PAGE - 1024),
                (OS_PAGE, OS_PAGE),
            ]
        );
    }

    #[test]
    fn copy_back_writes_mapped_pages_only() {
        let from = pattern();
        let to = holed(0xee);
        assert_eq!(write_mapped(to, from.as_ptr(), 3 * BLOCK), Some(2 * BLOCK));
        assert_eq!(block(to, 0), &from[..BLOCK]);
        assert_eq!(block(to, 2), &from[2 * BLOCK..]);
        release(to);
        // Neither aligned nor whole: the bytes around the range stay.
        let to = holed(0xee);
        // SAFETY: inside the first block.
        let dst_off = unsafe { to.add(100) };
        assert_eq!(
            write_mapped(dst_off, from.as_ptr(), BLOCK - 300),
            Some(BLOCK - 300)
        );
        let first = block(to, 0);
        assert_eq!(first[100..BLOCK - 200], from[..BLOCK - 300]);
        let around = first[..100].iter().chain(&first[BLOCK - 200..]);
        assert!(around.into_iter().all(|&b| b == 0xee));
        // A page that is mapped but not writable is not written either
        // (guarded title blocks are opened before they are copied back).
        let mut old = 0;
        // SAFETY: a committed block of ours.
        let protected =
            unsafe { VirtualProtect(to.add(2 * BLOCK).cast(), BLOCK, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        // SAFETY: the third block.
        let third_block = unsafe { to.add(2 * BLOCK) };
        assert_eq!(write_mapped(third_block, from.as_ptr(), BLOCK), Some(0));
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
    }

    #[test]
    fn title_memory_write_reports_unmapped_destination_without_faulting() {
        let from = pattern();
        let to = holed(0xee);
        assert!(write_title_memory(to as usize, &from[..BLOCK]));
        assert_eq!(block(to, 0), &from[..BLOCK]);
        assert!(!write_title_memory(to as usize, &from));
        assert_eq!(block(to, 0), &from[..BLOCK]);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
    }
}

/// Queries the adapter's current local VRAM budget in bytes via DXGI.
fn query_local_vram_budget(adapter: &IDXGIAdapter3) -> Option<u64> {
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: valid adapter and output struct.
    unsafe {
        adapter
            .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)
            .ok()?;
    }
    (info.Budget > 0).then_some(info.Budget)
}

/// Decides `(rebar_scratch_enabled, rebar_shadow_budget_bytes)` for the active
/// adapter.
///
/// - Disabled when `allow_rebar` is false (`VICTORY_REBAR=0` in diagnostics
///   builds), when the adapter/driver does not support `D3D12_HEAP_TYPE_GPU_UPLOAD`,
///   or on UMA/APU adapters (where `L0` write-back `shared_buffer` already lives
///   in unified local memory without write-combining read penalties).
/// - On discrete GPUs with ReBAR support, scales the shadow VRAM budget by local
///   VRAM so low-VRAM cards (`<= 8 GiB`) stay safely on system-RAM shadows while
///   12-32+ GiB cards keep hot buffers resident in VRAM.
fn rebar_policy(
    gpu_upload_supported: bool,
    uma: bool,
    local_vram_budget: Option<u64>,
    allow_rebar: bool,
) -> (bool, u64) {
    let (scratch, shadow_budget, _) =
        rebar_tier_budgets(gpu_upload_supported, uma, local_vram_budget, allow_rebar);
    (scratch, shadow_budget)
}

/// Decides the ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) budget in bytes for `LOOSE`
/// windows on the active adapter.
fn rebar_loose_policy(
    gpu_upload_supported: bool,
    uma: bool,
    local_vram_budget: Option<u64>,
    allow_rebar: bool,
) -> u64 {
    rebar_tier_budgets(gpu_upload_supported, uma, local_vram_budget, allow_rebar).2
}

fn rebar_tier_budgets(
    gpu_upload_supported: bool,
    uma: bool,
    local_vram_budget: Option<u64>,
    allow_rebar: bool,
) -> (bool, u64, u64) {
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    if !allow_rebar || !gpu_upload_supported || uma {
        return (false, 0, 0);
    }
    let Some(budget) = local_vram_budget else {
        return (false, 0, 0);
    };
    if budget <= 8 * GIB {
        (false, 0, 0)
    } else if budget <= 14 * GIB {
        (true, 512 * MIB, 0)
    } else if budget < 20 * GIB {
        (true, GIB, 2 * GIB)
    } else if budget < 28 * GIB {
        (true, 2 * GIB, 4 * GIB)
    } else {
        (true, 3 * GIB, 6 * GIB)
    }
}

fn parse_rebar_mb_override(raw: Option<&str>) -> Option<u64> {
    raw.and_then(|v| v.trim().parse::<u64>().ok())
        .map(|mb| mb.saturating_mul(1 << 20))
}

fn resolve_rebar_budgets(
    gpu_upload_supported: bool,
    uma: bool,
    local_vram_budget: Option<u64>,
    allow_rebar: bool,
    rebar_mb_override: Option<&str>,
    rebar_loose_mb_override: Option<&str>,
) -> (bool, u64, u64) {
    if !allow_rebar || !gpu_upload_supported || uma {
        return (false, 0, 0);
    }
    let (scratch, mut shadow_budget) =
        rebar_policy(gpu_upload_supported, uma, local_vram_budget, allow_rebar);
    let mut loose_budget =
        rebar_loose_policy(gpu_upload_supported, uma, local_vram_budget, allow_rebar);
    if let Some(bytes) = parse_rebar_mb_override(rebar_mb_override) {
        shadow_budget = bytes;
    }
    if let Some(bytes) = parse_rebar_mb_override(rebar_loose_mb_override) {
        loose_budget = bytes;
    }
    (scratch, shadow_budget, loose_budget)
}

/// Merges overlapping or adjacent half-open byte intervals `[from, to)` into
/// sorted disjoint intervals.
fn merge_ranges(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.retain(|&(from, to)| from < to);
    if ranges.len() <= 1 {
        return ranges;
    }
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match merged.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => merged.push((from, to)),
        }
    }
    merged
}

/// Converts sorted disjoint buffer-offset byte intervals `[from, to)` into
/// sorted, disjoint, non-adjacent 64 KiB piece runs `(first_piece, count)` so
/// each overlapping piece is hashed and compared at most once per submission.
fn spans_to_piece_runs(spans: &[(usize, usize)]) -> Vec<(u32, u32)> {
    use crate::watch::PAGE;
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &(from, to) in spans {
        if from >= to {
            continue;
        }
        let Ok(first) = u32::try_from(from / PAGE) else {
            continue;
        };
        let Ok(end) = u32::try_from(to.div_ceil(PAGE)) else {
            continue;
        };
        if first >= end {
            continue;
        }
        match runs.last_mut() {
            Some(last) if first <= last.0 + last.1 => {
                let new_end = (last.0 + last.1).max(end);
                last.1 = new_end - last.0;
            }
            _ => runs.push((first, end - first)),
        }
    }
    runs
}

/// Walks `piece_runs` paired with their per-piece GPU comparison `flags`,
/// coalescing contiguous flagged (`!= 0`) 64 KiB pieces into sorted disjoint
/// buffer-offset byte intervals clamped to `len`. Returns `(byte_runs, changed_pieces)`.
fn flagged_piece_byte_runs(
    piece_runs: &[(u32, u32)],
    flags: &[u32],
    len: usize,
) -> (Vec<(usize, usize)>, u64) {
    use crate::watch::PAGE;
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut cur: Option<(u32, u32)> = None;
    let mut idx = 0usize;
    let mut changed = 0u64;
    for &(first, count) in piece_runs {
        for off in 0..count {
            let flagged = flags.get(idx).copied().unwrap_or(1) != 0;
            idx += 1;
            if flagged {
                changed += 1;
                let piece = first + off;
                match cur {
                    Some((s, e)) if e == piece => cur = Some((s, piece + 1)),
                    Some((s, e)) => {
                        let lo = (s as usize) * PAGE;
                        let hi = ((e as usize) * PAGE).min(len);
                        if lo < hi {
                            out.push((lo, hi));
                        }
                        cur = Some((piece, piece + 1));
                    }
                    None => cur = Some((piece, piece + 1)),
                }
            } else if let Some((s, e)) = cur.take() {
                let lo = (s as usize) * PAGE;
                let hi = ((e as usize) * PAGE).min(len);
                if lo < hi {
                    out.push((lo, hi));
                }
            }
        }
    }
    if let Some((s, e)) = cur {
        let lo = (s as usize) * PAGE;
        let hi = ((e as usize) * PAGE).min(len);
        if lo < hi {
            out.push((lo, hi));
        }
    }
    (out, changed)
}

/// Intersects two sorted disjoint half-open interval lists `a` and `b`.
fn intersect_ranges(a: &[(usize, usize)], b: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0);
        let hi = a[i].1.min(b[j].1);
        if lo < hi {
            out.push((lo, hi));
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

impl Host {
    /// Local and non-local video memory `[local_budget, local_usage, shared_budget, shared_usage]`
    /// in bytes, or `None` on adapters without `IDXGIAdapter3` (e.g. WARP).
    pub(crate) fn video_memory(&self) -> Option<[u64; 4]> {
        let adapter = self.adapter.as_ref()?;
        let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: valid adapter and output structs.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        Some([
            local.Budget,
            local.CurrentUsage,
            shared.Budget,
            shared.CurrentUsage,
        ])
    }

    /// Adapter local VRAM budget in bytes, if reported by DXGI.
    pub(crate) fn local_vram_budget(&self) -> Option<u64> {
        self.video_memory().map(|[budget, ..]| budget)
    }

    /// Whether allocating `extra_bytes` of local VRAM stays under 85% of the
    /// adapter's current local VRAM budget (leaves headroom for textures and
    /// render targets).
    fn has_vram_headroom(&self, extra_bytes: u64) -> bool {
        let Some([budget, used, ..]) = self.video_memory() else {
            return true;
        };
        if budget == 0 {
            return false;
        }
        let limit = (budget / 100).saturating_mul(85);
        used.saturating_add(extra_bytes) <= limit
    }

    /// Byte budget reserved for ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) shadow
    /// buffers on this adapter (0 when ReBAR shadows are disabled).
    pub(crate) fn rebar_shadow_budget(&self) -> u64 {
        self.rebar_shadow_budget
    }

    /// Byte budget reserved for ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) `LOOSE`
    /// windows on this adapter (0 when ReBAR `LOOSE` windows are disabled).
    pub(crate) fn rebar_loose_budget(&self) -> u64 {
        self.rebar_loose_budget
    }

    /// A buffer in `D3D12_HEAP_TYPE_GPU_UPLOAD` (CPU-writable local VRAM via
    /// Resizable BAR), starting in `COMMON` and mapped for sequential CPU
    /// writes. Callers MUST NOT read from the returned pointer on the CPU.
    fn gpu_upload_buffer(
        &self,
        size: u64,
        flags: D3D12_RESOURCE_FLAGS,
    ) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
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
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-writable VRAM buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    let read_range = D3D12_RANGE { Begin: 0, End: 0 };
                    b.Map(0, Some(&read_range), Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                static WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        error = %e,
                        size,
                        "GPU upload heap allocation failed; falling back to system memory"
                    );
                }
                None
            }
        }
    }

    /// A scratch buffer for write-only per-draw host data: placed in
    /// `D3D12_HEAP_TYPE_GPU_UPLOAD` when ReBAR is enabled and VRAM has
    /// headroom, otherwise falls back to standard `D3D12_HEAP_TYPE_UPLOAD`.
    pub(crate) fn scratch_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if self.rebar_scratch
            && self.has_vram_headroom(size)
            && let Some(buf) = self.gpu_upload_buffer(size, D3D12_RESOURCE_FLAG_NONE)
        {
            return Some(buf);
        }
        self.upload_buffer(size)
    }

    /// Attempts to allocate a UAV-capable ReBAR shadow buffer in local VRAM.
    /// Returns `None` if ReBAR shadows are disabled, VRAM headroom is low, or
    /// allocation fails (callers fall back to [`Host::shared_buffer`]).
    pub(crate) fn try_rebar_shadow_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if (self.rebar_shadow_budget == 0 && self.rebar_loose_budget == 0)
            || !self.has_vram_headroom(size)
        {
            return None;
        }
        self.gpu_upload_buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    }
}

/// One unique ReBAR shadow with declared `GpuWrite` ranges in a submission.
struct RebarWriteEntry {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    declared_spans: Vec<(usize, usize)>,
    piece_runs: Vec<(u32, u32)>,
    /// Starting index in the GPU hash-compare `flags` array if GPU hashing ran
    /// for this entry; `None` falls back to reading back all `declared_spans`.
    compare_out_base: Option<usize>,
    compare_pieces: usize,
}

type DeclaredWrite = (std::sync::Arc<crate::shadow::Shadow>, usize, usize);

/// Groups unique declared GPU writes in O(R) into per-shadow `back` ranges
/// (for card/async readback), plus ReBAR entries and non-ReBAR/non-carded
/// writes when `!late`.
fn collect_declared_writes(
    writes: Vec<DeclaredWrite>,
    late: bool,
) -> (
    Vec<relay::Declared>,
    Vec<RebarWriteEntry>,
    Vec<DeclaredWrite>,
) {
    let mut back: Vec<relay::Declared> = Vec::new();
    let mut back_by_shadow: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut seen_back: std::collections::HashSet<(usize, usize, usize)> =
        std::collections::HashSet::new();
    let mut rebar_entries: Vec<RebarWriteEntry> = Vec::new();
    let mut by_shadow: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut seen_rebar: std::collections::HashSet<(usize, usize, usize)> =
        std::collections::HashSet::new();
    let mut non_rebar_writes = Vec::new();
    for (shadow, address, len) in writes {
        let ptr = std::sync::Arc::as_ptr(&shadow) as usize;
        let end = address.saturating_add(len);
        if address < end {
            let idx = *back_by_shadow.entry(ptr).or_insert_with(|| {
                let i = back.len();
                back.push((shadow.clone(), Vec::new()));
                i
            });
            if seen_back.insert((ptr, address, end)) {
                back[idx].1.push((address, end));
            }
        }
        if !late {
            if shadow.is_rebar() {
                let (from, to) = shadow.inside(address, len);
                let idx = *by_shadow.entry(ptr).or_insert_with(|| {
                    let i = rebar_entries.len();
                    rebar_entries.push(RebarWriteEntry {
                        shadow,
                        declared_spans: Vec::new(),
                        piece_runs: Vec::new(),
                        compare_out_base: None,
                        compare_pieces: 0,
                    });
                    i
                });
                if from < to && seen_rebar.insert((ptr, from, to)) {
                    rebar_entries[idx].declared_spans.push((from, to));
                }
            } else if !shadow.carded() {
                non_rebar_writes.push((shadow, address, len));
            }
        }
    }
    (back, rebar_entries, non_rebar_writes)
}

/// Optional pre-submission rehash and post-submission hash-compare command
/// lists recorded for ReBAR UAV shadows in one submission.
#[derive(Default)]
struct RebarHashPlan {
    pre_list: Option<ID3D12CommandList>,
    post_list: Option<ID3D12CommandList>,
    total_flag_pieces: usize,
}

impl Host {
    /// Ensures a reusable closed direct command list exists in `slot`.
    fn ensure_direct_list<'a>(
        &self,
        slot: &'a mut Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    ) -> windows::core::Result<&'a (ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
        if slot.is_none() {
            // SAFETY: valid device creating a direct allocator and closed list.
            let pair = unsafe {
                let allocator: ID3D12CommandAllocator = self
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &allocator,
                    None,
                )?;
                list.Close()?;
                (allocator, list)
            };
            *slot = Some(pair);
        }
        Ok(slot.as_ref().unwrap())
    }

    /// Submits a single closed helper command list on `self.queue` and waits
    /// on `self.fence` (`submitted` must be held by the caller).
    fn submit_helper_list(&self, submitted: &mut u64, cmd: ID3D12CommandList, label: &str) -> bool {
        *submitted += 1;
        // SAFETY: `cmd` is closed; `fence` and `event` are owned by `self` and
        // serialized under `submitted`.
        unsafe {
            let signalled = {
                let _lazy_order = crate::lazy_back::direct_submit_guard();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                let res = self.queue.Signal(&self.fence, *submitted);
                if res.is_ok() {
                    self.note_signalled(*submitted);
                }
                res
            };
            match signalled {
                Ok(()) => match self.fence.SetEventOnCompletion(*submitted, self.event) {
                    Ok(()) => {
                        let waited = WaitForSingleObject(self.event, INFINITE);
                        if waited != WAIT_OBJECT_0 {
                            self.poll_fence(*submitted);
                        }
                    }
                    Err(_) => self.poll_fence(*submitted),
                },
                Err(e) => {
                    tracing::warn!(error = %e, "{label} fence signal failed");
                    return false;
                }
            }
        }
        true
    }

    /// Builds optional pre-submission hash-update and post-submission
    /// hash-compare command lists for UAV-written ReBAR shadows so copy-back
    /// only DMAs 64 KiB pieces whose GPU hash actually changed. Falls back
    /// cleanly to full-range readbacks if the compute pipeline or buffers are
    /// unavailable.
    fn prepare_rebar_hash_plan(&self, entries: &mut [RebarWriteEntry]) -> RebarHashPlan {
        let Some(pipe) = rebar_hash_pipeline(&self.device) else {
            return RebarHashPlan::default();
        };
        struct PreparedHash {
            entry_idx: usize,
            hashes_gpu: u64,
            aligned_len: u32,
            pre_runs: Vec<(u32, u32)>,
            out_base: usize,
            pieces: usize,
        }
        let mut prepared: Vec<PreparedHash> = Vec::with_capacity(entries.len());
        let mut total_flag_pieces = 0usize;
        for (entry_idx, entry) in entries.iter().enumerate() {
            let entry_pieces: usize = entry.piece_runs.iter().map(|&(_, c)| c as usize).sum();
            if entry_pieces == 0 {
                continue;
            }
            let Some(crate::shadow::PreparedRebarHashes {
                hashes_gpu,
                aligned_len,
                pre_runs,
            }) = entry.shadow.prepare_rebar_hashes(|n| {
                self.default_buffer_in_state(n, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            })
            else {
                continue;
            };
            let Some(next_total) = total_flag_pieces.checked_add(entry_pieces) else {
                entry.shadow.mark_rebar_dirty_runs(&pre_runs);
                continue;
            };
            if (next_total as u64).saturating_mul(4) > REBAR_FLAGS_MAX_BYTES {
                entry.shadow.mark_rebar_dirty_runs(&pre_runs);
                continue;
            }
            prepared.push(PreparedHash {
                entry_idx,
                hashes_gpu,
                aligned_len,
                pre_runs,
                out_base: total_flag_pieces,
                pieces: entry_pieces,
            });
            total_flag_pieces = next_total;
        }
        if prepared.is_empty() {
            return RebarHashPlan::default();
        }
        let restore_dirty = |entries: &[RebarWriteEntry], prepared: &[PreparedHash]| {
            for p in prepared {
                if !p.pre_runs.is_empty() {
                    entries[p.entry_idx]
                        .shadow
                        .mark_rebar_dirty_runs(&p.pre_runs);
                }
            }
        };
        let mut rb = self
            .rebar_readback
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let need_bytes = ((total_flag_pieces as u64) * 4).max(4);
        let need_flags_alloc = rb
            .flags_uav
            .as_ref()
            .is_none_or(|(_, _, c)| *c < need_bytes)
            || rb
                .flags_readback
                .as_ref()
                .is_none_or(|(_, _, c)| *c < need_bytes);
        if need_flags_alloc {
            let cap = need_bytes
                .next_power_of_two()
                .clamp(REBAR_FLAGS_MIN_BYTES, REBAR_FLAGS_MAX_BYTES);
            let Some(uav) =
                self.default_buffer_in_state(cap, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            else {
                restore_dirty(entries, &prepared);
                return RebarHashPlan::default();
            };
            let Some((rb_buf, rb_ptr)) = self.readback_buffer(cap) else {
                restore_dirty(entries, &prepared);
                return RebarHashPlan::default();
            };
            name(&uav, || format!("ReBAR hash flags UAV {cap:#x}"));
            name(&rb_buf, || format!("ReBAR hash flags readback {cap:#x}"));
            // SAFETY: live D3D12 buffer.
            let uav_gpu = unsafe { uav.GetGPUVirtualAddress() };
            rb.flags_uav = Some((uav, uav_gpu, cap));
            rb.flags_readback = Some((rb_buf, rb_ptr as usize, cap));
        }
        let (flags_uav_res, flags_uav_gpu, _) = rb.flags_uav.as_ref().unwrap().clone();
        let flags_rb_res = rb.flags_readback.as_ref().unwrap().0.clone();

        // Record pre-submission hash pass for shadows that just created `hashes`
        // or received CPU writes (`Shadow::copy` / `copy_back_rebar`) since last hash.
        let pre_list = if prepared.iter().any(|p| !p.pre_runs.is_empty()) {
            let Ok((allocator, list)) = self.ensure_direct_list(&mut rb.pre_hash_list) else {
                restore_dirty(entries, &prepared);
                return RebarHashPlan::default();
            };
            let to_srv: Vec<D3D12_RESOURCE_BARRIER> = prepared
                .iter()
                .filter(|p| !p.pre_runs.is_empty())
                .map(|p| {
                    transition(
                        &entries[p.entry_idx].shadow.buffer,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    )
                })
                .collect();
            let to_common: Vec<D3D12_RESOURCE_BARRIER> = prepared
                .iter()
                .filter(|p| !p.pre_runs.is_empty())
                .map(|p| {
                    transition(
                        &entries[p.entry_idx].shadow.buffer,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    )
                })
                .collect();
            // SAFETY: `submitted` is held exclusively so no prior pre-hash list
            // is in flight; all shadow buffers enter in COMMON and return to
            // COMMON; `hashes` and `flags_uav` live in UNORDERED_ACCESS.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    list.ResourceBarrier(&to_srv);
                    list.SetComputeRootSignature(&pipe.root);
                    list.SetPipelineState(&pipe.pso);
                    list.SetComputeRootUnorderedAccessView(3, flags_uav_gpu);
                    for p in prepared.iter().filter(|p| !p.pre_runs.is_empty()) {
                        list.SetComputeRootShaderResourceView(
                            1,
                            entries[p.entry_idx].shadow.base_gpu(),
                        );
                        list.SetComputeRootUnorderedAccessView(2, p.hashes_gpu);
                        for &(first, count) in &p.pre_runs {
                            let constants = [first, p.aligned_len, 0u32, 0u32, count];
                            list.SetComputeRoot32BitConstants(0, 5, constants.as_ptr().cast(), 0);
                            list.Dispatch(count, 1, 1);
                        }
                    }
                    list.ResourceBarrier(&to_common);
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(cmd) => Some(cmd),
                Err(e) => {
                    tracing::warn!(error = %e, "could not record ReBAR pre-hash list");
                    restore_dirty(entries, &prepared);
                    return RebarHashPlan::default();
                }
            }
        } else {
            None
        };

        // Record post-submission hash-compare pass over all declared piece runs,
        // copying the tiny `flags` dword array to `flags_readback`.
        let Ok((allocator, list)) = self.ensure_direct_list(&mut rb.post_compare_list) else {
            restore_dirty(entries, &prepared);
            return RebarHashPlan::default();
        };
        let mut before_barriers = Vec::with_capacity(prepared.len() + 1);
        before_barriers.push(crate::indirect::uav_barrier());
        for p in &prepared {
            before_barriers.push(transition(
                &entries[p.entry_idx].shadow.buffer,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            ));
        }
        let mut after_barriers = Vec::with_capacity(prepared.len() + 1);
        after_barriers.push(transition(
            &flags_uav_res,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        ));
        for p in &prepared {
            after_barriers.push(transition(
                &entries[p.entry_idx].shadow.buffer,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            ));
        }
        // SAFETY: `submitted` is held exclusively so no prior post-compare list
        // is in flight; `HostList::close` returned every shadow buffer to
        // COMMON; `flags_uav` transitions UAV -> COPY_SOURCE -> UAV and
        // `flags_rb_res` stays permanently in COPY_DEST covering `total_flag_pieces * 4`.
        let recorded = unsafe {
            (|| -> windows::core::Result<ID3D12CommandList> {
                allocator.Reset()?;
                list.Reset(allocator, None)?;
                list.ResourceBarrier(&before_barriers);
                list.SetComputeRootSignature(&pipe.root);
                list.SetPipelineState(&pipe.pso);
                list.SetComputeRootUnorderedAccessView(3, flags_uav_gpu);
                for p in &prepared {
                    list.SetComputeRootShaderResourceView(
                        1,
                        entries[p.entry_idx].shadow.base_gpu(),
                    );
                    list.SetComputeRootUnorderedAccessView(2, p.hashes_gpu);
                    let mut out_idx = p.out_base;
                    for &(first, count) in &entries[p.entry_idx].piece_runs {
                        let constants = [first, p.aligned_len, (out_idx * 4) as u32, 1u32, count];
                        list.SetComputeRoot32BitConstants(0, 5, constants.as_ptr().cast(), 0);
                        list.Dispatch(count, 1, 1);
                        out_idx += count as usize;
                    }
                }
                list.ResourceBarrier(&after_barriers);
                list.CopyBufferRegion(
                    &flags_rb_res,
                    0,
                    &flags_uav_res,
                    0,
                    (total_flag_pieces as u64) * 4,
                );
                list.ResourceBarrier(&[transition(
                    &flags_uav_res,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
                list.Close()?;
                list.cast()
            })()
        };
        let post_list = match recorded {
            Ok(cmd) => Some(cmd),
            Err(e) => {
                tracing::warn!(error = %e, "could not record ReBAR post-compare list");
                restore_dirty(entries, &prepared);
                return RebarHashPlan::default();
            }
        };
        drop(rb);
        for p in prepared {
            entries[p.entry_idx].compare_out_base = Some(p.out_base);
            entries[p.entry_idx].compare_pieces = p.pieces;
        }
        RebarHashPlan {
            pre_list,
            post_list,
            total_flag_pieces,
        }
    }

    /// Reads GPU hash-compare flags (if run), coalesces contiguous flagged
    /// 64 KiB pieces intersected with declared UAV spans, and DMAs only those
    /// changed runs into `READBACK` for [`crate::shadow::Shadow::copy_back_rebar`].
    ///
    /// Returns `false` if readback or mirror merge fails; any entry with
    /// undelivered spans keeps its `pending` ranges intact so refreshes and
    /// loose-cache eviction cannot overwrite or drop GPU results.
    fn finish_rebar_copy_back(
        &self,
        submitted: &mut u64,
        entries: Vec<RebarWriteEntry>,
        total_flag_pieces: usize,
    ) -> bool {
        self.finish_rebar_copy_back_with(entries, total_flag_pieces, |jobs, f| {
            self.copy_rebar_readback(submitted, jobs, f)
        })
    }

    fn finish_rebar_copy_back_with(
        &self,
        entries: Vec<RebarWriteEntry>,
        total_flag_pieces: usize,
        readback: impl FnOnce(&[(&ID3D12Resource, u64, u64)], &mut dyn FnMut(&[&[u8]])) -> bool,
    ) -> bool {
        struct PendingCopy {
            shadow: std::sync::Arc<crate::shadow::Shadow>,
            spans: Vec<(usize, usize)>,
            first: usize,
            end: usize,
        }
        let mut pending: Vec<PendingCopy> = Vec::with_capacity(entries.len());
        let mut total_jobs = 0usize;
        let mut compared_total = 0u64;
        let mut changed_total = 0u64;
        {
            let rb = self
                .rebar_readback
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let flags_all: Option<&[u32]> = if total_flag_pieces > 0 {
                rb.flags_readback.as_ref().and_then(|(_, ptr, cap)| {
                    ((total_flag_pieces as u64) * 4 <= *cap).then(|| {
                        // SAFETY: submission fence completed and `ptr` is mapped
                        // for `*cap >= total_flag_pieces * 4` bytes of `READBACK`.
                        unsafe {
                            std::slice::from_raw_parts(
                                (*ptr as *const u8).cast::<u32>(),
                                total_flag_pieces,
                            )
                        }
                    })
                })
            } else {
                None
            };
            for entry in entries {
                let spans = if let (Some(base), Some(all)) = (entry.compare_out_base, flags_all)
                    && base + entry.compare_pieces <= all.len()
                {
                    let slice = &all[base..base + entry.compare_pieces];
                    let (flagged, changed) = flagged_piece_byte_runs(
                        &entry.piece_runs,
                        slice,
                        entry.shadow.len() as usize,
                    );
                    compared_total += entry.compare_pieces as u64;
                    changed_total += changed;
                    intersect_ranges(&flagged, &entry.declared_spans)
                } else {
                    // Fallback: read back the entire merged declared spans and
                    // mark them dirty in any existing hash buffer.
                    entry.shadow.mark_rebar_dirty_spans(&entry.declared_spans);
                    entry.declared_spans
                };
                let first = total_jobs;
                total_jobs += spans.len();
                pending.push(PendingCopy {
                    shadow: entry.shadow,
                    spans,
                    first,
                    end: total_jobs,
                });
            }
        }
        #[cfg(test)]
        if compared_total > 0 {
            self.rebar_hash_compared_pieces
                .fetch_add(compared_total, Ordering::Relaxed);
            self.rebar_hash_changed_pieces
                .fetch_add(changed_total, Ordering::Relaxed);
        }
        let _ = (compared_total, changed_total);
        if total_jobs == 0 {
            for entry in &pending {
                entry.shadow.clear_pending_rebar();
            }
            return true;
        }
        let mut copy_jobs: Vec<(&ID3D12Resource, u64, u64)> = Vec::with_capacity(total_jobs);
        let mut copy_bytes = 0u64;
        for entry in &pending {
            for &(from, to) in &entry.spans {
                let len = (to - from) as u64;
                copy_bytes += len;
                copy_jobs.push((&entry.shadow.buffer, from as u64, len));
            }
        }
        #[cfg(test)]
        {
            self.rebar_post_readback_calls
                .fetch_add(1, Ordering::Relaxed);
            self.rebar_post_readback_bytes
                .fetch_add(copy_bytes, Ordering::Relaxed);
        }
        let _ = copy_bytes;
        let mut all_delivered = true;
        let mut deliver = |slices: &[&[u8]]| {
            for entry in &pending {
                if entry.first < entry.end {
                    match entry
                        .shadow
                        .copy_back_rebar(&entry.spans, &slices[entry.first..entry.end])
                    {
                        Some(written) => {
                            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                        }
                        None => {
                            entry.shadow.mark_rebar_dirty_spans(&entry.spans);
                            all_delivered = false;
                        }
                    }
                } else {
                    entry.shadow.clear_pending_rebar();
                }
            }
        };
        let ok = readback(&copy_jobs, &mut deliver);
        if !ok {
            for entry in &pending {
                if !entry.spans.is_empty() {
                    // Keep `pending` ranges intact so refreshes and loose-cache
                    // eviction cannot overwrite or drop undelivered GPU bytes.
                    entry.shadow.mark_rebar_dirty_spans(&entry.spans);
                } else {
                    entry.shadow.clear_pending_rebar();
                }
            }
            return false;
        }
        all_delivered
    }

    /// DMAs `copies` (`(src_buffer, src_offset, len)`) from ReBAR buffers into
    /// a cached system-RAM readback buffer on `self.queue`, waits for `submitted`
    /// (already locked by the caller), and invokes `f` with the resulting slices.
    fn copy_rebar_readback(
        &self,
        submitted: &mut u64,
        copies: &[(&ID3D12Resource, u64, u64)],
        f: impl FnOnce(&[&[u8]]),
    ) -> bool {
        if copies.is_empty() {
            f(&[]);
            return true;
        }
        let mut total: u64 = 0;
        let mut offsets: Vec<(usize, usize)> = Vec::with_capacity(copies.len());
        for &(_, _, len) in copies {
            let Some(end) = total.checked_add(len) else {
                return false;
            };
            offsets.push((total as usize, len as usize));
            total = end;
        }
        if total == 0 {
            let empty: Vec<&[u8]> = vec![&[]; copies.len()];
            f(&empty);
            return true;
        }
        let mut rb = self
            .rebar_readback
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Err(e) = self.ensure_direct_list(&mut rb.list) {
            tracing::warn!(error = %e, "could not create ReBAR readback command list");
            return false;
        }
        let mut transient: Option<(ID3D12Resource, *const u8)> = None;
        let (dst_res, dst_ptr): (ID3D12Resource, *const u8) = if total > REBAR_READBACK_KEEP_CAP {
            let Some((buf, ptr)) = self.readback_buffer(total) else {
                return false;
            };
            transient = Some((buf.clone(), ptr));
            (buf, ptr)
        } else {
            let need_new = rb.buffer.as_ref().is_none_or(|(_, _, cap)| *cap < total);
            if need_new {
                let cap = total
                    .next_power_of_two()
                    .clamp(1 << 20, REBAR_READBACK_KEEP_CAP);
                let Some((buf, ptr)) = self.readback_buffer(cap) else {
                    return false;
                };
                name(&buf, || format!("ReBAR readback {cap:#x}"));
                rb.buffer = Some((buf, ptr as usize, cap));
            }
            let (buf, ptr, _) = rb.buffer.as_ref().unwrap();
            (buf.clone(), *ptr as *const u8)
        };
        let (allocator, list) = rb.list.as_ref().unwrap();
        // SAFETY: `submitted` is held exclusively so no previous readback is in
        // flight; all source buffers are in COMMON state (implicitly promoting
        // to COPY_SOURCE and decaying back to COMMON on completion), and
        // `dst_res` is a READBACK buffer in COPY_DEST state covering `total`.
        let recorded = unsafe {
            (|| -> windows::core::Result<()> {
                allocator.Reset()?;
                list.Reset(allocator, None)?;
                for (&(src, src_off, len), &(dst_off, _)) in copies.iter().zip(&offsets) {
                    if len > 0 {
                        list.CopyBufferRegion(&dst_res, dst_off as u64, src, src_off, len);
                    }
                }
                list.Close()?;
                Ok(())
            })()
        };
        if let Err(e) = recorded {
            tracing::warn!(error = %e, "could not record ReBAR readback list");
            drop(transient);
            return false;
        }
        let Ok(cmd): windows::core::Result<ID3D12CommandList> = list.cast() else {
            drop(transient);
            return false;
        };
        if !self.submit_helper_list(submitted, cmd, "ReBAR readback") {
            drop(transient);
            return false;
        }
        let slices: Vec<&[u8]> = offsets
            .iter()
            .map(|&(off, len)| {
                // SAFETY: GPU copy completed and `dst_ptr` covers `total` bytes.
                unsafe { std::slice::from_raw_parts(dst_ptr.add(off), len) }
            })
            .collect();
        f(&slices);
        drop(transient);
        true
    }

    /// Reads the entire ReBAR shadow buffer into a cached system-RAM `Vec<u8>`
    /// once, on its first UAV write, without ever reading write-combining memory
    /// from the CPU.
    fn readback_rebar_to_vec(
        &self,
        submitted: &mut u64,
        buffer: &ID3D12Resource,
        size: u64,
    ) -> Option<Vec<u8>> {
        if size == 0 {
            return Some(Vec::new());
        }
        let mut out = None;
        let ok = self.copy_rebar_readback(submitted, &[(buffer, 0, size)], |slices| {
            out = Some(slices[0].to_vec());
        });
        ok.then_some(out).flatten()
    }

    /// A buffer in the upload heap, mapped for writing.
    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
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
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    let read_range = D3D12_RANGE { Begin: 0, End: 0 };
                    b.Map(0, Some(&read_range), Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }

    /// A buffer in the readback heap, mapped for reading.
    pub(crate) fn readback_buffer(&self, size: u64) -> Option<(ID3D12Resource, *const u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
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
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a readback buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no readback buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

/// Luminance bounds of the host display when HDR (HDR10 / PQ) is active on
/// Windows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdrDisplayInfo {
    pub min_luminance: f32,
    pub max_luminance: f32,
    pub max_full_frame_luminance: f32,
}

/// Whether this process uses more than 85% of the video memory budget the
/// OS gives it on the host adapter. Asked at most every 250 ms; false if
/// the adapter cannot say.
pub(crate) fn video_memory_tight() -> bool {
    use std::sync::atomic::{AtomicBool, AtomicU64};
    static ASKED_MS: AtomicU64 = AtomicU64::new(0);
    static TIGHT: AtomicBool = AtomicBool::new(false);
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let now = START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
        + 1;
    let asked = ASKED_MS.load(Ordering::Relaxed);
    if asked != 0 && now - asked < 250
        || ASKED_MS
            .compare_exchange(asked, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return TIGHT.load(Ordering::Relaxed);
    }
    let Some(host) = host() else { return false };
    // SAFETY: plain DXGI factory and adapter queries.
    let tight = unsafe {
        CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter3>(host.device.GetAdapterLuid()))
            .and_then(|a| {
                let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                a.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)
                    .map(|()| info.Budget > 0 && info.CurrentUsage > info.Budget / 100 * 85)
            })
            .unwrap_or(false)
    };
    if tight && !TIGHT.load(Ordering::Relaxed) {
        tracing::info!("video memory near its budget: new read-only textures go to system memory");
    }
    TIGHT.store(tight, Ordering::Relaxed);
    tight
}

/// Bytes of video memory made since start (frees not counted): render
/// targets and depth, other textures, GPU-only buffers.
#[cfg(feature = "diagnostics")]
pub(crate) static VRAM_MADE: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Counts `resource` in [`VRAM_MADE`] slot `kind`.
#[cfg(feature = "diagnostics")]
pub(crate) fn note_vram(kind: usize, resource: &ID3D12Resource) {
    let Some(host) = host() else { return };
    // SAFETY: plain queries on a live resource and device.
    let bytes = unsafe {
        let desc = resource.GetDesc();
        host.device
            .GetResourceAllocationInfo(0, &[desc])
            .SizeInBytes
    };
    VRAM_MADE[kind].fetch_add(bytes, Ordering::Relaxed);
}

/// Logs the host adapter's video memory (local) and the system memory it
/// maps (non-local): what this process uses of each against the budget the
/// OS gives it.
#[cfg(feature = "diagnostics")]
fn log_video_memory() {
    let Some(host) = host() else { return };
    // SAFETY: plain DXGI factory and adapter queries.
    unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return;
        };
        let Ok(adapter) = factory.EnumAdapterByLuid::<IDXGIAdapter3>(host.device.GetAdapterLuid())
        else {
            return;
        };
        let mb = |group| {
            let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            adapter
                .QueryVideoMemoryInfo(0, group, &mut info)
                .map(|()| (info.CurrentUsage >> 20, info.Budget >> 20))
                .unwrap_or_default()
        };
        let (local, local_budget) = mb(DXGI_MEMORY_SEGMENT_GROUP_LOCAL);
        let (shared, shared_budget) = mb(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL);
        let [targets, textures, buffers] = VRAM_MADE
            .each_ref()
            .map(|n| n.load(Ordering::Relaxed) >> 20);
        tracing::info!(
            "video memory {local}/{local_budget} MB, shared {shared}/{shared_budget} MB; made \
             since start: targets {targets} MB, textures {textures} MB, GPU buffers {buffers} MB"
        );
    }
}

/// Queries the host display outputs for an active HDR10 (`G2084_NONE_P2020`)
/// desktop mode. Returns `None` when HDR is disabled or unsupported.
pub fn display_hdr_info() -> Option<HdrDisplayInfo> {
    // SAFETY: plain DXGI factory, adapter and output enumeration.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let adapter: IDXGIAdapter1 = if let Some(host) = host() {
            let luid = host.device.GetAdapterLuid();
            factory
                .EnumAdapterByLuid::<IDXGIAdapter1>(luid)
                .or_else(|_| factory.EnumAdapters1(0))
                .ok()?
        } else {
            factory.EnumAdapters1(0).ok()?
        };
        let mut i = 0u32;
        while let Ok(output) = adapter.EnumOutputs(i) {
            i += 1;
            let Ok(output6) = output.cast::<IDXGIOutput6>() else {
                continue;
            };
            let Ok(desc) = output6.GetDesc1() else {
                continue;
            };
            if desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 {
                return Some(HdrDisplayInfo {
                    min_luminance: desc.MinLuminance,
                    max_luminance: desc.MaxLuminance,
                    max_full_frame_luminance: desc.MaxFullFrameLuminance,
                });
            }
        }
        None
    }
}

fn desired_color_space(format: DXGI_FORMAT, hdr_active: bool) -> DXGI_COLOR_SPACE_TYPE {
    if !hdr_active {
        return DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    }
    match format {
        DXGI_FORMAT_R10G10B10A2_UNORM => DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
        DXGI_FORMAT_R16G16B16A16_FLOAT => DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,
        _ => DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
    }
}

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    color_space: DXGI_COLOR_SPACE_TYPE,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
    last_present_issue: Option<HRESULT>,
    compositor: Option<PresentCompositor>,
}

/// A successful DXGI status may still mean nothing was presented.
fn present_result(status: HRESULT) -> windows::core::Result<bool> {
    status.ok()?;
    Ok(status != DXGI_STATUS_OCCLUDED)
}

impl Presenter {
    fn sync_color_space(&mut self, hdr_active: bool) {
        let wanted = desired_color_space(self.format, hdr_active);
        if self.color_space == wanted {
            return;
        }
        // SAFETY: `self.swap_chain` is a live `IDXGISwapChain3`.
        let supported = unsafe { self.swap_chain.CheckColorSpaceSupport(wanted) }.unwrap_or(0);
        if supported & (DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT.0 as u32) == 0 {
            tracing::debug!(
                color_space = wanted.0,
                supported,
                "swap chain color space unsupported on host output"
            );
            return;
        }
        // SAFETY: `wanted` is supported for presentation on `self.swap_chain`.
        match unsafe { self.swap_chain.SetColorSpace1(wanted) } {
            Ok(()) => {
                self.color_space = wanted;
                tracing::info!(
                    format = self.format.0,
                    color_space = wanted.0,
                    "swap chain color space set"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, color_space = wanted.0, "SetColorSpace1 failed");
            }
        }
    }
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    logged_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        #[cfg(feature = "diagnostics")]
        crate::profile::start();
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            logged_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        #[cfg(feature = "diagnostics")]
        crate::timeline::wait_stats::frame();
        self.frames += 1;
        self.logged_frames += 1;
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] =
            [0, 1].map(|i| SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] =
            [0, 1].map(|i| COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames);
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        let [fence, memory, running, called] = crate::timeline::HELD_US
            .each_ref()
            .map(|n| n.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
        let [mesh_direct, mesh_indirect] =
            crate::draw::take_mesh_without_mesh_shaders().map(|n| n as f64 / frames);
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            let logged_frames = (std::mem::take(&mut self.logged_frames) as f64).max(1.0);
            crate::upload_pool::trim_idle(ram_idle_ms(available_physical_ram()));
            #[cfg(feature = "diagnostics")]
            log_video_memory();
            tracing::info!(
                "queues per frame (ms, summed over queues): held behind fences {fence:.1},                  behind memory waits {memory:.1}; running work {running:.1}; in the driver's                  ExecuteCommandLists {called:.1}"
            );
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect",
                making = crate::pipeline::making()
            );
            if mesh_direct + mesh_indirect > 0.0 {
                tracing::info!(
                    "mesh dispatches run as a compute pass and a draw (no mesh shaders on this                      host): {mesh_direct:.1}/frame direct, {mesh_indirect:.1}/frame indirect"
                );
            }
            let twins = crate::resource::TWINS[0].load(Ordering::Relaxed);
            if twins > 0 {
                let [to_depth, to_twin] = [2, 3]
                    .map(|i| crate::resource::TWINS[i].swap(0, Ordering::Relaxed) as f64 / frames);
                tracing::info!(
                    "depth twins: {twins} alive, {} depth textures left without one; \
                     copies per frame: {to_depth:.1} into depth textures, {to_twin:.1} into twins",
                    crate::resource::TWINS[1].load(Ordering::Relaxed)
                );
            }
            let rebar_b = crate::shadow::rebar_shadow_bytes();
            let rebar_loose_b = crate::shadow::rebar_loose_bytes();
            if rebar_b > 0 || rebar_loose_b > 0 {
                tracing::info!(
                    "rebar residency: rebar={} MiB, rebar_loose={} MiB",
                    rebar_b >> 20,
                    rebar_loose_b >> 20,
                );
            }
            let [changes, waiting, under_way] = [0, 1, 2].map(|i| {
                crate::watch::REMAPS[i].swap(0, Ordering::Relaxed) as f64
                    / [1.0, 1e3, 1e3][i]
                    / logged_frames
            });
            if changes > 0.0 {
                tracing::info!(
                    "changes of title memory's mapping per frame: {changes:.0}, {waiting:.1} ms of them waiting for reads in place, \
                     {under_way:.1} ms under way (each holds {}; VICTORY_MAPPING_RANGES=0 holds all of title memory)",
                    if crate::watch::by_range() {
                        "the bytes it changes"
                    } else {
                        "all of title memory"
                    }
                );
            }
            let [through, left, filled, short] = [0, 1, 2, 3].map(|i| {
                crate::texture::UNLOCKED[i].swap(0, Ordering::Relaxed) as f64 / logged_frames
            });
            if through + left + filled > 0.0 {
                tracing::info!(
                    "texture refreshes during active remaps, per frame: \
                     {through:.1} read through the kernel, {left:.1} left for the next refresh \
                     (a page was not there), {filled:.1} first fillings from the pages there \
                     (VICTORY_TEXTURE_FILL=0: left empty), {short:.1} of them with less than was \
                     mapped (looked at again)"
                );
            }
            crate::watch::asks_say(logged_frames);
            relay::log_stats(logged_frames);
            crate::lazy_back::log_stats(logged_frames);
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::log(logged_frames);
                tracing::info!(
                    conflicts = crate::shadow::CONFLICTS.swap(0, Ordering::Relaxed),
                    "copy-back conflicts (dwords left to a newer CPU write) since the last line"
                );
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown or the diagnostic probe has no reading. Otherwise diagnostics
    /// builds report whether the sampled pixels were black.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        self.present_overlay_with_compositor(
            texture,
            state,
            None,
            frame,
            PresentCompositor::new,
            #[cfg(all(test, feature = "diagnostics"))]
            |_, _| {},
        )
    }

    /// Presents the base with a premultiplied overlay when both can be sampled.
    pub(crate) fn present_overlay(
        &self,
        texture: &ID3D12Resource,
        state: u32,
        overlay: &ID3D12Resource,
        overlay_state: u32,
        frame: u64,
    ) -> Option<bool> {
        self.present_overlay_with_compositor(
            texture,
            state,
            Some((overlay, overlay_state)),
            frame,
            PresentCompositor::new,
            #[cfg(all(test, feature = "diagnostics"))]
            |_, _| {},
        )
    }

    fn present_overlay_with_compositor(
        &self,
        texture: &ID3D12Resource,
        state: u32,
        overlay: Option<(&ID3D12Resource, u32)>,
        frame: u64,
        create: impl Fn(&ID3D12Device, DXGI_FORMAT) -> windows::core::Result<PresentCompositor>,
        #[cfg(all(test, feature = "diagnostics"))] inspect: impl FnOnce(
            &ID3D12Resource,
            &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        ),
    ) -> Option<bool> {
        #[cfg(feature = "diagnostics")]
        let overlay = overlay.filter(|_| crate::capture::show_spec().is_none());
        let mut overlay = overlay.and_then(|(overlay, state)| {
            let extent = crate::present_compositor::extent(texture, overlay)?;
            Some((overlay, D3D12_RESOURCE_STATES(state as i32), extent))
        });
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        let mut retry = self.present_retry.lock().unwrap_or_else(|e| e.into_inner());
        let now = std::time::Instant::now();
        // After a failure, wait for the backoff: one attempt per state, not
        // one per frame.
        if !retry.due(now) {
            return None;
        }
        // Initialize before choosing the swap-chain extent or opening a list.
        // A failed attempt can still present the base and retry on a later call.
        let mut new_compositor = None;
        if overlay.is_some() && presenter.as_ref().is_none_or(|p| p.compositor.is_none()) {
            let mut compositor_retry = self
                .present_compositor_retry
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if compositor_retry.due(now) {
                match create(&self.device, format) {
                    Ok(compositor) => {
                        new_compositor = Some(compositor);
                        compositor_retry.reset();
                    }
                    Err(error) => {
                        compositor_retry.failed(std::time::Instant::now());
                        if compositor_retry.attempts == 1 {
                            tracing::warn!(%error, "no PresentX compositor; showing the base plane");
                        }
                    }
                }
            }
            if new_compositor.is_none() {
                overlay = None;
            }
        }
        let (width, height) = overlay.map_or((desc.Width as u32, desc.Height), |(_, _, size)| size);
        if presenter.is_none() {
            match self.presenter(width, height, format) {
                Ok(p) => {
                    let attempts = retry.reset();
                    if attempts > 0 {
                        tracing::info!(attempts, "swap chain created after the failed attempts");
                    }
                    *presenter = Some(p);
                }
                Err((step, e)) => {
                    retry.failed(now);
                    if retry.attempts == 1 {
                        tracing::warn!(
                            error = %e,
                            step = step.name(),
                            width,
                            height,
                            format = format.0,
                            "no swap chain; the next attempt waits for the backoff"
                        );
                    } else {
                        tracing::debug!(
                            error = %e,
                            step = step.name(),
                            attempts = retry.attempts,
                            "no swap chain"
                        );
                    }
                    return None;
                }
            }
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => {
                    retry.reset();
                    (p.width, p.height, p.format) = (width, height, format);
                }
                Err(e) => {
                    retry.failed(now);
                    if retry.attempts == 1 {
                        tracing::warn!(
                            error = %e,
                            width,
                            height,
                            "swap chain not resized; the next attempt waits for the backoff"
                        );
                    } else {
                        tracing::debug!(
                            error = %e,
                            attempts = retry.attempts,
                            "swap chain not resized"
                        );
                    }
                    return None;
                }
            }
        }
        if matches!(
            format,
            DXGI_FORMAT_R10G10B10A2_UNORM | DXGI_FORMAT_R16G16B16A16_FLOAT
        ) {
            p.sync_color_space(display_hdr_info().is_some());
        } else {
            p.sync_color_space(false);
        }
        if let Some(compositor) = new_compositor {
            p.compositor = Some(compositor);
            tracing::info!(width, height, "compositing two PresentX planes");
        }
        let composition = overlay.zip(p.compositor.as_ref());
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and live source textures. Lists are
        // waited for before the presenter can reuse its descriptors.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            #[cfg(feature = "diagnostics")]
            let plane_probes = composition
                .filter(|_| crate::capture::plane_stats_wanted(frame))
                .map(|((overlay, overlay_state, _), _)| {
                    [
                        crate::capture::probe_plane(&self.device, list, texture, state),
                        crate::capture::probe_plane(&self.device, list, overlay, overlay_state),
                    ]
                });
            let capture_source =
                if let Some(((overlay, overlay_state, extent), compositor)) = composition {
                    compositor.record(
                        &self.device,
                        list,
                        &back,
                        [(texture, state), (overlay, overlay_state)],
                        extent,
                    );
                    #[cfg(feature = "diagnostics")]
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    &back
                } else {
                    if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                        list.ResourceBarrier(&[transition(
                            texture,
                            state,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        )]);
                    }
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_PRESENT,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )]);
                    // Diagnostics: show a selected title render target instead.
                    #[cfg(feature = "diagnostics")]
                    let shown = crate::capture::show(list, &back);
                    #[cfg(not(feature = "diagnostics"))]
                    let shown = false;
                    if !shown {
                        list.CopyResource(&back, texture);
                    }
                    texture
                };
            #[cfg(feature = "diagnostics")]
            let capture = (cfg!(test) || crate::capture::wanted(frame))
                .then(|| crate::capture::read_back(&self.device, list, capture_source))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = crate::capture::probe(&self.device, list, capture_source);
            #[cfg(not(feature = "diagnostics"))]
            let _ = capture_source;
            let back_state = if composition.is_some() {
                if cfg!(feature = "diagnostics") {
                    D3D12_RESOURCE_STATE_COPY_SOURCE
                } else {
                    D3D12_RESOURCE_STATE_RENDER_TARGET
                }
            } else {
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    list.ResourceBarrier(&[transition(
                        texture,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        state,
                    )]);
                }
                D3D12_RESOURCE_STATE_COPY_DEST
            };
            list.ResourceBarrier(&[transition(&back, back_state, D3D12_RESOURCE_STATE_PRESENT)]);
            p.copy.close();
            self.execute(&[&p.copy]);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                #[cfg(test)]
                inspect(&buffer, &footprint);
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let status = p.swap_chain.Present(0, DXGI_PRESENT(0));
            match present_result(status) {
                Ok(false) => {
                    if p.last_present_issue.replace(status) != Some(status) {
                        tracing::info!(frame, "DXGI reports the window is occluded");
                    }
                    return None;
                }
                Ok(true) => {
                    if p.last_present_issue.take().is_some() {
                        tracing::info!(frame, "DXGI resumed presenting frames");
                    }
                }
                Err(error) => {
                    if self.device.GetDeviceRemovedReason().is_err() {
                        fail(format_args!(
                            "host GPU device removed during Present: {error}{}{}",
                            self.removal_data(&[&p.copy]),
                            last_notes(&[&p.copy])
                        ));
                    }
                    if p.last_present_issue.replace(status) != Some(status) {
                        tracing::warn!(%error, frame, "swap chain present failed");
                    }
                    return None;
                }
            }
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            {
                let composite = probe.as_ref().and_then(crate::capture::grid_samples);
                if let Some([primary, overlay]) = plane_probes {
                    for (name, buffer) in [("primary", primary), ("overlay", overlay)] {
                        crate::capture::report_grid(
                            frame,
                            name,
                            buffer.as_ref().and_then(crate::capture::grid_samples),
                        );
                    }
                    crate::capture::report_grid(frame, "composite", composite);
                }
                composite.map(crate::capture::note_presented)
            }
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<Presenter, (PresentStep, windows::core::Error)> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height).map_err(|e| (PresentStep::Window, e))?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
                .map_err(|e| (PresentStep::Factory, e))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)
                .map_err(|e| (PresentStep::SwapChain, e))?
                .cast::<IDXGISwapChain3>()
                .map_err(|e| (PresentStep::SwapChain, e))?;
            let copy = self.new_list().map_err(|e| (PresentStep::List, e))?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            let mut p = Presenter {
                swap_chain,
                width,
                height,
                format,
                color_space: DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
                copy,
                hwnd,
                meter: Meter::new(),
                last_present_issue: None,
                compositor: None,
            };
            p.sync_color_space(display_hdr_info().is_some());
            Ok(p)
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// Whether the host may make render target views of `texture`. A texture
/// the host refused as a render target is made without that use (see
/// `resource::host_texture`; NVIDIA refuses B4G4R4A4), and a render target
/// view of it is an invalid call that removes the device (0x887A0001).
pub(crate) fn renders_to(texture: &ID3D12Resource) -> bool {
    // SAFETY: a live resource.
    let flags = unsafe { texture.GetDesc() }.Flags;
    flags.0 & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 != 0
}

/// A typed null render target view for a slot whose texture the host
/// cannot render to: draws to it are discarded and the other slots keep
/// their place.
pub(crate) fn null_target() -> D3D12_RENDER_TARGET_VIEW_DESC {
    D3D12_RENDER_TARGET_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
        ..Default::default()
    }
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
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
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// A GPU-only default-heap buffer that shaders can write (unordered
    /// access). Starts in the common state.
    pub(crate) fn default_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        self.default_buffer_in_state(size, D3D12_RESOURCE_STATE_COMMON)
    }

    /// A GPU-only default-heap buffer starting in `UNORDERED_ACCESS` (e.g.
    /// card shadow hash buffers).
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        self.default_buffer_in_state(size, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
    }

    /// A GPU-only default-heap UAV buffer created in `initial_state`.
    fn default_buffer_in_state(
        &self,
        size: u64,
        initial_state: D3D12_RESOURCE_STATES,
    ) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
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
        // SAFETY: plain creation of a default-heap buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                initial_state,
                None,
                &mut buffer,
            )
        };
        #[cfg(feature = "diagnostics")]
        if let Some(b) = &buffer {
            note_vram(2, b);
        }
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no default buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!(
                    "; list {i} of {}, of its {}: {n}",
                    runs.len(),
                    notes.len()
                ));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(
    history: &[D3D12_AUTO_BREADCRUMB_OP],
    markers: &[(u32, String)],
    done: usize,
) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done]
        .iter()
        .rposition(barrier)
        .map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

fn adapter_name(raw: &[u16]) -> String {
    let len = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
    String::from_utf16_lossy(&raw[..len])
}

fn adapter_matches(name: &str, filter: &str) -> bool {
    name.to_ascii_lowercase()
        .contains(&filter.trim().to_ascii_lowercase())
}

#[derive(Debug)]
struct PickedAdapter {
    adapter: IDXGIAdapter1,
    desc: DXGI_ADAPTER_DESC1,
    device: ID3D12Device,
}

/// Picks the adapter that runs the title: the first, in high-performance order (so a
/// hybrid laptop's discrete GPU comes before its integrated one), with shader model 6.6,
/// resource binding tier 3 and mesh shaders; when none has all three, the one that runs
/// most of the title, with a warning naming what each adapter lacks (`crate::adapter`).
/// Returns the live D3D12 device created during the capability probe so the host does
/// not initialize a second device on the chosen adapter, releasing probe devices for
/// candidates that are no longer in contention.
/// When `filter` is given (`VICTORY_ADAPTER`), only adapters whose description contains
/// it (case-insensitively) take part, software ones included.
fn pick_hardware_adapter(filter: Option<&str>) -> windows::core::Result<PickedAdapter> {
    // SAFETY: plain DXGI enumeration and D3D12 capability probes.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let factory6 = factory.cast::<IDXGIFactory6>().ok();
        let mut selector = crate::adapter::Selector::new();
        let mut i = 0u32;
        loop {
            let adapter: IDXGIAdapter1 = match &factory6 {
                Some(f6) => {
                    match f6.EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE) {
                        Ok(a) => a,
                        Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
                        Err(e) => return Err(e),
                    }
                }
                None => match factory.EnumAdapters1(i) {
                    Ok(a) => a,
                    Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
                    Err(e) => return Err(e),
                },
            };
            i += 1;
            let desc = adapter.GetDesc1()?;
            let name = adapter_name(&desc.Description);
            let is_sw = desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
            if let Some(needle) = filter {
                if !adapter_matches(&name, needle) {
                    continue;
                }
            } else if is_sw {
                continue;
            }
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                tracing::info!(adapter = %name, "adapter skipped: no D3D12 feature level 12_0");
                continue;
            }
            let Some(device) = device else { continue };
            let caps = adapter_caps(&device);
            #[cfg(feature = "diagnostics")]
            tracing::debug!(adapter = %name, ?caps, "adapter candidate");
            let picked = PickedAdapter {
                adapter,
                desc,
                device,
            };
            if selector.push(crate::adapter::Candidate { name, caps }, picked) {
                break;
            }
        }
        let Some((picked, shortfall)) = selector.finish() else {
            return Err(windows::core::Error::from_hresult(DXGI_ERROR_NOT_FOUND));
        };
        if let Some(s) = shortfall {
            tracing::warn!(
                adapter = %s.chosen,
                lacks = %s.lacks,
                adapters = %s.adapters,
                "no adapter has everything the title needs; using the one that runs most"
            );
        }
        Ok(picked)
    }
}

/// What `device` offers of the capabilities the title needs.
fn adapter_caps(device: &ID3D12Device) -> crate::adapter::Caps {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    let mut options7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL_6_6,
    };
    // SAFETY: each query writes its own structure of the given size; a failed query leaves
    // the defaults, which read as "not supported".
    let (options_ok, options7_ok, model_ok) = unsafe {
        (
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
                .is_ok(),
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS7,
                    (&raw mut options7).cast(),
                    size_of_val(&options7) as u32,
                )
                .is_ok(),
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_SHADER_MODEL,
                    (&raw mut model).cast(),
                    size_of_val(&model) as u32,
                )
                .is_ok(),
        )
    };
    crate::adapter::Caps {
        shader_model_6_6: model_ok && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0,
        binding_tier_3: options_ok
            && options.ResourceBindingTier.0 >= D3D12_RESOURCE_BINDING_TIER_3.0,
        mesh_shaders: options7_ok
            && options7.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED,
    }
}

#[cfg(test)]
pub(crate) struct DropCounter(std::sync::Arc<std::sync::atomic::AtomicUsize>);

#[cfg(test)]
impl DropCounter {
    pub(crate) fn new() -> (Self, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        (Self(count.clone()), count)
    }
}

#[cfg(test)]
impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// Whether D3D12 wave-ops capabilities allow fixed `WaveSize(32)`.
fn wave32_supported(wave_ops: bool, min: u32, max: u32) -> bool {
    wave_ops && min <= 32 && max >= 32
}

#[cfg(test)]
#[path = "host_tests.rs"]
mod tests;
