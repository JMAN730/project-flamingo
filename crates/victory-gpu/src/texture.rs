//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles every mip covered by the upload.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (from the first mip of a placed 2D view, otherwise every mip and array slice).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8_UNORM;

use crate::host::{HostList, host};
use crate::object::This;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// Monotonic tick stamped on every staging cache hit or insert for LRU eviction.
static USE_TICK: AtomicU64 = AtomicU64::new(1);

fn next_use_tick() -> u64 {
    USE_TICK.fetch_add(1, Ordering::Relaxed)
}

/// Host physical RAM reported by Windows, or `None` if the query fails.
fn total_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: `MEMORYSTATUSEX` is plain data and `dwLength` is initialized
    // before calling `GlobalMemoryStatusEx`.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: valid out pointer with `dwLength` matching `MEMORYSTATUSEX`.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    (ok != 0 && status.ullTotalPhys != 0).then_some(status.ullTotalPhys)
}

/// Upload-heap byte budget for [`STAGING`], scaled to host physical RAM.
fn staging_budget_for_ram(total_ram_bytes: Option<u64>) -> u64 {
    match total_ram_bytes {
        None => 512 * MIB,
        Some(ram) if ram <= 16 * GIB => 256 * MIB,
        Some(ram) if ram <= 32 * GIB => 512 * MIB,
        Some(_) => GIB,
    }
}

/// Cached staging upload-heap byte budget for this host.
fn staging_budget() -> u64 {
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let ram = total_physical_ram();
        let budget = staging_budget_for_ram(ram);
        tracing::info!(
            total_ram_mib = ram.map(|b| b / MIB),
            budget_mib = budget / MIB,
            "texture staging cache budget"
        );
        budget
    })
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    upload: crate::upload_pool::PooledUpload,
    /// Bytes in the host mapping (`GetCopyableFootprints` total).
    size: u64,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// First host subresource and bytes read from the title allocation.
    first: u32,
    tiled_bytes: usize,
    /// Mips per array slice.
    mips: u32,
    /// Top levels of the title's texture the host copy lacks.
    skipped: u32,
    /// Made by the bindless pass over textures picked by index, for one upload
    /// ([`once`]): read when the list is submitted, and not re-checked unless a
    /// descriptor-table draw later samples it.
    once: AtomicBool,
    memory: *const u8,
    /// Whether unchanged view guards can defer hashing this texture.
    guard_view: bool,
    /// Title resource this staging buffer is of: host SRVs are narrowed to
    /// its resident mip window ([`note_resident`]).
    resource: This,
    last: Mutex<Last>,
    /// Monotonic resource creation sequence for epoch-based retirement.
    born_seq: u64,
    /// Monotonic [`USE_TICK`] stamped on every staging cache hit or insert.
    last_used: AtomicU64,
}

/// What a texture's staging buffer holds: the digest of the memory detiled
/// (memory not watched), whether all `tiled_bytes` of `first..` have been
/// filled (separately from the digest so partial mip tails / partial unlocked
/// fills keep refreshing), and the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<(u32, Digest)>,
    /// Bytes of `self.tiled_bytes` actually read into the current staging
    /// snapshot (`0` if never filled or after a short fill).
    filled_bytes: u64,
    complete: bool,
    asked: u64,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// Bounded LRU cache of per-texture upload staging buffers.
///
/// Evicted entries with `Arc::strong_count == 1` are safe to drop directly
/// without going through [`crate::host::retire`]:
/// - `Staging.upload` is only used by [`upload`]/[`upload_from`], which
///   immediately move their `Arc<Staging>` into `list.deferred.copies` via
///   `list.defer_texture(staging)` until `list.reset()` or `list.drop()`.
/// - `Host::execute` holds `&HostList` until the GPU fence completes, so any
///   staging buffer referenced by an open or in-flight command list has
///   `Arc::strong_count >= 2` and is skipped by eviction.
/// - No descriptor heap ever references `Staging.upload`.
#[derive(Default)]
struct StagingCache {
    entries: HashMap<usize, Vec<Arc<Staging>>>,
    total_bytes: u64,
}

impl StagingCache {
    fn insert(&mut self, key: usize, staging: Arc<Staging>) -> Option<Arc<Staging>> {
        self.total_bytes = self.total_bytes.saturating_add(staging.size);
        let ranges = self.entries.entry(key).or_default();
        if let Some(existing) = ranges.iter_mut().find(|s| s.first == staging.first) {
            let previous = std::mem::replace(existing, staging);
            self.total_bytes = self.total_bytes.saturating_sub(previous.size);
            Some(previous)
        } else {
            ranges.push(staging);
            None
        }
    }

    fn remove(&mut self, key: usize) -> Vec<Arc<Staging>> {
        let Some(removed) = self.entries.remove(&key) else {
            return Vec::new();
        };
        for staging in &removed {
            self.total_bytes = self.total_bytes.saturating_sub(staging.size);
        }
        removed
    }

    /// Evicts least-recently-used idle entries (`Arc::strong_count == 1`)
    /// other than `(keep_key, keep_first)` until `self.total_bytes <= budget`
    /// or no idle candidates remain. Callers must drop the returned `Vec`
    /// after releasing the [`STAGING`] lock.
    fn evict_to_budget(
        &mut self,
        budget: u64,
        keep_key: usize,
        keep_first: u32,
    ) -> Vec<Arc<Staging>> {
        if self.total_bytes <= budget {
            return Vec::new();
        }
        let mut candidates: Vec<(u64, usize, u32)> = self
            .entries
            .iter()
            .flat_map(|(&key, ranges)| {
                ranges.iter().filter_map(move |staging| {
                    if (key == keep_key && staging.first == keep_first)
                        || Arc::strong_count(staging) != 1
                    {
                        return None;
                    }
                    Some((
                        staging.last_used.load(Ordering::Relaxed),
                        key,
                        staging.first,
                    ))
                })
            })
            .collect();
        candidates.sort_unstable_by_key(|&(tick, _, _)| tick);
        let mut evicted = Vec::new();
        for (_, key, first) in candidates {
            if self.total_bytes <= budget {
                break;
            }
            let mut remove_key = false;
            if let Some(ranges) = self.entries.get_mut(&key) {
                if let Some(pos) = ranges.iter().position(|s| s.first == first) {
                    let staging = ranges.swap_remove(pos);
                    self.total_bytes = self.total_bytes.saturating_sub(staging.size);
                    evicted.push(staging);
                }
                remove_key = ranges.is_empty();
            }
            if remove_key {
                self.entries.remove(&key);
            }
        }
        evicted
    }
}

static STAGING: Mutex<Option<StagingCache>> = Mutex::new(None);

thread_local! {
    /// Texture memory read for hashing and detiling: one per submitting
    /// thread, as large as the largest texture it read.
    static SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Calls `f` with `size` bytes of this thread's scratch buffer (their contents
/// are whatever the last texture left). Moves the buffer out of the
/// thread-local `RefCell` during `f` so same-thread VEH re-entrancy can never
/// panic with `RefCell already borrowed`. Keeps the initialized length so
/// alternating large and small textures do not re-zero the tail.
fn with_scratch<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    let mut read = SCRATCH.with(|slot| {
        slot.try_borrow_mut()
            .map(|mut b| std::mem::take(&mut *b))
            .unwrap_or_default()
    });
    if size > read.len() {
        read.reserve_exact(size - read.len());
        read.resize(size, 0);
    }
    let out = f(&mut read[..size]);
    SCRATCH.with(|slot| {
        if let Ok(mut saved) = slot.try_borrow_mut()
            && read.capacity() > saved.capacity()
        {
            *saved = read;
        }
    });
    out
}

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let first = {
        let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        set.get_or_insert_with(Default::default)
            .insert(resource as usize)
    };
    if first && cpu_refresh().is_some() {
        let print = title_print(resource);
        WRITTEN_OVER
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(Default::default)
            .insert(resource as usize, print);
    }
    // The host mip chain is complete; drop any CPU-streamed resident window
    // so narrowed SRVs remake to the title's full range (#616).
    note_resident(resource, None);
}

/// `VICTORY_TEXTURE_CPU_REFRESH`: whether a texture the host GPU wrote is
/// checked for later CPU writes to its title memory, and if so whether one
/// found is uploaded over the GPU's contents (`1`) or only logged (`0`).
/// Unset: neither, as before.
fn cpu_refresh() -> Option<bool> {
    static ON: OnceLock<Option<bool>> = OnceLock::new();
    *ON.get_or_init(
        || match std::env::var("VICTORY_TEXTURE_CPU_REFRESH").as_deref() {
            Ok("1") => Some(true),
            Ok("0") => Some(false),
            _ => None,
        },
    )
}

/// Title memory fingerprints of host-written textures as they were when the
/// host GPU first wrote them ([`title_print`]), with
/// `VICTORY_TEXTURE_CPU_REFRESH` set.
static WRITTEN_OVER: Mutex<Option<HashMap<usize, Option<u64>>>> = Mutex::new(None);

/// A sampled fingerprint of `resource`'s title memory ([`fingerprint_raw`]);
/// `None` if it cannot be read.
// shortcut: sampled windows, so a CPU write outside them is not seen.
fn title_print(resource: This) -> Option<u64> {
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let size = crate::resource::texture_layout(&desc)?.planes[0]
        .layout
        .size;
    fingerprint_raw(memory as u64, size, &[])
}

/// What an upload of a texture the host GPU wrote does with
/// `VICTORY_TEXTURE_CPU_REFRESH` (`upload` true: `1`): title memory's
/// fingerprint when the GPU wrote it, `then`, against `now`.
#[derive(Debug, PartialEq, Eq)]
enum AfterGpu {
    /// Title memory as it was (or unreadable): the GPU's contents stay.
    Unchanged,
    /// The CPU wrote it since: uploaded over the GPU's contents.
    Reupload,
    /// The CPU wrote it since, but only logging was asked for.
    KeptGpu,
}

fn after_gpu(then: Option<u64>, now: Option<u64>, upload: bool) -> AfterGpu {
    match (then, now) {
        (Some(then), Some(now)) if then != now => {
            if upload {
                AfterGpu::Reupload
            } else {
                AfterGpu::KeptGpu
            }
        }
        _ => AfterGpu::Unchanged,
    }
}

/// With `VICTORY_TEXTURE_CPU_REFRESH`: whether the host-written `resource`
/// is to be uploaded from title memory again because the CPU wrote it after
/// the GPU did (then it is no longer host-written), logging each texture
/// once for either outcome.
fn cpu_wrote_after_gpu(resource: This) -> bool {
    let Some(upload) = cpu_refresh() else {
        return false;
    };
    let then = WRITTEN_OVER
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|m| m.get(&(resource as usize)).copied())
        .flatten();
    let verdict = after_gpu(then, then.and_then(|_| title_print(resource)), upload);
    static LOGGED: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);
    let once = |what: &str| {
        let new = LOGGED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(Default::default)
            .insert(resource as usize);
        if new {
            tracing::debug!(
                resource = format_args!("{:#x}", resource as usize),
                "texture written by the GPU, then by the CPU: {what}"
            );
        }
    };
    match verdict {
        AfterGpu::Unchanged => false,
        AfterGpu::KeptGpu => {
            once("kept GPU copy");
            false
        }
        AfterGpu::Reupload => {
            once("re-uploaded");
            if let Some(set) = HOST_WRITTEN
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_mut()
            {
                set.remove(&(resource as usize));
            }
            if let Some(m) = WRITTEN_OVER
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_mut()
            {
                m.remove(&(resource as usize));
            }
            true
        }
    }
}

/// Whether the host GPU wrote `resource` (copy/resolve), so its contents are
/// on the host rather than in title memory.
pub(crate) fn is_host_written(resource: This) -> bool {
    HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)))
}

/// One level×layer of an [`alias`] copy: the source and dest subresource
/// indices (they differ when the placed host copy dropped top mips the
/// source kept) and the through-buffer footprints in each format.
struct AliasPart {
    from_sub: u32,
    to_sub: u32,
    written: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    read: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
}

/// The buffer a texture the host writes is copied through into the texture
/// of a view the title placed in its memory in another format
/// ([`alias`]): every level of every layer as each of the two formats lays
/// it out for a copy (the same bytes: the texels are as wide), the written
/// texture's layout first.
struct Through {
    buffer: ID3D12Resource,
    parts: Vec<AliasPart>,
}

/// The most levels times layers [`alias`] copies (a cube of nine levels).
const ALIAS_PARTS: u32 = 54;

/// Source subresource for dest mip `mip` of `layer` when the dest host copy
/// lacks the source's top `dropped` levels (`MipSlice + ArraySlice * MipLevels`).
fn alias_from_sub(mip: u32, layer: u32, dropped: u32, from_mips: u32) -> u32 {
    (mip + dropped) + layer * from_mips
}

// SAFETY: a device-local buffer only lists record copies with.
unsafe impl Send for Through {}
// SAFETY: as above.
unsafe impl Sync for Through {}

/// The copy buffers of [`alias`], by the placed view's texture.
static THROUGH: Mutex<Option<HashMap<usize, Arc<Through>>>> = Mutex::new(None);

/// Brings `placed`, the texture of a view the title placed in the memory
/// of `source` in another format (`crate::resource::alias_of`), up to date
/// with what the host wrote to `source`: the texels of every level and
/// layer are copied as they are, through a buffer (the host copies between
/// two formats of one texel width in no other way). When the placed host
/// copy dropped top mips the source kept (host memory pressure), only the
/// retained levels are copied from the matching source levels. Recorded the
/// first time the list reads the view and again whenever the list used
/// `source` in another way since (wrote it, or read it itself). Leaves
/// `placed` in the copy destination state, as [`upload`] does. False if the
/// copy cannot be made: the caller fills the texture from title memory, as
/// before.
pub(crate) fn alias(list: &HostList, placed: This, source: This) -> bool {
    const TARGET_OR_UNORDERED: u32 = 0x4 | 0x8;
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(to), Some(from)) = (
        crate::resource::host_resource(placed),
        crate::resource::host_resource(source),
    ) else {
        return false;
    };
    let Some(host) = host() else { return false };
    let through = {
        let mut map = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(HashMap::new);
        match map.get(&(placed as usize)) {
            Some(through) => through.clone(),
            None => {
                // SAFETY: live textures.
                let (of_from, of_to) = unsafe { (from.GetDesc(), to.GetDesc()) };
                let layers = |d: &D3D12_RESOURCE_DESC| {
                    if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                        1
                    } else {
                        u32::from(d.DepthOrArraySize)
                    }
                };
                let (from_mips, to_mips) =
                    (u32::from(of_from.MipLevels), u32::from(of_to.MipLevels));
                let (from_layers, to_layers) = (layers(&of_from), layers(&of_to));
                // The placed host may have dropped top mips the source kept;
                // dest level 0 is then source level `dropped`.
                if to_mips == 0
                    || from_mips < to_mips
                    || from_layers != to_layers
                    || from_layers == 0
                {
                    return false;
                }
                let dropped = from_mips - to_mips;
                if dropped > 0
                    && (of_to.Width != (of_from.Width >> dropped).max(1)
                        || of_to.Height != (of_from.Height >> dropped).max(1))
                {
                    return false;
                }
                let parts = to_mips.saturating_mul(to_layers);
                let from_total = from_mips.saturating_mul(from_layers);
                if parts == 0 || parts > ALIAS_PARTS || from_total > ALIAS_PARTS {
                    return false;
                }
                let mut from_rows =
                    vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); from_total as usize];
                let mut to_rows =
                    vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); parts as usize];
                let mut bytes = [0u64; 2];
                // SAFETY: plain queries; the out pointers are ours and
                // hold every subresource layout of each texture.
                unsafe {
                    host.device.GetCopyableFootprints(
                        &of_from,
                        0,
                        from_total,
                        0,
                        Some(from_rows.as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[0]),
                    );
                    host.device.GetCopyableFootprints(
                        &of_to,
                        0,
                        parts,
                        0,
                        Some(to_rows.as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[1]),
                    );
                }
                if bytes[1] == 0 {
                    return false;
                }
                // Pack the through buffer as the dest lays out its levels;
                // pair each with the source level that has the same size.
                let mut paired = Vec::with_capacity(parts as usize);
                for layer in 0..to_layers {
                    for mip in 0..to_mips {
                        let from_sub = alias_from_sub(mip, layer, dropped, from_mips);
                        let to_sub = mip + layer * to_mips;
                        let src = from_rows[from_sub as usize];
                        let dst = to_rows[to_sub as usize];
                        let (fs, fd) = (src.Footprint, dst.Footprint);
                        if (fs.Width, fs.Height, fs.Depth, fs.RowPitch)
                            != (fd.Width, fd.Height, fd.Depth, fd.RowPitch)
                        {
                            return false;
                        }
                        paired.push(AliasPart {
                            from_sub,
                            to_sub,
                            written: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                Offset: dst.Offset,
                                Footprint: fs,
                            },
                            read: dst,
                        });
                    }
                }
                let Some(buffer) = host.default_buffer(bytes[1]) else {
                    return false;
                };
                crate::host::name(&buffer, || {
                    format!(
                        "alias copy {}x{} f{} to f{}",
                        of_from.Width, of_from.Height, of_from.Format.0, of_to.Format.0
                    )
                });
                tracing::info!(
                    view = ?placed,
                    texture = ?source,
                    width = of_from.Width,
                    height = of_from.Height,
                    parts,
                    dropped,
                    written_as = of_from.Format.0,
                    read_as = of_to.Format.0,
                    "a view placed in the memory of a texture the host writes, in another format: read from that texture by a copy on the host"
                );
                let through = Arc::new(Through {
                    buffer,
                    parts: paired,
                });
                map.insert(placed as usize, through.clone());
                through
            }
        }
    };
    let first = list.use_texture(placed as usize, 0);
    // The source's first use in the list records nothing (its prelude
    // brings it to the state); a use in another way since the last copy
    // shows as a change of state.
    let before = list
        .transition(source, COPY_SOURCE)
        .map(|(texture, before)| {
            let state = crate::host::legal_state(texture, COPY_SOURCE);
            let barrier = crate::host::transition(
                texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(state as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
            before
        });
    if !first && before.is_none() {
        return true;
    }
    crate::list::ensure_state(list, placed, COPY_DEST);
    let part = |texture: &ID3D12Resource, sub: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: sub,
        },
    };
    let rows = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&through.buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, read, written) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("alias copy of {source:?} into {placed:?}"));
    // SAFETY: an open list and live resources; the buffer is as big as
    // the footprints and rests in the common state between copies.
    unsafe {
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, common, written)]);
        for p in &through.parts {
            list.list
                .CopyTextureRegion(&rows(&p.written), 0, 0, 0, &part(from, p.from_sub), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, written, read)]);
        for p in &through.parts {
            list.list
                .CopyTextureRegion(&part(to, p.to_sub), 0, 0, 0, &rows(&p.read), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, read, common)]);
    }
    list.keep(&through.buffer);
    // A draw that also writes the source (a view of it handled before
    // this one) needs it in that state again; the next read of the alias
    // then copies once more, as it must after a write.
    if let Some(before) = before
        && before & TARGET_OR_UNORDERED != 0
    {
        crate::list::ensure_state(list, source, before);
    }
    true
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let stagings = {
        let mut cache = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        cache
            .as_mut()
            .map(|cache| cache.remove(resource))
            .unwrap_or_default()
    };
    {
        let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(set) = set.as_mut() {
            set.remove(&resource);
        }
    }
    if let Some(m) = WRITTEN_OVER
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
    {
        m.remove(&resource);
    }
    note_resident(resource as This, None);
    crate::descriptor::host::clear_short_read(resource);
    for s in &stagings {
        crate::descriptor::host::read_all(s.memory as u64);
    }
    if !stagings.is_empty() {
        let born = crate::retire::safe_batch_born_seq(stagings.iter().map(|s| s.born_seq));
        crate::host::retire_batch_born(stagings, born);
    }
    let mut through = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(through) = through.as_mut() {
        through.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

thread_local! {
    /// The thread records the upload of a texture picked by index (bindless
    /// sweep), of which that many bytes were found mapped (`u64::MAX`: not asked).
    pub(crate) static PICKING: Cell<Option<u64>> = const { Cell::new(None) };
}

/// Whether a staging buffer made for one upload of a texture picked by index
/// is filled without asking the write watch and skipped on subsequent bindless
/// sweeps (`VICTORY_TEXTURE_ONCE=0`: checked every time, as before).
fn once() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_ONCE").is_none_or(|v| v != "0"))
}

/// Whether a refresh that finds a mapping change under way and a page missing
/// fills a staging buffer that was never filled from the pages that are there
/// (`VICTORY_TEXTURE_FILL=0`: left as it was made, for the next refresh).
fn fill_unlocked() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_FILL").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line, refreshes that ran while the runtime was
/// remapping title memory: whole reads through the kernel, refreshes left for
/// the next submission because a page was missing, first fillings from the
/// pages there ([`fill_unlocked`]), and of them those that got less than was
/// mapped (looked at again).
pub(crate) static UNLOCKED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A 128-bit digest of `bytes`, to tell a texture's memory changed: four
/// lanes of folded multiplications over 64 bytes at a time. (Not a
/// cryptographic one, which took 0.27 ms a megabyte, four times the detiling
/// it was to spare; nobody makes up textures to collide.)
pub(crate) fn digest(bytes: &[u8]) -> Digest {
    const K: [u64; 8] = [
        0xa076_1d64_78bd_642f,
        0xe703_7ed1_a0b4_28db,
        0x8ebc_6af0_9c88_c6e3,
        0x5899_65cc_7537_4cc3,
        0x1d8e_4e27_c47d_124f,
        0xeb44_acca_b455_d165,
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
    ];
    let fold = |a: u64, b: u64| {
        let p = u128::from(a) * u128::from(b);
        p as u64 ^ (p >> 64) as u64
    };
    let word = |b: &[u8]| {
        let mut w = [0; 8];
        w[..b.len()].copy_from_slice(b);
        u64::from_le_bytes(w)
    };
    let mut lanes = [K[0], K[1], K[2], K[3]];
    let mut step = |block: &[u8; 64]| {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let a = word(&block[16 * i..16 * i + 8]);
            let b = word(&block[16 * i + 8..16 * i + 16]);
            // (What went in is added again: a product with a factor of zero
            // forgets neither the lane nor the block.)
            *lane = fold(a ^ K[4 + i], b ^ *lane)
                .wrapping_add(a.rotate_left(23) ^ lane.rotate_left(41))
                ^ b;
        }
    };
    let (blocks, rest) = bytes.as_chunks::<64>();
    for block in blocks {
        step(block);
    }
    let mut last = [0u8; 64];
    last[..rest.len()].copy_from_slice(rest);
    step(&last);
    let len = bytes.len() as u64;
    let low = fold(lanes[0] ^ K[1], lanes[1] ^ len) ^ fold(lanes[2] ^ K[3], lanes[3] ^ K[5]);
    let high = fold(lanes[0].rotate_left(32) ^ K[6], lanes[2] ^ len)
        ^ fold(lanes[1].rotate_left(32) ^ K[7], lanes[3] ^ K[2]);
    Digest(u128::from(low) << 64 | u128::from(high))
}

/// See [`digest`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Digest(u128);

/// 4 KiB OS-page coverage bitset for `[base_page * 4096, ...)`, recording which
/// pages [`read_there_pages`] actually read so [`mapped_coverage`] can verify
/// that every currently committed page was covered by the read (not merely that
/// the scalar byte counts matched across a concurrent page swap).
struct PageBits {
    base_page: usize,
    num_pages: usize,
    all: bool,
    bits: Vec<u64>,
}

impl PageBits {
    const PAGE: usize = 4096;

    fn new(start: usize, len: usize) -> Self {
        let base_page = start / Self::PAGE;
        let num_pages = if len == 0 {
            0
        } else {
            let end_page = start.saturating_add(len - 1) / Self::PAGE;
            end_page - base_page + 1
        };
        Self {
            base_page,
            num_pages,
            all: false,
            bits: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.all = false;
        self.bits.fill(0);
    }

    fn mark_all(&mut self) {
        self.all = true;
    }

    fn mark_range(&mut self, from: usize, len: usize) {
        if len == 0 || self.all {
            return;
        }
        let words = self.num_pages.div_ceil(64);
        if self.bits.len() < words {
            self.bits.resize(words, 0);
        }
        let first = (from / Self::PAGE).saturating_sub(self.base_page);
        let last = (from.saturating_add(len - 1) / Self::PAGE)
            .saturating_sub(self.base_page)
            .min(self.num_pages.saturating_sub(1));
        for page in first..=last {
            self.bits[page / 64] |= 1u64 << (page % 64);
        }
    }

    fn covers_range(&self, from: usize, to: usize) -> bool {
        if from >= to || self.all {
            return true;
        }
        if self.bits.is_empty() {
            return false;
        }
        let first = (from / Self::PAGE).saturating_sub(self.base_page);
        let last = (to.saturating_sub(1) / Self::PAGE)
            .saturating_sub(self.base_page)
            .min(self.num_pages.saturating_sub(1));
        for page in first..=last {
            if (self.bits[page / 64] & (1u64 << (page % 64))) == 0 {
                return false;
            }
        }
        true
    }
}

/// Reads what is there of the `dst.len()` bytes of title memory at `src`,
/// zeros for the rest; returns how many bytes were there and records every read
/// 4 KiB page in `pages`.
fn read_there_pages(src: *const u8, dst: &mut [u8], pages: &mut PageBits) -> usize {
    if dst.is_empty() {
        pages.mark_all();
        return 0;
    }
    dst.fill(0);
    let (start, len) = (src as usize, dst.len());
    if crate::host::read_into(start as *const u8, dst) {
        pages.mark_all();
        return len;
    }
    // (What failed may have written its first pages.)
    dst.fill(0);
    let mut got = 0;
    crate::host::for_pieces(start, len, 64 << 10, |from, n| {
        let piece = &mut dst[from - start..from - start + n];
        if crate::host::read_into(from as *const u8, piece) {
            pages.mark_range(from, n);
            got += n;
            return;
        }
        piece.fill(0);
        crate::host::for_pieces(from, n, 4 << 10, |from, n| {
            let piece = &mut dst[from - start..from - start + n];
            if crate::host::read_into(from as *const u8, piece) {
                pages.mark_range(from, n);
                got += n;
            } else {
                piece.fill(0);
            }
        });
    });
    got
}

/// Reads what is there of the `dst.len()` bytes of title memory at `src`,
/// zeros for the rest; returns how many bytes were there. Safe while the
/// runtime remaps title memory (the kernel reads): tries the whole span in one
/// go, falling back to 64 KiB and 4 KiB pieces where that fails.
#[cfg(test)]
pub(crate) fn read_there(src: *const u8, dst: &mut [u8]) -> usize {
    let mut pages = PageBits::new(src as usize, dst.len());
    read_there_pages(src, dst, &mut pages)
}

/// Records the full-chain copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    upload_from(list, resource, texture, 0);
}

/// Uploads the levels the placed view can sample, starting at the title's
/// mip `first` (a view's `MostDetailedMip`), from the host mip its host view
/// starts at ([`upload_start`]).
pub(crate) fn upload_from(list: &HostList, resource: This, texture: &ID3D12Resource, first: u32) {
    let first = upload_start(first, crate::resource::dropped_mips(resource));
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too, unless
    // `VICTORY_TEXTURE_CPU_REFRESH=1` finds that it did.
    if host_written && !cpu_wrote_after_gpu(resource) {
        crate::descriptor::host::clear_short_read(resource as usize);
        return;
    }
    let picking = PICKING.get();
    if list.covers_texture(resource as usize, first) {
        // Even when an earlier upload on this list already covers `first` (e.g.
        // a bindless pass recorded mip 0 and a table draw now samples mip 0),
        // record the table use by clearing `once` and preserve the highest
        // mapped-byte baseline without recording a second GPU copy.
        let covering: Vec<Arc<Staging>> = STAGING
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .and_then(|cache| cache.entries.get(&(resource as usize)))
            .map(|ranges| {
                ranges
                    .iter()
                    .filter(|s| s.first <= first)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        for s in &covering {
            if picking.is_none() {
                s.once.store(false, Ordering::Relaxed);
            }
            if fill_unlocked() {
                let there = match picking {
                    Some(there) if there != u64::MAX => there,
                    _ => mapped_bytes(s.memory, s.tiled_bytes),
                };
                list.raise_texture_expected(resource as usize, s.first, there);
            }
        }
        crate::descriptor::host::clear_short_read(resource as usize);
        return;
    }
    let Some(staging) = staging(resource, texture, first) else {
        return;
    };
    if picking.is_none() {
        // A draw samples it through a table: kept and checked from here on.
        staging.once.store(false, Ordering::Relaxed);
    }
    let expected = if fill_unlocked() {
        match picking {
            Some(there) if there != u64::MAX => there,
            _ => mapped_bytes(staging.memory, staging.tiled_bytes),
        }
    } else {
        0
    };
    if !list.use_texture(resource as usize, staging.first) {
        list.raise_texture_expected(resource as usize, staging.first, expected);
        crate::descriptor::host::clear_short_read(resource as usize);
        return;
    }
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        let subresource = staging.first + i as u32;
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe {
                (
                    staging.upload.buffer().GetGPUVirtualAddress(),
                    texture.GetDesc(),
                )
            };
            let len = staging.footprint_bytes(i) as u64;
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {subresource} from {} levels from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(staging.upload.buffer()) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging, expected);
    crate::descriptor::host::clear_short_read(resource as usize);
}

impl Staging {
    /// Title resource pointer (`This as usize`) owning this staging buffer.
    pub(crate) fn resource(&self) -> usize {
        self.resource as usize
    }

    /// First host subresource covered by this staging buffer.
    pub(crate) fn first(&self) -> u32 {
        self.first
    }

    /// Title memory byte range `[start, end)` backing this texture.
    pub(crate) fn memory_range(&self) -> (usize, usize) {
        let start = self.memory as usize;
        (start, start.saturating_add(self.layout.size as usize))
    }

    /// Marks this staging entry as most recently used.
    fn touch(&self) {
        self.last_used.store(next_use_tick(), Ordering::Relaxed);
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self, expected: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now(expected);
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self, expected: u64) -> usize {
        let size = self.tiled_bytes;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        let alone = self.once.load(Ordering::Relaxed);
        if alone && self.guard_view && last.complete {
            return 0;
        }
        if !alone
            && self.guard_view
            && let Some((now, written)) = crate::watch::writes(start, start + size, false)
        {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            last.asked = now;
            last.hash = None;
            last.filled_bytes = size as u64;
            last.complete = true;
            // Write-watched pools are mapped for the whole texture.
            note_resident(self.resource, None);
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            with_tiled_copy(self.memory, size, |tiled| self.detile(tiled));
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        last.asked = 0;
        let Some(_reading) = crate::watch::title_memory_reading_now(start, size) else {
            return self.refresh_unlocked(last, size, expected);
        };
        let resident = committed_prefix(self.memory, size);
        // A mip tail can be selected by its resident prefix only when the
        // smallest mip occupies less than the whole layout. Arrays and 3D
        // textures can have resident slices after a hole; copy their full
        // range through the sparse reader below.
        if resident < size
            && self.layout.bytes_from(self.mips - 1 + self.skipped) < self.layout.size
        {
            let end = self.mips;
            let first = (self.first..end)
                .find(|&mip| self.layout.bytes_from(mip + self.skipped) <= resident as u64);
            let bytes = first.map_or(0, |mip| self.layout.bytes_from(mip + self.skipped) as usize);
            // Host SRVs of this texture are narrowed to the same suffix.
            let window = first.map(|mip| (mip, end - mip));
            note_resident(self.resource, window);
            last.filled_bytes = bytes as u64;
            last.complete = false;
            with_tiled_copy(self.memory, bytes, |tiled| {
                self.changed(last, tiled, first.unwrap_or(end));
            });
            return bytes;
        }
        // All of it is there (or the layout has no separable mip suffix):
        // host views keep the title's full mip window.
        note_resident(self.resource, None);
        // Read through the kernel, as above: the title may unmap the pages
        // (evicting a texture a big table lists) while we read them, and a
        // direct read would fault. The copy is reused, not kept per texture.
        with_scratch(size, |tiled| {
            crate::host::copy_mapped(tiled.as_mut_ptr(), self.memory, size, false);
            if resident >= size {
                crate::descriptor::host::read_all(start as u64);
                last.filled_bytes = size as u64;
                last.complete = true;
            } else {
                last.filled_bytes = resident as u64;
                last.complete = false;
            }
            self.changed(last, tiled, self.first)
        })
    }

    /// A refresh while the runtime is remapping title memory: the texture is
    /// read through the kernel, which fails if a page of it is not there at
    /// this moment. Then a staging buffer that was filled before keeps what it
    /// holds and the texture is looked at again at its next refresh; one never
    /// filled (or only partially filled below `expected`) takes the pages that
    /// are there ([`Staging::fill`]).
    fn refresh_unlocked(&self, last: &mut Last, size: usize, expected: u64) -> usize {
        let read = SCRATCH.with_borrow_mut(|buf| {
            if size > buf.len() {
                buf.reserve_exact(size - buf.len());
                buf.resize(size, 0);
            }
            let slice = &mut buf[..size];
            if !crate::host::read_into(self.memory, slice) {
                return false;
            }
            note_resident(self.resource, None);
            crate::descriptor::host::read_all(self.memory as u64);
            last.filled_bytes = size as u64;
            last.complete = true;
            self.changed(last, slice, self.first);
            true
        });
        if read {
            UNLOCKED[0].fetch_add(1, Ordering::Relaxed);
            size
        } else if (last.hash.is_none() || (!last.complete && last.filled_bytes < expected))
            && fill_unlocked()
        {
            self.fill(last, size, expected)
        } else {
            UNLOCKED[1].fetch_add(1, Ordering::Relaxed);
            last.asked = 0;
            0
        }
    }

    /// The first filling of the staging buffer while the runtime is remapping
    /// title memory, of a texture not all of which can be read: with the pages
    /// that are there, zeros for the others. Pages that are away only for the
    /// moment of a change are read once that change finishes
    /// ([`crate::watch::change_over`]). A snapshot is only accepted (caching
    /// `last.hash` / `last.complete` and calling `read_all`) when the range
    /// read guard is actually held across both the read and `mapped_coverage`,
    /// no concurrent remap bumped `remap_seq`, every currently mapped page was
    /// covered, and at least `there.max(expected)` bytes were read; otherwise
    /// the texture is noted as short so the next eligible bindless pass looks
    /// at it again.
    fn fill(&self, last: &mut Last, size: usize, expected: u64) -> usize {
        const ATTEMPTS: usize = 4;
        let start = self.memory as u64;
        let mut pages = PageBits::new(self.memory as usize, size);
        let (got, verified, complete_hash) = SCRATCH.with_borrow_mut(|buf| {
            if size > buf.len() {
                buf.reserve_exact(size - buf.len());
                buf.resize(size, 0);
            }
            let slice = &mut buf[..size];
            let mut got = 0u64;
            let mut verified = false;
            for attempt in 0..ATTEMPTS {
                // Require the range read guard across both the read and
                // `mapped_coverage` so no active remap can replace a page at
                // the same virtual address before remap notification runs.
                let _reading = crate::watch::title_memory_reading_now(self.memory as usize, size);
                let seq_before = crate::watch::remap_seq();
                pages.reset();
                got = read_there_pages(self.memory, slice, &mut pages) as u64;
                let (there, covered) = mapped_coverage(self.memory, size, &pages);
                let seq_after = crate::watch::remap_seq();
                verified = _reading.is_some()
                    && covered
                    && seq_before == seq_after
                    && got >= there.max(expected);
                if verified {
                    break;
                }
                drop(_reading);
                if attempt + 1 < ATTEMPTS {
                    crate::watch::change_over(self.memory as usize, size);
                }
            }
            if got != 0 {
                self.detile_from(slice, self.first);
            }
            let complete_hash = (verified && got != 0).then(|| digest(slice));
            (got, verified, complete_hash)
        });
        UNLOCKED[2].fetch_add(1, Ordering::Relaxed);
        if !verified {
            UNLOCKED[3].fetch_add(1, Ordering::Relaxed);
            // Keep `last.hash` empty so both `once`-tagged bindless staging
            // and subsequent remap retries look at this texture again.
            last.hash = None;
            last.filled_bytes = 0;
            last.complete = false;
            last.asked = 0;
            crate::descriptor::host::read_short(start, self.resource as usize);
        } else {
            if let Some(hash) = complete_hash {
                last.hash = Some((self.first, hash));
            }
            last.filled_bytes = got;
            last.complete = got == size as u64;
            crate::descriptor::host::read_all(start);
        }
        got as usize
    }

    /// After reading the texture's memory into `tiled`: detiles it unless
    /// its digest is the last one's. Returns the bytes read.
    fn changed(&self, last: &mut Last, tiled: &[u8], first: u32) -> usize {
        let size = tiled.len();
        let hash = digest(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some((first, hash)) {
            return size;
        }
        last.hash = Some((first, hash));
        self.detile_from(tiled, first);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    // D3D12 may omit the padding after the final row of the final footprint.
    fn footprint_bytes(&self, i: usize) -> usize {
        let fp = self.footprints[i];
        let padded = u64::from(fp.Footprint.RowPitch)
            * u64::from(self.rows[i])
            * u64::from(fp.Footprint.Depth);
        padded.min(self.size - fp.Offset) as usize
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        self.detile_from(tiled, self.first);
    }

    fn detile_from(&self, tiled: &[u8], first: u32) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let sub = self.first + i as u32;
            let (mip, slice) = (sub % self.mips, sub / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = self.footprint_bytes(i);
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(
                    self.upload.data().add(footprint.Offset as usize),
                    len,
                )
            };
            if sub < first {
                linear.fill(0);
                continue;
            }
            if let Err(e) = self.layout.detile(
                tiled,
                mip + self.skipped,
                slice,
                linear,
                pitch,
                pitch * rows as usize,
            ) {
                tracing::warn!(
                    ?e,
                    mip,
                    slice,
                    skipped = self.skipped,
                    "texture not detiled"
                );
                return;
            }
        }
    }
}

/// Reads a texture for the duration of its detile.
fn with_tiled_copy(memory: *const u8, size: usize, detile: impl FnOnce(&[u8])) {
    with_scratch(size, |tiled| {
        crate::host::copy_mapped(tiled.as_mut_ptr(), memory, size, false);
        detile(tiled);
    });
}

/// How many bytes of `[at, at + len)` are committed memory, and whether every
/// committed page in that span was covered by `read`.
fn mapped_coverage(at: *const u8, len: usize, read: &PageBits) -> (u64, bool) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    if len == 0 {
        return (0, true);
    }
    let mut p = at as usize;
    let end = p.saturating_add(len);
    let mut total = 0u64;
    let mut covered = true;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.RegionSize == 0 {
            break;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            break;
        }
        let to = next.min(end);
        if info.State == MEM_COMMIT {
            total += (to - p) as u64;
            if !read.covers_range(p, to) {
                covered = false;
            }
        }
        p = to;
    }
    (total, covered)
}

/// How many bytes of `[at, at + len)` are committed memory.
pub(crate) fn mapped_bytes(at: *const u8, len: usize) -> u64 {
    let mut all = PageBits::new(at as usize, len);
    all.mark_all();
    mapped_coverage(at, len, &all).0
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    // A range past the top of the address space is never all committed.
    if (at as usize).checked_add(len).is_none() {
        return false;
    }
    committed_prefix(at, len) == len
}

/// Whether residency probes (`committed`, `committed_prefix`) fall back to
/// `VirtualQuery` instead of batched `K32QueryWorkingSetEx`
/// (`VICTORY_COMMITTED_QUERY=1` restores legacy `VirtualQuery`).
fn by_query() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMMITTED_QUERY").is_some_and(|v| v == "1"))
}

fn committed_prefix(at: *const u8, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    if by_query() {
        return committed_prefix_by_query(at, len);
    }
    readable_prefix(at as usize, len)
}

/// Contiguous readable prefix length of `[start, start.saturating_add(len))`
/// using batched 256-page `K32QueryWorkingSetEx` syscalls (with
/// [`crate::host::read_into_probe`] fallback for pages not yet in the working
/// set), avoiding `VirtualQuery`'s full-region page-table walks.
fn readable_prefix(start: usize, len: usize) -> usize {
    use windows_sys::Win32::System::ProcessStatus::{
        K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const KERNEL_PAGE: usize = 4096;
    let end = start.saturating_add(len);
    if start >= end {
        return 0;
    }
    let first = start / KERNEL_PAGE;
    let last = (end - 1) / KERNEL_PAGE;
    // SAFETY: plain data filled by `K32QueryWorkingSetEx`.
    let mut info: [PSAPI_WORKING_SET_EX_INFORMATION; 256] = unsafe { std::mem::zeroed() };
    let mut page = first;
    while page <= last {
        let n = (last - page + 1).min(info.len());
        for (i, entry) in info[..n].iter_mut().enumerate() {
            entry.VirtualAddress = ((page + i) * KERNEL_PAGE) as *mut _;
        }
        // SAFETY: `n` entries of the given size; our own process.
        let asked = unsafe {
            K32QueryWorkingSetEx(
                GetCurrentProcess(),
                info.as_mut_ptr().cast(),
                (n * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
            )
        } != 0;
        for (i, entry) in info[..n].iter().enumerate() {
            // Bit 0 of the answer: the page is in the working set.
            // SAFETY: plain data filled by the query.
            let valid = asked && unsafe { entry.VirtualAttributes.Flags } & 1 != 0;
            let page_addr = (page + i) * KERNEL_PAGE;
            let mut byte = [0u8];
            if !(valid || crate::host::read_into_probe(page_addr as *const u8, &mut byte)) {
                return page_addr.saturating_sub(start);
            }
        }
        page += n;
    }
    end - start
}

fn committed_prefix_by_query(at: *const u8, len: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = at as usize;
    // A range past the top of the address space (a title address and size
    // taken as they come) is cut at the top, so it is never all committed.
    let end = p.saturating_add(len);
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0
        {
            break;
        }
        p = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    }
    p.min(end) - at as usize
}

/// Narrows a title view's mip window to the resident ones; see
/// [`victory_formats::texture::narrowed_view`].
pub(crate) fn narrowed(asked: (u32, u32), resident: (u32, u32), mips: u32) -> (u32, u32) {
    victory_formats::texture::narrowed_view(asked, resident, mips)
}

/// Last-known resident mip window of a title texture, set by its staging
/// refresh. `None` while it is fully mapped (or never partly streamed).
pub(crate) fn resident(resource: This) -> Option<(u32, u32)> {
    let map = RESIDENT.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref()?.get(&(resource as usize)).copied()
}

/// Notes the resident mip window of `resource` (`None` when fully mapped).
/// Host SRVs are narrowed to it while it stands and remade when it widens.
pub(crate) fn note_resident(resource: This, window: Option<(u32, u32)>) {
    let mut map = RESIDENT.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    match window {
        Some(w) => {
            map.insert(resource as usize, w);
        }
        None => {
            map.remove(&(resource as usize));
        }
    }
}

/// Live resident mip window of `resource` from its mapped prefix, for the
/// first host view before a staging refresh has noted one. `None` when fully
/// mapped or the layout cannot express a mip suffix.
pub(crate) fn live_resident(resource: This) -> Option<(u32, u32)> {
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let layout = crate::resource::texture_layout(&desc)?.planes[0]
        .layout
        .clone();
    let committed = committed_prefix(memory, layout.size as usize);
    layout.resident_from_prefix(committed)
}

static RESIDENT: Mutex<Option<HashMap<usize, (u32, u32)>>> = Mutex::new(None);

/// The host mip a view from title mip `title_first` starts at, on a host
/// copy that lacks the title's top `dropped` levels: where its host view
/// starts (`crate::resource::rebase_mip_range`), so the upload covers every
/// level that view can sample.
fn upload_start(title_first: u32, dropped: u32) -> u32 {
    crate::resource::rebase_mip_range(title_first, u32::MAX, dropped).0
}

/// The first host subresource to stage for host mip `requested` of a host
/// copy of `mips` levels that lacks the title's top `skipped` (host mip `m`
/// is the title's `m + skipped` in `layout`): it, if the title's bytes from
/// there are a proper suffix of the layout, else the whole chain.
fn first_mip(
    layout: &victory_formats::texture::Layout,
    mips: u32,
    skipped: u32,
    requested: u32,
) -> u32 {
    if requested < mips && layout.bytes_from(requested + skipped) < layout.size {
        requested
    } else {
        0
    }
}

/// Allocates a fresh staging buffer for a title texture without inserting it
/// into [`STAGING`].
fn create_staging(
    resource: This,
    texture: &ID3D12Resource,
    requested: u32,
) -> Option<Arc<Staging>> {
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    // Software-decoded movie planes are linear R8 textures. The decoder can
    // write through another view of the same pages, bypassing the guards on
    // the sampled view. Check their content on every submission so cycling
    // through the movie's textures does not replay old frames.
    let guard_view = !(d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.Format == DXGI_FORMAT_R8_UNORM
        && d.MipLevels == 1
        && d.DepthOrArraySize == 1
        && layout.desc.swizzle == victory_formats::texture::Swizzle::LINEAR);
    // Top levels the host copy lost: host mip `m` is the title's `m + skipped`.
    let skipped = crate::resource::dropped_mips(resource);
    let first = first_mip(&layout, u32::from(d.MipLevels), skipped, requested);
    let tiled_bytes = layout.bytes_from(first + skipped) as usize;
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    } - first;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            first,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let upload = crate::upload_pool::acquire(host, total)?;
    crate::host::name(upload.buffer(), || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(
        ?desc,
        total,
        skipped,
        "texture contents uploaded from title memory"
    );
    Some(Arc::new(Staging {
        upload,
        size: total,
        first,
        tiled_bytes,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        skipped,
        once: AtomicBool::new(once() && guard_view && PICKING.get().is_some()),
        memory,
        guard_view,
        resource,
        last: Mutex::default(),
        born_seq: crate::host::birth_seq(),
        last_used: AtomicU64::new(next_use_tick()),
    }))
}

/// The staging state of a title texture, made on first use (or after LRU
/// eviction).
fn staging(resource: This, texture: &ID3D12Resource, requested: u32) -> Option<Arc<Staging>> {
    let key = resource as usize;
    let (s, replaced, evicted) = {
        let mut cache = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        let cache = cache.get_or_insert_with(StagingCache::default);
        if let Some(s) = cache.entries.get(&key).and_then(|ranges| {
            let cached = ranges.first()?;
            let first = first_mip(&cached.layout, cached.mips, cached.skipped, requested);
            ranges.iter().find(|s| s.first == first)
        }) {
            s.touch();
            return Some(s.clone());
        }
        let s = create_staging(resource, texture, requested)?;
        let keep_first = s.first;
        let replaced = cache.insert(key, s.clone());
        let evicted = cache.evict_to_budget(staging_budget(), key, keep_first);
        (s, replaced, evicted)
    };
    // Drop replaced/evicted staging buffers after releasing `STAGING` so freeing
    // D3D12 upload resources does not stall concurrent cache lookups.
    drop((replaced, evicted));
    Some(s)
}

/// Fast sampled fingerprint of title memory `[at, at + len)` plus optional
/// `mips` offsets, used by `host::fill_ahead` to detect CPU rewrites since a
/// fill read ahead of in-flight GPU writes. Moves the TLS scratch `Vec` out
/// across reads so VEH re-entrancy cannot panic on `RefCell`.
fn fingerprint_inner(at: u64, len: u64, mips: &[(u64, u64)], raw: bool) -> Option<u64> {
    const WINDOW: usize = 16 << 10;
    thread_local! {
        static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let len = len as usize;
    let mut print = 0u128;
    let mut read = READ.with(|slot| {
        slot.try_borrow_mut()
            .map(|mut b| std::mem::take(&mut *b))
            .unwrap_or_default()
    });
    let longest = mips.iter().map(|&(_, n)| n as usize).max().unwrap_or(0);
    read.resize((3 * WINDOW).max(longest), 0);
    let mut ok = true;
    let mut take = |from: usize, bytes: usize| {
        if !ok {
            return;
        }
        read[..bytes].fill(0);
        if raw {
            if !crate::host::copy_mapped_raw_checked(
                read.as_mut_ptr(),
                (at as usize + from) as *const u8,
                bytes,
                false,
            ) {
                ok = false;
                return;
            }
        } else {
            crate::host::copy_mapped(
                read.as_mut_ptr(),
                (at as usize + from) as *const u8,
                bytes,
                false,
            );
        }
        let h = blake3::hash(&read[..bytes]);
        let b = h.as_bytes();
        let d = u128::from_le_bytes(b[..16].try_into().unwrap_or([0; 16]));
        print = print.rotate_left(37) ^ d;
    };
    if len <= 3 * WINDOW {
        take(0, len);
    } else {
        take(0, WINDOW);
        take((len / 2) & !0xfff, WINDOW);
        take(len - WINDOW, WINDOW);
        for &(from, bytes) in mips {
            let from = from as usize;
            if from < len {
                take(from, (bytes as usize).min(len - from));
            }
        }
    }
    READ.with(|slot| {
        if let Ok(mut saved) = slot.try_borrow_mut()
            && read.capacity() > saved.capacity()
        {
            *saved = read;
        }
    });
    if !ok {
        return None;
    }
    let first = (print as u64 ^ (print >> 64) as u64).to_le_bytes();
    Some(u64::from_le_bytes(first) | 1)
}

pub(crate) fn fingerprint(at: u64, len: u64, mips: &[(u64, u64)]) -> u64 {
    fingerprint_inner(at, len, mips, false).unwrap_or(1)
}

/// Non-materializing [`fingerprint`]: returns `None` if any sampled window
/// cannot be read raw (safe to call while holding per-`Shadow` locks).
pub(crate) fn fingerprint_raw(at: u64, len: u64, mips: &[(u64, u64)]) -> Option<u64> {
    fingerprint_inner(at, len, mips, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A texture the GPU wrote is uploaded again only when its title memory
    /// changed since, and only with uploading asked for; one whose memory
    /// could not be read either time keeps the GPU's contents.
    #[test]
    fn a_cpu_write_after_the_gpus_is_uploaded_only_when_asked() {
        assert_eq!(after_gpu(Some(3), Some(5), true), AfterGpu::Reupload);
        assert_eq!(after_gpu(Some(3), Some(5), false), AfterGpu::KeptGpu);
        assert_eq!(after_gpu(Some(3), Some(3), true), AfterGpu::Unchanged);
        assert_eq!(after_gpu(None, Some(5), true), AfterGpu::Unchanged);
        assert_eq!(after_gpu(Some(3), None, true), AfterGpu::Unchanged);
    }

    /// With top levels dropped, a placed view from title mip `n` uploads
    /// from the host mip its host view starts at, `n - dropped` (none below
    /// 0), and staging of that host mip reads the title's bytes from
    /// `n`: eight title levels, two dropped, a view at title mip 2 starts at
    /// host mip 0 and stages the whole host chain.
    #[test]
    fn a_placed_view_uploads_from_the_host_mip_its_host_view_starts_at() {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        assert_eq!(upload_start(2, 2), 0);
        assert_eq!(upload_start(5, 2), 3);
        assert_eq!(upload_start(1, 2), 0, "a view of dropped levels only");
        assert_eq!(upload_start(4, 0), 4);
        for (title, dropped) in [(2, 2), (5, 2), (1, 2), (4, 0)] {
            assert_eq!(
                upload_start(title, dropped),
                crate::resource::rebase_mip_range(title, 3, dropped).0,
                "the upload starts where the host view does"
            );
        }
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 128,
            height: 128,
            depth: 1,
            mips: 8,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        })
        .expect("a layout");
        let host_mips = 6;
        // Host mip 0 is the title's 2: the whole host chain, staged whole.
        assert_eq!(first_mip(&layout, host_mips, 2, upload_start(2, 2)), 0);
        // Host mip 3 is the title's 5: a suffix of the title's bytes.
        assert!(layout.bytes_from(5) < layout.size);
        assert_eq!(first_mip(&layout, host_mips, 2, upload_start(5, 2)), 3);
        // Past the host chain: the whole chain.
        assert_eq!(first_mip(&layout, host_mips, 2, host_mips), 0);
    }

    #[test]
    fn a_range_past_the_address_space_is_not_committed() {
        let data = [0u8; 64];
        assert!(committed(data.as_ptr(), data.len()));
        assert!(!committed(data.as_ptr(), usize::MAX));
        assert!(!committed(usize::MAX as *const u8, 2));
    }

    /// What is there of a range some of whose pages are not: those bytes,
    /// counted, and zeros for the rest, whatever the buffer held.
    #[test]
    fn what_is_there_is_read_and_the_rest_is_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const SIZE: usize = 0x8_0000;
        // SAFETY: a reservation of ours; only committed pages are written.
        unsafe {
            let at = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!at.is_null());
            // 0-0x1_1000 and 0x3_0000-0x4_0000 are there.
            for (from, len) in [(0, 0x1_1000), (0x3_0000, 0x1_0000)] {
                assert!(
                    !VirtualAlloc(at.add(from).cast(), len, MEM_COMMIT, PAGE_READWRITE).is_null()
                );
                at.add(from).write_bytes(0x5a, len);
            }
            // From an address inside a page, as textures are packed.
            let mut out = vec![0x77u8; SIZE - 0x300];
            assert_eq!(
                super::read_there(at.add(0x300), &mut out),
                0x1_1000 - 0x300 + 0x1_0000
            );
            for (i, &b) in out.iter().enumerate() {
                let from = i + 0x300;
                let there = from < 0x1_1000 || (0x3_0000..0x4_0000).contains(&from);
                assert_eq!(b, if there { 0x5a } else { 0 }, "byte {from:#x}");
            }
            // All there: one read.
            let mut out = vec![0x77u8; 0x1_0000];
            assert_eq!(super::read_there(at.add(0x1000), &mut out), 0x1_0000);
            assert!(out.iter().all(|&b| b == 0x5a));
            // Nothing there.
            let mut out = vec![0x77u8; 0x1_0000];
            assert_eq!(super::read_there(at.add(0x5_0000), &mut out), 0);
            assert!(out.iter().all(|&b| b == 0));
            assert_ne!(VirtualFree(at.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// The digest tells every length and every changed byte apart, zeros
    /// among them.
    #[test]
    fn a_digest_tells_bytes_apart() {
        use super::digest;
        let zeros = [0u8; 400];
        let mut seen: Vec<super::Digest> = (0..400).map(|n| digest(&zeros[..n])).collect();
        for at in 0..400 {
            for value in [1u8, 0x80, 0xff] {
                let mut bytes = zeros;
                bytes[at] = value;
                seen.push(digest(&bytes));
            }
        }
        let text: Vec<u8> = (0..400u32).map(|i| ((i * 37) >> 2) as u8).collect();
        for at in 0..399 {
            let mut bytes = text.clone();
            bytes.swap(at, at + 1);
            if bytes != text {
                seen.push(digest(&bytes));
            }
        }
        seen.push(digest(&text));
        let n = seen.len();
        seen.sort_unstable_by_key(|d| d.0);
        seen.dedup();
        assert_eq!(seen.len(), n);
        assert_eq!(digest(&text), digest(&text.clone()));
    }

    #[test]
    fn texture_once_skips_repeat_refresh_until_table_upload() {
        let Some(host) = host() else {
            return;
        };
        let (resource, texture) = test_texture(64, 1, 1, None);
        let list = host.new_list().expect("host list");
        PICKING.set(Some(u64::MAX));
        upload_from(&list, resource, texture, 0);
        PICKING.set(None);

        let s = staging(resource, texture, 0).expect("staging");
        assert_eq!(s.once.load(Ordering::Relaxed), once());
        let first_read = s.refresh_now(0);
        assert!(first_read > 0);
        let second_read = s.refresh_now(0);
        if once() {
            assert_eq!(
                second_read, 0,
                "VICTORY_TEXTURE_ONCE must skip repeat refresh of once-tagged bindless staging"
            );
        }

        // A subsequent descriptor-table upload clears `once` so future refreshes run.
        let list2 = host.new_list().expect("host list 2");
        upload_from(&list2, resource, texture, 0);
        assert!(!s.once.load(Ordering::Relaxed));
        assert!(s.refresh_now(0) > 0);

        drop(s);
        list.close();
        list.reset();
        list2.close();
        list2.reset();
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(resource, 0x10);
        assert_eq!(release(resource), 0);
    }

    #[test]
    fn table_upload_clears_once_even_when_list_already_covers_texture() {
        let Some(host) = host() else {
            return;
        };
        let (resource, texture) = test_texture(64, 1, 1, None);
        let list = host.new_list().expect("host list");
        PICKING.set(Some(u64::MAX));
        upload_from(&list, resource, texture, 0);
        PICKING.set(None);

        let s = staging(resource, texture, 0).expect("staging");
        assert_eq!(s.once.load(Ordering::Relaxed), once());
        assert!(list.covers_texture(resource as usize, 0));

        // A table draw sampling the same texture on the same list hits
        // `covers_texture`, must clear `once`, and must not record a second copy.
        upload_from(&list, resource, texture, 0);
        assert!(
            !s.once.load(Ordering::Relaxed),
            "table upload on an already-covered list must clear bindless-only `once`"
        );
        assert!(s.refresh_now(0) > 0);
        assert!(s.refresh_now(0) > 0);

        drop(s);
        list.close();
        list.reset();
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(resource, 0x10);
        assert_eq!(release(resource), 0);
    }

    #[test]
    fn fill_unlocked_handles_partial_pages_during_active_remap() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        if host().is_none() {
            return;
        }
        const RESERVED: usize = 2 << 20;
        // SAFETY: reserve space and commit only the first 64 KiB page run.
        let memory =
            unsafe { VirtualAlloc(std::ptr::null(), RESERVED, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!memory.is_null());
        // SAFETY: commit the first 64 KiB within our reservation.
        let mapped = unsafe { VirtualAlloc(memory, 64 << 10, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(mapped, memory);
        // SAFETY: the committed 64 KiB is writable.
        unsafe { memory.cast::<u8>().write_bytes(0x5a, 64 << 10) };

        let (resource, texture) = test_texture(256, 1, 1, Some(memory as u64));
        let s = staging(resource, texture, 0).expect("staging");
        assert!(s.tiled_bytes > (64 << 10));

        // Hold an active mapping change over the texture's memory range so
        // `title_memory_reading_now` returns `None` and exercises `refresh_unlocked`.
        crate::descriptor::host::read_all(memory as u64);
        let _remap = crate::watch::title_memory_remapping(memory as usize, s.tiled_bytes);
        let read = s.refresh_now(64 << 10);
        if fill_unlocked() {
            assert_eq!(
                read,
                64 << 10,
                "VICTORY_TEXTURE_FILL must report the bytes actually read on a partial fill"
            );
            let last = s.last.lock().unwrap();
            assert!(
                last.hash.is_none(),
                "a fill while a remap is still active must not cache last.hash"
            );
            assert!(
                !last.complete,
                "a partial fill with got < size must not mark staging complete"
            );
            drop(last);
            assert!(
                crate::descriptor::host::is_short_read_pending(resource as usize),
                "a fill while a remap is still active must leave a pending short-read retry"
            );
            // SAFETY: staging mapping covers the first subresource footprint.
            let staged =
                unsafe { std::slice::from_raw_parts(s.upload.data(), s.footprint_bytes(0)) };
            assert!(staged.contains(&0x5a));
        } else {
            assert_eq!(
                read, 0,
                "VICTORY_TEXTURE_FILL=0 must leave an unfilled staging buffer empty during remap"
            );
            let last = s.last.lock().unwrap();
            assert!(last.hash.is_none());
            assert!(!last.complete);
        }
        drop(_remap);
        crate::descriptor::host::read_all(memory as u64);

        drop(s);
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(resource, 0x10);
        assert_eq!(release(resource), 0);
        // SAFETY: freeing our reservation.
        assert_ne!(unsafe { VirtualFree(memory, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn fill_requires_read_guard_against_same_va_replacement() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE,
            VirtualAlloc, VirtualFree,
        };
        if host().is_none() || !fill_unlocked() {
            return;
        }
        const RESERVED: usize = 2 << 20;
        const PART: usize = 64 << 10;
        // SAFETY: reserve space and commit the first 64 KiB with initial bytes `0x5a`.
        let memory =
            unsafe { VirtualAlloc(std::ptr::null(), RESERVED, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!memory.is_null());
        // SAFETY: commit the first 64 KiB within our reservation.
        let mapped = unsafe { VirtualAlloc(memory, PART, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(mapped, memory);
        // SAFETY: committed prefix is writable.
        unsafe { memory.cast::<u8>().write_bytes(0x5a, PART) };

        let (resource, texture) = test_texture(256, 1, 1, Some(memory as u64));
        let s = staging(resource, texture, 0).expect("staging");
        assert!(s.tiled_bytes > PART);
        crate::descriptor::host::read_all(memory as u64);

        // Hold an active remap guard over the staging range: `title_memory_reading_now`
        // returns `None`, so `fill` must not accept the snapshot even though `got ==
        // there == expected` and every committed VA is covered.
        let remap = crate::watch::title_memory_remapping(memory as usize, s.tiled_bytes);
        let read_during_remap = s.refresh_now(PART as u64);
        assert_eq!(read_during_remap, PART);
        {
            let last = s.last.lock().unwrap();
            assert!(
                last.hash.is_none(),
                "fill during an active remap must not cache last.hash"
            );
            assert_eq!(last.filled_bytes, 0);
            assert!(!last.complete);
        }
        assert!(
            crate::descriptor::host::is_short_read_pending(resource as usize),
            "fill during an active remap must keep the short-read notice pending instead of calling read_all"
        );

        // Replace the committed page at the exact same virtual address while the
        // remap guard is still held (before any remap notification runs).
        // SAFETY: decommit and re-commit the same 64 KiB virtual address range.
        assert_ne!(unsafe { VirtualFree(memory, PART, MEM_DECOMMIT) }, 0);
        // SAFETY: re-committing the same VA within our reservation.
        let remapped = unsafe { VirtualAlloc(memory, PART, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(remapped, memory);
        // SAFETY: re-committed prefix is writable.
        unsafe { memory.cast::<u8>().write_bytes(0xa5, PART) };

        // While the remap guard is still active, another direct `fill` must still
        // refuse to cache the snapshot or call `read_all`.
        {
            let mut last = s.last.lock().unwrap();
            assert_eq!(s.fill(&mut last, s.tiled_bytes, PART as u64), PART);
            assert!(last.hash.is_none());
            assert_eq!(last.filled_bytes, 0);
            assert!(!last.complete);
        }
        assert!(
            crate::descriptor::host::is_short_read_pending(resource as usize),
            "pending short-read notice must remain set while remap guard is held"
        );

        // Once the remap guard drops, `fill` acquires `title_memory_reading_now`,
        // accepts the fresh `0xa5` snapshot, caches its digest, and calls `read_all`.
        drop(remap);
        let mut expected_buf = vec![0u8; s.tiled_bytes];
        expected_buf[..PART].fill(0xa5);
        let expected_fresh_digest = digest(&expected_buf);
        let mut stale_buf = vec![0u8; s.tiled_bytes];
        stale_buf[..PART].fill(0x5a);
        assert_ne!(expected_fresh_digest, digest(&stale_buf));

        {
            let mut last = s.last.lock().unwrap();
            assert_eq!(s.fill(&mut last, s.tiled_bytes, PART as u64), PART);
            assert_eq!(last.hash, Some((0, expected_fresh_digest)));
            assert_eq!(last.filled_bytes, PART as u64);
            assert!(!last.complete);
        }
        assert!(
            !crate::descriptor::host::is_short_read_pending(resource as usize),
            "verified fill after remap completes must clear pending short-read state via read_all"
        );
        // SAFETY: staging mapping covers the first subresource footprint.
        let staged = unsafe { std::slice::from_raw_parts(s.upload.data(), s.footprint_bytes(0)) };
        assert!(staged.contains(&0xa5));
        assert!(!staged.contains(&0x5a));

        // Commit the remaining bytes and verify a full `fill` marks `last.complete`.
        let rest_len = s.tiled_bytes - PART;
        // SAFETY: committing remaining pages within our reservation.
        let rest = unsafe {
            VirtualAlloc(
                memory.cast::<u8>().add(PART).cast(),
                rest_len,
                MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!rest.is_null());
        // SAFETY: newly committed tail is writable.
        unsafe { memory.cast::<u8>().add(PART).write_bytes(0xa5, rest_len) };
        let full_digest = digest(&vec![0xa5u8; s.tiled_bytes]);
        {
            let mut last = s.last.lock().unwrap();
            assert_eq!(
                s.fill(&mut last, s.tiled_bytes, s.tiled_bytes as u64),
                s.tiled_bytes
            );
            assert_eq!(last.hash, Some((0, full_digest)));
            assert_eq!(last.filled_bytes, s.tiled_bytes as u64);
            assert!(last.complete);
        }

        crate::descriptor::host::read_all(memory as u64);
        drop(s);
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(resource, 0x10);
        assert_eq!(release(resource), 0);
        // SAFETY: freeing our reservation.
        assert_ne!(unsafe { VirtualFree(memory, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn short_fill_on_once_staging_retries_once_remaining_pages_arrive() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        let Some(host) = host() else {
            return;
        };
        const RESERVED: usize = 2 << 20;
        // SAFETY: reserve space and commit only the first 64 KiB page run.
        let memory =
            unsafe { VirtualAlloc(std::ptr::null(), RESERVED, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!memory.is_null());
        // SAFETY: commit the first 64 KiB within our reservation.
        let mapped = unsafe { VirtualAlloc(memory, 64 << 10, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(mapped, memory);
        // SAFETY: the committed 64 KiB is writable.
        unsafe { memory.cast::<u8>().write_bytes(0x5a, 64 << 10) };

        let (resource, texture) = test_texture(256, 1, 1, Some(memory as u64));
        let list = host.new_list().expect("host list");
        PICKING.set(Some(u64::MAX));
        upload_from(&list, resource, texture, 0);
        PICKING.set(None);

        let s = staging(resource, texture, 0).expect("staging");
        assert!(s.tiled_bytes > (64 << 10));
        assert_eq!(s.once.load(Ordering::Relaxed), once());
        let expected_full = s.tiled_bytes as u64;
        crate::descriptor::host::read_all(memory as u64);

        let _remap = crate::watch::title_memory_remapping(memory as usize, s.tiled_bytes);
        let short_read = s.refresh_now(expected_full);
        if fill_unlocked() {
            assert_eq!(short_read, 64 << 10);
            assert!(
                s.last.lock().unwrap().hash.is_none(),
                "short fill must leave last.hash empty so once-tagged staging retries"
            );
            assert!(
                crate::descriptor::host::is_short_read_pending(resource as usize),
                "short fill must record a pending short read for this resource"
            );
            assert!(crate::descriptor::host::clear_short_read(resource as usize));
            // A second pass while still remapping and short must also invoke fill(),
            // not bail out as already hashed.
            assert_eq!(s.refresh_now(expected_full), 64 << 10);
            assert!(s.last.lock().unwrap().hash.is_none());
            assert!(crate::descriptor::host::clear_short_read(resource as usize));
        } else {
            assert_eq!(short_read, 0);
            assert!(s.last.lock().unwrap().hash.is_none());
        }
        drop(_remap);

        // Now commit the remaining pages and verify the `once`-tagged staging
        // retries cleanly, fills the tail, and caches its final digest.
        let rest_len = s.tiled_bytes - (64 << 10);
        // SAFETY: committing the remaining pages of our reservation.
        let rest = unsafe {
            VirtualAlloc(
                memory.cast::<u8>().add(64 << 10).cast(),
                rest_len,
                MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!rest.is_null());
        // SAFETY: the newly committed tail is writable.
        unsafe {
            memory
                .cast::<u8>()
                .add(64 << 10)
                .write_bytes(0x7c, rest_len)
        };

        let retry_read = s.refresh_now(expected_full);
        assert_eq!(
            retry_read, s.tiled_bytes,
            "once-tagged staging must retry after a short fill once pages arrive"
        );
        {
            let last = s.last.lock().unwrap();
            assert!(last.hash.is_some());
            assert!(last.complete);
        }
        // SAFETY: staging mapping covers the entire first subresource footprint.
        let staged = unsafe { std::slice::from_raw_parts(s.upload.data(), s.footprint_bytes(0)) };
        assert!(
            staged.contains(&0x7c),
            "retried upload must detile newly arrived tail bytes instead of staying zeroed"
        );
        if once() {
            assert_eq!(
                s.refresh_now(expected_full),
                0,
                "once-tagged staging should skip further refreshes after a complete fill"
            );
        }

        crate::descriptor::host::read_all(memory as u64);
        drop(s);
        list.close();
        list.reset();
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(resource, 0x10);
        assert_eq!(release(resource), 0);
        // SAFETY: freeing our reservation.
        assert_ne!(unsafe { VirtualFree(memory, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn once_partial_mip_tail_keeps_refreshing_when_missing_mips_arrive() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        let Some(host) = host() else {
            return;
        };
        const RESERVED: usize = 2 << 20;
        // SAFETY: reserve space for the full mip chain without committing it.
        let memory =
            unsafe { VirtualAlloc(std::ptr::null(), RESERVED, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!memory.is_null());
        let (resource, texture) = test_texture(256, 9, 1, Some(memory as u64));
        let list = host.new_list().expect("host list");
        PICKING.set(Some(u64::MAX));
        upload_from(&list, resource, texture, 0);
        PICKING.set(None);

        let s = staging(resource, texture, 0).expect("staging");
        assert_eq!(s.once.load(Ordering::Relaxed), once());
        let tail_bytes = s.layout.bytes_from(8) as usize;
        assert!(tail_bytes < s.tiled_bytes);
        // SAFETY: commit only the mip-8 tail.
        let mapped_tail = unsafe { VirtualAlloc(memory, tail_bytes, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(mapped_tail, memory);
        // SAFETY: committed tail is writable.
        unsafe { memory.cast::<u8>().write_bytes(0x33, tail_bytes) };

        assert_eq!(s.refresh_now(0), tail_bytes);
        {
            let last = s.last.lock().unwrap();
            assert!(last.hash.is_some());
            assert!(
                !last.complete,
                "partial mip-tail refresh must not mark once-tagged staging complete"
            );
        }

        // Commit the rest of the mip chain; `once`-tagged staging must still refresh!
        let rest_len = s.tiled_bytes - tail_bytes;
        // SAFETY: commit remaining pages in our reservation.
        let mapped_rest = unsafe {
            VirtualAlloc(
                memory.cast::<u8>().add(tail_bytes).cast(),
                rest_len,
                MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!mapped_rest.is_null());
        // SAFETY: committed rest is writable.
        unsafe {
            memory
                .cast::<u8>()
                .add(tail_bytes)
                .write_bytes(0x66, rest_len)
        };

        assert_eq!(
            s.refresh_now(0),
            s.tiled_bytes,
            "once-tagged staging must refresh full chain once missing mips arrive"
        );
        assert!(s.last.lock().unwrap().complete);
        if once() {
            assert_eq!(s.refresh_now(0), 0);
        }

        drop(s);
        list.close();
        list.reset();
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(resource, 0x10);
        assert_eq!(release(resource), 0);
        // SAFETY: release our reservation.
        assert_ne!(unsafe { VirtualFree(memory, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn mapped_coverage_rejects_page_swap_with_equal_byte_count() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE,
            VirtualAlloc, VirtualFree,
        };
        const PAGE: usize = 4096;
        // SAFETY: isolated 2-page reservation freed at the end of the test.
        unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            // Commit only page 0 and read via `read_there_pages`.
            assert!(!VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            base.write_bytes(0xaa, PAGE);
            let mut buf = vec![0u8; 2 * PAGE];
            let mut pages = PageBits::new(base as usize, 2 * PAGE);
            let got = read_there_pages(base, &mut buf, &mut pages);
            assert_eq!(got, PAGE);
            let (there_same, covered_same) = mapped_coverage(base, 2 * PAGE, &pages);
            assert_eq!(there_same, PAGE as u64);
            assert!(covered_same);

            // Simulate a concurrent remap replacing page 0 with page 1: equal
            // committed byte count (PAGE), but page 1 was NOT covered by the read.
            assert_ne!(VirtualFree(base.cast(), PAGE, MEM_DECOMMIT), 0);
            assert!(
                !VirtualAlloc(base.add(PAGE).cast(), PAGE, MEM_COMMIT, PAGE_READWRITE).is_null()
            );
            let (there_swapped, covered_swapped) = mapped_coverage(base, 2 * PAGE, &pages);
            assert_eq!(there_swapped, PAGE as u64);
            assert!(
                !covered_swapped,
                "mapped_coverage must reject a page swap even when committed byte counts match"
            );

            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn ordinary_complete_refresh_resets_short_fill_counter() {
        let Some(host) = host() else {
            return;
        };
        let (resource, texture) = test_texture(64, 1, 1, None);
        let s = staging(resource, texture, 0).expect("staging");
        let start = s.memory as u64;
        crate::descriptor::host::read_all(start);

        // Exhaust the 8 consecutive short-fill allowance for `start`.
        for _ in 0..8 {
            crate::descriptor::host::read_short(start, resource as usize);
            assert!(crate::descriptor::host::clear_short_read(resource as usize));
        }
        crate::descriptor::host::read_short(start, resource as usize);
        assert!(
            !crate::descriptor::host::is_short_read_pending(resource as usize),
            "9th consecutive short fill must be suppressed"
        );

        // An ordinary guarded full refresh must reset the counter via `read_all`.
        assert_eq!(s.refresh_now(0), s.tiled_bytes);
        crate::descriptor::host::read_short(start, resource as usize);
        assert!(
            crate::descriptor::host::is_short_read_pending(resource as usize),
            "short fill after an ordinary complete refresh must queue a retry again"
        );

        crate::descriptor::host::read_all(start);
        drop(s);
        let _ = host;
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(resource, 0x10);
        assert_eq!(release(resource), 0);
    }

    #[test]
    fn per_list_expected_baseline_survives_partial_recording_on_other_list() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE,
            VirtualAlloc, VirtualFree,
        };
        if !fill_unlocked() {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        const RESERVED: usize = 2 << 20;
        // SAFETY: reserve and commit full texture memory for list A's recording.
        let memory =
            unsafe { VirtualAlloc(std::ptr::null(), RESERVED, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!memory.is_null());
        let (resource, texture) = test_texture(256, 1, 1, Some(memory as u64));
        let s = staging(resource, texture, 0).expect("staging");
        assert!(s.tiled_bytes > (64 << 10));
        // SAFETY: commit the full texture span.
        let full = unsafe { VirtualAlloc(memory, s.tiled_bytes, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!full.is_null());
        // SAFETY: full span is writable.
        unsafe { memory.cast::<u8>().write_bytes(0x41, s.tiled_bytes) };

        // List A records while fully mapped (`expected == s.tiled_bytes`).
        let list_a = host.new_list().expect("list A");
        upload_from(&list_a, resource, texture, 0);

        // Decommit the tail, then List B records while only 64 KiB is mapped.
        let tail_len = s.tiled_bytes - (64 << 10);
        // SAFETY: decommitting the page-aligned tail in our reservation.
        let decommitted = unsafe {
            VirtualFree(
                memory.cast::<u8>().add(64 << 10).cast(),
                tail_len,
                MEM_DECOMMIT,
            )
        };
        assert_ne!(decommitted, 0);
        let list_b = host.new_list().expect("list B");
        upload_from(&list_b, resource, texture, 0);

        // Submitting List B + List A during an active remap must preserve List A's
        // full-coverage baseline instead of letting List B overwrite it.
        crate::descriptor::host::read_all(memory as u64);
        list_a.close();
        list_b.close();
        let _remap = crate::watch::title_memory_remapping(memory as usize, s.tiled_bytes);
        host.execute(&[&list_b, &list_a]);
        drop(_remap);

        assert!(
            crate::descriptor::host::is_short_read_pending(resource as usize),
            "List A's higher expected baseline must trigger a short-read notice even when List B recorded partial"
        );
        crate::descriptor::host::read_all(memory as u64);

        drop(s);
        list_a.reset();
        list_b.reset();
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(resource, 0x10);
        assert_eq!(release(resource), 0);
        // SAFETY: release our reservation.
        assert_ne!(unsafe { VirtualFree(memory, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn batched_query_working_set_prefix_spans_multiple_batches_and_stops_at_holes() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree, VirtualProtect,
        };
        const MEM_TOP_DOWN: u32 = 0x10_0000;
        const KERNEL_PAGE: usize = 4096;
        // > 256 pages so `readable_prefix` crosses multiple 256-entry
        // `K32QueryWorkingSetEx` batches (300 pages = 1.2 MiB).
        const TOTAL_PAGES: usize = 300;
        const COMMIT_PAGES: usize = 270;
        const TOTAL_BYTES: usize = TOTAL_PAGES * KERNEL_PAGE;
        const COMMIT_BYTES: usize = COMMIT_PAGES * KERNEL_PAGE;

        // SAFETY: isolated reservation freed at the end of the test.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                TOTAL_BYTES,
                MEM_RESERVE | MEM_TOP_DOWN,
                PAGE_NOACCESS,
            )
            .cast::<u8>()
        };
        assert!(!base.is_null());
        // SAFETY: commit the first 270 pages; leave pages 270..300 reserved.
        let mapped = unsafe { VirtualAlloc(base.cast(), COMMIT_BYTES, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(mapped, base.cast());

        // Touch only the first half so the second half exercises the
        // `read_into_probe` fallback for committed pages not yet in working set.
        // SAFETY: pages 0..135 are committed and writable.
        unsafe { base.write_bytes(0x5A, 135 * KERNEL_PAGE) };

        assert_eq!(readable_prefix(base as usize, TOTAL_BYTES), COMMIT_BYTES);
        assert_eq!(
            readable_prefix(base as usize + 123, TOTAL_BYTES - 123),
            COMMIT_BYTES - 123
        );
        assert!(committed(base, COMMIT_BYTES));
        assert!(!committed(base, TOTAL_BYTES));
        // Unmapped start returns 0 immediately.
        assert_eq!(
            readable_prefix(base as usize + COMMIT_BYTES, 16 * KERNEL_PAGE),
            0
        );

        // `PAGE_NOACCESS` mid-range stops `readable_prefix` before the protected page.
        let mut old = 0u32;
        // SAFETY: page 260 is committed within our reservation.
        let protected = unsafe {
            VirtualProtect(
                base.add(260 * KERNEL_PAGE).cast(),
                KERNEL_PAGE,
                PAGE_NOACCESS,
                &mut old,
            )
        };
        assert_ne!(protected, 0);
        assert_eq!(
            readable_prefix(base as usize, TOTAL_BYTES),
            260 * KERNEL_PAGE
        );
        assert!(!committed(base, COMMIT_BYTES));

        // SAFETY: release our reservation.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn watched_texture_reads_reuse_space_and_expose_only_current_bytes() {
        #[cfg(feature = "diagnostics")]
        if crate::copystat::mode() == 2 {
            return; // this mode skips all copies
        }
        let large = [0x67; 8192];
        let small = [0x21; 128];
        let next = [0x43; 8192];
        let mut allocation = None;
        for source in [&large[..], &small[..], &next[..]] {
            with_tiled_copy(source.as_ptr(), source.len(), |tiled| {
                assert_eq!(tiled, source);
                if let Some(first) = allocation {
                    assert_eq!(tiled.as_ptr(), first);
                } else {
                    allocation = Some(tiled.as_ptr());
                }
            });
        }
    }

    #[test]
    fn unmapped_texture_pages_do_not_keep_previous_pixels() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };

        #[cfg(feature = "diagnostics")]
        if crate::copystat::mode() != 0 {
            return; // diagnostic copy overrides do not handle unmapped pages
        }
        const PAGE: usize = 4096;
        let previous = [0x65; 2 * PAGE];
        with_tiled_copy(previous.as_ptr(), previous.len(), |tiled| {
            assert_eq!(tiled, previous);
        });
        // SAFETY: reserve two pages at an address chosen by Windows.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!base.is_null());
        // SAFETY: commit the first page of that reservation.
        let mapped = unsafe { VirtualAlloc(base, PAGE, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(mapped, base);
        // SAFETY: the first page is writable.
        unsafe { base.cast::<u8>().write_bytes(0x32, PAGE) };
        with_tiled_copy(base.cast(), 2 * PAGE, |tiled| {
            assert!(tiled[..PAGE].iter().all(|&b| b == 0x32));
            assert!(tiled[PAGE..].iter().all(|&b| b == 0));
        });
        // SAFETY: this reservation is no longer used.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
    }

    fn test_texture(
        width: u32,
        mips: u16,
        slices: u16,
        address: Option<u64>,
    ) -> (This, &'static ID3D12Resource) {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        use windows::core::Interface;

        let null = std::ptr::null();
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: width,
            DepthOrArraySize: slices,
            MipLevels: mips,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let iid = ID3D12Resource::IID;
        let mut tex = null;
        let result = if let Some(at) = address {
            crate::resource::create_at_address(
                null,
                at,
                (&raw const desc).cast(),
                0,
                null.cast(),
                (&raw const iid).cast(),
                &mut tex,
            )
        } else {
            crate::resource::create_committed(
                null,
                (&raw const heap).cast(),
                0,
                (&raw const desc).cast(),
                0,
                null.cast(),
                (&raw const iid).cast(),
                &mut tex,
            )
        };
        assert_eq!(result, crate::object::S_OK);
        let host_tex = crate::resource::host_resource(tex).expect("host texture");
        (tex, host_tex)
    }

    #[test]
    fn partially_resident_arrays_keep_slices_on_both_sides_of_a_hole() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };

        if host().is_none() {
            eprintln!("skipping texture array staging: no D3D12 device");
            return;
        }
        #[cfg(feature = "diagnostics")]
        if crate::copystat::mode() != 0 {
            return;
        }
        const RESERVED: usize = 2 << 20;
        // SAFETY: create a new writable reservation for all three slices.
        let memory = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                RESERVED,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!memory.is_null());
        let (resource, texture) = test_texture(128, 1, 3, Some(memory as u64));
        let s = staging(resource, texture, 0).unwrap();
        let stride = s.layout.slice_size as usize;
        assert_eq!(s.tiled_bytes, 3 * stride);
        assert!(s.tiled_bytes <= RESERVED && stride.is_multiple_of(4096));
        let slice = |n: usize| memory.cast::<u8>().wrapping_add(n * stride);
        for (n, byte) in [0x16, 0x58, 0x9a].into_iter().enumerate() {
            // SAFETY: each slice fits within the committed reservation.
            unsafe { slice(n).write_bytes(byte, stride) };
        }
        let pixels = || {
            s.footprints
                .iter()
                .map(|fp| {
                    // SAFETY: each footprint begins within the live mapped staging buffer.
                    unsafe { s.upload.data().add(fp.Offset as usize).read() }
                })
                .collect::<Vec<_>>()
        };
        s.refresh_now(0);
        assert_eq!(pixels(), [0x16, 0x58, 0x9a]);
        // SAFETY: slice 1 is page-aligned and occupies only its own pages.
        let freed = unsafe { VirtualFree(slice(1).cast(), stride, MEM_DECOMMIT) };
        assert_ne!(freed, 0);
        s.refresh_now(0);
        assert_eq!(pixels(), [0x16, 0, 0x9a]);
        // SAFETY: the middle slice remains reserved and may be committed again.
        let mapped = unsafe { VirtualAlloc(slice(1).cast(), stride, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(mapped, slice(1).cast());
        // SAFETY: the preceding call recommitted this slice.
        unsafe { slice(1).write_bytes(0xbc, stride) };
        s.refresh_now(0);
        assert_eq!(pixels(), [0x16, 0xbc, 0x9a]);

        drop(s);
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(resource, 0x10);
        assert_eq!(release(resource), 0);
        // SAFETY: the resource and staging no longer refer to the reservation.
        assert_ne!(unsafe { VirtualFree(memory, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn placed_texture_upload_reads_the_resident_span_and_copies_to_the_requested_mip() {
        let Some(host) = host() else {
            eprintln!("skipping texture readback: no D3D12 device");
            return;
        };
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        // SAFETY: reserve space for the complete texture without committing its pages.
        let memory = unsafe { VirtualAlloc(std::ptr::null(), 2 << 20, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!memory.is_null());
        let (resource, texture) = test_texture(256, 9, 1, Some(memory as u64));
        let mut words = [0; 16];
        words[1] = D3D12_SRV_DIMENSION_TEXTURE2D.0 as u32;
        words[4] = 8;
        words[5] = 1;
        let view = crate::descriptor::View::Placed(false, [0; 14], words);
        let first = crate::descriptor::host::placed_first_mip(&view);
        assert_eq!(first, 8);

        let list = host.new_list().unwrap();
        upload_from(&list, resource, texture, first);
        let tail = staging(resource, texture, first).unwrap();
        assert_eq!(tail.first, 8);
        assert_eq!(tail.footprints.len(), 1);
        assert_eq!(tail.tiled_bytes as u64, tail.layout.bytes_from(first));
        assert!((tail.tiled_bytes as u64) < tail.layout.size);
        // SAFETY: the tail fits in the reservation and is the only committed span.
        let mapped_tail =
            unsafe { VirtualAlloc(memory, tail.tiled_bytes, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(mapped_tail, memory);
        assert!(!committed(memory.cast(), tail.layout.size as usize));
        // SAFETY: the committed tail is writable through the resource's address.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(tail.memory.cast_mut(), tail.tiled_bytes) };
        let pixel = [0xa3, 0xb5, 0xc7, 0xd9];
        tail.layout.tile(bytes, first, 0, &pixel, 4, 4).unwrap();
        assert_eq!(tail.refresh_now(0), tail.tiled_bytes);
        // Snapshot the partial copy on the GPU before another upload can overwrite it.
        let read_mip = || {
            crate::list::ensure_state(&list, resource, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
            let (back, mapped) = host.readback_buffer(tail.size).unwrap();
            let dest = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: the readback buffer lives through execution.
                pResource: unsafe { std::mem::transmute_copy(&back) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: tail.footprints[0],
                },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: the host texture lives through execution.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: first,
                },
            };
            // SAFETY: both locations describe the same 1x1 mip in the copy states.
            unsafe { list.list.CopyTextureRegion(&dest, 0, 0, 0, &src, None) };
            (back, mapped)
        };
        let (_partial_buffer, partial_pixel) = read_mip();
        // The placed table, full-chain copy source, and another list can all use
        // this texture before any of their deferred detiles have run.
        let other = host.new_list().unwrap();
        upload_from(&other, resource, texture, 2);
        let broader = staging(resource, texture, 2).unwrap();
        assert!(other.covers_texture(resource as usize, 2));
        assert!(!list.covers_texture(resource as usize, 2));
        upload(&list, resource, texture);
        assert!(list.covers_texture(resource as usize, 0));
        assert!(Arc::ptr_eq(
            &tail,
            &staging(resource, texture, first).unwrap()
        ));
        assert!(Arc::ptr_eq(
            &broader,
            &staging(resource, texture, 2).unwrap()
        ));
        let (_full_buffer, full_pixel) = read_mip();
        list.close();
        host.execute(&[&list]);
        // SAFETY: execution finished both copies and both mapped buffers are still live.
        let (partial, full) = unsafe {
            (
                partial_pixel.cast::<[u8; 4]>().read_unaligned(),
                full_pixel.cast::<[u8; 4]>().read_unaligned(),
            )
        };
        assert_eq!(partial, pixel, "partial upload wrote mip 8");
        assert_eq!(full, pixel, "full-chain upload preserved mip 8");
        other.close();
        other.reset();
        list.reset();
        assert!(!list.covers_texture(resource as usize, first));
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(resource, 0x10);
        assert_eq!(release(resource), 0);
        // SAFETY: the lists have been reset and the resource has been released.
        assert_ne!(unsafe { VirtualFree(memory, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn forget_drops_staging_immediately_when_idle() {
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let (tex, host_tex) = test_texture(4, 1, 1, None);
        let weak = {
            let s = staging(tex, host_tex, 0).expect("staging");
            Arc::downgrade(&s)
        };
        assert_eq!(weak.strong_count(), 1);
        forget(tex as usize);
        assert_eq!(weak.strong_count(), 0);
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(tex, 0x10);
        assert_eq!(release(tex), 0);
    }

    #[test]
    fn forget_retires_staging_until_active_epoch_finishes() {
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let (tex, host_tex) = test_texture(4, 1, 1, None);
        let weak = {
            let s = staging(tex, host_tex, 0).expect("staging");
            Arc::downgrade(&s)
        };
        assert_eq!(weak.strong_count(), 1);

        let list = host.new_list().expect("host list");
        list.mark_work();

        forget(tex as usize);
        assert_eq!(
            weak.strong_count(),
            1,
            "forgotten texture staging must wait for active epochs that recorded after its birth"
        );

        list.close();
        list.reset();
        assert_eq!(weak.strong_count(), 0);
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(tex, 0x10);
        assert_eq!(release(tex), 0);
    }

    #[test]
    fn forget_drops_new_staging_despite_older_pinned_list() {
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let stale = host.new_list().expect("stale list");
        stale.mark_work();

        let (tex, host_tex) = test_texture(4, 1, 1, None);
        let weak = {
            let s = staging(tex, host_tex, 0).expect("staging");
            Arc::downgrade(&s)
        };
        assert_eq!(weak.strong_count(), 1);

        forget(tex as usize);
        assert_eq!(
            weak.strong_count(),
            0,
            "staging created after stale list's last work must not leak behind it"
        );
        stale.close();
        stale.reset();
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(tex, 0x10);
        assert_eq!(release(tex), 0);
    }

    #[test]
    fn staging_budget_scales_across_ram_tiers() {
        assert_eq!(staging_budget_for_ram(None), 512 * MIB);
        assert_eq!(staging_budget_for_ram(Some(8 * GIB)), 256 * MIB);
        assert_eq!(staging_budget_for_ram(Some(16 * GIB)), 256 * MIB);
        assert_eq!(staging_budget_for_ram(Some(16 * GIB + 1)), 512 * MIB);
        assert_eq!(staging_budget_for_ram(Some(32 * GIB)), 512 * MIB);
        assert_eq!(staging_budget_for_ram(Some(32 * GIB + 1)), GIB);
        assert_eq!(staging_budget_for_ram(Some(64 * GIB)), GIB);
    }

    #[test]
    fn staging_evicts_lru_idle_entries_when_over_budget_and_skips_in_use_entries() {
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };

        let (t1, h1) = test_texture(4, 1, 1, None);
        let (t2, h2) = test_texture(4, 1, 1, None);
        let (t3, h3) = test_texture(4, 1, 1, None);
        let (t4, h4) = test_texture(4, 1, 1, None);

        let (mem1, desc1) = crate::resource::texture_memory(t1).expect("texture memory");
        let size1 = crate::resource::texture_layout(&desc1)
            .expect("texture layout")
            .size as usize;
        // SAFETY: committed title memory for `t1` holds at least `size1` writable bytes.
        unsafe { std::slice::from_raw_parts_mut(mem1.cast_mut(), size1).fill(0x5a) };

        let read_rgba_pixels = |s: &Staging| -> Vec<u8> {
            let fp = s.footprints[0];
            let pitch = fp.Footprint.RowPitch as usize;
            let row_bytes = fp.Footprint.Width as usize * 4;
            let rows = s.rows[0] as usize;
            let mut out = Vec::with_capacity(row_bytes * rows);
            if rows == 0 {
                return out;
            }
            let span = (rows - 1) * pitch + row_bytes;
            assert!(fp.Offset as usize + span <= s.size as usize);
            // SAFETY: `s.upload.data()` is a mapped upload buffer covering `fp.Offset..fp.Offset + span`.
            let mapped = unsafe {
                std::slice::from_raw_parts(s.upload.data().add(fp.Offset as usize), span)
            };
            for row in 0..rows {
                out.extend_from_slice(&mapped[row * pitch..row * pitch + row_bytes]);
            }
            out
        };

        let s1 = create_staging(t1, h1, 0).expect("staging 1");
        s1.refresh(0);
        assert_eq!(read_rgba_pixels(&s1), vec![0x5a; 64]);

        let s2 = create_staging(t2, h2, 0).expect("staging 2");
        let s3 = create_staging(t3, h3, 0).expect("staging 3");
        let entry_bytes = s1.size;
        assert!(entry_bytes > 0);
        assert_eq!(s2.size, entry_bytes);
        assert_eq!(s3.size, entry_bytes);

        let mut cache = StagingCache::default();
        assert!(cache.insert(t1 as usize, s1.clone()).is_none());
        // Re-inserting the same key returns the replaced entry to the caller so it
        // can be dropped after releasing the `STAGING` lock.
        let s1_dup = s1.clone();
        assert!(cache.insert(t1 as usize, s1_dup).is_some());
        assert_eq!(cache.total_bytes, entry_bytes);
        assert!(cache.insert(t2 as usize, s2.clone()).is_none());
        assert!(cache.insert(t3 as usize, s3.clone()).is_none());
        assert_eq!(cache.total_bytes, 3 * entry_bytes);

        // Touch `t1` so LRU order is `t2` (oldest) < `t3` < `t1` (newest).
        s1.touch();

        // Hold `t2` in an open `HostList` so its strong count is 2 (in-use).
        let list = host.new_list().expect("host list");
        list.defer_texture(s2.clone(), 0);

        let (w1, w2, w3) = (
            Arc::downgrade(&s1),
            Arc::downgrade(&s2),
            Arc::downgrade(&s3),
        );
        drop((s1, s2, s3));
        assert_eq!(w1.strong_count(), 1);
        assert_eq!(w2.strong_count(), 2);
        assert_eq!(w3.strong_count(), 1);

        // Insert `t4`, pushing `total_bytes` to 4 * `entry_bytes`.
        let s4 = create_staging(t4, h4, 0).expect("staging 4");
        assert_eq!(s4.size, entry_bytes);
        assert!(cache.insert(t4 as usize, s4).is_none());
        assert_eq!(cache.total_bytes, 4 * entry_bytes);

        // Budget of 3 entries must skip in-use `t2` and evict oldest idle `t3`,
        // leaving newer `t1` and kept `t4`.
        let evicted = cache.evict_to_budget(3 * entry_bytes, t4 as usize, 0);
        assert_eq!(evicted.len(), 1);
        assert_eq!(cache.total_bytes, 3 * entry_bytes);
        assert!(!cache.entries.contains_key(&(t3 as usize)));
        assert!(cache.entries.contains_key(&(t2 as usize)));
        assert!(cache.entries.contains_key(&(t1 as usize)));
        assert!(cache.entries.contains_key(&(t4 as usize)));
        drop(evicted);
        assert_eq!(
            w3.strong_count(),
            0,
            "evicted idle staging must drop immediately"
        );
        assert_eq!(w2.strong_count(), 2);
        assert_eq!(w1.strong_count(), 1);

        // Tightening to 2 entries evicts `t1` while still protecting in-use `t2`.
        let evicted = cache.evict_to_budget(2 * entry_bytes, t4 as usize, 0);
        assert_eq!(evicted.len(), 1);
        assert_eq!(cache.total_bytes, 2 * entry_bytes);
        assert!(!cache.entries.contains_key(&(t1 as usize)));
        assert!(cache.entries.contains_key(&(t2 as usize)));
        assert!(cache.entries.contains_key(&(t4 as usize)));
        drop(evicted);
        assert_eq!(
            w1.strong_count(),
            0,
            "evicted idle staging must drop immediately"
        );

        // Tightening to 1 entry while `t2` is still in-use evicts nothing:
        // neither in-use `t2` nor `keep_key` `t4` may be evicted.
        let evicted = cache.evict_to_budget(entry_bytes, t4 as usize, 0);
        assert!(evicted.is_empty());
        assert_eq!(cache.total_bytes, 2 * entry_bytes);

        // Resetting `list` drops its reference to `t2`; now `t2` becomes idle
        // and is evicted on the next budget pass.
        list.close();
        list.reset();
        assert_eq!(w2.strong_count(), 1);
        let evicted = cache.evict_to_budget(entry_bytes, t4 as usize, 0);
        assert_eq!(evicted.len(), 1);
        assert_eq!(cache.total_bytes, entry_bytes);
        assert!(!cache.entries.contains_key(&(t2 as usize)));
        assert!(cache.entries.contains_key(&(t4 as usize)));
        drop(evicted);
        assert_eq!(w2.strong_count(), 0);

        // Re-creating and refreshing evicted `t1` fully re-detiles its title
        // memory without zeroing or stale contents.
        let recreated = create_staging(t1, h1, 0).expect("recreated staging");
        recreated.refresh(0);
        assert_eq!(read_rgba_pixels(&recreated), vec![0x5a; 64]);
        drop(recreated);

        // Also run a live upload + GPU execution for `t1` after eviction.
        let upload_list = host.new_list().expect("upload list");
        upload(&upload_list, t1, h1);
        upload_list.close();
        host.execute(&[&upload_list]);
        upload_list.reset();
        forget(t1 as usize);

        drop(cache);
        for tex in [t1, t2, t3, t4] {
            let release = crate::object::slot::<extern "system" fn(This) -> u32>(tex, 0x10);
            assert_eq!(release(tex), 0);
        }
    }

    #[test]
    fn alias_from_sub_skips_dropped_top_mips_per_layer() {
        // Source has 5 mips; dest dropped 2 → dest mip 0 is source mip 2.
        assert_eq!(alias_from_sub(0, 0, 2, 5), 2);
        assert_eq!(alias_from_sub(2, 0, 2, 5), 4);
        // Second array slice starts after the source's full chain.
        assert_eq!(alias_from_sub(0, 1, 2, 5), 7);
        assert_eq!(alias_from_sub(1, 1, 0, 3), 4);
    }

    /// A synthetic version of a title that packs light into an integer
    /// texture from compute and samples it through a view of another format
    /// placed at the same address: the view's texels are the written
    /// texture's, not the (empty) title memory behind it.
    #[test]
    fn view_placed_over_a_compute_written_texture_reads_what_compute_wrote() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_UINT, DXGI_SAMPLE_DESC,
        };
        use windows::core::{Interface, s};

        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        const UNORDERED_ACCESS: u32 = 0x8;
        const TEXELS: usize = 16;

        /// A title object, released when the test ends.
        struct Title(This);
        impl Drop for Title {
            fn drop(&mut self) {
                crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
            }
        }

        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return; // no host GPU
        };

        // The title's memory for both textures. Nothing writes it, so a
        // view filled from it reads zeros. The placed view is never
        // released, so the memory is never freed.
        let memory: &'static mut [u8] = Box::leak(vec![0u8; 1 << 20].into_boxed_slice());
        let address = memory.as_ptr() as u64;

        // The words of a `ResourceDesc`: 4x4, one level and layer.
        let words = |format: i32, flags: u32| {
            let mut w = [0u32; 14];
            w[0] = 3; // texture 2D
            w[4] = 4; // width
            w[6] = 4; // height
            w[7] = 1 | (1 << 16); // layers, levels
            w[8] = format as u32;
            w[9] = 1; // samples
            w[12] = flags;
            w
        };
        const ALLOW_UNORDERED_ACCESS: u32 = 0x4;
        #[repr(C, align(8))]
        struct Aligned([u32; 14]);
        let written_desc = Aligned(words(DXGI_FORMAT_R32_UINT.0, ALLOW_UNORDERED_ACCESS));
        let iid = ID3D12Resource::IID;
        let mut written = std::ptr::null();
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                address,
                written_desc.0.as_ptr().cast(),
                0,
                std::ptr::null(),
                (&raw const iid).cast(),
                &mut written,
            ),
            crate::object::S_OK
        );
        let _written = Title(written);
        let written_host = crate::resource::host_resource(written).expect("host texture");
        let placed = crate::resource::texture_at(address, &words(DXGI_FORMAT_R32_FLOAT.0, 0))
            .expect("placed view");
        let placed_host = crate::resource::host_resource(placed).expect("host texture");
        assert_ne!(placed, written);
        assert_eq!(
            crate::resource::alias_of(placed),
            Some(written),
            "the view is found to lie over the texture the host writes"
        );

        // Stores a distinct, ordinary float bit pattern per texel; reads
        // the view as floats and stores their bits.
        const SOURCE: &str = r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=2), SRV(t0, numDescriptors=1))"
RWTexture2D<uint> light : register(u0);
RWByteAddressBuffer dst : register(u1);
Texture2D<float> sampled : register(t0);
[RootSignature(RS)]
[numthreads(4, 4, 1)]
void write(uint3 id : SV_DispatchThreadID) {
    light[id.xy] = 0x3F000000u | ((id.y * 4 + id.x + 1) * 0x1357u);
}
[RootSignature(RS)]
[numthreads(4, 4, 1)]
void read(uint3 id : SV_DispatchThreadID) {
    dst.Store((id.y * 4 + id.x) * 4, asuint(sampled.Load(int3(id.xy, 0))));
}
"#;
        let compile = |entry: windows::core::PCSTR| -> ID3DBlob {
            let mut blob: Option<ID3DBlob> = None;
            // SAFETY: the source and names are static strings.
            unsafe {
                D3DCompile(
                    SOURCE.as_ptr().cast(),
                    SOURCE.len(),
                    s!("alias_view"),
                    None,
                    None,
                    entry,
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .unwrap();
            }
            blob.unwrap()
        };
        let (write_code, read_code) = (compile(s!("write")), compile(s!("read")));
        let bytes = |blob: &ID3DBlob| -> Vec<u8> {
            // SAFETY: the blob holds its own bytes.
            unsafe {
                std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                )
            }
            .to_vec()
        };
        let (write_code, read_code) = (bytes(&write_code), bytes(&read_code));
        // SAFETY: the bytecode includes the root signature.
        let root: ID3D12RootSignature =
            unsafe { host.device.CreateRootSignature(0, &write_code) }.unwrap();
        let pipeline = |code: &[u8]| -> ID3D12PipelineState {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: borrowed for the call only.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            };
            // SAFETY: a valid description.
            unsafe { host.device.CreateComputePipelineState(&desc) }.unwrap()
        };
        let (write_pipeline, read_pipeline) = (pipeline(&write_code), pipeline(&read_code));

        let buffer = |kind, flags, state| -> ID3D12Resource {
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
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: valid buffer creation parameters.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut made,
                    )
                    .unwrap();
            }
            made.unwrap()
        };
        let out = buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let readback = buffer(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        // SAFETY: plain object creation and views of live resources, into
        // slots of a heap with room for all three.
        let (heap, table) = unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 3,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            let step = host
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let start = heap.GetCPUDescriptorHandleForHeapStart();
            let slot = |n: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + n * step,
            };
            let light = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_UAV {
                        MipSlice: 0,
                        PlaneSlice: 0,
                    },
                },
            };
            host.device.CreateUnorderedAccessView(
                written_host,
                None::<&ID3D12Resource>,
                Some(&light),
                slot(0),
            );
            let dst = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: 0,
                        NumElements: TEXELS as u32,
                        StructureByteStride: 0,
                        CounterOffsetInBytes: 0,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    },
                },
            };
            host.device.CreateUnorderedAccessView(
                &out,
                None::<&ID3D12Resource>,
                Some(&dst),
                slot(1),
            );
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_FLOAT,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            };
            host.device
                .CreateShaderResourceView(placed_host, Some(&view), slot(2));
            let table = heap.GetGPUDescriptorHandleForHeapStart();
            (heap, table)
        };

        let list = host.new_list().expect("host list");
        crate::list::ensure_state(&list, written, UNORDERED_ACCESS);
        // SAFETY: `list` is open; the pipelines, heap and textures are live
        // and the textures are in the states the barriers above name.
        unsafe {
            let l = &list.list;
            l.SetComputeRootSignature(&root);
            l.SetDescriptorHeaps(&[Some(heap.clone())]);
            l.SetComputeRootDescriptorTable(0, table);
            l.SetPipelineState(&write_pipeline);
            l.Dispatch(1, 1, 1);
        }
        // What a draw does for a view it samples.
        let aliased = crate::texture::alias(&list, placed, written);
        assert!(aliased, "the written texture is copied into the view's");
        crate::list::ensure_state(&list, placed, SHADER_RESOURCE);
        // SAFETY: as above; the view's texture is now readable.
        unsafe {
            let l = &list.list;
            l.SetPipelineState(&read_pipeline);
            l.Dispatch(1, 1, 1);
            l.ResourceBarrier(&[crate::host::transition(
                &out,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyBufferRegion(&readback, 0, &out, 0, (TEXELS * 4) as u64);
        }
        list.close();
        host.execute(&[&list]);

        let mut got = [0u32; TEXELS];
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the submission completed and the readback holds the texels.
        unsafe {
            let range = D3D12_RANGE {
                Begin: 0,
                End: TEXELS * 4,
            };
            readback.Map(0, Some(&range), Some(&mut mapped)).unwrap();
            std::ptr::copy_nonoverlapping(mapped.cast::<u32>(), got.as_mut_ptr(), TEXELS);
            readback.Unmap(0, None);
        }
        let expected: [u32; TEXELS] =
            std::array::from_fn(|i| 0x3F00_0000 | ((i as u32 + 1) * 0x1357));
        assert_eq!(got, expected);
        list.reset();
    }
}

#[cfg(test)]
mod resident_tests {
    use super::{forget, host_written, is_host_written, note_resident, resident};

    #[test]
    fn noted_resident_window_updates_when_more_arrive() {
        let resource = 0x616u64 as _;
        note_resident(resource, Some((5, 4)));
        assert_eq!(resident(resource), Some((5, 4)));
        note_resident(resource, Some((2, 7)));
        assert_eq!(resident(resource), Some((2, 7)));
        note_resident(resource, None);
        assert_eq!(resident(resource), None);
    }

    #[test]
    fn host_written_clears_resident_window() {
        let resource = 0x617u64 as _;
        note_resident(resource, Some((3, 5)));
        assert_eq!(resident(resource), Some((3, 5)));
        host_written(resource);
        assert!(is_host_written(resource));
        assert_eq!(resident(resource), None);
        // Leave no cross-test residue in the process-wide sets.
        forget(resource as usize);
    }

    #[test]
    fn with_scratch_is_reentrant_safe() {
        let outer = super::with_scratch(64, |outer_buf| {
            outer_buf.fill(0xaa);
            let inner = super::with_scratch(256, |inner_buf| {
                inner_buf.fill(0x55);
                inner_buf.len()
            });
            assert!(outer_buf.iter().all(|&b| b == 0xaa));
            (outer_buf.len(), inner)
        });
        assert_eq!(outer, (64, 256));
    }
}
