//! Translated shaders and translation refusals kept on disk (decision 0003):
//! under the configured cache folder, keyed by the container's content and by
//! the VIctory build that translated it (the BLAKE3 of our executable and
//! translation settings), so a rebuilt translator never serves an old
//! translation or an old refusal. Entries go through `victory-pipeline`'s
//! content-addressed cache, which writes them atomically; an entry that does
//! not read back is translated again.
//!
//! Each entry is our own encoding of `Result<HostShader, String>`: a field
//! added to [`HostShader`] must be added to `encode` and `decode` too (entries
//! of older builds are ignored anyway).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, StructuredSrv, UserData};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// Whether the active host adapter supports `WaveSize(32)` compute shaders.
fn allow_wave32() -> bool {
    #[cfg(windows)]
    {
        crate::host::host().is_none_or(|h| h.supports_wave32())
    }
    #[cfg(not(windows))]
    {
        true
    }
}

/// What identifies this build: the BLAKE3 of our executable file and of
/// the translation settings it runs with (the loop limit and the clean-up
/// passes, which a diagnostics build can change at run time, and whether
/// Wave32 lowering is enabled for the active host adapter).
fn build(wave32: bool) -> &'static [u8; 32] {
    static BUILDS: [OnceLock<[u8; 32]>; 2] = [const { OnceLock::new() }; 2];
    BUILDS[usize::from(wave32)].get_or_init(|| {
        // Diagnostics builds: `VICTORY_SHADER_CACHE_AS=<exe>` keys the cache by
        // that executable instead, so a rebuild that changed no translation
        // (only logging, say) reuses the translations an earlier build made
        // rather than starting the world cold. Stale if the translator did
        // change between the two.
        let keyed_by = std::env::var_os("VICTORY_SHADER_CACHE_AS")
            .filter(|_| cfg!(feature = "diagnostics"))
            .map(std::path::PathBuf::from);
        if let Some(path) = &keyed_by {
            tracing::warn!(path = %path.display(), "VICTORY_SHADER_CACHE_AS: shader cache keyed by another executable");
        }
        let exe = keyed_by.map_or_else(std::env::current_exe, Ok).and_then(std::fs::read);
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        match exe {
            Ok(mut bytes) => {
                bytes.extend_from_slice(&limit.to_le_bytes());
                bytes.extend_from_slice(&victory_shader::emit::shader_opt().to_le_bytes());
                bytes.push(
                    u8::from(wave32)
                        | (u8::from(wave32 && victory_shader::emit::wave64_double()) << 1)
                        | (u8::from(wave32 && victory_shader::emit::wave64_console_flow()) << 2)
                        | (u8::from(victory_shader::emit::indirect_amplify_enabled()) << 3),
                );
                if wave32 {
                    bytes.extend_from_slice(&victory_shader::emit::wave_wide_cache_key());
                }
                bytes.push(u8::from(victory_shader::emit::host_wave64()));
                // `VICTORY_WAVE64_SPLIT_ONLY` counts per shader (`cached`).
                bytes.extend_from_slice(&victory_shader::emit::fixes().to_le_bytes());
                bytes.extend_from_slice(&victory_shader::switch::bits().to_le_bytes());
                *blake3::hash(&bytes).as_bytes()
            }
            Err(e) => {
                // Without it no entry can be trusted: a fresh key per run.
                tracing::warn!(error = %e, "cannot read our executable; shader cache unused");
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                *blake3::hash(&now.as_nanos().to_le_bytes()).as_bytes()
            }
        }
    })
}

/// The translation of a console shader container, from the cache when
/// there is one.
pub(crate) fn host_shader(container: &[u8]) -> Result<HostShader, String> {
    let wave32 = allow_wave32();
    let translate = |c: &[u8]| {
        let hash = blake3::hash(c).to_hex();
        victory_shader::emit::translate_with_wave32_for_hash(c, wave32, &hash[..16])
            .map_err(|e| e.to_string())
    };
    match CACHE.get() {
        Some(cache) => cached(cache, build(wave32), container, translate),
        None => translate(container),
    }
}

/// The translation of a pixel shader container for a host without
/// barycentrics (`victory_shader::emit::translate_without_barycentrics`),
/// from the cache when there is one. Its entries have keys of their own:
/// the build fingerprint is salted with the variant.
pub(crate) fn host_shader_without_barycentrics(container: &[u8]) -> Result<HostShader, String> {
    let wave32 = allow_wave32();
    let translate = |c: &[u8]| {
        victory_shader::emit::translate_without_barycentrics(c, wave32).map_err(|e| e.to_string())
    };
    match CACHE.get() {
        Some(cache) => {
            let build = without_barycentrics_build(build(wave32));
            cached(cache, &build, container, translate)
        }
        None => translate(container),
    }
}

/// The fingerprint of translations without barycentrics made by `build`.
fn without_barycentrics_build(build: &[u8; 32]) -> [u8; 32] {
    *blake3::hash(&[&build[..], b"without barycentrics"].concat()).as_bytes()
}

/// The cache entry's converter: the build and the container in, the encoded
/// translation or refusal out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
    /// The last translation outcome produced by `convert`, kept so a cache
    /// write failure returns it directly instead of re-running translation.
    result: Mutex<Option<Result<HostShader, String>>>,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning; 2: buffer
        // accesses and first-lane addresses on title memory by address.
        2
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        // The input is the 32-byte build fingerprint, then the container.
        let container = input
            .get(32..)
            .ok_or_else(|| anyhow::anyhow!("shader cache input shorter than its build key"))?;
        let outcome = (self.translate)(container);
        let bytes = encode(&outcome);
        *self.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(outcome);
        Ok(bytes)
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    // Whether the split filter takes this shader is part of its key, so a
    // shader keeps its entry across bisection runs with different filters.
    let input = if victory_shader::emit::split_takes(container) {
        [&build[..], container].concat()
    } else {
        let left_out = blake3::hash(&[&build[..], b"split left out"].concat());
        [&left_out.as_bytes()[..], container].concat()
    };
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
        result: Mutex::new(None),
    };
    for _ in 0..2 {
        converter.ran.store(false, Ordering::Relaxed);
        *converter.result.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            // A full disk, a file in the way, a scanner holding the new
            // entry: the translation outcome is still valid, only not kept.
            Err(e) => {
                tracing::warn!(error = format!("{e:#}"), "shader cache entry not written");
                break;
            }
        };
        if let Some(outcome) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let ran = converter.ran.load(Ordering::Relaxed);
            let counter = if ran { &MISSES } else { &HITS };
            let what = match (outcome.is_ok(), ran) {
                (true, true) => "translated and cached",
                (true, false) => "from the cache",
                (false, true) => "refusal cached",
                (false, false) => "refusal from the cache",
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return outcome;
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    converter
        .result
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .unwrap_or_else(|| (converter.translate)(container))
}

const MAGIC: &[u8; 4] = b"VSH4";
const REFUSAL_MAGIC: &[u8; 4] = b"VSE1";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

/// The tag word of a root type, and for [`RootType::Other`] the raw value
/// in a word of its own: the value is any `u32` the title's metadata holds,
/// so folding it into the tag would overflow or alias a named type.
fn root_code(r: RootType) -> (u32, Option<u32>) {
    match r {
        RootType::Constants => (0, None),
        RootType::Descriptor => (1, None),
        RootType::Table => (2, None),
        RootType::ImmediateSampler => (3, None),
        RootType::ResourceHeap => (4, None),
        RootType::SamplerHeap => (5, None),
        RootType::Other(x) => (OTHER_ROOT, Some(x)),
    }
}

const OTHER_ROOT: u32 = 6;

/// The root type of a tag word; `other` reads the raw value that follows
/// [`OTHER_ROOT`]. An unknown tag is damage.
fn root_type(code: u32, other: impl FnOnce() -> Option<u32>) -> Option<RootType> {
    Some(match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        OTHER_ROOT => RootType::Other(other()?),
        _ => return None,
    })
}

/// Little-endian words and length-prefixed lists.
fn encode(outcome: &Result<HostShader, String>) -> Vec<u8> {
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let s = match outcome {
        Ok(s) => s,
        Err(reason) => {
            let mut out = REFUSAL_MAGIC.to_vec();
            word(&mut out, reason.len() as u32);
            out.extend_from_slice(reason.as_bytes());
            return out;
        }
    };
    let mut out = MAGIC.to_vec();
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        let (root, other) = root_code(u.root);
        word(&mut out, u.index);
        word(&mut out, u32::from(u.slot));
        word(&mut out, root);
        if let Some(other) = other {
            word(&mut out, other);
        }
        word(&mut out, u.dword);
        word(&mut out, u32::from(u.indexed_buffer));
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, s.structured_srvs.len() as u32);
    for srv in &s.structured_srvs {
        word(&mut out, u32::from(srv.slot));
        word(&mut out, srv.offset_dwords);
        word(&mut out, srv.stride);
    }
    let reads = victory_shader::emit::heap_reads(&s.dxil);
    word(&mut out, reads.len() as u32);
    for r in reads.iter() {
        for v in [
            u32::from(r.slot),
            r.offset_dwords,
            u32::from(r.sampler),
            r.props.0,
            r.props.1,
            u32::from(r.dynamic) | u32::from(r.absolute) << 1,
            r.index_from.map_or(0, |(slot, _)| u32::from(slot) + 1),
            r.index_from.map_or(0, |(_, dword)| dword),
            r.index_load.map_or(0, |l| u32::from(l.slot) + 1),
            r.index_load.map_or(0, |l| l.dword),
            r.index_load.map_or(0, |l| l.byte),
            r.index_load.map_or(0, |l| l.mask),
        ] {
            word(&mut out, v);
        }
    }
    out
}

fn decode(bytes: &[u8]) -> Option<Result<HostShader, String>> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    let magic = r.take(4)?;
    if magic == REFUSAL_MAGIC {
        let raw = r.list()?;
        let reason = String::from_utf8(raw).ok()?;
        return r.0.is_empty().then_some(Err(reason));
    }
    if magic != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        let index = r.word()?;
        let slot = u8::try_from(r.word()?).ok()?;
        let code = r.word()?;
        let root = root_type(code, || r.word())?;
        user_data.push(UserData {
            index,
            slot,
            root,
            dword: r.word()?,
            indexed_buffer: match r.word()? {
                0 => false,
                1 => true,
                _ => return None,
            },
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let n = r.word()? as usize;
    let mut structured_srvs = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        structured_srvs.push(StructuredSrv {
            slot: u8::try_from(r.word()?).ok()?,
            offset_dwords: r.word()?,
            stride: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut heap_reads = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        heap_reads.push(victory_shader::emit::HeapRead {
            slot: u8::try_from(r.word()?).ok()?,
            offset_dwords: r.word()?,
            sampler: r.word()? != 0,
            props: (r.word()?, r.word()?),
            dynamic: false,
            absolute: false,
            index_from: None,
            index_load: None,
        });
        let flags = r.word()?;
        let (from_slot, from_dword) = (r.word()?, r.word()?);
        let load = [r.word()?, r.word()?, r.word()?, r.word()?];
        let last = heap_reads.last_mut()?;
        last.dynamic = flags & 1 != 0;
        last.absolute = flags & 2 != 0;
        last.index_from = from_slot
            .checked_sub(1)
            .and_then(|s| u8::try_from(s).ok())
            .map(|s| (s, from_dword));
        last.index_load = load[0]
            .checked_sub(1)
            .and_then(|s| u8::try_from(s).ok())
            .map(|slot| victory_shader::emit::IndexLoad {
                slot,
                dword: load[1],
                byte: load[2],
                mask: load[3],
            });
    }
    if !r.0.is_empty() {
        return None;
    }
    victory_shader::emit::note_heap_reads(&dxil, heap_reads);
    r.0.is_empty().then_some(Ok(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        structured_srvs,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                indexed_buffer: true,
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![1],
            structured_srvs: vec![StructuredSrv {
                slot: 0,
                offset_dwords: 8,
                stride: 64,
            }],
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH2 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
    }

    /// A translation without barycentrics has an entry of its own: the
    /// plain translation of the same container is not served for it, nor
    /// the other way round.
    #[test]
    fn translations_without_barycentrics_have_entries_of_their_own() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a pixel shader with barycentrics".as_slice();
        let mut fallback = shader();
        fallback.dxil = b"DXBC without barycentrics".to_vec();
        assert_eq!(
            cached(&cache, &build, container, |_: &[u8]| Ok(shader())),
            Ok(shader())
        );
        let variant = without_barycentrics_build(&build);
        assert_ne!(variant, build);
        assert_eq!(
            cached(&cache, &variant, container, |_: &[u8]| Ok(fallback.clone())),
            Ok(fallback.clone())
        );
        let never = |_: &[u8]| -> Result<HostShader, String> { panic!("translated again") };
        assert_eq!(cached(&cache, &build, container, never), Ok(shader()));
        assert_eq!(cached(&cache, &variant, container, never), Ok(fallback));
    }

    #[test]
    fn refusals_come_back_from_the_cache_until_the_build_changes() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build_v1 = [7u8; 32];
        let build_v2 = [8u8; 32];
        let container = b"a refused console container".as_slice();
        let runs = AtomicU64::new(0);
        let refuse = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("unsupported: test refusal".to_owned())
        };

        assert_eq!(
            cached(&cache, &build_v1, container, refuse),
            Err("unsupported: test refusal".to_owned())
        );
        assert_eq!(
            cached(&cache, &build_v1, container, refuse),
            Err("unsupported: test refusal".to_owned())
        );
        assert_eq!(
            runs.load(Ordering::Relaxed),
            1,
            "the second refusal came from disk"
        );

        // A new translator build retries the container without clearing the cache.
        let fixed = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build_v2, container, fixed), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        assert_eq!(cached(&cache, &build_v2, container, fixed), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged refusal entry is removed and re-translated.
        let damaged_dir = tempfile::tempdir().unwrap();
        let damaged_cache = Cache::open(damaged_dir.path()).unwrap();
        let damaged_runs = AtomicU64::new(0);
        let refuse_again = |_: &[u8]| {
            damaged_runs.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("unsupported: damaged".to_owned())
        };
        assert!(cached(&damaged_cache, &build_v1, container, refuse_again).is_err());
        assert_eq!(damaged_runs.load(Ordering::Relaxed), 1);
        for e in walk(damaged_dir.path()) {
            std::fs::write(e, b"VSE1 broken").unwrap();
        }
        assert_eq!(
            cached(&damaged_cache, &build_v1, container, refuse_again),
            Err("unsupported: damaged".to_owned())
        );
        assert_eq!(damaged_runs.load(Ordering::Relaxed), 2);
        assert_eq!(
            cached(&damaged_cache, &build_v1, container, refuse_again),
            Err("unsupported: damaged".to_owned())
        );
        assert_eq!(damaged_runs.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn a_cache_that_cannot_keep_an_entry_still_gives_the_translation() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        // A file where every entry's folder would go: nothing can be written.
        for shard in 0..=0xff_u8 {
            std::fs::write(dir.path().join(format!("{shard:02x}")), b"").unwrap();
        }
        let ok_runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            ok_runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        let container = b"a console container".as_slice();
        assert_eq!(cached(&cache, &[7; 32], container, translate), Ok(shader()));
        assert_eq!(ok_runs.load(Ordering::Relaxed), 1);
        // A shader that does not translate still fails once, with its own error.
        let fail_runs = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            fail_runs.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("unsupported".to_owned())
        };
        let error = cached(&cache, &[7; 32], b"other", fail).unwrap_err();
        assert_eq!(error, "unsupported");
        assert_eq!(fail_runs.load(Ordering::Relaxed), 1);
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn heap_reads_come_back_with_their_entry() {
        use victory_shader::emit::{HeapRead, IndexLoad, heap_reads, note_heap_reads};
        let mut s = shader();
        // A container digest of its own, so other tests do not share it.
        s.dxil = b"DXBCheap-reads-test-digest".to_vec();
        let reads = vec![
            HeapRead {
                slot: 0,
                offset_dwords: 16,
                sampler: false,
                props: (2, 0x409),
                dynamic: false,
                absolute: false,
                index_from: None,
                index_load: None,
            },
            HeapRead {
                slot: 1,
                offset_dwords: 0,
                sampler: false,
                props: (11, 0),
                dynamic: true,
                absolute: true,
                index_from: Some((1, 1)),
                index_load: Some(IndexLoad {
                    slot: 1,
                    dword: 1,
                    byte: 8,
                    mask: 0xfffff,
                }),
            },
        ];
        note_heap_reads(&s.dxil, reads.clone());
        let bytes = encode(&Ok(s.clone()));
        note_heap_reads(&s.dxil, Vec::new());
        assert_eq!(decode(&bytes), Some(Ok(s.clone())));
        assert_eq!(&*heap_reads(&s.dxil), reads.as_slice());
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let ok = Ok(shader());
        let bytes = encode(&ok);
        assert_eq!(decode(&bytes), Some(ok));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);

        let err: Result<HostShader, String> = Err("unsupported: opcode".to_owned());
        let err_bytes = encode(&err);
        assert_eq!(decode(&err_bytes), Some(err));
        assert_eq!(decode(&err_bytes[..err_bytes.len() - 1]), None);
        assert_eq!(decode(&[err_bytes.as_slice(), &[0]].concat()), None);

        let empty_err: Result<HostShader, String> = Err(String::new());
        assert_eq!(decode(&encode(&empty_err)), Some(empty_err));

        // Invalid UTF-8 inside a refusal entry is treated as damaged.
        let mut bad_utf8 = REFUSAL_MAGIC.to_vec();
        bad_utf8.extend(2u32.to_le_bytes());
        bad_utf8.extend([0xff, 0xfe]);
        assert_eq!(decode(&bad_utf8), None);
    }

    #[test]
    fn every_root_type_round_trips() {
        let roots = [
            RootType::Constants,
            RootType::Descriptor,
            RootType::Table,
            RootType::ImmediateSampler,
            RootType::ResourceHeap,
            RootType::SamplerHeap,
            RootType::Other(0),
            RootType::Other(2),
            RootType::Other(u32::MAX - 5),
            RootType::Other(u32::MAX),
        ];
        for root in roots {
            let mut s = shader();
            s.user_data[0].root = root;
            let ok = Ok(s);
            assert_eq!(decode(&encode(&ok)), Some(ok), "{root:?}");
        }
        // A tag past the known ones is damage, not a guess.
        let mut bytes = encode(&Ok(shader()));
        let root_at = 4 + 4 + 4 + shader().dxil.len() + 4 + 4 + 4;
        bytes[root_at..root_at + 4].copy_from_slice(&7u32.to_le_bytes());
        assert_eq!(decode(&bytes), None);
    }

    #[test]
    fn an_input_shorter_than_the_build_key_is_an_error() {
        let converter = Translate {
            translate: |_: &[u8]| Ok(shader()),
            ran: AtomicBool::new(false),
            result: Mutex::new(None),
        };
        assert!(converter.convert(&[0; 31]).is_err());
        assert!(converter.convert(&[0; 32]).is_ok());
    }
}
