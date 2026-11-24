//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! The library is free-threaded but for loads of one pipeline on several
//! threads (`ID3D12Device1::CreatePipelineLibrary`, "Thread Safety"), so
//! pipelines of different names load at the same time. A load is not always
//! quick: the library of an AMD driver holds about 140 bytes per pipeline,
//! and what its own cache lacks is compiled inside the load, for seconds.
//! Loaded one at a time, 650 pipelines kept the eight workers to one core
//! for minutes, while each title frame could wait out its 500 ms for
//! them (`make`). Stores and serialising keep the lock: the size asked for
//! must still be right when the bytes are written (upstream #1000).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
    /// Pipelines stored since the last save.
    added: u64,
}

// SAFETY: D3D12 objects are free-threaded (a pipeline library with the
// exception `NAMES` takes care of).
unsafe impl Send for Library {}

/// Names one thread at a time holds.
struct Names {
    held: Mutex<Option<HashSet<String>>>,
    released: Condvar,
}

/// A name held until this is dropped.
struct Held<'a> {
    names: &'a Names,
    name: &'a str,
}

impl Names {
    const fn new() -> Names {
        Names {
            held: Mutex::new(None),
            released: Condvar::new(),
        }
    }

    /// Holds `name`, once no other thread does.
    fn hold<'a>(&'a self, name: &'a str) -> Held<'a> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        while held.as_ref().is_some_and(|h| h.contains(name)) {
            held = self.released.wait(held).unwrap_or_else(|e| e.into_inner());
        }
        held.get_or_insert_default().insert(name.to_owned());
        Held { names: self, name }
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        let mut held = self.names.held.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = held.as_mut() {
            h.remove(self.name);
        }
        drop(held);
        self.names.released.notify_all();
    }
}

/// Pipelines being loaded, or made and stored.
static NAMES: Names = Names::new();

const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most: the most recently
/// drawn with are kept (upstream #983).
const EARLY_MAX: usize = 8192;

fn library() -> Option<&'static Mutex<Library>> {
    static LIBRARY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let blob = DIR
                .get()
                .and_then(|d| std::fs::read(d.join(LIBRARY_FILE)).ok())
                .unwrap_or_default();
            let library = open(&device, blob);
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(Mutex::new)
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device
                .CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?
                .cast()
        }
    };
    if !blob.is_empty() {
        match make(&blob) {
            Ok(library) => {
                tracing::info!(bytes = blob.len(), "host pipeline library loaded");
                return Some(Library {
                    library,
                    _blob: blob,
                    added: 0,
                });
            }
            Err(e) => {
                tracing::info!(error = %e, "host pipeline library rejected; starting an empty one")
            }
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    // An empty library is asked for with no blob at all, which `make` cannot
    // say: an empty slice still has an address, and vkd3d-proton refuses an
    // address with no bytes behind it. That left it with no library at all.
    let mut raw = std::ptr::null_mut();
    // SAFETY: no blob is read; a success that wrote a pointer wrote an owned
    // ID3D12PipelineLibrary.
    let made = unsafe {
        (Interface::vtable(device).CreatePipelineLibrary)(
            Interface::as_raw(device),
            std::ptr::null(),
            0,
            &ID3D12PipelineLibrary::IID,
            &mut raw,
        )
        .ok()
        .and_then(|()| {
            if raw.is_null() {
                return Err(windows::Win32::Foundation::E_POINTER.into());
            }
            ID3D12PipelineLibrary::from_raw(raw).cast::<ID3D12PipelineLibrary1>()
        })
    };
    match made {
        Ok(library) => Some(Library {
            library,
            _blob: Vec::new(),
            added: 0,
        }),
        Err(e) => {
            tracing::info!(error = %e, "the host refused an empty pipeline library");
            None
        }
    }
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root]
        .into_iter()
        .chain(code.iter().copied())
        .chain([state.as_bytes()])
    {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    // One thread per name loads or makes it; the others then load it.
    let _held = NAMES.hold(name);
    // The interface alone under the lock: loads take up to seconds. Its
    // blob lives as long as the library, which is never dropped.
    let loads = library
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .library
        .clone();
    if let Ok(pso) = load(&loads, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        l.added += 1;
    }
    Ok(pso)
}

/// Writes the library and the early list if pipelines were added since the
/// last save and `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    // Held over serialisation and both writes, so overlapping saves cannot
    // replace files under each other.
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(library)) = (DIR.get(), library()) else {
        return;
    };
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
        return;
    }
    *last = Some(std::time::Instant::now());
    let (blob, saved_additions) = {
        let l = library.lock().unwrap_or_else(|e| e.into_inner());
        if l.added == 0 && !USED_CHANGED.load(Ordering::Relaxed) {
            return;
        }
        // SAFETY: a buffer of the size the library asks for.
        let bytes = unsafe {
            let mut bytes = vec![0u8; l.library.GetSerializedSize()];
            l.library.Serialize(&mut bytes).map(|()| bytes)
        };
        match bytes {
            Ok(bytes) => (bytes, l.added),
            Err(e) => {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                return;
            }
        }
    };
    let used_changed = USED_CHANGED.swap(false, Ordering::Relaxed);
    // Through `used`: an early list not read yet is read before it is
    // written, not replaced by an empty one.
    let (early, listed) = used()
        .as_ref()
        .map(|e| (e.to_text(), e.listed()))
        .unwrap_or_default();
    let written = std::fs::create_dir_all(dir)
        .and_then(|()| write_atomically(&dir.join(LIBRARY_FILE), &blob))
        .and_then(|()| write_atomically(&dir.join(EARLY_FILE), early.as_bytes()));
    match written {
        Ok(()) => {
            let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
            // Keep pipelines stored while the files were being written.
            l.added = l.added.saturating_sub(saved_additions);
            tracing::info!(
                bytes = blob.len(),
                early = listed,
                made = MADE.load(Ordering::Relaxed),
                loaded = LOADED.load(Ordering::Relaxed),
                started_early = EARLY.load(Ordering::Relaxed),
                "host pipeline library saved"
            );
        }
        Err(e) => {
            // At least one cache file was not published. Keep the counters
            // dirty for the next save.
            if used_changed {
                USED_CHANGED.store(true, Ordering::Relaxed);
            }
            tracing::warn!(error = %e, "host pipeline library not saved");
        }
    }
}

/// Writes `bytes` to a unique temporary file beside `path`, then renames it
/// over `path`: a reader never sees a part, and overlapping saves do not
/// share a temporary file.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let extension = format!(
        "tmp-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let tmp = path.with_extension(extension);
    let written = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

/// Title pipeline keys drawn or dispatched with, each with the last run
/// that did. The list keeps the `EARLY_MAX` most recently used, so the
/// pipelines a run draws with for the first time replace those no run has
/// used for longest instead of being left out once the list is full.
#[derive(Debug)]
struct Early {
    /// This run's number: one past the newest listed.
    run: u64,
    keys: HashMap<String, u64>,
}

impl Early {
    /// The list as saved: `<run> <key>` lines, most recent first. A bare
    /// key (as earlier builds wrote) is older than any numbered one.
    fn parse(text: &str) -> Early {
        let mut keys: HashMap<String, u64> = HashMap::new();
        for line in text.lines().filter(|l| !l.is_empty()) {
            let (run, key) = line
                .split_once(' ')
                .and_then(|(run, key)| Some((run.parse().ok()?, key)))
                .unwrap_or((0, line));
            let last = keys.entry(key.to_owned()).or_insert(run);
            *last = (*last).max(run);
        }
        let run = keys.values().max().map_or(1, |r| r + 1);
        let mut early = Early { run, keys };
        if early.keys.len() > EARLY_MAX {
            early.keys = early.newest().map(|(k, r)| (k.clone(), r)).collect();
        }
        early
    }

    /// The keys to keep, most recently used first (ties by key, so the list
    /// is the same for the same uses).
    fn newest(&self) -> impl Iterator<Item = (&String, u64)> {
        let mut keys: Vec<(&String, u64)> = self.keys.iter().map(|(k, &r)| (k, r)).collect();
        keys.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        keys.into_iter().take(EARLY_MAX)
    }

    fn to_text(&self) -> String {
        self.newest().map(|(key, run)| format!("{run} {key}\n")).collect()
    }

    /// Keys the saved list holds.
    fn listed(&self) -> usize {
        self.keys.len().min(EARLY_MAX)
    }

    fn contains(&self, key: &str) -> bool {
        self.keys.contains_key(key)
    }

    /// Notes `key` used this run; whether the saved list changes.
    fn note(&mut self, key: &str) -> bool {
        match self.keys.get_mut(key) {
            Some(run) if *run == self.run => false,
            Some(run) => {
                *run = self.run;
                true
            }
            None => {
                self.keys.insert(key.to_owned(), self.run);
                true
            }
        }
    }
}

static USED: Mutex<Option<Early>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<Early>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let early = Early::parse(&listed);
        tracing::info!(pipelines = early.keys.len(), "title pipelines to start early");
        *used = Some(early);
    }
    used
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|e| e.contains(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    if used.get_or_insert_with(|| Early::parse("")).note(key) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_early_list_takes_new_pipelines_in_place_of_the_oldest() {
        // A list filled by earlier runs: run 1's keys, then run 2's.
        let mut text = String::new();
        for i in 0..EARLY_MAX {
            let run = if i < EARLY_MAX / 2 { 1 } else { 2 };
            text.push_str(&format!("{run} old{i:05}\n"));
        }
        let mut early = Early::parse(&text);
        assert_eq!(early.run, 3);
        assert!(early.contains("old00000"));

        assert!(early.note("new"));
        assert!(early.note(&format!("old{:05}", EARLY_MAX - 1)));
        let kept = Early::parse(&early.to_text());
        assert_eq!(kept.listed(), EARLY_MAX);
        assert!(kept.contains("new"), "a new pipeline is listed though the list was full");
        assert!(kept.contains(&format!("old{:05}", EARLY_MAX / 2)), "run 2 is kept");
        // One of run 1's keys made room.
        let run1 = (0..EARLY_MAX / 2).filter(|i| kept.contains(&format!("old{i:05}"))).count();
        assert_eq!(run1, EARLY_MAX / 2 - 1);
        assert_eq!(kept.run, 4);
    }

    #[test]
    fn early_list_reads_bare_keys_as_oldest() {
        let mut early = Early::parse("aaaa\nbbbb\n\n");
        assert_eq!(early.run, 1);
        assert!(early.contains("aaaa") && early.contains("bbbb"));
        assert!(early.note("aaaa"), "a listed key used again moves to this run");
        assert!(!early.note("aaaa"), "and is noted once per run");
        assert_eq!(early.to_text(), "1 aaaa\n0 bbbb\n");
    }

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let library =
            open(&device, b"not a pipeline library".repeat(16)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    #[test]
    fn concurrent_atomic_writes_publish_one_complete_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        let start = std::sync::Arc::new(std::sync::Barrier::new(8));
        let mut threads = Vec::new();
        for thread in 0..8 {
            let path = path.clone();
            let start = start.clone();
            threads.push(std::thread::spawn(move || {
                start.wait();
                let byte = thread as u8 + 1;
                for _ in 0..32 {
                    write_atomically(&path, &[byte; 64 * 1024]).unwrap();
                }
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }

        let published = std::fs::read(&path).unwrap();
        assert_eq!(published.len(), 64 * 1024);
        assert!(published.iter().all(|&byte| byte == published[0]));
        assert!((1..=8).contains(&published[0]));
        assert!(
            !std::fs::read_dir(dir.path())
                .unwrap()
                .flatten()
                .any(|entry| entry.file_name().to_string_lossy().contains(".tmp-"))
        );
    }

    #[test]
    fn failed_atomic_writes_remove_their_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        std::fs::create_dir(&path).unwrap();

        assert!(write_atomically(&path, b"not published").is_err());
        assert!(
            !std::fs::read_dir(dir.path())
                .unwrap()
                .flatten()
                .any(|entry| entry.file_name().to_string_lossy().contains(".tmp-"))
        );
    }

    #[test]
    fn different_names_are_held_at_the_same_time() {
        static NAMES: Names = Names::new();
        // Each thread holds its name until every thread holds its own.
        let all = std::sync::Barrier::new(8);
        std::thread::scope(|s| {
            for i in 0..8 {
                let all = &all;
                s.spawn(move || {
                    let name = format!("pipeline {i}");
                    let _held = NAMES.hold(&name);
                    all.wait();
                });
            }
        });
        assert!(NAMES.held.lock().unwrap().as_ref().unwrap().is_empty());
    }

    #[test]
    fn one_name_is_held_by_one_thread_at_a_time() {
        static NAMES: Names = Names::new();
        static INSIDE: AtomicU64 = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..500 {
                        let _held = NAMES.hold("pipeline");
                        assert_eq!(INSIDE.fetch_add(1, Ordering::SeqCst), 0);
                        std::hint::spin_loop();
                        INSIDE.fetch_sub(1, Ordering::SeqCst);
                    }
                });
            }
        });
        // A holder that panics lets go of it.
        let panicked = std::thread::spawn(|| {
            let _held = NAMES.hold("pipeline");
            panic!("on purpose");
        });
        assert!(panicked.join().is_err());
        drop(NAMES.hold("pipeline"));
    }

    /// A library's pipelines load on many threads at once, each with its
    /// own description (a load with another's is refused), while others are
    /// stored and the library is serialised.
    #[test]
    fn pipelines_load_on_many_threads_at_once() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        const N: usize = 48;
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let shader = |i: usize| {
            let source = format!(
                "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n\
                 [RootSignature(RS)] [numthreads(1, 1, 1)] void main() {{ b.Store(0, b.Load(4) * {i} + {i}); }}"
            );
            let mut blob = None;
            // SAFETY: the source outlives the call; the blob is read within it.
            unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    None,
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .ok()?;
                let blob = blob?;
                Some(
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    )
                    .to_vec(),
                )
            }
        };
        let Some(codes) = (0..2 * N).map(shader).collect::<Option<Vec<_>>>() else {
            return; // no shader compiler here
        };
        // SAFETY: the code holds its root signature.
        let root: ID3D12RootSignature =
            unsafe { device.CreateRootSignature(0, &codes[0]) }.unwrap();
        struct Shared(ID3D12Device1, ID3D12RootSignature, ID3D12PipelineLibrary1);
        // SAFETY: D3D12 objects are free-threaded.
        unsafe impl Sync for Shared {}
        impl Shared {
            fn desc(&self, code: &[u8]) -> D3D12_COMPUTE_PIPELINE_STATE_DESC {
                D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    // SAFETY: a borrowed pointer the description never releases.
                    pRootSignature: unsafe { std::mem::transmute_copy(&self.1) },
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                }
            }
            fn store(&self, i: usize, code: &[u8]) {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: the description points at live data.
                unsafe {
                    let pso: ID3D12PipelineState =
                        self.0.CreateComputePipelineState(&self.desc(code)).unwrap();
                    self.2.StorePipeline(&name, &pso).unwrap();
                }
            }
            fn load(&self, i: usize, code: &[u8]) -> windows::core::Result<ID3D12PipelineState> {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: as above.
                unsafe { self.2.LoadComputePipeline(&name, &self.desc(code)) }
            }
            fn serialized(&self) -> Vec<u8> {
                // SAFETY: a buffer of the size asked for; nothing is stored
                // meanwhile.
                unsafe {
                    let mut blob = vec![0; self.2.GetSerializedSize()];
                    self.2.Serialize(&mut blob).unwrap();
                    blob
                }
            }
        }
        let first = open(&device, Vec::new()).expect("an empty library");
        let made = Shared(device.clone(), root.clone(), first.library.clone());
        (0..N).for_each(|i| made.store(i, &codes[i]));
        let again = open(&device, made.serialized()).expect("the library again");
        let shared = Shared(device, root, again.library.clone());
        let stores = Mutex::new(());
        std::thread::scope(|s| {
            for t in 0..8 {
                let (shared, codes, stores) = (&shared, &codes, &stores);
                s.spawn(move || {
                    for i in (t..N).step_by(8) {
                        shared.load(i, &codes[i]).expect("a stored pipeline");
                        assert!(
                            shared.load(i, &codes[(i + 1) % N]).is_err(),
                            "another description"
                        );
                        assert!(shared.load(N + i, &codes[N + i]).is_err(), "not stored yet");
                        let _one = stores.lock().unwrap();
                        shared.store(N + i, &codes[N + i]);
                        shared.serialized();
                    }
                });
            }
        });
        for (i, code) in codes.iter().enumerate() {
            shared.load(i, code).expect("every pipeline");
        }
    }
}
