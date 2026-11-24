//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and Grand
//! Theft Auto VI's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! shortcut: the budget (half a second of waiting per title frame,
//! `VICTORY_PIPELINE_WAIT_MS`) is a guess; a pass the title runs only once and whose pipeline takes longer
//! is lost.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    key: String,
    used: AtomicBool,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(
            graphics
                .as_ref()
                .map(|g| g.describe())
                .unwrap_or_default()
                .as_bytes(),
        );
        Pending {
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
        }
    }

    /// Whether a shader's container is past [`HEAVY_BYTES`].
    fn heavy(&self) -> bool {
        self.shaders.iter().any(|s| s.size() > HEAVY_BYTES)
    }

    fn make(&self) {
        let started = Instant::now();
        self.made
            .get_or_init(|| HostPipeline::new(&self.shaders, self.graphics.as_deref()));
        let took = started.elapsed();
        if took > Duration::from_secs(2) {
            tracing::info!(ms = took.as_millis() as u64, heavy = self.heavy(), shaders = ?self.shaders, "slow host pipeline");
            // With `VICTORY_DUMP_FAILED=<dir>`: their containers in `slow`,
            // to find out offline what makes the driver's compile slow.
            if let Some(dir) = std::env::var_os("VICTORY_DUMP_FAILED") {
                let dir = std::path::Path::new(&dir).join("slow");
                for s in &self.shaders {
                    let file = dir.join(format!("{}-{}.dxbc", s.stage, s.hash));
                    if !file.exists() {
                        let _ = std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&file, &s.container));
                    }
                }
            }
        }
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads: 500 ms unless
/// `VICTORY_PIPELINE_WAIT_MS` says otherwise (0: never wait, a draw whose
/// pipeline is not made yet is dropped until it is; one-off passes such as
/// a colour-grading bake may then be lost).
///
/// Without the variable, a long backlog (the world's first frames queue
/// over a thousand) cuts the budget to [`BACKLOG_BUDGET_MS`]: waiting does
/// not make the workers faster, and half a second per frame held the world
/// at 1 fps for minutes while they caught up (RTX 4060, 955-1,101 being
/// made). Frames go on with what is made; the rest comes in as it is.
fn budget() -> Duration {
    static MS: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let set = *MS.get_or_init(|| {
        std::env::var("VICTORY_PIPELINE_WAIT_MS").ok().and_then(|v| v.trim().parse().ok())
    });
    Duration::from_millis(set.unwrap_or_else(|| {
        if REQUESTED_WAITING.load(Ordering::Relaxed) > BACKLOG { BACKLOG_BUDGET_MS } else { 500 }
    }))
}

/// Pipelines a draw or dispatch asked for and not made yet (those queued
/// early, at creation, during loading, do not count: the menus' one-off
/// passes keep their whole budget then).
static REQUESTED_WAITING: AtomicU64 = AtomicU64::new(0);

/// Requested pipelines waiting past which the budget is [`BACKLOG_BUDGET_MS`].
const BACKLOG: u64 = 64;
/// The budget per frame while the backlog is long.
const BACKLOG_BUDGET_MS: u64 = 33;

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.made.get().is_none() {
        start(p, REQUESTED);
        wait(p, crate::frame::current());
    }
    let made = p.made.get()?.as_ref();
    if made.is_some() && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    made
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, EARLY);
    }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static WAITED_US: AtomicU64 = AtomicU64::new(0);
    if FRAME.swap(frame, Ordering::Relaxed) != frame {
        WAITED_US.store(0, Ordering::Relaxed);
    }
    let left = budget().saturating_sub(Duration::from_micros(WAITED_US.load(Ordering::Relaxed)));
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done
            .wait_timeout(held, rest)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    WAITED_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

/// A shader container past this size makes its pipeline heavy: the
/// world's biggest compute programs (275 of 60-78 KB in Grand Theft Auto
/// VI's first world frames) each took the RTX 4060's driver ~18 s to
/// compile. Queued with the rest, they held every worker for minutes while
/// 1,600 small draw pipelines waited behind them, so the world's draws
/// were dropped and the picture stayed black.
const HEAVY_BYTES: usize = 32 * 1024;

struct Queue {
    early: VecDeque<Arc<Pending>>,
    requested: VecDeque<Arc<Pending>>,
    /// Heavy pipelines, made by at most `heavy_limit` workers at once so
    /// the others keep making the light ones.
    heavy_early: VecDeque<Arc<Pending>>,
    heavy_requested: VecDeque<Arc<Pending>>,
    heavy_making: usize,
    heavy_limit: usize,
}

impl Queue {
    const fn new() -> Self {
        Self {
            early: VecDeque::new(),
            requested: VecDeque::new(),
            heavy_early: VecDeque::new(),
            heavy_requested: VecDeque::new(),
            heavy_making: 0,
            heavy_limit: 1,
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        let heavy = p.heavy();
        if priority == REQUESTED {
            REQUESTED_WAITING.fetch_add(1, Ordering::Relaxed);
            let lane = if heavy { &mut self.heavy_requested } else { &mut self.requested };
            lane.push_back(p.clone());
        } else {
            let lane = if heavy { &mut self.heavy_early } else { &mut self.early };
            lane.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    /// The next pipeline to make: light requested ones first, then heavy
    /// requested ones while a heavy slot is free, then those queued early.
    fn pop(&mut self) -> Option<Arc<Pending>> {
        fn early(lane: &mut VecDeque<Arc<Pending>>) -> Option<Arc<Pending>> {
            // An early entry can have been promoted and claimed already.
            std::iter::from_fn(|| lane.pop_front())
                .find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        }
        let heavy_free = self.heavy_making < self.heavy_limit;
        let requested = |lane: &mut VecDeque<Arc<Pending>>| {
            lane.pop_front().inspect(|_| {
                REQUESTED_WAITING.fetch_sub(1, Ordering::Relaxed);
            })
        };
        let p = requested(&mut self.requested)
            .or_else(|| heavy_free.then(|| requested(&mut self.heavy_requested)).flatten())
            .or_else(|| early(&mut self.early))
            .or_else(|| heavy_free.then(|| early(&mut self.heavy_early)).flatten())?;
        if p.heavy() {
            self.heavy_making += 1;
        }
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(
                WAITING.fetch_add(1, Ordering::Relaxed) + 1,
                Ordering::Relaxed,
            );
        }
        AVAILABLE.notify_one();
    }
}

fn next() -> Arc<Pending> {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop() {
            return p;
        }
        queue = AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// A worker is done with `p`: frees its heavy slot.
fn finished(p: &Pending) {
    if p.heavy() {
        let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
        queue.heavy_making = queue.heavy_making.saturating_sub(1);
        AVAILABLE.notify_all();
    }
}

/// Starts the workers on first use: two thirds of the logical processors,
/// 2 to 10, half of them (1 to 3: the driver ran out of memory with six
/// heavy compiles at once) allowed on heavy pipelines.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = std::thread::available_parallelism().map_or(2, |n| (n.get() * 2 / 3).clamp(2, 10));
        let heavy = (workers / 2).clamp(1, 3);
        QUEUE.lock().unwrap_or_else(|e| e.into_inner()).heavy_limit = heavy;
        tracing::info!(workers, heavy, "pipeline workers");
        for i in 0..workers {
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(|| {
                    let mut raised = None;
                    loop {
                        let p = next();
                        // Below the title's own threads (loading, recording)
                        // unless draws are waiting on a long backlog: the
                        // title then spins at full load and the workers
                        // starved (4-9 pipelines a minute in the world).
                        let raise = REQUESTED_WAITING.load(Ordering::Relaxed) > BACKLOG;
                        if raised != Some(raise) {
                            raised = Some(raise);
                            // SAFETY: a plain call on our own thread.
                            unsafe {
                                use windows::Win32::System::Threading::*;
                                let _ = SetThreadPriority(
                                    GetCurrentThread(),
                                    if raise { THREAD_PRIORITY_NORMAL } else { THREAD_PRIORITY_BELOW_NORMAL },
                                );
                            }
                        }
                        p.make();
                        finished(&p);
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save(false);
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            } else {
                running = true;
            }
        }
        running
    })
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &earlier));
        assert!(queue.pop().is_none());
    }

    /// Heavy pipelines wait while their slots are taken; light ones and
    /// those queued early go on past them.
    #[test]
    fn heavy_pipelines_do_not_hold_up_light_ones() {
        let big = super::super::shader("compute", &vec![7u8; HEAVY_BYTES + 1]);
        let heavy = || Arc::new(Pending::new(vec![big.clone()], None));
        let (h1, h2) = (heavy(), heavy());
        let light = Arc::new(Pending::new(Vec::new(), None));
        let early = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();
        assert_eq!(queue.heavy_limit, 1);

        queue.push(&h1, REQUESTED);
        queue.push(&h2, REQUESTED);
        queue.push(&early, EARLY);
        queue.push(&light, REQUESTED);
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &light));
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &h1));
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &early));
        assert!(queue.pop().is_none(), "the heavy slot is taken");
        queue.heavy_making -= 1;
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &h2));
    }
}
