//! What releases GPU memory waits (`diagnostics` builds): per wait, the
//! queue, the address and comparison, how long it held the queue and how
//! it was released, aggregated and logged every 5 s from the present path.
//!
//! How a release is told apart: whoever scans the registered waits
//! (`fence::scan_memory_waiters`) and finds one met records why, from the
//! scan it is in ([`Scan`], set by its caller for the thread); the queue's
//! `Timeline::run` then consumes the wait and closes the record. A wait met
//! without any scan noticing (`Timeline::submit`'s own check) counts as a
//! re-check; one met by a submission in flight as an early meet; one past
//! its deadline as a timeout. Releases noticed outside a done write are
//! split by whether the address lies in a range the last few completed
//! submissions copied back (what shaders wrote) or not (the title's CPU,
//! most likely).
//!
//! Locks: [`STATS`] is a leaf lock (taken under `MEMORY_WAITERS` and a
//! timeline's state, never taking anything itself).

use std::cell::Cell;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::{MemoryWait, Timeline};

/// The scan of registered memory waits a thread is in.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Scan {
    None,
    /// A done write (bottom-of-pipe write) of host submission `sub` (0:
    /// not a host submission's completion) at `[at, at + bytes)`.
    DoneWrite {
        sub: u64,
        at: usize,
        bytes: usize,
    },
    /// The scan after a host submission's copy-back and done writes.
    Completion {
        sub: u64,
    },
    /// The scan after a host submission was handed to the host queue.
    AfterSubmit,
    /// A fence signal (the title's CPU or a queue's signal).
    Signal,
    /// The fence watcher thread's poll.
    Watcher,
}

thread_local! {
    static SCAN: Cell<Scan> = const { Cell::new(Scan::None) };
    /// The timeline `Timeline::submit` is running after its own check.
    static RECHECK: Cell<usize> = const { Cell::new(0) };
}

/// Restores the thread's previous [`Scan`] on drop.
#[must_use]
pub(crate) struct Scanning(Scan);

impl Drop for Scanning {
    fn drop(&mut self) {
        SCAN.with(|s| s.set(self.0));
    }
}

/// The thread is in `scan` until the guard drops.
pub(crate) fn scanning(scan: Scan) -> Scanning {
    Scanning(SCAN.with(|s| s.replace(scan)))
}

/// [`scanning`] unless the thread already is in a scan.
pub(crate) fn scanning_unless_set(scan: Scan) -> Scanning {
    SCAN.with(|s| {
        let was = s.get();
        if was == Scan::None {
            s.set(scan);
        }
        Scanning(was)
    })
}

/// Clears the re-check mark on drop.
#[must_use]
pub(crate) struct Rechecking;

impl Drop for Rechecking {
    fn drop(&mut self) {
        RECHECK.with(|r| r.set(0));
    }
}

/// `Timeline::submit` found `timeline`'s memory wait met by itself.
pub(crate) fn rechecking(timeline: &Arc<Timeline>) -> Rechecking {
    RECHECK.with(|r| r.set(key(timeline)));
    Rechecking
}

/// How a wait was released. `*Cb`: the address lies in a range a recent
/// completed submission copied back; `*Cpu`: it does not.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum Kind {
    Done,
    ComplCb,
    ComplCpu,
    WatchCb,
    WatchCpu,
    SignalCb,
    SignalCpu,
    SubmitScanCb,
    SubmitScanCpu,
    RecheckCb,
    RecheckCpu,
    EarlyDone,
    EarlyShaders,
    Timeout,
    OtherCb,
    OtherCpu,
}

const KINDS: usize = 16;
const NAMES: [&str; KINDS] = [
    "done",
    "compl-cb",
    "compl-cpu",
    "watch-cb",
    "watch-cpu",
    "signal-cb",
    "signal-cpu",
    "subscan-cb",
    "subscan-cpu",
    "recheck-cb",
    "recheck-cpu",
    "early-done",
    "early-shaders",
    "timeout",
    "other-cb",
    "other-cpu",
];

/// How `Timeline::run` consumed a registered wait.
pub(crate) enum How {
    Met,
    Early { shaders: bool },
    Timeout,
}

struct Pending {
    queue: u32,
    registered: Instant,
}

#[derive(Default)]
struct PerAddress {
    count: u64,
    total_us: u64,
    max_us: u64,
    kinds: [u32; KINDS],
    queues: BTreeSet<u32>,
    comparison: u32,
    reference: u64,
    width: u8,
    /// The last submission whose done write or copy-back released it.
    last_sub: u64,
}

struct Stats {
    since: Instant,
    frames: u64,
    ids: HashMap<usize, u32>,
    pending: HashMap<usize, Pending>,
    /// Why a scan woke a timeline's wait: kind, submission.
    causes: HashMap<usize, (Kind, u64)>,
    by_address: HashMap<u64, PerAddress>,
    kinds: [(u64, u64); KINDS],
    /// Copied-back ranges `[from, to)` of the last completed submissions.
    copied: VecDeque<(u64, Vec<(usize, usize)>)>,
    /// `watch-cpu` releases: ms since the last queue call, buckets
    /// <1, <5, <20, <100, >=100 ms; sum in µs.
    gap: [u64; 5],
    gap_us: u64,
}

static STATS: Mutex<Option<Stats>> = Mutex::new(None);
/// µs since [`start`] of the last `Timeline::submit` (a title queue call).
static LAST_CALL_US: AtomicU64 = AtomicU64::new(0);

fn start() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

fn key(timeline: &Arc<Timeline>) -> usize {
    Arc::as_ptr(timeline) as usize
}

fn with<R>(f: impl FnOnce(&mut Stats) -> R) -> R {
    let mut stats = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let stats = stats.get_or_insert_with(|| Stats {
        since: Instant::now(),
        frames: 0,
        ids: HashMap::new(),
        pending: HashMap::new(),
        causes: HashMap::new(),
        by_address: HashMap::new(),
        kinds: [(0, 0); KINDS],
        copied: VecDeque::new(),
        gap: [0; 5],
        gap_us: 0,
    });
    f(stats)
}

/// A title queue call (`Timeline::submit`).
pub(crate) fn queue_called() {
    LAST_CALL_US.store(start().elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// Host submission `sub` copied back (or read back) `ranges` (address, bytes).
pub(crate) fn copied_back(sub: u64, ranges: Vec<(usize, usize)>) {
    let ranges = ranges
        .into_iter()
        .map(|(at, bytes)| (at, at.saturating_add(bytes)))
        .collect();
    with(|s| {
        s.copied.push_back((sub, ranges));
        while s.copied.len() > 4 {
            s.copied.pop_front();
        }
    });
}

impl Stats {
    /// The newest recent submission whose copy-back covers the wait's bytes.
    fn copied_by(&self, wait: &MemoryWait) -> Option<u64> {
        let from = wait.address as usize;
        let to = from.saturating_add(usize::from(wait.width));
        self.copied
            .iter()
            .rev()
            .find(|(_, ranges)| ranges.iter().any(|&(a, b)| a < to && b > from))
            .map(|&(sub, _)| sub)
    }

    /// `cb` or `cpu` by whether a copy-back covers the wait.
    fn split(&self, wait: &MemoryWait, cb: Kind, cpu: Kind) -> (Kind, u64) {
        match self.copied_by(wait) {
            Some(sub) => (cb, sub),
            None => (cpu, 0),
        }
    }
}

/// A queue is held behind `wait` (registered with the fence watcher).
pub(crate) fn held(timeline: &Arc<Timeline>) {
    let k = key(timeline);
    with(|s| {
        let next = s.ids.len() as u32;
        let queue = *s.ids.entry(k).or_insert(next);
        s.pending.entry(k).or_insert(Pending {
            queue,
            registered: Instant::now(),
        });
    });
}

/// A scan found `timeline`'s registered `wait` met.
pub(crate) fn woke(timeline: &Arc<Timeline>, wait: &MemoryWait) {
    let scan = SCAN.with(Cell::get);
    with(|s| {
        let cause = match scan {
            Scan::DoneWrite { sub, at, bytes }
                if (wait.address as usize) < at.saturating_add(bytes)
                    && wait.address as usize + usize::from(wait.width) > at =>
            {
                (Kind::Done, sub)
            }
            Scan::DoneWrite { sub, .. } | Scan::Completion { sub } => match s.copied_by(wait) {
                Some(copied) => (Kind::ComplCb, copied),
                None => (Kind::ComplCpu, sub),
            },
            Scan::AfterSubmit => s.split(wait, Kind::SubmitScanCb, Kind::SubmitScanCpu),
            Scan::Signal => s.split(wait, Kind::SignalCb, Kind::SignalCpu),
            Scan::Watcher => {
                let cause = s.split(wait, Kind::WatchCb, Kind::WatchCpu);
                if cause.0 == Kind::WatchCpu {
                    let now = start().elapsed().as_micros() as u64;
                    let gap = now.saturating_sub(LAST_CALL_US.load(Ordering::Relaxed));
                    s.gap_us += gap;
                    let bucket = match gap {
                        0..1_000 => 0,
                        1_000..5_000 => 1,
                        5_000..20_000 => 2,
                        20_000..100_000 => 3,
                        _ => 4,
                    };
                    s.gap[bucket] += 1;
                }
                cause
            }
            Scan::None => s.split(wait, Kind::OtherCb, Kind::OtherCpu),
        };
        s.causes.insert(key(timeline), cause);
    });
}

/// `Timeline::run` consumed `timeline`'s registered `wait`.
pub(crate) fn released(timeline: &Arc<Timeline>, wait: &MemoryWait, how: How) {
    let k = key(timeline);
    let recheck = RECHECK.with(Cell::get) == k;
    with(|s| {
        let cause = s.causes.remove(&k);
        let Some(pending) = s.pending.remove(&k) else {
            return;
        };
        let (kind, sub) = match how {
            How::Early { shaders: false } => (Kind::EarlyDone, 0),
            How::Early { shaders: true } => (Kind::EarlyShaders, 0),
            How::Timeout => (Kind::Timeout, 0),
            How::Met => match cause {
                Some(cause) => cause,
                None if recheck => s.split(wait, Kind::RecheckCb, Kind::RecheckCpu),
                None => s.split(wait, Kind::OtherCb, Kind::OtherCpu),
            },
        };
        let us = pending.registered.elapsed().as_micros() as u64;
        s.kinds[kind as usize].0 += 1;
        s.kinds[kind as usize].1 += us;
        let a = s.by_address.entry(wait.address).or_default();
        a.count += 1;
        a.total_us += us;
        a.max_us = a.max_us.max(us);
        a.kinds[kind as usize] += 1;
        a.queues.insert(pending.queue);
        (a.comparison, a.reference, a.width) = (wait.comparison, wait.reference, wait.width);
        if sub != 0 {
            a.last_sub = sub;
        }
    });
}

/// A frame was presented: every 5 s, logs and resets the aggregate.
pub(crate) fn frame() {
    let Some(text) = with(|s| {
        s.frames += 1;
        let elapsed = s.since.elapsed();
        if elapsed < Duration::from_secs(5) {
            return None;
        }
        let text = report(s, elapsed);
        s.since = Instant::now();
        s.frames = 0;
        s.by_address.clear();
        s.kinds = [(0, 0); KINDS];
        s.gap = [0; 5];
        s.gap_us = 0;
        Some(text)
    }) else {
        return;
    };
    for line in text {
        tracing::info!("{line}");
    }
}

fn kinds_text(kinds: impl Iterator<Item = (usize, String)>) -> String {
    let parts: Vec<String> = kinds.map(|(i, v)| format!("{} {v}", NAMES[i])).collect();
    if parts.is_empty() {
        "-".into()
    } else {
        parts.join(", ")
    }
}

fn report(s: &Stats, elapsed: Duration) -> Vec<String> {
    let frames = s.frames.max(1) as f64;
    let (count, us) = s
        .kinds
        .iter()
        .fold((0, 0), |(c, u), &(n, t)| (c + n, u + t));
    let mut lines = vec![format!(
        "memory waits by release ({:.1} s, {} frames): {:.1} held/frame, {:.1} ms/frame; per frame: {}",
        elapsed.as_secs_f64(),
        s.frames,
        count as f64 / frames,
        us as f64 / 1000.0 / frames,
        kinds_text(
            s.kinds
                .iter()
                .enumerate()
                .filter(|(_, k)| k.0 > 0)
                .map(|(i, &(n, t))| (
                    i,
                    format!(
                        "{:.1}x/{:.1}ms",
                        n as f64 / frames,
                        t as f64 / 1000.0 / frames
                    )
                ))
        ),
    )];
    let mut top: Vec<(&u64, &PerAddress)> = s.by_address.iter().collect();
    top.sort_by_key(|(_, a)| std::cmp::Reverse(a.total_us));
    for (rank, (address, a)) in top.iter().take(10).enumerate() {
        lines.push(format!(
            "memory wait top {}: {address:#x} w{} cmp {} ref {:#x} queues {:?}: n {} total {:.1} ms, mean {:.2} ms, max {:.1} ms, last sub {}; {}",
            rank + 1,
            a.width,
            a.comparison,
            a.reference,
            a.queues,
            a.count,
            a.total_us as f64 / 1000.0,
            a.total_us as f64 / 1000.0 / a.count.max(1) as f64,
            a.max_us as f64 / 1000.0,
            a.last_sub,
            kinds_text(
                a.kinds
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| **n > 0)
                    .map(|(i, n)| (i, n.to_string()))
            ),
        ));
    }
    let cpu: u64 = s.gap.iter().sum();
    if cpu > 0 {
        lines.push(format!(
            "memory waits CPU-written (watch-cpu): n {cpu}, since last queue call mean {:.2} ms; <1ms {}, <5ms {}, <20ms {}, <100ms {}, >=100ms {}",
            s.gap_us as f64 / 1000.0 / cpu as f64,
            s.gap[0],
            s.gap[1],
            s.gap[2],
            s.gap[3],
            s.gap[4],
        ));
    }
    lines
}
