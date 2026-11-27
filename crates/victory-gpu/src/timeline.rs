//! A queue's timeline: its work in submission order, held back behind a
//! GPU-side wait (`ID3D12CommandQueue::Wait`) until the fence reaches the
//! value.
//!
//! Work runs on the thread that submits it, as before, unless the queue is
//! held back: then the submission returns at once and the work runs later,
//! on the thread that moves the fence (a signal or a bottom-of-pipe write
//! from another queue, or the title's CPU signal). Nothing blocks a
//! submitting thread behind a wait, so a thread may submit a queue's wait
//! before the signal it waits for.
//!
//! Locks: a timeline's state is taken before a fence's waiters, never
//! after; work runs with no lock held.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

use crate::fence::Held;

#[cfg(feature = "diagnostics")]
pub(crate) mod wait_stats;

/// A GPU wait for a memory location to satisfy a D3D12 comparison.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MemoryWait {
    pub(crate) address: u64,
    pub(crate) reference: u64,
    pub(crate) mask: u32,
    pub(crate) width: u8,
    pub(crate) comparison: u32,
}

/// Count of GPU memory waits that timed out and failed open (diagnostics).
#[cfg(feature = "diagnostics")]
static MEMORY_WAIT_TIMEOUTS: AtomicU64 = AtomicU64::new(0);

/// Total GPU memory waits that have timed out so far (diagnostics).
#[cfg(feature = "diagnostics")]
#[allow(dead_code)]
pub(crate) fn memory_wait_timeouts() -> u64 {
    MEMORY_WAIT_TIMEOUTS.load(Ordering::Relaxed)
}

/// Whether `[address, address + width)` is committed, remembering pages
/// found committed for up to a second. Waits are checked on every memory
/// write notification; asking the kernel each time (`VirtualQuery`) took
/// most of a world frame's CPU time. A page the title decommits is asked
/// again within a second.
#[cfg(windows)]
fn committed_recently(address: u64, width: usize) -> bool {
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    static SEEN: Mutex<Option<(Instant, HashSet<u64>)>> = Mutex::new(None);
    let first = address >> 12;
    let last = (address + width as u64 - 1) >> 12;
    {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let (since, pages) = seen.get_or_insert_with(|| (Instant::now(), HashSet::new()));
        if since.elapsed() > Duration::from_secs(1) {
            *since = Instant::now();
            pages.clear();
        }
        if (first..=last).all(|p| pages.contains(&p)) {
            return true;
        }
    }
    if !crate::texture::committed(address as *const u8, width) {
        return false;
    }
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, pages)) = seen.as_mut() {
        pages.extend(first..=last);
    }
    true
}

impl MemoryWait {
    /// Whether the memory location currently satisfies the wait condition.
    ///
    /// On Windows, checks that `[address, address + width)` is committed before
    /// reading so a freed or decommitted title address resolves cleanly instead
    /// of faulting the process.
    pub(crate) fn satisfied(&self) -> bool {
        if self.address == 0 {
            return true;
        }
        let width = usize::from(self.width);
        if self.address.checked_add(width as u64).is_none() {
            tracing::warn!(
                address = format_args!("{:#x}", self.address),
                width = self.width,
                "GPU memory wait address overflows; resolving wait"
            );
            return true;
        }
        #[cfg(windows)]
        if !committed_recently(self.address, width) {
            tracing::warn!(
                address = format_args!("{:#x}", self.address),
                width = self.width,
                "GPU memory wait address is not committed; resolving wait"
            );
            return true;
        }
        let (lhs, rhs) = if self.width == 4 {
            let val = if self.address.is_multiple_of(align_of::<AtomicU32>() as u64) {
                // SAFETY: verified committed on Windows (or live test memory); aligned 32-bit read.
                unsafe { AtomicU32::from_ptr(self.address as *mut u32) }.load(Ordering::Acquire)
            } else {
                std::sync::atomic::fence(Ordering::Acquire);
                // SAFETY: verified committed on Windows (or live test memory); 32-bit read.
                unsafe { (self.address as *const u32).read_unaligned() }
            };
            (u64::from(val & self.mask), u64::from(self.reference as u32))
        } else {
            let val = if self.address.is_multiple_of(align_of::<AtomicU64>() as u64) {
                // SAFETY: verified committed on Windows (or live test memory); aligned 64-bit read.
                unsafe { AtomicU64::from_ptr(self.address as *mut u64) }.load(Ordering::Acquire)
            } else {
                std::sync::atomic::fence(Ordering::Acquire);
                // SAFETY: verified committed on Windows (or live test memory); 64-bit read.
                unsafe { (self.address as *const u64).read_unaligned() }
            };
            (val, self.reference)
        };
        self.holds(lhs, rhs)
    }

    /// Whether the (masked) memory value `lhs` against the (cut) reference
    /// `rhs` meets the comparison.
    fn holds(&self, lhs: u64, rhs: u64) -> bool {
        match self.comparison {
            2 => lhs < rhs,
            3 => lhs == rhs,
            4 => lhs <= rhs,
            5 => lhs > rhs,
            6 => lhs != rhs,
            7 => lhs >= rhs,
            _ => true,
        }
    }

    /// Whether `value`, stored by a `bytes`-byte write at the wait's
    /// address, would meet it (None: the write does not cover the wait's
    /// bytes as it reads them).
    pub(crate) fn met_by(&self, bytes: usize, value: u64) -> Option<bool> {
        match (self.width, bytes) {
            (4, 4 | 8) => Some(self.holds(
                u64::from(value as u32 & self.mask),
                u64::from(self.reference as u32),
            )),
            (8, 8) => Some(self.holds(value, self.reference)),
            _ => None,
        }
    }
}

/// Host submissions in flight (`VICTORY_IN_FLIGHT`, see
/// `host::in_flight_limit`): the host fence value of the last submission
/// whose completion (copy-back, done writes) has run. Completions run in
/// submission order, so every submission up to it is complete.
static COMPLETED: AtomicU64 = AtomicU64::new(0);

/// Queues holding work back until [`COMPLETED`] reaches a value. Leaf lock,
/// taken under a timeline's state, never the other way round.
static COMPLETION_WAITERS: Mutex<Vec<(u64, std::sync::Weak<Timeline>)>> = Mutex::new(Vec::new());

/// The done writes (bottom-of-pipe writes of known values) of host
/// submissions in flight: (host fence value, address, bytes, value), in
/// submission order. Leaf lock.
static FLYING_WRITES: Mutex<Vec<(u64, usize, usize, u64)>> = Mutex::new(Vec::new());
/// Whether [`FLYING_WRITES`] has entries (so waits need not lock it).
static FLYING: AtomicU32 = AtomicU32::new(0);

/// GPU memory waits met by a submission in flight before it completed, by
/// its done writes and by what its shaders wrote, since the frame-rate
/// line last read them.
pub(crate) static MET_IN_FLIGHT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A submission (host fence value `value`) was handed to the host queue;
/// `writes` are its done writes of known values (address, bytes, value).
pub(crate) fn fly_writes(value: u64, writes: impl IntoIterator<Item = (usize, usize, u64)>) {
    let mut flying = FLYING_WRITES.lock().unwrap_or_else(|e| e.into_inner());
    flying.extend(
        writes
            .into_iter()
            .map(|(at, bytes, v)| (value, at, bytes, v)),
    );
    FLYING.store(u32::from(!flying.is_empty()), Ordering::Release);
}

/// The submission's done writes landed in title memory.
pub(crate) fn land_writes(value: u64) {
    let mut flying = FLYING_WRITES.lock().unwrap_or_else(|e| e.into_inner());
    flying.retain(|w| w.0 != value);
    FLYING.store(u32::from(!flying.is_empty()), Ordering::Release);
}

/// The done writes in flight that touch `[from, to)`: (host fence value,
/// address, bytes, value).
pub(crate) fn flying_writes(from: usize, to: usize) -> Vec<(u64, usize, usize, u64)> {
    if FLYING.load(Ordering::Acquire) == 0 {
        return Vec::new();
    }
    let flying = FLYING_WRITES.lock().unwrap_or_else(|e| e.into_inner());
    flying
        .iter()
        .filter(|&&(_, at, bytes, _)| at < to && at + bytes > from)
        .copied()
        .collect()
}

/// The first submission in flight whose done write meets `wait` (its host
/// fence value), if any.
///
/// Why a queue may go on behind such a wait before that submission
/// completes: the wait only orders GPU work. The host has one in-order
/// queue and that submission is already on it, so whatever the waiting
/// queue submits next runs on the GPU after it (ExecuteCommandLists
/// boundaries are full barriers on the host: work and caches of one call
/// are done before the next starts). What must not move ahead is what a
/// CPU observes: the waiting queue's later non-GPU work (fence signals,
/// presents, page mappings) is held until that submission has completed
/// ([`Timeline`]'s `need`), and a later submission whose preparation reads,
/// on the CPU, title memory that one writes at completion first waits for
/// it (`host`'s landing check); where the GPU reads such a value through a
/// host copy, the copy is given it (`host`'s refresh patch). The title's
/// CPU never sees the value early: title memory gets it at completion, as
/// before.
///
/// Values shaders of a submission in flight write count too, once they
/// are in its host copy (`host::shader_meets`): the same ordering holds.
pub(crate) fn in_flight_meets(wait: &MemoryWait) -> Option<u64> {
    met_in_flight(wait).map(|(value, _)| value)
}

/// [`in_flight_meets`], and whether shaders (not a done write) met it.
fn met_in_flight(wait: &MemoryWait) -> Option<(u64, bool)> {
    if wait.address == 0 {
        return None;
    }
    if FLYING.load(Ordering::Acquire) != 0 {
        let flying = FLYING_WRITES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(value) = first_meeting(wait, &flying) {
            return Some((value, false));
        }
    }
    #[cfg(windows)]
    if let Some(value) = crate::host::shader_meets(wait) {
        return Some((value, true));
    }
    None
}

/// [`in_flight_meets`] over `writes` (in submission order).
fn first_meeting(wait: &MemoryWait, writes: &[(u64, usize, usize, u64)]) -> Option<u64> {
    writes
        .iter()
        .filter(|w| w.1 as u64 == wait.address)
        .find(|w| wait.met_by(w.2, w.3) == Some(true))
        .map(|w| w.0)
}

/// The host fence value of the last submission completed in flight.
pub(crate) fn completed_value() -> u64 {
    COMPLETED.load(Ordering::Acquire)
}

/// Submissions up to host fence value `value` have completed: the queues
/// that waited for it, to wake once no lock is held.
pub(crate) fn completed(value: u64) -> Vec<Arc<Timeline>> {
    COMPLETED.fetch_max(value, Ordering::AcqRel);
    let now = COMPLETED.load(Ordering::Acquire);
    let mut woken = Vec::new();
    let mut waiters = COMPLETION_WAITERS.lock().unwrap_or_else(|e| e.into_inner());
    waiters.retain(|(wanted, queue)| {
        if *wanted <= now {
            woken.extend(queue.upgrade());
            false
        } else {
            queue.strong_count() > 0
        }
    });
    woken
}

/// Whether submissions up to `value` have completed; if not, `queue` is
/// woken once they have (checked under the waiters' lock, so no completion
/// is missed).
fn completion_reached(value: u64, queue: &Arc<Timeline>) -> bool {
    if value == 0 || COMPLETED.load(Ordering::Acquire) >= value {
        return true;
    }
    let mut waiters = COMPLETION_WAITERS.lock().unwrap_or_else(|e| e.into_inner());
    if COMPLETED.load(Ordering::Acquire) >= value {
        return true;
    }
    let weak = Arc::downgrade(queue);
    match waiters.iter_mut().find(|(_, q)| q.ptr_eq(&weak)) {
        Some(existing) => existing.0 = existing.0.max(value),
        None => waiters.push((value, weak)),
    }
    false
}

/// One step of a queue's work.
pub(crate) enum Op {
    /// Hold back what follows until the fence reaches the value.
    Wait(Held, u64),
    /// Hold back what follows until the memory comparison holds (or times out).
    WaitMemory(MemoryWait, Option<Instant>),
    /// Command lists to run, a fence to signal.
    Run(Box<dyn FnOnce() + Send>),
    /// Work that finishes later (a host submission the GPU still runs):
    /// what follows is held back until it calls the resume it is given.
    RunAsync(Box<dyn FnOnce(Resume) + Send>),
    /// A host submission left in flight (`VICTORY_IN_FLIGHT`): the work
    /// prepares and submits it and returns its host fence value (0: none).
    /// What follows runs at once if it is GPU work too (another
    /// `RunInFlight`) or a wait; [`Op::Run`] and [`Op::RunAsync`] (fence
    /// signals, presents, page mappings: what the title's CPU sees) are
    /// held until the submission has completed.
    RunInFlight(Box<dyn FnOnce() -> u64 + Send>),
}

/// Called once when an [`Op::RunAsync`] has finished.
pub(crate) type Resume = Box<dyn FnOnce() + Send>;

#[derive(Default)]
pub(crate) struct Timeline {
    state: Mutex<State>,
    /// Signalled whenever work is done or the queue stops running it.
    moved: Condvar,
}

#[derive(Default)]
struct State {
    ops: VecDeque<Op>,
    /// Ops submitted and ops done (a submission's ticket is its count).
    pushed: u64,
    done: u64,
    /// A thread is running the queue's work.
    busy: bool,
    /// The first op is a wait not reached yet, registered with its fence.
    held: bool,
    /// An [`Op::RunAsync`] started and has not resumed yet.
    running_async: bool,
    /// Since when, and why ([`HELD_US`] index), the queue has been held.
    held_since: Option<(Instant, usize)>,
    /// The host fence value whose submission must have completed before
    /// the queue's next [`Op::Run`] or [`Op::RunAsync`]: its own
    /// submissions in flight, and those whose done writes met its memory
    /// waits early (0: none).
    need: u64,
}

/// Microseconds queues spent held, summed over queues: behind a fence wait,
/// a GPU memory wait, a submission still finishing; then running work, and
/// time inside the driver's ExecuteCommandLists. The frame-rate line reads
/// and resets them.
pub(crate) static HELD_US: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

impl Timeline {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Adds work to the queue and runs it unless the queue is held back.
    /// Work another thread is running for the queue is waited for, as a
    /// submission's own work is.
    pub(crate) fn submit(self: &Arc<Self>, op: Op) {
        // `VICTORY_ASYNC_SUBMIT` 3 to 5: a queue call finishes submission
        // work left for the title's threads first (nothing otherwise).
        #[cfg(windows)]
        crate::host::run_pending();
        let called = Instant::now();
        #[cfg(feature = "diagnostics")]
        wait_stats::queue_called();
        let mut s = self.lock();
        s.ops.push_back(op);
        s.pushed += 1;
        let ticket = s.pushed;
        while s.busy && s.done < ticket {
            s = self.moved.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        let _ = called;
        if s.done >= ticket {
            return;
        }
        // Behind work that finishes later: its resume runs what follows.
        if s.running_async {
            return;
        }
        // Still held back unless the value was reached by a path that does
        // not wake us (a CPU write to the fence's memory).
        if s.held {
            match s.ops.front() {
                Some(Op::Wait(fence, value)) if !fence.reached(*value) => return,
                Some(Op::WaitMemory(wait, deadline))
                    if !wait.satisfied()
                        && in_flight_meets(wait).is_none()
                        && deadline.is_none_or(|d| Instant::now() < d) =>
                {
                    return;
                }
                Some(Op::Run(_) | Op::RunAsync(_)) if !completion_reached(s.need, self) => return,
                _ => {}
            }
        }
        #[cfg(feature = "diagnostics")]
        let _recheck = wait_stats::rechecking(self);
        self.run(s);
    }

    /// The fence this queue waits for moved: runs what it held back.
    pub(crate) fn wake(self: &Arc<Self>) {
        let s = self.lock();
        if !s.busy && !s.running_async {
            self.run(s);
        }
    }

    /// An [`Op::RunAsync`] finished: counts it done and runs what it held
    /// back (unless the thread that started it is still running the queue,
    /// which goes on by itself).
    fn resume(self: &Arc<Self>) {
        let mut s = self.lock();
        s.running_async = false;
        s.done += 1;
        self.moved.notify_all();
        if !s.busy {
            self.run(s);
        }
    }

    fn run<'a>(self: &'a Arc<Self>, mut s: MutexGuard<'a, State>) {
        s.busy = true;
        s.held = false;
        if let Some((since, why)) = s.held_since.take() {
            HELD_US[why].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        loop {
            // Behind submissions in flight: what the title's CPU sees waits
            // for their completion.
            if matches!(s.ops.front(), Some(Op::Run(_) | Op::RunAsync(_)))
                && !completion_reached(s.need, self)
            {
                s.held = true;
                s.held_since = Some((Instant::now(), 2));
                break;
            }
            let Some(op) = s.ops.pop_front() else {
                break;
            };
            match op {
                Op::Wait(fence, value) => {
                    // Checked and registered under the fence's lock, so a
                    // signal after this check wakes us.
                    if !fence.wait_on_gpu(value, self) {
                        s.ops.push_front(Op::Wait(fence, value));
                        s.held = true;
                        s.held_since = Some((Instant::now(), 0));
                        break;
                    }
                }
                Op::WaitMemory(wait, deadline) => {
                    let was_registered = deadline.is_some();
                    let met = wait.satisfied();
                    if !met && let Some((value, shaders)) = met_in_flight(&wait) {
                        // Met by a submission already on the host queue.
                        s.need = s.need.max(value);
                        MET_IN_FLIGHT[usize::from(shaders)].fetch_add(1, Ordering::Relaxed);
                        #[cfg(feature = "diagnostics")]
                        if was_registered {
                            wait_stats::released(self, &wait, wait_stats::How::Early { shaders });
                        }
                    } else if !met {
                        let deadline = deadline.unwrap_or_else(|| {
                            Instant::now() + crate::fence::memory_wait_timeout()
                        });
                        if Instant::now() < deadline
                            && !crate::fence::wait_memory(wait, deadline, self)
                        {
                            s.ops.push_front(Op::WaitMemory(wait, Some(deadline)));
                            s.held = true;
                            s.held_since = Some((Instant::now(), 1));
                            #[cfg(feature = "diagnostics")]
                            wait_stats::held(self);
                            break;
                        }
                        let timed_out = !wait.satisfied();
                        #[cfg(feature = "diagnostics")]
                        if was_registered {
                            let how = if timed_out {
                                wait_stats::How::Timeout
                            } else {
                                wait_stats::How::Met
                            };
                            wait_stats::released(self, &wait, how);
                        }
                        if timed_out {
                            // Fail-open is intentional: if a console memory wait is
                            // never satisfied (for example, an unimplemented signal
                            // path), continuing after the 2 s stall avoids permanently
                            // deadlocking the queue even though subsequent work runs
                            // with the condition unmet.
                            #[cfg(feature = "diagnostics")]
                            let timeouts = MEMORY_WAIT_TIMEOUTS.fetch_add(1, Ordering::Relaxed) + 1;
                            #[cfg(feature = "diagnostics")]
                            tracing::warn!(
                                address = format_args!("{:#x}", wait.address),
                                comparison = wait.comparison,
                                reference = wait.reference,
                                width = wait.width,
                                timeouts,
                                "GPU memory wait timed out; continuing (fail-open)"
                            );
                            #[cfg(not(feature = "diagnostics"))]
                            tracing::warn!(
                                address = format_args!("{:#x}", wait.address),
                                comparison = wait.comparison,
                                reference = wait.reference,
                                width = wait.width,
                                "GPU memory wait timed out; continuing (fail-open)"
                            );
                        }
                    }
                    #[cfg(feature = "diagnostics")]
                    if was_registered && met {
                        wait_stats::released(self, &wait, wait_stats::How::Met);
                    }
                    if was_registered {
                        crate::fence::cancel_memory_wait(self);
                    }
                }
                Op::Run(work) => {
                    drop(s);
                    let started = Instant::now();
                    work();
                    HELD_US[3].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                    s = self.lock();
                }
                Op::RunInFlight(work) => {
                    drop(s);
                    let started = Instant::now();
                    let value = work();
                    HELD_US[3].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                    s = self.lock();
                    s.need = s.need.max(value);
                }
                Op::RunAsync(work) => {
                    s.running_async = true;
                    drop(s);
                    let me = Arc::clone(self);
                    work(Box::new(move || me.resume()));
                    s = self.lock();
                    if s.running_async {
                        // Finishes later; its resume runs the rest.
                        s.held_since = Some((Instant::now(), 2));
                        break;
                    }
                    // It finished already, and its resume counted it.
                    continue;
                }
            }
            s.done += 1;
            self.moved.notify_all();
        }
        s.busy = false;
        self.moved.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    #[test]
    fn memory_wait_comparisons_and_32_bit_mask() {
        let val32 = AtomicU32::new(0xabcd_0005);
        let addr32 = (&raw const val32) as u64;
        let check32 = |comparison, reference, mask| {
            MemoryWait {
                address: addr32,
                reference,
                mask,
                width: 4,
                comparison,
            }
            .satisfied()
        };
        // Masked value is 5.
        assert!(!check32(2, 5, 0xffff)); // 5 < 5
        assert!(check32(2, 6, 0xffff)); // 5 < 6
        assert!(check32(3, 5, 0xffff)); // 5 == 5
        assert!(!check32(3, 6, 0xffff)); // 5 == 6
        assert!(check32(4, 5, 0xffff)); // 5 <= 5
        assert!(!check32(4, 4, 0xffff)); // 5 <= 4
        assert!(check32(5, 4, 0xffff)); // 5 > 4
        assert!(!check32(5, 5, 0xffff)); // 5 > 5
        assert!(check32(6, 4, 0xffff)); // 5 != 4
        assert!(!check32(6, 5, 0xffff)); // 5 != 5
        assert!(check32(7, 5, 0xffff)); // 5 >= 5
        assert!(!check32(7, 6, 0xffff)); // 5 >= 6
        // Unmasked checks all 32 bits, ignoring upper 32 bits of reference.
        assert!(check32(3, 0x1_0000_0000 | 0xabcd_0005, u32::MAX));
        // 1 (NEVER) and 8 (ALWAYS) fallback to true in MemoryWait::satisfied.
        assert!(check32(1, 999, 0xffff));
        assert!(check32(8, 999, 0xffff));

        // 64-bit aligned & unaligned.
        let val64 = AtomicU64::new(0x0102_0304_0506_0708);
        assert!(
            MemoryWait {
                address: (&raw const val64) as u64,
                reference: 0x0102_0304_0506_0708,
                mask: 0,
                width: 8,
                comparison: 3,
            }
            .satisfied()
        );
        let mut buf = [0u8; 16];
        buf[1..9].copy_from_slice(&0x1122_3344_5566_7788_u64.to_le_bytes());
        assert!(
            MemoryWait {
                address: buf[1..].as_ptr() as u64,
                reference: 0x1122_3344_5566_7788,
                mask: 0,
                width: 8,
                comparison: 3,
            }
            .satisfied()
        );
        assert!(
            MemoryWait {
                address: buf[1..].as_ptr() as u64,
                reference: 0x5566_7788,
                mask: u32::MAX,
                width: 4,
                comparison: 3,
            }
            .satisfied()
        );
    }

    #[test]
    fn a_done_write_in_flight_meets_a_wait_as_the_memory_would() {
        let wait = |width, comparison, reference| MemoryWait {
            address: 0x1000,
            reference,
            mask: u32::MAX,
            width,
            comparison,
        };
        let writes = [
            (5, 0x1000, 4, 3),
            (6, 0x1000, 8, 0x1_0000_0009),
            (7, 0x2000, 4, 9),
        ];
        // The first submission whose write meets it: the low 32 bits of
        // the 8-byte write are 9.
        assert_eq!(first_meeting(&wait(4, 7, 9), &writes), Some(6));
        assert_eq!(first_meeting(&wait(4, 7, 3), &writes), Some(5));
        assert_eq!(first_meeting(&wait(8, 3, 0x1_0000_0009), &writes), Some(6));
        // A 4-byte write does not cover an 8-byte wait; nothing meets 10.
        assert_eq!(first_meeting(&wait(8, 7, 0), &writes), Some(6));
        assert_eq!(first_meeting(&wait(8, 3, 3), &writes), None);
        assert_eq!(first_meeting(&wait(4, 7, 10), &writes), None);
    }

    /// Behind a memory wait a submission in flight meets, a queue goes on
    /// with GPU work at once, but its fence signals (what the title's CPU
    /// sees) wait until every submission it is behind has completed.
    #[test]
    fn work_behind_submissions_in_flight_waits_for_their_completion() {
        let timeline = Arc::new(Timeline::default());
        let cell = AtomicU32::new(0);
        let address = (&raw const cell) as u64;
        // Far above any value another test completes.
        let value = 1 << 62;
        fly_writes(value, [(address as usize, 4, 9)]);
        let wait = MemoryWait {
            address,
            reference: 7,
            mask: u32::MAX,
            width: 4,
            comparison: 7,
        };
        let met = MET_IN_FLIGHT[0].load(Ordering::Relaxed);
        timeline.submit(Op::WaitMemory(
            wait,
            Some(Instant::now() + Duration::from_secs(5)),
        ));
        let submitted = Arc::new(AtomicBool::new(false));
        let flag = submitted.clone();
        timeline.submit(Op::RunInFlight(Box::new(move || {
            flag.store(true, Ordering::Release);
            value + 1
        })));
        assert!(
            submitted.load(Ordering::Acquire),
            "GPU work goes on behind a wait met in flight"
        );
        assert!(MET_IN_FLIGHT[0].load(Ordering::Relaxed) > met);
        assert!(!crate::fence::has_memory_waiter(&timeline));
        let signalled = Arc::new(AtomicBool::new(false));
        let flag = signalled.clone();
        timeline.submit(Op::Run(Box::new(move || {
            flag.store(true, Ordering::Release);
        })));
        assert!(!signalled.load(Ordering::Acquire), "held behind both");
        land_writes(value);
        for queue in completed(value) {
            queue.wake();
        }
        assert!(
            !signalled.load(Ordering::Acquire),
            "still behind its own submission"
        );
        for queue in completed(value + 1) {
            queue.wake();
        }
        assert!(signalled.load(Ordering::Acquire));
        assert_eq!(cell.load(Ordering::Relaxed), 0, "memory itself untouched");
    }

    #[test]
    fn memory_wait_timeout_unblocks_stalled_wait() {
        let timeline = Arc::new(Timeline::default());
        let never_written = AtomicU32::new(0);
        let ran = Arc::new(AtomicBool::new(false));
        let wait = MemoryWait {
            address: (&raw const never_written) as u64,
            reference: 1,
            mask: u32::MAX,
            width: 4,
            comparison: 3,
        };
        timeline.submit(Op::WaitMemory(
            wait,
            Some(Instant::now() + Duration::from_millis(20)),
        ));
        let ran_work = ran.clone();
        timeline.submit(Op::Run(Box::new(move || {
            ran_work.store(true, Ordering::Release);
        })));
        assert!(!ran.load(Ordering::Acquire));

        let limit = Instant::now() + Duration::from_secs(2);
        while !ran.load(Ordering::Acquire) && Instant::now() < limit {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            ran.load(Ordering::Acquire),
            "timed-out memory wait did not advance timeline"
        );
        #[cfg(feature = "diagnostics")]
        assert!(memory_wait_timeouts() >= 1);
    }

    #[cfg(windows)]
    #[test]
    fn memory_wait_resolves_uncommitted_address_without_faulting() {
        use windows_sys::Win32::System::Memory::{
            MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc, VirtualFree,
        };
        // SAFETY: reserving a single uncommitted page for read-safety verification.
        let reserved = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!reserved.is_null());
        let wait32 = MemoryWait {
            address: reserved as u64,
            reference: 1,
            mask: u32::MAX,
            width: 4,
            comparison: 3,
        };
        let wait64 = MemoryWait {
            address: reserved as u64,
            reference: 1,
            mask: u32::MAX,
            width: 8,
            comparison: 3,
        };
        assert!(wait32.satisfied());
        assert!(wait64.satisfied());
        // SAFETY: freeing the page we reserved above.
        unsafe { VirtualFree(reserved, 0, MEM_RELEASE) };
        // Freed (unmapped) address must also resolve cleanly without faulting.
        assert!(wait32.satisfied());
        assert!(wait64.satisfied());
    }

    #[test]
    fn consuming_satisfied_memory_wait_unregisters_waiter_before_post_wait_work() {
        let timeline = Arc::new(Timeline::default());
        let cell = AtomicU32::new(0);
        let wait = MemoryWait {
            address: (&raw const cell) as u64,
            reference: 1,
            mask: u32::MAX,
            width: 4,
            comparison: 3,
        };
        timeline.submit(Op::WaitMemory(
            wait,
            Some(Instant::now() + Duration::from_secs(5)),
        ));
        assert!(crate::fence::has_memory_waiter(&timeline));

        // Satisfy on the CPU and immediately submit post-wait work so
        // `Timeline::submit` consumes the wait directly (without waiting for
        // the background watcher scan).
        cell.store(1, Ordering::Release);
        let tl_check = timeline.clone();
        let saw_registered = Arc::new(AtomicBool::new(true));
        let saw_registered_work = saw_registered.clone();
        timeline.submit(Op::Run(Box::new(move || {
            saw_registered_work.store(
                crate::fence::has_memory_waiter(&tl_check),
                Ordering::Release,
            );
        })));
        assert!(
            !saw_registered.load(Ordering::Acquire),
            "MEMORY_WAITERS entry must be unregistered before post-wait work runs"
        );
        assert!(!crate::fence::has_memory_waiter(&timeline));
    }
}
