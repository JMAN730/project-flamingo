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
#[cfg(any(not(windows), test))]
use std::sync::atomic::AtomicU32;
#[cfg(any(not(windows), feature = "diagnostics", test))]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

use crate::fence::Held;

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

impl MemoryWait {
    /// Whether the memory location currently satisfies the wait condition.
    ///
    /// On Windows, reads only the 4 or 8 target bytes through the kernel
    /// (`ReadProcessMemory`) instead of querying `VirtualQuery` first, avoiding
    /// multi-millisecond page-table region walks on large title allocations
    /// while still resolving uncommitted, freed, or inaccessible addresses
    /// cleanly without faulting.
    pub(crate) fn satisfied(&self) -> bool {
        if self.address == 0 {
            return true;
        }
        let width = if self.width == 4 { 4 } else { 8 };
        if self.address.checked_add(width as u64).is_none() {
            tracing::warn!(
                address = format_args!("{:#x}", self.address),
                width = self.width,
                "GPU memory wait address overflows; resolving wait"
            );
            return true;
        }
        #[cfg(windows)]
        let (lhs, rhs) = {
            let mut raw = [0u8; 8];
            if !crate::host::read_into(self.address as *const u8, &mut raw[..width]) {
                tracing::warn!(
                    address = format_args!("{:#x}", self.address),
                    width = self.width,
                    "GPU memory wait address is not committed; resolving wait"
                );
                return true;
            }
            // Pair with Release stores from CPU or bottom-of-pipe writers.
            std::sync::atomic::fence(Ordering::Acquire);
            if width == 4 {
                let val = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                (u64::from(val & self.mask), u64::from(self.reference as u32))
            } else {
                (u64::from_le_bytes(raw), self.reference)
            }
        };
        #[cfg(not(windows))]
        let (lhs, rhs) = if width == 4 {
            let val = if self.address.is_multiple_of(align_of::<AtomicU32>() as u64) {
                // SAFETY: live aligned 32-bit test memory.
                unsafe { AtomicU32::from_ptr(self.address as *mut u32) }.load(Ordering::Acquire)
            } else {
                std::sync::atomic::fence(Ordering::Acquire);
                // SAFETY: live 32-bit test memory.
                unsafe { (self.address as *const u32).read_unaligned() }
            };
            (u64::from(val & self.mask), u64::from(self.reference as u32))
        } else {
            let val = if self.address.is_multiple_of(align_of::<AtomicU64>() as u64) {
                // SAFETY: live aligned 64-bit test memory.
                unsafe { AtomicU64::from_ptr(self.address as *mut u64) }.load(Ordering::Acquire)
            } else {
                std::sync::atomic::fence(Ordering::Acquire);
                // SAFETY: live 64-bit test memory.
                unsafe { (self.address as *const u64).read_unaligned() }
            };
            (val, self.reference)
        };
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
}

/// One step of a queue's work.
pub(crate) enum Op {
    /// Hold back what follows until the fence reaches the value.
    Wait(Held, u64),
    /// Hold back what follows until the memory comparison holds (or times out).
    WaitMemory(MemoryWait, Option<Instant>),
    /// Command lists to run, a fence to signal.
    Run(Box<dyn FnOnce() + Send>),
}

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
}

impl Timeline {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Adds work to the queue and runs it unless the queue is held back.
    /// Work another thread is running for the queue is waited for, as a
    /// submission's own work is.
    pub(crate) fn submit(self: &Arc<Self>, op: Op) {
        let mut s = self.lock();
        s.ops.push_back(op);
        s.pushed += 1;
        let ticket = s.pushed;
        while s.busy && s.done < ticket {
            s = self.moved.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        if s.done >= ticket {
            return;
        }
        // Still held back unless the value was reached by a path that does
        // not wake us (a CPU write to the fence's memory).
        if s.held {
            match s.ops.front() {
                Some(Op::Wait(fence, value)) if !fence.reached(*value) => return,
                Some(Op::WaitMemory(wait, deadline))
                    if !wait.satisfied() && deadline.is_none_or(|d| Instant::now() < d) =>
                {
                    return;
                }
                _ => {}
            }
        }
        self.run(s);
    }

    /// The fence this queue waits for moved: runs what it held back.
    pub(crate) fn wake(self: &Arc<Self>) {
        let s = self.lock();
        if !s.busy {
            self.run(s);
        }
    }

    fn run<'a>(self: &'a Arc<Self>, mut s: MutexGuard<'a, State>) {
        s.busy = true;
        s.held = false;
        while let Some(op) = s.ops.pop_front() {
            match op {
                Op::Wait(fence, value) => {
                    // Checked and registered under the fence's lock, so a
                    // signal after this check wakes us.
                    if !fence.wait_on_gpu(value, self) {
                        s.ops.push_front(Op::Wait(fence, value));
                        s.held = true;
                        break;
                    }
                }
                Op::WaitMemory(wait, deadline) => {
                    let was_registered = deadline.is_some();
                    if !wait.satisfied() {
                        let deadline = deadline.unwrap_or_else(|| {
                            Instant::now() + crate::fence::memory_wait_timeout()
                        });
                        if Instant::now() < deadline
                            && !crate::fence::wait_memory(wait, deadline, self)
                        {
                            s.ops.push_front(Op::WaitMemory(wait, Some(deadline)));
                            s.held = true;
                            break;
                        }
                        if !wait.satisfied() {
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
                    if was_registered {
                        crate::fence::cancel_memory_wait(self);
                    }
                }
                Op::Run(work) => {
                    drop(s);
                    work();
                    s = self.lock();
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

    #[cfg(windows)]
    #[test]
    fn memory_wait_resolves_noaccess_committed_page_without_faulting() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc, VirtualFree,
        };
        // SAFETY: committing a single `PAGE_NOACCESS` page for read-safety verification.
        let page = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4096,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_NOACCESS,
            )
        };
        assert!(!page.is_null());
        let wait = MemoryWait {
            address: page as u64,
            reference: 1,
            mask: u32::MAX,
            width: 4,
            comparison: 3,
        };
        assert!(wait.satisfied());
        // SAFETY: freeing the page we allocated above.
        unsafe { VirtualFree(page, 0, MEM_RELEASE) };
    }

    #[cfg(windows)]
    #[test]
    fn memory_wait_reads_committed_value_inside_large_allocation() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const SIZE: usize = 16 << 20;
        // SAFETY: a 16 MiB committed allocation freed before returning.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                SIZE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!base.is_null());
        let wait = MemoryWait {
            address: base as u64,
            reference: 42,
            mask: u32::MAX,
            width: 4,
            comparison: 3,
        };
        assert!(!wait.satisfied());
        // SAFETY: aligned 32-bit cell at the start of our committed allocation.
        unsafe { (*(base as *const AtomicU32)).store(42, Ordering::Release) };
        assert!(wait.satisfied());
        // SAFETY: freeing our allocation.
        unsafe { VirtualFree(base, 0, MEM_RELEASE) };
    }
}
