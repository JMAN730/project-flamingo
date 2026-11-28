//! The title's thread creation (`CreateThread`, `CreateRemoteThread[Ex]`,
//! `_beginthread[ex]`), passed to the host unchanged, with the stack each
//! thread asks for logged: our HLE and the GPU replay run on title threads,
//! inside whatever stack the title reserved for them.
//!
//! * Each distinct size and flag the title passes logs once at debug
//!   (`victory_hle=debug`), with the reserve Windows gives it.
//! * A reserve under [`REPLAY_STACK`] warns, once per start routine, with
//!   that routine and the caller as `module+rva`.
//!
//! The size is a commit size unless the flags have
//! `STACK_SIZE_PARAM_IS_A_RESERVATION`; a commit size leaves the reserve at
//! the executable's (16 MiB, `apps/victory/build.rs`) unless it is larger.

#![allow(non_snake_case)]

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Mutex;

/// `STACK_SIZE_PARAM_IS_A_RESERVATION`: the size is the reserve.
const RESERVATION: u32 = 0x1_0000;

/// The least stack reserve a title thread can run our code in: the
/// linker's default reserve, which host drivers were found to overflow
/// (`apps/victory/build.rs`), and the replay calls into them on the
/// title's thread.
const REPLAY_STACK: usize = 1 << 20;

/// The reserve Windows gives a thread asked for `size` bytes with `flags`,
/// in a process whose executable reserves `exe` (rounded to 64 KiB, the
/// allocation granularity; a commit size larger than the reserve takes a
/// reserve of it rounded up to 1 MiB).
fn reserve_of(size: usize, flags: u32, exe: usize) -> usize {
    const GRANULE: usize = 64 << 10;
    const MIB: usize = 1 << 20;
    let reserve = match size {
        0 => exe,
        size if flags & RESERVATION != 0 => size,
        size if size > exe => size.div_ceil(MIB) * MIB,
        _ => exe,
    };
    reserve.div_ceil(GRANULE) * GRANULE
}

/// What [`note`] logs for a thread: whether this size and flag are new
/// (log at debug) and whether this start routine is to be warned about.
fn first_sight(
    seen: &mut HashSet<(usize, bool)>,
    warned: &mut HashSet<usize>,
    size: usize,
    flags: u32,
    reserve: usize,
    start: usize,
) -> (bool, bool) {
    let new = seen.insert((size, flags & RESERVATION != 0));
    let warn = reserve < REPLAY_STACK && warned.insert(start);
    (new, warn)
}

/// Logs the stack a title thread asks for (see the module docs).
fn note(size: usize, flags: u32, start: usize) {
    struct Seen {
        sizes: HashSet<(usize, bool)>,
        starts: HashSet<usize>,
    }
    static SEEN: Mutex<Option<Seen>> = Mutex::new(None);
    let reserve = reserve_of(size, flags, host::exe_reserve());
    let (new, warn) = {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let seen = seen.get_or_insert_with(|| Seen {
            sizes: HashSet::new(),
            starts: HashSet::new(),
        });
        first_sight(
            &mut seen.sizes,
            &mut seen.starts,
            size,
            flags,
            reserve,
            start,
        )
    };
    if new {
        tracing::debug!(
            size,
            reservation = flags & RESERVATION != 0,
            reserve_kib = reserve >> 10,
            start = %host::locate(start),
            "a title thread's stack"
        );
    }
    if warn {
        tracing::warn!(
            reserve_kib = reserve >> 10,
            least_kib = REPLAY_STACK >> 10,
            start = %host::locate(start),
            from = %host::callers(),
            "a title thread reserves less stack than our code running on it may need"
        );
    }
}

/// A thread's start routine, as the system calls it.
pub(crate) type Start = Option<unsafe extern "system" fn(*mut c_void) -> u32>;
/// `_beginthreadex`'s start routine.
pub(crate) type CrtStartEx = Option<unsafe extern "system" fn(*mut c_void) -> u32>;
/// `_beginthread`'s start routine.
pub(crate) type CrtStart = Option<unsafe extern "C" fn(*mut c_void)>;

fn address<T: Copy>(f: Option<T>) -> usize {
    // SAFETY: `T` is a function pointer type, the size of a pointer.
    f.map_or(0, |f| unsafe { std::mem::transmute_copy::<T, usize>(&f) })
}

/// `CreateThread`, logged ([`note`]).
pub extern "system" fn CreateThread(
    security: *const c_void,
    stack: usize,
    start: Start,
    parameter: *const c_void,
    flags: u32,
    id: *mut u32,
) -> *mut c_void {
    note(stack, flags, address(start));
    host::create_thread(security, stack, start, parameter, flags, id)
}

/// `CreateRemoteThread`, logged ([`note`]).
pub extern "system" fn CreateRemoteThread(
    process: *mut c_void,
    security: *const c_void,
    stack: usize,
    start: Start,
    parameter: *const c_void,
    flags: u32,
    id: *mut u32,
) -> *mut c_void {
    note(stack, flags, address(start));
    host::create_remote_thread(process, security, stack, start, parameter, flags, id)
}

/// `CreateRemoteThreadEx`, logged ([`note`]).
#[allow(clippy::too_many_arguments)]
pub extern "system" fn CreateRemoteThreadEx(
    process: *mut c_void,
    security: *const c_void,
    stack: usize,
    start: Start,
    parameter: *const c_void,
    flags: u32,
    attributes: *mut c_void,
    id: *mut u32,
) -> *mut c_void {
    note(stack, flags, address(start));
    host::create_remote_thread_ex(
        process, security, stack, start, parameter, flags, attributes, id,
    )
}

/// `_beginthreadex`, logged ([`note`]).
pub extern "C" fn _beginthreadex(
    security: *mut c_void,
    stack: u32,
    start: CrtStartEx,
    parameter: *mut c_void,
    flags: u32,
    id: *mut u32,
) -> usize {
    note(stack as usize, flags, address(start));
    host::begin_thread_ex(security, stack, start, parameter, flags, id)
}

/// `_beginthread`, logged ([`note`]).
pub extern "C" fn _beginthread(start: CrtStart, stack: u32, parameter: *mut c_void) -> usize {
    note(stack as usize, 0, address(start));
    host::begin_thread(start, stack, parameter)
}

#[cfg(windows)]
mod host {
    use super::{CrtStart, CrtStartEx, Start};
    use std::ffi::c_void;
    use std::sync::OnceLock;
    use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::System::Threading::{
        CreateRemoteThread, CreateRemoteThreadEx, CreateThread,
    };

    unsafe extern "C" {
        #[link_name = "_beginthreadex"]
        fn crt_beginthreadex(
            security: *mut c_void,
            stack: u32,
            start: CrtStartEx,
            parameter: *mut c_void,
            flags: u32,
            id: *mut u32,
        ) -> usize;
        #[link_name = "_beginthread"]
        fn crt_beginthread(start: CrtStart, stack: u32, parameter: *mut c_void) -> usize;
    }

    // SAFETY (every forwarding function): the title's own arguments, passed
    // to the host function of the same name.

    pub(super) fn create_thread(
        security: *const c_void,
        stack: usize,
        start: Start,
        parameter: *const c_void,
        flags: u32,
        id: *mut u32,
    ) -> *mut c_void {
        // SAFETY: see above.
        unsafe { CreateThread(security.cast(), stack, start, parameter, flags, id) }
    }

    pub(super) fn create_remote_thread(
        process: *mut c_void,
        security: *const c_void,
        stack: usize,
        start: Start,
        parameter: *const c_void,
        flags: u32,
        id: *mut u32,
    ) -> *mut c_void {
        // SAFETY: see above.
        unsafe { CreateRemoteThread(process, security.cast(), stack, start, parameter, flags, id) }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_remote_thread_ex(
        process: *mut c_void,
        security: *const c_void,
        stack: usize,
        start: Start,
        parameter: *const c_void,
        flags: u32,
        attributes: *mut c_void,
        id: *mut u32,
    ) -> *mut c_void {
        // SAFETY: see above.
        unsafe {
            CreateRemoteThreadEx(
                process,
                security.cast(),
                stack,
                start,
                parameter,
                flags,
                attributes.cast(),
                id,
            )
        }
    }

    pub(super) fn begin_thread_ex(
        security: *mut c_void,
        stack: u32,
        start: CrtStartEx,
        parameter: *mut c_void,
        flags: u32,
        id: *mut u32,
    ) -> usize {
        // SAFETY: see above.
        unsafe { crt_beginthreadex(security, stack, start, parameter, flags, id) }
    }

    pub(super) fn begin_thread(start: CrtStart, stack: u32, parameter: *mut c_void) -> usize {
        // SAFETY: see above.
        unsafe { crt_beginthread(start, stack, parameter) }
    }

    /// Our executable's stack reserve (`SizeOfStackReserve` of its PE
    /// header), the reserve of a thread created without one.
    pub(super) fn exe_reserve() -> usize {
        static RESERVE: OnceLock<usize> = OnceLock::new();
        *RESERVE.get_or_init(|| {
            // SAFETY: our own executable's image, mapped for the process's
            // life; `e_lfanew` (+0x3c) leads to the NT headers, whose
            // optional header holds `SizeOfStackReserve` at +0x48 from it
            // (+0x18 + 0x48 from the signature) in a PE32+ image.
            unsafe {
                let base = GetModuleHandleW(std::ptr::null()).cast::<u8>();
                if base.is_null() {
                    return 1 << 20;
                }
                let nt = base.add(base.add(0x3c).cast::<u32>().read_unaligned() as usize);
                nt.add(0x18 + 0x48).cast::<u64>().read_unaligned() as usize
            }
        })
    }

    pub(super) fn locate(address: usize) -> String {
        victory_loader::locate(address)
    }

    /// Our callers, as `module+rva`, nearest first: the export's own
    /// frames, then the title's call site.
    pub(super) fn callers() -> String {
        let mut frames = [std::ptr::null_mut(); 8];
        // SAFETY: fills at most `frames.len()` return addresses.
        let n = unsafe {
            RtlCaptureStackBackTrace(
                1,
                frames.len() as u32,
                frames.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        let names: Vec<_> = frames[..usize::from(n)]
            .iter()
            .map(|&f| victory_loader::locate(f as usize))
            .collect();
        names.join(" < ")
    }
}

#[cfg(not(windows))]
mod host {
    use super::{CrtStart, CrtStartEx, Start};
    use std::ffi::c_void;

    pub(super) fn create_thread(
        _: *const c_void,
        _: usize,
        _: Start,
        _: *const c_void,
        _: u32,
        _: *mut u32,
    ) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn create_remote_thread(
        _: *mut c_void,
        _: *const c_void,
        _: usize,
        _: Start,
        _: *const c_void,
        _: u32,
        _: *mut u32,
    ) -> *mut c_void {
        std::ptr::null_mut()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_remote_thread_ex(
        _: *mut c_void,
        _: *const c_void,
        _: usize,
        _: Start,
        _: *const c_void,
        _: u32,
        _: *mut c_void,
        _: *mut u32,
    ) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn begin_thread_ex(
        _: *mut c_void,
        _: u32,
        _: CrtStartEx,
        _: *mut c_void,
        _: u32,
        _: *mut u32,
    ) -> usize {
        0
    }

    pub(super) fn begin_thread(_: CrtStart, _: u32, _: *mut c_void) -> usize {
        usize::MAX
    }

    pub(super) fn exe_reserve() -> usize {
        1 << 20
    }

    pub(super) fn locate(address: usize) -> String {
        format!("{address:#x}")
    }

    pub(super) fn callers() -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;

    /// A commit size keeps the executable's reserve unless it is larger;
    /// a reservation is the reserve itself, to 64 KiB.
    #[test]
    fn a_thread_reserves_what_windows_gives_its_size_and_flags() {
        let exe = 16 * MIB;
        assert_eq!(reserve_of(0, 0, exe), exe);
        assert_eq!(reserve_of(0, RESERVATION, exe), exe);
        assert_eq!(reserve_of(64 << 10, 0, exe), exe, "a commit size");
        assert_eq!(reserve_of(20 * MIB + 1, 0, exe), 21 * MIB);
        assert_eq!(reserve_of(256 << 10, RESERVATION, exe), 256 << 10);
        assert_eq!(reserve_of(100_000, RESERVATION, exe), 128 << 10);
    }

    /// Each size and flag logs once; each start routine with a small
    /// reserve warns once.
    #[test]
    fn sizes_log_once_and_small_reserves_warn_once_per_start() {
        let (mut seen, mut warned) = (HashSet::new(), HashSet::new());
        let mut sight = |size, flags, start| {
            let reserve = reserve_of(size, flags, 16 * MIB);
            first_sight(&mut seen, &mut warned, size, flags, reserve, start)
        };
        assert_eq!(sight(0, 0, 0x10), (true, false));
        assert_eq!(sight(0, 0, 0x20), (false, false));
        assert_eq!(
            sight(0, RESERVATION, 0x10),
            (true, false),
            "the flag counts"
        );
        assert_eq!(sight(256 << 10, RESERVATION, 0x30), (true, true));
        assert_eq!(sight(256 << 10, RESERVATION, 0x30), (false, false));
        assert_eq!(
            sight(256 << 10, RESERVATION, 0x40),
            (false, true),
            "another start"
        );
        assert_eq!(sight(256 << 10, 0, 0x50), (true, false), "a commit size");
    }
}
