//! The console's DirectStorage (`XCRDAPI.dll`): a factory that opens files
//! and creates request queues and status arrays the title polls. Its
//! interfaces have grown since 2020 (new interface ids, new slots); the
//! slots here are the ones the title calls, read from its call
//! sites. Slots nobody has needed yet log their offset.
//!
//! Submitted requests of every queue run on one pool of workers, as the
//! console's hardware runs them: in parallel, started in queue order,
//! queues served by weighted round robin over their priorities (about 10x
//! per level, counted in bytes), with at most the staging buffer's worth of
//! compressed source in flight. Each request reads its source (a file or
//! memory), inflates zlib data and/or unpacks BCPack data, and writes the
//! destination (an address, or physical pages). Completion is what stays
//! in order: a status entry is written once every request enqueued before
//! it on its queue has finished, with the first failure (in queue order)
//! since the previous status entry, or success.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::ffi::{CStr, c_char, c_void};
use std::os::windows::fs::FileExt;
use std::os::windows::io::FromRawHandle;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

use victory_formats::bcpack::{self, Tables};
use windows_sys::Win32::Foundation::HANDLE;

use super::release_ref;
use super::xasync::{E_INVALIDARG, E_NOTIMPL, E_PENDING};
use super::xgameruntime::{
    Guid, HRESULT, Methods, Object, S_OK, add_ref, missing, put, query_interface,
};
use crate::{HleModule, hle_export};

mod errors;
use errors::{ErrorRecord, QueueErrors, ReadFailure};

/// The factory interface the title asks for.
const FACTORY: Guid = Guid::new(
    0x6295_9d11,
    0xd45f,
    0x4884,
    [0xa0, 0xa3, 0xf7, 0x86, 0x2e, 0xa2, 0x43, 0xae],
);
const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_HANDLE_EOF)`: the source ends early.
const END_OF_FILE: HRESULT = 0x8007_0026_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_INVALID_DATA)`: a bad compressed stream.
const INVALID_DATA: HRESULT = 0x8007_000D_u32 as i32;

type Out = *mut *mut c_void;

static FACTORY_OBJECT: Object = Object::with_iids("dstorage factory", &FACTORY_VTABLE, &[FACTORY]);

pub extern "system" fn DStorageGetFactory(iid: *const Guid, out: Out) -> HRESULT {
    // SAFETY: the title passes a GUID (null tolerated).
    match unsafe { iid.as_ref() } {
        Some(&iid) if iid == FACTORY => {
            put(out, std::ptr::from_ref(&FACTORY_OBJECT).cast_mut().cast())
        }
        iid => {
            tracing::warn!(iid = ?iid.map(Guid::to_string), "DStorageGetFactory: unknown interface");
            E_NOINTERFACE
        }
    }
}

/// One of our objects with state: the runtime's object header (method
/// table first), a reference count, then `T`.
#[repr(C)]
struct Instance<T> {
    base: Object,
    refs: AtomicU32,
    state: T,
}

impl<T> Instance<T> {
    fn create(base: Object, state: T) -> *mut c_void {
        Box::into_raw(Box::new(Instance {
            base,
            refs: AtomicU32::new(1),
            state,
        }))
        .cast()
    }
}

extern "system" fn add_ref_instance<T>(this: *const Instance<T>) -> u32 {
    // SAFETY: an instance we created and the title still holds.
    unsafe { &*this }.refs.fetch_add(1, Ordering::Relaxed) + 1
}

extern "system" fn release<T>(this: *mut Instance<T>) -> u32 {
    // SAFETY: as above.
    let left = release_ref(&unsafe { &*this }.refs, "dstorage object");
    if left == Some(0) {
        // SAFETY: the last reference; created by `Instance::create`.
        drop(unsafe { Box::from_raw(this) });
    }
    left.unwrap_or(0)
}

fn iid_text(iid: *const Guid) -> String {
    // SAFETY: the title passes a GUID (null tolerated).
    unsafe { iid.as_ref() }.map_or_else(|| "null".into(), Guid::to_string)
}

/// `HRESULT_FROM_WIN32` of a host error.
fn from_win32(error: &std::io::Error) -> HRESULT {
    error
        .raw_os_error()
        .map_or(E_INVALIDARG, |e| (0x8007_0000 | (e as u32 & 0xffff)) as i32)
}

// The factory.

type Factory = *const Object;

/// Opens `path` for reading; the file object owns the handle.
extern "system" fn open_file(_: Factory, path: *const u16, iid: *const Guid, out: Out) -> HRESULT {
    const GENERIC_READ: u32 = 0x8000_0000;
    // Match typical archive readers: allow others to write/delete while we
    // read, and capture LastError before any allocation (VIC-BT-037 / #440).
    const FILE_SHARE_ALL: u32 = 0x7; // READ | WRITE | DELETE
    const OPEN_EXISTING: u32 = 3;
    if path.is_null() {
        return E_INVALIDARG;
    }
    // Goes through `files`, so paths on our virtual drives work too.
    let handle = super::files::CreateFileW(
        path,
        GENERIC_READ,
        FILE_SHARE_ALL,
        std::ptr::null(),
        OPEN_EXISTING,
        0,
        std::ptr::null_mut(),
    );
    if handle as isize == -1 {
        let error = std::io::Error::last_os_error();
        // SAFETY: the title passes a NUL-terminated path.
        let filename = unsafe {
            let len = (0..).take_while(|&i| *path.add(i) != 0).count();
            std::slice::from_raw_parts(path, len)
        };
        let name = String::from_utf16_lossy(filename);
        tracing::warn!(path = name, %error, "dstorage: opening a file failed");
        return from_win32(&error);
    }
    // SAFETY: the title passes a NUL-terminated path.
    let filename = unsafe {
        let len = (0..).take_while(|&i| *path.add(i) != 0).count();
        std::slice::from_raw_parts(path, len)
    };
    let name = String::from_utf16_lossy(filename);
    tracing::info!(path = name, iid = iid_text(iid), "dstorage: file opened");
    // SAFETY: a handle we just opened and nobody else owns.
    let file = unsafe { std::fs::File::from_raw_handle(handle) };
    let files = Files::new(file, Io::chosen());
    let file = Instance::create(
        Object::new("dstorage file", &FILE_VTABLE),
        File {
            files,
            filename: filename.into(),
        },
    );
    if stream_log::on() {
        stream_log::line(format_args!("F\t{:x}\t{name}", file as usize));
    }
    put(out, file)
}

/// `(capacity, name, iid, out)`.
extern "system" fn create_status_array(
    _: Factory,
    capacity: u32,
    name: *const c_char,
    iid: *const Guid,
    out: Out,
) -> HRESULT {
    tracing::info!(
        capacity,
        name = text(name),
        iid = iid_text(iid),
        "dstorage: status array"
    );
    let entries = (0..capacity).map(|_| AtomicI32::new(S_OK)).collect();
    let written = (0..capacity).map(|_| AtomicU64::new(0)).collect();
    let array = Instance::create(
        Object::new("dstorage status array", &STATUS_VTABLE),
        StatusArray { entries, written },
    );
    put(out, array)
}

/// The queue's description: as in 2020 (source type, capacity, priority,
/// name), plus a field the title sets to 1.
#[repr(C)]
struct QueueDesc {
    source_type: u64,
    capacity: u16,
    priority: i8,
    name: *const c_char,
    unknown: u64,
}

extern "system" fn create_queue(
    _: Factory,
    desc: *const QueueDesc,
    iid: *const Guid,
    out: Out,
) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes its description (null tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let name = text(desc.name);
    tracing::info!(
        source = desc.source_type,
        capacity = desc.capacity,
        priority = desc.priority,
        name,
        unknown = desc.unknown,
        iid = iid_text(iid),
        "dstorage: queue"
    );
    let Some(scheduler) = scheduler() else {
        return E_OUTOFMEMORY;
    };
    let lane = match Lane::new(desc.priority, desc.capacity) {
        Ok(lane) => lane,
        Err(hr) => return hr,
    };
    let mut lanes = scheduler.lock();
    let id = lanes.next_id;
    lanes.next_id += 1;
    lanes.lanes.insert(id, lane);
    drop(lanes);
    put(
        out,
        Instance::create(Object::new("dstorage queue", &QUEUE_VTABLE), Queue { id }),
    )
}

/// The title asks for 32 MiB and stops if this fails.
extern "system" fn set_staging_buffer_size(_: Factory, size: u32) -> HRESULT {
    tracing::info!(size, "dstorage: staging buffer size");
    if let Some(scheduler) = scheduler() {
        scheduler.lock().staging = u64::from(size);
    }
    S_OK
}

fn text(name: *const c_char) -> String {
    if name.is_null() {
        return String::new();
    }
    // SAFETY: a NUL-terminated name from the title.
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

static FACTORY_VTABLE: Methods<16> = Methods([
    query_interface as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),            // 0x18 (2020: queue creation)
    open_file as *const (),               // 0x20
    create_status_array as *const (),     // 0x28
    missing::<6> as *const (),            // 0x30 (2020: CPU affinity)
    missing::<7> as *const (),            // 0x38 (2020: debug flags)
    set_staging_buffer_size as *const (), // 0x40
    missing::<9> as *const (),
    create_queue as *const (), // 0x50
    missing::<11> as *const (),
    missing::<12> as *const (),
    missing::<13> as *const (),
    missing::<14> as *const (),
    missing::<15> as *const (),
]);

// Files.

struct File {
    files: Files,
    filename: Box<[u16]>,
}

/// How files are read (`VICTORY_DSTORAGE_IO=one|handles|overlapped|direct`;
/// defaults to `handles`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Io {
    /// `one`: all workers through the one handle, as before. The host does
    /// the reads of a handle one after the other.
    One,
    /// `handles`: a handle for each worker.
    Handles,
    /// `overlapped`: one handle, whose reads the host does side by side.
    Overlapped,
    /// `direct`: a handle for each worker, past the host's file cache.
    Direct,
}

impl Io {
    fn chosen() -> Io {
        static CHOSEN: OnceLock<Io> = OnceLock::new();
        *CHOSEN.get_or_init(|| {
            // `VICTORY_DSTORAGE_IO`: `one`, `handles` (default), `overlapped`, or `direct`.
            let io = match std::env::var("VICTORY_DSTORAGE_IO")
                .as_deref()
                .map(str::trim)
            {
                Ok("one") => Io::One,
                Ok("handles") => Io::Handles,
                Ok("overlapped") => Io::Overlapped,
                Ok("direct") => Io::Direct,
                _ => Io::DEFAULT,
            };
            tracing::info!(?io, "dstorage: how files are read (VICTORY_DSTORAGE_IO)");
            io
        })
    }

    const DEFAULT: Io = Io::Handles;
}

fn worker_count() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// A file as the workers read it: the handle the title opened, and those
/// made from it when a worker first reads.
struct Files {
    first: std::fs::File,
    others: Box<[OnceLock<Option<std::fs::File>>]>,
    io: Io,
}

impl Files {
    fn new(first: std::fs::File, io: Io) -> Files {
        let workers = worker_count();
        let others = match io {
            Io::One => 0,
            Io::Overlapped => 1,
            Io::Handles | Io::Direct => workers,
        };
        Files {
            first,
            others: (0..others).map(|_| OnceLock::new()).collect(),
            io,
        }
    }

    #[cfg(test)]
    fn open_for_test(path: &str, io: Io) -> Files {
        Files::new(std::fs::File::open(path).unwrap(), io)
    }

    /// Reads at `offset`; as [`FileExt::seek_read`].
    fn read_at(&self, out: &mut [u8], offset: u64) -> std::io::Result<usize> {
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED,
        };
        let flags = match self.io {
            Io::One | Io::Handles => 0,
            Io::Overlapped => FILE_FLAG_OVERLAPPED,
            Io::Direct => FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING,
        };
        let other = match self.others.len() {
            0 => None,
            n => self.others[this_thread() % n]
                .get_or_init(|| reopen(&self.first, flags))
                .as_ref(),
        };
        match (other, self.io) {
            (Some(file), Io::Overlapped) => read_overlapped(file, out, offset),
            (Some(file), Io::Direct) => read_direct(file, out, offset),
            (file, _) => file.unwrap_or(&self.first).seek_read(out, offset),
        }
    }
}

/// A number for this thread: 0 for the first that asks, and so on.
fn this_thread() -> usize {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    thread_local! {
        static NUMBER: usize = NEXT.fetch_add(1, Ordering::Relaxed) as usize;
    }
    NUMBER.with(|n| *n)
}

/// Another handle to `file`, for reading; `None` if the host gives none
/// (then the first one is used).
fn reopen(file: &std::fs::File, flags: u32) -> Option<std::fs::File> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{GENERIC_READ, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, ReOpenFile,
    };
    // SAFETY: a file we hold open.
    let handle = unsafe {
        ReOpenFile(
            file.as_raw_handle(),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            flags,
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        tracing::warn!(
            error = %std::io::Error::last_os_error(),
            flags,
            "dstorage: a file could not be opened again"
        );
        return None;
    }
    // SAFETY: a handle just opened, ours alone.
    Some(unsafe { std::fs::File::from_raw_handle(handle) })
}

/// A read through a handle opened for overlapped reads, waited for.
fn read_overlapped(file: &std::fs::File, out: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{CloseHandle, ERROR_HANDLE_EOF, ERROR_IO_PENDING, HANDLE};
    use windows_sys::Win32::Storage::FileSystem::ReadFile;
    use windows_sys::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};
    use windows_sys::Win32::System::Threading::CreateEventW;
    /// Set when this thread's read is done.
    struct Event(HANDLE);
    impl Drop for Event {
        fn drop(&mut self) {
            // SAFETY: our own event.
            unsafe { CloseHandle(self.0) };
        }
    }
    thread_local! {
        static DONE: Event =
            // SAFETY: a plain manual-reset event.
            Event(unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) });
    }
    DONE.with(|done| {
        if done.0.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let mut overlapped = OVERLAPPED {
            hEvent: done.0,
            ..Default::default()
        };
        overlapped.Anonymous.Anonymous.Offset = offset as u32;
        overlapped.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
        let len = out.len().min(1 << 30) as u32;
        let mut n = 0;
        // SAFETY: `out` and `overlapped` stay until the read is done, which
        // is waited for.
        let mut read = unsafe {
            ReadFile(
                file.as_raw_handle(),
                out.as_mut_ptr(),
                len,
                std::ptr::null_mut(),
                &mut overlapped,
            )
        } != 0;
        let mut error = std::io::Error::last_os_error();
        if read || error.raw_os_error() == Some(ERROR_IO_PENDING as i32) {
            // SAFETY: as above.
            read =
                unsafe { GetOverlappedResult(file.as_raw_handle(), &overlapped, &mut n, 1) } != 0;
            error = std::io::Error::last_os_error();
        }
        match error.raw_os_error() {
            _ if read => Ok(n as usize),
            Some(code) if code == ERROR_HANDLE_EOF as i32 => Ok(0),
            _ => Err(error),
        }
    })
}

/// A read past the file cache: of the whole sectors around the bytes, into
/// a buffer of this thread's, unless `out` takes them as they are.
fn read_direct(file: &std::fs::File, out: &mut [u8], offset: u64) -> std::io::Result<usize> {
    /// Every disk's sectors are at most this; offsets, sizes and addresses
    /// are multiples of it.
    const SECTOR: usize = 4096;
    /// Read at a time.
    const MOST: usize = 1 << 20;
    let sector = SECTOR as u64;
    if offset.is_multiple_of(sector)
        && out.len().is_multiple_of(SECTOR)
        && (out.as_ptr() as usize).is_multiple_of(SECTOR)
    {
        return read_overlapped(file, out, offset);
    }
    thread_local! {
        static BUFFER: std::cell::RefCell<Vec<u8>> =
            std::cell::RefCell::new(vec![0; MOST + SECTOR]);
    }
    BUFFER.with_borrow_mut(|buffer| {
        let skip = buffer.as_ptr().align_offset(SECTOR);
        let before = (offset % sector) as usize;
        let len = (before + out.len()).next_multiple_of(SECTOR).min(MOST);
        let aligned = &mut buffer[skip..skip + len];
        let read = read_overlapped(file, aligned, offset - before as u64)?;
        let n = read.saturating_sub(before).min(out.len());
        out[..n].copy_from_slice(&aligned[before..before + n]);
        Ok(n)
    })
}

static FILE_VTABLE: Methods<8> = Methods([
    query_interface as *const (),
    add_ref_instance::<File> as *const (),
    release::<File> as *const (),
    missing::<3> as *const (),
    missing::<4> as *const (),
    missing::<5> as *const (),
    missing::<6> as *const (),
    missing::<7> as *const (),
]);

// Status arrays.

struct StatusArray {
    entries: Box<[AtomicI32]>,
    /// For the log: when each entry was written ([`stream_log::now`]); 0
    /// once the title has seen it.
    written: Box<[AtomicU64]>,
}

type Array = *mut Instance<StatusArray>;

/// The result of the requests before entry `index`: pending, the first
/// failure, or success.
extern "system" fn status_result(this: Array, index: u32) -> HRESULT {
    // SAFETY: an array we created and the title still holds.
    let entries = &unsafe { &*this }.state.entries;
    entries
        .get(index as usize)
        .map_or(E_INVALIDARG, |e| e.load(Ordering::Acquire))
}

/// The title polls this (entry 0, a `Sleep(1)` loop) and never
/// reads the result itself.
extern "system" fn is_complete(this: Array, index: u32) -> bool {
    let done = status_result(this, index) != E_PENDING;
    if done && stream_log::on() {
        // SAFETY: as above.
        let written = &unsafe { &*this }.state.written;
        if let Some(at) = written
            .get(index as usize)
            .map(|w| w.swap(0, Ordering::Relaxed))
            .filter(|&at| at != 0)
        {
            stream_log::line(format_args!(
                "P\t{:x}\t{at}\t{}",
                this as usize,
                stream_log::now()
            ));
        }
    }
    done
}

static STATUS_VTABLE: Methods<5> = Methods([
    query_interface as *const (),
    add_ref_instance::<StatusArray> as *const (),
    release::<StatusArray> as *const (),
    is_complete as *const (),   // 0x18
    status_result as *const (), // 0x20
]);

// Queues.

/// A queue object: its lane in the scheduler.
struct Queue {
    id: u64,
}

impl Drop for Queue {
    fn drop(&mut self) {
        close_lane(self.id);
    }
}

type QueueRef = *const Instance<Queue>;

/// Every queue's requests, run by a pool of workers.
struct Scheduler {
    lanes: Mutex<Lanes>,
    /// Requests to start, or staging space freed.
    work: Condvar,
    /// A request finished (for full queues and `close`).
    finished: Condvar,
}

/// Priority weights per class (low, normal, high and real-time), as the
/// console's: about 10x per level, counted against request sizes.
const COST: [u64; 3] = [100, 10, 1];
/// The console's default staging buffer.
const STAGING: u64 = 32 << 20;

struct Lanes {
    lanes: HashMap<u64, Lane>,
    next_id: u64,
    /// Submission order over all queues.
    next_seq: u64,
    /// Bytes of staging held by running requests, and the limit.
    staged: u64,
    staging: u64,
    /// A start waited for staging space since the last finish.
    starved: bool,
    /// Weighted round robin: each class's virtual time (bytes × cost), and
    /// the time of the last start; within a class, the lane clock of the
    /// last start.
    clock: [u64; 3],
    now: u64,
    lane_now: [u64; 3],
    /// Since the last debug summary: requests, bytes written, time spent
    /// running them.
    done: (u64, u64, std::time::Duration),
    refused: u64,
    /// BCPack bytes written and time spent on them.
    bcpack: (u64, std::time::Duration),
    since: Option<std::time::Instant>,
}

/// A queue's requests in the scheduler.
struct Lane {
    class: usize,
    capacity: usize,
    /// Bytes started, for round robin among the lanes of a class.
    clock: u64,
    /// Enqueued, not yet submitted.
    unsubmitted: Vec<Command>,
    /// Submitted reads and status entries, by sequence number.
    reads: VecDeque<(u64, Request, ReadNote)>,
    statuses: VecDeque<(u64, Status)>,
    running: BTreeSet<u64>,
    /// First completed failure since the previous status entry.
    first_status_failure: Option<HRESULT>,
    errors: QueueErrors,
    closed: bool,
}

// SAFETY: requests point into title memory and our own objects, which stay
// valid until the title has seen them complete; status arrays count their
// references atomically.
unsafe impl Send for Lane {}

/// Timestamps kept per read while `VICTORY_DSTORAGE_LOG` is on.
#[derive(Clone, Copy, Default)]
struct ReadNote {
    enqueued: Option<std::time::Instant>,
    submitted: Option<std::time::Instant>,
}

/// A read a worker runs.
struct Job {
    lane: u64,
    seq: u64,
    request: Request,
    staging: u64,
    note: ReadNote,
}

// SAFETY: as for `Lane`.
unsafe impl Send for Job {}

/// The streaming log (`VICTORY_DSTORAGE_LOG=<file>`): a line for every
/// request (`R`: queue, number, file, offset, source bytes, destination
/// bytes, options, when enqueued, submitted, started, done, result), every
/// status entry written (`S`: array, number, when enqueued, when written),
/// the title's first look at it after (`P`: array, when written, when
/// seen), every file opened (`F`), and each second what the title's mapping
/// calls cost (`M`, `xmem::map_costs`). Times are milliseconds since the
/// Unix epoch.
pub(crate) mod stream_log {
    use super::*;
    use std::io::Write;

    type Log = Mutex<std::io::BufWriter<std::fs::File>>;

    fn log() -> Option<&'static Log> {
        static LOG: OnceLock<Option<Log>> = OnceLock::new();
        LOG.get_or_init(|| {
            // `VICTORY_DSTORAGE_LOG=<path>`: writes per-request/status/mapping telemetry TSV.
            let path = std::env::var_os("VICTORY_DSTORAGE_LOG")?;
            let file = std::fs::File::create(&path).ok()?;
            std::thread::spawn(|| {
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    if let Some(costs) = super::super::xmem::map_costs() {
                        line(format_args!("M\t{}\t{costs}", now()));
                    }
                    if let Some(log) = log() {
                        let _ = log.lock().unwrap_or_else(|e| e.into_inner()).flush();
                    }
                }
            });
            Some(Mutex::new(std::io::BufWriter::with_capacity(1 << 20, file)))
        })
        .as_ref()
    }

    pub(crate) fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_DSTORAGE_LOG").is_some())
    }

    /// Milliseconds since the Unix epoch.
    pub(super) fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |t| t.as_millis() as u64)
    }

    pub(super) fn line(text: std::fmt::Arguments) {
        if let Some(log) = log() {
            let _ = writeln!(log.lock().unwrap_or_else(|e| e.into_inner()), "{text}");
        }
    }

    pub(super) fn request(job: &Job, started: std::time::Instant, result: &Result<(), HRESULT>) {
        if !on() {
            return;
        }
        let now = now();
        let ago = |t: Option<std::time::Instant>| {
            now.saturating_sub(t.map_or(0, |t| t.elapsed().as_millis() as u64))
        };
        let r = &job.request;
        line(format_args!(
            "R\t{}\t{}\t{:x}\t{}\t{}\t{}\t{:x}\t{}\t{}\t{}\t{now}\t{:x}",
            job.lane,
            job.seq,
            r.source as usize,
            r.offset,
            r.source_size,
            r.destination_size,
            r.options,
            ago(job.note.enqueued),
            ago(job.note.submitted),
            ago(Some(started)),
            result.err().unwrap_or(0)
        ));
    }
}

/// The scheduler, with its workers started on first use; `None` if not
/// even one worker could start.
fn scheduler() -> Option<&'static Scheduler> {
    static SCHEDULER: OnceLock<Option<Scheduler>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let scheduler = SCHEDULER
        .get_or_init(|| {
            Some(Scheduler {
                lanes: Mutex::new(Lanes {
                    lanes: HashMap::new(),
                    next_id: 0,
                    next_seq: 0,
                    staged: 0,
                    staging: STAGING,
                    starved: false,
                    clock: [0; 3],
                    now: 0,
                    lane_now: [0; 3],
                    done: Default::default(),
                    refused: 0,
                    bcpack: Default::default(),
                    since: None,
                }),
                work: Condvar::new(),
                finished: Condvar::new(),
            })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let workers = worker_count();
        let started = (0..workers)
            .filter(|i| {
                std::thread::Builder::new()
                    .name(format!("dstorage {i}"))
                    .spawn(|| work(scheduler))
                    .is_ok()
            })
            .count();
        tracing::info!(workers = started, "dstorage: workers");
    });
    Some(scheduler)
}

impl Scheduler {
    fn lock(&self) -> MutexGuard<'_, Lanes> {
        self.lanes.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// A worker: runs the next read of the most deserving queue.
fn work(scheduler: &'static Scheduler) {
    let mut lanes = scheduler.lock();
    loop {
        let Some(job) = lanes.start() else {
            lanes = scheduler
                .work
                .wait(lanes)
                .unwrap_or_else(|e| e.into_inner());
            continue;
        };
        drop(lanes);
        let started = std::time::Instant::now();
        let result = execute(&job.request);
        stream_log::request(&job, started, &result);
        let failure = result
            .err()
            .map(|hr| Box::new(ReadFailure::new(&job.request, hr)));
        lanes = scheduler.lock();
        lanes.finish(&job, failure);
        lanes.count(&job, started);
        if std::mem::take(&mut lanes.starved) {
            scheduler.work.notify_all();
        }
        scheduler.finished.notify_all();
    }
}

impl Lanes {
    /// Counts a finished request; every 5 s a debug line sums them up.
    fn count(&mut self, job: &Job, started: std::time::Instant) {
        let (requests, bytes, busy) = &mut self.done;
        *requests += 1;
        *bytes += u64::from(job.request.destination_size);
        *busy += started.elapsed();
        if job.request.options & BCPACK != 0 {
            self.bcpack.0 += u64::from(job.request.destination_size);
            self.bcpack.1 += started.elapsed();
        }
        let since = *self.since.get_or_insert(started);
        let elapsed = since.elapsed();
        if elapsed.as_secs() >= 5 {
            let waiting: usize = self.lanes.values().map(|l| l.reads.len()).sum();
            let reading = READING.swap(0, Ordering::Relaxed) as f64 / 1e9;
            tracing::debug!(
                requests = *requests,
                mb_per_s = format_args!("{:.1}", *bytes as f64 / elapsed.as_secs_f64() / 1e6),
                busy_workers = format_args!("{:.2}", busy.as_secs_f64() / elapsed.as_secs_f64()),
                reading_workers = format_args!("{:.2}", reading / elapsed.as_secs_f64()),
                staging_waits = std::mem::take(&mut self.refused),
                bcpack_mb = self.bcpack.0 / 1_000_000,
                bcpack_workers =
                    format_args!("{:.2}", self.bcpack.1.as_secs_f64() / elapsed.as_secs_f64()),
                waiting,
                "dstorage: last {:.1} s",
                elapsed.as_secs_f64()
            );
            self.done = Default::default();
            self.bcpack = Default::default();
            self.since = None;
        }
    }

    /// Starts the next read of the class with the least virtual time, from
    /// its lane that has started the fewest bytes, if the staging buffer
    /// has room for it (or nothing else holds any).
    fn start(&mut self) -> Option<Job> {
        let busy = || self.lanes.iter().filter(|(_, l)| !l.reads.is_empty());
        let (_, class) = busy().map(|(_, l)| (self.clock[l.class], l.class)).min()?;
        let (_, _, id) = busy()
            .filter(|(_, l)| l.class == class)
            .filter_map(|(&id, l)| Some((l.clock, l.reads.front()?.0, id)))
            .min()?;
        let lane = self.lanes.get_mut(&id)?;
        let &(seq, request, note) = lane.reads.front()?;
        let staging = staging_bytes(&request);
        if self.staged > 0 && self.staged + staging > self.staging {
            self.starved = true;
            self.refused += 1;
            return None;
        }
        lane.reads.pop_front();
        lane.running.insert(seq);
        self.staged += staging;
        let size = u64::from(request.destination_size.max(request.source_size)).max(1);
        self.now = self.clock[class];
        self.clock[class] += size * COST[class];
        self.lane_now[class] = lane.clock;
        lane.clock += size;
        Some(Job {
            lane: id,
            seq,
            request,
            staging,
            note,
        })
    }

    fn finish(&mut self, job: &Job, failure: Option<Box<ReadFailure>>) {
        self.staged -= job.staging;
        if let Some(lane) = self.lanes.get_mut(&job.lane) {
            lane.running.remove(&job.seq);
            if let Some(failure) = failure {
                lane.errors.defer(job.seq, failure);
            }
            lane.flush();
        }
    }

    /// Moves a queue's enqueued commands to the workers.
    fn submit(&mut self, id: u64) {
        let Some(class) = self.lanes.get(&id).map(|l| l.class) else {
            return;
        };
        // A class that had nothing waiting starts from now, not from the
        // credit it left behind.
        if !self
            .lanes
            .values()
            .any(|l| l.class == class && !l.reads.is_empty())
        {
            self.clock[class] = self.clock[class].max(self.now);
        }
        let Some(lane) = self.lanes.get_mut(&id) else {
            return;
        };
        if lane.reads.is_empty() {
            lane.clock = lane.clock.max(self.lane_now[class]);
        }
        for command in std::mem::take(&mut lane.unsubmitted) {
            let seq = self.next_seq;
            self.next_seq += 1;
            match command {
                Command::Read(request, mut note) => {
                    if stream_log::on() {
                        note.submitted = Some(std::time::Instant::now());
                    }
                    lane.reads.push_back((seq, request, note));
                }
                Command::Status(status) => lane.statuses.push_back((seq, status)),
            }
        }
        lane.flush();
    }
}

impl Lane {
    fn new(priority: i8, capacity: u16) -> Result<Self, HRESULT> {
        Ok(Self {
            class: priority.clamp(-1, 1).wrapping_add(1) as usize,
            capacity: usize::from(capacity),
            clock: 0,
            unsubmitted: Vec::new(),
            reads: VecDeque::new(),
            statuses: VecDeque::new(),
            running: BTreeSet::new(),
            first_status_failure: None,
            errors: QueueErrors::new()?,
            closed: false,
        })
    }

    /// Requests and status entries held, submitted or not.
    fn entries(&self) -> usize {
        self.unsubmitted.len() + self.reads.len() + self.statuses.len() + self.running.len()
    }

    /// Publishes completed errors and writes ready status entries in queue order.
    fn flush(&mut self) {
        let next_read = self
            .reads
            .front()
            .map(|(seq, _, _)| *seq)
            .into_iter()
            .chain(self.running.first().copied())
            .min()
            .unwrap_or(u64::MAX);
        while let Some(&(seq, _)) = self.statuses.front().filter(|(seq, _)| *seq < next_read) {
            let failure = self.errors.publish_before(seq);
            let result = self.first_status_failure.take().or(failure).unwrap_or(S_OK);
            let (_, status) = self.statuses.pop_front().unwrap();
            if stream_log::on() {
                let at = stream_log::now();
                stream_log::line(format_args!(
                    "S\t{:x}\t{seq}\t{}\t{at}",
                    status.array as usize, status.enqueued
                ));
                status.written().store(at, Ordering::Relaxed);
            }
            status.entry().store(result, Ordering::Release);
        }
        let failure = self.errors.publish_before(next_read);
        self.first_status_failure = self.first_status_failure.or(failure);
    }
}

/// Staging bytes a request holds while it runs: what it buffers between
/// the file and the destination. Raw reads go straight to their
/// destination, memory sources need none (as on the console).
fn staging_bytes(request: &Request) -> u64 {
    let buffered = request.options & (ZLIB | BCPACK | PAGE_DESTINATION) != 0;
    if buffered && request.options & MEMORY_SOURCE == 0 {
        u64::from(request.source_size)
    } else {
        0
    }
}

/// Ends a queue: what was enqueued but not submitted is dropped, and this
/// waits until what was submitted has completed or failed, its status
/// entries written, as the console's `Close` does.
fn close_lane(id: u64) {
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let Some(lane) = lanes.lanes.get_mut(&id) else {
        return;
    };
    lane.closed = true;
    lane.unsubmitted.clear();
    let pending = |l: &Lane| !(l.reads.is_empty() && l.running.is_empty() && l.statuses.is_empty());
    while lanes.lanes.get(&id).is_some_and(pending) {
        lanes = scheduler
            .finished
            .wait(lanes)
            .unwrap_or_else(|e| e.into_inner());
    }
    lanes.lanes.remove(&id);
    scheduler.finished.notify_all();
}

/// A read request as the title builds it, 0x48 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct Request {
    options: u64,
    /// An address, or with `PAGE_DESTINATION` an array of 64 KiB page
    /// numbers.
    destination: *mut c_void,
    /// With `PAGE_DESTINATION`, the low 16 bits are where in the first page
    /// the destination starts.
    page_offset: u64,
    destination_size: u32,
    /// Our file object, or with `MEMORY_SOURCE` an address.
    source: *const c_void,
    offset: u64,
    source_size: u32,
    /// Compressed size between two decompression stages; unused here.
    intermediate_size: u32,
    /// Matched by cancellation; the title leaves it 0.
    tag: u64,
    name: *const c_char,
}

// Request options.
/// Inflate zlib (RFC 1950) data.
const ZLIB: u64 = 1;
/// Bits 1-4: decode BCPack data of this BC format number (0: none).
const BCPACK: u64 = 0xf << 1;
const PAGE_DESTINATION: u64 = 1 << 21;
const MEMORY_SOURCE: u64 = 1 << 22;
// Bits 5-20 select a further transform, bit 23 a source in physical pages;
// The title's packages need neither, so requests with them fail.

const PAGE: usize = 64 * 1024;

enum Command {
    Read(Request, ReadNote),
    Status(Status),
}

/// A status entry to write, holding a reference on its array.
struct Status {
    array: Array,
    index: u32,
    /// For the log.
    enqueued: u64,
}

impl Drop for Status {
    fn drop(&mut self) {
        release(self.array);
    }
}

impl Status {
    fn entry(&self) -> &AtomicI32 {
        // SAFETY: we hold a reference; the index was checked when enqueued.
        &unsafe { &*self.array }.state.entries[self.index as usize]
    }

    fn written(&self) -> &AtomicU64 {
        // SAFETY: as above.
        &unsafe { &*self.array }.state.written[self.index as usize]
    }
}

extern "system" fn enqueue_request(this: QueueRef, request: *const Request) {
    // SAFETY: the title passes its request (null tolerated).
    let Some(&request) = (unsafe { request.as_ref() }) else {
        return;
    };
    let note = if stream_log::on() {
        ReadNote {
            enqueued: Some(std::time::Instant::now()),
            submitted: None,
        }
    } else {
        ReadNote::default()
    };
    queue(this, Command::Read(request, note));
}

extern "system" fn enqueue_status(this: QueueRef, array: Array, index: u32) {
    // SAFETY: the title passes an array we created (null tolerated).
    let Some(entries) = (unsafe { array.as_ref() }).map(|a| &a.state.entries) else {
        return;
    };
    let Some(entry) = entries.get(index as usize) else {
        tracing::warn!(index, "dstorage: status entry out of range");
        return;
    };
    entry.store(E_PENDING, Ordering::Release);
    add_ref_instance(array);
    let enqueued = if stream_log::on() {
        stream_log::now()
    } else {
        0
    };
    queue(
        this,
        Command::Status(Status {
            array,
            index,
            enqueued,
        }),
    );
}

/// Adds a command to the queue; blocks while the queue is full, and
/// submits on its own once half of it is enqueued, as the console does.
fn queue(this: QueueRef, command: Command) {
    // SAFETY: a queue we created and the title still holds.
    let id = unsafe { &*this }.state.id;
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    loop {
        let Some(lane) = lanes.lanes.get(&id).filter(|l| !l.closed) else {
            return; // closed: dropped (a status releases its array)
        };
        // One slot stays free, as the console's count of empty slots says.
        if lane.entries() + 1 < lane.capacity.max(2) {
            break;
        }
        lanes = scheduler
            .finished
            .wait(lanes)
            .unwrap_or_else(|e| e.into_inner());
    }
    let Some(lane) = lanes.lanes.get_mut(&id) else {
        return;
    };
    lane.unsubmitted.push(command);
    if lane.unsubmitted.len() >= (lane.capacity / 2).max(1) {
        lanes.submit(id);
        scheduler.work.notify_all();
    }
}

extern "system" fn submit(this: QueueRef) {
    // SAFETY: as above.
    let id = unsafe { &*this }.state.id;
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let pending = lanes.lanes.get(&id).map_or(0, |l| l.unsubmitted.len());
    tracing::debug!(commands = pending, "dstorage: submit");
    lanes.submit(id);
    scheduler.work.notify_all();
}

/// Ends the queue whatever its reference count: what was not submitted
/// never completes, and this returns once everything submitted has
/// completed or failed (a title may still be polling its status entries).
/// The title calls this, not `Release`.
extern "system" fn close(this: QueueRef) {
    // SAFETY: as above.
    close_lane(unsafe { &*this }.state.id);
}

extern "system" fn get_error_event(this: QueueRef) -> HANDLE {
    // SAFETY: a queue we created and the title still holds.
    let id = unsafe { &*this }.state.id;
    scheduler()
        .and_then(|scheduler| {
            let lanes = scheduler.lock();
            lanes
                .lanes
                .get(&id)
                .map(|lane| lane.errors.duplicate_event())
        })
        .unwrap_or(std::ptr::null_mut())
}

extern "system" fn retrieve_error_record(this: QueueRef, out: *mut ErrorRecord) {
    if out.is_null() {
        return;
    }
    // SAFETY: a queue we created and the title still holds.
    let id = unsafe { &*this }.state.id;
    let record = scheduler()
        .and_then(|scheduler| {
            let mut lanes = scheduler.lock();
            lanes.lanes.get_mut(&id).map(|lane| lane.errors.take())
        })
        .unwrap_or_default();
    // SAFETY: the title supplies room for its error record.
    unsafe { out.write(record) };
}

/// Reads, inflates and/or unpacks, and writes one request.
fn execute(request: &Request) -> Result<(), HRESULT> {
    let options = request.options;
    let fail = |hr: HRESULT, why: &str| {
        tracing::warn!(
            options = format_args!("{options:#x}"),
            offset = request.offset,
            source_size = request.source_size,
            destination_size = request.destination_size,
            intermediate_size = request.intermediate_size,
            tag = request.tag,
            name = text(request.name),
            hr = format_args!("{:#010x}", hr as u32),
            "dstorage: request failed: {why}"
        );
        hr
    };
    if options & !(ZLIB | BCPACK | PAGE_DESTINATION | MEMORY_SOURCE) != 0 {
        return Err(fail(E_NOTIMPL, "unsupported options"));
    }
    if request.source.is_null() || request.destination.is_null() {
        return Err(fail(E_INVALIDARG, "no source or destination"));
    }
    let bc = ((options & BCPACK) >> 1) as u8;
    let size = request.destination_size as usize;
    if options & (ZLIB | BCPACK) == 0 {
        if request.source_size > request.destination_size {
            return Err(fail(E_INVALIDARG, "source larger than destination"));
        }
        let size = request.source_size as usize;
        return into_destination(request, size, |out| read_into(request, out).map(|()| size))
            .map_err(|hr| fail(hr, "copying"));
    }
    let tables = if bc != 0 {
        Some(bcpack_tables()?)
    } else {
        None
    };
    let source = read_source(request).map_err(|hr| fail(hr, "reading the source"))?;
    // The console's order: zlib first, then BCPack.
    let source = if bc != 0 && options & ZLIB != 0 {
        let mut packed = vec![0; request.intermediate_size as usize];
        let n = inflate(&source, &mut packed).map_err(|hr| fail(hr, "zlib"))?;
        packed.truncate(n);
        Cow::Owned(packed)
    } else {
        source
    };
    into_destination(request, size, |out| match tables {
        Some(tables) => bcpack::decode(bc, &source, out, tables)
            .map(|()| out.len())
            .map_err(|e| fail(INVALID_DATA, &e.to_string())),
        None => inflate(&source, out).map_err(|hr| fail(hr, "zlib")),
    })?;
    tracing::trace!(
        options = format_args!("{options:#x}"),
        offset = request.offset,
        size,
        "dstorage: request done"
    );
    Ok(())
}

/// Inflates a zlib stream into `out`, returning the bytes written. A stream
/// that ends short of `out` passes: the title's chunks fill it exactly, and
/// nothing says that is required.
fn inflate(source: &[u8], out: &mut [u8]) -> Result<usize, HRESULT> {
    miniz_oxide::inflate::decompress_slice_iter_to_slice(out, std::iter::once(source), true, false)
        .map_err(|status| {
            tracing::debug!(?status, "dstorage: zlib stream");
            INVALID_DATA
        })
}

/// The BCPack format's constant tables, parsed once for the process. If
/// they do not parse every BCPack request fails: one warning, then debug.
fn bcpack_tables() -> Result<&'static Tables, HRESULT> {
    static TABLES: OnceLock<Result<Tables, String>> = OnceLock::new();
    static WARNED: AtomicBool = AtomicBool::new(false);
    match TABLES.get_or_init(load_bcpack_tables) {
        Ok(tables) => Ok(tables),
        Err(why) if !WARNED.swap(true, Ordering::Relaxed) => {
            tracing::warn!("dstorage: BCPack requests fail: {why} (more at debug)");
            Err(E_NOTIMPL)
        }
        Err(_) => {
            tracing::debug!("dstorage: BCPack request failed: no tables");
            Err(E_NOTIMPL)
        }
    }
}

/// The tables VIctory ships with.
#[cfg(not(test))]
fn load_bcpack_tables() -> Result<Tables, String> {
    Tables::builtin().map_err(|e| format!("the built-in BCPack tables: {e}"))
}

/// Tests decode streams made against tables of their own, from this folder.
#[cfg(test)]
static BCPACK_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

#[cfg(test)]
fn use_bcpack_tables(dir: std::path::PathBuf) {
    let _ = BCPACK_DIR.set(dir);
}

#[cfg(test)]
fn load_bcpack_tables() -> Result<Tables, String> {
    let dir = BCPACK_DIR
        .get()
        .ok_or("no folder set for the test tables")?;
    let bytes = std::fs::read(dir.join("tables.bin")).map_err(|e| e.to_string())?;
    Tables::parse(&bytes).map_err(|e| e.to_string())
}

/// A compressed request's source: borrowed from memory, or read from the
/// file.
fn read_source(request: &Request) -> Result<Cow<'static, [u8]>, HRESULT> {
    let size = request.source_size as usize;
    if request.options & MEMORY_SOURCE != 0 {
        // SAFETY: the title's source holds `size` bytes until completion.
        let bytes = unsafe { std::slice::from_raw_parts(request.source.cast(), size) };
        return Ok(Cow::Borrowed(bytes));
    }
    let mut bytes = vec![0; size];
    read_into(request, &mut bytes)?;
    Ok(Cow::Owned(bytes))
}

/// Nanoseconds workers spent reading files, for the debug summary.
static READING: AtomicU64 = AtomicU64::new(0);

/// Fills `out` (the source size) from the request's file or memory.
fn read_into(request: &Request, out: &mut [u8]) -> Result<(), HRESULT> {
    if request.options & MEMORY_SOURCE != 0 {
        // SAFETY: the title's source holds `out.len()` bytes until completion.
        out.copy_from_slice(unsafe {
            std::slice::from_raw_parts(request.source.cast(), out.len())
        });
        return Ok(());
    }
    // SAFETY: a file object we created; the title keeps it open.
    let files = &unsafe { &*request.source.cast::<Instance<File>>() }
        .state
        .files;
    let started = std::time::Instant::now();
    let mut done = 0;
    let result = loop {
        if done == out.len() {
            break Ok(());
        }
        match files.read_at(&mut out[done..], request.offset + done as u64) {
            Ok(0) => break Err(END_OF_FILE),
            Ok(n) => done += n,
            Err(error) => break Err(from_win32(&error)),
        }
    };
    READING.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    result
}

/// Runs `fill` on the destination's first `len` bytes; `fill` returns how
/// many it produced. GPU-visible destinations are host memory like the
/// rest (GPU address = CPU address), so an address is written in place;
/// physical pages are too when they are consecutive pages of one
/// allocation, and otherwise go through a buffer.
fn into_destination(
    request: &Request,
    len: usize,
    fill: impl FnOnce(&mut [u8]) -> Result<usize, HRESULT>,
) -> Result<(), HRESULT> {
    if request.options & PAGE_DESTINATION == 0 {
        if request.options & (ZLIB | BCPACK) == 0 {
            // Raw reads are bounded by the OS file read and do not need a
            // second potentially large buffer.
            victory_gpu::kernel_will_write(request.destination as usize, len);
            // SAFETY: the title's destination holds `destination_size` bytes,
            // at least `len`, and the worker owns this request until completion.
            let out =
                unsafe { std::slice::from_raw_parts_mut(request.destination.cast::<u8>(), len) };
            let filled = fill(out);
            victory_gpu::kernel_wrote(request.destination as usize, len);
            return filled.map(drop);
        }
        // Decode into host-owned memory first. A title mapping can disappear
        // while a worker is running; copying through the kernel reports that
        // instead of faulting this worker while it writes the destination.
        let mut data = vec![0; len];
        let written = fill(&mut data)?;
        if written > len {
            return Err(E_INVALIDARG);
        }
        if !victory_gpu::write_title_memory(request.destination as usize, &data[..written]) {
            tracing::warn!(
                bytes = written,
                "dstorage: decompressed destination write failed"
            );
            return Err(E_INVALIDARG);
        }
        return Ok(());
    }
    let offset = (request.page_offset & 0xffff) as usize;
    // SAFETY: the title passes a page number per 64 KiB of destination,
    // which holds at least `len` bytes from `offset`.
    let pages = unsafe {
        std::slice::from_raw_parts(
            request.destination.cast::<usize>(),
            (offset + len).div_ceil(PAGE),
        )
    };
    // Consecutive pages of one allocation (the usual case): written in
    // place through one view.
    if let Some(window) = super::xmem::page_window(pages, offset, len) {
        return window.fill(fill);
    }
    let mut data = vec![0; len];
    let n = fill(&mut data)?;
    if super::xmem::write_physical(pages, offset, &data[..n]) {
        return Ok(());
    }
    // Page number 0 is no page (numbers start at 0x10_0000): a list naming
    // it is one the title has not filled yet. The title writes a request's
    // page list after enqueueing it, and a queue half full is submitted on
    // its own, so a read can start first. Read the list again for a few
    // milliseconds before failing: the title never looks at a request's
    // result, so a failed one is a model or texture left unwritten.
    if pages.contains(&0) {
        for _ in 0..LATE_LIST_TRIES {
            std::thread::sleep(std::time::Duration::from_millis(1));
            // SAFETY: as above; the title's list, read again.
            let again = unsafe {
                std::slice::from_raw_parts(request.destination.cast::<usize>(), pages.len())
            };
            if !again.contains(&0) && super::xmem::write_physical(again, offset, &data[..n]) {
                let late = LATE_LISTS.fetch_add(1, Ordering::Relaxed) + 1;
                if late.is_power_of_two() {
                    tracing::info!(
                        late,
                        "dstorage: a page list filled after its request started: written late"
                    );
                }
                return Ok(());
            }
        }
    }
    tracing::warn!(
        offset,
        len,
        pages = pages.len(),
        "dstorage: a destination's physical pages could not be written; its request fails"
    );
    Err(E_INVALIDARG)
}

/// Milliseconds a page list naming page 0 is read again for.
const LATE_LIST_TRIES: u32 = 8;
/// Page lists filled after their request started (written late).
static LATE_LISTS: AtomicU32 = AtomicU32::new(0);

/// Slots by the 2020 order, which the argument shapes at the
/// title's call sites match (0x18, 0x20, 0x30 and 0x40 are called).
static QUEUE_VTABLE: Methods<12> = Methods([
    query_interface as *const (),
    add_ref_instance::<Queue> as *const (),
    release::<Queue> as *const (),
    enqueue_request as *const (),       // 0x18
    enqueue_status as *const (),        // 0x20
    missing::<5> as *const (),          // 0x28 (2020: fence signal)
    submit as *const (),                // 0x30
    missing::<7> as *const (),          // 0x38 (2020: cancel by tag)
    close as *const (),                 // 0x40
    get_error_event as *const (),       // 0x48
    retrieve_error_record as *const (), // 0x50
    missing::<11> as *const (),         // 0x58 (2020: queue info)
]);

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "dstorage",
        dlls: &["xcrdapi.dll"],
        exports: vec![hle_export!(DStorageGetFactory)],
    }
}

#[cfg(test)]
mod tests {
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
    use std::time::{Duration, Instant};

    use windows_sys::Win32::Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT};
    use windows_sys::Win32::System::Threading::WaitForSingleObject;

    use super::*;
    use crate::modules::xgameruntime::{E_POINTER, IID_IUNKNOWN, method};

    fn call<F: Copy>(object: *mut c_void, slot: usize) -> F {
        // SAFETY: every object here starts with the runtime header.
        method(unsafe { &*object.cast::<Object>() }, slot)
    }

    fn factory() -> *mut c_void {
        let mut factory = std::ptr::null_mut();
        assert_eq!(DStorageGetFactory(&FACTORY, &mut factory), S_OK);
        factory
    }

    fn open(path: &std::path::Path) -> *mut c_void {
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        let open: extern "system" fn(*mut c_void, *const u16, *const Guid, Out) -> HRESULT =
            call(factory(), 0x20);
        let mut file = std::ptr::null_mut();
        assert_eq!(
            open(factory(), wide.as_ptr(), std::ptr::null(), &mut file),
            S_OK
        );
        file
    }

    fn status_array(capacity: u32) -> *mut c_void {
        let create: extern "system" fn(
            *mut c_void,
            u32,
            *const c_char,
            *const Guid,
            Out,
        ) -> HRESULT = call(factory(), 0x28);
        let mut array = std::ptr::null_mut();
        let name = c"test".as_ptr();
        assert_eq!(
            create(factory(), capacity, name, std::ptr::null(), &mut array),
            S_OK
        );
        array
    }

    fn new_queue(source_type: u64) -> *mut c_void {
        let create: extern "system" fn(*mut c_void, *const QueueDesc, *const Guid, Out) -> HRESULT =
            call(factory(), 0x50);
        let desc = QueueDesc {
            source_type,
            capacity: 0x80,
            priority: 0,
            name: c"test".as_ptr(),
            unknown: 1,
        };
        let mut queue = std::ptr::null_mut();
        assert_eq!(create(factory(), &desc, std::ptr::null(), &mut queue), S_OK);
        queue
    }

    fn request(
        options: u64,
        destination: *mut c_void,
        destination_size: usize,
        source: *const c_void,
        offset: u64,
        source_size: usize,
    ) -> Request {
        Request {
            options,
            destination,
            page_offset: 0,
            destination_size: destination_size as u32,
            source,
            offset,
            source_size: source_size as u32,
            intermediate_size: 0,
            tag: 0,
            name: std::ptr::null(),
        }
    }

    fn enqueue(queue: *mut c_void, request: &Request) {
        let f: extern "system" fn(*mut c_void, *const Request) = call(queue, 0x18);
        f(queue, request);
    }

    fn enqueue_status(queue: *mut c_void, array: *mut c_void, index: u32) {
        let f: extern "system" fn(*mut c_void, *mut c_void, u32) = call(queue, 0x20);
        f(queue, array, index);
    }

    fn void(object: *mut c_void, slot: usize) {
        let f: extern "system" fn(*mut c_void) = call(object, slot);
        f(object);
    }

    fn error_event(queue: *mut c_void) -> OwnedHandle {
        let get: extern "system" fn(*mut c_void) -> HANDLE = call(queue, 0x48);
        let handle = get(queue);
        assert!(!handle.is_null());
        // SAFETY: the queue returned a caller-owned event handle.
        unsafe { OwnedHandle::from_raw_handle(handle) }
    }

    fn poll_event(event: &OwnedHandle) -> u32 {
        // SAFETY: the caller holds the event.
        unsafe { WaitForSingleObject(event.as_raw_handle(), 0) }
    }

    fn result(array: *mut c_void, index: u32) -> HRESULT {
        let f: extern "system" fn(*mut c_void, u32) -> HRESULT = call(array, 0x20);
        f(array, index)
    }

    /// Polls as the title does, then returns the entry's result.
    fn wait(array: *mut c_void, index: u32) -> HRESULT {
        let done: extern "system" fn(*mut c_void, u32) -> bool = call(array, 0x18);
        let start = Instant::now();
        while !done(array, index) {
            assert!(start.elapsed() < Duration::from_secs(10), "never completed");
            std::thread::sleep(Duration::from_millis(1));
        }
        result(array, index)
    }

    fn file_with(bytes: &[u8]) -> (tempfile::TempDir, *mut c_void) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, bytes).unwrap();
        let file = open(&path);
        (dir, file)
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 7 + i / 251) as u8).collect()
    }

    #[test]
    fn request_layout_is_the_titles() {
        assert_eq!(size_of::<Request>(), 0x48);
        assert_eq!(std::mem::offset_of!(Request, destination_size), 0x18);
        assert_eq!(std::mem::offset_of!(Request, source), 0x20);
        assert_eq!(std::mem::offset_of!(Request, source_size), 0x30);
        assert_eq!(std::mem::offset_of!(Request, name), 0x40);
        assert_eq!(std::mem::offset_of!(QueueDesc, unknown), 0x18);
    }

    #[test]
    fn factory_query_interface_accepts_iunknown_and_factory_iid() {
        type Qi = extern "system" fn(*mut c_void, *const Guid, Out) -> HRESULT;
        let factory = factory();
        let qi: Qi = call(factory, 0x00);
        let mut out = std::ptr::null_mut();
        assert_eq!(qi(factory, &IID_IUNKNOWN, &mut out), S_OK);
        assert_eq!(out, factory);
        out = std::ptr::null_mut();
        assert_eq!(qi(factory, &FACTORY, &mut out), S_OK);
        assert_eq!(out, factory);

        let unknown = Guid::new(0xdead_beef, 0, 0, [0; 8]);
        out = 0x1234 as *mut c_void;
        assert_eq!(qi(factory, &unknown, &mut out), E_NOINTERFACE);
        assert!(out.is_null());
        out = 0x1234 as *mut c_void;
        assert_eq!(qi(factory, std::ptr::null(), &mut out), E_POINTER);
        assert!(out.is_null());
        assert_eq!(qi(factory, &IID_IUNKNOWN, std::ptr::null_mut()), E_POINTER);
    }

    #[test]
    fn queue_query_interface_accepts_iunknown_and_addrefs() {
        type Qi = extern "system" fn(*mut c_void, *const Guid, Out) -> HRESULT;
        let queue = new_queue(0);
        // SAFETY: `new_queue` returns an `Instance<Queue>` we still own.
        let refs_before = unsafe { &*queue.cast::<Instance<Queue>>() }
            .refs
            .load(Ordering::Relaxed);
        let qi: Qi = call(queue, 0x00);
        let mut out = std::ptr::null_mut();
        assert_eq!(qi(queue, &IID_IUNKNOWN, &mut out), S_OK);
        assert_eq!(out, queue);
        // SAFETY: as above; QI AddRef'd through the instance's slot.
        let refs_after = unsafe { &*queue.cast::<Instance<Queue>>() }
            .refs
            .load(Ordering::Relaxed);
        assert_eq!(refs_after, refs_before + 1);
        void(queue, 0x10); // Release the QI retain
        void(queue, 0x10); // Release the create retain
    }

    #[test]
    fn raw_and_zlib_reads_from_a_file() {
        let plain = pattern(300_000);
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
        let mut contents = plain.clone();
        contents.extend_from_slice(&packed);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(1));

        let mut raw = vec![0u8; 1000];
        let mut inflated = vec![0u8; plain.len()];
        let at = |b: &mut Vec<u8>| b.as_mut_ptr().cast();
        enqueue(queue, &request(0, at(&mut raw), 1000, file, 5000, 1000));
        let offset = plain.len() as u64;
        let zlib = request(
            ZLIB,
            at(&mut inflated),
            plain.len(),
            file,
            offset,
            packed.len(),
        );
        enqueue(queue, &zlib);
        enqueue_status(queue, array, 0);
        assert_eq!(result(array, 0), E_PENDING, "pending until done");
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(raw, plain[5000..6000]);
        assert_eq!(inflated, plain);
        void(queue, 0x40);
    }

    #[test]
    fn memory_sources_inflate_too() {
        let plain = pattern(70_000);
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
        let (queue, array) = (new_queue(1), status_array(1));
        let mut out = vec![0u8; plain.len()];
        let options = ZLIB | MEMORY_SOURCE;
        let r = request(
            options,
            out.as_mut_ptr().cast(),
            out.len(),
            packed.as_ptr().cast(),
            0,
            packed.len(),
        );
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, plain);
    }

    #[test]
    fn statuses_complete_in_order_with_the_first_failure() {
        let contents = pattern(4096);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(3));
        // Requests in flight together run in any order, so none of them
        // shares a destination (failed ones leave theirs undefined).
        let mut buffers = [[0u8; 16]; 5];
        let [failed, failed_too, out, other, bcpack] =
            buffers.each_mut().map(|b| b.as_mut_ptr().cast());

        // Segment 0: a read past the end, then a bad stream; the first wins.
        enqueue(queue, &request(0, failed, 16, file, 4090, 16));
        enqueue(queue, &request(ZLIB, failed_too, 16, file, 0, 16));
        enqueue_status(queue, array, 0);
        // Segment 1: two reads.
        enqueue(queue, &request(0, other, 16, file, 0, 16));
        enqueue(queue, &request(0, out, 16, file, 100, 16));
        enqueue_status(queue, array, 1);
        void(queue, 0x30);
        // Segment 2: a BCPack request (BC7) whose stream is empty fails.
        with_test_tables();
        enqueue(queue, &request(7 << 1, bcpack, 16, file, 0, 0));
        enqueue_status(queue, array, 2);
        void(queue, 0x30);

        assert_eq!(wait(array, 2), INVALID_DATA);
        assert_eq!(result(array, 0), END_OF_FILE);
        assert_eq!(result(array, 1), S_OK);
        assert_eq!(buffers[2], contents[100..116]);
        assert_eq!(buffers[3], contents[..16]);
    }

    #[test]
    fn queue_reports_and_retrieves_read_errors() {
        let contents = pattern(64);
        let (dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(2));
        let idle_queue = new_queue(0);
        let event = error_event(queue);
        let idle_event = error_event(idle_queue);
        drop(error_event(queue)); // Each caller can close its own handle.
        assert_eq!(poll_event(&event), WAIT_TIMEOUT);

        let mut buffers = [[0u8; 16]; 3];
        let [short, bad_zlib, out] = buffers.each_mut().map(|b| b.as_mut_ptr().cast());
        let mut first = request(0, short, 16, file, 60, 16);
        first.name = c"short read".as_ptr();
        first.tag = 23;
        enqueue(queue, &first);
        enqueue(queue, &request(ZLIB, bad_zlib, 16, file, 0, 16));
        enqueue_status(queue, array, 0);
        enqueue(queue, &request(0, out, 16, file, 8, 16));
        enqueue_status(queue, array, 1);
        void(queue, 0x30);
        assert_eq!(wait(array, 1), S_OK);
        assert_eq!(result(array, 0), END_OF_FILE);
        assert_eq!(buffers[2], contents[8..24]);
        assert_eq!(poll_event(&event), WAIT_OBJECT_0);
        assert_eq!(poll_event(&idle_event), WAIT_TIMEOUT);

        #[repr(C)]
        struct Guarded {
            before: u64,
            record: ErrorRecord,
            after: u64,
        }
        let mut out = Guarded {
            before: 17,
            record: ErrorRecord::default(),
            after: 29,
        };
        let retrieve: extern "system" fn(*mut c_void, *mut ErrorRecord) = call(queue, 0x50);
        retrieve(queue, std::ptr::null_mut());
        retrieve(queue, &mut out.record);
        assert_eq!((out.before, out.after), (17, 29));
        assert_eq!(out.record.failure_count, 2);
        let metadata = &out.record.first;
        assert_eq!(metadata.result, END_OF_FILE);
        assert_eq!(metadata.command_type, 0);
        assert_eq!(metadata.options, first.options);
        assert_eq!(metadata.destination, short as usize);
        assert_eq!(metadata.destination_size, 16);
        assert_eq!(metadata.file_offset, 60);
        assert_eq!(metadata.source, 0);
        assert_eq!(metadata.source_size, 16);
        assert_eq!(metadata.tag, 23);
        assert_eq!(&metadata.name[..11], b"short read\0");
        let filename: Vec<_> = dir
            .path()
            .join("data.bin")
            .as_os_str()
            .encode_wide()
            .take(metadata.filename.len() - 1)
            .chain([0])
            .collect();
        assert_eq!(&metadata.filename[..filename.len()], filename.as_slice());
        assert_eq!(poll_event(&event), WAIT_TIMEOUT);
        retrieve(queue, &mut out.record);
        assert_eq!(
            (out.record.failure_count, out.record.first.command_type),
            (0, -1),
        );

        // A later failure starts a fresh record. Retrieve while the event
        // is still signaled, before a wait can consume the signal.
        enqueue(queue, &first);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), END_OF_FILE);
        retrieve(queue, &mut out.record);
        assert_eq!(
            (out.record.failure_count, out.record.first.result),
            (1, END_OF_FILE),
        );
        assert_eq!(poll_event(&event), WAIT_TIMEOUT);

        for object in [queue, idle_queue, array, file] {
            let release: extern "system" fn(*mut c_void) -> u32 = call(object, 0x10);
            assert_eq!(release(object), 0);
        }
        // Closing the queue releases its handle. The caller's stays valid.
        assert_eq!(poll_event(&event), WAIT_TIMEOUT);
    }

    #[test]
    fn half_a_queue_submits_itself() {
        let (_dir, file) = file_with(&pattern(64));
        let create: extern "system" fn(*mut c_void, *const QueueDesc, *const Guid, Out) -> HRESULT =
            call(factory(), 0x50);
        let desc = QueueDesc {
            source_type: 0,
            capacity: 4,
            priority: 1,
            name: c"small".as_ptr(),
            unknown: 1,
        };
        let mut queue = std::ptr::null_mut();
        assert_eq!(create(factory(), &desc, std::ptr::null(), &mut queue), S_OK);
        let array = status_array(1);
        let mut out = [0u8; 16];
        enqueue(queue, &request(0, out.as_mut_ptr().cast(), 16, file, 8, 16));
        enqueue_status(queue, array, 0); // the second of four: no submit needed
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, pattern(64)[8..24]);
    }

    /// Lanes without workers: what `start` picks, and when statuses land.
    fn lanes(staging: u64) -> Lanes {
        Lanes {
            lanes: HashMap::new(),
            next_id: 0,
            next_seq: 0,
            staged: 0,
            staging,
            starved: false,
            clock: [0; 3],
            now: 0,
            lane_now: [0; 3],
            done: Default::default(),
            refused: 0,
            bcpack: Default::default(),
            since: None,
        }
    }

    fn read(lanes: &mut Lanes, id: u64, options: u64, size: u32) {
        let mut r = request(options, std::ptr::null_mut(), 0, std::ptr::null(), 0, 0);
        (r.destination_size, r.source_size) = (size, size);
        let lane = lanes.lanes.get_mut(&id).unwrap();
        lane.unsubmitted.push(Command::Read(r, ReadNote::default()));
    }

    #[test]
    fn priorities_share_by_bytes_about_tenfold() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 4096).unwrap());
        l.lanes.insert(1, Lane::new(1, 4096).unwrap());
        l.lanes.insert(2, Lane::new(2, 4096).unwrap()); // real-time: as high
        for _ in 0..400 {
            for id in 0..3 {
                read(&mut l, id, 0, 4096);
            }
        }
        (0..3).for_each(|id| l.submit(id));
        let mut started = [0u32; 3];
        for _ in 0..330 {
            let job = l.start().unwrap();
            started[job.lane as usize] += 1;
            l.finish(&job, None);
        }
        // Normal gets about a tenth of what high and real-time get together.
        assert_eq!(started[1] + started[2], 300, "{started:?}");
        assert_eq!(started[0], 30, "{started:?}");
        assert!(started[1].abs_diff(started[2]) <= 1, "{started:?}");
    }

    #[test]
    fn staging_bounds_compressed_bytes_in_flight() {
        let mut l = lanes(100);
        l.lanes.insert(0, Lane::new(0, 64).unwrap());
        read(&mut l, 0, ZLIB, 60);
        read(&mut l, 0, ZLIB, 60);
        read(&mut l, 0, 0, 1000); // raw: straight to its destination
        l.submit(0);
        let first = l.start().unwrap();
        assert!(l.start().is_none(), "60 + 60 > 100");
        l.finish(&first, None);
        let second = l.start().unwrap();
        let raw = l.start().unwrap();
        assert_eq!((second.staging, raw.staging), (60, 0));
        // Alone, a request larger than the buffer still runs.
        read(&mut l, 0, ZLIB, 500);
        l.submit(0);
        assert!(l.start().is_none());
        l.finish(&second, None);
        assert!(l.start().is_some());
    }

    #[test]
    fn a_status_waits_for_every_earlier_read() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 64).unwrap());
        let array = status_array(2).cast::<Instance<StatusArray>>();
        let status = |index| {
            add_ref_instance(array);
            Command::Status(Status {
                array,
                index,
                enqueued: 0,
            })
        };
        read(&mut l, 0, 0, 16);
        read(&mut l, 0, 0, 16);
        l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(0));
        read(&mut l, 0, 0, 16);
        l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(1));
        // SAFETY: an array we created.
        for entry in unsafe { &(*array).state.entries }.iter() {
            entry.store(E_PENDING, Ordering::Release);
        }
        l.submit(0);
        let jobs: Vec<Job> = (0..3).map(|_| l.start().unwrap()).collect();
        // The last read and the first finish; the second then fails.
        l.finish(&jobs[2], None);
        l.finish(&jobs[0], None);
        let results = || (result(array.cast(), 0), result(array.cast(), 1));
        assert_eq!(results(), (E_PENDING, E_PENDING));
        let failure = ReadFailure::new(&jobs[1].request, INVALID_DATA);
        l.finish(&jobs[1], Some(Box::new(failure)));
        assert_eq!(results(), (INVALID_DATA, S_OK));
    }

    #[test]
    fn error_event_reports_failures_in_enqueue_order_without_status_entries() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 64).unwrap());
        read(&mut l, 0, MEMORY_SOURCE, 16);
        read(&mut l, 0, MEMORY_SOURCE, 16);
        l.submit(0);
        let data = [0u8; 16];
        let mut first = l.start().unwrap();
        first.request.source = data.as_ptr().cast();
        let second = l.start().unwrap();
        let handle = l.lanes[&0].errors.duplicate_event();
        assert!(!handle.is_null());
        // SAFETY: a new duplicate owned by the test.
        let event = unsafe { OwnedHandle::from_raw_handle(handle) };
        let failure = ReadFailure::new(&second.request, INVALID_DATA);
        l.finish(&second, Some(Box::new(failure)));
        assert_eq!(poll_event(&event), WAIT_TIMEOUT);
        assert_eq!(l.lanes.get_mut(&0).unwrap().errors.take().failure_count, 0);

        let failure = ReadFailure::new(&first.request, END_OF_FILE);
        l.finish(&first, Some(Box::new(failure)));
        assert_eq!(poll_event(&event), WAIT_OBJECT_0);
        let record = l.lanes.get_mut(&0).unwrap().errors.take();
        assert_eq!(
            (record.failure_count, record.first.result),
            (2, END_OF_FILE)
        );
        assert_eq!(record.first.source, data.as_ptr() as usize);
        assert_eq!(record.first.file_offset, 0);
        assert!(record.first.filename.iter().all(|&unit| unit == 0));

        // A status entry submitted after retrieval still sees the failure.
        let array = status_array(1).cast::<Instance<StatusArray>>();
        add_ref_instance(array);
        let lane = l.lanes.get_mut(&0).unwrap();
        lane.unsubmitted.push(Command::Status(Status {
            array,
            index: 0,
            enqueued: 0,
        }));
        l.submit(0);
        assert_eq!(result(array.cast(), 0), END_OF_FILE);
        assert_eq!(release(array), 0);
    }

    #[test]
    fn zlib_then_bcpack() {
        with_test_tables();
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&BC4_STREAM, 6);
        let (_dir, file) = file_with(&packed);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        let options = ZLIB | 4 << 1;
        let mut r = request(options, out.as_mut_ptr().cast(), 16, file, 0, packed.len());
        r.intermediate_size = BC4_STREAM.len() as u32;
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, bc4_blocks());
    }

    /// BCPack tables we made (uniform models; BC4 index pairs: low 3 bits,
    /// next 3 bits), in a folder kept for the test process.
    fn with_test_tables() {
        static DIR: OnceLock<()> = OnceLock::new();
        DIR.get_or_init(|| {
            let mut t = Vec::new();
            for m in 0..128 {
                let g = m / 16;
                for b in 0..9 {
                    let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
                    t.extend_from_slice(&f.to_le_bytes());
                }
            }
            t.extend([0; 256 + 64 + 256]);
            t.extend((0..64u8).map(|v| (v & 7) | (v >> 3) << 4));
            t.resize(bcpack::TABLES_SIZE, 0);
            let dir = tempfile::tempdir().unwrap().keep();
            std::fs::write(dir.join("tables.bin"), t).unwrap();
            use_bcpack_tables(dir);
        });
    }

    /// Two BC4 blocks from the format's test encoder: endpoints 10, -3 with
    /// indices p -> p & 7, then the same block plus 1 everywhere.
    const BC4_STREAM: [u8; 88] = [
        0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 4, 4, 4, 4, 1, 1, 1, 1, 0x10, 0x10, 0x10, 0x10, 2, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0x14, 5, 0x20, 0x68, 0xb0, 0xf8, 3, 3, 2, 2, 0x24, 0x24, 0x24,
        0x24, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0x10, 0x10, 0x10, 0x10, 0, 0, 0, 0,
        0x20, 0x68, 0xb0, 0xf8, 0x24, 0x24, 0x24, 0x24, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// What `BC4_STREAM` unpacks to.
    fn bc4_blocks() -> [u8; 16] {
        let code = [0u64, 2, 3, 4, 5, 6, 7, 1];
        let block = |e0: u64, e1: u64, add: usize| {
            let bits = (0..16).fold(0, |a, p| a | code[(p + add) & 7] << (16 + 3 * p));
            (e0 | e1 << 8 | bits).to_le_bytes()
        };
        let mut out = [0; 16];
        out[..8].copy_from_slice(&block(138, 125, 0));
        out[8..].copy_from_slice(&block(139, 126, 1));
        out
    }

    #[test]
    fn bcpack_requests_unpack_into_the_destination() {
        with_test_tables();
        let mut contents = pattern(100);
        contents.extend_from_slice(&BC4_STREAM);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        enqueue(
            queue,
            &request(4 << 1, out.as_mut_ptr().cast(), 16, file, 100, 88),
        );
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, bc4_blocks());
    }

    #[test]
    fn a_closed_queue_completes_nothing() {
        let (_dir, file) = file_with(&pattern(64));
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        enqueue(queue, &request(0, out.as_mut_ptr().cast(), 16, file, 0, 16));
        enqueue_status(queue, array, 0);
        void(queue, 0x40);
        void(queue, 0x30);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(result(array, 0), E_PENDING);
        assert_eq!(out, [0; 16]);
    }

    /// Close blocks until every submitted request has completed or failed,
    /// and writes their status entries.
    #[test]
    fn closing_a_queue_completes_what_was_submitted() {
        let data = pattern(8 << 20);
        let (_dir, file) = file_with(&data);
        let (queue, array) = (new_queue(0), status_array(1));
        let chunk = 256 << 10;
        let mut out = vec![0u8; data.len()];
        for k in 0..32 {
            let dst = out[k * chunk..].as_mut_ptr().cast();
            enqueue(
                queue,
                &request(0, dst, chunk, file, (k * chunk) as u64, chunk),
            );
        }
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        void(queue, 0x40);
        assert_eq!(result(array, 0), S_OK);
        assert!(out == data);
    }

    #[test]
    fn physical_page_destinations() {
        use crate::modules::xmem;
        let mut pages = [0usize; 2];
        let mut n = 2;
        assert_eq!(
            xmem::XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()),
            1
        );
        // Where the title would see them: a reservation of 64 KiB pages.
        let mapped = xmem::XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            0x2000 | 0x2040_0000, // reserve, 64 KiB pages
            0,
            0x04,
        );
        assert!(!mapped.is_null());
        assert_eq!(
            xmem::XMemMapPhysicalPages(mapped, 2, pages.as_ptr()),
            mapped
        );

        let plain = pattern(100);
        let (queue, array) = (new_queue(1), status_array(1));
        let mut r = request(
            PAGE_DESTINATION | MEMORY_SOURCE,
            pages.as_mut_ptr().cast(),
            plain.len(),
            plain.as_ptr().cast(),
            0,
            plain.len(),
        );
        r.page_offset = (PAGE - 40) as u64; // across the page boundary
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        // SAFETY: both pages are mapped read-write.
        let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>().add(PAGE - 40), 100) };
        assert_eq!(seen, plain);
    }
    #[test]
    fn zlib_into_pages_in_place_and_across_allocations() {
        use crate::modules::xmem;
        let allocate = |n: usize| {
            let (mut pages, mut given) = (vec![0usize; n], n);
            assert_eq!(
                xmem::XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
                1
            );
            pages
        };
        let (a, b) = (allocate(3), allocate(1));
        let plain = pattern(2 * PAGE + 100);
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
        // One allocation's pages in order (one window), then pages of two
        // (through a buffer), both from 50 bytes into the first.
        let mut destinations = [a.clone(), vec![a[0], a[1], b[0]]];
        let (queue, array) = (new_queue(1), status_array(1));
        for pages in &mut destinations {
            let mut r = request(
                ZLIB | PAGE_DESTINATION | MEMORY_SOURCE,
                pages.as_mut_ptr().cast(),
                plain.len(),
                packed.as_ptr().cast(),
                0,
                packed.len(),
            );
            r.page_offset = 50;
            enqueue(queue, &r);
        }
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        for pages in &destinations {
            let mapped = xmem::XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                0x2000 | 0x2040_0000, // reserve, 64 KiB pages
                0,
                0x04,
            );
            assert!(!mapped.is_null());
            assert_eq!(
                xmem::XMemMapPhysicalPages(mapped, 3, pages.as_ptr()),
                mapped
            );
            // SAFETY: all three pages are mapped read-write.
            let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), 3 * PAGE) };
            assert!(seen[..50].iter().all(|&b| b == 0));
            assert_eq!(seen[50..50 + plain.len()], plain);
            assert_eq!(
                xmem::XMemMapPhysicalPages(mapped, 3, std::ptr::null()),
                mapped
            );
        }
        assert_eq!(xmem::XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(xmem::XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// A title releasing once too often: the count stays 0 and nothing is
    /// freed again (the old `fetch_sub(1) - 1` overflowed).
    #[test]
    fn release_past_zero_leaves_the_count_at_zero() {
        let object = Instance::create(Object::new("dstorage queue", &QUEUE_VTABLE), ())
            .cast::<Instance<()>>();
        assert_eq!(add_ref_instance(object), 2);
        assert_eq!(release(object), 1);
        // SAFETY: still allocated; the test holds the object.
        let refs = &unsafe { &*object }.refs;
        refs.store(0, Ordering::Relaxed);
        assert_eq!(release(object), 0);
        assert_eq!(refs.load(Ordering::Relaxed), 0);
        // SAFETY: made by `Instance::create` and not freed by `release`.
        drop(unsafe { Box::from_raw(object) });
    }

    #[test]
    fn a_page_list_filled_after_its_request_starts() {
        use crate::modules::xmem;
        let (mut given, mut page) = (1usize, [0usize; 1]);
        assert_eq!(
            xmem::XMemAllocatePhysicalPages(0, &mut given, page.as_mut_ptr()),
            1
        );
        let mapped = xmem::XMemVirtualAlloc(
            std::ptr::null_mut(),
            PAGE,
            0x2000 | 0x2040_0000, // reserve, 64 KiB pages
            0,
            0x04,
        );
        assert!(!mapped.is_null());
        assert_eq!(xmem::XMemMapPhysicalPages(mapped, 1, page.as_ptr()), mapped);

        // The title's list, still empty when the request is submitted.
        let list = Box::leak(Box::new([0usize; 1]));
        let at = std::ptr::from_mut(list) as usize;
        let plain = pattern(4096);
        let (queue, array) = (new_queue(1), status_array(1));
        let r = request(
            PAGE_DESTINATION | MEMORY_SOURCE,
            list.as_mut_ptr().cast(),
            plain.len(),
            plain.as_ptr().cast(),
            0,
            plain.len(),
        );
        let number = page[0];
        let filler = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(2));
            // SAFETY: the leaked list, alive for the whole test.
            unsafe { std::ptr::write_volatile(at as *mut usize, number) };
        });
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        filler.join().unwrap();
        // SAFETY: the page is mapped read-write.
        let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), plain.len()) };
        assert_eq!(seen, plain);
    }

    /// What ways of reading a file give, at `VICTORY_BENCH_FILE` (a file of
    /// the title's, only read): 24 threads, requests of 48 KiB at random,
    /// first from places nobody read lately, then the same once more.
    /// `VICTORY_BENCH_FILE=... cargo test --release -p victory-hle --lib
    /// read_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "a benchmark"]
    fn read_bench() {
        const THREADS: usize = 24;
        const SIZE: usize = 48 << 10;
        const READS: usize = 12_000;
        let path = std::env::var("VICTORY_BENCH_FILE").expect("VICTORY_BENCH_FILE");
        let len = std::fs::metadata(&path).unwrap().len();
        let mut seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            | 1;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..2 {
            for mode in [Io::One, Io::Handles, Io::Overlapped, Io::Direct] {
                let file = Files::open_for_test(&path, mode);
                let at: Vec<u64> = (0..READS)
                    .map(|_| (random() % (len - SIZE as u64)) & !15)
                    .collect();
                for pass in ["cold", "again"] {
                    let next = std::sync::atomic::AtomicUsize::new(0);
                    let busy = AtomicU64::new(0);
                    let start = std::time::Instant::now();
                    std::thread::scope(|s| {
                        for _ in 0..THREADS {
                            s.spawn(|| {
                                let mut out = vec![0u8; SIZE];
                                while let Some(&at) = at.get(next.fetch_add(1, Ordering::Relaxed)) {
                                    let began = std::time::Instant::now();
                                    let mut done = 0;
                                    while done < SIZE {
                                        done += file
                                            .read_at(&mut out[done..], at + done as u64)
                                            .unwrap();
                                    }
                                    busy.fetch_add(
                                        began.elapsed().as_micros() as u64,
                                        Ordering::Relaxed,
                                    );
                                }
                            });
                        }
                    });
                    let took = start.elapsed().as_secs_f64();
                    println!(
                        "{round} {mode:?} {pass}: {:.0} MB/s, {:.2} ms a read",
                        (READS * SIZE) as f64 / took / 1e6,
                        busy.load(Ordering::Relaxed) as f64 / READS as f64 / 1e3
                    );
                }
            }
        }
    }

    /// Every way of reading a file gives its bytes, also past its end.
    #[test]
    fn every_way_of_reading_gives_the_same_bytes() {
        let plain = pattern(300_001);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, &plain).unwrap();
        for mode in [Io::One, Io::Handles, Io::Overlapped, Io::Direct] {
            let file = Files::open_for_test(path.to_str().unwrap(), mode);
            for (at, len) in [
                (0, 4096),
                (1, 100),
                (4095, 2),
                (8192, 200_000),
                (12_345, 250_000),
                (299_990, 11),
                (300_000, 1),
            ] {
                let mut out = vec![0u8; len];
                let mut done = 0;
                while done < len {
                    let n = file.read_at(&mut out[done..], (at + done) as u64).unwrap();
                    assert_ne!(n, 0, "{mode:?} {at} {len}");
                    done += n;
                }
                assert!(out == plain[at..at + len], "{mode:?} {at} {len}");
            }
            let mut out = [0u8; 16];
            assert_eq!(file.read_at(&mut out, 299_999).unwrap(), 2, "{mode:?}");
            assert_eq!(file.read_at(&mut out, 300_001).unwrap(), 0, "{mode:?}");
            assert_eq!(file.read_at(&mut out, 400_000).unwrap(), 0, "{mode:?}");
        }
    }
}
