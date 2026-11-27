//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATES, ID3D12Resource,
};

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of submissions not yet copied back may be writing,
    /// with the submission that declared them: title memory there is older
    /// than the copy.
    pending: Mutex<Vec<(u64, usize, usize)>>,
    /// Per submission not yet copied back (in submission order): the copy's
    /// bytes before it ran, from an offset on, covering the ranges its
    /// shaders write: what the GPU wrote differs.
    ///
    /// One submission at a time (the default) has one entry. With several
    /// in flight (`VICTORY_IN_FLIGHT`), later submissions may declare
    /// ranges an earlier one's shaders still write; then the bytes are
    /// handed on in order ("rolling" snapshots, see [`Shadow::copy_back`]):
    /// the earlier copy-back puts what it saw into title memory and into the
    /// later snapshots, and leaves the copy itself alone (the later GPU work
    /// may be writing it); the last one to cover the bytes brings the copy
    /// in line with title memory, as one submission at a time does.
    before: Mutex<Vec<(u64, Before)>>,
    /// Ranges GPU work of submissions in flight changes in place without
    /// declaring them, with the submission ([`Shadow::refresh_in_place`]).
    /// Empty without `VICTORY_IN_FLIGHT`.
    in_place: Mutex<Vec<(u64, usize, usize)>>,
}

#[derive(Default)]
struct Before {
    snapshot: Option<(usize, Vec<u8>)>,
    unchanged: UnchangedPages,
}

/// Pages already compared in full against this submission's snapshot.
#[derive(Default)]
struct UnchangedPages {
    start: usize,
    end: usize,
    pages: Vec<bool>,
}

impl UnchangedPages {
    fn begin(&mut self, start: usize, end: usize) {
        if self.pages.is_empty() {
            self.start = start;
            self.end = end;
            self.pages.resize(end.div_ceil(PAGE) - start / PAGE, false);
        }
    }

    fn differs(&mut self, address: usize, current: &[u8], before: &[u8]) -> bool {
        let page = address / PAGE;
        let unchanged = &mut self.pages[page - self.start / PAGE];
        if *unchanged {
            return false;
        }
        if current != before {
            return true;
        }
        // A later range can touch bytes a partial comparison missed. At
        // the edges, only bytes covered by the snapshot can be compared.
        *unchanged = address == (page * PAGE).max(self.start)
            && address + current.len() == ((page + 1) * PAGE).min(self.end);
        false
    }

    #[cfg(test)]
    fn clear(&mut self) {
        self.pages.clear();
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The copy's bytes for `len` bytes of title memory at `address`, as
    /// they are now (tests look at what a submission's shaders left there).
    #[cfg(all(test, feature = "diagnostics"))]
    pub(crate) fn host_bytes(&self, address: u64, len: usize) -> Vec<u8> {
        assert!(address >= self.start && address + len as u64 <= self.start + self.len);
        // SAFETY: within the copy's persistent mapping.
        unsafe {
            std::slice::from_raw_parts(self.data.add((address - self.start) as usize), len).to_vec()
        }
    }

    /// Writes `bytes` into the copy at a title address, as the GPU would
    /// (tests of copy-back).
    #[cfg(test)]
    fn gpu_wrote(&self, address: u64, bytes: &[u8]) {
        assert!(address >= self.start && address + bytes.len() as u64 <= self.start + self.len);
        // SAFETY: within the copy's persistent mapping.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.data.add((address - self.start) as usize),
                bytes.len(),
            );
        }
    }

    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// `len` bytes at a title address from the host copy, for the log: the
    /// bytes the copy holds, as of its last refresh (None outside the buffer).
    pub(crate) fn bytes(&self, address: u64, len: u64) -> Option<&[u8]> {
        if address < self.start || len == 0 || len > self.left(address) {
            return None;
        }
        let at = usize::try_from(address - self.start).ok()?;
        let len = usize::try_from(len).ok()?;
        // SAFETY: inside the mapping, which is as long as the buffer.
        Some(unsafe { std::slice::from_raw_parts(self.data.add(at), len) })
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        self.refresh_as(list, address, len, D3D12_RESOURCE_STATE_GENERIC_READ);
    }

    /// Refresh a range and prepare its host buffer for the specified GPU use.
    pub(crate) fn refresh_as(
        self: &Arc<Self>,
        list: &HostList,
        address: u64,
        len: u64,
        access: D3D12_RESOURCE_STATES,
    ) {
        list.use_shadow(&self.buffer, access);
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    /// [`Shadow::refresh_as`] of a range the list's GPU work changes in the
    /// copy without declaring it (indirect arguments clamped in place: what
    /// it writes must not go back to title memory). With submissions in
    /// flight, the range is noted while the submission is: no refresh of a
    /// later one copies title memory over it meanwhile (a later one that
    /// asks for it first waits for this one), and no earlier copy-back
    /// writes the copy there.
    pub(crate) fn refresh_in_place(
        self: &Arc<Self>,
        list: &HostList,
        address: u64,
        len: u64,
        access: D3D12_RESOURCE_STATES,
    ) {
        self.refresh_as(list, address, len, access);
        if crate::host::in_flight_limit() > 1 {
            let len = len.min(self.left(address));
            list.defer_in_place(self, address as usize, len as usize);
        }
    }

    /// A submission in flight (`sub`) changes `[start, end)` of the copy in
    /// place ([`Shadow::refresh_in_place`]).
    pub(crate) fn in_place(&self, sub: u64, start: usize, end: usize) {
        self.in_place
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((sub, start, end));
    }

    /// The submission's GPU work is done with what it changes in place.
    pub(crate) fn in_place_done(&self, sub: u64) {
        self.in_place
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|r| r.0 != sub);
    }

    /// With submissions in flight: the value (`width` bytes, 4 or 8) the
    /// GPU work of submission `sub` has left at title address `address` in
    /// the copy, if it differs from that submission's snapshot there (so
    /// that work, not an older one, wrote it). None if the submission did
    /// not declare it or has not changed it (yet).
    pub(crate) fn gpu_changed(&self, address: usize, width: usize, sub: u64) -> Option<u64> {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        if address < start || address + width > end || !matches!(width, 4 | 8) {
            return None;
        }
        let at = address - start;
        let all = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (taken, before) = all.iter().find(|(s, _)| *s == sub)?.1.snapshot.as_ref()?;
        if at < *taken || at + width > taken + before.len() {
            return None;
        }
        let mut now = [0u8; 8];
        for (i, b) in now[..width].iter_mut().enumerate() {
            // SAFETY: inside the mapping (checked above); the GPU may be
            // writing it, so read it as it is.
            *b = unsafe { self.data.add(at + i).read_volatile() };
        }
        (now[..width] != before[at - taken..at - taken + width]).then(|| u64::from_le_bytes(now))
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
            in_place: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        #[cfg(feature = "diagnostics")]
        note_writer(address, len);
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame).
    ///
    /// `sub` names the submission (increasing in submission order); each
    /// has its own snapshot. Bytes an earlier submission in flight still
    /// writes are taken from the copy as it is now, and replaced by what
    /// that submission's copy-back sees (see [`Shadow::copy_back`]).
    pub(crate) fn writing(&self, address: usize, len: usize, sub: u64) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((sub, address, address + len));
        let (from, to) = self.inside(address, len);
        // SAFETY: inside the mapping.
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut all = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let at = match all.iter().position(|(s, _)| *s >= sub) {
            Some(i) if all[i].0 == sub => i,
            Some(i) => {
                all.insert(i, (sub, Before::default()));
                i
            }
            None => {
                all.push((sub, Before::default()));
                all.len() - 1
            }
        };
        let before = &mut all[at].1.snapshot;
        let (start, end) = match &*before {
            Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
            None => (from, to),
        };
        if let Some((at, bytes)) = &*before
            && (*at, at + bytes.len()) == (start, end)
        {
            return;
        }
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        match before.take() {
            Some((at, bytes)) => {
                grown.extend_from_slice(host(start, at));
                grown.extend_from_slice(&bytes);
                grown.extend_from_slice(host(at + bytes.len(), end));
                give_back(bytes);
            }
            None => grown.extend_from_slice(host(start, end)),
        }
        *before = Some((start, grown));
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    ///
    /// `gap` (`Some`): the submitting thread is not waiting (asynchronous
    /// submission), so the title's busiest thread writes these pages while
    /// this runs. Then only the runs of bytes the GPU changed are written,
    /// runs less than `gap` bytes apart joined (a whole piece read and
    /// written back loses the title's writes in between), the blocks stay
    /// open until the write lands (another thread's refresh guards them
    /// again otherwise, and the write fails), and pages are taken as
    /// current as of before they were read.
    ///
    /// `sub`: the submission (as given to [`Shadow::writing`]). Copy-backs
    /// run in submission order. A piece a later submission in flight also
    /// declared is handed on instead ([`Shadow::hand_on`]): its GPU work
    /// may be writing the copy, so the copy is only read.
    pub(crate) fn copy_back(
        &self,
        address: usize,
        len: usize,
        gap: Option<usize>,
        sub: u64,
    ) -> usize {
        let overlapped = gap.is_some();
        let gap = gap.unwrap_or(1).max(1);
        let mut all = self.before.lock().unwrap_or_else(|e| e.into_inner());
        // Ranges later submissions declared or change in place (none with
        // one at a time).
        let mut later: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|p| p.0 > sub)
            .map(|p| (p.1, p.2))
            .collect();
        later.extend(
            self.in_place
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .filter(|p| p.0 > sub)
                .map(|p| (p.1, p.2)),
        );
        let mine = all.iter().position(|(s, _)| *s == sub);
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some(i) = mine
            && let (head, rest) = all.split_at_mut(i + 1)
            && let Before {
                snapshot,
                unchanged,
            } = &mut head[i].1
            && let Some((taken, before)) = &*snapshot
            && *taken <= from
            && to <= taken + before.len()
            && from < to
        {
            unchanged.begin(start + taken, start + taken + before.len());
            let before = &before[from - taken..to - taken];
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            let watched = first < last && self.watched.load(Ordering::Relaxed) != NOT_WATCHED;
            // Overlapped: asked before title memory is read, so a title write
            // after the read shows as newer (our own writes do too, and those
            // pages are copied again at the next refresh).
            let asked_before = if overlapped && watched {
                crate::watch::writes(first * PAGE, last * PAGE, true)
            } else {
                None
            };
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            // An unchanged page is never modified by copy-back, so later
            // overlapping ranges can skip it for the rest of the submission.
            let mut title = None;
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PAGE - (start + at) % PAGE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                if later
                    .iter()
                    .any(|&(a, e)| a < start + at + n && e > start + at)
                {
                    written +=
                        self.hand_on(start + at, host, old, &mut title, rest, gap, overlapped);
                    at += n;
                    continue;
                }
                if unchanged.differs(start + at, host, old)
                    && let title = &mut title.get_or_insert_with(|| vec![0u8; PAGE])[..n]
                    && crate::host::read_into((start + at) as *const u8, title)
                {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    let (changed, runs) = merge(start + at, title, host, old, gap, overlapped);
                    host.copy_from_slice(title);
                    if changed {
                        written += put_back(start + at, title, runs, overlapped);
                    }
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            if watched
                && !synced.is_empty()
                && let Some((now, _)) = match overlapped {
                    true => asked_before,
                    false => crate::watch::writes(first * PAGE, last * PAGE, true),
                }
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        // Every step of this submission that declared the range is done with
        // it, and its snapshot with its last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (sub, address, address + len));
        if !pending.iter().any(|r| r.0 == sub)
            && let Some(i) = mine
            && let (_, Before { snapshot, .. }) = all.remove(i)
            && let Some((_, bytes)) = snapshot
        {
            give_back(bytes);
        }
        written
    }

    /// [`Shadow::copy_back`] of one piece (at title address `address`) that
    /// a later submission in flight also declared: the GPU's bytes (the copy
    /// against `old`, this submission's snapshot) go into title memory where
    /// it differs, as there, but the copy is read once and left alone (the
    /// later GPU work may be writing it; what it writes after the read shows
    /// in its own copy-back), and the later snapshots take the bytes read:
    /// title memory holds them now, so the later copy-back does not put
    /// them back over what the title writes meanwhile. The page is not
    /// counted current (a refresh copies it again once nothing declares it).
    #[allow(clippy::too_many_arguments)]
    fn hand_on(
        &self,
        address: usize,
        host: &[u8],
        old: &[u8],
        title: &mut Option<Vec<u8>>,
        later: &mut [(u64, Before)],
        gap: usize,
        overlapped: bool,
    ) -> usize {
        let seen = host.to_vec();
        let n = seen.len();
        let mut written = 0;
        if seen.as_slice() != old
            && let title = &mut title.get_or_insert_with(|| vec![0u8; PAGE])[..n]
            && crate::host::read_into(address as *const u8, title)
        {
            let (changed, runs) = merge(address, title, &seen, old, gap, overlapped);
            if changed {
                written = put_back(address, title, runs, overlapped);
            }
        }
        let at = address - self.start as usize;
        for (_, b) in later {
            if let Some((taken, bytes)) = &mut b.snapshot {
                let (a, e) = (at.max(*taken), (at + n).min(*taken + bytes.len()));
                if a < e {
                    bytes[a - *taken..e - *taken].copy_from_slice(&seen[a - at..e - at]);
                }
            }
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|&(_, a, e)| (a, e))
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        // And what GPU work in flight changes in place (nothing without
        // `VICTORY_IN_FLIGHT`; a refresh that asks for it waited for that
        // work before it got here).
        busy.extend(
            self.in_place
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|&(_, a, e)| (a, e))
                .filter(|&(a, e)| a < to && e > from),
        );
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// With submissions in flight (`VICTORY_IN_FLIGHT`), after a refresh of
    /// `[from, to)`: puts the done writes of earlier submissions still in
    /// flight that lie there into the copy. Title memory gets them only at
    /// their completion, but this submission's GPU work runs after theirs
    /// on the host queue, where a console GPU would see them written.
    /// Bytes shaders declared are left alone. Returns the writes put in.
    pub(crate) fn patch_flying(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let writes = crate::timeline::flying_writes(from.max(start), to.min(end));
        if writes.is_empty() {
            return 0;
        }
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        pending.extend(
            self.in_place
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .copied(),
        );
        let mut put = 0;
        for (_, at, bytes, value) in writes {
            if at < start
                || at + bytes > end
                || at < from
                || at + bytes > to
                || pending.iter().any(|&(_, a, e)| a < at + bytes && e > at)
            {
                continue;
            }
            let value = value.to_le_bytes();
            // SAFETY: inside the mapping (checked above); 4 or 8 bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    value.as_ptr(),
                    self.data.add(at - start),
                    bytes.min(8),
                );
            }
            put += 1;
        }
        put
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            // Only mark the shadow permanently unwatched if it has never been
            // observed as watched. A sub-range that lands entirely in an
            // unmapped hole inside an already-watched view pool must not
            // disable write-watching for the rest of the pool.
            let _ =
                self.watched
                    .compare_exchange(0, NOT_WATCHED, Ordering::Relaxed, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

/// Writes a copy-back piece (`title`: title memory at `address` with the
/// GPU's bytes put in) into title memory: the runs of GPU-changed bytes
/// when `overlapped`, else the whole piece. Returns the bytes written.
fn put_back(address: usize, title: &[u8], runs: Vec<(usize, usize)>, overlapped: bool) -> usize {
    if overlapped {
        let mut written = 0;
        for (a, b) in runs {
            let to = address + a;
            crate::watch::kernel_will_write(to, b - a);
            crate::host::write_mapped(to as *mut u8, title[a..].as_ptr(), b - a);
            crate::watch::kernel_wrote(to, b - a);
            written += b - a;
        }
        written
    } else {
        crate::watch::writing(address, title.len());
        crate::host::write_mapped(address as *mut u8, title.as_ptr(), title.len());
        title.len()
    }
}

/// Copy-back as a three-way merge of one piece at title address `address`:
/// `title` (title memory now) takes `host`'s bytes (the GPU's copy) where
/// the GPU changed them against `old` (the snapshot before the submission),
/// by aligned dword, and only where the title has not changed that dword
/// since the snapshot either: a CPU write made while the GPU ran is newer
/// than the snapshot the GPU worked from, and wins (the title's memory may
/// have been reused for its own data in the meantime). Returns whether
/// `title` changed and, when `overlapped`, the runs of changed bytes with
/// gaps under `gap` joined. Diagnostics builds count the conflicts
/// (`copy-back conflicts` in the copy line); `VICTORY_COPY_BACK_GPU_WINS`
/// restores the old rule (the GPU's bytes always win) for comparison.
fn merge(
    address: usize,
    title: &mut [u8],
    host: &[u8],
    old: &[u8],
    gap: usize,
    overlapped: bool,
) -> (bool, Vec<(usize, usize)>) {
    #[cfg(feature = "diagnostics")]
    let gpu_wins = {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::switch::on("VICTORY_COPY_BACK_GPU_WINS"))
    };
    #[cfg(not(feature = "diagnostics"))]
    let gpu_wins = false;
    let n = title.len();
    let mut changed = false;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        let end = (i + 4 - (address + i) % 4).min(n);
        let (h, o) = (&host[i..end], &old[i..end]);
        if h != o && title[i..end] != *h {
            if gpu_wins || title[i..end] == *o {
                title[i..end].copy_from_slice(h);
                changed = true;
                if overlapped {
                    match runs.last_mut() {
                        Some((_, e)) if i - *e < gap => *e = end,
                        _ => runs.push((i, end)),
                    }
                }
            } else {
                CONFLICTS.fetch_add(1, Ordering::Relaxed);
            }
        }
        i = end;
    }
    (changed, runs)
}

/// Dwords copy-back left to a newer CPU write (see [`merge`]).
pub(crate) static CONFLICTS: AtomicU64 = AtomicU64::new(0);

/// Diagnostics (`VICTORY_WATCH_HIGH_ONES`): the host pipeline the
/// recording thread bound last, and the GPU write ranges recent draws and
/// dispatches declared with it, so a suspicious copy-back can name them.
#[cfg(feature = "diagnostics")]
mod writers {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    thread_local! {
        static PIPELINE: RefCell<String> = const { RefCell::new(String::new()) };
    }
    static RANGES: Mutex<VecDeque<(u64, u64, String)>> = Mutex::new(VecDeque::new());

    pub(crate) fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::switch::on("VICTORY_WATCH_HIGH_ONES"))
    }

    pub(crate) fn bound(name: &str) {
        if on() {
            PIPELINE.with(|p| name.clone_into(&mut p.borrow_mut()));
        }
    }

    pub(super) fn declared(address: u64, len: u64) {
        if !on() {
            return;
        }
        let name = PIPELINE.with(|p| p.borrow().clone());
        let mut ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner());
        if ranges.len() >= 8192 {
            ranges.pop_front();
        }
        ranges.push_back((address, len, name));
    }

    /// Pipelines that declared writes covering `at`, most recent first.
    pub(super) fn covering(at: u64) -> Vec<String> {
        let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner());
        let mut out: Vec<String> = Vec::new();
        for (a, n, name) in ranges.iter().rev() {
            if *a <= at && at < a + n && !out.contains(name) {
                out.push(name.clone());
                if out.len() >= 6 {
                    break;
                }
            }
        }
        out
    }
}

#[cfg(feature = "diagnostics")]
pub(crate) use writers::bound as note_bound_pipeline;

#[cfg(feature = "diagnostics")]
fn note_writer(address: u64, len: u64) {
    writers::declared(address, len);
}

/// Diagnostics (`VICTORY_WATCH_HIGH_ONES`): logs a write of ours into
/// title memory (copy-back, readback) that turns the high dword of an
/// 8-byte-aligned qword that held a title pointer into 0xFFFFFFFF, with
/// the qword before and after and the title buffer it belongs to, so a
/// write that clobbers the upper half of 64-bit values can be traced. At
/// most 200 lines.
#[cfg(feature = "diagnostics")]
pub(crate) fn high_ones(address: usize, bytes: &[u8]) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static LOGGED: AtomicU32 = AtomicU32::new(0);
    if !writers::on() || LOGGED.load(Ordering::Relaxed) >= 200 {
        return;
    }
    let end = address + bytes.len();
    let dword = |at: usize| -> u32 {
        u32::from_le_bytes(bytes[at - address..at - address + 4].try_into().unwrap())
    };
    // Qwords whose high dword lies wholly inside the write.
    let mut q = address.saturating_sub(4) & !7;
    while q + 8 <= end {
        let hi = q + 4;
        if hi >= address && dword(hi) == u32::MAX {
            let mut old = [0u8; 8];
            if crate::host::read_into(q as *const u8, &mut old) {
                let old = u64::from_le_bytes(old);
                // Only where a title pointer was (high dword 1-15, as
                // title addresses here are below 64 GiB): plain -1 data
                // over other data is common and not what is looked for.
                if (1..16).contains(&(old >> 32)) {
                    let lo = if q >= address { dword(q) } else { old as u32 };
                    let buffer = find(q as u64, 8).map(|s| format!("{:#x}+{:#x}", s.start, s.len));
                    tracing::warn!(
                        qword = format_args!("{q:#x}"),
                        before = format_args!("{old:#018x}"),
                        after = format_args!("{:#018x}", (u64::from(u32::MAX) << 32) | u64::from(lo)),
                        write = format_args!("{address:#x}+{:#x}", bytes.len()),
                        buffer = ?buffer,
                        writers = ?writers::covering(q as u64),
                        "a write of ours sets a qword's high dword to 0xFFFFFFFF"
                    );
                    if LOGGED.fetch_add(1, Ordering::Relaxed) >= 200 {
                        return;
                    }
                }
            }
        }
        q += 8;
    }
}

/// A refresh that copies none of its range is skipped: the writes it was
/// asked about were already in the copy, the range is outside the buffer,
/// or shaders were still writing all of it (that part is left to a later
/// refresh). Counted for the frame-rate line as one skipped refresh of
/// `len` bytes, so `0 skipped` there means no refresh was answered
/// without a copy, not that nothing was checked. A range copied in part is
/// not skipped; of what it did not copy, `unchanged` counts only the parts
/// this refresh checked, not the sub-ranges shaders still write.
#[cfg(any(feature = "diagnostics", test))]
pub(crate) fn skipped(len: usize, copied: usize) -> Option<u64> {
    (len > 0 && copied == 0).then_some(len as u64)
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive,
/// and any active epoch keeps it until that work completes.
pub(crate) fn unregister(resource: usize, start: u64) {
    let shadow = {
        let mut map = buffers();
        map.remove(&(start, resource)).and_then(|e| e.shadow)
    };
    if let Some(shadow) = shadow {
        crate::host::retire(shadow);
    }
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, if one has already been made: the same lookup as [`find`] without
/// making a copy, for paths that must not allocate one.
pub(crate) fn peek(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let map = buffers();
    let shadow = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .and_then(|(_, e)| e.shadow.clone());
    if shadow.is_some() {
        return shadow;
    }
    drop(map);
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    LOOSE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(start, len))
        .cloned()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    /// A refresh that copied nothing of its range is what the frame-rate
    /// line's `skipped` counts; one that copied any part of it is not, and
    /// an empty range is neither.
    #[test]
    fn a_refresh_that_copies_nothing_is_counted_as_skipped() {
        assert_eq!(skipped(0, 0), None);
        assert_eq!(skipped(4 << 20, 0), Some(4 << 20));
        assert_eq!(skipped(4 << 20, 1), None);
        assert_eq!(skipped(4 << 20, 4 << 20), None);
        // Nothing in the buffer: the range was outside it.
        assert_eq!(skipped(4096, 0), Some(4096));
    }

    #[test]
    fn partial_match_leaves_unchecked_bytes_visible() {
        let mut unchanged = UnchangedPages::default();
        let before = vec![0; PAGE];
        let mut host = before.clone();
        host[PAGE - 1] = 1;
        unchanged.begin(PAGE, 2 * PAGE);

        assert!(!unchanged.differs(PAGE, &host[..4], &before[..4]));
        assert!(!unchanged.pages[0]);
        assert!(unchanged.differs(PAGE, &host, &before));
        assert!(!unchanged.pages[0]);
    }

    #[test]
    fn snapshot_edges_are_cacheable_and_survive_later_ranges() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 3, &[1; 3], &[1; 3]));
        assert!(!unchanged.differs(PAGE, &[2; 4], &[2; 4]));
        assert_eq!(unchanged.pages, [true, true]);

        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 2, &[1; 2], &[1; 2]));
        assert!(!unchanged.differs(PAGE + 1, &[2; 2], &[2; 2]));
        assert_eq!(unchanged.pages, [true, true]);
    }

    #[test]
    fn next_submission_checks_the_same_page_again() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, 4);
        assert!(!unchanged.differs(0, &[0; 4], &[0; 4]));
        unchanged.clear();
        unchanged.begin(0, 4);
        assert!(unchanged.differs(0, &[0, 0, 0, 1], &[0; 4]));
    }

    // Both versions walk exactly the same declared ranges, in their
    // original order. Only the page comparison differs.
    fn scan(
        host: &[u8],
        before: &[u8],
        ranges: &[Range<usize>],
        mut differs: impl FnMut(usize, &[u8], &[u8]) -> bool,
    ) -> usize {
        let mut changed = 0;
        for range in ranges {
            let mut at = range.start;
            while at < range.end {
                let n = (PAGE - at % PAGE).min(range.end - at);
                changed += usize::from(differs(at, &host[at..at + n], &before[at..at + n]));
                at += n;
            }
        }
        changed
    }

    #[test]
    fn overlapping_ranges_match_uncached_comparisons() {
        let before = vec![0; 4 * PAGE];
        let mut host = before.clone();
        for at in [2, PAGE - 1, PAGE + 5, 3 * PAGE + 7] {
            host[at] = 1;
        }
        let ranges = [
            0..4,
            2..2 * PAGE + 3,
            PAGE - 2..4 * PAGE,
            2 * PAGE..3 * PAGE,
        ];
        let uncached = scan(&host, &before, &ranges, |_, host, before| host != before);
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, before.len());
        let cached = scan(&host, &before, &ranges, |at, host, before| {
            unchanged.differs(at, host, before)
        });
        assert_eq!(uncached, 6);
        assert_eq!(cached, uncached);
        assert_eq!(unchanged.pages, [false, false, true, false]);
    }

    // cargo test --release -p victory-gpu benchmark_overlapping_copy_back -- --ignored --nocapture
    #[test]
    #[ignore = "synthetic CPU release benchmark"]
    fn benchmark_overlapping_copy_back() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        const SIZE: usize = 128 << 20;
        let before = vec![0xA5; SIZE];
        let mut host = before.clone();
        for page in (0..SIZE / PAGE).step_by(64) {
            host[page * PAGE + PAGE / 2] ^= 0xFF;
        }
        let ranges: Vec<_> = (0..24).map(|i| i * PAGE + i..SIZE - i * 13).collect();
        let mut baseline = [Duration::ZERO; 5];
        let mut cached = [Duration::ZERO; 5];
        let mut unchanged = UnchangedPages::default();
        for (baseline, cached) in baseline.iter_mut().zip(&mut cached) {
            let started = Instant::now();
            let expected = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |_, host, before| host != before,
            );
            *baseline = started.elapsed();

            unchanged.clear();
            let started = Instant::now();
            unchanged.begin(0, before.len());
            let actual = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |at, host, before| unchanged.differs(at, host, before),
            );
            *cached = started.elapsed();
            assert_eq!(actual, expected);
        }
        baseline.sort_unstable();
        cached.sort_unstable();
        println!(
            "median of five runs, 24 overlapping ranges across 128 MiB: {:?} -> {:?} ({:.1}x)",
            baseline[2],
            cached[2],
            baseline[2].as_secs_f64() / cached[2].as_secs_f64(),
        );
    }

    #[cfg(windows)]
    #[test]
    fn unregister_drops_shadow_immediately_when_idle() {
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0001_usize;
        register(resource, start, data.len() as u64);
        let weak = {
            let shadow = find(start, 64).expect("shadow");
            Arc::downgrade(&shadow)
        };
        assert_eq!(weak.strong_count(), 1);
        unregister(resource, start);
        assert_eq!(weak.strong_count(), 0);
    }

    #[cfg(windows)]
    #[test]
    fn bytes_reads_the_copy_inside_the_buffer_only() {
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0002_usize;
        register(resource, start, data.len() as u64);
        let shadow = find(start, 64).expect("shadow");
        assert_eq!(shadow.bytes(start, 16).map(<[u8]>::len), Some(16));
        assert_eq!(
            shadow
                .bytes(start + data.len() as u64 - 16, 16)
                .map(<[u8]>::len),
            Some(16),
            "a range ending in the buffer's last bytes is inside it"
        );
        assert!(shadow.bytes(start - 4, 4).is_none(), "before the buffer");
        assert!(
            shadow.bytes(start + data.len() as u64 - 8, 16).is_none(),
            "past the buffer"
        );
        assert!(shadow.bytes(start, 0).is_none(), "nothing to read");
        unregister(resource, start);
    }

    /// Two submissions in flight declare the same range. The first one's
    /// copy-back puts its GPU bytes into title memory and hands them to the
    /// second's snapshot, leaving the copy alone; the second's then puts
    /// back only what its own GPU work wrote (not the first's bytes over
    /// what the title wrote after seeing them) and brings the copy in line.
    #[cfg(windows)]
    #[test]
    fn copy_backs_in_flight_hand_bytes_on_in_order() {
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let mut title = vec![0u8; 4096];
        let start = title.as_ptr() as u64;
        let resource = 0xdead_beef_0003_usize;
        register(resource, start, title.len() as u64);
        let shadow = find(start, 64).expect("shadow");
        let (a, n) = (start as usize, title.len());
        shadow.update(a, n);
        shadow.writing(a, n, 1);
        shadow.writing(a, n, 2);
        // The first submission's GPU work; the title writes elsewhere.
        shadow.gpu_wrote(start + 10, &[0xaa]);
        title[20] = 0x55;
        assert_eq!(shadow.update(a, n), 0, "declared: no refresh over it");
        shadow.copy_back(a, n, None, 1);
        assert_eq!((title[10], title[20]), (0xaa, 0x55));
        assert_eq!(
            shadow.bytes(start + 20, 1),
            Some(&[0][..]),
            "the copy is left to the second submission"
        );
        // The title overwrites the first one's output; the second one's
        // GPU work writes elsewhere.
        title[10] = 0x11;
        shadow.gpu_wrote(start + 30, &[0xbb]);
        shadow.copy_back(a, n, None, 2);
        assert_eq!((title[10], title[20], title[30]), (0x11, 0x55, 0xbb));
        for at in [10, 20, 30] {
            assert_eq!(
                shadow.bytes(start + at, 1),
                Some(&title[at as usize..][..1])
            );
        }
        assert!(
            shadow.before.lock().unwrap().is_empty(),
            "snapshots given back"
        );
        title[40] = 1;
        assert_eq!(shadow.update(a, n), n, "nothing declared any more");
        unregister(resource, start);
    }

    /// What a later submission in flight changes in place (clamped
    /// indirect arguments) is neither refreshed over nor written by an
    /// earlier copy-back; a shader value counts only once it differs from
    /// the submission's snapshot.
    #[cfg(windows)]
    #[test]
    fn in_place_ranges_and_shader_values_in_flight() {
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let mut title = vec![0u8; 4096];
        let start = title.as_ptr() as u64;
        let resource = 0xdead_beef_0005_usize;
        register(resource, start, title.len() as u64);
        let shadow = find(start, 64).expect("shadow");
        let (a, n) = (start as usize, title.len());
        shadow.update(a, n);
        shadow.writing(a, 256, 1);
        assert_eq!(shadow.gpu_changed(a + 16, 4, 1), None, "not written yet");
        shadow.gpu_wrote(start + 16, &5u32.to_le_bytes());
        assert_eq!(shadow.gpu_changed(a + 16, 4, 1), Some(5));
        assert_eq!(shadow.gpu_changed(a + 512, 4, 1), None, "not declared");
        // Submission 2 clamps arguments at 64..96 in place.
        shadow.in_place(2, a + 64, a + 96);
        shadow.gpu_wrote(start + 64, &[7]);
        title[65] = 9;
        title[600] = 3;
        shadow.copy_back(a, 256, None, 1);
        assert_eq!(title[16], 5);
        assert_eq!(shadow.bytes(start + 64, 2), Some(&[7, 0][..]), "left alone");
        shadow.update(a, n);
        assert_eq!(
            shadow.bytes(start + 64, 2),
            Some(&[7, 0][..]),
            "not refreshed"
        );
        assert_eq!(shadow.bytes(start + 600, 1), Some(&[3][..]));
        shadow.in_place_done(2);
        shadow.update(a, n);
        assert_eq!(shadow.bytes(start + 64, 2), Some(&title[64..66]));
        unregister(resource, start);
    }

    /// One submission at a time: its copy-back puts the GPU's bytes back
    /// and takes the title's into the copy, as before snapshots were per
    /// submission.
    #[cfg(windows)]
    #[test]
    fn one_submission_copies_back_and_syncs_the_copy() {
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let mut title = vec![0u8; 4096];
        let start = title.as_ptr() as u64;
        let resource = 0xdead_beef_0004_usize;
        register(resource, start, title.len() as u64);
        let shadow = find(start, 64).expect("shadow");
        let (a, n) = (start as usize, title.len());
        shadow.update(a, n);
        shadow.writing(a + 8, 64, 7);
        shadow.gpu_wrote(start + 9, &[0xcc]);
        title[12] = 0x22;
        shadow.copy_back(a + 8, 64, None, 7);
        assert_eq!((title[9], title[12]), (0xcc, 0x22));
        assert_eq!(shadow.bytes(start + 12, 1), Some(&[0x22][..]));
        assert!(shadow.before.lock().unwrap().is_empty());
        assert!(shadow.pending.lock().unwrap().is_empty());
        unregister(resource, start);
    }

    #[cfg(windows)]
    #[test]
    fn unregister_retires_shadow_until_active_epoch_finishes() {
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let list = host.new_list().expect("host list");
        list.mark_work();

        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0002_usize;
        register(resource, start, data.len() as u64);
        let weak = {
            let shadow = find(start, 64).expect("shadow");
            Arc::downgrade(&shadow)
        };
        assert_eq!(weak.strong_count(), 1);

        unregister(resource, start);
        assert_eq!(
            weak.strong_count(),
            1,
            "unregistered shadow must wait for active epochs"
        );

        list.close();
        list.reset();
        assert_eq!(weak.strong_count(), 0);
    }
}
