//! Query heaps. Occlusion heaps get a host query heap of the same size
//! plus a few spare queries, so the title's occlusion queries run on the
//! host GPU (see `list.rs`); other kinds only remember their description.
//!
//! A host query may not span host lists, and a title's list can be cut into
//! several (a memory wait, a hoist). A query open at a cut is ended there
//! and goes on in the next host list as one of the spare queries; resolving
//! it combines the parts ([`combined`]). With no spare query left it is cut
//! short there, and resolves as passed rather than hide what the draws
//! after the cut would have shown.
//!
//! Which parts a resolve combines follows the order the host lists run in,
//! which is the order they are submitted to the one host queue, not the
//! order they are recorded in: each begin of a title query is a run with
//! its own spare queries, and a resolve takes the run submitted last before
//! its list (or one begun earlier in its own list), chosen when its list is
//! submitted ([`Resolve`]). A run's spare queries stay its own while its
//! recording lasts, and after that while it is the last submitted run of
//! its query.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUERY_HEAP, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x0d9658ae,
        0xed45,
        0x469e,
        [0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Desc {
    kind: u32,
    count: u32,
    nodes: u32,
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: Desc,
    /// The host occlusion heap, made on first use.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<HostQueries>>,
}

#[cfg(windows)]
struct HostQueries {
    heap: windows::Win32::Graphics::Direct3D12::ID3D12QueryHeap,
    /// The title's query count: the spare host queries follow.
    count: u32,
    /// How many spare host queries follow.
    spare: u32,
    parts: std::sync::Arc<std::sync::Mutex<Parts>>,
}

/// Host queries each host occlusion heap has past the title's, for the
/// parts of queries that go on after their host list was cut.
#[cfg(windows)]
const SPARE_QUERIES: u32 = 64;

/// One begin of a title query: the spare host queries holding its later
/// parts, in order, and whether it was cut short.
#[cfg(any(windows, test))]
#[derive(Debug)]
struct Run {
    /// The title query.
    index: u32,
    /// The host list its begin was recorded in, while that recording lasts
    /// (0 after).
    list: usize,
    spares: Vec<u32>,
    /// No spare host query was left at a cut, so the draws after it are in
    /// no host query.
    short: bool,
}

/// The runs of a heap's title queries and the spare host queries they
/// hold.
#[cfg(any(windows, test))]
#[derive(Debug, Default)]
struct Parts {
    /// Spare host queries holding no part.
    free: Vec<u32>,
    /// Runs by id, which counts up in recording order.
    runs: std::collections::BTreeMap<u64, Run>,
    /// Title query to its run submitted last.
    latest: std::collections::HashMap<u32, u64>,
    next: u64,
}

/// The parts a resolve combines: (place in the resolved range, spare host
/// query), in order, and the places of queries cut short.
#[cfg(any(windows, test))]
type Chosen = (Vec<(u32, u32)>, Vec<u32>);

#[cfg(any(windows, test))]
impl Parts {
    /// For a heap of `count` title queries followed by `spare` more.
    fn new(count: u32, spare: u32) -> Self {
        Self {
            free: (count..count.saturating_add(spare)).rev().collect(),
            ..Default::default()
        }
    }

    /// Title query `index` begins anew in host list `list`: a new run,
    /// whose result is its own parts alone.
    fn begins(&mut self, index: u32, list: usize) -> u64 {
        self.next += 1;
        self.runs.insert(
            self.next,
            Run {
                index,
                list,
                spares: Vec::new(),
                short: false,
            },
        );
        self.next
    }

    /// A spare host query for the next part of run `run`, if one is left;
    /// with none the run is cut short.
    fn continues(&mut self, run: u64) -> Option<u32> {
        let run = self.runs.get_mut(&run)?;
        let Some(spare) = self.free.pop() else {
            run.short = true;
            return None;
        };
        run.spares.push(spare);
        Some(spare)
    }

    /// For a resolve of title queries `first..first + count` recorded now
    /// in host list `list`: per place, the run of that query begun last in
    /// `list`, which runs before the resolve whatever else is submitted.
    fn fixed(&self, first: u32, count: u32, list: usize) -> Vec<Option<u64>> {
        let mut fixed = vec![None; count as usize];
        for (&id, run) in &self.runs {
            if run.list == list
                && let Some(place) = run.index.checked_sub(first).filter(|&p| p < count)
            {
                fixed[place as usize] = Some(id);
            }
        }
        fixed
    }

    /// The parts of a resolve of title queries from `first` whose list is
    /// being submitted now: per place the run `fixed` names, else the run
    /// of that query submitted last (none: the query's own result alone).
    fn chosen(&self, first: u32, fixed: &[Option<u64>]) -> Chosen {
        let (mut parts, mut short) = (Vec::new(), Vec::new());
        for (place, fixed) in (0u32..).zip(fixed) {
            let Some(run) = fixed
                .or_else(|| self.latest.get(&first.checked_add(place)?).copied())
                .and_then(|id| self.runs.get(&id))
            else {
                continue;
            };
            parts.extend(run.spares.iter().map(|&spare| (place, spare)));
            if run.short {
                short.push(place);
            }
        }
        (parts, short)
    }

    /// Host list `list` is submitted: its runs, in recording order, are each
    /// its query's last submitted, and an earlier one whose recording is
    /// over is let go.
    fn submitted(&mut self, list: usize) {
        let ids: Vec<u64> = self
            .runs
            .iter()
            .filter(|(_, run)| run.list == list)
            .map(|(&id, _)| id)
            .collect();
        for id in ids {
            let index = self.runs[&id].index;
            if let Some(was) = self.latest.insert(index, id)
                && was != id
                && self.runs.get(&was).is_some_and(|run| run.list == 0)
            {
                self.release(was);
            }
        }
    }

    /// The recording of host list `list` is over (reset or dropped): its
    /// runs are let go unless one is still its query's last submitted.
    fn forget(&mut self, list: usize) {
        let ids: Vec<u64> = self
            .runs
            .iter()
            .filter(|(_, run)| run.list == list)
            .map(|(&id, _)| id)
            .collect();
        for id in ids {
            let index = self.runs[&id].index;
            if self.latest.get(&index) == Some(&id) {
                if let Some(run) = self.runs.get_mut(&id) {
                    run.list = 0;
                }
            } else {
                self.release(id);
            }
        }
    }

    /// Run `id` is gone: its spare host queries are free again.
    fn release(&mut self, id: u64) {
        if let Some(run) = self.runs.remove(&id) {
            self.free.extend(run.spares.into_iter().rev());
        }
    }

    /// The spare host queries of each title query of `first..first + count`
    /// as its run recorded last has them, as (place, spare host query).
    #[cfg(test)]
    fn recorded(&self, first: u32, count: u32) -> Vec<(u32, u32)> {
        let mut last = std::collections::BTreeMap::new();
        for run in self.runs.values() {
            if let Some(place) = run.index.checked_sub(first).filter(|&p| p < count) {
                last.insert(place, &run.spares);
            }
        }
        last.into_iter()
            .flat_map(|(place, spares)| spares.iter().map(move |&spare| (place, spare)))
            .collect()
    }
}

/// `D3D12_QUERY_TYPE_BINARY_OCCLUSION`.
#[cfg(any(windows, test))]
const BINARY_OCCLUSION: u32 = 1;

/// The `results` of resolved occlusion queries of kind `kind` (8 bytes
/// each) with the results of their later parts taken in: `extra` holds one
/// 8-byte result per entry of `places`, which names the result it belongs
/// to. Sample counts add up; a binary query passed if any part did. The
/// results `short` names are of queries cut short, whose later draws no
/// part counted: one that counted nothing says one sample passed, so the
/// title does not hide what those draws may have shown.
#[cfg(any(windows, test))]
pub(crate) fn combined(
    kind: u32,
    results: &[u8],
    extra: &[u8],
    places: &[u32],
    short: &[u32],
) -> Vec<u8> {
    let mut out = results.to_vec();
    for (part, &place) in extra.as_chunks::<8>().0.iter().zip(places) {
        let Some(whole) = out.as_chunks_mut::<8>().0.get_mut(place as usize) else {
            continue;
        };
        let (so_far, part) = (u64::from_le_bytes(*whole), u64::from_le_bytes(*part));
        let both = if kind == BINARY_OCCLUSION {
            u64::from(so_far != 0 || part != 0)
        } else {
            so_far.saturating_add(part)
        };
        *whole = both.to_le_bytes();
    }
    for &place in short {
        if let Some(whole) = out.as_chunks_mut::<8>().0.get_mut(place as usize)
            && *whole == [0; 8]
        {
            *whole = 1u64.to_le_bytes();
        }
    }
    out
}

/// `D3D12_QUERY_HEAP_TYPE_OCCLUSION`.
const OCCLUSION_HEAP: u32 = 0;

#[cfg(windows)]
fn heap(this: This) -> Option<&'static Heap> {
    // SAFETY: the title passes one of our query heaps, alive while it uses it.
    unsafe { this.cast::<Heap>().as_ref() }
}

#[cfg(windows)]
fn host_queries(this: This) -> Option<&'static HostQueries> {
    let heap = heap(this)?;
    if heap.desc.kind != OCCLUSION_HEAP || heap.desc.count == 0 {
        return None;
    }
    heap.host
        .get_or_init(|| {
            use windows::Win32::Graphics::Direct3D12::*;
            let device = &crate::host::host()?.device;
            let spare = SPARE_QUERIES.min(u32::MAX - heap.desc.count);
            let total = heap.desc.count + spare;
            let mut made: Option<ID3D12QueryHeap> = None;
            // SAFETY: plain object creation.
            let result = unsafe {
                device.CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_OCCLUSION,
                        Count: total,
                        NodeMask: 0,
                    },
                    &mut made,
                )
            };
            if let Err(e) = result {
                tracing::warn!(error = %e, count = heap.desc.count, "no host occlusion query heap");
                return None;
            }
            let made = made?;
            if let Err(e) = end_every_query(&made, total) {
                tracing::warn!(error = %e, count = heap.desc.count, "host occlusion queries not set up");
                return None;
            }
            Some(HostQueries {
                heap: made,
                count: heap.desc.count,
                spare,
                parts: std::sync::Arc::new(std::sync::Mutex::new(Parts::new(
                    heap.desc.count,
                    spare,
                ))),
            })
        })
        .as_ref()
}

/// Begins and ends each of the `count` queries of a new host occlusion heap
/// in a list of its own, run before any list that names the heap: every
/// query then has a result (0) before the title's first one ends, so a
/// resolve recorded or run before that is valid and needs no record of
/// which queries ended, which recording order cannot tell (lists run in the
/// order they are submitted, not recorded).
#[cfg(windows)]
fn end_every_query(
    heap: &windows::Win32::Graphics::Direct3D12::ID3D12QueryHeap,
    count: u32,
) -> windows::core::Result<()> {
    use windows::Win32::Graphics::Direct3D12::D3D12_QUERY_TYPE_OCCLUSION;
    let host = crate::host::host().ok_or_else(windows::core::Error::empty)?;
    let list = host.new_list()?;
    for index in 0..count {
        // SAFETY: an open list; the index is inside the heap.
        unsafe {
            list.list
                .BeginQuery(heap, D3D12_QUERY_TYPE_OCCLUSION, index);
            list.list.EndQuery(heap, D3D12_QUERY_TYPE_OCCLUSION, index);
        }
    }
    list.close();
    host.execute(&[&list]);
    Ok(())
}

/// The host occlusion heap behind the title's query heap `this`, if it is an
/// occlusion heap and the host GPU made one.
#[cfg(windows)]
pub(crate) fn occlusion_heap(
    this: This,
) -> Option<windows::Win32::Graphics::Direct3D12::ID3D12QueryHeap> {
    host_queries(this).map(|q| q.heap.clone())
}

/// What each host list noted of the occlusion queries recorded in it: the
/// heaps it began queries of and the resolves it recorded, until its
/// recording is over ([`forget_list`]).
#[cfg(windows)]
#[derive(Default)]
struct Noted {
    heaps: Vec<std::sync::Arc<std::sync::Mutex<Parts>>>,
    resolves: Vec<std::sync::Arc<Resolve>>,
}

#[cfg(windows)]
static NOTED: std::sync::Mutex<Option<std::collections::HashMap<usize, Noted>>> =
    std::sync::Mutex::new(None);

#[cfg(windows)]
fn noted<T>(list: usize, f: impl FnOnce(&mut Noted) -> T) -> T {
    let mut noted = NOTED.lock().unwrap_or_else(|e| e.into_inner());
    f(noted
        .get_or_insert_with(Default::default)
        .entry(list)
        .or_default())
}

/// Title query `index` of `this` begins anew in host list `list`: a new
/// run, whose id the query's notes carry ([`begun`]).
#[cfg(windows)]
pub(crate) fn begins(this: This, index: u32, list: usize) -> u64 {
    let Some(queries) = host_queries(this) else {
        return 0;
    };
    let run = queries
        .parts
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .begins(index, list);
    noted(list, |noted| {
        if !noted
            .heaps
            .iter()
            .any(|h| std::sync::Arc::ptr_eq(h, &queries.parts))
        {
            noted.heaps.push(queries.parts.clone());
        }
    });
    run
}

/// A spare host query for run `run` of a title query of `this` to go on in
/// after its host list was cut, if one is left; with none the run is noted
/// as cut short.
#[cfg(windows)]
pub(crate) fn continues(this: This, run: u64) -> Option<u32> {
    let mut parts = host_queries(this)?
        .parts
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    parts.continues(run)
}

/// A resolve of the title queries `first..first + count` of `this`, of kind
/// `kind`, recorded in host list `list`: which later parts it combines is
/// chosen when the list is submitted ([`submitting`]). None if `this` has
/// no host occlusion heap.
#[cfg(windows)]
pub(crate) fn resolving(
    this: This,
    kind: u32,
    first: u32,
    count: u32,
    list: usize,
) -> Option<std::sync::Arc<Resolve>> {
    let queries = host_queries(this)?;
    let fixed = queries
        .parts
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .fixed(first, count, list);
    let resolve = std::sync::Arc::new(Resolve {
        parts: queries.parts.clone(),
        kind,
        first,
        count,
        spare_first: queries.count,
        spare: queries.spare,
        fixed,
        chosen: Default::default(),
    });
    noted(list, |noted| noted.resolves.push(resolve.clone()));
    Some(resolve)
}

/// Host list `list` is being submitted (in the order lists reach the host
/// queue): its resolves choose their parts from the runs submitted before
/// it, then its own runs are the last submitted of their queries.
#[cfg(windows)]
pub(crate) fn submitting(list: usize) {
    let (heaps, resolves) = {
        let noted = NOTED.lock().unwrap_or_else(|e| e.into_inner());
        let Some(noted) = noted.as_ref().and_then(|n| n.get(&list)) else {
            return;
        };
        (noted.heaps.clone(), noted.resolves.clone())
    };
    for resolve in &resolves {
        resolve.choose();
    }
    for heap in heaps {
        heap.lock()
            .unwrap_or_else(|e| e.into_inner())
            .submitted(list);
    }
}

/// The recording of host list `list` is over (it is reset or dropped): its
/// runs no later list resolves are let go.
#[cfg(windows)]
pub(crate) fn forget_list(list: usize) {
    let noted = NOTED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        .and_then(|n| n.remove(&list));
    for heap in noted.into_iter().flat_map(|n| n.heaps) {
        heap.lock().unwrap_or_else(|e| e.into_inner()).forget(list);
    }
}

/// The spare host queries of the title queries `first..first + count` of
/// `this`, as their runs recorded last have them: (place, spare host
/// query).
#[cfg(all(test, windows))]
pub(crate) fn recorded_parts(this: This, first: u32, count: u32) -> Vec<(u32, u32)> {
    host_queries(this).map_or_else(Vec::new, |queries| {
        let parts = queries.parts.lock().unwrap_or_else(|e| e.into_inner());
        parts.recorded(first, count)
    })
}

/// One resolve of occlusion queries recorded in a host list: it resolves
/// the title's queries and every spare host query, and once the list has
/// run combines with each result the parts of the run chosen for it when
/// the list was submitted.
// shortcut: a list submitted again before its last submission's results
// were copied back copies them with the parts chosen for the new one.
#[cfg(windows)]
pub(crate) struct Resolve {
    parts: std::sync::Arc<std::sync::Mutex<Parts>>,
    kind: u32,
    first: u32,
    count: u32,
    /// The first spare host query and how many there are.
    spare_first: u32,
    spare: u32,
    /// Per place, a run begun earlier in the resolve's own host list.
    fixed: Vec<Option<u64>>,
    chosen: std::sync::Mutex<Chosen>,
}

#[cfg(windows)]
impl Resolve {
    /// The spare host queries resolved behind the title's, as (first,
    /// count): one 8-byte result each.
    pub(crate) fn spares(&self) -> (u32, u32) {
        (self.spare_first, self.spare)
    }

    fn choose(&self) {
        let chosen = self
            .parts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .chosen(self.first, &self.fixed);
        *self.chosen.lock().unwrap_or_else(|e| e.into_inner()) = chosen;
    }

    /// The results as the title reads them, from `resolved`: the title's
    /// `count` results followed by one for each spare host query.
    pub(crate) fn combined(&self, resolved: &[u8]) -> Vec<u8> {
        let split = (self.count as usize * 8).min(resolved.len());
        let (results, spares) = resolved.split_at(split);
        let chosen = self.chosen.lock().unwrap_or_else(|e| e.into_inner());
        picked(self.kind, results, spares, self.spare_first, &chosen)
    }
}

/// [`combined`] with the parts `chosen` names, whose results are in
/// `spares`, one 8-byte result per spare host query from `spare_first`.
#[cfg(any(windows, test))]
fn picked(kind: u32, results: &[u8], spares: &[u8], spare_first: u32, chosen: &Chosen) -> Vec<u8> {
    let (parts, short) = chosen;
    let (mut extra, mut places) = (Vec::new(), Vec::new());
    for &(place, spare) in parts {
        let Some(at) = spare.checked_sub(spare_first) else {
            continue;
        };
        if let Some(result) = spares.as_chunks::<8>().0.get(at as usize) {
            extra.extend_from_slice(result);
            places.push(place);
        }
    }
    combined(kind, results, &extra, &places, short)
}

/// Queries begun on the host and not yet ended, by (host list, heap,
/// index): the query's kind, the host query it runs as (its own index, or a
/// spare one after a cut) and its run ([`begins`]). Each recording list pairs its own
/// `BeginQuery` and `EndQuery`, so lists recording the same query at once
/// do not end each other's. An entry lasts no longer than the recording it
/// was made in: a split or the title's close ends the query, and a reset or
/// the release of the list forgets it ([`take_open`]), so a later recording
/// in a host list at the same address finds none of them.
#[cfg(windows)]
static OPEN: std::sync::Mutex<Option<OpenQueries>> = std::sync::Mutex::new(None);

/// (host list, heap, index) to the query's kind, its host query and its
/// run.
#[cfg(windows)]
type OpenQueries = std::collections::HashMap<(usize, usize, u32), (u32, u32, u64)>;

/// A query begun on the host and not yet ended: its heap, kind, index, the
/// host query it runs as, and its run.
#[cfg(windows)]
pub(crate) type Open = (This, u32, u32, u32, u64);

/// Notes query `index` of `this`, of kind `kind`, begun in host list `list`
/// as host query `host_index`, for run `run`.
#[cfg(windows)]
pub(crate) fn begun(this: This, kind: u32, index: u32, host_index: u32, list: usize, run: u64) {
    OPEN.lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(Default::default)
        .insert((list, this as usize, index), (kind, host_index, run));
}

/// The queries begun in host list `list` and not ended there, forgetting
/// them: the list is about to be closed or dropped, and a query may not
/// stay open past the end of its list.
#[cfg(windows)]
pub(crate) fn take_open(list: usize) -> Vec<Open> {
    let mut open = OPEN.lock().unwrap_or_else(|e| e.into_inner());
    let Some(open) = open.as_mut() else {
        return Vec::new();
    };
    let mut taken: Vec<(usize, u32, u32, u32, u64)> = open
        .extract_if(|key, _| key.0 == list)
        .map(|((_, heap, index), (kind, host_index, run))| (heap, kind, index, host_index, run))
        .collect();
    taken.sort_unstable();
    taken
        .into_iter()
        .map(|(heap, kind, index, host_index, run)| (heap as This, kind, index, host_index, run))
        .collect()
}

/// Forgets every query of heap `this` still open in any host list: the
/// title released the heap, and the notes of its open queries name it by
/// address, which a split or a close of their list would otherwise look up
/// after it is gone.
// shortcut: such a query is left begun in its host list and not ended (the
// host heap itself lives on with the list); no title of ours releases a
// query heap with a query of it open.
#[cfg(windows)]
fn forget_heap(this: This) {
    if let Some(open) = OPEN.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        open.retain(|key, _| key.1 != this as usize);
    }
}

/// The host query that query `index` of `this` was begun as in host list
/// `list`, if it was and has not ended there since, forgetting it.
#[cfg(windows)]
pub(crate) fn take_begun(this: This, index: u32, list: usize) -> Option<u32> {
    let (_, host_index, _) = OPEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()?
        .remove(&(list, this as usize, index))?;
    Some(host_index)
}

/// Device method.
pub(crate) extern "system" fn create(
    _: This,
    desc: *const Desc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "query heap");
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
    });
    object::hand_out(QUERY_HEAP, heap, iid, &HEAP_IIDS, out)
}

/// An occlusion heap of `count` queries, as the title would make it.
#[cfg(all(test, windows))]
pub(crate) fn test_occlusion_heap(count: u32) -> This {
    let desc = Desc {
        kind: OCCLUSION_HEAP,
        count,
        nodes: 0,
    };
    let mut heap = std::ptr::null();
    let made = create(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut heap);
    assert_eq!(made, crate::object::S_OK);
    heap
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUERY_HEAP,
        iid,
        object::is(iid, &HEAP_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    let left = object::release_boxed::<Heap>(this);
    #[cfg(windows)]
    if left == 0 {
        forget_heap(this);
    }
    left
}

/// IUnknown 3, object 4, device child 1.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<QUERY_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{E_NOINTERFACE, E_POINTER, S_FALSE, S_OK, slot};

    #[test]
    fn null_descriptions_are_rejected() {
        let mut heap = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), std::ptr::null(), &HEAP_IIDS[0], &mut heap),
            E_INVALIDARG
        );
        assert!(heap.is_null());
        // The description is checked first: a probe with no room for the
        // heap fails on it rather than answering S_FALSE.
        assert_eq!(
            create(
                std::ptr::null(),
                std::ptr::null(),
                &HEAP_IIDS[0],
                std::ptr::null_mut()
            ),
            E_INVALIDARG
        );
    }

    #[test]
    fn creation_without_an_out_pointer_only_probes() {
        let desc = Desc {
            kind: 1, // timestamps
            count: 64,
            nodes: 0,
        };
        // A known interface with no out pointer asks whether creation would
        // succeed; the heap is dropped instead of handed out.
        assert_eq!(
            create(std::ptr::null(), &desc, &HEAP_IIDS[0], std::ptr::null_mut()),
            S_FALSE
        );
        // A null interface id is tolerated: no interface, no heap.
        let mut heap = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), &desc, std::ptr::null(), &mut heap),
            E_NOINTERFACE
        );
        assert!(heap.is_null());
    }

    #[test]
    fn heaps_answer_their_interfaces_and_count_references() {
        let desc = Desc {
            kind: 0, // occlusion
            count: 8,
            nodes: 0,
        };
        let null = std::ptr::null();
        let mut heap = null;
        assert_eq!(create(null, &desc, &HEAP_IIDS[0], &mut heap), S_OK);
        assert!(!heap.is_null());
        let query = slot::<extern "system" fn(This, *const Iid, *mut This) -> HRESULT>(heap, 0x0);
        // Every interface the heap answers to hands out the same object.
        for iid in &HEAP_IIDS {
            let mut out = null;
            assert_eq!(query(heap, iid, &mut out), S_OK);
            assert_eq!(out as usize, heap as usize);
        }
        // An unknown interface writes null behind and reports it missing.
        let unknown = [0u8; 16];
        let mut out = heap;
        assert_eq!(query(heap, &unknown, &mut out), E_NOINTERFACE);
        assert!(out.is_null());
        // A null interface id is tolerated rather than matched.
        let mut out = heap;
        assert_eq!(query(heap, std::ptr::null(), &mut out), E_NOINTERFACE);
        assert!(out.is_null());
        // No room for the answer is a pointer error.
        assert_eq!(query(heap, &unknown, std::ptr::null_mut()), E_POINTER);
        // References: one from creation, four from the queries, one added;
        // each release hands back what is left, the last frees the heap.
        assert_eq!(slot::<extern "system" fn(This) -> u32>(heap, 0x8)(heap), 6);
        let release = slot::<extern "system" fn(This) -> u32>(heap, 0x10);
        for left in [5, 4, 3, 2, 1, 0] {
            assert_eq!(release(heap), left);
        }
    }

    /// A heap the title releases takes the notes of its open queries with
    /// it, in every list, and leaves another heap's alone.
    #[cfg(windows)]
    #[test]
    fn a_released_heap_leaves_no_open_query() {
        let desc = Desc {
            kind: OCCLUSION_HEAP,
            count: 4,
            nodes: 0,
        };
        let mut gone = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut gone),
            S_OK
        );
        // Another heap's address, never dereferenced.
        let other = 0x5157_2000_usize as This;
        let (a, b) = (0xa200_usize, 0xb200_usize);
        begun(gone, 0, 1, 1, a, 1);
        begun(gone, 1, 2, 2, b, 2);
        begun(other, 0, 1, 1, a, 3);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(gone, 0x10)(gone), 0);
        assert_eq!(take_open(b), []);
        assert_eq!(take_open(a), [(other, 0, 1, 1, 3)]);
    }

    /// Two lists recording the same query interleaved (A begins, B begins,
    /// A ends, B ends) each end their own.
    #[cfg(windows)]
    #[test]
    fn each_list_pairs_its_own_begin_and_end() {
        // A heap address no other test uses; never dereferenced.
        let heap = 0x5157_0000_usize as This;
        let (a, b) = (0xa000_usize, 0xb000_usize);
        begun(heap, 0, 3, 3, a, 1);
        begun(heap, 0, 3, 3, b, 2);
        assert_eq!(take_begun(heap, 3, a), Some(3));
        assert_eq!(take_begun(heap, 3, b), Some(3));
        // Each end is taken once, and an end without a begin is not.
        assert_eq!(take_begun(heap, 3, a), None);
        assert_eq!(take_begun(heap, 3, b), None);
        assert_eq!(take_begun(heap, 4, a), None);
        // Another index of the same list is its own query.
        begun(heap, 0, 5, 5, a, 3);
        assert_eq!(take_begun(heap, 6, a), None);
        assert_eq!(take_begun(heap, 5, a), Some(5));
        // A query going on after a cut ends as the host query it runs as.
        begun(heap, 0, 5, 70, a, 3);
        assert_eq!(take_begun(heap, 5, a), Some(70));
    }

    /// A list about to be closed gives up the queries still open in it,
    /// with their kinds, and only its own.
    #[cfg(windows)]
    #[test]
    fn a_closing_list_gives_up_its_open_queries() {
        let heap = 0x5157_1000_usize as This;
        let (a, b) = (0xa100_usize, 0xb100_usize);
        begun(heap, 0, 7, 7, a, 1);
        begun(heap, 1, 2, 66, a, 2);
        begun(heap, 0, 7, 7, b, 3);
        assert_eq!(take_begun(heap, 7, a), Some(7));
        begun(heap, 0, 9, 9, a, 4);
        assert_eq!(take_open(a), [(heap, 0, 9, 9, 4), (heap, 1, 2, 66, 2)]);
        assert!(take_open(a).is_empty());
        // An end the title records later in this list finds nothing to end.
        assert_eq!(take_begun(heap, 9, a), None);
        assert_eq!(take_open(b), [(heap, 0, 7, 7, 3)]);
    }

    /// Sample counts of a query's parts add up; a binary query passed if
    /// any part did; results without later parts stay as resolved.
    #[test]
    fn the_parts_of_a_query_combine_by_its_kind() {
        let bytes =
            |values: &[u64]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
        // Query 0 has no later part, query 1 has two, query 2 has one.
        let (extra, places) = (bytes(&[5, 7, 0]), [1, 1, 2]);
        assert_eq!(
            combined(0, &bytes(&[3, 10, 0]), &extra, &places, &[]),
            bytes(&[3, 22, 0])
        );
        assert_eq!(
            combined(BINARY_OCCLUSION, &bytes(&[1, 0, 0]), &extra, &places, &[]),
            bytes(&[1, 1, 0])
        );
        let late = bytes(&[0, 0, 1]);
        assert_eq!(
            combined(BINARY_OCCLUSION, &bytes(&[0, 1, 0]), &late, &places, &[]),
            bytes(&[0, 1, 1])
        );
        // A count does not wrap, and a part outside the results is left out.
        assert_eq!(
            combined(0, &bytes(&[u64::MAX]), &bytes(&[2, 9]), &[0, 4], &[]),
            bytes(&[u64::MAX])
        );
        assert_eq!(combined(0, &bytes(&[4, 6]), &[], &[], &[]), bytes(&[4, 6]));
    }

    /// A query cut short with nothing counted before the cut resolves as
    /// passed, of either kind: the draws after the cut are in no host
    /// query. One that counted samples keeps its count, and queries not cut
    /// short keep a zero.
    #[test]
    fn a_query_cut_short_does_not_resolve_as_hidden() {
        let bytes =
            |values: &[u64]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
        for kind in [0, BINARY_OCCLUSION] {
            assert_eq!(
                combined(kind, &bytes(&[0, 0, 1, 0]), &[], &[], &[1, 2, 9]),
                bytes(&[0, 1, 1, 0])
            );
        }
        assert_eq!(
            combined(0, &bytes(&[0, 40]), &bytes(&[0]), &[0], &[0, 1]),
            bytes(&[1, 40])
        );

        // The run that finds no spare host query at a cut is the one cut
        // short; a new run of its query, once submitted, is not.
        let (a, b) = (0xa0, 0xb0);
        let mut parts = Parts::new(8, 1);
        let three = parts.begins(3, a);
        let six = parts.begins(6, a);
        let two = parts.begins(2, a);
        assert_eq!(parts.continues(three), Some(8));
        assert_eq!(parts.continues(six), None);
        assert_eq!(parts.continues(two), None);
        parts.submitted(a);
        let all = [None; 8];
        assert_eq!(parts.chosen(0, &all), (vec![(3, 8)], vec![2, 6]));
        assert_eq!(
            parts.chosen(5, &all[..3]).1,
            [1],
            "places from the first resolved"
        );
        assert!(parts.chosen(3, &all[..1]).1.is_empty());
        parts.begins(6, b);
        parts.submitted(b);
        assert_eq!(parts.chosen(0, &all).1, [2]);
    }

    /// Each cut of a run's host list takes one spare host query, which
    /// stays the run's until a later run of its query is submitted and its
    /// own recording is over; with none left the run does not go on.
    #[test]
    fn spare_queries_hold_the_parts_until_a_later_run_supersedes_them() {
        let (a, b) = (0xa0, 0xb0);
        let mut parts = Parts::new(8, 3);
        let five = parts.begins(5, a);
        let two = parts.begins(2, a);
        assert_eq!(parts.continues(five), Some(8));
        assert_eq!(parts.continues(five), Some(9));
        assert_eq!(parts.continues(two), Some(10));
        let six = parts.begins(6, a);
        assert_eq!(parts.continues(six), None, "no spare query is left");
        parts.submitted(a);
        let all = [None; 8];
        assert_eq!(parts.chosen(0, &all).0, [(2, 10), (5, 8), (5, 9)]);
        // Places are counted from the first query resolved.
        assert_eq!(parts.chosen(4, &all[..2]).0, [(1, 8), (1, 9)]);
        assert_eq!(parts.chosen(3, &all[..2]).0, []);
        // Query 5 begins again in `b`: while `a`'s recording lasts its
        // spare queries stay taken, even once `b` is submitted.
        parts.begins(5, b);
        parts.submitted(b);
        assert_eq!(parts.chosen(0, &all).0, [(2, 10)]);
        let seven = parts.begins(7, b);
        assert_eq!(parts.continues(seven), None);
        // `a` is reset: query 5's old run is let go, lowest spare first,
        // and query 2's stays, being its last submitted.
        parts.forget(a);
        assert_eq!(parts.chosen(0, &all).0, [(2, 10)]);
        let six = parts.begins(6, b);
        assert_eq!(parts.continues(six), Some(8));
    }

    /// The case resolving in recording order got wrong: list A runs query 0
    /// over a cut, list B begins it again without a cut but is never
    /// submitted, and a resolve in a third list after A is submitted takes
    /// A's spare query. Once B is submitted, a later resolve takes B's run.
    #[test]
    fn a_resolve_takes_the_run_submitted_last_not_recorded_last() {
        let (a, b, c, d) = (0xa0, 0xb0, 0xc0, 0xd0);
        let mut parts = Parts::new(4, 2);
        let in_a = parts.begins(0, a);
        assert_eq!(parts.continues(in_a), Some(4));
        parts.begins(0, b);
        let in_c = parts.fixed(0, 1, c);
        assert_eq!(in_c, [None], "nothing of query 0 was begun in c");
        parts.submitted(a);
        parts.forget(a);
        assert_eq!(parts.chosen(0, &in_c), (vec![(0, 4)], vec![]));
        // B never runs and is dropped: A's run stays query 0's.
        parts.forget(b);
        assert_eq!(parts.chosen(0, &in_c), (vec![(0, 4)], vec![]));
        // A new run, submitted: its result is its own alone, and A's spare
        // query is free again.
        let in_d = parts.begins(0, d);
        parts.submitted(d);
        assert_eq!(parts.chosen(0, &in_c), (vec![], vec![]));
        assert_eq!(parts.continues(in_d), Some(4));
    }

    /// A resolve recorded before a later begin of its query, in a list that
    /// runs after the begin's, takes that begin's run: the order the host
    /// queue runs them in, not the order they were recorded in.
    #[test]
    fn a_resolve_recorded_first_and_run_later_takes_the_later_run() {
        let (r, m) = (0x10, 0x20);
        let mut parts = Parts::new(2, 2);
        let fixed = parts.fixed(1, 1, r);
        let in_m = parts.begins(1, m);
        assert_eq!(parts.continues(in_m), Some(2));
        parts.submitted(m);
        assert_eq!(parts.chosen(1, &fixed).0, [(0, 2)]);
        // Submitted in the other order (the resolve's list before `m`), the
        // resolve finds no run yet, though `m`'s begin was recorded.
        let mut parts = Parts::new(2, 2);
        let fixed = parts.fixed(1, 1, r);
        let in_m = parts.begins(1, m);
        assert_eq!(parts.continues(in_m), Some(2));
        assert_eq!(parts.chosen(1, &fixed), (vec![], vec![]));
    }

    /// A resolve after a begin in its own host list takes that run, whatever
    /// another list submitted before it ran; one before the begin in that
    /// list does not.
    #[test]
    fn a_resolve_takes_the_run_begun_earlier_in_its_own_list() {
        let (l, other) = (0x30, 0x40);
        let mut parts = Parts::new(2, 2);
        let before = parts.fixed(0, 2, l);
        let mine = parts.begins(0, l);
        assert_eq!(parts.continues(mine), Some(2));
        let after = parts.fixed(0, 2, l);
        assert_eq!(after, [Some(mine), None]);
        let theirs = parts.begins(0, other);
        assert_eq!(parts.continues(theirs), Some(3));
        parts.submitted(other);
        assert_eq!(parts.chosen(0, &after).0, [(0, 2)]);
        assert_eq!(parts.chosen(0, &before).0, [(0, 3)]);
        // `l` is submitted: its run is the last submitted, and `other`'s
        // goes once its recording is over.
        parts.submitted(l);
        parts.forget(other);
        assert_eq!(parts.chosen(0, &[None]).0, [(0, 2)]);
        assert_eq!(parts.free, [3]);
    }

    /// Resolved spare results are picked by spare query: each chosen part's
    /// result is taken in, one outside what was resolved is left out.
    #[test]
    fn chosen_parts_are_read_from_the_resolved_spares() {
        let bytes =
            |values: &[u64]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let spares = bytes(&[100, 20, 3]);
        let chosen = (vec![(0, 9), (0, 11), (1, 10), (1, 40)], vec![2]);
        assert_eq!(
            picked(0, &bytes(&[1, 2, 0]), &spares, 9, &chosen),
            bytes(&[104, 22, 1])
        );
        assert_eq!(
            picked(0, &bytes(&[1, 2]), &spares, 9, &(vec![(0, 3)], vec![])),
            bytes(&[1, 2]),
            "a part below the spare queries is left out"
        );
    }

    /// A new host occlusion heap has a result for every query before the
    /// title ends one: resolving a query no list ended gives 0.
    #[cfg(windows)]
    #[test]
    fn host_occlusion_queries_resolve_before_any_ends() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let desc = Desc {
            kind: OCCLUSION_HEAP,
            count: 4,
            nodes: 0,
        };
        let mut title = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut title),
            S_OK
        );
        let Some(heap) = occlusion_heap(title) else {
            let _ = release(title);
            return; // the host made no occlusion heap
        };
        let list = host.new_list().expect("host command list");
        let (staging, data) = list.readback(host, 32).expect("readback memory");
        // SAFETY: an open list, a live heap and a readback buffer of 32 bytes.
        unsafe {
            list.list.ResolveQueryData(
                &heap,
                windows::Win32::Graphics::Direct3D12::D3D12_QUERY_TYPE_OCCLUSION,
                0,
                4,
                &staging,
                0,
            );
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: the list ran; the readback buffer is mapped.
        let results = unsafe { data.cast::<[u64; 4]>().read_unaligned() };
        assert_eq!(results, [0; 4]);
        assert_eq!(release(title), 0);
    }
}
