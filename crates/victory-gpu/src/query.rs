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
    parts: std::sync::Mutex<Parts>,
}

/// Host queries each host occlusion heap has past the title's, for the
/// parts of queries that go on after their host list was cut.
#[cfg(windows)]
const SPARE_QUERIES: u32 = 64;

/// Which spare host queries hold the later parts of which title queries.
#[cfg(any(windows, test))]
#[derive(Debug, Default)]
struct Parts {
    /// Spare host queries holding no part.
    free: Vec<u32>,
    /// Title query to the spare host queries its result continues in.
    of: std::collections::HashMap<u32, Vec<u32>>,
    /// Title queries cut short: no spare host query was left at a cut, so
    /// the draws after it are in no host query.
    short: std::collections::HashSet<u32>,
}

#[cfg(any(windows, test))]
impl Parts {
    /// For a heap of `count` title queries followed by `spare` more.
    fn new(count: u32, spare: u32) -> Self {
        Self {
            free: (count..count.saturating_add(spare)).rev().collect(),
            of: Default::default(),
            short: Default::default(),
        }
    }

    /// Title query `index` begins anew: its result is the new query's
    /// alone, and the spare queries of its last run are free again.
    fn starts_over(&mut self, index: u32) {
        if let Some(parts) = self.of.remove(&index) {
            self.free.extend(parts.into_iter().rev());
        }
        self.short.remove(&index);
    }

    /// A spare host query for the next part of title query `index`, if one
    /// is left; with none the query is cut short.
    fn continues(&mut self, index: u32) -> Option<u32> {
        let Some(spare) = self.free.pop() else {
            self.short.insert(index);
            return None;
        };
        self.of.entry(index).or_default().push(spare);
        Some(spare)
    }

    /// The title queries of `first..first + count` that were cut short, as
    /// places in that range, in order.
    fn cut_short(&self, first: u32, count: u32) -> Vec<u32> {
        let mut places: Vec<u32> = self
            .short
            .iter()
            .filter_map(|index| index.checked_sub(first).filter(|&place| place < count))
            .collect();
        places.sort_unstable();
        places
    }

    /// The parts of the title queries `first..first + count`, as (place in
    /// that range, spare host query), in order.
    fn within(&self, first: u32, count: u32) -> Vec<(u32, u32)> {
        let mut parts: Vec<(u32, u32)> = self
            .of
            .iter()
            .filter_map(|(&index, spares)| {
                let place = index.checked_sub(first).filter(|&place| place < count)?;
                Some(spares.iter().map(move |&spare| (place, spare)))
            })
            .flatten()
            .collect();
        parts.sort_unstable();
        parts
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
                parts: std::sync::Mutex::new(Parts::new(heap.desc.count, spare)),
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

/// Title query `index` of `this` begins anew: the parts of its last run
/// ([`continues`]) are no longer its result.
// shortcut: by recording order, as [`parts`] reads them, not by the order
// the lists run in: a resolve recorded before a later begin of the same
// query and run after it, two lists recording the same query at once, or a
// begin recorded in a list that runs later or never (the earlier run's
// spare queries are freed, so a resolve recorded after it reads that run
// without its later parts) can combine the wrong parts.
#[cfg(windows)]
pub(crate) fn starts_over(this: This, index: u32) {
    if let Some(queries) = host_queries(this) {
        let mut parts = queries.parts.lock().unwrap_or_else(|e| e.into_inner());
        parts.starts_over(index);
    }
}

/// A spare host query for title query `index` of `this` to go on in after
/// its host list was cut, if one is left; with none the query is noted as
/// cut short ([`parts`]).
#[cfg(windows)]
pub(crate) fn continues(this: This, index: u32) -> Option<u32> {
    let mut parts = host_queries(this)?
        .parts
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    parts.continues(index)
}

/// The later parts of the title queries `first..first + count` of `this`,
/// as (place in that range, host query), in order, and the places of the
/// queries among them that were cut short.
#[cfg(windows)]
pub(crate) fn parts(this: This, first: u32, count: u32) -> (Vec<(u32, u32)>, Vec<u32>) {
    host_queries(this).map_or_else(Default::default, |queries| {
        let parts = queries.parts.lock().unwrap_or_else(|e| e.into_inner());
        (parts.within(first, count), parts.cut_short(first, count))
    })
}

/// Queries begun on the host and not yet ended, by (host list, heap,
/// index): the query's kind and the host query it runs as (its own index,
/// or a spare one after a cut). Each recording list pairs its own
/// `BeginQuery` and `EndQuery`, so lists recording the same query at once
/// do not end each other's. An entry lasts no longer than the recording it
/// was made in: a split or the title's close ends the query, and a reset or
/// the release of the list forgets it ([`take_open`]), so a later recording
/// in a host list at the same address finds none of them.
#[cfg(windows)]
static OPEN: std::sync::Mutex<Option<OpenQueries>> = std::sync::Mutex::new(None);

/// (host list, heap, index) to the query's kind and its host query.
#[cfg(windows)]
type OpenQueries = std::collections::HashMap<(usize, usize, u32), (u32, u32)>;

/// Notes query `index` of `this`, of kind `kind`, begun in host list `list`
/// as host query `host_index`.
#[cfg(windows)]
pub(crate) fn begun(this: This, kind: u32, index: u32, host_index: u32, list: usize) {
    OPEN.lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(Default::default)
        .insert((list, this as usize, index), (kind, host_index));
}

/// The queries begun in host list `list` and not ended there, as (heap,
/// kind, index, host query), forgetting them: the list is about to be
/// closed or dropped, and a query may not stay open past the end of its
/// list.
#[cfg(windows)]
pub(crate) fn take_open(list: usize) -> Vec<(This, u32, u32, u32)> {
    let mut open = OPEN.lock().unwrap_or_else(|e| e.into_inner());
    let Some(open) = open.as_mut() else {
        return Vec::new();
    };
    let mut taken: Vec<(usize, u32, u32, u32)> = open
        .extract_if(|key, _| key.0 == list)
        .map(|((_, heap, index), (kind, host_index))| (heap, kind, index, host_index))
        .collect();
    taken.sort_unstable();
    taken
        .into_iter()
        .map(|(heap, kind, index, host_index)| (heap as This, kind, index, host_index))
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
    let (_, host_index) = OPEN
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
        begun(gone, 0, 1, 1, a);
        begun(gone, 1, 2, 2, b);
        begun(other, 0, 1, 1, a);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(gone, 0x10)(gone), 0);
        assert_eq!(take_open(b), []);
        assert_eq!(take_open(a), [(other, 0, 1, 1)]);
    }

    /// Two lists recording the same query interleaved (A begins, B begins,
    /// A ends, B ends) each end their own.
    #[cfg(windows)]
    #[test]
    fn each_list_pairs_its_own_begin_and_end() {
        // A heap address no other test uses; never dereferenced.
        let heap = 0x5157_0000_usize as This;
        let (a, b) = (0xa000_usize, 0xb000_usize);
        begun(heap, 0, 3, 3, a);
        begun(heap, 0, 3, 3, b);
        assert_eq!(take_begun(heap, 3, a), Some(3));
        assert_eq!(take_begun(heap, 3, b), Some(3));
        // Each end is taken once, and an end without a begin is not.
        assert_eq!(take_begun(heap, 3, a), None);
        assert_eq!(take_begun(heap, 3, b), None);
        assert_eq!(take_begun(heap, 4, a), None);
        // Another index of the same list is its own query.
        begun(heap, 0, 5, 5, a);
        assert_eq!(take_begun(heap, 6, a), None);
        assert_eq!(take_begun(heap, 5, a), Some(5));
        // A query going on after a cut ends as the host query it runs as.
        begun(heap, 0, 5, 70, a);
        assert_eq!(take_begun(heap, 5, a), Some(70));
    }

    /// A list about to be closed gives up the queries still open in it,
    /// with their kinds, and only its own.
    #[cfg(windows)]
    #[test]
    fn a_closing_list_gives_up_its_open_queries() {
        let heap = 0x5157_1000_usize as This;
        let (a, b) = (0xa100_usize, 0xb100_usize);
        begun(heap, 0, 7, 7, a);
        begun(heap, 1, 2, 66, a);
        begun(heap, 0, 7, 7, b);
        assert_eq!(take_begun(heap, 7, a), Some(7));
        begun(heap, 0, 9, 9, a);
        assert_eq!(take_open(a), [(heap, 0, 9, 9), (heap, 1, 2, 66)]);
        assert!(take_open(a).is_empty());
        // An end the title records later in this list finds nothing to end.
        assert_eq!(take_begun(heap, 9, a), None);
        assert_eq!(take_open(b), [(heap, 0, 7, 7)]);
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

        // The query that finds no spare host query at a cut is the one cut
        // short, until it begins anew.
        let mut parts = Parts::new(8, 1);
        assert_eq!(parts.continues(3), Some(8));
        assert!(parts.cut_short(0, 8).is_empty());
        assert_eq!(parts.continues(6), None);
        assert_eq!(parts.continues(2), None);
        assert_eq!(parts.cut_short(0, 8), [2, 6]);
        assert_eq!(parts.cut_short(5, 3), [1], "places from the first resolved");
        assert!(parts.cut_short(3, 3).is_empty());
        parts.starts_over(6);
        assert_eq!(parts.cut_short(0, 8), [2]);
    }

    /// Each cut of a query's host list takes one spare host query, which
    /// stays the query's until it begins anew; with none left the query
    /// does not go on.
    #[test]
    fn spare_queries_hold_the_parts_until_the_query_begins_anew() {
        let mut parts = Parts::new(8, 3);
        assert_eq!(parts.continues(5), Some(8));
        assert_eq!(parts.continues(5), Some(9));
        assert_eq!(parts.continues(2), Some(10));
        assert_eq!(parts.continues(6), None, "no spare query is left");
        assert_eq!(parts.within(0, 8), [(2, 10), (5, 8), (5, 9)]);
        // Places are counted from the first query resolved.
        assert_eq!(parts.within(4, 2), [(1, 8), (1, 9)]);
        assert_eq!(parts.within(3, 2), []);
        assert_eq!(parts.within(6, 2), []);
        // Beginning query 5 again frees its spare queries, lowest first.
        parts.starts_over(5);
        parts.starts_over(7);
        assert_eq!(parts.within(0, 8), [(2, 10)]);
        assert_eq!(parts.continues(6), Some(8));
        assert_eq!(parts.within(0, 8), [(2, 10), (6, 8)]);
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
