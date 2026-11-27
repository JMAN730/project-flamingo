//! Query heaps. Occlusion heaps get a host query heap of the same size, so
//! the title's occlusion queries run on the host GPU (see `list.rs`); other
//! kinds only remember their description.
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
            let mut made: Option<ID3D12QueryHeap> = None;
            // SAFETY: plain object creation.
            let result = unsafe {
                device.CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_OCCLUSION,
                        Count: heap.desc.count,
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
            if let Err(e) = end_every_query(&made, heap.desc.count) {
                tracing::warn!(error = %e, count = heap.desc.count, "host occlusion queries not set up");
                return None;
            }
            Some(HostQueries { heap: made })
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

/// Queries begun on the host and not yet ended, by (host list, heap,
/// index): the query's kind. Each recording list pairs its own `BeginQuery`
/// and `EndQuery`, so lists recording the same query at once do not end
/// each other's.
// shortcut: a list reset between a title's begin and end keeps the entry,
// which an end in its next recording would then match.
#[cfg(windows)]
static OPEN: std::sync::Mutex<Option<OpenQueries>> = std::sync::Mutex::new(None);

/// (host list, heap, index) to the query's kind.
#[cfg(windows)]
type OpenQueries = std::collections::HashMap<(usize, usize, u32), u32>;

/// Notes query `index` of `this`, of kind `kind`, begun in host list `list`.
#[cfg(windows)]
pub(crate) fn begun(this: This, kind: u32, index: u32, list: usize) {
    OPEN.lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(Default::default)
        .insert((list, this as usize, index), kind);
}

/// The queries begun in host list `list` and not ended there, as (heap,
/// kind, index), forgetting them: the list is about to be closed, and a
/// query may not stay open past the end of its list.
#[cfg(windows)]
pub(crate) fn take_open(list: usize) -> Vec<(This, u32, u32)> {
    let mut open = OPEN.lock().unwrap_or_else(|e| e.into_inner());
    let Some(open) = open.as_mut() else {
        return Vec::new();
    };
    let mut taken: Vec<(usize, u32, u32)> = open
        .extract_if(|key, _| key.0 == list)
        .map(|((_, heap, index), kind)| (heap, kind, index))
        .collect();
    taken.sort_unstable();
    taken
        .into_iter()
        .map(|(heap, kind, index)| (heap as This, kind, index))
        .collect()
}

/// Whether query `index` of `this` was begun in host list `list` and not
/// ended there since, forgetting it.
#[cfg(windows)]
pub(crate) fn take_begun(this: This, index: u32, list: usize) -> bool {
    OPEN.lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        .is_some_and(|open| open.remove(&(list, this as usize, index)).is_some())
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

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUERY_HEAP,
        iid,
        object::is(iid, &HEAP_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
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

    /// Two lists recording the same query interleaved (A begins, B begins,
    /// A ends, B ends) each end their own.
    #[cfg(windows)]
    #[test]
    fn each_list_pairs_its_own_begin_and_end() {
        // A heap address no other test uses; never dereferenced.
        let heap = 0x5157_0000_usize as This;
        let (a, b) = (0xa000_usize, 0xb000_usize);
        begun(heap, 0, 3, a);
        begun(heap, 0, 3, b);
        assert!(take_begun(heap, 3, a));
        assert!(take_begun(heap, 3, b));
        // Each end is taken once, and an end without a begin is not.
        assert!(!take_begun(heap, 3, a));
        assert!(!take_begun(heap, 3, b));
        assert!(!take_begun(heap, 4, a));
        // Another index of the same list is its own query.
        begun(heap, 0, 5, a);
        assert!(!take_begun(heap, 6, a));
        assert!(take_begun(heap, 5, a));
    }

    /// A list about to be closed gives up the queries still open in it,
    /// with their kinds, and only its own.
    #[cfg(windows)]
    #[test]
    fn a_closing_list_gives_up_its_open_queries() {
        let heap = 0x5157_1000_usize as This;
        let (a, b) = (0xa100_usize, 0xb100_usize);
        begun(heap, 0, 7, a);
        begun(heap, 1, 2, a);
        begun(heap, 0, 7, b);
        assert!(take_begun(heap, 7, a));
        begun(heap, 0, 9, a);
        assert_eq!(take_open(a), [(heap, 0, 9), (heap, 1, 2)]);
        assert!(take_open(a).is_empty());
        // The ends the title records later find nothing to end.
        assert!(!take_begun(heap, 9, a));
        assert_eq!(take_open(b), [(heap, 0, 7)]);
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
