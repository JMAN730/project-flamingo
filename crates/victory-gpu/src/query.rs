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
    /// The host occlusion heap (made on first use) and, per query, whether
    /// one was ended on the host since it was made.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<HostQueries>>,
}

#[cfg(windows)]
struct HostQueries {
    heap: windows::Win32::Graphics::Direct3D12::ID3D12QueryHeap,
    ended: std::sync::Mutex<Vec<bool>>,
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
            Some(HostQueries {
                heap: made?,
                ended: std::sync::Mutex::new(vec![false; heap.desc.count as usize]),
            })
        })
        .as_ref()
}

/// The host occlusion heap behind the title's query heap `this`, if it is an
/// occlusion heap and the host GPU made one.
#[cfg(windows)]
pub(crate) fn occlusion_heap(
    this: This,
) -> Option<windows::Win32::Graphics::Direct3D12::ID3D12QueryHeap> {
    host_queries(this).map(|q| q.heap.clone())
}

/// Notes that query `index` of `this` was ended on the host: its result
/// can be resolved from then on.
#[cfg(windows)]
pub(crate) fn set_ended(this: This, index: u32) {
    if let Some(q) = host_queries(this)
        && let Some(e) = q
            .ended
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(index as usize)
    {
        *e = true;
    }
}

/// Whether query `index` of `this` was ever ended on the host.
#[cfg(windows)]
pub(crate) fn ended(this: This, index: u32) -> bool {
    host_queries(this).is_some_and(|q| {
        q.ended
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .copied()
            .unwrap_or(false)
    })
}

/// Queries begun on the host and not yet ended: by (heap, index), the host
/// list they were begun in (an end must be recorded in the same list).
#[cfg(windows)]
static OPEN: std::sync::Mutex<Option<std::collections::HashMap<(usize, u32), usize>>> =
    std::sync::Mutex::new(None);

/// Notes query `index` of `this` begun in host list `list`.
#[cfg(windows)]
pub(crate) fn begun(this: This, index: u32, list: usize) {
    OPEN.lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(Default::default)
        .insert((this as usize, index), list);
}

/// The host list query `index` of `this` was begun in, forgetting it.
#[cfg(windows)]
pub(crate) fn take_begun(this: This, index: u32) -> Option<usize> {
    OPEN.lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()?
        .remove(&(this as usize, index))
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
}
