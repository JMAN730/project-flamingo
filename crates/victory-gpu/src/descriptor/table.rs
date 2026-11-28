//! The buffer view table of a host view heap: one record per title slot,
//! for translated buffer loads whose element stride only the view knows (a
//! descriptor the program picks at run time, such as each lane's own in a
//! waterfall loop over heap indices; see `victory_shader::emit::UserData`).
//!
//! The console reads the stride from the descriptor. Here a slot holds our
//! own record of the view instead, and the host slot behind it is a view of
//! one kind, so a shader cannot take the stride from either. The table
//! gives it, per slot, a raw view holding the view's bytes, the stride (0
//! for a raw view), the view's first byte in that raw view and its element
//! count. Raw views are made per window of a title buffer's host copy
//! ([`Shadow`]), shared by every slot whose view lies in it, and kept while
//! a record names it: one no record names is dropped from the table, and its
//! heap slot is given back once every host list that bound the table while
//! it was there has finished ([`Alias`]).
//!
//! shortcut: only shader resource views of buffers get records (a slot with
//! another view reads zeros). As for other buffers read through directly
//! indexed heaps (see [`crate::shadow`]), a listed view's bytes are copied
//! when it is listed, at the next draw or dispatch reading the table, and
//! keep what that copied: the shader may reach any listed view, and
//! bringing them all up to date at every such dispatch copied gigabytes a
//! frame (run 189).

use std::collections::HashMap;
use std::sync::Arc;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::host::HostHeap;
use super::{VIEW_TAG, view_at};
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Bytes of a host copy one raw view starts within: a view starts at a
/// multiple of this and reaches twice as far, so every buffer view up to
/// this long fits in the window it starts in.
const WINDOW: u64 = 1 << 28;

const RECORD: usize = victory_shader::emit::BUFFER_TABLE_RECORD as usize;

/// Raw views made and given back over the run, across every table: what
/// the "buffer view table bound" log line reports, to see whether they grow.
static MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RELEASED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One raw view of a window of a host copy, in a dynamic slot of its heap.
/// The table holds one reference while a record names it, and each host
/// list that bound the table while it was there holds another until it is
/// reset or dropped ([`HostList::hold`]): the slot is given back with the
/// last, so no list that may still run reads a slot made again for another
/// view.
struct Alias {
    slot: u32,
    shadow: Arc<Shadow>,
    /// The title heap (address and creation id,
    /// [`crate::descriptor::heap_bind_id`]) whose host heap the slot is in:
    /// a heap released since, or another made at its address, is not given
    /// the slot.
    heap: (usize, u64),
}

impl Drop for Alias {
    fn drop(&mut self) {
        RELEASED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (key, id) = self.heap;
        crate::descriptor::release_dynamic_of(key, id, self.slot, 1);
    }
}

/// A copy buffer and the start of a window in it.
type AliasKey = (usize, u64);

/// The raw views of a table by copy and window, each with how many title
/// slots' records name it, and which one each slot's record names.
struct Aliases<V> {
    by_key: HashMap<AliasKey, (V, u32)>,
    of_slot: Vec<Option<AliasKey>>,
}

impl<V> Aliases<V> {
    fn new(slots: usize) -> Self {
        Self {
            by_key: HashMap::new(),
            of_slot: vec![None; slots],
        }
    }

    fn get(&self, key: &AliasKey) -> Option<&V> {
        self.by_key.get(key).map(|(v, _)| v)
    }

    /// A new raw view, named by no record yet.
    fn insert(&mut self, key: AliasKey, value: V) {
        self.by_key.insert(key, (value, 0));
    }

    /// Title slot `index`'s record now names the raw view `key` (None: no
    /// raw view). One that no record names any more stays until
    /// [`Self::prune`], so another slot changed in the same look can still
    /// find it.
    fn name(&mut self, index: usize, key: Option<AliasKey>) {
        let Some(was) = self.of_slot.get_mut(index) else {
            return;
        };
        if *was == key {
            return;
        }
        if let Some(old) = was.take()
            && let Some((_, uses)) = self.by_key.get_mut(&old)
        {
            *uses -= 1;
        }
        if let Some(new) = key
            && let Some((_, uses)) = self.by_key.get_mut(&new)
        {
            *uses += 1;
            *was = key;
        }
    }

    /// Takes out the raw views no record names.
    fn prune(&mut self) -> Vec<V> {
        let unused: Vec<AliasKey> = self
            .by_key
            .iter()
            .filter(|(_, (_, uses))| *uses == 0)
            .map(|(&key, _)| key)
            .collect();
        unused
            .into_iter()
            .filter_map(|key| self.by_key.remove(&key).map(|(v, _)| v))
            .collect()
    }

    fn len(&self) -> usize {
        self.by_key.len()
    }

    fn values(&self) -> impl Iterator<Item = &V> {
        self.by_key.values().map(|(v, _)| v)
    }
}

/// The raw view of a host copy a buffer view lies in: where it starts in
/// the copy, how long it is, and where the buffer view starts in it.
#[derive(Debug, PartialEq, Eq)]
struct Window {
    start: u64,
    len: u64,
    bias: u32,
}

/// The [`Window`] of `len` bytes at byte `offset` of a host copy with
/// `left` bytes from there to its end: from the multiple of [`WINDOW`] at
/// or below `offset`, up to twice as far (whole 16-byte blocks: a copy is
/// made 64 KiB at a time). None for a view that does not fit, or one not at
/// a dword.
fn window(offset: u64, len: u64, left: u64) -> Option<Window> {
    let start = offset & !(WINDOW - 1);
    let bias = offset - start;
    if !offset.is_multiple_of(4) || len > left || bias + len > 2 * WINDOW {
        return None;
    }
    let len = (offset + left - start).next_multiple_of(16).min(2 * WINDOW);
    Some(Window {
        start,
        len,
        bias: bias as u32,
    })
}

pub(crate) struct BufferTable {
    /// The records (CPU-visible memory the GPU reads).
    buffer: ID3D12Resource,
    data: usize,
    /// The heap slot of the raw view of `buffer`.
    slot: u32,
    /// Per title slot, the record (view index and generation) its entry
    /// was made from (0: none).
    made: Vec<u64>,
    /// How many slots list a view.
    listed: usize,
    /// Raw views of host copies, by copy buffer and window.
    aliases: Aliases<Arc<Alias>>,
    /// The title heap this table's host heap is behind (address and
    /// creation id), for the slots of its raw views.
    title_heap: (usize, u64),
    /// The heap's write count `made` was last brought up to date at
    /// ([`HostHeap`]'s `writes`; `u64::MAX` before the first time).
    seen: u64,
    /// Views listed since the last draw or dispatch that read the table:
    /// their copy and range, to copy once.
    fresh: Vec<(Arc<Shadow>, u64, u64)>,
    /// Whether the last [`Self::alias`] found the heap out of free slots
    /// for a new raw view: slots come back as recordings finish, so the
    /// record is worth making again.
    starved: bool,
}

// SAFETY: `data` is the persistent mapping of `buffer`, written only under
// the heap's table lock.
unsafe impl Send for BufferTable {}

impl BufferTable {
    fn new(heap: &HostHeap, title_heap: (usize, u64)) -> Option<Self> {
        let count = heap.count() as usize;
        let bytes = (count * RECORD).max(RECORD) as u64;
        let (buffer, data) = host()?.shared_buffer(bytes.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || "buffer view table".to_owned());
        // A record of zeros lists no view.
        // SAFETY: the mapping covers `bytes`.
        unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
        let slot = heap.dynamic(1)?;
        heap.raw_resource_at(slot, &buffer, 0, bytes);
        Some(Self {
            buffer,
            data: data as usize,
            slot,
            made: vec![0; count],
            listed: 0,
            aliases: Aliases::new(count),
            title_heap,
            seen: u64::MAX,
            fresh: Vec::new(),
            starved: false,
        })
    }

    fn write(&self, index: u32, record: [u32; 4]) {
        // SAFETY: `index` is below the title slot count the mapping covers.
        unsafe {
            ((self.data + index as usize * RECORD) as *mut [u32; 4]).write_unaligned(record);
        }
    }

    /// The raw view holding `len` bytes at `offset` of `shadow`'s copy
    /// (`left` more to its end from there): its key, its slot and the offset
    /// in it; made on first use.
    fn alias(
        &mut self,
        heap: &HostHeap,
        shadow: &Arc<Shadow>,
        (offset, len, left): (u64, u64, u64),
    ) -> Option<(AliasKey, u32, u32)> {
        let w = window(offset, len, left)?;
        let key = (shadow.buffer.as_raw() as usize, w.start);
        if let Some(alias) = self.aliases.get(&key) {
            return Some((key, alias.slot, w.bias));
        }
        let Some(slot) = heap.dynamic(1) else {
            self.starved = true;
            return None;
        };
        heap.raw_at(slot, shadow, w.start, w.len);
        MADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.aliases.insert(
            key,
            Arc::new(Alias {
                slot,
                shadow: shadow.clone(),
                heap: self.title_heap,
            }),
        );
        Some((key, slot, w.bias))
    }

    /// Brings the records up to date with what the title's slots hold,
    /// unless no slot of the heap changed since the last time. Returns the
    /// raw views no record names any more, taken out of the table.
    fn sync(&mut self, heap: &HostHeap, start: u64) -> Vec<Arc<Alias>> {
        let writes = heap.writes.load(std::sync::atomic::Ordering::Acquire);
        if writes == self.seen {
            return Vec::new();
        }
        self.seen = writes;
        let mut again = false;
        for index in 0..self.made.len() as u32 {
            let at = start + u64::from(index) * 32;
            // SAFETY: a descriptor slot in title memory.
            let [tag, i, generation] = unsafe { (at as *const [u32; 3]).read_unaligned() };
            let key = if tag == VIEW_TAG {
                u64::from(generation) << 32 | u64::from(i)
            } else {
                0
            };
            if self.made[index as usize] == key {
                continue;
            }
            let was = self.made[index as usize];
            self.made[index as usize] = key;
            let read = (key != 0)
                .then(|| view_at(i, generation))
                .flatten()
                .and_then(|view| super::host::selected_buffer(&view));
            self.starved = false;
            let made = read.and_then(|read| {
                let offset = read.shadow.offset(read.address);
                let left = read.shadow.left(read.address);
                let (key, alias, bias) =
                    self.alias(heap, &read.shadow, (offset, read.bytes, left))?;
                // A raw view (stride 0) counts bytes; the shader drops its index.
                let elements = u32::try_from(read.bytes / u64::from(read.stride.max(1))).ok()?;
                self.fresh
                    .push((read.shadow.clone(), read.address, read.bytes));
                Some((key, [alias, read.stride, bias, elements]))
            });
            self.aliases.name(index as usize, made.map(|(key, _)| key));
            let record = made.map(|(_, record)| record);
            let before = self.read(index)[3] != 0 && was != 0;
            match (before, record.is_some()) {
                (false, true) => self.listed += 1,
                (true, false) => self.listed -= 1,
                _ => {}
            }
            self.write(index, record.unwrap_or_default());
            if record.is_none() && self.starved {
                // Not made for want of a slot: forgotten, so the next
                // look makes it, and the next bind looks.
                self.made[index as usize] = 0;
                again = true;
            }
        }
        if again {
            self.seen = u64::MAX;
        }
        self.aliases.prune()
    }

    fn read(&self, index: u32) -> [u32; 4] {
        // SAFETY: `index` is below the title slot count the mapping covers.
        unsafe { ((self.data + index as usize * RECORD) as *const [u32; 4]).read_unaligned() }
    }
}

/// The heap slot of the buffer view table of `heap` (the host heap of the
/// title view heap `key`, whose slots are at `start`), its records brought
/// up to date and the copies they name readable by `list`'s shaders
/// (`graphics`: pixel shaders too). None if the host cannot make it.
pub(crate) fn bind(
    heap: &HostHeap,
    key: usize,
    start: u64,
    list: &HostList,
    graphics: bool,
) -> Option<u32> {
    let mut guard = heap.buffer_table.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let title_heap = (key, crate::descriptor::heap_bind_id(key));
        *guard = Some(BufferTable::new(heap, title_heap)?);
        tracing::info!(slots = heap.count(), "buffer view table made");
    }
    let table = guard.as_mut()?;
    let started = std::time::Instant::now();
    // Raw views no record names any more: let go of once the table's lock
    // is (the lists that bound the table while they were in it hold them
    // until they finish).
    let unnamed = table.sync(heap, start);
    let state = if graphics {
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    } else {
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    };
    list.use_shadow(&table.buffer, state);
    // Every copy a listed view lies in, readable by the shaders and its raw
    // view's slot kept while the list may run; views listed since the last
    // time copied.
    for alias in table.aliases.values() {
        list.use_shadow(&alias.shadow.buffer, state);
        list.hold(alias.clone());
    }
    for (shadow, address, len) in table.fresh.drain(..) {
        shadow.refresh_as(list, address, len, state);
    }
    static BINDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = BINDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        tracing::info!(
            binds = n,
            views = table.listed,
            raw_views = table.aliases.len(),
            raw_views_made = MADE.load(std::sync::atomic::Ordering::Relaxed),
            raw_views_released = RELEASED.load(std::sync::atomic::Ordering::Relaxed),
            us = started.elapsed().as_micros() as u64,
            "buffer view table bound"
        );
    }
    let slot = table.slot;
    drop(guard);
    drop(unnamed);
    Some(slot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_view_shares_the_raw_view_of_the_window_it_starts_in() {
        // Near the start of a 1 MiB copy.
        assert_eq!(
            window(0x120, 0x40, 0x10_0000 - 0x120),
            Some(Window {
                start: 0,
                len: 0x10_0000,
                bias: 0x120
            })
        );
        // In the second window of a large copy, reaching past it.
        let offset = WINDOW + 0x1000;
        let left = 3 * WINDOW - offset;
        assert_eq!(
            window(offset, WINDOW, left),
            Some(Window {
                start: WINDOW,
                len: 2 * WINDOW,
                bias: 0x1000
            })
        );
    }

    #[test]
    fn a_copy_ending_inside_a_block_is_viewed_to_the_block_end() {
        let w = window(8, 4, 6).expect("a window");
        assert_eq!((w.start, w.len, w.bias), (0, 16, 8));
    }

    /// A raw view stays while some slot's record names it and is taken out
    /// once none does, not before the look that changed them ends.
    #[test]
    fn raw_views_no_record_names_are_taken_out() {
        let (a, b) = ((1, 0), (1, WINDOW));
        let mut aliases = Aliases::new(4);
        aliases.insert(a, "a");
        aliases.name(0, Some(a));
        aliases.name(1, Some(a));
        assert!(aliases.prune().is_empty());
        // Slot 0 moves to a new raw view, slot 1 to no view: `a` is named
        // by none, but slot 2 finds it before the prune.
        aliases.insert(b, "b");
        aliases.name(0, Some(b));
        aliases.name(1, None);
        aliases.name(2, Some(a));
        assert!(aliases.prune().is_empty());
        aliases.name(2, None);
        assert_eq!(aliases.prune(), ["a"]);
        assert_eq!(aliases.len(), 1);
        // Naming the same raw view again counts it once.
        aliases.name(0, Some(b));
        aliases.name(0, None);
        assert_eq!(aliases.prune(), ["b"]);
        assert_eq!(aliases.len(), 0);
    }

    /// A raw view made but named by no record (its record could not be
    /// made) goes at the next prune; a name of a key not in the table, or
    /// of a slot past the table, changes nothing.
    #[test]
    fn a_raw_view_named_by_none_does_not_stay() {
        let mut aliases = Aliases::new(2);
        aliases.insert((7, 0), 7);
        aliases.name(0, Some((8, 0)));
        aliases.name(9, Some((7, 0)));
        assert_eq!(aliases.prune(), [7]);
        aliases.insert((8, 0), 8);
        aliases.name(0, Some((8, 0)));
        assert!(aliases.prune().is_empty());
        assert_eq!(aliases.values().copied().collect::<Vec<_>>(), [8]);
    }

    #[test]
    fn views_that_do_not_fit_have_no_window() {
        // Longer than what it starts in can reach.
        assert_eq!(window(WINDOW - 16, 2 * WINDOW, 4 * WINDOW), None);
        // Past the copy's end.
        assert_eq!(window(0, 32, 16), None);
        // Not at a dword.
        assert_eq!(window(2, 4, 64), None);
    }
}
