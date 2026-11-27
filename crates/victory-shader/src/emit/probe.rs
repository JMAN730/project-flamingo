//! Knobs and records of the split differential harness (`split_diff`):
//! thread-local, inert unless the harness sets them on its own thread, so
//! translation elsewhere is unchanged.

use std::cell::{Cell, RefCell};

/// A heap descriptor the translated shader reads: from root parameter
/// `param` (layout index) at byte `offset` into it, as `props` (the DXIL
/// resource properties its handle is annotated with).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HeapUse {
    pub param: usize,
    pub offset: u32,
    /// The code adds a term of its own to the address: the entry is not
    /// known before the program runs.
    pub dynamic: bool,
    pub sampler: bool,
    pub props: (u32, u32),
}

/// What [`record`] collected while translating.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Uses {
    pub heap: Vec<HeapUse>,
    /// Descriptors whose heap entry the code computes per lane or reads
    /// from memory: no fixed binding can be prepared for them.
    pub untracked: u32,
}

thread_local! {
    static USES: RefCell<Option<Uses>> = const { RefCell::new(None) };
    static SPLIT_WAVES: Cell<Option<u32>> = const { Cell::new(None) };
    static SPIN_LIMIT: Cell<Option<u32>> = const { Cell::new(None) };
    static FORCE_SPLIT: Cell<bool> = const { Cell::new(false) };
    static LOOP_LIMIT: Cell<Option<u32>> = const { Cell::new(None) };
    static STRICT_LDS: Cell<bool> = const { Cell::new(false) };
    static CLEAR_LDS: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f` with split programs clearing their LDS on entry (what the
/// program reads before writing then does not depend on what ran before).
pub fn with_clear_lds<T>(clear: bool, f: impl FnOnce() -> T) -> T {
    let old = CLEAR_LDS.with(|c| c.replace(clear));
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            CLEAR_LDS.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(old);
    f()
}

pub(crate) fn clear_lds() -> bool {
    CLEAR_LDS.with(Cell::get)
}

/// Runs `f` with the split lowering syncing its halves before every LDS
/// access, runs of plain reads or writes included.
pub fn with_strict_lds<T>(strict: bool, f: impl FnOnce() -> T) -> T {
    let old = STRICT_LDS.with(|c| c.replace(strict));
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            STRICT_LDS.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(old);
    f()
}

pub(crate) fn strict_lds() -> bool {
    STRICT_LDS.with(Cell::get)
}

/// Runs `f` with loops of translated shaders stopped after `limit`
/// iterations instead of [`super::loop_limit`]'s.
pub fn with_loop_limit<T>(limit: Option<u32>, f: impl FnOnce() -> T) -> T {
    let old = LOOP_LIMIT.with(|c| c.replace(limit));
    struct Restore(Option<u32>);
    impl Drop for Restore {
        fn drop(&mut self) {
            LOOP_LIMIT.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(old);
    f()
}

pub(crate) fn loop_limit() -> Option<u32> {
    LOOP_LIMIT.with(Cell::get)
}

/// Runs `f` recording the heap descriptors the translation reads.
pub fn record<T>(f: impl FnOnce() -> T) -> (T, Uses) {
    let old = USES.with(|u| u.replace(Some(Uses::default())));
    let r = f();
    let uses = USES.with(|u| u.replace(old)).unwrap_or_default();
    (r, uses)
}

pub(crate) fn recording() -> bool {
    USES.with(|u| u.borrow().is_some())
}

pub(crate) fn note(u: HeapUse) {
    USES.with(|r| {
        if let Some(r) = r.borrow_mut().as_mut()
            && !r.heap.contains(&u)
        {
            r.heap.push(u);
        }
    });
}

pub(crate) fn note_untracked() {
    USES.with(|r| {
        if let Some(r) = r.borrow_mut().as_mut() {
            r.untracked += 1;
        }
    });
}

/// Runs `f` with knobs set on this thread: `waves` lowers a split program
/// as for that many console waves per group (its exchange mechanism; the
/// group itself is unchanged), `spin_limit` bounds a half's wait for its
/// partner, `force_split` ignores the lists of programs left out of the
/// split.
pub fn with_knobs<T>(
    waves: Option<u32>,
    spin_limit: Option<u32>,
    force_split: bool,
    f: impl FnOnce() -> T,
) -> T {
    let old = (
        SPLIT_WAVES.with(|c| c.replace(waves)),
        SPIN_LIMIT.with(|c| c.replace(spin_limit)),
        FORCE_SPLIT.with(|c| c.replace(force_split)),
    );
    struct Restore((Option<u32>, Option<u32>, bool));
    impl Drop for Restore {
        fn drop(&mut self) {
            SPLIT_WAVES.with(|c| c.set(self.0.0));
            SPIN_LIMIT.with(|c| c.set(self.0.1));
            FORCE_SPLIT.with(|c| c.set(self.0.2));
        }
    }
    let _restore = Restore(old);
    f()
}

pub(crate) fn split_waves() -> Option<u32> {
    SPLIT_WAVES.with(Cell::get)
}

pub(crate) fn spin_limit() -> Option<u32> {
    SPIN_LIMIT.with(Cell::get)
}

pub(crate) fn force_split() -> bool {
    FORCE_SPLIT.with(Cell::get)
}
