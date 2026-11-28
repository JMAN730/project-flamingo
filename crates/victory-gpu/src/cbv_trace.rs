//! `VICTORY_TRACE_CBV=<hash>[,<hash>...]` (diagnostics): traces the constant
//! buffer views the stages with those runtime hashes read, from the view to
//! the bytes the GPU sees:
//!
//! - a fresh table's entry: title address, size, the scratch GPU address
//!   its bytes are filled into at submission and the host slot given it;
//! - a constant view made at a title slot: title address, size and the GPU
//!   address in the buffer's host copy;
//! - a fill of the same title range reused from earlier in the list;
//! - after the list has run, the first 64 bytes of each traced fill next to
//!   the same bytes of title memory.
//!
//! A fill that fails stops the run with a crash report already
//! (`title-to-upload fill failed`).

use std::cell::Cell;
use std::collections::HashSet;
use std::sync::OnceLock;

/// Bytes of each traced fill compared after the list has run.
pub(crate) const COMPARED: usize = 64;

fn hashes() -> &'static HashSet<String> {
    static HASHES: OnceLock<HashSet<String>> = OnceLock::new();
    HASHES.get_or_init(|| parse(&std::env::var("VICTORY_TRACE_CBV").unwrap_or_default()))
}

/// The runtime hashes a `VICTORY_TRACE_CBV` value names (lowercase).
fn parse(value: &str) -> HashSet<String> {
    value
        .split(',')
        .map(|h| h.trim().to_ascii_lowercase())
        .filter(|h| !h.is_empty())
        .collect()
}

thread_local! {
    /// The traced stage being bound on this thread, if any.
    static ON: Cell<Option<&'static str>> = const { Cell::new(None) };
}

/// While alive, this thread binds a stage whose views are traced.
pub(crate) struct Scope(Option<&'static str>);

impl Drop for Scope {
    fn drop(&mut self) {
        ON.with(|on| on.set(self.0));
    }
}

/// Binding the stage with runtime hash `hash`: traced if named.
pub(crate) fn enter(hash: &str) -> Scope {
    let wanted = hashes().get(&hash.to_ascii_lowercase()).map(String::as_str);
    Scope(ON.with(|on| on.replace(wanted)))
}

/// The traced stage being bound on this thread.
pub(crate) fn on() -> Option<&'static str> {
    ON.with(Cell::get)
}

/// `bytes` as hex, for the log.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hashes are named comma-separated, in either case; a scope restores
    /// what the thread traced before.
    #[test]
    fn hashes_name_the_traced_stages() {
        let named = parse(" F0C91B4D2578F5BF, 7fe75645c406cac6,,");
        assert_eq!(named.len(), 2);
        assert!(named.contains("f0c91b4d2578f5bf"));
        assert!(parse("").is_empty());
        assert_eq!(on(), None);
        {
            let _scope = enter("not traced");
            assert_eq!(on(), None);
        }
        assert_eq!(hex(&[0, 0xab, 0x10]), "00ab10");
    }
}
