//! What the title passes to `PresentX` beyond the resources we show: each
//! plane's fields after `resources` (0x18 up to the plane stride 0x48, most
//! likely source rects, placement, colour space and flags; the layout is
//! not confirmed) and the call's last argument, logged as numbers so a run
//! shows whether they stay at full-plane defaults. Logged once, then again
//! when they change, at most every [`CHANGE_EVERY`] and [`CHANGE_LOGS`]
//! times. The second plane's
//! format, size and flags log once, whether or not
//! `VICTORY_PRESENT_TWO_PLANES` composites it.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Bytes from one plane description to the next (the Series S layout,
/// larger than the prefix `queue::Plane` describes).
pub(crate) const PLANE_STRIDE: usize = 0x48;
/// Where the fields after a plane's `resources` start.
const TAIL_START: usize = 0x18;
/// Dwords of each plane's tail.
const TAIL_DWORDS: usize = (PLANE_STRIDE - TAIL_START) / 4;
/// Dwords read at the last argument (its size is not known; this many are
/// logged when they can be read).
const LAST_DWORDS: usize = 8;
/// Planes logged.
const PLANES: usize = 2;
/// The least time between two logs of changed parameters.
const CHANGE_EVERY: Duration = Duration::from_secs(5);
/// Logs of changed parameters per run (a field that changes every frame
/// would otherwise log every [`CHANGE_EVERY`] for the whole run).
const CHANGE_LOGS: u32 = 20;

/// The numbers one present passes: per plane its tail (`None` past
/// `planes` or when unreadable), and the dwords at the last argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Parameters {
    planes: u32,
    tails: [Option<[u32; TAIL_DWORDS]>; PLANES],
    last: Option<[u32; LAST_DWORDS]>,
}

/// The title's description of a presented resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlaneDesc {
    pub(crate) format: u32,
    pub(crate) width: u64,
    pub(crate) height: u32,
    pub(crate) mips: u16,
    pub(crate) samples: u32,
    pub(crate) dimension: u32,
    pub(crate) flags: u32,
}

/// Whether to log `now`'s parameters, given what was logged last and when:
/// the first ones always; changed ones once [`CHANGE_EVERY`] has passed.
fn should_log(last: Option<(&Parameters, Instant)>, now: &Parameters, at: Instant) -> bool {
    match last {
        None => true,
        Some((last, when)) => last != now && at.duration_since(when) >= CHANGE_EVERY,
    }
}

fn words<const N: usize>(at: usize) -> Option<[u32; N]> {
    if at == 0 {
        return None;
    }
    let mut bytes = [[0u8; 4]; N];
    if !crate::host::read_into(at as *const u8, bytes.as_flattened_mut()) {
        return None;
    }
    Some(bytes.map(u32::from_le_bytes))
}

fn read_usize(at: usize) -> Option<usize> {
    let mut bytes = [0; size_of::<usize>()];
    crate::host::read_into(at as *const u8, &mut bytes).then(|| usize::from_ne_bytes(bytes))
}

/// The first resource of plane `index`, when the plane lists one.
fn first_resource(planes: usize, index: usize) -> Option<usize> {
    let plane = planes.checked_add(index * PLANE_STRIDE)?;
    let mut count = [0; 4];
    if !crate::host::read_into((plane + 8) as *const u8, &mut count)
        || u32::from_ne_bytes(count) == 0
    {
        return None;
    }
    let resources = read_usize(plane + 0x10).filter(|&r| r != 0)?;
    read_usize(resources).filter(|&r| r != 0)
}

/// Logs the present's parameters (see the module docs). `planes` plane
/// descriptions start at `plane_parameters`; `last` is the last argument.
pub(crate) fn note(planes: u32, plane_parameters: *const u8, last: *const u8) {
    struct Logged {
        parameters: Parameters,
        at: Instant,
        changes: u32,
        second_plane: bool,
    }
    static LOGGED: Mutex<Option<Logged>> = Mutex::new(None);
    let base = plane_parameters as usize;
    let mut tails = [None; PLANES];
    for (i, tail) in tails.iter_mut().enumerate().take(planes as usize) {
        *tail = base
            .checked_add(i * PLANE_STRIDE + TAIL_START)
            .filter(|_| base != 0)
            .and_then(words::<TAIL_DWORDS>);
    }
    let now = Parameters {
        planes,
        tails,
        last: words::<LAST_DWORDS>(last as usize),
    };
    let at = Instant::now();
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    let first = logged.is_none();
    let changes = logged.as_ref().map_or(0, |l| l.changes);
    if changes <= CHANGE_LOGS
        && should_log(logged.as_ref().map(|l| (&l.parameters, l.at)), &now, at)
    {
        tracing::info!(
            planes,
            plane0 = ?now.tails[0],
            plane1 = ?now.tails[1],
            last = ?now.last,
            "{}",
            if first {
                "present parameters (plane fields 0x18..0x48 as dwords, and the last argument)"
            } else if changes < CHANGE_LOGS {
                "present parameters changed"
            } else {
                "present parameters changed (further changes are not logged)"
            }
        );
        let second_plane = logged.as_ref().is_some_and(|l| l.second_plane);
        *logged = Some(Logged {
            parameters: now,
            at,
            changes: if first { 0 } else { changes + 1 },
            second_plane,
        });
    }
    let Some(l) = logged.as_mut() else {
        return;
    };
    if planes >= 2 && base != 0 && !l.second_plane {
        l.second_plane = true;
        drop(logged);
        match first_resource(base, 1).and_then(crate::resource::present_plane_desc) {
            Some(d) => tracing::info!(
                format = d.format,
                width = d.width,
                height = d.height,
                mips = d.mips,
                samples = d.samples,
                dimension = d.dimension,
                flags = format_args!("{:#x}", d.flags),
                "the second present plane"
            ),
            None => tracing::info!("the second present plane lists no known render target"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(x: u32) -> Parameters {
        Parameters {
            planes: 1,
            tails: [Some([x; TAIL_DWORDS]), None],
            last: None,
        }
    }

    /// The first parameters log; the same ones never again; changed ones
    /// once enough time has passed since the last log.
    #[test]
    fn present_parameters_log_once_and_then_on_change_at_most_every_few_seconds() {
        let t0 = Instant::now();
        let (a, b) = (parameters(0), parameters(1));
        assert!(should_log(None, &a, t0));
        assert!(!should_log(Some((&a, t0)), &a, t0 + CHANGE_EVERY * 10));
        assert!(!should_log(Some((&a, t0)), &b, t0 + CHANGE_EVERY / 2));
        assert!(should_log(Some((&a, t0)), &b, t0 + CHANGE_EVERY));
    }

    /// A plane's tail is the dwords from 0x18 to the next plane.
    #[test]
    fn a_plane_tail_is_twelve_dwords_up_to_the_stride() {
        assert_eq!(TAIL_DWORDS, 12);
        assert_eq!(TAIL_START, size_of::<crate::queue::Plane>());
    }
}
