//! The world's render target presented in place of the title's frame
//! (`VICTORY_PRESENT_1440P`), for titles whose final upscale stays black.

use windows::Win32::Graphics::Direct3D12::*;

/// `VICTORY_PRESENT_1440P` (any value but `0`, or `<w>x<h>`, or `auto`):
/// present the world's own render target in place of plane 0. On Xbox
/// Series X the title draws the world at 2560x1440 and upscales it into the
/// 3840x2160 plane, a pass that stays black here (upstream #841); the
/// 1440p target itself holds the picture (mpDom's finding). `<w>x<h>` picks
/// another size (Series S renders lower); `auto` the largest 8-bit color
/// target the frame renders to that is not plane 0 itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorldPick {
    Size(u64, u32),
    Largest,
}

fn world_pick() -> Option<WorldPick> {
    static PICK: std::sync::OnceLock<Option<WorldPick>> = std::sync::OnceLock::new();
    *PICK.get_or_init(|| parse_world_pick(&std::env::var("VICTORY_PRESENT_1440P").ok()?))
}

fn parse_world_pick(v: &str) -> Option<WorldPick> {
    let v = v.trim();
    if v.is_empty() || v == "0" {
        return None;
    }
    if v.eq_ignore_ascii_case("auto") {
        return Some(WorldPick::Largest);
    }
    if let Some((w, h)) = v.split_once(['x', 'X'])
        && let (Ok(w), Ok(h)) = (w.trim().parse(), h.trim().parse())
    {
        return Some(WorldPick::Size(w, h));
    }
    Some(WorldPick::Size(2560, 1440))
}

/// The world target noted for the frame being built: (frame, area, copy).
static WORLD: std::sync::Mutex<Option<(u64, u64, std::sync::Arc<crate::resource::HostTexture>)>> =
    std::sync::Mutex::new(None);

/// Notes a render target bound by the frame being built, for
/// [`take_world`] (cheap when `VICTORY_PRESENT_1440P` is unset).
pub(crate) fn note_world(resource: crate::object::This) {
    let Some(pick) = world_pick() else { return };
    if resource.is_null() {
        return;
    }
    let Some(copy) = crate::resource::host_copy(resource) else {
        return;
    };
    // SAFETY: a live host texture.
    let d = unsafe { copy.texture.GetDesc() };
    // 8-bit color (RGBA and BGRA, linear and sRGB) and 10-bit color: what a
    // swap chain shows without tone mapping.
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || d.SampleDesc.Count != 1
        || !matches!(d.Format.0, 24 | 27 | 28 | 29 | 87 | 88 | 90 | 91)
    {
        return;
    }
    let area = d.Width * u64::from(d.Height);
    let fits = match pick {
        WorldPick::Size(w, h) => d.Width == w && d.Height == h,
        WorldPick::Largest => area >= 640 * 360,
    };
    if !fits {
        return;
    }
    let frame = crate::queue::presented();
    let mut world = WORLD.lock().unwrap_or_else(|e| e.into_inner());
    let keep = match &*world {
        // The last one bound wins at the asked size; the largest in auto.
        Some((f, a, _)) if *f == frame && pick == WorldPick::Largest => area > *a,
        _ => true,
    };
    if keep {
        *world = Some((frame, area, copy));
    }
}

/// The world target to present in place of `plane0` (a host texture), if
/// `VICTORY_PRESENT_1440P` asks for one and the frame (or the one before)
/// rendered to one; logged once.
pub(crate) fn take_world(
    plane0: &ID3D12Resource,
) -> Option<std::sync::Arc<crate::resource::HostTexture>> {
    world_pick()?;
    let (frame, _, copy) = WORLD.lock().unwrap_or_else(|e| e.into_inner()).clone()?;
    // A frame without one shows the title's own (menus, loading screens).
    if frame + 2 < crate::queue::presented() || copy.texture == *plane0 {
        return None;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: a live host texture.
        let d = unsafe { copy.texture.GetDesc() };
        tracing::info!(
            width = d.Width,
            height = d.Height,
            format = d.Format.0,
            "VICTORY_PRESENT_1440P: presenting the world's render target as plane 0"
        );
    });
    Some(copy)
}

#[cfg(test)]
mod world_tests {
    use super::{WorldPick, parse_world_pick};

    #[test]
    fn world_pick_reads_the_switch() {
        assert_eq!(parse_world_pick("1"), Some(WorldPick::Size(2560, 1440)));
        assert_eq!(parse_world_pick(" 1600x900 "), Some(WorldPick::Size(1600, 900)));
        assert_eq!(parse_world_pick("AUTO"), Some(WorldPick::Largest));
        assert_eq!(parse_world_pick("0"), None);
        assert_eq!(parse_world_pick(""), None);
    }
}
