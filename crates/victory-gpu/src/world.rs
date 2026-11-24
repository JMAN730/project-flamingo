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

/// A color target a frame rendered to: the frame (presents before it),
/// its area and its host copy.
struct Candidate {
    frame: u64,
    area: u64,
    copy: std::sync::Arc<crate::resource::HostTexture>,
}

/// The color targets of the last frames, at most [`KEPT`] (newest first).
static WORLD: std::sync::Mutex<Vec<Candidate>> = std::sync::Mutex::new(Vec::new());
const KEPT: usize = 16;

/// Draws run on the host since the last present from which `auto` takes
/// the frame for one of the world.
const AUTO_DRAWS: u64 = 800;

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
    // The last bound comes first; one entry per texture.
    world.retain(|c| c.copy.texture != copy.texture && c.frame + 2 >= frame);
    world.insert(0, Candidate { frame, area, copy });
    world.truncate(KEPT);
}

/// The world target to present in place of the presented planes
/// (`planes`, host textures, plane 0 first), if `VICTORY_PRESENT_1440P`
/// asks for one and the last frames rendered to one that is no plane
/// (`auto` would otherwise find the 4K interface plane): at a size, the
/// last bound; in `auto`, the largest of the newest frame. The frame's
/// targets are logged a few times, to choose a size.
pub(crate) fn take_world(
    planes: &[&ID3D12Resource],
) -> Option<std::sync::Arc<crate::resource::HostTexture>> {
    let pick = world_pick()?;
    let now = crate::queue::presented();
    let world = WORLD.lock().unwrap_or_else(|e| e.into_inner());
    let fresh = || {
        world
            .iter()
            .filter(|c| c.frame + 2 >= now && !planes.iter().any(|&p| c.copy.texture == *p))
    };
    if now.is_power_of_two() && now >= 256 {
        let sizes: Vec<String> = world
            .iter()
            .map(|c| {
                // SAFETY: a live host texture.
                let d = unsafe { c.copy.texture.GetDesc() };
                let plane = planes.iter().any(|&p| c.copy.texture == *p);
                format!("{}x{} f{}{}", d.Width, d.Height, d.Format.0, if plane { " (plane)" } else { "" })
            })
            .collect();
        tracing::info!(frame = now, targets = ?sizes, "VICTORY_PRESENT_1440P: color targets of the last frames");
    }
    // `auto` only for a frame of the world (menus draw a few hundred, the
    // world thousands), so menus keep showing the title's own planes.
    static LAST_DRAWN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let drawn = crate::draw::drawn();
    let since = drawn.saturating_sub(LAST_DRAWN.swap(drawn, std::sync::atomic::Ordering::Relaxed));
    let chosen = match pick {
        WorldPick::Size(..) => fresh().next(),
        WorldPick::Largest if since < AUTO_DRAWS => None,
        WorldPick::Largest => {
            let newest = fresh().map(|c| c.frame).max()?;
            fresh().filter(|c| c.frame == newest).max_by_key(|c| c.area)
        }
    }?;
    let copy = chosen.copy.clone();
    drop(world);
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
