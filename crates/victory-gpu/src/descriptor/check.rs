//! The kind of each host descriptor we write, for comparing with what the
//! translated shaders declare at the slots they read (`VICTORY_CHECK_DESCRIPTORS`,
//! diagnostics builds). NVIDIA stops the device with a TEX LAYOUT exception
//! when a shader reads a texture or buffer through a descriptor of another
//! kind (dimension, array-ness, buffer against texture).
//!
//! Every `Create*View` and `CreateSampler` into a host heap notes the kind it
//! made at that CPU handle. A slot never noted holds the null 2D view every
//! view heap starts with (a point sampler in sampler heaps).

use std::collections::HashMap;
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;

/// What a descriptor is, as far as a shader's declaration can disagree.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Kind {
    pub(crate) class: Class,
    /// The view dimension (`D3D12_SRV_DIMENSION` / `D3D12_UAV_DIMENSION`
    /// values); 0 for constant buffers and samplers, or a view made from a
    /// resource without a description.
    pub(crate) dimension: u8,
    /// Buffers: typed, raw or structured.
    pub(crate) buffer: Buffer,
    /// A null view (no resource).
    pub(crate) null: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Class {
    Srv,
    Uav,
    Cbv,
    Sampler,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Buffer {
    NotOne,
    Typed,
    Raw,
    Structured,
}

impl Kind {
    pub(crate) const INITIAL_VIEW: Kind = Kind {
        class: Class::Srv,
        dimension: D3D12_SRV_DIMENSION_TEXTURE2D.0 as u8,
        buffer: Buffer::NotOne,
        null: true,
    };
    pub(crate) const SAMPLER: Kind = Kind {
        class: Class::Sampler,
        dimension: 0,
        buffer: Buffer::NotOne,
        null: false,
    };

    pub(crate) fn srv(desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>, null: bool) -> Kind {
        let Some(desc) = desc else {
            return Kind {
                class: Class::Srv,
                dimension: 0,
                buffer: Buffer::NotOne,
                null,
            };
        };
        let buffer = if desc.ViewDimension == D3D12_SRV_DIMENSION_BUFFER {
            // SAFETY: the buffer member of a buffer view.
            let b = unsafe { desc.Anonymous.Buffer };
            over_limit("srv", b.NumElements, b.FirstElement);
            if b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
                Buffer::Raw
            } else if b.StructureByteStride != 0 {
                Buffer::Structured
            } else {
                Buffer::Typed
            }
        } else {
            Buffer::NotOne
        };
        Kind {
            class: Class::Srv,
            dimension: desc.ViewDimension.0 as u8,
            buffer,
            null,
        }
    }

    pub(crate) fn uav(desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>, null: bool) -> Kind {
        let Some(desc) = desc else {
            return Kind {
                class: Class::Uav,
                dimension: 0,
                buffer: Buffer::NotOne,
                null,
            };
        };
        let buffer = if desc.ViewDimension == D3D12_UAV_DIMENSION_BUFFER {
            // SAFETY: the buffer member of a buffer view.
            let b = unsafe { desc.Anonymous.Buffer };
            over_limit("uav", b.NumElements, b.FirstElement);
            if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
                Buffer::Raw
            } else if b.StructureByteStride != 0 {
                Buffer::Structured
            } else {
                Buffer::Typed
            }
        } else {
            Buffer::NotOne
        };
        Kind {
            class: Class::Uav,
            dimension: desc.ViewDimension.0 as u8,
            buffer,
            null,
        }
    }

    pub(crate) fn cbv(null: bool) -> Kind {
        Kind {
            class: Class::Cbv,
            dimension: 0,
            buffer: Buffer::NotOne,
            null,
        }
    }

    /// The kind a shader declares with the DXIL resource properties
    /// `props` (kind in the low byte, `0x1000` for unordered access); `None`
    /// for kinds this check does not know.
    pub(crate) fn declared(props: (u32, u32), sampler: bool) -> Option<Kind> {
        if sampler {
            return Some(Kind::SAMPLER);
        }
        let uav = props.0 & 0x1000 != 0;
        let class = if uav { Class::Uav } else { Class::Srv };
        // DXIL resource kinds → view dimensions (SRV and UAV numbering agree
        // for the kinds both have).
        let (dimension, buffer) = match props.0 & 0xff {
            1 => (2, Buffer::NotOne),
            2 => (4, Buffer::NotOne),
            3 => (6, Buffer::NotOne),
            4 => (8, Buffer::NotOne),
            5 if !uav => (9, Buffer::NotOne),
            6 => (3, Buffer::NotOne),
            7 => (5, Buffer::NotOne),
            8 => (7, Buffer::NotOne),
            9 if !uav => (10, Buffer::NotOne),
            10 => (1, Buffer::Typed),
            11 => (1, Buffer::Raw),
            12 => (1, Buffer::Structured),
            13 => return Some(Kind::cbv(false)),
            16 if !uav => (11, Buffer::NotOne),
            _ => return None,
        };
        Some(Kind {
            class,
            dimension,
            buffer,
            null: false,
        })
    }

    /// How `self` (the descriptor placed) disagrees with `declared`.
    pub(crate) fn mismatch(&self, declared: &Kind) -> Mismatch {
        if self.dimension == 0 && matches!(self.class, Class::Srv | Class::Uav) {
            // A view made from its resource alone: dimension unknown here.
            return if self.class == declared.class {
                Mismatch::None
            } else {
                Mismatch::Hard
            };
        }
        if self.class != declared.class {
            return Mismatch::Hard;
        }
        if self.dimension == declared.dimension {
            return if self.buffer == declared.buffer {
                Mismatch::None
            } else {
                // Typed, raw and structured buffers read through one another:
                // zeros on NVIDIA, not a TEX fault.
                Mismatch::Soft
            };
        }
        // Buffer against texture, or textures of other dimensions.
        let pair = |a: u8, b: u8| {
            (self.dimension == a && declared.dimension == b)
                || (self.dimension == b && declared.dimension == a)
        };
        if self.class == Class::Srv && (pair(2, 3) || pair(4, 5) || pair(9, 10) || pair(6, 7)) {
            // Array against single: called out, but not known to fault.
            return Mismatch::Soft;
        }
        if self.class == Class::Uav && (pair(2, 3) || pair(4, 5) || pair(6, 7)) {
            return Mismatch::Soft;
        }
        Mismatch::Hard
    }

    pub(crate) fn text(&self) -> String {
        let class = match self.class {
            Class::Srv => "srv",
            Class::Uav => "uav",
            Class::Cbv => return format!("cbv{}", if self.null { " null" } else { "" }),
            Class::Sampler => return "sampler".into(),
        };
        let dim = match self.dimension {
            0 => "?",
            1 => match self.buffer {
                Buffer::Raw => "raw buffer",
                Buffer::Structured => "structured buffer",
                _ => "typed buffer",
            },
            2 => "1d",
            3 => "1d array",
            4 => "2d",
            5 => "2d array",
            6 => "2d ms",
            7 => "2d ms array",
            8 => "3d",
            9 => "cube",
            10 => "cube array",
            11 => "accel",
            _ => "other",
        };
        format!("{class} {dim}{}", if self.null { " null" } else { "" })
    }
}

/// Logs buffer views past the host limit of 2^27 elements, which the
/// runtime passes to the driver without the debug layer.
fn over_limit(what: &str, elements: u32, first: u64) {
    if elements > 1 << 27 && enabled() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 8 || n.is_power_of_two() {
            tracing::warn!(
                n,
                what,
                elements,
                first,
                "descriptor check: a buffer view past 2^27 elements"
            );
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mismatch {
    None,
    /// Undefined, but not known to stop the device.
    Soft,
    /// Buffer against texture, another texture dimension, another class.
    Hard,
}

/// Per CPU handle: its kind, and the lowest host fence value a submission
/// that read it can have (0 if no checked draw read it).
#[derive(Clone, Copy)]
struct Slot {
    kind: Kind,
    read_by: u64,
}

static SLOTS: Mutex<Option<HashMap<usize, Slot>>> = Mutex::new(None);

/// Whether descriptors are checked (`VICTORY_CHECK_DESCRIPTORS`, diagnostics
/// builds).
pub(crate) fn enabled() -> bool {
    #[cfg(feature = "diagnostics")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            let on = crate::switch::on("VICTORY_CHECK_DESCRIPTORS");
            if on {
                tracing::warn!(
                    "VICTORY_CHECK_DESCRIPTORS: descriptor kinds checked at every draw and dispatch"
                );
            }
            on
        })
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        false
    }
}

/// Notes that `kind` was written at `at`. With a submission in flight that
/// a checked draw reading the slot belongs to, a change of kind there is
/// logged: the GPU may read the new descriptor in place of the old.
pub(crate) fn note(at: D3D12_CPU_DESCRIPTOR_HANDLE, kind: Kind) {
    if !enabled() {
        return;
    }
    let mut slots = SLOTS.lock().unwrap_or_else(|e| e.into_inner());
    let slots = slots.get_or_insert_with(HashMap::new);
    let old = slots.insert(at.ptr, Slot { kind, read_by: 0 });
    let Some(old) = old else { return };
    if old.read_by == 0 || old.kind == kind {
        // The same kind written again: whoever read the slot still reads
        // it, so a later change of kind is still theirs to hear of.
        if old.kind == kind
            && let Some(slot) = slots.get_mut(&at.ptr)
        {
            slot.read_by = old.read_by;
        }
        return;
    }
    // SAFETY: the host's own fence.
    let completed = crate::host::host().map_or(0, |h| unsafe { h.fence.GetCompletedValue() });
    let submitted = crate::host::submitted_value();
    if completed < old.read_by && old.read_by <= submitted {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 32 || n.is_power_of_two() {
            tracing::warn!(
                n,
                at = format_args!("{:#x}", at.ptr),
                was = old.kind.text(),
                now = kind.text(),
                read_by = old.read_by,
                completed,
                submitted,
                "descriptor check: a slot a submission in flight reads was rewritten with another kind"
            );
        }
    }
}

/// The kind at `at` (`default` if never noted), marking it read by a
/// submission whose host fence value is at least `read_by`.
pub(crate) fn read(at: D3D12_CPU_DESCRIPTOR_HANDLE, default: Kind, read_by: u64) -> Kind {
    let mut slots = SLOTS.lock().unwrap_or_else(|e| e.into_inner());
    let slots = slots.get_or_insert_with(HashMap::new);
    let slot = slots.entry(at.ptr).or_insert(Slot {
        kind: default,
        read_by: 0,
    });
    slot.read_by = slot.read_by.max(read_by);
    slot.kind
}

/// Forgets the slots of a heap from `start` for `bytes` (made null again or
/// released).
pub(crate) fn forget(start: usize, bytes: usize) {
    if !enabled() {
        return;
    }
    let mut slots = SLOTS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(slots) = slots.as_mut() {
        slots.retain(|&at, _| !(start..start + bytes).contains(&at));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_kinds_match_the_views_made_for_them() {
        let tex2d = Kind {
            class: Class::Srv,
            dimension: 4,
            buffer: Buffer::NotOne,
            null: false,
        };
        let declared = Kind::declared((2, 0), false).unwrap();
        assert_eq!(tex2d.mismatch(&declared), Mismatch::None);
        let tex3d = Kind::declared((4, 0), false).unwrap();
        assert_eq!(tex2d.mismatch(&tex3d), Mismatch::Hard);
        let array = Kind::declared((7, 0), false).unwrap();
        assert_eq!(tex2d.mismatch(&array), Mismatch::Soft);
        let raw = Kind::declared((11, 0), false).unwrap();
        assert_eq!(tex2d.mismatch(&raw), Mismatch::Hard);
        assert_eq!(Kind::INITIAL_VIEW.mismatch(&raw), Mismatch::Hard);
        let raw_uav = Kind::declared((11 | 0x1000, 0), false).unwrap();
        assert_eq!(raw_uav.class, Class::Uav);
        let raw_srv = Kind {
            class: Class::Srv,
            dimension: 1,
            buffer: Buffer::Raw,
            null: false,
        };
        assert_eq!(raw_srv.mismatch(&raw), Mismatch::None);
        assert_eq!(raw_srv.mismatch(&raw_uav), Mismatch::Hard);
        let typed = Kind::declared((10, 0x109), false).unwrap();
        assert_eq!(raw_srv.mismatch(&typed), Mismatch::Soft);
    }
}
