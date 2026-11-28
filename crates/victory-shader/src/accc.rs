//! Native compute programs stored in an `ACCC` part. Some containers carry
//! one in place of a `DXIL` part (seen: prebuilt compute passes shipped in a
//! DLL beside the title). Unlike `XDAT` blobs ([`crate::native`]) the part is
//! not compressed. Layout as observed:
//!
//! ```text
//! +0x00 "FSPX"  u32 version (1), u32 entry count, u32 part size
//! +0x10 entries of 16 B: u32 kind, u32 offset, u32 size, u32 0
//! kind 0  +0x0c u16 x3 thread-group size, u16 flags (bit 4 wave32, else
//!         wave64), +0x14 u32 COMPUTE_PGM_RSRC1, +0x18 u32 COMPUTE_PGM_RSRC2
//! kind 1  20 B, a hash then 3 (not read)
//! kind 6  GFX10 machine code; it opens with `v_nop` markers carrying the
//!         shader hash
//! kind 7  resource records: u32 0x403, u32 bytes after +0x14, ..., then
//!         from +0x18 records of 32 B: u32 tag, 20 B, u32 -1, u32 0
//!         tag 0x41  a used-resource entry, laid out as an `XDAT` record's
//!         tag 0x42  a view's dimension by register (not read)
//!         tag 0x43  the 16-byte shader hash
//! ```
//!
//! No root layout is stored. User-data dword 4 on holds the parameters of
//! the container's root signature (`RTS0`) in order: a table one dword (its
//! descriptors 8 dwords apart), a root descriptor two, constants their
//! count; every observed program's `RSRC2` user SGPR count ends exactly
//! there, and its used entries name those dwords. A layout that disagrees
//! with the user SGPR count is refused rather than guessed at.

use crate::ShaderError::{self, Malformed};
use crate::dxbc::{Container, FourCC, u16_at, u32_at};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo, check_resources,
    checked_compute, group_size, used_entry,
};

pub const ACCC: FourCC = FourCC(*b"ACCC");
const MAGIC: &[u8; 4] = b"FSPX";
const VERSION: u32 = 1;
const HEADER: usize = 0x10;
const ENTRY: usize = 16;
const MAX_ENTRIES: usize = 16;
const KIND_SETTINGS: u32 = 0;
const KIND_CODE: u32 = 6;
const KIND_RESOURCES: u32 = 7;
const SETTINGS_THREADS: usize = 0x0c;
const SETTINGS_FLAGS: usize = 0x12;
const FLAG_WAVE32: u16 = 1 << 4;
const SETTINGS_RSRC1: usize = 0x14;
const SETTINGS_RSRC2: usize = 0x18;
const RESOURCES_MAGIC: u32 = 0x403;
const RESOURCES_SIZE_BASE: usize = 0x14;
const RECORDS: usize = 0x18;
const RECORD: usize = 32;
const TAG_USED: u32 = 0x41;
/// Distinct used-resource records an `ACCC` part may list (far more than
/// any root signature's ranges name).
const MAX_USED: usize = 4096;
const TAG_DIMENSION: u32 = 0x42;
const TAG_HASH: u32 = 0x43;
/// The user-data dword the first root parameter arrives in; dwords 0-3 are
/// the system's (shortcut: what they hold is not known; no observed program
/// reads them).
const FIRST_USER_DATA: u32 = 4;
/// Dwords between two descriptors of a table, as the used entries state.
/// shortcut: sampler tables are assumed to step the same (none seen).
const DESCRIPTOR_DWORDS: u32 = 8;
const APPEND: u32 = u32::MAX;
const MAX_ROOT_PARAMS: u32 = 64;
const MAX_RANGES: u32 = 64;

/// The compute program of a container's `ACCC` part; empty without one.
pub fn accc_programs(c: &Container<'_>) -> Result<Vec<NativeProgram>, ShaderError> {
    let Some(part) = c.part(ACCC) else {
        return Ok(Vec::new());
    };
    let rts0 = c
        .part(FourCC::RTS0)
        .ok_or(Malformed("ACCC program without a root signature"))?;
    program(part.data, rts0.data).map(|p| vec![p])
}

fn program(part: &[u8], rts0: &[u8]) -> Result<NativeProgram, ShaderError> {
    if part.get(0..4) != Some(MAGIC) {
        return Err(Malformed("ACCC part lacks FSPX magic"));
    }
    if u32_at(part, 4)? != VERSION {
        return Err(Malformed("unknown FSPX version"));
    }
    let count = u32_at(part, 8)? as usize;
    if count > MAX_ENTRIES {
        return Err(Malformed("implausible FSPX entry count"));
    }
    if u32_at(part, 12)? as usize != part.len() {
        return Err(Malformed("FSPX size disagrees with its part"));
    }
    let entry = |kind: u32| -> Result<&[u8], ShaderError> {
        for i in 0..count {
            let at = HEADER + ENTRY * i;
            if u32_at(part, at)? != kind {
                continue;
            }
            let offset = u32_at(part, at + 4)? as usize;
            let size = u32_at(part, at + 8)? as usize;
            return offset
                .checked_add(size)
                .and_then(|end| part.get(offset..end))
                .ok_or(Malformed("FSPX entry out of bounds"));
        }
        Err(Malformed("FSPX entry missing"))
    };
    let settings = entry(KIND_SETTINGS)?;
    let code = entry(KIND_CODE)?;
    let resources = entry(KIND_RESOURCES)?;

    let wave64 = u16_at(settings, SETTINGS_FLAGS)? & FLAG_WAVE32 == 0;
    let rsrc1 = u32_at(settings, SETTINGS_RSRC1)?;
    let rsrc2 = u32_at(settings, SETTINGS_RSRC2)?;
    // `VGPRS` counts granules of 8 VGPRs in wave32, 4 in wave64, less one.
    let vgprs = ((rsrc1 & 0x3F) + 1) * if wave64 { 4 } else { 8 };
    if vgprs > 256 {
        return Err(Malformed("implausible VGPR count"));
    }
    let mut threads = [0; 3];
    for (k, n) in threads.iter_mut().enumerate() {
        *n = u16_at(settings, SETTINGS_THREADS + 2 * k)?.into();
    }
    let cs = ComputeIo {
        rsrc2,
        threads: group_size(threads)?,
    };
    let layout = root_layout(rts0, cs.user_sgprs())?;
    let used = used_resources(resources)?;
    check_resources(HwStage::Cs, &layout, &used)?;
    let cs = checked_compute(cs, &layout)?;
    Ok(NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: vgprs as u16,
        layout,
        used,
        io: StageIo::Cs(cs),
        code: code.to_vec(),
    })
}

fn used_resources(table: &[u8]) -> Result<Vec<crate::UsedResource>, ShaderError> {
    if u32_at(table, 0)? != RESOURCES_MAGIC {
        return Err(Malformed("unknown ACCC resource table"));
    }
    if u32_at(table, 4)? as usize + RESOURCES_SIZE_BASE != table.len() {
        return Err(Malformed("ACCC resource table size disagrees"));
    }
    let records = table.get(RECORDS..).unwrap_or_default();
    let (records, rest) = records.as_chunks::<RECORD>();
    if !rest.is_empty() {
        return Err(Malformed("ACCC resource table ends inside a record"));
    }
    let mut used = Vec::new();
    for r in records {
        match u32::from_le_bytes([r[0], r[1], r[2], r[3]]) {
            TAG_USED => {
                // The same entry again adds nothing (and the checks of
                // `check_resources` compare every pair).
                let entry = used_entry(r[4..24].try_into().expect("20 bytes"));
                if !used.contains(&entry) {
                    if used.len() == MAX_USED {
                        return Err(Malformed("implausible ACCC used-resource count"));
                    }
                    used.push(entry);
                }
            }
            TAG_DIMENSION | TAG_HASH => {}
            _ => return Err(Malformed("unknown ACCC resource record")),
        }
    }
    Ok(used)
}

/// The user SGPRs of each root parameter (one entry per table range), from
/// the root signature in the order the console assigns them.
fn root_layout(rts0: &[u8], user_sgprs: u8) -> Result<Vec<RootParam>, ShaderError> {
    let version = u32_at(rts0, 0)?;
    // Version 1.1 ranges add flags before the offset.
    let range_size = match version {
        1 => 20,
        2 => 24,
        _ => return Err(Malformed("unknown root signature version")),
    };
    let count = u32_at(rts0, 4)?;
    if count > MAX_ROOT_PARAMS {
        return Err(Malformed("implausible root parameter count"));
    }
    let params = u32_at(rts0, 8)? as usize;
    let mut layout = Vec::new();
    let mut user = FIRST_USER_DATA;
    for slot in 0..count {
        let at = params + 12 * slot as usize;
        let kind = u32_at(rts0, at)?;
        let data = u32_at(rts0, at + 8)? as usize;
        let user_data =
            u8::try_from(user).map_err(|_| Malformed("root parameter past user data"))?;
        let slot = slot as u8;
        match kind {
            // A descriptor table: one address, its ranges at their offsets.
            0 => {
                let ranges = u32_at(rts0, data)?;
                if ranges > MAX_RANGES {
                    return Err(Malformed("implausible descriptor range count"));
                }
                let first = u32_at(rts0, data + 4)? as usize;
                let mut next = Some(0u32);
                for k in 0..ranges as usize {
                    let r = first + range_size * k;
                    let num = u32_at(rts0, r + 4)?;
                    let base_register = u32_at(rts0, r + 8)?;
                    let offset = match u32_at(rts0, r + range_size - 4)? {
                        APPEND => next.ok_or(Malformed("range appended after an unbounded one"))?,
                        o => o,
                    };
                    next = (num != u32::MAX).then(|| offset.checked_add(num)).flatten();
                    layout.push(RootParam {
                        root: RootType::Table,
                        table_offset_dwords: offset
                            .checked_mul(DESCRIPTOR_DWORDS)
                            .ok_or(Malformed("implausible descriptor range offset"))?,
                        sgprs: 1,
                        base_register,
                        user_data,
                        slot,
                    });
                }
                user += 1;
            }
            // Root constants: register, space, count.
            1 => {
                let sgprs = u32_at(rts0, data + 8)?;
                layout.push(RootParam {
                    root: RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs,
                    base_register: u32_at(rts0, data)?,
                    user_data,
                    slot,
                });
                user = user.saturating_add(sgprs);
            }
            // A root CBV, SRV or UAV: a 64-bit address.
            2..=4 => {
                layout.push(RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: u32_at(rts0, data)?,
                    user_data,
                    slot,
                });
                user += 2;
            }
            _ => return Err(Malformed("unknown root parameter type")),
        }
    }
    if user != u32::from(user_sgprs) {
        return Err(Malformed(
            "ACCC root signature disagrees with the user SGPR count",
        ));
    }
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ResourceType, UsedResource};

    fn words(out: &mut Vec<u8>, w: &[u32]) {
        for v in w {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// A version 2 root signature: a UAV table of two ranges (u0-u1, then
    /// u5 appended), then a root CBV b0.
    fn root_signature() -> Vec<u8> {
        let mut r = Vec::new();
        // Header: version, 2 parameters at 0x18, no static samplers, flags.
        words(&mut r, &[2, 2, 0x18, 0, 0, 0]);
        // Parameters: table at 0x30, CBV at 0x60.
        words(&mut r, &[0, 0, 0x30, 2, 0, 0x60]);
        // Table: 2 ranges at 0x38.
        words(&mut r, &[2, 0x38]);
        words(&mut r, &[1, 2, 0, 0, 0, APPEND]);
        words(&mut r, &[1, 1, 5, 0, 0, APPEND]);
        // CBV b0, space 0, flags.
        words(&mut r, &[0, 0, 0]);
        r
    }

    fn record(out: &mut Vec<u8>, tag: u32, body: &[u8; 20]) {
        words(out, &[tag]);
        out.extend_from_slice(body);
        words(out, &[u32::MAX, 0]);
    }

    /// A used entry: kind and root type; table offset, lower bound and
    /// stride; descriptor dwords, user-data dword and slot.
    fn used_body([kind, root]: [u8; 2], w: [u32; 3], tail: [u8; 3]) -> [u8; 20] {
        let mut b = [0; 20];
        b[..2].copy_from_slice(&[kind, root]);
        for (k, v) in w.iter().enumerate() {
            b[4 + 4 * k..8 + 4 * k].copy_from_slice(&v.to_le_bytes());
        }
        b[16..19].copy_from_slice(&tail);
        b
    }

    /// An `ACCC` part: a 64-lane group of wave32 code `code`, user SGPRs
    /// `users`, reading u5 of the table and b0.
    fn part(code: &[u32], users: u32, flags: u16) -> Vec<u8> {
        let mut settings = Vec::new();
        words(&mut settings, &[0, 0, 0]);
        for v in [64u16, 1, 1, flags] {
            settings.extend_from_slice(&v.to_le_bytes());
        }
        // RSRC1: 16 VGPRs in wave32; RSRC2: user SGPRs, TGID_X_EN.
        words(&mut settings, &[1, users << 1 | 1 << 7]);
        let mut resources = Vec::new();
        let uav = used_body([3, 3], [0x10, 5, 4], [8, 4, 0]);
        let cbv = used_body([2, 1], [0, 0, 0], [4, 5, 1]);
        record(&mut resources, TAG_USED, &uav);
        record(&mut resources, TAG_USED, &cbv);
        record(&mut resources, TAG_DIMENSION, &[1; 20]);
        record(&mut resources, TAG_HASH, &[7; 20]);
        let mut table = Vec::new();
        words(
            &mut table,
            &[
                RESOURCES_MAGIC,
                (RECORDS + resources.len() - RESOURCES_SIZE_BASE) as u32,
                0,
                0,
                0,
                0,
            ],
        );
        table.extend_from_slice(&resources);
        let mut code_bytes = Vec::new();
        words(&mut code_bytes, code);
        let blobs = [
            (KIND_SETTINGS, settings),
            (1, vec![0; 20]),
            (KIND_CODE, code_bytes),
            (KIND_RESOURCES, table),
        ];
        let mut out = Vec::new();
        let mut at = HEADER + ENTRY * blobs.len();
        let total = at + blobs.iter().map(|(_, b)| b.len()).sum::<usize>();
        out.extend_from_slice(MAGIC);
        words(&mut out, &[VERSION, blobs.len() as u32, total as u32]);
        for (kind, b) in &blobs {
            words(&mut out, &[*kind, at as u32, b.len() as u32, 0]);
            at += b.len();
        }
        for (_, b) in &blobs {
            out.extend_from_slice(b);
        }
        out
    }

    fn container(accc: &[u8]) -> Vec<u8> {
        let rts0 = root_signature();
        crate::dxbc::build(&[(FourCC::RTS0, &rts0), (ACCC, accc)])
    }

    /// `s_endpgm`.
    const ENDPGM: u32 = 0xBF81_0000;

    /// A resource table of `bodies` as used-resource records.
    fn used_table(bodies: impl Iterator<Item = [u8; 20]>) -> Vec<u8> {
        let mut resources = Vec::new();
        for b in bodies {
            record(&mut resources, TAG_USED, &b);
        }
        let mut table = Vec::new();
        let size = (RECORDS + resources.len() - RESOURCES_SIZE_BASE) as u32;
        words(&mut table, &[RESOURCES_MAGIC, size, 0, 0, 0, 0]);
        table.extend_from_slice(&resources);
        table
    }

    #[test]
    fn repeated_used_records_count_once_and_too_many_are_refused() {
        let uav = used_body([3, 3], [0x10, 5, 4], [8, 4, 0]);
        let used = used_resources(&used_table(std::iter::repeat_n(uav, 50_000))).unwrap();
        assert_eq!(used.len(), 1);
        let distinct = (0..=MAX_USED as u32).map(|k| used_body([3, 3], [k, 5, 4], [8, 4, 0]));
        assert_eq!(
            used_resources(&used_table(distinct)).unwrap_err(),
            Malformed("implausible ACCC used-resource count")
        );
    }

    #[test]
    fn reads_a_compute_program() {
        let bytes = container(&part(&[ENDPGM], 7, FLAG_WAVE32));
        let c = Container::parse(&bytes).unwrap();
        let p = &accc_programs(&c).unwrap()[0];
        assert_eq!(p.stage, HwStage::Cs);
        assert!(!p.wave64);
        assert_eq!(p.vgprs, 16);
        assert_eq!(p.code, ENDPGM.to_le_bytes());
        let StageIo::Cs(cs) = p.io else {
            panic!("{:?}", p.io)
        };
        assert_eq!((cs.threads, cs.user_sgprs()), ([64, 1, 1], 7));
        let table = |table_offset_dwords, base_register| RootParam {
            root: RootType::Table,
            table_offset_dwords,
            sgprs: 1,
            base_register,
            user_data: 4,
            slot: 0,
        };
        assert_eq!(
            p.layout,
            [
                table(0, 0),
                table(16, 5),
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 0,
                    user_data: 5,
                    slot: 1,
                },
            ]
        );
        assert_eq!(
            p.used,
            [
                UsedResource {
                    kind: ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 16,
                    lower_bound: 5,
                    stride: 4,
                    dwords: 8,
                    user_data: 4,
                    slot: 0,
                },
                UsedResource {
                    kind: ResourceType::Cbv,
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 5,
                    slot: 1,
                },
            ]
        );
        // The same program through the crate's entry point.
        assert_eq!(
            crate::container_programs(&c).unwrap(),
            std::slice::from_ref(p)
        );
    }

    #[test]
    fn reads_wave64() {
        let bytes = container(&part(&[ENDPGM], 7, 0));
        let p = &accc_programs(&Container::parse(&bytes).unwrap()).unwrap()[0];
        assert!(p.wave64);
        assert_eq!(p.vgprs, 8);
    }

    #[test]
    fn refuses_what_it_cannot_place() {
        let refused =
            |accc: &[u8]| accc_programs(&Container::parse(&container(accc)).unwrap()).unwrap_err();
        // User SGPRs past the root signature's would be guessed at.
        assert_eq!(
            refused(&part(&[ENDPGM], 9, FLAG_WAVE32)),
            Malformed("ACCC root signature disagrees with the user SGPR count")
        );
        let mut bad = part(&[ENDPGM], 7, FLAG_WAVE32);
        bad[0] = b'X';
        assert_eq!(refused(&bad), Malformed("ACCC part lacks FSPX magic"));
        let mut short = part(&[ENDPGM], 7, FLAG_WAVE32);
        short.pop();
        assert_eq!(
            refused(&short),
            Malformed("FSPX size disagrees with its part")
        );
        // Without a root signature there is no layout.
        let alone = crate::dxbc::build(&[(ACCC, &part(&[ENDPGM], 7, FLAG_WAVE32))]);
        assert_eq!(
            accc_programs(&Container::parse(&alone).unwrap()).unwrap_err(),
            Malformed("ACCC program without a root signature")
        );
    }

    #[test]
    fn translates_through_the_pipeline() {
        let bytes = container(&part(&[ENDPGM], 7, FLAG_WAVE32));
        let shader = crate::emit::translate_with_wave32(&bytes, true).unwrap();
        assert_eq!(shader.stage, HwStage::Cs);
    }
}
