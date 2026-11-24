//! Console wave64 compute programs on a host of 32-lane waves, lowered doubled
//! ([`lower::with_doubled`]): one host thread runs console lanes `l` and `l + 32`, a host wave
//! is a console wave. Each program shares LDS or reads lanes in ways the two-host-waves form
//! cannot take (it keeps them at 64 lanes, which a 32-lane GPU refuses). Hand-assembled GFX10
//! words, lowered, validated by `dxil.dll` when present, and run on the host GPU when it takes
//! waves of 32; every result is compared with a CPU model of the console's 64-lane lockstep.

use std::collections::HashMap;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

// ---- a small GFX10 assembler ----

const VCC_LO: u32 = 106;
const VCC_HI: u32 = 107;
const EXEC: u32 = 126;
/// Inline integer constant `n` (0..=64) as a source operand.
const fn k(n: u32) -> u32 {
    128 + n
}
/// Inline -1.
const MINUS_ONE: u32 = 193;
/// VGPR `n` as a 9-bit source operand.
const fn v(n: u32) -> u32 {
    256 + n
}

const CMP_EQ_U32: u32 = 0xC2;
const CMP_LE_U32: u32 = 0xC3;
const CMP_GT_U32: u32 = 0xC4;
const CMP_NE_U32: u32 = 0xC5;
/// `v_cmp_* vcc, src0, v[vsrc1]`: vcc = src0 OP vsrc1.
const fn cmp(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x3E << 25 | op << 17 | vsrc1 << 9 | src0
}

const MOV: u32 = 1;
const READFIRSTLANE: u32 = 2;
/// VOP1: `vdst` (an SGPR for `v_readfirstlane_b32`), `src0`.
const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x3F << 25 | vdst << 17 | op << 9 | src0
}

const LSHRREV: u32 = 0x16;
const LSHLREV: u32 = 0x1A;
const AND: u32 = 0x1B;
const OR: u32 = 0x1C;
const XOR: u32 = 0x1D;
const ADD: u32 = 0x25;
/// VOP2: `v[vdst] = src0 OP v[vsrc1]` (shifts: `v[vsrc1]` shifted by `src0`).
const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}

const READLANE: u32 = 0x360;
const MBCNT_LO: u32 = 0x365;
const MBCNT_HI: u32 = 0x366;
/// VOP3 with two sources.
const fn vop3(op: u32, vdst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0x35 << 26 | op << 16 | vdst, src1 << 9 | src0]
}

const S_MOV_B64: u32 = 0x04;
const S_BCNT1_I32_B64: u32 = 0x10;
const S_AND_SAVEEXEC_B64: u32 = 0x24;
/// SOP1: `s[sdst] = OP ssrc0`.
const fn sop1(op: u32, sdst: u32, ssrc0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | ssrc0
}
const S_AND_B64: u32 = 0x0F;
/// SOP2: `s[sdst] = ssrc0 OP ssrc1`.
const fn sop2(op: u32, sdst: u32, ssrc0: u32, ssrc1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | ssrc1 << 8 | ssrc0
}
const S_WAITCNT_ALL: u32 = 0xBF8C_0000;
const S_BARRIER: u32 = 0xBF8A_0000;
const EXECZ: u32 = 0xBF88_0000;
const EXECNZ: u32 = 0xBF89_0000;

const DS_WRITE_B32: u32 = 13;
const DS_READ_B32: u32 = 54;
/// `ds_write_b32 v[addr], v[data]`.
const fn ds_write(addr: u32, data: u32) -> [u32; 2] {
    [0xD800_0000 | DS_WRITE_B32 << 18, data << 8 | addr]
}
/// `ds_read_b32 v[vdst], v[addr]`.
const fn ds_read(vdst: u32, addr: u32) -> [u32; 2] {
    [0xD800_0000 | DS_READ_B32 << 18, vdst << 24 | addr]
}
/// `buffer_store_dword v[vdata], v[vaddr], s[8:11], 0 offen offset:bytes`.
const fn store(vdata: u32, vaddr: u32, bytes: u32) -> [u32; 2] {
    [0x38 << 26 | 0x1C << 18 | 1 << 12 | bytes, 0x80 << 24 | 2 << 16 | vdata << 8 | vaddr]
}

/// Words with labelled branches (`s_cbranch_*` to a label; offsets patched at the end).
#[derive(Default)]
struct Asm {
    words: Vec<u32>,
    labels: HashMap<&'static str, usize>,
    fixups: Vec<(usize, &'static str)>,
}

impl Asm {
    fn op(&mut self, w: u32) -> &mut Self {
        self.words.push(w);
        self
    }
    fn op2(&mut self, w: [u32; 2]) -> &mut Self {
        self.words.extend_from_slice(&w);
        self
    }
    fn branch(&mut self, base: u32, to: &'static str) -> &mut Self {
        self.fixups.push((self.words.len(), to));
        self.words.push(base);
        self
    }
    fn label(&mut self, name: &'static str) -> &mut Self {
        self.labels.insert(name, self.words.len());
        self
    }
    fn done(&mut self) -> Vec<u32> {
        for &(at, to) in &self.fixups {
            let target = self.labels[to] as i64;
            let rel = target - (at as i64 + 1);
            self.words[at] |= u32::from(rel as i16 as u16);
        }
        std::mem::take(&mut self.words)
    }
}

// ---- programs, lowering, the card ----

/// A wave64 compute program of `threads` (x, y, z) threads, the thread index in v0 (and y in
/// v1 when `threads[1] > 1`), 4 root constants in s0..s3, a raw UAV's descriptor in s[8:11]
/// and 512 bytes of LDS, then `body`.
fn native(body: &[u32], threads: [u32; 3]) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(0xBF81_0000); // s_endpgm
    let tidig = u32::from(threads[1] > 1) + u32::from(threads[2] > 1);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            // 5 user SGPRs (s0..s4), thread id components, 512 bytes of LDS
            rsrc2: 5 << 1 | tidig << 11 | 1 << 15,
            threads,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The IR of `native`, which must hold each instruction named in `names` (the hand encoding
/// decodes as meant).
fn build(native: &NativeProgram, names: &[&str]) -> ir::Program {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    for n in names {
        assert!(
            p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == *n),
            "{n} not in the program: {:?}",
            p.blocks.iter().flat_map(|b| &b.insts).map(|i| i.name()).collect::<Vec<_>>()
        );
    }
    p
}

/// `p` lowered for a host of 32-lane waves: doubled (every lane branch wave-wide), or as the
/// plain per-lane form is. The shader and the wave size it asks of the host.
fn lowered(native: &NativeProgram, p: &ir::Program, doubled: bool) -> Result<(HostShader, Option<u32>), ShaderError> {
    let st = if doubled {
        super::all_wide(p).expect("every lane branch wave-wide")
    } else {
        structure::structure(p).expect("structure")
    };
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let parts = lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    };
    let l = if doubled {
        lower::with_doubled(|| lower::lower(&parts))
    } else {
        lower::lower(&parts)
    };
    lower::TEST_HOST_WAVE.set(was);
    let l = l?;
    Ok((
        HostShader {
            stage: HwStage::Cs,
            dxil: l.dxil,
            user_data: l.user_data,
            samplers: l.samplers,
            static_samplers: Vec::new(),
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            root_indexed_views: res.root_indexed_views,
        },
        l.wave_size,
    ))
}

/// The doubled form of `native`: asks for waves of 32, validates. With `refused`, the plain
/// form keeps 64 lanes or stops (the refusal the doubled form is for).
fn doubled(native: &NativeProgram, names: &[&str], refused: bool) -> HostShader {
    let p = build(native, names);
    if refused {
        let plain = lowered(native, &p, false);
        assert!(
            matches!(plain, Ok((_, Some(64)))) || plain.is_err(),
            "the plain form already runs on 32 lanes: {:?}",
            plain.map(|(_, w)| w)
        );
    }
    let (shader, wave) = lowered(native, &p, true).expect("doubled lowering");
    assert_eq!(wave, Some(32));
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// Runs `shader` on the hardware device, one thread group, root constants s0..s3, a zeroed
/// raw UAV of `dwords`; its contents, or None (and a line) without a device that takes it.
fn on_the_card(shader: &HostShader, constants: [u32; 4], dwords: u32) -> Option<Vec<u32>> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let (root, pso) = match host.compute_pipeline(shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program: {e}; not run");
            return None;
        }
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=3) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(dwords)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0].clone())
}

// ---- (a) LDS written under a guard, read by every lane in the same lane branch ----

/// Inside `if (t < s1)`: lane 0 writes s0 to LDS dword 0 under its own guard, every lane reads
/// it; then lane 32 (the other half) writes s2 there, every lane reads again. Lane t stores
/// `first | second << 16`. On the console the 64 lanes run in lockstep, so the first read sees
/// s0 and the second s2 (when lane 32 is in the branch), with no barrier. The `2cc4c197`
/// pattern (`sr_tracing:CS_Tiling_Generate_Opaque`).
#[test]
fn lds_written_by_one_lane_is_read_by_all_in_the_same_lane_branch() {
    let words = Asm::default()
        .op(vop2(LSHLREV, 5, k(2), 0)) // v5 = t * 4
        .op(vop1(MOV, 2, k(0))) // v2 = 0 (LDS address)
        .op(cmp(CMP_GT_U32, 1, 0)) // vcc = s1 > t
        .op(sop1(S_AND_SAVEEXEC_B64, 14, VCC_LO))
        .branch(EXECZ, "end")
        .op(cmp(CMP_EQ_U32, k(0), 0)) // lane 0
        .op(sop1(S_AND_SAVEEXEC_B64, 12, VCC_LO))
        .branch(EXECZ, "skip1")
        .op(vop1(MOV, 1, 0)) // v1 = s0
        .op2(ds_write(2, 1))
        .label("skip1")
        .op(sop1(S_MOV_B64, EXEC, 12))
        .op2(ds_read(3, 2))
        .op(S_WAITCNT_ALL)
        .op(cmp(CMP_EQ_U32, k(32), 0)) // lane 32
        .op(sop1(S_AND_SAVEEXEC_B64, 12, VCC_LO))
        .branch(EXECZ, "skip2")
        .op(vop1(MOV, 1, 2)) // v1 = s2
        .op2(ds_write(2, 1))
        .label("skip2")
        .op(sop1(S_MOV_B64, EXEC, 12))
        .op2(ds_read(4, 2))
        .op(S_WAITCNT_ALL)
        .op(vop2(LSHLREV, 4, k(16), 4))
        .op(vop2(OR, 3, v(3), 4))
        .op2(store(3, 5, 0))
        .label("end")
        .op(sop1(S_MOV_B64, EXEC, 14))
        .done();
    let native = native(&words, [64, 1, 1]);
    let shader = doubled(&native, &["ds_write_b32", "ds_read_b32", "s_and_saveexec_b64"], true);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let (s0, s2) = (0x11, 0x22);
    for s1 in [64u32, 40, 33, 32, 1, 0] {
        let Some(out) = on_the_card(&shader, [s0, s1, s2, 0], 64) else { return };
        // the console: lockstep over the 64 lanes
        let second = if 32 < s1 { s2 } else { s0 };
        let want: Vec<u32> = (0..64).map(|t| if t < s1 { s0 | second << 16 } else { 0 }).collect();
        assert_eq!(out, want, "{s1} lanes in the branch");
    }
    eprintln!("ran on the hardware device: every lane read both writes in console order");
}

// ---- (b) lane reads across the halves ----

/// Inside `if (t < s2)`: `v_readlane` of lanes s0 and s1 (either half) of v1 = 3 t + 5, and
/// every lane stores `lane s0's | lane s1's << 16`.
#[test]
fn a_lane_read_takes_the_lane_from_either_half() {
    let words = Asm::default()
        .op(vop2(LSHLREV, 5, k(2), 0))
        .op(vop2(ADD, 1, v(0), 0)) // 2t
        .op(vop2(ADD, 1, v(1), 0)) // 3t
        .op(vop2(ADD, 1, k(5), 1)) // 3t + 5
        .op(cmp(CMP_GT_U32, 2, 0)) // vcc = s2 > t
        .op(sop1(S_AND_SAVEEXEC_B64, 14, VCC_LO))
        .branch(EXECZ, "end")
        .op2(vop3(READLANE, 20, v(1), 0)) // s20 = v1 of lane s0
        .op2(vop3(READLANE, 21, v(1), 1)) // s21 = v1 of lane s1
        .op(vop1(MOV, 2, 21))
        .op(vop2(LSHLREV, 2, k(16), 2))
        .op(vop2(OR, 2, 20, 2))
        .op2(store(2, 5, 0))
        .label("end")
        .op(sop1(S_MOV_B64, EXEC, 14))
        .done();
    let native = native(&words, [64, 1, 1]);
    let shader = doubled(&native, &["v_readlane_b32"], true);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (s0, s1, s2) in [(0u32, 63u32, 64u32), (5, 40, 64), (31, 32, 20), (45, 1, 50), (63, 0, 1)] {
        let Some(out) = on_the_card(&shader, [s0, s1, s2, 0], 64) else { return };
        let at = |l: u32| 3 * l + 5;
        let want: Vec<u32> = (0..64).map(|t| if t < s2 { at(s0) | at(s1) << 16 } else { 0 }).collect();
        assert_eq!(out, want, "lanes {s0}, {s1}, {s2} in the branch");
    }
    eprintln!("ran on the hardware device: lane reads of either half");
}

/// Inside `if (t >= s0)`: `v_readfirstlane` of v1 = 3 t + 5 is lane s0's, in either half.
#[test]
fn a_first_lane_read_finds_the_first_lane_in_either_half() {
    let words = Asm::default()
        .op(vop2(LSHLREV, 5, k(2), 0))
        .op(vop2(ADD, 1, v(0), 0))
        .op(vop2(ADD, 1, v(1), 0))
        .op(vop2(ADD, 1, k(5), 1)) // v1 = 3t + 5
        .op(cmp(CMP_LE_U32, 0, 0)) // vcc = s0 <= t
        .op(sop1(S_AND_SAVEEXEC_B64, 14, VCC_LO))
        .branch(EXECZ, "end")
        .op(vop1(READFIRSTLANE, 20, v(1)))
        .op(vop1(MOV, 2, 20))
        .op2(store(2, 5, 0))
        .label("end")
        .op(sop1(S_MOV_B64, EXEC, 14))
        .done();
    let native = native(&words, [64, 1, 1]);
    // (the two-host-waves form takes this one too, through its exchange)
    let shader = doubled(&native, &["v_readfirstlane_b32"], false);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for s0 in [0u32, 5, 31, 32, 40, 63] {
        let Some(out) = on_the_card(&shader, [s0, 0, 0, 0], 64) else { return };
        let want: Vec<u32> = (0..64).map(|t| if t >= s0 { 3 * s0 + 5 } else { 0 }).collect();
        assert_eq!(out, want, "first lane {s0}");
    }
    eprintln!("ran on the hardware device: the first lane of EXEC in either half");
}

// ---- (c) v_mbcnt and a mask read as data ----

/// Inside `if (t < s1)`: vcc = lanes t >= s0 (of those in the branch: [s0, s1)). Lane t
/// stores `v_mbcnt` of vcc (lanes of vcc below it) plus `s_bcnt1(vcc) << 8`, then vcc's low
/// and high dwords as data.
#[test]
fn mbcnt_and_a_mask_as_data_count_all_64_lanes() {
    let words = Asm::default()
        .op(vop2(LSHLREV, 5, k(2), 0))
        .op(cmp(CMP_GT_U32, 1, 0)) // vcc = s1 > t
        .op(sop1(S_AND_SAVEEXEC_B64, 14, VCC_LO))
        .branch(EXECZ, "end")
        .op(cmp(CMP_LE_U32, 0, 0)) // vcc = s0 <= t (lanes outside EXEC: 0)
        .op2(vop3(MBCNT_LO, 2, VCC_LO, k(0)))
        .op2(vop3(MBCNT_HI, 2, VCC_HI, v(2)))
        .op(sop1(S_BCNT1_I32_B64, 20, VCC_LO))
        .op(vop1(MOV, 3, 20))
        .op(vop2(LSHLREV, 3, k(8), 3))
        .op(vop2(OR, 2, v(2), 3))
        .op2(store(2, 5, 0))
        .op(vop1(MOV, 3, VCC_LO))
        .op2(store(3, 5, 256))
        .op(vop1(MOV, 3, VCC_HI))
        .op2(store(3, 5, 512))
        .label("end")
        .op(sop1(S_MOV_B64, EXEC, 14))
        .done();
    let native = native(&words, [64, 1, 1]);
    let shader = doubled(&native, &["v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32", "s_bcnt1_i32_b64"], true);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (s0, s1) in [(0u32, 64u32), (1, 64), (17, 40), (32, 64), (33, 50), (10, 33), (63, 64), (5, 5)] {
        let Some(out) = on_the_card(&shader, [s0, s1, 0, 0], 192) else { return };
        let lanes = |n: u32| if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
        let mask: u64 = lanes(s1) & !lanes(s0);
        let mut want = vec![0u32; 192];
        for t in (0..64u32).filter(|&t| t < s1) {
            let below = (mask & lanes(t)).count_ones();
            want[t as usize] = below | mask.count_ones() << 8;
            want[64 + t as usize] = mask as u32;
            want[128 + t as usize] = (mask >> 32) as u32;
        }
        assert_eq!(out, want, "lanes [{s0}, {s1})");
    }
    eprintln!("ran on the hardware device: mbcnt, bit count and mask dwords of all 64 lanes");
}

// ---- (d) a loop whose trip count differs per half, with LDS traffic ----

/// LDS dword t = t. Lane t loops `(t & 3) + 2 (t >> 5) + 1` times (half 1 longer); each
/// iteration it reads the dword of lane `t ^ 33` (the other half) and writes it plus one to
/// its own, the reads of an iteration all before its writes (lockstep). Lane t stores its
/// dword after the loop.
#[test]
fn a_loop_of_different_lengths_per_half_shares_lds_in_lockstep() {
    let words = Asm::default()
        .op(vop2(LSHLREV, 5, k(2), 0)) // own address
        .op(vop2(XOR, 4, k(33), 0))
        .op(vop2(LSHLREV, 4, k(2), 4)) // partner's address
        .op2(ds_write(5, 0)) // LDS[t] = t
        .op(S_WAITCNT_ALL)
        .op(vop2(AND, 6, k(3), 0))
        .op(vop2(LSHRREV, 7, k(5), 0))
        .op(vop2(LSHLREV, 7, k(1), 7))
        .op(vop2(ADD, 6, v(6), 7))
        .op(vop2(ADD, 6, k(1), 6)) // n
        .op(sop1(S_MOV_B64, 16, EXEC))
        .label("loop")
        .op2(ds_read(3, 4))
        .op(S_WAITCNT_ALL)
        .op(vop2(ADD, 3, k(1), 3))
        .op2(ds_write(5, 3))
        .op(vop2(ADD, 6, MINUS_ONE, 6))
        .op(cmp(CMP_NE_U32, k(0), 6))
        .op(sop2(S_AND_B64, EXEC, EXEC, VCC_LO))
        .branch(EXECNZ, "loop")
        .op(sop1(S_MOV_B64, EXEC, 16))
        .op2(ds_read(3, 5))
        .op(S_WAITCNT_ALL)
        .op2(store(3, 5, 0))
        .done();
    let native = native(&words, [64, 1, 1]);
    // (the two-host-waves form takes this one too, meeting around the loop: run below as well)
    let shader = doubled(&native, &["s_and_b64", "s_cbranch_execnz", "ds_read_b32", "ds_write_b32"], false);
    // the console: 64 lanes in lockstep
    let mut lds: Vec<u32> = (0..64).collect();
    let mut n: Vec<u32> = (0..64).map(|t| (t & 3) + 2 * (t >> 5) + 1).collect();
    let mut active = [true; 64];
    loop {
        let read: Vec<u32> = (0..64).map(|t| lds[t ^ 33]).collect();
        for t in 0..64 {
            if active[t] {
                lds[t] = read[t] + 1;
                n[t] -= 1;
                active[t] = n[t] != 0;
            }
        }
        if !active.contains(&true) {
            break;
        }
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(out) = on_the_card(&shader, [0; 4], 64) else { return };
    assert_eq!(out, lds);
    eprintln!("ran on the hardware device: the loop's LDS traffic in console order");
    // For the record: the two-host-waves form of the same program (not asserted).
    let p = build(&native, &[]);
    if let Ok((plain, Some(32))) = lowered(&native, &p, false)
        && let Some(out) = on_the_card(&plain, [0; 4], 64)
    {
        let wrong = out.iter().zip(&lds).filter(|(a, b)| a != b).count();
        eprintln!("the two-host-waves form: {wrong} of 64 lanes differ from the console");
    }
}

// ---- (f) a group of several console waves with the title's barrier ----

/// 256 threads, 4 console waves. In each wave, lane s0 writes `s1 + t` to LDS dword w (its
/// wave); `s_barrier`; every lane reads dword `(w + 1) & 3` and stores it with `t << 16`.
#[test]
fn several_console_waves_meet_at_the_titles_barrier() {
    let words = Asm::default()
        .op(vop2(LSHLREV, 5, k(2), 0))
        .op(vop2(LSHRREV, 6, k(6), 0)) // w
        .op(vop2(LSHLREV, 7, k(2), 6))
        .op(vop2(AND, 8, k(63), 0)) // lane in the wave
        .op(cmp(CMP_EQ_U32, 0, 8)) // vcc = s0 == lane
        .op(sop1(S_AND_SAVEEXEC_B64, 12, VCC_LO))
        .branch(EXECZ, "skip")
        .op(vop2(ADD, 9, 1, 0)) // s1 + t
        .op2(ds_write(7, 9))
        .label("skip")
        .op(sop1(S_MOV_B64, EXEC, 12))
        .op(S_WAITCNT_ALL)
        .op(S_BARRIER)
        .op(vop2(ADD, 6, k(1), 6))
        .op(vop2(AND, 6, k(3), 6))
        .op(vop2(LSHLREV, 7, k(2), 6))
        .op2(ds_read(3, 7))
        .op(S_WAITCNT_ALL)
        .op(vop2(LSHLREV, 4, k(16), 0))
        .op(vop2(OR, 3, v(3), 4))
        .op2(store(3, 5, 0))
        .done();
    let native = native(&words, [256, 1, 1]);
    let shader = doubled(&native, &["s_barrier", "ds_write_b32"], false);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let s1 = 1000;
    for s0 in [0u32, 33, 63] {
        let Some(out) = on_the_card(&shader, [s0, s1, 0, 0], 256) else { return };
        let want: Vec<u32> = (0..256u32)
            .map(|t| {
                let next = ((t >> 6) + 1) & 3;
                (s1 + 64 * next + s0) | t << 16
            })
            .collect();
        assert_eq!(out, want, "lane {s0} of each wave writes");
    }
    eprintln!("ran on the hardware device: 4 console waves, the barrier, every thread id once");
}

/// A 16 x 16 group (4 console waves): each thread stores `x | y << 8` at its flat index, so
/// the thread ids of the halved host group come out as the console's.
#[test]
fn a_two_dimensional_group_keeps_the_consoles_thread_ids() {
    let words = Asm::default()
        .op(vop2(LSHLREV, 5, k(4), 1)) // 16 y
        .op(vop2(ADD, 5, v(5), 0)) // flat = x + 16 y
        .op(vop2(LSHLREV, 5, k(2), 5))
        .op(vop2(LSHLREV, 3, k(8), 1))
        .op(vop2(OR, 3, v(3), 0)) // x | y << 8
        // a lane read so that the plain form keeps 64 lanes
        .op2(vop3(READLANE, 20, v(0), 0))
        .op(vop1(MOV, 4, 20))
        .op2(store(3, 5, 0))
        .op(vop2(ADD, 5, 0xFF, 5)) // (literal follows)
        .op(1024)
        .op2(store(4, 5, 0))
        .done();
    let native = native(&words, [16, 16, 1]);
    let shader = doubled(&native, &["v_readlane_b32"], false);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for s0 in [3u32, 40] {
        let Some(out) = on_the_card(&shader, [s0, 0, 0, 0], 512) else { return };
        let mut want = vec![0u32; 512];
        for y in 0..16u32 {
            for x in 0..16u32 {
                let flat = (x + 16 * y) as usize;
                want[flat] = x | y << 8;
                // lane s0 of this thread's console wave: its x (waves are 4 rows of 16)
                let lane_flat = (flat as u32 & !63) + s0;
                want[256 + flat] = lane_flat % 16;
            }
        }
        assert_eq!(out, want, "lane {s0}");
    }
    eprintln!("ran on the hardware device: the console's (x, y) in a halved group");
}
