//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, and turn-range reduction for `v_sin_*`/`v_cos_*`.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// VOP3 with output modifier (`omod`: 0 none, 1 mul:2, 2 mul:4, 3 div:2).
const fn vop3_omod(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16, omod: u8) -> [u32; 2] {
    let [w0, w1] = vop3(op, vdst, src0, src1, src2);
    [w0, w1 | (omod as u32 & 3) << 27]
}

/// VOP3 with integer/float `clamp` (unsigned saturation on integer add/sub).
const fn vop3_clamp(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    let [w0, w1] = vop3(op, vdst, src0, src1, src2);
    [w0 | 1 << 15, w1]
}

/// VOP3B (carry-out `sdst`) with `clamp`.
const fn vop3b_clamp(op: u16, vdst: u8, sdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | 1 << 15 | (sdst as u32 & 0x7F) << 8 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3P `op vdst, src0, src1, src2` (no modifiers; all f32 sources).
const fn vop3p(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x33 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    let shader = try_build_valu_cs(body).expect("lower");

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// [`build_valu_cs`] without the validation, returning a refusal instead of
/// panicking on it.
fn try_build_valu_cs(body: &[u32]) -> Result<HostShader, ShaderError> {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
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
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })?;

    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
    })
}

/// `s20` is above the 5 SGPRs the hardware initializes here and is written
/// only on the path the branch skips, so its read after the join is a path
/// artefact and reads zero.
#[test]
fn an_sgpr_written_on_another_path_reads_zero() {
    let body = [
        s_mov_b32(16, 0x80),      // s16 = 0
        0xBF06_8010,              // s_cmp_eq_u32 s16, 0
        0xBF85_0001,              // s_cbranch_scc1 past the next instruction
        s_mov_b32(20, 0x87),      // s20 = 7
        vop1(0x001, 0, sgpr(20)), // v_mov_b32 v0, s20
    ];
    build_valu_cs(&body);
}

/// An SGPR above the initialized ones that no instruction writes is more
/// likely an input the decoding misses than a path artefact, so it stays
/// refused.
#[test]
fn an_sgpr_no_instruction_writes_is_refused() {
    let body = [vop1(0x001, 0, sgpr(20))]; // v_mov_b32 v0, s20
    let Err(e) = try_build_valu_cs(&body) else {
        panic!("translated");
    };
    assert!(e.to_string().contains("no instruction writes"), "{e}");
}

/// One GPU test at a time. With these tests on the hardware adapter and WARP
/// from several test threads at once, now and then none of their dispatches
/// ever completes and the test binary hangs; and two GPU tests at once on
/// WARP can end the test process with an access violation. Other GPU test
/// modules take this lock too.
pub(super) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// `Host::new(false)` also selects WARP when the environment forces it.
/// Do not run that backend twice or label software results as hardware.
pub(super) fn backends(force_warp: bool) -> &'static [(&'static str, bool)] {
    if force_warp {
        &[("warp", true)]
    } else {
        &[("hardware", false), ("warp", true)]
    }
}

#[test]
fn forced_warp_runs_only_the_software_backend() {
    assert_eq!(backends(true), &[("warp", true)]);
    assert_eq!(backends(false), &[("hardware", false), ("warp", true)]);
}

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP,
/// or WARP alone with `VICTORY_WARP=1`) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let force_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    run_cases_on(shader, cases, backends(force_warp), check);
}

/// [`run_cases`] on the given backends only (WARP cannot pin a wave size,
/// which the lane-aware programs ask for).
fn run_cases_on(
    shader: &HostShader,
    cases: &[[u32; 4]],
    backends: &[(&str, bool)],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let (mut ran, mut no_wave) = (0, 0);
    for &(label, warp) in backends {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(made) => made,
            // The shader pins the console's wave size; a device without it
            // cannot build the pipeline.
            Err(e) if !super::lane_mask_tests::runs_wave32(&host) => {
                eprintln!("{label} device has no 32-lane wave ({e}); skipped");
                no_wave += 1;
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    assert!(
        ran + no_wave > 0,
        "neither hardware nor WARP SM 6.6 device available"
    );
}

#[test]
fn scalar_quad_masks_pack_nibbles_and_set_scc() {
    let pack_words = |opcode: u32| {
        [
            0xBE90_0000 | opcode << 8, // s_quadmask s16, s0 (pairs for b64)
            sop2(0x0a, 18, 129, 128),  // s_cselect_b32 s18, 1, 0
            sop2(0x1e, 18, 18, 144),   // s_lshl_b32 s18, s18, 16
            sop2(0x10, 16, 16, 18),    // s_or_b32 s16, s16, s18
            vop1(1, 0, sgpr(16)),      // v_mov_b32 v0, s16
        ]
    };
    let cases = [
        ([0, 0, 0, 0], [0, 0]),
        ([0x8000_0420, 0x0100_0001, 0, 0], [0x1_0086, 0x1_4186]),
        ([0, 0x8000_0000, 0, 0], [0, 0x1_8000]),
        ([u32::MAX, u32::MAX, 0, 0], [0x1_00ff, 0x1_ffff]),
    ];
    let inputs = cases.map(|(input, _)| input);
    for (wide, opcode) in [0x2c, 0x2d].into_iter().enumerate() {
        let shader = build_valu_cs(&pack_words(opcode));
        run_cases(&shader, &inputs, |backend, i, _, got| {
            assert_eq!(got, cases[i].1[wide], "{backend}, opcode {opcode:#x}");
        });
    }
    let high = build_valu_cs(&[
        0xBE90_2D00, // s_quadmask_b64 s[16:17], s[0:1]
        vop1(1, 0, sgpr(17)),
    ]);
    run_cases(&high, &inputs, |backend, _, _, got| {
        assert_eq!(got, 0, "{backend}: high dword");
    });
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

#[test]
fn a_branch_to_fallthrough_keeps_both_values_at_a_join() {
    let shader = build_valu_cs(&[
        0xBF06_0100,             // s_cmp_eq_u32 s0, s1
        0xBF85_0002,             // s_cbranch_scc1 -> the s2 = 1 path
        s_mov_b32(2, 128),       // s2 = 0
        0xBF82_0002,             // s_branch -> the join
        s_mov_b32(2, 129),       // s2 = 1
        0xBF85_0000,             // s_cbranch_scc1 -> the following instruction
        vop1(0x001, 0, sgpr(2)), // join: v0 = s2
    ]);
    run_cases(
        &shader,
        &[[7, 7, 0, 0], [7, 5, 0, 0]],
        |backend, _, inputs, actual| {
            assert_eq!(actual, u32::from(inputs[0] == inputs[1]), "{backend}");
        },
    );
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

/// MTBUF `op vdata, off, s[srsrc:srsrc+3], 0 format:format` (at byte 0).
const fn mtbuf(op: u8, vdata: u8, srsrc: u8, format: u8) -> [u32; 2] {
    [
        0x3A << 26 | (format as u32 & 0x7F) << 19 | (op as u32 & 7) << 16,
        0x80 << 24
            | (op as u32 >> 3 & 1) << 21
            | (srsrc as u32 >> 2 & 0x1F) << 16
            | (vdata as u32) << 8,
    ]
}

/// Stores `v1` to the buffer, then reads it back into `v0` with the tbuffer
/// load `op` in buffer format `format`.
fn tbuffer_load_of_v1(op: u8, format: u8) -> HostShader {
    let mut body = buffer_store_dword(1, 8, 0).to_vec();
    body.extend(mtbuf(op, 0, 8, format));
    build_valu_cs(&body)
}

#[test]
fn tbuffer_d16_loads_pack_two_channels_per_dword() {
    // tbuffer_load_format_d16_x and tbuffer_load_format_d16_xy.
    let (x, xy) = (8, 9);
    // (opcode, format, dword stored, dword loaded)
    let cases: &[(u8, u8, u32, u32)] = &[
        // 16_16_FLOAT: the f16 pair (1.0, 2.0) as it was stored.
        (xy, 29, 0x4000_3C00, 0x4000_3C00),
        // 16_16_UINT: the 16-bit integers as they were stored.
        (xy, 27, 0xBEEF_1234, 0xBEEF_1234),
        // 16_16_UNORM: 0.0 and 1.0, as f16.
        (xy, 23, 0xFFFF_0000, 0x3C00_0000),
        // 16_FLOAT, one channel: v0 starts clear, so the high half is 0.
        (x, 13, 0x4000_3C00, 0x0000_3C00),
    ];
    for &(op, format, stored, want) in cases {
        let shader = tbuffer_load_of_v1(op, format);
        run_cases(&shader, &[[stored, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                got, want,
                "{backend}: d16 opcode {op} format {format} of {stored:#010x} = {got:#010x}, want {want:#010x}",
            );
        });
    }
}

#[test]
fn tbuffer_d16_loads_saturate_32_bit_integers() {
    // VIC-BT-015: tbuffer_load_format_d16_x of a 32-bit integer saturates to
    // 16 bits (UINT to 0xFFFF, SINT to -32768..32767) rather than converting
    // through f16 or keeping the low half. Only the low half is compared:
    // what an odd last channel leaves in the high half is separate.
    let x = 8;
    let cases: &[(u8, u32, u32)] = &[
        (20, 70000, 0xFFFF),              // 32_UINT: past 0xFFFF
        (20, 5, 5),                       // in range: as stored
        (21, 70000, 0x7FFF),              // 32_SINT: past 32767
        (21, (-65536i32) as u32, 0x8000), // below -32768
        (21, (-5i32) as u32, 0xFFFB),     // in range: the low half
    ];
    for &(format, stored, want) in cases {
        let shader = tbuffer_load_of_v1(x, format);
        run_cases(&shader, &[[stored, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                got & 0xFFFF,
                want,
                "{backend}: d16_x format {format} of {stored:#x} = {got:#010x}, want low half {want:#06x}"
            );
        });
    }
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }
}

/// The f16 forms go through the same `max_min` helper, after widening to f32.
#[test]
fn f16_max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_u32);
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f16", 0x139, pz),
        ("v_min_f16", 0x13A, nz),
        ("v_max3_f16", 0x354, pz),
        ("v_min3_f16", 0x351, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            // The f16 result is the low half of v0.
            assert_eq!(got & 0xFFFF, want, "{backend}: {name} {inp:x?}");
        });
    }
}

#[test]
fn v_min3_max3_med3_integer_forms_match_rdna2() {
    // v_{min3,max3,med3}_{i32,u32} v0, v1, v2, v3 (VOP3 0x152..0x159).
    /// Name, VOP3 opcode and the RDNA2 result for three operands.
    type Op = (&'static str, u16, fn([u32; 3]) -> u32);
    let ops: [Op; 6] = [
        ("v_min3_i32", 0x152, |t| {
            t.into_iter().min_by_key(|&x| x as i32).unwrap()
        }),
        ("v_min3_u32", 0x153, |t| t.into_iter().min().unwrap()),
        ("v_max3_i32", 0x155, |t| {
            t.into_iter().max_by_key(|&x| x as i32).unwrap()
        }),
        ("v_max3_u32", 0x156, |t| t.into_iter().max().unwrap()),
        ("v_med3_i32", 0x158, |mut t| {
            t.sort_by_key(|&x| x as i32);
            t[1]
        }),
        ("v_med3_u32", 0x159, |mut t| {
            t.sort_unstable();
            t[1]
        }),
    ];
    // Every ordering of three values, with the sign bit set on some so that
    // the signed and unsigned answers differ.
    let values = [0u32, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF];
    let mut cases = Vec::new();
    for a in values {
        for b in values {
            for c in values {
                cases.push([a, b, c, 0]);
            }
        }
    }
    for (name, op, pick) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let want = pick([inp[0], inp[1], inp[2]]);
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1], inp[2],
            );
        });
    }
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// The f16 nearest to `x`, ties to even, as the console's conversions
/// round (MODE's default); found by search over the f16 values, so it does
/// not share the lowering's bit arithmetic.
fn f16_nearest_reference(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7E00;
    }
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = f64::from(x.abs());
    if a >= 65520.0 {
        return sign | 0x7C00;
    }
    // The largest finite half at or below `a` (halves order as their bits).
    let (mut lo, mut hi) = (0u16, 0x7BFF);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if f64::from(f16_to_f32(mid)) <= a {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let below = f64::from(f16_to_f32(lo));
    let above = if lo == 0x7BFF {
        65536.0
    } else {
        f64::from(f16_to_f32(lo + 1))
    };
    let pick = if a - below < above - a || (a - below == above - a && lo % 2 == 0) {
        lo
    } else {
        lo + 1
    };
    sign | pick
}

#[test]
fn f32_to_f16_rounds_to_nearest_even() {
    // Midpoints between neighbouring halves (ties), the f32 next to them, and
    // a spread of f32 values from below the smallest subnormal to past 65504.
    let mut inputs: Vec<f32> = Vec::new();
    for h in (0x0000u16..0x7BFF)
        .step_by(613)
        .chain([0x0000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x3C01, 0x7BFE])
    {
        let mid = ((f64::from(f16_to_f32(h)) + f64::from(f16_to_f32(h + 1))) / 2.0) as f32;
        inputs.extend([
            mid,
            -mid,
            f32::from_bits(mid.to_bits() + 1),
            f32::from_bits(mid.to_bits() - 1),
        ]);
    }
    let mut bits = 0x3280_0000u32; // 2^-26
    while bits < 0x4790_0000 {
        inputs.push(f32::from_bits(bits));
        bits += 0x0013_3F7D;
    }
    inputs.extend([
        65504.0,
        65519.99,
        65520.0,
        1.0e6,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -0.0,
        2.0f32.powi(-25),
        f32::MIN_POSITIVE,
    ]);
    // v_mov_b32 v0, 0; v_cvt_f16_f32 v0, v1 (VOP1 0x00a).
    let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(0x00A, 0, vgpr(1))]);
    let cases: Vec<[u32; 4]> = inputs.iter().map(|x| [x.to_bits(), 0, 0, 0]).collect();
    run_cases(&shader, &cases, |backend, i, _, got| {
        let (x, want) = (inputs[i], f16_nearest_reference(inputs[i]));
        let got_h = got as u16;
        if is_f16_nan(want) {
            assert!(
                is_f16_nan(got_h),
                "{backend}: v_cvt_f16_f32({x:e}) = {got_h:#06x}, want NaN"
            );
        } else {
            assert_eq!(
                got_h,
                want,
                "{backend}: v_cvt_f16_f32({x:e} = {:#010x}) = {got_h:#06x}, want {want:#06x}",
                x.to_bits()
            );
        }
    });
    // f16 arithmetic narrows the same way: 1 + 2^-10 plus 2^-11 is a tie
    // that goes to the even 1 + 2^-9 (toward zero would give 1 + 2^-10).
    let mut body = vec![vop1(0x001, 0, 128)];
    body.extend_from_slice(&vop3(0x132, 0, vgpr(1), vgpr(2), 0)); // v_add_f16 v0, v1, v2
    let add = build_valu_cs(&body);
    run_cases(
        &add,
        &[[0x3C01, 0x1000, 0, 0], [0x7BFF, 0x5000, 0, 0]],
        |backend, i, _, got| {
            let want = [0x3C02u32, 0x7C00][i];
            assert_eq!(
                got, want,
                "{backend}: v_add_f16 case {i} = {got:#06x}, want {want:#06x}"
            );
        },
    );
    // v_cvt_pkrtz_f16_f32 still rounds toward zero: 1 + 3/2048 to 1 + 2^-10,
    // 1e6 to the largest finite half, -1e-8 to -0.
    let pkrtz = build_valu_cs(&vop3(0x12F, 0, vgpr(1), vgpr(2), 0));
    let (a, b) = ((1.0f32 + 3.0 / 2048.0).to_bits(), 1.0e6f32.to_bits());
    run_cases(
        &pkrtz,
        &[[a, b, 0, 0], [(-1.0e-8f32).to_bits(), a, 0, 0]],
        |backend, i, _, got| {
            let want = [0x7BFF_3C01u32, 0x3C01_8000][i];
            assert_eq!(
                got, want,
                "{backend}: v_cvt_pkrtz_f16_f32 case {i} = {got:#010x}, want {want:#010x}"
            );
        },
    );
}

#[test]
fn write_f_omod_flushes_neg_zero() {
    // v_add_f32_e64 v0, v1, v2 with omod: RDNA2 flushes -0.0 after the scale.
    // VOP3 opcode 0x103 = VOP2 v_add_f32 (0x003) at the VOP2-in-VOP3 base.
    let f = f32::to_bits;
    let nz = f(-0.0);
    let pz = f(0.0);
    for (omod, scale) in [(1u8, 2.0f32), (2, 4.0), (3, 0.5)] {
        let shader = build_valu_cs(&vop3_omod(0x103, 0, vgpr(1), vgpr(2), 0, omod));
        let cases: &[([u32; 4], u32)] = &[
            // -0 + -0 = -0; omod must yield +0, not IEEE -0.
            ([nz, nz, 0, 0], pz),
            ([pz, pz, 0, 0], pz),
            ([f(1.0), f(1.0), 0, 0], f(2.0 * scale)),
            ([f(-1.0), f(0.0), 0, 0], f(-scale)),
        ];
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: v_add_f32 omod={omod}({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1]
            );
        });
    }
    // Without omod, -0 + -0 stays -0.
    let plain = build_valu_cs(&vop3(0x103, 0, vgpr(1), vgpr(2), 0));
    run_cases(&plain, &[[nz, nz, 0, 0]], |backend, _, _, got| {
        assert_eq!(
            got, nz,
            "{backend}: v_add_f32(-0, -0) without omod must keep -0 ({got:#010x})"
        );
    });
}

#[test]
fn v_fma_legacy_f32_follows_the_d3d9_zero_rule() {
    // v_fma_legacy_f32 v0, v1, v2, v3 (VOP3 opcode 0x140 on RDNA2)
    let shader = build_valu_cs(&vop3(0x140, 0, vgpr(1), vgpr(2), vgpr(3)));
    let f = f32::to_bits;
    let cases: &[([u32; 4], u32)] = &[
        ([f(2.0), f(3.0), f(1.0), 0], f(7.0)),
        // A zero factor gives a zero product, whatever the other one is.
        ([0, f(f32::INFINITY), f(1.5), 0], f(1.5)),
        ([f(f32::NAN), 0, f(-2.0), 0], f(-2.0)),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, _, got| {
        let want = cases[i].1;
        assert_eq!(got, want, "{backend}: case {i}: {got:#x}, want {want:#x}");
    });
}

/// `v_fma_f16` is fused; unfused `v_mad_f16` / `v_mac_f16` round the product
/// to f16 first. With `a = b = 256h` and `c = -65504h`, the product overflows
/// to +inf unfused, while the fused form yields `32h` (VIC-BT-042).
#[test]
fn v_fma_f16_is_fused_mul_add_rounds_the_product() {
    // 256.0h, 256.0h, -65504.0h
    let cases = [[0x5C00u32, 0x5C00, 0xFBFF, 0]];

    // v_fma_f16 v0, v1, v2, v3 (VOP3 0x34b)
    let fma = build_valu_cs(&vop3(0x34b, 0, vgpr(1), vgpr(2), vgpr(3)));
    run_cases(&fma, &cases, |backend, _, _, got| {
        assert_eq!(
            got & 0xFFFF,
            0x5000,
            "{backend}: v_fma_f16(256h, 256h, -65504h) = {:#06x}, want 0x5000 (32h)",
            got & 0xFFFF
        );
    });

    // v_mad_f16 v0, v1, v2, v3 (VOP3 0x203, GFX9 opcode retained for titles)
    let mad = build_valu_cs(&vop3(0x203, 0, vgpr(1), vgpr(2), vgpr(3)));
    run_cases(&mad, &cases, |backend, _, _, got| {
        assert_eq!(
            got & 0xFFFF,
            0x7C00,
            "{backend}: v_mad_f16(256h, 256h, -65504h) = {:#06x}, want 0x7C00 (+inf)",
            got & 0xFFFF
        );
    });

    // v_mac_f16 v0, v1, v2: dst is the accumulator. Seed v0 from v3 first.
    // VOP3 encoding of VOP2 0x023 → 0x123.
    let mut body = Vec::new();
    body.push(vop1(0x001, 0, vgpr(3))); // v_mov_b32 v0, v3
    body.extend_from_slice(&vop3(0x123, 0, vgpr(1), vgpr(2), 0));
    let mac = build_valu_cs(&body);
    run_cases(&mac, &cases, |backend, _, _, got| {
        assert_eq!(
            got & 0xFFFF,
            0x7C00,
            "{backend}: v_mac_f16(256h, 256h, -65504h) = {:#06x}, want 0x7C00 (+inf)",
            got & 0xFFFF
        );
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

#[test]
fn v_cvt_off_f32_i4_uses_the_signed_low_nibble() {
    // v_cvt_off_f32_i4 v0, v1 (VOP1 opcode 0x00e)
    let shader = build_valu_cs(&[vop1(0x00e, 0, vgpr(1))]);
    let nibble = |n: u32| (((n as i32) << 28) >> 28) as f32 / 16.0;

    let mut cases: Vec<u32> = (0..16).collect();
    // Bits above the low nibble are ignored.
    cases.extend([
        0xABCD_EF05,
        0xFFFF_FFF8,
        0x1234_5670,
        0x8000_000F,
        0x0000_00F7,
    ]);
    let inputs: Vec<[u32; 4]> = cases.iter().map(|&x| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = nibble(cases[i]).to_bits();
        assert_eq!(
            got,
            want,
            "{backend}: v_cvt_off_f32_i4({:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
            inp[0],
            f32::from_bits(got),
        );
    });
    // Spot-check the reference itself: 0x5 -> 0.3125, 0x8 -> -0.5, 0xF -> -0.0625.
    assert_eq!(nibble(0xABCD_EF05), 0.3125);
    assert_eq!(nibble(8), -0.5);
    assert_eq!(nibble(0xF), -0.0625);
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

#[test]
fn v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would
    // round before the rounding to an integer, ties, and the edges.
    let mut bits: Vec<u16> = (0..=u16::MAX).step_by(251).collect();
    bits.extend([
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
        0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0x3A9B, 0xBA9B, 0x2E66, 0x1400,
    ]);
    for (op, signed, what) in [
        (0x063u16, true, "v_cvt_norm_i16_f16"),
        (0x064, false, "v_cvt_norm_u16_f16"),
    ] {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = bits.iter().map(|&b| [u32::from(b), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = cvt_norm16_reference(bits[i], signed);
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                bits[i],
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    let cases: &[(u16, u16, u16)] = &[
        // (f16 input, u16 result, i16 result)
        (0x0000, 0, 0),          // 0
        (0x8000, 0, 0),          // -0
        (0x3E00, 1, 1),          // 1.5 -> 1 (toward zero)
        (0xBE00, 0, 0xFFFF),     // -1.5 -> 0 / -1
        (0x57D0, 125, 125),      // 125.0
        (0x7800, 32768, 32767),  // 32768 saturates i16
        (0x77FF, 32752, 32752),  // 32752, the largest f16 below 32768
        (0xF800, 0, 0x8000),     // -32768 fits i16
        (0xF801, 0, 0x8000),     // -32800 saturates i16
        (0x7BFF, 65504, 32767),  // max finite f16
        (0xFBFF, 0, 0x8000),     // -65504
        (0x7C00, 0xFFFF, 32767), // +inf
        (0xFC00, 0, 0x8000),     // -inf
        (0x7E00, 0, 0),          // NaN
        (0xFE00, 0, 0),          // -NaN
        (0x0001, 0, 0),          // smallest subnormal
    ];
    for (op, what, pick) in [
        (0x052u16, "v_cvt_u16_f16", 0usize),
        (0x053, "v_cvt_i16_f16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_cvt_f16_u16_and_i16_widen_the_integer_first() {
    // v_mov_b32 v0, 0; v_cvt_f16_u16 / v_cvt_f16_i16 v0, v1 (VOP1 0x050 / 0x051).
    let cases: &[(u16, u16, u16)] = &[
        // (u16 / i16 input bits, f16 of the unsigned value, f16 of the signed value)
        (0x0000, 0x0000, 0x0000), // 0
        (0x0001, 0x3C00, 0x3C00), // 1.0
        (0x07FF, 0x67FF, 0x67FF), // 2047, the last integer below 2048 that f16 holds
        (0x0801, 0x6800, 0x6800), // 2049 ties to even: 2048
        (0x7FFF, 0x7800, 0x7800), // 32767 rounds up to 32768
        (0x8000, 0x7800, 0xF800), // 32768 / -32768
        (0x8001, 0x7800, 0xF800), // 32769 rounds to 32768 / -32767 rounds to -32768
        (0xFC00, 0x7BE0, 0xE400), // 64512 / -1024
        (0xFFFE, 0x7C00, 0xC000), // 65534 overflows to +inf / -2
        (0xFFFF, 0x7C00, 0xBC00), // 65535 overflows to +inf / -1
    ];
    for (op, what, pick) in [
        (0x050u16, "v_cvt_f16_u16", 0usize),
        (0x051, "v_cvt_f16_i16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// A 32-bit literal in a signed 64-bit operand is sign-extended (an
/// unsigned one's is zero-extended): -100 compared with `v[1:2]`, and
/// shifted by `s_ashr_i64`.
#[test]
fn signed_64_bit_operands_sign_extend_their_literal() {
    const MINUS_100: u32 = -100_i32 as u32;
    let check = |name: &str, body: &[u32], cases: &[([u32; 4], u32)]| {
        let shader = build_valu_cs(body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    };
    // v_cmp_*_i64 s6, v[1:2], -100 (VOP3, literal src1), then
    // v_cndmask_b32 v0, 0, 1, s6.
    let compare = |op: u16| {
        let [a, b] = vop3(op, 6, vgpr(1), 255, 0);
        let [c, d] = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
        [a, b, MINUS_100, c, d]
    };
    // An i64 as the two input dwords of `v[1:2]`.
    let pair = |n: i64| [n as u32, (n >> 32) as u32, 0, 0];
    let eq = [(pair(-100), 1), ([MINUS_100, 0, 0, 0], 0), (pair(0), 0)];
    check("v_cmp_eq_i64", &compare(0x0A2), &eq);
    let lt = [(pair(-200), 1), (pair(-50), 0), (pair(0), 0)];
    check("v_cmp_lt_i64", &compare(0x0A1), &lt);
    // s_ashr_i64 s[6:7], -100, 4; v_mov_b32 v0, s7: the high dword.
    let ashr = [sop2(0x23, 6, 255, 0x84), MINUS_100, vop1(0x001, 0, sgpr(7))];
    check("s_ashr_i64", &ashr, &[([0; 4], u32::MAX)]);
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    // s_bfe_u32 s6, s0, s1 (SOP2 opcode 0x27): the same 7-bit width as the signed form,
    // so width=32 keeps all 32 bits and nothing is sign-extended.
    let s_unsigned = build_valu_cs(&[sop2(0x27, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);
    let mut su_cases = unsigned_cases.to_vec();
    su_cases.push(([0x8000_0000, 4, 32, 0], 0x0800_0000));
    let su_in: Vec<[u32; 4]> = su_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_unsigned, &su_in, |backend, i, inp, got| {
        let want = su_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_u32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn v_cmp_class_f32_matches_rdna2() {
    // v_cmp_class_f32_e64 s6, v1, v2 (VOP3 opcode 0x088); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp = build_valu_cs(&[
        vop3(0x088, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x088, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary32 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7F80_0001, 0), // +sNaN (quiet bit 0x0040_0000 clear)
        (0xFFBF_FFFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7FC0_0000, 1), // +qNaN
        (0xFFC0_0001, 1), // -qNaN
        (0xFF80_0000, 2), // -inf
        (0xBF80_0000, 3), // -normal (-1.0)
        (0x8000_0001, 4), // -denormal
        (0x8000_0000, 5), // -0.0
        (0x0000_0000, 6), // +0.0
        (0x0000_0001, 7), // +denormal
        (0x3F80_0000, 8), // +normal (+1.0)
        (0x7F80_0000, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f32({:#010x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // With neg on src0: only the sign bit flips, so a denormal stays a
    // denormal and a signalling NaN stays signalling.
    let [w0, w1] = vop3(0x088, 6, vgpr(1), vgpr(2), 0);
    let cmp_neg = build_valu_cs(&[
        w0,
        w1 | 1 << 29,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    let neg_cases: &[([u32; 4], u32)] = &[
        ([0x0000_0001, 1 << 4, 0, 0], 1), // +denormal -> -denormal
        ([0x0000_0001, 1 << 5, 0, 0], 0), // not -0.0
        ([0x7F80_0001, 1 << 0, 0, 0], 1), // sNaN stays signalling
        ([0x7F80_0001, 1 << 1, 0, 0], 0),
    ];
    let neg_in: Vec<[u32; 4]> = neg_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_neg, &neg_in, |backend, i, inp, got| {
        let want = neg_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f32(-{:#010x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp_lo = build_valu_cs(&[
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1 << 11; // opsel[0] = 1 (high half)
    w1 |= 1 << 29; // neg[0] = 1
    let cmp_hi_neg = build_valu_cs(&[
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0), // sNaN (+sNaN, quiet bit 0x0200 clear)
        (0xFDFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7E00, 1), // +qNaN
        (0xFE01, 1), // -qNaN
        (0xFC00, 2), // -inf
        (0xBC00, 3), // -normal (-1.0)
        (0x8001, 4), // -denormal
        (0x8000, 5), // -0.0
        (0x0000, 6), // +0.0
        (0x0001, 7), // +denormal
        (0x3C00, 8), // +normal (+1.0)
        (0x7C00, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_lo, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16({:#06x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3);
    // high half 0x7C01 (+sNaN) negated -> still sNaN (class 0, not quieted).
    let hi_cases: &[([u32; 4], u32)] = &[
        ([0x3C00_0000, 1 << 3, 0, 0], 1),
        ([0x3C00_0000, 1 << 8, 0, 0], 0),
        ([0x7C01_FFFF, 1 << 0, 0, 0], 1),
        ([0x7C01_FFFF, 1 << 1, 0, 0], 0),
    ];
    let hi_in: Vec<[u32; 4]> = hi_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_hi_neg, &hi_in, |backend, i, inp, got| {
        let want = hi_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

/// RDNA2 ignores VOP3 `op_sel` on 16-bit compares (reading the low halves
/// regardless), while non-compare 16-bit ops such as `v_add_nc_u16` honour it.
#[test]
fn cmp16_ignores_op_sel_as_rdna2_does() {
    let vop3_opsel = |op: u16, dst: u8, s0: u16, s1: u16, opsel: u8| -> [u32; 2] {
        let [w0, w1] = vop3(op, dst, s0, s1, 0);
        [w0 | (u32::from(opsel) << 11), w1]
    };

    // High and low halves give opposite outcomes for every non-zero source op_sel.
    let f16_cases: &[([u32; 4], u32)] = &[
        // hi 2.0 vs 1.0, lo 1.0 < 1.5 -> true only when both read the low half
        ([0x4000_3C00, 0x3C00_3E00, 0, 0], 1),
        // hi 1.0 vs 3.0, lo 2.0 < 1.5 -> false only when both read the low half
        ([0x3C00_4000, 0x4200_3E00, 0, 0], 0),
    ];
    let u16_cases: &[([u32; 4], u32)] = &[
        // hi 10 vs 1, lo 1 < 2 -> true
        ([0x000A_0001, 0x0001_0002, 0, 0], 1),
        // hi 1 vs 20, lo 10 < 2 -> false
        ([0x0001_000A, 0x0014_0002, 0, 0], 0),
    ];
    let i16_cases: &[([u32; 4], u32)] = &[
        // hi 1 vs -2, lo -1 < 0 -> true
        ([0x0001_FFFF, 0xFFFE_0000, 0, 0], 1),
        // hi -2 vs 2, lo 1 < -1 -> false
        ([0xFFFE_0001, 0x0002_FFFF, 0, 0], 0),
    ];

    let mut cmp_shaders = Vec::new();
    for (op, what, cases) in [
        (0x0c9u16, "v_cmp_lt_f16", f16_cases),
        (0x0a9, "v_cmp_lt_u16", u16_cases),
        (0x089, "v_cmp_lt_i16", i16_cases),
    ] {
        for opsel in [0b0001u8, 0b0010, 0b0011] {
            let mut body = vop3_opsel(op, 6, vgpr(1), vgpr(2), opsel).to_vec();
            body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(6)));
            cmp_shaders.push((what, opsel, build_valu_cs(&body), cases));
        }
    }

    // v_cmpx_lt_f16_e64 (0x0d9) writing EXEC directly also ignores op_sel.
    let [w0, w1] = vop3_opsel(0x0d9, 126, vgpr(1), vgpr(2), 0b0001);
    let cmpx = build_valu_cs(&[
        s_mov_b32(6, 126),    // save exec_lo
        vop1(0x001, 0, 0x80), // v0 = 0
        w0,
        w1,                   // v_cmpx_lt_f16_e64 exec_lo, v1, v2 op_sel:[1,0]
        vop1(0x001, 0, 0x81), // v0 = 1 if active
        s_mov_b32(126, 6),    // restore exec_lo
    ]);

    // Control: v_add_nc_u16 (0x303) still honours op_sel.
    let [w0, w1] = vop3_opsel(0x303, 0, vgpr(1), vgpr(2), 0b0001);
    let add_hi = build_valu_cs(&[vop1(0x001, 0, 0x80), w0, w1]);

    for (what, opsel, shader, cases) in &cmp_shaders {
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what} opsel={opsel:#b} ({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }

    let f16_in: Vec<[u32; 4]> = f16_cases.iter().map(|c| c.0).collect();
    run_cases(&cmpx, &f16_in, |backend, i, inp, got| {
        let want = f16_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmpx_lt_f16 opsel=0b1 ({:#010x}, {:#010x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    run_cases(
        &add_hi,
        &[[0x000A_0001, 0x0014_0002, 0, 0]],
        |backend, _, _, got| {
            assert_eq!(
                got, 12,
                "{backend}: v_add_nc_u16 op_sel:[1,0] = {got}, want 12"
            );
        },
    );
}

/// `v_fma_mixlo_f16` / `v_fma_mixhi_f16` round the exact `a*b + c` once to
/// binary16 (VIC-BT-041). An f32 FMA then `legacyF32ToF16` double-rounds the
/// midpoint-tie class below to the even half.
#[test]
fn fma_mix_f16_single_rounds_midpoint_ties() {
    let f = f32::to_bits;
    // 1*1 + (2^-11 + 1 ulp): exact sits just above the 1 + 2^-11 midpoint.
    let c = 0x3A00_0001u32;
    // 2^-11 - 2^-30: exact sits just below the same midpoint.
    let c_below = f(2.0f32.powi(-11) - 2.0f32.powi(-30));
    let cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(1.0), c, 0], 0x3C01),
        ([f(-1.0), f(1.0), c | 0x8000_0000, 0], 0xBC01),
        ([f(2.0), f(0.5), c, 0], 0x3C01),
        ([f(1.0), f(1.0), c_below, 0], 0x3C00),
        ([f(-1.0), f(1.0), c_below | 0x8000_0000, 0], 0xBC00),
        ([f(2.0), f(0.5), c_below, 0], 0x3C00),
        // Ordinary fused multiply-add still lands on the exact half.
        ([f(2.0), f(3.0), f(1.0), 0], 0x4700), // 7.0
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    // v_fma_mixlo_f16 v0, v1, v2, v3
    let lo = build_valu_cs(&vop3p(0x021, 0, vgpr(1), vgpr(2), vgpr(3)));
    run_cases(&lo, &inputs, |backend, i, _, got| {
        let want = cases[i].1;
        assert_eq!(
            got as u16, want as u16,
            "{backend}: v_fma_mixlo_f16 case {i} = {got:#06x}, want {want:#06x}"
        );
    });
    // High half: seed v0's low half and check the result lands in bits 31:16.
    let mut body = vec![vop1(0x001, 0, 255)]; // v_mov_b32 v0, lit
    body.push(0x1234_5678);
    body.extend_from_slice(&vop3p(0x022, 0, vgpr(1), vgpr(2), vgpr(3)));
    let hi = build_valu_cs(&body);
    run_cases(&hi, &inputs, |backend, i, _, got| {
        let want = cases[i].1 << 16 | 0x5678;
        assert_eq!(
            got, want,
            "{backend}: v_fma_mixhi_f16 case {i} = {got:#010x}, want {want:#010x}"
        );
    });
}

#[test]
fn cmp16_sdwa_preserves_word_selects() {
    // llvm-mc -mcpu=gfx1030: v_cmp_lt_{f16,i16,u16}_sdwa vcc_lo, v1, v2.
    let instructions = [
        (0x7d92_04f9, [0x4000_3c00, 0x3800_3e00, 0, 0]),
        (0x7d12_04f9, [0x0003_ffff, 0xfffe_0000, 0, 0]),
        (0x7d52_04f9, [0x0003_0001, 0x0000_0002, 0, 0]),
    ];
    let select = vop3(0x101, 0, 0x80, 0x81, sgpr(106));
    for (instruction, input) in instructions {
        // Low/low gives true. Selecting either high word gives false.
        for (extension, expected) in [(0x0404_0001, 1), (0x0405_0001, 0), (0x0504_0001, 0)] {
            let shader = build_valu_cs(&[instruction, extension, select[0], select[1]]);
            run_cases(&shader, &[input], |backend, _, _, actual| {
                assert_eq!(
                    actual, expected,
                    "{backend}: {instruction:#x} {extension:#x}"
                );
            });
        }
    }
}

/// A carry-in whose mask is a plain SGPR (a root constant) the structure pass
/// does not follow: this lane adds its own bit of the number.
#[test]
fn v_add_co_ci_on_an_untracked_mask_adds_this_lanes_bit() {
    // v_add_co_ci_u32_e64 v0, s12, v1, v2, s3 (VOP3B 0x128; s12 takes the carry out).
    let mut op = vop3(0x128, 0, vgpr(1), vgpr(2), sgpr(3));
    op[0] |= 12 << 8;
    let shader = build_valu_cs(&op);
    run_cases_on(
        &shader,
        &[
            [10, 20, 0, 0],
            [10, 20, 0, 1],
            [10, 20, 0, 2],
            [10, 20, 0, !0],
        ],
        &[("hardware", false)],
        |label, i, inp, got| {
            assert_eq!(got, inp[0] + inp[1] + (inp[3] & 1), "{label} case {i}");
        },
    );
}

/// Inline float constants on 16-bit integer ops keep their f32 bits: `1.0`
/// is `0x3F80_0000` (low half 0), not f16 `0x3C00` (VIC-BT-021 / #438).
#[test]
fn u16_inline_float_uses_f32_bits() {
    // v_add_nc_u16 v0, 1.0, v1  (inline const 242)
    let add = build_valu_cs(&vop3(0x303, 0, 242, vgpr(1), 0));
    // Hardware: low half of 0x3F80_0000 is 0, so v0 = v1's low half.
    // Bug: f16 decode of 1.0 is 0x3C00, so v0 = v1 + 0x3C00.
    run_cases(
        &add,
        &[[5, 0, 0, 0], [0x000A_0007, 0, 0, 0]],
        |backend, i, inp, got| {
            let want = inp[0] & 0xFFFF;
            assert_eq!(
                got & 0xFFFF,
                want,
                "{backend}: v_add_nc_u16 1.0, v1 case {i}: {got:#010x}, want low {want:#06x}"
            );
        },
    );
}

/// `v_mad_f32` / `v_mac_f32` flush subnormals before and after `omod`
/// regardless of `MODE.denorm32` (VIC-BT-043 / #440).
#[test]
fn v_mad_f32_flushes_subnormals_around_omod() {
    let flt_min = 0x0080_0000u32;
    let f = f32::to_bits;
    // v_mad_f32 v0, v1, v2, v3
    let mad = build_valu_cs(&vop3(0x141, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_mad_f32 v0, v1, v2, v3 mul:2
    let mad_mul2 = build_valu_cs(&vop3_omod(0x141, 0, vgpr(1), vgpr(2), vgpr(3), 1));
    // v_mad_f32 v0, v1, v2, v3 div:2
    let mad_div2 = build_valu_cs(&vop3_omod(0x141, 0, vgpr(1), vgpr(2), vgpr(3), 3));
    // v_mov_b32 v0, v4 then v_mac_f32 v0, v1, v2 (acc in v0 from s3/v4)
    let mac = build_valu_cs(&[vop1(0x001, 0, vgpr(4)), vop2(0x1f, 0, vgpr(1), 2)]);

    let mad_cases: &[([u32; 4], u32)] = &[
        ([f(2.0), f(3.0), f(1.0), 0], f(7.0)),
        // 1.5*FLT_MIN - FLT_MIN = 0.5*FLT_MIN (subnormal) → +0
        ([f(1.5), flt_min, flt_min ^ 0x8000_0000, 0], 0),
        // Smallest subnormal input is flushed before the mul.
        ([0x0000_0001, f(4.0), 0, 0], 0),
    ];
    let mad_mul2_cases: &[([u32; 4], u32)] = &[
        // Same sum flushed to 0 before mul:2 → +0 (not FLT_MIN).
        ([f(1.5), flt_min, flt_min ^ 0x8000_0000, 0], 0),
        ([f(2.0), f(3.0), f(1.0), 0], f(14.0)),
    ];
    let mad_div2_cases: &[([u32; 4], u32)] = &[
        // FLT_MIN / 2 is subnormal; flush after omod → +0.
        ([f(1.0), flt_min, 0, 0], 0),
        ([f(2.0), f(4.0), 0, 0], f(4.0)),
    ];
    // mac: v0 += v1*v2 with initial v0 from inputs[3]
    let mac_cases: &[([u32; 4], u32)] = &[
        ([f(2.0), f(3.0), 0, f(1.0)], f(7.0)),
        ([f(1.5), flt_min, 0, flt_min ^ 0x8000_0000], 0),
    ];

    for (shader, cases, what) in [
        (&mad, mad_cases, "v_mad_f32"),
        (&mad_mul2, mad_mul2_cases, "v_mad_f32 mul:2"),
        (&mad_div2, mad_div2_cases, "v_mad_f32 div:2"),
        (&mac, mac_cases, "v_mac_f32"),
    ] {
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(shader, &inputs, |backend, i, _, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what} case {i}: {got:#010x}, want {want:#010x}"
            );
        });
    }
}

/// The CPU model of the 24-bit multiplies: the 48-bit product of the low 24
/// bits of each operand, zero- or sign-extended; `hi` picks its high dword.
fn mul24(a: u32, b: u32, signed: bool, hi: bool) -> u32 {
    let ext = |x: u32| {
        if signed {
            i64::from(((x << 8) as i32) >> 8)
        } else {
            i64::from(x & 0xFF_FFFF)
        }
    };
    let p = ext(a) * ext(b);
    (if hi { p >> 32 } else { p }) as u32
}

#[test]
fn mul24_model_follows_the_isa() {
    assert_eq!(mul24(0xFF_FFFF, 0xFF_FFFF, false, true), 0xFFFF);
    assert_eq!(mul24(0xFF_FFFF, 0xFF_FFFF, false, false), 0xFE00_0001);
    // -1 * -1 = 1: the high dword is 0.
    assert_eq!(mul24(0xFF_FFFF, 0xFF_FFFF, true, true), 0);
    // -2^23 * 1: the high dword is all ones.
    assert_eq!(mul24(0x80_0000, 1, true, true), u32::MAX);
    // The top byte of each operand is ignored.
    assert_eq!(mul24(0xAB80_0000, 0xCD80_0000, false, true), 0x4000);
}

/// `v_mul_hi_u32_u24` and `v_mul_hi_i32_i24` give the high dword of the
/// 48-bit product whose low dword `v_mul_u32_u24`/`v_mul_i32_i24` give.
#[test]
fn mul_hi_24_bit_products() {
    let inputs: Vec<[u32; 4]> = [
        (0, 0),
        (0xFF_FFFF, 0xFF_FFFF),
        (0x80_0000, 0x80_0000),
        (0x80_0000, 1),
        (0x7F_FFFF, 0x80_0001),
        (0xFF12_3456, 0x00AB_CDEF),
        (0x1234_5678, 0x9ABC_DEF0),
    ]
    .iter()
    .map(|&(a, b)| [a, b, 0, 0])
    .collect();
    // VOP2 `op v0, v1, v2`.
    for (op, signed, what) in [
        (0x0A, true, "v_mul_hi_i32_i24"),
        (0x0C, false, "v_mul_hi_u32_u24"),
    ] {
        let shader = build_valu_cs(&[vop2(op, 0, vgpr(1), 2)]);
        run_cases(&shader, &inputs, |backend, i, x, got| {
            let want = mul24(x[0], x[1], signed, true);
            assert_eq!(
                got, want,
                "{backend}: {what} case {i}: {got:#010x}, want {want:#010x}"
            );
        });
    }
}

/// The CPU model of `v_sad_*`: `c` plus the sum of the unsigned differences
/// of `a` and `b`'s fields of `width` bits, wrapping; `hi` shifts the sum
/// to the high half first (`v_sad_hi_u8`).
fn sad(a: u32, b: u32, c: u32, width: u32, hi: bool) -> u32 {
    let mask = if width == 32 {
        u32::MAX
    } else {
        (1 << width) - 1
    };
    let sum = (0..32 / width).fold(0u32, |s, i| {
        let (x, y) = (a >> (width * i) & mask, b >> (width * i) & mask);
        s.wrapping_add(x.abs_diff(y))
    });
    (if hi { sum << 16 } else { sum }).wrapping_add(c)
}

#[test]
fn sad_model_follows_the_isa() {
    // |1-4| + |2-3| + |3-2| + |4-1| = 8.
    assert_eq!(sad(0x0403_0201, 0x0102_0304, 0, 8, false), 8);
    assert_eq!(sad(0x0403_0201, 0x0102_0304, 2, 8, true), 8 << 16 | 2);
    // |0xFFFF - 1| + |0 - 0xFFFF| = 0x1FFFD.
    assert_eq!(sad(0x0000_FFFF, 0xFFFF_0001, 0, 16, false), 0x1_FFFD);
    // |1 - 0xFFFF_FFFF| + 5 wraps to 3.
    assert_eq!(sad(1, u32::MAX, 5, 32, false), 3);
}

/// `v_sad_u8`, `v_sad_hi_u8`, `v_sad_u16` and `v_sad_u32`.
#[test]
fn sums_of_absolute_differences() {
    let inputs: Vec<[u32; 4]> = [
        [0, 0, 0, 0],
        [0x0403_0201, 0x0102_0304, 7, 0],
        [0xFF00_FF00, 0x00FF_00FF, 0, 0],
        [0x0000_FFFF, 0xFFFF_0001, 0x10, 0],
        [0x1234_5678, 0x9ABC_DEF0, 0xFFFF_FFF0, 0],
        [1, u32::MAX, 5, 0],
    ]
    .to_vec();
    for (op, width, hi, what) in [
        (0x15A, 8, false, "v_sad_u8"),
        (0x15B, 8, true, "v_sad_hi_u8"),
        (0x15C, 16, false, "v_sad_u16"),
        (0x15D, 32, false, "v_sad_u32"),
    ] {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &inputs, |backend, i, x, got| {
            let want = sad(x[0], x[1], x[2], width, hi);
            assert_eq!(
                got, want,
                "{backend}: {what} case {i}: {got:#010x}, want {want:#010x}"
            );
        });
    }
}

/// `v_msad_u8` skips bytes whose reference/mask byte (`src1`) is 0 and wraps
/// the 32-bit sum with `src2`.
#[test]
fn v_msad_u8_masks_zero_bytes_and_accumulates_sad() {
    let msad = |a: u32, b: u32, c: u32| -> u32 {
        (0..4).fold(c, |acc, i| {
            let x = (a >> (8 * i)) & 0xFF;
            let y = (b >> (8 * i)) & 0xFF;
            if y == 0 {
                acc
            } else {
                acc.wrapping_add(x.abs_diff(y))
            }
        })
    };
    let inputs: Vec<[u32; 4]> = vec![
        [0, 0, 0, 0],
        // Every mask byte is 0 -> result is just src2.
        [0xFFAA_5511, 0x0000_0000, 0x1234, 0],
        // Mixed zero and non-zero mask bytes: bytes 1 and 3 masked out.
        [0xFF10_2030, 0x0015_0025, 7, 0],
        // All non-zero mask bytes match v_sad_u8.
        [0x0403_0201, 0x0102_0304, 10, 0],
        // Wrapping 32-bit accumulator.
        [0xFF00_FF00, 0x0101_0101, 0xFFFF_FFF0, 0],
    ];
    let shader = build_valu_cs(&vop3(0x171, 0, vgpr(1), vgpr(2), vgpr(3)));
    run_cases(&shader, &inputs, |backend, i, x, got| {
        let want = msad(x[0], x[1], x[2]);
        assert_eq!(
            got, want,
            "{backend}: v_msad_u8 case {i}: {got:#010x}, want {want:#010x}"
        );
    });
}

/// The CPU model of the integer dot products: `c` plus the products of
/// `a` and `b`'s fields of `width` bits, sign- or zero-extended, wrapping.
fn dot(a: u32, b: u32, c: u32, width: u32, signed: bool) -> u32 {
    let field = |x: u32, n: u32| {
        if signed {
            (((x << (32 - width * (n + 1))) as i32) >> (32 - width)) as u32
        } else {
            x >> (width * n) & ((1u64 << width) - 1) as u32
        }
    };
    (0..32 / width).fold(c, |s, n| {
        s.wrapping_add(field(a, n).wrapping_mul(field(b, n)))
    })
}

#[test]
fn dot_model_follows_the_isa() {
    // 1*5 + 2*6 + 3*7 + 4*8 = 70.
    assert_eq!(dot(0x0403_0201, 0x0807_0605, 0, 8, false), 70);
    // Every byte -1 times 2: -8, plus 10.
    assert_eq!(dot(0xFFFF_FFFF, 0x0202_0202, 10, 8, true), 2);
    // Unsigned, the same bytes are 255: 4 * 510.
    assert_eq!(dot(0xFFFF_FFFF, 0x0202_0202, 0, 8, false), 2040);
    // -32768 * -32768 twice is 2^31: it wraps.
    assert_eq!(dot(0x8000_8000, 0x8000_8000, 0, 16, true), 0x8000_0000);
    // Nibbles: 8 * (-8 * -8) = 512; unsigned 8 * 64 the same.
    assert_eq!(dot(0x8888_8888, 0x8888_8888, 0, 4, true), 512);
    assert_eq!(dot(0xFFFF_FFFF, 0x7777_7777, 100, 4, true), 44);
    assert_eq!(dot(0xFFFF_FFFF, 0x7777_7777, 100, 4, false), 100 + 8 * 105);
}

/// `v_dot2_{i32_i16,u32_u16}`, `v_dot4_{i32_i8,u32_u8}` and
/// `v_dot8_{i32_i4,u32_u4}` with op_sel_hi as assemblers write it.
#[test]
fn integer_dot_products() {
    let inputs: Vec<[u32; 4]> = vec![
        [0, 0, 0, 0],
        [0x0403_0201, 0x0807_0605, 7, 0],
        [0xFFFF_FFFF, 0x0202_0202, 10, 0],
        [0x8000_8000, 0x8000_8000, 0x7FFF_FFFF, 0],
        [0x8888_8888, 0x7777_7777, 0xFFFF_FF00, 0],
        [0x1234_5678, 0x9ABC_DEF0, 0x0BAD_F00D, 0],
    ];
    for (op, width, signed, what) in [
        (0x14, 16, true, "v_dot2_i32_i16"),
        (0x15, 16, false, "v_dot2_u32_u16"),
        (0x16, 8, true, "v_dot4_i32_i8"),
        (0x17, 8, false, "v_dot4_u32_u8"),
        (0x18, 4, true, "v_dot8_i32_i4"),
        (0x19, 4, false, "v_dot8_u32_u4"),
    ] {
        let [w0, w1] = vop3p(op, 0, vgpr(1), vgpr(2), vgpr(3));
        // op_sel_hi:[1,1,1]
        let shader = build_valu_cs(&[w0 | 1 << 14, w1 | 3 << 27]);
        run_cases(&shader, &inputs, |backend, i, x, got| {
            let want = dot(x[0], x[1], x[2], width, signed);
            assert_eq!(
                got, want,
                "{backend}: {what} case {i}: {got:#010x}, want {want:#010x}"
            );
        });
    }
}

/// `v_mad_u32_u16` / `v_mad_i32_i16` (op_sel halves) and `v_lerp_u8` (#405).
#[test]
fn mad_u32_u16_i32_i16_and_lerp_u8() {
    let vop3_opsel = |op: u16, vdst: u8, s0: u16, s1: u16, s2: u16, opsel: u8| -> [u32; 2] {
        let [w0, w1] = vop3(op, vdst, s0, s1, s2);
        [w0 | (u32::from(opsel) << 11), w1]
    };

    // v_mad_u32_u16 v0, v1, v2, v3
    let mad_u = build_valu_cs(&vop3(0x373, 0, vgpr(1), vgpr(2), vgpr(3)));
    // opsel[0:1] = high halves of src0/src1
    let mad_u_hi = build_valu_cs(&vop3_opsel(0x373, 0, vgpr(1), vgpr(2), vgpr(3), 0b0011));
    // v_mad_i32_i16 v0, v1, v2, v3
    let mad_i = build_valu_cs(&vop3(0x375, 0, vgpr(1), vgpr(2), vgpr(3)));
    let mad_i_hi = build_valu_cs(&vop3_opsel(0x375, 0, vgpr(1), vgpr(2), vgpr(3), 0b0011));
    // v_lerp_u8 v0, v1, v2, v3
    let lerp = build_valu_cs(&vop3(0x14d, 0, vgpr(1), vgpr(2), vgpr(3)));

    let mad_u_cases: &[([u32; 4], u32)] = &[
        ([3, 5, 7, 0], 3 * 5 + 7),
        ([0x0001_0003, 0x0002_0005, 7, 0], 3 * 5 + 7),
        // wrap: 0xFFFF * 0xFFFF + 1 = 0xFFFE_0002
        ([0xFFFF, 0xFFFF, 1, 0], 0xFFFE_0002),
    ];
    let mad_u_hi_cases: &[([u32; 4], u32)] = &[
        // hi(0x0004_xxxx)=4, hi(0x0006_xxxx)=6 → 4*6+7
        ([0x0004_0003, 0x0006_0005, 7, 0], 4 * 6 + 7),
        ([0xFFFF_0001, 0x0002_FFFF, 0, 0], 0xFFFF * 2),
    ];
    let mad_i_cases: &[([u32; 4], u32)] = &[
        ([3, 5, 7, 0], 22),
        // (-2) * 3 + 1 = -5
        ([0xFFFE, 3, 1, 0], (-5i32) as u32),
        // (-1) * (-1) + 0 = 1
        ([0xFFFF, 0xFFFF, 0, 0], 1),
        // wrap: 0x8000 * 0x8000 = 0x4000_0000; + 0xC000_0000 wraps
        ([0x8000, 0x8000, 0xC000_0000, 0], 0),
    ];
    let mad_i_hi_cases: &[([u32; 4], u32)] = &[
        // hi 0xFFFE (-2) * hi 0x0003 (3) + 1 = -5
        ([0xFFFE_0001, 0x0003_FFFF, 1, 0], (-5i32) as u32),
    ];
    // Per byte: (a + b + (c & 1)) >> 1
    let lerp_cases: &[([u32; 4], u32)] = &[
        (
            [0x03_02_01_00, 0x07_06_05_04, 0x00_00_00_00, 0],
            0x05_04_03_02,
        ),
        // Rounding bit lifts odd sums: (1+0+1)>>1 = 1 vs (1+0+0)>>1 = 0.
        (
            [0x00_00_00_01, 0x00_00_00_00, 0x00_00_00_01, 0],
            0x00_00_00_01,
        ),
        // byte0: (1+2+1)>>1=2; byte1: (3+4+1)>>1=4; byte2: (5+6+0)>>1=5; byte3: (7+8+1)>>1=8
        (
            [0x07_05_03_01, 0x08_06_04_02, 0x01_00_01_01, 0],
            0x08_05_04_02,
        ),
        (
            [0xFF_00_80_40, 0x01_00_80_40, 0x01_01_00_00, 0],
            0x80_00_80_40,
        ),
    ];

    for (shader, cases, what) in [
        (&mad_u, mad_u_cases, "v_mad_u32_u16"),
        (&mad_u_hi, mad_u_hi_cases, "v_mad_u32_u16 opsel hi"),
        (&mad_i, mad_i_cases, "v_mad_i32_i16"),
        (&mad_i_hi, mad_i_hi_cases, "v_mad_i32_i16 opsel hi"),
        (&lerp, lerp_cases, "v_lerp_u8"),
    ] {
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(shader, &inputs, |backend, i, _, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what} case {i}: {got:#010x}, want {want:#010x}"
            );
        });
    }
}

/// VOP3 integer `clamp` saturates unsigned add/sub on overflow/underflow
/// (VIC-BT-010). Without the fix these refuse to lower, or wrap.
#[test]
fn integer_vop3_clamp_saturates_add_and_sub() {
    // VOP2 ops in VOP3 encoding use opcode + 0x100 (0x025 alone is VOPC).
    // v_add_nc_u32 / v_sub_nc_u32 / v_subrev_nc_u32 with clamp.
    let add_nc = build_valu_cs(&vop3_clamp(0x125, 0, vgpr(1), vgpr(2), 0));
    let sub_nc = build_valu_cs(&vop3_clamp(0x126, 0, vgpr(1), vgpr(2), 0));
    let subrev_nc = build_valu_cs(&vop3_clamp(0x127, 0, vgpr(1), vgpr(2), 0));
    // v_add_co_u32 / v_sub_co_u32 with clamp (carry → s16, unused).
    let add_co = build_valu_cs(&vop3b_clamp(0x30f, 0, 16, vgpr(1), vgpr(2), 0));
    let sub_co = build_valu_cs(&vop3b_clamp(0x310, 0, 16, vgpr(1), vgpr(2), 0));

    let add_cases: &[([u32; 4], u32)] = &[
        ([1, 2, 0, 0], 3),
        ([u32::MAX, 1, 0, 0], u32::MAX),
        ([0x8000_0000, 0x8000_0000, 0, 0], u32::MAX),
        ([0, 0, 0, 0], 0),
    ];
    let sub_cases: &[([u32; 4], u32)] = &[
        ([5, 3, 0, 0], 2),
        ([1, 2, 0, 0], 0),
        ([0, 1, 0, 0], 0),
        ([u32::MAX, u32::MAX, 0, 0], 0),
    ];
    // subrev: dst = src1 - src0 (v2 - v1).
    let subrev_cases: &[([u32; 4], u32)] =
        &[([3, 5, 0, 0], 2), ([2, 1, 0, 0], 0), ([1, 0, 0, 0], 0)];

    for (shader, cases, what) in [
        (&add_nc, add_cases, "v_add_nc_u32 clamp"),
        (&sub_nc, sub_cases, "v_sub_nc_u32 clamp"),
        (&subrev_nc, subrev_cases, "v_subrev_nc_u32 clamp"),
        (&add_co, add_cases, "v_add_co_u32 clamp"),
        (&sub_co, sub_cases, "v_sub_co_u32 clamp"),
    ] {
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(shader, &inputs, |backend, i, _, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what} case {i}: {got:#010x}, want {want:#010x}"
            );
        });
    }
}

/// `victory_formats::bcpack::zz`, the function the test below runs as console code.
fn zz_reference(v: u8, bits: u32) -> u8 {
    let mask = ((1u32 << bits) - 1) as u8;
    ((v >> 1) ^ if v & 1 != 0 { mask } else { 0 }) & mask
}

/// BCPack's zigzag decode written as console code matches the formats crate's function on
/// the GPU: every symbol at every width, widths past a byte included.
#[test]
fn bcpack_zigzag_decode_matches_the_formats_function() {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = the symbol, v2 = its width in bits; v5 = the width's mask, as a byte.
    body.extend(vop3(0x11A, 5, vgpr(2), 129, 128)); // v_lshlrev_b32 v5, v2, 1
    body.extend(vop3(0x125, 5, 193, vgpr(5), 128)); // v_add_nc_u32 v5, -1, v5
    body.extend(vop3(0x148, 5, vgpr(5), 128, 128 + 8)); // v_bfe_u32 v5, v5, 0, 8
    body.extend(vop3(0x116, 6, 129, vgpr(1), 128)); // v_lshrrev_b32 v6, 1, v1
    body.extend(vop3(0x148, 7, vgpr(1), 128, 129)); // v_bfe_u32 v7, v1, 0, 1
    body.extend(vop3(0x0C5, 106, 128, vgpr(7), 128)); // v_cmp_ne_u32 vcc, 0, v7
    body.extend(vop3(0x101, 8, 128, vgpr(5), vcc)); // v_cndmask_b32 v8, 0, v5, vcc
    body.extend(vop3(0x11D, 9, vgpr(6), vgpr(8), 128)); // v_xor_b32 v9, v6, v8
    body.extend(vop3(0x11B, 0, vgpr(9), vgpr(5), 128)); // v_and_b32 v0, v9, v5
    let shader = build_valu_cs(&body);
    let mut cases = Vec::new();
    let mut want = Vec::new();
    for bits in (0u32..=9).chain([16, 31]) {
        for v in 0u32..256 {
            cases.push([v, bits, 0, 0]);
            want.push(u32::from(zz_reference(v as u8, bits)));
        }
    }
    run_cases(&shader, &cases, |backend, i, inputs, got| {
        assert_eq!(got, want[i], "{backend}: zz{inputs:?}");
    });
}

/// `victory_formats::bcpack::interleave`, the function the test below runs as console code.
fn interleave_reference(a: u8, b: u8) -> u8 {
    let spread = |v: u8| (0..4).fold(0, |acc, k| acc | (v >> k & 1) << (2 * k));
    spread(a) | spread(b) << 1
}

/// BCPack's interleave of two 4-bit symbols, written as console code, matches the formats
/// crate's function on the GPU: every pair of symbols, with and without bits above the
/// nibble that the function ignores.
#[test]
fn bcpack_symbol_interleave_matches_the_formats_function() {
    // v1 = the first symbol, on the even bits; v2 = the second, on the odd ones.
    let mut body = vec![vop1(0x001, 0, 128)]; // v_mov_b32 v0, 0
    for bit in 0u16..4 {
        for (symbol, position) in [(1, 2 * bit), (2, 2 * bit + 1)] {
            body.extend(vop3(0x148, 3, vgpr(symbol), 128 + bit, 129)); // v_bfe_u32 v3, symbol, bit, 1
            body.extend(vop3(0x36F, 0, vgpr(3), 128 + position, vgpr(0))); // v_lshl_or_b32 v0, v3, position, v0
        }
    }
    let shader = build_valu_cs(&body);
    let mut cases: Vec<[u32; 4]> = (0u32..256)
        .map(|pair| [pair & 15, pair >> 4, 0, 0])
        .collect();
    cases.extend((0u32..16).map(|a| [a | 0xF0, (15 - a) | 0xA0, 0, 0]));
    run_cases(&shader, &cases, |backend, _, [a, b, ..], got| {
        let want = u32::from(interleave_reference(a as u8, b as u8));
        assert_eq!(got, want, "{backend}: interleave({a:#x}, {b:#x})");
    });
}

/// `victory_formats::bcpack::add_index`, the function the test below runs as console code.
fn add_index_reference(cur: i16, bits: u8, refv: i16, ref_bits: u8) -> i16 {
    let up = |v: i16, b: u8| {
        if b == 0 || b > 16 {
            0
        } else {
            (v as u16) << (16 - b)
        }
    };
    let sum = up(cur, bits).wrapping_add(up(refv, ref_bits)) as i16;
    if bits == 0 { 0 } else { sum >> (16 - bits) }
}

fn add_index_body() -> Vec<u32> {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = cur, v2 = bits, v3 = refv, v4 = ref_bits. v6 = cur left-aligned to 16 bits.
    body.extend(vop3(0x126, 5, 128 + 16, vgpr(2), 128)); // v_sub_nc_u32 v5, 16, v2
    body.extend(vop3(0x11A, 6, vgpr(5), vgpr(1), 128)); // v_lshlrev_b32 v6, v5, v1
    body.extend(vop3(0x148, 6, vgpr(6), 128, 128 + 16)); // v_bfe_u32 v6, v6, 0, 16
    body.extend(vop3(0x125, 7, 193, vgpr(2), 128)); // v_add_nc_u32 v7, -1, v2
    body.extend(vop3(0x0C4, 106, 128 + 16, vgpr(7), 128)); // v_cmp_gt_u32 vcc, 16, v7
    body.extend(vop3(0x101, 6, 128, vgpr(6), vcc)); // v_cndmask_b32 v6, 0, v6, vcc
    // v9 = refv left-aligned the same way.
    body.extend(vop3(0x126, 8, 128 + 16, vgpr(4), 128));
    body.extend(vop3(0x11A, 9, vgpr(8), vgpr(3), 128));
    body.extend(vop3(0x148, 9, vgpr(9), 128, 128 + 16));
    body.extend(vop3(0x125, 10, 193, vgpr(4), 128));
    body.extend(vop3(0x0C4, 106, 128 + 16, vgpr(10), 128));
    body.extend(vop3(0x101, 9, 128, vgpr(9), vcc));
    // The 16-bit sum, signed, back down to `bits` bits; nothing at a width of 0.
    body.extend(vop3(0x125, 11, vgpr(6), vgpr(9), 128)); // v_add_nc_u32 v11, v6, v9
    body.extend(vop3(0x149, 11, vgpr(11), 128, 128 + 16)); // v_bfe_i32 v11, v11, 0, 16
    body.extend(vop3(0x118, 0, vgpr(5), vgpr(11), 128)); // v_ashrrev_i32 v0, v5, v11
    body.extend(vop3(0x0C2, 106, 128, vgpr(2), 128)); // v_cmp_eq_u32 vcc, 0, v2
    body.extend(vop3(0x101, 0, vgpr(0), 128, vcc)); // v_cndmask_b32 v0, v0, 0, vcc
    body
}

/// BCPack's index residual plus its reference, written as console code, matches the formats
/// crate's function on the GPU at every width, reference widths out of range included.
#[test]
fn bcpack_index_residual_sum_matches_the_formats_function() {
    let shader = build_valu_cs(&add_index_body());
    let values: [i16; 6] = [0, 1, -1, 0x7FFF, -0x8000, -0x1235];
    let refs: [i16; 3] = [0, 0x2A5, -7];
    let mut cases = Vec::new();
    let mut want = Vec::new();
    for bits in 0u8..=16 {
        for ref_bits in [0u8, 1, 4, 8, 15, 16, 17] {
            for cur in values {
                for refv in refs {
                    cases.push([
                        cur as i32 as u32,
                        u32::from(bits),
                        refv as i32 as u32,
                        u32::from(ref_bits),
                    ]);
                    want.push(add_index_reference(cur, bits, refv, ref_bits) as i32 as u32);
                }
            }
        }
    }
    run_cases(&shader, &cases, |backend, i, inputs, got| {
        assert_eq!(got, want[i], "{backend}: add_index{inputs:?}");
    });
}

/// `victory_formats::bcpack::nibbles`, the function the test below runs as console code.
fn nibbles_reference(b: u8, odd_first: bool) -> [i16; 2] {
    let half = |v: u8| (0..4).fold(0u8, |acc, k| acc | (v >> (2 * k) & 1) << k);
    let z = |u: u8| i16::from(u >> 1) ^ -i16::from(u & 1);
    let (even, odd) = (z(half(b)), z(half(b >> 1)));
    if odd_first { [odd, even] } else { [even, odd] }
}

fn nibbles_body() -> Vec<u32> {
    let vcc = sgpr(106);
    // v1 = the byte, v2 = whether the odd bits come first. v5 and v6 gather the even and
    // the odd bits.
    let mut body = vec![vop1(0x001, 5, 128), vop1(0x001, 6, 128)]; // v_mov_b32 v5, 0; v6, 0
    for k in 0u16..4 {
        for (half, bit) in [(5, 2 * k), (6, 2 * k + 1)] {
            body.extend(vop3(0x148, 7, vgpr(1), 128 + bit, 129)); // v_bfe_u32 v7, v1, bit, 1
            body.extend(vop3(0x36F, half, vgpr(7), 128 + k, vgpr(half))); // v_lshl_or_b32
        }
    }
    // Zigzag to signed, kept as 16 bits: v8 from the even half, v11 from the odd one.
    for (half, out) in [(5, 8), (6, 11)] {
        body.extend(vop3(0x116, 9, 129, vgpr(half), 128)); // v_lshrrev_b32 v9, 1, half
        body.extend(vop3(0x148, 10, vgpr(half), 128, 129)); // v_bfe_u32 v10, half, 0, 1
        body.extend(vop3(0x126, 10, 128, vgpr(10), 128)); // v_sub_nc_u32 v10, 0, v10
        body.extend(vop3(0x11D, out, vgpr(9), vgpr(10), 128)); // v_xor_b32 out, v9, v10
        body.extend(vop3(0x148, out, vgpr(out), 128, 128 + 16)); // v_bfe_u32 out, out, 0, 16
    }
    // The pair in the order asked for, the second in the result's high half.
    body.extend(vop3(0x0C5, 106, 128, vgpr(2), 128)); // v_cmp_ne_u32 vcc, 0, v2
    body.extend(vop3(0x101, 14, vgpr(8), vgpr(11), vcc)); // v_cndmask_b32 v14, v8, v11, vcc
    body.extend(vop3(0x101, 15, vgpr(11), vgpr(8), vcc)); // v_cndmask_b32 v15, v11, v8, vcc
    body.extend(vop3(0x36F, 0, vgpr(15), 128 + 16, vgpr(14))); // v_lshl_or_b32 v0, v15, 16, v14
    body
}

/// BCPack's split of a byte into two signed nibbles, written as console code, matches the
/// formats crate's function on the GPU for every byte in both orders.
#[test]
fn bcpack_nibble_split_matches_the_formats_function() {
    let shader = build_valu_cs(&nibbles_body());
    let mut cases = Vec::new();
    let mut want = Vec::new();
    for odd_first in [false, true] {
        for b in 0u32..256 {
            cases.push([b, u32::from(odd_first), 0, 0]);
            let [first, second] = nibbles_reference(b as u8, odd_first);
            want.push(u32::from(second as u16) << 16 | u32::from(first as u16));
        }
    }
    run_cases(&shader, &cases, |backend, i, inputs, got| {
        assert_eq!(got, want[i], "{backend}: nibbles{inputs:?}");
    });
}

/// `victory_formats::bcpack::bc7_precision` with its table and `pbits`, the function the
/// test below runs as console code.
fn bc7_precision_reference(ctrl: u16) -> [u8; 4] {
    const BC7_BITS: [[u8; 4]; 8] = [
        [4, 4, 4, 0],
        [6, 6, 6, 0],
        [5, 5, 5, 0],
        [7, 7, 7, 0],
        [5, 5, 5, 6],
        [7, 7, 7, 8],
        [7, 7, 7, 7],
        [5, 5, 5, 5],
    ];
    fn pbits(mode: u16) -> bool {
        matches!(mode, 0 | 1 | 3 | 6 | 7)
    }
    let (mode, rot) = (ctrl & 7, usize::from(ctrl >> 3 & 3));
    let mut p = BC7_BITS[usize::from(mode)];
    if rot != 0 {
        p.swap(3, rot - 1);
    } else if pbits(mode) {
        p[0] += 1;
    }
    p
}

fn bc7_precision_body() -> Vec<u32> {
    let vcc = sgpr(106);
    // The table's rows as little-endian dwords, and the modes with a p-bit as a bit set.
    const ROWS: [u32; 8] = [
        0x0004_0404,
        0x0006_0606,
        0x0005_0505,
        0x0007_0707,
        0x0605_0505,
        0x0807_0707,
        0x0707_0707,
        0x0505_0505,
    ];
    let mut body = Vec::new();
    // v1 = the control word; v2 = its mode, v3 = its rotation, v5 = the mode's row.
    body.extend(vop3(0x148, 2, vgpr(1), 128, 128 + 3)); // v_bfe_u32 v2, v1, 0, 3
    body.extend(vop3(0x148, 3, vgpr(1), 128 + 3, 128 + 2)); // v_bfe_u32 v3, v1, 3, 2
    body.extend([vop1(0x001, 5, 255), ROWS[0]]); // v_mov_b32 v5, row 0
    for (mode, row) in ROWS.iter().enumerate().skip(1) {
        body.extend([vop1(0x001, 6, 255), *row]); // v_mov_b32 v6, row
        body.extend(vop3(0x0C2, 106, 128 + mode as u16, vgpr(2), 128)); // v_cmp_eq_u32 vcc, mode, v2
        body.extend(vop3(0x101, 5, vgpr(5), vgpr(6), vcc)); // v_cndmask_b32 v5, v5, v6, vcc
    }
    // No rotation: red gains the p-bit. v7 = the row with it.
    body.extend([vop1(0x001, 7, 255), 0b1100_1011]); // v_mov_b32 v7, modes 0, 1, 3, 6, 7
    body.extend(vop3(0x116, 7, vgpr(2), vgpr(7), 128)); // v_lshrrev_b32 v7, v2, v7
    body.extend(vop3(0x148, 7, vgpr(7), 128, 129)); // v_bfe_u32 v7, v7, 0, 1
    body.extend(vop3(0x125, 7, vgpr(5), vgpr(7), 128)); // v_add_nc_u32 v7, v5, v7
    // A rotation swaps alpha with channel `rot - 1`: v8 = that channel's shift, v9 = the
    // two bytes' difference, v10 = the row with them exchanged.
    body.extend(vop3(0x125, 8, 193, vgpr(3), 128)); // v_add_nc_u32 v8, -1, v3
    body.extend(vop3(0x11A, 8, 128 + 3, vgpr(8), 128)); // v_lshlrev_b32 v8, 3, v8
    body.extend(vop3(0x116, 9, vgpr(8), vgpr(5), 128)); // v_lshrrev_b32 v9, v8, v5
    body.extend(vop3(0x148, 9, vgpr(9), 128, 128 + 8)); // v_bfe_u32 v9, v9, 0, 8
    body.extend(vop3(0x148, 10, vgpr(5), 128 + 24, 128 + 8)); // v_bfe_u32 v10, v5, 24, 8
    body.extend(vop3(0x11D, 9, vgpr(9), vgpr(10), 128)); // v_xor_b32 v9, v9, v10
    body.extend(vop3(0x11A, 10, vgpr(8), vgpr(9), 128)); // v_lshlrev_b32 v10, v8, v9
    body.extend(vop3(0x11D, 10, vgpr(5), vgpr(10), 128)); // v_xor_b32 v10, v5, v10
    body.extend(vop3(0x11A, 9, 128 + 24, vgpr(9), 128)); // v_lshlrev_b32 v9, 24, v9
    body.extend(vop3(0x11D, 10, vgpr(10), vgpr(9), 128)); // v_xor_b32 v10, v10, v9
    body.extend(vop3(0x0C5, 106, 128, vgpr(3), 128)); // v_cmp_ne_u32 vcc, 0, v3
    body.extend(vop3(0x101, 0, vgpr(7), vgpr(10), vcc)); // v_cndmask_b32 v0, v7, v10, vcc
    body
}

/// BC7's channel precisions for a control word, written as console code, match the
/// formats crate's function on the GPU for every mode and rotation.
#[test]
fn bcpack_bc7_precisions_match_the_formats_function() {
    let shader = build_valu_cs(&bc7_precision_body());
    // Every mode and rotation, alone and under each of the control word's higher bits.
    let cases: Vec<[u32; 4]> = (0u32..32)
        .flat_map(|low| [0u32, 0x20, 0x8000, 0xFFE0].map(|high| [high | low, 0, 0, 0]))
        .collect();
    run_cases(&shader, &cases, |backend, _, inputs, got| {
        let want = u32::from_le_bytes(bc7_precision_reference(inputs[0] as u16));
        assert_eq!(got, want, "{backend}: bc7_precision({:#x})", inputs[0]);
    });
}

/// `victory_formats::bcpack::pick` with its history length, the function the test below
/// runs as console code.
fn pick_reference<'a, T>(
    i: usize,
    r: usize,
    history: &'a [T],
    dictionary: &'a [T],
    zero: &'a T,
) -> &'a T {
    const HISTORY: usize = 256;
    match r {
        0 => zero,
        r if r <= i => &history[(i - r) % HISTORY],
        r => &dictionary[r - i - 1],
    }
}

/// BCPack's choice of a block's reference, written as console code, matches the formats
/// crate's function on the GPU: none, a block of the history ring, or a dictionary entry
/// before the first block, on both sides of each boundary.
#[test]
fn bcpack_reference_choice_matches_the_formats_function() {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = the block, v2 = its reference. The result names the source in its high half
    // (1 the history, 2 the dictionary) and the entry in its low half.
    body.extend(vop3(0x126, 5, vgpr(1), vgpr(2), 128)); // v_sub_nc_u32 v5, v1, v2
    body.extend(vop3(0x148, 5, vgpr(5), 128, 128 + 8)); // v_bfe_u32 v5, v5, 0, 8
    body.extend(vop3(0x36F, 5, 129, 128 + 16, vgpr(5))); // v_lshl_or_b32 v5, 1, 16, v5
    body.extend(vop3(0x126, 6, vgpr(2), vgpr(1), 128)); // v_sub_nc_u32 v6, v2, v1
    body.extend(vop3(0x125, 6, 193, vgpr(6), 128)); // v_add_nc_u32 v6, -1, v6
    body.extend(vop3(0x36F, 6, 130, 128 + 16, vgpr(6))); // v_lshl_or_b32 v6, 2, 16, v6
    body.extend(vop3(0x0C3, 106, vgpr(2), vgpr(1), 128)); // v_cmp_le_u32 vcc, v2, v1
    body.extend(vop3(0x101, 7, vgpr(6), vgpr(5), vcc)); // v_cndmask_b32 v7, v6, v5, vcc
    body.extend(vop3(0x0C2, 106, 128, vgpr(2), 128)); // v_cmp_eq_u32 vcc, 0, v2
    body.extend(vop3(0x101, 0, vgpr(7), 128, vcc)); // v_cndmask_b32 v0, v7, 0, vcc
    let shader = build_valu_cs(&body);
    // Entries that say where they are, so the function's choice reads as the result does.
    let history: Vec<u32> = (0..256).map(|k| 1 << 16 | k).collect();
    let dictionary: Vec<u32> = (0..255).map(|k| 2 << 16 | k).collect();
    // Blocks and references on both sides of `r == 0`, `r == i` and the ring's length.
    let cases: Vec<[u32; 4]> = [0u32, 1, 2, 254, 255, 256, 1000]
        .into_iter()
        .flat_map(|i| [0u32, 1, 2, 3, 128, 254, 255].map(|r| [i, r, 0, 0]))
        .collect();
    run_cases(&shader, &cases, |backend, _, [i, r, ..], got| {
        let want = *pick_reference(i as usize, r as usize, &history, &dictionary, &0);
        assert_eq!(got, want, "{backend}: pick(block {i}, reference {r})");
    });
}

/// `victory_formats::bcpack::bc6_index_bits`, the function the test below runs as console
/// code.
fn bc6_index_bits_reference(mode: u8) -> u8 {
    match mode {
        1..=10 => 3,
        11..=14 => 4,
        _ => 0,
    }
}

/// BC6H's index width for a mode, written as console code, matches the formats crate's
/// function on the GPU for every byte.
#[test]
fn bcpack_bc6_index_width_matches_the_formats_function() {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = the mode. Two range tests as unsigned compares of the mode less the range's start.
    body.extend(vop3(0x125, 5, 193, vgpr(1), 128)); // v_add_nc_u32 v5, -1, v1
    body.extend(vop3(0x0C4, 106, 128 + 10, vgpr(5), 128)); // v_cmp_gt_u32 vcc, 10, v5
    body.extend(vop3(0x101, 6, 128, 128 + 3, vcc)); // v_cndmask_b32 v6, 0, 3, vcc
    body.extend(vop3(0x126, 5, vgpr(1), 128 + 11, 128)); // v_sub_nc_u32 v5, v1, 11
    body.extend(vop3(0x0C4, 106, 128 + 4, vgpr(5), 128)); // v_cmp_gt_u32 vcc, 4, v5
    body.extend(vop3(0x101, 7, 128, 128 + 4, vcc)); // v_cndmask_b32 v7, 0, 4, vcc
    body.extend(vop3(0x11C, 0, vgpr(6), vgpr(7), 128)); // v_or_b32 v0, v6, v7
    let shader = build_valu_cs(&body);
    let cases: Vec<[u32; 4]> = (0u32..256).map(|mode| [mode, 0, 0, 0]).collect();
    run_cases(&shader, &cases, |backend, _, inputs, got| {
        let want = u32::from(bc6_index_bits_reference(inputs[0] as u8));
        assert_eq!(got, want, "{backend}: bc6_index_bits({})", inputs[0]);
    });
}

/// `victory_formats::bcpack::quads`, the function the test below runs as console code.
fn quads_reference(rank: &[u8; 256], s: &[u8], ix: &mut [u8]) {
    for (q, &sym) in s.iter().enumerate() {
        let v = rank[usize::from(sym)];
        let base = (q >> 1) * 8 + (q & 1) * 2;
        for (k, p) in [0, 1, 4, 5].into_iter().enumerate() {
            ix[base + p] = v >> (2 * k) & 3;
        }
    }
}

/// BCPack's 2-bit index residuals, one symbol per 2x2 quad, written as console code, match
/// the formats crate's function on the GPU: each quad's four residuals and where the quad
/// starts in the block row, for a rank table and a symbol order that are both permutations.
#[test]
fn bcpack_quad_residuals_match_the_formats_function() {
    let rank: [u8; 256] = std::array::from_fn(|i| (i * 167 + 13) as u8);
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = the symbol, v2 = the quad's number; v3 = the symbol's dword in the table, v5 =
    // that dword.
    body.extend(vop3(0x148, 1, vgpr(1), 128, 128 + 8)); // v_bfe_u32 v1, v1, 0, 8
    body.extend(vop3(0x116, 3, 128 + 2, vgpr(1), 128)); // v_lshrrev_b32 v3, 2, v1
    for (d, row) in rank.as_chunks::<4>().0.iter().enumerate() {
        let reg = if d == 0 { 5 } else { 6 };
        body.extend([vop1(0x001, reg, 255), u32::from_le_bytes(*row)]); // v_mov_b32 reg, row
        if d != 0 {
            body.extend(vop3(0x0C2, 106, 128 + d as u16, vgpr(3), 128)); // v_cmp_eq_u32 vcc, d, v3
            body.extend(vop3(0x101, 5, vgpr(5), vgpr(6), vcc)); // v_cndmask_b32 v5, v5, v6, vcc
        }
    }
    // The entry's byte, then its four residuals, one to a nibble of the result.
    body.extend(vop3(0x148, 7, vgpr(1), 128, 128 + 2)); // v_bfe_u32 v7, v1, 0, 2
    body.extend(vop3(0x11A, 7, 128 + 3, vgpr(7), 128)); // v_lshlrev_b32 v7, 3, v7
    body.extend(vop3(0x116, 8, vgpr(7), vgpr(5), 128)); // v_lshrrev_b32 v8, v7, v5
    body.extend(vop3(0x148, 0, vgpr(8), 128, 128 + 2)); // v_bfe_u32 v0, v8, 0, 2
    for k in 1u16..4 {
        body.extend(vop3(0x148, 9, vgpr(8), 128 + 2 * k, 128 + 2)); // v_bfe_u32 v9, v8, 2k, 2
        body.extend(vop3(0x36F, 0, vgpr(9), 128 + 4 * k, vgpr(0))); // v_lshl_or_b32 v0, v9, 4k, v0
    }
    // Where the quad starts: eight pixels per pair of quads, two for the odd one. It goes
    // in the result's high half.
    body.extend(vop3(0x116, 10, 129, vgpr(2), 128)); // v_lshrrev_b32 v10, 1, v2
    body.extend(vop3(0x148, 11, vgpr(2), 128, 129)); // v_bfe_u32 v11, v2, 0, 1
    body.extend(vop3(0x11A, 11, 129, vgpr(11), 128)); // v_lshlrev_b32 v11, 1, v11
    body.extend(vop3(0x36F, 10, vgpr(10), 128 + 3, vgpr(11))); // v_lshl_or_b32 v10, v10, 3, v11
    body.extend(vop3(0x36F, 0, vgpr(10), 128 + 16, vgpr(0))); // v_lshl_or_b32 v0, v10, 16, v0
    let shader = build_valu_cs(&body);
    let symbols: Vec<u8> = (0..256).map(|k| (k * 91 + 7) as u8).collect();
    let mut ix = vec![0xFFu8; 4 * symbols.len()];
    quads_reference(&rank, &symbols, &mut ix);
    let cases: Vec<[u32; 4]> = symbols
        .iter()
        .enumerate()
        .map(|(q, &s)| [u32::from(s), q as u32, 0, 0])
        .collect();
    // Each backend must read every cell the function wrote, each once.
    let mut read = std::collections::BTreeMap::new();
    run_cases(&shader, &cases, |backend, q, _, got| {
        let base = (got >> 16) as usize;
        let seen: &mut Vec<bool> = read
            .entry(backend.to_string())
            .or_insert(vec![false; ix.len()]);
        for (k, p) in [0, 1, 4, 5].into_iter().enumerate() {
            let cell = base + p;
            assert!(
                cell < ix.len() && !seen[cell],
                "{backend}: quad {q} lands on cell {cell}"
            );
            seen[cell] = true;
            assert_eq!(
                got >> (4 * k) & 15,
                u32::from(ix[cell]),
                "{backend}: quad {q}, pixel {k}"
            );
        }
    });
    for (backend, seen) in read {
        assert!(seen.iter().all(|&s| s), "{backend}: cells never read");
    }
}

/// `victory_formats::bcpack::pairs`, the function the test below runs as console code.
fn pairs_reference(rank: &[u8; 64], s: &[u8], ix: &mut [u8]) {
    for (k, &sym) in s.iter().enumerate() {
        let v = rank[usize::from(sym & 63)];
        ix[2 * k] = v & 7;
        ix[2 * k + 1] = v >> 4 & 7;
    }
}

/// BCPack's 3-bit index residuals, one symbol per two pixels, written as console code,
/// match the formats crate's function on the GPU: a rank table that tells the two pixels
/// apart, read through every symbol, the symbols' two unused bits included.
#[test]
fn bcpack_pixel_pair_residuals_match_the_formats_function() {
    // No entry gives its two pixels the same residual, and the pair is never symmetric.
    let rank: [u8; 64] = std::array::from_fn(|i| ((i % 8) | (((i * 3 + 1) % 8) << 4)) as u8);
    assert!(rank.iter().all(|v| v & 7 != v >> 4 & 7));
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = the symbol; v2 = its table index, v3 = the index's dword, v5 = that dword.
    body.extend(vop3(0x148, 2, vgpr(1), 128, 128 + 6)); // v_bfe_u32 v2, v1, 0, 6
    body.extend(vop3(0x116, 3, 128 + 2, vgpr(2), 128)); // v_lshrrev_b32 v3, 2, v2
    for (d, row) in rank.as_chunks::<4>().0.iter().enumerate() {
        let reg = if d == 0 { 5 } else { 6 };
        body.extend([vop1(0x001, reg, 255), u32::from_le_bytes(*row)]); // v_mov_b32 reg, row
        if d != 0 {
            body.extend(vop3(0x0C2, 106, 128 + d as u16, vgpr(3), 128)); // v_cmp_eq_u32 vcc, d, v3
            body.extend(vop3(0x101, 5, vgpr(5), vgpr(6), vcc)); // v_cndmask_b32 v5, v5, v6, vcc
        }
    }
    // The entry's byte, then its two residuals, the second pixel's in the next byte.
    body.extend(vop3(0x148, 7, vgpr(2), 128, 128 + 2)); // v_bfe_u32 v7, v2, 0, 2
    body.extend(vop3(0x11A, 7, 128 + 3, vgpr(7), 128)); // v_lshlrev_b32 v7, 3, v7
    body.extend(vop3(0x116, 8, vgpr(7), vgpr(5), 128)); // v_lshrrev_b32 v8, v7, v5
    body.extend(vop3(0x148, 9, vgpr(8), 128, 128 + 3)); // v_bfe_u32 v9, v8, 0, 3
    body.extend(vop3(0x148, 10, vgpr(8), 128 + 4, 128 + 3)); // v_bfe_u32 v10, v8, 4, 3
    body.extend(vop3(0x36F, 0, vgpr(10), 128 + 8, vgpr(9))); // v_lshl_or_b32 v0, v10, 8, v9
    let shader = build_valu_cs(&body);
    let symbols: Vec<u8> = (0..=255).collect();
    let mut ix = vec![0u8; 2 * symbols.len()];
    pairs_reference(&rank, &symbols, &mut ix);
    let cases: Vec<[u32; 4]> = symbols.iter().map(|&s| [u32::from(s), 0, 0, 0]).collect();
    run_cases(&shader, &cases, |backend, k, inputs, got| {
        let want = u32::from(ix[2 * k]) | u32::from(ix[2 * k + 1]) << 8;
        assert_eq!(got, want, "{backend}: pairs, symbol {}", inputs[0]);
    });
}

/// A select on an inline 0 or -1 mask picks src0 or src1 in every lane.
#[test]
fn v_cndmask_on_an_inline_mask_picks_one_side() {
    for (mask, pick) in [(128u16, 0usize), (193, 1)] {
        let k = build_valu_cs(&vop3(0x101, 0, vgpr(1), vgpr(2), mask));
        run_cases_on(
            &k,
            &[[10, 20, 0, 0]],
            &[("hardware", false)],
            |label, _, inp, got| {
                assert_eq!(got, inp[pick], "{label} inline {mask}");
            },
        );
    }
}

/// `victory_formats::xma::log2`, the function the test below runs as console code.
fn xma_log2_reference(v: usize) -> u32 {
    usize::BITS - 1 - v.max(1).leading_zeros()
}

/// XMA's integer base-2 logarithm, written as console code, matches the formats crate's
/// function on the GPU: zero, every power of two and both of its neighbours.
#[test]
fn xma_integer_log2_matches_the_formats_function() {
    let mut body = Vec::new();
    // v1 = the value; zero counts as one.
    body.extend(vop3(0x114, 5, vgpr(1), 129, 128)); // v_max_u32 v5, v1, 1
    body.push(vop1(0x039, 6, vgpr(5))); // v_ffbh_u32 v6, v5
    body.extend(vop3(0x126, 0, 128 + 31, vgpr(6), 128)); // v_sub_nc_u32 v0, 31, v6
    let shader = build_valu_cs(&body);
    let mut values = vec![0u32, u32::MAX];
    for bit in 0..32 {
        let p = 1u32 << bit;
        values.extend([p.wrapping_sub(1), p, p.wrapping_add(1)]);
    }
    let cases: Vec<[u32; 4]> = values.into_iter().map(|v| [v, 0, 0, 0]).collect();
    run_cases(&shader, &cases, |backend, _, inputs, got| {
        let want = xma_log2_reference(inputs[0] as usize);
        assert_eq!(got, want, "{backend}: log2({:#x})", inputs[0]);
    });
}

/// `victory_formats::xma::to_i16`, the function the test below runs as console code.
fn xma_to_i16_reference(v: f32) -> i16 {
    (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16
}

/// XMA's sample conversion from float to 16 bits, written as console code, matches the
/// formats crate's function on the GPU: ties round away from zero, both ends saturate.
#[test]
fn xma_sample_to_i16_matches_the_formats_function() {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = the sample. v5 = it scaled, v6 = that truncated, v7 = the fraction cut off.
    body.extend([vop1(0x001, 5, 255), 32768f32.to_bits()]); // v_mov_b32 v5, 32768.0
    body.extend(vop3(0x108, 5, vgpr(1), vgpr(5), 128)); // v_mul_f32 v5, v1, v5
    body.push(vop1(0x021, 6, vgpr(5))); // v_trunc_f32 v6, v5
    body.extend(vop3(0x104, 7, vgpr(5), vgpr(6), 128)); // v_sub_f32 v7, v5, v6
    // A fraction of a half or more moves one further from zero.
    body.extend(vop3(0x103, 8, vgpr(6), 242, 128)); // v_add_f32 v8, v6, 1.0
    body.extend(vop3(0x006, 106, vgpr(7), 240, 128)); // v_cmp_ge_f32 vcc, v7, 0.5
    body.extend(vop3(0x101, 9, vgpr(6), vgpr(8), vcc)); // v_cndmask_b32 v9, v6, v8, vcc
    body.extend(vop3(0x103, 8, vgpr(6), 243, 128)); // v_add_f32 v8, v6, -1.0
    body.extend(vop3(0x003, 106, vgpr(7), 241, 128)); // v_cmp_le_f32 vcc, v7, -0.5
    body.extend(vop3(0x101, 9, vgpr(9), vgpr(8), vcc)); // v_cndmask_b32 v9, v9, v8, vcc
    // Saturate, then convert.
    body.extend([vop1(0x001, 10, 255), (-32768f32).to_bits()]); // v_mov_b32 v10, -32768.0
    body.extend(vop3(0x110, 9, vgpr(9), vgpr(10), 128)); // v_max_f32 v9, v9, v10
    body.extend([vop1(0x001, 10, 255), 32767f32.to_bits()]); // v_mov_b32 v10, 32767.0
    body.extend(vop3(0x10F, 9, vgpr(9), vgpr(10), 128)); // v_min_f32 v9, v9, v10
    body.push(vop1(0x008, 0, vgpr(9))); // v_cvt_i32_f32 v0, v9
    // Not a number converts to zero: `v_max_f32` alone would have saturated it.
    body.extend(vop3(0x008, 106, vgpr(1), vgpr(1), 128)); // v_cmp_u_f32 vcc, v1, v1
    body.extend(vop3(0x101, 0, vgpr(0), 128, vcc)); // v_cndmask_b32 v0, v0, 0, vcc
    let shader = build_valu_cs(&body);
    let step = 1.0 / 32768.0;
    let mut samples = vec![
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    // Ties and their neighbours on both sides of zero, and just inside each end.
    for k in [0.5f32, 1.5, 2.5, 3.5, 32766.5, 32767.5] {
        for x in [
            k,
            f32::from_bits(k.to_bits() - 1),
            f32::from_bits(k.to_bits() + 1),
        ] {
            samples.extend([x * step, -x * step]);
        }
    }
    samples.extend((0..64).map(|i| (i as f32 * 0.618_034).sin() * 1.1));
    let cases: Vec<[u32; 4]> = samples.iter().map(|s| [s.to_bits(), 0, 0, 0]).collect();
    let mut wrong = Vec::new();
    run_cases(&shader, &cases, |backend, _, inputs, got| {
        let v = f32::from_bits(inputs[0]);
        let want = xma_to_i16_reference(v) as i32 as u32;
        if got != want {
            wrong.push(format!(
                "{backend}: to_i16({v:e}) = {} want {}",
                got as i32, want as i32
            ));
        }
    });
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// `victory_formats::texture::dim`, one of the two functions the test below runs as
/// console code.
fn texture_dim_reference(texels: u32, mip: u32, block: u32) -> u32 {
    (texels >> mip).max(1).div_ceil(block)
}

/// `victory_formats::texture::padded_dim`, the other one.
fn texture_padded_dim_reference(texels: u32, mip: u32, block: u32) -> u32 {
    texels.max(1).div_ceil(block).div_ceil(1 << mip)
}

/// A mip's size in elements, as data sees it and as placement pads it, written as console
/// code, matches the formats crate's two functions on the GPU for texel and compressed
/// block sizes. The block is a power of two, so each division is a shift.
#[test]
fn texture_mip_sizes_match_the_formats_functions() {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = texels, v2 = the mip, v3 = the block, v4 = which size. v5 = the block's shift,
    // v6 = the block less one.
    body.push(vop1(0x03A, 5, vgpr(3))); // v_ffbl_b32 v5, v3
    body.extend(vop3(0x125, 6, 193, vgpr(3), 128)); // v_add_nc_u32 v6, -1, v3
    // As data sees it: halve rounding down, at least one, round up to whole elements.
    body.extend(vop3(0x116, 7, vgpr(2), vgpr(1), 128)); // v_lshrrev_b32 v7, v2, v1
    body.extend(vop3(0x114, 7, vgpr(7), 129, 128)); // v_max_u32 v7, v7, 1
    body.extend(vop3(0x125, 7, vgpr(7), vgpr(6), 128)); // v_add_nc_u32 v7, v7, v6
    body.extend(vop3(0x116, 7, vgpr(5), vgpr(7), 128)); // v_lshrrev_b32 v7, v5, v7
    // As placement pads it: mip 0's elements, halved rounding up.
    body.extend(vop3(0x114, 8, vgpr(1), 129, 128)); // v_max_u32 v8, v1, 1
    body.extend(vop3(0x125, 8, vgpr(8), vgpr(6), 128)); // v_add_nc_u32 v8, v8, v6
    body.extend(vop3(0x116, 8, vgpr(5), vgpr(8), 128)); // v_lshrrev_b32 v8, v5, v8
    body.extend(vop3(0x11A, 9, vgpr(2), 129, 128)); // v_lshlrev_b32 v9, v2, 1
    body.extend(vop3(0x125, 9, 193, vgpr(9), 128)); // v_add_nc_u32 v9, -1, v9
    body.extend(vop3(0x125, 8, vgpr(8), vgpr(9), 128)); // v_add_nc_u32 v8, v8, v9
    body.extend(vop3(0x116, 8, vgpr(2), vgpr(8), 128)); // v_lshrrev_b32 v8, v2, v8
    body.extend(vop3(0x0C5, 106, 128, vgpr(4), 128)); // v_cmp_ne_u32 vcc, 0, v4
    body.extend(vop3(0x101, 0, vgpr(7), vgpr(8), vcc)); // v_cndmask_b32 v0, v7, v8, vcc
    let shader = build_valu_cs(&body);
    let mut cases = Vec::new();
    for texels in [0u32, 1, 2, 3, 5, 8, 9, 100, 255, 257, 4096, 16383] {
        for mip in [0u32, 1, 2, 3, 5, 13] {
            for block in [1u32, 4] {
                cases.extend([[texels, mip, block, 0], [texels, mip, block, 1]]);
            }
        }
    }
    // The two sizes must differ somewhere, or the test could not tell them apart.
    assert!(cases.iter().any(|&[t, m, b, _]| {
        texture_dim_reference(t, m, b) != texture_padded_dim_reference(t, m, b)
    }));
    run_cases(
        &shader,
        &cases,
        |backend, _, [texels, mip, block, padded], got| {
            let want = if padded == 0 {
                texture_dim_reference(texels, mip, block)
            } else {
                texture_padded_dim_reference(texels, mip, block)
            };
            assert_eq!(
                got, want,
                "{backend}: {texels} texels, mip {mip}, block {block}, padded {padded}"
            );
        },
    );
}

/// `victory_formats::texture::narrowed_view`, the function the test below runs
/// as console code.
fn texture_narrowed_view_reference(
    asked: (u32, u32),
    resident: (u32, u32),
    mips: u32,
) -> (u32, u32) {
    let (first, levels) = asked;
    let end = if levels == 0 || levels == u32::MAX {
        mips
    } else {
        first.saturating_add(levels).min(mips)
    };
    let (rfirst, rcount) = resident;
    let rend = rfirst.saturating_add(rcount).min(mips);
    // Nothing the view asked for is there: its most detailed resident mip
    // alone (the picture is the smallest mip, not black).
    let start = first.max(rfirst).min(rend.saturating_sub(1));
    let stop = end.min(rend);
    let count = stop.saturating_sub(start).max(1);
    (start, count)
}

/// Narrowing a shader view's mip range to the resident mips, written as console
/// code, matches `victory_formats::texture::narrowed_view` on the GPU:
/// saturating integer add/sub (`v_add_nc_u32`/`v_sub_nc_u32` with `clamp`),
/// sentinel checks (`0` and `u32::MAX`) joined with `s_or_b32`, and packing the
/// resulting `(start, count)` pair with `v_lshl_or_b32`.
#[test]
fn texture_narrowed_view_matches_the_formats_function() {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = first, v2 = levels, v3 = rfirst | (rcount << 16), v4 = mips.
    // Unpack rfirst -> v5, rcount -> v6.
    body.extend(vop3(0x148, 5, vgpr(3), 128, 144)); // v_bfe_u32 v5, v3, 0, 16
    body.extend(vop3(0x148, 6, vgpr(3), 144, 144)); // v_bfe_u32 v6, v3, 16, 16
    // end = if levels == 0 || levels == u32::MAX { mips } else { first.saturating_add(levels).min(mips) }.
    body.extend(vop3_clamp(0x125, 7, vgpr(1), vgpr(2), 128)); // v_add_nc_u32 v7, v1, v2 clamp
    body.extend(vop3(0x113, 7, vgpr(7), vgpr(4), 128)); // v_min_u32 v7, v7, v4
    body.extend(vop3(0x0C2, 16, vgpr(2), 128, 128)); // v_cmp_eq_u32 s16, v2, 0
    body.extend(vop3(0x0C2, 17, vgpr(2), 193, 128)); // v_cmp_eq_u32 s17, v2, -1
    body.push(sop2(0x10, 106, 16, 17)); // s_or_b32 vcc_lo, s16, s17
    body.extend(vop3(0x101, 7, vgpr(7), vgpr(4), vcc)); // v_cndmask_b32 v7, v7, v4, vcc
    // rend = rfirst.saturating_add(rcount).min(mips) -> v6.
    body.extend(vop3_clamp(0x125, 6, vgpr(5), vgpr(6), 128)); // v_add_nc_u32 v6, v5, v6 clamp
    body.extend(vop3(0x113, 6, vgpr(6), vgpr(4), 128)); // v_min_u32 v6, v6, v4
    // start = first.max(rfirst).min(rend.saturating_sub(1)) -> v5.
    body.extend(vop3(0x114, 5, vgpr(1), vgpr(5), 128)); // v_max_u32 v5, v1, v5
    body.extend(vop3_clamp(0x126, 8, vgpr(6), 129, 128)); // v_sub_nc_u32 v8, v6, 1 clamp
    body.extend(vop3(0x113, 5, vgpr(5), vgpr(8), 128)); // v_min_u32 v5, v5, v8
    // stop = end.min(rend); count = stop.saturating_sub(start).max(1) -> v7.
    body.extend(vop3(0x113, 7, vgpr(7), vgpr(6), 128)); // v_min_u32 v7, v7, v6
    body.extend(vop3_clamp(0x126, 7, vgpr(7), vgpr(5), 128)); // v_sub_nc_u32 v7, v7, v5 clamp
    body.extend(vop3(0x114, 7, vgpr(7), 129, 128)); // v_max_u32 v7, v7, 1
    // Pack (start << 16) | count into v0.
    body.extend(vop3(0x36F, 0, vgpr(5), 144, vgpr(7))); // v_lshl_or_b32 v0, v5, 16, v7

    let shader = build_valu_cs(&body);
    let mut cases = Vec::new();
    for mips in [0u32, 1, 6, 12] {
        for &(rfirst, rcount) in &[(0u32, 0u32), (0, 12), (2, 3), (5, 7)] {
            for first in [0u32, 2, 5, u32::MAX - 2] {
                for levels in [0u32, 1, 3, u32::MAX - 1, u32::MAX] {
                    cases.push([first, levels, rfirst | (rcount << 16), mips]);
                }
            }
        }
    }
    // Sanity-check that cases exercise overflow saturation, underflow saturation,
    // both sentinel values, and non-overlapping ranges.
    assert!(cases.iter().any(|&[first, levels, _, mips]| {
        levels != 0 && levels != u32::MAX && first.checked_add(levels).is_none() && mips > 0
    }));
    assert!(cases.iter().any(|&[first, levels, packed, mips]| {
        let (rfirst, rcount) = (packed & 0xFFFF, packed >> 16);
        let (start, count) =
            texture_narrowed_view_reference((first, levels), (rfirst, rcount), mips);
        start > first && count == 1
    }));
    run_cases(
        &shader,
        &cases,
        |backend, _, [first, levels, packed, mips], got| {
            let (rfirst, rcount) = (packed & 0xFFFF, packed >> 16);
            let (want_start, want_count) =
                texture_narrowed_view_reference((first, levels), (rfirst, rcount), mips);
            let want = (want_start << 16) | want_count;
            assert_eq!(
                got,
                want,
                "{backend}: asked=({first}, {levels}) resident=({rfirst}, {rcount}) mips={mips}: got ({}, {}), want ({want_start}, {want_count})",
                got >> 16,
                got & 0xFFFF
            );
        },
    );
}

/// `victory_formats::texture::Swizzle::block_log2`, the function the test below
/// runs as console code (`u32::MAX` encodes `None`).
fn texture_swizzle_block_log2_reference(mode: u32) -> Option<u32> {
    Some(match mode {
        0 => 0,
        1 | 2 => 8,
        5 | 6 | 21..=23 => 12,
        9 | 10 | 17 | 18 | 24..=27 => 16,
        _ => return None,
    })
}

/// Mapping a GFX10 swizzle mode number to log2 of its block size in bytes,
/// written as console code using 32-bit literal bitmasks and shift-wrap guarding
/// (`v_lshlrev_b32`, `v_cmp_lt_u32`, `v_and_b32`, `v_cndmask_b32`), matches
/// `victory_formats::texture::Swizzle::block_log2` on the GPU.
#[test]
fn texture_swizzle_block_log2_matches_the_formats_function() {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = mode. v5 = if mode < 32 { 1u << mode } else { 0 } (guards v_lshlrev_b32 & 31 wrap).
    body.extend(vop3(0x11A, 5, vgpr(1), 129, 128)); // v_lshlrev_b32 v5, v1, 1
    body.extend(vop3(0x0C1, 106, vgpr(1), 160, 128)); // v_cmp_lt_u32 vcc, v1, 32
    body.extend(vop3(0x101, 5, 128, vgpr(5), vcc)); // v_cndmask_b32 v5, 0, v5, vcc
    // Default result: u32::MAX (None).
    body.push(vop1(0x001, 0, 193)); // v_mov_b32 v0, -1
    // mode == 0 -> 0 (linear).
    body.extend(vop3(0x0C2, 106, vgpr(1), 128, 128)); // v_cmp_eq_u32 vcc, v1, 0
    body.extend(vop3(0x101, 0, vgpr(0), 128, vcc)); // v_cndmask_b32 v0, v0, 0, vcc
    // 256 B (log2 = 8): modes 1, 2.
    body.extend([vop1(0x001, 2, 255), 0x0000_0006]); // v_mov_b32 v2, 0x00000006
    body.extend(vop3(0x11B, 6, vgpr(5), vgpr(2), 128)); // v_and_b32 v6, v5, v2
    body.extend(vop3(0x0C5, 106, vgpr(6), 128, 128)); // v_cmp_ne_u32 vcc, v6, 0
    body.extend(vop3(0x101, 0, vgpr(0), 136, vcc)); // v_cndmask_b32 v0, v0, 8, vcc
    // 4 KiB (log2 = 12): modes 5, 6, 21..=23.
    body.extend([vop1(0x001, 2, 255), 0x00E0_0060]); // v_mov_b32 v2, 0x00E00060
    body.extend(vop3(0x11B, 6, vgpr(5), vgpr(2), 128)); // v_and_b32 v6, v5, v2
    body.extend(vop3(0x0C5, 106, vgpr(6), 128, 128)); // v_cmp_ne_u32 vcc, v6, 0
    body.extend(vop3(0x101, 0, vgpr(0), 140, vcc)); // v_cndmask_b32 v0, v0, 12, vcc
    // 64 KiB (log2 = 16): modes 9, 10, 17, 18, 24..=27.
    body.extend([vop1(0x001, 2, 255), 0x0F06_0600]); // v_mov_b32 v2, 0x0F060600
    body.extend(vop3(0x11B, 6, vgpr(5), vgpr(2), 128)); // v_and_b32 v6, v5, v2
    body.extend(vop3(0x0C5, 106, vgpr(6), 128, 128)); // v_cmp_ne_u32 vcc, v6, 0
    body.extend(vop3(0x101, 0, vgpr(0), 144, vcc)); // v_cndmask_b32 v0, v0, 16, vcc

    let shader = build_valu_cs(&body);
    let mut cases: Vec<[u32; 4]> = (0..=64u32).map(|m| [m, 0, 0, 0]).collect();
    cases.extend([[128, 0, 0, 0], [255, 0, 0, 0], [u32::MAX, 0, 0, 0]]);
    // Confirm every outcome (0, 8, 12, 16, and None including wrapped >= 32 modes) is covered.
    for expected in [Some(0), Some(8), Some(12), Some(16), None] {
        assert!(
            cases
                .iter()
                .any(|&[m, ..]| texture_swizzle_block_log2_reference(m) == expected)
        );
    }
    run_cases(&shader, &cases, |backend, _, [mode, ..], got| {
        let want = texture_swizzle_block_log2_reference(mode).unwrap_or(u32::MAX);
        assert_eq!(got, want, "{backend}: swizzle mode {mode}");
    });
}

/// `victory_formats::bcpack::references`, the function the test below runs as
/// console code.
fn bcpack_references_reference(s: [u8; 4], paired: bool) -> (usize, usize) {
    let (d, ix) = if paired {
        (
            interleave_reference(s[0], s[1]),
            interleave_reference(s[2], s[3]),
        )
    } else {
        (s[0], s[1])
    };
    let delta = (d >> 1) ^ (d & 1).wrapping_neg();
    (usize::from(ix.wrapping_add(delta)), usize::from(ix))
}

/// Emits RDNA2 instructions to interleave two 4-bit symbols (`a` on even bits,
/// `b` on odd bits) into `dst`, using `mask55` (= `0x55`) and `tmp` as scratch.
fn emit_interleave_pair(body: &mut Vec<u32>, dst: u8, a: u8, b: u8, mask55: u8, tmp: u8) {
    // Spread a's low 4 bits to even positions -> dst.
    body.extend(vop3(0x11B, dst, vgpr(a), 143, 128)); // v_and_b32 dst, a, 15
    body.extend(vop3(0x36F, tmp, vgpr(dst), 130, vgpr(dst))); // v_lshl_or_b32 tmp, dst, 2, dst
    body.extend(vop3(0x11B, dst, vgpr(tmp), 179, 128)); // v_and_b32 dst, tmp, 0x33 (51)
    body.extend(vop3(0x36F, tmp, vgpr(dst), 129, vgpr(dst))); // v_lshl_or_b32 tmp, dst, 1, dst
    body.extend(vop3(0x11B, dst, vgpr(tmp), vgpr(mask55), 128)); // v_and_b32 dst, tmp, 0x55
    // Spread b's low 4 bits to even positions -> tmp, then dst |= tmp << 1.
    body.extend(vop3(0x11B, tmp, vgpr(b), 143, 128)); // v_and_b32 tmp, b, 15
    body.extend(vop3(0x36F, tmp, vgpr(tmp), 130, vgpr(tmp))); // v_lshl_or_b32 tmp, tmp, 2, tmp
    body.extend(vop3(0x11B, tmp, vgpr(tmp), 179, 128)); // v_and_b32 tmp, tmp, 0x33 (51)
    body.extend(vop3(0x36F, tmp, vgpr(tmp), 129, vgpr(tmp))); // v_lshl_or_b32 tmp, tmp, 1, tmp
    body.extend(vop3(0x11B, tmp, vgpr(tmp), vgpr(mask55), 128)); // v_and_b32 tmp, tmp, 0x55
    body.extend(vop3(0x36F, dst, vgpr(tmp), 129, vgpr(dst))); // v_lshl_or_b32 dst, tmp, 1, dst
}

/// BCPack's record endpoint/index reference decoding (optional 4-bit symbol
/// interleaving, signed zigzag delta, and 8-bit wrapping sum), written as
/// console code, matches `victory_formats::bcpack::references` on the GPU.
#[test]
fn bcpack_references_match_the_formats_function() {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = s[0] | (s[1] << 8) | (s[2] << 16) | (s[3] << 24), v2 = paired (0 or 1).
    body.extend(vop3(0x148, 3, vgpr(1), 128, 136)); // v_bfe_u32 v3, v1, 0, 8   (s0)
    body.extend(vop3(0x148, 4, vgpr(1), 136, 136)); // v_bfe_u32 v4, v1, 8, 8   (s1)
    body.extend(vop3(0x148, 5, vgpr(1), 144, 136)); // v_bfe_u32 v5, v1, 16, 8  (s2)
    body.extend(vop3(0x148, 6, vgpr(1), 152, 136)); // v_bfe_u32 v6, v1, 24, 8  (s3)
    // Paired mode: d = interleave(s0, s1) -> v7, ix = interleave(s2, s3) -> v8.
    body.extend([vop1(0x001, 9, 255), 0x0000_0055]); // v_mov_b32 v9, 0x55
    emit_interleave_pair(&mut body, 7, 3, 4, 9, 10);
    emit_interleave_pair(&mut body, 8, 5, 6, 9, 10);
    // Select (d, ix) into (v3, v4) based on paired != 0.
    body.extend(vop3(0x0C5, 106, vgpr(2), 128, 128)); // v_cmp_ne_u32 vcc, v2, 0
    body.extend(vop3(0x101, 3, vgpr(3), vgpr(7), vcc)); // v_cndmask_b32 v3, v3, v7, vcc
    body.extend(vop3(0x101, 4, vgpr(4), vgpr(8), vcc)); // v_cndmask_b32 v4, v4, v8, vcc
    // delta = (d >> 1) ^ (d & 1).wrapping_neg(); ep = ix.wrapping_add(delta) as u8.
    body.extend(vop3(0x116, 5, 129, vgpr(3), 128)); // v_lshrrev_b32 v5, 1, v3
    body.extend(vop3(0x11B, 6, vgpr(3), 129, 128)); // v_and_b32 v6, v3, 1
    body.extend(vop3(0x126, 6, 128, vgpr(6), 128)); // v_sub_nc_u32 v6, 0, v6
    body.extend(vop3(0x11D, 5, vgpr(5), vgpr(6), 128)); // v_xor_b32 v5, v5, v6
    body.extend(vop3(0x125, 5, vgpr(4), vgpr(5), 128)); // v_add_nc_u32 v5, v4, v5
    body.extend(vop3(0x148, 5, vgpr(5), 128, 136)); // v_bfe_u32 v5, v5, 0, 8
    // Pack (ep << 8) | ix into v0.
    body.extend(vop3(0x36F, 0, vgpr(5), 136, vgpr(4))); // v_lshl_or_b32 v0, v5, 8, v4

    let shader = build_valu_cs(&body);
    let mut cases = Vec::new();
    // Unpaired: cover zero, odd/even zigzag deltas, wrapping underflow/overflow, and extremes.
    for d in [0u8, 1, 2, 3, 20, 127, 128, 254, 255] {
        for ix in [0u8, 1, 10, 127, 128, 250, 255] {
            let packed = u32::from_le_bytes([d, ix, 0xA5, 0x5A]);
            cases.push([packed, 0, 0, 0]);
        }
    }
    // Paired: 4-bit symbol pairs (plus upper-nibble noise to confirm masking to 4 bits).
    for &(s0, s1) in &[
        (0u8, 0u8),
        (1, 0),
        (0, 1),
        (0x0A, 0x05),
        (0x0F, 0x0F),
        (0xF3, 0xE7),
    ] {
        for &(s2, s3) in &[(0u8, 0u8), (1, 0), (0x05, 0x0A), (0x0F, 0x0F), (0xC2, 0xD9)] {
            let packed = u32::from_le_bytes([s0, s1, s2, s3]);
            cases.push([packed, 1, 0, 0]);
        }
    }
    // Confirm wrapping underflow (ix == 0 -> ep == 255) and paired/unpaired divergence are tested.
    assert!(cases.iter().any(|&[p, paired, ..]| {
        let s = p.to_le_bytes();
        paired == 0 && bcpack_references_reference(s, false) == (255, 0)
    }));
    assert!(cases.iter().any(|&[p, ..]| {
        let s = p.to_le_bytes();
        bcpack_references_reference(s, true) != bcpack_references_reference(s, false)
    }));
    run_cases(&shader, &cases, |backend, _, [packed, paired, ..], got| {
        let s = packed.to_le_bytes();
        let (want_ep, want_ix) = bcpack_references_reference(s, paired != 0);
        let want = ((want_ep as u32) << 8) | (want_ix as u32);
        assert_eq!(
            got,
            want,
            "{backend}: s={s:?} paired={paired}: got ({}, {}), want ({want_ep}, {want_ix})",
            got >> 8,
            got & 0xFF
        );
    });
}

/// `s_lshlN_add_u32` sets SCC to the carry of its add, which is not lowered:
/// a shader reading it is refused rather than given "result != 0".
#[test]
fn lshl_add_carry_read_is_refused() {
    for op in 0x2e..=0x31u8 {
        let refused = try_build_valu_cs(&[
            sop2(op, 12, 0, 1),
            sop2(0x0a, 13, 0x81, 0x80), // s_cselect_b32 s13, 1, 0
            vop1(0x001, 0, sgpr(13)),
        ]);
        let err = format!("{:?}", refused.expect_err("refused"));
        assert!(err.contains("carry"), "op {op:#x}: {err}");
    }
}

/// A 16-bit result written to WORD_0 with UNUSED_SEXT sign-extends bit 15
/// into the high word; UNUSED_PAD zeroes it.
#[test]
fn sdwa_16bit_word0_sext_sign_extends() {
    // llvm-mc -mcpu=gfx1030: v_cvt_i16_f16_sdwa v0, v1 dst_sel:WORD_0
    // dst_unused:UNUSED_SEXT (then UNUSED_PAD) src0_sel:WORD_0.
    for (extension, want) in [(0x0004_0C01, 0xFFFF_FFFF), (0x0004_0401, 0x0000_FFFF)] {
        let shader = build_valu_cs(&[0x7E00_A6F9, extension]);
        run_cases(&shader, &[[0xBC00, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(got, want, "{backend}: {extension:#x}: {got:#010x}");
        });
    }
}

/// `v_perm_b32`: byte `i` of the result is byte `S` of `{src0, src1}` (`src1`
/// low) for `S < 8`, sign-fill of bytes 1 and 3 of `src1` then `src0` for
/// `8..=11`, `0x00` for `12`, and `0xFF` for `13..=255` (RDNA2 ISA VOP3).
/// Checks both runtime VGPR selectors and the literal constant-selector path.
#[test]
fn v_perm_b32_matches_rdna2() {
    fn perm_byte(src0: u32, src1: u32, sel: u32) -> u32 {
        match sel & 0xFF {
            0..=3 => (src1 >> (8 * (sel & 3))) & 0xFF,
            4..=7 => (src0 >> (8 * (sel & 3))) & 0xFF,
            8 => {
                if (src1 >> 15) & 1 != 0 {
                    0xFF
                } else {
                    0
                }
            }
            9 => {
                if (src1 >> 31) & 1 != 0 {
                    0xFF
                } else {
                    0
                }
            }
            10 => {
                if (src0 >> 15) & 1 != 0 {
                    0xFF
                } else {
                    0
                }
            }
            11 => {
                if (src0 >> 31) & 1 != 0 {
                    0xFF
                } else {
                    0
                }
            }
            12 => 0x00,
            _ => 0xFF,
        }
    }
    fn perm(src0: u32, src1: u32, sel: u32) -> u32 {
        (0..4).fold(0u32, |acc, k| {
            acc | (perm_byte(src0, src1, sel >> (8 * k)) << (8 * k))
        })
    }

    let words = [
        0x0000_0000u32,
        0x7F7F_7F7F,
        0x8080_8080,
        0x817E_00FF,
        0x12A4_7F80,
        0xFE01_807F,
    ];
    let selectors = [
        0x0302_0100u32, // bytes 0..=3 of src1
        0x0706_0504,    // bytes 0..=3 of src0
        0x0B0A_0908,    // sign-fill of bytes 1,3 of src1 then src0
        0xFF0E_0D0C,    // 12 -> 0x00, 13/14/255 -> 0xFF
        0x0C09_0401,    // mixed bytes, sign-fill, zero
        0x100B_0806,    // mixed bytes, sign-fill, 0xFF
    ];

    let mut dyn_cases = Vec::new();
    for &a in &words {
        for &b in &words {
            for &sel in &selectors {
                dyn_cases.push([a, b, sel, 0]);
            }
        }
    }
    let dyn_shader = build_valu_cs(&vop3(0x344, 0, vgpr(1), vgpr(2), vgpr(3)));
    run_cases(&dyn_shader, &dyn_cases, |backend, i, x, got| {
        let want = perm(x[0], x[1], x[2]);
        assert_eq!(
            got, want,
            "{backend}: dynamic v_perm_b32 case {i} ({:#010x}, {:#010x}, {:#010x}): {got:#010x}, want {want:#010x}",
            x[0], x[1], x[2],
        );
    });

    let const_sel = 0x0D0C_0904u32;
    let [lo, hi] = vop3(0x344, 0, vgpr(1), vgpr(2), 255);
    let const_shader = build_valu_cs(&[lo, hi, const_sel]);
    let mut const_cases = Vec::new();
    for &a in &words {
        for &b in &words {
            const_cases.push([a, b, 0, 0]);
        }
    }
    run_cases(&const_shader, &const_cases, |backend, i, x, got| {
        let want = perm(x[0], x[1], const_sel);
        assert_eq!(
            got, want,
            "{backend}: literal v_perm_b32 case {i} ({:#010x}, {:#010x}): {got:#010x}, want {want:#010x}",
            x[0], x[1],
        );
    });
}

/// `s_absdiff_i32` (wrapping signed difference, absolute value, `SCC = result != 0`)
/// and `s_pack_{ll,lh,hh}_b32_b16` (16-bit half packing, leaving `SCC` untouched).
#[test]
fn s_absdiff_and_pack_match_rdna2() {
    let absdiff_cases: &[[u32; 4]] = &[
        [0, 0, 0, 0],
        [5, 5, 0, 0],
        [5, 12, 0, 0],
        [12, 5, 0, 0],
        [(-5i32) as u32, 3, 0, 0],
        [3, (-5i32) as u32, 0, 0],
        [i32::MIN as u32, 0, 0, 0],
        [i32::MAX as u32, i32::MIN as u32, 0, 0],
    ];
    let absdiff_val = build_valu_cs(&[
        sop2(0x2c, 16, 0, 1), // s_absdiff_i32 s16, s0, s1
        vop1(1, 0, sgpr(16)), // v_mov_b32 v0, s16
    ]);
    let absdiff_scc = build_valu_cs(&[
        sop2(0x2c, 16, 0, 1),     // s_absdiff_i32 s16, s0, s1
        sop2(0x0a, 17, 129, 128), // s_cselect_b32 s17, 1, 0
        vop1(1, 0, sgpr(17)),     // v_mov_b32 v0, s17
    ]);
    let model_absdiff = |a: u32, b: u32| -> u32 {
        let diff = (a as i32).wrapping_sub(b as i32);
        diff.wrapping_abs() as u32
    };
    run_cases(&absdiff_val, absdiff_cases, |backend, i, x, got| {
        let want = model_absdiff(x[0], x[1]);
        assert_eq!(got, want, "{backend}: s_absdiff_i32 value case {i}");
    });
    run_cases(&absdiff_scc, absdiff_cases, |backend, i, x, got| {
        let want = u32::from(model_absdiff(x[0], x[1]) != 0);
        assert_eq!(got, want, "{backend}: s_absdiff_i32 scc case {i}");
    });

    let pack_cases: &[[u32; 4]] = &[
        [0x0000_0000, 0x0000_0000, 1, 0],
        [0x1234_5678, 0x9ABC_DEF0, 0, 0],
        [0x1234_5678, 0x9ABC_DEF0, 1, 0],
        [0x0000_FFFF, 0xFFFF_0000, 1, 0],
        [0xFFFF_0000, 0x0000_FFFF, 0, 0],
        [0xDEAD_BEEF, 0xCAFE_BABE, 1, 0],
    ];
    type PackCase = (u8, &'static str, fn(u32, u32) -> u32);
    let packs: &[PackCase] = &[
        (0x32, "s_pack_ll_b32_b16", |a, b| (a & 0xFFFF) | (b << 16)),
        (0x33, "s_pack_lh_b32_b16", |a, b| {
            (a & 0xFFFF) | (b & 0xFFFF_0000)
        }),
        (0x34, "s_pack_hh_b32_b16", |a, b| {
            (a >> 16) | (b & 0xFFFF_0000)
        }),
    ];
    for &(op, name, model) in packs {
        let val_shader = build_valu_cs(&[
            sop2(op, 16, 0, 1),   // s_pack_* s16, s0, s1
            vop1(1, 0, sgpr(16)), // v_mov_b32 v0, s16
        ]);
        run_cases(&val_shader, pack_cases, |backend, i, x, got| {
            let want = model(x[0], x[1]);
            assert_eq!(
                got, want,
                "{backend}: {name} case {i}: {got:#010x}, want {want:#010x}"
            );
        });
        // Verify s_pack_* leaves SCC untouched (even when its result is 0).
        let scc_shader = build_valu_cs(&[
            0xBF06_8102,              // s_cmp_eq_u32 s2, 1
            sop2(op, 16, 0, 1),       // s_pack_* s16, s0, s1 (must preserve SCC)
            sop2(0x0a, 17, 129, 128), // s_cselect_b32 s17, 1, 0
            vop1(1, 0, sgpr(17)),     // v_mov_b32 v0, s17
        ]);
        run_cases(&scc_shader, pack_cases, |backend, i, x, got| {
            assert_eq!(got, x[2], "{backend}: {name} scc preservation case {i}");
        });
    }
}
