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
use crate::{io, ir, resource, structure};

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
/// VOP1 SDWA `op_sdwa vdst, vsrc0 dst_sel dst_unused src0_sel`.
const fn vop1_sdwa(
    op: u16,
    vdst: u8,
    vsrc0: u8,
    dst_sel: u8,
    dst_unused: u8,
    src0_sel: u8,
) -> [u32; 2] {
    [
        vop1(op, vdst, 0xF9),
        vsrc0 as u32
            | (dst_sel as u32 & 7) << 8
            | (dst_unused as u32 & 3) << 11
            | (src0_sel as u32 & 7) << 16,
    ]
}

/// VOP2 SDWA `op_sdwa vdst, vsrc0, vsrc1 dst_sel dst_unused src0_sel src1_sel`.
const fn vop2_sdwa(
    op: u8,
    vdst: u8,
    vsrc0: u8,
    vsrc1: u8,
    dst: (u8, u8),
    src_sel: [u8; 2],
) -> [u32; 2] {
    [
        (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | 0xF9,
        vsrc0 as u32
            | (dst.0 as u32 & 7) << 8
            | (dst.1 as u32 & 3) << 11
            | (src_sel[0] as u32 & 7) << 16
            | (src_sel[1] as u32 & 7) << 24,
    ]
}


/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
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
        host_wave64: true,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// One GPU test at a time. With these tests on the hardware adapter and WARP
/// from several test threads at once, now and then none of their dispatches
/// ever completes and the test binary hangs; and two GPU tests at once on
/// WARP can end the test process with an access violation. Other GPU test
/// modules take this lock too.
pub(super) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
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
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
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

/// `v_cmp_f_*` is false and `v_cmp_tru_*` true whatever the operands, NaNs
/// included (RDNA2 ISA). As `fcmp false`/`fcmp true` WARP removed its device
/// at the first dispatch.
#[test]
fn float_compares_that_are_always_false_or_true() {
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;
    let f32_inputs = [
        [f(1.0), f(2.0), 0, 0],
        [f(1.0), f(1.0), 0, 0],
        [nan, f(1.0), 0, 0],
        [nan, nan, 0, 0],
    ];
    let f16_inputs = [
        [0x3C00, 0x4000, 0, 0],
        [0x3C00, 0x3C00, 0, 0],
        [0x7E00, 0x3C00, 0, 0],
        [0x7E00, 0x7E00, 0, 0],
    ];
    // v_cmp_*_e64 s20, v1, v2; v_cndmask_b32_e64 v0, 0, 1, s20
    for (op, what, inputs, want) in [
        (0x000u16, "v_cmp_f_f32", f32_inputs, 0),
        (0x00f, "v_cmp_tru_f32", f32_inputs, 1),
        (0x0c8, "v_cmp_f_f16", f16_inputs, 0),
        (0x0ef, "v_cmp_tru_f16", f16_inputs, 1),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        run_cases(&shader, &inputs, |backend, _, inp, got| {
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
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

/// VOP3B `op vdst, sdst, src0, src1, src2` (no modifiers).
const fn vop3b(op: u16, vdst: u8, sdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    let [w0, w1] = vop3(op, vdst, src0, src1, src2);
    [w0 | (sdst as u32 & 0x7F) << 8, w1]
}

/// 32 × 32-bit inputs where the product or the 64-bit addend reach their
/// extremes, so both the multiply and the add wrap.
const MAD64_CASES: &[[u32; 4]] = &[
    [0, 0, 0, 0],
    [1, 1, 0, 0],
    [0xffff_ffff, 0xffff_ffff, 0, 0],
    [0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff],
    [0xffff_ffff, 0xffff_ffff, 0x0000_0002, 0],
    [0x8000_0000, 0x8000_0000, 0, 0x8000_0000],
    [0x8000_0000, 0x7fff_ffff, 0x1234_5678, 0x9abc_def0],
    [0x7fff_ffff, 0x7fff_ffff, 0xffff_ffff, 0x7fff_ffff],
    [0xffff_fffe, 3, 1, 0],
    [0x0001_0000, 0x0001_0000, 0xffff_ffff, 0xffff_ffff],
    [0xdead_beef, 0x1337_c0de, 0x0bad_f00d, 0xfeed_face],
];

/// The 64-bit result and the carry out of the addition (unsigned form).
fn mad64(inp: [u32; 4], signed: bool) -> (u64, bool) {
    let addend = u64::from(inp[2]) | u64::from(inp[3]) << 32;
    let product = if signed {
        (i64::from(inp[0] as i32) * i64::from(inp[1] as i32)) as u64
    } else {
        u64::from(inp[0]) * u64::from(inp[1])
    };
    product.overflowing_add(addend)
}

#[test]
fn v_mad_u64_u32_and_v_mad_i64_i32_match_rdna2() {
    // v_mad_{u64_u32,i64_i32} v[5:6], vcc_lo, v1, v2, v[3:4] (VOP3B 0x176,
    // 0x177), then v_mov_b32 v0, v5 (low dword) or v6 (high dword).
    for (op, signed, form) in [
        (0x176, false, "v_mad_u64_u32"),
        (0x177, true, "v_mad_i64_i32"),
    ] {
        let mad = vop3b(op, 5, 106, vgpr(1), vgpr(2), vgpr(3));
        for (half, reg) in [("low", 5), ("high", 6)] {
            let shader = build_valu_cs(&[mad[0], mad[1], vop1(0x001, 0, vgpr(reg))]);
            run_cases(&shader, MAD64_CASES, |backend, _, inp, got| {
                let r = mad64(inp, signed).0;
                let want = if reg == 5 { r as u32 } else { (r >> 32) as u32 };
                assert_eq!(
                    got, want,
                    "{backend}: {form}({:#x}, {:#x}, {:#x}{:08x}) {half} = {got:#010x}, \
                     want {want:#010x}",
                    inp[0], inp[1], inp[3], inp[2],
                );
            });
        }
    }
    // The unsigned form's carry out: v_cndmask_b32 v0, 0, 1, vcc_lo.
    let mad = vop3b(0x176, 5, 106, vgpr(1), vgpr(2), vgpr(3));
    let pick = vop3(0x101, 0, 128, 129, 106);
    let shader = build_valu_cs(&[mad[0], mad[1], pick[0], pick[1]]);
    run_cases(&shader, MAD64_CASES, |backend, _, inp, got| {
        let want = u32::from(mad64(inp, false).1);
        assert_eq!(
            got, want,
            "{backend}: v_mad_u64_u32({:#x}, {:#x}, {:#x}{:08x}) carry = {got}, want {want}",
            inp[0], inp[1], inp[3], inp[2],
        );
    });
}

/// f32 bits of values whose significand and exponent RDNA2 returns exactly:
/// zeros, infinities, NaNs, normals at both ends of the range and
/// denormals (normalized: v_frexp works on the bits).
const FREXP_CASES: &[u32] = &[
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7fc0_0000, // NaN
    0xffc0_0001, // NaN with a payload
    0x3f80_0000, // 1.0
    0xbf80_0000, // -1.0
    0x3eff_ffff, // just below 0.5
    0x4049_0fdb, // pi
    0x0080_0000, // smallest normal
    0x7f7f_ffff, // largest finite
    0xff7f_ffff,
    0x0000_0001, // smallest denormal
    0x8000_0001,
    0x007f_ffff, // largest denormal
    0x0040_0000,
    0x0000_ffff,
];

/// (significand bits, exponent) as RDNA2's v_frexp_mant_f32 and
/// v_frexp_exp_i32_f32 give them.
fn frexp_ref(x: u32) -> (u32, i32) {
    let field = (x >> 23 & 0xFF) as i32;
    let frac = x & 0x7F_FFFF;
    if x & 0x7FFF_FFFF == 0 || field == 0xFF {
        return (x, 0);
    }
    let (frac, e) = if field == 0 {
        let lz = frac.leading_zeros();
        ((frac << (lz - 8)) & 0x7F_FFFF, -117 - lz as i32)
    } else {
        (frac, field - 126)
    };
    (x & 0x8000_0000 | 0x3F00_0000 | frac, e)
}

#[test]
fn v_frexp_mant_and_exp_f32_match_rdna2() {
    let cases: Vec<[u32; 4]> = FREXP_CASES.iter().map(|&x| [x, 0, 0, 0]).collect();
    // Every case: the significand times 2^exponent gives the input back.
    for &x in FREXP_CASES {
        let (m, e) = frexp_ref(x);
        let back = f64::from(f32::from_bits(m)) * 2f64.powi(e);
        let want = f64::from(f32::from_bits(x));
        assert!(back == want || want.is_nan(), "{x:#010x}");
    }
    // v_frexp_mant_f32 v0, v1 (VOP1 0x40)
    let mant = build_valu_cs(&[vop1(0x040, 0, vgpr(1))]);
    run_cases(&mant, &cases, |backend, _, inp, got| {
        let want = frexp_ref(inp[0]).0;
        assert_eq!(
            got, want,
            "{backend}: v_frexp_mant_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    // v_frexp_exp_i32_f32 v0, v1 (VOP1 0x3f)
    let exp = build_valu_cs(&[vop1(0x03f, 0, vgpr(1))]);
    run_cases(&exp, &cases, |backend, _, inp, got| {
        let want = frexp_ref(inp[0]).1;
        assert_eq!(
            got as i32, want,
            "{backend}: v_frexp_exp_i32_f32({:#010x}) = {}, want {want}",
            inp[0], got as i32,
        );
    });
}

#[test]
fn v_ldexp_f32_scales_by_any_exponent() {
    // v_ldexp_f32 v0, v1, v2 (VOP3 0x362). Results stay normal (or zero,
    // or infinity) so the host's denormal mode cannot change them.
    let shader = build_valu_cs(&vop3(0x362, 0, vgpr(1), vgpr(2), 0));
    let inf = f32::INFINITY.to_bits();
    let cases: &[(u32, i32, u32)] = &[
        (1.0f32.to_bits(), 0, 1.0f32.to_bits()),
        (1.0f32.to_bits(), 127, 0x7f00_0000),
        (1.0f32.to_bits(), 128, inf),
        (1.0f32.to_bits(), -126, 0x0080_0000),
        (1.5f32.to_bits(), -1, 0.75f32.to_bits()),
        ((-3.0f32).to_bits(), 4, (-48.0f32).to_bits()),
        // Exponents past what one power-of-two factor holds.
        (0x0080_0000, 253, 0x7f00_0000),
        (0x7f00_0000, -253, 0x0080_0000),
        (0x0080_0000, 254, inf),
        (0x7f7f_ffff, -253, 0x00ff_ffff),
        // Overflow and underflow from any finite nonzero input.
        (0x0080_0000, i32::MAX, inf),
        ((-1.0e-30f32).to_bits(), 1000, 0xff80_0000),
        (0x7f7f_ffff, i32::MIN, 0),
        ((-0.5f32).to_bits(), -400, 0x8000_0000),
        // Zeros, infinities and NaN keep their value.
        (0x8000_0000, 1000, 0x8000_0000),
        (inf, -1000, inf),
        (0xff80_0000, 0, 0xff80_0000),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, e, _)| [x, e as u32, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].2;
        assert_eq!(
            got, want,
            "{backend}: v_ldexp_f32({:#010x}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1] as i32,
        );
    });
    run_cases(&shader, &[[0x7fc0_0000, 10, 0, 0]], |backend, _, _, got| {
        assert!(f32::from_bits(got).is_nan(), "{backend}: {got:#010x}");
    });
}

/// Translates a 64-thread wave64 program from `wave64_fixtures` for a host
/// without 64-lane waves (lane exchanges through group-shared memory) and
/// checks what its group stores on every available SM 6.6 host. The
/// translation declares no wave size, so it runs on any GPU: on NVIDIA's
/// 32-lane waves this is the only way these programs run at all.
fn run_wave64_group(body: &[u32], lds_bytes: u32, expected: &[u32]) {
    use super::wave64_fixtures::{lower_for, program};
    let lowered = lower_for(&program(body, 64, lds_bytes), false).expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Constants, 0) | (RootType::Table, 1) => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(256)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        for (n, (&got, &want)) in out[0].iter().zip(expected).enumerate() {
            assert_eq!(
                got,
                want,
                "{label}: dword {n} (lane {}, result {}) = {got:#x}, want {want:#x}",
                n % 64,
                n / 64,
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn wave64_lane_reads_ballots_and_lds_run_without_64_lane_waves() {
    use super::wave64_fixtures::{CROSS_LANE, cross_lane_expected};
    run_wave64_group(CROSS_LANE, 512, &cross_lane_expected());
}

#[test]
fn wave64_dpp_first_lane_and_swizzle_run_without_64_lane_waves() {
    use super::wave64_fixtures::{DPP_FIRST_SWIZZLE, dpp_first_swizzle_expected};
    run_wave64_group(DPP_FIRST_SWIZZLE, 0, &dpp_first_swizzle_expected());
}

#[test]
fn v_cmp_class_f32_matches_rdna2() {
    let build = |abs0: bool, neg0: bool| {
        let [mut w0, mut w1] = vop3(0x088, 6, vgpr(1), vgpr(2), 0);
        if abs0 {
            w0 |= 1 << 8; // abs[0] = 1
        }
        if neg0 {
            w1 |= 1 << 29; // neg[0] = 1
        }
        let sel = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
        build_valu_cs(&[w0, w1, sel[0], sel[1]])
    };

    // All 10 IEEE-754 binary32 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7F80_0001, 0), // +sNaN (quiet bit 0x40_0000 clear)
        (0xFF80_0001, 0), // -sNaN
        (0x7FC0_0000, 1), // +qNaN (quiet bit set)
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
    let mut plain_cases = Vec::new();
    for &(bits, cls) in classes {
        plain_cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        plain_cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let cmp_plain = build(false, false);
    let plain_in: Vec<[u32; 4]> = plain_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_plain, &plain_in, |backend, i, inp, got| {
        let want = plain_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f32({:#010x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // RDNA2 input modifiers on v_cmp_class_f32 manipulate the sign bit bitwise:
    // they must not quiet sNaNs or flush denormals to zero.
    struct ModCases {
        label: &'static str,
        abs: bool,
        neg: bool,
        items: &'static [(u32, u32)],
    }
    let mod_sets = [
        ModCases {
            label: "neg",
            abs: false,
            neg: true,
            items: &[
                (0x3F80_0000, 3), // -(+1.0) -> -normal
                (0x0000_0000, 5), // -(+0.0) -> -0.0
                (0x8000_0000, 6), // -(-0.0) -> +0.0
                (0x0000_0001, 4), // -(+denormal) -> -denormal (not flushed to -0.0)
                (0x8000_0001, 7), // -(-denormal) -> +denormal (not flushed to +0.0)
                (0x7F80_0001, 0), // -(+sNaN) -> sNaN (not quieted to qNaN)
                (0xFF80_0001, 0), // -(-sNaN) -> sNaN (not quieted to qNaN)
            ],
        },
        ModCases {
            label: "abs",
            abs: true,
            neg: false,
            items: &[
                (0xBF80_0000, 8), // |-1.0| -> +normal
                (0x8000_0000, 6), // |-0.0| -> +0.0
                (0xFF80_0000, 9), // |-inf| -> +inf
                (0x8000_0001, 7), // |-denormal| -> +denormal
                (0xFF80_0001, 0), // |-sNaN| -> sNaN
            ],
        },
        ModCases {
            label: "abs+neg",
            abs: true,
            neg: true,
            items: &[
                (0x3F80_0000, 3), // -|+1.0| -> -normal
                (0xBF80_0000, 3), // -|-1.0| -> -normal
                (0x0000_0001, 4), // -|+denormal| -> -denormal
                (0x8000_0001, 4), // -|-denormal| -> -denormal
                (0x7F80_0001, 0), // -|+sNaN| -> sNaN
            ],
        },
    ];
    for set in mod_sets {
        let shader = build(set.abs, set.neg);
        let mut cases = Vec::new();
        for &(bits, cls) in set.items {
            cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
            cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
        }
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: v_cmp_class_f32[{}]({:#010x}, {:#05x}) = {got}, want {want}",
                set.label, inp[0], inp[1],
            );
        });
    }
}

#[test]
fn sdwa_16bit_dst_unused_sext_sign_extends_word0() {
    // SDWA dst_sel: 4 = WORD_0, 5 = WORD_1.
    // SDWA dst_unused: 0 = UNUSED_PAD, 1 = UNUSED_SEXT, 2 = UNUSED_PRESERVE.
    // Pre-seed v0 from v4 (0xABCD_1234) so UNUSED_SEXT/UNUSED_PAD/UNUSED_PRESERVE
    // are distinguishable from each other.
    let seed = 0xABCD_1234u32;

    // v_cvt_i16_f16_sdwa v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_SEXT src0_sel:WORD_0
    let mut body = vec![vop1(0x001, 0, vgpr(4))];
    body.extend_from_slice(&vop1_sdwa(0x053, 0, 1, 4, 1, 4));
    let cvt_sext = build_valu_cs(&body);
    let cvt_cases: &[(u32, u32)] = &[
        (0xBC00, 0xFFFF_FFFF), // -1.0 -> -1i16 sign-extended to -1i32
        (0xF800, 0xFFFF_8000), // -32768.0 -> -32768i16 sign-extended
        (0x3C00, 0x0000_0001), // +1.0 -> +1i16 zero-extended (old v0 upper bits cleared)
        (0x7BFF, 0x0000_7FFF), // +65504.0 -> saturated +32767i16
        (0x0000, 0x0000_0000), // 0.0 -> 0
    ];
    let cvt_in: Vec<[u32; 4]> = cvt_cases.iter().map(|&(x, _)| [x, 0, 0, seed]).collect();
    run_cases(&cvt_sext, &cvt_in, |backend, i, inp, got| {
        let want = cvt_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cvt_i16_f16_sdwa({:#06x}, WORD_0, UNUSED_SEXT) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });

    // v_cvt_norm_i16_f16_sdwa v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_SEXT src0_sel:WORD_1
    let mut body = vec![vop1(0x001, 0, vgpr(4))];
    body.extend_from_slice(&vop1_sdwa(0x063, 0, 1, 4, 1, 5));
    let norm_sext = build_valu_cs(&body);
    let norm_cases: &[(u32, u32)] = &[
        (0xBC00_3C00, 0xFFFF_8001), // high half -1.0 -> -32767i16 sign-extended
        (0x3C00_BC00, 0x0000_7FFF), // high half +1.0 -> +32767i16
    ];
    let norm_in: Vec<[u32; 4]> = norm_cases.iter().map(|&(x, _)| [x, 0, 0, seed]).collect();
    run_cases(&norm_sext, &norm_in, |backend, i, inp, got| {
        let want = norm_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cvt_norm_i16_f16_sdwa({:#010x}, WORD_0, UNUSED_SEXT) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });

    // v_add_f16_sdwa v0, v1, v2 dst_sel:WORD_0 dst_unused:UNUSED_SEXT src0_sel:WORD_0 src1_sel:WORD_0
    let mut body = vec![vop1(0x001, 0, vgpr(4))];
    body.extend_from_slice(&vop2_sdwa(0x32, 0, 1, 2, (4, 1), [4, 4]));
    let add_sext = build_valu_cs(&body);
    let add_cases: &[(u32, u32, u32)] = &[
        (0xBC00, 0xC000, 0xFFFF_C200), // -1.0 + -2.0 = -3.0 (0xC200, bit 15 set -> sign-extended)
        (0x3C00, 0x4000, 0x0000_4200), // +1.0 + +2.0 = +3.0 (0x4200, bit 15 clear)
    ];
    let add_in: Vec<[u32; 4]> = add_cases.iter().map(|&(a, b, _)| [a, b, 0, seed]).collect();
    run_cases(&add_sext, &add_in, |backend, i, inp, got| {
        let want = add_cases[i].2;
        assert_eq!(
            got, want,
            "{backend}: v_add_f16_sdwa({:#06x}, {:#06x}, WORD_0, UNUSED_SEXT) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    // Regression check for the other dst_sel / dst_unused modes on v_cvt_i16_f16_sdwa(-1.0):
    // WORD_0 + UNUSED_PAD -> 0x0000_FFFF, WORD_0 + UNUSED_PRESERVE -> 0xABCD_FFFF,
    // WORD_1 + UNUSED_PAD / UNUSED_SEXT -> 0xFFFF_0000 (bits below WORD_1 pad with 0),
    // WORD_1 + UNUSED_PRESERVE -> 0xFFFF_1234, DWORD -> 0x0000_FFFF.
    let modes: &[(u8, u8, u32)] = &[
        (4, 0, 0x0000_FFFF),
        (4, 2, 0xABCD_FFFF),
        (5, 0, 0xFFFF_0000),
        (5, 1, 0xFFFF_0000),
        (5, 2, 0xFFFF_1234),
        (6, 0, 0x0000_FFFF),
    ];
    for &(dst_sel, dst_unused, want) in modes {
        let mut body = vec![vop1(0x001, 0, vgpr(4))];
        body.extend_from_slice(&vop1_sdwa(0x053, 0, 1, dst_sel, dst_unused, 4));
        let shader = build_valu_cs(&body);
        run_cases(&shader, &[[0xBC00, 0, 0, seed]], |backend, _, _, got| {
            assert_eq!(
                got, want,
                "{backend}: v_cvt_i16_f16_sdwa(-1.0, dst_sel={dst_sel}, dst_unused={dst_unused}) = {got:#010x}, want {want:#010x}",
            );
        });
    }
}

/// [`build_valu_cs`], with the lowering's refusal as an error.
fn try_build_valu_cs(body: &[u32]) -> Result<HostShader, String> {
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
        host_wave64: true,
    })
    .map_err(|e| format!("{e:?}"))?;

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    Ok(shader)
}

/// MTBUF `op vdata, v5, s[8:11], 0 format:fmt offset:16` (no offen/idxen;
/// `op` 4..7 stores x..xyzw, 12..15 their d16 forms).
const fn mtbuf_store(op: u8, fmt: u8, vdata: u8) -> [u32; 2] {
    [
        0x3A << 26 | (fmt as u32) << 19 | (op as u32 & 7) << 16 | 16,
        0x80 << 24 | (op as u32 >> 3 & 1) << 21 | (8 >> 2) << 16 | (vdata as u32) << 8 | 5,
    ]
}

/// Runs `shader` once with root constants `inputs` on every available
/// SM 6.6 host and returns the first `dwords` of its raw UAV, per device.
fn run_dwords(shader: &HostShader, inputs: [u32; 4], dwords: u32) -> Vec<(&'static str, Vec<u32>)> {
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Constants, 0) => inputs[u.dword as usize],
                (RootType::Table, 1) => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let r = host
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
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, r[0].clone()));
    }
    assert!(
        !out.is_empty(),
        "neither hardware nor WARP SM 6.6 device available"
    );
    out
}

#[test]
fn tbuffer_store_format_writes_the_formats_layout() {
    let f = |x: f32| x.to_bits();
    // (what, op, format, data in v1..v4, dwords 4..8 afterwards)
    type Case = (&'static str, u8, u8, [u32; 4], [u32; 4]);
    let cases: &[Case] = &[
        (
            "32_32_32_32 float xyzw",
            7,
            77,
            [f(1.5), f(-0.0), 0x7FC0_1234, 0x0000_0001],
            [f(1.5), f(-0.0), 0x7FC0_1234, 0x0000_0001],
        ),
        (
            "32_32_32_32 sint xyzw",
            7,
            76,
            [u32::MAX, 0x8000_0000, 7, 0x1234_5678],
            [u32::MAX, 0x8000_0000, 7, 0x1234_5678],
        ),
        (
            "32_32_32 float xyz",
            6,
            74,
            [f(2.0), f(3.0), f(4.0), 0xDEAD_BEEF],
            [f(2.0), f(3.0), f(4.0), 0],
        ),
        (
            "32_32 uint xy",
            5,
            63,
            [0xAABB_CCDD, 9, 0xDEAD_BEEF, 0xDEAD_BEEF],
            [0xAABB_CCDD, 9, 0, 0],
        ),
        (
            "32 float x",
            4,
            22,
            [f(-7.25), 1, 2, 3],
            [f(-7.25), 0, 0, 0],
        ),
        (
            // d16: the halves are stored as they are.
            "16_16 float d16_xy",
            13,
            29,
            [0xBC00_3555, 0xDEAD_BEEF, 0, 0],
            [0xBC00_3555, 0, 0, 0],
        ),
        (
            "16_16_16_16 float d16_xyzw",
            15,
            71,
            [0x1234_5678, 0x7C00_FC00, 0, 0],
            [0x1234_5678, 0x7C00_FC00, 0, 0],
        ),
    ];
    for &(what, op, fmt, inputs, want) in cases {
        let mut body = vec![vop1(0x001, 5, 0x80)]; // v_mov_b32 v5, 0
        body.extend(mtbuf_store(op, fmt, 1));
        let shader = build_valu_cs(&body);
        for (backend, out) in run_dwords(&shader, inputs, 8) {
            assert_eq!(
                out[4..8],
                want,
                "{backend}: tbuffer_store {what}: wrote {:#010x?}, want {want:#010x?}",
                &out[4..8],
            );
        }
    }
}

#[test]
fn tbuffer_store_format_refuses_what_it_does_not_convert() {
    for (what, op, fmt, reason) in [
        ("8_8_8_8 unorm", 7, 56, "a converted or narrow format"),
        ("16_16 uint", 5, 27, "a converted or narrow format"),
        ("16 float", 4, 13, "an element under a dword"),
        (
            "32_32_32_32 float, xy only",
            5,
            77,
            "some of its format's channels",
        ),
        ("32 float d16_x", 12, 22, "a converted or narrow format"),
        // f32 to f16 waits for nearest-even narrowing (#422).
        ("16_16 float xy", 5, 29, "a converted or narrow format"),
        (
            "16_16_16_16 float xyzw",
            7,
            71,
            "a converted or narrow format",
        ),
    ] {
        let mut body = vec![vop1(0x001, 5, 0x80)];
        body.extend(mtbuf_store(op, fmt, 1));
        match try_build_valu_cs(&body) {
            Ok(_) => panic!("tbuffer_store {what} lowered; it should be refused"),
            Err(e) => assert!(e.contains(reason), "tbuffer_store {what}: refused with {e}"),
        }
    }
}

#[test]
fn v_mul_legacy_f32_treats_denormals_as_zero() {
    // v_mul_legacy_f32 v0, v1, v2 (VOP2 opcode 0x007 in its VOP3 encoding)
    let shader = build_valu_cs(&vop3(0x107, 0, vgpr(1), vgpr(2), vgpr(0)));
    let f = f32::to_bits;
    let (denorm, neg_denorm, min_normal) = (0x0000_0001u32, 0x8040_0000u32, 0x0080_0000u32);

    let cases: &[(u32, u32, u32)] = &[
        // Zero or denormal times anything, non-finite included, is +0.
        (f(0.0), f(f32::INFINITY), f(0.0)),
        (f(-0.0), f(f32::NAN), f(0.0)),
        (f(f32::NAN), f(0.0), f(0.0)),
        (denorm, f(f32::INFINITY), f(0.0)),
        (f(f32::NAN), neg_denorm, f(0.0)),
        (denorm, f(2.0f32.powi(100)), f(0.0)),
        // The smallest normal is not zero.
        (min_normal, f(2.0), f(f32::MIN_POSITIVE * 2.0)),
        // Ordinary products.
        (f(1.5), f(-4.0), f(-6.0)),
        (f(3.0), f(0.5), f(1.5)),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(a, b, _)| [a, b, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].2;
        assert_eq!(
            got, want,
            "{backend}: v_mul_legacy_f32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

/// VOP3 `op vdst, src0, src1, src2` with explicit `abs`, `neg`, `omod`, `clamp`.
#[allow(clippy::too_many_arguments)]
const fn vop3_mod(
    op: u16,
    vdst: u8,
    src0: u16,
    src1: u16,
    src2: u16,
    abs: u8,
    neg: u8,
    omod: u8,
    clamp: bool,
) -> [u32; 2] {
    [
        0x35 << 26
            | (op as u32 & 0x3FF) << 16
            | (clamp as u32) << 15
            | (abs as u32 & 7) << 8
            | vdst as u32,
        (neg as u32 & 7) << 29
            | (omod as u32 & 3) << 27
            | (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18,
    ]
}

#[test]
fn f16_omod_applies_before_clamp() {
    // VOP3 v_add_f16 (0x132 = 0x100 + VOP2 0x32): v0 = (v1 + v2) * omod [clamp]
    let mul2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 1, false));
    let mul4 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 2, false));
    let div2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, false));
    let div2_clamp = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, true));

    // f16 constants: 0.25=0x3400, 0.5=0x3800, 0.75=0x3A00, 1.0=0x3C00, 1.5=0x3E00, 2.0=0x4000, -0.5=0xB800
    run_cases(
        &mul2,
        &[[0x3400, 0x3800, 0, 0]], // (0.25 + 0.5) * 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 mul:2");
        },
    );
    run_cases(
        &mul4,
        &[[0x3400, 0x3400, 0, 0]], // (0.25 + 0.25) * 4 = 2.0 (0x4000)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x4000, "{backend}: v_add_f16 mul:4");
        },
    );
    run_cases(
        &div2,
        &[[0x3E00, 0x3E00, 0, 0]], // (1.5 + 1.5) / 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 div:2");
        },
    );
    // Ordering check: (0.75 + 0.75) = 1.5; div:2 before clamp gives 0.75 (0x3A00),
    // whereas clamp before div:2 would give 0.5 (0x3800).
    let clamp_cases: &[(u16, u16, u16)] = &[
        (0x3A00, 0x3A00, 0x3A00), // (0.75 + 0.75) / 2 = 0.75 -> 0.75
        (0x3E00, 0x3E00, 0x3C00), // (1.5 + 1.5) / 2 = 1.5 -> clamped to 1.0
        (0xB800, 0xB800, 0x0000), // (-0.5 + -0.5) / 2 = -0.5 -> clamped to 0.0
    ];
    let inputs: Vec<[u32; 4]> = clamp_cases
        .iter()
        .map(|&(a, b, _)| [u32::from(a), u32::from(b), 0, 0])
        .collect();
    run_cases(&div2_clamp, &inputs, |backend, i, _, got| {
        let want = clamp_cases[i].2;
        assert_eq!(
            got as u16, want,
            "{backend}: v_add_f16 div:2 clamp case {i}: got {:#06x}, want {want:#06x}",
            got as u16,
        );
    });
}

#[test]
fn cvt_16bit_int_from_f16_saturates_and_honors_modifiers() {
    // VOP3 v_cvt_u16_f16 = 0x1D2 (0x180 + VOP1 0x52)
    // VOP3 v_cvt_i16_f16 = 0x1D3 (0x180 + VOP1 0x53)
    let cvt_u16 = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_u16_abs = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 1, 0, 0, false));
    let cvt_i16 = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_i16_neg = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 1, 0, false));

    let u16_cases: &[(u16, u16)] = &[
        (0xBC00, 0),      // -1.0 -> 0
        (0xFC00, 0),      // -inf -> 0
        (0x7E00, 0),      // NaN -> 0
        (0x57BC, 123),    // 123.75 -> 123
        (0x7BFF, 65504),  // 65504.0 (max finite f16) -> 65504
        (0x7C00, 0xFFFF), // +inf -> 65535 (saturated)
    ];
    let u16_in: Vec<[u32; 4]> = u16_cases.iter().map(|&(x, _)| [u32::from(x), 0, 0, 0]).collect();
    run_cases(&cvt_u16, &u16_in, |backend, i, inp, got| {
        let want = u16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_u16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_u16_abs,
        &[[0xD7BC, 0, 0, 0]], // |-123.75| -> 123
        |backend, _, _, got| {
            assert_eq!(got as u16, 123, "{backend}: v_cvt_u16_f16 |src0|");
        },
    );

    let i16_cases: &[(u16, u16)] = &[
        (0xFC00, 0x8000),       // -inf -> -32768
        (0xF8E2, 0x8000),       // -40000.0 -> -32768 (saturated, not wrapped)
        (0xD7BC, (-123i16) as u16), // -123.75 -> -123
        (0x7E00, 0),            // NaN -> 0
        (0x57BC, 123),          // 123.75 -> 123
        (0x78E2, 0x7FFF),       // +40000.0 -> 32767 (saturated, not wrapped)
        (0x7C00, 0x7FFF),       // +inf -> 32767
    ];
    let i16_in: Vec<[u32; 4]> = i16_cases.iter().map(|&(x, _)| [u32::from(x), 0, 0, 0]).collect();
    run_cases(&cvt_i16, &i16_in, |backend, i, inp, got| {
        let want = i16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_i16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_i16_neg,
        &[[0x57BC, 0, 0, 0], [0x78E2, 0, 0, 0]], // -(123.75) -> -123; -(40000.0) -> -32768
        |backend, i, _, got| {
            let want = [(-123i16) as u16, 0x8000][i];
            assert_eq!(got as u16, want, "{backend}: v_cvt_i16_f16 -src0 case {i}");
        },
    );
}

/// MUBUF `buffer_load_dword vdata, vaddr, s[srsrc:srsrc+3], 0 offen`.
const fn buffer_load_dword_offen(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x0C << 18 | 1 << 12,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// A root SRV has no bounds on the host: a read past what the host bound
/// reads 0, as a console buffer view past its records does, instead of
/// memory after it (or an unmapped page, which some GPUs stall on).
#[test]
fn root_view_reads_stop_at_the_bound_size() {
    // s0..s3 root constants, s4 the UAV table, s[8:9] the root SRV's address.
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(12, 4));
    // The buffer descriptor built around the root address: all records.
    words.push(s_mov_b32(10, 0xC1));
    words.push(s_mov_b32(11, 0x80));
    words.push(vop1(0x001, 1, sgpr(0))); // v_mov_b32 v1, s0: the byte offset
    words.extend_from_slice(&buffer_load_dword_offen(0, 1, 8));
    words.extend_from_slice(&buffer_store_dword(0, 12, 0));
    words.push(S_ENDPGM);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let root = |root, sgprs, user_data, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let used = |kind, root, user_data, slot| UsedResource {
        kind,
        root,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            root(RootType::Constants, 4, 0, 0),
            root(RootType::Table, 1, 4, 1),
            root(RootType::Descriptor, 2, 8, 2),
        ],
        used: vec![
            used(ResourceType::Uav, RootType::Table, 4, 1),
            used(ResourceType::Srv, RootType::Descriptor, 8, 2),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 10 << 1, // 10 user SGPRs: s0..s9
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
        host_wave64: true,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    assert_eq!(shader.root_srvs, [2]);
    assert!(
        shader
            .user_data
            .iter()
            .any(|u| u.root == RootType::Descriptor && u.slot == 2 && u.index == 8),
        "the root view's size is user data: {:?}",
        shader.user_data
    );
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    // 64 dwords bound, of which the host says 16 (64 bytes) are the view's.
    let data: Vec<u32> = (0..64).map(|i| 0x100 + i).collect();
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        for (offset, want) in [
            (0, 0x100),
            (60, 0x10F),
            (64, 0),
            (128, 0),
            (u32::MAX - 3, 0),
        ] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match u.root {
                    RootType::Constants => offset,
                    RootType::Descriptor => 64,
                    _ => 0,
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: data.clone(),
                                },
                            ),
                        ],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(out[0][0], want, "{label}: offset {offset:#x}");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
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
