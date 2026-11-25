//! Hardware and WARP execution tests for computed barycentric interpolation,
//! directly read barycentrics (`SV_Barycentrics`), and standalone first-half
//! (`v_interp_p1_f32`) interpolation across perspective, linear, and centroid
//! modes.

use super::bitcode::{Attr, Term, pred};
use super::container::{self, SigElement};
use super::shader::Shader;
use super::{HostShader, emit};
use crate::dxbc::{Container, FourCC, build};
use crate::native::{HwStage, NativeProgram, PsInput, StageIo, Usage};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// Builds a perspective vertex shader outputting a triangle covering the
/// render target with distinct non-unit `w` values (`1.0, 2.0, 4.0`) at the
/// three vertices so perspective and linear interpolation diverge, plus
/// `TEXCOORD0` with distinct per-vertex values.
fn perspective_triangle_vs(texcoord_interp: u8) -> HostShader {
    let mut s = Shader::new(false);
    s.inputs.push(SigElement {
        name: "SV_VertexID".into(),
        index: 0,
        kind: 1,
        comp: 5,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 1,
        usage: 1,
        half: false,
    });
    s.outputs.push(SigElement {
        name: "TEXCOORD".into(),
        index: 0,
        kind: 0,
        comp: 9,
        interp: texcoord_interp,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0b1111,
        half: false,
    });
    s.outputs.push(SigElement {
        name: "SV_Position".into(),
        index: 0,
        kind: 3,
        comp: 9,
        interp: 4,
        row: Some(1),
        col: 0,
        cols: 4,
        usage: 0b1111,
        half: false,
    });

    let entry = s.entry;
    let vid_args = [s.m.i32(0), s.m.i32(0), s.m.i8(0), s.m.undef(s.t.i32)];
    let vid = s.op(
        entry,
        "loadInput.i32",
        Attr::ReadNone,
        s.t.i32,
        4,
        &vid_args,
    );
    let k1 = s.m.i32(1);
    let k2 = s.m.i32(2);
    let is1 = s.m.cmp(entry, pred::EQ, vid, k1);
    let is2 = s.m.cmp(entry, pred::EQ, vid, k2);

    let pick = |s: &mut Shader, v0: f32, v1: f32, v2: f32| {
        let f0 = s.m.f32(v0.to_bits());
        let f1 = s.m.f32(v1.to_bits());
        let f2 = s.m.f32(v2.to_bits());
        let t = s.m.select(entry, is1, f1, f0);
        s.m.select(entry, is2, f2, t)
    };

    // Clip-space positions (x*w, y*w, 0, w) inside NDC [-1, 1]^2 (so neither
    // hardware nor WARP clips the triangle) with w = 1.0, 2.0, 4.0 so
    // perspective vs linear barycentrics differ strongly.
    let pos = [
        pick(&mut s, -1.0, 2.0, -4.0),
        pick(&mut s, -1.0, -2.0, 4.0),
        pick(&mut s, 0.0, 0.0, 0.0),
        pick(&mut s, 1.0, 2.0, 4.0),
    ];
    let tc = [
        pick(&mut s, 1.0, 2.0, 4.0),
        pick(&mut s, 10.0, 20.0, 40.0),
        pick(&mut s, 0.25, 0.5, 0.75),
        pick(&mut s, -1.0, 0.5, 3.0),
    ];

    for (col, v) in tc.into_iter().enumerate() {
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col as u8), v];
        s.op(entry, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
    }
    for (col, v) in pos.into_iter().enumerate() {
        let args = [s.m.i32(1), s.m.i32(0), s.m.i8(col as u8), v];
        s.op(entry, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
    }
    s.m.terminate(entry, Term::Ret);

    HostShader {
        stage: HwStage::Vs,
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
    }
}

fn translate_ps(words: &[u32], input_addr: u32) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 16,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: vec![PsInput {
                usage: Usage::Generic,
                row: 0,
                mask: 0b1111,
                attr: 0,
                flat: false,
                fp16: false,
                pack2: false,
                row2: 0,
            }],
            input_addr,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let el = |name: &str, kind, usage| SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage,
        half: false,
    };
    let isg1 = container::signature(&[el("TEXCOORD", 0, 0)], true);
    let osg1 = container::signature(&[el("SV_Target", 16, 15)], false);
    let bytes = build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes).unwrap();
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    emit(&c, &native, &p, &st, &io, &res).expect("emit PS")
}

/// Encodes `v_interp_p1_f32 v[dst], v[src_i], attr0.[ch]`.
fn p1(dst: u32, src_i: u32, ch: u32) -> u32 {
    0xC800_0000 | (dst << 18) | (ch << 8) | src_i
}

/// Encodes `v_interp_p2_f32 v[dst], v[src_j], attr0.[ch]`.
fn p2(dst: u32, src_j: u32, ch: u32) -> u32 {
    0xC801_0000 | (dst << 18) | (ch << 8) | src_j
}

/// Encodes `v_add_f32 v[dst], 0.0, v[src]` (VOP2, forces computed barycentrics).
fn add_zero(dst: u32, src: u32) -> u32 {
    0x0600_0080 | (dst << 17) | (src << 9)
}

/// Builds a PS that interpolates all 4 channels of `attr0` using `input_addr`.
/// When `computed` is true, adds `0.0` to `v0`/`v1` so the lowering takes the
/// hand-interpolation (`SV_Barycentrics` + `attributeAtVertex`) path.
fn interp_ps(input_addr: u32, computed: bool) -> HostShader {
    let mut words = Vec::new();
    let (vi, vj) = if computed {
        words.push(add_zero(6, 0));
        words.push(add_zero(7, 1));
        (6, 7)
    } else {
        (0, 1)
    };
    for ch in 0..4 {
        words.push(p1(2 + ch, vi, ch));
        words.push(p2(2 + ch, vj, ch));
    }
    // exp mrt0, v2, v3, v4, v5 done vm
    words.extend([0xF800_180F, 0x0504_0302, S_ENDPGM]);
    translate_ps(&words, input_addr)
}

#[cfg(windows)]
fn render_pair(vs: &HostShader, ps: &HostShader, label: &str, warp: bool) -> Option<Vec<[f32; 4]>> {
    use super::host::{Draw, Host};
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS3, D3D12_FEATURE_DATA_D3D12_OPTIONS3,
    };

    let Ok(host) = Host::new(warp) else {
        eprintln!("no SM 6.6 {label} device; skipped");
        return None;
    };
    let mut opts3 = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
    // SAFETY: query writes its own `D3D12_FEATURE_DATA_D3D12_OPTIONS3` struct.
    let bary_ok = unsafe {
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS3,
                (&raw mut opts3).cast(),
                size_of_val(&opts3) as u32,
            )
            .is_ok()
    } && opts3.BarycentricsSupported.as_bool();
    if !bary_ok {
        eprintln!("no BarycentricsSupported on {label} device; skipped");
        return None;
    }
    let (root, pso) = host
        .pipeline(vs, ps)
        .unwrap_or_else(|e| panic!("{label}: pipeline failed: {e}"));
    let out = host
        .draw(
            &root,
            &pso,
            &Draw {
                roots: Vec::new(),
                textures: &[],
                samplers: &[],
                vertices: &[],
                stride: 0,
                size: (8, 8),
                mesh_groups: 0,
            },
        )
        .unwrap_or_else(|e| panic!("{label}: draw failed: {e}"));
    Some(out)
}

/// Compares hand interpolation (`SV_Barycentrics` + `attributeAtVertex`)
/// against native host interpolation (`loadInput`) across perspective,
/// linear, perspective-centroid, and linear-centroid modes on hardware GPU
/// and WARP.
#[test]
fn hand_interpolation_matches_host_for_perspective_linear_and_centroid() {
    // (name, SPI_PS_INPUT_ADDR bit, host DXIL interp mode)
    let modes = [
        ("perspective_center", 1u32 << 1, 2u8),
        ("perspective_centroid", 1u32 << 2, 3u8),
        ("linear_center", 1u32 << 4, 4u8),
        ("linear_centroid", 1u32 << 5, 5u8),
    ];

    let mut pairs = Vec::new();
    for &(name, addr, dxil_mode) in &modes {
        let mut vs_host = perspective_triangle_vs(dxil_mode);
        let mut vs_hand = perspective_triangle_vs(1);
        let mut ps_host = interp_ps(addr, false);
        let mut ps_hand = interp_ps(addr, true);

        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            for s in [&mut vs_host, &mut vs_hand, &mut ps_host, &mut ps_hand] {
                v.validate(&mut s.dxil)
                    .unwrap_or_else(|e| panic!("{name}: DXIL validation failed: {e}"));
            }
        }
        #[cfg(not(windows))]
        let _ = (&mut vs_host, &mut vs_hand, &mut ps_host, &mut ps_hand);
        pairs.push((name, vs_host, ps_host, vs_hand, ps_hand));
    }

    #[cfg(windows)]
    {
        use super::valu_tests::{ONE_AT_A_TIME, backends};

        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let force_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
        let mut ran = 0;
        for &(label, warp) in backends(force_warp) {
            let mut persp_pixels = None;
            let mut linear_pixels = None;
            let mut backend_ok = true;
            for (name, vs_host, ps_host, vs_hand, ps_hand) in &pairs {
                let Some(host_out) = render_pair(vs_host, ps_host, label, warp) else {
                    backend_ok = false;
                    break;
                };
                let hand_out = render_pair(vs_hand, ps_hand, label, warp).unwrap();
                assert_eq!(host_out.len(), hand_out.len());
                let max_diff = host_out
                    .iter()
                    .flatten()
                    .zip(hand_out.iter().flatten())
                    .map(|(&a, &b)| {
                        assert!(
                            a.is_finite() && b.is_finite(),
                            "{label}/{name}: non-finite output (host={a}, hand={b})"
                        );
                        (a - b).abs()
                    })
                    .fold(0.0f32, f32::max);
                assert!(
                    max_diff < 1e-4,
                    "{label}/{name}: hand interpolation diverged from host (max_diff={max_diff})"
                );
                if *name == "perspective_center" {
                    persp_pixels = Some(hand_out);
                } else if *name == "linear_center" {
                    linear_pixels = Some(hand_out);
                }
            }
            if !backend_ok {
                continue;
            }
            // Verify perspective and linear actually produced distinct results.
            let persp = persp_pixels.unwrap();
            let linear = linear_pixels.unwrap();
            let mode_diff = persp
                .iter()
                .flatten()
                .zip(linear.iter().flatten())
                .map(|(&a, &b)| {
                    assert!(
                        a.is_finite() && b.is_finite(),
                        "{label}: non-finite perspective/linear output (persp={a}, linear={b})"
                    );
                    (a - b).abs()
                })
                .fold(0.0f32, f32::max);
            assert!(
                mode_diff > 0.1,
                "{label}: expected perspective and linear to differ (mode_diff={mode_diff})"
            );
            ran += 1;
        }
        assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
    }
}

/// Verifies directly read barycentrics (`SV_Barycentrics`) and a standalone
/// `v_interp_p1_f32` (`P0 + i * (P1 - P0)`) exported directly as data on
/// hardware GPU and WARP.
#[test]
fn direct_barycentrics_and_first_half_on_hardware() {
    // v0 = i (PerspectiveCenter), v1 = j (PerspectiveCenter)
    // v2 = i, v3 = j, v4 = v_interp_p1_f32(v0, attr0.x) = 1.0 + i * (2.0 - 1.0) = 1.0 + i
    // v5 = v_interp_mov_f32(p20, attr0.x) = 4.0 (vertex 2)
    let words = [
        p1(4, 0, 0), // v_interp_p1_f32 v4, v0, attr0.x -> 1.0 + i * (2.0 - 1.0)
        0xC816_0001, // v_interp_mov_f32 v5, p20, attr0.x -> 4.0 (vertex 2)
        0xF800_180F, // exp mrt0, v0, v1, v4, v5 done vm
        0x0504_0100,
        S_ENDPGM,
    ];
    let mut vs = perspective_triangle_vs(1);
    let mut ps = translate_ps(&words, 1 << 1);

    #[cfg(windows)]
    {
        use super::valu_tests::{ONE_AT_A_TIME, backends};

        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil).expect("VS validation");
            v.validate(&mut ps.dxil).expect("PS validation");
        }
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let force_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
        let mut ran = 0;
        for &(label, warp) in backends(force_warp) {
            let Some(out) = render_pair(&vs, &ps, label, warp) else {
                continue;
            };
            assert_eq!(out.len(), 64);
            let mut covered = 0;
            for (idx, &[i, j, p1_x, v2_x]) in out.iter().enumerate() {
                if v2_x == 0.0 {
                    continue;
                }
                covered += 1;
                // Independently derive perspective-correct (i, j) for this pixel
                // center from triangle vertices (-1,-1,w=1), (1,-1,w=2), (-1,1,w=4).
                let (px, py) = (idx % 8, idx / 8);
                let ndc_x = ((px as f32 + 0.5) / 8.0) * 2.0 - 1.0;
                let ndc_y = 1.0 - ((py as f32 + 0.5) / 8.0) * 2.0;
                let l1 = (ndc_x + 1.0) * 0.5;
                let l2 = (ndc_y + 1.0) * 0.5;
                let l0 = 1.0 - l1 - l2;
                let (q0, q1, q2) = (l0, l1 / 2.0, l2 / 4.0);
                let sum = q0 + q1 + q2;
                let expected_i = q1 / sum;
                let expected_j = q2 / sum;
                assert!(
                    (0.0..=1.0).contains(&i) && (0.0..=1.0).contains(&j) && i + j <= 1.001,
                    "{label}: pixel {idx} out-of-range barycentrics ({i}, {j})"
                );
                assert!(
                    (i - expected_i).abs() < 1e-4 && (j - expected_j).abs() < 1e-4,
                    "{label}: pixel {idx} barycentrics mismatch: got ({i}, {j}), expected ({expected_i}, {expected_j})"
                );
                let expected_p1 = 1.0 + expected_i * (2.0 - 1.0);
                assert!(
                    (p1_x - expected_p1).abs() < 1e-4,
                    "{label}: pixel {idx} p1 mismatch: got {p1_x}, expected {expected_p1}"
                );
                assert!(
                    (v2_x - 4.0).abs() < 1e-4,
                    "{label}: pixel {idx} vertex 2 mismatch: got {v2_x}, expected 4.0"
                );
            }
            assert!(
                covered >= 24,
                "{label}: expected at least 24 covered pixels, got {covered}"
            );
            ran += 1;
        }
        assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
    }
    #[cfg(not(windows))]
    let _ = (&mut vs, &mut ps);
}

#[test]
fn barycentrics_disabled_refuses_hand_and_direct_paths() {
    let try_emit = |words: &[u32], allow_barycentrics: bool| {
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 16,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 0b1111,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 1 << 1,
                db_shader_control: 0,
                patches: Vec::new(),
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, usage| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage,
            half: false,
        };
        let isg1 = container::signature(&[el("TEXCOORD", 0, 0)], true);
        let osg1 = container::signature(&[el("SV_Target", 16, 15)], false);
        let bytes = build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        super::emit_with_caps(&c, &native, &p, &st, &io, &res, true, allow_barycentrics)
    };

    // Normal host interpolation succeeds even without SV_Barycentrics.
    let host_words = [p1(2, 0, 0), p2(2, 1, 0), 0xF800_180F, 0x0202_0202, S_ENDPGM];
    assert!(try_emit(&host_words, false).is_ok());

    // Computed barycentrics require SV_Barycentrics / attributeAtVertex.
    let computed_words = [
        add_zero(6, 0),
        add_zero(7, 1),
        p1(2, 6, 0),
        p2(2, 7, 0),
        0xF800_180F,
        0x0202_0202,
        S_ENDPGM,
    ];
    assert!(try_emit(&computed_words, false).is_err());
    assert!(try_emit(&computed_words, true).is_ok());

    // Directly read barycentrics require SV_Barycentrics.
    let direct_words = [0xF800_180F, 0x0100_0100, S_ENDPGM];
    assert!(try_emit(&direct_words, false).is_err());
    assert!(try_emit(&direct_words, true).is_ok());
}
