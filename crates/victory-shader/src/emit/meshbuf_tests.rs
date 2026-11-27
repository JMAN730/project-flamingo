//! The sample mesh program (`super::sample`) drawn the way hosts without
//! mesh shaders draw it (`super::meshbuf`): its compute half fills the
//! buffer, the generated vertex shader reads it back, and the texels are the
//! ones the program describes. Where a host has mesh shaders, the same
//! program as a mesh shader must draw the same texels.

use super::host::{Draw, Fill, Host, Root};
use super::{sample, validator};

const SIZE: (u32, u32) = (16, 16);

fn assert_close(what: &str, got: &[[f32; 4]], want: &[[f32; 4]]) {
    assert_eq!(got.len(), want.len(), "{what}: texel count");
    for (n, (g, w)) in got.iter().zip(want).enumerate() {
        let close = g.iter().zip(w).all(|(a, b)| (a - b).abs() <= 1e-4);
        assert!(close, "{what}: texel {n} is {g:?}, not {w:?}");
    }
}

fn draw(size: (u32, u32)) -> Draw<'static> {
    Draw {
        roots: vec![(0, Root::Constants(vec![0])), (1, Root::Constants(vec![0]))],
        textures: &[],
        samplers: &[],
        vertices: &[],
        stride: 0,
        size,
        mesh_groups: 1,
    }
}

#[test]
fn a_mesh_program_draws_the_same_without_mesh_shaders() {
    draws_the_sample(sample::mesh());
}

const fn sop1(op: u32, sdst: u32, src0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src0
}
const fn sop2(op: u32, sdst: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0
}
const VCC_LO: u32 = 106;
const EXEC_LO: u32 = 126;
/// `v_cmp_eq_u32 vcc, v1, v1`: every lane (v1 is the lane's index).
const V_CMP_ALL: u32 = 0x7D84_0000 | 1 << 9 | 257;
/// `v_cvt_f32_u32 v13, s20` and `v_add_f32 v6, v6, v13`.
const V_CVT_F32_U32_V13_S20: u32 = 0x7E00_0000 | 13 << 17 | 0x06 << 9 | 20;
const V_ADD_F32_V6_V13: u32 = 0x03 << 25 | 6 << 17 | 13 << 9 | (256 + 6);

/// A whole-wave count of the lanes whose compare differs from EXEC (the
/// console compiler's "is every lane equal" test, here none of them) is
/// read in a mesh program, and added to the position's w: 1.0 + 0.
#[test]
fn a_mesh_program_counts_the_lanes_a_compare_differs_in() {
    let extra = [
        V_CMP_ALL,
        sop2(0x13, 20, VCC_LO, EXEC_LO), // s_xor_b64 s[20:21], vcc, exec
        sop1(0x10, 20, 20),              // s_bcnt1_i32_b64 s20, s[20:21]
        V_CVT_F32_U32_V13_S20,
        V_ADD_F32_V6_V13,
    ];
    draws_the_sample(sample::mesh_with(&extra));
}

/// Lanes are not numbered like the console's in a mesh program, so the
/// lowest set lane of a mask (a lane number) stays refused.
#[test]
fn a_mesh_program_does_not_read_a_lane_number_from_a_mask() {
    let extra = [
        V_CMP_ALL,
        sop1(0x13, 20, VCC_LO), // s_ff1_i32_b32 s20, vcc_lo
        V_CVT_F32_U32_V13_S20,
        V_ADD_F32_V6_V13,
    ];
    let e = sample::mesh_error(&extra).expect("refused");
    assert!(e.to_string().contains("lane mask"), "{e}");
}

/// A register that held a lane mask's dword and gets data on the way to a
/// join is read after the join: the mask edge is never taken, and the data
/// reaches the position's w (1.0).
#[test]
fn a_mesh_program_joins_a_dead_mask_dword_with_data() {
    const LITERAL: u32 = 255;
    let extra = [
        V_CMP_ALL,
        sop1(0x04, 20, VCC_LO),                      // s_mov_b64 s[20:21], vcc
        0xBF00_0000 | 0x07 << 16 | 0x80 << 8 | 0x80, // s_cmp_lg_u32 0, 0: SCC clear
        0xBF85_0002,                                 // s_cbranch_scc1 over the next two words
        sop1(0x03, 21, LITERAL),                     // s_mov_b32 s21, 1.0
        0x3F80_0000,
        0x7E00_0000 | 6 << 17 | 0x01 << 9 | 21, // v_mov_b32 v6, s21
    ];
    draws_the_sample(sample::mesh_with(&extra));
}

fn draws_the_sample(
    sample::MeshSample {
        mesh,
        fallback,
        pixel: ps,
    }: sample::MeshSample,
) {
    // Three vertices of 4 + 4 dwords; one primitive of its indices and 2.
    let buffer = &fallback.buffer;
    assert_eq!((buffer.vertex_dwords, buffer.primitive_dwords), (8, 3));
    assert_eq!(buffer.group_dwords(), 3 * 8 + 3);
    assert_eq!(buffer.draw_vertices(), 3);

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let shaders = [
            ("mesh", &mesh),
            ("compute half", &fallback.compute),
            ("vertex half", &fallback.vertex),
            ("pixel", &ps),
        ];
        for (what, shader) in shaders {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy)
                .unwrap_or_else(|e| panic!("{what}: DXIL validation failed: {e}"));
        }
    }

    let want = sample::mesh_texels(SIZE.0, SIZE.1);
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let ((compute_root, compute_pso), (root, pso)) = host
            .mesh_fallback_pipelines(&fallback, &ps)
            .unwrap_or_else(|e| panic!("{label}: fallback pipelines: {e}"));
        // Group 0, then a part of a dispatch that starts at group 5: the
        // program counts from the number it is told.
        for first_group in [0, 5] {
            let fill = Fill {
                root: &compute_root,
                pso: &compute_pso,
                user_data: vec![0],
                uav: 1,
                srv: 2,
                bytes: u64::from(fallback.buffer.group_dwords()) * 4,
                vertices: fallback.buffer.draw_vertices(),
                first_group,
            };
            let got = host
                .draw_filled(&root, &pso, &draw(SIZE), Some(&fill))
                .unwrap_or_else(|e| panic!("{label}: fallback draw: {e}"));
            let want = sample::group_texels(SIZE.0, SIZE.1, first_group);
            let what = format!("{label}, without mesh shaders, from group {first_group}");
            assert_close(&what, &got, &want);
        }
        ran += 1;
        // The same program as a mesh shader, where the host runs those.
        match host.mesh_pipeline(&mesh, &ps) {
            Ok((root, pso)) => {
                let got = host
                    .draw(&root, &pso, &draw(SIZE))
                    .unwrap_or_else(|e| panic!("{label}: mesh draw: {e}"));
                assert_close(&format!("{label}, as a mesh shader"), &got, &want);
                eprintln!("{label}: mesh shader and fallback agree");
            }
            Err(e) => eprintln!("{label}: no mesh pipeline ({e}); fallback checked alone"),
        }
    }
    if ran == 0 {
        eprintln!("no SM 6.6 device; skipped");
    }
}
