//! Pixel shaders that compute the weights they interpolate with, or read
//! the barycentric registers as data. The host gives them SM 6.1
//! `GetAttributeAtVertex` and `SV_Barycentrics` (last chance of the
//! emitter), checked here against the weights of a known triangle; a host
//! without them gets the same from the geometry shader of `bary.rs`,
//! checked against the same triangle and a strip on every device.

use super::container::{SigElement, signature};
use super::{HostShader, emit};
use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, PsInput, StageIo, Usage, VsInput, VsOutput};

/// The vertices' attribute values: x of TEXCOORD0 at each.
const ATTR: [f32; 3] = [2.0, 10.0, 34.0];
/// Channel y of the same attribute.
const ATTR_Y: [f32; 3] = [0.5, -1.5, 4.5];

fn element(name: &str, index: u32, kind: u8, row: u32, cols: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind,
        comp: 9,
        interp: 0,
        row: Some(row),
        col: 0,
        cols,
        usage: (1 << cols) - 1,
        half: false,
    }
}

fn translate(
    stage: HwStage,
    io: StageIo,
    words: &[u32],
    ins: &[SigElement],
    outs: &[SigElement],
) -> Result<HostShader, crate::ShaderError> {
    translate_for(false, stage, io, words, ins, outs)
}

/// As [`translate`]; `fallback` for a host without barycentrics.
fn translate_for(
    fallback: bool,
    stage: HwStage,
    io: StageIo,
    words: &[u32],
    ins: &[SigElement],
    outs: &[SigElement],
) -> Result<HostShader, crate::ShaderError> {
    let isg = signature(ins, true);
    let osg = signature(outs, false);
    let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg), (FourCC::OSG1, &osg)]);
    let native = NativeProgram {
        stage,
        wave64: false,
        vgprs: 12,
        layout: vec![],
        used: vec![],
        io,
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::ir::build(&native.code, false, native.vgprs)?;
    let st = crate::structure::structure(&p)?;
    let io = crate::io::raise(&p, &native)?;
    let res = crate::resource::track(&p, &native.layout, stage, Some(&st))?;
    let c = Container::parse(&bytes).unwrap();
    if fallback {
        super::emit_without_barycentrics(&c, &native, &p, &st, &io, &res)
    } else {
        emit(&c, &native, &p, &st, &io, &res)
    }
}

fn varying() -> SigElement {
    element("TEXCOORD", 0, 0, 0, 3)
}

fn vertex_shader() -> HostShader {
    let outs = [varying(), element("SV_Position", 0, 3, 1, 4)];
    translate(
        HwStage::Vs,
        StageIo::Vs {
            inputs: [(0, 4, 15), (1, 8, 7)]
                .map(|(row, vgpr, mask)| VsInput {
                    usage: Usage::Generic,
                    row,
                    vgpr,
                    mask,
                })
                .to_vec(),
            outputs: vec![VsOutput {
                usage: Usage::Generic,
                row: 0,
                param: 0,
                fp16: false,
                pack2: false,
            }],
        },
        &[
            0xF800_0207,
            0x000A_0908, // exp param0 v8, v9, v10, off
            0xF800_08CF,
            0x0706_0504, // exp pos0 v4, v5, v6, v7 done
            0xBF81_0000,
        ],
        &[
            element("POSITION", 0, 0, 0, 4),
            element("TEXCOORD", 0, 0, 1, 3),
        ],
        &outs,
    )
    .expect("vertex shader")
}

/// A pixel shader reading TEXCOORD0 with the barycentrics of `input_addr`
/// (bit 1: perspective center, in v0 and v1), exporting `words`' MRT0.
fn pixel_shader(words: &[u32]) -> Result<HostShader, crate::ShaderError> {
    pixel_shader_for(false, words)
}

/// [`pixel_shader`] for a host without barycentrics.
fn fallback_pixel_shader(words: &[u32]) -> HostShader {
    pixel_shader_for(true, words).expect("translate without barycentrics")
}

fn pixel_shader_for(fallback: bool, words: &[u32]) -> Result<HostShader, crate::ShaderError> {
    translate_for(
        fallback,
        HwStage::Ps,
        StageIo::Ps {
            inputs: vec![PsInput {
                usage: Usage::Generic,
                row: 0,
                mask: 7,
                attr: 0,
                flat: false,
                fp16: false,
                pack2: false,
                row2: 0,
            }],
            input_addr: 1 << 1,
            rsrc2: Default::default(),
            db_shader_control: 0,
            patches: vec![],
        },
        words,
        &[varying()],
        &[element("SV_Target", 0, 16, 0, 4)],
    )
}

const V_SUB_F32_V2_ONE_V0: u32 = 0x0804_00F2; // v_sub_f32 v2, 1.0, v0
const V_SUB_F32_V2_V2_V1: u32 = 0x0804_0302; // v_sub_f32 v2, v2, v1
const EXPORT: [u32; 2] = [0xF800_180F, 0x0000_0000]; // exp mrt0 ... done vm; sources below
const S_ENDPGM: u32 = 0xBF81_0000;

/// Weights `1 - i` and `j` into the attribute's two halves.
fn computed_weights() -> Vec<u32> {
    vec![
        V_SUB_F32_V2_ONE_V0,
        0xC80C_0002, // v_interp_p1_f32 v3, v2, attr0.x
        0xC80D_0001, // v_interp_p2_f32 v3, v1, attr0.x
        EXPORT[0],
        0x0303_0303, // v3 in all four channels
        S_ENDPGM,
    ]
}

/// Constant weights 0.5 and 2.0 (no barycentrics involved).
fn constant_weights() -> Vec<u32> {
    vec![
        0x7E04_02F0, // v_mov_b32 v2, 0.5
        0x7E06_02F4, // v_mov_b32 v3, 2.0
        0xC810_0002, // v_interp_p1_f32 v4, v2, attr0.x
        0xC811_0003, // v_interp_p2_f32 v4, v3, attr0.x
        EXPORT[0],
        0x0404_0404,
        S_ENDPGM,
    ]
}

/// Channel x interpolated as the hardware does, channel y with the weights
/// 1 - i and j: one attribute read both ways.
fn mixed() -> Vec<u32> {
    vec![
        0xC80C_0000, // v_interp_p1_f32 v3, v0, attr0.x
        0xC80D_0001, // v_interp_p2_f32 v3, v1, attr0.x
        V_SUB_F32_V2_ONE_V0,
        0xC810_0102, // v_interp_p1_f32 v4, v2, attr0.y
        0xC811_0101, // v_interp_p2_f32 v4, v1, attr0.y
        EXPORT[0],
        0x0403_0403, // v3, v4, v3, v4
        S_ENDPGM,
    ]
}

/// The barycentrics as data: i, j and 1 - i - j.
fn direct_read() -> Vec<u32> {
    vec![
        V_SUB_F32_V2_ONE_V0,
        V_SUB_F32_V2_V2_V1,
        EXPORT[0],
        0x0202_0100, // v0, v1, v2, v2
        S_ENDPGM,
    ]
}

fn validate(shader: &HostShader) {
    match super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        Ok(mut validator) => validator
            .validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed"),
        Err(e) => eprintln!("{e}; validation skipped"),
    }
}

/// The signature element `name` of `shader`'s input.
fn input_named(shader: &HostShader, name: &str) -> Option<SigElement> {
    super::signature_part(&Container::parse(&shader.dxil).unwrap(), FourCC::ISG1)
        .unwrap()
        .into_iter()
        .find(|e| e.name == name)
}

#[test]
fn computed_weights_read_the_attribute_at_each_vertex() {
    let ps = pixel_shader(&computed_weights()).expect("translate");
    validate(&ps);
    // The weights are the program's own (1 - i) of the host's barycentrics.
    assert!(input_named(&ps, "SV_Barycentrics").is_some());
}

#[test]
fn constant_weights_need_no_host_barycentrics() {
    let ps = pixel_shader(&constant_weights()).expect("translate");
    validate(&ps);
    assert!(input_named(&ps, "SV_Barycentrics").is_none());
}

#[test]
fn barycentrics_read_as_data_come_from_the_host() {
    let ps = pixel_shader(&direct_read()).expect("translate");
    validate(&ps);
    let bary = input_named(&ps, "SV_Barycentrics").expect("SV_Barycentrics");
    assert_eq!((bary.kind, bary.cols, bary.row), (28, 3, None));
}

/// A barycentric register some lanes overwrite on one path, joined, then
/// read only as the high half of 64-bit shifts that drop it: none of its
/// bits reach the export, so the host's barycentrics are not needed (hosts
/// without them can run it).
#[test]
fn a_barycentric_whose_bits_are_shifted_out_needs_no_host_barycentrics() {
    let ps = pixel_shader(&[
        0xC80C_0000, // v_interp_p1_f32 v3, v0, attr0.x
        0xC80D_0001, // v_interp_p2_f32 v3, v1, attr0.x
        0xBE84_037E, // s_mov_b32 s4, exec_lo
        0x7C22_06F0, // v_cmpx_lt_f32 exec_lo, 0.5, v3
        0xBF88_0001, // s_cbranch_execz over the next one
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0x7E00_0F03, // v_cvt_u32_f32 v0, v3
        0xD6FF_0000,
        0x0002_0097, // v_lshlrev_b64 v[0:1], 23, v[0:1]
        0xD6FF_0000,
        0x0002_0091, // v_lshlrev_b64 v[0:1], 17, v[0:1]
        EXPORT[0],
        0x0101_0101, // v1 in all four channels
        S_ENDPGM,
    ])
    .expect("translate");
    validate(&ps);
    assert!(input_named(&ps, "SV_Barycentrics").is_none());
}

/// The three vertices' values and the weights of pixel (x, y) in the
/// 2x2 target for the triangle (-1,-1), (-1,3), (3,-1): i weighs the second
/// vertex, j the third.
fn weights(px: u32, py: u32) -> (f32, f32) {
    let (x, y) = (px as f32 - 0.5, 0.5 - py as f32);
    ((y + 1.0) / 4.0, (x + 1.0) / 4.0)
}

fn run(ps: &HostShader, host: &super::host::Host) -> Vec<[f32; 4]> {
    use super::host::Draw;
    let vs = vertex_shader();
    let vertices: Vec<f32> = [[-1.0, -1.0], [-1.0, 3.0], [3.0, -1.0]]
        .into_iter()
        .enumerate()
        .flat_map(|(k, [x, y])| [x, y, 0.5, 1.0, ATTR[k], ATTR_Y[k], 0.0])
        .collect();
    let (root, pso) = host.pipeline(&vs, ps).expect("pipeline");
    host.draw(
        &root,
        &pso,
        &Draw {
            roots: vec![],
            textures: &[],
            samplers: &[],
            vertices: &vertices,
            stride: 28,
            size: (2, 2),
            mesh_groups: 0,
        },
    )
    .expect("draw")
}

/// Whether `host` reports barycentrics (and so `GetAttributeAtVertex`).
fn barycentrics_supported(host: &super::host::Host) -> bool {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS3, D3D12_FEATURE_DATA_D3D12_OPTIONS3,
    };
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
    // SAFETY: the options struct matches the feature asked for.
    let ok = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS3,
            (&raw mut options).cast(),
            size_of_val(&options) as u32,
        )
    }
    .is_ok();
    ok && options.BarycentricsSupported.as_bool()
}

fn close(got: f32, want: f32, what: &str) {
    assert!(
        (got - want).abs() <= 1e-4 * want.abs().max(1.0),
        "{what}: {got} != {want}"
    );
}

/// The outputs of the four programs drawn over the reference triangle
/// (computed, direct, constant, mixed) against what the console computes.
fn check_reference(label: &str, [out, direct_out, constant_out, mixed_out]: [&[[f32; 4]]; 4]) {
    for py in 0..2u32 {
        for px in 0..2u32 {
            let (i, j) = weights(px, py);
            let at = (py * 2 + px) as usize;
            // P0 + (1 - i) * P10 + j * P20
            let want = ATTR[0] + (1.0 - i) * (ATTR[1] - ATTR[0]) + j * (ATTR[2] - ATTR[0]);
            for v in out[at] {
                close(v, want, &format!("{label} computed ({px},{py})"));
            }
            // P0 + 0.5 * P10 + 2 * P20, whatever the pixel.
            let want = ATTR[0] + 0.5 * (ATTR[1] - ATTR[0]) + 2.0 * (ATTR[2] - ATTR[0]);
            close(
                constant_out[at][0],
                want,
                &format!("{label} constant ({px},{py})"),
            );
            let x = ATTR[0] + i * (ATTR[1] - ATTR[0]) + j * (ATTR[2] - ATTR[0]);
            let y = ATTR_Y[0] + (1.0 - i) * (ATTR_Y[1] - ATTR_Y[0]) + j * (ATTR_Y[2] - ATTR_Y[0]);
            close(mixed_out[at][0], x, &format!("{label} mixed x ({px},{py})"));
            close(mixed_out[at][1], y, &format!("{label} mixed y ({px},{py})"));
            close(direct_out[at][0], i, &format!("{label} i ({px},{py})"));
            close(direct_out[at][1], j, &format!("{label} j ({px},{py})"));
            close(
                direct_out[at][2],
                1.0 - i - j,
                &format!("{label} 1-i-j ({px},{py})"),
            );
        }
    }
}

#[test]
fn computed_and_direct_barycentrics_match_a_reference_on_gpu() {
    use super::host::Host;
    let computed = pixel_shader(&computed_weights()).expect("translate");
    let direct = pixel_shader(&direct_read()).expect("translate");
    let constant = pixel_shader(&constant_weights()).expect("translate");
    let mixed = pixel_shader(&mixed()).expect("translate");
    validate(&mixed);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // Hardware only where it reports support; WARP always does.
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("{label}: no device");
            continue;
        };
        let supported = barycentrics_supported(&host);
        eprintln!("{label}: barycentrics supported: {supported}");
        if !supported {
            continue;
        }
        let outs = [&computed, &direct, &constant, &mixed].map(|ps| run(ps, &host));
        check_reference(label, outs.each_ref().map(Vec::as_slice));
        eprintln!("{label}: computed and direct barycentrics matched");
        ran += 1;
    }
    assert!(
        ran > 0 || std::env::var_os("VICTORY_NO_DEVICE_OK").is_some(),
        "no host"
    );
}

/// The four programs translated for a host without barycentrics.
fn fallback_programs() -> [HostShader; 4] {
    [
        computed_weights(),
        direct_read(),
        constant_weights(),
        mixed(),
    ]
    .map(|w| fallback_pixel_shader(&w))
}

/// The names of `shader`'s inputs.
fn input_names(shader: &HostShader) -> Vec<(String, u32)> {
    super::signature_part(&Container::parse(&shader.dxil).unwrap(), FourCC::ISG1)
        .unwrap()
        .into_iter()
        .map(|e| (e.name, e.index))
        .collect()
}

/// Without host barycentrics, the values at the vertices are constant
/// inputs of their own and the weights an input interpolated as the
/// console's (perspective, at the centre: mode 2); the programs need
/// neither `SV_Barycentrics` nor `GetAttributeAtVertex`, and validate.
#[test]
fn fallback_translations_need_no_host_barycentrics() {
    let [computed, direct, constant, both] = fallback_programs();
    let copies = [0, 1, 2].map(|k| (super::bary::vertex_copy_name(k, "TEXCOORD"), 0));
    let weights = (super::bary::BARYCENTRICS.to_owned(), 2);
    for (label, shader, want_copies, want_weights) in [
        ("computed", &computed, true, true),
        ("direct", &direct, false, true),
        ("constant", &constant, true, false),
        ("mixed", &both, true, true),
    ] {
        validate(shader);
        assert!(!shader.needs_barycentrics(), "{label}");
        assert!(input_named(shader, "SV_Barycentrics").is_none(), "{label}");
        let names = input_names(shader);
        for copy in &copies {
            assert_eq!(names.contains(copy), want_copies, "{label}: {copy:?}");
        }
        assert_eq!(names.contains(&weights), want_weights, "{label}");
        let gs = super::barycentric_gs(&vertex_shader().dxil, &shader.dxil).expect(label);
        validate_gs(&gs);
    }
    // On hosts with barycentrics the same programs use them.
    for words in [
        computed_weights(),
        direct_read(),
        constant_weights(),
        mixed(),
    ] {
        assert!(pixel_shader(&words).unwrap().needs_barycentrics());
    }
}

fn validate_gs(gs: &[u8]) {
    match super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        Ok(mut validator) => validator
            .validate(&mut gs.to_vec())
            .expect("DXIL validation failed"),
        Err(e) => eprintln!("{e}; validation skipped"),
    }
}

/// A program that needs no barycentrics comes out byte for byte the same
/// when translated for a host without them.
#[test]
fn programs_without_barycentrics_translate_the_same_either_way() {
    let plain = [
        0xC80C_0000, // v_interp_p1_f32 v3, v0, attr0.x
        0xC80D_0001, // v_interp_p2_f32 v3, v1, attr0.x
        EXPORT[0],
        0x0303_0303,
        S_ENDPGM,
    ];
    let shifted = [
        0xC80C_0000, // v_interp_p1_f32 v3, v0, attr0.x
        0xC80D_0001, // v_interp_p2_f32 v3, v1, attr0.x
        0xBE84_037E, // s_mov_b32 s4, exec_lo
        0x7C22_06F0, // v_cmpx_lt_f32 exec_lo, 0.5, v3
        0xBF88_0001, // s_cbranch_execz over the next one
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0x7E00_0F03, // v_cvt_u32_f32 v0, v3
        0xD6FF_0000,
        0x0002_0097, // v_lshlrev_b64 v[0:1], 23, v[0:1]
        0xD6FF_0000,
        0x0002_0091, // v_lshlrev_b64 v[0:1], 17, v[0:1]
        EXPORT[0],
        0x0101_0101,
        S_ENDPGM,
    ];
    for words in [&plain[..], &shifted[..]] {
        let native = pixel_shader(words).expect("translate");
        assert!(!native.needs_barycentrics());
        assert_eq!(fallback_pixel_shader(words), native);
    }
}

/// `ps` translated without barycentrics, drawn behind the geometry shader
/// of `bary.rs` over `vertices` (position, then TEXCOORD0), as `command`
/// says, culling back faces.
fn run_fallback(
    ps: &HostShader,
    host: &super::host::Host,
    vertices: &[f32],
    size: (u32, u32),
    command: super::host::TriangleDraw,
) -> Vec<[f32; 4]> {
    use super::host::Draw;
    use windows::Win32::Graphics::Direct3D12::D3D12_CULL_MODE_BACK;
    let vs = vertex_shader();
    let gs = super::barycentric_gs(&vs.dxil, &ps.dxil).expect("geometry shader");
    let root = host.root_signature(&vs, ps).expect("root signature");
    let pso = host
        .pipeline_with_cull(&root, &vs.dxil, Some(&gs), &ps.dxil, D3D12_CULL_MODE_BACK)
        .expect("pipeline");
    host.draw_triangle(
        &root,
        &pso,
        &Draw {
            roots: vec![],
            textures: &[],
            samplers: &[],
            vertices,
            stride: 28,
            size,
            mesh_groups: 0,
        },
        command,
    )
    .expect("draw")
}

/// Position, then TEXCOORD0 (x, y and an unused z), per vertex.
fn vertex_data(positions: &[[f32; 2]], attr: impl Fn(usize, [f32; 2]) -> [f32; 2]) -> Vec<f32> {
    positions
        .iter()
        .enumerate()
        .flat_map(|(k, &p)| {
            let [x, y] = attr(k, p);
            [p[0], p[1], 0.5, 1.0, x, y, 0.0]
        })
        .collect()
}

/// The barycentric weights of `p` in the triangle `t`.
fn barycentric(p: [f32; 2], t: [[f32; 2]; 3]) -> [f32; 3] {
    let [a, b, c] = t;
    let area = |p: [f32; 2], q: [f32; 2], r: [f32; 2]| {
        (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])
    };
    let whole = area(a, b, c);
    [
        area(p, b, c) / whole,
        area(a, p, c) / whole,
        area(a, b, p) / whole,
    ]
}

/// On every device, hardware with or without barycentrics and WARP, the
/// programs translated without barycentrics give over the reference
/// triangle what the console computes: the computed weights, constant
/// weights, an attribute read both ways, and i, j and 1 - i - j read
/// directly.
#[test]
fn fallback_barycentrics_match_a_reference_on_gpu() {
    use super::host::{Host, TriangleDraw};
    let programs = fallback_programs();
    let vertices = vertex_data(&[[-1.0, -1.0], [-1.0, 3.0], [3.0, -1.0]], |k, _| {
        [ATTR[k], ATTR_Y[k]]
    });
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("{label}: no device");
            continue;
        };
        eprintln!(
            "{label}: barycentrics supported: {}",
            barycentrics_supported(&host)
        );
        let outs = programs
            .each_ref()
            .map(|ps| run_fallback(ps, &host, &vertices, (2, 2), TriangleDraw::Direct));
        check_reference(
            &format!("{label} without barycentrics"),
            outs.each_ref().map(Vec::as_slice),
        );
        eprintln!("{label}: barycentrics without host barycentrics matched");
        ran += 1;
    }
    assert!(
        ran > 0 || std::env::var_os("VICTORY_NO_DEVICE_OK").is_some(),
        "no host"
    );
}

/// A triangle strip of two triangles over a 4x4 target, back faces culled:
/// both triangles are drawn (the geometry shader keeps the strip's winding),
/// an attribute interpolated through the values at the vertices is the
/// plain interpolation everywhere, and in each triangle the weights read
/// directly are those of the pixel centre over the triangle's vertices in
/// one order, the same one the computed weights use. The first triangle's
/// order is the strip's; the second's is D3D's (it reorders a strip's odd
/// triangles), and keeps the winding.
#[test]
fn fallback_barycentrics_follow_a_strip_on_gpu() {
    use super::host::{Host, TriangleDraw};
    // The shared edge runs along x + y = 0.25, past every pixel centre;
    // the corners are on the rasterizer's grid (1/256 pixel), so that the
    // weights are those of these very positions.
    let strip = [[-1.0, -1.0], [-1.0, 1.25], [1.25, -1.0], [1.25, 1.25]];
    // An affine attribute: every triangle's interpolation is the function.
    let affine = |[x, y]: [f32; 2]| 2.0 + 3.0 * x + 5.0 * y;
    let values = |[x, y]: [f32; 2]| -4.0 + 7.0 * x - 2.0 * y * y;
    let vertices = vertex_data(&strip, |_, p| [affine(p), values(p)]);
    let [computed, direct, _, both] = fallback_programs();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("{label}: no device");
            continue;
        };
        let draw = |ps| run_fallback(ps, &host, &vertices, (4, 4), TriangleDraw::Strip);
        let (computed_out, direct_out, mixed_out) = (draw(&computed), draw(&direct), draw(&both));
        let mut odd_orders = Vec::new();
        for py in 0..4u32 {
            for px in 0..4u32 {
                let at = (py * 4 + px) as usize;
                let p = [(px as f32 + 0.5) / 2.0 - 1.0, 1.0 - (py as f32 + 0.5) / 2.0];
                let what = |w: &str| format!("{label} strip {w} ({px},{py})");
                close(mixed_out[at][0], affine(p), &what("mixed x"));
                let [i, j, rest] = [direct_out[at][0], direct_out[at][1], direct_out[at][2]];
                close(rest, 1.0 - i - j, &what("1-i-j"));
                // The triangle's vertices in the order whose weights these
                // are: the strip's for the first triangle, one of the second
                // triangle's three clockwise orders for the other.
                let first = p[0] + p[1] < 0.25;
                let orders: &[[usize; 3]] = if first {
                    &[[0, 1, 2]]
                } else {
                    &[[2, 1, 3], [1, 3, 2], [3, 2, 1]]
                };
                let order = orders.iter().copied().find(|&o| {
                    let w = barycentric(p, o.map(|k| strip[k]));
                    (w[0] - rest).abs() < 1e-4 && (w[1] - i).abs() < 1e-4 && (w[2] - j).abs() < 1e-4
                });
                let Some(order) = order else {
                    panic!("{}: weights {i} {j} fit no order", what("direct"));
                };
                if !first {
                    odd_orders.push(order);
                }
                // The computed weights: P0 + (1 - i) * P10 + j * P20, of
                // channel x (computed) and y (mixed).
                let computed = |f: &dyn Fn([f32; 2]) -> f32| {
                    let v = order.map(|k| f(strip[k]));
                    v[0] + (1.0 - i) * (v[1] - v[0]) + j * (v[2] - v[0])
                };
                close(computed_out[at][0], computed(&affine), &what("computed"));
                close(mixed_out[at][1], computed(&values), &what("mixed y"));
            }
        }
        odd_orders.dedup();
        assert_eq!(
            odd_orders.len(),
            1,
            "{label}: the second triangle's order varies"
        );
        eprintln!(
            "{label}: strip matched; second triangle as vertices {:?}",
            odd_orders[0]
        );
        ran += 1;
    }
    assert!(
        ran > 0 || std::env::var_os("VICTORY_NO_DEVICE_OK").is_some(),
        "no host"
    );
}

/// Validates the emitted shaders named in the file `VICTORY_VALIDATE_LIST`
/// (one path per line, from a `VICTORY_EMIT_DUMP` run outside the repo).
#[test]
#[ignore = "needs VICTORY_VALIDATE_LIST"]
fn listed_dumps_validate() {
    let list = std::env::var("VICTORY_VALIDATE_LIST").expect("VICTORY_VALIDATE_LIST");
    let mut validator =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
            .expect("dxil.dll");
    let mut failed = Vec::new();
    let mut total = 0;
    for path in std::fs::read_to_string(list).unwrap().lines() {
        let mut dxil = std::fs::read(path).unwrap();
        total += 1;
        if let Err(e) = validator.validate(&mut dxil) {
            failed.push(format!("{path}: {}", e.lines().next().unwrap_or("")));
        }
    }
    eprintln!("validated {} of {total}", total - failed.len());
    assert!(failed.is_empty(), "{failed:#?}");
}

/// The first half of an interpolation read by an in-place instruction
/// (`v_add_f32 v3, v3, v1`: v3 is an explicit source as well as the old
/// destination every vector write reads) is read as data. The last chance
/// must not take that read for the old destination alone and lower the
/// first half as zero, which would export the wrong value.
#[test]
fn an_in_place_reader_of_an_interpolation_first_half_reads_it() {
    let r = pixel_shader(&[
        0xC80C_0000, // v_interp_p1_f32 v3, v0, attr0.x
        0x0606_0303, // v_add_f32 v3, v3, v1
        EXPORT[0],
        0x0303_0303, // v3 in all four channels
        S_ENDPGM,
    ]);
    assert!(r.is_err(), "lowered with the first half as zero");
}
