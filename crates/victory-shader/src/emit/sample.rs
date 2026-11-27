//! A synthetic mesh program, hand-assembled, with the pixel shader that
//! shows what it draws: for tests of how mesh programs run, here and in the
//! crates that draw with them (feature `samples`).
//!
//! The program is one thread group of 64 lanes drawing one triangle over the
//! whole target. Lane n is vertex n: its texture coordinate is
//! (n & 1, n >> 1) and its position four times that minus one, so the
//! coordinate at a pixel is a quarter of its clip position plus one. Lane 0
//! is the primitive (vertices 0, 2, 1: clockwise on the screen), with the
//! per-primitive values 0.5 and 2 plus the group's first thread
//! ([`GROUP_THREADS`] per group, so every group draws its own value). The
//! pixel shader writes the coordinate's x and y, then the two per-primitive
//! values.

use super::bitcode::{Attr, Term};
use super::container::SigElement;
use super::shader::Shader;
use super::{HostShader, MeshFallback, lower};
use crate::native::{HwStage, MeshIo, NativeProgram, StageIo, Usage, VsOutput};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// What the console adds to the program's group input per thread group.
pub const GROUP_THREADS: u32 = 64;

/// Inline constants: an integer 0..=64, -1, and the floats.
const fn int(n: u16) -> u16 {
    128 + n
}
const MINUS_ONE: u16 = 193;
const HALF: u16 = 240;
const ONE: u16 = 242;
const MINUS_ONE_F: u16 = 243;
const TWO: u16 = 244;
const FOUR: u16 = 246;
/// A 32-bit literal follows the instruction.
const LITERAL: u16 = 255;

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// VOP1 `op vdst, src0`.
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1`.
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9,
    ]
}

/// `exp target` of the VGPRs `v` (as many as given), `done` on the last one.
fn exp(target: u8, v: &[u8], done: bool) -> [u32; 2] {
    let enabled = (1u32 << v.len()) - 1;
    let mut sources = 0;
    for (k, &r) in v.iter().enumerate() {
        sources |= u32::from(r) << (8 * k);
    }
    [
        0x3E << 26 | u32::from(done) << 11 | u32::from(target) << 4 | enabled,
        sources,
    ]
}

const V_MOV_B32: u16 = 0x001;
const V_READFIRSTLANE_B32: u16 = 0x002;
const V_CVT_F32_U32: u16 = 0x006;
const V_ADD_F32: u8 = 0x03;
const V_MUL_F32: u8 = 0x08;
const V_LSHRREV_B32: u8 = 0x16;
const V_AND_B32: u8 = 0x1B;
const V_MBCNT_LO_U32_B32: u16 = 0x365;
const POS0: u8 = 12;
const PRIM: u8 = 20;
const PARAM0: u8 = 32;

fn program(extra: &[u32]) -> NativeProgram {
    let mut words = Vec::new();
    // v1: the lane's index (the lanes below it in a full mask).
    words.extend(vop3(V_MBCNT_LO_U32_B32, 1, MINUS_ONE, int(0)));
    words.push(vop2(V_AND_B32, 2, int(1), 1));
    words.push(vop2(V_LSHRREV_B32, 3, int(1), 1));
    // v7, v8: the texture coordinate.
    words.push(vop1(V_CVT_F32_U32, 7, vgpr(2)));
    words.push(vop1(V_CVT_F32_U32, 8, vgpr(3)));
    // v2, v3: the position.
    words.push(vop2(V_MUL_F32, 2, FOUR, 7));
    words.push(vop2(V_MUL_F32, 3, FOUR, 8));
    words.push(vop2(V_ADD_F32, 2, MINUS_ONE_F, 2));
    words.push(vop2(V_ADD_F32, 3, MINUS_ONE_F, 3));
    words.push(vop1(V_MOV_B32, 4, int(0)));
    words.push(vop1(V_MOV_B32, 6, ONE));
    // The caller's instructions: they may change v6 (1.0) and the SGPRs from
    // s16 up, and must leave the drawn triangle as it is.
    words.extend_from_slice(extra);
    words.extend(exp(POS0, &[2, 3, 4, 6], false));
    words.extend(exp(PARAM0, &[7, 8, 4, 6], false));
    // The per-primitive output.
    words.push(vop1(V_MOV_B32, 9, HALF));
    // s4: the group's first thread (v5 on entry, the same in every lane).
    words.push(vop1(V_READFIRSTLANE_B32, 4, vgpr(5)));
    words.push(vop1(V_CVT_F32_U32, 12, 4));
    words.push(vop2(V_ADD_F32, 10, TWO, 12));
    words.extend(exp(PARAM0 + 1, &[9, 10], false));
    // Vertices 0, 2 and 1, ten bits each.
    words.push(vop1(V_MOV_B32, 11, LITERAL));
    words.push(2 << 10 | 1 << 20);
    words.extend(exp(PRIM, &[11], true));
    words.push(S_ENDPGM);
    let output = |row, param| VsOutput {
        usage: Usage::Generic,
        row,
        param,
        fp16: false,
        pack2: false,
    };
    NativeProgram {
        stage: HwStage::Ms,
        wave64: true,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Ms(MeshIo {
            // Bit 7: per primitive, a row of the primitive signature.
            outputs: vec![output(0, 0), output(2, 0x80 | 1)],
            clip_cntl: 0,
            threads: 64,
            max_vertices: 3,
            max_primitives: 1,
            topology: 2,
            lds_bytes: 0,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn element(name: &str, index: u32, kind: u8, interp: u8, row: u32, cols: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind,
        comp: 9,
        interp,
        row: Some(row),
        col: 0,
        cols,
        usage: 0,
        half: false,
    }
}

/// The per-vertex and the per-primitive outputs.
fn signatures() -> (Vec<SigElement>, Vec<SigElement>) {
    (
        vec![
            element("TEXCOORD", 0, 0, 2, 0, 4),
            element("SV_Position", 0, 3, 4, 1, 4),
        ],
        vec![element("TEXCOORD", 1, 0, 1, 2, 2)],
    )
}

/// The program as a mesh shader, or its compute half with the buffer's layout.
fn lowered(native: &NativeProgram, mesh_buffer: bool) -> lower::Lowered {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let (outputs, prims) = signatures();
    lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs,
        prims,
        mesh_stride: GROUP_THREADS,
        mesh_buffer,
    })
    .expect("lower")
}

fn host_shader(stage: HwStage, l: lower::Lowered) -> HostShader {
    HostShader {
        stage,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        structured_srvs: l.structured_srvs,
    }
}

fn pixel_shader() -> HostShader {
    let mut s = Shader::new(true);
    let b = s.entry;
    let (outputs, prims) = signatures();
    s.inputs = outputs.into_iter().chain(prims).collect();
    let undef = s.m.undef(s.t.i32);
    // (input element, channel) per target channel.
    for (channel, (id, col)) in [(0, 0), (0, 1), (2, 0), (2, 1)].into_iter().enumerate() {
        s.inputs[id as usize].usage |= 1 << col;
        let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), undef];
        let f32t = s.t.f32;
        let x = s.op(b, "loadInput.f32", Attr::ReadNone, f32t, 4, &args);
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(channel as u8), x];
        let void = s.t.void;
        s.op(b, "storeOutput.f32", Attr::NoUnwind, void, 5, &args);
    }
    let mut target = element("SV_Target", 0, 16, 0, 0, 4);
    target.usage = 0xF;
    s.outputs = vec![target];
    s.m.terminate(b, Term::Ret);
    HostShader {
        stage: HwStage::Ps,
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        structured_srvs: Vec::new(),
    }
}

/// The sample program in the forms a host draws it with.
pub struct MeshSample {
    /// As a mesh shader.
    pub mesh: HostShader,
    /// As hosts without mesh shaders run it.
    pub fallback: MeshFallback,
    /// The pixel shader for either: four 32-bit float channels to target 0.
    pub pixel: HostShader,
}

/// Translates the sample program.
pub fn mesh() -> MeshSample {
    mesh_with(&[])
}

/// The sample program with `extra` instructions after it makes v6 (the
/// position's w, 1.0), before the exports. Drawn the same if they leave v6
/// at 1.0.
pub(super) fn mesh_with(extra: &[u32]) -> MeshSample {
    mesh_of(&program(extra))
}

/// The sample program's translation attempt, as the error it stops with.
#[cfg(all(test, windows))]
pub(super) fn mesh_error(extra: &[u32]) -> Option<crate::ShaderError> {
    let native = program(extra);
    let p = ir::build(&native.code, native.wave64, native.vgprs).ok()?;
    let st = structure::structure(&p).ok()?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).ok()?;
    let io = io::raise(&p, &native).ok()?;
    let (outputs, prims) = signatures();
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs,
        prims,
        mesh_stride: GROUP_THREADS,
        mesh_buffer: false,
    })
    .err()
}

fn mesh_of(native: &NativeProgram) -> MeshSample {
    let l = lowered(native, true);
    let buffer = l
        .mesh_buffer
        .clone()
        .expect("the compute half has the buffer's layout");
    MeshSample {
        mesh: host_shader(HwStage::Ms, lowered(native, false)),
        fallback: MeshFallback {
            vertex: buffer.vertex_shader().expect("the vertex half"),
            compute: host_shader(HwStage::Ms, l),
            buffer,
        },
        pixel: pixel_shader(),
    }
}

/// What thread group 0 of the sample draws at each pixel of a target of
/// this size, row by row.
pub fn mesh_texels(width: u32, height: u32) -> Vec<[f32; 4]> {
    group_texels(width, height, 0)
}

/// What thread group `group` of the sample draws at each pixel of a target
/// of this size, row by row. Every group covers the whole target, so a
/// dispatch of several leaves the last one's.
pub fn group_texels(width: u32, height: u32, group: u32) -> Vec<[f32; 4]> {
    let last = 2.0 + (group * GROUP_THREADS) as f32;
    let mut texels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let clip_x = 2.0 * (x as f32 + 0.5) / width as f32 - 1.0;
            let clip_y = 1.0 - 2.0 * (y as f32 + 0.5) / height as f32;
            texels.push([(clip_x + 1.0) / 4.0, (clip_y + 1.0) / 4.0, 0.5, last]);
        }
    }
    texels
}
