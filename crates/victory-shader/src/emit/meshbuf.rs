//! Mesh programs on hosts without mesh shaders.
//!
//! A console mesh program is one thread group whose lane n exports vertex n
//! and primitive n. A host without mesh shaders runs it in two halves:
//!
//! * the program itself as a **compute shader** (the same lowering, the same
//!   thread group, group-shared memory and barriers included) that stores
//!   what it would export into a buffer, the root UAV `u0` in
//!   [`MESH_BUFFER_SPACE`](super::MESH_BUFFER_SPACE);
//! * a generated **vertex shader** ([`MeshBuffer::vertex_shader`]) that reads
//!   that buffer back, the root SRV `t0` in the same space, and feeds the
//!   title's pixel shader what the mesh shader would have.
//!
//! The buffer holds one record per thread group, group `g`'s at dword
//! `g * group_dwords()`:
//!
//! ```text
//! max_vertices   x vertex:    one dword per output channel, elements in
//!                             signature order
//! max_primitives x primitive: the three vertex indices, 10 bits each from
//!                             bit 0 (all 0: culled, no area), then one dword
//!                             per per-primitive output channel
//! ```
//!
//! The host dispatches the compute half with the title's group count in x and
//! then draws `draw_vertices()` vertices as a triangle list with one instance
//! per group: vertex `3p + c` of instance `g` is corner `c` of primitive `p`
//! of group `g`. Corners are not shared between primitives, so per-primitive
//! outputs are plain vertex outputs with the same value at all three.
//!
//! A dispatch of more groups than the host's buffer holds runs in parts, a
//! compute pass and a draw each. The records of a part start at dword 0, and
//! its groups count from 0 on the host, so the compute half adds the part's
//! first group (the constant buffer `b0` in the same space) to the group
//! number the program reads.

use super::HostShader;
use super::bitcode::{Attr, Bin, Cast, Term, V};
use super::container::SigElement;
use super::shader::{Mesh, Shader};
use crate::ShaderError;

/// The layout of the buffer a mesh program's compute half fills.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshBuffer {
    /// Vertices and primitives each thread group has room for.
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// Dwords of one vertex.
    pub vertex_dwords: u32,
    /// Dwords of one primitive, its index dword included.
    pub primitive_dwords: u32,
    outputs: Vec<SigElement>,
    prims: Vec<SigElement>,
}

/// The dword of element `id` within its vertex or primitive data.
fn offset(elements: &[SigElement], id: usize) -> u32 {
    elements[..id].iter().map(|e| u32::from(e.cols)).sum()
}

impl MeshBuffer {
    pub(crate) fn new(mesh: Mesh, outputs: &[SigElement], prims: &[SigElement]) -> Self {
        MeshBuffer {
            max_vertices: mesh.max_vertices,
            max_primitives: mesh.max_primitives,
            vertex_dwords: offset(outputs, outputs.len()),
            primitive_dwords: 1 + offset(prims, prims.len()),
            outputs: outputs.to_vec(),
            prims: prims.to_vec(),
        }
    }

    /// Dwords of one thread group's record.
    pub fn group_dwords(&self) -> u32 {
        self.max_vertices * self.vertex_dwords + self.max_primitives * self.primitive_dwords
    }

    /// Vertices the host draws per instance (thread group).
    pub fn draw_vertices(&self) -> u32 {
        3 * self.max_primitives
    }

    /// The dword, within a group's record, where its primitives start.
    pub(crate) fn primitives_at(&self) -> u32 {
        self.max_vertices * self.vertex_dwords
    }

    /// The dword of channel `col` of per-vertex output `id` within a vertex.
    pub(crate) fn vertex_dword(&self, id: usize, col: u8) -> u32 {
        offset(&self.outputs, id) + u32::from(col)
    }

    /// The dword of channel `col` of per-primitive output `id` within a
    /// primitive (past its index dword).
    pub(crate) fn primitive_dword(&self, id: usize, col: u8) -> u32 {
        1 + offset(&self.prims, id) + u32::from(col)
    }

    /// The vertex shader that reads the buffer back: `SV_VertexID` and
    /// `SV_InstanceID` in, the mesh program's per-vertex and per-primitive
    /// outputs out, under their own semantics and registers.
    pub fn vertex_shader(&self) -> Result<HostShader, ShaderError> {
        let unsupported = |what| ShaderError::Unsupported { offset: 0, what };
        // One register space on the way to the pixel shader: a per-primitive
        // output cannot sit where a per-vertex one does.
        let clash = self.prims.iter().any(|p| {
            self.outputs.iter().any(|o| {
                p.row.is_some()
                    && p.row == o.row
                    && p.col < o.col + o.cols
                    && o.col < p.col + p.cols
            })
        });
        if clash {
            return Err(unsupported(
                "a per-primitive output in a per-vertex output's register",
            ));
        }
        if self
            .outputs
            .iter()
            .chain(&self.prims)
            .any(|e| e.row.is_none())
        {
            return Err(unsupported("a mesh output without a register"));
        }

        let mut s = Shader::new(false);
        let b = s.entry;
        let id_input = |name: &str, kind, row| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 5,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        s.inputs = vec![
            id_input("SV_VertexID", 1, 0),
            id_input("SV_InstanceID", 2, 1),
        ];
        let undef = s.m.undef(s.t.i32);
        let input = |s: &mut Shader, id: u32| {
            let args = [s.m.i32(id), s.m.i32(0), s.m.i8(0), undef];
            let i32t = s.t.i32;
            s.op(b, "loadInput.i32", Attr::ReadNone, i32t, 4, &args)
        };
        let (vertex, group) = (input(&mut s, 0), input(&mut s, 1));

        let k = |s: &mut Shader, n: u32| s.m.i32(n);
        let three = k(&mut s, 3);
        let primitive = s.m.bin(b, Bin::UDiv, vertex, three);
        let corner = s.m.bin(b, Bin::URem, vertex, three);
        let n = k(&mut s, self.group_dwords());
        let record = s.m.bin(b, Bin::Mul, group, n);
        // The primitive's dword in the buffer.
        let n = k(&mut s, self.primitive_dwords);
        let p = s.m.bin(b, Bin::Mul, primitive, n);
        let n = k(&mut s, self.primitives_at());
        let p = s.m.bin(b, Bin::Add, p, n);
        let p = s.m.bin(b, Bin::Add, p, record);

        let buffer = s.mesh_buffer_handle(false);
        // `count` consecutive dwords from `dword` on, as the load returns them.
        let load = |s: &mut Shader, dword: V, count: u8| {
            let two = s.m.i32(2);
            let byte = s.m.bin(b, Bin::Shl, dword, two);
            let args = [buffer, byte, undef, s.m.i8((1 << count) - 1), s.m.i32(4)];
            let ret = s.t.res_ret;
            s.op(b, "rawBufferLoad.f32", Attr::ReadOnly, ret, 139, &args)
        };
        let indices = load(&mut s, p, 1);
        let indices = s.m.extract(b, indices, 0);
        let i32t = s.t.i32;
        let indices = s.m.cast(b, Cast::Bitcast, indices, i32t);
        let ten = k(&mut s, 10);
        let shift = s.m.bin(b, Bin::Mul, corner, ten);
        let index = s.m.bin(b, Bin::LShr, indices, shift);
        let mask = k(&mut s, 0x3FF);
        let index = s.m.bin(b, Bin::And, index, mask);
        // The corner's vertex.
        let n = k(&mut s, self.vertex_dwords);
        let v = s.m.bin(b, Bin::Mul, index, n);
        let v = s.m.bin(b, Bin::Add, v, record);

        let mut id = 0u32;
        let sources = [
            (&self.outputs, v, 0),
            // Past the primitive's index dword.
            (&self.prims, p, 1),
        ];
        for (elements, base, skip) in sources {
            let mut at = skip;
            for e in elements {
                let n = k(&mut s, at);
                let dword = s.m.bin(b, Bin::Add, base, n);
                let data = load(&mut s, dword, e.cols);
                for col in 0..e.cols {
                    let x = s.m.extract(b, data, u32::from(col));
                    let (suffix, x) = if e.comp == 9 {
                        (".f32", x)
                    } else {
                        (".i32", s.m.cast(b, Cast::Bitcast, x, i32t))
                    };
                    let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), x];
                    let void = s.t.void;
                    s.op(
                        b,
                        &format!("storeOutput{suffix}"),
                        Attr::NoUnwind,
                        void,
                        5,
                        &args,
                    );
                }
                at += u32::from(e.cols);
                id += 1;
            }
        }
        s.outputs = self
            .outputs
            .iter()
            .chain(&self.prims)
            .cloned()
            .map(|mut e| {
                e.usage = (1 << e.cols) - 1;
                e
            })
            .collect();
        s.m.terminate(b, Term::Ret);
        Ok(HostShader {
            stage: crate::native::HwStage::Vs,
            dxil: s.finish(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            structured_srvs: Vec::new(),
        })
    }
}
