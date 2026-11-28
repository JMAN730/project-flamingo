//! Barycentrics on a host without them (`D3D12_FEATURE_DATA_D3D12_OPTIONS3
//! .BarycentricsSupported` false, as on Pascal GPUs). A console pixel shader
//! that computes the weights it interpolates with, or reads the barycentric
//! registers as data, is translated with `SV_Barycentrics` and SM 6.1
//! `GetAttributeAtVertex` (see the module notes in `mod.rs`); such a host
//! refuses the pipeline. The same program translated without them
//! ([`super::translate_without_barycentrics`]) reads instead:
//!
//! * for each input it reads at one vertex, three constant
//!   (nointerpolation) inputs `VICTORY_VERTEX<k>_<semantic>` (same semantic
//!   index, columns and type), the value at vertex k of the primitive;
//! * for each kind of barycentrics it reads, the input
//!   `VICTORY_BARYCENTRICS`, semantic index the DXIL interpolation mode
//!   (linear or noperspective, centre, centroid or sample), three channels
//!   with the weights of the three vertices, as `SV_Barycentrics` has them.
//!
//! Both go in rows of their own past the program's other inputs. The
//! geometry shader made here ([`barycentric_gs`]) sits between the vertex
//! shader and that pixel shader: it takes each triangle and emits it again,
//! each vertex carrying its own outputs as before (register for register,
//! as the pixel shader declares them), all three vertices' values in the
//! copies, and the weights (1, 0, 0), (0, 1, 0) and (0, 0, 1), which the
//! rasterizer interpolates into the barycentrics in the mode the pixel
//! shader declares.
//!
//! Vertex k is the geometry shader's input vertex k, which is the vertex
//! `GetAttributeAtVertex(k)` reads on a host with barycentrics: D3D orders
//! a primitive's vertices the same way for both, so the console's P0, P10
//! and P20 map as they do there (the shortcut on the provoking vertex in
//! `lower.rs` holds for both). Triangle lists and strips both arrive as
//! triangles: D3D already reorders the vertices of a strip's odd triangles
//! so that they turn the same way as the even ones, and the triangle goes
//! out in the order it came, so culling and the front face are unchanged.
//! Other topologies (points, lines, patches, adjacency, the console's
//! rectangle lists) get no such stage: the pipeline layer refuses them.
//!
//! shortcut: which vertex of a strip's odd triangle is the console's P0 is
//! taken to be D3D's choice (the GTX 1080 gives vertices i, i + 2, i + 1 of
//! triangle i); it is not measured on the console, and matters only for
//! programs whose weights are not symmetric in the vertices.

use super::bitcode::{Attr, Cast, Term, V};
use super::container::SigElement;
use super::shader::{Geometry, Shader};
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// The name prefix of a pixel input that holds one vertex's value of
/// another: `VICTORY_VERTEX<k>_<semantic>`.
pub(crate) const VERTEX_COPY: &str = "VICTORY_VERTEX";

/// The name of the pixel input that holds the barycentrics, whose semantic
/// index is its DXIL interpolation mode.
pub(crate) const BARYCENTRICS: &str = "VICTORY_BARYCENTRICS";

/// The name of the copy of the input `semantic` at vertex `vertex`.
pub(crate) fn vertex_copy_name(vertex: u8, semantic: &str) -> String {
    format!("{VERTEX_COPY}{vertex}_{semantic}")
}

/// The vertex and semantic a copy's name stands for.
fn copy_of(name: &str) -> Option<(u32, &str)> {
    let (k, semantic) = name.strip_prefix(VERTEX_COPY)?.split_once('_')?;
    let k: u32 = k.parse().ok()?;
    (k < 3 && !semantic.is_empty()).then_some((k, semantic))
}

/// The scalars a geometry shader may emit per invocation (vertices times
/// output components).
const MAX_SCALARS: u32 = 1024;

/// Registers of a stage's signature.
const MAX_ROWS: u32 = 32;

/// The geometry shader (a DXIL container) between the translated vertex
/// shader `vs` and the pixel shader `ps` translated without barycentrics
/// (both as [`super::HostShader::dxil`]). Its inputs are the vertex
/// shader's outputs; its outputs are the pixel shader's inputs, register
/// for register, plus the position, clip and cull distances and render
/// target and viewport indices the vertex shader writes and the pixel
/// shader does not read.
pub fn barycentric_gs(vs: &[u8], ps: &[u8]) -> Result<Vec<u8>, ShaderError> {
    let outputs = super::signature_part(&Container::parse(vs)?, FourCC::OSG1)?;
    let c = Container::parse(ps)?;
    let mut inputs = super::signature_part(&c, FourCC::ISG1)?;
    if let Some(psv) = c.part(FourCC::PSV0) {
        super::container::input_interpolation(psv.data, &mut inputs);
    }
    build(outputs, &inputs)
}

/// Where one output component of an emitted vertex comes from.
#[derive(Clone, Copy)]
enum Source {
    /// Column `col` of input `id` at the vertex being emitted.
    Own { id: usize, col: u8 },
    /// Column `col` of input `id` at vertex `vertex`, whichever is emitted.
    At { id: usize, col: u8, vertex: u32 },
    /// The weight of vertex `of`: 1 at that vertex, 0 at the others.
    Weight { of: u32 },
    /// The primitive's number.
    Primitive,
    /// Zero (what the rasterizer gives an index no stage writes).
    Zero,
}

fn unsupported(what: &'static str) -> ShaderError {
    ShaderError::Unsupported { offset: 0, what }
}

/// The vertex output with the semantic of `e` (a system value by kind,
/// another by name, ignoring case), and the column of it that holds column
/// `col` of `e`.
fn source_of<'a>(
    outputs: &'a [SigElement],
    semantic: &str,
    kind: u8,
    index: u32,
) -> Option<(usize, &'a SigElement)> {
    outputs.iter().enumerate().find(|(_, o)| {
        o.index == index
            && if kind == 0 {
                o.kind == 0 && o.name.eq_ignore_ascii_case(semantic)
            } else {
                o.kind == kind
            }
    })
}

/// The columns of `e` as columns of the vertex output `o`.
fn columns(e: &SigElement, o: &SigElement) -> Result<Vec<u8>, ShaderError> {
    (e.col..e.col + e.cols)
        .map(|c| {
            c.checked_sub(o.col)
                .filter(|&k| k < o.cols)
                .ok_or(unsupported(
                    "a pixel input the vertex shader does not write",
                ))
        })
        .collect()
}

/// [`barycentric_gs`] from the vertex shader's outputs and the pixel
/// shader's inputs (with their interpolation modes).
fn build(
    mut vs_outputs: Vec<SigElement>,
    ps_inputs: &[SigElement],
) -> Result<Vec<u8>, ShaderError> {
    if vs_outputs.iter().any(|e| e.row.is_none()) {
        return Err(unsupported(
            "a barycentric fallback over a vertex output without a register",
        ));
    }
    if !vs_outputs
        .iter()
        .any(|e| e.kind == 3 && e.comp == 9 && e.cols == 4)
    {
        return Err(unsupported(
            "a barycentric fallback without a four-component position",
        ));
    }
    for e in &mut vs_outputs {
        // As `rect.rs`: the modes `emit_program` declares a vertex shader's
        // outputs with; a geometry shader's inputs carry the same.
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
        e.usage = (1 << e.cols) - 1;
    }
    let mut outputs: Vec<SigElement> = Vec::new();
    let mut sources: Vec<Vec<Source>> = Vec::new();
    for e in ps_inputs {
        if e.kind == 28 {
            return Err(unsupported(
                "a pixel shader that reads the host's barycentrics",
            ));
        }
        // Values the rasterizer makes, and inputs without a register.
        if e.row.is_none() || matches!(e.kind, 12..=15) {
            continue;
        }
        let from = match e.kind {
            0 if e.name == BARYCENTRICS => {
                if e.comp != 9 {
                    return Err(unsupported("barycentrics that are not floats"));
                }
                (e.col..e.col + e.cols)
                    .map(|c| Source::Weight { of: u32::from(c) })
                    .collect()
            }
            10 => {
                if e.cols != 1 || e.comp == 9 {
                    return Err(unsupported("a primitive id that is no integer"));
                }
                vec![Source::Primitive]
            }
            _ => match copy_of(&e.name).filter(|_| e.kind == 0) {
                Some((vertex, semantic)) => {
                    let kind = match semantic.to_ascii_uppercase().as_str() {
                        "SV_POSITION" => 3,
                        "SV_CLIPDISTANCE" => 6,
                        "SV_CULLDISTANCE" => 7,
                        _ => 0,
                    };
                    let (id, o) = source_of(&vs_outputs, semantic, kind, e.index).ok_or(
                        unsupported("a pixel input the vertex shader does not write"),
                    )?;
                    columns(e, o)?
                        .into_iter()
                        .map(|col| Source::At { id, col, vertex })
                        .collect()
                }
                None => match source_of(&vs_outputs, e.semantic_name(), e.kind, e.index) {
                    Some((id, o)) => columns(e, o)?
                        .into_iter()
                        .map(|col| Source::Own { id, col })
                        .collect(),
                    // No stage writes them: the rasterizer's zero.
                    None if matches!(e.kind, 4 | 5) => vec![Source::Zero],
                    None => {
                        return Err(unsupported(
                            "a pixel input the vertex shader does not write",
                        ));
                    }
                },
            },
        };
        let interp = match e.kind {
            3 => 4,
            4 | 5 | 10 => 1,
            _ if copy_of(&e.name).is_some() => 1,
            _ if e.interp != 0 => e.interp,
            _ if e.comp == 9 => 2,
            _ => 1,
        };
        outputs.push(SigElement {
            interp,
            usage: (1 << e.cols) - 1,
            half: false,
            ..e.clone()
        });
        sources.push(from);
    }
    // What the vertex shader writes for the rasterizer and the output
    // merger that the pixel shader does not read: in rows of their own.
    let mut next = outputs
        .iter()
        .filter_map(|e| e.row)
        .max()
        .map_or(0, |r| r + 1);
    for (id, o) in vs_outputs.iter().enumerate() {
        if !matches!(o.kind, 3..=7) {
            continue;
        }
        if let Some(k) = outputs
            .iter()
            .position(|e| e.kind == o.kind && e.index == o.index)
        {
            // Clip and cull distances the pixel shader reads only some of
            // still all clip or cull: this stage is the last before the
            // rasterizer, so it passes every component the vertex shader
            // wrote (the pixel input reads its subset of the same row).
            // The widened output takes the vertex shader's whole component
            // range, start column included, so the pixel input's columns
            // must lie inside it and no other output may share those columns.
            if matches!(o.kind, 6 | 7) && outputs[k].cols < o.cols {
                let (start, end) = (o.col, o.col + o.cols);
                let read = (outputs[k].col, outputs[k].col + outputs[k].cols);
                let row = outputs[k].row;
                let overlaps = outputs.iter().enumerate().any(|(j, e)| {
                    j != k
                        && e.row.is_some()
                        && e.row == row
                        && e.col < end
                        && start < e.col + e.cols
                });
                if read.0 < start || read.1 > end || overlaps {
                    return Err(unsupported(
                        "a barycentric fallback whose clip or cull distances do not line up with the vertex shader's",
                    ));
                }
                outputs[k].col = o.col;
                outputs[k].cols = o.cols;
                outputs[k].usage = (1 << o.cols) - 1;
                sources[k] = (0..o.cols).map(|col| Source::Own { id, col }).collect();
            }
            continue;
        }
        outputs.push(SigElement {
            row: Some(next),
            ..o.clone()
        });
        sources.push((0..o.cols).map(|col| Source::Own { id, col }).collect());
        next += 1;
    }
    if outputs.iter().any(|e| e.row.is_some_and(|r| r >= MAX_ROWS)) {
        return Err(unsupported(
            "a barycentric fallback with more pixel inputs than registers",
        ));
    }
    let scalars: u32 = outputs.iter().map(|e| u32::from(e.cols)).sum();
    if scalars * 3 > MAX_SCALARS {
        return Err(unsupported(
            "a barycentric fallback with more output components than a geometry shader emits",
        ));
    }

    let mut s = Shader::new(false);
    s.geometry = Some(Geometry { max_vertices: 3 });
    s.inputs = vs_outputs;
    s.outputs = outputs.clone();
    let b = s.entry;
    // Each input column at each vertex, loaded once: [element][column][vertex].
    let mut loaded: Vec<Vec<[Option<V>; 3]>> = s
        .inputs
        .iter()
        .map(|e| vec![[None; 3]; usize::from(e.cols)])
        .collect();
    let mut load = |s: &mut Shader, id: usize, col: u8, vertex: u32| -> V {
        let slot = &mut loaded[id][usize::from(col)][vertex as usize];
        *slot.get_or_insert_with(|| {
            let (name, ty) = if s.inputs[id].comp == 9 {
                ("loadInput.f32", s.t.f32)
            } else {
                ("loadInput.i32", s.t.i32)
            };
            let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), s.m.i32(vertex)];
            s.op(b, name, Attr::ReadNone, ty, 4, &args)
        })
    };
    let primitive = sources
        .iter()
        .flatten()
        .any(|f| matches!(f, Source::Primitive))
        .then(|| s.op(b, "primitiveID.i32", Attr::ReadNone, s.t.i32, 108, &[]));
    let stream = s.m.i8(0);
    for vertex in 0..3 {
        for (id, from) in sources.iter().enumerate() {
            let float = s.outputs[id].comp == 9;
            for (col, &source) in from.iter().enumerate() {
                let (value, is_float) = match source {
                    Source::Own { id: i, col: c } => {
                        (load(&mut s, i, c, vertex), s.inputs[i].comp == 9)
                    }
                    Source::At {
                        id: i,
                        col: c,
                        vertex: k,
                    } => (load(&mut s, i, c, k), s.inputs[i].comp == 9),
                    Source::Weight { of } => {
                        let one = if of == vertex { 1f32 } else { 0f32 };
                        (s.m.f32(one.to_bits()), true)
                    }
                    Source::Primitive => (primitive.expect("loaded above"), false),
                    Source::Zero if float => (s.m.f32(0), true),
                    Source::Zero => (s.m.i32(0), false),
                };
                // The pixel shader's type: the bits as they are.
                let value = match (is_float, float) {
                    (true, false) => s.m.cast(b, Cast::Bitcast, value, s.t.i32),
                    (false, true) => s.m.cast(b, Cast::Bitcast, value, s.t.f32),
                    _ => value,
                };
                let name = if float {
                    "storeOutput.f32"
                } else {
                    "storeOutput.i32"
                };
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col as u8), value];
                s.op(b, name, Attr::NoUnwind, s.t.void, 5, &args);
            }
        }
        s.op(b, "emitStream", Attr::NoUnwind, s.t.void, 97, &[stream]);
    }
    s.op(b, "cutStream", Attr::NoUnwind, s.t.void, 98, &[stream]);
    s.m.terminate(b, Term::Ret);
    let gs = s.finish();
    // The registers must be the pixel shader's (the container may move an
    // element whose row mixes interpolation modes).
    let written = super::signature_part(&Container::parse(&gs)?, FourCC::OSG1)?;
    for e in &outputs {
        let same = written
            .iter()
            .any(|w| w.name == e.semantic_name() && w.index == e.index && w.row == e.row);
        if !same {
            return Err(unsupported(
                "a barycentric fallback whose outputs do not keep the pixel shader's registers",
            ));
        }
    }
    Ok(gs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn element(name: &str, index: u32, kind: u8, comp: u8, row: u32, cols: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index,
            kind,
            comp,
            interp: 0,
            row: Some(row),
            col: 0,
            cols,
            usage: 0,
            half: false,
        }
    }

    fn vs_outputs() -> Vec<SigElement> {
        vec![
            element("TEXCOORD", 0, 0, 9, 0, 3),
            element("SV_Position", 0, 3, 9, 1, 4),
            element("TEXCOORD", 1, 0, 5, 2, 1),
        ]
    }

    /// The pixel inputs a fallback translation declares: the original
    /// element (constant), its three copies, the barycentrics of two modes
    /// and an integer the vertex shader writes.
    fn ps_inputs() -> Vec<SigElement> {
        let constant = |e: SigElement| SigElement { interp: 1, ..e };
        vec![
            constant(element("TEXCOORD", 0, 0, 9, 0, 3)),
            constant(element("TEXCOORD", 1, 0, 5, 2, 1)),
            constant(element("VICTORY_VERTEX0_TEXCOORD", 0, 0, 9, 3, 3)),
            constant(element("VICTORY_VERTEX1_TEXCOORD", 0, 0, 9, 4, 3)),
            constant(element("VICTORY_VERTEX2_TEXCOORD", 0, 0, 9, 5, 3)),
            SigElement {
                interp: 2,
                ..element(BARYCENTRICS, 2, 0, 9, 6, 3)
            },
            SigElement {
                interp: 4,
                ..element(BARYCENTRICS, 4, 0, 9, 7, 3)
            },
        ]
    }

    #[test]
    fn copy_names_say_their_vertex_and_semantic() {
        assert_eq!(
            copy_of(&vertex_copy_name(2, "TEXCOORD")),
            Some((2, "TEXCOORD"))
        );
        assert_eq!(copy_of("VICTORY_VERTEX3_TEXCOORD"), None);
        assert_eq!(copy_of("VICTORY_VERTEX1_"), None);
        assert_eq!(copy_of("TEXCOORD"), None);
    }

    /// Triangles in, three vertices out; every pixel input at its own
    /// register, and the position (which the pixel shader does not read)
    /// in a row past them. The validator, where there is one, accepts it.
    #[test]
    fn the_barycentric_stage_is_a_geometry_shader_that_keeps_the_pixel_registers() {
        let gs = build(vs_outputs(), &ps_inputs()).unwrap();
        let c = Container::parse(&gs).unwrap();
        assert_eq!(c.part(FourCC::DXIL).unwrap().data[..4], [0x66, 0, 2, 0]);
        let psv = c.part(FourCC::PSV0).unwrap().data;
        assert_eq!(psv[4..17], [3, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 1]);
        assert_eq!(psv[28..34], [2, 0, 3, 0, 3, 8]);
        let outs = super::super::signature(c.part(FourCC::OSG1).unwrap().data).unwrap();
        for want in ps_inputs() {
            assert!(
                outs.iter()
                    .any(|o| o.name == want.name && o.index == want.index && o.row == want.row),
                "{} {} not at row {:?}",
                want.name,
                want.index,
                want.row
            );
        }
        let position = outs.iter().find(|o| o.kind == 3).unwrap();
        assert_eq!(position.row, Some(8));
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut gs.clone()).unwrap();
        }
    }

    /// Clip distances the pixel shader reads only two of still pass all
    /// four the vertex shader writes: the stage is the last before the
    /// rasterizer, which clips by every one of them.
    #[test]
    fn the_barycentric_stage_passes_every_clip_distance_the_vertex_shader_writes() {
        let mut vs = vs_outputs();
        vs.push(element("SV_ClipDistance", 0, 6, 9, 2, 4));
        let mut ps = ps_inputs();
        ps.push(element("SV_ClipDistance", 0, 6, 9, 8, 2));
        let gs = build(vs, &ps).unwrap();
        let c = Container::parse(&gs).unwrap();
        let outs = super::super::signature(c.part(FourCC::OSG1).unwrap().data).unwrap();
        let clip = outs.iter().find(|o| o.kind == 6).unwrap();
        assert_eq!(clip.cols, 4);
        assert_eq!(clip.row, Some(8));
    }

    /// A pixel input reading the last two of four clip distances (columns
    /// 2 and 3) widens to the vertex shader's columns 0 to 3, not 2 to 5.
    #[test]
    fn widened_clip_distances_start_at_the_vertex_shader_s_column() {
        let mut vs = vs_outputs();
        vs.push(element("SV_ClipDistance", 0, 6, 9, 2, 4));
        let mut ps = ps_inputs();
        ps.push(SigElement {
            col: 2,
            ..element("SV_ClipDistance", 0, 6, 9, 8, 2)
        });
        let gs = build(vs, &ps).unwrap();
        let c = Container::parse(&gs).unwrap();
        let outs = super::super::signature(c.part(FourCC::OSG1).unwrap().data).unwrap();
        let clip = outs.iter().find(|o| o.kind == 6).unwrap();
        assert_eq!((clip.col, clip.cols), (0, 4));
        assert_eq!(clip.row, Some(8));
    }

    /// Clip distances the pixel shader reads outside the vertex shader's
    /// columns cannot widen to them: refused, not given a wrong extent.
    #[test]
    fn clip_distances_that_do_not_line_up_are_refused() {
        let mut vs = vs_outputs();
        vs.push(SigElement {
            col: 1,
            ..element("SV_ClipDistance", 0, 6, 9, 2, 3)
        });
        let mut ps = ps_inputs();
        ps.push(element("SV_ClipDistance", 0, 6, 9, 8, 2));
        assert!(build(vs, &ps).is_err());
    }

    /// Refused with a reason: a copy of something the vertex shader does
    /// not write, no position, the host's own barycentrics.
    #[test]
    fn barycentric_stages_that_cannot_be_made_are_refused() {
        let mut missing = ps_inputs();
        missing.push(element("VICTORY_VERTEX0_COLOR", 0, 0, 9, 8, 4));
        assert!(build(vs_outputs(), &missing).is_err());
        let mut without_position = vs_outputs();
        without_position.remove(1);
        assert!(build(without_position, &ps_inputs()).is_err());
        let mut host = ps_inputs();
        host.push(SigElement {
            row: None,
            ..element("SV_Barycentrics", 0, 28, 9, 0, 3)
        });
        assert!(build(vs_outputs(), &host).is_err());
        let mut wide = ps_inputs();
        wide.push(element(BARYCENTRICS, 6, 0, 9, 40, 3));
        assert!(build(vs_outputs(), &wide).is_err());
    }
}
