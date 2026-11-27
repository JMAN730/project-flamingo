//! End-to-end tests of waterfall loops over per-lane descriptors: the
//! synthetic GFX10 programs of `resource::tests` lowered to DXIL, validated
//! by `dxil.dll` when present, and run on the host GPU and WARP, each lane
//! reading the resource its own descriptor names.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Lowers a wave32 compute program of 32 threads with the table of views at
/// s0 (textures, or with `buffers` structured buffers of dwords, at entries
/// 0 and 1, a raw UAV at entry 2) and the table of samplers at s1.
fn build(words: &[u32], buffers: bool) -> HostShader {
    let (shader, lowered) = build_any(words, buffers);
    // One access with each lane's own heap index, no wave operation.
    assert_eq!(lowered.nonuniform_handles, 1);
    assert!(!lowered.wave_ops);
    shader
}

/// [`build`] without its checks: the shader (validated by `dxil.dll` when
/// present) and what lowering reported.
fn build_any(words: &[u32], buffers: bool) -> (HostShader, lower::Lowered) {
    build_with_view(words, buffers.then_some(4))
}

fn build_with_view(words: &[u32], buffer_stride: Option<u32>) -> (HostShader, lower::Lowered) {
    let native = program(words, buffer_stride);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
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
        mesh_buffer: false,
    })
    .expect("lower");
    let static_samplers = lowered
        .samplers
        .iter()
        .enumerate()
        .map(|(n, &w)| {
            let mut s = super::StaticSampler::from_words(w);
            if lowered.cmp_samplers >> n & 1 != 0 {
                s.filter |= 0x80;
            }
            s
        })
        .collect();
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil.clone(),
        user_data: lowered.user_data.clone(),
        samplers: lowered.samplers.clone(),
        static_samplers,
        root_cbvs: lowered.root_cbvs.clone(),
        root_srvs: lowered.root_srvs.clone(),
        root_uavs: lowered.root_uavs.clone(),
        structured_srvs: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    (shader, lowered)
}

/// The wave32 compute program of 32 threads `build` lowers.
fn program(words: &[u32], buffer_stride: Option<u32>) -> NativeProgram {
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, stride, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data,
        slot: user_data,
    };
    let views = if let Some(stride) = buffer_stride {
        range(ResourceType::Srv, 0, 4, stride, 0)
    } else {
        range(ResourceType::Srv, 0, 8, 0, 0)
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: vec![
            views,
            range(ResourceType::Uav, 16, 4, 0, 0),
            range(ResourceType::Sampler, 0, 4, 0, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// Runs `shader` with `views` at heap indices 0.. (the UAV last) on the
/// hardware GPU and on WARP (each skipped without a device) and returns,
/// per one that ran, the UAV's 32 dwords. Empty when no SM 6.6 device is
/// available: callers return early (`docs/code-standards.md`).
fn run(shader: &HostShader, views: &[View]) -> Vec<(&'static str, Vec<u32>)> {
    run_with_table(shader, views, None)
}

/// [`run`], the host's buffer view table (see `UserData`) at heap index
/// `table` when the shader reads one.
fn run_with_table(
    shader: &HostShader,
    views: &[View],
    table: Option<u32>,
) -> Vec<(&'static str, Vec<u32>)> {
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let mut out = Vec::new();
    // One GPU test at a time, with the VALU tests: in parallel on the
    // hardware adapter and WARP the test process sometimes died of an
    // access violation (7 of 24 runs of these four tests).
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let force_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    for &(label, warp) in super::valu_tests::backends(force_warp) {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        // Both tables start at heap index 0.
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match u.root {
                RootType::Table => 0,
                RootType::ResourceHeap => table.expect("a buffer view table"),
                _ => panic!("{u:?}"),
            };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views,
                    samplers: &samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0].clone()));
    }
    out
}

/// False when no SM 6.6 hardware GPU or WARP ran; the caller returns early.
fn device_ran<T>(results: &[T]) -> bool {
    if results.is_empty() {
        eprintln!("neither hardware nor WARP SM 6.6 device available; skipped");
        false
    } else {
        true
    }
}

/// The red channel each lane sampled from two 1x1 textures, red 1.0 at
/// heap index 0 and 2.0 at 1.
fn sampled(shader: &HostShader) -> Vec<(&'static str, Vec<f32>)> {
    let first = [[1.0f32, 0.0, 0.0, 1.0]];
    let second = [[2.0f32, 0.0, 0.0, 1.0]];
    let views = [
        View::Texture(1, 1, &first),
        View::Texture(1, 1, &second),
        View::RawUav(32),
    ];
    run(shader, &views)
        .into_iter()
        .map(|(label, words)| (label, words.into_iter().map(f32::from_bits).collect()))
        .collect()
}

#[test]
fn lanes_sample_the_texture_their_if_else_arm_chose() {
    use crate::resource::tests::WATERFALL_JOINED_WAVE32;
    let shader = build(WATERFALL_JOINED_WAVE32, false);
    let results = sampled(&shader);
    if !device_ran(&results) {
        return;
    }
    for (label, red) in results {
        // Lanes below 16 took the first texture, the others the second.
        let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_texture_their_arm_loaded_or_picked() {
    use crate::resource::tests::WATERFALL_JOINED_MIXED_WAVE32;
    let shader = build(WATERFALL_JOINED_MIXED_WAVE32, false);
    let results = sampled(&shader);
    if !device_ran(&results) {
        return;
    }
    for (label, red) in results {
        // Lanes below 16 loaded entry (lane & 1); of the others, lanes
        // below 24 picked the first, the rest the second.
        let want: Vec<f32> = (0..32)
            .map(|l| match l {
                0..16 => [1.0, 2.0][l & 1],
                16..24 => 1.0,
                _ => 2.0,
            })
            .collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_holding_data_for_a_descriptor_sample_heap_index_zero() {
    use crate::resource::tests::waterfall_joined_data_wave32;
    // Lanes below 16 hold the second entry's descriptor, the others what the
    // program computed in its registers (a lane's data, no descriptor).
    let shader = build(&waterfall_joined_data_wave32(false), false);
    let results = sampled(&shader);
    if !device_ran(&results) {
        return;
    }
    for (label, red) in results {
        // shortcut: the lanes that hold data read heap index 0 (the first
        // texture); on the console they read whatever those words name.
        let want: Vec<f32> = (0..32).map(|l| if l < 16 { 2.0 } else { 1.0 }).collect();
        assert_eq!(red, want, "{label}");
    }
}

/// A uniform if/else (on the sampler table's address in s1, which is 0 in
/// the tests) joining the two samplers a 2x1 texture is sampled at u = 1.25
/// with. `first` takes the first arm: clamp (word 0 is 2) as an immediate
/// sampler or, with `heap`, the heap sampler at index 0 (clamp, as `run`
/// makes it). The other arm is the immediate wrap sampler (word 0 is 0).
fn joined_samplers(first: bool, heap: bool) -> Vec<u32> {
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_02FF,
        0x3FA0_0000,                                   // v_mov_b32 v2, 1.25
        0x7E06_02F0,                                   // v_mov_b32 v3, 0.5
        0x3420_0082,                                   // v_lshlrev_b32 v16, 2, v0
        if first { 0xBF06_8001 } else { 0xBF06_8101 }, // s_cmp_eq_u32 s1, 0 or 1
    ];
    if heap {
        words.extend([
            0xBF84_0003, // s_cbranch_scc0 -> else
            0xF408_0203,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0 (the heap sampler)
            0xBF82_0004, // s_branch -> join
            0xBE88_0380, // s_mov_b32 s8, 0
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
        ]);
    } else {
        words.extend([
            0xBF84_0002, // s_cbranch_scc0 -> else
            0xB008_0002, // s_movk_i32 s8, 2 (clamp)
            0xBF82_0001, // s_branch -> join
            0xBE88_0380, // s_mov_b32 s8, 0 (wrap)
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
        ]);
    }
    words.extend([
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    words
}

#[test]
fn a_sampler_chosen_by_control_flow_is_sampled_with_each_choice() {
    for heap in [false, true] {
        for first in [true, false] {
            let words = joined_samplers(first, heap);
            // Not what every other program is lowered as: a join of
            // immediate samplers is refused in the first tries.
            let native = program(&words, None);
            let p = ir::build(&native.code, false, native.vgprs).unwrap();
            let st = structure::structure(&p).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let parts = lower::Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: Vec::new(),
                outputs: Vec::new(),
                prims: Vec::new(),
                mesh_stride: 0,
                mesh_buffer: false,
            };
            let refusal = lower::lower_with_wave32(&parts, true, None, None, false);
            let refusal = refusal.err().expect("refused outside the last chance");
            assert!(
                refusal.to_string().contains("chosen by control flow"),
                "{refusal}"
            );
            // Lowered (and validated by `dxil.dll` when present) with the
            // last chance. The immediate samplers: wrap and, but with the
            // heap's, clamp (the heap one is a descriptor, not a binding).
            let (shader, lowered) = build_any(&words, false);
            assert_eq!(
                lowered.samplers.len(),
                if heap { 1 } else { 2 },
                "{heap} {first}"
            );
            // At u = 1.25 of two texels wrap reads the first, clamp the
            // second. The UAV is the table's third entry.
            let texels = [[1.0f32, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]];
            let spare = [[0.0f32; 4]];
            let views = [
                View::Texture(2, 1, &texels),
                View::Texture(1, 1, &spare),
                View::RawUav(32),
            ];
            let results = run(&shader, &views);
            if !device_ran(&results) {
                return;
            }
            for (label, got) in results {
                let red: Vec<f32> = got.into_iter().map(f32::from_bits).collect();
                // The first arm's sampler clamps, the other wraps.
                let want = if first { 2.0 } else { 1.0 };
                assert_eq!(red, vec![want; 32], "{label} heap {heap} first {first}");
            }
        }
    }
}

/// Cube sampling of lane `l`'s face `l & 3` at (1.25, 1.75): `d` with the
/// gradients 0 or, for `steep`, 1.0 (`image_sample_d`, face coordinates'
/// gradients first), else `image_sample_lz`. The red channel of the cube's
/// texels is their face number.
fn cube_sample(d: bool, steep: bool) -> Vec<u32> {
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x3612_0083, // v_and_b32 v9, 3, v0
        0x7E10_0D09, // v_cvt_f32_u32 v8, v9
        0x7E0C_02FF,
        0x3FA0_0000, // v_mov_b32 v6, 1.25
        0x7E0E_02FF,
        0x3FE0_0000, // v_mov_b32 v7, 1.75
    ];
    if d {
        let gradient = if steep { 0x7E00_02F2 } else { 0x7E00_0280 }; // v_mov_b32 v0, 1.0 or 0
        for v in 2..6u32 {
            // v_mov_b32 v<v>, 1.0 or 0
            words.push(gradient | v << 17);
        }
    }
    words.extend([
        0xBF8C_C07F,                               // s_waitcnt lgkmcnt(0)
        if d { 0xF088_0F18 } else { 0xF09C_0F18 }, // image_sample_d or _lz, dim:CUBE
        if d { 0x0043_0C02 } else { 0x0043_0C06 }, // address from v2 (v6 for lz)
        0xBF8C_3F70,                               // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    words
}

#[test]
fn a_cube_sample_with_gradients_takes_the_face_the_coordinates_name() {
    // 4x4 texels per face, 6 faces: the red of a texel is its face.
    let texels: Vec<[f32; 4]> = (0..96).map(|i| [(i / 16) as f32, 0.0, 0.0, 1.0]).collect();
    // Faces 0-3 by lane: the same with or without gradients, steep or not
    // (the view has one level).
    let want: Vec<f32> = (0..32).map(|l| (l & 3) as f32).collect();
    for (d, steep) in [(false, false), (true, false), (true, true)] {
        let (shader, _) = build_any(&cube_sample(d, steep), false);
        // The UAV is the table's third entry.
        let spare = [[0.0f32; 4]];
        let views = [
            View::Cube(4, &texels),
            View::Texture(1, 1, &spare),
            View::RawUav(32),
        ];
        let results = run(&shader, &views);
        if !device_ran(&results) {
            return;
        }
        for (label, got) in results {
            let red: Vec<f32> = got.into_iter().map(f32::from_bits).collect();
            assert_eq!(red, want, "{label}, gradients {d} steep {steep}");
        }
    }
}

#[test]
fn lanes_load_from_the_buffer_their_nested_arms_chose() {
    use crate::resource::tests::WATERFALL_JOINED_BUFFER_WAVE32;
    let shader = build(WATERFALL_JOINED_BUFFER_WAVE32, true);
    let first: Vec<u32> = (0..32).map(|i| 100 + i).collect();
    let second: Vec<u32> = (0..32).map(|i| 200 + i).collect();
    let views = [
        View::Buffer {
            data: &first,
            stride: Stride::Structured(4),
        },
        View::Buffer {
            data: &second,
            stride: Stride::Structured(4),
        },
        View::RawUav(32),
    ];
    let results = run(&shader, &views);
    if !device_ran(&results) {
        return;
    }
    for (label, got) in results {
        // Dword `lane` of the first buffer below 16, of the second from 16
        // to 23; lanes from 24 on load and store nothing.
        let want: Vec<u32> = (0..32)
            .map(|l| match l {
                0..16 => 100 + l,
                16..24 => 200 + l,
                _ => 0,
            })
            .collect();
        assert_eq!(got, want, "{label}");
    }
}

/// `WATERFALL_INDEXED_WAVE32` (each lane's entry at a byte offset only the
/// loop's first lane names) loading element `lane` of a buffer instead of
/// sampling: a view no fixed table entry names.
fn indexed_buffer_waterfall() -> Vec<u32> {
    use crate::resource::tests::WATERFALL_INDEXED_WAVE32;
    let mut words = WATERFALL_INDEXED_WAVE32.to_vec();
    let at = words.iter().position(|&w| w == 0xF40C_0302).unwrap();
    // s_load_dwordx4 s[12:15], s[4:5], s20
    words[at] = 0xF408_0302;
    let at = words.iter().position(|&w| w == 0xF09C_0F08).unwrap();
    // buffer_load_dword v12, v0, s[12:15], 0 idxen
    words[at] = 0xE030_2000;
    words[at + 1] = 0x8003_0C00;
    words
}

#[test]
fn lanes_load_with_the_stride_and_view_the_host_table_lists() {
    // No stride in the program's metadata (0): only the views know it.
    let (shader, _) = build_with_view(&indexed_buffer_waterfall(), Some(0));
    assert!(
        shader
            .user_data
            .iter()
            .any(|u| u.root == RootType::ResourceHeap && !u.indexed_buffer),
        "{:?}",
        shader.user_data
    );
    // The table at heap index 3 lists entry 0 as 8-byte elements from byte
    // 0 of the raw view at 4, entry 1 as 12-byte elements from byte 16 of
    // the one at 5 (20 elements: lanes past them read zeros).
    let mut table = vec![0u32; 4 * 8];
    table[..4].copy_from_slice(&[4, 8, 0, 32]);
    table[4..8].copy_from_slice(&[5, 12, 16, 20]);
    let first: Vec<u32> = (0..64)
        .map(|i| if i % 2 == 0 { 100 + i / 2 } else { 0xDEAD })
        .collect();
    let mut second = vec![0xBEEF; 4];
    for i in 0..32 {
        second.extend([200 + i, 0xDEAD, 0xDEAD]);
    }
    let unread = [0u32; 4];
    let views = [
        View::Buffer {
            data: &unread,
            stride: Stride::Raw,
        },
        View::Buffer {
            data: &unread,
            stride: Stride::Raw,
        },
        View::RawUav(32),
        View::Buffer {
            data: &table,
            stride: Stride::Raw,
        },
        View::Buffer {
            data: &first,
            stride: Stride::Raw,
        },
        View::Buffer {
            data: &second,
            stride: Stride::Raw,
        },
    ];
    let results = run_with_table(&shader, &views, Some(3));
    if !device_ran(&results) {
        return;
    }
    for (label, got) in results {
        // Element `lane` of the first view in even lanes, of the second in
        // odd ones below 20 (its element count), zeros past it.
        let want: Vec<u32> = (0..32)
            .map(|l| match (l % 2, l) {
                (0, _) => 100 + l,
                (_, 0..20) => 200 + l,
                _ => 0,
            })
            .collect();
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn a_buffer_of_known_stride_reads_no_host_table() {
    let (shader, _) = build_with_view(&indexed_buffer_waterfall(), Some(4));
    assert!(
        shader
            .user_data
            .iter()
            .all(|u| u.root != RootType::ResourceHeap),
        "{:?}",
        shader.user_data
    );
}

#[test]
fn lanes_sample_the_entry_their_offset_names_in_the_console_waterfall() {
    use crate::resource::tests::WATERFALL_INDEXED_WAVE32;
    let shader = build(WATERFALL_INDEXED_WAVE32, false);
    let results = sampled(&shader);
    if !device_ran(&results) {
        return;
    }
    for (label, red) in results {
        // Even lanes hold entry 0's offset, odd lanes entry 1's.
        let want: Vec<f32> = (0..32).map(|l| [1.0, 2.0][l & 1]).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_entry_their_index_names_when_scaled_in_the_loop() {
    use crate::resource::tests::{
        WATERFALL_ADDED_BASE_INDEX_WAVE32, WATERFALL_SCALED_INDEX_WAVE32,
    };
    let want: Vec<f32> = (0..32).map(|l| [1.0, 2.0][l & 1]).collect();
    for words in [
        WATERFALL_SCALED_INDEX_WAVE32,
        WATERFALL_ADDED_BASE_INDEX_WAVE32,
    ] {
        let shader = build(words, false);
        let results = sampled(&shader);
        if !device_ran(&results) {
            return;
        }
        for (label, red) in results {
            assert_eq!(red, want, "{label}");
        }
    }
}

#[test]
fn lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx() {
    use crate::resource::tests::cmpx_waterfall;
    // Lanes below 16 hold the first descriptor, the others the second.
    let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
    for folded in [false, true] {
        let shader = build(&cmpx_waterfall(false, folded), false);
        let results = sampled(&shader);
        if !device_ran(&results) {
            return;
        }
        for (label, red) in results {
            assert_eq!(red, want, "{label}, folded {folded}");
        }
    }
}

#[test]
fn lanes_sample_their_arm_after_a_redundant_masked_write() {
    use crate::resource::tests::WATERFALL_JOINED_WAVE32;
    let mut words = WATERFALL_JOINED_WAVE32.to_vec();
    let at = words.iter().position(|&w| w == 0x887E_147E).unwrap() + 1;
    words.splice(
        at..at,
        [
            0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0 (lane < 16)
            0xBE92_3C6A, // s_and_saveexec_b32 s18, vcc_lo
            0x7E08_0218, // v_mov_b32 v4, s24 (same word the then arm wrote)
            0xBEFE_0312, // s_mov_b32 exec_lo, s18
        ],
    );
    let shader = build(&words, false);
    let results = sampled(&shader);
    if !device_ran(&results) {
        return;
    }
    let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
    for (label, red) in results {
        assert_eq!(red, want, "{label}");
    }
}

/// `image_gather4_l` at (0.375, 0.375) of a 4x4 texture whose red is
/// 10 * y + x, level 2.0 (the view has one level: clamped to level 0). Lanes
/// 0-7 store the four results at dwords 4 * lane...
const GATHER4_L_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF40C_0302,
    0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
    0xF408_0203,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    0x3620_0087, // v_and_b32 v16, 7, v0
    0x3420_2084, // v_lshlrev_b32 v16, 4, v16
    0x7E04_02FF,
    0x3EC0_0000, // v_mov_b32 v2, 0.375
    0x7E06_02FF,
    0x3EC0_0000, // v_mov_b32 v3, 0.375
    0x7E08_02FF,
    0x4000_0000, // v_mov_b32 v4, 2.0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF110_0108,
    0x0043_0C02, // image_gather4_l v[12:15], v[2:4], s[12:19], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_1000,
    0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
    0xE070_1004,
    0x800A_0D10, // buffer_store_dword v13, v16, s[40:43], 0 offen offset:4
    0xE070_1008,
    0x800A_0E10, // buffer_store_dword v14, v16, s[40:43], 0 offen offset:8
    0xE070_100C,
    0x800A_0F10, // buffer_store_dword v15, v16, s[40:43], 0 offen offset:12
    0xBF81_0000, // s_endpgm
];

#[test]
fn gather4_with_a_level_returns_the_four_texels_of_that_level_in_gather_order() {
    let (shader, _) = build_any(GATHER4_L_WAVE32, false);
    let texels: Vec<[f32; 4]> = (0..16)
        .map(|i| [(10 * (i / 4) + i % 4) as f32, 0.0, 0.0, 1.0])
        .collect();
    let views = [
        View::Texture(4, 4, &texels),
        View::Texture(4, 4, &texels),
        View::RawUav(32),
    ];
    let results = run(&shader, &views);
    if !device_ran(&results) {
        return;
    }
    for (label, words) in results {
        let got: Vec<f32> = words.into_iter().map(f32::from_bits).collect();
        // u * 4 - 0.5 = 1: texels (1, 2), (2, 2), (2, 1), (1, 1).
        let want: Vec<f32> = (0..32).map(|i| [21.0, 22.0, 12.0, 11.0][i % 4]).collect();
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn gather4_with_a_level_far_outside_the_view_reads_the_nearest_edge() {
    let texels: Vec<[f32; 4]> = (0..16)
        .map(|i| [(10 * (i / 4) + i % 4) as f32, 0.0, 0.0, 1.0])
        .collect();
    // (u, v) and the texel (10 * y + x) all four of the footprint clamp to.
    // u * 4 above i32 (or infinite) must give the last column, not wrap
    // to the first when the footprint's second column is added; far below,
    // the first; NaN, the first.
    let cases = [
        (1e9, -1e9, 3.0),
        (-1e9, 1e9, 30.0),
        (f32::INFINITY, f32::NEG_INFINITY, 3.0),
        (f32::NAN, f32::INFINITY, 30.0),
    ];
    for (u, v, texel) in cases {
        let mut words = GATHER4_L_WAVE32.to_vec();
        // The literals of `v_mov_b32 v2, u` and `v_mov_b32 v3, v`.
        assert_eq!((words[12], words[14]), (0x7E04_02FF, 0x7E06_02FF));
        words[13] = f32::to_bits(u);
        words[15] = f32::to_bits(v);
        let (shader, _) = build_any(&words, false);
        let views = [
            View::Texture(4, 4, &texels),
            View::Texture(4, 4, &texels),
            View::RawUav(32),
        ];
        let results = run(&shader, &views);
        if !device_ran(&results) {
            return;
        }
        for (label, words) in results {
            let got: Vec<f32> = words.into_iter().map(f32::from_bits).collect();
            assert_eq!(got, vec![texel; 32], "{label}: ({u}, {v})");
        }
    }
}

#[test]
fn scalar_structured_load_and_record_count_use_the_same_view() {
    #[rustfmt::skip]
    let words = [
        0xbe840300, 0xbe850380,     // s[4:5] = view table
        0xf4080202, 0xfa000000,     // s_load_dwordx4 s[8:11], s[4:5], 0
        0xf4080302, 0xfa000040,     // s_load_dwordx4 s[12:15], s[4:5], 64
        0xf4200404, 0xfa000014,     // s_buffer_load_dword s16, s[8:11], 20
        0x80110a10,                 // s_add_u32 s17, s16, s10 (record count)
        0x7e020211,                 // v_mov_b32 v1, s17
        0x34040082,                 // v_lshlrev_b32 v2, 2, v0 (lane * 4)
        0xe0701000, 0x80030102,     // buffer_store_dword v1, v2, s[12:15], 0 offen
        0xbf810000,                 // s_endpgm
    ];
    // The second program uses descriptor word 2 only as the byte offset
    // of the scalar buffer load that also consumes the whole descriptor.
    let mut offset_words = words;
    offset_words[7] = 0x14000000; // soffset = s10, immediate offset = 0
    offset_words[8] = 0xbe910310; // s_mov_b32 s17, s16
    let (sum, _) = build_with_view(&words, Some(16));
    let (offset, _) = build_with_view(&offset_words, Some(16));
    for records in [4, 8] {
        let mut data = vec![0u32; records * 4];
        data[1] = 0x1234;
        data[2] = 0x5678;
        data[5] = 0x9012;
        let views = [
            View::Buffer {
                data: &data,
                stride: Stride::Structured(16),
            },
            View::Texture(1, 1, &[[0.0; 4]]),
            View::RawUav(32),
        ];
        for (shader, expected) in [
            (&sum, data[5] + records as u32),
            (&offset, data[records / 4]),
        ] {
            let results = run(shader, &views);
            if !device_ran(&results) {
                return;
            }
            for (label, result) in results {
                assert_eq!(result, vec![expected; 32], "{label}");
            }
        }
    }
}

/// A raw view's record count is its size in bytes, read from the same
/// descriptor the scalar load takes whole: the program has no stride to
/// give, so the view is raw like the load's.
#[test]
fn scalar_load_and_record_count_use_the_same_raw_view() {
    #[rustfmt::skip]
    let words = [
        0xbe840300, 0xbe850380,     // s[4:5] = view table
        0xf4080202, 0xfa000000,     // s_load_dwordx4 s[8:11], s[4:5], 0
        0xf4080302, 0xfa000040,     // s_load_dwordx4 s[12:15], s[4:5], 64
        0xf4200404, 0xfa000014,     // s_buffer_load_dword s16, s[8:11], 20
        0x80110a10,                 // s_add_u32 s17, s16, s10 (record count)
        0x7e020211,                 // v_mov_b32 v1, s17
        0x34040082,                 // v_lshlrev_b32 v2, 2, v0 (lane * 4)
        0xe0701000, 0x80030102,     // buffer_store_dword v1, v2, s[12:15], 0 offen
        0xbf810000,                 // s_endpgm
    ];
    let (shader, _) = build_with_view(&words, Some(0));
    for dwords in [8usize, 16] {
        let data: Vec<u32> = (0..dwords as u32).map(|n| 0x100 + n).collect();
        let views = [
            View::Buffer {
                data: &data,
                stride: Stride::Raw,
            },
            View::Texture(1, 1, &[[0.0; 4]]),
            View::RawUav(32),
        ];
        let results = run(&shader, &views);
        if !device_ran(&results) {
            return;
        }
        for (label, result) in results {
            let want = data[5] + 4 * dwords as u32;
            assert_eq!(result, vec![want; 32], "{label}: {dwords} dwords");
        }
    }
}
