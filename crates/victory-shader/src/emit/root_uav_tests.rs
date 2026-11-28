//! A synthetic compute shader writes one word through a spilled root descriptor.

use super::{HostShader, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

#[test]
fn compute_writes_through_spilled_root_uav() {
    let slot = 3;
    let words: [u32; 8] = [
        0xF404_0200, // s_load_dwordx2 s[8:9], s[0:1], 0
        0xFA00_0000,
        0xBE8A_0384, // descriptor size
        0xBE8B_0380, // remaining descriptor flags
        0x7E00_029D, // v_mov_b32 v0, 29
        0xE070_0004, // buffer_store_dword v0, off, s[8:11], 0 offset:4
        0x8002_0000,
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 1,
        layout: vec![RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 16,
            slot,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 16,
            slot,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p =
        ir::build_with(&native.code, native.wave64, native.vgprs, true).expect("ir::build_with");
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
        mesh_buffer: false,
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
        root_uavs: lowered.root_uavs,
        structured_srvs: Vec::new(),
    };

    assert_eq!(shader.root_uavs, [slot]);
    assert!(shader.root_srvs.is_empty());
    #[cfg(windows)]
    check_readback(&shader);
}

#[cfg(windows)]
fn check_readback(shader: &HostShader) {
    use super::host::{Dispatch, Host, Root, View};
    use super::valu_tests::{ONE_AT_A_TIME, backends};

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    let force_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    for &(label, warp) in backends(force_warp) {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (
                            0,
                            Root::Constants(vec![0; shader.user_data_dwords().max(1) as usize]),
                        ),
                        (
                            1,
                            Root::Table {
                                sampler: false,
                                index: 1,
                            },
                        ),
                    ],
                    views: &[View::RawUav(4), View::RawUav(4)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out, [vec![0; 4], vec![0, 29, 0, 0]], "{label}");
        eprintln!("{label}: root UAV write verified");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// A load, a store and an atomic through one root UAV, lowered: the shader
/// reads the host's bias word for that root slot
/// ([`super::USER_ROOT_VIEW_BIAS`]), so the host can bind a root view whose
/// address is not on 16 bytes at the boundary below it, and the DXIL
/// validates.
#[test]
fn every_access_through_a_root_uav_adds_the_view_bias() {
    let slot = 3;
    let words: [u32; 16] = [
        0xF404_0200, // s_load_dwordx2 s[8:9], s[0:1], 0
        0xFA00_0000,
        0xBE8A_0384, // descriptor size
        0xBE8B_0380, // remaining descriptor flags
        0x7E00_029D, // v_mov_b32 v0, 29
        0xE070_0004, // buffer_store_dword v0, off, s[8:11], 0 offset:4
        0x8002_0000,
        0xE030_0008, // buffer_load_dword v1, off, s[8:11], 0 offset:8
        0x8002_0100,
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_0010, // buffer_store_dword v1, off, s[8:11], 0 offset:16
        0x8002_0100,
        0xE0C8_000C, // buffer_atomic_add v0, off, s[8:11], 0 offset:12
        0x8002_0000,
        0xBF81_0000, // s_endpgm
        0xBF81_0000,
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 2,
        layout: vec![RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 16,
            slot,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 16,
            slot,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p =
        ir::build_with(&native.code, native.wave64, native.vgprs, true).expect("ir::build_with");
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
        mesh_buffer: false,
    })
    .expect("lower");
    assert_eq!(lowered.root_uavs, [slot]);
    let bias: Vec<_> = lowered
        .user_data
        .iter()
        .filter(|u| u.root == RootType::Other(super::USER_ROOT_VIEW_BIAS))
        .collect();
    assert_eq!(bias.len(), 1, "{:?}", lowered.user_data);
    assert_eq!(bias[0].slot, slot);
    #[cfg(windows)]
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut lowered.dxil.clone())
            .expect("DXIL validation failed");
    }
}
