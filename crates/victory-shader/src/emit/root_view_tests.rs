//! A synthetic compute shader writes by index through a buffer descriptor it
//! builds around a spilled root descriptor that its resource list leaves out:
//! a raw root UAV addressed at index * stride + offset.

use super::{HostShader, lower};
use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
use crate::{io, ir, resource, structure};

#[test]
fn compute_writes_by_index_through_an_unlisted_root_descriptor() {
    let slot = 3;
    let words: [u32; 11] = [
        0xF404_0200, // s_load_dwordx2 s[8:9], s[0:1], 0 (user SGPRs 16, 17)
        0xFA00_0000,
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x8809_FF09, // s_or_b32 s9, s9, 0x80000 (stride 8)
        0x0008_0000,
        0xBE8A_03C1, // s_mov_b32 s10, -1 (records)
        0xBE8B_0380, // s_mov_b32 s11, 0
        0x7E02_0300, // v_mov_b32 v1, v0
        0xE070_2004, // buffer_store_dword v1, v0, s[8:11], 0 idxen offset:4
        0x8002_0100,
        0xBF81_0000, // s_endpgm
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
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [8, 1, 1],
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
                        (0, Root::Constants(vec![0])),
                        (
                            1,
                            Root::Table {
                                sampler: false,
                                index: 1,
                            },
                        ),
                    ],
                    views: &[View::RawUav(4), View::RawUav(16)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let expected: Vec<u32> = (0..16)
            .map(|d| if d % 2 == 1 { d / 2 } else { 0 })
            .collect();
        assert_eq!(out[1], expected, "{label}");
        eprintln!("{label}: indexed root UAV writes verified");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
