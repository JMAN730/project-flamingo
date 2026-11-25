use super::super::emit;
use crate::native::{ComputeIo, RootParam};
use crate::{Container, FourCC, HwStage, NativeProgram, ResourceType, RootType, UsedResource};

#[test]
fn selects_from_independently_written_mask_words() {
    let words = [
        0xbe81_0380u32, // s1 = 0 (table address high)
        0xf408_0200,
        0xfa00_0000, // load UAV into s[8:11]
        0xbe84_03ff,
        0x8000_0000, // s4 = lane 31
        0xbe85_0381, // s5 = lane 32 in wave64
        0x7e02_028b,
        0x7e04_029d, // v1 = 11, v2 = 29
        0xd501_0003,
        0x0012_0501, // v_cndmask_b32 v3, v1, v2, s[4:5]
        0x3402_0082, // v1 = 4 * v0
        0xe070_1000,
        0x8002_0301, // store v3 to UAV at v1
        0xbf81_0000,
    ];
    let container = crate::dxbc::build(&[]);
    let c = Container::parse(&container).unwrap();
    for (wave64, uses_select) in [(false, true), (true, true), (true, false)] {
        let mut words = words;
        if !uses_select {
            words[12] = 0x8002_0101; // store v1 instead of the select result
        }
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 4,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: crate::native::StageIo::Cs(ComputeIo {
                rsrc2: 2,
                threads: [if wave64 { 64 } else { 32 }, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let shader = emit(&c, &native, &p, &st, &io, &res).unwrap();
        let emitted = Container::parse(&shader.dxil).unwrap();
        let features = emitted.part(FourCC::SFI0).unwrap();
        let wave_ops = u64::from_le_bytes(features.data.try_into().unwrap()) & 0x4000 != 0;
        assert_eq!(wave_ops, uses_select);
    }
}

#[test]
fn scalar_buffer_load_from_structured_srv_uses_structured_resource() {
    // Load a structured SRV descriptor (stride 24) from table s0 into s[4:7],
    // read two dwords at byte offset 8 with s_buffer_load_dwordx2, and store
    // the second dword to a raw UAV at table offset 4 dwords.
    let words = [
        0xbe81_0380u32, // s1 = 0 (table address high)
        0xf408_0100,
        0xfa00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0x0 (structured SRV)
        0xf408_0200,
        0xfa00_0010, // s_load_dwordx4 s[8:11], s[0:1], 0x10 (raw UAV)
        0xf424_0302,
        0xfa00_0008, // s_buffer_load_dwordx2 s[12:13], s[4:7], 0x8
        0x7e02_020d, // v1 = s13
        0x3404_0082, // v2 = 4 * v0
        0xe070_1000,
        0x8002_0102, // buffer_store_dword v1, v2, s[8:11], 0 offen
        0xbf81_0000, // s_endpgm
    ];
    // The shader for an SRV of `stride` bytes per element.
    let lowered = |stride: u32| {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![
                UsedResource {
                    kind: ResourceType::Srv,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride,
                    dwords: 4,
                    user_data: 0,
                    slot: 0,
                },
                UsedResource {
                    kind: ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 4,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 0,
                    slot: 0,
                },
            ],
            io: crate::native::StageIo::Cs(ComputeIo {
                rsrc2: 2,
                threads: [32, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        let p = crate::ir::build(&native.code, false, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        emit(&c, &native, &p, &st, &io, &res).unwrap()
    };
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut shader = lowered(24);
    assert!(!shader.dxil.is_empty());
    // The stride is what makes the load a structured one: the same
    // program over an SRV without one (a raw buffer) lowers differently.
    assert_ne!(shader.dxil, lowered(0).dxil);
    #[cfg(windows)]
    if let Ok(mut v) = crate::emit::validator::Validator::load(None) {
        v.validate(&mut shader.dxil)
            .expect("structured scalar SRV load DXIL must validate");
    }
}

#[test]
fn scalar_structured_load_with_wave_uniform_offset_compiles_for_mesh_and_pixel() {
    use crate::emit::container;
    use crate::native::{MeshIo, StageIo};

    // Load a structured table SRV (non-power-of-two stride 1580 and 68)
    // into s[4:7], derive a wave-uniform byte offset via
    // `v_readfirstlane_b32 s12, v0`, read via `s_buffer_load_dword`, and
    // export the result. Without reciprocal-multiply lowering, NVIDIA's
    // DXIL compiler (`nvgpucomp64.dll`) crashes with E_OUTOFMEMORY inside
    // `CreatePipelineState` on graphics stages (`ms`/`ps`/`vs`).
    let mesh_words = [
        0xbe89_0380u32, // s_mov_b32 s9, 0 (user data starts at s8 in mesh)
        0xf408_0104,
        0xfa00_0000, // s_load_dwordx4 s[4:7], s[8:9], 0x0 (structured SRV)
        0x7e18_0500, // v_readfirstlane_b32 s12, v0
        0x8f0c_820c, // s_lshl_b32 s12, s12, 2
        0xf420_0342,
        0x1800_0000, // s_buffer_load_dword s13, s[4:7], s12
        0x7e02_020d, // v_mov_b32 v1, s13
        0xf800_00cf,
        0x0101_0101, // exp pos0 v1, v1, v1, v1
        0xbf81_0000, // s_endpgm
    ];
    let ps_words = [
        0xbe81_0380u32, // s_mov_b32 s1, 0
        0xf408_0100,
        0xfa00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0x0 (structured SRV)
        0x7e18_0500, // v_readfirstlane_b32 s12, v0
        0x8f0c_820c, // s_lshl_b32 s12, s12, 2
        0xf420_0342,
        0x1800_0000, // s_buffer_load_dword s13, s[4:7], s12
        0x7e02_020d, // v_mov_b32 v1, s13
        0xf800_080f,
        0x0101_0101, // exp mrt0 v1, v1, v1, v1 done
        0xbf81_0000, // s_endpgm
    ];
    let pos_sig = container::signature(
        &[container::SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xf,
            half: false,
        }],
        false,
    );
    let target_sig = container::signature(
        &[container::SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xf,
            half: false,
        }],
        false,
    );
    let mesh_container = crate::dxbc::build(&[(FourCC::OSG1, &pos_sig)]);
    let ps_container = crate::dxbc::build(&[(FourCC::ISG1, &pos_sig), (FourCC::OSG1, &target_sig)]);

    for stride in [68u32, 1580] {
        let layout = vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }];
        let used = vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }];
        let ms_native = NativeProgram {
            stage: HwStage::Ms,
            wave64: false,
            vgprs: 4,
            layout: layout.clone(),
            used: used.clone(),
            io: StageIo::Ms(MeshIo {
                outputs: vec![],
                clip_cntl: 0,
                threads: 32,
                max_vertices: 3,
                max_primitives: 1,
                topology: 2,
                lds_bytes: 0,
            }),
            code: mesh_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let ps_native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout,
            used,
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0x100, // SV_Position.x in v0
                rsrc2: Default::default(),
                db_shader_control: 0,
                patches: vec![],
            },
            code: ps_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mc = Container::parse(&mesh_container).unwrap();
        let mp = crate::ir::build(&ms_native.code, false, ms_native.vgprs).unwrap();
        let mst = crate::structure::structure(&mp).unwrap();
        let mio = crate::io::raise(&mp, &ms_native).unwrap();
        let mres =
            crate::resource::track(&mp, &ms_native.layout, ms_native.stage, Some(&mst)).unwrap();
        let ms = emit(&mc, &ms_native, &mp, &mst, &mio, &mres).unwrap();

        let pc = Container::parse(&ps_container).unwrap();
        let pp = crate::ir::build(&ps_native.code, false, ps_native.vgprs).unwrap();
        let pst = crate::structure::structure(&pp).unwrap();
        let pio = crate::io::raise(&pp, &ps_native).unwrap();
        let pres =
            crate::resource::track(&pp, &ps_native.layout, ps_native.stage, Some(&pst)).unwrap();
        let ps = emit(&pc, &ps_native, &pp, &pst, &pio, &pres).unwrap();

        #[cfg(not(windows))]
        let _ = (&ms, &ps);

        #[cfg(windows)]
        {
            if let Ok(mut v) = crate::emit::validator::Validator::load(None) {
                v.validate(&mut ms.dxil.clone())
                    .unwrap_or_else(|e| panic!("ms stride {stride} validation failed: {e}"));
                v.validate(&mut ps.dxil.clone())
                    .unwrap_or_else(|e| panic!("ps stride {stride} validation failed: {e}"));
            }
            let _one = super::super::valu_tests::ONE_AT_A_TIME
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Ok(host) = super::super::host::Host::new(false) {
                let stub_ps = super::super::host::stub_ps();
                // Mesh pipeline creation requires mesh shader capability; check via CreatePipelineState.
                let mut opts7 =
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
                // SAFETY: querying D3D12_FEATURE_D3D12_OPTIONS7 into a zeroed struct.
                let has_mesh = unsafe {
                    host.device.CheckFeatureSupport(
                        windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut opts7).cast(),
                        size_of_val(&opts7) as u32,
                    )
                }
                .is_ok()
                    && opts7.MeshShaderTier
                        != windows::Win32::Graphics::Direct3D12::D3D12_MESH_SHADER_TIER_NOT_SUPPORTED;
                if has_mesh {
                    host.mesh_pipeline(&ms, &stub_ps).unwrap_or_else(|e| {
                        panic!("hardware mesh pipeline stride {stride} failed: {e}")
                    });
                    host.mesh_pipeline(&ms, &ps).unwrap_or_else(|e| {
                        panic!("hardware mesh+pixel pipeline stride {stride} failed: {e}")
                    });
                }
                let vs = super::super::host::stub_vs(&ps).unwrap();
                host.pipeline(&vs, &ps).unwrap_or_else(|e| {
                    panic!("hardware pixel pipeline stride {stride} failed: {e}")
                });
            }
        }
    }
}

#[test]
fn structured_div_magic_is_exact_across_u32_boundaries() {
    let half_mul_div_rem = |addr: u32, stride: u32| -> (u32, u32) {
        let tz = stride.trailing_zeros();
        let odd = stride >> tz;
        if odd == 1 {
            return (addr >> tz, addr & (stride - 1));
        }
        let (magic, shift) = super::structured_div_magic(odd);
        let shifted = addr >> tz;
        let a_lo = shifted & 0xffff;
        let a_hi = shifted >> 16;
        let m_lo = magic & 0xffff;
        let m_hi = magic >> 16;
        let p0 = a_lo.checked_mul(m_lo).unwrap();
        let p1 = a_lo.checked_mul(m_hi).unwrap();
        let p2 = a_hi.checked_mul(m_lo).unwrap();
        let p3 = a_hi.checked_mul(m_hi).unwrap();
        let mid = (p0 >> 16)
            .checked_add(p1 & 0xffff)
            .unwrap()
            .checked_add(p2 & 0xffff)
            .unwrap();
        let hi = p3
            .checked_add(p1 >> 16)
            .unwrap()
            .checked_add(p2 >> 16)
            .unwrap()
            .checked_add(mid >> 16)
            .unwrap();
        let q = hi >> shift;
        let r = addr.checked_sub(q.checked_mul(stride).unwrap()).unwrap();
        (q, r)
    };
    let mut strides: Vec<u32> = (4..=2048).step_by(4).collect();
    strides.extend([
        0x0001_0004,
        0x3fff_fffc,
        0x8000_0004,
        0xc000_000c,
        0xffff_fffc,
    ]);
    for stride in strides {
        let samples = [
            0,
            1,
            3,
            4,
            stride - 1,
            stride,
            stride + 1,
            stride.saturating_mul(2).saturating_sub(1),
            stride.saturating_mul(2),
            stride.saturating_mul(7).saturating_add(4),
            0x3fff_fffc,
            0x7fff_ffff,
            0x8000_0000,
            u32::MAX - stride,
            u32::MAX - stride + 1,
            u32::MAX - 4,
            u32::MAX - 1,
            u32::MAX,
        ];
        for addr in samples {
            assert_eq!(
                half_mul_div_rem(addr, stride),
                (addr / stride, addr % stride),
                "addr={addr:#x} stride={stride}"
            );
        }
    }
}

#[test]
fn scalar_structured_load_dynamic_offset_reads_exact_element_and_field() {
    // Compute shader: s0 = descriptor table, s1 = dynamic byte offset (root constant).
    // Loads structured SRV (slot 0, stride S) into s[4:7], raw UAV (slot 1) into s[8:11],
    // reads two dwords at byte offset s1 via s_buffer_load_dwordx2, and stores both to UAV.
    let words = [
        0xbe82_0300u32, // s_mov_b32 s2, s0
        0xbe83_0380,    // s_mov_b32 s3, 0
        0xf408_0101,
        0xfa00_0000, // s_load_dwordx4 s[4:7], s[2:3], 0x0 (structured SRV)
        0xf408_0201,
        0xfa00_0020, // s_load_dwordx4 s[8:11], s[2:3], 0x20 (raw UAV at heap slot 1)
        0xf424_0302,
        0x0200_0000, // s_buffer_load_dwordx2 s[12:13], s[4:7], s1
        0x7e02_020c, // v_mov_b32 v1, s12
        0x7e04_020d, // v_mov_b32 v2, s13
        0xe070_1000,
        0x8002_0180, // buffer_store_dword v1, off, s[8:11], 0
        0xe070_1004,
        0x8002_0280, // buffer_store_dword v2, off, s[8:11], 0 offset:4
        0xbf81_0000, // s_endpgm
    ];
    for stride in [16u32, 24, 68] {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![
                RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                RootParam {
                    root: RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 1,
                    slot: 1,
                },
            ],
            used: vec![
                UsedResource {
                    kind: ResourceType::Srv,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride,
                    dwords: 4,
                    user_data: 0,
                    slot: 0,
                },
                UsedResource {
                    kind: ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 8,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 0,
                    slot: 0,
                },
            ],
            io: crate::native::StageIo::Cs(ComputeIo {
                rsrc2: 2,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        let p = crate::ir::build(&native.code, false, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let shader = emit(&c, &native, &p, &st, &io, &res).unwrap();
        let sfi0 = Container::parse(&shader.dxil)
            .unwrap()
            .part(FourCC::SFI0)
            .unwrap()
            .data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_eq!(
            features & (0x4000 | 0x8000),
            0,
            "structured offset split must not require WAVE_OPS or INT64"
        );

        #[cfg(windows)]
        {
            use super::super::host::{Dispatch, Host, Root, Stride, View};
            if let Ok(mut v) = crate::emit::validator::Validator::load(None) {
                v.validate(&mut shader.dxil.clone())
                    .unwrap_or_else(|e| panic!("cs stride {stride} validation failed: {e}"));
            }
            let _one = super::super::valu_tests::ONE_AT_A_TIME
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let elem_dwords = (stride / 4) as usize;
            let elements = 8usize;
            let data: Vec<u32> = (0..(elem_dwords * elements) as u32)
                .map(|i| 0x1000_0000 + i)
                .collect();
            // Test offsets crossing element boundaries (last dword of element 2 + first of 3,
            // plus middle of element 5).
            let test_offsets = [0u32, stride - 4, stride * 2 + stride - 4, stride * 5 + 8];
            for (label, warp) in [("hardware", false), ("warp", true)] {
                let Ok(host) = Host::new(warp) else {
                    continue;
                };
                let (root, pso) = host
                    .compute_pipeline(&shader)
                    .unwrap_or_else(|e| panic!("{label} stride {stride}: {e}"));
                for byte_off in test_offsets {
                    let mut ud = vec![0u32; shader.user_data_dwords().max(2) as usize];
                    ud[0] = 0; // heap slot 0
                    ud[1] = byte_off;
                    let out = host
                        .dispatch(
                            &root,
                            &pso,
                            &Dispatch {
                                roots: vec![(0, Root::Constants(ud))],
                                views: &[
                                    View::Buffer {
                                        data: &data,
                                        stride: Stride::Structured(stride),
                                    },
                                    View::RawUav(2),
                                ],
                                samplers: &[],
                                groups: [1, 1, 1],
                            },
                        )
                        .unwrap_or_else(|e| panic!("{label} stride {stride} off {byte_off}: {e}"));
                    let w0 = data[(byte_off / 4) as usize];
                    let w1 = data[(byte_off / 4) as usize + 1];
                    assert_eq!(
                        out[0],
                        vec![w0, w1],
                        "{label} stride {stride} byte_off {byte_off}"
                    );
                }
            }
        }
    }
}
