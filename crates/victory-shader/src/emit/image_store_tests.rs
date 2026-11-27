//! Synthetic image stores of some channels (`dmask` not 0xF) to an RGBA32F
//! texture UAV. DXIL's typed UAV stores write all four components, so the
//! texel's other channels are read back and written again.

use super::{HostShader, emit};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{Container, io, ir, resource, structure};

const LANES: usize = 32;

/// `v_mov_b32 vdst, src0`.
const fn v_mov_b32(vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | 0x01 << 9 | src0
}

/// `image_store` of `dmask`'s channels, 2D: the data from `vdata`, the
/// coordinates from v2 and v3, the descriptor s[12:19].
const fn image_store(dmask: u32, vdata: u32) -> [u32; 2] {
    [
        0xF000_0000 | 0x08 << 18 | dmask << 8 | 2 << 2,
        0x0003_0002 | vdata << 8,
    ]
}

/// Lane `i` stores to texel (i, 0): all four channels (1, 2, 3, 4), then
/// y alone (9), then x and z (7, 8), which are packed from one VGPR pair.
fn words() -> Vec<u32> {
    let mut w = vec![
        0xBE81_0380, // s_mov_b32 s1, 0 (high half of the table address)
        0xF40C_0300,
        0xFA00_0000,         // s_load_dwordx8 s[12:19], s[0:1], 0
        v_mov_b32(2, 0x100), // v2 = lane
        v_mov_b32(3, 0x80),  // v3 = 0
        v_mov_b32(4, 0xF2),  // v4 = 1.0
        v_mov_b32(5, 0xF4),  // v5 = 2.0
        v_mov_b32(6, 0xFF),
        3.0f32.to_bits(),   // v6 = 3.0
        v_mov_b32(7, 0xF6), // v7 = 4.0
        v_mov_b32(8, 0xFF),
        9.0f32.to_bits(), // v8 = 9.0
        v_mov_b32(10, 0xFF),
        7.0f32.to_bits(), // v10 = 7.0
        v_mov_b32(11, 0xFF),
        8.0f32.to_bits(), // v11 = 8.0
        0xBF8C_C07F,      // s_waitcnt lgkmcnt(0)
    ];
    w.extend(image_store(0xF, 4));
    w.extend(image_store(0x2, 8));
    w.extend(image_store(0x5, 10));
    w.push(0xBF81_0000); // s_endpgm
    w
}

fn shader(code: &[u32]) -> Result<HostShader, crate::ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
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
            dwords: 8,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1, // one user SGPR
            threads: [LANES as u32, 1, 1],
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    let bytes = crate::dxbc::build(&[]);
    let container = Container::parse(&bytes).unwrap();
    emit(&container, &native, &p, &st, &io, &res)
}

fn validate(shader: &HostShader) {
    if let Ok(mut validator) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        validator
            .validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
}

#[test]
fn image_stores_of_some_channels_translate() {
    validate(&shader(&words()).expect("translate"));
}

/// A channel mask no run of leading channels covers (y alone) is stored too.
#[test]
fn a_store_of_one_inner_channel_translates() {
    let mut w = words();
    w.truncate(w.len() - 1);
    w.extend(image_store(0x4, 8));
    w.push(0xBF81_0000);
    validate(&shader(&w).expect("translate"));
}

#[test]
fn image_stores_of_some_channels_keep_the_others() {
    use super::host::{Dispatch, Host, Root, View};
    let shader = shader(&words()).expect("translate");
    validate(&shader);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let force_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    let mut ran = 0;
    for &(name, warp) in super::valu_tests::backends(force_warp) {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        let output = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(
                        0,
                        Root::Constants(vec![0; shader.user_data_dwords().max(1) as usize]),
                    )],
                    views: &[View::TextureUav(LANES as u32, 1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        // (1, 2, 3, 4), then y = 9, then x = 7 and z = 8.
        let want: Vec<u32> = (0..LANES)
            .flat_map(|_| [7.0f32, 9.0, 8.0, 4.0].map(f32::to_bits))
            .collect();
        assert_eq!(output[0], want, "{name}");
        eprintln!("{name}: verified image stores of some channels");
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}
