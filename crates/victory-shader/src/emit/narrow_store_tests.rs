//! Synthetic byte and halfword buffer stores (`buffer_store_byte`,
//! `buffer_store_short` and their `_d16_hi` forms) to a raw UAV, where
//! neighbouring lanes write the two halves (or four bytes) of one dword.

use super::{HostShader, emit};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{Container, io, ir, resource, structure};

/// Lane `i` (of 64) writes the halfword `0xA000 + i` at byte `2i` (from the
/// low half of a VGPR) and at byte `128 + 2i` (from the high half), and the
/// byte `i` at byte `256 + i` (twice: low and high half).
const WORDS: &[u32] = &[
    0xBE81_0380, // s_mov_b32 s1, 0 (high half of the table address)
    0xF408_0300,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[0:1], 0
    0x4A02_00FF,
    0x0000_A000, // v_add_nc_u32 v1, 0xa000, v0
    0x3404_0081, // v_lshlrev_b32 v2, 1, v0
    0x3406_0290, // v_lshlrev_b32 v3, 16, v1
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE068_1000,
    0x8003_0102, // buffer_store_short v1, v2, s[12:15], 0 offen
    0xE06C_1080,
    0x8003_0302, // buffer_store_short_d16_hi v3, v2, s[12:15], 0 offen offset:128
    0xE060_1100,
    0x8003_0100, // buffer_store_byte v1, v0, s[12:15], 0 offen offset:256
    0xE064_1140,
    0x8003_0300, // buffer_store_byte_d16_hi v3, v0, s[12:15], 0 offen offset:320
    0xBF81_0000, // s_endpgm
];

fn shader() -> HostShader {
    shader_of(WORDS)
}

fn shader_of(words: &[u32]) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
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
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1, // one user SGPR
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let bytes = crate::dxbc::build(&[]);
    let container = Container::parse(&bytes).unwrap();
    emit(&container, &native, &p, &st, &io, &res).unwrap()
}

fn expected() -> Vec<u32> {
    let mut bytes = vec![0u8; 384];
    for i in 0..64usize {
        let half = (0xA000 + i as u16).to_le_bytes();
        bytes[2 * i..2 * i + 2].copy_from_slice(&half);
        bytes[128 + 2 * i..130 + 2 * i].copy_from_slice(&half);
        bytes[256 + i] = i as u8;
        bytes[320 + i] = i as u8;
    }
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(u32::from_le_bytes)
        .collect()
}

#[test]
fn byte_and_halfword_stores_translate() {
    let shader = shader();
    #[cfg(windows)]
    if let Ok(mut validator) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        validator.validate(&mut shader.dxil.clone()).unwrap();
    }
}

#[cfg(windows)]
#[test]
fn byte_and_halfword_stores_keep_the_rest_of_each_dword() {
    use super::host::{Dispatch, Host, Root, View};
    let shader = shader();
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
                        Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                    )],
                    views: &[View::RawUav(96)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        assert_eq!(output[0], expected(), "{name}");
        eprintln!("{name}: verified byte and halfword stores");
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

/// Every lane stores its own one-bit byte (`1 << (lane & 7)`) to byte 0:
/// one lane's store wins, so the byte holds one bit, never the bits of
/// several lanes that cleared it before any of them set theirs.
const RACE: &[u32] = &[
    0xBE81_0380, // s_mov_b32 s1, 0
    0xF408_0300,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[0:1], 0
    0x3608_0087, // v_and_b32 v4, 7, v0
    0x7E0A_0281, // v_mov_b32 v5, 1
    0x3402_0B04, // v_lshlrev_b32 v1, v4, v5
    0x7E0C_0280, // v_mov_b32 v6, 0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE060_1000,
    0x8003_0106, // buffer_store_byte v1, v6, s[12:15], 0 offen
    0xBF81_0000, // s_endpgm
];

#[cfg(windows)]
#[test]
fn lanes_storing_the_same_byte_leave_one_lanes_value() {
    use super::host::{Dispatch, Host, Root, View};
    let shader = shader_of(RACE);
    if let Ok(mut validator) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        validator.validate(&mut shader.dxil.clone()).unwrap();
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let force_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    let mut ran = 0;
    for &(name, warp) in super::valu_tests::backends(force_warp) {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        ran += 1;
        for _ in 0..8 {
            let output = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(
                            0,
                            Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                        )],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            let word = output[0][0];
            assert_eq!(word.count_ones(), 1, "{name}: {word:#x}");
            assert_eq!(word & !0xFF, 0, "{name}: {word:#x}");
        }
    }
    if ran == 0 {
        // A GPU test without the capability says so and returns (see the
        // code standards), rather than failing on that machine.
        eprintln!("skipped: lanes storing the same byte needs an SM 6.6 device");
    }
}
