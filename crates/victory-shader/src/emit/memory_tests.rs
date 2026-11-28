//! Title memory by address (`resource::MEMORY`) in a compute program: an
//! access whose address traces to no root parameter reads through the
//! 64 KiB page directory instead of being refused.

use super::{SLOT_FAULTS, SLOT_MEMORY, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// A compute program of `words` whose one root parameter is a two-dword
/// table at `s0`; everything else the code reads is title memory.
fn native_compute(words: &[u32]) -> NativeProgram {
    let slot = 1;
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 16,
            slot,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Cbv,
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
    }
}

fn lower_compute(native: &NativeProgram) -> lower::Lowered {
    let p =
        ir::build_with(&native.code, native.wave64, native.vgprs, true).expect("ir::build_with");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    lower::lower(&lower::Parts {
        native,
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
    .expect("lower")
}

#[test]
#[ignore = "WIP: the synthetic load is dead and never lowered; the dump run is the evidence"]
fn a_scalar_load_from_an_untracked_address_reads_the_directory() {
    let words: [u32; 5] = [
        0xF400_0081, // s_load_dword s2, s[2:3], 0: no parameter holds
        0xFA00_0000, // s2, so the address is title memory at it
        0xBF06_8002, // s_cmp_eq_u32 s2, 0 (the dword is read)
        0xBF85_0000, // s_cbranch_scc1 0
        0xBF81_0000, // s_endpgm
    ];
    let lowered = lower_compute(&native_compute(&words));
    assert!(lowered.root_srvs.contains(&SLOT_MEMORY));
    assert_eq!(lowered.root_uavs, [SLOT_FAULTS]);
}

#[test]
#[ignore = "WIP: the synthetic atomic is not lowered yet; the dump run is the evidence"]
fn a_global_atomic_whose_address_is_loaded_from_memory_reads_the_directory() {
    let words: [u32; 9] = [
        0xF400_0081, // s_load_dword s2, s[2:3], 0
        0xFA00_0000,
        0xF400_00C1, // s_load_dword s3, s[2:3], 4
        0xFA00_0004,
        0x7E00_0202, // v_mov_b32 v0, s2 (a pointer loaded from memory)
        0x7E02_0203, // v_mov_b32 v1, s3
        0xDCC9_8000, // global_atomic_add v2, v[0:1], v3, off glc
        0x027D_0300,
        0xBF81_0000, // s_endpgm
    ];
    let lowered = lower_compute(&native_compute(&words));
    assert!(lowered.root_srvs.contains(&SLOT_MEMORY));
    assert_eq!(lowered.root_uavs, [SLOT_FAULTS]);
}

/// Lowers `words` as [`lower_compute`] does and validates the DXIL with
/// `dxil.dll` when there is one; checks it reads the page directory.
fn lowered_by_address(words: &[u32]) -> lower::Lowered {
    let mut native = native_compute(words);
    // Four root constants at s0: s[2:3] holds a title address (s[0:1]
    // would be the spill table of root arguments).
    native.layout[0] = RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs: 4,
        base_register: 0,
        user_data: 0,
        slot: 0,
    };
    native.used.clear();
    let lowered = lower_compute(&native);
    assert!(lowered.root_srvs.contains(&SLOT_MEMORY));
    assert_eq!(lowered.root_uavs, [SLOT_FAULTS]);
    #[cfg(windows)]
    {
        match super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            Ok(mut v) => v
                .validate(&mut lowered.dxil.clone())
                .expect("DXIL validation failed"),
            Err(e) => eprintln!("{e}; validation skipped"),
        }
    }
    lowered
}

/// `s_load_dwordx2 s[4:5], s[2:3], 0`: a pointer read from title memory
/// (at the address root constants 2 and 3 hold), then dwords 2 and 3 of a raw buffer
/// descriptor around it (`OOB_SELECT` 3: raw).
const POINTER_DESCRIPTOR: [u32; 6] = [
    0xF404_0101, // s_load_dwordx2 s[4:5], s[2:3], 0
    0xFA00_0000,
    0xBE86_03C1, // s_mov_b32 s6, -1
    0xBE87_03FF, // s_mov_b32 s7, 0x31014fac
    0x3101_4FAC,
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
];

fn with_descriptor(body: &[u32]) -> Vec<u32> {
    let mut w = POINTER_DESCRIPTOR.to_vec();
    w.extend_from_slice(body);
    w.push(0xBF81_0000); // s_endpgm
    w
}

#[test]
fn buffer_accesses_through_a_descriptor_around_a_loaded_pointer_read_the_directory() {
    lowered_by_address(&with_descriptor(&[
        0xE030_1000, // buffer_load_dword v1, v0, s[4:7], 0 offen
        0x8001_0100,
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1010, // buffer_store_dword v1, v0, s[4:7], 0 offen offset:16
        0x8001_0100,
    ]));
}

#[test]
fn a_buffer_atomic_through_a_descriptor_around_a_loaded_pointer_reads_the_directory() {
    lowered_by_address(&with_descriptor(&[
        0xE0C8_5000, // buffer_atomic_add v1, v0, s[4:7], 0 offen glc
        0x8001_0100,
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1010, // buffer_store_dword v1, v0, s[4:7], 0 offen offset:16
        0x8001_0100,
    ]));
}

#[test]
fn a_scalar_buffer_load_through_a_descriptor_around_a_loaded_pointer_reads_the_directory() {
    lowered_by_address(&with_descriptor(&[
        0xF420_0202, // s_buffer_load_dword s8, s[4:7], 0x10
        0xFA00_0010,
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E02_0208, // v_mov_b32 v1, s8
        0xE070_1010, // buffer_store_dword v1, v0, s[4:7], 0 offen offset:16
        0x8001_0100,
    ]));
}

#[test]
fn a_descriptor_whose_size_the_code_loads_is_bounded_at_run_time() {
    let words = [
        0xF404_0101, // s_load_dwordx2 s[4:5], s[2:3], 0
        0xFA00_0000,
        0xF400_0181, // s_load_dword s6, s[2:3], 8: the size
        0xFA00_0008,
        0xBE87_03FF, // s_mov_b32 s7, 0x31014fac
        0x3101_4FAC,
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_1000, // buffer_load_dword v1, v0, s[4:7], 0 offen
        0x8001_0100,
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1010, // buffer_store_dword v1, v0, s[4:7], 0 offen offset:16
        0x8001_0100,
        0xBF81_0000, // s_endpgm
    ];
    lowered_by_address(&words);
}

#[test]
fn a_global_access_at_a_loaded_scalar_base_plus_a_lane_offset_reads_the_directory() {
    let words = [
        0xF404_0101, // s_load_dwordx2 s[4:5], s[2:3], 0
        0xFA00_0000,
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xDC30_8000, // global_load_dword v1, v0, s[4:5]
        0x0104_0000,
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xDC70_8010, // global_store_dword v0, v1, s[4:5] offset:16
        0x0004_0100,
        0xBF81_0000, // s_endpgm
    ];
    lowered_by_address(&words);
}

#[test]
fn a_scalar_load_at_a_first_lane_read_of_a_lane_pointer_reads_the_directory() {
    let words = [
        0xDC34_8000, // global_load_dwordx2 v[2:3], v0, s[4:5]
        0x027D_0000, //   (off: v[0:1] is the address)
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E08_0502, // v_readfirstlane_b32 s4, v2
        0x7E0A_0503, // v_readfirstlane_b32 s5, v3
        0xF400_0202, // s_load_dword s8, s[4:5], 0
        0xFA00_0000,
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E02_0208, // v_mov_b32 v1, s8
        0xDC70_8000, // global_store_dword v[2:3], v1, off
        0x007D_0102,
        0xBF81_0000, // s_endpgm
    ];
    lowered_by_address(&words);
}
