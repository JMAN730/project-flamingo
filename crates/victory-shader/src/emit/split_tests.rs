//! A wave64 compute program on a host without 64-lane waves runs each
//! console wave as two 32-lane halves (`lower::split`): synthetic GFX10
//! programs of one thread group (one console wave or several), lowered
//! with `WaveSize(32)`, validated by `dxil.dll` when present, and run on a
//! hardware GPU with 32-lane waves (skipped without one; `SPLIT_WARP` set
//! runs them on WARP instead) against the console's result, thread by
//! thread. `SPLIT_DUMP=<file>` writes the last shader built.

use super::host::{Dispatch, Host, Root, View};
use super::{FourCC, HostShader, lower, validator, with_host_wave64};
use crate::dxbc::Container;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const LANES: usize = 64;
const VCC: u32 = 106;
const EXEC: u32 = 126;

const fn sop1(op: u32, sdst: u32, src0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src0
}
const fn sop2(op: u32, sdst: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0
}
const fn s_not_b64(sdst: u32, src0: u32) -> u32 {
    sop1(0x08, sdst, src0)
}
const fn s_bcnt1_i32_b64(sdst: u32, src0: u32) -> u32 {
    sop1(0x10, sdst, src0)
}
const fn s_and_saveexec_b64(sdst: u32, src0: u32) -> u32 {
    sop1(0x24, sdst, src0)
}
const fn s_or_b64(sdst: u32, src0: u32, src1: u32) -> u32 {
    sop2(0x11, sdst, src0, src1)
}
const fn s_cbranch_execz(words: u16) -> u32 {
    0xBF88_0000 | words as u32
}
/// `v_cmp_gt_u32 vcc, n, v0`: the lanes below `n` (v0 is the lane).
const fn v_cmp_lanes_below(n: u32) -> u32 {
    0x7D88_0000 | (0x80 + n)
}
/// `v_cmp_lt_u32 vcc, n, v0`: the lanes above `n`.
const fn v_cmp_lanes_above(n: u32) -> u32 {
    0x7D82_0000 | (0x80 + n)
}
/// `v_mov_b32 vdst, src0` (`src0`: an SGPR, inline constant or 256 + VGPR).
const fn v_mov_b32(vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | 0x01 << 9 | src0
}
/// `v_readfirstlane_b32 sdst, v[src]`.
const fn v_readfirstlane_b32(sdst: u32, src: u32) -> u32 {
    0x7E00_0000 | sdst << 17 | 0x02 << 9 | (0x100 + src)
}
/// `v_readlane_b32 sdst, v[src], lane` (lane 0-64, an inline constant).
const fn v_readlane_b32(sdst: u32, src: u32, lane: u32) -> [u32; 2] {
    [0xD760_0000 | sdst, (0x80 + lane) << 9 | (0x100 + src)]
}
/// DS instruction with a 16-bit byte offset.
const fn ds(op: u32, addr: u32, data0: u32, dst: u32) -> [u32; 2] {
    [0xD800_0000 | op << 18, dst << 24 | data0 << 8 | addr]
}

/// Wraps `body` (its result in v1, 0 before it) in a wave64 program that
/// stores v1 at dword `lane` of the raw UAV at table entry 0.
fn program(body: &[u32]) -> Vec<u32> {
    let mut w = vec![
        0xBE81_0380, // s_mov_b32 s1, 0: s[0:1] is the table's address
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0x0
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        v_mov_b32(1, 0x80),
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    w.extend_from_slice(body);
    w.extend_from_slice(&[
        0xE070_1000,
        0x8002_0110, // buffer_store_dword v1, v16, s[8:11], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    w
}

/// `inside` runs in the lanes the compare `select` leaves in VCC
/// (`s_and_saveexec_b64 s[6:7], vcc` and `s_cbranch_execz` over it); EXEC
/// is restored from s[6:7] after it.
fn branch(select: u32, inside: &[u32]) -> Vec<u32> {
    let mut body = vec![
        select,
        s_and_saveexec_b64(6, VCC),
        s_cbranch_execz(inside.len() as u16),
    ];
    body.extend_from_slice(inside);
    body.push(s_or_b64(EXEC, EXEC, 6));
    program(&body)
}

/// Lowers `words` as a wave64 compute program of `threads` threads with
/// the table of views at s0 (the raw UAV at entry 0) for a host without
/// 64-lane waves.
fn lowered(
    words: &[u32],
    threads: u32,
    lds_blocks: u32,
) -> Result<lower::Lowered, crate::ShaderError> {
    lowered_group(words, [threads, 1, 1], lds_blocks)
}

/// [`lowered`] for a group of `dims` threads (thread id VGPRs to match).
fn lowered_group(
    words: &[u32],
    dims: [u32; 3],
    lds_blocks: u32,
) -> Result<lower::Lowered, crate::ShaderError> {
    let ids = u32::from(dims[1] > 1) + u32::from(dims[2] > 1);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
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
            rsrc2: 1 << 1 | ids << 11 | lds_blocks << 15,
            threads: dims,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    with_host_wave64(false, || {
        lower::lower(&lower::Parts {
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
    })
}

/// The lowered shader, validated, after checking it declares `WaveSize(32)`.
fn build(words: &[u32], threads: u32, lds_blocks: u32) -> HostShader {
    build_group(words, [threads, 1, 1], lds_blocks)
}

/// [`build`] for a group of `dims` threads.
fn build_group(words: &[u32], dims: [u32; 3], lds_blocks: u32) -> HostShader {
    let lowered = lowered_group(words, dims, lds_blocks).expect("lower");
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
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    if let Ok(d) = std::env::var("SPLIT_DUMP") {
        std::fs::write(d, &shader.dxil).unwrap();
    }
    let c = Container::parse(&shader.dxil).unwrap();
    let psv = c.part(FourCC::PSV0).unwrap().data;
    let min = u32::from_le_bytes(psv[20..24].try_into().unwrap());
    let max = u32::from_le_bytes(psv[24..28].try_into().unwrap());
    assert_eq!(
        (min, max),
        (32, 32),
        "the lowering must declare WaveSize(32)"
    );
    shader
}

/// Runs `shader` (one group) on a hardware GPU with 32-lane waves and
/// checks that every lane stored `want[lane]`.
fn check(shader: &HostShader, want: &[u32]) {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(std::env::var_os("SPLIT_WARP").is_some()) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    if !runs_wave32(&host) {
        eprintln!("hardware device has no 32-lane wave; not run");
        return;
    }
    let (root, pso) = host
        .compute_pipeline(shader)
        .unwrap_or_else(|e| panic!("compute_pipeline failed: {e}"));
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let views = [View::RawUav(want.len() as u32)];
    let words = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &views,
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    assert_eq!(words[0], want);
}

fn runs_wave32(host: &Host) -> bool {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
    };
    let mut opts1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the options struct matches the feature asked for.
    let ok = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&raw mut opts1).cast(),
            size_of_val(&opts1) as u32,
        )
    }
    .is_ok();
    ok && (opts1.WaveLaneCountMin..=opts1.WaveLaneCountMax).contains(&32)
}

fn lanes(f: impl Fn(usize) -> u32) -> Vec<u32> {
    (0..LANES).map(f).collect()
}

/// A compare read as a number holds both halves' lanes: 40 lanes counted,
/// 8 of them in the upper half.
#[test]
fn a_mask_read_as_a_number_counts_both_halves() {
    let shader = build(
        &program(&[
            v_cmp_lanes_below(40),
            s_bcnt1_i32_b64(4, VCC),
            v_mov_b32(1, 4),
        ]),
        64,
        0,
    );
    check(&shader, &lanes(|_| 40));
}

/// The high dword of a compare: the upper half's ballot.
#[test]
fn a_masks_high_dword_is_the_upper_half() {
    let shader = build(
        &program(&[v_cmp_lanes_below(40), v_mov_b32(1, VCC + 1)]),
        64,
        0,
    );
    check(&shader, &lanes(|_| 0xFF));
}

/// Inside a branch both halves run, the upper half with 4 of its lanes:
/// a mask negated there counts the lanes outside the branch too
/// (64 - 36 of the 40 lanes the branch holds).
#[test]
fn a_negated_mask_in_a_branch_counts_the_lanes_outside_it() {
    let shader = build(
        &branch(
            v_cmp_lanes_below(40),
            &[
                v_cmp_lanes_below(36),
                s_not_b64(20, VCC),
                s_bcnt1_i32_b64(4, 20),
                v_mov_b32(1, 4),
            ],
        ),
        64,
        0,
    );
    check(&shader, &lanes(|l| if l < 40 { 28 } else { 0 }));
}

/// The first lane EXEC holds, in a branch only the upper half's lanes take
/// (EXEC read as a number makes the program depend on the lane layout).
#[test]
fn the_first_active_lane_can_be_in_the_upper_half() {
    let shader = build(
        &branch(
            v_cmp_lanes_above(36),
            &[
                v_readfirstlane_b32(4, 0),
                s_bcnt1_i32_b64(5, EXEC),
                sop2(0x00, 4, 4, 5), // s_add_u32 s4, s4, s5
                v_mov_b32(1, 4),
            ],
        ),
        64,
        0,
    );
    // Lane 37, plus the 27 lanes above 36.
    check(&shader, &lanes(|l| if l > 36 { 64 } else { 0 }));
}

/// A lane read from the other half.
#[test]
fn a_lane_is_read_from_the_other_half() {
    let mut body = v_readlane_b32(4, 0, 50).to_vec();
    body.push(v_mov_b32(1, 4));
    let shader = build(&program(&body), 64, 0);
    check(&shader, &lanes(|_| 50));
}

/// The compiler's expansion of a whole-wave sum (DPP within rows,
/// `v_permlane(x)16` within halves, then lanes 31 and 63 read and added):
/// 1 + 2 + ... + 64 in every lane.
#[test]
fn a_whole_wave_sum_adds_both_halves() {
    let words = [
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null
        0x340C_0082, // v_lshlrev_b32_e32 v6, 2, v0
        0x4A0E_0081, // v_add_nc_u32_e32 v7, 1, v0
        0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
        0xD501_0001,
        0x0002_0E80, // v_cndmask_b32_e64 v1, 0, v7, s[0:1]
        0x4A02_02FA,
        0xFF01_6101, // v_add_nc_u32_dpp v1, v1, v1 row_xmask:1
        0x4A02_02FA,
        0xFF01_6201, // v_add_nc_u32_dpp v1, v1, v1 row_xmask:2
        0x4A02_02FA,
        0xFF01_6401, // v_add_nc_u32_dpp v1, v1, v1 row_xmask:4
        0x4A02_02FA,
        0xFF01_6801, // v_add_nc_u32_dpp v1, v1, v1 row_xmask:8
        0xD777_0002,
        0x0305_8301, // v_permlane16_b32 v2, v1, -1, -1
        0xD778_0003,
        0x0305_8302, // v_permlanex16_b32 v3, v2, -1, -1
        0x4A08_0702, // v_add_nc_u32_e32 v4, v2, v3
        0xD760_0002,
        0x0001_3F04, // v_readlane_b32 s2, v4, 31
        0xD760_0003,
        0x0001_7F04, // v_readlane_b32 s3, v4, 63
        0x7E0A_0203, // v_mov_b32_e32 v5, s3
        0x4A08_0A02, // v_add_nc_u32_e32 v4, s2, v5
        0xBEFE_0400, // s_mov_b64 exec, s[0:1]
        0xE070_1000,
        0x8002_0406, // buffer_store_dword v4, v6, s[8:11], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let shader = build(&words, 64, 0);
    check(&shader, &lanes(|_| 64 * 65 / 2));
    // A group of 32 threads: the absent upper half adds nothing.
    let shader = build(&words, 32, 0);
    check(&shader, &[32 * 33 / 2; 32]);
}

/// The compiler's expansion of a whole-wave sum of 16-bit floats in a
/// branch of the lanes below 40, its identity put into the lanes outside
/// the mask the section computes (the entry EXEC and the saved one). The
/// console rounds to f16 at each add of its expansion, in its order; a host
/// wave sum adds in another, so the sum is refused, not approximated.
#[test]
fn a_whole_wave_sum_of_halves_is_refused() {
    let inside = [
        0x7E0E_02FF,
        0x0000_3C00,           // v_mov_b32 v7, 1.0 (a half)
        0xBE84_25C1,           // s_or_saveexec_b64 s[4:5], -1
        sop2(0x0F, 12, 20, 4), // s_and_b64 s[12:13], s[20:21], s[4:5]
        0xD501_0009,
        0x0032_0E80, // v_cndmask_b32 v9, 0, v7, s[12:13]
        0x6412_12FA,
        0xFF01_6109, // v_add_f16_dpp v9, v9, v9 row_xmask:1
        0x6412_12FA,
        0xFF01_6209, // v_add_f16_dpp v9, v9, v9 row_xmask:2
        0x6412_12FA,
        0xFF01_6409, // v_add_f16_dpp v9, v9, v9 row_xmask:4
        0x6412_12FA,
        0xFF01_6809, // v_add_f16_dpp v9, v9, v9 row_xmask:8
        0xD777_0002,
        0x0305_8309, // v_permlane16_b32 v2, v9, -1, -1
        0xD778_0003,
        0x0305_8302, // v_permlanex16_b32 v3, v2, -1, -1
        0x6408_0702, // v_add_f16 v4, v2, v3
        0xD760_0002,
        0x0001_3F04, // v_readlane_b32 s2, v4, 31
        0xD760_0003,
        0x0001_7F04, // v_readlane_b32 s3, v4, 63
        0x7E0A_0203, // v_mov_b32 v5, s3
        0x6408_0A02, // v_add_f16 v4, s2, v5
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        v_mov_b32(1, 0x104),
    ];
    let mut words = branch(v_cmp_lanes_below(40), &inside);
    // The entry EXEC, kept before the branch.
    words.insert(6, sop1(0x04, 20, EXEC)); // s_mov_b64 s[20:21], exec
    let e = lowered(&words, 64, 0).err().expect("refused");
    assert!(
        e.to_string().contains("a whole-wave sum of 16-bit floats"),
        "{e}"
    );
}

/// One console wave shares LDS without barriers: each lane writes its id
/// and reads the one the mirror lane in the other half wrote.
#[test]
fn lds_written_by_the_other_half_is_read_after_it() {
    let mut body = ds(0x0D, 16, 0, 0).to_vec(); // ds_write_b32 v16, v0
    body.extend([
        0x4C22_20FF, // v_sub_nc_u32 v17, 252, v16
        252,
    ]);
    body.extend(ds(0x36, 17, 0, 1)); // ds_read_b32 v1, v17
    body.push(0xBF8C_C07F); // s_waitcnt lgkmcnt(0)
    let shader = build(&program(&body), 64, 1);
    check(&shader, &lanes(|l| 63 - l as u32));
}

/// A group of at most 32 threads has no upper half: plain `WaveSize(32)`.
#[test]
fn a_group_without_an_upper_half_needs_no_split() {
    let shader = build(
        &program(&[
            v_cmp_lanes_below(40),
            s_bcnt1_i32_b64(4, VCC),
            v_mov_b32(1, 4),
        ]),
        32,
        0,
    );
    check(&shader, &[32; 32]);
}

/// A partial console wave above 32 threads is refused on such hosts, not
/// run with the halves out of step.
#[test]
fn groups_of_a_partial_console_wave_are_refused() {
    let words = program(&[
        v_cmp_lanes_below(40),
        s_bcnt1_i32_b64(4, VCC),
        v_mov_b32(1, 4),
    ]);
    let e = lowered(&words, 96, 0).err().expect("refused");
    assert!(e.to_string().contains("32-lane host"), "{e}");
}

/// Several console waves per group, each counting a compare's lanes (a
/// mask read as a number): lane & 63 below 40, in every wave.
#[test]
fn every_wave_of_a_group_counts_its_own_lanes() {
    let body = [
        0x3604_00BF, // v_and_b32 v2, 63, v0: the lane
        0x7D88_04A8, // v_cmp_gt_u32 vcc, 40, v2
        s_bcnt1_i32_b64(4, VCC),
        v_mov_b32(1, 4),
    ];
    for threads in [128, 256, 1024] {
        let shader = build(&program(&body), threads, 0);
        check(&shader, &vec![40; threads as usize]);
    }
}

/// A lane mask a select takes as data (`v_cndmask_b32` of the mask's low
/// dword, its own lane mask all lanes): every lane holds the number, in
/// every wave of the group (plus the mask's lane count, which splits the
/// program).
#[test]
fn a_mask_a_select_takes_as_data_is_its_number_in_every_wave() {
    let body = [
        0x3604_00BF, // v_and_b32 v2, 63, v0: the lane
        0x7D88_048A, // v_cmp_gt_u32 vcc, 10, v2
        0xBE84_046A, // s_mov_b64 s[4:5], vcc
        0xBE8C_04C1, // s_mov_b64 s[12:13], -1
        0xD501_0001,
        0x0030_0880, // v_cndmask_b32_e64 v1, 0, s4, s[12:13]
        s_bcnt1_i32_b64(6, VCC),
        0x4A02_0206, // v_add_nc_u32 v1, s6, v1
    ];
    for threads in [128, 256] {
        let shader = build(&program(&body), threads, 0);
        check(&shader, &vec![0x3FF + 10; threads as usize]);
    }
}

/// Waves of one group going round a loop different numbers of times, each
/// round reading a lane of the upper half: wave w goes round w + 1 times,
/// adding its lane 40's thread id (64 w + 40) each time. Group barriers
/// would pair one wave's halves with another's; the halves of each wave
/// sync on their own.
#[test]
fn waves_of_a_group_loop_on_their_own() {
    let mut body = vec![
        0x2C04_0086, // v_lshrrev_b32 v2, 6, v0: the wave
        v_readfirstlane_b32(4, 2),
        sop2(0x00, 4, 4, 0x81), // s_add_u32 s4, s4, 1
    ];
    // loop:
    body.extend(v_readlane_b32(5, 0, 40));
    body.extend([
        0x4A02_0205,            // v_add_nc_u32 v1, s5, v1
        sop2(0x01, 4, 4, 0x81), // s_sub_u32 s4, s4, 1
        sopc(0x07, 4, 0x80),    // s_cmp_lg_u32 s4, 0
        sopp(0x05, -6),         // s_cbranch_scc1 loop
    ]);
    let shader = build(&program(&body), 256, 0);
    check(
        &shader,
        &(0..256u32)
            .map(|t| (t / 64 + 1) * (t / 64 * 64 + 40))
            .collect::<Vec<_>>(),
    );
}

/// LDS each wave of a group uses alone, without barriers (the console's
/// lanes of one wave need none): each lane reads what the mirror lane of
/// its wave wrote.
#[test]
fn lds_a_wave_uses_alone_in_a_group_of_several() {
    let mut body = ds(0x0D, 16, 0, 0).to_vec(); // ds_write_b32 v16, v0
    body.extend(v_mbcnt_lo(2, EXEC, 0x80));
    body.extend(v_mbcnt_hi(2, EXEC + 1, 0x102)); // v2: the lane
    body.extend([
        0x4C04_04BF, // v_sub_nc_u32 v2, 63, v2: the mirror lane
        0x3606_00FF, // v_and_b32 v3, ~63, v0: the wave's first thread
        !63u32,
        0x4A06_0503, // v_add_nc_u32 v3, v3, v2
        0x3422_0682, // v_lshlrev_b32 v17, 2, v3
    ]);
    body.extend(ds(0x36, 17, 0, 1)); // ds_read_b32 v1, v17
    body.push(0xBF8C_C07F); // s_waitcnt lgkmcnt(0)
    let shader = build(&program(&body), 256, 2);
    check(
        &shader,
        &(0..256u32)
            .map(|t| (t & !63) + 63 - (t & 63))
            .collect::<Vec<_>>(),
    );
}

/// The program's own barrier orders LDS between the waves of a group:
/// each thread reads what the mirror thread of the group wrote, plus its
/// wave's first thread id (`v_readlane_b32`, which makes it split).
#[test]
fn a_run_of_lds_writes_is_read_after_it_by_the_other_half() {
    // Two writes with no read between them (one pair sync before the run),
    // then a read of the mirror lane's second write.
    let mut body = ds(0x0D, 16, 0, 0).to_vec(); // ds_write_b32 v16, v0
    let mut second = ds(0x0D, 16, 0, 0);
    second[0] |= 1024; // offset:1024
    body.extend(second); // ds_write_b32 v16, v0 offset:1024
    body.extend(v_mbcnt_lo(2, EXEC, 0x80));
    body.extend(v_mbcnt_hi(2, EXEC + 1, 0x102)); // v2: the lane
    body.extend([
        0x4C04_04BF, // v_sub_nc_u32 v2, 63, v2: the mirror lane
        0x3606_00FF, // v_and_b32 v3, ~63, v0: the wave's first thread
        !63u32,
        0x4A06_0503, // v_add_nc_u32 v3, v3, v2
        0x3422_0682, // v_lshlrev_b32 v17, 2, v3
    ]);
    let mut read = ds(0x36, 17, 0, 1);
    read[0] |= 1024;
    body.extend(read); // ds_read_b32 v1, v17 offset:1024
    body.push(0xBF8C_C07F); // s_waitcnt lgkmcnt(0)
    let shader = build(&program(&body), 256, 4);
    check(
        &shader,
        &(0..256u32)
            .map(|t| (t & !63) + 63 - (t & 63))
            .collect::<Vec<_>>(),
    );
}

#[test]
fn a_wave_that_skips_a_barrier_to_its_end_does_not_hold_the_others() {
    // Wave 1 of the group has no lane below 64, so its EXEC is empty and
    // it jumps past the barrier to the end; on the console only the waves
    // still running count at a barrier, and the host's group barrier does
    // not wait for the halves that ended either.
    let shader = build(
        &branch(v_cmp_lanes_below(64), &{
            let mut inside = vec![0xBF8A_0000]; // s_barrier
            inside.extend(v_readlane_b32(4, 0, 5)); // s4: lane 5's v0
            inside.push(0x4A02_0004); // v_add_nc_u32 v1, s4, v0
            inside
        }),
        128,
        0,
    );
    check(
        &shader,
        &(0..128u32)
            .map(|t| if t < 64 { t + 5 } else { 0 })
            .collect::<Vec<_>>(),
    );
}

#[test]
fn a_barrier_orders_lds_between_the_waves_of_a_group() {
    let mut body = ds(0x0D, 16, 0, 0).to_vec(); // ds_write_b32 v16, v0
    body.extend([
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0x4C22_20FF, // v_sub_nc_u32 v17, 508, v16
        508,
    ]);
    body.extend(ds(0x36, 17, 0, 1)); // ds_read_b32 v1, v17
    body.push(0xBF8C_C07F); // s_waitcnt lgkmcnt(0)
    body.extend(v_readlane_b32(4, 0, 0));
    body.push(0x4A02_0204); // v_add_nc_u32 v1, s4, v1
    let shader = build(&program(&body), 128, 1);
    check(
        &shader,
        &(0..128u32).map(|t| 127 - t + (t & !63)).collect::<Vec<_>>(),
    );
}

/// A loop each lane leaves after `lane & 3` iterations: both halves go
/// round while any lane of the wave does, and each iteration counts the
/// lanes still in it (48, 32, 16).
#[test]
fn a_loop_runs_while_any_lane_of_the_wave_is_in_it() {
    let body = [
        sop1(0x04, 6, EXEC), // s_mov_b64 s[6:7], exec
        0x3604_0083,         // v_and_b32 v2, 3, v0
        v_mov_b32(3, 0x80),
        // loop:
        0x7D82_0480,                 // v_cmp_lt_u32 vcc, 0, v2
        sop2(0x0F, EXEC, EXEC, VCC), // s_and_b64 exec, exec, vcc
        s_cbranch_execz(4),
        s_bcnt1_i32_b64(4, EXEC),
        0x4A06_0604, // v_add_nc_u32 v3, s4, v3
        0x4A04_04C1, // v_add_nc_u32 v2, -1, v2
        0xBF82_FFF9, // s_branch loop
        // end:
        sop1(0x04, EXEC, 6), // s_mov_b64 exec, s[6:7]
        v_mov_b32(1, 0x103),
    ];
    for threads in [64, 256] {
        let shader = build(&program(&body), threads, 0);
        check(
            &shader,
            &group(threads, |t| [0, 48, 80, 96][t as usize & 3]),
        );
    }
}

/// `f` of each thread id of a group of `threads`.
fn group(threads: u32, f: impl Fn(u32) -> u32) -> Vec<u32> {
    (0..threads).map(f).collect()
}

/// An if/else whose arms both run, each with its lanes: the lanes of one
/// arm keep their value through the other, and each arm counts its lanes.
#[test]
fn both_arms_of_a_divergent_if_run_with_their_own_lanes() {
    let arm = |k: u32| {
        [
            s_bcnt1_i32_b64(4, EXEC),
            sop2(0x00, 4, 4, 0x80 + k),
            v_mov_b32(1, 4),
        ]
    };
    let mut body = vec![
        v_cmp_lanes_below(40),
        s_and_saveexec_b64(6, VCC),
        sop2(0x13, 6, EXEC, 6), // s_xor_b64 s[6:7], exec, s[6:7]
        s_cbranch_execz(3),
    ];
    body.extend(arm(1));
    body.extend([
        sop1(0x25, 6, 6),          // s_or_saveexec_b64 s[6:7], s[6:7]
        sop2(0x13, EXEC, EXEC, 6), // s_xor_b64 exec, exec, s[6:7]
        s_cbranch_execz(3),
    ]);
    body.extend(arm(2));
    body.push(s_or_b64(EXEC, EXEC, 6));
    let shader = build(&program(&body), 64, 0);
    check(&shader, &lanes(|l| if l < 40 { 41 } else { 26 }));
}

/// `v_readfirstlane_b32` with no lane in EXEC reads lane 0.
#[test]
fn reading_the_first_lane_of_none_reads_lane_0() {
    let body = [
        v_mov_b32(2, 0x100),    // v2 = lane
        sop1(0x04, 6, EXEC),    // s_mov_b64 s[6:7], exec
        sop1(0x04, EXEC, 0x80), // s_mov_b64 exec, 0
        v_readfirstlane_b32(4, 2),
        sop1(0x04, EXEC, 6), // s_mov_b64 exec, s[6:7]
        s_bcnt1_i32_b64(5, EXEC),
        sop2(0x00, 4, 4, 5), // s_add_u32 s4, s4, s5
        v_mov_b32(1, 4),
    ];
    let shader = build(&program(&body), 64, 0);
    check(&shader, &lanes(|_| 64));
}

/// DPP without fetch-inactive or bound_ctrl from a lane EXEC switches off
/// leaves the destination alone: the odd lanes read their (off) even
/// neighbour.
#[test]
fn dpp_from_a_switched_off_lane_does_not_write() {
    let body = [
        v_mov_b32(2, 0x100), // v2 = lane
        v_mov_b32(1, 0xC1),  // v1 = -1
        0x3606_0081,         // v_and_b32 v3, 1, v0
        0x7D8A_0680,         // v_cmp_ne_u32 vcc, 0, v3: the odd lanes
        s_and_saveexec_b64(6, VCC),
        0x7E02_02FA,
        0xFF01_1102, // v_mov_b32_dpp v1, v2 row_shr:1
        s_or_b64(EXEC, EXEC, 6),
    ];
    let shader = build(&program(&body), 64, 0);
    check(&shader, &lanes(|_| u32::MAX));
}

/// `v_mbcnt_lo_u32_b32 vdst, src0, src1` (VOP3; `src1` 256 + VGPR or const).
const fn v_mbcnt_lo(vdst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD765_0000 | vdst, src1 << 9 | src0]
}
const fn v_mbcnt_hi(vdst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD766_0000 | vdst, src1 << 9 | src0]
}
/// SOPC `op src0, src1`.
const fn sopc(op: u32, src0: u32, src1: u32) -> u32 {
    0xBF00_0000 | op << 16 | src1 << 8 | src0
}
/// `s_cbranch_*` (SOPP `op`) by `words` (negative: back).
const fn sopp(op: u32, words: i16) -> u32 {
    0xBF80_0000 | op << 16 | words as u16 as u32
}

/// Lane counts below each lane of EXEC (`v_mbcnt` of EXEC) in a branch
/// whose lanes straddle the halves: lane - 21 above lane 20.
#[test]
fn mbcnt_of_exec_counts_lanes_in_both_halves() {
    let mut inside = v_mbcnt_lo(2, EXEC, 0x80).to_vec();
    inside.extend(v_mbcnt_hi(1, EXEC + 1, 0x102));
    let shader = build(&branch(v_cmp_lanes_above(20), &inside), 64, 0);
    check(&shader, &lanes(|l| if l > 20 { l as u32 - 21 } else { 0 }));
}

/// `v_mbcnt` of a compare's mask: the lanes of it below each lane.
#[test]
fn mbcnt_of_a_compare_counts_the_lower_halfs_lanes() {
    let mut body = vec![v_cmp_lanes_below(40)];
    body.extend(v_mbcnt_lo(2, VCC, 0x80));
    body.extend(v_mbcnt_hi(1, VCC + 1, 0x102));
    let shader = build(&program(&body), 64, 0);
    check(&shader, &lanes(|l| l.min(40) as u32));
}

/// The compiler's scalarisation of a lane set: `s_ff1_i32_b64` picks the
/// next lane, `v_readlane_b32` reads it, its bit is cleared, round again
/// while any is left (a scalar loop both halves must leave together).
/// Lanes 0, 4, ..., 60 summed.
#[test]
fn a_loop_over_a_masks_lanes_reads_each_lane() {
    let mut body = vec![
        0x3604_0083,         // v_and_b32 v2, 3, v0
        0x7D84_0480,         // v_cmp_eq_u32 vcc, 0, v2
        sop1(0x04, 6, VCC),  // s_mov_b64 s[6:7], vcc
        sop1(0x03, 4, 0x80), // s_mov_b32 s4, 0
        // loop:
        sop1(0x14, 5, 6), // s_ff1_i32_b64 s5, s[6:7]
    ];
    body.extend([0xD760_000C, 5 << 9 | 0x100]); // v_readlane_b32 s12, v0, s5
    body.extend([
        sop2(0x00, 4, 4, 12),    // s_add_u32 s4, s4, s12
        sop2(0x1F, 12, 0x81, 5), // s_lshl_b64 s[12:13], 1, s5
        sop2(0x15, 6, 6, 12),    // s_andn2_b64 s[6:7], s[6:7], s[12:13]
        sopc(0x13, 6, 0x80),     // s_cmp_lg_u64 s[6:7], 0
        sopp(0x05, -8),          // s_cbranch_scc1 loop
        v_mov_b32(1, 4),
    ]);
    // In wave w the lanes' thread ids are 64 w higher (16 of them).
    for threads in [64, 256] {
        let shader = build(&program(&body), threads, 0);
        check(&shader, &group(threads, |t| t / 64 * 1024 + 480));
    }
}

/// A waterfall loop over a per-lane value: each round takes the first
/// lane's value and the lanes that share it, until no lane is left. With
/// `key` the lane's value, every lane sees its own value and how many lanes
/// share it.
fn waterfall(key: u32) -> Vec<u32> {
    program(&[
        key,                 // v2 = the lane's value
        sop1(0x04, 6, EXEC), // s_mov_b64 s[6:7], exec
        v_readfirstlane_b32(4, 2),
        0x7D84_0404, // v_cmp_eq_u32 vcc, s4, v2
        s_and_saveexec_b64(12, VCC),
        s_bcnt1_i32_b64(5, EXEC),
        sop2(0x1E, 14, 4, 0x88), // s_lshl_b32 s14, s4, 8
        sop2(0x00, 14, 14, 5),   // s_add_u32 s14, s14, s5
        v_mov_b32(1, 14),
        sop2(0x13, EXEC, 12, EXEC), // s_xor_b64 exec, s[12:13], exec
        sopp(0x09, -9),             // s_cbranch_execnz loop
        sop1(0x04, EXEC, 6),        // s_mov_b64 exec, s[6:7]
    ])
}

/// Values of 16 lanes each, two per half.
#[test]
fn a_waterfall_loop_over_runs_of_lanes() {
    for threads in [64, 128] {
        let shader = build(&waterfall(0x2C04_0084), threads, 0); // v_lshrrev_b32 v2, 4, v0
        check(&shader, &group(threads, |t| ((t >> 4) << 8) + 16));
    }
}

/// Values shared by lanes of both halves.
#[test]
fn a_waterfall_loop_over_values_in_both_halves() {
    for threads in [64, 128] {
        let shader = build(&waterfall(0x3604_0083), threads, 0); // v_and_b32 v2, 3, v0
        check(&shader, &group(threads, |t| ((t & 3) << 8) + 16));
    }
}

/// One lane per wave adds the wave's count to an LDS counter, the others
/// take their place from the old value (`v_mbcnt`, `v_readfirstlane`); the
/// one lane is in the upper half.
#[test]
fn one_lane_atomic_shared_through_readfirstlane() {
    let mut inside = vec![v_mov_b32(17, 0x80)]; // v17 = 0 (the counter)
    inside.extend([0x7E06_02FF, 1000]); // v_mov_b32 v3, 1000
    inside.extend(ds(0x0D, 17, 3, 0)); // ds_write_b32 v17, v3
    inside.extend(v_mbcnt_lo(2, EXEC, 0x80));
    inside.extend(v_mbcnt_hi(2, EXEC + 1, 0x102));
    inside.extend([
        sop1(0x04, 20, EXEC), // s_mov_b64 s[20:21], exec
        s_bcnt1_i32_b64(4, 20),
        v_mov_b32(3, 4),
        0x7D84_0480, // v_cmp_eq_u32 vcc, 0, v2
        s_and_saveexec_b64(12, VCC),
    ]);
    inside.extend(ds(0x20, 17, 3, 4)); // ds_add_rtn_u32 v4, v17, v3
    inside.extend([
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        s_or_b64(EXEC, EXEC, 12),
        v_readfirstlane_b32(5, 4),
        0x4A02_0405, // v_add_nc_u32 v1, s5, v2
    ]);
    let shader = build(&branch(v_cmp_lanes_above(36), &inside), 64, 1);
    check(
        &shader,
        &lanes(|l| if l > 36 { 1000 + l as u32 - 37 } else { 0 }),
    );
}

/// A loop both halves leave on a scalar count, with the program's own
/// barrier in it and a lane-dependent branch inside: both halves reach
/// every barrier as often.
#[test]
fn a_barrier_in_a_loop_with_a_divergent_branch_inside() {
    // v1 += bcnt(exec) of the lanes above 40, three times.
    let body = [
        sop1(0x03, 4, 0x83), // s_mov_b32 s4, 3
        // loop:
        v_cmp_lanes_above(40),
        s_and_saveexec_b64(6, VCC),
        s_cbranch_execz(2),
        s_bcnt1_i32_b64(5, EXEC),
        0x4A02_0205, // v_add_nc_u32 v1, s5, v1
        s_or_b64(EXEC, EXEC, 6),
        0xBF8A_0000,            // s_barrier
        sop2(0x01, 4, 4, 0x81), // s_sub_u32 s4, s4, 1
        sopc(0x07, 4, 0x80),    // s_cmp_lg_u32 s4, 0
        sopp(0x05, -10),        // s_cbranch_scc1 loop
    ];
    let shader = build(&program(&body), 64, 0);
    check(&shader, &lanes(|l| if l > 40 { 3 * 23 } else { 0 }));
}

/// An 8x8 group: console lanes are flat thread ids (x first), in both
/// halves; each lane stores its lane (`v_mbcnt` of EXEC) plus the flat id
/// lane 50 reads, at its flat id.
#[test]
fn a_two_dimensional_group_keeps_lanes_in_flat_order() {
    let mut w = vec![
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0x0
        0x3406_0283, // v_lshlrev_b32 v3, 3, v1
        0x4A06_0700, // v_add_nc_u32 v3, v0, v3: the flat id
        0x3420_0682, // v_lshlrev_b32 v16, 2, v3
    ];
    w.extend(v_mbcnt_lo(2, EXEC, 0x80));
    w.extend(v_mbcnt_hi(2, EXEC + 1, 0x102));
    w.extend(v_readlane_b32(4, 3, 50));
    w.extend([
        0x4A02_0404, // v_add_nc_u32 v1, s4, v2
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x8002_0110, // buffer_store_dword v1, v16, s[8:11], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    let shader = build_group(&w, [8, 8, 1], 0);
    check(&shader, &lanes(|l| l as u32 + 50));
}

/// The bisect filter: hash prefixes and equal-length prefix ranges.
#[test]
fn the_split_filter_takes_prefixes_and_ranges() {
    let m = super::split_filter_matches;
    let h = "5c09eb42b7a3c362";
    assert!(m("", h));
    assert!(m("5c09eb42b7a3c362", h));
    assert!(m("e9,5c", h));
    assert!(m("0-7", h));
    assert!(m("50-5f", h));
    assert!(!m("8-f", h));
    assert!(!m("5d", h));
    assert!(!m("none", h));
    assert!(!m("0-ff", h));
}

/// The selection tools apply by hash (`victory translate`) is the one
/// translation applies: a shader left out of the split is lowered as for a
/// 64-lane host, any other one as for the host it runs on.
#[test]
fn a_shader_left_out_of_the_split_is_lowered_as_for_a_64_lane_host() {
    with_host_wave64(false, || {
        assert!(super::with_split_selection(
            super::SPLIT_LEFT_OUT_WRONG[0],
            super::host_wave64
        ));
        assert!(!super::with_split_selection(
            "0123456789abcdef",
            super::host_wave64
        ));
        assert!(!super::host_wave64(), "the selection ends with its call");
    });
}

/// The shape of a classification pass of a group of many console waves:
/// in each round, each wave's first active lane alone (`s_ff1_i32_b64` of
/// EXEC, then `v_cmpx`) writes the wave's count of a compare's lanes (a
/// mask read as a number) to LDS; after the program's barrier, every
/// thread reads one of the counts. Round k of wave w counts the lanes
/// below 3k + 4w.
#[test]
fn each_waves_first_lane_writes_its_waves_count_to_lds() {
    let mut body = wave_counts();
    body.extend([
        0x360E_00BF, // v_and_b32 v7, 63, v0
        0x340E_0E82, // v_lshlrev_b32 v7, 2, v7
    ]);
    body.extend(ds(0x36, 7, 0, 1)); // ds_read_b32 v1, v7
    body.push(0xBF8C_C07F);
    let shader = build(&program(&body), 1024, 1);
    check(&shader, &group(1024, |t| wave_count(t & 63)));
}

/// What [`wave_counts`] leaves at LDS dword `j`.
fn wave_count(j: u32) -> u32 {
    (3 * (j / 16) + 4 * (j % 16)).min(64)
}

/// [`each_waves_first_lane_writes_its_waves_count_to_lds`] up to the
/// barrier after the rounds.
fn wave_counts() -> Vec<u32> {
    let mut body = vec![
        0x2C04_0086, // v_lshrrev_b32 v2, 6, v0: the wave
        v_readfirstlane_b32(7, 2),
        sop1(0x03, 5, 0x80), // s_mov_b32 s5, 0
    ];
    let start = body.len();
    body.extend(v_mbcnt_lo(3, 0xC1, 0x80));
    body.extend(v_mbcnt_hi(3, 0xC1, 0x103)); // v3: the lane
    body.extend([
        sop2(0x1E, 6, 5, 0x81),  // s_lshl_b32 s6, s5, 1
        sop2(0x00, 6, 6, 5),     // s_add_u32 s6, s6, s5
        sop2(0x1E, 20, 7, 0x82), // s_lshl_b32 s20, s7, 2
        sop2(0x00, 6, 6, 20),    // s_add_u32 s6, s6, s20
        0x7D88_0606,             // v_cmp_gt_u32 vcc, s6, v3
        sop1(0x04, 0, VCC),      // s_mov_b64 s[0:1], vcc
        sop1(0x14, 4, EXEC),     // s_ff1_i32_b64 s4, exec
        sop1(0x04, 2, EXEC),     // s_mov_b64 s[2:3], exec
        0x7DA4_0604,             // v_cmpx_eq_u32 exec, s4, v3
        s_bcnt1_i32_b64(21, 0),
        v_mov_b32(5, 21),
        sop2(0x1E, 22, 5, 0x86), // s_lshl_b32 s22, s5, 6
        0x340C_0482,             // v_lshlrev_b32 v6, 2, v2
        0x4A0C_0C16,             // v_add_nc_u32 v6, s22, v6
    ]);
    body.extend(ds(0x0D, 6, 5, 0)); // ds_write_b32 v6, v5
    body.extend([
        sop1(0x04, EXEC, 2),    // s_mov_b64 exec, s[2:3]
        sop2(0x00, 5, 5, 0x81), // s_add_u32 s5, s5, 1
        sopc(0x07, 5, 0x84),    // s_cmp_lg_u32 s5, 4
    ]);
    let back = start as i32 - (body.len() as i32 + 1);
    body.push(sopp(0x05, back as i16)); // s_cbranch_scc1 loop
    body.extend([
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
    ]);
    body
}

/// After [`wave_counts`], thread 0 alone turns round 0's counts into a
/// running sum over the waves in LDS (a scalar loop reading and writing
/// what the other waves wrote, the sum carried by `v_readfirstlane_b32`)
/// and writes the total; after a barrier every thread reads one entry plus
/// the total.
#[test]
fn thread_0_sums_what_every_wave_wrote() {
    let shader = build(&program(&prefix_sum(wave_counts())), 1024, 3);
    check(&shader, &group(1024, |t| summed(t & 63, wave_count)));
}

/// What [`prefix_sum`] leaves a thread reading entry `j`, the counts at
/// each entry being `count`.
fn summed(j: u32, count: impl Fn(u32) -> u32) -> u32 {
    let sum: u32 = (0..16).map(&count).sum();
    let entry = if j < 16 {
        (0..j).map(&count).sum()
    } else {
        count(j)
    };
    entry + sum
}

/// `body` (LDS counts in place, and a barrier), then thread 0's running
/// sum of [`thread_0_sums_what_every_wave_wrote`].
fn prefix_sum(mut body: Vec<u32>) -> Vec<u32> {
    body.extend([
        0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
        s_and_saveexec_b64(12, VCC),
    ]);
    let skip = body.len();
    body.push(0); // s_cbranch_execz end (patched)
    body.extend([
        sop1(0x03, 26, 0x80), // s_mov_b32 s26, 0: the wave
        sop1(0x03, 27, 0x80), // s_mov_b32 s27, 0: the sum before it
    ]);
    let start = body.len();
    body.extend([
        sop2(0x1E, 28, 26, 0x82), // s_lshl_b32 s28, s26, 2
        v_mov_b32(9, 28),
        v_mov_b32(10, 27),
        sop2(0x00, 26, 26, 0x81), // s_add_u32 s26, s26, 1
        sopc(0x08, 26, 0x8F),     // s_cmp_gt_u32 s26, 15
    ]);
    body.extend(ds(0x36, 9, 0, 8)); // ds_read_b32 v8, v9
    body.extend(ds(0x0D, 9, 10, 0)); // ds_write_b32 v9, v10
    body.extend([
        0xBF8C_C17F, // s_waitcnt lgkmcnt(1)
        0x4A10_101B, // v_add_nc_u32 v8, s27, v8
    ]);
    let out = body.len();
    body.push(0); // s_cbranch_scc1 done (patched)
    body.push(v_readfirstlane_b32(27, 8));
    let back = start as i32 - (body.len() as i32 + 1);
    body.push(sopp(0x02, back as i16)); // s_branch loop
    body[out] = sopp(0x05, (body.len() - out - 1) as i16);
    body.extend([0x7E16_02FF, 1024]); // v_mov_b32 v11, 1024
    body.extend(ds(0x0D, 11, 8, 0)); // ds_write_b32 v11, v8
    body[skip] = s_cbranch_execz((body.len() - skip - 1) as u16);
    body.extend([
        s_or_b64(EXEC, EXEC, 12),
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0x360E_00BF, // v_and_b32 v7, 63, v0
        0x340E_0E82, // v_lshlrev_b32 v7, 2, v7
        v_mov_b32(11, 0x80),
    ]);
    body.extend(ds(0x36, 7, 0, 1)); // ds_read_b32 v1, v7
    let mut total = ds(0x36, 11, 0, 12);
    total[0] |= 1024; // ds_read_b32 v12, v11 offset:1024
    body.extend(total);
    body.extend([
        0xBF8C_C07F,
        0x4A02_030C, // v_add_nc_u32 v1, v12, v1
    ]);
    body
}

/// [`thread_0_sums_what_every_wave_wrote`] as the title's tile
/// classification runs it: thread 0 first clears the counts (a run of
/// writes, then a barrier); only lanes at or above 5w of wave w take part,
/// the others leave the program (waves 13-15 entirely, past both later
/// barriers), so a wave's first lane may be in its upper half.
#[test]
fn waves_leaving_early_and_first_lanes_in_the_upper_half_sum_in_lds() {
    let mut body = vec![
        0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
        s_and_saveexec_b64(12, VCC),
        s_cbranch_execz(2 + 2 * 64),
        v_mov_b32(13, 0x80),
        v_mov_b32(14, 0x80),
    ];
    for j in 0..64 {
        let mut w = ds(0x0D, 14, 13, 0);
        w[0] |= 4 * j; // ds_write_b32 v14, v13 offset:4j
        body.extend(w);
    }
    body.extend([
        s_or_b64(EXEC, EXEC, 12),
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0x361E_00BF, // v_and_b32 v15, 63, v0
        0x2C30_0086, // v_lshrrev_b32 v24, 6, v0
        v_readfirstlane_b32(24, 24),
        sop2(0x1E, 23, 24, 0x82), // s_lshl_b32 s23, s24, 2
        sop2(0x00, 23, 23, 24),   // s_add_u32 s23, s23, s24
        0x7D86_1E17,              // v_cmp_le_u32 vcc, s23, v15
        s_and_saveexec_b64(14, VCC),
    ]);
    let skip = body.len();
    body.push(0); // s_cbranch_execz end (patched)
    let rest = prefix_sum(wave_counts());
    body[skip] = s_cbranch_execz(rest.len() as u16);
    body.extend(rest);
    let count = |j: u32| {
        let (w, k) = (j % 16, j / 16);
        if w >= 13 {
            0
        } else {
            (3 * k + 4 * w).min(64).saturating_sub(5 * w)
        }
    };
    let shader = build(&program(&body), 1024, 3);
    check(
        &shader,
        &group(1024, |t| {
            let w = t / 64;
            if w < 13 && t % 64 >= 5 * w {
                summed(t & 63, count)
            } else {
                0
            }
        }),
    );
}

/// LDS reads outside the program's LDS read 0, as on the console, not what
/// the halves last exchanged: after a lane read across the halves, each
/// lane reads one of the 16 dwords below the program's and adds lane 40's
/// thread id.
#[test]
fn lds_reads_out_of_bounds_read_0() {
    let mut body = v_readlane_b32(4, 0, 40).to_vec();
    body.extend([
        0x3604_008F, // v_and_b32 v2, 15, v0
        0x3404_0482, // v_lshlrev_b32 v2, 2, v2
        0x4A04_04FF, // v_add_nc_u32 v2, -64, v2: below the program's LDS
        (-64i32) as u32,
    ]);
    body.extend(ds(0x36, 2, 0, 1)); // ds_read_b32 v1, v2
    body.extend([
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x4A02_0204, // v_add_nc_u32 v1, s4, v1
    ]);
    for threads in [64, 128] {
        let shader = super::probe::with_knobs(None, Some(1 << 14), false, || {
            build(&program(&body), threads, 1)
        });
        check(&shader, &group(threads, |t| t / 64 * 64 + 40));
    }
}

/// Returning LDS atomics outside the program's LDS return 0, as an
/// out-of-bounds console read does, not what a dropped write left in the
/// dword such writes go to: every lane writes 1000 out of bounds, then
/// adds 1000 there, exchanges it, or compares it with 1000, and adds lane
/// 40's thread id (a lane read across the halves).
#[test]
fn lds_atomics_out_of_bounds_return_0() {
    let start = |body: &mut Vec<u32>| {
        body.extend(v_readlane_b32(4, 0, 40));
        body.extend([
            0x3404_0082, // v_lshlrev_b32 v2, 2, v0
            0x4A04_04FF, // v_add_nc_u32 v2, 0x10000, v2: past the program's LDS
            0x1_0000,
            0x7E06_02FF, // v_mov_b32 v3, 1000
            1000,
        ]);
        body.extend(ds(0x0D, 2, 3, 0)); // ds_write_b32 v2, v3
        body.push(0xBF8C_C07F); // s_waitcnt lgkmcnt(0)
    };
    // (opcode, data1): ds_add_rtn_u32, ds_wrxchg_rtn_b32, ds_cmpst_rtn_b32.
    for (op, data1) in [(0x20, 0), (0x2D, 0), (0x30, 3)] {
        let mut body = Vec::new();
        start(&mut body);
        let [w0, w1] = ds(op, 2, 3, 1); // v1 = op(v2, v3[, v3])
        body.extend([w0, w1 | data1 << 16]);
        body.extend([
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x4A02_0204, // v_add_nc_u32 v1, s4, v1
        ]);
        for threads in [64, 128] {
            let l = lowered(&program(&body), threads, 1).expect("lower");
            assert_eq!(
                l.lds_atomics_zero_outside, 1,
                "op {op:#x}, {threads} threads"
            );
            let shader = build(&program(&body), threads, 1);
            check(&shader, &group(threads, |t| t / 64 * 64 + 40));
        }
    }
    // Without a result (ds_add_u32) there is no old value to zero.
    let mut body = Vec::new();
    start(&mut body);
    body.extend(ds(0x00, 2, 3, 0)); // ds_add_u32 v2, v3
    let l = lowered(&program(&body), 64, 1).expect("lower");
    assert_eq!(l.lds_atomics_zero_outside, 0);
}

/// Title memory by address (`resource::MEMORY`) in a wave64 program split
/// into halves: a buffer load and store through a descriptor built around
/// a pointer the program loads, and a global load and store at that
/// pointer plus each lane's offset, lower with `WaveSize(32)` and validate.
#[test]
fn accesses_to_title_memory_by_address_lower_in_halves() {
    let mut words = vec![
        0xF404_0101, // s_load_dwordx2 s[4:5], s[2:3], 0
        0xFA00_0000,
        0xBE86_03C1, // s_mov_b32 s6, -1
        0xBE87_03FF, // s_mov_b32 s7, 0x31014fac
        0x3101_4FAC,
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xE030_1000, // buffer_load_dword v1, v16, s[4:7], 0 offen
        0x8001_0110,
        0xDC30_8000, // global_load_dword v2, v16, s[4:5]
        0x0204_0010,
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x4A02_0501, // v_add_nc_u32 v1, v1, v2
    ];
    // Lane 40's sum, from the upper half.
    words.extend(v_readlane_b32(9, 1, 40));
    words.extend([
        0x4A02_0209, // v_add_nc_u32 v1, s9, v1
        0xE070_1100, // buffer_store_dword v1, v16, s[4:7], 0 offen offset:256
        0x8001_0110,
        0xDC70_8200, // global_store_dword v16, v1, s[4:5] offset:512
        0x0004_0110,
        0xBF81_0000, // s_endpgm
    ]);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 4,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = with_host_wave64(false, || {
        lower::lower(&lower::Parts {
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
    })
    .expect("lower");
    assert!(lowered.root_srvs.contains(&super::SLOT_MEMORY));
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut lowered.dxil.clone())
            .expect("DXIL validation failed");
    }
    let c = Container::parse(&lowered.dxil).unwrap();
    let psv = c.part(FourCC::PSV0).unwrap().data;
    let min = u32::from_le_bytes(psv[20..24].try_into().unwrap());
    let max = u32::from_le_bytes(psv[24..28].try_into().unwrap());
    assert_eq!(
        (min, max),
        (32, 32),
        "the lowering must declare WaveSize(32)"
    );
}
