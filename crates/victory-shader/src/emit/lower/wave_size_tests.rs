use super::super::{emit, emit_with_wave32};
use crate::native::{ComputeIo, RootParam};
use crate::{Container, FourCC, HwStage, NativeProgram, ResourceType, RootType, UsedResource};

const S_ENDPGM: u32 = 0xBF81_0000;

/// Assembles a wave64 compute program with one raw UAV table at `s0`,
/// lowers it, validates DXIL when `dxil.dll` is present, and returns the
/// lowered `HostShader` along with its `PSV0` `(min, max)` wave size.
fn compile_wave64_cs(
    words: &[u32],
    threads: [u32; 3],
    rsrc2_extra: u32,
) -> (super::super::HostShader, (u32, u32)) {
    compile_wave64_cs_mode(words, threads, rsrc2_extra, true)
}

fn compile_wave64_cs_mode(
    words: &[u32],
    threads: [u32; 3],
    rsrc2_extra: u32,
    allow_wave32: bool,
) -> (super::super::HostShader, (u32, u32)) {
    compile_wave64_cs_threads(words, threads, rsrc2_extra, allow_wave32, threads)
}

fn compile_wave64_cs_threads(
    words: &[u32],
    threads: [u32; 3],
    rsrc2_extra: u32,
    allow_wave32: bool,
    expected_threads: [u32; 3],
) -> (super::super::HostShader, (u32, u32)) {
    let shader = try_compile_wave64_cs(words, threads, rsrc2_extra, allow_wave32).unwrap();
    check_wave64_cs(shader, expected_threads)
}

fn try_compile_wave64_cs(
    words: &[u32],
    threads: [u32; 3],
    rsrc2_extra: u32,
    allow_wave32: bool,
) -> Result<super::super::HostShader, crate::ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
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
        io: crate::native::StageIo::Cs(ComputeIo {
            rsrc2: (2 << 1) | rsrc2_extra,
            threads,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let container = crate::dxbc::build(&[]);
    let c = Container::parse(&container).unwrap();
    let p = crate::ir::build(&native.code, true, native.vgprs).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    if allow_wave32 {
        emit(&c, &native, &p, &st, &io, &res)
    } else {
        emit_with_wave32(&c, &native, &p, &st, &io, &res, false)
    }
}

fn check_wave64_cs(
    shader: super::super::HostShader,
    expected_threads: [u32; 3],
) -> (super::super::HostShader, (u32, u32)) {
    #[cfg(windows)]
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let emitted = Container::parse(&shader.dxil).unwrap();
    let sfi0 = u64::from_le_bytes(emitted.part(FourCC::SFI0).unwrap().data.try_into().unwrap());
    assert_ne!(sfi0 & 0x4000, 0, "expected WAVE_OPS feature bit");
    let psv = emitted.part(FourCC::PSV0).unwrap().data;
    let min_wave = u32::from_le_bytes(psv[20..24].try_into().unwrap());
    let max_wave = u32::from_le_bytes(psv[24..28].try_into().unwrap());
    let psv_threads = [
        u32::from_le_bytes(psv[40..44].try_into().unwrap()),
        u32::from_le_bytes(psv[44..48].try_into().unwrap()),
        u32::from_le_bytes(psv[48..52].try_into().unwrap()),
    ];
    assert_eq!(psv_threads, expected_threads);
    (shader, (min_wave, max_wave))
}

/// Wave64 compute program that computes `v7` from `v0` (or `v1, v0`),
/// runs `extra_before`, executes a whole-wave sum reduction of `v7` into
/// `v4`, and stores `v4` to `out[flat_tid]`.
fn wave64_sum_cs_words(setup_2d: bool, extra_before: &[u32], extra_after: &[u32]) -> Vec<u32> {
    let mut words = vec![
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null
    ];
    if setup_2d {
        words.extend([
            0x340E_0283, // v_lshlrev_b32_e32 v7, 3, v1
            0x4A0E_0107, // v_add_nc_u32_e32 v7, v7, v0
            0x340C_0E82, // v_lshlrev_b32_e32 v6, 2, v7
            0x4A0E_0E81, // v_add_nc_u32_e32 v7, 1, v7
        ]);
    } else {
        words.extend([
            0x340C_0082, // v_lshlrev_b32_e32 v6, 2, v0
            0x4A0E_0081, // v_add_nc_u32_e32 v7, 1, v0
        ]);
    }
    words.extend_from_slice(extra_before);
    words.extend([
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
    ]);
    words.extend_from_slice(extra_after);
    words.push(S_ENDPGM);
    words
}

#[cfg(windows)]
fn run_cs_uav(shader: &super::super::HostShader, dwords: u32) -> Vec<(&'static str, Vec<u32>)> {
    use super::super::host::{Dispatch, Host, Root, View};
    use windows::Win32::Graphics::Direct3D12::*;
    let _one = super::super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let mut opts1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        // SAFETY: querying D3D12_FEATURE_D3D12_OPTIONS1 into a zeroed struct.
        let supported = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut opts1).cast(),
                size_of_val(&opts1) as u32,
            )
        }
        .is_ok()
            && opts1.WaveOps.as_bool()
            && (opts1.WaveLaneCountMin..=opts1.WaveLaneCountMax).contains(&32);
        if !supported {
            // WARP reports WaveLaneCountMin/Max = 4, so fixed WaveSize(32)
            // pipelines are only executable on hardware Wave32 devices.
            eprintln!(
                "{label}: skipped WaveSize(32) execution (device supports {}..{})",
                opts1.WaveLaneCountMin, opts1.WaveLaneCountMax
            );
            continue;
        }
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let res = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(dwords)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, res[0].clone()));
    }
    out
}

#[test]
fn doubled_lane_reads_broadcast_registers_from_both_halves_without_scratch_lds() {
    // Each lane writes tid + 1 near the end of the 32 KiB LDS allocation
    // and reads its opposite half. Sum registers from lanes 31 and 63,
    // then add the cross-half LDS result. Every output tests both paths.
    let mut words = vec![
        0xBE81_0380,
        0xF408_0200,
        0xFA00_0000, // load UAV descriptor
        0x340C_0082, // v_lshlrev_b32 v6, 2, v0 (byte address)
        0x4A0E_0081, // v_add_nc_u32 v7, 1, v0
        0xD760_0002,
        0x0001_3F07, // v_readlane_b32 s2, v7, 31
        0xD760_0003,
        0x0001_7F07, // v_readlane_b32 s3, v7, 63
        0x7E0A_0203, // v_mov_b32 v5, s3
        0x4A08_0A02, // v_add_nc_u32 v4, s2, v5
        0xD834_7F00,
        0x0000_0706, // ds_write_b32 v6, v7 offset:32512
        0x3A00_00A0, // v_xor_b32 v0, 32, v0
        0x3400_0082, // v_lshlrev_b32 v0, 2, v0
        0xD8D8_7F00,
        0x0500_0000, // ds_read_b32 v5, v0 offset:32512
        0x4A08_0905, // v_add_nc_u32 v4, v5, v4
        0xE070_1000,
        0x8002_0406, // buffer_store_dword v4, v6
        S_ENDPGM,
    ];
    let full_lds = 64 << 15;
    let group = [64, 1, 1];
    super::TEST_DOUBLE_LANES.set(Some(false));
    let (_, off) = compile_wave64_cs_mode(&words, group, full_lds, true);
    assert_eq!(off, (64, 64));
    super::TEST_DOUBLE_LANES.set(Some(true));
    let (_, native) = compile_wave64_cs_mode(&words, group, full_lds, false);
    assert_eq!(native, (64, 64));
    let (shader, doubled) = compile_wave64_cs_threads(&words, group, full_lds, true, [32, 1, 1]);
    assert_eq!(doubled, (32, 32));
    #[cfg(windows)]
    for (host, result) in run_cs_uav(&shader, 64) {
        assert_eq!(
            result,
            (0..64).map(|lane| 97 + (lane ^ 32)).collect::<Vec<_>>(),
            "{host}"
        );
        eprintln!("{host}: read back all 64 logical lanes");
    }
    #[cfg(not(windows))]
    let _ = shader;
    // Inter-thread state is outside the straight-line indexed-read proof.
    words.insert(words.len() - 1, 0xBF8A_0000); // s_barrier
    let (_, with_barrier) = compile_wave64_cs_mode(&words, group, full_lds, true);
    assert_eq!(with_barrier, (64, 64));
    super::TEST_DOUBLE_LANES.set(None);
}

#[test]
fn split_console_flow_preserves_lds_instruction_order() {
    // Each lane stores its id in LDS, reads from the opposite half,
    // then publishes the result through its own UAV element.
    let words = [
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0x3A0A_00A0, // v_xor_b32 v5, 32, v0
        0x340C_0A82, // v_lshlrev_b32 v6, 2, v5
        0xD834_0000,
        0x0000_0004, // ds_write_b32 v4, v0
        0xD8D8_0000,
        0x0200_0006, // ds_read_b32 v2, v6
        0xD760_0002,
        0x0001_5100, // v_readlane_b32 s2, v0, 40
        0x4A04_0402, // v_add_nc_u32 v2, s2, v2
        0xE070_1000,
        0x8002_0204, // buffer_store_dword v2, v4, s[8:11], 0 offen
        S_ENDPGM,
    ];
    let group = [64, 1, 1];
    let lds_512_bytes = 1 << 15;
    super::super::TEST_CONSOLE_FLOW.set(Some(false));
    let (_, native) = compile_wave64_cs(&words, group, lds_512_bytes);
    assert_eq!(native, (64, 64));

    super::super::TEST_CONSOLE_FLOW.set(Some(true));
    let (shader, split) = compile_wave64_cs(&words, group, lds_512_bytes);
    let _ = &shader;
    assert_eq!(split, (32, 32));
    #[cfg(windows)]
    for (host, output) in run_cs_uav(&shader, 64) {
        assert_eq!(
            output,
            (0..64).map(|lane| (lane ^ 32) + 40).collect::<Vec<_>>(),
            "{host}"
        );
    }

    let (_, native_only) = compile_wave64_cs_mode(&words, group, lds_512_bytes, false);
    assert_eq!(native_only, (64, 64));
    super::super::TEST_CONSOLE_FLOW.set(None);
}

#[test]
fn console_flow_broadcasts_the_first_lane_to_both_halves() {
    let words = [
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0x7E04_0500, // v_readfirstlane_b32 s2, v0
        0x7E04_0281, // v_mov_b32 v2, 1
        0x4A04_0402, // v_add_nc_u32 v2, s2, v2
        0xE070_1000,
        0x8002_0204, // buffer_store_dword v2, v4, s[8:11], 0 offen
        S_ENDPGM,
    ];
    super::super::TEST_CONSOLE_FLOW.set(Some(true));
    let (shader, bounds) = compile_wave64_cs(&words, [64, 1, 1], 0);
    super::super::TEST_CONSOLE_FLOW.set(None);
    let _ = &shader;
    assert_eq!(bounds, (32, 32));
    #[cfg(windows)]
    for (host, values) in run_cs_uav(&shader, 64) {
        assert_eq!(values, vec![1; 64], "{host}");
    }
}

/// A wave64 program: a first-lane read forces the console flow, then
/// `branch` runs between the stores of 1 and 2 into `out[tid]`.
fn console_flow_branch_words(branch: &[u32]) -> Vec<u32> {
    let mut words = vec![
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0x7E04_0500, // v_readfirstlane_b32 s2, v0
        0x7E04_0281, // v_mov_b32 v2, 1
    ];
    words.extend_from_slice(branch);
    words.extend([
        0xE070_1000,
        0x8002_0204, // buffer_store_dword v2, v4, s[8:11], 0 offen
        S_ENDPGM,
    ]);
    words
}

#[test]
fn console_flow_branches_on_a_numeric_vcc() {
    let words = console_flow_branch_words(&[
        0xBEEA_0480, // s_mov_b64 vcc, 0
        0xBF86_0001, // s_cbranch_vccz +1
        0x7E04_0282, // v_mov_b32 v2, 2
    ]);
    super::super::TEST_CONSOLE_FLOW.set(Some(true));
    let (_, bounds) = compile_wave64_cs(&words, [64, 1, 1], 0);
    super::super::TEST_CONSOLE_FLOW.set(None);
    assert_eq!(bounds, (32, 32));
}

#[test]
fn console_flow_scc_join_of_mask_operations_is_resolved_or_refused() {
    let words = console_flow_branch_words(&[
        0x7D88_0088, // v_cmp_gt_u32_e32 vcc, 8, v0
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0002, // s_cbranch_scc1 +2
        0x8782_7E6A, // s_and_b64 s[2:3], vcc, exec
        0xBF82_0001, // s_branch +1
        0x8784_7E6A, // s_and_b64 s[4:5], vcc, exec
        0xBF85_0001, // s_cbranch_scc1 +1
        0x7E04_0282, // v_mov_b32 v2, 2
    ]);
    super::super::TEST_CONSOLE_FLOW.set(Some(false));
    let off = try_compile_wave64_cs(&words, [64, 1, 1], 0, true);
    super::super::TEST_CONSOLE_FLOW.set(Some(true));
    let on = try_compile_wave64_cs(&words, [64, 1, 1], 0, true);
    super::super::TEST_CONSOLE_FLOW.set(None);
    eprintln!("off: {:?}", off.as_ref().err());
    eprintln!("on: {:?}", on.as_ref().err());
    assert!(on.is_err());
}

#[test]
fn console_flow_exec_hi_only_write_terminates() {
    // A mask whose only change is its high dword is keyed by the
    // unchanged low dword: reading it must stop, not recurse.
    let words = console_flow_branch_words(&[
        0xBEFF_0380, // s_mov_b32 exec_hi, 0
        0xBF86_0001, // s_cbranch_vccz +1
        0x7E04_0282, // v_mov_b32 v2, 2
    ]);
    super::super::TEST_CONSOLE_FLOW.set(Some(true));
    let result = try_compile_wave64_cs(&words, [64, 1, 1], 0, true);
    super::super::TEST_CONSOLE_FLOW.set(None);
    let _ = result;
}

#[test]
fn console_flow_keeps_the_native_whole_wave_reduction() {
    let words = wave64_sum_cs_words(false, &[], &[]);
    super::super::TEST_CONSOLE_FLOW.set(Some(true));
    let (_, wave_bounds) = compile_wave64_cs(&words, [64, 1, 1], 0);
    super::super::TEST_CONSOLE_FLOW.set(None);
    assert_eq!(wave_bounds, (64, 64));
}

#[test]
fn small_wave64_whole_wave_reduction_emits_wave32_and_runs() {
    for (threads, setup_2d, rsrc2_extra, expected_sum) in [
        ([32, 1, 1], false, 0, 32 * 33 / 2),
        ([8, 4, 1], true, 1 << 11, 32 * 33 / 2),
    ] {
        let words = wave64_sum_cs_words(setup_2d, &[], &[]);
        let (shader, wave_bounds) = compile_wave64_cs(&words, threads, rsrc2_extra);
        let _ = (&shader, expected_sum);
        assert_eq!(wave_bounds, (32, 32), "threads={threads:?}");
        #[cfg(windows)]
        {
            let count = threads.iter().product::<u32>();
            for (label, data) in run_cs_uav(&shader, count) {
                assert_eq!(
                    data,
                    vec![expected_sum; count as usize],
                    "{label} threads={threads:?}"
                );
            }
        }
    }

    // Divergent control flow (`if (tid < 8)`) around the whole-wave
    // reduction still lowers with WaveSize(32) and computes 1+...+8 = 36.
    let body = wave64_sum_cs_words(false, &[], &[]);
    let sec_start = body.iter().position(|&w| w == 0xBE80_25C1).unwrap();
    let section_and_store = &body[sec_start..body.len() - 1];
    let mut divergent = body[..sec_start].to_vec();
    divergent.extend([
        0x7D88_0088,                                  // v_cmp_gt_u32_e32 vcc, 8, v0
        0xBE84_246A,                                  // s_and_saveexec_b64 s[4:5], vcc
        0xBF88_0000 | section_and_store.len() as u32, // s_cbranch_execz -> restore
    ]);
    divergent.extend_from_slice(section_and_store);
    divergent.extend([
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        S_ENDPGM,
    ]);
    let (shader, wave_bounds) = compile_wave64_cs(&divergent, [32, 1, 1], 0);
    let _ = &shader;
    assert_eq!(wave_bounds, (32, 32));
    #[cfg(windows)]
    for (label, data) in run_cs_uav(&shader, 32) {
        let want: Vec<u32> = (0..32).map(|i| if i < 8 { 36 } else { 0 }).collect();
        assert_eq!(data, want, "{label} divergent");
    }
}

#[test]
fn large_group_and_lds_and_disallowed_wave64_reductions_keep_wave64() {
    let base = wave64_sum_cs_words(false, &[], &[]);

    // Non-32-thread groups ([1, 1, 1], [16, 1, 1], [33, 1, 1], [64, 1, 1]) keep WaveSize(64).
    for threads in [[1, 1, 1], [16, 1, 1], [33, 1, 1], [64, 1, 1]] {
        let (_, bounds) = compile_wave64_cs(&base, threads, 0);
        assert_eq!(bounds, (64, 64), "threads={threads:?}");
    }

    // [32, 1, 1] with LDS keeps WaveSize(64).
    let (_, bounds) = compile_wave64_cs(&base, [32, 1, 1], 1 << 15);
    assert_eq!(bounds, (64, 64));

    // [32, 1, 1] with independent cross-lane exchange outside reduction
    // sections keeps WaveSize(64).
    for outside_op in [
        &[0x4A0E_0EFA, 0xFF01_6107][..], // v_add_nc_u32_dpp v7, v7, v7 row_xmask:1
        &[0xD778_0007, 0x0305_8307][..], // v_permlanex16_b32 v7, v7, -1, -1
        &[0xD760_0004, 0x0001_0107][..], // v_readlane_b32 s4, v7, 0
    ] {
        let words = wave64_sum_cs_words(false, outside_op, &[]);
        let (_, bounds) = compile_wave64_cs(&words, [32, 1, 1], 0);
        assert_eq!(bounds, (64, 64), "outside_op={outside_op:#x?}");
    }

    // [32, 1, 1] with a held constant mask (`masks_held`) keeps WaveSize(64).
    let held = wave64_sum_cs_words(
        false,
        &[0xBE86_24FF, 0x0000_1234], // s_and_saveexec_b64 s[6:7], 0x1234
        &[0xBEFE_0406],              // s_mov_b64 exec, s[6:7]
    );
    let (_, bounds) = compile_wave64_cs(&held, [32, 1, 1], 0);
    assert_eq!(bounds, (64, 64));

    // On adapters that only support 64-lane waves (`allow_wave32 = false`),
    // an otherwise eligible <=32-thread reduction keeps WaveSize(64).
    let (_, bounds) = compile_wave64_cs_mode(&base, [32, 1, 1], 0, false);
    assert_eq!(bounds, (64, 64));
}

#[test]
fn sparse_mask_32_thread_wave64_float_reduction_emits_wave32_and_runs() {
    let words = [
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0100,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], null
        0x7E0A_02FF,
        0x3F80_0000, // v_mov_b32 v5, 1.0
        0x3606_0081, // v_and_b32 v3, 1, v0
        0x7D84_0680, // v_cmp_eq_u32 vcc, 0, v3
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBE8A_25C1, // s_or_saveexec_b64 s[10:11], -1
        0xD501_0001,
        0x002A_0A80, // v_cndmask_b32 v1, 0, v5, s[10:11]
        0x0602_02E9,
        0xDE54_C101,
        0x0602_02E9,
        0x9360_1201,
        0x0602_02E9,
        0x0009_2401,
        0x0602_02FA,
        0xFF09_4001,
        0xD778_0002,
        0x0201_0101,
        0x0602_0302,
        0xD760_0002,
        0x0001_4101, // v_readlane_b32 s2, v1, 32
        0x0602_0202,
        0xD760_0003,
        0x0001_0101, // v_readlane_b32 s3, v1, 0
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
        0x7E04_0203, // v_mov_b32 v2, s3
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        S_ENDPGM,
    ];
    let (shader, wave_bounds) = compile_wave64_cs(&words, [32, 1, 1], 0);
    let _ = &shader;
    assert_eq!(wave_bounds, (32, 32));
    let (_, bounds_33) = compile_wave64_cs(&words, [33, 1, 1], 0);
    assert_eq!(
        bounds_33,
        (64, 64),
        "33-thread variant must stay out of the Wave32 reduction path"
    );
    #[cfg(windows)]
    for (label, data) in run_cs_uav(&shader, 32) {
        let expected: Vec<u32> = (0..32)
            .map(|i| if i % 2 == 0 { 16.0f32.to_bits() } else { 0 })
            .collect();
        assert_eq!(data, expected, "{label} sparse-mask float reduction");
    }
}
