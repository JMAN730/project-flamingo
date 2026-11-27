//! End-to-end tests of the program's constant table (the console compiler
//! appends it to the code and reads it through `s_getpc_b64`) read at an
//! address that control flow joins: synthetic wave32 GFX10 compute
//! programs lowered to DXIL, validated by `dxil.dll` when present, and run
//! on the host GPU and WARP against the console's result, lane by lane.

use super::lane_mask_tests::{build, check};

/// The constant table: eight dwords after the code.
const TABLE: [u32; 8] = [11, 22, 33, 44, 55, 66, 77, 88];

/// A program whose lanes read dword `lane & 3` of the table, at an address
/// joined from two paths a uniform branch picks (`s_cmp_eq_u32 s0, 0`, or
/// `s_cmp_lg_u32` with `taken_when_zero` false): the taken path adds 16
/// bytes to its index, the other adds none. Each path computes its own
/// address from the program counter, so the join is of two sums with
/// different computed terms.
fn joined(taken_when_zero: bool) -> Vec<u32> {
    let mut w = vec![
        0xBE81_0380, // s_mov_b32 s1, 0: s[0:1] is the table of views
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0x0
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    let pc = 4 * w.len() as u32 + 4;
    w.extend_from_slice(&[
        0xBE84_1F00, // s_getpc_b64 s[4:5]
        0x8004_FF04,
        0,           // s_add_u32 s4, s4, lit (the table's offset from the PC)
        0x8205_8005, // s_addc_u32 s5, s5, 0
        0x3604_0083, // v_and_b32 v2, 3, v0
        0x3404_0482, // v_lshlrev_b32 v2, 2, v2
        if taken_when_zero {
            0xBF06_8000 // s_cmp_eq_u32 s0, 0
        } else {
            0xBF07_8000 // s_cmp_lg_u32 s0, 0
        },
        0xBF85_0005, // s_cbranch_scc1 taken
        // Not taken: the table plus the index.
        0xD70F_6A03,
        0x0000_0902, // v_add_co_u32 v3, vcc_lo, v2, s4
        0xD528_6A04,
        0x01A8_0A80, // v_add_co_ci_u32 v4, vcc_lo, 0, s5, vcc_lo
        0xBF82_0005, // s_branch join
        // Taken: 16 bytes further.
        0x4A0A_0490, // v_add_nc_u32 v5, 16, v2
        0xD70F_6A03,
        0x0000_0905, // v_add_co_u32 v3, vcc_lo, v5, s4
        0xD528_6A04,
        0x01A8_0A80, // v_add_co_ci_u32 v4, vcc_lo, 0, s5, vcc_lo
        // Join.
        0xDC30_8000,
        0x017D_0003, // global_load_dword v1, v[3:4], off
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x8002_0110, // buffer_store_dword v1, v16, s[8:11], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    let table = 4 * w.len() as u32;
    let lit = w.iter().position(|&x| x == 0x8004_FF04).unwrap() + 1;
    w[lit] = table - pc;
    w.extend_from_slice(&TABLE);
    w
}

/// The table's address is 0 in the test: the taken path's dwords 4-7 when
/// the branch is taken on 0, dwords 0-3 when it is not.
#[test]
fn a_code_table_address_joined_from_two_paths_reads_each_paths_entry() {
    for (taken_when_zero, skip) in [(true, 4), (false, 0)] {
        let shader = build(&joined(taken_when_zero));
        check(&shader, &std::array::from_fn(|l| TABLE[skip + (l & 3)]));
    }
}
