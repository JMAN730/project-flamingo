//! Demanded bits: for each SSA value, the bits of it that can reach
//! anything the program does. A value whose bits are all shifted out or
//! masked off before they are used (a register still holding a hardware
//! input, read as the other half of a 64-bit shift whose result keeps only
//! the low half's bits, say) demands nothing, and needs no value.
//!
//! The analysis runs backwards over the reads of each instruction: the
//! operations below say which bits of each operand their result's demanded
//! bits come from; any other read demands every bit. The old value of a
//! VGPR an instruction writes passes through to the lanes it leaves out,
//! so it demands what the result does where the write is guarded, unless
//! those lanes never run again (EXEC is not widened before the program
//! ends).

use std::collections::HashMap;

use crate::gfx10::{Fields, Operand, Valu};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};

const ALL: u32 = u32::MAX;
const MAX_ROUNDS: usize = 64;

/// The bits of every value `p` reads that its reads as an operand (not
/// as the old value of a register written) pass on to anything the program
/// does; `guarded(b, i)` tells whether instruction `i` of block `b` leaves
/// some lanes that reach it out (they keep the old value of what it
/// writes). Values missing from the map demand nothing.
// shortcut: lowering gives a guarded write whose old value is missing 0 in
// the lanes it leaves out; so the old value read last is not counted here,
// as if those lanes never used what they keep.
pub(super) fn demanded(p: &Program, guarded: &dyn Fn(usize, usize) -> bool) -> HashMap<Value, u32> {
    // All demanded bits, old values included, and those read as operands.
    let mut d: HashMap<Value, u32> = HashMap::new();
    let mut s: HashMap<Value, u32> = HashMap::new();
    // Per block that ends the program, the instructions after its last
    // write of EXEC: the lanes EXEC leaves out there never run again, so
    // what they keep is never read.
    let dead_after: Vec<usize> = p
        .blocks
        .iter()
        .map(|block| match block.exit {
            Exit::End => block
                .insts
                .iter()
                .rposition(|i| {
                    i.write(Reg::S(EXEC)).is_some() || i.write(Reg::S(EXEC + 1)).is_some()
                })
                .map_or(0, |k| k + 1),
            _ => usize::MAX,
        })
        .collect();
    for _ in 0..MAX_ROUNDS {
        let mut changed = false;
        for (b, block) in p.blocks.iter().enumerate().rev() {
            for (i, inst) in block.insts.iter().enumerate().rev() {
                let out = |r: Reg| inst.write(r).map_or(0, |w| d.get(&w).copied().unwrap_or(0));
                let old_only = |r: Reg| inst.write(r).is_some() && !named(inst, r);
                let operands: Vec<(Value, u32)> = match through(inst, &out) {
                    Some(reads) => reads,
                    None => inst
                        .reads
                        .iter()
                        .filter(|&&(r, _)| !old_only(r))
                        .map(|&(_, v)| (v, ALL))
                        .collect(),
                };
                // The old value of each VGPR written passes to the lanes a
                // guarded write leaves out.
                let olds: Vec<(Value, u32)> = if guarded(b, i) && i < dead_after[b] {
                    inst.writes
                        .iter()
                        .filter_map(|&(r, _)| match (r, inst.read(r)) {
                            (Reg::V(_), Some(old)) => Some((old, out(r))),
                            _ => None,
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                for (v, bits) in operands {
                    changed |= add(&mut d, v, bits);
                    changed |= add(&mut s, v, bits);
                }
                for (v, bits) in olds {
                    changed |= add(&mut d, v, bits);
                }
            }
            for phi in &block.phis {
                let all = d.get(&phi.value).copied().unwrap_or(0);
                let operand = s.get(&phi.value).copied().unwrap_or(0);
                for &a in &phi.args {
                    changed |= add(&mut d, a, all);
                    changed |= add(&mut s, a, operand);
                }
            }
        }
        if !changed {
            return s;
        }
    }
    // Not settled: every value read demands everything.
    p.blocks
        .iter()
        .flat_map(|b| {
            b.insts
                .iter()
                .flat_map(|i| i.reads.iter().map(|r| r.1))
                .chain(b.phis.iter().flat_map(|phi| phi.args.iter().copied()))
        })
        .map(|v| (v, ALL))
        .collect()
}

/// Whether an operand of `inst` names register `r` (a vector instruction's
/// sources; any read of another kind of instruction).
fn named(inst: &Inst, r: Reg) -> bool {
    let Fields::Valu(v) = &inst.fields else {
        return true;
    };
    v.src.iter().any(|o| match (*o, r) {
        (Operand::Vector { reg, dwords }, Reg::V(x))
        | (Operand::Scalar { code: reg, dwords }, Reg::S(x)) => {
            (u16::from(reg)..u16::from(reg) + u16::from(dwords.max(1))).contains(&u16::from(x))
        }
        _ => false,
    })
}

/// Adds `bits` to what `v` demands; whether that changed it.
fn add(d: &mut HashMap<Value, u32>, v: Value, bits: u32) -> bool {
    if bits == 0 {
        return false;
    }
    let e = d.entry(v).or_insert(0);
    let new = *e | bits;
    std::mem::replace(e, new) != new
}

/// The values `inst` reads as operands, with the bits of each that its
/// results' demanded bits (`out` of a written register) come from; `None`
/// for an instruction this does not follow. EXEC and other implicit reads
/// select lanes only.
fn through(inst: &Inst, out: &dyn Fn(Reg) -> u32) -> Option<Vec<(Value, u32)>> {
    let Fields::Valu(v) = &inst.fields else {
        return None;
    };
    if v.sdwa.is_some()
        || v.dpp.is_some()
        || v.abs != 0
        || v.neg != 0
        || v.clamp
        || v.omod != 0
        || v.opsel != 0
    {
        return None;
    }
    let Some(Operand::Vector { reg: dst, .. }) = v.dst else {
        return None;
    };
    let lo = out(Reg::V(dst));
    let hi = out(Reg::V(dst + 1));
    let k = |n: usize| constant(v, n);
    let mut reads = Vec::new();
    let mut op = |n: usize, dword: u8, bits: u32| -> Option<()> {
        match *v.src.get(n)? {
            Operand::Vector { reg, .. } => reads.push((inst.read(Reg::V(reg + dword))?, bits)),
            Operand::Scalar { code, .. } => {
                if let Some(x) = inst.read(Reg::S(code + dword)) {
                    reads.push((x, bits));
                }
            }
            _ => {}
        }
        Some(())
    };
    match inst.name() {
        "v_mov_b32" => op(0, 0, lo)?,
        "v_or_b32" | "v_xor_b32" => {
            op(0, 0, lo)?;
            op(1, 0, lo)?;
        }
        "v_and_b32" => {
            let (a, b) = (k(0).unwrap_or(ALL), k(1).unwrap_or(ALL));
            op(0, 0, lo & b)?;
            op(1, 0, lo & a)?;
        }
        "v_and_or_b32" => {
            let (a, b) = (k(0).unwrap_or(ALL), k(1).unwrap_or(ALL));
            op(0, 0, lo & b)?;
            op(1, 0, lo & a)?;
            op(2, 0, lo)?;
        }
        "v_lshlrev_b32" => {
            let s = k(0)? & 31;
            op(1, 0, lo >> s)?;
        }
        "v_lshrrev_b32" => {
            let s = k(0)? & 31;
            op(1, 0, lo << s)?;
        }
        "v_lshl_or_b32" => {
            let s = k(1)? & 31;
            op(0, 0, lo >> s)?;
            op(2, 0, lo)?;
        }
        "v_alignbit_b32" => {
            // ({src0, src1} >> s), low dword.
            let s = k(2)? & 31;
            op(1, 0, lo << s)?;
            op(0, 0, if s == 0 { 0 } else { lo >> (32 - s) })?;
        }
        "v_cndmask_b32" => {
            op(0, 0, lo)?;
            op(1, 0, lo)?;
            op(2, 0, ALL)?;
            op(2, 1, ALL)?;
        }
        "v_lshlrev_b64" => {
            let s = k(0)? & 63;
            let (l, h) = match s {
                0 => (lo, hi),
                1..=31 => ((lo >> s) | (hi << (32 - s)), hi >> s),
                _ => (hi >> (s - 32), 0),
            };
            op(1, 0, l)?;
            op(1, 1, h)?;
        }
        "v_lshrrev_b64" => {
            let s = k(0)? & 63;
            let (l, h) = match s {
                0 => (lo, hi),
                1..=31 => (lo << s, (lo >> (32 - s)) | (hi << s)),
                _ => (0, lo << (s - 32)),
            };
            op(1, 0, l)?;
            op(1, 1, h)?;
        }
        _ => return None,
    }
    Some(reads)
}

/// Operand `n` of `v` as a constant.
fn constant(v: &Valu, n: usize) -> Option<u32> {
    match *v.src.get(n)? {
        Operand::Const(c) => crate::resource::inline(c),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(words: &[u32]) -> Program {
        let code: Vec<u8> = words
            .iter()
            .chain(&[0xBF81_0000]) // s_endpgm
            .flat_map(|w| w.to_le_bytes())
            .collect();
        crate::ir::build(&code, false, 32).unwrap()
    }

    /// The input a value of the program reads first in `reg`.
    fn input(p: &Program, reg: Reg) -> Value {
        p.inputs.iter().find(|i| i.0 == reg).unwrap().1
    }

    #[test]
    fn the_high_half_of_a_shift_that_drops_it_demands_nothing() {
        let p = program(&[
            0xD6FF_0000,
            0x0002_0097, // v_lshlrev_b64 v[0:1], 23, v[0:1]
            0xD6FF_0000,
            0x0002_0091, // v_lshlrev_b64 v[0:1], 17, v[0:1]
            0x7E04_0301, // v_mov_b32 v2, v1 (bits 40.. of the input: the low half's)
            0x7E06_0300, // v_mov_b32 v3, v0
            0xE070_1000,
            0x8000_0202, // buffer_store_dword v2, v2, s[0:3], 0 offen
            0xE070_1000,
            0x8000_0303, // buffer_store_dword v3, v3, s[0:3], 0 offen
        ]);
        let d = demanded(&p, &|_, _| false);
        assert_eq!(d.get(&input(&p, Reg::V(1))).copied().unwrap_or(0), 0);
        assert_eq!(d.get(&input(&p, Reg::V(0))).copied(), Some(0x00FF_FFFF));
    }

    #[test]
    fn an_old_value_read_last_by_a_write_demands_nothing() {
        let p = program(&[
            0x7E02_0280, // v_mov_b32 v1, 0
            0x7E00_0302, // v_mov_b32 v0, v2
            0xE070_1000,
            0x8000_0000, // buffer_store_dword v0, v0, s[0:3], 0 offen
        ]);
        let v0 = input(&p, Reg::V(0));
        let d = demanded(&p, &|_, _| false);
        assert_eq!(d.get(&v0).copied().unwrap_or(0), 0);
        // Read last as the old value of a guarded write, it still demands
        // nothing (lowering gives such lanes 0); what it is copied to does.
        let d = demanded(&p, &|_, _| true);
        assert_eq!(d.get(&v0).copied().unwrap_or(0), 0);
    }
}
