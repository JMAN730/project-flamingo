//! A console wave64 compute program on a host whose waves hold at most 32
//! lanes: each console wave runs as two host waves, its halves (console
//! lanes 0-31 and 32-63), under `WaveSize(32)`.
//!
//! The halves must see what one console wave sees, so the program runs as
//! the console runs it rather than one lane at a time: both halves take
//! every branch the console's wave takes (decided from the whole wave's
//! EXEC, VCC or SCC), and EXEC and every other lane mask are kept exactly,
//! as each lane's bit, including lanes that are switched off. A vector
//! instruction is guarded by its lane's EXEC bit (an off lane keeps its old
//! VGPR value and has no effect), so SSA joins hold the console's register
//! contents. What one half needs of the other (a mask as a number, a lane
//! read, a reduction's other half, whether any lane takes a branch) goes
//! through group-shared memory, which is sound because both halves of a
//! console wave always run the same blocks.
//!
//! A group of one console wave (64 threads) is the two halves, so a
//! thread-group barrier is their pairwise sync. In a group of several
//! console waves, the waves take their own branches, so group barriers
//! would count differently in different waves: there each pair of halves
//! syncs on its own, through a counter per half in group-shared memory
//! that it bumps and its partner waits for (see [`Lower::pair_exchange`]).
//! The console program's own `s_barrier`s stay group barriers (every
//! console wave reaches them as often). The console compiler leaves out
//! barriers between the lanes of one wave sharing LDS; here a pair sync goes
//! between LDS accesses that may depend on each other.

use std::collections::{HashMap, HashSet};

use super::super::bitcode::{Attr, Bin, Cast, Rmw, Term, V, pred};
use super::super::wave::Op;
use super::{Lower, Val, Why};
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Exit, Inst, Reg, VCC, Value};

/// Dwords each half can pass in one exchange.
const SLOTS: u32 = 4;

/// Thread-group barrier with a group-shared memory fence.
const SYNC: u32 = 1 | 8;

/// Group-shared memory fence alone.
const FENCE: u32 = 8;

/// Dwords of a console wave's exchange area when the group holds several
/// waves: two buffers (exchanges alternate between them) of both halves'
/// slots, then each half's count of exchanges done.
const STRIDE: u32 = 2 * 2 * SLOTS + 2;

/// Polls of the partner's count after which a half stops waiting. Both
/// halves of a group are resident and always reach the same exchanges, so
/// the wait ends unless this lowering is at fault. The bound must not be
/// reached by a partner that is merely slow (with one lane polling, 2^16
/// polls is well under a millisecond, which a busy GPU can exceed): a half
/// that gives up reads stale slots and may then take other branches than
/// its partner. This bound is only reached after far longer than the
/// host's GPU timeout, so a lowering fault shows as a hang, not as data.
const SPIN_LIMIT: u32 = 1 << 30;

/// Group-shared memory a host thread group may use.
const HOST_LDS_BYTES: u32 = 32 * 1024;

#[derive(Default)]
pub(super) struct Split {
    /// The console lane's exact bit of each lane mask, by its low dword.
    exact: HashMap<Value, V>,
    /// The low dword of each mask's high dword.
    high: HashMap<Value, Value>,
    /// Mask joins to fill: the phi, IR block, phi index.
    pending: Vec<(V, usize, usize)>,
    /// Values some instruction or join reads.
    read: HashSet<Value>,
    /// Group-shared `i32`s the halves exchange through.
    xchg: Option<V>,
    /// Made in the entry block: host lane, console lane, half (0 or 1).
    lanes: Option<(V, V, V)>,
    /// Made in the entry block: the console wave's index in the group.
    wave: Option<V>,
    /// Console (IR) block whose last LDS access, with no sync since, was a
    /// plain read (`true`) or a plain write (`false`).
    lds_last: Option<(usize, bool)>,
    /// The program's LDS in dwords; the host's has one more, which takes
    /// the accesses past it.
    pub(super) lds_bound: u32,
    /// Values that may hold lane masks: what structurisation tracks, EXEC,
    /// what a `saveexec` saves, and joins of them.
    masks: HashSet<Value>,
    /// DXIL blocks only some lanes run (guarded stores): no exchange there.
    pub(super) guarded: HashSet<usize>,
    /// Console waves per thread group.
    waves: u32,
    /// Values (mask dwords) some reader may take as a number. With several
    /// waves a mask's numbers are made where it is written (a pair sync
    /// splits the block, which code reading a value cannot allow).
    numbers: HashSet<Value>,
    /// Mask joins of the block being lowered whose numbers are made after
    /// its joins: low dword, exact bit.
    phi_numbers: Vec<(Value, V)>,
}

impl Split {
    pub(super) fn new(
        p: &crate::ir::Program,
        st: &crate::structure::Structured,
        waves: u32,
    ) -> Self {
        let mut read = HashSet::new();
        let mut masks: HashSet<Value> = st.masks.keys().copied().collect();
        masks.extend(p.inputs.iter().filter(|i| i.0 == Reg::S(EXEC)).map(|i| i.1));
        for b in &p.blocks {
            read.extend(b.phis.iter().flat_map(|phi| phi.args.iter().copied()));
            for i in &b.insts {
                read.extend(i.reads.iter().map(|r| r.1));
                if let Some(e) = i.write(Reg::S(EXEC)) {
                    masks.insert(e);
                    if let Fields::Sop {
                        dst: Some(Operand::Scalar { code, .. }),
                        ..
                    } = i.fields
                        && let Some(d) = i.write(Reg::S(code))
                    {
                        masks.insert(d);
                    }
                }
            }
        }
        loop {
            let before = masks.len();
            for b in &p.blocks {
                for phi in &b.phis {
                    if matches!(phi.reg, Reg::S(c) if c % 2 == 0 || c == EXEC)
                        && phi.args.iter().any(|a| masks.contains(a))
                    {
                        masks.insert(phi.value);
                    }
                }
            }
            if masks.len() == before {
                break;
            }
        }
        let numbers = if waves > 1 {
            number_reads(p, st, &masks)
        } else {
            HashSet::new()
        };
        Split {
            read,
            masks,
            waves,
            numbers,
            ..Split::default()
        }
    }

    pub(super) fn is_mask(&self, v: Value) -> bool {
        self.exact.contains_key(&v) || self.high.contains_key(&v)
    }

    /// Why a program with `lds_dwords` of its own LDS cannot be split.
    pub(super) fn refusal(&self, lds_dwords: u32) -> Option<Why> {
        let xchg = if self.waves > 1 {
            self.waves * STRIDE
        } else {
            2 * SLOTS
        };
        ((lds_dwords + xchg) * 4 > HOST_LDS_BYTES)
            .then_some("LDS too large for a host group with the halves' exchange")
    }

    fn several(&self) -> bool {
        self.waves > 1
    }
}

/// SGPR values some instruction or join may read as numbers rather than
/// as lane masks: every read except EXEC as a vector or memory
/// instruction's lane set, a mask operation's sources (as `split_sop`
/// takes them), a branch condition, a select's or carry's mask and a mask
/// join's arguments. A reader left out here is refused later, one taken in
/// needlessly costs an exchange.
fn number_reads(
    p: &crate::ir::Program,
    st: &crate::structure::Structured,
    masks: &HashSet<Value>,
) -> HashSet<Value> {
    let mut out = HashSet::new();
    let sgpr = |r: Reg| matches!(r, Reg::S(_));
    for b in &p.blocks {
        for phi in &b.phis {
            let Reg::S(c) = phi.reg else { continue };
            let mask_phi =
                c == EXEC || st.masks.contains_key(&phi.value) || masks.contains(&phi.value);
            let high_of_mask = c % 2 == 1
                && b.phis.iter().any(|q| {
                    q.reg == Reg::S(c - 1)
                        && (c - 1 == EXEC
                            || st.masks.contains_key(&q.value)
                            || masks.contains(&q.value))
                });
            if !mask_phi && !high_of_mask {
                out.extend(phi.args.iter().copied());
            }
        }
        for i in &b.insts {
            let name = i.name();
            match &i.fields {
                Fields::Sopp { .. } if name.starts_with("s_cbranch") => {}
                Fields::Sop { dst, .. } => {
                    let dst = match dst {
                        Some(Operand::Scalar { code, .. }) => i.write(Reg::S(*code)),
                        _ => None,
                    };
                    let mask_op = i.write(Reg::S(EXEC)).is_some()
                        || dst.is_some_and(|v| st.masks.contains_key(&v));
                    if !mask_op {
                        out.extend(i.reads.iter().filter(|r| sgpr(r.0)).map(|r| r.1));
                    }
                }
                Fields::Valu(v) => {
                    let selects = name.starts_with("v_cndmask")
                        || name.contains("_co_ci_")
                        || name.starts_with("v_div_fmas");
                    let explicit: Vec<u8> = v
                        .src
                        .iter()
                        .filter_map(|o| match *o {
                            Operand::Scalar { code, dwords } => Some((code, dwords)),
                            _ => None,
                        })
                        .flat_map(|(code, dwords)| (0..dwords.max(1)).map(move |k| code + k))
                        .collect();
                    // A select's lane mask (its third operand, or VCC when it
                    // has two) is read per lane; its data operands are
                    // numbers like any other instruction's.
                    let data: Vec<u8> = if selects {
                        let n = if name.starts_with("v_div_fmas") { 3 } else { 2 };
                        v.src
                            .iter()
                            .take(n)
                            .filter_map(|o| match *o {
                                Operand::Scalar { code, dwords } => Some((code, dwords)),
                                _ => None,
                            })
                            .flat_map(|(code, dwords)| (0..dwords.max(1)).map(move |k| code + k))
                            .collect()
                    } else {
                        Vec::new()
                    };
                    for &(r, val) in &i.reads {
                        let Reg::S(c) = r else { continue };
                        let lane_set = (c == EXEC || c == EXEC + 1) && !explicit.contains(&c);
                        if (!lane_set && !selects) || data.contains(&c) {
                            out.insert(val);
                        }
                    }
                }
                _ => {
                    for &(r, val) in &i.reads {
                        if matches!(r, Reg::S(c) if c != EXEC && c != EXEC + 1) {
                            out.insert(val);
                        }
                    }
                }
            }
        }
    }
    out
}

impl Lower<'_> {
    fn sp(&mut self) -> &mut Split {
        self.split.as_mut().expect("split lowering")
    }

    /// Host lane, console lane and half, made once in the entry block. The
    /// console lane follows the relabelled thread id (see `thread_id`).
    pub(super) fn split_lanes(&mut self) -> (V, V, V) {
        if let Some(l) = self.sp().lanes {
            return l;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let host = self
            .s
            .op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let (k32, k63, k5, k6) = (
            self.s.m.i32(32),
            self.s.m.i32(63),
            self.s.m.i32(5),
            self.s.m.i32(6),
        );
        let upper = self.s.m.bin(e, Bin::And, first, k32);
        let console = self.s.m.bin(e, Bin::Or, upper, host);
        let console = self.s.m.bin(e, Bin::And, console, k63);
        let half = self.s.m.bin(e, Bin::LShr, upper, k5);
        let wave = self.s.m.bin(e, Bin::LShr, first, k6);
        let l = (host, console, half);
        self.sp().lanes = Some(l);
        self.sp().wave = Some(wave);
        l
    }

    fn barrier(&mut self, at: usize, mode: u32) {
        let mode = self.s.m.i32(mode);
        let void = self.s.t.void;
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// The group-shared exchange area.
    fn xchg(&mut self) -> V {
        if let Some(g) = self.sp().xchg {
            return g;
        }
        let n = if self.sp().several() {
            self.sp().waves * STRIDE
        } else {
            2 * SLOTS
        };
        let arr = self.s.m.array(n, self.s.t.i32);
        let g = self.s.m.global("xchg", arr, 3);
        self.sp().xchg = Some(g);
        g
    }

    /// Each half's `vals` (i32, the same in every lane of the half), as
    /// (lower half's, upper half's), exchanged in block `at`, which with
    /// several waves per group must be the current block (the exchange
    /// then continues the code in a block of its own).
    pub(super) fn pair_exchange(&mut self, at: usize, vals: &[V]) -> Vec<(V, V)> {
        assert!(vals.len() as u32 <= SLOTS);
        self.s.flags |= super::flags::WAVE_OPS;
        if self.sp().several() {
            assert_eq!(at, self.cur, "a pair exchange away from the current block");
            return self.pair_exchange_waves(vals);
        }
        let (_, _, half) = self.split_lanes();
        let g = self.xchg();
        let k = self.s.m.i32(SLOTS);
        let mine = self.s.m.bin(at, Bin::Mul, half, k);
        for (j, &v) in vals.iter().enumerate() {
            let kj = self.s.m.i32(j as u32);
            let idx = self.s.m.bin(at, Bin::Add, mine, kj);
            let ptr = self.s.m.element(at, g, idx);
            self.s.m.store(at, ptr, v);
        }
        self.barrier(at, SYNC);
        let out = (0..vals.len() as u32)
            .map(|j| {
                let (k0, k1) = (self.s.m.i32(j), self.s.m.i32(SLOTS + j));
                let p0 = self.s.m.element(at, g, k0);
                let p1 = self.s.m.element(at, g, k1);
                (self.s.m.load(at, p0), self.s.m.load(at, p1))
            })
            .collect();
        // Nobody writes the slots again before both halves have read them.
        self.barrier(at, SYNC);
        out
    }

    /// [`Self::pair_exchange`] in a group of several console waves. Each
    /// half keeps a count of the exchanges it has done in its wave's area
    /// (only that half writes it). Exchange `n` writes buffer `n & 1`, bumps
    /// the count to `n + 1` and waits until the partner's count passes `n`,
    /// then reads the partner's slots. A half is at most one exchange ahead
    /// of its partner, which therefore never still reads the buffer it
    /// writes.
    fn pair_exchange_waves(&mut self, vals: &[V]) -> Vec<(V, V)> {
        let (_, _, half) = self.split_lanes();
        let wave = self.sp().wave.expect("made with the lanes");
        let g = self.xchg();
        let c = self.cur;
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (stride, counts, slots, one) = (
            k(self, STRIDE),
            k(self, 4 * SLOTS),
            k(self, SLOTS),
            k(self, 1),
        );
        let base = self.s.m.bin(c, Bin::Mul, wave, stride);
        let counts = self.s.m.bin(c, Bin::Add, base, counts);
        let mine_at = self.s.m.bin(c, Bin::Add, counts, half);
        let other = self.s.m.bin(c, Bin::Xor, half, one);
        let other_at = self.s.m.bin(c, Bin::Add, counts, other);
        let mine_count = self.s.m.element(c, g, mine_at);
        let other_count = self.s.m.element(c, g, other_at);
        let n = self.s.m.load(c, mine_count);
        let parity = self.s.m.bin(c, Bin::And, n, one);
        let two_slots = k(self, 2 * SLOTS);
        let buffer = self.s.m.bin(c, Bin::Mul, parity, two_slots);
        let buffer = self.s.m.bin(c, Bin::Add, base, buffer);
        let own = self.s.m.bin(c, Bin::Mul, half, slots);
        let own = self.s.m.bin(c, Bin::Add, buffer, own);
        for (j, &v) in vals.iter().enumerate() {
            let kj = k(self, j as u32);
            let idx = self.s.m.bin(c, Bin::Add, own, kj);
            let ptr = self.s.m.element(c, g, idx);
            self.s.m.store(c, ptr, v);
        }
        self.barrier(c, FENCE);
        let next = self.s.m.bin(c, Bin::Add, n, one);
        self.s.m.atomic(c, Rmw::Xchg, mine_count, next);
        // Wait for the partner. One lane of the host wave polls (an LDS
        // atomic from every lane to one address is 32 serialized accesses
        // per poll, which also slows the partner being waited for) and
        // hands the count to the others, so the whole wave leaves together.
        let (host, _, _) = self.split_lanes();
        let (spin, poll, join, done) = (
            self.s.m.new_block(),
            self.s.m.new_block(),
            self.s.m.new_block(),
            self.s.m.new_block(),
        );
        self.s.m.terminate(c, Term::Br(spin));
        let i32t = self.s.t.i32;
        let polls = self.s.m.phi(spin, i32t);
        let z = k(self, 0);
        let first = self.s.m.cmp(spin, pred::EQ, host, z);
        self.s.m.terminate(spin, Term::CondBr(first, poll, join));
        let polled = self.s.m.atomic(poll, Rmw::Or, other_count, z);
        self.s.m.terminate(poll, Term::Br(join));
        let theirs = self.s.m.phi(join, i32t);
        self.s.m.set_phi(theirs, vec![(polled, poll), (z, spin)]);
        let theirs = self.s.op(
            join,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[theirs],
        );
        let ahead = self.s.m.bin(join, Bin::Sub, theirs, n);
        let ready = self.s.m.cmp(join, pred::SGT, ahead, z);
        let polls_next = self.s.m.bin(join, Bin::Add, polls, one);
        let limit = super::super::probe::spin_limit().unwrap_or(SPIN_LIMIT);
        let limit = k(self, limit);
        let late = self.s.m.cmp(join, pred::IUGE, polls_next, limit);
        let stop = self.s.m.bin(join, Bin::Or, ready, late);
        self.s.m.set_phi(polls, vec![(z, c), (polls_next, join)]);
        self.s.m.terminate(join, Term::CondBr(stop, done, spin));
        self.cur = done;
        self.barrier(done, FENCE);
        (0..vals.len() as u32)
            .map(|j| {
                let (kj, kh) = (k(self, j), k(self, SLOTS + j));
                let i0 = self.s.m.bin(done, Bin::Add, buffer, kj);
                let i1 = self.s.m.bin(done, Bin::Add, buffer, kh);
                let p0 = self.s.m.element(done, g, i0);
                let p1 = self.s.m.element(done, g, i1);
                (self.s.m.load(done, p0), self.s.m.load(done, p1))
            })
            .collect()
    }

    /// Orders the halves' LDS accesses: a group barrier when the group is
    /// one console wave, else a pair sync.
    fn pair_sync(&mut self) {
        if self.sp().several() {
            self.pair_exchange_waves(&[]);
        } else {
            let c = self.cur;
            self.barrier(c, SYNC);
        }
    }

    /// Whether any lane of the console wave has `bit` (exchanged in the
    /// current block).
    fn any64(&mut self, bit: V) -> V {
        let i1 = self.s.t.i1;
        let i32t = self.s.t.i32;
        let c = self.cur;
        let any = self.s.op(c, "waveAnyTrue", Attr::NoUnwind, i1, 113, &[bit]);
        let any = self.s.m.cast(c, Cast::ZExt, any, i32t);
        let [(a, b)] = self.pair_exchange(c, &[any])[..] else {
            unreachable!()
        };
        let c = self.cur;
        let both = self.s.m.bin(c, Bin::Or, a, b);
        let z = self.s.m.i32(0);
        self.s.m.cmp(c, pred::NE, both, z)
    }

    /// The console lane's bit of a 64-bit mask given as two numbers.
    fn number_bit(&mut self, at: usize, lo: V, hi: V) -> V {
        let (_, lane, _) = self.split_lanes();
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let low = self.s.m.cmp(at, pred::IULT, lane, k32);
        let word = self.s.m.select(at, low, lo, hi);
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    fn const_bits(&mut self, at: usize, k: u64) -> V {
        match k {
            0 => self.s.m.bool(false),
            u64::MAX => self.s.m.bool(true),
            _ => {
                let (lo, hi) = (self.s.m.i32(k as u32), self.s.m.i32((k >> 32) as u32));
                self.number_bit(at, lo, hi)
            }
        }
    }

    fn word(&mut self, v: Value) -> Result<V, Why> {
        match self.get(v)? {
            Val::I32(x) => Ok(x),
            _ => Err("a lane mask dword without a value"),
        }
    }

    /// The console lane's exact bit of the mask in `lo`, `hi`.
    pub(super) fn exact_pair(&mut self, lo: Value, hi: Option<Value>, at: usize) -> Result<V, Why> {
        let sp = self.split.as_ref().expect("split lowering");
        if let Some(&x) = sp.exact.get(&lo) {
            let highs = &self.x.st.highs;
            let paired =
                hi.is_none_or(|h| sp.high.get(&h) == Some(&lo) || highs.get(&h) == Some(&lo));
            if paired {
                return Ok(x);
            }
        }
        // A mask held as numbers (or a pair whose dwords come from
        // different masks): the lane tests its own bit.
        let hi = hi.ok_or("a lane mask without its high dword")?;
        let (l, h) = (self.word(lo)?, self.word(hi)?);
        Ok(self.number_bit(at, l, h))
    }

    /// The exact bit of the mask an operand of `inst` names.
    fn exact_operand(&mut self, inst: &Inst, op: &Operand, at: usize) -> Result<V, Why> {
        match *op {
            Operand::Scalar { code, .. } if code == NULL => Ok(self.s.m.bool(false)),
            Operand::Scalar { code, .. } => {
                let lo = inst.read(Reg::S(code)).ok_or("an unread lane mask")?;
                self.exact_pair(lo, inst.read(Reg::S(code + 1)), at)
            }
            Operand::Const(k) => {
                let x = crate::resource::inline(k).ok_or("a special lane mask")?;
                Ok(self.const_bits(at, x as i32 as i64 as u64))
            }
            _ => Err("a literal lane mask"),
        }
    }

    /// The exact bit of the EXEC `inst` runs under.
    pub(super) fn split_exec(&mut self, inst: &Inst) -> Result<V, Why> {
        let lo = inst
            .read(Reg::S(EXEC))
            .ok_or("a vector instruction without EXEC")?;
        let c = self.cur;
        self.exact_pair(lo, inst.read(Reg::S(EXEC + 1)), c)
    }

    /// A vector instruction's guard: its lane's EXEC bit.
    pub(super) fn split_guard(&mut self, inst: &Inst) -> Result<Option<V>, Why> {
        let e = self.split_exec(inst)?;
        Ok((self.s.m.int_value(e) != Some(1)).then_some(e))
    }

    fn note_mask(&mut self, inst: &Inst, code: u8, bit: V) {
        let Some(lo) = inst.write(Reg::S(code)) else {
            return;
        };
        self.sp().exact.insert(lo, bit);
        let hi = inst.write(Reg::S(code + 1));
        if let Some(hi) = hi {
            self.sp().high.insert(hi, lo);
        }
        let numbers = &self.sp().numbers;
        if numbers.contains(&lo) || hi.is_some_and(|h| numbers.contains(&h)) {
            self.ballot_here(lo, bit);
        }
    }

    /// The numbers of the mask `lo` of exact bits `bit`, made in the current
    /// block (or constants).
    fn ballot_here(&mut self, lo: Value, bit: V) {
        if self.ballots.contains_key(&lo) {
            return;
        }
        let xy = match self.constant_ballot(bit) {
            Some(xy) => xy,
            None => {
                let c = self.cur;
                let w = self.half_ballot(c, bit);
                let [xy] = self.pair_exchange(c, &[w])[..] else {
                    unreachable!()
                };
                xy
            }
        };
        self.ballots.insert(lo, xy);
    }

    /// A mask of the same bit in every console lane, as numbers (every
    /// lane of a split group exists).
    fn constant_ballot(&mut self, bit: V) -> Option<(V, V)> {
        let k = self.s.m.int_value(bit)?;
        let w = self.s.m.i32(if k != 0 { u32::MAX } else { 0 });
        Some((w, w))
    }

    fn half_ballot(&mut self, at: usize, bit: V) -> V {
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.m.extract(at, r, 0)
    }

    /// A VALU write of a lane mask (a compare or carry) whose bits before
    /// EXEC are `raw`: off lanes get 0.
    pub(super) fn split_lanes_written(&mut self, inst: &Inst, code: u8, raw: V) {
        let Ok(e) = self.split_exec(inst) else {
            return;
        };
        let c = self.cur;
        let bit = self.s.m.bin(c, Bin::And, raw, e);
        self.note_mask(inst, code, bit);
    }

    /// The program's EXEC on entry: every lane. With several waves per
    /// group, each half's count of exchanges starts at 0 before any
    /// exchange (the entry block runs in every thread, so a group barrier
    /// can order this).
    pub(super) fn split_inputs(&mut self) {
        let t = self.s.m.bool(true);
        let lo = self
            .x
            .p
            .inputs
            .iter()
            .find(|i| i.0 == Reg::S(EXEC))
            .map(|i| i.1);
        let hi = self
            .x
            .p
            .inputs
            .iter()
            .find(|i| i.0 == Reg::S(EXEC + 1))
            .map(|i| i.1);
        if let Some(lo) = lo {
            self.sp().exact.insert(lo, t);
            if let Some(hi) = hi {
                self.sp().high.insert(hi, lo);
            }
        }
        let bound = self.sp().lds_bound;
        if bound > 0 {
            // The dword out-of-bounds reads take holds 0 (see
            // `split_lds_index`); with several waves the barrier below
            // orders this store, else a barrier of its own.
            let e = self.s.entry;
            let (z, at) = (self.s.m.i32(0), self.s.m.i32(bound));
            let lds = self.s.lds();
            let p = self.s.m.element(e, lds, at);
            self.s.m.store(e, p, z);
            if !self.sp().several() {
                self.barrier(e, SYNC);
            }
        }
        if super::super::probe::clear_lds() && self.s.lds_dwords > 0 {
            // The harness's: every thread zeroes its share of LDS.
            let e = self.s.entry;
            let flat = self.s.op(
                e,
                "flattenedThreadIdInGroup.i32",
                Attr::ReadNone,
                self.s.t.i32,
                96,
                &[],
            );
            let threads = 64 * self.sp().waves;
            let z = self.s.m.i32(0);
            let lds = self.s.lds();
            for k in 0..self.s.lds_dwords.div_ceil(threads) {
                let base = self.s.m.i32(k * threads);
                let at = self.s.m.bin(e, Bin::Add, flat, base);
                let n = self.s.m.i32(self.s.lds_dwords);
                let inside = self.s.m.cmp(e, pred::IULT, at, n);
                let at = self.s.m.select(e, inside, at, z);
                let p = self.s.m.element(e, lds, at);
                // Index 0 may be written twice: with 0 both times.
                self.s.m.store(e, p, z);
            }
            self.barrier(e, SYNC);
        }
        if self.sp().several() {
            let (_, _, half) = self.split_lanes();
            let wave = self.sp().wave.expect("made with the lanes");
            let g = self.xchg();
            let e = self.s.entry;
            let (stride, counts, z) = (
                self.s.m.i32(STRIDE),
                self.s.m.i32(4 * SLOTS),
                self.s.m.i32(0),
            );
            let at = self.s.m.bin(e, Bin::Mul, wave, stride);
            let at = self.s.m.bin(e, Bin::Add, at, counts);
            let at = self.s.m.bin(e, Bin::Add, at, half);
            let p = self.s.m.element(e, g, at);
            self.s.m.store(e, p, z);
            self.barrier(e, SYNC);
        }
    }

    /// A join of IR block `b`: an exact phi for the low dword of a lane
    /// mask (EXEC or one structurisation tracks).
    pub(super) fn split_phi(&mut self, b: usize, i: usize) {
        let p = self.x.p;
        let phi = &p.blocks[b].phis[i];
        let Reg::S(c) = phi.reg else { return };
        let st = self.x.st;
        if c == EXEC || st.masks.contains_key(&phi.value) || self.sp().masks.contains(&phi.value) {
            let v = self.s.m.phi(self.cur, self.s.t.i1);
            self.sp().exact.insert(phi.value, v);
            self.sp().pending.push((v, b, i));
            let h = p.blocks[b].phis.iter().find(|q| q.reg == Reg::S(c + 1));
            if let Some(h) = h {
                self.sp().high.insert(h.value, phi.value);
            }
            let numbers = &self.sp().numbers;
            let as_number =
                numbers.contains(&phi.value) || h.is_some_and(|h| numbers.contains(&h.value));
            if as_number {
                self.sp().phi_numbers.push((phi.value, v));
            }
        }
    }

    /// After the joins of a block: the numbers of its mask joins some
    /// reader takes as numbers (several waves per group).
    pub(super) fn split_phi_numbers(&mut self) {
        for (lo, bit) in std::mem::take(&mut self.sp().phi_numbers) {
            self.ballot_here(lo, bit);
        }
    }

    pub(super) fn split_fill_phis(&mut self) -> Result<(), Why> {
        let p = self.x.p;
        while let Some((v, b, i)) = self.sp().pending.pop() {
            let phi = &p.blocks[b].phis[i];
            let Reg::S(c) = phi.reg else { unreachable!() };
            let high = p.blocks[b].phis.iter().find(|q| q.reg == Reg::S(c + 1));
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let x = self.exact_pair(phi.args[n], high.map(|h| h.args[n]), qb)?;
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        Ok(())
    }

    /// A scalar instruction on lane masks, or `None` for one this lowering
    /// leaves to the scalar code (no mask written).
    pub(super) fn split_sop(&mut self, inst: &Inst) -> Option<Result<(), Why>> {
        let Fields::Sop { dst, src, .. } = &inst.fields else {
            return None;
        };
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        let writes_exec = inst.write(Reg::S(EXEC)).is_some();
        if !writes_exec && !lo.is_some_and(|v| self.x.st.masks.contains_key(&v)) {
            return None;
        }
        Some(self.split_mask_op(inst, dst_code, src))
    }

    fn split_mask_op(&mut self, inst: &Inst, dst: Option<u8>, src: &[Operand]) -> Result<(), Why> {
        let name = inst.name();
        let base = name
            .strip_suffix("_b64")
            .and_then(|n| n.strip_prefix("s_"))
            .ok_or("a 32-bit operation on a wave64 lane mask")?;
        let c = self.cur;
        let t = self.s.m.bool(true);
        let src_bit = |l: &mut Self, k: usize| -> Result<V, Why> {
            let op = src.get(k).ok_or("a missing operand")?;
            l.exact_operand(inst, op, c)
        };
        let not = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Xor, x, t);
        let combine = |l: &mut Self, op: &str, a: V, b: V| -> Result<V, Why> {
            let m = &mut l.s.m;
            Ok(match op {
                "and" => m.bin(c, Bin::And, a, b),
                "or" => m.bin(c, Bin::Or, a, b),
                "xor" => m.bin(c, Bin::Xor, a, b),
                "andn2" => {
                    let nb = m.bin(c, Bin::Xor, b, t);
                    m.bin(c, Bin::And, a, nb)
                }
                "orn2" => {
                    let nb = m.bin(c, Bin::Xor, b, t);
                    m.bin(c, Bin::Or, a, nb)
                }
                "andn1" => {
                    let na = m.bin(c, Bin::Xor, a, t);
                    m.bin(c, Bin::And, na, b)
                }
                "orn1" => {
                    let na = m.bin(c, Bin::Xor, a, t);
                    m.bin(c, Bin::Or, na, b)
                }
                "nand" => {
                    let x = m.bin(c, Bin::And, a, b);
                    m.bin(c, Bin::Xor, x, t)
                }
                "nor" => {
                    let x = m.bin(c, Bin::Or, a, b);
                    m.bin(c, Bin::Xor, x, t)
                }
                "xnor" => {
                    let x = m.bin(c, Bin::Xor, a, b);
                    m.bin(c, Bin::Xor, x, t)
                }
                _ => return Err("a lane-mask operation the split lowering does not model"),
            })
        };
        let scc_of = |l: &mut Self| -> Result<V, Why> {
            match inst.read(Reg::Scc).map(|v| l.get(v)) {
                Some(Ok(Val::Bit(x))) => Ok(x),
                _ => Err("SCC without a value"),
            }
        };
        // (destination's bits, EXEC's new bits)
        let (to_dst, to_exec) = if let Some(op) = base.strip_suffix("_saveexec") {
            let e = self.split_exec_read(inst)?;
            let a = src_bit(self, 0)?;
            (Some(e), Some(combine(self, op, a, e)?))
        } else if let Some(op) = base.strip_suffix("_wrexec") {
            let e = self.split_exec_read(inst)?;
            let a = src_bit(self, 0)?;
            let x = combine(self, op, a, e)?;
            (Some(x), Some(x))
        } else {
            let r = match base {
                "mov" => src_bit(self, 0)?,
                "not" => {
                    let a = src_bit(self, 0)?;
                    not(self, a)
                }
                "cselect" => {
                    let (a, b, s) = (src_bit(self, 0)?, src_bit(self, 1)?, scc_of(self)?);
                    self.s.m.select(c, s, a, b)
                }
                "cmov" => {
                    let code = dst.ok_or("s_cmov without a destination")?;
                    let lo = inst.read(Reg::S(code)).ok_or("s_cmov's old value")?;
                    let old = self.exact_pair(lo, inst.read(Reg::S(code + 1)), c)?;
                    let (a, s) = (src_bit(self, 0)?, scc_of(self)?);
                    self.s.m.select(c, s, a, old)
                }
                "wqm" => {
                    let a = src_bit(self, 0)?;
                    self.quad_any(a)
                }
                op => {
                    let (a, b) = (src_bit(self, 0)?, src_bit(self, 1)?);
                    combine(self, op, a, b)?
                }
            };
            if dst == Some(EXEC) {
                (None, Some(r))
            } else {
                (Some(r), None)
            }
        };
        // SCC: whether the result has any lane set (s_mov, s_cmov and
        // s_cselect leave it alone; their SCC is no write). Exchanges come
        // last: with several waves they move to a block of their own.
        let scc = inst.write(Reg::Scc).map(|scc| {
            let r = if base.ends_with("_saveexec") || base.ends_with("_wrexec") {
                to_exec.expect("EXEC")
            } else {
                to_exec.or(to_dst).expect("a result")
            };
            (scc, r)
        });
        if let (Some(x), Some(code)) = (to_dst, dst)
            && code != EXEC
        {
            self.note_mask(inst, code, x);
        }
        if let Some(x) = to_exec {
            self.note_mask(inst, EXEC, x);
        }
        if let Some((scc, r)) = scc {
            let val = if self.sp().read.contains(&scc) {
                Val::Bit(self.any64(r))
            } else {
                Val::Missing("an unread SCC")
            };
            self.vals.insert(scc, val);
        }
        Ok(())
    }

    fn split_exec_read(&mut self, inst: &Inst) -> Result<V, Why> {
        let lo = inst.read(Reg::S(EXEC)).ok_or("EXEC unread")?;
        let c = self.cur;
        self.exact_pair(lo, inst.read(Reg::S(EXEC + 1)), c)
    }

    /// Whether any lane of this lane's quad has `bit` (quads stay inside a
    /// half).
    fn quad_any(&mut self, bit: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let x = self.s.m.cast(c, Cast::ZExt, bit, i32t);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self
                .s
                .op(c, "quadReadLaneAt.i32", Attr::NoUnwind, i32t, 122, &[x, kn]);
            r = self.s.m.bin(c, Bin::Or, r, read);
        }
        let z = self.s.m.i32(0);
        self.s.m.cmp(c, pred::NE, r, z)
    }

    /// A mask dword as a number: each half's ballot of the exact bits,
    /// exchanged where the mask is written. With several waves per group
    /// the numbers of masks read so are made where they are written
    /// ([`Split::numbers`]); one this lowering did not foresee is refused.
    pub(super) fn split_mask_dword(&mut self, v: Value) -> Result<V, Why> {
        let (lo, high) = match self.split.as_ref().and_then(|sp| sp.high.get(&v)).or(self
            .x
            .st
            .highs
            .get(&v))
        {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        if let Some(&(x, y)) = self.ballots.get(&lo) {
            return Ok(if high { y } else { x });
        }
        let bit = *self
            .sp()
            .exact
            .get(&lo)
            .ok_or("a lane mask the split lowering does not follow")?;
        if let Some((x, y)) = self.constant_ballot(bit) {
            self.ballots.insert(lo, (x, y));
            return Ok(if high { y } else { x });
        }
        if self.sp().several() {
            return Err("a lane mask read as a number the split lowering did not foresee");
        }
        let at = match self.def_block.get(&lo) {
            None => self.s.entry,
            Some(&b) if b == self.cur_ir => self.cur,
            Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
        };
        if self.sp().guarded.contains(&at) {
            return Err("a lane mask read as a number where only some lanes run");
        }
        let w = self.half_ballot(at, bit);
        let [(x, y)] = self.pair_exchange(at, &[w])[..] else {
            unreachable!()
        };
        self.ballots.insert(lo, (x, y));
        Ok(if high { y } else { x })
    }

    /// A branch as the console's wave takes it: the condition for `taken`.
    pub(super) fn split_branch(&mut self, b: usize) -> Result<(V, usize, usize), Why> {
        let p = self.x.p;
        let Exit::Branch { taken, not_taken } = p.blocks[b].exit else {
            unreachable!()
        };
        let inst = p.blocks[b]
            .insts
            .last()
            .ok_or("a branch without its instruction")?;
        let nonzero = |l: &mut Self, code: u8| -> Result<V, Why> {
            let lo = inst
                .read(Reg::S(code))
                .ok_or("a branch on an unread mask")?;
            let hi = inst.read(Reg::S(code + 1));
            let c = l.cur;
            if l.sp().is_mask(lo) || l.x.st.masks.contains_key(&lo) {
                let bit = l.exact_pair(lo, hi, c)?;
                return Ok(l.any64(bit));
            }
            let x = l.word(lo)?;
            let x = match hi {
                Some(h) => {
                    let y = l.word(h)?;
                    l.s.m.bin(c, Bin::Or, x, y)
                }
                None => x,
            };
            let z = l.s.m.i32(0);
            Ok(l.s.m.cmp(c, pred::NE, x, z))
        };
        let (cond, set) = match inst.name() {
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                let scc = inst.read(Reg::Scc).ok_or("a branch on unread SCC")?;
                let Val::Bit(x) = self.get(scc)? else {
                    return Err("SCC without a value");
                };
                (x, inst.name() == "s_cbranch_scc1")
            }
            "s_cbranch_vccz" | "s_cbranch_vccnz" => {
                (nonzero(self, VCC)?, inst.name() == "s_cbranch_vccnz")
            }
            "s_cbranch_execz" | "s_cbranch_execnz" => {
                (nonzero(self, EXEC)?, inst.name() == "s_cbranch_execnz")
            }
            _ => return Err("a branch the split lowering does not model"),
        };
        // With several waves an exchange moves the block's end to a block
        // of its own: the branch leaves from there.
        self.tail[b] = Some(self.cur);
        // A branch on a clear condition goes the other way round (no `xor`).
        Ok(if set {
            (cond, taken, not_taken)
        } else {
            (cond, not_taken, taken)
        })
    }

    /// `v_readlane_b32`: lane `lane` (0-63) of `x`, read in its half.
    pub(super) fn split_readlane(&mut self, x: V, lane: V) -> V {
        let c = self.cur;
        let k31 = self.s.m.i32(31);
        let within = self.s.m.bin(c, Bin::And, lane, k31);
        let r = self.read_host_lane(x, within);
        let [(r0, r1)] = self.pair_exchange(c, &[r])[..] else {
            unreachable!()
        };
        let c = self.cur;
        let (k32, z) = (self.s.m.i32(32), self.s.m.i32(0));
        let upper = self.s.m.bin(c, Bin::And, lane, k32);
        let low = self.s.m.cmp(c, pred::EQ, upper, z);
        self.s.m.select(c, low, r0, r1)
    }

    /// `v_readfirstlane_b32`: `x` of the first lane EXEC holds.
    pub(super) fn split_readfirstlane(&mut self, inst: &Inst, x: V) -> Result<V, Why> {
        let e = self.split_exec(inst)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let w = self.half_ballot(c, e);
        let first = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[w]);
        let z = self.s.m.i32(0);
        let has = self.s.m.cmp(c, pred::NE, w, z);
        let first = self.s.m.select(c, has, first, z);
        let r = self.read_host_lane(x, first);
        let has = self.s.m.cast(c, Cast::ZExt, has, i32t);
        let [(has0, has1), (r0, r1)] = self.pair_exchange(c, &[has, r])[..] else {
            unreachable!()
        };
        let c = self.cur;
        let has1 = self.s.m.cmp(c, pred::NE, has1, z);
        // No lane at all: lane 0's.
        let upper = self.s.m.cmp(c, pred::EQ, has0, z);
        let upper = self.s.m.bin(c, Bin::And, upper, has1);
        Ok(self.s.m.select(c, upper, r1, r0))
    }

    /// `op` of `a` and `b` (i32 bit patterns).
    pub(super) fn split_combine(&mut self, op: Op, a: V, b: V) -> V {
        let c = self.cur;
        let int = |l: &mut Self, p: u8| {
            let lt = l.s.m.cmp(c, p, a, b);
            l.s.m.select(c, lt, a, b)
        };
        match op {
            Op::AddU => self.s.m.bin(c, Bin::Add, a, b),
            Op::Or => self.s.m.bin(c, Bin::Or, a, b),
            Op::And => self.s.m.bin(c, Bin::And, a, b),
            Op::Xor => self.s.m.bin(c, Bin::Xor, a, b),
            Op::MinU => int(self, pred::IULT),
            Op::MaxU => int(self, pred::IUGT),
            Op::MinI => int(self, pred::SLT),
            Op::MaxI => int(self, pred::SGT),
            Op::AddH => {
                let f32t = self.s.t.f32;
                let mut half = |x: V| {
                    self.s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                };
                let (fa, fb) = (half(a), half(b));
                let r = self.s.m.bin(c, Bin::Add, fa, fb);
                self.f16_nearest(r)
            }
            Op::AddF | Op::MinF | Op::MaxF => {
                let (fa, fb) = (self.as_float(a), self.as_float(b));
                let r = match op {
                    Op::AddF => self.s.m.bin(c, Bin::Add, fa, fb),
                    _ => self.max_min(op == Op::MaxF, fa, fb),
                };
                self.as_int(r)
            }
        }
    }

    /// A whole-wave reduction or prefix over the console wave, from this
    /// half's: `half_result` over the half's lanes, `x` this lane's term.
    pub(super) fn split_reduce(
        &mut self,
        op: Op,
        x: V,
        half_result: V,
        prefix: bool,
        total: impl FnOnce(&mut Self) -> V,
    ) -> V {
        let _ = x;
        if !prefix {
            let c = self.cur;
            let [(r0, r1)] = self.pair_exchange(c, &[half_result])[..] else {
                unreachable!()
            };
            return self.split_combine(op, r0, r1);
        }
        // The upper half's lanes follow every lane of the lower half.
        let t = total(self);
        let c = self.cur;
        let [(t0, _)] = self.pair_exchange(c, &[t])[..] else {
            unreachable!()
        };
        let (_, _, half) = self.split_lanes();
        let with = self.split_combine(op, t0, half_result);
        let c = self.cur;
        let z = self.s.m.i32(0);
        let lower = self.s.m.cmp(c, pred::EQ, half, z);
        self.s.m.select(c, lower, half_result, with)
    }

    /// Before an LDS access: a sync of the halves unless it and the access
    /// before it in this console block (with no sync between) are both plain reads
    /// or both plain writes (`kind`; `None` for any other access).
    // shortcut: a run of plain writes is not ordered between the halves; two
    // halves writing the same LDS address in different writes of one run may
    // land in the other order than on the console. Programs clear LDS so
    // (hundreds of writes in a row), and a pair sync before each one makes
    // a group of several waves spin through hundreds of exchanges.
    pub(super) fn split_lds_barrier(&mut self, kind: Option<bool>) {
        // By console block: a guarded write ends the DXIL block it is in.
        let b = self.cur_ir;
        let same_run = kind.is_some()
            && !super::super::probe::strict_lds()
            && self.sp().lds_last == kind.map(|k| (b, k));
        if !same_run {
            self.pair_sync();
        }
        self.sp().lds_last = kind.map(|k| (b, k));
    }

    /// The LDS dword `index` of the program's `bound`, or outside them one
    /// of the two dwords the host's LDS has after the program's: for a read
    /// the first, which holds 0 (the console reads 0 out of bounds), for
    /// anything else the second (the console drops such writes). Unbounded,
    /// such accesses reach the halves' exchange area, which the host places
    /// next to the program's LDS (an address below it, as a negative offset
    /// wraps, reads the halves' last exchange): there a write makes the
    /// halves take each other's numbers wrong or wait for each other
    /// forever, and a read takes what the halves last exchanged.
    pub(super) fn split_lds_index(&mut self, index: V, bound: u32, read: bool) -> V {
        let c = self.cur;
        let n = self.s.m.i32(bound);
        let inside = self.s.m.cmp(c, pred::IULT, index, n);
        let outside = self.s.m.i32(if read { bound } else { bound + 1 });
        self.s.m.select(c, inside, index, outside)
    }
}
