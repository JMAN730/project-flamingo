//! SCC of an EXEC write that a branch on EXEC then tests: the compiler's
//! waterfall loops (`s_andn2_wrexec`, `s_cbranch_execnz` back) leave it set
//! to "some lane is left", and code after the loop may read it (an
//! `s_cselect` of EXEC or 0, say). On each edge of that branch the wave
//! knows its value: zero where EXEC was found empty, one elsewhere.

use super::{EXEC, Lower};
use crate::gfx10::Fields;
use crate::ir::{Exit, Reg, Value};

impl Lower<'_> {
    /// The value of `scc` wherever it is read, when it is the SCC of an
    /// EXEC write whose result ("not zero") the branch ending the writing
    /// block tests, and every read lies past one edge of that branch.
    pub(super) fn exec_edge_scc(&self, scc: Value) -> Option<bool> {
        let p = self.x.p;
        let (b, inst) = p.blocks.iter().enumerate().find_map(|(b, block)| {
            block
                .insts
                .iter()
                .find(|i| i.write(Reg::Scc) == Some(scc))
                .map(|i| (b, i))
        })?;
        if !matches!(inst.fields, Fields::Sop { .. }) {
            return None;
        }
        let base = inst.name().replace("_saveexec", "").replace("_wrexec", "");
        if !super::sets_scc_nonzero(&base) {
            return None;
        }
        let exec_lo = inst.write(Reg::S(EXEC))?;
        let exec_hi = inst.write(Reg::S(EXEC + 1));
        if p.wave64 && exec_hi.is_none() {
            return None;
        }
        // The block ends in a branch on that EXEC.
        let Exit::Branch { taken, not_taken } = p.blocks[b].exit else {
            return None;
        };
        let branch = p.blocks[b].insts.last()?;
        let zero_taken = match branch.name() {
            "s_cbranch_execz" => true,
            "s_cbranch_execnz" => false,
            _ => return None,
        };
        if branch.read(Reg::S(EXEC)) != Some(exec_lo)
            || (p.wave64 && branch.read(Reg::S(EXEC + 1)) != exec_hi)
        {
            return None;
        }
        let (zero, nonzero) = if zero_taken {
            (taken, not_taken)
        } else {
            (not_taken, taken)
        };
        // Each successor reached only by this edge dominates what it says.
        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut index = vec![usize::MAX; p.blocks.len()];
        for (n, &blk) in order.iter().enumerate() {
            index[blk] = n;
        }
        let idom = crate::ir::idoms(&p.blocks, &order, &index);
        let dominates = |d: usize, mut x: usize| loop {
            if x == d {
                break true;
            }
            if idom[x] == x || idom[x] == usize::MAX {
                break false;
            }
            x = idom[x];
        };
        // Blocks where `scc` is read: instructions' own, and phi arguments'
        // predecessors.
        let mut reads = Vec::new();
        for (u, block) in p.blocks.iter().enumerate() {
            if block
                .insts
                .iter()
                .any(|i| i.reads.iter().any(|r| r.1 == scc))
            {
                reads.push(u);
            }
            for phi in &block.phis {
                for (k, &a) in phi.args.iter().enumerate() {
                    if a == scc {
                        reads.push(*block.preds.get(k)?);
                    }
                }
            }
        }
        if reads.is_empty() {
            return None;
        }
        for (edge, value) in [(zero, false), (nonzero, true)] {
            if p.blocks[edge].preds == [b] && reads.iter().all(|&u| dominates(edge, u)) {
                return Some(value);
            }
        }
        None
    }
}
