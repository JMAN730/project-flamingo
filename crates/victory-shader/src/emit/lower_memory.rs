//! Title-memory addressing for the emitter: 40-bit addresses, page
//! directory lookup and the flat/smem instruction forms. A child module of
//! [`super::lower`] so the general emitter stays under the repo's per-file
//! size limit (docs/asset-policy.md).

use super::*;

impl<'a> Lower<'a> {
    /// `lo`, `hi` plus the instruction's byte offset `off`, with the carry.
    pub(super) fn address_plus(&mut self, lo: V, hi: V, off: i32) -> (V, V) {
        if off == 0 {
            return (lo, hi);
        }
        let c = self.cur;
        let i32t = self.s.t.i32;
        let k = self.s.m.i32(off as u32);
        let sum = self.s.m.bin(c, Bin::Add, lo, k);
        // Up: a carry when the sum wrapped below `lo`. Down: a borrow when
        // it wrapped above.
        let wrapped = self
            .s
            .m
            .cmp(c, if off > 0 { pred::IULT } else { pred::IUGT }, sum, lo);
        let one = self.s.m.cast(c, Cast::ZExt, wrapped, i32t);
        let hi = self
            .s
            .m
            .bin(c, if off > 0 { Bin::Add } else { Bin::Sub }, hi, one);
        (sum, hi)
    }

    /// `lo`, `hi` plus the unsigned 32-bit `x`, with the carry.
    pub(super) fn address_add(&mut self, lo: V, hi: V, x: V) -> (V, V) {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let sum = self.s.m.bin(c, Bin::Add, lo, x);
        let carry = self.s.m.cmp(c, pred::IULT, sum, lo);
        let carry = self.s.m.cast(c, Cast::ZExt, carry, i32t);
        let hi = self.s.m.bin(c, Bin::Add, hi, carry);
        (sum, hi)
    }

    /// Title memory at the 40-bit address `lo`, `hi` (`shader.rs`
    /// `SLOT_MEMORY`): the handle of the raw view of the host copy holding
    /// it and the byte in that view. A page without a copy reads zeros
    /// (and drops writes) through the null view at the first slot, and its
    /// number is left for the host, for lanes that run (`guard`).
    pub(super) fn memory(&mut self, lo: V, hi: V, guard: Option<V>) -> (V, V) {
        use super::super::shader::{
            MEMORY_FAULTS, MEMORY_PAGES, SLOT_FAULTS, SLOT_MEMORY, USER_MEMORY_BASE,
        };
        let c = self.cur;
        let (i32t, res) = (self.s.t.i32, self.s.t.res_ret);
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k16, k8) = (k(self, 16), k(self, 0xFF));
        let low = self.s.m.bin(c, Bin::LShr, lo, k16);
        let high = self.s.m.bin(c, Bin::And, hi, k8);
        let high = self.s.m.bin(c, Bin::Shl, high, k16);
        let page = self.s.m.bin(c, Bin::Or, low, high);
        // Inside the directory whatever the address (lanes that do not run
        // compute one too); its last entry is never filled.
        let last = k(self, MEMORY_PAGES - 1);
        let page = self
            .s
            .op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[page, last]);
        let k3 = k(self, 3);
        let at = self.s.m.bin(c, Bin::Shl, page, k3);
        let directory = self.s.root_srv_handle(SLOT_MEMORY);
        let (iu, both, align) = (self.s.m.undef(i32t), self.s.m.i8(3), k(self, 4));
        let entry = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            res,
            139,
            &[directory, at, iu, both, align],
        );
        let (copy, start) = (self.s.m.extract(c, entry, 0), self.s.m.extract(c, entry, 1));
        let (copy, start) = (self.as_int(copy), self.as_int(start));
        // A page without a copy: its number, at a place by the number, or
        // at the sink for lanes that do not run and pages that have one.
        let zero = k(self, 0);
        let mut missing = self.s.m.cmp(c, pred::EQ, copy, zero);
        if let Some(g) = guard {
            missing = self.s.m.bin(c, Bin::And, missing, g);
        }
        let places = k(self, MEMORY_FAULTS - 1);
        let place = self.s.m.bin(c, Bin::URem, page, places);
        let one = k(self, 1);
        let place = self.s.m.bin(c, Bin::Add, place, one);
        let k2 = k(self, 2);
        let place = self.s.m.bin(c, Bin::Shl, place, k2);
        let place = self.s.m.select(c, missing, place, zero);
        let faults = self.s.root_uav_handle(SLOT_FAULTS);
        let void = self.s.t.void;
        let mask = self.s.m.i8(1);
        self.s.op(
            c,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            void,
            140,
            &[faults, place, iu, page, iu, iu, iu, mask, align],
        );
        let base = self.host_word(RootType::Other(USER_MEMORY_BASE), 0, 0);
        let slot = self.s.m.bin(c, Bin::Add, base, copy);
        let h = self.s.heap_handle(c, slot, false, true, (11 | 0x1000, 0));
        let k64k = k(self, 0xFFFF);
        let within = self.s.m.bin(c, Bin::And, lo, k64k);
        let byte = self.s.m.bin(c, Bin::Add, start, within);
        (h, byte)
    }

    /// `global_load_*`, `global_store_*` and `global_atomic_*` on title
    /// memory by address (`resource::MEMORY`): the address is the VGPR
    /// pair, each lane its own.
    pub(super) fn memory_flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat {
            dst,
            addr,
            data,
            saddr,
            offset,
            lds,
            cache,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *lds {
            return Err("a global access to LDS");
        }
        let Some(Operand::Vector { reg: areg, .. }) = *addr else {
            return Err("a global address not in VGPRs");
        };
        let word = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a global address")?;
            l.i32_of(v)
        };
        let (lo, hi) = match *saddr {
            None | Some(Operand::Scalar { code: NULL, .. }) => (word(self, 0)?, word(self, 1)?),
            // An SGPR pair plus each lane's 32-bit offset, zero-extended.
            Some(Operand::Scalar { code, .. }) => {
                let sword = |l: &mut Self, k: u8| -> Result<V, Why> {
                    let v = inst
                        .read(Reg::S(code + k))
                        .ok_or("an address not in SGPRs")?;
                    l.i32_of(v)
                };
                let (lo, hi) = (sword(self, 0)?, sword(self, 1)?);
                let voff = word(self, 0)?;
                self.address_add(lo, hi, voff)
            }
            Some(_) => return Err("a global address with a scalar base"),
        };
        let (lo, hi) = self.address_plus(lo, hi, i32::from(*offset));
        let guard = self.guard(b, i)?;
        let name = inst.name();
        let count = |rest: &str| match rest {
            "dword" => Ok(1u8),
            "dwordx2" => Ok(2),
            "dwordx3" => Ok(3),
            "dwordx4" => Ok(4),
            _ => Err(name),
        };
        // Each dword with its own directory entry: the next one may be in
        // another copy.
        if let Some(rest) = name.strip_prefix("global_load_") {
            let n = count(rest)?;
            let Some(Operand::Vector { reg, .. }) = *dst else {
                return Err("a global load result");
            };
            for k in 0..n {
                let x = self.memory_dword(lo, hi, k, guard);
                self.write(inst, Reg::V(reg + k), x, guard);
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("global_store_") {
            let n = count(rest)?;
            let d = data.as_ref().ok_or("global store data")?;
            let vals = self.vgprs(inst, d, n)?;
            for (k, &x) in (0..n).zip(&vals) {
                self.memory_store_dword(lo, hi, k, x, guard);
            }
            return Ok(());
        }
        let Some(op) = name.strip_prefix("global_atomic_") else {
            return Err(name);
        };
        let d = data.as_ref().ok_or("global atomic data")?;
        let v = self.vgprs(inst, d, 1)?[0];
        // With `glc` the value from before comes back, in the result
        // register the instruction names.
        let back = match (*dst, cache.glc) {
            (Some(Operand::Vector { reg, .. }), true) => Some(reg),
            _ => None,
        };
        let old = match back.and_then(|reg| inst.read(Reg::V(reg))) {
            Some(x) => self.i32_of(x)?,
            None => self.s.m.i32(0),
        };
        let (h, byte) = self.memory(lo, hi, guard);
        let was = self.memory_atomic(name, op, h, byte, v, guard, old)?;
        if let Some(reg) = back {
            self.write(inst, Reg::V(reg), was, None);
        }
        Ok(())
    }

    /// A `global_atomic_*` or `buffer_atomic_*` operation `op` (the name's
    /// suffix) of `v` on the dword at `byte` of the view `h` (title memory,
    /// [`Self::memory`]), in the lanes `guard` leaves; the dword from before
    /// (`old` in other lanes).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn memory_atomic(
        &mut self,
        name: &'static str,
        op: &str,
        h: V,
        byte: V,
        v: V,
        guard: Option<V>,
        old: V,
    ) -> Result<V, Why> {
        let c = self.cur;
        let (i32t, res) = (self.s.t.i32, self.s.t.res_ret);
        let iu = self.s.m.undef(i32t);
        let align = self.s.m.i32(4);
        // DXIL's atomic operations: add 0, and 1, or 2, xor 3, signed
        // minimum 4 and maximum 5, unsigned 6 and 7, exchange 8.
        let simple = match op {
            "add" => Some((0, v)),
            "sub" => {
                let zero = self.s.m.i32(0);
                Some((0, self.s.m.bin(c, Bin::Sub, zero, v)))
            }
            "and" => Some((1, v)),
            "or" => Some((2, v)),
            "xor" => Some((3, v)),
            "smin" => Some((4, v)),
            "smax" => Some((5, v)),
            "umin" => Some((6, v)),
            "umax" => Some((7, v)),
            "swap" => Some((8, v)),
            "fmin" | "fmax" => None,
            _ => return Err(name),
        };
        let at = self.begin_guarded(guard);
        let was = match simple {
            Some((code, x)) => {
                let code = self.s.m.i32(code);
                self.s.op(
                    self.cur,
                    "atomicBinOp.i32",
                    Attr::NoUnwind,
                    i32t,
                    78,
                    &[h, code, byte, iu, iu, x],
                )
            }
            None => {
                // A float minimum or maximum has no atomic operation on
                // the host: the dword is exchanged for the smaller (larger)
                // of itself and the value until no other thread got in
                // between. Each failed pass means another one's succeeded;
                // the limit only keeps a broken card from hanging.
                let head = self.cur;
                let mask = self.s.m.i8(1);
                let first = self.s.op(
                    head,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    res,
                    139,
                    &[h, byte, iu, mask, align],
                );
                let first = self.s.m.extract(head, first, 0);
                let first = self.as_int(first);
                let (pass, done) = (self.s.m.new_block(), self.s.m.new_block());
                self.s.m.terminate(head, Term::Br(pass));
                self.cur = pass;
                let seen = self.s.m.phi(pass, i32t);
                let n = self.s.m.phi(pass, i32t);
                let (f32t, fv) = (self.s.t.f32, self.as_float(v));
                let fseen = self.as_float(seen);
                // `dx.op.binary`: FMax 35, FMin 36.
                let pick = if op == "fmin" { 36 } else { 35 };
                let wanted =
                    self.s
                        .op(pass, "binary.f32", Attr::ReadNone, f32t, pick, &[fseen, fv]);
                let wanted = self.as_int(wanted);
                let found = self.s.op(
                    pass,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &[h, byte, iu, iu, seen, wanted],
                );
                let one = self.s.m.i32(1);
                let n1 = self.s.m.bin(pass, Bin::Add, n, one);
                let lost = self.s.m.cmp(pass, pred::NE, found, seen);
                let limit = self.s.m.i32(1 << 16);
                let more = self.s.m.cmp(pass, pred::IULT, n1, limit);
                let again = self.s.m.bin(pass, Bin::And, lost, more);
                self.s.m.terminate(pass, Term::CondBr(again, pass, done));
                let zero = self.s.m.i32(0);
                self.s.m.set_phi(seen, vec![(first, head), (found, pass)]);
                self.s.m.set_phi(n, vec![(zero, head), (n1, pass)]);
                self.cur = done;
                found
            }
        };
        Ok(self.end_guarded(at, &[(was, old)])[0])
    }

    /// The descriptor `rsrc` of `inst` built around title memory
    /// (`resource::MEMORY`, [`Source::Built`]): its 48-bit base (dword 0,
    /// dword 1 bits 0-15), its stride (dword 1 bits 16-29) and its size in
    /// bytes or records (dword 2, a constant or the value the code
    /// computes), from the registers the access reads.
    fn memory_descriptor(
        &mut self,
        inst: &Inst,
        rsrc: &Operand,
        src: &Source,
    ) -> Result<[V; 4], Why> {
        let Operand::Scalar { code, .. } = *rsrc else {
            return Err("a descriptor not in SGPRs");
        };
        let Source::Built {
            bits,
            words,
            records,
            ..
        } = *src
        else {
            unreachable!("a descriptor built around title memory")
        };
        // GFX10 dword 1 bits 30 and 31 swizzle; dword 3 bit 23 adds the
        // thread id to the index.
        if bits >> 30 != 0 || words[1] & 1 << 23 != 0 {
            return Err("a swizzled buffer in title memory");
        }
        let word = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst
                .read(Reg::S(code + k))
                .ok_or("a descriptor not in SGPRs")?;
            l.i32_of(v)
        };
        let (lo, w1) = (word(self, 0)?, word(self, 1)?);
        let c = self.cur;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let low16 = k(self, 0xFFFF);
        let hi = self.s.m.bin(c, Bin::And, w1, low16);
        let k16 = k(self, 16);
        let stride = self.s.m.bin(c, Bin::LShr, w1, k16);
        let bits14 = k(self, 0x3FFF);
        let stride = self.s.m.bin(c, Bin::And, stride, bits14);
        let records = match records {
            Some(v) => self.i32_of(v)?,
            None => k(self, words[0]),
        };
        Ok([lo, hi, stride, records])
    }

    /// `buffer_load_dword*`, `buffer_store_dword*` and `buffer_atomic_*`
    /// through a descriptor built around title memory (`resource::MEMORY`):
    /// the base plus `soffset`, the immediate, the offset VGPR and, with
    /// `idxen`, the index times the stride; bounded as dword 3 selects
    /// (GFX10 `OOB_SELECT`: an access out of range reads 0 and writes
    /// nothing).
    pub(super) fn memory_buffer(
        &mut self,
        inst: &Inst,
        b: usize,
        i: usize,
        src: &Source,
    ) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            rsrc,
            soffset,
            offset,
            offen,
            idxen,
            format,
            cache,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if format.is_some() || name.contains("_format_") {
            return Err("a typed access to title memory by address");
        }
        let Source::Built { words, .. } = *src else {
            unreachable!("a descriptor built around title memory")
        };
        let count = |rest: &str| match rest {
            "dword" => Ok(1u8),
            "dwordx2" => Ok(2),
            "dwordx3" => Ok(3),
            "dwordx4" => Ok(4),
            _ => Err("a byte or halfword access to title memory by address"),
        };
        let [lo, hi, stride, records] = self.memory_descriptor(inst, rsrc, src)?;
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        // The offset the range check sees: the immediate, the offset VGPR
        // and the offset register.
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let mut within = off;
        if let Some(index) = index {
            let start = self.s.m.bin(c, Bin::Mul, index, stride);
            within = self.s.m.bin(c, Bin::Add, start, off);
        }
        // shortcut: the offset from the base is taken below 4 GiB.
        let (lo, hi) = self.address_add(lo, hi, within);
        // Whether dword `k` of the access is in range.
        let oob = words[1] >> 28 & 3;
        let bound = Bound::of(oob, index.is_some());
        let in_range = |l: &mut Self, k: u32| -> V {
            let c = l.cur;
            let zero = l.s.m.i32(0);
            let index = index.unwrap_or(zero);
            let at = l.add_bytes(off, 4 * k);
            match bound {
                Bound::NotEmpty => l.s.m.cmp(c, pred::NE, records, zero),
                Bound::Bytes => l.s.m.cmp(c, pred::IULT, at, records),
                Bound::Records => l.s.m.cmp(c, pred::IULT, index, records),
                Bound::RecordAndStride => {
                    let ok = l.s.m.cmp(c, pred::IULT, index, records);
                    let inside = l.s.m.cmp(c, pred::IULT, at, stride);
                    l.s.m.bin(c, Bin::And, ok, inside)
                }
            }
        };
        let in_range = |l: &mut Self, k: u32| Some(in_range(l, k));
        let and = |l: &mut Self, a: Option<V>, b: Option<V>| match (a, b) {
            (Some(a), Some(b)) => Some(l.s.m.bin(l.cur, Bin::And, a, b)),
            (a, b) => a.or(b),
        };
        let iu = self.s.m.undef(i32t);
        // Each dword on its own: its own range check (GFX10 checks each),
        // and its own directory entry (the next one may be in another copy).
        if let Some(rest) = name.strip_prefix("buffer_load_") {
            let n = count(rest)?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for k in 0..n {
                let ok = in_range(self, u32::from(k));
                let run = and(self, ok, guard);
                let mut x = self.memory_dword(lo, hi, k, run);
                if let Some(ok) = ok {
                    let zero = self.s.m.i32(0);
                    x = self.s.m.select(self.cur, ok, x, zero);
                }
                self.write(inst, Reg::V(reg + k), x, guard);
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("buffer_store_") {
            let n = count(rest)?;
            let d = data.as_ref().ok_or("buffer store data")?;
            let vals = self.vgprs(inst, d, n)?;
            for (k, &x) in (0..n).zip(&vals) {
                let ok = in_range(self, u32::from(k));
                let run = and(self, ok, guard);
                self.memory_store_dword(lo, hi, k, x, run);
            }
            return Ok(());
        }
        let Some(op) = name.strip_prefix("buffer_atomic_") else {
            return Err(name);
        };
        let d = data.as_ref().ok_or("buffer atomic data")?;
        let Operand::Vector { reg: dreg, .. } = *d else {
            return Err("buffer atomic data");
        };
        let back = cache.glc;
        let old = match inst.read(Reg::V(dreg)) {
            Some(x) if back => self.i32_of(x)?,
            _ => self.s.m.i32(0),
        };
        let first = in_range(self, 0);
        let run = and(self, first, guard);
        let was = if op == "cmpswap" {
            // Data is (new, compare).
            let vals = self.vgprs(inst, d, 2)?;
            let (h, byte) = self.memory(lo, hi, run);
            let at = self.begin_guarded(run);
            let was = self.s.op(
                self.cur,
                "atomicCompareExchange.i32",
                Attr::NoUnwind,
                i32t,
                79,
                &[h, byte, iu, iu, vals[1], vals[0]],
            );
            self.end_guarded(at, &[(was, old)])[0]
        } else {
            let v = self.vgprs(inst, d, 1)?[0];
            let (h, byte) = self.memory(lo, hi, run);
            self.memory_atomic(name, op, h, byte, v, run, old)?
        };
        if back {
            // An access out of range returns 0.
            let was = match first {
                Some(ok) => {
                    let zero = self.s.m.i32(0);
                    self.s.m.select(self.cur, ok, was, zero)
                }
                None => was,
            };
            self.write(inst, Reg::V(dreg), was, guard);
        }
        Ok(())
    }

    /// `s_buffer_load_dword*` through a descriptor built around title
    /// memory (`resource::MEMORY`): the base plus the offsets (their low
    /// two bits cleared), each dword bounded by the size in bytes.
    pub(super) fn memory_sbuffer(&mut self, inst: &Inst, src: &Source) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            base,
            offset,
            soffset,
            ..
        } = inst.fields
        else {
            return Err("a scalar buffer load");
        };
        let count = match inst.name().strip_prefix("s_buffer_load_dword") {
            Some("") => 1u8,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar buffer access to title memory by address"),
        };
        let [lo, hi, _, records] = self.memory_descriptor(inst, &base, src)?;
        let c = self.cur;
        let not3 = self.s.m.i32(!3);
        let mut off = self.s.m.i32((offset as u32) & !3);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, &soffset)?;
            let so = self.s.m.bin(c, Bin::And, so, not3);
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let (lo, hi) = self.address_add(lo, hi, off);
        // Each dword with its own directory entry: the next one may be in
        // another copy.
        for k in 0..count {
            let x = self.memory_dword(lo, hi, k, None);
            let at = self.add_bytes(off, 4 * u32::from(k));
            let ok = self.s.m.cmp(c, pred::IULT, at, records);
            let zero = self.s.m.i32(0);
            let x = self.s.m.select(c, ok, x, zero);
            self.write(inst, Reg::S(code + k), x, None);
        }
        Ok(())
    }

    /// Dword `k` of title memory from `lo`, `hi` up, looked up in the
    /// directory on its own ([`Self::memory`]; `guard` the lanes that read
    /// it): consecutive dwords may lie in different copies, a buffer's and
    /// then the window after it, say.
    pub(super) fn memory_dword(&mut self, lo: V, hi: V, k: u8, guard: Option<V>) -> V {
        let (lo, hi) = self.address_plus(lo, hi, 4 * i32::from(k));
        let (h, byte) = self.memory(lo, hi, guard);
        let c = self.cur;
        let (i32t, res) = (self.s.t.i32, self.s.t.res_ret);
        let (iu, one, align) = (self.s.m.undef(i32t), self.s.m.i8(1), self.s.m.i32(4));
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            res,
            139,
            &[h, byte, iu, one, align],
        );
        let x = self.s.m.extract(c, r, 0);
        self.as_int(x)
    }

    /// Stores `x` as dword `k` of title memory from `lo`, `hi` up, in the
    /// lanes `run` leaves (none: every lane), the dword looked up in the
    /// directory on its own (see [`Self::memory_dword`]).
    pub(super) fn memory_store_dword(&mut self, lo: V, hi: V, k: u8, x: V, run: Option<V>) {
        let (lo, hi) = self.address_plus(lo, hi, 4 * i32::from(k));
        let (h, byte) = self.memory(lo, hi, run);
        let (i32t, void) = (self.s.t.i32, self.s.t.void);
        let (iu, one, align) = (self.s.m.undef(i32t), self.s.m.i8(1), self.s.m.i32(4));
        let at = self.begin_guarded(run);
        self.s.op(
            self.cur,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            void,
            140,
            &[h, byte, iu, x, iu, iu, iu, one, align],
        );
        self.end_guarded(at, &[]);
    }

    /// `s_load_dword*` from title memory by address (`resource::MEMORY`):
    /// the SGPR pair plus the instruction's offsets.
    pub(super) fn memory_smem(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            base: Operand::Scalar { code: base, .. },
            offset,
            soffset,
            ..
        } = inst.fields
        else {
            return Err("a scalar load from memory");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u8,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory access by address"),
        };
        let word = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst
                .read(Reg::S(base + k))
                .ok_or("an address not in SGPRs")?;
            l.i32_of(v)
        };
        let (mut lo, mut hi) = (word(self, 0)?, word(self, 1)?);
        let c = self.cur;
        let i32t = self.s.t.i32;
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            // An offset register counts up from the address.
            let so = self.operand(inst, &soffset)?;
            let sum = self.s.m.bin(c, Bin::Add, lo, so);
            let carry = self.s.m.cmp(c, pred::IULT, sum, lo);
            let carry = self.s.m.cast(c, Cast::ZExt, carry, i32t);
            hi = self.s.m.bin(c, Bin::Add, hi, carry);
            lo = sum;
        }
        // Each dword with its own directory entry: the next one may be in
        // another copy.
        let (lo, hi) = self.address_plus(lo, hi, offset);
        for k in 0..count {
            let x = self.memory_dword(lo, hi, k, None);
            self.write(inst, Reg::S(code + k), x, None);
        }
        Ok(())
    }

    /// Whether an f32 value is NaN, tested in integer arithmetic so host
    /// drivers (such as NVIDIA) do not fold `fcmp uno` away under fast-math.
    pub(super) fn is_nan_f32(&mut self, f: V) -> V {
        let c = self.cur;
        let bits = self.as_int(f);
        let mask = self.s.m.i32(0x7FFF_FFFF);
        let abs = self.s.m.bin(c, Bin::And, bits, mask);
        let inf = self.s.m.i32(0x7F80_0000);
        self.s.m.cmp(c, pred::IUGT, abs, inf)
    }
}

/// What GFX10 checks each dword of a buffer access against, by dword 3's
/// `OOB_SELECT` (an access that fails reads 0 and writes nothing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Bound {
    /// Checking disabled (2), but a buffer of no records takes nothing.
    NotEmpty,
    /// Raw (3): the dword's byte offset against the size in bytes.
    Bytes,
    /// Structured (1; 0 without an index): the index against the size in
    /// records.
    Records,
    /// Structured with an index (0): the index against the size in records
    /// and the dword's offset in its record against the stride.
    // shortcut: without an index the offset is not checked against the
    // stride (raw descriptors built with stride 0 would take nothing).
    RecordAndStride,
}

impl Bound {
    /// The check of `OOB_SELECT` `oob`, for an access with an index or
    /// without.
    pub(super) fn of(oob: u32, indexed: bool) -> Self {
        match (oob & 3, indexed) {
            (2, _) => Bound::NotEmpty,
            (3, _) => Bound::Bytes,
            (0, true) => Bound::RecordAndStride,
            _ => Bound::Records,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Bound;

    /// Each `OOB_SELECT` checks what GFX10 does: an empty buffer takes no
    /// access even with checking disabled, and mode 0 checks the offset in
    /// the record against the stride where mode 1 checks the index only.
    #[test]
    fn each_oob_select_checks_what_gfx10_checks() {
        assert_eq!(Bound::of(2, false), Bound::NotEmpty);
        assert_eq!(Bound::of(2, true), Bound::NotEmpty);
        assert_eq!(Bound::of(3, true), Bound::Bytes);
        assert_eq!(Bound::of(0, true), Bound::RecordAndStride);
        assert_eq!(Bound::of(1, true), Bound::Records);
        assert_eq!(Bound::of(1, false), Bound::Records);
        assert_eq!(Bound::of(0, false), Bound::Records);
    }
}
