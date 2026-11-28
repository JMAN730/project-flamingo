//! Image and buffer memory instructions of [`Lower`], a child module so
//! `lower.rs` stays under the asset policy's size limit.

use super::*;

impl Lower<'_> {
    pub(super) fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            rsrc,
            dmask,
            dim,
            unorm,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if name.starts_with("image_bvh") {
            return self.bvh_miss(inst, b, i);
        }
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        let no_channels = *dmask == 0;
        // A load or sample with no channel enabled returns the first.
        let dmask = if *dmask == 0 { &1 } else { dmask };
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let (kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only.
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (5, 3)
            }
            4 => (6, 2),
            5 => (7, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        // Address VGPRs in order.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        // Full 32-bit address VGPR at `reg_idx` (offset, bias, z-compare).
        let addr_u32 = |l: &mut Self, reg_idx: usize| -> Result<V, Why> {
            let r = *regs.get(reg_idx).ok_or("an image address")?;
            #[cfg(test)]
            l.s.image_address_reads.push(reg_idx);
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            l.i32_of(v)
        };
        // With a16, two 16-bit components per VGPR starting at `base_reg`, low half first.
        let addr_halves =
            |l: &mut Self, base_reg: usize, k: usize, packed: bool| -> Result<V, Why> {
                let x = addr_u32(l, base_reg + if packed { k / 2 } else { k })?;
                if !packed {
                    return Ok(x);
                }
                let c = l.cur;
                let (sh, mask) = (l.s.m.i32(16 * (k as u32 % 2)), l.s.m.i32(0xFFFF));
                let x = l.s.m.bin(c, Bin::LShr, x, sh);
                Ok(l.s.m.bin(c, Bin::And, x, mask))
            };
        let addr_packed = |l: &mut Self, base_reg: usize, k: usize| -> Result<V, Why> {
            addr_halves(l, base_reg, k, *a16)
        };
        let addr_val = |l: &mut Self, k: usize| -> Result<V, Why> { addr_packed(l, 0, k) };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked image")?;
        let store = name.starts_with("image_store");
        let uav = store
            || self
                .used_kind(src)
                .is_some_and(|u| u.kind == ResourceType::Uav);
        let props = (kind | if uav { 0x1000 } else { 0 }, 0x409);
        if let Some(op) = name.strip_prefix("image_atomic_") {
            // GLC returns the old value. DXIL implements subtraction by adding
            // the two's complement of the operand at the same width.
            let code = match op {
                "add" | "sub" => 0,
                "and" => 1,
                "or" => 2,
                "xor" => 3,
                "smin" => 4,
                "smax" => 5,
                "umin" => 6,
                "umax" => 7,
                "swap" => 8,
                "cmpswap" => u32::MAX,
                _ => return Err(name),
            };
            let Fields::Image { cache, .. } = &inst.fields else {
                unreachable!()
            };
            // (a16 packs only the coordinates, which `addr_val` unpacks.)
            if *d16 {
                return Err("a 16-bit image atomic");
            }
            // Compare-exchange consumes a replacement followed by a comparand.
            let compare = code == u32::MAX;
            let wide = match (compare, *dmask) {
                (false, 1) | (true, 3) => false,
                (false, 3) | (true, 15) => true,
                _ => return Err("an image atomic data mask"),
            };
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image atomic data");
            };
            let width = if wide { 2 } else { 1 };
            let vals = self.vgprs(inst, data, if compare { 2 * width } else { width })?;
            let (ty, component) = if wide {
                self.s.flags |= flags::INT64 | flags::ATOMIC_TYPED64 | flags::ATOMIC_HEAP64;
                (self.s.m.int(64), 0x107) // one uint64_t in an R32G32_UINT view
            } else {
                (self.s.t.i32, 0x105)
            };
            let original = if wide {
                self.join64(vals[0], vals[1])
            } else {
                vals[0]
            };
            let value = if op == "sub" {
                let zero = self.s.m.int_const(ty, 0);
                self.s.m.bin(c, Bin::Sub, zero, original)
            } else {
                original
            };
            let h = self.handle(site, src, false, (kind | 0x1000, component))?;
            let mut args = vec![h];
            let (intrinsic, opcode) = if compare {
                let comparand = if wide {
                    self.join64(vals[2], vals[3])
                } else {
                    vals[1]
                };
                args.extend(cs);
                args.extend([comparand, value]);
                let intrinsic = if wide {
                    "atomicCompareExchange.i64"
                } else {
                    "atomicCompareExchange.i32"
                };
                (intrinsic, 79)
            } else {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                (
                    if wide {
                        "atomicBinOp.i64"
                    } else {
                        "atomicBinOp.i32"
                    },
                    78,
                )
            };
            let at = self.begin_guarded(guard);
            let r = self
                .s
                .op(self.cur, intrinsic, Attr::NoUnwind, ty, opcode, &args);
            if cache.glc {
                let old = self.end_guarded(at, &[(r, original)])[0];
                if wide {
                    let (lo, hi) = self.split64(old);
                    self.write(inst, Reg::V(reg), lo, None);
                    self.write(inst, Reg::V(reg + 1), hi, None);
                } else {
                    self.write(inst, Reg::V(reg), old, None);
                }
            } else {
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if store {
            if name != "image_store" {
                return Err("an image store to a mip level");
            }
            if no_channels {
                return Err("an image store of no channels");
            }
            // shortcut: stores are taken to write all four channels of a
            // float format (the console converts by the view's format).
            // The channels `dmask` selects come from consecutive data
            // registers (halves with d16); the others are written as zero
            // (RDNA2 ISA, MIMG stores).
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image store data");
            };
            let mut vals = Vec::new();
            let mut k = 0u8;
            for ch in 0..4u8 {
                if dmask >> ch & 1 == 0 {
                    vals.push(self.s.m.f32(0));
                    continue;
                }
                // d16: two f16 channels per VGPR, low half first.
                let r = if *d16 { k / 2 } else { k };
                let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
                vals.push(if *d16 {
                    let mut x = self.i32_of(v)?;
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                } else {
                    self.float_of(v)?
                });
                k += 1;
            }
            let h = self.handle(site, src, false, props)?;
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            args.extend(cs);
            args.extend(vals);
            args.push(self.s.m.i8(0xF));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "textureStore.f32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // GFX10.3 `_g16` opcodes pack gradients into 16-bit pairs; `a16` (coordinates, LOD,
        // clamp) is independent of it.
        let (base_name, g16) = match name.strip_suffix("_g16") {
            Some(base) => (base, true),
            None => (name, false),
        };
        let (op, sample) = match base_name {
            "image_sample" | "image_sample_cl" | "image_sample_o" => (60, true),
            "image_sample_lz" | "image_sample_l" | "image_sample_lz_o" | "image_sample_l_o" => {
                (62, true)
            }
            "image_sample_b" | "image_sample_b_cl" | "image_sample_b_o" | "image_sample_b_cl_o" => {
                (61, true)
            }
            "image_sample_c_lz" | "image_sample_c_lz_o" => (65, true),
            "image_sample_d" | "image_sample_d_cl" | "image_sample_d_o" | "image_sample_d_cl_o" => {
                (63, true)
            }
            "image_get_lod" if self.s.pixel => (81, true),
            // Implicit derivatives: pixel shaders only.
            "image_sample_c" | "image_sample_c_cl" | "image_sample_c_o" if self.s.pixel => {
                (64, true)
            }
            "image_gather4" | "image_gather4_lz" | "image_gather4_lz_o" | "image_gather4_l"
            | "image_gather4_l_o" => (73, true),
            "image_gather4_c_lz" | "image_gather4_c_lz_o" => (74, true),
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            _ => return Err(name),
        };
        if op == 72 {
            // Width, height, depth or layers, levels (per dmask) of the level
            // in the first address: DXIL's dimensions. Unused size channels
            // read 0 (RDNA); UAVs report one mip level (VIC-BT-038 / #440).
            let h = self.handle(site, src, false, props)?;
            let mip = if uav {
                self.s.m.undef(self.s.t.i32)
            } else {
                addr_val(self, 0)?
            };
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let one = self.s.m.i32(1);
            let zero = self.s.m.i32(0);
            let present = match kind {
                1 => [true, false, false, true],
                2 | 5 | 6 => [true, true, false, true],
                _ => [true, true, true, true],
            };
            let Operand::Vector { reg, .. } = *data else {
                return Err("an image result");
            };
            for (k, ch) in (0..4u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                let x = if ch == 3 && uav {
                    one
                } else if !present[ch as usize] {
                    zero
                } else {
                    self.s.m.extract(c, d, ch)
                };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(site, src, false, props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        // The other choices of a sampler chosen by control flow: whether the
        // path taken brought it, and its handle (see `sampler_choices`).
        let mut variants: Vec<(V, V)> = Vec::new();
        let r = if sample {
            let ss = self
                .descriptors
                .get(&(site, true))
                .copied()
                .ok_or("an untracked sampler")?;
            // Comparison: the reference value is the first address after
            // the bias, and the sampler is a comparison sampler.
            let cmp = matches!(op, 64 | 65 | 74);
            if cmp && (*a16 || dmask.count_ones() != 1) {
                return Err("a comparison sample of 16-bit coordinates or several channels");
            }
            let sampler_props = (14 | if cmp { 0x8000 } else { 0 }, 0);
            let (sh, others) = match self.sampler_choices(ss, sampler_props)? {
                Some(choices) => choices,
                None => (self.handle(site, ss, true, sampler_props)?, Vec::new()),
            };
            if !others.is_empty() && op == 81 {
                return Err("a level query with a sampler chosen by control flow");
            }
            variants = others;
            // Texel offsets (`_o`) come first: x in bits 0-5, y in 8-13,
            // z in 16-21, signed (programmable offsets: gathers only).
            // Offset, bias and z-compare are always full 32-bit VGPRs even
            // with a16. Token split (not ends_with) so `_o_g16` / `_cl_o` work.
            let offset_word = base_name.split('_').any(|t| t == "o");
            let gather = matches!(op, 73 | 74);
            if offset_word && kind == 5 {
                return Err("a texel offset on a cube");
            }
            if gather && !matches!(kind, 2 | 5 | 7) {
                return Err("a gather of a view other than 2D or cube");
            }
            let has_cl = base_name.split('_').any(|t| t == "cl");
            let first = usize::from(offset_word) + usize::from(op == 61) + usize::from(cmp);
            // Coordinates the size divides (not the array slice); explicit
            // gradients (x then y) come before the coordinates and remain
            // 32-bit floats on GFX10.3 unless the opcode is a `_g16` form.
            let spatial = coords - usize::from(kind == 6 || kind == 7);
            let gradient_dims = if kind == 5 { 2 } else { spatial };
            let deriv_dwords = match (op == 63, g16) {
                (false, _) => 0,
                (true, false) => gradient_dims,
                (true, true) => gradient_dims.div_ceil(2),
            };
            let coord_base = first + 2 * deriv_dwords;
            let mut cs = vec![fu; 4];
            // With a16, sampling coordinates, bias (low half), LOD and clamp are f16.
            let to_float = |l: &mut Self, x: V| {
                if *a16 {
                    let f = l.s.t.f32;
                    l.s.op(l.cur, "legacyF16ToF32", Attr::ReadNone, f, 131, &[x])
                } else {
                    l.as_float(x)
                }
            };
            // Bias is a whole VGPR even with a16; only its low half is the f16.
            // The bias follows the offset word.
            let bias_float = |l: &mut Self| -> Result<V, Why> {
                let bias = addr_u32(l, usize::from(offset_word))?;
                if *a16 {
                    let mask = l.s.m.i32(0xFFFF);
                    let lo = l.s.m.bin(l.cur, Bin::And, bias, mask);
                    Ok(to_float(l, lo))
                } else {
                    Ok(l.as_float(bias))
                }
            };
            let grad_float = |l: &mut Self, x: V| {
                if g16 {
                    let f = l.s.t.f32;
                    l.s.op(l.cur, "legacyF16ToF32", Attr::ReadNone, f, 131, &[x])
                } else {
                    l.as_float(x)
                }
            };
            if kind == 5 && *unorm {
                return Err("a cube sample with texel coordinates");
            }
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                let x = addr_packed(self, coord_base, k)?;
                *slot = to_float(self, x);
            }
            let face = cs[2];
            if kind == 5 {
                let dir = self.cube_direction(cs[0], cs[1], face);
                cs[..3].copy_from_slice(&dir);
            }
            if *unorm {
                // Texel coordinates: divided by the size of level 0.
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let z = self.s.m.i32(0);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                    let n = self.s.m.extract(c, d, k as u32);
                    let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                    *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
                }
            }
            if op == 81 {
                // The clamped then the unclamped level of detail.
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let f32t = self.s.t.f32;
                for (k, ch) in (0..2u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                    let clamped = self.s.m.bool(ch == 0);
                    let lod = self.s.op(
                        c,
                        "calculateLOD.f32",
                        Attr::ReadOnly,
                        f32t,
                        81,
                        &[h, sh, cs[0], cs[1], cs[2], clamped],
                    );
                    let x = self.as_int(lod);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let offsets = if kind == 5 {
                0
            } else {
                (coords - usize::from(kind == 6 || kind == 7)).min(3)
            };
            let zero = self.s.m.i32(0);
            let mut os = vec![iu; 3];
            for o in os.iter_mut().take(offsets) {
                *o = zero;
            }
            let mut args = vec![h, sh];
            args.extend(&cs);
            let clamp = if has_cl {
                let x = addr_packed(self, coord_base, coords)?;
                to_float(self, x)
            } else {
                fu
            };
            if offset_word && !gather {
                // DXIL takes immediate offsets in -8..=7; the console's are
                // 6-bit and may be computed.
                let w = addr_u32(self, 0)?;
                let w = self
                    .s
                    .m
                    .int_value(w)
                    .ok_or("a sample offset that is not constant")? as u32;
                for (k, o) in os.iter_mut().enumerate().take(offsets) {
                    let x = (w >> (8 * k) & 0x3F) as i32;
                    let x = if x >= 32 { x - 64 } else { x };
                    if !(-8..=7).contains(&x) {
                        return Err("a sample offset past -8..7");
                    }
                    *o = self.s.m.i32(x as u32);
                }
            }
            match op {
                60 | 61 if !self.s.pixel => {
                    // Outside pixel shaders the console still takes the level
                    // from each quad of lanes (lane 1 minus lane 0 across,
                    // lane 2 minus lane 0 down); DXIL before validator 1.8
                    // has no implicit derivatives there, so the quad's
                    // differences become explicit gradients. A bias scales
                    // them by 2^bias, the same level for isotropic filtering.
                    self.cross_lane(true)?;
                    let scale = if op == 61 {
                        let bias = bias_float(self)?;
                        Some(self.unary(21, bias))
                    } else {
                        None
                    };
                    let g = self.quad_gradients(&cs[..spatial], scale);
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                60 => {
                    args.extend(&os);
                    args.push(clamp);
                    self.s
                        .op(c, "sample.f32", Attr::ReadOnly, self.s.t.res_ret, 60, &args)
                }
                61 => {
                    let bias = bias_float(self)?;
                    args.extend(&os);
                    args.extend([bias, clamp]);
                    self.s.op(
                        c,
                        "sampleBias.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        61,
                        &args,
                    )
                }
                63 => {
                    let mut g = [[fu; 3]; 2];
                    for (axis, derivative) in g.iter_mut().enumerate() {
                        for (k, dst) in derivative[..gradient_dims].iter_mut().enumerate() {
                            let value = addr_halves(self, first + axis * deriv_dwords, k, g16)?;
                            *dst = grad_float(self, value);
                        }
                        if kind == 5 {
                            // A unit-major direction maps back to face coordinates with a
                            // factor of 1/2. Its derivative has a zero major component.
                            let two = self.s.m.f32(2.0f32.to_bits());
                            let du = self.s.m.bin(c, Bin::Mul, derivative[0], two);
                            let dv = self.s.m.bin(c, Bin::Mul, derivative[1], two);
                            let zero = self.s.m.f32(0);
                            derivative.copy_from_slice(&self.cube_axes(du, dv, zero, face));
                        }
                    }
                    args.extend(&os);
                    args.extend(g.into_iter().flatten());
                    args.push(clamp);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                64 | 65 => {
                    let reference = addr_u32(self, first - 1)?;
                    let reference = self.as_float(reference);
                    args.extend(&os);
                    args.push(reference);
                    let name = if op == 64 {
                        args.push(clamp);
                        "sampleCmp.f32"
                    } else {
                        "sampleCmpLevelZero.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
                62 => {
                    let lod = if matches!(name, "image_sample_l" | "image_sample_l_o") {
                        let x = addr_packed(self, coord_base, coords)?;
                        to_float(self, x)
                    } else {
                        self.s.m.f32(0)
                    };
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        c,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                _ if op == 73
                    && name.starts_with("image_gather4_l")
                    && !name.starts_with("image_gather4_lz") =>
                {
                    if *unorm || *d16 {
                        return Err("a gather with a level of texel coordinates or 16-bit data");
                    }
                    let lod = addr_packed(self, coord_base, coords)?;
                    let lod = to_float(self, lod);
                    let offset = if offset_word {
                        Some(addr_u32(self, 0)?)
                    } else {
                        None
                    };
                    let texels = self.gather_at_level(h, kind, &cs, lod, offset, *dmask)?;
                    let Operand::Vector { reg, .. } = *data else {
                        return Err("an image result");
                    };
                    for (k, x) in texels.into_iter().enumerate() {
                        let x = self.as_int(x);
                        self.write(inst, Reg::V(reg + k as u8), x, guard);
                    }
                    return Ok(());
                }
                _ => {
                    if name == "image_gather4" {
                        return Err("a gather with implicit level");
                    }
                    if dmask.count_ones() != 1 {
                        return Err("a gather of several channels");
                    }
                    if offset_word {
                        let w = addr_u32(self, 0)?;
                        for (k, at) in [(0, 26), (1, 18)] {
                            // Bits at..at+5 sign-extended: shift up, then down.
                            let left = self.s.m.i32(at);
                            let t = self.s.m.bin(c, Bin::Shl, w, left);
                            let k26 = self.s.m.i32(26);
                            os[k] = self.s.m.bin(c, Bin::AShr, t, k26);
                        }
                    }
                    args.extend(&os[..2]);
                    args.push(self.s.m.i32(dmask.trailing_zeros()));
                    if op == 74 {
                        let reference = addr_u32(self, first - 1)?;
                        args.push(self.as_float(reference));
                        self.s.op(
                            c,
                            "textureGatherCmp.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            74,
                            &args,
                        )
                    } else {
                        self.s.op(
                            c,
                            "textureGather.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            73,
                            &args,
                        )
                    }
                }
            }
        } else {
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let mip = if name == "image_load_mip" {
                addr_val(self, coords)?
            } else if uav {
                // Our declarations have four channels.
                self.s.flags |= flags::TYPED_UAV_LOADS;
                iu
            } else {
                self.s.m.i32(0)
            };
            // VIC-BT-017: d16 image_load packs two channels per VGPR and must
            // keep an odd trailing high half; UINT/SINT texels saturate to 16
            // bits via textureLoad.i32 rather than going through f16.
            if *d16 {
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let channels: Vec<u32> = (0..4).filter(|ch| dmask >> ch & 1 != 0).collect();
                let mut args = vec![h, mip];
                args.extend(&cs);
                args.extend([iu, iu, iu]);
                let r_f = self.s.op(
                    c,
                    "textureLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    66,
                    &args,
                );
                // U32 component type, four channels (matches 0x409 for F32).
                let h_i = self.handle(site, src, false, (props.0, 0x405))?;
                let i32t = self.s.t.i32;
                let res_i = self
                    .s
                    .m
                    .named_struct("dx.types.ResRet.i32", vec![i32t, i32t, i32t, i32t, i32t]);
                let mut args_i = vec![h_i, mip];
                args_i.extend(&cs);
                args_i.extend([iu, iu, iu]);
                let r_i = self
                    .s
                    .op(c, "textureLoad.i32", Attr::ReadOnly, res_i, 66, &args_i);
                // SQ_IMG_RSRC_WORD1.FORMAT at bits 20..28 (GFX10).
                let (is_uint, is_sint) = match rsrc {
                    Operand::Scalar { code, dwords } if *dwords >= 2 => {
                        let w1 = inst
                            .read(Reg::S(code + 1))
                            .ok_or("an unread image descriptor")?;
                        let w1 = self.i32_of(w1)?;
                        let sh = self.s.m.i32(20);
                        let mask = self.s.m.i32(0x1FF);
                        let fmt = self.s.m.bin(c, Bin::LShr, w1, sh);
                        let fmt = self.s.m.bin(c, Bin::And, fmt, mask);
                        self.format_is_uint_sint(fmt)
                    }
                    _ => (
                        self.s.m.int_const(self.s.t.i1, 0),
                        self.s.m.int_const(self.s.t.i1, 0),
                    ),
                };
                let mut halves = Vec::new();
                for ch in channels {
                    let f = self.s.m.extract(c, r_f, ch);
                    let f16 = self.f16_nearest(f);
                    let i = self.s.m.extract(c, r_i, ch);
                    let u = self.saturate_u16(i);
                    let s = self.saturate_i16(i);
                    let int = self.s.m.select(c, is_sint, s, u);
                    let is_int = self.s.m.bin(c, Bin::Or, is_uint, is_sint);
                    halves.push(self.s.m.select(c, is_int, int, f16));
                }
                let k16 = self.s.m.i32(16);
                let n = halves.len();
                let mut xs = Vec::new();
                for (k, pair) in halves.chunks(2).enumerate() {
                    let packed = match *pair {
                        [lo, hi] => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, lo, hi)
                        }
                        [lo] => lo,
                        _ => unreachable!("chunks of one or two"),
                    };
                    let reg = reg + k as u8;
                    let x = if k * 2 + 1 == n {
                        #[cfg(test)]
                        {
                            self.s.d16_kept_high_halves += 1;
                        }
                        self.low_half(inst, reg, packed)
                    } else {
                        packed
                    };
                    xs.push((reg, x));
                }
                for (reg, x) in xs {
                    self.write(inst, Reg::V(reg), x, guard);
                }
                return Ok(());
            }
            let mut args = vec![h, mip];
            args.extend(&cs);
            args.extend([iu, iu, iu]);
            self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            )
        };
        // The sampling repeated with each other choice of the sampler.
        let mut repeated = Vec::new();
        if !variants.is_empty() {
            let call = self
                .s
                .sample_call
                .clone()
                .filter(|call| call.result == r && call.args.get(1).is_some())
                .ok_or("a sample with a sampler chosen by control flow")?;
            for (taken, handle) in variants {
                let mut args = call.args.clone();
                args[1] = handle;
                let again = self.s.op(
                    call.block,
                    &call.name,
                    call.attr,
                    call.ret,
                    call.opcode,
                    &args,
                );
                repeated.push((taken, again));
            }
        }
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        let channels: Vec<u32> = if matches!(op, 73 | 74) {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let mut x = self.s.m.extract(c, r, ch);
            for &(taken, again) in &repeated {
                let y = self.s.m.extract(c, again, ch);
                x = self.s.m.select(c, taken, y, x);
            }
            xs.push(if *d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                self.f16_nearest(x)
            } else {
                self.as_int(x)
            });
        }
        if *d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// `image_gather4_l`: DXIL's gathers take no level, so the four texels
    /// a bilinear fetch at level `lod` reads are loaded from that level, in
    /// the gather's order: (i0, j1), (i1, j1), (i1, j0), (i0, j0). Channel
    /// `dmask` of each, as floats.
    ///
    /// The level is the nearest one, clamped to the view's levels (a
    /// sampler with point mip filtering; the GPU culling that uses this
    /// passes whole levels of a depth pyramid).
    // shortcut: texels past the edge are clamped (clamp-to-edge
    // addressing, what depth pyramids use); wrap and border samplers would
    // read other texels.
    pub(super) fn gather_at_level(
        &mut self,
        h: V,
        kind: u32,
        cs: &[V],
        lod: V,
        offset: Option<V>,
        dmask: u8,
    ) -> Result<[V; 4], Why> {
        if dmask.count_ones() != 1 {
            return Err("a gather of several channels");
        }
        // 2D (2) or 2D array (7) views only.
        if !matches!(kind, 2 | 7) {
            return Err("a gather with a level of a view other than 2D");
        }
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let dims_t = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let (zero, one) = (self.s.m.i32(0), self.s.m.i32(1));
        let half = self.s.m.f32(0.5f32.to_bits());
        let clamp = |l: &mut Self, x: V, top: V| {
            let lo = l.s.m.cmp(c, pred::SLT, x, zero);
            let x = l.s.m.select(c, lo, zero, x);
            let hi = l.s.m.cmp(c, pred::SGT, x, top);
            l.s.m.select(c, hi, top, x)
        };
        // Nearest whole level, within the view.
        let base = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims_t, 72, &[h, zero]);
        let levels = self.s.m.extract(c, base, 3);
        let last = self.s.m.bin(c, Bin::Sub, levels, one);
        let level = self.s.m.bin(c, Bin::Add, lod, half);
        let level = self.unary(27, level);
        let level = self.s.m.cast(c, Cast::FpToSi, level, i32t);
        let level = clamp(self, level, last);
        let size = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims_t, 72, &[h, level]);
        // The texel below and left of the sample point: floor(u * size - 0.5),
        // then the gather's programmable offset (6-bit signed x, then y).
        // Before it becomes an integer it is kept within 64 texels of the
        // view (FMax/FMin, which also turn NaN into the low edge): past
        // that, the offset (-32..31) and the clamp to the edge give the
        // same texel, and a huge or infinite coordinate can neither fall
        // outside i32 nor wrap round when the offset is added.
        let margin = self.s.m.f32(64f32.to_bits());
        let below = self.s.m.f32((-64f32).to_bits());
        let mut lows = [zero; 2];
        let mut tops = [zero; 2];
        for (k, (low, top)) in lows.iter_mut().zip(&mut tops).enumerate() {
            let n = self.s.m.extract(c, size, k as u32);
            *top = self.s.m.bin(c, Bin::Sub, n, one);
            let n = self.s.m.cast(c, Cast::UiToFp, n, f32t);
            let x = self.s.m.bin(c, Bin::Mul, cs[k], n);
            let x = self.s.m.bin(c, Bin::Sub, x, half);
            let x = self.unary(27, x);
            let x = self.binary(35, x, below);
            let above = self.s.m.bin(c, Bin::Add, n, margin);
            let x = self.binary(36, x, above);
            *low = self.s.m.cast(c, Cast::FpToSi, x, i32t);
            if let Some(w) = offset {
                let left = self.s.m.i32(if k == 0 { 26 } else { 18 });
                let t = self.s.m.bin(c, Bin::Shl, w, left);
                let k26 = self.s.m.i32(26);
                let o = self.s.m.bin(c, Bin::AShr, t, k26);
                *low = self.s.m.bin(c, Bin::Add, *low, o);
            }
        }
        // The array slice: the nearest, within the view.
        let slice = if kind == 7 {
            let s = self.s.m.bin(c, Bin::Add, cs[2], half);
            let s = self.unary(27, s);
            let s = self.s.m.cast(c, Cast::FpToSi, s, i32t);
            let layers = self.s.m.extract(c, base, 2);
            let top = self.s.m.bin(c, Bin::Sub, layers, one);
            clamp(self, s, top)
        } else {
            self.s.m.undef(i32t)
        };
        let iu = self.s.m.undef(i32t);
        let channel = dmask.trailing_zeros();
        let texel = |l: &mut Self, dx: u32, dy: u32| {
            let at = |l: &mut Self, k: usize, d: u32| {
                let d = l.s.m.i32(d);
                let x = l.s.m.bin(c, Bin::Add, lows[k], d);
                clamp(l, x, tops[k])
            };
            let (x, y) = (at(l, 0, dx), at(l, 1, dy));
            let r = l.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                66,
                &[h, level, x, y, slice, iu, iu, iu],
            );
            l.s.m.extract(c, r, channel)
        };
        Ok([
            texel(self, 0, 1),
            texel(self, 1, 1),
            texel(self, 1, 0),
            texel(self, 0, 0),
        ])
    }

    /// The direction a cube sample's face coordinates stand for: the
    /// console takes s and t in [1, 2] (`v_cubesc`/`v_cubetc` over
    /// `|v_cubema|`, plus 1.5) and the face id (`v_cubeid`), which the
    /// host's cube sampling turns back into the same texel.
    // shortcut: the view is taken as a cube, not a cube array (whose face
    // operand adds 8 × the slice).
    pub(super) fn cube_direction(&mut self, s: V, t: V, face: V) -> [V; 3] {
        let c = self.cur;
        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
        let (two, three) = (k(self, 2.0), k(self, 3.0));
        let u = self.s.m.bin(c, Bin::Mul, s, two);
        let u = self.s.m.bin(c, Bin::Sub, u, three);
        let v = self.s.m.bin(c, Bin::Mul, t, two);
        let v = self.s.m.bin(c, Bin::Sub, v, three);
        let one = k(self, 1.0);
        self.cube_axes(u, v, one, face)
    }

    /// Rotate the two face axes and the major axis into cube direction space.
    pub(super) fn cube_axes(&mut self, u: V, v: V, major: V, face: V) -> [V; 3] {
        let c = self.cur;
        let z = self.s.m.f32(0x8000_0000);
        let nu = self.s.m.bin(c, Bin::Sub, z, u);
        let nv = self.s.m.bin(c, Bin::Sub, z, v);
        let negative_major = self.s.m.bin(c, Bin::Sub, z, major);
        let f = self.s.m.cast(c, Cast::FpToUi, face, self.s.t.i32);
        // Per face +x, -x, +y, -y, +z, -z.
        let table = [
            [major, negative_major, u, u, u, nu],
            [nv, nv, major, negative_major, nv, nv],
            [nu, u, v, nv, major, negative_major],
        ];
        table.map(|row| {
            let mut r = row[0];
            for (n, &x) in row.iter().enumerate().skip(1) {
                let kn = self.s.m.i32(n as u32);
                let is = self.s.m.cmp(c, pred::EQ, f, kn);
                r = self.s.m.select(c, is, x, r);
            }
            r
        })
    }

    pub(super) fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if name == "buffer_gl0_inv" || name == "buffer_gl1_inv" {
            // Console cache coherence: gl0 writes back and invalidates the
            // per-CU shader L0, gl1 invalidates GL1, so one wavefront's
            // writes become visible to others. D3D12 already guarantees UAV
            // coherence across wavefronts and dispatches, DXIL has no
            // equivalent instruction, and this project's host-visible
            // writeback comes from command-stream GpuWrite ranges rather
            // than shader cache hints. Ordering is kept by the surrounding
            // s_waitcnt/s_barrier, which the translator does lower.
            return Ok(());
        }
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err(name);
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let atomic = match name.strip_prefix("buffer_atomic_") {
            None => None,
            Some("add") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap") => Some(u32::MAX),
            Some(_) => return Err(name),
        };
        // These unformatted loads return the same register bits as the
        // corresponding single-channel GFX10 integer formats.
        let narrow_format = match name {
            "buffer_load_ubyte" => Some(5),   // 8_UINT
            "buffer_load_sbyte" => Some(6),   // 8_SINT
            "buffer_load_ushort" => Some(11), // 16_UINT
            "buffer_load_sshort" => Some(12), // 16_SINT
            _ => None,
        };
        // Byte and halfword stores: (bits, from the high half of the VGPR).
        let narrow_store = match name {
            _ if super::super::fixes() & super::super::FIX_NARROW_STORE == 0 => None,
            "buffer_store_byte" => Some((8, false)),
            "buffer_store_byte_d16_hi" => Some((8, true)),
            "buffer_store_short" => Some((16, false)),
            "buffer_store_short_d16_hi" => Some((16, true)),
            _ => None,
        };
        // d16: two 16-bit channels per VGPR, low half first (MUBUF and MTBUF).
        let d16 = name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let sub_name = name.trim_end_matches("_hi").trim_end_matches("_d16");
        let sub = match sub_name {
            "buffer_load_ubyte" | "buffer_load_sbyte" | "buffer_store_byte" => Some(1),
            "buffer_load_ushort" | "buffer_load_sshort" | "buffer_load_short"
            | "buffer_store_short" => Some(2),
            _ => None,
        };
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() || sub.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
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
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        if matches!(src, Source::Built { address, .. } if address.param == MEMORY) {
            return self.memory_buffer(inst, b, i, src);
        }
        // A descriptor made raw has no stride: only byte addressing reads it
        // as the console does.
        if (index.is_some() || typed) && self.x.res.raw.contains(&site) {
            return Err("an indexed or typed access through a descriptor made raw");
        }
        let used = self.used_kind(src);
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            if sub.is_some() {
                return Err("a byte or halfword load through a constant or structured buffer");
            }
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let x = self.cbuffer_dword(h, a);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let uav = store
            || atomic.is_some()
            || used.is_some_and(|u| u.kind == ResourceType::Uav)
            || (index.is_some() && !typed && used.is_none() && !self.buffer_loads_only(src));
        // A root descriptor is a raw view: an indexed access through the
        // descriptor the code builds around it is at index * stride + offset.
        let root_index = super::super::fixes() & super::super::FIX_ROOT_INDEX != 0;
        let (index, off) = match (index, self.root_view_stride(src)) {
            (Some(i), Some(stride)) if !typed && root_index => {
                let stride = self.s.m.i32(stride);
                let start = self.s.m.bin(c, Bin::Mul, i, stride);
                (None, self.s.m.bin(c, Bin::Add, start, off))
            }
            (index, _) => (index, off),
        };
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        let flat_stride = if index.is_none() && (!typed || format.is_some()) {
            self.table_buffer_stride(src, used, uav)
        } else {
            None
        };
        if sub.is_some() && flat_stride.is_none() {
            if narrow_format.is_none() && narrow_store.is_none() {
                return Err(name);
            }
            if used.is_some_and(|u| u.stride > 1) {
                return Err("a byte or halfword load through a constant or structured buffer");
            }
        }
        if let Some(op) = atomic {
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
            // A zero-stride view is raw: the index drops out.
            let indexed = match index {
                Some(index) => self.structured_index(src, uav, index)?,
                None => None,
            };
            let stride = indexed.map_or(flat_stride, |(_, stride)| Some(stride));
            let index = indexed.map(|(index, _)| index);
            let h = self.handle(site, src, false, Self::buffer_props(stride, uav))?;
            let (index, off) = match (index, stride) {
                (Some(index), _) => (index, off),
                (None, Some(stride)) => self.structured_coords(off, stride),
                (None, None) => (off, iu),
            };
            let coords = [index, off, iu];
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if returns => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let at = self.begin_guarded(guard);
            let i32t = self.s.t.i32;
            let r = if op == u32::MAX {
                // cmpswap: data is (new, compare).
                let mut args = vec![h];
                args.extend(coords);
                args.extend([vals[1], vals[0]]);
                self.s.op(
                    self.cur,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &args,
                )
            } else {
                let mut args = vec![h, self.s.m.i32(op)];
                args.extend(coords);
                args.push(vals[0]);
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
            };
            let r = self.end_guarded(at, &[(r, old)])[0];
            if returns {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        // A view of known stride takes the structured store below.
        if let Some((bits, high)) = narrow_store.filter(|_| flat_stride.is_none()) {
            // DXIL stores whole dwords: clear the byte or halfword in its
            // dword, then set it, each with an atomic so that lanes writing
            // the rest of that dword keep their bytes.
            // shortcut: a halfword at an odd address is written at the even
            // one below it (the console splits it across bytes).
            let d = data.as_ref().ok_or("buffer store data")?;
            let mut x = self.vgprs(inst, d, 1)?[0];
            if high {
                let k16 = self.s.m.i32(16);
                x = self.s.m.bin(c, Bin::LShr, x, k16);
            }
            let ones = self.s.m.i32((1u32 << bits) - 1);
            x = self.s.m.bin(c, Bin::And, x, ones);
            let within = self.s.m.i32(if bits == 8 { 3 } else { 2 });
            let byte = self.s.m.bin(c, Bin::And, off, within);
            let three = self.s.m.i32(3);
            let shift = self.s.m.bin(c, Bin::Shl, byte, three);
            let x = self.s.m.bin(c, Bin::Shl, x, shift);
            let field = self.s.m.bin(c, Bin::Shl, ones, shift);
            let all = self.s.m.i32(u32::MAX);
            let keep = self.s.m.bin(c, Bin::Xor, field, all);
            let dword = self.s.m.i32(!3);
            let at_dword = self.s.m.bin(c, Bin::And, off, dword);
            let (h, coords) = match index {
                Some(index) => match self.indexed_stride(src, uav)? {
                    IndexedStride::Structured(stride) => {
                        let h = self.handle(site, src, false, (12 | u, stride))?;
                        (h, [index, at_dword, iu])
                    }
                    // Zero-stride view: the index drops out.
                    IndexedStride::Raw => (
                        self.handle(site, src, false, (11 | u, 0))?,
                        [at_dword, iu, iu],
                    ),
                },
                None => (
                    self.handle(site, src, false, (11 | u, 0))?,
                    [at_dword, iu, iu],
                ),
            };
            let at = self.begin_guarded(guard);
            // One compare-exchange per try, merged with the dword it found,
            // so that lanes storing into the same dword keep each other's
            // bytes and no byte ends up with two lanes' bits.
            let start = self.cur;
            let (retry, latch, done) = (
                self.s.m.new_block(),
                self.s.m.new_block(),
                self.s.m.new_block(),
            );
            let zero = self.s.m.i32(0);
            self.s.m.terminate(start, Term::Br(retry));
            let old = self.s.m.phi(retry, self.s.t.i32);
            let rest = self.s.m.bin(retry, Bin::And, old, keep);
            let new = self.s.m.bin(retry, Bin::Or, rest, x);
            self.s.m.terminate(retry, Term::Br(latch));
            let mut args = vec![h];
            args.extend(coords);
            args.extend([old, new]);
            let seen = self.s.op(
                latch,
                "atomicCompareExchange.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                79,
                &args,
            );
            self.s.m.set_phi(old, vec![(zero, start), (seen, latch)]);
            let stored = self.s.m.cmp(latch, pred::EQ, old, seen);
            self.s.m.terminate(latch, Term::CondBr(stored, done, retry));
            self.cur = done;
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            if d16 {
                // shortcut: d16 data is taken as f16 (integer formats would
                // keep 16 bits), widened for the 32-bit float store.
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    wide.push(self.as_int(f));
                }
                vals = wide;
            }
            if let Some(stride) = flat_stride.filter(|_| !typed) {
                let h = self.handle(site, src, false, Self::buffer_props(Some(stride), uav))?;
                let at = self.begin_guarded(guard);
                for (k, mut val) in vals.into_iter().enumerate() {
                    let addr = self.add_bytes(off, 4 * k as u32);
                    if name.ends_with("_hi") {
                        let half = self.s.m.i32(16);
                        val = self.s.m.bin(self.cur, Bin::LShr, val, half);
                    }
                    self.structured_store(h, addr, stride, val, sub.unwrap_or(4));
                }
                self.end_guarded(at, &[]);
                return Ok(());
            }
            let mut args = Vec::new();
            let (op, name) = if typed {
                // shortcut: typed stores are taken to write all four channels
                // of a float format.
                if count != 4 || *offen || *offset != 0 {
                    return Err("a typed buffer store of some channels or at an offset");
                }
                args.push(self.handle(site, src, false, (10 | u, 0x409))?);
                args.extend([index.unwrap_or(self.s.m.i32(0)), iu]);
                for x in vals {
                    args.push(self.as_float(x));
                }
                (69, "bufferStore.f32")
            } else {
                let indexed = match index {
                    Some(index) => self.structured_index(src, uav, index)?,
                    None => None,
                };
                match indexed {
                    Some((index, stride)) => {
                        args.push(self.handle(
                            site,
                            src,
                            false,
                            Self::buffer_props(Some(stride), uav),
                        )?);
                        args.extend([index, off]);
                    }
                    None => {
                        args.push(self.handle(site, src, false, (11 | u, 0))?);
                        args.extend([off, iu]);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = format.or(narrow_format.filter(|_| flat_stride.is_none())) {
            if index.is_some() {
                return Err("a typed-format load by index");
            }
            let h = self.handle(site, src, false, Self::buffer_props(flat_stride, uav))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let mut xs = self.format_load(h, off, f, count, flat_stride)?;
            if d16 {
                xs = self.d16_channels(f, xs)?;
            }
            let n = xs.len();
            for (k, x) in xs.into_iter().enumerate() {
                let reg = reg + k as u8;
                let x = if d16 && k + 1 == n && count % 2 == 1 {
                    #[cfg(test)]
                    {
                        self.s.d16_kept_high_halves += 1;
                    }
                    self.low_half(inst, reg, x)
                } else {
                    x
                };
                self.write(inst, Reg::V(reg), x, guard);
            }
            return Ok(());
        }
        if let Some(stride) = flat_stride.filter(|_| !typed) {
            let h = self.handle(site, src, false, Self::buffer_props(Some(stride), uav))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for k in 0..count {
                let addr = self.add_bytes(off, 4 * k);
                let mut x = self.structured_load(h, addr, stride, sub.unwrap_or(4));
                let reg = reg + k as u8;
                if let Some(bytes) = sub {
                    let bits = 8 * bytes;
                    if matches!(sub_name, "buffer_load_sbyte" | "buffer_load_sshort") {
                        let shift = self.s.m.i32(32 - bits);
                        let lo = self.s.m.bin(c, Bin::Shl, x, shift);
                        x = self.s.m.bin(c, Bin::AShr, lo, shift);
                    } else {
                        let mask = self.s.m.i32((1 << bits) - 1);
                        x = self.s.m.bin(c, Bin::And, x, mask);
                    }
                    if name.ends_with("_hi") {
                        x = self.high_half(inst, reg, x);
                    } else if name.ends_with("_d16") {
                        x = self.low_half(inst, reg, x);
                    }
                }
                self.write(inst, Reg::V(reg), x, guard);
            }
            return Ok(());
        }
        let r = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(site, src, false, (10 | u, 0x409))?;
            let index = index.unwrap_or(self.s.m.i32(0));
            self.s.op(
                c,
                "bufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                68,
                &[h, index, iu],
            )
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            let selector = (!uav && used.is_none())
                .then(|| self.buffer_selector(src))
                .flatten();
            match (index, selector) {
                (Some(index), Some((slot, word))) => {
                    let [alias, stride, bias] = self.selected_buffer(slot, word)?;
                    let start = self.s.m.bin(c, Bin::Mul, index, stride);
                    let offset = self.s.m.bin(c, Bin::Add, start, off);
                    let byte = self.s.m.bin(c, Bin::Add, offset, bias);
                    // A wrapped byte offset must stay outside the bounded raw view.
                    let high = self.mul_hi(index, stride, false);
                    let zero = self.s.m.i32(0);
                    let product_wrapped = self.s.m.cmp(c, pred::NE, high, zero);
                    let offset_wrapped = self.s.m.cmp(c, pred::IULT, offset, start);
                    let bias_wrapped = self.s.m.cmp(c, pred::IULT, byte, offset);
                    let wrapped = self.s.m.bin(c, Bin::Or, product_wrapped, offset_wrapped);
                    let overflow = self.s.m.bin(c, Bin::Or, wrapped, bias_wrapped);
                    // The host leaves this final block outside every alias. All
                    // four component addresses still fit in a 32-bit offset.
                    let invalid = self.s.m.i32(u32::MAX - 15);
                    let in_last_block = self.s.m.cmp(c, pred::IUGE, byte, invalid);
                    let invalid_address = self.s.m.bin(c, Bin::Or, overflow, in_last_block);
                    let byte = self.s.m.select(c, invalid_address, invalid, byte);
                    let h = self.s.heap_handle(c, alias, false, false, (11, 0));
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, byte, iu, mask, align],
                    )
                }
                (Some(index), None) => match self.structured_index(src, uav, index) {
                    Ok(Some((index, stride))) => {
                        let h =
                            self.handle(site, src, false, Self::buffer_props(Some(stride), uav))?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, index, off, mask, align],
                        )
                    }
                    // Zero-stride view: index * 0 drops out, so this is a raw
                    // byte load at `off` (#462).
                    Ok(None) => {
                        let h = self.handle(site, src, false, (11 | u, 0))?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, off, iu, mask, align],
                        )
                    }
                    // Only the view knows the stride: read it from the host's
                    // table (see `UserData`).
                    Err(why) if !uav => self
                        .table_load(src, index, off, mask, count)
                        .map_err(|_| why)?,
                    Err(why) => return Err(why),
                },
                (None, _) => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, off, iu, mask, align],
                    )
                }
            }
        };
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if d16 {
            // Two f16 channels per VGPR. An odd last channel leaves its high
            // half intact. shortcut: MUBUF typed loads use a float view, so
            // UINT/SINT saturation for d16 is handled on the MTBUF path where
            // the instruction names the format (`d16_channels`).
            let k16 = self.s.m.i32(16);
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let h = self.f16_nearest(f);
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                let reg = reg + k as u8;
                let mut x = x.expect("a channel");
                if 2 * k + 1 == count {
                    #[cfg(test)]
                    {
                        self.s.d16_kept_high_halves += 1;
                    }
                    x = self.low_half(inst, reg, x);
                }
                self.write(inst, Reg::V(reg), x, guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }
}
