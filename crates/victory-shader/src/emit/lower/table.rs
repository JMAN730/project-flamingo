//! Buffer loads whose element stride only the view knows (a descriptor the
//! program picks at run time, such as each lane's own in a waterfall loop
//! over heap indices): read through the host's buffer view table (see
//! [`UserData`]), which gives each heap slot's raw view, stride, first byte
//! and element count.

use super::{Lower, Val, Why};
use crate::emit::bitcode::{Attr, Bin, V, pred};
use crate::emit::{self, UNKNOWN_STRUCTURED_STRIDE, UserData};
use crate::ir::{Inst, Reg};
use crate::native::RootType;
use crate::resource::Source;

/// A buffer view's record in the host's buffer view table
/// ([`Lower::table_buffer`]).
pub(super) struct TableBuffer {
    handle: V,
    stride: V,
    bias: V,
    elements: V,
}

impl Lower<'_> {
    /// Whether every view `src` may name sits at a fixed table entry, whose
    /// stride the host specializes the pipeline with instead (see
    /// [`emit::StructuredBinding`]): such loads keep that path.
    fn view_specializable(&self, src: &Source) -> bool {
        src.leaves().into_iter().all(|leaf| {
            let Source::Loaded { load, first } = *leaf else {
                return false;
            };
            let Some(address) = self.scalar.get(&load) else {
                return false;
            };
            let Some(param) = self.x.native.layout.get(address.param) else {
                return false;
            };
            param.root == RootType::Table
                && address.dynamic.is_none()
                && address
                    .offset
                    .checked_add(4 * u32::from(first))
                    .is_some_and(|o| o % 32 == 0)
        })
    }

    /// The heap index of the buffer descriptor `src` and whether it may
    /// differ between lanes, as [`Lower::handle`] takes them.
    pub(super) fn buffer_heap_index(&mut self, src: &Source) -> Result<(V, bool), Why> {
        match src {
            Source::Loaded { load, first } => {
                let index = self.heap_index(*load, *first, false)?;
                Ok((index, self.x.res.waterfall.loads.contains(load)))
            }
            Source::Phi { .. } => Ok((self.uniform_heap_index(src, false)?, false)),
            Source::Select { .. } => Ok((self.lane_heap_index(src, false)?, false)),
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => {
                Ok((self.lane_heap_index(src, false)?, true))
            }
            Source::Constant(_) | Source::Built { .. } => Err(UNKNOWN_STRUCTURED_STRIDE),
        }
    }

    /// The user-data dword holding the heap index of the host's buffer
    /// view table (see [`UserData`]).
    pub(super) fn buffer_table_slot(&mut self) -> Result<V, Why> {
        let existing = self
            .user_reads
            .values()
            .find(|u| !u.indexed_buffer && u.root == RootType::ResourceHeap)
            .map(|u| u.index);
        let index = match existing {
            Some(index) => index,
            None => {
                // After every input, which keeps its own position.
                let inputs = self
                    .params
                    .values()
                    .filter_map(|&(p, word)| {
                        let rp = self.x.native.layout[p];
                        matches!(rp.root, RootType::Constants | RootType::Table)
                            .then_some(u32::from(rp.user_data) + word + 1)
                    })
                    .max()
                    .unwrap_or(0);
                let index = inputs.max(self.s.user_data);
                if index + 1 > 64 {
                    return Err("a buffer view table beyond the root constant budget");
                }
                self.user_reads.insert(
                    index,
                    UserData {
                        indexed_buffer: false,
                        index,
                        slot: 0,
                        root: RootType::ResourceHeap,
                        dword: 0,
                    },
                );
                index
            }
        };
        Ok(self.user_word(index))
    }

    /// The host's record of the buffer view `src` names (see
    /// [`UserData`]): a raw handle of the memory holding it, its
    /// stride, its first byte there and its element count. A slot the host
    /// has no record for gives the table's own view and no elements.
    pub(super) fn table_buffer(&mut self, src: &Source) -> Result<TableBuffer, Why> {
        if crate::switch::no_buffer_table() || self.view_specializable(src) {
            return Err(UNKNOWN_STRUCTURED_STRIDE);
        }
        let (index, per_lane) = self.buffer_heap_index(src)?;
        let c = self.cur;
        let slot = self.buffer_table_slot()?;
        let table = self.s.heap_handle(c, slot, false, false, (11, 0));
        let invalid = self.s.m.i32(u32::MAX - 15);
        let limit = self.s.m.i32(emit::VIEW_HEAP_LIMIT);
        let in_heap = self.s.m.cmp(c, pred::IULT, index, limit);
        let size = self.s.m.i32(emit::BUFFER_TABLE_RECORD);
        let at = self.s.m.bin(c, Bin::Mul, index, size);
        let at = self.s.m.select(c, in_heap, at, invalid);
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(0xF);
        let align = self.s.m.i32(4);
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[table, at, iu, mask, align],
        );
        let mut words = [iu; 4];
        for (k, w) in words.iter_mut().enumerate() {
            let x = self.s.m.extract(c, r, k as u32);
            *w = self.as_int(x);
        }
        let [alias, stride, bias, elements] = words;
        let zero = self.s.m.i32(0);
        let known = self.s.m.cmp(c, pred::NE, elements, zero);
        let alias = self.s.m.select(c, known, alias, slot);
        let handle = self.s.heap_handle(c, alias, false, per_lane, (11, 0));
        Ok(TableBuffer {
            handle,
            stride,
            bias,
            elements,
        })
    }

    /// An indexed load of `mask`'s dwords at byte `off` of element `index`
    /// of a buffer whose stride only its view knows, through the host's
    /// table. An element past the view's, or an address that wraps, reads
    /// zeros (the console's structured bounds check).
    // shortcut: an offset past the element's stride reads into the next
    // element, or past the view into the memory after it, as far as the
    // raw view holding it reaches.
    pub(super) fn table_load(&mut self, src: &Source, index: V, off: V, mask: V) -> Result<V, Why> {
        let t = self.table_buffer(src)?;
        let c = self.cur;
        let start = self.s.m.bin(c, Bin::Mul, index, t.stride);
        let within = self.s.m.bin(c, Bin::Add, start, off);
        let high = self.mul_hi(index, t.stride, false);
        let zero = self.s.m.i32(0);
        let product_wrapped = self.s.m.cmp(c, pred::NE, high, zero);
        let offset_wrapped = self.s.m.cmp(c, pred::IULT, within, start);
        // A raw view (stride 0) is not indexed: the index moves nothing,
        // and the count is of bytes, not of elements.
        let past = self.s.m.cmp(c, pred::IUGE, index, t.elements);
        let structured = self.s.m.cmp(c, pred::NE, t.stride, zero);
        let past = self.s.m.bin(c, Bin::And, past, structured);
        let wrapped = self.s.m.bin(c, Bin::Or, product_wrapped, offset_wrapped);
        let outside = self.s.m.bin(c, Bin::Or, wrapped, past);
        let byte = self.s.m.bin(c, Bin::Add, within, t.bias);
        let bias_wrapped = self.s.m.cmp(c, pred::IULT, byte, within);
        let outside = self.s.m.bin(c, Bin::Or, outside, bias_wrapped);
        // The host leaves this final block outside every view it lists.
        let invalid = self.s.m.i32(u32::MAX - 15);
        let byte = self.s.m.select(c, outside, invalid, byte);
        let iu = self.s.m.undef(self.s.t.i32);
        let align = self.s.m.i32(4);
        Ok(self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[t.handle, byte, iu, mask, align],
        ))
    }

    /// A buffer descriptor the program reads as data whose stride only its
    /// view knows: word 2 (the record count) from the host's table, as the
    /// loads through it take the stride ([`Lower::table_load`]).
    pub(super) fn table_descriptor_data(
        &mut self,
        inst: &Inst,
        code: u8,
        count: u32,
        first: u8,
        src: &Source,
    ) -> Result<(), Why> {
        let t = self.table_buffer(src)?;
        for k in 0..count {
            let Some(v) = inst.write(Reg::S(code + k as u8)) else {
                continue;
            };
            if k.wrapping_sub(u32::from(first)) == 2 {
                self.write(inst, Reg::S(code + k as u8), t.elements, None);
            } else {
                self.vals.insert(
                    v,
                    Val::Missing("a descriptor's address or format read as data"),
                );
            }
        }
        Ok(())
    }
}
