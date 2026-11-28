//! A root SRV or UAV may name any byte, but the raw view the host binds for
//! it starts on 16 bytes: the host binds it at the boundary below the
//! title's address and gives the shader how far below that is
//! ([`emit::USER_ROOT_VIEW_BIAS`]), which every byte offset into the view
//! adds (loads, stores and atomics alike), as the indexed read path does
//! with its table record's bias.

use super::{Lower, Why};
use crate::emit;
use crate::emit::bitcode::{Bin, V};
use crate::native::{ResourceType, RootType};
use crate::resource::Source;

impl Lower<'_> {
    /// `off`, a byte offset into the view of descriptor `src`, plus the
    /// host's bias word for its root slot when `src` is built around a
    /// root descriptor the host may bind as a raw view (not a constant
    /// buffer, which it binds by address). Other descriptors keep `off`.
    pub(super) fn plus_root_view_bias(&mut self, src: &Source, off: V) -> Result<V, Why> {
        let Source::Built { address, .. } = src else {
            return Ok(off);
        };
        let Some(rp) = self.x.native.layout.get(address.param) else {
            // The code's own tables and title memory by address.
            return Ok(off);
        };
        if rp.root != RootType::Descriptor
            || self
                .used_kind(src)
                .is_some_and(|u| u.kind == ResourceType::Cbv)
        {
            return Ok(off);
        }
        let slot = rp.slot;
        let bias = self.host_word(RootType::Other(emit::USER_ROOT_VIEW_BIAS), slot, 0);
        Ok(self.s.m.bin(self.cur, Bin::Add, off, bias))
    }
}
