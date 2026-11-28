//! Xbox Series shaders to host shaders.
//!
//! An Xbox Series shader is a standard DXBC container ([`dxbc`]). Its `DXIL`
//! part ([`dxil`]) may carry Xbox additions around the LLVM bitcode: an
//! `XDAT` table before it indexing blobs after it, among them `xbop`
//! (compiler options) and the zlib-compressed GFX10 code. The bitcode may use
//! console-only intrinsics, so the GFX10 code is translated in general.

pub mod accc;
pub mod agility;
pub mod bvh;
pub mod dxbc;
pub mod dxil;
pub mod emit;
pub mod gfx10;
pub mod io;
pub mod ir;
pub mod native;
pub mod resource;
pub mod structure;
pub mod switch;

pub use dxbc::{Container, DispatchInfo, FourCC, Fspx, Part, inspect_fspx};
pub use dxil::{DxilProgram, ShaderKind, XboxExtras};
pub use native::{
    HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, container_programs,
    native_programs,
};

/// The name a shader container goes by in the runtime's logs and in
/// `victory translate --hash`: the first 16 hex digits of the BLAKE3 of its
/// bytes, as the title passes them. Two-half lowering selection
/// (`emit::with_split_selection`) keys on it too.
pub fn runtime_hash(container: &[u8]) -> String {
    blake3::hash(container).to_hex()[..16].to_owned()
}

#[cfg(test)]
mod runtime_hash_tests {
    /// A synthetic container is named by the first 16 hex digits of its
    /// BLAKE3, lowercase, as the logs print it.
    #[test]
    fn the_runtime_hash_is_the_blake3_prefix() {
        let bytes = crate::dxbc::build(&[(crate::FourCC::DXIL, &[1, 2, 3, 4])]);
        let hash = super::runtime_hash(&bytes);
        assert_eq!(hash.len(), 16);
        assert!(
            hash.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
        assert!(blake3::hash(&bytes).to_hex().starts_with(&hash));
        assert_ne!(hash, super::runtime_hash(&bytes[1..]));
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ShaderError {
    #[error("not a DXBC container")]
    NotDxbc,
    #[error("malformed shader: {0}")]
    Malformed(&'static str),
    #[error("GFX10 code at {offset:#x}: {reason}")]
    Isa { offset: usize, reason: &'static str },
    /// Valid code the translator cannot handle yet: an opcode name or a construct.
    #[error("GFX10 code at {offset:#x}: {what} is not supported")]
    Unsupported { offset: usize, what: &'static str },
}
