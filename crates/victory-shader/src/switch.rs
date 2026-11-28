//! Switches that turn translator paths off, for telling which of them breaks
//! a title on a host (bisecting in the world). Read once, in every build (not
//! only with `diagnostics`). Each is part of the translation, so the shader
//! cache must key on [`bits`] (`victory-gpu`'s does).
//!
//! * `VICTORY_SHADER_NO_LAST_CHANCE=1`: programs that need the last chance
//!   (`leftovers`, in `emit` and in `resource::track`) are refused.
//! * `VICTORY_SHADER_NO_SECOND_PASS=1`: no second try at a refused program
//!   (SCC of a lane mask as a wave-any, lane reads from a spill).
//! * `VICTORY_NO_ACCC=1`: `ACCC` containers have no programs.
//! * `VICTORY_SHADER_NO_BUFFER_TABLE=1`: buffer loads whose stride only the
//!   view knows are refused instead of read through the host's buffer view
//!   table (`emit::UserData`).
//!
//! Test builds ignore the environment: every path is on.

use std::sync::OnceLock;

const NO_LAST_CHANCE: u32 = 1;
const NO_SECOND_PASS: u32 = 2;
const NO_ACCC: u32 = 4;
const NO_BUFFER_TABLE: u32 = 8;

/// The set of switches that are on, as bits (for the shader cache key).
pub fn bits() -> u32 {
    static BITS: OnceLock<u32> = OnceLock::new();
    if cfg!(test) {
        return 0;
    }
    *BITS.get_or_init(|| {
        // Like `victory-gpu`'s switches: set, and not `0`.
        let on = |name: &str| std::env::var_os(name).is_some_and(|v| v != "0");
        [
            ("VICTORY_SHADER_NO_LAST_CHANCE", NO_LAST_CHANCE),
            ("VICTORY_SHADER_NO_SECOND_PASS", NO_SECOND_PASS),
            ("VICTORY_NO_ACCC", NO_ACCC),
            ("VICTORY_SHADER_NO_BUFFER_TABLE", NO_BUFFER_TABLE),
        ]
        .into_iter()
        .filter(|&(name, _)| on(name))
        .fold(0, |bits, (_, bit)| bits | bit)
    })
}

/// The last chance of `emit` and `resource::track` is off.
pub fn no_last_chance() -> bool {
    bits() & NO_LAST_CHANCE != 0
}

/// The second try of `lower` is off.
pub fn no_second_pass() -> bool {
    bits() & NO_SECOND_PASS != 0
}

/// `ACCC` containers have no programs.
pub fn no_accc() -> bool {
    bits() & NO_ACCC != 0
}

/// Buffer loads whose stride only the view knows are refused.
pub fn no_buffer_table() -> bool {
    bits() & NO_BUFFER_TABLE != 0
}
