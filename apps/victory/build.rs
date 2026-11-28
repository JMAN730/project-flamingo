//! Link settings for the Windows executable.

use std::path::PathBuf;

fn main() {
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        return;
    }
    // Threads created without a stack size take the executable's default.
    // Host graphics drivers compiling the title's large shaders on such
    // threads (vkd3d-proton's and the drivers' own, under Wine too) overflow
    // the linker's default of 1 MiB: reserve 16 MiB instead.
    println!("cargo:rustc-link-arg-bins=/STACK:16777216");
    agility_sdk();
}

/// Opts the executable into an app-local Agility SDK runtime when one is
/// installed beside it, in `target/<profile>/D3D12/D3D12Core.dll`.
///
/// Windows 10's own D3D12 stops at shader model 6.5 and translated shaders
/// are 6.6, so there no pipeline of the title's can be created without it.
/// `d3d12.dll` loads that runtime when the executable exports the two
/// symbols of `victory_shader::agility`. `victory-shader`'s build script
/// exports them from its test binaries; link arguments do not reach
/// dependents, so the executable asks for them here, on the same condition:
/// with the exports and no DLL, `d3d12.dll` creates no device at all, so a
/// build without the DLL exports nothing and runs on the in-box runtime.
fn agility_sdk() {
    // `OUT_DIR` is `target/<profile>/build/<pkg>-<hash>/out`.
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let profile = out
        .ancestors()
        .nth(3)
        .expect("a profile directory above OUT_DIR");
    let runtime = profile.join("D3D12").join("D3D12Core.dll");
    println!("cargo:rerun-if-changed={}", runtime.display());
    if runtime.is_file() {
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKPath,DATA");
    }
}
