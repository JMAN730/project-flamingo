use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(victory_bundled_hlsl)");
    println!("cargo:rerun-if-env-changed=VICTORY_BUILD_DXC");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        agility_sdk();
    }
    bundled_hlsl();
}

/// Compiles the helper shaders at build time when `VICTORY_BUILD_DXC` names
/// a compiler.
fn bundled_hlsl() {
    let Some(compiler) = env::var_os("VICTORY_BUILD_DXC").map(PathBuf::from) else {
        return;
    };
    println!("cargo:rerun-if-changed={}", compiler.display());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    for (name, entry, profile) in [
        ("argument_clamp", "main", "cs_6_0"),
        ("rect_rewrite", "main", "cs_6_0"),
        ("rect_expand", "main", "cs_6_0"),
        ("plane_corners", "corner", "vs_6_0"),
        ("plane_steps", "plane", "ps_6_0"),
        ("plane_floats", "plane", "ps_6_0"),
        ("rebar_hash", "main", "cs_6_0"),
        ("mesh_draws", "main", "cs_6_0"),
    ] {
        let source = format!("shaders/{name}.hlsl");
        println!("cargo:rerun-if-changed={source}");
        let status = Command::new(&compiler)
            .args(["-T", profile, "-E", entry, "-O3", "-Fo"])
            .arg(out.join(format!("{name}.dxil")))
            .arg(&source)
            .status()
            .unwrap_or_else(|e| panic!("could not start {} for {source}: {e}", compiler.display()));
        assert!(status.success(), "DXC failed for {source} ({status})");
    }
    println!("cargo:rustc-cfg=victory_bundled_hlsl");
}

/// Opts this crate's test binaries into an app-local Agility SDK runtime
/// when one is installed in `target/<profile>/D3D12/D3D12Core.dll`, as
/// `victory-shader`'s build script does for its own and `apps/victory`'s for
/// the executable.
///
/// Windows 10's own D3D12 stops at shader model 6.5, and tests that draw
/// with translated shaders (6.6) need a device that takes them. `d3d12.dll`
/// loads that runtime when the running module exports the two symbols of
/// `victory_shader::agility`; with the exports and no DLL it creates no
/// device at all, so nothing is exported without the DLL.
fn agility_sdk() {
    // `OUT_DIR` is `target/<profile>/build/<pkg>-<hash>/out`.
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile = out
        .ancestors()
        .nth(3)
        .expect("a profile directory above OUT_DIR")
        .to_path_buf();
    let runtime = profile.join("D3D12").join("D3D12Core.dll");
    println!("cargo:rerun-if-changed={}", runtime.display());
    if !runtime.is_file() {
        return;
    }
    // The loader resolves `D3D12SDKPath` next to the running module, and
    // test binaries run from `deps`.
    let deps = profile.join("deps").join("D3D12");
    let _ = std::fs::create_dir_all(&deps);
    let _ = std::fs::copy(&runtime, deps.join("D3D12Core.dll"));
    // For every binary linked from this package: its test binaries (the
    // library itself is not linked).
    println!("cargo:rustc-link-arg=/EXPORT:D3D12SDKVersion,DATA");
    println!("cargo:rustc-link-arg=/EXPORT:D3D12SDKPath,DATA");
}
