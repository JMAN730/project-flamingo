use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Runs `victory config` from `cwd` (so no `./victory.toml` is read) with the
/// cache and data folders set to the given paths.
fn config(cwd: &Path, cache: &Path, data: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_victory"))
        .arg("config")
        .current_dir(cwd)
        .env_remove("VICTORY_CONFIG")
        .env_remove("VICTORY_GAME_DIR")
        .env_remove("VICTORY_TEST_TITLES")
        .env("VICTORY_CACHE_DIR", cache)
        .env("VICTORY_DATA_DIR", data)
        .output()
        .unwrap()
}

/// A folder inside the source tree, which the asset policy refuses.
fn inside_repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/config-policy-test")
}

#[test]
fn config_succeeds_with_paths_outside_the_repo() {
    let dir = tempfile::tempdir().unwrap();
    let output = config(
        dir.path(),
        &dir.path().join("cache"),
        &dir.path().join("data"),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{output:?}");
    assert!(!stdout.contains("POLICY VIOLATION"), "{stdout}");
}

#[test]
fn config_fails_when_a_path_violates_the_asset_policy() {
    let dir = tempfile::tempdir().unwrap();
    let output = config(dir.path(), &inside_repo(), &dir.path().join("data"));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "config succeeded: {stdout}");
    assert!(stdout.contains("POLICY VIOLATION"), "{stdout}");
    assert!(stderr.contains("1 configured path(s)"), "{stderr}");
}
