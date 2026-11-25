//! Finding the files a title names in its folder.

use std::path::{Path, PathBuf};

/// The file `name` in `dir`, by the name as written or else by the entry
/// whose name matches it ignoring ASCII case, as the Windows loader looks
/// DLLs up. A folder may be case-sensitive (WSL makes them so on Windows
/// drives), where a title's `GameLib.dll` and its import of `gamelib.dll`
/// are different names. A directory spelled like the file does not hide it.
pub(crate) fn file_in(dir: &Path, name: &str) -> Option<PathBuf> {
    let exact = dir.join(name);
    if exact.is_file() {
        return Some(exact);
    }
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.eq_ignore_ascii_case(name))
        })
        .map(|e| e.path())
        .find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, empty folder under the system temp directory.
    fn folder(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("victory-loader-files-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_file_is_found_by_any_spelling() {
        let dir = folder("spelling");
        std::fs::write(dir.join("Late.dll"), b"x").unwrap();
        for name in ["Late.dll", "LATE.DLL", "late.dll"] {
            let found = file_in(&dir, name).unwrap_or_else(|| panic!("{name}"));
            assert!(found.is_file(), "{name}");
        }
        assert_eq!(file_in(&dir, "other.dll"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_folder_spelled_like_the_file_does_not_hide_it() {
        let dir = folder("folder");
        std::fs::create_dir(dir.join("LATE.DLL")).unwrap();
        std::fs::write(dir.join("late.dll"), b"x").unwrap();
        let found = file_in(&dir, "Late.dll").unwrap();
        assert!(found.is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
