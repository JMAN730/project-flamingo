//! Portable publication of derived files into a shared cache directory.
//!
//! Several copies of the runner can prepare the same title at once. Each
//! writer stages into its own file, so two of them cannot truncate each
//! other's bytes or rename a file the other already published. The final
//! rename is atomic within the directory, and the staging file is removed
//! if publication fails. An unchanged file is checked with a bounded buffer.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

const COMPARE_CHUNK_BYTES: usize = 256 * 1024;
static NEXT_STAGING: AtomicU64 = AtomicU64::new(0);

fn cached_bytes_match(path: &Path, bytes: &[u8]) -> io::Result<bool> {
    let mut file = File::open(path)?;
    if file
        .metadata()
        .is_ok_and(|metadata| metadata.len() != bytes.len() as u64)
    {
        return Ok(false);
    }

    let mut buffer = vec![0; bytes.len().min(COMPARE_CHUNK_BYTES)];
    for chunk in bytes.chunks(COMPARE_CHUNK_BYTES) {
        let buffer = &mut buffer[..chunk.len()];
        file.read_exact(buffer)?;
        if buffer != chunk {
            return Ok(false);
        }
    }

    // Check for growth since the metadata check. read_exact retries interrupted reads.
    match file.read_exact(&mut [0]) {
        Ok(()) => Ok(false),
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(true),
        Err(error) => Err(error),
    }
}

/// Writes `bytes` to `path` unless it is already there, using a staging file
/// unique to this process and call.
pub fn write_if_changed(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if cached_bytes_match(path, bytes).unwrap_or(false) {
        return Ok(());
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image");
    let sequence = NEXT_STAGING.fetch_add(1, Ordering::Relaxed);
    let staging = path.with_file_name(format!(".{name}.{}.{sequence}.tmp", std::process::id()));
    std::fs::write(&staging, bytes)?;
    if let Err(error) = std::fs::rename(&staging, path) {
        let _ = std::fs::remove_file(&staging);
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn cached_contents_match_across_chunks_and_at_the_end() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.bin");
        let mut bytes = vec![0x51; 2 * COMPARE_CHUNK_BYTES + 11];
        bytes[COMPARE_CHUNK_BYTES..].fill(0x86);
        bytes[2 * COMPARE_CHUNK_BYTES..].fill(0xb3);
        write_if_changed(&path, &bytes).unwrap();
        assert!(cached_bytes_match(&path, &bytes).unwrap());
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(UNIX_EPOCH + Duration::from_secs(1_600_000_000))
            .unwrap();
        let modified = path.metadata().unwrap().modified().unwrap();
        write_if_changed(&path, &bytes).unwrap();
        assert_eq!(path.metadata().unwrap().modified().unwrap(), modified);

        for offset in [
            COMPARE_CHUNK_BYTES - 1,
            COMPARE_CHUNK_BYTES,
            bytes.len() - 1,
        ] {
            bytes[offset] ^= 0xff;
            assert!(!cached_bytes_match(&path, &bytes).unwrap());
            write_if_changed(&path, &bytes).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), bytes);
        }
        assert!(!cached_bytes_match(&path, &bytes[..bytes.len() - 1]).unwrap());
    }

    /// Two publishers staging the same destination at once both succeed,
    /// and neither leaves a staging file behind.
    #[test]
    fn concurrent_publishers_do_not_share_a_staging_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.bin");
        let bytes = b"a prepared image".to_vec();
        let (first, second) = (path.clone(), path.clone());
        let (first_bytes, second_bytes) = (bytes.clone(), bytes.clone());
        let (one, two) = (
            std::thread::spawn(move || write_if_changed(&first, &first_bytes)),
            std::thread::spawn(move || write_if_changed(&second, &second_bytes)),
        );
        one.join().unwrap().unwrap();
        two.join().unwrap().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path() != path)
            .collect();
        assert!(
            leftovers.is_empty(),
            "staging files were removed: {leftovers:?}"
        );
    }

    #[test]
    fn a_failed_publication_removes_its_staging_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing-parent").join("image.bin");
        assert!(write_if_changed(&path, b"bytes").is_err());
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .collect();
        assert!(leftovers.is_empty(), "no staging file survives an error");
    }
}
