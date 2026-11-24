//! `victory prepare <game files>`: translates every shader the game files
//! hold into the shader cache a run uses, before the run, so the first
//! world visit after an update does not translate them while the title
//! waits (its draws dropped until then: objects see-through, minutes of
//! 1 fps). Run it with the same settings as the run (`VICTORY_CACHE_DIR`,
//! `VICTORY_SHADER_LOOP_LIMIT`, the `VICTORY_WAVE_*` switches): they are
//! part of each translation's key. VIctory Try does that after each build.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;

use crate::shader::{MAX_FILE, sources};

pub fn run(paths: Vec<PathBuf>, jobs: Option<usize>) -> Result<()> {
    let paths = crate::default_paths(paths)?;
    let cfg = victory_core::Config::load()?;
    victory_hle::modules::d3d12x::use_shader_cache(cfg.cache_subdir("shaders")?);
    // Half the logical processors by default: the PC stays usable.
    let threads = jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).max(1)));
    let seen = Mutex::new(HashSet::new());
    let (done, failed) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let started = std::time::Instant::now();
    let mut last_said = std::time::Instant::now();
    for root in &paths {
        for entry in walkdir::WalkDir::new(root).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                continue;
            };
            let jobs = sources(&data);
            if jobs.is_empty() {
                continue;
            }
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads.min(jobs.len()) {
                    // The run's pipeline workers' stack: deep programs overflowed 2 MB.
                    let worker = std::thread::Builder::new().stack_size(64 << 20);
                    let _ = worker.spawn_scoped(s, || {
                        while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let Some(bytes) = job.bytes() else { continue };
                            let hash = blake3::hash(&bytes);
                            if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert(hash) {
                                continue;
                            }
                            // One shader that panics the translator is left to the run.
                            let ok = std::panic::catch_unwind(|| victory_hle::modules::d3d12x::prepare_shader(&bytes))
                                .unwrap_or(false);
                            if ok {
                                done.fetch_add(1, Ordering::Relaxed);
                            } else {
                                failed.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    });
                }
            });
            if last_said.elapsed().as_secs() >= 15 {
                last_said = std::time::Instant::now();
                let (hits, made) = victory_hle::modules::d3d12x::prepared_shader_counts();
                println!(
                    "prepare      {} shaders ready ({hits} already cached, {made} translated now), {} do not translate, {:.0} s",
                    done.load(Ordering::Relaxed),
                    failed.load(Ordering::Relaxed),
                    started.elapsed().as_secs_f64()
                );
            }
        }
    }
    let (hits, made) = victory_hle::modules::d3d12x::prepared_shader_counts();
    println!(
        "prepared     {} shaders ready ({hits} already cached, {made} translated now), {} do not translate, in {:.0} s",
        done.load(Ordering::Relaxed),
        failed.load(Ordering::Relaxed),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
