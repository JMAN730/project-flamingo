//! `victory split-diff`: the title's wave64 compute programs, each run as
//! the two-half lowering of a 32-lane host makes it and compared with the
//! variants that must agree with it (`victory_shader::emit::split_diff`).
//! Prints and writes hashes and counts only, never shader contents.

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use victory_shader::emit::split_diff::{Harness, Options, Outcome};
use victory_shader::native::StageIo;
use victory_shader::{Container, FourCC, HwStage};

use crate::shader::{MAX_FILE, scan_entries, sources};

pub fn run(
    paths: Vec<PathBuf>,
    out: PathBuf,
    only: Option<String>,
    groups: [u32; 3],
    dry_run: bool,
) -> Result<()> {
    let paths = crate::default_paths(paths)?;
    victory_core::ensure_outside_repo(&out)?;
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let only: Vec<String> = only
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    // The programs, by the hash victory-gpu logs (container bytes).
    let mut programs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut seen = HashSet::new();
    let mut walk_errors = 0;
    let mut unread = 0;
    for root in &paths {
        for entry in scan_entries(root, &mut walk_errors) {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                // Counted with the entries the walk could not read: the
                // scan is then reported as incomplete.
                unread += 1;
                continue;
            };
            for source in sources(&data) {
                let Some(bytes) = source.bytes() else {
                    continue;
                };
                let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
                if !seen.insert(hash.clone()) {
                    continue;
                }
                if !only.is_empty() && !only.iter().any(|p| hash.starts_with(p.as_str())) {
                    continue;
                }
                if is_wave64_compute(&bytes) {
                    programs.insert(hash, bytes.into_owned());
                }
            }
        }
    }
    walk_errors += unread;
    // Made only now, after the inputs are read: a report path that is also
    // an input would otherwise be emptied before it is scanned.
    let mut report =
        std::fs::File::create(&out).with_context(|| format!("creating {}", out.display()))?;
    let harness = Harness::new().map_err(anyhow::Error::msg)?;
    let header = format!(
        "split-diff: {} wave64 compute programs; groups {groups:?}; reference {}",
        programs.len(),
        harness
            .reference()
            .unwrap_or("none (no device with 64-lane waves and SM 6.6)")
    );
    println!("{header}");
    writeln!(report, "{header}")?;
    // `SPLIT_DIFF_DUMP=<dir>` (outside the repository): each variant's DXIL.
    let dump = std::env::var_os("SPLIT_DIFF_DUMP").map(PathBuf::from);
    if let Some(d) = &dump {
        victory_core::ensure_outside_repo(d)?;
        std::fs::create_dir_all(d)?;
    }
    let mut opts = Options {
        groups,
        dry_run,
        dump,
        ..Options::default()
    };
    // `SPLIT_DIFF_SPIN_LOG2=<n>`: a half gives up waiting after 2^n polls
    // (a hang of a faulty lowering lasts that long).
    if let Some(n) = std::env::var("SPLIT_DIFF_SPIN_LOG2")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
    {
        opts.spin_limit = 1 << n.min(24);
    }
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut differing = Vec::new();
    for (hash, bytes) in &programs {
        let outcome = harness.check(bytes, &opts);
        let (class, line) = match &outcome {
            Outcome::NotSplit(why) => ("not split", format!("{hash} not split: {why}")),
            Outcome::Skipped(why) => ("skipped", format!("{hash} skipped: {why}")),
            Outcome::Planned(t, w) => {
                ("planned", format!("{hash} planned: {t} threads, {w} waves"))
            }
            Outcome::Ran(r) => {
                let mut parts = Vec::new();
                let mut class = "agree";
                // Agreement takes a comparison that ran: a variant that
                // failed, or none compared at all, leaves it unknown.
                let (mut compared, mut failed) = (false, false);
                for v in &r.variants {
                    compared |= v.result.is_ok() && v.diff.is_some();
                    failed |= v.result.is_err();
                    let s = match (&v.result, v.diff) {
                        (Err(e), _) => {
                            class = if e.contains("device removed") {
                                "fault"
                            } else {
                                class
                            };
                            format!("{}=error({e})", v.name)
                        }
                        (Ok(h), None) => format!("{}={h}", v.name),
                        (Ok(_), Some((0, _))) => format!("{}=same", v.name),
                        (Ok(h), Some((n, o))) if v.reordered => {
                            if class == "agree" {
                                class = "reordered";
                            }
                            format!("{}={h} ({n} dwords in {o} outputs reordered)", v.name)
                        }
                        (Ok(h), Some((n, o))) => {
                            if class != "fault" {
                                // A variant that differs from itself run
                                // again says nothing by differing.
                                class = match v.name.split(':').nth(1).unwrap_or("") {
                                    "split-again" | "split-pairs-again" => "nondeterministic",
                                    _ if class == "nondeterministic" => class,
                                    _ => "differ",
                                };
                            }
                            format!("{}={h} ({n} dwords in {o} outputs differ)", v.name)
                        }
                    };
                    parts.push(s);
                }
                if class == "agree" && (failed || !compared) {
                    class = "incomplete";
                }
                if class == "differ" || class == "nondeterministic" {
                    differing.push((hash.clone(), class));
                }
                let line = format!(
                    "{hash} {class}: {} threads, {} waves, {} outputs of {} dwords, {} written in {}; {}",
                    r.threads,
                    r.waves,
                    r.outputs,
                    r.words,
                    r.written.0,
                    r.written.1,
                    parts.join(", ")
                );
                (class, line)
            }
        };
        *counts.entry(class).or_default() += 1;
        println!("{line}");
        writeln!(report, "{line}")?;
        report.flush()?;
        if class == "fault" {
            let msg = "stopped: the device was removed";
            println!("{msg}");
            writeln!(report, "{msg}")?;
            break;
        }
    }
    let summary = format!(
        "summary: {}",
        counts
            .iter()
            .map(|(k, n)| format!("{n} {k}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("{summary}");
    writeln!(report, "{summary}")?;
    for (hash, class) in &differing {
        writeln!(report, "listed {hash} {class}")?;
    }
    println!("report: {}", out.display());
    anyhow::ensure!(
        walk_errors == 0,
        "scan incomplete (traversal errors: {walk_errors})"
    );
    Ok(())
}

/// Whether `bytes` is a container of a wave64 compute program.
fn is_wave64_compute(bytes: &[u8]) -> bool {
    let Ok(c) = Container::parse(bytes) else {
        return false;
    };
    if c.part(FourCC::DXIL).is_none() && c.part(victory_shader::accc::ACCC).is_none() {
        return false;
    }
    victory_shader::container_programs(&c).is_ok_and(|ps| {
        ps.iter()
            .any(|p| p.stage == HwStage::Cs && p.wave64 && matches!(p.io, StageIo::Cs(_)))
    })
}
