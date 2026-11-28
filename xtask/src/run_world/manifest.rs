//! The run manifest: what was launched, from which commit, with which
//! settings, and how it ended. One `manifest.json` per run folder, written
//! when the run starts and again when it ends, so that two runs can be
//! compared without anybody's memory of how they were started.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::settings::Settings;

pub const FILE_NAME: &str = "manifest.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub run_id: String,
    pub session: String,
    /// The commit the working tree was at, and whether it had changes.
    pub commit: String,
    pub dirty: bool,
    /// The executable that was copied, and the copy that ran.
    pub exe: PathBuf,
    pub exe_copy: PathBuf,
    /// The arguments the copy was started with.
    pub args: Vec<String>,
    pub settings: Settings,
    /// Every variable the tool set for the title.
    pub env: BTreeMap<String, String>,
    /// The caller's `VICTORY_*` path variables the title inherited.
    pub inherited_env: BTreeMap<String, String>,
    /// The title's output (stdout and stderr).
    pub log: PathBuf,
    /// Where presented frames were saved, if the build can capture.
    pub capture_dir: Option<PathBuf>,
    pub pid: Option<u32>,
    /// When the run was asked for: the time in the run id (UTC, RFC 3339).
    /// The build and the copy of the exe come after it.
    pub started: String,
    /// When the title process was started. The length of the run counts
    /// from here. Absent in manifests written before this was recorded.
    #[serde(default)]
    pub title_started: Option<String>,
    /// Absent while the run is going (or if its launcher was killed).
    pub end: Option<End>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct End {
    /// When the title process was found to have ended.
    pub ended: String,
    /// How long the title ran: from its start to its end, without the
    /// build and the copy before it.
    pub seconds: u64,
    /// The process exit code; absent when it could not be read.
    pub exit_code: Option<i64>,
    /// The same as Windows prints it (`0xC0000005`).
    pub exit_code_hex: Option<String>,
    pub ended_by: EndedBy,
    /// The `crashes.log` files of the data folder that this run wrote to,
    /// each with the copy of what it added.
    pub crashes: Vec<CrashLog>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EndedBy {
    /// The title ended by itself (a clean exit or a crash: see the code).
    Title,
    /// `run-world stop`.
    Stop,
    /// `--max-seconds` ran out.
    MaxSeconds,
    /// The launcher was gone when `stop` cleaned up; no exit code.
    LauncherGone,
    /// The launcher ended it: the title got the system's D3D12 instead of
    /// the Agility SDK beside the exe, so nothing it drew counts.
    SystemD3d12,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrashLog {
    pub path: PathBuf,
    /// What the run appended, saved beside the manifest.
    pub copy: PathBuf,
    pub bytes_added: u64,
}

impl Manifest {
    pub fn write(&self, run_dir: &Path) -> Result<()> {
        let path = run_dir.join(FILE_NAME);
        victory_core::ensure_outside_repo(&path)?;
        let text = serde_json::to_string_pretty(self).expect("a manifest is plain data");
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
    }

    pub fn read(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    /// One line for `run-world list`.
    pub fn summary(&self) -> String {
        let end = match &self.end {
            Some(end) => format!(
                "{:>6} s  {:<13} exit {}",
                end.seconds,
                format!("{:?}", end.ended_by).to_lowercase(),
                end.exit_code_hex.as_deref().unwrap_or("unknown"),
            ),
            None => "not ended".into(),
        };
        format!(
            "{}  {}{}  {}  {}",
            self.run_id,
            &self.commit[..self.commit.len().min(10)],
            if self.dirty { "+" } else { " " },
            self.settings.build.name(),
            end
        )
    }
}

impl End {
    /// The end of a title that ran from `title_started` to `ended`.
    pub fn new(
        title_started: SystemTime,
        ended: SystemTime,
        exit_code: Option<i64>,
        ended_by: EndedBy,
        crashes: Vec<CrashLog>,
    ) -> Self {
        Self {
            ended: rfc3339(ended),
            seconds: ended
                .duration_since(title_started)
                .unwrap_or_default()
                .as_secs(),
            exit_code,
            // Windows exit codes are 32 bits; a crash shows as an NTSTATUS.
            exit_code_hex: exit_code.map(|code| format!("0x{:08X}", code as u32)),
            ended_by,
            crashes,
        }
    }
}

/// `time` as UTC, `2026-11-28T03:04:05Z`.
pub fn rfc3339(time: SystemTime) -> String {
    let (y, mo, d, h, mi, s) = civil(time);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// A run id: when it started (UTC) and who started it. Sorts by time.
pub fn run_id(time: SystemTime, session: &str) -> String {
    let (y, mo, d, h, mi, s) = civil(time);
    let session: String = session
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    format!("{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}-{session}")
}

/// Year, month, day, hour, minute, second of `time` in UTC.
fn civil(time: SystemTime) -> (i64, u32, u32, u32, u32, u32) {
    let secs = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();
    let (days, rest) = ((secs / 86_400) as i64, (secs % 86_400) as u32);
    // Days to a date in the proleptic Gregorian calendar, by 400-year eras
    // that start on 1 March.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day, rest / 3600, rest / 60 % 60, rest % 60)
}

/// The size of every `crashes.log` one folder below `data_dir` (the title
/// writes it in its own folder there).
pub fn crash_log_sizes(data_dir: &Path) -> BTreeMap<PathBuf, u64> {
    let mut sizes = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(data_dir) else {
        return sizes;
    };
    for entry in entries.flatten() {
        let path = entry.path().join("crashes.log");
        if let Ok(meta) = std::fs::metadata(&path) {
            sizes.insert(path, meta.len());
        }
    }
    sizes
}

/// Saves what was appended to each `crashes.log` since `before` into
/// `run_dir`, and lists it.
pub fn collect_crashes(
    before: &BTreeMap<PathBuf, u64>,
    data_dir: &Path,
    run_dir: &Path,
) -> Result<Vec<CrashLog>> {
    let mut crashes = Vec::new();
    for (path, size) in crash_log_sizes(data_dir) {
        let old = before.get(&path).copied().unwrap_or(0);
        // A log that shrank was replaced: all of it is new.
        let from = if size < old { 0 } else { old };
        if size == from {
            continue;
        }
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let added = &bytes[(from as usize).min(bytes.len())..];
        let name = if crashes.is_empty() {
            "crashes.log".to_string()
        } else {
            format!("crashes-{}.log", crashes.len() + 1)
        };
        let copy = run_dir.join(name);
        victory_core::ensure_outside_repo(&copy)?;
        std::fs::write(&copy, added).with_context(|| format!("writing {}", copy.display()))?;
        crashes.push(CrashLog {
            path,
            copy,
            bytes_added: added.len() as u64,
        });
    }
    Ok(crashes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_world::settings::{Overrides, Profile};

    fn at(secs: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(secs)
    }

    fn manifest(run_dir: &Path) -> Manifest {
        let settings = Settings::resolve(Profile::default(), Overrides::default()).unwrap();
        Manifest {
            run_id: run_id(at(1_795_662_245), "card"),
            session: "card".into(),
            commit: "2d1e37d5aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            dirty: true,
            exe: PathBuf::from("victory.exe"),
            exe_copy: PathBuf::from("victory-run.exe"),
            args: vec!["run".into(), "--execute".into()],
            env: settings.child_env(Some(&run_dir.join("frames"))),
            settings,
            inherited_env: BTreeMap::new(),
            log: run_dir.join("victory.log"),
            capture_dir: Some(run_dir.join("frames")),
            pid: Some(4242),
            started: rfc3339(at(1_795_662_245)),
            title_started: Some(rfc3339(at(1_795_662_545))),
            end: None,
        }
    }

    #[test]
    fn times_are_utc_calendar_dates() {
        assert_eq!(rfc3339(at(0)), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(at(951_782_400)), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339(at(1_795_662_245)), "2026-11-26T03:04:05Z");
        assert_eq!(rfc3339(at(4_102_444_799)), "2099-12-31T23:59:59Z");
    }

    #[test]
    fn a_run_id_sorts_by_time_and_is_a_safe_folder_name() {
        let id = run_id(at(1_795_662_245), "card agent/2");
        assert_eq!(id, "20261126-030405-card-agent-2");
        assert!(id < run_id(at(1_795_662_246), "a"));
    }

    #[test]
    fn a_manifest_reads_back_as_written_before_and_after_the_end() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = manifest(dir.path());
        m.write(dir.path()).unwrap();
        let path = dir.path().join(FILE_NAME);
        let started = Manifest::read(&path).unwrap();
        assert_eq!(started, m);
        assert_eq!(started.end, None);
        assert_eq!(started.env["VICTORY_TEXTURE_MIP_DROP"], "2");
        assert!(started.env.contains_key("VICTORY_CAPTURE"));

        m.end = Some(End::new(
            at(100),
            at(1_900),
            Some(0xC000_0005),
            EndedBy::Title,
            Vec::new(),
        ));
        m.write(dir.path()).unwrap();
        let ended = Manifest::read(&path).unwrap().end.unwrap();
        assert_eq!(ended.seconds, 1_800);
        assert_eq!(ended.exit_code_hex.as_deref(), Some("0xC0000005"));
        assert_eq!(ended.ended_by, EndedBy::Title);
    }

    #[test]
    fn the_length_of_a_run_counts_from_the_title_start_not_from_the_run_id() {
        let dir = tempfile::tempdir().unwrap();
        // Asked for at 02:00:00, title started five minutes later (the
        // build), ended twenty minutes after that.
        let (asked, launched, ended) = (at(7_200), at(7_500), at(8_700));
        let mut m = manifest(dir.path());
        m.run_id = run_id(asked, "card");
        m.started = rfc3339(asked);
        m.title_started = Some(rfc3339(launched));
        m.end = Some(End::new(
            launched,
            ended,
            Some(0),
            EndedBy::Title,
            Vec::new(),
        ));
        assert_eq!(m.run_id, "19700101-020000-card");
        assert_eq!(m.started, "1970-01-01T02:00:00Z");
        assert_eq!(m.title_started.as_deref(), Some("1970-01-01T02:05:00Z"));
        let end = m.end.as_ref().unwrap();
        assert_eq!(end.ended, "1970-01-01T02:25:00Z");
        assert_eq!(end.seconds, 1_200);
    }

    #[test]
    fn a_manifest_without_the_title_start_still_reads() {
        let dir = tempfile::tempdir().unwrap();
        let mut value = serde_json::to_value(manifest(dir.path())).unwrap();
        value.as_object_mut().unwrap().remove("title_started");
        let old: Manifest = serde_json::from_value(value).unwrap();
        assert_eq!(old.title_started, None);
    }

    #[test]
    fn the_summary_line_names_commit_build_and_outcome() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = manifest(dir.path());
        assert!(m.summary().ends_with("not ended"), "{}", m.summary());
        m.end = Some(End::new(at(0), at(90), Some(1), EndedBy::Stop, Vec::new()));
        let line = m.summary();
        assert!(line.contains("2d1e37d5aa+"), "{line}");
        assert!(line.contains("diag"), "{line}");
        assert!(line.contains("stop"), "{line}");
        assert!(line.contains("0x00000001"), "{line}");
    }

    #[test]
    fn a_manifest_is_not_written_into_the_source_tree() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let m = manifest(&inside);
        assert!(m.write(&inside).is_err());
        assert!(!inside.join(FILE_NAME).exists());
    }

    #[test]
    fn only_what_a_run_added_to_crashes_log_is_collected() {
        let data = tempfile::tempdir().unwrap();
        let run = tempfile::tempdir().unwrap();
        let (quiet, noisy) = (data.path().join("0000000A"), data.path().join("0000000B"));
        std::fs::create_dir_all(&quiet).unwrap();
        std::fs::create_dir_all(&noisy).unwrap();
        std::fs::write(quiet.join("crashes.log"), "old\n").unwrap();
        std::fs::write(noisy.join("crashes.log"), "old\n").unwrap();
        let before = crash_log_sizes(data.path());
        assert_eq!(before.len(), 2);

        assert!(
            collect_crashes(&before, data.path(), run.path())
                .unwrap()
                .is_empty()
        );
        std::fs::write(noisy.join("crashes.log"), "old\nnew fault\n").unwrap();
        let crashes = collect_crashes(&before, data.path(), run.path()).unwrap();
        assert_eq!(crashes.len(), 1);
        assert_eq!(crashes[0].path, noisy.join("crashes.log"));
        assert_eq!(crashes[0].bytes_added, 10);
        assert_eq!(
            std::fs::read_to_string(&crashes[0].copy).unwrap(),
            "new fault\n"
        );
    }

    #[test]
    fn a_crashes_log_made_or_replaced_during_the_run_is_taken_whole() {
        let data = tempfile::tempdir().unwrap();
        let run = tempfile::tempdir().unwrap();
        let title = data.path().join("0000000A");
        std::fs::create_dir_all(&title).unwrap();
        let none = crash_log_sizes(data.path());
        std::fs::write(title.join("crashes.log"), "first\n").unwrap();
        let made = collect_crashes(&none, data.path(), run.path()).unwrap();
        assert_eq!(made[0].bytes_added, 6);

        let mut longer = BTreeMap::new();
        longer.insert(title.join("crashes.log"), 100);
        let replaced = collect_crashes(&longer, data.path(), run.path()).unwrap();
        assert_eq!(replaced[0].bytes_added, 6);
    }
}
