//! The card lock: one title run on the GPU at a time, across sessions.
//!
//! The lock is a small JSON file. A run creates it before it starts the
//! title and removes it when the title has ended. A lock is live while the
//! title process or the launcher that waits on it still runs; anything else
//! is stale and is cleared by the next run. A process is recognized by its
//! id and its image name together, so a recycled id is not taken for it.

use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// What the lock file holds. Only `pid` matters to another tool that
/// shares the file: a lock written elsewhere with fewer fields still reads,
/// and is live while its `pid` runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Lock {
    /// Who claimed the card (`--session`).
    pub session: String,
    pub run_id: String,
    /// When the claim was made (UTC, RFC 3339).
    pub started: String,
    /// The `xtask` process that waits on the title (0: there is none).
    pub launcher_pid: u32,
    pub launcher_image: String,
    /// The title process, once it has started.
    pub pid: Option<u32>,
    /// The image name of the title process: the run's own copy of the exe.
    pub image: Option<String>,
    /// The run's manifest.
    pub manifest: PathBuf,
    /// Set by `run-world stop` before it ends the title.
    pub stop_requested: bool,
}

impl Default for Lock {
    fn default() -> Self {
        Self {
            session: "unknown".into(),
            run_id: "unknown".into(),
            started: "unknown".into(),
            launcher_pid: 0,
            launcher_image: String::new(),
            pid: None,
            image: None,
            manifest: PathBuf::new(),
            stop_requested: false,
        }
    }
}

/// Looks up running processes.
pub trait Processes {
    /// The image name of the running process `pid`, if there is one.
    fn image_of(&self, pid: u32) -> Option<String>;
}

/// Whether `pid` runs and is `image` (compared without case).
pub fn runs_as(processes: &dyn Processes, pid: u32, image: &str) -> bool {
    processes
        .image_of(pid)
        .is_some_and(|found| found.eq_ignore_ascii_case(image))
}

impl Lock {
    /// Whether the title of this lock still runs.
    pub fn title_runs(&self, processes: &dyn Processes) -> bool {
        match (self.pid, &self.image) {
            (Some(pid), Some(image)) => runs_as(processes, pid, image),
            // No image to know it by: any process with that id.
            (Some(pid), None) => processes.image_of(pid).is_some(),
            (None, _) => false,
        }
    }

    /// Whether the launcher of this lock still runs.
    pub fn launcher_runs(&self, processes: &dyn Processes) -> bool {
        self.launcher_pid != 0 && runs_as(processes, self.launcher_pid, &self.launcher_image)
    }

    /// Whether the card is still claimed: the title runs, or its launcher
    /// does (it is about to start the title or to release the lock).
    pub fn live(&self, processes: &dyn Processes) -> bool {
        self.title_runs(processes) || self.launcher_runs(processes)
    }
}

/// What a lock file was found to be.
#[derive(Debug, PartialEq, Eq)]
pub enum Found {
    Free,
    Live(Lock),
    Stale(Lock),
    /// A lock as scripts older than this tool wrote it, `<pid> <session>`
    /// on one line: live while that process runs, whatever its image.
    /// `text` is the line; `image` what the process runs, if it does.
    BarePid {
        pid: u32,
        text: String,
        image: Option<String>,
    },
    /// Not a lock we can read: left alone, since another run may be in the
    /// middle of writing it.
    Unreadable(String),
}

/// Reads the lock file at `path` and says whether its run is alive.
pub fn inspect(path: &Path, processes: &dyn Processes) -> Result<Found> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Found::Free),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let line = text.trim();
    if line.lines().count() == 1
        && let Some(Ok(pid)) = line.split_whitespace().next().map(str::parse::<u32>)
    {
        return Ok(Found::BarePid {
            pid,
            text: line.to_string(),
            image: processes.image_of(pid),
        });
    }
    Ok(match serde_json::from_str::<Lock>(&text) {
        Ok(lock) if lock.live(processes) => Found::Live(lock),
        Ok(lock) => Found::Stale(lock),
        Err(e) => Found::Unreadable(e.to_string()),
    })
}

/// How long a run waits for another that is claiming the card right now.
const GUARD_WAIT: Duration = Duration::from_secs(10);

/// The right to look at the lock file and replace it: a second file,
/// `<lock>.guard`, made exclusively and removed when dropped. It is held
/// only while the lock is read, cleared and written (milliseconds), never
/// during a run. Without it, two runs that both find a stale lock could
/// each clear the other's fresh one and both start a title.
struct Guard(PathBuf);

impl Guard {
    fn take(lock: &Path, wait: Duration) -> Result<Self> {
        let mut name = lock.file_name().unwrap_or_default().to_os_string();
        name.push(".guard");
        let path = lock.with_file_name(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("making {}", parent.display()))?;
        }
        let start = Instant::now();
        loop {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    // Who holds it, for whoever finds it left behind.
                    let _ = write!(file, "{}", std::process::id());
                    return Ok(Self(path));
                }
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
                // Windows refuses a name whose file is being deleted.
                Err(e) if e.kind() == ErrorKind::PermissionDenied => {}
                Err(e) => return Err(e).with_context(|| format!("creating {}", path.display())),
            }
            if start.elapsed() >= wait {
                // Not removed for the holder: taking it away from a run
                // that is only slow would bring the race back.
                bail!(
                    "{} is held: another run-world is claiming the card. If none is \
                     running (one was killed in the act), delete that file",
                    path.display()
                );
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Claims the card for `lock`: fails while a live lock exists, clears a
/// stale one. Looking, clearing and writing happen under the guard, so of
/// two runs that start together only one gets the card.
pub fn acquire(path: &Path, lock: &Lock, processes: &dyn Processes) -> Result<()> {
    acquire_within(path, lock, processes, GUARD_WAIT)
}

fn acquire_within(
    path: &Path,
    lock: &Lock,
    processes: &dyn Processes,
    wait: Duration,
) -> Result<()> {
    let _guard = Guard::take(path, wait)?;
    match inspect(path, processes)? {
        Found::Free => {}
        Found::Live(held) => bail!(
            "the card is taken: session {} holds {} since {} (run {}, title pid {}). \
             Wait for it, or end it with `cargo xtask run-world stop`",
            held.session,
            path.display(),
            held.started,
            held.run_id,
            held.pid.map_or("not started".into(), |p| p.to_string()),
        ),
        Found::BarePid {
            pid,
            text,
            image: Some(image),
        } => bail!(
            "the card is taken: {} holds `{text}` and pid {pid} ({image}) still runs. \
             Wait for it, or end it with `cargo xtask run-world stop`",
            path.display(),
        ),
        Found::BarePid {
            pid,
            text,
            image: None,
        } => {
            remove_if_same(path, &text)?;
            println!("cleared a stale lock: pid {pid} is gone");
        }
        Found::Stale(stale) => {
            remove_if_same(path, &text_of(&stale))?;
            println!(
                "cleared a stale lock: run {} of session {} (its processes are gone)",
                stale.run_id, stale.session
            );
        }
        Found::Unreadable(why) => bail!(
            "{} is not a lock this tool wrote ({why}); look at it and delete it by hand \
             if no run is using the card",
            path.display()
        ),
    }
    // Exclusive all the same: a tool that does not know the guard may have
    // written its own lock since we looked.
    match create(path, lock) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::AlreadyExists => bail!(
            "{} appeared while the card was being claimed: something else took it",
            path.display()
        ),
        Err(e) => Err(e).with_context(|| format!("creating {}", path.display())),
    }
}

fn create(path: &Path, lock: &Lock) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(text_of(lock).as_bytes())
}

fn text_of(lock: &Lock) -> String {
    serde_json::to_string_pretty(lock).expect("a lock is plain data")
}

/// Removes the lock file if it still holds `stale`, under the guard.
pub fn clear_stale(path: &Path, stale: &str) -> Result<()> {
    let _guard = Guard::take(path, GUARD_WAIT)?;
    remove_if_same(path, stale)
}

/// Removes the lock file if it holds `stale`; refuses if it now holds
/// something else. The caller holds the guard.
fn remove_if_same(path: &Path, stale: &str) -> Result<()> {
    let now = match std::fs::read_to_string(path) {
        Ok(now) => now,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    if !same_lock(&now, stale) {
        bail!("another run took the card while a stale lock was being cleared");
    }
    std::fs::remove_file(path).with_context(|| format!("removing {}", path.display()))
}

/// Whether two lock texts say the same, however they are laid out and
/// whichever fields they leave to their defaults.
fn same_lock(a: &str, b: &str) -> bool {
    let lock = |t: &str| serde_json::from_str::<Lock>(t).ok();
    a.trim() == b.trim() || (lock(a).is_some() && lock(a) == lock(b))
}

/// Rewrites the lock of run `lock.run_id` (to add the title's pid, or the
/// stop request). Fails if the file now belongs to another run.
pub fn update(path: &Path, lock: &Lock) -> Result<()> {
    let held = read(path)?;
    if held.as_ref().map(|h| &h.run_id) != Some(&lock.run_id) {
        bail!(
            "{} no longer belongs to run {}",
            path.display(),
            lock.run_id
        );
    }
    std::fs::write(path, text_of(lock)).with_context(|| format!("writing {}", path.display()))
}

/// The lock at `path`, if the file exists and reads as one.
pub fn read(path: &Path) -> Result<Option<Lock>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(serde_json::from_str(&text).ok()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// Gives the card back, if the lock is still that of `run_id`.
pub fn release(path: &Path, run_id: &str) -> Result<()> {
    if read(path)?.is_some_and(|held| held.run_id == run_id) {
        std::fs::remove_file(path).with_context(|| format!("removing {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A process table for tests.
    #[derive(Default)]
    pub struct Table(pub HashMap<u32, String>);

    impl Processes for Table {
        fn image_of(&self, pid: u32) -> Option<String> {
            self.0.get(&pid).cloned()
        }
    }

    pub fn table(rows: &[(u32, &str)]) -> Table {
        Table(rows.iter().map(|(p, i)| (*p, i.to_string())).collect())
    }

    pub fn lock(run_id: &str, pid: Option<u32>) -> Lock {
        Lock {
            session: "card".into(),
            run_id: run_id.into(),
            started: "2026-11-28T03:00:00Z".into(),
            launcher_pid: 10,
            launcher_image: "xtask.exe".into(),
            pid,
            image: pid.map(|_| format!("victory-{run_id}.exe")),
            manifest: PathBuf::from("manifest.json"),
            stop_requested: false,
        }
    }

    #[test]
    fn a_lock_is_live_while_its_title_or_its_launcher_runs() {
        let l = lock("a", Some(20));
        assert!(l.live(&table(&[(20, "victory-a.exe")])));
        assert!(l.live(&table(&[(10, "XTASK.EXE")])));
        assert!(!l.live(&table(&[])));
    }

    #[test]
    fn a_recycled_pid_does_not_keep_a_lock_alive() {
        let l = lock("a", Some(20));
        assert!(!l.live(&table(&[(20, "notepad.exe"), (10, "cargo.exe")])));
        // Another run's copy of the exe is not this run either.
        assert!(!l.live(&table(&[(20, "victory-b.exe")])));
    }

    #[test]
    fn a_live_lock_refuses_a_second_run_and_names_the_holder() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("card.lock");
        let running = table(&[(20, "victory-a.exe")]);
        acquire(&path, &lock("a", Some(20)), &running).unwrap();
        let refused = acquire(&path, &lock("b", None), &running).unwrap_err();
        let message = refused.to_string();
        assert!(message.contains("session card"), "{message}");
        assert!(message.contains("pid 20"), "{message}");
        assert_eq!(read(&path).unwrap().unwrap().run_id, "a");
    }

    #[test]
    fn a_stale_lock_is_cleared_and_the_card_claimed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("card.lock");
        acquire(&path, &lock("a", Some(20)), &table(&[])).unwrap();
        assert_eq!(
            inspect(&path, &table(&[])).unwrap(),
            Found::Stale(lock("a", Some(20)))
        );
        acquire(&path, &lock("b", None), &table(&[])).unwrap();
        assert_eq!(read(&path).unwrap().unwrap().run_id, "b");
        // Nothing is left beside the lock.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn a_file_that_is_not_a_lock_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("card.lock");
        std::fs::write(&path, "").unwrap();
        assert!(acquire(&path, &lock("b", None), &table(&[])).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "");
    }

    #[test]
    fn a_lock_that_is_only_a_pid_holds_the_card_while_that_pid_runs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("card.lock");
        std::fs::write(&path, "22624 card\r\n").unwrap();
        let running = table(&[(22624, "victory-card.exe")]);
        let refused = acquire(&path, &lock("b", None), &running).unwrap_err();
        let message = refused.to_string();
        assert!(message.contains("`22624 card`"), "{message}");
        assert!(
            message.contains("pid 22624 (victory-card.exe)"),
            "{message}"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "22624 card\r\n");
        // A pid alone reads the same way.
        std::fs::write(&path, "22624").unwrap();
        assert!(acquire(&path, &lock("b", None), &running).is_err());

        acquire(&path, &lock("b", None), &table(&[])).unwrap();
        assert_eq!(read(&path).unwrap().unwrap().run_id, "b");
    }

    #[test]
    fn a_lock_taken_while_a_stale_one_is_cleared_is_given_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("card.lock");
        // The file no longer holds what was judged stale: another run's.
        acquire(&path, &lock("fresh", Some(20)), &table(&[])).unwrap();
        assert!(clear_stale(&path, &text_of(&lock("old", Some(7)))).is_err());
        assert_eq!(read(&path).unwrap().unwrap().run_id, "fresh");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn while_another_run_claims_the_card_nothing_is_cleared() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("card.lock");
        acquire(&path, &lock("old", Some(20)), &table(&[])).unwrap();
        let guard = Guard::take(&path, Duration::ZERO).unwrap();
        assert!(dir.path().join("card.lock.guard").is_file());
        let wait = Duration::from_millis(60);
        let refused = acquire_within(&path, &lock("b", None), &table(&[]), wait).unwrap_err();
        assert!(refused.to_string().contains("card.lock.guard"), "{refused}");
        assert_eq!(read(&path).unwrap().unwrap().run_id, "old");
        drop(guard);
        assert!(!dir.path().join("card.lock.guard").exists());
        acquire(&path, &lock("b", None), &table(&[])).unwrap();
        assert_eq!(read(&path).unwrap().unwrap().run_id, "b");
    }

    #[test]
    fn of_many_runs_that_find_a_stale_lock_together_one_gets_the_card() {
        for _ in 0..20 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("card.lock");
            acquire(&path, &lock("old", Some(20)), &table(&[])).unwrap();
            // Every racer's launcher runs, so whichever lock is written is
            // live; the old lock's processes are gone.
            let winners: Vec<String> = std::thread::scope(|scope| {
                let racers: Vec<_> = (0..8u32)
                    .map(|i| {
                        let path = &path;
                        scope.spawn(move || {
                            let mut mine = lock(&format!("racer-{i}"), None);
                            mine.launcher_pid = 100 + i;
                            let running: Vec<(u32, &str)> =
                                (0..8).map(|j| (100 + j, "xtask.exe")).collect();
                            acquire(path, &mine, &table(&running))
                                .ok()
                                .map(|()| mine.run_id)
                        })
                    })
                    .collect();
                racers
                    .into_iter()
                    .filter_map(|racer| racer.join().unwrap())
                    .collect()
            });
            assert_eq!(winners.len(), 1, "{winners:?}");
            assert_eq!(read(&path).unwrap().unwrap().run_id, winners[0]);
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        }
    }

    #[test]
    fn the_lock_folder_is_made_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deep").join("card.lock");
        acquire(&path, &lock("a", None), &table(&[])).unwrap();
        assert!(path.is_file());
    }

    #[test]
    fn only_the_owning_run_updates_or_releases_the_lock() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("card.lock");
        acquire(&path, &lock("a", None), &table(&[])).unwrap();
        update(&path, &lock("a", Some(20))).unwrap();
        assert_eq!(read(&path).unwrap().unwrap().pid, Some(20));
        assert!(update(&path, &lock("b", Some(30))).is_err());
        release(&path, "b").unwrap();
        assert!(path.is_file());
        release(&path, "a").unwrap();
        assert!(!path.exists());
        // Releasing twice is fine.
        release(&path, "a").unwrap();
    }

    #[test]
    fn a_lock_another_tool_wrote_with_fewer_fields_still_holds_the_card() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("card.lock");
        let theirs = r#"{ "run_id": "gpu-tests", "pid": 23272, "session": "card" }"#;
        std::fs::write(&path, theirs).unwrap();
        let running = table(&[(23272, "some-test.exe")]);
        let Found::Live(held) = inspect(&path, &running).unwrap() else {
            panic!("a lock whose pid runs is live");
        };
        assert_eq!((held.session.as_str(), held.pid), ("card", Some(23272)));
        assert!(!held.launcher_runs(&table(&[(0, "")])));
        let refused = acquire(&path, &lock("b", None), &running).unwrap_err();
        assert!(refused.to_string().contains("session card"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), theirs);

        acquire(&path, &lock("b", None), &table(&[])).unwrap();
        assert_eq!(read(&path).unwrap().unwrap().run_id, "b");
    }

    #[test]
    fn json_that_is_not_an_object_is_not_a_lock() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("card.lock");
        std::fs::write(&path, "[1, 2]").unwrap();
        assert!(matches!(
            inspect(&path, &table(&[])).unwrap(),
            Found::Unreadable(_)
        ));
    }
}
