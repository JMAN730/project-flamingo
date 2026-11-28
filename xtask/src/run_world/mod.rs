//! `cargo xtask run-world`: one way to launch a world run of the title, so
//! that several sessions can share one card and compare what they ran.
//!
//! A run claims the card with a lock file, copies `victory.exe` to a name
//! of its own, starts the title with the known-good environment (changed
//! by a profile file or flags), saves the title's output and its captured
//! frames in a run folder outside the repository, and writes a manifest
//! there. `stop` ends the run by the process id in the lock, never by
//! image name: other sessions run `victory.exe` tools of their own.

mod lock;
mod manifest;
mod settings;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, bail, ensure};
use victory_core::ensure_outside_repo;

use lock::{Found, Lock, Processes};
use manifest::{End, EndedBy, Manifest};
use settings::{Build, Overrides, Profile, Settings};

/// Where runs and the lock live unless told otherwise: the system drive,
/// outside every checkout.
const DEFAULT_RUNS_DIR: &str = r"C:\victory-data\runs";
const DEFAULT_LOCK: &str = r"C:\victory-data\card.lock";

/// How long `stop` waits for an ended title to leave the process list:
/// Windows takes its time to tear down a process with gigabytes mapped and
/// GPU work in flight (more than 20 s was seen).
const STOP_WAIT: Duration = Duration::from_secs(180);

#[derive(clap::Args)]
#[command(args_conflicts_with_subcommands = true)]
pub struct Args {
    #[command(subcommand)]
    action: Option<Action>,
    #[command(flatten)]
    start: Start,
}

#[derive(clap::Subcommand)]
enum Action {
    /// End the run that holds the card, by the process id in the lock.
    Stop(Shared),
    /// Say who holds the card.
    Status(Shared),
    /// List the runs in the runs folder, oldest first.
    List(Shared),
}

#[derive(clap::Args)]
struct Shared {
    /// The lock file that claims the card.
    #[arg(long, default_value = DEFAULT_LOCK)]
    lock: PathBuf,
    /// The folder that holds one folder per run.
    #[arg(long, default_value = DEFAULT_RUNS_DIR)]
    runs_dir: PathBuf,
}

#[derive(clap::Args)]
struct Start {
    #[command(flatten)]
    shared: Shared,
    /// Who is running: goes in the lock, the run id and the manifest.
    #[arg(long, env = "RUN_WORLD_SESSION", default_value = "unnamed")]
    session: String,
    /// A TOML file with any of: build, console, keys, max_seconds, [env],
    /// [capture] frames/seconds. Flags win over it.
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Which victory.exe: diag (default; the skip list, capture and
    /// scripted keys only exist there) or release.
    #[arg(long, value_enum)]
    build: Option<Build>,
    /// Use this victory.exe instead of building one.
    #[arg(long)]
    exe: Option<PathBuf>,
    /// Do not run cargo build: use the exe already in the target folder.
    #[arg(long)]
    no_build: bool,
    /// The config file with the game paths (sets VICTORY_CONFIG).
    #[arg(long)]
    config: Option<PathBuf>,
    /// The title's executable inside game_dir, when it holds several.
    #[arg(long)]
    title_exe: Option<String>,
    /// The console model: series-s (default) or series-x.
    #[arg(long)]
    console: Option<String>,
    /// Set a VICTORY_ variable (NAME=VALUE); an empty value removes a default.
    #[arg(long = "env", value_name = "NAME=VALUE")]
    env: Vec<String>,
    /// Also save the presented frames <first>-<last>[:<step>].
    #[arg(long, value_name = "RANGE")]
    capture_frames: Option<String>,
    /// Also save every frame presented <from>-<to> seconds after the first.
    #[arg(long, value_name = "SPAN")]
    capture_seconds: Option<String>,
    /// Keys to press, as VICTORY_KEYS takes them (40:Enter,60:Enter); none
    /// drops the profile's.
    #[arg(long)]
    keys: Option<String>,
    /// End the run after this many seconds (default 2700, so that no run
    /// holds the card for good; 0: no limit).
    #[arg(long)]
    max_seconds: Option<u64>,
    /// Where the run's copy of the exe and its D3D12 folder go, in a
    /// folder per run. Default: victory-data\runs in your user folder.
    #[arg(long)]
    bin_dir: Option<PathBuf>,
    /// Print what would run and stop: no lock, no build, no title.
    #[arg(long)]
    dry_run: bool,
}

pub fn run(root: &Path, args: Args) -> Result<()> {
    match args.action {
        Some(Action::Stop(shared)) => stop(&shared),
        Some(Action::Status(shared)) => status(&shared),
        Some(Action::List(shared)) => list(&shared),
        None => start(root, args.start),
    }
}

fn start(root: &Path, args: Start) -> Result<()> {
    let profile = match &args.profile {
        Some(path) => Profile::read(path)?,
        None => Profile::default(),
    };
    let settings = Settings::resolve(
        profile,
        Overrides {
            build: args.build,
            console: args.console,
            env: args.env,
            capture_frames: args.capture_frames,
            capture_seconds: args.capture_seconds,
            keys: args.keys,
            max_seconds: args.max_seconds,
        },
    )?;
    if settings.build == Build::Release {
        let ignored = settings.needs_diagnostics();
        ensure!(
            ignored.is_empty(),
            "a release build has no {}: use --build diag, or drop them \
             (--env VICTORY_SKIP_PIPELINES= removes the skip list)",
            ignored.join(", ")
        );
    }

    if let Some(config) = &args.config {
        // SAFETY: nothing else runs in this process yet; the variable is
        // read by the config loader below and inherited by the title.
        unsafe { std::env::set_var("VICTORY_CONFIG", config) };
    }
    let config = victory_core::Config::load()?;
    let title = title_exe(config.require_game_dir()?, args.title_exe.as_deref())?;

    let started = SystemTime::now();
    let run_id = manifest::run_id(started, &args.session);
    let run_dir = args.shared.runs_dir.join(&run_id);
    let capture_dir = (settings.build == Build::Diag).then(|| run_dir.join("frames"));
    let bin_dir = match &args.bin_dir {
        Some(dir) => dir.clone(),
        None => default_bin_dir(&args.shared.runs_dir),
    }
    .join(&run_id);
    for path in [
        Some(&args.shared.lock),
        Some(&run_dir),
        Some(&bin_dir),
        capture_dir.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        ensure_outside_repo(path)?;
    }

    let env = settings.child_env(capture_dir.as_deref());
    let title_args = vec![
        "run".to_string(),
        title.display().to_string(),
        "--console".into(),
        settings.console.clone(),
        "--execute".into(),
    ];
    if args.dry_run {
        println!("run {run_id} would start in {}", run_dir.display());
        println!("  exe copied to {}", bin_dir.display());
        match settings.max_seconds {
            Some(max) => println!("  ended after {max} s"),
            None => println!("  no time limit"),
        }
        println!(
            "  victory.exe ({}) {}",
            settings.build.name(),
            title_args.join(" ")
        );
        for (name, value) in &env {
            println!("  {name}={value}");
        }
        return Ok(());
    }

    let mut held = Lock {
        session: args.session.clone(),
        run_id: run_id.clone(),
        started: manifest::rfc3339(started),
        launcher_pid: std::process::id(),
        launcher_image: own_image()?,
        pid: None,
        image: None,
        manifest: run_dir.join(manifest::FILE_NAME),
        stop_requested: false,
    };
    // Built before the card is claimed: a build takes minutes and needs no
    // GPU, and the claim is what the other sessions wait on.
    let exe = match &args.exe {
        Some(exe) => exe.clone(),
        None => built_exe(root, settings.build, args.no_build)?,
    };
    // Checked before the claim too: a run that cannot start must not clear
    // or take anybody's lock.
    ensure!(exe.is_file(), "{} is not there", exe.display());
    let beside = exe.parent().context("the exe has a folder")?;
    ensure!(
        beside.join("D3D12").join("D3D12Core.dll").is_file(),
        "the Agility SDK is missing: copy a D3D12 folder (with D3D12Core.dll) to {}",
        beside.display()
    );
    lock::acquire(&args.shared.lock, &held, &System)?;
    let run = Run {
        exe,
        bin_dir,
        settings,
        env,
        title_args,
        started,
    };
    let outcome = launch(root, &args.shared.lock, run, &config, &mut held);
    if let Err(e) = lock::release(&args.shared.lock, &run_id) {
        eprintln!("the lock was not released: {e:#}");
    }
    let code = outcome?;
    // The run's outcome is this command's outcome, so scripts can tell.
    std::process::exit(code.map_or(1, |c| if c == 0 { 0 } else { 1 }));
}

/// What `launch` starts.
struct Run {
    exe: PathBuf,
    /// The run's own folder for the copy of the exe.
    bin_dir: PathBuf,
    settings: Settings,
    env: BTreeMap<String, String>,
    title_args: Vec<String>,
    started: SystemTime,
}

/// Copies the exe and starts the title, waits for it, and writes the
/// manifest. Returns the title's exit code.
fn launch(
    root: &Path,
    lock_path: &Path,
    run: Run,
    config: &victory_core::Config,
    held: &mut Lock,
) -> Result<Option<i64>> {
    let Run {
        exe,
        bin_dir,
        settings,
        env,
        title_args,
        started,
    } = run;
    // The name of the copy is what `stop` and the lock recognize the
    // process by.
    let image = format!("victory-{}.exe", held.run_id);
    let copy = copy_with_agility(&exe, &bin_dir, &image)?;

    let run_dir = held.manifest.parent().expect("set by start").to_path_buf();
    let log = run_dir.join("victory.log");
    let capture_dir = env.get("VICTORY_CAPTURE").map(PathBuf::from);
    std::fs::create_dir_all(&run_dir).with_context(|| format!("making {}", run_dir.display()))?;
    ensure_outside_repo(&log)?;
    let out = std::fs::File::create(&log).with_context(|| format!("creating {}", log.display()))?;
    let crashes_before = manifest::crash_log_sizes(&config.data_dir);

    let mut command = Command::new(&copy);
    command
        .args(&title_args)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(out.try_clone()?)
        .stderr(out);
    let mut inherited = BTreeMap::new();
    for (name, value) in std::env::vars() {
        if !name.starts_with("VICTORY_") {
            continue;
        }
        if settings::KEPT_AMBIENT.contains(&name.as_str()) {
            inherited.insert(name, value);
        } else {
            command.env_remove(&name);
        }
    }
    command.envs(&env);
    let mut child = command
        .spawn()
        .with_context(|| format!("starting {}", copy.display()))?;

    held.pid = Some(child.id());
    held.image = Some(image);
    let (commit, dirty) = commit(root);
    let mut manifest = Manifest {
        run_id: held.run_id.clone(),
        session: held.session.clone(),
        commit,
        dirty,
        exe,
        exe_copy: copy.clone(),
        args: title_args,
        settings: settings.clone(),
        env,
        inherited_env: inherited,
        log: log.clone(),
        capture_dir,
        pid: held.pid,
        started: held.started.clone(),
        end: None,
    };
    // From here on the title runs: whatever fails, wait for it and say how
    // it ended.
    let bookkeeping = lock::update(lock_path, held).and(manifest.write(&run_dir));
    if let Err(e) = &bookkeeping {
        eprintln!("the run goes on, but: {e:#}");
    }
    println!("run {} started: pid {}", held.run_id, child.id());
    println!("  output    {}", log.display());
    println!("  manifest  {}", held.manifest.display());

    let clock = Instant::now();
    let mut ended_by = EndedBy::Title;
    let mut adapter_checked = false;
    let status = loop {
        if let Some(status) = child.try_wait().context("waiting for the title")? {
            break status;
        }
        if !adapter_checked {
            match adapter_verdict(&head_of(&log)) {
                None => {}
                Some(Ok(())) => adapter_checked = true,
                Some(Err(lacks)) => {
                    adapter_checked = true;
                    eprintln!(
                        "WRONG D3D12: the adapter lacks {lacks}. The title did not load the \
                         Agility SDK beside {} and got the system's D3D12 instead; the run \
                         is worthless and is ended. Put the exe folder (--bin-dir) where \
                         only your user can write.",
                        copy.display()
                    );
                    ended_by = EndedBy::SystemD3d12;
                    child.kill().context("ending the title")?;
                }
            }
        }
        if ended_by == EndedBy::Title
            && settings
                .max_seconds
                .is_some_and(|max| clock.elapsed() >= Duration::from_secs(max))
        {
            println!("--max-seconds ran out: ending pid {}", child.id());
            ended_by = EndedBy::MaxSeconds;
            // Our own child: its handle names the process, not an image.
            child.kill().context("ending the title")?;
        }
        std::thread::sleep(Duration::from_millis(500));
    };
    if ended_by == EndedBy::Title
        && lock::read(lock_path)?.is_some_and(|l| l.run_id == held.run_id && l.stop_requested)
    {
        ended_by = EndedBy::Stop;
    }
    let exit_code = status.code().map(i64::from);
    let crashes = manifest::collect_crashes(&crashes_before, &config.data_dir, &run_dir)
        .unwrap_or_else(|e| {
            eprintln!("crashes.log was not collected: {e:#}");
            Vec::new()
        });
    let end = End::new(started, SystemTime::now(), exit_code, ended_by, crashes);
    println!(
        "run {} ended after {} s ({:?}), exit code {}",
        held.run_id,
        end.seconds,
        end.ended_by,
        end.exit_code_hex.as_deref().unwrap_or("unknown")
    );
    for crash in &end.crashes {
        println!(
            "  crashes   {} (+{} bytes)",
            crash.copy.display(),
            crash.bytes_added
        );
    }
    manifest.end = Some(end);
    manifest.write(&run_dir)?;
    // Only what this run put there: the folder is named after the run.
    if let Err(e) = std::fs::remove_dir_all(&bin_dir) {
        eprintln!("{} was not removed: {e}", bin_dir.display());
    }
    if ended_by == EndedBy::SystemD3d12 {
        bail!("the title ran on the system's D3D12, not the Agility SDK");
    }
    Ok(exit_code)
}

/// Where exe copies go unless told otherwise: under the user's own folder.
/// A folder that other users may write to (as `C:\victory-data` is) makes
/// Windows ignore the `D3D12` folder beside the exe and load the system's
/// D3D12, which is too old for the title. Without a user folder, `runs_dir`.
fn default_bin_dir(runs_dir: &Path) -> PathBuf {
    match std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        Some(home) if !home.is_empty() => PathBuf::from(home).join("victory-data").join("runs"),
        _ => runs_dir.to_path_buf(),
    }
}

/// Copies `exe` to `bin_dir` as `image`, with the `D3D12` folder beside it
/// (the Agility SDK the exe loads from its own folder). Returns the copy.
fn copy_with_agility(exe: &Path, bin_dir: &Path, image: &str) -> Result<PathBuf> {
    ensure_outside_repo(bin_dir)?;
    let from = exe.parent().context("the exe has a folder")?.join("D3D12");
    let to = bin_dir.join("D3D12");
    std::fs::create_dir_all(&to).with_context(|| format!("making {}", to.display()))?;
    let copy = bin_dir.join(image);
    std::fs::copy(exe, &copy).with_context(|| format!("copying to {}", copy.display()))?;
    for entry in std::fs::read_dir(&from).with_context(|| format!("reading {}", from.display()))? {
        let path = entry?.path();
        if path.is_file() {
            let name = path.file_name().expect("a listed file has a name");
            std::fs::copy(&path, to.join(name))
                .with_context(|| format!("copying {}", path.display()))?;
        }
    }
    Ok(copy)
}

/// The start of the title's output, where the adapter is reported.
fn head_of(log: &Path) -> String {
    use std::io::Read;
    let mut head = Vec::new();
    if let Ok(file) = std::fs::File::open(log) {
        // Enough for everything up to the device; the log grows to megabytes.
        let _ = file.take(256 * 1024).read_to_end(&mut head);
    }
    String::from_utf8_lossy(&head).into_owned()
}

/// What the title's output so far says about the D3D12 it got: nothing
/// yet (`None`), fine, or the capabilities the chosen adapter lacks when
/// shader model 6.6 is among them. Every D3D12 the Agility SDK brings has
/// shader model 6.6; the one in Windows 10 does not, so lacking it means
/// the SDK beside the exe was not loaded.
fn adapter_verdict(output: &str) -> Option<Result<(), String>> {
    for line in output.lines() {
        let line = without_colours(line);
        if let Some((_, rest)) = line.split_once("lacks=") {
            // `lacks=<what the chosen one lacks> adapters=<every adapter>`.
            let lacks = rest.split(" adapters=").next().unwrap_or(rest).trim();
            if lacks.contains("shader model 6.6") {
                return Some(Err(lacks.to_string()));
            }
        }
        if line.contains("host D3D12 device ready") {
            return Some(Ok(()));
        }
    }
    None
}

/// `line` without terminal colour sequences (the log colours field names,
/// which splits `lacks=` in two).
fn without_colours(line: &str) -> String {
    let mut plain = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // ESC [ ... up to the letter that ends the sequence.
            for end in chars.by_ref() {
                if end.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            plain.push(c);
        }
    }
    plain
}

/// The title's executable: `name` inside `game_dir`, or the only `.exe`
/// at its top.
fn title_exe(game_dir: &Path, name: Option<&str>) -> Result<PathBuf> {
    if let Some(name) = name {
        let exe = game_dir.join(name);
        ensure!(exe.is_file(), "{} is not there", exe.display());
        return Ok(exe);
    }
    let mut found = Vec::new();
    for entry in
        std::fs::read_dir(game_dir).with_context(|| format!("reading {}", game_dir.display()))?
    {
        let path = entry?.path();
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("exe"))
            && path.is_file()
        {
            found.push(path);
        }
    }
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => bail!("no .exe at the top of {}", game_dir.display()),
        n => bail!(
            "{n} executables at the top of {}: name the title's with --title-exe",
            game_dir.display()
        ),
    }
}

/// The `victory.exe` of `build` in the target folder, built first unless
/// told not to.
fn built_exe(root: &Path, build: Build, no_build: bool) -> Result<PathBuf> {
    if !no_build {
        let mut cargo = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
        cargo.current_dir(root).args(["build", "-p", "victory"]);
        match build {
            Build::Diag => cargo.args(["--profile", "diag", "--features", "diagnostics"]),
            Build::Release => cargo.arg("--release"),
        };
        let status = cargo.status().context("running cargo build")?;
        ensure!(status.success(), "cargo build failed");
    }
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .map(|dir| {
            if dir.is_absolute() {
                dir
            } else {
                root.join(dir)
            }
        })
        .unwrap_or_else(|| root.join("target"));
    let name = format!("victory{}", std::env::consts::EXE_SUFFIX);
    Ok(target.join(build.name()).join(name))
}

/// The commit of the working tree and whether it has uncommitted changes.
fn commit(root: &Path) -> (String, bool) {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    (
        git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into()),
        git(&["status", "--porcelain"]).is_some_and(|changes| !changes.is_empty()),
    )
}

fn own_image() -> Result<String> {
    let exe = std::env::current_exe().context("finding this executable")?;
    Ok(exe
        .file_name()
        .context("this executable has a name")?
        .to_string_lossy()
        .into_owned())
}

fn stop(shared: &Shared) -> Result<()> {
    ensure_outside_repo(&shared.lock)?;
    let mut held = match lock::inspect(&shared.lock, &System)? {
        Found::Free => {
            println!("the card is free: nothing to stop");
            return Ok(());
        }
        Found::Unreadable(why) => bail!("{} is not a lock ({why})", shared.lock.display()),
        Found::BarePid { pid, text, image } => {
            return stop_bare_pid(&shared.lock, pid, &text, image);
        }
        Found::Live(held) | Found::Stale(held) => held,
    };
    if held.title_runs(&System) {
        let pid = held.pid.expect("a running title has a pid");
        if held.image.is_none() {
            // A lock from another tool, with no image to know its process
            // by: the id may have been given to something else since.
            let image = System.image_of(pid).unwrap_or_default();
            ensure!(
                is_victory_image(&image),
                "pid {pid} in {} is {image}, not a victory exe: not ending it",
                shared.lock.display()
            );
        }
        held.stop_requested = true;
        lock::update(&shared.lock, &held)?;
        println!(
            "ending run {} of session {}: pid {pid}",
            held.run_id, held.session
        );
        kill(pid)?;
        let gone = wait_until(STOP_WAIT, || !held.title_runs(&System));
        ensure!(gone, "pid {pid} is still running");
    } else if let Some(pid) = held.pid {
        println!("pid {pid} of run {} is already gone", held.run_id);
    }
    // A live launcher writes the end of the manifest and releases the lock.
    if wait_until(Duration::from_secs(10), || !held.launcher_runs(&System)) {
        finish_for_gone_launcher(&held);
        lock::release(&shared.lock, &held.run_id)?;
    } else if lock::read(&shared.lock)?.is_some_and(|l| l.run_id == held.run_id) {
        println!(
            "the launcher (pid {}) still holds the lock; it releases it when it is done",
            held.launcher_pid
        );
        return Ok(());
    }
    println!("the card is free");
    Ok(())
}

/// Ends the run of a lock that holds only a process id. With no run id to
/// know the process by, it is ended only if its image is a `victory` exe:
/// the id may have been given to something else since.
fn stop_bare_pid(path: &Path, pid: u32, text: &str, image: Option<String>) -> Result<()> {
    if let Some(image) = image {
        ensure!(
            is_victory_image(&image),
            "pid {pid} in {} is now {image}, not a victory exe: not ending it. \
             Delete the lock by hand if no run is using the card",
            path.display()
        );
        println!("ending pid {pid} ({image})");
        kill(pid)?;
        let gone = wait_until(STOP_WAIT, || System.image_of(pid).is_none());
        ensure!(gone, "pid {pid} is still running");
    } else {
        println!("pid {pid} is already gone");
    }
    lock::clear_stale(path, text)?;
    println!("the card is free");
    Ok(())
}

/// Whether `image` is one of our executables: `victory.exe`, a copy named
/// `victory-<something>.exe`, or a test binary (`victory_gpu-<hash>.exe`).
fn is_victory_image(image: &str) -> bool {
    let name = image.to_ascii_lowercase();
    let stem = name.strip_suffix(".exe").unwrap_or(&name);
    stem == "victory" || stem.starts_with("victory-") || stem.starts_with("victory_")
}

/// Writes an end into the manifest of a run whose launcher did not.
fn finish_for_gone_launcher(held: &Lock) {
    let Ok(mut manifest) = Manifest::read(&held.manifest) else {
        return;
    };
    if manifest.end.is_some() {
        return;
    }
    let now = SystemTime::now();
    let mut end = End::new(now, now, None, EndedBy::LauncherGone, Vec::new());
    // The start is only kept as text; the length of the run is not known.
    end.seconds = 0;
    manifest.end = Some(end);
    if let Some(dir) = held.manifest.parent()
        && let Err(e) = manifest.write(dir)
    {
        eprintln!("the manifest was not finished: {e:#}");
    }
}

fn wait_until(limit: Duration, done: impl Fn() -> bool) -> bool {
    let start = Instant::now();
    loop {
        if done() {
            return true;
        }
        if start.elapsed() >= limit {
            return false;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn status(shared: &Shared) -> Result<()> {
    match lock::inspect(&shared.lock, &System)? {
        Found::Free => println!("the card is free"),
        Found::Unreadable(why) => println!("{} is not a lock ({why})", shared.lock.display()),
        Found::Live(held) => println!(
            "taken by session {} since {}: run {}, title pid {}, launcher pid {}\n  {}",
            held.session,
            held.started,
            held.run_id,
            held.pid.map_or("not started".into(), |p| p.to_string()),
            held.launcher_pid,
            held.manifest.display()
        ),
        Found::Stale(held) => println!(
            "free: the lock of run {} (session {}) is stale and the next run clears it",
            held.run_id, held.session
        ),
        Found::BarePid {
            pid,
            text,
            image: Some(image),
        } => println!("taken: the lock holds `{text}` and pid {pid} ({image}) runs"),
        Found::BarePid {
            pid, image: None, ..
        } => println!("free: pid {pid} of the lock is gone and the next run clears it"),
    }
    Ok(())
}

fn list(shared: &Shared) -> Result<()> {
    let mut runs = Vec::new();
    let entries = match std::fs::read_dir(&shared.runs_dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!("no runs yet in {}", shared.runs_dir.display());
            return Ok(());
        }
        Err(e) => return Err(e).with_context(|| format!("reading {}", shared.runs_dir.display())),
    };
    for entry in entries.flatten() {
        if let Ok(manifest) = Manifest::read(&entry.path().join(manifest::FILE_NAME)) {
            runs.push(manifest);
        }
    }
    runs.sort_by(|a, b| a.run_id.cmp(&b.run_id));
    for run in &runs {
        println!("{}", run.summary());
    }
    println!("{} runs in {}", runs.len(), shared.runs_dir.display());
    Ok(())
}

/// The processes of this machine, asked from the system's own tools.
struct System;

impl Processes for System {
    #[cfg(windows)]
    fn image_of(&self, pid: u32) -> Option<String> {
        let out = Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output()
            .ok()?;
        tasklist_image(&String::from_utf8_lossy(&out.stdout), pid)
    }

    #[cfg(not(windows))]
    fn image_of(&self, pid: u32) -> Option<String> {
        let out = Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "comm="])
            .output()
            .ok()?;
        let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let name = name.rsplit('/').next().unwrap_or_default().to_string();
        (!name.is_empty()).then_some(name)
    }
}

/// The image name in `tasklist /FO CSV /NH` output for `pid`: the first
/// field of the row whose second field is the pid. With no such process
/// tasklist prints a sentence instead of a row.
#[cfg_attr(not(any(windows, test)), allow(dead_code))]
fn tasklist_image(output: &str, pid: u32) -> Option<String> {
    output.lines().find_map(|line| {
        let mut fields = line.trim().split("\",\"");
        let image = fields.next()?.strip_prefix('"')?;
        let found: u32 = fields.next()?.trim_matches('"').parse().ok()?;
        (found == pid).then(|| image.to_string())
    })
}

/// Ends process `pid`. By id only: an image name would also end the
/// tools other sessions run.
fn kill(pid: u32) -> Result<()> {
    let status = if cfg!(windows) {
        Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .status()
    } else {
        Command::new("kill").args(["-9", &pid.to_string()]).status()
    }
    .context("ending the process")?;
    ensure!(status.success(), "pid {pid} could not be ended");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasklist_rows_give_the_image_of_the_pid_asked_for() {
        let row =
            "\"victory-20261128-030000-card.exe\",\"4242\",\"Console\",\"1\",\"812,000 K\"\r\n";
        assert_eq!(
            tasklist_image(row, 4242).as_deref(),
            Some("victory-20261128-030000-card.exe")
        );
        assert_eq!(tasklist_image(row, 42), None);
        let nothing = "INFO: No tasks are running which match the specified criteria.\r\n";
        assert_eq!(tasklist_image(nothing, 4242), None);
        assert_eq!(tasklist_image("", 4242), None);
    }

    #[test]
    fn an_adapter_without_shader_model_6_6_means_the_system_d3d12() {
        let warn = "WARN victory_gpu::host: no adapter has everything the title needs; \
                    using the one that runs most adapter=Some GPU ";
        let ready = "INFO victory_gpu::host: host D3D12 device ready adapter=Some GPU\n";
        assert_eq!(adapter_verdict("starting\n"), None);
        assert_eq!(adapter_verdict(ready), Some(Ok(())));
        // Mesh shaders alone are missing on the card itself: fine.
        let ours = format!("{warn}lacks=mesh shaders adapters=Some GPU (lacks mesh shaders)\n");
        assert_eq!(adapter_verdict(&ours), None);
        assert_eq!(adapter_verdict(&format!("{ours}{ready}")), Some(Ok(())));
        // Another adapter lacking it does not matter, the chosen one does.
        let other = format!(
            "{warn}lacks=mesh shaders adapters=Some GPU (lacks mesh shaders), \
             Basic (lacks shader model 6.6)\n{ready}"
        );
        assert_eq!(adapter_verdict(&other), Some(Ok(())));
        let system = format!(
            "{warn}\u{1b}[3mlacks\u{1b}[0m\u{1b}[2m=\u{1b}[0mshader model 6.6, mesh shaders \
             \u{1b}[3madapters\u{1b}[0m\u{1b}[2m=\u{1b}[0mSome GPU\n{ready}"
        );
        assert_eq!(
            adapter_verdict(&system),
            Some(Err("shader model 6.6, mesh shaders".into()))
        );
    }

    #[test]
    fn the_exe_is_copied_with_the_agility_folder_beside_it() {
        let built = tempfile::tempdir().unwrap();
        let bin = tempfile::tempdir().unwrap();
        let exe = built.path().join("victory.exe");
        std::fs::write(&exe, "exe").unwrap();
        std::fs::create_dir(built.path().join("D3D12")).unwrap();
        std::fs::write(built.path().join("D3D12").join("D3D12Core.dll"), "core").unwrap();
        let run = bin.path().join("run-1");
        let copy = copy_with_agility(&exe, &run, "victory-run-1.exe").unwrap();
        assert_eq!(copy, run.join("victory-run-1.exe"));
        assert_eq!(std::fs::read_to_string(&copy).unwrap(), "exe");
        assert_eq!(
            std::fs::read_to_string(run.join("D3D12").join("D3D12Core.dll")).unwrap(),
            "core"
        );
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("bin");
        assert!(copy_with_agility(&exe, &inside, "victory-x.exe").is_err());
        assert!(!inside.exists());
    }

    #[test]
    fn exe_copies_default_to_the_users_own_folder() {
        let dir = default_bin_dir(Path::new("runs"));
        if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
            assert_eq!(dir, Path::new(&home).join("victory-data").join("runs"));
        } else {
            assert_eq!(dir, Path::new("runs"));
        }
    }

    #[test]
    fn only_our_own_executables_count_as_victory_images() {
        for ours in [
            "victory.exe",
            "Victory-Card.EXE",
            "victory-20261128-a.exe",
            "victory_gpu-0123abcd.exe",
        ] {
            assert!(is_victory_image(ours), "{ours}");
        }
        for other in ["victoryx.exe", "notepad.exe", "xtask.exe", "my-victory.exe"] {
            assert!(!is_victory_image(other), "{other}");
        }
    }

    #[test]
    fn this_process_is_found_by_its_own_pid_and_image() {
        let image = own_image().unwrap();
        assert!(lock::runs_as(&System, std::process::id(), &image));
        assert!(!lock::runs_as(&System, std::process::id(), "not-this.exe"));
    }

    #[test]
    fn the_title_exe_is_the_named_one_or_the_only_one() {
        let dir = tempfile::tempdir().unwrap();
        assert!(title_exe(dir.path(), None).is_err());
        std::fs::write(dir.path().join("title.EXE"), "").unwrap();
        std::fs::write(dir.path().join("data.bin"), "").unwrap();
        assert_eq!(
            title_exe(dir.path(), None).unwrap(),
            dir.path().join("title.EXE")
        );
        std::fs::write(dir.path().join("other.exe"), "").unwrap();
        assert!(title_exe(dir.path(), None).is_err());
        assert_eq!(
            title_exe(dir.path(), Some("other.exe")).unwrap(),
            dir.path().join("other.exe")
        );
        assert!(title_exe(dir.path(), Some("missing.exe")).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn the_default_folders_are_outside_the_source_tree() {
        ensure_outside_repo(Path::new(DEFAULT_RUNS_DIR)).unwrap();
        ensure_outside_repo(Path::new(DEFAULT_LOCK)).unwrap();
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("runs");
        assert!(ensure_outside_repo(&inside).is_err());
    }
}
