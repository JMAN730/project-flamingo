//! The offline batch: every shader found goes through the translator, and
//! what it cannot handle yet is tallied (decision 0003).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use victory_shader::emit;
#[cfg(windows)]
use victory_shader::emit::host::{self, Host};
#[cfg(windows)]
use victory_shader::emit::validator::Validator;
use victory_shader::structure::{self, Atom, LaneExit};
use victory_shader::{Container, FourCC, HwStage, NativeProgram, RootType, ShaderError, ir};
use victory_shader::{io, resource};

use crate::shader::{MAX_FILE, Source, scan_entries, sources};

/// Per-stage program counts.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct StageCounts {
    vs: usize,
    ps: usize,
    cs: usize,
    ms: usize,
    ls: usize,
    hs: usize,
    es: usize,
    gs: usize,
}

impl StageCounts {
    fn record(&mut self, stage: HwStage) {
        match stage {
            HwStage::Vs => self.vs += 1,
            HwStage::Ps => self.ps += 1,
            HwStage::Cs => self.cs += 1,
            HwStage::Ms => self.ms += 1,
            HwStage::Ls => self.ls += 1,
            HwStage::Hs => self.hs += 1,
            HwStage::Es => self.es += 1,
            HwStage::Gs => self.gs += 1,
        }
    }

    fn merge(&mut self, other: &StageCounts) {
        self.vs += other.vs;
        self.ps += other.ps;
        self.cs += other.cs;
        self.ms += other.ms;
        self.ls += other.ls;
        self.hs += other.hs;
        self.es += other.es;
        self.gs += other.gs;
    }

    fn total(&self) -> usize {
        self.vs + self.ps + self.cs + self.ms + self.ls + self.hs + self.es + self.gs
    }

    /// Stage breakdown printed on each program-level reason line.
    fn format_reason_breakdown(&self) -> Option<String> {
        if self.total() == 0 {
            return None;
        }
        let mut out = format!(
            "vs {}, ps {}, cs {}, ms {}",
            self.vs, self.ps, self.cs, self.ms
        );
        for (name, count) in [
            ("ls", self.ls),
            ("hs", self.hs),
            ("es", self.es),
            ("gs", self.gs),
        ] {
            if count > 0 {
                out.push_str(&format!(", {name} {count}"));
            }
        }
        Some(out)
    }

    /// Breakdown of non-zero unemitted stages (`LS`, `HS`, `ES`, `GS`).
    fn format_unemitted(&self) -> String {
        let mut parts = Vec::new();
        for (name, count) in [
            ("LS", self.ls),
            ("HS", self.hs),
            ("ES", self.es),
            ("GS", self.gs),
        ] {
            if count > 0 {
                parts.push(format!("{name} {count}"));
            }
        }
        parts.join(", ")
    }
}

/// Programs stopped by one reason, split by stage, with the first example.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Failed {
    total: usize,
    stages: StageCounts,
    example: String,
}

fn is_emitted_stage(stage: HwStage) -> bool {
    matches!(stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms)
}

#[derive(Default)]
struct Tally {
    containers: usize,
    duplicates: usize,
    programs: usize,
    translated: usize,
    instructions: usize,
    structured: usize,
    vector: usize,
    guarded: usize,
    /// Guards that read a mask the pass cannot see into.
    opaque: usize,
    lane_branches: usize,
    uniform_branches: usize,
    tracked: usize,
    loaded: usize,
    built: usize,
    constant: usize,
    /// Descriptors and scalar accesses with a computed address term.
    dynamic: usize,
    scalar: usize,
    used: usize,
    /// Used-resource entries whose root parameter no descriptor traces
    /// to, by resource and root type.
    untraced_used: BTreeMap<String, usize>,
    /// Root parameters descriptors trace to that no used entry names.
    unlisted: usize,
    /// Vertex and pixel programs, and those whose I/O was raised.
    io_programs: usize,
    io_raised: usize,
    interpolated: usize,
    vertex_values: usize,
    /// Exports by target kind.
    exports: BTreeMap<&'static str, usize>,
    /// Exports still packed as f16 pairs after the export patch.
    compressed: usize,
    /// Vertex and pixel programs every earlier stage handled, and how far
    /// emission got with them.
    emit_programs: usize,
    emitted: usize,
    validated: usize,
    pipelines: usize,
    /// The same for compute programs.
    cs_programs: usize,
    cs_emitted: usize,
    cs_validated: usize,
    cs_pipelines: usize,
    /// The same for mesh programs.
    ms_programs: usize,
    ms_emitted: usize,
    ms_validated: usize,
    ms_pipelines: usize,
    /// Mesh programs as hosts without mesh shaders run them
    /// (`--mesh-fallback`): tried, emitted, validated, pipelines.
    fallback: [usize; 4],
    /// Pixel programs that need barycentrics, emitted as hosts without them
    /// run them (`--barycentric-fallback`): tried, emitted, validated (with
    /// their geometry shader), pipelines.
    barycentric: [usize; 4],
    /// Pixel programs that need no barycentrics whose emission for hosts
    /// without them is byte for byte their plain one, and those whose is not.
    barycentric_same: [usize; 2],
    /// Programs in stages the emitter does not emit (`LS`, `HS`, `ES`, `GS`)
    /// that were not refused by an earlier pass.
    not_emitted: StageCounts,
    /// Distinct programs refused before or during emission.
    refused: usize,
    /// Emitted programs rejected by the DXIL validator (`--validate`).
    validator_rejected: usize,
    /// Reason → programs stopped by it, split by stage, and the first one.
    failed: BTreeMap<String, Failed>,
    /// With `--list`: per-program result rows, including runtime hashes.
    list: bool,
    lines: Vec<String>,
}

impl Tally {
    fn fail(&mut self, stage: Option<HwStage>, reason: String, example: impl FnOnce() -> String) {
        let entry = if self.list {
            let e = example();
            self.lines.push(format!("{e}  {reason}"));
            self.failed.entry(reason).or_insert(Failed {
                total: 0,
                stages: StageCounts::default(),
                example: e,
            })
        } else {
            self.failed.entry(reason).or_insert_with(|| Failed {
                total: 0,
                stages: StageCounts::default(),
                example: example(),
            })
        };
        entry.total += 1;
        if let Some(stage) = stage {
            entry.stages.record(stage);
        }
    }

    fn emitted_total(&self) -> usize {
        self.emitted + self.cs_emitted + self.ms_emitted
    }

    fn validated_total(&self) -> usize {
        self.validated + self.cs_validated + self.ms_validated
    }

    fn summary_lines(&self, validate: bool) -> Vec<String> {
        let not_emitted = self.not_emitted.total();
        let not_emitted_line = if not_emitted > 0 {
            format!(
                "             {not_emitted} programs in stages not emitted ({})",
                self.not_emitted.format_unemitted()
            )
        } else {
            "             0 programs in stages not emitted".to_owned()
        };
        let emitted = self.emitted_total();
        let outcome_line = if validate || self.validator_rejected > 0 {
            format!(
                "             {emitted} emitted ({} validated, {} rejected by validator), \
                 {} refused, {not_emitted} in stages not emitted ({} total)",
                self.validated_total(),
                self.validator_rejected,
                self.refused,
                self.programs
            )
        } else {
            format!(
                "             {emitted} emitted, {} refused, {not_emitted} in stages not emitted \
                 ({} total)",
                self.refused, self.programs
            )
        };
        vec![
            format!(
                "containers   {} ({} duplicates skipped)",
                self.containers, self.duplicates
            ),
            format!(
                "programs     {} translated of {} ({} IR instructions)",
                self.translated, self.programs, self.instructions
            ),
            format!(
                "structured   {} of {} ({} vector instructions, {} guarded, {} on opaque masks; \
                 branches {} per lane, {} uniform)",
                self.structured,
                self.translated,
                self.vector,
                self.guarded,
                self.opaque,
                self.lane_branches,
                self.uniform_branches
            ),
            format!(
                "resources    {} of {} tracked (descriptors {} loaded, {} built, {} constant; \
                 {} scalar accesses; {} with a computed term)",
                self.tracked,
                self.translated,
                self.loaded,
                self.built,
                self.constant,
                self.scalar,
                self.dynamic
            ),
            format!(
                "  used-resource entries {}: {} not traced {:?}; {} traced parameters not listed",
                self.used,
                self.untraced_used.values().sum::<usize>(),
                self.untraced_used,
                self.unlisted
            ),
            format!(
                "I/O          {} of {} vertex and pixel programs raised ({} interpolations, \
                 {} vertex values; exports {:?}, {} still f16 pairs)",
                self.io_raised,
                self.io_programs,
                self.interpolated,
                self.vertex_values,
                self.exports,
                self.compressed
            ),
            format!(
                "DXIL         {} emitted of {} vertex and pixel programs; {} validated, {} pipelines",
                self.emitted, self.emit_programs, self.validated, self.pipelines
            ),
            format!(
                "             {} emitted of {} compute programs; {} validated, {} pipelines",
                self.cs_emitted, self.cs_programs, self.cs_validated, self.cs_pipelines
            ),
            format!(
                "             {} emitted of {} mesh programs; {} validated, {} pipelines",
                self.ms_emitted, self.ms_programs, self.ms_validated, self.ms_pipelines
            ),
            not_emitted_line,
            outcome_line,
        ]
    }

    fn failure_lines(&self) -> Vec<String> {
        let mut failed: Vec<_> = self.failed.iter().collect();
        failed.sort_by_key(|(_, f)| std::cmp::Reverse(f.total));
        failed
            .into_iter()
            .map(|(reason, f)| match f.stages.format_reason_breakdown() {
                Some(stages) => {
                    format!(
                        "  {:>7}  {reason}  [{stages}]  (e.g. {})",
                        f.total, f.example
                    )
                }
                None => format!("  {:>7}  {reason}  (e.g. {})", f.total, f.example),
            })
            .collect()
    }

    fn merge(&mut self, other: Tally) {
        self.containers += other.containers;
        self.duplicates += other.duplicates;
        self.programs += other.programs;
        self.translated += other.translated;
        self.instructions += other.instructions;
        self.structured += other.structured;
        self.vector += other.vector;
        self.guarded += other.guarded;
        self.opaque += other.opaque;
        self.lane_branches += other.lane_branches;
        self.uniform_branches += other.uniform_branches;
        self.tracked += other.tracked;
        self.loaded += other.loaded;
        self.built += other.built;
        self.constant += other.constant;
        self.dynamic += other.dynamic;
        self.scalar += other.scalar;
        self.used += other.used;
        for (k, n) in other.untraced_used {
            *self.untraced_used.entry(k).or_default() += n;
        }
        self.unlisted += other.unlisted;
        self.io_programs += other.io_programs;
        self.io_raised += other.io_raised;
        self.interpolated += other.interpolated;
        self.vertex_values += other.vertex_values;
        self.compressed += other.compressed;
        self.emit_programs += other.emit_programs;
        self.emitted += other.emitted;
        self.validated += other.validated;
        self.pipelines += other.pipelines;
        self.cs_programs += other.cs_programs;
        self.cs_emitted += other.cs_emitted;
        self.cs_validated += other.cs_validated;
        self.cs_pipelines += other.cs_pipelines;
        self.ms_programs += other.ms_programs;
        self.ms_emitted += other.ms_emitted;
        self.ms_validated += other.ms_validated;
        self.ms_pipelines += other.ms_pipelines;
        self.not_emitted.merge(&other.not_emitted);
        self.refused += other.refused;
        self.validator_rejected += other.validator_rejected;
        for (mine, theirs) in self.fallback.iter_mut().zip(other.fallback) {
            *mine += theirs;
        }
        for (mine, theirs) in self.barycentric.iter_mut().zip(other.barycentric) {
            *mine += theirs;
        }
        for (mine, theirs) in self.barycentric_same.iter_mut().zip(other.barycentric_same) {
            *mine += theirs;
        }
        for (k, n) in other.exports {
            *self.exports.entry(k).or_default() += n;
        }
        self.lines.extend(other.lines);
        for (reason, f) in other.failed {
            let entry = self.failed.entry(reason).or_insert_with(|| Failed {
                total: 0,
                stages: StageCounts::default(),
                example: f.example,
            });
            entry.total += f.total;
            entry.stages.merge(&f.stages);
        }
    }
}

/// Emitted shaders by label, shared between threads.
type Kept = std::sync::Arc<Mutex<Vec<(String, emit::HostShader)>>>;

/// The optional checks after emission, one set per thread.
/// Dumped shaders and validator reports (`VICTORY_EMIT_DUMP`) are
/// game-derived (rule #1), so their directory must be outside the repo, as
/// `Config` requires of the cache and data directories.
fn emit_dump_outside_repo(dir: Option<&std::ffi::OsStr>) -> Result<()> {
    if let Some(dir) = dir {
        victory_core::ensure_outside_repo(Path::new(dir))?;
    }
    Ok(())
}

struct Checks {
    /// Shaders named in `VICTORY_PAIRS` (`vs+ps,...`, label substrings),
    /// kept to build those pipelines after the walk.
    keep: Option<Kept>,
    validator: Option<Validator>,
    host: Option<(Host, emit::HostShader)>,
    /// Also emit mesh programs for hosts without mesh shaders.
    mesh_fallback: bool,
    /// Also emit pixel programs for hosts without barycentrics.
    barycentric_fallback: bool,
}

impl Checks {
    /// Whether the geometry shader of a pixel program's fallback is built.
    /// It is built from a stub vertex shader, which the host's compiler
    /// makes: on Windows always, elsewhere only for a validator or a host
    /// that was asked for (neither exists there), so a scan that only
    /// emits does not fail every such program for the platform it runs on.
    fn builds_geometry(&self) -> bool {
        cfg!(windows) || self.validator.is_some() || self.host.is_some()
    }
}

pub fn run(
    paths: Vec<PathBuf>,
    validate: bool,
    pipelines: bool,
    (mesh_fallback, barycentric_fallback): (bool, bool),
    list: bool,
    jobs: Option<usize>,
) -> Result<()> {
    emit_dump_outside_repo(std::env::var_os("VICTORY_EMIT_DUMP").as_deref())?;
    let paths = crate::default_paths(paths)?;
    let dll = validate.then(Validator::default_path).flatten();
    if validate {
        Validator::load(dll.as_deref()).map_err(anyhow::Error::msg)?;
        println!(
            "validator    {}",
            dll.as_ref()
                .map_or("dxil.dll".into(), |p| p.display().to_string())
        );
    }
    if pipelines {
        Host::new(false).map_err(anyhow::Error::msg)?;
        // As the running title does on a 32-lane host (`victory-gpu`'s
        // `supports_wave32`): `VICTORY_WAVE64_SPLIT=1` runs Wave64 programs as
        // two 32-lane halves instead of declaring `WaveSize(64)`.
        // shortcut: assumes the host has 32-lane waves.
        if std::env::var_os("VICTORY_WAVE64_SPLIT").is_some_and(|v| v != "0") {
            victory_shader::emit::set_host_wave64(false);
        }
    }
    let pairs: Vec<(String, String)> = std::env::var("VICTORY_PAIRS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|p| p.split_once('+'))
        .map(|(v, p)| (v.trim().to_owned(), p.trim().to_owned()))
        .collect();
    let kept = std::sync::Arc::new(Mutex::new(Vec::new()));
    let threads =
        jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let seen = Mutex::new(HashSet::new());
    let total = Mutex::new(Tally::default());
    let mut walk_errors = 0;
    for root in &paths {
        for entry in scan_entries(root, &mut walk_errors) {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                continue;
            };
            let jobs = sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads.min(jobs.len()) {
                    s.spawn(|| {
                        let mut t = Tally {
                            list,
                            ..Tally::default()
                        };
                        let mut checks = Checks {
                            keep: (!pairs.is_empty()).then(|| kept.clone()),
                            validator: validate
                                .then(|| Validator::load(dll.as_deref()).ok())
                                .flatten(),
                            host: pipelines
                                .then(|| Host::new(false).ok().map(|h| (h, host::stub_ps())))
                                .flatten(),
                            mesh_fallback,
                            barycentric_fallback,
                        };
                        while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                            translate(job, entry.path(), &seen, &mut t, &mut checks);
                        }
                        lock(&total).merge(t);
                    });
                }
            });
        }
    }
    let mut t = total.into_inner().unwrap_or_else(|e| e.into_inner());
    for line in t.summary_lines(validate) {
        println!("{line}");
    }
    if mesh_fallback {
        let [tried, emitted, validated, made] = t.fallback;
        println!(
            "             {emitted} emitted of {tried} mesh programs without mesh shaders; \
             {validated} validated, {made} pipelines"
        );
    }
    if barycentric_fallback {
        let [tried, emitted, validated, made] = t.barycentric;
        let [same, different] = t.barycentric_same;
        println!(
            "             {emitted} emitted of {tried} pixel programs that need barycentrics, \
             for hosts without them; {validated} validated, {made} pipelines"
        );
        println!(
            "             {same} pixel programs without barycentrics emit the same for such \
             hosts, {different} differ"
        );
    }
    if !pairs.is_empty() {
        let host = Host::new(false).map_err(anyhow::Error::msg)?;
        let kept = lock(&kept);
        let find = |name: &str| kept.iter().find(|(l, _)| l.contains(name));
        for (v, p) in &pairs {
            let made = match (find(v), find(p)) {
                (Some((_, vs)), Some((_, ps))) => if vs.stage == HwStage::Ms {
                    host.mesh_pipeline(vs, ps)
                } else {
                    host.pipeline(vs, ps)
                }
                .map(|_| "pipeline created".to_owned())
                .unwrap_or_else(|e| e),
                _ => "not emitted".to_owned(),
            };
            println!("pair {v} + {p}: {made}");
        }
    }
    t.lines.sort();
    for l in &t.lines {
        println!("{l}");
    }
    for line in t.failure_lines() {
        println!("{line}");
    }
    anyhow::ensure!(
        walk_errors == 0,
        "scan incomplete (traversal errors: {walk_errors})"
    );
    Ok(())
}

fn translate(
    source: &Source,
    path: &Path,
    seen: &Mutex<HashSet<blake3::Hash>>,
    t: &mut Tally,
    checks: &mut Checks,
) {
    let location = || format!("{}{}", path.display(), source.label());
    let Some(bytes) = source.bytes() else {
        return t.fail(None, "corrupt record".into(), location);
    };
    let list = t.list;
    let hash = blake3::hash(&bytes).to_hex().to_string();
    let short_hash = &hash[..16];
    let example = || {
        let location = location();
        if list {
            format!("{location} hash={short_hash}")
        } else {
            location
        }
    };
    t.containers += 1;
    let Ok(container) = Container::parse(&bytes) else {
        return t.fail(None, "no DXIL part (SM5 or older)".into(), example);
    };
    // A compute container may carry its program in an `ACCC` part instead.
    if ![FourCC::DXIL, victory_shader::accc::ACCC]
        .into_iter()
        .any(|f| container.part(f).is_some())
    {
        return t.fail(None, "no DXIL part (SM5 or older)".into(), example);
    }
    // A container is tallied once. By its own digest, not its program
    // part's: the lowering is chosen by the container's hash (a built-in
    // list, or `VICTORY_WAVE_WIDE_ONLY`), so two containers sharing a
    // program can lower differently.
    let key = blake3::hash(&bytes);
    if !lock(seen).insert(key) && !list {
        t.duplicates += 1;
        return;
    }
    let programs = match victory_shader::container_programs(&container) {
        Ok(p) => p,
        Err(e) => return t.fail(None, e.to_string(), example),
    };
    for p in programs {
        tally_program(&container, &p, short_hash, &example, t, checks);
    }
}

fn tally_program(
    container: &Container<'_>,
    p: &NativeProgram,
    hash: &str,
    example: &impl Fn() -> String,
    t: &mut Tally,
    checks: &mut Checks,
) {
    t.programs += 1;
    match ir::build_with(&p.patched_code(), p.wave64, p.vgprs, p.uses_spill_table()) {
        Ok(program) => {
            t.translated += 1;
            t.instructions += program.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
            // Per-lane descriptors are traced on the per-lane form; its
            // failure is listed after the resources' and I/O's.
            let structured = structure::structure(&program);
            let lanes = structured.as_ref().ok();
            let res = match resource::track(&program, &p.layout, p.stage, lanes) {
                Ok(r) => {
                    tally_resources(p, &r, t);
                    Some(r)
                }
                Err(ShaderError::Unsupported { offset, what }) => {
                    t.fail(Some(p.stage), format!("resources: {what}"), || {
                        format!("{} {:?} at {offset:#x}", example(), p.stage)
                    });
                    None
                }
                Err(e) => {
                    t.fail(Some(p.stage), e.to_string(), || {
                        format!("{} {:?}", example(), p.stage)
                    });
                    None
                }
            };
            let graphics = matches!(p.stage, HwStage::Vs | HwStage::Ps);
            let mut raised = None;
            if is_emitted_stage(p.stage) {
                t.io_programs += usize::from(graphics);
                match io::raise(&program, p) {
                    Ok(r) => {
                        if graphics {
                            tally_io(&r, t);
                        }
                        raised = Some(r);
                    }
                    Err(ShaderError::Unsupported { offset, what }) => {
                        t.fail(Some(p.stage), format!("I/O: {what}"), || {
                            format!("{} {:?} at {offset:#x}", example(), p.stage)
                        });
                    }
                    Err(e) => t.fail(Some(p.stage), e.to_string(), || {
                        format!("{} {:?}", example(), p.stage)
                    }),
                }
            }
            let structured = match structured {
                Ok(s) => {
                    tally_structured(&program, &s, t);
                    Some(s)
                }
                Err(e) => {
                    let reason = match e {
                        ShaderError::Unsupported { what, .. } => format!("structure: {what}"),
                        ShaderError::Isa { reason, .. } => format!("structure: {reason}"),
                        e => e.to_string(),
                    };
                    t.fail(Some(p.stage), reason, || {
                        format!("{} {:?}", example(), p.stage)
                    });
                    None
                }
            };
            if !is_emitted_stage(p.stage) {
                if res.is_some() && structured.is_some() {
                    t.not_emitted.record(p.stage);
                    if t.list {
                        t.lines
                            .push(format!("{} {:?}  stage not emitted", example(), p.stage));
                    }
                } else {
                    t.refused += 1;
                }
            } else if let (Some(r), Some(i), Some(s)) = (res, raised, structured) {
                let label = || format!("{} {:?}", example(), p.stage);
                *counter(t, p.stage, 0) += 1;
                match emit::emit_with_wave32_for_hash(
                    container, p, &program, &s, &i, &r, true, hash,
                ) {
                    Ok(shader) => {
                        *counter(t, p.stage, 1) += 1;
                        if p.stage == HwStage::Ps && checks.barycentric_fallback {
                            let fallback =
                                emit::emit_without_barycentrics(container, p, &program, &s, &i, &r);
                            check_barycentric(&shader, fallback, t, checks, &label);
                        }
                        check(shader, t, checks, label);
                    }
                    Err(ShaderError::Unsupported { offset, what }) => {
                        t.refused += 1;
                        t.fail(Some(p.stage), format!("emit: {what}"), || {
                            format!("{} at {offset:#x}", label())
                        });
                    }
                    Err(e) => {
                        t.refused += 1;
                        t.fail(Some(p.stage), format!("emit: {e}"), label);
                    }
                }
                if p.stage == HwStage::Ms && checks.mesh_fallback {
                    t.fallback[0] += 1;
                    match emit::emit_mesh_fallback(container, p, &program, &s, &i, &r) {
                        Ok(fallback) => {
                            t.fallback[1] += 1;
                            check_fallback(fallback, t, checks, label);
                        }
                        Err(ShaderError::Unsupported { offset, what }) => {
                            t.fail(Some(p.stage), format!("mesh fallback: {what}"), || {
                                format!("{} at {offset:#x}", label())
                            });
                        }
                        Err(e) => t.fail(Some(p.stage), format!("mesh fallback: {e}"), label),
                    }
                }
            } else {
                t.refused += 1;
            }
        }
        Err(e) => {
            t.refused += 1;
            let reason = match e {
                ShaderError::Unsupported { what, .. } => format!("unsupported: {what}"),
                e => e.to_string(),
            };
            t.fail(Some(p.stage), reason, || {
                format!("{} {:?}", example(), p.stage)
            });
        }
    }
}

/// A pixel program emitted for hosts without barycentrics (`fallback`)
/// beside its plain emission `native`: one that needs none must be the
/// same; one that does is validated with its geometry shader and gets a
/// pipeline behind a stub vertex shader, if asked to.
fn check_barycentric(
    native: &emit::HostShader,
    fallback: Result<emit::HostShader, ShaderError>,
    t: &mut Tally,
    checks: &mut Checks,
    label: &impl Fn() -> String,
) {
    let needs = native.needs_barycentrics();
    let fallback = match fallback {
        Ok(f) => f,
        Err(e) => {
            // A program that needs none and whose fallback did not emit is
            // a failure (reported below), not one that emits differently.
            if needs {
                t.barycentric[0] += 1;
            }
            return t.fail(
                Some(HwStage::Ps),
                format!("barycentric fallback: {e}"),
                label,
            );
        }
    };
    if !needs {
        if fallback == *native {
            t.barycentric_same[0] += 1;
        } else {
            t.barycentric_same[1] += 1;
            t.fail(
                Some(HwStage::Ps),
                "barycentric fallback: a program without barycentrics emits differently".into(),
                label,
            );
        }
        return;
    }
    t.barycentric[0] += 1;
    if fallback.needs_barycentrics() {
        return t.fail(
            Some(HwStage::Ps),
            "barycentric fallback: still needs the host's barycentrics".into(),
            label,
        );
    }
    t.barycentric[1] += 1;
    if !checks.builds_geometry() {
        return;
    }
    // The stub vertex shader feeds the plain program's inputs, as `check`'s.
    let made = host::stub_vs(native).and_then(|vs| {
        emit::barycentric_gs(&vs.dxil, &fallback.dxil)
            .map(|gs| (vs, gs))
            .map_err(|e| e.to_string())
    });
    let (vs, mut gs) = match made {
        Ok(made) => made,
        Err(e) => {
            return t.fail(
                Some(HwStage::Ps),
                format!("barycentric fallback geometry shader: {e}"),
                label,
            );
        }
    };
    if let Some(v) = checks.validator.as_mut() {
        let mut ps = fallback.dxil.clone();
        for (what, dxil) in [("pixel", &mut ps), ("geometry", &mut gs)] {
            if let Err(text) = v.validate(dxil) {
                let first = text
                    .lines()
                    .find(|l| l.contains("error"))
                    .unwrap_or(text.lines().next().unwrap_or(""))
                    .to_owned();
                let reason = format!("barycentric fallback validator ({what}): {first}");
                return t.fail(Some(HwStage::Ps), reason, label);
            }
        }
        t.barycentric[2] += 1;
    }
    if let Some((h, _)) = &checks.host {
        match h.pipeline_with_geometry(&vs, &gs, &fallback) {
            Ok(_) => {
                t.barycentric[3] += 1;
                if t.list {
                    t.lines.push(format!(
                        "{}  barycentric fallback pipeline created",
                        label()
                    ));
                }
            }
            Err(e) => t.fail(
                Some(HwStage::Ps),
                format!("barycentric fallback pipeline: {e}"),
                label,
            ),
        }
    }
}

/// Validates both halves of a mesh program emitted for hosts without mesh
/// shaders, and makes their pipelines, if asked to.
fn check_fallback(
    mut fallback: emit::MeshFallback,
    t: &mut Tally,
    checks: &mut Checks,
    label: impl Fn() -> String,
) {
    if let Some((h, stub_ps)) = &checks.host {
        match h.mesh_fallback_pipelines(&fallback, stub_ps) {
            Ok(_) => t.fallback[3] += 1,
            Err(e) => t.fail(
                Some(HwStage::Ms),
                format!("mesh fallback pipeline: {e}"),
                &label,
            ),
        }
    }
    let Some(v) = checks.validator.as_mut() else {
        return;
    };
    let halves = [
        ("compute", &mut fallback.compute.dxil),
        ("vertex", &mut fallback.vertex.dxil),
    ];
    for (half, dxil) in halves {
        if let Err(text) = v.validate(dxil) {
            let first = text
                .lines()
                .find(|l| l.contains("error"))
                .unwrap_or(text.lines().next().unwrap_or(""))
                .to_owned();
            let reason = format!("mesh fallback validator ({half}): {first}");
            return t.fail(Some(HwStage::Ms), reason, &label);
        }
    }
    t.fallback[2] += 1;
}

/// Validates an emitted shader if asked to.
fn check(
    mut shader: emit::HostShader,
    t: &mut Tally,
    checks: &mut Checks,
    label: impl Fn() -> String,
) {
    if let Some(keep) = &checks.keep {
        lock(keep).push((label(), shader.clone()));
    }
    // `VICTORY_EMIT_DUMP` with `--list`: every emitted shader as a file.
    if t.list
        && let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP")
    {
        let name: String = label()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let _ = std::fs::write(
            format!("{dir}/{}.dxil", &name[name.len().saturating_sub(100)..]),
            &shader.dxil,
        );
    }
    if let Some((h, stub_ps)) = &checks.host {
        let made = match shader.stage {
            HwStage::Vs => h.pipeline(&shader, stub_ps).map(drop),
            HwStage::Cs => h.compute_pipeline(&shader).map(drop),
            HwStage::Ms => h.mesh_pipeline(&shader, stub_ps).map(drop),
            _ => host::stub_vs(&shader).and_then(|vs| h.pipeline(&vs, &shader).map(drop)),
        };
        match made {
            Ok(()) => {
                *counter(t, shader.stage, 3) += 1;
                if t.list {
                    t.lines.push(format!("{}  pipeline created", label()));
                }
            }
            Err(e) => t.fail(
                Some(shader.stage),
                format!("pipeline: {e} ({})", shader.declared()),
                &label,
            ),
        }
    }
    let Some(v) = checks.validator.as_mut() else {
        return;
    };
    match v.validate(&mut shader.dxil) {
        Ok(()) => {
            *counter(t, shader.stage, 2) += 1;
            if t.list {
                t.lines.push(format!("{}  validated", label()));
            }
        }
        Err(text) => {
            let first = text
                .lines()
                .find(|l| l.contains("error"))
                .unwrap_or(text.lines().next().unwrap_or(""))
                .to_owned();
            if let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP") {
                let n = t
                    .failed
                    .get(&format!("validator: {first}"))
                    .map_or(0, |f| f.total);
                if n < 3 {
                    let name = format!(
                        "{dir}/{}_{n}",
                        first
                            .chars()
                            .filter(char::is_ascii_alphanumeric)
                            .take(40)
                            .collect::<String>()
                    );
                    let _ = std::fs::write(format!("{name}.dxil"), &shader.dxil);
                    let _ = std::fs::write(format!("{name}.txt"), format!("{}\n{text}", label()));
                }
            }
            t.validator_rejected += 1;
            t.fail(Some(shader.stage), format!("validator: {first}"), label);
        }
    }
}

/// Emission counts of `stage`'s kind: 0 programs, 1 emitted, 2 validated,
/// 3 pipelines.
fn counter(t: &mut Tally, stage: HwStage, k: usize) -> &mut usize {
    let [a, b, c, d] = match stage {
        HwStage::Cs => [
            &mut t.cs_programs,
            &mut t.cs_emitted,
            &mut t.cs_validated,
            &mut t.cs_pipelines,
        ],
        HwStage::Ms => [
            &mut t.ms_programs,
            &mut t.ms_emitted,
            &mut t.ms_validated,
            &mut t.ms_pipelines,
        ],
        _ => [
            &mut t.emit_programs,
            &mut t.emitted,
            &mut t.validated,
            &mut t.pipelines,
        ],
    };
    [a, b, c, d].into_iter().nth(k).expect("0 to 3")
}

fn tally_io(r: &io::Io, t: &mut Tally) {
    t.io_raised += 1;
    for (_, i) in &r.interps {
        match i {
            io::Interp::Interpolated { .. } => t.interpolated += 1,
            io::Interp::Vertex { .. } => t.vertex_values += 1,
            io::Interp::Computed { second, .. } => t.interpolated += usize::from(*second),
        }
    }
    for e in &r.exports {
        let kind = match e.target {
            io::Target::Color(_) => "color",
            io::Target::Depth => "depth",
            io::Target::Null => "null",
            io::Target::Position(_) => "position",
            io::Target::Param { .. } => "param",
            io::Target::Unread => "unread param",
            io::Target::Primitive => "primitive",
            io::Target::PrimitiveParam { .. } => "primitive param",
        };
        *t.exports.entry(kind).or_default() += 1;
        t.compressed += usize::from(e.compressed);
    }
}

fn tally_resources(p: &NativeProgram, r: &resource::Resources, t: &mut Tally) {
    t.tracked += 1;
    t.scalar += r.scalar.len();
    t.dynamic += r.scalar.iter().filter(|s| s.1.dynamic.is_some()).count();
    let address = |load| r.scalar.iter().find(|s| s.0 == load).map(|s| s.1);
    let mut traced = HashSet::new();
    for leaf in r.descriptors.iter().flat_map(|d| d.source.leaves()) {
        let a = match *leaf {
            resource::Source::Loaded { load, .. } | resource::Source::Fetched { load, .. } => {
                t.loaded += 1;
                address(load)
            }
            resource::Source::Built { address, .. } => {
                t.built += 1;
                t.dynamic += usize::from(address.dynamic.is_some());
                Some(address)
            }
            resource::Source::Constant(_) => {
                t.constant += 1;
                None
            }
            resource::Source::Phi { .. }
            | resource::Source::Select { .. }
            | resource::Source::Lanes { .. }
            | resource::Source::LanePhi { .. } => {
                unreachable!("leaves are no phis, selects or per-lane choices")
            }
        };
        // Heaps hold dynamic resources, which no used entry lists; code
        // tables are no root parameter.
        if let Some(a) = a
            && a.param != resource::CODE
            && a.param != resource::MEMORY
            && !matches!(
                p.layout[a.param].root,
                RootType::ResourceHeap | RootType::SamplerHeap
            )
        {
            traced.insert(p.layout[a.param].user_data);
        }
    }
    // Root constants are listed as a CBV, read straight from the SGPRs.
    traced.extend(
        r.params
            .iter()
            .map(|&(_, param, _)| &p.layout[param])
            .filter(|l| l.root == RootType::Constants)
            .map(|l| l.user_data),
    );
    let listed: HashSet<u8> = p.used.iter().map(|u| u.user_data).collect();
    t.used += p.used.len();
    // Immediate samplers are literals in the code: the distinct sampler
    // values (decoded) are matched against the listed entries, which do not
    // say which value is whose.
    let mut values: Vec<emit::StaticSampler> = Vec::new();
    for d in r
        .descriptors
        .iter()
        .filter(|d| d.role == resource::Role::Sampler)
    {
        for leaf in d.source.leaves() {
            if let resource::Source::Constant(words) = leaf
                && let Ok(w) = <[u32; 4]>::try_from(&words[..words.len().min(4)])
            {
                let s = emit::StaticSampler::from_words(w);
                if !values.contains(&s) {
                    values.push(s);
                }
            }
        }
    }
    // Engines often declare equal samplers at several registers, so equal
    // values may stand for several entries, but never more values than
    // entries.
    let immediate = p
        .used
        .iter()
        .filter(|u| u.root == RootType::ImmediateSampler)
        .count();
    let samplers_match = !values.is_empty() && values.len() <= immediate;
    let matched = |u: &&victory_shader::UsedResource| {
        if u.root == RootType::ImmediateSampler {
            samplers_match
        } else {
            traced.contains(&u.user_data)
        }
    };
    for u in p.used.iter().filter(|u| !matched(u)) {
        *t.untraced_used
            .entry(format!("{:?} via {:?}", u.kind, u.root))
            .or_default() += 1;
    }
    t.unlisted += traced.difference(&listed).count();
}

fn tally_structured(program: &ir::Program, s: &structure::Structured, t: &mut Tally) {
    t.structured += 1;
    for (block, guards) in program.blocks.iter().zip(&s.guards) {
        t.vector += block
            .insts
            .iter()
            .filter(|i| i.read(ir::Reg::S(ir::EXEC)).is_some())
            .count();
        for &g in guards.iter().flatten() {
            t.guarded += 1;
            let support = s.bools.support(g);
            t.opaque += usize::from(support.iter().any(|a| matches!(a, Atom::Opaque(_))));
        }
    }
    for exit in &s.exits {
        if let LaneExit::Branch { when, .. } = *exit {
            if s.bools
                .support(when)
                .iter()
                .all(|a| matches!(a, Atom::Uniform(_)))
            {
                t.uniform_branches += 1;
            } else {
                t.lane_branches += 1;
            }
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(not(windows))]
use offline::{Host, Validator, host};

/// Off Windows there is no D3D12 host or DXIL validator: asking for either
/// fails, and the translation itself still runs.
#[cfg(not(windows))]
mod offline {
    use victory_shader::emit::HostShader;

    const NEEDS_WINDOWS: &str = "needs Windows (D3D12 and dxil.dll)";

    pub enum Validator {}

    impl Validator {
        pub fn default_path() -> Option<std::path::PathBuf> {
            None
        }

        pub fn load(_: Option<&std::path::Path>) -> Result<Self, String> {
            Err(NEEDS_WINDOWS.into())
        }

        pub fn validate(&mut self, _: &mut [u8]) -> Result<(), String> {
            match *self {}
        }
    }

    pub enum Host {}

    impl Host {
        pub fn new(_: bool) -> Result<Self, String> {
            Err(NEEDS_WINDOWS.into())
        }

        pub fn pipeline(&self, _: &HostShader, _: &HostShader) -> Result<(), String> {
            match *self {}
        }

        pub fn mesh_pipeline(&self, _: &HostShader, _: &HostShader) -> Result<(), String> {
            match *self {}
        }

        pub fn compute_pipeline(&self, _: &HostShader) -> Result<(), String> {
            match *self {}
        }

        pub fn pipeline_with_geometry(
            &self,
            _: &HostShader,
            _: &[u8],
            _: &HostShader,
        ) -> Result<(), String> {
            match *self {}
        }

        pub fn mesh_fallback_pipelines(
            &self,
            _: &victory_shader::emit::MeshFallback,
            _: &HostShader,
        ) -> Result<(), String> {
            match *self {}
        }
    }

    pub mod host {
        use victory_shader::emit::HostShader;

        /// Only asked for alongside a [`super::Host`], which never exists here.
        pub fn stub_ps() -> HostShader {
            unreachable!("no host GPU off Windows")
        }

        pub fn stub_vs(_: &HostShader) -> Result<HostShader, String> {
            Err(super::NEEDS_WINDOWS.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shader dumps are game-derived, so a dump directory in the repo is
    /// refused; one outside it, or none, is not.
    #[test]
    fn emit_dump_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("no-such-dump-dir");
        let err = emit_dump_outside_repo(Some(inside.as_os_str())).unwrap_err();
        assert!(
            matches!(
                err.downcast_ref(),
                Some(victory_core::PolicyError::InsideRepo { .. })
            ),
            "{err}"
        );
        let outside = tempfile::tempdir().unwrap();
        emit_dump_outside_repo(Some(outside.path().as_os_str())).unwrap();
        emit_dump_outside_repo(None).unwrap();
    }
    use victory_shader::native::{ComputeIo, StageIo};

    fn empty_dxbc() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        bytes.extend_from_slice(b"DXBC");
        bytes.extend_from_slice(&[0u8; 16]);
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&32u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes
    }

    fn endpgm_program(stage: HwStage, io: StageIo) -> NativeProgram {
        // GFX10 `s_endpgm` (0xBF81_0000).
        NativeProgram {
            stage,
            wave64: false,
            vgprs: 4,
            layout: Vec::new(),
            used: Vec::new(),
            io,
            code: 0xBF81_0000u32.to_le_bytes().to_vec(),
        }
    }

    #[test]
    fn reason_lines_report_per_stage_counts_and_merge_across_workers() {
        let mut a = Tally::default();
        a.fail(Some(HwStage::Vs), "emit: unknown stride".into(), || {
            "a.bin Vs".into()
        });
        a.fail(Some(HwStage::Vs), "emit: unknown stride".into(), || {
            "a2.bin Vs".into()
        });
        a.fail(Some(HwStage::Ps), "emit: unknown stride".into(), || {
            "a3.bin Ps".into()
        });
        a.fail(Some(HwStage::Cs), "emit: unknown stride".into(), || {
            "a4.bin Cs".into()
        });
        a.fail(
            Some(HwStage::Ps),
            "I/O: computed barycentrics".into(),
            || "b.bin Ps".into(),
        );
        a.fail(None, "corrupt record".into(), || "bad.bin".into());

        let mut b = Tally::default();
        b.fail(Some(HwStage::Ps), "emit: unknown stride".into(), || {
            "c.bin Ps".into()
        });
        b.fail(Some(HwStage::Ms), "emit: unknown stride".into(), || {
            "d.bin Ms".into()
        });
        b.fail(Some(HwStage::Gs), "structure: irreducible".into(), || {
            "g.bin Gs".into()
        });

        a.merge(b);
        assert_eq!(
            a.failure_lines(),
            vec![
                "        6  emit: unknown stride  [vs 2, ps 2, cs 1, ms 1]  (e.g. a.bin Vs)",
                "        1  I/O: computed barycentrics  [vs 0, ps 1, cs 0, ms 0]  (e.g. b.bin Ps)",
                "        1  corrupt record  (e.g. bad.bin)",
                "        1  structure: irreducible  [vs 0, ps 0, cs 0, ms 0, gs 1]  (e.g. g.bin Gs)",
            ]
        );
    }

    #[test]
    fn list_mode_records_rows_and_preserves_stage_breakdown() {
        let mut t = Tally {
            list: true,
            ..Tally::default()
        };
        t.fail(Some(HwStage::Vs), "emit: test".into(), || {
            "x.bin hash=0000 Vs".into()
        });
        t.fail(Some(HwStage::Ms), "emit: test".into(), || {
            "y.bin hash=1111 Ms".into()
        });
        assert_eq!(
            t.lines,
            vec![
                "x.bin hash=0000 Vs  emit: test",
                "y.bin hash=1111 Ms  emit: test",
            ]
        );
        assert_eq!(
            t.failure_lines(),
            vec!["        2  emit: test  [vs 1, ps 0, cs 0, ms 1]  (e.g. x.bin hash=0000 Vs)"]
        );
    }

    #[test]
    fn summary_program_counts_add_up_without_subtraction() {
        let mut a = Tally {
            programs: 10,
            emit_programs: 5,
            emitted: 4,
            validated: 3,
            validator_rejected: 1,
            cs_programs: 2,
            cs_emitted: 1,
            cs_validated: 1,
            refused: 3,
            ..Tally::default()
        };
        a.not_emitted.record(HwStage::Ls);
        a.not_emitted.record(HwStage::Hs);

        let mut b = Tally {
            programs: 7,
            ms_programs: 3,
            ms_emitted: 2,
            ms_validated: 1,
            validator_rejected: 1,
            refused: 1,
            ..Tally::default()
        };
        b.not_emitted.record(HwStage::Hs);
        b.not_emitted.record(HwStage::Es);
        b.not_emitted.record(HwStage::Gs);
        b.not_emitted.record(HwStage::Gs);

        a.merge(b);
        assert_eq!(
            a.emitted_total() + a.refused + a.not_emitted.total(),
            a.programs
        );

        let without_validate = a.summary_lines(false);
        assert!(
            without_validate.contains(
                &"             6 programs in stages not emitted (LS 1, HS 2, ES 1, GS 2)"
                    .to_owned()
            ),
            "{without_validate:#?}"
        );
        // Because `validator_rejected > 0`, the breakdown stays visible even if `validate` is false.
        assert!(
            without_validate.contains(
                &"             7 emitted (5 validated, 2 rejected by validator), 4 refused, 6 in stages not emitted (17 total)".to_owned()
            ),
            "{without_validate:#?}"
        );

        let plain = Tally {
            programs: 3,
            cs_programs: 3,
            cs_emitted: 2,
            refused: 1,
            ..Tally::default()
        };
        let plain_lines = plain.summary_lines(false);
        assert!(
            plain_lines.contains(&"             0 programs in stages not emitted".to_owned()),
            "{plain_lines:#?}"
        );
        assert!(
            plain_lines.contains(
                &"             2 emitted, 1 refused, 0 in stages not emitted (3 total)".to_owned()
            ),
            "{plain_lines:#?}"
        );
    }

    #[test]
    fn tally_program_separates_emitted_refused_and_unemitted_stages() {
        let raw = empty_dxbc();
        let container = Container::parse(&raw).unwrap();
        let mut checks = Checks {
            keep: None,
            validator: None,
            host: None,
            mesh_fallback: false,
            barycentric_fallback: false,
        };
        let mut t = Tally {
            list: true,
            ..Tally::default()
        };
        let example = || "synth.bin".to_owned();

        // Compute shader with `s_endpgm` emits cleanly.
        let cs = endpgm_program(
            HwStage::Cs,
            StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [1, 1, 1],
            }),
        );
        tally_program(&container, &cs, "", &example, &mut t, &mut checks);

        // Tessellation and geometry stages are counted in `not_emitted`, not refused.
        for stage in [HwStage::Ls, HwStage::Hs, HwStage::Es, HwStage::Gs] {
            let prog = endpgm_program(stage, StageIo::None);
            tally_program(&container, &prog, "", &example, &mut t, &mut checks);
        }

        // An unemitted stage that fails IR building is counted once as refused, not `not_emitted`.
        let mut bad_gs = endpgm_program(HwStage::Gs, StageIo::None);
        bad_gs.code = 0xFFFF_FFFFu32.to_le_bytes().to_vec();
        tally_program(&container, &bad_gs, "", &example, &mut t, &mut checks);

        // A vertex shader that fails IR building records `vs 1` on its refusal line.
        let mut bad_vs = endpgm_program(
            HwStage::Vs,
            StageIo::Vs {
                inputs: Vec::new(),
                outputs: Vec::new(),
            },
        );
        bad_vs.code = 0xFFFF_FFFFu32.to_le_bytes().to_vec();
        tally_program(&container, &bad_vs, "", &example, &mut t, &mut checks);

        assert_eq!(t.programs, 7);
        assert_eq!(t.emitted_total(), 1);
        assert_eq!(t.cs_emitted, 1);
        assert_eq!(t.not_emitted.total(), 4);
        assert_eq!(t.not_emitted.ls, 1);
        assert_eq!(t.not_emitted.hs, 1);
        assert_eq!(t.not_emitted.es, 1);
        assert_eq!(t.not_emitted.gs, 1);
        assert_eq!(t.refused, 2);
        assert_eq!(
            t.emitted_total() + t.refused + t.not_emitted.total(),
            t.programs
        );

        let reasons = t.failure_lines();
        assert_eq!(reasons.len(), 1, "{reasons:#?}");
        assert!(
            reasons[0].contains("[vs 1, ps 0, cs 0, ms 0, gs 1]"),
            "{}",
            reasons[0]
        );
        assert!(
            t.lines
                .iter()
                .any(|l| l == "synth.bin Gs  stage not emitted"),
            "{:#?}",
            t.lines
        );
    }

    /// A DXBC container holding `parts`, each a tag and its data.
    fn dxbc(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let mut bytes = empty_dxbc();
        let mut offset = 32 + 4 * parts.len();
        for (_, data) in parts {
            bytes.extend_from_slice(&(offset as u32).to_le_bytes());
            offset += 8 + data.len();
        }
        for (tag, data) in parts {
            bytes.extend_from_slice(*tag);
            bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
            bytes.extend_from_slice(data);
        }
        bytes[0x18..0x1C].copy_from_slice(&(offset as u32).to_le_bytes());
        bytes[0x1C..0x20].copy_from_slice(&(parts.len() as u32).to_le_bytes());
        bytes
    }

    fn no_checks() -> Checks {
        Checks {
            keep: None,
            validator: None,
            host: None,
            mesh_fallback: false,
            barycentric_fallback: false,
        }
    }

    /// The lowering is chosen by the container's digest, so two containers
    /// that share a program part and differ elsewhere are both tallied;
    /// only a byte-identical container is a duplicate.
    #[test]
    fn containers_sharing_a_program_part_are_tallied_separately() {
        let program = [0u8; 16];
        let a = dxbc(&[(b"DXIL", &program), (b"STAT", &[1; 4])]);
        let b = dxbc(&[(b"DXIL", &program), (b"STAT", &[2; 4])]);
        let (ca, cb) = (Container::parse(&a).unwrap(), Container::parse(&b).unwrap());
        assert_eq!(
            ca.part(FourCC::DXIL).unwrap().data,
            cb.part(FourCC::DXIL).unwrap().data
        );
        assert_ne!(a, b);

        let seen = Mutex::new(HashSet::new());
        let mut t = Tally::default();
        let mut checks = no_checks();
        let path = Path::new("synth.bin");
        for (offset, data) in [(0, &a), (0x100, &b)] {
            let source = Source::Raw { offset, data };
            translate(&source, path, &seen, &mut t, &mut checks);
        }
        assert_eq!((t.containers, t.duplicates), (2, 0));
        assert_eq!(lock(&seen).len(), 2);

        // The same bytes again, found elsewhere in the file.
        let source = Source::Raw {
            offset: 0x200,
            data: &a,
        };
        translate(&source, path, &seen, &mut t, &mut checks);
        assert_eq!((t.containers, t.duplicates), (3, 1));
        assert_eq!(lock(&seen).len(), 2);
    }

    /// A pixel shader as the emitter hands it over, whose feature flags
    /// (`SFI0`) ask for the host's barycentrics or do not.
    fn pixel_shader(barycentrics: bool) -> emit::HostShader {
        // The barycentrics bit of the DXIL shader feature flags.
        let flags: u64 = if barycentrics { 0x2_0000 } else { 0 };
        emit::HostShader {
            stage: HwStage::Ps,
            dxil: dxbc(&[(b"SFI0", &flags.to_le_bytes())]),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            structured_srvs: Vec::new(),
        }
    }

    fn refused() -> Result<emit::HostShader, ShaderError> {
        Err(ShaderError::Unsupported {
            offset: 0,
            what: "a synthetic refusal",
        })
    }

    /// A program that needs no barycentrics whose fallback emission fails is
    /// a failure row, not a program that "emits differently".
    #[test]
    fn a_failed_fallback_without_barycentrics_is_a_failure_not_a_difference() {
        let native = pixel_shader(false);
        assert!(!native.needs_barycentrics());
        let mut t = Tally::default();
        let label = || "synth.bin Ps".to_owned();
        check_barycentric(&native, refused(), &mut t, &mut no_checks(), &label);

        assert_eq!(t.barycentric_same, [0, 0]);
        assert_eq!(t.barycentric, [0; 4]);
        assert_eq!(t.failed.len(), 1, "{:#?}", t.failure_lines());
        let (reason, failed) = t.failed.iter().next().unwrap();
        assert!(reason.starts_with("barycentric fallback: "), "{reason}");
        assert!(reason.contains("a synthetic refusal"), "{reason}");
        assert_eq!((failed.total, failed.stages.ps), (1, 1));
    }

    /// A program that needs barycentrics counts as tried even when its
    /// fallback emission fails.
    #[test]
    fn a_failed_fallback_with_barycentrics_still_counts_as_tried() {
        let native = pixel_shader(true);
        assert!(native.needs_barycentrics());
        let mut t = Tally::default();
        let label = || "synth.bin Ps".to_owned();
        check_barycentric(&native, refused(), &mut t, &mut no_checks(), &label);

        assert_eq!(t.barycentric, [1, 0, 0, 0]);
        assert_eq!(t.barycentric_same, [0, 0]);
        assert_eq!(t.failed.values().map(|f| f.total).sum::<usize>(), 1);
    }

    /// Off Windows no stub vertex shader can be made, and a scan that only
    /// emits does not need one: a fallback that emitted counts as emitted
    /// and is not a failure.
    #[cfg(not(windows))]
    #[test]
    fn an_emitted_fallback_is_no_failure_where_no_geometry_shader_is_built() {
        let native = pixel_shader(true);
        let mut t = Tally::default();
        let label = || "synth.bin Ps".to_owned();
        let fallback = Ok(pixel_shader(false));
        check_barycentric(&native, fallback, &mut t, &mut no_checks(), &label);

        assert_eq!(t.barycentric, [1, 1, 0, 0]);
        assert!(t.failed.is_empty(), "{:#?}", t.failure_lines());
    }
}
