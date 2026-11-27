//! Differential checks of the two-half lowering of wave64 compute programs
//! (`lower::split`): a program is translated in several ways that must
//! agree, each run on a host device with the same synthetic inputs, and
//! the contents of every view it writes compared.
//!
//! The variants:
//! - `split`: the lowering as the running title gets it on a 32-lane host
//!   (the lists of programs left out of the split are ignored);
//! - `split-again`: the same pipeline run again (outputs that change from
//!   run to run are races, or atomics whose order the program leaves open);
//! - `split-pairs`: a group of one console wave lowered with the exchange
//!   mechanism of a group of several (counters per half instead of group
//!   barriers). The single-wave lowering is the reference: both are valid
//!   for such a group, so a difference is a fault of the several-wave
//!   mechanism, which every multi-wave program depends on;
//! - `native`: the program declaring `WaveSize(64)`, on a reference device
//!   whose waves hold 64 lanes (none on a host with only 32-lane waves; WARP
//!   before shader model 6.6 cannot run translated shaders at all).
//!
//! Inputs follow the program's resource layout: what the translation reads
//! from the heaps is recorded while it is made ([`super::probe`]), each
//! table gets a region of the heap, each entry a view of the kind the
//! shader declares, filled with deterministic pseudo-random contents in
//! several profiles (`Rng`: small integers, small floats, mostly zero...;
//! a profile the program writes nothing on is not reported); root constants
//! are small integers. Programs whose descriptors cannot be bound up front
//! (indices computed by the code, descriptors read from memory) are skipped
//! rather than run with views of the wrong kind, which NVIDIA reports as a
//! device fault. A difference where every output holds the same dwords in
//! another order is told apart (`reordered`: appends through an atomic
//! counter). Loops stop early and a half waits for its partner only so
//! long ([`Options`]), so that synthetic bounds or a faulty lowering do not
//! run into the host's GPU timeout.
//!
//! Nothing of a program is kept or printed but its hash and the counts of
//! what differs.
// shortcut: with no device of 64-lane waves (a 32-lane GPU and WARP before
// shader model 6.6), multi-wave programs are only checked against
// themselves run again; nothing tells a deterministic fault of theirs.

use std::collections::{BTreeMap, BTreeSet};

use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

use super::host::Host;
use super::probe::{self, HeapUse, Uses};
use super::{FourCC, HostShader};
use crate::dxbc::Container;
use crate::native::{HwStage, NativeProgram, RootType, StageIo};

/// How the programs are run.
#[derive(Debug, Clone)]
pub struct Options {
    /// Thread groups per dispatch.
    pub groups: [u32; 3],
    /// Seed of the synthetic contents.
    pub seed: u64,
    /// Polls after which a half stops waiting for its partner: a lowering
    /// fault then shows as data rather than as a GPU hang.
    pub spin_limit: u32,
    /// Translate and plan the bindings, run nothing.
    pub dry_run: bool,
    /// Iterations after which a loop stops (lower than the title's, so
    /// that loops on synthetic bounds cannot run into the GPU timeout).
    pub loop_limit: u32,
    /// Writes each variant's DXIL here as `<hash>.<variant>.dxil`.
    pub dump: Option<std::path::PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            groups: [1, 1, 1],
            seed: 0x5EED,
            spin_limit: 1 << 14,
            dry_run: false,
            loop_limit: 1 << 10,
            dump: None,
        }
    }
}

/// What a check found for one program.
#[derive(Debug, Clone)]
pub enum Outcome {
    /// Not a program the split lowering takes (why).
    NotSplit(String),
    /// Split, but not run (why).
    Skipped(String),
    /// Split and bindable ([`Options::dry_run`]): threads, console waves.
    Planned(u32, u32),
    Ran(Report),
}

/// The variants of one program that ran.
#[derive(Debug, Clone)]
pub struct Report {
    pub threads: u32,
    /// Console waves per group.
    pub waves: u32,
    /// Views the program writes.
    pub outputs: usize,
    /// Their dwords.
    pub words: usize,
    /// Dwords `split` changed (against a dispatch of no groups), and in how
    /// many outputs: how much the comparison sees.
    pub written: (usize, usize),
    pub variants: Vec<Variant>,
}

#[derive(Debug, Clone)]
pub struct Variant {
    /// `p<profile>:<variant>`.
    pub name: String,
    /// Hash of all outputs (16 hex digits), or why the variant did not run.
    pub result: Result<String, String>,
    /// Against `split`: dwords that differ, and in how many outputs.
    pub diff: Option<(usize, usize)>,
    /// When it differs: whether each output holds the same dwords in
    /// another order (as when lanes append through an atomic counter in an
    /// order the program leaves open).
    pub reordered: bool,
    pub outputs: Vec<Vec<u32>>,
}

impl Report {
    /// Variants whose outputs differ from `split`'s.
    pub fn differing(&self) -> Vec<&Variant> {
        self.variants
            .iter()
            .filter(|v| v.diff.is_some_and(|(n, _)| n > 0))
            .collect()
    }
}

/// The devices checks run on.
pub struct Harness {
    host: Host,
    /// A device with 64-lane waves and shader model 6.6, if any.
    reference: Option<(Host, String)>,
}

impl Harness {
    /// The default hardware device (it must run 32-lane waves) and a
    /// reference device with 64-lane waves when one is present.
    pub fn new() -> Result<Self, String> {
        let host = Host::new(false)?;
        let (min, max) = lanes(&host.device);
        if !(min..=max).contains(&32) {
            return Err(format!("the device runs {min}..{max}-lane waves, not 32"));
        }
        Ok(Harness {
            host,
            reference: reference_device(),
        })
    }

    /// The reference device's name, if there is one.
    pub fn reference(&self) -> Option<&str> {
        self.reference.as_ref().map(|r| r.1.as_str())
    }

    /// Checks the wave64 compute program of `container`.
    pub fn check(&self, container: &[u8], opts: &Options) -> Outcome {
        match self.check_inner(container, opts) {
            Ok(o) => o,
            Err(e) => Outcome::Skipped(e),
        }
    }

    fn check_inner(&self, container: &[u8], opts: &Options) -> Result<Outcome, String> {
        let c = Container::parse(container).map_err(|e| e.to_string())?;
        let native = crate::container_programs(&c)
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|p| p.stage == HwStage::Cs)
            .ok_or("no compute program")?;
        let StageIo::Cs(cs) = native.io else {
            return Ok(Outcome::NotSplit("no compute I/O".into()));
        };
        let threads: u32 = cs.threads.iter().product();
        if !native.wave64 {
            return Ok(Outcome::NotSplit("wave32".into()));
        }
        if threads <= 32 || !threads.is_multiple_of(64) {
            return Ok(Outcome::NotSplit(format!("{threads} threads")));
        }
        let waves = threads / 64;
        let split = translate(container, false, None, opts);
        let (split, uses) = match split {
            Ok(s) => s,
            Err(e) => return Ok(Outcome::NotSplit(format!("not translated: {e}"))),
        };
        if wave_size(&split) != Some(32) {
            return Ok(Outcome::NotSplit(
                "no cross-lane work (no wave size)".into(),
            ));
        }
        let plans = (0..PROFILES)
            .map(|p| Plan::new(&native, &split, &uses, opts.seed, p))
            .collect::<Result<Vec<_>, _>>()?;
        let pairs = (waves == 1).then(|| match translate(container, false, Some(2), opts) {
            Ok((s, u)) if u == uses => Ok(s),
            Ok(_) => Err("the variant reads other descriptors".to_owned()),
            Err(e) => Err(format!("not translated: {e}")),
        });
        let native_cs =
            self.reference
                .as_ref()
                .map(|_| match translate(container, true, None, opts) {
                    Ok((s, u)) if u == uses => Ok(s),
                    Ok(_) => Err("the variant reads other descriptors".to_owned()),
                    Err(e) => Err(format!("not translated: {e}")),
                });
        if std::env::var_os("SPLIT_DIFF_PLAN").is_some() {
            eprintln!(
                "plan: views {:?} samplers {:?} root cbv/srv/uav {:?}/{:?}/{:?}",
                plans[0].views,
                plans[0].samplers,
                split.root_cbvs,
                split.root_srvs,
                split.root_uavs
            );
        }
        if opts.dry_run && std::env::var_os("SPLIT_DIFF_PSO").is_some() {
            self.host.compute_pipeline(&split)?;
            eprintln!("split pipeline made");
            if let Some(Ok(p)) = &pairs {
                self.host.compute_pipeline(p)?;
                eprintln!("pairs pipeline made");
            }
        }
        if opts.dry_run {
            return Ok(Outcome::Planned(threads, waves));
        }
        let none = Options {
            groups: [0, 0, 0],
            ..opts.clone()
        };
        let mut variants = Vec::new();
        let (mut outputs, mut words, mut written) = (0, 0, (0, 0));
        for (n, plan) in plans.iter().enumerate() {
            let name = |v: &str| format!("p{n}:{v}");
            let initial = self.run(&self.host, &split, plan, &none)?;
            let first = self.run(&self.host, &split, plan, opts);
            let base = first.as_ref().ok().cloned();
            let w = base.as_ref().map_or((0, 0), |o| diff(&initial, o));
            if let Some(o) = &base {
                outputs = o.len();
                words = o.iter().map(Vec::len).sum();
            }
            written.0 = written.0.max(w.0);
            written.1 = written.1.max(w.1);
            let quiet = n > 0 && w.0 == 0 && first.is_ok();
            let mut profile = Vec::new();
            profile.push(variant(name("split"), first, None));
            let again = self.run(&self.host, &split, plan, opts);
            profile.push(variant(name("split-again"), again, base.as_ref()));
            if let Some(pairs) = &pairs {
                let r = pairs
                    .clone()
                    .and_then(|s| self.run(&self.host, &s, plan, opts));
                let pairs_base = r.as_ref().ok().cloned();
                profile.push(variant(name("split-pairs"), r, base.as_ref()));
                if let Ok(s) = pairs {
                    let again = self.run(&self.host, s, plan, opts);
                    profile.push(variant(
                        name("split-pairs-again"),
                        again,
                        pairs_base.as_ref(),
                    ));
                }
            }
            if let (Some((reference, _)), Some(cs)) = (&self.reference, &native_cs) {
                let r = cs.clone().and_then(|s| self.run(reference, &s, plan, opts));
                profile.push(variant(name("native"), r, base.as_ref()));
            }
            let removed = profile
                .iter()
                .any(|v| matches!(&v.result, Err(e) if e.contains("device removed")));
            // A profile on which nothing writes tells nothing (the first is
            // always reported). Drop it only once every comparison ran and left
            // the outputs as the split did: a reference that writes where the
            // split does not is exactly the fault to report.
            let silent = quiet
                && profile
                    .iter()
                    .all(|v| v.result.is_ok() && v.diff.is_none_or(|d| d.0 == 0));
            if !silent {
                variants.append(&mut profile);
            }
            if removed {
                break;
            }
        }
        Ok(Outcome::Ran(Report {
            threads,
            waves,
            outputs,
            words,
            written,
            variants,
        }))
    }

    fn run(
        &self,
        host: &Host,
        cs: &HostShader,
        plan: &Plan,
        opts: &Options,
    ) -> Result<Vec<Vec<u32>>, String> {
        let (root, pso) = host.compute_pipeline(cs)?;
        // SAFETY: the plan's descriptions match the resources made for
        // them; everything stays alive until the fence passes.
        unsafe { dispatch(&host.device, &root, &pso, cs, plan, opts.groups) }
    }
}

fn variant(
    name: String,
    r: Result<Vec<Vec<u32>>, String>,
    base: Option<&Vec<Vec<u32>>>,
) -> Variant {
    match r {
        Ok(outputs) => {
            let diff = base.map(|b| diff(b, &outputs));
            let reordered = base.is_some_and(|b| {
                diff.is_some_and(|d| d.0 > 0)
                    && b.iter().zip(&outputs).all(|(x, y)| {
                        let (mut x, mut y) = (x.clone(), y.clone());
                        x.sort_unstable();
                        y.sort_unstable();
                        x == y
                    })
            });
            Variant {
                name,
                result: Ok(hash(&outputs)),
                diff,
                reordered,
                outputs,
            }
        }
        Err(e) => Variant {
            name,
            result: Err(e),
            diff: None,
            reordered: false,
            outputs: Vec::new(),
        },
    }
}

/// Dwords that differ between `a` and `b`, and in how many outputs.
pub fn diff(a: &[Vec<u32>], b: &[Vec<u32>]) -> (usize, usize) {
    let mut words = 0;
    let mut outputs = 0;
    for (x, y) in a.iter().zip(b) {
        let n = x.iter().zip(y).filter(|(p, q)| p != q).count() + x.len().abs_diff(y.len());
        words += n;
        outputs += usize::from(n > 0);
    }
    (words, outputs)
}

fn hash(outputs: &[Vec<u32>]) -> String {
    let mut h = blake3::Hasher::new();
    for o in outputs {
        h.update(&(o.len() as u64).to_le_bytes());
        for w in o {
            h.update(&w.to_le_bytes());
        }
    }
    h.finalize().to_hex()[..16].to_owned()
}

/// Translates `container` for a host with (`wave64`) or without 64-lane
/// waves, recording the heap entries it reads.
fn translate(
    container: &[u8],
    wave64: bool,
    waves: Option<u32>,
    opts: &Options,
) -> Result<(HostShader, Uses), String> {
    let (r, uses) = probe::record(|| {
        let strict = std::env::var_os("SPLIT_DIFF_STRICT_LDS").is_some();
        let clear = std::env::var_os("SPLIT_DIFF_CLEAR_LDS").is_some();
        probe::with_clear_lds(clear, || {
            probe::with_strict_lds(strict, || {
                probe::with_loop_limit(Some(opts.loop_limit), || {
                    probe::with_knobs(waves, Some(opts.spin_limit), true, || {
                        super::with_host_wave64(wave64, || {
                            super::translate_program(container, true, "", None)
                        })
                    })
                })
            })
        })
    });
    let (shader, _) = r.map_err(|e| e.to_string())?;
    if let Some(dir) = &opts.dump {
        let hash = blake3::hash(container).to_hex();
        let name = match (wave64, waves) {
            (true, _) => "native",
            (false, Some(_)) => "pairs",
            (false, None) => "split",
        };
        let _ = std::fs::write(
            dir.join(format!("{}.{name}.dxil", &hash[..16])),
            &shader.dxil,
        );
        let _ = std::fs::write(dir.join(format!("{}.dxbc", &hash[..16])), container);
    }
    Ok((shader, uses))
}

/// The wave size a translated shader declares.
fn wave_size(s: &HostShader) -> Option<u32> {
    let c = Container::parse(&s.dxil).ok()?;
    let psv = c.part(FourCC::PSV0)?.data;
    let min = u32::from_le_bytes(psv.get(20..24)?.try_into().ok()?);
    let max = u32::from_le_bytes(psv.get(24..28)?.try_into().ok()?);
    (min == max && min != 0).then_some(min)
}

fn lanes(device: &ID3D12Device) -> (u32, u32) {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the options struct matches the feature asked for.
    let ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&raw mut o).cast(),
            size_of_val(&o) as u32,
        )
    }
    .is_ok();
    if ok && o.WaveOps.as_bool() {
        (o.WaveLaneCountMin, o.WaveLaneCountMax)
    } else {
        (0, 0)
    }
}

/// An adapter (WARP included) with shader model 6.6 whose waves can hold
/// 64 lanes.
fn reference_device() -> Option<(Host, String)> {
    // SAFETY: plain DXGI enumeration and D3D12 device creation.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let mut i = 0;
        while let Ok(adapter) = factory.EnumAdapters1(i) {
            i += 1;
            let desc = adapter.GetDesc1().ok()?;
            let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(0);
            let name = String::from_utf16_lossy(&desc.Description[..len]);
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                continue;
            }
            let Some(device) = device else { continue };
            let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            let sm_ok = device
                .CheckFeatureSupport(
                    D3D12_FEATURE_SHADER_MODEL,
                    (&raw mut sm).cast(),
                    size_of_val(&sm) as u32,
                )
                .is_ok()
                && sm.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            let (min, max) = lanes(&device);
            if sm_ok && (min..=max).contains(&64) {
                return Some((Host { device }, name));
            }
        }
        None
    }
}

/// The kind of view a heap entry gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ViewKind {
    /// A texture: DXIL resource kind, UAV, element class.
    Texture(u8, bool, Class),
    /// A typed buffer of 4-channel elements of this class.
    Typed(bool, Class),
    Raw(bool),
    /// Structured, of this stride.
    Structured(bool, u32),
    Cbv,
    Sampler(bool),
}

/// What an element holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Class {
    /// Channels of 32-bit floats (count 1, 2 or 4).
    F32(u8),
    F16,
    U32(u8),
    I32(u8),
}

impl Class {
    fn of(props1: u32) -> Result<Class, String> {
        let count = match (props1 >> 8 & 0xFF) as u8 {
            1 => 1,
            2 => 2,
            _ => 4,
        };
        Ok(match props1 & 0xFF {
            9 | 13 | 14 => Class::F32(count),
            8 | 11 | 12 => Class::F16,
            5 => Class::U32(count),
            4 => Class::I32(count),
            t => return Err(format!("element type {t}")),
        })
    }

    fn format(self) -> DXGI_FORMAT {
        match self {
            Class::F32(1) => DXGI_FORMAT_R32_FLOAT,
            Class::F32(2) => DXGI_FORMAT_R32G32_FLOAT,
            Class::F32(_) => DXGI_FORMAT_R32G32B32A32_FLOAT,
            Class::F16 => DXGI_FORMAT_R16G16B16A16_FLOAT,
            Class::U32(1) => DXGI_FORMAT_R32_UINT,
            Class::U32(2) => DXGI_FORMAT_R32G32_UINT,
            Class::U32(_) => DXGI_FORMAT_R32G32B32A32_UINT,
            Class::I32(1) => DXGI_FORMAT_R32_SINT,
            Class::I32(2) => DXGI_FORMAT_R32G32_SINT,
            Class::I32(_) => DXGI_FORMAT_R32G32B32A32_SINT,
        }
    }

    fn bytes(self) -> u32 {
        match self {
            Class::F32(n) | Class::U32(n) | Class::I32(n) => 4 * u32::from(n),
            Class::F16 => 8,
        }
    }

    /// A pseudo-random dword of this class's elements.
    fn word(self, rng: &mut Rng) -> u32 {
        match self {
            Class::F32(_) => rng.float(),
            Class::F16 => {
                // Two halves in [-2, 2).
                let h = |r: &mut Rng| {
                    let x = (r.next() % 4096) as f32 / 1024.0 - 2.0;
                    f32_to_f16(x)
                };
                u32::from(h(rng)) | u32::from(h(rng)) << 16
            }
            Class::U32(_) | Class::I32(_) => rng.small(),
        }
    }
}

fn f32_to_f16(x: f32) -> u16 {
    // Exact for the multiples of 1/1024 in [-2, 2) used here.
    if x == 0.0 {
        return 0;
    }
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        return sign as u16;
    }
    (sign | (exp as u32) << 10 | mant) as u16
}

impl ViewKind {
    fn of(u: &HeapUse) -> Result<ViewKind, String> {
        let (p0, p1) = u.props;
        let kind = (p0 & 0xFF) as u8;
        let uav = p0 & 0x1000 != 0;
        Ok(match kind {
            1..=9 => ViewKind::Texture(kind, uav, Class::of(p1)?),
            10 => ViewKind::Typed(uav, Class::of(p1)?),
            11 => ViewKind::Raw(uav),
            12 => ViewKind::Structured(uav, p1.max(4)),
            13 => ViewKind::Cbv,
            14 => ViewKind::Sampler(p0 & 0x8000 != 0),
            k => return Err(format!("resource kind {k}")),
        })
    }
}

/// Dwords of each buffer view.
const BUFFER_DWORDS: u32 = 16 * 1024;
/// Dwords of each constant buffer.
const CBV_DWORDS: u32 = 4 * 1024;
/// First heap entry of the tables' regions (entries below are the dynamic
/// heap's, at the offsets the code gives).
const TABLES_AT: u32 = 4096;

/// The bindings of one program.
struct Plan {
    /// Root constants of user data (the shader's `b0`).
    user_data: Vec<u32>,
    /// View heap entries.
    views: BTreeMap<u32, ViewKind>,
    /// Sampler heap entries (comparison or not).
    samplers: BTreeMap<u32, bool>,
    seed: u64,
    /// How contents are drawn (see [`Rng`]).
    profile: u8,
}

impl Plan {
    fn new(
        native: &NativeProgram,
        cs: &HostShader,
        uses: &Uses,
        seed: u64,
        profile: u8,
    ) -> Result<Self, String> {
        if uses.untracked > 0 {
            return Err(format!(
                "{} descriptors chosen per lane or read from memory",
                uses.untracked
            ));
        }
        if cs.user_data.iter().any(|u| u.indexed_buffer) {
            return Err("an indexed heap buffer in user data".into());
        }
        // Each table's region: its first entry, by layout index.
        let mut region: BTreeMap<usize, u32> = BTreeMap::new();
        let mut next_view = TABLES_AT;
        let mut next_sampler = 256u32;
        let mut views = BTreeMap::new();
        let mut samplers = BTreeMap::new();
        let mut sampler_tables = BTreeSet::new();
        // Regions in layout order, sized by the largest entry read.
        let mut extent: BTreeMap<usize, u32> = BTreeMap::new();
        for u in &uses.heap {
            if u.dynamic {
                return Err("a descriptor at an index the code computes".into());
            }
            let rp = native
                .layout
                .get(u.param)
                .ok_or("a descriptor of no parameter")?;
            if rp.root == RootType::Table {
                let unit = if u.sampler { 16 } else { 32 };
                let e = extent.entry(u.param).or_default();
                *e = (*e).max(u.offset / unit + 1);
                if u.sampler {
                    sampler_tables.insert(u.param);
                }
            }
        }
        for (&param, &n) in &extent {
            if sampler_tables.contains(&param) {
                region.insert(param, next_sampler);
                next_sampler += n.next_multiple_of(16);
            } else {
                region.insert(param, next_view);
                next_view += n.next_multiple_of(64);
            }
        }
        for u in &uses.heap {
            let rp = native.layout[u.param];
            let unit = if u.sampler { 16 } else { 32 };
            if u.offset % unit != 0 {
                return Err("a descriptor read at an unaligned offset".into());
            }
            let index = match rp.root {
                RootType::Table => region[&u.param] + u.offset / unit,
                RootType::ResourceHeap | RootType::SamplerHeap => {
                    let i = u.offset / unit;
                    if !u.sampler && i >= TABLES_AT || u.sampler && i >= 256 {
                        return Err("a dynamic heap entry past the harness's".into());
                    }
                    i
                }
                r => return Err(format!("a heap descriptor from a {r:?} parameter")),
            };
            let kind = ViewKind::of(u)?;
            match kind {
                ViewKind::Sampler(cmp) => {
                    if samplers.insert(index, cmp).is_some_and(|c| c != cmp) {
                        return Err("a sampler read with and without comparison".into());
                    }
                }
                _ => {
                    if let Some(old) = views.insert(index, kind)
                        && !compatible(old, kind)
                    {
                        return Err(format!("one entry read as {old:?} and {kind:?}"));
                    }
                }
            }
        }
        if samplers.keys().any(|&i| i >= super::SAMPLER_HEAP_LIMIT) {
            return Err("too many samplers".into());
        }
        let mut rng = Rng::new(seed ^ 0xC0157A27, profile);
        let mut user_data = vec![0u32; cs.user_data_dwords().max(1) as usize];
        for u in &cs.user_data {
            let rp = native
                .layout
                .iter()
                .find(|p| p.slot == u.slot)
                .ok_or("user data of no parameter")?;
            let value = match u.root {
                RootType::Table => {
                    let param = native
                        .layout
                        .iter()
                        .position(|p| p.slot == u.slot && p.root == RootType::Table)
                        .ok_or("a table of no parameter")?;
                    match region.get(&param) {
                        Some(&r) if sampler_tables.contains(&param) => r * 16,
                        Some(&r) => r * 32,
                        // A table the shader reads nothing from.
                        None => 0,
                    }
                }
                RootType::Constants => {
                    let _ = rp;
                    rng.small()
                }
                _ => 0,
            };
            user_data[u.index as usize] = value;
        }
        Ok(Plan {
            user_data,
            views,
            samplers,
            seed,
            profile,
        })
    }
}

/// Whether one view can serve reads declared as `a` and as `b`.
fn compatible(a: ViewKind, b: ViewKind) -> bool {
    match (a, b) {
        (ViewKind::Texture(k, u, x), ViewKind::Texture(l, v, y)) => {
            k == l && u == v && x.format() == y.format()
        }
        _ => a == b,
    }
}

/// Input profiles: 0 half small integers and half small floats, 1 small
/// integers 0-15, 2 floats in [0, 1), 3 mostly zero, 4 ones, 5 records of
/// a small size (1-4) under a pseudo-random top byte.
const PROFILES: u8 = 6;

/// splitmix64, drawing values as the profile says.
struct Rng(u64, u8);

impl Rng {
    fn new(seed: u64, profile: u8) -> Self {
        Rng(seed ^ u64::from(profile) << 56, profile)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A small integer (root constants, integer elements).
    fn small(&mut self) -> u32 {
        let r = self.next();
        match self.1 {
            1 => (r % 16) as u32,
            3 if !r.is_multiple_of(8) => 0,
            3 => (r >> 8) as u32 % 4 + 1,
            4 => 1,
            5 => ((r % 4) as u32 + 1) | (((r >> 8) as u32 & 0xFF) << 24),
            _ => (r % 64) as u32,
        }
    }

    /// A small float.
    fn float(&mut self) -> u32 {
        let r = self.next();
        match self.1 {
            2 => ((r % 1024) as f32 / 1024.0).to_bits(),
            4 => 1f32.to_bits(),
            3 if !r.is_multiple_of(8) => 0,
            _ => ((r % 1024) as f32 / 256.0 - 2.0).to_bits(),
        }
    }

    /// A dword of a buffer read in no known way.
    fn mixed(&mut self) -> u32 {
        let profile = self.1;
        match profile {
            1 | 3 | 4 | 5 => self.small(),
            2 => self.float(),
            _ if self.next() & 1 == 0 => self.small(),
            _ => self.float(),
        }
    }
}

fn err(e: windows::core::Error) -> String {
    e.message().to_string()
}

fn transition(
    r: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn buffer_desc(bytes: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// A texture of DXIL resource kind `kind`: description and view
/// dimension.
fn texture_desc(kind: u8, uav: bool, class: Class) -> Result<D3D12_RESOURCE_DESC, String> {
    let (dim, w, h, d) = match kind {
        1 => (D3D12_RESOURCE_DIMENSION_TEXTURE1D, 64, 1, 1),
        6 => (D3D12_RESOURCE_DIMENSION_TEXTURE1D, 64, 1, 4),
        2 => (D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1),
        7 => (D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 4),
        5 => (D3D12_RESOURCE_DIMENSION_TEXTURE2D, 8, 8, 6),
        9 => (D3D12_RESOURCE_DIMENSION_TEXTURE2D, 8, 8, 12),
        4 => (D3D12_RESOURCE_DIMENSION_TEXTURE3D, 8, 8, 8),
        k => return Err(format!("texture kind {k}")),
    };
    Ok(D3D12_RESOURCE_DESC {
        Dimension: dim,
        Width: w,
        Height: h,
        DepthOrArraySize: d,
        MipLevels: 1,
        Format: class.format(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: if uav {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        },
        ..Default::default()
    })
}

/// An output to read back: the resource, its copyable layout (none for a
/// buffer of this many bytes) and the readback buffer.
struct Output {
    resource: ID3D12Resource,
    layout: Option<Vec<(D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u32, u64)>>,
    bytes: u64,
    readback: ID3D12Resource,
}

/// Runs `pso` once with `plan`'s bindings and returns the contents of
/// every view it may write, in heap order, then the root UAVs.
///
/// # Safety
/// `root` and `pso` must be made from `cs` on `dev`.
unsafe fn dispatch(
    dev: &ID3D12Device,
    root: &ID3D12RootSignature,
    pso: &ID3D12PipelineState,
    cs: &HostShader,
    plan: &Plan,
    groups: [u32; 3],
) -> Result<Vec<Vec<u32>>, String> {
    // SAFETY: (all below) D3D12 calls with descriptions that outlive them;
    // resources stay alive until the fence says the GPU is done.
    unsafe {
        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .map_err(err)?;
        let alloc: ID3D12CommandAllocator = dev
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .map_err(err)?;
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso)
            .map_err(err)?;
        let make = |heap: D3D12_HEAP_TYPE, desc: &D3D12_RESOURCE_DESC, state| {
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: heap,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                desc,
                state,
                None,
                &mut r,
            )
            .map_err(err)?;
            r.ok_or_else(|| "no resource".to_string())
        };
        // An upload buffer holding `words`.
        let upload = |words: &[u32]| -> Result<ID3D12Resource, String> {
            let size = (words.len() * 4).next_multiple_of(256).max(256);
            let b = make(
                D3D12_HEAP_TYPE_UPLOAD,
                &buffer_desc(size as u64, D3D12_RESOURCE_FLAG_NONE),
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut p = std::ptr::null_mut();
            b.Map(0, None, Some(&mut p)).map_err(err)?;
            std::ptr::copy_nonoverlapping(words.as_ptr(), p.cast::<u32>(), words.len());
            b.Unmap(0, None);
            Ok(b)
        };
        // Root views in root-signature order (see `Host::root_signature_for`).
        let mut root_views: Vec<(u8, u8)> = cs
            .root_cbvs
            .iter()
            .map(|&r| (r, 0))
            .chain(cs.root_srvs.iter().map(|&r| (r, 1)))
            .chain(cs.root_uavs.iter().map(|&r| (r, 2)))
            .collect();
        root_views.sort_unstable();
        root_views.dedup();
        let root_uavs = root_views.iter().filter(|v| v.1 == 2).count() as u32;
        let heap_views = plan.views.keys().next_back().map_or(0, |&i| i + 1);
        let n_views = heap_views + root_uavs;
        let heap = |ty, n: u32| -> Result<ID3D12DescriptorHeap, String> {
            dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: ty,
                NumDescriptors: n.max(1),
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .map_err(err)
        };
        let views = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, n_views)?;
        let n_samplers = plan.samplers.keys().next_back().map_or(0, |&i| i + 1);
        let samplers = heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, n_samplers)?;
        let view_size =
            dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let sampler_size =
            dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
        let view_at = |i: u32| {
            let mut at = views.GetCPUDescriptorHandleForHeapStart();
            at.ptr += i as usize * view_size;
            at
        };
        let mut keep: Vec<ID3D12Resource> = Vec::new();
        let mut outputs: Vec<Output> = Vec::new();
        let mut to_uav = Vec::new();
        let mut to_srv = Vec::new();
        // A default-heap buffer holding `words`, for a view; readback if
        // written.
        let buffer_with = |words: &[u32], uav: bool, list: &ID3D12GraphicsCommandList| {
            let bytes = words.len() as u64 * 4;
            let flags = if uav {
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_FLAG_NONE
            };
            let b = make(
                D3D12_HEAP_TYPE_DEFAULT,
                &buffer_desc(bytes, flags),
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?;
            let up = upload(words)?;
            list.CopyBufferRegion(&b, 0, &up, 0, bytes);
            Ok::<_, String>((b, up))
        };
        for (&index, &kind) in &plan.views {
            // Contents depend on the seed and the entry only.
            let mut rng = Rng::new(
                plan.seed ^ u64::from(index).wrapping_mul(0x9E37_79B9),
                plan.profile,
            );
            let at = view_at(index);
            match kind {
                ViewKind::Raw(uav) | ViewKind::Structured(uav, _) | ViewKind::Typed(uav, _) => {
                    let (format, elements, stride, class) = match kind {
                        ViewKind::Raw(_) => (DXGI_FORMAT_R32_TYPELESS, BUFFER_DWORDS, 0, None),
                        ViewKind::Structured(_, s) => {
                            (DXGI_FORMAT_UNKNOWN, BUFFER_DWORDS * 4 / s, s, None)
                        }
                        ViewKind::Typed(_, c) => {
                            (c.format(), BUFFER_DWORDS * 4 / c.bytes(), 0, Some(c))
                        }
                        _ => unreachable!(),
                    };
                    let words: Vec<u32> = (0..BUFFER_DWORDS)
                        .map(|_| match class {
                            Some(c) => c.word(&mut rng),
                            None => rng.mixed(),
                        })
                        .collect();
                    let (b, up) = buffer_with(&words, uav, &list)?;
                    keep.push(up);
                    if uav {
                        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                            Format: format,
                            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_UAV {
                                    NumElements: elements,
                                    StructureByteStride: stride,
                                    Flags: if stride == 0 && class.is_none() {
                                        D3D12_BUFFER_UAV_FLAG_RAW
                                    } else {
                                        D3D12_BUFFER_UAV_FLAG_NONE
                                    },
                                    ..Default::default()
                                },
                            },
                        };
                        dev.CreateUnorderedAccessView(&b, None, Some(&desc), at);
                        let bytes = u64::from(BUFFER_DWORDS) * 4;
                        let readback = make(
                            D3D12_HEAP_TYPE_READBACK,
                            &buffer_desc(bytes, D3D12_RESOURCE_FLAG_NONE),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        to_uav.push(b.clone());
                        outputs.push(Output {
                            resource: b,
                            layout: None,
                            bytes,
                            readback,
                        });
                    } else {
                        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: format,
                            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_SRV {
                                    FirstElement: 0,
                                    NumElements: elements,
                                    StructureByteStride: stride,
                                    Flags: if stride == 0 && class.is_none() {
                                        D3D12_BUFFER_SRV_FLAG_RAW
                                    } else {
                                        D3D12_BUFFER_SRV_FLAG_NONE
                                    },
                                },
                            },
                        };
                        dev.CreateShaderResourceView(&b, Some(&desc), at);
                        to_srv.push(b.clone());
                        keep.push(b);
                    }
                }
                ViewKind::Cbv => {
                    let words: Vec<u32> = (0..CBV_DWORDS).map(|_| rng.mixed()).collect();
                    let b = upload(&words)?;
                    dev.CreateConstantBufferView(
                        Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC {
                            BufferLocation: b.GetGPUVirtualAddress(),
                            SizeInBytes: CBV_DWORDS * 4,
                        }),
                        at,
                    );
                    keep.push(b);
                }
                ViewKind::Texture(k, uav, class) => {
                    let desc = texture_desc(k, uav, class)?;
                    let t = make(
                        D3D12_HEAP_TYPE_DEFAULT,
                        &desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )?;
                    let subs = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                        1
                    } else {
                        u32::from(desc.DepthOrArraySize)
                    };
                    let mut layouts =
                        vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); subs as usize];
                    let mut rows = vec![0u32; subs as usize];
                    let mut row_bytes = vec![0u64; subs as usize];
                    let mut total = 0u64;
                    dev.GetCopyableFootprints(
                        &desc,
                        0,
                        subs,
                        0,
                        Some(layouts.as_mut_ptr()),
                        Some(rows.as_mut_ptr()),
                        Some(row_bytes.as_mut_ptr()),
                        Some(&mut total),
                    );
                    let up = make(
                        D3D12_HEAP_TYPE_UPLOAD,
                        &buffer_desc(total, D3D12_RESOURCE_FLAG_NONE),
                        D3D12_RESOURCE_STATE_GENERIC_READ,
                    )?;
                    let mut p = std::ptr::null_mut();
                    up.Map(0, None, Some(&mut p)).map_err(err)?;
                    for (s, l) in layouts.iter().enumerate() {
                        let depth = l.Footprint.Depth as usize;
                        for z in 0..depth {
                            for y in 0..rows[s] as usize {
                                let row = p.cast::<u8>().add(
                                    l.Offset as usize
                                        + (z * rows[s] as usize + y)
                                            * l.Footprint.RowPitch as usize,
                                );
                                for w in 0..row_bytes[s] as usize / 4 {
                                    let v = class.word(&mut rng);
                                    row.add(4 * w).cast::<u32>().write_unaligned(v);
                                }
                            }
                        }
                    }
                    up.Unmap(0, None);
                    for (s, l) in layouts.iter().enumerate() {
                        let dst = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: std::mem::transmute_copy(&t),
                            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                                SubresourceIndex: s as u32,
                            },
                        };
                        let src = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: std::mem::transmute_copy(&up),
                            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                                PlacedFootprint: *l,
                            },
                        };
                        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                    }
                    keep.push(up);
                    texture_view(dev, &t, k, uav, class, &desc, at);
                    if uav {
                        let readback = make(
                            D3D12_HEAP_TYPE_READBACK,
                            &buffer_desc(total, D3D12_RESOURCE_FLAG_NONE),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        to_uav.push(t.clone());
                        let layout = layouts
                            .iter()
                            .zip(&rows)
                            .zip(&row_bytes)
                            .map(|((l, &r), &b)| (*l, r, b))
                            .collect();
                        outputs.push(Output {
                            resource: t,
                            layout: Some(layout),
                            bytes: total,
                            readback,
                        });
                    } else {
                        to_srv.push(t.clone());
                        keep.push(t);
                    }
                }
                ViewKind::Sampler(_) => unreachable!(),
            }
        }
        for (&index, &cmp) in &plan.samplers {
            let mut at = samplers.GetCPUDescriptorHandleForHeapStart();
            at.ptr += index as usize * sampler_size;
            dev.CreateSampler(
                &D3D12_SAMPLER_DESC {
                    Filter: if cmp {
                        D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT
                    } else {
                        D3D12_FILTER_MIN_MAG_MIP_POINT
                    },
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                    MaxAnisotropy: 1,
                    ComparisonFunc: if cmp {
                        D3D12_COMPARISON_FUNC_LESS_EQUAL
                    } else {
                        D3D12_COMPARISON_FUNC_NEVER
                    },
                    MaxLOD: f32::MAX,
                    ..Default::default()
                },
                at,
            );
        }
        // Root views: contents by slot.
        let mut root_args = Vec::new();
        let mut next_uav = heap_views;
        for (n, &(slot, kind)) in root_views.iter().enumerate() {
            let param = 1 + n as u32;
            let mut rng = Rng::new(
                plan.seed ^ 0xB00F ^ (u64::from(slot) << 8 | u64::from(kind)),
                plan.profile,
            );
            let dwords = if kind == 0 { CBV_DWORDS } else { BUFFER_DWORDS };
            let words: Vec<u32> = (0..dwords).map(|_| rng.mixed()).collect();
            match kind {
                0 | 1 => {
                    let b = upload(&words)?;
                    root_args.push((param, kind, b.GetGPUVirtualAddress(), 0u32));
                    keep.push(b);
                }
                _ => {
                    let (b, up) = buffer_with(&words, true, &list)?;
                    keep.push(up);
                    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_TYPELESS,
                        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_UAV {
                                NumElements: dwords,
                                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                                ..Default::default()
                            },
                        },
                    };
                    dev.CreateUnorderedAccessView(&b, None, Some(&desc), view_at(next_uav));
                    root_args.push((param, 2, 0, next_uav));
                    next_uav += 1;
                    let bytes = u64::from(dwords) * 4;
                    let readback = make(
                        D3D12_HEAP_TYPE_READBACK,
                        &buffer_desc(bytes, D3D12_RESOURCE_FLAG_NONE),
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )?;
                    to_uav.push(b.clone());
                    outputs.push(Output {
                        resource: b,
                        layout: None,
                        bytes,
                        readback,
                    });
                }
            }
        }
        let barriers: Vec<_> = to_uav
            .iter()
            .map(|r| {
                transition(
                    r,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )
            })
            .chain(to_srv.iter().map(|r| {
                transition(
                    r,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )
            }))
            .collect();
        if !barriers.is_empty() {
            list.ResourceBarrier(&barriers);
        }
        list.SetDescriptorHeaps(&[Some(views.clone()), Some(samplers.clone())]);
        list.SetComputeRootSignature(root);
        list.SetComputeRoot32BitConstants(
            0,
            plan.user_data.len() as u32,
            plan.user_data.as_ptr().cast(),
            0,
        );
        for &(param, kind, va, index) in &root_args {
            match kind {
                0 => list.SetComputeRootConstantBufferView(param, va),
                1 => list.SetComputeRootShaderResourceView(param, va),
                _ => {
                    let mut at = views.GetGPUDescriptorHandleForHeapStart();
                    at.ptr += u64::from(index) * view_size as u64;
                    list.SetComputeRootDescriptorTable(param, at);
                }
            }
        }
        let [x, y, z] = groups;
        list.Dispatch(x, y, z);
        let barriers: Vec<_> = outputs
            .iter()
            .map(|o| {
                transition(
                    &o.resource,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )
            })
            .collect();
        if !barriers.is_empty() {
            list.ResourceBarrier(&barriers);
        }
        for o in &outputs {
            match &o.layout {
                None => list.CopyBufferRegion(&o.readback, 0, &o.resource, 0, o.bytes),
                Some(layout) => {
                    for (s, (l, _, _)) in layout.iter().enumerate() {
                        let src = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: std::mem::transmute_copy(&o.resource),
                            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                                SubresourceIndex: s as u32,
                            },
                        };
                        let dst = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: std::mem::transmute_copy(&o.readback),
                            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                                PlacedFootprint: *l,
                            },
                        };
                        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                    }
                }
            }
        }
        list.Close().map_err(err)?;
        queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
        queue.Signal(&fence, 1).map_err(err)?;
        while fence.GetCompletedValue() < 1 {
            if let Err(e) = dev.GetDeviceRemovedReason() {
                return Err(format!("device removed: {}", err(e)));
            }
            std::thread::yield_now();
        }
        if let Err(e) = dev.GetDeviceRemovedReason() {
            return Err(format!("device removed: {}", err(e)));
        }
        let mut out = Vec::new();
        for o in &outputs {
            let mut p = std::ptr::null_mut();
            o.readback.Map(0, None, Some(&mut p)).map_err(err)?;
            let base = p.cast::<u8>();
            let words = match &o.layout {
                None => {
                    std::slice::from_raw_parts(base.cast::<u32>(), o.bytes as usize / 4).to_vec()
                }
                Some(layout) => {
                    let mut w = Vec::new();
                    for (l, rows, row_bytes) in layout {
                        for z in 0..l.Footprint.Depth as usize {
                            for y in 0..*rows as usize {
                                let row = base.add(
                                    l.Offset as usize
                                        + (z * *rows as usize + y) * l.Footprint.RowPitch as usize,
                                );
                                for k in 0..*row_bytes as usize / 4 {
                                    w.push(row.add(4 * k).cast::<u32>().read_unaligned());
                                }
                            }
                        }
                    }
                    w
                }
            };
            o.readback.Unmap(0, None);
            out.push(words);
        }
        drop(keep);
        Ok(out)
    }
}

/// The SRV or UAV of texture `t` at `at`.
///
/// # Safety
/// `t` must be made from `desc`.
unsafe fn texture_view(
    dev: &ID3D12Device,
    t: &ID3D12Resource,
    kind: u8,
    uav: bool,
    class: Class,
    desc: &D3D12_RESOURCE_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let n = u32::from(desc.DepthOrArraySize);
    let format = class.format();
    // SAFETY: descriptions of `t`'s own dimensions.
    unsafe {
        if uav {
            let (dim, anon) = match kind {
                1 => (
                    D3D12_UAV_DIMENSION_TEXTURE1D,
                    D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Texture1D: D3D12_TEX1D_UAV::default(),
                    },
                ),
                6 => (
                    D3D12_UAV_DIMENSION_TEXTURE1DARRAY,
                    D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Texture1DArray: D3D12_TEX1D_ARRAY_UAV {
                            ArraySize: n,
                            ..Default::default()
                        },
                    },
                ),
                2 => (
                    D3D12_UAV_DIMENSION_TEXTURE2D,
                    D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_UAV::default(),
                    },
                ),
                4 => (
                    D3D12_UAV_DIMENSION_TEXTURE3D,
                    D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Texture3D: D3D12_TEX3D_UAV {
                            WSize: n,
                            ..Default::default()
                        },
                    },
                ),
                _ => (
                    D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                    D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                            ArraySize: n,
                            ..Default::default()
                        },
                    },
                ),
            };
            let d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: format,
                ViewDimension: dim,
                Anonymous: anon,
            };
            dev.CreateUnorderedAccessView(t, None, Some(&d), at);
            return;
        }
        let (dim, anon) = match kind {
            1 => (
                D3D12_SRV_DIMENSION_TEXTURE1D,
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture1D: D3D12_TEX1D_SRV {
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
            ),
            6 => (
                D3D12_SRV_DIMENSION_TEXTURE1DARRAY,
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture1DArray: D3D12_TEX1D_ARRAY_SRV {
                        MipLevels: 1,
                        ArraySize: n,
                        ..Default::default()
                    },
                },
            ),
            2 => (
                D3D12_SRV_DIMENSION_TEXTURE2D,
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
            ),
            7 => (
                D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MipLevels: 1,
                        ArraySize: n,
                        ..Default::default()
                    },
                },
            ),
            4 => (
                D3D12_SRV_DIMENSION_TEXTURE3D,
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture3D: D3D12_TEX3D_SRV {
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
            ),
            5 => (
                D3D12_SRV_DIMENSION_TEXTURECUBE,
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    TextureCube: D3D12_TEXCUBE_SRV {
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
            ),
            _ => (
                D3D12_SRV_DIMENSION_TEXTURECUBEARRAY,
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    TextureCubeArray: D3D12_TEXCUBE_ARRAY_SRV {
                        MipLevels: 1,
                        NumCubes: n / 6,
                        ..Default::default()
                    },
                },
            ),
        };
        let d = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: dim,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: anon,
        };
        dev.CreateShaderResourceView(t, Some(&d), at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_convert_exactly() {
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(0.0), 0);
    }

    #[test]
    fn diffs_count_words_and_outputs() {
        let a = vec![vec![1, 2, 3], vec![4]];
        let b = vec![vec![1, 0, 3], vec![5]];
        assert_eq!(diff(&a, &b), (2, 2));
        assert_eq!(diff(&a, &a), (0, 0));
    }
}
