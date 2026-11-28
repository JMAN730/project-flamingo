//! What a world run is launched with: the known-good environment, changed
//! by a profile file and then by flags.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

/// The environment world runs are known to work with on the 8 GB card.
pub const KNOWN_GOOD_ENV: &[(&str, &str)] = &[
    (
        "VICTORY_SKIP_PIPELINES",
        "5d113aa6830760b6,caa78b8ca62bc3d6,5552dac6a2b7a1fd",
    ),
    ("VICTORY_TEXTURE_MIP_DROP", "2"),
    ("VICTORY_TEXTURE_MIP_DROP_MIN_SIZE", "256"),
    ("VICTORY_WAVE64_SPLIT", "1"),
    ("VICTORY_TIME_SCALE", "auto"),
];

/// The console model world runs use.
pub const KNOWN_GOOD_CONSOLE: &str = "series-s";

/// `VICTORY_*` variables of the caller that a run keeps: they say where
/// files are, not how the title runs. Every other one is dropped, so two
/// runs with the same manifest ran with the same switches.
pub const KEPT_AMBIENT: &[&str] = &[
    "VICTORY_CONFIG",
    "VICTORY_GAME_DIR",
    "VICTORY_CACHE_DIR",
    "VICTORY_DATA_DIR",
    "VICTORY_TEST_TITLES",
];

/// Variables the tool sets itself from its capture and input flags.
const OWNED: &[&str] = &[
    "VICTORY_CAPTURE",
    "VICTORY_CAPTURE_FRAMES",
    "VICTORY_CAPTURE_SECONDS",
    "VICTORY_KEYS",
    "VICTORY_KEYS_FILE",
];

/// The most frames a range may list: each one is read back from the GPU.
const MOST_FRAMES: u64 = 4096;

/// Which `victory.exe` runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Build {
    /// Release code generation plus the debugging aids. The skip list,
    /// capture and scripted keys only exist in this build.
    Diag,
    /// The plain release build.
    Release,
}

impl Build {
    pub fn name(self) -> &'static str {
        match self {
            Build::Diag => "diag",
            Build::Release => "release",
        }
    }
}

/// A profile file: any part of a run's settings, in TOML.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub build: Option<Build>,
    pub console: Option<String>,
    /// Variables to set; an empty value removes a default.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub capture: CaptureProfile,
    pub keys: Option<String>,
    pub max_seconds: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureProfile {
    pub frames: Option<String>,
    pub seconds: Option<String>,
}

impl Profile {
    pub fn read(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}

/// What flags ask for, over the profile.
#[derive(Debug, Default)]
pub struct Overrides {
    pub build: Option<Build>,
    pub console: Option<String>,
    /// `NAME=VALUE`; an empty value removes the variable.
    pub env: Vec<String>,
    pub capture_frames: Option<String>,
    pub capture_seconds: Option<String>,
    pub keys: Option<String>,
    pub max_seconds: Option<u64>,
}

/// Presented frames `first..=last`, every `step`th.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameRange {
    pub first: u64,
    pub last: u64,
    pub step: u64,
}

impl FrameRange {
    /// `<first>-<last>` or `<first>-<last>:<step>`.
    pub fn parse(text: &str) -> Result<Self> {
        let (range, step) = match text.split_once(':') {
            Some((range, step)) => (range, step.trim().parse().context("the step")?),
            None => (text, 1),
        };
        let (first, last) = range
            .split_once('-')
            .with_context(|| format!("a frame range is <first>-<last>[:<step>], got {text}"))?;
        let first: u64 = first.trim().parse().context("the first frame")?;
        let last: u64 = last.trim().parse().context("the last frame")?;
        ensure!(step > 0, "the step of a frame range is at least 1");
        ensure!(first <= last, "frame range {text} ends before it starts");
        let range = Self { first, last, step };
        ensure!(
            range.count() <= MOST_FRAMES,
            "frame range {text} lists {} frames; at most {MOST_FRAMES} (use a step)",
            range.count()
        );
        Ok(range)
    }

    pub fn count(self) -> u64 {
        (self.last - self.first) / self.step + 1
    }

    /// The list `VICTORY_CAPTURE_FRAMES` takes.
    pub fn list(self) -> String {
        (self.first..=self.last)
            .step_by(self.step as usize)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Seconds `from..=to` after the first presented frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecondsSpan {
    pub from: u64,
    pub to: u64,
}

impl SecondsSpan {
    /// `<from>-<to>`.
    pub fn parse(text: &str) -> Result<Self> {
        let (from, to) = text
            .split_once('-')
            .with_context(|| format!("a span of seconds is <from>-<to>, got {text}"))?;
        let from: u64 = from.trim().parse().context("the first second")?;
        let to: u64 = to.trim().parse().context("the last second")?;
        ensure!(from <= to, "span {text} ends before it starts");
        Ok(Self { from, to })
    }
}

/// The settings of one run, as launched and as written to its manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    pub build: Build,
    pub console: String,
    /// Every `VICTORY_*` variable the tool sets, except the capture folder.
    pub env: BTreeMap<String, String>,
    pub capture_frames: Option<FrameRange>,
    pub capture_seconds: Option<SecondsSpan>,
    /// `VICTORY_KEYS`: `<second>:<key>,...`.
    pub keys: Option<String>,
    /// The run is ended after this many seconds.
    pub max_seconds: Option<u64>,
}

impl Settings {
    /// The known-good defaults, then the profile, then the flags.
    pub fn resolve(profile: Profile, flags: Overrides) -> Result<Self> {
        let mut env: BTreeMap<String, String> = KNOWN_GOOD_ENV
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let from_flags = flags
            .env
            .iter()
            .map(|pair| {
                pair.split_once('=')
                    .map(|(k, v)| (k.trim().to_string(), v.to_string()))
                    .with_context(|| format!("--env takes NAME=VALUE, got {pair}"))
            })
            .collect::<Result<Vec<_>>>()?;
        for (name, value) in profile.env.into_iter().chain(from_flags) {
            ensure!(
                name.starts_with("VICTORY_"),
                "{name} is not a VICTORY_ variable"
            );
            if OWNED.contains(&name.as_str()) {
                bail!("{name} is set by the capture and keys settings, not as a variable");
            }
            if KEPT_AMBIENT.contains(&name.as_str()) {
                bail!("{name} says where files are: set it in the shell or in victory.toml");
            }
            if value.is_empty() {
                env.remove(&name);
            } else {
                env.insert(name, value);
            }
        }
        let frames = flags.capture_frames.or(profile.capture.frames);
        let seconds = flags.capture_seconds.or(profile.capture.seconds);
        let keys = flags.keys.or(profile.keys).filter(|k| k != "none");
        let settings = Self {
            build: flags.build.or(profile.build).unwrap_or(Build::Diag),
            console: flags
                .console
                .or(profile.console)
                .unwrap_or_else(|| KNOWN_GOOD_CONSOLE.into()),
            env,
            capture_frames: frames.as_deref().map(FrameRange::parse).transpose()?,
            capture_seconds: seconds.as_deref().map(SecondsSpan::parse).transpose()?,
            keys,
            max_seconds: flags.max_seconds.or(profile.max_seconds),
        };
        ensure!(
            ["series-s", "series-x"].contains(&settings.console.as_str()),
            "the console is series-s or series-x, got {}",
            settings.console
        );
        Ok(settings)
    }

    /// What a release build would silently ignore of these settings.
    pub fn needs_diagnostics(&self) -> Vec<&'static str> {
        let mut needs = Vec::new();
        if self.env.contains_key("VICTORY_SKIP_PIPELINES") {
            needs.push("the skip list");
        }
        if self.capture_frames.is_some() || self.capture_seconds.is_some() {
            needs.push("capture");
        }
        if self.keys.is_some() {
            needs.push("scripted keys");
        }
        needs
    }

    /// The variables the title process gets beyond the kept ambient ones.
    /// `capture_dir` is where frames go, when the build can capture.
    pub fn child_env(&self, capture_dir: Option<&Path>) -> BTreeMap<String, String> {
        let mut env = self.env.clone();
        if let Some(dir) = capture_dir {
            env.insert("VICTORY_CAPTURE".into(), dir.display().to_string());
            if let Some(frames) = self.capture_frames {
                env.insert("VICTORY_CAPTURE_FRAMES".into(), frames.list());
            }
            if let Some(span) = self.capture_seconds {
                env.insert(
                    "VICTORY_CAPTURE_SECONDS".into(),
                    format!("{}-{}", span.from, span.to),
                );
            }
        }
        if let Some(keys) = &self.keys {
            env.insert("VICTORY_KEYS".into(), keys.clone());
        }
        env
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved(profile: &str, flags: Overrides) -> Result<Settings> {
        Settings::resolve(toml::from_str(profile).unwrap(), flags)
    }

    #[test]
    fn with_nothing_asked_a_run_gets_the_known_good_environment() {
        let s = resolved("", Overrides::default()).unwrap();
        assert_eq!(s.build, Build::Diag);
        assert_eq!(s.console, "series-s");
        assert_eq!(s.env.len(), KNOWN_GOOD_ENV.len());
        assert_eq!(s.env["VICTORY_TIME_SCALE"], "auto");
        assert_eq!(s.env["VICTORY_WAVE64_SPLIT"], "1");
        assert_eq!(s.keys, None);
    }

    #[test]
    fn flags_win_over_the_profile_which_wins_over_the_defaults() {
        let profile = r#"
            build = "release"
            console = "series-x"
            keys = "40:Enter"
            [env]
            VICTORY_TEXTURE_MIP_DROP = "1"
            VICTORY_LOG = "info"
            [capture]
            frames = "10-20"
        "#;
        let flags = Overrides {
            console: Some("series-s".into()),
            env: vec!["VICTORY_TEXTURE_MIP_DROP=3".into()],
            capture_frames: Some("100-110:5".into()),
            ..Overrides::default()
        };
        let s = resolved(profile, flags).unwrap();
        assert_eq!(s.build, Build::Release);
        assert_eq!(s.console, "series-s");
        assert_eq!(s.env["VICTORY_TEXTURE_MIP_DROP"], "3");
        assert_eq!(s.env["VICTORY_LOG"], "info");
        assert_eq!(s.env["VICTORY_TEXTURE_MIP_DROP_MIN_SIZE"], "256");
        assert_eq!(s.capture_frames.unwrap().list(), "100,105,110");
        assert_eq!(s.keys.as_deref(), Some("40:Enter"));
    }

    #[test]
    fn an_empty_value_removes_a_default_and_none_removes_the_keys() {
        let flags = Overrides {
            env: vec!["VICTORY_SKIP_PIPELINES=".into()],
            keys: Some("none".into()),
            ..Overrides::default()
        };
        let s = resolved("keys = \"40:Enter\"", flags).unwrap();
        assert!(!s.env.contains_key("VICTORY_SKIP_PIPELINES"));
        assert_eq!(s.keys, None);
        assert!(s.needs_diagnostics().is_empty());
    }

    #[test]
    fn variables_the_tool_owns_or_that_place_files_are_refused() {
        for name in ["VICTORY_CAPTURE", "VICTORY_DATA_DIR", "PATH"] {
            let flags = Overrides {
                env: vec![format!("{name}=x")],
                ..Overrides::default()
            };
            assert!(resolved("", flags).is_err(), "{name}");
        }
        assert!(toml::from_str::<Profile>("typo = 1").is_err());
    }

    #[test]
    fn frame_ranges_and_spans_are_checked() {
        let r = FrameRange::parse("1500-1560:20").unwrap();
        assert_eq!((r.count(), r.list().as_str()), (4, "1500,1520,1540,1560"));
        assert_eq!(FrameRange::parse("7-7").unwrap().list(), "7");
        for bad in ["20-10", "10", "1-9:0", "0-100000", "a-b"] {
            assert!(FrameRange::parse(bad).is_err(), "{bad}");
        }
        assert!(FrameRange::parse("0-100000:100").is_ok());
        assert_eq!(
            SecondsSpan::parse("300-360").unwrap(),
            SecondsSpan { from: 300, to: 360 }
        );
        assert!(SecondsSpan::parse("360-300").is_err());
    }

    #[test]
    fn capture_and_keys_reach_the_title_only_as_the_tool_sets_them() {
        let flags = Overrides {
            capture_frames: Some("8-10".into()),
            capture_seconds: Some("300-360".into()),
            keys: Some("40:Enter,60:Enter".into()),
            ..Overrides::default()
        };
        let s = resolved("", flags).unwrap();
        assert_eq!(
            s.needs_diagnostics(),
            ["the skip list", "capture", "scripted keys"]
        );
        let env = s.child_env(Some(Path::new("frames")));
        assert_eq!(env["VICTORY_CAPTURE"], "frames");
        assert_eq!(env["VICTORY_CAPTURE_FRAMES"], "8,9,10");
        assert_eq!(env["VICTORY_CAPTURE_SECONDS"], "300-360");
        assert_eq!(env["VICTORY_KEYS"], "40:Enter,60:Enter");
        let without = s.child_env(None);
        assert!(!without.contains_key("VICTORY_CAPTURE_FRAMES"));
    }
}
