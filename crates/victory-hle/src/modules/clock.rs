//! The title's clock: `QueryPerformanceCounter` and `GetTickCount(64)`,
//! passed through to the host.
//!
//! `VICTORY_TIME_SCALE=auto` (or `auto:<fps>`, 10 by default) slows the
//! title's clock while frames come slower than that rate: each frame then
//! sees about `1/<fps>` seconds pass, so a title running far below its
//! frame rate keeps what it does by the clock (streaming in and creating
//! objects) in pace with what it does per frame (freeing them). At or above
//! the rate the clock runs at real speed; it never runs slower than a tenth
//! of real time, and it never goes back.
//!
//! Diagnostics builds also take a fixed `VICTORY_TIME_SCALE=<s>`
//! (0 < s < 1): from the first call on, the clock runs at `s` times real
//! time.
//!
//! shortcut: only these imports are scaled; the frame-pacing clock, waits
//! with time-outs, wall-clock time and any counter the title reads without
//! an import still run in real time.
#![allow(non_snake_case)]

use std::sync::{Mutex, OnceLock};

use crate::HleModule;

/// How the title's clock runs.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// At `s` times real time (diagnostics builds).
    Fixed(f64),
    /// Slowed while frames come slower than `fps`.
    Auto { fps: f64 },
}

/// The slowest an automatic clock runs, against real time.
const AUTO_MIN_RATE: f64 = 0.1;

/// How often an automatic clock looks at the frame rate (real seconds).
const AUTO_WINDOW: f64 = 0.5;

/// The mode a `VICTORY_TIME_SCALE` value asks for; fixed scales only in
/// `diagnostics` builds.
fn parse(value: &str, diagnostics: bool) -> Option<Mode> {
    let value = value.trim();
    if let Some(rest) = value.strip_prefix("auto") {
        let fps = match rest.strip_prefix(':') {
            Some(n) => n
                .trim()
                .parse()
                .ok()
                .filter(|&f: &f64| f > 0.0 && f <= 1000.0)?,
            None if rest.is_empty() => 10.0,
            None => return None,
        };
        return Some(Mode::Auto { fps });
    }
    if !diagnostics {
        return None;
    }
    let s: f64 = value.parse().ok()?;
    (s > 0.0 && s < 1.0).then_some(Mode::Fixed(s))
}

/// The mode `VICTORY_TIME_SCALE` sets, if any.
fn mode() -> Option<Mode> {
    static MODE: OnceLock<Option<Mode>> = OnceLock::new();
    *MODE.get_or_init(|| {
        let value = std::env::var("VICTORY_TIME_SCALE").ok()?;
        let mode = parse(&value, cfg!(feature = "diagnostics"));
        match mode {
            Some(Mode::Fixed(s)) => tracing::warn!(
                scale = s,
                "VICTORY_TIME_SCALE: the title's clock runs slower than real time"
            ),
            Some(Mode::Auto { fps }) => tracing::warn!(
                fps,
                "VICTORY_TIME_SCALE: the title's clock slows while frames come slower than this"
            ),
            None => tracing::warn!(value, "VICTORY_TIME_SCALE: not understood; ignored"),
        }
        mode
    })
}

/// Whether the title's clock is scaled (the module is registered only then).
pub(crate) fn enabled() -> bool {
    mode().is_some()
}

/// Real counter values when the title first read each clock.
struct Base {
    counter: i64,
    ticks: u64,
    frequency: i64,
}

fn base() -> &'static Base {
    static BASE: OnceLock<Base> = OnceLock::new();
    BASE.get_or_init(|| Base {
        counter: host::counter(),
        ticks: host::ticks(),
        frequency: host::frequency().max(1),
    })
}

/// An automatic clock: the real and title counter readings at the last
/// step, the rate between them, and where the current frame-rate window
/// began (real reading, frames presented).
#[derive(Debug)]
struct Auto {
    real: i64,
    title: f64,
    rate: f64,
    window_real: i64,
    window_frames: u64,
}

impl Auto {
    fn new(now: i64, frames: u64) -> Self {
        Self {
            real: now,
            title: now as f64,
            rate: 1.0,
            window_real: now,
            window_frames: frames,
        }
    }

    /// The title's counter at real counter `now`, with `frames` presented
    /// so far; `frequency` is in counts per second, `fps` the rate below
    /// which the clock slows.
    fn step(&mut self, now: i64, frames: u64, frequency: i64, fps: f64) -> i64 {
        if now > self.real {
            self.title += (now - self.real) as f64 * self.rate;
            self.real = now;
        }
        let elapsed = (self.real - self.window_real) as f64 / frequency as f64;
        if elapsed >= AUTO_WINDOW {
            let shown = frames.saturating_sub(self.window_frames) as f64;
            // Before the first frame there is no rate to follow.
            let wanted = if frames == 0 {
                1.0
            } else {
                (shown / elapsed / fps).clamp(AUTO_MIN_RATE, 1.0)
            };
            // Half way each window: one slow frame does not stop the clock.
            self.rate = 0.5 * (self.rate + wanted);
            self.window_real = self.real;
            self.window_frames = frames;
        }
        self.title as i64
    }
}

/// `now` (a real reading) as the title's clock shows it.
fn scaled_counter(now: i64) -> i64 {
    match mode() {
        Some(Mode::Fixed(s)) => {
            let b = base().counter;
            b + ((now - b) as f64 * s) as i64
        }
        Some(Mode::Auto { fps }) => {
            static AUTO: Mutex<Option<Auto>> = Mutex::new(None);
            let frames = victory_gpu::queue::frames_presented();
            let b = base();
            let mut auto = AUTO.lock().unwrap_or_else(|e| e.into_inner());
            auto.get_or_insert_with(|| Auto::new(b.counter, frames))
                .step(now, frames, b.frequency, fps)
        }
        None => now,
    }
}

fn scaled_ticks(now: u64) -> u64 {
    match mode() {
        Some(Mode::Fixed(s)) => {
            let b = base().ticks;
            b + (now.saturating_sub(b) as f64 * s) as u64
        }
        Some(Mode::Auto { .. }) => {
            // The counter's clock, in milliseconds from the same start.
            let b = base();
            let title = scaled_counter(host::counter());
            let ms = i128::from((title - b.counter).max(0)) * 1000 / i128::from(b.frequency);
            b.ticks + ms as u64
        }
        None => now,
    }
}

pub extern "system" fn QueryPerformanceCounter(count: *mut i64) -> i32 {
    let now = scaled_counter(host::counter());
    if !count.is_null() {
        // SAFETY: the title passes where the count goes.
        unsafe { count.write(now) };
    }
    1
}

pub extern "system" fn GetTickCount64() -> u64 {
    scaled_ticks(host::ticks())
}

pub extern "system" fn GetTickCount() -> u32 {
    scaled_ticks(host::ticks()) as u32
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "clock",
        dlls: &[
            "kernel32.dll",
            "kernelbase.dll",
            "api-ms-win-core-profile-*",
            "api-ms-win-core-sysinfo-*",
        ],
        exports: vec![
            crate::hle_export!(QueryPerformanceCounter),
            crate::hle_export!(GetTickCount64),
            crate::hle_export!(GetTickCount),
        ],
    }
}

#[cfg(windows)]
mod host {
    pub(super) fn counter() -> i64 {
        let mut n = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut n) };
        n
    }

    pub(super) fn ticks() -> u64 {
        // SAFETY: no arguments.
        unsafe { windows_sys::Win32::System::SystemInformation::GetTickCount64() }
    }

    pub(super) fn frequency() -> i64 {
        let mut n = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceFrequency(&mut n) };
        n
    }
}

#[cfg(not(windows))]
mod host {
    use std::time::Instant;

    fn start() -> Instant {
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        *START.get_or_init(Instant::now)
    }

    pub(super) fn counter() -> i64 {
        start().elapsed().as_nanos() as i64
    }

    pub(super) fn ticks() -> u64 {
        start().elapsed().as_millis() as u64
    }

    pub(super) fn frequency() -> i64 {
        1_000_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clock_rises_with_the_hosts_unscaled() {
        let mut a = 0;
        let mut b = 0;
        assert_eq!(QueryPerformanceCounter(&mut a), 1);
        assert_eq!(QueryPerformanceCounter(&mut b), 1);
        assert!(b >= a);
        assert!(GetTickCount64() >= u64::from(GetTickCount()) / 2);
        // No scale set in tests: readings are the host's.
        assert_eq!(scaled_counter(12_345), 12_345);
    }

    #[test]
    fn the_switch_reads_auto_and_fixed_scales() {
        assert_eq!(parse("auto", false), Some(Mode::Auto { fps: 10.0 }));
        assert_eq!(parse(" auto:20 ", false), Some(Mode::Auto { fps: 20.0 }));
        assert_eq!(parse("auto:0", false), None);
        assert_eq!(parse("automatic", false), None);
        assert_eq!(parse("0.25", false), None);
        assert_eq!(parse("0.25", true), Some(Mode::Fixed(0.25)));
        assert_eq!(parse("1.5", true), None);
    }

    #[test]
    fn an_automatic_clock_slows_with_the_frame_rate_and_never_goes_back() {
        let f = 1000; // counts per second
        let mut a = Auto::new(0, 1);
        // Real speed while 30 frames a second come.
        let mut frames = 1;
        let mut last = 0;
        for t in (100..=2000).step_by(100) {
            frames += 3;
            let now = a.step(t, frames, f, 10.0);
            assert!(now >= last);
            last = now;
        }
        assert!((a.rate - 1.0).abs() < 1e-9, "{}", a.rate);
        assert!((1900..=2000).contains(&last), "{last}");
        // One frame a second: the clock settles near a tenth of real time.
        for t in (2100..=12000).step_by(100) {
            if t % 1000 == 0 {
                frames += 1;
            }
            let now = a.step(t, frames, f, 10.0);
            assert!(now >= last);
            last = now;
        }
        assert!(a.rate < 0.2, "{}", a.rate);
        // A reading from just before the last one does not go back.
        assert!(a.step(11_990, frames, f, 10.0) >= last);
    }
}
