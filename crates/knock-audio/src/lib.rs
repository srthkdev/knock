//! Knock's audio engine: the single source of truth for sound synthesis.
//!
//! * [`synthesize`]: contact events -> modal resonator bank (dry signal)
//! * [`room_and_mic`]: room reverb, mic gain jitter, absolute noise floor, optional grip rustle
//! * [`features`]: compact acoustic summary used by agents and baselines
//!
//! All randomness comes from a PCG64 generator seeded with a `u64`, so output is a pure
//! function of the inputs on every platform (native, Python, WASM).
//!
//! Feature flags: `python` builds the `knock_audio` PyO3 extension module, `wasm` the
//! wasm-bindgen API for the browser.

pub mod features;
pub mod filters;
pub mod room;
pub mod synth;

#[cfg(feature = "python")]
mod python;
#[cfg(feature = "wasm")]
mod wasm;

pub use features::{features, Features};
pub use room::room_and_mic;
pub use synth::{synthesize, Event};

/// Sample rate (Hz).
pub const FS: f64 = 44100.0;
/// Typical rms of a tap at the mic; sets the absolute noise floor.
pub const REF_LEVEL: f64 = 1e-3;
/// Default room reverb tail time constant (s). Short so the room does not cap modal decay
/// rates (a 0.05 s tail limited every decay to ~20/s, swamping wood/plastic damping).
pub const REVERB_TAIL_S: f64 = 0.01;
/// L2 norm of the room impulse response (~ that of the original 0.05 s tail), which keeps the
/// wet level, and so loudness over the absolute noise floor, independent of the tail time.
pub const REVERB_IR_GAIN: f64 = 0.14;

/// The engine's PRNG (PCG XSL-RR 128/64), seeded from a `u64`.
pub fn rng(seed: u64) -> rand_pcg::Pcg64 {
    use rand::SeedableRng;
    rand_pcg::Pcg64::seed_from_u64(seed)
}

thread_local! {
    // FFT plans are expensive to build for mixed-radix sizes; reuse them across calls.
    static PLANNER: std::cell::RefCell<realfft::RealFftPlanner<f64>> = std::cell::RefCell::new(realfft::RealFftPlanner::new());
}

pub(crate) fn plan_forward(n: usize) -> std::sync::Arc<dyn realfft::RealToComplex<f64>> {
    PLANNER.with(|p| p.borrow_mut().plan_fft_forward(n))
}

pub(crate) fn plan_inverse(n: usize) -> std::sync::Arc<dyn realfft::ComplexToReal<f64>> {
    PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(n))
}

/// Build events from a flat `[t0, J0, tau0, t1, J1, tau1, ...]` buffer.
pub fn events_from_flat(flat: &[f64]) -> Vec<Event> {
    assert!(flat.len() % 3 == 0, "events must be a flat array of (time_s, impulse_Ns, contact_time_s) triples");
    flat.chunks_exact(3).map(|c| Event { t: c[0], impulse: c[1], tau: c[2] }).collect()
}

/// Convenience: `room_and_mic(synthesize(...))` with independent sub-seeds.
#[allow(clippy::too_many_arguments)]
pub fn render(events: &[Event], freqs: &[f64], decay: &[f64], gains: &[f64], seed: u64, seconds: f64, rustle: bool, tail_s: f64) -> Vec<f64> {
    use rand::Rng;
    let mut r = rng(seed);
    let (s1, s2): (u64, u64) = (r.random(), r.random());
    room_and_mic(&synthesize(events, freqs, decay, gains, s1, seconds), s2, rustle, tail_s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plausible hollow-ish and solid-ish mode set: the engine should keep them apart.
    fn modes(f0: f64, n: usize, a: f64, b: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let f: Vec<f64> = (1..=n).map(|i| f0 * i as f64 * (1.0 + 0.07 * i as f64)).collect();
        let d = f.iter().map(|x| a + b * x).collect();
        let g = (1..=n).map(|i| 1.0 / i as f64).collect();
        (f, d, g)
    }

    #[test]
    fn full_chain_end_to_end() {
        let ev = [Event { t: 0.12, impulse: 0.02, tau: 0.0004 }];
        let (f, d, g) = modes(700.0, 8, 5.0, 0.01);
        let x = render(&ev, &f, &d, &g, 11, 1.4, false, REVERB_TAIL_S);
        assert_eq!(x, render(&ev, &f, &d, &g, 11, 1.4, false, REVERB_TAIL_S));
        let ft = features(&x);
        assert_eq!(ft.distinct_hits, 1);
        assert!(ft.loudness_db_over_floor > 30.0, "{ft:?}");
        assert!(ft.ring_time_s > 0.05, "{ft:?}");

        let (f2, d2, g2) = modes(3000.0, 8, 5.0, 0.01);
        let y = features(&render(&ev, &f2, &d2, &g2, 11, 1.4, false, REVERB_TAIL_S));
        assert!(y.brightness_hz > ft.brightness_hz);
    }

    #[test]
    fn flat_events() {
        let e = events_from_flat(&[0.1, 1.0, 0.001, 0.2, 2.0, 0.002]);
        assert_eq!(e.len(), 2);
        assert_eq!(e[1], Event { t: 0.2, impulse: 2.0, tau: 0.002 });
    }
}
