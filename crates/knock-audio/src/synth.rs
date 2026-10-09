//! Modal synthesis: contact events excite a bank of 2-pole resonators.

use rand::Rng;
use rand_distr::StandardNormal;
use std::f64::consts::PI;


use crate::FS;

/// One contact: start time (s), impulse (N s), contact duration (s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Event {
    pub t: f64,
    pub impulse: f64,
    pub tau: f64,
}

/// Port of `knock.sound.synthesize`: half-sine force pulses (unit area, contact-duration wide)
/// excite each mode with a random strike-position gain; a short noise click rides on top.
/// Each mode is `lfilter([sin w], [1, -2 r cos w, r^2])` with `r = exp(-decay/FS)`.
pub fn synthesize(events: &[Event], freqs: &[f64], decay: &[f64], gains: &[f64], seed: u64, seconds: f64) -> Vec<f64> {
    assert!(freqs.len() == decay.len() && freqs.len() == gains.len(), "freqs/decay/gains length mismatch");
    let n = (seconds * FS) as usize;
    let nm = freqs.len();
    let mut rng = crate::rng(seed);
    let mut out = vec![0.0f64; n];

    // (start index, half-sine pulse, per-mode amplitude) for every event that fits in the clip
    let mut hits: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::with_capacity(events.len());
    for e in events {
        let w = ((e.tau * FS) as i64).max(2) as usize;
        let i = (e.t * FS) as i64;
        if i < 0 || i as usize + w >= n {
            continue;
        }
        let i = i as usize;
        let mut pulse: Vec<f64> = (0..w).map(|k| (PI * k as f64 / w as f64).sin()).collect();
        let s: f64 = pulse.iter().sum();
        pulse.iter_mut().for_each(|p| *p /= s);
        let u: f64 = rng.random_range(0.1..0.9);
        let v: f64 = rng.random_range(0.1..0.9);
        let amps: Vec<f64> = (0..nm)
            .map(|k| {
                let idx = (k + 1) as f64;
                let pos = ((PI * u * idx).sin() * (PI * v * idx * 0.7 + 0.3).sin()).abs();
                gains[k] * (0.3 + pos) * e.impulse
            })
            .collect();
        for o in &mut out[i..i + w] {
            let g: f64 = rng.sample(StandardNormal);
            *o += e.impulse * g * 0.15;
        }
        hits.push((i, pulse, amps));
    }

    // Sparse excitation: only samples under a contact pulse carry input. `row_of[i]` points at
    // that sample's per-mode input row (modes padded to a multiple of LANES).
    let nmp = nm.div_ceil(LANES) * LANES;
    let mut row_of = vec![u32::MAX; n];
    let mut rows: Vec<f64> = Vec::new();
    for (i, pulse, amps) in &hits {
        for (j, p) in pulse.iter().enumerate() {
            let r = &mut row_of[i + j];
            if *r == u32::MAX {
                *r = (rows.len() / nmp) as u32;
                rows.resize(rows.len() + nmp, 0.0);
            }
            let row = &mut rows[*r as usize * nmp..][..nm];
            row.iter_mut().zip(amps).for_each(|(x, a)| *x += a * p);
        }
    }

    // Resonator coefficients, padded with silent modes.
    let mut b0 = vec![0.0f64; nmp];
    let mut a1 = vec![0.0f64; nmp];
    let mut a2 = vec![0.0f64; nmp];
    for k in 0..nm {
        let r = (-decay[k] / FS).exp();
        let wk = 2.0 * PI * freqs[k] / FS;
        b0[k] = wk.sin();
        a1[k] = -2.0 * r * wk.cos();
        a2[k] = r * r;
    }

    // Run LANES independent recursions side by side so their latency chains overlap (SIMD).
    for blk in (0..nmp).step_by(LANES) {
        let b: [f64; LANES] = b0[blk..blk + LANES].try_into().unwrap();
        let p: [f64; LANES] = a1[blk..blk + LANES].try_into().unwrap();
        let q: [f64; LANES] = a2[blk..blk + LANES].try_into().unwrap();
        let (mut y1, mut y2) = ([0.0f64; LANES], [0.0f64; LANES]);
        for (i, o) in out.iter_mut().enumerate() {
            let r = row_of[i];
            let mut y = [0.0f64; LANES];
            if r == u32::MAX {
                for j in 0..LANES {
                    y[j] = -p[j] * y1[j] - q[j] * y2[j];
                }
            } else {
                let x = &rows[r as usize * nmp + blk..][..LANES];
                for j in 0..LANES {
                    y[j] = b[j] * x[j] - p[j] * y1[j] - q[j] * y2[j];
                }
            }
            *o += y.iter().sum::<f64>();
            y2 = y1;
            y1 = y;
        }
    }
    out
}

const LANES: usize = 8;

/// Straightforward per-mode reference (one dense `lfilter` per mode, as in the Python code);
/// used to check the batched implementation.
#[cfg(test)]
fn synthesize_reference(events: &[Event], freqs: &[f64], decay: &[f64], gains: &[f64], seed: u64, seconds: f64) -> Vec<f64> {
    use crate::filters::resonator_add;
    let n = (seconds * FS) as usize;
    let nm = freqs.len();
    let mut rng = crate::rng(seed);
    let mut out = vec![0.0f64; n];
    let mut exc = vec![vec![0.0f64; n]; nm];
    for e in events {
        let w = ((e.tau * FS) as i64).max(2) as usize;
        let i = (e.t * FS) as i64;
        if i < 0 || i as usize + w >= n {
            continue;
        }
        let i = i as usize;
        let mut pulse: Vec<f64> = (0..w).map(|k| (PI * k as f64 / w as f64).sin()).collect();
        let s: f64 = pulse.iter().sum();
        pulse.iter_mut().for_each(|p| *p /= s);
        let u: f64 = rng.random_range(0.1..0.9);
        let v: f64 = rng.random_range(0.1..0.9);
        for (k, ek) in exc.iter_mut().enumerate() {
            let idx = (k + 1) as f64;
            let pos = ((PI * u * idx).sin() * (PI * v * idx * 0.7 + 0.3).sin()).abs();
            for j in 0..w {
                ek[i + j] += gains[k] * (0.3 + pos) * e.impulse * pulse[j];
            }
        }
        for o in &mut out[i..i + w] {
            let g: f64 = rng.sample(StandardNormal);
            *o += e.impulse * g * 0.15;
        }
    }
    for k in 0..nm {
        let r = (-decay[k] / FS).exp();
        let wk = 2.0 * PI * freqs[k] / FS;
        resonator_add(&mut out, wk.sin(), -2.0 * r * wk.cos(), r * r, |j| exc[k][j]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tap() -> Vec<Event> {
        vec![Event { t: 0.1, impulse: 0.01, tau: 0.0005 }]
    }

    #[test]
    fn deterministic_for_seed() {
        let f = [440.0, 1200.0];
        let a = synthesize(&tap(), &f, &[20.0, 40.0], &[1.0, 0.5], 7, 1.0);
        let b = synthesize(&tap(), &f, &[20.0, 40.0], &[1.0, 0.5], 7, 1.0);
        let c = synthesize(&tap(), &f, &[20.0, 40.0], &[1.0, 0.5], 8, 1.0);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 44100);
    }

    #[test]
    fn silent_before_event_and_without_events() {
        let x = synthesize(&tap(), &[500.0], &[10.0], &[1.0], 1, 0.5);
        assert!(x[..4410].iter().all(|&v| v == 0.0));
        assert!(x[4410..].iter().any(|&v| v != 0.0));
        let y = synthesize(&[], &[500.0], &[10.0], &[1.0], 1, 0.5);
        assert!(y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn batched_matches_per_mode_reference() {
        // 11 modes (not a multiple of the lane width) and overlapping contact pulses
        let f: Vec<f64> = (0..11).map(|k| 150.0 + 830.0 * k as f64).collect();
        let d: Vec<f64> = f.iter().map(|x| 3.0 + 0.004 * x).collect();
        let g: Vec<f64> = (0..11).map(|k| 1.0 / (k + 1) as f64).collect();
        let ev = [
            Event { t: 0.05, impulse: 0.02, tau: 0.002 },
            Event { t: 0.0505, impulse: 0.01, tau: 0.001 },
            Event { t: 0.4, impulse: 0.005, tau: 0.00001 },
        ];
        let a = synthesize(&ev, &f, &d, &g, 99, 0.8);
        let b = synthesize_reference(&ev, &f, &d, &g, 99, 0.8);
        let peak = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() <= 1e-12 * peak, "{x} vs {y}");
        }
    }

    #[test]
    fn events_past_the_end_are_dropped() {
        let late = [Event { t: 0.999, impulse: 1.0, tau: 0.01 }];
        let x = synthesize(&late, &[500.0], &[10.0], &[1.0], 1, 1.0);
        assert!(x.iter().all(|&v| v == 0.0));
    }
}
