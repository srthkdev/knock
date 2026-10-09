//! Compact acoustic summary given to agents (port of `knock.sound.features`).

use std::f64::consts::PI;

use crate::FS;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Features {
    pub ring_time_s: f64,
    pub distinct_hits: u32,
    pub brightness_hz: f64,
    pub dominant_pitch_hz: f64,
    pub loudness_db_over_floor: f64,
}

/// Python's `round(x, nd)`: round half to even.
pub fn round_half_even(x: f64, nd: i32) -> f64 {
    let m = 10f64.powi(nd);
    let y = x * m;
    let r = y.round();
    let r = if (y - y.trunc()).abs() == 0.5 { 2.0 * (y / 2.0).round() } else { r };
    r / m
}

impl Features {
    /// Rounded exactly as the Python dict reports them.
    pub fn rounded(&self) -> Features {
        Features {
            ring_time_s: round_half_even(self.ring_time_s, 3),
            distinct_hits: self.distinct_hits,
            brightness_hz: round_half_even(self.brightness_hz, 0),
            dominant_pitch_hz: round_half_even(self.dominant_pitch_hz, 0),
            loudness_db_over_floor: round_half_even(self.loudness_db_over_floor, 1),
        }
    }
}

/// `np.percentile(v, q)` with linear interpolation.
fn percentile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = q / 100.0 * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Unrounded features. `x` must hold at least one 5 ms frame.
pub fn features(x: &[f64]) -> Features {
    let hop = (0.005 * FS) as usize;
    let nf = x.len() / hop;
    assert!(nf > 0, "features: clip shorter than one 5 ms frame");
    let env: Vec<f64> = x[..nf * hop]
        .chunks_exact(hop)
        .map(|fr| 20.0 * ((fr.iter().map(|v| v * v).sum::<f64>() / hop as f64).sqrt() + 1e-12).log10())
        .collect();
    let floor = percentile(&env, 10.0);
    let peak_i = argmax(&env);
    let loudness = env[peak_i] - floor;
    let target = (env[peak_i] - 25.0).max(floor + 3.0);
    let ring_frames = env[peak_i..].iter().position(|&e| e < target).unwrap_or(nf - peak_i);
    let ring = ring_frames as f64 * 0.005;

    let mut hits = 0u32;
    let mut prev: i64 = -99;
    for i in 0..nf.saturating_sub(3) {
        let rise = env[i + 3] - env[i];
        if rise > 8.0 && env[i + 3] > floor + 12.0 {
            if i as i64 - prev > 4 {
                hits += 1;
            }
            prev = i as i64;
        }
    }

    let start = peak_i * hop;
    let seg = &x[start..(start + (0.2 * FS) as usize).min(x.len())];
    let m = seg.len();
    // one-sided taper `np.hanning(2m)[m:]`: keeps the onset, where fast-decaying (solid) modes
    // live, now that the short room tail no longer smears them past a symmetric window's rise
    let m2 = (2 * m - 1) as f64;
    let mut buf: Vec<f64> = seg
        .iter()
        .enumerate()
        .map(|(k, v)| v * (0.5 - 0.5 * (2.0 * PI * (k + m) as f64 / m2).cos()))
        .collect();
    let fft = crate::plan_forward(m);
    let mut spec_c = fft.make_output_vec();
    fft.process(&mut buf, &mut spec_c).unwrap();
    let (mut num, mut den) = (0.0, 0.0);
    let (mut peak_hz, mut peak_v) = (0.0, f64::NEG_INFINITY);
    for (k, c) in spec_c.iter().enumerate() {
        let s = c.norm() + 1e-12;
        let f = k as f64 * FS / m as f64;
        num += f * s;
        den += s;
        if f > 80.0 && s > peak_v {
            peak_v = s;
            peak_hz = f;
        }
    }
    Features {
        ring_time_s: ring,
        distinct_hits: hits,
        brightness_hz: num / den,
        dominant_pitch_hz: peak_hz,
        loudness_db_over_floor: loudness,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_rounding() {
        assert_eq!(round_half_even(2.5, 0), 2.0);
        assert_eq!(round_half_even(3.5, 0), 4.0);
        assert_eq!(round_half_even(1234.4, 0), 1234.0);
        assert_eq!(round_half_even(0.125, 2), 0.12);
    }

    #[test]
    fn percentile_matches_numpy() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 10.0), 1.4);
        assert_eq!(percentile(&[7.0], 10.0), 7.0);
    }

    #[test]
    fn decaying_tone_features() {
        let (f0, d) = (1500.0, 8.0);
        let mut x: Vec<f64> = (0..61740).map(|i| 1e-6 * (((i * 7919) % 1000) as f64 / 500.0 - 1.0)).collect();
        let s0 = 4410;
        for (i, v) in x.iter_mut().enumerate().skip(s0) {
            let t = (i - s0) as f64 / FS;
            *v += 1e-2 * (-d * t).exp() * (2.0 * PI * f0 * t).sin();
        }
        let ft = features(&x);
        assert_eq!(ft.distinct_hits, 1);
        assert!((ft.dominant_pitch_hz - f0).abs() < 10.0, "{ft:?}");
        // -25 dB of an exp(-8t) envelope: t = 25 / (20 log10 e * 8) = 0.36 s
        assert!((ft.ring_time_s - 0.36).abs() < 0.02, "{ft:?}");
        assert!(ft.loudness_db_over_floor > 60.0, "{ft:?}");
    }
}
