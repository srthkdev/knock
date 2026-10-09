//! Room reverb, mic gain jitter, absolute noise floor and optional grip rustle.

use rand::Rng;
use rand_distr::StandardNormal;
use std::f64::consts::PI;

use crate::filters::{butter_bandpass, butter_lowpass, sosfilt};
use crate::{FS, REF_LEVEL, REVERB_IR_GAIN};

/// `np.convolve(x, h)[:len(x)]` via FFT overlap-add (power-of-two blocks about twice the IR).
pub fn convolve_head(x: &[f64], h: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    if x.is_empty() || h.is_empty() {
        return out;
    }
    let nfft = (2 * h.len()).next_power_of_two().max(1024);
    let block = nfft - h.len() + 1;
    let fwd = crate::plan_forward(nfft);
    let inv = crate::plan_inverse(nfft);
    let mut scratch = vec![Default::default(); fwd.get_scratch_len().max(inv.get_scratch_len())];
    let mut buf = vec![0.0; nfft];
    buf[..h.len()].copy_from_slice(h);
    let mut fh = fwd.make_output_vec();
    fwd.process_with_scratch(&mut buf, &mut fh, &mut scratch).unwrap();
    let scale = 1.0 / nfft as f64;
    fh.iter_mut().for_each(|c| *c *= scale);
    let mut fx = fwd.make_output_vec();
    for start in (0..x.len()).step_by(block) {
        let seg = &x[start..(start + block).min(x.len())];
        buf.fill(0.0);
        buf[..seg.len()].copy_from_slice(seg);
        fwd.process_with_scratch(&mut buf, &mut fx, &mut scratch).unwrap();
        fx.iter_mut().zip(&fh).for_each(|(p, q)| *p *= *q);
        // a real signal's spectrum: DC and Nyquist bins must be real for the inverse
        fx[0].im = 0.0;
        fx[nfft / 2].im = 0.0;
        inv.process_with_scratch(&mut fx, &mut buf, &mut scratch).unwrap();
        let end = (start + nfft).min(x.len());
        out[start..end].iter_mut().zip(&buf).for_each(|(o, v)| *o += v);
    }
    out
}

fn normals(rng: &mut impl Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.sample::<f64, _>(StandardNormal)).collect()
}

fn set_rms(x: &mut [f64], target: f64) {
    let rms = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt();
    let k = if rms > 0.0 { target / rms } else { 0.0 };
    x.iter_mut().for_each(|v| *v *= k);
}

/// Port of `knock.sound.room_and_mic`. `tail_s` is the reverb tail's exponential time
/// constant (default [`crate::REVERB_TAIL_S`]); the IR is a direct path plus a noise tail,
/// normalised to L2 gain [`REVERB_IR_GAIN`], so shorter tails carry less energy relative to
/// the direct sound while the overall wet level stays calibrated.
pub fn room_and_mic(x: &[f64], seed: u64, rustle: bool, tail_s: f64) -> Vec<f64> {
    assert!(tail_s > 0.0, "tail_s must be positive");
    let mut rng = crate::rng(seed);
    let l = (((0.25f64).min(10.0 * tail_s) * FS) as usize).max(1);
    let mut ir: Vec<f64> = normals(&mut rng, l)
        .into_iter()
        .enumerate()
        .map(|(k, g)| g * (-(k as f64) / (tail_s * FS)).exp())
        .collect();
    ir[0] = 25.0;
    let k = REVERB_IR_GAIN / ir.iter().map(|v| v * v).sum::<f64>().sqrt();
    ir.iter_mut().for_each(|v| *v *= k);

    let mut y = convolve_head(x, &ir);
    let jitter = 10f64.powf(rng.random_range(-4.0..4.0) / 20.0);
    y.iter_mut().for_each(|v| *v *= jitter);
    if y.is_empty() {
        return y;
    }

    let mut noise = normals(&mut rng, y.len());
    sosfilt(&butter_lowpass(2, 2000.0, FS), &mut noise);
    set_rms(&mut noise, REF_LEVEL * 10f64.powf(-rng.random_range(32.0..42.0) / 20.0));

    if rustle {
        let mut r = normals(&mut rng, y.len());
        sosfilt(&butter_bandpass(2, 200.0, 3000.0, FS), &mut r);
        let f: f64 = rng.random_range(3.0..5.0);
        for (k, v) in r.iter_mut().enumerate() {
            *v *= 0.5 + 0.5 * (2.0 * PI * f * k as f64 / FS).sin().abs();
        }
        set_rms(&mut r, REF_LEVEL * 10f64.powf(-rng.random_range(34.0..44.0) / 20.0));
        noise.iter_mut().zip(&r).for_each(|(a, b)| *a += b);
    }
    y.iter_mut().zip(&noise).for_each(|(a, b)| *a += b);
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_convolution_matches_direct() {
        let x: Vec<f64> = (0..1000).map(|i| ((i * 37 % 101) as f64 - 50.0) / 50.0).collect();
        let h: Vec<f64> = (0..300).map(|i| (-(i as f64) / 40.0).exp() * if i % 3 == 0 { 1.0 } else { -0.5 }).collect();
        let fast = convolve_head(&x, &h);
        for n in 0..x.len() {
            let direct: f64 = (0..=n.min(h.len() - 1)).map(|k| h[k] * x[n - k]).sum();
            assert!((fast[n] - direct).abs() < 1e-9, "{n}");
        }
    }

    #[test]
    fn noise_floor_is_absolute_and_deterministic() {
        let silence = vec![0.0; 44100];
        let a = room_and_mic(&silence, 3, false, crate::REVERB_TAIL_S);
        assert_eq!(a, room_and_mic(&silence, 3, false, crate::REVERB_TAIL_S));
        let rms = (a.iter().map(|v| v * v).sum::<f64>() / a.len() as f64).sqrt();
        let db = 20.0 * (rms / REF_LEVEL).log10();
        assert!((-42.0..=-32.0).contains(&db), "{db}");
        let b = room_and_mic(&silence, 3, true, crate::REVERB_TAIL_S);
        let rb = (b.iter().map(|v| v * v).sum::<f64>() / b.len() as f64).sqrt();
        assert!(rb > rms);
    }

    #[test]
    fn shorter_tail_decays_faster_at_the_same_level() {
        // a loud click: the wet tail after it is set by the room, not by modes
        let mut x = vec![0.0; 44100];
        x[4410] = 1.0;
        let tail_db = |tail_s: f64| {
            let y = room_and_mic(&x, 9, false, tail_s);
            let e = |a: usize, b: usize| y[a..b].iter().map(|v| v * v).sum::<f64>();
            (10.0 * (e(4410 + 2205, 4410 + 3087) / e(4410, 4410 + 882)).log10(), e(0, 44100))
        };
        let (short, e_short) = tail_db(0.01);
        let (long, e_long) = tail_db(0.05);
        // 50-70 ms after the hit vs the first 20 ms
        assert!(short < -35.0, "{short}");
        assert!(long > -20.0, "{long}");
        assert!(short < long - 20.0);
        // the IR has a fixed L2 gain: overall wet energy is comparable
        let ratio_db = 10.0 * (e_short / e_long).log10();
        assert!(ratio_db.abs() < 6.0, "{ratio_db}");
    }

}
