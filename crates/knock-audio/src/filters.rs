//! IIR filters: the per-mode 2-pole resonator and Butterworth design as second-order sections,
//! matching `scipy.signal.butter(..., output="sos")` + `sosfilt`.

use std::f64::consts::PI;

/// `scipy.signal.lfilter([b0], [1, a1, a2], x)` with zero initial state, accumulated into `out`.
/// Only nonzero spans of the input need to be supplied: `x` is read lazily through `input`.
pub fn resonator_add(out: &mut [f64], b0: f64, a1: f64, a2: f64, input: impl Fn(usize) -> f64) {
    let (mut y1, mut y2) = (0.0f64, 0.0f64);
    for (i, o) in out.iter_mut().enumerate() {
        let y = b0 * input(i) - a1 * y1 - a2 * y2;
        *o += y;
        y2 = y1;
        y1 = y;
    }
}

/// One biquad section `[b0, b1, b2, 1, a1, a2]` (scipy sos row layout).
pub type Sos = [f64; 6];

/// `scipy.signal.sosfilt(sos, x)` in place (direct form II transposed, zero initial state).
pub fn sosfilt(sos: &[Sos], x: &mut [f64]) {
    for s in sos {
        let [b0, b1, b2, _a0, a1, a2] = *s;
        let (mut z1, mut z2) = (0.0f64, 0.0f64);
        for v in x.iter_mut() {
            let xi = *v;
            let y = b0 * xi + z1;
            z1 = b1 * xi - a1 * y + z2;
            z2 = b2 * xi - a2 * y;
            *v = y;
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct C(f64, f64);

impl C {
    fn add(self, o: C) -> C { C(self.0 + o.0, self.1 + o.1) }
    fn sub(self, o: C) -> C { C(self.0 - o.0, self.1 - o.1) }
    fn mul(self, o: C) -> C { C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0) }
    fn scale(self, k: f64) -> C { C(self.0 * k, self.1 * k) }
    fn div(self, o: C) -> C {
        let d = o.0 * o.0 + o.1 * o.1;
        C((self.0 * o.0 + self.1 * o.1) / d, (self.1 * o.0 - self.0 * o.1) / d)
    }
    fn abs(self) -> f64 { self.0.hypot(self.1) }
    fn sqrt(self) -> C {
        let r = self.abs();
        let re = ((r + self.0) / 2.0).max(0.0).sqrt();
        let im = ((r - self.0) / 2.0).max(0.0).sqrt();
        C(re, if self.1 < 0.0 { -im } else { im })
    }
}

/// Analog Butterworth prototype poles (scipy `buttap`).
fn buttap(order: usize) -> Vec<C> {
    let n = order as f64;
    (0..order)
        .map(|k| {
            let m = -(n - 1.0) + 2.0 * k as f64;
            let th = PI * m / (2.0 * n);
            // p = -exp(j*pi*m/(2N))
            C(-th.cos(), -th.sin())
        })
        .collect()
}

/// Bilinear transform of an analog zpk (scipy `bilinear_zpk`).
fn bilinear(z: &[C], p: &[C], k: f64, fs: f64) -> (Vec<C>, Vec<C>, f64) {
    let fs2 = C(2.0 * fs, 0.0);
    let mut zd: Vec<C> = z.iter().map(|&zi| fs2.add(zi).div(fs2.sub(zi))).collect();
    let pd: Vec<C> = p.iter().map(|&pi| fs2.add(pi).div(fs2.sub(pi))).collect();
    while zd.len() < pd.len() {
        zd.push(C(-1.0, 0.0));
    }
    let num = z.iter().fold(C(1.0, 0.0), |a, &zi| a.mul(fs2.sub(zi)));
    let den = p.iter().fold(C(1.0, 0.0), |a, &pi| a.mul(fs2.sub(pi)));
    (zd, pd, k * num.div(den).0)
}

/// Group a digital zpk with real zeros and conjugate pole pairs into biquads the way scipy's
/// `zpk2sos(pairing="nearest")` does: the pole pair closest to the unit circle is paired first
/// with its nearest zeros and placed last; the overall gain goes on the first section.
fn zpk2sos(mut z: Vec<C>, p: Vec<C>, k: f64) -> Vec<Sos> {
    // one representative (upper half-plane) pole per conjugate pair
    let mut pairs: Vec<C> = p.into_iter().filter(|c| c.1 >= 0.0).collect();
    pairs.sort_by(|a, b| (1.0 - a.abs()).abs().partial_cmp(&(1.0 - b.abs()).abs()).unwrap());
    let mut sections = Vec::new();
    for pp in pairs {
        let mut zs = Vec::new();
        for _ in 0..2 {
            let (i, _) = z
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.sub(pp).abs().partial_cmp(&b.1.sub(pp).abs()).unwrap())
                .unwrap();
            zs.push(z.remove(i));
        }
        let (z0, z1) = (zs[0].0, zs[1].0);
        sections.push([1.0, -(z0 + z1), z0 * z1, 1.0, -2.0 * pp.0, pp.0 * pp.0 + pp.1 * pp.1]);
    }
    sections.reverse();
    for v in &mut sections[0][..3] {
        *v *= k;
    }
    sections
}

/// `scipy.signal.butter(order, f, "low", fs=fs, output="sos")` for even `order`.
pub fn butter_lowpass(order: usize, f: f64, fs: f64) -> Vec<Sos> {
    let wo = 2.0 * fs * (PI * f / fs).tan();
    let p: Vec<C> = buttap(order).into_iter().map(|c| c.scale(wo)).collect();
    let (zd, pd, kd) = bilinear(&[], &p, wo.powi(order as i32), fs);
    zpk2sos(zd, pd, kd)
}

/// `scipy.signal.butter(order, [lo, hi], "band", fs=fs, output="sos")`.
pub fn butter_bandpass(order: usize, lo: f64, hi: f64, fs: f64) -> Vec<Sos> {
    let w1 = 2.0 * fs * (PI * lo / fs).tan();
    let w2 = 2.0 * fs * (PI * hi / fs).tan();
    let (bw, wo) = (w2 - w1, (w1 * w2).sqrt());
    let mut p = Vec::new();
    for c in buttap(order) {
        let lp = c.scale(bw / 2.0);
        let s = lp.mul(lp).sub(C(wo * wo, 0.0)).sqrt();
        p.push(lp.add(s));
        p.push(lp.sub(s));
    }
    let z = vec![C(0.0, 0.0); order];
    let (zd, pd, kd) = bilinear(&z, &p, bw.powi(order as i32), fs);
    zpk2sos(zd, pd, kd)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[Sos], b: &[Sos]) {
        assert_eq!(a.len(), b.len());
        for (ra, rb) in a.iter().zip(b) {
            for (x, y) in ra.iter().zip(rb) {
                assert!((x - y).abs() < 1e-12, "{ra:?} vs {rb:?}");
            }
        }
    }

    #[test]
    fn butter_matches_scipy() {
        // reference values from scipy 1.18 butter(..., fs=44100, output="sos")
        close(
            &butter_lowpass(2, 2000.0, 44100.0),
            &[[0.01681915010705713, 0.03363830021411426, 0.01681915010705713, 1.0, -1.6010923941836188, 0.6683689946118473]],
        );
        close(
            &butter_bandpass(2, 200.0, 3000.0, 44100.0),
            &[
                [0.03080173834487661, 0.06160347668975322, 0.03080173834487661, 1.0, -1.4642175285336692, 0.5919686591395004],
                [1.0, -2.0, 1.0, 1.0, -1.9603697363903296, 0.9612668665022573],
            ],
        );
    }

    #[test]
    fn sosfilt_dc_gain_of_lowpass_is_unity() {
        let sos = butter_lowpass(2, 2000.0, 44100.0);
        let mut x = vec![1.0; 5000];
        sosfilt(&sos, &mut x);
        assert!((x[4999] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn resonator_impulse_response_is_damped_sine() {
        let (fs, f, d) = (44100.0f64, 1000.0f64, 30.0f64);
        let r = (-d / fs).exp();
        let w = 2.0 * PI * f / fs;
        let mut out = vec![0.0; 200];
        resonator_add(&mut out, w.sin(), -2.0 * r * w.cos(), r * r, |i| if i == 0 { 1.0 } else { 0.0 });
        for (n, y) in out.iter().enumerate() {
            let expect = r.powi(n as i32) * (w * (n as f64 + 1.0)).sin();
            assert!((y - expect).abs() < 1e-12);
        }
    }
}
