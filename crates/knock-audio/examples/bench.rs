//! Per-call timings of the engine stages:
//! `cargo run --release --example bench -p knock-audio`
use std::hint::black_box;
use std::time::Instant;

use knock_audio::{features, room_and_mic, synthesize, Event, REVERB_TAIL_S};

fn time(name: &str, n: u32, f: impl Fn()) {
    f();
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    println!("{name:14} {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / n as f64);
}

fn main() {
    let x: Vec<f64> = (0..61740).map(|i| ((i * 7919 % 1000) as f64 / 500.0 - 1.0) * 1e-3).collect();
    let ev = [Event { t: 0.1, impulse: 0.02, tau: 0.0004 }, Event { t: 0.5, impulse: 0.01, tau: 0.0004 }];
    let f: Vec<f64> = (0..16).map(|k| 300.0 + 700.0 * k as f64).collect();
    let d: Vec<f64> = f.iter().map(|v| 5.0 + 0.002 * v).collect();
    let g = vec![1.0; 16];
    time("synthesize", 200, || {
        black_box(synthesize(&ev, &f, &d, &g, 1, 1.4));
    });
    time("room", 200, || {
        black_box(room_and_mic(&x, 1, false, REVERB_TAIL_S));
    });
    time("room+rustle", 200, || {
        black_box(room_and_mic(&x, 1, true, REVERB_TAIL_S));
    });
    time("features", 200, || {
        black_box(features(&x));
    });
    let ir: Vec<f64> = (0..4410).map(|i| (-(i as f64) / 441.0).exp()).collect();
    time("convolve", 200, || {
        black_box(knock_audio::room::convolve_head(&x, &ir));
    });
    time("normals", 200, || {
        use rand::Rng;
        let mut r = knock_audio::rng(1);
        black_box((0..61740).map(|_| r.sample::<f64, _>(rand_distr::StandardNormal)).sum::<f64>());
    });
    let sos = knock_audio::filters::butter_bandpass(2, 200.0, 3000.0, 44100.0);
    time("sosfilt", 200, || {
        let mut y = x.clone();
        knock_audio::filters::sosfilt(&sos, &mut y);
        black_box(y);
    });
}
