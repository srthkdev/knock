//! Browser API (wasm-bindgen). Seeds are JS `bigint`s (u64); audio is Float64Array, with
//! Float32Array helpers for feeding Web Audio `AudioBuffer`s directly.

use wasm_bindgen::prelude::*;

/// Acoustic summary, rounded like the Python engine reports it.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Features {
    pub ring_time_s: f64,
    pub distinct_hits: u32,
    pub brightness_hz: f64,
    pub dominant_pitch_hz: f64,
    pub loudness_db_over_floor: f64,
}

/// Sample rate the engine renders at (Hz).
#[wasm_bindgen]
pub fn sample_rate() -> f64 {
    crate::FS
}

/// `events` is a flat Float64Array `[t0, J0, tau0, t1, J1, tau1, ...]`.
#[wasm_bindgen]
pub fn synthesize(events: &[f64], freqs: &[f64], decay: &[f64], gains: &[f64], seed: u64, seconds: f64) -> Result<Vec<f64>, JsError> {
    if events.len() % 3 != 0 {
        return Err(JsError::new("events must be a flat array of (time_s, impulse_Ns, contact_time_s) triples"));
    }
    if freqs.len() != decay.len() || freqs.len() != gains.len() {
        return Err(JsError::new("freqs, decay and gains must have the same length"));
    }
    Ok(crate::synthesize(&crate::events_from_flat(events), freqs, decay, gains, seed, seconds))
}

/// Default reverb tail time constant (s) to pass as `tail_s`.
#[wasm_bindgen]
pub fn reverb_tail_s() -> f64 {
    crate::REVERB_TAIL_S
}

/// `tail_s`: reverb tail time constant in seconds (use `reverb_tail_s()` for the default).
#[wasm_bindgen]
pub fn room_and_mic(x: &[f64], seed: u64, rustle: bool, tail_s: f64) -> Result<Vec<f64>, JsError> {
    if !(tail_s > 0.0) {
        return Err(JsError::new("tail_s must be positive"));
    }
    Ok(crate::room_and_mic(x, seed, rustle, tail_s))
}

/// Accepts Float64Array; `features_f32` accepts Float32Array.
#[wasm_bindgen]
pub fn features(x: &[f64]) -> Result<Features, JsError> {
    if x.len() < (0.005 * crate::FS) as usize {
        return Err(JsError::new("clip shorter than one 5 ms frame"));
    }
    let f = crate::features(x).rounded();
    Ok(Features {
        ring_time_s: f.ring_time_s,
        distinct_hits: f.distinct_hits,
        brightness_hz: f.brightness_hz,
        dominant_pitch_hz: f.dominant_pitch_hz,
        loudness_db_over_floor: f.loudness_db_over_floor,
    })
}

#[wasm_bindgen]
pub fn features_f32(x: &[f32]) -> Result<Features, JsError> {
    features(&x.iter().map(|&v| v as f64).collect::<Vec<_>>())
}

/// Full chain (synthesize + room_and_mic) returned as Float32Array for `AudioBuffer.copyToChannel`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn render_f32(events: &[f64], freqs: &[f64], decay: &[f64], gains: &[f64], seed: u64, seconds: f64, rustle: bool, tail_s: f64) -> Result<Vec<f32>, JsError> {
    if !(tail_s > 0.0) {
        return Err(JsError::new("tail_s must be positive"));
    }
    if events.len() % 3 != 0 {
        return Err(JsError::new("events must be a flat array of (time_s, impulse_Ns, contact_time_s) triples"));
    }
    if freqs.len() != decay.len() || freqs.len() != gains.len() {
        return Err(JsError::new("freqs, decay and gains must have the same length"));
    }
    let x = crate::render(&crate::events_from_flat(events), freqs, decay, gains, seed, seconds, rustle, tail_s);
    Ok(x.into_iter().map(|v| v as f32).collect())
}
