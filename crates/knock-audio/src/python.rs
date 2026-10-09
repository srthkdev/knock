//! `knock_audio` Python extension module (PyO3 + numpy).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::Event;

fn vec1(a: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    a.as_array().iter().copied().collect()
}

/// synthesize(events, freqs, decay, gains, seed, seconds) -> float64 array
///
/// events: float64 array of shape (n, 3) with rows (time_s, impulse_Ns, contact_time_s).
#[pyfunction]
fn synthesize<'py>(
    py: Python<'py>,
    events: PyReadonlyArray2<'py, f64>,
    freqs: PyReadonlyArray1<'py, f64>,
    decay: PyReadonlyArray1<'py, f64>,
    gains: PyReadonlyArray1<'py, f64>,
    seed: u64,
    seconds: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ev = events.as_array();
    if ev.ncols() != 3 && ev.nrows() > 0 {
        return Err(PyValueError::new_err("events must have shape (n, 3)"));
    }
    let ev: Vec<Event> = ev.rows().into_iter().map(|r| Event { t: r[0], impulse: r[1], tau: r[2] }).collect();
    let (f, d, g) = (vec1(&freqs), vec1(&decay), vec1(&gains));
    if f.len() != d.len() || f.len() != g.len() {
        return Err(PyValueError::new_err("freqs, decay and gains must have the same length"));
    }
    let out = py.detach(|| crate::synthesize(&ev, &f, &d, &g, seed, seconds));
    Ok(out.into_pyarray(py))
}

/// room_and_mic(x, seed, rustle=False, tail_s=REVERB_TAIL_S) -> float64 array
#[pyfunction]
#[pyo3(signature = (x, seed, rustle=false, tail_s=crate::REVERB_TAIL_S))]
fn room_and_mic<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>, seed: u64, rustle: bool, tail_s: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !(tail_s > 0.0) {
        return Err(PyValueError::new_err("tail_s must be positive"));
    }
    let x = vec1(&x);
    let out = py.detach(|| crate::room_and_mic(&x, seed, rustle, tail_s));
    Ok(out.into_pyarray(py))
}

/// features(x) -> dict, rounded exactly like knock.sound.features
#[pyfunction]
fn features<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyDict>> {
    let x = vec1(&x);
    if x.len() < (0.005 * crate::FS) as usize {
        return Err(PyValueError::new_err("clip shorter than one 5 ms frame"));
    }
    let f = py.detach(|| crate::features(&x)).rounded();
    let d = PyDict::new(py);
    d.set_item("ring_time_s", f.ring_time_s)?;
    d.set_item("distinct_hits", f.distinct_hits)?;
    d.set_item("brightness_hz", f.brightness_hz as i64)?;
    d.set_item("dominant_pitch_hz", f.dominant_pitch_hz as i64)?;
    d.set_item("loudness_db_over_floor", f.loudness_db_over_floor)?;
    Ok(d)
}

#[pymodule]
fn knock_audio(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("FS", crate::FS)?;
    m.add("REF_LEVEL", crate::REF_LEVEL)?;
    m.add("REVERB_TAIL_S", crate::REVERB_TAIL_S)?;
    m.add("REVERB_IR_GAIN", crate::REVERB_IR_GAIN)?;
    m.add_function(wrap_pyfunction!(synthesize, m)?)?;
    m.add_function(wrap_pyfunction!(room_and_mic, m)?)?;
    m.add_function(wrap_pyfunction!(features, m)?)?;
    Ok(())
}
