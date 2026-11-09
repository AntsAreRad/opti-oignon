//! The value head's twin: the L2 logistic regression that orders a search's candidate files by their traits, as
//! the Python reference (`opti_oignon/mycelium/value.py`) learns and scores it.
//!
//! Python validates what it hands over (the rows, the labels, the strength) and refuses by name what the head
//! cannot take; this module takes rows already checked, and still refuses a shape that does not hold together. The
//! head is learned as the reference learns it: the traits centred on a mean summed around their first value and
//! scaled (a trait that never moves, or by a rounding only, is centred on its first value, scaled by 1 and read as
//! nought), Newton's method on the weights and the bias at once, the bias not held in, at
//! most the steps Python names, stopped once no coefficient moves by more than `TOLERANCE`; whether it settled is
//! answered, and Python refuses a head that did not. Each step's system is symmetric and positive definite and is
//! solved by Cholesky; the reference solves it by LU, and on the rows and strengths Python admits the two agree
//! within rounding. The interpreter's lock is let go while the head is learned or scored.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The shape of what the twin takes and answers. Python asks the twin only when this is the shape it hands over.
pub const FORMAT: u32 = 1;

const TOLERANCE: f64 = 1e-7;
/// A trait whose values lie apart by no more than this share of their size never moves: a rounding, not a spread.
const STILL: f64 = 1e-12;

fn refuse(why: &str) -> PyErr {
    PyValueError::new_err(format!("value head: {why}"))
}

/// Solve `a x = b` for a symmetric positive definite `a` (n x n, row-major) by Cholesky; None when `a` is not
/// positive definite.
fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(sum > 0.0) {
                    return None;
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    let mut z = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * z[k];
        }
        z[i] = sum / l[i * n + i];
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = z[i];
        for k in i + 1..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    Some(x)
}

/// The means and scales of the traits, and which never move. A mean is summed around the trait's first value, so
/// that a sum's rounding stays below the spread it measures; the scale is the population standard deviation, a
/// nought one replaced by 1. A trait whose least and greatest values lie apart by no more than `STILL` of their size
/// never moves: centred on its first value and scaled by 1, it is read as nought.
fn centre(width: usize, x: &[f64], rows: usize) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let (mut mean, mut scale, mut still) = (vec![0.0f64; width], vec![0.0f64; width], vec![false; width]);
    for j in 0..width {
        let first = x[j];
        let (mut shift, mut top, mut low) = (0.0f64, first, first);
        for r in 0..rows {
            let v = x[r * width + j];
            shift += v - first;
            top = top.max(v);
            low = low.min(v);
        }
        let m = first + shift / rows as f64;
        let mut spread = 0.0f64;
        for r in 0..rows {
            let d = x[r * width + j] - m;
            spread += d * d;
        }
        let s = (spread / rows as f64).sqrt();
        if top - low <= STILL * top.abs().max(low.abs()) {
            mean[j] = first;
            scale[j] = 1.0;
            still[j] = true;
        } else {
            mean[j] = m;
            scale[j] = if s == 0.0 { 1.0 } else { s };
        }
    }
    (mean, scale, still)
}

type Learned = (Vec<f64>, Vec<f64>, Vec<f64>, f64, usize, bool);

/// Learn the head: (mean, scale, weights, bias, steps, settled).
#[pyfunction]
pub fn value_fit(py: Python<'_>, width: usize, x: Vec<f64>, y: Vec<f64>, c: f64, max_steps: usize) -> PyResult<Learned> {
    py.allow_threads(|| learn(width, &x, &y, c, max_steps)).map_err(|why| refuse(&why))
}

fn learn(width: usize, x: &[f64], y: &[f64], c: f64, max_steps: usize) -> Result<Learned, String> {
    if width == 0 || x.is_empty() || x.len() % width != 0 {
        return Err("the rows do not hold whole rows of the traits".into());
    }
    let rows = x.len() / width;
    if y.len() != rows {
        return Err("not one label per row".into());
    }
    if !(c.is_finite() && c > 0.0) {
        return Err("the strength is not a positive number".into());
    }
    if !x.iter().chain(y.iter()).all(|v| v.is_finite()) || !y.iter().all(|&v| v == 0.0 || v == 1.0) {
        return Err("a value that is not finite, or a label neither 0 nor 1".into());
    }
    if max_steps == 0 {
        return Err("no step to learn in".into());
    }
    let (mean, scale, still) = centre(width, x, rows);
    let n = width + 1;
    // The rows centred and scaled, each with the bias's 1 at its end.
    let mut z = vec![0.0f64; rows * n];
    for r in 0..rows {
        for j in 0..width {
            z[r * n + j] = if still[j] { 0.0 } else { (x[r * width + j] - mean[j]) / scale[j] };
        }
        z[r * n + width] = 1.0;
    }
    let mut hold = vec![1.0 / c; n];
    hold[width] = 0.0;
    let mut theta = vec![0.0f64; n];
    let (mut steps, mut settled) = (0, false);
    for step in 1..=max_steps {
        steps = step;
        let mut gradient = vec![0.0f64; n];
        let mut h = vec![0.0f64; n * n];
        for r in 0..rows {
            let zr = &z[r * n..(r + 1) * n];
            let mut logit = 0.0;
            for k in 0..n {
                logit += zr[k] * theta[k];
            }
            let p = 1.0 / (1.0 + (-logit).exp());
            let weight = p * (1.0 - p);
            for i in 0..n {
                gradient[i] += zr[i] * (p - y[r]);
                let wi = zr[i] * weight;
                for j in 0..=i {
                    h[i * n + j] += wi * zr[j];
                }
            }
        }
        for i in 0..n {
            gradient[i] += hold[i] * theta[i];
            h[i * n + i] += hold[i];
            for j in 0..i {
                h[j * n + i] = h[i * n + j];
            }
        }
        let delta = cholesky_solve(&h, &gradient, n).ok_or("the step's system is not positive definite")?;
        let mut largest = 0.0f64;
        for k in 0..n {
            theta[k] -= delta[k];
            largest = largest.max(delta[k].abs());
        }
        if largest < TOLERANCE {
            settled = true;
            break;
        }
    }
    if !theta.iter().all(|v| v.is_finite()) {
        return Err("the head learned is not finite".into());
    }
    let bias = theta[width];
    theta.truncate(width);
    Ok((mean, scale, theta, bias, steps, settled))
}

/// The logits of rows under a head.
#[pyfunction]
pub fn value_scores(
    py: Python<'_>,
    mean: Vec<f64>,
    scale: Vec<f64>,
    weights: Vec<f64>,
    bias: f64,
    x: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let width = weights.len();
    if width == 0 || mean.len() != width || scale.len() != width || x.len() % width != 0 {
        return Err(refuse("the head and the rows do not hold the same traits"));
    }
    Ok(py.allow_threads(|| {
        let mut out = Vec::with_capacity(x.len() / width);
        for row in x.chunks(width) {
            let mut logit = bias;
            for j in 0..width {
                logit += (row[j] - mean[j]) / scale[j] * weights[j];
            }
            out.push(logit);
        }
        out
    }))
}
