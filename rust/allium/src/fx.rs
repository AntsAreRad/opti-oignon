//! Fixed point, Q16.16 in i32: the twin of `opti_oignon/allium/fx.py`.
//!
//! Intermediates are i128. For inputs held to i32 no primitive here comes
//! near the i128 bounds (the widest, `hill_up` with n = 4, stays under
//! 2^108), so every `saturating_*` below computes the exact value, the same
//! integer Python computes without bounds, and both then saturate to i32
//! identically. Shifts of a negative value floor, as Python's do, and
//! division is `div_euclid` by a positive divisor, which floors too.
//!
//! Every primitive is total: a correction of an input, a parameter or a
//! result increments `Work::alarm`, and nothing ever panics or refuses.

pub const ONE: i128 = 1 << 16;
pub const HALF: i128 = 1 << 15;
pub const I32_MIN: i128 = -(1 << 31);
pub const I32_MAX: i128 = (1 << 31) - 1;
pub const CMAX: i128 = 8 << 16;
pub const N_MIN: i128 = 1;
pub const N_MAX: i128 = 4;
pub const TAU_MIN: i128 = 1;
pub const TAU_MAX: i128 = 16;
pub const STEPS_MAX: i128 = 1 << 20;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Work {
    pub units: u64,
    pub alarm: u64,
}

impl Work {
    fn unit(&mut self) {
        self.units = self.units.saturating_add(1);
    }

    fn units_add(&mut self, n: i128) {
        let n = u64::try_from(n).unwrap_or(0);
        self.units = self.units.saturating_add(n);
    }

    fn alarm(&mut self) {
        self.alarm = self.alarm.saturating_add(1);
    }
}

fn input(value: i128, work: &mut Work) -> i128 {
    held(value, I32_MIN, I32_MAX, work)
}

fn output(value: i128, work: &mut Work) -> i128 {
    held(value, I32_MIN, I32_MAX, work)
}

fn held(value: i128, low: i128, high: i128, work: &mut Work) -> i128 {
    if value < low {
        work.alarm();
        return low;
    }
    if value > high {
        work.alarm();
        return high;
    }
    value
}

fn shr16(value: i128) -> i128 {
    value.wrapping_shr(16)
}

fn floor_div(a: i128, b: i128) -> i128 {
    // Callers guarantee b > 0; a zero divisor is not reachable, and would give 0.
    a.checked_div_euclid(b).unwrap_or(0)
}

fn pow_raw(c: i128, n: i128) -> i128 {
    let mut x = c;
    let mut round = 1;
    while round < n {
        x = shr16(x.saturating_mul(c));
        round = round.saturating_add(1);
    }
    x
}

pub fn mul(a: i128, b: i128, work: &mut Work) -> i128 {
    work.unit();
    let a = input(a, work);
    let b = input(b, work);
    output(shr16(a.saturating_mul(b)), work)
}

pub fn div(a: i128, b: i128, work: &mut Work) -> i128 {
    work.unit();
    let a = input(a, work);
    let b = input(b, work);
    if b <= 0 {
        work.alarm();
        if a > 0 {
            return I32_MAX;
        }
        if a < 0 {
            return I32_MIN;
        }
        return 0;
    }
    output(floor_div(a.wrapping_shl(16), b), work)
}

pub fn pow(c: i128, n: i128, work: &mut Work) -> i128 {
    work.unit();
    let c = input(c, work);
    let n = held(n, N_MIN, N_MAX, work);
    output(pow_raw(c, n), work)
}

pub fn hill_up(c: i128, k: i128, n: i128, work: &mut Work) -> i128 {
    work.unit();
    let c = held(input(c, work), 0, I32_MAX, work);
    let k = held(input(k, work), 1, CMAX, work);
    let n = held(n, N_MIN, N_MAX, work);
    let pc = pow_raw(c, n);
    let den = pow_raw(k, n).saturating_add(pc);
    if den <= 0 {
        return 0;
    }
    output(floor_div(pc.wrapping_shl(16), den), work)
}

pub fn hill_down(c: i128, k: i128, n: i128, work: &mut Work) -> i128 {
    ONE.saturating_sub(hill_up(c, k, n, work))
}

pub fn mm(c: i128, k: i128, work: &mut Work) -> i128 {
    work.unit();
    let c = held(input(c, work), 0, I32_MAX, work);
    let k = held(input(k, work), 1, CMAX, work);
    output(floor_div(c.wrapping_shl(16), k.saturating_add(c)), work)
}

pub fn sig(a: i128, work: &mut Work) -> i128 {
    work.unit();
    let a = input(a, work);
    let magnitude = if a < 0 { a.saturating_neg() } else { a };
    let q = floor_div(magnitude.saturating_mul(HALF), ONE.saturating_add(magnitude));
    if a < 0 {
        HALF.saturating_sub(q)
    } else {
        HALF.saturating_add(q)
    }
}

pub fn sat(x: i128, low: i128, high: i128, work: &mut Work) -> i128 {
    work.unit();
    let x = input(x, work);
    let low = input(low, work);
    let high = input(high, work);
    if low > high {
        work.alarm();
        return low;
    }
    if x < low {
        return low;
    }
    if x > high {
        return high;
    }
    x
}

pub fn isqrt(x: i128, work: &mut Work) -> i128 {
    work.unit();
    let x = held(input(x, work), 0, I32_MAX, work);
    let root = u64::try_from(x).unwrap_or(0).isqrt();
    i128::from(root)
}

pub fn sin_b(bam: i128, table: &[i128], work: &mut Work) -> i128 {
    work.unit();
    let index = usize::try_from(bam.rem_euclid(1024)).unwrap_or(0);
    table.get(index).copied().unwrap_or(0)
}

pub fn cos_b(bam: i128, table: &[i128], work: &mut Work) -> i128 {
    work.unit();
    let index = usize::try_from(bam.saturating_add(256).rem_euclid(1024)).unwrap_or(0);
    table.get(index).copied().unwrap_or(0)
}

pub fn lut(x: i128, table: &[i128], work: &mut Work) -> i128 {
    work.unit();
    let x = held(input(x, work), 0, ONE, work);
    let index = usize::try_from(x.wrapping_shr(8)).unwrap_or(0);
    if index >= 256 {
        return table.get(256).copied().unwrap_or(0);
    }
    let low = table.get(index).copied().unwrap_or(0);
    let next = table.get(index.saturating_add(1)).copied().unwrap_or(0);
    let step = next.saturating_sub(low).saturating_mul(x & 255).wrapping_shr(8);
    output(low.saturating_add(step), work)
}

pub fn decay_iter(x: i128, tau: i128, steps: i128, work: &mut Work) -> i128 {
    let mut x = input(x, work);
    let tau = held(tau, TAU_MIN, TAU_MAX, work);
    let steps = held(steps, 0, STEPS_MAX, work);
    let shift = u32::try_from(tau).unwrap_or(16);
    let mut done = 0;
    while done < steps {
        x = x.saturating_sub(x.wrapping_shr(shift));
        done = done.saturating_add(1);
    }
    work.units_add(steps);
    x
}

pub fn decay_lazy(v0: i128, tau: i128, age: i128, length: i128, work: &mut Work) -> i128 {
    work.unit();
    let v0 = input(v0, work);
    let tau = held(tau, TAU_MIN, TAU_MAX, work);
    let length = held(length, 1, STEPS_MAX, work);
    let age = held(age, 0, STEPS_MAX, work);
    if age >= length {
        return 0;
    }
    let shift = u32::try_from(tau).unwrap_or(16);
    let mut factor = ONE;
    let mut done = 0;
    while done < age {
        if factor == 0 {
            break;
        }
        factor = factor.saturating_sub(factor.wrapping_shr(shift));
        done = done.saturating_add(1);
    }
    output(shr16(v0.saturating_mul(factor)), work)
}

/// The primitives by name, with the number of integer arguments each takes.
pub fn arity(name: &str) -> Option<usize> {
    match name {
        "sig" | "isqrt" | "sin_b" | "cos_b" | "lut" => Some(1),
        "mul" | "div" | "pow" | "mm" => Some(2),
        "hill_up" | "hill_down" | "sat" | "decay_iter" => Some(3),
        "decay_lazy" => Some(4),
        _ => None,
    }
}
