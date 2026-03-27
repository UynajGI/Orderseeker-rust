// ============================================================
// xy_mc.rs — 2D XY Model  Monte Carlo (Metropolis)
// ============================================================
// Reference:
//   Mermin & Wagner (1966): equilibrium 2D XY has QLRO, no true LRO
//   Loos et al. (2023): non-reciprocal XY uses same lattice + Glauber MC
//
// Model:
//   H = -J * Σ_{<i,j>} cos(θ_i - θ_j)
//   θ_i ∈ [0, 2π),  L×L square lattice, periodic BC
//
// Observables (→ CSV):
//   m    = |⟨e^{iθ}⟩|           magnetisation / order parameter
//   C(r) = ⟨S_i · S_{i+r}⟩     spin-spin correlation (x-direction)
//   U_L  = 1 - ⟨m⁴⟩/(3⟨m²⟩²)   Binder cumulant (FSS)
//
// Cargo.toml dependencies:
//   rand       = { version = "0.8", features = ["small_rng"] }
//   rand_distr = "0.4"
//
// Build & run:
//   cargo build --release
//   ./target/release/xy_mc
//
// Output files:
//   xy_timeseries.csv    step, m
//   xy_correlation.csv   r, C_r
//   xy_binder_scan.csv   L, T, m_mean, m_std, binder
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Uniform;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

// ─────────────────────────────────────────────────────────────
// Section 1 — Lattice data structure
// ─────────────────────────────────────────────────────────────

/// Flat row-major L×L array of spin angles θ_i ∈ [0, 2π).
///
/// Layout: theta[i * l + j] = θ at row i, column j.
/// Periodic BC is handled everywhere via `wrap()`.
struct Lattice {
    l:     usize,
    theta: Vec<f64>,
}

impl Lattice {
    // ── flat index: (row i, col j) → i*L + j ───────────────
    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.l + j
    }

    // ── periodic wrap: coordinate x + delta  mod L ─────────
    // delta can be +1 or -1 (nearest-neighbour only).
    #[inline]
    fn wrap(&self, x: usize, delta: i64) -> usize {
        let l = self.l as i64;
        ((x as i64 + delta).rem_euclid(l)) as usize
    }

    // ── read θ[i,j] ────────────────────────────────────────
    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        self.theta[self.idx(i, j)]
    }

    // ── write θ[i,j] ───────────────────────────────────────
    #[inline]
    fn set(&mut self, i: usize, j: usize, val: f64) {
        let k = self.idx(i, j);
        self.theta[k] = val;
    }

    // ── Local field: Σ_{nb} cos(θ_i − θ_nb) for site (i,j)
    //    This equals −E_local / J  when all spins are included.
    fn local_field(&self, i: usize, j: usize) -> f64 {
        let ti    = self.get(i, j);
        let up    = self.get(self.wrap(i,  1), j);
        let down  = self.get(self.wrap(i, -1), j);
        let right = self.get(i, self.wrap(j,  1));
        let left  = self.get(i, self.wrap(j, -1));
        (ti - up).cos() + (ti - down).cos()
            + (ti - right).cos() + (ti - left).cos()
    }

    // ── Initialise with uniform random angles ───────────────
    fn random_init(l: usize, rng: &mut SmallRng) -> Self {
        let dist  = Uniform::new(0.0_f64, 2.0 * PI);
        let theta = (0..l * l).map(|_| rng.sample(dist)).collect();
        Lattice { l, theta }
    }

    // ── Initialise ferromagnetic state (all θ = 0) ──────────
    #[allow(dead_code)]
    fn ordered_init(l: usize) -> Self {
        Lattice { l, theta: vec![0.0; l * l] }
    }
}

// ─────────────────────────────────────────────────────────────
// Section 2 — One Metropolis sweep  (N = L² trials)
// ─────────────────────────────────────────────────────────────

/// Perform one full Metropolis sweep:
///   N = L² trials, each picking a random site, proposing
///   θ_new = θ_old + δ  (δ ~ Uniform(−delta, +delta)),
///   accepting with min(1, exp(−β ΔE)).
///
/// This is an exact translation of `mc_sweep` in `xy_model_mc.py`.
fn mc_sweep(lat: &mut Lattice, beta: f64, j_coup: f64, delta: f64, rng: &mut SmallRng) {
    let l         = lat.l;
    let n         = l * l;
    let idx_dist  = Uniform::new(0_usize, l);
    let dth_dist  = Uniform::new(-delta, delta);
    let acc_dist  = Uniform::new(0.0_f64, 1.0_f64);

    for _ in 0..n {
        let i = rng.sample(idx_dist);
        let j = rng.sample(idx_dist);

        let old_field = lat.local_field(i, j);
        let old_angle = lat.get(i, j);

        // propose new angle, wrap to [0, 2π)
        let new_angle = (old_angle + rng.sample(dth_dist)).rem_euclid(2.0 * PI);
        lat.set(i, j, new_angle);
        let new_field = lat.local_field(i, j);

        // ΔE = E_new − E_old = −J * (new_field − old_field)
        let de = -j_coup * (new_field - old_field);
        if de > 0.0 && rng.sample(acc_dist) >= (-beta * de).exp() {
            lat.set(i, j, old_angle); // reject: revert
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Section 3 — Observables
// ─────────────────────────────────────────────────────────────

/// m = |⟨e^{iθ}⟩| = sqrt(⟨cos θ⟩² + ⟨sin θ⟩²)
fn magnetisation(lat: &Lattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let mx = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let my = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (mx * mx + my * my).sqrt()
}

/// C(r) = ⟨cos(θ_{i,j} − θ_{i, j+r})⟩  averaged over all sites.
/// Returns Vec[0..=r_max], with C[0] = 1.
fn spin_correlation(lat: &Lattice, r_max: usize) -> Vec<f64> {
    let l = lat.l;
    let n = (l * l) as f64;
    let mut c = vec![0.0_f64; r_max + 1];
    c[0] = 1.0;
    for r in 1..=r_max {
        let s: f64 = (0..l)
            .flat_map(|i| (0..l).map(move |j| (i, j)))
            .map(|(i, j)| {
                let jr = (j + r) % l;
                (lat.get(i, j) - lat.get(i, jr)).cos()
            })
            .sum();
        c[r] = s / n;
    }
    c
}

/// U_L = 1 − ⟨m⁴⟩ / (3 ⟨m²⟩²)
/// → 2/3 in ordered phase, → 0 in disordered phase.
fn binder_cumulant(m_samples: &[f64]) -> f64 {
    let n  = m_samples.len() as f64;
    let m2 = m_samples.iter().map(|m| m * m).sum::<f64>() / n;
    let m4 = m_samples.iter().map(|m| m.powi(4)).sum::<f64>() / n;
    if m2 < 1e-15 {
        return 0.0;
    }
    1.0 - m4 / (3.0 * m2 * m2)
}

// ── statistics helpers ──────────────────────────────────────
fn mean_f(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
fn std_dev_f(v: &[f64]) -> f64 {
    let mu = mean_f(v);
    (v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

// ─────────────────────────────────────────────────────────────
// Section 4 — Main simulation runner
// ─────────────────────────────────────────────────────────────

/// Collected result of one `run_xy_mc` call.
struct XyResult {
    t:         f64,
    l:         usize,
    m_mean:    f64,
    m_std:     f64,
    binder:    f64,
    c_r:       Vec<f64>,
    m_samples: Vec<f64>,
}

/// Run equilibrium 2D XY Monte Carlo and collect statistics.
///
/// Exact parameter correspondence with Python `run_xy_mc`:
///   l            ↔  L
///   temp         ↔  T
///   j_coup       ↔  J
///   n_therm      ↔  n_therm
///   n_measure    ↔  n_measure
///   measure_every↔  measure_every
///   seed         ↔  seed
fn run_xy_mc(
    l:             usize,
    temp:          f64,
    j_coup:        f64,
    n_therm:       usize,
    n_measure:     usize,
    measure_every: usize,
    seed:          u64,
    verbose:       bool,
) -> XyResult {
    let beta    = 1.0 / temp;
    let delta   = PI;                        // trial-move half-width = π
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = Lattice::random_init(l, &mut rng);

    // ── thermalisation (discarded) ──────────────────────────
    for _ in 0..n_therm {
        mc_sweep(&mut lat, beta, j_coup, delta, &mut rng);
    }
    if verbose {
        println!("[XY-MC] L={l} T={temp:.3} — thermalisation done ({n_therm} sweeps)");
    }

    // ── production: measure every `measure_every` sweeps ────
    let mut m_samples: Vec<f64> = Vec::with_capacity(n_measure / measure_every + 1);
    for s in 0..n_measure {
        mc_sweep(&mut lat, beta, j_coup, delta, &mut rng);
        if s % measure_every == 0 {
            m_samples.push(magnetisation(&lat));
        }
    }

    let r_max  = l / 2;
    let c_r    = spin_correlation(&lat, r_max);
    let binder = binder_cumulant(&m_samples);
    let m_mean = mean_f(&m_samples);
    let m_std  = std_dev_f(&m_samples);

    if verbose {
        println!("        <m> = {m_mean:.4} ± {m_std:.4}   U_L = {binder:.4}");
    }

    XyResult { t: temp, l, m_mean, m_std, binder, c_r, m_samples }
}

// ─────────────────────────────────────────────────────────────
// Section 5 — Temperature scan  (finite-size scaling)
// ─────────────────────────────────────────────────────────────

/// One row in the Binder-cumulant scan table.
struct ScanRow {
    l:      usize,
    t:      f64,
    m_mean: f64,
    m_std:  f64,
    binder: f64,
}

/// Scan temperatures for multiple L values.
/// Mirrors Python `temperature_scan`.
fn temperature_scan(
    l_list:    &[usize],
    t_list:    &[f64],
    j_coup:    f64,
    n_therm:   usize,
    n_measure: usize,
    seed:      u64,
) -> Vec<ScanRow> {
    let mut rows = Vec::new();
    for &l in l_list {
        for &t in t_list {
            let res = run_xy_mc(l, t, j_coup, n_therm, n_measure, 10, seed, false);
            println!("  L={l:3}  T={t:.3}  <m>={:.4}  U_L={:.4}", res.m_mean, res.binder);
            rows.push(ScanRow { l, t, m_mean: res.m_mean, m_std: res.m_std, binder: res.binder });
        }
    }
    rows
}

// ─────────────────────────────────────────────────────────────
// Section 6 — CSV output helpers
// ─────────────────────────────────────────────────────────────

/// xy_timeseries.csv: columns  step, m
fn write_timeseries(path: &str, m_samples: &[f64], measure_every: usize) {
    let mut w = BufWriter::new(File::create(path).expect("cannot create file"));
    writeln!(w, "step,m").unwrap();
    for (k, &m) in m_samples.iter().enumerate() {
        writeln!(w, "{},{:.8}", k * measure_every, m).unwrap();
    }
    println!("Written: {path}");
}

/// xy_correlation.csv: columns  r, C_r
fn write_correlation(path: &str, c_r: &[f64]) {
    let mut w = BufWriter::new(File::create(path).expect("cannot create file"));
    writeln!(w, "r,C_r").unwrap();
    for (r, &c) in c_r.iter().enumerate() {
        writeln!(w, "{},{:.8}", r, c).unwrap();
    }
    println!("Written: {path}");
}

/// xy_binder_scan.csv: columns  L, T, m_mean, m_std, binder
fn write_binder_scan(path: &str, rows: &[ScanRow]) {
    let mut w = BufWriter::new(File::create(path).expect("cannot create file"));
    writeln!(w, "L,T,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}", r.l, r.t, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// Section 7 — Entry point  (mirrors Python __main__ block)
// ─────────────────────────────────────────────────────────────

fn main() {
    // ── Single production run  (L=32, T=0.8) ────────────────
    let res = run_xy_mc(
        32,     // l
        0.8,    // temp
        1.0,    // j_coup
        2000,   // n_therm
        5000,   // n_measure
        10,     // measure_every
        42,     // seed
        true,   // verbose
    );
    write_timeseries("xy_timeseries.csv", &res.m_samples, 10);
    write_correlation("xy_correlation.csv", &res.c_r);

    // ── Temperature scan  T ∈ [0.5, 1.5],  L ∈ {16, 32} ────
    // 11 evenly-spaced points, matches Python demo
    let t_list: Vec<f64> = (0..=10).map(|k| 0.5 + k as f64 * 0.1).collect();
    let scan = temperature_scan(
        &[16, 32],
        &t_list,
        1.0,  // j_coup
        500,  // n_therm
        1000, // n_measure
        0,    // seed
    );
    write_binder_scan("xy_binder_scan.csv", &scan);
}
