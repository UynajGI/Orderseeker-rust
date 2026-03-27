// ============================================================
// rotor_sde.rs — Self-Spinning Rotor XY Model
//               SDE Integration (Euler-Maruyama)
// ============================================================
// Reference:
//   Rouzaire & Levis (2021) "Defect Superdiffusion and Unbinding in a
//   2D XY Model of Self-Driven Rotors", PRL 127, 088004
//
// Model:
//   2D L×L square lattice, periodic BC.
//   Each site carries angle θ_i and quenched frequency Ω_i.
//
//   Langevin SDE (Euler-Maruyama):
//     θ_i(t+dt) = θ_i(t)
//               + [Ω_i + J Σ_{⟨ij⟩} sin(θ_j − θ_i)] dt
//               + sqrt(2 D dt) · ξ_i
//   ξ_i ~ N(0,1) i.i.d.
//   Ω_i ~ N(0, σ²)   (quenched disorder, fixed at t=0)
//
// Observables (→ CSV):
//   m(t)    = |⟨e^{iθ}⟩|              magnetisation / order parameter
//   C(r)    = ⟨cos(θ_i − θ_{i,j+r})⟩  spin-spin correlation (x-direction)
//   n_def(t)= number of +1 topological defects (vortices)
//
// Cargo.toml dependencies:
//   rand       = { version = "0.8", features = ["small_rng"] }
//   rand_distr = "0.4"
//
// Output files:
//   rotor_timeseries.csv   step, m, n_defects
//   rotor_correlation.csv  r, C_r
//   rotor_sigma_scan.csv   sigma, m_mean, m_std, avg_defects
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, Uniform};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

// ─────────────────────────────────────────────────────────────
// Section 1 — Lattice initialisation
// ─────────────────────────────────────────────────────────────

/// L×L lattice of spin angles and quenched frequencies.
/// Stored in flat row-major order: index = i * L + j.
struct RotorLattice {
    l:     usize,
    theta: Vec<f64>,   // spin angles  (NOT periodically wrapped; full real line)
    omega: Vec<f64>,   // quenched natural frequencies ~ N(0, sigma²)
}

impl RotorLattice {
    /// Initialise with random θ ∈ [0,2π) and ω ~ N(0,σ²).
    fn new(l: usize, sigma: f64, seed: u64) -> Self {
        let n          = l * l;
        let mut rng    = SmallRng::seed_from_u64(seed);
        let ang_dist   = Uniform::new(0.0_f64, 2.0 * PI);
        let omega_dist = Normal::new(0.0_f64, sigma).unwrap();
        let theta: Vec<f64> = (0..n).map(|_| rng.sample(ang_dist)).collect();
        let omega: Vec<f64> = (0..n).map(|_| rng.sample(omega_dist)).collect();
        RotorLattice { l, theta, omega }
    }

    // ── periodic flat-index helpers ─────────────────────────
    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }

    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i, j)] }

    #[inline]
    fn wrap(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
}

// ─────────────────────────────────────────────────────────────
// Section 2 — Deterministic drift  f(θ)  (vectorised over all sites)
// ─────────────────────────────────────────────────────────────

/// f_i = Ω_i + J Σ_{⟨ij⟩} sin(θ_j − θ_i)
///
/// Uses explicit periodic indexing (equivalent to Python `np.roll`).
/// Returns array of shape (L*L,).
fn drift(lat: &RotorLattice, j_coup: f64) -> Vec<f64> {
    let l = lat.l;
    let n = l * l;
    let mut f = lat.omega.clone();
    for i in 0..l {
        for j in 0..l {
            let k   = lat.idx(i, j);
            let ti  = lat.theta[k];
            let up    = lat.get(lat.wrap(i,  1), j);
            let down  = lat.get(lat.wrap(i, -1), j);
            let right = lat.get(i, lat.wrap(j,  1));
            let left  = lat.get(i, lat.wrap(j, -1));
            f[k] += j_coup * (
                (up    - ti).sin()
                + (down  - ti).sin()
                + (right - ti).sin()
                + (left  - ti).sin()
            );
        }
    }
    f
}

// ─────────────────────────────────────────────────────────────
// Section 3 — Euler-Maruyama step
// ─────────────────────────────────────────────────────────────

/// One EM step:
///   θ(t+dt) = θ(t) + f(θ)·dt + sqrt(2·D·dt)·ξ,   ξ ~ N(0,1)
///
/// Angles are NOT wrapped after the update (kept on the real line
/// for accurate ω_eff estimation; cos/sin observables handle wrapping).
///
/// Mirrors Python `em_step` exactly.
fn em_step(lat: &mut RotorLattice, j_coup: f64, d_noise: f64, dt: f64, rng: &mut SmallRng) {
    let n        = lat.l * lat.l;
    let noise_sd = (2.0 * d_noise * dt).sqrt();
    let f        = drift(lat, j_coup);
    let norm_d   = Normal::new(0.0_f64, 1.0_f64).unwrap();
    for k in 0..n {
        lat.theta[k] += f[k] * dt + noise_sd * rng.sample(norm_d);
    }
}

// ─────────────────────────────────────────────────────────────
// Section 4 — Observables
// ─────────────────────────────────────────────────────────────

/// m = |⟨e^{iθ}⟩|
fn magnetisation(lat: &RotorLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re * re + im * im).sqrt()
}

/// C(r) = ⟨cos(θ_{i,j} − θ_{i,j+r})⟩  averaged over all (i,j).
/// Returns Vec[0..=r_max], C[0] = 1.
fn spin_correlation(lat: &RotorLattice, r_max: usize) -> Vec<f64> {
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

// ─────────────────────────────────────────────────────────────
// Section 5 — Topological defect detection
//             (winding number on each elementary plaquette)
// ─────────────────────────────────────────────────────────────

/// Wrap angle difference to (−π, π].
#[inline]
fn angle_diff(a: f64, b: f64) -> f64 {
    let d = b - a;
    d - (2.0 * PI) * (d / (2.0 * PI)).round()
}

/// Count ±1 vortices by computing the discrete winding number on
/// every plaquette (i,j) → (i+1,j) → (i+1,j+1) → (i,j+1) → (i,j).
///
/// Returns (n_positive, n_negative).
/// Mirrors Python `detect_defects`.
fn count_defects(lat: &RotorLattice) -> (usize, usize) {
    let l = lat.l;
    let mut pos = 0_usize;
    let mut neg = 0_usize;

    for i in 0..l {
        let ip = (i + 1) % l;
        for j in 0..l {
            let jp = (j + 1) % l;
            // plaquette corners in counter-clockwise order:
            // (i,j) → (ip,j) → (ip,jp) → (i,jp) → (i,j)
            let d1 = angle_diff(lat.get(i,  j),  lat.get(ip, j));
            let d2 = angle_diff(lat.get(ip, j),  lat.get(ip, jp));
            let d3 = angle_diff(lat.get(ip, jp), lat.get(i,  jp));
            let d4 = angle_diff(lat.get(i,  jp), lat.get(i,  j));
            let winding = ((d1 + d2 + d3 + d4) / (2.0 * PI)).round() as i32;
            match winding {
                 1 => pos += 1,
                -1 => neg += 1,
                _  => {}
            }
        }
    }
    (pos, neg)
}

fn mean_f(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
fn std_dev_f(v: &[f64]) -> f64 {
    let mu = mean_f(v);
    (v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

// ─────────────────────────────────────────────────────────────
// Section 6 — Main simulation runner
// ─────────────────────────────────────────────────────────────

/// Collected result of one `run_spinning_rotor` call.
struct RotorResult {
    l:           usize,
    j_coup:      f64,
    d_noise:     f64,
    sigma:       f64,
    m_mean:      f64,
    m_std:       f64,
    c_r:         Vec<f64>,
    m_ts:        Vec<f64>,
    n_defects_ts: Vec<usize>,  // positive-defect count at each measurement
}

/// Run self-spinning rotor simulation.
///
/// Parameter correspondence with Python `run_spinning_rotor`:
///   l            ↔  L
///   j_coup       ↔  J
///   d_noise      ↔  D
///   sigma        ↔  sigma
///   dt           ↔  dt
///   t_transient  ↔  t_transient
///   t_measure    ↔  t_measure
///   measure_every↔  measure_every
///   seed         ↔  seed
fn run_spinning_rotor(
    l:             usize,
    j_coup:        f64,
    d_noise:       f64,
    sigma:         f64,
    dt:            f64,
    t_transient:   f64,
    t_measure:     f64,
    measure_every: usize,
    seed:          u64,
    verbose:       bool,
) -> RotorResult {
    let mut rng   = SmallRng::seed_from_u64(seed);
    let mut lat   = RotorLattice::new(l, sigma, seed);
    let n_trans   = (t_transient / dt).round() as usize;
    let n_measure = (t_measure   / dt).round() as usize;

    // ── transient (discarded) ───────────────────────────────
    for _ in 0..n_trans {
        em_step(&mut lat, j_coup, d_noise, dt, &mut rng);
    }
    if verbose {
        println!("[Rotor] L={l} J={j_coup} D={d_noise:.2} σ={sigma:.2} \
                  — transient done ({n_trans} steps)");
    }

    // ── production ──────────────────────────────────────────
    let mut m_ts:         Vec<f64>  = Vec::new();
    let mut n_defects_ts: Vec<usize> = Vec::new();

    for s in 0..n_measure {
        em_step(&mut lat, j_coup, d_noise, dt, &mut rng);
        if s % measure_every == 0 {
            m_ts.push(magnetisation(&lat));
            let (pos, _) = count_defects(&lat);
            n_defects_ts.push(pos);
        }
    }

    let r_max   = l / 2;
    let c_r     = spin_correlation(&lat, r_max);
    let m_mean  = mean_f(&m_ts);
    let m_std   = std_dev_f(&m_ts);
    let avg_def = n_defects_ts.iter().sum::<usize>() as f64 / n_defects_ts.len() as f64;

    if verbose {
        println!("        <m> = {m_mean:.4} ± {m_std:.4}   avg defects = {avg_def:.1}");
    }

    RotorResult { l, j_coup, d_noise, sigma, m_mean, m_std, c_r, m_ts, n_defects_ts }
}

// ─────────────────────────────────────────────────────────────
// Section 7 — Sigma scan  (disorder vs order)
// ─────────────────────────────────────────────────────────────

struct ScanRow {
    sigma:      f64,
    m_mean:     f64,
    m_std:      f64,
    avg_defects: f64,
}

/// Vary σ.  At σ=0 the model reduces to equilibrium XY with noise D.
/// Mirrors Python `sigma_scan`.
fn sigma_scan(
    l:           usize,
    sigma_arr:   &[f64],
    j_coup:      f64,
    d_noise:     f64,
    dt:          f64,
    t_transient: f64,
    t_measure:   f64,
    seed:        u64,
) -> Vec<ScanRow> {
    let mut rows = Vec::new();
    for &sigma in sigma_arr {
        let res = run_spinning_rotor(
            l, j_coup, d_noise, sigma, dt,
            t_transient, t_measure, 20, seed, false,
        );
        let avg_d = res.n_defects_ts.iter().sum::<usize>() as f64
                  / res.n_defects_ts.len() as f64;
        println!("  σ={sigma:.3}  <m>={:.4}  <n_def>={avg_d:.1}", res.m_mean);
        rows.push(ScanRow { sigma, m_mean: res.m_mean, m_std: res.m_std, avg_defects: avg_d });
    }
    rows
}

// ─────────────────────────────────────────────────────────────
// Section 8 — CSV output helpers
// ─────────────────────────────────────────────────────────────

/// rotor_timeseries.csv: columns  step, m, n_defects
fn write_timeseries(path: &str, res: &RotorResult, measure_every: usize) {
    let mut w = BufWriter::new(File::create(path).expect("cannot create file"));
    writeln!(w, "step,m,n_defects").unwrap();
    for (k, (&m, &nd)) in res.m_ts.iter().zip(&res.n_defects_ts).enumerate() {
        writeln!(w, "{},{:.8},{}", k * measure_every, m, nd).unwrap();
    }
    println!("Written: {path}");
}

/// rotor_correlation.csv: columns  r, C_r
fn write_correlation(path: &str, c_r: &[f64]) {
    let mut w = BufWriter::new(File::create(path).expect("cannot create file"));
    writeln!(w, "r,C_r").unwrap();
    for (r, &c) in c_r.iter().enumerate() {
        writeln!(w, "{},{:.8}", r, c).unwrap();
    }
    println!("Written: {path}");
}

/// rotor_sigma_scan.csv: columns  sigma, m_mean, m_std, avg_defects
fn write_sigma_scan(path: &str, rows: &[ScanRow]) {
    let mut w = BufWriter::new(File::create(path).expect("cannot create file"));
    writeln!(w, "sigma,m_mean,m_std,avg_defects").unwrap();
    for r in rows {
        writeln!(w, "{:.4},{:.6},{:.6},{:.3}", r.sigma, r.m_mean, r.m_std, r.avg_defects).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// Section 9 — Entry point  (mirrors Python __main__ block)
// ─────────────────────────────────────────────────────────────

fn main() {
    // ── Single run  (L=32, J=1.0, D=0.3, σ=0.2) ────────────
    let measure_every = 20_usize;
    let res = run_spinning_rotor(
        32,    // l
        1.0,   // j_coup
        0.3,   // d_noise
        0.2,   // sigma
        0.01,  // dt
        10.0,  // t_transient
        50.0,  // t_measure
        measure_every,
        42,    // seed
        true,  // verbose
    );
    write_timeseries("rotor_timeseries.csv", &res, measure_every);
    write_correlation("rotor_correlation.csv", &res.c_r);

    // ── Sigma scan  σ ∈ [0.0, 1.0], 8 points ────────────────
    // matches Python demo: L=16, J=1.0, D=0.3
    let sigma_arr: Vec<f64> = (0..8).map(|k| k as f64 / 7.0).collect();
    let scan = sigma_scan(
        16,          // l
        &sigma_arr,
        1.0,         // j_coup
        0.3,         // d_noise
        0.01,        // dt
        5.0,         // t_transient
        15.0,        // t_measure
        0,           // seed
    );
    write_sigma_scan("rotor_sigma_scan.csv", &scan);
}
