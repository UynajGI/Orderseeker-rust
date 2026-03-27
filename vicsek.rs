// ============================================================
// vicsek.rs — Vicsek Self-Propelled Particle Model
//             Discrete-Time Particle Dynamics
// ============================================================
// Reference:
//   Vicsek et al. (1995) "Novel type of phase transition in a system of
//   self-driven particles", PRL 75, 1226
//
// Model:
//   N particles in [0,L)² with periodic BC, constant speed v0.
//
//   Synchronous update (one discrete time step):
//     θ_i(t+1) = circular_mean_{j: |r_j−r_i|<R} θ_j(t)  +  η_i
//     r_i(t+1) = r_i(t) + v0 · (cos θ_i(t+1), sin θ_i(t+1))
//
//   η_i ~ Uniform(−η/2, +η/2)    (noise amplitude η)
//   Periodic BC: r_i %= L
//
// Observables (→ CSV):
//   φ_v = |⟨e^{iθ}⟩|  normalised mean velocity (order parameter)
//
// Cargo.toml dependencies:
//   rand       = { version = "0.8", features = ["small_rng"] }
//   rand_distr = "0.4"
//
// Output files:
//   vicsek_timeseries.csv   step, phi
//   vicsek_noise_scan.csv   rho, eta, phi_mean, phi_std
//   vicsek_snapshot.csv     x, y, cos_theta, sin_theta
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Uniform;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

// ─────────────────────────────────────────────────────────────
// Section 1 — Particle state
// ─────────────────────────────────────────────────────────────

/// Collection of N particles in a 2D square box of side L.
struct Particles {
    n:     usize,
    l:     f64,
    x:     Vec<f64>,   // x-positions  ∈ [0, L)
    y:     Vec<f64>,   // y-positions  ∈ [0, L)
    theta: Vec<f64>,   // heading angles  ∈ ℝ  (not wrapped to [0,2π) — for atan2)
}

impl Particles {
    /// Randomly initialise N particles in [0,L)² with uniform headings.
    fn random_init(n: usize, l: f64, seed: u64) -> Self {
        let mut rng   = SmallRng::seed_from_u64(seed);
        let pos_dist  = Uniform::new(0.0_f64, l);
        let ang_dist  = Uniform::new(0.0_f64, 2.0 * PI);
        let x:     Vec<f64> = (0..n).map(|_| rng.sample(pos_dist)).collect();
        let y:     Vec<f64> = (0..n).map(|_| rng.sample(pos_dist)).collect();
        let theta: Vec<f64> = (0..n).map(|_| rng.sample(ang_dist)).collect();
        Particles { n, l, x, y, theta }
    }
}

// ─────────────────────────────────────────────────────────────
// Section 2 — Core update step
// ─────────────────────────────────────────────────────────────

/// One synchronous Vicsek step.
///
/// For each particle i:
///   1. find all j with |r_j − r_i|_min-image < R  (includes self)
///   2. θ_i_new = atan2(Σ sin θ_j, Σ cos θ_j)  +  noise
///   3. move: r_i += v0 * (cos θ_i_new, sin θ_i_new),  then wrap to [0,L)
///
/// Mirrors Python `vicsek_step` exactly.
fn vicsek_step(
    p:      &mut Particles,
    v0:     f64,
    r_cut:  f64,
    eta:    f64,
    rng:    &mut SmallRng,
) {
    let n        = p.n;
    let l        = p.l;
    let r2_cut   = r_cut * r_cut;
    let noise_d  = Uniform::new(-eta / 2.0, eta / 2.0);

    // ── 1. Compute new headings from circular mean ──────────
    let mut theta_new = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum_s = 0.0_f64;
        let mut sum_c = 0.0_f64;
        for j in 0..n {
            // minimum-image displacement
            let mut dx = p.x[j] - p.x[i];
            let mut dy = p.y[j] - p.y[i];
            dx -= l * (dx / l).round();
            dy -= l * (dy / l).round();
            if dx * dx + dy * dy <= r2_cut {
                sum_s += p.theta[j].sin();
                sum_c += p.theta[j].cos();
            }
        }
        theta_new[i] = sum_s.atan2(sum_c) + rng.sample(noise_d);
    }

    // ── 2. Update positions with periodic wrap ──────────────
    for i in 0..n {
        p.x[i] = (p.x[i] + v0 * theta_new[i].cos()).rem_euclid(l);
        p.y[i] = (p.y[i] + v0 * theta_new[i].sin()).rem_euclid(l);
        p.theta[i] = theta_new[i];
    }
}

// ─────────────────────────────────────────────────────────────
// Section 3 — Observable
// ─────────────────────────────────────────────────────────────

/// φ_v = |⟨e^{iθ}⟩|  = sqrt(⟨cos θ⟩² + ⟨sin θ⟩²)
fn order_parameter(theta: &[f64]) -> f64 {
    let n  = theta.len() as f64;
    let re = theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re * re + im * im).sqrt()
}

fn mean_f(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
fn std_dev_f(v: &[f64]) -> f64 {
    let mu = mean_f(v);
    (v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

// ─────────────────────────────────────────────────────────────
// Section 4 — Main simulation runner
// ─────────────────────────────────────────────────────────────

/// Collected result of one `run_vicsek` call.
struct VicsekResult {
    n:        usize,
    rho:      f64,
    l:        f64,
    v0:       f64,
    r_cut:    f64,
    eta:      f64,
    phi_mean: f64,
    phi_std:  f64,
    phi_ts:   Vec<f64>,
    // final snapshot
    x_snap:   Vec<f64>,
    y_snap:   Vec<f64>,
    theta_snap: Vec<f64>,
}

/// Run Vicsek model simulation.
///
/// Parameter correspondence with Python `run_vicsek`:
///   n_par        ↔  N
///   rho          ↔  rho          (L = sqrt(N/rho))
///   v0           ↔  v0
///   r_cut        ↔  R
///   eta          ↔  eta
///   n_therm      ↔  n_therm
///   n_steps      ↔  n_steps
///   measure_every↔  measure_every
///   seed         ↔  seed
fn run_vicsek(
    n_par:        usize,
    rho:          f64,
    v0:           f64,
    r_cut:        f64,
    eta:          f64,
    n_therm:      usize,
    n_steps:      usize,
    measure_every: usize,
    seed:         u64,
    verbose:      bool,
) -> VicsekResult {
    let l       = (n_par as f64 / rho).sqrt();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut p   = Particles::random_init(n_par, l, seed);

    // ── thermalisation ──────────────────────────────────────
    for _ in 0..n_therm {
        vicsek_step(&mut p, v0, r_cut, eta, &mut rng);
    }
    if verbose {
        println!("[Vicsek] N={n_par} rho={rho:.2} eta={eta:.3} v0={v0} \
                  — thermalisation done ({n_therm} steps)");
    }

    // ── production ──────────────────────────────────────────
    let mut phi_ts: Vec<f64> = Vec::new();
    for s in 0..n_steps {
        vicsek_step(&mut p, v0, r_cut, eta, &mut rng);
        if s % measure_every == 0 {
            phi_ts.push(order_parameter(&p.theta));
        }
    }

    let phi_mean = mean_f(&phi_ts);
    let phi_std  = std_dev_f(&phi_ts);

    if verbose {
        println!("         <phi> = {phi_mean:.4} ± {phi_std:.4}");
    }

    VicsekResult {
        n: n_par, rho, l, v0, r_cut: r_cut, eta,
        phi_mean, phi_std, phi_ts,
        x_snap: p.x, y_snap: p.y, theta_snap: p.theta,
    }
}

// ─────────────────────────────────────────────────────────────
// Section 5 — Noise scan  (order–disorder transition)
// ─────────────────────────────────────────────────────────────

struct ScanRow {
    rho:      f64,
    eta:      f64,
    phi_mean: f64,
    phi_std:  f64,
}

/// Scan η values and collect ⟨φ⟩ for fixed (N, ρ).
/// Mirrors Python `noise_scan`.
fn noise_scan(
    n_par:    usize,
    rho:      f64,
    eta_arr:  &[f64],
    v0:       f64,
    r_cut:    f64,
    n_therm:  usize,
    n_steps:  usize,
    seed:     u64,
) -> Vec<ScanRow> {
    let mut rows = Vec::new();
    for &eta in eta_arr {
        let res = run_vicsek(n_par, rho, v0, r_cut, eta, n_therm, n_steps, 5, seed, false);
        println!("  N={n_par} rho={rho:.2} eta={eta:.3}  <phi>={:.4}", res.phi_mean);
        rows.push(ScanRow { rho, eta, phi_mean: res.phi_mean, phi_std: res.phi_std });
    }
    rows
}

// ─────────────────────────────────────────────────────────────
// Section 6 — CSV output helpers
// ─────────────────────────────────────────────────────────────

/// vicsek_timeseries.csv: columns  step, phi
fn write_timeseries(path: &str, phi_ts: &[f64], measure_every: usize) {
    let mut w = BufWriter::new(File::create(path).expect("cannot create file"));
    writeln!(w, "step,phi").unwrap();
    for (k, &phi) in phi_ts.iter().enumerate() {
        writeln!(w, "{},{:.8}", k * measure_every, phi).unwrap();
    }
    println!("Written: {path}");
}

/// vicsek_noise_scan.csv: columns  rho, eta, phi_mean, phi_std
fn write_noise_scan(path: &str, rows: &[ScanRow]) {
    let mut w = BufWriter::new(File::create(path).expect("cannot create file"));
    writeln!(w, "rho,eta,phi_mean,phi_std").unwrap();
    for r in rows {
        writeln!(w, "{:.4},{:.4},{:.6},{:.6}", r.rho, r.eta, r.phi_mean, r.phi_std).unwrap();
    }
    println!("Written: {path}");
}

/// vicsek_snapshot.csv: columns  x, y, cos_theta, sin_theta
fn write_snapshot(path: &str, res: &VicsekResult) {
    let mut w = BufWriter::new(File::create(path).expect("cannot create file"));
    writeln!(w, "x,y,cos_theta,sin_theta").unwrap();
    for i in 0..res.n {
        writeln!(
            w,
            "{:.6},{:.6},{:.6},{:.6}",
            res.x_snap[i], res.y_snap[i],
            res.theta_snap[i].cos(),
            res.theta_snap[i].sin()
        ).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// Section 7 — Entry point  (mirrors Python __main__ block)
// ─────────────────────────────────────────────────────────────

fn main() {
    // ── Single run  (N=300, ρ=4.0, η=0.1) ──────────────────
    let res = run_vicsek(
        300,  // n_par
        4.0,  // rho
        0.03, // v0
        1.0,  // r_cut  (= R)
        0.1,  // eta
        300,  // n_therm
        800,  // n_steps
        5,    // measure_every
        42,   // seed
        true, // verbose
    );
    write_timeseries("vicsek_timeseries.csv", &res.phi_ts, 5);
    write_snapshot("vicsek_snapshot.csv", &res);

    // ── Noise scan  η ∈ [0.01, 1.5],  ρ ∈ {2.0, 4.0} ───────
    // 15 evenly-spaced η values, matches Python demo
    let eta_arr: Vec<f64> = (0..15).map(|k| 0.01 + k as f64 * (1.5 - 0.01) / 14.0).collect();
    let mut all_rows = Vec::new();
    for rho in [2.0_f64, 4.0_f64] {
        let rows = noise_scan(
            300,      // n_par
            rho,      // rho
            &eta_arr,
            0.03,     // v0
            1.0,      // r_cut
            200,      // n_therm
            300,      // n_steps
            0,        // seed
        );
        all_rows.extend(rows);
    }
    write_noise_scan("vicsek_noise_scan.csv", &all_rows);
}
