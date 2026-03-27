// ============================================================
// kuramoto.rs — Kuramoto Oscillator Lattice  ODE (Euler / RK4)
// ============================================================
// References:
//   Sakaguchi, Shinomoto & Kuramoto (1987) [SSK]: d-dim lattice coupling
//   Daido (1988): generalised coupling, RG, lower critical dimension
//
// Model:
//   dφ_i/dt = ω_i + K Σ_{⟨ij⟩} sin(φ_j − φ_i)
//   ω_i ~ N(0, σ²)   (quenched disorder, fixed at t=0)
//   d-dimensional hypercubic lattice, periodic BC
//
// Observables (→ CSV):
//   r(t)      = |⟨e^{iφ}⟩|        global synchronisation order parameter
//   ω_eff,i   = d⟨φ_i⟩/dt          effective entrained frequency
//
// Cargo.toml dependencies:
//   rand       = { version = "0.8", features = ["small_rng"] }
//   rand_distr = "0.4"
//
// Output files:
//   kuramoto_timeseries.csv    step, t, r
//   kuramoto_omega_eff.csv     osc_index, omega_eff
//   kuramoto_coupling_scan.csv d, L, K, r_mean, r_std
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, Uniform};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

// ─────────────────────────────────────────────────────────────
// Section 1 — Lattice initialisation  (d-dimensional)
// ─────────────────────────────────────────────────────────────

/// Flat array of N = L^d phase values.
struct OscLattice {
    n:     usize,      // total oscillators = L^d
    l:     usize,      // linear size
    d:     usize,      // dimension
    phi:   Vec<f64>,   // phases  (NOT wrapped; unwrapped for ω_eff tracking)
    omega: Vec<f64>,   // quenched natural frequencies
    nb:    Vec<Vec<usize>>, // precomputed neighbour table
}

impl OscLattice {
    /// Build lattice, draw random φ ∈ [0,2π) and ω ~ N(0,σ²).
    fn new(l: usize, d: usize, sigma: f64, seed: u64) -> Self {
        let n          = l.pow(d as u32);
        let mut rng    = SmallRng::seed_from_u64(seed);
        let phi_dist   = Uniform::new(0.0_f64, 2.0 * PI);
        let omega_dist = Normal::new(0.0_f64, sigma).unwrap();
        let phi:   Vec<f64> = (0..n).map(|_| rng.sample(phi_dist)).collect();
        let omega: Vec<f64> = (0..n).map(|_| rng.sample(omega_dist)).collect();
        let nb = build_neighbour_table(l, d);
        OscLattice { n, l, d, phi, omega, nb }
    }
}

// ─────────────────────────────────────────────────────────────
// Section 2 — Neighbour table for L^d lattice (periodic BC)
// ─────────────────────────────────────────────────────────────

/// Precompute the list of 2*d neighbour flat-indices for every site.
///
/// Encoding: flat index = Σ_{axis=0}^{d-1}  coord[axis] * L^axis
/// (axis 0 is fastest-varying — same as the Python version)
fn build_neighbour_table(l: usize, d: usize) -> Vec<Vec<usize>> {
    let n = l.pow(d as u32);
    let mut nb = vec![Vec::with_capacity(2 * d); n];
    for flat_i in 0..n {
        // decode flat → multi-index
        let mut coords = vec![0_usize; d];
        let mut tmp    = flat_i;
        for ax in 0..d {
            coords[ax] = tmp % l;
            tmp /= l;
        }
        // ± 1 along each axis
        for ax in 0..d {
            for &delta in &[-1_i64, 1_i64] {
                let mut nb_coords = coords.clone();
                nb_coords[ax] = ((coords[ax] as i64 + delta).rem_euclid(l as i64)) as usize;
                // encode back to flat index
                let mut flat_j = 0_usize;
                let mut stride = 1_usize;
                for a in 0..d {
                    flat_j += nb_coords[a] * stride;
                    stride *= l;
                }
                nb[flat_i].push(flat_j);
            }
        }
    }
    nb
}

// ─────────────────────────────────────────────────────────────
// Section 3 — Right-hand side   dφ/dt
// ─────────────────────────────────────────────────────────────

/// dφ_i/dt = ω_i + K Σ_{j ∈ nb(i)} sin(φ_j − φ_i)
///
/// Exact translation of Python `dphi_dt`.
fn dphi_dt(phi: &[f64], omega: &[f64], k_coup: f64, nb: &[Vec<usize>]) -> Vec<f64> {
    let n = phi.len();
    let mut dph = omega.to_vec();
    for i in 0..n {
        for &j in &nb[i] {
            dph[i] += k_coup * (phi[j] - phi[i]).sin();
        }
    }
    dph
}

// ─────────────────────────────────────────────────────────────
// Section 4 — Time integrators: Euler and RK4
// ─────────────────────────────────────────────────────────────

/// Euler step: φ(t+dt) = φ(t) + dt * dφ/dt
fn euler_step(phi: &[f64], omega: &[f64], k_coup: f64, nb: &[Vec<usize>], dt: f64) -> Vec<f64> {
    let dph = dphi_dt(phi, omega, k_coup, nb);
    phi.iter().zip(&dph).map(|(p, d)| p + dt * d).collect()
}

/// RK4 step — more accurate than Euler for the same dt.
/// Mirrors Python `rk4_step` exactly.
fn rk4_step(phi: &[f64], omega: &[f64], k_coup: f64, nb: &[Vec<usize>], dt: f64) -> Vec<f64> {
    let n = phi.len();
    // helper: phi + s * k
    let add = |base: &[f64], k: &[f64], s: f64| -> Vec<f64> {
        base.iter().zip(k).map(|(b, ki)| b + s * ki).collect()
    };

    let k1 = dphi_dt(phi,                      omega, k_coup, nb);
    let k2 = dphi_dt(&add(phi, &k1, 0.5 * dt), omega, k_coup, nb);
    let k3 = dphi_dt(&add(phi, &k2, 0.5 * dt), omega, k_coup, nb);
    let k4 = dphi_dt(&add(phi, &k3, dt),        omega, k_coup, nb);

    (0..n).map(|i| phi[i] + (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])).collect()
}

// ─────────────────────────────────────────────────────────────
// Section 5 — Observables
// ─────────────────────────────────────────────────────────────

/// r = |⟨e^{iφ}⟩|  ∈ [0, 1]   (global synchronisation order parameter)
fn order_parameter(phi: &[f64]) -> f64 {
    let n  = phi.len() as f64;
    let re = phi.iter().map(|p| p.cos()).sum::<f64>() / n;
    let im = phi.iter().map(|p| p.sin()).sum::<f64>() / n;
    (re * re + im * im).sqrt()
}

/// ω_eff,i = (φ_i(t_end) − φ_i(t_start)) / (t_end − t_start)
/// Uses the raw (unwrapped) phase difference over the full measurement window.
fn effective_frequencies(phi_start: &[f64], phi_end: &[f64], elapsed: f64) -> Vec<f64> {
    phi_start.iter().zip(phi_end)
        .map(|(s, e)| (e - s) / elapsed)
        .collect()
}

fn mean_f(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
fn std_dev_f(v: &[f64]) -> f64 {
    let mu = mean_f(v);
    (v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

// ─────────────────────────────────────────────────────────────
// Section 6 — Main simulation runner
// ─────────────────────────────────────────────────────────────

/// Whether to use Euler or RK4 integration.
#[derive(Clone, Copy, PartialEq)]
enum Integrator { Euler, Rk4 }

/// Collected result of one `run_kuramoto` call.
struct KuramotoResult {
    l:         usize,
    d:         usize,
    k_coup:    f64,
    sigma:     f64,
    r_mean:    f64,
    r_std:     f64,
    omega_eff: Vec<f64>,
    r_ts:      Vec<f64>,  // r sampled every step during measurement window
    dt:        f64,
}

/// Run Kuramoto lattice ODE simulation.
///
/// Parameter correspondence with Python `run_kuramoto`:
///   l            ↔  L
///   d            ↔  d
///   k_coup       ↔  K
///   sigma        ↔  sigma
///   dt           ↔  dt
///   t_transient  ↔  t_transient
///   t_measure    ↔  t_measure
///   integrator   ↔  integrator  ('euler' or 'rk4')
///   seed         ↔  seed
fn run_kuramoto(
    l:           usize,
    d:           usize,
    k_coup:      f64,
    sigma:       f64,
    dt:          f64,
    t_transient: f64,
    t_measure:   f64,
    integrator:  Integrator,
    seed:        u64,
    verbose:     bool,
) -> KuramotoResult {
    let mut lat   = OscLattice::new(l, d, sigma, seed);
    let n_trans   = (t_transient / dt).round() as usize;
    let n_measure = (t_measure   / dt).round() as usize;
    let step_fn   = match integrator {
        Integrator::Euler => euler_step,
        Integrator::Rk4   => rk4_step,
    };

    // ── transient (discarded) ───────────────────────────────
    for _ in 0..n_trans {
        lat.phi = step_fn(&lat.phi, &lat.omega, k_coup, &lat.nb, dt);
    }
    if verbose {
        println!("[Kuramoto] L={l} d={d} K={k_coup:.2} σ={sigma:.2} \
                  — transient done ({n_trans} steps)");
    }

    // ── production: record r at every step ──────────────────
    let phi_start = lat.phi.clone();
    let mut r_ts  = Vec::with_capacity(n_measure);

    for _ in 0..n_measure {
        lat.phi = step_fn(&lat.phi, &lat.omega, k_coup, &lat.nb, dt);
        r_ts.push(order_parameter(&lat.phi));
    }

    let omega_eff = effective_frequencies(&phi_start, &lat.phi, t_measure);
    let r_mean    = mean_f(&r_ts);
    let r_std     = std_dev_f(&r_ts);

    if verbose {
        println!("           r = {r_mean:.4} ± {r_std:.4}");
    }

    KuramotoResult { l, d, k_coup, sigma, r_mean, r_std, omega_eff, r_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// Section 7 — Coupling-strength scan  (phase diagram)
// ─────────────────────────────────────────────────────────────

struct ScanRow {
    d:      usize,
    l:      usize,
    k_coup: f64,
    r_mean: f64,
    r_std:  f64,
}

/// Scan K values and collect ⟨r⟩ for fixed (L, d, σ).
/// Mirrors Python `coupling_scan`.
fn coupling_scan(
    l:           usize,
    d:           usize,
    k_arr:       &[f64],
    sigma:       f64,
    dt:          f64,
    t_transient: f64,
    t_measure:   f64,
    seed:        u64,
) -> Vec<ScanRow> {
    let mut rows = Vec::new();
    for &k in k_arr {
        let res = run_kuramoto(l, d, k, sigma, dt, t_transient, t_measure,
                               Integrator::Rk4, seed, false);
        println!("  d={d} L={l:3} K={k:.3}  r={:.4}", res.r_mean);
        rows.push(ScanRow { d, l, k_coup: k, r_mean: res.r_mean, r_std: res.r_std });
    }
    rows
}

// ─────────────────────────────────────────────────────────────
// Section 8 — CSV output helpers
// ─────────────────────────────────────────────────────────────

/// kuramoto_timeseries.csv: columns  step, t, r
fn write_timeseries(path: &str, r_ts: &[f64], dt: f64) {
    let mut w = BufWriter::new(File::create(path).expect("cannot create file"));
    writeln!(w, "step,t,r").unwrap();
    for (k, &r) in r_ts.iter().enumerate() {
        writeln!(w, "{},{:.6},{:.8}", k, k as f64 * dt, r).unwrap();
    }
    println!("Written: {path}");
}

/// kuramoto_omega_eff.csv: columns  osc_index, omega_eff
fn write_omega_eff(path: &str, omega_eff: &[f64]) {
    let mut w = BufWriter::new(File::create(path).expect("cannot create file"));
    writeln!(w, "osc_index,omega_eff").unwrap();
    for (i, &o) in omega_eff.iter().enumerate() {
        writeln!(w, "{},{:.8}", i, o).unwrap();
    }
    println!("Written: {path}");
}

/// kuramoto_coupling_scan.csv: columns  d, L, K, r_mean, r_std
fn write_coupling_scan(path: &str, rows: &[ScanRow]) {
    let mut w = BufWriter::new(File::create(path).expect("cannot create file"));
    writeln!(w, "d,L,K,r_mean,r_std").unwrap();
    for r in rows {
        writeln!(w, "{},{},{:.4},{:.6},{:.6}", r.d, r.l, r.k_coup, r.r_mean, r.r_std).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// Section 9 — Entry point  (mirrors Python __main__ block)
// ─────────────────────────────────────────────────────────────

fn main() {
    // ── Single run  (d=2, L=16, K=3.0) ─────────────────────
    let res = run_kuramoto(
        16,               // l
        2,                // d
        3.0,              // k_coup
        1.0,              // sigma
        0.05,             // dt
        50.0,             // t_transient
        100.0,            // t_measure
        Integrator::Rk4,  // integrator
        42,               // seed
        true,             // verbose
    );
    write_timeseries("kuramoto_timeseries.csv", &res.r_ts, res.dt);
    write_omega_eff("kuramoto_omega_eff.csv", &res.omega_eff);

    // ── Coupling scan  K ∈ [0.5, 5.0], d ∈ {2, 3} ──────────
    // 10 evenly-spaced K values, L=8 for speed (mirrors Python demo)
    let k_arr: Vec<f64> = (0..10).map(|k| 0.5 + k as f64 * (5.0 - 0.5) / 9.0).collect();
    let mut all_rows = Vec::new();
    for d in [2_usize, 3_usize] {
        let rows = coupling_scan(
            8,    // l
            d,    // d
            &k_arr,
            1.0,  // sigma
            0.05, // dt
            30.0, // t_transient
            50.0, // t_measure
            0,    // seed
        );
        all_rows.extend(rows);
    }
    write_coupling_scan("kuramoto_coupling_scan.csv", &all_rows);
}
