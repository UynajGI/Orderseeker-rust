// ============================================================
// model_e1_xy_nrec_nnn.rs  —  Model E1  🔴 P1
// XY Model: Reciprocal NN + Unidirectional NNN Coupling
// ============================================================
// Model family : E族 — 非互易相互作用的新变体
// Priority     : 🔴 P1  (LRO probability ≥ 70%)
//
// Physics motivation
// ------------------
//   Loos et al. (2023) proved that non-reciprocal XY (vision-cone
//   coupling) produces LRO in 2D.  Model E1 asks: can we obtain
//   the same effect by keeping NN coupling fully reciprocal but
//   introducing non-reciprocity only at next-nearest-neighbour (NNN)
//   distance?  Non-reciprocity at longer range is physically natural
//   in flocking / biological contexts.
//
//   Non-reciprocal mechanism: net probability current around loops
//   injects energy at a rate proportional to the entropy production.
//   Moving the non-reciprocity to NNN while keeping NN symmetric
//   tests whether the range of non-reciprocity matters.
//
// Hamiltonian / dynamics (Monte Carlo, Glauber-like Metropolis)
// -------------------------------------------------------------
//   Effective energy for site i:
//     H_i = −J  Σ_{j ∈ NN(i)}  cos(θ_i − θ_j)          [reciprocal]
//           − J' Σ_{j ∈ NNN_fwd(i)} cos(θ_i − θ_j)      [one-way forward]
//
//   NNN_fwd(i) = { i+2x̂, i+2ŷ }  (unidirectional: from i to i+2)
//   NNN_bwd(i) = { i-2x̂, i-2ŷ }  receives no coupling from i → asymmetry.
//
//   Monte Carlo update (Metropolis):
//     Propose θ_i → θ_i + δ,  δ ~ Uniform(−Δ, +Δ)
//     Compute ΔE = −(new H_i − old H_i)  for site i ONLY
//     Accept with min(1, exp(−β ΔE))
//
//   Note: because H_ij ≠ H_ji (non-reciprocal), the full system
//   Hamiltonian is ill-defined, but the LOCAL energy for site i
//   drives the Metropolis step exactly as in Loos et al.
//
// Observables (→ CSV)
// -------------------
//   m      = |⟨e^{iθ}⟩|       magnetisation (order parameter)
//   C(r)   = spatial correlation
//   U_L    = Binder cumulant
//
// Output files
// ------------
//   e1_timeseries.csv   sweep, m
//   e1_correlation.csv  r, C_r
//   e1_jprime_scan.csv  L, J_prime, m_mean, m_std, binder
//   e1_size_scan.csv    L, J_prime, m_mean, m_std, binder
//
// Cargo.toml deps:  rand = "0.8" (features=["small_rng"])
//                   rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Uniform;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

// ─────────────────────────────────────────────────────────────
// § 1  Lattice
// ─────────────────────────────────────────────────────────────

struct NRLattice {
    l:     usize,
    theta: Vec<f64>,
}

impl NRLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i, j)] }
    #[inline] fn set(&mut self, i: usize, j: usize, v: f64) {
        let k = self.idx(i,j); self.theta[k] = v;
    }

    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let d = Uniform::new(0.0f64, 2.0 * PI);
        NRLattice { l, theta: (0..l*l).map(|_| rng.sample(d)).collect() }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  Local energy for site (i,j)
// ─────────────────────────────────────────────────────────────

/// Local energy for site (i,j):
///   E_i = −J  [cos(θ_i−θ_up) + cos(θ_i−θ_dn) + cos(θ_i−θ_rt) + cos(θ_i−θ_lf)]
///         −J' [cos(θ_i−θ_{i,j+2}) + cos(θ_i−θ_{i+2,j})]   (forward NNN only)
///
/// Note: the backward NNN {i,j-2} and {i-2,j} do NOT appear in E_i
/// (those sites exert no force on i), creating the non-reciprocity.
fn local_energy(lat: &NRLattice, i: usize, j: usize, j_nn: f64, j_nnn: f64) -> f64 {
    let l  = lat.l;
    let th = lat.get(i, j);
    // NN (reciprocal)
    let e_nn = j_nn * (
          (th - lat.get(lat.w(i, 1),  j)).cos()
        + (th - lat.get(lat.w(i,-1),  j)).cos()
        + (th - lat.get(i, lat.w(j, 1))).cos()
        + (th - lat.get(i, lat.w(j,-1))).cos()
    );
    // NNN forward only (non-reciprocal)
    let e_nnn = j_nnn * (
          (th - lat.get(i,         lat.w(j, 2))).cos()  // x forward
        + (th - lat.get(lat.w(i, 2), j)).cos()          // y forward
    );
    -(e_nn + e_nnn)
}

// ─────────────────────────────────────────────────────────────
// § 3  Metropolis sweep
// ─────────────────────────────────────────────────────────────

fn mc_sweep(lat: &mut NRLattice, beta: f64, j_nn: f64, j_nnn: f64,
            delta: f64, rng: &mut SmallRng) {
    let l    = lat.l;
    let n    = l * l;
    let id   = Uniform::new(0usize, l);
    let dd   = Uniform::new(-delta, delta);
    let ud   = Uniform::new(0.0f64, 1.0);

    for _ in 0..n {
        let i   = rng.sample(id);
        let j   = rng.sample(id);
        let old = lat.get(i, j);
        let e_old = local_energy(lat, i, j, j_nn, j_nnn);
        let new_  = (old + rng.sample(dd)).rem_euclid(2.0 * PI);
        lat.set(i, j, new_);
        let e_new = local_energy(lat, i, j, j_nn, j_nnn);
        let de    = e_new - e_old;
        if de > 0.0 && rng.sample(ud) >= (-beta * de).exp() {
            lat.set(i, j, old); // reject
        }
    }
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &NRLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &NRLattice, r_max: usize) -> Vec<f64> {
    let l = lat.l;
    let n = (l * l) as f64;
    let mut c = vec![0.0f64; r_max + 1];
    c[0] = 1.0;
    for r in 1..=r_max {
        c[r] = (0..l).flat_map(|i| (0..l).map(move |j| (i,j)))
            .map(|(i,j)| (lat.get(i,j) - lat.get(i,(j+r)%l)).cos())
            .sum::<f64>() / n;
    }
    c
}

fn binder(ms: &[f64]) -> f64 {
    let n  = ms.len() as f64;
    let m2 = ms.iter().map(|m| m*m).sum::<f64>() / n;
    let m4 = ms.iter().map(|m| m.powi(4)).sum::<f64>() / n;
    if m2 < 1e-15 { return 0.0; }
    1.0 - m4 / (3.0 * m2 * m2)
}

fn mean_f(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
fn std_f(v: &[f64]) -> f64 {
    let mu = mean_f(v);
    (v.iter().map(|x| (x-mu).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

// ─────────────────────────────────────────────────────────────
// § 5  Runner
// ─────────────────────────────────────────────────────────────

struct RunResult { l: usize, j_nnn: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64> }

fn run(l: usize, temp: f64, j_nn: f64, j_nnn: f64,
       n_therm: usize, n_meas: usize, mevery: usize,
       seed: u64, verbose: bool) -> RunResult {
    let beta    = 1.0 / temp;
    let delta   = PI;
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = NRLattice::new_random(l, &mut rng);

    for _ in 0..n_therm { mc_sweep(&mut lat, beta, j_nn, j_nnn, delta, &mut rng); }
    if verbose {
        println!("[E1] L={l} T={temp:.2} J={j_nn} J'={j_nnn:.3} \
                  therm={n_therm} sweeps done");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_meas {
        mc_sweep(&mut lat, beta, j_nn, j_nnn, delta, &mut rng);
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, j_nnn, m_mean, m_std, binder: bdr, c_r, m_ts }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, j_nnn: f64, m_mean: f64, m_std: f64, binder: f64 }

/// Scan J' (NNN non-reciprocal coupling strength).
/// J'=0: symmetric NNN → equilibrium-like, no LRO expected.
/// J'>0: non-reciprocal NNN → test LRO onset.
fn jprime_scan(l: usize, jp_arr: &[f64], temp: f64, j_nn: f64,
               n_therm: usize, n_meas: usize, seed: u64) -> Vec<SRow> {
    jp_arr.iter().map(|&jp| {
        let r = run(l, temp, j_nn, jp, n_therm, n_meas, 10, seed, false);
        println!("  L={l} J'={jp:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, j_nnn: jp, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], temp: f64, j_nn: f64, j_nnn: f64,
             n_therm: usize, n_meas: usize, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, temp, j_nn, j_nnn, n_therm, n_meas, 10, seed, false);
        println!("  L={l} J'={j_nnn:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, j_nnn, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

// ─────────────────────────────────────────────────────────────
// § 7  CSV helpers
// ─────────────────────────────────────────────────────────────

fn write_ts(path: &str, m_ts: &[f64], mevery: usize) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "sweep,m").unwrap();
    for (k,&m) in m_ts.iter().enumerate() {
        writeln!(w, "{},{:.8}", k*mevery, m).unwrap();
    }
    println!("Written: {path}");
}

fn write_corr(path: &str, c: &[f64]) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "r,C_r").unwrap();
    for (r,&v) in c.iter().enumerate() { writeln!(w, "{},{:.8}", r, v).unwrap(); }
    println!("Written: {path}");
}

fn write_scan(path: &str, rows: &[SRow]) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "L,J_prime,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.j_nnn, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let mevery = 10usize;
    // single run: L=32, T=0.5, J=1, J'=0.5
    let res = run(32, 0.5, 1.0, 0.5, 2000, 5000, mevery, 42, true);
    write_ts("e1_timeseries.csv", &res.m_ts, mevery);
    write_corr("e1_correlation.csv", &res.c_r);

    // J' scan: J'=0 (no NNN) → J'=1.0 (strong NNN non-reciprocity)
    let jp_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.1).collect();
    let jr = jprime_scan(32, &jp_arr, 0.5, 1.0, 1000, 3000, 0);
    write_scan("e1_jprime_scan.csv", &jr);

    // finite-size scan at J'=0.5
    let lr = size_scan(&[16,24,32,48], 0.5, 1.0, 0.5, 1500, 3000, 1);
    write_scan("e1_size_scan.csv", &lr);
}
