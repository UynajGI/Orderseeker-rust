// ============================================================
// model_e2_xy_nrec_disorder.rs  —  Model E2  🔴 P1
// Non-Reciprocal XY (Vision-Cone) + Quenched Frequency Disorder
// ============================================================
// Model family : E族 — 非互易相互作用的新变体
// Priority     : 🔴 P1  (LRO probability ≥ 70% for small σ)
//
// Physics motivation
// ------------------
//   Loos et al. (2023) showed vision-cone XY (Ω_i=0) has LRO in 2D.
//   Model E2 adds quenched natural frequencies ω_i ~ N(0,σ²) to test
//   robustness.  Two competing effects:
//     • Non-reciprocity (vision cone) → stabilises LRO
//     • Frequency disorder σ         → disrupts synchronisation
//   Expected: LRO survives for σ < σ*(J, α), yielding a phase boundary.
//
// Equation of motion  (Euler-Maruyama SDE)
// -----------------------------------------
//   dθ_i = [ ω_i  +  Σ_{j ∈ nb(i)} J_{ij} sin(θ_j − θ_i) ] dt
//          + √(2D dt) ξ_i
//
//   Vision-cone non-reciprocal coupling:
//     J_{ij} = J  if site j lies in the forward half-plane of i
//              0  otherwise
//
//   "Forward half-plane" is defined per-site by a fixed orientation:
//     Site i at row r, col c  →  forward = {right, up}  (fixed π-cone)
//     So J_{i→right} = J,  J_{i→up} = J,
//        J_{i→left}  = 0,  J_{i→down} = 0
//     While from the neighbour's perspective the couplings are reversed.
//
//   ω_i ~ N(0, σ²)  quenched (fixed at t=0).
//
//   This is a direct generalisation of Loos 2023 (σ=0) to σ>0.
//
// Observables (→ CSV)
// -------------------
//   m(t) = |⟨e^{iθ}⟩|    order parameter
//   C(r) = correlation
//   U_L  = Binder cumulant
//
// Output files
// ------------
//   e2_timeseries.csv   step, t, m
//   e2_correlation.csv  r, C_r
//   e2_sigma_scan.csv   L, sigma, m_mean, m_std, binder
//   e2_size_scan.csv    L, sigma, m_mean, m_std, binder
//
// Cargo.toml deps:  rand = "0.8" (features=["small_rng"])
//                   rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, Uniform};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

// ─────────────────────────────────────────────────────────────
// § 1  Lattice
// ─────────────────────────────────────────────────────────────

struct NRDisorderLattice {
    l:     usize,
    theta: Vec<f64>,
    omega: Vec<f64>,   // quenched frequencies ~ N(0,σ²)
}

impl NRDisorderLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i, j)] }

    fn new(l: usize, sigma: f64, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let ud  = Uniform::new(0.0f64, 2.0 * PI);
        let nd  = Normal::new(0.0f64, sigma).unwrap();
        let theta: Vec<f64> = (0..l*l).map(|_| rng.sample(ud)).collect();
        let omega: Vec<f64> = (0..l*l).map(|_| rng.sample(nd)).collect();
        NRDisorderLattice { l, theta, omega }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  Drift with vision-cone non-reciprocity
// ─────────────────────────────────────────────────────────────

/// Vision-cone rule: site i receives coupling from {right, up} only.
///   f_i = ω_i + J [sin(θ_{right} − θ_i) + sin(θ_{up} − θ_i)]
///
/// The coupling is unidirectional: j=right/up → i,  but NOT left/down → i.
/// This matches the half-plane vision cone (α = π cone, fixed orientation).
fn drift(lat: &NRDisorderLattice, j: f64) -> Vec<f64> {
    let l   = lat.l;
    let mut f = lat.omega.clone();
    for i in 0..l {
        for jj in 0..l {
            let k  = lat.idx(i, jj);
            let th = lat.theta[k];
            // only RIGHT and UP neighbours contribute (vision-cone)
            let rt = lat.get(i,         lat.w(jj, 1));  // right
            let up = lat.get(lat.w(i,1), jj           );  // up (row+1)
            f[k] += j * ((rt - th).sin() + (up - th).sin());
        }
    }
    f
}

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut NRDisorderLattice, j: f64, d: f64,
           dt: f64, rng: &mut SmallRng) {
    let f  = drift(lat, j);
    let sd = (2.0 * d * dt).sqrt();
    let nd = Normal::new(0.0f64, 1.0).unwrap();
    for k in 0..lat.l*lat.l {
        lat.theta[k] += f[k] * dt + sd * rng.sample(nd);
    }
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &NRDisorderLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &NRDisorderLattice, r_max: usize) -> Vec<f64> {
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

struct RunResult { l: usize, sigma: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, sigma: f64, j: f64, d: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed + 7777);
    let mut lat = NRDisorderLattice::new(l, sigma, seed);
    let n_tr  = (t_trans / dt).round() as usize;
    let n_me  = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j, d, dt, &mut rng); }
    if verbose {
        println!("[E2] L={l} σ={sigma:.3} J={j} D={d:.2} \
                  transient={n_tr} steps done");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, d, dt, &mut rng);
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, sigma, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, sigma: f64, m_mean: f64, m_std: f64, binder: f64 }

/// σ scan: find critical σ* above which LRO is destroyed.
fn sigma_scan(l: usize, sig_arr: &[f64], j: f64, d: f64,
              dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    sig_arr.iter().map(|&sig| {
        let r = run(l, sig, j, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} σ={sig:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, sigma: sig, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

/// Size scan at fixed σ: confirm LRO (m→const) vs QLRO/disorder (m→0).
fn size_scan(l_arr: &[usize], sigma: f64, j: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, sigma, j, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} σ={sigma:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, sigma, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

// ─────────────────────────────────────────────────────────────
// § 7  CSV helpers
// ─────────────────────────────────────────────────────────────

fn write_ts(path: &str, m_ts: &[f64], dt: f64, me: usize) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "step,t,m").unwrap();
    for (k,&m) in m_ts.iter().enumerate() {
        let s = k*me; writeln!(w, "{},{:.6},{:.8}", s, s as f64*dt, m).unwrap();
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
    writeln!(w, "L,sigma,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.sigma, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: σ=0.1 (small disorder), J=1, D=0.2
    let res = run(32, 0.1, 1.0, 0.2, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("e2_timeseries.csv", &res.m_ts, res.dt, me);
    write_corr("e2_correlation.csv", &res.c_r);

    // σ scan: σ=0 (Loos 2023 baseline, expect LRO) → σ=1.5 (strong disorder)
    let sig_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.15).collect();
    let sr = sigma_scan(32, &sig_arr, 1.0, 0.2, 0.01, 15.0, 60.0, 0);
    write_scan("e2_sigma_scan.csv", &sr);

    // finite-size scan at σ=0.1
    let lr = size_scan(&[16,24,32,48], 0.1, 1.0, 0.2, 0.01, 15.0, 60.0, 1);
    write_scan("e2_size_scan.csv", &lr);
}
