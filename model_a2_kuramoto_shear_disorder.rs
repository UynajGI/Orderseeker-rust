// ============================================================
// model_a2_kuramoto_shear_disorder.rs  —  Model A2  🔴 P1
// Kuramoto Lattice + Shear Flow + Gaussian Frequency Disorder
// ============================================================
// Model family : A族 — Kuramoto振子格子 + 非平衡驱动
// Priority     : 🔴 P1  (LRO probability: medium-high)
//
// Physics motivation
// ------------------
//   Extension of A1: natural frequencies ω_i ~ N(0, σ²) (quenched).
//   Tests robustness of shear-induced LRO against frequency disorder.
//   For small σ ≪ J, shear should still dominate; a critical σ*(γ̇)
//   separates LRO from disorder.  This competition is unexplored.
//
// Equation of motion  (Euler-Maruyama SDE)
// -----------------------------------------
//   dθ_i = [ ω_i
//            + J  Σ_{⟨ij⟩} sin(θ_j − θ_i)
//            + γ̇  y_i  (θ_{i,j+1} − θ_{i,j-1}) / 2   ] dt
//          + √(2D dt) ξ_i
//
//   ω_i ~ N(0, σ²)   quenched, fixed at t=0
//   All other terms identical to A1.
//
// Key difference from A1
// ----------------------
//   • Each site has its own quenched frequency ω_i (drawn once).
//   • Extra parameter σ (disorder width).
//   • Scan over (γ̇, σ) to map out the phase boundary.
//
// Observables (→ CSV)
// -------------------
//   r(t)   = |⟨e^{iθ}⟩|     order parameter
//   C(r)   = spatial correlation along x
//   U_L    = Binder cumulant
//
// Output files
// ------------
//   a2_timeseries.csv    step, t, r
//   a2_correlation.csv   r, C_r
//   a2_sigma_scan.csv    L, sigma, gamma_dot, r_mean, r_std, binder
//   a2_shear_scan.csv    L, gamma_dot, sigma, r_mean, r_std, binder
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
// § 1  Lattice  (phases + quenched frequencies)
// ─────────────────────────────────────────────────────────────

struct ShearDisorderLattice {
    l:     usize,
    theta: Vec<f64>,  // phases (real line)
    omega: Vec<f64>,  // quenched natural frequencies ~ N(0, σ²)
}

impl ShearDisorderLattice {
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
        ShearDisorderLattice { l, theta, omega }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  Drift  f(θ)
// ─────────────────────────────────────────────────────────────

fn drift(lat: &ShearDisorderLattice, j: f64, gd: f64) -> Vec<f64> {
    let l   = lat.l;
    let mut f = lat.omega.clone();  // starts from ω_i
    for i in 0..l {
        let yi = i as f64;
        for jj in 0..l {
            let k  = lat.idx(i, jj);
            let th = lat.theta[k];
            let up = lat.get(lat.w(i, 1),  jj);
            let dn = lat.get(lat.w(i,-1),  jj);
            let rt = lat.get(i, lat.w(jj, 1));
            let lf = lat.get(i, lat.w(jj,-1));
            // XY coupling
            f[k] += j * ((up-th).sin() + (dn-th).sin()
                        + (rt-th).sin() + (lf-th).sin());
            // shear advection
            f[k] += gd * yi * (rt - lf) * 0.5;
        }
    }
    f
}

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut ShearDisorderLattice, j: f64, gd: f64,
           d: f64, dt: f64, rng: &mut SmallRng) {
    let f  = drift(lat, j, gd);
    let sd = (2.0 * d * dt).sqrt();
    let nd = Normal::new(0.0f64, 1.0).unwrap();
    for k in 0..lat.l*lat.l {
        lat.theta[k] += f[k] * dt + sd * rng.sample(nd);
    }
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &ShearDisorderLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &ShearDisorderLattice, r_max: usize) -> Vec<f64> {
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

fn binder(rs: &[f64]) -> f64 {
    let n  = rs.len() as f64;
    let r2 = rs.iter().map(|r| r*r).sum::<f64>() / n;
    let r4 = rs.iter().map(|r| r.powi(4)).sum::<f64>() / n;
    if r2 < 1e-15 { return 0.0; }
    1.0 - r4 / (3.0 * r2 * r2)
}

fn mean_f(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
fn std_f(v: &[f64]) -> f64 {
    let mu = mean_f(v);
    (v.iter().map(|x| (x-mu).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

// ─────────────────────────────────────────────────────────────
// § 5  Runner
// ─────────────────────────────────────────────────────────────

struct RunResult { l: usize, sigma: f64, gd: f64,
                   r_mean: f64, r_std: f64, binder: f64,
                   c_r: Vec<f64>, r_ts: Vec<f64>, dt: f64 }

fn run(l: usize, sigma: f64, j: f64, gd: f64, d: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed + 1000); // separate RNG for dynamics
    let mut lat = ShearDisorderLattice::new(l, sigma, seed);
    let n_tr  = (t_trans / dt).round() as usize;
    let n_me  = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j, gd, d, dt, &mut rng); }
    if verbose {
        println!("[A2] L={l} σ={sigma:.3} γ̇={gd:.3} J={j} D={d:.2} \
                  transient={n_tr} steps done");
    }

    let mut r_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, gd, d, dt, &mut rng);
        if s % mevery == 0 { r_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let r_mean = mean_f(&r_ts);
    let r_std  = std_f(&r_ts);
    let bdr    = binder(&r_ts);
    if verbose { println!("      r = {r_mean:.4} ± {r_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, sigma, gd, r_mean, r_std, binder: bdr, c_r, r_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, sigma: f64, gd: f64,
              r_mean: f64, r_std: f64, binder: f64 }

/// Scan disorder width σ at fixed γ̇ — find critical σ*(γ̇).
fn sigma_scan(l: usize, sigma_arr: &[f64], j: f64, gd: f64, d: f64,
              dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    sigma_arr.iter().map(|&sig| {
        let r = run(l, sig, j, gd, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} σ={sig:.3} γ̇={gd:.2}  r={:.4}  U_L={:.4}",
                 r.r_mean, r.binder);
        SRow { l, sigma: sig, gd, r_mean: r.r_mean, r_std: r.r_std, binder: r.binder }
    }).collect()
}

/// Scan shear rate γ̇ at fixed σ — compare to A1 (σ=0).
fn shear_scan(l: usize, gd_arr: &[f64], sigma: f64, j: f64, d: f64,
              dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    gd_arr.iter().map(|&gd| {
        let r = run(l, sigma, j, gd, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} γ̇={gd:.3} σ={sigma:.2}  r={:.4}  U_L={:.4}",
                 r.r_mean, r.binder);
        SRow { l, sigma, gd, r_mean: r.r_mean, r_std: r.r_std, binder: r.binder }
    }).collect()
}

// ─────────────────────────────────────────────────────────────
// § 7  CSV helpers
// ─────────────────────────────────────────────────────────────

fn write_ts(path: &str, r_ts: &[f64], dt: f64, me: usize) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "step,t,r").unwrap();
    for (k,&r) in r_ts.iter().enumerate() {
        let s = k*me; writeln!(w, "{},{:.6},{:.8}", s, s as f64*dt, r).unwrap();
    }
    println!("Written: {path}");
}

fn write_corr(path: &str, c_r: &[f64]) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "r,C_r").unwrap();
    for (r,&c) in c_r.iter().enumerate() { writeln!(w, "{},{:.8}", r, c).unwrap(); }
    println!("Written: {path}");
}

fn write_scan_sigma(path: &str, rows: &[SRow]) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "L,sigma,gamma_dot,r_mean,r_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.4},{:.6},{:.6},{:.6}",
                 r.l,r.sigma,r.gd,r.r_mean,r.r_std,r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: σ=0.1 (small disorder), γ̇=0.5
    let res = run(32, 0.1, 1.0, 0.5, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("a2_timeseries.csv", &res.r_ts, res.dt, me);
    write_corr("a2_correlation.csv", &res.c_r);

    // σ scan at γ̇=0.5: find critical disorder σ*
    let sigma_arr: Vec<f64> = (0..=8).map(|k| k as f64 * 0.15).collect();
    let sr = sigma_scan(32, &sigma_arr, 1.0, 0.5, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan_sigma("a2_sigma_scan.csv", &sr);

    // γ̇ scan at σ=0.1: compare with A1 (σ=0) baseline
    let gd_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.2).collect();
    let gr = shear_scan(32, &gd_arr, 0.1, 1.0, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan_sigma("a2_shear_scan.csv", &gr);
}
