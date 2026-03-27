// ============================================================
// model_a1_kuramoto_shear.rs  —  Model A1  🔴 P1
// Kuramoto Lattice + Uniform Shear Flow  (Ω₀ = const, no disorder)
// ============================================================
// Model family : A族 — Kuramoto振子格子 + 非平衡驱动
// Priority     : 🔴 P1  (LRO probability ≥ 70%)
//
// Physics motivation
// ------------------
//   Standard 2D Kuramoto lattice → no LRO (Daido 1988, r → 0).
//   Uniform shear flow v = (γ̇ y, 0) adds an advection term that
//   suppresses Goldstone-mode IR divergences via the |k_x|^{-2/3}
//   mechanism (Nakano & Sasa 2021).  All oscillators share the same
//   Ω₀ (quenched disorder removed) so only shear vs thermal noise
//   compete.  Statistical O(2) symmetry is preserved.
//
// Equation of motion  (Euler-Maruyama SDE, lattice spacing a = 1)
// ---------------------------------------------------------------
//   dθ_i = [ Ω₀
//            + J  Σ_{⟨ij⟩} sin(θ_j − θ_i)
//            + γ̇  y_i  (θ_{i,j+1} − θ_{i,j-1}) / 2   ] dt
//          + √(2D dt) ξ_i
//
//   i = (row, col),  y_i = row index,  PBC in both directions.
//   The shear advection uses a central-difference ∂_x θ.
//
// Observables (→ CSV)
// -------------------
//   r(t)  = |⟨e^{iθ}⟩|               order parameter
//   C(r)  = ⟨cos(θ_{0}−θ_{r})⟩_x    spatial correlation along x
//   U_L   = 1 − ⟨r⁴⟩/(3⟨r²⟩²)       Binder cumulant
//
// Output files
// ------------
//   a1_timeseries.csv   step, t, r
//   a1_correlation.csv  r, C_r
//   a1_shear_scan.csv   L, gamma_dot, r_mean, r_std, binder
//   a1_size_scan.csv    L, gamma_dot, r_mean, r_std, binder
//
// Cargo.toml deps:  rand = "0.8"  (features=["small_rng"])
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

struct ShearLattice {
    l:     usize,
    theta: Vec<f64>,  // phases on the real line (not wrapped)
}

impl ShearLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i, j)] }

    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let d = Uniform::new(0.0f64, 2.0 * PI);
        ShearLattice { l, theta: (0..l*l).map(|_| rng.sample(d)).collect() }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  Drift  f(θ)
// ─────────────────────────────────────────────────────────────

fn drift(lat: &ShearLattice, omega0: f64, j: f64, gd: f64) -> Vec<f64> {
    let l = lat.l;
    let mut f = vec![omega0; l * l];
    for i in 0..l {
        let yi = i as f64;
        for jj in 0..l {
            let k     = lat.idx(i, jj);
            let th    = lat.theta[k];
            let up    = lat.get(lat.w(i, 1),  jj);
            let dn    = lat.get(lat.w(i,-1),  jj);
            let rt    = lat.get(i, lat.w(jj, 1));
            let lf    = lat.get(i, lat.w(jj,-1));
            // XY coupling
            f[k] += j * ((up-th).sin() + (dn-th).sin()
                        + (rt-th).sin() + (lf-th).sin());
            // shear advection  γ̇ y_i  ∂_x θ
            f[k] += gd * yi * (rt - lf) * 0.5;
        }
    }
    f
}

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut ShearLattice, omega0: f64, j: f64, gd: f64,
           d: f64, dt: f64, rng: &mut SmallRng) {
    let f   = drift(lat, omega0, j, gd);
    let sd  = (2.0 * d * dt).sqrt();
    let nd  = Normal::new(0.0f64, 1.0).unwrap();
    for k in 0..lat.l*lat.l {
        lat.theta[k] += f[k] * dt + sd * rng.sample(nd);
    }
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &ShearLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &ShearLattice, r_max: usize) -> Vec<f64> {
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
    (v.iter().map(|x|(x-mu).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

// ─────────────────────────────────────────────────────────────
// § 5  Runner
// ─────────────────────────────────────────────────────────────

struct RunResult { l: usize, gd: f64, r_mean: f64, r_std: f64,
                   binder: f64, c_r: Vec<f64>, r_ts: Vec<f64>, dt: f64 }

fn run(l: usize, omega0: f64, j: f64, gd: f64, d: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = ShearLattice::new_random(l, &mut rng);
    let n_tr  = (t_trans / dt).round() as usize;
    let n_me  = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, omega0, j, gd, d, dt, &mut rng); }
    if verbose {
        println!("[A1] L={l} γ̇={gd:.3} J={j} D={d:.2} Ω₀={omega0} \
                  transient={n_tr} steps done");
    }

    let mut r_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, omega0, j, gd, d, dt, &mut rng);
        if s % mevery == 0 { r_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let r_mean = mean_f(&r_ts);
    let r_std  = std_f(&r_ts);
    let bdr    = binder(&r_ts);
    if verbose { println!("      r = {r_mean:.4} ± {r_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, gd, r_mean, r_std, binder: bdr, c_r, r_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, gd: f64, r_mean: f64, r_std: f64, binder: f64 }

fn shear_scan(l: usize, gd_arr: &[f64], omega0: f64, j: f64, d: f64,
              dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    gd_arr.iter().map(|&gd| {
        let r = run(l, omega0, j, gd, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} γ̇={gd:.3}  r={:.4}  U_L={:.4}", r.r_mean, r.binder);
        SRow { l, gd, r_mean: r.r_mean, r_std: r.r_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], omega0: f64, j: f64, gd: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, omega0, j, gd, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} γ̇={gd:.3}  r={:.4}  U_L={:.4}", r.r_mean, r.binder);
        SRow { l, gd, r_mean: r.r_mean, r_std: r.r_std, binder: r.binder }
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

fn write_scan(path: &str, rows: &[SRow], col: &str) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "L,{col},r_mean,r_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.6},{:.6},{:.6},{:.6}", r.l,r.gd,r.r_mean,r.r_std,r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: L=32, γ̇=0.5, J=1, D=0.5, Ω₀=0
    let res = run(32, 0.0, 1.0, 0.5, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("a1_timeseries.csv", &res.r_ts, res.dt, me);
    write_corr("a1_correlation.csv", &res.c_r);

    // shear scan γ̇ ∈ [0, 2.0]  (0 = baseline, expect r≈0 for 2D Kuramoto)
    let gd_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.2).collect();
    let sr = shear_scan(32, &gd_arr, 0.0, 1.0, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan("a1_shear_scan.csv", &sr, "gamma_dot");

    // finite-size scan at γ̇=0.5  (LRO: r→const; disorder: r→0 as L→∞)
    let lr = size_scan(&[16,24,32,48], 0.0, 1.0, 0.5, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("a1_size_scan.csv", &lr, "gamma_dot");
}
