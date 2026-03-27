// ============================================================
// model_i1_kuramoto_extensional.rs  —  Model I1  🔴 P1
// Kuramoto Lattice + Extensional (Stretching) Flow
// ============================================================
// Model family : I族 — 剪切流/流场机制的变体与扩展
// Priority     : 🔴 P1  (LRO probability ≥ 70%)
//
// Physics motivation
// ------------------
//   Minami & Nakano (2022) proved analytically that extensional flow
//   (stretching along x, compression along y) stabilises LRO in O(N)
//   field theory via the same |k_x|^{-2/3} IR-suppression mechanism
//   as uniform shear.  Model I1 applies this to the Kuramoto phase
//   field with quenched frequency disorder, completing the A1/B1
//   series:
//     A1  : Kuramoto + shear,      Ω_i = Ω₀  (no disorder)
//     I1  : Kuramoto + extension,  ω_i ~ N(0,σ²)  (with disorder)
//
//   Key question: does extensional flow suppress the Kuramoto
//   lattice's inherent disorder-induced decoherence as effectively
//   as it does the equilibrium Goldstone modes?
//
// Equation of motion  (Euler-Maruyama SDE, a = 1)
// -----------------------------------------------
//   dθ_i = [ ω_i
//            + J Σ_{⟨ij⟩} sin(θ_j − θ_i)
//            + ε̇  x_i (θ_{i,j+1} − θ_{i,j-1}) / 2   [stretch ∂_x]
//            − ε̇  y_i (θ_{i+1,j} − θ_{i-1,j}) / 2   [compress ∂_y] ] dt
//          + √(2D dt) ξ_i
//
//   ω_i ~ N(0, σ²)  quenched;  ε̇ > 0 is the extension rate.
//   x_i = col index,  y_i = row index,  standard PBC.
//
// Observables (→ CSV)
// -------------------
//   r(t)  = |⟨e^{iθ}⟩|     synchronisation order parameter
//   C(r)  = spatial correlation (x-direction)
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   i1_timeseries.csv     step, t, r
//   i1_correlation.csv    r, C_r
//   i1_eps_scan.csv       L, sigma, eps_dot, r_mean, r_std, binder
//   i1_size_scan.csv      L, sigma, eps_dot, r_mean, r_std, binder
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

struct ExtKuraLattice {
    l:     usize,
    theta: Vec<f64>,
    omega: Vec<f64>,  // quenched frequencies ~ N(0, σ²)
}

impl ExtKuraLattice {
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
        ExtKuraLattice { l, theta, omega }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  Drift  f(θ)
// ─────────────────────────────────────────────────────────────

/// f_i = ω_i
///       + J Σ sin(θ_j − θ_i)
///       + ε̇  x_i (∂_x θ)_i    (x stretching)
///       − ε̇  y_i (∂_y θ)_i    (y compression)
///
/// Central differences:  ∂_x θ_i ≈ (θ_{i,j+1} − θ_{i,j-1})/2
///                        ∂_y θ_i ≈ (θ_{i+1,j} − θ_{i-1,j})/2
fn drift(lat: &ExtKuraLattice, j_coup: f64, eps: f64) -> Vec<f64> {
    let l   = lat.l;
    let mut f = lat.omega.clone();
    for i in 0..l {
        let yi = i as f64;
        for jj in 0..l {
            let xj = jj as f64;
            let k  = lat.idx(i, jj);
            let th = lat.theta[k];
            let up  = lat.get(lat.w(i, 1),  jj);
            let dn  = lat.get(lat.w(i,-1),  jj);
            let rt  = lat.get(i, lat.w(jj, 1));
            let lf  = lat.get(i, lat.w(jj,-1));
            // Kuramoto-XY coupling
            f[k] += j_coup * ((up-th).sin() + (dn-th).sin()
                             + (rt-th).sin() + (lf-th).sin());
            // extensional advection
            f[k] += eps * xj * (rt - lf) * 0.5;   // +ε̇ x ∂_x θ
            f[k] -= eps * yi * (up - dn) * 0.5;   // −ε̇ y ∂_y θ
        }
    }
    f
}

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut ExtKuraLattice, j_coup: f64, eps: f64,
           d: f64, dt: f64, rng: &mut SmallRng) {
    let f  = drift(lat, j_coup, eps);
    let sd = (2.0 * d * dt).sqrt();
    let nd = Normal::new(0.0f64, 1.0).unwrap();
    for k in 0..lat.l*lat.l {
        lat.theta[k] += f[k] * dt + sd * rng.sample(nd);
    }
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &ExtKuraLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &ExtKuraLattice, r_max: usize) -> Vec<f64> {
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

struct RunResult { l: usize, sigma: f64, eps: f64,
                   r_mean: f64, r_std: f64, binder: f64,
                   c_r: Vec<f64>, r_ts: Vec<f64>, dt: f64 }

fn run(l: usize, sigma: f64, j_coup: f64, eps: f64, d: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed + 3333);
    let mut lat = ExtKuraLattice::new(l, sigma, seed);
    let n_tr  = (t_trans / dt).round() as usize;
    let n_me  = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j_coup, eps, d, dt, &mut rng); }
    if verbose {
        println!("[I1] L={l} σ={sigma:.3} ε̇={eps:.3} J={j_coup} D={d:.2} \
                  transient={n_tr} steps done");
    }

    let mut r_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j_coup, eps, d, dt, &mut rng);
        if s % mevery == 0 { r_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let r_mean = mean_f(&r_ts);
    let r_std  = std_f(&r_ts);
    let bdr    = binder(&r_ts);
    if verbose { println!("      r = {r_mean:.4} ± {r_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, sigma, eps, r_mean, r_std, binder: bdr, c_r, r_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, sigma: f64, eps: f64,
              r_mean: f64, r_std: f64, binder: f64 }

fn eps_scan(l: usize, sigma: f64, eps_arr: &[f64], j_coup: f64, d: f64,
            dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    eps_arr.iter().map(|&eps| {
        let r = run(l, sigma, j_coup, eps, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} σ={sigma:.2} ε̇={eps:.3}  r={:.4}  U_L={:.4}",
                 r.r_mean, r.binder);
        SRow { l, sigma, eps, r_mean: r.r_mean, r_std: r.r_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], sigma: f64, j_coup: f64, eps: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, sigma, j_coup, eps, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} σ={sigma:.2} ε̇={eps:.3}  r={:.4}  U_L={:.4}",
                 r.r_mean, r.binder);
        SRow { l, sigma, eps, r_mean: r.r_mean, r_std: r.r_std, binder: r.binder }
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

fn write_corr(path: &str, c: &[f64]) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "r,C_r").unwrap();
    for (r,&v) in c.iter().enumerate() { writeln!(w, "{},{:.8}", r, v).unwrap(); }
    println!("Written: {path}");
}

fn write_scan(path: &str, rows: &[SRow]) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "L,sigma,eps_dot,r_mean,r_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.sigma, r.eps, r.r_mean, r.r_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: σ=0.1, ε̇=0.3, J=1, D=0.5
    let res = run(32, 0.1, 1.0, 0.3, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("i1_timeseries.csv", &res.r_ts, res.dt, me);
    write_corr("i1_correlation.csv", &res.c_r);

    // ε̇ scan at σ=0.1: compare with A1 (shear) and B1 (XY extension)
    let eps_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.1).collect();
    let er = eps_scan(32, 0.1, &eps_arr, 1.0, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan("i1_eps_scan.csv", &er);

    // finite-size scan at ε̇=0.3, σ=0.1
    let lr = size_scan(&[16,24,32,48], 0.1, 1.0, 0.3, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("i1_size_scan.csv", &lr);
}
