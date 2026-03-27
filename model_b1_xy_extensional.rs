// ============================================================
// model_b1_xy_extensional.rs  —  Model B1  🔴 P1
// XY Model + Uniform Extensional (Stretching) Flow
// ============================================================
// Model family : B族 — XY模型 / O(2)场 + 新型非平衡驱动
// Priority     : 🔴 P1  (LRO probability ≥ 70%)
//
// Physics motivation
// ------------------
//   Minami & Nakano (2022) proved analytically that O(N) field theory
//   under extensional (stretching) flow also develops LRO via the same
//   |k_x|^{-2/3} suppression as shear.  The extensional flow
//   v_x = ε̇ x,  v_y = −ε̇ y   is incompressible (∇·v = 0) and
//   compatible with PBC when implemented with careful finite-difference
//   advection.  No lattice numerical study exists yet.
//
// Equation of motion  (Euler-Maruyama SDE, lattice spacing a = 1)
// ---------------------------------------------------------------
//   dθ_i = [ J Σ_{⟨ij⟩} sin(θ_j − θ_i)
//            + ε̇ x_i (θ_{i+1,j} − θ_{i-1,j}) / 2   [∂_x term]
//            − ε̇ y_i (θ_{i,j+1} − θ_{i,j-1}) / 2   [∂_y term] ] dt
//          + √(2D dt) ξ_i
//
//   x_i = col index  (fast axis),  y_i = row index  (slow axis)
//   ε̇  > 0 : stretching rate along x, compression along y
//   PBC: standard for both coupling and advection differences.
//
//   Note: extensional flow breaks the rotational symmetry of the
//   FLOW FIELD (x and y treated differently), but the XY coupling
//   retains O(2) symmetry; statistical symmetry is preserved in
//   ensemble average over initial conditions.
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨e^{iθ}⟩|     magnetisation / order parameter
//   C(r)  = ⟨cos(θ_i − θ_{i+r x̂})⟩   correlation along x
//   C_y(r)= ⟨cos(θ_i − θ_{i+r ŷ})⟩   correlation along y
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   b1_timeseries.csv      step, t, m
//   b1_correlation_x.csv   r, C_r
//   b1_correlation_y.csv   r, C_r
//   b1_eps_scan.csv        L, eps_dot, m_mean, m_std, binder
//   b1_size_scan.csv       L, eps_dot, m_mean, m_std, binder
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

struct ExtLattice {
    l:     usize,
    theta: Vec<f64>,  // phases (real line)
}

impl ExtLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i, j)] }

    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let d = Uniform::new(0.0f64, 2.0 * PI);
        ExtLattice { l, theta: (0..l*l).map(|_| rng.sample(d)).collect() }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  Drift  f(θ)
// ─────────────────────────────────────────────────────────────

/// f_i = J Σ sin(θ_j − θ_i)
///       + ε̇ x_i (θ_{i,j+1} − θ_{i,j-1})/2    (stretching along x)
///       − ε̇ y_i (θ_{i+1,j} − θ_{i-1,j})/2    (compression along y)
///
/// Coordinate convention:
///   row index i  → y-coordinate  (y_i = i)
///   col index j  → x-coordinate  (x_j = j)
fn drift(lat: &ExtLattice, j_coup: f64, eps: f64) -> Vec<f64> {
    let l   = lat.l;
    let mut f = vec![0.0f64; l * l];
    for i in 0..l {
        let yi = i as f64;  // y-coordinate
        for jj in 0..l {
            let xj = jj as f64; // x-coordinate
            let k  = lat.idx(i, jj);
            let th = lat.theta[k];
            let up  = lat.get(lat.w(i, 1),  jj);  // row i+1
            let dn  = lat.get(lat.w(i,-1),  jj);  // row i-1
            let rt  = lat.get(i, lat.w(jj, 1));   // col j+1
            let lf  = lat.get(i, lat.w(jj,-1));   // col j-1
            // XY coupling
            f[k] = j_coup * ((up-th).sin() + (dn-th).sin()
                            + (rt-th).sin() + (lf-th).sin());
            // extensional flow:  ε̇ x ∂_x θ  −  ε̇ y ∂_y θ
            f[k] += eps * xj * (rt - lf) * 0.5;   // +ε̇ x ∂_x
            f[k] -= eps * yi * (up - dn) * 0.5;   // −ε̇ y ∂_y
        }
    }
    f
}

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut ExtLattice, j_coup: f64, eps: f64,
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

fn order_param(lat: &ExtLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

/// Spatial correlation along the x-direction (col shift).
fn corr_x(lat: &ExtLattice, r_max: usize) -> Vec<f64> {
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

/// Spatial correlation along the y-direction (row shift).
fn corr_y(lat: &ExtLattice, r_max: usize) -> Vec<f64> {
    let l = lat.l;
    let n = (l * l) as f64;
    let mut c = vec![0.0f64; r_max + 1];
    c[0] = 1.0;
    for r in 1..=r_max {
        c[r] = (0..l).flat_map(|i| (0..l).map(move |j| (i,j)))
            .map(|(i,j)| (lat.get(i,j) - lat.get((i+r)%l,j)).cos())
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

struct RunResult { l: usize, eps: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   cx: Vec<f64>, cy: Vec<f64>,
                   m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j_coup: f64, eps: f64, d: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = ExtLattice::new_random(l, &mut rng);
    let n_tr  = (t_trans / dt).round() as usize;
    let n_me  = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j_coup, eps, d, dt, &mut rng); }
    if verbose {
        println!("[B1] L={l} ε̇={eps:.3} J={j_coup} D={d:.2} \
                  transient={n_tr} steps done");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j_coup, eps, d, dt, &mut rng);
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let cx     = corr_x(&lat, l/2);
    let cy     = corr_y(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, eps, m_mean, m_std, binder: bdr, cx, cy, m_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, eps: f64, m_mean: f64, m_std: f64, binder: f64 }

fn eps_scan(l: usize, eps_arr: &[f64], j_coup: f64, d: f64,
            dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    eps_arr.iter().map(|&eps| {
        let r = run(l, j_coup, eps, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} ε̇={eps:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, eps, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j_coup: f64, eps: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j_coup, eps, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} ε̇={eps:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, eps, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
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
    writeln!(w, "L,eps_dot,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l,r.eps,r.m_mean,r.m_std,r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: L=32, ε̇=0.3, J=1, D=0.5
    let res = run(32, 1.0, 0.3, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("b1_timeseries.csv", &res.m_ts, res.dt, me);
    write_corr("b1_correlation_x.csv", &res.cx);
    write_corr("b1_correlation_y.csv", &res.cy);

    // ε̇ scan  ε̇ ∈ [0, 1.0]
    // ε̇=0 : equilibrium XY (no LRO in 2D); ε̇>0 : test LRO onset
    let eps_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.1).collect();
    let er = eps_scan(32, &eps_arr, 1.0, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan("b1_eps_scan.csv", &er);

    // finite-size scan at ε̇=0.3
    let lr = size_scan(&[16,24,32,48], 1.0, 0.3, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("b1_size_scan.csv", &lr);
}
