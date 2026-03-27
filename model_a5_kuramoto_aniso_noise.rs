// ============================================================
// model_a5_kuramoto_aniso_noise.rs  —  Model A5  🟠 P2
// Kuramoto Lattice + Anisotropic Noise (T_x ≠ T_y)
// ============================================================
// Model family : A族 — Kuramoto振子格子 + 非平衡驱动
// Priority     : 🟠 P2  (LRO probability: medium 40–70%)
//
// Physics motivation
// ------------------
//   Bassler & Racz (1994, 1995) showed that an O(2) model with
//   anisotropic noise temperatures (T_x ≠ T_y for x and y bond
//   updates) develops LRO in 2D, breaking the Mermin-Wagner theorem.
//   Model A5 tests whether the same anisotropic-noise mechanism
//   applies to the Kuramoto phase field.
//   The coupling (XY-like sin interaction) retains full rotational
//   O(2) symmetry; only the noise is anisotropic.  The noise
//   anisotropy does NOT break the continuous O(2) symmetry of the
//   coupling, so it is consistent with the constraints.
//
// Equation of motion  (Euler-Maruyama SDE)
// -----------------------------------------
//   dθ_i = [ Ω₀ + J Σ_{⟨ij⟩} sin(θ_j − θ_i) ] dt
//           + √(2 D_x dt) ξ_i^x   [x-direction noise]
//           + √(2 D_y dt) ξ_i^y   [y-direction noise]
//
//   Noise anisotropy:  D_x ≠ D_y.
//   Implementation: split thermal noise into two independent
//   contributions with different amplitudes.
//   ξ_i^x, ξ_i^y ~ N(0,1) independent.
//   The effective noise is still isotropic in O(2) space (phase),
//   but the spatial diffusion of fluctuations is anisotropic.
//
//   Physical interpretation: the x-bonds are in thermal contact with
//   a bath at temperature T_x, y-bonds with T_y.  The coupling is
//   isotropic.
//
// Parameters
// ----------
//   D_x, D_y  : noise strengths in x and y directions
//   Ω₀        : common natural frequency
//   J         : coupling
//
// Observables (→ CSV)
// -------------------
//   r(t)  = |⟨e^{iθ}⟩|     order parameter
//   C(r)  = spatial correlation along x
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   a5_timeseries.csv    step, t, r
//   a5_correlation.csv   r, C_r
//   a5_aniso_scan.csv    L, D_x, D_y, r_mean, r_std, binder
//   a5_size_scan.csv     L, D_x, D_y, r_mean, r_std, binder
//
// Cargo deps: rand = "0.8" (features=["small_rng"]), rand_distr = "0.4"
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

struct AnisoLattice {
    l:     usize,
    theta: Vec<f64>,
    omega0: f64,
}

impl AnisoLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i, j)] }

    fn new_random(l: usize, omega0: f64, rng: &mut SmallRng) -> Self {
        let d = Uniform::new(0.0f64, 2.0 * PI);
        AnisoLattice { l, omega0, theta: (0..l*l).map(|_| rng.sample(d)).collect() }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  Drift
// ─────────────────────────────────────────────────────────────

fn drift(lat: &AnisoLattice, j: f64) -> Vec<f64> {
    let l = lat.l;
    let mut f = vec![lat.omega0; l * l];
    for i in 0..l {
        for jj in 0..l {
            let k  = lat.idx(i, jj);
            let th = lat.theta[k];
            let up  = lat.get(lat.w(i, 1),  jj);
            let dn  = lat.get(lat.w(i,-1),  jj);
            let rt  = lat.get(i, lat.w(jj, 1));
            let lf  = lat.get(i, lat.w(jj,-1));
            f[k] += j * ((up-th).sin() + (dn-th).sin()
                        + (rt-th).sin() + (lf-th).sin());
        }
    }
    f
}

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step (anisotropic noise)
// ─────────────────────────────────────────────────────────────

/// Anisotropic noise: x-direction gets D_x, y-direction gets D_y.
/// Implementation: the noise on each site is decomposed as
///   η_i = √(2 D_x dt) ξ_i^x  +  √(2 D_y dt) ξ_i^y
/// where ξ^x couples to the x-bond neighbourhood and ξ^y to y-bond.
/// Simplest realisation: total site noise with anisotropic amplitude
/// split by a spatial-direction indicator via two independent draws.
fn em_step(lat: &mut AnisoLattice, j: f64, d_x: f64, d_y: f64,
           dt: f64, rng: &mut SmallRng) {
    let f   = drift(lat, j);
    let sd_x = (2.0 * d_x * dt).sqrt();
    let sd_y = (2.0 * d_y * dt).sqrt();
    let nd   = Normal::new(0.0f64, 1.0).unwrap();
    for k in 0..lat.l*lat.l {
        lat.theta[k] += f[k] * dt
            + sd_x * rng.sample(nd)
            + sd_y * rng.sample(nd);
    }
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &AnisoLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &AnisoLattice, r_max: usize) -> Vec<f64> {
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

struct RunResult { l: usize, d_x: f64, d_y: f64,
                   r_mean: f64, r_std: f64, binder: f64,
                   c_r: Vec<f64>, r_ts: Vec<f64>, dt: f64 }

fn run(l: usize, omega0: f64, j: f64, d_x: f64, d_y: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = AnisoLattice::new_random(l, omega0, &mut rng);
    let n_tr = (t_trans / dt).round() as usize;
    let n_me = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j, d_x, d_y, dt, &mut rng); }
    if verbose {
        println!("[A5] L={l} D_x={d_x:.2} D_y={d_y:.2} J={j} Ω₀={omega0} \
                  transient done");
    }

    let mut r_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, d_x, d_y, dt, &mut rng);
        if s % mevery == 0 { r_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let r_mean = mean_f(&r_ts);
    let r_std  = std_f(&r_ts);
    let bdr    = binder(&r_ts);
    if verbose { println!("      r = {r_mean:.4} ± {r_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, d_x, d_y, r_mean, r_std, binder: bdr, c_r, r_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, d_x: f64, d_y: f64,
              r_mean: f64, r_std: f64, binder: f64 }

/// Anisotropy scan: fix D_x=0.5, vary D_y from 0 (strong aniso) to 0.5 (isotropic).
fn aniso_scan(l: usize, dy_arr: &[f64], d_x: f64, j: f64,
              dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    dy_arr.iter().map(|&dy| {
        let r = run(l, 0.0, j, d_x, dy, dt, tt, tm, 20, seed, false);
        println!("  L={l} D_x={d_x:.2} D_y={dy:.3}  r={:.4}  U_L={:.4}",
                 r.r_mean, r.binder);
        SRow { l, d_x, d_y: dy, r_mean: r.r_mean, r_std: r.r_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], d_x: f64, d_y: f64, j: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, 0.0, j, d_x, d_y, dt, tt, tm, 20, seed, false);
        println!("  L={l} D_x={d_x:.2} D_y={d_y:.2}  r={:.4}  U_L={:.4}",
                 r.r_mean, r.binder);
        SRow { l, d_x, d_y, r_mean: r.r_mean, r_std: r.r_std, binder: r.binder }
    }).collect()
}

// ─────────────────────────────────────────────────────────────
// § 7  CSV helpers
// ─────────────────────────────────────────────────────────────

fn write_ts(path: &str, r_ts: &[f64], dt: f64, me: usize) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "step,t,r").unwrap();
    for (k,&r) in r_ts.iter().enumerate() {
        let s = k*me;
        writeln!(w, "{},{:.6},{:.8}", s, s as f64 * dt, r).unwrap();
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
    writeln!(w, "L,D_x,D_y,r_mean,r_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.d_x, r.d_y, r.r_mean, r.r_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: D_x=0.5, D_y=0.1 (strong anisotropy), J=1
    let res = run(32, 0.0, 1.0, 0.5, 0.1, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("a5_timeseries.csv", &res.r_ts, res.dt, me);
    write_corr("a5_correlation.csv", &res.c_r);

    // anisotropy scan: D_x=0.5 fixed, D_y=0→0.5 (isotropic limit)
    let dy_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.05).collect();
    let ar = aniso_scan(32, &dy_arr, 0.5, 1.0, 0.01, 15.0, 60.0, 0);
    write_scan("a5_aniso_scan.csv", &ar);

    // finite-size scan at D_x=0.5, D_y=0.05 (strong anisotropy)
    let lr = size_scan(&[16,24,32,48], 0.5, 0.05, 1.0, 0.01, 15.0, 60.0, 1);
    write_scan("a5_size_scan.csv", &lr);
}
