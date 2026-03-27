// ============================================================
// model_a3_kuramoto_osc_drive.rs  —  Model A3  🟠 P2
// Kuramoto Lattice + Spatially-Modulated Oscillatory Drive
// ============================================================
// Model family : A族 — Kuramoto振子格子 + 非平衡驱动
// Priority     : 🟠 P2  (LRO probability: medium-high 40–70%)
//
// Physics motivation
// ------------------
//   Ikeda & Kuroda (2024) showed that a spatially non-uniform
//   oscillatory drive can stabilise LRO by suppressing Goldstone-mode
//   IR divergences.  Here we apply this mechanism to the Kuramoto
//   phase field: a long-wave-modulated sinusoidal drive
//     F_i(t) = A cos(q₀·r_i) cos(ω t)
//   with q₀ → 0 (long-wave limit, taken as q₀ = 2π/L for one wavelength).
//   This tests whether oscillatory drive on a phase-only degree of
//   freedom (no amplitude) achieves the same IR suppression.
//
// Equation of motion  (Euler-Maruyama SDE, a = 1)
// -----------------------------------------------
//   dθ_i = [ Ω₀
//            + J Σ_{⟨ij⟩} sin(θ_j − θ_i)
//            + A cos(q₀ x_i) cos(ω t)        ] dt
//          + √(2D dt) ξ_i
//
//   q₀ = 2π/L  (one full spatial wavelength),  x_i = col index.
//   The oscillatory drive has zero time-average and breaks no
//   spatial symmetry statistically.
//
// Parameters
// ----------
//   Ω₀    : common natural frequency (0 by default)
//   J     : coupling constant
//   A     : drive amplitude
//   ω     : drive angular frequency
//   D     : noise strength
//   dt    : time step (must satisfy dt < 2π/ω for resolution)
//
// Observables (→ CSV)
// -------------------
//   r(t)  = |⟨e^{iθ}⟩|           order parameter
//   C(r)  = spatial correlation along x
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   a3_timeseries.csv    step, t, r
//   a3_correlation.csv   r, C_r
//   a3_amp_scan.csv      L, A, omega, r_mean, r_std, binder
//   a3_size_scan.csv     L, A, omega, r_mean, r_std, binder
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

struct OscDriveLattice {
    l:     usize,
    theta: Vec<f64>,
}

impl OscDriveLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i, j)] }

    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let d = Uniform::new(0.0f64, 2.0 * PI);
        OscDriveLattice { l, theta: (0..l*l).map(|_| rng.sample(d)).collect() }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  Drift  f(θ, t)
// ─────────────────────────────────────────────────────────────

fn drift(lat: &OscDriveLattice, omega0: f64, j: f64,
         amp: f64, drv_omega: f64, t: f64) -> Vec<f64> {
    let l   = lat.l;
    let q0  = 2.0 * PI / l as f64;  // one wavelength across system
    let ft  = amp * drv_omega.mul_add(t, 0.0).cos(); // A cos(ω t)
    let mut f = vec![omega0; l * l];
    for i in 0..l {
        for jj in 0..l {
            let k  = lat.idx(i, jj);
            let th = lat.theta[k];
            let xj = jj as f64;
            let up  = lat.get(lat.w(i, 1),  jj);
            let dn  = lat.get(lat.w(i,-1),  jj);
            let rt  = lat.get(i, lat.w(jj, 1));
            let lf  = lat.get(i, lat.w(jj,-1));
            f[k] += j * ((up-th).sin() + (dn-th).sin()
                        + (rt-th).sin() + (lf-th).sin());
            // spatially-modulated oscillatory drive
            f[k] += (q0 * xj).cos() * ft;
        }
    }
    f
}

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut OscDriveLattice, omega0: f64, j: f64,
           amp: f64, drv_omega: f64, t: f64,
           d: f64, dt: f64, rng: &mut SmallRng) {
    let f  = drift(lat, omega0, j, amp, drv_omega, t);
    let sd = (2.0 * d * dt).sqrt();
    let nd = Normal::new(0.0f64, 1.0).unwrap();
    for k in 0..lat.l*lat.l {
        lat.theta[k] += f[k] * dt + sd * rng.sample(nd);
    }
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &OscDriveLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &OscDriveLattice, r_max: usize) -> Vec<f64> {
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

struct RunResult { l: usize, amp: f64, drv_omega: f64,
                   r_mean: f64, r_std: f64, binder: f64,
                   c_r: Vec<f64>, r_ts: Vec<f64>, dt: f64 }

fn run(l: usize, omega0: f64, j: f64, amp: f64, drv_omega: f64,
       d: f64, dt: f64, t_trans: f64, t_meas: f64,
       mevery: usize, seed: u64, verbose: bool) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = OscDriveLattice::new_random(l, &mut rng);
    let n_tr = (t_trans / dt).round() as usize;
    let n_me = (t_meas  / dt).round() as usize;

    let mut t = 0.0f64;
    for _ in 0..n_tr {
        em_step(&mut lat, omega0, j, amp, drv_omega, t, d, dt, &mut rng);
        t += dt;
    }
    if verbose {
        println!("[A3] L={l} A={amp:.3} ω={drv_omega:.3} J={j} D={d:.2} \
                  transient done");
    }

    let mut r_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, omega0, j, amp, drv_omega, t, d, dt, &mut rng);
        t += dt;
        if s % mevery == 0 { r_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let r_mean = mean_f(&r_ts);
    let r_std  = std_f(&r_ts);
    let bdr    = binder(&r_ts);
    if verbose { println!("      r = {r_mean:.4} ± {r_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, amp, drv_omega, r_mean, r_std, binder: bdr, c_r, r_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, amp: f64, drv_omega: f64,
              r_mean: f64, r_std: f64, binder: f64 }

fn amp_scan(l: usize, amp_arr: &[f64], drv_omega: f64, j: f64, d: f64,
            dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    amp_arr.iter().map(|&a| {
        let r = run(l, 0.0, j, a, drv_omega, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} A={a:.3} ω={drv_omega:.2}  r={:.4}  U_L={:.4}",
                 r.r_mean, r.binder);
        SRow { l, amp: a, drv_omega, r_mean: r.r_mean, r_std: r.r_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], amp: f64, drv_omega: f64, j: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, 0.0, j, amp, drv_omega, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} A={amp:.3} ω={drv_omega:.2}  r={:.4}  U_L={:.4}",
                 r.r_mean, r.binder);
        SRow { l, amp, drv_omega, r_mean: r.r_mean, r_std: r.r_std, binder: r.binder }
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
    writeln!(w, "L,A,omega,r_mean,r_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.amp, r.drv_omega, r.r_mean, r.r_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: L=32, A=0.5, ω=1.0, J=1, D=0.5
    let res = run(32, 0.0, 1.0, 0.5, 1.0, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("a3_timeseries.csv", &res.r_ts, res.dt, me);
    write_corr("a3_correlation.csv", &res.c_r);

    // amplitude scan A ∈ [0, 2.0] at ω=1.0
    let amp_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.2).collect();
    let ar = amp_scan(32, &amp_arr, 1.0, 1.0, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan("a3_amp_scan.csv", &ar);

    // finite-size scan at A=0.5, ω=1.0
    let lr = size_scan(&[16,24,32,48], 0.5, 1.0, 1.0, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("a3_size_scan.csv", &lr);
}
