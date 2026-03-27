// ============================================================
// model_b6_xy_oscillating_bath.rs  —  Model B6  🟠 P2
// XY Model + Time-Modulated Noise Strength D(t)
// ============================================================
// Model family : B族 — XY模型 / O(2)场 + 新型非平衡驱动
// Priority     : 🟠 P2  (LRO probability: medium 40–70%)
//
// Physics motivation
// ------------------
//   Ikeda & Kuroda (2024) introduced spatially non-uniform oscillatory
//   drives to stabilise LRO.  Model B6 asks the complementary question:
//   can a TEMPORALLY modulated noise strength D(t) = D₀[1 + A cos(ωd t)]
//   achieve the same IR suppression?
//   The noise amplitude varies periodically in time while the coupling
//   retains full O(2) symmetry.  The time-averaged noise strength is
//   D₀ (same as the equilibrium reference), so the effective
//   temperature is unchanged on long time scales.
//
// Equation of motion  (Euler-Maruyama SDE)
// -----------------------------------------
//   dθ_i = J Σ_{⟨ij⟩} sin(θ_j − θ_i) dt
//           + √(2 D(t) dt) ξ_i
//
//   D(t) = D₀ [1 + A_mod cos(ω_d t)],  |A_mod| < 1 for positivity.
//
// Parameters
// ----------
//   J      : coupling
//   D₀     : mean noise strength
//   A_mod  : modulation amplitude (0 → no modulation, pure white noise)
//   ω_d    : drive frequency
//   dt     : time step
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨e^{iθ}⟩|     magnetisation
//   C(r)  = spatial correlation along x
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   b6_timeseries.csv    step, t, m
//   b6_correlation.csv   r, C_r
//   b6_amp_scan.csv      L, A_mod, omega_d, m_mean, m_std, binder
//   b6_size_scan.csv     L, A_mod, omega_d, m_mean, m_std, binder
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

struct OscBathLattice {
    l:     usize,
    theta: Vec<f64>,
}

impl OscBathLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i, j)] }

    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let d = Uniform::new(0.0f64, 2.0 * PI);
        OscBathLattice { l, theta: (0..l*l).map(|_| rng.sample(d)).collect() }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  Drift
// ─────────────────────────────────────────────────────────────

fn drift(lat: &OscBathLattice, j: f64) -> Vec<f64> {
    let l = lat.l;
    let mut f = vec![0.0f64; l * l];
    for i in 0..l {
        for jj in 0..l {
            let k  = lat.idx(i, jj);
            let th = lat.theta[k];
            let up  = lat.get(lat.w(i, 1),  jj);
            let dn  = lat.get(lat.w(i,-1),  jj);
            let rt  = lat.get(i, lat.w(jj, 1));
            let lf  = lat.get(i, lat.w(jj,-1));
            f[k] = j * ((up-th).sin() + (dn-th).sin()
                       + (rt-th).sin() + (lf-th).sin());
        }
    }
    f
}

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step  (time-modulated D)
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut OscBathLattice, j: f64,
           d0: f64, a_mod: f64, omega_d: f64, t: f64,
           dt: f64, rng: &mut SmallRng) {
    let f  = drift(lat, j);
    let dt_eff = d0 * (1.0 + a_mod * (omega_d * t).cos());
    let dt_eff = dt_eff.max(0.0); // ensure non-negative
    let sd = (2.0 * dt_eff * dt).sqrt();
    let nd = Normal::new(0.0f64, 1.0).unwrap();
    for k in 0..lat.l*lat.l {
        lat.theta[k] += f[k] * dt + sd * rng.sample(nd);
    }
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &OscBathLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &OscBathLattice, r_max: usize) -> Vec<f64> {
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

struct RunResult { l: usize, a_mod: f64, omega_d: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, d0: f64, a_mod: f64, omega_d: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = OscBathLattice::new_random(l, &mut rng);
    let n_tr = (t_trans / dt).round() as usize;
    let n_me = (t_meas  / dt).round() as usize;

    let mut t = 0.0f64;
    for _ in 0..n_tr {
        em_step(&mut lat, j, d0, a_mod, omega_d, t, dt, &mut rng);
        t += dt;
    }
    if verbose {
        println!("[B6] L={l} J={j} D₀={d0:.2} A={a_mod:.3} ω_d={omega_d:.2} \
                  transient done");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, d0, a_mod, omega_d, t, dt, &mut rng);
        t += dt;
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, a_mod, omega_d, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, a_mod: f64, omega_d: f64,
              m_mean: f64, m_std: f64, binder: f64 }

fn amp_scan(l: usize, a_arr: &[f64], omega_d: f64, j: f64, d0: f64,
            dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    a_arr.iter().map(|&a| {
        let r = run(l, j, d0, a, omega_d, dt, tt, tm, 20, seed, false);
        println!("  L={l} A={a:.3} ω_d={omega_d:.2}  m={:.4}  U_L={:.4}",
                 r.m_mean, r.binder);
        SRow { l, a_mod: a, omega_d, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], a_mod: f64, omega_d: f64, j: f64, d0: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, d0, a_mod, omega_d, dt, tt, tm, 20, seed, false);
        println!("  L={l} A={a_mod:.3} ω_d={omega_d:.2}  m={:.4}  U_L={:.4}",
                 r.m_mean, r.binder);
        SRow { l, a_mod, omega_d, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

// ─────────────────────────────────────────────────────────────
// § 7  CSV helpers
// ─────────────────────────────────────────────────────────────

fn write_ts(path: &str, m_ts: &[f64], dt: f64, me: usize) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "step,t,m").unwrap();
    for (k,&m) in m_ts.iter().enumerate() {
        let s = k*me;
        writeln!(w, "{},{:.6},{:.8}", s, s as f64 * dt, m).unwrap();
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
    writeln!(w, "L,A_mod,omega_d,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.a_mod, r.omega_d, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: A_mod=0.8, ω_d=2.0, J=1, D₀=0.5
    let res = run(32, 1.0, 0.5, 0.8, 2.0, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("b6_timeseries.csv", &res.m_ts, res.dt, me);
    write_corr("b6_correlation.csv", &res.c_r);

    // modulation amplitude scan A ∈ [0, 0.9]
    let a_arr: Vec<f64> = (0..=9).map(|k| k as f64 * 0.1).collect();
    let ar = amp_scan(32, &a_arr, 2.0, 1.0, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan("b6_amp_scan.csv", &ar);

    // finite-size scan at A=0.8, ω_d=2.0
    let lr = size_scan(&[16,24,32,48], 0.8, 2.0, 1.0, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("b6_size_scan.csv", &lr);
}
