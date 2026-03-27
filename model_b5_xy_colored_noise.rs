// ============================================================
// model_b5_xy_colored_noise.rs  —  Model B5  🟠 P2
// XY Model + Exponentially Correlated (Coloured) Noise
// ============================================================
// Model family : B族 — XY模型 / O(2)场 + 新型非平衡驱动
// Priority     : 🟠 P2  (LRO probability: medium 40–70%)
//
// Physics motivation
// ------------------
//   White noise drives white-spectrum fluctuations in the XY model,
//   giving the standard Mermin-Wagner result.  Replacing white noise
//   with exponentially correlated (Ornstein-Uhlenbeck) noise of
//   correlation time τ_c changes the effective noise spectrum:
//     S(ω) ∝ τ_c / (1 + ω² τ_c²)   (Lorentzian)
//   For large τ_c this suppresses high-frequency fluctuations and
//   approaches a quasi-static coloured field, potentially modifying
//   the IR behaviour.  In the τ_c → ∞ limit the noise becomes
//   effectively frozen (acts like quenched disorder), while for
//   τ_c → 0 it recovers white noise.
//
// Equation of motion  (Euler-Maruyama SDE + OU noise)
// ----------------------------------------------------
//   dθ_i = J Σ_{⟨ij⟩} sin(θ_j − θ_i) dt  +  η_i(t) dt
//   dη_i = −(1/τ_c) η_i dt + √(2 D/τ_c) dW_i
//
//   η_i : Ornstein-Uhlenbeck process, ⟨η_i(t)η_i(t')⟩ = (D/τ_c) e^{−|t−t'|/τ_c}
//   For τ_c → 0: η_i(t) → √(2D) ξ_i(t)  (white noise limit)
//
// Parameters
// ----------
//   J     : coupling
//   D     : noise intensity (effective diffusivity)
//   τ_c   : noise correlation time
//   dt    : time step
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨e^{iθ}⟩|     magnetisation
//   C(r)  = spatial correlation along x
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   b5_timeseries.csv    step, t, m
//   b5_correlation.csv   r, C_r
//   b5_tauc_scan.csv     L, tau_c, m_mean, m_std, binder
//   b5_size_scan.csv     L, tau_c, m_mean, m_std, binder
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
// § 1  Lattice + OU noise state
// ─────────────────────────────────────────────────────────────

struct ColouredXYLattice {
    l:     usize,
    theta: Vec<f64>,
    eta:   Vec<f64>,  // OU noise per site
}

impl ColouredXYLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i, j)] }

    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let d = Uniform::new(0.0f64, 2.0 * PI);
        ColouredXYLattice {
            l,
            theta: (0..l*l).map(|_| rng.sample(d)).collect(),
            eta:   vec![0.0f64; l*l],
        }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  One SDE step
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut ColouredXYLattice, j: f64, d: f64,
           tau_c: f64, dt: f64, rng: &mut SmallRng) {
    let l  = lat.l;
    let n  = l * l;
    let nd = Normal::new(0.0f64, 1.0).unwrap();

    // ── update OU noise process ──────────────────────────────
    // Exact OU update: η(t+dt) = η(t) e^{-dt/τ} + √(D/τ_c (1-e^{-2dt/τ})) ξ
    let decay = (-dt / tau_c).exp();
    let ou_sd = (d / tau_c * (1.0 - decay * decay)).sqrt();
    for k in 0..n {
        lat.eta[k] = decay * lat.eta[k] + ou_sd * rng.sample(nd);
    }

    // ── XY drift + coloured noise ────────────────────────────
    let mut dth = vec![0.0f64; n];
    for i in 0..l {
        for jj in 0..l {
            let k  = lat.idx(i, jj);
            let th = lat.theta[k];
            let up  = lat.get(lat.w(i, 1),  jj);
            let dn  = lat.get(lat.w(i,-1),  jj);
            let rt  = lat.get(i, lat.w(jj, 1));
            let lf  = lat.get(i, lat.w(jj,-1));
            dth[k] = j * ((up-th).sin() + (dn-th).sin()
                         + (rt-th).sin() + (lf-th).sin()) * dt
                   + lat.eta[k] * dt;
        }
    }
    for k in 0..n { lat.theta[k] += dth[k]; }
}

// ─────────────────────────────────────────────────────────────
// § 3  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &ColouredXYLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &ColouredXYLattice, r_max: usize) -> Vec<f64> {
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
// § 4  Runner
// ─────────────────────────────────────────────────────────────

struct RunResult { l: usize, tau_c: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, d: f64, tau_c: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = ColouredXYLattice::new_random(l, &mut rng);
    let n_tr = (t_trans / dt).round() as usize;
    let n_me = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j, d, tau_c, dt, &mut rng); }
    if verbose {
        println!("[B5] L={l} J={j} D={d:.2} τ_c={tau_c:.3} transient done");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, d, tau_c, dt, &mut rng);
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, tau_c, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 5  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, tau_c: f64, m_mean: f64, m_std: f64, binder: f64 }

fn tauc_scan(l: usize, tc_arr: &[f64], j: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    tc_arr.iter().map(|&tc| {
        let r = run(l, j, d, tc, dt, tt, tm, 20, seed, false);
        println!("  L={l} τ_c={tc:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, tau_c: tc, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, d: f64, tau_c: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, d, tau_c, dt, tt, tm, 20, seed, false);
        println!("  L={l} τ_c={tau_c:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, tau_c, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

// ─────────────────────────────────────────────────────────────
// § 6  CSV helpers
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
    writeln!(w, "L,tau_c,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.tau_c, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 7  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: τ_c=1.0 (correlated), J=1, D=0.5
    let res = run(32, 1.0, 0.5, 1.0, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("b5_timeseries.csv", &res.m_ts, res.dt, me);
    write_corr("b5_correlation.csv", &res.c_r);

    // τ_c scan: τ_c=0.01 (≈white) → τ_c=5.0 (strongly correlated)
    let tc_arr: Vec<f64> = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0].to_vec();
    let tr = tauc_scan(32, &tc_arr, 1.0, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan("b5_tauc_scan.csv", &tr);

    // finite-size scan at τ_c=1.0
    let lr = size_scan(&[16,24,32,48], 1.0, 0.5, 1.0, 0.01, 15.0, 60.0, 1);
    write_scan("b5_size_scan.csv", &lr);
}
