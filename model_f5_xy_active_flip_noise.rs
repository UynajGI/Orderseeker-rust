// ============================================================
// model_f5_xy_active_flip_noise.rs  —  Model F5  🟡 P3
// XY Model + Active Poisson Flip Noise  (θ_i → θ_i + π  at rate p per step)
// ============================================================
// Model family : F族 — 噪声谱工程
// Priority     : 🟡 P3  (LRO probability: low-medium ~10–25%)
//
// Physics motivation
// ------------------
//   Instead of continuous Gaussian noise, each spin is occasionally
//   subject to a hard "active flip" that rotates it by π:
//     θ_i → θ_i + π  with probability p per time step.
//   This Poisson-process perturbation breaks detailed balance
//   (the reverse process θ_i → θ_i − π has a different acceptance
//   probability in the presence of coupling) and can be viewed as
//   discrete injection events from an external "active" reservoir.
//
//   The standard thermal noise (√(2D dt) ξ) is also retained, so
//   the model has two noise channels:
//     1. Continuous Gaussian: models ordinary thermal fluctuations.
//     2. Discrete Poisson flips: models active injection events
//        (amplitude π ≫ thermal fluctuation amplitude √(2Ddt) ≈ 0.14).
//
//   Physical question: can the Poisson flips generate a non-trivial
//   noise spectrum that circumvents the Mermin-Wagner theorem?
//   Likely answer: NO, because π-flips introduce large-amplitude
//   uncorrelated noise that tends to destroy order.  However, studying
//   the competition between flip rate p and coupling J is physically
//   informative as a limiting case of non-equilibrium noise.
//
// Equation of motion  (mixed continuous/discrete SDE)
// ----------------------------------------------------
//   Step 1 (Euler-Maruyama):
//     θ_i ← θ_i + [J Σ sin(θ_j−θ_i)] dt + √(2D dt) ξ_i
//
//   Step 2 (Poisson flip):
//     with probability p: θ_i ← θ_i + π
//
// Parameters
// ----------
//   J   : XY coupling
//   D   : continuous noise strength
//   p   : flip probability per site per step (0 ≤ p ≤ 1)
//   dt  : time step
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨e^{iθ}⟩|     order parameter
//   C(r)  = spatial correlation along x
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   f5_timeseries.csv    step, t, m
//   f5_correlation.csv   r, C_r
//   f5_p_scan.csv        L, flip_prob, m_mean, m_std, binder
//   f5_size_scan.csv     L, flip_prob, m_mean, m_std, binder
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

struct XYLattice {
    l:     usize,
    theta: Vec<f64>,
}

impl XYLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i, j)] }

    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let ud = Uniform::new(0.0f64, 2.0 * PI);
        XYLattice { l, theta: (0..l*l).map(|_| rng.sample(ud)).collect() }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  Combined EM + Poisson flip step
// ─────────────────────────────────────────────────────────────

fn em_flip_step(lat: &mut XYLattice, j: f64, d: f64, flip_prob: f64,
                dt: f64, rng: &mut SmallRng) {
    let l  = lat.l;
    let n  = l * l;
    let nd = Normal::new(0.0f64, 1.0).unwrap();
    let ud = Uniform::new(0.0f64, 1.0f64);
    let sd = (2.0 * d * dt).sqrt();

    // ── Step 1: Euler-Maruyama (XY drift + continuous Gaussian noise) ──
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
                    + sd * rng.sample(nd);
        }
    }
    for k in 0..n { lat.theta[k] += dth[k]; }

    // ── Step 2: Poisson active flip (θ_i → θ_i + π with prob p) ──────
    if flip_prob > 0.0 {
        for k in 0..n {
            if rng.sample(ud) < flip_prob {
                lat.theta[k] += PI;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// § 3  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &XYLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &XYLattice, r_max: usize) -> Vec<f64> {
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

struct RunResult { l: usize, flip_prob: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, d: f64, flip_prob: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = XYLattice::new_random(l, &mut rng);
    let n_tr  = (t_trans / dt).round() as usize;
    let n_me  = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_flip_step(&mut lat, j, d, flip_prob, dt, &mut rng); }
    if verbose {
        println!("[F5] L={l} J={j} D={d:.2} p_flip={flip_prob:.4} \
                  transient={n_tr} steps done");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        em_flip_step(&mut lat, j, d, flip_prob, dt, &mut rng);
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, flip_prob, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 5  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, flip_prob: f64, m_mean: f64, m_std: f64, binder: f64 }

/// Flip probability scan:
/// p=0   → standard XY (no active flips, small D); expect m→0 (MW theorem)
/// p>0   → active flips destroy order; m decreases rapidly with p
fn p_scan(l: usize, p_arr: &[f64], j: f64, d: f64,
          dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    p_arr.iter().map(|&p| {
        let r = run(l, j, d, p, dt, tt, tm, 20, seed, false);
        println!("  L={l} p={p:.4}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, flip_prob: p, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, d: f64, flip_prob: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, d, flip_prob, dt, tt, tm, 20, seed, false);
        println!("  L={l} p={flip_prob:.4}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, flip_prob, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
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
        writeln!(w, "{},{:.6},{:.8}", s, s as f64*dt, m).unwrap();
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
    writeln!(w, "L,flip_prob,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.6},{:.6},{:.6},{:.6}",
                 r.l, r.flip_prob, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 7  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: L=32, J=1, D=0.1 (small thermal), p_flip=0.001 (rare flips)
    let res = run(32, 1.0, 0.1, 0.001, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("f5_timeseries.csv", &res.m_ts, res.dt, me);
    write_corr("f5_correlation.csv", &res.c_r);

    // flip probability scan:
    // p=0       → equilibrium XY (no flips)
    // p=0.0001  → very rare active events
    // p=0.01    → moderately active
    // p=0.1     → strongly disruptive
    let p_arr = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1].to_vec();
    let pr = p_scan(32, &p_arr, 1.0, 0.1, 0.01, 15.0, 60.0, 0);
    write_scan("f5_p_scan.csv", &pr);

    // finite-size scan at p=0.001
    let lr = size_scan(&[16,24,32,48], 1.0, 0.1, 0.001, 0.01, 15.0, 60.0, 1);
    write_scan("f5_size_scan.csv", &lr);
}
