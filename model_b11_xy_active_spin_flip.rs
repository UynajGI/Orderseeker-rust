// ============================================================
// model_b11_xy_active_spin_flip.rs  —  Model B11  🟡 P3
// XY Model + Self-Consistent Active Angular Velocity
// ============================================================
// Model family : B族 — XY模型 / O(2)场 + 新型非平衡驱动
// Priority     : 🟡 P3  (LRO probability: medium ~20–40%)
//
// Physics motivation
// ------------------
//   Each spin carries an active angular velocity ω_i^act that is NOT
//   quenched but self-consistently determined by its local neighbourhood:
//
//     ω_i^act = ε Σ_{j∈nn(i)} cos(θ_j − θ_i)
//
//   This is a nonlinear active term: sites in a locally ordered patch
//   receive a positive angular push, while disordered sites receive
//   near-zero push.  Unlike quenched disorder (which breaks the
//   ergodic structure), this frequency is dynamic and configuration-
//   dependent.  The cos() coupling is the second harmonic of the
//   standard sin() coupling, effectively an XY model perturbation
//   that breaks time-reversal symmetry.
//
//   Physically this corresponds to an "active rotor" model where the
//   spinning rate adapts to local alignment — reminiscent of certain
//   motile agents.  The resulting dynamics are non-equilibrium and may
//   generate effective non-reciprocal interactions.
//
// Equation of motion  (Euler-Maruyama SDE, lattice spacing a = 1)
// ---------------------------------------------------------------
//   dθ_i = [ J Σ_{j∈nn(i)} sin(θ_j − θ_i)
//            + ε Σ_{j∈nn(i)} cos(θ_j − θ_i)  ] dt
//          + √(2D dt) ξ_i
//
//   The active term ε Σ cos(θ_j−θ_i) is a total-derivative-like
//   contribution in 1D but NOT derivable from a global Hamiltonian in 2D,
//   so the dynamics break detailed balance for ε ≠ 0.
//   The cos() term has zero mean in a disordered state and finite mean
//   in an ordered state, creating a positive feedback loop.
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨e^{iθ}⟩|     order parameter
//   C(r)  = spatial correlation along x
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   b11_timeseries.csv    step, t, m
//   b11_correlation.csv   r, C_r
//   b11_eps_scan.csv      L, epsilon, m_mean, m_std, binder
//   b11_size_scan.csv     L, epsilon, m_mean, m_std, binder
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
// § 2  Drift  f(θ)
// ─────────────────────────────────────────────────────────────

/// f_i = J Σ sin(θ_j − θ_i)     [XY coupling, breaks detail balance via time-reversal]
///       + ε Σ cos(θ_j − θ_i)   [active angular velocity: self-consistent, dynamic]
///
/// The cos() term represents ω_i^act = ε Σ_j cos(θ_j − θ_i), which is
/// configuration-dependent (not quenched) and breaks detailed balance.
fn drift(lat: &XYLattice, j: f64, eps: f64) -> Vec<f64> {
    let l   = lat.l;
    let mut f = vec![0.0f64; l * l];
    for i in 0..l {
        for jj in 0..l {
            let k  = lat.idx(i, jj);
            let th = lat.theta[k];
            let up  = lat.get(lat.w(i, 1),  jj);
            let dn  = lat.get(lat.w(i,-1),  jj);
            let rt  = lat.get(i, lat.w(jj, 1));
            let lf  = lat.get(i, lat.w(jj,-1));
            // standard XY sin coupling
            f[k] = j * ((up-th).sin() + (dn-th).sin()
                       + (rt-th).sin() + (lf-th).sin());
            // active cos coupling  ε Σ cos(θ_j − θ_i)
            f[k] += eps * ((up-th).cos() + (dn-th).cos()
                          + (rt-th).cos() + (lf-th).cos());
        }
    }
    f
}

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut XYLattice, j: f64, eps: f64,
           d: f64, dt: f64, rng: &mut SmallRng) {
    let f  = drift(lat, j, eps);
    let sd = (2.0 * d * dt).sqrt();
    let nd = Normal::new(0.0f64, 1.0).unwrap();
    for k in 0..lat.l*lat.l {
        lat.theta[k] += f[k] * dt + sd * rng.sample(nd);
    }
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
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
// § 5  Runner
// ─────────────────────────────────────────────────────────────

struct RunResult { l: usize, eps: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, eps: f64, d: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = XYLattice::new_random(l, &mut rng);
    let n_tr  = (t_trans / dt).round() as usize;
    let n_me  = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j, eps, d, dt, &mut rng); }
    if verbose {
        println!("[B11] L={l} J={j} ε={eps:.3} D={d:.2} \
                  transient={n_tr} steps done");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, eps, d, dt, &mut rng);
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, eps, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, eps: f64, m_mean: f64, m_std: f64, binder: f64 }

/// ε scan: ε=0 → standard equilibrium XY (no LRO); ε>0 → active perturbation.
fn eps_scan(l: usize, eps_arr: &[f64], j: f64, d: f64,
            dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    eps_arr.iter().map(|&e| {
        let r = run(l, j, e, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} ε={e:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, eps: e, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, eps: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, eps, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} ε={eps:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
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
    writeln!(w, "L,epsilon,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.eps, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: L=32, J=1, ε=0.5, D=0.5
    let res = run(32, 1.0, 0.5, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("b11_timeseries.csv", &res.m_ts, res.dt, me);
    write_corr("b11_correlation.csv", &res.c_r);

    // ε scan: ε=0 (standard XY) → ε=1.0 (strong active spinning)
    let eps_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.1).collect();
    let er = eps_scan(32, &eps_arr, 1.0, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan("b11_eps_scan.csv", &er);

    // finite-size scan at ε=0.5
    let lr = size_scan(&[16,24,32,48], 1.0, 0.5, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("b11_size_scan.csv", &lr);
}
