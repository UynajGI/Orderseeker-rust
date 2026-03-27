// ============================================================
// model_b4_xy_time_delay.rs  —  Model B4  🟠 P2
// XY Model + Time-Delayed Coupling (memory feedback)
// ============================================================
// Model family : B族 — XY模型 / O(2)场 + 新型非平衡驱动
// Priority     : 🟠 P2  (LRO probability: medium-high 40–70%)
//
// Physics motivation
// ------------------
//   A time delay τ in the coupling effectively introduces memory
//   into the dynamics, breaking time-reversal symmetry (non-equilibrium).
//   The delay is analogous to coloured noise / retarded interaction,
//   and may suppress long-wave fluctuations by introducing an effective
//   mass gap for the Goldstone mode.  This mechanism is unexplored
//   in the XY context.
//
// Equation of motion  (Euler-Maruyama SDE with delay)
// ----------------------------------------------------
//   dθ_i(t) = J Σ_{⟨ij⟩} sin[θ_j(t−τ) − θ_i(t)] dt
//             + √(2D dt) ξ_i
//
//   τ > 0 : delay time.
//   Implementation: store a ring buffer of past θ states
//   of depth n_delay = round(τ/dt).  The coupling uses the
//   buffered value θ(t−τ), while the noise and current state
//   use θ(t).
//
//   For τ=0 this reduces to the standard equilibrium XY model.
//
// Parameters
// ----------
//   J     : coupling
//   τ     : delay time
//   D     : noise strength
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
//   b4_timeseries.csv    step, t, m
//   b4_correlation.csv   r, C_r
//   b4_tau_scan.csv      L, tau, m_mean, m_std, binder
//   b4_size_scan.csv     L, tau, m_mean, m_std, binder
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
// § 1  Lattice + ring buffer for delay
// ─────────────────────────────────────────────────────────────

struct DelayLattice {
    l:       usize,
    theta:   Vec<f64>,
    // ring buffer of past states: shape [n_delay+1][l*l]
    history: Vec<Vec<f64>>,
    head:    usize,   // current write position in ring
    n_delay: usize,   // number of steps in delay
}

impl DelayLattice {
    fn new_random(l: usize, tau: f64, dt: f64, rng: &mut SmallRng) -> Self {
        let n_delay = (tau / dt).round() as usize + 1;
        let d = Uniform::new(0.0f64, 2.0 * PI);
        let theta: Vec<f64> = (0..l*l).map(|_| rng.sample(d)).collect();
        // initialise all history to the same random state
        let history = vec![theta.clone(); n_delay + 1];
        DelayLattice { l, theta, history, head: 0, n_delay }
    }

    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get_current(&self, i: usize, j: usize) -> f64 {
        self.theta[self.idx(i, j)]
    }
    /// Get θ(t − τ) from the ring buffer.
    #[inline] fn get_delayed(&self, i: usize, j: usize) -> f64 {
        let ring_len = self.n_delay + 1;
        let past = (self.head + 1) % ring_len;  // oldest slot
        self.history[past][self.idx(i, j)]
    }
    /// Push current theta into the ring buffer.
    fn push_history(&mut self) {
        let ring_len = self.history.len();
        self.history[self.head] = self.theta.clone();
        self.head = (self.head + 1) % ring_len;
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  One SDE step
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut DelayLattice, j: f64, d: f64,
           dt: f64, rng: &mut SmallRng) {
    let l  = lat.l;
    let n  = l * l;
    let nd = Normal::new(0.0f64, 1.0).unwrap();
    let sd = (2.0 * d * dt).sqrt();

    lat.push_history();  // save θ(t) before update

    let mut dth = vec![0.0f64; n];
    for i in 0..l {
        for jj in 0..l {
            let k   = lat.idx(i, jj);
            let th  = lat.get_current(i, jj);
            // neighbours at t − τ (from ring buffer)
            let up  = lat.get_delayed(lat.w(i, 1),  jj);
            let dn  = lat.get_delayed(lat.w(i,-1),  jj);
            let rt  = lat.get_delayed(i, lat.w(jj, 1));
            let lf  = lat.get_delayed(i, lat.w(jj,-1));
            // XY coupling with delayed neighbours
            dth[k] = j * ((up-th).sin() + (dn-th).sin()
                         + (rt-th).sin() + (lf-th).sin()) * dt
                   + sd * rng.sample(nd);
        }
    }
    for k in 0..n { lat.theta[k] += dth[k]; }
}

// ─────────────────────────────────────────────────────────────
// § 3  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &DelayLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &DelayLattice, r_max: usize) -> Vec<f64> {
    let l = lat.l;
    let n = (l * l) as f64;
    let mut c = vec![0.0f64; r_max + 1];
    c[0] = 1.0;
    for r in 1..=r_max {
        c[r] = (0..l).flat_map(|i| (0..l).map(move |j| (i,j)))
            .map(|(i,j)| (lat.get_current(i,j) - lat.get_current(i,(j+r)%l)).cos())
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

struct RunResult { l: usize, tau: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, tau: f64, d: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = DelayLattice::new_random(l, tau, dt, &mut rng);
    let n_tr = (t_trans / dt).round() as usize;
    let n_me = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j, d, dt, &mut rng); }
    if verbose {
        println!("[B4] L={l} J={j} τ={tau:.3} D={d:.2} n_delay={} \
                  transient done", lat.n_delay);
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, d, dt, &mut rng);
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, tau, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 5  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, tau: f64, m_mean: f64, m_std: f64, binder: f64 }

fn tau_scan(l: usize, tau_arr: &[f64], j: f64, d: f64,
            dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    tau_arr.iter().map(|&tau| {
        let r = run(l, j, tau, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} τ={tau:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, tau, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, tau: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, tau, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} τ={tau:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, tau, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
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
    writeln!(w, "L,tau,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.tau, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 7  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: L=32, τ=0.3, J=1, D=0.5
    let res = run(32, 1.0, 0.3, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("b4_timeseries.csv", &res.m_ts, res.dt, me);
    write_corr("b4_correlation.csv", &res.c_r);

    // τ scan: τ=0 (standard XY, no LRO) → τ=1.0 (strong delay)
    let tau_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.1).collect();
    let tr = tau_scan(32, &tau_arr, 1.0, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan("b4_tau_scan.csv", &tr);

    // finite-size scan at τ=0.3
    let lr = size_scan(&[16,24,32,48], 1.0, 0.3, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("b4_size_scan.csv", &lr);
}
