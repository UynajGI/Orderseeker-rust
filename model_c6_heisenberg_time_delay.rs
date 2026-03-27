// ============================================================
// model_c6_heisenberg_time_delay.rs  —  Model C6  🟡 P3
// O(3) Heisenberg Model + Time-Delayed Coupling (τ > 0)
// ============================================================
// Model family : C族 — O(N≥3)矢量场 + 各类非平衡机制
// Priority     : 🟡 P3  (LRO probability: medium ~20–40%)
//
// Physics motivation
// ------------------
//   Model B4 introduced time-delayed coupling to the O(2) XY model.
//   Model C6 extends this to the O(3) Heisenberg model.  The delayed
//   coupling n_j(t−τ)·n_i(t) form of interaction breaks time-reversal
//   symmetry (TRS) because under t → −t, the delay τ maps to −τ.
//   This non-equilibrium character may stabilise LRO through
//   mechanisms analogous to the active noise suppression of Goldstone
//   modes.
//
//   Implementation: the history of all spin configurations over a
//   delay time τ is stored in a ring buffer of size
//     n_delay = round(τ / dt)  steps.
//   The coupling is computed from the state at time (t − τ).
//
// Equation of motion  (Euler-Maruyama SDE on S², with delay)
// -----------------------------------------------------------
//   d n_i(t) = P_⊥(n_i(t)) [J Σ_{j∈nn(i)} n_j(t−τ)] dt
//              + √(2D dt) P_⊥(n_i(t)) ξ_i
//   n_i ← n_i / |n_i|
//
//   P_⊥(n) v = v − (v·n) n   (tangent space projection)
//   n_j(t−τ) is fetched from the ring buffer.
//
// Parameters
// ----------
//   J   : Heisenberg coupling
//   D   : noise strength
//   τ   : delay time (in simulation time units; τ=0 → standard Heisenberg)
//   dt  : time step (ring buffer stores n_delay = round(τ/dt) frames)
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨n⟩|   vector order parameter
//   C(r)  = ⟨n_0 · n_r⟩ spatial correlation along x
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   c6_timeseries.csv    step, t, m
//   c6_correlation.csv   r, C_r
//   c6_tau_scan.csv      L, tau, m_mean, m_std, binder
//   c6_size_scan.csv     L, tau, m_mean, m_std, binder
//
// Cargo deps: rand = "0.8" (features=["small_rng"]), rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, UnitSphere};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};

// ─────────────────────────────────────────────────────────────
// § 1  Types and helpers for O(3)
// ─────────────────────────────────────────────────────────────

type Vec3 = [f64; 3];

#[inline] fn dot(a: &Vec3, b: &Vec3) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
#[inline] fn add3(a: Vec3, b: Vec3)  -> Vec3 { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
#[inline] fn sub3(a: Vec3, b: Vec3)  -> Vec3 { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
#[inline] fn scale3(s: f64, v: Vec3) -> Vec3 { [s*v[0], s*v[1], s*v[2]] }

#[inline]
fn proj_tangent(n: &Vec3, v: Vec3) -> Vec3 {
    sub3(v, scale3(dot(n, &v), *n))
}

#[inline]
fn normalise(v: Vec3) -> Vec3 {
    let r = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if r < 1e-30 { [1.0, 0.0, 0.0] } else { [v[0]/r, v[1]/r, v[2]/r] }
}

// ─────────────────────────────────────────────────────────────
// § 2  Lattice with ring buffer for delay history
// ─────────────────────────────────────────────────────────────

/// The ring buffer stores the last `n_delay` full spin configurations.
/// `history[0]` = most recent snapshot added,
/// `history[n_delay-1]` = oldest snapshot = state at (t - τ).
struct DelayedHeisenbergLattice {
    l:       usize,
    spn:     Vec<Vec3>,             // current spin state
    history: VecDeque<Vec<Vec3>>,   // ring buffer of past states
    n_delay: usize,                 // = round(tau/dt)
}

impl DelayedHeisenbergLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get_cur(&self, i: usize, j: usize) -> Vec3 { self.spn[self.idx(i,j)] }
    #[inline] fn get_del(&self, i: usize, j: usize) -> Vec3 {
        // oldest entry = state at t - τ
        self.history.back().unwrap()[self.idx(i, j)]
    }

    fn new_random(l: usize, n_delay: usize, rng: &mut SmallRng) -> Self {
        let us  = UnitSphere;
        let spn: Vec<Vec3> = (0..l*l).map(|_| {
            let v = rng.sample(us); [v[0], v[1], v[2]]
        }).collect();
        // Initialise entire history with the same random state
        let mut history = VecDeque::with_capacity(n_delay.max(1));
        for _ in 0..n_delay.max(1) { history.push_front(spn.clone()); }
        DelayedHeisenbergLattice { l, spn, history, n_delay }
    }

    /// Push current state into ring buffer (discard oldest entry).
    fn push_history(&mut self) {
        self.history.push_front(self.spn.clone());
        if self.history.len() > self.n_delay.max(1) {
            self.history.pop_back();
        }
    }
}

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step on S² with time delay
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut DelayedHeisenbergLattice, j: f64,
           d: f64, dt: f64, rng: &mut SmallRng) {
    let l  = lat.l;
    let nd = Normal::new(0.0f64, 1.0).unwrap();
    let sd = (2.0 * d * dt).sqrt();

    let spn_cur = lat.spn.clone();

    for i in 0..l {
        for jj in 0..l {
            let k   = lat.idx(i, jj);
            let ni  = spn_cur[k];

            // Delayed neighbour states  n_j(t − τ)
            let up  = lat.get_del(lat.w(i, 1),  jj);
            let dn  = lat.get_del(lat.w(i,-1),  jj);
            let rt  = lat.get_del(i, lat.w(jj, 1));
            let lf  = lat.get_del(i, lat.w(jj,-1));

            // Coupling force using DELAYED spins
            let coupling = [
                j*(up[0]+dn[0]+rt[0]+lf[0]),
                j*(up[1]+dn[1]+rt[1]+lf[1]),
                j*(up[2]+dn[2]+rt[2]+lf[2]),
            ];
            let drift_j = proj_tangent(&ni, coupling);

            // Projected noise
            let noise_raw = [sd*rng.sample(nd), sd*rng.sample(nd), sd*rng.sample(nd)];
            let noise     = proj_tangent(&ni, noise_raw);

            let new_n = add3(ni, add3(scale3(dt, drift_j), noise));
            lat.spn[k] = normalise(new_n);
        }
    }

    // Push old current state into history buffer
    lat.push_history();
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &DelayedHeisenbergLattice) -> f64 {
    let n  = lat.l * lat.l;
    let m: Vec3 = lat.spn.iter().fold([0.0;3], |acc, s| add3(acc, *s));
    (dot(&m, &m)).sqrt() / n as f64
}

fn correlation(lat: &DelayedHeisenbergLattice, r_max: usize) -> Vec<f64> {
    let l  = lat.l;
    let nf = (l * l) as f64;
    let mut c = vec![0.0f64; r_max + 1];
    c[0] = 1.0;
    for r in 1..=r_max {
        c[r] = (0..l).flat_map(|i| (0..l).map(move |j| (i,j)))
            .map(|(i,j)| dot(&lat.get_cur(i,j), &lat.get_cur(i,(j+r)%l)))
            .sum::<f64>() / nf;
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

struct RunResult { l: usize, tau: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, d: f64, tau: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let n_delay = (tau / dt).round() as usize;
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = DelayedHeisenbergLattice::new_random(l, n_delay, &mut rng);
    let n_tr  = (t_trans / dt).round() as usize;
    let n_me  = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j, d, dt, &mut rng); }
    if verbose {
        println!("[C6] L={l} J={j} D={d:.2} τ={tau:.3} (n_delay={n_delay}) \
                  transient={n_tr} steps done  (O(3)+delay)");
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
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, tau: f64, m_mean: f64, m_std: f64, binder: f64 }

fn tau_scan(l: usize, tau_arr: &[f64], j: f64, d: f64,
            dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    tau_arr.iter().map(|&tau| {
        let r = run(l, j, d, tau, dt, tt, tm, 20, seed, false);
        println!("  L={l} τ={tau:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, tau, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, d: f64, tau: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, d, tau, dt, tt, tm, 20, seed, false);
        println!("  L={l} τ={tau:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, tau, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
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
    writeln!(w, "L,tau,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.tau, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: L=24, τ=0.5, J=1, D=0.5  (n_delay=50 steps)
    let res = run(24, 1.0, 0.5, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("c6_timeseries.csv", &res.m_ts, res.dt, me);
    write_corr("c6_correlation.csv", &res.c_r);

    // τ scan: τ=0 (no delay / standard Heisenberg) → τ=2.0 (long delay)
    let tau_arr = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0].to_vec();
    let tr = tau_scan(24, &tau_arr, 1.0, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan("c6_tau_scan.csv", &tr);

    // finite-size scan at τ=0.5
    let lr = size_scan(&[12,16,24,32], 1.0, 0.5, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("c6_size_scan.csv", &lr);
}
