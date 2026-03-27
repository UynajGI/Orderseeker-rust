// ============================================================
// model_b8_xy_poisson_update.rs  —  Model B8  🟡 P3
// XY Model + Asynchronous Poisson Clock Updates
// ============================================================
// Model family : B族 — XY模型 / O(2)场 + 新型非平衡驱动
// Priority     : 🟡 P3  (LRO probability: low-medium 20–40%)
//
// Physics motivation
// ------------------
//   Standard lattice dynamics use synchronous parallel updates.
//   Model B8 replaces this with asynchronous updates: each site i
//   has an independent Poisson clock with rate λ_i = λ (uniform).
//   Between events the site is frozen.  This breaks time-reversal
//   symmetry (non-equilibrium) but preserves the static O(2) symmetry.
//   The question is whether this asynchrony alone (without any
//   directed force) can alter the long-range behaviour of the XY model.
//
// Implementation  (continuous-time Monte Carlo / event-driven SDE)
// ---------------------------------------------------------------
//   Approach: simulate in discrete time steps dt, but each site is
//   updated with probability p = λ dt per step instead of always.
//   This gives independent Bernoulli(p) activation per site per step,
//   approximating independent Poisson(λ) clocks.
//
//   Update rule when site i fires:
//     θ_i ← θ_i + J Σ_{j∈NN} sin(θ_j − θ_i) Δt_i
//            + √(2D Δt_i) ξ_i
//   where Δt_i = 1/λ is the expected inter-event time.
//
//   For simplicity we use Δt_i = 1 (one "event" worth of drift),
//   scaled so that the average update rate matches a reference system.
//
// Parameters
// ----------
//   J      : coupling
//   D      : noise per update event
//   p_fire : probability of update per step per site (= λ dt)
//   dt     : base time step
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨e^{iθ}⟩|
//   C(r)  = spatial correlation
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   b8_timeseries.csv    step, t, m
//   b8_correlation.csv   r, C_r
//   b8_p_scan.csv        L, p_fire, m_mean, m_std, binder
//   b8_size_scan.csv     L, p_fire, m_mean, m_std, binder
//
// Cargo deps: rand = "0.8" (features=["small_rng"]), rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, Uniform, Bernoulli};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

// ─────────────────────────────────────────────────────────────
// § 1  Lattice
// ─────────────────────────────────────────────────────────────

struct PoissonXYLattice {
    l:     usize,
    theta: Vec<f64>,
}

impl PoissonXYLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i, j)] }

    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let d = Uniform::new(0.0f64, 2.0 * PI);
        PoissonXYLattice { l, theta: (0..l*l).map(|_| rng.sample(d)).collect() }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  One asynchronous step
// ─────────────────────────────────────────────────────────────

fn async_step(lat: &mut PoissonXYLattice, j: f64, d: f64,
              p_fire: f64, dt: f64, rng: &mut SmallRng) {
    let l    = lat.l;
    let n    = l * l;
    let nd   = Normal::new(0.0f64, 1.0).unwrap();
    let bern = Bernoulli::new(p_fire).unwrap();
    let sd   = (2.0 * d * dt).sqrt();

    // Snapshot current state (needed for consistent parallel-ish update)
    let theta_old = lat.theta.clone();

    for i in 0..l {
        for jj in 0..l {
            if !rng.sample(bern) { continue; }  // site does NOT fire this step
            let k  = lat.idx(i, jj);
            let th = theta_old[k];
            let up  = theta_old[lat.idx(lat.w(i, 1),  jj)];
            let dn  = theta_old[lat.idx(lat.w(i,-1),  jj)];
            let rt  = theta_old[lat.idx(i, lat.w(jj, 1))];
            let lf  = theta_old[lat.idx(i, lat.w(jj,-1))];
            lat.theta[k] += j * ((up-th).sin() + (dn-th).sin()
                                + (rt-th).sin() + (lf-th).sin()) * dt
                          + sd * rng.sample(nd);
        }
    }
    let _ = n; // suppress unused warning
}

// ─────────────────────────────────────────────────────────────
// § 3  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &PoissonXYLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &PoissonXYLattice, r_max: usize) -> Vec<f64> {
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

struct RunResult { l: usize, p_fire: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, d: f64, p_fire: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = PoissonXYLattice::new_random(l, &mut rng);
    let n_tr = (t_trans / dt).round() as usize;
    let n_me = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { async_step(&mut lat, j, d, p_fire, dt, &mut rng); }
    if verbose {
        println!("[B8] L={l} J={j} D={d:.2} p_fire={p_fire:.3} transient done");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        async_step(&mut lat, j, d, p_fire, dt, &mut rng);
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, p_fire, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 5  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, p_fire: f64, m_mean: f64, m_std: f64, binder: f64 }

fn p_scan(l: usize, p_arr: &[f64], j: f64, d: f64,
          dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    p_arr.iter().map(|&p| {
        let r = run(l, j, d, p, dt, tt, tm, 20, seed, false);
        println!("  L={l} p={p:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, p_fire: p, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, d: f64, p_fire: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, d, p_fire, dt, tt, tm, 20, seed, false);
        println!("  L={l} p={p_fire:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, p_fire, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
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
    writeln!(w, "L,p_fire,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.p_fire, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 7  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: p_fire=0.5 (half sites update each step)
    let res = run(32, 1.0, 0.5, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("b8_timeseries.csv", &res.m_ts, res.dt, me);
    write_corr("b8_correlation.csv", &res.c_r);

    // p_fire scan: p=1.0 (synchronous) → p=0.1 (sparse updates)
    let p_arr: Vec<f64> = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0].to_vec();
    let pr = p_scan(32, &p_arr, 1.0, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan("b8_p_scan.csv", &pr);

    // finite-size scan at p_fire=0.5
    let lr = size_scan(&[16,24,32,48], 1.0, 0.5, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("b8_size_scan.csv", &lr);
}
