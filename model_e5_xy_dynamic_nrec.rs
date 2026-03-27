// ============================================================
// model_e5_xy_dynamic_nrec.rs  —  Model E5  🟡 P3
// XY Model + Oscillating Non-Reciprocal Coupling  (J_NR(t) = J' cos(ω₀ t))
// ============================================================
// Model family : E族 — 非互易相互作用的新变体
// Priority     : 🟡 P3  (LRO probability: medium ~20–40%)
//
// Physics motivation
// ------------------
//   Models E1–E4 use static non-reciprocal couplings.  Model E5 makes
//   the non-reciprocal part TIME-DEPENDENT: the asymmetric coupling
//   coefficient oscillates at frequency ω₀.  This combines two
//   separately known LRO-generating mechanisms:
//     (1) Oscillatory driving (Ikeda-Kuroda 2024) — periodic force
//         can suppress Goldstone-mode IR divergences.
//     (2) Non-reciprocal coupling (Loos 2023) — static non-reciprocity
//         generates LRO via probability currents.
//
//   The coupling takes the form:
//     J_{ij}(t) = J     +   J' cos(ω₀ t)   if j is a "forward" neighbour of i
//     J_{ji}(t) = J     −   J' cos(ω₀ t)   (antisymmetric non-reciprocal part)
//     J_{ij}(t) = J_{ji}(t) = J              for "transverse" neighbours
//
//   "Forward" = {right, up} (fixed half-plane vision cone, same as E2).
//   The time-averaged non-reciprocal part ⟨J_{NR}⟩_t = 0, so the
//   mean coupling is symmetric — yet the instantaneous non-reciprocity
//   breaks detailed balance.
//
// Equation of motion  (Euler-Maruyama SDE, lattice spacing a = 1)
// ---------------------------------------------------------------
//   dθ_i = [ Σ_{j∈NN_sym(i)}  J  sin(θ_j − θ_i)
//            + Σ_{j∈NN_fwd(i)} J'cos(ω₀ t) sin(θ_j − θ_i)
//            − Σ_{j∈NN_bwd(i)} J'cos(ω₀ t) sin(θ_j − θ_i)
//           ] dt  +  √(2D dt) ξ_i
//
//   NN_fwd(i) = {right, up},  NN_bwd(i) = {left, down}  (i.e., transverse NN)
//   In practice this is equivalent to:
//     f_i = J Σ_{4 nn} sin(θ_j−θ_i)
//           + J'cos(ω₀t) [sin(θ_{right}−θ_i) + sin(θ_{up}−θ_i)
//                         −sin(θ_{left}−θ_i)  − sin(θ_{down}−θ_i)]
//
// Parameters
// ----------
//   J     : base symmetric coupling
//   J'    : amplitude of oscillating non-reciprocal part
//   ω₀    : oscillation frequency
//   D     : noise strength
//   dt    : time step (should satisfy dt ≪ 2π/ω₀)
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨e^{iθ}⟩|     order parameter
//   C(r)  = spatial correlation along x
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   e5_timeseries.csv    step, t, m
//   e5_correlation.csv   r, C_r
//   e5_jprime_scan.csv   L, J_prime, omega0, m_mean, m_std, binder
//   e5_size_scan.csv     L, J_prime, omega0, m_mean, m_std, binder
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
// § 2  Drift  f(θ, t)
// ─────────────────────────────────────────────────────────────

/// f_i = J  Σ_{4 nn}   sin(θ_j − θ_i)       [symmetric base]
///       + J'cos(ω₀t) [sin(θ_{rt}−θ_i) + sin(θ_{up}−θ_i)
///                     −sin(θ_{lf}−θ_i) − sin(θ_{dn}−θ_i)]  [antisymmetric NR]
fn drift(lat: &XYLattice, j: f64, j_prime: f64, omega0: f64, t: f64) -> Vec<f64> {
    let l      = lat.l;
    let jnr_t  = j_prime * (omega0 * t).cos();   // time-modulated non-reciprocal strength
    let mut f  = vec![0.0f64; l * l];
    for i in 0..l {
        for jj in 0..l {
            let k  = lat.idx(i, jj);
            let th = lat.theta[k];
            let up  = lat.get(lat.w(i, 1),  jj);
            let dn  = lat.get(lat.w(i,-1),  jj);
            let rt  = lat.get(i, lat.w(jj, 1));
            let lf  = lat.get(i, lat.w(jj,-1));
            // symmetric XY coupling (all 4 neighbours)
            f[k]  = j * ((up-th).sin() + (dn-th).sin()
                        + (rt-th).sin() + (lf-th).sin());
            // oscillating non-reciprocal: +fwd −bwd
            //   fwd = {right, up},  bwd = {left, down}
            f[k] += jnr_t * ((rt-th).sin() + (up-th).sin()
                             -(lf-th).sin() - (dn-th).sin());
        }
    }
    f
}

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut XYLattice, j: f64, j_prime: f64, omega0: f64,
           t: f64, d: f64, dt: f64, rng: &mut SmallRng) {
    let f  = drift(lat, j, j_prime, omega0, t);
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

struct RunResult { l: usize, j_prime: f64, omega0: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, j_prime: f64, omega0: f64, d: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = XYLattice::new_random(l, &mut rng);
    let n_tr  = (t_trans / dt).round() as usize;
    let n_me  = (t_meas  / dt).round() as usize;

    let mut t = 0.0f64;
    for _ in 0..n_tr {
        em_step(&mut lat, j, j_prime, omega0, t, d, dt, &mut rng);
        t += dt;
    }
    if verbose {
        println!("[E5] L={l} J={j} J'={j_prime:.3} ω₀={omega0:.3} D={d:.2} \
                  transient done");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, j_prime, omega0, t, d, dt, &mut rng);
        t += dt;
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, j_prime, omega0, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, j_prime: f64, omega0: f64,
              m_mean: f64, m_std: f64, binder: f64 }

fn jprime_scan(l: usize, jp_arr: &[f64], omega0: f64, j: f64, d: f64,
               dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    jp_arr.iter().map(|&jp| {
        let r = run(l, j, jp, omega0, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} J'={jp:.3} ω₀={omega0:.2}  m={:.4}  U_L={:.4}",
                 r.m_mean, r.binder);
        SRow { l, j_prime: jp, omega0, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, j_prime: f64, omega0: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, j_prime, omega0, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} J'={j_prime:.3} ω₀={omega0:.2}  m={:.4}  U_L={:.4}",
                 r.m_mean, r.binder);
        SRow { l, j_prime, omega0, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
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
    writeln!(w, "L,J_prime,omega0,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.j_prime, r.omega0, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: L=32, J=1, J'=0.5, ω₀=1.0, D=0.5
    let res = run(32, 1.0, 0.5, 1.0, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("e5_timeseries.csv", &res.m_ts, res.dt, me);
    write_corr("e5_correlation.csv", &res.c_r);

    // J' scan: J'=0 (no non-reciprocity) → J'=1.0 (strong oscillating NR)
    let jp_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.1).collect();
    let jr = jprime_scan(32, &jp_arr, 1.0, 1.0, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan("e5_jprime_scan.csv", &jr);

    // finite-size scan at J'=0.5, ω₀=1.0
    let lr = size_scan(&[16,24,32,48], 1.0, 0.5, 1.0, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("e5_size_scan.csv", &lr);
}
