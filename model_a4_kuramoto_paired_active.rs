// ============================================================
// model_a4_kuramoto_paired_active.rs  —  Model A4  🟠 P2
// Kuramoto Lattice + Paired Active Torque (centre-of-mass conserving)
// ============================================================
// Model family : A族 — Kuramoto振子格子 + 非平衡驱动
// Priority     : 🟠 P2  (LRO probability: medium-high 40–70%)
//
// Physics motivation
// ------------------
//   Keta & Henkes (2024) showed that antisymmetric paired active
//   forces with Σ_i F_i = 0 (centre-of-mass / momentum conservation)
//   produce a noise spectrum S(q) ∝ q² that stabilises LRO.
//   Model A4 transplants this to the Kuramoto phase field:
//   each bond (i,j) carries a short-range coloured active torque
//   F_{ij}^act = −F_{ji}^act  (Newton's 3rd law for torques).
//   The active torque is an Ornstein-Uhlenbeck (OU) process with
//   relaxation time τ_a.  For τ_a → 0 this reduces to model B3
//   (white paired noise); for τ_a → ∞ it approaches a constant bias.
//
// Equation of motion  (Euler-Maruyama SDE + OU noise)
// ----------------------------------------------------
//   dθ_i = [ Ω₀ + J Σ_{⟨ij⟩} sin(θ_j − θ_i) ] dt
//           + √(2D dt) ξ_i
//           + Σ_{⟨ij⟩} F_{ij}^act dt
//
//   dF_{ij} = −(1/τ_a) F_{ij} dt + √(2 D_a/τ_a) dW_{ij}
//   F_{ji}  = −F_{ij}  (antisymmetry enforced every step)
//
//   F_{ij} stored only for bonds (i,j) with i < j (canonical direction).
//
// Observables (→ CSV)
// -------------------
//   r(t)  = |⟨e^{iθ}⟩|     order parameter
//   C(r)  = spatial correlation along x
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   a4_timeseries.csv    step, t, r
//   a4_correlation.csv   r, C_r
//   a4_da_scan.csv       L, D_a, tau_a, r_mean, r_std, binder
//   a4_size_scan.csv     L, D_a, tau_a, r_mean, r_std, binder
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
// § 1  Lattice + OU bond state
// ─────────────────────────────────────────────────────────────

struct KuramotoActiveLattice {
    l:      usize,
    theta:  Vec<f64>,
    omega0: f64,
    // OU active torques on bonds; store 2L² bonds (horizontal + vertical)
    // bond index convention:
    //   horiz[i*l + j] = F_{(i,j)→(i,j+1)}
    //   vert [i*l + j] = F_{(i,j)→(i+1,j)}
    f_h: Vec<f64>,   // horizontal bond OU states
    f_v: Vec<f64>,   // vertical bond OU states
}

impl KuramotoActiveLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i, j)] }

    fn new_random(l: usize, omega0: f64, rng: &mut SmallRng) -> Self {
        let d = Uniform::new(0.0f64, 2.0 * PI);
        KuramotoActiveLattice {
            l, omega0,
            theta: (0..l*l).map(|_| rng.sample(d)).collect(),
            f_h: vec![0.0f64; l*l],
            f_v: vec![0.0f64; l*l],
        }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  One SDE step
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut KuramotoActiveLattice,
           j: f64, d: f64, d_a: f64, tau_a: f64,
           dt: f64, rng: &mut SmallRng) {
    let l  = lat.l;
    let n  = l * l;
    let nd = Normal::new(0.0f64, 1.0).unwrap();

    // ── update OU bond torques first (Euler-Maruyama for OU) ──
    let ou_decay = (-dt / tau_a).exp();   // exact OU decay
    let ou_sd    = (d_a * (1.0 - ou_decay * ou_decay)).sqrt();
    for k in 0..n {
        lat.f_h[k] = ou_decay * lat.f_h[k] + ou_sd * rng.sample(nd);
        lat.f_v[k] = ou_decay * lat.f_v[k] + ou_sd * rng.sample(nd);
    }

    // ── XY drift + bulk noise + active torques ────────────────
    let sd = (2.0 * d * dt).sqrt();
    let mut dth = vec![0.0f64; n];

    for i in 0..l {
        for jj in 0..l {
            let k  = lat.idx(i, jj);
            let th = lat.theta[k];
            let up  = lat.get(lat.w(i, 1),  jj);
            let dn  = lat.get(lat.w(i,-1),  jj);
            let rt  = lat.get(i, lat.w(jj, 1));
            let lf  = lat.get(i, lat.w(jj,-1));
            // drift
            dth[k] = lat.omega0 * dt
                   + j * ((up-th).sin() + (dn-th).sin()
                         + (rt-th).sin() + (lf-th).sin()) * dt;
            // bulk thermal noise
            dth[k] += sd * rng.sample(nd);

            // active torques from bonds incident on site (i,jj):
            //   horiz right:  F_{(i,jj)→(i,jj+1)}  → +F to jj, −F to jj+1
            //   horiz left :  F_{(i,jj-1)→(i,jj)}   = -F_{(i,jj)→(i,jj-1 canonical)}
            let fh_right = lat.f_h[lat.idx(i, jj)];
            let fh_left  = lat.f_h[lat.idx(i, lat.w(jj,-1))];
            let fv_up    = lat.f_v[lat.idx(i,           jj)];
            let fv_down  = lat.f_v[lat.idx(lat.w(i,-1), jj)];
            // site jj is the "origin" of f_h[idx(i,jj)]: receives +F
            // site jj is the "target" of f_h[idx(i,jj-1)]: receives -F
            dth[k] += (fh_right - fh_left + fv_up - fv_down) * dt;
        }
    }
    for k in 0..n { lat.theta[k] += dth[k]; }
}

// ─────────────────────────────────────────────────────────────
// § 3  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &KuramotoActiveLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &KuramotoActiveLattice, r_max: usize) -> Vec<f64> {
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
// § 4  Runner
// ─────────────────────────────────────────────────────────────

struct RunResult { l: usize, d_a: f64, tau_a: f64,
                   r_mean: f64, r_std: f64, binder: f64,
                   c_r: Vec<f64>, r_ts: Vec<f64>, dt: f64 }

fn run(l: usize, omega0: f64, j: f64, d: f64, d_a: f64, tau_a: f64,
       dt: f64, t_trans: f64, t_meas: f64, mevery: usize,
       seed: u64, verbose: bool) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = KuramotoActiveLattice::new_random(l, omega0, &mut rng);
    let n_tr = (t_trans / dt).round() as usize;
    let n_me = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j, d, d_a, tau_a, dt, &mut rng); }
    if verbose {
        println!("[A4] L={l} J={j} D={d:.2} D_a={d_a:.2} τ_a={tau_a:.2} \
                  transient done");
    }

    let mut r_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, d, d_a, tau_a, dt, &mut rng);
        if s % mevery == 0 { r_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let r_mean = mean_f(&r_ts);
    let r_std  = std_f(&r_ts);
    let bdr    = binder(&r_ts);
    if verbose { println!("      r = {r_mean:.4} ± {r_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, d_a, tau_a, r_mean, r_std, binder: bdr, c_r, r_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 5  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, d_a: f64, tau_a: f64,
              r_mean: f64, r_std: f64, binder: f64 }

fn da_scan(l: usize, da_arr: &[f64], tau_a: f64, j: f64, d: f64,
           dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    da_arr.iter().map(|&da| {
        let r = run(l, 0.0, j, d, da, tau_a, dt, tt, tm, 20, seed, false);
        println!("  L={l} D_a={da:.3} τ_a={tau_a:.2}  r={:.4}  U_L={:.4}",
                 r.r_mean, r.binder);
        SRow { l, d_a: da, tau_a, r_mean: r.r_mean, r_std: r.r_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], d_a: f64, tau_a: f64, j: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, 0.0, j, d, d_a, tau_a, dt, tt, tm, 20, seed, false);
        println!("  L={l} D_a={d_a:.3} τ_a={tau_a:.2}  r={:.4}  U_L={:.4}",
                 r.r_mean, r.binder);
        SRow { l, d_a, tau_a, r_mean: r.r_mean, r_std: r.r_std, binder: r.binder }
    }).collect()
}

// ─────────────────────────────────────────────────────────────
// § 6  CSV helpers
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
    writeln!(w, "L,D_a,tau_a,r_mean,r_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.d_a, r.tau_a, r.r_mean, r.r_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 7  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: L=32, D_a=0.5, τ_a=0.5, J=1, D=0.2
    let res = run(32, 0.0, 1.0, 0.2, 0.5, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("a4_timeseries.csv", &res.r_ts, res.dt, me);
    write_corr("a4_correlation.csv", &res.c_r);

    // D_a scan at τ_a=0.5
    let da_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.1).collect();
    let dr = da_scan(32, &da_arr, 0.5, 1.0, 0.2, 0.01, 15.0, 60.0, 0);
    write_scan("a4_da_scan.csv", &dr);

    // finite-size scan at D_a=0.5, τ_a=0.5
    let lr = size_scan(&[16,24,32,48], 0.5, 0.5, 1.0, 0.2, 0.01, 15.0, 60.0, 1);
    write_scan("a4_size_scan.csv", &lr);
}
