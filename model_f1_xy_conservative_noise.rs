// ============================================================
// model_f1_xy_conservative_noise.rs  —  Model F1  🔴 P1
// XY Model + Conservative (Bond-Divergence) Noise
// ============================================================
// Model family : F族 — 噪声谱工程
// Priority     : 🔴 P1  (LRO probability: VERY HIGH)
//
// Physics motivation
// ------------------
//   The Mermin-Wagner theorem requires ∫ d²k / k² to diverge (2D).
//   A noise spectrum S(k) ∝ k² (instead of S ∝ const for white noise)
//   exactly cancels this IR divergence, stabilising true LRO.
//
//   Implementation: bond noise ξ_{bond} in the divergence form
//     noise_i = Σ_{μ=x,y} (η_{i+μ̂} − η_{i−μ̂})
//   where η_{bond} ~ N(0,1) independent scalars per bond per step.
//   In Fourier space this gives the noise spectrum:
//     S̃(k) ∝ Σ_μ sin²(k_μ/2) ~ |k|²  as k→0
//   which precisely removes the IR divergence.
//
//   This is the most direct lattice realisation of the Keta-Henkes
//   mechanism (q² noise) transplanted to the XY phase field.
//
// Equation of motion  (Euler-Maruyama SDE, a = 1)
// -----------------------------------------------
//   dθ_i = J Σ_{⟨ij⟩} sin(θ_j − θ_i) dt
//          + Σ_{μ} (η_{i+μ̂,μ} − η_{i,μ}) √(2 D_c dt)
//
//   η_{b,μ} : independent N(0,1) for each bond b=(i,μ̂) per step.
//   For site i:
//     Δθ_i^noise = √(2 D_c dt) [
//          (ζ_{i,+x} − ζ_{i,-x})    [x-bond difference]
//         +(ζ_{i,+y} − ζ_{i,-y})    [y-bond difference]
//     ]
//   where ζ_{i,+μ} is the bond noise on the bond from i to i+μ̂.
//
//   Exact implementation: draw one ζ per bond per step, add/subtract
//   at the two endpoints.
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨e^{iθ}⟩|     order parameter
//   C(r)  = spatial correlation
//   U_L   = Binder cumulant
//   n_def = topological defect count
//
// Output files
// ------------
//   f1_timeseries.csv    step, t, m, n_defects
//   f1_correlation.csv   r, C_r
//   f1_dc_scan.csv       L, D_c, m_mean, m_std, binder
//   f1_size_scan.csv     L, D_c, m_mean, m_std, binder
//
// Cargo.toml deps:  rand = "0.8" (features=["small_rng"])
//                   rand_distr = "0.4"
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
        let d = Uniform::new(0.0f64, 2.0 * PI);
        XYLattice { l, theta: (0..l*l).map(|_| rng.sample(d)).collect() }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  One SDE step with conservative bond noise
// ─────────────────────────────────────────────────────────────

/// One Euler-Maruyama step:
///   1. Compute XY drift and accumulate into dtheta.
///   2. For each horizontal bond (i,j)→(i,j+1): draw ζ ~ N(0,1),
///         add +√(2 D_c dt) ζ to site (i,j)
///         sub −√(2 D_c dt) ζ from site (i,j+1)
///   3. Same for vertical bonds.
///   No bulk (white) noise in the pure F1 model  (D_bulk = 0 by default).
///   If desired, set d_bulk > 0 to add a regular thermal bath as well.
fn em_step(
    lat:     &mut XYLattice,
    j_coup:  f64,
    d_bulk:  f64,   // bulk white-noise strength (can be 0)
    d_c:     f64,   // conservative bond-noise strength
    dt:      f64,
    rng:     &mut SmallRng,
) {
    let l  = lat.l;
    let n  = l * l;
    let nd = Normal::new(0.0f64, 1.0).unwrap();

    // ── XY drift ────────────────────────────────────────────
    let mut dth = vec![0.0f64; n];
    for i in 0..l {
        for jj in 0..l {
            let k  = lat.idx(i, jj);
            let th = lat.theta[k];
            let up  = lat.get(lat.w(i, 1),  jj);
            let dn  = lat.get(lat.w(i,-1),  jj);
            let rt  = lat.get(i, lat.w(jj, 1));
            let lf  = lat.get(i, lat.w(jj,-1));
            dth[k] = j_coup * ((up-th).sin() + (dn-th).sin()
                              + (rt-th).sin() + (lf-th).sin()) * dt;
        }
    }

    // ── bulk white noise (optional) ─────────────────────────
    if d_bulk > 0.0 {
        let sd = (2.0 * d_bulk * dt).sqrt();
        for k in 0..n { dth[k] += sd * rng.sample(nd); }
    }

    // ── conservative bond noise ─────────────────────────────
    //   noise spectrum S(k) ∝ |k|²  → removes IR divergence
    let sd_c = (2.0 * d_c * dt).sqrt();
    // horizontal bonds (i,j)—(i,j+1)
    for i in 0..l {
        for jj in 0..l {
            let zeta = sd_c * rng.sample(nd);
            let k0   = lat.idx(i, jj);
            let k1   = lat.idx(i, lat.w(jj, 1));
            dth[k0] += zeta;
            dth[k1] -= zeta;
        }
    }
    // vertical bonds (i,j)—(i+1,j)
    for i in 0..l {
        for jj in 0..l {
            let zeta = sd_c * rng.sample(nd);
            let k0   = lat.idx(i,           jj);
            let k1   = lat.idx(lat.w(i, 1), jj);
            dth[k0] += zeta;
            dth[k1] -= zeta;
        }
    }

    for k in 0..n { lat.theta[k] += dth[k]; }
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

#[inline]
fn adiff(a: f64, b: f64) -> f64 {
    let d = b - a;
    d - (2.0 * PI) * (d / (2.0 * PI)).round()
}

fn count_defects(lat: &XYLattice) -> (usize, usize) {
    let l = lat.l;
    let (mut pos, mut neg) = (0, 0);
    for i in 0..l {
        let ip = (i+1) % l;
        for j in 0..l {
            let jp = (j+1) % l;
            let d1 = adiff(lat.get(i,  j),  lat.get(ip, j));
            let d2 = adiff(lat.get(ip, j),  lat.get(ip, jp));
            let d3 = adiff(lat.get(ip, jp), lat.get(i,  jp));
            let d4 = adiff(lat.get(i,  jp), lat.get(i,  j));
            match ((d1+d2+d3+d4)/(2.0*PI)).round() as i32 {
                 1 => pos += 1,
                -1 => neg += 1,
                _  => {}
            }
        }
    }
    (pos, neg)
}

fn mean_f(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
fn std_f(v: &[f64]) -> f64 {
    let mu = mean_f(v);
    (v.iter().map(|x| (x-mu).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

// ─────────────────────────────────────────────────────────────
// § 4  Runner
// ─────────────────────────────────────────────────────────────

struct RunResult { l: usize, d_c: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>,
                   ndef_ts: Vec<usize>, dt: f64 }

fn run(l: usize, j: f64, d_bulk: f64, d_c: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = XYLattice::new_random(l, &mut rng);
    let n_tr  = (t_trans / dt).round() as usize;
    let n_me  = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j, d_bulk, d_c, dt, &mut rng); }
    if verbose {
        println!("[F1] L={l} J={j} D_bulk={d_bulk:.2} D_c={d_c:.2} \
                  transient={n_tr} steps done");
    }

    let mut m_ts    = Vec::new();
    let mut ndef_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, d_bulk, d_c, dt, &mut rng);
        if s % mevery == 0 {
            m_ts.push(order_param(&lat));
            ndef_ts.push(count_defects(&lat).0);
        }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    let nd_avg = ndef_ts.iter().sum::<usize>() as f64 / ndef_ts.len() as f64;
    if verbose {
        println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}  \
                  avg_def = {nd_avg:.1}");
    }
    RunResult { l, d_c, m_mean, m_std, binder: bdr, c_r, m_ts, ndef_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 5  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, d_c: f64, m_mean: f64, m_std: f64, binder: f64 }

/// D_c scan: D_c=0 → no conservative noise (expect no LRO); D_c>0 → LRO onset.
fn dc_scan(l: usize, dc_arr: &[f64], j: f64, d_bulk: f64,
           dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    dc_arr.iter().map(|&dc| {
        let r = run(l, j, d_bulk, dc, dt, tt, tm, 20, seed, false);
        println!("  L={l} D_c={dc:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, d_c: dc, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, d_bulk: f64, d_c: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, d_bulk, d_c, dt, tt, tm, 20, seed, false);
        println!("  L={l} D_c={d_c:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, d_c, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

// ─────────────────────────────────────────────────────────────
// § 6  CSV helpers
// ─────────────────────────────────────────────────────────────

fn write_ts(path: &str, res: &RunResult, me: usize) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "step,t,m,n_defects").unwrap();
    for (k,(&m,&nd)) in res.m_ts.iter().zip(&res.ndef_ts).enumerate() {
        let s = k*me;
        writeln!(w, "{},{:.6},{:.8},{}", s, s as f64*res.dt, m, nd).unwrap();
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
    writeln!(w, "L,D_c,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.d_c, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 7  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // pure conservative noise: D_bulk=0, D_c=0.5
    let res = run(32, 1.0, 0.0, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("f1_timeseries.csv", &res, me);
    write_corr("f1_correlation.csv", &res.c_r);

    // D_c scan: D_c=0 (white noise, no LRO) → D_c=1.0 (strong conserv. noise)
    let dc_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.1).collect();
    let dr = dc_scan(32, &dc_arr, 1.0, 0.0, 0.01, 15.0, 60.0, 0);
    write_scan("f1_dc_scan.csv", &dr);

    // finite-size scan at D_c=0.5, D_bulk=0
    let lr = size_scan(&[16,24,32,48], 1.0, 0.0, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("f1_size_scan.csv", &lr);
}
