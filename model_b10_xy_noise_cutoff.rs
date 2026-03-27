// ============================================================
// model_b10_xy_noise_cutoff.rs  —  Model B10  🟡 P3
// XY Model + Low-frequency Noise Cutoff  (D(k)=0 for |k|<k_c)
// ============================================================
// Model family : B族 — XY模型 / O(2)场 + 新型非平衡驱动
// Priority     : 🟡 P3  (LRO probability: medium-high ~30–50%)
//
// Physics motivation
// ------------------
//   The Mermin-Wagner theorem relies on the divergence of long-wave
//   (IR) thermal fluctuations.  If the noise is spectrally filtered
//   so that long-wavelength modes (|k| < k_c) receive no thermal
//   excitation, the IR divergence is cut off and LRO may survive.
//   This directly simulates the "centre-of-mass bath temperature → 0"
//   core mechanism of Maire & Plati (2024) on a phase-only model.
//
//   Implementation strategy (real-space lattice):
//   Rather than a sharp Fourier-space cutoff (expensive FFT per step),
//   we approximate the filtered noise by computing the noise increment
//   in Fourier space, zeroing all modes with |k| < k_c, and
//   transforming back via a DFT (O(N²) per step, feasible for L≤32).
//
//   The DFT is performed over the L×L lattice using direct summation.
//   Modes are indexed by (kx, ky) ∈ {0,...,L-1}².  The physical
//   wavenumber magnitude is |k| = 2π/L · √(kx²+ky²) (with aliasing:
//   kx → kx if kx ≤ L/2, else kx − L).
//
// Equation of motion  (Euler-Maruyama SDE)
// -----------------------------------------
//   dθ_i = J Σ_{⟨ij⟩} sin(θ_j − θ_i) dt  +  η̃_i dt
//
//   where η̃ is the filtered noise:
//     1. Draw raw noise ξ_k ~ N(0,1) for each site k.
//     2. DFT: Ξ̂(q) = Σ_k ξ_k e^{-iq·r_k}  / N
//     3. Zero all modes with |q̃| < k_c:  Ξ̂(q) ← 0  if |q̃| < k_c
//     4. iDFT back to real space, scale by √(2D dt)
//
// Parameters
// ----------
//   J    : XY coupling
//   D    : noise amplitude (for |k| ≥ k_c modes)
//   k_c  : cutoff wavenumber fraction (0 ≤ k_c ≤ 1, in units of π)
//          k_c = 0 → no cutoff (standard XY), k_c > 0 → IR suppressed
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨e^{iθ}⟩|     order parameter
//   C(r)  = spatial correlation along x
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   b10_timeseries.csv    step, t, m
//   b10_correlation.csv   r, C_r
//   b10_kc_scan.csv       L, k_c, m_mean, m_std, binder
//   b10_size_scan.csv     L, k_c, m_mean, m_std, binder
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
// § 2  Filtered noise generation via direct DFT  (O(N²))
// ─────────────────────────────────────────────────────────────

/// Generate filtered noise for all L×L sites.
///
/// Steps:
///  1. Draw raw real-space noise ξ_k ~ N(0,1) for each site.
///  2. 2D DFT: Ξ̂(qx,qy) = Σ_k ξ_k exp(-2πi(qx*cx+qy*cy)/L) / N
///  3. Zero modes where √(q̃x²+q̃y²) / (L/2) < k_c_frac
///     (q̃x = qx if qx ≤ L/2 else qx-L, similarly for y)
///  4. 2D iDFT back to real space, return real part scaled by √(2D dt)
///
/// k_c_frac ∈ [0,1]: fraction of Nyquist; 0 = no cutoff, 1 = all zeroed.
fn filtered_noise(l: usize, d: f64, dt: f64, k_c_frac: f64,
                  rng: &mut SmallRng) -> Vec<f64> {
    let n    = l * l;
    let nf   = n as f64;
    let nd   = Normal::new(0.0f64, 1.0).unwrap();

    // raw noise in real space
    let xi: Vec<f64> = (0..n).map(|_| rng.sample(nd)).collect();

    // 2D DFT (complex result stored as (re, im) pairs)
    let mut xi_hat_re = vec![0.0f64; n];
    let mut xi_hat_im = vec![0.0f64; n];

    for qy in 0..l {
        for qx in 0..l {
            let q_idx = qy * l + qx;
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for cy in 0..l {
                for cx in 0..l {
                    let r_idx = cy * l + cx;
                    let phase = -2.0 * PI * ((qx*cx + qy*cy) as f64) / l as f64;
                    re += xi[r_idx] * phase.cos();
                    im += xi[r_idx] * phase.sin();
                }
            }
            xi_hat_re[q_idx] = re / nf;
            xi_hat_im[q_idx] = im / nf;
        }
    }

    // zero modes below cutoff
    let lh = l as i64 / 2;
    for qy in 0..l {
        for qx in 0..l {
            let q_idx = q_idx(qy, qx, l);
            let qx_t  = if qx as i64 <= lh { qx as f64 } else { qx as f64 - l as f64 };
            let qy_t  = if qy as i64 <= lh { qy as f64 } else { qy as f64 - l as f64 };
            let k_norm = (qx_t*qx_t + qy_t*qy_t).sqrt() / (l as f64 / 2.0);
            if k_norm < k_c_frac {
                xi_hat_re[q_idx] = 0.0;
                xi_hat_im[q_idx] = 0.0;
            }
        }
    }

    // iDFT back to real space
    let sd = (2.0 * d * dt).sqrt();
    let mut noise = vec![0.0f64; n];
    for cy in 0..l {
        for cx in 0..l {
            let r_idx = cy * l + cx;
            let mut val = 0.0f64;
            for qy in 0..l {
                for qx in 0..l {
                    let q_idx = q_idx(qy, qx, l);
                    let phase = 2.0 * PI * ((qx*cx + qy*cy) as f64) / l as f64;
                    val += xi_hat_re[q_idx] * phase.cos()
                         - xi_hat_im[q_idx] * phase.sin();
                }
            }
            noise[r_idx] = sd * val;  // already includes √(2D dt) factor
        }
    }
    noise
}

#[inline]
fn q_idx(qy: usize, qx: usize, l: usize) -> usize { qy * l + qx }

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step with filtered noise
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut XYLattice, j: f64, d: f64, k_c: f64,
           dt: f64, rng: &mut SmallRng) {
    let l = lat.l;
    let n = l * l;

    // XY drift
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
                         + (rt-th).sin() + (lf-th).sin()) * dt;
        }
    }

    // filtered noise
    let noise = filtered_noise(l, d, dt, k_c, rng);
    for k in 0..n { lat.theta[k] += dth[k] + noise[k]; }
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

struct RunResult { l: usize, k_c: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, d: f64, k_c: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = XYLattice::new_random(l, &mut rng);
    let n_tr  = (t_trans / dt).round() as usize;
    let n_me  = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j, d, k_c, dt, &mut rng); }
    if verbose {
        println!("[B10] L={l} J={j} D={d:.2} k_c={k_c:.3} \
                  transient={n_tr} steps done");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, d, k_c, dt, &mut rng);
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, k_c, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, k_c: f64, m_mean: f64, m_std: f64, binder: f64 }

fn kc_scan(l: usize, kc_arr: &[f64], j: f64, d: f64,
           dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    kc_arr.iter().map(|&kc| {
        let r = run(l, j, d, kc, dt, tt, tm, 20, seed, false);
        println!("  L={l} k_c={kc:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, k_c: kc, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, d: f64, k_c: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, d, k_c, dt, tt, tm, 20, seed, false);
        println!("  L={l} k_c={k_c:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, k_c, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
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
    writeln!(w, "L,k_c,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.k_c, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    // NOTE: L≤16 recommended for interactive runs (DFT is O(N²) per step).
    //       Use L=16 for scans, L=24 for single run.
    let me = 20usize;
    // single run: L=16, k_c=0.3 (filter ~9% of modes by area), D=0.5
    let res = run(16, 1.0, 0.5, 0.3, 0.01, 20.0, 80.0, me, 42, true);
    write_ts("b10_timeseries.csv", &res.m_ts, res.dt, me);
    write_corr("b10_correlation.csv", &res.c_r);

    // k_c scan: k_c=0 (standard XY, no cutoff) → k_c=0.8 (heavy filtering)
    let kc_arr: Vec<f64> = (0..=8).map(|k| k as f64 * 0.1).collect();
    let kr = kc_scan(16, &kc_arr, 1.0, 0.5, 0.01, 10.0, 40.0, 0);
    write_scan("b10_kc_scan.csv", &kr);

    // finite-size scan at k_c=0.3
    let lr = size_scan(&[8,12,16,24], 1.0, 0.5, 0.3, 0.01, 10.0, 40.0, 1);
    write_scan("b10_size_scan.csv", &lr);
}
