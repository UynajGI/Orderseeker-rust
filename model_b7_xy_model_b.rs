// ============================================================
// model_b7_xy_model_b.rs  —  Model B7  🟠 P2
// XY Model B — Conserved Dynamics (Kawasaki-like, ∇·ζ noise)
// ============================================================
// Model family : B族 — XY模型 / O(2)场 + 新型非平衡驱动
// Priority     : 🟠 P2  (LRO probability: medium 40–70%)
//
// Physics motivation
// ------------------
//   "Model B" (Hohenberg-Halperin classification) has conserved
//   order-parameter dynamics: ∂_t θ = ∇²(δF/δθ) + ∇·ζ where ζ is
//   a conserved noise (divergence form).  This modifies the dynamical
//   universality class (z=4 vs z=2 for Model A) and changes the
//   effective noise spectrum to S(k) ∝ k² (from the ∇· operator).
//   The Bassler-Racz mechanism (1994) studied a related anisotropic
//   conserved model.  Here we test the single-temperature isotropic
//   conserved dynamics.
//
// Equation of motion  (Euler-Maruyama SDE, Model B type)
// -------------------------------------------------------
//   The conserved phase-field update is:
//     dθ_i = [−K Δ²θ_i + J Δθ_i] dt + Σ_μ (ζ_{i+μ} − ζ_{i}) √(2D dt)
//   In the pure Model-B limit (no Model-A relaxation term):
//     dθ_i = J Δθ_i dt + Σ_μ (ζ_{i+μ} − ζ_{i}) √(2D dt)
//
//   where Δθ_i = Σ_{j∈NN} (θ_j − θ_i)  is the discrete Laplacian.
//   The nonlinear XY coupling J sin(θ_j−θ_i) is approximated by
//   J(θ_j − θ_i) here, giving a linear Model-B benchmark.
//   A nonlinear version replaces Δθ_i with Σ sin(θ_j−θ_i).
//
//   For the full nonlinear XY Model B:
//     dθ_i = J Σ_{⟨ij⟩} sin(θ_j − θ_i) dt + Σ_μ (ζ_{i+μ} − ζ_i) √(2D dt)
//
// Parameters
// ----------
//   J      : coupling (nonlinear sin)
//   D      : noise strength (applied to divergence-form noise)
//   dt     : time step
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨e^{iθ}⟩|     magnetisation
//   C(r)  = spatial correlation along x
//   U_L   = Binder cumulant
//   n_def = topological defect count
//
// Output files
// ------------
//   b7_timeseries.csv    step, t, m, n_defects
//   b7_correlation.csv   r, C_r
//   b7_d_scan.csv        L, D, m_mean, m_std, binder
//   b7_size_scan.csv     L, D, m_mean, m_std, binder
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

struct ModelBLattice {
    l:     usize,
    theta: Vec<f64>,
}

impl ModelBLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i, j)] }

    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let d = Uniform::new(0.0f64, 2.0 * PI);
        ModelBLattice { l, theta: (0..l*l).map(|_| rng.sample(d)).collect() }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  One step: nonlinear XY Model B
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut ModelBLattice, j: f64, d: f64,
           dt: f64, rng: &mut SmallRng) {
    let l  = lat.l;
    let n  = l * l;
    let nd = Normal::new(0.0f64, 1.0).unwrap();

    // Nonlinear XY drift: J Σ sin(θ_j − θ_i)
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

    // Conserved noise: Σ_μ (ζ_{i+μ̂} − ζ_{i}) with bond noise ζ_{bond}
    let sd = (2.0 * d * dt).sqrt();
    // horizontal bonds
    for i in 0..l {
        for jj in 0..l {
            let zeta = sd * rng.sample(nd);
            let k0   = lat.idx(i, jj);
            let k1   = lat.idx(i, lat.w(jj, 1));
            dth[k0] += zeta;
            dth[k1] -= zeta;
        }
    }
    // vertical bonds
    for i in 0..l {
        for jj in 0..l {
            let zeta = sd * rng.sample(nd);
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

fn order_param(lat: &ModelBLattice) -> f64 {
    let n  = (lat.l * lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re + im*im).sqrt()
}

fn correlation(lat: &ModelBLattice, r_max: usize) -> Vec<f64> {
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

fn count_defects(lat: &ModelBLattice) -> (usize, usize) {
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

struct RunResult { l: usize, d: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>,
                   ndef_ts: Vec<usize>, dt: f64 }

fn run(l: usize, j: f64, d: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = ModelBLattice::new_random(l, &mut rng);
    let n_tr = (t_trans / dt).round() as usize;
    let n_me = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j, d, dt, &mut rng); }
    if verbose {
        println!("[B7] L={l} J={j} D={d:.2} (Model B / conserved noise) \
                  transient done");
    }

    let mut m_ts    = Vec::new();
    let mut ndef_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, d, dt, &mut rng);
        if s % mevery == 0 {
            m_ts.push(order_param(&lat));
            ndef_ts.push(count_defects(&lat).0);
        }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose {
        let nd_avg = ndef_ts.iter().sum::<usize>() as f64 / ndef_ts.len() as f64;
        println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}  \
                  avg_def = {nd_avg:.1}");
    }
    RunResult { l, d, m_mean, m_std, binder: bdr, c_r, m_ts, ndef_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 5  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, d: f64, m_mean: f64, m_std: f64, binder: f64 }

fn d_scan(l: usize, d_arr: &[f64], j: f64,
          dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    d_arr.iter().map(|&dv| {
        let r = run(l, j, dv, dt, tt, tm, 20, seed, false);
        println!("  L={l} D={dv:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, d: dv, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} D={d:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, d, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
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
        writeln!(w, "{},{:.6},{:.8},{}", s, s as f64 * res.dt, m, nd).unwrap();
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
    writeln!(w, "L,D,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.d, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 7  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: J=1, D=0.5 (Model B conserved noise)
    let res = run(32, 1.0, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("b7_timeseries.csv", &res, me);
    write_corr("b7_correlation.csv", &res.c_r);

    // D scan (compare with F1 which has same noise structure but no Model-B coupling)
    let d_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.1).collect();
    let dr = d_scan(32, &d_arr, 1.0, 0.01, 15.0, 60.0, 0);
    write_scan("b7_d_scan.csv", &dr);

    // finite-size scan at D=0.5
    let lr = size_scan(&[16,24,32,48], 1.0, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("b7_size_scan.csv", &lr);
}
