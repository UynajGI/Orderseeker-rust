// ============================================================
// model_b3_xy_paired_active.rs  —  Model B3  🔴 P1
// XY Model + Paired Active Noise  (centre-of-mass conserving)
// ============================================================
// Model family : B族 — XY模型 / O(2)场 + 新型非平衡驱动
// Priority     : 🔴 P1  (LRO probability ≥ 70%)
//
// Physics motivation
// ------------------
//   Keta & Henkes (2024) showed that paired active stresses with
//   zero net force (centre-of-mass conservation) generate
//   super-uniform fluctuations S(q) ~ q² in 2D elastic bodies,
//   stabilising quasi-long-range order beyond the Mermin-Wagner limit.
//   Here we transplant this mechanism to the pure rotational degree of
//   freedom (XY phase field): each nearest-neighbour bond (i,j) carries
//   an independent white-noise "active torque" η_{ij}(t) = −η_{ji}(t),
//   so that Σ_j η_{ij} sums to zero for each site (Newton's 3rd law).
//   The resulting noise spectrum is ∝ q² in Fourier space, which
//   exactly cancels the IR divergence responsible for MW theorem.
//
// Equation of motion  (Euler-Maruyama SDE, a = 1)
// -----------------------------------------------
//   dθ_i = [ J Σ_{⟨ij⟩} sin(θ_j − θ_i) ] dt
//           + √(2D  dt) ξ_i                     [bulk thermal noise]
//           + Σ_{⟨ij⟩} dB_{ij}                  [paired active noise]
//
//   dB_{ij} = √(2 D_a dt) ζ_{ij},  ζ_{ij} = −ζ_{ji} ~ N(0,1)
//             (antisymmetric bond noise, drawn once per step per bond)
//
//   The active contribution for site i is:
//     Σ_{j∈nb(i)} dB_{ij} = √(2 D_a dt) Σ_{j∈nb(i)} ζ_{ij}
//   where ζ_{ij} is drawn fresh each step and ζ_{ji} = −ζ_{ij}.
//
//   Implementation: iterate over all horizontal bonds then vertical
//   bonds; for each bond (i,j) draw one N(0,1) variate ζ and add
//   +√(2 D_a dt) ζ to site i, subtract from site j.
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨e^{iθ}⟩|     magnetisation
//   C(r)  = spatial correlation
//   U_L   = Binder cumulant
//   n_def = positive topological defect count
//
// Output files
// ------------
//   b3_timeseries.csv   step, t, m, n_defects
//   b3_correlation.csv  r, C_r
//   b3_da_scan.csv      L, D_a, m_mean, m_std, binder
//   b3_size_scan.csv    L, D_a, m_mean, m_std, binder
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
// § 2  One SDE step with paired active noise
// ─────────────────────────────────────────────────────────────

fn em_step_paired(
    lat:  &mut XYLattice,
    j:    f64,
    d:    f64,    // bulk thermal noise strength
    d_a:  f64,    // active (paired) noise strength
    dt:   f64,
    rng:  &mut SmallRng,
) {
    let l  = lat.l;
    let n  = l * l;
    let nd = Normal::new(0.0f64, 1.0).unwrap();

    // ── deterministic XY drift ──────────────────────────────
    let mut dtheta = vec![0.0f64; n];
    for i in 0..l {
        for jj in 0..l {
            let k  = lat.idx(i, jj);
            let th = lat.theta[k];
            let up  = lat.get(lat.w(i, 1),  jj);
            let dn  = lat.get(lat.w(i,-1),  jj);
            let rt  = lat.get(i, lat.w(jj, 1));
            let lf  = lat.get(i, lat.w(jj,-1));
            dtheta[k] = j * ((up-th).sin() + (dn-th).sin()
                            + (rt-th).sin() + (lf-th).sin()) * dt;
        }
    }

    // ── bulk thermal noise: √(2D dt) ξ_i ───────────────────
    let sd_bulk = (2.0 * d * dt).sqrt();
    for k in 0..n { dtheta[k] += sd_bulk * rng.sample(nd); }

    // ── paired active noise over all bonds ──────────────────
    // Each bond contributes once: +ζ to i, -ζ to j.
    // Horizontal bonds: (i,j) — (i,j+1)
    // Vertical   bonds: (i,j) — (i+1,j)
    let sd_a = (2.0 * d_a * dt).sqrt();
    for i in 0..l {
        for jj in 0..l {
            // horizontal bond (i,jj) -- (i,jj+1)
            {
                let zeta = sd_a * rng.sample(nd);
                let k0 = lat.idx(i, jj);
                let k1 = lat.idx(i, lat.w(jj, 1));
                dtheta[k0] += zeta;
                dtheta[k1] -= zeta;
            }
            // vertical bond (i,jj) -- (i+1,jj)
            {
                let zeta = sd_a * rng.sample(nd);
                let k0 = lat.idx(i,            jj);
                let k1 = lat.idx(lat.w(i, 1),  jj);
                dtheta[k0] += zeta;
                dtheta[k1] -= zeta;
            }
        }
    }

    // ── apply increments ────────────────────────────────────
    for k in 0..n { lat.theta[k] += dtheta[k]; }
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

// ── topological defect count (winding number on plaquettes) ──
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

struct RunResult { l: usize, d_a: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>,
                   ndef_ts: Vec<usize>, dt: f64 }

fn run(l: usize, j: f64, d: f64, d_a: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = XYLattice::new_random(l, &mut rng);
    let n_tr  = (t_trans / dt).round() as usize;
    let n_me  = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step_paired(&mut lat, j, d, d_a, dt, &mut rng); }
    if verbose {
        println!("[B3] L={l} J={j} D={d:.2} D_a={d_a:.2} \
                  transient={n_tr} steps done");
    }

    let mut m_ts    = Vec::new();
    let mut ndef_ts = Vec::new();
    for s in 0..n_me {
        em_step_paired(&mut lat, j, d, d_a, dt, &mut rng);
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
    RunResult { l, d_a, m_mean, m_std, binder: bdr, c_r, m_ts, ndef_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 5  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, d_a: f64, m_mean: f64, m_std: f64, binder: f64 }

fn da_scan(l: usize, da_arr: &[f64], j: f64, d: f64,
           dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    da_arr.iter().map(|&da| {
        let r = run(l, j, d, da, dt, tt, tm, 20, seed, false);
        println!("  L={l} D_a={da:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, d_a: da, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, d: f64, d_a: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, d, d_a, dt, tt, tm, 20, seed, false);
        println!("  L={l} D_a={d_a:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, d_a, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
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
    writeln!(w, "L,D_a,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l,r.d_a,r.m_mean,r.m_std,r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 7  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: L=32, J=1, D=0.2, D_a=0.5
    let res = run(32, 1.0, 0.2, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("b3_timeseries.csv", &res, me);
    write_corr("b3_correlation.csv", &res.c_r);

    // D_a scan: D_a=0 → equilibrium XY (no LRO); D_a>0 → test LRO
    let da_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.1).collect();
    let dr = da_scan(32, &da_arr, 1.0, 0.2, 0.01, 15.0, 60.0, 0);
    write_scan("b3_da_scan.csv", &dr);

    // finite-size scan at D_a=0.5
    let lr = size_scan(&[16,24,32,48], 1.0, 0.2, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("b3_size_scan.csv", &lr);
}
