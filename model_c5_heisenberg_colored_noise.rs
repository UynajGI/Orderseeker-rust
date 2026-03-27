// ============================================================
// model_c5_heisenberg_colored_noise.rs  —  Model C5  🟡 P3
// O(3) Heisenberg Model + Isotropic Colored (Exponentially Correlated) Noise
// ============================================================
// Model family : C族 — O(N≥3)矢量场 + 各类非平衡机制
// Priority     : 🟡 P3  (LRO probability: medium ~20–40%)
//
// Physics motivation
// ------------------
//   Model B5 introduced colored noise (exponential correlation time τ_c)
//   to the O(2) XY model.  Model C5 extends this to the O(3) Heisenberg
//   model, where two Goldstone modes (instead of one) are affected.
//   The colored noise is isotropic in the three vector components and
//   does NOT break the O(3) symmetry.
//
//   Each site carries a persistent noise vector η_i(t) ∈ R³ that evolves
//   as an Ornstein-Uhlenbeck process:
//     dη_i = −η_i/τ_c dt + √(2D/τ_c) dW_i   (3D OU)
//   The spin dynamics then become:
//     d n_i = P_⊥(n_i) [J Σ_j n_j + η_i] dt
//   followed by normalisation.
//
//   In the long-τ_c limit, η_i is slowly varying and approaches a
//   quasi-static random field, equivalent to a random anisotropy
//   that breaks O(3) → O(0).  In the short-τ_c limit (τ_c→0),
//   the model reduces to standard equilibrium Heisenberg (no LRO).
//   The interesting regime is intermediate τ_c.
//
// Equation of motion  (Euler-Maruyama SDE on S²)
// -----------------------------------------------
//   dη_i^α = −η_i^α/τ_c dt + √(2D/τ_c dt) ξ_i^α   [OU noise]
//   d n_i   = P_⊥(n_i) [J Σ_{j∈nn(i)} n_j + η_i] dt   [spin SDE]
//   n_i ← n_i / |n_i|                                    [normalise]
//
//   P_⊥(n) v = v − (v·n)n  projects to tangent space of S².
//
// Parameters
// ----------
//   J     : Heisenberg coupling
//   D     : noise variance (of OU process; effective temperature ~ D τ_c)
//   τ_c   : noise correlation time (τ_c=0 → white noise / standard Heisenberg)
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨n⟩|   vector order parameter
//   C(r)  = ⟨n_0 · n_r⟩ spatial correlation along x
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   c5_timeseries.csv    step, t, m
//   c5_correlation.csv   r, C_r
//   c5_tauc_scan.csv     L, tau_c, m_mean, m_std, binder
//   c5_size_scan.csv     L, tau_c, m_mean, m_std, binder
//
// Cargo deps: rand = "0.8" (features=["small_rng"]), rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, UnitSphere};
use std::fs::File;
use std::io::{BufWriter, Write};

// ─────────────────────────────────────────────────────────────
// § 1  Types and helpers for O(3)
// ─────────────────────────────────────────────────────────────

type Vec3 = [f64; 3];

#[inline] fn dot(a: &Vec3, b: &Vec3) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
#[inline] fn add3(a: Vec3, b: Vec3) -> Vec3 { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
#[inline] fn sub3(a: Vec3, b: Vec3) -> Vec3 { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
#[inline] fn scale3(s: f64, v: Vec3) -> Vec3 { [s*v[0], s*v[1], s*v[2]] }

#[inline]
fn proj_tangent(n: &Vec3, v: Vec3) -> Vec3 {
    sub3(v, scale3(dot(n, &v), *n))
}

#[inline]
fn normalise(v: Vec3) -> Vec3 {
    let r = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if r < 1e-30 { [1.0,0.0,0.0] } else { [v[0]/r, v[1]/r, v[2]/r] }
}

// ─────────────────────────────────────────────────────────────
// § 2  Lattice
// ─────────────────────────────────────────────────────────────

struct HeisenbergColoredLattice {
    l:   usize,
    spn: Vec<Vec3>,   // unit 3-vectors on S²
    eta: Vec<Vec3>,   // OU noise state for each site (3D)
}

impl HeisenbergColoredLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> Vec3 { self.spn[self.idx(i,j)] }

    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let us = UnitSphere;
        let spn: Vec<Vec3> = (0..l*l).map(|_| {
            let v = rng.sample(us); [v[0], v[1], v[2]]
        }).collect();
        let eta = vec![[0.0f64; 3]; l*l];  // initialise noise at zero
        HeisenbergColoredLattice { l, spn, eta }
    }
}

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step on S² with OU noise
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut HeisenbergColoredLattice, j: f64,
           d: f64, tau_c: f64, dt: f64, rng: &mut SmallRng) {
    let l  = lat.l;
    let nd = Normal::new(0.0f64, 1.0).unwrap();

    // OU noise update:  dη = −η/τ dt + √(2D/τ dt) ξ
    // For τ_c=0 limit: use white noise directly (avoid division by zero)
    let sd_ou = if tau_c > 1e-12 {
        (2.0 * d / tau_c * dt).sqrt()
    } else {
        (2.0 * d * dt).sqrt()
    };

    for k in 0..l*l {
        if tau_c > 1e-12 {
            for a in 0..3 {
                lat.eta[k][a] += -lat.eta[k][a] / tau_c * dt
                    + sd_ou * rng.sample(nd);
            }
        } else {
            // white noise limit: η is just the noise itself each step
            for a in 0..3 { lat.eta[k][a] = sd_ou * rng.sample(nd); }
        }
    }

    let spn_old = lat.spn.clone();
    let eta_old = lat.eta.clone();

    for i in 0..l {
        for jj in 0..l {
            let k   = lat.idx(i, jj);
            let ni  = spn_old[k];
            let up  = spn_old[lat.idx(lat.w(i, 1),  jj)];
            let dn  = spn_old[lat.idx(lat.w(i,-1),  jj)];
            let rt  = spn_old[lat.idx(i, lat.w(jj, 1))];
            let lf  = spn_old[lat.idx(i, lat.w(jj,-1))];

            // Heisenberg coupling force
            let coupling = [
                j*(up[0]+dn[0]+rt[0]+lf[0]),
                j*(up[1]+dn[1]+rt[1]+lf[1]),
                j*(up[2]+dn[2]+rt[2]+lf[2]),
            ];
            let drift_j = proj_tangent(&ni, coupling);

            // OU noise contribution (projected to tangent space)
            let noise_tang = proj_tangent(&ni, eta_old[k]);

            let new_n = add3(ni, add3(scale3(dt, drift_j), scale3(dt, noise_tang)));
            lat.spn[k] = normalise(new_n);
        }
    }
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &HeisenbergColoredLattice) -> f64 {
    let n  = lat.l * lat.l;
    let m: Vec3 = lat.spn.iter().fold([0.0;3], |acc, s| add3(acc, *s));
    let nf = n as f64;
    (dot(&m, &m)).sqrt() / nf
}

fn correlation(lat: &HeisenbergColoredLattice, r_max: usize) -> Vec<f64> {
    let l  = lat.l;
    let nf = (l * l) as f64;
    let mut c = vec![0.0f64; r_max + 1];
    c[0] = 1.0;
    for r in 1..=r_max {
        c[r] = (0..l).flat_map(|i| (0..l).map(move |j| (i,j)))
            .map(|(i,j)| dot(&lat.get(i,j), &lat.get(i,(j+r)%l)))
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

struct RunResult { l: usize, tau_c: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, d: f64, tau_c: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = HeisenbergColoredLattice::new_random(l, &mut rng);
    let n_tr  = (t_trans / dt).round() as usize;
    let n_me  = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j, d, tau_c, dt, &mut rng); }
    if verbose {
        println!("[C5] L={l} J={j} D={d:.2} τ_c={tau_c:.3} \
                  transient={n_tr} steps done  (O(3)+colored noise)");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, d, tau_c, dt, &mut rng);
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, tau_c, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, tau_c: f64, m_mean: f64, m_std: f64, binder: f64 }

fn tauc_scan(l: usize, tc_arr: &[f64], j: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    tc_arr.iter().map(|&tc| {
        let r = run(l, j, d, tc, dt, tt, tm, 20, seed, false);
        println!("  L={l} τ_c={tc:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, tau_c: tc, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, d: f64, tau_c: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, d, tau_c, dt, tt, tm, 20, seed, false);
        println!("  L={l} τ_c={tau_c:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, tau_c, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
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
    writeln!(w, "L,tau_c,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.tau_c, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: L=24, τ_c=1.0, J=1, D=0.5
    let res = run(24, 1.0, 0.5, 1.0, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("c5_timeseries.csv", &res.m_ts, res.dt, me);
    write_corr("c5_correlation.csv", &res.c_r);

    // τ_c scan: τ_c=0 (white noise / equilibrium Heisenberg) → τ_c=5.0 (slow noise)
    let tc_arr: Vec<f64> = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0].to_vec();
    let tr = tauc_scan(24, &tc_arr, 1.0, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan("c5_tauc_scan.csv", &tr);

    // finite-size scan at τ_c=1.0
    let lr = size_scan(&[12,16,24,32], 1.0, 0.5, 1.0, 0.01, 15.0, 60.0, 1);
    write_scan("c5_size_scan.csv", &lr);
}
