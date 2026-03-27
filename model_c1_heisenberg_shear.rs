// ============================================================
// model_c1_heisenberg_shear.rs  —  Model C1  🔴 P1
// O(3) Heisenberg Model + Uniform Shear Flow
// ============================================================
// Model family : C族 — O(N≥3)矢量场 + 各类非平衡机制
// Priority     : 🔴 P1  (LRO probability ≥ 70%)
//
// Physics motivation
// ------------------
//   Nakano & Sasa (2021) proved via large-N field theory that uniform
//   shear flow suppresses IR divergences of ALL Goldstone modes via
//   |k_x|^{-2/3}.  For O(3) (Heisenberg), breaking O(3) → O(2) leaves
//   2 Goldstone modes; both are suppressed by shear, so LRO is expected.
//   This is the first lattice numerical test of shear-induced LRO for
//   an O(3) order parameter.
//
// Equation of motion  (Euler-Maruyama SDE on S²)
// -----------------------------------------------
//   d n_i = P_⊥(n_i) [ J Σ_{j∈NN} n_j  +  γ̇ y_i  (n_{i,j+1} − n_{i,j-1})/2 ] dt
//           + √(2D dt) P_⊥(n_i) ξ_i
//   then normalise: n_i ← n_i / |n_i|
//
//   P_⊥(n) = I − n n^T  is the projector onto the tangent space of S².
//   ξ_i ~ N(0,I_3) independent 3D Gaussian noise.
//   The shear term advects the spin field with velocity v_x = γ̇ y_i.
//
//   Note: we use the XY-like nonlinear form for the coupling:
//     f_i = J Σ_{j∈NN} n_j  projected to tangent space.
//   This gives the correct SO(3)-invariant dynamics.
//
// Observables (→ CSV)
// -------------------
//   m(t)   = |⟨n⟩| = |Σ n_i / N|    vector order parameter
//   C(r)   = ⟨n_0 · n_r⟩_x           spatial correlation along x
//   U_L    = Binder cumulant (vector version)
//
// Output files
// ------------
//   c1_timeseries.csv    step, t, m
//   c1_correlation.csv   r, C_r
//   c1_shear_scan.csv    L, gamma_dot, m_mean, m_std, binder
//   c1_size_scan.csv     L, gamma_dot, m_mean, m_std, binder
//
// Cargo deps: rand = "0.8" (features=["small_rng"]), rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, UnitSphere};
use std::fs::File;
use std::io::{BufWriter, Write};

// ─────────────────────────────────────────────────────────────
// § 1  Lattice (O(3) unit vectors)
// ─────────────────────────────────────────────────────────────

type Vec3 = [f64; 3];

struct HeisenbergLattice {
    l:   usize,
    spn: Vec<Vec3>,   // unit 3-vectors
}

#[inline]
fn dot(a: &Vec3, b: &Vec3) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }

#[inline]
fn add3(a: Vec3, b: Vec3) -> Vec3 { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }

#[inline]
fn sub3(a: Vec3, b: Vec3) -> Vec3 { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }

#[inline]
fn scale3(s: f64, v: Vec3) -> Vec3 { [s*v[0], s*v[1], s*v[2]] }

/// Project v onto tangent space of n: P_⊥(n) v = v − (v·n) n
#[inline]
fn proj_tangent(n: &Vec3, v: Vec3) -> Vec3 {
    let c = dot(n, &v);
    sub3(v, scale3(c, *n))
}

/// Normalise to unit vector
#[inline]
fn normalise(v: Vec3) -> Vec3 {
    let r = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if r < 1e-30 { [1.0,0.0,0.0] } else { [v[0]/r, v[1]/r, v[2]/r] }
}

impl HeisenbergLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i * self.l + j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64 + d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> Vec3 { self.spn[self.idx(i,j)] }

    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let us = UnitSphere;
        let spn: Vec<Vec3> = (0..l*l).map(|_| {
            let v = rng.sample(us);
            [v[0], v[1], v[2]]
        }).collect();
        HeisenbergLattice { l, spn }
    }
}

// ─────────────────────────────────────────────────────────────
// § 2  Euler-Maruyama step on S²
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut HeisenbergLattice, j: f64, gd: f64,
           d: f64, dt: f64, rng: &mut SmallRng) {
    let l  = lat.l;
    let n  = l * l;
    let nd = Normal::new(0.0f64, 1.0).unwrap();
    let sd = (2.0 * d * dt).sqrt();

    let spn_old = lat.spn.clone();

    for i in 0..l {
        let yi = i as f64;
        for jj in 0..l {
            let k   = lat.idx(i, jj);
            let ni  = spn_old[k];
            let up  = spn_old[lat.idx(lat.w(i, 1),  jj)];
            let dn  = spn_old[lat.idx(lat.w(i,-1),  jj)];
            let rt  = spn_old[lat.idx(i, lat.w(jj, 1))];
            let lf  = spn_old[lat.idx(i, lat.w(jj,-1))];

            // XY coupling: J Σ n_j projected to tangent space
            let coupling = [
                j * (up[0]+dn[0]+rt[0]+lf[0]),
                j * (up[1]+dn[1]+rt[1]+lf[1]),
                j * (up[2]+dn[2]+rt[2]+lf[2]),
            ];
            let drift_xy = proj_tangent(&ni, coupling);

            // shear advection: γ̇ y_i (n_{rt} − n_{lf})/2  projected
            let shear_raw = scale3(gd * yi * 0.5, sub3(rt, lf));
            let drift_sh  = proj_tangent(&ni, shear_raw);

            // noise
            let noise_raw = [sd * rng.sample(nd),
                             sd * rng.sample(nd),
                             sd * rng.sample(nd)];
            let noise = proj_tangent(&ni, noise_raw);

            let new_n = add3(add3(add3(ni,
                                       scale3(dt, drift_xy)),
                                  scale3(dt, drift_sh)),
                             noise);
            lat.spn[k] = normalise(new_n);
        }
    }
    let _ = n; // suppress warning
}

// ─────────────────────────────────────────────────────────────
// § 3  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &HeisenbergLattice) -> f64 {
    let n = lat.l * lat.l;
    let m: Vec3 = lat.spn.iter().fold([0.0;3], |acc, s| add3(acc, *s));
    let nf = n as f64;
    dot(&scale3(1.0/nf, m), &scale3(1.0/nf, m)).sqrt()
}

fn correlation(lat: &HeisenbergLattice, r_max: usize) -> Vec<f64> {
    let l = lat.l;
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
// § 4  Runner
// ─────────────────────────────────────────────────────────────

struct RunResult { l: usize, gd: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, gd: f64, d: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = HeisenbergLattice::new_random(l, &mut rng);
    let n_tr = (t_trans / dt).round() as usize;
    let n_me = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat, j, gd, d, dt, &mut rng); }
    if verbose {
        println!("[C1] L={l} J={j} γ̇={gd:.3} D={d:.2} (O(3) Heisenberg+shear) \
                  transient done");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat, j, gd, d, dt, &mut rng);
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, gd, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 5  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, gd: f64, m_mean: f64, m_std: f64, binder: f64 }

fn shear_scan(l: usize, gd_arr: &[f64], j: f64, d: f64,
              dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    gd_arr.iter().map(|&gd| {
        let r = run(l, j, gd, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} γ̇={gd:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, gd, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, gd: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l, j, gd, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} γ̇={gd:.3}  m={:.4}  U_L={:.4}", r.m_mean, r.binder);
        SRow { l, gd, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
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
    writeln!(w, "L,gamma_dot,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.4},{:.6},{:.6},{:.6}",
                 r.l, r.gd, r.m_mean, r.m_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 7  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: L=24, γ̇=0.5, J=1, D=0.5
    let res = run(24, 1.0, 0.5, 0.5, 0.01, 20.0, 100.0, me, 42, true);
    write_ts("c1_timeseries.csv", &res.m_ts, res.dt, me);
    write_corr("c1_correlation.csv", &res.c_r);

    // shear scan γ̇ ∈ [0, 2.0]
    let gd_arr: Vec<f64> = (0..=10).map(|k| k as f64 * 0.2).collect();
    let sr = shear_scan(24, &gd_arr, 1.0, 0.5, 0.01, 15.0, 60.0, 0);
    write_scan("c1_shear_scan.csv", &sr);

    // finite-size scan at γ̇=0.5
    let lr = size_scan(&[12,16,24,32], 1.0, 0.5, 0.5, 0.01, 15.0, 60.0, 1);
    write_scan("c1_size_scan.csv", &lr);
}
