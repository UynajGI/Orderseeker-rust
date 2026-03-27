// ============================================================
// model_c4_o4_shear.rs  —  Model C4  🟠 P2
// O(4) Vector Model + Uniform Shear Flow
// ============================================================
// Model family : C族 — O(N≥3)矢量场 + 各类非平衡机制
// Priority     : 🟠 P2  (LRO probability: medium-high 40–70%)
//
// Physics motivation
// ------------------
//   The large-N field theory (Nakano & Sasa 2021) predicts LRO for
//   any O(N) model under shear.  O(4) lies between O(3) (C1) and the
//   large-N limit, and tests the N-dependence of shear-induced LRO.
//   O(4) also connects to the 4-component scalar field theory relevant
//   in particle physics (Higgs-like) and helps extrapolate to N→∞.
//
// Equation of motion  (Euler-Maruyama SDE on S³)
// -----------------------------------------------
//   d n_i = P_⊥(n_i) [ J Σ_{j∈NN} n_j
//                     + γ̇ y_i (n_{rt} − n_{lf})/2 ] dt
//           + √(2D dt) P_⊥(n_i) ξ_i
//   then normalise.  n_i ∈ S³ (4-component unit vector).
//
// Parameters
// ----------
//   J     : coupling
//   γ̇     : shear rate
//   D     : noise strength
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨n⟩|   O(4) order parameter
//   C(r)  = ⟨n_i · n_{i+r}⟩_x
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   c4_timeseries.csv    step, t, m
//   c4_correlation.csv   r, C_r
//   c4_shear_scan.csv    L, gamma_dot, m_mean, m_std, binder
//   c4_size_scan.csv     L, gamma_dot, m_mean, m_std, binder
//
// Cargo deps: rand = "0.8" (features=["small_rng"]), rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Normal;
use std::fs::File;
use std::io::{BufWriter, Write};

type Vec4 = [f64; 4];

// ─────────────────────────────────────────────────────────────
// § 1  Vec4 helpers
// ─────────────────────────────────────────────────────────────

#[inline] fn dot4(a: &Vec4, b: &Vec4) -> f64 {
    a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]
}
#[inline] fn add4(a: Vec4, b: Vec4) -> Vec4 {
    [a[0]+b[0],a[1]+b[1],a[2]+b[2],a[3]+b[3]]
}
#[inline] fn sub4(a: Vec4, b: Vec4) -> Vec4 {
    [a[0]-b[0],a[1]-b[1],a[2]-b[2],a[3]-b[3]]
}
#[inline] fn scale4(s: f64, v: Vec4) -> Vec4 {
    [s*v[0],s*v[1],s*v[2],s*v[3]]
}
#[inline] fn norm4(v: Vec4) -> Vec4 {
    let r = dot4(&v,&v).sqrt();
    if r < 1e-30 { [1.0,0.0,0.0,0.0] } else { scale4(1.0/r, v) }
}
#[inline] fn proj_tan4(n: &Vec4, v: Vec4) -> Vec4 {
    sub4(v, scale4(dot4(n,&v), *n))
}
#[inline] fn rand_unit4(rng: &mut SmallRng, nd: &Normal<f64>) -> Vec4 {
    let v = [rng.sample(*nd),rng.sample(*nd),rng.sample(*nd),rng.sample(*nd)];
    norm4(v)
}

// ─────────────────────────────────────────────────────────────
// § 2  Lattice (O(4) unit vectors)
// ─────────────────────────────────────────────────────────────

struct O4Lattice {
    l:   usize,
    spn: Vec<Vec4>,
}

impl O4Lattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i*self.l+j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64+d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> Vec4 { self.spn[self.idx(i,j)] }

    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let nd = Normal::new(0.0f64, 1.0).unwrap();
        O4Lattice {
            l,
            spn: (0..l*l).map(|_| rand_unit4(rng, &nd)).collect(),
        }
    }
}

// ─────────────────────────────────────────────────────────────
// § 3  Euler-Maruyama step on S³
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut O4Lattice, j: f64, gd: f64,
           d: f64, dt: f64, rng: &mut SmallRng) {
    let l  = lat.l;
    let nd = Normal::new(0.0f64, 1.0).unwrap();
    let sd = (2.0 * d * dt).sqrt();
    let spn_old = lat.spn.clone();

    for i in 0..l {
        let yi = i as f64;
        for jj in 0..l {
            let k   = lat.idx(i,jj);
            let ni  = spn_old[k];
            let up  = spn_old[lat.idx(lat.w(i, 1),  jj)];
            let dn  = spn_old[lat.idx(lat.w(i,-1),  jj)];
            let rt  = spn_old[lat.idx(i, lat.w(jj, 1))];
            let lf  = spn_old[lat.idx(i, lat.w(jj,-1))];

            // XY coupling (O(4))
            let coup = [j*(up[0]+dn[0]+rt[0]+lf[0]),
                        j*(up[1]+dn[1]+rt[1]+lf[1]),
                        j*(up[2]+dn[2]+rt[2]+lf[2]),
                        j*(up[3]+dn[3]+rt[3]+lf[3])];
            let d_xy = proj_tan4(&ni, coup);

            // Shear advection
            let sh_raw = scale4(gd*yi*0.5, sub4(rt,lf));
            let d_sh   = proj_tan4(&ni, sh_raw);

            // Noise
            let nr = [sd*rng.sample(nd),sd*rng.sample(nd),
                      sd*rng.sample(nd),sd*rng.sample(nd)];
            let noise = proj_tan4(&ni, nr);

            let new_n = add4(add4(add4(ni, scale4(dt, d_xy)),
                                  scale4(dt, d_sh)),
                             noise);
            lat.spn[k] = norm4(new_n);
        }
    }
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &O4Lattice) -> f64 {
    let n = lat.l*lat.l;
    let m = lat.spn.iter().fold([0.0;4], |acc,s| add4(acc,*s));
    let m = scale4(1.0/n as f64, m);
    dot4(&m,&m).sqrt()
}

fn correlation(lat: &O4Lattice, r_max: usize) -> Vec<f64> {
    let l = lat.l;
    let nf = (l*l) as f64;
    let mut c = vec![0.0f64; r_max+1];
    c[0] = 1.0;
    for r in 1..=r_max {
        c[r] = (0..l).flat_map(|i| (0..l).map(move |j| (i,j)))
            .map(|(i,j)| dot4(&lat.get(i,j), &lat.get(i,(j+r)%l)))
            .sum::<f64>() / nf;
    }
    c
}

fn binder(ms: &[f64]) -> f64 {
    let n  = ms.len() as f64;
    let m2 = ms.iter().map(|m| m*m).sum::<f64>() / n;
    let m4 = ms.iter().map(|m| m.powi(4)).sum::<f64>() / n;
    if m2 < 1e-15 { return 0.0; }
    1.0 - m4 / (3.0*m2*m2)
}

fn mean_f(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
fn std_f(v: &[f64]) -> f64 {
    let mu = mean_f(v);
    (v.iter().map(|x| (x-mu).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

// ─────────────────────────────────────────────────────────────
// § 5  Runner
// ─────────────────────────────────────────────────────────────

struct RunResult { l: usize, gd: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, gd: f64, d: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = O4Lattice::new_random(l, &mut rng);
    let n_tr = (t_trans/dt).round() as usize;
    let n_me = (t_meas /dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat,j,gd,d,dt,&mut rng); }
    if verbose {
        println!("[C4] L={l} J={j} γ̇={gd:.3} D={d:.2} (O(4) shear) transient done");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat,j,gd,d,dt,&mut rng);
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
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, gd: f64, m_mean: f64, m_std: f64, binder: f64 }

fn shear_scan(l: usize, gd_arr: &[f64], j: f64, d: f64,
              dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    gd_arr.iter().map(|&gd| {
        let r = run(l,j,gd,d,dt,tt,tm,20,seed,false);
        println!("  L={l} γ̇={gd:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
        SRow { l, gd, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, gd: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l,j,gd,d,dt,tt,tm,20,seed,false);
        println!("  L={l} γ̇={gd:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
        SRow { l, gd, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn write_ts(path: &str, m_ts: &[f64], dt: f64, me: usize) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w,"step,t,m").unwrap();
    for (k,&m) in m_ts.iter().enumerate() {
        let s=k*me; writeln!(w,"{},{:.6},{:.8}",s,s as f64*dt,m).unwrap(); }
    println!("Written: {path}");
}
fn write_corr(path: &str, c: &[f64]) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w,"r,C_r").unwrap();
    for (r,&v) in c.iter().enumerate() { writeln!(w,"{},{:.8}",r,v).unwrap(); }
    println!("Written: {path}");
}
fn write_scan(path: &str, rows: &[SRow]) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w,"L,gamma_dot,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w,"{},{:.4},{:.6},{:.6},{:.6}",
                 r.l,r.gd,r.m_mean,r.m_std,r.binder).unwrap(); }
    println!("Written: {path}");
}

fn main() {
    let me = 20usize;
    let res = run(24,1.0,0.5,0.5,0.01,20.0,100.0,me,42,true);
    write_ts("c4_timeseries.csv",&res.m_ts,res.dt,me);
    write_corr("c4_correlation.csv",&res.c_r);
    let gd_arr: Vec<f64> = (0..=10).map(|k| k as f64*0.2).collect();
    let sr = shear_scan(24,&gd_arr,1.0,0.5,0.01,15.0,60.0,0);
    write_scan("c4_shear_scan.csv",&sr);
    let lr = size_scan(&[12,16,24,32],1.0,0.5,0.5,0.01,15.0,60.0,1);
    write_scan("c4_size_scan.csv",&lr);
}
