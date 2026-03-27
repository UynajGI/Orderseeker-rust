// ============================================================
// model_e3_xy_antisymmetric_nrec.rs  —  Model E3  🟠 P2
// XY Model + Antisymmetric Non-Reciprocal Coupling (J_ij = −J_ji)
// ============================================================
// Model family : E族 — 非互易相互作用的新变体
// Priority     : 🟠 P2  (LRO probability: medium-high 40–70%)
//
// Physics motivation
// ------------------
//   Loos et al. (2023) used "vision-cone" non-reciprocity: J_{ij}=J
//   for j in cone, J_{ji}=0 (one-way coupling).  Model E3 explores
//   a different non-reciprocity: fully antisymmetric coupling where
//   J_{ij} = J, J_{ji} = −J.  This means each bond drives a ROTATION
//   rather than a simple alignment.  Antisymmetric coupling generates
//   persistent probability currents around loops, which is a source
//   of non-equilibrium driving.
//
// Equation of motion  (Euler-Maruyama SDE)
// -----------------------------------------
//   dθ_i = J Σ_{j∈NN} sin(θ_j − θ_i) dt                [symmetric part]
//           + J' Σ_{j∈NN_fwd} sin(θ_j − θ_i) dt         [non-recip part, fwd]
//           − J' Σ_{j∈NN_bwd} sin(θ_j − θ_i) dt         [non-recip part, bwd]
//           + √(2D dt) ξ_i
//
//   "fwd" = {right, up},  "bwd" = {left, down} (fixed asymmetry direction).
//   J' = 0: standard symmetric XY.
//   J' = J: right/up coupling is doubled, left/down is zero (Loos-like).
//   J' can be negative (reversed antisymmetry).
//
// Output files
// ------------
//   e3_timeseries.csv    step, t, m
//   e3_correlation.csv   r, C_r
//   e3_jprime_scan.csv   L, J_prime, m_mean, m_std, binder
//   e3_size_scan.csv     L, J_prime, m_mean, m_std, binder
//
// Cargo deps: rand = "0.8" (features=["small_rng"]), rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, Uniform};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

struct XYLat {
    l:     usize,
    theta: Vec<f64>,
}

impl XYLat {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i*self.l+j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64+d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i,j)] }
    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let d = Uniform::new(0.0f64, 2.0*PI);
        XYLat { l, theta: (0..l*l).map(|_| rng.sample(d)).collect() }
    }
}

fn em_step(lat: &mut XYLat, j: f64, jp: f64, d: f64, dt: f64, rng: &mut SmallRng) {
    let l = lat.l;
    let nd = Normal::new(0.0f64,1.0).unwrap();
    let sd = (2.0*d*dt).sqrt();
    let mut dth = vec![0.0f64; l*l];
    for i in 0..l {
        for jj in 0..l {
            let k  = lat.idx(i,jj);
            let th = lat.theta[k];
            let up  = lat.get(lat.w(i, 1), jj);
            let dn  = lat.get(lat.w(i,-1), jj);
            let rt  = lat.get(i, lat.w(jj, 1));
            let lf  = lat.get(i, lat.w(jj,-1));
            // symmetric part
            dth[k] = j*((up-th).sin()+(dn-th).sin()+(rt-th).sin()+(lf-th).sin())*dt;
            // antisymmetric non-reciprocal part: +J' for fwd, -J' for bwd
            dth[k] += jp*((rt-th).sin()+(up-th).sin())*dt;
            dth[k] -= jp*((lf-th).sin()+(dn-th).sin())*dt;
            dth[k] += sd*rng.sample(nd);
        }
    }
    for k in 0..l*l { lat.theta[k] += dth[k]; }
}

fn order_param(lat: &XYLat) -> f64 {
    let n = (lat.l*lat.l) as f64;
    let re = lat.theta.iter().map(|t| t.cos()).sum::<f64>() / n;
    let im = lat.theta.iter().map(|t| t.sin()).sum::<f64>() / n;
    (re*re+im*im).sqrt()
}

fn correlation(lat: &XYLat, r_max: usize) -> Vec<f64> {
    let l = lat.l; let nf = (l*l) as f64;
    let mut c = vec![0.0f64; r_max+1]; c[0]=1.0;
    for r in 1..=r_max {
        c[r] = (0..l).flat_map(|i| (0..l).map(move |j| (i,j)))
            .map(|(i,j)| (lat.get(i,j)-lat.get(i,(j+r)%l)).cos()).sum::<f64>()/nf;
    }
    c
}

fn binder(ms: &[f64]) -> f64 {
    let n=ms.len() as f64;
    let m2=ms.iter().map(|m|m*m).sum::<f64>()/n;
    let m4=ms.iter().map(|m|m.powi(4)).sum::<f64>()/n;
    if m2<1e-15 {0.0} else {1.0-m4/(3.0*m2*m2)}
}
fn mean_f(v: &[f64]) -> f64 { v.iter().sum::<f64>()/v.len() as f64 }
fn std_f(v: &[f64]) -> f64 {
    let mu=mean_f(v);
    (v.iter().map(|x|(x-mu).powi(2)).sum::<f64>()/v.len() as f64).sqrt()
}

struct RunResult { l: usize, jp: f64, m_mean: f64, m_std: f64,
                   binder: f64, c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, jp: f64, d: f64, dt: f64,
       tt: f64, tm: f64, me: usize, seed: u64, verbose: bool) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = XYLat::new_random(l, &mut rng);
    for _ in 0..(tt/dt).round() as usize { em_step(&mut lat,j,jp,d,dt,&mut rng); }
    if verbose { println!("[E3] L={l} J={j} J'={jp:.3} D={d:.2} transient done"); }
    let mut m_ts = Vec::new();
    for s in 0..(tm/dt).round() as usize {
        em_step(&mut lat,j,jp,d,dt,&mut rng);
        if s%me==0 { m_ts.push(order_param(&lat)); }
    }
    let c_r=correlation(&lat,l/2);
    let m_mean=mean_f(&m_ts); let m_std=std_f(&m_ts); let bdr=binder(&m_ts);
    if verbose { println!("      m={m_mean:.4} U_L={bdr:.4}"); }
    RunResult { l, jp, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

struct SRow { l: usize, jp: f64, m_mean: f64, m_std: f64, binder: f64 }
fn jp_scan(l: usize, jp_arr: &[f64], j: f64, d: f64, dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    jp_arr.iter().map(|&jp| {
        let r=run(l,j,jp,d,dt,tt,tm,20,seed,false);
        println!("  L={l} J'={jp:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
        SRow{l,jp,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}
    }).collect()
}
fn size_scan(l_arr: &[usize], j: f64, jp: f64, d: f64, dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r=run(l,j,jp,d,dt,tt,tm,20,seed,false);
        println!("  L={l} J'={jp:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
        SRow{l,jp,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}
    }).collect()
}

fn write_ts(path: &str, m_ts: &[f64], dt: f64, me: usize) {
    let mut w=BufWriter::new(File::create(path).unwrap());
    writeln!(w,"step,t,m").unwrap();
    for (k,&m) in m_ts.iter().enumerate() { let s=k*me; writeln!(w,"{},{:.6},{:.8}",s,s as f64*dt,m).unwrap(); }
    println!("Written: {path}");
}
fn write_corr(path: &str, c: &[f64]) {
    let mut w=BufWriter::new(File::create(path).unwrap());
    writeln!(w,"r,C_r").unwrap();
    for (r,&v) in c.iter().enumerate() { writeln!(w,"{},{:.8}",r,v).unwrap(); }
    println!("Written: {path}");
}
fn write_scan(path: &str, rows: &[SRow]) {
    let mut w=BufWriter::new(File::create(path).unwrap());
    writeln!(w,"L,J_prime,m_mean,m_std,binder").unwrap();
    for r in rows { writeln!(w,"{},{:.4},{:.6},{:.6},{:.6}",r.l,r.jp,r.m_mean,r.m_std,r.binder).unwrap(); }
    println!("Written: {path}");
}

fn main() {
    let me = 20usize;
    let res = run(32,1.0,0.5,0.5,0.01,20.0,100.0,me,42,true);
    write_ts("e3_timeseries.csv",&res.m_ts,res.dt,me);
    write_corr("e3_correlation.csv",&res.c_r);
    let jp_arr: Vec<f64> = (0..=10).map(|k| k as f64*0.1).collect();
    let jr = jp_scan(32,&jp_arr,1.0,0.5,0.01,15.0,60.0,0);
    write_scan("e3_jprime_scan.csv",&jr);
    let lr = size_scan(&[16,24,32,48],1.0,0.5,0.5,0.01,15.0,60.0,1);
    write_scan("e3_size_scan.csv",&lr);
}
