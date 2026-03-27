// ============================================================
// model_e4_xy_random_nrec.rs  —  Model E4  🟠 P2
// XY Model + Spatially Random Non-Reciprocal Coupling Strength
// ============================================================
// Model family : E族 — 非互易相互作用的新变体
// Priority     : 🟠 P2  (LRO probability: medium 40–70%)
//
// Physics motivation
// ------------------
//   Lorenzana (2025) analysed random non-reciprocity in field theories.
//   Model E4 implements this on the lattice: each bond (i,j) has a
//   symmetric part J and a random non-reciprocal part δJ_{ij} drawn
//   from N(0, σ_nr²) with δJ_{ij} = −δJ_{ji} (quenched antisymmetric).
//   The non-reciprocal strengths are spatially random but statistically
//   isotropic (each site has the same distribution).
//
// Equation of motion  (Euler-Maruyama SDE)
// -----------------------------------------
//   dθ_i = Σ_{j∈NN} [J + δJ_{ij}] sin(θ_j − θ_i) dt
//           + √(2D dt) ξ_i
//
//   δJ_{ij} ~ N(0, σ_nr²) quenched, δJ_{ji} = −δJ_{ij}.
//   Stored as: for each horizontal bond (i,j)→(i,j+1): dj_h[idx(i,j)]
//              for each vertical bond   (i,j)→(i+1,j): dj_v[idx(i,j)]
//
// Output files
// ------------
//   e4_timeseries.csv    step, t, m
//   e4_correlation.csv   r, C_r
//   e4_sigma_scan.csv    L, sigma_nr, m_mean, m_std, binder
//   e4_size_scan.csv     L, sigma_nr, m_mean, m_std, binder
//
// Cargo deps: rand = "0.8" (features=["small_rng"]), rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, Uniform};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

struct RandNRLattice {
    l:      usize,
    theta:  Vec<f64>,
    dj_h:   Vec<f64>,  // horiz bond non-recip: dj_h[idx(i,j)] = δJ_{(i,j)→(i,j+1)}
    dj_v:   Vec<f64>,  // vert  bond non-recip: dj_v[idx(i,j)] = δJ_{(i,j)→(i+1,j)}
}

impl RandNRLattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i*self.l+j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64+d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.theta[self.idx(i,j)] }

    fn new(l: usize, sigma_nr: f64, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let ud  = Uniform::new(0.0f64, 2.0*PI);
        let nd  = Normal::new(0.0f64, sigma_nr).unwrap();
        let theta: Vec<f64> = (0..l*l).map(|_| rng.sample(ud)).collect();
        let dj_h: Vec<f64>  = (0..l*l).map(|_| rng.sample(nd)).collect();
        let dj_v: Vec<f64>  = (0..l*l).map(|_| rng.sample(nd)).collect();
        RandNRLattice { l, theta, dj_h, dj_v }
    }
}

fn em_step(lat: &mut RandNRLattice, j: f64, d: f64, dt: f64, rng: &mut SmallRng) {
    let l  = lat.l;
    let nd = Normal::new(0.0f64,1.0).unwrap();
    let sd = (2.0*d*dt).sqrt();
    let mut dth = vec![0.0f64; l*l];
    for i in 0..l {
        for jj in 0..l {
            let k   = lat.idx(i,jj);
            let th  = lat.theta[k];
            let up  = lat.get(lat.w(i, 1), jj);
            let dn  = lat.get(lat.w(i,-1), jj);
            let rt  = lat.get(i, lat.w(jj, 1));
            let lf  = lat.get(i, lat.w(jj,-1));
            // j+1 direction (right): coupling = J + dj_h[idx(i,jj)]
            let jr  = j + lat.dj_h[lat.idx(i,jj)];
            // j-1 direction (left): bond origin is (i,jj-1); coupling from i sees
            //   −δJ_{(i,jj-1)→(i,jj)} = sign reversal
            let jl  = j - lat.dj_h[lat.idx(i, lat.w(jj,-1))];
            // i+1 direction (up)
            let ju  = j + lat.dj_v[lat.idx(i,jj)];
            // i-1 direction (down)
            let jd  = j - lat.dj_v[lat.idx(lat.w(i,-1), jj)];
            dth[k] = (jr*(rt-th).sin()+jl*(lf-th).sin()
                     +ju*(up-th).sin()+jd*(dn-th).sin())*dt
                   + sd*rng.sample(nd);
        }
    }
    for k in 0..l*l { lat.theta[k] += dth[k]; }
}

fn order_param(lat: &RandNRLattice) -> f64 {
    let n=(lat.l*lat.l) as f64;
    let re=lat.theta.iter().map(|t|t.cos()).sum::<f64>()/n;
    let im=lat.theta.iter().map(|t|t.sin()).sum::<f64>()/n;
    (re*re+im*im).sqrt()
}
fn correlation(lat: &RandNRLattice, r_max: usize) -> Vec<f64> {
    let l=lat.l; let nf=(l*l) as f64;
    let mut c=vec![0.0f64;r_max+1]; c[0]=1.0;
    for r in 1..=r_max {
        c[r]=(0..l).flat_map(|i|(0..l).map(move |j|(i,j)))
            .map(|(i,j)|(lat.get(i,j)-lat.get(i,(j+r)%l)).cos()).sum::<f64>()/nf;
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

struct RunResult { l: usize, sig: f64, m_mean: f64, m_std: f64,
                   binder: f64, c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, sig: f64, d: f64, dt: f64,
       tt: f64, tm: f64, me: usize, seed: u64, verbose: bool) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed+999);
    let mut lat = RandNRLattice::new(l, sig, seed);
    for _ in 0..(tt/dt).round() as usize { em_step(&mut lat,j,d,dt,&mut rng); }
    if verbose { println!("[E4] L={l} J={j} σ_nr={sig:.3} D={d:.2} transient done"); }
    let mut m_ts = Vec::new();
    for s in 0..(tm/dt).round() as usize {
        em_step(&mut lat,j,d,dt,&mut rng);
        if s%me==0 { m_ts.push(order_param(&lat)); }
    }
    let c_r=correlation(&lat,l/2);
    let m_mean=mean_f(&m_ts); let m_std=std_f(&m_ts); let bdr=binder(&m_ts);
    if verbose { println!("      m={m_mean:.4}  U_L={bdr:.4}"); }
    RunResult { l, sig, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

struct SRow { l: usize, sig: f64, m_mean: f64, m_std: f64, binder: f64 }
fn sig_scan(l: usize, sig_arr: &[f64], j: f64, d: f64, dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    sig_arr.iter().map(|&sig| {
        let r=run(l,j,sig,d,dt,tt,tm,20,seed,false);
        println!("  L={l} σ_nr={sig:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
        SRow{l,sig,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}
    }).collect()
}
fn size_scan(l_arr: &[usize], j: f64, sig: f64, d: f64, dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r=run(l,j,sig,d,dt,tt,tm,20,seed,false);
        println!("  L={l} σ_nr={sig:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
        SRow{l,sig,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}
    }).collect()
}

fn write_ts(path: &str, m_ts: &[f64], dt: f64, me: usize) {
    let mut w=BufWriter::new(File::create(path).unwrap()); writeln!(w,"step,t,m").unwrap();
    for (k,&m) in m_ts.iter().enumerate() { let s=k*me; writeln!(w,"{},{:.6},{:.8}",s,s as f64*dt,m).unwrap(); }
    println!("Written: {path}");
}
fn write_corr(path: &str, c: &[f64]) {
    let mut w=BufWriter::new(File::create(path).unwrap()); writeln!(w,"r,C_r").unwrap();
    for (r,&v) in c.iter().enumerate() { writeln!(w,"{},{:.8}",r,v).unwrap(); }
    println!("Written: {path}");
}
fn write_scan(path: &str, rows: &[SRow]) {
    let mut w=BufWriter::new(File::create(path).unwrap()); writeln!(w,"L,sigma_nr,m_mean,m_std,binder").unwrap();
    for r in rows { writeln!(w,"{},{:.4},{:.6},{:.6},{:.6}",r.l,r.sig,r.m_mean,r.m_std,r.binder).unwrap(); }
    println!("Written: {path}");
}

fn main() {
    let me=20usize;
    let res=run(32,1.0,0.3,0.5,0.01,20.0,100.0,me,42,true);
    write_ts("e4_timeseries.csv",&res.m_ts,res.dt,me);
    write_corr("e4_correlation.csv",&res.c_r);
    let sig_arr: Vec<f64>=(0..=10).map(|k| k as f64*0.1).collect();
    let sr=sig_scan(32,&sig_arr,1.0,0.5,0.01,15.0,60.0,0);
    write_scan("e4_sigma_scan.csv",&sr);
    let lr=size_scan(&[16,24,32,48],1.0,0.3,0.5,0.01,15.0,60.0,1);
    write_scan("e4_size_scan.csv",&lr);
}
