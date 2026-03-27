// ============================================================
// model_f4_xy_dual_bath.rs  —  Model F4  🟠 P2
// XY Model + Dual Bath (fast + slow bath, isotropic)
// ============================================================
// Model family : F族 — 噪声谱工程
// Priority     : 🟠 P2  (LRO probability: medium)
//
// Physics motivation
// ------------------
//   Maire & Plati (2024) showed that a system coupled to a fast
//   local bath AND a slow collective bath (centre-of-mass mode
//   selectively cooled) can exhibit enhanced positional order.
//   Model F4 applies this dual-bath idea to the XY spin:
//     • Bath 1: fast white noise, strength D₁ (local, all modes)
//     • Bath 2: slow Ornstein-Uhlenbeck, strength D₂, correlation τ₂
//   If the slow bath effectively suppresses long-wavelength modes,
//   LRO may emerge.  Unlike B5 (single OU bath), here both baths
//   are simultaneously present.
//
// Equation of motion
// ------------------
//   dθ_i = J Σ_{⟨ij⟩} sin(θ_j−θ_i) dt
//          + √(2D₁ dt) ξ_i^{fast}       [fast bath]
//          + η_i^{slow}(t) dt           [slow OU bath]
//   dη_i^{slow} = −(η_i/τ₂)dt + √(2D₂/τ₂) dW_i
//
// Output files
// ------------
//   f4_timeseries.csv   step, t, m
//   f4_correlation.csv  r, C_r
//   f4_d2_scan.csv      L, D2, tau2, m_mean, m_std, binder
//   f4_size_scan.csv    L, D2, tau2, m_mean, m_std, binder
//
// Cargo.toml deps:  rand = "0.8"  rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, Uniform};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

struct DualBathLattice {
    l:     usize,
    theta: Vec<f64>,
    eta:   Vec<f64>,   // slow OU bath field
}
impl DualBathLattice {
    #[inline] fn idx(&self,i:usize,j:usize)->usize{i*self.l+j}
    #[inline] fn w(&self,x:usize,d:i64)->usize{
        ((x as i64+d).rem_euclid(self.l as i64)) as usize}
    #[inline] fn get(&self,i:usize,j:usize)->f64{self.theta[self.idx(i,j)]}
    fn new_random(l:usize,rng:&mut SmallRng)->Self{
        let d=Uniform::new(0.0f64,2.0*PI);
        let theta:Vec<f64>=(0..l*l).map(|_|rng.sample(d)).collect();
        let eta=vec![0.0f64;l*l];
        DualBathLattice{l,theta,eta}
    }
}

fn em_step(lat:&mut DualBathLattice, j:f64, d1:f64, d2:f64, tau2:f64, dt:f64, rng:&mut SmallRng){
    let l=lat.l; let n=l*l;
    let nd=Normal::new(0.0f64,1.0).unwrap();
    // update slow OU bath
    let decay=(-dt/tau2).exp();
    let ou_sd=(d2*(1.0-decay*decay)).sqrt();
    for k in 0..n { lat.eta[k]=lat.eta[k]*decay+ou_sd*rng.sample(nd); }
    // XY drift + fast bath + slow bath
    let sd1=(2.0*d1*dt).sqrt();
    let mut dth=vec![0.0f64;n];
    for i in 0..l { for jj in 0..l {
        let k=lat.idx(i,jj); let th=lat.theta[k];
        let up=lat.get(lat.w(i,1),jj); let dn=lat.get(lat.w(i,-1),jj);
        let rt=lat.get(i,lat.w(jj,1)); let lf=lat.get(i,lat.w(jj,-1));
        dth[k]=j*((up-th).sin()+(dn-th).sin()+(rt-th).sin()+(lf-th).sin())*dt
              +sd1*rng.sample(nd)       // fast bath
              +lat.eta[k]*dt;           // slow bath
    }}
    for k in 0..n { lat.theta[k]+=dth[k]; }
}

fn order_param(lat:&DualBathLattice)->f64{
    let n=(lat.l*lat.l) as f64;
    let re=lat.theta.iter().map(|t|t.cos()).sum::<f64>()/n;
    let im=lat.theta.iter().map(|t|t.sin()).sum::<f64>()/n;
    (re*re+im*im).sqrt()
}
fn correlation(lat:&DualBathLattice,r_max:usize)->Vec<f64>{
    let l=lat.l; let n=(l*l) as f64;
    let mut c=vec![0.0f64;r_max+1]; c[0]=1.0;
    for r in 1..=r_max{
        c[r]=(0..l).flat_map(|i|(0..l).map(move|j|(i,j)))
            .map(|(i,j)|(lat.get(i,j)-lat.get(i,(j+r)%l)).cos()).sum::<f64>()/n;
    }
    c
}
fn binder(ms:&[f64])->f64{
    let n=ms.len() as f64; let m2=ms.iter().map(|m|m*m).sum::<f64>()/n;
    let m4=ms.iter().map(|m|m.powi(4)).sum::<f64>()/n;
    if m2<1e-15{return 0.0;} 1.0-m4/(3.0*m2*m2)
}
fn mean_f(v:&[f64])->f64{v.iter().sum::<f64>()/v.len() as f64}
fn std_f(v:&[f64])->f64{let mu=mean_f(v);(v.iter().map(|x|(x-mu).powi(2)).sum::<f64>()/v.len() as f64).sqrt()}

struct RunResult{l:usize,d2:f64,tau2:f64,m_mean:f64,m_std:f64,binder:f64,c_r:Vec<f64>,m_ts:Vec<f64>,dt:f64}
fn run(l:usize,j:f64,d1:f64,d2:f64,tau2:f64,dt:f64,t_trans:f64,t_meas:f64,mevery:usize,seed:u64,verbose:bool)->RunResult{
    let mut rng=SmallRng::seed_from_u64(seed);
    let mut lat=DualBathLattice::new_random(l,&mut rng);
    let n_tr=(t_trans/dt).round() as usize; let n_me=(t_meas/dt).round() as usize;
    for _ in 0..n_tr{em_step(&mut lat,j,d1,d2,tau2,dt,&mut rng);}
    if verbose{println!("[F4] L={l} D₁={d1:.3} D₂={d2:.3} τ₂={tau2:.2} transient done");}
    let mut m_ts=Vec::new();
    for s in 0..n_me{em_step(&mut lat,j,d1,d2,tau2,dt,&mut rng);if s%mevery==0{m_ts.push(order_param(&lat));}}
    let c_r=correlation(&lat,l/2); let m_mean=mean_f(&m_ts); let m_std=std_f(&m_ts); let bdr=binder(&m_ts);
    if verbose{println!("      m={m_mean:.4} U_L={bdr:.4}");}
    RunResult{l,d2,tau2,m_mean,m_std,binder:bdr,c_r,m_ts,dt}
}
struct SRow{l:usize,d2:f64,tau2:f64,m_mean:f64,m_std:f64,binder:f64}
fn d2_scan(l:usize,d2_arr:&[f64],j:f64,d1:f64,tau2:f64,dt:f64,tt:f64,tm:f64,seed:u64)->Vec<SRow>{
    d2_arr.iter().map(|&d2|{let r=run(l,j,d1,d2,tau2,dt,tt,tm,20,seed,false);
    println!("  L={l} D₂={d2:.3} τ₂={tau2:.2}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
    SRow{l,d2,tau2,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}}).collect()
}
fn size_scan(l_arr:&[usize],j:f64,d1:f64,d2:f64,tau2:f64,dt:f64,tt:f64,tm:f64,seed:u64)->Vec<SRow>{
    l_arr.iter().map(|&l|{let r=run(l,j,d1,d2,tau2,dt,tt,tm,20,seed,false);
    println!("  L={l} D₂={d2:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
    SRow{l,d2,tau2,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}}).collect()
}
fn write_ts(path:&str,m_ts:&[f64],dt:f64,me:usize){
    let mut w=BufWriter::new(File::create(path).unwrap()); writeln!(w,"step,t,m").unwrap();
    for(k,&m)in m_ts.iter().enumerate(){let s=k*me;writeln!(w,"{},{:.6},{:.8}",s,s as f64*dt,m).unwrap();}
    println!("Written: {path}");
}
fn write_corr(path:&str,c:&[f64]){
    let mut w=BufWriter::new(File::create(path).unwrap()); writeln!(w,"r,C_r").unwrap();
    for(r,&v)in c.iter().enumerate(){writeln!(w,"{},{:.8}",r,v).unwrap();}
    println!("Written: {path}");
}
fn write_scan(path:&str,rows:&[SRow]){
    let mut w=BufWriter::new(File::create(path).unwrap()); writeln!(w,"L,D2,tau2,m_mean,m_std,binder").unwrap();
    for r in rows{writeln!(w,"{},{:.4},{:.4},{:.6},{:.6},{:.6}",r.l,r.d2,r.tau2,r.m_mean,r.m_std,r.binder).unwrap();}
    println!("Written: {path}");
}
fn main(){
    let me=20usize;
    let res=run(32,1.0,0.1,0.5,5.0,0.01,20.0,100.0,me,42,true);
    write_ts("f4_timeseries.csv",&res.m_ts,res.dt,me);
    write_corr("f4_correlation.csv",&res.c_r);
    // D₂ scan: D₂=0 (no slow bath) → D₂=1.0
    let d2_arr:Vec<f64>=(0..=10).map(|k|k as f64*0.1).collect();
    let dr=d2_scan(32,&d2_arr,1.0,0.1,5.0,0.01,15.0,60.0,0);
    write_scan("f4_d2_scan.csv",&dr);
    let lr=size_scan(&[16,24,32,48],1.0,0.1,0.5,5.0,0.01,15.0,60.0,1);
    write_scan("f4_size_scan.csv",&lr);
}
