// ============================================================
// model_f3_xy_1f_noise.rs  —  Model F3  🟠 P2
// XY Model + 1/f^β Time-Correlated (Power-Law) Noise
// ============================================================
// Model family : F族 — 噪声谱工程
// Priority     : 🟠 P2  (LRO probability: medium-high)
//
// Physics motivation
// ------------------
//   Noise with power-law temporal spectrum S(ω) ∝ |ω|^{−β}:
//     β=0 → white noise    (standard, no LRO)
//     β=1 → 1/f noise      (ubiquitous in nature)
//     β=2 → Brownian noise (very long memory)
//   Long-time correlations (large β) effectively suppress the
//   exploration of Goldstone modes, potentially stabilising LRO.
//
//   Implementation: approximate 1/f^β via a sum of Ornstein-Uhlenbeck
//   processes with logarithmically-spaced correlation times:
//     η_i(t) = Σ_{k=1}^{K} η_i^{(k)}(t)
//   where each η^{(k)} has correlation time τ_k = τ_min * (τ_max/τ_min)^{k/K}.
//   The combined spectrum approximates 1/f^β for a range of frequencies.
//
//   For simplicity: 3 OU processes with τ = {0.1, 1.0, 10.0} and
//   equal amplitudes (approximates β≈0.5 spectrum).  The exponent β
//   controls the mixing weights.
//
// Equation of motion
// ------------------
//   dθ_i = J Σ_{⟨ij⟩} sin(θ_j−θ_i) dt + η_i(t) dt
//   η_i = Σ_k w_k η_i^{(k)},  each η^{(k)} ~ OU(τ_k, D_k)
//
// Output files
// ------------
//   f3_timeseries.csv   step, t, m
//   f3_correlation.csv  r, C_r
//   f3_beta_scan.csv    L, beta, m_mean, m_std, binder
//   f3_size_scan.csv    L, beta, m_mean, m_std, binder
//
// Cargo.toml deps:  rand = "0.8"  rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, Uniform};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

const N_OU: usize = 4;  // number of OU components

struct PLNLattice {
    l:   usize,
    theta: Vec<f64>,
    eta:   Vec<[f64; N_OU]>,  // OU noise components per site
}
impl PLNLattice {
    #[inline] fn idx(&self,i:usize,j:usize)->usize{i*self.l+j}
    #[inline] fn w(&self,x:usize,d:i64)->usize{
        ((x as i64+d).rem_euclid(self.l as i64)) as usize}
    #[inline] fn get(&self,i:usize,j:usize)->f64{self.theta[self.idx(i,j)]}
    fn new_random(l:usize,rng:&mut SmallRng)->Self{
        let d=Uniform::new(0.0f64,2.0*PI);
        let theta:Vec<f64>=(0..l*l).map(|_|rng.sample(d)).collect();
        let eta:Vec<[f64;N_OU]>=vec![[0.0;N_OU];l*l];
        PLNLattice{l,theta,eta}
    }
}

/// Compute OU mixing weights w_k for β-spectrum approximation.
/// Logarithmically-spaced τ_k from 0.1 to 10; weights ∝ τ_k^{β-1}.
fn ou_params(beta: f64) -> ([f64; N_OU], [f64; N_OU]) {
    let tau_min = 0.05f64; let tau_max = 20.0f64;
    let mut taus = [0.0f64; N_OU];
    let mut amps = [0.0f64; N_OU];
    for k in 0..N_OU {
        taus[k] = tau_min * (tau_max/tau_min).powf(k as f64/(N_OU-1) as f64);
        amps[k] = taus[k].powf(beta - 1.0);
    }
    // normalise amplitudes so total variance = D (set by caller)
    let sum: f64 = amps.iter().sum();
    for k in 0..N_OU { amps[k] /= sum; }
    (taus, amps)
}

fn em_step(lat:&mut PLNLattice, j:f64, d:f64, beta:f64, dt:f64, rng:&mut SmallRng){
    let l=lat.l; let n=l*l;
    let nd=Normal::new(0.0f64,1.0).unwrap();
    let (taus, amps) = ou_params(beta);
    // update OU components
    for k in 0..n {
        for c in 0..N_OU {
            let tau=taus[c]; let amp=amps[c];
            let d_k=d*amp;
            let decay=(-dt/tau).exp();
            let ou_sd=(d_k*(1.0-decay*decay)).sqrt();
            lat.eta[k][c]=lat.eta[k][c]*decay+ou_sd*rng.sample(nd);
        }
    }
    // XY drift + summed OU noise
    let mut dth=vec![0.0f64;n];
    for i in 0..l { for jj in 0..l {
        let k=lat.idx(i,jj); let th=lat.theta[k];
        let up=lat.get(lat.w(i,1),jj); let dn=lat.get(lat.w(i,-1),jj);
        let rt=lat.get(i,lat.w(jj,1)); let lf=lat.get(i,lat.w(jj,-1));
        let eta_tot: f64 = lat.eta[k].iter().sum();
        dth[k]=(j*((up-th).sin()+(dn-th).sin()+(rt-th).sin()+(lf-th).sin())
               +eta_tot)*dt;
    }}
    for k in 0..n { lat.theta[k]+=dth[k]; }
}

fn order_param(lat:&PLNLattice)->f64{
    let n=(lat.l*lat.l) as f64;
    let re=lat.theta.iter().map(|t|t.cos()).sum::<f64>()/n;
    let im=lat.theta.iter().map(|t|t.sin()).sum::<f64>()/n;
    (re*re+im*im).sqrt()
}
fn correlation(lat:&PLNLattice,r_max:usize)->Vec<f64>{
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

struct RunResult{l:usize,beta:f64,m_mean:f64,m_std:f64,binder:f64,c_r:Vec<f64>,m_ts:Vec<f64>,dt:f64}
fn run(l:usize,j:f64,d:f64,beta:f64,dt:f64,t_trans:f64,t_meas:f64,mevery:usize,seed:u64,verbose:bool)->RunResult{
    let mut rng=SmallRng::seed_from_u64(seed);
    let mut lat=PLNLattice::new_random(l,&mut rng);
    let n_tr=(t_trans/dt).round() as usize; let n_me=(t_meas/dt).round() as usize;
    for _ in 0..n_tr{em_step(&mut lat,j,d,beta,dt,&mut rng);}
    if verbose{println!("[F3] L={l} β={beta:.3} J={j} D={d:.2} transient done");}
    let mut m_ts=Vec::new();
    for s in 0..n_me{em_step(&mut lat,j,d,beta,dt,&mut rng);if s%mevery==0{m_ts.push(order_param(&lat));}}
    let c_r=correlation(&lat,l/2); let m_mean=mean_f(&m_ts); let m_std=std_f(&m_ts); let bdr=binder(&m_ts);
    if verbose{println!("      m={m_mean:.4} U_L={bdr:.4}");}
    RunResult{l,beta,m_mean,m_std,binder:bdr,c_r,m_ts,dt}
}
struct SRow{l:usize,beta:f64,m_mean:f64,m_std:f64,binder:f64}
fn beta_scan(l:usize,b_arr:&[f64],j:f64,d:f64,dt:f64,tt:f64,tm:f64,seed:u64)->Vec<SRow>{
    b_arr.iter().map(|&b|{let r=run(l,j,d,b,dt,tt,tm,20,seed,false);
    println!("  L={l} β={b:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
    SRow{l,beta:b,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}}).collect()
}
fn size_scan(l_arr:&[usize],j:f64,d:f64,beta:f64,dt:f64,tt:f64,tm:f64,seed:u64)->Vec<SRow>{
    l_arr.iter().map(|&l|{let r=run(l,j,d,beta,dt,tt,tm,20,seed,false);
    println!("  L={l} β={beta:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
    SRow{l,beta,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}}).collect()
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
    let mut w=BufWriter::new(File::create(path).unwrap()); writeln!(w,"L,beta,m_mean,m_std,binder").unwrap();
    for r in rows{writeln!(w,"{},{:.4},{:.6},{:.6},{:.6}",r.l,r.beta,r.m_mean,r.m_std,r.binder).unwrap();}
    println!("Written: {path}");
}
fn main(){
    let me=20usize;
    let res=run(32,1.0,0.5,1.5,0.01,20.0,100.0,me,42,true);
    write_ts("f3_timeseries.csv",&res.m_ts,res.dt,me);
    write_corr("f3_correlation.csv",&res.c_r);
    // β scan: β=0 (white) → β=2 (Brownian)
    let b_arr:Vec<f64>=(0..=10).map(|k|k as f64*0.2).collect();
    let br=beta_scan(32,&b_arr,1.0,0.5,0.01,15.0,60.0,0);
    write_scan("f3_beta_scan.csv",&br);
    let lr=size_scan(&[16,24,32,48],1.0,0.5,1.5,0.01,15.0,60.0,1);
    write_scan("f3_size_scan.csv",&lr);
}
