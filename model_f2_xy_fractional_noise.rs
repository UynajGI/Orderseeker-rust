// ============================================================
// model_f2_xy_fractional_noise.rs  —  Model F2  🔴 P1 (upgraded)
// XY Model + Fractional-Order Conservative Noise (∇^α η, 0<α<1)
// ============================================================
// Model family : F族 — 噪声谱工程
// Priority     : 🔴 P1  (LRO probability: very high, by construction)
//
// Physics motivation
// ------------------
//   Model F1 uses α=1 (full divergence form, q² spectrum → LRO).
//   Model F2 interpolates: noise spectrum S(k) ∝ |k|^{2α} for 0<α<1.
//   At α=0 we recover flat white noise (no LRO in 2D);
//   at α=1 we recover F1 (LRO guaranteed).
//   There should exist a critical α* such that LRO emerges for α > α*.
//   In 2D the MW condition requires S(k) to suppress ∫d²k/[k²S(k)],
//   which diverges if 2α ≤ 2, i.e., α ≤ 1.  So α=1 is the boundary.
//   This model directly measures the critical noise spectrum exponent.
//
// Implementation of fractional noise  (lattice, 0 < α ≤ 1)
// ----------------------------------------------------------
//   Fourier-space approach:
//   1. Draw iid N(0,1) for each lattice site  →  η̃(k)  (flat spectrum)
//   2. Multiply by |k|^α  in Fourier space    →  modified spectrum ∝ |k|^{2α}
//   3. iFFT back to real space               →  fractional noise field
//   4. Add as noise increment to θ_i
//
//   We use a real-to-real DFT via manual 2D cosine-based approximation
//   since we only have rand.  Alternatively: implement via iterative
//   differencing (for integer α) or use the exact spectral method.
//
//   Practical approximation for arbitrary α:
//   Noise = Σ_{bonds} w_{bond} * antisymmetric_bond_noise,
//   where bond weights w ∝ J^{α/1} encode the fractional Laplacian.
//   For lattice implementation we use the "fractional bond noise"
//   by drawing bond noise with amplitude |k|^α weighting.
//
//   Simplified discrete implementation:
//   For each site i, the noise increment is:
//   dθ_i = sd * [α * (bond noise, F1-type) + (1-α) * white noise]
//   This interpolates linearly between white noise (α=0) and F1 (α=1).
//   (Exact fractional noise requires FFT; this is a practical surrogate.)
//
// Output files
// ------------
//   f2_timeseries.csv   step, t, m, n_defects
//   f2_correlation.csv  r, C_r
//   f2_alpha_scan.csv   L, alpha, m_mean, m_std, binder
//   f2_size_scan.csv    L, alpha, m_mean, m_std, binder
//
// Cargo.toml deps:  rand = "0.8"  rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, Uniform};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

struct XYLattice { l: usize, theta: Vec<f64> }
impl XYLattice {
    #[inline] fn idx(&self,i:usize,j:usize)->usize{i*self.l+j}
    #[inline] fn w(&self,x:usize,d:i64)->usize{
        ((x as i64+d).rem_euclid(self.l as i64)) as usize}
    #[inline] fn get(&self,i:usize,j:usize)->f64{self.theta[self.idx(i,j)]}
    fn new_random(l:usize,rng:&mut SmallRng)->Self{
        let d=Uniform::new(0.0f64,2.0*PI);
        XYLattice{l,theta:(0..l*l).map(|_|rng.sample(d)).collect()}
    }
}

/// α-interpolated noise:
///   α=0 → pure white noise (no LRO)
///   α=1 → pure conservative bond noise (F1, LRO)
fn em_step(lat:&mut XYLattice, j:f64, d:f64, alpha:f64, dt:f64, rng:&mut SmallRng){
    let l=lat.l; let n=l*l;
    let nd=Normal::new(0.0f64,1.0).unwrap();
    // XY drift
    let mut dth=vec![0.0f64;n];
    for i in 0..l { for jj in 0..l {
        let k=lat.idx(i,jj); let th=lat.theta[k];
        let up=lat.get(lat.w(i,1),jj); let dn=lat.get(lat.w(i,-1),jj);
        let rt=lat.get(i,lat.w(jj,1)); let lf=lat.get(i,lat.w(jj,-1));
        dth[k]=j*((up-th).sin()+(dn-th).sin()+(rt-th).sin()+(lf-th).sin())*dt;
    }}
    let sd=(2.0*d*dt).sqrt();
    // white noise component (α=0 limit)
    let sd_w=sd*(1.0-alpha);
    if sd_w>1e-15 { for k in 0..n { dth[k]+=sd_w*rng.sample(nd); } }
    // conservative bond noise component (α=1 limit)
    let sd_c=sd*alpha;
    if sd_c>1e-15 {
        for i in 0..l { for jj in 0..l {
            let zeta=sd_c*rng.sample(nd);
            let k0=lat.idx(i,jj); let k1=lat.idx(i,lat.w(jj,1));
            dth[k0]+=zeta; dth[k1]-=zeta;
            let zeta=sd_c*rng.sample(nd);
            let k0=lat.idx(i,jj); let k1=lat.idx(lat.w(i,1),jj);
            dth[k0]+=zeta; dth[k1]-=zeta;
        }}
    }
    for k in 0..n { lat.theta[k]+=dth[k]; }
}

fn order_param(lat:&XYLattice)->f64{
    let n=(lat.l*lat.l) as f64;
    let re=lat.theta.iter().map(|t|t.cos()).sum::<f64>()/n;
    let im=lat.theta.iter().map(|t|t.sin()).sum::<f64>()/n;
    (re*re+im*im).sqrt()
}
fn correlation(lat:&XYLattice,r_max:usize)->Vec<f64>{
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
#[inline] fn adiff(a:f64,b:f64)->f64{let d=b-a;d-(2.0*PI)*(d/(2.0*PI)).round()}
fn count_defects(lat:&XYLattice)->usize{
    let l=lat.l; let mut pos=0usize;
    for i in 0..l{let ip=(i+1)%l; for j in 0..l{let jp=(j+1)%l;
        let w=adiff(lat.get(i,j),lat.get(ip,j))+adiff(lat.get(ip,j),lat.get(ip,jp))
             +adiff(lat.get(ip,jp),lat.get(i,jp))+adiff(lat.get(i,jp),lat.get(i,j));
        if((w/(2.0*PI)).round() as i32).abs()==1{pos+=1;}
    }}
    pos
}
fn mean_f(v:&[f64])->f64{v.iter().sum::<f64>()/v.len() as f64}
fn std_f(v:&[f64])->f64{let mu=mean_f(v);(v.iter().map(|x|(x-mu).powi(2)).sum::<f64>()/v.len() as f64).sqrt()}

struct RunResult{l:usize,alpha:f64,m_mean:f64,m_std:f64,binder:f64,c_r:Vec<f64>,m_ts:Vec<f64>,ndef_ts:Vec<usize>,dt:f64}
fn run(l:usize,j:f64,d:f64,alpha:f64,dt:f64,t_trans:f64,t_meas:f64,mevery:usize,seed:u64,verbose:bool)->RunResult{
    let mut rng=SmallRng::seed_from_u64(seed);
    let mut lat=XYLattice::new_random(l,&mut rng);
    let n_tr=(t_trans/dt).round() as usize; let n_me=(t_meas/dt).round() as usize;
    for _ in 0..n_tr{em_step(&mut lat,j,d,alpha,dt,&mut rng);}
    if verbose{println!("[F2] L={l} α={alpha:.3} J={j} D={d:.2} transient done");}
    let mut m_ts=Vec::new(); let mut ndef_ts=Vec::new();
    for s in 0..n_me{em_step(&mut lat,j,d,alpha,dt,&mut rng);
        if s%mevery==0{m_ts.push(order_param(&lat));ndef_ts.push(count_defects(&lat));}}
    let c_r=correlation(&lat,l/2); let m_mean=mean_f(&m_ts); let m_std=std_f(&m_ts); let bdr=binder(&m_ts);
    let nd_avg=ndef_ts.iter().sum::<usize>() as f64/ndef_ts.len() as f64;
    if verbose{println!("      m={m_mean:.4} U_L={bdr:.4} avg_def={nd_avg:.1}");}
    RunResult{l,alpha,m_mean,m_std,binder:bdr,c_r,m_ts,ndef_ts,dt}
}
struct SRow{l:usize,alpha:f64,m_mean:f64,m_std:f64,binder:f64}
fn alpha_scan(l:usize,a_arr:&[f64],j:f64,d:f64,dt:f64,tt:f64,tm:f64,seed:u64)->Vec<SRow>{
    a_arr.iter().map(|&a|{let r=run(l,j,d,a,dt,tt,tm,20,seed,false);
    println!("  L={l} α={a:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
    SRow{l,alpha:a,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}}).collect()
}
fn size_scan(l_arr:&[usize],j:f64,d:f64,alpha:f64,dt:f64,tt:f64,tm:f64,seed:u64)->Vec<SRow>{
    l_arr.iter().map(|&l|{let r=run(l,j,d,alpha,dt,tt,tm,20,seed,false);
    println!("  L={l} α={alpha:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
    SRow{l,alpha,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}}).collect()
}
fn write_ts(path:&str,res:&RunResult,me:usize){
    let mut w=BufWriter::new(File::create(path).unwrap()); writeln!(w,"step,t,m,n_defects").unwrap();
    for(k,(&m,&nd))in res.m_ts.iter().zip(&res.ndef_ts).enumerate(){
        let s=k*me;writeln!(w,"{},{:.6},{:.8},{}",s,s as f64*res.dt,m,nd).unwrap();}
    println!("Written: {path}");
}
fn write_corr(path:&str,c:&[f64]){
    let mut w=BufWriter::new(File::create(path).unwrap()); writeln!(w,"r,C_r").unwrap();
    for(r,&v)in c.iter().enumerate(){writeln!(w,"{},{:.8}",r,v).unwrap();}
    println!("Written: {path}");
}
fn write_scan(path:&str,rows:&[SRow]){
    let mut w=BufWriter::new(File::create(path).unwrap()); writeln!(w,"L,alpha,m_mean,m_std,binder").unwrap();
    for r in rows{writeln!(w,"{},{:.4},{:.6},{:.6},{:.6}",r.l,r.alpha,r.m_mean,r.m_std,r.binder).unwrap();}
    println!("Written: {path}");
}
fn main(){
    let me=20usize;
    let res=run(32,1.0,0.5,0.8,0.01,20.0,100.0,me,42,true);
    write_ts("f2_timeseries.csv",&res,me);
    write_corr("f2_correlation.csv",&res.c_r);
    // α scan: α=0 (white, no LRO) → α=1.0 (full F1, LRO)
    let a_arr:Vec<f64>=(0..=10).map(|k|k as f64*0.1).collect();
    let ar=alpha_scan(32,&a_arr,1.0,0.5,0.01,15.0,60.0,0);
    write_scan("f2_alpha_scan.csv",&ar);
    let lr=size_scan(&[16,24,32,48],1.0,0.5,0.8,0.01,15.0,60.0,1);
    write_scan("f2_size_scan.csv",&lr);
}
