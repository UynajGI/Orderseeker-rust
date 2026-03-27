// ============================================================
// model_i3_xy_oscillating_shear.rs  —  Model I3  🟠 P2
// XY Model + Oscillating Shear Flow  γ̇(t) = γ̇₀ cos(ωt)
// ============================================================
// Model family : I族 — 剪切流/流场机制的变体与扩展
// Priority     : 🟠 P2  (LRO probability: medium)
//
// Physics motivation
// ------------------
//   Steady shear (C1, A1) stabilises LRO via k^{-2/3} suppression.
//   Oscillating shear has zero time-average flow: ⟨γ̇⟩=0.
//   Does the instantaneous non-equilibrium still stabilise LRO,
//   or does the cancellation destroy the shear mechanism?
//   This probes whether the mechanism is "flow path" (trajectory
//   dependent) or "instantaneous" (driven by |γ̇(t)|).
//
// Equation of motion  (Euler-Maruyama SDE)
// -----------------------------------------
//   dθ_i = [ J Σ_{⟨ij⟩} sin(θ_j − θ_i)
//            + γ̇₀ cos(ω_osc t)  y_i  (θ_{i,j+1}−θ_{i,j-1})/2 ] dt
//          + √(2D dt) ξ_i
//
// Output files
// ------------
//   i3_timeseries.csv    step, t, m
//   i3_correlation.csv   r, C_r
//   i3_gd0_scan.csv      L, gamma_dot0, omega_osc, m_mean, m_std, binder
//   i3_size_scan.csv     L, gamma_dot0, omega_osc, m_mean, m_std, binder
//
// Cargo.toml deps:  rand = "0.8"  rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, Uniform};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

struct ShearLattice{l:usize,theta:Vec<f64>}
impl ShearLattice{
    #[inline] fn idx(&self,i:usize,j:usize)->usize{i*self.l+j}
    #[inline] fn w(&self,x:usize,d:i64)->usize{((x as i64+d).rem_euclid(self.l as i64)) as usize}
    #[inline] fn get(&self,i:usize,j:usize)->f64{self.theta[self.idx(i,j)]}
    fn new_random(l:usize,rng:&mut SmallRng)->Self{
        let d=Uniform::new(0.0f64,2.0*PI);
        ShearLattice{l,theta:(0..l*l).map(|_|rng.sample(d)).collect()}
    }
}

fn em_step(lat:&mut ShearLattice,j:f64,gd0:f64,omega_osc:f64,d:f64,dt:f64,t:f64,rng:&mut SmallRng){
    let gd=gd0*(omega_osc*t).cos();  // oscillating shear rate
    let l=lat.l; let n=l*l;
    let nd=Normal::new(0.0f64,1.0).unwrap();
    let sd=(2.0*d*dt).sqrt();
    let mut dth=vec![0.0f64;n];
    for i in 0..l { let yi=i as f64;
        for jj in 0..l {
            let k=lat.idx(i,jj); let th=lat.theta[k];
            let up=lat.get(lat.w(i,1),jj); let dn=lat.get(lat.w(i,-1),jj);
            let rt=lat.get(i,lat.w(jj,1)); let lf=lat.get(i,lat.w(jj,-1));
            dth[k]=j*((up-th).sin()+(dn-th).sin()+(rt-th).sin()+(lf-th).sin())*dt
                  +gd*yi*(rt-lf)*0.5*dt
                  +sd*rng.sample(nd);
        }
    }
    for k in 0..n{lat.theta[k]+=dth[k];}
}

fn order_param(lat:&ShearLattice)->f64{
    let n=(lat.l*lat.l) as f64;
    let re=lat.theta.iter().map(|t|t.cos()).sum::<f64>()/n;
    let im=lat.theta.iter().map(|t|t.sin()).sum::<f64>()/n;
    (re*re+im*im).sqrt()
}
fn correlation(lat:&ShearLattice,r_max:usize)->Vec<f64>{
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

struct RunResult{l:usize,gd0:f64,omega_osc:f64,m_mean:f64,m_std:f64,binder:f64,c_r:Vec<f64>,m_ts:Vec<f64>,dt:f64}
fn run(l:usize,j:f64,gd0:f64,omega_osc:f64,d:f64,dt:f64,t_trans:f64,t_meas:f64,mevery:usize,seed:u64,verbose:bool)->RunResult{
    let mut rng=SmallRng::seed_from_u64(seed);
    let mut lat=ShearLattice::new_random(l,&mut rng);
    let n_tr=(t_trans/dt).round() as usize; let n_me=(t_meas/dt).round() as usize;
    for s in 0..n_tr{em_step(&mut lat,j,gd0,omega_osc,d,dt,s as f64*dt,&mut rng);}
    if verbose{println!("[I3] L={l} γ̇₀={gd0:.3} ω_osc={omega_osc:.3} transient done");}
    let mut m_ts=Vec::new();
    for s in 0..n_me{
        let t=(n_tr+s) as f64*dt;
        em_step(&mut lat,j,gd0,omega_osc,d,dt,t,&mut rng);
        if s%mevery==0{m_ts.push(order_param(&lat));}
    }
    let c_r=correlation(&lat,l/2); let m_mean=mean_f(&m_ts); let m_std=std_f(&m_ts); let bdr=binder(&m_ts);
    if verbose{println!("      m={m_mean:.4} U_L={bdr:.4}");}
    RunResult{l,gd0,omega_osc,m_mean,m_std,binder:bdr,c_r,m_ts,dt}
}
struct SRow{l:usize,gd0:f64,omega_osc:f64,m_mean:f64,m_std:f64,binder:f64}
fn gd0_scan(l:usize,gd0_arr:&[f64],omega_osc:f64,j:f64,d:f64,dt:f64,tt:f64,tm:f64,seed:u64)->Vec<SRow>{
    gd0_arr.iter().map(|&gd0|{let r=run(l,j,gd0,omega_osc,d,dt,tt,tm,20,seed,false);
    println!("  L={l} γ̇₀={gd0:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
    SRow{l,gd0,omega_osc,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}}).collect()
}
fn size_scan(la:&[usize],j:f64,gd0:f64,omega_osc:f64,d:f64,dt:f64,tt:f64,tm:f64,seed:u64)->Vec<SRow>{
    la.iter().map(|&l|{let r=run(l,j,gd0,omega_osc,d,dt,tt,tm,20,seed,false);
    println!("  L={l} γ̇₀={gd0:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
    SRow{l,gd0,omega_osc,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}}).collect()
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
    let mut w=BufWriter::new(File::create(path).unwrap()); writeln!(w,"L,gamma_dot0,omega_osc,m_mean,m_std,binder").unwrap();
    for r in rows{writeln!(w,"{},{:.4},{:.4},{:.6},{:.6},{:.6}",r.l,r.gd0,r.omega_osc,r.m_mean,r.m_std,r.binder).unwrap();}
    println!("Written: {path}");
}
fn main(){
    let me=20usize;
    let res=run(32,1.0,1.0,1.0,0.5,0.01,20.0,100.0,me,42,true);
    write_ts("i3_timeseries.csv",&res.m_ts,res.dt,me);
    write_corr("i3_correlation.csv",&res.c_r);
    // γ̇₀ scan: compare with steady shear (A1 model)
    let gd0_arr:Vec<f64>=(0..=10).map(|k|k as f64*0.2).collect();
    let gr=gd0_scan(32,&gd0_arr,1.0,1.0,0.5,0.01,15.0,60.0,0);
    write_scan("i3_gd0_scan.csv",&gr);
    let lr=size_scan(&[16,24,32,48],1.0,1.0,1.0,0.5,0.01,15.0,60.0,1);
    write_scan("i3_size_scan.csv",&lr);
}
