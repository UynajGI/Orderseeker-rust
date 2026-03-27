// ============================================================
// model_i2_heisenberg_extensional.rs  —  Model I2  🟠 P2
// O(3) Heisenberg Model + Extensional (Stretching) Flow
// ============================================================
// Model family : I族 — 剪切流/流场机制的变体与扩展
// Priority     : 🟠 P2  (LRO probability: high, equivalent to C1)
//
// Physics motivation
// ------------------
//   Complement to C1 (O(3) + shear flow).  Uses extensional flow
//   v_x = ε̇ x, v_y = −ε̇ y (incompressible, ∇·v=0) instead of shear.
//   Minami & Nakano (2022) showed extensional and shear flows are
//   equivalent in the 1/N analysis.  Model I2 tests this equivalence
//   numerically for O(3) spins.
//
// Equation of motion (projected Langevin on S²)
// ----------------------------------------------
//   d n̂_i = P_i [ J Σ_{⟨ij⟩} n̂_j
//                 + ε̇ x_i (n̂_{i+x̂}−n̂_{i−x̂})/2   [stretch ∂_x]
//                 − ε̇ y_i (n̂_{i+ŷ}−n̂_{i−ŷ})/2 ] dt  [compress ∂_y]
//           + √(2D) P_i dW_i
//
// Output files
// ------------
//   i2_timeseries.csv   step, t, m
//   i2_correlation.csv  r, C_r
//   i2_eps_scan.csv     L, eps_dot, m_mean, m_std, binder
//   i2_size_scan.csv    L, eps_dot, m_mean, m_std, binder
//
// Cargo.toml deps:  rand = "0.8"  rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, Uniform};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

type Vec3 = [f64; 3];
#[inline] fn dot3(a:&Vec3,b:&Vec3)->f64{a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
#[inline] fn add3(a:&Vec3,b:&Vec3)->Vec3{[a[0]+b[0],a[1]+b[1],a[2]+b[2]]}
#[inline] fn sc3(a:&Vec3,s:f64)->Vec3{[a[0]*s,a[1]*s,a[2]*s]}
#[inline] fn sub3(a:&Vec3,b:&Vec3)->Vec3{[a[0]-b[0],a[1]-b[1],a[2]-b[2]]}
#[inline] fn norm3(a:&Vec3)->f64{(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]).sqrt()}
#[inline] fn proj3(n:&Vec3,v:&Vec3)->Vec3{sub3(v,&sc3(n,dot3(n,v)))}
#[inline] fn normalise3(mut a:Vec3)->Vec3{let r=norm3(&a);if r>1e-15{a[0]/=r;a[1]/=r;a[2]/=r;}a}

struct HeisenLattice{l:usize,spin:Vec<Vec3>}
impl HeisenLattice{
    #[inline] fn idx(&self,i:usize,j:usize)->usize{i*self.l+j}
    #[inline] fn w(&self,x:usize,d:i64)->usize{((x as i64+d).rem_euclid(self.l as i64)) as usize}
    #[inline] fn get(&self,i:usize,j:usize)->Vec3{self.spin[self.idx(i,j)]}
    fn new_random(l:usize,rng:&mut SmallRng)->Self{
        let nd=Normal::new(0.0f64,1.0).unwrap();
        let spin:Vec<Vec3>=(0..l*l).map(|_|normalise3([rng.sample(nd),rng.sample(nd),rng.sample(nd)])).collect();
        HeisenLattice{l,spin}
    }
}

fn em_step(lat:&mut HeisenLattice,j:f64,eps:f64,d:f64,dt:f64,rng:&mut SmallRng){
    let l=lat.l; let n=l*l;
    let nd=Normal::new(0.0f64,1.0).unwrap();
    let sd=(2.0*d*dt).sqrt();
    let mut new_spin=lat.spin.clone();
    for i in 0..l { let yi=i as f64;
        for jj in 0..l { let xj=jj as f64;
            let k=lat.idx(i,jj); let ni=lat.spin[k];
            let up=lat.get(lat.w(i,1),jj); let dn=lat.get(lat.w(i,-1),jj);
            let rt=lat.get(i,lat.w(jj,1)); let lf=lat.get(i,lat.w(jj,-1));
            let mut dv=[0.0f64;3];
            for c in 0..3{
                dv[c]=j*(up[c]+dn[c]+rt[c]+lf[c])*dt
                     +eps*xj*(rt[c]-lf[c])*0.5*dt   // stretch ∂_x
                     -eps*yi*(up[c]-dn[c])*0.5*dt;   // compress ∂_y
            }
            let f=proj3(&ni,&dv);
            let raw:Vec3=[rng.sample(nd)*sd,rng.sample(nd)*sd,rng.sample(nd)*sd];
            let noise=proj3(&ni,&raw);
            new_spin[k]=normalise3(add3(&ni,&add3(&f,&noise)));
        }
    }
    lat.spin=new_spin;
}

fn order_param(lat:&HeisenLattice)->f64{
    let n=(lat.l*lat.l) as f64; let mut avg=[0.0f64;3];
    for s in &lat.spin{for c in 0..3{avg[c]+=s[c];}}
    norm3(&avg)/n
}
fn correlation(lat:&HeisenLattice,r_max:usize)->Vec<f64>{
    let l=lat.l; let n=(l*l) as f64;
    let mut c=vec![0.0f64;r_max+1]; c[0]=1.0;
    for r in 1..=r_max{
        c[r]=(0..l).flat_map(|i|(0..l).map(move|j|(i,j)))
            .map(|(i,j)|dot3(&lat.get(i,j),&lat.get(i,(j+r)%l))).sum::<f64>()/n;
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

struct RunResult{l:usize,eps:f64,m_mean:f64,m_std:f64,binder:f64,c_r:Vec<f64>,m_ts:Vec<f64>,dt:f64}
fn run(l:usize,j:f64,eps:f64,d:f64,dt:f64,t_trans:f64,t_meas:f64,mevery:usize,seed:u64,verbose:bool)->RunResult{
    let mut rng=SmallRng::seed_from_u64(seed);
    let mut lat=HeisenLattice::new_random(l,&mut rng);
    let n_tr=(t_trans/dt).round() as usize; let n_me=(t_meas/dt).round() as usize;
    for _ in 0..n_tr{em_step(&mut lat,j,eps,d,dt,&mut rng);}
    if verbose{println!("[I2] L={l} ε̇={eps:.3} J={j} D={d:.2} O(3) extensional transient done");}
    let mut m_ts=Vec::new();
    for s in 0..n_me{em_step(&mut lat,j,eps,d,dt,&mut rng);if s%mevery==0{m_ts.push(order_param(&lat));}}
    let c_r=correlation(&lat,l/2); let m_mean=mean_f(&m_ts); let m_std=std_f(&m_ts); let bdr=binder(&m_ts);
    if verbose{println!("      m={m_mean:.4} U_L={bdr:.4}");}
    RunResult{l,eps,m_mean,m_std,binder:bdr,c_r,m_ts,dt}
}
struct SRow{l:usize,eps:f64,m_mean:f64,m_std:f64,binder:f64}
fn eps_scan(l:usize,ea:&[f64],j:f64,d:f64,dt:f64,tt:f64,tm:f64,seed:u64)->Vec<SRow>{
    ea.iter().map(|&eps|{let r=run(l,j,eps,d,dt,tt,tm,20,seed,false);
    println!("  L={l} ε̇={eps:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
    SRow{l,eps,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}}).collect()
}
fn size_scan(la:&[usize],j:f64,eps:f64,d:f64,dt:f64,tt:f64,tm:f64,seed:u64)->Vec<SRow>{
    la.iter().map(|&l|{let r=run(l,j,eps,d,dt,tt,tm,20,seed,false);
    println!("  L={l} ε̇={eps:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
    SRow{l,eps,m_mean:r.m_mean,m_std:r.m_std,binder:r.binder}}).collect()
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
    let mut w=BufWriter::new(File::create(path).unwrap()); writeln!(w,"L,eps_dot,m_mean,m_std,binder").unwrap();
    for r in rows{writeln!(w,"{},{:.4},{:.6},{:.6},{:.6}",r.l,r.eps,r.m_mean,r.m_std,r.binder).unwrap();}
    println!("Written: {path}");
}
fn main(){
    let me=20usize;
    let res=run(32,1.0,0.3,0.3,0.01,20.0,100.0,me,42,true);
    write_ts("i2_timeseries.csv",&res.m_ts,res.dt,me);
    write_corr("i2_correlation.csv",&res.c_r);
    let ea:Vec<f64>=(0..=10).map(|k|k as f64*0.2).collect();
    let er=eps_scan(32,&ea,1.0,0.3,0.01,15.0,60.0,0);
    write_scan("i2_eps_scan.csv",&er);
    let lr=size_scan(&[16,24,32,48],1.0,0.3,0.3,0.01,15.0,60.0,1);
    write_scan("i2_size_scan.csv",&lr);
}
