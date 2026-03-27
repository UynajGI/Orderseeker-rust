// ============================================================
// model_c3_heisenberg_paired_active.rs  —  Model C3  🟠 P2
// O(3) Heisenberg + Paired Active Torques (conserving total angular momentum)
// ============================================================
// Model family : C族 — O(N≥3)矢量场 + 各类非平衡机制
// Priority     : 🟠 P2  (LRO probability: medium-high 40–70%)
//
// Physics motivation
// ------------------
//   Keta & Henkes (2024) showed that antisymmetric paired forces
//   with Σ_i F_i = 0 generate q² noise spectra in elastic bodies.
//   Model C3 generalises this to O(3) rotational degrees of freedom:
//   each bond (i,j) carries a random active torque τ_{ij} = −τ_{ji}
//   (conservation of angular momentum).  The active torque acts
//   tangentially on each spin, preserving |n_i| = 1.
//
// Equation of motion  (Euler-Maruyama SDE on S²)
// -----------------------------------------------
//   d n_i = P_⊥(n_i) [ J Σ_{j∈NN} n_j
//                     + Σ_{b∋i} τ_b e_b ]  dt
//           + √(2D dt) P_⊥(n_i) ξ_i
//   then normalise.
//
//   τ_b : scalar active torque on bond b = (i,j),  τ_{ij} = −τ_{ji}
//         OU process: dτ_b = −(1/τ_a) τ_b dt + √(2 D_a/τ_a) dW_b
//   e_b : unit vector perpendicular to n_i in the local tangent plane
//         (a "rotation axis" direction drawn randomly once per step)
//
// Parameters
// ----------
//   J     : coupling
//   D     : bulk thermal noise
//   D_a   : active torque noise
//   τ_a   : OU correlation time of active torques
//   dt    : time step
//
// Observables (→ CSV)
// -------------------
//   m(t)  = |⟨n⟩|   O(3) order parameter
//   C(r)  = spatial correlation along x
//   U_L   = Binder cumulant
//
// Output files
// ------------
//   c3_timeseries.csv    step, t, m
//   c3_correlation.csv   r, C_r
//   c3_da_scan.csv       L, D_a, tau_a, m_mean, m_std, binder
//   c3_size_scan.csv     L, D_a, tau_a, m_mean, m_std, binder
//
// Cargo deps: rand = "0.8" (features=["small_rng"]), rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, UnitSphere};
use std::fs::File;
use std::io::{BufWriter, Write};

type Vec3 = [f64; 3];

// ─────────────────────────────────────────────────────────────
// § 1  Helpers
// ─────────────────────────────────────────────────────────────

#[inline] fn dot3(a: &Vec3, b: &Vec3) -> f64 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
#[inline] fn add3(a: Vec3, b: Vec3) -> Vec3 { [a[0]+b[0],a[1]+b[1],a[2]+b[2]] }
#[inline] fn sub3(a: Vec3, b: Vec3) -> Vec3 { [a[0]-b[0],a[1]-b[1],a[2]-b[2]] }
#[inline] fn scale3(s: f64, v: Vec3) -> Vec3 { [s*v[0],s*v[1],s*v[2]] }
#[inline] fn norm3(v: Vec3) -> Vec3 {
    let r = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();
    if r < 1e-30 { [1.0,0.0,0.0] } else { [v[0]/r,v[1]/r,v[2]/r] }
}
#[inline] fn proj_tan(n: &Vec3, v: Vec3) -> Vec3 {
    sub3(v, scale3(dot3(n,&v), *n))
}

// ─────────────────────────────────────────────────────────────
// § 2  Lattice
// ─────────────────────────────────────────────────────────────

struct HA3Lattice {
    l:   usize,
    spn: Vec<Vec3>,
    // OU active torques on bonds:  horiz[i*l+j] for bond (i,j)→(i,j+1)
    //                              vert [i*l+j] for bond (i,j)→(i+1,j)
    fh: Vec<f64>,
    fv: Vec<f64>,
}

impl HA3Lattice {
    #[inline] fn idx(&self, i: usize, j: usize) -> usize { i*self.l+j }
    #[inline] fn w(&self, x: usize, d: i64) -> usize {
        ((x as i64+d).rem_euclid(self.l as i64)) as usize
    }
    #[inline] fn get(&self, i: usize, j: usize) -> Vec3 { self.spn[self.idx(i,j)] }

    fn new_random(l: usize, rng: &mut SmallRng) -> Self {
        let us = UnitSphere;
        let spn: Vec<Vec3> = (0..l*l).map(|_| { let v = rng.sample(us); [v[0],v[1],v[2]] }).collect();
        HA3Lattice { l, spn, fh: vec![0.0;l*l], fv: vec![0.0;l*l] }
    }
}

// ─────────────────────────────────────────────────────────────
// § 3  One SDE step
// ─────────────────────────────────────────────────────────────

fn em_step(lat: &mut HA3Lattice, j: f64, d: f64, d_a: f64,
           tau_a: f64, dt: f64, rng: &mut SmallRng) {
    let l  = lat.l;
    let n  = l * l;
    let nd = Normal::new(0.0f64, 1.0).unwrap();
    let sd = (2.0 * d * dt).sqrt();

    // ── update OU bond torques ───────────────────────────────
    let decay = (-dt/tau_a).exp();
    let ou_sd = (d_a*(1.0-decay*decay)).sqrt();
    for k in 0..n {
        lat.fh[k] = decay*lat.fh[k] + ou_sd*rng.sample(nd);
        lat.fv[k] = decay*lat.fv[k] + ou_sd*rng.sample(nd);
    }

    let spn_old = lat.spn.clone();

    for i in 0..l {
        for jj in 0..l {
            let k  = lat.idx(i,jj);
            let ni = spn_old[k];
            let up  = spn_old[lat.idx(lat.w(i, 1),  jj)];
            let dn  = spn_old[lat.idx(lat.w(i,-1),  jj)];
            let rt  = spn_old[lat.idx(i, lat.w(jj, 1))];
            let lf  = spn_old[lat.idx(i, lat.w(jj,-1))];

            // XY coupling
            let coup = [j*(up[0]+dn[0]+rt[0]+lf[0]),
                        j*(up[1]+dn[1]+rt[1]+lf[1]),
                        j*(up[2]+dn[2]+rt[2]+lf[2])];
            let d_xy = proj_tan(&ni, coup);

            // Active torques: each bond contributes a tangential push
            // Use a random tangent direction e_b for each bond
            let make_tan = |n: &Vec3, scalar: f64| -> Vec3 {
                let raw = [rng.sample(nd), rng.sample(nd), rng.sample(nd)];
                scale3(scalar, proj_tan(n, raw))
            };
            // Horiz right bond (i,jj)→(i,jj+1):  +τ_h to jj
            let torq = lat.fh[lat.idx(i,jj)] * dt;
            let act_hr = make_tan(&ni, torq);
            // Horiz left bond (i,jj-1)→(i,jj):  −τ_h to jj
            let torq2 = -lat.fh[lat.idx(i, lat.w(jj,-1))] * dt;
            let act_hl = make_tan(&ni, torq2);
            // Vert up bond (i,jj)→(i+1,jj):  +τ_v to jj
            let torq3 = lat.fv[lat.idx(i,jj)] * dt;
            let act_vu = make_tan(&ni, torq3);
            // Vert down bond (i-1,jj)→(i,jj):  −τ_v to jj
            let torq4 = -lat.fv[lat.idx(lat.w(i,-1), jj)] * dt;
            let act_vd = make_tan(&ni, torq4);

            let act_total = add3(add3(act_hr,act_hl),add3(act_vu,act_vd));

            // Thermal noise
            let noise_raw = [sd*rng.sample(nd),sd*rng.sample(nd),sd*rng.sample(nd)];
            let noise = proj_tan(&ni, noise_raw);

            let new_n = add3(add3(add3(ni, scale3(dt, d_xy)), act_total), noise);
            lat.spn[k] = norm3(new_n);
        }
    }
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
// ─────────────────────────────────────────────────────────────

fn order_param(lat: &HA3Lattice) -> f64 {
    let n = lat.l * lat.l;
    let m = lat.spn.iter().fold([0.0;3], |acc,s| add3(acc,*s));
    let m = scale3(1.0/n as f64, m);
    dot3(&m,&m).sqrt()
}

fn correlation(lat: &HA3Lattice, r_max: usize) -> Vec<f64> {
    let l = lat.l;
    let nf = (l*l) as f64;
    let mut c = vec![0.0f64; r_max+1];
    c[0] = 1.0;
    for r in 1..=r_max {
        c[r] = (0..l).flat_map(|i| (0..l).map(move |j| (i,j)))
            .map(|(i,j)| dot3(&lat.get(i,j), &lat.get(i,(j+r)%l)))
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

struct RunResult { l: usize, d_a: f64, tau_a: f64,
                   m_mean: f64, m_std: f64, binder: f64,
                   c_r: Vec<f64>, m_ts: Vec<f64>, dt: f64 }

fn run(l: usize, j: f64, d: f64, d_a: f64, tau_a: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut lat = HA3Lattice::new_random(l, &mut rng);
    let n_tr = (t_trans/dt).round() as usize;
    let n_me = (t_meas /dt).round() as usize;

    for _ in 0..n_tr { em_step(&mut lat,j,d,d_a,tau_a,dt,&mut rng); }
    if verbose {
        println!("[C3] L={l} J={j} D={d:.2} D_a={d_a:.2} τ_a={tau_a:.2} transient done");
    }

    let mut m_ts = Vec::new();
    for s in 0..n_me {
        em_step(&mut lat,j,d,d_a,tau_a,dt,&mut rng);
        if s % mevery == 0 { m_ts.push(order_param(&lat)); }
    }
    let c_r    = correlation(&lat, l/2);
    let m_mean = mean_f(&m_ts);
    let m_std  = std_f(&m_ts);
    let bdr    = binder(&m_ts);
    if verbose { println!("      m = {m_mean:.4} ± {m_std:.4}  U_L = {bdr:.4}"); }
    RunResult { l, d_a, tau_a, m_mean, m_std, binder: bdr, c_r, m_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { l: usize, d_a: f64, tau_a: f64,
              m_mean: f64, m_std: f64, binder: f64 }

fn da_scan(l: usize, da_arr: &[f64], tau_a: f64, j: f64, d: f64,
           dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    da_arr.iter().map(|&da| {
        let r = run(l,j,d,da,tau_a,dt,tt,tm,20,seed,false);
        println!("  L={l} D_a={da:.3} τ_a={tau_a:.2}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
        SRow { l, d_a: da, tau_a, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], j: f64, d: f64, d_a: f64, tau_a: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let r = run(l,j,d,d_a,tau_a,dt,tt,tm,20,seed,false);
        println!("  L={l} D_a={d_a:.3}  m={:.4}  U_L={:.4}",r.m_mean,r.binder);
        SRow { l, d_a, tau_a, m_mean: r.m_mean, m_std: r.m_std, binder: r.binder }
    }).collect()
}

// ─────────────────────────────────────────────────────────────
// § 7  CSV helpers
// ─────────────────────────────────────────────────────────────

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
    writeln!(w,"L,D_a,tau_a,m_mean,m_std,binder").unwrap();
    for r in rows {
        writeln!(w,"{},{:.4},{:.4},{:.6},{:.6},{:.6}",
                 r.l,r.d_a,r.tau_a,r.m_mean,r.m_std,r.binder).unwrap(); }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    let res = run(24,1.0,0.2,0.5,0.5,0.01,20.0,100.0,me,42,true);
    write_ts("c3_timeseries.csv",&res.m_ts,res.dt,me);
    write_corr("c3_correlation.csv",&res.c_r);
    let da_arr: Vec<f64> = (0..=10).map(|k| k as f64*0.1).collect();
    let dr = da_scan(24,&da_arr,0.5,1.0,0.2,0.01,15.0,60.0,0);
    write_scan("c3_da_scan.csv",&dr);
    let lr = size_scan(&[12,16,24,32],1.0,0.2,0.5,0.5,0.01,15.0,60.0,1);
    write_scan("c3_size_scan.csv",&lr);
}
