// ============================================================
// model_c2_heisenberg_vicsek.rs  —  Model C2  🔴 P1
// O(3) Heisenberg + Vicsek-Type Self-Propulsion
// ============================================================
// Model family : C族 — O(N≥3)矢量场 + 各类非平衡机制
// Priority     : 🔴 P1  (LRO probability ≥ 70%)
//
// Physics motivation
// ------------------
//   Toner & Tu (1995, 1998) proved that self-propelled flocking
//   (Vicsek) has LRO in 2D for ANY N-component order parameter
//   via the non-linear advective term.  Model C2 tests this for
//   O(3) Heisenberg spins: each spin n_i is attached to a particle
//   moving with velocity v₀ n_i, and the local alignment rule
//   replaces thermal equilibrium.
//
// Dynamics
// --------
//   Particles move: r_i(t+dt) = r_i(t) + v₀ n_i(t) dt  (on torus)
//   Spin update (discrete Vicsek-type):
//     n_i(t+dt) = Normalise( ⟨n_j⟩_{j∈B(r_i,R)} + √(2D dt) ξ_i )
//
//   ⟨n_j⟩ is the average of all spins (including i) within radius R
//   of particle i at time t (metric neighbourhood).
//   ξ_i ~ N(0,I₃) projected to tangent space of the updated direction.
//
//   PBC: positions on [0,L)², distances computed with wrap-around.
//   For efficiency (small L), we use a brute-force O(N²) neighbour search.
//
// Parameters
// ----------
//   N_part : number of particles (default L²/4 for ~0.25 density)
//   v₀     : self-propulsion speed
//   R      : alignment radius
//   D      : rotational noise
//   L      : system size
//
// Observables (→ CSV)
// -------------------
//   phi(t) = |⟨n⟩|           polar order parameter
//   C(r)   ≈ ⟨n_i·n_j⟩ as function of |r_i−r_j| (binned)
//   U_L    = Binder cumulant
//
// Output files
// ------------
//   c2_timeseries.csv     step, t, phi
//   c2_correlation.csv    r_bin, C_r
//   c2_noise_scan.csv     N_part, D, phi_mean, phi_std, binder
//   c2_size_scan.csv      L, D, phi_mean, phi_std, binder
//
// Cargo deps: rand = "0.8" (features=["small_rng"]), rand_distr = "0.4"
// ============================================================

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal, Uniform, UnitSphere};
use std::fs::File;
use std::io::{BufWriter, Write};

type Vec3 = [f64; 3];

// ─────────────────────────────────────────────────────────────
// § 1  Helpers
// ─────────────────────────────────────────────────────────────

#[inline] fn dot3(a: &Vec3, b: &Vec3) -> f64 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
#[inline] fn add3(a: Vec3, b: Vec3) -> Vec3 { [a[0]+b[0],a[1]+b[1],a[2]+b[2]] }
#[inline] fn scale3(s: f64, v: Vec3) -> Vec3 { [s*v[0],s*v[1],s*v[2]] }
#[inline] fn norm3(v: Vec3) -> Vec3 {
    let r = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();
    if r < 1e-30 { [1.0,0.0,0.0] } else { [v[0]/r,v[1]/r,v[2]/r] }
}

#[inline]
fn wrap_dist_sq(dx: f64, dy: f64, l: f64) -> f64 {
    let dx2 = dx - l * (dx / l).round();
    let dy2 = dy - l * (dy / l).round();
    dx2*dx2 + dy2*dy2
}

// ─────────────────────────────────────────────────────────────
// § 2  Particle system
// ─────────────────────────────────────────────────────────────

struct VicsekO3 {
    np:  usize,
    l:   f64,
    pos: Vec<[f64;2]>,   // (x, y) positions
    spn: Vec<Vec3>,      // O(3) spin directions
}

impl VicsekO3 {
    fn new_random(np: usize, l: f64, rng: &mut SmallRng) -> Self {
        let ux = Uniform::new(0.0f64, l);
        let us = UnitSphere;
        let pos: Vec<[f64;2]> = (0..np).map(|_| [rng.sample(ux), rng.sample(ux)]).collect();
        let spn: Vec<Vec3>    = (0..np).map(|_| { let v = rng.sample(us); [v[0],v[1],v[2]] }).collect();
        VicsekO3 { np, l, pos, spn }
    }
}

// ─────────────────────────────────────────────────────────────
// § 3  One Vicsek step
// ─────────────────────────────────────────────────────────────

fn vicsek_step(sys: &mut VicsekO3, v0: f64, r: f64, d: f64,
               dt: f64, rng: &mut SmallRng) {
    let np = sys.np;
    let l  = sys.l;
    let nd = Normal::new(0.0f64, 1.0).unwrap();
    let r2 = r * r;
    let sd = (2.0 * d * dt).sqrt();

    // Compute new spin directions (vectorised average in neighbourhood)
    let mut new_spn = vec![[0.0f64;3]; np];
    for i in 0..np {
        let mut avg = sys.spn[i]; // include self
        let (xi, yi) = (sys.pos[i][0], sys.pos[i][1]);
        for j in 0..np {
            if j == i { continue; }
            let dx = sys.pos[j][0] - xi;
            let dy = sys.pos[j][1] - yi;
            if wrap_dist_sq(dx, dy, l) <= r2 {
                avg = add3(avg, sys.spn[j]);
            }
        }
        // add noise in tangent space of avg direction
        let n_avg = norm3(avg);
        let noise_raw = [sd * rng.sample(nd),
                         sd * rng.sample(nd),
                         sd * rng.sample(nd)];
        // project noise perpendicular to n_avg
        let c = dot3(&n_avg, &noise_raw);
        let noise = [noise_raw[0] - c*n_avg[0],
                     noise_raw[1] - c*n_avg[1],
                     noise_raw[2] - c*n_avg[2]];
        new_spn[i] = norm3(add3(n_avg, noise));
    }
    sys.spn = new_spn;

    // Move particles
    for i in 0..np {
        sys.pos[i][0] = (sys.pos[i][0] + v0 * sys.spn[i][0] * dt).rem_euclid(l);
        sys.pos[i][1] = (sys.pos[i][1] + v0 * sys.spn[i][1] * dt).rem_euclid(l);
    }
}

// ─────────────────────────────────────────────────────────────
// § 4  Observables
// ─────────────────────────────────────────────────────────────

fn polar_order(sys: &VicsekO3) -> f64 {
    let nf = sys.np as f64;
    let m: Vec3 = sys.spn.iter().fold([0.0;3], |acc, s| add3(acc, *s));
    let m = scale3(1.0/nf, m);
    dot3(&m, &m).sqrt()
}

fn corr_binned(sys: &VicsekO3, n_bins: usize) -> Vec<f64> {
    let l = sys.l;
    let r_max = l * 0.5;
    let dr = r_max / n_bins as f64;
    let mut sum  = vec![0.0f64; n_bins];
    let mut cnt  = vec![0usize; n_bins];
    for i in 0..sys.np {
        for j in (i+1)..sys.np {
            let dx = sys.pos[j][0] - sys.pos[i][0];
            let dy = sys.pos[j][1] - sys.pos[i][1];
            let r  = wrap_dist_sq(dx, dy, l).sqrt();
            let b  = (r / dr) as usize;
            if b < n_bins {
                sum[b]  += dot3(&sys.spn[i], &sys.spn[j]);
                cnt[b]  += 1;
            }
        }
    }
    sum.iter().zip(&cnt)
       .map(|(s,&c)| if c > 0 { s / c as f64 } else { 0.0 })
       .collect()
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
// § 5  Runner
// ─────────────────────────────────────────────────────────────

struct RunResult { np: usize, d: f64,
                   phi_mean: f64, phi_std: f64, binder: f64,
                   c_r: Vec<f64>, phi_ts: Vec<f64>, dt: f64 }

fn run(np: usize, l: f64, v0: f64, r: f64, d: f64, dt: f64,
       t_trans: f64, t_meas: f64, mevery: usize, seed: u64, verbose: bool
) -> RunResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut sys = VicsekO3::new_random(np, l, &mut rng);
    let n_tr = (t_trans / dt).round() as usize;
    let n_me = (t_meas  / dt).round() as usize;

    for _ in 0..n_tr { vicsek_step(&mut sys, v0, r, d, dt, &mut rng); }
    if verbose {
        println!("[C2] N={np} L={l:.1} v₀={v0:.2} R={r:.2} D={d:.2} \
                  transient done");
    }

    let mut phi_ts = Vec::new();
    for s in 0..n_me {
        vicsek_step(&mut sys, v0, r, d, dt, &mut rng);
        if s % mevery == 0 { phi_ts.push(polar_order(&sys)); }
    }
    let c_r     = corr_binned(&sys, l as usize / 2);
    let phi_mean = mean_f(&phi_ts);
    let phi_std  = std_f(&phi_ts);
    let bdr      = binder(&phi_ts);
    if verbose { println!("      φ = {phi_mean:.4} ± {phi_std:.4}  U_L = {bdr:.4}"); }
    RunResult { np, d, phi_mean, phi_std, binder: bdr, c_r, phi_ts, dt }
}

// ─────────────────────────────────────────────────────────────
// § 6  Scans
// ─────────────────────────────────────────────────────────────

struct SRow { np: usize, l: f64, d: f64, phi_mean: f64, phi_std: f64, binder: f64 }

fn noise_scan(np: usize, l: f64, d_arr: &[f64], v0: f64, r: f64,
              dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    d_arr.iter().map(|&dv| {
        let r2 = run(np, l, v0, r, dv, dt, tt, tm, 20, seed, false);
        println!("  N={np} D={dv:.3}  φ={:.4}  U_L={:.4}", r2.phi_mean, r2.binder);
        SRow { np, l, d: dv, phi_mean: r2.phi_mean, phi_std: r2.phi_std, binder: r2.binder }
    }).collect()
}

fn size_scan(l_arr: &[usize], np_fn: impl Fn(usize)->usize,
             v0: f64, r: f64, d: f64,
             dt: f64, tt: f64, tm: f64, seed: u64) -> Vec<SRow> {
    l_arr.iter().map(|&l| {
        let np = np_fn(l);
        let lf = l as f64;
        let r2 = run(np, lf, v0, r, d, dt, tt, tm, 20, seed, false);
        println!("  L={l} N={np} D={d:.2}  φ={:.4}  U_L={:.4}", r2.phi_mean, r2.binder);
        SRow { np, l: lf, d, phi_mean: r2.phi_mean, phi_std: r2.phi_std, binder: r2.binder }
    }).collect()
}

// ─────────────────────────────────────────────────────────────
// § 7  CSV helpers
// ─────────────────────────────────────────────────────────────

fn write_ts(path: &str, phi_ts: &[f64], dt: f64, me: usize) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "step,t,phi").unwrap();
    for (k,&p) in phi_ts.iter().enumerate() {
        let s = k*me;
        writeln!(w, "{},{:.6},{:.8}", s, s as f64 * dt, p).unwrap();
    }
    println!("Written: {path}");
}

fn write_corr(path: &str, c: &[f64]) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "r_bin,C_r").unwrap();
    for (r,&v) in c.iter().enumerate() { writeln!(w, "{},{:.8}", r, v).unwrap(); }
    println!("Written: {path}");
}

fn write_scan(path: &str, rows: &[SRow]) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    writeln!(w, "N_part,L,D,phi_mean,phi_std,binder").unwrap();
    for r in rows {
        writeln!(w, "{},{:.1},{:.4},{:.6},{:.6},{:.6}",
                 r.np, r.l, r.d, r.phi_mean, r.phi_std, r.binder).unwrap();
    }
    println!("Written: {path}");
}

// ─────────────────────────────────────────────────────────────
// § 8  main
// ─────────────────────────────────────────────────────────────

fn main() {
    let me = 20usize;
    // single run: N=100, L=20, v₀=0.3, R=1.0, D=0.1
    let res = run(100, 20.0, 0.3, 1.0, 0.1, 0.1, 50.0, 200.0, me, 42, true);
    write_ts("c2_timeseries.csv", &res.phi_ts, res.dt, me);
    write_corr("c2_correlation.csv", &res.c_r);

    // noise scan D ∈ [0.05, 1.0]
    let d_arr: Vec<f64> = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0].to_vec();
    let dr = noise_scan(100, 20.0, &d_arr, 0.3, 1.0, 0.1, 30.0, 100.0, 0);
    write_scan("c2_noise_scan.csv", &dr);

    // finite-size scan: N ∝ L² (density=0.25), D=0.1
    let lr = size_scan(&[10,15,20,25], |l| l*l/4,
                       0.3, 1.0, 0.1, 0.1, 30.0, 100.0, 1);
    write_scan("c2_size_scan.csv", &lr);
}
