# Rust Simulation Codes — README
*Translated from Python; grounded on `model_simulation_summary.md` + `Knowledge/method.md`*

---

## File Inventory

| Source file | Cargo manifest | Model | Method |
|-------------|---------------|-------|--------|
| `xy_mc.rs` | `xy_mc_Cargo.toml` | 2D XY (equilibrium) | Monte Carlo – Metropolis |
| `kuramoto.rs` | `kuramoto_Cargo.toml` | Kuramoto oscillator lattice | ODE – Euler / RK4 |
| `vicsek.rs` | `vicsek_Cargo.toml` | Vicsek self-propelled particles | Discrete-time particle dynamics |
| `rotor_sde.rs` | `rotor_sde_Cargo.toml` | Self-spinning rotor XY | SDE – Euler-Maruyama |

---

## Quick Start

```bash
# One-shot: create all Cargo projects, copy sources, build everything
chmod +x setup_rust_projects.sh
./setup_rust_projects.sh

# Then run each binary (CSV outputs appear in the project folder):
cd xy_mc      && ./target/release/xy_mc
cd kuramoto   && ./target/release/kuramoto
cd vicsek     && ./target/release/vicsek
cd rotor_sde  && ./target/release/rotor_sde
```

Or build and run a single model manually:

```bash
mkdir -p xy_mc/src
cp xy_mc.rs         xy_mc/src/main.rs
cp xy_mc_Cargo.toml xy_mc/Cargo.toml
cd xy_mc && cargo run --release
```

---

## Dependencies  (`Cargo.toml` for all four)

```toml
[dependencies]
rand       = { version = "0.8", features = ["small_rng"] }
rand_distr = "0.4"
```

No other external crates needed.

---

## Model Details & Output Files

### 1. `xy_mc.rs` — 2D XY Model  (Monte Carlo / Metropolis)
**Equation:**  H = −J Σ_{⟨i,j⟩} cos(θᵢ − θⱼ),  θᵢ ∈ [0,2π),  L×L lattice, periodic BC

| Output CSV | Columns | Description |
|-----------|---------|-------------|
| `xy_timeseries.csv` | `step, m` | Magnetisation time series (production run) |
| `xy_correlation.csv` | `r, C_r` | Spin-spin correlation C(r) at end of run |
| `xy_binder_scan.csv` | `L, T, m_mean, m_std, binder` | Binder cumulant U_L scan over T and L |

**Key functions:**

| Rust fn | Python equivalent | Role |
|---------|------------------|------|
| `mc_sweep` | `mc_sweep` | N=L² Metropolis trials per sweep |
| `magnetisation` | `magnetisation` | m = \|⟨e^{iθ}⟩\| |
| `spin_correlation` | `spin_correlation` | C(r) = ⟨cos(θᵢ−θᵢ₊ᵣ)⟩ |
| `binder_cumulant` | `binder_cumulant` | U_L = 1−⟨m⁴⟩/(3⟨m²⟩²) |
| `run_xy_mc` | `run_xy_mc` | Main runner |
| `temperature_scan` | `temperature_scan` | FSS scan over T and L |

---

### 2. `kuramoto.rs` — Kuramoto Oscillator Lattice  (ODE)
**Equation:**  dφᵢ/dt = ωᵢ + K Σ_{⟨ij⟩} sin(φⱼ−φᵢ),  ωᵢ~N(0,σ²),  d-dim hypercubic lattice

| Output CSV | Columns | Description |
|-----------|---------|-------------|
| `kuramoto_timeseries.csv` | `step, t, r` | Synchronisation order parameter r(t) |
| `kuramoto_omega_eff.csv` | `osc_index, omega_eff` | Effective frequency per oscillator |
| `kuramoto_coupling_scan.csv` | `d, L, K, r_mean, r_std` | r vs K for d=2,3 |

**Key functions:**

| Rust fn | Python equivalent | Role |
|---------|------------------|------|
| `build_neighbour_table` | `_build_neighbour_table` | Precompute d-dim periodic neighbours |
| `dphi_dt` | `dphi_dt` | RHS of Kuramoto ODE |
| `euler_step` / `rk4_step` | same | Time integration (selectable) |
| `order_parameter` | `order_parameter` | r = \|⟨e^{iφ}⟩\| |
| `effective_frequencies` | `effective_frequencies` | ω_eff,i = Δφᵢ/Δt |
| `run_kuramoto` | `run_kuramoto` | Main runner |
| `coupling_scan` | `coupling_scan` | Scan K for fixed (L,d,σ) |

---

### 3. `vicsek.rs` — Vicsek Model  (Particle Dynamics)
**Equations:**  θᵢ(t+1) = circmean_{|rⱼ−rᵢ|<R}θⱼ(t) + ηᵢ;  rᵢ(t+1) = rᵢ(t) + v₀ê(θᵢ)

| Output CSV | Columns | Description |
|-----------|---------|-------------|
| `vicsek_timeseries.csv` | `step, phi` | Order parameter φᵥ(t) |
| `vicsek_noise_scan.csv` | `rho, eta, phi_mean, phi_std` | φᵥ vs η for ρ={2,4} |
| `vicsek_snapshot.csv` | `x, y, cos_theta, sin_theta` | Final particle snapshot |

**Key functions:**

| Rust fn | Python equivalent | Role |
|---------|------------------|------|
| `vicsek_step` | `vicsek_step` | Synchronous update (min-image BC) |
| `order_parameter` | `order_parameter_vicsek` | φᵥ = \|⟨e^{iθ}⟩\| |
| `run_vicsek` | `run_vicsek` | Main runner |
| `noise_scan` | `noise_scan` | Scan η for fixed (N,ρ) |

---

### 4. `rotor_sde.rs` — Self-Spinning Rotor  (SDE / Euler-Maruyama)
**Equation:**  dθᵢ = [Ωᵢ + J Σ sin(θⱼ−θᵢ)]dt + √(2D dt) ξᵢ,  Ωᵢ~N(0,σ²)

| Output CSV | Columns | Description |
|-----------|---------|-------------|
| `rotor_timeseries.csv` | `step, m, n_defects` | Magnetisation + defect count time series |
| `rotor_correlation.csv` | `r, C_r` | Spin-spin correlation |
| `rotor_sigma_scan.csv` | `sigma, m_mean, m_std, avg_defects` | m vs σ (disorder scan) |

**Key functions:**

| Rust fn | Python equivalent | Role |
|---------|------------------|------|
| `drift` | `drift` | Deterministic part f(θ) |
| `em_step` | `em_step` | Euler-Maruyama SDE step |
| `magnetisation` | `magnetisation_r` | m = \|⟨e^{iθ}⟩\| |
| `spin_correlation` | `spin_correlation` | C(r) |
| `count_defects` | `detect_defects` | Winding-number vortex count |
| `run_spinning_rotor` | `run_spinning_rotor` | Main runner |
| `sigma_scan` | `sigma_scan` | Scan σ for fixed (L,J,D) |

---

## Design Principles

1. **One-to-one correspondence with Python** — every function name, parameter order,
   and algorithm step directly mirrors the Python version.
2. **No unsafe code** — all array accesses are bounds-checked; periodic wrapping
   uses `.rem_euclid()`.
3. **Single external dependency** — only `rand` + `rand_distr`; no special
   linear-algebra or simulation library needed.
4. **CSV output instead of matplotlib** — use any plotting tool (Python/matplotlib,
   gnuplot, Julia, R) to visualise the results.
5. **`SmallRng`** — fast, seedable PRNG (xoshiro128++) from the `rand` crate,
   analogous to `numpy.random.default_rng(seed)`.
