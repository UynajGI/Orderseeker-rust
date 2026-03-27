# Rust 统计物理模型模拟代码库

> 从Python翻译而来，基于 `model_simulation_summary.md` 和 `Knowledge/method.md`

---

## 项目概述

本项目包含四个统计物理/非平衡态物理模型的Rust实现，用于研究**Mermin-Wagner定理**及其打破机制。所有代码均采用单一外部依赖策略（仅使用 `rand` + `rand_distr`），输出CSV格式数据，便于后续可视化分析。

---

## 文件结构

| 源文件 | Cargo配置 | 模型 | 数值方法 |
|--------|-----------|------|----------|
| `xy_mc.rs` | `xy_mc_Cargo.toml` | 2D XY模型（平衡态） | Monte Carlo – Metropolis |
| `kuramoto.rs` | `kuramoto_Cargo.toml` | Kuramoto振子晶格 | ODE – Euler / RK4 |
| `vicsek.rs` | `vicsek_Cargo.toml` | Vicsek自驱动粒子 | 离散时间粒子动力学 |
| `rotor_sde.rs` | `rotor_sde_Cargo.toml` | 自旋转转子XY | SDE – Euler-Maruyama |

---

## 模型详解

### 1. XY模型 (`xy_mc.rs`)

**物理背景**: 经典2D XY模型，研究BKT相变和准长程序（QLRO）

**哈密顿量**:
```
H = -J Σ_{⟨i,j⟩} cos(θᵢ - θⱼ)
```

**关键函数**:
| 函数 | 功能 |
|------|------|
| `mc_sweep` | 执行L²次Metropolis尝试 |
| `magnetisation` | 序参量 m = \|⟨e^{iθ}⟩\| |
| `spin_correlation` | 自旋-自旋关联 C(r) |
| `binder_cumulant` | Binder累积量 U_L（FSS分析） |

**输出文件**:
- `xy_timeseries.csv` — 磁化强度时间序列
- `xy_correlation.csv` — 自旋关联函数
- `xy_binder_scan.csv` — Binder累积量扫描（确定T_c）

---

### 2. Kuramoto振子模型 (`kuramoto.rs`)

**物理背景**: 研究同步相变，下临界维度 d_L = 2

**动力学方程**:
```
dφᵢ/dt = ωᵢ + K Σ_{⟨ij⟩} sin(φⱼ - φᵢ)
```
其中 ωᵢ ~ N(0, σ²) 为淬火无序

**关键函数**:
| 函数 | 功能 |
|------|------|
| `build_neighbour_table` | 预计算d维周期边界邻居表 |
| `dphi_dt` | ODE右端项 |
| `euler_step` / `rk4_step` | 时间积分器（可选） |
| `order_parameter` | 同步序参量 r = \|⟨e^{iφ}⟩\| |
| `effective_frequencies` | 有效频率 ω_eff |

**输出文件**:
- `kuramoto_timeseries.csv` — 同步序参量时间序列
- `kuramoto_omega_eff.csv` — 各振子有效频率
- `kuramoto_coupling_scan.csv` — 耦合强度扫描

---

### 3. Vicsek模型 (`vicsek.rs`)

**物理背景**: 活性物质原型模型，研究集体运动相变

**更新规则**:
```
θᵢ(t+1) = circmean_{|rⱼ-rᵢ|<R} θⱼ(t) + ηᵢ
rᵢ(t+1) = rᵢ(t) + v₀ ê(θᵢ)
```

**关键函数**:
| 函数 | 功能 |
|------|------|
| `vicsek_step` | 同步更新（最小镜像周期边界） |
| `order_parameter` | 归一化平均速度 φᵥ |

**输出文件**:
- `vicsek_timeseries.csv` — 序参量时间序列
- `vicsek_noise_scan.csv` — 噪声强度扫描
- `vicsek_snapshot.csv` — 粒子快照

---

### 4. 自旋转转子模型 (`rotor_sde.rs`)

**物理背景**: 非平衡XY变体，研究缺陷超扩散

**SDE方程**:
```
dθᵢ = [Ωᵢ + J Σ sin(θⱼ-θᵢ)]dt + √(2D dt) ξᵢ
```

**关键函数**:
| 函数 | 功能 |
|------|------|
| `drift` | 确定性漂移项 |
| `em_step` | Euler-Maruyama步进 |
| `count_defects` | 拓扑缺陷计数（缠绕数方法） |

**输出文件**:
- `rotor_timeseries.csv` — 磁化强度和缺陷数时间序列
- `rotor_correlation.csv` — 自旋关联函数
- `rotor_sigma_scan.csv` — 无序强度扫描

---

## 设计原则

1. **Python一一对应**: 函数名、参数顺序、算法步骤完全镜像Python版本
2. **无unsafe代码**: 所有数组访问边界检查，周期边界使用 `.rem_euclid()`
3. **最小依赖**: 仅 `rand` + `rand_distr`，无需线性代数库
4. **CSV输出**: 便于Python/gnuplot/Julia/R可视化
5. **SmallRng**: 快速可种子化PRNG（xoshiro128++）

---

## 快速开始

```bash
# 一键构建所有项目
chmod +x setup_rust_projects.sh
./setup_rust_projects.sh

# 单独构建运行
mkdir -p xy_mc/src
cp xy_mc.rs xy_mc/src/main.rs
cp xy_mc_Cargo.toml xy_mc/Cargo.toml
cd xy_mc && cargo run --release
```

---

## 依赖配置

```toml
[dependencies]
rand       = { version = "0.8", features = ["small_rng"] }
rand_distr = "0.4"
```

---

## 理论背景：Mermin-Wagner定理及其打破

### MW定理核心结论
- d ≤ 2时，连续对称性（O(N≥2)）的短程相互作用系统**不存在真正长程序**
- 物理机制：长波自旋波涨落发散

### 打破MW机制的途径（本项目覆盖）

| 机制 | 模型 | 关键特征 |
|------|------|----------|
| 剪切流驱动 | — | k^{-2/3}涨落抑制 |
| 非互易相互作用 | — | Jᵢⱼ ≠ Jⱼᵢ |
| 自驱动/活性 | Vicsek | 等效长程作用 |
| 自旋转注入 | Rotor SDE | 能量持续注入 |

---

## 代码风格

- **结构体封装**: 每个模型使用结构体存储状态（`Lattice`, `OscLattice`, `Particles`, `RotorLattice`）
- **函数式设计**: 核心物理计算为纯函数
- **显式周期边界**: 使用 `wrap()` / `rem_euclid()` 处理
- **CSV输出分离**: 独立的 `write_*` 函数处理文件输出

---

## 参考文献

1. Mermin & Wagner (1966) — MW定理
2. Vicsek et al. (1995) — Vicsek模型
3. Sakaguchi, Shinomoto & Kuramoto (1987) — 晶格Kuramoto
4. Rouzaire & Levis (2021) — 自旋转转子

---

*文档生成日期: 2026年3月27日*