# 候选模型库

## 三、候选模型库

以下所有模型均满足约束（经典、保持连续对称性、短程相互作用、PBC）并排除已知结论。

模型按**"出现LRO的可能性"**和**"数值模拟优先级"**综合评分，分为四个优先级档次：

- 🔴 **P1（最高优先级）**：物理机制清晰、LRO可能性高（≥70%）、数值可行性强
- 🟠 **P2（高优先级）**：有合理物理依据、LRO可能性中高（40–70%）
- 🟡 **P3（中优先级）**：机制探索性强、LRO可能性中等（20–40%）
- ⚪ **P4（探索优先级）**：高度创新但LRO可能性较低或不确定（<20%）

### 第A族：Kuramoto振子格子 + 非平衡驱动

（基础：二维Kuramoto格子无LRO，Daido 1988；本族探索加入新机制后的行为）

**A1** 🔴 **Kuramoto格子 + 均匀剪切流平流（各向同性噪声，相同本征频率 \Omega_i=\Omega_0）**

- 模型：\dot\theta_i = \Omega_0 + J\sum_{j\sim i}\sin(\theta_j-\theta_i) + \dot\gamma y_i(\theta_{i+\hat{x}}-\theta_{i-\hat{x}})/(2a) + \sqrt{2D}\xi_i
- 说明：所有振子本征频率相同（去除淬火无序），在此基础上加剪切流平流。与Nakano 2021最近，但序参量为相位同步而非磁化。
- LRO可能性：高（剪切流抑制 |k_x|^{-2/3} 机制预期适用于相位场）
- 注意：各向同性噪声；剪切流本身不破缺O(2)对称性（统计上仍各向同性）。

**A2** 🔴 **Kuramoto格子 + 均匀剪切流平流（Gaussian频率分布，小色散 \sigma\ll J）**

- 模型：同A1但 \Omega_i\sim\mathcal{N}(0,\sigma^2)，探索频率色散与剪切流的竞争
- 说明：A1的推广，检验LRO是否对小频率色散鲁棒
- LRO可能性：中高（小σ时预期稳健）

**A3** 🟠 **Kuramoto格子 + 非均匀振荡驱动（空间调制周期力）**

- 模型：\dot\theta_i = \Omega_0 + J\sum_{j\sim i}\sin(\theta_j-\theta_i) + A\cos(\mathbf{q}_0\cdot\mathbf{r}_i)\cos(\omega t) + \sqrt{2D}\xi_i，取 \mathbf{q}_0\to 0 极限（长波调制）
- 说明：对应Ikeda-Kuroda 2024机制在相位振子上的推广
- LRO可能性：中高（振荡驱动抑制Goldstone模机制预期适用）

**A4** 🟠 **Kuramoto格子 + 配对活性驱动（等大反向，质心守恒）**

- 模型：\dot\theta_i = \Omega_0 + J\sum_{j\sim i}\sin(\theta_j-\theta_i) + \sum_{j\sim i}F_{ij}^{\text{act}}(t) + \sqrt{2D}\xi_i，其中 F_{ij}^{\text{act}}=-F_{ji}^{\text{act}}，F_{ij}^{\text{act}} 为短程有色噪声
- 说明：对应Keta-Henkes机制在相位场上的推广；\sum_i F_i=0 质心守恒，噪声谱 \propto q^2
- LRO可能性：中高（超均匀机制预期适用于相位场）

**A5** 🟠 **Kuramoto格子 + 两种不同噪声温度（x方向 T_x，y方向 T_y，T_x\neq T_y）**

- 模型：各向异性Langevin噪声，但保持O(2)对称性（仅噪声各向异性，耦合项各向同性）
- 说明：对应Bassler-Racz机制在Kuramoto格子上的推广；**注意：各向异性噪声是否违反约束2？** 此处噪声各向异性仅改变动力学，不改变耦合的旋转对称性，故符合约束。
- LRO可能性：中（Bassler-Racz中O(2)有LRO，但Kuramoto加入频率色散后是否保持需检验）

### 第B族：XY模型 / O(2)场 + 新型非平衡驱动

（基础：平衡态XY无LRO；非互易视野锥XY已有LRO不再探索）

**B1** 🔴 **XY模型 + 均匀拉伸流（extensional flow，新约束下的实现）**

- 模型：格点XY模型加拉伸流平流项 v_x=\dot\epsilon x,v_y=-\dot\epsilon y（不可压缩）；采用PBC下的Lees-Edwards型边界
- 说明：Minami-Nakano 2022证明拉伸流（O(N)场论）有LRO；此处为格点XY模型的数值验证，尚无人做
- LRO可能性：高（场论预测LRO，待格点验证）

**B3** 🔴 **XY模型 + 配对活性驱动（格点版质心守恒噪声）**

- 模型：格点XY自旋 \theta_i，加入成对噪声力 \eta_{ij}=-\eta_{ji}，满足 \sum_i\eta_i=0
- 说明：Keta-Henkes在弹性体上的版本；在纯旋转自由度（相位）上的推广，尚未有人研究
- LRO可能性：高（质心守恒机制预期适用于旋转序参量）

**B4** 🟠 **XY模型 + 时间延迟反馈（time-delayed coupling）**

- 模型：\dot\theta_i = J\sum_{j\sim i}\sin[\theta_j(t-\tau)-\theta_i(t)] + \sqrt{2D}\xi_i，\tau>0 为延迟时间
- 说明：时间延迟打破时间反演对称性，等效产生非平衡；延迟耦合在神经网络和光学系统中常见，其对XY模型序的影响尚未研究
- LRO可能性：中高（延迟等效为有色噪声/记忆，可能压缩IR涨落）

**B5** 🟠 **XY模型 + 有色噪声（指数关联噪声，相关时间 \tau_c）**

- 模型：\dot\theta_i = J\sum_{j\sim i}\sin(\theta_j-\theta_i) + \eta_i(t)，\langle\eta_i(t)\eta_i(t')\rangle = (D/\tau_c)e^{-|t-t'|/\tau_c}（各向同性，不破缺空间对称性）
- 说明：有色噪声等效于与记忆核的自旋耦合，可能改变Goldstone模的有效质量；长相关时间极限类似于Ikeda-Kuroda振荡驱动
- LRO可能性：中（\tau_c\to\infty 时等效为受迫振荡，与已知机制相连）

**B6** 🟠 **XY模型 + 周期振荡外浴（驱动频率 \omega_d，噪声强度时间调制）**

- 模型：\dot\theta_i = J\sum_{j\sim i}\sin(\theta_j-\theta_i) + \sqrt{2D(t)}\xi_i，D(t)=D_0[1+A\cos(\omega_d t)]
- 说明：噪声强度时间调制，等效于周期驱动的热浴；与Ikeda-Kuroda振荡驱动（空间非均匀）互补（此处为时间调制）
- LRO可能性：中（时间调制噪声能否起到类似空间非均匀驱动的IR抑制效果尚不清楚）

**B7** 🟠 **XY模型 + 守恒动力学（Model B型，\partial_t\theta = \nabla^2\delta F/\delta\theta + \nabla\cdot\boldsymbol\zeta，守恒噪声 \nabla\cdot\boldsymbol\zeta）**

- 模型：序参量守恒的XY动力学（类Kawasaki自旋交换），即Bassler-Racz中 T_x=T_y 情况的纯守恒极限
- 说明：Bassler-Racz的守恒动力学在各向同性温度时是否仍有LRO？原始文献用 T_x\neq T_y；单温守恒版本需要检验
- LRO可能性：中（守恒动力学改变有效噪声谱，但单温下是否突破MW存疑）

**B8** 🟡 **XY模型 + 随机更新节律（Poisson过程时钟，非同步动力学）**

- 模型：格点XY模型，每个自旋以独立Poisson速率 \lambda_i 被更新（而非同步并行更新）
- 说明：非同步更新打破时间反演（不同于Glauber动力学），可能等效产生某种非平衡效应
- LRO可能性：低中（仅更新时序的改变，可能不足以改变静态相结构）

**B9** 🟡 **XY模型 + 驱动耗散（driven-dissipative，注入/耗散对）**

- 模型：\dot\theta_i = J\sum_{j\sim i}\sin(\theta_j-\theta_i) - \gamma(\theta_i-\bar\theta) + \sqrt{2D}\xi_i，其中 \bar\theta 为瞬时全局均值（非守恒但全局耦合到均值）
- 说明：注入-耗散形式，类似开放量子系统的经典极限，保持O(2)（因 \bar\theta 随系统旋转）
- LRO可能性：中（全局耦合到均值类似于mean-field，可能稳定LRO，但须确认约束合规性）

**B10** 🟡 **XY模型 + 阶梯式噪声谱（低频噪声被截断，D(k)=0 for |k|<k_c）**

- 模型：在Fourier空间对噪声实施低频截断，仅高频噪声存在
- 说明：直接模拟Maire-Plati "质心浴温度→0" 的核心机制，但在相位模型而非位移模型上
- LRO可能性：中高（直接移除长波噪声应能稳定LRO，但格点实现有技术细节）

**B11** 🟡 **XY模型 + 相互作用的活性翻转（active spin-flip：自旋有自驱动角速度，但模长固定）**

- 模型：\dot\theta_i = \omega_i^{\text{act}} + J\sum_{j\sim i}\sin(\theta_j-\theta_i) + \sqrt{2D}\xi_i，其中活性角速度 \omega_i^{\text{act}} 由局部邻居场决定（如 \omega_i^{\text{act}}=\epsilon\sum_{j\sim i}\cos(\theta_j-\theta_i)），而非淬火随机
- 说明：活性频率依赖于局部构型，非淬火无序；等效于XY模型的特定非线性扰动
- LRO可能性：中（自洽活性频率可能产生有效非互易效果）

### 第C族：O(N≥3)矢量场 + 各类非平衡机制

（基础：平衡态O(3) Heisenberg无LRO；非平衡O(N)研究极为稀少）

**C1** 🔴 **O(3) Heisenberg模型 + 均匀剪切流（格点版）**

- 模型：格点O(3)自旋 \hat{\mathbf{n}}_i\in S^2，加剪切流平流项 \dot\gamma y_i(\hat{\mathbf{n}}_{i+\hat{x}}-\hat{\mathbf{n}}_{i-\hat{x}})/(2a)，保持 |\hat{\mathbf{n}}_i|=1
- 说明：Nakano 2021证明剪切流稳定O(2) LRO；O(3)有2个Goldstone模（O(3)\to O(2)），每个模都经历 |k_x|^{-2/3} 压缩，预期同样产生LRO
- LRO可能性：高（剪切流机制应适用于任意O(N)，已有大N场论支持）

**C2** 🔴 **O(3) Heisenberg模型 + Vicsek型自驱动**

- 模型：格点O(3)自旋附着于以 v_0\hat{\mathbf{n}}_i 速度运动的粒子，局部平均对齐（Vicsek规则推广到O(3)）
- 说明：Toner-Tu场论对O(N)均有效；O(3)自驱动系统是否同样产生极性LRO？
- LRO可能性：高（场论预测，但格点O(3) Vicsek模型未见数值研究）

**C3** 🟠 **O(3) Heisenberg + 配对活性力**

- 模型：格点O(3)自旋 \hat{\mathbf{n}}_i，成对活性力矩 \tau_{ij}=-\tau_{ji}（守恒总角动量）
- 说明：配对活性应力机制推广到O(3)旋转自由度；守恒总扭矩使噪声谱 \propto q^2
- LRO可能性：中高（质心守恒机制的O(3)推广，机制清晰）

**C4** 🟠 **O(4) 矢量模型 + 剪切流**

- 模型：格点O(4)自旋（4分量单位矢量）加剪切流平流
- 说明：连接O(3)和大N极限，测试LRO对N的依赖性；有助于理解O(N)→∞的连续极限
- LRO可能性：中高（大N极限已有理论支持，O(4)介于中间）

**C5** 🟡 **O(3) Heisenberg + 有色噪声（时间关联）**

- 模型：格点O(3)自旋，各向同性有色噪声 \langle\eta_i^\alpha(t)\eta_j^\beta(t')\rangle=D\delta_{ij}\delta^{\alpha\beta}e^{-|t-t'|/\tau_c}
- 说明：B5在O(3)上的推广；有色噪声对多分量序参量的作用与O(2)情况可能有所不同
- LRO可能性：中（参照B5的预期）

**C6** 🟡 **O(3) + 时间延迟耦合**

- 模型：格点O(3)自旋，延迟耦合 \hat{\mathbf{n}}_i(t-\tau)\cdot\hat{\mathbf{n}}_j(t) 型相互作用
- 说明：B4在O(3)上的推广
- LRO可能性：中

### 第E族：非互易相互作用的新变体

（基础：视野锥XY（\Omega_i=0）已知有LRO；以下是未研究的变体）

**E1** 🔴 **距离依赖非互易XY（近邻完全互易，次近邻单向耦合）**

- 模型：最近邻互易耦合 J_{ij}=J；次近邻单向耦合 J_{ij}^{(2)}=J',J_{ji}^{(2)}=0（根据相对位置方向确定单向性）
- 说明：非互易性在更长距离引入，保持局部互易；测试非互易作用距离对LRO的影响
- LRO可能性：高（非互易机制已证明有效，距离推远后应仍有效）

**E2** 🔴 **非互易XY + 本征频率（Loos 2023的Kuramoto推广）**

- 模型：视野锥XY（保持Loos 2023的结构），加入随机本征频率 \Omega_i\sim\mathcal{N}(0,\sigma^2)
- 说明：直接检验非互易LRO对频率色散的鲁棒性；非互易（稳定LRO）vs 频率色散（破坏同步）的竞争
- LRO可能性：高（小σ时LRO应保持；存在临界σ*）

**E3** 🟠 **符号随机非互易XY（耦合随机正/负，但 |J_{ij}|=|J_{ji}|，仅符号不同）**

- 模型：J_{ij}=J,J_{ji}=-J（反对称非互易）或随机化符号
- 说明：完全反对称耦合，不同于视野锥的"有/无"非互易；探索非互易形式对LRO的影响
- LRO可能性：中高（反对称耦合会产生旋转流动力学，但与旋转流的关系需检验）

**E4** 🟠 **强度梯度非互易（非互易强度沿特定格点方向线性变化，但统计各向同性）**

- 模型：J_{ij} 的非互易部分 \delta J_{ij} 在空间上有梯度，但梯度方向随机（保持统计各向同性）
- 说明：空间随机非互易，参照Lorenzana 2025的随机非互易分析
- LRO可能性：中（随机非互易的relevant性取决于临界指数，条件苛刻）

**E5** 🟡 **动态非互易（非互易强度本身有时间动力学，如振荡非互易系数）**

- 模型：J_{ij}^{\text{NR}}(t) = J'\cos(\omega_0 t)，非互易部分周期振荡
- 说明：振荡非互易结合了振荡驱动机制（Ikeda-Kuroda）和非互易机制
- LRO可能性：中

### 第F族：噪声谱工程（在不破缺对称性的前提下设计特殊噪声）

（基础：已知质心守恒噪声 \propto q^2 可产生LRO；以下探索更广泛的噪声谱）

**F1** 🔴 **XY模型 + 保守噪声（守恒荷的扩散噪声，\nabla\cdot\mathbf{J} 型）**

- 模型：\dot\theta_i = J\sum_{j\sim i}\sin(\theta_j-\theta_i) + \sum_\mu(\eta_{i+\hat\mu}-\eta_{i-\hat\mu})，\eta_{i\mu} 为标量白噪声（键噪声，自动守恒）
- 说明：噪声为散度形式，Fourier空间 \tilde\eta_k\propto|k|，谱 \propto|k|^2；这是Keta-Henkes机制在纯相位场上的直接格点实现
- LRO可能性：极高（直接实现 q^2 噪声谱，MW定理被精确抵消）

**F2** 🔴 **XY模型 + 分数阶守恒噪声（\nabla^\alpha\eta，0<\alpha<1）**

- 模型：噪声谱 \tilde\eta_k\propto|k|^\alpha，0<\alpha<1（分数阶Laplacian噪声）
- 说明：连续调节噪声谱指数，探索从 \alpha=0（白噪声，无LRO）到 \alpha=1（F1，LRO）的过渡；存在临界 \alpha^* 使LRO出现
- LRO可能性：极高（直接测试MW定理被突破的充分条件）

**F3** 🟠 **XY模型 + 非白色时间关联噪声（1/f^\beta 噪声，0<\beta<2）**

- 模型：\langle\tilde\eta(\omega)\tilde\eta(-\omega)\rangle\propto|\omega|^{-\beta}，长时间关联噪声
- 说明：时间上的有色噪声改变动力学类（z指数），可能等效地压缩IR发散
- LRO可能性：中高（大 \beta 极限等效于慢噪声，可能抑制Goldstone模涨落）

**F4** 🟠 **XY模型 + 双层噪声（不同时间尺度的两个独立噪声浴叠加，各向同性）**

- 模型：\dot\theta_i = J\sum_{j\sim i}\sin(\theta_j-\theta_i) + \sqrt{2D_1}\xi_i^{(1)} + \sqrt{2D_2}\xi_i^{(2)}，两浴相关时间不同（\tau_1\ll\tau_2）
- 说明：Maire-Plati双浴机制在旋转自由度上的推广；若慢浴不激发长波模则产生LRO
- LRO可能性：中（双浴机制需要选择性加热，仅两浴叠加可能不够）

**F5** 🟡 **XY模型 + 活性翻转噪声（以有限速率主动翻转方向，而非连续白噪声）**

- 模型：每步以概率 p 随机翻转 \theta_i\to\theta_i+\pi（硬翻转），加正常耦合
- 说明：离散活性翻转破坏细致平衡，等效为大振幅噪声的Poisson注入
- LRO可能性：低中（翻转会强烈破坏序，LRO难以稳定）

### 第I族：剪切流/流场机制的变体与扩展

**I1** 🔴 **Kuramoto格子 + 拉伸流（extensional flow，对称轴随机取向，统计各向同性）**

- 模型：\dot\theta_i = J\sum_{j\sim i}\sin(\theta_j-\theta_i) + \dot\epsilon(x_i\partial_x-y_i\partial_y)\theta_i + \sqrt{2D}\xi_i，在PBC中实现时使用统计各向同性的拉伸方向
- 说明：拉伸流（已知对O(2)场有LRO）在Kuramoto相位场上的应用，紧随A1但流场类型不同
- LRO可能性：高（拉伸流机制与剪切流等效，Kuramoto相位场预期同样响应）

**I2** 🟠 **O(3) Heisenberg + 均匀拉伸流**

- 模型：格点O(3)自旋加拉伸流平流
- 说明：C1（剪切流）的拉伸流版本；两者机制相同，但流场形态不同
- LRO可能性：高（与C1等价分析）

**I3** 🟠 **O(2)格点模型 + 振荡剪切（频率 \omega 的时间周期剪切 v=\dot\gamma(t)y\hat{x}，\dot\gamma(t)=\dot\gamma_0\cos(\omega t)）**

- 模型：稳态剪切换成振荡剪切，时间平均流速为零
- 说明：振荡剪切是否同样稳定LRO？相当于B6（时间调制噪声）的剪切流版本；净流为零时剪切流机制是否仍有效
- LRO可能性：中（时间平均流为零可能抵消 k_x^{-2/3} 效应，但瞬时剪切仍产生非平衡）