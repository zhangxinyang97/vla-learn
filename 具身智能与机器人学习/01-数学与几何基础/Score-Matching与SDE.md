---
title: Score-Matching与SDE
tags:
  - score-matching
  - sde
  - diffusion
  - math-foundation
aliases:
  - 得分匹配
  - SDE
  - 朗之万动力学
created: 2026-08-12
---

## 1. Score Function（得分函数）

### 1.1 定义

Score function 定义为对数概率密度对输入的梯度：

$$
\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})
$$

参数说明：
- $\mathbf{x} \in \mathbb{R}^d$：数据点，$d$ 为数据维度（如图像的像素数）
- $p(\mathbf{x})$：数据分布的概率密度函数
- $\nabla_{\mathbf{x}}$：对 $\mathbf{x}$ 的梯度算子，结果为 $d$ 维向量
- $\mathbf{s}(\mathbf{x})$：得分（score），指向概率密度增长最快的方向

### 1.2 直观理解

> **Score 指向更高概率密度的区域。**

把 $p(\mathbf{x})$ 想象成一座山。Score 就是 "上山" 的方向向量：
- 在概率密度高的地方（山顶），score 较小（梯度趋近零）
- 在概率密度低的地方（山脚），score 指向山顶方向，幅度较大
- Score 告诉我们在数据空间中 "往哪走" 才能遇到更像训练数据样本的点

这一直觉直接催生了**朗之万动力学采样**（见第 4 节）。

### 1.3 为什么 Score 避免了归一化常数

许多概率模型的形式为：

$$
p_\theta(\mathbf{x}) = \frac{e^{-E_\theta(\mathbf{x})}}{Z_\theta}
$$

其中 $E_\theta(\mathbf{x})$ 为能量函数（unnormalized log-probability），$Z_\theta = \int e^{-E_\theta(\mathbf{x})} d\mathbf{x}$ 为归一化常数（partition function）。

**关键洞察：取对数后求梯度，$Z_\theta$ 消失！**

$$
\begin{aligned}
\log p_\theta(\mathbf{x}) &= -E_\theta(\mathbf{x}) - \log Z_\theta \\[4pt]
\nabla_{\mathbf{x}} \log p_\theta(\mathbf{x}) &= -\nabla_{\mathbf{x}} E_\theta(\mathbf{x}) - \underbrace{\nabla_{\mathbf{x}} \log Z_\theta}_{=\;0}
\end{aligned}
$$

因为 $Z_\theta$ 不依赖于 $\mathbf{x}$，其梯度为零。因此：

$$
\boxed{\mathbf{s}_\theta(\mathbf{x}) = -\nabla_{\mathbf{x}} E_\theta(\mathbf{x})}
$$

**这对扩散模型为什么重要：**
- 我们不需要计算难以处理的归一化常数 $Z_\theta$
- 只需学习 score function $\mathbf{s}_\theta(\mathbf{x})$ 即可描述整个分布
- 这使得 score-based 模型避开了传统能量模型的瓶颈

---

## 2. Score Matching（得分匹配）

### 2.1 原始 Score Matching（Hyvärinen, 2005）

目标：训练一个 score 网络 $\mathbf{s}_\theta(\mathbf{x})$ 去逼近真实数据的 score $\nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$。

最直接的思路是最小化 Fisher divergence：

$$
\mathcal{L}_{\text{SM}} = \frac{1}{2} \mathbb{E}_{p_{\text{data}}(\mathbf{x})} \left[ \|\mathbf{s}_\theta(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})\|^2 \right]
$$

但 $\nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$ 是不可知的（我们不知道真实数据分布的解析形式）。

**Hyvärinen 的关键贡献：** 通过分部积分（integration by parts），上述损失等价于以下可计算的式子：

$$
\mathcal{L}_{\text{SM}} = \mathbb{E}_{p_{\text{data}}(\mathbf{x})} \left[ \operatorname{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x})) + \frac{1}{2} \|\mathbf{s}_\theta(\mathbf{x})\|^2 \right]
$$

其中 $\operatorname{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}))$ 是 score 网络的 Jacobian 的迹。

**问题：** 对高维数据（如图像），计算 $\operatorname{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}))$ 需要 $d$ 次反向传播，计算量太大。

### 2.2 Denoising Score Matching（DSM，去噪得分匹配）

Vincent (2011) 提出了 Denoising Score Matching，解决了高维计算问题。

**核心思想：** 对干净数据 $\mathbf{x}$ 添加噪声 $\sigma \cdot \boldsymbol{\epsilon}$（$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$），得到扰动数据 $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\epsilon}$。扰动后的条件分布是已知的：

$$
p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 \mathbf{I})
$$

其 score 有闭式解：

$$
\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2} = -\frac{\boldsymbol{\epsilon}}{\sigma}
$$

**DSM 损失：**

$$
\boxed{\mathcal{L}_{\text{DSM}} = \frac{1}{2} \mathbb{E}_{p_{\text{data}}(\mathbf{x})} \mathbb{E}_{\tilde{\mathbf{x}} \sim \mathcal{N}(\mathbf{x}, \sigma^2 \mathbf{I})} \left[ \left\| \mathbf{s}_\theta(\tilde{\mathbf{x}}) - \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2} \right\|^2 \right]}
$$

化简后即为噪声预测形式：

$$
\mathcal{L}_{\text{DSM}} = \frac{1}{2\sigma^2} \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}} \left[ \left\| \mathbf{s}_\theta(\mathbf{x} + \sigma\boldsymbol{\epsilon}) + \frac{\boldsymbol{\epsilon}}{\sigma} \right\|^2 \right]
$$

参数说明：
- $\sigma$：噪声标准差，控制扰动程度
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$：标准高斯噪声
- $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\epsilon}$：加噪后的数据
- $\mathbf{s}_\theta(\cdot)$：需要学习的 score 网络

> **DSM 与 DDPM 的噪声预测的联系：**  
> 若我们参数化 $\mathbf{s}_\theta(\tilde{\mathbf{x}}) = -\frac{\boldsymbol{\epsilon}_\theta(\tilde{\mathbf{x}}, \sigma)}{\sigma}$，则 DSM 损失等价于预测噪声 $\boldsymbol{\epsilon}$。这正是 DDPM 的核心损失函数！详见 [[扩散模型基础]]。

### 2.3 Sliced Score Matching（切片得分匹配）

Song et al. (2019) 提出的替代方案，通过随机投影避免计算完整 Jacobian 的迹：

$$
\mathcal{L}_{\text{SSM}} = \mathbb{E}_{p_{\text{data}}} \mathbb{E}_{\mathbf{v} \sim \mathcal{N}(0, \mathbf{I})} \left[ \mathbf{v}^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}) \mathbf{v} + \frac{1}{2} \|\mathbf{s}_\theta(\mathbf{x})\|^2 \right]
$$

其中 $\mathbf{v}$ 为随机投影向量，$\mathbf{v}^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}) \mathbf{v}$ 可通过一次向量-Jacobian 乘积高效计算。

### 2.4 多噪声尺度下的 Score Matching

NCSN（Noise Conditional Score Network）的核心创新：**在多个噪声尺度上同时学习 score**。

不同 $\sigma$ 下的扰动分布：

$$
p_{\sigma_i}(\tilde{\mathbf{x}}) = \int p_{\text{data}}(\mathbf{x}) \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma_i^2 \mathbf{I}) d\mathbf{x}
$$

加权 DSM 损失：

$$
\mathcal{L}_{\text{NCSN}} = \frac{1}{L} \sum_{i=1}^{L} \lambda(\sigma_i) \cdot \mathcal{L}_{\text{DSM}}(\sigma_i)
$$

其中 $\lambda(\sigma_i) = \sigma_i^2$ 是经验权重（使不同噪声尺度的损失 magnitude 对齐），$L$ 为噪声尺度个数。

参数说明：
- $L$：噪声尺度总数（NCSN 中通常取 10）
- $\sigma_i$：第 $i$ 个噪声尺度，满足 $\sigma_1 < \sigma_2 < \dots < \sigma_L$
- $\lambda(\sigma_i)$：权重函数，通常取 $\sigma_i^2$

---

## 3. NCSN 架构细节

NCSN 使用基于 U-Net 的 score 网络，通过实例归一化（Instance Normalization）处理不同噪声尺度。

### 3.1 噪声条件 Score 网络

$$
\mathbf{s}_\theta(\mathbf{x}, \sigma) = \frac{\mathbf{x} - \mathbf{D}_\theta(\mathbf{x}, \sigma)}{\sigma^2}
$$

参数化技巧：学去噪器 $\mathbf{D}_\theta(\mathbf{x}, \sigma)$ 而非直接学 score，数值更稳定。

### 3.2 噪声尺度的几何级数设计

NCSNv2 改进了噪声尺度的选择策略，使其适应数据分布：

$$
\sigma_i = \sigma_1 \cdot \left( \frac{\sigma_L}{\sigma_1} \right)^{\frac{i-1}{L-1}}, \quad i = 1, \dots, L
$$

其中 $\sigma_1$ 足够小（不破坏数据结构），$\sigma_L$ 足够大（使扰动分布接近高斯）。

---

## 4. Langevin Dynamics（朗之万动力学）

### 4.1 基本原理

朗之万动力学是一种 MCMC 采样方法，仅需 score function 即可从分布中采样：

$$
\boxed{\mathbf{x}_{t+1} = \mathbf{x}_t + \eta \cdot \nabla_{\mathbf{x}} \log p(\mathbf{x}_t) + \sqrt{2\eta} \cdot \mathbf{z}_t, \quad \mathbf{z}_t \sim \mathcal{N}(0, \mathbf{I})}
$$

参数说明：
- $\mathbf{x}_t$：第 $t$ 步的样本（$t$ 为采样步数，与扩散模型的时间步不同）
- $\eta$：步长（step size），控制每次更新的幅度
- $\nabla_{\mathbf{x}} \log p(\mathbf{x}_t)$：当前 $\mathbf{x}_t$ 处的 score
- $\mathbf{z}_t$：随机噪声项，防止陷入局部模式
- $\sqrt{2\eta}$：噪声系数，保证理论收敛性（满足涨落耗散定理）

**两项的物理意义：**
| 项 | 名称 | 作用 |
|-------|------|------|
| $\eta \cdot \nabla_{\mathbf{x}} \log p(\mathbf{x}_t)$ | 漂移项（Drift） | 将样本推向高概率密度区域 |
| $\sqrt{2\eta} \cdot \mathbf{z}_t$ | 扩散项（Diffusion） | 注入随机性，保证遍历整个分布 |

### 4.2 收敛条件

当 $\eta \to 0$ 且 $T \to \infty$ 时，$\mathbf{x}_T$ 的分布收敛到 $p(\mathbf{x})$。实际中取 $T$ 足够大（如 1000 步），$\eta$ 足够小。

### 4.3 Annealed Langevin Dynamics（退火朗之万动力学）

NCSN 使用多噪声尺度的退火策略：

**算法流程：**
1. 初始化 $\tilde{\mathbf{x}}_0 \sim \mathcal{N}(0, \mathbf{I})$（白噪声）
2. 对每个噪声尺度 $\sigma_i$（从大到小）：
   - 设置步长 $\eta_i = \epsilon \cdot \sigma_i^2 / \sigma_L^2$
   - 执行 $T$ 步朗之万动力学：$\tilde{\mathbf{x}}_{t+1} = \tilde{\mathbf{x}}_t + \eta_i \cdot \mathbf{s}_\theta(\tilde{\mathbf{x}}_t, \sigma_i) + \sqrt{2\eta_i} \cdot \mathbf{z}_t$
3. 最终 $\tilde{\mathbf{x}}$ 即为采样结果

**退火的作用：**
- 大 $\sigma$：探索全局结构，快速移动到数据流形附近
- 小 $\sigma$：精修局部细节，生成高保真样本
- 逐步降噪避免陷入局部最优

### 4.4 PyTorch 伪代码

```python
import torch

def annealed_langevin_sampling(score_model, sigmas, T=100, epsilon=2e-5):
    """
    Annealed Langevin dynamics for NCSN.
    
    Args:
        score_model: noise-conditional score network s_θ(x, σ)
        sigmas: list of noise levels [σ₁, σ₂, ..., σ_L] (descending)
        T: number of Langevin steps per noise level
        epsilon: base step size factor
    
    Returns:
        sampled image (shape: [B, C, H, W])
    """
    device = next(score_model.parameters()).device
    B, C, H, W = 1, 3, 32, 32  # example shape
    
    # Step 1: Initialize from pure noise
    x = torch.randn(B, C, H, W, device=device)
    
    sigma_L = sigmas[-1]
    
    # Step 2: Descend through noise levels
    for sigma in reversed(sigmas):  # from large σ to small σ
        # Adaptive step size proportional to σ²
        alpha = epsilon * (sigma / sigma_L) ** 2
        
        # Step 3: Run T Langevin steps
        for t in range(T):
            z = torch.randn_like(x)
            # Score evaluation
            score = score_model(x, sigma)
            # Langevin update: drift + diffusion
            x = x + alpha * score + torch.sqrt(2 * alpha) * z
    
    return x
```

---

## 5. 随机微分方程（SDE）统一框架

Song et al. (2021) 提出用 SDE 统一 NCSN 和 DDPM，揭示了 score-based 模型的连续时间极限。

### 5.1 前向 SDE（Forward / Perturbation Process）

前向过程将数据逐步扰动为噪声：

$$
\boxed{\mathrm{d}\mathbf{x} = \mathbf{f}(\mathbf{x}, t) \mathrm{d}t + g(t) \mathrm{d}\mathbf{w}}
$$

参数说明：
- $\mathbf{x}$：数据（在连续时间 $t \in [0, T]$ 中演化）
- $\mathbf{f}(\mathbf{x}, t)$：漂移系数（drift），控制确定性演化方向
- $g(t)$：扩散系数（diffusion），控制随机噪声注入强度
- $\mathbf{w}$：标准维纳过程（Wiener process / Brownian motion）
- $\mathrm{d}\mathbf{w} \sim \mathcal{N}(0, \mathrm{d}t \cdot \mathbf{I})$

#### VP-SDE（Variance Preserving SDE）——对应 DDPM

$$
\mathrm{d}\mathbf{x} = -\frac{1}{2}\beta(t) \mathbf{x} \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}\mathbf{w}
$$

参数说明：
- $\beta(t)$：连续时间的噪声调度函数，$\beta(t) > 0$，通常取线性递增
- $-\frac{1}{2}\beta(t) \mathbf{x}$：均值回归项，将 $\mathbf{x}$ 拉向原点
- $\sqrt{\beta(t)}$：噪声注入强度
- "Variance Preserving"：若初始 $\operatorname{Var}(\mathbf{x}_0) = 1$，则全过程 $\operatorname{Var}(\mathbf{x}_t) = 1$

> **VP-SDE 的离散化等价于 DDPM 的前向加噪过程。**  
> 令 $\beta_i = \beta(t_i) \Delta t$，即可恢复 DDPM 的递推公式 $\mathbf{x}_t = \sqrt{1-\beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}$。详见 [[扩散模型基础]]。

#### VE-SDE（Variance Exploding SDE）——对应 NCSN

$$
\mathrm{d}\mathbf{x} = \sqrt{\frac{\mathrm{d}[\sigma^2(t)]}{\mathrm{d}t}} \mathrm{d}\mathbf{w}
$$

参数说明：
- $\sigma^2(t)$：方差函数，随 $t$ 单调递增
- 漂移项为零（$\mathbf{f} = 0$），纯扩散过程
- "Variance Exploding"：$\operatorname{Var}(\mathbf{x}_t)$ 随 $t$ 不断增大

#### sub-VP SDE

$$
\mathrm{d}\mathbf{x} = -\frac{1}{2}\beta(t) \mathbf{x} \mathrm{d}t + \sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s)\mathrm{d}s})} \mathrm{d}\mathbf{w}
$$

该 SDE 的方差始终小于 VP-SDE，介于 VP 和 VE 之间。

### 5.2 反向 SDE（Reverse SDE）——Anderson 定理

Anderson (1982) 证明了逆向时间 SDE 的形式：

$$
\boxed{\mathrm{d}\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] \mathrm{d}t + g(t) \mathrm{d}\bar{\mathbf{w}}}
$$

参数说明：
- $\bar{\mathbf{w}}$：逆向时间的标准维纳过程
- $p_t(\mathbf{x})$：前向过程在时间 $t$ 的边缘分布
- $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$：时间 $t$ 的（未知）score function
- $g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$：score 修正项，将扩散引导回数据分布方向

**与正向 SDE 对比：**

| 分量 | 正向 SDE | 反向 SDE |
|------|----------|----------|
| 漂移项 | $\mathbf{f}(\mathbf{x}, t)$ | $\mathbf{f}(\mathbf{x}, t) - g^2(t) \nabla_{\mathbf{x}} \log p_t$ |
| 扩散项 | $g(t) \mathrm{d}\mathbf{w}$ | $g(t) \mathrm{d}\bar{\mathbf{w}}$ |
| 演化方向 | 数据 → 噪声 | 噪声 → 数据 |

**Anderson 定理的含义：** 如果我们能估计每一时刻的 score $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$，就可以通过求解反向 SDE 从噪声生成数据。

这就是扩散模型的数学核心：**用神经网络 $\mathbf{s}_\theta(\mathbf{x}, t)$ 拟合 $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$，然后解反向 SDE 采样。**

### 5.3 SDE 训练的连续时间损失

将 DSM 推广到连续时间：

$$
\mathcal{L}_{\text{SDE}} = \mathbb{E}_{t \sim \mathcal{U}(0,T)} \mathbb{E}_{\mathbf{x}(0) \sim p_{\text{data}}} \mathbb{E}_{\mathbf{x}(t) \sim p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))} \left[ \lambda(t) \left\| \mathbf{s}_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) \right\|^2 \right]
$$

参数说明：
- $t \sim \mathcal{U}(0,T)$：均匀采样的时间步
- $\mathbf{x}(0)$：干净数据
- $p_{0t}$：前向 SDE 从时间 0 到 $t$ 的转移概率（高斯分布，均值与方差由 SDE 决定）
- $\lambda(t)$：时间相关的权重函数

对于 VP-SDE，条件 score 的闭式解为：

$$
\nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - e^{-\int_0^t \beta(s) \mathrm{d}s}}}
$$

取 $\lambda(t) = 1 - e^{-\int_0^t \beta(s) \mathrm{d}s}$ 可使不同 $t$ 的损失量级对齐。

---

## 6. Probability Flow ODE（概率流常微分方程）

### 6.1 从 SDE 到 ODE

**关键洞察：** 每一个 SDE 都有一个等价的 ODE，其边缘分布演化完全相同！

对于同一个前向 SDE，存在一个确定性过程（Probability Flow ODE）：

$$
\boxed{\mathrm{d}\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - \frac{1}{2} g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] \mathrm{d}t}
$$

参数说明：
- 相比反向 SDE，扩散项 $g(t) \mathrm{d}\bar{\mathbf{w}}$ 被移除
- 漂移项中 $g^2(t)$ 的系数从 $1$ 变为 $\frac{1}{2}$（保证边缘分布等价）
- 这是一个**确定性**过程 —— 给定初始噪声，采样结果唯一确定

### 6.2 SDE 与 ODE 对比

| 性质 | 反向 SDE | Probability Flow ODE |
|------|----------|---------------------|
| 随机性 | 随机 | 确定性 |
| 采样速度 | 需多步（~1000） | 可大幅减少步数 |
| 隐空间插值 | 不保证平滑 | 平滑可逆，支持语义插值 |
| 似然评估 | 困难 | 支持（通过瞬时变变量公式） |
| 典型方法 | SDE Solver（Euler-Maruyama） | ODE Solver（DDIM, DPM-Solver） |

### 6.3 DDIM 就是 PF-ODE 的离散化

DDIM（Denoising Diffusion Implicit Models）的采样公式等价于 VP-SDE 的 Probability Flow ODE 的离散化。这是 DDIM 可以少步采样（如 50 步）的理论基础。

### 6.4 似然计算

Probability Flow ODE 让精确似然计算成为可能。根据瞬时变变量公式（instantaneous change-of-variables）：

$$
\log p_0(\mathbf{x}(0)) = \log p_T(\mathbf{x}(T)) + \int_0^T \operatorname{tr}\left( \nabla_{\mathbf{x}} \mathbf{v}(\mathbf{x}(t), t) \right) \mathrm{d}t
$$

其中 $\mathbf{v}(\mathbf{x}, t) = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ 为 ODE 的速度场。

---

## 7. SDE 框架——模型统一视图

| 模型 | 离散 / 连续 | SDE 类型 | Score 估计方式 | 采样方式 | 特点 |
|------|------------|---------|---------------|---------|------|
| NCSN (2019) | 离散（多噪声尺度） | VE-SDE（极限） | DSM 多尺度 | Annealed Langevin | 开创 score-based 生成 |
| NCSNv2 (2020) | 离散 | VE-SDE | DSM + 自适应噪声尺度 | Annealed Langevin | 改进噪声尺度选择 |
| DDPM (2020) | 离散（马尔可夫链） | VP-SDE（极限） | 噪声预测 | 祖先采样 | 奠基扩散模型 |
| DDIM (2021) | 离散 | PF-ODE（VP） | 噪声预测 | 确定性跳步采样 | 加速采样 |
| SDE Unified (2021) | 连续 | VP / VE / sub-VP | 连续时间 DSM | SDE Solver / PF-ODE | 统一框架 |
| Flow Matching (2023) | 连续 | ODE（无扩散项） | 向量场回归 | ODE Solver | 更通用框架 |
| DPM-Solver (2022) | 连续 | PF-ODE | 噪声预测 | 高阶 ODE Solver | 极致加速（10-20 步） |

---

## 8. 从 Score Matching 到扩散模型：逻辑链

```
Score Function 定义
    ↓ (避开归一化常数 Z_θ)
Score Matching (Hyvärinen, 2005)
    ↓ (引入噪声解决高维计算)
Denoising Score Matching (Vincent, 2011)
    ↓ (多噪声尺度)
NCSN / Annealed Langevin Sampling (Song & Ermon, 2019)
    ↓ (连续时间极限)
SDE 统一框架 (Song et al., 2021)
    ↓ (PF-ODE 分支)
DDIM / DPM-Solver 加速采样
    ↓ (ODE 泛化)
Flow Matching (Lipman et al., 2023)
```

每一步的推动力都是解决上一步的计算或理论瓶颈。

---

## 9. 关键公式速查

| 公式 | 名称 | 用途 |
|------|------|------|
| $\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$ | Score function | 定义 |
| $\mathcal{L}_{\text{DSM}} = \mathbb{E}[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - (\mathbf{x} - \tilde{\mathbf{x}})/\sigma^2\|^2]$ | DSM 损失 | 训练 |
| $\mathbf{x}_{t+1} = \mathbf{x}_t + \eta \mathbf{s}(\mathbf{x}_t) + \sqrt{2\eta}\mathbf{z}_t$ | 朗之万动力学 | NCSN 采样 |
| $\mathrm{d}\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}\mathrm{d}t + \sqrt{\beta(t)}\mathrm{d}\mathbf{w}$ | VP-SDE | DDPM 前向 |
| $\mathrm{d}\mathbf{x} = [\mathbf{f} - g^2\nabla_{\mathbf{x}}\log p_t]\mathrm{d}t + g\mathrm{d}\bar{\mathbf{w}}$ | 反向 SDE | 采样 |
| $\mathrm{d}\mathbf{x} = [\mathbf{f} - \frac{1}{2}g^2\nabla_{\mathbf{x}}\log p_t]\mathrm{d}t$ | PF-ODE | 确定性采样 |

---

## 10. 总结

Score Matching 和 SDE 为扩散模型提供了完整的数学基础：

1. **Score function** 避免了归一化常数，使高维分布建模成为可能
2. **Denoising Score Matching** 将不可计算的分部积分转化为简单的噪声预测
3. **朗之万动力学** 提供了一种仅用 score 即可采样的 MCMC 方法
4. **SDE 统一框架** 将 NCSN 和 DDPM 统一在连续时间视角下
5. **Probability Flow ODE** 提供了确定性采样路径，支持加速和似然计算

> **理解扩散模型的关键：一切都可以归结为"学习 score function，然后解反向 SDE/ODE"。**

---

## 相关笔记

- [[扩散模型基础]] — DDPM 的详细推导与实现
- [[Flow-Matching流匹配]] — 更通用的生成建模范式
- [[变分推断与ELBO]] — 扩散模型的变分视角
- [[KL散度推导]] — 损失推导中反复使用的信息论工具

---

## 参考文献

1. Hyvärinen, A. (2005). Estimation of Non-Normalized Statistical Models by Score Matching. *JMLR*.
2. Vincent, P. (2011). A Connection Between Score Matching and Denoising Autoencoders. *Neural Computation*.
3. Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. *NeurIPS*.
4. Song, Y., & Ermon, S. (2020). Improved Techniques for Training Score-Based Generative Models. *NeurIPS*.
5. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.
6. Song, Y., Sohl-Dickstein, J., et al. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. *ICLR*.
7. Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. *ICLR*.
8. Anderson, B. D. O. (1982). Reverse-Time Diffusion Equation Models. *Stochastic Processes and their Applications*.
9. Lipman, Y., Chen, R. T. Q., et al. (2023). Flow Matching for Generative Modeling. *ICLR*.
10. Lu, C., et al. (2022). DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. *NeurIPS*.
