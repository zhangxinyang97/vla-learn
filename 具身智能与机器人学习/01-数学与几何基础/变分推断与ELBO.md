---
title: 变分推断与ELBO
tags: [variational-inference, elbo, bayesian, vae, math-foundation]
aliases: [VI, 证据下界, Evidence Lower BOund, 变分贝叶斯]
created: 2026-08-12
---

# 变分推断与ELBO

> **一句话**：当后验概率 $p(\mathbf{z}\mid\mathbf{x})$ 无法直接计算时，用一个可优化的分布 $q_\phi(\mathbf{z}\mid\mathbf{x})$ 去近似它，并通过最大化 ELBO 来最小化两个分布的 KL 散度。

---

## 1. 贝叶斯公式与后验推断

### 1.1 贝叶斯公式回顾

$$p(\mathbf{z} \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \mathbf{z})\,p(\mathbf{z})}{p(\mathbf{x})}$$

| 符号 | 名称 | 含义 | 维度 |
|------|------|------|------|
| $\mathbf{x}$ | 观测数据 (Observation) | 我们能看到的变量，如图像像素、文本token | $\mathbb{R}^D$，$D$ = 数据维度（例如 28×28=784 的 MNIST 图像） |
| $\mathbf{z}$ | 隐变量 (Latent Variable) | 不可直接观测的潜在因素，如生成图像背后的"风格"、"内容"、"类别" | $\mathbb{R}^d$，$d \ll D$，通常 $d=2\sim 256$ |
| $p(\mathbf{z}\mid\mathbf{x})$ | **后验分布** (Posterior) | 给定观测数据后，隐变量的条件分布——"看到图像后，推断其潜在表示" | $\mathbf{x},\mathbf{z}$ 上的概率密度 |
| $p(\mathbf{x}\mid\mathbf{z})$ | **似然** (Likelihood) | 给定隐变量后生成观测数据的概率——解码器 | $\mathbf{x},\mathbf{z}$ 上的条件概率密度 |
| $p(\mathbf{z})$ | **先验分布** (Prior) | 在没有观测数据时，对隐变量的先验信念，通常取 $\mathcal{N}(\mathbf{0},\mathbf{I})$ | $\mathbb{R}^d$ 上的标准正态 |
| $p(\mathbf{x})$ | **证据/边际似然** (Evidence/Marginal Likelihood) | 数据的边际分布，对 $\mathbf{z}$ 积分得到 | 标量（归一化常数） |

### 1.2 直观理解

**类比场景**：你走进一个房间，看到桌上放着一幅画（$\mathbf{x}$）。

- **先验 $p(\mathbf{z})$**：在没有看到画之前，你对"画家可能用了什么技法"的先验猜测——大部分画法都差不多，所以假设 $\mathcal{N}(0,1)$。
- **似然 $p(\mathbf{x}\mid\mathbf{z})$**：如果画家用了某种技法（$\mathbf{z}$），画出这幅画（$\mathbf{x}$）的概率有多大。
- **后验 $p(\mathbf{z}\mid\mathbf{x})$**：看到这幅画之后，你应该如何更新对画家技法的推断——这是贝叶斯推断的核心。
- **证据 $p(\mathbf{x})$**：这幅画本身出现的概率——对所有可能的技法求和（积分）。

### 1.3 后验为什么是"不可计算的"？

$$p(\mathbf{x}) = \int_{\mathbf{z}} p(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})\,d\mathbf{z}$$

**问题**：
- 对隐变量 $\mathbf{z}$ 的积分**在高维空间不可行**。当 $d$ 为几十到几百维时，数值积分复杂度呈指数增长（维度灾难）。
- 即使 $\mathbf{z}$ 只有 20 维，遍历每维 100 个采样点也需要 $100^{20}$ 次计算。
- 这就是 **Bayesian Inference 的核心困境**：后验 $p(\mathbf{z}\mid\mathbf{x})$ 有解析解，但需要算 $p(\mathbf{x})$，而 $p(\mathbf{x})$ 不可积。

**两种解决路线**：

| 路线 | 方法 | 代表作 |
|------|------|--------|
| **MCMC 采样** | 从后验中采样而不计算归一化常数 | Metropolis-Hastings, Hamiltonian Monte Carlo |
| **变分推断 (VI)** | 用一个简单分布族近似后验，转化为优化问题 | Mean-Field VI, VAE (Amortized VI) |

> 本文走 **VI → VAE** 路线。

---

## 2. 变分推断：从采样到优化

### 2.1 核心思路

我们引入一个**参数化的近似分布**（变分分布）：

$$q_\phi(\mathbf{z}\mid\mathbf{x})$$

| 符号 | 含义 |
|------|------|
| $q$ | 变分分布（通常是高斯），是我们**可控的**近似后验 |
| $\phi$ | 变分参数（神经网络权重），由编码器学习 |
| $q_\phi(\mathbf{z}\mid\mathbf{x})$ | 给定输入 $\mathbf{x}$，神经网络输出 $\mathbf{z}$ 的分布参数（均值 $\boldsymbol{\mu}_\phi(\mathbf{x})$ 和对数方差 $\log\boldsymbol{\sigma}_\phi^2(\mathbf{x})$） |

我们希望 $q_\phi(\mathbf{z}\mid\mathbf{x})$ 尽可能接近真实后验 $p(\mathbf{z}\mid\mathbf{x})$。用 KL 散度衡量"距离"：

$$D_{\text{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x}) \parallel p(\mathbf{z}\mid\mathbf{x})\big) = \mathbb{E}_{\mathbf{z}\sim q_\phi}\left[\log\frac{q_\phi(\mathbf{z}\mid\mathbf{x})}{p(\mathbf{z}\mid\mathbf{x})}\right]$$

### 2.2 直观理解：为什么叫"变分"？

"变分"（Variational）的含义是：我们不是在积分（无法计算），而是在**一个分布族中搜索最优分布**——即对一个泛函（函数的函数）求极值。

类比：不是算 $\int f(x)dx$，而是找 $\arg\min_q D_{\text{KL}}(q\parallel p)$。这就是变分法思想。

---

## 3. ELBO 推导（逐步展开）

### 3.1 从 KL 散度出发

目标：最小化 $D_{\text{KL}}(q_\phi \parallel p)$。

$$\begin{aligned}
D_{\text{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x}) \parallel p(\mathbf{z}\mid\mathbf{x})\big) 
&= \mathbb{E}_{\mathbf{z}\sim q_\phi}\left[\log q_\phi(\mathbf{z}\mid\mathbf{x}) - \log p(\mathbf{z}\mid\mathbf{x})\right] & \text{(KL 定义)} \\[6pt]
&= \mathbb{E}_{\mathbf{z}\sim q_\phi}\left[\log q_\phi(\mathbf{z}\mid\mathbf{x}) - \log \frac{p(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})}{p(\mathbf{x})}\right] & \text{(代入贝叶斯公式)} \\[6pt]
&= \mathbb{E}_{\mathbf{z}\sim q_\phi}\left[\log q_\phi(\mathbf{z}\mid\mathbf{x}) - \log p(\mathbf{x}\mid\mathbf{z}) - \log p(\mathbf{z}) + \log p(\mathbf{x})\right] & \text{(展开对数)}
\end{aligned}$$

关键一步：$\log p(\mathbf{x})$ 不依赖于 $\mathbf{z}$，所以可以从期望中提取出来：

$$\begin{aligned}
D_{\text{KL}}(q_\phi \parallel p) 
&= \mathbb{E}_{\mathbf{z}\sim q_\phi}\left[\log q_\phi(\mathbf{z}\mid\mathbf{x}) - \log p(\mathbf{x}\mid\mathbf{z}) - \log p(\mathbf{z})\right] + \log p(\mathbf{x}) \\[6pt]
&= -\underbrace{\mathbb{E}_{\mathbf{z}\sim q_\phi}\big[\log p(\mathbf{x}\mid\mathbf{z})\big]}_{\text{重构项}} + \underbrace{\mathbb{E}_{\mathbf{z}\sim q_\phi}\big[\log q_\phi(\mathbf{z}\mid\mathbf{x}) - \log p(\mathbf{z})\big]}_{\text{KL}(q_\phi \parallel p(\mathbf{z}))} + \log p(\mathbf{x})
\end{aligned}$$

### 3.2 定义 ELBO

移项得：

$$\log p(\mathbf{x}) = D_{\text{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x}) \parallel p(\mathbf{z}\mid\mathbf{x})\big) + \mathcal{L}_{\text{ELBO}}(\mathbf{x}; \theta, \phi)$$

其中 **ELBO (Evidence Lower BOund)**：

$$\boxed{\mathcal{L}_{\text{ELBO}}(\mathbf{x}; \theta, \phi) = \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big] - D_{\text{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x}) \parallel p(\mathbf{z})\big)}$$

> **为什么叫"证据下界"**：因为 $D_{\text{KL}} \geq 0$，所以 $\log p(\mathbf{x}) \geq \mathcal{L}_{\text{ELBO}}$。ELBO 是证据 $\log p(\mathbf{x})$ 的下界。最大化 ELBO 就是同时做到：(1) 提高重构质量 (2) 让近似后验逼近先验 (3) 隐式地最小化 $D_{\text{KL}}(q_\phi \parallel p)$。

### 3.3 逐项解释

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big] \;-\; D_{\text{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x}) \parallel p(\mathbf{z})\big)$$

| 项 | 符号 | 直观含义 | 数学角色 |
|----|------|----------|----------|
| **重构损失** (Reconstruction) | $\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}\mid\mathbf{z})]$ | "解码器还原输入的能力"——给定 $\mathbf{z}$ 后 $\mathbf{x}$ 的似然越大越好 | 鼓励 $\mathbf{z}$ 保留足够信息重建 $\mathbf{x}$。对于连续数据通常取高斯似然 → MSE；对于二值数据取伯努利似然 → BCE |
| **KL 正则项** (KL Regularization) | $D_{\text{KL}}(q_\phi \parallel p(\mathbf{z}))$ | "编码器输出不要太离谱"——迫使 $q_\phi$ 靠近先验 $\mathcal{N}(0,1)$ | 防止 $\mathbf{z}$ 退化为确定性编码（此时 VAE 退化为普通 Auto-Encoder），保持潜在空间的连续性和可采样性 |
| $\theta$ | 解码器参数 | 生成网络（Decoder）的权重 | $p_\theta(\mathbf{x}\mid\mathbf{z})$ 的参数 |
| $\phi$ | 编码器参数 | 推断网络（Encoder）的权重 | $q_\phi(\mathbf{z}\mid\mathbf{x})$ 的参数 |

### 3.4 直观理解：ELBO 在做什么？

**重构项** 像是老师检查学生的笔记：给定压缩后的摘要 $\mathbf{z}$，你能多好地还原原文 $\mathbf{x}$？

**KL 项** 像是图书馆的分类规则：你的摘要格式应该符合标准规范 $\mathcal{N}(0,1)$，这样才能和其他笔记放在同一书架上。

> **没有 KL 项** = 普通 Auto-Encoder，$\mathbf{z}$ 可能散落在空间中任意位置，中间区域不可采样。
> **有 KL 项** = VAE，$\mathbf{z}$ 集中在标准正态附近，任意采样都能生成合理样本。

---

## 4. 重参数化技巧（Reparameterization Trick）

### 4.1 问题

ELBO 的期望项需要对 $\mathbf{z}\sim q_\phi$ 采样，而 $\mathbf{z}$ 的分布依赖于 $\phi$：

$$\mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big]$$

如果直接在采样路径上反向传播，梯度无法流过随机节点（采样操作 $\mathbf{z}\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\sigma}^2)$ 不可导）。

### 4.2 解决方案

将随机性"外移"到一个**不依赖参数**的噪声变量 $\boldsymbol{\epsilon}$：

$$\boxed{\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}$$

| 符号 | 含义 | 维度 | 来源 |
|------|------|------|------|
| $\boldsymbol{\mu}_\phi(\mathbf{x})$ | 编码器输出的均值向量 | $\mathbb{R}^d$ | 神经网络 Encoder 的输出（无激活函数） |
| $\boldsymbol{\sigma}_\phi(\mathbf{x})$ | 编码器输出的标准差向量 | $\mathbb{R}^d$ | 神经网络 Encoder 的输出，需保证正值（通过 $\exp$ 或 softplus） |
| $\odot$ | 逐元素乘法 (Hadamard product) | — | — |
| $\boldsymbol{\epsilon}$ | 从固定分布采样的噪声 | $\mathbb{R}^d$ | $\mathcal{N}(\mathbf{0},\mathbf{I})$，**不依赖任何参数** |
| $\mathbf{z}$ | 重参数化后的隐变量 | $\mathbb{R}^d$ | 可微的确定性变换 + 独立随机噪声 |

### 4.3 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # --- Encoder ---
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)       # μ(x)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)   # log σ²(x)

        # --- Decoder ---
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """编码器：x → μ, log σ²"""
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：
        z = μ + σ ⊙ ε,   ε ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)           # σ = exp(0.5 * log σ²)
        eps = torch.randn_like(std)              # ε ~ N(0, I)，不依赖参数
        z = mu + std * eps                       # 确定性变换（可微）
        return z

    def decode(self, z):
        """解码器：z → x̂"""
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))        # [0,1] 像素值

    def forward(self, x):
        mu, logvar = self.encode(x)              # 编码
        z = self.reparameterize(mu, logvar)      # 重参数化采样
        x_recon = self.decode(z)                 # 解码
        return x_recon, mu, logvar

    def loss_function(self, x, x_recon, mu, logvar):
        """
        ELBO 损失 = 重构损失 + KL 散度
        """
        # 重构损失: BCE (二值像素) 或 MSE (连续值)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

        # KL 散度: D_KL( q(z|x) || p(z) )，p(z) = N(0, I)
        # 闭式解: -0.5 * Σ(1 + log σ² - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # ELBO = recon_loss + kl_loss → 最小化负 ELBO
        return recon_loss + kl_loss, recon_loss, kl_loss
```

### 4.4 直观理解

**为什么需要"噪声不在参数上"？**

在没有重参数化之前，计算图是这样的：

```
φ → N(μ,σ²) → z → loss
     ↑ 采样不可导！
```

有了重参数化之后：

```
φ → μ,σ → [μ + σ⊙ε] → z → loss
                ↑ ε 固定，μ,σ 可导
```

这就像是：你需要在考试中写一篇作文（$\mathbf{z}$），但你不能直接抄字典（采样不可导）。于是你准备了一个模板 $\boldsymbol{\mu}$ 和一个风格参数 $\boldsymbol{\sigma}$，然后掷骰子 $\boldsymbol{\epsilon}$ 决定一些随机因素。最终作文 $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$ 既保留了你的写作风格（可导的 $\boldsymbol{\mu},\boldsymbol{\sigma}$），又有随机性（独立的 $\boldsymbol{\epsilon}$）。

---

## 5. VAE 与 CVAE：从 ELBO 到具体模型

### 5.1 VAE（变分自编码器）

**设定**：
- 先验：$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$
- 近似后验（编码器）：$q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \operatorname{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x})))$
- 似然（解码器）：$p_\theta(\mathbf{x}\mid\mathbf{z}) = \mathcal{N}(\mathbf{x}_{\text{recon}}, \mathbf{I})$ 或 $\text{Bernoulli}(\mathbf{x}_{\text{recon}})$

**ELBO**：

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big] - D_{\text{KL}}\big(\mathcal{N}(\boldsymbol{\mu}_\phi,\boldsymbol{\sigma}_\phi^2) \parallel \mathcal{N}(\mathbf{0},\mathbf{I})\big)$$

KL 项闭式解（见 [[KL散度推导]]）：

$$D_{\text{KL}} = \frac{1}{2}\sum_{j=1}^{d}\big(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\big)$$

**生成过程**：$\mathbf{z}\sim\mathcal{N}(0,1) \;\to\; \mathbf{x}\sim p_\theta(\mathbf{x}\mid\mathbf{z})$

### 5.2 CVAE（条件变分自编码器）

**动机**：VAE 的生成是不可控的。CVAE 引入条件 $\mathbf{c}$（类别标签、文本描述、控制信号）来实现可控生成。

**设定**：
- 先验：$p(\mathbf{z}\mid\mathbf{c})$ = $\mathcal{N}(\mathbf{0},\mathbf{I})$（通常条件独立于先验）
- 近似后验：$q_\phi(\mathbf{z}\mid\mathbf{x}, \mathbf{c})$
- 似然：$p_\theta(\mathbf{x}\mid\mathbf{z}, \mathbf{c})$

**CVAE ELBO**：

$$\boxed{\mathcal{L}_{\text{CVAE}} = \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x},\mathbf{c})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z},\mathbf{c})\big] - D_{\text{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x},\mathbf{c}) \parallel p(\mathbf{z}\mid\mathbf{c})\big)}$$

**关键区别**：编码器和解码器都额外接收条件 $\mathbf{c}$ 作为输入。

| | VAE | CVAE |
|---|---|---|
| 编码器输入 | $\mathbf{x}$ | $\mathbf{x}, \mathbf{c}$ |
| 解码器输入 | $\mathbf{z}$ | $\mathbf{z}, \mathbf{c}$ |
| 生成 | 随机采样 → 随机图像 | 给定 $\mathbf{c}$ → 可控生成 |
| 具身智能应用 | 一般特征学习 | 行为克隆中**条件动作生成**（给定观测生成动作分布） |

**直观理解**：

VAE 像是闭着眼睛画画——你不知道会画出什么。
CVAE 像是命题作文——给定标题 $\mathbf{c}$（"猫"），你画的内容受标题约束，但笔触风格仍有随机性。

在机器人学习中，$\mathbf{c}$ 可以是当前观测（图像+关节状态），$\mathbf{x}$ 是下一时刻的动作。CVAE 学习 $p(\text{动作}\mid\text{观测})$ 的多模态分布。

详见 [[CVAE条件变分自编码器]]

---

## 6. $\beta$-VAE：解耦表征学习

### 6.1 动机

标准 VAE 的 ELBO 同时优化重构质量和 KL 正则。$\beta$-VAE 引入权重 $\beta > 1$ 来加大 KL 项的惩罚，迫使 $\mathbf{z}$ 的各维度更**独立**和**解耦**（disentangled）。

### 6.2 损失函数

$$\boxed{\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big] - \beta \cdot D_{\text{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x}) \parallel p(\mathbf{z})\big)}$$

| $\beta$ 值 | 效果 |
|------------|------|
| $\beta = 1$ | 标准 VAE |
| $\beta > 1$ | 更强正则化 → $\mathbf{z}$ 各维度更独立（解耦），但重构质量可能下降 |
| $\beta < 1$ | 更弱正则化 → 重构更好，但潜在空间更"纠缠"，接近 Auto-Encoder |

### 6.3 直观理解

把隐变量 $\mathbf{z}$ 想像成描述"人物头像"的 3 个维度：

- **VAE** ($\beta=1$)：$\mathbf{z}_1$ 可能同时控制"发色"和"脸型"（纠缠），只要能还原就行。
- **$\beta$-VAE** ($\beta=4$)：$\mathbf{z}_1$ 只控制"发色"，$\mathbf{z}_2$ 只控制"脸型"，$\mathbf{z}_3$ 只控制"表情"（解耦）。代价是图像可能稍微模糊。

> **信息瓶颈视角**：KL 项限制了 $\mathbf{z}$ 能传递的信息量。$\beta$ 越大，"管道"越窄，$\mathbf{z}$ 只能编码最高效（独立）的特征。

**$\beta$-VAE 在具身智能中的应用**：解耦的表征使机器人能从视觉观测中分离出"物体位置"、"自身姿态"、"任务进度"等独立因子，提升策略的泛化能力。

---

## 7. 模型对比总表

| 特性 | VAE | CVAE | $\beta$-VAE | VQ-VAE |
|------|-----|------|-------------|--------|
| **先验 $p(\mathbf{z})$** | $\mathcal{N}(\mathbf{0},\mathbf{I})$ | $\mathcal{N}(\mathbf{0},\mathbf{I})$ | $\mathcal{N}(\mathbf{0},\mathbf{I})$ | 均匀分布（Codebook 中离散索引） |
| **后验 $q(\mathbf{z}\mid\mathbf{x})$** | $\mathcal{N}(\boldsymbol{\mu}_\phi, \boldsymbol{\sigma}_\phi^2)$ | $\mathcal{N}(\boldsymbol{\mu}_\phi, \boldsymbol{\sigma}_\phi^2)$，条件于 $\mathbf{c}$ | 同 VAE | 确定性映射到最近邻 Codebook 向量 |
| **ELBO 变体** | 标准 ELBO | 条件 ELBO（编码/解码均加 $\mathbf{c}$） | ELBO 带 $\beta$ 权重 | 重构 + Codebook 损失 + 承诺损失 |
| **隐空间类型** | 连续 (Gaussian) | 连续 (Gaussian) | 连续 (Gaussian) | **离散** (Codebook 向量索引) |
| **KL 项** | $D_{\text{KL}}(q\parallel\mathcal{N}(0,1))$ | $D_{\text{KL}}(q\parallel\mathcal{N}(0,1))$ | $\beta\cdot D_{\text{KL}}(q\parallel\mathcal{N}(0,1))$ | 无（用 VQ 替代） |
| **生成质量** | 中等（模糊） | 条件可控，中等 | $\beta$↑→更模糊但更解耦 | **高**（离散表征避免后验坍塌） |
| **训练难点** | 后验坍塌 (Posterior Collapse) | 同 VAE + 条件注入方式 | 平衡 $\beta$ 值 | Codebook 坍塌、Straight-Through Estimator |
| **代表作** | Kingma & Welling, 2013 | Sohn et al., 2015 (CVAE) | Higgins et al., 2017 ($\beta$-VAE) | van den Oord et al., 2017 (VQ-VAE) |
| **具身智能典型用途** | 状态表征学习 | 行为克隆 / 轨迹生成 | 解耦技能学习 | 视觉 tokenizer (如 RT-2 的图像分词) |

---

## 8. 常见问题与误区

### 8.1 后验坍塌 (Posterior Collapse)

**现象**：训练过程中，解码器变得过于强大，$\mathbf{z}$ 被忽略，$q_\phi(\mathbf{z}\mid\mathbf{x})$ 直接等于先验 $\mathcal{N}(0,1)$。

**本质**：$\log p_\theta(\mathbf{x}\mid\mathbf{z})$ 已经不依赖 $\mathbf{z}$ 即可很好地重构 $\mathbf{x}$，$\mathbf{z}$ 不再传递信息 → KL 项驱动 $q_\phi\to p(\mathbf{z})$。

**缓解**：KL annealing（逐步增加 KL 权重）、free bits（KL 低于阈值时不惩罚）、更弱的解码器。

### 8.2 "ELBO 最大化的方向"容易搞混

$$\log p(\mathbf{x}) = \underbrace{D_{\text{KL}}(q_\phi \parallel p)}_{\geq 0} + \mathcal{L}_{\text{ELBO}}$$

- 训练目标：$\max_{\theta,\phi} \mathcal{L}_{\text{ELBO}}$
- 等价于：$\min_{\phi} D_{\text{KL}}(q_\phi\parallel p)$（同时 $\max_\theta$ 重构似然）
- 损失函数一般为：`loss = -ELBO` → 最小化负 ELBO

### 8.3 VAE 生成的图像为什么模糊？

因为 VAE 用逐像素独立的高斯/伯努利似然，本质上是逐像素 MSE/BCE，无法捕捉像素间的结构化依赖。后续工作（VQ-VAE、扩散模型）在不同层面解决了这一问题。

---

## 9. 与扩散模型的联系

变分推断是**扩散模型**的理论基石之一：

- **DDPM** 可以看作一个多层 VAE：$\mathbf{x}_0\to\mathbf{x}_1\to\cdots\to\mathbf{x}_T$，每步 $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ 是固定的高斯噪声注入，学习目标是去噪（ELBO 的变体）。
- **Score-Matching** 提供了另一条等价路径：不最大化 ELBO，而是匹配数据分布的梯度场。

详见 [[Score-Matching与SDE]]

---

[[KL散度推导]] | [[VAE与KL散度深入理解]] | [[CVAE条件变分自编码器]] | [[Score-Matching与SDE]] | [[Policy与损失函数]]
