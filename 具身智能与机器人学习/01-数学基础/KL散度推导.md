---
title: KL散度推导
tags: [kl-divergence, vae, information-theory, math-foundation]
aliases: [KL散度, 相对熵, 交叉熵]
created: 2026-08-13
---

# KL 散度推导

> **定位**：数学地基之一。VAE/CVAE 的训练目标、RL 的策略约束、扩散模型的 ELBO 里，KL 散度无处不在。本篇教学目标：**从定义手推出"两个高斯之间的 KL"与"VAE 特例"，并看懂每一项在干什么**。
>
> 前置：[[向量内积与外积]]（期望与基本运算）。关联：[[变分推断与ELBO]] [[VAE与KL散度深入理解]] [[Policy与损失函数]]。

## 一、定义与直觉

$$D_{\text{KL}}(P \parallel Q) = \mathbb{E}_{x\sim P}\left[\log\frac{P(x)}{Q(x)}\right] = \sum_x P(x)\log\frac{P(x)}{Q(x)}$$

**直觉**：用分布 $Q$ 去"编码"真实分布 $P$ 的样本，平均要多花多少额外代价（相对熵）。

两个关键性质（必须记住）：

1. **非负**：$D_{\text{KL}}(P\parallel Q) \ge 0$，仅当 $P=Q$ 时为零（由吉布斯不等式 / Jensen 不等式保证）
2. **不对称**：$D_{\text{KL}}(P\parallel Q) \ne D_{\text{KL}}(Q\parallel P)$——所以它**不是距离**，只是"散度"

> 不对称的直觉：$P$ 是真实分布，$Q$ 是近似。$D_{\text{KL}}(P\parallel Q)$ 惩罚"真实有值而近似没值"（漏掉模式），是 VAE 里常用的方向。

## 二、两个高斯之间的 KL（分步推导）

设 $P=\mathcal{N}(\mu_1,\sigma_1^2)$，$Q=\mathcal{N}(\mu_2,\sigma_2^2)$，概率密度 $\mathcal{N}(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$。

**第 1 步**：写出 $\log\frac{P}{Q}$：

$$\log\frac{P}{Q} = \log\frac{\sigma_2}{\sigma_1} - \frac{(x-\mu_1)^2}{2\sigma_1^2} + \frac{(x-\mu_2)^2}{2\sigma_2^2}$$

**第 2 步**：对 $x\sim P$ 取期望，用两个事实：$\mathbb{E}[x]=\mu_1$、$\mathbb{E}[(x-\mu_1)^2]=\sigma_1^2$：

- 第一项是常数：$\log\frac{\sigma_2}{\sigma_1}$
- 第二项：$-\frac{\mathbb{E}[(x-\mu_1)^2]}{2\sigma_1^2} = -\frac{1}{2}$
- 第三项：$\frac{\mathbb{E}[(x-\mu_2)^2]}{2\sigma_2^2} = \frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}$

（第三项用 $\mathbb{E}[(x-\mu_2)^2]=\mathbb{E}[(x-\mu_1+\mu_1-\mu_2)^2]=\sigma_1^2+(\mu_1-\mu_2)^2$，交叉项期望为 0。）

**第 3 步**：合并：

$$\boxed{D_{\text{KL}}(P\parallel Q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}}$$

## 三、VAE 特例：$Q=\mathcal{N}(0,1)$

代入 $\mu_2=0,\ \sigma_2=1$：

$$D_{\text{KL}} = \log\frac{1}{\sigma_1} + \frac{\sigma_1^2+\mu_1^2}{2} - \frac{1}{2} = \frac{1}{2}\big(\mu_1^2 + \sigma_1^2 - 1 - \ln\sigma_1^2\big)$$

$$\boxed{D_{\text{KL}}\big(\mathcal{N}(\mu,\sigma^2)\,\big\Vert\,\mathcal{N}(0,1)\big) = \frac{1}{2}\big(\mu^2 + \sigma^2 - 1 - \ln\sigma^2\big)}$$

$d$ 维（各维独立）推广：

$$D_{\text{KL}} = \frac{1}{2}\sum_{i=1}^{d}\big(\mu_i^2 + \sigma_i^2 - 1 - \ln\sigma_i^2\big)$$

## 四、各项在"干什么"（VAE 的正则化机制）

| 项 | 作用 | 直觉 |
|----|------|------|
| $\mu^2$ | 推动 $\mu\to 0$ | 把隐变量中心拉到原点 |
| $\sigma^2$ | 推动 $\sigma^2\to 1$ | 让隐变量方差接近标准正态 |
| $-\ln\sigma^2$ | 当 $\sigma^2\to 0$ 时 $\to+\infty$ | **防止方差塌缩**（退化成点） |
| $-1$ | 常数 | 保证 $P=Q$ 时散度为 0 |

> **为什么防止方差塌缩**：VAE 若让 $\sigma\to0$，隐变量就退化成确定性点，失去"从 $N(0,1)$ 采样"的生成能力。$-\ln\sigma^2$ 这一项像弹簧一样把它拉回来。

## 五、信息论视角：KL 与交叉熵

$$D_{\text{KL}}(P\parallel Q) = \underbrace{H(P,Q)}_{\text{交叉熵}} - \underbrace{H(P)}_{\text{熵}}$$

- $H(P,Q) = -\mathbb{E}_{x\sim P}[\log Q(x)]$：用 $Q$ 编码 $P$ 的总代价
- $H(P) = -\mathbb{E}_{x\sim P}[\log P(x)]$：$P$ 本身的信息量（常数）

**推论**：当 $P$ 固定时，最小化 KL $\iff$ 最小化交叉熵。这就是分类任务用交叉熵损失的理论依据（见 [[Policy与损失函数]]）。

## 六、在具身/ML 里的落点

| 场景 | KL 怎么用 | 关联 |
|------|----------|------|
| VAE/CVAE 训练 | ELBO 里的正则项（隐变量 → $N(0,1)$） | [[变分推断与ELBO]] [[CVAE条件变分自编码器]] |
| ACT | β-VAE 的 KL 权重 β 控制隐空间约束强度 | [[ACT算法详解]] |
| RL 策略约束 | TRPO 用 KL 约束策略更新幅度，PPO 用 clip 近似 | [[强化学习策略-PPO与SAC]] |
| 蒸馏/对齐 | 教师-学生分布对齐（KL 或反向 KL） | — |

## 检查点

1. ✅ 手推 $Q=\mathcal{N}(0,1)$ 时的 KL 公式
2. ✅ 说清 $\mu^2,\sigma^2,-\ln\sigma^2$ 三项各自在干什么
3. ✅ 解释"P 固定时最小化 KL = 最小化交叉熵"

## 前置 / 关联

前置：[[向量内积与外积]]
关联：[[变分推断与ELBO]] | [[VAE与KL散度深入理解]] | [[Policy与损失函数]] | [[CVAE条件变分自编码器]]
