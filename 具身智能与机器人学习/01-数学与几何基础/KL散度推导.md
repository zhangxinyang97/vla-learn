---
title: KL散度推导
tags: [kl-divergence, vae, information-theory, math-foundation]
---

# KL 散度推导

$$D_{\text{KL}}(P \parallel Q) = \mathbb{E}_{x\sim P}\left[\log\frac{P(x)}{Q(x)}\right]$$

≥ 0，仅当 P=Q 时为零。不对称。

## 两个高斯分布之间

$P=\mathcal{N}(\mu_1,\sigma_1^2)$, $Q=\mathcal{N}(\mu_2,\sigma_2^2)$：

$$D_{\text{KL}} = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

## VAE特例：$Q=\mathcal{N}(0,1)$

$$\boxed{D_{\text{KL}} = \frac{1}{2}(\mu^2 + \sigma^2 - 1 - \ln\sigma^2)}$$

$d$维推广：$\frac{1}{2}\sum_{i=1}^d(\mu_i^2 + \sigma_i^2 - 1 - \ln\sigma_i^2)$

## 各项含义

| 项 | 作用 |
|---|---|
| $\mu^2$ | 推动 $\mu\to 0$ |
| $\sigma^2$ | 推动 $\sigma^2\to 1$ |
| $-\ln\sigma^2$ | 防止方差塌缩（$\sigma^2\to 0$ 时 → +∞） |

## 信息论视角

$D_{\text{KL}}(P\parallel Q) = H(P,Q) - H(P)$ = 交叉熵 - 熵

P固定时，最小化KL等价于最小化交叉熵。

[[VAE与KL散度深入理解]] | [[CVAE条件变分自编码器]] | [[Policy与损失函数]]
