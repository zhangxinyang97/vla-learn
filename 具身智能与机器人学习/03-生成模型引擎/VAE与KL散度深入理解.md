---
title: VAE与KL散度深入理解
tags: [vae, kl-divergence, latent-space]
---

# VAE 与 KL 散度深入理解

> 破解困惑："KL要求z~N(0,I)，直接抽样不就行了？"

## 核心答案

$\mathcal{N}(0,I)$是**规矩**，Encoder输出的$\mu_X,\sigma_X$是**答案**。

- 跑步 → $\mu=[0.8,-0.5,\dots]$
- 深蹲 → $\mu=[-0.9,0.3,\dots]$

如果都用$\mu=0$，所有动作混在一起，$z$变纯噪声，Policy不知道该跑还是蹲。

## KL散度的作用

**没有KL**：Encoder走捷径，门牌号设到±10000，中间无人区→采样崩盘。

**有KL**：强制紧凑压缩在N(0,I)范围内——有区分但连续平滑，支持插值。

## 数学事实

KL直接对$\mu,\sigma$下手：
$$D_{\text{KL}} = \frac{1}{2}\sum_{i=1}^d(\mu_i^2 + \sigma_i^2 - 1 - \ln\sigma_i^2)$$

间接塑造所有采样的$z$贴近N(0,I)。

## 训练什么？

Encoder学会分配互斥门牌号；Policy学会"看区域行事"。

[[KL散度推导]] | [[CVAE条件变分自编码器]] | [[ACT算法详解]]
