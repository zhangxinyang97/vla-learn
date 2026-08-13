---
title: Flow-Matching流匹配
tags: [flow-matching, diffusion, ode, optimal-transport]
aliases: [Rectified Flow, 概率流ODE]
---

# Flow Matching（流匹配）

> 扩散模型的进化版——将弯曲去噪轨迹"拉直"，1-10步推理。

## 核心思想

扩散路径弯曲(100步)→Flow Matching直线(1-10步)：
```
DDPM: 噪声 ~ ~ ~ 弯曲 ~ ~ ~ > 数据 (100步)
FM:   噪声 ──────── 直线 ────> 数据 (1-10步)
```

## 数学框架

概率流ODE：$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{c})$

训练目标（最优传输OT路径）：
$$\mathcal{L} = \mathbb{E}\left[\|\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{c}) - (\mathbf{x}_1 - \mathbf{x}_0)\|^2\right]$$

其中$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$（直线插值）。

## Rectified Flow

训练后若路径不够直→用生成样本再训练→2-3轮Reflow后几乎完全直线化。

## 在机器人中的应用

**π₀** (Physical Intelligence, 2024)：Flow Matching统一引擎，1-10步ODE积分，跨多种机器人形态，推理<5ms。

## 对比

| | Diffusion | Flow Matching |
|---|---|---|
| 路径 | 弯曲SDE | 直线ODE |
| 步数 | 10-100 | 1-10 |
| 代表 | Diffusion Policy, DP3 | π₀, Octo v2 |

**结合了CVAE的速度和Diffusion的质量——目前具身智能大模型首选引擎。**

[[扩散模型基础]] | [[VLA大模型]] | [[Diffusion-Policy详解]]
