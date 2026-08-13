---
title: Flow-Matching流匹配
tags: [flow-matching, diffusion, ode, optimal-transport, pi-zero]
aliases: [Rectified Flow, 概率流ODE, 流匹配]
---

# Flow Matching（流匹配）

> **一句话**：把扩散模型「弯曲的多步去噪路径」换成「**直线的一步到位路径**」，用 ODE 在 1~10 步内生成，速度逼近 CVAE、质量看齐扩散。

## 为什么需要 Flow Matching？（动机）

扩散模型质量高但**慢**（要 100 步去噪）。机器人实时控制要求动作生成在 **<10ms** 内完成——100 步根本来不及。

根本原因：扩散的路径是**弯曲的 SDE**（每一步都带随机抖动），逼得你必须小步走。

Flow Matching 的洞察：**如果路径是直的，理论上一步就能到终点**。所以它训练网络直接学「**速度场**」——从噪声到数据的一条直线。

> **直觉类比**：扩散是「迷宫里走弯路」，Flow Matching 是「两点之间拉一条直线」。

## 怎么工作（核心机制）

### 概率流 ODE

$$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{c})$$

- $\mathbf{v}_\theta$：速度场（网络输出），指向数据方向
- 沿 ODE 积分即可从噪声 $\mathbf{x}_0$ 到达数据 $\mathbf{x}_1$

### 直线路径 + 训练目标

路径取直线插值：$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$

训练目标（最优传输 OT 路径）：让速度场逼近「终点减起点」的直线速度：

$$\mathcal{L} = \mathbb{E}\left[\|\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{c}) - (\mathbf{x}_1 - \mathbf{x}_0)\|^2\right]$$

> 这正是 [[最优传输理论]] 的结论：**所有分布变换中，直线是最短路径**。

### Rectified Flow（拉直）

若训练后路径仍不够直 → 用生成样本再训练一轮 → 2~3 轮 **Reflow** 后几乎完全直线化。

## 在模型中的角色（SOTA 关联）

**π0（Physical Intelligence, 2024）**：以 Flow Matching 为统一引擎，替代扩散。
- **条件流匹配**（CFM）：以观测+指令为条件 $\mathbf{c}$
- **Action Chunk**：一次生成整块动作
- **50Hz** 高频生成连续动作分布，跨多种机器人形态
- 1-10 步 ODE 积分，推理 <5ms

## 工程化实现

- **步数**：1-10 步（扩散要 10-100），越少越快但质量略降
- **ODE 求解器**：Euler 即可（直线路径简单），或用更高阶 RK 提高精度
- **Reflow 蒸馏**：多轮 reflow 拉直路径 → 实现「一步生成」
- **训练稳定性**：OT 路径（直线插值）比随机配对更稳定、收敛更快

## 对比与现状

| | Diffusion | Flow Matching |
|---|---|--|
| 路径 | 弯曲 SDE | 直线 ODE |
| 步数 | 10-100 | 1-10 |
| 速度 | 10-100ms | 1-10ms |
| 代表 | Diffusion Policy, DP3 | π0, Octo v2 |

**结论：Flow Matching 结合了 CVAE 的速度和 Diffusion 的质量——当前具身智能大模型的首选引擎。**

[[扩散模型基础]] | [[VLA大模型]] | [[Diffusion-Policy详解]] | [[最优传输理论]]
