---
title: VQ-VAE与动作Token化
tags: [vq-vae, tokenization, rt-2, openvla, discrete-action]
aliases: [向量量化, 动作离散化]
---

# VQ-VAE 与动作 Token 化

> **一句话**：让 VLM 输出连续机器人动作的唯一办法是——**把连续动作变成离散 Token**。本文讲清两种离散化方案（可学习码本 VQ-VAE vs 固定均匀分箱），以及它们在 RT-2/OpenVLA 里的实际用法。

## 为什么需要离散化动作？（动机）

VLA 的本质是 VLM（如 LLaMA、PaLI）——一个**离散 Token 自回归预测器**，只能输出离散 Token ID。但机器人动作是连续浮点向量（7 维 EEF 位姿：$\Delta x,\Delta y,\Delta z,\Delta roll,\Delta pitch,\Delta yaw,gripper$）。

**核心矛盾**：连续动作 vs 离散 Token 输出。解法就是把连续动作「翻译」成离散 Token，让 VLM 像「预测下一个单词」一样预测下一个动作分量。

## 两种离散化方案（先厘清概念）

| 方案 | 码本 | 是否训练 | 代表 |
|---|---|---|---|
| **VQ-VAE（可学习码本）** | 可学习嵌入向量 | 需要训练 | 图像 token 化（DALL-E）、某些动作隐空间 |
| **均匀分箱（Uniform Binning）** | 固定 bin | 无需训练 | **RT-2 / OpenVLA 的动作离散化** |

> ⚠️ **常见误解**：RT-2 和 OpenVLA 的动作离散化**不是 VQ-VAE**，而是简单的均匀分箱（256 bins）。VQ-VAE 是离散化的理论背景，工程上动作 token 化选择了更简单稳定的分箱。

## VQ-VAE 是什么（核心机制）

标准 VQ-VAE（van den Oord, NeurIPS 2017）：用**可学习码本**把连续编码离散化。

```
连续输入 x → [Encoder] → z_e → 最近邻查码本 → e_k（离散索引 k）→ [Decoder] → x̂
                                ↑
                        Codebook K×D 可学习向量
```

- 码本 $e \in \mathbb{R}^{K\times D}$：$K$ 个可学习向量，$K$ 典型 512~8192
- 最近邻量化：$k = \arg\min_j \|z_e - e_j\|_2$

### 损失函数

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \underbrace{\|\text{sg}[z_e] - e\|_2^2}_{\text{Codebook Loss}} + \underbrace{\beta\|z_e - \text{sg}[e]\|_2^2}_{\text{Commitment Loss}}$$

- **sg（stop gradient）**：让 Codebook 和 Encoder **轮流**靠近对方，解决 $\arg\min$ 不可导问题
- **STE（直通估计器）**：前向用 $e_k$，反向假设量化「不存在」，梯度直通 Encoder

### Codebook 坍塌与对策

| 方法 | 原理 |
|---|---|
| EMA 更新码本 | 指数移动平均替代梯度下降 |
| Codebook Reset | 重置长期未使用的向量 |
| FSQ（有限标量量化） | 直接舍入到 bin，不用查表，天然无坍塌 |

## 在模型中的角色（SOTA 关联）

### RT-2 / OpenVLA：用均匀分箱（不用 VQ-VAE）

RT-2 的做法：7 维动作，每维独立分 **256 个均匀 bin**：

```
动作 a ∈ R^7 → 逐维度分箱 → [t₁,...,t₇]，tᵢ ∈ {0,...,255}
```

- 256 = $2^8$，一个字节，自然量化粒度；平移 ±0.1m 时分辨率 ≈ 0.78mm，够用
- 给 VLM 词汇表加 **256×7 = 1792 个动作 token**
- VLM 自回归预测：$P(\text{action}|\text{img},\text{instr}) = \prod_{i=1}^{7} P(t_i | t_{<i}, \text{img}, \text{instr})$

### 为什么 RT-2 不用 VQ-VAE？

- 动作维度低（7 维）、结构简单 → 均匀分箱已够用
- 分箱**无需训练**、无 codebook 坍塌风险、工程简单稳定
- VQ-VAE 的可学习码本在**图像 token 化**（高维、需要学语义）里才真正发挥价值

## 工程化实现

- **分箱粒度**：256 bins 是动作离散化的默认选择（精度/词汇量的平衡）；更高精度可用 512/1024
- **token 注入**：动作 token 的 embedding 初始化为 0 或小随机值，随训练学习
- **OpenVLA 的做法**：同样离散化（256 bins），但动作是 EEF 相对增量（7 维），与 RT-2 一致

## 局限与现状

- **分箱的量化误差**：离散化必然损失精度（±0.78mm 级），对高精度插销类任务可能不够 → 后续工作探索连续动作头（如 π0 的 Flow Matching 直接输出连续动作）
- **自回归的顺序依赖**：7 个 token 顺序生成，有累积误差

[[VLA大模型]] | [[VAE与KL散度深入理解]] | [[CVAE条件变分自编码器]]
