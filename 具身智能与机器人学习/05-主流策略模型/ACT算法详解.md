---
title: ACT算法详解
tags: [act, cvae, transformer, imitation-learning, action-chunking]
aliases: [Action Chunking with Transformers]
---

# ACT：Action Chunking with Transformers

Stanford ALOHA, RSS 2023。CVAE + Transformer Decoder → 一次预测100帧动作块。

## 架构

```
观测图像→ResNet→视觉特征─┐
关节状态→MLP→本体特征────┤→Transformer Decoder→[a_0,...,a_99]
隐变量z~N(0,I)→拼接─────┘
```

## CVAE的角色

| | Encoder | Decoder |
|---|---|---|
| 训练 | 从演示动作推断z(μ,σ) | z+观测→预测动作，算重建损失 |
| 推理 | **不使用** | 从N(0,I)采样z→生成动作 |

总损失：$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}$

## Action Chunking核心

- 块长K=100@50Hz=2秒动作块
- Transformer自注意力保证块内时间一致性
- 避免逐帧预测的复合误差漂移

## 关键参数

14维绝对关节角、ResNet-18视觉编码、β-VAE KL权重。

[[CVAE条件变分自编码器]] | [[Diffusion-Policy详解]] | [[动作空间表征总览]]
