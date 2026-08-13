---
title: Diffusion-Policy详解
tags: [diffusion-policy, imitation-learning, ddpm, action-chunking]
aliases: [DP, 扩散策略]
---

# Diffusion Policy (DP)

Columbia/TRI, RSS 2023。机器人模仿学习主流标杆——扩散生成 + Action Chunking。

## 核心架构

```
观测o_t→条件编码(ResNet/ViT)→c
噪声ε~N(0,I)→x_T→U-Net去噪(c, k)→x_0→动作块A_t
```

输入：最近2帧RGB+本体感觉 | 输出：未来16步EEF相对增量

## 训练

```python
noise = randn_like(action_seq)
k = randint(0, 100)
noisy = add_noise(action_seq, noise, k)
pred_noise = unet(noisy, k, cond=obs_encoder(obs))
loss = MSE(pred_noise, noise)
```

## 滚动时域控制

时间t：预测[a_t,...,a_{t+15}]→执行前N步→时间t+N：重新预测→...

## 为什么比ACT好？

| | ACT(CVAE) | Diffusion Policy |
|---|---|---|
| 分布 | 高斯假设 | **任意分布** |
| 精细操作 | 一般 | **优秀** |
| 速度 | <5ms | 10-100ms(DDIM 10步≈20ms) |
| 数据需求 | 中 | **更少** |

## DP3 (3D升级版)

视觉编码2D ResNet→**PointNet++点云编码**，天然SE(3)等变性，Sim-to-Real更鲁棒。

[[扩散模型基础]] | [[ACT算法详解]] | [[动作空间表征总览]]
