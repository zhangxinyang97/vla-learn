---
title: 端到端VLA模型盘点-上
tags: [vla, octo, openvla, openvla-oft, gr-2, gr-1, end-to-end]
aliases: [Octo, OpenVLA, OpenVLA-OFT+, GR-2]
---

# 端到端 VLA 模型盘点（上）：Octo / OpenVLA / OpenVLA-OFT+ / GR-1/2

> 端到端 = 视觉+语言输入 → 动作输出，无中间规划/控制层。本文盘点 2023-2025 头部可实践性工作。

## Octo (UCB/Stanford/CMU/DeepMind, 2024)

大型 Transformer 策略框架：**任意输入 token → 输出 token（编码为动作）**。

- 输入端：预训练语言模型（任务描述）+ 轻量 CNN（观察）分别 token 化
- Transformer 主干处理任务+观察 token 序列 → 读数 token → 输出头生成动作
- **逐块注意力结构**：微调时可增删输入输出（新观察/新动作空间）而不动预训练参数
- Open X-Embodiment 800k 轨迹训练；9 个机器人平台验证
- 无需额外训练即可切换相机配置/机器人本体/语言指令/目标图像引导
- 可作通用策略初始化，消费级 GPU 数小时微调适配新机器人

## OpenVLA (Stanford, 2024)

- Open X-Embodiment 970k 轨迹；微调预训练 **Prismatic-7B VLM** backbone
- 动作预测表述为「视觉-语言」任务：观测图 + 语言指令 → 动作序列
- 视觉编码器融合 **DINOv2 + SigLIP** 预训练特征
- 连续动作 → 离散 token（复用语言模型 tokenizer）
- 29 任务 + 多本体：绝对成功率比 RT-2-X(55B) **高 16.5%**，参数量 **少 7 倍**

## OpenVLA-OFT+

OpenVLA 增强版，把 **自回归(AR) → Parallel Decoding**：

| 维度 | OpenVLA | OpenVLA-OFT+ |
|---|---|---|
| 解码 | 自回归逐 token | 并行一次性多步 |
| 注意力 | 因果 Mask | 双向 Attention Mask |
| 损失 | CE | L1 回归 loss |
| 动作 | 离散 token | 连续动作（支持 ACT） |

> 收敛/推理速度比 Diffusion 更快，兼顾 ACT 式多步连续输出。

## GR-1 / GR-2 (字节跳动)

生成式预训练 VLM：把未来动作预测为 **视频帧(OBS Token) + 末端轨迹(ACT Token)**。

- GR-1：0.8M 视频，CALVIN 仿真验证
- GR-2：38M 视频大规模预训练，部署到实体机械臂
- 主干：Transformer 自回归，海量视频预训练 → 机械臂数据微调
- **部署后处理**：CVAE 生成轨迹 → 自研 **WBC(Whole-Body Control)** 逆解优化 → 关节空间 → 插帧 200Hz，兼顾碰撞/可操作性

> 关键点：GR-2 虽是端到端，执行层仍用 WBC 兜底 → 端到端与经典控制的边界并非非此即彼。

[[VLA大模型]] | [[CLIP与SigLIP对比]] | [[DINOv2与自监督视觉特征]] | [[VQ-VAE与动作Token化]]
