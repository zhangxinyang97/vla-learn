---
title: VLA大模型
tags: [vla, rt-2, openvla, octo, pi-zero, embodied-llm]
aliases: [RT-2, OpenVLA, Octo, π₀]
---

# VLA 大模型：RT-2 / OpenVLA / Octo / π₀

VLA = VLM(视觉-语言模型) + Action Head → 理解语言+感知场景+输出动作。

## RT-2 (Google, 2023)

VLM输出Token直接当动作：连续动作离散化为256 bins×7维→7个Token，LLM自回归预测。

Co-fine-tuning(互联网+机器人数据)效果>纯机器人微调。涌现理解"拿起将要掉落的物体"等抽象概念。

## OpenVLA (Stanford, 2024)

开源VLA标杆。SigLIP+DINOv2双视觉→Llama2 7B→动作头。LoRA微调，单GPU可用。基于Open X-Embodiment数据集(100万+轨迹)。

## Octo (UCB, 2024)

通用多embodiment统一Transformer，扩散头或自回归头生成动作。

## π₀ (Physical Intelligence, 2024)

Flow Matching统一引擎，1-10步ODE积分。跨多种机器人形态(单臂/双臂/移动操作)。**当前最接近"通用具身基础模型"的工作**。

## VLA对比

| | RT-2 | OpenVLA | Octo | π₀ |
|---|---|---|---|---|
| 年份 | 2023 | 2024 | 2024 | 2024 |
| 参数 | 55B | 7B | 27M* | ~3B |
| 引擎 | 自回归Token | 自回归Token | Diffusion | Flow Matching |
| 开源 | ❌ | ✅ | ✅ | ❌ |

*仅策略网络

## 挑战

数据瓶颈、大模型延迟vs实时控制、泛化vs精度权衡。解决方向：Flow Matching加速、分层架构(VLA高层规划+专用策略底层执行)。

[[策略与生成的统一趋势]] | [[Flow-Matching流匹配]] | [[扩散模型基础]] | [[CLIP与SigLIP对比]]
