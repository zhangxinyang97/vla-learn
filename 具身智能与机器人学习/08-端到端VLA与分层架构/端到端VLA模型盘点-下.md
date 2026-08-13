---
title: 端到端VLA模型盘点-下
tags: [vla, rdt, pi-zero, graspvla, go-1, villa, flow-matching, diffusion]
aliases: [RDT-1B, π0, GraspVLA, GO-1, ViLLA]
---

# 端到端 VLA 模型盘点（下）：RDT-1B / π0 / GraspVLA / GO-1

## RDT-1B (Robot Diffusion Transformer)

双手操作扩散基础模型，**1.2B 参数**（最大扩散式操作基础模型）。

- 扩散模型表示多模态；可扩展 Transformer 处理多模态输入异质性，捕获机器人数据非线性+高频
- **物理可解释的统一动作空间**：统一不同机器人动作表示，保留物理意义 → 迁移物理知识
- 最大多机器人数据集预训练；自建 6K+ 双手数据微调
- 零样本泛化、1~5 次演示学新技能、灵巧长任务

## π0 (Physical Intelligence)

跨本体多任务通用 VLA，**预训练 VLM + Flow Matching**，曾被誉为「地表最强」。

- 借鉴 **Transfusion**：图/语言走标准 VLM 骨架(PaliGemma)，机器人动作/状态走 **Action Expert 模块**
- **条件流匹配 + Action Chunk**：50Hz 生成连续动作分布
- 预训练：OXE(22 机器人) + π 数据集(7 机器人/68 任务)，>10000 小时
- 预训练(以量取胜) + 微调(少量高质量) 两段式

## GraspVLA (银河通用, 王鹤团队)

全球第一个**完全基于仿真合成大数据**预训练的端到端抓取基础模型。

- 预训练：**十亿帧「视觉-语言-动作」对**（史上最大体量），纯合成
- Sim2Real 零样本泛化到未见场景/物体；后训练小样本适配
- **七大泛化金标准**：光照 / 干扰物 / 平面位置 / 高度 / 背景 / 物体类别 / 闭环
- 团队底色：CV+Diffusion 小模型 + DexGraspNet/GAPartNet 大规模仿真数据集

## GO-1 / ViLLA (智元, 2025)

智元「启元大模型」，首个通用具身基座模型，提出 **ViLLA = VLM + MoE**。

- **VLM**：海量互联网图文 → 通用场景感知 + 语言理解
- **MoE**：
  - Latent Planner（隐式规划器）：跨本体 + 人类操作视频 → 通用动作理解
  - Action Expert（动作专家）：百万真机数据 → 精细动作执行
- 预测 **Latent Action Tokens（隐式动作标记）**，弥合「图像-文本输入」与「机器人执行动作」的鸿沟
- 小样本快速泛化；真实灵巧操作 + 长时任务超越开源 SOTA

> ViLLA vs VLA：核心区别是把「A」拆成隐式动作标记 + 专家分工，降低具身智能门槛。

## 八大模型速查

| 模型 | 机构 | 引擎 | 参数 | 数据 | 特色 |
|---|---|---|---|---|---|
| Octo | UCB等 | Transformer | 27M* | OXE 800k | 逐块注意力可增删 IO |
| OpenVLA | Stanford | AR Token | 7B | OXE 970k | 开源标杆 |
| OpenVLA-OFT+ | - | Parallel Decoding | 7B | - | 双向注意力+L1 |
| GR-2 | 字节 | AR+CVAE | - | 38M视频 | OBS+ACT Token，WBC兜底 |
| RDT-1B | - | Diffusion | 1.2B | 多机器人 | 统一动作空间 |
| π0 | PI | Flow Matching | ~3B | OXE+π | Transfusion，50Hz |
| GraspVLA | 银河通用 | Diffusion | - | 十亿帧合成 | Sim2Real 零样本 |
| GO-1 | 智元 | MoE+VLM | - | 互联网+视频+真机 | ViLLA 隐式动作 |

[[VLA大模型]] | [[Flow-Matching流匹配]] | [[扩散模型基础]] | [[CVAE条件变分自编码器]]
