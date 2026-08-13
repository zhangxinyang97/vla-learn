---
title: 世界模型与VLA集成
tags: [world-model, lecun, i-jepa, flip, mdp, dynamics]
aliases: [世界模型, FLIP, I-JEPA, LeCun世界模型]
---

# 世界模型与 VLA：如何正确集成？

## LeCun 的世界模型观
- 长期公开反对 Scaling Law 的 AGI 路线（GTC 与黄仁勋观点对立）
- 主张：通过观察+交互学习**事物背后运行机理的世界模型**，预测未来状态变化
- 代表作 I-JEPA；坦言可能需 10 年

## 世界模型与 VLA 的元素一致性
- 都含感知(VL) + 交互(A)；VLM 本身多模态、可大规模预训练、能预测推理未来 → 看似 VLA 就是世界模型的答案？

## 问题：现有 VLA 是「知其然，再推所以然」
- 现有框架逻辑：**先知其然（VL 感知），再推所以然（A）** = VL 引导 A
- 但按 MDP：**Action 决定 State 与 Value 变化** → 交互在 Dynamics 中起决定性作用 → 这个「所以然」就是 Action

> **正确世界模型 = 「知其所以然（Action→Dynamics），推出其然（State）」**
> VLA 是「已知世界模型 → 应用到动作推理」的推理框架，**不是**学习世界模型的训练框架。

## FLIP：世界模型视频空间规划（NUS 邵林, ICLR 2025）
- Flow-Centric Generative Planning，通用操作世界模型
- 三模块对应 MDP 核心要素：

| 模块 | 对应 | 作用 |
|---|---|---|
| 动作生成 | Action | 生成候选动作 |
| 动力学预测 | State Transition | 动作 → 视频流 |
| 价值函数预测 | Value | 评估规划 |

- 关键：**Action-Dynamics-Value 顺序**（区别于 VLA 的 VL→A）
- Action 生成在前 → Dynamics 转为 VLM 可理解的视频流 → VLM 评估完成规划 → 可同时训练+推理

## 未来方向
世界模型的学习+推理双向通路加入 VLA（VLADV？ADVLM？），待观察。

[[端到端与分层架构之争]] | [[扩散模型基础]] | [[Score-Matching与SDE]] | [[最优传输理论]]
