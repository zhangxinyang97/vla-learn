---
title: 端到端VLA与分层架构-概览
tags: [MOC, vla, layered-architecture, embodied-ai, world-model, survey]
updated: 2026-08-13
aliases: [端到端vs分层, VLA分层架构概览]
---

# 端到端 VLA 与分层架构：概览与学习路径

> **本模块**：从「端到端 VLA」到「分层 VL+A 架构」的完整技术地图。
> 覆盖 2023-2025 头部开源/商业 VLA 模型盘点 + 分层架构盘点 + 架构之争的理论思辨 + 未来方向。

---

## 目录树

```
08-端到端VLA与分层架构/
│
├── 端到端VLA与分层架构-概览与学习路径.md   ← 你在这里
│
├── 端到端VLA模型盘点-上.md                ← Octo / OpenVLA / OpenVLA-OFT+ / GR-1/2
├── 端到端VLA模型盘点-下.md                ← RDT-1B / π0 / GraspVLA / GO-1(ViLLA)
│
├── 端到端与分层架构之争.md                ← 暴力美学 vs 模块化，Scaling Law 困境
├── 世界模型与VLA集成.md                  ← LeCun 世界模型、FLIP、MDP
│
├── 分层架构-VLM加运动规划.md              ← VoxPoser / ReKep / Vila+CoPa / OmniManip
├── 分层架构-VLM加技能控制器.md            ← HumanPlus / Susie / ATM / ManipGen / DexGraspVLA
│
├── VLA下一步-动作形式与整身协同.md         ← 工作空间vs关节空间、移动操作、WBC/MPC
├── 从VLM到空间智能.md                    ← 3D-GS / RoboGSim / CAST
├── ChatVLA与训练范式解耦.md               ← 虚假遗忘 / 任务干扰 / MoE
├── 2025-VLA新进展盘点.md                  ← Gemini Robotics / π0.5 / GR00T / OpenVLA 2
├── 灵巧操作与触觉感知.md                  ← 灵巧手 / 触觉 / DexGraspVLA
├── 分层架构总览-从意图到执行.md            ← 🆕 分层路线教学：指令→执行五步链路
└── 端到端合一路径详解.md                  ← 🆕 合一路线教学：动作空间/统一动作
```

## 核心结论（TL;DR）

1. **端到端 VLA** 商业落地更快，适合打造**专用场景智能**；**分层架构**更接近具身智能终极愿景（通用泛化的物理世界智能）。
2. 当前 VLA 的「A」受框架先天制约：Scaling Law 数据困境、RL 的 EvE 博弈、long-horizon 能力缺失。
3. 分层「VLM + MotionPlanning / 技能控制器」盘活既有机器人算法，是当前可解释、可组合的务实路线。
4. 世界模型的正确姿势是「知其所以然（Action 决定 Dynamics），推出其然」——与现有「VL 引导 A」范式相反。
5. 未来三件事：整身移动操作协同、VLM→空间智能对齐、训练范式解耦（ChatVLA）。

## 学习路径

### 路径 A：先看全景（1 小时）⭐
```
端到端与分层架构之争 → 端到端VLA模型盘点-上 → 端到端VLA模型盘点-下
  → 分层架构-VLM加运动规划 → 分层架构-VLM加技能控制器
  → 2025-VLA新进展盘点 → 灵巧操作与触觉感知
```

### 路径 B：理论思辨（研究导向）⭐⭐⭐⭐
```
端到端与分层架构之争 → 世界模型与VLA集成 → ChatVLA与训练范式解耦
  → 关联阅读：[[Score-Matching与SDE]] [[最优传输理论]] [[变分推断与ELBO]]
```

### 路径 C：未来方向（工程导向）⭐⭐
```
VLA下一步-动作形式与整身协同 → 从VLM到空间智能
  → 关联阅读：[[正逆运动学与PID控制]] [[轨迹规划-RRT与样条插值]] [[3D-Gaussian-Splatting]]
```

## 关联既有笔记

[[VLA大模型]] | [[策略与生成的统一趋势]] | [[Flow-Matching流匹配]] | [[扩散模型基础]] | [[动作空间表征总览]] | [[正逆运动学与PID控制]] | [[轨迹规划-RRT与样条插值]] | [[3D-Gaussian-Splatting]] | [[2025-VLA新进展盘点]] | [[灵巧操作与触觉感知]] | [[分层架构总览-从意图到执行]] | [[端到端合一路径详解]]
