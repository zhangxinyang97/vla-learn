---
title: NVIDIA-Sim-to-Real技术
tags: [sim-to-real, nvidia, isaac-sim, domain-randomization, neural-actuator]
aliases: [Neural Actuator, Sim-to-Real Gap]
---

# NVIDIA Sim-to-Real 技术

两大Gap：**动力学Gap**(电机非线性) + **视觉Gap**(光照纹理)。

## Neural Actuator

问题：仿真假设电机理想(τ=K_t·I)，真实有齿隙/摩擦/延迟/热衰减。

方案：数据驱动MLP学习真实电机行为→冻结权重插入Isaac Sim→RL训练面对真实响应。

训练：真机采集轨迹→监督学习(指令→传感器)→冻结插入仿真。

## 域随机化

物理随机化(质量/摩擦/阻尼/延迟) + 视觉随机化(光照/纹理/相机噪点)。

> 随机化范围覆盖真实世界→策略自然鲁棒。

## Teacher-Student蒸馏

| Teacher(特权Critic) | Student(Policy) |
|---|---|
| 仿真中"开挂"读精确物理参数 | 仅真实传感器数据 |
| 提供准确Value | 在Teacher指导下学习→Zero-shot真机 |

## 生成式AI

Eureka(LLM写RL奖励函数)、GR00T-Mimic(少量演示→海量仿真数据)、Re3Sim(3DGS视频→仿真场景)。

[[仿真引擎与ROS2部署]] | [[人形机器人开源框架]] | [[正逆运动学与PID控制]]
