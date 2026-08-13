---
title: CVAE条件变分自编码器
tags: [cvae, imitation-learning, generative-model, robotics]
---

# CVAE 条件变分自编码器

在条件$c$下生成多样化数据，建模"一对多"的多模态映射。

机器人领域几乎所有"VAE"算法底层都是CVAE——必须根据环境做动作($p(\text{action}\mid\text{state})$)。

## 四大应用场景

| 场景 | 条件$c$ | 生成目标 |
|---|---|---|
| 轨迹规划 | 障碍物+起终点 | 多条无碰撞路径 |
| 6D抓取 | 物体3D点云 | 多个抓取位姿SE(3) |
| 行为克隆 | 环境感知特征 | 多模态动作分布 |
| 世界模型 | 历史观测+指令 | 未来视觉预测 |

## SOTA工作

| 工作 | 领域 | CVAE解决问题 |
|---|---|---|
| **ACT** (RSS'23) | 模仿学习 | 动作多模态歧义(左绕/右绕) |
| **6-DOF GraspNet** | 抓取 | 多角度无碰撞采样 |
| **MPNet** (IJRR) | 运动规划 | 启发式采样分布 |
| **Trajectron++** (ECCV) | 轨迹预测 | 意图不确定性 |

## CVAE vs Diffusion

CVAE推理<5ms适合高频控制；Diffusion 10-100ms但质量极高。

[[ACT算法详解]] | [[VAE与KL散度深入理解]] | [[扩散模型基础]]
