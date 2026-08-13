---
title: 正逆运动学与PID控制
tags: [robotics, kinematics, pid, impedance-control, fk-ik]
aliases: [FK/IK, 阻抗控制]
---

# 正逆运动学与 PID 控制

策略输出"目标位姿"→IK→关节角→PID→电机指令。

## 正运动学FK

已知关节角$\boldsymbol{\theta}$，求末端位姿$\mathbf{T}_{\text{EE}}$。DH参数法链式相乘。

## 逆运动学IK

已知目标位姿，反算关节角。雅可比迭代：$\Delta\boldsymbol{\theta} = \mathbf{J}^+\Delta\mathbf{x}$

常见问题：多解(选最近解)、奇异点(阻尼最小二乘)、关节限位(投影)、不可达。

## PID控制

$$u(t) = K_p e(t) + K_i\int e(\tau)d\tau + K_d\frac{de}{dt}$$

| 项 | 作用 |
|---|---|
| P | 响应当前误差(大→快但可能振荡) |
| I | 消除稳态误差(大→消静差但可能过冲) |
| D | 抑制振荡(预测趋势、增加阻尼) |

机器人中：关节空间PID(每关节独立) 或 笛卡尔空间PID(末端误差×雅可比转置)。

## 阻抗控制

使机械臂呈现"弹簧-阻尼-质量"特性，实现柔顺抓取和人机安全：

$$\boldsymbol{\tau} = \mathbf{J}^T(\mathbf{K}_d\mathbf{e} + \mathbf{D}_d\dot{\mathbf{e}}) + \mathbf{g}(\boldsymbol{\theta})$$

阻抗(测位移→出力) vs 导纳(测力→出位移)。

## 完整控制链路

```
策略→目标位姿(10-50Hz)→IK求解→目标关节角→PID(1000Hz+)→电机→编码器反馈→闭环
```

[[SE3与SO3空间变换]] | [[仿真引擎与ROS2部署]] | [[动作空间表征总览]]
