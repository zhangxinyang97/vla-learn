---
title: 仿真引擎与ROS2部署
tags: [simulation, mujoco, isaac-gym, ros2, deployment]
---

# 仿真引擎与 ROS2 部署

## 主流仿真引擎

| 引擎 | 特点 | 场景 |
|---|---|---|
| **MuJoCo** | 接触力学精准、轻量 | 学术RL基准 |
| **Isaac Sim/Lab** | GPU加速+光追、与Isaac Lab集成 | Sim-to-Real视觉策略 |
| **Isaac Gym** | GPU并行万级环境 | 大规模RL训练 |
| **SAPIEN** | PartNet铰接物体 | 操作任务 |
| **Genesis** | 比Isaac Gym快10-80x | 超大规模并行 |

## 实时控制环路 (ROS2)

```
相机→观测节点→/observations Topic→策略推理(GPU)→/actions Topic→IK/PID(实时线程)→电机
```

延迟预算：相机5-10ms + 策略5-50ms + IK<1ms + PID<0.1ms = **20-70ms总延迟**。

> 当总延迟>控制周期时，需要Action Chunking补偿。

## 部署工具链

PyTorch训练→ONNX→TensorRT量化→C++ ROS2节点→EtherCAT/CAN→电机

Python做策略推理+C++做底层实时控制。

## 数据格式

HDF5(本地大文件) vs Zarr(分布式/云存储)。LeRobot基于Zarr。

[[NVIDIA-Sim-to-Real技术]] | [[正逆运动学与PID控制]] | [[人形机器人开源框架]]
