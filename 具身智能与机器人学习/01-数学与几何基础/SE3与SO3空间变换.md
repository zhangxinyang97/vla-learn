---
title: SE3与SO3空间变换
tags: [3d-geometry, robotics, math-foundation, camera-calibration]
aliases: [3D刚体变换, 四元数, 齐次变换]
---

# SE(3) 与 SO(3) 空间变换

## SO(3)：旋转群

$$\text{SO}(3) = \{\mathbf{R} \in \mathbb{R}^{3\times3} \mid \mathbf{R}^T\mathbf{R}=\mathbf{I}, \det(\mathbf{R})=1\}$$

3自由度纯旋转。

## SE(3)：旋转+平移

$$\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \in \text{SE}(3)$$

6自由度（3旋转+3平移）。链式组合：$\mathbf{T}_{\text{EE}}^{\text{World}} = \mathbf{T}_{\text{Base}}^{\text{World}} \mathbf{T}_{\text{EE}}^{\text{Base}}$

## 四种旋转表示

| 表示 | 优点 | 缺点 |
|---|---|---|
| 欧拉角 (3D) | 直观 | **万向节死锁** |
| 旋转矩阵 (9D) | 无歧义 | 冗余 |
| 四元数 (4D) | 紧凑、无死锁、SLERP平滑 | 不直观 |
| **6D连续表示** (6D) | 深度学习友好、连续无歧义 | 需额外正交化 |

**6D表示**（Diffusion Policy / RT-2采用）：取旋转矩阵前两列，Gram-Schmidt恢复完整 $\mathbf{R}$。

## 相机标定

针孔模型：$s\mathbf{p}_{\text{pixel}} = \mathbf{K}[\mathbf{R}\mid\mathbf{t}]\mathbf{P}_{\text{world}}$

手眼标定：$\mathbf{AX}=\mathbf{XB}$，Eye-in-Hand vs Eye-to-Hand。

## SE(3)在策略中的应用

Diffusion Policy动作：$\mathbf{a}_t = [\Delta x,\Delta y,\Delta z,\Delta r_x,\Delta r_y,\Delta r_z, g]$

[[正逆运动学与PID控制]] | [[动作空间表征总览]] | [[向量内积与外积]]
