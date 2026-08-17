---
title: SE3与SO3空间变换
tags: [3d-geometry, robotics, math-foundation, camera-calibration, quaternion, lie-algebra]
aliases: [3D刚体变换, 四元数, 齐次变换, SE3, SO3]
created: 2026-08-13
---

# SE(3) 与 SO(3) 空间变换

> **定位**：具身智能的**位姿语言**。机器人本体、末端、相机、物体、世界——所有坐标系之间的转换都用 SE(3) 描述。DP3 的等变性、动作空间的参考系、手眼标定、运动学，全依赖这篇。
>
> 前置：[[向量内积与外积]]。关联：[[相机模型与手眼标定]] [[正逆运动学与PID控制]] [[动作空间表征总览]]。

## 一、SO(3)：旋转群

$$\text{SO}(3) = \{\mathbf{R} \in \mathbb{R}^{3\times3} \mid \mathbf{R}^T\mathbf{R}=\mathbf{I},\ \det(\mathbf{R})=1\}$$

- 3 自由度纯旋转
- $\mathbf{R}^T\mathbf{R}=\mathbf{I}$ 表示正交（保长保角），$\det=1$ 排除镜像（纯旋转而非反射）

**旋转矩阵的两个作用**：① 旋转向量：$\mathbf{v}'=\mathbf{R}\mathbf{v}$；② 表示坐标系姿态（列向量是新坐标轴在世界系的方向）。

## 二、四种旋转表示（怎么选）

| 表示 | 维度 | 优点 | 缺点 |
|------|------|------|------|
| 欧拉角 | 3 | 直观 | **万向节死锁** |
| 旋转矩阵 | 9 | 无歧义、直接运算 | 冗余、需正交约束 |
| **四元数** | 4 | 紧凑、无死锁、可 SLERP 平滑插值 | 不直观、双覆盖（$q$ 与 $-q$ 同旋转） |
| **6D 连续表示** | 6 | 深度学习友好、连续无歧义 | 需额外正交化恢复 |

### 万向节死锁（为什么欧拉角会出问题）

三轴转动的中间轴转到 ±90° 时，第一轴与第三轴**重合**，丢一个自由度——某些姿态无法唯一表示，插值会"抽搐"。四元数用 4 维表示 3 维旋转，绕开了这个奇点。

## 三、四元数基础

$$\mathbf{q} = (w,\ x,\ y,\ z) = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$$

- 旋转 $\theta$ 角绕单位轴 $\mathbf{u}=(u_x,u_y,u_z)$：

$$\mathbf{q} = \left(\cos\frac{\theta}{2},\ u_x\sin\frac{\theta}{2},\ u_y\sin\frac{\theta}{2},\ u_z\sin\frac{\theta}{2}\right)$$

- 旋转向量 $\mathbf{v}$（写成纯四元数）用共轭作用：$\mathbf{v}' = \mathbf{q}\mathbf{v}\mathbf{q}^{-1}$
- **SLERP（球面线性插值）**：在两姿态间沿单位球面最短路径平滑插值，机器人轨迹平滑、动画姿态过渡都用它

## 四、李代数 so(3)：角速度向量（直觉）

旋转矩阵的变化率 $\dot{\mathbf{R}} = [\boldsymbol{\omega}]_\times \mathbf{R}$，其中 $\boldsymbol{\omega}$ 是**角速度向量**（方向=旋转轴，模长=转速），$[\boldsymbol{\omega}]_\times$ 是由外积构造的反对称矩阵：

$$[\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix}$$

**直觉**：so(3) 是 SO(3) 在单位元处的切空间——"小旋转"由角速度向量描述（这正连回 [[向量内积与外积]] 的外积）。指数映射 $\exp([\boldsymbol{\omega}]_\times)$ 把角速度"积分"成旋转矩阵（罗德里格斯公式）。

## 五、SE(3)：旋转 + 平移

$$\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \in \text{SE}(3)$$

- 6 自由度（3 旋转 + 3 平移）
- 齐次坐标作用：把 3D 点 $\mathbf{p}$ 变到新系：$\begin{bmatrix}\mathbf{R}&\mathbf{t}\\0&1\end{bmatrix}\begin{bmatrix}\mathbf{p}\\1\end{bmatrix} = \begin{bmatrix}\mathbf{R}\mathbf{p}+\mathbf{t}\\1\end{bmatrix}$

**链式组合**（坐标系层层嵌套的核心）：

$$\mathbf{T}_{\text{camera}}^{\text{world}} = \mathbf{T}_{\text{base}}^{\text{world}}\,\mathbf{T}_{\text{end}}^{\text{base}}\,\mathbf{T}_{\text{camera}}^{\text{end}}$$

（读作：从相机到末端，到基座，到世界——逐级复合，见 [[相机模型与手眼标定]] 的手眼标定 AX=XB。）

## 六、复合变换例题

机器人基座在世界原点，末端在基座前方 0.5m（$\mathbf{t}=(0.5,0,0)$），末端绕 z 轴转 90°：

- $\mathbf{T}_{\text{end}}^{\text{base}} = \begin{bmatrix}\cos90^\circ&-\sin90^\circ&0&0.5\\ \sin90^\circ&\cos90^\circ&0&0\\ 0&0&1&0\\0&0&0&1\end{bmatrix}$
- 末端上一点 $\mathbf{p}_{\text{end}}=(0.1,0,0)$ 在基座系为 $\mathbf{p}_{\text{base}} = \mathbf{R}\mathbf{p}_{\text{end}}+\mathbf{t} = (0,0.1,0)+(0.5,0,0)=(0.5,0.1,0)$

（验证：末端局部 x 轴绕 z 转 90° 后指向基座 y 方向，点 $(0.1,0,0)$ 变到 $(0.5,0.1,0)$。✅）

## 七、6D 连续表示（深度学习用）

Diffusion Policy / RT-2 采用：取旋转矩阵**前两列**（6 个数），因为第三列可由前两列外积得到，再用 Gram-Schmidt 正交化恢复完整 $\mathbf{R}$。好处：6 个连续实数、无死锁、可直接喂神经网络回归。

## 八、SE(3) 在策略里的应用速查

| 场景 | 怎么用 SE(3) | 关联 |
|------|-------------|------|
| 动作表征 | 末端位姿 = 位置 + 旋转（EEF 6D/7D 增量） | [[动作空间表征总览]] |
| DP3 等变性 | 点云+动作同步旋转 → 视角不变 | [[DP3-3D扩散策略]] |
| 手眼标定 | 相机↔基座关系 AX=XB | [[相机模型与手眼标定]] |
| 运动学 | FK/IK 的链式变换 | [[正逆运动学与PID控制]] |

## 检查点

1. ✅ 说清四种旋转表示各自的优劣，解释万向节死锁
2. ✅ 解释 so(3) 角速度向量与旋转矩阵的关系（外积）
3. ✅ 做一个两层坐标系的复合变换例题

## 前置 / 关联

前置：[[向量内积与外积]]
关联：[[相机模型与手眼标定]] | [[正逆运动学与PID控制]] | [[动作空间表征总览]] | [[DP3-3D扩散策略]] | [[机器人动力学基础]]
