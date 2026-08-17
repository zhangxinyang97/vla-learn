---
title: 轨迹规划-RRT与样条插值
tags: [trajectory-planning, rrt, spline, motion-planning, robotics]
aliases: [RRT, RRT*, 样条插值]
---

# 轨迹规划：RRT 与样条插值

策略输出"目标动作"→**轨迹规划**生成连续路径→IK求解→PID执行。轨迹规划是策略与控制之间的桥梁：将离散的动作指令转化为光滑、可执行的关节/末端轨迹。

## 问题定义

给定起始位姿 $\mathbf{x}_{\text{start}} \in \mathcal{C}$ 和目标位姿 $\mathbf{x}_{\text{goal}} \in \mathcal{C}$（$\mathcal{C}$ 为配置空间），规划一条无碰撞的连续路径：

$$\tau: [0, T] \to \mathcal{C}_{\text{free}},\quad \tau(0)=\mathbf{x}_{\text{start}},\;\tau(T)=\mathbf{x}_{\text{goal}}$$

| 概念 | 含义 |
|---|---|
| 配置空间 (C-space) | 机器人所有可能关节角/位姿的集合 |
| $\mathcal{C}_{\text{free}}$ | 无碰撞区域 |
| $\mathcal{C}_{\text{obs}}$ | 障碍物区域 |
| 路径 (path) | 几何曲线，不考虑时间 |
| 轨迹 (trajectory) | 路径 + 时间参数化，含速度和加速度 |

## RRT：快速探索随机树

**核心思想**：在 C-space 中随机采样，逐步生长一棵树，直到连接到目标。

### 算法流程

```
1. 初始化树 T，根节点 = x_start
2. repeat K 次:
   a. x_rand ← RandomSample()           # 在 C-space 随机采样
   b. x_near ← NearestNeighbor(T, x_rand) # 找树上最近节点
   c. x_new ← Steer(x_near, x_rand, η)   # 朝 x_rand 步进 η
   d. if CollisionFree(x_near, x_new):
        T.add_node(x_new)
        T.add_edge(x_near, x_new)
        if dist(x_new, x_goal) < ε:       # 到达目标
            return ExtractPath(x_new)
3. return FAILURE
```

### 关键操作

**RandomSample**：均匀随机采样 vs 目标偏置采样（以概率 $p_{\text{goal}}$ 直接采样目标点）。

**NearestNeighbor**：欧氏距离或加权距离。工程上用 KD-Tree 加速（$O(\log n)$）。

**Steer**：朝向控制，步长 $\eta$ 限制每次扩展距离：

$$\mathbf{x}_{\text{new}} = \mathbf{x}_{\text{near}} + \eta \cdot \frac{\mathbf{x}_{\text{rand}} - \mathbf{x}_{\text{near}}}{\|\mathbf{x}_{\text{rand}} - \mathbf{x}_{\text{near}}\|}$$

**CollisionFree**：检查路径是否与障碍物相交。对机械臂需做正向运动学逐点验证。

### RRT 伪代码

```python
def rrt(x_start, x_goal, max_iter=5000, step_size=0.1, goal_bias=0.05):
    tree = {x_start: None}  # node -> parent

    for _ in range(max_iter):
        if random() < goal_bias:
            x_rand = x_goal
        else:
            x_rand = random_config(c_space)

        x_near = nearest_neighbor(tree, x_rand)  # KD-Tree O(log n)
        x_new = steer(x_near, x_rand, step_size)

        if collision_free(x_near, x_new, obstacles):
            tree[x_new] = x_near
            if distance(x_new, x_goal) < step_size:
                tree[x_goal] = x_new
                return extract_path(tree, x_goal)

    return None  # no path found
```

### RRT 特性

| 特性 | 说明 |
|---|---|
| 概率完备 | 只要存在可行路径，采样次数足够必定找到 |
| 非最优 | 路径曲折，不保证最短 |
| 高维友好 | 不像 A* 需要网格离散化，适合 6-DOF+ 机械臂 |
| Voronoi 偏置 | 树天然向未探索区域生长（大 Voronoi 区域优先） |

## RRT*：渐进最优

**改进**：在 RRT 基础上加入 **Rewire（重连）** 操作，随着采样增加，路径收敛到最优解。

```
Steer + CollisionFree 同 RRT，额外步骤：

e. x_near_nodes ← NearNeighbors(T, x_new, radius)  # 半径内所有邻居
f. x_min ← argmin_{x∈x_near_nodes}{cost(x) + dist(x, x_new)}  # 找最优父节点
g. T.add_edge(x_min, x_new)

h. foreach x ∈ x_near_nodes:            # Rewire: 检查新节点能否优化邻居
       if cost(x_new) + dist(x_new, x) < cost(x):
           T.remove_edge(x.parent, x)
           T.add_edge(x_new, x)
```

**Rewire 半径**：$r = \gamma\left(\frac{\log n}{n}\right)^{1/d}$，其中 $n$ 为节点数，$d$ 为维度。随节点增多半径缩小，保证渐进最优但不牺牲太多效率。

| RRT vs RRT* | RRT | RRT* |
|---|---|---|
| 完备性 | 概率完备 | 概率完备 |
| 最优性 | 无 | **渐进最优** |
| 速度 | 快 | 较慢（每步 Rewire） |
| 适用 | 实时在线 | 离线一次性规划 |

## 样条插值：从路径到光滑轨迹

RRT/RRT* 输出的是折线段路径。需要通过样条插值转化为光滑、连续可导的轨迹，才能输入控制器。

### 三次样条 (Cubic Spline)

每段为三次多项式，保证 **位置连续** 和 **速度连续**：

$$s_i(t) = a_i + b_i(t-t_i) + c_i(t-t_i)^2 + d_i(t-t_i)^3,\quad t \in [t_i, t_{i+1}]$$

**约束条件**：
- $s_i(t_{i+1}) = s_{i+1}(t_{i+1})$ — 位置连续（$C^0$）
- $\dot{s}_i(t_{i+1}) = \dot{s}_{i+1}(t_{i+1})$ — 速度连续（$C^1$）
- 边界条件：自然边界（$\ddot{s}_0(0)=\ddot{s}_n(T)=0$）或夹持边界（指定首末速度）

### 五次样条 (Quintic Spline)

每段五次多项式，额外保证 **加速度连续**（$C^2$），减少 jerk（加加速度）冲击：

$$s_i(t) = a_i + b_i t + c_i t^2 + d_i t^3 + e_i t^4 + f_i t^5$$

$$s, \dot{s}, \ddot{s}\text{ 三阶连续} \;\Rightarrow\; \text{电机力矩平滑，减少振动}$$

### 三次样条插值 Python 实现

```python
import numpy as np
from scipy.interpolate import CubicSpline

def cubic_spline_trajectory(waypoints, total_time=5.0, dt=0.01):
    """
    waypoints: (N, dim) — RRT输出的路径点
    total_time: 总时间
    dt: 控制周期
    返回: (T, dim) 位置, (T, dim) 速度
    """
    N = len(waypoints)
    t_knots = np.linspace(0, total_time, N)  # 等距时间节点

    # 对每个维度独立插值
    dim = waypoints.shape[1]
    pos_splines = [CubicSpline(t_knots, waypoints[:, d],
                                bc_type='natural') for d in range(dim)]

    t = np.arange(0, total_time, dt)
    pos = np.column_stack([sp(t) for sp in pos_splines])
    vel = np.column_stack([sp(t, 1) for sp in pos_splines])  # 1阶导数

    return pos, vel
```

### 关节空间 vs 笛卡尔空间

| 规划空间 | 优点 | 缺点 |
|---|---|---|
| 关节空间 | 无奇异点；避关节限位直接 | 末端轨迹不可预测 |
| 笛卡尔空间 | 末端路径直观可控 | 需实时 IK；可能遇奇异点 |

**常见做法**：笛卡尔空间 RRT 规划末端路径 → IK 映射到关节空间 → 关节空间样条平滑。

## 梯形速度剖面

将路径时间参数化，使运动在约束内高效执行：

```
速度
 ^        ┌──────────┐
 |       /│ 匀速段    │\
 |      / │          │ \
 |     /  │          │  \
 |    /   │          │   \
 |   /    │          │    \
 |  / 加速│          │减速 \
 +-+------+----------+------+--> t
   t0     t1         t2    t3
```

**三段式**：
1. **加速段** $[t_0, t_1]$：以最大加速度 $a_{\max}$ 加速至 $v_{\max}$
2. **匀速段** $[t_1, t_2]$：以 $v_{\max}$ 巡航
3. **减速段** $[t_2, t_3]$：以 $-a_{\max}$ 减速至 0

$$v(t) =
\begin{cases}
a_{\max} t, & t \in [0, t_1]\\
v_{\max}, & t \in [t_1, t_2]\\
v_{\max} - a_{\max}(t - t_2), & t \in [t_2, t_3]
\end{cases}$$

短路程场景退化：若无法达到 $v_{\max}$，则变为三角剖面（纯加速+减速）。

## 策略→轨迹→控制全流程

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  策略输出     │────▶│  轨迹规划        │────▶│  IK 求解      │────▶│  PID 控制    │
│  (10-50 Hz)  │     │  (RRT + 样条)    │     │  (100-500 Hz) │     │  (1-10 kHz)  │
└──────────────┘     └─────────────────┘     └──────────────┘     └──────────────┘
       │                      │                      │                     │
  动作chunk:             连续轨迹:             关节角序列:           力矩/电流指令:
  Δx, Δθ, 目标位姿      x(t)光滑可导          θ₁..θ₆(t)           τ₁..τ₆,  PWM
```

**详细流程**：
1. **策略** 输出动作 chunk（如 ACT 的 100 步目标位姿序列 / Diffusion Policy 的末端位移）
2. **轨迹规划** 将离散航点插值为光滑轨迹（样条），同时验证无碰撞（RRT 生成参考路径）
3. **IK** 在每个控制周期将末端位姿转为关节角
4. **PID** 以高频闭环跟踪目标关节角，输出电机电流/力矩指令

**频率层次**：策略低频推理 + 规划中频生成轨迹 + IK/PID 高频闭环。高频层不关心语义，只负责跟踪。

## MoveIt / OMPL 概览

**OMPL (Open Motion Planning Library)**：C++ 运动规划算法库，提供 RRT、RRT*、PRM、EST 等多种规划器。不包含碰撞检测，需外部提供。

**MoveIt**：ROS 生态的机械臂运动规划框架，集成 OMPL + FCL 碰撞检测 + IK 求解器。

```
MoveIt 架构：
  MoveGroup (Python/C++ API)
    ├── 运动规划 → OMPL (RRTconnect / RRT* / PRM)
    ├── 碰撞检测 → FCL (Flexible Collision Library)
    ├── IK 求解   → KDL / TRAC-IK / BioIK
    └── 轨迹执行   → FollowJointTrajectory Action
```

常用 OMPL 规划器：
- **RRTConnect**：从起点终点同时生长两棵树，速度快，工业首选
- **RRT***：渐进最优，适合离线一次性规划
- **PRM (Probabilistic Roadmap)**：预建路图 + 多次查询，适合静态环境多任务

## SOTA：学习式规划

传统规划在高维空间和复杂环境中效率骤降，学习式方法成为前沿方向。

### MPNet (Motion Planning Networks)

用神经网络直接学习"采样→路径"的映射，替代 RRT 的随机采样过程：

- **Encoder**：将点云环境编码为隐空间特征
- ** Planner**：给定起止点 + 环境特征，生成路径点
- 训练数据：RRT* 生成的专家路径

优势：推理速度远快于 RRT（前向传播 vs 上万次采样+碰撞检测），且隐式学习环境结构。

### Neural Motion Planning

- **Neural RRT***：用 GNN 预测 RRT* 的 Rewire 最优半径，加速收敛
- **扩散策略用于规划**：Diffusion Policy 本质也是一种"规划"——从噪声逐步去噪生成平滑动作轨迹，天然具备多模态和光滑性
- **STORM**：结合 Transformer 和流模型的运动规划
- **学习式碰撞检测**：用 Occupancy Network / NeRF 隐式表示场景，替代显式 mesh 碰撞

**趋势**：End-to-End 感知→规划一体化（如 RT-2、Octo），但当前工业部署仍以经典规划为主流，学习式方法作为加速器或难场景兜底。

> 拓展阅读：NVIDIA cuMotion (基于 GPU 并行采样的 RRT)、轨迹优化 (TrajOpt / CHOMP)、动力学约束规划 (Kinodynamic RRT)。

[[正逆运动学与PID控制]] | [[动作空间表征总览]] | [[仿真引擎与ROS2部署]]
