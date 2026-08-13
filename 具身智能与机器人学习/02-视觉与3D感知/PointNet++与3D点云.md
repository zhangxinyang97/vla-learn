---
title: PointNet++与3D点云
tags: [pointnet, pointnet++, point-cloud, 3d-perception, dp3]
aliases: [点云处理, PointNet, PointNet++]
---

# PointNet++ 与 3D 点云

## 1. 点云的核心性质

点云是 $N\times D$ 的无序点集（$D$ 通常为 3：$x,y,z$），核心挑战：

| 性质 | 含义 | 设计要求 |
|------|------|---------|
| **无序性 (Unordered)** | 点的排列顺序不影响物体语义 | 对称函数：$f(x_1,...,x_n)=f(x_{\pi(1)},...,x_{\pi(n)})$ |
| **置换不变性 (Permutation Invariant)** | 任何排列输入应得到相同输出 | 不可使用 RNN/LSTM；卷积需特殊处理 |
| **稀疏性 (Sparse)** | 大部分空间为空（不同于密集像素） | 高效的非空区域计算 |
| **不规则采样** | 近处点密、远处点疏 | 局部邻域定义需自适应 |

与 2D 图像对比：图像是规则网格上的密集信号 → 标准 CNN 天然适用；点云是非结构化数据 → 需专门架构。

## 2. PointNet (CVPR 2017)

核心思想：用**对称函数**保证置换不变性。

### 2.1 架构

$$
f(\{x_1, \ldots, x_n\}) \approx g\left( h(x_1), \ldots, h(x_n) \right)
$$

其中 $h$ 为逐点 MLP（共享权重），$g$ 为对称函数（max pooling）。

- 每个点经共享 MLP 升维到 1024D
- **Max Pooling** 聚合全局特征 → 1024D 全局描述子
- 全局特征 + 逐点特征拼接 → 分类/分割

### 2.2 对称函数选择

Max Pooling 是唯一保证以下所有性质的对称函数：
- 排列不变
- 可微（支持反向传播）
- 信息保留（关键点集理论证明）

### 2.3 T-Net（变换网络）

点云缺乏旋转/平移不变性 → T-Net 学习 $3\times3$ 仿射变换矩阵：
- **Input Transform**：对齐输入点云
- **Feature Transform**：对齐中间特征（64×64 矩阵，加正则化 $\|I-AA^T\|_F^2$ 约束正交性）

### 2.4 PointNet 局限

| 局限 | 原因 | 后果 |
|------|------|------|
| 无局部结构 | 仅全局 max pooling，无层次化 | 无法捕获细粒度局部几何 |
| 无上下文比例 | 逐点 MLP 感受野固定 | 对尺度变化敏感 |
| 平移依赖 | 全局特征丢失空间分布 | 大场景分割困难 |

## 3. PointNet++ (NeurIPS 2017)

核心创新：**层次化局部特征学习** —— 递归地将点集划分为重叠的局部区域，逐层提取局部特征。

### 3.1 层次结构

```
输入点云 (N, 3)
  ↓
Set Abstraction L1: N1 × (d+C1)   ← 局部区域 + PointNet
  ↓
Set Abstraction L2: N2 × (d+C2)
  ↓
Set Abstraction L3: N3 × (d+C3)
  ↓
全局特征 / 逐点上采样
```

### 3.2 Set Abstraction（SA）模块

每个 SA 层由三个子步骤组成：

**Step 1 — Sampling (FPS, 最远点采样)**:
从 $N$ 个点中选 $N'$ 个中心点，使彼此尽可能远，确保均匀覆盖：

```
输入: points (N, 3), 目标数量 N'
初始化: centroids = [随机选1个点]
for i in 1..N':
    对每个剩余点，计算到已有 centroids 集的最小距离
    选距离最大的点加入 centroids
返回: centroids (N', 3)
```

FPS 复杂度 $O(N'N)$，是 SA 模块的计算瓶颈。

**Step 2 — Grouping (Ball Query, 球查询)**:
以每个中心点为圆心，半径 $r$ 内采样最多 $K$ 个邻居点：

$$
\mathcal{N}(c_i) = \{ x_j \mid \|x_j - c_i\|_2 \leq r \}
$$

- $r$ 控制感受野大小（对尺度敏感）
- $K$ 限制最大邻居数（固定输出尺寸）
- 若球内点数不足 $K$，复制最近点填充

**Step 3 — PointNet Layer**:
每个局部区域独立送入轻量 PointNet（共享 MLP → Max Pooling），输出该区域的局部特征向量。

### 3.3 多尺度分组

解决点云密度不均匀问题：

**MSG (Multi-Scale Grouping)**：
同一 SA 层使用多个 $(r, K)$ 组合，提取不同尺度的局部特征后拼接。每个 $(r_1, K_1), (r_2, K_2)$ 各自 Ball Query → PointNet → 拼接输出。

**MRG (Multi-Resolution Grouping)**：
融合当前层和上一层的特征 —— 对密度鲁棒、计算效率更高。

### 3.4 分割头 (Segmentation Head)

编码器-解码器结构，上采样通过**距离加权插值 + 跳跃连接**：

$$
f^{(j)}(x) = \frac{\sum_{i=1}^k w_i(x) f_i^{(j)}}{\sum_{i=1}^k w_i(x)}, \quad w_i(x) = \frac{1}{d(x, x_i)^p}
$$

- 对每个待插值点，找 $k=3$ 个最近邻中心点
- 距离倒数加权，$p=2$
- 插值特征与编码器对应层特征拼接 → Unit PointNet 精炼

### 3.5 分类头 (Classification Head)

多层 SA 后得到全局点集（如 1 个点），直接接全连接分类。

## 4. PyTorch 伪代码：Set Abstraction

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def farthest_point_sample(xyz, npoint):
    """
    xyz: (B, N, 3)  点云坐标
    npoint: 采样点数
    返回: (B, npoint) 采样点的索引
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].unsqueeze(1)   # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)             # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]                   # (B,)
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    radius: 球半径
    nsample: 最大邻居数 K
    xyz: (B, N, 3) 所有点
    new_xyz: (B, S, 3) 中心点（S 个）
    返回: (B, S, nsample) 邻居索引
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=xyz.device)
    group_idx = group_idx.view(1, 1, N).repeat(B, S, 1)

    # 计算中心点到所有点的距离
    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)

    # 球外点标记为 N (无效索引)
    group_idx[sqrdists > radius ** 2] = N

    # 按距离排序，取前 nsample 个
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # 替换无效索引为组内第一个有效点
    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


class SetAbstraction(nn.Module):
    """PointNet++ Set Abstraction 模块"""
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        """
        npoint: FPS 采样点数
        radius: Ball Query 半径
        nsample: Ball Query 最大邻居数
        in_channel: 输入特征维度 (如 3 或 3+C)
        mlp: MLP 各层输出维度列表，如 [64, 64, 128]
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        # 构建 MLP (Conv1d)
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # xyz 相对坐标拼接
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: (B, N, 3) 坐标
        points: (B, N, C) 特征（可为 None，首次用坐标代替）
        返回:
          new_xyz: (B, npoint, 3)
          new_points: (B, npoint, \sum mlp)
        """
        B, N, C = xyz.shape

        # Step 1: FPS 采样中心点
        fps_idx = farthest_point_sample(xyz, self.npoint)            # (B, npoint)
        new_xyz = xyz[torch.arange(B).unsqueeze(-1), fps_idx]        # (B, npoint, 3)

        # Step 2: Ball Query 分组
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # (B, npoint, nsample)
        grouped_xyz = xyz[torch.arange(B).unsqueeze(-1).unsqueeze(-1), idx]  # (B, npoint, nsample, 3)

        # 相对坐标
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)        # 归一化到中心点

        if points is not None:
            grouped_points = points[torch.arange(B).unsqueeze(-1).unsqueeze(-1), idx]
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        # (B, npoint, nsample, 3+C) → (B, 3+C, npoint, nsample) for Conv2d
        new_points = new_points.permute(0, 3, 1, 2)

        # Step 3: PointNet (MLP → Max Pooling)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=-1)[0]                # (B, C_out, npoint)
        new_points = new_points.permute(0, 2, 1)                     # (B, npoint, C_out)

        return new_xyz, new_points
```

## 5. 方法对比

| 方法 | 年份 | 核心机制 | 置换不变 | 局部结构 | 感受野 | 计算效率 | 典型场景 |
|------|------|---------|---------|---------|--------|---------|---------|
| **PointNet** | 2017 | 全局 Max Pooling | ✅ | ❌ | 全局 | 高 | 分类 |
| **PointNet++** | 2017 | 层次化 SA + FPS | ✅ | ✅ 多尺度 | 分层扩展 | 中 | 分类/分割/检测 |
| **Point Transformer** | 2021 | Self-Attention + 向量位置编码 | ✅ | ✅ 全局+局部 | 自适应 | 中-低 | 分割/检测(SOTA) |
| **3D CNN (Voxel)** | — | 3D 卷积 | ✅ grid dependent | ✅ | 固定核 | 低 ($O(r^3)$) | 均匀体素输入 |
| **SparseConvNet** | 2018 | 稀疏 3D 卷积 | ✅ | ✅ | 固定核 | 中 | 大场景检测 |

### 选型指南

- **PointNet++**：点云处理**通用基线**，3D 感知任务首选，DP3 视觉骨干
- **Point Transformer**：精度 SOTA，但计算量大，适合高精度分割
- **3D CNN**：仅当输入已体素化时考虑

## 6. PointNet++ 在 DP3 中的角色

DP3 (3D Diffusion Policy) 选择 PointNet++ 作为视觉编码器的原因：

### 6.1 DP3 视觉流水线

```
RGB-D 图像
  ↓ 反投影
3D 点云 (N×6: xyz + rgb)
  ↓ PointNet++ 骨干
逐点特征 (N×C)
  ↓ 池化/投影
全局特征 + 局部特征
  ↓ Diffusion Head
动作序列
```

### 6.2 为什么用 PointNet++

| 需求 | PointNet++ 如何满足 |
|------|-------------------|
| **SE(3) 等变性** | PointNet++ 处理点云坐标 $(x,y,z)$，通过相对坐标归一化（平移不变）和逐点 MLP（旋转可泛化），配合坐标系选择实现 SE(3) 等变 |
| **点云原生** | 直接消费反投影 3D 点云，无需体素化（省内存、保精度） |
| **层次化特征** | 多尺度 SA 同时捕获局部几何（抓取点）和全局语义（物体类别） |
| **轻量高效** | 比 Point Transformer 快 3-5×，适合实时机器人控制 |
| **置换不变** | 点云来自深度图的随机采样，PointNet++ 天然鲁棒 |

### 6.3 SE(3) 等变性

DP3 输出 SE(3) 空间的动作序列（末端位姿）。SE(3) 等变性要求：

- **平移等变**：物体平移 $\Delta p$ → 预测动作平移 $\Delta p$（PointNet++ 天然满足，使用相对坐标）
- **旋转等变**：物体旋转 $R$ → 预测动作旋转 $R$（需规范坐标系或数据增强）

PointNet++ 通过处理点云相对坐标（`grouped_xyz_norm = grouped_xyz - centroid`）天然获得**平移不变的特征**，避免了绝对位置的过拟合。旋转方面，DP3 在训练时大量使用 SO(3) 数据增强，使 PointNet++ 学会从点云结构推断 SE(3) 等变的特征表示。

```
PointNet++ 视觉编码器
    ↓
  SE(3)-等变特征
    ↓
 Diffusion 去噪 → SE(3) 动作轨迹
```

## 7. 关键概念速查

| 概念 | 公式/含义 |
|------|----------|
| **对称函数** | $f(x_1,...,x_n)=f(x_{\pi(1)},...,x_{\pi(n)})$ |
| **Max Pooling** | $\max\{h(x_1),...,h(x_n)\}$ — PointNet 核心 |
| **FPS** | 迭代选最远点，复杂度 $O(N'N)$ |
| **Ball Query** | 球半径 $r$，最大 $K$ 邻居 |
| **MSG** | 多 $(r,K)$ 组合并行 → 拼接 |
| **MRG** | 当前层 + 上层特征加权融合 |
| **插值上采样** | $f(x)=\frac{\sum w_i f_i}{\sum w_i},\ w_i=1/d(x,x_i)^2$ |

## See Also

- [[DP3-3D扩散策略]] — PointNet++ 是 DP3 的视觉骨干
- [[3D-Gaussian-Splatting]] — 另一种 3D 表示（显式辐射场）
- [[DINOv2与自监督视觉特征]] — 2D 视觉预训练方法对比
- [[SE3与SO3空间变换]] — 刚体变换群的基础数学
