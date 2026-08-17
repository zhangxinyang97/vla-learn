---
title: DP3-3D扩散策略
tags: [dp3, diffusion-policy, pointnet++, 3d-vision, se3-equivariance]
aliases: [3D Diffusion Policy, 三维扩散策略]
---

# DP3：3D Diffusion Policy

3D Vision + Diffusion Policy = SE(3) 等变动作生成。用 PointNet++ 点云编码替换 2D CNN，天然获得对相机视角和刚体变换的不变性。

## 1. 动机：为什么 2D RGB 不够？

### 2D 视觉的根本局限

```
场景示意：
          真实世界物体                相机A看到的        相机B看到的
         ┌──────────┐              ┌──────┐           ┌──────┐
         │    🏐     │              │  ○   │           │      │
         │   球体    │              │  ○   │           │  ○   │
         └──────────┘              └──────┘           └──────┘
         物理上同一物体             俯视→圆            侧视→仍是圆
                                  但缺少深度信息！
```

| 问题 | 2D RGB | 3D 点云 |
|------|--------|---------|
| **视角依赖性** | 换个角度输入完全不同 → 策略不稳定 | 点云在 SE(3) 变换下可等变处理 |
| **几何信息丢失** | 缺乏深度、遮挡推理 | 显式编码空间位置 (x,y,z) |
| **Sim-to-Real 泛化** | 仿真/真实 RGB 域差距大 | 点云更接近物理世界表示 |
| **多视角融合** | 需多目相机 + 复杂融合 | 单帧点云天然多视角 |

### DP3 的核心洞察

> 标准 Diffusion Policy：**RGB 图像 → 2D CNN → 条件特征 → 去噪**    
> DP3 替换为：**深度/点云 → PointNet++ → 全局特征 → 去噪**  

视觉编码器从 ResNet/ViT 换成 PointNet++ 后，整个策略获得 **SE(3)-等变性**：对点云施加刚体变换 $T \in \text{SE}(3)$，输出的动作也以相同方式变换。

$$
\pi(a \mid T \cdot P) = T \cdot \pi(a \mid P)
$$

其中 $P \in \mathbb{R}^{N \times 3}$ 为点云（N个点的xyz坐标），$T \in \text{SE}(3)$ 为刚体变换（旋转+平移），$\pi$ 为策略。

## 2. 架构总览

### ASCII 架构图

```
                          ┌──────────────────────────────────────────┐
                          │              训练阶段                     │
                          └──────────────────────────────────────────┘
                                                        演示动作序列
    深度相机/RGBD                                              │
        │                                              a_0, a_1, ..., a_{H-1}
        ▼                                                     │
  ┌──────────┐                                          ┌─────▼─────┐
  │ 点云 P_t  │                                          │  加噪 k 步  │
  │ N×3 坐标 │                                          │ ~N(0,I)ε  │
  └────┬─────┘                                          └─────┬─────┘
       │                                                      │
       ▼                                                      ▼
┌──────────────┐                                     ┌──────────────────┐
│  PointNet++  │                                     │ x_k (噪声动作块)  │
│   Hierarchical│                                    │   H × D_action    │
│    编码器    │                                     └────────┬─────────┘
└──────┬───────┘                                              │
       │ 全局特征 f ∈ R^256                    噪声步 k ─────┤
       │  (SE(3)等变)                                      │
       ▼                                                    ▼
  ┌─────────┐          ┌──────────────────────────────────────────┐
  │ f       │─────────►│             1D U-Net 去噪器               │
  │ SE(3)   │  FiLM    │  ┌───┐    ┌───┐    ┌───┐    ┌───┐       │
  │ 条件    │  调制    │  │Res│→ds │Res│→ds │Res│→us │Res│→us    │
  └─────────┘          │  │ 1 │    │ 2 │    │ 3 │    │ 4 │       │
                       │  └───┘    └───┘    └───┘    └───┘       │
                       └──────────────────────────────────────────┘
                                                      │
                                                      ▼
                                              ┌──────────────────┐
                                              │  预测噪声 ε̂       │
                                              │  L = MSE(ε̂, ε)   │
                                              └──────────────────┘

                          ┌──────────────────────────────────────────┐
                          │              推理阶段 (DDIM)              │
                          └──────────────────────────────────────────┘

    点云 P_t ──► PointNet++ ──► f
                                  │
    噪声 x_T ~ N(0,I) ────────────┤
                                  ▼
                        ┌──────────────────┐
                        │  1D U-Net 去噪器  │ ← FiLM 调制 by f
                        └────────┬─────────┘
                                 │ × DDIM 10 步
                                 ▼
                        ┌──────────────────┐
                        │ 动作块 Â_t        │
                        │ H=16步 × D=7维   │
                        │ [Δx,Δy,Δz,      │
                        │  Δrx,Δry,Δrz,g] │
                        └────────┬─────────┘
                                 │ 执行前 N 步
                                 ▼
                           机器人执行
```

### 组件拆解

| 组件 | 功能 | 关键参数 |
|------|------|----------|
| **PointNet++ 编码器** | N×3 点云 → 全局特征向量 f∈R^256 | SA层数=4，每层半径=0.1/0.2/0.4/∞ |
| **FiLM 条件注入** | 将 f 注入 U-Net 各层（缩放γ + 偏移β） | $\gamma, \beta = \text{MLP}(f,t)$ |
| **1D U-Net 去噪器** | 噪声动作块 x_k → 去噪动作块 x_{k-1} | 3层下采样+3层上采样，Conv1D |
| **DDIM 采样器** | 加速推理，100步→10步 | η=0（确定性），10步 |

## 3. SE(3) 等变性：DP3 的核心创新

### 为什么 PointNet++ 天然提供 SE(3) 等变性？

PointNet++ 的核心操作是 **Set Abstraction (SA) 层**：

**SA 层计算流程：**

给定点集 $\{p_1, ..., p_N\}$，对每个中心点 $p_c$：

1. 球查询：找到半径 r 内的邻域点 $\{p_j \mid \|p_j - p_c\| < r\}$
2. 局部特征：$h_c = \text{MLP}\left(\{ (p_j - p_c) \}\right)$ — **相对坐标**
3. 最大池化：$h_c' = \max_{j} h_c$ — **对称函数**

$$
\text{SA}(\{p_i\}) = \text{MaxPool}\left(\text{MLP}\left(\{p_j - p_c\}_{j \in \mathcal{N}(p_c)}\right)\right)
$$

**关键性质：对 SE(3) 变换等变**

对任意旋转 $\mathbf{R} \in \text{SO}(3)$ 和平移 $\mathbf{t} \in \mathbb{R}^3$：

$$
\text{SA}(\{\mathbf{R}p_i + \mathbf{t}\}) = \text{SA}(\{p_i\})
$$

因为：
- 相对坐标：$(\mathbf{R}p_j + \mathbf{t}) - (\mathbf{R}p_c + \mathbf{t}) = \mathbf{R}(p_j - p_c)$
- 仅改变局部坐标系的朝向，但 MLP 只需学习对相对几何的模式
- 最大池化消除剩余的方向敏感性

**实际处理：** DP3 通过 **坐标系归一化** 进一步强化等变性：

1. 将点云平移到机器人基座坐标系
2. 对末端执行器位置做局部归一化
3. 动作输出也定义在 EE 局部坐标系

### 与标准 Diffusion Policy 的对比

```
标准 DP (2D CNN):
  相机旋转 30° → RGB 完全改变 → 策略输出完全不同 ❌

DP3 (PointNet++):
  点云旋转 30° → 相对坐标不变 → 策略输出在局部坐标系中不变 ✓
```

**Sim-to-Real 优势：** 仿真和真实世界的 RGB 域差异极大（纹理、光照），但点云的几何结构（只要深度准确）在 sim 和 real 之间高度一致。

## 4. 数学公式详解

### 4.1 正向扩散过程

给定演示动作块 $\mathbf{a}^0 = [a_0, a_1, ..., a_{H-1}] \in \mathbb{R}^{H \times D}$，逐步加噪：

$$
q(\mathbf{a}^k \mid \mathbf{a}^{k-1}) = \mathcal{N}\left(\mathbf{a}^k; \sqrt{1 - \beta_k} \,\mathbf{a}^{k-1}, \beta_k \mathbf{I}\right)
$$

- $\beta_k \in (0,1)$：第 k 步的噪声调度（余弦调度）
- $\mathbf{a}^k$：加噪 k 步后的动作块
- $k \in \{0, 1, ..., K\}$：扩散步，K=100（DDIM 推导时只用 10 步）

**一步到位公式：**

$$
\mathbf{a}^k = \sqrt{\bar{\alpha}_k} \,\mathbf{a}^0 + \sqrt{1 - \bar{\alpha}_k} \,\boldsymbol{\epsilon}
$$

- $\alpha_k = 1 - \beta_k$：信号保留率
- $\bar{\alpha}_k = \prod_{i=1}^k \alpha_i$：累积信号保留率
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$：从标准高斯采样的噪声

### 4.2 去噪网络（条件DDPM）

训练一个网络 $\epsilon_\theta$ 从噪声观测中预测所加的噪声：

$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{\mathbf{a}^0, k, \boldsymbol{\epsilon}}\left[\left\|\boldsymbol{\epsilon} - \epsilon_\theta(\mathbf{a}^k, k, \mathbf{f})\right\|^2\right]
$$

- $\epsilon_\theta$：去噪 U-Net，参数为 θ
- $\mathbf{a}^k$：加噪 k 步后的动作块（网络输入）
- $k$：扩散步索引（通过正弦位置编码注入）
- $\mathbf{f} = \text{PointNet++}(P)$：点云全局特征（条件，FiLM 注入）

**FiLM (Feature-wise Linear Modulation)：**

在第 l 层 U-Net 特征图上：

$$
\text{FiLM}(\mathbf{h}_l, \mathbf{f}, k) = \gamma_l \odot \mathbf{h}_l + \beta_l
$$

其中 $[\gamma_l, \beta_l] = \text{MLP}_l(\mathbf{f}, \text{SinPosEmb}(k))$

- $\gamma_l$：逐通道缩放因子
- $\beta_l$：逐通道偏移因子
- $\odot$：逐元素乘法

### 4.3 DDIM 采样（推理加速）

DDIM 跳步去噪，从纯噪声 $\mathbf{a}^K \sim \mathcal{N}(0,\mathbf{I})$ 开始：

$$
\mathbf{a}^{k-1} = \sqrt{\bar{\alpha}_{k-1}} \cdot \underbrace{\left(\frac{\mathbf{a}^k - \sqrt{1-\bar{\alpha}_k}\,\epsilon_\theta(\mathbf{a}^k, k, \mathbf{f})}{\sqrt{\bar{\alpha}_k}}\right)}_{\text{估计的干净数据 }\hat{\mathbf{a}}^0} + \sqrt{1-\bar{\alpha}_{k-1}}\,\epsilon_\theta(\mathbf{a}^k, k, \mathbf{f})
$$

- 采样步数：K=100（训练）→ 推理只用 T=10 步（均匀间隔子序列）
- 确定性采样（η=0）：给定初始噪声，输出固定
- 推理速度：约 20ms（RTX 3090）

### 4.4 动作表示

每个时间步的动作向量：

$$
a_t = [\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z, g] \in \mathbb{R}^7
$$

- $\Delta x, \Delta y, \Delta z$：末端执行器（EEF）位置增量（m）
- $\Delta r_x, \Delta r_y, \Delta r_z$：EEF 姿态增量，用 6D 连续旋转表示的前 3 维
- $g \in [0, 1]$：夹爪开合度（二值化：>0.5=闭合）
- 动作块长度：H=16 步 @ 10Hz = 1.6 秒

## 5. 训练流程

```
Algorithm: DP3 训练
────────────────────────────────────────────────────
输入: 演示数据集 D = {(P_i, a_i^0)}，其中
      P_i: 第 i 个时间步的点云 (N_points × 3)
      a_i^0: 干净的动作块 (H × 7)

超参数:
  - 扩散步数 K = 100
  - 学习率 η = 1e-4
  - Batch Size = 256
  - 点云点数 N = 1024 (FPS采样)
  - 动作块长度 H = 16

for epoch = 1 to num_epochs:
    for batch in D:
        P ← 点云批次 (B × N × 3)
        a0 ← 动作块批次 (B × H × 7)

        # Step 1: 随机采样扩散步
        k ← Uniform(1, K)  # (B,) 每个样本独立采样

        # Step 2: 加噪
        ε ← randn_like(a0)   # (B × H × 7)
        ak ← √ᾱₖ · a0 + √(1-ᾱₖ) · ε

        # Step 3: 编码点云 (SE(3)等变)
        f ← PointNet++(P)    # (B × 256)

        # Step 4: 预测噪声
        ε̂ ← UNet1D(ak, k, cond=f)  # FiLM调制

        # Step 5: MSE损失
        L ← MSE(ε̂, ε)
        L.backward()
        optimizer.step()
────────────────────────────────────────────────────
```

### 数据增强（强化等变性）

训练时对每批点云随机施加 SE(3) 变换：

```python
# 随机旋转（绕z轴 ±180°，任意轴 ±15°）
R = random_rotation_matrix(z_range=360, other_range=15)
# 随机平移（±5cm）
t = uniform(-0.05, 0.05, (3,))
# 变换点云
P_aug = (R @ P.T + t[:, None]).T  # (N × 3)
# 同步变换动作（仅在EE局部坐标中需调整参考系）
```

## 6. PyTorch 伪代码：DP3 Forward Pass

```python
import torch
import torch.nn as nn

class DP3(nn.Module):
    """3D Diffusion Policy: PointNet++ encoder + DDPM denoiser"""
    
    def __init__(self, action_dim=7, horizon=16, cond_dim=256):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        # SE(3)-等变视觉编码器
        self.pointnet = PointNetPlusPlusEncoder(
            in_channels=3,      # xyz坐标
            out_dim=cond_dim,   # 全局特征维度
            sa_layers_config=[  # Set Abstraction层配置
                (512, 0.1, 32),   # (npoints, radius, mlp_dims)
                (256, 0.2, 64),
                (128, 0.4, 128),
                (None, None, 256) # 全局层
            ]
        )
        
        # 1D U-Net去噪器 + FiLM条件注入
        self.denoiser = ConditionalUNet1D(
            input_dim=action_dim * horizon,  # 112 = 16×7
            cond_dim=cond_dim,               # 256
            dims=[256, 512, 1024],           # 各层通道
            kernel_size=5,
            n_groups=8
        )
        
        # 扩散调度
        self.betas = cosine_beta_schedule(timesteps=100)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def forward(self, x, t, pointcloud):
        """
        单步训练/推理forward
        
        Args:
            x: 噪声动作块 (B, horizon, action_dim) 或 (B, horizon*action_dim)
            t: 扩散步索引 (B,)
            pointcloud: 点云 (B, N_points, 3)
        
        Returns:
            预测的噪声 (B, horizon, action_dim)
        """
        B = x.shape[0]
        
        # 1. 点云编码 → SE(3)等变特征
        #    PointNet++: 相对坐标 + 最大池化 → 旋转不变
        point_feat = self.pointnet(pointcloud)  # (B, 256)
        
        # 2. 扩散步编码
        #    正弦位置编码 → 时间嵌入
        t_emb = sinusoidal_embedding(t, dim=128)  # (B, 128)
        
        # 3. 合并条件
        cond = torch.cat([point_feat, t_emb], dim=-1)  # (B, 384)
        
        # 4. 平坦化动作块
        x_flat = x.view(B, -1)  # (B, horizon*action_dim)
        
        # 5. 条件1D U-Net去噪
        noise_pred = self.denoiser(x_flat, cond)  # (B, horizon*action_dim)
        
        # 6. 恢复形状
        noise_pred = noise_pred.view(B, self.horizon, self.action_dim)
        
        return noise_pred


def train_step(model, batch, optimizer):
    """单个训练步"""
    pointcloud, actions = batch  # (B,N,3), (B,H,7)
    B = actions.shape[0]
    
    # 随机采样扩散步
    t = torch.randint(1, model.num_timesteps, (B,))
    
    # 加噪
    noise = torch.randn_like(actions)
    alpha_bar = model.alphas_cumprod[t]  # (B,)
    noisy_actions = (
        torch.sqrt(alpha_bar)[:, None, None] * actions +
        torch.sqrt(1 - alpha_bar)[:, None, None] * noise
    )
    
    # 预测噪声
    pred_noise = model(noisy_actions, t, pointcloud)
    
    # MSE损失
    loss = nn.functional.mse_loss(pred_noise, noise)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def ddim_sample(model, pointcloud, num_steps=10):
    """DDIM采样：从纯噪声生成动作块"""
    B = pointcloud.shape[0]
    device = pointcloud.device
    
    # 纯噪声起点
    x = torch.randn(B, model.horizon, model.action_dim, device=device)
    
    # DDIM跳步子序列 (均匀间隔)
    step_indices = torch.linspace(
        model.num_timesteps - 1, 0, num_steps, dtype=torch.long
    )
    
    for i, k in enumerate(step_indices):
        t = torch.full((B,), k, device=device)
        
        # 预测噪声
        eps = model(x, t, pointcloud)  # (B, H, 7)
        
        # DDIM更新（确定性采样, η=0）
        alpha_bar = model.alphas_cumprod[k]
        alpha_bar_prev = model.alphas_cumprod[step_indices[i+1]] \
                         if i < num_steps - 1 else torch.tensor(1.0)
        
        # 估计干净数据
        x0_pred = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        
        # 向前一步
        x = (
            torch.sqrt(alpha_bar_prev) * x0_pred +
            torch.sqrt(1 - alpha_bar_prev) * eps
        )
    
    return x  # 去噪的动作块 (B, H, 7)
```

## 7. 关键实验结果

### vs 标准 Diffusion Policy (仿真实验)

在 Robomimic / RLBench / ManiSkill 等基准上：

| 指标 | Diffusion Policy (2D RGB) | DP3 (3D PointCloud) |
|------|---------------------------|----------------------|
| **仿真成功率** | 85-92% | **90-98%** |
| **视角鲁棒性**（相机转30°） | 成功率下降 30-50% | **下降 < 5%** |
| **Sim-to-Real 迁移** | 需域随机化 | **直接迁移，< 10% 性能损失** |
| **推理速度** | 18ms (DDIM 10) | 22ms (DDIM 10) |
| **数据效率** | 100% 数据 | **60% 数据可达同等性能** |

### 视角鲁棒性实验（核心优势）

```
实验设置：训练时相机固定位置A，测试时旋转相机到位置B、C、D

成功率 (%)
100 ┤  ■ DP3
    │  □ Diffusion Policy (2D)
 80 ┤  ■
    │  ■  □
 60 ┤  ■     □
    │  ■        □
 40 ┤  ■           □
    │
 20 ┤
    │
  0 ┼──┬──────┬──────┬──────┬──
      A   B(+15°) C(+30°) D(+45°)
         测试相机角度

DP3在各角度成功率几乎持平 ← SE(3)等变性的直接体现
```

### 消融实验

| 消融项 | 成功率 | 关键发现 |
|--------|--------|----------|
| **完整 DP3** | 94% | Baseline |
| 替换 PointNet++ 为 PointNet (无层级) | 78% | 层级特征对精细操作至关重要 |
| 替换点云为 RGB (即标准 DP) | 68% | 3D 信息的核心价值 |
| 移除 FiLM 条件，改用拼接 | 87% | FiLM 调制 > 简单拼接 |
| 移除训练时 SE(3) 增强 | 81% | 数据增强强化等变性 |
| 点云点数 N=256 (vs 1024) | 83% | 点数显著影响精度 |

## 8. 对比总览

| 维度 | ACT (CVAE) | Diffusion Policy (DP) | **DP3 (Ours)** |
|------|------------|----------------------|----------------|
| **视觉输入** | 2D RGB (ResNet) | 2D RGB (ResNet/ViT) | **3D 点云 (PointNet++)** |
| **生成模型** | CVAE | DDPM | DDPM |
| **动作块长** | 100步 @ 50Hz | 16步 @ 10Hz | 16步 @ 10Hz |
| **分布假设** | 高斯（单峰） | 任意分布 | **任意分布** |
| **SE(3) 等变性** | ❌ 无 | ❌ 无（需数据增强补救） | ✅ **天然等变** |
| **视角鲁棒性** | 差 | 差 | **优秀** |
| **Sim-to-Real** | 需域随机化 | 需域随机化 | **直接迁移** |
| **推理速度** | < 1ms（单次前传） | 18ms (DDIM×10) | 22ms (DDIM×10) |
| **参数量** | ~20M | ~80M | ~60M |
| **数据效率** | 中 | 中-高 | **高** |
| **适用场景** | 高频控制、ALOHA | 精细操作、通用 | **需要泛化/多视角的场景** |

## 9. 局限性

| 局限 | 说明 | 缓解方案 |
|------|------|----------|
| **对深度传感器依赖** | 需要 RGBD 相机或深度估计模型 | 单目深度估计（如 DepthAnything）可生成伪点云 |
| **透明/反光物体** | 深度相机对镜面、玻璃失效 | 多传感器融合（RGB辅助） |
| **计算开销** | PointNet++ 比 ResNet 慢约 3-5ms | FPS 采样降点数、TensorRT 加速 |
| **训练数据获取** | 仿真中获取点云容易，真实世界需 RGBD 标注 | 仿真预训练 + 少量真实适配 |
| **稀疏点云** | 远距离物体点稀疏，特征变差 | 多尺度球查询半径 |
| **Scale 敏感性** | PointNet++ 的球查询半径是超参数 | 根据任务尺度调整半径值 |

## 10. 与相关工作的联系

- **R3M / VIP**（视觉预训练）：DP3 的点云编码器同样可以预训练，但 SE(3) 等变性是结构属性而非学习得到的
- **RT-2**（VLA）：DP3 是"观测→动作"的底层策略，RT-2 是高层语义规划，可以组合成分层架构
- **Diffusion-CCSP**：同时用点云和 RGB 作为条件，进一步融合多模态
- **3D Diffuser Actor**：类似的 PointNet++ + Diffusion 思路，但在 Transformer 中处理时序

[[Diffusion-Policy详解]] | [[PointNet++与3D点云]] | [[扩散模型基础]] | [[SE3与SO3空间变换]] | [[ACT算法详解]] | [[动作空间表征总览]]
