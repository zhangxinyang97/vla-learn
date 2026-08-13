---
title: 3D Gaussian Splatting
tags: [3dgs, gaussian-splatting, nerf, 3d-reconstruction, sim-to-real]
aliases: [3DGS, 三维高斯泼溅]
---

# 3D Gaussian Splatting

## 1. 为什么需要 3DGS：超越 NeRF

NeRF (Neural Radiance Fields, Mildenhall et al., ECCV 2020) 颠覆了 3D 重建——用 MLP 隐式编码场景，但代价高昂：

| 维度 | NeRF | 3D Gaussian Splatting (3DGS) |
|------|------|------------------------------|
| **渲染速度** | 逐像素 MLP 查询，~30s/帧 | Rasterization，≥30 FPS 实时 |
| **训练时间** | 数小时 ~ 数天 | 数十分钟 ~ 1 小时 |
| **场景表示** | 隐式 (MLP 权重) | 显式 (数百万 3D 高斯) |
| **可编辑性** | ❌ MLP 是黑箱，无法直接编辑 | ✅ 高斯可移动/旋转/缩放/删除 |
| **新视角质量** | PSNR ~30-32 dB | PSNR ~33-35 dB（相当或更好） |
| **动态场景** | 困难，需额外时间维输入 | 天然支持（每帧独立高斯） |
| **与图形学管线兼容** | ❌ 需额外网格提取步骤 | ✅ 高斯 → Mesh/点云，可直接导入 Blender/Unity |

**核心权衡**：NeRF 是"用神经网络记住场景"，3DGS 是"用显式基元重建场景"——后者牺牲了连续表示的优雅，换来了实时渲染和可编辑性，这对**机器人仿真**至关重要。

## 2. 3D Gaussian Primitives（三维高斯基元）

### 2.1 为什么要用高斯？

一个 3D 高斯完全由以下参数定义：

$$G(x) = e^{-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

其中：
- $\mu \in \mathbb{R}^3$：高斯中心（均值），决定空间位置
- $\Sigma \in \mathbb{R}^{3 \times 3}$：协方差矩阵，决定形状、方向和尺度

**选择高斯的理由**：
1. 可微分 → 梯度可反向传播
2. 紧凑：3D 协方差可分解为 **旋转 $R$** + **缩放 $S$** → $\Sigma = RSS^T R^T$
3. 在 2D 投影后仍保持高斯形式（投影近似为高斯）→ 高效光栅化
4. Alpha blending（透明度混合）解析可微

### 2.2 协方差分解：旋转 + 缩放

为保证 $\Sigma$ 半正定且物理有效，3DGS 将其分解为：

$$\Sigma = R S S^T R^T$$

参数化方式：
- **旋转 $R$**：用四元数 $q \in \mathbb{R}^4$ 表示（避免万向节锁，可微）
- **缩放 $S$**：对角矩阵 $S = \text{diag}(s_x, s_y, s_z)$，其中 $s \in \mathbb{R}^3$（各向异性缩放）

优势：优化 $q$ 和 $s$ 比直接优化 $\Sigma_{3\times3}$ 更稳定，且自动满足半正定性。

### 2.3 每个高斯的完整参数

| 参数 | 符号 | 维度 | 说明 |
|------|------|------|------|
| 位置 (Mean) | $\mu$ | $\mathbb{R}^3$ | 高斯中心坐标 $(x,y,z)$ |
| 协方差 (Covariance) | $\Sigma$ | 等效 7 参数 | $q \in \mathbb{R}^4$ (旋转) + $s \in \mathbb{R}^3$ (缩放) |
| 不透明度 (Opacity) | $\alpha$ | $\mathbb{R}^1$ | $\alpha \in [0, 1]$，通过 sigmoid 激活 |
| 颜色 (Color) | $c$ | $\mathbb{R}^{48}$ | 球谐系数 (SH)，见 §2.4 |

总计每个高斯约 **59 个可学习参数**，场景通常包含 **100万 ~ 600万** 个高斯。

### 2.4 球谐函数 (Spherical Harmonics) 编码视角依赖颜色

3DGS 不使用 MLP 解码颜色，而是用 **球谐函数 (Spherical Harmonics, SH)** 显式编码每个高斯在不同视角下的颜色变化：

$$c(\mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} k_l^m Y_l^m(\mathbf{d})$$

其中：
- $\mathbf{d} \in \mathbb{R}^3$：观察方向向量（从高斯中心指向相机）
- $L$：球谐阶数，3DGS 使用 $L=3$（前 4 阶）→ 每通道 $(L+1)^2 = 16$ 个系数
- $Y_l^m(\mathbf{d})$：实球谐基函数，正交归一化
- $k_l^m$：该高斯的 SH 系数（可学习参数）
- RGB 3 通道 × 16 系数 = **48 个 SH 参数** 每高斯

> **为什么 $L=3$ 足够？** 低阶 SH 捕获漫反射（视角无关），高阶捕获镜面反射（高光），$L=3$ 在效果和参数量之间取平衡。大多数自然场景镜面反射有限。

**与 NeRF 的视角依赖对比**：

| | NeRF | 3DGS |
|---|------|------|
| 颜色编码 | MLP: $F_\Theta(x, d) \to (c, \sigma)$ | SH 系数显式存储 |
| 参数量 | 固定（MLP 权重 ~1M） | 随场景增长（$N \times 48$） |
| 推理速度 | 每像素 MLP 前向 | 解析计算（查表 + 点积） |

## 3. 可微光栅化 (Differentiable Rasterization)

这是 3DGS 的核心引擎——将 3D 高斯投影到 2D，逐像素合成最终图像，且全程可微。

### 3.1 投影：3D → 2D 高斯溅射 (Splatting)

将 3D 协方差矩阵投影到图像平面：

$$\Sigma' = J W \Sigma W^T J^T$$

其中：
- $W \in \mathbb{R}^{3\times3}$：世界到相机坐标的旋转矩阵（view transform）
- $J \in \mathbb{R}^{2\times3}$：投影变换的雅可比矩阵（局部仿射近似，因为透视投影非线性）

**关键技巧**：投影后取 $\Sigma'$ 的左上 $2\times2$ 子块 → 得到 2D 屏幕空间高斯：

$$G^{2D}(p) = e^{-\frac{1}{2}(p - \mu')^T (\Sigma'_{2\times2})^{-1} (p - \mu')}$$

其中 $p$ 为像素坐标，$\mu'$ 为投影后的 2D 均值。

### 3.2 基于 Tile 的快速排序

直接对所有高斯按深度排序 → $O(N\log N)$，N 可达数百万 → 太慢。

3DGS 的加速策略：
1. 将屏幕划分为 $16\times16$ 的 tile
2. 每个 tile 只保留覆盖到该 tile 的高斯（通过 2D bounding box 判断）
3. 在每个 tile 内按深度排序（quick sort，但排序数量远小于全场景）
4. 使用 **CUDA 自定义核函数** 实现并行 tile-based rasterization

```
屏幕 (1920×1080)
  ↓ 划分
Tile 0,0   Tile 0,1   ...   Tile 0,119
Tile 1,0   Tile 1,1   ...   Tile 1,119
  ...
Tile 67,0  Tile 67,1  ...   Tile 67,119

对每个 Tile：
  1. 收集覆盖该 tile 的高斯列表（去重）
  2. 按深度由近到远排序
  3. 串行 alpha blending（tile 内像素并行）
```

### 3.3 Alpha Blending（透明度混合）

像素颜色通过由近到远累积高斯贡献得到：

$$C(p) = \sum_{i=1}^{N} c_i \cdot \alpha_i' \cdot \prod_{j=1}^{i-1} (1 - \alpha_j')$$

其中：
- $C(p)$：像素 $p$ 的最终颜色
- $c_i$：第 $i$ 个高斯的颜色（由 SH 系数和观察方向 $d$ 计算）
- $\alpha_i' = \alpha_i \cdot G_i^{2D}(p)$：第 $i$ 个高斯在像素 $p$ 处的有效不透明度
  - $\alpha_i$：该高斯的基础不透明度（sigmoid 激活）
  - $G_i^{2D}(p)$：2D 高斯在像素 $p$ 处求值（权重越大 → 该位置越接近高斯中心 → 贡献越大）
- $\prod_{j=1}^{i-1}(1-\alpha_j')$：透射率（Transmittance），前方所有高斯阻挡后剩下的光量

> **类比**：高斯的 Alpha Blending 与 NeRF 的体积渲染公式 $C = \sum T_i(1-e^{-\sigma_i\delta_i})c_i$ 结构相同——区别是 NeRF 沿射线采样，3DGS 直接对排序后的高斯混合。

**提前终止**：当透射率 $T = \prod_{j=1}^{i}(1-\alpha_j') < \epsilon$（通常 $\epsilon = 10^{-4}$），停止混合——剩余高斯对像素贡献可忽略。

### 3.4 反向传播

梯度从像素损失回传到高斯参数：

$$\frac{\partial \mathcal{L}}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial C} \cdot \frac{\partial C}{\partial \alpha'} \cdot \frac{\partial \alpha'}{\partial G^{2D}} \cdot \frac{\partial G^{2D}}{\partial \mu'}$$

- 链式法则经过：alpha blending → 2D 高斯求值 → 投影变换 → 3D 参数
- **所有步骤解析可微**，CUDA 实现自定义 backward kernel

## 4. 训练策略

### 4.1 初始化

使用 SfM (Structure from Motion, 如 COLMAP) 输出的稀疏点云作为初始高斯中心：

- 每个 SfM 点 → 一个高斯的 $\mu$（位置）
- 初始 $\alpha$：0.1（半透明）
- 初始 $s$：最近邻三个点的平均距离（自适应尺度）
- 初始 $q$：单位四元数（无旋转）
- 初始 SH：$k_0^0 = \text{RGB}$，高阶系数全 0（纯漫反射）

### 4.2 损失函数

$$\mathcal{L} = (1 - \lambda) \cdot \mathcal{L}_1 + \lambda \cdot \mathcal{L}_{\text{SSIM}}$$

其中：
- $\mathcal{L}_1 = \|I_{\text{render}} - I_{\text{gt}}\|_1$：像素级 L1 损失，捕获颜色差异
- $\mathcal{L}_{\text{SSIM}} = 1 - \text{SSIM}(I_{\text{render}}, I_{\text{gt}})$：结构相似性损失 (SSIM)，捕捉感知质量
- $\lambda = 0.2$：SSIM 权重（经验值，过大会产生模糊）

> **为什么 L1 而不是 L2？** L1 对离群点（如高光、反射）更鲁棒，避免过惩罚个别像素。

### 4.3 自适应密度控制 (Adaptive Density Control, ADC)

训练的**核心创新**——动态调整高斯数量和分布：

```
每 3000 步检查每个高斯的梯度累积 ──→ 决定是否 clone / split / prune

                    累积梯度 > 阈值 τ_pos？
                   /                    \
                 是                      否
                 │                        │
         高斯 "大" 还是 "小"？          α < ε_min？
          /              \                 │
        大              小               是 → prune
        │               │
      split           clone
  (一分为二)       (复制平移)
  → 增加细节       → 填补空洞
```

**三个操作详解**：

| 操作 | 触发条件 | 机制 | 目的 |
|------|---------|------|------|
| **Clone** | $\|\nabla L\| > \tau_{\text{pos}}$ 且 $\max(s) < \tau_{\text{scale}}$ | 在与梯度方向一致处复制一个相同高斯，平移一小步 | 填补几何空洞（小高斯 + 高梯度 = 该区域高斯不足） |
| **Split** | $\|\nabla L\| > \tau_{\text{pos}}$ 且 $\max(s) > \tau_{\text{scale}}$ | 一分为二，缩放因子 $\phi=1.6$，用采样位置初始化 | 将过大高斯分裂为更细粒度的表示（覆盖过多区域 → 过模糊） |
| **Prune** | $\alpha < \epsilon_\alpha$ 或 $\max(s) > \tau_{\text{max}}$ | 删除高斯 | 去除透明/世界空间过大的无效高斯 |

参数：$\tau_{\text{pos}} = 0.0002$（梯度阈值），$\tau_{\text{scale}} = 0.01$（大小阈值），$\epsilon_\alpha = 0.005$

### 4.4 完整训练流程

```
初始化：SfM 稀疏点云 → N₀ 个高斯 (~5000)
  ↓
迭代 (默认 30,000 步)：
  ├─ 随机采样训练视角
  ├─ Forward: 投影 → Tile 排序 → Alpha Blending → 渲染图像
  ├─ 计算 L = (1-λ)L1 + λ·L_SSIM(I_render, I_gt)
  ├─ Backward: 梯度 → μ, Σ, α, SH
  ├─ Optimizer step (Adam, lr 递减)
  └─ 每 3000 步: 自适应密度控制 (Clone/Split/Prune)
  ↓
收敛: ~1M-6M 高斯
  ↓
输出: .ply 文件 (包含所有高斯参数)
```

**学习率调度**：
- $\mu$ (位置)：指数衰减，初始 $1.6\times10^{-4}$，最终 $1.6\times10^{-6}$
- $\alpha$ (不透明度)：固定 $0.05$
- $s$ (缩放)：固定 $0.005$
- $q$ (旋转)：固定 $1\times10^{-3}$
- SH 系数：固定 $2.5\times10^{-3}$

## 5. PyTorch 伪代码：3D 高斯前向传播

```python
import torch
import torch.nn.functional as F

class GaussianModel:
    def __init__(self, sh_degree=3):
        self.sh_degree = sh_degree
        # 每个高斯的可学习参数
        self._xyz = None          # (N, 3) 位置 μ
        self._features_dc = None  # (N, 1, 3) SH 零阶系数 (RGB base)
        self._features_rest = None # (N, 15, 3) SH 高阶系数
        self._scaling = None      # (N, 3) 对数缩放 log(s)
        self._rotation = None     # (N, 4) 四元数 q
        self._opacity = None      # (N, 1) 对数不透明度 logit(α)

    def build_covariance(self, scaling, rotation):
        """Σ = R S S^T R^T，S = diag(exp(scaling))"""
        S = torch.diag_embed(torch.exp(scaling))   # (N, 3, 3)
        R = self.build_rotation_matrix(rotation)     # (N, 3, 3)
        L = R @ S  # (N, 3, 3)
        return L @ L.transpose(-1, -2)  # Σ = LL^T, 保证半正定

    def build_rotation_matrix(self, q):
        """四元数 → 旋转矩阵，q=(r, x, y, z)"""
        r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            torch.stack([1-2*(y**2+z**2), 2*(x*y-r*z), 2*(x*z+r*y)], dim=-1),
            torch.stack([2*(x*y+r*z), 1-2*(x**2+z**2), 2*(y*z-r*x)], dim=-1),
            torch.stack([2*(x*z-r*y), 2*(y*z+r*x), 1-2*(x**2+y**2)], dim=-1)
        ], dim=-2)
        return R  # (N, 3, 3)

    def compute_color_from_sh(self, sh_features, view_dir):
        """视角依赖颜色：c(d) = Σ k_l^m Y_l^m(d)"""
        # SH 基函数求值 Y_l^m(view_dir) → (N, 16)
        sh_basis = eval_sh_basis(self.sh_degree, view_dir)
        # 点积：SH 系数 (N, 3, 16) @ SH 基 (N, 16, 1) → (N, 3)
        color = (sh_features @ sh_basis.unsqueeze(-1)).squeeze(-1)
        return torch.sigmoid(color)  # 压缩到 [0, 1]

    def compute_2d_covariance(self, means3D, cov3D, viewmatrix, focal):
        """投影 3D 协方差到 2D: Σ' = J W Σ W^T J^T"""
        # W: world-to-camera 旋转部分
        W = viewmatrix[:3, :3]  # (3, 3)
        # 投影点
        t = (viewmatrix @ torch.cat([means3D, torch.ones_like(means3D[:, :1])],
              dim=-1).T).T  # (N, 4)
        # 透视除法 → NDC
        t_ndc = t[:, :2] / t[:, 3:4]
        # 雅可比 J = ∂(pixel_x, pixel_y) / ∂(cam_x, cam_y)
        # 透视投影近似
        J = torch.zeros((means3D.shape[0], 2, 3), device=means3D.device)
        J[:, 0, 0] = focal / t[:, 2]  # ∂px/∂cx
        J[:, 0, 2] = -focal * t[:, 0] / (t[:, 2] ** 2)  # ∂px/∂cz
        J[:, 1, 1] = focal / t[:, 2]  # ∂py/∂cy
        J[:, 1, 2] = -focal * t[:, 1] / (t[:, 2] ** 2)  # ∂py/∂cz
        # Σ'_2D = J W Σ W^T J^T
        cov_cam = W @ cov3D @ W.T  # (N, 3, 3)
        cov_2d = J @ cov_cam @ J.transpose(-1, -2)  # (N, 2, 2)
        return cov_2d

    def forward(self, viewpoint_camera):
        """完整前向：参数 → 渲染图像"""
        N = self._xyz.shape[0]

        # 1. 激活参数
        opacity = torch.sigmoid(self._opacity)  # α: logit → [0,1]
        scaling = torch.exp(self._scaling)      # s: log → >0

        # 2. 构建 3D 协方差
        cov3D = self.build_covariance(scaling, self._rotation)  # (N, 3, 3)

        # 3. 投影到 2D
        cov2D = self.compute_2d_covariance(
            self._xyz, cov3D,
            viewpoint_camera.world_view_transform,
            viewpoint_camera.focal_length
        )  # (N, 2, 2)

        # 4. 计算 2D 高斯在屏幕上的 bounding box
        #    3σ 原则：高斯 99.7% 能量在 3σ 内
        det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
        # 特征值 λ₁, λ₂ 决定 blob 半径
        lambda1 = 0.5 * (cov2D[:, 0, 0] + cov2D[:, 1, 1] +
                         torch.sqrt((cov2D[:, 0, 0] - cov2D[:, 1, 1])**2 +
                          4 * cov2D[:, 0, 1]**2))
        lambda2 = 0.5 * (cov2D[:, 0, 0] + cov2D[:, 1, 1] -
                         torch.sqrt((cov2D[:, 0, 0] - cov2D[:, 1, 1])**2 +
                          4 * cov2D[:, 0, 1]**2))
        radius = 3.0 * torch.sqrt(torch.max(lambda1, lambda2))
        # bounding box = [μ_x - radius, μ_y - radius, μ_x + radius, μ_y + radius]

        # 5. 视角依赖颜色 (SH)
        view_dir = F.normalize(viewpoint_camera.center - self._xyz, dim=-1)
        sh_features = torch.cat([self._features_dc, self._features_rest], dim=1)
        color = self.compute_color_from_sh(sh_features, view_dir)  # (N, 3)

        # 6. Tile-based 排序 + Alpha Blending (CUDA kernel)
        #    此为简化版本，实际由 CUDA 核函数完成
        rendered_image = differentiable_splatting(
            self._xyz, cov2D, color, opacity,
            viewpoint_camera.image_height,
            viewpoint_camera.image_width
        )
        return rendered_image
```

**关键实现细节**：
- `torch.sigmoid(opacity)` 确保 $\alpha \in [0, 1]$
- `torch.exp(scaling)` 确保 $s > 0$
- `build_rotation_matrix` 四元数 → 旋转矩阵（标准公式），自动保持正交性
- `compute_2d_covariance` 中的雅可比 $J$ 是 EWA (Elliptical Weighted Average) splatting 的关键——透视投影的非线性部分

## 6. 对比：NeRF vs 3DGS vs InstantNGP

| 特性 | NeRF (2020) | InstantNGP (2022) | 3DGS (2023) |
|------|-------------|-------------------|-------------|
| **表示** | MLP $F_\Theta(x,d) \to (c,\sigma)$ | Hash Grid + tiny MLP | 显式 3D Gaussian 集合 |
| **渲染方式** | 体积渲染（射线采样） | 体积渲染（射线采样） | 可微光栅化 (Splatting) |
| **训练时间** | ~12-24h (单场景) | ~5-10 min | ~30-60 min |
| **渲染速度** | <1 FPS | ~10 FPS (tiny MLP) | ≥30 FPS (实时) |
| **参数量** | ~1.2M (固定) | ~12M (hash table) + 64K (MLP) | 59 × N (N~1-6M 高斯，~60-350M 参数) |
| **存储** | ~5 MB (MLP 权重) | ~50 MB (hash table) | ~50-500 MB (.ply 文件) |
| **质量 PSNR** | ~31 dB (Mip-NeRF 360) | ~33 dB | ~33-35 dB |
| **反走样** | ✅ 多尺度采样 (Mip-NeRF) | ❌ 需手动处理 | ✅ 3D 高斯自适应缩放 |
| **动态场景** | 困难 | 困难 | ✅ 天然适配（每帧独立高斯） |
| **编辑性** | ❌ 黑箱 | ❌ 黑箱 | ✅ 直接编辑/删除/移动高斯 |
| **GPU 内存** | ~4-8 GB | ~2-4 GB | ~4-12 GB |
| **代码复杂度** | ~3K 行 | ~5K 行 (CUDA hash) | ~15K 行 (CUDA rasterizer) |

### 什么时候用哪个？

```
需要快速原型 + 小存储？  → InstantNGP
需要实时渲染 + 可编辑？  → 3DGS ✅
需要理论优雅 + 连续表示？ → NeRF (Mip-NeRF 360 变体)
机器人仿真 (Real-to-Sim)？ → 3DGS ✅✅ (可编辑 + 实时 + 显式)
```

## 7. Re3Sim：3DGS 在 Real-to-Sim 中的应用

### 7.1 动机

机器人仿真（Sim-to-Real Transfer）的核心瓶颈：**仿真环境与真实世界的视觉差距 (Visual Domain Gap)**。

传统方案：
```
真实场景 → 人工建模 (Blender/Unity) → 仿真环境
          ↑ 耗时数周，纹理/光照无法准确复现
```

Re3Sim (Real-to-Sim via 3DGS) 的思路：
```
真实场景 → 多视角采集 → 3DGS 训练 → 高斯 → Mesh → 仿真环境
          ↑ 数十分钟               ↑ 可编辑中间表示
```

### 7.2 Re3Sim 工作流程

```
Step 1: 数据采集
  ├─ 手持相机 / 机器人视角环绕拍摄 (~200-500 张)
  ├─ COLMAP SfM → 相机位姿 + 稀疏点云
  └─ 可选：深度传感器辅助 (RGB-D)

Step 2: 3DGS 训练 (30 min)
  ├─ SfM 点云初始化高斯
  ├─ L1 + SSIM 损失优化
  └─ 自适应密度控制 → 高保真场景重建

Step 3: 高斯 → 仿真资产
  ├─ SuGaR / 2DGS: 高斯 → Mesh 提取 (Poisson Surface Reconstruction)
  ├─ 纹理烘焙: SH 系数 → 漫反射 + 镜面反射贴图
  └─ 物理属性标注: 碰撞体、质量、摩擦系数 (可选)
  
Step 4: 导入仿真器
  ├─ Isaac Sim / MuJoCo / Gazebo
  └─ 域随机化 (Domain Randomization) 自动扩充
```

### 7.3 关键技术：从高斯到网格

高斯是体积基元（volumetric primitives），无法直接做碰撞检测。需转换为表面 Mesh：

**SuGaR (Surface-Aligned Gaussians, CVPR 2024)**：
1. 引入正则化：强制高斯扁化为与表面对齐的薄片（最小化法向方向缩放）
2. 从对齐的高斯中采样有符号距离函数 (SDF)
3. Poisson Surface Reconstruction → 水密 Mesh
4. 纹理烘焙 → 标准 PBR 材质

**2DGS (2D Gaussian Splatting, SIGGRAPH 2024)**：
- 将 3D 椭球体退化为 2D 圆盘（surfels）
- 天然与表面贴合 → 直接提取法向和 Mesh
- 渲染质量与 3DGS 相当，但几何更精确

## 8. 在机器人学中的应用

### 8.1 仿真环境中的抓取

```
真实桌面场景
  ↓ 多视角采集 (3 min)
3DGS 重建 (30 min)
  ↓ 高斯 → Mesh → Isaac Sim
仿真环境中的抓取策略训练
  ↓ RL / 模仿学习
策略可直接迁移到真实环境 (Sim-to-Real Gap 大幅缩小)
```

**优势**：
- 真实纹理和光照（SH 编码的视角依赖反射）
- 高保真几何（ADC 自适应密度 → 精细边缘和薄物体）
- 快速迭代：修改场景 → 重新训练 3DGS → 新仿真资产（分钟级）

### 8.2 Sim-to-Real Visual Gap 缩减

| Gap 来源 | 传统方法 | 3DGS 方法 |
|----------|---------|-----------|
| 几何精度 | CAD 模型（理想化） | 真实重建（含缺陷、变形） |
| 纹理/材质 | 手工贴图 | SH 编码的真实外观 |
| 光照 | 手动调参 | SH 自动捕获环境光 |
| 反射/高光 | PBR 近似 | 高阶 SH 自然建模 |
| 透明/半透明 | 困难 | Alpha blending 天然支持 |

### 8.3 更多机器人应用场景

| 应用 | 描述 | 关键优势 |
|------|------|---------|
| **视觉导航** | 在 3DGS 重建的环境中训练导航策略 | 可生成任意新视角训练数据 |
| **6-DoF 抓取** | 在重建场景中进行抓取位姿采样和评估 | 精确几何 + 真实外观 → 可靠抓取成功率预测 |
| **物体操纵** | 编辑高斯（移动/旋转）模拟物体交互 | 显式编辑 = 交互式仿真 |
| **SLAM** | 3DGS 作为地图表示 → 实时定位与建图 | 比点云更紧凑，比 Mesh 更灵活 |
| **NeRF-to-Grasp** | 从 NeRF/3DGS 场景直接生成抓取位姿 | 无需显式 3D 重建步骤 |

## 9. 局限与前沿

| 局限 | 描述 | 前沿方案 |
|------|------|---------|
| **反射/折射** | 高斯假设朗伯 + 镜面混合，无法正确建模镜面反射 | Ref-GS, Mirror-3DGS |
| **天空/远背景** | 高斯无法延伸到无穷远 | 背景球 + 天空盒混合 |
| **无界场景** | 远处大高斯导致数值不稳定 | Mip-Splatting (多尺度高斯) |
| **动态物体** | 静态训练假设被打破 | 4D-GS (时间维高斯), Deformable-3DGS |
| **存储膨胀** | 百万级高斯 → GB 级存储 | 压缩：量化 + 剪枝 + VQ-VAE |
| **训练需要 COLMAP** | 依赖 SfM 提供位姿和初始点云 | DUSt3R / MASt3R (端到端替代) |
| **雨天/夜景** | 训练视角不足导致空洞/漂浮物 | Scaffold-GS (锚点引导初始化) |

## 10. 相关阅读

- [[PointNet++与3D点云]] — 3D 感知的基础：从无序点云提取特征，与 3DGS 的 SfM 初始化互补
- [[NVIDIA-Sim-to-Real技术]] — Isaac Sim 域随机化 + 3DGS 真实场景重建的完整管线
- [[仿真引擎与ROS2部署]] — 将 3DGS 重建资产导入仿真器并部署到真实机器人
- [[DINOv2与自监督视觉特征]] — 在 3DGS 渲染图像上提取密集视觉特征用于机器人策略
- [[CLIP与SigLIP对比]] — 多模态引导的 3D 场景理解与语义编辑

---

**参考论文**：
- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
- Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields", ECCV 2020
- Müller et al., "Instant Neural Graphics Primitives", SIGGRAPH 2022
- Guédon & Lepetit, "SuGaR: Surface-Aligned Gaussian Splatting", CVPR 2024
- Huang et al., "2D Gaussian Splatting for Geometrically Accurate Radiance Fields", SIGGRAPH 2024
