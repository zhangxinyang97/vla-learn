---
title: DINOv2与自监督视觉特征
tags: [dinov2, self-supervised, vision, dense-features, openvla]
aliases: [DINOv2, 自监督密集特征]
---

# DINOv2 与自监督视觉特征

## 为什么自监督 > 监督学习（对机器人学而言）

| 维度 | 监督预训练 (ImageNet) | 自监督 (DINOv2) |
| ---- | -------------------- | ---------------- |
| 标签依赖 | 需要人工标注 | 零标注 |
| 特征粒度 | 偏向分类 token，丢失空间 | 保留密集 patch 特征 |
| 泛化能力 | 过拟合标注类别 | 学到通用视觉基元 |
| 机器人适配 | 识别物体 → 不够；需要空间推理 | 密集对应 + 深度线索 → 直接可用 |

机器人任务（抓取、导航、操控）需要的不是"这是什么"，而是**像素级空间理解**——DINOv2 正是为此而生。

## DINO 核心框架：Teacher-Student 知识蒸馏

两个结构相同的 ViT，无需标签，通过自蒸馏学习：

```
┌──────────────┐         ┌──────────────┐
│   Teacher    │ ──→EMA──│   Student    │
│  (momentum)  │   Loss  │  (learnable) │
└──────────────┘         └──────────────┘
      ↑                       ↑
  global views          global + local views
  (224×224)             (224×224 + 96×96)
```

Teacher 通过 EMA（指数移动平均）更新，不从梯度学习；Student 正常反向传播。

### 蒸馏损失

Cross-entropy between teacher softmax output and student softmax output:

$$\mathcal{L}_{DINO} = -P_t \log P_s$$

其中：

- $P_t = \text{softmax}\left(\frac{g_t(x) - c}{\tau_t}\right)$ —— Teacher 输出概率，$g_t(x)$ 为 teacher logits，$\tau_t$ 为 teacher 温度
- $P_s = \text{softmax}\left(\frac{g_s(x)}{\tau_s}\right)$ —— Student 输出概率，$\tau_s$ 为学生温度
- $c$：centering 偏置向量（EMA 累积），防止某一维度坍塌

### Centering（中心化）

$$c \leftarrow m \cdot c + (1-m) \cdot \frac{1}{B}\sum_{i=1}^B g_t(x_i)$$

- 对 teacher 输出减去运行均值 $c$
- 防止 teacher 输出坍塌到单一维度（所有概率集中在一个 token 上）
- $m$ 为动量系数（典型值 $0.9$）

### Sharpening（锐化）

$$\tau_t < \tau_s$$

- Teacher 温度 $\tau_t$ 更低（典型 $0.04$），Student 温度 $\tau_s$ 更高（典型 $0.1$）
- 低温度 → softmax 更尖锐 → teacher 预测更自信 → 提供更强的学习信号
- Centering 与 Sharpening 互为制衡：centering 防止单一维度坍塌，sharpening 防止均匀分布坍塌

### DINO 中 Centering & Sharpening 的协同机制

```
无约束            Centering only      Sharpening only      Centering + Sharpening
  ↓                   ↓                    ↓                       ↓
均匀分布坍塌       某一维坍塌            某一维坍塌             稳定均衡
(所有token同概率)  (所有概率集于1个token) (所有概率集于1个token)  ✅ 正常工作
```

为防止 collapse（模型输出 constant），DINO 采用 centering + sharpening 双重约束：

- **Centering** 防止某一维度过度占优（避免模型只输出"狗"这一个 class）
- **Sharpening** 防止 teacher 输出过于均匀（避免蒸馏信号消失，所有 token 同概率则没有信息量）
- 两者共同作用 → Teacher 输出既多样化又有区分度 → Student 学到丰富表示

## 多尺度裁剪训练（Multi-Crop）

DINO 引入多尺度策略以增强区域级理解：

$$\mathcal{L}_{DINO}^{multi} = \sum_{x_g \in \{g_1,g_2\}} \sum_{x \in V,\; x \neq x_g} H\!\left(P_t(x_g),\, P_s(x)\right)$$

- $g_1,g_2$：2 个 global crop（大尺寸，如 224×224）→ 只给 Teacher
- $l_1,...,l_N$：$N$ 个 local crop（小尺寸，如 96×96）→ 只给 Student
- $V = \{g_1,g_2,l_1,...,l_N\}$：所有视图集合
- $H$：交叉熵 $H(p,q)=-\sum p\log q$
- Teacher 只处理 global views（大局观），Student 处理所有 views（细节 + 大局）

效果：模型学会"局部细节属于哪个全局物体"——即**局部到全局的对应关系**，这正是密集特征的关键。

## DINOv2 的改进 (Oquab et al., 2023)

DINOv2 在 DINO 基础上做了关键升级：

| 改进 | DINO (2021) | DINOv2 (2023) |
| ---- | ----------- | ------------- |
| 模型规模 | ViT-S/B | ViT-g (1B+) |
| 训练数据 | ImageNet (1.2M) | 大规模精选数据 (LVD-142M) |
| 训练目标 | DINO only | DINO + iBOT (masked image modeling) |
| 正则化 | — | KoLeo regularizer（特征 spread） |
| 分辨率 | 224² | 支持 518² 高分辨率 |
| Patch 级别 | 有密集特征但弱 | 极强密集特征（patch 级语义） |

### iBOT 辅助损失

DINOv2 联合使用 DINO 损失和 iBOT（image BERT pre-training）损失：

$$\mathcal{L}_{DINOv2} = \lambda_{DINO} \cdot \mathcal{L}_{DINO} + \lambda_{iBOT} \cdot \mathcal{L}_{iBOT}$$

- $\mathcal{L}_{iBOT}$：随机 mask 输入 patches → teacher 处理完整图 → student 从 masked 图预测被 mask 的 patch 特征
- $\lambda_{DINO}$、$\lambda_{iBOT}$：加权系数（通常各 1.0）
- iBOT 迫使模型学习 patch 级别的上下文推理，大幅提升密集特征质量

### KoLeo 正则化

$$\mathcal{L}_{KoLeo} = -\frac{1}{n}\sum_{i=1}^n \log \min_{j\neq i} \lVert f_i - f_j \rVert$$

- $f_i$：第 $i$ 个样本的归一化特征向量
- $\min_{j\neq i} \lVert f_i - f_j \rVert$：$f_i$ 到最近邻的距离
- 目标：最大化特征之间的最小距离 → **防止特征坍缩到低维流形** → 特征在整个空间中均匀分布
- 理解：每个样本的特征都要"有自己的位置"，不与邻居挤在一起

## 涌现能力（Emergence）

DINOv2 在**没有**任何像素级标注的情况下，自然涌现出以下能力：

### 1. 语义分割（无需微调）

```
输入图片 ──→ DINOv2 ViT ──→ Patch Features (H/P × W/P × D)
                                   │
                                   ├──→ PCA 降维到 3 通道 → 可视化
                                   │    自动呈现语义边界
                                   │
                                   └──→ K-Means 聚类 → 物体分割
                                        无需任何标注
```

Attention maps 自动对应物体部件——狗的 attention 在狗身上，背景的 attention 在背景上。这正是**自监督学习学到语义**的直接证据。

### 2. 密集对应（Dense Correspondence）

同一物体的不同视角/姿态，DINOv2 能在 patch 级别匹配对应点：

```
Image A 某 patch 特征  ←──余弦相似度──→  Image B 某 patch 特征
                                            ↓
                                       找到最优匹配点
```

这使机器人可以跨视角追踪同一物体——抓取任务的核心能力（"从侧视图看的手柄，从顶视图在哪个位置？"）。

### 3. 深度线索（Depth from Features）

DINOv2 的 patch 特征隐式编码了深度信息：距离相近的物体在特征空间中也相近。通过特征相似度矩阵可以直接估计相对深度排序，无需双目摄像头。

## OpenVLA 的双编码器架构

OpenVLA（Open Vision-Language-Action Model）是当前最强的开源 VLA 模型，其核心设计选择是**双视觉编码器**：

```
          ┌──────────────────────┐
          │      Robot Image     │
          └──────────┬───────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌───────────────┐        ┌────────────────┐
│    SigLIP     │        │    DINOv2      │
│  (全局语义)    │        │  (密集空间特征)  │
│               │        │                │
│  Output:      │        │  Output:       │
│  [1 × D_sig]  │        │  [N_patch × D_dino]│
└───────┬───────┘        └───────┬────────┘
        │                        │
        │  ┌─────────────────┐   │
        │  │  全局 avg pool   │   │
        │  └─────────────────┘   │
        │                        │
        └────────┬───────────────┘
                 ▼
        ┌────────────────┐
        │  Concat + Proj │
        │  [1 × D_fused] │
        └───────┬────────┘
                 ▼
        ┌────────────────┐
        │  Llama 2 (LLM) │
        └───────┬────────┘
                 ▼
        ┌────────────────┐
        │  Action Tokens │
        └────────────────┘
```

### 为什么需要两个编码器？

| 编码器 | 提供的信息 | 机器人中的作用 |
| ------ | ---------- | -------------- |
| SigLIP | 全局语义："这是什么场景？" | 理解任务上下文 |
| DINOv2 | 密集空间："物体在哪里？形状如何？" | 精确操控定位 |

单一编码器的局限：
- **只用 SigLIP**：知道"桌上有个杯子"，但不知道杯子的精确位置和朝向
- **只用 DINOv2**：知道每个像素的语义，但缺乏全局场景理解
- **两者结合**：既有场景语义，又有空间精度 → 可以执行"拿起桌子右上角的蓝色杯子"

### 融合方式

$$\mathbf{f}_{fused} = \text{Proj}\left(\left[\mathbf{f}_{SigLIP} \;\|\; \text{AvgPool}(\mathbf{f}_{DINOv2})\right]\right)$$

- $\mathbf{f}_{SigLIP} \in \mathbb{R}^{D_{sig}}$：SigLIP 的 [CLS] token 或全局池化特征
- $\mathbf{f}_{DINOv2} \in \mathbb{R}^{N_{patch} \times D_{dino}}$：DINOv2 的 patch 特征（例如 ViT-L/14 @ 224² → $256 \times 1024$）
- $\text{AvgPool}$：对 patch 维度取平均 → $\mathbb{R}^{D_{dino}}$
- $[\cdot \|\cdot]$：通道维度拼接
- $\text{Proj}$：线性投影（或小 MLP）→ 对齐到 LLM 的输入维度
- 最终 $\mathbf{f}_{fused} \in \mathbb{R}^{D_{llm}}$ 作为视觉 token 输入 Llama 2

## 与其他视觉模型的对比

| 模型 | 训练方式 | 特征类型 | 密集特征 | 语义理解 | 机器人适配 |
| ---- | -------- | -------- | -------- | -------- | ---------- |
| CLIP | 对比学习 | 全局 | ❌ 弱 | ✅ 强 | 分类/检索尚可，操控精度不足 |
| SigLIP | 对比学习（逐对Sigmoid） | 全局 | ❌ 弱 | ✅ 强 | 全局语义好，空间定位弱 |
| DINOv2 | 自蒸馏+iBOT | 密集 Patch | ✅ 极强 | ✅ 强 | 空间理解极好，语义也够 |
| SAM | 提示分割 | Mask | ✅ 仅 mask | ❌ 无语义 | 分割强但需要 prompt，无语义 |
| MAE | 掩码重建 | 密集 Patch | ✅ 中等 | ⚠️ 弱 | 重建能力好但缺少语义判别力 |
| DINO | 自蒸馏 | 密集 Patch | ✅ 强 | ✅ 中 | DINOv2 的前身，规模较小 |

关键洞察：**SigLIP + DINOv2 是互补关系，不是互斥**。OpenVLA 的设计证明了这一点——用 SigLIP 回答"是什么"，用 DINOv2 回答"在哪里"。

## DINOv2 特征提取（PyTorch 伪代码）

```python
import torch
from transformers import AutoImageProcessor, Dinov2Model
from PIL import Image

# 加载预训练模型
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
model = Dinov2Model.from_pretrained("facebook/dinov2-large")
model.eval()

def extract_dinov2_features(image: Image.Image):
    """
    提取 DINOv2 特征
    
    Args:
        image: PIL Image (RGB)
    
    Returns:
        cls_token:   [1, D]        全局特征 (类似 CLIP 的 image embedding)
        patch_feats: [N_patches, D] 密集特征 (空间保留)
    """
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs.last_hidden_state: [1, 1+N_patches, D]
    #   - position 0:  CLS token
    #   - position 1+: patch tokens
    
    cls_token   = outputs.last_hidden_state[:, 0, :]        # [1, D]
    patch_feats = outputs.last_hidden_state[:, 1:, :]       # [1, N_patches, D]
    
    # 可选：reshape 回空间布局
    H_patches = image.height // model.config.patch_size
    W_patches = image.width  // model.config.patch_size
    patch_feats_spatial = patch_feats.reshape(1, H_patches, W_patches, -1)
    # [1, H_patches, W_patches, D]
    
    return cls_token, patch_feats

# ==== OpenVLA 风格的双编码器融合 ====
from transformers import SiglipModel

siglip = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384")
dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-large")

def openvla_style_fuse(image: Image.Image, proj_out_dim: int = 4096):
    """
    OpenVLA 风格：SigLIP 全局 + DINOv2 密集 → 融合投影
    """
    # Step 1: SigLIP 全局特征
    sig_out = siglip.get_image_features(image)              # [1, D_sig]
    
    # Step 2: DINOv2 密集特征 → 全局池化
    _, dino_patches = extract_dinov2_features(image)        # [1, N, D_dino]
    dino_global = dino_patches.mean(dim=1)                  # [1, D_dino]
    
    # Step 3: 拼接 + 投影
    fused = torch.cat([sig_out, dino_global], dim=-1)       # [1, D_sig + D_dino]
    proj  = torch.nn.Linear(fused.shape[-1], proj_out_dim)
    visual_tokens = proj(fused)                             # [1, D_llm]
    
    # Step 4: 输入 LLM
    # visual_tokens → concat with text tokens → Llama 2 → action
    return visual_tokens
```

## 关键要点

1. **自监督 > 监督**：DINOv2 不需要任何标注，却学到了比监督预训练更丰富的视觉表示
2. **密集特征是核心**：patch-level 特征保留了空间信息，这是机器人操作的基础
3. **Centering + Sharpening** 是防止特征坍塌的经典设计，理解它们的协同是理解自监督学习的关键
4. **OpenVLA 的双编码器不是 hack**，是有理论依据的设计：全局语义 + 密集空间 = 完整的视觉理解
5. **DINOv2 的涌现能力**（语义分割、密集对应、深度线索）说明自监督学习能学到超越单一任务的结构化知识

[[CLIP与SigLIP对比]] | [[VLA大模型]] | [[PointNet++与3D点云]] | [[3D-Gaussian-Splatting]]
