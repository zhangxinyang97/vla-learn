---
title: CLIP与SigLIP对比
tags: [vision-language, contrastive-learning, clip, siglip, multimodal, vla-vision-tower]
aliases: [CLIP, SigLIP, 视觉语言对齐, 对比学习]
---

# CLIP 与 SigLIP：VLA 的「视觉-语言对齐塔」

> **一句话**：CLIP/SigLIP 把「图像」和「文字」编码到**同一个向量空间**，让语义相近的图文对在空间里彼此靠近。VLA 用它们当「眼睛」，让机器人能**看懂场景、听懂指令**。

## 为什么 VLA 需要它？（动机）

机器人策略的输入是「图像 + 语言指令」。要输出正确动作，模型必须先具备两个能力：

1. **图像语义理解**：图里有什么、物体在哪
2. **图文对齐**：语言里的「红色杯子」对应图像里哪个物体

传统做法是分别训练「视觉分类器 + 文本编码器」再拼起来——但它们活在**不同空间**，无法直接比较「文字」和「图像」是否匹配。

CLIP 的革命：**用对比学习把图像和文本拉进同一个嵌入空间**。于是「拿起红色杯子」这句指令，能直接和图像里的红杯子对齐。这就是 VLA 视觉塔的基石。

> **直觉类比**：CLIP 像一本「双语词典」。以前是「中文词典 + 英文词典」两本，查「苹果」和「apple」对不上；CLIP 把它俩编译成**同一个索引**，一查就通。

## 怎么工作（核心机制）

### 双塔结构

```
图像 → [图像塔 ViT]            ──→ 图像向量 v_i ∈ R^d  ┐
                                                      ├→ 余弦相似度矩阵 (B×B)
文本 → [文本塔 Transformer]      ──→ 文本向量 t_j ∈ R^d  ┘
```

- 两个独立编码器，各自输出 **d 维向量**（如 512）
- 匹配的图文对（矩阵对角线）相似度应**高**，不匹配的应**低**

### CLIP 训练目标：InfoNCE

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(v_i \cdot t_i / \tau)}{\sum_{j=1}^{B}\exp(v_i \cdot t_j / \tau)}$$

- 对角线 $v_i \cdot t_i$ 是**正样本对**（图配对应文字），其余 $B-1$ 对是负样本
- 本质是**分类**：在一批负样本里，把正确的文字挑出来
- $\tau$：温度系数，控制分布尖锐度

> 关键点：CLIP 的「对比」是**全局 softmax**——一个 batch 里所有 $B\times B$ 对一起归一化。

### SigLIP 的改进：逐对 Sigmoid

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^B\sum_{j=1}^B \log\sigma\big(z_{ij} \cdot (-1)^{\mathbb{I}(i\neq j)}\big)$$

- 不再全局 softmax，而是**每个 $(i,j)$ 对独立**做二分类：匹配 or 不匹配
- $(-1)^{\mathbb{I}(i\neq j)}$：正样本对 $z_{ij}$ 取正（推向 +∞），负样本取负（推向 -∞）

## 为什么 SigLIP 取代了 CLIP？（工程动机）

| 维度 | CLIP (InfoNCE) | SigLIP (逐对 Sigmoid) |
|---|---|---|
| 归一化 | 全局 softmax | 逐对独立 |
| **通信** | 极高（多卡要 All-Gather 所有特征） | 极低（无需全局） |
| **小 batch** | 差（负样本少，对比学不好） | 稳定 |
| 数据效率 | 低 | 更高 |
| 现代 MLLM | LLaVA 1.5 | PaliGemma、Qwen-VL 新版、OpenVLA |

**根本原因**：CLIP 的 InfoNCE 依赖**大量负样本**，所以需要巨大 batch（32,768）。多卡训练时，全局 softmax 要求每张卡都拿到所有卡的特征，**All-Gather 通信成为瓶颈**。SigLIP 把「全局对比」拆成「逐对二分类」，**解耦计算**，小 batch 也能训，通信量骤降——这就是大模型（PaliGemma 等）纷纷转投 SigLIP 的原因。

## 在 VLA 模型中的角色（在哪用）

| 模型 | 视觉塔 | 说明 |
|---|---|---|
| OpenVLA | **SigLIP + DINOv2 双塔** | SigLIP 给语义对齐，DINOv2 给密集空间特征，两者拼接 |
| π0 | **SigLIP**（PaliGemma 内） | PaliGemma VLM 的视觉编码器就是 SigLIP |
| RT-2 | ViT（PaLI-X 内置） | 谷歌自研 PaLI 视觉塔 |
| LLaVA 1.5 | CLIP | 早期 MLLM 标杆 |

> **为什么 OpenVLA 用双塔**：SigLIP 是「全局语义」（知道这是杯子），但机器人操作需要「像素级空间位置」（杯子在坐标 (x,y,z)、朝向如何）。DINOv2 补上这个缺口——这就是「语义塔 + 空间塔」的组合拳。

## 工程化实现

- **视觉塔冻结**：VLA 微调时通常**冻结视觉塔**（预训练权重已很好），只训投影层 + LLM 的 LoRA，省显存、防过拟合
- **特征投影**：视觉向量先过一层 MLP 投影到 LLM 的 token 维度，再与文本 token 拼接
- **分辨率权衡**：分辨率越高细节越多，但 token 数**平方级增长** → 显存爆炸；OpenVLA 用 ~384×384
- **训练技巧**：温度 $\tau$ 可学习；难负样本挖掘（hard negative mining）提升区分度

## 局限与现状

- **局限**：对比学习学的是「全局语义」，天然丢失**空间细节**（物体具体位置/姿态），所以纯 CLIP/SigLIP 不够，需 DINOv2 等密集特征补充
- **现状**：SigLIP 已是视觉塔主流；CLIP 更多作为历史基准

[[VLA大模型]] | [[DINOv2与自监督视觉特征]] | [[Policy与损失函数]] | [[向量内积与外积]]
