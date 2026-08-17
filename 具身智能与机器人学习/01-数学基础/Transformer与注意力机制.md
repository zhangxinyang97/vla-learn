---
title: Transformer与注意力机制
tags: [transformer, attention, math-foundation, vla, llm, backbone]
aliases: [自注意力, Attention机制, Multi-Head Attention, RoPE]
created: 2026-08-13
---

# Transformer 与注意力机制

> **为什么是基石？** ACT、Diffusion Policy、RDT-1B、OpenVLA、π₀ 的骨干全是 Transformer。不理解注意力，就看不懂 08 模块里"因果 Mask""逐块注意力""双向注意力"这些词的来由。这篇补上 VLA 全系的数学骨架。

## 一、从向量点积到注意力

注意力本质是**带权求和**：用"查询"去匹配一组"键"，按匹配度加权聚合"值"。

[[向量内积与外积]] 里讲过：$\mathbf{q} \cdot \mathbf{k} = |\mathbf{q}||\mathbf{k}|\cos\theta$，点积越大 → 方向越一致 → 匹配度越高。

**Scaled Dot-Product Attention**（Vaswani et al., 2017）：

$$\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

- $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$：查询（Query），"我在找什么"
- $\mathbf{K} \in \mathbb{R}^{m \times d_k}$：键（Key），"我有什么"
- $\mathbf{V} \in \mathbb{R}^{m \times d_v}$：值（Value），"拿什么聚合"
- $n$ = 查询数，$m$ = 键值数（自注意力时 $n=m$ = 序列长度 $L$）
- **除以 $\sqrt{d_k}$**：当 $d_k$ 大时点积方差变大，softmax 梯度消失，缩放保持数值稳定

**复杂度 O(L²·d)**：序列越长越贵，这是所有 Transformer 策略的推理瓶颈（也是动作 chunk 只取 K 步的原因之一）。

## 二、Multi-Head Attention：多视角并行

$$\text{MultiHead}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\mathbf{W}^O$$

- 把 $d$ 维拆成 $h$ 个 $d/h$ 维子空间，每个 head 独立做注意力
- 直觉：不同 head 学不同的关系——有的看空间邻近，有的看颜色，有的看时序
- 在动作序列上：一个 head 管"当前帧与 3 帧前的关系"，一个 head 管"与初始帧的关系"

## 三、残差连接与 LayerNorm

每一层都是：

$$\mathbf{x}' = \mathbf{x} + \text{SubLayer}(\mathbf{x}), \qquad \text{再 LayerNorm}$$

- **残差**：让梯度直达底层，深度网络可训的关键（50+ 层不退化）
- **LayerNorm**：对特征维归一化（对每个 token 独立），稳定训练
- **Pre-Norm vs Post-Norm**：现代模型（LLaMA/OpenVLA）多用 Pre-Norm，训练更稳

## 四、位置编码：给无序注意力注入顺序

注意力本身是**置换不变**的（打乱 token 顺序结果不变）——必须显式编码位置。

| 方案 | 做法 | 谁在用 |
|------|------|--------|
| 正弦绝对位置 | 固定三角函数，外推性差 | 原始 Transformer |
| 可学习位置 | 学一组位置向量 | ViT、Diffusion Policy (1D) |
| **RoPE（旋转位置编码）** | 把位置编码进 Q/K 的旋转矩阵，相对位置衰减，外推好 | **LLaMA 系 → OpenVLA 2 / π0 / 现代 VLA 主流** |
| 3D 位置编码 | 把 3D 坐标编码进 token | 3D-VLA、点云 Transformer |

## 五、因果 Mask vs 双向注意力

- **因果 Mask（causal）**：每个 token 只能看自己和之前 → 自回归预测（RT-2、OpenVLA 的 LLM 骨干）。适合"预测下一个动作 token"。
- **双向注意力（bidirectional）**：全序列互相可见 → 编码器/扩散 Transformer（Octo 的动作头）。适合"看到整段观测后一次性生成动作块"。
- OpenVLA-OFT+ 用双向注意力 + L1 回归替代自回归采样，推理更快（见 [[端到端VLA模型盘点-上]]）。

## 六、推理优化：KV Cache

自回归每步只生成一个新 token，但前面的 Q/K 不必重算——缓存 Key/Value 矩阵。代价是显存随序列增长，这也是 VLA 在真机上延迟的瓶颈来源。

## 七、MoE（混合专家）：稀疏化

$$\mathbf{y} = \sum_{i \in \text{Top-k}(\text{Router}(\mathbf{x}))} g_i(\mathbf{x}) \cdot \mathbf{E}_i(\mathbf{x})$$

- Router 把每个 token 分给 Top-2 个专家网络，参数巨大但每次只激活一小部分
- 用途：**ChatVLA 用 MoE 解耦"理解"与"控制"**（见 [[ChatVLA与训练范式解耦]]），也是 VLA 扩展参数的常见手段

## 八、在具身策略中的角色速查

| 模型 | Transformer 用在哪 | 关键点 |
|------|-------------------|--------|
| ACT | CVAE 的 Transformer Decoder | 一次预测 K=100 帧动作块，块内一致性靠自注意力 |
| Diffusion Policy | 1D 时序 U-Net / ViT 骨干 | 观测条件 + 动作去噪 |
| RT-2 / OpenVLA | LLM 自回归骨干 | 视觉+语言+动作全部 Token 化 |
| RDT-1B | Diffusion Transformer | 多模态输入异构处理 |
| π0 / π0.5 | Flow Matching + Transformer | RoPE + 因果结构，1-10 步 ODE 推理 |

## 前置依赖

[[向量内积与外积]]（QK 点积的几何意义）

## SOTA 关联

[[ACT算法详解]] | [[Diffusion-Policy详解]] | [[VLA大模型]] | [[端到端VLA模型盘点-上]] | [[RDT-1B]] | [[ChatVLA与训练范式解耦]]
