---
title: VQ-VAE与动作Token化
tags: [vq-vae, tokenization, rt-2, openvla, discrete-action]
aliases: [向量量化, 动作离散化]
---

# VQ-VAE 与动作 Token 化

> **核心命题：如何让 VLM 输出连续的机器人动作？答案——把动作变成 Token。**

## 为什么需要离散化动作？

VLA 的本质是 VLM（视觉-语言模型）→ 输出动作。但 VLM 架构（LLaMA、PaLI）本质上是 **离散 Token 自回归预测器**，只能输出离散的 Token ID。机器人动作却是连续的浮点数向量（如 7 维 EEF 位姿：$\Delta x, \Delta y, \Delta z, \Delta roll, \Delta pitch, \Delta yaw, gripper$）。要让 VLM 输出动作，必须把连续动作"翻译"成离散 Token。

这就是 VQ-VAE 在 RT-2/OpenVLA 中的核心作用——**不是生成图像，而是充当连续动作到离散 Token 的桥梁**。

## VQ-VAE 架构

标准 VQ-VAE（van den Oord, NeurIPS 2017）包含三个核心组件：

```
连续输入 x (如 7维动作向量)
       ↓
   [Encoder]  →  z_e = f_φ(x)   编码到连续潜空间
       ↓
   [Codebook] →  e_k = argmin_k ‖z_e - e_k‖₂  最近邻查找→离散索引k
       ↓
   [Decoder]  →  x̂ = g_θ(e_k)   从码本向量解码重建
```

### Codebook（码本）

码本是一组可学习的嵌入向量：

$$e \in \mathbb{R}^{K \times D}$$

- $K$：码本大小（词汇量），典型值 512~8192
- $D$：每个码向量的维度，典型值 64~512

最近邻量化（最邻近查询）：

$$k = \arg\min_j \| z_e(x) - e_j \|_2$$

量化后的潜变量：

$$z_q(x) = e_k$$

### VQ-VAE 损失函数

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \underbrace{\|\text{sg}[z_e] - e\|_2^2}_{\text{Codebook Loss}} + \underbrace{\beta\|z_e - \text{sg}[e]\|_2^2}_{\text{Commitment Loss}}$$

**逐项解释：**

| 项 | 含义 | 更新谁 | 目的 |
|---|---|---|---|
| $\mathcal{L}_{\text{recon}}$ | 重建损失 | Encoder + Decoder | 保证输出 $x̂ ≈ x$ |
| $\|\text{sg}[z_e]-e\|^2$ | Codebook Loss | **仅 Codebook** | 让码本向量靠近 Encoder 输出 |
| $\beta\|z_e-\text{sg}[e]\|^2$ | Commitment Loss | **仅 Encoder** | 让 Encoder 输出不要远离码本向量 |

**`sg`（stop gradient，停止梯度）的关键作用**：

- $\text{sg}[z_e]$：计算 Codebook Loss 时，固定 $z_e$，梯度不回传到 Encoder，只更新 Codebook
- $\text{sg}[e]$：计算 Commitment Loss 时，固定 $e$，梯度不回传到 Codebook，只更新 Encoder

这解决了 $\arg\min$ 算子不可导的问题——Codebook 和 Encoder **轮流靠近对方**，避免两方同时移动导致训练不稳定。

## 直通估计器（Straight-Through Estimator, STE）

$\arg\min$ 的梯度为 0（或无穷），无法反向传播。

**VQ-VAE 的解决方案**：

$$z_q = z_e + \text{sg}[e_k - z_e]$$

- 前向传播：$z_q = e_k$（实际使用码本向量）
- 反向传播：梯度直接从 Decoder 传入 Encoder（Bypass 量化操作）

**直观理解**：前向使用量化的精确结果，反向假设量化操作"不存在"，梯度抄近路直通 Encoder。这就是 "Straight-Through" 的名字来源。

代码实现本质：
```python
# 前向：z_q = e_k
# 反向：∂L/∂z_e = ∂L/∂z_q （直接复制梯度）
z_q = z_e + (e_k - z_e).detach()  # .detach() = sg
```

## Codebook 坍塌（Codebook Collapse）

**问题**：训练中大部分码本向量"死亡"——永远不被选中，码本利用率从 $K$ 退化到 $\ll K$（例如 $512 \rightarrow 20$）。

**原因**：
- Encoder 输出集中到少数几个模式 → 码本少数向量高频更新、多数永不激活
- 初始化不良 → 部分向量一开始就远离数据分布

**解决方案**：

| 方法 | 原理 |
|---|---|
| **EMA 更新码本** | 用指数移动平均更新码本向量，替代梯度下降。缓解部分向量"暂停更新"问题 |
| **Codebook Reset** | 定期将长期未使用的码本向量重置为当前 Encoder 输出的随机样本 |
| **分治码本** | FSQ（Finite Scalar Quantization）：直接舍入到离散 bin，不用查表，天然无坍塌 |
| **降低 Commitment Loss 权重** | $\beta$ 从 0.25 起步 |

EMA 更新公式（$\lambda \in [0.99, 0.999]$）：

$$e_k \leftarrow \lambda \cdot e_k + (1-\lambda) \cdot \mathbb{E}[z_e \mid \text{分配给} e_k]$$

## PyTorch 伪代码：VQ-VAE 前向传播

```python
class VectorQuantizer(nn.Module):
    def __init__(self, K, D, beta=0.25):
        self.codebook = nn.Embedding(K, D)  # K×D
        self.beta = beta

    def forward(self, z_e):
        # z_e: [B, D]  编码器输出
        # Step 1: 计算距离矩阵（最近邻查找）
        # ‖z_e - e‖² = ‖z_e‖² + ‖e‖² - 2·z_e·eᵀ
        dist = (
            (z_e ** 2).sum(dim=1, keepdim=True)   # [B, 1]
            + (self.codebook.weight ** 2).sum(dim=1)  # [K]
            - 2 * z_e @ self.codebook.weight.T      # [B, K]
        )

        # Step 2: argmin → 编码索引
        indices = dist.argmin(dim=1)  # [B]

        # Step 3: 查表获取量化向量
        z_q = self.codebook(indices)  # [B, D]

        # Step 4: 损失计算
        codebook_loss = F.mse_loss(z_q.detach(), z_e)       # ‖sg[z_e]-e‖²
        commitment_loss = F.mse_loss(z_q, z_e.detach())     # ‖z_e-sg[e]‖²
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Step 5: Straight-Through Estimator
        z_q_ste = z_e + (z_q - z_e).detach()

        return z_q_ste, vq_loss, indices
```

## RT-2 的动作 Token 化

RT-2（Google DeepMind, 2023）的做法：**不训练 VQ-VAE，而是直接做均匀分箱离散化**。

### 离散化方案

将一个 7 维动作向量 $a = [\Delta x, \Delta y, \Delta z, \Delta roll, \Delta pitch, \Delta yaw, gripper] \in \mathbb{R}^7$ 离散化：

1. 每个维度独立分成 **256 个均匀 bin**
2. 每个 bin 编号为 0~255（$2^8$，对应一个 8-bit 整数）
3. 每个 bin 对应一个 **离散 Token**
4. 7 个维度 → 7 个 Token

```
动作向量 a ∈ ℝ⁷
    ↓ 逐维度离散化
Token序列: [t₁, t₂, t₃, t₄, t₅, t₆, t₇]
其中 tᵢ ∈ {0, 1, ..., 255}
```

### 为什么是 256 个 bin？

- 256 = $2^8$，恰好一个字节，是计算机自然的量化粒度
- 对机器人控制精度足够：假设平移范围 ±0.1m，256 bin 的分辨率 ≈ 0.78mm
- 与 VLM 的词汇量兼容：32000 token 词汇 + 256×7 = 1792 个额外动作 token，几乎不影响词汇分布

### 动作 Token 注入 VLM 词汇表

RT-2 在 PaLI-X / PaLM-E 的基础上，给词汇表增加 **1792 个特殊动作 token**（256 bins × 7 dims = 1792）：

```
原始词汇: [the, robot, move, ..., <eos>]
动作token: [<a_x_0>, <a_x_1>, ..., <a_x_255>,  ← Δx
            <a_y_0>, ..., <a_y_255>,           ← Δy
            ...
            <grip_0>, ..., <grip_255>]         ← gripper
```

VLM 在自回归生成时，像预测下一个单词一样预测下一个动作 token：

$$P(\text{action} \mid \text{image}, \text{instruction}) = \prod_{i=1}^{7} P(t_i \mid t_1, ..., t_{i-1}, \text{image}, \text{instruction})$$

### 为什么离散化让 VLM 能输出动作？

**本质上利用了 LLM 自回归生成的天性**：

1. LLM 已经被训练成"给定上文，预测下一个 token"的强大引擎
2. 把动作变成 token 后，预测动作 = 预测下一个 token，这正是 LLM 的强项
3. **Co-fine-tuning 的关键作用**：用互联网数据（图文理解）+ 机器人数据（动作预测）联合微调，LLM 可以同时理解"拿起香蕉"的语义和"要输出哪些动作 token 才能做到"
4. 涌现能力：训练后 RT-2 能理解"捡起快要掉落的物体"这种从未在机器人数据中出现的抽象指令——因为 LLM 从互联网文本中学到了这个概念，只需学会映射到动作 token 即可

## OpenVLA 的适配

OpenVLA（Stanford, 2024）在 RT-2 的基础上做了改进：

| 对比项 | RT-2 | OpenVLA |
|---|---|---|
| 离散化方式 | 256 均匀 bin | 256 均匀 bin |
| 动作维度 | 7 维（EEF 绝对位姿） | 7 维（EEF 相对增量） |
| Token 数量 | 1792 | 1792 |
| 离散化是否训练 | 否（固定 bin） | 否（固定 bin） |
| 基座 VLM | PaLI-X / PaLM-E (55B) | Llama2 7B + SigLIP + DINOv2 |
| 动作头 | 直接 token 预测 | 额外动作 MLP 头 + LoRA 微调 |

OpenVLA 的核心改进：用 **LoRA 微调** 替代全参数微调，使得 7B 模型在单 GPU 上可用，且保持与 RT-2 同级别的泛化能力。

## 连续动作 vs 离散动作

| | 连续动作（ACT / Diffusion Policy） | 离散动作（RT-2 / OpenVLA） |
|---|---|---|
| 表示 | 浮点数向量 $\mathbf{a} \in \mathbb{R}^d$ | Token 序列 $t_1,...,t_d$ |
| 生成引擎 | CVAE / Diffusion / Flow Matching | LLM 自回归 |
| 分辨率 | 理论上无限（浮点精度） | 256 bin / 维 |
| 语言理解 | 无 | 强（从 LLM 继承） |
| 指令泛化 | 弱（需训练数据覆盖） | 强（涌现理解抽象指令） |
| 推理速度 | < 5ms（CVAE） | 50-200ms（LLM 自回归） |
| 典型工作 | ACT, Diffusion Policy, π₀ | RT-2, OpenVLA, RoboFlamingo |

**关键取舍**：离散化允许利用 LLM 的语言理解和泛化能力，代价是量化精度损失和推理速度。但随着更高效的 LLM 推理（vLLM, TensorRT-LLM）和流式解码，延迟正快速缩小。

## 高级话题：FSQ（Finite Scalar Quantization）

FSQ 是 2023 年提出的替代方案，避免码本坍塌：

- 直接对每个潜变量维度做 **舍入到有限层级**（如 $\{-3, -2, -1, 0, 1, 2, 3\}$）
- 无需维护码本，不会坍塌
- 表示能力由舍入层级数决定：$L$ 层级 × $d$ 维 = $L^d$ 种组合
- 在图像 VQ-VAE 中表现与标准码本相当甚至更好

FSQ 思路与 RT-2 的均匀分箱其实异曲同工——都在说一件事：**与其学码本，不如直接分箱**。

## 总结

| 问题 | VQ-VAE 的答案 |
|---|---|
| 连续动作如何变成 Token？ | Codebook 量化：$a \to z_e \to e_k$（Token ID） |
| 梯度如何通过 argmin？ | Straight-Through Estimator：反向复制梯度 |
| 码本不用怎么办？ | EMA 更新 / Codebook Reset / 改用 FSQ |
| RT-2 的简化做法 | 不学 Codebook，直接 256 均匀 bin → Token |
| 为什么有效？ | 让 LLM 用自回归的天性去预测动作 |

**一句话总结：VQ-VAE 的核心思想——用 Codebook 做连续与离散的桥梁。RT-2 把这个思想简化到极致（直接分箱不用学），证明了一个反直觉的事实：对机器人控制来说，256 级离散化已经足够好，换取的语言理解和泛化能力才是真金白银。**

[[VLA大模型]] | [[CVAE条件变分自编码器]] | [[动作空间表征总览]] | [[VAE与KL散度深入理解]]
