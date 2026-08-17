---
title: Policy与损失函数
tags: [pytorch, softmax, cross-entropy, policy-learning, loss-function]
aliases: [Softmax NLLLoss CrossEntropyLoss, 损失函数, 策略梯度]
created: 2026-08-13
---

# Policy 与损失函数

> **定位**：数学地基的应用层——**策略学习本质上就是最小化某个损失**。本篇分两部分：① 机器人学习里最常用的几类损失（MSE/L1/Huber/交叉熵）；② 策略梯度目标函数（RL 的"损失"）。并把它们对接到 ACT/Diffusion/PPO 的具体实现。
>
> 前置：[[KL散度推导]]。关联：[[强化学习基础-贝尔曼方程与动态规划]] [[强化学习策略-PPO与SAC]]。

## 一、损失函数全景：三个家族

| 家族 | 代表 | 适用 | 机器人用例 |
|------|------|------|-----------|
| **回归（连续值）** | MSE / L1 / Huber | 预测连续动作、位姿 | ACT 用 L1、扩散用 MSE、位姿回归用 L1 |
| **分类（离散值）** | 交叉熵 / BCE | 预测离散类别、Token | VLA 动作 Token 化后的自回归 |
| **分布距离** | KL | 对齐两个分布 | VAE/ACT 隐变量正则、RL 策略约束 |

## 二、回归损失：MSE vs L1 vs Huber

| 损失 | 公式 | 对离群点 | 特点 |
|------|------|---------|------|
| MSE | $\frac{1}{n}\sum(y-\hat y)^2$ | 敏感（平方放大） | 平滑可导、**扩散模型默认** |
| L1 | $\frac{1}{n}\sum\|y-\hat y\|$ | 稳健 | 不放大异常、**ACT/位姿回归常用** |
| Huber | 分段（小误差 MSE、大误差 L1） | 折中 | 兼顾平滑与稳健 |

**为什么扩散用 MSE**：去噪目标是"预测噪声 $\epsilon$"，噪声本身是零均值高斯，MSE 恰好是它的最大似然（等价于 Score Matching 损失，见 [[Score-Matching与SDE]]）。

**为什么 ACT 用 L1**：动作是关节角，L1 对偶发大偏差更稳健，不易被个别帧带偏。

## 三、分类损失：Softmax / NLLLoss / CrossEntropy（拆解）

**核心等价**：$\text{CrossEntropyLoss}(z,y) = \text{NLLLoss}(\text{LogSoftmax}(z), y)$

| 步骤 | 公式 | 说明 |
|------|------|------|
| Softmax | $P(i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | 把 logits 变成概率 |
| LogSoftmax | $\log P(i) = z_i - \log\sum_j e^{z_j}$ | Log-Sum-Exp 防数值溢出 |
| NLLLoss | $-\log P(y)$ | **自身不算 log，仅索引+取负** |
| CrossEntropyLoss | $-(z_y - \log\sum_j e^{z_j})$ | 上面三步融合 |

**常见误解**：NLLLoss 名字带 "Log" 但**不计算对数**——唯一一次 log 在 LogSoftmax 里。

```python
# ❌ 误区：两层 log
loss = -math.log(math.log(softmax(z)[y]))
# ✅ 实际：一层 log
loss = -F.log_softmax(z, dim=0)[y]
```

**工程建议**：直接用 `nn.CrossEntropyLoss`（底层算子融合）。**不要在最后一层加 Softmax！**

### 交叉熵与 KL 的关系

单热标签 $y$ 下，交叉熵 $=-\log P(y)$；它与 KL 只差一个常数（真实分布的熵），所以"最小化交叉熵 = 最小化预测分布与真实分布的 KL"（见 [[KL散度推导]] 第五节）。

## 四、策略梯度：RL 的"损失"（其实是目标）

RL 的目标是最大化期望回报 $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[\sum_t\gamma^t r_t]$。**策略梯度定理**给出它的梯度：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t \nabla_\theta\log\pi_\theta(a_t|s_t)\cdot\hat{A}_t\right]$$

- $\hat{A}_t$：优势函数（这个动作比平均水平好多少）
- **直觉**：好动作（$A_t>0$）→ 提高概率；坏动作（$A_t<0$）→ 降低概率
- 实际实现里写成"负对数概率 × 优势"当 loss 来最小化

> 注意：策略梯度**不是**监督学习的损失——它没有"正确标签"，只有"回报高低"作为加权信号。这是 RL 与模仿学习的根本区别（见 [[强化学习基础-贝尔曼方程与动态规划]]）。

## 五、三大策略模型各自最小化什么

| 模型 | 损失/目标 | 直觉 |
|------|----------|------|
| ACT | L1 重建 + β·KL（隐变量正则） | 学动作 + 压隐空间（见 [[ACT算法详解]]） |
| Diffusion Policy | 去噪 MSE（预测噪声） | 学"从噪声回到动作"（见 [[Diffusion-Policy详解]]） |
| PPO | Clip 后的策略梯度（负 log 概率 × 优势） | 稳定地"好动作加概率"（见 [[强化学习策略-PPO与SAC]]） |

## 检查点

1. ✅ 说清 MSE/L1/Huber 各自何时用、为什么扩散用 MSE 而 ACT 用 L1
2. ✅ 解释"CrossEntropyLoss = NLLLoss(LogSoftmax)"，指出常见 double-log 误区
3. ✅ 说清策略梯度与监督学习损失的本质区别

## 前置 / 关联

前置：[[KL散度推导]]
关联：[[强化学习基础-贝尔曼方程与动态规划]] | [[强化学习策略-PPO与SAC]] | [[ACT算法详解]] | [[Diffusion-Policy详解]] | [[Score-Matching与SDE]] | [[CLIP与SigLIP对比]]
