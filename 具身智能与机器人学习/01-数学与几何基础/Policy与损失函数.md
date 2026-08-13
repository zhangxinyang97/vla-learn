---
title: Policy与损失函数
tags: [pytorch, softmax, cross-entropy, policy-learning]
aliases: [Softmax NLLLoss CrossEntropyLoss]
---

# Softmax / NLLLoss / CrossEntropyLoss

**核心等价**：$\text{CrossEntropyLoss}(z,y) = \text{NLLLoss}(\text{LogSoftmax}(z), y)$

## 数学拆解

| 步骤 | 公式 |
|---|---|
| Softmax | $P(i) = e^{z_i} / \sum_j e^{z_j}$ |
| LogSoftmax | $\log P(i) = z_i - \log\sum_j e^{z_j}$ (Log-Sum-Exp防溢出) |
| NLLLoss | $-\log P(y)$ (**自身不计算log，仅索引+取负**) |
| CrossEntropyLoss | 以上三步融合：$-(z_y - \log\sum_j e^{z_j})$ |

## 常见误解

NLLLoss 名字带"Log"但**不计算对数**。唯一一次 log 在 LogSoftmax 中。

```python
# ❌ 误区：两层log
loss = -math.log(math.log(softmax(z)[y]))
# ✅ 实际：一层log
loss = -F.log_softmax(z, dim=0)[y]
```

## 工程建议

直接用 `nn.CrossEntropyLoss`——底层算子融合。**不要在最后一层加 Softmax！**

[[CLIP与SigLIP对比]] | [[KL散度推导]]
