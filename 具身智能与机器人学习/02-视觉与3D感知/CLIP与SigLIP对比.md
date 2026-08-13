---
title: CLIP与SigLIP对比
tags: [vision-language, contrastive-learning, clip, siglip, multimodal]
---

# CLIP 与 SigLIP 对比

## CLIP (OpenAI, 2021)

双塔结构。InfoNCE Loss：$B\times B$ 相似度矩阵 → Softmax归一化 → 交叉熵。

瓶颈：全局归一化导致多卡All-Gather通信巨大；极度依赖大Batch(32,768)。

## SigLIP (Google, 2023)

逐对Sigmoid二分类——每个$(i,j)$ pair独立判断是否匹配：

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^B\sum_{j=1}^B \log\sigma(z_{ij} \cdot (-1)^{\mathbb{I}(i\neq j)})$$

优势：解耦计算（不需全局通信）、小Batch也有效、数据效率更高。

## 对比

|        | CLIP       | SigLIP                        |
| ------ | ---------- | ----------------------------- |
| 损失     | 行/列Softmax | 逐对Sigmoid                     |
| 通信     | 极高         | 极低                            |
| 小Batch | 差          | 稳定                            |
| 现代MLLM | LLaVA 1.5  | PaliGemma, Qwen-VL新版, OpenVLA |

SigLIP 已取代 CLIP 成为主流视觉塔。

[[VLA大模型]] | [[Policy与损失函数]]
