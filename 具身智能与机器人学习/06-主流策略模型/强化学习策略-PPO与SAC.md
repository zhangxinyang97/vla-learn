---
title: 强化学习策略-PPO与SAC
tags: [reinforcement-learning, ppo, sac, rlpd, robotics]
aliases: [PPO, SAC, 强化学习策略]
---

# 强化学习策略：PPO、SAC 与 RLPD

模仿学习学"怎么做"，强化学习学"怎样做得更好"——两者互补。Isaac Gym/Lab 大规模并行 RL 训练是具身智能的核心工程范式。

> 本篇是 **L1 RL 地基**的应用篇。值函数与动态规划的系统教学见 [[强化学习基础-贝尔曼方程与动态规划]]，Q-learning/DQN 见 [[强化学习基础-Q-learning与DQN]]——先吃透那两篇再看本片效果最佳。

## 1. 强化学习基础

### 1.1 MDP 五元组

$$(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$

| 符号 | 含义 | 机器人场景示例 |
|------|------|--------------|
| $\mathcal{S}$ | 状态空间 | 关节角 + 本体速度 + 目标位姿 |
| $\mathcal{A}$ | 动作空间 | 各关节力矩/目标位置/速度指令 |
| $\mathcal{P}(s'|s,a)$ | 状态转移概率 | 仿真物理引擎前向模拟 |
| $\mathcal{R}(s,a)$ | 即时奖励 | 离目标越近奖励越高、能耗惩罚 |
| $\gamma \in [0,1]$ | 折扣因子 | 通常 0.99，远期奖励指数衰减 |

### 1.2 策略梯度

目标：最大化期望累积回报 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t r_t]$

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t\right]$$

- $\pi_\theta(a|s)$：参数化策略，输出动作分布
- $\hat{A}_t$：优势函数估计（动作比平均水平好多少）
- 直觉：好动作（$A_t>0$）↑概率，坏动作（$A_t<0$）↓概率

### 1.3 Actor-Critic 架构

```
状态 s_t ──→ [Actor 网络 π_θ]  ──→ 动作 a_t ──→ 环境 ──→ r_t, s_{t+1}
              │
              └──→ [Critic 网络 V_φ] ──→ V(s_t)   ← 监督信号：TD误差 δ_t
```

Actor 负责决策（选动作），Critic 负责评价（算优势），两者协同训练。

---

## 2. PPO（Proximal Policy Optimization）

OpenAI, 2017。TRPO 的简化版——**Clip 代替 KL 约束**，工程友好，Isaac Gym 默认算法。

### 2.1 Clipped Surrogate Objective

PPO 核心：限制策略更新幅度，防止一步更新过大导致训练崩溃。

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \cdot \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot \hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是新旧策略的概率比。

| 参数 | 含义 | 典型值 |
|------|------|--------|
| $\epsilon$ | Clip 范围 | 0.1~0.2 |
| $r_t(\theta)$ | 概率比（>1 表示新策略更倾向该动作） | — |
| $\hat{A}_t$ | 优势估计（正=好动作，负=坏动作） | — |

**直觉**：
- 优势为正时：$r_t$ 被 clip 到 ≤ 1+ε，防止对该动作过度自信
- 优势为负时：$r_t$ 被 clip 到 ≥ 1-ε，防止对该动作过度惩罚
- 取 min 保证：只在新策略确实更好时提升目标，否则截断

### 2.2 GAE（Generalized Advantage Estimation）

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差。

| 参数 | 含义 | 典型值 |
|------|------|--------|
| $\gamma$ | 折扣因子 | 0.99 |
| $\lambda$ | GAE 衰减因子，平衡偏差-方差 | 0.95 |
| $\lambda=0$ | 1步 TD（低方差高偏差） | — |
| $\lambda=1$ | Monte Carlo（无偏高方差） | — |

### 2.3 PPO 总损失

$$\mathcal{L}_{\text{PPO}} = \mathcal{L}^{\text{CLIP}} - c_1 \cdot \mathcal{L}^{\text{VF}} + c_2 \cdot S[\pi_\theta]$$

- $\mathcal{L}^{\text{VF}} = (V_\phi(s_t) - V_t^{\text{target}})^2$：Critic 价值函数损失
- $S[\pi_\theta]$：策略熵奖励，鼓励探索（防止过早收敛到次优确定性策略）
- $c_1, c_2$：损失权重系数

### 2.4 PyTorch 伪代码

```python
# PPO Clipped Loss
def ppo_loss(actor, critic, old_log_probs, states, actions, advantages, returns, epsilon=0.2):
    # Actor loss (clipped surrogate)
    new_log_probs = actor(states).log_prob(actions)
    ratio = (new_log_probs - old_log_probs).exp()  # r_t(θ)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - epsilon, 1 + epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    # Critic loss (value function)
    values = critic(states)
    critic_loss = F.mse_loss(values, returns)

    # Entropy bonus (鼓励探索)
    entropy = actor(states).entropy().mean()

    return actor_loss + 0.5 * critic_loss - 0.01 * entropy
```

---

## 3. SAC（Soft Actor-Critic）

UC Berkeley, 2018。Off-policy 最大熵 RL——**最大化奖励的同时最大化策略熵**，样本效率远高于 PPO。

### 3.1 最大熵目标

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi\left[\sum_t \gamma^t (r_t + \alpha \cdot \mathcal{H}(\pi(\cdot|s_t)))\right]$$

| 参数 | 含义 |
|------|------|
| $\alpha$ | 温度系数，控制熵权重（探索程度） |
| $\mathcal{H}(\pi)$ | 策略熵 $= -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$ |
| $\alpha$ 大 | 强随机性，更多探索 |
| $\alpha$ 小 | 趋近确定性策略 |

### 3.2 Soft Q-Learning

SAC 维护两个 Q 网络（取 min 抑制过高估计）+ 两个 Target Q 网络（软更新）：

$$\mathcal{L}_Q(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[(Q_\phi(s,a) - (r + \gamma \cdot \bar{V}(s')))^2\right]$$

其中 Soft Value Target：

$$\bar{V}(s') = \mathbb{E}_{a' \sim \pi_\theta}\left[\min_{i=1,2} Q_{\bar{\phi}_i}(s',a') - \alpha \log \pi_\theta(a'|s')\right]$$

### 3.3 策略更新

$$\mathcal{L}_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}}\left[\alpha \log \pi_\theta(a|s) - \min_{i=1,2} Q_{\phi_i}(s,a)\right],\ a \sim \pi_\theta(s)$$

使用重参数化技巧（Reparameterization Trick）：$a = f_\theta(s, \xi),\ \xi \sim \mathcal{N}(0,I)$ 使梯度可回传。

### 3.4 自动温度调节

$$\mathcal{L}(\alpha) = \mathbb{E}_{a \sim \pi}\left[-\alpha \log \pi(a|s) - \alpha \cdot \bar{\mathcal{H}}\right]$$

- $\bar{\mathcal{H}}$：目标熵（通常设为 $-\dim(\mathcal{A})$）
- 自动调节：熵不够→↑α 增加探索，熵过剩→↓α 趋向利用

### 3.5 PyTorch 伪代码

```python
# SAC Entropy-Regularized Loss
def sac_loss(actor, critic1, critic2, states, actions, rewards, next_states, dones,
             alpha, gamma=0.99):
    # Critic loss (soft Q-learning)
    with torch.no_grad():
        next_actions, next_log_probs = actor.sample(next_states)
        q_next = torch.min(critic1(next_states, next_actions),
                           critic2(next_states, next_actions))
        q_target = rewards + gamma * (1 - dones) * (q_next - alpha * next_log_probs)
    q1_loss = F.mse_loss(critic1(states, actions), q_target)
    q2_loss = F.mse_loss(critic2(states, actions), q_target)

    # Actor loss (最大化 Q + 熵)
    new_actions, new_log_probs = actor.sample(states)
    q_new = torch.min(critic1(states, new_actions), critic2(states, new_actions))
    actor_loss = (alpha * new_log_probs - q_new).mean()

    return q1_loss + q2_loss, actor_loss
```

---

## 4. RLPD：强化学习 + 示范数据

UC Berkeley, 2023。**在 Replay Buffer 中混合示范数据和在线交互数据**，极大加速 RL 训练。

### 4.1 核心机制

```
Replay Buffer:
┌──────────────────────────────────────┐
│  50% 离线示范数据（遥操作采集）         │
│  50% 在线交互数据（当前策略采集）         │
│  采样策略：均匀采样 → 一定概率仅采样示范   │
└──────────────────────────────────────┘
```

训练时从混合 Buffer 中采样，Q 更新在示范数据上使用 **MC Return** 而非 TD：

$$\mathcal{L}_Q^{\text{demo}} = (Q_\phi(s,a) - G_{\text{demo}})^2$$

其中 $G_{\text{demo}} = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$ 是完整的示范轨迹回报。

### 4.2 效果

- 标准 RL 需要 1B+ 步训练 → RLPD 仅需 50k-100k 步（**10,000x 样本效率提升**）
- 利用示范："好动作"的高价值锚定 → Critic 更快收敛
- SAC + RLPD 是目前具身 RL 的最强组合之一

---

## 5. 强化学习在机器人学中的应用

### 5.1 Causal Chain

```
仿真物理引擎 → 并行环境（万级） → RL训练 → Sim-to-Real → 真机部署
    ↑                                    │
    └─── 奖励函数 ← 人工设计 / LLM生成 ───┘
```

### 5.2 为什么 RL 在机器人学中关键

| 维度 | 说明 |
|------|------|
| **奖励驱动** | 不需要人类示范——奖励函数定义"好"，RL 自己找最优解 |
| **探索能力** | RL 在仿真中可尝试亿级交互，发现人类不会的操控策略 |
| **Sim-to-Real** | Isaac Gym 万级并行 → 数小时训练 → 域随机化 → 零样本真机部署 |
| **端到端优化** | 视觉→动作映射直接用奖励优化，无需行为克隆标签 |
| **持续进化** | 在线 RL 可在真机上持续 fine-tune |

### 5.3 奖励工程

> "Reward is enough." — David Silver

| 奖励类型 | 示例 | 问题 |
|----------|------|------|
| 稠密奖励 | 距离目标指数衰减 | 需精心设计，易诱导局部最优 |
| 稀疏奖励 | 成功标志位=1，否则=0 | 天然无偏但极难学习 |
| 课程奖励 | 逐步提高要求 | 需要人设计课程 |
| LLM 生成 | Eureka 自动写奖励代码 | 新兴方向，可扩展性好 |

---

## 6. 方法对比

### 6.1 PPO vs SAC vs RLPD vs Imitation Learning

| 维度 | PPO | SAC | RLPD | Imitation Learning |
|------|-----|-----|------|-------------------|
| **学习范式** | On-policy RL | Off-policy RL | Off-policy RL + Demo | 监督学习 |
| **数据来源** | 当前策略采样 | Replay Buffer | Buffer + 示范数据 | 人类演示 |
| **样本效率** | 低（M+ 步） | 中（100k+ 步） | **高（50k 步）** | 极高（几百条演示） |
| **探索机制** | Entropy bonus | 最大熵目标 | 最大熵 + 示范引导 | 无需探索 |
| **行为多样性** | 中 | **高** | 中 | 低（模仿给定分布） |
| **超越人类** | ✅ 可能 | ✅ 可能 | ✅ 可能 | ❌ 受限于演示质量 |
| **部署安全** | 低（需大量试错） | 低 | 中（示范提供先验） | **高（已知行为）** |
| **Sim-to-Real** | ✅ 核心应用 | ✅ | ✅ | 中（分布偏移） |
| **工程复杂度** | 低（Isaac Gym 内置） | 中 | 中-高 | 低（行为克隆） |
| **典型场景** | 运动控制、灵巧手 | 操作任务、Sample-efficient | 需少量演示加速 | 遥操作复制、精细操作 |

### 6.2 何时用哪种方法？

```
                    有高质量演示数据？
                   /              \
                 是                否
                 |                 |
          数据量大（1000+）？      仿真中可定义奖励？
         /        \              /           \
       是          否          是             否
       |           |           |              |
   IL(BC)      RLPD       PPO / SAC      ⚠️ 需先搞数据/奖励
  (模仿即可)   (RL+Demo)   (纯RL)         (先回到数据采集)
       |
  需要提升鲁棒性/超越演示？
    → RL 微调（RL fine-tune BC policy）
```

---

## 7. Isaac Gym / Lab 中的 PPO 实践

```python
# Isaac Lab 简化训练循环
env = ManagerBasedRLEnv(cfg)          # 万级并行环境
agent = PPO(actor_net, critic_net)    # Actor-Critic

for iteration in range(max_iterations):
    # 1. 并行采集 rollout（N 个环境 × T 步）
    rollouts = env.step(agent.act(obs))  # GPU 并行

    # 2. 计算 GAE 优势
    advantages = compute_gae(rollouts, gamma=0.99, lam=0.95)

    # 3. 多 epoch PPO 更新
    for epoch in range(K_epochs):
        agent.update(rollouts, advantages, clip_eps=0.2)

    # 4. 周期性 Sim-to-Real 评估
    if iteration % eval_interval == 0:
        deploy_and_test_on_real_robot(agent)
```

> Isaac Gym 的万级并行使 PPO 能在数小时内完成原本需要数周的 RL 训练。

---

[[Diffusion-Policy详解]] | [[ACT算法详解]] | [[NVIDIA-Sim-to-Real技术]] | [[仿真引擎与ROS2部署]] | [[策略与生成的统一趋势]]
