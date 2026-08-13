---
title: 分层架构-VLM加运动规划
tags: [layered-architecture, vlm, motion-planning, voxposer, rekep, vila-copa, omnimanip]
aliases: [VoxPoser, ReKep, Vila+CoPa, OmniManip]
---

# 分层架构（一）：High-level Task Planner + MotionPlan

> VLM/空间智能 + 机器人 MotionPlanning。任务主体引导的语义+空间关系学习 + 通用运动执行。代表性：VoxPoser / ReKep / Vila+CoPa / OmniManip。

## VoxPoser (李飞飞团队，开山之作)
- 用 VLM 双模态语义+空间对齐，在 LLM 任务指令引导下生成：
  - **Affordance Map**（可供性，操作交互部位）
  - **Constraint Map**（类似碰撞距离场）
- 过程称 **3D Value Map Composition** → 通用 Motion Planning 套件完成运动规划

## ReKep (Relational Keypoint Constraints)
- 关系关键点约束作为任务规划输出
- DINOv2 全局采样候选关键点 → GPT-4o grounding → 生成任务约束（子目标 + Path 约束）描述 keypoint 相对位置/距离
- 约束 → 代价函数 → 优化解算器求末端轨迹
- 大模型常识(任务决策) + 传统优化(高效通用可靠执行) 的典型结合

## Vila+CoPa (清华 高阳团队)
- ViLa：高层任务规划（基于 GPT-4V，首个 GPT-4V 具身应用，long-horizon 远超 RL/SayCan）
- CoPa：低层子任务执行（基于目标位姿空间约束）
- 三模块：
  1. **Grounding**：VLM 常识 + SoM 语义分割，由粗到细选出物体/Part
  2. **Task-Oriented Grasping**：GraspNet 生成抓取候选 + Grounding 识别抓取部位 → 筛选
  3. **Task-Aware Motion Planning**：Grounding 识别任务部位 → 建模 3D → VLM 生成空间约束 → 求解器算抓取后姿势

## OmniManip (北大+智元)
- **以对象为中心的表示**，弥合 VLM 高层推理与操作低层精度的差距
- 对象规范空间（功能 affordance）→ 交互原语（关键点/方向）→ 桥接 VLM 常识 → 3D 空间约束
- VFM+VLM：比 ReKep 多一步 3D AIGC + 6D 位姿估计抽象对象主体
- **双闭环**：高层（primitive 重采样 + 交互渲染 → VLM 规划闭环）+ 低层（6D 姿态跟踪 → 轨迹优化执行闭环）
- 无需微调即可 zero-shot 泛化

## 路线归类：task-oriented vs object-oriented

| | task-oriented policy learning | object-oriented constraint learning |
|---|---|---|
| 学习主体 | 抓取策略/技能 | 物体空间约束关系 |
| 泛化考察 | 对象+本体的泛化 | 任务类别+空间拓扑的泛化 |
| 适合 | SkillSet 技能模型 | TaskPlan 生成 |
| 代表 | CoPa/OmniManip抓取部分 | ReKep/OmniManip约束部分 |

[[端到端与分层架构之争]] | [[DINOv2与自监督视觉特征]] | [[SE3与SO3空间变换]] | [[轨迹规划-RRT与样条插值]]
