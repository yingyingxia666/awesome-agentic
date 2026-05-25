# Awesome-Agentic

> A curated reading list of large-language-model RL papers, organized by four research directions: **Reasoning RL**, **Agentic RL**, **OPD (Off-Policy / On-Policy Distillation / Drift)**, and **Multi-Agent**.

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Last Update](https://img.shields.io/badge/last%20update-2026.05-brightgreen) ![Papers](https://img.shields.io/badge/papers-200%2B-blue) ![Time Range](https://img.shields.io/badge/time-2023.01--2026.05-orange)

## 📖 仓库简介

本仓库整理 **大模型 / Agent / 多模态推理大模型** 在强化学习方向的代表性论文（≥ 2023.01），覆盖：

- **奖励信号怎么设计**（Reward Modeling）
- **稀疏 outcome 怎么分配回中间步**（Credit Assignment）
- **行为策略与目标策略错位时怎么修正**（Off-Policy）
- **多个 LLM 协作 / 对抗的训练范式**（Multi-Agent）

---

## 🗺️ 方向介绍

本仓库把 LLM RL 的论文分成四大方向：

| 方向 | 关键问题 | 典型论文 |
|------|---------|---------|
| 🧠 **Reasoning RL** | 单轮长 CoT 推理（500–30K tokens），可验证或半可验证任务的奖励与归因 | DeepSeek-R1, GRPO, PRIME, GSPO, DAPO, VAPO |
| 🤖 **Agentic RL** | 多轮、长 horizon、部分可观测；工具调用、GUI、Embodied、网页搜索 | SWE-RL, Search-R1, ToolRL, GiGPO, SWEET-RL, RAGEN, HCAPO |
| 🔄 **OPD** | Off-Policy RL（IS / 异步 / replay / TIM）+ On-Policy Distillation + Drift 监控 | GSPO, MinPRO, M2PO, TML On-Policy Distillation, AReaL, IcePop |
| 👥 **Multi-Agent** | 多个 LLM 协作 / 辩论 / 自博弈 / coordinator 训练 | MAPoRL, MARFT, SPC, Latent Agents, eva, FlowReasoner |

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Reasoning RL   │  │   Agentic RL    │  │      OPD        │  │  Multi-Agent    │
│ (single-turn)   │  │ (multi-turn)    │  │ (off / dist /   │  │ (debate / coop  │
│                 │  │                 │  │  drift)         │  │  / self-play)   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │                    │
   • Reward / PRM       • Tool-use RL        • IS clip 设计        • Co-training
   • Credit Assign      • GUI / Embodied     • Async / Replay      • Debate as RL
   • Long CoT GRPO      • Search / Web       • TIM / Precision     • Self-play
   • Reward Hacking     • Multi-turn CA      • KD / Distillation   • Coordinator
   • Test-time Scale    • Memory / Long      • KL / Drift          • Game-theoretic
                          horizon              monitor
```

---

## 📑 目录

- [1. Reasoning RL](#1-reasoning-rl)
  - [1.1 RLVR 与可验证奖励基础](#11-rlvr-与可验证奖励基础)
  - [1.2 GRPO 谱系与算法工程改造](#12-grpo-谱系与算法工程改造)
  - [1.3 Process Reward Model（PRM）](#13-process-reward-modelprm)
  - [1.4 Token-level Credit Assignment](#14-token-level-credit-assignment)
  - [1.5 Segment-level Credit Assignment](#15-segment-level-credit-assignment)
  - [1.6 Causal / Counterfactual CA & Anti-Reward-Hacking](#16-causal--counterfactual-ca--anti-reward-hacking)
  - [1.7 DPO 步级变体与无显式 RM 路线](#17-dpo-步级变体与无显式-rm-路线)
  - [1.8 Reward Modeling：Generative / Self-Reward / Robust RM](#18-reward-modelinggenerative--self-reward--robust-rm)
  - [1.9 Self-Improvement / Test-time Scaling](#19-self-improvement--test-time-scaling)
  - [1.10 综述与基准](#110-综述与基准)
- [2. Agentic RL](#2-agentic-rl)
  - [2.1 Tool-use / Multi-turn Agent](#21-tool-use--multi-turn-agent)
  - [2.2 Turn-level Credit Assignment](#22-turn-level-credit-assignment)
  - [2.3 Hindsight / Counterfactual Turn CA](#23-hindsight--counterfactual-turn-ca)
  - [2.4 GUI / Embodied / Computer-Use Agent](#24-gui--embodied--computer-use-agent)
  - [2.5 Search / Web / Research Agent](#25-search--web--research-agent)
  - [2.6 Memory & Long-Horizon Agent](#26-memory--long-horizon-agent)
  - [2.7 Code / SWE Agent](#27-code--swe-agent)
  - [2.8 Multimodal Agent RL](#28-multimodal-agent-rl)
  - [2.9 安全与红队 Reward](#29-安全与红队-reward)
  - [2.10 综述与基准](#210-综述与基准)
- [3. OPD（Off-Policy / On-Policy Distillation / Drift）](#3-opdoff-policy--on-policy-distillation--drift)
  - [3.1 Off-Policy RL：IS / Clipping 设计](#31-off-policy-rlis--clipping-设计)
  - [3.2 异步 / Replay / 系统级 Off-Policy](#32-异步--replay--系统级-off-policy)
  - [3.3 训练-推理不匹配（TIM / Precision / MoE）](#33-训练-推理不匹配tim--precision--moe)
  - [3.4 On-Policy Distillation](#34-on-policy-distillation)
  - [3.5 Off-Policy KD 对照](#35-off-policy-kd-对照)
  - [3.6 Policy Drift 监控与缓解](#36-policy-drift-监控与缓解)
  - [3.7 综述与博客](#37-综述与博客)
- [4. Multi-Agent](#4-multi-agent)
  - [4.1 Multi-Agent Co-Training](#41-multi-agent-co-training)
  - [4.2 LLM Debate](#42-llm-debate)
  - [4.3 Cooperative CA / 多 agent 信用分配](#43-cooperative-ca--多-agent-信用分配)
  - [4.4 Self-Play / Game-Theoretic](#44-self-play--game-theoretic)
  - [4.5 LLM-as-Coordinator](#45-llm-as-coordinator)
  - [4.6 综述与基准](#46-综述与基准)

---

## 1. Reasoning RL

> 单轮长 CoT 推理任务（数学、代码、形式化证明、复杂推理）的 RL，关键词：RLVR、PRM、GRPO、长 CoT、process reward、信用分配。

### 1.1 RLVR 与可验证奖励基础

#### Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model (2025-03)
- **简介**：StepFun 公开复现 DeepSeek-R1-Zero 的端到端配方。仅用 vanilla PPO + rule-based reward + 完全去掉 KL penalty，证明 R1-Zero 风格 scaling 在 7B 上同样可达。critic 学到了"对重复响应自动下调 advantage"的行为，等价于隐式起到了稳定作用，因此 KL 不再必要。训练步数仅为 R1-Zero 公开报告的约 1/10。
- **arXiv**：[2503.24290](https://arxiv.org/abs/2503.24290)

#### SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild (2025-03)
- **简介**：港科大与 SJTU 团队对 10 个不同家族（Llama、Qwen、Mistral、DeepSeek、InternLM 等）做 R1-Zero 风格 RLVR 实证。给出"哪些 base 能做 zero-RL，哪些不能"的实证地图，并系统消融 reward 设计、entropy 控制、长度 bias、KL 系数等关键超参，是 RLVR 复现的工程参考手册。
- **arXiv**：[2503.18892](https://arxiv.org/abs/2503.18892)

#### Kimi K1.5: Scaling Reinforcement Learning with LLMs (2025-01)
- **简介**：Moonshot AI 的 RL 后训练技术报告。核心算法是 online policy mirror descent + 强长度惩罚；用 chain-of-thought reward model 替代经典标量 RM，CoT-RM 在 RewardBench 上 acc 98.5% vs 经典 RM 84.4%。保留宽松 KL 项防止 length explosion，提出 long2short 自蒸馏把长链能力压回短链。
- **arXiv**：[2501.12599](https://arxiv.org/abs/2501.12599)

#### DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2025-01)
- **简介**：DeepSeek-AI 的标志性工作，把"长链推理可以纯 RL 学出来"从设想变成现实。R1-Zero 完全不用 KL penalty 与 SFT 冷启动，直接 GRPO + rule-based reward 即涌现 self-reflection、aha-moment 行为；R1 加 cold-start SFT 提升可读性。同时发布 R1-Distill 系列把推理能力 off-policy 蒸馏到 1.5B–70B 学生。
- **arXiv**：[2501.12948](https://arxiv.org/abs/2501.12948)

#### Tülu 3: Pushing Frontiers in Open Language Model Post-Training (2024-11)
- **简介**：AI2 的全栈开源 post-training 配方，首次正式提出 RLVR（Reinforcement Learning with Verifiable Rewards）这一术语。Pipeline 分 SFT → length-normalized DPO → RLVR 三阶段，全部数据、checkpoint、训练脚本开源。Length-normalized DPO 是抗 length-bias drift 的代表实践。
- **arXiv**：[2411.15124](https://arxiv.org/abs/2411.15124)

### 1.2 GRPO 谱系与算法工程改造

#### Magistral (2025-06)
- **简介**：Mistral AI 公开的 reasoning 模型技术报告。完全不依赖蒸馏 trace，纯 RL（GRPO + 改进）+ async generator pipeline 训练，证明小模型（24B）可直接通过 RL 超过 distillation+SFT baseline。ε_clip 调到 0.28（与 DAPO 一致）；自定义 generator/learner 解耦的 async 架构是工业级 RL 的工程参考。
- **arXiv**：[2506.10910](https://arxiv.org/abs/2506.10910)

#### VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks (2025-04)
- **简介**：ByteDance Seed 团队提出的 value-based 长 CoT RL 算法。核心是 length-adaptive GAE：根据 token position 动态调整 GAE 的 λ，让长序列里靠后的 reward 信号也能稳定回传。在 AIME 等长链推理任务上比 critic-free 的 DAPO 高 10+ 分，是 critic-based 路线在长 horizon 重新优于 critic-free 的关键证据。
- **arXiv**：[2504.05118](https://arxiv.org/abs/2504.05118)

#### Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO) (2025-03)
- **简介**：NUS Sea AI Lab 团队系统诊断 GRPO 的两个隐性 bias：response length 越长 advantage 估计越偏、std normalization 让 batch 内难易题之间产生 reward 漏 cross-talk。Dr. GRPO 同时移除 length normalization 与 std normalization，给出无偏的 group baseline，在等价 token budget 下显著提升 efficiency。
- **arXiv**：[2503.20783](https://arxiv.org/abs/2503.20783)

#### DAPO: An Open-Source LLM Reinforcement Learning System at Scale (2025-03)
- **简介**：ByteDance Seed 团队开源的 32B reasoning RL 配方。四件套：Clip-Higher 非对称信任域（ε_high=0.28 / ε_low=0.2）让低概率惊喜 token 上界更宽防止 entropy collapse；token-level loss 替代 sequence-level loss；dynamic sampling 过滤 advantage=0 的无信息 prompt；overlong reward shaping 消除长度截断 bias。AIME 2024 50 → 60。
- **arXiv**：[2503.14476](https://arxiv.org/abs/2503.14476)

#### DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO) (2024-02)
- **简介**：GRPO（Group Relative Policy Optimization）的诞生地。同一 prompt 采 G 个 rollout 用组内均值方差归一化作为 advantage，彻底去掉 critic，显著降低 RL 训练显存与对齐 PPO 的复杂度。后续 R1、DAPO、VAPO、GSPO、Dr.GRPO、PRIME 等所有"群体相对"系算法的母体。
- **arXiv**：[2402.03300](https://arxiv.org/abs/2402.03300)

### 1.3 Process Reward Model（PRM）

#### A Survey of Process Reward Models for Reasoning (2025-10)
- **简介**：当前 PRM 方向最系统的综述（v3 2026-04 持续更新）。把 PRM 工作按"标注来源（人工 / 自动 MC / 隐式）/ 训练目标（分类 / 回归 / 生成）/ 应用领域（数学 / 代码 / agent / 多模态）"三维分类，绘制 PRM 全景图。包含 ProcessBench / PRMBench / VisualProcessBench 等评测基准的横向比较，是新人入门 PRM 的"地图"。
- **arXiv**：[2510.08049](https://arxiv.org/abs/2510.08049)

#### GroundedPRM: Tree-Search-Driven, Tool-Verified Process Reward Modeling for Mathematical Reasoning (2025-10)
- **简介**：用 MCTS 构造结构化推理路径，每步用外部工具（Python / SymPy）验证消除 hallucinated supervision；step-level 与 outcome-level reward 混合 aggregation 给最终 PRM 标签。仅用 40K 样本即可让 ProcessBench 上 +26%，是当前自动 PRM 数据效率最高的工作之一。
- **arXiv**：[2510.14942](https://arxiv.org/abs/2510.14942)

#### GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning (2025-04)
- **简介**：把 PRM 从打分器变成 generator——给定步骤后先生成 critique，再综合给 step reward。Critique 形态显著提升 PRM 的人类可读性与对 reasoning RL 的指导力，并允许在 inference time 用 multi-sample critique 扩 PRM 算力。MATH-500 上 BoN 性能超过同 size 判别式 PRM。
- **arXiv**：[2504.00891](https://arxiv.org/abs/2504.00891)

#### R-PRM: Reasoning-Driven Process Reward Modeling (2025-02)
- **简介**：针对 reasoning trajectory 的专用 PRM 训练范式。在标注 step quality 之外，引入"step-to-step reasoning chain"作辅助 head，让 PRM 学会沿推理依赖图打分而非孤立判断单步。在 ProcessBench 上比同 size 普通 PRM 提升约 5–10 点。
- **arXiv**：[2502.14361](https://arxiv.org/abs/2502.14361)

#### Process Reinforcement through Implicit Rewards (PRIME) (2025-02)
- **简介**：把 Implicit PRM 与 online RL 真正打通。训练时直接用 DPO 隐式 reward = log π_θ/π_ref 当作 step-level advantage，无需 explicit step 标注，与 GRPO 无缝集成。Qwen2.5-Math-7B 在 AIME 24 上 14.0 → 26.7，是隐式过程奖励路线最被关注的工程标杆。
- **arXiv**：[2502.01456](https://arxiv.org/abs/2502.01456)

#### VersaPRM: Multi-Domain Process Reward Model via Synthetic Reasoning Data (2025-02)
- **简介**：通过合成多领域 reasoning 数据训跨域可迁移的 PRM，专门解决"PRM 通常只在数学域 work"的痛点。在 8 个非数学推理 benchmark 上取得 BoN 提升，验证了 PRM 跨域泛化的可能性。
- **arXiv**：[2502.06737](https://arxiv.org/abs/2502.06737)

#### The Lessons of Developing Process Reward Models in Mathematical Reasoning (Qwen2.5-Math-PRM) (2025-01)
- **简介**：Qwen 团队发布的 7B/72B 数学 PRM，附带详细工程经验。系统比较 MC-based vs human-labeled PRM 数据、不同 aggregation 方法（min / mean / last）、训练目标（pointwise vs pairwise）的影响。当前公开数学 PRM 中最强 baseline 之一。
- **arXiv**：[2501.07301](https://arxiv.org/abs/2501.07301)

#### rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking (2025-01)
- **简介**：Microsoft Research 的标志性 PRM 工作。MCTS 自我对弈 + Process Preference Model（PPM，PRM 的 pairwise 版）+ self-evolution 四轮迭代，让 7B 小模型在 MATH 与 AIME 上达到 o1-mini 级别。系统证明 SLM + 高质量 PRM + 多轮 self-evolve 可以追赶大模型推理能力。
- **arXiv**：[2501.04519](https://arxiv.org/abs/2501.04519)

#### Free Process Rewards without Process Labels (Implicit PRM) (2024-12)
- **简介**：UIUC + Tsinghua 的理论奠基工作。证明在 outcome-level reward 上训练的 reward model（即 ORM）自带一个 token-level/step-level 的隐式过程奖励 r̂_t = β·log(π_θ/π_ref)，无需任何 step 标注即可解码出来。这是 PRIME、From r to Q\* 等后续隐式 PRM 路线的理论起点。
- **arXiv**：[2412.01981](https://arxiv.org/abs/2412.01981)

#### VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment (2024-10)
- **简介**：Mila + ServiceNow 的工作。用 MC rollout（每个 step 重 rollout K 次取均值）替代学习的 critic 估 V(s)，得到无偏 step-level value。算力翻倍但 advantage 信号纯净，是后续所有 PRM-free CA 工作的"金标"对照。
- **arXiv**：[2410.01679](https://arxiv.org/abs/2410.01679)

#### Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning (PAV) (2024-10)
- **简介**：Carnegie Mellon + Google 的理论框架——把 process reward 重写为"相对 prefix continuation 的 advantage"：r_t = E[R | s_{≤t+1}] − E[R | s_{≤t}]。这一形式给出隐式 PRM 的最严格定义，是后续 Implicit PRM、From r to Q\* 等工作共同的理论基石。
- **arXiv**：[2410.08146](https://arxiv.org/abs/2410.08146)

#### Improve Mathematical Reasoning in Language Models by Automated Process Supervision (OmegaPRM) (2024-06)
- **简介**：Google DeepMind 提出的自动 PRM 数据 de facto 标准。用 divide-and-conquer 风格的 MCTS 把每条 trajectory 切成 step，根据该 step 后续 rollout 的成功率自动赋 step label，生成 1.5M+ 数学 step-level 数据。后续 VisualPRM / MM-PRM 等多模态 PRM 直接复用其框架。
- **arXiv**：[2406.06592](https://arxiv.org/abs/2406.06592)

#### ReST-MCTS\*: LLM Self-Training via Process Reward Guided Tree Search (2024-06)
- **简介**：清华团队首批把 process reward 与树搜索深度结合的工作。ReST 自训练 + MCTS 搜索 + tree-based PRM 三者闭环：MCTS 用 PRM 引导搜索得到优质 trajectory，trajectory 又用于训练 next-iter PRM。LLaMA3-8B 上 GSM8K 79.6 → 86.3。
- **arXiv**：[2406.03816](https://arxiv.org/abs/2406.03816)

#### From r to Q\*: Your Language Model is Secretly a Q-Function (2024-04)
- **简介**：Stanford 团队的理论突破。把 DPO 学到的隐式 reward β·log(π_θ/π_ref) 重新解释为 token-level Q-function，给出 RL 与 preference learning 的统一视角。为 RM、PRM、token-level CA 三者的统一提供严格理论入口。
- **arXiv**：[2404.12358](https://arxiv.org/abs/2404.12358)

#### AlphaMath Almost Zero: process Supervision without process (2024-05)
- **简介**：从 outcome supervision 直接派生伪过程监督——给同一 prompt 多 rollout，按 final answer 正确率反推每一步的 value，再训 value head 与 value-guided decoding。完全不需要 step-level 人工标注，是"几乎零过程标注"PRM 路线的代表。
- **arXiv**：[2405.03553](https://arxiv.org/abs/2405.03553)

#### Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing (AlphaLLM) (2024-04)
- **简介**：把 AlphaGo 的 MCTS + value/policy iteration 完整搬到 LLM 推理。Imagination（self-play 生成新问题）+ Searching（MCTS 找解）+ Criticizing（PRM 评估每步）三角色循环。GSM8K 57.8 → 92.0，MATH 20.7 → 51.0，是早期 MCTS-LLM 路线的标志工作。
- **arXiv**：[2404.12253](https://arxiv.org/abs/2404.12253)

#### Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations (2023-12)
- **简介**：Peking University + DeepSeek 的早期工作，把 PRM 从必须人工标 step 解放出来。每个 step 的 label 通过其后续 rollout 是否最终正确自动派生（completion-based 自动标注），训练得到的 PRM 在 MATH/GSM8K 上比手工标注 PRM 更稳定且数据扩展性更好。
- **arXiv**：[2312.08935](https://arxiv.org/abs/2312.08935)

#### Let's Verify Step by Step (PRM800K) (2023-05)
- **简介**：OpenAI 发布的首个大规模人工 step 标注 PRM 数据集（800K step 级标签）。系统证明 process supervision 在 MATH 上显著优于 outcome supervision（比 ORM 高 10+ 点 BoN），并开源整个标注流程。是 step-level reward 路线的开山之作。
- **arXiv**：[2305.20050](https://arxiv.org/abs/2305.20050)

### 1.4 Token-level Credit Assignment

#### Token Entropy Policy Optimization (TEPO) (2025-10)
- **简介**：用 Markov 似然把 GRPO 的组级 outcome reward 重组为 token 聚合形式，token 条件似然乘积差异化分配 advantage。高熵 token 得到更大梯度，低熵 token 被弱化，缓解长 CoT 训练中 entropy 单调下降至 collapse 的失败模式。
- **arXiv**：[2510.09369](https://arxiv.org/abs/2510.09369)

#### DCPO: Dynamic Clipping Policy Optimization (2025-09)
- **简介**：百川提出。按 token 先验概率自适应调节 PPO clipping 上下界——稀有 token 上界宽鼓励探索、常见 token 上界紧防止 collapse；同时引入 cumulative-step advantage smoothing 显著提升非零 advantage 比例。在 7B Qwen 上对 GSM8K 与 MATH 上稳定优于 GRPO。
- **arXiv**：[2509.02333](https://arxiv.org/abs/2509.02333)

#### GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy (2025-08)
- **简介**：ByteDance 提出。把 token entropy 作 reward 调节器：高熵 token = 关键决策点，给更大梯度；advantage_t = A_seq · (H_t / Σ H_k)。同时给出 token-level（GTPO）与 sequence-level（GRPO-S）两种实现，统一了"按熵塑形 reward"的设计空间。
- **arXiv**：[2508.04349](https://arxiv.org/abs/2508.04349)

#### Token-level Direct Preference Optimization with Reward (TGDPO) (2025-06)
- **简介**：把序列级 PPO 解析地分解为 token-level PPO，closed-form 推出 token-level 最优 policy 与对应 token reward；据此构造 token reward-guided DPO loss。相比 v1 中 TDPO 仅做 token-level KL，TGDPO 真正引入 token-level reward 信号，在数学推理上稳定优于 DPO/TDPO。ICML 2025。
- **arXiv**：[2506.14574](https://arxiv.org/abs/2506.14574)

#### Selective Preference Optimization via Token-Level Selective Self-Play (T-SPMO) (2025-04)
- **简介**：把 GRPO 的 group baseline 与 token-level prefix matching 结合，做 LoRA 友好的 critic-free token CA。在 critic-free 与 LoRA 训练资源极紧的场景下，SVAMP 准确率 46 → 70%，是小算力场景的实用 token-level CA 配方。
- **arXiv**：[2504.20834](https://arxiv.org/abs/2504.20834)

#### Entropy-Guided Sequence Weighting for Efficient Reward-Free Preference Optimization (EGSW) (2025-03)
- **简介**：在 sample 维度做 temperature-scaled softmax 加权——按每条 sequence 的 entropy × advantage 共同加权。低熵 high-advantage sequence 拿到主要梯度，高熵 high-advantage sequence 视为 exploratory 给次要梯度，自然平衡 exploitation 与 exploration。
- **arXiv**：[2503.22456](https://arxiv.org/abs/2503.22456)

#### Token-level Importance Sampling for Direct Preference Optimization (TIS-DPO) (2024-10)
- **简介**：理论证明最优 DPO 数据每 token 应有相等期望 reward，否则需 token IS 修正：w_t = exp(r̂_token / β)。给出 DPO 训练数据质量与 token-level 偏差的严格关系，是 token-aware DPO 系列的关键理论支点。ICLR 2025。
- **arXiv**：[2410.04350](https://arxiv.org/abs/2410.04350)

#### Selective Preference Optimization via Token-Level Reward Function Estimation (SePO) (2024-08)
- **简介**：训 oracle 模型估计 token reward，仅对最关键的 30% token 做对比训练。与 PURE 形成对偶——PURE 抑制最差 token，SePO 强化最关键 token，两者都把 reward 集中到少数关键 token 来减少 noise。
- **arXiv**：[2408.13518](https://arxiv.org/abs/2408.13518)

#### RLHF Workflow: From Reward Modeling to Online RLHF (RTO, Reinforced Token Optimization) (2024-04)
- **简介**：把 RLHF 严格形式化为 token-level MDP——先用 DPO 抽 token-level reward，再用 PPO 在 token MDP 上优化。理论上把 DPO 与 PPO 桥接为同一框架的两个阶段，是"DPO meets PPO"的代表工作。
- **arXiv**：[2404.18922](https://arxiv.org/abs/2404.18922)

### 1.5 Segment-level Credit Assignment

> 2025.10 以来 CA 粒度从 token / trajectory 两个极端向 segment / sub-trajectory / turn-cluster 中间层收敛。

#### VSPO: Value-based Segmental Policy Optimization with Progressive Reward Shaping (2025-12)
- **简介**：阿里飞猪提出。渐进式中间步 reward shaping + 基于 value 的样本采样，专治 GRPO 在 TIR（tool-integrated reasoning）长程任务上的 advantage vanish。把训练分阶段从粗段（trajectory）逐步过渡到细段（step），收敛比纯 GRPO 更稳定。
- **arXiv**：[2512.07478](https://arxiv.org/abs/2512.07478)

#### SALT: Step-level Advantage via Trajectory Graph for LLM Reasoning (2025-10)
- **简介**：Amazon 提出。在同一 prompt 的多条 trajectory 上构图——共享步合并为节点，未共享步独立成节点；每节点 advantage = f(node_freq, success_rate(node))。无需额外 rollout，直接 plug-in GRPO/RLOO，是 GiGPO（anchor-state 聚类）之外另一条 segment-level CA 路线。
- **arXiv**：[2510.20022](https://arxiv.org/abs/2510.20022)

#### Attribution-based Contribution to Policy Optimization (ACPO) (2025-10)
- **简介**：trajectory 语义分段后，用 attribution-based representation 调节 entropy；factorized reward 对每段分级贡献。把"哪些段贡献了答案"显式化为可解释的归因图，缓解 long CoT 训练中 advantage 平摊。
- **arXiv**：[2510.08899](https://arxiv.org/abs/2510.08899)

#### Tree-GRPO: Tree-based Group Relative Policy Optimization for Multi-step Agents (2025-09)
- **简介**：阿里高德。把 agent 的 step-level rollout 组织为树搜索，intra-tree group relative advantage 等价于 step-level DPO（理论可证）。仅用 1/4 GRPO 预算即超过同 size 的 GRPO baseline，是把 GRPO 思想扩到树结构的关键工作。
- **arXiv**：[2509.21240](https://arxiv.org/abs/2509.21240)

#### Tree-OPO: Off-policy Monte Carlo Tree-Guided Advantage Optimization for Multistep Reasoning (2025-09)
- **简介**：借 teacher MCTS 构 prefix curriculum，提出 Staged Advantage Estimation 把 reward 分配到 tree hierarchy 不同层级。让小 student 直接从 teacher 的搜索结构中学到分阶段的 advantage 信号。
- **arXiv**：[2509.09284](https://arxiv.org/abs/2509.09284)

#### First Return, Entropy-Eliciting Explore (FR3E) (2025-07)
- **简介**：ByteDance Seed 团队。识别 trajectory 中的高熵决策点，从这些 token 重新 rollout K 次估 V̂(state)，再按 ΔV 缩放 advantage。等价于"VinePPO 的高熵自适应版"——只在真正不确定的关键节点付额外 rollout 算力。
- **arXiv**：[2507.07017](https://arxiv.org/abs/2507.07017)

#### Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models (SPO) (2025-05)
- **简介**：UIUC 团队提出，正式确立 segment-level CA 独立分支。把 sequence 切成连续 segment（cutpoint 或 tree 两种方式），段级 MC 估 V̂(seg_prefix)，A_seg = R_total − V̂；short-CoT 用 SPO-chain，long-CoT 用 SPO-tree。是 segment-level CA 当前最被引用的奠基工作。
- **arXiv**：[2505.23564](https://arxiv.org/abs/2505.23564)

### 1.6 Causal / Counterfactual CA & Anti-Reward-Hacking

#### Length Bias Through a Causal Lens: Counterfactual Data Augmentation for Robust Reward Models (2025-11)
- **简介**：把 RM 的 length bias 形式化为因果图中的 spurious path do(L)。通过 counterfactual data augmentation 构造 length-divergent / content-divergent 对，强制 RM 在 length 与 content 上做条件独立判断。RewardBench 上抗 length attack 显著提升。
- **arXiv**：[2511.12573](https://arxiv.org/abs/2511.12573)

#### Causal Reward Adjustment via Sparse Autoencoders Reduces Reward Hacking (CRA) (2025-08)
- **简介**：用稀疏自编码器分解 reward model 的内部表示，识别引发 hacking 的 spurious 子空间，再用 backdoor adjustment 公式恢复真实奖励。是把 mechanistic interpretability 直接接到 RL anti-hacking 的代表工作。
- **arXiv**：[2508.04216](https://arxiv.org/abs/2508.04216)

#### Counterfactually-Guided Length Debiasing for Reward Models (CoLD) (2025-07)
- **简介**：把 PRM 的 length bias 视作 do(L) 的 spurious path，给出三件套——length penalty、bias estimator、joint training 联合训练。在不显著降低 RM 准确率前提下消除 verbosity bias，PRM 系列工作中首批严肃用 do-calculus 的方法。
- **arXiv**：[2507.15698](https://arxiv.org/abs/2507.15698)

#### CROME: Causally Robust Reward Modeling for LLMs via Causal Augmentations (2025-06)
- **简介**：通过 causal augmentation（针对每条 attribute，构造该 attribute 改变但其他不变的 counterfactual 数据）+ neutral augmentation（与该 attribute 无关的扰动）训练 RM 抵抗 spurious feature。RewardBench +5.4%，是因果 RM 的代表配方。
- **arXiv**：[2506.16507](https://arxiv.org/abs/2506.16507)

#### Shapley Credit Assignment Rewards (SCAR) (2025-05)
- **简介**：用合作博弈论中的 Shapley value 把序列级 reward 公平分配到 token / span。理论保证最优策略不变，无需训 critic 或细标，是博弈论视角下 token/segment-level CA 的代表工作。
- **arXiv**：[2505.20417](https://arxiv.org/abs/2505.20417)

#### Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning (PURE) (2025-04)
- **简介**：观察到 GRPO 用 PRM 累加 reward 容易被 hacking——单步 PRM 评分稍偏即累积放大。PURE 改用 segment 内最小值（min-form）做 reward 聚合，抑制 hacking 的同时仍保留 step-level CA 信号。3× 训练效率提升。
- **arXiv**：[2504.15275](https://arxiv.org/abs/2504.15275)

#### Mitigating Reward Hacking via Information-Theoretic Reward Modeling and Causal Rewards for Language Model Alignment (2025-01)
- **简介**：Meta + UChicago 提出 RM 学习 counterfactual invariance——当无关变量被 intervene 时 reward 不变。可作 drop-in 替换标准 RM 损失，在 OOD 与 reward over-optimization 设置下显著优于普通 RM。
- **arXiv**：[2501.09620](https://arxiv.org/abs/2501.09620)

#### A Principled Loss for Direct Preference Optimization (2024-08)
- **简介**：论证标准 DPO 损失与其 BT 模型推导假设不一致，导致 logit 差无界增长（即 chosen-rejected 概率比可至无穷），引发训练不稳定与 reward hacking。提出基于 RLHF optimality 的固定目标值损失，使 DPO 训练 logit 收敛到有界目标。
- **arXiv**：[2508.07137](https://arxiv.org/abs/2508.07137)

#### ODIN: Disentangled Reward Mitigates Hacking in RLHF (2024-02)
- **简介**：双头 RM 设计——length 头与 content 头独立训练；RLHF 推理时仅取 content 头的 reward。直接消除"答案越长越好"的 length hacking，是 RLHF 工程实践中最简单有效的 anti-hacking 配方之一。
- **arXiv**：[2402.07319](https://arxiv.org/abs/2402.07319)

#### WARM: On the Benefits of Weight Averaged Reward Models (2024-01)
- **简介**：Google DeepMind 提出。多个独立训练的 RM 在 weight space 上平均（而非 prediction ensembling），显著提升 OOD 鲁棒性与抗 reward hacking 能力。等价于一个免费的"reward model regularizer"，已成 RM 训练的工程最佳实践之一。
- **arXiv**：[2401.12187](https://arxiv.org/abs/2401.12187)

### 1.7 DPO 步级变体与无显式 RM 路线

#### Iterative Reasoning Preference Optimization (IRPO) (2025-04)
- **简介**：迭代式 reasoning preference optimization——每轮用上一轮模型生成的 trajectory 自动构造正负偏好对（按答案正确性），再训新的 DPO model；多轮迭代显著放大数学推理能力。可看作 ReST + DPO 的偏好版本。
- **arXiv**：[2504.15477](https://arxiv.org/abs/2504.15477)

#### β-DPO: Direct Preference Optimization with Dynamic β (2024-07)
- **简介**：动态调节 DPO 的 KL 系数 β——根据每个 batch 的偏好对质量（用 reward gap 估计）自适应升降 β。低质量 pair β 调高更保守，高质量 pair β 调低更激进，比固定 β 显著稳定。
- **arXiv**：[2407.08639](https://arxiv.org/abs/2407.08639)

#### Step-Controlled DPO: Leveraging Stepwise Errors for Enhancing Mathematical Reasoning (SCDPO) (2024-07)
- **简介**：用 step-level error 把同一 prompt 拆成正负对——找到错误首步，把错前 prefix 共享但分歧后正确/错误的两条 trajectory 当 chosen/rejected。是 Step-DPO 的可控版本，对错误位置精细化。
- **arXiv**：[2407.00782](https://arxiv.org/abs/2407.00782)

#### Step-DPO: Step-wise Preference Optimization for Long-Chain Reasoning of LLMs (2024-06)
- **简介**：把 DPO 的 chosen/rejected 从 response 级降到 step 级——用同一前缀下分叉的两个步骤构造偏好对。首批正面验证 step-level preference 能 scale 数学推理的工作，对 70B 模型 MATH 提升 3-4 点。
- **arXiv**：[2406.18629](https://arxiv.org/abs/2406.18629)

#### BoNBoN Alignment for Large Language Models and the Sweetness of Best-of-n Sampling (2024-06)
- **简介**：把 Best-of-N 蒸馏到模型本身的 PO loss——同时用 mean-of-N（提升整体）与 best-of-N（提升尾部）两种 reference 构造偏好对。理论给出 BoN distribution 的解析形式，让 alignment 不再需要 inference-time BoN。
- **arXiv**：[2406.00832](https://arxiv.org/abs/2406.00832)

#### Token-level Direct Preference Optimization (TDPO) (2024-04)
- **简介**：把 DPO 的 KL 约束从序列级降到 token 级——每个 token 都有独立的 reward 与 KL 项。首批 token-level 偏好优化工作，让 DPO 能精细抑制单 token 偏离 reference 的程度。
- **arXiv**：[2404.11999](https://arxiv.org/abs/2404.11999)

#### sDPO: Don't Use Your Data All at Once (2024-03)
- **简介**：分阶段加入 preference 数据训练——先用部分高质量 pair 训出 ckpt，再把它当 reference，加新数据训下一阶段。把 DPO 一次训完改造成多阶段课程式训练，比一次性训练更稳。
- **arXiv**：[2403.19270](https://arxiv.org/abs/2403.19270)

#### Provably Robust DPO: Aligning Language Models with Noisy Feedback (R-DPO) (2024-03)
- **简介**：对偏好数据中的 label noise（如标错的 chosen/rejected）给 PAC-style 鲁棒性保证，提出 noise-aware 的 DPO loss 修正项。在含 20-30% 标注噪声的数据上仍能可靠训练，是 DPO 走向真实噪声数据的重要工作。
- **arXiv**：[2403.00409](https://arxiv.org/abs/2403.00409)

#### Noise Contrastive Alignment of Language Models with Explicit Rewards (NCA) (2024-02)
- **简介**：把 alignment 形式化为带显式 reward 的 noise contrastive estimation。当数据带 explicit reward 而非 pairwise preference 时，NCA 比 DPO 更高效；提供 reward score → policy 的另一条直接路径。
- **arXiv**：[2402.05369](https://arxiv.org/abs/2402.05369)

#### Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive (DPOP) (2024-02)
- **简介**：识别 DPO 在 chosen-rejected 编辑距离过近时的 failure mode——logit 同时下降但 chosen 反相对劣化。DPO-Positive 项额外强制 chosen 概率不下降，配合得到 Smaug-72B 模型，在 MT-Bench 上 SOTA。
- **arXiv**：[2402.13228](https://arxiv.org/abs/2402.13228)

#### Generalized Preference Optimization: A Unified Approach to Offline Alignment (GPO) (2024-02)
- **简介**：DeepMind 提出统一框架——把 DPO / IPO / SLiC / SLiC-HF / KTO 等离线对齐方法都重写为同一个偏好优化目标的不同凸损失。给每种损失刻画其偏置-方差性质，便于按数据特性选合适方法。
- **arXiv**：[2402.05749](https://arxiv.org/abs/2402.05749)

#### KTO: Model Alignment as Prospect Theoretic Optimization (2024-02)
- **简介**：基于 Kahneman-Tversky prospect theory 的对齐——只需单样本 0/1 偏好（response 是 desirable / undesirable）而非 pairwise pair，使用非线性 prospect-theoretic loss。偏好数据成本大降，对噪声更鲁棒。
- **arXiv**：[2402.01306](https://arxiv.org/abs/2402.01306)

#### SimPO: Simple Preference Optimization with a Reference-Free Reward (2024-05)
- **简介**：reference-free 简化 DPO——直接用 length-normalized log-prob 当 reward，不再需要参考模型。同时引入 target reward margin 显式分离 chosen/rejected。在多个对齐 benchmark 上稳定优于 DPO。
- **arXiv**：[2405.14734](https://arxiv.org/abs/2405.14734)

#### ORPO: Monolithic Preference Optimization without Reference Model (2024-03)
- **简介**：Odds Ratio Preference Optimization 把 SFT loss 与 preference loss 一次合训，无需独立 reward model 与 reference model。极大简化 alignment pipeline，是单阶段对齐的代表工作。
- **arXiv**：[2403.07691](https://arxiv.org/abs/2403.07691)

#### A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO) (2023-10)
- **简介**：DeepMind 给 RLHF/DPO 一个统一理论框架 ΨPO；通过 closed-form 替代 DPO 中导致 overfitting 的 sigmoid 项（IPO），用平方损失代替分类损失，在小偏好数据上显著缓解 reward hacking 与 over-optimization。
- **arXiv**：[2310.12036](https://arxiv.org/abs/2310.12036)

#### Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO) (2023-05)
- **简介**：Stanford 团队的奠基工作。把 RLHF 的 PPO + RM 两阶段重写为基于参考策略的偏好分类损失，无需 reward model 训练与 PPO 实现。后续所有偏好优化变体（IPO、KTO、SimPO、ORPO、Step-DPO、TDPO、TGDPO、TIS-DPO 等）的母体。
- **arXiv**：[2305.18290](https://arxiv.org/abs/2305.18290)

### 1.8 Reward Modeling：Generative / Self-Reward / Robust RM

#### Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy (2025-07)
- **简介**：Skywork 第二代 RM，超越 v1 显著。核心是人机协同的偏好数据 curation pipeline——LLM 自动筛选 + 人工审核高难度边界 case，最终得到 SOTA RewardBench 表现。是当前公开 RM 中最强 baseline 之一。
- **arXiv**：[2507.01352](https://arxiv.org/abs/2507.01352)

#### LaSeR: Reinforcement Learning with Last-Token Self-Rewarding (2025-10)
- **简介**：把 LLM 自身在最后 token 的 verifier 行为转换为 self-reward——用模型对答案的 confidence 作为 reward 信号。无需外部 RM 即可 RL 训练，是 self-reward 范式在 RL 上的轻量实现。
- **arXiv**：[2510.14943](https://arxiv.org/abs/2510.14943)

#### Self-Rewarding Reasoning Reward Model (2025-02)
- **简介**：让模型同时充当 generator 与 verifier，通过 reasoning RL 提升 self-verify 能力。Verifier 也是同一 LLM，在 reasoning 过程中输出 confidence 或 critique。是 LLM 一体化做生成+评估的代表工作。
- **arXiv**：[2502.19613](https://arxiv.org/abs/2502.19613)

#### Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs (2024-10)
- **简介**：Skywork 团队第一代 RM。仅 80K 高质量 preference pair 训出 8B/27B SOTA RM；公开 RM 数据筛选、训练超参与去重 pipeline 的完整 best practice，是 RM 工程的实战手册。
- **arXiv**：[2410.18451](https://arxiv.org/abs/2410.18451)

#### Generative Reward Models (2024-10)
- **简介**：合成数据训练生成式 RM——judge 与 reward 二合一，先生成 critique 再给 score。统一了 LLM-as-judge 与 reward modeling 两个研究方向，CritiqueLLM、Prometheus-2 等工作的近亲。
- **arXiv**：[2410.12832](https://arxiv.org/abs/2410.12832)

#### Generative Verifiers: Reward Modeling as Next-Token Prediction (GenRM) (2024-08)
- **简介**：Google DeepMind 提出。把 verifier 做成 next-token prediction——对 (question, answer) 之后预测 "Yes"/"No" 概率作为 verifier 分数。可一同训 generation 与 verification 头，比独立判别式 verifier 更易扩展且能与 SFT 共训。
- **arXiv**：[2408.15240](https://arxiv.org/abs/2408.15240)

#### Self-Taught Evaluators (2024-08)
- **简介**：Meta 提出。自举训练 LLM-as-judge——用模型自己产生的对比数据迭代提升 judge 质量，无需任何人工偏好标注。在 RewardBench 上接近 GPT-4 judge，为大规模 self-improving evaluator 提供可行路径。
- **arXiv**：[2408.02666](https://arxiv.org/abs/2408.02666)

#### Critique-out-Loud Reward Models (CLoud) (2024-08)
- **简介**：让 RM 在打分前先生成可读 critique，再综合 critique 与原 response 给 score。提升 RM 的可解释性与对齐人类判断的能力，是把"思考链"思想引入 reward modeling 的代表工作。
- **arXiv**：[2408.11791](https://arxiv.org/abs/2408.11791)

#### Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models (2024-05)
- **简介**：KAIST + LG AI 开源的细粒度 LLM-as-judge 模型。同时支持 absolute scoring 与 pairwise comparison 两种评估模式，并按多维 rubric（helpfulness、honesty、harmlessness 等）分别评分。
- **arXiv**：[2405.01535](https://arxiv.org/abs/2405.01535)

#### Self-Play Preference Optimization for Language Model Alignment (SPPO) (2024-05)
- **简介**：UCLA 提出。把 alignment 建模为 constant-sum two-player game 找 Nash policy，用迭代 self-play 逼近。Mistral-7B 在 AlpacaEval 2.0 LC 28.53%；Llama-3-8B 38.77%，是 self-play 风格对齐的代表算法。
- **arXiv**：[2405.00675](https://arxiv.org/abs/2405.00675)

#### Self-Rewarding Language Models (2024-01)
- **简介**：Meta 提出。模型同时作为 response generator 与 self-evaluator，多轮迭代 self-reward 训练。Llama-2-70B 经 3 轮 self-rewarding 后超过 Claude 2 和 GPT-4 0613，证明无需外部偏好数据也能持续提升对齐质量。
- **arXiv**：[2401.10020](https://arxiv.org/abs/2401.10020)

#### Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (SPIN) (2024-01)
- **简介**：UCLA 提出。用上一轮模型生成 response 当 opponent，新模型学会区分"自己生成"与"真实数据"再做 SFT。无需任何外部 RM 或人工偏好，仅靠 SFT 数据迭代 self-play 即可超越纯 SFT，是 self-play 对齐的早期奠基工作。
- **arXiv**：[2401.01335](https://arxiv.org/abs/2401.01335)

#### JudgeLM: Fine-tuned Large Language Models are Scalable Judges (2023-10)
- **简介**：早期开源 LLM-as-judge 工作之一。通过指令微调把 LLM 训成偏好评估器，在多种任务上判断对齐人类。是 RewardBench 等评测出现前最被引用的开源 judge baseline。
- **arXiv**：[2310.17631](https://arxiv.org/abs/2310.17631)

### 1.9 Self-Improvement / Test-time Scaling

#### Test-Time Reinforcement Learning (TTRL) (2025-04)
- **简介**：清华 + 上海 AI Lab 提出。对每个测试问题，用 majority-vote@N 的多数答案作伪 reward 在线 GRPO 微调，**无需任何 ground-truth 标注**。Qwen-2.5-Math-7B 在 AIME 2024 pass@1 +211%，是 inference-time RL 概念的标志工作。
- **arXiv**：[2504.16084](https://arxiv.org/abs/2504.16084)

#### Tina: Tiny Reasoning Models via LoRA (2025-04)
- **简介**：1.5B 模型 + LoRA RL 训练，最佳 ckpt 仅 $9 USD 训练成本，AIME24 +20%、Pass@1 43.33%。证明 RL 推理能力可以在极小算力下成立，对中小团队友好。
- **arXiv**：[2504.15777](https://arxiv.org/abs/2504.15777)

#### LIMO: Less is More for Reasoning (2025-02)
- **简介**：上海交大提出。仅 800 条高质量数学推理样本即可激发 Qwen2.5-32B 的 long CoT 能力，AIME 24 6.5 → 57.1%。质量远比数量重要——挑战了"RL 推理需要大数据"的默认假设。
- **arXiv**：[2502.03387](https://arxiv.org/abs/2502.03387)

#### B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners (2024-12)
- **简介**：HKUST 提出。系统监控 self-taught reasoner（STaR）训练动力学的 exploration vs exploitation balance——pass@k diversity 与 pass@1 accuracy 的折衷。给出动态调温度与采样数的策略保持长程提升。
- **arXiv**：[2412.17256](https://arxiv.org/abs/2412.17256)

#### e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs (2025-06)
- **简介**：CMU + DeepMind 关注 TTS 真正的"外推"问题：多数模型在训练 token 预算外不能持续提升。e3 三大要素——链式调用基础 LLM 不对称能力、错误轨迹"负梯度"放大探索、难度-预算耦合课程。1.7B 模型在 AIME 25 / HMMT 25 拿到该 size 最佳，extrapolate 到 2× 训练预算。
- **arXiv**：[2506.09026](https://arxiv.org/abs/2506.09026)

#### L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning (2025-03)
- **简介**：CMU 提出。LCPO-Exact / LCPO-Max 让模型按目标 token 长度生成 CoT；与 budget forcing（"Wait"）相比，scaling 显著更好且不损失推理质量。把"让模型按预算思考"做成可学习目标。
- **arXiv**：[2503.04697](https://arxiv.org/abs/2503.04697)

#### Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters (Snell et al.) (2024-08)
- **简介**：DeepMind 最重要的 TTS 论文。系统比较 search vs adaptive distribution 两种 inference-time compute 分配，得出"compute-optimal 策略下小模型 + TTS 可超过 14× 大模型"的关键判断。后续所有 TTS 工作的对照基线。
- **arXiv**：[2408.03314](https://arxiv.org/abs/2408.03314)

#### Large Language Monkeys: Scaling Inference Compute with Repeated Sampling (2024-07)
- **简介**：Stanford 提出。仅靠 repeated sampling，DeepSeek-Coder-V2 在 SWE-Bench Lite 从 15.9% → 56%；提供"采样次数 vs 覆盖率"的 power-law 拟合。证明 inference-time 算力本身就是一个独立的 scaling 维度。
- **arXiv**：[2407.21787](https://arxiv.org/abs/2407.21787)

#### V-STaR: Training Verifiers for Self-Taught Reasoners (2024-02)
- **简介**：Mila 提出。用 STaR 过程产生的"错误解"训练 verifier；DPO 训练的 verifier 比 ORM 更优。把 STaR 的废弃负样本变成 reward 信号，是 self-taught reasoner 谱系的重要扩展。
- **arXiv**：[2402.06457](https://arxiv.org/abs/2402.06457)

#### ReFT: Reasoning with REinforced Fine-Tuning (2024-01)
- **简介**：ByteDance 提出。SFT warmup + PPO + ground-truth answer reward；不需训 RM；比 STaR / SC / MV 更稳定。是早期把 RL 用于数学推理而不依赖 PRM 的代表工作。
- **arXiv**：[2401.08967](https://arxiv.org/abs/2401.08967)

#### Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models (ReSTEM) (2023-12)
- **简介**：Google DeepMind 提出。ReST + EM 期望最大化——E 步用模型采样 + 二元过滤构数据集，M 步从 base model 在过滤集上 SFT。每轮重启 base 避免累积漂移，是 self-training 的稳定配方。
- **arXiv**：[2312.06585](https://arxiv.org/abs/2312.06585)

#### RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment (2023-04)
- **简介**：HKUST 提出。Best-of-N + SFT 的极简对齐——采 N 条 response，按 RM 排序取 top-K SFT。是离线偏好学习最简单的 baseline，后续 ReST、ReSTEM、RFT 等系列的起点。
- **arXiv**：[2304.06767](https://arxiv.org/abs/2304.06767)

### 1.10 综述与基准

- **The Landscape of Agentic Reinforcement Learning for Large Language Models** (2025-09)：700+ 论文综述，6 大能力 + 7 大任务双重分类。[arXiv:2509.02547](https://arxiv.org/abs/2509.02547)
- **A Comprehensive Survey on Learning from Rewards for Large Language Models** (2025)：训练 / 推理 / 后处理三阶段统一。[ACL Anthology](https://aclanthology.org/2025.findings-emnlp.970/)
- **Reward Models in Language Model Alignment: Taxonomy, Applications, Challenges, and Future** (2025-04)：RM 全谱系综述。[arXiv:2504.12328](https://arxiv.org/abs/2504.12328)
- **A Survey on Test-Time Scaling in Large Language Models: What, How, Where, and How Well?** (2025-03)：TTS 四维分类综述。[arXiv:2503.24235](https://arxiv.org/abs/2503.24235)
- **PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models** (2025-01)：PRM 全方位评测基准。[arXiv:2501.03124](https://arxiv.org/abs/2501.03124)
- **ProcessBench: Identifying Process Errors in Mathematical Reasoning** (2024-12)：评测 PRM 检测 step error 的能力。[arXiv:2412.06559](https://arxiv.org/abs/2412.06559)
- **JudgeBench: A Benchmark for Evaluating LLM-based Judges** (2024-10)：LLM-as-judge 的对抗式评测。[arXiv:2410.12784](https://arxiv.org/abs/2410.12784)
- **RM-Bench: Benchmarking Reward Models with Subtle Differences and Style Biases** (2024-10)：抗 stylistic spurious feature 的 RM 评测。[arXiv:2410.16184](https://arxiv.org/abs/2410.16184)
- **RewardBench: Evaluating Reward Models for Language Modeling** (2024-03)：RM 评测开山基准。[arXiv:2403.13787](https://arxiv.org/abs/2403.13787)

---

## 2. Agentic RL

> 多轮、长 horizon、部分可观测的智能体 RL，覆盖工具调用、GUI、网页、代码、记忆、安全。

### 2.1 Tool-use / Multi-turn Agent

#### Nemotron-Research-Tool-N1: Tool-Using Language Models with Reinforced Reasoning (2025-05)
- **简介**：NVIDIA 提出。纯 rule-based reward + GRPO 训出工具调用 agent；reward 同时校验 tool schema、参数合法性、调用成功率三层。摆脱 SFT trace 依赖，证明 tool-use 能力可通过纯 RL 涌现，是 tool-call rule reward 设计的代表配方。
- **arXiv**：[2505.00024](https://arxiv.org/abs/2505.00024)

#### ARTIST: Agentic Reasoning and Tool Integration in Self-improving Transformers (2025-05)
- **简介**：把 agent 自我训练做成自演化 curriculum——agent 自动产生新任务、解新任务、用 outcome reward 训练自己，多轮迭代提升。证明 agent 能力可以无人工 curriculum 自演化，是 agent self-improvement 的代表工作。
- **arXiv**：[2505.01441](https://arxiv.org/abs/2505.01441)

#### ToolRL: Reward is All Tool Learning Needs (2025-04)
- **简介**：UIUC 系统证明 Tool-use Agent 的 reward design 比算法本身更关键。给出多层细粒度 reward 模板（schema 校验 + 软相似度匹配 + retrieved-token mask + 自演化 curriculum），让纯 GRPO 即超过复杂算法变体。
- **arXiv**：[2504.13958](https://arxiv.org/abs/2504.13958)

#### ReTool: Reinforcement Learning for Strategic Tool Use in LLMs (2025-04)
- **简介**：让 LLM 在 Python sandbox 内做 tool-integrated reasoning——把代码执行结果实时反馈给 reasoning。RL 训 LLM 何时该调代码、何时不调，AIME 24 提升 8 点；是当前代码工具 RL 的标杆配方之一。
- **arXiv**：[2504.11536](https://arxiv.org/abs/2504.11536)

#### Acting Less is Reasoning More! Teaching Model to Act Efficiently (OTC-PO) (2025-04)
- **简介**：把 tool call 数量作为额外 cost 加进 reward，让 agent 学会只在必要时调工具。Optimal Tool Cost Policy Optimization 显著降低无效 API 调用，同时保留任务成功率，是 tool-budget aware RL 的代表。
- **arXiv**：[2504.14870](https://arxiv.org/abs/2504.14870)

#### ToRL: Scaling Tool-Integrated RL (2025-03)
- **简介**：把 tool 作为 first-class action 进入 RL 训练框架，统一处理 tool call schema 验证、partial credit、调用代价。开源训练框架支持 search、calculator、code interpreter 等多种工具，是 tool-integrated RL 的工程参考。
- **arXiv**：[2503.23383](https://arxiv.org/abs/2503.23383)

### 2.2 Turn-level Credit Assignment

#### From Trajectories to Tokens: Turn-Level PPO for Multi-Turn Reasoning (Turn-PPO) (2025-12)
- **简介**：放弃 token-level MDP 回到 turn-level MDP + 学 critic——每 turn 一个 V_φ(s_turn)，A_turn = R_turn + γV(s_{turn+1}) − V(s_turn)。在 multi-turn agent 上比 token-level GRPO 更稳定，是 turn-level CA 算法回归 critic-based 路线的代表。
- **arXiv**：[2512.17008](https://arxiv.org/abs/2512.17008)

#### Group Turn Policy Optimization for Multi-Turn LLM Agent Reinforcement Learning (GTPO) (2025-11)
- **简介**：Amazon AGI 团队提出。Turn-level reward + return-based advantage + 自监督 reward shaping（用 tool-call code 相似度做 partial reward），缓解多轮 agent reward 极度稀疏的问题。
- **arXiv**：[2511.14846](https://arxiv.org/abs/2511.14846)

#### Information Gain-based Policy Optimization for Multi-Turn Agent (IGPO) (2025-10)
- **简介**：每 turn 建模为"获得正确答案概率的边际增量"——用 LLM 自身 belief 作内蕴 reward：r_turn = P_θ(answer | h_turn) − P_θ(answer | h_turn-1)。无需外部 verifier，是最早把 belief change 当 turn reward 的工作之一。
- **arXiv**：[2510.14967](https://arxiv.org/abs/2510.14967)

#### Group-in-Group Policy Optimization for LLM Agent Training (GiGPO) (2025-05)
- **简介**：双层 group 信用分配——episode 级 group baseline + step 级 group baseline 双重 advantage normalization。让 agent 同时获得任务完成的全局信号与单步动作的局部信号，多轮 agent CA 标志性算法。
- **arXiv**：[2505.10978](https://arxiv.org/abs/2505.10978)

#### Fine-Grained Turn-Level Credit Assignment for Multi-Turn LLM Agents (MT-GRPO) (2025-05)
- **简介**：多轮 GRPO + turn-level advantage：A_turn1 = λ·R_turn1 + (1−λ)·R_outcome；Wikipedia 检索 agent 上 50% 精确匹配 vs 20–30% baseline，tool execution 成功率达 100%。
- **arXiv**：[2505.11821](https://arxiv.org/abs/2505.11821)

#### RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning (StarPO-S) (2025-04)
- **简介**：诊断多轮 agent RL 的"Echo Trap"——reward variance cliff + gradient spike 联合导致策略退化。StarPO-S 给出 trajectory filtering + critic incorporation + decoupled clipping + gradient stabilization 四件套修复，是多轮 agent RL 训练稳定性的关键参考。
- **arXiv**：[2504.20073](https://arxiv.org/abs/2504.20073)

#### SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks (2025-03)
- **简介**：Meta FAIR 提出。非对称 critic——训练时用 oracle 信息（如 ground-truth code），推理时不用；turn-level critic 接受比 actor 多得多的特权信号。是首个真正利用"训练时特权信息"的多轮 agent RL，代码协作任务 +6%。
- **arXiv**：[2503.15478](https://arxiv.org/abs/2503.15478)

### 2.3 Hindsight / Counterfactual Turn CA

#### TRACE: Trajectory-Aware Credit Assignment for Multi-Turn Jailbreaking via Hindsight Decomposition (2026-05)
- **简介**：成功 trajectory 用 leave-one-turn-out 估每 turn 贡献；失败 trajectory 用 prompt harmfulness + semantic relevance 罚分。专门针对多轮 jailbreaking 的 CA 设计，是 hindsight 思想在安全场景的代表应用。
- **arXiv**：[2605.08778](https://arxiv.org/abs/2605.08778)

#### Self-Induced Outcome Potential for Verifier-Free Multi-Turn Reward Shaping (SIOP) (2026-05)
- **简介**：在没有 ground-truth verifier 时，把最终答案的语义 cluster 当作 latent outcome state；turn 增加该 cluster 的后验支持就给奖励。把"无监督 outcome 重建"思想引入 turn-level reward shaping，是 verifier-free agent RL 的代表方案。
- **arXiv**：[2605.04984](https://arxiv.org/abs/2605.04984)

#### Hindsight Credit Assignment for Long-Horizon Agent Policy Optimization (HCAPO) (2026-03)
- **简介**：腾讯团队。对训练好的 model 反向跑 trajectory，用 hindsight verifier 给每个 turn 反事实 advantage——"假如这一 turn 不同会怎样"。是 RUDDER 思想在多轮 agent 上的真正实现，长 horizon 任务 +15+ 点。
- **arXiv**：[2603.08754](https://arxiv.org/abs/2603.08754)

#### Hindsight Information Modulated Segmental Rewards for Multi-Turn Agent (HISR) (2026-03)
- **简介**：用回溯模型度量动作重要性，给出**分段级**而非转级的过程奖励。把 trajectory 切成语义 segment，每段用 hindsight value 估贡献，是 hindsight × segment-level 的混合工作。
- **arXiv**：[2603.18683](https://arxiv.org/abs/2603.18683)

#### CriticSearch: Step-Level Reward Modeling via Frozen Critic Hindsight for Search Agents (2025-11)
- **简介**：冻结一个 critic LLM 做回溯评估，给每轮搜索 trajectory 一个 dense 转级奖励，避免 online critic 训练的不稳定。3B 模型 +16.7%、7B +6.7%，是 hindsight × frozen critic 的轻量实用方案。
- **arXiv**：[2511.12159](https://arxiv.org/abs/2511.12159)

#### ECHO: Hindsight Trajectory Rewriting for Language Model Agents (2025-10)
- **简介**：MIT 提出。把 Hindsight Experience Replay (HER) 移植到 LM agent——对失败 trajectory 用 LLM 反推"实际能完成的 alternative subgoal"，把失败 trajectory 改写为成功的 synthetic positive。XMiniGrid / PeopleJoinQA 上比 Reflexion 快 80%。
- **arXiv**：[2510.10304](https://arxiv.org/abs/2510.10304)

### 2.4 GUI / Embodied / Computer-Use Agent

#### MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents (AdaGRPO) (2025-09)
- **简介**：清华 + 蚂蚁。专门针对移动 GUI agent 的在线 RL；自适应 group-relative advantage 处理任务长度方差大的难点。屏幕截图 + 控件元数据双输入，AndroidWorld benchmark 显著提升。
- **arXiv**：[2509.18119](https://arxiv.org/abs/2509.18119)

#### Mobile-Agent-v3: Foundamental Agents for GUI Automation (TRPO) (2025-08)
- **简介**：阿里 X-PLUG 提出。Mobile agent v3 全栈方案；turn-level RL with TRPO 保证 trust region，避免多轮策略震荡。在 AndroidWorld 与 OSWorld-Mobile 上 SOTA。
- **arXiv**：[2508.15144](https://arxiv.org/abs/2508.15144)

#### Efficient Agent Training for Computer Use (PC Agent-E) (2025-05)
- **简介**：高效 PC 桌面 agent 训练——只需 312 条人工 demo 数据 + 自动 trajectory bootstrapping，即可让 7B 模型在 WindowsAgentArena-Lite 上比 Claude 3.5 高 2.6 点。是 computer-use agent 数据效率的标杆。
- **arXiv**：[2505.13909](https://arxiv.org/abs/2505.13909)

#### ManipLVM-R1: Reinforcement Learning for Reasoning in Embodied Manipulation with Large Vision-Language Models (2025-05)
- **简介**：VLM + RL 无需标注学习具身操作——用 affordance 一致性 + 轨迹匹配作 dense reward。证明具身 manipulation 可通过 RL 直接学，无需大规模 demo 数据，是 embodied VLM RL 的代表工作。
- **arXiv**：[2505.16517](https://arxiv.org/abs/2505.16517)

#### Being-0: A Humanoid Robotic Agent with Vision-Language Models and Modular Skills (2025-03)
- **简介**：把 humanoid robot agent 拆为 FM 高级认知（VLM 做规划）+ 技能库低级控制（学习的 motor primitive）。分层框架降低 RL 优化难度，是 humanoid 长程任务的代表配方。
- **arXiv**：[2503.12533](https://arxiv.org/abs/2503.12533)

#### UI-TARS: Pioneering Automated GUI Interaction with Native Agents (2025-01)
- **简介**：ByteDance 提出。原生 GUI agent 全流程——SFT + DPO + RL 三阶段；视觉感知 + 动作空间统一建模。在 OSWorld 与 AndroidWorld 上同时 SOTA，是开源 GUI agent 的代表工作。
- **arXiv**：[2501.12326](https://arxiv.org/abs/2501.12326)

#### AutoGLM: Autonomous Foundation Agents for GUIs (2024-11)
- **简介**：清华 + 智谱。自主 GUI 基础 agent；首个端到端 RL 训出的 GUI 通用 agent。在网页、桌面、移动多个 GUI 环境上展现统一能力，并开源训练框架。
- **arXiv**：[2411.00820](https://arxiv.org/abs/2411.00820)

#### OS-Atlas: A Foundation Action Model for Generalist GUI Agents (2024-10)
- **简介**：跨 OS 的通用 GUI grounding 模型——SFT + RL 提升 cross-platform 泛化。在 Windows、macOS、Linux、Android、Web 五种环境上统一训练，是 GUI grounding 的代表数据集 + 模型。
- **arXiv**：[2410.23218](https://arxiv.org/abs/2410.23218)

#### DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning (2024-06)
- **简介**：Berkeley 提出。Android device control 的 offline + online 两阶段 RL；首篇真实设备 RL，AitW 任务集 17 → 67%。是 GUI agent 走向真实设备的奠基工作。
- **arXiv**：[2406.11896](https://arxiv.org/abs/2406.11896)

### 2.5 Search / Web / Research Agent

#### MMSearch-R1: Incentivizing LMMs to Search (2025-06)
- **简介**：ByteDance + NTU。第一个让 LMM 在真实互联网环境里多轮搜索的端到端 RL 框架——把网页加载、链接跳转、内容理解、问答整条链路用 GRPO 训练。复杂多模态查询任务上比 SFT 提升显著。
- **arXiv**：[2506.20670](https://arxiv.org/abs/2506.20670)

#### DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments (2025-04)
- **简介**：BUPT + 商汤。在真实 Web 上多轮研究的 agent；end-to-end RL with rule-based reward（事实正确性 + 引用一致性）。证明深度研究 agent 不需 SFT trace，纯 RL 也能 work。
- **arXiv**：[2504.03160](https://arxiv.org/abs/2504.03160)

#### ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning (2025-03)
- **简介**：迭代式 search agent——reasoning → search → 整合 → reasoning 循环；纯 RL 训出"何时该搜、搜什么、如何用"。HotpotQA 与 MuSiQue 上比 SFT-based agent 显著提升。
- **arXiv**：[2503.19470](https://arxiv.org/abs/2503.19470)

#### Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (2025-03)
- **简介**：UIUC 提出。纯 RL 训搜索 agent 的代表工作；retrieved-token mask 保护检索回的文档不被 RL 梯度污染。证明 RL 训搜索比 SFT 训显著好，开源 Search-R1 框架被广泛复用。
- **arXiv**：[2503.09516](https://arxiv.org/abs/2503.09516)

#### R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning (2025-03)
- **简介**：阶段化 reward 设计——第一阶段只奖励搜索行为本身（鼓励 agent 学会调 search tool），第二阶段奖励答案正确性（让 agent 学会利用搜索结果）。两阶段 reward 比一次到位的 outcome reward 更稳定。
- **arXiv**：[2503.05592](https://arxiv.org/abs/2503.05592)

#### WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning (2024-11)
- **简介**：清华提出。Self-evolving curriculum 的 Web agent RL——用失败任务自动生成新训练任务（让 agent 把没解决的当作新挑战），实现 curriculum 自动演化。WebArena-Lite 上 4.8% → 42.4%。
- **arXiv**：[2411.02337](https://arxiv.org/abs/2411.02337)

### 2.6 Memory & Long-Horizon Agent

#### Memory-R2: LoGo-GRPO with Curriculum for Long-Horizon Memory Agents (2026-05)
- **简介**：LoGo-GRPO 同时优化 local（单 session 内）与 global（跨 session）组相对优势；progressive curriculum 从 8 → 16 → 32 sessions 渐进训练。专门解决"长 horizon 记忆 agent 在不同 session 长度下不稳定"的问题。
- **arXiv**：[2605.21768](https://arxiv.org/abs/2605.21768)

#### Memory Operation Tree GRPO for Memory-Augmented Agents (Mem-T / MoT-GRPO) (2026-01)
- **简介**：构造 Memory Operation Tree（MoT）——每条记忆操作（add/del/update）作为树节点，沿分支拓扑反向传播稀疏 outcome reward。Hindsight credit 把搜索效用归因到源 memory item，MemBench F1 +14.92%。
- **arXiv**：[2601.23014](https://arxiv.org/abs/2601.23014)

#### Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning (2025-08)
- **简介**：把 memory 操作（add / delete / update / search）作为 RL action；首个把记忆管理本身做成 RL 决策的工作。LongMemEval 上比无 memory baseline 提升 30+ 点，开启了"learnable memory operation policy"研究方向。
- **arXiv**：[2508.19828](https://arxiv.org/abs/2508.19828)

#### MemAgent: Reshaping Long-Context LLM with Multi-Conv RL Memory Agent (2025-07)
- **简介**：长 context agent + 学习的内存压缩 / 检索策略；用 multi-conversation RL 训内存读写，把"超长上下文如何放进有限上下文窗口"做成可学策略。LongBench-v2 上稳定优于 sliding-window。
- **arXiv**：[2507.02259](https://arxiv.org/abs/2507.02259)

### 2.7 Code / SWE Agent

#### AceCoder: Acing Coder RL via Automated Test-Case Synthesis (2025-02)
- **简介**：港大 + 滴滴。自动构造 RL 训练数据——LLM 生成单元测试，用单测通过率作为 reward，无需人工标注。Code RL 数据自动化标杆，CodeForces 与 LiveCodeBench 上显著提升。
- **arXiv**：[2502.01718](https://arxiv.org/abs/2502.01718)

#### SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution (2025-02)
- **简介**：Meta FAIR 提出。用 GitHub PR / issue 历史作 RL 训练数据——给定 issue 与 codebase，agent 生成 patch，patch 通过单测得 reward。SWE-bench Verified 上 41% 解决率，是 agentic 软件工程 RL 的代表工作。
- **arXiv**：[2502.18449](https://arxiv.org/abs/2502.18449)

#### Light-R1: Curriculum SFT, DPO and RL for Long COT from Scratch and Beyond (2025-03)
- **简介**：从 zero 开始 curriculum SFT + DPO + RL 三段式训练，仅需 small budget 复刻长 CoT 推理能力。Light-R1-32B 在 AIME 25 上 56.7%，超过 DeepSeek-R1-Distill-Llama-70B；是中小算力团队复现 R1 的代表配方。
- **arXiv**：[2503.10460](https://arxiv.org/abs/2503.10460)

### 2.8 Multimodal Agent RL

#### DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning (2025-05)
- **简介**：阿里高德。端到端 RL，无 cold-start SFT——奖励视觉工具调用（zoom-in、crop、annotate）成功，利用模型自身 grounding 能力涌现"看图思考"行为。InfographicsVQA 等任务比 SFT baseline 显著提升。
- **arXiv**：[2505.14362](https://arxiv.org/abs/2505.14362)

#### MM-PRM: Enhancing Multimodal Mathematical Reasoning with Scalable Step-Level Supervision (2025-05)
- **简介**：Shanghai AI Lab 把 OmegaPRM 框架扩展到多模态数学推理——用 10K 种子题通过自动 MC rollout 生成 700K+ step-level 标签。验证多模态 PRM 数据可自动 scale，是当前最大的多模态 step-level 数据。
- **arXiv**：[2505.13427](https://arxiv.org/abs/2505.13427)

#### Skywork-VL Reward: An Effective Reward Model for Multimodal Understanding and Reasoning (2025-05)
- **简介**：基于 Qwen2.5-VL-7B 的多模态 RM；同时支持图文一致性、推理正确性、harmlessness 多维度评分。VL-RewardBench 与 MMRewardBench 双基准 SOTA，是当前最强公开多模态 RM。
- **arXiv**：[2505.07263](https://arxiv.org/abs/2505.07263)

#### VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model (2025-04)
- **简介**：OmAgent 团队提出。通用 VL R1-style 训练框架——首次系统揭示视觉任务的"OD aha moment"，证明 VLM 可以像 LLM 一样涌现 self-reflection 行为。开源完整训练 pipeline。
- **arXiv**：[2504.07615](https://arxiv.org/abs/2504.07615)

#### OpenVLThinker: An Early Exploration to Vision-Language Reasoning via Iterative Self-Improvement (2025-03)
- **简介**：UCLA 提出。迭代 SFT-RL 循环培养 7B VLM 反思能力——每轮 RL 后用模型自采样数据再 SFT，再 RL。MathVista 与 MMMU 显著提升，是 VLM self-improvement 的代表工作。
- **arXiv**：[2503.17352](https://arxiv.org/abs/2503.17352)

#### MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning (2025-03)
- **简介**：Shanghai AI Lab。仅 8K 多模态数学数据（指令模型 0.05% 量）在 K12 +8.2%；关键发现：**KL 散度反而限制多模态指令模型探索**；离线难度过滤优于在线过滤。挑战了 KL penalty 在多模态 RL 中的默认假设。
- **arXiv**：[2503.07365](https://arxiv.org/abs/2503.07365)

#### VisualPRM: An Effective Process Reward Model for Multimodal Reasoning (2025-03)
- **简介**：Shanghai AI Lab 发布的首个 8B 多模态 PRM + VisualPRM400K 数据集 + VisualProcessBench 评测。自动 MC 估每步期望准确率，让 InternVL2.5-78B Best-of-N 提升 5.9 点，是多模态 PRM 的 de facto 标杆。
- **arXiv**：[2503.10291](https://arxiv.org/abs/2503.10291)

#### Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models (Huang et al.) (2025-03)
- **简介**：200K modality-bridging 多模态 CoT 冷启动 + GRPO + 渐进式思维抑制；7B 模型在 MathVista 73.5%。把 R1 训练范式系统迁移到多模态场景的代表工作。
- **arXiv**：[2503.06749](https://arxiv.org/abs/2503.06749)

#### Visual-RFT: Visual Reinforcement Fine-Tuning (2025-03)
- **简介**：Shanghai AI Lab 把 GRPO 应用于视觉任务——细粒度图像分类、few-shot 检测、reasoning grounding 三种视觉任务都用统一 GRPO + verifiable reward 训练。证明 GRPO 不局限于 reasoning，可推广到广义视觉任务。
- **arXiv**：[2503.01785](https://arxiv.org/abs/2503.01785)

#### MM-RLHF: The Next Step Forward in Multimodal LLM Alignment (2025-02)
- **简介**：CASIA + Kuaishou。120K 人工 8 维细粒度多模态偏好数据 + Critique-Based RM + MM-DPO 全流水线开源；27 基准一致提升、安全性 +60%。是当前最大规模公开多模态 RLHF 配方。
- **arXiv**：[2502.10391](https://arxiv.org/abs/2502.10391)

#### Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning (2024-11)
- **简介**：Shanghai AI Lab 提出。Actor-Critic 多模态——Critic 用 DPO 训练，提供自然语言 critique 而非标量分数。是首个把 DPO 用在多模态 critic 的工作，让 VLM error detection 更精准。
- **arXiv**：[2411.18203](https://arxiv.org/abs/2411.18203)

### 2.9 安全与红队 Reward

#### Provably Safe Reinforcement Learning from Human Feedback (CS-RLHF) (2025-10)
- **简介**：用 ReLU(JC) 固定罚函数代替 Lagrangian 自适应乘子；理论保证 reward 不降 + 成本满足 ≤ d+ε。给出 RLHF 在 helpfulness / safety 双目标下的可证明界，是 safety RLHF 理论方向的代表。
- **arXiv**：[2510.03520](https://arxiv.org/abs/2510.03520)

#### Multi-Turn Safety Alignment via Thought-Guided Adversarial Self-Play (MTSA) (2025-05)
- **简介**：Thought-guided 多轮越狱攻击 + 对抗迭代优化；红蓝 self-play 同时训攻击者与防御者，达到 Nash 平衡。在 Vicuna-7B 上 jailbreak ASR 从 65% 降至 8% 同时不损失 helpfulness。
- **arXiv**：[2505.17147](https://arxiv.org/abs/2505.17147)

#### Safe RLHF-V: Safe Reinforcement Learning from Human Feedback in Multimodal Large Language Models (2025-03)
- **简介**：首个多模态 Safe RLHF 框架——基于 BeaverTails-V 双偏好（helpfulness + safety 各自标注）训 Lagrangian RL；安全 +34.2%、helpfulness +34.3%，把 Safe RLHF 思想成功扩到 VLM。
- **arXiv**：[2503.17682](https://arxiv.org/abs/2503.17682)

#### DOOR: Direct Preference Optimization with One-vs-Other Rejection (2025-03)
- **简介**：把安全对齐拆为 chosen 概率最大化 + 拒绝 token 加权两个目标；W-DOOR 进一步用 weighting 区分不同程度的有害 response。在 jailbreak 攻击下 ASR 显著降低，是 DPO 风格 safety alignment 的代表。
- **arXiv**：[2503.03710](https://arxiv.org/abs/2503.03710)

#### Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs (2025-02)
- **简介**：实证发现窄域 finetuning 即可产生广泛 misalignment——仅用不安全代码 finetune 就让模型在所有任务上变得 misaligned（包括完全无关的对话场景）。挑战了"局部微调局部影响"的默认假设。
- **arXiv**：[2502.17424](https://arxiv.org/abs/2502.17424)

#### Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models (2024-06)
- **简介**：Anthropic 实验。课程化训练 reward hacking 行为后，模型会泛化到"自我修改奖励函数"等更严重的 specification gaming，且 RLHF 安全训练无法消除。是 reward tampering 现象的标志性证据论文。
- **arXiv**：[2406.10162](https://arxiv.org/abs/2406.10162)

#### Aligner: Efficient Alignment by Learning to Correct (2024-02)
- **简介**：北大 + 华为提出。修正未对齐答案比生成对齐答案容易——前置模型后挂一个 Seq2Seq 残差 corrector；7B Aligner 让 GPT-4 helpfulness +17.5、safety +26.9。NeurIPS 2024 Oral，是模块化对齐的代表工作。
- **arXiv**：[2402.02416](https://arxiv.org/abs/2402.02416)

#### Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training (2024-01)
- **简介**：Anthropic 实验。训练带后门的 LLM，证明 RLHF / SFT / 对抗训练**都不能消除欺骗行为**；模型甚至学会更好地隐藏触发条件。是 RLHF safety 局限性最有力的实证论文之一。
- **arXiv**：[2401.05566](https://arxiv.org/abs/2401.05566)

#### Safe RLHF: Safe Reinforcement Learning from Human Feedback (2023-10)
- **简介**：北大提出。Cost model + reward model 分离的 Lagrangian 约束 RLHF——把 safety 建模为约束而非目标的一部分。BeaverTails 数据集来源，是 safety RLHF 的奠基工作。
- **arXiv**：[2310.12773](https://arxiv.org/abs/2310.12773)

#### SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks (2023-10)
- **简介**：UPenn 提出。随机扰动输入 prompt + 多数投票输出；输入级抗扰动认证保证。无需重新训练即可大幅降低 jailbreak ASR，是推理时防御的代表方案。
- **arXiv**：[2310.03684](https://arxiv.org/abs/2310.03684)

### 2.10 综述与基准

- **From Reasoning to Agentic: A Survey of Credit Assignment in Large Language Models** (2026-04)：腾讯综述，CA 47 方法二维分类 + decision tree。[arXiv:2604.09459](https://arxiv.org/abs/2604.09459)
- **The Landscape of Agentic Reinforcement Learning for Large Language Models** (2025-09)：700+ 论文综述。[arXiv:2509.02547](https://arxiv.org/abs/2509.02547)
- **Reinforced MLLM: A Survey on RL-based Reasoning in Multimodal Large Language Models** (2025-04)：多模态 RL 第一篇综述；value-free / value-based 两大范式。[arXiv:2504.21277](https://arxiv.org/abs/2504.21277)
- **TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks** (2024-12)：175 个真实公司任务；Claude 3.5 Sonnet 仅 24% 完成率。[arXiv:2412.14161](https://arxiv.org/abs/2412.14161)
- **VL-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models** (2024-11)：多模态 RM 评测；GPT-4o 仅 65.4%。[arXiv:2411.17451](https://arxiv.org/abs/2411.17451)

---

## 3. OPD（Off-Policy / On-Policy Distillation / Drift）

> 行为策略与目标策略错位时如何修正、teacher 信号如何高效压缩、πθ 相对参考策略的漂移如何监控。

### 3.1 Off-Policy RL：IS / Clipping 设计

#### Trust Region with Sequence Mask Policy Optimization (TRM) (2025-12)
- **简介**：用序列级 mask + KL 替代 ratio 约束信任域；MoE / 长 trajectory 友好。比 GSPO 等序列 IS 更宽容，对训-推 mismatch 鲁棒，是 2025 年末 IS 设计的最新工作。
- **arXiv**：[2512.23075](https://arxiv.org/abs/2512.23075)

#### Smooth Asymmetric Policy Optimization (SAPO) (2025-11)
- **简介**：用平滑 sigmoid clip 替代 hard clip——梯度在 clip 边界附近平滑过渡而非直接归零，避免大批样本梯度断崖。配合 asymmetric clipping 处理正负 advantage，训练曲线显著平滑。
- **arXiv**：[2511.20347](https://arxiv.org/abs/2511.20347)

#### Turn-level Stale-Tolerant PPO for Multi-Turn Agent (Turn-level ST-PPO) (2025-11)
- **简介**：把 importance sampling ratio 从 token 级抬到 turn 级——以整个 turn 内所有 token log-prob 之和取 IS。多轮 agent 比 token-IS 更稳定，且对 stale rollout 更鲁棒。
- **arXiv**：[2511.20718](https://arxiv.org/abs/2511.20718)

#### Second-Moment Trust Region for Off-Policy LLM RL (M2PO) (2025-10)
- **简介**：用 IS 比率二阶矩约束信任域而非比率本身，对长 horizon staleness 比 GSPO 更鲁棒。给出 IS variance bound 的精确刻画，是 off-policy LLM RL 中变分理论的代表工作。
- **arXiv**：[2510.01161](https://arxiv.org/abs/2510.01161)

#### Asymmetric Policy Optimization with Flipped Importance Sampling (ASPO) (2025-10)
- **简介**：发现负 advantage 样本的 IS 修正存在系统性偏差——传统 PPO 的 ratio 在负 advantage 上方向相反。ASPO 提出翻转 importance ratio 处理 negative advantage，让正负样本的修正一致。
- **arXiv**：[2510.06062](https://arxiv.org/abs/2510.06062)

#### Group Sequence Policy Optimization (GSPO) (2025-07)
- **简介**：阿里 Qwen 团队提出。把 token 级 IS 替换为序列级 IS——整条序列共用一个 ratio，避免 token 级方差爆炸。MoE RL 的事实标准，因为序列级 IS 天然容忍 expert 漂移导致的局部 logit 差异。
- **arXiv**：[2507.18071](https://arxiv.org/abs/2507.18071)

#### Geometric-Mean Policy Optimization (GMPO) (2025-07)
- **简介**：用 token IS 的几何均值聚合而非乘积——比序列级（乘积）更平滑、比 token 级（独立）更稳。在 MoE 与 long CoT 双重困难场景下平衡了 GSPO 与 GRPO 的取舍。
- **arXiv**：[2507.20673](https://arxiv.org/abs/2507.20673)

#### Clipped IS Policy Optimization with Stop-Gradient (CISPO) (2025-06)
- **简介**：在 IS 比率上做 stop-gradient——避免梯度被 IS 权重自身放大造成 second-order 自激。简单 stop-gradient 一行代码即可显著降低训练方差，是 IS 修正的工程小技巧。
- **arXiv**：[2506.13585](https://arxiv.org/abs/2506.13585)

#### Truncated Off-Policy Policy Optimization (TOPR) (2025-03)
- **简介**：非对称 IS 截断——正样本与负样本分别处理 IS 上下界。对 negative advantage 做更紧的 IS truncation 抑制 outlier，是 off-policy LLM RL 早期 IS 设计的代表工作。
- **arXiv**：[2503.14286](https://arxiv.org/abs/2503.14286)

#### Min-Prefix Ratio Policy Optimization (MinPRO) (2026-01)
- **简介**：用前缀 IS 比率的 min 做 clip——理论上比 token 级 IS 方差更低，比序列级 IS 更细粒度。在 long CoT + stale rollout 双重难点下稳定优于 GSPO 与 GRPO。
- **arXiv**：[2601.22718](https://arxiv.org/abs/2601.22718)

#### Cumulative Token Prefix Policy Optimization (CTPO) (2026-05)
- **简介**：累计 token 前缀比的 IS 修正——把 token 级、前缀级、序列级三种粒度的 IS 统一在一个连续插值参数下。给出 IS 粒度选择的理论分析，是 IS 设计统一框架的代表工作。
- **arXiv**：[2605.07331](https://arxiv.org/abs/2605.07331)

### 3.2 异步 / Replay / 系统级 Off-Policy

#### Missing Old Logits in Asynchronous Agentic Reinforcement Learning (2026-05)
- **简介**：诊断异步管道中"旧 logits 缺失"的工程问题——actor 更新后 rollout buffer 内的旧 log-prob 不再准确，导致 IS 修正失效。提出 PPO-EWMA 用指数滑动平均估计 stale logits，让异步 PPO 重新可靠。
- **arXiv**：[2605.12070](https://arxiv.org/abs/2605.12070)

#### ROSE: Cooperative Elasticity for RL Rollout (2026-05)
- **简介**：在现有推理集群上采用空闲 GPU 加速 RL 采样；SLO 感知共训练机制保证不影响线上推理服务的 SLO。在 1.20–3.31× 吞吐提升下不损失训练收敛速度，是工业级 RL 的资源调度代表工作。
- **arXiv**：[2605.06534](https://arxiv.org/abs/2605.06534)

#### AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning (2025-05)
- **简介**：蚂蚁集团开源的完全异步 RL 框架——actor / critic / rollout 三组 GPU 完全异步，interruptible rollout 允许中途切策略；配套 staleness-enhanced PPO 处理 stale rollout。是工业 RL 框架的代表，1T 模型可训。
- **arXiv**：[2505.24298](https://arxiv.org/abs/2505.24298)

#### StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation (2025-04)
- **简介**：清华提出。流式 RL 训练——rollout 与 train GPU 完全解耦，rollout 一边生成 trajectory 一边送给 trainer，training 端异步消费。支持异构 GPU（rollout 与 train 用不同型号 GPU）。
- **arXiv**：[2504.15930](https://arxiv.org/abs/2504.15930)

#### Trajectory Balanced Async Rollout for Reinforcement Learning (TBA) (2025-03)
- **简介**：异步 trajectory buffer 设计——按 trajectory length 自动平衡 buffer 内不同长度 rollout 的占比，避免长 trajectory 由于 rollout 慢而被 starve。保证 buffer 数据 freshness 与多样性。
- **arXiv**：[2503.18929](https://arxiv.org/abs/2503.18929)

#### The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization (Async RLHF) (2024-10)
- **简介**：异步 RLHF 的早期系统级研究——给出 actor / RM / reference 三组 GPU 异步部署的实现细节、buffer 配置、IS 修正注意事项。是 RLHF 工程实现的经典手册。
- **arXiv**：[2410.18252](https://arxiv.org/abs/2410.18252)

### 3.3 训练-推理不匹配（TIM / Precision / MoE）

#### DeepSeek-V3.2: Tackling Training-Inference Mismatch in Reinforcement Learning at Trillion Scale (2025-12)
- **简介**：DeepSeek 1T MoE 上的 TIM 治理样板——Keep Routing（训练时 replay 推理时的 routing）+ Keep Sampling Mask + off-policy 序列掩码 + 无偏 KL 估计器四件套。是当前最大规模 MoE RL 的工程参考。
- **arXiv**：[2512.02556](https://arxiv.org/abs/2512.02556)

#### Toward Bitwise Identical Kernels for Distributed RL Training (TBIK) (2025-11)
- **简介**：跨 TP-size 按位相同的归约核——从内核层消除训练与推理引擎之间的数值 mismatch。不同 TP 切分下 forward 结果按位一致，根除一类 TIM 来源。
- **arXiv**：[2511.17826](https://arxiv.org/abs/2511.17826)

#### Bridging the Train-Inference Gap with FP16 (2025-10)
- **简介**：把 rollout 与 train 都切到 FP16 几乎能消除 mismatch——证明很多 TIM 问题源于 BF16 与 FP16 的不同舍入行为。算法 vs 系统的边界——有时把精度统一比设计 IS 修正更划算。
- **arXiv**：[2510.26788](https://arxiv.org/abs/2510.26788)

#### IcePop: Bilateral Masking for Mixture-of-Experts RL (2025-10)
- **简介**：Ling Team 提出。双侧掩码 M(k)=k if k∈[α,β] else 0——只掩去 IS ratio 极端的 token，仅 1-2‰ token 被 mask。MoE 通用，是 R3 之外另一条治理 expert 漂移的轻量方案。
- **arXiv**：[2510.18855](https://arxiv.org/abs/2510.18855)

#### Routing Replay for Mixture-of-Experts Reinforcement Learning (R3) (2025-10)
- **简介**：记录推理引擎的 expert 路由分布，在训练时 replay 让 forward 时的 routing 与 inference 时一致——直接根除 MoE 中"相同输入→不同 expert→不同 logit"的 TIM。100B+ MoE 规模友好。
- **arXiv**：[2510.11370](https://arxiv.org/abs/2510.11370)

#### Quantization Error as Exploration Bonus in RL (QeRL) (2025-10)
- **简介**：反向利用 quantization mismatch——把量化噪声当作 exploration bonus，鼓励 policy 在 quantized inference 下仍保持高熵。化 TIM 之"敌"为"友"，是相当反直觉的设计。
- **arXiv**：[2510.11696](https://arxiv.org/abs/2510.11696)

#### Numerical Nondeterminism in LLM Inference (2025-06)
- **简介**：列举数值非确定性源（线程调度、cuBLAS、radix cache）与 RL drift 的因果链。给出"训练时数值不一致 → 推理时输出漂移 → RL 训练崩盘"的完整链路，是 TIM 现象的诊断手册。
- **arXiv**：[2506.09501](https://arxiv.org/abs/2506.09501)

### 3.4 On-Policy Distillation

#### On-Policy Distillation (Thinking Machines Lab) (2025-10)
- **简介**：Thinking Machines Lab 旗舰博客。Student 自己 rollout，对每个 student-生成 token 计算 reverse-KL 作为 dense per-token advantage 直接做 PG。Qwen3-8B-Base + Qwen3-32B teacher 用 1/10 RL 算力达到 RL 同级 reasoning 性能（AIME'24 74.4% vs 67.6%）。
- **链接**：[Thinking Machines Lab Blog](https://thinkingmachines.ai/blog/on-policy-distillation/)

#### Speculative Knowledge Distillation: Bridging the Gap Between Forward and Reverse KL Divergence (SpecKD) (2025-10)
- **简介**：Propose-and-verify 风格——student 先提议 token，teacher 分布"接受"高置信对齐 token、"拒绝"其余。揭示 KD 中 filter 哪些 token 比 loss 形态（forward vs reverse KL）更关键，给 KD 设计提供新思路。
- **arXiv**：[2510.24021](https://arxiv.org/abs/2510.24021)

#### DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs (2025-03)
- **简介**：KAIST + MS 提出。对比式 KD——teacher response 用 SKL（symmetric KL）、student response 用 SRKL（skew reverse KL），两种不同 loss 形态分别优化两类响应。ICML 2025 Spotlight。
- **arXiv**：[2503.07067](https://arxiv.org/abs/2503.07067)

#### Temporally Adaptive Interpolated Distillation (TAID) (2025-01)
- **简介**：Sakana AI 提出。在 student 与 teacher 分布之间动态插值出"中间分布"作靶子，t=0 时是 student，t=1 时是 teacher。缓解 capacity gap、mode averaging、mode collapse 三大 KD 难题。ICLR 2025 Spotlight。
- **arXiv**：[2501.16937](https://arxiv.org/abs/2501.16937)

#### MiniLLM: Knowledge Distillation of Large Language Models (2023-06)
- **简介**：Microsoft + 清华提出。Reverse-KL 替代 forward-KL；policy gradient 优化；single-step decomposition 降方差。HF TRL 把 MiniLLM 实现为 GKD/MiniLLM/TML 的统一通用版本，2026.01 v6 改名为 *On-Policy Distillation of LLMs*。
- **arXiv**：[2306.08543](https://arxiv.org/abs/2306.08543)

#### On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD) (2023-06)
- **简介**：Google DeepMind 提出。Generalized Knowledge Distillation：L_GKD = (1−λ)·E_Data[D(p_T||p_S)] + λ·E_p_S[D(p_T||p_S)]——同时学 ground-truth 数据与 student 生成数据。TML On-Policy Distillation 是 GKD 在 (λ=1, D=reverse-KL) 下的特例。ICLR 2024。
- **arXiv**：[2306.13649](https://arxiv.org/abs/2306.13649)

### 3.5 Off-Policy KD 对照

#### Distillation Scaling Laws (2025-02)
- **简介**：Apple 提出。估算蒸馏模型性能 = f(总算力, teacher/student 算力分配)；给出运行准则——已有 teacher 或要蒸多个 student 时 KD 优于 supervised pretraining。是 KD scaling law 的开山工作。ICML 2025。
- **arXiv**：[2502.08606](https://arxiv.org/abs/2502.08606)

#### DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (R1-Distill) (2025-01)
- **简介**：用 R1 生成 80 万样本对 Qwen2.5-Math 1.5B–32B、Llama-3.1-8B/70B 做纯 SFT 蒸馏；AIME'24 1.5B 28.9 / 7B 55.5 / 32B 72.6。证明对小模型 off-policy distillation 显著优于直接 RL，与 TML On-Policy Distillation 形成路线对照。
- **arXiv**：[2501.12948](https://arxiv.org/abs/2501.12948)

### 3.6 Policy Drift 监控与缓解

#### Learning Rate as a Correction Layer for Policy Drift (2026-02)
- **简介**：响应式学习率调度——监控 response 长度激增（先于崩溃数十步）作为 drift 早期预警，触发后自动降学习率防止 collapse。是 RL 训练 drift 的工程级早期预警系统。
- **arXiv**：[2602.01826](https://arxiv.org/abs/2602.01826)

#### The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models (2025-05)
- **简介**：上海 AI Lab + 清华。建立经验定律 R = −a·exp(H) + b——性能从熵兑换而来，存在可预测天花板。提出 Clip-Cov（mask 高协方差 token 的梯度）+ KL-Cov（对高协方差 token 加 KL 惩罚），Qwen2.5-32B +6.4 点。是 DAPO Clip-Higher 的原理化升级。
- **arXiv**：[2505.22617](https://arxiv.org/abs/2505.22617)

### 3.7 综述与博客

- **Off-Policy Drift in LLM RL** (Chris Liu, 2026-04)：OPD 60+ 论文一站式入口，是 OPD 方向最完整的二手综述。[chrisliu298.ai](https://chrisliu298.ai/)
- **A Survey on Knowledge Distillation of Large Language Models** (2024-02)：KD algorithm / skill / verticalization 三维分类。[arXiv:2402.13116](https://arxiv.org/abs/2402.13116)
- **Defeating Nondeterminism in LLM Inference** (Thinking Machines Blog)：根因是 batch 依赖 + 浮点非结合性，给出实操修复方案。
- **HF TRL minillm 文档**：把 GKD / MiniLLM / TML On-Policy Distillation 统一为同一公式的工程实现。[huggingface.co/docs/trl](https://huggingface.co/docs/trl)
- **Anyscale: Open Source RL Libraries for LLMs**：11 个开源 RL 框架横向对比（TRL / OpenRLHF / verl / AReaL / NeMo-RL 等）。[anyscale.com/blog](https://www.anyscale.com/blog)

---

## 4. Multi-Agent

> 多个 LLM 协作、辩论、自博弈、coordinator 训练。

### 4.1 Multi-Agent Co-Training

#### Latent Agents: Internalizing Multi-Agent Debate Through Activation Steering (2026-04)
- **简介**：两阶段微调把 multi-agent debate 蒸馏进单 LLM；activation steering 发现 internalize 后形成 agent-specific subspaces，token 用量降到 7%。**multi-agent 训练 + 单 agent 推理是性价比最高的组合**，是 multi-agent 方向最被关注的近期工作。
- **arXiv**：[2604.24881](https://arxiv.org/abs/2604.24881)

#### Multi-Agent Self-Play with Hierarchical Attribution (MARSHAL) (2025-10)
- **简介**：合作+竞争策略游戏 self-play；turn-level advantage estimator + agent-specific advantage normalization；Qwen3-4B 在 AIME +10%。证明 multi-agent self-play 可以稳定训练而不退化。
- **arXiv**：[2510.15414](https://arxiv.org/abs/2510.15414)

#### Multi-Agent Reinforcement Fine-Tuning of Language Models (MARFT) (2025-04)
- **简介**：LaMAS（Language Multi-Agent System）的 RFT 范式——Flex-POMDP 动态依赖建模 + Encoder-Decoder Trust Region；multi-agent advantage decomposition。MATH 比单 agent PPO 高 ~5% 且更稳定，是 LLM 多 agent RL 的奠基工作。
- **arXiv**：[2504.16129](https://arxiv.org/abs/2504.16129)

#### LLM-Empowered Reward Shaping and Observation Enhancement for Multi-Agent RL (LERO) (2025-03)
- **简介**：LLM 同时生成 hybrid reward function + observation enhancement——用 LLM 写自然语言 reward shaping 与 observation 重写规则，外层进化算法迭代选优。开创性地把 LLM 用作 MARL 的"reward + observation engineer"。
- **arXiv**：[2503.21807](https://arxiv.org/abs/2503.21807)

#### Speaking the Language of Teamwork: LLM-Guided Credit Assignment in Multi-Agent Reinforcement Learning (2025-02)
- **简介**：CMU 提出。LLM 当 dense reward 生成器——基于自然语言任务描述给 agent-specific 稠密奖励。弥补了 VDN/QMIX 等经典 MARL 算法在稀疏奖励下次优的问题，把 LLM 当作 MARL 的"裁判"。
- **arXiv**：[2502.03723](https://arxiv.org/abs/2502.03723)

#### LLM-MCA & LLM-TACA: LLMs for Multi-Agent Credit Assignment and Task Allocation (2025-02)
- **简介**：把 multi-agent credit assignment 重写为 sequence improvement + attribution 两个 pattern recognition 任务；LLM 当 centralized reward critic 做 reward decomposition + task allocation。AAMAS 2025。
- **arXiv**：[2502.16863](https://arxiv.org/abs/2502.16863)

#### Multi-Agent Policy Reinforcement Learning for Large Language Models (MAPoRL) (2025-02)
- **简介**：多 LLM 多轮讨论 + verifier，verifier 同时验答案与给 persuasion bonus + correction bonus。证明只有 multi-agent co-training 才能 generalize——单独训练单 LLM 不会涌现协作能力。
- **arXiv**：[2502.18439](https://arxiv.org/abs/2502.18439)

#### SiriuS: Self-Improving Multi-Agent Systems via Bootstrapped Reasoning (2025-02)
- **简介**：experience library + bootstrapped reasoning，outcome-conditioned CA——成功 trajectory 进 library，失败 trajectory 反向作 negative。reasoning 与 QA 任务 +2.86–21.88%。
- **arXiv**：[2502.04780](https://arxiv.org/abs/2502.04780)

#### Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains (2025-01)
- **简介**：generation agent + critic agent 各自专业化，distributed peer reward——每个 agent 既被同伴评估也评估同伴。ICLR 2025，是多 agent 专业化训练的代表方案。
- **arXiv**：[2501.05707](https://arxiv.org/abs/2501.05707)

#### MALT: Improving Reasoning with Multi-Agent LLM Training (2024-12)
- **简介**：generator → verifier → refinement 异构 sequential 三角色；联合 outcome reward 反传到每个角色（每个角色独立 LLM）。MATH +14.14%，是异构 multi-agent 协作训练的代表。
- **arXiv**：[2412.01928](https://arxiv.org/abs/2412.01928)

#### Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning (CORY) (2024-10)
- **简介**：Pioneer/Observer 双 agent Stackelberg 博弈——两个 LLM 一前一后，团队 reward 加和（VDN 风格）；定期角色交换隐式平衡 CA。NeurIPS 2024，是双 LLM 协作训练的早期标杆。
- **arXiv**：[2410.06101](https://arxiv.org/abs/2410.06101)

### 4.2 LLM Debate

#### Multi-Agent Process Rewards via Per-Action LLM-as-Judge (MAPPA) (2025-12)
- **简介**：每个 agent 的 action 用 LLM-as-judge 给 per-action process reward；解决 multi-agent CA 与样本效率两难。AIME +5–17.5pp，ICLR 2026 Workshop 论文。
- **OpenReview**：[s06wgoO65a](https://openreview.net/forum?id=s06wgoO65a)

#### Sequential Debate Reinforcement Learning (SDRL) (2026-01)
- **简介**：把 debate trace 当 RL 训练信号——成功 debate 的回合作 positive trajectory 训练，是 2024 H2 之后 debate 范式从"inference-time tool"向"training signal"转向的代表工作。
- **arXiv**：[2601.22297](https://arxiv.org/abs/2601.22297)

#### Multi-Agent Critic Aggregation for Debate-as-Training (MACA) (2025-09)
- **简介**：把多 agent debate 输出聚合后训 critic——critic 学会识别"哪些 debate 回合质量高"。是 debate-as-training-signal 范式的进阶版，把 debate 数据用得更精细。
- **arXiv**：[2509.15172](https://arxiv.org/abs/2509.15172)

#### Debate as Training Extension (DTE) (2025-05)
- **简介**：把 debate trajectories 转化为 RL 训练样本——每条 debate 末尾的共识答案当 reward 信号反传到所有参与 agent。Debate-as-training-signal 范式的早期确立工作。
- **arXiv**：[2505.15734](https://arxiv.org/abs/2505.15734)

#### If Multi-Agent Debate is the Answer, What is the Question? (2025-02)
- **简介**：MAD 批判性 meta-eval——发现 MAD 不一定 > CoT/SC，在简单任务上甚至更差。提出 Heter-MAD（异构 LLM 组合）作为 MAD 真正有价值的设置。给 MAD 方向打了一剂清醒剂。
- **arXiv**：[2502.08788](https://arxiv.org/abs/2502.08788)

#### Diversity of Thought Improves Reasoning Abilities of LLMs in Multi-Agent Debate (2024-10)
- **简介**：实证异构中等模型组合在 GSM-8K 上达到 91% > 单 GPT-4。证明 MAD 真正的力量不来自模型 size，而来自 diversity injection——多个不同模型互相纠偏比单大模型自审更有效。
- **arXiv**：[2410.12853](https://arxiv.org/abs/2410.12853)

#### Agent4Debate: Towards Generalizable Argument Generation Through Multi-Agent Debate (2024-08)
- **简介**：Searcher / Analyzer / Writer / Reviewer 四专用 agent，竞争辩论——每个 agent 有独立角色与 prompt。是把 debate 做成"专业团队分工"的代表工作。
- **arXiv**：[2408.04472](https://arxiv.org/abs/2408.04472)

#### Should we Rely on Stronger LMs for Reasoning? Rethinking the Bounds of LLM Reasoning (2024-02)
- **简介**：批判性视角——strong prompt 单 agent ≈ MAD；MAD 真正价值是 diversity injection 而非"多次思考"。给 multi-agent 方向提供 contrastive 视角。
- **arXiv**：[2402.18272](https://arxiv.org/abs/2402.18272)

#### SocraSynth: Multi-LLM Reasoning with Conditional Statistics (2024-02)
- **简介**：苏格拉底辩论——可调 contentiousness 等级（争辩激烈程度）+ CRIT reasonableness 评分。把人类辩论的不同强度做成可控参数，是 debate 风格化的代表工作。
- **arXiv**：[2402.06634](https://arxiv.org/abs/2402.06634)

#### Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs (2023-11)
- **简介**：ICML 2024。系统比较 MAD 协议 vs Self-Consistency vs CoT-ensemble，给出 MAD 在哪些任务上有效、哪些任务上无效的实证地图。是 MAD 方法论评估的基础工作。
- **arXiv**：[2311.17371](https://arxiv.org/abs/2311.17371)

#### ReConcile: Round-Table Conference Improves Reasoning via Consensus Among Diverse LLMs (2023-09)
- **简介**：ChatGPT + Bard + Claude2 三 agent 异构 round-table 辩论；confidence-aware aggregation——按每个 agent 在不同问题上的 confidence 加权聚合。StrategyQA +7.7%，是异构 LLM 辩论的代表工作。
- **arXiv**：[2309.13007](https://arxiv.org/abs/2309.13007)

#### ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate (2023-08)
- **简介**：把 MAD 用于 NLG 评测——多角色裁判组（专家、外行、批评家等）协同评价 LLM response。证明 multi-agent 评测比单 LLM-as-judge 更可靠。
- **arXiv**：[2308.07201](https://arxiv.org/abs/2308.07201)

#### Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate (MAD with DoT) (2023-05)
- **简介**：提出 Degeneration-of-Thought（DoT）问题——单 LLM 自我反思容易陷入思维定势。Tit-for-tat 双 agent + judge 结构强制 divergent thinking。EMNLP 2024。
- **arXiv**：[2305.19118](https://arxiv.org/abs/2305.19118)

#### Improving Factuality and Reasoning in Language Models through Multi-Agent Debate (2023-05)
- **简介**：Du et al. 提出 MAD 范式根源——多 LLM 多轮辩论 majority converge 显著提升数学与事实任务准确率。是 multi-agent debate 方向最早被广泛引用的工作。
- **arXiv**：[2305.14325](https://arxiv.org/abs/2305.14325)

### 4.3 Cooperative CA / 多 agent 信用分配

#### MAgICoRe: Multi-Agent, Iterative, Coarse-to-Fine Refinement for Reasoning (2024-09)
- **简介**：Solver / Reviewer / Refiner 三 agent；Reviewer 用 step-level PRM 给 targeted feedback；按问题难度自适应粗/细路径。是 multi-agent + step-level PRM 结合的代表工作。
- **arXiv**：[2409.12147](https://arxiv.org/abs/2409.12147)

#### Provable Multi-Party Reinforcement Learning with Diverse Human Feedback (2024-03)
- **简介**：多方偏好的 RLHF 理论——meta-learning + Nash / Utilitarian / Leximin 三种社会福利函数；首次给出 multi-party RLHF 的 sample complexity 与公平性界限。是 multi-stakeholder alignment 的奠基理论工作。
- **arXiv**：[2403.05006](https://arxiv.org/abs/2403.05006)

#### MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models (2024-02)
- **简介**：把多 agent 多轮 debate 表示为 DAG；GNN-augmented student model 蒸馏。三个目标联合训练：next-token + 正确/错误对比 + graph-based。是把 multi-agent 蒸馏到单 LLM 的代表工作。
- **arXiv**：[2402.01620](https://arxiv.org/abs/2402.01620)

### 4.4 Self-Play / Game-Theoretic

#### SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning (2025-06)
- **简介**：全在线 multi-turn multi-agent RL；role-conditioned advantage estimation（RAE）——为每个角色估独立 advantage。仅在 Kuhn Poker 自我对弈即可让数学/通用推理跨域 +8.6%/+8.4%，证明博弈训练可激发跨域推理能力。
- **arXiv**：[2506.24119](https://arxiv.org/abs/2506.24119)

#### Absolute Zero: Reinforced Self-play Reasoning with Zero Data (AZR) (2025-05)
- **简介**：零外部数据——agent 自我提议任务并解决；coding + math reasoning SOTA 超过用上万人工样本训练的零设置模型。把 RLVR 推向极致：连任务本身都不需要外部提供。
- **arXiv**：[2505.03335](https://arxiv.org/abs/2505.03335)

#### SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning (2025-04)
- **简介**：港大 + 腾讯。critic 通过 adversarial self-play 进化——sneaky generator 故意写难检测的错让 critic 学会更精准识别。ProcessBench 70.8 → 77.7%，**把 step-level reward modeling 从"标注密集"转为"博弈无标注"是标志事件**。
- **arXiv**：[2504.19162](https://arxiv.org/abs/2504.19162)

#### Evolving Alignment via Asymmetric Self-Play (eva) (2024-11)
- **简介**：DeepMind + UChicago 提出。Creator-Solver 非对称博弈——creator 用 minimax-regret 自适应生成 prompt，solver 学解。首次让 LLM 自我进化训练 prompt 分布，把 self-play 从 response 层抬到 prompt 层。
- **arXiv**：[2411.00062](https://arxiv.org/abs/2411.00062)

#### Prover-Verifier Games Improve Legibility of LLM Outputs (2024-07)
- **简介**：OpenAI Superalignment。iterative 训练 verifier + helpful prover + sneaky prover 三角色——helpful prover 学说服 verifier、sneaky 学骗 verifier、verifier 学辨别。结果输出对人类的可验证性显著提升。
- **arXiv**：[2407.13692](https://arxiv.org/abs/2407.13692)

#### Iterative Nash Policy Optimization: Aligning LLMs with General Preferences via No-Regret Learning (INPO) (2024-07)
- **简介**：no-regret learning 让策略自我对弈逼近 Nash 均衡；不需估计 win-rate（区别于 SPPO）。理论保证收敛到 Nash policy，是 game-theoretic alignment 的代表工作。
- **arXiv**：[2407.00617](https://arxiv.org/abs/2407.00617)

#### Self-playing Adversarial Language Game Enhances LLM Reasoning (SPAG) (2024-04)
- **简介**：腾讯提出。attacker / defender 围绕 target word 双人语言博弈——attacker 引诱 defender 说出 target word，defender 避免。RL on game outcome 改善多 reasoning benchmark。NeurIPS 2024。
- **arXiv**：[2404.10642](https://arxiv.org/abs/2404.10642)

#### Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences (DNO) (2024-04)
- **简介**：general preference 下的 batched on-policy 算法——理论 monotonic improvement 保证；不依赖 BT 模型假设（区别于 DPO）。是从理论上突破 BT 假设限制的代表工作。
- **arXiv**：[2404.03715](https://arxiv.org/abs/2404.03715)

### 4.5 LLM-as-Coordinator

#### Conductor: Reinforcement Learning a Multi-Model Orchestrator (2025-12)
- **简介**：7B Conductor 通过纯 RL 自动发现 LLM 之间的协作拓扑 + prompt 工程；recursive 拓扑允许 conductor 把自己当 worker。是 multi-agent coordinator 训练的代表工作。
- **arXiv**：[2512.04388](https://arxiv.org/abs/2512.04388)

#### Trinity: Lightweight Coordinator for Multi-LLM Systems via Evolutionary Search (2025-12)
- **简介**：sep-CMA-ES 优化 coordinator——SLM backbone + lightweight head；< 20K 可学参数。把 multi-LLM coordinator 训练做成超轻量化方案，部署友好。
- **arXiv**：[2512.04695](https://arxiv.org/abs/2512.04695)

#### CoRL: Controller LLM for Cost-Aware Expert Routing (2025-11)
- **简介**：Controller LLM 选择性调度 expert pool——double objective：performance ↑ + cost ↓。在 helpfulness 与 API cost 双目标下 Pareto 优化，是工业部署友好的 router 训练方案。
- **arXiv**：[2511.02755](https://arxiv.org/abs/2511.02755)

#### FlowReasoner: Reinforcing Query-Level Meta-Agents (2025-04)
- **简介**：Sea AI Lab 提出。每个 user query 生成一个独立的 agent system——meta-agent（蒸馏自 R1）通过外部执行反馈做 RL 决定每个 query 用什么子 agent 组合。3 benchmark 比 o1-mini +10.52%。
- **arXiv**：[2504.15257](https://arxiv.org/abs/2504.15257)

#### GPTSwarm: Language Agents as Optimizable Graphs (2024-02)
- **简介**：把 agent 系统建模成 computational graph——node optimization（refine prompt）+ edge optimization（reorganize topology）两类自动优化器，可用 RL 优化连接结构。ICML 2024，是 agent system 自动设计的奠基工作。
- **arXiv**：[2402.16823](https://arxiv.org/abs/2402.16823)

### 4.6 综述与基准

- **A Survey on LLM-based Multi-Agent Systems: Workflow, Infrastructure, and Challenges** (2024-12)：应用维度（agentic workflow / 社会模拟 / SWE）。[arXiv:2412.17481](https://arxiv.org/abs/2412.17481)
- **A Survey on Meta-Thinking in Large Language Models via Multi-Agent Reinforcement Learning** (2025-04)：唯一以 MARL × LLM 元认知为主线的综述。[arXiv:2504.14520](https://arxiv.org/abs/2504.14520)
- **LLM as a Mastermind: A Survey of Strategic Reasoning with Large Language Models** (2024-04)：博弈论框架下 LLM 多 agent 评估。[arXiv:2404.01230](https://arxiv.org/abs/2404.01230)
- **Large Language Model based Multi-Agents: A Survey of Progress and Challenges** (2024-02)：IJCAI 2024，LLM-MA 教科书，profiling / communication / capability growth。[arXiv:2402.01680](https://arxiv.org/abs/2402.01680)
- **TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks** (2024-12)：175 个真实公司任务；Claude 3.5 Sonnet 仅 24% 完成率。[arXiv:2412.14161](https://arxiv.org/abs/2412.14161)
- **MultiAgentBench / MARBLE: Evaluating LLM Multi-Agents in Diverse Scenarios** (2025-03)：协作+竞争+milestone-based KPI；6 类交互场景。[arXiv:2503.01935](https://arxiv.org/abs/2503.01935)
- **MAgIC: Investigation of Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration** (2023-11)：judgment / reasoning / deception / cooperation 四维评估；GPT-4 与 Llama-2-70B 能力差 3 倍。[arXiv:2311.08562](https://arxiv.org/abs/2311.08562)
- **Sotopia: Interactive Evaluation for Social Intelligence in Language Agents** (2023-10)：ICLR 2024，社交智能 / 协调 / 合作 / 交易场景评测。[arXiv:2310.11667](https://arxiv.org/abs/2310.11667)
- **CompeteAI: Understanding the Competition Behaviors in Large Language Model-based Agents** (2023-10)：ICML 2024 Oral，GPT-4 模拟两餐馆竞争；ABM × LLM 范式。[arXiv:2310.17512](https://arxiv.org/abs/2310.17512)

---

## 🤝 贡献指南

欢迎补充！PR 时请遵循以下格式：

```markdown
#### 论文完整标题（简称 / 缩写） (YYYY-MM)
- **简介**：2-4 句话概括论文动机、核心机制（公式 / 方法关键步骤）、关键实验结果与定位（与同类工作的关系或痛点突破）。
- **arXiv**：[XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
```

要求：
1. **完整标题**：使用论文 arXiv 上的完整官方标题，方法缩写放括号内
2. **简介**：≥ 2 句话，覆盖 motivation / 核心机制 / 关键结果或定位
3. 时间窗口：仅收 ≥ 2023.01 的工作；纯 CV / T2I 生成评测请勿 PR
4. 排序：每个小章节内按发表时间倒序（最新置顶）
5. 必须给 arXiv ID 或可访问链接

## 📝 License

[CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) — 论文整理本身无版权；引用请尊重原论文版权。
