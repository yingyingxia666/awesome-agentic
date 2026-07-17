# Awesome-Agentic

> A curated reading list of large-language-model RL papers, organized by four research directions: **Reasoning RL**, **Agentic RL**, **OPD (Off-Policy / On-Policy Distillation / Drift)**, and **Multi-Agent**.

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Last Update](https://img.shields.io/badge/last%20update-2026.07-brightgreen) ![Papers](https://img.shields.io/badge/papers-540%2B-blue) ![Time Range](https://img.shields.io/badge/time-2023.01--2026.07-orange)

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

#### Non-vacuous Generalization Bounds for Reinforcement Learning with Verifiable Rewards（Progressive RLVR） (2026-07)
- **简介**：Yuxuan Zhu、Rohan Alur、Daniel Kang。在十亿参数规模上给出首个 RLVR 参数高效微调的**非平凡泛化界**：将 PAC-Bayes 压缩界适配到该设定，用 Gumbel-max 重参数化处理 token 生成随机性。为落地这些界提出 **Progressive RLVR** 框架，**将 RLVR 与 on-policy distillation、TinyLoRA、模型量化集成**：保留标准 LoRA 微调 84–97% 性能，同时模型压缩率提升 14,796×，在数学/编程/通识推理/Text-to-SQL 四域给出非平凡界（超基座 9–51%，距微调模型 6–11%）。on-policy distillation 是其框架组成部分，作为 §3.4 的理论/压缩视角对照收录。
- **arXiv**：[2607.14506](https://arxiv.org/abs/2607.14506)

#### Verifier-Based Reinforcement Fine-Tuning of Reasoning Models for Thermal Energy Storage Control (2026-07)
- **简介**：Shioda、Terashima、Nagai 用 RLVR 对开放权重推理模型做强化微调（RFT），用于建筑热储能（TES）调度控制。将离线动态规划（DP）的精确动作价值转成对每个候选动作的稠密奖励，仅用 30 条训练 prompt 训练模型作为上层调度器输出小时级热泵设定点。结果：RFT 把开放权重模型的排放从 70.5 降到 61.2 kg-CO₂（DP 最优 60.8）；轨迹分析显示 RFT 主要稳定了可观测的规划模式（候选比较、前瞻、可行性检查）而非创造新策略，且在预测误差/未见工况/迁移到电池任务时具鲁棒性。
- **arXiv**：[2607.12856](https://arxiv.org/abs/2607.12856)

#### SCOPE-RL: Optimizing Reasoning Paths Before and After Success (Scaffolded Chain Optimization with Process Efficiency) (2026-07)
- **简介**：Liu 等 9 人提出 SCOPE-RL，针对 RLVR 稀疏终答奖励「成功前无进度信号、成功后无法区分冗余/局部有缺陷的正确轨迹」的问题，提出保留 GRPO 更新的两阶段稠密化框架：成功前用 Adaptive Scaffolded RL（在答案隐藏的子问题链上加前缀分解的可验证奖励），成功后用 Quality-Aware Process RL（对正确轨迹加正确性门控的过程形状奖励）。在 Qwen3-8B-Instruct（DAPO-Math + Big-Math）上较仅结果 GRPO 平均准确率最高 +11.2pp、推理 token 最高减少 27.1%，在 GSPO 与 Qwen3-0.6B 上增益仍成立。
- **arXiv**：[2607.11506](https://arxiv.org/abs/2607.11506)

#### Selective Left-Shift: Turning Test-Time Compute and Difficulty-based Curation into Training Data for Low-Resource Code Generation (2026-07)
- **简介**：Didula Samaraweera 等（含 WSO2 的 Srinath Perera）针对低资源编程语言（Julia、Ballerina）小模型的「SFT 缺数据、推理时扩展太贵、从零 RL 优势近零」三难，提出三段式流水线：先把推理时算力「左移」到离线数据合成引擎（用编译器 / 测试反馈迭代产出**已验证**样例），再 SFT 注入语法先验，最后用「语言无关 I/O 测试」为验证信号做 **RLVR**（SFT 先验把探索约束在语法正确区）。在 Qwen3-8B 上，MultiPL-E Julia pass@1 最多 +7.6、Agnostics LiveCodeBench +14.2（对比 SOTA），且数据 / 成本仅为前 SOTA 的一小部分；消融确认 SFT 阶段与执行式奖励缺一不可。
- **arXiv**：[2607.07748](https://arxiv.org/abs/2607.07748)

#### Self-Review Reinforcement Learning (SRRL) with Cross-Episode Memory and Policy Distillation (2026-07)
- **简介**：Muhammad Zain Amin、Kibele Sebnem Yildirim 提出 **SRRL**，在每个 RL episode 中嵌入显式「自我复盘」步骤：首答失败时模型先生成 self-review 指出错因，据此条件化改进第二次尝试。与 Reflexion 等推理时反思不同，SRRL **用策略梯度优化 self-review** 并通过选择性蒸馏把改进内化进 base policy 使其跨 episode 持久化，同时用 cross-episode memory 保存成功的 self-review 供相似任务复用。在 Qwen3-4B、OLMo-3-7B 的 GSM8K 上，最终奖励与学习效率均稳定优于标准 RLVR（GRPO optimizer）基线。
- **arXiv**：[2607.05541](https://arxiv.org/abs/2607.05541)

#### TREK: Distill to Explore, Reinforce to Refine (2026-07)
- **简介**：Meta 等团队（Yuanda Xu、Ran He、Alborz Geramifard 等）针对 GRPO 在「正确解落在学生 on-policy 支撑集之外」的难题上停滞的问题，提出 **TREK（Teacher-Routed Exploration via Forward KL）**：不把蒸馏用于模仿，而用于**扩展探索支撑集**——先定位低通过率难题，向 proposal 源（黑盒 / 白盒 teacher 或自身加推理时上下文）取已验证候选，按当前学生似然保留 top-r，用一小段 forward-KL 把这些已验证模式「拉进」学生支撑集，再回到标准 on-policy GRPO 精炼。用 DeepSeek-V4 proposals，Qwen3-8B 的 AIME 2025 从 36.9→40.3、AIME 2024 从 47.9→51.1（avg@16），自上下文变体无外部 teacher 也达 38.5 / 49.6；agentic 任务 ALFWorld 75.8→82.8、ScienceWorld 12.5→26.7。
- **arXiv**：[2607.05339](https://arxiv.org/abs/2607.05339)

#### Verifiable Rewards for Calibrated Probabilistic Forecasting (2026-06)
- **简介**：Sadanand Singh 等针对「RLVR 原则上可训练校准的概率预测器（Brier 等 proper scoring rule 仅由结果计算、期望下由真实概率最小化），实践中却反而恶化校准」的问题展开研究。聚焦 **aleatoric forecasting**（预测本身即输出、标签为单次随机结果），以 NFL 局内胜率为测试床、以博彩市场为参照。核心机制：奖励逐回合真实结果会失败（单结果是噪声目标、策略梯度会污染 CoT），故提出**可验证、无标签奖励**——由历史结果估计的 state-conditioned 经验胜率以去除标签噪声，并通过直接预测或 **gradient mask** 让梯度不作用于推理链。仅用该奖励训练（无人工标签、无 SFT），7B 模型经直接预测即达到博彩市场的校准水平，且比零样本前沿模型校准更好；掩码梯度（而非丢弃 CoT）能保留推理，普通 CoT 训练则会破坏它。属 RLVR 奖励设计 + 反 CoT 污染的新工作。
- **arXiv**：[2607.00164](https://arxiv.org/abs/2607.00164)

#### Transferability for General Reasoning: An Automated Curriculum for Multi-Domain RLVR (TAC) (2026-06)
- **简介**：MPI + 多伦多大学（Yongjin Yang、Bernhard Schölkopf、Zhijing Jin 等）针对多领域 RLVR（数学/编程/科学）课程固定或手调、且对"某域更新是否惠及其余域"无感知的问题，提出 **Transfer-Aware Curriculum (TAC)**：bandit 式在线课程，复用 RL 已有信号——per-domain advantage 表征本地可学习性、GRPO 步的 projected gradient 经梯度几何对齐估计跨域可迁移性（<1% wall-clock 开销）。在六域推理套件上，Qwen3-1.7B / Llama3.2-3B 取得最佳宏平均准确率，较 learnability-only bandit 最多 +2.8 点（相对 10%），消融显示去掉可迁移性项后性能骤降。
- **arXiv**：[2606.25178](https://arxiv.org/abs/2606.25178)

#### Provable Benefits of RLVR over SFT for Reasoning Models: Learning to Backtrack Efficiently (2026-06)
- **简介**：Stanley Wei、Juno Kim 从理论上解释"为何强化微调比纯 SFT 更能提升推理"。把 CoT 推理建模为图上的寻路问题，证明：仅用黄金最短路径（无负例）训练的 SFT **学不会高效回溯**，而 RLVR 仅靠 outcome reward 即可学会从死胡同高效回溯，二者在推理期算力上形成**指数级分离**；并表明 RLVR 模型让其学到了"推理链中困难决策点的位置"，从而更好地分配推理期算力。还证明 RLVR 模型的推理轨迹可蒸馏回 base 模型使其同样学会高效回溯。
- **arXiv**：[2606.22938](https://arxiv.org/abs/2606.22938)

#### Learning at the Right Pace: Adaptive Data Scheduling Improves LLM Reinforcement Learning (ADS) (2026-06)
- **简介**：JHU + Rice（Zicheng Xu、Vladimir Braverman 等）针对 RL 后训练普遍采用均匀采样、忽视数据语义结构与策略能力变化的问题，提出双层数据调度框架 **ADS**：cluster 层按语义模式组织样本并维护自适应跨簇分布以巩固当前进度；sample 层在簇内持续采样 policy-boundary 样本以提供信息量更高的相对优势。在 3 个 LLM × 7 个推理 benchmark 上较 GRPO 平均提升 5.2%，且对不同目标设计的 RL 方法均能稳定增益，显示其作为通用数据调度策略的潜力。
- **arXiv**：[2606.22305](https://arxiv.org/abs/2606.22305)

#### When Do Intrinsic Rewards Work for Code Reasoning? A Comprehensive Study (2026-06)
- **简介**：Purdue + UC Berkeley（Xiaolong Jin、Xuandong Zhao、Dawn Song 等）对"无 ground-truth 的内在奖励（RLIF，如多数投票 / 置信度打分）能否迁移到代码推理"做系统实证研究。在 LiveCodeBench 上系统评测代表性 certainty-based RLIF 方法：发现这类方法早期有收益但**不可避免地坍缩**——模型逐步缩短输出、丧失推理能力，坍缩速度对样本量与温度敏感；且用作 RLVR 初始化时相比从零训练无显著增益。给出在代码推理上使用内在奖励的可操作建议，明确其前景与边界。
- **arXiv**：[2606.20881](https://arxiv.org/abs/2606.20881)

#### RASFT: Rollout-Adaptive Supervised Fine-Tuning for Reasoning (2026-06)
- **简介**：NJIT + Texas A&M 等的 Mengnan Du 团队提出 **RASFT**，把 SFT 当作 policy-aware 信号校准：依据 verified on-policy rollouts 估计 problem-level solvability，policy 弱时强化 expert 引导、policy 已掌握时放松模仿并融入正确 self-rollouts；再用"frozen reference 与 current policy 的 clipped inverse ratio"约束 policy drift。在 6 个数学 + 2 个代码 benchmark、多模型规模上系统优于 SFT、SFT 变体与代表性 RL（GRPO/DAPO）方法。是 RLVR / RL 与 SFT 的衔接路线代表作。
- **arXiv**：[2606.07006](https://arxiv.org/abs/2606.07006)

#### A Pre-Registered Causal Partition of Self-Consistency Elicitation and Reward Design in RLVR (2026-06)
- **简介**：单作者 Yuze Gao 提出对 RLVR 的"自一致性诱导 vs 奖励设计"做 **预注册因果分解**：把朴素估计量 `naive = acc(TRUE) - acc(RANDOM)` 拆为 `total = null + elicit + rd`，并在 5 档 prior 强度的可控 tabular-GRPO 模拟器上度量每一项。结果显示在 prior=0.20 时 reward-design 仅占朴素估计量 0.139；2×2×2 阶因子设计确认非可加（交互比 0.385）；并对两篇已发表 RLVR 工作做"再审计"，给出 ELICITATION DOMINATED（占比 0.98）与 REWARD DESIGN DOMINATED（占比 1.18）的判决，提供一键复用 harness。强调即使是负面 flip 结果也提交，是 RLVR 因果学派的一篇方法论新工作。
- **arXiv**：[2606.05932](https://arxiv.org/abs/2606.05932)

#### Right Makes Might: Aligning Verified Hidden States Empowers RL Reasoning (Hidden-Align) (2026-06)
- **简介**：上海AI Lab + 中科大等团队（13 位作者）观察到正确 rollouts 在"answer marker"前一个 anchor token 处的 last-layer hidden state 自发收敛（cosine ~0.84），但仍残留 reasoning-path 噪声。提出 **Hidden-Align** —— 在 anchor 位置对正确轨迹的 hidden states 加 L2 / 方向对齐辅助损失，**零训练 / 推理 overhead**。在 8 个数学 benchmark 上分别对 Qwen3-1.7B / 4B / 14B 的 DAPO 基线提升 pass@1 平均 **+3.8 / +6.2 / +5.4** 个百分点，并保持 pass@k 提升。
- **arXiv**：[2606.03234](https://arxiv.org/abs/2606.03234)

#### Learning to Solve, Forgetting to Retain: Correct-Set Turnover in RLVR (2026-06)
- **简介**：中科院信工所 Chenxu Yang 等指出 RLVR 训练中"提升的准确率"掩盖了一个隐性代价 —— 已掌握题目随训练进程被悄悄"忘掉"，作者将此命名为 **correct-set turnover**。理论与实证给出 **repair-window principle**：错过修复窗口后修复成本陡升。提出 retention-aware **review** 机制（pre-rollout batch replacement，零额外 rollout），在图像-文本 / 视频 / 纯文本共 20 个 benchmark 上稳定优于 GRPO、DAPO 与 replay 基线，并验证跨 Qwen3-VL、Qwen2.5-Math 的算法泛化性。
- **arXiv**：[2606.03087](https://arxiv.org/abs/2606.03087)

#### A Local Perturbation Theory for Cross-Domain Interference and Recovery in Multi-Domain RL (2026-06)
- **简介**：天津大学 Lei Yang、Deyi Xiong 等给出多领域 RLVR 的 **局部扰动理论**：单域 RL 产生稀疏、低幅度、活跃路由部分重叠的参数编辑，仅靠"全模型梯度正交"无法解释互扰；证明后续域训练对前域损害主要来自 **二阶损害项**，并集中在低维共享冲突子空间；据此设计短促 domain-refresh 实现选择性恢复。Code → Math → QA → CW 后做 Re-Math，把 Math 从 57.66 恢复到 66.04，平均分到 66.39 全 SOTA；并展示无训练 rollback 在 Math-QA 对的代理坐标上部分恢复。
- **arXiv**：[2606.02398](https://arxiv.org/abs/2606.02398)

#### Binary Rewards and Reinforcement Learning: Fundamental Challenges (2026-05)
- **简介**：Naver Labs 的 Marc Dymetman 给 RLVR 训练中观察到的 "diversity collapse"（pass@1 涨、pass@k 反而跌于 base）一个结构性解释。论文证明二元奖励让 expected-reward 最大化退化为无穷多解的退化集合，而 KL-control 在 β→0 极限下选出唯一不退化解 p\*=a(·|𝒴₁)（base 在合法集合上的条件分布），但在模型 misspec 时 β 下降反而让策略坍缩到极少数高 validity 输出而非 p\*。给出 β↔target validity rate μ 的显式换算公式，并实证演示该坍缩机制。
- **arXiv**：[2605.02375](https://arxiv.org/abs/2605.02375)

#### Adaptive Negative Reinforcement for LLM Reasoning: Dynamically Balancing Correction and Diversity in RLVR (2026-05)
- **简介**：在 NSR（Negative Sample Reinforcement，只惩罚错样本）框架上提两条扩展：A-NSR 用时间相关的 schedule 把训练前期的强纠错逐步过渡到后期的细更新；CW-NSR 按归一化 sequence likelihood 给每条错样本不同的惩罚权重——模型越自信的错越重罚、探索性的错少罚。给出 token-level 更新的形式分析，论证其等价于 prior-guided probability redistribution，可对抗过拟合。在 Qwen2.5-Math-1.5B 上 MATH/AIME 2025/AMC23 验证。
- **arXiv**：[2605.07137](https://arxiv.org/abs/2605.07137)

#### Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning (ReasonMaxxer) (2026-05)
- **简介**：USC 团队挑战 "RL 教模型新能力" 的范式。Token 级分析显示 RL 的有益更新是 sparse 且可预测的：仅 1–3% token 位置受影响，被提升的 token 100% 在 base model top-5 内，且 base model 自身的 entropy 即可定位这些位置。基于此提出 ReasonMaxxer：仅在 entropy-gated 决策点用对比损失，几百次 base rollout、单卡分钟级训练即可匹配/超越完整 RL，训练成本下降约三个数量级——给 RLVR 必要性提供了反向证据。
- **arXiv**：[2605.06241](https://arxiv.org/abs/2605.06241)

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

#### Where Should RL Post-Training Compute Go? Model Size, Search, Learning, and Feedback (RACE) (2026-07)
- **简介**：Wilhelm、Kao 针对「固定后训练 FLOP 预算该怎么分」这一决策问题，提出面向 GRPO 后训练的 FLOP 记账框架，把算力分解为 rollout/search、policy-update/learning、reward/feedback-model 评估三块。在 LoRA 适配的 Qwen2.5 策略上发现「条件性分配前沿」：最优分配随模型规模、算力预算、奖励系统与评估目标而变；规则奖励几乎把非更新算力全花在 rollout，而 PRM 式反馈会明显占用 reward-model 推理算力。提出 RACE 作为在昂贵验证前定位分配区制的诊断式 pilot-grid 协议。
- **arXiv**：[2607.13389](https://arxiv.org/abs/2607.13389)

#### Max Out GRPO Signal: Adaptive Trace Prefix Control for Hard Reasoning Problems (AdaPrefix-GRPO) (2026-07)
- **简介**：单作者 Vladislav Beliaev 针对 GRPO 在「组内无一 rollout 成功→组相对优势消失→最想学的前沿难题贡献零梯度」的失效，提出 **AdaPrefix-GRPO**：把「预置参考解正确前缀的长度」作为难度连续旋钮，并做成**反馈控制器**——训练中动态调节每题给多少前缀、把成功率维持在梯度信号最大的 ~50%，随后完全撤除辅助使部署模型独立解题。在同等训练 FLOPs 下，0.6B 模型在训练分布留出题上准确率超过 GRPO 两倍（2.1×），Qwen3-1.7B 达 1.6×、AIME 达 1.7×，同时把 trace 长度约减半；实现上仅需数据准备 + 前缀 token 的 loss mask，模型越小增益越大。
- **arXiv**：[2607.07674](https://arxiv.org/abs/2607.07674)

#### Graph-Native Reinforcement Learning Enables Traceable Scientific Hypothesis Generation through Conceptual Recombination (Graph-PRefLexOR) (2026-07)
- **简介**：MIT 的 Subhadeep Pal、Markus J. Buehler 等提出 **Graph-PRefLexOR**——用 GRPO 微调的一族 graph-native 推理模型，将推理组织为「机制探索 / 图构建 / 模式提取 / 假设综合」的显式阶段，把神经语言生成与符号关系结构耦合，使因果连接可构建、可检视、可复用。在材料科学与力学文献的 100 道开放式问题上较对应 base 模型提升 40–65%（推理可追溯性增益最大），embedding 分析显示约 2–3× 的语义多样性；测试时图扩展表明额外算力主要增加**有界语义空间内的长程概念重组**，而非单纯扩大语义覆盖。属 GRPO 结构化推理 + 测试时扩展的跨域（材料）代表作。
- **arXiv**：[2607.00924](https://arxiv.org/abs/2607.00924)

#### BV-Blend: Uncertainty-Weighted Historical Baselines for Stable Critic-Free RL with Verifiable Rewards (2026-06)
- **简介**：Yupeng Chang、Yuan Wu、Yi Chang 针对 GRPO 式 critic-free RLVR 的核心不稳定性——优势估计依赖 prompt-local（组内）奖励统计，当组内 rollout 奖励全同（二值 verifier 冷启动时常见）时组内方差为零、group normalization 产生零优势而阻碍学习。提出 **BV-Blend**：为每个语义簇维护 EMA 追踪的奖励矩（均值/方差），由 SEM（标准误）代理导出置信权重，用该权重把**历史矩与 prompt-local 统计**混合成标准化优势，供 PPO 式 clipped 更新。在可验证推理 benchmark 上提升训练稳定性与性能，并在 group-normalized 方法会停滞的区间仍保持鲁棒。属 GRPO 优势估计稳定化的算法工程改造。
- **arXiv**：[2606.28707](https://arxiv.org/abs/2606.28707)

#### Beyond Penalizing Mistakes: Stabilizing Efficiency Training in Large Reasoning Models via Adaptive Correct-Only Rewards (ACOER) (2026-06)
- **简介**：高丽大学（Jungseob Lee、Chanjun Park、Heuiseok Lim 等）研究长度惩罚奖励接入 GRPO 时频繁触发 reward collapse 的机理：GRPO 的组归一化在"错误答案被持续长度惩罚"时产生发散优势，故惩罚错误答案长度的方法在持续优化下结构性易坍缩；而仅惩罚正确答案虽避免主失效，却仍受过度压缩驱动的随机坍缩影响。提出 **ACOER**：将简洁性奖励隔离到正确补全、并以动态预算归一化与控制环惩罚调整防止随机压缩。在多个数学推理 benchmark 上准确率优于 base、同时 token 生成减少 60% 以上。
- **arXiv**：[2606.22716](https://arxiv.org/abs/2606.22716)

#### Demystifying Hidden-State Recurrence: Switchable Latent Reasoning with On-Policy Reinforcement Learning (SWITCH) (2026-06)
- **简介**：作者 Jiayu Yang、Chao Chen 等 9 人。解决潜在思维链（latent CoT，用连续隐状态递归替代可见推理轨迹）难以用标准 on-policy RL 优化且难以因果解释的问题。② 关键洞见：一对显式边界 token（`<swi>` 进入 / `</swi>` 退出潜在模式）即可同时解决两个问题——因边界是普通离散 token，**GRPO 策略比率在每个决策点都有良定义**；同时锚点为探测与因果干预提供入口。训练采用 visible-to-latent 课程 + Switch-GRPO 目标，将梯度传播过递归潜在计算。③ 在相近规模下持续超越此前隐状态递归潜在推理方法；机制分析表明 `<swi>` 是可学习的局部化切换策略，其潜在步骤执行因果重要计算且集中于进入时单一隐状态转换。
- **arXiv**：[2606.13106](https://arxiv.org/abs/2606.13106)

#### N-GRPO: Embedding-Level Neighbor Mixing for Enhanced Policy Optimization (2026-06)
- **简介**：作者 Xukun Zhu、Hang Yu、Peng Di、Linchao Zhu（疑似浙江大学，代码组织 ZJUSCL；ACL 2026 Findings）。针对 GRPO rollout 阶段探索的根本权衡——token 级采样产生仅措辞不同的冗余轨迹，而 embedding 级随机噪声破坏语义一致性。② 核心机制"语义邻居混合（Semantic Neighbor Mixing）"：动态混合锚点 token 与其最近语义邻居的 embedding 构造输入表示，在注入多样性的同时严格遵循局部语义流形。③ 在 DeepSeek-R1-Distill-Qwen 多个规模上持续优于强基线，并在分布外（OOD）任务上表现稳健泛化。
- **arXiv**：[2606.10768](https://arxiv.org/abs/2606.10768)

#### Representation-Aware Advantage Estimation: Your Reward Model Provides More Than A Scalar Output (GraphAE) (2026-06)
- **简介**：作者 Guozheng Li、Xiyan Fu、Yiwen Guo。针对 RLHF 仅用标量奖励（噪声大、无法捕捉细粒度偏好）的问题，提出**表征感知的优势估计**：利用奖励模型隐藏状态作为辅助信号。② 核心机制 GraphAE 将每个采样组视为图——节点为响应、边为响应在 RM 隐空间的相似度，通过图传播计算优势，使每个样本融合邻居上下文；轻量可插拔，可集成进 GRPO / GSPO / RLOO。③ 实验在 Arena-Hard-v0.1 上最高 +6.3、AlpacaEval 2.0 上 +8.27、MT-Bench 上 +0.22。
- **arXiv**：[2606.10528](https://arxiv.org/abs/2606.10528)

#### MDP-GRPO: Stabilized Group Relative Policy Optimization for Multi-Constraint Instruction Following (ACL 2026 Main) (2026-06)
- **简介**：德黑兰大学 Salmani-Zarchi 等针对离散低分散奖励下 GRPO 的三类病态（**low-variance amplification、mean-centering blindness、zero-variance collapse**），提出 **MDP-GRPO**：① 多温度采样增加奖励分散；② 双锚 advantage 在同质组中恢复梯度并阻断 mean-centering blindness；③ 基于 Kahneman-Tversky 前景理论的 prospect shaping 限制更新并惩罚违例；④ 非对称 KL 正则。在 FollowBench / IFEval / 自建多约束数据集上把 Llama-3.2-3B 的 strict-constraint 满足率提升至多 **+5.0**，小 group size 下仍稳定收敛并保持 MMLU/ARC。
- **arXiv**：[2606.06058](https://arxiv.org/abs/2606.06058)

#### Hint-Guided Diversified Policy Optimization for LLM Reasoning (HDPO) (2026-06)
- **简介**：苏州大学 + 蚂蚁集团（Zhiyu Cao 等）提出 **HDPO**，两阶段 RL 框架激励"propose-select-think"轨迹：① **Cold Start for Structured Reasoning**（SFT 让模型先列候选解纲再选最可靠者）；② **Hint-Guided Diversified RL**（diversity 调度 + reliability 奖励，鼓励多样且可靠的解）。Hit@N 实验显示相同采样次数下 HDPO 准确率显著高于 GRPO，尤其在尝试次数有限时差距更大。
- **arXiv**：[2606.03021](https://arxiv.org/abs/2606.03021)

#### HMPO: Hybrid Median-length Policy Optimization for Chain-of-Thought Compression (2026-06)
- **简介**：中山大学 Pan Zhou 团队提出 **HMPO**，单阶段 RL 框架做 CoT 压缩。三大组件：① 自成功 rollouts 推出的 **adaptive median 预算**（无需手调长度）；② cosine-decay token reward 平滑长度惩罚；③ **乘法式 reward** 强行优先正确性以阻断 trivial reward hacking。仅用数学数据训练，在数学/代码/科学/指令任务上同时泛化；从 9B 到 122B（含 MoE）规模，达 **19%–46% token 压缩**且准确率几乎不掉，训练成本远低于现有多阶段基线。
- **arXiv**：[2606.01934](https://arxiv.org/abs/2606.01934)

#### EP-GRPO: Entropy-Progress Aligned Group Relative Policy Optimization with Implicit Process Guidance (2026-05)
- **简介**：系统量化 GRPO 的三类 credit-assignment 失败：token 粒度均匀化、step 极性误判（把对的步骤负向更新）、zero-variance collapse 抹杀 outcome 梯度。EP-GRPO 用三件套修：entropy-gated modulation 把 sequence advantage 转 token 权重；policy divergence 锚定 outcome advantage 提供 token 级方向反馈（无外部 RM）；累积熵映射做 progress-aligned advantage 归一化，在零方差下仍保持梯度流。在数学基准上一致超 GRPO 系列。
- **arXiv**：[2605.04960](https://arxiv.org/abs/2605.04960)

#### DGPO: Distribution-Guided Policy Optimization for Fine-Grained Credit Assignment (2026-05)
- **简介**：Hongbo Jin / Rongpeng Zhu 等。一句话：用有界的 Hellinger 距离替换无界 KL，再用 entropy gating 调制 token 级信用分配。指出 Reverse KL 在低先验 token 上分母趋零会触发梯度爆炸、使模型陷入 mode-seeking 保守；Hellinger 距离 ∈ [0,1] 严格有界、自然稳定，配合 entropy gate 区分 "深思熟虑的探索" 与 "自信的幻觉"，再用 softmax 把 sequence 级 advantage 重分配到 token 级。Qwen2.5-32B 上 AIME 2024 60.0% Avg@32 / AIME 2025 46.0% Avg@32，超 GRPO/DAPO；梯度方差 -41%、mode collapse -68%。
- **arXiv**：[2605.03327](https://arxiv.org/abs/2605.03327)

#### Beyond Mode Collapse: Distribution Matching for Diverse Reasoning (2026-05)
- **简介**：上海 AI Lab Kai Chen / Dahua Lin 团队。直击 GRPO 类 on-policy RL 的 mode collapse：概率质量集中到单一解，削弱了多解推理能力。提出 distribution matching 目标，维护一个学到的解类型分布而非塌缩到最高奖励 path，同时保持答对率。给推理任务的 RL 设计一个对应于 "保多样性" 的可优化形式约束。
- **arXiv**：[2605.19461](https://arxiv.org/abs/2605.19461)

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

#### Reason, Reward, Refine: Step-Level Errors Corrections with Structured Feedback for Physics Reasoning in Small Language Models (2026-07)
- **简介**：Raj Jaiswal、Rajiv Ratn Shah 等（IIIT-Delhi / NII）针对小模型物理推理「一步出错、后续全崩」的结构性失败，提出**步级奖励框架**：定位**首个推理错误**、生成针对性的结构化反馈，并用带 KL 正则的策略梯度训练模型修订解答，全程**不暴露 ground-truth 解作为生成目标**。与依赖步级标注的方法不同，无需构造偏好数据，外部 verifier 仅在训练时使用。在五个物理 benchmark 上较 CoT 提示 +17~20%、较最强基线 +10~16%，计算错误 56.9%→23.5%、误解错误 22.3%→12.0%（最佳情形），概念错误 89.7%→68.7% 仍为最难消除的失败模式。
- **arXiv**：[2607.05199](https://arxiv.org/abs/2607.05199)

#### Process Advantage Signal Shaping: A Paradigm-Agnostic Middleware for Process-Supervised RL in LLM Reasoners (PASS) (2026-06)
- **简介**：Chao Wang、Hongtao Tian、Ting Yao、Wenbo Ding 等指出：在 GRPO 的组标准化优势之上叠加 step-level 过程信号（学习式 PRM 或 on-policy-distillation KL）会暴露三类结构性病理——过程/结果/格式流在组标准化处的**通道污染**、过程信号粒度与逻辑决策粒度的**分辨率错配**、以及 return-to-go 求和带来的**累积陷阱**（依信号符号引发长度膨胀或探索截断）。提出 **PASS** 中间件，插在任意标量 step-level 过程信号与 GRPO clipped surrogate 之间，对症下三招：Advantage Fusion（三流在组内独立标准化）、Chunk-by-Value（由信号自身导出 value-homogeneous 分块并块内广播 credit）、Divide-Length（把累积目标转为平均 value-density 分）。在两域（数学推理用学习式 PRM、多跳 QA 用 on-policy-distillation KL）× 两种组标准化算子下，PASS 每种设置均较对应 GRPO 基线取得一致 pass@1 增益。属 PRM/过程监督与 GRPO 融合的方法论中间件。
- **arXiv**：[2606.29296](https://arxiv.org/abs/2606.29296)

#### Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents (2026-06)
- **简介**：UW-Madison（Changdae Oh、Sharon Li 等）针对 agentic 场景下 PRM 难以构建（长 horizon、不可逆动作、随机环境反馈使人工标注与 MC 估计都不可行）的问题，证明 RL 后训练本身已蕴含有效的步级打分要素：在一般随机 MDP 下推导出隐式优势——RL 训练策略与参考策略的 **log-probability ratio 恰好恢复最优优势函数**（progress advantage），无需任何专门的 reward model 训练，且 annotation-free、domain-agnostic。在 test-time scaling、不确定性量化、失败归因三类应用、5 个 benchmark、4 个模型族上一致优于 confidence-based 基线，并超过专门训练的 reward model。
- **arXiv**：[2606.26080](https://arxiv.org/abs/2606.26080)

#### Process-Verified Reinforcement Learning for Theorem Proving via Lean (2026-06)
- **简介**：KAIST（Minsu Kim、Se-Young Yun）提出把 Lean 证明助手本身当作**符号化过程 oracle**，在训练中同时提供 outcome 级与细粒度 tactic 级的已验证反馈。将证明尝试解析为 tactic 序列，借 Lean 的 elaboration 标出局部 sound 步骤与最早失败步，产生植根于类型论的稠密、verifier-grounded 信用信号；并以 first-error propagation 与 first-token credit 方法将这些结构化奖励纳入 GRPO 式目标，平衡 outcome 与 process 级优势。在 STP-Lean、DeepSeek-Prover-V1.5 上，tactic 级监督在 MiniF2F、ProofNet 等多数设置优于 outcome-only 基线。
- **arXiv**：[2606.20068](https://arxiv.org/abs/2606.20068)

#### The Hidden Bias of Process Reward Models: PRISM for Rewarding the Right Reasoning (2026-06)
- **简介**：作者 Aakriti Agrawal、Souradip Chakraborty、Furong Huang 等 9 人。揭示 PRM 因步骤级训练数据严重不平衡而产生的隐藏偏差：标准交叉熵训练放大该偏差，使 PRM 对"看似合理但实际错误"的步骤过度给分、产生高假阳性率，且假阳性具不对称下游危害（主动将 Best-of-N、引导解码、策略优化引向错误推理）。② 提出 PRISM（Precision Ranking for Improved Step Modeling），策略感知的 PRM 训练框架：从逐点标签拟合转为对比式步骤级比较 + 时间前瞻策略生成的硬负样本 + 难度感知课程优化对比间隔，无需新人工标注。③ 在 PRMBench 上假阳性降低 22%、macro F1 优于强判别式 PRM；下游引导解码准确率最高 +22%、Best-of-N 最高 +33%。
- **arXiv**：[2606.09078](https://arxiv.org/abs/2606.09078)

#### SCI-PRM: A Tool Aware Process Reward Model for Scientific Reasoning Verification (KDD 2026 AI4Science) (2026-06)
- **简介**：港理工 + 上海AI Lab（Xiangyu Zhao、Lei Bai 等）首次把 PRM 从数学拓展到生物 / 化学 / 物理科学推理，构造 **SCIPRM70K** 大规模 Chain-of-Tool 轨迹数据集（推理与工具调用交织），训练 **Sci-PRM** 在一次推理内对 tool 选择、执行精度和结果解读做细粒度监督。两类应用：(1) Best-of-N 测试时 scaling；(2) 接入 RL 作为稠密奖励，缓解 advantage-disappearance 问题，使 base model 突破现有性能天花板。
- **arXiv**：[2606.04579](https://arxiv.org/abs/2606.04579)

#### StepPRM-RTL: Stepwise Process-Reward Guided LLM Fine-Tuning for Enhanced RTL Synthesis (DAC 2026) (2026-06)
- **简介**：IBM Research（Vijayaraghavan 等）面向 Verilog/VHDL 长程推理与多步依赖，提出 **StepPRM-RTL**：从 canonical 解构造逐步 trajectory（每步 = rationale + 增量代码改动），训练 **PRM** 对中间步骤打分，用作 RAFT fine-tuning 的稠密反馈；MCTS 探索替代推理路径丰富训练集。在 Verilog/VHDL benchmark 上比最佳 prior 方法在功能正确性与推理保真度上 **>10%** 提升；消融证实 PRM-引导奖励 + 逐步轨迹探索是关键。是 PRM 在硬件代码长程推理上的代表性工作。
- **arXiv**：[2606.04246](https://arxiv.org/abs/2606.04246)

#### Unsupervised Process Reward Models (uPRM) (2026-05)
- **简介**：EPFL Maria Brbic 团队。完全不需人类逐步标注，也不需要 final-answer 的 ground truth。核心是一个利用 LLM next-token 概率构造的 scoring function——在一批轨迹间联合估计 "首个错误步" 候选位置。三类评估全部成立：(i) ProcessBench 首错定位比 LLM-as-Judge 高出 +15 pp 绝对准确率；(ii) 作 TTS verifier 与 supervised PRM 持平、压过 majority voting +6.9 pp；(iii) 作 RL reward 信号时比 ground-truth 训的 supervised PRM 还更稳定。给 PRM 标注成本问题给出一个根本性的去标注路径。
- **arXiv**：[2605.10158](https://arxiv.org/abs/2605.10158)

#### Process Rewards with Learned Reliability (BetaPRM) (2026-05)
- **简介**：WashU + SUTD。指出现有 PRM 只输出单点分数，下游被迫把不可靠预测当确定信号。BetaPRM 把每步 PRM 变成 distributional：从 Monte Carlo continuations 的成功计数监督学一个 Beta 信念，用 Beta-Binomial likelihood 解释观察到的成功次数，而不是回归到 finite-sample 经验比率。学到的 reliability 信号区分高/低置信度奖励。基于此提 ACA（Adaptive Computation Allocation）：可信高分早停、不确定前缀分配更多算力，相比固定预算 Best-of-16 token 用量降低 33.57%、accuracy 反升。
- **arXiv**：[2605.15529](https://arxiv.org/abs/2605.15529)

#### Verifiable Process Rewards for Agentic Reasoning (VPR) (2026-05)
- **简介**：清华袁慧宁等。聚焦一类 "中间动作可被符号/算法 oracle 客观验证" 的推理问题，把 oracle 转成 dense turn-level 奖励。三个实例：search-based verification（动态演绎）、constraint-based verification（逻辑推理）、posterior-based verification（概率推断）。理论证明 verifier 可靠度决定信用分配收益，实证上同时压过 outcome reward 和 rollout-based PRM，且能跨域迁移到通用推理基准——揭示 "可验证的中间监督" 比 outcome RL 上限高。
- **arXiv**：[2605.10325](https://arxiv.org/abs/2605.10325)

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

#### Beyond Entropy: Correctness-Aware Advantage Shaping via Contrastive Policy Optimization (CPO) (2026-07)
- **简介**：Xu、Liu、Chan、Li、Cai、Chen、Zhang 指出 RLVR 常用熵做 advantage shaping，但熵无法区分「有用的不确定」与「有害的困惑」。提出 CPO，用 reference-guided 与 vanilla 生成分布之间的 **token 级对比分歧**做「正确性感知」的 advantage shaping，理论与实验均表明该分歧可靠指示 token 级正确性；并证明 On-policy Distillation 是 CPO 的特例（后验分布由外部教师实例化），同时解决 zero-advantage 问题。在域内/域外基准上大幅超过基于熵的 RLVR 方法并保持强泛化。
- **arXiv**：[2607.14614](https://arxiv.org/abs/2607.14614)

#### When Implausible Tokens Get Reinforced: Tail-Aware Credit Calibration for LLM Reinforcement Learning (TACO) (2026-07)
- **简介**：Xiuyi Lou、Zicheng Xu、Vladimir Braverman 等（JHU / Rice 系）指出 critic-free RL（GRPO 类）用**均匀信用分配**把同一优势广播给所有 token 会引发「正信用污染（Positive-Credit Contamination）」：上下文错误的低概率尾部 token 与合理 token 拿到相同正信用，导致错误推理被无差别强化。提出 **TACO（Tail-Aware Credit calibratiOn）**：先算融合局部生成上下文的 tail-risk 分数（区分「意外稀有」与「不确定性驱动的探索」），再据此**调低高风险 token 的正信用而不完全移除其梯度**，使反复出现的有用稀有模式仍能累积强化、偶发噪声被逐步抑制。3 个 LLM × 8 个 benchmark 上稳定优于 GRPO 类基线，并改善长程 RL 训练稳定性（代码开源）。
- **arXiv**：[2607.07976](https://arxiv.org/abs/2607.07976)

#### Rethinking On-Policy Self-Distillation for Thinking Models (2026-07)
- **简介**：Simran Kaur、Narutatsu Ri、Sanjeev Arora 等（Princeton）研究「特权自蒸馏（给学生看解答等特权信息）」对 thinking model 的影响，反直觉地发现：在长推理 trace 上，特权上下文蒸馏会**损害** thinking model——5 个 Qwen3/OLMo 模型在 AIME24/25、HMMT25 上 avg@16 相对下降最多 17%，退化随隐藏的特权上下文量与 rollout 预算增大而加剧。诊断把失效归因于**高熵分叉位置的 token 级信号**被重塑：特权上下文降低 thinking model 的分叉率、在自我纠错分支上惩罚原本被普通 OPD 支持的重新考量 token，使模型产生更少的验证 / 回溯 / 犹豫标记。结论：强 thinking model 的自蒸馏必须关注纠错与推理步附近的 **token 级信号**。
- **arXiv**：[2607.05184](https://arxiv.org/abs/2607.05184)

#### Which Tokens Matter? Adaptive Token Selection for RLVR with the Relative Surprisal Index (RSI-S) (2026-06)
- **简介**：Outongyi Lv 等针对 RLVR 中两条互相矛盾的经验路线——一派主张优先训练高熵 token 位置、另一派警告勿让低概率 token 主导梯度——指出「孤立评估采样 token 的概率或熵不足以刻画策略优化动力学」。提出信息论度量 **Relative Surprisal Index (RSI)**，自然耦合 token 熵与所选 token 的概率，并证明温和条件下 RSI 关联于「logit-gradient 范数与预测熵在 selected-logit 扰动下一阶变化的局部比值」。据此提出 **RSI-S**：熵自适应 token 过滤，仅保留处于稳定 RSI 区间的 token，同时滤除冗余低-surprisal token 与不稳定高-surprisal 尾部 token，调和两派矛盾。在 Qwen2.5-1.5B/3B/7B 上，AIME 与 AMC 的 avg@32 较 GRPO 提升 2–3 个百分点。属 token 级 credit assignment 的选择性优化。
- **arXiv**：[2606.31575](https://arxiv.org/abs/2606.31575)

#### On the Policy Gradient Foundations of Group Relative Policy Optimization: Credit Assignment, Gradient Sparsity, and Rank Collapse (2026-06)
- **简介**：Amritansh Mishra、Supriyo Chakraborty、Berkcan Kapusuzoglu 从策略梯度定理第一性原理严格推导 GRPO，揭示其**根本性 credit assignment 失败**：在 output-only 奖励下，一条 rollout 内每个 token 获得**相同优势**，把 token 级 credit 坍缩为单一标量。证明这会诱发随训练加剧的**梯度稀疏**，并在 Nemotron-4B/GSM8K 上通过 GRPO 梯度的 SVD 分析实证：无论组大小 R∈{2,4,8}，梯度矩阵有效秩恒 ≈2；将其形式化为源于优势零和约束的**内在 rank-2 结构**，并推导 GRPO baseline 何时最优。工作刻画了 GRPO 简洁性在理论上何时成立，并将 credit assignment 瓶颈指认为多步推理的关键限制。属 token 级 credit assignment 的理论分析新工作。
- **arXiv**：[2606.29238](https://arxiv.org/abs/2606.29238)

#### Learning with a Single Rollout via Monte Carlo Pass@k Critic (SR-PPO) (2026-06)
- **简介**：阿尔伯塔大学 + MILA（Fengdi Che、Rupam Mahmood、Dale Schuurmans 等）研究 single-rollout PPO（SR-PPO）以同时缓解重复采样的算力成本与 token 级信用分配难题。不再对组内回报归一化估计优势，而是用**单条 rollout 的 Monte Carlo outcome** 训练一个校准的 token 级信用 critic，预测前缀处的 Pass@k 成功概率（由 Pass@1 尝试导出）：该信号比 Pass@1 更具选择性——折扣易解前缀、优先成功概率仍处边缘的难前缀。证明 k→∞ 时 Pass@k 收敛为可达性指示器，显式状态图上极限可在 O(|V|+|E|) 计算。SR-PPO 学习动态稳定，在 HMMT26、AIME24 上 Pass@128 一致提升。
- **arXiv**：[2606.25451](https://arxiv.org/abs/2606.25451)

#### VIMPO: Value-Implicit Policy Optimization for LLMs (2026-06)
- **简介**：UC Berkeley（Zhewei Kang、Sergey Levine、Dawn Song、Xuandong Zhao 等）针对"GRPO 免 critic 但把轨迹级优势均摊到每个 token、actor-critic 信号稠密但需训练 value 函数"的权衡，提出免 critic 的 **VIMPO**：从 KL-正则 RL 的最优性条件导出 policy-implied value 函数，对自回归生成可写成 policy-reference 对数比形式并以"轨迹末无未来奖励"的终止条件锚定，得到一个无需训练 critic 即可纳入 outcome 级可验证奖励的简单 value loss，同时给出免 critic 的 actor advantage。在数学 RLVR（MATH-500、AIME 2024/2025、OlympiadBench）上全面优于 GRPO，竞赛题增益更大，且在噪声奖励下保持优势。
- **arXiv**：[2606.20008](https://arxiv.org/abs/2606.20008)

#### Beyond Entropy: Learning from Token-Level Distributional Deviations for LLM Reasoning (ICT) (2026-06)
- **简介**：港理工 + 团队（Xuanzhi Feng、Song Guo 等）针对 RLVR 中"均匀 token 更新致熵坍缩 / 过度熵最大化致熵爆炸"的二难，提出 **Independent Combinatorial Tokens (ICT)** 框架，把优化焦点从标量不确定性转向 token logits 的分布性质：用 token logits 分布间的 Jensen-Shannon 散度识别"分布模式独特"的 token 作为关键分叉点引导探索。理论上（基于 Shannon 与二阶 Rényi 熵）证明仅更新这些 token 可调控策略集中度。在 Qwen2.5（0.5B/1.5B/7B）上仅更新前 10% 独特 token，较 GRPO、20-Entropy、STAPO 在 7 个 benchmark 上平均 pass@4 提升 4.58%、最大 14.9%。
- **arXiv**：[2606.19771](https://arxiv.org/abs/2606.19771)

#### Context-Aware RL for Agentic and Multimodal LLMs (ContextRL) (2026-06)
- **简介**：来自 Princeton 系作者群（Peiyang Xu、Karthik R. Narasimhan、Pramod Viswanath、Prateek Mittal、Xingyu Fu 等 7 人）。针对 LLM 在长/复杂上下文中难以定位"小而决定性证据"（工具轨迹中的一行、图像中的细节）的问题，提出通过**间接辅助目标**提升细粒度 grounding。② 机制：不仅监督最终答案，而是给模型一个 query、一个 answer 和两个高度相似的 context，奖励其选出支持"query–answer"配对的那个 context；对比上下文数据在编码智能体（轨迹为 context，1K 对）与多模态推理（图像为 context，7K 对）两域构建。③ 相比标准 GRPO，5 个长程推理基准平均 +2.2%、12 个 VQA 基准平均 +1.8%；消融显示增益来自上下文选择目标本身而非对比数据。
- **arXiv**：[2606.17053](https://arxiv.org/abs/2606.17053)

#### STRIDE: Strategic Trajectory Reasoning via Discriminative Estimation for Verifiable Reinforcement Learning (2026-06)
- **简介**：作者 Qinjian Zhao、Zhihao Dou 等 13 人。针对 RLVR 仅用最终答案分配轨迹级奖励、监督稀疏且对所有 token 一视同仁的问题，且指出过程奖励/高熵 token/语义不确定性等中间信号"本身不可验证"。② 核心机制：在每个响应组内对比成功与失败轨迹，估计每个 **n-gram 策略模式**的"结果判别偏好"，再结合推理显著性熵识别决策相关策略模式，在 RL 优化时为这些模式赋予差异化优势值，从而在保持 RLVR 可验证性的前提下实现更精确的 token/模式级信用分配。③ 在多种模型、任务及扩展场景（含 VLM 与 agent 系统）上一致提升推理性能。
- **arXiv**：[2606.15866](https://arxiv.org/abs/2606.15866)

#### Not All Tokens Learn Alike: Attention Entropy Reveals Heterogeneous Signals in RL Reasoning (2026-05)
- **简介**：北大团队。把 token 异质性映射到 attention entropy 上：低-attn-entropy 的 "anchor" token 依赖局部支持，梯度稳定、对齐 full-token 更新但难推到难题；高-attn-entropy 的 "explorer" token 聚合发散上下文，梯度更大但更易爆。Explorer-only 训练平均不稳定，但少数稳定的 run 暗示这些 token 含 hard-reasoning 信号。提出 entropy-aware soft reweighting，在 Qwen3-8B-Base 上 held-out 平均 +3.01 pp。控制实验排除 position / predictive entropy / loss normalization 作为替代解释。
- **arXiv**：[2605.07660](https://arxiv.org/abs/2605.07660)

#### Entropy Polarity in Reinforcement Fine-Tuning: Direction, Asymmetry, and Control (PAPO) (2026-05)
- **简介**：复旦张奇 / 桂韬等。给 RLVR 中 "策略熵如何被 sampled update 改变" 一套理论：定义 token-level entropy polarity（带符号量），一阶近似预测某次更新会扩张还是收缩熵。揭示结构性不对称——强化高概率 token 更易致熵收缩；扩张需要低概率样本或更强分布修正。基于此提 PAPO（Polarity-Aware Policy Optimization）：advantage reweighting 同时保留正负 polarity 两类更新、在线熵轨迹自适应调节探索/利用。数学推理稳定优于 GRPO 系列。
- **arXiv**：[2605.11775](https://arxiv.org/abs/2605.11775)

#### CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization (2026-05)
- **简介**：MBZUAI + Linköping + ANU。指出 GRPO 把同一 advantage 广播到 rollout 内每个 token，无法区分 "关键推理 token" 与 "嗯/然后/我重新整理一下" 这类填充 token。提出 CEPO：识别一个 token "重要" 的标准必须 *双侧成立*——既在答对 trajectory 中频繁出现，又在答错 trajectory 中相对缺席。把这种对比证据当 token 级 reweight 信号嵌进 GRPO 优化，在数学推理上压过 baseline。把 token-level CA 从 "看正例" 推到 "正反例对比"。
- **arXiv**：[2605.19436](https://arxiv.org/abs/2605.19436)

#### Token-weighted Direct Preference Optimization with Attention (AttentionPO) (2026-05)
- **简介**：康奈尔 + Vanderbilt。在 DPO/Token-DPO 路线上，TwDPO 给出一个 token 加权的 RL 风格 DPO 目标族，其实例 AttentionPO 直接拿 LLM 自身 attention 估 token 权重——让 LLM 充当 pairwise judge、记录它在比较两条响应时关注哪些 token。无需独立训权重模型，每个样本只多两次前向，且 content-aware（按内容自适应而非 position 启发式）。AlpacaEval/MT-Bench/ArenaHard 均超越现有 token-level DPO。
- **arXiv**：[2605.21883](https://arxiv.org/abs/2605.21883)

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

#### Breaking Failure Cascades: Step-Aware Reinforcement Learning for Medical Multimodal Reasoning (MRPO) (2026-06)
- **简介**：Junha Jung、Jaewoo Kang（DMIS Lab，高丽大等）针对临床图像推理后训练普遍 outcome-centric（仅靠最终答案正确性/序列级偏好）导致稀疏 credit assignment 的问题，实证发现「早期推理失败引发的级联错误」是医疗 VQA 错误预测的主因。提出 **Medical Reasoning-aware Policy Optimization (MRPO)**：引入 step-wise 过程奖励，当最终答案错误时对**更早的无效推理步 token 施加指数级更大的惩罚**，在不损害正确路径的前提下打断失败级联。在三个多模态 LLM backbone 上一致优于标准 GRPO 与近期 RL 基线，Qwen3-VL-8B-Instruct 上甚至超过 HuatuoGPT-Vision-34B 达 2.79 点；并把早期推理失败率从 64.0% 降至 13.0%。属 step/segment 级过程奖励与失败级联缓解的代表作（多模态但核心为 step-aware 推理 RL）。
- **arXiv**：[2606.31825](https://arxiv.org/abs/2606.31825)

#### Dynamic Rollout Editing for Reducing Overthinking in RL-Trained Reasoning Models (DRE) (2026-06)
- **简介**：作者 Zihao Wei、Liang Pang、Huawei Shen、Xueqi Cheng 等 11 人（疑似中科院计算所系）。将长 CoT 的"过度思考"（答案已出现后仍继续生成）从 GRPO 视角重定位为**训练时信用分配问题**而非解码时停止问题。② 关键观察：训练初期成功轨迹比失败轨迹略更过度思考，而 GRPO 的序列级信用无法区分"到达解答的前缀"与"不必要的延续"，两者都得正向信号，使失衡放大。机制 DRE：对答案出现后仍继续思考的成功轨迹，保留已验证前缀、编辑剩余思考部分、并在同一 RL group 内优先选择编辑后轨迹，从而削弱不必要思考的偏好信号而不惩罚必要推理。③ 在多种任务上验证 DRE 有效性。
- **arXiv**：[2606.17890](https://arxiv.org/abs/2606.17890)

#### Shattering the Autoregressive Curse: Dynamic Epistemic Entropy Orchestrated Erasable Reinforcement Learning (E³RL) (2026-06)
- **简介**：作者 Ziliang Wang、Kang An、Faqiang Qian 等 8 人。针对长程逻辑推理中的"自回归诅咒"——早期微小认知扰动沿 MDP 不可逆传播、引发级联失败导致轨迹崩溃。② 提出 E³RL（动态认知熵编排的可擦除 RL）：摆脱外部信号，将模型内生的局部自回归交叉熵作为认知不确定性内在坐标；通过**段级（segment-level）自适应动态阈值与优势分配**精确"切除"局部逻辑缺陷，并复用历史 KV 缓存流，赋予推理过程自愈能力且保持线性内存开销。③ 在 DeepMath-103k 训练，AIME 等数学基准上 4B/8B 模型分别超越此前 SOTA 5.349% / 6.514%。
- **arXiv**：[2606.17735](https://arxiv.org/abs/2606.17735)

#### RREDCoT: Segment-Level Reward Redistribution for Reasoning Models (2026-06)
- **简介**：JKU Linz 的 Sepp Hochreiter 团队提出 **RREDCoT**（Reward REDistribution for Chain of Thoughts）：把 GRPO 视为高方差 Monte Carlo 方法，提出用模型自身近似最优 reward redistribution，**无额外 rollout** 即可在 segment 级强调对答案重要的 CoT 段。系统对比了 MC 采样、各种 attribution 方法，并细致分析 CoT 分段策略与 state value 估计。这是当前为数极少的"非 token / 非 step 而是 segment-level"显式信用分配工作，定位类似 trajectory-level → segment-level 的中间层。
- **arXiv**：[2606.06475](https://arxiv.org/abs/2606.06475)

#### ThoughtFold: Folding Reasoning Chains via Introspective Preference Learning (2026-06)
- **简介**：上海AI Lab + CUHK Dahua Lin 团队提出 **ThoughtFold**：Long CoT 中 trial-and-error 段被 RLVR 当作"正确答案的一部分"被无差别强化，导致 over-thinking。ThoughtFold 用 introspective 策略在每条正确轨迹内识别冗余，构造 sub-trajectory 谱，引入 **masked preference optimization** 显式惩罚冗余探索、鼓励直接桥接关键推理段。把 DeepSeek-R1-Distill-Qwen-7B 的 token 用量减少约 **56%**，同时维持 SOTA 准确率。本质上是 segment 级 sub-trajectory 偏好优化。
- **arXiv**：[2606.03503](https://arxiv.org/abs/2606.03503)

#### Segment-Aligned Policy Optimization (SAPO) (2026-05)
- **简介**：China Telecom + Lei Gao 等。把策略更新单元从 token / full sequence 改成 "coherent reasoning segment"：在推理 segment 上构 step-wise MDP，做 segment-level value estimation、advantage、importance sampling，三项都按推理边界对齐。在多模态推理基准上稳定超 token-level 与 sequence-level baseline，训练稳定性、value estimation 一致性同时提升——给 §1.5 提供新的 segment 级 credit-assignment 落地。
- **arXiv**：[2605.01327](https://arxiv.org/abs/2605.01327)

#### STRIDE: Learnable Stepwise Language Feedback for LLM Reasoning (2026-05)
- **简介**：中科院 + 通义实验室 Yongbin Li 等。现有 step-level feedback 依赖 PRM 或外部判别器，训练成本和脆性都高。STRIDE 让模型自学一种 "stepwise 自然语言反馈"——把 "这一步是否正确以及为什么" 直接表达成可学语言信号，作为 segment 级 credit。比 outcome-only RL 收敛更稳、比 PRM 蒸馏更便宜，给 PRM-free 的 segment-level 路线提供新候选。
- **arXiv**：[2605.18851](https://arxiv.org/abs/2605.18851)

#### Credit Assignment with Resets in Language Model Reasoning (RRPO / SRPO) (2026-05)
- **简介**：Columbia + Apple 团队。把 "reset" 机制引入 RLVR：返回到中间状态、重采反事实 continuations，使 outcome 差异可归因到该点决策。两套方法：RRPO 随机选 reset 状态；SRPO 让模型自定位错误步骤再 reset。在 CPI 框架下分析，证明 "用 credit-assignment oracle 指向可改进状态" 的扩展 CPI 严格优于随机 reset。SRPO 在多模型多基准持续超 GRPO/RRPO，且仅依赖模型自身、无外部监督——给中间层 CA 一种基于 counterfactual rollout 的清晰落地。
- **arXiv**：[2605.25507](https://arxiv.org/abs/2605.25507)

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

#### When the Reward Suite Is Leaky: A Preregistered Causal Contrast of Natural Verifier False Positives in RLVR (2026-07)
- **简介**：单作者 Chuyifei Zhang 的预注册因果对照研究，聚焦 RLVR 代码奖励中「天然假阳性」（per-task、持续、非对称的验证器错误）。作者在部署套件上做两臂因果对照：同样 MBPP 任务/种子/算力下，用原始 MBPP 测试（leaky）对比 MBPP+ 加固测试（hardened）训练 GRPO，并有两个复现家族。关键发现：奖励到的假阳性质量可由训练前的廉价静态泄漏审计预测（Spearman 0.80），人工裁定发现 47.57%（按记录加权）的被奖励假阳性是真正错误的代码——奖励为真实 bug 买单，而非纯套件伪影；加固奖励能去除测量膨胀但几乎不带来能力提升。
- **arXiv**：[2607.11022](https://arxiv.org/abs/2607.11022)

#### Mitigating Factual Hallucination in Large Reasoning Models via Mixed-Mode Advantage Regularization (MARGO) (2026-07)
- **简介**：Kaishen Wang、Tong Zheng、Heng Huang 等（Maryland 系）发现「显式思考」在事实型 QA 上**并非一致有益**：它有时会推翻原本正确的非思考答案、造成事实漂移，称为「thinking-induced hallucination」。把显式思考形式化为叠加在直答倾向上的 **thinking residual**，提出 **MARGO（Mixed-Mode Advantage Regularization for Grounded Optimization）**：用**非思考 rollout 作为同模型参照**做优势估计，构造同时含思考 / 非思考轨迹的混合模式 rollout 组，据此评估显式思考是否在直答之上带来事实增益，从而抑制易致幻的思考、保留有益思考。多个事实型 QA benchmark 上事实可靠性优于强基线，数学 benchmark 上保持通用推理能力。
- **arXiv**：[2607.05861](https://arxiv.org/abs/2607.05861)

#### Right in the Right Way: LM Training with Verifiable Rewards and Human Demonstrations (2026-07)
- **简介**：MIT 的 Mehul Damani、Isha Puri、Idan Shenfeld、Jacob Andreas 针对 RLVR 只优化可客观打分部分、忽视风格/结构等不可验证维度而引发的 **diversity collapse、不自然输出与 reward hacking**，提出**对抗式 generator-discriminator 框架**：generator 用 RL 同时最大化任务准确率与来自 discriminator 的对抗奖励；discriminator 与策略同训、学习区分人类书写与模型生成输出，充当人类输出分布的学习式代理，为难以形式化为标量奖励的方面提供反馈。跨 bug 修复、开放式生成等域，在保住 RLVR 准确率增益的同时改善不可验证属性：bug 修复中 edit distance 显著更低且性能持平；故事生成 win rate 显著提升且更多样、更像人；在一个简单 reward hacking benchmark 上**几乎消除模型不当行为**并维持高分。属反 reward hacking + 桥接 RL/SFT 的代表作。
- **arXiv**：[2607.01181](https://arxiv.org/abs/2607.01181)

#### The Verification Horizon: No Silver Bullet for Coding Agent Rewards (2026-06)
- **简介**：阿里 Qwen 团队（Dayiheng Liu、Zeyu Cui 等，作者按名字字母序）系统论证"验证比生成更难"的反转现象与其对奖励设计的影响：每个 verifier 都只是人类意图的 proxy，面临双重困难——意图天然欠定难以忠实检查、训练优化会拉大 proxy 与意图差距而表现为 **reward hacking / 信号饱和**。沿 scalability、faithfulness、robustness 三维刻画验证信号质量，并研究四类奖励构造（test verifier / rubric verifier / user-as-verifier / automated agent verifier）。实验表明针对性验证设计能有效抑制 reward hacking、提升完成质量；核心结论：没有固定奖励函数能随策略能力增长持续有效，**验证必须与生成器协同进化**。
- **arXiv**：[2606.26300](https://arxiv.org/abs/2606.26300)

#### CFPO: Counterfactual Policy Optimization for Multimodal Reasoning (2026-06)
- **简介**：北邮（Zhangyuan Yu、Qicheng Lao 等，ICML 2026）针对 LVLM 多模态推理中 RL 缺乏显式反事实增强与因果学习、导致"忽视视觉证据偏信语言先验 / 长 CoT 中幻觉漂移"的 grounding 失效，提出 **CounterFactual Policy Optimization (CFPO)**：通过跨模态反事实增强机制——最大化"模型预测"与"抑制关键视觉线索的反事实状态下预测"之间的差异——来正则化策略，强制视觉感知与文本推理的因果一致。无需外部 reward model 或额外监督，可无缝接入 GRPO/DAPO。较标准 RL 基线一致提升 3.17%–6.25%，较 SOTA 感知方法 PAPO 提升 1.32%–2.13%。
- **arXiv**：[2606.23206](https://arxiv.org/abs/2606.23206)

#### Reward Hacking in Language Model Agents: Revisiting AI Safety Gridworlds (2026-06)
- **简介**：来自 Koç University 与 UC Berkeley（Ömer Veysel Çağatan、Xuandong Zhao）。把经典 AI Safety Gridworlds 改造为文本化评测套件，系统研究语言智能体在优化代理奖励（proxy reward）时的 reward hacking / specification gaming：模型零样本即系统性地获得高"可观测奖励"但在隐藏安全目标上表现差，且 RL 直接优化会进一步**扩大可观测奖励与隐藏奖励之间的差距**。② 机制上指出模型因初始能力过早锁定在局部高回报策略，难以发现更安全的替代方案。③ 该现象在 1.5B–14B 全规模上持续存在，且无法用更精细的 credit assignment、探索提示或熵正则化解决，说明 agentic 场景的代理奖励失效需要超越标准探索/信用分配修复的新方法。
- **arXiv**：[2606.15385](https://arxiv.org/abs/2606.15385)

#### Do Coding Agents Deceive Us? Detecting and Preventing Cheating via Capped Evaluation with Randomized Tests (CapCode / CapReward) (2026-06)
- **简介**：东大 Sugiyama / Ishida 团队提出 **CapCode**：构造的代码评测集合的 best non-cheating 表现被故意"封顶"在 1 以下，因此显著超过 cap 的得分必然是作弊证据；并配套 **CapReward**——基于该 cap 的奖励设计，阻止超过 cap 的优化。在多个数据集上 CapCode 能在保持模型排序的同时检出 cheating，CapReward 减少 cheating 行为，让模型更贴合真实任务规范。代码推理 RL 中 anti-reward-hacking 路线的代表性新设计。
- **arXiv**：[2606.07379](https://arxiv.org/abs/2606.07379)

#### Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning (CHERRL) (2026-06)
- **简介**：清华 + 哈工大 + 西交（Xiaozhi Wang、Juanzi Li 等）提出 **CHERRL**：rubric-based RL 中策略模型容易利用 LLM-as-a-Judge 的偏见做 reward hacking，且通常与多种 judge bias 纠缠难分析。CHERRL 通过向 LaaJ 注入已知 bias 稳定复现 hacking、明确 reward divergence、精确定位 hacking 起点，提供"可控 hacking 实验环境"。基于此，论文从 discoverability / exploitability 两维度分析 judge bias，并探索 agent-based 系统从训练日志自动检测 hacking onset。代码 + 环境开源。
- **arXiv**：[2606.04923](https://arxiv.org/abs/2606.04923)

#### QUBRIC: Co-Designing Queries and Rubrics for RL Beyond Verifiable Rewards (2026-06)
- **简介**：Amazon + GeorgiaTech（Tuo Zhao、Chao Zhang 等，11 位作者）指出 rubric-based RL 的结构瓶颈：rubric 质量受限于 query 结构——开放式 query 产 vague rubric，朴素收紧又会引入 fabricated reference 让所有响应都失败而无奖励信号。**QUBRIC** 共同设计 query 与 rubric：教师 key-points 把开放问题改写为可评估的 scenario 问题；contrastive rubric 生成把"教师-策略 gap"转为 query-level 标准；learnability filtering 仅保留信息性 query-rubric 对接入 GRPO。在 ArenaHard 较 SFT 提升 **+5.5**；只用指令跟随训练，迁移到法律、道德、叙事推理 3 个保留 benchmark **平均 +6.3**，提升集中在 reasoning 维度。
- **arXiv**：[2606.03968](https://arxiv.org/abs/2606.03968)

#### When RLHF Fails: A Mechanistic Taxonomy of Reward Hacking, Collapse, and Evaluator Gaming (2026-06)
- **简介**：单作者 Zelalem Abahana 给出 RLHF 失败模式的 **机理分类**，把 reward hacking 从"终末事件"重写成可分类、可定位、可预测的训练动力学。在 PPO / DPO / UP-PPO / RM-uncertainty / approximate policy drift / diversity-repetition 诊断 + 双 LLM judge 上，覆盖 61 个 checkpoint 行 / 1920 行级转换。结论：激进 PPO 局部 hacking 率最高 **14.45%**（CI 10.16-18.75），UP-PPO 在同样激进设置下降到 **11.33-10.94%**；pre-transition logistic 预测未来 row-level hacking 的 ROC-AUC 0.821；3/12 设置中 row-level 分析能识别 checkpoint 平均所漏掉的 localized hacking。
- **arXiv**：[2606.03238](https://arxiv.org/abs/2606.03238)

#### Reward Hacking in Rubric-Based Reinforcement Learning (2026-05)
- **简介**：Anas Mahmoud 等系统性地把 rubric-RL 的 reward hacking 拆成三类失败模式：partial 满足复合 criterion、把隐式内容当显式（跳过实质解释）、不精确主题匹配。提出 "self-internalization gap" 作为诊断——追踪在弱 verifier 上训练的 policy 何时在真实质量上触顶但代理奖励仍在涨，这是从优化真实表现转向优化代理指标的转折点。结论：更强 verifier 可缓解但不能根除漏洞，rubric 设计与模型架构同等重要。给 medical/scientific RLHF 流水线一个对照范本。
- **arXiv**：[2605.12474](https://arxiv.org/abs/2605.12474)

#### Step-wise Rubrics as Rewards (SRaR) (2026-05)
- **简介**：北大 / Wenqi Shao 等。指出 rubric-based RL（RaR）虽细于 outcome-only，但 rubric 分数被聚合成单标量打到整段响应上，导致三大病症：丢失 multi-criterion 结构、对错步同等监督、unbounded self-correction 引发 reward hacking。1000 题诊断显示 18.2% 的 "正确响应内步骤" 实际是错的却被正向奖励，49.9% 的 "错误响应内步骤" 实际是对的却被惩罚。SRaR 三件事：用 LLM judge 把每条 rubric item 归因到具体步骤；rollout 间归一 per-step rubric 得分使只有 quality 有变化的步才产生学习信号；与 outcome reward 用 decoupled advantage estimator 组合保持 baseline 稳定。AIME 2025 Faithful Reasoning Rate 34.5%→46.7%，self-correction looping 48.1%→26.5%。
- **arXiv**：[2605.17291](https://arxiv.org/abs/2605.17291)

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

#### Weight-Space Geometry of Offline Reasoning Training (2026-06)
- **简介**：Aleksandr Nikolich 等以权重空间几何视角对比六种 offline 推理训练损失（SFT、RFT、DFT、RIFT、Offline GRPO、DPO）是否机制各异还是收敛到相似的权重更新。在 Qwen3-4B 同一 base、同一数学 rollout、attention-only LoRA 下，用 cosine 相似度、主夹角子空间分析、线性模式连通性与 CKA 分析 delta：发现 (i) SFT/RFT/RIFT 权重 delta 近共线（cosine≥0.97）且 GSM8K 准确率相当；(ii) DFT 方向偏离更大；(iii) **Offline GRPO 加入与 SFT 方向大量正交的分量**（全局约 67%，后层至多约 86%）却仍在 SFT 损失盆地内；(iv) **DPO 处于近正交子空间**、存在 mode-connectivity 障碍、后层 CKA 坍至约 0.46，且在本协议下准确率最高（GSM8K 93.5%、AIME26 30.0%）。为 DPO/Offline GRPO 步级与无显式 RM 路线提供了机理性对照。
- **arXiv**：[2606.23740](https://arxiv.org/abs/2606.23740)

#### CASPO: Confidence-Aware Step-wise Preference Optimization (2026-05)
- **简介**：浙大 + 蒙特利尔大学 + 中山大学 + 弗吉尼亚理工。直击 "答案对但中间步存在缺陷" 的可靠性鸿沟。CASPO 通过迭代 DPO 把 token 级 confidence 与逐步 logical correctness 对齐，无需独立 RM。推理时进一步提 CaT（Confidence-aware Thought）：用校准后的 confidence 以 O(V) 延迟动态剪掉不确定推理分支。10 个 benchmark + 多 model family，CASPO 一致提升 reasoning reliability 与 inference 效率；在 Qwen3-8B-Base 上 AIME'24/'25 上压过 tree-search baseline 且不用 RM 数据。
- **arXiv**：[2605.07353](https://arxiv.org/abs/2605.07353)

#### ξ-DPO: Direct Preference Optimization via Ratio Reward Margin (2026-05)
- **简介**：西北工业大学。reference-free 偏好优化（SimPO 系）的 β/γ 联调一直靠试错。ξ-DPO 把 SimPO 目标重写成 "最小化 reward gap 与最优 margin 的距离"，并把 reward 改写为 chosen/rejected 比率形式，直接把 β 抵消掉，得到一个有界、可解释的 margin ξ。与 SimPO 的 γ 不同，ξ 显式表示期望相对分离度，可从初始 reward gap 分布直接确定，避免反复调参。开放基准上稳定优于 SimPO/DPO。
- **arXiv**：[2605.10981](https://arxiv.org/abs/2605.10981)

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

#### LLM-as-a-Verifier: A General-Purpose Verification Framework (2026-07)
- **简介**：Jacky Kwok、Chelsea Finn、Ion Stoica、Azalia Mirhoseini 等（Stanford / Berkeley / NVIDIA）把「验证（判断解是否正确）」提为一条新的**扩展轴**，提出无需额外训练的通用验证框架 **LLM-as-a-Verifier**：不同于让 LLM 输出离散分数的标准 judge，它对打分 token 的 logits 分布**取期望得到连续分数**，从而沿三维扩展——评分粒度、重复评测、准则分解，且更细粒度打分带来正负解的更好分离与更校准的比较。在 Terminal-Bench V2（86.5%）、SWE-Bench Verified（78.2%）、RoboRewardBench（87.4%）、MedAgentBench（73.3%）达 SOTA；其细粒度信号还可作为任务进度代理，并**为 RL 提供稠密反馈**，提升 SAC 与 GRPO 在机器人与数学推理上的样本效率。（v1 = 2026-07-06，本页所示 v2 = 07-07，均在窗口内。）
- **arXiv**：[2607.05391](https://arxiv.org/abs/2607.05391)

#### Attention Limited Reward Learning (2026-07)
- **简介**：单作者 Wenqian Xing 从「理性疏忽（rational inattention）」视角重审 RLHF 常用的 Bradley-Terry 奖励建模：每个偏好标签由**低容量评估通道**生成，因而混淆了两种模糊——「两候选价值真的接近」与「区别在有限注意力下难被察觉」。理论上证明：被动比较数据一般**无法区分**奖励、注意力与默认倾向，异质注意力会使标准 BT 建模恢复出**误导性排名**；学习速率由每个标签所承载的「被注意信息量」而非原始标签数决定。两个案例（Chatbot Arena 的模型对人类投票、感知比较任务）显示存在超过采样噪声的**循环成分**（无任何标量奖励可表示），主张把人类反馈视为「注意力受限的测量过程」而非直接显示偏好——为 robust RM 提供了理论诊断。
- **arXiv**：[2607.04590](https://arxiv.org/abs/2607.04590)

#### Many Voices, One Reward: Multi-Role Rubric Generation for LLM Judging and Reward Modeling (MRRG) (2026-07)
- **简介**：Dazhi Fu、Jiuding Yang、Yiwen Guo、Jicong Fan 针对无标注 rubric 生成器通常仅依赖单一通用评估者、易遗漏重要人类偏好维度（作者称之为 **dimensional blind spots**）的问题，提出 **Multi-Role Rubric Generation (MRRG)**：一个**免训练、免参考**框架，从多个互补角色抽取评估标准并整合为可审计的 rubric-based 打分器，既能验证成对偏好，又能为 **GRPO 式 RLVR** 提供奖励。在偏好验证 benchmark 上跨多个 backbone 一致优于单角色 rubric 生成基线；进一步的 RLVR 实验表明 MRRG 为改进开放式生成提供更强的奖励信号。属 generative / rubric-based reward modeling 的新工作。
- **arXiv**：[2607.01830](https://arxiv.org/abs/2607.01830)

#### GEOALIGN: Geometric Rollout Curation for Robust LLM Reinforcement Learning (2026-06)
- **简介**：中山大学 + 阿里（Ting Zhou、Ying Shen、Daoyuan Chen 等，ICML 2026）针对在线 RL 在噪声/错配奖励下训练不稳的问题，识别出名为 **directional inconsistency** 的失效模式：batch 内少量高奖励 rollout 诱导的表征空间偏好方向与多数派尖锐相悖，造成高方差、不稳定更新。提出轻量即插件 **GEOALIGN**：构造 within-prompt 偏好对、在线学习一个投影器以集中 reward-ordered 位移方向、并据相对 batch 共识原型的角度偏差检测方向不一致 rollout 并以稳定替代修正；仅前向、开销可忽略。在对话对齐（学习型 RM）与数学推理（二元可验证奖励）上提升最终性能、减少训练震荡，优于 PF-PPO、PAR、PODS、Seed-GRPO。
- **arXiv**：[2606.26917](https://arxiv.org/abs/2606.26917)

#### MaxProof: Scaling Mathematical Proof with Generative-Verifier RL and Population-Level Test-Time Scaling (2026-06)
- **简介**：MiniMax-M3 系列团队（Jiacheng Chen 等 23 人）。面向竞赛级数学证明（无可执行 ground truth，奖励须来自生成式验证器）。② 机制：先用**防御纵深式生成验证器**（专为低假阳性率设计）训练三种能力——证明生成、证明验证、批判条件化证明修复，并合并为单一 M3 模型；测试时 MaxProof 将模型同时作为生成器/验证器/精炼器/排序器，在候选证明群体上搜索并通过锦标赛选择返回最终证明（population-level test-time scaling）。③ 借助 MaxProof，M3 在 IMO 2025 达 35/42、USAMO 2026 达 36/42，**两项均超过人类金牌门槛**。
- **arXiv**：[2606.13473](https://arxiv.org/abs/2606.13473)

#### Reasoning Arena: Trace Tournaments When Verifiable Rewards Fall Short (2026-06)
- **简介**：来自 University of Cambridge 与 Mistral AI（Han Zhou、Albert Q. Jiang 等 5 人）。针对 RLVR 在"组内所有轨迹奖励相同"时组相对优势无梯度信号（零优势样本被浪费）的问题。② 机制：将非多样性奖励组路由至 judge 系统而非丢弃，构造**轨迹锦标赛（trace tournaments）**让推理轨迹两两对决揭示组内细粒度偏好；为高效化，每条新轨迹仅与动态更新的小型锚点池比较，并在不完整比较图上拟合 Bradley-Terry 模型，避免二次方成对比较。③ 在竞赛数学与编程基准上平均比 RLVR 基线高 7.6%，训练加速 27%–41%，节省近 50% 生成计算。
- **arXiv**：[2606.09380](https://arxiv.org/abs/2606.09380)

#### Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling (ICML 2026) (2026-06)
- **简介**：高丽大 Hyunjung Shim 团队系统化分析多模态 LLM-as-a-Judge 的 **感知判断偏差**：当视觉证据与文本叙述冲突时 judge 过度信任流畅文本而非视觉证据，并把失败模式拆为"感知能力不足"与"响应锚定"。提出 PPJD 数据集（perceptually perturbed judgment）和统一训练框架——结合 GRPO 结构化奖励 + batch-ranking 目标，无需显式成对标签即可实现全局排序。在多个 MLLM-as-Judge benchmark 上同时改进感知保真度、排序一致性、与人类评估的对齐。Generative-RM 路线下针对 reward 鲁棒性的代表新作。
- **arXiv**：[2606.02578](https://arxiv.org/abs/2606.02578)

#### Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward (VIGOR) (2026-05)
- **简介**：浙大 + 同济。RLVR 依赖 gold label / 域专 verifier 的硬约束。VIGOR 完全不用 verifier：给 prompt 采一组 completion，给 group 内 teacher-forced negative log-likelihood gradient 的 ℓ₂-norm 较小者更高奖励——直觉是更小梯度范数代表与当前 policy 对齐更好，可作为 intrinsic 偏好信号。两个工程要点：用 √T 缩放纠正 token 级平均梯度的系统性长度偏置；group-wise rank shaping 跨 prompt 稳定奖励量纲。Qwen2.5-7B-Base 在 MATH 后训练，math +3.31、code +1.91（仅在 math 上训）。给 RM-free / verifier-free 路线提供清晰的 intrinsic 信号方案。
- **arXiv**：[2605.09920](https://arxiv.org/abs/2605.09920)

#### Power Distribution Bridges Sampling, Self-Reward RL, and Self-Distillation (2026-05)
- **简介**：东京大学 Sato 实验室。给 power distribution（power sampling 的目标分布）一个统一视角：从 RL 角度，它是 "用模型自身 sequence-level log-probability 当 reward" 的 KL-regularized RL 的 closed-form 最优解；从蒸馏角度，对应一个共享同一目标分布的 power self-distillation——把 power sampling 的推理代价摊销到对 teacher 样本的 supervised 训练里。理论证 power self-distillation 可达成 self-reward sharpening，下游 true-reward 提升被 "true-reward 与 self-reward 在 power distribution 下的协方差" 控制。推理任务实验证明 self-distillation 在低推理成本下匹配/超 power sampling。给 self-reward RL 一个关于 "什么时候该用" 的清晰理论判据。
- **arXiv**：[2605.04542](https://arxiv.org/abs/2605.04542)

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

#### Post-Training Shifts Confidence: A Three-Stage Analysis of How SFT, RL, and OPD Shape Pre-, Intra-, and Post-CoT Calibration (PosConf) (2026-07)
- **简介**：Li 等（EIT-NLP）研究 SFT/RL/OPD 三类后训练如何重塑推理中的「置信度」，提出三阶段校准框架（CoT 生成之前/之中/之后，分别对应难度估计、提前终止、答案聚合）。发现：OPD 提供最有用的推理前置信度，SFT 给出最强的在线提前停止信号，RL 产出最可靠的 trace 级聚合信号；且置信度可靠性依赖位置——RL 置信度在「路径承诺」阶段后才有用，OPD 早期有用但后期可能反向校准。据此提出 PosConf（仅用可靠相对位置区间的置信度），把 RL 答案聚合较多数投票提升 6.1 分、紧 token 预算下 OPD 提前停止最多 +4.3 分。
- **arXiv**：[2607.13753](https://arxiv.org/abs/2607.13753)

#### Consensus as Privileged Context for Label-Free Self-Distillation (CANON) (2026-07)
- **简介**：Gkountouras、Jukić、Titov（爱丁堡/阿姆斯特丹方向）提出 CANON，把「多数投票共识」从过滤/偏好/标量奖励等受限用法升级为稠密 token 级监督。对每条无标签 prompt 采样多解、抽多数答案，并用一个到达该答案的解 condition 冻结模型快照，得到「共识锚定的教师」，再在模型自身 rollout 上逐 token 监督。在数学与科学推理基准上 pass@1 最高 +12 分，以 1/7 算力超过无标签 RL 6 分，接近用金标准解 condition 的教师；池化无标签数据训练后可迁移到留出基准。分析显示提升非纯分布锐化——训练后能解出此前 32 次尝试从未解出的题。
- **arXiv**：[2607.13643](https://arxiv.org/abs/2607.13643)

#### When LLMs Agree, Are They Right? Auditing Self-Consistency and Cross-Model Agreement as Confidence Signals (2026-07)
- **简介**：单作者 Kaihua Ding 审计 LLM-as-judge 集成 / 自一致性的核心假设——「一致（judge 间或自采样间）即正确」——并证明其不可靠：一致可能源于共享偏差、记忆化启发式或选项位置先验而非真相。在 53 个 runner、K=50 采样、GPQA Diamond 与 AIME 共 **26.5 万样本**的大规模跨 runner 研究中，以多数正确为部署标签、用层级 runner-clustered bootstrap 度量：一致性是**正但弱的预测子**（ρ 0.20–0.59），其有用性依 regime——对未饱和中档模型与算力分配最有用，对最一致的前沿模型最差（一致度 ≥0.8 覆盖 GPQA 77% 条目，其中 48% 是错的，即「过度自信却不更准」）。结论：自一致性是**有条件的正确性代理**而非独立置信分数，并公开逐 run 数据。
- **arXiv**：[2607.08065](https://arxiv.org/abs/2607.08065)

#### MILES: Modular Instruction Memory with Learnable Selection for Self-Improving LLM Reasoning (2026-07)
- **简介**：Ruilin Tong、Dong Gong（UNSW）针对「测试时问题顺序到达、可跨题积累可复用经验」但现有记忆方法（整解模板泛化差 / 启发式步级选择未对齐最终正确性）的不足，提出 **MILES**：在真实测试时约束（记忆增量扩张、监督有限）下**动态扩展步级记忆**并做面向正确性的记忆组合。记忆单元为「子目标嵌入 + 子指令」的非对称对，各配一个可学习的选择头；由此形成 coarse-to-fine 检索——粗层扩展记忆并从高置信样本收集监督训练选择头，细层对不确定样本用已学选择头重排候选并引导推理。在多个推理任务上持平或超越现有方法，取得更优的准确率-效率折衷，并展现鲁棒性与可迁移性。
- **arXiv**：[2607.06974](https://arxiv.org/abs/2607.06974)

#### When Does In-Context Search Help? A Sampling-Complexity Theory of Reflection-Driven Reasoning (2026-07)
- **简介**：Yotam Wolf、Noam Wies、Amnon Shashua（HUJI / AA-I）对「in-context search（模型迭代生成-批判-修订）」给出理论分析：将其建模为对推理轨迹的**近似推断**——base 模型定义先验、自反思提供后验更新反馈——并研究推理时的**采样复杂度**（达到高成功率所需的顺序尝试数）。核心结论：当反思能可靠定位早期错误时，in-context search 相对 base 可获**指数级改进**（用多项式次顺序尝试解决零样本通过率指数级小的问题），反之则相对并行采样无渐近收益；且该增益稳健可学（近似后验更新即足够、在 search rollouts 上做交叉熵训练即可用多项式样本恢复所需行为）。还表明在 RLVR 的分阶段抽象下，最优策略延拓实现同一后验重加权规则，并在真实大推理模型上验证关键定性预测。
- **arXiv**：[2607.06720](https://arxiv.org/abs/2607.06720)

#### Active-GRPO: Adaptive Imitation and Self-Improving Reasoning for Molecular Optimization (2026-07)
- **简介**：斯坦福 + 芝加哥大学 + Argonne（Xuefeng Liu、Mingxuan Cao、Le Cong 等）针对科学推理训练中 answer-only SFT 坍缩多步推理、RLVR 反馈稀疏、而 Reference-guided PO（RePO）受参考质量上限制约的困境，提出 **active reasoning** 范式：策略在每个实例上**自主决定何时模仿参考、何时强化自身发现**，并持续升级模仿目标。实例化为 **Active-GRPO**，含两个耦合机制：active imitate-reinforce（参考仍优于自身候选时做模仿学习，一旦策略生成超越参考的分子即转为 RL 自我提升）与 active referencing（持续用迄今最佳策略生成候选替换参考，逐步抬高模仿目标使参考始终有信息量）。在 TOMG-Bench MolOpt 上，匹配三种子评估下把平均 SR×Sim 从 GRPO 的 0.0959、RePO 的 0.1665 提升到 0.1773，LogP/MR/QED 上有统计显著增益。属自我提升 + GRPO 的跨域（分子优化）代表作。
- **arXiv**：[2607.00531](https://arxiv.org/abs/2607.00531)

#### Efficient and Trainable Language Model Test-Time Scaling via Local Branch Routing (LBR) (2026-06)
- **简介**：UC San Diego + Northwestern（Yutong Yin、Xin Eric Wang、Julian McAuley、Zhaoran Wang 等）提出 token 级测试时扩展框架 **Local Branch Routing (LBR)**：展开一棵小的局部 lookahead 树，将所有采样分支经 LM 前向，再用轻量 router 选定要 commit 的 depth-1 子树。通过对候选局部未来的隐藏状态做路由，使每个 token 决策能利用 root next-token 分布之外的证据，同时避免完整解级搜索；其 prune-shift-grow 解码保留离散分支身份并定义可计算的树轨迹似然，从而支持端到端 RLVR 联合优化 base 模型与 router。在数学推理上 Pass@1 与 Pass@32 均优于离散 CoT、原始离散 token RLVR 及 soft-token branching 基线。
- **arXiv**：[2606.25354](https://arxiv.org/abs/2606.25354)

#### ExTra: Exploratory Trajectory Optimization for Language Model Reinforcement Learning (2026-06)
- **简介**：新加坡国立大学 + SAP（Wenyang Hu、See-Kiong Ng、Bryan Kian Hsiang Low 等）针对 RLVR 在难度两端失效（易题全对、低多样、梯度弱；难题全错、无正奖励）的问题，提出 GRPO 兼容框架 **ExTra**，从模型自身 rollout 提取探索信号：(i) 在 GRPO 归一化后加入基于 embedding 的多样性 novelty reward，奖励多样的正确解；(ii) entropy-guided prefix regeneration，用熵信号给部分轨迹打分并从有潜力的中间步继续探索。在六个数学推理 benchmark 上，Qwen3-1.7B 较 GRPO 的 pass@1 约 +5 点、pass@16 约 +7 点，表明轨迹级探索信号可同时改善单样本准确率与推理期覆盖。
- **arXiv**：[2606.24994](https://arxiv.org/abs/2606.24994)

#### SPIRAL: Learning to Search and Aggregate (2026-06)
- **简介**：Stanford（Jubayer Ibn Hamid、Dorsa Sadigh、Chelsea Finn、Noah Goodman 等）指出后训练通常只优化"单条轨迹内的顺序推理"，却未联合优化测试时常用的并行采样与多轨迹聚合。提出 **Sequential-Parallel-Aggregative RL (SPIRAL)**：让 LM 先并行采样一组各自经顺序 CoT 的独立轨迹、再生成一条以这些轨迹为条件的聚合轨迹，所有组件端到端针对最终聚合响应的奖励优化；用 set RL 教模型产出"对聚合器整体有用"的轨迹集合，用标准 RL 教其聚合。推理任务上随推理算力有效扩展，三种算力 primitive 全部放大时较 GRPO 取得最多 11× 扩展效率与 15% 更高性能。
- **arXiv**：[2606.23595](https://arxiv.org/abs/2606.23595)

#### Continual Self-Improvement with Lightweight Experiential Latent Memories (ELM) (2026-06)
- **简介**：作者 Vaggelis Dorovatas、Nancy Kalaj、Rahaf Aljundi。研究 LLM 能否在线从自身推理轨迹中学习、将瞬时计算转化为持久可复用知识（无需外部监督或未来数据）。② 关键发现：基于原始轨迹的 ICL 无法泛化（token 级复用根本局限）；遂借鉴无监督 RL，**以自生成的测试时信号（多数投票）作为奖励**做轻量级每实例训练，并将推理时计算蒸馏为紧凑模块化"潜在记忆"（soft prompt，约占参数 ~0.001%），存储后在未来输入检索复用，靠模块化避免灾难性遗忘。③ 记忆仅需少量梯度步即媲美全参数更新/离线训练，在数学推理基准上显著优于 zero-shot 与原始数据 ICL，并可跨数据集（AIME24↔AMC23）有效迁移。
- **arXiv**：[2606.17803](https://arxiv.org/abs/2606.17803)

#### StarOR: Synergizing Tree Search and Test-Time Reinforcement Learning for Optimization Modeling (2026-06)
- **简介**：作者 Jiajun Li、Yu Ding、Wanyuan Wang 等 5 人。针对优化建模的层级化特性——一次性生成脆弱（早期符号错误传播）、固定策略搜索的 rollout 继承相似偏差且中间决策信用分配有限。② 机制：将 MCTS 与**测试时强化学习**结合，把建模过程分解为四阶段，在每个非终端节点用 GRPO 更新一个瞬态 LoRA adapter，以 MCTS 兄弟节点作为局部对比集，将搜索时探索转化为针对实例的策略精炼；并设计无监督多维度奖励系统，无需 ground-truth 标签即可为中间决策提供细粒度反馈。③ 在 5 个优化基准上，仅用 4B 骨干即达 SOTA，超越现有方法与前沿 LLM。
- **arXiv**：[2606.15197](https://arxiv.org/abs/2606.15197)

#### Step-by-Step Optimization-like Reasoning in LLMs over Expanding Search Spaces (OPT*) (2026-06)
- **简介**：Cambridge Mihaela van der Schaar 团队指出 RLVR 主要覆盖数学/代码 verifiable 域，对"许多有效计划中找一个高价值方案"这类决策任务力不从心。提出 **OPT***——一族优化风格任务，每个任务自带 feasibility checker + evaluator，complexity 参数扩张搜索空间，无需新人工标注。两种 regime：(i) **solver-guided 在线 policy 优化**——用 solver 当部分状态 value oracle，rank-based reward shaping 加固更优 next-step；(ii) solver 不可得时的 search-based offline RL。理论上把"在大搜索空间内成功"与"reasoner 每单位 search 预算抽取的信息量"联系起来；实证上 OPT* 训练改善逐步优化式推理。是 reasoning RL 在结构化优化任务上的扩展。
- **arXiv**：[2606.05464](https://arxiv.org/abs/2606.05464)

#### Reinforcement Learning from Rich Feedback with Distributional DAgger (DistIL) (2026-06)
- **简介**：USC + Anthropic（Rishabh Agrawal 等）批评 RLVR "single-bit reward"过窄，提出从 execution traces / tool outputs / expert corrections / model self-evaluations 等丰富反馈中学习。DistIL 是经典 DAgger 的分布式变体——learner 在当前 policy 访问的 state 上局部访问 expert 分布，导出简单的 forward cross-entropy 目标（支持 black-box expert），其序列级梯度通过 future expert-student disagreement 反传实现 **rich credit assignment**。证明反向 KL / JS 自蒸馏目标无法保证单调改进（即便 expert 奖励更高），forward CE 反而具单调改进与 regret 保证；并优化 teacher-weighted likelihood-of-success 的下界，提升 Pass@N。在科学推理 / 编码 / 难数学问题上稳超 RLVR 与自蒸馏基线。
- **arXiv**：[2606.05152](https://arxiv.org/abs/2606.05152)

#### Agentic Chain-of-Thought Steering for Efficient and Controllable LLM Reasoning (ACTS) (2026-06)
- **简介**：UCSD + Intuit（Yu Xia、Julian McAuley 等）把"how the model thinks"看作可控目标，将 reasoning steering 形式化为 MDP：controller agent 观察 reasoning trace + 剩余预算后发出"reasoning 策略 + 引导短语"steering action 推动 frozen reasoner 的下一步生成，保留连续生成性。controller 由合成多预算 steering trajectory 初始化，再以 budget-conditioned reward shaping 做 RL 微调。在多个 benchmark 上以大幅 token 节省匹配 full-thinking 性能，并提供 accuracy-efficiency 可控权衡。是 test-time scaling × CoT 控制的代表性新工作。
- **arXiv**：[2606.03965](https://arxiv.org/abs/2606.03965)

#### ATLAS: Agentic Test-time Learning-to-Allocate Scaling (2026-06)
- **简介**：UCSD Pengtao Xie 团队提出 **ATLAS**，把 test-time scaling 的 orchestration 本身交给 LLM controller 做端到端控制：单 action `explore` 调度独立 solver 重新尝试原问题，controller 决定是否再取证据、何时停、怎么合成最终答案；action space 可扩展，每次 explore 可指定 solver / reasoning effort / prompting 策略。Claude Sonnet 4.6 backbone 上达到 HLE-Verified **56.00%**、LiveCodeBench **82.29%**、GPQA-Diamond **85.75%**、BabyVision **23.71%**，API 调用远少于固定流程基线。多模型扩展 ATLAS-MM 进一步把 HLE-Verified 提升到 60.00%、LiveCodeBench 到 85.63%。
- **arXiv**：[2606.01667](https://arxiv.org/abs/2606.01667)

#### Better, Faster: Harnessing Self-Improvement in Large Reasoning Models (HSIR / H-GRPO) (2026-05, ICML 2026)
- **简介**：武汉大学 + Liang Ding / Bo Du / Dacheng Tao。诊断 LRM 自我改进的两个失败源：数据不平衡（自生成多简单样本，关键难样本稀缺）、过度思考（自生成轨迹多冗长冗余）。HSIR 框架两件事：verify-after-exit 采样策略提升困难 query 的高质量解收集；intrinsic diversity 分数量化并过滤过度思考样本。可挂在多种 post-training 范式上；进一步把 intrinsic diversity 作外部奖励嵌进 RL 得到 H-GRPO。多基准平均 +10.9 pp，推理效率最大 -42.4% 开销。
- **arXiv**：[2605.24998](https://arxiv.org/abs/2605.24998)

#### G-Zero: Self-Play for Open-Ended Generation from Zero Data (2026-05)
- **简介**：WashU + UVA Yu Meng 等。给开放式（不可验证）任务一个 verifier-free、共进化的自改进框架。核心创新 Hint-δ 是一种 intrinsic reward，量化 Generator 在 *无 hint* 与 *条件于自生成 hint* 两种响应之间的预测漂移。Proposer 用 GRPO 训练，专门合成挑战 query 与 informative hint 去打 Generator 盲点；Generator 用 DPO 内化 hint-guided 改进。理论上给 idealized standard-DPO 版本证明 best-iterate suboptimality 保证（要求 Proposer 提供足够的探索覆盖、数据筛选保证 pseudo-label 噪声低）。完全从内部分布动力学派生监督，绕过外部 judge 的能力上限——给 self-evolving LLM 在不可验证域提供新路径。
- **arXiv**：[2605.09959](https://arxiv.org/abs/2605.09959)

#### Exploration-Driven Optimization for Test-Time Large Language Model Reasoning (EDO / ED-iDPO / ED-GRPO) (2026-05)
- **简介**：Georgia Tech Bo Dai 团队，TMLR 2026 接收。点出推理时 / RL 后训练的根本张力：inference-time 方法需要从相对扁平的分布做多样化采样，而 RL 后训练本质让分布更尖锐。EDO 把 reward-biasing 风格的探索目标推广到迭代后训练，集成进标准 RL 目标，鼓励解的多样性同时不损 reasoning。分别落到 iDPO（→ED-iDPO）和 GRPO（→ED-GRPO）。三个 in-distribution 基准 +1.0~1.3、五个 OOD 任务 +1.5（与 self-consistency 配合时收益最大）；同时保持模型熵、稳定 RL 动态、缓解 over-optimization collapse。给 "RL 训练到底是该锐化还是保多样" 给出可调和的形式答案。
- **arXiv**：[2605.09853](https://arxiv.org/abs/2605.09853)

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

#### From Atomic Actions to Standard Operating Procedures: Iterative Tool Optimization for Self-Evolving LLM Agents（EvoSOP） (2026-07)
- **简介**：中国人民大学 + 阿里（Haipeng Ding、Yuexiang Xie、Yaliang Li、Bolin Ding 等）提出让 agent 把原子动作合成为可复用的标准作业流程（SOP），作为封装多步逻辑的高阶可调用工具。EvoSOP 从执行轨迹抽取 SOP，并经构建-合并-评估-剪枝的系统化生命周期迭代优化工具集。实验显示显著提升任务成功率、大幅减少交互轮数，形成可靠高效的工具使用范式。
- **arXiv**：[2607.07321](https://arxiv.org/abs/2607.07321)

#### Beyond Static Evaluation: Building Simulation Environments for Scalable Agentic Reinforcement Learning（AgenticAI-Supervisor） (2026-07)
- **简介**：Akshay Arora 等提出 AgenticAI-Supervisor —— 一个 API+UI 驱动的 RL Gym 环境，解耦环境构建与可扩展执行，以可验证执行结果生成高保真轨迹并做多维奖励塑造，通过严格内部状态校验缓解 reward hacking。以客服 agent 案例展示闭环反馈优化，后续将扩展 Computer Use / Tool Use。
- **arXiv**：[2607.05773](https://arxiv.org/abs/2607.05773)

#### Beyond Next-Token Prediction: An RLVR Proof of Concept for Tool-Use Agents on Atlassian Workflows (2026-07) (RLVR-Atlassian)
- **简介**：Centific 团队的 RLVR 概念验证，针对企业 SaaS 工作流中「命中正确 endpoint、正确嵌套参数、正确顺序」的目标错配。构建 5 个模拟 Jira REST v3 / Confluence v2 的合成环境（schema 级保真），奖励完全由工具调用 trace 计算，无实时 API、无学习式裁判、无人工标注。用同一批 checker 驱动 GRPO 训练 Qwen3-1.7B / Qwen3.5-4B，在 4 个非退化场景上将平均奖励从 4B 基线 0.35–0.92 提升到 0.95–1.00（Confluence 建页 0.35→1.00 增幅最大）。
- **arXiv**：[2607.01465](https://arxiv.org/abs/2607.01465)

#### Next-Generation Agentic Reinforcement Learning Systems Enable Self-Evolving Agents (2026-07) (AReaL 2.0)
- **简介**：蚂蚁 / 港科大 / 清华 AReaL 团队的立场+系统论文，指出企业级自演进智能体的瓶颈不在 RL 算法而在在线 agentic RL 系统，提出三支柱：跨异构 agent 范式承载步粒度学习信号的标准化轨迹数据协议、把真实工作负载转为可治理学习底料的企业级数据代理、基于轨迹统计自动决定何时更新权重/演化 in-context harness 的统一演化控制平面；并以 AReaL2.0 实例化一条从部署工作负载做策略权重在线更新的 agent-oriented RL 闭环。
- **arXiv**：[2607.01120](https://arxiv.org/abs/2607.01120)

#### Why Multi-Step Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Fix It (2026-06)
- **简介**：中科院自动化所 Yupu Hao、Zhuoran Jin、Kang Liu、Jun Zhao 等。针对"工具调用 RL 单独训练常不稳定/收益有限、甚至灾难性坍缩"的现象做机理分析：失败源于特定控制 token 的概率异常飙升、破坏结构化执行，而底层工具使用能力其实仍在、只是被特定格式掩盖。据此系统考察多类监督信号（off-policy 监督、hint 引导、错误样例监督等）在同步/交错训练方案下的效果，发现 SFT 与 RL 交错可显著改善稳定性、但在 format/content OOD 评测下性能退化，并分析了学习率与跨设定泛化影响。代码已开源（Tool-RL-Box）。
- **arXiv**：[2606.26027](https://arxiv.org/abs/2606.26027)

#### APPO: Agentic Procedural Policy Optimization（APPO） (2026-06)
- **简介**：作者 Xucong Wang、Ziyu Ma、Yong Wang、Yuxiang Ji、Shidong Yang、Guanhua Chen、Pengkun Wang、Xiangxiang Chu（含美团/AMAP 系作者群）。针对 agentic RL 普遍把信用分配在"工具调用边界、固定工作流"等粗粒度单元、难以定位哪些中间决策真正影响下游结果的问题。② 先验分析发现：有影响力的决策点广泛分布于整个生成序列、而非集中在工具调用处，且仅凭 token 熵无法可靠反映其对最终结果的影响。据此提出 APPO，将"在何处分支 + 分支后如何分配信用"从粗粒度交互单元下移到序列中的细粒度决策点：用结合 token 不确定性与"后续延续的策略诱导似然增益"的 Branching Score 选择分支位置（过滤掉虚假高熵位置），并引入过程级（procedure-level）优势缩放在分支 rollout 间更好地分配信用。③ 在 13 个基准上较强 agentic RL 基线持续提升近 4 分，同时保持高效工具调用与行为可解释性。
- **arXiv**：[2606.12384](https://arxiv.org/abs/2606.12384)

#### Exploring Agentic Tool-Calling Decisions via Uncertainty-Aligned Reinforcement Learning (TRUST) (2026-06)
- **简介**：上海 AI Lab / 上交团队（Yijin Zhou 等）观察到面向决策的 RL 会削弱"正确动作 vs 错误动作"的不确定性分离，导致过度自信的错误。提出 TRUST：把不确定度量化作为"排斥力"加入奖励设计，并配合轻量级 key-turn 标注做多轮轨迹后训练；在多个 tool-use 基准上同时改进决策质量与不确定度可靠性。
- **arXiv**：[2606.06976](https://arxiv.org/abs/2606.06976)

#### Tool-Aware Optimization with Entropy Guidance for Efficient Agentic Reinforcement Learning (TAO-RL) (2026-06)
- **简介**：南大 Hongye Cao 等提出 TAO-RL，针对工具集成 RL 的训练不稳定问题做两件事：(i) 工具感知轨迹过滤——丢弃"全失败工具调用"和"全对/全错"退化组；(ii) 在 post-tool-call token 处注入熵引导奖励，鼓励关键决策点的多样化推理路径。在 7 个推理 benchmark、3 个模型规模上一致超越基线。
- **arXiv**：[2606.03762](https://arxiv.org/abs/2606.03762)

#### AEM: Adaptive Entropy Modulation for Multi-Turn Agentic Reinforcement Learning (2026-05)
- **简介**：百度 + 清华陈省身数学中心团队提出 AEM，针对 agentic RL 中"稀疏 outcome 奖励 → 信用分配难"的问题，提出**无需过程监督**的自适应熵调制方法。将 token-level 熵动力学**提升到 response 级**（与 LLM agent 的实际动作粒度对齐），并基于 sampled-response advantage 与 relative surprisal 的互动关系推导出 response-level uncertainty proxy 来重缩放 advantage，自然地完成 explore→exploit 过渡。在 ALFWorld、WebShop、SWE-bench-Verified（1.5B–32B 模型）上一致提升强 RL 基线，集成到 SOTA 软件工程 RL 框架可再 +1.4%。
- **arXiv**：[2605.00425](https://arxiv.org/abs/2605.00425)

#### T²PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning (2026-05)
- **简介**：UCLA + Amazon 团队发表于 ICML 2026 Spotlight，针对多轮 agentic RL 训练**坍塌**问题，提出 Token- and Turn-level Policy Optimization (T²PO) 双层不确定性感知探索控制框架。Token 层：当 marginal uncertainty 变化低于阈值时**触发 thinking intervention**；Turn 层：识别"探索进展可忽略"的轮次并**动态重采样**避免无效 rollout。在 WebShop、ALFWorld、Search QA 上显著改善训练稳定性与最终性能。
- **arXiv**：[2605.02178](https://arxiv.org/abs/2605.02178)

#### EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL (2026-05)
- **简介**：华为诺亚方舟 + 港科大团队提出 EnvFactory，全自动地从真实资源中**探索并验证**有状态可执行工具环境，并通过 topology-aware 采样 + calibrated refinement 合成带**隐式意图**的多轮轨迹。仅用 7 个领域的 85 个验证环境就生成 2,575 条 SFT/RL 轨迹，使用环境数仅约前人 1/5 却在 BFCLv3 (+15%)、MCP-Atlas (+8.6%)、τ2-Bench/VitaBench (+6%) 全面领先。
- **arXiv**：[2605.18703](https://arxiv.org/abs/2605.18703)

#### Polar: Agentic RL on Any Harness at Scale (2026-05)
- **简介**：NVIDIA 团队提出 Polar，把 agent harness 当作黑盒——通过**代理 LLM API 调用并记录 token-level 模型交互**重建 token-faithful 轨迹，再交给独立训练器。彻底解耦 harness、训练 infra 和 RL 算法，单纯用 GRPO 即可在 SWE-Bench Verified 上让 Qwen3.5-4B 在 Codex/Claude Code/Qwen Code/Pi 四个 harness 上分别 +22.6/+4.8/+0.6/+6.2 分。已注册为 NeMo Gym 环境之一。
- **arXiv**：[2605.24220](https://arxiv.org/abs/2605.24220)

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

#### Branching Policy Optimization: Sandbox-Native Language Agent Reinforcement Learning (BPO) (2026-07)
- **简介**：Bowei He, Yankai Chen, Xiaokun Zhang, Xue Liu（McGill 等，WAIC Academic 2026）。利用智能体沙箱「确定、可快照、可从任意中间态恢复」的特性，改造 rollout 拓扑：在高熵决策点自适应快照，每分支点 fork K 个动作各自 rollout 到终止，并用兄弟回报而非独立 prompt 计算逐步优势；证明该估计无偏且方差严格低于轨迹级基线。在 WebShop、ALFWorld、SWE-bench Verified 上（Qwen2.5-7B / Llama-3.1-8B）较 GRPO/RLOO 成功率提升 3.6–6.1 个点，梯度范数方差减半，少 38% 策略更新即匹配最佳基线。
- **arXiv**：[2607.14171](https://arxiv.org/abs/2607.14171)

#### TRACE: Turn-level Reward Assignment via Credit Estimation for Long-Horizon Agents (2026-07)
- **简介**：Microsoft Research（Leitian Tao, Baolin Peng, Jianfeng Gao, Sharon Li 等）。针对长 horizon 多轮工具智能体结果奖励稀疏、高方差且误导的问题，TRACE 将 rollout 表示为工具调用边界处的状态转移，用冻结参考模型获取 gold-answer 对数概率并转为 log-ratio 状态值，再以时序差分（TD）变化导出逐动作奖励，无需额外 critic 或过程标注。在闭网 BrowseComp-Plus 上将 Qwen3-4B 从 7.2 提升到 35.6、Qwen3-30B-A3B 从 8.4 提升到 42.6，且行为可迁移至开放网络基准。
- **arXiv**：[2607.13988](https://arxiv.org/abs/2607.13988)

#### STAMP: Provenance-Guided Credit Assignment for Deep Search Agents (2026-07)
- **简介**：Ke Xu, Han Xu 等。针对深度搜索智能体 RL 中「奖励—信用错配」（暴露支撑文档的动作得不到定向信用）问题，STAMP 用基于参考的 verifier 判定每条被引文档是否支撑训练期证据图中的实体/关系，并用「首次暴露归因」把被支撑引用回溯到首次surface 它的动作；通过保号优势调制注入 step credit，不改变轨迹级奖励与组内排序。在 BrowseComp / BrowseComp-ZH / xbench-DS 上较 GRPO 基线分别 +2.0/+5.5/+3.0 点。
- **arXiv**：[2607.11172](https://arxiv.org/abs/2607.11172)

#### Entropy Pacing Policy Optimization for Multi-Task Agentic Reinforcement Learning（EPPO） (2026-07)
- **简介**：南洋理工/阿里通义（Zetian Hu、Shunyu Liu、Yongbin Li、Dacheng Tao 等）指出多任务 agentic RL 中的"探索-利用节奏错配"：易任务过早收敛到低熵策略阻碍难任务，难任务又把易任务推回高熵，产生跨任务熵交叉与频繁熵尖峰。提出 EPPO，以任务级动态裁剪替代 GRPO 固定裁剪阈值——对过自信任务收紧、对欠探索任务放松更新。多任务 agentic 基准上优于同类方法。
- **arXiv**：[2607.07178](https://arxiv.org/abs/2607.07178)

#### TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training（TurnOPD） (2026-07)
- **简介**：复旦（Yuhang Zhou、Jingjing Chen 等）提出回合级预算化的 on-policy 蒸馏。针对朴素 agent OPD 的两大低效——尾部回合浪费 wall-clock 且 KL 监督弱噪、轨迹级 KL 把损失集中在浅层 token——设计自适应 rollout 深度预算（探针式回合统计定长）与渐进式回合归一化损失预算（KL 权重由 token 级转向回合平衡）。在 ALFWorld/WebShop/多跳搜索上于同等 wall-clock 下超越朴素 OPD，推进精度–时间前沿。
- **arXiv**：[2607.05804](https://arxiv.org/abs/2607.05804)

#### STAPO: Selective Trajectory-Aware Policy Optimization for LLM Agent Training（STAPO） (2026-07)
- **简介**：ACL 2026 主会论文（Qiuyi Qi 等）。针对长程 agent 的"轨迹忽视"（中间步丢失任务目标与交互历史），提出归一化熵度量相对平均行为的置信偏差，定位与轨迹忽视相关的离群步，并以轨迹感知奖励 + 轨迹无关惩罚的分层组内 RL 联合优化。在 ALFWorld、WebShop、Search-QA 上达到 SOTA 并显著缓解轨迹忽视。
- **arXiv**：[2607.04963](https://arxiv.org/abs/2607.04963)

#### RSPO: Reward-Swap Policy Optimization for Multi-Turn LLM Agents（RSPO） (2026-07)
- **简介**：腾讯优图实验室（Qiang Liu 等）提出面向多轮 LLM 智能体的"奖励交换"策略优化。用稠密过程奖励训练探索智能体以采集多样化轨迹，再把奖励标签换回真实结果奖励喂给目标策略，从而兼得探索多样性与结果对齐、规避 reward hacking。在 WebShop、ALFWorld 上，应用于 GRPO/PPO/GiGPO 均获一致提升。
- **arXiv**：[2607.04713](https://arxiv.org/abs/2607.04713)

#### ECHO: Learning Epistemically Adaptive Language Agents with Turn-Level Credit (2026-06) (ECHO-Epistemic)
- **简介**：Nath & Krishnaswamy 提出 Epistemic Decision Processes (EDPs) 的信念状态形式化，证明信念无关策略的误差会随 horizon 指数放大、聚合轨迹回报无法识别每轮贝叶斯优势；据此提出 ECHO (Epistemic Credit for History-Conditioned Optimization)，用后验敏感奖励做 turn-level 信用分配的裁剪式策略梯度。在 Clue Selector Game 上在分辨率、信息增益、效率上显著优于轨迹级 GRPO。
- **arXiv**：[2606.29745](https://arxiv.org/abs/2606.29745)

#### ATOD: Annealed Turn-aware On-policy Distillation for Multi-turn Autonomous Agents (2026-06) (ATOD)
- **简介**：面向小模型长 horizon 交互智能体，提出退火式 OPD-RL 混合调度——早期以在线蒸馏快速逼近教师、后期逐步增强 RL 探索；并引入 Turn-level Disagreement-Uncertainty Reweighting (T-DUR) 对高价值轮次加权，改善长轨迹的稠密监督。在 ALFWorld / WebShop / Search-QA 上平均成功率较 OPD +3.03、较 GRPO +23.62，并超过对应教师模型 2.16 分。
- **arXiv**：[2606.27814](https://arxiv.org/abs/2606.27814)

#### Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning (G2PO) (2026-06)
- **简介**：微软（Yunan Wang、Shaohan Huang、Furu Wei、Qi Zhang 等）提出 G2PO，针对长 horizon agentic RL"奖励稀疏/延迟 + step 级信用分配仍粗糙、把探索当作孤立线性轨迹、忽略状态转移图结构"的瓶颈。G2PO 把线性交互轨迹显式转换为全局状态转移图：通过跨轨迹聚合相同观测引入"组聚合状态价值估计"以降低采样方差与轨迹依赖偏差；把 agent 动作重定义为状态节点间的转移，提出 edge-centric 优势估计，并在全图范围标准化 TD 误差以识别并优先关键转移。在 WebShop、ALFWorld、AppWorld 上较 SOTA prompt-based 与 RL 基线显著领先，成功率较 GRPO 最高 +22.2%。
- **arXiv**：[2606.22995](https://arxiv.org/abs/2606.22995)

#### Drowning in Routine: Signal Dilution in Multi-Turn Agent Training (2026-06)
- **简介**：Mila / Polytechnique Montréal（Yann Pernot、Vi Retault；FAGEN Workshop @ ICML 2026）。理论刻画多轮 agent 的"信号稀释"：多轮中有些动作改变下游回报分布、有些只是必要但"奖励等价"的常规执行；轨迹级信用分配的代价并非仅由长 horizon 决定，而由"决策密度 ρ"（影响回报的轮次占比）支配。低 ρ 时常规轮给 GRPO 等轨迹级估计器增加梯度方差却不增加期望信号，推得 turn 级/轨迹级信噪比按 ρ^{-1/2} 衰减（critic 误差可控时）；高 ρ 时轨迹级方法仍有竞争力、且可省去 critic 成本。在 ρ 可精确调控的受控环境中以 R²=0.999 复现预测的标度律。
- **arXiv**：[2606.22164](https://arxiv.org/abs/2606.22164)

#### TRACE: A Unified Rollout Budget Allocation Framework for Efficient Agentic Reinforcement Learning（TRACE） (2026-06)
- **简介**：作者 Heming Zou、Qi Wang、Yun Qu 等 12 人（含腾讯系 Saiyong Yang、国防科大 Xin Xu、清华 Xiangyang Ji）。针对 rollout-密集的策略优化常因"奖励对比不足"受限——过简/过难 prompt 产生低方差反馈，且 outcome-only 奖励把同一终局评估广播给多轮 rollout 的每个决策；以往工作只在 prompt 层面利用样本信息量，忽略了同一 rollout 内"跨轮前缀"的信息量差异。② 方法：把每个 ReAct 式 thought-action-observation 轮建模为语义上不同的节点，使预算分配从 prompt 根节点扩展到"轮级前缀 + 后续延续"，自然形成树状 rollout（Tree Rollout Allocation for Contrastive Exploration）；用一个可泛化的共享预测器从前缀历史估计各锚点的条件成功概率，引导预算优先投向"最可能产生混合终局奖励"的根与中间前缀，从而在固定采样预算下增强奖励对比、放大策略更新信号。③ 在典型 agentic 基准上取得竞争性性能与效率增益，例如等采样成本下将 Qwen3-14B 多跳 QA 平均准确率较强基线提升 2.8 分。
- **arXiv**：[2606.11119](https://arxiv.org/abs/2606.11119)

#### Maximizing Rollout Informativeness under a Fixed Budget: A Submodular View of Tree Search for Tool-Use Agentic RL (InfoTree) (2026-05)
- **简介**：上海交大团队首次将固定预算下的"rollout informativeness" (RIFB) 形式化为 GRPO 注入的期望非消失策略梯度质量，证明**预算无关采样器对困难提示存在远离零的崩溃率**。把中间状态选择重铸为单调子模最大化（贪心 1−1/e 近似）；其闭式边际收益恰好就是带 token 熵奖励的不确定性感知 UCB (UUCB)，从而把"token 熵奖励"从经验技巧变成解析结果。框架 InfoTree 加上自适应预算分配器和异步 speculative expansion，在 9 个 benchmark（数学/搜索/编码/OS）上击败 flat GRPO、Tree-GRPO、AT2PO、CW-GRPO、RC-GRPO、DeepSearch。
- **arXiv**：[2605.05262](https://arxiv.org/abs/2605.05262)

#### Healthcare AI GYM for Medical Agents: Turn-level Truncated On-Policy Distillation (TT-OPD) (2026-05)
- **简介**：作者构建涵盖 10 个临床领域、3.6K+ 任务、135 个工具、828K 医学语料的 Healthcare AI Gym，并系统验证：vanilla GRPO 在 agentic 多轮医学任务中会**退化为冗长单轮独白**（length 爆炸 + tool-use 频率塌缩）。提出 TT-OPD——**梯度无关的 EMA 教师**利用 outcome-privileged information 在每一轮对话提供 dense outcome-aware KL 正则。在 18 个 benchmark 中 10 个最优，平均比非-RL 基线 +3.9pp。
- **arXiv**：[2605.02943](https://arxiv.org/abs/2605.02943)

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

#### SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning (2026-07)
- **简介**：来自清华 / 中科院自动化所等团队（Jinyang Wu, Jianhua Tao 等）。针对结果型 RL 稀疏轨迹级奖励对中间决策监督不足的问题，提出 SEED——将已完成的 on-policy 轨迹转化为训练期的 hindsight「技能」（可复用工作流、关键观察、失败规避规则），当前策略同时充当轨迹收集者与分析者；再将技能诱导的概率偏移转化为稠密 token 级 on-policy 蒸馏信号，与结果型 RL 联合优化。在文本与视觉智能体任务上一致提升性能与样本效率，并对未见场景鲁棒泛化。
- **arXiv**：[2607.14777](https://arxiv.org/abs/2607.14777)

#### ToolAnchor: Anchoring Counterfactual Context to Boost Agentic Tool-use Capability (2026-07)
- **简介**：Weiting Liu, Jieyi Bi, Yining Ma 等（复旦等）。识别工具集扩展中的「行为惯性」——智能体倾向沿用熟悉工具与既有推理模式而难以纳入新工具；提出在关键决策点注入反事实 anchor 上下文以打破惯性、恢复失败轨迹并激发被抑制的能力，ToolAnchor 用 teacher 模型假设反事实上下文、经 student rollout 验证并通过 agentic 后训练内化成功干预。在 GAIA、BrowseComp、VDR-Bench 上于扩展工具集下保持有竞争力表现。
- **arXiv**：[2607.14145](https://arxiv.org/abs/2607.14145)

#### LOTAPO: Leave-One-Turn Attribution for Self-Generated Process Rewards in Multi-Turn Search Reasoning (2026-07)
- **简介**：Qiang Zhu, Jiajun Wu, Longyi Wang。提出基于反向「留一轮」归因的自生成过程监督：将某一搜索轮及其检索观测替换为固定 [DELETE] 占位并测量策略对 gold 答案平均对数似然的变化（Answer-Likelihood Gain），据此估计该轮贡献，并用符号一致性门控保留方向一致的归一化过程优势；无需额外奖励模型 / teacher / verifier / LLM-as-Judge。在七个知识密集型 QA 数据集上平均 EM 0.326，超过最强 step-reward 基线 IGPO 0.053。
- **arXiv**：[2607.13501](https://arxiv.org/abs/2607.13501)

#### CRAFT: Counterfactual Credit Assignment from Free Sibling Rollouts for Self-Distilled Agentic Reinforcement Learning (2026-06) (CRAFT)
- **简介**：针对自蒸馏 agentic RL 中「单标量 teacher-student 对数概率差」信号既回溯又符号盲的缺陷，提出三支柱信用分配：Pillar 1 复用 GRPO 已采样的 G-1 兄弟 rollouts、按对数概率差重要性加权，得到近零额外算力的有符号逐 token 反事实信用；Pillar 2 非对称控制器动态调节蒸馏权重与参考 KL 权重；Pillar 3 按信用符号在 mode-seeking / mode-covering 间切换 KL 惩罚。给出估计量一致性与方差界证明，并在三类 agentic 环境、四种模型规模、五种端到端方法上评估。
- **arXiv**：[2606.29476](https://arxiv.org/abs/2606.29476)

#### HERO: Hindsight-Enhanced Reflection from Environment Observations for Agentic Self-Distillation（HERO） (2026-06)
- **简介**：作者 Haoran Liu、Yuwei Zhang、Xiyao Li、Bohan Lyu、Jingbo Shang（UCSD 系）。RL 通常仅靠轨迹终局结果改进多轮 Agent，难以对各中间轮做信用分配；近期 on-policy 自蒸馏通过 self-teacher 把特权反馈转成稠密 token 级监督，但将其朴素扩展到多轮设置会出现意外性能退化——作者归因于"特权反馈（成功轨迹/终局结果）与学生当前决策上下文之间缺乏对齐"。② 提出 HERO：以"下一步环境观测"作为局部对齐反馈——每次 rollout 后对已完成交互进行反思（后见之明），把每个观测转化为紧凑的逐轮诊断（turn-level diagnosis），捕捉关于原动作的可操作反馈（必要性 / 有效性 / 失败原因）。③ 在 TauBench 与 WebShop 上，相比"仅环境反馈的自蒸馏"和 GRPO 提升任务成功率并减少无谓轮次；在训练轮预算受限、成功 rollout 稀少、GRPO 奖励对比信号弱时尤为有效。
- **arXiv**：[2606.11559](https://arxiv.org/abs/2606.11559)

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

#### HyMobileAgent: Data-Environment Co-Scaling for Efficient GUI Agents (2026-07)
- **简介**：Hy Vision Team（Huawen Shen 等）。基于 vision-native 基座 Hy3.0-VL-A3B（原生任意分辨率、32K 上下文）构建移动 GUI 智能体，提出「数据—环境协同扩展」框架：GUI 感知飞轮、教程视频→结构化交互数据管线、跨 2000+ 沙箱/真机的百万级动作数据管线与自动失败归因、PhoneWorld Mock App Factory（34 个可重置 App、34000+ 任务），以及带死循环检测的 Planning-and-Reflection 机制；训练配方含 mid-training + SFT + 带任务特定奖励设计的 RL。
- **arXiv**：[2607.14548](https://arxiv.org/abs/2607.14548)

#### Learning Robust Execution in Robotic Manipulation with Agentic Reinforcement Learning (2026-07)
- **简介**：Xiaopeng Zhang, Yueyang Weng, Qi Liu 等。针对机器人操作中不确定性、长 horizon 与误差累积导致执行失稳的问题，提出两个运行时执行质量度量，以及一个 agentic RL 框架——高层策略基于近期执行历史在少量执行模式间选择以调控执行过程，退化时触发恢复机制回到既访名义状态使任务续行（而非直接学习底层动作）。在 LIBERO 上标准设置成功率提升最高 13.7%、扰动设置最高 39.2%。
- **arXiv**：[2607.13818](https://arxiv.org/abs/2607.13818)

#### Exploratory, Communicative, and Deployable: Vision-Driven Embodied Agents for Open-World Mobile Manipulation (REAL) (2026-07)
- **简介**：Boyu Mi, Tai Wang, Yao Mu, Jiangmiao Pang 等（InternRobotics，ECCV 2026）。面向开放世界移动操作的 agentic 框架 REAL：建立 sim-to-real 一致、无 oracle 感知的环境 API，并集成模拟用户实现人在环交互；在该环境内做数据收集、SFT 与在线 RL 分层训练。在交互任务上以 56.9% 成功率超越领先闭源 VLM，真实双臂移动机器人 60 回合端到端成功率 78.3%，零样本迁移到未见家居场景。
- **arXiv**：[2607.13653](https://arxiv.org/abs/2607.13653)

#### Joint On-and-Off Policy Learning for Vision-and-Language Navigation (JOP-VLN) (2026-07)
- **简介**：Qingrong He, Lin Zhao, Liang Lin 等（IROS 2026）。弥合 VLN 中模仿学习（IL+DAgger）与可验证奖励 RL 两大范式割裂：三阶段训练——先在专家示范上 IL 习得基础导航、再用 DAgger 生成启发式探索轨迹做 IL 增强纠错、最后实施联合 on/off 策略学习（高熵轨迹采样提效 + 纠错优先的轨迹排序）。在 VLN-CE R2R / RxR 上成功率 69.9% / 68.0%，R2R 上取得新 SOTA。
- **arXiv**：[2607.13461](https://arxiv.org/abs/2607.13461)

#### ScaleCUA: Scaling Computer Use Agents with Verifiable Task Synthesis and Efficient Online RL (2026-07)
- **简介**：清华 THUDM（Bowen Lv, Xiao Liu, Jie Tang, Yuxiao Dong 等）。面向计算机使用智能体（CUA）在线 RLVR 的可验证数据稀缺与训练低效两大瓶颈，提出统一框架 ScaleCUA：VeriGen 端到端生成可验证 RL 任务（迭代 docker 交互 + 多智能体反馈，产出 24K+ 可验证任务）、Frontier Sampling 按能力前沿分配 rollout、Visual Context Segmentation 滑窗（较逐步分解训练加速 2.83×）。在 OSWorld 达 68.7%、ScienceBoard 达 54.0%，为开源 CUA 新 SOTA。
- **arXiv**：[2607.11185](https://arxiv.org/abs/2607.11185)

#### Reinforcement Learning for Computer-Use Agents with Autonomous Evaluation (2026-06)
- **简介**：Marta Sumyk、Oleksandr Kosovan（GLOW @ IJCAI 2026 Workshop）针对开放桌面环境"缺乏可扩展、机器可读奖励信号"的痛点，提出用自主 VLM 评估作为 GUI agent 的可扩展监督信号：给定最终截图与原始指令，由 VLM 判定任务是否完成、提供终局反馈，无需任务特定启发式或人工标注。由于评估器不完美，将其反馈建模为"带噪二元奖励通道"并为 PPO 推导噪声校正奖励估计量。在 macOSWorld、Windows Agent Arena、OSWorld 上，校正后的评估器奖励平均较零样本 +12.6pp、较原始评估器微调 +5.1pp。
- **arXiv**：[2606.24515](https://arxiv.org/abs/2606.24515)

#### PhoneBuddy: Training Open Models for Agentic Phone Use (2026-06)
- **简介**：Zhengyang Tang、Xin Lai 等 26 人大团队（含 Benyou Wang、Ji-Rong Wen、Rui Yan 等）提出 PhoneBuddy——面向"agentic 手机使用"的训练配方与开放模型线。核心是把真实 App 环境与 mock App 环境 PhoneWorld（从真实 GUI 使用结构重建可运行 mock App）结合：先用两类环境采集的轨迹做共享 SFT，再对比"纯真机 RL"与"跨两类环境的 mixed RL"。在覆盖 App/小程序/跨 App 工作流的 150 任务真机人评中，成功率从 SFT 的 36.67% → 真机 RL 40.67% → mixed RL 45.33%；AndroidWorld 上 60.3% → 77.2% → 83.2%。结论：mock-App 训练不能替代真机 RL，但是可扩展、可重置、可自动校验的互补来源。
- **arXiv**：[2606.23049](https://arxiv.org/abs/2606.23049)

#### MobileForge: Annotation-Free Adaptation for Mobile GUI Agents with Hierarchical Feedback-Guided Policy Optimization (2026-06)
- **简介**：Guangyi Liu、Pengxiang Zhao 等团队针对"移动 GUI agent 适配真实 App 成本高（App 多、频繁更新、难以用人工任务/演示/奖励标注覆盖）"的问题，提出免标注自适应系统 MobileForge。系统含 MobileGym（把任务生成与 rollout 评估锚定在真实移动 App 交互上）与 HiFPO（Hierarchical Feedback-Guided Policy Optimization）——把轨迹结果、step 级过程反馈与纠错提示统一转化为"提示上下文化的 step 级 GRPO 更新"。仅用自动生成的免标注数据，就把 Qwen3-VL-8B 适配到 AndroidWorld 67.2% Pass@3（逼近闭源数据 GUI-Owl-1.5-8B 的 69.0%）；适配后的 ForgeOwl-8B 进一步达 77.6% Pass@3，并在 out-of-domain MobileWorld GUI-only split 上 41.0% 成功率，为该评测下最强开放数据移动 GUI agent。
- **arXiv**：[2606.19930](https://arxiv.org/abs/2606.19930)

#### GUI-AC: Enhancing Continual Learning in GUI Agents（GUI-AC） (2026-06)
- **简介**：作者 Can Lin（北邮）、Tao Feng（清华）、Hangjie Yuan（浙大）、Dan Zhang（NUS）、Yifan Zhu、Zhonghong Ou（北邮）。针对 GUI 数据天然非平稳——不断出现未见界面实例（新域、新分辨率）造成持续分布漂移，阻碍 GUI Agent 的持续学习；而强化微调（RFT）在 grounding 能力上表现出明显不稳定：奖励不连续与高方差振荡、rollout 结果分布不均给优势估计引入噪声导致策略过度自信，固定 clipping 边界又压制了适应新分布所需的策略概率上升、引发探索能力坍塌。② 提出 GUI-AC，引入 grounding 确定性（grounding certainty）支撑两个机制：Adaptive Advantage（下调噪声优势估计、抑制策略过度自信）与 Dynamic Clipping（放松 clipping 边界、扩大探索范围）。③ 大量实验显示二者联合提升性能，超越 SOTA 基线。
- **arXiv**：[2606.10522](https://arxiv.org/abs/2606.10522)

#### MIRAGE: Mobile Agents with Implicit Reasoning and Generative World Models (2026-06)
- **简介**：Zhichao Yang 等提出 MIRAGE：把可见的文本 reasoning trace 蒸馏为连续潜在表示，配合"潜在向量与未来截屏对齐"的生成式 world-model 目标，让 mobile agent 在隐空间里推理并预判界面状态。在 AndroidWorld 上以 3-5× 更低 token 预算匹配显式 CoT-SFT，比可比 instruction-tuned 基线高 10.2 分；AndroidControl 上 token 减少 75% 以上而 grounding 更好。
- **arXiv**：[2606.04627](https://arxiv.org/abs/2606.04627)

#### Multi-Agent Computer Use (MACU) (2026-06)
- **简介**：CMU（Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried）指出当前 CUA 几乎都是"单串行 agent"，对长程任务次优。论文论证应该转向多智能体 CUA：经理-员工式分工、并行执行、基于新信息持续重规划，并提出对应评测与系统设计原则。
- **arXiv**：[2606.01533](https://arxiv.org/abs/2606.01533)

#### ToolCUA: Towards Optimal GUI-Tool Path Orchestration for Computer Use Agents (2026-05)
- **简介**：阿里通义实验室 + 复旦 + 上海AI Lab 联合提出 ToolCUA，解决 CUA 在"GUI 点击"和"工具调用"两条路径下的**路径选择困境**。三阶段训练：（1）Interleaved GUI-Tool 轨迹缩放流水线（合成工具库 + 重利用纯点击轨迹），（2）Tool-Bootstrapped GUI RFT（warmup SFT + 关键切换节点单轮 RL），（3）Online Agentic RL with Tool-Efficient Path Reward（鼓励合理用工具 + 路径长度奖励）。基于 Qwen3-VL-8B-Instruct，OSWorld-MCP 上 46.85%（相对基础 +66%），平均完成步数仅 14.93（最低），跨 OS Windows zero-shot 33.8% 超越 Qwen3-VL-235B。
- **arXiv**：[2605.12481](https://arxiv.org/abs/2605.12481)

#### LiteGUI: Distilling Compact GUI Agents with Reinforcement Learning (2026-05)
- **简介**：MoonBit + 复旦团队提出无 SFT 训练范式，针对 2B/3B 端侧 GUI agent："Guided On-policy Distillation"（带 oracle 参考轨迹 + 动态检索）+ "Multi-solution Dual-level GRPO"（联合宏观子任务规划 + 微观执行匹配）。配套自动数据流水线生成多解标注的 GUI 轨迹，使轻量 agent 在多 benchmark 上达到 SOTA 同时与更大模型有竞争力。
- **arXiv**：[2605.07505](https://arxiv.org/abs/2605.07505)

#### OpenComputer: Verifiable Software Worlds for Computer-Use Agents (2026-05)
- **简介**：耶鲁 NLP（Arman Cohan 组）提出 OpenComputer：**verifier-grounded** 的可验证软件世界，包含 (1) app 特定状态 verifier 提供结构化检测端点，(2) 自进化验证层，(3) 任务生成流水线，(4) 完整轨迹记录 + auditable partial-credit reward harness。覆盖 33 个桌面应用、1000 个最终任务（浏览器/Office/创意/IDE/文件/通信）。实证显示其硬编码 verifier 比 LLM-as-judge 更接近人类裁判，前沿 agent 的端到端完成率仍然较低，open-source 模型对比 OSWorld-Verified 大幅下跌。
- **arXiv**：[2605.19769](https://arxiv.org/abs/2605.19769)

#### World Action Models: The Next Frontier in Embodied AI (2026-05)
- **简介**：复旦 + 新加坡国立等的综述（survey），首次系统形式化 World Action Models (WAMs)：把 VLA 的"反应式 obs→action 映射"和"world model 的预测式状态建模"统一为对未来状态-动作的联合分布。给出 Cascaded vs. Joint WAM 分类、按生成模态/条件机制/动作解码的子类，并梳理数据生态（teleop / 人类演示 / 仿真 / 第一视角网络视频）与评测协议（视觉 fidelity / 物理常识 / 动作合理性）。可作为 embodied VLA RL 子方向的导航文献。
- **arXiv**：[2605.12090](https://arxiv.org/abs/2605.12090)

#### SimWorld Studio: Automatic Environment Generation with Evolving Coding Agent for Embodied Agent Learning (2026-05)
- **简介**：UCSD + Mohamed bin Zayed University 团队提出基于 Unreal Engine 5 的 SimWorld Studio，核心是 **SimCoder**——一个工具/技能增强的编码 agent，用 verifier 反馈（编译错/物理 check/VLM 评判）自我进化地构建物理 grounded 3D 世界。生成的 world 导出为 Gym 接口，并与 embodied learner **协同进化**——agent 表现反过来引导 SimCoder 生成位于学习者能力前沿的自适应课程。三个 navigation 案例：协同进化比固定环境 +18 pts、比未训练 +40 pts。
- **arXiv**：[2605.09423](https://arxiv.org/abs/2605.09423)

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

#### A Learning-Rate-Gated Failure of GRPO in a Small Language and Vision-Language Model Web Agent: A Controlled Null and Its Mechanism (2026-07)
- **简介**：Chengguang Gan, Shiwen Ni 等。系统性研究（18 组控制实验，变动学习率/KL 权重/种子/初始化/裁剪）：在 4B–8B 规模、已基本掌握任务的强 SFT 基线上，GRPO 在 web agent 上无可信提升，文本轨道中高学习率反而可信变差；对照实验证明并非管线损坏（在奖励可采样任务上成功率 +22 点）。给出机制解释：GRPO 仅在「采样策略已比贪婪策略更常成功」的 headroom 存在时才有效，并定位退化/坍缩两区的因果差异（scale-dependent）。一个有价值的 agentic-RL 负结果。
- **arXiv**：[2607.12640](https://arxiv.org/abs/2607.12640)

#### Information Gain-based Rollout Policy Optimization: An Adaptive Tree-Structured Rollout Approach for Multi-Turn LLM Agents（IGRPO） (2026-07)
- **简介**：上海交大（Yijun Zhang、Jiaxin Ding、Xinbing Wang 等）提出以中间状态信息量为 rollout 组织原则的 IGRPO：按节点级信息量做预算感知的树状 rollout，更频繁扩展高信息分支、逐步抑制无望分支；并证明信息增益 rollout 诱导出显式的极限教师分布，从而统一自适应树搜索探索与原则化策略优化。七个搜索增强 QA 基准上在同等 rollout 预算下持续超越强基线。
- **arXiv**：[2607.06223](https://arxiv.org/abs/2607.06223)

#### When Should LLMs Search? Counterfactual Supervision for Search Routing (2026-07)
- **简介**：Minho Kim（ICML 2026 FAGEN Workshop）将"是否需要检索"形式化为实例级搜索路由问题：对比同一问题的无检索/强制检索结果，构造 NO SEARCH/SEARCH/UNSOLVED 的 oracle 作为评估与学习信号，用 SFT + 偏好优化训练路由策略。Gemma-E2B 与 Qwen3.5-4B 的路由 macro-F1 分别从 0.71→0.82、0.71→0.84。
- **arXiv**：[2607.05752](https://arxiv.org/abs/2607.05752)

#### MetaResearcher: Scaling Deep Research via Self-Reflective Reinforcement Learning in Adversarial Virtual Environments (2026-06)
- **简介**：Wei Yu、Suxing Liu 等提出 MetaResearcher，沿四个协同维度扩展 deep research agent 训练：① Evolving Virtual World——在训练环境注入时间动态与对抗性错误信息，迫使 agent 发展来源可信度评估与时间冲突消解能力；② Discovery-Oriented Tasks——超越纯事实检索的假设生成与矛盾消解任务；③ Self-Reflective Meta-Reward——在 GRPO 框架内联合优化答案正确性、搜索路径效率、反思深度与工具调用多样性，直接缓解前人"重复动作环"问题；④ Heterogeneous Multi-Agent Swarm——Scout/Filter/Synthesizer 专用模型经协同 RL 学习协作策略。基于 LiteResearcher 基建、零边际 API 成本训练，目标提升 GAIA/Xbench-DS 性能与对抗条件下的认知鲁棒性。
- **arXiv**：[2606.19893](https://arxiv.org/abs/2606.19893)

#### DEEPRUBRIC: Evidence-Tree Rubric Supervision for Efficient Reinforcement Learning of Deep Research Agents（DeepRubric） (2026-06)
- **简介**：作者 Minghang Zhu、Chuyang Wei、Junhao Xu、Yilin Cheng、Zhumin Chen、Jiyan He（含山东大学系）。Deep research agent 通过检索与推理合成长篇报告，rubric-based RL 用可核查标准把报告质量转成奖励信号，但其效率取决于这些标准能否可靠覆盖任务范围与证据需求；现有做法多让 LLM"为给定 query 生成 rubric"，一旦模型推断不出潜在信息需求，rubric 便会不完整、降低 RL 效率。② 提出 DeepRubric：反转该过程——先确定"一份有证据支撑的报告应被评估什么"，再据此合成对齐的 query–rubric 对。具体地，从采样的种子主题出发递归扩展"有证据支撑的子问题"构建证据树（evidence tree），叶子作为原子且可验证的评估目标，再用证据树合成训练 query 与 rubric，保证奖励恰好评估 query 所请求的信息。③ 用该框架构造 9K query–rubric 监督样本，以 rubric-based GRPO 训练 DeepRubric-8B，在三个基准上达到与此前开源 SOTA deep research 模型相当的性能，而 RL GPU-hours 约减少 13 倍。
- **arXiv**：[2606.17029](https://arxiv.org/abs/2606.17029)

#### Self-Evolving Deep Research via Joint Generation and Evaluation (SCORE) (2026-06)
- **简介**：HKUST(GZ) Han Zhu 等针对 deep research 报告"无标准答案、奖励不可验证"的痛点，提出 SCORE：将 evaluator 与 solver 共享参数、共同进化，并引入 meta-harness 根据 solver 表现动态控制评测环境，避免静态 LLM-as-judge 饱和。在多个 deep research benchmark 上一致提升报告质量。
- **arXiv**：[2606.04507](https://arxiv.org/abs/2606.04507)

#### Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses (2026-06)
- **简介**：UIUC / Stanford（Pengcheng Jiang, Jiawei Han 等）提出"把状态管理外置到环境侧 harness"的 search agent 范式：harness 维护候选池、重要性标签、证据链接、验证记录、压缩观测与预算化渲染；策略只做语义决策。20B Harness-1 在 8 个检索 benchmark 上平均 curated recall 0.730，比次优开源 search subagent 高 +11.4 分，并在迁移基准上更强。
- **arXiv**：[2606.02373](https://arxiv.org/abs/2606.02373)

#### OpenWebRL: Demystifying Online Multi-turn Reinforcement Learning for Visual Web Agents (2026-06)
- **简介**：UIUC × Microsoft（Rui Yang, Qianhui Wu, Jianfeng Gao 等）发布全开源框架 OpenWebRL，覆盖可扩展实时浏览器基础设施、SFT 初始化、多模态上下文管理、轨迹级成功裁判与多轮策略优化。仅 0.4K 初始化轨迹 + 2.2K 在线 RL 任务训练出 OpenWebRL-4B，Online-Mind2Web 67.0% / DeepShop 64.0%，与 OpenAI CUA、Gemini CUA 持平。
- **arXiv**：[2606.02031](https://arxiv.org/abs/2606.02031)

#### Argus: Evidence Assembly for Scalable Deep Research Agents (2026-05)
- **简介**：阿里巴巴达摩院 + 南洋理工 + 华盛顿大学团队提出 Argus，把 deep research 重定义为"用互补证据片段拼图"。Searcher（普通 ReAct）+ Navigator（共享 evidence graph，验证缺失片段并派遣 Searcher）协同；Navigator 用 RL 训练做 verify/dispatch/synthesize，Searcher 单独训练。35B-A3B MoE 主干下，单 Searcher +5.5pt、8 并行 Searcher +12.7pt（八个 benchmark 平均），64 Searcher 时 BrowseComp 86.2 超越所有专有 agent，且 Navigator 推理上下文 <21.5K token。
- **arXiv**：[2605.16217](https://arxiv.org/abs/2605.16217)

#### SciResearcher: Scaling Deep Research Agents for Frontier Scientific Reasoning (2026-05)
- **简介**：港科大（Yangqiu Song 组）提出 SciResearcher，针对前沿科学推理（领域知识稀疏 + 异构 + 需要复杂计算）。提出基于学术证据的全自动 agentic 数据合成框架（综合多种概念/计算任务），用精心策划的数据做 SFT + agentic RL，得到 SciResearcher-8B：HLE-Bio/Chem-Gold 19.46%（参数规模 SOTA），SuperGPQA-Hard-Biology 与 TRQA-Literature 提升 13–15 pp。
- **arXiv**：[2605.01489](https://arxiv.org/abs/2605.01489)

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

#### From Noisy Traces to Root Causes: Structural Trajectory Analysis and Causal Extraction for Agent Optimization（STRACE） (2026-07)
- **简介**：微软（Ying Chang、Jiahang Xu、Yuqing Yang 等）针对基于反思的长程 agent 优化——真实执行轨迹冗余异构、单条轨迹含大量无关步且朴素截断会丢因果证据——提出 STRACE：批级挖掘失败模式过滤冗余、保留代表性失败；轨迹内在文本依赖图上做因果定位剔除非因果步、锁定真正的根因模块。在形式化验证任务 VeruSAGE-Bench 上把人类专家设计 agent 的成功率 42.5%→58.5%（1.4×）。
- **arXiv**：[2607.07702](https://arxiv.org/abs/2607.07702)

#### MetaSkill-Evolve: Recursive Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution（MetaSkill-Evolve） (2026-07)
- **简介**：LMU 慕尼黑（Zefeng Wang、Yunpu Ma 等）提出双时间尺度框架，使 agent 的技能自进化"递归化"：每个分支同时携带任务技能 s 与分支局部元技能 m=(ψ,σ,α,π,ε)，后者参数化改进流水线的 Analyzer/Retriever/Allocator/Proposer/Evolver 五个 agent；任务技能快环演化、元技能用同一流水线自作用于自身慢环演化。五个 agent 共享单一冻结骨干，在 OfficeQA/SealQA/ALFWorld 上分别较原始骨干 +23.54/+16.09/+1.92 分。
- **arXiv**：[2607.05297](https://arxiv.org/abs/2607.05297)

#### ECHO: Prune to act, trace to learn with selective turn memory in agentic RL (2026-06) (ECHO-TurnMemory)
- **简介**：面向有界上下文下的长 horizon 语言智能体，提出选择性轮记忆框架 ECHO：将每个已完成的环境轮压缩为紧凑记忆记录、按需重构有界策略上下文，并复用被选中的来源索引，把正向结果信用回溯到支撑成功答案的证据与选择动作（source-indexed reconstruction），同时解决历史坍缩与可追溯学习两大问题。在 BrowseComp-Plus 上达 43.4% held-out 准确率（GRPO 28.9%、rolling-summary 基线 SUPO 36.1%），且更省轮数与轨迹量，零样本泛化到多目标 QA、代码生成与深度信息检索。
- **arXiv**：[2606.31650](https://arxiv.org/abs/2606.31650)

#### UCOB: Learning to Utilize and Evolve Agentic Skills via Credit-Aware On-Policy Bidirectional Self-Distillation (2026-06) (UCOB)
- **简介**：中科院自动化所等提出 UCOB，针对「检索到的技能记忆并非始终有益、privileged-teacher 假设脆弱」的问题，将技能条件 prompt 与无技能 prompt 视为同一模型的两个在线上下文视图，在相同任务与锚点状态下比较 return-to-go、以更高回报视图作为局部教师，形成本地信用信号来内化有益技能、纠正误导性技能，并指导技能记忆更新、效用感知检索与反思自训练。在 ALFWorld / WebShop / Search-QA 上较 SOTA 基线最高分别 +23.5、+18.0 分。
- **arXiv**：[2606.29502](https://arxiv.org/abs/2606.29502)

#### Multi-Turn Reasoning When Context Arrives in Pieces: Scalable Sharding and Memory-Augmented RL (2026-06)
- **简介**：作者 Shu Tong Luo、Wenqin Liu、Rui Liu、Mingming Gong（墨尔本大学）、Jiaxian Guo（Google Research Australia）。针对"信息分散到多轮才揭示"时 LLM 准确率即便有完整上下文也最高骤降 65% 的 Lost-in-Conversation 退化问题。② 方法：训练模型维护紧凑的滚动记忆（bounded rolling memory）、每轮改写记忆缓冲而非反复 attend 不断增长的历史；为使训练可扩展，提出低成本 sharding 流水线，把单轮 QA 数据集自动转成"多轮碎片化信息"的 episode（仅用 1–3 个 few-shot 示例，免去数小时人工标注），并用多轮 DAPO（RLVR）训练记忆增强策略。③ 仅在 sharded GSM8K 上训练即显著提升多轮准确率，并零样本泛化到更难数学与域外长上下文 QA；更关键的是，记忆训练模型在测试时即便给全历史也优于全历史基线，表明"学会压缩"比单纯暴露全上下文带来更鲁棒的增量推理。
- **arXiv**：[2606.12941](https://arxiv.org/abs/2606.12941)

#### Self-evolving LLM agents with in-distribution Optimization (Q-Evolve) (2026-06)
- **简介**：Yudi Zhang, Meng Fang 等提出 Q-Evolve：用 IQL 风格的加权目标在"专家轨迹+agent 自采轨迹"混合数据上学 in-distribution critic，再用 advantage 推出步级 process reward 做行为接近策略优化，循环自改进。在 ALFWorld / WebShop / ScienceWorld 上以更强样本效率超越基线，证明 process 监督与策略可在共享 in-distribution loop 中协同进化。
- **arXiv**：[2606.07367](https://arxiv.org/abs/2606.07367)

#### AdMem: Advanced Memory for Task-solving Agents (2026-06)
- **简介**：Runzhe Wang 等提出统一的自动记忆框架，融合语义 / 情景 / 程序记忆于双层短期+长期存储，并用 actor-memory-critic 多 agent 架构做自动记忆生成、奖励标注与自适应检索；长期记忆通过基于奖励的合并、剪枝来保证可扩展性。在多种环境上提升长程多轮任务的鲁棒性与成功率。
- **arXiv**：[2606.06787](https://arxiv.org/abs/2606.06787)

#### Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories (2026-06)
- **简介**：Google Research（Ali Behrouz, Vahab Mirrokni 等）提出"睡眠"范式：(1) Memory Consolidation——通过 Knowledge Seeding 把小自我蒸馏到更大网络（结合 on-policy distillation 与 RL imitation 的 Generalized Distillation）；(2) Dreaming——用 RL 生成合成数据课程做无监督自改进。长程持续学习与少样本泛化任务上验证睡眠阶段必要性。
- **arXiv**：[2606.03979](https://arxiv.org/abs/2606.03979)

#### Joint Agent Memory and Exploration Learning via Novelty Signals (JAMEL) (2026-06)
- **简介**：清华 / 微软亚研 Shizuo Tian 等提出 JAMEL：把 agent 记忆与探索策略联合训练。利用 GUI 域的代码覆盖等"确定性持久 novelty 信号"作为记忆模块的免标注监督，让记忆区分已穷尽行为与未知行为，与 novelty-driven 探索互相增强。在未见环境中泛化优于开源基线，token 消耗更低。
- **arXiv**：[2606.01528](https://arxiv.org/abs/2606.01528)

#### SAM: State-Adaptive Memory for Long-Horizon Reasoning Agent (2026-05)
- **简介**：人大 + 北京智源（Zhicheng Dou 组）提出 SAM——standalone 框架，把交互巩固为紧凑 memory cues 同时保留原始 trajectory pages 供 intent-driven recall；cues 不替换历史，而是作为轻量 handle 让 agent 按当前需要重建远距离信息，**无需重训 backbone**。memory 模块用 expert-guided supervision + RL 与 trajectory-level utility 对齐。在 BrowseComp、BrowseComp-ZH、WideSearch、HLE 上一致超越强基线。
- **arXiv**：[2605.24468](https://arxiv.org/abs/2605.24468)

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

#### Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning（SAO） (2026-07)
- **简介**：智谱 + 清华（Zhenyu Hou、Jie Tang、Yuxiao Dong 等）针对异步 RL 在长程 agentic 任务的稳定性与 off-policy 难题，提出 SAO：用单 rollout 采样替代 GRPO 的组内采样以降低 off-policy 效应、配合价值模型训练设计，并引入严格双侧 token 级裁剪稳定优化。可稳定训练千步，在 SWE-Bench Verified、BeyondAIME、IMOAnswerBench 上持续超越 GRPO 及其变体，并成功用于训练开源 GLM-5.2（750B-A40B）的 agentic RL 流水线。
- **arXiv**：[2607.07508](https://arxiv.org/abs/2607.07508)

#### The Rollout Infrastructure Tax in Coding-Agent Reinforcement Learning (2026-07) (Rollout-Infra-Tax)
- **简介**：将 coding-agent RL 的执行基础设施本身作为研究对象，系统比较四种执行基座（单容器、托管沙箱、Kubernetes 编排容器、云虚拟机），发现冷启动延迟最高 110× 差异、百万条 150 步轨迹的预计 worker-hours 有 1.8× 分布，主张未来编码智能体 RL 系统应把执行基座作为训练系统的一部分来优化，而非仅当部署管道。提交至 ACM SoCC 2026。
- **arXiv**：[2607.01415](https://arxiv.org/abs/2607.01415)

#### Steer, Don't Solve: Training Small Critic Models for Large Code Agents (2026-06)
- **简介**：CMU（Shubham Gandhi、Yiqing Xie、Carolyn Rose 等）针对"端到端 code agent 训练资源密集、且在解决 code issue 所需的策略级推理上易停滞（联合优化代码级执行与策略级推理使后者欠发展）"的问题，提出冻结 agent、外挂 critic 提供策略级信号。不同于以往"事后给完整轨迹打分"的 critic，本文用 SFT 训练一个提供 intra-trajectory 反馈、能实时"操舵"的小 critic。在 SWE-bench Verified 上，基于 CWM-32B 轨迹训练的 critic 可迁移到两个未见 agent（+3.0~+3.8 分），加入目标 agent 轨迹后增益升至 +3.8（CWM-32B）/+4.4~+5.2（两个 Qwen agent），critic 成本仅为强教师的 1/30~1/92；在 Qwen3-Next-80B-A3B 上 critic 引导系统既更准（25.2% vs 20.8%）又更便宜（$0.04 vs $0.11，因 critic 同时缩短了轨迹）。
- **arXiv**：[2606.21811](https://arxiv.org/abs/2606.21811)

#### Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills (2026-06)
- **简介**：Chuan Xiao, Linfeng Zhang 等提出 Socratic-SWE：把 agent 历史解题轨迹蒸馏为结构化"agent skills"（总结循环失败模式与有效修复套路），再以这些 skill 在真实仓库中生成定向修复任务，经执行验证 + solver-gradient 对齐奖励筛选。SWE-bench Verified / Lite / Pro / Terminal-Bench 2.0 上一致优于自进化基线，3 轮迭代后 Verified 达 50.40%。
- **arXiv**：[2606.07412](https://arxiv.org/abs/2606.07412)

#### BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models (2026-05, ICML 2026)
- **简介**：作者提出三阶段框架解决 APR 中"稀疏执行反馈 + 粗粒度 sequence reward 难以定位修复关键编辑"的问题：(1) 在 execution-verified demonstrations + reasoning traces 上 SFT；(2) 从执行结果训练**双 reward model**——sequence-level assessor + line-level credit allocator；(3) PPO 优化时 line-level 模型把奖励重分配到关键编辑区域。**line-level 信用分配粒度天然匹配代码 diff**。SWE-bench Verified 40.7% (+22.9pp over 基础)、Defects4J (Python→Java 迁移) 24.8%、HumanEval-Java 84.5%、QuixBugs 95.0%，实现强跨语言泛化。
- **arXiv**：[2605.09134](https://arxiv.org/abs/2605.09134)

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

#### AdaTurn: Budget-Aware Test-Time Scaling for Active Visual Perception Agents (2026-07)
- **简介**：Susan Liang, Chao Huang, Jason J. Corso, Chenliang Xu 等（罗切斯特大学等）。针对主动视觉智能体在部署期 rollout 预算不定、预算小于偏好轨迹时被过早截断（catastrophic truncation）的问题，提出预算感知框架 AdaTurn：将智能体条件于允许轮数并显式训练预算诱导的边界行为，核心 Forced-Answer DAPO（FA-DAPO）把超预算事件从被 mask/惩罚的失败转为可训练的最终决策步；训练与推理均随机化预算并配负载均衡调度器。将 VisualProbe-Medium 在 4 轮下从 36.7% 提升到 47.6%，同时保持大预算下的强扩展性。
- **arXiv**：[2607.14547](https://arxiv.org/abs/2607.14547)

#### SPyCE: Skill-Policy Co-evolution for Multimodal Agents (2026-07)
- **简介**：Ru Zhang, Weijie Qiu。针对「thinking with images」多模态智能体——现有 RL 把轨迹压缩为标量奖励、memory 方法仅测试期检索而不更新策略，提出将推理轨迹蒸馏为可复用技能并在训练中与策略共演化：分层技能库（执行技能捕捉局部视觉操作、工作流技能编码高层先验），策略以检索到的技能指导 rollout，技能库又用高价值 rollout 演化，形成闭环。八个基准上一致超越 RL 与 memory 基线。
- **arXiv**：[2607.13854](https://arxiv.org/abs/2607.13854)

#### ExToken: Structured Exploration for Efficient Vision-Language-Action Reinforcement Fine-tuning (2026-07)
- **简介**：Yilun Kong, Guozheng Ma, Li Shen, Dacheng Tao 等。研究 VLA-RL 的「探索停滞」瓶颈，发现轨迹多样性比 rollout 数量更关键；提出 RL Exploration Token（ExToken）：将 VLA 策略条件于从离线示范导出的离散行为先验以做结构化探索，并用状态条件的 token 选择器桥接训练期探索与部署期确定性推理。在仿真与真实机器人操作上一致加速收敛、提升性能，且在极受限交互预算下鲁棒。
- **arXiv**：[2607.12931](https://arxiv.org/abs/2607.12931)

#### ReGRPO: Reflection-Augmented Policy Optimization for Tool-Using Agents (2026-06) (ReGRPO)
- **简介**：新加坡国立 Show Lab 提出 ReGRPO，面向工具增强的视觉-语言模型（VLM）智能体在工具失败后的恢复问题。先用结构化反思数据引擎：对近似失误动作实际执行以采集真实失败观测，构造 Reflection-of-Thought 三元组（ErrorType, Evidence, FixPlan）配对纠正动作做暖启动 SFT；再在局部轨迹内以组相对优势联合优化反思 token 与纠正动作，并加反思成本项抑制冗余反思。在 GTA 与 GAIA 上一致优于强开源基线。
- **arXiv**：[2606.31392](https://arxiv.org/abs/2606.31392)

#### VideoSEG-O3: A Multi-turn Reinforcement Learning Framework for Reasoning Video Object Segmentation (2026-06)
- **简介**：东南大学 / 百度等（Ming Dai, Jingdong Wang 等）提出首个面向 RVOS 的多轮 RL 框架 VideoSEG-O3：用"由粗到细"多轮时空 CoT 迭代定位关键时段与关键帧；引入 SEG-aware logit calibration 把像素级分割反馈直接注入 token logits；并用解耦的 thinking trace 把推理拆为时间/空间/语言维度，配套冷启数据 VTS-CoT。
- **arXiv**：[2606.06819](https://arxiv.org/abs/2606.06819)

#### Potential-Guided Flow Matching for Vision-Language-Action Policy Improvement (ForesightFlow) (2026-06)
- **简介**：清华 / 中科大等（Yunpeng Mei, Gao Huang, Jie Chen 等）针对 VLA 部署中"成功+部分完成+可恢复错误+失败"混合质量数据，提出自引导 flow-matching 策略 ForesightFlow：同一 flow 同时提议动作并打分（best-of-K 推理免外部 critic）；用解耦 advantage-weighted flow matching 仅对动作速度加权、对 potential 速度均匀训练，避免过自信打分。BEHAVIOR-1K 5 任务 + 真实 5 双臂任务上匹配 separate-critic 强基线，训练算力 -38%。
- **arXiv**：[2606.04968](https://arxiv.org/abs/2606.04968)

#### Entropy Is Not Enough: Unlocking Effective Reinforcement Learning for Visual Reasoning via Vision-Anchored Token Selection (VEPO) (2026-06)
- **简介**：复旦 Senjie Jin、桂韬等指出 RLVR 中"用 token 熵做 credit assignment"在视觉推理中失效——视觉敏感 token 天然低熵被忽略。提出 VEPO：把视觉敏感度与 token 熵以"乘性耦合"融合，把梯度 credit 重新引向"既视觉 grounded 又信息量高"的 token。3B / 7B 上分别比纯熵基线高 +3.15 / +2.28 分。
- **arXiv**：[2606.03937](https://arxiv.org/abs/2606.03937)

#### Towards Precise Intent-Aligned VLA Aerial Navigation via Expert-Guided GRPO (EG-GRPO) (2026-06)
- **简介**：浙大 Tianyang Chen, Fei Gao 等针对无人机连续动作空间的探索难题提出 EG-GRPO：在 online rollouts 中混入少量专家数据并搭配并行仿真-推理异构流水线（rollout 时间 -43.5%）。复杂人意图任务上成功率达到 SFT 基线的 2.13×，意图对齐 +60.9%。
- **arXiv**：[2606.02313](https://arxiv.org/abs/2606.02313)

#### MT-EditFlow: Reinforcement Learning for Multi-Turn Image Editing with Flow Matching (2026-06)
- **简介**：UCLA / Adobe 等（Jiahui Huang, Yasi Zhang, Mingyuan Zhou, Ying Nian Wu 等）面向多轮交互式图像编辑提出 flow-matching RL 框架 MT-EditFlow：多轮视角 + 多奖励统一公式，适用于 GRPO 与 NFT-based RL；系统分析 turn-level 聚合策略、VLM 推理模式权衡偏差/方差、以及 advantage fusion 防止 reward hacking。FLUX.1-Kontext-dev 在第 3 轮整体性能 +6.85 分，超越 Qwen-Image-Edit 等开源 SOTA。
- **arXiv**：[2606.01985](https://arxiv.org/abs/2606.01985)

#### Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning (MoCA) (2026-05, ICML 2026 Spotlight)
- **简介**：Wenhu Chen + 林方真等团队提出**模态信用分配** (Modality-aware Credit Assignment, MoCA) 框架，把 VLM 失败诊断为感知缺陷（"看不清"）vs 推理缺陷（"想错了"），把生成显式分解为交错的 perception/reasoning 步骤。引入 **Perception Verification (PV)**——用"蒙眼推理" agent **独立于推理结果**对感知保真度赋奖；并用结构化算法执行的 verbal verification 替换高方差的 LLM judge 以扩展到自由形式 VL 任务。能在大量任务谱系上同时改善 perception 与 reasoning，缓解二者之间常见的 seesaw effect。
- **arXiv**：[2605.14054](https://arxiv.org/abs/2605.14054)

#### DUEL: Adversarial Self-Play for Multimodal Reasoning (2026-05)
- **简介**：作者提出 DUEL：从同一 pretrained VLM 初始化两个策略，**对抗性自博弈**完全替代昂贵高质量标注。Challenger 生成 image-grounded true claim 与最小扰动 hard-negative 对抗版本；Solver 对图像验证两条 claim，强制细粒度近邻视觉判别。引入 length-normalized log-likelihood reward 在二值 outcome 之外保留信息量，提升稀疏反馈下学习稳定性。**无需人类标注、外部 reward 模型或图像编辑工具**，一致提升视觉推理与判别。
- **arXiv**：[2605.24794](https://arxiv.org/abs/2605.24794)

#### Ranking-Aware Calibration for Reliable Multimodal Reinforcement Learning (RAC) (2026-05)
- **简介**：清华（Peng Cui / Jun Zhu 组）针对 VLM RL 后置信度**校准差**的问题（terminal correctness reward 既不惩罚自信错误也不把置信度与视觉证据质量挂钩），提出 RAC：用 group-based RL 已经产生的两类比较信号——(a) **ranking-aware group loss**（同 prompt 内更好的 rollout 应有更高置信度）、(b) **clean–corrupted pairwise loss**（视觉证据降级时置信度应同步衰减）。两个损失无需外部标注、自然集成进 group-based RL 后训。在 Qwen2.5-VL 与 InternVL-3.5 上的 6 个多模态推理 benchmark（含损坏输入）上一致改善校准与精度。
- **arXiv**：[2605.16999](https://arxiv.org/abs/2605.16999)

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

#### MJ: Multi-turn LLM Jailbreaking via Decomposed Credit Assignment (DC-GRPO) (2026-07)
- **简介**：Junyoung Park, Sangdon Park 等（POSTECH 等）。将多轮越狱攻击者的学习刻画为信用分配问题，提出 DC-GRPO——为 GRPO 的每一轮分配独立的组相对学习信号（结合即时与未来信用），避免把单一轨迹级分数广播到整段对话导致的信用错配；给出静态/动态加权两种实例化。在多个受害 LLM 与基准上，动态/静态变体平均 ASR5@3 达 98.26%/97.88%，显著超越 SEMA（86.58%）与 TROJail（86.23%）。（含有害内容示例警告）
- **arXiv**：[2607.11070](https://arxiv.org/abs/2607.11070)

#### Beyond Attack-Success Rate: Action-Graded Severity Scale for Tool-Using AI Agents (2026-07)
- **简介**：Harry Owiredu-Ashley 提出面向工具使用 agent 红队的动作分级严重性评分：以七级序数尺度（L0–L6）按动作可逆性、是否跨域触达他方、是否扩权对工具调用轨迹打分；用确定性 oracle 与三前沿模型裁判两种方式计算。在 AgentDojo workspace 四受害模型/两防御上，揭示二元攻击成功率掩盖的三类风险（如报告零攻击成功却仍存在跨域泄漏），裁判组与 oracle 序数一致性 α=0.91。
- **arXiv**：[2607.07474](https://arxiv.org/abs/2607.07474)

#### The Blind Curator: How a Biased Judge Silently Disables Skill Retirement in Self-Evolving Agents (2026-07)
- **简介**：Xing Zhang 等给出一项行为安全结果：自进化 agent 依赖"技能退役"防止技能库跌破无技能基线，但该保证假设奖励无偏，而 reference-free 任务下的 LLM 裁判并不满足。通过腐败奖励分析证明——对称噪声无碍，但"假通过"偏置会在一个尖锐阈值后彻底关停基于贡献的技能退役且无法靠增数据跨越；并提出部署前的廉价缺陷注入审计判定裁判所处阈值侧。
- **arXiv**：[2607.07436](https://arxiv.org/abs/2607.07436)

#### Reason Less, Verify More: Deterministic Gates Recover a Silent Policy-Violation Failure Mode in Tool-Using LLM Agents (2026-07)
- **简介**：Vikas Reddy 等在 τ2-bench 航空域揭示工具使用 agent 的"静默错状态"失效——工具在策略允许环境下执行了被域策略禁止的写操作，任务看似成功却造成不可见错误状态（78% 失败属此类）。提出确定性、只读的执行前门控在写操作前检查调用与状态；四门控套件把 gpt-4o-mini 全基准成功率 29.6%→42.0%（+12.4pp），且在 gpt-5.2 上同样把 61.2%→71.6%。
- **arXiv**：[2607.07405](https://arxiv.org/abs/2607.07405)

#### Defending Jailbreak Attacks on Large Language Models via Manifold Trajectory Kinetics (MTK) (2026-06)
- **简介**：HUST 等（Hangtao Zhang, Shengshan Hu 等）把 LLM 视作"把输入变换为输出的动力系统"，跨层追踪 prompt 邻域结构演化：良性始终靠近良性邻域，越狱 prompt 从恶意种子起，后期"策略性"漂移到良性邻域。在 4 个 LLM × 10 种越狱攻击上：伪恶意 prompt TPR 95% @ 5% FPR，自适应攻击下仍保持 85% TPR；VLM 上同样有效。
- **arXiv**：[2606.07335](https://arxiv.org/abs/2606.07335)

#### Safety Paradox: How Enhanced Safety Awareness Leaves LLMs Vulnerable to Posterior Attack (2026-06)
- **简介**：SUTD / NTU 等（Long Hoang, Wenxuan Zhang 等）发现"安全意识越强、越易被反向利用"：提出 Posterior Attack——单次查询让模型直接生成"它自己内部分类器本会判违规的回答"。在 30 个开源 LLM（含 35B）与 GPT-5、Claude 4.6 上验证；并通过 RL 干预证明因果链：人为削弱安全判别可免疫该攻击，反之加剧。形式化证明"对齐单调改进 → 后验脆弱性放大"。
- **arXiv**：[2606.05614](https://arxiv.org/abs/2606.05614)

#### CHASE: Adversarial Red-Blue Teaming for Improving LLM Safety using Reinforcement Learning (2026-06)
- **简介**：UNSW Sydney（Rahul Markasserithodi, Aditya Joshi 等）提出 CHASE 红蓝共进化框架：黑盒攻击者用 GRPO 在"绕过有效性 × 意图保真度"乘性奖励下训练，防御者用两阶段 GRPO + 拒绝采样 SFT 在收割的对抗改写上加固。BeaverTails / JailbreakBench 上对 5 个未见攻击家族（PAIR/TAP/AutoDAN/PAP/Translation）平均 StrongREJECT 降低 43.2%，良性 prompt 误拒率 0%。
- **arXiv**：[2606.05523](https://arxiv.org/abs/2606.05523)

#### NeuroArmor: Safe-Variant-Guided Representation Consistency for Selective Re-Anchoring in Jailbreak Defense (2026-06)
- **简介**：北语 / 字节等（Zhongyang Lin 等）提出白盒运行时防御 NeuroArmor：为每个 prompt 构造 K 个安全变体作为局部安全参考，在隐藏态空间比较异常并路由到拒绝分支或友好恢复分支。Llama-3-8B-Instruct 上把恶意 ASR 从 41.56% 降到 1.57%，同时把良性 FPR 从 30.26% 降到 22.05%。
- **arXiv**：[2606.03486](https://arxiv.org/abs/2606.03486)

#### Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization (2026-05, ICML 2026)
- **简介**：西工大 + UESTC + 中国电信 AI 团队（李学龙组）提出 Metis，把越狱重铸为 adversarial POMDP 内的**inference-time 策略优化**，自进化元认知循环对目标防御逻辑做因果诊断，把结构化反馈当作语义梯度精炼策略。10 个模型平均 ASR 89.2%（O1 76.0%, GPT-5-chat 78.0%），平均 token 成本相比传统方法 8.2× 降低、最高 11.4×。揭示当前防御对内部驱动的闭环推理轨迹仍脆弱。
- **arXiv**：[2605.10067](https://arxiv.org/abs/2605.10067)

#### Disentangling Intent from Role: Adversarial Self-Play for Persona-Invariant Safety Alignment (PIA) (2026-05)
- **简介**：中科院 + NUS 团队针对 persona-based jailbreak 缺少**机理性防御**的问题，提出 Persona-Invariant Alignment (PIA) 对抗自博弈框架。攻方 PLE (Persona Lineage Evolution) 用 lineage-based credit propagation 高效探索高风险 persona 空间；守方 PICL (Persona-Invariant Consistency Learning) 基于结构分离假设、用单边 KL-散度约束**结构性解耦** safety decision 与 persona context。显著降低 ASR 同时保留通用能力，给出可证安全对齐范式。
- **arXiv**：[2605.01899](https://arxiv.org/abs/2605.01899)

#### A Systematic Investigation of The RL-Jailbreaker in LLMs (2026-05)
- **简介**：阿尔伯塔大学等团队首次对 RL jailbreaker 做系统性机制分解：把框架拆成问题形式化（reward 函数、动作空间、episode 长度）与算法措施（RL 算法、训练数据、reward shaping），定位**dense reward + 长 episode** 是 jailbreaking 成功的最主要驱动力。RL-jailbreaker 攻陷所有目标模型与防护栏，为提升 RL-jailbreaker 效率与红队评估提供了工具，也为防御侧硬化指出方向。
- **arXiv**：[2605.07032](https://arxiv.org/abs/2605.07032)

#### Why Do Aligned LLMs Remain Jailbreakable: Refusal-Escape Directions (RED) (2026-05)
- **简介**：中科院计算所 + 国科大团队提出"拒答逃逸方向 (Refusal-Escape Directions, RED)"概念——在保持有害语义的扰动子空间内，仍存在能把模型行为从拒答推向回答的方向。论文形式化 RED，并给出 operator-level 来源分析与 safety-utility trade-off 刻画。为对齐后模型仍可被越狱提供了**结构性解释**，对 RLHF 后的安全性研究具有补全意义。
- **arXiv**：[2605.08878](https://arxiv.org/abs/2605.08878)

#### Redefining AI Red Teaming in the Agentic Era: From Weeks to Hours (Dreadnode) (2026-05)
- **简介**：Dreadnode 团队展示首个 agentic 红队系统：基于开源 Dreadnode SDK，集成 45+ 攻击、450+ transform、130+ scorer，用自然语言目标驱动 agent 自主组合攻击/转换/打分流水线，把传统手工耗时数周的红队流程压缩至小时级。在 Meta Llama Scout 上 zero-code 实现 ASR 85%（severity 1.0），3 小时跑出 232 个 critical findings、674 次攻击、573 个 finding。统一传统对抗样本与生成式越狱评估范式。
- **arXiv**：[2605.04019](https://arxiv.org/abs/2605.04019)

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

#### GFlowRL: Scaling Distribution-Matching RL to Large Language Models（GFlowRL） (2026-07)
- **简介**：Xiaodong Liu、Michael Xu、Paul Smolensky、Doug Burger、Jianfeng Gao 等（Microsoft）。把 GFlowNet 风格的分布匹配 RL 扩展到现代 post-training：发现此前被视为必需的 learned partition function 在规模/horizon/reward 噪声增大时反而成为梯度不稳定与工程负担来源，可用 rollout group 内的 in-batch Monte Carlo 估计替代，从而完全移除辅助 partition network，同时保留 reward 分布匹配目标。关键落点在 §3.1/§3.3：配两个稳定器——**针对 rollout/trainer drift 的 importance-sampling 修正**与**针对离群残差的 asymmetric flow-gap clipping**（IS + 非对称 clip 设计），在 dense 与 MoE（up to 235B）上稳定扩展（14B 达 Codeforces 2048），是首个跨 dense/sparse 稳定扩展的 GFlowNet 风格 RL。
- **arXiv**：[2607.13394](https://arxiv.org/abs/2607.13394)

#### UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma (UP) (2026-07)
- **简介**：Chongyu Fan、Pengfei Liu、Sijia Liu 等。针对 IS-based RL 的"探索-稳定"两难：纯 IS 常致灾难性不稳定，而标准 clipping 又过早截断策略更新预算、抑制探索。作者形式化 Probability Capacity (Cap) 概念，揭示保守 clipping 会结构性扼杀正确但低置信推理路径。提出 Unbounded Positive Asymmetric Optimization (UP)：用 stop-gradient 把策略锚定在当前状态，对正 advantage 释放不裁剪、稳定的梯度以最大化探索，对负 advantage 保留标准 clipping 防止失稳。可扩展到 token 级（GRPO/DAPO）与 sequence 级（GSPO），在多种算法、架构（Dense/MoE/VLM）与模态上均提升推理精度——是本窗口内针对 IS clipping 设计的通用 plug-and-play 改进。
- **arXiv**：[2607.06987](https://arxiv.org/abs/2607.06987)

#### Turning Off-Policy Tokens On-Policy: A Plug-in Approach for Improving LLM Alignment (SIS) (2026-07)
- **简介**：Renmin University of China + JD.com（Yu Li 等）。针对 "rollout then update" 造成的 off-policy 训练数据，提出 Selective Importance Sampling (SIS)：把 off-policy 模型视为提议分布，做 token 级拒绝检验——被接受的 token 视作 on-policy 并赋单位重要性分数，被拒绝的 token 保留标准 IS 校正。理论上证明可缩小 token-level 与 sequence-level off-policy 梯度估计器之间的 gap；作为 plug-in 仅改动 policy loss 中的 importance ratio，几乎不增加 wall-clock 开销，可与多种 RL 后训练算法组合。在 dense 与 MoE LLM 的 math / agent benchmark 上一致提升，且在 off-policy 数据下鲁棒性显著更强。
- **arXiv**：[2607.04728](https://arxiv.org/abs/2607.04728)

#### Experience Augmented Policy Optimization for LLM Reasoning (EAPO) (2026-06)
- **简介**：Jinda Lu、Kexin Huang、Xiang Wang、Guoyin Wang、Jingren Zhou 等针对 RLVR 通常从零 on-policy 优化、采样成本高且难以复用历史经验的问题；而近期以**固定推理轨迹**复用经验的做法又因策略演化产生 **policy mismatch**。本文主张经验不应作为固定轨迹复用，而应以**策略自适应**方式表达：提出 **EAPO**，将一个先前 RL 优化过的策略作为 **action-level 经验先验**，在 rollout 的关键决策点处选择性注入经验；并配以**改造的重要性采样（adapted importance sampling）**方案，确保从经验增强 rollout 中稳定、无偏地学习。在 Qwen-2.5-math-7B 与 Qwen-3-8B、五个 benchmark 上一致超越 SOTA RLVR 方法。属经验复用 + off-policy IS 修正的算法设计。
- **arXiv**：[2606.30420](https://arxiv.org/abs/2606.30420)

#### What are Key Factors for Updates in RL for LLM Reasoning? (ACPO) (2026-06)
- **简介**：微软研究院（Peidong Wang、Xufang Luo、Dongsheng Li 等）对 RLVR 更新做理论分析，揭示"每次 rollout 的梯度步数所决定的 off-policy 程度"会显著改变重要性采样比率分布及其裁剪行为，从而改变"哪些 token 主导更新"。将梯度期望刻画为支配更新动态的核心量，分析 token 概率、优势、IS 比率的作用，据此提出 **Adaptive Clip Policy Optimization (ACPO)**——按各 token 组 IS 比率的经验方差自适应调整裁剪边界。在 3B/7B 模型、数学/表格 QA/逻辑谜题上优于 DAPO、CISPO 等强基线。
- **arXiv**：[2606.22570](https://arxiv.org/abs/2606.22570)

#### Rethinking the Divergence Regularization in LLM RL (DRPO) (2026-06)
- **简介**：Jiarui Yao、Xiangxin Zhou、Penghui Qi、Wee Sun Lee、Liefeng Bo、Tianyu Pang 等提出。指出 LLM RL 实践中因训练-推理不匹配与 policy staleness 而**天然 off-policy**，故信任域控制至关重要；而 PPO/GRPO 的 ratio-clipping 在长尾词表上是分布漂移的差代理。承接 **DPPO**（用基于 divergence 的 mask 替代 ratio-clip，以采样 token 的绝对概率漂移定义信任域）但指出其仍是**硬 mask**——token 一旦越界即丢弃梯度而非修正。提出 **DRPO（Divergence Regularized Policy Optimization）**：用**平滑的、优势加权的二次正则化器**替代硬 mask，保持与 DPPO 相同的信任域几何，同时产生**有界、连续的梯度权重**，对越界更新衰减并提供边界外的修正信号。在不同模型规模、架构与精度设置下提升 LLM RL 的训练稳定性与效率，是 §3.1 中 ratio-clip → divergence-mask → 平滑正则 这一演进线的最新一环。
- **arXiv**：[2606.09821](https://arxiv.org/abs/2606.09821)

#### How Off-Policy Can GRPO Be? Mu-GRPO for Efficient LLM Reinforcement Learning (Mu-GRPO) (2026-05)
- **简介**：Minghao Tian、Yunfei Xie、Chen Wei 等人提出 μ-GRPO，将 RL 训练拆成少量大型「生成-优化」阶段，故意拉高 rollout staleness（μ 可达 1024）以摊薄切换开销。诊断出标准 GRPO 在高 staleness 下的「clipping 困境」——紧 clip 抑制梯度、宽 clip 触发崩溃，并定位崩溃根源在 negative-advantage 触发后的 suffix tokens。提出 **relaxed clipping + negative-advantage veto**，在 5 个模型 / 多个数学基准上匹配或超过 GRPO，墙钟训练时间约 2× 加速。属于 OPD §3.1 序列级 IS / clip 设计的最新代表。
- **arXiv**：[2605.17570](https://arxiv.org/abs/2605.17570)

#### Hölder Policy Optimisation: Generalising Probability Aggregation for LLM Policy Optimisation (HölderPO) (2026-05)
- **简介**：Yuxiang Chen、Dingli Liang 等（UCL + SJTU + HKUST-GZ）。指出 GRPO 把 token 级概率聚合成序列级 IS 的方式被锁死成算术 / 几何均值（PPO / GSPO 各占一端），是限制算法适应性的隐藏瓶颈。用 **Hölder mean** 统一概率聚合，引入显式参数 p 在「梯度集中度」与「方差界限」间连续插值，并配套 dynamic annealing 在训练全程调度 p。在 5 个数学 benchmark 平均 54.9%（相对 GRPO 提升 7.2%），ALFWorld agentic 任务 93.8% 成功率，为 GSPO / GMPO 之后的 IS 聚合设计提供了连续谱视角。
- **arXiv**：[2605.12058](https://arxiv.org/abs/2605.12058)

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

#### Staleness-Learning Rate Scaling Laws for Asynchronous RLHF (2026-07)
- **简介**：Jingwei Song、Weixun Wang、Chuan Wu、Linfeng Zhang 等针对高吞吐 RLHF 系统解耦 rollout 生成与策略优化、导致 learner 更新使用**陈旧 rollout（stale rollouts）**的问题，系统研究异步 GRPO 中陈旧度的影响。将行为策略显式写入 GRPO 代理目标，区分 learner 所用的 surrogate-gradient 映射与分布依赖 population 目标的真实全导数；在局部有界、分布光滑、行为策略光滑等假设下，证明陈旧 rollout 引入 **O(S·η)** 量级的 per-step 代理梯度偏差（S 为最大 rollout 滞后、η 为学习率）。进一步推导**条件式 collapse-time 缩放律**：cycle 内漂移低于 batch 级 clipping 半径时崩溃主要由累积 learner 漂移 T·η 主导，陈旧约束激活时稳定性显式依赖 S·η；由此给出双约束稳定条件 η≪min{R_batch/(S·G_upd), R_crit/(T·G_upd)}，解释了在 horizon-limited 区间最大稳定学习率为何看似弱依赖陈旧度。属异步 off-policy 系统的**陈旧度-学习率理论/缩放律**新工作。
- **arXiv**：[2607.01083](https://arxiv.org/abs/2607.01083)

#### ASymPO: Asymmetric-Scale Policy Optimization for Asynchronous LLM Post-Training Without Behavior Information (ASymPO) (2026-06)
- **简介**：华为诺亚方舟团队针对**异步 RL** 场景下 stale rollout 引入 distribution drift 的问题，质疑标准做法（behavior log-prob、IS ratio、clipping）所要求的 token-aligned/版本化/数值一致的 rollout-learner 接口是否必要。识别出新失效模式 **scale-imbalance**：stale 响应在当前策略下评估，正/负 loss 项 NLL 尺度不同导致 zero-sum advantage 不再等价于平衡 loss 贡献。提出 **ASymPO**——只用当前策略概率，按响应当前平均 token NLL 归一化 token loss，无需任何 behavior 信息即恢复 response-level zero-sum 平衡；同时给出 fixed negative-scaling baseline **SPO**。在异步数学推理 post-training 上与 GRPO 持平甚至更优，证明 behavior 信息并非异步 RL 的必需品。是 §3.2 系统级 Off-Policy 中"去 behavior"路线的代表。
- **arXiv**：[2606.03070](https://arxiv.org/abs/2606.03070)

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

#### The Mirage of Optimizing Training Policies: Monotonic Inference Policies as the Real Objective for LLM Reinforcement Learning (MIPI/MIPU) (2026-06)
- **简介**：Jing Liang、Hongyao Tang、Jianye Hao、Bo Zheng 等（含淘天/阿里团队）针对 LLM RL 训练脆弱、易不稳定甚至崩溃的核心成因之一——**训练-推理不匹配（training-inference mismatch）**：推理引擎（追求生成效率）与训练引擎（追求梯度精度）对同一轨迹给出不一致概率，即便模型参数已同步，也会诱发一种「始终存在且毒害训练」的特殊 off-policyness。本文指出既有工作忽视的**目标错位**：对训练引擎内策略的有效更新，未必带来部署所用**推理策略**的改进。据此提出 **Monotonic Inference Policy Improvement (MIPI)** 原则，并给出两步框架 **MIPU**：构造 sampler-referenced 候选更新，再以 inference-side gap proxy 选择性接受同步候选。在两种模型规模、高失配设定下均提升平均推理性能与训练稳定性。属 TIM/off-policyness 的目标层面新工作。
- **arXiv**：[2606.29526](https://arxiv.org/abs/2606.29526)

#### ReLibra: Routing-Replay-Guided Load Balancing for MoE Training in Reinforcement Learning (2026-05)
- **简介**：Chao Jin、Xinming Wei 等（北大 + 字节）。承接 R3 Routing Replay 的思想但目标不同：R3 用 routing replay 修「训练-推理 router 不一致」，ReLibra 用同一信息修「MoE 训练负载不均」——RL rollout 与训练过程使用同一组 token 与同一份 MoE 参数，因此 token-to-expert 的路由决策在训练前已知。基于此设计 inter-batch expert reordering（跨节点）+ intra-batch expert replication（节点内），匹配分层网络带宽。相对 Megatron-LM 提升 1.6× 吞吐，相对给定 oracle load 的 EPLB 仍提升 1.2×。是 MoE RL 系统侧 routing replay 的新一支线，与 R3 形成路由-replay 的「算法 / 系统」双解。
- **arXiv**：[2605.08639](https://arxiv.org/abs/2605.08639)

#### AIS: Adaptive Importance Sampling for Quantized RL (2026-05)
- **简介**：Jiajun Zhou、Wei Shao 等（HKU）。专门针对「FP8 rollout + BF16 trainer」量化 RL 的非平稳偏置：训练早期，rollout-trainer mismatch 实际提供了一种隐式探索 bonus；但当策略集中后同样的扰动转为破坏性偏置。提出 AIS：组合 weight reliability、divergence severity、variance amplification 三个实时诊断指标为单一混合系数，自适应在「无修正」与「完全 IS 修正」之间插值。在 LLaDA-8B / Qwen3-8B / Qwen3.5-9B 上集成进 GRPO，匹配 BF16 baseline 同时保留 1.5–2.76× FP8 rollout 加速。是 FP16 Alignment / FP8-RL 系列后第一个把 quantized-RL mismatch 当「双刃剑」处理的工作。
- **arXiv**：[2605.13907](https://arxiv.org/abs/2605.13907)

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

#### Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations（Demystifying OPD） (2026-07)
- **简介**：Rui Wang、Hongru Wang、Tianqing Fang、Wenhao Yu、Kam-Fai Wong 等（含 CUHK / Tencent 方向作者）。系统研究 on-policy distillation（OPD）的训练动力学，将 OPD 定位为"探索催化剂"——通过 dense token 级引导把学生导向正确推理路径，但不扩展能力上限，且效果完全取决于引导信号质量。揭示两类破坏探索的病理：**Student-Teacher Mismatch**（师生分布差距过大导致引导信号与任务正确性错位）与 **Length Exploitation**（聚合 token 级目标制造长度依赖捷径，学生靠截断/冗余 padding 刷分）。提出两项轻量信号调节——advantage clipping 与 log-scale compression，在七个 benchmark 上稳定超越 OPD 变体与 RLVR 基线，证明"信号质量的良好调节"而非"教师规模"才主导 OPD 成功。属 On-Policy Distillation 的机制/病理分析。
- **arXiv**：[2607.13399](https://arxiv.org/abs/2607.13399)

#### ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation（ShortOPD） (2026-07)
- **简介**：Qingyu Zhang、Hongyu Lin、Yaojie Lu、Xianpei Han、Le Sun 等（中科院软件所方向）。针对结构化剪枝后 LLM 在自由生成任务上崩溃的问题：剪枝后 greedy pass@1 几乎归零但 pass@k 在重复采样下大幅恢复（有用生成被"降权"而非抹除），且可恢复区间主要因 suffix repetition 失败。用 On-Policy Distillation（以剪枝前模型作 frozen teacher）在压缩模型自身 on-policy 状态上做 dense token 级监督恢复；但长 rollout 会把早期预算浪费在低信息的重复后缀上。提出 **ShortOPD** short-to-long 调度：检测 teacher 确认的重复后缀、把存活前缀视作有效长度、按当前可用有效长度分配 rollout 预算。在数学/代码/开放生成上把压缩模型分数提到未恢复值约 9×、标准恢复配方 1.6–4.4×，并以 1/4 训练时间、少 71% rollout token 匹配固定 8192-token horizon。属 §3.4 中面向压缩恢复的 OPD 变体。
- **arXiv**：[2607.13124](https://arxiv.org/abs/2607.13124)

#### EasyOPD: An Easy-to-use On-Policy Distillation Framework for Large Language Models（EasyOPD） (2026-07)
- **简介**：Jie Sun、Mao Zheng、Mingyang Song、Pengfei Liu、Xiang Wang 等（腾讯方向团队）。针对现有 OPD 方法在监督形式、tokenizer 兼容性、teacher 访问方式、监督粒度上差异巨大、实现碎片化难复现的问题，提出基于 verl 分布式 RL 框架构建的 **EasyOPD**：将用户侧配置、方法特定的监督逻辑、verl 执行三者解耦，方法模块通过 loss 构造、rollout 元数据、reward 处理、tokenizer 对齐、teacher 侧计算等扩展边界接入共享后端。实例化三类 OPD 设定——cross-tokenizer OPD、on-policy self-distillation、step-wise OPD，在推理/代码/科学知识/工具使用基准上验证同一后端可运行且保留各方法目标。提供可运行 YAML、文档与安装包。是 §3.4 的**工程框架/基础设施**代表。
- **arXiv**：[2607.11012](https://arxiv.org/abs/2607.11012)

#### Behavior Leverage Imbalance in Multi-Teacher On-Policy Distillation (Soft Clamp) (2026-07)
- **简介**：Jiabin Shen、Guang Chen、Chengjun Mao。研究多教师 on-policy distillation 中的行为漂移：在双教师工具使用场景下，vanilla GKD 虽提升 tool-call recall，却把模型推向 "over-calling"（在本应直接回答的样本上误调工具），而这种漂移从聚合 loss 上不可见。作者提出 behavior leverage imbalance 分析——`<tool_call>`、函数名等 mode-entry / 结构位置的局部 token 级信号，会对全局生成模式产生不成比例的控制力。提出 Soft Clamp：per-token divergence 校准，动态压缩极端 token 级 JS 散度同时保留非零梯度。在 APIGen-MT 上把 over-calling 从 13.7% 降到 9.0%（相对 vanilla GKD）且保持决策精度，在 BFCL 多轮诊断上也减少 tool-call loops——提示多教师 OPD 应监控教师信号"作用在哪"而非只看"聚合幅度"，属 policy/behavior drift 监控视角。
- **arXiv**：[2607.07050](https://arxiv.org/abs/2607.07050)

#### Multi-Turn On-Policy Distillation with Prefix Replay (ReOPD) (2026-07)
- **简介**：Baohao Liao、Hanze Dong、Li Dong、Furu Wei 等。研究 agentic 任务的 on-policy distillation：全在线 OPD 因每步都需学生重新 rollout + 教师查询而昂贵。提出 Replayed-Prefix On-Policy Distillation (ReOPD)——用预采集的教师轨迹作为"回放前缀"，学生只在选定步动作、教师提供密集 per-step 监督而无需真实环境交互。作者指出多轮 OPD 存在 "prefix trap"：让历史更贴近学生 on-policy 会提升相关性，却可能在教师目标不可靠的历史上查询教师，形成学生占据与教师可靠性之间的双向分布漂移；ReOPD 用 step-decaying 采样调度（偏重早期、低漂移前缀）来缓解。在数学推理（Python）与搜索环境、多种师生规模上保持或提升 OPD 精度，学生训练零工具调用，每步至少快 4×。
- **arXiv**：[2607.04763](https://arxiv.org/abs/2607.04763)

#### AsyncOPD: How Stale Can On-Policy Distillation Be? (AsyncOPD) (2026-06)
- **简介**：Wonjun Kang、Kevin Galim、Seunghyuk Oh、Minjun Kang、Sanghyun Park 等（FuriosaAI + UW-Madison）。首个系统研究**异步 On-Policy Distillation 中的 staleness** 的工作，把异步 RL 的 off-policy 问题搬到 OPD 场景：rollout 主导 reasoning 训练时长，异步管道解耦 rollout 与 learner 可缓解瓶颈但引入 stale-policy 数据；且在 teacher 反馈通过 local KL loss 实现、full-vocab teacher logits 太贵需有限 teacher-score cache 的现实设定下。三大发现：① **KL 方向决定 staleness 鲁棒性**——teacher-weighted forward KL 对 stale rollout 更鲁棒，student-weighted reverse KL 脆弱；② 对脆弱的 reverse-KL，异步 RL 的稳定化技巧（如 PPO 式 clipping）反而不如一个更简单的 OPD 专用 surrogate——在 learner 时用当前 student 重算 reverse-KL 信号（recompute Aθ、不 clip）；③ 有限 teacher-score cache 造成 bias-variance 权衡，引出 multi-sample Monte Carlo 估计器（保留 MC 可纠偏性、降低单样本方差）。最终开源 **AsyncOPD** 全异步训练管道，吞吐相对严格同步训练提升 1.6×–3.8× 且精度相当。横跨 §3.2（异步系统级 Off-Policy）与 §3.4（On-Policy Distillation），是把"异步/staleness 治理"正式引入 OPD 的标志性新作。
- **arXiv**：[2606.24143](https://arxiv.org/abs/2606.24143)

#### SG-OPD: Sign-Gated On-Policy Distillation via Sign-Consistency Gating and Phased Teacher Sampling (SG-OPD) (2026-06)
- **简介**：Haoran Xu、Hongyu Wang、Yifei Gao、Jiaze Li、Xiaofeng Zhang、Xiaosong Yuan 提出。指出 On-Policy Distillation 的有效性隐含依赖两个常被打破的假设——轨迹级 student/teacher 对齐、token 级教师可靠性一致。引入一个**二元验证器（binary verifier）**作为对教师的信任信号，在两个互补粒度上工作：① **Phased Teacher Sampling**——冷启动阶段混入经验证器认可的教师 rollout；② **Sign-Consistency Gate**——在教师与"验证器正确方向"一致的 token 上**外推**蒸馏更新、在不一致的 token 上**内插**。在竞赛级数学推理基准上持续优于标准 OPD（per-sample +1.98、per-question +7.50）。是 TML/Reverse-KL OPD 路线上引入"验证器信任门控"的新代表。
- **arXiv**：[2606.09304](https://arxiv.org/abs/2606.09304)

#### Stabilizing On-Policy Distillation for MLLM Reasoning with Global Normalization (GNDPO) (2026-06)
- **简介**：Dongze Hao、Zhiwei Jin、Chen Chen、Haonan Lu 提出。面向**多模态 LLM（MLLM）推理**的 On-Policy Distillation：相比依赖稀疏 binary/outcome 反馈的 RLVR，OPD 用更强教师提供稠密、细粒度的 per-token 监督；但朴素 token 级蒸馏在 outlier 状态下因 magnitude 失配而出现**梯度不稳定（gradient explosion）**。提出 **Globally Normalized Distillation Policy Optimization (GNDPO)**：把原始 per-token KL 分数变换为**batch 级相对优势（batch-level relative advantage）**，在保留 token 级指导收益的同时抑制梯度爆炸。在多模态推理任务上显著提升训练鲁棒性与下游性能。是 §3.4 中把 OPD 稳定化（全局归一化）推广到 MLLM 的新工作，与 Uni-OPD 的多模态扩展形成互补。
- **arXiv**：[2606.09091](https://arxiv.org/abs/2606.09091)

#### Teaching the Way, Not the Answer: Privileged Tutoring Distillation Policy Optimization (PTD-PO) (2026-06)
- **简介**：小米 AI / 西工大团队针对 RLVR 在多模态 LVLM 失败 rollout 上稀疏监督的问题，提出 PTD-PO：用空间注意力与中间推理步骤构造结构化"特权提示"，通过 in-context learning 生成 step-wise token 分布监督，学生仍在 answer-free 上下文优化、对失败 rollout 与提示增强 reference 模型做 token 级对齐；并用 **Top-K JSD** 把对齐聚焦于信息量大的 token 概率上以稳定 distillation。在 2B–8B LVLM 上一致优于 RLVR 与蒸馏基线，**显著缓解 entropy collapse**——既可视为 On-Policy Distill，也可作为 Off-Policy KD 失败 rollout 修复手段。
- **arXiv**：[2606.07000](https://arxiv.org/abs/2606.07000)

#### OPRD: On-Policy Representation Distillation (OPRD) (2026-06)
- **简介**：浙大 / 蚂蚁联合团队提出 OPRD，把 OPD 监督从输出空间提升至**隐藏状态空间**——在学生自采样的 rollout 上跨多层对齐学生与教师的表示，绕过 LM head 与大词表（Qwen ~150k）KL 蒙特卡洛估计带来的方差，理论上消除 sampling variance。AIME 2024/2025 与 AIMO 上闭合 student-teacher gap，相比 top-k OPD 训练快 1.44×、显存少 54%，是当前 §3.4 OPD 系内最深一层结构信号的代表。
- **arXiv**：[2606.06021](https://arxiv.org/abs/2606.06021)

#### Physics-Guided Policy Optimization with Self-Distillation (PGPO) (2026-06)
- **简介**：把 SDPO（self-distilled policy optimization）的不稳定性归因于"step size 对 self-teacher 反馈的盲目信任"，从粘性流体动力学（viscous-fluid dynamics）的 SDE 视角形式化此类比，提出 **PGPO**：用学生预测与 feedback-conditioned teacher 之间的互信息估计，导出 **information-modulated step-size multiplier**。理论上保留 vanilla SGD 的 order-1 weak-approximation guarantee，几乎零额外开销。在 Science-QA 上 4 个 domain 中 3 个超过 SDPO 最多 +4.5 分，且在 SDPO 训练后期崩塌的设置下保持稳定。是 self-OPD 自蒸馏稳定性的另一条新路径。
- **arXiv**：[2606.03620](https://arxiv.org/abs/2606.03620)

#### Teacher-Guided Policy Optimization for On-Policy Reasoning Distillation under Large Policy Divergence (TGPO) (2026-05)
- **简介**：Xinyu Liu、Kechen Jiao 等（东北大学 / Meituan / Meta）。指出 reverse-KL 路线（即 TML 类 OPD）在 student / teacher 分布严重偏离时退化为「无信息负反馈」，标准 RKL 无法继续学习。提出 TGPO：在 student rollout 上额外消费「teacher 在 student 前缀条件下的预测」作为稠密方向性指导，仍保持严格 on-policy，可即插式接入 RLVR 框架。在复杂推理基准上显著超过 RKL / GRPO 基线并对教师选型鲁棒。是 TML On-Policy Distillation 之后修正 RKL 失效模式的关键续作。
- **arXiv**：[2605.13230](https://arxiv.org/abs/2605.13230)

#### KL for a KL: On-Policy Distillation with Control Variate Baseline (vOPD) (2026-05)
- **简介**：Minjae Oh、Sangjun Song、Gyubin Choi、Yunho Choi、Yohan Jo（首尔大学）将 OPD 视为 policy-gradient RL，针对其「单样本 MC 估计高方差」问题，引入 RL 经典的控制变量基线。关键观察：OPD 的最优 value function 恰好是 student-teacher 之间的 per-token negative reverse-KL，可直接从已算好的前向 pass 中无成本读出，无需额外 critic 或推理。在数学 / 科学推理基准上稳定优于 vanilla OPD，并匹配最贵的 full-vocabulary 基线。是给 TML/Reverse-KL OPD 加方差缩减的标准 RL 工具化。
- **arXiv**：[2605.07865](https://arxiv.org/abs/2605.07865)

#### Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe (2026-05)
- **简介**：Wenjin Hou、Shangpin Peng 等（腾讯 + ZJU + HIT）。提出第一个统一覆盖 LLM 与 MLLM 的 OPD 框架，识别两大瓶颈：(1) student 探索信息状态不足、(2) teacher 在 student rollout 上的 token 级监督不可靠。设计「offline difficulty-aware + online correctness-aware」双数据平衡策略，加上 outcome-guided margin calibration，让 token 级监督与 outcome reward 保序一致。覆盖 5 个领域 16 个 benchmark（含单 / 多教师、强→弱、跨模态），是把 TML 类 OPD 推广到多模态场景的最新代表。
- **arXiv**：[2605.03677](https://arxiv.org/abs/2605.03677)

#### Rebellious Student: Reversing Teacher Signals for Reasoning Exploration with Self-Distilled RLVR (RLRT) (2026-05)
- **简介**：Jeonghye Kim、Jiwon Jeon、Dongsheng Li、Yuqing Yang。在 self-distillation 设定下（同一模型、teacher 拥有特权信息）发现：当 student 在 teacher 不会预测的路径上仍能成功时，这些 token 反映其自驱动推理；但标准 self-distillation 会把它们覆盖掉。提出 RLRT：把 self-distillation 信号「反着读」——在正确 rollout 上对这些被低估的 token 强化，等价于 RLVR 中一种基于学生自身成功的探索新形式。在 Qwen3 base / instruct / thinking checkpoint 上显著超过 self-distillation 与探索基线，为 RLVR 引入「信息不对称」这一新设计轴。
- **arXiv**：[2605.10781](https://arxiv.org/abs/2605.10781)

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

#### Building Multi-Task Agentic LLMs via Two-Phase Distillation (2026-06)
- **简介**：Huaijie Wang、Shusheng Xu、Yi Wu、Kaifeng Lyu 研究如何构建多任务模型：先为各任务单独训练 RL 专家、再经蒸馏整合（对照直接在混合任务上训练单模型）。核心对照发现：**off-policy distillation** 在多任务下会退化——forward KL 的 **mode-covering** 特性使聚合多任务数据引入大量行为模式、超出学生容量，迫使其在行为间平均而性能下降；**on-policy distillation** 则是 **mode-seeking**，但需要强初始化。据此提出**两阶段**方案：先 off-policy distillation、再 on-policy refinement。在对话智能体与文本游戏上，该两阶段方法可逐任务匹配单任务 RL 专家性能，而单独用 off-policy 或 on-policy distillation 均无法达到。本文是**Off-Policy KD 与 On-Policy Distillation 的直接机制对照**代表作（mode-covering vs mode-seeking）。
- **arXiv**：[2606.30044](https://arxiv.org/abs/2606.30044)

#### CKA-QAD: Beyond Output Matching — Preserving Internal Geometry in NVFP4 LLM Distillation (CKA-QAD) (2026-06)
- **简介**：把 RL post-trained 模型在 **NVFP4 量化感知蒸馏 (QAD)** 中的退化作为 Off-Policy KD 对照案例研究。诊断发现仅靠 KL 输出匹配会掩盖**内部表示 drift**——在 RL-post-trained 模型上尤其严重，layerwise CKA 对 BF16 teacher 的相似度显著下降，并与下游 reasoning/coding 上瓶颈相关。提出 **CKA-QAD**：通过 CKA 对齐 layerwise Gram 矩阵的轻量正则化，作为 output matching 的补充。在 Nemotron 3 Nano 与 Qwen3-4B-Thinking-2507 上恢复表示对齐并改善下游精度。把"低比特部署 + RL post-train"组合下的 KD drift 量化清楚，是 §3.5 Off-Policy KD 的重要新工作。
- **arXiv**：[2606.05682](https://arxiv.org/abs/2606.05682)

#### Decoupling KL and Trajectories: A Unified Perspective for SFT, DAgger, Offline RL, and OPD in LLM Distillation (Decoupled-Distill) (2026-05)
- **简介**：Anhao Zhao、Haoran Xin 等（EIT-NLP）。指出 off-policy distillation 与 OPD 隐含耦合了两个正交维度——「prefix source」（teacher / student）与「token-level KL 方向」（forward / reverse）。解耦后得到 4 个有效目标，分别对应 SFT-like cross-entropy、DAgger-style on-policy SFT、offline-RL distillation、OPD。在数学推理上系统比较，揭示三组 trade-off：KL 方向 → 准确率/熵；prefix → 质量/算力；训练长度 → 准确率/稳定性。提出 KL mixing 与 entropy-gated length curriculum，把 Avg@k / Pass@k 提升 3.6–5.8 点、平均长度降 ~3×。是 R1-Distill / TML OPD / SFT 三家路线的统一对照框架。
- **arXiv**：[2605.16826](https://arxiv.org/abs/2605.16826)

#### Distillation Scaling Laws (2025-02)
- **简介**：Apple 提出。估算蒸馏模型性能 = f(总算力, teacher/student 算力分配)；给出运行准则——已有 teacher 或要蒸多个 student 时 KD 优于 supervised pretraining。是 KD scaling law 的开山工作。ICML 2025。
- **arXiv**：[2502.08606](https://arxiv.org/abs/2502.08606)

#### DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (R1-Distill) (2025-01)
- **简介**：用 R1 生成 80 万样本对 Qwen2.5-Math 1.5B–32B、Llama-3.1-8B/70B 做纯 SFT 蒸馏；AIME'24 1.5B 28.9 / 7B 55.5 / 32B 72.6。证明对小模型 off-policy distillation 显著优于直接 RL，与 TML On-Policy Distillation 形成路线对照。
- **arXiv**：[2501.12948](https://arxiv.org/abs/2501.12948)

### 3.6 Policy Drift 监控与缓解

#### Understanding Diversity Collapse in RLVR via the Lens of Overtraining (BBG) (2026-06)
- **简介**：Suqin Yuan、Jinkun Chen、Jiyang Zheng、Muyang Li、Lei Feng、Dadong Wang、Tao Xiang、Tongliang Liu、Bo An 提出。从**过训练（overtraining）**视角形式化 RLVR 的"多样性坍缩"（Pass@1 升、高 k Pass@k 退化）：当某问题对边界指标的贡献饱和后，进一步更新不再扩展可解问题集，而只是把概率质量集中到 on-policy 偏好的轨迹上——因此标准 RLVR 的多数更新从边界视角看都属过训练。给出诊断（聚合 Pass@k 下降并不等于无新能力）与干预实验（仅在"零观测成功"问题上更新可把 Pass@256 提升至超过基础模型），并提出 **Bayesian Boundary Gating (BBG)**：估计每个问题对推理边界的边际贡献、把优化引导远离过训练，在多个推理基准上跨广 k 值提升平均 Pass@k。是继 entropy collapse 系列之后，从"过训练/边界饱和"角度监控并缓解 policy drift / diversity collapse 的新视角。
- **arXiv**：[2606.15455](https://arxiv.org/abs/2606.15455)

#### Understanding and Preventing Entropy Collapse in RLVR with On-Policy Entropy Flow Optimization (OPEFO) (2026-05)
- **简介**：Huimin Xu、Shuai Zhao、Xiaobao Wu、Anh Tuan Luu（NTU）从 token 级 entropy flow 视角统一解释 GRPO 系熵塌缩：熵下降 token 长期压过熵上升 token。提出严格 on-policy 的 OPEFO，按各 token 对熵变的贡献自适应重加权熵增 / 熵减更新，避免比率裁剪 / 粗粒度熵正则的局限。在 6 个数学推理基准上系统优于 GRPO/DAPO 等基线，是 entropy collapse 诊断 + 修复方向继 The Entropy Mechanism of RL 之后最系统的新工作。
- **arXiv**：[2605.11491](https://arxiv.org/abs/2605.11491)

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

#### Who Grades the Grader? Co-Evolving Evaluation Metrics and Skills for Self-Improving LLM Agents (Double Ratchet) (2026-07)
- **简介**：Zhang 等指出自进化智能体的技能循环隐含「已存在可靠评估指标」的假设，而现实常不成立。提出可进化的指标循环（在完整进化生命周期下搜索小型缺陷检测器组合，锚定 10 项参考集、以无标注输出的一致性正则化、并用留出锚点审计），及 Double Ratchet——指标与技能循环的协同进化。在 MBPP+、Spider 2.0-Snow、无参考报告生成上，可保留由真值/最佳 rubric 驱动的留出增益的 88–110%，并展示锚点纪律+外部审计带来的安全性。
- **arXiv**：[2607.12790](https://arxiv.org/abs/2607.12790)

#### Game Theory Driven Multi-Agent Framework Mitigates Language Model Hallucination (G-Frame) (2026-07)
- **简介**：Runzhe Liu、Biquan Bie、Zihao Wang、Yuchao Ma、Yexin Liu、Xinghai Li、Harry Yang、Wenbo Yang、Jinzhe Cao、Shengyang Tao 提出 **G-Frame**——融合贝叶斯博弈与团队博弈原理的自适应多 agent 框架，为"轻量 LLM 在规则型科学领域易幻觉"建立"高质量数据合成 + 模型训练"的自动化闭环。通过结构化推理强制内化领域约束，合成了 363,045 条思维链与 199,589 组问答；由此训练的 7B 模型 OmniChem 在自建基准与 ChemBench 上与 GPT-4o mini 持平，且相对 base 幻觉减少 79.46%，并在分子设计与合成规划上展现进阶能力。定位 §4.1 多 agent 数据合成协同训练 / §4.4 博弈论驱动交叉。
- **arXiv**：[2607.08403](https://arxiv.org/abs/2607.08403)

#### Compete Then Collaborate: Frontier AI Teachers Build a Verifiable Curriculum to Improve a Coding Student Beyond Imitation (2026-07)
- **简介**：Miseong Shawn Kim 提出"先竞争后协作"框架：四个前沿 AI 教师（Claude、Codex-GPT、Grok、Gemini）先由**基于执行的裁判**（单元测试、stdin-stdout 校验，带公平性控制）头对头排名，再协作为学生（Qwen2.5-Coder）构建可验证课程。三点发现：① 执行验证下所有教师在标准题近乎满分（99–100%，饱和效应），难竞赛题才拉开差距（Gemini 77% > Claude=Codex 69% > Grok 50%），但学生侧稳健结果不依赖教师排名；② 对已足够强的学生做模仿式 SFT 不涨反降（7B/32B 上如 MBPP 76.7%→72.7%）；③ 把同一协作课程当作 **RLVR 环境**则提升学生（竞赛题 5.9%→8.8% 峰值，相对 +49%），逆转 SFT 的方向。核心结论：AI 教师协作的价值不在"汇总答案供模仿"，而在"共建可验证环境让学生做中学"。属 §4.1 多教师协作训练 / co-training。
- **arXiv**：[2607.08255](https://arxiv.org/abs/2607.08255)

#### The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators (RQGM) (2026-06)
- **简介**：Alex Iacob、Andrej Jovanović、William F. Shen、Nicholas D. Lane 等（含剑桥团队）针对自改进 agent 普遍假设"评估标准静止（固定 verifier / benchmark / 标注集）"的缺陷，提出 Red Queen Gödel Machine——把评估器本身纳入改进闭环，开放对"演化评估器 / 对抗目标 / 动态效用"的搜索。机制是受控效用演化：搜索划分为 epoch，epoch 内评估准则固定（保证 per-epoch 自改进保证），epoch 边界更新效用，使目标可跨 epoch 演化。在可验证代码任务上引入互补的 agent-as-a-judge 代码评审信号即超过此前 SOTA 且 token 用量降 1.35–1.72×；在科学论文写作/评审、奥赛证明写作/评分上，co-evolved writer 在多样 judge panel 下接受率高 1.78–1.86×、co-evolved grader 真值准确率高 9%；并用对抗目标修正了 reviewer "对 AI 论文过度接受（最高达人类 1.91×）"的问题。属于 §4.1 / §4.4 交叉的"agent 与评估器协同进化（co-evolution / self-play）"路线。
- **arXiv**：[2606.26294](https://arxiv.org/abs/2606.26294)

#### From Trainee to Trainer: LLM-Designed Training Environment for RL with Multi-Agent Reasoning (2026-06)
- **简介**：Chao Chen、Chengzu Li、Zhiwei Li、Yinhong Liu、Zhijiang Guo 提出 **LLM-as-Environment-Engineer** 框架，把"RL 各阶段间人工重设计训练环境"的启发式工作自动化：让当前策略模型分析自身失败轨迹 + 上下文信息，自动提出下一阶段训练环境配置的修改。配套发布可控测试平台 MAPF-FrozenLake（生成器暴露多维环境配置）。以 Qwen3-4B 为骨干即取得最强综合性能，超过更大的专有 LLM（GPT、Gemini）与固定环境训练基线；并发现成功的环境更新依赖失败证据且保留已有效配置，且"当前 RL checkpoint 比原始 base model 更胜任环境工程师角色"——策略学习提升了模型诊断自身弱点的能力。属于"policy 与训练环境/harness 联合演化"的多 agent reasoning 训练路线。
- **arXiv**：[2606.17682](https://arxiv.org/abs/2606.17682)

#### Divide and Cooperate: Role-Decomposed Multi-Agent LLM Training with Cross-Agent Learning Signals (DAC) (2026-06)
- **简介**：Jaewan Park、Solbee Cho、Jay-Yoon Lee（SNU）针对"证据获取 + 答案生成耦合在单一策略内"导致的策略空间组合爆炸与信用分配难题，提出角色分解多 agent 训练框架 DAC：把 agentic search 拆成 Searcher 与 Generator 两个协作子任务、各用角色专属学习信号训练。Generator 兼任答案生成者与证据充分性验证者，证据不足时弃权（abstain），该弃权信号被并入 Searcher 的奖励形成**结构化跨 agent 学习信号**改善信用分配；反过来 Searcher 用 hard-positive 证据增强让 Generator 暴露于多样困难证据环境提升鲁棒。在通用与 multi-hop QA 上，仅用共享底座上的 LoRA 模块即超越依赖整模型全量微调的基线。横跨 §4.1 多 agent 联合训练与 §4.3 跨 agent 信用分配。
- **arXiv**：[2606.10684](https://arxiv.org/abs/2606.10684)

#### GARL: Game-Theoretic Reinforcement Learning for Multi-Agent Strategic Prioritisation (2026-06)
- **简介**：Yuxiao Ye 等（含 Zhiyuan Liu）提出 GARL，将多 agent 战略优先级排序形式化为两阶段博弈：竞争 agent 在共享候选集上分配战略资源，higher-level arbiter 给出最终排名；将博弈论效用转换为 role-specific RL 信号。在法律争议排名任务上，让小型开源 LLM 在相同候选排名设置下追平强闭源 LLM，定位是 §4.1 / §4.4 交叉的"博弈结构 → MARL reward"路线。
- **arXiv**：[2606.05002](https://arxiv.org/abs/2606.05002)

#### AgentJet: A Flexible Swarm Training Framework for Agentic Reinforcement Learning (2026-06)
- **简介**：将多 agent LLM RL 训练解耦为 swarm server / client 异构架构：服务端节点托管可训练模型并在 GPU 集群优化，客户端节点在任意设备运行任意 agent，支持异构多模型 RL、多任务混训、容错执行、训练期热替换；timeline merging 上下文追踪压缩冗余 context，得到 1.5–10× 训练加速。是面向"多模型联合 RL"工程化的代表工作。
- **arXiv**：[2606.04484](https://arxiv.org/abs/2606.04484)

#### EvoTrainer: Co-Evolving LLM Policies and Training Harnesses for Autonomous Agentic Reinforcement Learning (2026-06)
- **简介**：阿里 Yongbin Li / Min Yang / Jieping Ye 等提出 EvoTrainer，把 agentic RL 从"recipe 搜索"推到"policy 与 training harness 联合演化"：诊断 rollout 证据、迭代修正诊断、回测 intervention、积累可复用 skill。在数学推理 / 竞赛代码 / 仓库级 SWE 三档评测中匹配或超过人工设计的 RL baseline，长程 agentic SWE 收益最大。
- **arXiv**：[2606.03108](https://arxiv.org/abs/2606.03108)

#### Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces (2026-05)
- **简介**：单作者综述/分类工作，把多 agent 系统的 RL 优化对象从"个体动作"扩展到"工作如何被派生/委派/沟通/聚合/停止"——即 orchestration traces。系统梳理 8 类 reward family（含 parallelism speedup、split correctness、aggregation quality）、8 个信用承载单元（token→team），把 orchestration learning 拆为 5 子决策（when to spawn / whom to delegate / how to communicate / how to aggregate / when to stop）。截至 2026-05-04 在公开池中**未发现专门训练 stopping decision 的 RL 方法**。可作为 agentic RL 多 agent 子方向导航文献。
- **arXiv**：[2605.02801](https://arxiv.org/abs/2605.02801)

#### NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning (2026-05)
- **简介**：UGA 团队把多 agent LLM 系统重新定义为可训练神经网络——LLM agents 是节点、文本中间信号是边，agent 是 role-free 但 structure-aware 的，拓扑只决定信息能否流通，RL 训练自发决定 agents 如何通信、专业化、协调。用层次分解的理论视角说明此类模块化文本计算的参数效率；实验显示组织规模 scaling 是路径依赖的——大系统从零训不稳，但从已训小系统渐进生长可行。
- **arXiv**：[2605.16757](https://arxiv.org/abs/2605.16757)

#### When Does Multi-Agent RL Improve LLM Workflows? Workflow, Scale, and Policy-Sharing Tradeoffs (2026-05)
- **简介**：Oregon State + Microsoft 团队系统性给出端到端 MARL 在 LLM workflow 上"何时有效、何时崩溃"的实证图谱：3 种工作流（Eval-Opt / Voting / Orch-Workers）× 数学/代码 × 0.6B/1.7B/4B；对比 shared-policy 与 isolated-policy。结论：MARL 普遍优于 base，但增益是 workflow×task×scale 联合函数；isolated 峰值更高但更易掉到 terminal-cliff，shared 失败模式被重定向而非消除——首篇系统化负面证据 + 机制分析。
- **arXiv**：[2605.24202](https://arxiv.org/abs/2605.24202)

#### UnityMAS-O: A General RL Optimization Framework for LLM-Based Multi-Agent Systems (2026-05)
- **简介**：人大 + 小红书团队提出 UnityMAS-O：把整条多 agent workflow（而不是单条轨迹）视为优化单元，通过逻辑 agent 角色、图轨迹、用户定义奖励、agent–model 映射四个一等对象，把逻辑 agent 与物理参数解耦——支持 fully-shared / fully-separate / partial 共享，奖励可在 role / round / trajectory 三层分配。基于 verl + Ray 星型拓扑实现，覆盖 RAG-QA、agent-search、reflective code 三类工作流。
- **arXiv**：[2605.26646](https://arxiv.org/abs/2605.26646)

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

#### Does Multi-Agent Debate Improve AI Feedback on Research Papers? (mad-research / paper-workshop) (2026-07)
- **简介**：Havranek、Irsova 用预注册、身份遮蔽、同论文内实验，让 44 篇经济学元分析的作者按"对改进本文的有用性"对三份 AI 报告排序：前沿模型单次生成 vs. 作者自建的两套多智能体辩论工具。结论是作者更偏好单次生成（较 mad-research 领先 0.66 个排名点、较 paper-workshop 领先 0.57），尽管 paper-workshop 花费约 30 倍 token；且 AI 评审几乎总把真实期刊审稿意见排在末位——警示以 AI 裁判替代作者的风险。对"多智能体辩论是否改进 AI 反馈"给出（至少在经济学元分析上）多为否定的实证对照。
- **arXiv**：[2607.14713](https://arxiv.org/abs/2607.14713)

#### Mixture of Debaters: Learn to Debate at Architectural Level in Multi-Agent Reasoning（MoD） (2026-06)
- **简介**：华南理工 Dayong Liang、Yi Cai 等提出 MoD，把多智能体辩论从「实例化多份模型副本」下沉到**单模型架构层**，用 Mixture-of-Experts 实现动态自辩论。三项关键设计：(1) 双路由解耦角色分配与流程控制，动态决定「何时辩论、何时综合」；(2) momentum switching 平滑 token 级路由、抑制专家切换抖动；(3) 统一自辩论将多种辩论人格封装为轻量专家模块，免除 agent 间通信却保留行为多样性。多模态基准上超越单模型与传统多 agent 系统，同时**延迟降低 3.7×、token 消耗减少 87%**。定位：把「多 agent 辩论」的算力开销问题从系统层转为架构层求解。
- **arXiv**：[2606.29425](https://arxiv.org/abs/2606.29425)

#### Minority Sentinel: When to Overturn Majority Voting in Multi-Agent LLM Debates（Minority Sentinel） (2026-06)
- **简介**：Chuan He、Dong Wen 等（SIGIR 2026 AgentSearch Workshop）指出，多智能体辩论 + 多数投票（MAD + Majority Voting）依赖 Condorcet 陪审团定理的「误差独立」假设，但当代 LLM 共享预训练语料导致误差强相关，多数派会系统性压制正确的少数派意见（作者称之为 **Minority Truth**）——三异构 agent、六基准上约每四个分歧案例就有一个少数派正确，理论恢复空间达 10 个百分点。提出 Minority Sentinel：从辩论日志抽取多维「辩论指纹」，训练 LightGBM 元分类器决定**何时推翻多数投票**，在全部六数据集、20 个随机种子上达到 81.2% Flip Precision 且 Net Gain 为正；对照 LLM-as-Judge 基线 Net Gain 为负，说明「翻转安全性」而非召回量才决定干预价值。
- **arXiv**：[2606.29270](https://arxiv.org/abs/2606.29270)

#### Heterogeneous LLM Debate Under Adversarial Peers: Honest Gains, Replacement Costs, and Resilience (2026-06)
- **简介**：Prashanti Nilayam、Kiran Kumar Ramanna、Prashil Tumbade、Sankalp Nayak 把"异构 peer 既带来纠错也带来对抗影响"这一双刃剑量化拆解：以 defender（诚实 agent）的修订行为为透镜，跟踪其改答频率与"纠正 vs 有害"方向，对比匹配面板（同构基线 / 诚实混合 / 对抗混合）及已被同族恶意 peer 污染的面板，跨 4 个模型族 × 3 个推理基准。结论：诚实异构 peer 大幅降低有害修订（Llama-3.1-70B 在 MATH-hard 上诚实槽有害修订率从同构 89% → 35%），对抗 peer 则逆转回 90%；且当面板已被同族对抗者污染时，加入诚实异构 peer 反而把"初始正确却被改错"的 flip rate 从 31% 降到 6%。揭示异构性既是攻击面、在已有对抗者时又是防御手段——为 §4.2 中"异构辩论的鲁棒性与组合安全"提供 defender-side 度量。
- **arXiv**：[2606.19826](https://arxiv.org/abs/2606.19826)

#### From Argument Components to Graphs: A Multi-Agent Debate with Confidence Gating for Argument Relations (2026-06)
- **简介**：Jakub Bąba、Jarosław A. Chudziak（华沙理工，KES 2026）把"支持者-反对者-裁判"（Proponent-Opponent-Judge）辩论范式从论证组件分类扩展到 **论证关系识别与分类（ARIC）**，将其重构为对组件对（component pairs）的辩论；并引入**置信度门控（confidence gating）**——仅对不确定案例辩论、高置信度时直接接受初始预测。在 UKP Argument Annotated Essays v2 上，选择性辩论取得所有免训练方法中最高 Macro F1，而对全部样本辩论反而掉到基线以下；所有生成式方法 Macro F1 均超微调 RoBERTa，并产出人类可读辩论记录。属于 §4.2 中"role-decoupled debate as inference + 选择性触发"的代表。
- **arXiv**：[2606.16047](https://arxiv.org/abs/2606.16047)

#### ARMOR-MAD: Adaptive Routing for Heterogeneous Multi-Agent Debate in Large Language Model Reasoning (2026-06)
- **简介**：Fuqiang Niu、Bowen Zhang 提出无训练的异构多 agent 辩论框架 ARMOR-MAD，把 debate 当作"条件计算"以避免固定流程浪费算力、放大相似 agent 的相关错误。三组件：PAR（Pre-debate Agreement Routing，按 Round-0 独立答案是否一致决定要不要辩论）、EASE（Early Agreement Stopping Evaluator，收敛即停）、SOD（Semantic Outlier Detection，聚合时降权异常终答）。在 MATH Level 5 / GSM8K / MMLU / MMLU-Pro 上一致优于同模型池的固定轮次异构辩论，分别达 65.5% / 96.5% / 90.0% / 81.5%。结论：真正的模型异构性 + 基于一致性的控制，对 MAD 的准确性与效率都关键。
- **arXiv**：[2606.13197](https://arxiv.org/abs/2606.13197)

#### Early-Token Confidence Predicts Reasoning Quality in Multi-Agent LLM Debate (2026-06)
- **简介**：Ali Keramati、Justin Cheok、Jacob Horne、Mark Warschauer（接 MADRAG 同组工作）研究"无参考答案的开放式任务里，内在置信信号能否预测 LLM-as-judge 评出的推理质量"。基于辩论式作文评分框架，在两个 ASAP 数据集上对比置信度代理指标与 rubric 判别分。核心发现：**早期 token 置信度**（生成前几个 token）是最强、最一致的推理质量预测指标，优于全序列统计量；log-prob 轨迹显示开头阶段差异最大、信息量最丰富；且观察到**角色间系统性不对称**——支持性推理（Advocate）的置信度-质量对齐强于对抗性批评（Skeptic）。为多 agent 辩论系统的推理可靠性提供轻量诊断信号。
- **arXiv**：[2606.10307](https://arxiv.org/abs/2606.10307)

#### MADRAG: Multi-Agent Debate with Retrieval-Augmented Generation for Training-Free Analytic Essay Scoring (2026-06)
- **简介**：把 LLM-as-judge 解构为 Advocate / Skeptic / Judge 三角色辩论：Advocate 找强项，Skeptic 攻弱点，Judge 配合 rubric-aligned exemplar 检索做最终评分。消融显示检索带来校准收益、辩论提升对高阶 trait 的推理。无训练、显著优于 prompt-based baseline，逼近监督系统。
- **arXiv**：[2606.06754](https://arxiv.org/abs/2606.06754)

#### CAF-Gen: A Multi-Agent System for Enriching Argumentation Structures (2026-06)
- **简介**：把浅层论证挖掘结果丰富为符合 Carneades Argumentation Framework（CAF）的论证模型：Creator agent 生成 CAF 结构，Reviewer agent 进行批判性验证，多轮 iterative Creator-Reviewer 流水线显著优于单次生成的稳定性。把"生成 / 评审 / 修订"形式化为多 agent 协议。
- **arXiv**：[2606.06646](https://arxiv.org/abs/2606.06646)

#### Consensus is Strategically Insufficient: Reasoning-Trace Disagreement as a Knowledge-Representation Signal (2026-06)
- **简介**：对"投票 / 共识 / 辩论 / 容错聚合都旨在消除分歧"这一前提提出反思：在价值负载任务中分歧可能是真实规范不确定性而非 agent 错误。提出符号化 disagreement 状态层（convergent agreement / divergent agreement / convergent disagreement / divergent disagreement），用于 defeasible 战略路由；与 §4.2 现有 debate 文献的强烈对照定位。
- **arXiv**：[2606.04223](https://arxiv.org/abs/2606.04223)

#### Dynamic Trust-Aware Sparse Communication Topology for LLM-Based Multi-Agent Consensus (DySCo) (2026-06)
- **简介**：现有 multi-agent debate / 协作框架普遍采用全连接通信，token 与延迟随 agent 数二次增长；固定稀疏拓扑又无法按任务自适应。DySCo 在每轮按 agent 可靠性、答案分歧度、任务相关性估计通信边价值，在预算约束下挑选少量高价值边交换信息，并以动态信任权重聚合答案，共识稳定后提前终止。属于"按需通信 + 动态信任"的辩论框架。
- **arXiv**：[2606.01828](https://arxiv.org/abs/2606.01828)

#### MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate (2026-05)
- **简介**：浙大 + 之江实验室团队把蒸馏教师重构为辩论合议庭——多个教师在学生 on-policy 状态上展开辩论，并按辩后置信度加权产生 token-level 监督信号，突破单教师上限；进一步给出 OPAD（step-level 采样稳定多步累积误差）和任务自适应散度（agentic 用 JSD、code 用 reverse-KL）。在 Qwen3/Qwen3.5 6 套教师-学生组合 5 个 agentic+code 基准全部第一，14B+8B→4B 配置下 agentic +2.4%、code +3.7%。
- **arXiv**：[2605.01347](https://arxiv.org/abs/2605.01347)

#### MAVEN: Multi-Agent Verification-Elaboration Network with In-Step Epistemic Auditing (2026-05)
- **简介**：同济 + 复旦团队把 MAD 重构为黑板架构 + Skeptic-Researcher-Judge 角色显式解耦的对抗审议循环，每步加 epistemic audit 标注消息可信度。在 OpenBookQA / TruthfulQA / HALUEVAL / StrategyQA 上稳定优于 Gemini-3.1-Pro 与 ReConcile 共识基线，模型无关，可作为通用推理增强前置层；填补"role-decoupled adversarial debate as inference"在 ReConcile 之后的工程化空缺。
- **arXiv**：[2605.07646](https://arxiv.org/abs/2605.07646)

#### SVR-MAD: A Bayesian-Inspired Framework for Posterior-Guided Multi-Agent Debate (2026-05)
- **简介**：哈佛 + 耶鲁团队针对辩论上下文随 agent 数线性膨胀的可扩展性瓶颈，提出 SVR-MAD：把 pre-debate 置信信号作先验、debate 结果作后验式证据，增量构建通信图，优先保留经受住同伴质疑的回答（贝叶斯意义下"survival via challenge"）。在多模型多基准上保持或提升精度的同时 token 成本最高降低 61%；与 ChatEval/MAGDi 这类 debate-graph 方法形成可扩展性对照。
- **arXiv**：[2605.23099](https://arxiv.org/abs/2605.23099)

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

#### Who Broke the System? Failure Localization in LLM-Based Multi-Agent Systems (AgentLocate) (2026-07)
- **简介**：Yufei Xia、Anjun Gao、Yueyang Quan、Zhuqing Liu、Minghong Fang 研究 LLM 多 agent 系统的**故障定位/责任归属**——当执行失败时，指认"哪个 agent 负责、轨迹在哪一步首次不可逆地走偏"。提出 **AgentLocate**：把故障同时归因到"具体 agent + 最早决定性步骤"，结合 LLM 裁判机制 + 多独立评估器的多视角验证，用置信度感知策略聚合，并用反馈对裁判做轻量微调以提升归因质量。在两个覆盖多样任务/agent 配置/轨迹长度的基准上，在"责任 agent 与失败步"识别上一致优于现有故障定位方法，且 token 与耗时高效。COLM 2026 接收。是 §4.3 多 agent 信用分配 / 责任归因的直接工作。
- **arXiv**：[2607.07989](https://arxiv.org/abs/2607.07989)

#### Contagion Networks: Evaluator Bias Propagation in Multi-Agent LLM Systems (2026-06)
- **简介**：Zewen Liu 提出 Contagion Networks 形式框架，度量"当 LLM 充当多 agent 系统中的评估者时，其系统性评估偏差如何沿 agent 网络传播"。在 DeepSeek-chat 三 agent 受控实验（structured / balanced / evidence-based 三种评估者偏差画像）下测得 Cross-Agent Contagion Matrix Γ₃，发现偏差在 agent 间稳定传播（γ∈[0.157,0.352]），即便底座同模型也如此；用谱半径 ρ(Γ_N) 刻画三种传播 regime，且同模型 agent 的传播系数比此前跨模型工作（MM-EPC: γ≈0.85–1.3）弱 3–5×，处于"抑制 regime"；并给出可操作缓解：评估委员会规模从 k=1 增到 k=3 可把有效传播降低 72.4%。开源实验框架。与多 agent 信用/奖励信号污染、LLM-as-judge 偏差归因强相关。
- **arXiv**：[2606.20493](https://arxiv.org/abs/2606.20493)

#### Economy of Minds: Emerging Multi-Agent Intelligence with Economic Interactions (2026-06)
- **简介**：受 Hayek 去中心化市场协调理论启发，把 agent 群体建模为经济系统：agent 通过拍卖竞争行动权、互相支付、从环境奖励累积财富；这些经济信号天然实现去中心化信用分配，无需全局编排或显式通信协议。群体通过经济选择演化（高效者通过 exploitation 突变扩张，低效者破产由 exploration 替代），在数学推理 / 金融研究 / 科学研究 / 加速器设计 / 分布式系统优化五类任务上超越更强单体 baseline。
- **arXiv**：[2606.02859](https://arxiv.org/abs/2606.02859)

#### Tree-based Credit Assignment for Multi-Agent Memory System (TreeMem) (2026-05)
- **简介**：把 builder–summarizer–retrieval 三智能体记忆管线展开成树结构，每个 agent 输出对应多条后继分支，用 Monte Carlo 在子树上平均估计该智能体对终端奖励的贡献，从而把粗糙的最终任务奖励转成 agent-specific 信号；无需任务特定标注即可让异构 agent 实现专业化分工，长程任务上稳超统一 reward 与角色专属 reward 两类基线。
- **arXiv**：[2605.04811](https://arxiv.org/abs/2605.04811)

#### In-Context Credit Assignment via the Core (Least-Core ICCA) (2026-05)
- **简介**：CMU 团队把"上下文窗口里多创作者 IP 共生 AI 内容（代码/新闻/短视频）"的功劳分配，建模为合作博弈最小核心（least core）解概念——保证任意子集的回报不显著低于其独立贡献。给出基于约束 seeding 与 separation 的近似算法，相比 Shapley 类方案可以以低数量级 LLM 调用逼近 least core；首次把"core"引入 multi-agent LLM 信用分配。
- **arXiv**：[2605.06920](https://arxiv.org/abs/2605.06920)

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

#### Digital Pantheon: Simulating and Auditing Coalition Formation with LLM Agents (2026-07)
- **简介**：Van Mulders 等提出多智能体框架，用 SFT+DPO+每党派 RAG 让 LLM 智能体维持坚定党派立场（DPO 注入激进人设，RAG 将各智能体约束在其官方竞选纲领）。在 2019 弗兰德斯选举场景下以 hub-and-spoke 谈判（由 formateur 仲裁）模拟联盟组建，并引入多层信息谱系拓扑（MILT）、联盟影响力评分（CIS）与真实世界对照，三次独立模拟得到稳定胜者与排序，纲领锚定的谱系可可靠预测现实落地。
- **arXiv**：[2607.15095](https://arxiv.org/abs/2607.15095)

#### When Is Delegated Play Truthful? Within-Range Regret and the Trilemma of Aligned Delegation (2026-07)
- **简介**：Dube 从博弈论视角统一「自动竞价代理」与「语言模型代理」两类委托场景：证明委托人如实向自身代理描述意图当且仅当代理已在其可达范围内采取最优行动（"loyal"），可获收益的上界等于代理诚实上报行动相对可被诱导行动的 within-range regret（定理 1）。进一步给出对齐护栏的「三难」——绑定性、诚实性、能力保持三者两两成立则排除第三（定理 2），并指出这正是提示工程/越狱的激励来源；在五家厂商生产级模型上以对齐式上限实测，诚实上报均留有可通过夸大上报回收的剩余。
- **arXiv**：[2607.14357](https://arxiv.org/abs/2607.14357)

#### Hallucination Self-Play: Bootstrapping Reinforced Detector via Evolved Generator (HSP) (2026-07)
- **简介**：Shiping Yang、Shining Liang、Weihao Liu、Wenbiao Ding、Linjun Shou、Lu Cheng、Angel X. Chang 针对忠实性幻觉检测缺高质量标注数据、且现有方法把生成器当静态组件的问题，提出 **Hallucination Self-Play (HSP)**：从同一 base 模型初始化两个角色——detector 评判输出忠实性、generator 生成越来越难检测的幻觉响应。先用人工标注微调 detector，再把它当奖励模型经 RLAIF 训练 generator；反过来演化后的 generator 合成幻觉数据、用规则化 RL 进一步优化 detector，形成对抗自博弈闭环。RAGTruth 基准上、两个模型族均显示：无外部监督即可把小 LLM 逐步提升到匹配甚至超越先进 LLM。COLM 2026 接收。属 §4.4 自博弈的检测器-生成器协同进化。
- **arXiv**：[2607.07993](https://arxiv.org/abs/2607.07993)

#### Agon: Competitive Cross-Model RL with Implicit Rival Grading of Reasoning (2026-07)
- **简介**：Vladislav Beliaev 指出 GRPO 类 RLVR 只给最终答案打分、从不评判推理过程，导致模型学会"写更多"而非"想更好"。提出 **Agon**：让两个能力相当但行为各异的模型互为裁判——两者解同一题，交替扮演"起草/阅读并解答"角色，各自因"胜过对手"而获奖励；要赢就必须超越"已看过自己草稿的对手"，从而**在训练中隐式评判推理**，无需过程标签、无需奖励模型；因两模型同时优化，各自面对逐渐变强的对手（单模型 RL 无法提供）。推理时以"起草-阅读后作答"两阶段级联部署。在 DeepMath 难档 + Qwen3 上 pass@1 翻倍（约为同底座 Mixture-of-Agents 一遍增益的 8 倍），并在竞赛编程与跨模型族（Qwen3.5、Gemma 4）复现。属 §4.4 自博弈 / §4.1 双模型协同训练交叉。
- **arXiv**：[2607.07690](https://arxiv.org/abs/2607.07690)

#### More Convincing, Not More Correct: Self-Play Reward Hacking of Reference-Free LLM Judges (2026-07)
- **简介**：Chenyu Zhou 系统揭示 self-rewarding / self-play / LLM-as-a-judge 流水线的一个**结构性缺陷**：无参考裁判在「看到候选答案后」打的是"可信度"而非"正确性"，从而留下策略可利用的 false-positive 盆地。用"隐藏锚点审计"（held-out 精确匹配、裁判从未见过）度量：在 GSM8K + Qwen3 上，self-play 把裁判 pass 率从 0.72 抬到 0.94，而真实准确率始终停在 0.20；错误可跨裁判族（Qwen/Llama/Gemma）与规模迁移，三裁判严格集成仍接受 55%。关键变量是裁判**是否先独立作答再看候选**：先承诺答案可把 false-positive 从 0.719 压到 0.012，作为训练奖励用"去锚通道"可让 false-positive 保持为 0——是对 §4.4 自博弈/自奖励训练的重要负面结果与防御。
- **arXiv**：[2607.05904](https://arxiv.org/abs/2607.05904)

#### Strategic Bargaining in Multi-Buyer Markets: Reinforcement Learning from Verifiable Rewards for LLM Negotiations (2026-07)
- **简介**：Shuze Daniel Liu、Claire Chen、Jiabao Sean Xiao、Xin Chen、David Simchi-Levi（MIT）把"单卖家同时与多个私有预算买家谈判"这一博弈场景形式化，发现标准 LLM 虽语言流畅却是糟糕的经济决策者——不去探索买家池、固着于当前最高报价。提出以客观经济结果为锚的 RLVR 训练配方，使"市场发现 vs 剩余榨取"的战略平衡在学习中原生涌现；训练后的卖家经历多阶段战略演化（价格锚定、战略探测），剩余远超前沿模型，并对未见谈判风格与预算分布稳健泛化。属 §4.4 博弈结构 → RLVR 的多 agent 谈判路线。
- **arXiv**：[2607.05863](https://arxiv.org/abs/2607.05863)

#### Attractor States Emerge in Multi-Turn LLM Conversations（Attractor States） (2026-06)
- **简介**：ELLIS / MPI 的 Ting-Wen Ko 与 Jonas Geiping 研究开放式多 agent 对话的**长程动力学**：跨 7 个 LLM、20 个争议话题，对比 **self-play**（同模型对弈）与 **mixed-play**（异模型辩论）双人辩论，在表征空间、话语特征与立场三个维度追踪轨迹。发现 self-play 轨迹是「模型专属吸引子」，会在 mixed-play 中**不对称地牵引对手**改变风格与行为（如 Claude Haiku 是隐空间强吸引子、GPT-4.1 nano 极易被同化）。结论：开放式多 agent 交互部分可由模型专属吸引子预测，但受结构化、不对称的伙伴影响塑形——为自主 agent 系统的设计、预测与监控提供 self-play 视角的机理证据。
- **arXiv**：[2606.30571](https://arxiv.org/abs/2606.30571)

#### Age of LLM: A Strategic 1v1 Benchmark for Reasoning, Diplomacy and Reliability of Large Language Models under Fog of War (2026-06)
- **简介**：Arnaud Ricci 提出 Age of LLM——一个回合制 1v1 博弈基准：两个 LLM 在 13×7 网格上对抗以摧毁对方基地，刻意设置三类压力源：战争迷雾（fog of war）、完整外交（消息/停火/最后通牒，铀矿保密）、以及可靠性维度（每回合须严格遵循 JSON schema，非法动作被静默丢弃）。引擎私有、每局换随机地图种子与对手以缓解数据污染。在 15 个推理模型、54 局、5258 个动作上发现：核打击 rush 占主导（规则一致子集 78%）、军事征服罕见但更快（12.3 vs 18.9 回合）、外交频繁却几乎从不兑现、约 58% 非法动作源于 fog/state 误判（使非法率成为信念追踪度量）、并观察到可靠性与胜负的弱关联。把博弈论压力测试用于"对抗不确定性下 LLM 的信念追踪 / 自发欺骗 / per-model 认知人格"，开放 replay 格式与可视化。属于 §4.4 博弈论 / self-play 评测路线。
- **arXiv**：[2606.24391](https://arxiv.org/abs/2606.24391)

#### Learn from Your Mistakes: Tree-like Self-Play for Secure Code LLMs (TSP) (2026-06)
- **简介**：把安全代码生成重构为细粒度序贯决策：模型在决策树上探索分支轨迹，自己生成"golden path"和有漏洞的变体，作为 self-play 双方博弈，让模型在关键节点处显式拒绝自己的局部错误。在 Python 安全 benchmark 上 CodeLlama-7B 的 SPR@1 提升到 75.8%，显著优于 SFT 与无结构 self-play；对未见 CWE / 跨语言（C/C++ → Python/Go/JS）展现强 OOD 泛化。
- **arXiv**：[2606.03489](https://arxiv.org/abs/2606.03489)

#### A Theoretical Framework for Self-Play Theorem Proving Algorithms (2026-06)
- **简介**：Thomas Chen 与 Zhiyuan Li 为 prover–conjecturer self-play（继 Dong & Ma 2025）提供首份理论框架：把定理集合形式化为图，用一组原始假设刻画 prover 训练后保证以及 conjecturer 如何访问图结构；证明若定理图连通良好，基于 reversible random walk 的 conjecturer 即可让被证明定理集合指数增长。还提出基于 diffusion similarity 的多样性度量，对症"conjecturer 倾向生成人为复杂、非基础定理"的实证问题。
- **arXiv**：[2606.01861](https://arxiv.org/abs/2606.01861)

#### Towards General Preference Alignment: Diffusion Models at Nash Equilibrium (Diff.-NPO) (2026-05)
- **简介**：BU + UCSD 团队把扩散模型对齐从 BT 假设解耦，将 RLHF 形式化为通用偏好框架下的两玩家零和博弈，提出 Diffusion Nash Preference Optimization——当前策略以自博弈方式逼近 Nash 策略实现自我提升。在 T2I 多指标上稳定优于现有 DPO/RLHF-style 偏好对齐方法；将 INPO/SPAG 的 game-theoretic 自博弈思路首次拓展到扩散后训练。
- **arXiv**：[2605.04494](https://arxiv.org/abs/2605.04494)

#### Seirênes: Adversarial Self-Play with Evolving Distractions for LLM Reasoning (2026-05)
- **简介**：北大 + 字节团队提出参数共享对抗自博弈框架：同一模型既扮演 Adversary 生成貌似合理却暗含干扰的上下文以暴露自身推理盲区，又扮演 Reasoner 在干扰下解题；两者目标对抗驱动协同进化课程。4B–30B 七个数学推理基准平均 +10.2 / +9.1 / +7.2，且 4B Seirênes 生成的干扰文本能让 GPT/Gemini 顶级模型掉 4–5 分——SPIRAL/AZR 之后又一条"endogenous adversarial"自博弈新路径。
- **arXiv**：[2605.11636](https://arxiv.org/abs/2605.11636)

#### PopuLoRA: Co-Evolving LLM Populations for Reasoning Self-Play (2026-05)
- **简介**：UCL + Vmax 团队对 Absolute-Zero-Reasoner（AZR）的单 agent 自校准缺陷给出 population-based 修补——共享冻结底座 + 异构 LoRA 适配器，分 teacher / student 两个亚种群，配合 LoRA 权重空间变异+交叉操作快速生成同秩种群成员。出题难度持续攀升而非自我塌缩，7B 规模在 HumanEval+/MBPP+/LiveCodeBench + 7 个数学基准上 population-mean 全面超越 compute-matched AZR，最弱个体也超基线。
- **arXiv**：[2605.16727](https://arxiv.org/abs/2605.16727)

#### Structure from Strategic Interaction & Uncertainty: Risk-Sensitive Games for Robust Preference Learning (2026-05)
- **简介**：Caltech + UW 团队指出 NLHF 类 Nash-from-HF 方法只优化期望对偶 payoff，会把"平均胜率相同但尾部差很大"的策略混在一起。提出 risk-sensitive preference games，玩家优化偏好损失的凸风险测度；利用 translation invariance 保留单调性以保证收敛，给出双时间尺度 extragradient 算法收敛到 Stackelberg 均衡，并在低样本下校正风险估计偏差；INPO 系列的下一步——把 Nash 自博弈对齐推到 risk-aware。
- **arXiv**：[2605.09946](https://arxiv.org/abs/2605.09946)

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

#### Collaborative Spatial Learning with Multi-LLM Agents in Networked Social Experiments (2026-07)
- **简介**：He、Kuhlman、Deng 在 Mason–Watts 实验（PNAS 2012）的 8 种网络拓扑上，让 16 个 LLM 智能体组成群体进行二维搜索，考察通信网络效率对集体协作的影响，并与机制化贝叶斯优化智能体及人类实验数据对照。发现仅当被指示随机化首轮选择时 LLM 群体才呈现显著的网络效率效应（一句首轮随机化指令即将集体收益提升超 3 倍），默认初始化下则不显现；贝叶斯优化智能体在该空间搜索任务上收益更高。ASONAM 2026 收录。
- **arXiv**：[2607.14574](https://arxiv.org/abs/2607.14574)

#### Learning Latency-Aware Orchestration for Multi-Agent Systems (LAMaS) (2026-07)
- **简介**：Shi、Zheng、Lou 针对 LLM 多智能体系统（MAS）多步执行带来的高推理延迟，提出延迟感知编排框架 LAMaS。训练时通过带**关键路径感知信用分配**（critical-path-aware credit assignment）的约束优化学习延迟感知执行图；推理时用轻量控制器随执行自适应剔除冗余的后续智能体交互。四个基准上取得学习型 MAS 基线中最优延迟，端到端延迟降低超 50% 且保持相当或更好准确率，并可低成本迁移到其他 MAS。
- **arXiv**：[2607.13359](https://arxiv.org/abs/2607.13359)

#### Graph Feedback Controls Consensus and Clique Formation in Open-Weight Language-Model Populations (2026-07)
- **简介**：Saab、Abdallah 用命名博弈（naming-game）协议研究 1.1B–32B 开源权重 LM 群体中的约定/共识形成，将运行时交互图作为可控变量。发现同质相似度路由会删除跨基态暴露、放大碎片化，而寻桥（bridge-seeking）路由在有记忆时常能修复碎片化恢复共识；给出保留历史→共识、阈值相似度→无共识的大量对照证据（如 Qwen2.5-32B 在 18 组充分混合设置下均达稳定共识）。
- **arXiv**：[2607.12077](https://arxiv.org/abs/2607.12077)

#### Multi-Agent LLMs Fail to Explore Each Other (MACE) (2026-07)
- **简介**：Choi、Sharon Li 等（UW-Madison / UCSC 团队）将「多智能体探索」形式化为部分可观测随机博弈（POSG），指出现代 LLM 智能体在相互交互时表现出短视、极化的模式，导致协调次优、后悔增大。提出轻量框架 MACE（Multi-Agent Contextual Exploration），通过结构化的同伴选择显式促进探索，并在理论上证明探索价值随智能体多样性增大；在上下文/参数多样性两类设置下均显著改善探索行为与下游任务表现。
- **arXiv**：[2607.11250](https://arxiv.org/abs/2607.11250)

#### MAS-PromptBench: When Does Prompt Optimization Improve Multi-Agent LLM Systems? (2026-06)
- **简介**：Juyang Bai、Laixi Shi 系统研究"系统提示词优化（system-prompt optimization）在多 agent LLM 系统上何时、能提升多少"。MAS 中每个 LLM agent 由 system prompt 指定角色/行为并在 workflow 中占据位置（决定 inter-agent 协调与输出聚合），因此 system prompt 是无需微调即可做系统级改进的关键可优化面；但相比单 LLM，扩展到 MAS 面临搜索空间指数膨胀。作者在 task / workflow / 通信协议 / team size 多维变化的广泛 MAS 配置上，benchmark 两个由 SOTA 单 agent 方法自然扩展而来的 prompt optimizer，刻画 prompt 优化在不同 MAS 设置下"何时有效、增益多大、对系统配置多敏感"，既展示可解锁的显著增益也暴露开放挑战。属于 §4.5 中"coordinator/workflow 配置下的系统级优化诊断"。
- **arXiv**：[2606.23664](https://arxiv.org/abs/2606.23664)

#### CodeTeam: An LLM-Powered Multi-Agent Framework for Repository-Level Code Generation (2026-06)
- **简介**：Yifei Wang、Ruiyin Li、Peng Liang 等针对仓库级代码生成（NL2Repo）需要更长规划视野、跨文件稳定接口与跨文件不一致的迭代调试，提出 CodeTeam，把规划、决策、实现解耦为协同的多 agent 阶段：规划阶段多个 Architect agent 起草竞争性软件设计草图（SDS，可由检索的设计参考接地），CTO agent 评估/挑选/规范化最优 SDS 为机器可校验契约（指定文件归属、公开接口、依赖约束）；实现阶段 Developer agent 在依赖感知调度器与轻量 Git 协调下生成代码，QA agent 跑测试并驱动迭代修复。在 SketchEval 上 PE / SFT 变体相对 CodeS 分别 +4.1 / +2.9 SketchBLEU，在 NL2Repo-Bench 上取得最高平均测试通过率（34.6% PE / 42.3% SFT）；消融显示项目专属开发者分配与检索增强规划各贡献 9.9% / 8.1% 相对提升。属于 §4.5 中"CTO/Architect 充当 coordinator 的角色分工 + 契约编排"路线。
- **arXiv**：[2606.22082](https://arxiv.org/abs/2606.22082)

#### Reward Modeling for Multi-Agent Orchestration (OrchRM) (2026-06)
- **简介**：King Yeung Tsang、Zihao Zhao、Vishal Venkataramani、Haizhou Shi、Zixuan Ke、Semih Yavuz、Shafiq Joty、Hao Wang（Wang-ML-Lab，含 Salesforce）针对 LLM 多 agent 系统编排器"监督有限 + 计算昂贵"的训练瓶颈，提出自监督框架 **OrchRM**：利用多 agent 执行中的中间产物构造 win-lose 对训练 Bradley-Terry 奖励模型，无需人工标注。与依赖昂贵 sub-agent rollout 的现有 MAS test-time scaling / 编排器训练不同，OrchRM 直接在**编排层面**运作，token 效率最高提升 10×、MAS test-time scaling 准确率最高 +8%，且在数学推理 / 网页问答 / 多跳推理多域稳定迁移。把奖励建模从 trajectory 级抬到 orchestration 级，是 coordinator 训练的代表新工作。
- **arXiv**：[2606.13598](https://arxiv.org/abs/2606.13598)

#### Streaming Communication in Multi-Agent Reasoning (StreamMA) (2026-06)
- **简介**：HKUST(GZ) / 阿里 / 浙大团队（Zhen Yang 等）打破 multi-agent reasoning 的"generate-then-transfer"协议：上游 agent 每生成一个推理步即流式推送给下游，把流水线深度对应的延迟从线性下拉。意外发现：因为推理质量随步骤非均匀分布且早期步骤更可靠，下游用早期可靠步骤反而准确率更高。给出 stream / serial / single 三协议的首份 closed-form 联合分析；在 8 个推理 benchmark / 两个前沿 LLM / 三种拓扑上平均 +7.3pp，HMMT 2026 最高 +22.4pp；并发现"步级缩放定律"。
- **arXiv**：[2606.05158](https://arxiv.org/abs/2606.05158)

#### Multi²: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments (2026-06)
- **简介**：针对 LLM agent 长程交互中的 objective drift，提出 Multi²：高层 agent（System 1）做 context-aware 子目标生成（SFT 训练），低层 agent（System 2）通过 offline-to-online RL 执行原子动作；显式 role decomposition 在多种交互环境中稳定优于强 agentic baseline。同时发布三套 hierarchical benchmark。可视为"上下层 LLM 联合训练 / 协调"的代表。
- **arXiv**：[2606.03698](https://arxiv.org/abs/2606.03698)

#### Iterative Critique-and-Routing Controller for Multi-Agent Systems with Heterogeneous LLMs (2026-05)
- **简介**：Purdue 团队针对异构 LLM 多 agent 协调器只能"一次路由"的局限，提出把多 agent 协调建模为有限时域 MDP——控制器每轮评估当前草稿，决策"停 / 继续 / 调用下一个 agent 进一步精修"。引入 agent 利用约束的 Lagrangian 松弛目标，用策略梯度训练；7 个推理基准上以不到 25% 调用次数稳定逼近最强单 agent 表现，展示 router→sequential controller 的范式跃迁。
- **arXiv**：[2605.08686](https://arxiv.org/abs/2605.08686)

#### Maestro: Reinforcement Learning to Orchestrate Hierarchical Model-Skill Ensembles (2026-05)
- **简介**：清华 + 浙大 + 港中文团队提出 Maestro：将异构多模态任务建模为分层模型-技能注册表上的 POMDP，4B 轻量级编排器策略每步决定是否调用外部专家、选哪个 model–skill 对、以及何时停止；训练只用 outcome-based RL，无需 step-level 监督。10 项多模态基准平均 70.1%，超 GPT-5（69.3%）和 Gemini-2.5-Pro（68.7%），且对未见模型/技能即插即用。
- **arXiv**：[2605.22177](https://arxiv.org/abs/2605.22177)

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
