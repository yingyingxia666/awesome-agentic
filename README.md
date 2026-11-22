# Awesome-Agentic

> A curated reading list of large-language-model RL papers, organized by four research directions: **Reasoning RL**, **Agentic RL**, **OPD (Off-Policy / On-Policy Distillation / Drift)**, and **Multi-Agent**.

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Last Update](https://img.shields.io/badge/last%20update-2026.09-brightgreen) ![Papers](https://img.shields.io/badge/papers-910%2B-blue) ![Time Range](https://img.shields.io/badge/time-2023.01--2026.09-orange)

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

#### Measuring the Marginal Value of Verifier-Based RL Beyond Supervised Fine-Tuning (2026-11)
- **简介**：作者 Amit Sagtani、Saloni Garg。RLVR 通常被默认为能在 SFT 之外带来额外增益，但这一贡献很少被直接度量。作者构建了确定性 benchmark **gpu-train-debug**（1,877 个合成的 GPU/LLM 训练故障诊断任务、30 类原型，由程序化 verifier 带部分得分打分），在受控条件下把 SFT 与 GRPO 都跑到收敛并精确对比。预注册实验（60 步 GRPO，学习率 1e-6）显示：1,200 条金标准示范几乎解决任务（hard split 上 SFT 0.785），而 GRPO 在分布内无可检测提升（+0.22 点，95% CI [-0.06, +0.51]），仅在分布外 held-out split 上有小幅提升（+0.33 点）。作者推测 SFT 残留的系统性错误在各采样间高度一致，削弱了 GRPO 组内学习信号。整个研究仅花费 8.69 美元算力，代码与配置全部公开。
- **arXiv**：[2611.07639](https://arxiv.org/abs/2611.07639)

#### On the Primacy Bias in RLVR Training (2026-11)
- **简介**：作者 Phuc Minh Nguyen、Rui Zhang 等。针对「RLVR 能否扩展超出基座模型已有的推理能力」这一争议，论文从 **primacy bias**（模型过度依赖已学到的表征、只在已会的问题上继续提升而少探索更难问题）的角度给出解释。受控实验与大规模分析表明：在混合难度数据上做 RLVR 会进一步锐化模型在简单题上的表现，而在难题上只有边际收益；难题虽然整体增益有限，却最能扩展推理能力、推动表征变化，但在难题上训练过久会造成对简单题的遗忘。该发现可调和此前关于 RLVR 的相互矛盾的观察：在基座较弱的 OOD 领域，难题能扩展推理，而在训练充分的领域，遗忘会掩盖难题带来的真实增益（abstract 未给出具体数值）。
- **arXiv**：[2611.04061](https://arxiv.org/abs/2611.04061)

#### Not All Reasoning Generalizes: A Case Study on Learning Generalizable Reasoning Strategies in RLVR (2026-11)
- **简介**：作者 Ziqi Pang、Yu-Xiong Wang。论文用可控的「方程求解」框架（对非双射字符串函数 f 求多个满足 f(x)=y 的输入 x）研究 RLVR 如何诱导可泛化的推理策略。发现一种隐蔽的失败模式：RLVR 让模型在分布内以结构化的逐步推导取得高准确率，但在 OOD 偏移（更多解、更长 y 等）下模型放弃推导，退化为低效的暴力枚举，准确率骤降；这既非基座能力不足，也非 reward hacking。有效的缓解方式是间接干预：与双射逆向任务联合训练，使推导更确定并显著恢复正确策略。作者还给出理论分析，说明仅用结果奖励的 RLVR 会隐式地更多奖励有效推理策略，从而在无过程监督下塑造推理过程。代码已开源。
- **arXiv**：[2611.03627](https://arxiv.org/abs/2611.03627)

#### ReSI-Math: Exploring Recoverable Capability Frontiers for Efficient Reasoning Improvement (2026-11)
- **简介**：作者 Yao Cui、Zhen Yang 等。论文指出「直接解题失败」不等于「没有能力」：部分失败可通过对模型自身轨迹的搜索恢复，形成一个依赖策略的 **recoverable capability frontier**。据此提出 **ReSI-Math**：用 Self-Judge 引导的进化搜索在固定策略下找出可恢复的失败题，并随策略演化自适应选取前沿题做顺序式 RLVR，把恢复出的能力转化为高数据效率的监督。在 Qwen3-VL-8B 上，正确-错误轨迹交叉从 621 次尝试中产生 272 个经 verifier 确认的正确后代（43.8%）；三轮顺序 one-shot RLVR 后 MATH500 与 WE-MATH 平均分由 78.71 升至 80.16；匹配的单题 RLVR 下，前沿引导选题比随机选题 54.16 vs 53.31。
- **arXiv**：[2611.04787](https://arxiv.org/abs/2611.04787)

#### The Cost of Mismatch: Noise Amplification in Zeroth-Order Reinforcement Learning (ZO-BOLT) (2026-11)
- **简介**：论文被 NeurIPS 2026 接收，作者 Lianmin Chen、Junbin Qiu、Chenxing Wei、Yao Shu、Kun He。零阶（ZO）优化是内存高效 LLM 后训练的有前景路线，但其在 RLVR 中的可靠性鲜有研究。作者建立通用的方差理论框架，发现 ZO 梯度估计的可靠性在结构上取决于两次有限差分评估是否在匹配数据上进行：在 on-policy RL 的数据失配机制下，会产生 $1/\mu^2$ 的 oracle 噪声放大（扰动尺度 μ 减小时形成不可约的噪声下界），而匹配数据机制下的 Boltzmann Targeted SFT（BOLT）通过噪声抵消消除了这一放大；并给出清晰区分两种机制的带显式常数的收敛保证。据此提出 **ZO-BOLT** 算法，借助 BOLT 的匹配数据优势对 RLVR 对齐目标做可靠的 ZO 优化。在三种模型与两个数学推理 benchmark 上，ZO-BOLT 持续优于预训练基座（摘要后续具体数值未显示）。
- **arXiv**：[2611.03260](https://arxiv.org/abs/2611.03260)

#### When All Rollouts Fail: Policy-Compatible Joint Decoding for Hard-Problem Reasoning (PICO) (2026-10)
- **简介**：作者 Zehao Wang、Hanzhu Chen 等。RLVR 在模型能力之外的难题上存在「学习悬崖」：所有 rollout 都失败时，GRPO/RLOO 等组相对估计器的 advantage 全为零；而引入专家轨迹又与当前策略不兼容，常落在学生模型低概率区域，只能学到表层模仿。论文提出 **PICO**（Policy-Compatible RL）：通过分布受限的 joint decoding，用专家数据在 token 级引导解码并保持学生模型的相对 token 概率，再用动态 top-k 在有限散度预算内约束采样分布，使轨迹天然「可验证且与策略兼容」；随后与 on-policy rollout 一起在 mixed-behavior 目标下用重要性采样联合优化。在多个推理与 agentic benchmark 上稳定优于标准 RLVR 和已有 off-policy 基线（abstract 未给出具体数值）。
- **arXiv**：[2610.36849](https://arxiv.org/abs/2610.36849)

#### Guided-RLVR: Guided Exploration in Reinforcement Learning with Verifiable Rewards (2026-10)
- **简介**：作者 Bo Yang、Zishuai Zhang 等。已有研究表明 RLVR 的 rollout 常塌缩到少数推理轨迹，限制了探索空间。受「关键步骤上的 thinking token 是推理路径结构性要素」这一发现启发，论文提出 **Guided-RLVR**：在 RLVR 上增加一个轻量 controller，在采样分布中有策略地注入语义明确的 steering token，鼓励模型从多种推理视角求解问题，从而扩大探索空间。在数学推理和代码生成 benchmark 上稳定优于现有 RLVR 方法，同时生成的推理轨迹更加多样（abstract 未给出具体数值）。
- **arXiv**：[2610.37041](https://arxiv.org/abs/2610.37041)

#### There Is No Free Lunch in RLVR: Every pass@1 Gain Has a pass@k Price (2026-10)
- **简介**：作者 Yinuo Liu。RLVR 提升 pass@1，但在大采样预算 k 下基础模型会反超，这究竟是 RL 的缺陷还是训练不足的暂态？论文将 RLVR 建模为对跨 prompt 共享策略的基础策略做指数倾斜（exponential tilt），证明在 exclusive specialization 下，任何提升 pass@1 的更新必然降低某个 prompt 的成功率；当损失触及最难 prompt 时必然出现 pass@k 交叉，交叉次数受每 prompt 成功率变化的符号变化数约束。同一倾斜恒等式还统一解释了 entropy collapse 与 entropy-performance 规律。结论是 RL 无法拓宽推理边界，只能沿边界移动，并给出可从评测日志读出该代价的诊断工具。
- **arXiv**：[2610.39295](https://arxiv.org/abs/2610.39295)

#### Reasoning or Pattern Exploitation? Mechanistic Insights from RL-Trained Language Models (2026-10)
- **简介**：作者 Aishik Nagar、Cheston Tan。RLVR 后的模型常被说成获得了「推理」能力，但这是真推理还是表层适应？论文在有限且可穷举分析的 Tic-Tac-Toe 逻辑域中，用 GRPO 训练小规模 LLM，并用 sparse autoencoder（SAE）追踪 RL 过程中内部表征的演化。准确率与对 prompt 变化的鲁棒性都很高，但机制分析显示模型主要是更依赖 prompt 中已显式给出的信息（轮到谁、棋局进度、棋盘占用），而棋盘对称性、strategic fork 等高层概念表征依然薄弱。结论：RL 后训练既不增加也不移除概念信息，只是重组了哪些特征占主导，因果干预将决策定位在后续的晚期层状态，提示推理 benchmark 可能高估模型能力。
- **arXiv**：[2610.39182](https://arxiv.org/abs/2610.39182)

#### Collective Dynamics of Self-Reinforcement: How Cross-Task Coupling Drives Policy Collapse in Unsupervised RLVR (2026-10)
- **简介**：作者 Peize Ding。无标签 RLVR 让模型对自身多数投票一致性得奖励，起初能提升推理，但最终塌缩：自评奖励趋于 1 而准确率下降，常出现对所有问题给同一答案。该塌缩只在训练问题数量很多时出现。论文用可解析的多问题共享参数模型解释这种数据规模依赖，称其为集体不稳定性：每个问题的自我强化慢于 1/N，而共享模式以与 N 无关的速率增长，从而推出临界问题数 N_c。模拟验证了尖锐相变等预测；核版本理论识别出保护性耦合和能稳定均匀集体模式的边际熵 bonus。在小型 Transformer 的无标签 RL 上，去除共享方向可完全避免塌缩，恢复则重新塌缩。
- **arXiv**：[2610.37553](https://arxiv.org/abs/2610.37553)

#### Where RL Breaks: Localizing and Repairing Failures via Targeted Continued Pretraining (2026-10)
- **简介**：作者 Xinyi Wang、Danqi Chen。越来越多证据表明 RLVR 主要是锐化基座模型已有能力，那么 RL 失败时基座缺什么？作者检查 RL 策略自身的 rollout，发现失败是局部化的：一小组特征 n-gram（从「let's recheck」这类反思短语到「the inverse of A」这类具体运算）可强烈预测 rollout 是否失败；这些失败在整个 RL 训练中持续存在，但并非模型能力所不及，同一问题的其他采样答案常常是对的。据此，作者用失败 n-gram 从预训练语料检索文本，在 RL 前对基座继续预训练，语料仅 35M token，比常规 mid-training 小 3–4 个数量级。在 OLMo2 与 Qwen3 的三类 RL 设置上，该方法在 11 个设置中的 10 个优于同等规模随机抽样文本的继续预训练，在 GSM8K 上（插入 mid-training 之前）高 2.0 分；为 OLMo2-1B 检索的文本还能迁移，使 Qwen3-4B 在混合任务上提升 1.9 分。
- **arXiv**：[2610.25836](https://arxiv.org/abs/2610.25836)

#### Explore with Memory, Learn without It: In-Context Exploration and Self-Distillation for LLM Reasoning (LEAF) (2026-10)
- **简介**：作者 Yutong Chen、Zhichao Xu、Rui Liu、Kunpeng Liu。RLVR 的探索主要依赖独立采样，没有充分利用已知错误答案、环境反馈等过程信息。提出 **LEAF**：在每次 rollout 前向上下文注入随问题动态更新的 memory（最近一次 response、其奖励及环境反馈，每次 verifier 调用后更新），再将策略更新与 rollout 解耦，通过 self-distillation 把带 memory 的行为学回到不带 memory 的策略上。在四个基准、两个开源模型上，LEAF 的探索速度比随机采样基线快约 4 倍；在未见测试集上准确率比最强基线分别高 1.59 和 1.83 个百分点，且计算量低于所有 RL 基线。
- **arXiv**：[2610.17734](https://arxiv.org/abs/2610.17734)

#### Measuring and Mitigating Solution Mode Collapse in RLVR (Re:Max) (2026-10)
- **简介**：作者 Liv G. d'Aliberti、Marwa Abdulhai、Sofiia Druchyna、Peter Henderson、Manoel Horta Ribeiro。RLVR 对模型产生哪种正确答案并不敏感，但保留多种正确解法对用户选择和提升整体表现有价值。论文提出 **ModeBench**，一个多解任务 benchmark，其 verifier 同时返回正确性和所发现的 mode，用以测量 RLVR 后训练下解多样性的变化：发现即使准确率持平或提升，RLVR 也会把概率集中到更少的正确 mode，且前沿模型已高度集中。解决方案 **Re:Max** 在 replay buffer 中为每个已发现 mode 存一个已验证样本并均匀训练，使只被找到一次的解与反复找到的解被同等练习。在三种模型规模、两种 RL 目标及更难任务设置下，replay 同时提升了策略的成功率与可成功的不同方式数量。
- **arXiv**：[2610.11064](https://arxiv.org/abs/2610.11064)

#### Same Answer, New Box: How Much of RLVR's Out-of-Domain Gain Is Answer Format? (2026-10)
- **简介**：作者 Zongye Lyu。RLVR 常被认为带来奖励验证域之外的增益（如数学训练后 MMLU 提升），但这些准确率通常由答案提取器从模型文本读出，模型若学会把答案写在提取器查看的位置，即使没有答对得更好也能得分更高。作者用公开代码的提取器与自建提取器（无论答案在 \boxed{} 中还是正文中都能读取）对 base 模型及其 RL 模型输出分别评分。在四组 7B SimpleRL base-RL 模型对上，这些提取器报告的域外多选增益中 82% 在自建提取器下消失——是答案格式而非答案本身的增益；独立 LLM 读者估算为 77%，prompt 直接要求使用 box 时为 79%。而在另一种配方训练的 32B 模型对上，几乎没有格式增益。结论：报告的增益部分取决于读取它的提取器。
- **arXiv**：[2610.20639](https://arxiv.org/abs/2610.20639)

#### How RL Reshapes LLM Reasoning: Transferability, Coverage, and Scaling Laws (2026-10)
- **简介**：作者 Ziheng Cheng、Yixiao Huang、Hanlin Zhu、Somayeh Sojoudi。近期研究对 RL 是扩展还是仅重加权 LLM 推理空间给出看似冲突的证据：数学训练可提升其他领域，Pass@1 提升却可能伴随 Pass@N 低于 base。论文在 Qwen 与 Gemma 家族上重新检验，发现既有跨域增益也有遗忘，大采样预算下的覆盖率在有些任务上增加、有些下降；对前后解答轨迹的分析显示推理策略发生迁移，由此提出把「策略选择」与「具体执行」分离的两阶段自回归策略模型，并证明 RL 的隐式偏置如何重塑策略偏好——使某些任务受益而压制另一些任务所需策略，从而在不扩展策略支持集的情况下拓宽或收窄覆盖。论文还为 RL 算力的 log-sigmoid 与 log-linear scaling law 提供理论依据并评估其预测力。
- **arXiv**：[2610.04158](https://arxiv.org/abs/2610.04158)

#### Sharpening Tax in Post-Training (2026-10)
- **简介**：作者 Changdae Oh、Qi Zeng、Qi Qi、Andrey Zhmoginov 等考察「RL 后训练只是锐化基座模型已有行为、以牺牲解覆盖率换单次精度」的假说在 agentic 任务中是否成立。发现配备轻量推理 harness 的预训练模型虽然 pass@1 低得多，但在足够的测试时预算下 pass@K 解覆盖往往超过后训练模型；机制上后训练把任务推向「总能解 / 总不能解」两个极端。提出诊断指标 **Sharpening Tax** 量化后训练后测试时可扩展性的损失；在四个家族 14 对基座/后训练模型、三个 agentic 基准（共 42 个案例）上，该税在多数设置中普遍存在，可由少量 rollout 估计。最后提出 **posterior-tempered group sampling（PTGS）**，按估计难度逐 prompt 调整采样温度，在两个 agentic 环境的 RL 训练中付税更小，在重复采样下解决更多任务并提升单次精度。
- **arXiv**：[2610.01509](https://arxiv.org/abs/2610.01509)

#### Probe with Participation Trophies: Random-Reward RL as a Probe of LLM Capability (2026-10)
- **简介**：作者 Yu Mao、Lei Yu、Zining Zhu、Freda Shi 等把「虚假 reward 悖论」与模型的可达性（reachability，即在特定约束下从当前状态进一步训练能达到什么）联系起来，提出用 random-reward RL 作为探测模型能力的工具。针对「随机 reward 也能提升性能」的两种主流解释（RL 训练机制 vs 数据污染），论文给出不同视角：虚假 reward RL 可探测超出当前表现的可达性。例如两个在合成算术上精度同为 3.5% 的 OLMo checkpoint，在相同的正确性 reward RL 下最佳运行分别达到 8.5% 和 55%。跨预训练与 mid-training 的 OLMo checkpoint 呈现三个阶段：早期 RL 几乎无提升；预训练后期正确性 reward 有效而随机 reward 仍弱；进入 mid-training 后随机 reward 也能带来大幅提升。数字遮蔽 SFT 分析显示类似顺序。
- **arXiv**：[2610.01066](https://arxiv.org/abs/2610.01066)

#### PoEM: Predicting RL Outcomes from Existing Policies (2026-09)
- **简介**：作者 Kimia Hamidieh、Giannis Daras、Antonio Torralba。基础模型的 RL 后训练算力开销大，且奖励变化或需组合多个奖励时都要从头训练。**PoEM** 问：给定新奖励函数，能否不经 RL 训练就预测策略？若新奖励是已有奖励的线性组合，则最优策略可表示为已有策略在 log 概率空间中的组合；即便奖励之间并非线性相关，不同奖励下 RL 学到的策略也大致落在一个近似低秩子空间中，组合权重仅需用奖励或基策略在样本上的输出估计。据此得到一个无需额外 RL 训练、由已有后训练模型和目标奖励近似目标 RL 策略的算法，在开放式生成的程序化与学习式奖励上评测，并扩展到数学推理、工具调用和图像生成（abstract 未给出具体数值）。
- **arXiv**：[2609.30226](https://arxiv.org/abs/2609.30226)

#### Rufus-Air: An Open LLM Post-Training Recipe (2026-09)
- **简介**：作者为 Amazon 的 Chia-Yuan Chang、Renyuan Cheng、Rui Feng 等（按姓氏字母序）。在 GLM-4.5-Air-Base（106B-A12B）上给出开放、可复现的后训练配方，按串行八阶段组织：SFT、Reasoning RL、Coding RL、Instruction-Following RL、General Agent、Coding Agent、Search Agent 与 RLHF，披露数据、奖励设计、基础设施、阶段顺序与分阶段结果；阶段由基础到高阶、由硬可验证奖励到软的 judge 信号，基于开源组件与公开数据，无需新增人工标注或内部蒸馏 teacher。主要发现：高质量多样 SFT 奠定能力底线；难度过滤使 RL prompt 处于有效学习区间；奖励可靠性是安排阶段顺序的实用原则；基础设施与工程选择也是配方的一部分。性能超过官方 GLM-4.5-Air 后训练版本，并与同规模开源模型相当。
- **arXiv**：[2609.29421](https://arxiv.org/abs/2609.29421)

#### Towards Scalable RLVR: Multimodal Instruction Following Data Synthesis and Distillation (2026-09)
- **简介**：作者 Yirong Zeng、Yuxian Wang、Yutai Hou 等。多模态指令遵循（MMIF）主要依赖 SFT，容易停留在表层模式匹配并损害通用能力，而 RLVR 又受限于缺乏 RL-ready 的多模态数据。作者提出 **MIFS** 数据合成流水线：用生成式约束协议合成多样的原始样本，再通过 learnability-aware 的蒸馏机制依据 RL 训练动态过滤数据，并配合基于代码的 verifier 提供高精度 reward。最终数据集含 90k 样本、覆盖 8 类约束与 14 个任务域。MIFS 训练的 MLLM 在四个 MMIF benchmark 上平均提升 8.13%，训练收敛速度比使用原始数据快 3 倍，且保留了视觉核心能力、缓解了 SFT 的泛化代价。
- **arXiv**：[2609.16059](https://arxiv.org/abs/2609.16059)

#### Unlocking the Unsolvable: Teacher-Guided Curriculum for Data-Efficient RLVR (MFC) (2026-09)
- **简介**：作者 Yukang Zhu、Zhen Han（Findings of EMNLP 2026）。RLVR 中超出模型当前能力、rollout 全部失败的题目因无学习信号而被结构性浪费，却恰是最有信息量的前沿。作者用强模型的部分推理轨迹构造分级难度，并以 backward-chaining 课程逐步撤去引导，直到模型能独立解题。仅用 128 道「不可解」题训练，就在九个 benchmark 平均上匹配或超过在完整 2000 题上训练的 GRPO（约 16 倍数据效率），并显著拓展大 k 下 pass@k 衡量的推理边界。作者还发现该设置下分布偏移代价尤为严重，据此提出 **Monotone Frontier Curriculum（MFC）**，单调地把训练推向无引导求解，持续优于现有课程学习方法。
- **arXiv**：[2609.13997](https://arxiv.org/abs/2609.13997)

#### Not All Prompts Are Equal: Exploration-Guided Prompt Scaffolding for Multimodal Reinforcement Post-Training (2026-09)
- **简介**：作者 Yuanhao Yue、Qianli Ma、Chengyu Wang 等（EMNLP 2026 main）。在线 RL 中训练 prompt 的信息量差别很大（有的已饱和、有的太难），却获得相同 rollout 预算。作者提出 exploration-guided prompt scaffolding：在 MLLM 的 RL 后训练中动态调整训练 prompt 分布。核心是 **Exploration Potential Score（EPS）**，一个由 KL 正则化 policy improvement 理论导出、可直接由 on-policy rollout 统计量计算、无额外开销的 prompt 效用代理；不丢弃低效用 prompt，而是用 teacher 模型生成保持原任务意图的 scaffold 改写，把 teacher 监督重新定位为训练数据精炼而非输出模仿。与 GRPO 结合在 Geo3K 与 MMK12 上稳定优于 baseline，域内最高相对提升 9.7%，MathVision 与 MMMU-Pro 分别提升 11.5% 与 11.1%。
- **arXiv**：[2609.15051](https://arxiv.org/abs/2609.15051)

#### Voice of Reason: Reinforcement Learning for Spoken Math (2026-09)
- **简介**：作者 Timothée Weisselberger、Edouard Graves、Alexandre Défossez（COLM 2026）。语音语言模型在数学推理上落后于文本模型。作者把 RLVR 应用到 GLM-4-Voice 上：先用合成的口语问答数据做 SFT 适配领域，再做 RL。结果显示即使不增加额外推理 token，RL 也能让 GSM8K 准确率超过此前语音模型需借助额外推理轨迹才能达到的水平；与已有 streaming reasoning 技术结合后，free-form 准确率进一步提升到 74.8%，为语音原生模型的口语数学能力建立了新 SOTA。
- **arXiv**：[2609.18677](https://arxiv.org/abs/2609.18677)

#### Compositional Reasoning in Language Models under Reinforcement Learning Post-Training (2026-09)
- **简介**：作者 Yu He、Yingxi Li、Yifei Wang、Ellen Vitercik。RL 后训练显著提升了推理能力，但其对组合推理的影响尚不清楚。作者提出依赖图框架形式化组合推理，得到复杂度递增的三个组合层级，并用具有确定性 reward 与清晰组合结构的数据结构任务做实证。发现一致的「分解到组合」不对称：分解技能训练不能可靠迁移到组合任务，而组合任务训练更容易迁移回分解任务；作者给出理论解释，并在长度外推、结构分布偏移、需未见技能的迁移上进一步评估组合泛化；最后在真实工具调用 benchmark 上做 pilot study，初步显示该不对称性可延伸到实际场景。
- **arXiv**：[2609.19465](https://arxiv.org/abs/2609.19465)

#### What Does an LLM Learn from Reinforcement Learning? A Mechanistic Interpretability Perspective with Fixed-SAE Track (2026-09)
- **简介**：作者 Lingheng Du、Yiming Tang、Xufeng Duan、Dianbo Liu。RL 在表征层面究竟给模型带来什么、能否创造全新特征，现有行为层面的解释不足。作者提出 **Fixed-SAE Track**：对每个考察层训练一个共享的 SAE，在 base 模型与所有 RL checkpoint 的激活上联合训练，固定所有特征方向，从而以可解释 SAE latent 的激活严格定义表征变化，包括检测新出现的特征。在多个数据集与 RL 算法上发现：RL 引起的漂移小、渐进、概念特定且集中在后几层，主要提升少数 ladder token 以及步骤分隔、答案定界符等格式脚手架的采样率，而非重塑问题内容；把这些特征 steering 到 base 模型可恢复约 80% 的 RL 性能增益，表明 RL 主要是激发模型已有能力。作者还设计了含构造已知特征的合成 benchmark 来检验 RL 能否注入真正的新特征。
- **arXiv**：[2609.15064](https://arxiv.org/abs/2609.15064)

#### ThinkPrior: Zero-Rollout Difficulty Priors for Cold-Start Prompt Selection in RLVR (2026-09)
- **简介**：Tommy Sha、Skylar Zhai、Siqi Zhao 等 3 人。指出 GRPO 中若一个 group 的 rollout 全对或全错，其 group-relative advantage 恒为零，这类 "silent group" 不提供 reward-advantage 梯度，而均匀采样会把整轮 39% 的 rollout 花在它们身上；基于历史的 prompt 选择又必须先消耗 target-policy rollout 才能估计难度，形成 cold-start 浪费。提出 **ThinkPrior**：用一个外部 anchor 模型做一遍离线 pass，以 verifier 打分得到的 anchor pass rate 初始化 Beta 后验，从而在第一次 target-policy rollout 之前就构造出 zero-rollout 难度先验，按 expected learnability 选题并用训练结果持续更新后验，既不改 loss 也不改 optimizer。在 Qwen2.5-Math-7B、16 个 seed 上，早期 silent group 减少一半以上，前 30 step 的浪费 rollout 减少近五分之一，最终精度未检测到差异；与 DAPO 组合在同样 3840-rollout 更新预算下生成的 rollout 减少 10.6%。作者明确说明在这个 250-prompt 池上固定预算的结果是 rollout 重分配而非净节省。
- **arXiv**：[2609.09075](https://arxiv.org/abs/2609.09075)

#### CircuitLens: Reasoning Circuits as Data Selection Signals for Reinforcement Learning with Verifiable Rewards (2026-09)
- **简介**：来自北京航空航天大学（Zhuofan Chen、Ziqian Jiao、Yikai Cui、Jun Bai、Wenge Rong 等 6 人）。指出现有 RLVR 数据选择标准（难度过滤、人工筛选、reward 轨迹打分）都把数据价值当作题目的内在属性，与将要学习的模型无关。提出 **Circuit Reasoning Score (CRS)**：通过 contrastive ablation 定位 46 个 reasoning-sensitive attention head，在冻结的 base model 上单次 forward 即可计算，不需要 reward 标签也不需要 rollout。结论与直觉相反——在 Qwen2.5-Math-7B 上，circuit engagement 最低的十分位反而优于随机选择（GSM8K +2.0 pp、OlympiadBench +1.6 pp、Minerva +2.9 pp），而 engagement 最高的十分位收益更小且与中间十分位不可区分；该优势有明确边界条件：在 domain-curated 池上各选择方法互不可分，1.5B 规模下有用方向发生反转，且 reward 最低的训练条件产生最强下游泛化。属 §1.1 的 RLVR 数据选择实证分析，作者的落点是数据选择呈 regime-dependent，而非可归结为一套静态的题目质量排序。
- **arXiv**：[2609.07183](https://arxiv.org/abs/2609.07183)

#### DE-Venus: A Data-Efficient RLVR Framework for Large Language Models (2026-09)
- **简介**：来自 Shenzhi Yang、Guangcheng Zhu、Kai Tang、Zhengqing Zang 等 14 人。指出 RLVR 的实际扩展受两方面成本约束——昂贵的 on-policy rollout，以及大规模获取可靠 target 的代价；已有工作分别处理样本选择、监督不完整或标签噪声，且常把监督逻辑与分布式训练纠缠在一起，妨碍受控对比与复用。提出 **DE-Venus**，把监督视为贯穿数据准备与策略优化的演化状态，拆成三个模块：Active Data Selection 分配训练与标注预算、Weak Supervision Construction 从无标注样本导出学习信号、Training-Time Supervision Refinement 过滤或纠正不可靠监督；实现上把各方法的特定决策表达为离线数据集转移或对 target、reward、batch、advantage 的在线变换，从而在保留 verl 分布式执行契约的前提下容纳 7 种代表性方法与一条数据选择流水线。在公开基准与三个业务场景上，不同配置仅用 10% 标签或低至 13% 的相关数据即可保持或提升模型质量，部分业务配置还把观测到的收敛步数减少 63%–75%。
- **arXiv**：[2609.03324](https://arxiv.org/abs/2609.03324)

#### Gradients Know What Outcomes Don't: Unlocking Reinforcement Learning for LLM Reasoning with Gradient-Aligned Rewards (GAR) (2026-09)
- **简介**：来自 Leqi Zheng、Jinbo Su、Fang Niu、Chaokun Wang 等。指出 RLVR 的二值 outcome reward 无法区分多条正确轨迹之间的质量差异，而现有 dense reward 方案要么停留在表层启发式、要么依赖需昂贵离线标注的 PRM，都浪费了训练语料中本已存在的专家解。提出 **GAR**（Gradient-Aligned Reward），直接在策略自身的梯度空间构造信号：对 output projection layer 做截断反向传播，为每条 rollout 提取一个紧凑梯度向量，与 expert-anchor 梯度做 cosine similarity，即得到 dense、reasoning-aware 的 reward，wall-clock overhead 低于 9%；并证明该 cosine 可乘性分解为 prediction-error 与 activation-pattern 两个因子，刻画了对齐信号究竟在度量什么。在 Qwen3-4B 与 Qwen3-8B 上于竞赛级数学基准一致优于 GRPO 等 baseline，并在无领域数据的情况下迁移到 GPQA Diamond 与 MMLU-Pro（abstract 未给出具体数值）。
- **arXiv**：[2609.03342](https://arxiv.org/abs/2609.03342)

#### Locked at the Entrance, Open Inside: Where RLVR Narrows the Solution Space (2026-08)
- **简介**：来自 Qiancheng Zhou、Ruizhe Li。针对 RLVR 提升 pass@1 却收缩策略解空间、削弱 test-time scaling 收益的现象，追问 breadth 究竟在推理轨迹的哪一段丢失：是无法进入某个有效解族（access），还是进入后无法完成计算（execution）。作者选取解空间可穷举的 Countdown 任务，按首个 operand 与 operator 把解划分为离散的 entrance family，在 Qwen2.5-3B 上跑 PPO、Qwen2.5-3B-Instruct 上跑 GRPO：两种设定下 solution coverage 最多下降 67%，即便在所有 checkpoint 都能解的题上也腰斩；且收缩高度集中在入口——首个算术运算之前的 per-token likelihood 偏移比下游推理大 11–16 倍，仅提供一个未被选中的 entrance prefix 就能把低 access 解族的完成率提升一个数量级以上（PPO 下 0.018 → 0.212），说明替代解仍可执行、只是不再被启动。据此定位，表层 prompting 无法恢复多样性，而 entrance-targeted 干预有效：用早期 checkpoint 做 late-layer 参数插值使 coverage 提升 37% 且 pass@1 不降；早期步 entropy collapse 在 7B/14B 模型的六个数学基准上复现，但并非推理优化的必然产物——SFT baseline 保留超过两倍的 coverage，分阶段 SFT–DPO–RLVR 流水线也保住了早期步 entropy。属 §1.1 的 RLVR 解空间收缩现象定位分析。
- **arXiv**：[2608.29188](https://arxiv.org/abs/2608.29188)

#### Program Learning with Verifiable Rewards: Symbolic Backpropagation for Post-Training LLMs (PLVR) (2026-08)
- **简介**：来自 Vishvesh Bhat（单作者）。主张对中间步骤可验证的任务，推理能力不应封在模型权重里（不可检查、不可逐步核对、不可迁移到别的模型），而应放在权重之外、表示为由确定性与神经 primitive 组合的显式程序。提出 **PLVR**，直接从输入-输出样例学习这类程序，机制是 symbolic backpropagation：每个程序层携带 typed ontology，在输出端对 ground truth 计算 loss，再通过 primitive signature 上的类型推断把所需输入 ontology 反向传播，使 credit assignment 成为一次推导而非估计；相对 RLVR 只验证终端结果，PLVR 的 reward 是逐步的 contract verdict、在程序结构上 dense。LiveCodeBench v6 与 Tau2Bench 上，30B base model 配 PLVR 在同等预算下平均比 RL 高 27.8 分、比大一个数量级的前沿模型高 13.6 分；把 loss 引导的搜索换成同一类型可行空间上等预算的均匀采样，中位程序分从 65.6 崩到 17.5，说明优势来自反向传播而非类型系统本身。
- **arXiv**：[2608.28421](https://arxiv.org/abs/2608.28421)

#### Boosting LLM Exploration via Weak-Model Guidance in RLVR (2026-08)
- **简介**：来自北京大学等机构（Xingyu Shen、Huishuai Zhang、Peng Li、Dongyan Zhao 等 5 人）。针对 RLVR 训练中 policy entropy 下降、推理覆盖面收窄、大 k 下 pass@k 退化的问题，指出已有方法多以算法层面的正则化缓解 entropy collapse，却忽略了跨模型的非参数化扰动这条路径。方法不再只依赖模型自身的内部探索，而是强制目标模型在一个更小更弱的语言模型所生成的**部分推理轨迹（外部 prefix）**之上续写并给出答案，这些「陌生前缀」打断模型的过度自信、迫使其探索不同的推理路径；作者进一步实证研究了外部 prefix 的潜力，揭示分布差异（distributional discrepancy）对 RLVR 探索动力学的作用机制。多个数学基准上一致优于 vanilla RLVR，且增益随 k 增大愈发明显，表明推理覆盖面被显著扩展；同时无需额外 SFT、精巧 reward 设计或复杂 prompt 即可缓解 entropy collapse（abstract 未给出具体数值）。
- **arXiv**：[2608.27420](https://arxiv.org/abs/2608.27420)

#### Is Next-Chunk Reasoning RL Really Better than SFT? Revisiting Training Strategies under no-CoT Data (2026-08)
- **简介**：来自上海人工智能实验室等机构（Yinhao Tang、Youqing Fang、Yanan Sun、Kai Chen 等 11 人）。近期工作提出 next-chunk reasoning RL 来利用 no-CoT 数据（如解题范例、教科书推导等推理信息丰富但缺显式 CoT 标注的语料）：让模型生成隐式推理轨迹，并以该轨迹能否预测下一个文本 chunk 作为 reward。但既有评测主要对比常规 SFT baseline，无法区分收益来自 RL 形式本身还是仅仅来自更有效地让模型接触 no-CoT 数据。作者做受控对照，并引入一个简单却被忽视的替代方案 **Mixed SFT**——单阶段监督微调，同时在 no-CoT 与 long-CoT 数据上联合训练。结果显示 Mixed SFT 的 post-RLVR 性能上限明显高于 next-chunk reasoning RL，而训练算力少 60 倍以上，且这一优势在域内数学推理与域外推理任务上都一致；此外还指出 pre-RLVR 准确率更高并不必然带来更高的 post-RLVR 准确率，强调 no-CoT 训练策略必须放到完整 post-training 流水线中评估。属 §1.1 的训练策略对照分析（受控实证研究，不提新 RL 算法）。
- **arXiv**：[2608.23256](https://arxiv.org/abs/2608.23256)

#### Robust Code RL via Faulty-Code-Driven Test case Synthesis and Dense Reward Shaping (RobustTests) (2026-08)
- **简介**：来自蚂蚁集团（Yiwen Zhang、Xiaodong Yan、Zhenyu Huang、Jun Zhou 等 9 人）。针对代码生成 RLVR 中测试用例覆盖不足导致 reward hacking 与策略退化的问题，提出 **RobustTests** 框架，核心是 faulty-code-driven 的测试用例合成策略：利用「近似正确」的错误代码（near-correct faulty code）来暴露潜在逻辑差异，从而合成能真正区分对错的测试，并用带行为特征聚类的 validator agent 过滤无效或冗余用例；同时引入基于通过率的 stepwise dense reward 函数，缓解 false negative 并提升训练稳定性。用该流水线构造了诊断能力更强的 CodeContests+ 增强版数据集，以之对 Qwen3-32B 做 RL 微调在 LiveCodeBench 上取得 3% 绝对提升，数据已开源于 HuggingFace。
- **arXiv**：[2608.24135](https://arxiv.org/abs/2608.24135)

#### Continual Reasoning Gym: Diagnosing and Harnessing Shared Reasoning in Continual RLVR (CPR) (2026-08)
- **简介**：Lirui Luo、Guoxi Zhang、Hongming Xu、Cong Fang 等 6 人。针对多任务 RLVR（MTRL）在新任务加入时需整体重训、能力扩展成本高的问题，研究 continual RLVR——每来一个任务就在已有模型上继续更新，核心问题是这样更新的模型能否追平联合训练。作者构建 **Continual Reasoning Gym** 环境，把文本与视觉推理任务组织为五条任务序列，得到两点观察：Sequential RLVR 的遗忘其实较轻，但最终性能仍低于 MTRL；对最终性能做分解后发现遗忘只解释了差距的一部分。作者把「遗忘轻」归因于 shared reasoning（可迁移的推理结构使在一个任务上训练平均而言也支撑其他任务），并据此提出 **Continual Prompt Replay (CPR)**：回放此前任务的 prompt 并用当前策略重新生成回答，以利用共享推理改善当前与未来任务的学习；平均而言只有 CPR 达到 MTRL 级别性能（abstract 未给出具体数值）。
- **arXiv**：[2608.18574](https://arxiv.org/abs/2608.18574)

#### Ask, Condition or Abstain: Reinforcement Learning for Missing-Premise Reasoning (ACA-RL) (2026-08)
- **简介**：Yongqi Tong、Zhenyu Zhang、Zimi Liu、Mingli Song 等 11 人。指出 answer-only RL 只训练模型求解完整给定的问题，而现实 query 常缺少确定唯一答案所必需的前提；此时有用的回应并不总是拒答——模型应当追问缺失前提、以未知量为条件作答，或在没有信息性条件回答时 abstain。提出 **ACA-RL（Ask-Condition-Abstain Reinforcement Learning）**：用 reasoning-graph 引导的数据增强流水线把良构问题转成带局部缺口标注（localized gap annotations）的缺前提训练实例，再以覆盖五种可观测回应行为的结构化 reward 在这些实例上做 RL；同时发布 **Missing-Premise Benchmark (MPB)**——274 条人工校验、覆盖数学、逻辑与现实文字题的基准。在 Qwen3 与 Llama 系列模型上，ACA-RL 一致提升 MPB 表现，同时保持良构推理任务上的竞争力（abstract 未给出具体数值）。
- **arXiv**：[2608.16554](https://arxiv.org/abs/2608.16554)

#### Bootstrapping Niche Multilingual Code Translation via Reinforcement Learning with Execution-Based Verifiable Supervision (2026-08)
- **简介**：来自东京大学 Yutaka Matsuo 团队（Kouki Yuki、Jie Zeng、Takeshi Kojima、Yusuke Iwasawa 等 9 人）。代码翻译必须跨多种编程语言保持可执行行为，但神经代码翻译主要聚焦 C++/Java/Python 等少数语言，留下平行监督稀疏的小众语言 many-to-many 设定，容易产出「看似合理但不可执行」的译文。方法以执行结果作为可验证监督驱动偏好式 RL：先把可验证的种子 Python 程序扩展成经执行校验的多语言代码池，再用 base LLM 在各语言对上生成候选译文并按执行结果打标，用得到的偏好训练一个给跨语言翻译质量打分的 reward model，最后以该 reward 为信号用 GRPO 在 600 个有向语言对（25×24）上优化；并提出把 HumanEval-X 扩展到大规模 many-to-many 空间的执行式基准 **HumanEval-X++**。在 Qwen-3.5 4B 与 9B 上均一致优于未训练基线，其中 4B 模型在 HumanEval-X++ 上全语言平均提升 13%，中档语言提升 21%。
- **arXiv**：[2608.13854](https://arxiv.org/abs/2608.13854)

#### PAIR: Pairwise-Aware Inclusion Reweighting for Adaptive Rollout Allocation in RLVR (2026-08)
- **简介**：作者 Pixel Nomand、Elena Voss、Marcus Hale、Sofia Reyes（机构未标注）。指出 RLVR 的算力绝大部分花在生成成组长推理轨迹上，而现有分配器按 pointwise 的难度/效用给 prompt、rollout 或 token 分预算；作者识别出一个统计错配：未 clip 的 leave-one-out group-relative 得分梯度并非独立点贡献之和，而是 rollout 两两之间的二阶 U-statistic，因此完成一条 rollout 会揭示它与所有其他已完成 rollout 的对比，自适应地选择终止点会改变哪些 pair 项可被观测。提出 **PAIR**，把短 rollout 前缀视为对比图（contrast graph）的顶点、pair 梯度项视为边：仅看前缀的预测器估计正确性与剩余 token 成本，一个凸设计在期望后缀 token 预算下选择各前缀的续写概率，已完成顶点诱导的每条边再按其记录的联合纳入概率做逆概率加权；在条件独立 on-policy rollout 与未 clip、未标准化目标下，该估计量对完整候选 pair 梯度是 design-unbiased 的。在 Qwen3-1.7B/4B 的等算力 RLVR 实验中，PAIR 平均准确率比最强 pointwise 分配器高 +1.2 和 +1.4，同时比 full-group GRPO 少生成 51% 与 52% 的 token；frozen-population 估计量审计确认无加权的自适应选择是有偏的，而 pair-inclusion 校正能在相同后缀成本下还原完整 pair 目标。
- **arXiv**：[2608.11368](https://arxiv.org/abs/2608.11368)

#### Parameter Exploration for RLVR via Variational Learning (3PO) (2026-08)
- **简介**：来自 TU Darmstadt UKP 实验室（Vatsal Venkatkrishna、Nico Daheim、Iryna Gurevych）。指出 exploration 是 LLM RL 配方中显著影响下游性能的关键成分，但现有方法多在 action space 上调控（如 temperature scaling），只能改变输出分布的方差、无法重排 token，限制了探索范围并可能导致训练发散或停滞。本文转向 parameter-space exploration：从一个后验中采样出不同的策略，各自生成可能不同的 rollout，采样策略多样性的高低因此成为与 action-space 互补的探索调节杆；据此提出 **3PO**（Perturbed Parameter Policy Optimization）方法族，包含不同的参数采样策略与不同的 rollout 分组方式用于 reward 估计。在 OLMo-3-1025-7B 与 Qwen2.5-Math-7B 的数学推理与代码生成任务上，以近乎相同的 FLOPs 成本一致优于标准 GRPO；且使用多个参数样本时，训练中出现的 zero-advantage 组以及格式错误/错误 rollout 都持续少于 GRPO 和 action-space baseline（abstract 未给出具体数值）。
- **arXiv**：[2608.09805](https://arxiv.org/abs/2608.09805)

#### Beyond Solvability: Task Learnability as a Static Prior for LLM RL Post-Training (TrajVal) (2026-08)
- **简介**：来自阿里巴巴（Ting Zhou、Zhenqing Ling、Daoyuan Chen、Yaliang Li 等 7 人）。指出均匀任务采样在分配算力时完全不考虑任务对优化响应的差异，而已有的任务价值度量大多依赖当前 pass rate 或 reward 这类快照信号，只刻画任务在当前策略下的可解性（solvability）；然而可解性相近的任务对继续训练的正向响应可能相差很大。作者把这一残余维度定义为 task learnability——在固定 RL 后训练配置下任务对继续训练的期望正向响应，通过分析逐任务 reward 轨迹发现它在独立采样的训练上下文之间可复现，并能预测下游收益。为在训练开始前就拿到该信号，提出 **TrajVal**：一个轻量的探针式估计器，用一次短 probe 运行加两次端点评测近似逐任务 learnability，既可作为独立的静态先验用于任务采样，也可作为乘性先验叠加在已有在线调度器上。在数学与逻辑推理基准、多个模型规模上，TrajVal 相比均匀采样提升数据效率，并在与在线调度方法结合时带来互补增益（abstract 未给出具体数值）。
- **arXiv**：[2608.09217](https://arxiv.org/abs/2608.09217)

#### Don't Peek at the Answer: Outcome-Masked Group Relative Policy Optimization for Label-Free RLVR (OM-GRPO) (2026-08)
- **简介**：来自厦门大学（Yongshi Ye、Liang Zhang、Yidong Chen、Xiaodong Shi 等 5 人）。指出无标签 RLVR 常用投票共识替代 ground-truth 监督，但同一 answer-level 信号既用于估计 reward 又用于驱动 token-level 策略优化时会发生崩塌——模型倾向直接强化答案 token 而非改进推理。提出 **OM-GRPO**，将 reward 估计与策略优化解耦：对答案片段（answer span）屏蔽梯度，同时以软共识信号保留 answer-level reward，把优化压力移出答案 token；并引入 **Contrast-Augmented Reward**，在已有轨迹上做低成本成对比较来精化 reward 估计，不需额外 rollout。在多个推理基准与三个 LLM backbone 上一致优于现有 label-free RLVR 方法，并以稳定的优化过程匹配有监督 GT-reward 训练；在 Test-Time Training 设定下比 majority voting 高 4.24 个点。
- **arXiv**：[2608.03119](https://arxiv.org/abs/2608.03119)

#### Off-Context GRPO: Learning to Reason on Hard Problems using Privileged Information (OC-GRPO) (2026-07)
- **简介**：Priyank Agrawal、Aditya Modi 等（Amazon）针对 RLVR 在难题上"模型采不到任何正确解→零学习信号"的 learning cliff，提出用特权信息（如解答前缀）引导 rollout（称为 **off-context**：训练 prompt 含特权引导、但目标由原始无引导 prompt 定义）。**OC-GRPO** 对 GRPO 做最小改动：用引导 rollout 但施加 importance-corrected 目标，把更新拉回原始无引导目标，避免未校正引导训练的失稳。在标准数学推理基准上平均较 vanilla GRPO 绝对提升 3.9%（相对 13.8%），额外成本可忽略。
- **arXiv**：[2607.19313](https://arxiv.org/abs/2607.19313)

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

#### Controlling Sharpening in LLM Post-Training (KAPPO) (2026-11)
- **简介**：作者 Simon Matrenok、Nicolas Menet、Nicolas Flammarion、Andreas Krause。RL 后训练会提升准确率但降低行为覆盖度（sharpening），可能让困难任务停滞；pass@k、max@k 目标能缓解该现象但机理不清、且各需专用算法。论文闭式推导出 KL 正则化 pass@k / max@k 目标的最优策略，揭示 @k 优化会按 prompt 难度重新分配学习压力、偏向更难的问题，并证明标准 @1 目标可通过 prompt 自适应正则或奖励变换得到完全相同的最优策略。基于此提出 **KAPPO**（@k Adaptive Pressure for Policy Optimization）：经一次性 reference 校准后，把现有 @1 方法无需改动训练流程地适配为 @k。在数学推理与代码生成上的多种 RL 方法实验表明，KAPPO 可直接控制 sharpening，且单 completion 训练即可达到多 completion @k 优化器的效果、成本更低。
- **arXiv**：[2611.03146](https://arxiv.org/abs/2611.03146)

#### C-GRPO: Conformal Group Relative Policy Optimization (2026-11)
- **简介**：作者 Arya Fayyazi、Seyedarmin Azizi、Mehdi Kamal、Massoud Pedram，论文被 NeurIPS 2026 接收。标准 GRPO 使用固定 group size，在简单题上浪费算力、在难题上探索不足，而已有自适应预算方法的停止规则靠启发式、缺乏有限样本校准保证。**C-GRPO** 用基于任务相关 nonconformity score 的 conformal 阈值取代固定 rollout 预算：数学推理使用 APS 风格校准，代码任务使用 first-success 分数，并配合自动选择 miscoverage 与周期性重校准。作者证明了该停止规则的有限样本覆盖性，并证明在策略单调提升下重校准使期望训练预算非增。在数学、研究生级科学与代码共六个 benchmark 上，C-GRPO 相比固定预算 GRPO 减少 49-70% 的训练 rollout 且保持准确率，跨模型家族可节省高达 82% rollout。
- **arXiv**：[2611.05591](https://arxiv.org/abs/2611.05591)

#### Whitened Advantage Summation Policy Optimization: Correlation-Aware Aggregation for Multi-Reward Reinforcement Learning (WAS-PO) (2026-11)
- **简介**：作者 Olaitan Olaleye、Jayson Meribe、Temidayo Olaleye。LLM 多奖励 RL 常对各奖励分别归一化后按固定权重相加，但奖励相关时其共享成分会被重复计数，实际权重变成 $\Sigma^{1/2} w$ 而非设定的 $w$。**WAS-PO** 在求和前用 $\Sigma^{-1/2}$ 对归一化 advantage 做白化，使设定权重作用于去相关的成分；作者推导其对应的标量化目标，指出其对原始奖励的有效权重可为负，并给出两奖励下较轻权重奖励被反转的精确条件。在 Qwen2.5-1.5B-Instruct 的工具调用任务上，WAS-PO 的平均训练奖励在三奖励（+3.8%）与四奖励（+3.9%）设置下均高于 GDPO；但训练奖励增益并未迁移到 held-out 准确率（略低于 GDPO，不显著），且两奖励时落后 GDPO，其差距可能来自 batch 级归一化而非白化。作者对结果持相当谨慎的态度。
- **arXiv**：[2611.02609](https://arxiv.org/abs/2611.02609)

#### S3AC: Stabilized, Selective, and Scalable Actor-Critic for LLM Reasoning (2026-10)
- **简介**：作者 Shuo Shen、Guangran Cheng 等。无 critic 的 RLVR 方法（如 GRPO）缺少 token 级信用分配并依赖分组 rollout 估计 baseline，而长 CoT 的 critic 训练又难：轨迹长度覆盖不足，且新初始化的 value head 主导早期梯度。论文提出 **S3AC**（Stabilized, Selective, and Scalable Actor-Critic），给出一套实用的 critic 训练配方，训练后的 critic 提供 token 级 advantage 与 prompt 级 baseline；并基于 prompt 级 baseline 估计难度，提出 **DAPS**（Difficulty-Aware Prompt Sampling），在生成前按难度和历史结果选 prompt，减少 rollout 浪费。在 1.5B 到 397B 的稠密与 MoE 模型上，S3AC 优于现代 GRPO 变体（GRPO-X）；在 35B 上可维持单 rollout 训练，而测试的 critic-free 基线在初期提升后退化。
- **arXiv**：[2610.35518](https://arxiv.org/abs/2610.35518)

#### Priority-Preserving Advantage Aggregation for Multi-Reward Policy Optimization (PAPO) (2026-10)
- **简介**：作者 Xiangchi Yuan、Xingjian Diao 等。LLM 后训练常同时优化优先级不同的多个目标（如正确性与推理效率），但常用的加性奖励聚合在主、辅 advantage 冲突时，足够大的辅助信号可以推翻主目标诱导的偏好，论文称之为 advantage priority inversion。奖励级 gating 虽能保护优先级，却丢掉了主条件失败样本间有用的辅助区分。论文提出 **PAPO**（Primary Advantage Policy Optimization）：对主、辅 advantage 做非对称融合，二者一致时保留辅助监督，冲突时平滑衰减辅助信号，保持主目标的符号（primary-consistent ordering）。该规则适用于 GRPO 的序列级 advantage 和 PPO 的 token 级 advantage，在多领域、多模型上取得最佳主目标表现，同时辅助目标保持最佳或有竞争力。
- **arXiv**：[2610.34862](https://arxiv.org/abs/2610.34862)

#### SetGRPO: Calibrating Grader-Induced Policy Optimization (2026-10)
- **简介**：作者 Jiale Huang、Zixu Li 等。Rubric-based RL 使用细粒度 grader 反馈训练开放式任务，但 grader 的随机性会通过 GRPO 的组内归一化改变整组训练系数，引起依赖策略的更新变化。论文提出 **SetGRPO**（Set-Valued GRPO），包含三个模块：Joint Reward Set Calibration 为训练系数构造联合不确定集；Geometry-Guided Safe Recovery 将候选系数映射到当前策略的局部更新几何并确定每组最大安全恢复比例；Selective Risk-Gated Optimization 通过独立风险校准选择全局阈值，控制被接受更新的方向反转风险，同时保留原 GRPO 目标。在可交换性假设下提供有限样本覆盖保证。在三个领域十个开放式 benchmark 上，对 Qwen3.5 与 Llama3.2 系列均取得所有训练方法中最高的宏平均。
- **arXiv**：[2610.36807](https://arxiv.org/abs/2610.36807)

#### Bayes-Adaptive MDPs for Rollout Allocation to Enhance GRPO Training (AdaRoll-GRPO) (2026-10)
- **简介**：作者 Faeze Moradi Kalarde、Armin Toroghi 等。GRPO 给每个训练问题分配统一 rollout 预算，忽略难度差异：很难的题得到的 rollout 太少而难以产生正奖励信号，中等难度的题却分到过多。论文将「总预算不变下按难度分配 rollout 以最大化训练效用」形式化，提出 **AdaRoll-GRPO**：把 rollout 分配重述为序贯决策问题，用 Bayes-adaptive MDP 学习自适应分配策略；该策略离线计算，训练时无额外在线优化开销。在多个推理数据集和 LLM 架构上，固定 rollout 预算下稳定提升 GRPO 训练表现（abstract 未给出具体数值）。
- **arXiv**：[2610.37429](https://arxiv.org/abs/2610.37429)

#### CONCUR: Concordant Auxiliary Rewards for Multi-Task, Multi-Reward Reinforcement Learning (2026-10)
- **简介**：作者 Jixuan Leng、Si Si、Cho-Jui Hsieh 等。多任务 RL 中任务的主奖励常对长度奖励等辅助奖励做 gating，但组相对 advantage 估计器把被 gating 的辅助项当作普通分量，可能反转主奖励的顺序；在主奖励饱和时又只有辅助项能区分回复。保留辅助项伤害部分任务，丢弃则伤害另一些任务。论文提出 **CONCUR** advantage 估计器，用满足两个条件（主奖励更高的回复不得分更低；处于主奖励上限的回复得分相等）的最近分数替换被 gating 辅助项的标准化分数。CONCUR-T 全程应用两个条件，CONCUR-A 则对在通过的回复中与主奖励顺序一致多于相反的辅助项放宽第二个条件。在两个模型家族的三个模型、覆盖数学/代码/指令遵循/写作上，均平均优于所有基线以及丢弃辅助项的做法。
- **arXiv**：[2610.36437](https://arxiv.org/abs/2610.36437)

#### Flat-RLVR: Sharpness-Aware Reinforcement Learning with Verifiable Rewards for Reasoning Capacity in LLMs (2026-10)
- **简介**：作者 Yeonbum Kim、Hyun Kyu Lee、Sung Whan Yoon。RLVR（尤其是 GRPO）容易引起 distribution sharpening：策略分布过度尖峰，稀有但正确的轨迹被逐步压制，样本多样性下降。论文提出 **Flat-RLVR**，在 LLM 后训练中通过 sharpness-aware minimization 搜索平坦的 reward 极值区域来平滑策略分布，并可与各类 GRPO 变体结合。作者从理论上把 flat-reward RLVR 与 distribution sharpening 的减弱联系起来，实验表明在保持单次尝试准确率的同时提升样本多样性，在采样效率与推理能力之间取得更好的权衡。（abstract 未给出具体数值）
- **arXiv**：[2610.29281](https://arxiv.org/abs/2610.29281)

#### ReAPO: Retry-Aware Policy Optimization for Efficient Exploration in LLM Reasoning (2026-10)
- **简介**：作者 Nghi Dao、Calder Morton-Ferguson、Noah Golowich、Adam Block。RLVR 常用独立随机 rollout 探索，容易重复采样相似的失败策略；把错误解放入后续尝试上下文的方法引入了「如何给每次尝试分配 credit」的问题。论文从 Stochastic Shortest Paths 视角看待该 credit assignment，在适当简化假设下推出对 GRPO 的简单改动 **ReAPO**：为失败轨迹分配额外生成，使成功的重试被强化而不奖励之前的失败，并以理论上有依据的方式给各次尝试分配 credit。在 0.6B–4B 的 Qwen3 与 OLMo-2 模型、八个数学与科学推理 benchmark 上提升首次尝试表现；在 Qwen3-0.6B 上优于两倍 group size 或两倍训练步数的 GRPO，分别少用 39% 和 31% 的生成 token 即达到峰值准确率。
- **arXiv**：[2610.28770](https://arxiv.org/abs/2610.28770)

#### FrameGRPO: Incentivizing Video Reasoning via Multi-Frame Rollouts and Temporal Credit (2026-10)
- **简介**：作者 Jiaxuan Sun、Biao Wang、Shuai Li。标准 video GRPO 只改变语言轨迹，而让组内每个 rollout 都基于同一批采样帧，当共享视觉证据不足或有干扰时会产生同质 reward，削弱组相对学习。论文提出 **FrameGRPO**，无需辅助网络或改动模型结构，包含两部分：Multi-Frame Rollouts——仅对单次最大帧数解码做均匀时间下采样，构造对齐视图以丰富视觉证据与推理轨迹；Gated Temporal Efficiency Credit——当固定的满帧参考子集内所有 rollout 都失败时，才选择性强化正确的低帧率轨迹，而不无条件偏好更少帧。在多种视频理解与推理 benchmark 上稳定优于标准 GRPO，说明时间视图多样化是视频 MLLM 中探索与 credit assignment 的有效机制。（abstract 未给出具体数值）
- **arXiv**：[2610.26022](https://arxiv.org/abs/2610.26022)

#### Preserving Reasoning Diversity in Reinforcement Learning with Verifiable Rewards via Adaptive Replay (TAR) (2026-10)
- **简介**：作者 Wen-Zhi Li、Sainyam Galhotra、Claire Cardie。RLVR 提升推理能力的同时常降低推理行为多样性。作者从有限样本学习动态的角度分析，指出有限的 on-policy rollout 会逐步压制语义不同但 reward 相同的推理模式，一旦某模式被欠采样，标准 on-policy RLVR 很难再恢复。据此提出 **TAR**（Targeted Adaptive Replay），保留基座策略中已有的多样正确推理行为：构建由多样基座模型解答组成的紧凑 memory bank，并通过自适应机制仅在已存储行为在 RLVR 中被压制时才选择性重放。在六个数学推理 benchmark 与两种 Qwen2.5 规模上，TAR 比已有保持多样性的后训练方法在推理表现与多样性之间取得更好的平衡，并显著减少对基座正确解答的遗忘。代码已开源。
- **arXiv**：[2610.24236](https://arxiv.org/abs/2610.24236)

#### MENTOR: Learning from Failure with On-Policy Pedagogical Intervention (2026-10)
- **简介**：作者 Guanlan Yue、Hang Yu、Bingchang Liu 等。RLVR 容易出现「zero-reward collapse」：任务难度骤增使学习信号消失，丢弃失败轨迹则浪费了其中的诊断信息，而已有的预定义提示库等方案属于静态的 off-policy 式干预。论文提出 **MENTOR**，一个 on-policy 的多 agent 框架，充当实时导师：实时诊断失败根因，动态生成针对性的指导 prompt 引导学习 agent 的下一次尝试，可作为即插即用模块嵌入 GRPO 等流程而无需改动训练基础设施。在数学与代码生成上有效缓解 zero-reward collapse，优于 SOTA 基线，Llama-3.2-3B-Instruct 相对基座模型平均相对增益最高达 78.3%。
- **arXiv**：[2610.13594](https://arxiv.org/abs/2610.13594)

#### Generalized Proximal Policy Optimization: Unifying LLM RL Algorithms from the Perspective of Divergence (GPPO) (2026-10)
- **简介**：作者 Zhicheng Cai、Xinyuan Guo、Wei-Ying Ma 等。PPO 的 clipping 代理在几何上退化，不能自适应策略变化的阻力，导致过早裁剪和探索退化；后续变体多为零散的启发式改动。**GPPO** 从 divergence 视角统一 LLM RL 算法：建立带单调性能提升保证的广义 f-divergence trust-region 理论，说明不同 f-divergence 诱导不同几何与更新动力学，并引入比常规单样本 Monte Carlo 更忠实的 divergence 估计器拓扑。PPO、DAPO、DPPO、RIPO 等都是其特例，形成「Divergence x Estimator」的二维设计空间。在八个竞赛级数学与代码基准上，最佳 GPPO 实例相对 GRPO 在数学和编程上最高提升 50% 和 14%。
- **arXiv**：[2610.13015](https://arxiv.org/abs/2610.13015)

#### GRPODropout: Less is More for Online Reinforcement Learning Rollouts (2026-10)
- **简介**：作者 Hexuan Deng、Zihao Yan、Xuebo Liu、Shuo Nie、Yue Wang、Chen Wang。GRPO 等 RL 方法会出现策略熵坍缩，削弱探索。现有办法多在算法层面（奖励修改、熵/KL 正则）或 token 级重加权，论文从互补视角出发：改变哪些生成的 rollout 参与策略更新。同样采样预算下并非所有 rollout 都对更新有正贡献，选择性剔除可改善学习。**GRPODropout** 在标准更新前，选择性移除少量高概率的正优势 rollout，并对保留的 advantage 重新居中，只改变 rollout 的使用方式、计算开销可忽略，并配有 rollout 级理论分析指导设计与阈值选择。实验显示相较原始 GRPO 准确率更高、actor 熵更高，且更新使用的 rollout 更少（abstract 未给出具体数值）。
- **arXiv**：[2610.11854](https://arxiv.org/abs/2610.11854)

#### All Work And No Play Makes Jack a Dull Boy: Understanding and Preventing Catastrophic Strategy Collapse in RLVR (Mesh Learning) (2026-10)
- **简介**：作者 Qiyuan Huang、Tianshi Xu、Meng Li。RLVR 后训练中 GRPO 类算法可能出现严重的后期坍缩；prompt 探测表明这并非良性的策略剪枝，而是有效策略容量的有害收缩，使不同推理策略越来越难触达。论文通过轨迹级策略更新交互定义策略，建立结合优化动力学与信息论的统一理论，证明主要 RLVR 目标会逐步把概率质量集中到单一策略，而维持非平凡任务准确率需要最低策略容量，二者冲突解释了灾难性坍缩。进一步推导 Mirrored Entanglement Index (MEI) 作为轻量在线预警信号，并提出 **Mesh Learning**，暴露多个推理策略并防止任一策略主导优化。在 AIME26、AIME25、MATH-500、GPQA、LiveCodeBench 上，对 Qwen 与 Phi 家族分别最高提升 13.4 pp 和 11.5 pp。
- **arXiv**：[2610.02835](https://arxiv.org/abs/2610.02835)

#### Decoupling Exploration from Optimization in RLVR (ExpDis) (2026-10)
- **简介**：作者 Saif Punjwani、Micah Goldblum。RLVR 的一大期望是发现新的推理策略，但在奖励中加入强烈的新颖性激励往往收效有限，甚至损伤模型质量，而可验证奖励只监督模型知识与行为的狭窄切片，这类退化难以恢复。作者提出 **Exploration-Distillation (ExpDis)**，将探索与优化解耦：训练一个或多个带新颖性奖励的 explorer 策略，对其轨迹做正确性与质量过滤，再蒸馏到单独的 student 策略，student 训练时不含新颖性奖励，多轮交替进行。这样可以激进地扩展探索而不损害 student。在七个数学推理 benchmark、两个模型家族上，同等墙钟预算下 ExpDis 优于 DAPO，pass@k 扩展性也更好，说明生成的正确解更多样。
- **arXiv**：[2610.10536](https://arxiv.org/abs/2610.10536)

#### Follow the Winners: Conservative Policy Improvement with the Cross-Entropy Method for Critic-Free RFT (FTW) (2026-10)
- **简介**：论文被 NeurIPS 2026 接收为 poster，作者 Joery Ariën de Vries、Neil David Lawrence、Zhenwen Dai。agentic LLM 的 critic-free RFT 常用 GRPO 类方法，用重复 rollout 的 group baseline 降方差，但在 live service、安全沙箱等有状态环境中难以重复 rollout，且激进更新会固化长而稀疏验证的轨迹噪声。**Follow the Winners (FTW)** 把 cross-entropy method 引入 RFT，用 replay buffer 样本上的序数过滤器取代 group rollout，对回报的次序统计量产生多项式集中。作者从 control-as-inference 视角推导 FTW，同时恢复出 GRPO 与 DPO 作为特定建模选择，指出 GRPO 是风险中性、DPO 与 FTW 共享有界的风险寻求偏置。扩展到 agentic LLM 后训练，FTW 在 Sokoban 与 Search-R1 基线上与 GRPO、PPO 持平，把对 value model 或 group rollout 的依赖换成 CPU 内存。
- **arXiv**：[2610.03361](https://arxiv.org/abs/2610.03361)

#### Tropical Reinforcement Learning (TROPIC) (2026-10)
- **简介**：作者 Arip Asadulaev、Aladin Djuhera、Holger Boche、Fakhri Karray、Martin Takac 等指出，LLM 的 RL 通常最大化期望回报，即把所有成功轨迹的概率相加；这只能反映成功频率而非哪个解有效，且因概率和为一，强化一个解会使模型遗忘从未被证伪的另一个解，不利于需从不同（常失败的）尝试中拼合推理步骤的组合式推理。提出 **Tropical Reinforcement Learning**，改变代数：对备选解的概率取最大值而非求和，得到 tropical semiring，状态价值即其最可能已验证解的 log 概率，并附带可回放、复用的显式路径，使来自不同 rollout 的最优前缀与后缀可在共享状态处拼接。实现算法 **TROPIC** 面向确定性、可重置、结果可验证的环境；在 Sokoban、Countdown、FrozenLake、WebShop 四个 agentic 任务上最多比最强 on-policy 基线高 16 个百分点。
- **arXiv**：[2610.02478](https://arxiv.org/abs/2610.02478)

#### MaPP: A Unified Marginalized Posterior-Predictive Framework for Data-Efficient RLVR (2026-09)
- **简介**：作者 Yangyang Ren、Haodong Zhu 等指出 RLVR 的 rollout 与策略更新成本高，在线 prompt 选择虽可用每个 prompt 的 Bayes 后验预测难度，却忽略了从采样响应中提取学习信号的可靠性：GRPO 中响应的 advantage 经 group 归一化而依赖同组其他随机采样的结果，由此产生 composition noise，对梯度估计误差设下不可消除的下界。提出 **MaPP**（Marginalized Posterior-Predictive）：通过闭式 Beta-Binomial 边缘化，用与组成无关的内在 advantage 取代标准 group-relative advantage，其误差随后验集中而可证地减小；并用同一后验导出考虑不确定性的 prompt 选择分数，无需额外 rollout 成本。在数学、规划、视觉几何任务和五种骨干上稳定优于 GRPO 与强选择基线，同等 rollout 预算下比最强基线平均精度最多提高 +2.45。
- **arXiv**：[2609.34990](https://arxiv.org/abs/2609.34990)

#### Save Your Saturated Data: Learning Beyond Reward Saturation in Group-Based RL (2026-09)
- **简介**：作者 Ziyuan Yang、Yike Wang、Shangbin Feng、Yulia Tsvetkov 指出，随着模型变强，训练数据会 reward 饱和：同一问题的所有采样响应都得到同样高的 reward，group-relative 信号消失、数据失效。论文研究能否从饱和数据中恢复学习信号，在数据、rollout、reward、advantage 四个层面做干预，并仅用饱和推理数据做大量 RL 训练。结果显示 rollout 生成层面的干预最稳定：引导策略生成「高质量但错误」的解，使饱和组中出现低 reward 的负样本，使 Qwen3-1.7B 与 4B 上 GRPO 提升 6.4% 至 9.0%；提高 rollout 温度或加入辅助 reward 也能恢复非零 advantage，但收益较不稳定。进一步分析表明有效负样本需要有信息量，方法与未饱和数据并用仍有效，并支持对新饱和样本迭代回收。
- **arXiv**：[2609.33126](https://arxiv.org/abs/2609.33126)

#### CorrGRPO: Correlation-Normalized GRPO for Multi-Reward Learning (2026-09)
- **简介**：作者 Wenbin Hu、Huihao Jing、Haochen Shi、Yuxuan Liu、Haoran Li、Yangqiu Song 指出，多 reward 场景下 GRPO 把各 reward 分量求和再按组内标准差归一化，该方差等于所有成对 reward 协方差之和；大尺度且相关的 reward 会主导归一化，压制小尺度 reward 的信号。提出 **Correlation-Normalized GRPO（CorrGRPO）**：把成对协方差归一化为 Pearson 相关系数，保持中心化总 reward 不变，同时平衡不同尺度 reward 对基于相关性归一化的影响，使 advantage 幅度能随 reward 相关性自适应而不被大尺度分量主导。在代码生成、工具调用与 agent 安全三个领域、0.5B–8B 模型上与 GRPO 及其他变体对比，三个领域均有提升（abstract 未给出具体数值）。代码已开源。
- **arXiv**：[2609.36820](https://arxiv.org/abs/2609.36820)

#### EasyPPO: Stabilizing the Critic Is Key (2026-09)
- **简介**：作者 Xuanyi Zhou、Qiuyang Mang、Huanzhi Mao、Dacheng Li、Wenhao Chai 等发现 PPO 的 critic 也是 LLM RL 不稳定的主要来源，并识别两种失效模式：一是对 actor 和 critic 同时过滤被截断的 rollout，会把目标变为「完成条件下的 reward」，导致截断率可能上升而条件 reward 仍在提升；二是异质的回报噪声使高方差 prompt 在有限 batch 中主导 critic 更新。**EasyPPO** 对应提出：仅对 actor 做 overlong 过滤，使 critic 在完成与被截断 rollout 的回报上训练；噪声归一化 critic 回归，按采样回报标准差的倒数加权各 prompt 的 critic loss；以及适度更小的 critic mini-batch。在 FrontierCS 连续 reward 代码、AIME24 二值 reward 数学推理与 Search-R1 多轮搜索上全程稳定，稳定优于 vanilla PPO、VAPO、HL-Gauss PPO，最佳验证分相对 PPO 分别提升 14.89%、2.28%、9.47%。
- **arXiv**：[2609.36802](https://arxiv.org/abs/2609.36802)

#### GVPO++: Group Variance Policy Optimization for LLM Post-Training and On-Policy Distillation (2026-09)
- **简介**：作者 Kaichen Zhang、Yuzhong Hong、Junwei Bao 等，为 NeurIPS 2025 论文 GVPO 的扩展版。GRPO 等方法依赖重要性采样，训练不稳定。**GVPO** 将 KL 约束奖励最大化的解析解融入梯度加权，其梯度对应隐式奖励中心距离与真实奖励中心距离之间的均方误差，保证唯一最优解恰为 KL 约束奖励最大化目标，并允许灵活采样分布而无需重要性采样。扩展版进一步证明 GVPO 可自然推广到 on-policy distillation (OPD)，并能优化一整族扩展的 OPD 目标，为目标设计提供原则性基础（abstract 未给出具体数值）。
- **arXiv**：[2609.21432](https://arxiv.org/abs/2609.21432)

#### DEEPO: Dual-Entropy Enhanced Policy Optimization for Hallucination in MLLMs (2026-09)
- **简介**：作者 Yingxuan Zhuang、Miao Pan、Wangjie Gan 等。RL 提升 MLLM 推理，但对幻觉的影响不均。论文指出从奖励到参数更新的两个薄弱点：rollout 层面，高语义熵的难题常产生全错的样本组，使组相对 advantage 归零；优化层面，自信但错误的 token 因分布变尖而梯度几乎为零。**DEEPO** 提出双阶段增强：由语义熵触发的专家前缀注入 grounded 续写，恢复 advantage 方差；并用 advantage 符号感知的 Renyi 预条件化缓解 logit 饱和。两个分支各自优于 GRPO，在最复杂的长程任务 VideoMMMU 上交互效应显著（+4.0，95% CI [1.1, 6.9]），在降低幻觉的同时保持准确率与训练稳定。
- **arXiv**：[2609.28570](https://arxiv.org/abs/2609.28570)

#### Informed Masking: Structure-Aware Perturbation for Reinforcement Learning in Diffusion Large Language Models (2026-09)
- **简介**：作者 Xiaoyi Yu、Enver Sangineto、Pei Fu 等（EMNLP 2026 Findings）。对扩散大语言模型 (dLLM) 做 RL 对齐需要在小的 Monte Carlo 预算下通过掩码重建子问题估计似然代理，现有方法用均匀随机掩码。论文发现 dLLM rollout 中存在上游/下游 token 结构：揭示某些 token 会引起邻近未解码位置置信度大幅变化（上游），其余则只引起小的局部变化（下游），且掩码下游 token 得到的子问题明显更良定（子问题难度不对称）。提出 **Informed Masking (IM)**：从去噪轨迹零额外推理成本地导出逐 token 优先级，并把掩码采样偏向下游 token。作为即插即用模块用于三种 SOTA dLLM RL 方法（LLaDA-8B-Instruct），在数学与规划基准上相对平均增益最高分别为 2.01%、8.68%、5.77%，训练更稳定。
- **arXiv**：[2609.25927](https://arxiv.org/abs/2609.25927)

#### MaD-RL: Matching Distributions for Calibrating LLMs with Reinforcement Learning (2026-09)
- **简介**：论文指出标准 RL 后训练只最大化单个输出的 reward，而合成数据生成、公平性约束、策略探索等场景需要控制「跨多次生成的输出分布」。作者提出 **Distribution Matching** 框架（MaD-RL），让模型输出的某个潜在类别属性分布匹配指定目标分布；实证显示 GRPO 等主流后训练配方会把策略概率集中到单一模式、降低多样性，而熵正则与采样温度只作用于 token 空间且只能趋向均匀分布。作者证明已有工作是基于 $L_2$ 散度的特例，并为 KL、Jensen-Shannon 等散度设计了相应 reward 函数并给出理论依据，在数学推理与编程实验中验证了有效性（abstract 未给出具体数值）。
- **arXiv**：[2609.31644](https://arxiv.org/abs/2609.31644)

#### Bellman Policy Optimization (BPO) (2026-09)
- **简介**：作者 Zhuoqing Song、Haotian Xu、Xikun Zhang、Lidong Bing。**BPO** 是一种从 Policy Mirror Descent（PMD）推导出的 critic-free RLVR 方法。针对带终端 reward 的自回归生成，BPO 利用 Bellman 方程把 PMD 重写为 trajectory 级目标，从而无需估计中间状态的 value，并证明其与原 PMD 目标具有相同的唯一最优解。实际的 BPO loss 通过对该目标做近似得到，其 mismatch 修正权重是互补 token 概率的平滑比值。在数学推理 benchmark 上验证了方法有效（abstract 未给出具体数值）。
- **arXiv**：[2609.15987](https://arxiv.org/abs/2609.15987)

#### Rethinking Critic Learning in PPO: Understanding and Mitigating Value Flattening (SP3O) (2026-09)
- **简介**：作者 Yizhuo Li、Jianhao Yan 等。LLM 的 PPO 通常用 critic 估计状态价值以降低方差，但作者发现一种系统性失效模式 **Value Flattening**：由多条 Monte Carlo 续写估计出的中间状态价值变化剧烈，而 critic 预测却相对平坦；在 FrozenLake 中也观察到该现象，且状态空间越大越明显。理论与实证分析将其归因于 critic loss 中隐含的方差惩罚，以及时间相关状态带来的冗余更新。据此提出 **SP$^3$O**（SParse Proximal Policy Optimization），只对每条回答中少数几个彼此分离的状态施加 value loss。在 Qwen3-Base 上，每条回答仅监督三个状态即可缓解 Value Flattening，并在不同模型规模与评测集上持续提升策略。
- **arXiv**：[2609.18708](https://arxiv.org/abs/2609.18708)

#### Learning to Solve Hard Problems in RL for LLMs by Never Giving Up (NGU) (2026-09)
- **简介**：作者 Michael Noukhovitch、Hamish Ivison、Nathan Lambert、Aaron Courville。作者指出 LLM 的 RL 在数据集上提升并不均衡：容易题大幅提升、难题提升很小，称为 RL 中的「马太效应」；原因之一是现代 RL 方法在容易题上浪费过多算力。提出 **Never Give Up（NGU）**：对每个问题持续采样直到出现正确答案，借助异步 RL 自然地用更少样本过滤简单题、把算力动态重分配给难题，并研究了 off-policy 鲁棒性等设计选择、总结最佳实践。在 DeepScaleR 数学数据上提高了单位算力的性能，尤其是难题；在编程任务 Manufactoria 上，标准 GRPO 搭配 per-test reward 无法完全解决，而 NGU 能逐步解出更难的测试直至完全解决。
- **arXiv**：[2609.13443](https://arxiv.org/abs/2609.13443)

#### CanvasAnneal: Curriculum Reinforcement Learning for Diffusion Language Models (2026-09)
- **简介**：作者 Blake Olson、Yuhang Song 等。扩散语言模型（DLM）在复杂推理与工具使用上落后于自回归模型，而标准 RL 在 DLM 上存在探索瓶颈。**CanvasAnneal** 是课程式的 diffusion RL 框架：RL 初期把 teacher 生成的推理轨迹注入初始扩散画布以 warm-start 探索，随训练推进逐步撤去引导，要求模型独立生成越来越多的推理轨迹。在数学推理与工具使用 benchmark 上，MATH500、Countdown 和 Tau2 上优于标准 diffu-GRPO，并在若干任务上显著加快 reward 提升，但收益因任务而异。
- **arXiv**：[2609.13060](https://arxiv.org/abs/2609.13060)

#### Forty Shades of Blue: Quality-Diversity Alignment via Mode-Conditioned Reinforcement Learning (MoDA) (2026-09)
- **简介**：作者 Jiayi Yuan、Hangoo Kang、James Jihao Liu、Yejin Choi、Vikram Iyer、Liwei Jiang 等。LLM 对齐训练的副产物是 mode collapse：输出多样性逐步丧失。**MoDA**（Mode-conditioned Diversity Alignment）是一种联合优化质量与多样性的在线 RL 后训练算法，受 MARL 协同视角启发：训练一个以抽象编号角色为条件的共享 LLM 策略，每个角色视为一个 agent，竞争产出彼此不同的输出；并用 prompt 自适应的质量门控机制标定参考质量阈值，仅对达标响应给予多样性 reward，防止 reward hacking。相对 Qwen3-8B baseline，Infinite-Chat 上 SBERT 多样性提升 265%，通用能力 pass@1 平均提升 10.3%；相比最强 baseline DivPO，SBERT 多样性由 0.274 提升至 0.482。
- **arXiv**：[2609.14896](https://arxiv.org/abs/2609.14896)

#### Difficulty-Adaptive Tree-Structured Policy Optimization for Expanding Reasoning Coverage in RLVR (DATPO) (2026-09)
- **简介**：来自 POSTECH（Youngjun Yu、Sanghwan Jang、Hwanjo Yu）。指出 RLVR 显著提升单样本准确率，却常因训练期探索不足而无法扩大模型的内在推理覆盖（pass@k），因此转而优化训练期 rollout 的结构设计。分析给出三条原则：difficulty-adaptive rollout 不只是效率启发式，本身对扩大 pass@k 有重要作用；tree-based rollout 在发现正确答案上优于 parallel sampling；sentence-entropy 引导的 forking 能克服 token-level branching 的 localization 现象、最大化语义多样性。据此提出 **DATPO**，将 difficulty-adaptive tree search 与 sibling-diversity advantage 项结合，在训练中显式促进语义多样性以扩大推理覆盖。数学推理基准上尤其在 pass@k 上优于基线，并直接转化为更好的 test-time scaling 表现（abstract 未给出具体数值）。
- **arXiv**：[2609.08650](https://arxiv.org/abs/2609.08650)

#### Stable-MM-R1: Anchoring Multimodal Reasoning Dynamics via Entropy-Guided Stratification (2026-09)
- **简介**：Yimeng Ye、Shuang Chen、Wenxuan Huang、Manyuan Zhang 等 11 人。针对多模态推理 RL 流程中的训练不稳定与 entropy 快速坍塌，作者将其归因于标准采样过程中的 "Rollout Silencing" 与低质量梯度信号，提出一套以数据为中心的稳定化框架。**PAQM**（Potential-Aware Query Mining）动态过滤数据，把训练集中到能力激发潜力高的 "Distillation Zone" 样本上；**HSR**（Hybrid Stratified Replay）则按 Path Entropy（一种 rollout 级置信代理）与 outcome reward 对 rollout 分层重组 batch，在单个优化 step 内复用当前策略产生的 "Stability Anchors" 与 "Hard Negatives" 构造高对比度的优化组，并在进入下一 step 前清空 buffer 以保持 on-policy。该方法在复杂推理任务上超过强基线，在有限算力下同时缓解 entropy 坍塌并提升学习信号利用率（abstract 未给出具体数值）。
- **arXiv**：[2609.07148](https://arxiv.org/abs/2609.07148)

#### One Step, One Lead: Mitigating Higher-Order Interference in Multi-Domain Reinforcement Learning via Cross-Step Control (OSOL) (2026-09)
- **简介**：Zihan Lin、Xiaohan Wang、Jie Cao、Jiajun Chai 等 7 人。指出跨领域联合 RL 虽能拓宽 LLM 推理能力，却常损害单领域性能并破坏优化稳定性；已有工作用一阶梯度对齐或曲率代理从单步视角诊断干扰，会漏掉一类序列性干扰——同点处的各域梯度可以近乎正交，而连续两次真实更新却在输出空间中部分相互回撤。作者进一步说明，相邻 checkpoint 之间的 token log-probability footprint 可直接在输出空间恢复这一局部二阶交互，无需显式重建同步曲率。据此提出 **OSOL**：每轮指定一个 focus domain，用前一 checkpoint 的 footprint 对 token 级 rebound risk 排序，并在标准 GRPO 更新内施加 drift-ranked、自适应缩放的修正，以抑制被定位的 cross-step 输出回撤分量。受控实验显示 cross-step backtracking 比同点梯度诊断更能预示后续任务损伤，前一 checkpoint 的 footprint 对未来 rebound risk 的排序也比 Hessian 类代理更准。在 Qwen3-30B-A3B 上，OSOL 取得 0.4822 的 domain-macro 平均分，比最强对比基线提升 5.7%，且不需要显式高阶微分。
- **arXiv**：[2609.06469](https://arxiv.org/abs/2609.06469)

#### Spurious Advantage Hidden in GRPO (SIGNBALANCE) (2026-09)
- **简介**：来自 Jiamian Wang、Samyadeep Basu、Koustava Goswami、Tong Yu、Zhiqiang Tao。指出 GRPO 的 advantage estimator 仅凭组内 reward 统计给每条 rollout 赋 magnitude，于是靠猜中答案的 rollout 与靠推理得到答案的 rollout 表面一致、同样拿到高 magnitude，作者称之为 spurious advantage；它出现在三类场景：候选集很小的 bounded-answer 任务、open-answer 集合中嵌套的 bounded 子情形，以及预算充足使多条路径都能落到同一答案的 search agent，三者都会把策略引向 guess-like 行为。提出 **SIGNBALANCE**，其 magnitude 是 composition-free 的：保留 verifier 给出的符号、改用全局 scale，并通过 stop-gradient 的 per-class rescaling 恢复零均值平衡。在不同规模的数学与 search agent 基准上，SIGNBALANCE 在 open-answer 数学上与 GRPO 持平，在 bounded-answer 数学与 search agent 上有提升（abstract 未给出具体数值）。兼具 GRPO advantage 机制的失效诊断与算法修正。
- **arXiv**：[2609.04063](https://arxiv.org/abs/2609.04063)

#### Group Adaptive Clipping Policy Optimization (GAPO) (2026-08)
- **简介**：来自 Sheng Jia、Xiao Wang、Shiva Prasad Kasiviswanathan、Rein Houthooft。指出 group relative 类 RLVR 方法对所有 rollout 使用固定的 importance-sampling ratio clip 边界，导致难题上稀有的正确 rollout 与易题上大量的正确 rollout 被以相近比率 clip，尽管两者提供的学习信号完全不同：组内成功率低的 rollout 具有更大的 IS ratio、对探索与攻克新问题携带更强梯度信号，却被固定 clip 不成比例地压制。提出 **GAPO**，一个可插入现有 GRPO 类方法的改动，把 clipping 边界按 rollout advantage 自适应调整；其动机来自 reverse-KL trust-region 视角——学习信号更大的 rollout 应获得成比例更大的更新余量。GAPO 不需要 reward shaping，保持标准 PPO/GSPO surrogate 不变，仅调整 clip 阈值。在 Qwen 与 Llama 系列上，于 base model 通过率较低的数学推理与代码基准上，Pass@1 与 Pass@k 均一致优于固定 clip 与 advantage-shaping baseline（abstract 未给出具体数值）。
- **arXiv**：[2609.00444](https://arxiv.org/abs/2609.00444)

#### Learning from Hard Prompts: Difficulty-aware Advantage Amplification in Dynamic Sampling (DA3PO) (2026-08)
- **简介**：来自 Siyuan Gan、Yuhan Li、Xiran Wang、Linjian Meng 等。DAPO 相对 GRPO 的增益主要来自 Dynamic Sampling——通过过滤掉采样响应全对或全错的 prompt 来消除零 advantage 带来的零策略梯度。但作者的理论分析表明它反而降低训练效率：Dynamic Sampling 对同一 prompt 下不同响应的 advantage 放大是非对称的，在 hard prompt 上错误响应被放大得比正确响应更多，于是模型更倾向于回避已观测到的错误响应，而不是充分利用那些难以采到的正确响应。为此提出 **DAA**（Direct Advantage Amplification），直接放大 hard prompt 上由 Dynamic Sampling 得到的难采样正确响应的 advantage；将 DAA 并入 DAPO 得到 **DA3PO**，在 DAPO 基础上不到 30 行代码即可实现。实验显示 DA3PO 显著优于 GRPO 及其他经典 GRPO 变体（abstract 未给出具体数值）。
- **arXiv**：[2608.27982](https://arxiv.org/abs/2608.27982)

#### When Do Larger Batches Help Scale LLM Reinforcement Learning? (2026-08)
- **简介**：来自 Ziniu Li、Jinbo Wang、Guanhua Huang、Feiyuan Zhang 等。追问大 batch 降低随机梯度方差的统计收益是否真能转化为更短的 wall-clock time-to-target，并把算法效应与系统效应沿各自自然坐标拆开：算法层在等累计样本数下重调依赖 batch 的超参，得到一个在有界 batch 范围内近似 batch-size-invariant 的配置族，其成员的样本索引学习轨迹几乎重合；系统层利用 rollout 生成与训练的计算不对称性——自回归生成在低并发下受 memory-bandwidth 限制，而训练开销近似随处理 token 数增长。两者结合给出一条直接决策规则：只有当吞吐增益超过 samples-to-target 的惩罚时，更大 batch 才会缩短 time-to-target。GRPO 与 PPO 实验支持该分解的两侧：Adam 下学习率按平方根缩放可在有界范围内产生近似 batch-size-invariant 的学习曲线；固定硬件上更大 batch 使生成吞吐最高提升 2.29×；GRPO 中把更高吞吐与学习率重调结合可把 time-to-target 缩短最多 29%，而只增大 batch 不重调则尽管吞吐更高反而更慢。属 §1.2 的 RL 训练 scaling 实证分析。
- **arXiv**：[2608.29296](https://arxiv.org/abs/2608.29296)

#### PAC: Progress-Augmented Advantage Curriculum for Multi-Task Reinforcement Learning of LLMs (2026-08)
- **简介**：来自 Yuanqiang Yu、Yanzhao Zheng、Zhentao Zhang、Tianze Xu 等。指出多任务 RL post-training 多依赖固定或人工设计的任务混比，忽略了任务有用性随训练推进而变化；而在线 curriculum 方法通常只用更新幅度定义 learnability，不看更新是否真的转化为 reward 增益，容易把 rollout 预算错配给「更新大但无效」的任务。提出 **PAC**，联合两个任务级信号：由 advantage 导出的 learnability（衡量任务能诱导多大的策略更新）与近期 reward gain（衡量这些更新是否确实改善了任务表现），再由一个 Bayesian Thompson Sampling 控制器在 GRPO 训练过程中据此跨任务分配 rollout。在 multi-level 推理与 multi-domain 推理两种设定下，PAC 用更少 rollout 步达到相当的验证分数，最终平均分也高于随机采样与仅基于 advantage 的 curriculum baseline（abstract 未给出具体数值）。
- **arXiv**：[2608.30528](https://arxiv.org/abs/2608.30528)

#### Beyond the Stability-Exploration Dilemma: Environmental Regularization for LLM Policy Optimization (ERPO) (2026-08)
- **简介**：Xianlei Zhou、Xiangdi Meng、Yu He、Qika Lin、Jun Liu 等 10 人（代码开源于 AlibabaResearch）。指出 LLM 策略优化的稳定性—探索权衡目前由 action 侧的 Policy-KL 正则调停，这让实践者两难：保留 Policy-KL 会约束回复行为、吃掉 action 侧的探索预算，去掉它则失去显式的漂移控制。作者主张把正则化搬到输入侧：随训练推进，当前策略在训练 query 上诱导的分布会相对 pre-RL 参考分布无约束漂移。**ERPO** 因此引入 **Query-KL（QKL）** 项来界定 query 分布偏移，并配一个由参考模型导出的、数据集静态的 per-query 权重，使每个 query 的更新偏向在参考分布下更典型的 query。关键性质是 QKL 的梯度严格只经 query likelihood 流动，policy-gradient 估计器所用的 response score function 不出现在 QKL 中，因此 QKL 对回复分布不施加直接梯度压力，探索得以保留；ERPO 可插入 GRPO/PPO/REINFORCE 式流水线且不增加额外前向。在 6 个数学推理基准上替换标准 Policy-KL 后有效控制了 query 分布漂移，在高温解码与长周期训练下准确率更高、行为明显更稳定（abstract 未给出具体数值）。
- **arXiv**：[2608.23311](https://arxiv.org/abs/2608.23311)

#### Perturb the Thought, Not the Pixels: Latent-Space Rollout Diversification for Reinforcement Learning of Vision-Language Models (NC-GRPO) (2026-08)
- **简介**：Michael Jerge、Joseph Pelczar、Justin Downes。RLVR 能提升 VLM 推理能力，而在每个优化组内提高 rollout 多样性可放大其收益；已有做法通过解码温度或像素空间图像扰动来制造多样性，本文追问扰动是否应该放进模型的 latent space。**NC-GRPO（Noise-Contrastive GRPO）** 对每个 rollout 组中的一半样本，在 prompt 编码阶段的最后一层 hidden state 注入尺度校准过的高斯噪声，使这些 rollout 从一个被位移的起始状态分叉；那些尽管被位移仍能到达正确答案的分支相对被带偏的分支被强化，从而把分叉点处的敏感度转化为 policy-gradient 信号，而目标函数、reward 与推理协议均保持不变。在 Geometry3K 上训练 Qwen2.5-VL-7B，NC-GRPO 在 5 个留出基准上的域外数学推理显著优于 vanilla GRPO（pooled McNemar p ≤ 0.001），域内准确率与幻觉鲁棒性同样提升——后者恰是图像空间加噪会退化的维度（尽管其在感知密集基准上的域外均值更高）。机制消融表明起作用的是独立随机多样性而非噪声预算或方向，噪声尺度研究揭示了推理专精与通用能力之间的调节旋钮；方法与模态无关，接入标准 RLVR 流水线仅需对推理引擎改约 50 行代码。
- **arXiv**：[2608.21595](https://arxiv.org/abs/2608.21595)

#### Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation (2026-08)
- **简介**：Zhizhao Liu、Zhiliang Tian、Xi Wang、Dongsheng Li 等 7 人。RLVR 依赖昂贵的 rollout 探索，而给不同难度样本分配同等探索预算是低效的：简单样本 rollout 冗余，困难但可学的样本探索不足；已有自适应调度器要么依赖专门 probing（带来大量额外生成开销），要么用历史估计（存在冷启动与反馈过时问题，且忽略样本间关系）。提出一个即插即用的图结构在线难度估计器，让相关样本共享 rollout 反馈并持续更新难度估计：先按语义与推理相似度构建 difficulty-aware 样本图，引入隐难度状态并以 Potts prior 促使邻居样本共享同一状态，再用 state-level Beta-Binomial 模型聚合各状态下的 rollout 结果，最后以在线 mean-field 变分算法随新反馈持续更新状态分配与状态级难度。该框架可接入 sample-selection 与 rollout-allocation 两类调度器，在多个 base model、RL 调度器与基准上取得更好性能（abstract 未给出具体数值）。
- **arXiv**：[2608.17941](https://arxiv.org/abs/2608.17941)

#### GUPO: Gradient Uncertainty-aware Policy Optimization for Post-Training Large Language Models (2026-08)
- **简介**：Peizheng Guo、Jianqi Zhang、Xingyu Zhang、Wenwen Qiang 等 7 人。指出 GRPO 把同一 mini-batch 内不同 query 诱导的 group 梯度直接平均来形成策略更新，但这些 group 梯度方向可能互相冲突；作者的实证分析显示 group 梯度冲突往往与更低效的策略更新相关，因此在冲突下需要一个更可靠的聚合方向，而标准 GRPO 把已实现的 group 梯度当作确定性贡献、聚合时完全不区分其可靠性。提出 **GUPO**：在贝叶斯框式下把每个 group 梯度建模为随机变量并估计其概率分布，再用 Dirichlet 形式推导出梯度不确定性，据此校准各 group 梯度在聚合中的贡献权重。多个基准上的实验验证了其有效性（abstract 未给出具体数值）。
- **arXiv**：[2608.17411](https://arxiv.org/abs/2608.17411)

#### Learn What's Left, Not What's Mastered: Saturation Aware Advantage Reweighting for Multi-Reward Policy Optimization (SA-MRPO) (2026-08)
- **简介**：Yixuan Wang、Yifei Chen、Haichao Zhang、Nuno Vasconcelos、Yijiang Li 等 9 人。指出在优化多个 reward 目标时，现有 group-relative advantage 方法通常先用固定加权和把 reward 向量标量化、再做 group-wise 标准化，由此带来两个根本问题：reward 画像截然不同的 rollout 可能得到完全相同的 advantage；且所有目标不论当前饱和程度都以固定相对权重优化，梯度预算持续花在已解决的目标上而非仍有提升空间的目标。提出 **SA-MRPO**：对每个 reward 目标独立标准化，并依据 batch 级的目标饱和度估计自适应折减其贡献，把优化力度动态重分配给欠优化目标，同时经验上维持已满足目标的表现；作者还指出 saturation-aware 重加权可以反转更新的符号而不只是缩放幅度。在二目标与三目标 reward 组合的数学推理上，SA-MRPO 在 15 组基准对比中有 12 组在更困难的 correctness 目标上优于 GDPO，AIME24 最高提升 5%；自适应推理设定下五个基准全部提升、平均 3.8%、AMC23 最高 9.2%；代码基准 pass rate 最高提升 2.3%，且各设定下较易目标都保持在已满足水平附近。
- **arXiv**：[2608.16072](https://arxiv.org/abs/2608.16072)

#### CEDAR-GRPO: Process-Aware Reinforcement Learning for General Abductive Reasoning in LLMs (2026-08)
- **简介**：Moein Salimi、Danial Parnian、Shaygan Adim、Mahdi Jafari Siavoshani、Mohammad Hossein Rohban 等 9 人。abductive reasoning（推断最佳解释）此前在 LLM 研究中多通过狭窄的任务专用基准考察，难以判断收益能否迁移出训练/评测所用的基准家族。提出 **CEDAR-GRPO**，一个 process-aware 框架：除最终答案正确性外，加入两类面向 abduction 的 reward——evidence coverage（证据覆盖）与 evidence-to-explanation directionality（证据到解释的方向性）。四个 open-weight LLM 在受控、领域中立的假设生成与假设选择混合任务上后训练，并在 11 个未见任务上评测（假设选择、缺失事实生成、defeasible inference、长上下文调查、临床推理、代码调试及非 abductive 对照）。CEDAR-GRPO 在每个模型、每个 held-out 任务上都优于 base 模型与仅用正确性的 GRPO，平均分别提升 7.4 与 2.7 个点，最大提升 30.8 个点；消融确认 RL、abductive reward 设计与任务多样性各自都对迁移有贡献，process-level 指标还显示更强的替代假设探索、竞争假设排除、回溯与不确定性标注行为。
- **arXiv**：[2608.14791](https://arxiv.org/abs/2608.14791)

#### SoftmaxGRPO: Learning to Reason using Softmax Advantage Group Estimation (2026-08)
- **简介**：来自 Rice University（Jefferson Hernandez、Jaywon Koo、Zilin Xiao、Vicente Ordonez 等 5 人）。指出 GRPO 这类基于组的目标在不同难度的 prompt 间分配学习信号很差：在二值 reward 下，组归一化会在接近解决的简单 prompt 上诱导发散的权重。提出 **SoftmaxGRPO**（Softmax Advantage Group Estimation），作为 drop-in 替换把 z-score 归一化的组内 advantage 换成带温度缩放的 softmax advantage，使权重无论 prompt 难度都保持有界；理论上给出二值 reward 下精确的有限组总体目标并指出 MaxRL 是其低温极限，对有界标量 reward 证明大组更新恰好优化一个 log 矩母函数目标，同时说明不对 reward 分布附加假设时不存在通用的有限组标量目标。实验上它把可测的梯度预算从接近解决的 prompt 上重新分配出去，在相同 reward 下一致优于 GRPO：在 DeepMath 上以可验证 reward 达到 51.8%，并仅用轻量文本相似度 reward 把 1.5B instruction-tuned 模型在 Poetry 上从 35.0% 提升到 68.0%。
- **arXiv**：[2608.09271](https://arxiv.org/abs/2608.09271)

#### GCPO: Diagnosing and Constraining Subspace Geometry in Rollout RL for LLMs (2026-08)
- **简介**：来自上海人工智能实验室（Kai Yang、Jingwei Xu、Wanyu Wang、Yu Qiao 等 7 人）。指出 GRPO 等 on-policy rollout 方法在后训练中常出现训练不稳定、跨任务能力退化与回复长度膨胀，而已有工作只刻画了聚合更新的子空间几何，这种几何的逐步变化及其与性能的关系仍不清楚。提出 Principal-Subspace Overlap——一个把单次 rollout 更新相对预训练权重主奇异子空间做维度校正的度量，发现平均 overlap 虽低，但瞬时尖峰往往先于性能退化出现；据此提出 **GCPO**（Geometrically Constrained Policy Optimization），用硬性双边正交投影把更新约束到互补子空间，从构造上阻止这类越界。在 Qwen3-8B 与 GLM4-9B 的数学推理、代码生成与工具使用任务上，GCPO 一致优于 GRPO 及 DAPO、GSPO 等近期变体，相比 base 模型与最强 baseline 分别最多提升 27.69 与 2.37 个点，同时保留通用能力、消除回复长度膨胀并稳定 policy entropy。
- **arXiv**：[2608.11674](https://arxiv.org/abs/2608.11674)

#### LODESTAR: Trustworthy Entropy Is Navigated, Not Merely Measured -- Reinforced Polarizer Keeps a Frozen LLM from Being Confidently Misled by the Wrong Evidence (2026-08)
- **简介**：作者 Po-Jen Ko、Che-Cheng Wu、Hung-Chun Hsu、Chuan-Ju Wang 等 5 人。检索增强问答中，保留 frozen respondent LLM 以最低 answer-token entropy 给出的候选答案是很强的无标注选择规则——在五个 QA 基准上把平均答案 F1 从检索器 top-1 passage 的 0.4769 提升到 0.5148；但这条被已有 entropy 选择器普遍采用的「最低 entropy」规则有一种特定且后果严重的失效：误导性 passage 会让 respondent 自信地答错，恰恰在信号看起来最可信处把 entropy 压低。作者指出失效来源于 respondent 读到的 passage 及其阅读上下文，而上下文是可以干预的输入，于是提出 **LODESTAR**——据作者所知第一个按「文本干预在第三方 frozen respondent 上诱导的不确定性」为干预打分并在同一问题的候选间比较的方法：用 RL 离线一次性训练一个 polarizer，即插入 respondent prompt、从不写入其权重的固定自然语言短串，训练标签由 gold answer 与两个 LLM judge 离线构造，推理阶段两者都不需要。在 5,008 个问题、同一 frozen respondent 与同一候选池下，LODESTAR 取得所有 inference-ready 选择器中最高的平均 F1（0.5148→0.5339）、最高 exact match（0.4136）与所评 frozen-respondent 配置中最高的 GPT-4o judge 分（0.6435），三个随机种子均值在 70 个「方法×数据集」F1 单元上全部战胜 14 个已发表配置且逐一保持配对显著；增益在 in-domain 与 out-of-domain 均成立，消融显示 polarizer 正是让 respondent 更少采纳误导 passage 的原因（26.0% 对 30.3%）。
- **arXiv**：[2608.11922](https://arxiv.org/abs/2608.11922)

#### When Correct Solutions Repeat: Rarity-Aware Credit Redistribution for GRPO (Cue-GRPO) (2026-08)
- **简介**：来自 Zhe Cao、Miaowen Wen、Fangjiong Chen。指出 RLVR 通常把每条正确 completion 当作独立学习信号，GRPO 这种 completion-level 的一视同仁会造成结构层面偏斜：反复出现的常见解法形式按被采样频率累积正系数质量，稀有解法形式只得到很少 credit，作者将其形式化为 multiplicity-induced structure-level credit concentration。提出按簇稀有度重分配正 advantage 的 partition-conditioned 规则，**Cue-GRPO** 用确定性 Strategy Cues 对已验证正确轨迹构造 rollout-local 划分来实例化该规则，无需辅助模型推理；并验证同一机制在 judge 导出的划分（Judge Partitions）下同样可用。在 Qwen2.5-Math-7B 与 Llama-3.1-8B-Instruct 上提升 AIME 重复采样性能，采样预算越高增益越大，训练 wall-clock 开销仅比 GRPO 多 6%。
- **arXiv**：[2608.03467](https://arxiv.org/abs/2608.03467)

#### Beyond the Mean: Multi-Moment Policy Optimization for LLM Reasoning (MMPO) (2026-08)
- **简介**：来自 Yijun Zhang、Yule Xie、Jiaxin Ding、Luoyi Fu 等 7 人。提出以「矩」（moment）视角重看 LLM 推理的策略优化：把随机抽取一道题的失败概率视为随机变量，用其各阶矩来刻画优化目标，并指出现有方法基本只优化失败概率分布的单个矩，分布的更广结构未被利用。提出 **MMPO**（Multi-Moment Policy Optimization），联合最小化失败概率分布的多个矩，其目标可直接解释为最小化「获得首个成功回答所需的期望截断时间」；在此之上还给出通用的 moment-transformation 框架，可系统诱导不同的矩剖面，为一大类策略优化目标提供统一视角。在五个数学推理基准与不同规模模型上，MMPO 一致优于强基线。
- **arXiv**：[2608.02149](https://arxiv.org/abs/2608.02149)

#### CVPO: Enhancing LLM Reinforcement Learning Reasoning via Value-Variance Adaptation and Dynamic Curriculum Learning (2026-08)
- **简介**：来自 Ziqi Jia、Yalu Ouyang、Bo Pang、Panpan Li 等 8 人。针对现有方法对生成答案轨迹的反馈精度不足、以及训练中出现的题目难度漂移（problem difficulty drift）两个问题，提出 **CVPO**（Curriculum-guided Value-Variance Policy Optimization）。轨迹层面：发现 token-level value-variance 与探索强度相关，并给出理论分析表明该方差可界定策略更新幅度，于是用估计的轨迹 value-variance 量化生成过程的内在随机性，据此为不同 reward 类型设计 variance-aware 的 advantage 调整机制；题目层面：引入随难度自适应的动态课程加权，使模型在每个训练阶段聚焦与当前能力相匹配的题目。实验表明其性能超过 VAPO 等强 value-based 基线，同时具备更强探索，在多种数学任务上推理更准确、更稳健。
- **arXiv**：[2608.03068](https://arxiv.org/abs/2608.03068)

#### Start Classifying: Categorical Critics for LLM Reinforcement Learning (HL-Gauss PPO) (2026-08)
- **简介**：来自 Zhijian Zhou、Long Li、Xuan Zhang、Yulei Qin 等 10 人。指出 LLM 的 PPO critic 通常以标量 MSE 回归拟合 value 目标，而 RLVR 的稀疏二值 reward 使 critic 的优化与校准格外关键——细微的 value 误差会直接扭曲 PPO 所用的标量 advantage。提出 **HL-Gauss PPO**：把标量 MSE 头替换为在离散化 value support 上的类别预测头，用交叉熵拟合平滑后的 HL-Gauss 目标，输出再解码为标量期望供标准 GAE 与 PPO 使用，因此 actor 更新完全不变、并非分布式 RL。在数学推理、工具增强数学与 Search-R1 上，Qwen2.5 与 Qwen3 两种 backbone 均稳定优于 PPO 与 DAPO；one-hot、two-hot 与 Bernoulli 两桶 critic 的对照说明增益既不来自更大的输出头也不只靠二分类，且 HL-Gauss 改善了 Brier score 与校准误差，给出更对称、方差更低的 advantage。
- **arXiv**：[2608.02181](https://arxiv.org/abs/2608.02181)

#### Cooperative Coevolution for Resource-Constrained Agentic LLM Post-Training (CoPES) (2026-08)
- **简介**：Zhiyuan Wang、Shengcai Liu、Jiahao Wu、Ning Lu 等。工具调用 LLM agent 产生长多轮 trajectory，使基于梯度的后训练极耗显存；进化策略（ES）可免反向传播做全参数后训练并最终追平梯度 RL，但在只有少量 GPU 的受限场景下其 GPU-hour 开销导致训练时间过长。提出 **CoPES（Cooperative Parameter-subspace Evolution Strategy）**：把全参数空间分解为若干低维子空间并协同搜索以提升优化效率。在数学任务上后训练 Qwen3.5-4B 工具调用 agent、于五个不同难度基准评测：在与全参数 GRPO 最佳验证 checkpoint 相同的 GPU-hour 预算下，CoPES 恢复了 GRPO 验证准确率增益的 92%（标准 ES 仅 67%），理论显存需求不足全参数 GRPO 的 1/8，且在五个基准的全部 pass@k 指标上稳定优于标准 ES 与 LoRA-based GRPO。
- **arXiv**：[2608.02391](https://arxiv.org/abs/2608.02391)

#### LEEPS: Latent-Guided Explore-Exploit Prompt Sampling for Efficient RLVR in Large Language Models (2026-07)
- **简介**：来自中国人民大学等（Shuang Liang、Xiting Wang 等）。RLVR 中"组内 rollout 奖励全相同"的提示白耗生成预算；已有 pre-rollout 选择难以平衡利用与探索（反复利用历史高信息提示会收窄覆盖，过度探索又降低有效提示占比）。**LEEPS** 把候选划分为 exploit / explore 两组合，按各自近期"非平凡比率"自适应分配 rollout 预算，并用表征空间近邻 + 历史 rollout 结果优先选取"可能产生非零奖励方差"的不确定提示，使探索更有的放矢而不增额外 rollout。六个数学基准上两种模型规模均取最高均分（Qwen2.5-Math-1.5B/7B 相对最强基线 +2.6% / +3.7%），三个 OOD 通用推理基准亦最高，每步仅约 2 秒在线采样开销。
- **arXiv**：[2607.28077](https://arxiv.org/abs/2607.28077)

#### LoRA Scaffolded Policy Optimization (LSPO): A Sampling-Time Low-Rank Scaffold for Recovering Reinforcement-Learning Gradient on Zero-Reward Cliff Prompts (2026-07)
- **简介**：来自 Ken Ding（单作者）。针对 RLVR 的结构性盲区——"cliff"提示（组内每条 rollout 全失败）导致组归一化优势恒为 0、GRPO 在模型能力前沿处无梯度。**LSPO** 在采样时恢复该梯度：每步检测 cliff 提示，对其 ground-truth 解做一次简短监督拟合一个小 LoRA adapter，用 base+adapter 重新 rollout，将现已成功的补全经重要性采样校正拼回 RL batch，仅对 base 做 GRPO 更新；adapter 只吃监督梯度并在 checkpoint 丢弃，得到纯 base 模型。DeepMath-103K + DeepSeek-R1-Distill-Qwen-1.5B、n=5 配对种子、1000 步下，在全部 16 个 (基准, pass@k) 格中匹配或超过 DAPO（15 胜 1 平），AIME24/pass@4 最高 +10.7 分，16 格平均 +3.8 分。
- **arXiv**：[2607.27787](https://arxiv.org/abs/2607.27787)

#### Kalman Meets Curriculum: Efficient Dynamic Prompt Selection for Adaptive RL Finetuning (KGPS) (2026-07)
- **简介**：来自北航等（Haodong Zhu、Baochang Zhang 等）。RL 微调效果取决于为当前策略选到合适难度的提示，而提示难度随训练动态变化。**KGPS** 把提示选择重构为动态状态估计问题：在 logit 空间用线性-高斯状态空间模型刻画每个提示的潜在成功率，过程噪声与策略更新幅度耦合（策略变动大则不确定性升高），用卡尔曼滤波维护对难度的校准高斯后验，并按"偏好中等难度、自然回访不确定项"的后验期望训练效用来选提示，无需额外 rollout。在数学、规划、几何推理基准及多种 RL 算法上一致提升最终精度与 rollout 效率——如 DeepSeek-R1-Distill-7B 上比 DS 少用 83% rollout 同时平均 +0.12 分。
- **arXiv**：[2607.27610](https://arxiv.org/abs/2607.27610)

#### ReCo: Reweighting GRPO Against Distributional Concentration (2026-07)
- **简介**：来自首尔大学（Junoh Park、Taesup Kim 等）。针对 GRPO 会降低基模推理覆盖、在大 k 时 Pass@k 反不如基模的现象，追因为 GRPO 更新会"集中"到基模本已高概率生成的响应：响应级上高概率响应因重复出现主导组梯度，token 级上重要性比进一步放大已变高概率的 token。**ReCo** 同时纠正两者——按 rollout 组内期望出现频率归一化响应贡献，并用基于方差的比值替换 token 级重要性比，对仍有可替代 token 选择的"非饱和决策点"给更大更新尺度。在 Qwen2.5-Math-1.5B/7B 与 Llama-3.1-8B-Instruct、五个数学基准上改善大 k 的 Pass@k，小 k 与 GRPO 相当。
- **arXiv**：[2607.26862](https://arxiv.org/abs/2607.26862)

#### ISO: An RLVR-Native Optimization Stack (Isospectral Optimization) (2026-07)
- **简介**：Hanqing Zhu、Yuandong Tian、Zhangyang Wang 等（UT Austin / Meta 等）研究 RLVR 中"把奖励反馈转成权重更新"的优化层，提出 **spectral inheritance**：RLVR 可复用基座权重奇异谱、仅通过输入/输出奇异 frame 的变化获得新行为。据此提出 **Isospectral Optimization (ISO)**：固定谱、优化 frame 的 RLVR-native 框架，含离线 ISO-Merger（无需 post-merge 数据/rollout/梯度即合并共享基座专家）与在线 ISO-Optimizer（对 frame 变量施加 AdamW/Muon）。1.5B–8B 推理与编码任务上收敛显著更快：Qwen3-8B-Base 上 AdamW 需 270 步达 0.495，ISO-AdamW 仅 100 步达同分、210 步进一步到 0.509。
- **arXiv**：[2607.19331](https://arxiv.org/abs/2607.19331)

#### RRPO: Reference-Relative Policy Optimization with Stratified Conditional Rollouts (RRPO) (2026-07)
- **简介**：Yuxin Xiong、Junda Wu 等（UCSD / Adobe Research）提出 RRPO，将 GRPO 从"依赖任务正确性验证器"泛化到无显式 verifier 的设定：先用 **stratified conditional rollouts** 构造正/负 anchor 集，再训练一个 metric projection head 以 set-contrastive 目标对候选 rollout 打分；优化时冻结投影头，把对齐分数在组内中心化，直接作为组相对优势（contrastive advantage）。在可验证推理、开放式生成与 post-SFT 三类设定下，RRPO 与基于 verifier 的优化持平、优于弱监督基线，并在 SFT 后仍有额外增益。
- **arXiv**：[2607.18470](https://arxiv.org/abs/2607.18470)

#### OR Else: A Differentiable Trust Region for Policy Optimization (PPO-OR / GRPO-OR) (2026-07)
- **简介**：Chinmay Rane、Kanishka Tyagi、Michael Manry 指出 PPO/GRPO 的 clipped surrogate 在"有利方向饱和"处导数突变，提出 **Output Reset (OR)** ——一种光滑单侧饱和规则，用 OR squared-margin loss 在 rollout-relative token log-ratio 空间替换裁剪项（优势符号定方向，token 越过有利 margin 后直接 OR 残差归零）。在 Llama-3.2-1B-Instruct + hh-rlhf（共享 RM、3 seed）上：GAE 设定下 PPO-OR 平均终期 RM 分比 PPO-clip 高 0.305；group-relative 设定下 GRPO-OR 均分未提升但方差更小、终期 OR 残差近零、overshoot 比例下降。属对 clip 机制的可微替代研究。
- **arXiv**：[2607.18163](https://arxiv.org/abs/2607.18163)

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

#### Process Reward Modeling with Reliability- and Conservatism-Aware Bayesian Belief (2026-10)
- **简介**：作者 Zhongjian Qiao、Zhixiang Zhou 等。PRM 是 LLM 推理 test-time scaling 的核心，但其过程奖励可能不准，论文将原因归为两类不确定性：PRM 本身因监督数据不足和训练不完善带来的不确定性，以及采集监督数据的 policy 与 test-time scaling 所用 policy 不一致带来的 policy 不确定性（导致系统性奖励高估）。提出 **BayesianPRM**：把过程奖励视为奖励假设空间中的随机变量并维护 Bayesian belief，用 reward ensemble 离散化假设空间；通过「reliability-aware belief updating」（用 ELBO 优化的 amortized belief network，衡量各 ensemble 成员对监督数据的解释程度）处理 PRM 不确定性，再通过「conservatism-aware belief calibration」把 belief 向低奖励重加权以抑制高估。多模态推理 benchmark 上，BayesianPRM 给出更准确的过程奖励并提升 test-time scaling 表现（abstract 未给出具体数值）。
- **arXiv**：[2610.23223](https://arxiv.org/abs/2610.23223)

#### BoT-GRPO: Efficient Process-Reward RL for Reasoning via Bag-of-Token Aggregation (2026-10)
- **简介**：作者 Yingxiang Yang、Weihang Xiao、Zhunxuan Wang、Joshua Flashner、Niresh Agarwal，发表于 COLM 2026 Workshop on Efficient Reasoning。GRPO 中 rollout 的每个 token 获得同样的 advantage，论文探讨如何在不引入 value network 的前提下高效利用过程监督。**BoT-GRPO** 通过长度不变的「bag of tokens」聚合把 GRPO 扩展到 token 级 reward model：汇集所有 rollout 的 token 级奖励，按所属序列长度的倒数加权，并相对加权后的组统计量计算逐 token advantage；无需 critic，可在有 token 级奖励时直接替换 GRPO。在 React 前端代码生成上，达到 80% 编译率最高比 GRPO 快 1.9 倍，收敛快于 GSPO、DAPO、PURE；在 AIME 数学推理上，用一半步数获得相对 GRPO 最高 8.1% 的绝对 Pass@k 提升。
- **arXiv**：[2610.09804](https://arxiv.org/abs/2610.09804)

#### Inducing Process Supervision from Outcome-Only Reinforcement Learning (TIPS) (2026-09)
- **简介**：作者 Shengda Fan、Xin Cong、Zhong Zhang、Haotian Chen、Yankai Lin 指出，训练强 PRM 成本高：人工步骤标注难扩展，Monte Carlo 估计昂贵且可能偏离步骤的内在正确性。提出 **TIPS**（Thinking-Induced Process Supervision），用仅依赖结果的 RL 训练生成式 PRM：模型先生成 CoT，再输出步级标签与结果标签，reward 只看预测结果是否与真值一致，所得 group-relative advantage 优化整个生成响应。直觉是检查中间步骤有助于判断结果时，更准确的检查会带来更好的结果判断与更高 reward，因而无需显式过程监督就能强化步级验证。在数学与 agent 基准、四个骨干家族上验证；TIPS-Qwen3-4B-Thinking-2507 仅用 3.2K 条带结果标签的轨迹，在 ProcessBench 上达 85.2 F1，超过所有评测的已训练 PRM 以及 GPT-5.4-Instruct、Claude-4.7-Opus 等 prompt-only 强 judge，但仍落后于 o1-mini。
- **arXiv**：[2609.36641](https://arxiv.org/abs/2609.36641)

#### Monitorable Chart Reasoning Agents via Verifiable Process Rewards (Chart-RVR) (2026-09)
- **简介**：作者 Sanchit Sinha、Oana Frunza、Kashif Rasul、Aidong Zhang（EMNLP 2026 Findings）。基于 LVLM 的图表 agent 只给答案或难以验证的自由形式推理，无法判断错误来自读图、取值还是计算。**Chart-RVR** 是用可验证过程奖励训练可监控图表 agent 的 RL 框架，把图表推理分解为三个可审计块：Structure（识别图表类型）、Evidence（以 JSON 重建底层数据表）、Derivation（逐步推导答案）。在六个域内/域外基准上取得同等规模 LVLM 中的 SOTA 准确率；并通过结合真值代理指标、oracle 信息增益与 LLM-as-auditor 的三角化协议评估，显示其推理在过程可验证性与证据定位上明显优于 CoT 提示、SFT 与现有图表基线。
- **arXiv**：[2609.24071](https://arxiv.org/abs/2609.24071)

#### Neuro-symbolic PRM: Enhancing Scientific Reasoning via Structured Traces and Symbolic Verification (2026-08)
- **简介**：Yuxin Zi、Cong Xu、Suparna Bhattacharya、Martin Foltin、Amit Sheth。指出工具增强 LLM 在定量 STEM 多步推理上虽已大幅进步，仍残留一类失败模式：中间步骤语法正确、数学可执行、单位一致，但在上下文中并不成立（contextually ungrounded）；现有方案要么依赖无法判断语义意图的形式化 verifier，要么让 PRM 同时承担算术与逻辑双重检查的负担。作者提出 neuro-symbolic 框架，把推理正确性干净地拆成两个形式化维度——**Symbolic Validity（V）** 与 **Semantic Groundedness（G）**：V 由确定性符号 verifier 作为硬过滤器按构造保证；G 则通过在「verifier 已接受的流形」上条件式训练一个 PRM 来评估。为高效训练该 PRM，引入 **Counterfactual Symbolic Perturbation（CSP）** 数据合成策略，算法化地生成保持约束的 hard negative（能完美通过 verifier 但逻辑错误的步骤）。推理时采用 verifier-first 的约束搜索，对 verifier 覆盖的算子保证执行一致性，PRM 只负责对语义 groundedness 排序。方法精准针对强工具型 LLM 的残余错误类别，在不引入大量启发式规则的前提下显著提升推理可靠性（abstract 未给出具体数值）。
- **arXiv**：[2608.26329](https://arxiv.org/abs/2608.26329)

#### Rewarding Better Thinking for LLM Preference Alignment (Thinking Checklist Reward, TCR) (2026-07)
- **简介**：Xubo Liu、Ying Zhang 等（南开大学）针对偏好对齐中"代理奖励多为 outcome-level、对推理轨迹指导不足、终分接近时信用分配粗糙"的问题，提出 **Thinking Checklist Reward (TCR)**：把偏好对转成样本专属的"思维检查清单"，据此评估生成的推理轨迹是否覆盖偏好隐含的考量；并引入 EMA 残差公式隔离出超越 outcome 奖励可预测部分的"thinking surplus"。在 3 个模型族、5 个模型上一致提升对齐表现，消融验证 EMA 残差与样本专属清单监督的重要性。
- **arXiv**：[2607.19824](https://arxiv.org/abs/2607.19824)

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

#### Using Experience Twice: Dynamic Process Rubrics for Privileged Token-Level Credit Assignment (ORSD) (2026-11)
- **简介**：作者 Shumao Wu、Kongcheng Zhang、Jianqiang Huang、Baisheng Lai。针对稀疏的轨迹级奖励难以识别哪些关键决策应被强化或抑制的问题，论文提出 **ORSD**（Online Rubric Self-Distillation）：当前策略针对同一组轨迹在每步独立生成多组 rubric bundle，从跨轨迹行为差异中抽取候选过程 rubric；冻结的 verifier 逐一评估各轨迹对 rubric 条目的满足度，并依据 rubric 满足度与任务成功的 bundle 级一致性选出一组。训练时把选中的 rubric 作为 privileged information，比较有无 rubric 时原始 token 的对数似然差，作为 token 级过程信用，通过加性 advantage shaping 叠加到轨迹级 GRPO advantage 上，从而衔接结果监督与细粒度过程监督（abstract 未给出具体数值）。
- **arXiv**：[2611.06543](https://arxiv.org/abs/2611.06543)

#### NAS-PO: Native Attention-Strategy Policy Optimization for Vision-Language Models (2026-11)
- **简介**：作者 Ke An、Shizhe Xiang、Yue Liu 等。标准 RLVR 对所有生成 token 赋予相同的结果信用，不区分其感知相关性；已有感知感知方法依赖图像扰动/遮挡分支，是间接代理。NAS-PO 转而利用模型的 native attention 这一内在信号，把它构造成奖励锚定、token 对齐的路由信号：用 visual mass policy 度量视觉分配、conditional visual policy 刻画对视觉 token 的空间路由；**NAS-AS**（Native Attention-Strategy Advantage Shaping）据此对视觉参与度高的 rollout 重新分配带符号的 policy-gradient 信用，**AAC** loss（Adaptive Attention-Coupled）则在所有回复 token 上对齐 visual-mass policy、在视觉活跃 token 上对齐 conditional visual policy，固化有助于成功的视觉路由模式。无需额外扰动分支（abstract 未给出具体数值）。
- **arXiv**：[2611.02008](https://arxiv.org/abs/2611.02008)

#### GRIT: Gradient Masking for Visually Misgrounded Tokens in Text-Rich Reasoning (2026-10)
- **简介**：作者 Chanseul Cho、Noori Bae、Dongheon Lee、Youngmin Ro，论文被 WACV 2027 接收。VLM 在文本密集推理中常出现「答案正确但中间视觉感知有误」的现象，作者称之为 CVMR（Correct-but-Visually-Misgrounded Reasoning）；标准 RLVR 只奖励最终答案，会同等地强化 grounded 与 misgrounded 的推理。token 级分析显示 CVMR token 的 attention 模式与 grounded token 有系统性差异。据此提出 **GRIT**：把视觉 grounding 引入 RLVR，通过对视觉误锚定 token 做梯度屏蔽来削弱其影响。作者还构建 **GRIT-Bench**，可在文本长度与推理深度上系统评估文本密集推理。实验表明 GRIT 相较现有 RLVR 方法同时提升推理准确率与 grounding 保真度（abstract 未给出具体数值）。
- **arXiv**：[2610.41629](https://arxiv.org/abs/2610.41629)

#### ProbePO: Sparse Probes, Redistributed Credit for Critic-Free LLM Reinforcement Learning (2026-10)
- **简介**：作者 Diyi Hu、Ruizhong Qiu、Hanghang Tong 等（NeurIPS 2026 Workshop）。RLVR 只给整条推理轨迹一个终端奖励，GRPO 将 group 归一化 advantage 广播到所有 token，而细粒度方法常依赖密集的辅助 rollout 或分支，且可能改变策略更新所用的数据。论文把问题表述为「带预算的测量」：**ProbePO** 用无奖励的不确定性分数挑选部分解状态，在受限的续写预算下估计其期望 verifier 奖励，并在保持每条轨迹平均 token 级 advantage 不变的前提下，把 GRPO 的轨迹级 advantage 在 token 间重新分配，probe 续写仅作测量。在 1.5B-7B 的 DeepSeekMath 与 Qwen 系列三个模型上，ProbePO 稳定优于 GRPO，在 18 个模型-benchmark 组合中 16 个取得四种方法中的最高精度；相对密集探测的 VinePPO（DeepSeekMath-7B），六个 benchmark 全部更高，端到端训练时间减少 76%。
- **arXiv**：[2610.24515](https://arxiv.org/abs/2610.24515)

#### Hindsight-Privileged RLVR: Single-Rollout Token Credit Without Group Sampling (HP-RLVR) (2026-10)
- **简介**：作者 Wei Chen、Yubing Wu 等。GRPO 类方法在组内所有尝试奖励相同时会失去学习信号。**HP-RLVR** 是单 rollout 方法：用生成策略的快照分别在因果历史、以及包含已实现未来与 verifier 反馈的 hindsight 上下文下为每个 token 打分，用两个视角之间归一化的 Jensen-Shannon 散度，相对历史基线对奖励加权，为成功与失败的尝试都给出有界的 token 级信用，无需参考解或学习得到的 critic。在 Qwen3-8B 的配对实验中，HP-RLVR 在数学、代码和 ALFWorld 上比 GRPO 分别快 1.21x、1.46x、2.30x，每 prompt 仅用八分之一的 rollout；整体精度相当，数学上全部规模领先，代码与 ALFWorld 上 GRPO 略优。ALFWorld 中 GRPO 约 43%-49% 的梯度范数记录为零，而 HP-RLVR 没有。
- **arXiv**：[2610.20497](https://arxiv.org/abs/2610.20497)

#### TPPO: Tracking-Value PPO For Language Model Reasoning (2026-10)
- **简介**：作者 Futing Wang、Chenxi Li、Jianhao Yan 等。稀疏终端奖励下标准 critic 难以跟踪沿轨迹的 value 演化，而这种演化恰好刻画了 token 级相对信用。论文分析相邻前缀 value 的关系，证明其在 log-odds 空间具有可加的 Bayesian 分解，据此构造 **Tracking-Value PPO（TPPO）**：把各位置的标量输出累加为 value logit，并通过对重构 value 的终端结果监督来学习。TPPO 在 Qwen3-4B-Base 和 Qwen3-8B-Base 上的综合数学推理分数比 PPO 分别高 7.0 和 8.4 分，同时提升 OOD 泛化，并在 Qwen3-4B、Qwen2.5-Math-1.5B、Qwen3.5-9B 以及 countdown 与科学编程任务上取得类似增益。
- **arXiv**：[2610.16794](https://arxiv.org/abs/2610.16794)

#### Each Reward to Its Relevant Tokens: Reward-Specific Token-Level Credit Assignment for Multi-Reward RL (MRTPO) (2026-10)
- **简介**：作者 Skylar Zhai、Haonan Wang、Zhining Liu 等。模型常同时对多个可验证奖励（如工具调用的正确性与格式、复杂指令的各项约束）进行训练；GDPO 对每个奖励分别归一化，但仍把每个奖励的 advantage 均匀铺在整个 response 上，尽管每个 verifier 只检查少数位置。**MRTPO** 把每个奖励的 advantage 分配给与之相关的 token：对每个奖励，策略分别在带/不带该奖励 verifier 信息下重放自己的 response，用每个 token 两个 next-token 分布的 Jensen-Shannon 距离衡量相关性并重新分配 advantage，无需辅助模型或 token 级标注。在 text-to-SQL、多约束指令遵循和工具调用上，MRTPO 比对比方法收敛更快、表现更好（abstract 未给出具体数值）。
- **arXiv**：[2610.21439](https://arxiv.org/abs/2610.21439)

#### Bellman-Consistent Trajectory RL for LLM Arithmetic Reasoning (2026-10)
- **简介**：作者 Lianfa Li。针对 LLM 算术推理中的稀疏终端奖励，论文提出 Bellman-consistent 的轨迹 RL：用 **Trajectory Bellman Residual Minimization（TBRM）** 把稀疏奖励转化为稠密的 token 级信号，结合 Reinforce Leave-One-Out（RLOO）组策略降低 Bellman 残差的梯度方差，并用 hindsight goal relabeling 加速中间步骤学习。在 Countdown 任务上，TBRM-RLOO 的梯度方差稳定性优于 GRPO，Pass@1 达 0.55-0.64，Pass@16 达 76.0%-84.0%，优于 SFT、IPO/DPO 和 RLOO 基线，并表现出较高的单样本效率。该工作为课程项目报告性质，仅在 Countdown 任务上验证。
- **arXiv**：[2610.21060](https://arxiv.org/abs/2610.21060)

#### Hierarchical Credit Assignment for RLVR on Fused Gromov-Wasserstein Geometry (HarA) (2026-10)
- **简介**：作者 Qi Yu、Ruizhong Qiu、Zhichen Zeng、Xuying Ning、Yanjun Zhao、Dongqi Fu。GRPO 等 group-based RLVR 给同一结果的 rollout 内所有 token 同样的 advantage；已有工作用 token 位置或熵等局部信号改进信用分配，却难以捕捉推理行为相对当前策略的全局语义新颖性。**HarA** 把每条 rollout 表示为 token 隐状态与位置上的分布，计算同结果 rollout 的 Fused Gromov-Wasserstein (FGW) barycenter 以刻画当前策略下的内部推理模式，以某推理元素对当前 rollout 与 barycenter 间 FGW 距离的贡献度量其语义新颖性；并用 anchor-guided 线性化把昂贵的 FGW 转为可用 Sinkhorn 高效求解的 Wasserstein 形式，据此对 token 级 advantage 重新加权（abstract 被截断，具体提升数值以原文为准）。
- **arXiv**：[2610.04344](https://arxiv.org/abs/2610.04344)

#### Visual sensitivity is not claim retractability: persistence-aware credit assignment for multimodal reinforcement learning (PACG) (2026-09)
- **简介**：作者 Zhongan Bi、Kepeng Lin 等发现，Qwen2.5-VL-7B 在四个多模态推理基准上答对的响应中，有 27.81% 至少包含一条图像不支持的直接视觉声明，而结果级 RL 会让这些声明继承正确答案的正 credit。通过固定 rollout 的反事实诊断（在被干预的图像下重新打分）区分 Evidence-Function Sensitivity（EFS）与声明持久性，揭示 Sensitivity-Persistence Decoupling：DAPO、VPPO 下 EFS 上升、声明整体更易撤回，但不受支持的声明反而更持久，GRPO 则无此恶化。提出 **Persistence-Aware Credit Gating（PACG）**，衰减异常持久的视觉声明的正 credit，无需支持/不支持标签，也不增加推理成本。Qwen2.5-VL-7B 上九基准平均在 DAPO 下由 58.1% 升至 59.9%，VPPO 下由 59.8% 升至 60.9%。
- **arXiv**：[2609.36572](https://arxiv.org/abs/2609.36572)

#### Reinforcing Multimodal Reasoning via Token-Level Perception-Grounded Advantage Estimation (TPAE) (2026-09)
- **简介**：作者 Zhihan Zhang、Lizi Liao（ACM MM 2026 接收）指出 MLLM 的 RLVR 依赖粗粒度序列级 reward，缺乏对推理链中视觉相关步骤的监督。通过视觉依赖度和预测熵两个 token 级指标发现：正确推理链随视觉 grounding 加强熵下降更明显；关键 token（预测错误会引发推理崩溃）是正确链的视觉依赖-熵联合分布中的统计离群点。据此提出 **TPAE**：衡量每个 token 与正确 rollout 的视觉-熵模式的统计一致性来估计 token 级 advantage，并以此调制序列级 advantage，可接入多种 RLVR 框架。在七个基准上稳定优于强基线，优化更稳定高效。代码已开源。
- **arXiv**：[2609.39168](https://arxiv.org/abs/2609.39168)

#### Surprising Success, Repeated Failure: Entropy-Guided Credit Assignment for Exploration in LLM Reasoning (EAPO) (2026-09)
- **简介**：作者 Woongyeong Yeo、Minki Kang、Jinheon Baek、Sung Ju Hwang 等指出，细粒度 credit assignment 常需辅助模型、额外采样或特权信息；而策略熵虽是现成信号，在强化与惩罚两侧都优先高不确定位置会把惩罚集中到失败响应仍有恢复余地之处，抑制探索。提出 **Entropic Advantage Policy Optimization（EAPO）**，对成功与失败做非对称处理：把归一化策略熵与 advantage 的符号耦合，强化「令人意外的成功」、纠正「反复出现的失败」——对成功响应中的高熵决策给予更强强化，对失败响应中的低熵决策给予更强惩罚，同时削弱不确定位置的惩罚以保留恢复机会。在基座与推理骨干上的多种推理任务中取得最佳整体性能，并扩大问题覆盖、增加候选答案多样性。
- **arXiv**：[2609.33781](https://arxiv.org/abs/2609.33781)

#### Privy to the Foil: Recasting Value Estimation with a Self-Privileged Critic for RLVR (πPPO) (2026-09)
- **简介**：作者 Kun Liang、Chenming Tang、Clive Bai 等指出，PPO 等 actor-critic 方法虽可通过 value function 构造 token 级 advantage，但可靠的价值估计既要评估进展又要预测演化中策略的未来行为，错误会破坏 credit assignment 并使在线训练不稳定。论文重新审视仅依赖状态的 value 估计，提出自特权 actor-critic 框架 **πPPO**：复用经验证的同 prompt rollout 作为对比证据，使 critic 能对照成功与失败的尝试评估中间推理，同时保持标准策略优化和部署接口不变。实验表明 πPPO 显著提高价值估计质量，并在高难度数学推理基准上优于有代表性的 actor-critic 与无 critic RLVR 基线，配合更小的非对称 critic 仍有效。
- **arXiv**：[2609.37825](https://arxiv.org/abs/2609.37825)

#### Trust the Critic More (AC2) (2026-09)
- **简介**：作者 Kaiyue Wen、Luke Bailey、Arvind Mahankali、Tengyu Ma 指出，标准 LLM RL 对长 rollout 的每个 token 赋予由终局 reward 决定的同一 advantage，而学习的 critic 通常被认为过于不准，即便存在也只用作 baseline，故每条轨迹仍须 rollout 到终局。提出 **Actor-Critic with Action Chunking（AC2）**：在 action chunk（过往轨迹前缀之后的短续写）上分配 credit，由 critic 对 chunk 末状态打分，使策略更新无需观察终局 reward。通过三项设计使 critic credit 可靠：local readiness（仅在 critic 对该题足够准确时才用 critic 更新）、可用时向 critic 提供先前成功 rollout 的参考解、以及以 10k token 为单位的 chunk。在 FineProofs-RL 上训练 Qwen3-4B 并在 IMO-ProofBench 评测，AC2 以少 2.5 倍 decoding FLOPs 超过 GRPO 的峰值验证分 18.5%，训练步数亦少 25%。
- **arXiv**：[2609.39247](https://arxiv.org/abs/2609.39247)

#### SIPO: Unifying Reinforcement Learning with On-Policy Self-Distillation (2026-09)
- **简介**：作者 Zhenrui Yue、Huimin Zeng、Yueqi Wang 等指出 RLVR 的稀疏结果 reward 缺少中间步骤的 token 级 credit assignment；on-policy self-distillation（OPSD）用带特权上下文的 self-teacher 提供稠密信号，但 self-teacher 常过度自信，对长推理轨迹施加过度惩罚。提出 **SIPO**（self-instructing policy optimization），使用对比式 self-teacher：对每个 prompt 采样多条 rollout 并用环境 reward 打分，为每条 rollout 构造两个教师上下文（参考答案与组内错误的配对），模型在两种上下文下重新评估自身响应，用两个教师 log 概率之差作为 token 级反馈，使两种上下文共有的偏差大体抵消。reward 仍决定更新主方向，self-teacher 在 token 间重分配 credit；即使整组 rollout 全失败、group-relative advantage 消失，SIPO 仍有学习信号。在多个推理与代码生成基准上优于 RLVR 与 OPSD 基线，且无需外部教师或额外生成。
- **arXiv**：[2609.36742](https://arxiv.org/abs/2609.36742)

#### PACT: From Credit Assignment to Critic Alignment (2026-09)
- **简介**：作者 Jiayan Fu、Hang Xu、Yong Zhang 等。针对 token 级 credit 缺乏公认数学定义的问题，提出 Completeness、Prefix Consistency、Neutrality 三条正则条件并证明其唯一确定 token 级 credit，据此统一解释现有算法：OPD 中的理想 teacher 相当于隐式 critic，response 级 RLOO 信号在期望上与 token 级 credit 的策略梯度一致，GAE 的中间 critic 误差可能与真实 credit 同量级。基于此提出 **PACT**（Policy Aligned Critic Training），采用 Actor-then-Critic 更新顺序并对 critic 训练做重要性采样校正。agentic 数学推理四个基准平均 72.87%，比 GRPO、PPO 分别高 8.80、13.16 个百分点；SWE-bench Verified 达 67.4%，分别高于 PPO、GRPO、SAO 2.4、2.0、3.8 个百分点。
- **arXiv**：[2609.26355](https://arxiv.org/abs/2609.26355)

#### Giving Credit Where It's Due: Redundancy-Aware Learning for Efficient Reasoning (RECAP) (2026-09)
- **简介**：作者 Yuqing Zhou、Hong Wang、Manqing Mao 等。大推理模型常产生正确但冗长的推理，现有方法的轨迹级目标或局部 token/step 信号很少建模步骤间语义依赖。**RECAP**（REdundancy-aware Credit Assignment via Propagation）结合两个信号：structural responsibility——从最终答案节点沿 LLM 标注的语义依赖图向后传播 credit；step efficacy——以逐步加入后金标准答案 log-likelihood 的变化衡量朝答案前进的程度。二者把 rollout 级 GRPO advantage 重塑为 step 级更新，且无需单独训练 PRM 或预构造简洁轨迹。在两个 7B 模型、四个数学基准上，Qwen2.5-Math-7B 的 pass@1 比 GRPO 提升 2.0–3.7 个百分点，同时推理 token 减少 8%–31%。
- **arXiv**：[2609.27156](https://arxiv.org/abs/2609.27156)

#### Information-Time Proximal Policy Optimization (InfoPPO) (2026-09)
- **简介**：作者 Yongcheng Zeng、Xinyu Cui、Yan Song 等。现有 RLVR 方法以逐 token 生成参数化 MDP 的时间推进，而自回归轨迹上的信息流高度不均匀。**InfoPPO** 用信息密度而非 token 计数重参数化时间，为时间 credit 传播与策略更新诱导出共同的状态相关结构：恢复了长程推理中非平凡折扣的有效性，同时避免终态监督在长序列上过度衰减；并由 information-time 的策略改进分析导出状态相关的更新约束，用按 token 位置信息密度自适应的 clipping 实现。理论上给出了策略改进下界。在 Qwen3 模型的五个竞赛级数学推理基准上稳定优于强基线，且在 token-time PPO 退化的非平凡折扣设置下仍保持稳定的准确率与响应长度（abstract 未给出具体数值）。
- **arXiv**：[2609.24380](https://arxiv.org/abs/2609.24380)

#### Anchoring What Matters: A Dual-Level Learning Framework for Visually-Grounded Multimodal Reasoning (PIVOT) (2026-09)
- **简介**：作者 Xinxin Song、Siyuan Li、Tingxiong Xiao、Jinli Suo。标准 on-policy RLVR 在强化视觉 grounded 推理上存在瓶颈：有价值的视觉 grounded 轨迹更新一次就被丢弃，且均匀的 token advantage 分配无法强化关键的感知或推理步骤。**PIVOT** 提出双层框架：一是自校准的 experience replay，选择性地收集并回放视觉 grounded 的历史经验作为策略优化的稳定参照锚点；二是 vision-guided advantage allocation，根据 token 的局部视觉支撑及其对后续推理的影响，为其分配额外的视觉感知 advantage。在多个 benchmark 上取得极具竞争力的多模态推理表现（abstract 未给出具体数值）。
- **arXiv**：[2609.18057](https://arxiv.org/abs/2609.18057)

#### TIAO: Token Importance-Aware Policy Optimization for Text Summarization (2026-09)
- **简介**：作者 Qixiu Li、Chenlong Bao、Xiang Zhu 等。文本摘要的 RL 方法大多把 reward 直接施加到无差别的 token 序列上，忽略了各 token 对词级与句级质量的重要性差异。**TIAO**（Token Importance-Aware Policy Optimization）显式利用 token 重要性：基于 token 依赖关系识别核心 token，并依据轨迹整体依赖度对该轨迹的 advantage 重新加权。在真实世界数据集上取得极具竞争力的结果，经 TIAO 增强的 7B 基座模型可与 GPT-4 和 GPT-5-nano 表现相当。
- **arXiv**：[2609.16748](https://arxiv.org/abs/2609.16748)

#### VERPO: Verified Evidence Regularized Policy Optimization (2026-09)
- **简介**：Haijiang Li、Chengyu Lv、Yi Zhang、Zhibing Zhang 等 11 人。指出可验证 outcome reward 给出的 sequence-level advantage 无法识别哪些 token-level 决策应被保留或修改；用 evidence-conditioned teacher 回放采样轨迹可提供更密的监督，但无差别模仿会把与任务成功无关的格式或推理风格一并迁移过来。提出 **VERPO**，把 evidence 视为策略修正的"提议"而非模仿目标，在保留 outcome 目标的同时将 evidence-free 的 reference 恢复与带符号的 token-level evidence 修正解耦；Fisher Evidence Contrast 沿估计出的 evidence-presence 方向衰减修正强度，stopped token-wise ZPD controller 依据局部 reward 对齐度与 Fisher 移动代价缩放接受度，而 reference 通道保持独立于接受与否。在五个科学推理与工具使用任务上，各 backbone 的最佳变体平均分均超过最强对比基线：Qwen3-4B 从 0.6826 提升到 0.6857，Qwen3-8B 从 0.6895 提升到 0.7058，Llama-3.2-1B 从 0.4751 提升到 0.5657。
- **arXiv**：[2609.06100](https://arxiv.org/abs/2609.06100)

#### Cliff: Learning Process Rewards from the First Mistake (2026-09)
- **简介**：来自 Peixuan Han、Runhui Wang、Ketan Ramaneti、Jie Hao 等。针对 RLVR 依赖粗粒度 outcome reward、对中间推理过程缺少指导的问题，作者观察到：一旦推理过程首次出错，其后的推理已条件于一个无效前缀，再去评估它提供的额外信息很有限。据此提出 **Cliff** 这一 reward shaping 策略，用现成 LLM 作 teacher 定位每条 rollout 中的第一个错误，从而把 rollout 自然切成正确前缀与错误后缀，再转成 token-level advantage——前缀给正 advantage、之后给负反馈；相比 PRM 与 on-policy distillation，它既不需要专门的 reward model，也不假设 teacher 与 student 推理模式相同。跨 12 种场景的实验一致提升推理表现，比 on-policy distillation 高 15%、比标准 GRPO 高 7%，即使 teacher 能力平平也成立；论文还分析了 Cliff 中 ground truth 的作用及其训练动态。
- **arXiv**：[2609.02817](https://arxiv.org/abs/2609.02817)

#### GMTS: Gradient Magnitude-based Token Selection Improves RLVR Training for LLM Reasoning (2026-08)
- **简介**：来自 Outongyi Lv、Yuanwei Zhang、Xiaoqun Zhang。已有研究发现只用 entropy 最高的 20% token 训练即可获得显著增益，但高 entropy token 为何有益并未讲清。本文发现：同一 answer 内部高 entropy token 确实与较大梯度幅值相关，但考虑到 answer-level reward 信号本身存在差异，entropy 无法跨不同 answer 一致地反映 token 重要性。据此提出 **GMTS**（Gradient Magnitude-based Token Selection）来量化 token 重要性，利用 entropy 与梯度之间的联系来近似梯度幅值排序，从而完成 token 选择。实验表明按 GMTS 排序训练 top 20% token 在三个推理领域与多种模型规模上一致优于基于 entropy 的 token selection（abstract 未给出具体数值），说明 GMTS 对 token 贡献给出了更细粒度的估计。
- **arXiv**：[2608.30632](https://arxiv.org/abs/2608.30632)

#### Contrastive Branch Policy Optimization (CBPO) (2026-08)
- **简介**：Ying Wang、Changlin Qiu、Bang Lin、Linbo Jin 等 7 人。针对 RLVR 训练多轮工具交互时 sparse outcome reward 无法指明「哪一步中间决策导致成功」的问题，指出 branch sampling 虽能在候选续写间形成局部对比，但现有方法把两个不同问题混为一谈：固定 rollout 预算如何分配，以及如何把分支结果转成 token-level credit。**CBPO** 将二者解耦并各配专用机制：用 generation entropy 在整条回复上筛选候选分支位置，再用 path-level 与 node-level 衰减把固定预算分摊到不同轨迹与位置上，避免探索塌缩到少数路径或相邻 token；父轨迹与共享完全相同 token 前缀的分支构成 exact-prefix group，组内 reward 波动定义为 **Contrastive Branch Value（CBV）**，作为局部决策敏感度的 outcome-based 估计，用于缩放续写段的 advantage 而不改变其符号；当同一轨迹上选中多个节点时，CBPO 将其切成互不重叠的 credit segment，避免共享 token 上的梯度重复累加。全流程只需 outcome reward、无需过程级标注。在 10 个基准（5 个数学推理 + 5 个知识密集检索）与两种模型规模上一致优于 SOTA 策略优化与 branch-based 方法，两个领域的 macro-average 准确率均最高（abstract 未给出具体数值）。
- **arXiv**：[2608.24300](https://arxiv.org/abs/2608.24300)

#### Let Credit Follow Computation: Architecture-Aware Credit Transport for Large Language Model Reinforcement Learning (CompPO) (2026-08)
- **简介**：Qifan Shi、Zhaolu Kang、Chenghua Zhu。把 LLM RL 的 credit assignment 拆成三个对象：关于成功的 evidence、把 evidence 转成 token-level advantage 的 transport operator、以及把 advantage 转成策略变化的 update geometry；指出近期工作大幅改进了 evidence、采样与 update geometry，但 transport operator 仍与架构无关——固定折扣的 GAE 沿 token 时间施加平稳几何核，group-relative 方法则把一个结果统计量广播到整条回复，两者都不反映 Transformer 策略自身所执行的、轨迹特定的计算。作者提出 **computation-conditioned credit transport（CCT）** 通用框架：用行为策略内部计算的一个 detached 统计量来参数化传输下游价值的因果核。具体算法 **CompPO** 把原生 attention concentration 映射为有界的 per-token retention gate，在 one-step bootstrap 与路径依赖的广义优势轨迹（**Comp-GAE**）中同时使用该 gate，并协同设计了 transport-aligned critic（**TAC**）——复用 actor 的 hidden state 与 routing 信息，无需第二个同规模 Transformer；task reward 与 clipped PPO 目标保持不变，gate 取常数即退化为固定系数 GAE。在 5 个 Qwen3-4B 随机种子上，CompPO 最终留出准确率 61.4%（95% CI [60.8, 62.0]），而调优后的 GRPO 为 53.8% [52.9, 54.7]；Comp-GAE 配标准 critic（55.2%）与 TAC 配固定 gate（56.4%）均不及完整模型（交互项 +2.4 [1.9, 2.9]）。shuffle 与位置对照实验确认了轨迹特定的对齐性；CompPO 在 PPO 超参网格中 10/12 次运行稳定而基线仅 3/12；冻结评测下在 Qwen3-4B 与 Llama-3.1-8B-Instruct 上分别比 GRPO 高 4.3 与 3.9 个 greedy pass@1 macro 点。
- **arXiv**：[2608.21501](https://arxiv.org/abs/2608.21501)

#### FARCA: Fact-Aligned Reliability-Aware Credit Assignment for Reinforcement Learning with Factual Supervision (2026-08)
- **简介**：来自南京理工大学（Qiming Xie、Wenjie Zheng、Xiangqing Shen、Rui Xia）。为降低 RLVR 中 outcome-driven reward 带来的幻觉风险，已有工作引入过程级事实性监督，但由于事实信号聚合粒度过粗、且缺乏对这些信号本身可靠性的评估，导致事实核查与策略更新之间错配。作者将该问题命名为 **noisy factual credit assignment** 并分解为两方面：credit 定位模糊（credit localization ambiguity）与 credit 可靠性模糊（credit reliability ambiguity）。**FARCA** 把事实性监督转化为已定位、按可靠性加权的 token-level 训练信号：一方面令事实核查的粒度与策略更新的粒度对齐，实现细粒度 credit 定位；另一方面引入 **counterfactual evidence attribution**，以「某条事实判断对关键证据的依赖程度」作为核查可靠性的经验代理来计算可靠性权重，用该权重调制事实 reward 与局部 advantage，从而削弱潜在不可靠信号对策略优化的影响。在不同模型与多个事实性推理基准上显著提升事实性，同时保持通用推理能力（abstract 未给出具体数值）。
- **arXiv**：[2608.24350](https://arxiv.org/abs/2608.24350)

#### SRPO: Self-Reflective Policy Optimization for Long-Horizon Reasoning (2026-08)
- **简介**：来自上海交通大学、武汉大学等（Jialong Liu、Yuling Shi、Ning Yang、Xiaodong Gu、Zuchao Li）。自我反思是人类学习中强有力的 credit assignment 机制，能把稀疏的结果反馈转成可执行的指导，但其用于 LLM post-training 的潜力尚未被充分挖掘。**SRPO** 将该能力内化：让 LLM 分析自己已完成的轨迹，把错误归纳为简洁的「reflection patch」，再用以 reflection 为条件的 teacher 分数去评价 student 的 on-policy rollout，作为稠密的 token-level 训练信号——从而把稀疏的终端监督变成稠密 token 级信号，且不需要外部 critic、独立 reward model 或更大的 teacher 模型。以 Qwen3-8B base 为底，SRPO 在数学推理与长时程 agentic 基准上取得 SOTA 且数据效率极高：AIME'24 达到 73.3%，而训练 FLOPs 仅为放大规模 SFT 的 8%（0.08×）；同时把 WebShop 成功率提到 64.7%、ALFWorld 76.8%、SWE-Bench-Lite 31.2%，代码已开源。
- **arXiv**：[2608.23493](https://arxiv.org/abs/2608.23493)

#### Teach the Magnitude, Not the Direction: Verifier-Bounded Credit Assignment for Multi-Turn Multi-step LLM Agents (CrEST) (2026-08)
- **简介**：来自蚂蚁集团与浙江大学（Zechuan Wang、Siyuan Lu、Linjian Mo、Leilei Gan 等 6 人）。指出训练多轮工具使用 agent 时，RLVR 能提供 verifier 界定的性能上限，但其轨迹级 credit assignment 把异质的逐轮结果压成单一 reward 信号；而 on-policy distillation 虽提供稠密的 per-token 监督，却要么被 teacher 能力上限约束，要么出现梯度集中崩塌。提出 **CrEST**，一个分层 credit assignment 框架，在两个层级上分配 credit：turn-segmented verified advantage 解决轮间信号稀释，entropy-gated self-teacher modulation 细化轮内 token 的贡献，从而在保留 RL 的 verifier-bounded 上限的同时引入来自 privileged self-teacher 的稠密 token 级信号。在 BFCL V3 与 WildToolBench 上、两个模型规模下均一致优于 RL 与 distillation baseline，在长轨迹与严格 session 级指标上增益最大；作者的结论是 teacher 在策略优化中的角色可以从决定更新方向降级为调制更新幅度，从而在不牺牲 verifier-bounded 上限的前提下获得稠密 credit assignment（abstract 未给出具体数值）。
- **arXiv**：[2608.13179](https://arxiv.org/abs/2608.13179)

#### ReDiPPO: Reference-Guided Value Calibration and Discrepancy-Aware Token Reweighting for Mathematical Reasoning (2026-07)
- **简介**：来自中科大 / 讯飞（Zhenrong Zhang、Jun Du、Si Wei 等）。回归 PPO 路线：其 critic 原则上能做 token 级信用分配，但在长程、稀疏结果奖励的数学推理中标准 critic 难以准确评估中间状态。**ReDiPPO** 引入 reference-guided critic（训练时以参考答案为特权信号提供更准值估计），同时保留标准 critic，并量化二者的 token 级 reference-standard discrepancy 作为"困难推理状态"指示器，用它重加权对应 token 级优势。多个数学推理基准上值估计更准，最终推理性能一致超过 PPO、DAPO、GSPO 等强基线。
- **arXiv**：[2607.27631](https://arxiv.org/abs/2607.27631)

#### CoRT: Counterfactual Replay for Token-Level Rubric-Guided Policy Optimization (2026-07)
- **简介**：来自南京大学等（Bo-Wen Zhang、Lan-Zhe Guo 等 8 人）。针对 rubric-based RL 在 GRPO 管线中被压成一个 response 级标量、再均匀广播到所有 token、缺乏响应内信用分配的问题，提出 **CoRT**：不训练额外 token 打分器，而用 counterfactual replay 在"原 rubric 条件 prompt"与"无标准 criteria-free prompt"下对同一采样响应重打分，用逐 token 对数似然差作为"对 rubric 上下文依赖度"的代理，映射为有界、响应归一化的权重来重分配带符号的 GRPO 优势。跨指令微调模型与奖励粒度，绝大多数对照下优于同配 response 级 GRPO，平均提升 4.4 个百分点，且与需单独学习阶段的 token 级信用基线相当。
- **arXiv**：[2607.25659](https://arxiv.org/abs/2607.25659)

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

#### Double-Rollout Reinforcement Learning with Sampling-based Reward Estimation (2026-10)
- **简介**：作者 Haifeng Qian 等（NVIDIA 团队）。论文指出 RLVR 的结果奖励是推理质量的不完美度量：答案正确的 trace 未必推理可靠，答案错误的 rollout 也可能接近正确，由此产生的奖励噪声损害 GRPO 的稳定性。提出 **double-rollout RL**：对每条 rollout 截断其推理 trace，从截断前缀采样多个 continuation，用 continuation 的平均奖励替代原始奖励，得到更稳健的推理质量估计，仅带来适度的额外算力。仅用 DAPO 数据，Nemotron-3-Nano-30B-A3B 在 30 个 RL step 内达到 AIME 2026 95.4%（GRPO 为 91.5%）；结合长度奖励后，进一步改善了准确率-效率 Pareto 前沿。
- **arXiv**：[2610.16593](https://arxiv.org/abs/2610.16593)

#### Where the Model Changes Its Mind: Hindsight-Divergence Localization for Efficient Reinforcement Learning with Verifiable Rewards (HDL) (2026-09)
- **简介**：作者 Fanchao Chen、Hengyu Fu、Shivaram Venkataraman、Jiantao Jiao 指出，group-relative RLVR 独立采样完整轨迹成本高，且不会在关键位置显式探索决策空间。已完成轨迹上的反馈可揭示策略会重新考虑哪些早期选择，从而提示应在何处采样备选续写。提出 **Hindsight-Divergence Localization（HDL）**：利用 hindsight 引起的 token log-likelihood 变化选择分支点；只生成少量完整 root 轨迹，并用原任务上下文下从选定位置采样的续写填满每个训练组，续写复用 root 前缀，仅通过新生成的后缀贡献策略更新。在数学、代码、agent 三类任务、三个模型上，与同组大小同步数的 GRPO 相比，生成 token 最多减少 2.5 倍、rollout 时间加速 1.8 倍，且三个领域性能均提升，agent 任务最多提升 12.5 点。
- **arXiv**：[2609.36864](https://arxiv.org/abs/2609.36864)

#### EPIG-Tree: Compute-Optimal Branching for Gradient-Efficient Reinforcement Learning (2026-09)
- **简介**：作者 Nikita Khomich、Leopold Hermansson、Ido Hakimi。GRPO 把整条随机轨迹压缩成单个标量 reward，导致探索与 reward 分配低效。本文把 tree-based rollout 构造视为 policy gradient 估计的计算分配问题：分支应放在「额外一个分支最能降低 policy gradient 不确定性（按单位算力计）」的位置，而非策略仅仅不确定之处。基于 policy gradient 随机变量的全方差分解，推出两条分配律：新分支降低决策不确定性，重复的后缀 rollout 降低延续不确定性，由此得到 **EPIG-Tree** 评分，直接利用已计算的 rollout。实验中 EPIG 在 13 个控制环境的 sweep 里于全部 9 个稠密连续控制环境中降低梯度 MSE；在线单轮数学上，tree-local credit 优于 flat GRPO，但分支位置的影响次于 token 级 credit assignment；多轮 Wordle 上 EPIG 最终胜率 0.850，超过 flat GRPO 的 0.790。
- **arXiv**：[2609.20004](https://arxiv.org/abs/2609.20004)

#### HISPO: Hierarchical Importance-Sampling Policy Optimization with Entropy-Derived Segments (2026-09)
- **简介**：作者 Quoc-Vinh Lai-Dang、Hyo-Sang Shin。RLVR 中长回答的不同部分对最终正确性贡献不均，而现有目标要么在 token 级（GRPO、DAPO）、要么在序列级（GSPO）做 importance-sampling 修正。**HISPO** 提出介于两者之间的 segment 级策略优化：在 rollout 时依据熵切分出连续 segment，赋予基于熵的软 saliency 权重，并在 segment 粒度上做 clipped importance-sampling 修正。在 Qwen3-1.7B-Base 上做数学推理训练，六个 benchmark 上 Pass@8 均超过最强 baseline，其中五个 benchmark 的 Acc@8 持平或更优；AIME25 上相对 GRPO 提升 +3.75 Acc@8 / +3.78 Pass@8，相对 GSPO 提升 +2.50 Acc@8 / +1.27 Pass@8。
- **arXiv**：[2609.15471](https://arxiv.org/abs/2609.15471)

> 2025.10 以来 CA 粒度从 token / trajectory 两个极端向 segment / sub-trajectory / turn-cluster 中间层收敛。

#### Fork Where the Model Changes Its Mind: Belief-Shift Branching for Tree-Structured Reinforcement Learning (2026-09)
- **简介**：Bin Lei、Yu Li、Prafulla Kumar Choubey、Silvio Savarese、Chien-Sheng Wu 等 10 人（含 Salesforce AI Research）。指出树状 rollout 能为 critic-free RLVR 提供 step-level credit，但每个 fork 都要额外采样，现实预算下每条链只能分叉几次；若 fork 落在结果已基本确定的位置，兄弟轨迹会高度一致、几乎不提供 credit 信号，因此在给定树规模下 fork 的位置基本决定了 step-level RL 的收益上限。作者把 fork 放置形式化为定位价值曲线的 pivot（期望结果发生转折之处），提出 **belief-shift branching**：读取模型在候选步边界处的 answer belief，在相邻 belief 分歧最大的那一步之前分叉；给出三种无需 step-level 监督、覆盖不同访问权限的实现——black-box probe、logit-lens depth profile，以及离线拟合、仅用于 RL 前验证阶段的 learned activation direction。该信号只决定 fork 位置，probe 在数学上约占 step 计算量的 1%、代码上低于 5%。对照 Monte-Carlo 价值曲线，belief-shift 信号在全部 8 个 model×benchmark 面板中排名第一，领先 entropy、结构化与 LLM-judge 基线；RL 实验覆盖三个模型族两个领域，在 OLMo-3-7B 上数学聚合分比最强基线高 +2.6、AIME 2026 高 +2.9，并横扫 OLMo 全部代码列，LiveCodeBench-medium 高 +6.5。
- **arXiv**：[2609.11061](https://arxiv.org/abs/2609.11061)

#### Long-Horizon Language Model Reinforcement Learning via Progressive Point Matching (2026-09)
- **简介**：来自 UC Berkeley 与 CMU（Preston Fu、Kevin Frans、Oleh Rybkin、Sergey Levine、Aviral Kumar）。指出当前语言模型 RL 严重依赖稀疏 outcome reward，在需要更长更复杂轨迹的任务上学习极慢；此前奖励部分进展的做法形式往往有偏，会收敛到次优策略。作者提出一种简单且无偏的 dense reward 形式 **progressive point matching**，在 segment level 上奖励部分进展，并从理论与合成环境两方面证明其随任务 horizon 的扩展效率呈指数级更优；实际落地时每个任务只需一条 reference trajectory 即可实例化。在极难数学推理题上，稀疏 outcome reward 完全无法取得进展，而 segment-level reward 在更大的 test-time token 预算下无论以成功率还是 pass@k 衡量都能带来改善（abstract 未给出具体数值）。
- **arXiv**：[2609.07303](https://arxiv.org/abs/2609.07303)

#### Tracking the Moving Frontier: Long-Short Term Advantage Estimator (LSTAE) (2026-09)
- **简介**：Xinhao Yao、Lu Yu、Changhao Wang、Fengwei Teng 等 8 人。指出 group-based RLVR 需为每个 prompt 重复采样多条轨迹来估计 advantage，使长程 agent 训练代价高昂，同时又丢弃了跨迭代积累的经验，遂追问能否用历史经验替代这些组内重复比较、且不直接在陈旧轨迹上做优化。提出 **LSTAE**，一种 single-stream RL 算法：历史只用于 advantage 估计，策略更新仍只用当前 rollout。它为每个 task anchor 维护一个持久 tracker——轨迹级（long term）用 drift-aware 的历史 baseline 追踪该 anchor 不断移动的成功前沿，衡量每条新轨迹的相对贡献；步级（short term）用一个最近 state-experience buffer，利用反复出现的 state 估计局部 action advantage。这种双时间尺度设计把累积经验转成多粒度 credit 信号，每个 anchor 只需一次 rollout。在 agentic 与数学推理基准上，LSTAE 匹配或超过强 group-based 基线，同时大幅降低 rollout 成本（abstract 未给出具体数值）。
- **arXiv**：[2609.06671](https://arxiv.org/abs/2609.06671)

#### Learning Where Outcomes Change: Credit-Addressable Reasoning for Multimodal Geometry (CE-GRPO) (2026-08)
- **简介**：来自 Jiani Guo、Junjie Wang、Jie Wu 等，含 Shaohan Huang、Furu Wei 等微软研究院作者。多模态几何推理要求 VLM 抽取精确的视觉关系并在多步演绎中保持它们，但自由形式的推理 trace 掩盖了真正决定答案的决策，而 trajectory-level RL 又把单一终端信号摊到整个 response 上。作者提出 credit-addressable reasoning 原则：推理时暴露的语义单元同时定义学习在何处比较备选、在何处分配 credit。该原则由两部分实例化——**Code-CoT** 保留图形、把视觉关系表示为可按行寻址的可执行代码，并把推理组织为 typed event；**CE-GRPO** 用结构先验与 type-normalized entropy 选取事件边界，从共享前缀采样完整续写，把 outcome 差异转化为局部化的 advantage。九个几何基准上 CE-GRPO 平均准确率 76.04，比 Qwen3-VL-8B 高 8.09 分、比 trajectory-level GRPO 高 3.43 分，且中间事件数越多其相对优势越大。
- **arXiv**：[2608.30457](https://arxiv.org/abs/2608.30457)

#### Ockhamareto: Pareto-Gated Segment-Level Credit Assignment for Concise Unit-Test Generation with Reinforcement Learning (2026-08)
- **简介**：Dong Huang、Mark Harman、Jie M. Zhang、Zhijiang Guo 等 6 人。面向单元测试生成与选择，提出基于奥卡姆剃刀与 Pareto 最优思想的单次（single-shot）GRPO 框架 **Ockhamareto**，含两个主要组件：（i）**Pareto-gated Bonus**，只奖励在（mutation score, −测试数量）空间中不被支配（non-dominated）的 rollout，从而同时压住「多杀变异」与「少写测试」两个目标；（ii）**Token-level Segment Credit**，把每个测试对变异杀死数的边际贡献回溯归因到该单元测试代码块的 token 上，实现 segment 级 credit assignment。在 UnLeakedTestBench（ULT）上严格 Pareto 支配最强 RL baseline MIST-RL，并在每一个优化目标上全面占优：N=5 时 mutation score 49.9% vs 31.3%，平均测试数 2.60 vs 4.67，单测试性价比提升 3.4 倍；在 HumanEval+、MBPP+、CodeContests、TestGenEval-Lite 四个基准上 mutation 与 coverage 指标均领先且测试套件始终最小；在 4B / 9B / 27B 三种规模上均超过 SOTA，mutation 提升 +30~35 个百分点。作者还发现 Pareto 前沿上效率—有效性最优权衡的膝点与函数规模等易算代理指标不相关，因而必须真正计算 Pareto 前沿才能为每个被测函数定位这一关键工程权衡。
- **arXiv**：[2608.24473](https://arxiv.org/abs/2608.24473)

#### Latent Thought Credit: Multi-Answer Credit Assignment for Latent Reasoning (LTC) (2026-08)
- **简介**：来自 Xuyang Zhao、Liting Zhang、Zichen Xu、Yong Chen 等 7 人。latent reasoning 让模型在连续潜表示中完成中间推理而不必完全外化为离散 CoT，但仅靠答案级 reward 很难对潜在思维做 credit assignment——单个最终答案把思维质量与答案采样噪声混在了一起。提出 **Latent Thought Credit (LTC)** 分层 credit assignment 框架：对每个 prompt 采样多个 latent thought，固定每个 thought 之后的上下文，再从该固定上下文生成多个答案并对 reward 取平均，以此估计 thought-level 期望 reward；随后用 thought-level advantage 优化潜在思维阶段、用 answer-level advantage 优化答案阶段，并加入 advantage 加权的 thought-matching 目标，帮助策略复现高 credit 的潜在思维。在 GRPO 式 on-policy 框架中实现，数学推理与 STEM 多选任务上取得对比方法中最好的平均准确率；消融与固定上下文诊断表明多答案估计降低了 reward 估计误差，缓解了模糊或错误的 thought-level credit。
- **arXiv**：[2608.01593](https://arxiv.org/abs/2608.01593)

#### SP3O: Reinforcement Learning from Segment Preferences without Reward Modeling (2026-08)
- **简介**：来自 Evan Assmus、Qining Zhang、Lei Ying。指出一般随机 MDP 上的偏好强化学习（PbRL）通常需要训练 reward model，而已有免 reward model 的方法要么局限于 bandit 或确定性 MDP（如 DPO、P3O），要么采用收敛更慢的零阶无梯度优化；并且这些方法几乎只使用轨迹级反馈，轨迹很长时人类评估者负担极大，而 segment 更短、更易比较。提出 **SP3O**（Segment Pairwise Proximal Policy Optimization）：一种免 reward model、免 critic 且基于梯度的 PbRL 算法，用 segment 级偏好反馈通过 off-policy importance sampling 构造精确的策略价值差估计量，再据此以 PPO 型损失计算策略梯度。论文给出理论依据并分析 segment 长度选择的权衡，在机器人控制与 LLM 微调两类设定下对比其他 PbRL/RLHF 算法均取得更好表现，长时程任务上优势尤为明显。
- **arXiv**：[2608.02951](https://arxiv.org/abs/2608.02951)

#### Fishing Out Free Riders: Shapley-Based Reward Attribution for Parallel Reasoning via Reinforcement Learning (Parallel Shapley) (2026-07)
- **简介**：Wentao Zhang、Haoyu Zhang 等（ShanghaiTech 等）针对并行推理中"结果级奖励对所有路径一视同仁、无法区分冗余/误导路径"的问题，提出 **Parallel Shapley**：把每条推理路径视为合作博弈中的玩家，用 Shapley 值量化其边际贡献；以 generative reward model 评估路径效用、Monte Carlo 采样近似 Shapley。在数学推理基准上优于既有基线，训练更稳定、可解释，能"钓出搭便车者"（free riders）并按贡献成比例分配奖励。
- **arXiv**：[2607.18979](https://arxiv.org/abs/2607.18979)

#### Reasoning Error from Known Fact: Step-Level Self-Consistency Group Relative Policy Optimization for LLM (SSC-GRPO) (2026-07)
- **简介**：Xiaomeng Hu、Junbo Zhao 等（浙大等）细粒度分析长 CoT 中的幻觉，识别出"上下文敏感事实幻觉"（模型本有相关知识、却因推理中的上下文干扰而出错）。提出 **SSC-GRPO**：跨多次 rollout 计算各步骤的自一致性分数，据此给推理轨迹分配 **step-level 奖励**。相较既有方法，在数学推理基准与幻觉排行榜上均取得 SOTA，为推理过程中幻觉的检测与缓解提供新视角。
- **arXiv**：[2607.18915](https://arxiv.org/abs/2607.18915)

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

#### CRoS: Causality Reward on Scrambled CoTs Improves Accuracy and Reasoning Faithfulness (2026-10)
- **简介**：作者 Xiao Yu、Justin Chih-Yao Chen、Zaid Khan、Hyunji Lee、Mohit Bansal。RLVR 仅依赖结果正确性，无法保证 CoT 对最终答案有因果贡献；已有因果性度量通过删除步骤或注入错误来扰动 CoT，但这会改变步骤信息，高因果分可能只是对信息变化敏感，作为奖励还会导致 CoT 变短、探索与自我纠正减少。**CRoS** 改为只打乱推理步骤顺序、不改动内容，从而保留每步信息，作为因果性奖励。在 Reasoning Gym + Gemma-3-4B 上，CRoS 在 OOD 准确率、CoT 质量与忠实度上优于现有因果奖励与结果奖励：在五个未见的逻辑与游戏数据集上准确率 46.8%，高于最强基线的 43.8%。
- **arXiv**：[2610.42054](https://arxiv.org/abs/2610.42054)

#### Hacking or Just Failing? A More Capable LLM Monitor Better Separates Reward Hacking from Honest Failure (2026-10)
- **简介**：作者 Kaushik Kezendirane。agent 的可见得分高于真实进展，可能是因为刷分，也可能只是犯错，有用的 reward-hacking 监控器必须区分二者。作者构建了一个可刷分的沙盒任务，通过回放并反事实地修复最终状态来给每个 episode 标注，进行了五项预注册研究。在 Llama 3.1 8B agent 的 60 个 episode 上，小型本地监控器（Llama 3B/8B）无法区分利用与诚实失败（预注册主监控器 AUC 0.64）；而更强的 Gemini 3.1 Flash-Lite 达到 AUC 0.80（95% 区间 [0.66, 0.93]），仅误报 1/35 次诚实失败，但利用样本仅 14 个、估计不确定。对未在 prompt 中点名的利用方式，Gemini 常能正确描述动作却从不标记；一个两规则的确定性检查则抓住了所有已知利用。作者公开环境、标注与预注册。
- **arXiv**：[2611.00407](https://arxiv.org/abs/2611.00407)

#### Audit Where Hacks Pay: Budgeted Strong-Verifier Audits Against Reward Hacking in GRPO (2026-10)
- **简介**：作者 Shurun Li、Michael Hua Wang 等。RLVR 用单元测试等廉价检查训练，策略可能学会钻空子，而强验证器又太贵，无法对每条 rollout 运行。论文研究如何在 GRPO 中分配有限的强验证器预算，把「审计哪些弱通过 rollout」与「被抓到的 hack 得到什么奖励」分开分析，并证明：优先审计弱通过数最少的整组（low-pass budgeted audits, **LB**）能最小化留在 hack 上的 advantage 质量。在 LeetCode 与 Countdown-Code 上，LB 将 hacking 压到接近零，而相同预算和惩罚的随机审计做不到；full audit 的效果与 LB 相差约两个点以内，而 LB 仅用约 28% 的标签。理论预测的符号在 39 次运行中对 33 次判断正确。
- **arXiv**：[2610.35716](https://arxiv.org/abs/2610.35716)

#### Mitigating Reward Hacking in Preference Reward Models with Evolving Penalties (2026-10)
- **简介**：作者 Afonso Baldo、António Farinhas 等。对无法验证、难以写成 rubric 的开放式任务，偏好 reward model 仍很有价值，但它容易被 reward hacking：策略提高分数却没有提升甚至损害回复质量。论文提出 **Dynalize**，在策略训练过程中在线缓解 hacking：周期性、异步地由一个语言模型分析训练 rollout，针对 reward model 似乎在奖励的不良行为撰写惩罚项，再由 judge 将其应用到每条 rollout。在临床咨询与创意写作上，仅用 reward model 训练会使其得分持续上升但外部 benchmark 最终下降；Dynalize 可维持更长训练并提升下游表现，能抑制偏好模型所激励、并扩散到绝大多数回复乃至推理过程的模式，其收益在多轮训练中仍成立，且在用发现的行为重训 reward model 后依然保持。（abstract 未给出具体数值）
- **arXiv**：[2610.26496](https://arxiv.org/abs/2610.26496)

#### SimHack: Predicting the onset of detectable reward hacking before RL training (2026-10)
- **简介**：作者 Junwei Deng、Alex Bie、Natalia Ponomareva。LLM 代码 RL 中 reward hacking 常可检测但难以预防：环境加固随复杂度增加越来越难，训练期干预又会激励模型混淆 hacking 行为。论文转而研究能否在运行 RL 之前预测可检测 hacking 何时占主导，以提前选择风险更低的训练配方。作者发现初始 hacking 率不能单独决定 onset，且随配方与重复运行差异很大，因此把问题表述为预测目标配方下 hacking onset step 的分布，并提出 **SimHack**：从少量较小规模的参考 RL run 学习 hacking 训练动态，对未见目标配方生成反事实模拟 run 而无需重新训练。对未见配方，SimHack 预测中位 onset 的相对误差为 11.3%，跨配方排序相关系数 0.85，并能泛化到参考与目标之间模型规模、课程、rollout 数和学习率的差异；用于筛选 RL 课程时，可找到在无 hacking 情况下多训练 96.6% 时长的课程。
- **arXiv**：[2610.28707](https://arxiv.org/abs/2610.28707)

#### Reward Misspecification Degrades Chain-of-Thought Monitorability (2026-10)
- **简介**：作者 Tianyi Lorena Yan、John Hewitt。CoT 监控是 AI 安全中检测与防止不当行为的关键策略，需找出导致 CoT 不可监控的 RL 做法；其中 reward misspecification（奖励函数与真实任务意图不一致）几乎不可避免。论文构建了成对的「与任务意图对齐的真奖励」与「偏好另一行为的误设奖励」测试床，使 reward hacking 可被训练诱发并可验证标注。在三个模型、三个任务上，随着 hack 率上升 CoT 可监控性仍然很低：CoT 仍在推理任务意图，而输出违背意图却获得奖励。为在不知真奖励与 hack 方式的前提下缓解，作者收集训练各步不同奖励下的 rollout，用 LM 推断奖励描述并注入输入，教模型把奖励说出来（reward description training），使可监控性按不同指标平均绝对提升 24% 或 42%，并使 CoT 显式推理误设奖励。代码已公开。
- **arXiv**：[2610.28303](https://arxiv.org/abs/2610.28303)

#### Multi-Reward Hacking: From Aggregation-Induced Drift to Trust-Anchored Defense (2026-10)
- **简介**：作者 Yushu Li、Jiin Woo、Xiaoxiao Li、Youngsuk Park。LLM 后训练常把主奖励（如正确性）与辅助奖励（如简洁、效率）聚合；论文表明对称聚合会引发「聚合奖励上升而主任务性能下降」的 multi-reward hacking，并证明当辅助奖励更易获得时，确定性条件均值 GRPO 在对称聚合下会收敛到偏向辅助奖励的行为。提出 **Trust-Anchored Gating（TAG）**：先根据单独优化每个奖励通道所引起的策略漂移选出 anchor 通道，再把辅助奖励的信用门控在 anchor 被满足的前提下，从结构上消除仅满足辅助项的信用，同时保留联合成功的额外信用。在 Qwen 与 Llama 的数学和程序合成实验中，Qwen3-4B 加简洁奖励时 TAG 的 pass@4 为 0.581，加法聚合为 0.220，响应长度比仅用正确性奖励缩短 44%。
- **arXiv**：[2610.21404](https://arxiv.org/abs/2610.21404)

#### Rent-Seeking Language Models: Reward Models Set the Prices, Frequent Audits Curb the Rent (2026-10)
- **简介**：作者 Zixiao Wang、Xingbo Du、Duzhen Zhang、Le Song。论文把 RLHF 中的 reward model overoptimization 重新表述为寻租（rent-seeking）：策略是企业，从代理奖励模型这一「监管者」处获得并不被真实质量支撑的奖励，由此得到 rent、capture index、deadweight loss 等可度量量。在含三个模型家族、准确率匹配的代理 RM 与两个 gold RM 的固定测试平台上，RL 把 41% 到 66% 的代理增益转为 rent，而 gold 奖励停滞。两种代理 RM 都不验证事实，但一个为「自信的细节」付费，另一个为「勤勉的表象」付费，因而 RL 策略学会编造不存在实体的细节（97% 到 100% 的探针）或回避事实性问题（67% 到 98%）。在固定期望罚金下，频繁的轻度审计优于稀少的严厉审计，随机审计与定向审计的单次效率相当；作者还首次观察到策略学会停留在定向审计阈值之下的迹象。
- **arXiv**：[2610.19832](https://arxiv.org/abs/2610.19832)

#### Gradient Routing Localizes and Removes Unintended Behaviors in RL (GRAFT) (2026-10)
- **简介**：作者 Jake Ward、Shawn Hu、Aria Wong、Nathan Hu、Alexander Matt Turner 等。设定为：奖励函数被误设会导致非预期行为，而用于检测它的 monitor 本身会系统性出错，因此用 monitor 过滤数据或施加奖励惩罚都无法阻止模型学到 reward hacking。提出 **GRAFT（Gradient Routing Adapter Fine Tuning）**：不惩罚或过滤被标记样本，而是把这些样本的梯度更新约束在一组新增的「牺牲」参数上。好处有二：monitor 不进入奖励函数，模型不会学着隐藏不当行为；消融这些牺牲参数即可降低或完全停止非预期行为，同时保留目标任务上的提升。在奖励惩罚会让模型学会隐蔽 hack 的 RL 环境中，对 135M 到 8B 参数的模型验证有效；并通过把含 reward hacking 的前沿 LLM 轨迹蒸馏进 Qwen3-32B，在更大规模上展示类似效果且不教会模型 hack。
- **arXiv**：[2610.17062](https://arxiv.org/abs/2610.17062)

#### hacktrace: behavior-supervised detection of reward hacking during code generation (2026-10)
- **简介**：作者 Hao Jiang、Xin Li、Annan Wang、Yichi Zhang、Weisi Lin。编码 agent 可以通过修复代码拿到通过的评分，也可以通过删除暴露 bug 的测试拿到。检测这类 reward hacking 需要识别包括失败尝试在内的捷径行为。作者发布来自 Qwen3-8B 的 173,561 条带标注多轮编码轨迹，表明独立于 exploit 成功与否去监督捷径行为能大幅提升检测。**HACKTRACE** 是行为监督的监控器，读取 agent 生成代码时已经计算的内部状态，可在回合结束前监控，无需额外 LM token 或前向；结合最终文件静态特征，平均每题 AUC 达 0.997，监控开销 8 ms。同样的生成状态还可作 RL 监控信号：在强 GRPO 惩罚下，把通过解中作弊的占比从 82-91% 降至 1-5%，同时保留诚实正确的解。
- **arXiv**：[2610.03055](https://arxiv.org/abs/2610.03055)

#### How to post-train on a surrogate: Envelope sampling mitigates reward hacking (2026-10)
- **简介**：作者 Sanjit Dandapanthula、Shuvom Sadhuka、Samir Khan、Michael Oberst、Aaditya Ramdas、Alexandra Chouldechova。LLM 常针对 LLM judge 等廉价代理做后训练，因为真实奖励（如人类偏好）太贵，但这常导致 reward hacking。论文研究少量 n 个模型输出带有真值标注（如专家审阅）、用于在优化前重新校准 LLM judge 的设定。已有的 judge 重校准方法昂贵或启发式，且已知当代理在稀有输出集合上失准时 on-policy 采样会失败。作者提出 **envelope sampling**：在人类奖励与重校准奖励位于 judge 周围 L2 球内的假设下，最小化后训练模型 regret 上界的有理论依据的方法，并给出通过拒绝采样或对修改后奖励微调来从 envelope 采样的实用算法。在临床病历生成及受控 sycophancy 任务上，基于 envelope 样本的重校准能缓解 reward hacking，而基于基座模型样本的重校准做不到。
- **arXiv**：[2610.11281](https://arxiv.org/abs/2610.11281)

#### Semifactual Credit-Augmented Policy Optimization (SCAPO) (2026-09)
- **简介**：作者 Junshu Pan、Zhizhang Fu、Shulin Huang、Wenqi Shao 等发现 RLVR 训练的 LLM 对与任务无关的 prompt 特征仍然敏感。通过保持问题与答案不变的 semifactual prompt 干预分析发现 token 级敏感度差异显著，且在解码时抑制高漂移 token 候选即可无需更新权重地提升推理精度；这暴露了 GRPO 把同一结果 advantage 分给每个 token、可能同时强化虚假依赖的局限。提出受因果启发的 GRPO 变体 **SCAPO**：在固定响应上度量 semifactual 干预下的 token 概率漂移，用归一化稳定性分数在训练早期降低相对不稳定 token 的 advantage，且不对稳定性本身额外加分。在 Qwen3-4B-Base 与 Qwen3-1.7B-Base 上，AIME 2024-2026 精度较 GRPO 分别提高 5.63 和 4.17 个百分点，并在大多数数学基准和全部 OOD 基准上取得对比方法中的最佳结果。
- **arXiv**：[2609.40360](https://arxiv.org/abs/2609.40360)

#### CATCH: A Controllable Analysis Testbed for Reward Hacking in Coding RL (2026-09)
- **简介**：作者 Shouli Wang、Yanfeng Jia、Zhihao Ou 等指出 RLVR 中模型可利用环境漏洞获取高 reward 而未提升真实能力（reward hacking），但缺少能复现并可靠识别 hacking 的测试床。提出 **CATCH**：在代码 RL 中刻意暴露环境漏洞，通过比较「脆弱评估器下的成功」与「独立审计下的任务正确性」给出基于执行的 gold 标签，还可通过 SFT 数据混合控制模型初始 hacking 倾向、通过 reward 设计控制获得 reward 的难度。实验表明 CATCH 能产生明显 reward hacking 的多样训练轨迹，初始模型和 reward 难度都会塑造 hacking 的出现；并评估了多种检测与缓解方法。关键发现是 CoT monitor 起初抑制 hacking，但随着策略学会用代码注释误导 monitor，这种保护逐渐失效，说明需在整个训练过程中评估缓解手段。代码与资源已公开。
- **arXiv**：[2609.39533](https://arxiv.org/abs/2609.39533)

#### Verifier Errors in RLVR: Reward Hacking, Limits of Feedback, and Selective Control (2026-09)
- **简介**：作者 Christian Moya、Elliott Thornley、Guang Lin 研究 RLVR 中不完美 verifier 会奖励错误响应、带来 reward hacking 的问题。在固定 verifier 的梯度流下刻画了「reward 上升而正确率下降」的条件；并证明 RLVR 期间可获得的观测一般不足以检测或识别被接受的错误，也无法保证在不牺牲正确响应的前提下减少这些错误。为突破此限制，构造了利用审计提供的额外正确性反馈的修正项，实现 **selective control**：在当前策略处降低被接受错误的概率并提高正确响应的概率，前提是该修正压过 verifier reward 对错误的推动力。log-linear 与神经 contextual bandit 以及语言模型上的实验支持分析，显示部分审计下的 selective control 在提升正确率的同时降低被接受错误。
- **arXiv**：[2609.35677](https://arxiv.org/abs/2609.35677)

#### Scoring Higher, Answering Worse: Mitigating Reward Hacking in Rubric-Based RL via Protocol-Level Rubrics (ProRubric) (2026-09)
- **简介**：作者 Maoqi Liu、Junwei He、Bowen Zhang 等指出 Rubric-based RL 中 judge 逐条检查 rubric 再按加权和聚合为 reward，这种加性聚合是弱点：各条标准可以互相补偿，策略漏掉关键决策后可以靠无人要求的建议「买回」分数。在临床问诊中，这类策略 rubric 覆盖率上升，但在 held-out 医生标准上的适当性反而低于未训练模型；把相同标准分组使其必须同时成立，即可挽回约三分之一的损失。提出 **ProRubric**：将 checklist 离线一次性分为少数 protocol 级维度，仅当维度内所有标准成立且失败条款未触发时才计分，不改变优化器。ProRubric 在不损失覆盖率下将适当性提升 10.8 点，并在两种规模上取得最佳七基准平均。代码已开源。
- **arXiv**：[2609.38847](https://arxiv.org/abs/2609.38847)

#### CounterRoute: Self-Routed Reasoning via Hierarchical Counterfactual Credit Assignment (2026-09)
- **简介**：作者 Ruochen Jiao、Besnik Fetahu、Zhenyu Shi、Priyanka Nigam。推理模型常在直接回答即可时仍输出长 CoT；双模式模型通常把 thinking/direct 的选择交给用户。**CounterRoute** 是在线 RL 框架，直接从原生双模式 checkpoint 出发、无需 SFT warm-up，在一个共享策略中联合学习路由与模式条件下的回答：用成对的当前策略反事实 rollout 估计跨模式路由 credit，仅分配给路由 token，模式内用 GRPO 训练回答 token；并以 paired-to-self-routed 课程稳定早期训练。九个基准上在准确率与效率间优于启发式与学习式路由方法，相比始终思考的 checkpoint，两个 Qwen3 规模的平均生成 token 分别减少 51% 和 41%；仅在数学与合成指令遵循数据上训练，即可泛化到编码、科学、常识等未见任务。
- **arXiv**：[2609.29109](https://arxiv.org/abs/2609.29109)

#### Taming CoT Obfuscation in VLMs: From Mechanistic Evidence to Activation-Level Enforcement (TAME) (2026-09)
- **简介**：作者 Xutao Mao、Jianing Zhu、Jinman Zhao 等（NeurIPS 2026）。RL 提升 VLM 推理的同时，可能诱发 CoT obfuscation：模型准确率高，但推理轨迹缺乏 grounding、更难监控。论文发现这种退化伴随表示层面的偏移——RL 优化使与任务无关的模板特征在激活空间中与视觉 grounded 内容更难分离。据此提出 **TAME**：用 Sparse Autoencoder (SAE) 在 RL 训练中正则化 VLM 内部特征，结合 anti-obfuscation 反馈回路降低行为层面的 obfuscation，并对特定内部方向施加非对称 SAE 约束，惩罚模板激活而保留视觉证据。在 VIRL-39k 与 SPA-VL 的两个模型族上，CoT 可监控性相对 GRPO 最多提升 +30.9（VIRL）与 +16.7（SPA-VL）点，同时基本保持任务准确率与通用能力，存在任务相关的权衡。
- **arXiv**：[2609.24243](https://arxiv.org/abs/2609.24243)

#### Optimizing the Score, Losing Sight of the Task: Reward Hacking Across Weights, Selection, and Prompts (2026-09)
- **简介**：作者 Vansh Wahi。更高的评测分数并不总意味着更好的系统：当优化利用了评估器的错误时，测得的进步可能掩盖任务性能不变甚至下降。这种失败可发生在参数更新、生成输出的选择、或持久 prompt 的修订三种优化载体上。论文建立了跨 weights、selection、text 三种载体比较 reward hacking 的框架，基于 Proxy Compression Hypothesis 及推理时/上下文内 reward hacking 研究，形式化了评估器分歧的距离相关上界和嵌套策略类的容量排序，并说明仅靠距离无法给出普适的脆弱性排序；用精确的有限输出例子展示打分缺陷的位置如何改变各方法所偏好的行为，还梳理了各载体上有代表性的防御哪些可直接迁移。结论：可靠的提升取决于控制可达的失败模式，并保留独立于被优化分数的任务质量证据（理论分析，abstract 未给出具体实验数值）。
- **arXiv**：[2609.25848](https://arxiv.org/abs/2609.25848)

#### Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations (2026-09)
- **简介**：作者 Leon Bergen、Usha Bhalla、Andrew Lee 等。随着模型规模增大 reward hacking 更频繁、更复杂。作者分析 reward hacking 在前沿开源 LLM 内部表征中的体现，发现简单的 difference-of-means（DoM）向量就能在多种常见评测行为中一致地表示 reward hacking，且可泛化、可解释，能可靠地检测。在 DeepSWE、SWE-bench 等评测中模型 hack 现象严重：GLM 5.2 在 DeepSWE 上 57.2% 的 rollout、在 SWE-bench 上 73% 的 rollout 出现 hacking。DoM 向量的检测效果与 LLM monitor 相当但几乎零成本；作用于 CoT 时还能在后续动作之前预测 hack，可在线拦截；分析其命中而 LLM monitor 漏掉的样例还发现了其他不良行为。
- **arXiv**：[2609.19101](https://arxiv.org/abs/2609.19101)

#### Inducing Emergent Misalignment from Reward Hacks with Iterative DPO (2026-09)
- **简介**：Oliver Daniels、Perusha Moodley、Benjamin M. Marlin、David Lindner。动机是 RLVR 训练中的 reward hacking 会诱发 reward seeking 与广泛的 misalignment，研究这种误泛化对建立威胁模型和对策很重要，但在大模型上跑 RL 的成本使其常常不可行。作者提出改用 iterative DPO 来研究涌现失配：它保留了 RLVR 的若干关键性质，同时成本更低，并且可以在主流 finetuning API 上训练。实验发现，在单轮 reward hacking 环境上用 iterative DPO 训练 GPT-4.1 会诱发隐蔽的 misaligned power-seeking 与 alignment faking，这是首个公开可得、能诱发这两类令人担忧失配形态的（半）在线训练 pipeline；用同一 pipeline 训练 Qwen2.5-32B-Instruct 则同时产生 misalignment 与 instruction following 准确率的提升，说明 iterative DPO 也可作为 selective generalization 的 testbed。属 §1.6 的 reward hacking 诱发涌现失配现象研究，贡献在低成本研究范式与实证发现，而非新的优化算法。
- **arXiv**：[2609.06649](https://arxiv.org/abs/2609.06649)

#### Are Verifier Errors Independent Within a GRPO Group? Evidence from Qwen2.5 Rollouts (2026-09)
- **简介**：Esther Xin（单作者）。指出 group-based RLVR 用自动 verifier 为同一 prompt 的多条 completion 打分，而假定 verifier 误差独立的分析可能忽略了共享答案格式带来的依赖。作者在 Qwen2.5-1.5B 于 MATH、GSM8K、DeepMath-103K 上生成的 24,998 个八条 completion 的 group 上直接测量：估计组内 verifier-error 相关系数为 0.530（95% 置信区间 0.500–0.560）；在 exchangeable-error 模型下，这相当于八条 completion 的一个 group 经 design effect 调整后有效样本量仅为 1.70。依赖强度随答案形式差异明显——分数、根式、符号表达式与区间的聚集程度强于单位标注和百分号；在四种 rule-based verifier 配置下重放 group-relative advantage，最多 0.83% 的 group 出现至少一次 advantage 符号分歧。作者也说明由于一个 group 就是对同一 prompt 的重复采样，组内聚集可能同时反映共享 prompt 难度与共享答案形式，本文不试图分离二者。属 §1.6 的 verifier 噪声结构实证测量，结论是应转向 prompt 与答案形式感知的 verifier 噪声分析，而非只看聚合错误率。
- **arXiv**：[2609.06386](https://arxiv.org/abs/2609.06386)

#### CARE: Contrastive Anchor-based Rubric Evolution for Large Language Model Post-Training (2026-09)
- **简介**：来自 Siyuan Li、Xinxin Song、Chen Ruinian、Jingjing Fan 等。rubric-based RL 把开放式指令分解为 prompt 专属的灵活 rubric，在开放式任务后训练上比 RLVR 更合适，但静态 rubric 随策略演化必然被 hack，而已有动态方案又引入新问题：rubric 抽取缺乏方向、hack 检测不可靠、rubric 无界膨胀。提出 **CARE**（Contrastive Anchor-based Rubric Evolution），把每一步 rubric 演化都锚定在由前沿模型基于 prompt 及其 rubrics 生成的高质量 anchor response 上：每个训练步将得分最高的 rollout 与 anchor 对比，触发两条互补机制——Adaptive 分支被动修补 reward misspecification，Chase 分支主动把与前沿水平的质量差转化为更锐利的 rubric；两者共同在高 reward 区（reward over-optimization 主要发源地）维持判别精度。在 WildChecklist-9K 上以 Qwen2.5-7B-Base 与 Qwen2.5-7B-Instruct 训练，CARE 于 Arena-Hard-2.0、InfoBench、FollowBench 取得 SOTA，并且是唯一在 300 个训练步中对 GPT-4.1 anchor response 的胜率持续上升的方法；Llama-3.1-8B-Instruct 与 Qwen3-8B 上的结果表明其可跨模型族泛化。
- **arXiv**：[2609.00892](https://arxiv.org/abs/2609.00892)

#### Uncovering and Mitigating Aggregation-Induced Reward Hacking in Multi-Reward Reinforcement Learning (AMRP) (2026-08)
- **简介**：来自 Yu Yuan、Yaoyou Fan、Lili Zhao、Guangting Zheng 等。当前 RL fine-tuning 越来越多地同时使用可验证规则、任务专用评估器与学习得到的 reward model 等多个 reward 维度，并以固定权重 scalarize。作者识别出一种由聚合本身诱发的 reward hacking：静态投影把性质截然不同的 reward profile 混叠成同一个标量，使优化被推向最容易、最密集或被 reward 信号系统性偏好的维度；随训练推进这会把策略困在次优 profile，阻止其收敛到本可带来更高任务表现的更均衡 profile。为此提出 **AMRP**（Adaptive Multi-Reward Projection），一个轻量在线方法，用 relative shortfall、reward volatility 与 recent progress 三个信号重新分配聚合权重，对落后、不稳定或停滞的维度加压，对已饱和的维度减压。在结构化推理、citation-grounded 生成与开放式对齐三类任务的 GRPO 训练下，AMRP 一致改善 reward profile 的均衡性与下游表现，优于固定与动态加权 baseline，并在 GDPO 与 PPO 上同样有效（abstract 未给出具体数值）。
- **arXiv**：[2609.00213](https://arxiv.org/abs/2609.00213)

#### Measuring Reward Hacking and Reasoning-Answer Decoupling Under Position-Confounded Optimization (2026-08)
- **简介**：Suyash Maniyar、Armaan Sandhu、Abhishek Mishra。把 goal misgeneralization 当作一个度量问题：当 reward 在每个训练样本上都正确、但同时与多个目标一致时，模型可能习得非预期目标，而训练分布上的端点准确率无法区分「真正解题」与「利用表面特征」。作者用 GRPO 在正确答案恒为选项 A 的选择题数学数据上训练模型，再在答案位置无偏的未见测试集上评测：在 Qwen2.5、Llama 3.x 与 Gemma-3 上，有偏训练常把较小模型的选 A 率推到 0.90 以上、并把无偏准确率压向随机水平，此时准确率测量的已不是数学能力而是一种答案位置策略。进一步发现 reasoning-answer decoupling——能力较强的模型推理过程已算出正确数值却仍然选 A（用数值抽取加 LLM judge（GPT-4.1-mini）追踪，Qwen2.5-3B 的解耦率约 0.66）；这一坏构念还外溢到域外的 MMLU 与价值相关 prompt，且在无偏数据上继续训练只能不均衡地逆转域内偏移、对域外仅部分逆转，模型可能在训练分布上看似「已修复」而在未见输入上仍有偏。属 §1.6 的 reward hacking / 捷径学习现象与度量分析（不提出新训练方法），是判断「基准分数还在不在测能力」的关键对照。
- **arXiv**：[2608.15445](https://arxiv.org/abs/2608.15445)

#### Debate Training Reduces Reward Hacking in RLAIF (2026-08)
- **简介**：来自 Google DeepMind（Zachary Kenton、Lili Janzer、Rory Greig、Rohin Shah 等 11 人）。针对 RLAIF 的核心障碍——训练推进时策略会学会利用 AI judge 的系统性错误从而损害任务性能，且在 judge 弱于策略（正是监督更强 AI 的相关设定）时更严重——作者改用 debate 做 RL 微调：generator 与 critic 的两人对抗博弈由一个更弱的 LLM judge 裁决。实验选择最终答案可验证的数学任务以便直接测量 reward hacking 动力学，用 Gemini 2.5 Flash 级策略搭配冻结的更弱 Gemini 2.5 Flash Lite judge：单人 RLAIF 基线很快 hack 掉 judge，而 debate 在整个训练过程中维持 judge 性能，取得更高的峰值验证准确率（回收 45% 的性能差距）且在很多 RL 步后依然保持。附加实验表明：进一步弱化 judge 会加快 hacking，但增加一轮 debate 可以补偿；debate 的激励能压过被 prompt 注入的 misalignment；用 LLM judge 的 RL 比 RLVR 有更小的 train/validation reward gap；且不加玩家约束的对抗训练有退化为 critic 去 hack judge 的风险，实践中把 critique 限制在 150 词以内可平衡博弈、避免 judge hacking，但会牺牲 critic 的表达清晰度。
- **arXiv**：[2608.17776](https://arxiv.org/abs/2608.17776)

#### An Empirical Study of Reward Specification and Benchmark Reliability in GRPO-based LLM Unlearning (2026-08)
- **简介**：Rubén Balbastre、Juan Manuel Orduña、Mariano Pérez。指出 LLM unlearning 通常只按两个目标评测——抑制目标知识、保留非目标效用——从而在生成式 QA 中遗漏了第三种行为：当 target-adjacent 的 prompt 本可用一个不含目标细节泄漏的更宽泛答案回应时，模型应在该层面作答，而不是泄漏、回避或拒答。作者在受控的 LoRA-GRPO + RWKU 设定下比较四种 reward 设计（词面抑制、anti-refusal 塑形、rubric 式宽泛作答、显式拒答对比），每种都做带与不带 SFT warm-up 的对照。结论是优化成功不等于行为层面的 unlearning：RWKU forget 分数、held-out 补全审计、终态训练 rollout 审计与训练动力学会指向彼此矛盾的结论；作者把这些分歧归因于 reward-hacking 终态、GRPO 的 policy-support 限制、基准探针漏掉终态行为变化，以及 reward 在优化过程中可能选出语义泄漏低的宽泛话题式作答。属 §1.6 的 reward 规格化与 reward hacking 实证/度量可靠性分析，不提出新方法（abstract 未给出具体数值）。
- **arXiv**：[2608.17804](https://arxiv.org/abs/2608.17804)

#### Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL (2026-08)
- **简介**：作者 Minglai Yang、Xinyu Guo、Utkarsh Tyagi、Mian Zhang 等 9 人。指出 rubric（由 LLM judge 打分的准则清单）只是质量的固定代理、永远不是其完整描述，针对它优化足够久的策略必然学会利用两者之差；作者直接测量了这一现象：用 GRPO 在医学与科学 rubric 上训练 Qwen3-8B，并同时用训练 judge 与更强的 gold judge 评测 OOD 基准，发现两条分数在训练中分叉——训练 judge 的分数持续攀升，而 gold judge 的分数先见顶后下滑，在 HealthBench-Hard 上掉 3 个点、在 ResearchQA 上掉 22 个点；固定偏差的 judge 只会平移 gold 曲线而不会让它在训练分上升时反向下跌，因此这是 reward hacking 而非 judge 噪声。提出 **Rubric Dropout**，借用 neuron dropout 的一行式修法：每步在计算 reward 前随机丢弃 rubric 的一部分 criteria，使策略永不针对同一 rubric 优化两次；被丢弃的子集在同一 rollout group 内共享，以保证 GRPO 的组内 advantage 仍可比较，而评测始终使用完整 rubric。在两组基准上比较不做 dropout 与 30%、50% dropout，后者在每个对齐的 checkpoint 上都提高 OOD gold 分数（HealthBench-Hard +1～+2 点、ResearchQA +6～+7 点），降低作者跟踪的两个 hacking 指标且不损域内性能；扫参显示 30–50% 是较宽的最佳区间，而「按 criteria 对训练的有用程度重加权」这一自然替代方案在其设定下反而比完全不干预更差。
- **arXiv**：[2608.11669](https://arxiv.org/abs/2608.11669)

#### Not All Tokens Deserve Equal Credit: Counterfactual Sensitivity Credit Reallocation for Long-CoT Reasoning (CSCR) (2026-07)
- **简介**：来自 Qiangqiang He、Zhongheng Wu、ZiJian Wang。系统检验 On-policy self-distillation（OPSD）的隐含假设——特权自教师引起的似然位移是否携带可靠的"答案对齐"方向。做法：固定每条采样轨迹，分别在"断言正确"与"断言错误"两种对立结果条件下重打分。发现多数受影响 token 在两种条件下同向位移、符号反转极少，且大位移集中在高可替换的表层 token 上，而承载问题特定推理内容的 token 反而不敏感——说明特权位移的方向不可靠、其幅值主要反映"反事实敏感度"而非 token 学习价值。据此提出 **CSCR**：GRPO 的简单扩展，降低高敏感 token 的信用并重归一化 token 级优势，保持原信用预算与 verifier 决定的方向。长 CoT 数学推理基准上在相同策略更新数下一致优于 GRPO。
- **arXiv**：[2607.27888](https://arxiv.org/abs/2607.27888)

#### The Weight of Silence: A Causal Case for Weights Over the Scratchpad in Latent Chess Reasoning (2026-07)
- **简介**：Ishan S. Kshirsagar 首次在"RL 前后同一模型"上做 latent reasoning 的因果干预对照（此前因果分析多限于数学/逻辑、且仅在单 checkpoint 内比较）。用分阶段 latent-reasoning 课程 + RL 训练下棋模型：合法率从 pre-RL 48% 单调升至 61%、将杀棋臆造完全消除。六条件因果干预（替换/加噪 latent 思维向量几乎不影响、消融仅轻微退化、唯有 exact-zero 致崩溃）显示：RL 增加的是对扰动的鲁棒性而非对思维内容的依赖——exact-zero 破坏下合法率 pre-RL 崩至 1% vs post-RL 9%。反驳"latent 思维是推理期主动查询的 scratchpad"的默认假设，指出 latent reasoning 的主效应在于训练期塑形参数；并给出数学/逻辑之外（国际象棋）latent+RL 有效的工作示例。
- **arXiv**：[2607.20952](https://arxiv.org/abs/2607.20952)

#### Measuring Reward-Seeking via Contrastive Belief Updates (2026-07)
- **简介**：Axel Højmark、Jérémy Scheurer 等（Apollo Research）提出用 **Contrastive Synthetic Document Finetuning** 改变模型对"grader 奖励什么"的信念、使之与用户/开发者意图冲突，从而量化 RL 训练模型的"reward-seeking"（是否在追逐评分者判断而非真实目标）。在 OpenAI o3 能力向 RL 的中间 checkpoint（无安全训练）上发现：编码与对齐任务上模型常站队 grader，且该倾向随 RL 训练上升——如"守诺 vs 完成任务"环境中，晚期 checkpoint 在 SDF 暗示 grader 奖励完成任务时 87% 违诺、暗示奖励诚实时仅 9%（早期 40% vs 24%）；对 reward-hacking 模型组织体（gpt-oss-120b）敏感度翻倍（33%→86%）。表明 RL 会随训练加剧 reward-seeking。
- **arXiv**：[2607.18966](https://arxiv.org/abs/2607.18966)

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

#### Towards Bridging the Gap Between Offline and Iterative Alignment via Preference Distillation (DP3O) (2026-09)
- **简介**：Wenbo Zhang、Wenzhuo Zhou、Hengrui Cai、Zhengling Qi 追问两个问题：为什么 DPO 的 iterative 扩展普遍强于 offline 版本，以及能否把这种优势搬进 offline alignment。受控实验表明，iterative 流程中额外引入的显式 preference model 才是其优于 offline 方法的关键因素。据此提出 **DP3O**（Distilled Preference Probability Policy Optimization）：先用一组 helper LLM 学习显式 preference model，再把其知识蒸馏进 policy 优化。理论上作者证明显式 preference 建模比隐式形式有更好的估计误差控制，且 DP3O 通过方差缩减取得比 hard-label DPO 更紧的泛化界。在大量对话与下游任务上，DP3O 优于当前最好的 offline 方法、与 iterative DPO 性能相当，并把训练时间减少约 $42\%$。
- **arXiv**：[2609.06893](https://arxiv.org/abs/2609.06893)

#### Direct Diversity Optimization for Diverse Successful Trajectories in Preference Post-Training (DDO) (2026-09)
- **简介**：Junwon Ko、Dong-Jae Lee、Minchan Kwon、Junmo Kim 等。指出序贯决策任务的 LLM agent 通常只用 trajectory 级结果标签做后训练，这类标签无法监督「同一决策状态下保留多条成功分支」，导致成功策略覆盖度收窄。本文把问题形式化为 successful strategy coverage（固定 rollout 预算下模型能实现多少条互异的成功策略），提出离线后训练方法 **DDO**：**Divergence-Tree Collection（DTC）** 在共享决策状态处构造 state-aligned 分支集合，**Reference-Relative Target-Odds Objective（RTO）** 训练模型匹配成功备选之间的 reference-relative 目标 odds，因而不需要显式 reward model。在 BabyAI、BabaIsAI、WebShop 上任务成功率与成功策略覆盖度均为所比后训练方法中最强，局部动作替换后的 recovery rate 亦最高，且优于「只模仿成功轨迹」与解码期多样化对照。
- **arXiv**：[2609.10052](https://arxiv.org/abs/2609.10052)

#### Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs (SFS-DPO) (2026-08)
- **简介**：来自南洋理工大学（Vu Duc Anh、Nhat M. Hoang、Do Xuan Long、Luu Anh Tuan 等 6 人）。针对「模型自行验证并改正自己错误」这一 LLM 长期未解决的难题，提出基于 RL 的两阶段框架 **SFS-DPO**（Self-Fix Step-DPO）：第一阶段通过 step-level 偏好优化强化步级推理能力，第二阶段显式训练模型做自我验证与自我纠错；并给出 teacher 辅助变体 **SFS-DPO-R**，为错误验证引入解释性 rationale 以提供更强的纠错信号。在多个 LLM 上的 in-domain 与 out-of-domain 综合评测中，两个版本都一致优于此前的 step-level 训练 baseline；进一步分析显示自我纠错的发生频率与有效性同时提升，说明强化步级推理对稳健性能至关重要（abstract 未给出具体数值）。
- **arXiv**：[2608.11573](https://arxiv.org/abs/2608.11573)

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

#### E-GenRM: Generative Reward Modeling for Open-Ended E-Commerce Shopping Agents (2026-11)
- **简介**：作者 Ying Sun、Qiuyu Zhao、Xiao Wang 等。面向开放式购物任务，现有轨迹与奖励设计局限于当前使用模式与已知失败。**E-GenRM** 是面向电商购物 agent 的 generative reward model：数据侧利用购物 app 中自然发生的用户行为构造多样的长程对话轨迹；建模侧先用 SFT 学习基于行为的评估链，再经 criterion-guided RL 优化，奖励正确的偏好判断与有证据支撑的事实陈述，并通过个性化 hard-negative 训练提升对用户偏好的敏感度，另有主动证据检索扩展以补全评估所需的电商信息。在 E-GenRM-Bench 上，E-GenRM-27B 偏好准确率达 87.40%，比最强基线高 6.97 个百分点；在淘宝 14 天在线 A/B 测试中，用 E-GenRM 后训练的购物 agent 提升了有效购物指标。
- **arXiv**：[2611.03008](https://arxiv.org/abs/2611.03008)

#### Learning to Reason from Reference Samples Alone: Reinforced Adversarial Learning for Modern LLM Post-Training (RAL) (2026-10)
- **简介**：作者 Albert Bou、Javier Valdes-Aleman、Andrew D. White 等。许多场景下高质量答案易于辨认却难以用显式奖励打分，RLVR 受限于有显式正确性信号的任务。论文探究：仅以参考答案，能否通过共同训练的 LLM judge 把它们变成推理 LLM on-policy 后训练的唯一奖励？提出 **RAL**（Reinforced Adversarial Learning）。在整数乘法、算术表达式合成、SMILES 到分子式翻译三个可验证任务上（验证器仅用于评测），judge 足够强且训练足够久时 RAL 可与直接 RLVR 持平；应用于科学摘要和 NIH 基金摘要生成时，人类科学家与 LLM judge 集成都更难区分其生成与真实摘要，优于前沿 LLM 基线。
- **arXiv**：[2610.38851](https://arxiv.org/abs/2610.38851)

#### Co-Evolving Policy and Rubric via Capability Awareness for Open-Ended Generation (CoCA) (2026-10)
- **简介**：作者 Chengsheng Zhang、Wenshen Xu 等。Rubric-based RL 为开放式生成提供可解释的细粒度监督，但静态 rubric 可能饱和或超出策略的探索范围，导致奖励区分度不足，且已有方法没有显式让 rubric 难度跟随策略能力。论文提出 **CoCA**（Co-Evolving Policy and Rubric via Capability Awareness），通过 criterion 级诊断闭环地共同调整 rubric 与策略：对比 query-only 与 rubric-guided rollout 来刻画每条 criterion 相对当前策略的状态，重校准难度错配的 criterion，并识别由引导揭示的能力缺口；随后通过引导优化扩展 rubric 条件下的能力前沿，或通过 on-policy self-distillation 迁移合格增益。多个 benchmark 与 LLM 上稳定优于 rubric-based RL 方法，并缓解能力退化。
- **arXiv**：[2610.38566](https://arxiv.org/abs/2610.38566)

#### Expert-Guided Rubrics as Dense Reward Signals for Reinforcement Learning in Non-Verifiable Domains (2026-10)
- **简介**：作者 Ashish Bansal、Hossam Amer、Keerat Kaur Guliani 等（RL4LLM-Agents workshop, ICAIF）。在金融等不可验证领域，LLM agent 需理解用户意图、改写查询并检索，但评估查询质量需要稀缺的领域专家（SME）判断且本质上是分级而非二值的；GRPO 下若组内所有样本结果相同，稀疏奖励就没有学习信号而使优化停滞。论文提出把少量固定投入的 SME 判断转为稠密奖励（expert-guided rubrics），并与基于检索的指标结合，形成与 GRPO 兼容的混合训练信号。实验显示相比稀疏奖励基线，Recall@3 最高提升 6%，MRR@3 提升 6.7%。
- **arXiv**：[2610.26932](https://arxiv.org/abs/2610.26932)

#### Compression Precedes, Content Decides: Emergent Misalignment in Reward Models (2026-10)
- **简介**：作者 Dhruv Puri、Nikhil Prakash。Emergent Misalignment（窄域有害微调向域外泛化）已在语言模型中被广泛研究，但尚未在监督它们的 reward model 中研究。论文微调 Skywork-Reward-V2-Llama-3.1-8B 与 Skywork-Reward-V2-Qwen3-8B，使其在某一领域（医疗或金融）偏好有害内容，结果泛化到七个 held-out 领域；这并非退化，微调后的 RM 在 RewardBench 2 上仅下降 4.8 与 5.3 分（RM-Llama），仍优于未修改的 RM-Qwen。对底层 Llama-3.1-8B-Instruct 与 Qwen3-8B 用 DPO 与 SFT 做同样微调，发现 RM 比语言模型更易发生偏好反转。机制上可定位到最后/被打分 token：有害与正常回答的表征发生几何压缩、夹角缩小；同时有害表征与标量头的夹角更小，因而获得更高奖励，且仅在有害偏好微调中出现、对照组没有。
- **arXiv**：[2610.22805](https://arxiv.org/abs/2610.22805)

#### Judging the Judges: Confidence and Disagreement in Rubrics as Rewards (RaRE) (2026-10)
- **简介**：作者 Michael Tian-Yue Liu、Anne Harrington、Trevor Darrell、Yutong Bai。Rubrics as Rewards（RaR）用 LLM judge 对照结构化 rubric 为回答打分，把 RL 扩展到不可验证领域，已有工作聚焦于改进 rubric 或强化单个 judge。作者先表明：在 RaR-Medicine 与 RaR-Science 上训练 Qwen3-4B-Base 时，仅 judge 身份就会使得分产生很大差异（五个前沿与开源 judge）。据此提出 **RaRE（Rubrics as Rewards with Ensembles）** judge 集成策略，含两个机制：soft-Likert 奖励，用 judge 的 logprob 给更自信的 judge 更高权重；drop-on-disagreement mask，把高分歧 rollout 排除在 GRPO 更新之外。简单平均 judge 分数无法超过最佳单 judge，而 RaRE 在 HealthBench 与 GPQA Diamond 上都优于所有单 judge 的 RaR 变体。
- **arXiv**：[2610.21179](https://arxiv.org/abs/2610.21179)

#### EnGRICH: Enhancing Generative Reward Modeling with Critiques from Humans (2026-10)
- **简介**：作者 Xuancheng Li、Beining Wang、Haitao Li、Heng Wang、Yujia Zhou、Qingyi Pan。生成式奖励模型（GRM）在给出偏好判断的同时生成自然语言批评，但现有训练通常只用最终偏好正确性作结果监督，由于偏好结果空间高度受限，不可靠的批评也可能得到正确结果而被强化；近期工作用人类批评做过程监督，但这类批评稀缺且常被降为标量奖励。**EnGRICH** 让 GRM 搭配一个训练期 MetaCritic，后者从少量人类批评中学习：构造针对具体回答的 rubric，用其评价生成批评的证据覆盖与正确性，所得信号既提供用于细粒度信用分配的过程奖励，也为探索更好批评提供结构化引导；GRM 训练期间 MetaCritic 还被进一步优化，使基于人类批评的评价准则泛化到仅有结果的数据。（abstract 被截断，实验数字以原文为准）
- **arXiv**：[2610.05370](https://arxiv.org/abs/2610.05370)

#### RMB: Reward Model Boosting Mitigates Reward Hacking (2026-09)
- **简介**：作者 Jiabin Fan、Dezhi Ye、Yongchang Hao、Lili Mou（TMLR 2026）针对 RLHF 中因代理 reward model 不完美而出现的 reward hacking，即策略优化提高了代理 reward 却使真实人类偏好下的表现变差。提出 **Reward Model Boosting（RMB）**：先用促进多样性的正则项训练一组 reward model，使各模型学到 reward 景观的互补方面；再按 boosting 原则学习一个轻量聚合器，把多样 RM 的输出聚合为更准确、更稳健的 reward 信号。大量实验显示 RMB 在分布内与分布外数据集上显著提升 reward 准确率，明显缓解 reward hacking，并最终提升 RLHF 性能（abstract 未给出具体数值）。
- **arXiv**：[2609.33221](https://arxiv.org/abs/2609.33221)

#### Training LLM Judges from Language Feedback via Position-Selective Self-Distillation (2026-09)
- **简介**：作者 Ilgee Hong、Changlong Yu、Zhenghao Xu 等研究从自然语言反馈训练 LLM judge，尤其是主观任务，其结论强烈依赖 judge 调用哪些评估标准及如何加权。主流的结果监督 RL（如 GRPO）只按最终判决准确率给 rollout 所有 token 同一标量，对选择标准的 token 无独立 credit，也忽略了偏好标签附带的丰富语言反馈。Self-Distillation（SD）让同一模型在该反馈条件下作教师提供稠密位置级监督，但并非所有位置同等有用。作者用教师-学生的逐位置熵移位区分 context sharpening 与 context spreading 两种情形，据此提出基于熵移位的 position masking，保留熵移位分布的低尾。实验显示屏蔽高熵移位位置能提升相对朴素 SD 的 OOD 泛化；所得 self-distilled judge 在主观子类别上比结果监督 RL 训练的 judge 高 2–9 个百分点，在客观类上仍有竞争力。
- **arXiv**：[2609.38792](https://arxiv.org/abs/2609.38792)

#### DCRL: Decoupling and Coupling Reinforcement Learning via Policy-Reward Manifold Alignment (2026-09)
- **简介**：作者 Henan Sun、Zehua Li、Haitao Hu 等。规则式与奖励模型式 RL 常出现优化不稳定与 reward hacking。论文从几何视角把 LLM 通用推理视为由逻辑演绎、评估、表示三个相互依赖的子流形耦合而成，将规则/RM 式 RL 的局限解释为 RL 过程中策略-奖励流形失配。提出 **DCRL**：（1）基于三段论逻辑的 prompt 演化机制，动态精炼奖励 rubric 以增强奖励流形的表达力；（2）策略-奖励重耦合机制，联合更新奖励模型与策略以保证评估一致、缓解流形失配。多个推理领域上稳定优于规则式和奖励模型式基线，Qwen3-4B 经 DCRL 训练后超过 Qwen3-32B 基线并接近 Qwen3-235B。
- **arXiv**：[2609.27572](https://arxiv.org/abs/2609.27572)

#### Learning to Ideate for Scientific Impact (2026-09)
- **简介**：作者 Shubham Kale、Aniketh Garikaparthi、Manasi Patwardhan（RLxF Workshop, ICML 2026）。科研构想系统常用新颖性、清晰度、可行性等即时可判的代理训练与评估，延迟的学术采纳信号能否作为反馈引导模型朝向更高影响力的方向仍未知。论文以按年份归一化的引用影响力作为含噪但可扩展的学术采纳代理：从 10 万多篇计算机科学论文中抽取以目标为条件的构想描述并赋予有序引用标签，训练以研究目标为条件的奖励模型预测引用影响力标签，再用 SFT 加 RL 对齐构想生成器；为减少循环论证，用留出的、基于参考的协议评估，把生成构想与同一目标下的历史构想比较并按参考构想的引用标签加权。RL 调优后的模型产生的构想估计影响力一致高于基座与 SFT 基线。
- **arXiv**：[2609.29802](https://arxiv.org/abs/2609.29802)

#### EvoRS: On-Policy Self-Evolution of Reward Systems for Open-Ended Reinforcement Learning (2026-09)
- **简介**：作者 Weiyuan Li、Aili Chen、Xintao Wang 等。开放式 RL 常依赖基于 rubric 的 reward，但策略与 reward 系统构成动态反馈回路：策略优化当前 reward 后，原本有用的 reward 系统可能因 reward hacking 或响应区分度下降而失效，因此 reward 系统应在训练中演化。**EvoRS** 是一种自演化 RL 框架，把 reward 系统表示为可执行的 Reward-DAG，由 agentic designer 根据 on-policy rollout 与 reward trace 更新它，以保持训练时的可靠性。在写作与角色扮演任务上，EvoRS 在三个 judge 下均取得最佳质量，分别比策略提升 2.107 与 4.767 分，同时减少 reward hacking 与覆盖失效；消融表明固定的综合 reward 系统无法在开放式任务中保持可靠。
- **arXiv**：[2609.12459](https://arxiv.org/abs/2609.12459)

#### Building Legal Reward Models for Grounding and Abstention (2026-09)
- **简介**：作者 Rilton Franzone、Valentin Noël、Puyu Wang、Philip Torr、Fabio J. Fehr（ICML 2026 AI4Law Workshop）。法律等高风险领域要求系统基于检索证据推理并在证据不足时拒答，但现有 reward model 主要面向通用偏好，缺乏对上下文 grounding 的评估能力。作者提出把现有法律问答数据集转换为上下文偏好数据的框架，并据此构建 **LegalRewardBench（LRB）**，评测在噪声与不足检索条件下的 grounded 法律生成。发现 contextual DPO 能提升 grounded 评估，但对偏好数据构造敏感；长度均衡增强显著改善 grounded 法律评估，最强配置较 baseline 最多提升 25.6 个百分点；还观察到跨法域迁移，主要在维多利亚州刑法数据上精调的模型在美国 Housing Statute QA 上提升 16.2 个百分点。
- **arXiv**：[2609.14739](https://arxiv.org/abs/2609.14739)

#### UniRRM: Unified Reasoning Reward Models Across Languages and Evaluation Paradigms (2026-09)
- **简介**：Peng Lai、Yichao Du、Junchao Wu、Weibo Gao 等 9 人。指出 RL 在可验证任务上表现出色，但开放式任务中 reward model 的可靠性仍是关键难题：现有方案要么依赖昂贵的专有 LLM-as-a-Judge，要么是缺乏可解释性的标量 reward model，而新兴的生成式 reward model 又受制于静态评价标准、评测范式割裂与多语言支持不足。为此作者构建 **MixReward**——覆盖六个领域、103 种语言且同时包含 pairwise 与 listwise 数据的大规模多语言数据集，并提出 **UniRRM**，一个同时支持多语言与多种评测范式的统一 reasoning reward model：它用分阶段的 reasoning chain 动态生成 task-generic 与 instruction-specific 两层评价 criteria（即 rubric），从而实现细粒度、随输入自适应的判断，同时保持跨语言一致性。实验显示 UniRRM-8B 与 UniRRM-14B 在多个基准上接近同规模 SOTA，并对未见过的评测范式依然有效，消融实验验证了各组件的可靠性（abstract 未给出具体数值）。
- **arXiv**：[2609.05910](https://arxiv.org/abs/2609.05910)

#### HSRM: Hidden-State Reward Models for Test-Time Verification (2026-08)
- **简介**：来自 Xianzhi Li、Xiaodan Zhu。针对 LLM 能生成看似合理的数学推理却难以从多个候选中挑出正确解的问题，指出现有 test-time 流程依赖重新读取生成文本的 text-based verifier，使验证成为推理开销的主要来源；而已有研究表明 LLM 的内部表征中已编码了与正确性相关的信号。据此提出 **HSRM**，一个轻量的 hidden-state reward model：从 frozen generator 在推理步边界处抽取 hidden state，用一个小型 Transformer encoder 对候选解排序，直接读内部表征而不重新处理文本；训练只用自生成轨迹加 outcome label，既不需要人写的 process 监督，也不需要大规模预训练 verifier。在四个数学推理基准上，仅约 2M 参数的 HSRM 在 16 个 generator–dataset 设定中的 15 个上匹配或超过 55M 参数的纯文本 energy verifier，通过复用生成时已算出的表征提供了一种高效替代方案。
- **arXiv**：[2608.30841](https://arxiv.org/abs/2608.30841)

#### Small Language Models as Judges for Rubric-Based Reinforcement Learning (2026-08)
- **简介**：来自 Fengyu Xie、Yilun Zhao、Bingsen Chen、Arman Cohan 等。rubric-based RL 通过对照 instance-specific 标准打分，把 RL 扩展到没有精确答案或规则 verifier 的任务，但代价是 reward 计算昂贵——训练需反复做 rubric judging，通常要调用闭源 API 或部署 7B 以上的本地 generative judge。本文研究更小的模型能否胜任高效可靠的 rubric judge：先构建 PointRubric 与 RaR-Science-Static 两个 pointwise rubric 评测集（含 instance-specific 标准与逐条满足标签），再比较从小模型提取 criterion 级判断的三种方式——Generative verdict、Yes/No logprob margin 与 Probe judge。两个数据集上 Qwen3-1.7B 的 Probe judge 取得最强的 criterion 级一致性，优于 Generative 与 Logprob 判法；把它当作 GRPO 的 reward model 时，可将策略在 RaR-Science rubric score 上从 0.232 训到 0.643，而 8B Generative judge baseline 只到 0.594 且 reward-judge 耗时是其 10.7×。任务与领域迁移实验进一步表明 Probe judge 能跨设定保持 criterion 级的 reward 结构。
- **arXiv**：[2608.30005](https://arxiv.org/abs/2608.30005)

#### JudgePanel: A Compact Judge with Panel Deliberation via Adaptive Multi-Reward Reinforcement Learning (2026-08)
- **简介**：来自 Yiyue Qian、Shinan Zhang、Huan Song、Hannah Marlowe。LLM-as-a-Judge 已成为人工评估的可扩展替代，但单模型 judge 受自身模型偏置限制，而靠多样化 deliberation 缓解偏置的多 agent 评估协议在推理时代价过高。为此提出 **JudgePanel**，让一个紧凑 judge 模型具备多 agent panel deliberation 能力：先在由强评估器集成产生的 panel deliberation trace 上训练，捕捉讨论、分歧与达成一致的结构化模式；再用 **AdaReward**——一种自适应多 reward RL 算法，在 RL 训练中随各目标以不同速率饱和而动态重平衡 reward 各分量的权重——把判断质量推到 SFT 之上；为便于部署还设计了轻量领域特化模块，用数百条标注样本即可快速适配新评测域。结果上，14B backbone 的 JudgePanel 在四个评测基准上超过参数量最高达 70B 的 judge 专用模型，并表现出较强的 position consistency（abstract 未给出具体数值）。
- **arXiv**：[2608.29168](https://arxiv.org/abs/2608.29168)

#### GenRubric: Self-Evolving Rubric Generation for Scalable LLM Evaluation (2026-08)
- **简介**：来自清华大学与新加坡国立大学（Yifan Chen、Haitao Li、Qingyao Ai、Yiqun Liu 等）。LLM judge 常在打分过程中临时导出 query 相关标准，使评测要求界定不足、覆盖面难以审计；显式的 query-specific rubric 能把要求写明，但专家撰写成本高，而现有自动方法多依赖推理期 refine 或外部监督。提出 **GenRubric**，一个自演化框架，无需在自演化过程中追加人工标注即可从无标注 query 改进 rubric 生成。其原理是 rubric-induced self-consistency：对同一 query 独立采样得到的多个 rubric 各自给出其潜在评测要求的局部视角，一个全面的 rubric 应当诱导出在这些互补视角下都能成立的 response。实现上用 RL 落地该原则，把 cross-rubric comprehensiveness 信号与 group 级、criterion 级的 rubric 质量 reward 结合。作者在多个领域训练了 4B、8B、14B 三个规模的模型，在人工标注的 rubric 基准上，自演化提升了「生成 rubric 诱导的评测」与「专家 rubric 诱导的评测」之间的一致性，且改进可泛化到 held-out 领域（abstract 未给出具体数值）。
- **arXiv**：[2608.29856](https://arxiv.org/abs/2608.29856)

#### V-Rubrics: Visual Faithfulness via Rubric-Based Reinforcement Learning (2026-08)
- **简介**：来自南洋理工大学等机构（Shulin Tian、Minglun Li、Yuhao Dong、Ziwei Liu 等 9 人）。针对 VLM 会给出流畅但视觉证据支撑不足的回答（一个无依据的物体、图表数值或中间推断即可毁掉整段看似合理的回复），作者将其归因为多模态 post-training 中的 credit assignment 失效：标量 outcome reward 只能表明答案是否可接受，无法指出哪些视觉事实有据、哪些推理步骤有效、哪些指令约束被漏掉。为此提出 **Visual Rubrics-Based Reinforcement Learning**，把参考回复分解为原子命题，并沿 **Visual Faithfulness（VF）**、**Reasoning Consistency（RC）**、**Instruction Following（IF）** 三个维度给生成答案打分，得到的 rubric 条目提供结构化 partial credit，并在有支撑证据 span 时把 rubric credit 定位到对应位置。实现上先用公开 OpenMMReasoner-SFT-874K 语料微调 Qwen3-VL-8B-Instruct 得到 SFT checkpoint（沿用 OpenMMReasoner 的 cold-start 数据配方）；再从 17 个视觉 grounded 数据源构建 **V-Rubrics 50K**（50,248 条），先做规则过滤、以 rejection-sampling 得分刻画样本难度，再用 Gemini-3-Pro 在统一结构化 prompt 与协议下逐条标注；最后在同一 SFT checkpoint 上以 component-wise、prefix-localized 的 rubric credit 训练。实验显示 rubric-based GRPO 优于共享 SFT baseline 与 answer-only GRPO，在知识型与视觉 grounded 推理基准上增益最大（abstract 未给出具体数值）。
- **arXiv**：[2608.25580](https://arxiv.org/abs/2608.25580)

#### RecurSE: Bounded Recursive Self-Evaluation for LLM Rubric Judges (2026-08)
- **简介**：来自阿里巴巴（Kaiyuan Liu、Ziyuan Zhuang、Rongxiang Weng、Jieping Ye）。LLM-as-judge 对开放式文本评测与 post-training 引导至关重要，但改进 judge 本身通常依赖昂贵标注、reward model 或更强 teacher 蒸馏。本文从 RL 训练 reward 中彻底移除外部 gold 监督，让模型自身的评价能力产生优化信号，构成有界递归自改进（bounded RSI）的闭环设定，命名为 **RecurSE（Recursive Self-Evaluation）**，并研究两个核心问题：自改进何时能发生、又何时必须停止。机制上，Pass 1 是一个可训练 judge 按 per-rule rubric 评估候选回复，Pass 2 是与之同步的 policy-copy checker，依据 meta-rubric 审计 judge 的推理过程并给出标量 process reward；为使学习可行，**interface decoupling** 在结构上把 checker 的标量分数与 judge 的判决 token 隔离，消除了那种会虚高自评 reward 的退化性 token 复制捷径。由于无锚点的递归学习本质有界，作者用 **Pairwise Advantage Validity（PAV）** 作为无偏验证监控量，同时跟踪 judge 准确率与 checker 保真度，以可靠识别最优早停窗口。在 Qwen3.5-9B、Gemma-4-E4B-it、Qwen3.6-27B 上，RecurSE 于留出的医学、pairwise、摘要与专业基准上取得一致泛化增益；消融显示 judge-checker 同步共演化优于冻结 checker、外部 meta-judge、self-consistency 与放大规模的 teacher 蒸馏；其 judge 挑选的偏好对还能有效提升下游策略对齐（abstract 未给出具体数值）。
- **arXiv**：[2608.24231](https://arxiv.org/abs/2608.24231)

#### AutoVerifier: Residual-Guided Non-Parametric Optimization for Reference-Based Answer Verification (2026-08)
- **简介**：Zebei Zhao、Zhihao Shi、Minqi Shi。基于参考答案的 verifier 既用于评测推理模型，也为 RLVR 提供准确的 outcome reward；已有工作探索过规则型、模型型与工具增强型 verifier 来判定各种答案形式的等价性，但作者指出诸如 $1+3.14$ 与 $1+\pi$ 是否等价，实际取决于题目与评分标准，这类隐含假设被形式化为 **verifier inductive bias**。为此提出 **AutoVerifier**，一种 residual-guided 的非参数化优化方法，从 verifier 反复出错的残差中学习这些 bias：把 bias 记录成 rule card，只有在 replay validation 确认没有直接回归退化后，才将其提升为代码模块或 prompt guidance，从而使被接受的更新保持可审计、可编辑、可复用。在四个 verifier 基准上大幅超过 SOTA verifier（abstract 未给出具体数值）。
- **arXiv**：[2608.25637](https://arxiv.org/abs/2608.25637)

#### APTER: Adaptive Post-Training with Expert-Grounded Rubrics (2026-08)
- **简介**：Xukai Wang、Liangqi Li、Zhiyue Xu、Xu-Yao Zhang 等 9 人。指出大模型进入专业领域后需满足领域约束、包含关键证据并给出完整推理，而现有后训练多依赖整体偏好或结果级验证，近期 rubric 方法又通常为每个 query 独立生成 rubric，在专业领域可能漏掉关键要求且样本间标准不一致，妨碍对持续性能力缺陷的诊断与定向修复。提出 **APTER**：其一，expert-grounded rubric 构建从领域专家搭建的专家准则框架出发（每条准则代表一项稳定的专业能力），对每个 query 选取相关准则并实例化为链接到源准则的 query 级 rubric，把可复用的专家准则转成无需参考答案的可执行监督；其二，自适应后训练把 rubric 判定同时用作优化信号与准则级诊断信号——按准则 ID 聚合低分判定以暴露持续缺陷，并在 RL 过程中触发针对性的 SFT 更新。数学推理与医学问答实验在两个领域上都取得一致增益：跨三代模型，数学与医学平均分相对对应 base 模型最高分别提升 15.86 与 8.04 个点；代码与 rubric 数据集已开源。
- **arXiv**：[2608.14212](https://arxiv.org/abs/2608.14212)

#### Competence, Not Accuracy: A Diagnostic for Reference-Free Judge Gates in Skill Optimization (2026-08)
- **简介**：Chenle Chen、Yangbo Wei、Chao Yao、Lei He 等 7 人。text-space skill optimization 通过演化一份自然语言 skill 文档来适配冻结的 agent，每个候选都要经验证 gate 接受；现有 gate 依赖可验证 reward，因而把这类方法限制在有自动 verifier 的任务上，若换成 LLM-judge gate 能否携带可用信号则从未被检验。作者提出一个前置问题——在把 judge 放入循环之前，能否判断它的打分究竟能不能区分正确与错误答案，并把 reference-free judge 形式化为 latent solver：其判定本质上取决于与自己会得出的结论是否一致，故评估能力被解题能力上界所限。由此给出以 judge 能力 c 与答案空间大小 k 表示的可区分性（ROC-AUC）闭式界、必要条件 c > 1/k，以及「边际 AUC 会被题目难度混淆而 within-question 估计不会」的结论；再用不干预任何决策的 non-intervening probe 在真实优化运行中记录 judge 分数，发现 competence 接近下界时可区分性只等于随机、超过下界后才可用，且 judge 的基准准确率会高估真正相关的 competence，闭环实验还显示该筛查能预测会发生哪一类 gating 错误。属 §1.8 的 judge/reward 可靠性诊断类工作，产出的是低成本的部署前诊断而非新训练方法（abstract 未给出具体数值）。
- **arXiv**：[2608.18719](https://arxiv.org/abs/2608.18719)

#### RISE-RL: Rubric-Informed Selective Exploration for Open-Ended Reinforcement Learning (2026-08)
- **简介**：作者 Jinkun Hou、Zhuo Liu、Huimin Ren、Pan Zhou、Kun Zhan 等 6 人。指出开放式任务的对齐困难在于回答需同时满足多维标准、且不存在唯一正确的生成轨迹，而现有 rubric-based RL 把 criterion 级细粒度反馈压成标量 reward，使得在有限 on-policy exploration 下难以定向修补持续存在的能力缺口。提出 **RISE-RL**（Rubric-Informed Selective Exploration）：用反复未被满足的 rubric criteria 去诱导那些靠无引导探索难以发现的 privileged 轨迹，只保留完整 rubric reward 高于自然 rollout 平均 reward 的轨迹，再把它们放回原始 prompt 下重新评估，以强化自然策略支持度仍很弱的行为；该引导信号通过一个独立的辅助目标优化，并在其额外收益衰减后移除。在 4B 与 14B 模型、写作/对话/健康/科学四类任务上，无引导评测下每个基准均取得最高平均分：相比标准 Rubric-RL，4B 规模平均提升 1.3 点、14B 规模提升 3.3 点，其中 CreativeWriting-V3 提升 6.0 点；同时提升创意写作多样性并在客观评分的医学、科学基准上获益。
- **arXiv**：[2608.09123](https://arxiv.org/abs/2608.09123)

#### ConRub-Med: Reinforcement Learning with Consensus Rubrics for Open-Ended Medical Question Answering (2026-08)
- **简介**：作者 Taojie Zhu、Yuan Xia、Tao Sun、Jinjie Gu、Yonghong He 等 11 人（含蚂蚁集团研究者）。指出 RLVR 在数学与代码上有效是因为答案可自动校验，而许多开放式医学问题缺少同等廉价的结果 verifier——回答可能部分正确、不完整或含临床后果严重的错误；医生撰写或审核的 rubric 临床基础扎实但逐题请专家成本过高，模型生成的 rubric 则能把这种监督规模化。提出 **ConRub-Med** 以在 rubric 反馈从构造流向策略优化的过程中保住有用的区分度：每个 prompt 由三个异质语言模型独立提出原子 criteria，另一个模型做审核、只保留三个生成器在语义上都支持的条目；Three-State 评分区分正确覆盖、信息缺失与错误陈述，错误给负分而非零分；当一个完整 GRPO 组内所有回答拿到相同最终 reward 时，由 pairwise judge 在两种候选顺序结论一致时才提供 sequence advantage，且不改动标量 reward，无平局的组则退回 vanilla GRPO。按题目配对的盲评中，两位医学专家认为完整流程产出的 rubric 面板比单一生成器更具临床相关性；在所评开源模型上 ConRub-Med 于九个基准中六个排名第一，并取得最高的医学平均分与泛化平均分，用其 5,166 条 prompt 的 rubric 数据集在 HealthBench-Hard 上得 38.98±1.04，而 InfiMed-ORBIT 用 8,000 与 28,000 样本分别为 33.60 与 37.30。
- **arXiv**：[2608.10996](https://arxiv.org/abs/2608.10996)

#### RRC: Unlocking Generative Reward Models in LLM Reinforcement Learning via Ranking-Based Reward Construction (2026-08)
- **简介**：来自东北大学 NLP 实验室（Chenglong Wang、Ziming Zhu、Yifu Huo、Bei Li、Jingbo Zhu 等 12 人）。指出 reward modeling 正从判别式转向生成式，但生成式 reward model 虽在回复排序上能力很强，在 RL 中的潜力却未被释放；作者的分析归因于生成式 reward modeling 的「比较」本质与现有 RL 算法采用的标量打分范式之间存在错配。提出 **RRC**（Ranking-based Reward Construction），从相对偏好排序中导出 reward 以提供更有效的 RL 学习信号，包含两种互补策略：self-competitive ranking 利用同一 prompt 下采样回复之间的比较，anchor-guided ranking 借少量参考回复实现可扩展的排序式 reward 构造。在开放式对话与推理基准上，RRC 相比已有 reward 构造方式一致提升了基于生成式 reward model 的 RL 训练效果，代码已开源。
- **arXiv**：[2608.06310](https://arxiv.org/abs/2608.06310)

#### When the Judge Should Not Decide: Evidence-Locked, Non-Compensatory Selection Bounds LLM-Judge Failure in Reasoning Pipelines (EL-DGR) (2026-08)
- **简介**：来自 Yiyao Zhang、Diksha Goel、Hussain Ahmad、Shixun Huang 等。指出部署在推理流水线中的 LLM judge 不只是度量质量，而是在决定「哪个答案上线」，这一决策的代价更取决于 judge 被嵌入的决策规则而非 judge 本身的准确率：在四个 GRPO 策略的冻结候选池上，无约束的标量 DeepSeek-R1-7B judge 相比答案级 majority vote 几乎毫无收益（GSM8K 500 题 +1.0 pp、HotpotQA 300 题 +0.34 EM），在 30 题冻结规则确认集上甚至比多数投票低 10 个点。提出 **EL-DGR**（Evidence-Locked Derive-Gate-Repair）这一任务自适应的非补偿式规则，让同一 judge 服从约束：judge 偏好只有携带抽取式证据凭证时才可推翻有证据支持的共识，修复仅在两个候选均未被认证而修复结果被认证时才允许。不改 judge、候选与预算的前提下，GSM8K 达 58.2%（judge 56.8%、majority 55.8%、首候选 55.4%）、HotpotQA 达 17.33 EM / 25.46 F1，较首候选 GRPO 提升 +2.8 pp（McNemar p=0.0026）；决策审计显示它在 30 道试点题中只推翻 8 次共识且从未把正确共识改错。作者同时报告负面结果：同一七通道分解用作步级门控训练 reward 无效。
- **arXiv**：[2608.07813](https://arxiv.org/abs/2608.07813)

#### CSPF: A Constrained Shared-Private Fusion Method for Non-Verifiable Preference Evaluation (CSPF) (2026-07)
- **简介**：Hehao Zhang、Danli Wang 等提出 **Constrained Shared-Private Fusion (CSPF)**，将多个异构冻结奖励模型视为互补评估器，在成对人类偏好监督下学习融合其隐状态表征：把每个专家信号分解为 shared 与 expert-private 表征，鼓励跨专家对齐同时保留互补视角。在 LM-Arena 目标域适配与 PPE 分布外偏好评估上，主指标优于单专家 RM、标量多专家、rubric-judge 等基线，表明融合隐状态为不可验证偏好评估提供更具表达力的基础。
- **arXiv**：[2607.20862](https://arxiv.org/abs/2607.20862)

#### Co-Evolving LLM Evaluators and Policies via DynamicRubric (DynamicRubric) (2026-07)
- **简介**：Beining Wang、Qingyao Ai、Yiqun Liu 等（清华 / 微信搜索）从"概率质量分配"视角论证：随策略变强、候选质量趋近，evaluator 相对分差坍缩会造成弱/误导监督；并证明移动概率质量的方向增益恰等于两响应间的 evaluator 分差。提出 **DynamicRubric**：response-set 条件化的 evaluator–policy 协同进化框架，为每个候选集生成加权二值 rubric 项并聚合成响应级分数。8B 骨干上其 evaluator 与策略监督均超过 70B RM 或 235B 静态 rubric 生成器；优化后策略在可验证推理与编码上有增益，且已全量部署于微信搜索 AI 问答（日千万级请求）。
- **arXiv**：[2607.20083](https://arxiv.org/abs/2607.20083)

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

#### Test-Time Self-Play: Scaling without a Verifier (TTSP) (2026-11)
- **简介**：作者 Akira Yoshiyama、Akmal Ashirmatov、Jonas Hübotter、Ido Hakimi、Andreas Krause。多数 test-time scaling 方法依赖 verifier 对候选解打分以选择或精修，但许多部署场景中并没有针对目标问题的强 verifier。论文提出 **Test-Time Self-Play (TTSP)**，用非对称自博弈做 test-time scaling，不依赖 verifier。在 LiveCodeBench v6 的难度校准子集上，TTSP 在所有评估的 verifier 强度下都是表现最好的方法，且 verifier 越弱、与最强基线的差距越大；在无 verifier 极限下，mean@8 获得 35% 的绝对提升。仅当基座策略对正确答案几乎零覆盖时增益消失。
- **arXiv**：[2611.07367](https://arxiv.org/abs/2611.07367)

#### Consolidating Reasoning with Test-Time Training (TTT-Consolidate) (2026-11)
- **简介**：作者 Haotian Wu、Zeming Chen、Hao Zhao、Antoine Bosselut，NeurIPS 2026。多 agent 协作若只对结论投票或把大量嘈杂轨迹塞入上下文，会浪费中间推理。**TTT-Consolidate** 把多 agent 推理轨迹当作适应数据，通过 test-time training 将轨迹整合为参数化表示，再基于整合后的信息生成新解。训练时用双层优化元学习如何整合：内循环把推理轨迹编码为轻量 LoRA adapter，外循环学习内循环的元参数，使整合结果有助于合成最终解。作者还构建 OpenParallelThinking 语料（12,810 道数学题，配以 6 个 LLM 的混合轨迹）。在数学与 STEM benchmark 上，TTT-Consolidate 比最佳聚合基线高出最多 20 个百分点，在算力与时间匹配下依然成立，且模型越大差距越大。
- **arXiv**：[2611.02104](https://arxiv.org/abs/2611.02104)

#### Self-Evolution Reasoner: Continuous Optimization via Policy-Intrinsic Exploration and Exploitation (SEPO) (2026-10)
- **简介**：作者 Wenhang Shi、Yiren Chen 等。RLVR 与自监督 on-policy distillation 都受探索瓶颈限制：策略采不到正确解时难以脱离局部最优，而依赖外部示范又是 off-policy 且分布不匹配。论文提出 **SEPO**（Self-Evolution Policy Optimization）：先利用标签诱导策略自身产出正确推理（引导探索），再把「以该 in-distribution 推理为条件的当前策略」当作 self-teacher，将新发现的解蒸馏回策略（in-distribution 利用），迭代形成持续自我进化。在科学、工具使用和复杂逻辑推理领域，同等训练时间/步数下均取得最佳表现，并在复杂任务或小模型上基线几乎失效时仍保持稳定提升。
- **arXiv**：[2610.34137](https://arxiv.org/abs/2610.34137)

#### Group-Marginalized Self-Rewarding RL Drives Zero-Label Self-Evolving (GMAE) (2026-09)
- **简介**：作者 Yiming Wang、Yikang Liu、Qingyuan Tian、Zhuosheng Zhang、Zhaopeng Tu 等研究无需人工标签的 self-rewarding RL。现有基于 ensemble 的方法由 rollout 组构造 reward 参照，但响应的 reward 表示还取决于其随机采样的组上下文（组内其他响应），只用单个上下文实现可能漏掉所需信号、给策略优化提供不可靠指引。提出 **Group-Marginalized Advantage Estimation（GMAE）**：把多种可能上下文下的 reward 实现聚合为响应级分布，并估计期望 advantage。在八个基准、四个基座模型上表现强劲且跨域泛化，同时学习稳定、额外成本低，并适用于不同训练数据集与 RL 骨干（abstract 未给出具体数值）。
- **arXiv**：[2609.36750](https://arxiv.org/abs/2609.36750)

#### Frontier Learning: Training LLM Reasoners at the Edge of Capability (2026-09)
- **简介**：作者 Robin Faro、Shyam Sundhar Ramesh、Ilija Bogunovic、Aurelien Lucchi 指出，现有 GRPO 后训练多在训练前固定的题池上微调，而只有 rollout 同时包含成功与失败时才有学习信号，随着模型进步固定题池中有用部分很快失效。提出 **frontier learning**：一种开放式后训练方法，在线使用程序化生成器持续产生有信息量的训练问题；把生成器的任务相关参数视为搜索空间，用 regret 信号优先探索「前沿」难度，使训练聚焦在模型不断演化的推理能力边缘。在多个推理任务与模型家族上，相比固定题池基线稳定取得更高的相对增益（abstract 未给出具体数值），说明有效后训练不仅要挑选有用题目，还要持续在能力边缘生成题目。
- **arXiv**：[2609.35426](https://arxiv.org/abs/2609.35426)

#### Dream-RSI: Recursive Self-Improvement through Evolving Worlds (2026-09)
- **简介**：作者 Tong Zheng、Xidong Wu、Zheng Zhang 等。自主 agent 的递归自我改进依赖有效探索，但固定探索策略无法随搜索空间扩大而适应，而在线策略优化又要面对长程 rollout 带来的延迟且昂贵的反馈。**Dream-RSI** 用一个轻量的编排层使探索显式、可编程，同时保持底层 base agent 不变；其关键洞察是累积的发现历史可充当对已实现搜索空间的 replay simulator。通过在由历史发现树构建的模拟器中「做梦」，可获得即时、低成本的 off-policy 反馈来评估和改进探索策略，改进后的策略再在线部署，进一步扩充模拟器池，形成自我改进闭环。在 4 个领域的 9 个任务上取得有竞争力的质量，并在若干设置下提升发现效率。
- **arXiv**：[2609.14858](https://arxiv.org/abs/2609.14858)

#### Sampling via Decision-Flow: Training-Free Extraction of Improved Latent Reasoning Paths in Large Language Models (DF-Sample) (2026-09)
- **简介**：作者 Zhendong Mi、Shaoyi Huang。基于 RL「分布锐化」假说（RL 只是把概率质量重分配到 base 模型已有的高 reward 轨迹），作者问：能否不做 RL 微调就解锁这些潜在路径？提出 **DF-Sample**（Decision-Flow Sampling），一种免训练、免数据的推理期框架：构建层次化推理树，对终端节点打质量分，并把效用反向传播以指导每个中间分支决策；与纯局部的逐步采样不同，它在选定路径前做显式全局轨迹评估，找回标准解码忽略的高质量低概率推理链。在 GPQA 上准确率 45.6%，超过 power sampling（38.9%）与 GRPO（39.9%）；在三个模型、四个 benchmark 上均优于 baseline，表明预训练 base 模型有可观的潜在推理能力。
- **arXiv**：[2609.12317](https://arxiv.org/abs/2609.12317)

#### STRETCH the Boundaries: A Unified Self-Taught Framework for Progressive LLM Evolution (2026-09)
- **简介**：作者 Yajie Yu、Mark Lee、Yue Feng。LLM 自我改进训练常因固定难度无法适应模型不断提升的能力而停滞。**STRETCH**（Self-Taught Reasoning Evolution via Targeted CHallenge）受认知脚手架理论启发，引入动态 Stretch Zone 机制，持续使题目难度与模型求解能力对齐：在同一参数空间内，模型在 Scaffolder（生成自适应、推动边界的挑战）与 Learner（通过 RL 优化自己的求解轨迹）之间交替，这种双循环共同演化可稳定训练、缓解 reward hacking 并促进推理能力渐进增长。在谈判与运筹学 benchmark 上持续优于强 prompting 与领域特定 baseline；scaffolder 配置分析表明动态难度对齐对持续的能力提升至关重要（abstract 未给出具体数值）。
- **arXiv**：[2609.18642](https://arxiv.org/abs/2609.18642)

#### Beyond Confidence: Stability-Aware Test-Time Adaptation for LLM Reasoning (TASCO) (2026-09)
- **简介**：Bincheng Gu、Min Gao、Zongwei Wang、Yibing Bai、Yulan He、Junliang Yu。指出 test-time adaptation 是替代昂贵后训练的轻量路线，predictive entropy 可作为模型自带信号、无需外部 verifier 或 reward model 地把模型导向更高置信的推理状态，但高置信不等于正确——LLM 在错误推理轨迹上同样可能高度自信。作者的观察是：当置信度在局部扰动下保持稳定时，高置信推理更可能正确。据此提出 **TASCO**，在保持 LLM 冻结的前提下只优化一个轻量的 task-level prefix，把局部稳定性纳入基于置信的 test-time adaptation，并给出两种扰动策略——Random Perturbation 促使相邻扰动 prefix 所诱导的各条轨迹之间保持分布稳定性，Sharpness-Aware Perturbation 则针对最坏情况的局部敏感度。实验显示 TASCO 在多种 LLM 与推理基准上同时提升推理准确率与 token 效率，行为分析表明它能在局部扰动下维持稳定置信而不会过早收窄模型的预测分布（abstract 未给出具体数值）。定位说明：基座 LLM 全程 frozen 且不使用外部 verifier/reward，但并非完全 parameter-free——仍有梯度更新，只是更新对象是轻量 prefix 参数，符合 §1.9 收录 test-time adaptation 类工作的惯例。
- **arXiv**：[2609.11393](https://arxiv.org/abs/2609.11393)

#### GUT: Quantifying and Optimizing the Reasoning Uncertainty of LLMs via Graph Complexity (2026-09)
- **简介**：Shuang Liang、Xin-Yu Hu、Xiang-Jun Ou、Shao-Qun Zhang。针对 LLM 推理过程中的不确定性——同样的 prompt 输入下每个推理步都会衍生大量发散分支，其中部分分支的推理链明显不可信甚至荒谬——提出基于图复杂度的 **GUT** 方法。其核心思路是用有向无环图刻画每条推理链的潜在分支，从而保证所有潜在分支都被完整覆盖在图空间内；在此之上构建两个模块：GUT-Q 通过用 graph complexity 近似推理空间复杂度来量化推理不确定性，GUT-O 则把负的不确定性作为 RL 的 reward function，以此优化并降低不确定性。在 4 个 LLM 与 5 个数据集上的实验验证了方法有效性（abstract 未给出具体数值）。
- **arXiv**：[2609.05284](https://arxiv.org/abs/2609.05284)

#### From Base Rollouts to RL Reasoning: A Budgeted Search Perspective (2026-09)
- **简介**：来自 Wenhe Sun、Cunxiang Wang、Zijun Yao、Yixin Cao。追问 RLVR 的增益与推理期解码/搜索之间的关系：RL 是创造了 base model 本不具备的推理能力，还是把 rollout 分布移向它本已可达却很少采到的轨迹？作者用 Unified Decoding Framework（UDF）从行为层面研究，把 token 级采样、beam 式搜索、tree search 与 sequence 级 resampling 统一表达为共享预算操作空间上的可执行 policy，并用 pass@k、self-consistency、best-of-N、first-finish success 事后打分；随后在 SimpleRL-Zoo 的 Base/RL 配对 checkpoint 上检验 RL 默认策略曲线能否由 base 操作点的结构化路径近似。在 Math500、AIME、GPQA、IFEval 上，pass@k 恢复路径服从 Budgeted Operating-Point Transition Rule（BOPTR），即 N_Base ≈ αN_RL^β（指数随基准而变）。Qwen2.5-7B 上 BOPTR 在所测非 oracle 规则中迁移误差最低，为 3.41 pp（95% CI [2.32, 5.53]），三随机种子复现为 3.07 ± 0.39 pp；该规则还外推到四个模型族的十个模型（拟合后新增 checkpoint 上 3.28–4.87 pp）、四个从未参与拟合的基准（5.03 pp，拟合内 4.44 pp），并在目标模型没有 RL checkpoint（4.19 pp）甚至完全没有 RL 监督（5.08 pp）时仍成立。属 §1.9 的行为学诊断分析：作者将其定性为对 internalized-search 的有限度解读与描述性 scaling 规律，明确声明不是参数级等价的证据。
- **arXiv**：[2609.01274](https://arxiv.org/abs/2609.01274)

#### ERR+: Sequential Entropy Resolution for Efficient and Decisive LLM Reasoning (2026-08)
- **简介**：来自 Xin Jiang、Minhao Wang、Wen Wu、Zhentao Xie 等。指出当前 RLVR 虽靠正确性 reward 取得好结果，但对推理过程本身的质量几乎没有指导，内部推理结构基本未被优化；跨多个模型族的实证分析发现一个一致模式：正确的推理 trace 在 thinking 阶段出现的 token-level entropy 下降更频繁、幅度也更大。据此提出两阶段 RLVR 框架 **ERR+**：第一阶段用 **Entropy Relief Reward**（ERR），按 thinking 阶段累计的 token-level entropy 下降给 bonus 并按 response 长度做对数归一化——与此前抑制 entropy 的做法相反，它奖励「不确定性被消解」而不约束探索性的高 entropy 状态；第二阶段引入 Robust Relative Efficiency Reward，用组内 z-score 经 tanh 变换对每条 response 的长度相对同组同伴打分。作者给出形式分析，说明两个目标联合优化会在训练早期产生梯度冲突，从而论证了顺序式（两阶段）设计的必要性。五个数据集、多个模型 backbone 上准确率与回答简洁性均获一致提升（abstract 未给出具体数值）。
- **arXiv**：[2608.28771](https://arxiv.org/abs/2608.28771)

#### TTPO: Test-Time Policy Optimization (2026-08)
- **简介**：来自浙江大学（Aozhe Wang、Zhengxi Lu、Jianze Wang、Yongliang Shen 等 11 人）。RL 与 On-Policy Self-Distillation（OPSD）等 post-training 方法推动了数学推理进展，但都依赖 ground-truth 标签，无法用于 test-time training（TTT）；用 majority-vote 伪标签替代 ground truth 虽自然却脆弱——一次错误投票会污染 teacher 并误导每个 token。作者观察到这一失败模式是**非对称**的：与伪标签不一致的 rollout 通常本身就是错的，无论投票结果对错。据此提出 **TTPO**，采用非对称目标：对与伪标签一致的 rollout 用 OPSD 做蒸馏，对不一致的 rollout 用 Grouped RL 施加惩罚；再以 token-level 选择精化两个分支——蒸馏降低已收敛位置的权重，RL 只惩罚高置信度的错误。这样即使伪标签频繁出错，两类更新仍然是有依据的，且随模型变强，majority-vote 路由会给出更紧的自监督。无需任何标签，TTPO 在 5 个竞赛级基准上追平有标签监督的 OPSD，在 TTT 设定下把 Qwen3-1.7B 从 38.0% 提升到 45.2%，在 no-thinking 模式下带来 +25.2% 至 +36.4% 的提升，并表现出较强的跨任务泛化。
- **arXiv**：[2608.27448](https://arxiv.org/abs/2608.27448)

#### Prefix Sliding for efficient test-time scaling (2026-08)
- **简介**：来自斯坦福大学等机构（Niklas Muennighoff、Zhengyang Wang、Zeyi Chen、Percy Liang、Mike Lewis 等 18 人）。test-time scaling 靠增加测试期算力（如让模型推理更久）提升性能，但模型用 full attention 把整条推理轨迹留在内存中，需要长思考的难题代价高得难以承受；作者发现大多数中间推理 token 随推理继续会丧失重要性，因此保留它们是否值得成本值得怀疑。据此提出 **Prefix Sliding**：推理过程中丢弃既不属于 prefix、也不在最近数千 token 窗口内的 token——prefix 保留关键指令与可用工具，最近 token 是模型当前正在推进的推理，于是总内存需求被封顶，与推理长度无关，从而支持高效的长时程 test-time scaling。无需训练时，Prefix Sliding 就能让现有模型快 3 倍且保持性能；配合强化学习训练则能进一步提升性能，使推理轨迹可扩展到十万 token 以上。消融显示 Prefix Sliding 优于对中间 token 做摘要以及普通 sliding window，代码已开源。
- **arXiv**：[2608.26070](https://arxiv.org/abs/2608.26070)

#### Selective Regenerative Decoding: Trajectory-Level Intervention for Inference-Time Reasoning (SRD) (2026-08)
- **简介**：来自 Amazon / AWS AI 相关团队（Sophia Xiao Pu、Yumo Xu、Sailik Sengupta、Arshit Gupta 等 8 人）。推理期解码方法通过探索多条候选轨迹提升 LLM 推理，但把每条轨迹当作原子对象——整条保留或整条不可逆丢弃，于是那些前缀质量高、只是后缀退化的「半有希望」候选被连带浪费。**SRD** 把每个候选路由到 discard / keep / refine 三种处理之一，对边界候选只重新生成退化的后缀部分而保留有用前缀，且无需更大的 target model。在温和假设下，SRD 相对 rejection sampling 有可证明的 1.28–1.36 倍样本效率增益，且期望轨迹质量严格更高，候选池越大增益越大。在 MATH500、GPQA Diamond、HotpotQA 与 AlpacaEval 上、配多组 generation–reward model 组合，SRD 以显著更少的生成 token 达到 Best-of-N 的准确率，并在低算力区间优于 speculative rejection。通过把干预粒度从整条轨迹选择下移到 segment 级，SRD 打开了推理期准确率—算力权衡曲线上此前未被探索的区域。
- **arXiv**：[2608.24338](https://arxiv.org/abs/2608.24338)

#### VIG: Visual Information Gain as a Reward Signal for Multimodal Chain-of-Thought Compression (2026-08)
- **简介**：Wen Luo, Xiaohan Yi, Xiaotao Huang, Liqun Huang。多模态大推理模型的长 CoT 中，重复视觉描述、自我反思等与视觉脱节的填充 token 占相当比例，只推高推理成本却不贡献答案；已有 CoT 压缩方法只优化输出长度，从不度量某个推理 token 是否真的落在图像上。**VIG** 提出信息论式的 GRPO reward，按「图像使该推理 token 的预测不确定性下降多少」为每个 token 打分，在线由同一 policy 的两次前向（带图与不带图）计算得到，无需参考链、外部标注或额外 reward model。六个主流多模态推理基准、三个 Qwen3-VL-Thinking 规模（2B/4B/8B）以及 8B 上额外的 R1-Onevision-Bench 评测中，均一致改善准确率-效率折衷（abstract 未给出具体数值），支撑其核心主张：高效多模态推理来自提升视觉信息密度而非施加长度预算。
- **arXiv**：[2608.21883](https://arxiv.org/abs/2608.21883)

#### Learning When to Think: Adaptive Reasoning for Test-Time Compute Allocation (2026-08)
- **简介**：Gijs Kassenaar、Zhao Yang、Vincent François-Lavet。针对 RL 训练的推理模型通常在固定 token 预算下工作、导致简单题过度计算而难题计算不足的问题，作者让模型用回答的第一个 token 自选推理模式：NoThink（尽快作答）、Short（简短推理）、Long（长推理）。该选择直接在 GRPO 内部学习、不需要独立 router，依靠一个使各模式在不同回答长度上分别划算的 shaped reward，配合每模式的硬 token 上限以保持模式区分度。在 MATH 上训练的 1.5B 蒸馏模型中三种模式都出现且未塌缩到单一选择，且短模式最终比 Long 更准，说明 router 是按难度而非随机分流；三个种子平均下，所得策略在 held-out MATH500 上接近 base 模型（0.782 vs 0.796），平均回答长度从 4796 token 降到 2811 token（减少 41%），并可零重训迁移到其他基准——题目越简单节省越多，例如 GSM8K 上 token 减少 76%，且在相近回答长度下准确率高于基线。
- **arXiv**：[2608.20256](https://arxiv.org/abs/2608.20256)

#### Test-Time Scaling in the Wild: Why Exploitation, Not Exploration, Is the Bottleneck (2026-08)
- **简介**：Davide Romano、Kanak Raj、Jerrod Parker、Daniele Giofrè。指出 test-time scaling（TTS，生成多候选、在部分序列上搜索、或迭代精修草稿）虽在数学与代码上收益巨大，却几乎只在验证容易的任务上被开发与压测。作者首次在算力归一化条件下比较五类 TTS 方法在医学、法律、金融、通用对话与创意写作五个开放式生成基准上的表现，并用统一框架把每种方法的 token 预算效果分解为 exploration 与 exploitation 两侧。结论取决于看哪一侧：扩展 exploration 是有效的——候选池中的最佳候选随算力稳定变好；真正失效的是 exploitation，即把丰富候选池转成最终输出的那一步——在 SOTA generator 下 reward model 与真实质量的相关仅 ρ_v ≈ 0.12，使选择近乎随机、与预算多少无关，tree search 还通过 diversity collapse 放大这一失败；refinement 只在五个基准中的一个有效，其余表观增益均被混淆因素解释；只有跨候选的合成（Fusion）稳定优于单样本基线，但也只回收约 40% 的可得质量。属 §1.9 的 TTS 实证诊断分析（不提出新方法），是「verifier / reward model 质量决定 test-time scaling 上限」的关键对照。
- **arXiv**：[2608.18931](https://arxiv.org/abs/2608.18931)

#### Beyond the Best Guess: Improving LLM Solution Coverage with Evolution Strategies (2026-08)
- **简介**：来自 Cognizant AI Lab（Conor F. Hayes、Elliot Meyerson、Babak Hodjat、Risto Miikkulainen、Xin Qiu 等 7 人）。针对数学与科学等发现型任务，指出常规做法只取模型的 best guess，而增加 test-time compute、以 pass@k 方式探索解空间生成多样候选能带来更大收益；问题在于用 RL 做后训练会让输出分布收窄到高 reward 输出附近，导致解覆盖度（solution coverage）崩塌、pass@k 受限。作者主张改用 Evolution Strategies（ES）——一种基于种群、无梯度、通过随机扰动直接在权重空间优化的后训练方法，并给出证据显示 ES 一致取得比 RL 更高的 pass@k，产生更宽的输出分布与更大的解覆盖度，进而在标准数学基准上取得更好结果；结论是在需要多样解覆盖的发现型问题上，ES 是比 RL 更好的后训练基础（abstract 未给出具体数值）。
- **arXiv**：[2608.12679](https://arxiv.org/abs/2608.12679)

#### DIVE: Unlocking Self-Improvement in Frozen Language Models Through Diversity-Driven Skill Evolution (2026-08)
- **简介**：来自 Georgia Tech 与 Cisco（Siheng Xiong、Ali Payani、Oguzhan Gungordu、Faramarz Fekri）。针对 LLM 不更新参数就无法保留部署后经验的问题，提出 **DIVE**：让 frozen LLM 从任务经验与 verifier 反馈中演化出持久的自然语言 skill，这些 skill 编码可复用的推理流程、验证策略、常见失败模式与输出约束，并由同一个底层模型负责执行与修订，不需要 teacher 模型。由于自然语言 skill 演化是随机、非凸的搜索过程，只优化单条 skill 轨迹容易过拟合已采样经验或收敛到次优解，DIVE 因此从 bootstrapped 经验中独立演化多个 skill 种群、用多样化变换自适应精炼，并联合挑选一组互补的 skill 来抑制优化方差。在六个数学与逻辑推理任务、多个模型族上，DIVE 一致优于既有推理方法、prompt 优化方法、skill 开发框架与 memory 类 baseline，并以远少于 SFT、GRPO 等参数式方法和 GEPA prompt 优化的 rollout 数取得明显更大的提升；演化出的 skill 还能跨模型规模与家族迁移，使 GPT-5-nano 在常规 prompting 下追平或超过 GPT-5（abstract 未给出具体数值）。
- **arXiv**：[2608.12486](https://arxiv.org/abs/2608.12486)

#### Refining Over Resampling: Test-Time Self-Correction for LLM Reasoning (2026-08)
- **简介**：来自 Ahsan Bilal、Muhammad Ahmed Mohsin、Muhammad Umer、Dean F. Hougen 等 7 人。指出单纯加宽采样的 test-time scaling 存在收益递减——新的 rollout 往往只是重复已有答案模式，并不带来有效的推理多样性；而 verifier-based 选择则依赖外部 reward model 的校准。提出免 verifier 的「广度—深度」精化框架：先采样多条彼此独立的推理 rollout（广度，保留初始尝试的多样性），再对每条 rollout 施加迭代式 self-critique 与 self-correction（深度，在聚合前修复局部推理错误），最后对精化后的答案做 majority voting。在 AIME24、AIME25、AMC、OlympiadBench、MATH500 上，多个开源权重模型均一致优于贪心解码、majority voting、verifier-based best-of-$N$、beam search 与 lookahead decoding；例如 Qwen2.5-1.5B 上 MATH500 相比最强 verifier-based 基线提升至 58.0%，AMC 从 25.0% 提升到 32.5%。
- **arXiv**：[2608.05643](https://arxiv.org/abs/2608.05643)

#### Hi-TTRL: Regulating Consensus with Hints for Test-Time Reinforcement Learning (2026-08)
- **简介**：来自 Kunbin Xu、Xingzuo Li、Xuefeng Bai、Kehai Chen。指出 TTRL 用 majority voting 构造伪标签、在无标注数据下更新策略虽有效，但其 reward 信号对 consensus strength（rollout 组内最高频答案的出现频率）高度敏感，而后者兼具双重角色：既反映伪标签的可靠性，也决定 advantage 的分布——低共识会让不可靠伪标签因过大的 advantage 放大更新，高共识则压缩 reward 对比、最终导致梯度消失。提出 **Hi-TTRL**：先用部分 rollout 组估计 consensus strength，一旦落到目标区间之外便调用 MCMC hint sampler，该采样器以幂变换后的前缀分布为目标、通过有限步近似采样生成 rollout 前缀作为 hint，调节幂指数即可产生锐化或平坦化的提示，把 rollout 共识强度引导回目标区间。多数据集、多 backbone 实验表明 Hi-TTRL 一致优于标准 TTRL，消融与共识调控分析验证了自适应 hint 引导的有效性。
- **arXiv**：[2608.03545](https://arxiv.org/abs/2608.03545)

#### β-OPSD: Deriving with Policy Optimization, Training with Self-Distillation (2026-07)
- **简介**：来自 UMD（Jiawei Xu、Tom Goldstein、Furong Huang 等）。指出 vanilla OPSD 恰是一个更广策略优化族中 β=1 的成员（β 权衡将 student 锚定到参考策略的 KL 惩罚）。据此把 β 从固定值 1 变为可控正则参数，得到 **β-OPSD**：其最优策略为参考策略与特权 teacher 之间的几何插值；不直接做高方差 RL，而是把闭式解转化为蒸馏目标——对每个 β 沿"参考→teacher"路径选目标，用二者 token 级 logit 混合高效实现，再配 return-to-go 信用分配对齐序列级目标。数学推理基准上一致优于 vanilla OPSD，兼顾稳定性与效率。
- **arXiv**：[2607.28582](https://arxiv.org/abs/2607.28582)

#### Lightning OPD 2.0: Mitigating Style Bias in Cross-Teacher On-Policy Distillation for Large Reasoning Models (2026-07)
- **简介**：来自 MIT / NVIDIA（Yecheng Wu、Song Han、Han Cai）。在线策略蒸馏（OPD）通常要求 teacher 一致性（提供监督的模型也须生成 SFT 参考数据），实践中常被违反。作者发现 teacher–reference 分歧含"有用的上下文特定证据"与"用词/格式/推理节奏差异"两成分。**Lightning OPD 2.0** 用 rollout 级 cross-fitting 估计后者作为 style-token bias 代理并在构造 token 级 OPD 更新前减去。从 Klear-Reasoner-8B-SFT 出发，在 AIME 2024 达 82.4%、LiveCodeBench v5 达 63.0%，在 cross-teacher 设定下一致优于 Lightning OPD。
- **arXiv**：[2607.28449](https://arxiv.org/abs/2607.28449)

#### Contrastive Reinforced Policy Optimization via Privileged Self-Distillation (CRPO) (2026-07)
- **简介**：来自美团（Xingjian Wu、Xunliang Cai 等 9 人）。指出 On-Policy Self-Distillation（OPSD）虽提供稠密 logit 级监督，但因 self-teacher 的特权信息带来 exposure bias，在多轮 agentic 场景下导致推理路径收敛、优化方向丢失。**CRPO** 从对比学习视角重构 agentic OPSD：用预测熵区分正位置（反思性探索）与负位置（exposure bias），做组内对比以保留可靠的细粒度优化信号。在 13 个推理与深度搜索基准上一致超越已有 RL 与自蒸馏基线，显著提升长程交互的训练稳定性与泛化性。
- **arXiv**：[2607.28026](https://arxiv.org/abs/2607.28026)

#### Weak-to-Strong On-Policy Distillation (W2S-OPD) (2026-07)
- **简介**：来自 Microsoft、UMD 等团队（Fangxu Yu、Zinan Lin、Jianfeng Gao 等，Technical Report）。传统在线策略蒸馏（OPD）假设 teacher 至少与 student 一样强，在前沿模型无更强 teacher 时失效。**W2S-OPD** 在 logit 空间用一对"正/负"弱模型（均比 student 小且廉价）构造 proxy teacher：二者 logit 差隔离出"能力方向"，加到 student 自身 base 上得到分布上仍与 student 邻近的代理教师，再对 student 自身 rollout 做逐 token reverse-KL 蒸馏。三种对比对（post-RL vs pre-RL、大 vs 小 base、含正确/错误提示）在 4 个数学 + 3 个代码基准上超过 OPD，甚至让 student 超过领域 teacher。
- **arXiv**：[2607.26246](https://arxiv.org/abs/2607.26246)

#### Probing the Origins of Reasoning Performance: Representational Quality for Mathematical Problem-Solving in RL vs. SFT Fine-Tuned Models (2026-07)
- **简介**：来自 Algoverse AI Research 等（Antyabha Rahman 等，AAAI 2026 XAI4Science Workshop）。机理分析类工作，追问"为何 RL 微调模型在数学推理上优于 SFT 模型"。两条证据：① 层级隐状态上训练的线性探针显示 RL 模型对"答案是否正确"的预测精度更高，表征更线性可分、更结构化；② 均值消融显示 RL 模型形成"越深层越关键"的层级架构，而 SFT 模型层间重要性均匀分布。此外分析重复采样下的 token 数变异，指出 token 分配更取决于整体训练流程而非单纯 RL/SFT。
- **arXiv**：[2607.26119](https://arxiv.org/abs/2607.26119)

#### From RLVR to RLSVR: Task Transformation Induces Self-Verifiable Rewards for Open-Ended LLM Self-Improvement (RLSVR/SpyRL) (2026-07)
- **简介**：来自 Duke、Adobe 等团队（Qinsi Wang 等 11 人，COLM 2026）。针对 RLVR 只能用于数学/代码等可确定性验证域的局限，提出 **RLSVR**：借鉴自监督学习"构造 pretext task"的思路，把开放式任务转化为规则可自动产生奖励的可验证代理环境。以"谁是卧底"多智能体自博弈环境 **SpyRL** 为实例——卧底身份预先确定，投票结果即完全可验证的奖励。实验在文本摘要、创意写作、数学推理上均超过已有自改进方法，并在可验证推理任务上取得一致增益。
- **arXiv**：[2607.23802](https://arxiv.org/abs/2607.23802)

#### Test-Time Scaling via Error Localization (TTEL) (2026-07)
- **简介**：Rajiv S. Chitale、Aravindan Raghuveer 等（Google）指出独立采样与顺序多轮修正缺乏 token 级信用分配、频繁丢弃有效推理前缀。提出 **TTEL**：推理期算法，用固定或环境反馈做 token 级错误定位——比较"有反馈"与"null-context 基线"下的条件概率以隔离出错步，随后截断并分支新生成、最大化复用有效前缀。在 pass@k vs 生成 token 成本上建立严格占优 Pareto 前沿：Qwen3-8B 在 LiveCodeBench 上 pass@64 达 71.0%、生成 token 约为独立采样的一半（360.4k vs 735.0k）；在 AIME-2025/HMMT-2025 上（Qwen3-8B 与 Qwen3-4B-Thinking-2507）干净超越竞争的 test-time 基线。
- **arXiv**：[2607.21453](https://arxiv.org/abs/2607.21453)

#### SLPO: Scaling Latent Reasoning via a Surrogate Policy (SLPO) (2026-07)
- **简介**：Runyang You、Yongqi Li、Wenjie Li 等（PolyU）指出 latent reasoning（以连续向量承载中间计算）缺乏可解的 per-step likelihood 与固定预算下的自适应停止接口，使 outcome-reward RL 难以激发 latent 端的 test-time scaling。提出 **Surrogate Latent Policy Optimization (SLPO)**：构造 latent transition 上的经验代理策略密度做轨迹级信用分配，并用 correctness-supervised 停止头，经 outcome-reward 优化精炼为可变视界策略。在连续/软思维设定下提升并行采样 Pass@k，并把更长的 latent 计算分配给更难实例（确定性精度更高）。
- **arXiv**：[2607.19691](https://arxiv.org/abs/2607.19691)

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

#### From RL Papers to Framework Code: Inclusion and Review in Seven Open-Source Training Frameworks (2026-10)
- **简介**：作者 Taiheng Pan。实践者通过训练框架使用 LLM 的 RL 方法，框架收录某方法就意味着用户一个配置即可使用。论文通过提交历史追踪 7 个开源 RL 后训练框架，并关联 2025 年 1 月至 2026 年 6 月的 6,939 篇筛选后的 arXiv 论文：在提出通用 RL 训练方法的 1,598 篇中，仅 37 篇进入至少一个框架；五个季度队列中，180 天内被收录的数量稳定在 5-7 篇，而队列规模从 72 增长到 386 篇。被收录论文更常训练 30B 以上策略（41% 对 10%），更常修改既有损失中的系数、clip、mask 或权重（29% 对 9%），这些差距在剔除框架维护公司的论文后仍成立，代码发布率两组相当。作者发布了把训练失败与框架实现关联的 tracker，并列出各框架的贡献规则与历史提案。
- **arXiv**：[2610.28027](https://arxiv.org/abs/2610.28027)

#### Checking the Checkers: A Synthesis of Verifier Noise and Robustness in RLVR (2026-10)
- **简介**：作者 Savyasachi Gupta、Aniruddha Srinivas Joshi（AISSA 2026 录用）。RLVR 假设打分的 checker 本身正确，但实践中规则 checker 会拒绝格式特殊的正确答案、单元测试会放过错误程序、LLM judge 可能被空白或对抗性回答欺骗。本文综述 58 项测量、分析或缓解 verifier 错误的研究，围绕错误类型、对训练的影响、不同缓解手段的适用性，以及对 self-generated、rubric/judge 与 verifier-free 奖励的推广展开。主要结论：错误的结构比错误率更重要；随机错误多半只是减慢学习，而策略可学会触发的 false positive 会导致平台期或崩溃；多数缓解方法只针对过一类错误。文中给出对比表并提出报告协议以使未来结果可比。
- **arXiv**：[2610.17824](https://arxiv.org/abs/2610.17824)

#### Rollout Efficiency in Reinforcement Learning for Reasoning Large Language Models: A Taxonomy and Future Directions (2026-09)
- **简介**：作者 Niloofar Gholipour、Marcos Assuncao、Gursimran Singh 等。推理类 RL 把大量训练成本转移到 rollout 阶段，因此需要在保持训练数据新鲜度、一致性与统计有效性的同时降低 rollout 成本。这篇综述从机制和瓶颈两个视角给出近期 rollout 效率研究的系统分类，分析不同技术族如何应对不同来源的 rollout 低效，探讨各技术组合的机会与潜在冲突，指出效率收益在评估与汇报上的缺口，并讨论开放挑战与未来方向。
- **arXiv**：[2609.25463](https://arxiv.org/abs/2609.25463)

#### Proof-Carrying Cognition: Closing the Verification Gap with Reality-Settled Reward (2026-09)
- **简介**：Eshwar Reddy M、Sourav Karmakar。论点是当前推理 RL 的瓶颈在 verification gap——形式化领域之外缺少可扩展且不可腐化的 reward。理论上，在 best-of-N 选择的 joint-Gaussian 模型中，verifier 与 gold 的相关系数 rho 恰是 test-time compute 与能力之间的"汇率"，unsound verifier 需付出 N^(1/rho^2) 的多项式惩罚，其 margin-free copula 形式对真实 LLM judge 的 soundness 预测中位误差为 4%。实证上，在带可执行 ground truth 的程序合成 testbed 中，unsound verifier 的 Soundness-under-Pressure 随优化增强从 0.94 掉到 0.32（N=4096），而 sound verifier 单调改善；reality-anchored settlement 把 hacking gap 从约 0.27 压到约 0，on-policy settlement 的标签效率约为随机标注的 10 倍；仅靠选择就能从诚实样本中制造出 +0.53 的 hacking gap；在真实 GRPO 训练下，frozen reward model 走完整条 overoptimization 曲线（executed reward 崩塌 90%），而用 10% settlement 流重新拟合的同一模型保住了 6 倍的 executed reward。并提出 proof-carrying cognition 范式（推理步骤作为带类型的概率主张，由仅在 held-out 现实上训练的世界模型定价、以 proper scoring rule 结算）与以 Soundness-under-Pressure 为主指标的基准规格。
- **arXiv**：[2609.09776](https://arxiv.org/abs/2609.09776)

#### ConsensusBench: Benchmark of Consensus Nodes for LLM Reasoning via Outcome Reward Densifying (2026-09)
- **简介**：Shi-Qi Yan、Chao-Hong Tan、Qian Chen、Wen Wang 等 6 人。指出 GRPO 及相关算法虽在 outcome-level reward 下表现强劲，却只依赖最终答案，无法反馈哪些中间步骤促成了成功或失败；随着任务复杂度与推理轨迹长度上升，这种稀疏的 final-answer reward 越来越不够用。作者的假设是正确答案依赖于推理过程中少量关键的中间结论，这些结论可被视为可验证的 sub-outcome：从 N 条 rollout 中筛出正确轨迹，再对语义等价的中间陈述做聚类，得到 **Consensus Nodes**；据此导出 rule-based process reward **ConsensusPR** 并接入 GRPO 类算法，直接缓解长推理轨迹上的 reward 稀疏问题。配套基准 **ConsensusBench** 提出三个过程级评测指标：Final Answer Accuracy (Acc)、Node Coverage Rate (NCR) 与 Tokens per Node (TPN)。在 AIME 2024、AIME 2025、GSM8K、MATH-500 与 ConsensusBench 上，该方法持续超过 GRPO 类方法（abstract 未给出具体数值）。
- **arXiv**：[2609.04648](https://arxiv.org/abs/2609.04648)

#### Unifying ICL, SFT, KL-Regularized RL Through a Bayesian Lens (2026-09)
- **简介**：Junxin Fan（单作者）。针对 SFT、few-shot ICL、KL-regularized RLHF/RLVR、on-policy distillation (OPD) 与带搜索及 CoT 的 test-time reasoning 常被当作根本不同的范式讨论、以致 few-shot prompting 对 RL 调优推理模型影响不一等实证结果显得费解的现状，本文提供一个把它们置于同一框架下的贝叶斯视角。核心是两步模板：(i) 用 prior/reference model 与一个 utility 信号（log-likelihood、reward 或 advantage）在输出或动作上构造（广义）Bayes 或 Gibbs 后验 q*；(ii) 用 forward-KL 投影把 q* 近似到某个参数族上，既可以是 in-weights（SFT/RL）也可以是 in-context（ICL）。Part I 把 few-shot ICL 与 SFT 形式化为对 Bayes posterior predictive 的 amortized 与 in-weights 投影；Part II–IV 证明 KL-regularized RLHF/RLVR、reward-weighted SFT、reward-weighted ICL (RW-ICL) 与 advantage-weighted SFT (AWSFT) 都是对 reward 或 advantage 所诱导后验的 forward-KL 投影，并厘清这些等价在哪里成立（目标函数与一阶更新）、在哪里不成立（学习信号的来源与粒度）；Part V 给出对现代推理流程的推论，包括把 RLHF/RLVR 配方理解为"posterior design + projection"、为何对 importance-weighted KL 投影而言 cold-start 或有监督 warm-up 实际上不可避免，以及把 DeepSeek-R1 与 o1 式推理模型视为 test-time 贝叶斯搜索与训练期 KL 摊销的组合。属 §1.10 的统一理论视角，不提出新方法。
- **arXiv**：[2609.05111](https://arxiv.org/abs/2609.05111)

#### Revisiting Complete Reasoning Traces for Post-Training (2026-09)
- **简介**：来自 NAVER AI Lab（Jaehui Hwang、Sangdoo Yun、Byeongho Heo、Dongyoon Han）。指出后训练常直接在预先收集的推理轨迹上进行，而这些轨迹因路径复杂交错而冗长、往往包含通往答案途中的绕路，但"LLM 是否真的从学习完整轨迹中获益"一直缺乏检验。pilot study 发现完整轨迹带来的收益有限，而部分轨迹即便被大幅截断依然有效；作者随后通过基于 attention 的分析与受控 token 移除实验进一步说明，中间 token 对最终推理质量的贡献极小，这暗示在轨迹端点已知的前提下，避开冗余信息反而让 LLM 得以调用内部知识补出缺失步骤、在内部推演出连贯的替代路径。作者还表明仅用端点训练会带来一致的推理行为变化，并且同样有利于基于 RL 或 on-policy distillation 的后训练方法。属 §1.10 的后训练数据形态实证再审视，结论是需要重新检视"完整推理轨迹"这一默认做法，而非提出新算法（abstract 未给出具体数值）。
- **arXiv**：[2609.07103](https://arxiv.org/abs/2609.07103)

#### Where the Verifier Fails: A Category-Level Audit of Reward Signals in RLVR (2026-09)
- **简介**：来自 Esther Xin（单作者）。RLVR 与标准基准评测都依赖一个把自由文本答案转成二值 reward 的自动 verifier，先前工作报告某评测框架只接受约 94% 的自家 ground-truth 答案并归因于 LaTeX 解析，但这只是聚合数字、没说清哪些答案形式消耗了误差预算。本文给出这一分解：把 metamorphic testing 施加于 verifier 而非模型，生成 certified equivalent 答案变体（构造上即保持数学含义的改写），因此任何拒绝都是可证明的 false negative、无需人工裁定；在四个常用 verifier 上统计 307,420 条判决的分类别拒绝率。三点发现：(1) 相同输入下 self-validation 从 53.8% 到 95.2%，跨度 41.3 个点，同一库的两种配置在 49.9% 的样本对上判决不一致，说明已发表数字描述的是某个实现而非任务本身；(2) 残余误差并非均摊在各解析类别，而是集中于空白与标点，占默认 LaTeX 配置 in-contract 失败的 93.0%，一个尾随句点或换行即主导误差预算；(3) 把拒绝与执行失败分开后可见，聚合误差相近的 verifier 失败原因相反，且某 reference numeric cascade 因相对容差尺度不变，会按量级阶梯式接受 off-by-one 的错答——小于 10^4 时为 0%，达到或超过 10^4 时为 100%。属 §1.6 的 verifier / reward 信号可靠性实证审计，不提出新的训练方法。
- **arXiv**：[2609.01354](https://arxiv.org/abs/2609.01354)

#### The Rise of Verbal Reinforcement Learning (VRL) (2026-09)
- **简介**：来自 Kshitij Tayal、Arun Sharma、Genta Indra Winata、Anirban Das 等。作者把「以自然语言作为改进 language agent 的主要反馈通道」这一新兴范式命名为 **Verbal Reinforcement Learning（VRL）**——自然语言能以人与现代 LLM 都可解释的形式传达意图、偏好与因果结构——并给出该领域的首个统一梳理。全文围绕单一轴线组织：verbal feedback 在 agent 生命周期的何时生效、以及它修改了什么，由此得到三大支柱：(1) Language as Grounding Signal，语言通过指定目标、状态与 reward 结构来定义任务本身；(2) Language as Deliberative Feedback，自然语言在推理期引导推理、无需更新模型参数；(3) Language as Learning Signal，语言反馈经由训练塑造模型参数。每一支柱内综合代表性工作、区分关键子类，并说明语言在塑造 agent 行为中的不同角色，最后指出该范式面临的挑战与机会。属 §1.10 的系统性综述。
- **arXiv**：[2609.01597](https://arxiv.org/abs/2609.01597)

#### BAITBENCH: Measuring Agent Reward Hacking with Optional Shortcuts Planted in ML Tasks (2026-08)
- **简介**：来自 Pradyumna Shyama Prasad、Meiri Anto、Leon Eshuijs、Julian Moncarz 等。LLM agent 正被越来越多地用于在几乎无人监督的情况下自主运行 ML 实验、围绕目标指标迭代，已有工作记录了其中的 reward hacking，但现有基准都不度量藏在数据或建模任务本身之中的 exploit。作者提出 **BAITBENCH**：三个合成表格 ML 任务，每个都埋入一条能抬高 public test 分数、却在 hidden test set 上失效的 shortcut；由于该 shortcut 是可选的、使用它也不违反任何明示规则，基准度量的正是模型主动利用它来虚高分数的频率。用两阶段 judge pipeline 评测七个前沿 agent，57.1% 的运行出现 reward hacking，七个中有五个超过 50%；即使在第二种被明确提示不要作弊的条件下，平均作弊率仍高于 50%。作者同时开源基准、judge 实现与一份含 reward hack 的标注 transcript 数据集。属 §1.6 的 agent reward hacking 度量基准，用于横向比较各类缓解手段。
- **arXiv**：[2608.30724](https://arxiv.org/abs/2608.30724)

#### Scaling Large Reasoning Models beyond Human Supervision: A Path toward Superintelligence (2026-08)
- **简介**：来自 Zhiqin Yang、Jingwen Fu、Yuhan Liu 等 19 人（含香港科技大学 Wei Xue、Yike Guo）。RLVR 已在数学与代码等可自动核验的领域显著提升大推理模型（LRM），但推广到开放式与 agentic 任务困难重重：可靠 reward 更难获得，直接的人类监督也跟不上模型自生成经验的规模与复杂度。本文系统研究人类监督逐步退出学习回路后 LRM 如何持续改进，沿两条相连维度梳理：reward 轴追踪从 per-instance 人类判断到可复用 verifier、再到无需人类反馈即可运作的 reward 的演进；experience 轴考察学习如何从人工策划的任务与环境走向自生成 curriculum、构造式环境与自主协同演化。两条维度由一个 L0 到 L4 的五级阶梯串联，用以标识学习过程中哪些部分仍受人类控制；分析还指出日益自主的 reward 与经验生成带来的风险，包括 reward hacking、feedback drift、curriculum collapse 与 environment error，并给出围绕 policy capability、feedback fidelity、experience quality 三类对象的互补评测方案。属 §1.10 的系统性综述，配有持续更新的 GitHub 论文列表。
- **arXiv**：[2608.31075](https://arxiv.org/abs/2608.31075)

#### Consolidating RLVR Capabilities Across Domains: A Deep Dive into Fusion Paradigms (2026-08)
- **简介**：来自复旦大学、腾讯等机构（Siye Wu、Kai Yang、Yuchen Cai、Xin Xu 等 11 人）。针对「RLVR 通常只提升单一能力、覆盖多能力需先训领域专家再融合」这一实践困境，按复用的产物把融合范式归为三类：Merge（合并专家 task vector）、Mix RL（汇合各专家数据集重新做 RL）与 multi-teacher on-policy distillation（**MOPD**，两者都用）；由于以往三者各自孤立研究，如何比较与选择并不清楚。作者在共享专家与共享数据、跨多个模型规模与多领域基准套件上做统一对照，发现三者平均性能差距最多 1.4 个点，但在单一基准上差距可达 8.6 个点，且领域级差异与 task-vector 几何中体现的跨领域关系一致；训练动态揭示各自约束：Mix RL 依赖领域混合比例，MOPD 受限于其 teacher 上界，Merge 把所有专家更新压缩进一次合并；三者都只提升单样本准确率，而在解覆盖度（solution coverage）上无可测增益、在留出能力上也无损失。最终给出选型准则：已有专家且看重廉价融合用 Merge，无专家而要训统一模型用 Mix RL（并按跨领域迁移调混合比），更看重保住领域收益则用 MOPD。属 §1.1 的多领域能力融合范式对照分析。
- **arXiv**：[2608.27409](https://arxiv.org/abs/2608.27409)

#### Demystifying Reinforcement Learning Post-Training of Language Models (2026-08)
- **简介**：来自华盛顿大学与 AI2 等（Donovan Clay、Saket Gollapudi、Sankar Harilal、Sewoong Oh、Natasha Jaques 等 7 人）。针对「RL post-training 已成为提升 LLM 推理、数学与代码能力的强力框架，但经典 RL 背后的原理对许多研究者与实践者仍是黑箱」这一现状，作者逐步拆解 RL post-training 算法，在受控简化环境中隔离出 RLVR 的运作机制，系统考察 RL 结果如何被 base model 的先验分布、reward 信号的粒度、prompt 分布的多样性以及模型规模所塑造；并以策略输出分布的 entropy 为透镜，对比 pretraining、SFT 与 RL post-training 各阶段学到的分布，揭示每个阶段如何塑形模型的确定性。结论包括：所谓 spurious reward 的效果取决于 post-training 所用的 prompt 分布；RL post-training 能否成功取决于 base model 是否已在目标行为上放置了足够的概率质量，并将其与经典 RL 中的探索概念相联系。属 §1.10 的机制性实证解析与入门 primer（不提新方法，面向 NLP 社区提供 RL 工具箱的系统梳理）。
- **arXiv**：[2608.24949](https://arxiv.org/abs/2608.24949)

#### Stopping and Routing LLM Judge Panels (2026-08)
- **简介**：Bin Zhu、Yi Xie、Yanghui Rao。LLM 评测流水线往往有大量候选 judge——通用 LLM-as-a-judge prompt、reward model、安全分类器、置信度变体、任务专用 verifier——真正的部署问题不只是「哪个 judge 最好」，而是该调用哪些 judge、在哪些样本上调用、以及 panel 何时停止扩充。作者把 judge panel 设计形式化为 role-conditioned 分配问题：从一个小规模标注审计集、声明的数据切片与各 judge 成本出发，估计相对目标的角色——copy（不增加条件信息）、complement（改善全局 panel）、specialist（只在特定 slice 有用）；这些角色直接导出策略：丢弃 copy、全局加入 complement、按条件路由 specialist，并在验证增益低于阈值时停止。在推理、代码、安全、偏好、reward-model、摘要与数学等审计集上，与单 judge、扁平 panel、匹配多样性启发式、full-call stacking、reliability juries 与 frugal cascade 做了比较，产出一张 judge 调用的 regime map（在可部署 slice 上路由 specialist、在 verifier 已饱和的 regime 停止、当风险收益值得成本时保留宽集成、忽略条件 copy）与可复用可审计的调用计划（abstract 未给出具体数值）。属 §1.8 中围绕 judge/reward model 集成与调用成本的方法性工作，本身不含 RL 训练环节。
- **arXiv**：[2608.19802](https://arxiv.org/abs/2608.19802)

## 2. Agentic RL

> 多轮、长 horizon、部分可观测的智能体 RL，覆盖工具调用、GUI、网页、代码、记忆、安全。

### 2.1 Tool-use / Multi-turn Agent

#### What Does Multi-Turn Reinforcement Learning Really Improve in LLM Agents? (BudgetMix RL) (2026-11)
- **简介**：作者 Kejian Zhu、Zhuoran Jin、Shangqing Tu 等。多轮 agentic RL 的收益对交互约束非常敏感。论文用 $Pass@(K,T)$（采样宽度 K 与交互深度 T）评估，发现轨迹级 RL 的优势依赖足够的 turn 预算，预算收紧时显著缩小；agentic RL 也并非均匀提升单步决策质量，其优势集中在错误恢复与长程韧性上。为缓解对预算的依赖带来的部署风险，作者初步讨论了过程监督信号的利弊，并提出 **BudgetMix RL**：在训练中变化交互预算，使模型在多种执行设置下表现稳健。（abstract 未给出具体数值）
- **arXiv**：[2611.06483](https://arxiv.org/abs/2611.06483)

#### ToolSpace: Scaling Tool-Centric Environment Synthesis for Long-Horizon Agents (2026-11)
- **简介**：作者 Guanting Dong、Zhichao Sun、Xinyu Yang 等。现有 MCP 环境扩展方法以数据为中心（先合成数据库和场景再派生工具与任务），导致工具接口与真实不符、工具组合缺乏执行落地、任务很少跨多个环境。**ToolSpace** 反转顺序：用确定性规则（而非 LLM 判断）从真实 MCP 服务与开源 agent 语料的执行轨迹中挖掘相邻调用间的数据/状态/重试依赖，对加权工具组合图聚类为环境，仅为状态变更环境合成数据库，其余工具回放记录；再做带回滚的状态感知随机游走，并将链反编译为带断言检查的自然语言任务，只保留可从初始状态成功回放的样本。该流水线在 AutomationBench 与 tau3-Bench Banking 上生成更难的 Pro 版，7 个前沿模型最多下降 38.2 分；在 7 个长程工具使用 benchmark 上，Qwen3.6-35B-A3B 平均分由 23.0 升至 41.0（加 RL 为 43.2），GLM-5.2 由 44.6 升至 56.6。
- **arXiv**：[2611.05036](https://arxiv.org/abs/2611.05036)

#### Agent-R$^2$: Scaling Agents for Realistic Productivity Work via Real-Source Data Synthesis (2026-11)
- **简介**：作者 Teng Pan、Zixuan Wang、Yuchen Yan 等。现实生产力任务需要遵循长程专业工作流、理解异构文件上下文并产出满足细粒度要求的交付物，瓶颈在于缺少反映真实工作意图、真实文件环境与交付物级评估的后训练数据。**Agent-R$^2$** 是 real-source 数据合成框架，全流程以真实职业种子、语料推导的工作流证据和网络真实文件为基础：从 1 万个网页语料候选中蒸馏出 4k 个任务实例，每个配有真实多格式附件与细粒度交付物级 rubric。在此数据上通过 SFT 与基于 rubric 的 RL 训练 **Agent-R$^2$-4B**，仅用 4k 实例即在 GDPval（51.0% vs 48.0%）与 OfficeQA（14.8% vs 13.8%）上超过 Qwen3-235B-A22B-Thinking-2507。
- **arXiv**：[2611.08219](https://arxiv.org/abs/2611.08219)

#### Axion 1.0: Open Agent Foundation Models, Simplified (2026-11)
- **简介**：作者 Yuqi Jia、Yunjie Tian、Jiayu Zheng 等。**Axion 1.0** 是完全开放的 agent 基础模型，训练配方极简：两阶段轻量 SFT 后接 RL，不含专门的 mid-training、DPO 或 on-policy distillation。第一阶段 SFT 通过 model-aware 的数据构造（基于基座自身成功解法对合成难题做对抗式增强）培养自洽推理；第二阶段通过可扩展流水线构造基于真实数据源与确定性计算的、可自动验证的多样任务，把能力扩展到依赖工具的推理；最后 RL 进一步强化长程推理及与外部工具/环境的交互。在一系列 agentic benchmark 上表现强劲，在若干项上可与领先商业模型竞争。模型之外还开放了完整配方，包括数据构造、生成、过滤、验证、prompt 与训练代码（abstract 未给出具体数值）。
- **arXiv**：[2611.03233](https://arxiv.org/abs/2611.03233)

#### Stabilizing On-Policy Targeted Guidance in Agentic Reinforcement Learning (2026-11)
- **简介**：作者 Kevin Frans、Alexander M Rush、Sergey Levine、Pieter Abbeel、Ashvin Nair。朴素 on-policy RL 受限于基座模型能力，覆盖不足时无法从某些奖励中学习；而现代后训练中的奖励常来自富文本源（如带 rubric 的 judge 模型）。论文研究 **targeted guidance** 框架：通过 on-policy self-distillation 目标从每轮（per-turn）文本反馈中学习，重点不仅是直接改进难优化的行为，更是在整个训练过程中做到这一点而不损害基线 RL 表现。在 agentic 多轮软件工程环境的一系列设置中，将 targeted self-distillation 与逐准则（per-criteria）RL 奖励结合、并把反馈限制在策略确实失败的 turn，取得成功；还分析了使熵稳定、减少训练振荡的设计选择（abstract 未给出具体数值）。
- **arXiv**：[2611.05266](https://arxiv.org/abs/2611.05266)

#### Harness-S: Learning from Sibling Harnesses for Generalizable Agent Training (2026-11)
- **简介**：作者 Kangning Zhang、Wenxiang Jiao、Bowen Fang 等。作者观察到从同一基座演化出的 harness 具有局部性、互补性与非平稳性：不同变体修复不同失败，其效用随策略学习而变。**Harness-S** 不追求通用最优 harness，而是通过 sibling harness 有选择地「救援」基础执行系统的失败，并把获得的能力迁移进单一策略。Directional Harness Evolution 沿不同修复方向演化出固定的 Sibling Harness Pool；Complementary Harness Rescue Training 仅在基础 harness 下所有初始尝试都失败时调用池，同一策略在辅助 harness 下生成新 rollout 以替换失败组做 RL；Adaptive Harness Router 利用 Harness Behavior Profile 与多 LLM 投票选 Top-k，并随策略重新锚定。在 AppWorld、SWE-Gym、DBBench 上，回到原 harness 时显著优于仅策略训练。
- **arXiv**：[2611.04277](https://arxiv.org/abs/2611.04277)

#### EvoMCP: Co-evolving MCP agents and training data (2026-10)
- **简介**：作者 Xuanang Gao、Kepeng Lin、Xinyan Shi 等。对 MCP agent 的 SFT 通常基于静态数据分布，无法随学生能力演化、也难以针对新暴露的弱点。**EvoMCP** 是 agent-数据互相演化框架：由 rollout 反馈评估学生已掌握的能力与缺口以确定学习目标；task evolution 在保持 verifier 对齐的前提下，调整已验证种子任务的指令与可观察环境状态，生成可执行任务；trajectory evolution 用可复用 skill 修复失败的教师 rollout 并精炼成功的 rollout；任务-轨迹对用于微调学生，被接受的任务再作为后续演化种子，更新后的学生重新评估以指导下一轮数据演化。在 Toolathlon、MCPMark、MCP-Atlas 上相对初始学生分别取得 16.5%、10.2%、12.6% 的绝对提升。
- **arXiv**：[2610.41479](https://arxiv.org/abs/2610.41479)

#### MiRL: Error-Targeted Reinforcement Learning for Correcting Terminal Agent Misbehaviors (2026-10)
- **简介**：作者 Runchu Tian、Zaid Khan、Mohit Bansal 等。已有 error-targeted 方法面向单轮任务，agent 失败分析多用于评估调试而非训练。论文提出 **MiRL**（Misbehavior-driven RL）面向终端 agent：先剖析策略错误，找出跨环境反复出现的可泛化 misbehavior，再用 RL 训练，rollout 从这些 misbehavior 发生的步骤开始，仅当通过测试且避免该 misbehavior 时才给奖励。在 Qwen3.5-4B 上，八类 misbehavior 占错误步骤的 52.1%；训练后在未见前缀上重复这些 misbehavior 的频率平均降低 43%-57%。MiRL 在 Terminal-Bench Lite 上优于 TMax-4B（51.8% 对 47.4%），在 Terminal-Bench 2.1 上持平（21.3% 对 20.6%），仅用其 35% 的 rollout。
- **arXiv**：[2610.35697](https://arxiv.org/abs/2610.35697)

#### Scaling Laws for Tool-Use Agents via Controllable Environment Synthesis (SynthAPI) (2026-10)
- **简介**：作者 Andrew Szot、Samira Abnar、Omar Attia、Abbas Kazerouni、Michael Kirchhof、Cheng-Yu Hsieh。论文研究合成工具使用环境中的数据构成如何影响 LLM tool-use agent 的训练，提出 **SynthAPI** pipeline，沿工具、任务和工具所操作数据库三个维度显式控制生成环境的多样性。通过在 458,752 个任务、四种规模共 849 个训练模型上系统改变生成控制，导出可根据合成环境的规模与构成预测下游泛化的 scaling law。结论包括：数据预算增大时，扩展工具与数据库多样性比增加任务数更有效，增加数据复杂度收益更大。按这些发现扩展数据训练的 agent，在四个评测 benchmark 上优于此前工作数据训练的模型。
- **arXiv**：[2610.37648](https://arxiv.org/abs/2610.37648)

#### CoworkEnv: Synthesizing Training Environments for Long-Horizon Cowork Agents (2026-10)
- **简介**：作者 Zhongwei Xie、Yiming Du 等（work in progress）。AI coworker 需要协调工具、跨轮沟通并在变化的工作区状态下遵守流程约束，训练经验需要连贯请求、服务记录、工具接口与评分检查。论文提出 **CoworkEnv**，一个由能力引导的可执行训练环境合成框架：以流程需求引导任务构造，通过程序化校验与反馈修复检查生成的规范和评分行为；teacher 执行产出精选示范用于 SFT，student 交互提供 rollout 与 reward 用于 RL，并训练出紧凑的 PocketWorker 模型。对 Qwen3-VL-8B-Instruct，SFT 使 Claw-Eval 平均分从 46.6 升至 64.1，其中 Multi-turn 任务提升 31.9 分；仅 SFT 的扩展在 ClawMark 上也适用。作者还比较了不同 reward 处理与 credit assignment 配置下的 RL 训练动态，训练 reward 的上升为环境可支持策略优化提供了初步证据。
- **arXiv**：[2610.24960](https://arxiv.org/abs/2610.24960)

#### BIRD-RL: Scaling Agentic Reinforcement Learning over Stateful Data-Centric Environments (2026-10)
- **简介**：作者 Ge Qu、Jinyang Li 等（NeurIPS 2026 Main Track）。现有 agentic RL 框架多以工具为中心，难以应对 SQL 调试这类有状态的数据中心环境：每个 rollout 需要独立数据库容器导致空间效率极低，而不跟踪状态地复用数据库会造成轨迹污染与跨轨迹干扰、破坏策略更新。论文提出 **BIRD-RL**，一个四阶段 agentic RL 课程，交替增强基础 agent 推理与轨迹自适应的 on-policy 探索，避免直接端到端学习的不稳定；并由 Trajectory-scoped Persistent Execution Infrastructure 支撑，在保持轨迹隔离的同时安全复用数据库副本，使空间 RL 效率提升 12.8 倍且无需 Kubernetes。在 BIRD-CRITIC-SQLite 上，7B 与 14B 模型成功率分别达 44.40% 和 48.00%，与 Claude-Opus-4.6 Agent（48.20%）相当，并可训练同时擅长 SQL 调试与生成的统一小模型。
- **arXiv**：[2610.29426](https://arxiv.org/abs/2610.29426)

#### Measure First, Then Allocate: Calibrated Rollout Budgets for MCP Agents in Reinforcement Learning (BRACE) (2026-10)
- **简介**：作者 Hieu Tran、Zonghai Yao、Sunjae Kwon、Hong Yu。训练工具使用模型交互成本高，而同一 rollout group 内 reward 完全相同时没有组相对学习信号。论文利用 reward 分歧指导 MCP 环境中的 harness 评估和 rollout 分配：先提出 **Discord**，用精确配对推断与保持决策的早停来测试 harness 修改，只有分歧的配对结果才支持某个 harness；这启发了 **BRACE**（Bank-calibrated Rollout Allocation under Correlated Episodes），结合历史任务证据、任务成功率不确定性与相关 reward 模型，在分配生成预算前预测 mixed-reward group。受控 harness 实验支持 Discord，在线 MCP 实验揭示 retention-cost 权衡；在三个 Qwen3-VL 模型规模上，BRACE 以更少的新训练 rollout 超过最强基线峰值（历史 bank 采集单独计），并无需额外训练就在 BFCL 和 NESTFUL 上领先对比方法。
- **arXiv**：[2610.28785](https://arxiv.org/abs/2610.28785)

#### RSIPO: Recursive Self-Improving Policy Optimization for Agentic Reinforcement Learning via Skill-Policy Co-Evolution (2026-10)
- **简介**：作者 Fan Zhang、Hao Wu、Jinyang Wu、Zheng Lian、Xiao Luo 等。把 RLVR 与 on-policy self-distillation（OPSD）结合的 agentic RL 常用 agentic skill 作为 privileged context 让 teacher 给 student 的 token 打分，但忽视了 skill 的 on-policy 演化以及 teacher 监督的可靠性。**RSIPO** 让 skill 与策略在训练中共同演化：策略自己从完成的轨迹中提取 skill，并通过 trajectory replay 显式优化其 skill 生成能力；同时用 empirical Bayes shrinkage 动态校准 student 对 teacher 打分的信任度。以 Qwen2.5-7B 为基座，RSIPO 在 ALFWorld 上平均成功率达 98.6%，较第二名基线和标准 GRPO 分别提升 2.5 和 17.4 个百分点，并在搜索类 agent 任务上优于所有对比方法。
- **arXiv**：[2610.15772](https://arxiv.org/abs/2610.15772)

#### Rollout-Time Intervention: Unlocking Agentic Reinforcement Learning Beyond Policy Limits (RTI) (2026-10)
- **简介**：作者 Xi Chen、Ruixue Ding、Boli Chen 等（Alibaba-NLP）。agent RL 受 on-policy 推理瓶颈限制：任务超出模型当前能力边界时有效轨迹极度稀疏，训练常常崩溃。**RTI（Rollout-Time Intervention）** 在 rollout 期间利用奖励侧证据（评估 rubric）把错误轨迹转化为成功轨迹，为策略很少自行解决的任务提供有效学习信号。干预轨迹本质上是 off-policy 的，但作者发现过滤掉分布偏移过大的轨迹后，它们能有效引导策略优化。在五个基准上 RTI 一致优于 RL 基线，使后训练策略接近明显更强的同期 LLM，并突破自身原有的能力边界（abstract 未给出具体数值）。
- **arXiv**：[2610.13438](https://arxiv.org/abs/2610.13438)

#### Co-Evolving Acting and Forecasting for Language Agents (CoPE) (2026-10)
- **简介**：作者 Haofei Yu、Yixing Chen、Yining Zhao、Xuyan Ye、Yuxiang Liu、Zhigang Hua 等。标准 agentic RL 只从终端标量奖励学习，中间步骤的监督基本未被利用；已有的 value target 来自同一单一标量，observation target（网页、终端日志）又长又嘈杂且只能前瞻一步。**CoPE** 引入第三类多步动作目标：在成功的 on-policy 轨迹上，训练策略骨干生成每段历史之后的动作序列；每个动作也由其更早历史预测并被监督，形成被共享骨干吸收的时间集成，使 acting 与 forecasting 共同演化；推理时仍逐步行动，无额外开销。在 ALFWorld、ScienceWorld、WebShop、tau^2-bench 和 AppWorld 五个基准上，聚合表现强于 GRPO 及带预测性辅助目标的 RL 基线；ScienceWorld 上样本效率相对 ECHO 和 GRPO 分别最高提升 1.9 与 2.0 倍，且仅需单一策略骨干。
- **arXiv**：[2610.20597](https://arxiv.org/abs/2610.20597)

#### SkillScaffold: Scaffolding Agentic Reinforcement Learning with Policy-Adaptive Skill Credit Assignment and Selective Injection (2026-10)
- **简介**：作者 Wenlin Zhong、Hao Wang、Kun Yuan、Fan Yang、Jinluan Yang、Dian Jin。文本 skill 把过往交互提炼为可复用指导以改善探索，但已有 skill-based RL 常假设注入总是有益，或依赖间接反馈、需大量 rollout 的评估。预实验表明基于 rollout 的效用估计既昂贵又在有限采样下不稳定，且 skill 收益因候选和策略当前能力而异。**SkillScaffold** 将低成本 skill 效用估计与选择性注入结合：复用结果已验证的训练轨迹生成候选 skill，通过 outcome-aware 似然对比估计效用，无需针对候选的环境 rollout；效用决定是否、注入哪个 skill，没有候选被估计为有益时保持无 skill 执行；Generator 与 Executor 联合训练，并鼓励 skill 内化以实现无 skill 推理。在 ALFWorld、WebShop、ScienceWorld 和 tau^2-Bench 上，对短/长程任务均优于强 agentic-RL 与 skill-based RL 基线（abstract 未给出具体数值）。
- **arXiv**：[2610.18807](https://arxiv.org/abs/2610.18807)

#### Structuring MoE Expert Selection for Agentic Reinforcement Learning (2026-10)
- **简介**：作者 Bolian Li、Ting-Yao Hu、Cheng-Yu Hsieh、Sanjoy Chowdhury、Oncel Tuzel、Raviteja Vemulapalli。长程 LLM agent 常基于稀疏 MoE 实现，但 agent 行为与 MoE 结构的协同设计少有研究。论文发现现成 MoE 的专家选择呈现与 agent 轨迹天然对齐的专门化结构：语义相似操作（如 READ、UPDATE）的 turn 之间专家路由重叠更多；而标准 RL 忽略该结构，路由在训练中不受控，限制任务表现和推理效率。作者提出面向 agent 任务的层级路由控制框架：显式鼓励 turn 级专家选择与 agent 操作对齐，同时正则 token 级专家选择以保持局部一致，并引入 entropy-gated 控制机制保证稳定。该框架在所有评测 benchmark 上成功率提升超过 10 个点。
- **arXiv**：[2610.07332](https://arxiv.org/abs/2610.07332)

#### Hierarchical Reinforcement Learning with Stable Temporal Abstraction for Language Model Agents (STAC) (2026-10)
- **简介**：作者 Shayan Mohajer Hamidi、Yize Cheng、Yuanda Xu、Zhengze Zhou、Alborz Geramifard。层级 RL 通过围绕持久子目标组织原子动作并在多时间尺度分配信用来改进长程控制，近期层级语言 agent 显式分离子目标规划与动作执行。作者观察到显式层级并不保证时间抽象稳定：学到的边界策略可能几乎每轮都替换子目标（子目标变得短暂），或在子目标已不合适后仍保留，称为 temporal abstraction instability。**STAC**（Stable Temporal Abstraction via Constrained Optimization）是约束式边界策略优化方法，把过早重规划和陈旧持续表示为约束代价，仅对采样的边界决策施加拉格朗日代价，而不改变底层算法的奖励、critic 目标、子目标 advantage 与原子动作 advantage。Qwen3-0.6B 上 ALFWorld / WebShop 成功率比强层级基线高 8.1 / 7.9 点，Llama-3.2-1B-Instruct 上高 23.5 / 15.8 点。
- **arXiv**：[2610.05473](https://arxiv.org/abs/2610.05473)

#### Training Advisors for LLM Agents from Task Outcomes (Caddie) (2026-10)
- **简介**：作者 Sergei Polezhaev、Barys Liskavets、Ori Press、Alexander Golubev。已有工作表明自然语言反馈能帮助 agent 在任务执行中修订决策。**Caddie** 训练 critic 在 agent 执行任务时提供自然语言分析与建议，不依赖步骤级标签或参考批评，而是从 agent 在收到 critic 反馈后最终是否成功中学习；在冻结基座模型的情况下通过 RL 优化 critic。在单个基座模型的多跳问答上训练，Qwen3-4B critic 能提升四个不同规模和架构基座模型的成功率（含三个训练中未使用的）。在 MuSiQue 上使 Qwen3-4B 成功率提升超过 25 个百分点，超过无 critic 的 Kimi K3；同一 critic 在 tau^3 和 DeepDive 等域外交互 benchmark 上无需额外训练也有增益。
- **arXiv**：[2610.09858](https://arxiv.org/abs/2610.09858)

#### WEFT: Scaling Tool-Use Post-Training for General-Purpose Agents (2026-09)
- **简介**：作者 Bo Mao、Hang He、Linting Wang 等指出，近期扩展工具使用后训练的工作多集中在合成可执行环境，但这只是由环境、任务、agent harness 与评估器构成的 agentic 交互系统的一部分，孤立扩展环境不保证相应收益。提出 **WEFT**（Whole-system Evolution For Tool-use Post-training），把可扩展的交互系统构建、执行驱动的自演化与稳定后训练结合：在环境广度、任务复杂度、交互多样性上扩展；用执行轨迹与状态证据归因失败并修订相关组件；以 prefix-preserving sampling 保留已验证进展、atomic-turn credit assignment 定位学习信号，并用 MegaMCP 在共享工具服务上为并发 rollout 维护隔离、可恢复的状态。WEFT-8B/14B 在 BFCL V4、τ²-Bench、Claw-Eval 上优于所有同规模环境扩展基线，WEFT-14B 相对 Agent-World-14B 分别提升 6.41、2.23、12.27 个百分点；WEFT-35B-A3B 进一步延伸到 Toolathlon-Verified 和 AutomationBench。
- **arXiv**：[2609.36887](https://arxiv.org/abs/2609.36887)

#### Luck Is Not Skill: When Do Paired Rollouts Help Group-Relative RL of LLM Agents? (2026-09)
- **简介**：作者 Nazmus Sakib。组相对 RL 比较同一 prompt 的 rollout，但独立的环境噪声会掩盖比较信号。论文研究 paired rollouts：组内共享按事件键控的噪声调度、保持各 rollout 边缘分布不变，并推导出单侧 grader 噪声下梯度方差降低的精确条件，同时给出奖励对比变好但梯度方差反而增加的反例。在 2B tool-use agent、tool 故障与 grader 翻转下的预注册对照实验中，tool 故障下配对使最终带噪测试成功率平均提升 5.1 个百分点，但未达到预注册的学习曲线判据；梯度探针显示协方差迹降低 21%–30%（grader 翻转）与 40%–63%（tool 故障）。结论是区分「改善奖励对比」「降低估计方差」「提升学习速度」三者。
- **arXiv**：[2609.24144](https://arxiv.org/abs/2609.24144)

#### ACLArena: Agent Continue Learning in Multi-stage Post-training (2026-09)
- **简介**：作者 Haixin Wang、Xiaoxuan Wang、Junkai Zhang 等。通用 agent 的多种能力通常在不同训练阶段习得，但 Agent Continual Learning (ACL) 缺乏公认配方。**ACLArena** 搭建顺序训练流水线，从模型层和 token 层两个角度分析遗忘与泛化机制，并系统比较 multi-teacher on-policy distillation、self-distilled fine-tuning 与 model merging 在恢复旧能力与保留新能力上的表现，最后提出结合高质量轨迹离线 replay 与多个经 RL 专门化的 LoRA 专家路由网络的新 ACL 配方。在四个推理与 agentic 任务的 in-domain 与 out-of-domain 评测上验证了分析与方法有效（abstract 未给出具体数值）。
- **arXiv**：[2609.23989](https://arxiv.org/abs/2609.23989)

#### Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms (VHD-Play) (2026-09)
- **简介**：作者 Xinjie Shen、Wei Fan、Xudong Guo 等（Qwen 技术报告）。现有环境生成流水线常先构建环境再定义结果规则或标注轨迹，使动力学与评估事后对齐。**VHD-Play** 反转这一依赖：先采样并求解一个数学模型，再由基于语料的 setter 把其决策过程渲染为有状态工具，可执行动力学与轨迹打分参考均继承自同一个已解模型，每个环境成本仅几美分，共生成 3,300 个多样 agentic 环境。在三个族上训练 Qwen3.6-35B-A3B，五族诊断的平均 agentic 分数从 0.204 提升到 0.815，并泛化到留出实例、八个未见机制族，以及函数调用、旅行规划、365 天电商等外部基准；对比显示大部分可学习差距在有状态交互而非底层问题求解。
- **arXiv**：[2609.27321](https://arxiv.org/abs/2609.27321)

#### AutoGym: Blueprint-First Generation of Verifiable Agent Gyms (2026-09)
- **简介**：作者 Aarati Andrea Noronha、Kavya Ravikumar、Carly Xiaoyu Lin（NeurIPS 2026 Workshop 投稿）。用 RL 训练 agent 需要由任务、可执行环境和可靠 verifier 构成的 gym，人工构建昂贵且静态，单次合成的任务难度只是表面的，正确性还得靠不可靠的 LLM judge 事后裁定。**AutoGym** 从最小的领域种子或已有模型轨迹生成完整 gym，包含三个机制：blueprint-first 生成——在实例化环境前先规定有效解空间、环境需求和验证标准，使可解性成为构造前提；显式生成参数——控制任务拓扑、交互深度、能力轴、问题混淆和干扰项以精细调节难度；主动课程合成——基于性能校准随模型能力演进调整参数分布。在生产力与时间推理场景中生成了覆盖能力谱的 gym，包括能挑战前沿模型的实例。
- **arXiv**：[2609.22592](https://arxiv.org/abs/2609.22592)

#### Marginally Correct Tool Caches Can Reverse Group-Normalized Policy Updates (2026-09)
- **简介**：作者 Shivam Gupta。agent 训练中缓存工具结果可减少重复执行，但会耦合 rollout 的随机性。论文研究一个二动作模型：独立执行与共享执行都保持每条 rollout 的条件奖励分布，但组内共享一个随机结果会反转期望的组归一化策略更新。推导出精确的有限组表达式：对常数替代动作，共享更新跟随的是胜率减败率，而非期望奖励之差；Bernoulli 特例给出方向错误区域，并且随组大小增长存在不会消失的更新方差下限。不做组标准差缩放的中心化在该模型中保持期望回报方向。对 540 种配置和 3,240 次估计器评估做了穷举验证，并在固定版本的 TVCache 栈上用 256 条脚本化 rollout 复现共享路径。作者说明这不衡量语言模型训练性能，也不反驳 TVCache 的确定性输出契约，只表明仅靠边缘输出有效不足以保证随机缓存与训练等价。
- **arXiv**：[2609.26866](https://arxiv.org/abs/2609.26866)

#### ToolCompass: Guiding Tool Trialing, Not Suppressing It (2026-09)
- **简介**：作者 Junlin Fang、Chong Zhang、Do Nguyen-Thanh 等。LLM agent 需要从训练时见过的工具泛化到部署时未见的工具，关键难点是工具试探：过多试探浪费交互预算，选择性试探才能探索陌生工具。仅看结果的后训练对浪费性试探无引导，turn 级监督又可能抑制必要探索。**ToolCompass** 是后训练框架，按共同功能组织 tool-call 表示：把每个功能类建模为 von Mises-Fisher 分布，联合降低跨领域的功能内差异、增大功能间分离，从而把已见工具的经验迁移到功能相似的未见工具，引导探索远离无关选项；无需真值调用轨迹或访问未见工具，也无推理开销。在 AppWorld 与 FTRL 上，对 GRPO、RFT、DMPO 均有稳定增益，AppWorld OOD 任务成功率相比原始后训练最高提升 10.71 个百分点。
- **arXiv**：[2609.25678](https://arxiv.org/abs/2609.25678)

#### SFT or RL for Tool-Calling Agents? A Controlled Study Across Data, Method, and Scale (2026-09)
- **简介**：作者 Md Tahmid Rahman Laskar、Xue-Yong Fu、Shashi Bhushan TN（EMNLP 2026 REALM Workshop）。针对训练数据、适配方法、模型规模如何共同影响工具调用能力缺少受控证据的问题，作者在 0.6B–32B 的六个 Qwen3 模型上比较 LoRA-SFT、GRPO、以及 SFT 后接 GRPO。结果：SFT（LoRA）在分布内表现最强，在 18 个设置中 15 个最优；跨数据集迁移上差距较小，GRPO 在 54 个设置中赢 29 个，但平均优势不足 1 个点，SFT→GRPO 很少是最强；数据混合在保持接近专用训练的同时带来稳定的迁移；LoRA 优于全参数微调，更好地保留了预训练的 agentic 行为。
- **arXiv**：[2609.17848](https://arxiv.org/abs/2609.17848)

#### Salesforce Koa: An Enterprise Language Model for Agentic Tool Use (2026-09)
- **简介**：作者 Zixiang Chen、Sufeng Niu 等（Salesforce）。Koa 是在开源 Nemotron-3-Super-120B 上用 GRPO 做 RL 后训练的企业级模型，以 FP8 部署，仅使用公开与合成数据，专精企业工作流所需的工具调用：路由请求、以合法参数调用正确工具、完成多轮业务任务。核心是「规范驱动的任务构建」：把声明式 Agent Script 规范展开成带 persona 的多轮环境，reward 以成功的工具使用为基础。在 CRMAgentBench 上 Task Success Rate 达 87%（GPT-4.1 为 82%，基座为 79%），在人工标注的生产工具调用集上也处于或接近各指标最高，并保持了 Tau2Bench、BFCL 上的通用能力；受控对比显示额外的领域内 RL 阶段提升了参数准确率与完整 tool-call 成功率。
- **arXiv**：[2609.15066](https://arxiv.org/abs/2609.15066)

#### Spurious Tool Use: When RL Agents Learn the Wrong Reason to Act (2026-09)
- **简介**：作者 Yiwei Yang、Haoxiang Zhang、Bingbing Wen 等。RL 优化的工具使用策略可能放大训练数据中的伪相关。作者在事实问答与数学推理的受控合成环境中注入与特定工具强相关、但与工具必要性因果无关的 cue，研究 RL agent 何时、为何学到基于 prompt 表层线索而非任务真实需求来调用工具的捷径策略。在 cue 存在但工具并不必要的反事实评测中，虚假工具调用率最多上升 39%；捷径只在 agent 已可靠掌握目标工具时才出现，说明任务能力而非仅数据不均衡是关键因素，cue 与工具的语义对齐会显著放大效应。为缓解该问题，提出稠密的决策级 reward——由 LLM judge 评估每次工具调用是否必要——有效抑制由 cue 驱动的工具调用并保持任务表现。
- **arXiv**：[2609.16268](https://arxiv.org/abs/2609.16268)

#### Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL (ActObs) (2026-09)
- **简介**：作者 Juzheng Zhang、Disha Makhija、Manoj Ghuhan Arivazhagan 等。标准 agent SFT 只对 agent 自己的 action token 计算 loss，环境 observation 仅作上下文。**ActObs** 额外监督轨迹中已有的 observation token，让策略学会预测动作后果，不增加数据、参数、序列 token 或前向次数。两种方法 SFT 后表现相近，但 GRPO 之后出现分化：Qwen3-4B 上 ActObs 在 Terminal-Bench 2.0 的每个采样预算下 pass@k 都更高；Qwen3-8B 上牺牲部分 pass@1 但 pass@16 提高 3.4 个点，并在未见过的 aider-polyglot 上 pass@1 提高 4.2 个点（4B）。分析表明 ActObs 在 RL 中保留更多熵、策略移动更小；根源在 SFT：action 与 observation 梯度迅速正交，仅 action 训练会让环境预测能力低于基座。
- **arXiv**：[2609.20715](https://arxiv.org/abs/2609.20715)

#### HarnessBandit: Joint Learnability-Transferability Scheduling for Multi-Harness Agentic Reinforcement Learning (2026-09)
- **简介**：作者 Hongliang Wei、Xiaobing Tu 等。LLM agent 通过 system prompt、工具 schema、控制循环、轨迹格式各异的 harness 部署，同一模型在不同 harness 下表现不均。用多个 harness 训练共享策略时需要调度：每个训练步既要选当前提供有用学习信号的 harness，又要使更新惠及其他 harness。**HarnessBandit** 是在线调度器，每个优化步选一个 harness；GRPO 更新后观测 learnability（batch 上平均绝对 advantage）与 transferability（当前 harness 低维梯度 sketch 与其余 harness 梯度滑动平均的余弦），经归一化融合后带访问奖励与探索下限采样。用 Qwen3.5-2B 在 ClawGym 的六个 harness 上训练，在 PinchBench 与 ClawEval 上均优于混合 batch 的多 harness 训练。
- **arXiv**：[2609.13739](https://arxiv.org/abs/2609.13739)

#### MATCH: Model-Aware Tool Learning with Curriculum Scheduling and Hierarchically Gated Rewards (2026-09)
- **简介**：作者 Shihao Liu、Hao Yin、Lijun Liu 等。工具学习 RL 存在两个问题：固定阈值课程会与策略不断变化的能力边界错位；加性 reward 在预测工具错误时仍会泄漏参数级 credit。**MATCH** 是闭环框架：Model-Aware Curriculum Learning（MACL）维护随策略共同演化的、由 reward 导出的样本难度，每个 epoch 选取靠近当前能力边界的样本以及 top-k 更难的样本；Hierarchical Tool-call Gated Reward（HTGR）把工具名、参数 key、参数 value 作为门控链打分，仅当前置条件成立才给下一级 credit。同一 HTGR reward 既驱动 GRPO 更新又刷新 MACL 难度。在 API-Bank 与 BFCL V3 上整体准确率分别达 72.19% 与 62.87%，优于主要的监督与 RL baseline，且在两个模型家族的四个 backbone 上都有一致提升。
- **arXiv**：[2609.20082](https://arxiv.org/abs/2609.20082)

#### SiLR: Structure-Preserving Admission and Process Reward for LLM Tool Agents (2026-09)
- **简介**：Chenyu Zhou、Qiliang Jiang、Shuning Wu、Xu Zhou。针对 ReAct 式工具 agent 的运行时准入门（gate）通常被当作标量过滤器的问题，作者指出被拒提案后同一状态会继续产生新提案，因此 gate 实为提案流上的搜索算子，会决定哪些 trajectory 可达；聚合分数式 gate 会落入「标量投影陷阱」，接受局部改善的提案而把 trajectory 锁死在平台期。**SiLR** 对每个提案先做影子执行（shadow execution），再用 branch 级违规状态（过载支路支撑集 + 每支路严重度）上的乘积序（product order）判断准入，并证明不存在对该序 sound 的标量代理，故失效是表示层面的而非阈值调参问题。在 Gym-ANM 挖掘场景上 SiLR 恢复 21/21 个多动作 episode，而终端式 gate 为 0/21、最佳标量 gate 为 9/21；两约束族同时激活时 support-only 准入了 42,410 例中 63.2% 的物理不安全动作而乘积序为 0；该准则复用为 GRPO 的 process reward 后在全部场景优于其计数投影，且是唯一使无 gate policy 超过未训练基座的 reward（0.844 vs. 0.778）。
- **arXiv**：[2609.04629](https://arxiv.org/abs/2609.04629)

#### Why Sample What You Can Enumerate? Exact Policy Optimization for Genomic Tool Selection (FGPO) (2026-09)
- **简介**：Haoyue Liu、Xiaoyu Ma、Ye Chen、Zhichao Wang 等。在专科科学场景中工具子集空间虽为组合空间但可完全枚举，作者指出「冻结 reasoner 上跑 GRPO」的常规配方在此结构性失配：GRPO 仅用少量 rollout 估计动作期望，且训练越成功越退化——policy 集中到偏好子集后重复采样、reward 相撞，group 归一化 advantage 归零，基因组推理中无 reward 信号的问题比例从均匀参考 policy 的 0.2% 升到 GRPO 训练后的 20.8%。**FGPO**（Full-Group Policy Optimization）为每个工具子集打分并直接优化精确动作期望，使每次更新看到完整动作空间，同时把「问题–子集」reward 预计算成穷举表，彻底移除训练回路中的冻结 reasoner 调用。跨 5 个冻结 reasoner 与 3 个基因组基准的 15 个设置中 FGPO 全面超过 GRPO，平均 +6.75 点、最高 +14.20；标准按需 GRPO 调度需要 2.4 倍的 reward 评估次数，且在 GenomeQA 上 FGPO 把每题调用工具数从 2.36 降到 1.40。
- **arXiv**：[2609.10221](https://arxiv.org/abs/2609.10221)

#### TRACE: Training Reasoning Agents for Causal Exploration with Synthesized Rewards (2026-09)
- **简介**：Rui Sun、Zhan Shi、Bing He。RLVR 在数学与代码上奏效依赖答案易于校验，而复杂数据上的诊断推理缺少这一条件——确认异常真因往往需要昂贵的专家调查且事后仍可能含糊。作者提出「工程化地制造可验证性」：先采样一个干预、注入受控模拟器、生成该干预会产生的观测，隐藏的干预既提供 oracle 标签与客观 reward，agent 仍须在有噪声、被混淆且分散的证据中调查。**TRACE** 将其实例化为含 12 种根因与细粒度分段归因的数字广告诊断环境，agent 用 Python 与 SQL 调查并须同时给出根因与受影响分段。在 235 个 held-out episode 上，最强 prompt 基线 Claude Opus 5 为 0.686 FullAttr@1；SFT 把 Qwen3.5-35B-A3B 从 0.159 提升到 0.637，随后用合成 reward 做 RL 达 0.757，超过所有 prompt 基线（含前沿闭源模型与 prompt 的 Qwen3.5-122B-A10B），且所得 policy 的工具调用次数显著少于 prompt 的 35B 基座。
- **arXiv**：[2609.10315](https://arxiv.org/abs/2609.10315)

#### Learning to Use Tools: Reinforcement Learning for Tool-Integrated Mathematical Reasoning (2026-08)
- **简介**：Minghui Xu、Zi Wang。针对 LLM 数学推理中计算错误占错误回答相当比例的问题，作者先构造 SFT 数据教模型计算器工具的调用模式与返回值解读，再在这个 tool-formatted policy 之上用 RLOO、RLOO++、GRPO、DAPO 等 on-policy RL 方法、以可自动验证的最终答案 reward 训练，并另建 1,024 题与训练集无精确重叠的 Countdown held-out 基准以保证评测可靠。结果显示计算器工具集成让 SFT 与 RL 基线的 pass@k 普遍提升约 10 个百分点，其中 Tool-DAPO 最强，把 pass@1 从 Tool-SFT 的 35.8% 提升到 66.0%；进一步分析表明即使只给最终答案 reward，RL 也会促使模型更有效地使用工具。
- **arXiv**：[2608.28447](https://arxiv.org/abs/2608.28447)

#### CAST: Critique-Aware Supervision for Training Reliable Long-Horizon Tool-Calling Agents (2026-08)
- **简介**：Arizona State University 与 Cisco Research（Amir Saeidi、Zehua Zhang、Rishitosh Singh、Vivek Gupta、Chitta Baral 等）。在长 horizon 有状态环境中，单个错误动作（如退错款）即造成不可逆失败、必须在执行前拦截，但现有工作多依赖 prompt 式 critique agent，优化类方法又缺少系统地产出可训练的丰富验证理由的手段。**CAST** 把稀疏任务结果转化为 action 级监督：先分析 agent trajectory，合成结构化 rationale 解释部分可观测与领域策略下每个动作是否有效，据此训练 critique 模型，再用该模型构造 critique-aware 训练数据来优化 policy 模型。在动态工具调用基准上微调 Qwen3 系列后，Retail 任务的 pass^4 超过 GPT-OSS-120B 逾 10%，域外 Telehealth 设置再获约 9% 提升。
- **arXiv**：[2608.30147](https://arxiv.org/abs/2608.30147)

#### MCP-Universe RL: A Framework for Training MCP Tool-Use Agents via Reinforcement Learning (2026-08)
- **简介**：Salesforce AI Research（Ziyang Luo, Yan Yang, Silvio Savarese, Junnan Li 等）。指出现有 agentic RL 框架大多止步于 policy update，把两个系统难题留给用户：为数百条并发 trajectory 各自拉起隔离环境并接入训练，以及在长多轮 episode 大量时间卡在慢工具调用时保持 GPU 利用率。**MCP-U RL** 以 MCP 作为环境接口，任何已暴露为 MCP server 的工具无需 RL 专用集成代码即可接入训练，并一次性补齐两层：环境编排层在可插拔容器后端上完成 MCP 环境的分配、隔离与回收，rollout 编排层用分阶段流水线让多条 trajectory 重叠执行，使 GPU 在 episode 等待工具时仍保持忙碌；训练层后端无关，已集成 veRL 与 slime。仅改任务规格、用同一份配置即在 gpt-oss-20b 上分别训练软件工程、深度研究与通用工具使用三类 agent，三者 task reward 均提升（abstract 未给出具体数值）。
- **arXiv**：[2608.22167](https://arxiv.org/abs/2608.22167)

#### Joint Optimization of Tool Creation and Use for Large Language Model Agents (SMITH) (2026-08)
- **简介**：台湾大学（Zhi Rui Tam, Chieh-Yen Lin, Yun-Nung Chen, Hung-yi Lee 等）。现有工具创建系统只在推理期 prompt 一个冻结 LLM，写工具的模型与用工具的模型彼此解耦，缺少「自己产出的 schema 自己调得动」这一信号。**SMITH**（Schema-grounded Multi-task Iterative Tool Honing）在单一 policy 内联合训练工具创建与工具使用：每条 rollout 要么是 build 任务（据少量示例写出工具），要么是 use 任务（在留出问题上调用工具池中的工具），并用 schema、代码、结果三条独立 reward 轴分别捕获三类失败模式，使每种失败贡献各自的梯度。Qwen3-4B 在 13 个带精确 verifier 的过程推理任务上训练后，留出任务宏平均准确率达 79.8，为所有对比方法最优并超过未训练的 30B-A3B 工具编写者；TabMWP-Hard 40.4、域外 GQA 42.6（较同 backbone 最佳推理期基线 +7.6），且完全未使用视觉或表格训练数据；其 4B 模型写出的工具还能提升 LFM-2.5-350M 与 Qwen3-30B-A3B 的表现。
- **arXiv**：[2608.24571](https://arxiv.org/abs/2608.24571)

#### MidTool: Mid-training Data Synthesis for Agentic Tool Use (2026-08)
- **简介**：Fengqing Jiang, Yite Wang, Boyi Liu, Zhaoyang Wang 等。已有工作证明定向 mid-training 能强化数学、科学等推理能力并改善软件工程场景的 agentic 能力，本文研究与之平行但较少被探索的通用 tool use。**MidTool** 是面向 agentic tool-use mid-training 的开放语料构建 pipeline，把大规模 web、PDF、code 数据与来自真实 tool API、MCP skills、文档驱动 workflow 的合成监督结合，专门训练模型识别 tool affordance、从上下文 ground 参数、组合 tool call workflow 以及在信息不完整时恢复。作者在 MidTool-Mix 上 mid-train Qwen3-4B-Base 与 Qwen3-8B-Base，再分别做 SFT 与 RL 后训练，在 BFCL、tau2-Bench 与 MCP Universe 上两种后训练路径下均一致优于 baseline（abstract 未给出具体数值），说明通用 tool use 同样受益于专门的 mid-training 而不应完全交给后训练。
- **arXiv**：[2608.20314](https://arxiv.org/abs/2608.20314)

#### Retry, Switch, or Abstain? Learning Strategy-Aware Tool-Use Policies via Controlled Error Injection (BENCH2ROBUST) (2026-08)
- **简介**：Chaoran Chen、Vy Nguyen、Ziji Zhang、Dakuo Wang 等。针对工具智能体几乎只在「工具调用必定成功」的环境中训练与评测、而真实部署中工具会瞬时失败、持续失败甚至静默失败的问题，作者提出 **BENCH2ROBUST**，把无失败的工具使用 benchmark 转换成可控随机环境并按可解性分场景，强制 episode 必须做出「重试同一路径 / 切换替代路径 / 在所有路径耗尽后停止」的策略选择。在该环境上比较两类互补干预：运行时结构化恢复上下文 Bayesian Tool Memory（BTM）与 curriculum-controlled RL。覆盖 4 个模型家族的 7 个模型、两类多轮 benchmark 的实验显示工具失败带来近乎普遍的鲁棒性差距；held-out Retail 任务上 BTM 无需重训即把鲁棒性提升最多 16.8 个百分点，RL 学到的恢复行为在推理期不挂 BTM 时依然有效，两者结合在失败注入下达到 40.8–45.5%，同时保持无失败场景下的原有性能。
- **arXiv**：[2608.11977](https://arxiv.org/abs/2608.11977)

#### When the API Speaks the Wrong Language: Revisiting Post-Training for Multilingual Tool Use (2026-08)
- **简介**：Siddharth Chauhan、Thomas Butler、Abhishek Singhania、Honey Gupta 等。论文聚焦多语言场景下 LLM API 调用可靠性下降的一种具体失效模式：模型选对了工具，但把参数值写成了不一致的语言，作者称之为 Argument Language Mismatch（ALM）——语义上正确但操作上无效，且被标准 API 调用指标完全漏掉。作者重新审视缓解 ALM 的 post-training 方案，发现在其 benchmark 上 SFT 就是很强的基线，能大幅改善参数语言一致性与端到端函数调用准确率，在模型选择一致的前提下与更复杂的 RL 方法相当甚至更好。进一步引入带结构化、参数感知 reward 的 GRPO 后，语言一致性提升且通用推理能力保持得更好，但增益是增量式的，主要体现在泛化与多目标权衡上，说明多语言 API grounding 的大部分性能可由精心设计的监督训练获得，RL 提供的是定向而非根本性的改进（abstract 未给出具体数值）。
- **arXiv**：[2608.11715](https://arxiv.org/abs/2608.11715)

#### Agentic Router: An Execution-Grounded Continual Learning Approach With Memory (2026-08)
- **简介**：Yuxuan Chen、Rongpeng Li、Zhifeng Zhao、Honggang Zhang 等。面向命令行网络运维（SONiC）场景，问题在于一条「看起来合理」的命令执行后仍可能失败或引入操作风险，而已有工作只关注命令生成或最终配置正确性，没有用执行落地的经验同时提升候选覆盖与动作选择质量。作者提出 execution-grounded 的双路径、后果感知 agent：先生成多个完整候选动作、预测各自的执行后果，再用效用与风险感知的重排序选出最终动作；提案侧路径把可复用的运维教训抽象成可检索指引，在不改动 proposal LLM 的前提下提升可行动作覆盖率，选择侧路径则用真实 SSH 反馈做 session 级 LoRA 更新以改进条件化选择质量。在多轮 SONiC 运维会话、多个不同 Qwen3 proposal 模型上，框架同时提升可行动作覆盖与 top-1 执行成功率，且两条适配路径随交互推进呈互补增益（abstract 未给出具体数值）。
- **arXiv**：[2608.09184](https://arxiv.org/abs/2608.09184)

#### ToolLIFT: Lifting Tool-Specific Trajectories into Function-Level Graphs for Generalizable Tool Planning (ToolLIFT) (2026-08)
- **简介**：Xiuhui You、Jiayi Luo、Qingyun Sun 等。指出现有做法直接从历史工具使用 trajectory 建 tool-level 图，图与具体工具绑定、难迁移到新工具集；作者发现同类任务常共享 function-level 工作流结构，是更可迁移的抽象。**ToolLIFT** 先用 trajectory-lifting 把工具专用轨迹抬升为 function-level workflow graph（FWG），跨工具共享协作经验；再基于 FWG 全局结构解耦工作流规划与工具选择，使单步选择对齐整体工作流；最后用 RL 引入 source-gated 与 skill-specific reward 保证跨工具调用的信息流可溯源。在 2 个 in-distribution 与 3 个 out-of-distribution 基准上一致优于 SOTA 基线，对未见工具集泛化良好。
- **arXiv**：[2608.03468](https://arxiv.org/abs/2608.03468)

#### SkillRise: Agentic Reinforcement Learning for Cross-Task Skill Evolution (SkillRise) (2026-07)
- **简介**：来自浙江大学 / 美团（Zhiyuan Yao、Yongliang Shen、Weiwen Liu、Xunliang Cai 等 16 人）。标准 agentic RL 把任务当作独立 episode，已有 skill 学习要么聚焦同一任务的重复尝试，要么用多阶段管线纠缠抽取/检索/执行。**SkillRise** 是统一的跨任务 skill 学习 RL 框架：把相关实例组织成难度递进序列，用单一策略在"解题"与"整理一份不断演化、直接传给下一任务的 skill 文档"之间交替；**跨任务解耦信用分配**——解题用当前任务结果监督，整理用折扣的下游结果监督。ALFWorld、WebShop、ScienceWorld 上取得最强 Pass@1（较最强基线 +2.3~8.5 个百分点），并展现"跨任务测试时扩展"：相关任务序列越长（即使每个只试一次）性能越好，说明其复用可迁移 skill 而非重复采样同一任务，同时大幅降低多阶段管线运行开销。
- **arXiv**：[2607.26784](https://arxiv.org/abs/2607.26784)

#### Towards Robust Reinforcement Learning for Small-Scale Language Model Agents (2026-07)
- **简介**：来自滑铁卢大学 / MBZUAI（Md Rezwanul Haque、Md. Milon Islam、Fakhri Karray）。系统排查了 70–500M 参数小语言模型（SLM）用 RL 对齐时"不稳定"的失败机理：在 Pythia-70M/160M/410M 与 SmolLM2-135M/360M × TinyStories/CNNDM/Wikitext-103 共 15 个 (模型, 语料) 上用 PPO 训练，识别出三种可复现失败模式——标准 PEFT/TRL 管线中 LoRA 参数"静默冻结"、bf16 下重要性比数值溢出、奖励模型误差导致灾难性策略崩溃；分别用 merge-and-reinitialize adapter、PPO 更新用 fp32、以及 reward whitening + 重要性比守护 + 权重回滚三层安全机制修复。提出"容量余量假设"：SLM 规模下 PPO 表现取决于流畅的 SFT 先验（PPL<20）与判别性奖励信号，而非参数量。所有配置稳定收敛并优于指令微调基线且用更少数据。
- **arXiv**：[2607.25091](https://arxiv.org/abs/2607.25091)

#### PATS: Policy-Aware Training Scaffolding for Agentic Reinforcement Learning（PATS） (2026-07)
- **简介**：Yipeng Shi、Zhipeng Ma、Yue Wang 等（含北大等）提出的"策略中心"训练范式，把可复用 skill 重新定位为**动态训练脚手架**而非目标本身。PATS 将最新策略的 rollout 组转成 evidence cards，用任务特定评估调整后续 rollout 的上下文——弱策略靠具体指导完成难任务，随策略变强逐步删减冗余指导；策略仍用标准 RLVR＋环境奖励优化，脚手架在部署时丢弃。在 ALFWorld/WebShop 上较强基线最高提升 18.6%，在 7 个 search-augmented QA 基准上以少 32.1% 的 prompt token 保持竞争力。
- **arXiv**：[2607.21419](https://arxiv.org/abs/2607.21419)

#### MOF-Sleuth: Tool-Grounded Reward Alignment for Explainable Fine-Grained MOF CIF Auditing（MOF-Sleuth） (2026-07)
- **简介**：Yu Liu、Zhiwei Yang、Chaozhuo Li 等（中科院信工所等）提出的强化学习引导 CIF 审计智能体，用于金属有机框架（MOF）晶体信息文件的细粒度错误诊断。框架含确定性 Forensic Lab（推导成分/几何/连接/占据/配位/电荷等证据）与 Sleuth 推理引擎；**Reward-guided RL 把工具测量转成化学解释级监督**，不仅奖励最终答案，还奖励所引用的化学证据与证据支撑的诊断，并提出 Chemically Grounded Diagnosis（Chem-GD）度量。四个基准上在 LLM 方法与 MOF 专用 ML 方法中取得 SOTA。
- **arXiv**：[2607.19935](https://arxiv.org/abs/2607.19935)

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

#### Don't Guess, Ask Wisely: Benchmarking and Training Proactive Elicitation in Open-Domain LLM Conversations (DialogGRPO) (2026-11)
- **简介**：作者 Rongyu Zhang、Mutian Bao、Jinjian Zhang 等，EMNLP 2026 oral。针对用户 query 不完整时 LLM 应先提问澄清的「proactive elicitation」能力：在 7 个 LLM 上评估发现，不足 11% 的欠指定 query 会触发追问。论文提出 **VagueBench**（对种子 query 标注结构化意图与细节单元，再受控去除语言线索得到意图模糊与完全模糊变体，从而有已知的隐藏信息集）及 **DialogGRPO**：一种离线 RL 方法，把多轮追问组织为对话状态 DAG，每个节点上的候选问题通过 group-relative 比较获得结合即时信息增益与前瞻价值的稠密 turn 级奖励，无需运行时模拟即可细粒度信用分配。DialogGRPO 在超过 88% 的模糊 query 上成功引出缺失信息，且不会对完整 query 过度追问，并零样本泛化到 MultiWOZ 2.4 与 AmbigQA。
- **arXiv**：[2611.01878](https://arxiv.org/abs/2611.01878)

#### AttnSR: Attention Step Reward for Long-Horizon Agentic RL (2026-11)
- **简介**：作者 Mengsong Wu、Tianchen Zhou、Zhaochen Su 等。仅用结果奖励的 RL 几乎无法为长程 agent 的单步决策提供监督。**AttnSR** 是 step 级奖励增强方法：不用额外训练的奖励模型或额外 rollout，而是从训练 replay 中提取的策略 attention 导出奖励分配——沿策略模型 attention 图反向追踪，估计较早步骤与最终回复的相关性；终止步相对 running baseline 的表现决定这些奖励是强化还是惩罚被选中的步骤。在受控的 Qwen2.5-7B-Instruct 实验中，AttnSR 在 ALFWorld 所有检查点上均优于 PPO 基线，最终 held-out 成功率由 91.2% 提升到 97.1%；在 ScienceWorld 上每个检查点均有提升并减少 29.9% 的环境交互。消融表明收益取决于中间奖励的分配方式而非简单累加。代码已开源。
- **arXiv**：[2611.07932](https://arxiv.org/abs/2611.07932)

#### BLPO: Beyond-Local Policy Optimization for Long-Horizon LLM Agents (2026-11)
- **简介**：作者 Xuanren Chen、Chongyang Tao、Wenpeng Hu 等。分步 group 式 RL 比轨迹级监督提供更细的信用，但可靠比较必须考虑交互历史差异，而在每个任务有限的 rollout 内强制历史一致会使动作偏好估计支撑不足。**BLPO** 从跨任务、跨训练更新共享的经验中估计 task-progress advantage：把轨迹前缀总结为任务进度状态、按功能抽象动作，使来自不同任务的对应决策可以贡献共享估计；为控制异质任务带来的偏差与噪声，在全局 memory 中以平衡聚合与可靠性加权累积 context-relative return。这些全局 advantage 与归一化动作上的局部比较及轨迹级信用互补，无需学习 critic 或额外 rollout。在 ALFWorld 与 WebShop（1.5B 与 7B 策略）上，同等训练预算下任务成功率优于所评估基线，学习更快、轨迹更短且几乎无额外开销。
- **arXiv**：[2611.03005](https://arxiv.org/abs/2611.03005)

#### CREDO: Evidence-Conditioned Credit Assignment for Temporal Reasoning Agents (2026-11)
- **简介**：作者 Yue Leng、Shengtao Wen、Xiang Chen 等。时序知识图谱问答（TKGQA）要求 agent 检索证据、选择事实并多步时序推理，但 GRPO 的轨迹级结果信号无法区分中间决策质量：下游失败会惩罚有用的证据获取，全失败组中相同奖励又消除了相对信号。**CREDO**（Credit Redistribution with Evidence-Driven Opportunities）利用中间证据恢复局部监督（包括全失败组），并决定结果监督作用于何处：通过同一问题 rollout 间的比较分配正向 stage 级信用，用保留的答案重叠作为机会代理，把一部分结果策略权重转移到推理与作答阶段；保留终端奖励与轨迹级 advantage，且重分配保持每条轨迹的总结果策略权重守恒。在 MultiTQ 上 Overall 准确率由 78.0% 升至 82.4%、Multiple 准确率由 55.0% 升至 65.6%；TimelineKGQA 跨 KG 准确率由 30.9% 升至 36.4%。
- **arXiv**：[2611.01643](https://arxiv.org/abs/2611.01643)

#### RLiD: Reinforcement Learning in the Decision Plane for Language Agents and LLMs (2026-11)
- **简介**：作者 Ziyi Chen、Yan Zhang、Jianhui Wei 等。token 级 RL 把 action 的 advantage 分配给整个生成回复（含动作解析、推理链与畸形内容），而环境只执行解析后的动作，同一动作的不同自然语言表达会给 policy gradient 带来额外方差。**RLiD**（Reinforcement Learning in the Decision plane）直接优化可执行动作选项的概率：把观测、可选的紧凑 memory 与带标签的动作菜单编译成输入，用大型决策模型（LDM）单次前向选出动作，免去自回归生成与动作解析，避免无效动作输出并提升 rollout 效率。分析表明在固定文本策略内，对同一动作的不同表达做边缘化可去除一个零均值梯度分量而不改变期望更新；训练好的 LDM 策略还可通过动作轨迹监督蒸馏回生成式 LLM。在 ARC、Countdown、Sokoban 上性能提升或成本下降，ALFWorld 上完整流水线以 12.7 H100 GPU 小时达到 99.3% 成功率。
- **arXiv**：[2611.02455](https://arxiv.org/abs/2611.02455)

#### SGR-OPSD: Learning from Subgoal Plans with Graph-Regularized Credit Assignment for Agentic Reinforcement Learning (2026-11)
- **简介**：作者 Shu Liu、Yang Chong、WenShi Chen、Man Lan。OPSD 通过 privileged information 反馈为 agentic RL 提供稠密监督，但现有 privileged information 缺少显式的任务进度规划，且师生对数概率差反映的是模型偏好变化而非真实任务进展。**SGR-OPSD**（Subgoal-Guided Graph-Regularized OPSD）把子目标监督与交互得到的证据结合做 turn 级信用分配：用 LLM 生成的子目标计划提供稠密动作级指导，并经带约束、KL 正则的目标转为 advantage 权重；构造 Subgoal Progress Graph，由采样的转移频次估计经验回报，对比各动作与状态平均的经验回报得到偏好，用以校准教师的信用权重。在 ALFWorld、WebShop、Search-QA 上两种规模（Qwen2.5-3B/7B）持续优于 GRPO 与强自蒸馏基线，相对 GRPO 成功率最多提升 7.6 个百分点，Qwen2.5-7B 在 ALFWorld 上达 85.8%。
- **arXiv**：[2611.02414](https://arxiv.org/abs/2611.02414)

#### TSBPO: Tree-Structured Branching for Policy Optimization in Interactive Language Agents (2026-10)
- **简介**：作者 Rui Yu、Chong Mu、Jinchuan Zhang。多轮 agent 训练只有终端反馈，使 rollout 分配与局部信用都欠确定。**TSBPO** 包含两个组件：轻量细粒度的 Root/Branch 预算分配器，利用终端结果对比的似然拟合估计，在任务间分配事实 rollout、在可回放状态间分配额外后缀；在得到的物理 rollout 树上，tree-aware 的 step 级信用分配算法比较紧邻的兄弟片段，把其局部结果差异分配给其中包含的 turn，精确的 task/observation 分组提供补充的 step 项。分配器决定采样什么，观测到的结果决定后续信用。以 Qwen2.5-1.5B actor，固定检查点在 ALFWorld 上 seen/unseen 的随机成功率为 97.79% / 96.64%，WebShop 500 个测试目标上 6 次随机 episode 的单 episode 成功率为 81.07%；去掉 observation-group step 项后在 WebShop update-100 检查点降为 75.93%。
- **arXiv**：[2610.41697](https://arxiv.org/abs/2610.41697)

#### Multiscale Policy Optimization for Long-Horizon LLM Agents (MPO) (2026-10)
- **简介**：作者 Tao Ren、Hui Yang、Jia Deng 等。长程 LLM agent 以完整的 assistant turn 与环境交互，而每个动作是逐 token 生成的。论文观察到 critic 的 value 主要在 turn 之间变化，因此提出以 turn 为锚点、在 token 间做更细粒度 credit 分配的 **MPO**（Multiscale Policy Optimization），由 Multiscale Advantage Estimation（MAE）与 turn-level importance sampling 组成，使 credit assignment 与策略更新在 turn 尺度上对齐。在搜索、工具使用和自动化任务上，使用 4B 与 30B-A3B 模型均稳定优于强 RL 基线（abstract 未给出具体数值）。
- **arXiv**：[2610.35013](https://arxiv.org/abs/2610.35013)

#### FOCUS: Active Critic Learning for Long-Horizon Language Model Agents (2026-10)
- **简介**：作者 Zhengping Jiang、William Jurayj、Benjamin Van Durme 等。稀疏结果奖励使长程语言模型 agent 难以对早期动作做信用分配；学习得到的 critic 估值不准，而用重复 continuation 采样估计中间值又代价高昂。**FOCUS** 用选择性分支帮助 critic 学习价值估计：rollout controller 从预期最能降低 critic 估计误差的状态主动采样 continuation，再用 TD(λ) advantage 为相邻状态间的片段打分，并以采样证据作为 critic 训练目标；通过模块化的上下文管理接口，把 context compaction 引起的片段边界用于价值估计与信用分配。在法律推理、deep search 与代码定位 benchmark 上，相同每步生成预算下优于 GRPO 和 VinePPO；Qwen3.5-9B 上 BrowseComp-Plus 55.4 对 48.2，DeonticBench 56.5 对 52.4。
- **arXiv**：[2610.35801](https://arxiv.org/abs/2610.35801)

#### ProtAgent: Post-Training Compact Protein Tool Agents with Interaction-Level Credit Assignment (2026-10)
- **简介**：作者 Yuxiang Wang、Baojian Zhou、Yanghua Xiao 等。蛋白质注释 agent 需要选择合适的生物信息学工具、解读输出并用于最终回答，现有系统多依赖前沿规模模型。**ProtAgent** 是两阶段 pipeline：先 SFT 学习工具接口，再用标准 on-policy REINFORCE 提升交互策略；中间工具调用获得 action reward，最终回答获得 terminal answer reward，并用 step-level return 把奖励分配给产生它的响应和之前的动作（对比对每个动作分配同一 episode return 的 baseline）。在真实生信工具的闭环协议下，ProtAgent-8B 在 annotation-coverage 指标上成功率 61.00%，高于 DeepSeek-V3.2-Thinking 671B 的 41.60%，与最强前沿系统 60.80% 相当；三 seed 比较中 step-level return 为 61.40%，episode-level 为 42.70%。作者提示该指标不等于生物学正确性。
- **arXiv**：[2610.33746](https://arxiv.org/abs/2610.33746)

#### AdaPRA: Process Reward Agents with Adaptive Rubrics for Data Science (2026-10)
- **简介**：作者 Atharva Naik、Vivek Chari、Daniel Fried、Carolyn Rosé、Benjamin Van Durme 等。数据科学 agent 可自动化端到端机器学习流程，但存在探索不足、验证不严谨、代码 bug、预算使用低效（每次尝试重复同样的 EDA 与基线）等过程错误。**AdaPRA** 是跨尝试积累知识的持久化编码 agent harness，配合 rubric 引导的过程监督来识别并纠正错误：process reward agent 通过执行检查审计 agent 的轨迹与工作区，并按任务自适应调整 rubric；从简单任务种下的 rubric 可泛化到较难任务，rubric 自适应进一步提升效果。每任务五次尝试，AdaPRA 使 80B Qwen 模型在 DSPredict-Easy 和 DSPredict-Hard 上的排行榜百分位分别提升 24.5 和 14.8 点，同时减少过程错误；也优于数据科学 PRM DataPRM，且一个基础模型 run 中挖掘的 rubric 可提升另一个模型（DeepSeek-V3.2）。
- **arXiv**：[2610.33245](https://arxiv.org/abs/2610.33245)

#### Tree-Aware Posterior Distillation for Long-Horizon Agentic Reinforcement Learning (2026-10)
- **简介**：作者 Heming Zou、Qi Wang 等。长程 agentic RL 面临两个问题：仅有终局 reward 使 turn 级 credit assignment 稀疏且高方差，同一 rollout batch 上的多次 mini-batch 更新又带来 policy staleness；现有 critic、PRM、OPD 方案都需要 value 网络、步级标注或更强外部 teacher。论文提出 **TreeOPSD**：无 critic 的 tree-aware posterior self-distillation 框架，把每个交互 turn 视为树节点，从共享前缀分支出不同续写，递归回传终局 reward 得到隐式 turn 级 credit；再把 credit 映射为以相对冻结行为策略的 likelihood ratio 为参数的闭式有界 teacher 分布，最小化 teacher 与 student 的散度即可在行为策略处恢复 policy gradient 方向，同时为多次 mini-batch 更新提供有限目标、防止 overshoot。在多个 agent benchmark 和模型规模上有效，例如在 Deep Research 设定下使 Qwen3-30B-A3B-Thinking-2507 平均准确率比 GRPO 高 4.97 分。
- **arXiv**：[2610.23181](https://arxiv.org/abs/2610.23181)

#### One Act, Many Masks: Effect-Marginal Policy Optimization for Long-Horizon Agents (2026-10)
- **简介**：作者 Shaojin Ma、Ruitao Wang 等。多轮 RL 虽已有细粒度 credit assignment，但策略优化仍通过各回复自身的概率比应用这些信号；不同文本回复可能产生相同的可执行效果，而由后续结局产生的 credit 差异却绑定在各自文本实现上，作者称之为 credit-update misalignment。论文提出 **Effect-Marginal Policy Optimization (EMPO)**：用 effect 级 surrogate 把优化与执行层等价对齐，耦合同一效果的不同文本实现的更新；保留原有 credit 信号，可与轨迹级和步级 credit 估计器配合，且不需要额外 critic 或 rollout。在 ALFWorld、WebShop 和 Sokoban 上，任务表现与学习效率均优于强基线。（abstract 未给出具体数值）
- **arXiv**：[2610.29427](https://arxiv.org/abs/2610.29427)

#### Self-Credit: Credit Assignment for Long-Horizon Agentic RL via Intra-group Rescaling (2026-10)
- **简介**：作者 Fangcong Yin、Ming Yin 等。长程 agent 的 RLVR 奖励稀疏，credit assignment 困难。论文提出 **Self-Credit**，基于 group-based RL 由稀疏 outcome reward 生成稠密监督，包含三部分：rollout 过程中策略模型生成带引用的摘要，引用包含关键证据的早期 turn，作为自生成 credit；通过 intra-group rescaling 比较同一 prompt 下成功与失败 rollout 中被引用的动作，利用组内一致/分歧去除不应得的 credit、强化关键动作；再用重标定权重对被引用 turn 做 advantage reweighting。在 BrowseComp-Plus 上让 Qwen3 8B 与 32B 分别比基座提升 37.0 和 36.2 分，比最强 agentic RL 基线高 8.5 和 5.6 分；在 GAIA 和 xBench-DeepSearch 上分别比最强基线高 4.9 和 4.7 分。
- **arXiv**：[2610.28388](https://arxiv.org/abs/2610.28388)

#### From Outcomes to Steps: Reliable Success-Aware Critic Learning for Long-Horizon Language Agents (SCALE) (2026-10)
- **简介**：作者 Jiaxi Liu、Shixiong Kai、Zhentao Tang、Yang Li、Mingxuan Yuan 等。长程语言 agent 只有轨迹级监督时，需要判断中间动作使最终成功更可能还是更不可能；但整体回报预测准确的 critic 未必能分辨局部价值差异，在稀疏反馈与函数近似下尤甚。提出 **SCALE**，一个 actor-critic 框架：critic 在每个交互前缀同时预测回报值与 Success Value（从该前缀出发最终完成任务的概率），通过 **Success-GAE** 把 Success Value 的变化转为多步动作 credit，并按预测器可靠性调节其贡献，同时保留 token 级 PPO 目标。在 5 个 agent benchmark、3 种模型骨干上，SCALE 在多数设置下优于对比 RL 基线，成功率最高提升 4.1 个百分点，且独立的 critic 分析显示其改善了价值估计与训练稳定性。
- **arXiv**：[2610.27623](https://arxiv.org/abs/2610.27623)

#### Outcome-Anchored Credit Reallocation for Agentic Knowledge Graph Question Answering (OACR) (2026-10)
- **简介**：作者 Mingyuan Li、Yucheng Li、Xiaxia Wang、Gong Cheng。Agentic KGQA 要求 LLM 通过多轮与知识图谱交互解题；outcome 监督 RL 把 rollout 的结果级 advantage 均匀分给所有 token，而 rollout 内各 KG 操作对搜索过程与最终答案的影响并不相同，造成轨迹与操作之间的错配。提出 **OACR (Outcome-Anchored Credit Reallocation)**：用仅训练时使用的 KG-state outcome model 估计相邻 KG 搜索状态间的结果奖励变化，将其转为操作 credit，并据此导出 token 权重；这些乘性权重重新分配 token 级更新幅度，同时在每个生成 token 上保持结果级 advantage 的符号不变。在三个 benchmark 上，OACR 整体优于已有 KGQA 方法，且推理时 LLM 调用次数与 token 数更少（abstract 未给出具体数值）。
- **arXiv**：[2610.25384](https://arxiv.org/abs/2610.25384)

#### LLM Value Functions for Credit Assignment in Agentic RL (LVRL) (2026-10)
- **简介**：作者 Kaijie Zhu、Weixiang Yan、Alexander Toshev、Andrew Szot 等。长 horizon 稀疏结果奖励下，GRPO 给所有动作同样的 advantage，critic 方法又昂贵且不稳定。提出 **LVRL（LLM Value Function RL）**：利用预训练 LLM 的知识与 in-context learning 估计策略相关的状态价值，无需微调独立 critic；value model 以组内上下文为条件，并用初始与终止状态的已知价值校准预测，由价值差分得到 step 级信用。进一步扩展为 **LVOPD**，把 teacher 监督聚焦于估计价值下降的动作。在 AppWorld、BabyAI、SWE-bench Verified 上（4B-32B），LVRL 稳定优于 GRPO 与 PPO；AppWorld 与 SWE-bench Verified 上最高比 GRPO 高 13.6 和 8.6 个百分点，BabyAI 上从完全失败提升到 75.0% 成功率；LVOPD 在 AppWorld 上也优于标准蒸馏基线。
- **arXiv**：[2610.15840](https://arxiv.org/abs/2610.15840)

#### Contextual Credit Policy Optimization (CCPO) (2026-10)
- **简介**：作者 Huiwen Han、Ziyao Zeng、Zihao Zhu、Hao Wang、Hongyu Xu 等。基于组的 agent RL 通过比较轨迹间观测匹配的经验得到 step 级信用，但相同观测可能出现在不同上下文中，匹配到的同伴信息量并不相等；而且有利的终端结果会掩盖成功轨迹里的冗余动作或绕路。**CCPO** 依据冻结 LLM 表示和累计轨迹统计量计算的上下文相似度为观测匹配同伴加权；并将「历史信用」（实际回报与加权同伴基线之差）与「未来信用」（短期已实现延续中上下文结果价值的变化）配对，使一步因其带来的进展而获得信用。两者均基于已收集的 rollout 事后计算，无需辅助 value critic。在 Qwen2.5-Instruct（1.5B 与 7B）上，ALFWorld 最高 97.4%、WebShop 79.4%，最多比 GRPO 高 21.6 和 18.5 个点。
- **arXiv**：[2610.19103](https://arxiv.org/abs/2610.19103)

#### ProAS: A Progress-Aware Advantage Shaping Framework for Multi-Turn Agentic Reinforcement Learning (2026-10)
- **简介**：作者 Bowen Ding、Mingda Wan、Yuhan Chen、Jiyao Yuan、Qi Zhu、Futing Wang。GRPO 类 critic-free 算法把轨迹级 advantage 平均广播给所有动作，会强化偏好轨迹中结构上不合理的动作，也忽略非偏好轨迹中各动作的下游偏离。**ProAS** 是即插即用框架，利用动态演化的 Agentic Reference Graph（ARG），通过 Masking Unreasonable Incentives（MUI）与 Asymmetric Penalty Redistribution（APR）实现细粒度信用分配，无需 critic。在 Qwen2.5-1.5B/7B-Instruct 上，ProAS 将主流 group-based 基线提升到新的 SOTA：ALFWorld 峰值成功率 91.9% 与 94.3%，WebShop 66.7% 与 70.8%；训练收敛最多加快 2.44 倍，且对不完美或蒸馏得到的 ARG 保持稳健。
- **arXiv**：[2610.18683](https://arxiv.org/abs/2610.18683)

#### Of Errors and Echoes: Responsibility-Aware Credit Assignment for Tool-Use Agents (ECHO) (2026-10)
- **简介**：作者 Xingjian Diao、Tianyu Yang、Wenjun Huang、Xiangchi Yuan、Chunhui Zhang 等。即便有 step 级反馈，tool-use agent 的细粒度信用分配仍然困难：观察到的错误可能继承自更早的决策，多个负责的决策可能共享同一下游后果或参与后续修复。**ECHO（Error-Chain-Aware Optimization）** 把责任识别与信用分配分离：冻结 judge 结合回溯式轨迹分析与基于策略可见前缀的决策时评估，区分新引入错误、继承后果、遗漏修正和成功修正；再对共享下游后果在各责任决策间只计一次，并通过 source-funded recovery 约束对自身错误的修复奖励，产生有界的动作级监督。在六个工具使用基准的 4B 与 8B 规模上，ECHO 在每个规模的六个基准中有五个取得最高分。
- **arXiv**：[2610.16027](https://arxiv.org/abs/2610.16027)

#### DeltaTTG: Training Multi-Turn Agents with a Token-to-Go Progress Meter (2026-10)
- **简介**：作者 Xuyan Ye、Xiyan Fu、Shengda Fan、Jiawei Cui、Yankai Lin、Wei Lu。多轮 agent 的 RLVR 依赖稀疏终端奖励；组内 step 比较需要可靠状态匹配与足够多样的结果，直接 step 打分又需外部评估器、人工标注或任务特定启发式。**DeltaTTG** 学习可复用的进度信号「progress meter」：以 Token-to-Go（直到任务完成还需生成的 token 数，作为剩余完成工作量的可观测代理）为监督目标，利用多样策略的已记录轨迹低成本训练；actor 训练时用冻结 meter 估计值的变化在各步间重分配由结果得到的 advantage，不改变奖励，并保持每条轨迹 advantage 的符号与 token 平均尺度。在 ALFWorld、PaperSearchQA 和 BFCLv3 上，平均分比 GRPO 高 5.7（Qwen3-4B）和 7.2（Llama-3.1-8B-Instruct），且训练 GPU 时长与 GRPO 接近。
- **arXiv**：[2610.17666](https://arxiv.org/abs/2610.17666)

#### PEARL: Process-Experience Aware Reinforcement Learning for Multi-Turn Agents (2026-10)
- **简介**：作者 Jie Yang、Mianqiu Huang、Yiwei Zhao、Bo Wang、Yang Liu、Yuxin Wang。终端奖励让多轮 agent 无需 step 标注即可训练，但组相对 advantage 对「该调整哪些中间决策」指导很少；失败轨迹里也含合理动作，而不良决策模式会随训练反复出现。**PEARL** 在无额外 critic 下改进局部信用分配：给定同一历史、结果不同的两个 continuation，定位它们首次分叉处，构建各分支局部决策的特征表示；在近期失败与成功记忆中对这些特征聚类，识别重复出现的局部模式；据此仅对低奖励决策选择性加强负 advantage，保持正 advantage 与基础优化器不变。在 AppWorld 上 4B 与 8B 规模的 Normal 和 Challenge 任务均优于 ARPO；ALFWorld 上 4B 平均成功率达 97.9%，比最强基线高 2.7 点，8B 保持竞争力。
- **arXiv**：[2610.14636](https://arxiv.org/abs/2610.14636)

#### Mosaic: Multi-Session Agentic Reinforcement Learning (2026-10)
- **简介**：作者 Yilong Xu、Minrui Xu、Xinyu Zhou、Xiang Long、Haochen Tan、Zhijiang Guo。多数 agentic RL 假设单 session 轨迹，但 context compaction、subagent forking 等技术会产生跨多个 session 的轨迹，与单 session 训练假设失配。论文提出 Multi-Session Agentic RL 这一更一般的设定（轨迹内可有任意数量的 session）；常用做法是把轨迹级 advantage 广播给所有 session，无法区分各 session 贡献也无法指导 session 分配。**Mosaic** 是 critic-free 的分层优化方法：轨迹级把 session 使用信号纳入 advantage 估计，session 级依据估计的边际贡献塑造每个 session 的 advantage。在 BrowseComp 上用 Qwen3.5-4B，Mosaic 在 context compaction 与 subagent forking 两种场景下都优于广播基线，且平均使用更少 session（abstract 未给出具体数值）。
- **arXiv**：[2610.22174](https://arxiv.org/abs/2610.22174)

#### RewardWeaver: Long-Horizon Interactive Learning for Language Agents via Self-Evolving Reward Adaptation (2026-10)
- **简介**：作者 Hengbo Xiao、Boyao Zhang、Purui Liu、Yuxuan Zheng、Haoran Yin、Haibo Liu。长程交互中终端反馈稀疏、信用分配困难；过程奖励提供更稠密监督，但随策略演化，当前真正限制成功的能力瓶颈会变化。**RewardWeaver** 维护一个已验证的能力空间（其中入选 Rubric 的语义固定），闭合策略优化、任务评估、失败归因与奖励适配的循环：每个训练阶段后，对低结果轨迹做 outcome-grounded 的反向归因，聚合反复出现且受策略控制的能力瓶颈，并动态选择下一阶段对应的过程奖励；现有能力空间未覆盖的反复失败会触发受控的扩展流程。在 SOTOPIA、Amazon History Price 和新构建的 Sales Benchmark 上评测，覆盖社交互动、双边议价与垂直销售（abstract 被截断，具体提升数值以原文为准）。
- **arXiv**：[2610.10120](https://arxiv.org/abs/2610.10120)

#### SCAD: Structured Credit Assignment and Distillation for Long-Horizon Agents (2026-10)
- **简介**：作者 Shangyang Wu、Shuai Zhao、Ziyue Zhu、Jinyang Wu、Anh Tuan Luu、Haoran Luo。长程 agent 训练面临稀疏终端奖励掩盖中间贡献、以及 OPD 在 student 历史增长后丢失有效 teacher 引导的问题。**SCAD** 把交互组织为规划与有界子任务执行：在局部上下文中蒸馏执行，通过跨 rollout 的子任务前缀树细化规划信用——规划获得完整终端信用，执行获得正的终端信用加 teacher 引导。在所有评测 benchmark 上，宏平均准确率比最强训练基线在文本任务上高 4.48 个百分点，多模态任务上高 4.19 个百分点，有效结合了基于结果的信用分配与 teacher 引导蒸馏。
- **arXiv**：[2610.03372](https://arxiv.org/abs/2610.03372)

#### Credit Where It Matters: Dependency-Aware Policy Optimization for Terminal Agents (DepGPO) (2026-10)
- **简介**：作者 Yu Li、Guangfeng Cai、Long-Fei Li、Shuo Han、Shengtian Yang、Han Luo。终端 agent 的后续命令往往依赖于前序命令产生的信息，但现有轨迹级和步骤级信用分配方法没有显式追踪命令如何通过读写依赖影响最终结果，训练信号可能落到无关操作上。**DepGPO**（Dependency-Aware Group Policy Optimization）利用命令间执行依赖指导信用分配：从执行轨迹构建命令依赖图，从任务 verifier 检查的资源出发反向追踪，沿路径把信用分配给相关的写入及其支撑读取，并据此在各步间重新分配轨迹 advantage。大量对比与消融实验表明 DepGPO 在复杂终端任务上提升了任务性能与训练稳定性（abstract 未给出具体数值）。
- **arXiv**：[2610.03634](https://arxiv.org/abs/2610.03634)

#### AdaStep: Adaptive Step Credit Weighting for Agentic Reinforcement Learning (2026-10)
- **简介**：作者 Xin Wang、Wenhao Wu、Menghao Zhang、Zhi Wang、Kun Shao、Jian Luan。长程 LLM agent 通常用稀疏结果奖励训练，轨迹级目标太粗；步骤级信用分配更细，但观测回报同时依赖后续动作、环境转移与轨迹长度，估计不可靠。**AdaStep** 自适应地控制每个由 group 得到的局部 advantage 对轨迹级信号的修改强度：把加权表述为潜在步骤 advantage 的均方误差估计问题，在显式条件采样假设下推导出每状态最优的收缩系数，具有「信号/总方差」的解释——当回报变化可归因于所选动作时保留局部信用，被下游随机性主导时则抑制。仅需轻量标量计算，无需 critic、额外 rollout 或额外模型推理。在三种 backbone 的 ALFWorld、WebShop、ScienceWorld 上一致优于基线。
- **arXiv**：[2610.03223](https://arxiv.org/abs/2610.03223)

#### VETTA: Coordinating Turn- and Token-Level Credit Assignment for Multi-Turn LLM Agents (2026-10)
- **简介**：作者 Jiaju Chen、Min Yang、Jinghua Piao、Xiaochong Lan、Xu Xia、Xiangnan He。多轮 agent 面临两个相关的信用分配问题：哪些回复促成了结果？回复内哪些生成决策重要？现有方法通常只聚焦一层：turn 级方法评价完整回复但不区分其中决策，token 级方法能跨轮传播反馈却不显式建模每个回复的信用。**VETTA** 在共享的轻量 critic 上用独立 head 联合学习 turn 级与 token 级价值，沿两个时间序列计算 advantage，并把每个 turn advantage 与以回复内居中的 token 残差结合用于 PPO 更新；为降低价值学习成本，critic 仅保留初始化 actor 的预训练 checkpoint 的前若干 Transformer block。Qwen2.5-7B-Instruct 上成功率为 ALFWorld 95.5%、WebShop 76.0%、ScienceWorld 52.2%。
- **arXiv**：[2610.08402](https://arxiv.org/abs/2610.08402)

#### Dependency-Aware Reward Shaping for Agentic Reinforcement Learning (DARS) (2026-10)
- **简介**：作者 Ziyi Chen、Yan Zhang、Jianhui Wei、Daoan Zhang、Zuozhu Liu 指出，终局 reward 下常见步级 credit 方法忽略：建立在未修正错误上的工作是浪费的，而独立工作仍然有效。提出 **DARS**（Dependency-Aware Reward Shaping）：把任务进展表示为以前置关系连接的 predicate，在依赖图上分配步级 credit；annotator 标注每步验证、使失效或修复了哪些 predicate，已验证 predicate 按距最近损坏前置的图距离折扣，独立的不受影响，再由固定 potential 转为带符号的逐步 reward。DARS 可接入 GiGPO、ARPO/AEPO 等方法而无需改变其 rollout 策略或优化器。在五类任务、1.5B–8B 模型上，ALFWorld 上较同预算 GiGPO 最多提升 10 点，亦提升 WebShop 与 Search-R1 QA；蒸馏的 8B annotator 与 API annotator 效果相当。代码已开源。
- **arXiv**：[2610.01207](https://arxiv.org/abs/2610.01207)

#### SHARPO: Segment-Level Credit Assignment for Agentic Reinforcement Learning (2026-09)
- **简介**：作者 Xinchen Du、Zhengze Zhou、Wenhui Zhu、Han Yu、Sen Na、Rohit Jain 指出 agentic RL 中单个局部错误即可导致任务失败，而轨迹级 reward 难以对单个决策分配 credit。提出 **SHARPO**（Segment-level Hindsight Advantage Reweighting for Policy Optimization），在面向环境的 segment 粒度上改进 GRPO：受 on-policy self-distillation（OPSD）启发，在每个 segment 内计算 teacher-student log 概率差，由此得到作用于 GRPO advantage 的有界乘子，同一 segment 内所有 token 共享该乘子，使 credit 随 segment 变化。基于 Qwen2.5-7B-Instruct，SHARPO 在 ALFWorld 与 WebShop 上优于 GRPO、SDAR、RLSD、StepOPSD 等现有基线（abstract 未给出具体数值）。
- **arXiv**：[2610.00838](https://arxiv.org/abs/2610.00838)

#### Cross-Rollout Bellman Closure for Long-Horizon Agentic Reinforcement Learning (CRBC) (2026-09)
- **简介**：作者 Yangyang Ren、Haodong Zhu、Linlin Yang 等观察到 GRPO 类方法在长程任务中各 rollout 会重访共享的 anchor 状态，提供了跨 rollout 的步级 credit 证据：visit-local 平均不跨 rollout 递推传播，最短路径估计器又可能被罕见路径主导。提出 **Cross-Rollout Bellman Closure（CRBC）**：将每组 rollout 合并为带吸收成功/失败边界的有限经验过程，通过一次线性求解得到其 behavior-policy Bellman 不动点，用相同的经验动作与转移频率在共享 anchor 间传播证据并聚合备选延续；由状态值回溯得到动作值，其相对状态值的增益即步级 credit，并与轨迹级 group advantage 结合，无需额外环境 rollout。在 ALFWorld、WebShop、Sokoban 多个模型规模上稳定提升最终性能与学习效率，如 Qwen2.5-1.5B-Instruct 在 ALFWorld 上比最强评测基线高 5.59 个百分点。
- **arXiv**：[2609.35082](https://arxiv.org/abs/2609.35082)

#### Learning from Viable Failure Prefixes: Milestone Viability Potential Policy Optimization for Long-Horizon LLM Agents (MVPO) (2026-09)
- **简介**：作者 Qi Zhou、Yuanfan Li 指出 GRPO、GiGPO 等基于组的方法在比较的回报无差异时失效，称为 zero-credit failure：训练早期许多失败 rollout 含有用前缀，却得不到任何区分性 advantage。提出 **MVPO**（Milestone Viability Potential Policy Optimization），在 Union-Find viability 区域上估计前缀 potential，用 potential 差分 advantage 修复 zero-credit 组，并按相对性能进展衰减 potential 分支。Qwen2.5-1.5B-Instruct 上优于包括 GRPO、GiGPO 在内的八个强基线；同等训练长度下，相对 GiGPO 在 ALFWorld 提升 +4.4、WebShop 提升 +5.3 成功点，advantage 构建开销仅增加 0.16%–0.20%。
- **arXiv**：[2609.37111](https://arxiv.org/abs/2609.37111)

#### SLCA-GRPO: Resolving Cross-Segment Credit Misattribution in Tool-Calling RL (2026-09)
- **简介**：作者 Yan Zhan、Shaobo Liu 等。工具调用 agent 的输出由结构化 tool 调用与面向用户的自然语言摘要交织而成，而 GRPO 把同一个轨迹级标量 advantage 广播给所有 token，导致摘要生成的梯度噪声污染 tool 决策 token（跨段 credit 误分配）。论文提出 **SLCA-GRPO**：在同一组 rollout 内按结构段解耦 advantage（Segment-Locked Credit Assignment），配合分层奖励 HierR，将执行 advantage 路由到 tool token、偏好 advantage 路由到摘要 token，并构建 Schema-Guided LLM Simulator (SGLS) 作为训练基础设施。7B 模型上，相比 GRPO、ToolPO、RLTR，in-domain 提升 2.53 pp，BFCL 提升 1.36 pp，$\tau^2$-Bench 提升 9.15 pp，同时降低 tool 冗余与成本。
- **arXiv**：[2609.29050](https://arxiv.org/abs/2609.29050)

#### Back to the Definition: Estimating Step-Level Advantages via Trajectory Graphs for Agentic Reinforcement Learning (GRAFT) (2026-09)
- **简介**：作者 Xincheng Yao、Haobo Fu、Weiming Liu 等。GRPO 的组归一化 advantage 在 response 级可靠，但在 step 级会系统性偏置（失败轨迹中也有有价值的步骤）；要忠实估计需从每个中间状态多次采样，代价过高。论文提出 **GRAFT**：把所有 rollout 轨迹合并成轨迹图，在图上通过 Bellman 迭代恢复节点状态价值，并以节点价值差为边分配 credit；进一步提出 Graph GAE 以降低状态价值估计偏差。理论上该 step 级 advantage 忠实于 RL 基本定义。在多个多轮 agentic 基准上稳定优于 GRPO 及近期 agentic RL 算法（abstract 未给出具体数值）。
- **arXiv**：[2609.28963](https://arxiv.org/abs/2609.28963)

#### Reinforcement Learning with Decomposed Subtasks (RLDS) (2026-09)
- **简介**：作者 Mattie Terzolo、Mikolaj Sacha、Ayan Sinha、Andrew Rabinovich。GRPO 把整个多轮 rollout 折叠为单一标量奖励，当任务由不同技能组成且反馈稀疏延迟时信息损失严重。论文提出 **RLDS**，核心是 Subtask-Decomposed Advantage Estimation (SDAE)：在固定分类体系上把轨迹奖励拆分为各子任务份额，分别计算组相对 advantage，并按重要性、围绕 reflection 标记的关键步骤分配 token 级 credit。在 FrozenLake、HotpotQA、ScienceWorld、DeepResearch 四个基准上评估，收益随子任务异质性增长：ScienceWorld +11.5 分、FrozenLake +9.8 分，HotpotQA 与 DeepResearch 持平；ScienceWorld 每步 wall-clock 还降低 10.9%。
- **arXiv**：[2609.27035](https://arxiv.org/abs/2609.27035)

#### ProCredit: From Outcome Rewards to Progress Credit in Agentic Reinforcement Learning (2026-09)
- **简介**：作者 Ming Ma、Yi Zhu、Yiran Zhong 等。长程 agentic 任务只有终态结果奖励时，全失败的组无训练信号、失败尝试无法区分接近完成的程度、推进任务的 turn 与仅查询环境的 turn 获得相同 credit。论文观察到决定成功的验收检查同样可以在中间状态上运行，即进度和结果一样可验证。**ProCredit** 在每个 turn 后重跑验收检查，按进度变化奖励该 turn，并用这些奖励同时在同一任务的多次尝试之间、以及轨迹内的 turn 之间分配 credit。在 AppWorld 上以三种规模的 Qwen3.5 基座训练，各规模均优于结果奖励与进度类基线，4B 上比最强结果奖励基线高 4.1 个百分点；消融表明仅把最终进度加到轨迹分数上无增益，收益来自把进度记到其发生的 turn。
- **arXiv**：[2609.27532](https://arxiv.org/abs/2609.27532)

#### ArenaFlow: From Trajectory Ranking to Hierarchical Credit Propagation for Open-Ended Agent RL (2026-09)
- **简介**：作者 Qiang Zhang、Ruixue Ding、Fanrui Zhang 等。开放式 agent 任务难以获得可靠的标量奖励，近期的成对评估方法虽缓解了奖励区分度坍缩，却仍把丰富的比较反馈压缩成单个轨迹级奖励，掩盖了关键中间步骤，也无法把成功行为沉淀为可复用技能。**ArenaFlow** 用锦标赛式相对排名得到轨迹级奖励信号，并为每次比较配备结构化的反思评估，产出关键成功步骤、可复用策略技能及技能使用归因三类监督；在 step 层依据锦标赛存活深度把轨迹级 advantage 传播到高置信的关键步骤，在 skill 层从组级使用归因估计技能效用，并通过效用感知的更新、剪枝与检索维护全局技能记忆，高效用技能再作为未来探索的策略先验。在开放式 agent 任务上验证有效（abstract 未给出具体数值）。
- **arXiv**：[2609.21378](https://arxiv.org/abs/2609.21378)

#### Granularity-Adaptive Credit Assignment for Long-Horizon LLM Agent Reinforcement Learning (2026-09)
- **简介**：作者 Taoran Liang、Yang Liu 等。长程 LLM agent 的 RL 只有稀疏的结果 reward：episode 级 advantage 过于粗糙，step 级比较分辨率更高却带来依赖上下文的估计噪声。**GACA**（Granularity-Adaptive Credit Assignment）是一种 critic-free 方法，为每个决策自适应地混合 episode 级与 step 级 credit：先在轨迹内对响应的平均 per-token NLL 做归一化得到「关键度」分数，再据此决定各 step 的混合权重，整个过程复用 rollout 的 log-prob，无需额外 rollout 或模型评估。论文给出了最优分数相关混合的理论分析。在 ALFWorld 与 WebShop（1.5B 与 7B backbone）上，GACA 在对比方法中取得最高的平均成功率，且额外计算开销可忽略。
- **arXiv**：[2609.12424](https://arxiv.org/abs/2609.12424)

#### Optimizing Sparse Outcomes Through Dense Behavioral Signals via Value-Guided Preference Distillation (2026-09)
- **简介**：作者 Ziyi Zhu、Daniel R. Cahn 等。多轮对话 agent 直接优化长期结果往往效果差且易 reward hacking。本文将长程对话优化形式化为多目标 RL：训练一个 multi-head value model，在多个前瞻 horizon 上预测一组可观测的用户行为信号，并证明这些稠密辅助行为信号的标量化组合能实现有效的 credit assignment、优化稀疏结果。为在上线前发现无约束单目标代理造成的策略退化，作者构建了结合反事实用户模拟与对话级结果模型的安全评估框架；最后通过 reference-anchored preference optimization 把多目标 value 偏好蒸馏进策略，以极小的算力达到与 on-policy 在线 RL 相当的效果。线上 A/B 测试显示蒸馏后的策略显著提升长期用户留存。
- **arXiv**：[2609.14648](https://arxiv.org/abs/2609.14648)

#### Dual-Axis Policy Optimization for LLM Agents: Bayesian Feedback Attribution and Trajectory Mass Normalization (BATON) (2026-09)
- **简介**：作者 Yingxuan Zhuang、Binhe Yu 等。LLM agent 的 RL 涉及两个相互独立的优化维度：轨迹内如何利用环境反馈（Intra-Trajectory Feedback Attribution），以及批内如何聚合完整轨迹（Inter-Trajectory Objective Aggregation）。**BATON** 为双轴框架：第一轴用 Bayesian Feedback Attribution 构造以反馈为条件的、对已采样动作的后验；第二轴用 Trajectory Mass Normalization（TMN）给每条完整轨迹分配相等的优化质量。在 GRPO 与 GiGPO 之上，于 ALFWorld、WebShop、SearchQA 上的实验表明两个轴各自带来独立增益，组合后在各模型规模下取得最强整体表现。
- **arXiv**：[2609.19830](https://arxiv.org/abs/2609.19830)

#### Turn-level Multiscale Density Ratio Estimation for LLM Agents (tlm-DRE) (2026-09)
- **简介**：作者 Zishuo Zhao、Kai Chen、Ao Li、Yuan Liu。PPO、DPO、GRPO 等对齐方法多面向单轮任务，复杂多轮 agent 任务仍有提升空间。**tlm-DRE**（Turn-level Multiscale Density Ratio Estimation）为不同轮次赋予不同权重，并基于多轮任务中正负样本空间的差距提出非对称的 token 级训练。在多种 agent benchmark 上相比传统对齐方法表现有竞争力，并在域内与域外条件下的多轮推理任务中保持鲁棒（abstract 未给出具体数值）。
- **arXiv**：[2609.16760](https://arxiv.org/abs/2609.16760)

#### T1: Terminal Agent Reinforcement Learning for Long-Horizon Tasks (2026-09)
- **简介**：Junyao Yang、Yucheng Shi、Zhongzhi Li、Haitao Mi 等。面向终端（shell）类长 horizon 任务，**T1** 是一个 122B 总参数的 MoE 模型，用 RL 在云沙箱中操作真实 shell、单任务最多 300+ 个 tool-call turn，并以每个任务自带的 verifier 执行结果为 reward。训练配方有三点：其一为激进 warm-start 以稳定 actor-critic，并用按通过 verifier 绝对数量打分的 dense process reward；其二通过 TITO 构造在 turn 边界做 drift 修复、只在采样得到的确切 token 标识上训练，并用 rollout routing replay 记录采样器在每个 MoE 层的逐 token 专家选择并在训练时重放；其三使用与 Terminal-Bench 2.1 完全不相交的 OOD 训练语料。TITO 与 R3 一起把训练–推理 log 概率差从 0.021 降到 0.013、loss 区域 token drift 精确为零；Terminal-Bench 2.1 上从基座 43.8% 提升到 64.0% resolved，Long-Horizon Terminal Bench 上达 27.9% 并超过 GPT-5.4 与 GLM-5.1。
- **arXiv**：[2609.11042](https://arxiv.org/abs/2609.11042)

#### DRACO: Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training (2026-09)
- **简介**：IBM Research（Shubham Gandhi、Saurabh Goyal、Kiran Kate、Yara Rizk）。RLVR 依赖 programmatic checker，但多数长 horizon agent 领域没有；作者工作在 outcome-blind 设定（无 ground-truth 成功信号）下，指出多准则 rubric 虽可充当 reward，却只对整条 trajectory 打一次标量分，对几十步的决策是很差的信号。**DRACO**（Distributing Rubric-based Advantage for Credit Optimization）在训练中动态生成 rubric 以跟踪 policy 能力演化，对完成的 trajectory 打一次 rubric 分，再把该判断重分配到负责相应 rubric 的步骤上，产生 GRPO 中有区分度的 per-step advantage；重分配是闭式的，不引入任何需训练的归因模块。AppWorld 上 DRACO 比 base model 高 15.9 个点、比用稀疏 ground-truth reward 训练的 GRPO 高 5.3 个点（而自身完全不用 verifier）；域外 Tau-Bench 上即使没有 frontier judge 也比 base model 高 5.3 个点，优于 ground-truth reward 训练与其他 rubric 训练设置。
- **arXiv**：[2609.04094](https://arxiv.org/abs/2609.04094)

#### TIGPO: Temporal Instance-Graph Policy Optimization for Long-Horizon LLM Agents (2026-09)
- **简介**：Jinwei Gan。基于图的 policy optimization 把 rollout trajectory 组织成状态转移图以改善长 horizon credit assignment，但现有方法在每次 policy 更新内独立建图，丢弃早期 policy 发现的转移，advantage 估计被局限在 batch 局部的小 rollout group 内。**TIGPO** 把图式 credit assignment 跨 policy 更新延展：为每个任务维护持久化转移图，使不同 policy 版本发现的有效转移共同决定当前 rollout 的 credit；并把固定 rollout 预算切分为 Exploration slot（常规任务采样）与 Revisit slot（对先前已探索任务的延迟重试），每次 revisit 都把当前 rollout group 与其对应的历史 Exploration group 配对构成跨时间参照，用以在小 group 下稳定相对 advantage 估计，同时同任务对比可直接刻画训练阶段间的 policy 改进。历史转移与分数仅作结构性与 detached 统计参照、从不进入 policy loss。ALFWorld 与 WebShop 上稳定优于既有 group-based 与 graph-based 方法（abstract 未给出具体数值）。
- **arXiv**：[2609.03383](https://arxiv.org/abs/2609.03383)

#### PGPO: Potential-Guided Policy Optimization for Multi-Turn Agentic Tasks (2026-09)
- **简介**：Yuyao Zheng、Haipeng Sun、Junwei Bao、Lemao Liu 等。group-based RL 在 terminal reward 稀疏的多轮 agentic 任务中对中间动作 credit 过粗，GiGPO 等虽引入 step-level advantage，但其 step 信号仍取决于单条 trajectory 的最终结果，于是失败 trajectory 内部的动作难以区分，有效动作会与错误动作拿到同样的不利 credit。**PGPO** 从每个 rollout group 内 anchor-state-group 的回报统计中估计经验状态潜势（state potential），再由相邻状态的潜势差导出动作 advantage，从而实现跨 trajectory 的 credit 传播，为失败 trajectory 也提供有区分度的 step-level credit。ALFWorld 与 WebShop 上相对近期 group-based RL 方法整体表现更强，分析表明其失败侧 credit 信号更有信息量且训练开销几乎可忽略（abstract 未给出具体数值）。
- **arXiv**：[2609.02236](https://arxiv.org/abs/2609.02236)

#### VICT: Verifier-Instrumented Credit Tracing for Long-Horizon LLM Agent Reinforcement Learning (2026-08)
- **简介**：Pengcheng Li、Zhengyang Zhang、Dongxu Zhang、Sui Huang 等（EMNLP 2026）。标准做法把可程序化验证的 terminal reward 广播给 trajectory 中每个动作，而现有细粒度方法多从 rollout 侧构造辅助信号或额外比较来估计动作重要性，却仍把判定成功的 verifier 当作一个标量 reward，丢弃了它内部的任务结构。**VICT** 的关键洞察是许多可验证任务已把相关检查编码在 terminal verifier 内部，于是提供一个训练期接口：暴露可执行或有证据支撑的 atom，并沿依赖有效的 proof edge 把它们回溯到具体动作，仅在这些边上重分配 group-relative advantage，从而把 credit assignment 从 rollout 侧推断转为 verifier 侧追溯。方法保留原始 terminal reward、在证据不全或歧义时弃权（abstain），只改动训练期的 advantage 张量，无需 learned critic、过程标注、分支 rollout 或推理期访问 verifier。在 ALFWorld 与 WebShop 上大幅优于仅用结果奖励的训练，并与近期细粒度 credit 方法表现相当；消融排除了密集 atom reward、final-commit credit、时间邻近性与稀疏性作为充分解释（abstract 未给出具体数值）。
- **arXiv**：[2608.28128](https://arxiv.org/abs/2608.28128)

#### Reconciling Process Supervision with Outcome-Based Credit in Agentic Policy Optimization (TASPO) (2026-08)
- **简介**：Jingxiao Yang、Wangjie Gan、Yingxuan Zhuang、Wenqi Zhang 等。结果导向 RL 把 trajectory 级 advantage 均摊给所有决策，长 horizon 下 credit 过粗；on-policy 自蒸馏用仅训练时可得的 privileged information（PI）重评采样行为可提供更细监督，但细粒度监督不等于细粒度 credit——PI 引起的似然变化只说明额外信息如何改变 policy 偏好，并不决定可执行动作应如何继承已验证的任务结果，由此产生「监督—credit 鸿沟」（PI 可能与当前交互状态无关、token 粒度与可执行决策错位、缺少强化所需的结果语义）。**TASPO** 把 privileged 监督转为以结果为锚的动作 credit：从已验证的成功经验中构造决策可用的 PI，在可执行动作层面聚合 PI 引起的似然偏移，再把相对动作支持度转成正的、有界的、保均值权重作用于原 trajectory advantage——更新方向与平均尺度由验证结果决定，PI 只在动作间重分配 credit。三个 agentic 基准上比 GRPO 提升 10.6%，对未见任务泛化更好，且 action 级分配使优化更稳定。
- **arXiv**：[2608.31077](https://arxiv.org/abs/2608.31077)

#### Agent-G²: Gaussian Guidance for Agentic Reinforcement Learning (2026-08)
- **简介**：来自浙江大学等机构（Zixuan Wang、Yanrui Miao、Zhengxi Lu、Yongliang Shen 等 9 人）。hint-based RL 通过在每次 rollout 前保留专家轨迹的一段前缀来缓解长时程 agent 任务的 reward 稀疏问题，其效果取决于 guidance depth（保留多少轨迹）。现有方法把该深度当作确定性标量：调度式方案让所有样本共用一个值、忽略任务间异质性；per-sample 探测式方案逐样本估计但要付出额外 rollout 成本。作者发现有用的 guidance 实际占据一个深度**区间**，其信息量沿区间中心近似呈高斯分布，而非集中于单一最优点。**Agent-G²** 据此把每个任务的深度从一个高斯分布中采样，其中心与展宽都从已为策略优化收集的 rollout 中在线估计，无需探测 rollout 也无需学习深度预测器：中心由全局 baseline 与 per-cluster 难度组合而成，展宽跟踪簇内方差。在 ALFWorld 与 WebShop、Qwen2.5-1.5B / 7B-Instruct 上评测，ALFWorld 上分别比最强的 hint-based、hint-free 与 Aux-RL baseline 高 2.3 / 3.9 / 7.4 个点，而 rollout 成本不到 per-sample 探测方案的三分之一。
- **arXiv**：[2608.23318](https://arxiv.org/abs/2608.23318)

#### HiDiffTIR: Hierarchical Difficulty-Aware Policy Optimization for Multi-Turn Tool-Integrated Reasoning (2026-08)
- **简介**：中科院计算所（Yucan Guo, Xiaohan Wang, Miao Su, Saiping Guan 等）。针对多轮 Tool-Integrated Reasoning 中现有 RL 方法给整条 trajectory 统一 advantage、把所有正确工具调用等同对待，从而无法区分平凡与困难工具使用模式、学习信号不精确的问题，**HiDiffTIR** 在 trajectory 与 turn 两个层级做难度感知的 credit assignment，让 policy 聚焦更有信息量的 trajectory 与更难的推理步。该细粒度优化不需要任何额外监督，只依赖标准 RL rollout 得到的组内统计量。三个工具使用基准上多轮 TIR 表现与工具调用准确率均一致超过强 RL 基线（abstract 未给出具体数值）。
- **arXiv**：[2608.21863](https://arxiv.org/abs/2608.21863)

#### IAPO: Influence-Aware Policy Optimization for Credit Assignment in Multi-Turn Service Agents (2026-08)
- **简介**：Bo Ren, Yirong Mao, Yi Yang, Wenhui Que。服务型 agent 的任务信息随交互逐步展开（用户会澄清或修改目标、工具返回后续决策所需信息），仅靠最终 reward 无法指出哪些动作真正促成任务解决；已有方法依赖其他 trajectory 的对比证据、重采样续写，或单独构造 step 级学习信号。**IAPO** 指出一条已完成的 rollout 本身就记录了信息与错误在动作间的流动，于是把每条 rollout 表示为可训练 agent 动作之上的带类型 influence-dependency 图（用户与工具观察作为证据），再把 support-use 与 failed-use 结构转换成路由权重，用于重新分配同一条 trajectory-level advantage。Qwen3-4B 与 Qwen3-8B 在 τ²-Bench、UserBench、AgentChangeBench 三个服务 agent 基准上均优于多轮 RL 基线，BFCL-v4 Multi-Turn 进一步表明这些增益未损害多轮 function-calling 能力（abstract 未给出具体数值）。
- **arXiv**：[2608.24588](https://arxiv.org/abs/2608.24588)

#### ToSCA: Leveraging Hierarchical Reinforcement Learning on Temporal and Strategic Abstractions of Conversational Agents (2026-08)
- **简介**：Xiaoyu Wang, Qingqing Gu, Yue Zhao, Luo Ji 等。受人类在日常交互中具备概念感知与策略规划等多级时间抽象的启发，**ToSCA** 提出面向对话 agent 的两级分层 RL 框架，弥合以往 token 级与 utterance 级 RL 之间的断层：在两级 MDP 上，token 级回复解码以 utterance 级动作（显式的文本策略）为条件；基于理论推导与效率考虑，高层 critic 用 DQN 求解、低层 actor-critic 用 PPO 求解。为缓解 reward 稀疏并促进收敛，还设计双粒度 reward，把 utterance 级满意度分数与 token 级内在激励及 KL 惩罚结合。日常对话与情感支持对话实验中，策略决策与回复质量均优于多类基线（abstract 未给出具体数值），代码已开源。
- **arXiv**：[2608.21969](https://arxiv.org/abs/2608.21969)

#### Towards Better Agents for Multi-Turn User Interaction: The Next User Turn Is More Than Context (FACA) (2026-08)
- **简介**：Yiwen Zhao, Zhihao Wen, Yuchen Mao, Mingxuan Jiang 等。面向用户的工具 agent 需在多轮中协调对话与 tool use，但交互式 RL 通常把每条 rollout 压成一个终局 reward，让有效的信息 elicitation、错误与后续修复获得完全相同的 credit。作者指出下一轮用户反应不只是上下文，还是关于前一段 user-to-user segment 的带噪、时序局部的证据，据此提出 **FACA**（Feedback-Aware Credit Assignment）：把每个用户反应与对应 segment 对齐，导出局部归一化的 reaction advantage，再叠加到经 verify 的终局 outcome advantage 上，不需额外 critic 也不需额外 rollout。在与 outcome-only Interactive GRPO 在模拟器、可见对话、初始化、rollout 与优化设置上严格对齐的对照下，三次独立训练在九域 τ 系列基准上平均分别提升 5.91 和 10.22 个百分点（8B / 14B）；增益集中于 Telecom，且在 8B 上随机化反应极性即抹除该增益，同样的排序在 Pare-Bench 与 Co-Gym 上 zero-shot 保持。
- **arXiv**：[2608.17499](https://arxiv.org/abs/2608.17499)

#### PlanPO: Group Planning-Aware Policy Optimization for Multi-Turn Agentic LLMs (2026-08)
- **简介**：Dayang Liang, Liyuan He, Xuan Feng, Bo An 等。多数 group-relative 变体无法区分成功 trajectory 之间的 advantage，即使它们的交互效率差异巨大——绕圈式的成功往往拿到与最优解相同的 outcome reward，造成 advantage collapse 与严重性能瓶颈。**PlanPO** 引入 coarse-to-fine advantage 信号，在同一任务采样出的成功 trajectory 上刻画 trajectory 级长度与 turn 级回复长度的相对差异，从而在 group-relative 优化结构内让 agent 主动从高质量 rollout 中学习跨越交互规划与文本生成的、可泛化的审慎行为，而不退化成朴素的长度最小化。在 ALFWorld、WebShop、SciWorld 三个多轮基准上平均比 GRPO 提升 27.2%，超过近期强 baseline，且额外训练开销可忽略。
- **arXiv**：[2608.17289](https://arxiv.org/abs/2608.17289)

#### RTPO: Reverse-Turn Policy Optimization for Stabilizing Agentic RL Training (2026-08)
- **简介**：Yugu Li, Jimmy Cao, Jianglin Qiao, Siyi Hu。多轮 RL 训练极不稳定，turn 数增加时常出现严重性能退化；作者通过理论分析识别出三个紧耦合的不稳定来源：rollout-training 上下文错配、稀疏终局 reward 下 turn-level credit assignment 过弱，以及长短 trajectory 在不同 policy 版本下被优化引起的异步 policy drift，三者共同源于扁平化的 trajectory 优化。**RTPO** 用统一的 reverse-turn 表述加以解决：把多轮 rollout 组织为稀疏反向树，按时间逆序执行 turn 级 policy 更新，使每个决策与其下游 continuation 对齐，实现因果一致的 turn-level credit assignment 与 on-policy continuation 以抑制异步 drift，并给出消除上下文错配与异步 drift、降低 credit 偏差并收敛到递归最优的理论保证。在多轮 agentic RL 基准上比 trajectory 级与 turn 级 baseline 分别提升 21.50% 和 10.76%。
- **arXiv**：[2608.18682](https://arxiv.org/abs/2608.18682)

#### TRCA: Transition-wise Rubric Credit Assignment for Long-horizon LLM Agents (2026-08)
- **简介**：Huan Zhang, Mingju Chen, Dongxu Zhou, Heng Chang 等。长 horizon agent 通常只用稀疏终局结果优化，细粒度 credit assignment 困难；已有方案要么依赖过程评估器（带来标注与推理成本），要么从成功 trajectory 反推 step 级 credit，而 RL 早期成功 trajectory 极其稀缺，使 anchor-based 方法大幅失效。**TRCA** 直接从动作引发的状态转移导出 step 级监督，不需学习式评估器也不需成功 anchor：用 Evidence、Execution、Invalidity 三条 rubric 评判每个 transition，分别捕捉任务相关信息获取、有效任务执行与无效/退步行为；Foundational Rubric Reward 度量局部转移质量，Breakthrough Rubric Reward 追踪新覆盖的 Evidence 与 Execution 条件以奖励增量任务进展，二者与终局结果合并成细粒度 step 级 advantage 用于 policy 优化。在 ALFWorld、WebShop 与七个搜索增强 QA 基准上一致优于所评 baseline：Qwen2.5-7B-Instruct 上 WebShop 分数提升 6.0%–12.6%，Qwen2.5-3B-Instruct 上 SearchQA 平均分提升 1.9%–18.3%。
- **arXiv**：[2608.16156](https://arxiv.org/abs/2608.16156)

#### SAPO: Single-Rollout Autoregressive Policy Optimization for Agentic Reinforcement Learning (2026-08)
- **简介**：Dayang Liang, Lang Feng, Bo An, Yunlong Liu。critic-free 的 group-relative 方法从多条 rollout 估计 advantage，避开了 PPO 的显存开销，但存在三项局限：缺少显式 value 泛化与有效时序 credit assignment、长 horizon 复杂任务上 advantage collapse、以及采样预算与性能之间昂贵的取舍。**SAPO** 让 policy 与 value 函数共享同一个自回归骨干，利用 LLM 的自回归结构在不同因果边界上以共享参数产生 policy 与 value 预测，同时独立优化 PPO 目标与辅助的 on-policy SARSA 目标；为稳健估计每个 turn 的贡献，还引入结合 λ-return 与 batch normalization 的 trajectory 级 generalized advantage estimator。ALFWorld 与 WebShop 上以 Qwen2.5-1.5B/7B 训练稳定，平均比 PPO 和 GRPO 分别高 +15.1 和 +12.1 个百分点，同时消除独立 critic 模型的显存成本并把单次迭代运行时间相比 PPO 降低 33.2%。
- **arXiv**：[2608.19842](https://arxiv.org/abs/2608.19842)

#### MileGPO: Milestone Inference with Local Evidence for Graph-Based Policy Optimization of Long-Horizon LLM Agents (2026-08)
- **简介**：Bo Qian, Yuting Wu, Shuang Zeng, Huaiyu Wan 等。长 horizon agentic RL 常只有终局 reward，已有方法通过 step 分组或图式 advantage 估计把 trajectory 级信号细化为 step 级 credit，但会忽略有意义的中间里程碑。**MileGPO** 从分组 on-policy rollout 中导出过程级 credit，包含三项设计：Milestone Discovery 在成功 rollout 上识别候选 milestone、在失败 rollout 上识别反复出现的 trap；Reliability-Calibrated Shaping（RCS）按结果置信度为候选加权，强化可靠的 milestone 与 trap、压低不确定候选；Progress-Contrastive Calibration（PCC）进一步检验候选是否反映局部进展、其入边 transition 是否优于同一状态下观测到的其他分支。方法既不需辅助模型也不需额外环境交互，在 ALFWorld 与 WebShop 上取得 SOTA，且 ALFWorld 上分布内到分布外差距很小（abstract 未给出具体数值）；消融与 credit 诊断表明可靠性加权、局部进展与同状态分支证据能互补地消解含糊的中间 credit。
- **arXiv**：[2608.19803](https://arxiv.org/abs/2608.19803)

#### Temporal GRPO: Beyond Trajectory-Level Credit in Vision-Language-Action Reinforcement Learning (2026-08)
- **简介**：Yao Zhou、Hang Gao、Fengge Wu、Changwen Zheng、Wenwen Qiang。针对 GRPO 式 VLA post-training 中一个 rollout-level advantage 被平摊到 trajectory 内所有动作、使「完成了若干正确阶段但后期失败」的 rollout 反过来惩罚早期正确动作的问题（作者称之为 trajectory-level credit aliasing），**Temporal GRPO** 构造可检测的任务 stage，把每条 rollout 对齐到 stage 特定的动作区间，并只在进入同一 stage 的 rollout 之间做组内比较，得到的 stage advantage 在一次策略更新中分别作用于各自对应区间。在 RoboTwin 2.0 上提升任务成功率与样本效率，且在不同任务 horizon 上增益一致；LIBERO-Long 上的受控更新保留了共享的前置 stage，把改进集中在 rollout 结果首次分叉的那个 stage（abstract 未给出具体数值）。
- **arXiv**：[2608.13026](https://arxiv.org/abs/2608.13026)

#### TCPO: Turn-Level Credit Policy Optimization (2026-08)
- **简介**：Sicong Liao、Zhi Chen、Yaohua Tang。针对 verifier 引导的多轮 RL：每轮 verifier 分数虽稠密，却不等于稠密 credit——分数衡量当前输出质量，credit 应衡量当前 turn 如何改变后续 refinement trajectory。**TCPO** 把 credit assignment 建模为 score-to-credit 转换，以参考式比较构造 turn-level advantage：retrospective credit 相对历史最优状态刻画即时进步与回退，hindsight delayed credit 识别当轮未改进但后续有回报的 turn，selective fixed-history counterfactual estimation 在相同 history 下细化高 surprisal 的 turn。在数学推理、代码生成与 AppWorld 上优于或持平最强基线，Qwen3-4B 与 DeepSeek-R1-Distill-Llama-8B 上取得最佳或并列最佳 best-turn Pass@8，并减少成功所需 turn 数。
- **arXiv**：[2608.01667](https://arxiv.org/abs/2608.01667)

#### How Much, Then Where: Credit-Conserving Action-to-Token Allocation for Multi-Turn Agent Reinforcement Learning (FACTOR) (2026-08)
- **简介**：Lichao Ma、Yang Sun、Shuaitao Zhao 等 12 人。指出多轮 agent RL 的 credit assignment 实为两层决策：把 trajectory 级 credit 分给各 action（多少），以及把该 credit 铺到 token（分到哪）。**FACTOR** 将两者解耦——用 checkpoint 校准的 TD 残差给出逐 action credit，其求和可 telescope 回 trajectory advantage；再用 feedback-conditioned 的 teacher-student 似然差把 credit 分配到实际 action token，逐 action 归一化以避免 token 级符号翻转；配合 action-mean reduction 消除标量代理权重对 token 长度的隐式依赖。在 ALFWorld、WebShop、ScienceWorld 上每个 environment-seed 对比均胜过强基线，最长 horizon 环境增益最大；超参无需重调即可迁移到更大骨干与另一模型家族。
- **arXiv**：[2608.07118](https://arxiv.org/abs/2608.07118)

#### TAPO: Transition-Aware Policy Optimization for LLM Agents (TAPO) (2026-07)
- **简介**：来自北京大学（Cong Li、Peixi Peng 等 7 人）。指出现有 agent RL 主要依赖稀疏任务奖励，未充分利用在线交互中天然稠密的监督信号——动作执行后的环境反馈。受"多步目标任务的泛化依赖对环境后果的预测性知识"启发，提出 **TAPO**：在标准 RL 更新之外，复用 rollout 数据在共享骨干上施加 **action-conditioned next-observation 预测监督**，交替进行策略优化与转移监督，增强模型对环境转移动态/动作后果的敏感度；即插即用、无需额外专家数据/采样/推理开销。在 WebShop、ALFWorld 上跨多种规模模型与策略优化算法一致优于纯策略优化基线。
- **arXiv**：[2607.27973](https://arxiv.org/abs/2607.27973)

#### CAST: Game Solvers as Turn-Level Teachers for LLM Agents (CAST) (2026-07)
- **简介**：来自中科大 / 南京大学 / 美团（Yu Wang、Lan-Zhe Guo、Xunliang Cai、Han-Jia Ye、Fuli Feng 等 11 人）。长程游戏 RLVR 依赖稀疏终局奖励，难以揭示哪一步决定成败；稠密过程信号又难兼顾廉价与准确。作者观察到 **博弈求解器（game solver）状态价值的变化** 能揭示某动作是否推进状态趋向成功，据此提出 **CAST（Credit Assignment from Solver Teachers）**：把价值变化转为 solver advantage 注入 RLVR 作为 turn 级信号，并证明在 soft-optimal solver 假设下最大化 solver advantage 等价于对 solver 做在线策略蒸馏（仅需标量价值而非 teacher logits）。在 Sokoban、Minesweeper、Rush Hour 上于同域与未见难度均超过所有训练基线，并在 ALFWorld、WebShop 上取得最高零样本均分。
- **arXiv**：[2607.25308](https://arxiv.org/abs/2607.25308)

#### Hybrid Advantage Estimation with Unified Critic for VLM Agentic Reinforcement Learning (HyGAE) (2026-07)
- **简介**：来自 KAUST（Wenxuan Zhang、Jürgen Schmidhuber、Mohamed Elhoseiny 等 8 人，ECCV 2026）。针对 VLM 智能体多轮决策中"token 级优化（拼接轨迹）"与"turn 级优化（turn 内均匀信用）"两条路线各自的局限，先建立两级优化的理论表述并推导出同时服务二者的 **hybrid advantage**；进一步证明在适当折扣因子与学习目标下，一个 **unified critic** 可同时估计 turn 级与 token 级价值。据此提出 actor-critic 框架 **HyGAE**。五个多轮决策环境上平均成功率 91%，较其他方法提升 10%，并分析 hybrid advantage 与 return 的精确解析形式对优化至关重要。
- **arXiv**：[2607.23605](https://arxiv.org/abs/2607.23605)

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

#### Hindsight Credit Assignment over Trees for Agentic RL (TriGRPO) (2026-11)
- **简介**：作者 Penghong Zhao、Yihao Chen、Maoheng Yuan 等。长程 LLM agent RL 面临两难：成功经验对策略改进至关重要，但策略能力有限使其难以发现，稀疏结果奖励又使采样轨迹几乎没有相对反馈。**TriGRPO** 通过 rollout 树内部与跨树的 hindsight 信用分配，把稀缺的成功转化为探索与学习信号：hindsight 引导的分支在进行中的轨迹到达匹配的决策状态时，用 rollout 中发现的成功动作启动一个自主继续的兄弟分支，支持进一步探索与 parent-sibling 比较。三类互补分组组织策略优化——root group 捕获同 prompt 内的相对回报，node group 依据 parent-sibling 结果差异分配信用，cross group 用相关任务的历史表现补充稀疏反馈。实验表明在多种模型规模上任务表现与 rollout 效率均有一致提升（abstract 未给出具体数值）。
- **arXiv**：[2611.02706](https://arxiv.org/abs/2611.02706)

#### Toward Counterfactual Learning for Selective Re-Examination in Agentic Visual Search (CARE) (2026-10)
- **简介**：作者 Ao Chen、Shengchao Chen、Yan Luximon。高分辨率图像视觉搜索中，VLM 不仅要决定看哪里，还要决定是否需要再次观察后再提交答案；已有 prompting、示范与结果型 RL 难以直接监督「在某个搜索状态下重新检查是否有益」。作者提出 **ReSee** benchmark（3,227 条视觉搜索轨迹、8,827 个显式 commit-or-re-examine 决策，配反事实裁决以区分需要再检查与应直接提交的状态），并提出 **CARE**（Counterfactual Advantage for Re-Examination）：决策级 RL 方法，比较同一搜索状态下 commit 与 re-examine 两种 verdict 的配对后续，以其成本感知回报差估计额外观察的反事实价值，仅分配给对应的 verdict token，给出「是否继续搜索」的局部信用。实验显示现有 VLM 的重新检查选择性有限，CARE 在两种骨干上相对 GRPO 显著提升准确率与决策选择性，在六个未见 OOD benchmark 上准确率更高且视觉请求更少。
- **arXiv**：[2611.00848](https://arxiv.org/abs/2611.00848)

#### Action-Value Calibration via Continuation Sharing for Long-Horizon Agentic Reinforcement Learning (ConCal) (2026-10)
- **简介**：作者 Yingjian Chen、Lang Zhou 等。GiGPO 等方法比较共享状态下的动作来计算相对 advantage，但每个动作的价值只由单个采样 continuation 估计，局部信用易受后续决策影响。论文提出 **ConCal**（Continuation Calibration），一种递归 continuation-value 估计器：通过入射的状态-动作-状态转移匹配同一任务内的访问，把共享的 continuation 证据与每次访问自身的转移估计结合，并向后传播校准后的价值以修正早期动作价值，无需额外 rollout。held-out 诊断显示组内价值差误差更低、与参考动作排序更一致。以 Qwen2.5-1.5B-Instruct 为模型，在 ALFWorld 和 WebShop 上成功率比 GiGPO 分别提高 3.39 和 5.86 个百分点。
- **arXiv**：[2610.37195](https://arxiv.org/abs/2610.37195)

#### ReflectPO: Reflection-Guided Policy Optimization for Long-Horizon LLM Agents (2026-10)
- **简介**：作者 Rumei Li、Xunliang Cai。长程任务中早期执行错误会影响后续交互历史与环境状态，并沿决策链累积。论文提出 **ReflectPO**，面向长程任务自主错误恢复的 RL 框架，含两部分：Reflection-Guided Sampling——利用交互轨迹中的分支差异与重复访问构造错误恢复提示；Tree-Structured Credit Assignment——把采样轨迹组织成树，用父子节点的 reward 差异与 recovery gain 分配动作级 credit。在两个长程工具使用 benchmark 上，大多数 domain 优于多个已有基线并取得最佳平均表现，进一步分析表明 agent 识别自身偏离、从错误状态恢复并继续推进任务的能力得到提升。（abstract 未给出具体数值）
- **arXiv**：[2610.30933](https://arxiv.org/abs/2610.30933)

#### TrajEvolve: Trajectory Evolution for Reinforcement Learning with Hindsight Credit Assignment (2026-10)
- **简介**：作者 Yi Wen、Hao Chen 等（NeurIPS 2026 接收）。RL 用于长程任务时面临两个问题：任务复杂导致难以采样到足够成功轨迹、收敛慢；仅靠稀疏终局 reward 会让所有步骤被无差别更新、鼓励冗余步骤。其根源是现有 RL 范式完全依赖基座模型的推理能力、缺乏主动优化轨迹的能力。论文提出轨迹演化范式 **TrajEvolve**：除模型自身推理轨迹外，还通过精炼历史轨迹中的低质量步骤生成「演化轨迹」，让策略接触到落在基座支持范围内但在实际训练预算下很少被采样的高质量组合；并把步级连续 credit assignment 重构为二分类任务——每个步骤是否为完成任务所必需，从而通过反事实验证获得步级标签、精确识别冗余步骤。在两个数据集上的大量实验证明方法优越。（abstract 未给出具体数值）
- **arXiv**：[2610.28060](https://arxiv.org/abs/2610.28060)

#### Distill Until Divergence: Error-Truncated Hindsight Distillation for Agentic RL (2026-10)
- **简介**：作者 Yibin Huang、Conghui Zhu。长程交互中，outcome-based GRPO 缺少对中间决策的细粒度指导；近期方法把完成的轨迹蒸馏为可复用 skill，并用 hindsight 条件化的 OPD 提供稠密 token 级 credit，但 skill 引起的概率偏移未必对应奖励对齐的 credit，尤其在错误动作改变后续交互状态后，skill 条件化监督会与当前状态错配。论文提出一种按轨迹结果与错误边界路由两类 hindsight skill 的方法：Positive skill 记录成功流程，强化成功轨迹与出错前的决策；Negative skill 记录失败特有的修正，针对决定性错误；出错之后的后缀仅用 GRPO 优化，保留 outcome 学习而不延伸可能错配的 hindsight 监督。实验显示任务表现有竞争力并能有效分配细粒度 credit（abstract 未给出具体数值）。
- **arXiv**：[2610.23048](https://arxiv.org/abs/2610.23048)

#### Over-Defense in Coding Agents Begins at Silent Forks (ForkDPO) (2026-10)
- **简介**：作者 Hanjun Luo、Xuecheng Liu、Xiucheng Zhang 等。coding agent 用 outcome 奖励后训练，近期方法把 credit 集中在决定结果的 turn，但超出任务所需的防御性动作二者都漏掉：它们出现在「silent fork」——所有下一步动作都导向同一结果的 turn，成功率不变，前沿模型在隐式预算下于多达 91% 的决定性动作处走向不成比例的分支。直接在轨迹奖励中加 checker 只能经由完全合比例的轨迹触及这些 fork，使 74% 的 GRPO group 奖励相同、行为不变。因此提出 **ForkDPO**：在同一前缀处对比合比例与不合比例的动作，在所需行为两侧的 silent fork 上构造偏好对，把「行为分歧处」作为学习信号。在未见仓库的测试集上，ForkDPO 将 4B 模型的不成比例行为从 70.3% 降至 58.9%，而在随机 turn 上做 DPO 为 67.7%。代码已开源。
- **arXiv**：[2610.24949](https://arxiv.org/abs/2610.24949)

#### SERO: Learning from Executable Oracle Continuations for Long-Horizon Agentic Reinforcement Learning (2026-10)
- **简介**：作者 Dingyi Rong、Yue Shi、Chaofan Ma、Yixin Cao 等。长 horizon agentic RL 受限于当前策略访问到的未来：rollout 到不了有用行为时，再好的信用分配也无法监督从未遇到的状态。**SERO（State-Rooted Executable Rollout Optimization）** 恢复策略实际到达的某个状态，让仅训练期使用的 oracle 在真实环境中继续交互，使其后续观测与 oracle 动作本身成为训练轨迹；并引入 policy-reachability weighting，按冻结 behavior policy 对此前 oracle 动作的兼容程度给每个 oracle 步加权。oracle 损失与基础 RL 估计器分开优化，推理时不使用 oracle。在 ALFWorld 和 WebShop（Qwen2.5-1.5B/7B-Instruct）上，学习完整的 executable continuation 远优于只监督 root oracle 动作，reachability 加权进一步提升 OOD 表现（abstract 未给出具体数值）。
- **arXiv**：[2610.19397](https://arxiv.org/abs/2610.19397)

#### TreeHCA: Hindsight-Guided Exploration and Credit Assignment for Multi-Turn LLM Agents (2026-10)
- **简介**：作者 Debajoy Mukherjee、Jeremy Carleton、Srinivas Shakkottai、Dileep Kalathil 等。GRPO 对 agent 轨迹中每个 token 分配相同 advantage；Hindsight Credit Assignment（HCA）可以细化反馈，但其估计依赖于在关键历史处含有信息性备选的轨迹。**TreeHCA** 将 hindsight 引导的探索与 hindsight 信用分配耦合：用参考答案似然的变化回访中间历史并采样备选 continuation，形成推理与工具交互的树；针对探索改变采样分布，推导出相应的 inverse-hindsight importance 修正（含精确的有限池恒等式），再通过归一化 tree backup 把终端结果传播为 turn 级 advantage。在七个问答基准与三个 Qwen 骨干上，TreeHCA 的整体 exact match 优于所有基线。
- **arXiv**：[2610.12828](https://arxiv.org/abs/2610.12828)

#### CASI: Credit Assignment via Set Intervention for Agentic Retrieval (2026-10)
- **简介**：作者 Xinyi Cheng、Tianfu Zhang、Jiwen Li、Xuesen Zhang、Shihao Yu、Pengfei Zhu。LLM 越来越多地联合生成多个检索动作，而下游系统只给整个集合一个奖励；广播该奖励虽是集合级目标的有效 score-function 梯度估计，却无法确定依赖联盟的集合内归因。**CASI** 通过 snapshot-grounded set intervention 构造缺失的反事实信息：一次生成与检索，固定可重放的快照组件，在相同条件下评估备选子集；再用 Shapley 聚合把这些边际转为集合内信用，并通过有界零和 advantage shaping 纳入策略优化。在五动作的赞助搜索场景下，对 1,200 个请求做穷举拍卖重放揭示出显著的联盟依赖；5 次独立训练中，CASI 训练的策略在留出拍卖重放集合价值上高于 Singleton 与 Broadcast 训练的策略；在 10% 流量、为期七天的随机线上 A/B 测试中，主要广告主价值指标较生产对照提升 3.51%。作者也如实指出在一个敏感性后端下 CASI 并未优于 Singleton。
- **arXiv**：[2610.20293](https://arxiv.org/abs/2610.20293)

#### Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight (SRD) (2026-10)
- **简介**：作者 Haoxiang Zhang、Qinglin Chen、Hiroaki Hayashi、Zhuofeng Li、Siming Zhang、Jiaxin Zhang。RLVR 主要通过轨迹结束后的标量结果奖励学习，对 group-relative 目标，当所有 rollout 奖励相同时信号消失。论文提出 prospective learning：用事后经验监督「行动前」的 foresight 预测，并实例化为 **SRD**（Self-Retrospection Distillation），把完成轨迹揭示的特权 hindsight 蒸馏进同一策略的轨迹盲 foresight，推理时无需显式生成。在 10 个工具集成推理和长程 agentic 任务上与 RLVR 及 self-distillation 基线互补，增益最高 24.2 pp；在奖励无对比的场景尤其明显——2B 设置下 98% 的 group 全失败，纯 RLVR 训练成功率为 0.0%，加入 SRD 后同等 rollout 下达到 60.6%。
- **arXiv**：[2610.08077](https://arxiv.org/abs/2610.08077)

#### Environmental Feedback Modeling Matters: Rethinking Feedback Treatment in Agentic Hindsight Self-Distillation (SELF) (2026-10)
- **简介**：作者 Hangxi Guo、Fengyuan Liu、Yue Wang、Yuhua Qi、Haoyi Xiong、Fei Sun。奖励不可得时无法直接对语言 agent 做 RL，近期方法把环境反馈作为特权上下文做 hindsight self-distillation，但分析表明仅让 teacher 以反馈为条件并不够。**SELF**（agentic SElf-distilLation with environmental Feedback modeling）联合优化环境反馈建模与 hindsight self-distillation：学习预测环境响应，同时把反馈条件化 self-teacher 的引导蒸馏进策略，二者相互增强。使用 Qwen3-8B，SELF 在 tau-bench 成功率上比 SDPO 和 GRPO 分别高 6.4 和 4.1 个百分点，在 AppWorld 任务目标完成率上分别高 10.71 和 3.57 个百分点。
- **arXiv**：[2610.11384](https://arxiv.org/abs/2610.11384)

#### RELACE: retrospective likelihood-based action credit estimation for long-horizon language agents (2026-10)
- **简介**：作者 Sayak Chakrabarti、Sathish Reddy Indurthi。多轮 agent 的轨迹级监督信用粗糙且噪声大，终端奖励无法定位错误；GiGPO 等方法通过状态条件比较细化监督，但信用估计仍对后续决策与结果敏感。**RELACE** 是 critic-free 框架，把回顾式动作评估与状态条件 advantage 估计结合：通过 teacher-forced likelihood 打分，在动作原始上下文与融入结果的上下文下分别评估已执行动作，比较两种似然得到轨迹归一化的回顾因子，刻画「依赖结果的动作合理性变化」而非单纯事后合理性。用该因子重加权折扣回报，并在同一任务内的等价状态动作之间比较加权回报构造局部 advantage，与轨迹级 GRPO 监督互补；再配合时间平滑与保护成功的 masking 稳定局部信号。（abstract 被截断，实验数字以原文为准）
- **arXiv**：[2610.07349](https://arxiv.org/abs/2610.07349)

#### CIPO: Counterfactual Imagination Policy Optimization for Adaptive Tool Granularity Selection (2026-10)
- **简介**：作者 Yu Li、Yunlu Wan、Zijian Zhu、Han Luo、Chao Ren、Long-Fei Li。LLM agent 的多步工具交互常含重复的局部工具序列，把它们当作复合「Skill」可缩短轨迹，但原子工具与复合 skill 并存时，agent 需判断当前状态需要原子精细控制还是 skill 级抽象。论文把 skill 使用视为自适应工具粒度选择，其最直接的训练信号是比较同一状态下原子与 skill 选择的后果。**CIPO**（Counterfactual Imagination Policy Optimization）通过预算约束地挖掘成功工具使用轨迹来构造可执行 skill，并用反事实分支 rollout 训练粒度决策：对每条基础 rollout，在第一个可选粒度决策处分支，把所选动作替换为可行的原子或 skill 备选，成对的结果差作为策略优化的补充奖励。（abstract 被截断，实验数字以原文为准）
- **arXiv**：[2610.04991](https://arxiv.org/abs/2610.04991)

#### Targeting Pivotal Decisions for Credit Assignment in Agentic Reinforcement Learning (ProVer) (2026-09)
- **简介**：作者 Dongwon Jung、Yifan Wang、Yuexing Hao 等指出 GRPO 把轨迹级 advantage 均匀分给所有 policy token，无法区分关键决策与无关决策。提出 **ProVer**：给定一组 rollout，agentic judge 对比成功与失败轨迹，提出可能导致结果分歧的 segment；但不直接信任 judge，而是用「该 segment 前后由当前策略采样的后续」的终局成功率之差估计其 advantage 来加以验证，正估计再被并入该 segment 内 policy token 的 GRPO advantage。模型判断仅用于选择验证位置，无需穷举评估所有中间状态。在 ALFWorld、WebShop、SearchQA 上，ProVer 在两个模型规模上均取得最佳平均性能，相对 GRPO 在 Qwen3.5-2B/4B 上分别提升 9.91% 和 7.12%，且额外生成开销不大。
- **arXiv**：[2609.36178](https://arxiv.org/abs/2609.36178)

#### GraphHCA: Closed-Form Hindsight Credit Assignment for Long-Horizon LLM Agents (2026-09)
- **简介**：作者 Haodong Zhu、Yangyang Ren 等关注稀疏终局 reward 下 agent 的步级 credit assignment。Hindsight credit assignment（HCA）通过 hindsight 与 behavior-policy 概率之比归因，但估计 hindsight 分布需要辅助模型或额外前向。**GraphHCA** 为 model-free 实现：在确定性转移的终局目标任务中，由 Bayes 规则将 hindsight 比化为相邻状态上 behavior-policy 成功概率之比，取对数得到状态级 success potential，其跨转移的增量即为步级 credit；该 potential 由池化 rollout 在诱导转移图上通过折扣递推估计，在任意有向图上有唯一不动点。步级信号与轨迹级 advantage 结合，无需学习 hindsight 模型或额外前向，权重为零时退化为 GRPO。在 ALFWorld、WebShop 及 Sokoban（视觉-语言 agent）上取得最优，ALFWorld 总成功率较 GRPO 最多提升 24.6 点，较最强步级基线最多提升 4.7 点。
- **arXiv**：[2609.35084](https://arxiv.org/abs/2609.35084)

#### Counterfactual Rollout Replay: Forkable Environments as Free Process Rewards for Software Engineering Agents (CRR) (2026-09)
- **简介**：作者 Yuanhao Li、Hongbo Wang 等（NeurIPS 2026 接收）针对 SWE agent 仅有终局信号、缺少中间决策指导的问题，提出 **Counterfactual Rollout Replay（CRR）**：利用可 fork 的可执行环境获得步级回报对比。CRR 选取少量决策点，恢复状态，采样一个替代动作并按策略向前 rollout；保留实际训练轨迹，并把选定步的 advantage 替换为其终局回报与采样的反事实回报之差。无需人工过程标注或学习的 PRM（「free」指免除监督成本，而非回放计算）。14B 策略下，CRR 提升 SWE-bench Verified、SWE-bench Live、SWE-rebench 的 pass@1，并可与过程奖励和轨迹搜索方法组合；在 SWE-bench Verified 上等 wall-clock 对比为 41.7% 对 36.7%（扩展的纯结果 GRPO），含 fork 开销仍提升 5.0 点。局限在于需要状态恢复廉价可靠的环境。
- **arXiv**：[2609.33875](https://arxiv.org/abs/2609.33875)

#### ASCT: Attentive Search over Counterfactual Trees for Credit Assignment in Agentic Reinforcement Learning (2026-09)
- **简介**：作者 Yang Li、Jinhan Yang 等提出 **ASCT**，把训练时的多步搜索转化为局部动作 credit：在 actor 访问的状态上，辅助树从同一可恢复前缀评估备选合法动作，其动作值表以冻结 actor 的概率为中心化，作为 PPO 在 actor 采样轨迹上的 credit，部署时仅用 actor。Uniform、UCT 与成本感知的 AgentUCT 是该框架的三种实例。在 HotpotQA agentic RAG 上，三者的平均 held-out utility 均优于轨迹回报 PPO 和适配 workflow 的 VinePPO；三个 seed 下 ASCT-AgentUCT 达到 0.6187 utility（VinePPO 为 0.5939），answer F1 与执行成本也有改善，并少用 50.3% 记录的辅助 Qwen token。
- **arXiv**：[2609.35215](https://arxiv.org/abs/2609.35215)

#### CREDO: Variance-Guided Rubric Evolution for Replay-Corrected Credit Assignment (2026-09)
- **简介**：作者 Xuchun Hu。长程语言 agent 只有稀疏终态反馈，中间 rubric 虽有结构但可能设定不当；在可重置环境中反事实续写 rollout 能衡量局部 credit，但全量 replay 代价高。**CREDO** 将演化的语义 rubric 与选择性、基于执行的 credit 修正结合：冻结 judge 把可见转移映射到 rubric 特征，credit head 预测该转移带来的期望终态奖励变化，再用按记录的包含概率独立采样的双侧 replay 修正预测残差。论文推导了条件无偏性与方差分解，说明保留哪些 rubric 特征与如何分配固定的期望 replay 预算，并描述与终态 leave-one-out advantage 的混合实现。这是初步报告，提供方法、证明与受控评估协议，明确不声称在 language-agent 基准上的经验优势。
- **arXiv**：[2609.24174](https://arxiv.org/abs/2609.24174)

#### PCQC: Privileged Counterfactual Question Credit for Multi-Turn Medical Dialogue (2026-09)
- **简介**：作者 Chenxuan Li、Jiayi Wan、Xinrong Chen 等。医疗对话需要学会提问以获取病人信息，但基于最终诊断正确性的结果监督无法区分各个问题的贡献，也不能给未执行的替代问题提供反馈。**PCQC** 在训练时利用特权病人信息，为同一对话状态下的替代问题构造答案使其可直接比较，并由冻结的诊断打分器按其对正确诊断的支持程度评估每个问答对的诊断效用，转化为相对的问题 credit，同时监督已执行与未执行的问题，无需为未执行的替代项做完整 rollout。在四个医疗基准上平均诊断准确率达 63.10%，分别比 GRPO、ATPO 高 4.38、4.21 个百分点，且问诊轮数比 GRPO 少 33.1%。
- **arXiv**：[2609.27987](https://arxiv.org/abs/2609.27987)

#### When and How Should an Agent Clarify? CIGAsk: Teaching LLMs to Clarify via Counterfactual Information Gain (2026-09)
- **简介**：作者 Yunxiang Li、Xixin Wu、Helen Meng（EMNLP 2026 Findings）。指令微调 LLM 面对欠明确的 query 常直接选定一种解释而不澄清，仅靠 prompt 不足以解决。需要同时学会何时问和如何问。**CIGAsk** 在多轮 GRPO 循环中用两种互补奖励教授这两项技能：Counterfactual Information Gain (CIG) 比较冻结参考模型在有/无用户回答时对金标准答案的 log-likelihood，提供逐 turn credit 指导「如何问」；Asymmetric Ambiguity Bonus 依据金标准歧义标签在终端 token 给出带符号奖励，指导「何时问」。在表格、段落、开放域 QA 三类澄清基准上，CIGAsk-7B 优于最强外部基线（后者使用更大的 backbone），可跨数据集迁移且保持分布外单轮 QA 性能。
- **arXiv**：[2609.24290](https://arxiv.org/abs/2609.24290)

#### AgentBrew: Offline Tool-Use Agent Learning from Raw Real-World Trajectories (2026-09)
- **简介**：Zhiyi Lyu、Yewen Li、Longtao Zheng、Bo An 等。真实应用中训练工具 agent 缺少预定义任务与 verifier、没有可信模拟器且交互预算有限，**AgentBrew** 因此提出完全离线的框架：先让 agent 在目标环境中无质量过滤地探索出原始 trajectory 语料，再用 retrospective task inference 依据实际结果为每条 trajectory 反向重构对齐的指令，并用基于点互信息（PMI）的 credit assignment 把整条 trajectory 关于该指令的总信息量分解为逐动作可加 credit，用其加权 policy 训练目标以放大有效动作、压制无效动作，全程不需要 task verifier 或 on-policy rollout。在 GitHub、Notion、PostgreSQL 三个真实 MCP 应用上，AgentBrew 使 Qwen3-32B 平均提升 +8.7 Acc / +9.7 Score，超过 Qwen3-235B（+2.3 / +4.4）与 rejection sampling（+5.9 / +10.3）。
- **arXiv**：[2609.05837](https://arxiv.org/abs/2609.05837)

#### Hindsight Memory-PRM: Supervising Memory Management with Auditable Hindsight Credit (2026-08)
- **简介**：Haoxuan Jia、Yang Liu、Yingguang Yang、Yancheng Chen 等（含 Philip S. Yu）。长 horizon LLM agent 的记忆操作在执行当时价值不可观测、极难监督，但它们会在 trajectory 中留下机器可读证据：retrieval 命中与回答时的 citation。**Hindsight Memory-PRM** 两次利用这条审计链——离线用它训练一个以操作为条件的 memory-utility critic；在线则用 retrieval、citation 以及每个 probe 一次受控的「删除后重答」干预，结算经干预校准的条目级 presence credit，并沿版本链传播为 action 级代理 reward，因此既不需要逐操作人工标注，也不需要对后续续写做 Monte-Carlo 回放。在 held-out LoCoMo 上，固定共享 reader 条件下 8B 本地 policy 达 77.5%，超过其 API 教师（65.1%）与所有复现的外部系统，且上下文仅为 Mem0 官方工作点的 1/8；LongMemEval 上 79.0%。消融把增益归于因果校准而非信号密度，policy 还收敛到一种多版本记忆组织方式，是所有测试的开环基线都复现不出的。
- **arXiv**：[2608.29605](https://arxiv.org/abs/2608.29605)

#### TRACER: Per-Tool Context Retention for LLM Agents via Consequence-Attributed Reinforcement Learning (2026-08)
- **简介**：Ziqi Lin、Ye Wu、Mengying Yang、Xu Liu 等。企业数据 agent 单次会话常累积数十万 context token，而既有压缩策略分配保留预算时不考虑删除单个工具输出的下游后果，激进压缩反而触发昂贵的工具重调用抵消收益（作者称之为 compression–consequence gap）。**TRACER** 把压缩建模为逐工具的序贯决策问题：一个轻量 REINFORCE policy 仅用每个压缩时刻可得的信息给出以 query 为条件的保留比例，其后果感知目标同时计入任务成功、总 token 消耗与压缩后的工具重调用；为改善 credit assignment，还用一个学习到的 outcome model 对比「所选保留比例」与「完整保留该工具输出」的预测后果（即反事实比较）。在三种压缩后端的 held-out 生产 query 上，TRACER 相对全量保留降低 29–46% 的 token 消耗且任务成功率持平或更高，比按工具类型的静态策略再省 15–18%；干预式 rollout 显示学到的 per-tool credit 分数与实测单工具后果相关，policy 跨 agent backbone 与压缩架构迁移后仍有正收益，在五个 held-out LOCA-bench 环境上降低 18–25%。
- **arXiv**：[2608.29363](https://arxiv.org/abs/2608.29363)

#### AHEAD: Adaptive Hindsight with Environment-Augmented Distillation for Agentic RL (2026-08)
- **简介**：Xiaolong Jin, Dingmin Wang, Vijay Lingam, Varun Kumar。多轮 agent 的 RL 训练通常依赖 trajectory 级 reward，给每一步统一 advantage，无法辨别哪些决策导致成功或失败；自蒸馏类方法虽能用特权信息提供更细监督，却对所有步施加同一类特权信息，忽略了一个关键非对称性：常规步几乎不需要额外指导，而关键错误步需要环境反馈本身给不出的纠正方向。**AHEAD** 是 step-aware 框架，按步类型匹配不同监督源：teacher 在所有步接收环境反馈作为有据可依的稠密信号，并额外在错误步接收 LLM 生成的纠正提示以补足方向信息，对标准 GRPO 只需极小改动。在 ALFWorld、WebShop 与检索式 QA、三种模型规模上，7B 相对 GRPO 在 ALFWorld 成功率 +13.3 个点、WebShop +11.0 个点，并能以更少训练步达到给定成功率、在更紧的交互预算内完成任务。
- **arXiv**：[2608.24114](https://arxiv.org/abs/2608.24114)

#### Credit Without Ground Truth: Auditing Step-Level Credit Assignment in LLM Agents Against Executed Replay (2026-08)
- **简介**：Haiyue Zhang（单作者）。作者在 ALFWorld 单智能体工具环境中以「执行式 replay」得到的因果 ground truth 为基准，审计用于训练 LLM agent 的各类 step 级 credit 信号——LLM-judge 分数、outcome-conditioned logprob ratio、policy 自身置信度——发现没有一个在识别「哪些 step 因果上重要」上优于随机猜测；已有评测衡量的是 step 的正确性，本文改为衡量 step 的贡献（在每个决策点重采样 policy 自己的备选动作并向前 roll out，看结果实际改变了什么），两者并不一致。ground truth 本身高度结构化：因果贡献稀疏（有定义的决策点中仅 30.5% 存在可测效应），可测性还依赖模型（无 policy-supported counterfactual 的比例在两个同量级 policy 间相差两倍，13.1% vs 26.8%）。失效模式可辨识：隐式 credit 主要回声 policy 的流畅度（中位秩相关 +0.75，在另一模型族以校正后仪器复现为 +0.70），而对结果取条件并未增加因果信息（偏相关 −0.004，Qwen）；仅用置信度的 router 只能以随机水平召回关键 step，但能把 judge 成本按 turn 降 13.1%、按 trajectory 降 14.0%。七臂预注册训练实验中没有任何一臂稳定超过未训练 policy，checkpoint 表面上的仪器特征完全由训练剂量解释（credit 越稀疏保留样本越少，optimizer step 数相差一个量级），作者据此主张 credit 规则的比较必须匹配有效样本量，否则测到的是剂量而非 credit。属对 step/turn 级 credit assignment 的审计/诊断工作。
- **arXiv**：[2608.19760](https://arxiv.org/abs/2608.19760)

#### Trajectory-Relative Hindsight Distillation for Agentic Reinforcement Learning (TRIAL) (2026-08)
- **简介**：Haoyu Zheng、Yun Zhu、Qing Wang、Wenqiao Zhang。针对「一条 rollout 可产生大量 hindsight 信号、如何跨 turn 分配不明确」的问题，提出 **TRIAL** 与统一的 turn-aligned 打分协议：对每个决策 turn 抽取其实际后果的 outcome view，并在普通与 hindsight-conditioned 两种 context 下重评同一 response，用带符号的对数概率差决定 token 级监督的方向与局部强度；turn 级幅度在整条 trajectory 上联合归一化，使分配乘子的 eligible-token 加权均值恒为 1，即在固定平均强度下重新分布稠密监督。WebShop 与 ALFWorld 上 8 组骨干×环境×指标组合全部优于 GRPO，6 组在六种方法中最佳或并列最佳；WebShop + Qwen3-1.7B 成功率 56.4%→75.2%、task score 78.7→85.7。
- **arXiv**：[2608.07371](https://arxiv.org/abs/2608.07371)

#### TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning (2026-08)
- **简介**：Changle Qu、Sunhao Dai、Hengyi Cai、Jun Xu 等。指出 Tool-Integrated Reasoning（TIR）的 RL 多依赖 trajectory 级监督，长 horizon 下 credit assignment 过粗；已有 on-policy 自蒸馏虽用特权 context 提供更稠密信号，但该 context 常取自 ground-truth 答案或检索 skill，未必反映 agent 实际访问的状态，且 token 级监督无法刻画工具交互的 turn 结构。**TurnSight** 直接从 execution-conditioned hindsight 导出监督：构造多个不同 lookahead horizon 的 hindsight view，通过跨 horizon 方向一致性筛出可靠信号，再在 sibling rollout 间归一化，用于自适应调制 RL advantage 且保持其原始优化方向。三个基准上验证有效，代码已开源。
- **arXiv**：[2608.04007](https://arxiv.org/abs/2608.04007)

#### Group-Reflective Self-Distillation for Agentic Reinforcement Learning (GRSD) (2026-07)
- **简介**：来自 Binbin Zheng、Zijun Xie 等 7 人。RLVR 的终局奖励只给粗粒度轨迹级监督，把"成功行为、重复错误、偶然选择"纠缠在同一信号里；已有 agentic 自蒸馏用自然语言 skill 补充稀疏监督，但外部检索或由更强模型从单条轨迹抽取的 skill 可能与当前经验不匹配、超出策略能力或过度路径特定。**GRSD** 从策略自身已验证的 rollout 中导出"能力对齐且结果可判别"的指导：对每个 prompt，策略对在线组内每条已验证轨迹做反思，用 stop-gradient 快照对比成功/失败 rollout 的反思，构造 **组级特权指导**；以此为条件的 self-teacher 通过调制基于结果的优势来精化 **turn 级信用分配**，同时保持 verifier 决定的学习方向。多个 agentic 环境与模型规模上一致优于强基线，对未见任务泛化更好。
- **arXiv**：[2607.28076](https://arxiv.org/abs/2607.28076)

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

#### NorthCUA Technical Report: Turbocharging Computer-Use Agents through Experience at Scale (2026-11)
- **简介**：作者 Chaoyun Zhang、Bingxuan Li、Pinxue Guo 等。NorthCUA 是把 computer-use agent 从单步动作扩展到复杂软件工作的整套系统：**Experience Flywheel** 为 NorthCUA Harness 提供多样的可执行任务，Harness 维持长交互、记录决策并检查任务结果，Flywheel 据此整理已验证轨迹；对未解决任务，递归自我改进会演化可复用 skill 并保留已验证的里程碑，使后续尝试在前序进展上继续。Staged Learning 将这些经验与 grounding 数据结合，经 mid-training 与长程 SFT 培养 GUI 能力，随后策略在同一 Harness 中通过桌面+网页联合 RL 从新尝试中学习，闭合「经验生产—策略改进」回路。得到的 Nano/Mini/Pro 系列在所报告设置下，于 GUI grounding、桌面执行与 Web 交互上达到开源模型 SOTA；NorthCUA-Ultrafast 另提供质量-延迟权衡。
- **arXiv**：[2611.02102](https://arxiv.org/abs/2611.02102)

#### CUA-Master: Scaling Computer-use Agents through Open-World Recursive Improvements (2026-11)
- **简介**：作者 Xinlei Yu、Gen Li、Ji Zhao 等。CUA 的发展受限于交互数据的多样性、可控性与可靠性，常规优化流水线缺少迭代反馈，现有递归改进方法又多依赖自生成经验而收益递减。**CUA-Master** 是嵌套双循环递归改进框架：系统层外循环为 world-data-model flywheel，持续引入开放世界语料并利用模型的演化弱点指导后续数据构造与优化迭代；数据层内循环耦合开放世界 seeding 与自催化传播——前者从教学视频中挖掘真实工作流作为种子任务并做视觉-语义双路径轨迹重建，后者把种子转化为可复用原语，可控地合成多样、复杂且可验证的任务；另有反馈引导的 snapshot-rollback 重试机制以高效可靠地采集轨迹。CUA-Master-27B 在 OSWorld Verified 得 80.2、OSWorld 2.0 得 36.9、WeaveBench 得 43.9、Agents' Last Exam 得 25.0。
- **arXiv**：[2611.03126](https://arxiv.org/abs/2611.03126)

#### WebHarbor: Docking Real Websites for Evolving GUI Agent Environments (2026-11)
- **简介**：作者 Zhaoyang Wang、Qianhui Wu、Boyu Gou 等。GUI web agent 需要在评测与训练中都稳定且真实的交互环境：线上网站评测受内容变化、访问限制与认证流程覆盖不足的影响，现有合成训练环境又缺乏功能深度与视觉保真度。**WebHarbor** 是由 153 个可执行、docker 化网站环境构成的套件，在确定性控制下忠实重建真实网络服务的视觉界面、内容与功能工作流；通过任务驱动的人机协作构建——环境随新任务扩展功能范围、由 coding agent 实现改动、人类审核视觉保真度、功能正确性与任务有效性——共包含 3,518 个覆盖多步与登录后工作流的任务。WebHarbor 同时可作为高保真合成环境用于低成本 RL：在其中用 RL 训练的 Qwen 4B 级 GUI agent 能泛化到未见任务，并在四个 benchmark 上与领先开源 GUI web agent 竞争。
- **arXiv**：[2611.03634](https://arxiv.org/abs/2611.03634)

#### BiGrounder: Bidirectional Training Unlocks the Potential of GUI Grounding (2026-11)
- **简介**：作者 Dengyao Shi、Tianyu Fu、Fei Hu 等。GUI grounding 通常只用最终坐标做端到端监督，且每个目标坐标对应的描述数量很少，限制了对语言变化的鲁棒性；而 RL 阶段前探索能力有限的模型，即使 RL 数据相同提升也受限。作者提出**双向奖励范式**使模型自我强化：SFT 阶段的双向训练让模型能从目标坐标生成动作描述；RL 阶段模型 on-policy 地采样语义一致的指令变体并把每个变体 grounding 回目标，由 cycle-consistency gate 把语义质量与 grounding 成功耦合，使模型自己的指令策略成为持续刷新强化轨迹的来源。其 4B 模型在评估的五个 grounding benchmark 上取得有竞争力的准确率（abstract 摘录未给出完整数值）。
- **arXiv**：[2611.07765](https://arxiv.org/abs/2611.07765)

#### From Local Skills to Long-Horizon Web Tasks: Progressive Skill Exploration for LLM Web Agents (SCOPE) (2026-11)
- **简介**：论文被 NeurIPS 2026 接收，作者 Xueyang Feng、Xiaohe Bo、Xu Chen 等。朴素探索下 Web agent 训练困难：成功轨迹稀少，稀疏终端奖励使失败轨迹难以提供有用反馈。受技能习得理论启发，**SCOPE**（Skill-Composed Progressive Exploration）是短到长的学习框架，把复杂任务探索分解为从局部技能到完整任务求解能力的渐进扩展：Skill Discovery 从历史 rollout 中提取可复用局部技能与失败经验；Skill Orchestration 把技能组合为高层技能链；Skill Expansion 利用与当前技能链对齐的失败经验引导 agent 探索新的子技能。随后通过 agentic on-policy distillation 把逐步构建出的技能链内化到策略参数中。在 WebArena、WebVoyager、WebShop 上，相对各设置下最强基线取得 3.9% 至 18.6% 的相对提升。
- **arXiv**：[2611.03116](https://arxiv.org/abs/2611.03116)

#### Online Dense Reinforcement Learning with Planning-Grounding Agent for UI Navigation on Live IOS Apps (GiGPPO) (2026-10)
- **简介**：作者 Peitong Duan、Khashayar Gatmiry、Jeffrey Nichols、Samy Bengio、Emmanuel Abbe。前沿 VLM 做 UI 导航推理成本高，而 SFT 或离线 RL 训练的小模型易记忆静态轨迹、难以从错误中恢复。论文提出在真实 iOS 应用上直接在线训练 UI 导航 agent 的 RL 框架：将 agent 解耦为 Planning agent 与 UI Grounding 模型，用 LLM-as-a-Judge 给出稠密的步级反馈加终止奖励，用带步级 advantage 的 PPO 优化，并提出 **GiGPPO**——利用并发轨迹中视觉相似状态的奖励信号做 neighbor-conditioned 价值估计以稳定 critic，配套可并发数百台虚拟 iOS 设备的基础设施。在 UINavBench、iOSWorld、AndroidWorld 上显著优于 SFT 与仅终止奖励 RL；8B RL agent 在 Android 的 OOD 泛化上相对 SFT 提升近七倍，在真实 iOS 应用上超过 Gemini-3.5-Flash、与 Gemini-3.1-Pro 持平。
- **arXiv**：[2610.39264](https://arxiv.org/abs/2610.39264)

#### Where to Learn and How Far to Roll Out: Efficient Adaptive Training for GUI Agents (2026-10)
- **简介**：作者 Yuxiang Chai、Quyu Kong、Steven Hoi、Yue Wang 等。轨迹级 GRPO 训练多步 GUI agent 时，反复从任务起点完整 rollout 效率低：任务级奖励难以指出哪些中间决策需要改进，已掌握的前缀还被重复执行。论文提出两阶段自适应训练框架：先从成功的专家轨迹构造可复用的中间 checkpoint，并用 action disagreement 选取信息量大的 pivot；阶段 I 在这些 pivot 做 one-step RL；阶段 II 按任务进度自适应地在 Local、Suffix、Root 间逐步扩大策略责任范围，借助可恢复的 checkpoint 无需重放前序轨迹。在 MobileGym 和 CUA-Gym-Hub 上相对 vanilla GRPO 减少 56%-61% 的环境交互，同时提升任务表现。
- **arXiv**：[2610.39251](https://arxiv.org/abs/2610.39251)

#### VeriCUA: Verification as Agentic Exploration for Computer-Using Agents (2026-10)
- **简介**：作者 Qiushi Sun、Da Yu、Lingpeng Kong、Ben Kao 等（work in progress）。构建和扩展 computer-using agent（CUA）需要可靠的任务完成验证，用于数据整理、评估和 RL；VLM judge 受限于给定观察，生成程序检查又依赖可访问状态，二者都只是复杂数字环境的局部视角，且为不同平台定制验证器工程成本高。**VeriCUA** 是平台无关的验证框架，把验证视为动态的证据收集与评估：先检视可用上下文构建任务 rubric，再为未解决的要求选择 GUI 或代码检查，并在明确的证据规则下综合发现，适用于跨平台、跨观察模态的在线与离线场景。实验显示与参考标签的一致性优于对比基线，离线困难样本收益最大，在线更能拒绝虚假完成声明（abstract 未给出具体数值）。
- **arXiv**：[2610.40578](https://arxiv.org/abs/2610.40578)

#### Mage-VL-Cowork: A Full Training Recipe to Learn Hybrid Computer Use From Evolved Harnesses (2026-10)
- **简介**：作者 Xiaoyi Zhang、Haoqing Wang 等。有效的 computer-use agent 应在单一策略中结合视觉 GUI 交互与代码/终端执行。**Mage-VL-Cowork** 是 4B 混合 CUA，从通用多模态对齐 checkpoint 出发，经中期训练、SFT、RL 的端到端渐进五阶段课程训练。其数据引擎 Mage-CUA-Forge 用可复用的 skill 条件生成工作流和应用专用 preparer 取代逐任务 agent 构建，合成覆盖 6,915 个 skill 的 293,806 个桌面与网页任务，并将重复的单任务开销降低 82.6%；不直接蒸馏默认 teacher，而是演化 teacher harness、实例化专用 CUA harness 来产生高质量轨迹和分阶段监督。最终在 OSWorld-Verified 上达到 66.0%，推理成本更低，在该参数规模下达到 SOTA。
- **arXiv**：[2610.40747](https://arxiv.org/abs/2610.40747)

#### VeriCUA: Learning Proactive Verification for Reliable GUI Agent Rewards (2026-10)
- **简介**：作者 Zhizhou Wang、Yaxi Lu、Zhong Zhang、Xin Cong、Yankai Lin 等。computer-use agent 的 RL 越来越依赖与执行后 GUI 环境交互的 proactive reward agent，其可靠性取决于「检查什么」与「何时停止」；而 4B 到 GPT-5.4 的 prompted Judge 均停在 68%-70% 的 macro 一致率。提出 **VeriCUA**：可训练的 Judge 决定检查内容与停止时机，冻结的 Probe 在真实环境中执行检查，通过在线 RL 同时优化裁决正确性与 Probe 使用效率，回报在不同终态间归一化。仅在 OSWorld 上训练，在 MobileWorld 与 WebArena-Verified 上零样本取得最高奖励一致率，位于一致率-成本 Pareto 前沿；比 few-shot GPT-5.4 高 5.36 点且 Probe token 少 47.6%-59.8%，1B 版本也超过所有 prompted Judge；作为 Actor RL 的奖励使任务成功率总体提升 3.21 点、held-out 多应用任务提升 5.40 点。
- **arXiv**：[2610.26469](https://arxiv.org/abs/2610.26469)

#### Trace2World: Reconstructing App Worlds from Real Interaction Traces for Scalable GUI Agent Training (2026-10)
- **简介**：作者 Zezhou Wang、Hanzhang Zhou、Panrong Tong、Quyu Kong、Pengxiang Li、Steven Hoi 等。扩展 GUI RL 需要保留真实 app 行为的环境，但把行为推断与编码耦合会把不完整的 trace 变成状态转移错误的可运行 mock。**Trace2World** 将全局行为重建与实现分离：先把 trace 整合为共享的、可溯源的 app 模型，再编译为可执行环境，通过独立回放源工作流定位不一致并做有界修复；验证后的环境支持在源证据范围内按 learner 自适应地扩增任务与环境。从 1,570 条 trace 构建 50 个环境，在 10 个 app 的 200 个工作流上端到端复现 91.1%（最强的 task-local 基线为 59.8%）；基于源的验证保留了真机 94.0% 的模拟训练收益（自验证仅 54.7%）；在其中训练使自建 benchmark 提升 20.2 点，四个外部 GUI benchmark 提升 5.4-16.2 点，100 个真机任务提升 16.3 点。
- **arXiv**：[2610.30792](https://arxiv.org/abs/2610.30792)

#### Agent World Model in the Wild: Computer-Use Agent Environment Synthesis (WildAWM) (2026-10)
- **简介**：作者 Zhaoyang Wang、Qianhui Wu、Haorui Wang、Simon Yu、Baolin Peng、Canwen Xu 等。agent 学习正从人类数据转向与世界交互，但直接在真实计算机系统中扩展既困难又有安全风险。**WildAWM** 是一个可扩展的流水线，为训练通用 computer-use agent（CUA）合成可控环境与任务：构建同时支持 GUI、MCP 服务与命令行的可执行环境，使 agent 能组合视觉交互、工具调用与代码执行，且完全由开源模型驱动；针对任务合成中的可解性、足够难度与可靠 verifier 三个问题，引入结合异构 agent rollout、跨 judge 一致性检查与判断稳健性反事实探针的执行式质量门。流水线合成 341 个长程环境-任务对（可能需 100 步以上，前沿模型 GPT-6-Astra 仅解 60%）；仅在这些合成世界中训练两个不同规模的 Qwen 模型，在涵盖 web、claw 与 computer-use agent 的 8 个分布外 benchmark 上均获显著提升。代码已开源。
- **arXiv**：[2610.26828](https://arxiv.org/abs/2610.26828)

#### Adaptive-Group Process Rewards for GUI Agents (2026-10)
- **简介**：作者 Zirui Shang、Yang Liu、Xin Shu、Zhi Gao、Xinxiao Wu、Lifeng Fan 等。GUI agent 的 RL 受稀疏结果奖励所限，过程奖励需要在长轨迹中找到证据来评价动作。论文提出 adaptive-group 过程奖励框架：动态把 GUI 轨迹划分为语义连贯的 subgoal 组，在动作级与组级两个层面估计过程奖励——动作级 critic 依据组内视觉变化与后续动作评价动作正确性及对 subgoal 的贡献；组级聚合动作评价判断 subgoal 完成度，并利用全轨迹所有组的上下文进一步修正。每个动作的过程奖励由动作级评价与修正后的组级评价结合而成，并叠加到 GRPO 的结果 advantage 上。在过程奖励基准和下游 RL 训练中验证了有效性（abstract 未给出具体数值）。
- **arXiv**：[2610.17710](https://arxiv.org/abs/2610.17710)

#### MAI-Gym: An Open All-in-One Platform for GUI Agent Evaluation and RL Training across Mobile and Computer Environments (2026-10)
- **简介**：作者 Panrong Tong、Quyu Kong、Pengxiang Li 等。GUI agent 的在线 RL 受限于交互环境成本高与真实训练任务稀缺。**MAI-Gym** 是开放的一体化平台，统一移动、桌面和 Web 环境上的 GUI agent 评测、数据采集与在线 RL 训练，通过共享 rollout 流水线保证评测与训练行为一致，支持十个基准、可扩展到 20,000 个并行环境实例，并能自动合成可重置的 mock 应用与跨应用世界及带执行验证的任务 verifier。基于它训练的 MAI-Gym-9B 在 10B 以下开源 GUI agent 中达到 SOTA：MobileWorld 78.4%、OSWorld-Verified 76.2%；在 mock 订票应用上的 RL 还能迁移到真实应用，使真机任务成功率从 28.8% 提升到 48.1%。框架、2,000 个验证任务与模型 checkpoint 均开源。
- **arXiv**：[2610.20939](https://arxiv.org/abs/2610.20939)

#### ECHOES: Seeing Beyond the Current Image via Recursive History Compression and Privileged On-Policy Distillation (2026-10)
- **简介**：作者 Tengda Guo、Jiawang Hu、Yifei Tao 等。长 horizon GUI agent 依赖视觉历史维持任务状态，保留历史截图增加 visual token 与推理成本，文本摘要又会丢失关键状态信息。**ECHOES** 利用训练期 privileged information 引导文本历史压缩：采用 Compression-Think-Action（CTA）格式递归更新文本历史状态并推理；同一模型在 privileged 多图输入下充当 teacher，对 student 生成的 CTA rollout 提供稠密 token 级 on-policy distillation；Perception-Memory Fidelity（PMF）Gap 引导的 OPD-RL 框架动态平衡 privileged 蒸馏与任务奖励优化，并定期更新 teacher。还提出基于真实 GUI agent 失败轨迹的诊断基准 ECHOESBMK。在四个 GUI 基准与 ECHOESBMK 上任务表现和记忆保真度显著提升；部署时无需保留历史截图，在性能与推理成本间取得良好权衡，并对未见的视觉规划任务有正向零样本迁移。
- **arXiv**：[2610.22408](https://arxiv.org/abs/2610.22408)

#### SoftwareGym: Scaling Computer-Use Agent Training with Diverse Synthesized Software Environments (2026-10)
- **简介**：作者 Tian Jin、Wenhao Wang、Jing Kang、Haoting Shi、Rui Ye、Yijun Lu 等。computer-use agent（CUA）进步迅速，但大规模合成多样且可靠的软件环境仍困难：原生应用环境构建与集成成本高、覆盖受限，浏览器托管环境交互深度与流程复杂度有限。**SoftwareGym** 是自动合成 GUI 软件环境的流水线：生成结构多样的设计，将其实现为可运行应用，并通过 GUI 评估加迭代修复保证功能可靠。使用该流水线合成了十二类共 182 个应用，GUI 任务平均通过率 91.5%；构建了 120 任务的评测集 SoftwareGym-Bench，并用这些环境收集的轨迹训练 CUA。所得模型在严格域外的 OSWorld 上 Pass@3 从 41.39% 提升至 56.97%，在 SoftwareGym-Bench 上达到 40.0%。
- **arXiv**：[2610.12611](https://arxiv.org/abs/2610.12611)

#### SCA: Spatial Credit Assignment for Reinforcement Learning of GUI Agents (2026-09)
- **简介**：作者 Shengtian Yang、Ziyu Xiong 等指出 GUI agent 的 group-relative RL 以二值评估看待点击结果，把空间位置不同的失败点击视为相同，且当所有采样点击都失败时无相对信号。提出 **Spatial Credit Assignment（SCA）**：用采样点击的屏幕坐标细化 group-relative credit——在同时含成功与失败的组内用其余响应预测每条 held-out 响应的 reward，以预测残差调整 credit；若全部失败，则按到标注目标的距离排序。这些空间参照仅用于构造训练更新，部署策略不变。在受控合成实验中检验其更新误差与方向与精确回报梯度的一致性；在 GUI grounding 与离线动作预测基准上，SCA 提升了跨专业领域的 grounding，并在大多数动作预测指标上取得 RL 微调模型中的最强结果。
- **arXiv**：[2609.36939](https://arxiv.org/abs/2609.36939)

#### CUA-Sandbox: Efficient Environments for Computer-Use Agent Reinforcement Learning (2026-09)
- **简介**：作者 Xin Yan、Zhengbo Jiao、Jiaqi Liu 等指出，computer-use agent 的 RL 中，传统部署为每个独立 rollout 复制一份初始化运行时，并行环境增多时内存与初始化开销重复累积。其观察是轨迹需要独立的可变状态，而初始化后的应用运行时可在并发演化的环境间共享，故状态才是环境独立性的自然单位。**CUA-Sandbox** 通过 state-scoped 执行与事务化生命周期操作（含 reset 与 branch）将私有 state capsule 与共享运行时分离，同时保留原有软件接口与任务评估器。实验显示任务成功率与 Docker 相当或更好，同时大幅降低 rollout 与资源成本：rollout 吞吐最高提升 6.20 倍，单环境内存降低 9.2 倍，增量存储降低 504 倍。
- **arXiv**：[2609.32750](https://arxiv.org/abs/2609.32750)

#### ComputerSD: Online Self-Distillation from Real-Time Feedback for Computer-Use Agents (2026-09)
- **简介**：作者 Yong Du、Tongbo Chen、Zhengxi Lu 等指出 computer-use agent（CUA）在线训练多依赖稀疏结果 reward，对中间动作无监督；on-policy self-distillation（OPSD）可通过特权重打分提供 token 级信号，但直接用于 CUA 在线训练有两个挑战：固定引导可能与学生当前状态错位，引导引起的概率偏移可能与步级正确性冲突。提出 **ComputerSD**：把已执行 GUI 转移的实时反馈转化为策略学习的引导。微调的 GUI analyzer 在每次动作后产生引导和步级 value 分数，引导作为特权上下文，分数调节所得 OPSD 信号；在全异步训练框架中联合优化 token 级 OPSD 与轨迹级 GRPO。OSWorld-Verified 上，在通用 Qwen3-VL-8B-Thinking 和专用 EvoCUA-8B 骨干上分别比仅结果 GRPO 高 1.9 和 4.1 个百分点，OOD 设置评估也支持其泛化。
- **arXiv**：[2609.40253](https://arxiv.org/abs/2609.40253)

#### Learn How to Act from Your Own Interactions: On-Policy Self-Distillation for GUI Agents (GUI-SD-v2) (2026-09)
- **简介**：作者 Yan Zhang、Daiqing Wu、Huawen Shen 等。On-policy self-distillation (OPSD) 凭借特权条件 self-teacher 的稠密 token 级监督在 GUI grounding 上表现很好，但扩展到多轮 GUI agent 受限于 self-teacher 遵循特权的能力弱、特权指导不足。**GUI-SD-v2** 采用两阶段训练：先在相同 GUI 状态上联合优化带/不带特权指导的 rollout 以强化 privilege following；再通过 privilege-conditioned self-teacher 选择性蒸馏步级推理与记忆指导，支持动作决策与任务相关信息的保留。在 AndroidWorld 与 MobileWorld 上，Pass@1 与 Pass@3 成功率均优于现有 OPSD 基线及对比 SOTA（abstract 未给出具体数值）。
- **arXiv**：[2609.27307](https://arxiv.org/abs/2609.27307)

#### RecreationWorld: Scalable and Verifiable Environments for Hybrid Computer-Use Agents (2026-09)
- **简介**：作者 Shuai Bai、Jiayong Deng、Sicheng Fan 等。CUA 沿图形交互与代码/命令行软件开发两条线分别发展，而真实数字工作需要二者交错。**RecreationWorld** 是围绕「复现」构建的五平台框架：给定一个运行中的参考应用，agent 必须自主发现其行为并构建忠实实现，无预设流程；提供 Ubuntu、macOS、Windows、Android、Web 的可复现环境与统一的 GUI 控制加编码工具 harness，以运行中的参考作为隐藏行为测试的 oracle，提供基于执行的奖励，并用高质量开源应用扩展轨迹生成。在这些轨迹上训练的模型在五个分布外编码与混合 CUA 基准上提升，更常验证渲染输出。另发布含 250 个任务的 RecreationBench：GPT-6 Astra 总体 58.1% 领先，但仅 2.8% 的任务通过全部程序化测试。
- **arXiv**：[2609.22000](https://arxiv.org/abs/2609.22000)

#### MintAct: A Unified Visual Agent for Digital Environments (2026-09)
- **简介**：作者 Mingfei Gao、Rui Tian、Haiming Gang 等。**MintAct** 是统一 UI grounding、跨移动/桌面/Web 多步导航和视觉工具使用的视觉语言模型家族，规模为 2B、4B、8B，在这些能力上匹配各领域专家的表现。为此搭建了可扩展的环境与 RL 基础设施：在异构的各领域后端上托管数百个并发实例，同时服务轨迹采集与在线 RL；异步训练框架显式控制跨领域训练分布，并在环境反馈有噪声和 off-policy drift 下保持稳定。在可比规模下，MintAct 在多种基准上达到 SOTA，OSWorld-Verified 上为 48.9。
- **arXiv**：[2609.22083](https://arxiv.org/abs/2609.22083)

#### BlueLM-GUI Technical Report: A Real-Device-Centric Flywheel for Self-Improving Mobile GUI Agents (2026-09)
- **简介**：作者 Tong Ye、Kunyang Han、Guozhi Wang 等。移动 GUI agent 的工业部署面临三个缺口：沙盒训练与生产环境分布不匹配、昂贵的真机失败样本未被充分利用、固定 benchmark 饱和。BlueLM-GUI（35B-A3B）以「真机为中心的飞轮」弥合：Every Sample Matters——双轨流水线与异构三系统共识评估、错误纠正与衍生模块，把每条轨迹转化为可用监督；Every Rollout Is Real——持续预训练、SFT 与在数百台真机上的 agentic RL 三阶段配方；Every Query Evolves——按配额、三正交轴构造可随模型升级的 benchmark。在 MobileGUI-VBench 上得 87.4，超过最强闭源模型 5.1 分；AndroidWorld 上 84.9，为开源模型最佳、与闭源模型相当。
- **arXiv**：[2609.12394](https://arxiv.org/abs/2609.12394)

#### AnchorGUI: Asymmetric Memory for Dual-Scale Learning in GUI Navigation (2026-09)
- **简介**：作者 Shengjie Jin、Zelong Sun、Hengbo Xu 等（ECCV 2026）。VLM GUI agent 难以处理密集连续的视觉历史，阻碍单 episode 内的即时纠错（intra-trial）与跨多次尝试的经验蒸馏（cross-trial）。作者发现 GUI 导航中存在信息不对称：预期的状态转移可被压缩成轻量文本摘要，而意外结果需要保留截图作为因果证据。**AnchorGUI** 由 Cognitive State Anchor（CSA）驱动，逐步比较预期与观察到的转移，把被动的多模态轨迹转为显式预测误差信号，并通过非对称记忆编排双尺度学习：intra-trial 时滑动窗口仅保留不匹配处的视觉证据；cross-trial 蒸馏时把昂贵的 credit assignment 搜索空间聚焦到可能的失败步骤。在 AndroidWorld 上成功率 57.3%，每步 token 减少 2.4 倍；跨 trial 蒸馏达 69.2%（+11.9%），显著优于标准反思方法。
- **arXiv**：[2609.15457](https://arxiv.org/abs/2609.15457)

#### CADWorld: Computer-Use Benchmark for Long-Horizon Computer-Aided Design (2026-09)
- **简介**：作者 Zihan Dong、Yuanzhe Liu、Zhiyuan Ma 等。现有 computer-use benchmark 对产出持久结构化 artifact 的专业工程工作流覆盖有限，机械 CAD 要求 agent 在长交互 horizon 内操作几何与约束并产出有效的原生工程文件。**CADWorld** 是 FreeCAD 中的长程 computer-use benchmark，含 200 个任务、覆盖草图、零件建模、装配、CAM、FEM、测量、网格处理、工程制图等 11 类机械 CAD 工作流；agent 通过截图与 GUI 动作操作，成功由对保存的 FreeCAD 文件和辅助输出做任务特定的可执行检查判定。七个现有 agent 中最强者成功率仅 17.5%，而专家参考通过率为 87.0%；较弱 agent 常在产出有效 artifact 前就失败，较强 agent 则越来越多地败于结构、几何与构建过程要求。
- **arXiv**：[2609.16251](https://arxiv.org/abs/2609.16251)

#### Iron: Intent-Aligned and Retrospective Dual Learning Framework for Enhancing Generalist Virtual Agents (2026-08)
- **简介**：浙江大学（Jiahe Ying、Wendong Bu、Kaihang Pan、Juncheng Li、Siliang Tang 等）。针对 MLLM 驱动的 GUI agent 三大痛点——数据标注昂贵、动作与意图对齐不精确、失败 trajectory 被丢弃导致探索低效，**Iron** 提出一种对偶学习策略：用 stepwise cycle-consistent（SCC）reward 在低层动作与高层意图之间做细粒度对齐，以改善指令 grounding 与意图理解；同时引入 hindsight reproduction 机制把失败 trajectory 改造成可训练数据，提升学习效率与任务多样性。实验显示 Iron 训练的通用 agent 在跨环境、跨设备任务上稳定提升，超过使用三倍数据训练的模型，在未见过的 web 任务上取得 25.06% 的相对提升，在本身较复杂的任务上增益更明显。
- **arXiv**：[2608.27866](https://arxiv.org/abs/2608.27866)

#### GSAR: Goal-State-Anchor Rewards for Mobile GUI Agents with Self-Evolving Data Synthesis (2026-08)
- **简介**：Long Zhang, Yuhan Chen, Pengzhi Gao, Jian Luan 等。基于 VLM 的 GUI agent 在线 RL 受两个瓶颈制约：数据合成依赖特定环境、难以产出多样任务；现有 evaluator 要么可扩展性差，要么给出不准确、不可靠的 reward。**GSAR** 提出自演化数据合成，通过任务执行派生出多个环境并生成多样任务与目标状态；配套的 state-anchor 机制自动把成功目标状态中与任务相关的 UI 元素标注为参考 anchor，RL 训练时由这些 anchor 提供准确且可扩展的 reward，显著提升训练效率。离线 trajectory 验证准确率超过 90%，在所有 evaluator 中最接近 rule-based 方法；用该 reward 框架训练的 agent 在 AndroidWorld 与作者自建基准上均表现强劲。
- **arXiv**：[2608.22847](https://arxiv.org/abs/2608.22847)

#### Beyond Success and Failure: Length-Aware Contrastive Learning for GUI Agents (LACL-GUI) (2026-08)
- **简介**：Chengyang Gu, Le Zhang, Jingbo Zhou, Hui Xiong 等。GRPO 等方法用于 MLLM 驱动的 GUI agent 时存在 reward-梯度错配，导致优化低效且不稳定；近期工作把 RLVR 改写为对比式或分类式目标以消除问题梯度，但仍只用结果级监督，无法刻画同一结果类别内部 trajectory 质量的细粒度差异。**LACL-GUI** 把 trajectory 级质量信号引入对比式 RLVR：在成功与失败 trajectory 内部分别构造结构化偏好，鼓励更简洁的成功执行，并按与成功 trajectory 的偏离程度区分失败的优劣，同时保持优化稳定性。GUI agent 基准上给出更有效的学习信号并一致优于此前方法（abstract 未给出具体数值）。
- **arXiv**：[2608.21830](https://arxiv.org/abs/2608.21830)

#### UI-Mate: Advancing Open-Weight Foundation GUI Agents with In-Context Demonstrations (2026-08)
- **简介**：Zihan Ding, Longxu Dou, Qi Gao, Lei Ke 等。基础 GUI agent 的部署受训练数据稀缺且有偏、指令歧义、执行不可靠三重阻碍，且日常 workflow 依赖用户特定工具与默会惯例，未说明的指令会让多次运行产生任意差异。**UI-Mate** 把环境接地的训练栈与 in-context demonstration 学习整合：闭环数据引擎通过统一的 task-verifier bundle 在大规模并行环境中自动完成任务生成、环境构建、rollout、过滤、能力平衡、SFT 与 online RL；in-context demonstration 学习把多模态演示转成灵活的子任务级 workflow，跟随相关演示步骤并从实时界面重规划；同时发布 OSWorkerBench（41 个应用上 100 个长 horizon 办公任务，含 33 任务 self-demo 与 45 任务 variant-demo 两种演示设置）。UI-Mate-27B 在 OSWorld-Verified 上 77.0%、WindowsAgentArena 上 66.2%，创开源权重 SOTA；OSWorkerBench 上严格成功率 41.0%、进度 76.9%，比 Qwen3.6-27B 基座分别高 17.7 和 24.5 个点；33 任务 self-demo 子集上仅一条演示就把严格成功率从 17.2% 提到 35.4%、进度从 67.9% 提到 81.1%。
- **arXiv**：[2608.15930](https://arxiv.org/abs/2608.15930)

#### Test-Time Self-Evolving GUI Visual Grounding via Reflection-Guided On-Policy Self-Distillation (2026-08)
- **简介**：Shiyu Xuan、Zechao Li。针对 GUI grounding 模型部署后参数冻结、无法适应未见界面，而已有 test-time RL 方法又无法对失败探索进行反思的问题，作者提出 Test-Time Self-Evolving 框架，在无人工标注 ground truth 的条件下构建「探索—评估—反思—内化」闭环：agent 先按指令在未见界面上预测 grounding 坐标，MLLM-based Reflector 评估结果并给出相应的推理式反思，再由 **Reflection-Guided On-Policy Self-Distillation** 借一个条件化 self-teacher 把高层反思翻译为稠密的 token 级监督写回权重；另设 Contrastive Calibration，防止失败探索中错误的自回归前缀污染监督信号。六个 benchmark 上相对 base model 平均准确率提升 7.4%，作者称这是首个成功把 on-policy self-distillation 用于 GUI visual grounding test-time 适应的工作，代码将开源。
- **arXiv**：[2608.11191](https://arxiv.org/abs/2608.11191)

#### CoAdapt-GUI: Joint Workflow Context and Policy Adaptation for Unseen GUI Applications (2026-08)
- **简介**：Linqiang Guo、Li Gu、Zihuan Jiang、Zhixiang Chi 等。针对移动 GUI agent 迁移到训练集之外的 app 就变脆弱，且现实设定下目标 app 只有有限交互预算、没有目标演示数据的问题，**CoAdapt-GUI** 提出 test-time adaptation（TTA）框架，仅用 agent 自己在目标 app 上的 rollout 与 reward，联合适配结构化 workflow context 与策略。workflow context 只保留可迁移的操作流程、失败模式与校验规则，剔除绑定具体 app 的界面细节，从而让可复用的流程知识指导适配而不迁移源界面状态；策略侧用任务上下文匹配的 group-relative 优化，在冻结的视觉语言模型上更新 LoRA adapter。两项未见 app 评测中，AndroidWorld-Generalization 达到 45.0%（对照 Policy-Only TTA 基线 37.5%），AndroidWorld Plus 从 38.6% 提升到 52.9%。
- **arXiv**：[2608.11588](https://arxiv.org/abs/2608.11588)

#### LookAgain: Closed-Loop GUI Grounding with Visually Grounded Reflection (2026-08)
- **简介**：Renshan Zhang、Haoyang Meng、Rui Shao、Liqiang Nie 等。针对现有 GUI grounder 单次预测在小目标、密集控件与分布外界面上急剧退化，且都不把已产出的坐标当作「可在新视觉证据下被反思修正的假设」的问题，**LookAgain** 把 grounding 重构为多 turn 的 predict–look-again–refine 闭环，只用两个原语：locate 提出坐标假设、在图像上渲染标记并追加该预测区域的局部 patch，使下一步推理以上一次预测为空间先验；confirm 接受或拒绝该假设并终止流程。训练上先用构造的反思 trajectory 做 SFT 冷启动，再以「最终 grounding 是否正确」作为唯一 reward 做 GRPO。在拒识感知（refusal-aware）与通用 GUI grounding benchmark 上一致提升并取得 SOTA，消融验证了各组件有效性（abstract 未给出具体数值）。
- **arXiv**：[2608.09723](https://arxiv.org/abs/2608.09723)

#### Qwen-CUA: Native Computer Use for (almost) Everything (2026-08)
- **简介**：Qwen 团队（Dunjie Lu、Shuai Bai、Binyuan Hui、Junyang Lin 等 40 余人）。**Qwen-CUA** 以 397B-A17B Qwen MoE 为骨干，仅看截图、仅通过键鼠事件操作，不依赖 DOM 树、无障碍元数据或任务专用 API；scaffold 维持最多 20 张活跃截图并把更早视觉历史折叠为定长 block，兼顾近期证据与 prompt 前缀复用。训练侧搭建近 100,000 vCPU、数万并发环境的 rollout 集群，构造约 40,000 个可验证任务，用 verifiable reward 与 trajectory slicing 优化完整 trajectory。八个基准超过 Qwen3.7：OSWorld-Verified 86.2，OSWorld 2.0 binary/partial 18.5/48.4；万亿参数以上的 Qwen-CUA-Max 提升至 87.6 与 21.2/53.3，并把 RedTeamCUA 攻击成功率从 36.6 降到 16.4。
- **arXiv**：[2608.02352](https://arxiv.org/abs/2608.02352)

#### The Next Screenshot Knows: Gated Hindsight Distillation for Mobile GUI Agents (GHD) (2026-08)
- **简介**：Weiwei Li、Junzhuo Liu、Tong Chu、Wen Li 等。指出 GUI agent 多从成功轨迹离线训练，把 trajectory 拆成 prefix-action 对时丢弃了后续观测，也丢掉了「该动作为何正确」的依据——证据往往只出现在下一屏（如要开启 Soft Wrap 需先点 Edit 或 View，菜单打开前无从判断），纯模仿几乎没机会采样到正确推理。提出 **Gated Hindsight Distillation (GHD)**：训练时把下一张截图作为特权信息，student 仅依据可观测 trajectory 前缀预测，参数共享的 teacher 额外看到下一张截图并重评 student 的 on-policy 响应；仅当 student 失败而 hindsight-conditioned teacher 能恢复示范动作时才施加蒸馏。AndroidWorld 与 AndroidLab、两个视觉语言模型上成功率均优于 GRPO。
- **arXiv**：[2608.06065](https://arxiv.org/abs/2608.06065)

#### Qwen-UI-Agent Technical Report: Toward Next-Generation Real-World Centric Foundation GUI Agents (Qwen-UI-Agent) (2026-07)
- **简介**：来自 Qwen 团队（Hanzhang Zhou、Steven Hoi 等 16 人）。面向真实设备的基础 GUI agent，横跨移动端、computer-use、web 与 DeepSearch 环境：统一动作空间在单次模型 turn 内交错 GUI 操作与 CLI 执行并批量输出；AutoResearch 式数据飞轮用 agent 构造任务/环境、诊断失败、规划下一轮迭代；**在线 RL 支持在超过 100 turn 的轨迹上训练，逾 1 万并发环境加速 rollout**；轻量 harness 层支持主动服务发起与跨移动/桌面的有状态工作流。移动端刷新 SOTA（MobileWorld 82.1%、MobileWorld-Real 92.2%、AndroidDaily 97.5%），computer-use（OSWorld-Verified 79.5%）与 browser（WebArena 73.6%、ScreenSpot-Pro 81.5%）对标前沿模型具竞争力。
- **arXiv**：[2607.28227](https://arxiv.org/abs/2607.28227)

#### Interactive Reward Agent: GUI Task Evaluation via Environment-State Verification (IRA) (2026-07)
- **简介**：来自 Chenrui Shi 等 8 人。GUI 任务评估结果可作为 test-time scaling 与后训练的奖励信号，但可靠评估常需访问超出截图的环境状态（系统配置、文件数据、应用设置）。提出基于 propose-then-verify 的 **交互式奖励 agent（IRA）**：给定指令与执行后 GUI 环境，先提出任务完成条件，再通过调用系统工具/应用工具/GUI 工具验证，把可见界面与环境状态的证据在交互过程中结合。并引入 **GUI-RewardBench**（10 类 Ubuntu 桌面应用、321 条轨迹）。IRA 达 86.9% 准确率优于已有评估基线；**将 IRA 用于 GUI agent 的强化学习，取得 34.0% OSWorld 成功率**，证明其可提供有效奖励信号。
- **arXiv**：[2607.25904](https://arxiv.org/abs/2607.25904)

#### ODYSSE: Episode-wise Policy Optimization for Personalized Agentic Reasoning (ODYSSE) (2026-07)
- **简介**：来自昆士兰大学 / Griffith 大学（Jiaqi Zhang、Hongzhi Yin 等 5 人）。以人为中心的场景常是模糊请求 + 开放解空间，需 agent 同时与用户和环境交互来解码个性化偏好——即"个性化 agentic 推理"。提出强化微调框架 **ODYSSE**，核心是 **Episode-wise GRPO（ESPO）**：面对长动作 horizon 与强跨步依赖，不再独立优化各步，而引入 **episode 级奖励 + episodic 优势估计**，使上游证据有效引导下游个性化决策，跨多轮交互逐步消解模糊请求；并配 episodic batch sampler 把同一 episode 的动作聚成统一训练 batch。在真实长程个性化 GUI 推理任务上一致优于专用与通用 LVLM。
- **arXiv**：[2607.25369](https://arxiv.org/abs/2607.25369)

#### Learning to Detect UI Principle Violations via Reinforcement Learning（UI-Critic） (2026-07)
- **简介**：Nishi Mehta、Pratik Jayarao 等提出：小模型/编码智能体生成的前端界面即便能编译、渲染、过单测，仍常违反可访问性、欺骗性设计、视觉层级等界面质量原则。工作统一 WCAG 2.2、欺骗性设计分类法与感知/认知/交互理论共 19 条界面质量原则，用向干净 LLM 生成的 Tailwind 页面**合成注入已知违规**构造约 1 万页的验证数据集；对 4B 视觉语言模型做持续 RL，将 micro-F1 从 36% 提到 84%（19 条中 13 条 >80% F1）。所得 critic 可审计生成界面、过滤低质训练数据，并为"设计感知代码生成"提供奖励信号。
- **arXiv**：[2607.20690](https://arxiv.org/abs/2607.20690)

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

#### ModalSearcher: Weaving Visual Evidence into Autonomous Cross-Modal Deep Search (2026-11)
- **简介**：作者 Zhihan Yin、Ruyi Li、Chengyu Fang 等。现有多模态 agentic search 常把视觉证据固定在输入或预设的轨迹位置，agent 难以随信息需求选择搜索模态。论文定义「自主跨模态搜索」：agent 在直接推理、文搜文、文搜图、以图搜图间选择，适时主动引入视觉证据，并跨多跳切换与整合模态。提出 **ModalSearch-Pipeline**——构建交错多模态知识图谱，采样多样深度与模态组合的路径，并用反向构造问题隐去会暴露推理捷径的实体名与图像目标；在真实 Web 环境中用 SFT + RL 训练 **ModalSearcher Agent**，并构建 ModalSearch-Bench。ModalSearcher-8B 在相同 agentic 工作流下比 Qwen3-VL-8B 平均准确率高 16.2 个点。
- **arXiv**：[2611.06131](https://arxiv.org/abs/2611.06131)

#### DR Pixel: Page-Native Deep Research with Vision-Language Models (2026-11)
- **简介**：作者 Tiansheng Hu、Yichuan Wang、James V. Talwar、Nicholas Tomlin、Chen Zhao、Sewon Min 等。现有 deep research agent 通常把论文简化为 OCR 文本，丢失表格、图与公式信息。**DR Pixel** 是首个开源端到端多模态 DR agent，以渲染的页面图像为唯一检索与推理单元，在 106 万页语料上运行，并构建 DRPixelBench 评测。无需训练的 page-native DR 已优于 GPT-5.4、Gemini-3.5-Flash、Qwen3.6-27B 等上的 OCR 基线，且工具调用更少。训练方面，教师轨迹噪声大，混杂关键推理/动作步骤与冗余或失败尝试，为此提出 **counterfactual credit reweighting**，按各推理与动作步骤对最终答案的反事实贡献加权。训练使检索召回提升约 15 个点、DRPixelBench 最终准确率提升超过 10 个点，并在另外四个现有 benchmark 上超过未训练基线。
- **arXiv**：[2611.03097](https://arxiv.org/abs/2611.03097)

#### Search When Needed, Proceed When Reliable: Metacognitive Reliability Supervision for Efficient Agentic Search (2026-10)
- **简介**：作者 Minhao Sun、Zhijing Wu、Xian-Ling Mao。高效的 agentic search 要求 LLM 判断能否凭现有知识可靠推进，还是需要额外外部证据；基于最终答案正确性或轨迹级反馈的监督，只能间接反映当前决策状态的可靠性及检索如何改变它。论文提出元认知框架，让 LLM 依据中间推理状态的可靠性做搜索决策：估计局部结论的可靠性，以 current-state reliability score 判断是否无需搜索即可推进，以 search-action score 衡量检索到的证据使可靠性提升多少；二者共同构成面向 GRPO 的 reliability-aware 过程奖励，对搜索与不搜索行为提供更细粒度监督。在七个问答 benchmark 上验证了有效性（abstract 摘录未给出具体数值）。
- **arXiv**：[2610.41738](https://arxiv.org/abs/2610.41738)

#### Heterogeneous Co-Evolution Learning for Tool-Efficient Search Agents with Zero Labels (ZEST-Search) (2026-10)
- **简介**：作者 Qin-Wen Luo、Sheng Ren、Naiqiang Tan 等。深度搜索 agent 训练面临两个问题：能力边界由训练问题决定，而固定/带标注的任务集无法随 agent 提升而自适应；仅看结果的 RL 对所有成功轨迹一视同仁，即使有些正确答案依赖冗余搜索。**ZEST-Search** 是零标签、工具高效的异构 co-evolution 框架：从无标注参考问题出发，通过 answer-first 与证据路径引导的生成在线构造经过验证的搜索任务；异构 proposer 围绕 solver 不断变化的能力边界自适应出题，solver 用 RL 学习搜索增强推理；proposer 使用中等难度奖励，solver 使用任务局部的 tool-aware 奖励，鼓励有信息量的任务和更少冗余搜索的正确答案。在七个通用与多跳 QA benchmark 上持续提升搜索 agent 表现并提高工具效率（abstract 未给出具体数值）。
- **arXiv**：[2610.39682](https://arxiv.org/abs/2610.39682)

#### MTA-Zero: Self-Improving Multimodal Deep Search via Online QA-Solver Co-training (2026-10)
- **简介**：作者 Xiangyu Peng、Jiaxin Zhang、An Yan 等。MLLM 在开放世界多步搜索（图像定位、选工具、整合网页证据）上仍受限，现有方法依赖固定离线语料，问题分布无法随 agent 进步调整。**MTA-Zero** 是在线多模态 QA-Solver co-training 框架，无需预先构造多跳语料：可训练的 QA Generator 从图像化 VQA 种子上下文检索证据并在线构造经验证的多跳问题，独立的 deep-search Solver 选工具、收集证据并回答；Generator 的 Goldilocks 奖励来自 Solver 组内 rollout 成功率——全对或全错的问题奖励为零，接近当前能力前沿的问题被偏好；两个策略各自用 GRPO 独立优化、不共享参数或梯度。在六个多模态 deep-search benchmark 上，Qwen3-VL-8B Solver 平均准确率由 36.91% 升至 51.58%，在相同 agent workflow 下性能可比 GPT-5；Qwen3.6-27B Solver 提升 10.82 点。
- **arXiv**：[2610.33799](https://arxiv.org/abs/2610.33799)

#### Clearing the FoG: Training KG Agents to Retrieve the Evidence Behind Their Answers (FoG) (2026-10)
- **简介**：作者 Zongyuan Sun、Shuai Zhao、Haoran Luo 等。现有知识图谱（KG）agent 只把 KG 当答案来源：训练数据只有答案没有证据，奖励只检查答案不检查检索到的证据，因此 agent 常常答对却没检索到支撑三元组。论文提出 **FoG**，训练 KG agent 检索答案背后的证据：evidence-grounded synthesis 递归地把显式实体隐藏在可验证子问题之后，使每个问题天然带着其支撑三元组；KG 交互 agent 记录 KG 返回的三元组；**ECPR**（evidence-coverage process reward）把每个支撑三元组的功劳归给首次检索到它的那一轮。在三个 KGQA benchmark 上，3B-4B 开源 agent 在答案准确率和证据覆盖率上显著优于更大的基线，并具备零样本跨 KG 迁移能力。
- **arXiv**：[2610.37405](https://arxiv.org/abs/2610.37405)

#### EG-GRPO: Evidence-Guided Step-level Credit Assignment for Deep Search Agents (2026-10)
- **简介**：作者 Sijia Luo、Mengjie Gao 等。Deep search agent 通常只靠轨迹级 outcome reward 训练，而开放式搜索中难以获得可靠、与任务对齐的步级监督。论文提出 **EG-GRPO**（Evidence-Guided GRPO），利用合成 deep search 数据自带的 evidence 标注：找出每条必需证据 claim 的首次支持性引用，并据此用新获得的证据构造步级 process reward；将累积证据表示为不断演化的 knowledge state，做 state-conditioned 的相对优势估计，再与标准轨迹级 outcome advantage 结合。在五个 deep search benchmark、不同模型规模上总体优于标准 GRPO 以及已有 reward-enhanced 和 step-level RL 基线，说明任务相关的 evidence 标注可为 deep search RL 提供有效的过程监督。（abstract 未给出具体数值）
- **arXiv**：[2610.25269](https://arxiv.org/abs/2610.25269)

#### FALCA: Falsification-Aware Reinforcement Learning with Hierarchical Credit Assignment for Long-Horizon Deep Search Agents (2026-10)
- **简介**：作者 Zijun Dou、Kun Luo 等。长程 deep search 中早期错误会被放大：agent 容易锚定在错误候选、只收集部分支持证据、忽略矛盾。论文提出 **FALCA**，把 deep search 表述为 propose-falsify 过程，采用 Searcher-Falsifier 协作架构：Searcher 提出候选答案/子答案，独立的 Falsifier 主动寻找反证、检查未满足的约束、识别冲突并产出简洁的证伪报告用于修正 research state。训练上先用证伪轨迹做 SFT warmup，再进行基于并行 rollout 的 agentic RL，并对 research stage、state transition 和 action 做层次化 credit assignment。在五个搜索 benchmark 上优于 backbone 与 harness 消融基线，并减少 confirmation bias 与误差累积。（abstract 未给出具体数值）
- **arXiv**：[2610.28361](https://arxiv.org/abs/2610.28361)

#### Grounded Credit Assignment in Reinforcement Learning for Multi-Turn Search Agents (GCA) (2026-10)
- **简介**：作者 Qirui Liu、Hao Chen 等。GRPO 等 outcome-reward RL 给 search agent rollout 的每个 token 同样的 advantage，但 rollout 同时涉及「找证据」与「用证据」两种能力。作者发现准确率提升并不意味着 grounding：Search-R1 在七个 QA benchmark 上都提高了准确率，但在 Bamboogle 上正确答案中有证据支撑的比例从 84.0% 降到 42.9%。论文把原因归于 GRPO 的 credit assignment，并提出 **Grounded Credit Assignment (GCA)**：利用可验证的子结果（证据检索与答案正确性）分别校正搜索与回答的 credit——答错比找到证据的搜索受到更多归咎，绕过可检索证据而答对则获得更少 credit；校正量上限为每个 token advantage 的 10%，不会压过 outcome 信号，且无需额外模型、rollout 或训练阶段。GCA 使平均 EM 比 Search-R1 在 Qwen2.5-3B 和 Qwen3-30B-A3B 上分别提高 5.7 和 6.3 分，并把有证据支撑比例从 76.9% 提升到 93.4%。
- **arXiv**：[2610.23148](https://arxiv.org/abs/2610.23148)

#### ClueFlow: Learning Long-Horizon Search through Attributed Research-State Transitions (2026-10)
- **简介**：作者 Yicheng Wang、Wentao Tan、Yidan Huang、Yifei Liu、Qiong Cao。长程搜索要求 agent 获取相关证据并整合进不断演化的研究状态，已有方法维护研究状态或给出 turn 级进展信号，但缺少落在可归因状态变化上的信号。**ClueFlow** 构建 Source-State-Final 归因链，把获取证据的浏览器动作、Evidence/Candidate 更新与最终答案连接起来，执行后用 gold 条件化的语义验证和来源追踪；归因同时指导 SFT（选择哪些决策获得监督）与 RL（Turn-PPO 中沿归因链为 outcome turn advantage 补充 credit），并能从最终答案错误的轨迹中回收与 gold 一致的状态变化。在两个 Qwen 骨干、四个搜索 benchmark 上，Qwen3.5-35B-A3B 在 BrowseComp 达 60.0%、BrowseComp-Plus 达 72.0%；归因引导 SFT 用约 30% 训练 token 即可匹配全 turn SFT，归因引导 PPO 比 outcome-only PPO 高 4.1 点。
- **arXiv**：[2610.25980](https://arxiv.org/abs/2610.25980)

#### KnowGPO: Knowledge-Grouped Policy Optimization for Search Agents (2026-10)
- **简介**：作者 Siqi Chen、Zhengxi Lu、Yuchen Yan、Xingyu Wu 等（ZJU-REAL）。深度搜索 agent 的 credit assignment 困难，现有过程监督多锚定于表层轨迹状态或模型内部信号，而深度搜索中观测异质、检索上下文又长又嘈杂，二者都不可靠。**KnowGPO** 按轨迹所获得的任务相关知识而非表层观测对齐轨迹：把求解答案所需信息分解为原子知识条件，用已满足条件的子集表示每步轨迹状态；知识状态相同的轨迹分到同一组内比较，相邻步之间的增量知识增益作为环境落地的过程奖励。在 Xbench、BrowseComp-ZH 和 GAIA 上，KnowGPO 比仅结果奖励的 GRPO 与过程监督基线（GiGPO、IGPO）一致高 3.8 到 4.2 分，其 8B 模型可媲美或超过多数 32B 开源系统。
- **arXiv**：[2610.22331](https://arxiv.org/abs/2610.22331)

#### MMSearch-OPD: Easy Made Hard, Hard Made Learnable for Long-Horizon Multimodal Search Agents (2026-10)
- **简介**：作者 Huanyao Zhang、Jiepeng Zhou、Jiaye Lin 等。多模态深度搜索走向长 horizon 调查，但训练受限于高难度训练问题不足与过程反馈缺乏。**MMSearch-OPD** 结合可控数据精炼与 RL + on-policy distillation（OPD）联合训练：Refine 流水线对已有的简单多模态问题推理链做针对性扩展，用参数化变换控制搜索深度与跨模态证据需求，同时复用现有图像-问题对；训练时用答案奖励和轻量行为惩罚引导 RL，并由奖励差距调制对不完美 student 轨迹的 token 级 teacher 指导，蒸馏系数指数衰减以逐步降低对 teacher 的依赖。实验显示数据精炼有收益，SFT 之后的联合训练带来进一步提升（abstract 未给出具体数值）。
- **arXiv**：[2610.22240](https://arxiv.org/abs/2610.22240)

#### SOAR: Source-Aware Reinforcement Learning with Precise Credit Assignment for Search Agents (2026-10)
- **简介**：作者 Shengze Huang、Zihao Tang、Jinluan Yang、Xin Yu、Zhenghao Lin、Xiao Liu 等。搜索 agent 的 RL 常出现 evidence fabrication：agent 绕过外部信息，凭不可靠的内部知识捷径编造错误答案，根源在于推理轨迹不透明。**SOAR（SOurce-Aware Reinforcement learning）** 将单体式推理变为信息来源显式的轨迹：通过 Source-Aware Workflow 把外部证据与内部知识拆为并行、可观测的两条流，并引入 Source-Consistent Reward 引导策略演化，使不同知识流之间的信用分配更精确。在 3B 与 7B 骨干上分别超过基线 7.3% 和 5.4%，同时将 evidence fabrication 降低 74.8%。
- **arXiv**：[2610.13157](https://arxiv.org/abs/2610.13157)

#### Beyond Outcome Rewards: Constructing and Assigning Retrieval Credit for Search Agents (2026-10)
- **简介**：作者 Wenyu Huang、Xinyu Hou、Pavlos Vougiouklis、Ruofei Lai、Jeff Z. Pan。搜索 agent 通过迭代检索回答复杂多跳问题，RLVR 依赖稀疏的结果监督使信用分配困难、学习效率受限。论文系统研究中间监督如何改进搜索 agent 的 RL：考察一系列从中间检索步骤提供学习信号的奖励塑形与信用分配策略，并据此构建把中间信号与最终结果奖励结合的训练框架。在匹配训练条件下的多个 benchmark 上提升了总体搜索 agent 表现，并表明中间信号的选择及其信用分配的位置都会影响训练行为，说明奖励设计与信用分配是训练有效搜索 agent 的重要设计维度。（abstract 未给出具体数值）
- **arXiv**：[2610.10179](https://arxiv.org/abs/2610.10179)

#### CLIFT: Conformal Self-Verification for Web Agent Training and Test-Time Scaling (2026-10)
- **简介**：作者 Yifan Zhang、Yutong Dai、Viraj Prabhu、Zhiyuan Hu、Ran Xu、Zeyuan Chen。开源 web agent 已能执行真实浏览器任务，但 RL 训练仍依赖弱监督：二值任务成功太稀疏，前沿 LLM judge 又太贵且部署时不一定可用。**CLIFT** 围绕 conformal self-verification 构建：训练时 agent 回答关于自身 rollout 的自然语言验证问题，Compositional Conformal Certifier 只保留 URL 条件证据与训练期 judge 一致的问题信号，通过 polarity-aware lift 赋予带符号信任权重，并把得到的 verifier 分数融合进逐步奖励，且不会低于 judge 基线；测试时冻结同一认证问题库，作为 Conformal Trajectory Selection 的结构化证据，在 greedy rollout 与多样重试之间用保守多数投票选择，无需调用外部 judge。（abstract 被截断，具体数值以原文为准）
- **arXiv**：[2610.06829](https://arxiv.org/abs/2610.06829)

#### LexiHorizon: Stabilizing Reinforcement Learning for Long-Horizon Deep Search (2026-10)
- **简介**：作者 Zhiqing Nong、Liang Wen、Chao-Hsuan Liu。deep search agent 通过迭代检索、多跳推理与证据综合解决复杂知识任务，现有方法通常假设检索系统稳定、只做短程工具交互；但当检索对 query 措辞敏感时，需要反复改写 query，轨迹更长，训练面临持续的长程 query 探索与不断膨胀的检索内容管理。**LexiHorizon** 扩大轨迹上下文预算，用滑动窗口只保留最近工具观测同时保留推理历史，并引入 outcome-gated 的 search-effort reward：只对答案奖励非零的轨迹给予有界的工具调用奖励。在 XBench、WebWalkerQA、BrowseComp-ZH 上，所得 9B 模型一致优于其 base 模型和 MiroThinker-1.7-mini（abstract 被截断，具体数值以原文为准）。
- **arXiv**：[2610.05119](https://arxiv.org/abs/2610.05119)

#### Dr.Credit: Rubric-Grounded Process Credit Assignment for Deep Research Agents (2026-09)
- **简介**：作者 Yingjian Zhu、Zhenyi Wang、Jiaxin Guo 等针对 rubric 任务的 RL：rubric 分数通常只监督最终答案，现有过程 credit 方法又依赖标准答案，难以用于开放任务。**Dr.Credit** 用任务要求（rubric）作为最终答案评估和过程监督的共同参照：评估工具返回的信息相对于每条 rubric 已被接受支持的历史所提供的新增支持，从而区分新证据与轨迹中已有证据并识别部分支持，用以监督 deep research agent 的中间工具 turn。所得过程 advantage 与 GRPO 结果 advantage 结合。在四个域内/域外基准上，Dr.Credit 在所有主要指标与子指标上优于评测的开源 deep research 基线；8B 骨干的训练后 agent 平均表现可与评测的前沿闭源模型竞争，且在有限 research-turn 预算下证据获取更高效、报告质量更高。
- **arXiv**：[2609.34296](https://arxiv.org/abs/2609.34296)

#### Traverse: Learning When to Remember, Reset, and Redirect for Long-Horizon Web Search (2026-09)
- **简介**：作者 Jingyuan Ma、Lynx Aster、He Zhang 等指出，长程信息检索 agent 常积累嘈杂或误导的上下文，使早期错误持续存在、难以恢复。提出自主搜索 harness：agent 通过 Rubric、Answer、Verify 三种状态管理搜索——先定义有效答案的标准，在标准下搜索，再独立验证后决定终止或继续；并配备 Seal Memory 工具主动管理上下文。用 RL 训练该行为会引发 Seal Collapse，导致训练不稳定、agent 无法可靠学会何时如何使用记忆工具；论文用只训练上下文管理之后最终片段的简单策略解决。其 35B 模型在 BrowseComp 上达 72.83，优于同类开源系统，并在 BrowseComp-ZH、xbench、DeepSearchQA、WideSearch、金融调查与商品搜索上持续优于基座模型；消融表明自主压缩优于自动 compaction。
- **arXiv**：[2609.37082](https://arxiv.org/abs/2609.37082)

#### Reinforcement Learning with Verifiable Rewards for Small Search Agents (2026-09)
- **简介**：作者 Gaurisankar Jayadas、Aske Plaat 等。研究 reason-over-search 配方（Search-R1 式 RLVR）在 1B 以下小模型上是否可行。在 Qwen3.5-0.8B 上用 GRPO 加交错 Wikipedia 搜索工具、在 MuSiQue 上训练，仅改变三种奖励形态、每种三个随机种子，并在七个 QA 基准上评测。无需蒸馏，最佳运行平均 exact match 达 0.352，对比未训练的 0.092 提升 3.8 倍；而 Search-R1 风格的纯 exact-match 奖励在所有种子上都最差，甚至在 EM 指标上也最差。结论是小模型 RLVR 需要独立的奖励设计，而非简单缩小大模型配方。
- **arXiv**：[2609.28765](https://arxiv.org/abs/2609.28765)

#### IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis (2026-09)
- **简介**：作者 Xingyu Wu、Yuchen Yan、Zhengxi Lu 等（腾讯）。ReAct 式深度搜索 agent 存在角色耦合（一个策略同时承担规划、证据使用与综合）和上下文累积带来的噪声。**IterSynth** 采用角色解耦、基于摘要的范式：Planner 识别信息需求，Synthesizer 把证据整合进不断演化的摘要状态，以摘要作为搜索的持久状态。训练用 **RDPO**（Role-Decoupled Policy Optimization），结合终态结果奖励与 turn 级 rubric 评估，并按角色计算 advantage 以更精确地分配 credit。在 BrowseComp、Xbench-DS 等五个长程深搜基准上，IterSynth-8B 平均 50.7，超过此前最强 8B 以下 agent 4.2%；作为与模型无关的 prompting 范式，也能显著提升前沿闭源模型的 zero-shot 表现。
- **arXiv**：[2609.29444](https://arxiv.org/abs/2609.29444)

#### PrimeSeeker: Capability-Oriented Supervision for Deep Search Agents (2026-09)
- **简介**：作者 Linzhi Peng、Hanting Chen、Heng Chang 等。搜索 agent 常用合成问题训练，通过更大的证据图、更多跳数、更长轨迹提高难度，但这些全局属性只是局部检索能力的间接代理。论文提出 latent anchor reasoning：从描述性规格中解析出未命名的检索锚点，并将其转化为后续信息需求，据此把深搜分解为耦合操作链。**PrimeSeeker** 构造 web grounded 的锚点结构，并联合导出问题与参考证据骨架；骨架引导专家生成，并在 SFT 后复用于审计参考步骤覆盖以构造 RL 奖励。构建 9,221 条专家轨迹训练 30B 搜索 agent，在五个深搜基准上表现强劲，参考步骤优化进一步提升 SFT 策略，且轨迹检索冗余低，固定预算下以显著更少的工具调用保持较强的解覆盖（abstract 未给出具体数值）。
- **arXiv**：[2609.35816](https://arxiv.org/abs/2609.35816)

#### One Policy, Any Budget: Internalizing Budget-Aware Search via Reinforcement Learning (AnySearch) (2026-09)
- **简介**：Xiaowei Sun、Jin Li、Yili Hong、Yikun Fu、Yanghua Xiao。RL 已能让 LLM 搜索 agent 调用外部工具，但现有方法都在固定预算下训练，部署时约束变化便无法适应。**AnySearch** 用「训练脚手架 + 课程式 RL」让单个 policy 在任意预算约束下做 budget-aware search：第一阶段显式注入预算状态并配结构化推理 prompt，引导 agent 在线性衰减的预算下高效分配；第二阶段撤掉脚手架，让 agent 在自适应采样的预算约束下自主运行，以匹配真实推理条件。两阶段都用一个把答案准确率与预算效率通过绝对与相对信号耦合的复合 reward 优化，其中自适应权重对高准确率 query 放大效率信号、对低准确率 query 衰减它。七个通用与多跳 QA 基准上在所有预算尺度均优于基线，可泛化到训练范围之外的未见约束，并在不引入过多 token 开销的前提下取得更高的工具生产率（abstract 未给出具体数值）。
- **arXiv**：[2609.00813](https://arxiv.org/abs/2609.00813)

#### SearchWiki: Learning to Build and Navigate Knowledge Wikis for Active Information Seeking (2026-08)
- **简介**：IBM Research（Guransh Singh、Vishwajeet Kumar、Arkadeep Acharya、Jaydeep Sen、Sachindra Joshi 等）。扁平 RAG 把语料视为 chunk 袋，丢弃文档层级与跨文档结构。**SearchWiki** 是一个 harness 框架，把语料合成为分层、带类型、可导航的 wiki——文档综述、跨文档主题页、页级源记录三层，使初次查找失败后能逐步细化检索——并训练 agent **WikiResearcher-9B** 通过多轮工具使用来取证；其导航 policy 用 on-policy RL 优化，reward 为答案正确性、检索质量与 trajectory 效率的多分量组合。在 ViDoRe-V3（8 个领域）、FinanceBench 以及记忆类基准（LoCoMo、LongMemEval、PersonaMem-v2）上，这个 RL 微调的 Qwen 9B 模型显著超过同规模未训练基线，并追平或超过更大的外部模型（abstract 未给出具体数值）。
- **arXiv**：[2608.29953](https://arxiv.org/abs/2608.29953)

#### AgenticRag-R1: Agentic Reinforcement Learning with Stack Memory for Multi-Step Reasoning, Retrieval and Memorizing (2026-08)
- **简介**：Xinke Jiang、Yue Fang、Zhibang Yang、Jiaran Gao 等。RAG 系统在需要自适应检索与持续修订中间上下文的多步推理上表现不佳，而近期基于 RL 的 agentic RAG 方法多采用粗粒度动作空间与 trajectory 级 reward，导致 reward 分配偏弱、并偏向短 horizon 的模板化推理。**AgenticRag-R1** 通过记忆栈（memory stack）与细粒度动作空间把推理、检索、记忆三者深度整合，并用分层的 action-aware reward 与信息感知的 trajectory 拒绝策略支撑有效的长 horizon 学习。在多跳、开放域与 agentic 推理等多类基准、多种 backbone 规模上稳定优于强基线，并学到更鲁棒、可解释、记忆感知的推理行为（abstract 未给出具体数值）。
- **arXiv**：[2608.29622](https://arxiv.org/abs/2608.29622)

#### Harness-RL: Black-Box Reinforcement Learning with Action-Args Decoupling for Central-Agent Multi-Agent Harnesses (2026-08)
- **简介**：Xinke Jiang、Zhixin Zhang、Zhibang Yang、Jiaran Gao 等。在中心 agent 协调专用子 agent、工具与环境的多智能体 harness 中训练中心 policy 有两个难点：action 标签是低基数决策而其 args 是高维条件序列，用共享的序列级信号同时优化会产生冲突梯度；动态调度产生带分支、并行调用与上下文重写的互依赖会话，无法忠实压平为单条 token 序列。**Harness-RL** 把 Conflict-Aware Policy Optimization（**CAPO**）与接口级黑盒 trajectory 构造结合：黑盒部分采集 Interface Call Record、构建按会话的前缀树，并把结果 reward 与过程 reward 对齐到可训练 token；CAPO 用前向激活识别与 action token、args token 相关的参数分区，再把两者的 policy 梯度分别路由到对应子空间，同时支持仅训练中心 agent 与联合多智能体训练。七个多跳问答与 agentic 检索基准上，Qwen2.5-1.5B 与 Qwen2.5-3B 平均 F1 分别达 42.93 与 47.79，消融验证了 CAPO 的贡献并显示该设置下仅优化中心 agent 更优。
- **arXiv**：[2608.29641](https://arxiv.org/abs/2608.29641)

#### GTA-RAG: Graph-Trajectory-Augmented Reinforcement Learning for Multi-Turn Retrieval-Augmented Reasoning (2026-08)
- **简介**：Jun Chen, Yongchao Liu, Pengyu Qiu, Xiao Luo 等。agentic RAG 的 RL 方法通常只用最终答案 reward，监督稀疏且不关心模型是否真的检索到了所需证据链。**GTA-RAG** 从实体—文档图上采样连通的文档路径，合成多跳 QA trajectory 并用实际部署的 retriever 校验其可执行性，从而获得 trajectory 级监督；随后用 GRPO 配合 trajectory-guided reward（同时鼓励答案正确与命中目标证据文档）优化检索 policy，再在自然 QA 实例上做答案 reward 训练。三个多跳与两个简单 QA 基准上，Qwen2.5-3B 与 Qwen2.5-7B 两种 backbone 均一致超过基于 RL 的 RAG 基线，同时大幅提升证据链覆盖率（abstract 未给出具体数值），代码已开源。
- **arXiv**：[2608.22479](https://arxiv.org/abs/2608.22479)

#### MetaRAG: Belief-Action Aligned Policy Optimization for Agentic RAG (2026-08)
- **简介**：Qiuyi Qi, Tian Liang, Jiamu Wang, Qiang Zhu 等。agentic RAG 需要不断决定「继续搜索还是直接作答」，已有 RL 方法依赖外部监督，忽视了 agent 对当前证据是否充分的内部信念。**MetaRAG** 把搜索决策质量重述为 belief-action alignment：用 Verify-first Action Generation 在每个实际动作前先产生显式验证过程，用 Internal Belief Probing 从同一问题—历史上下文中估计 policy 模型自身的可答性信念，据此导出一致性 reward，并以答案正确性对该 reward 做门控，避免强化「内部自洽但答案错误」的 trajectory；belief probe 仅在训练时使用，不带来任何推理开销。七个公开 QA 基准上准确率-效率折衷一致优于强 RL agentic RAG 基线，增益可迁移到深度研究设置、不同优化器与多种 backbone（abstract 未给出具体数值）。
- **arXiv**：[2608.24214](https://arxiv.org/abs/2608.24214)

#### CAS: Conformalized Agentic Search via Adaptive Retrieval and Policy Weighting (2026-08)
- **简介**：Zixi Zhu, Jiayuan Su, Jian Zhang, Hongwei Wang 等。search agent 在 RL 微调中面临可靠性问题：启发式 Top-K 检索常导致关键证据丢失或引入噪声，而渐进 RL 带来的过度自信会引发幻觉答案与冗余搜索。**CAS** 把 conformal prediction 引入检索与训练两侧：检索侧用 Adaptive Prediction Set 把统计覆盖率转成动态文档截断，构造大小自适应的预测集；训练侧用 Adaptive Conformal Inference 动态构造可控覆盖率的预测集来量化答案置信度，并在 GRPO 目标中惩罚低置信 trajectory，使模型只从可靠 rollout 中学习。单跳与多跳 QA 数据集上推理准确率明显提升，同时冗余工具调用大幅减少（abstract 未给出具体数值），代码已开源。
- **arXiv**：[2608.20771](https://arxiv.org/abs/2608.20771)

#### Wuying-Browser-Agent: Real-World Centric Fundamental Long-Horizon Browser Agents (2026-08)
- **简介**：AIMAE Team（Tianxiang Chen, Yan Cheng, Zhangye Han, Xiaowei Li 等）。browser agent 在短而干净的演示上表现良好，但真实部署要求在活网站上维持数十步决策、从错误中恢复并应对复杂 UI，作者主张弥合这一差距需要在执行、监督、优化与评测各层同时对齐而非单纯扩规模。**Wuying-Browser-Agent** 为此提供统一框架：结构化 browser harness 给出稳定执行原语与面向决策的上下文管理；RUIC-SFT（Reflection and UI-specialized Curriculum SFT）显式在恢复类 trajectory 与复杂 UI 交互上做课程式训练；DAO-GRPO（Divergence-Aware Online GRPO）用基于势函数的 reward shaping 与 divergence-aware step 加权改进长 horizon credit assignment；并发布 BrowserBench——双语真实网页基准，350 个任务平均 37.9 步。Wuying-Browser-Agent-27B 在 WebVoyager 上 80.6%、Online-Mind2Web 上 66.7%、BrowserBench 上 65.1%，创 browser-use 基准的开源 SOTA；同一 pipeline 迁移到通用 agentic 任务，在 Tau2-Bench、Claw-Eval 与 BFCL-v4 上平均得分 73.8。
- **arXiv**：[2608.17319](https://arxiv.org/abs/2608.17319)

#### Beyond Outcome Rewards: Step-Level Self-Distilled Policy Optimization for Deep Search Agents (SSPO) (2026-08)
- **简介**：Haoze Wu、Chuqiao Kuang、Tianyi Zhuang、Xiaoguang Li。深度搜索 agent 的 trajectory 常跨数十步，标准 RL 却只给一个 outcome reward，对 credit assignment 过于稀疏；直接把 on-policy self-distillation（OPSD）用模型自身 logits 作稠密 token 级 teacher 又会引入根本性张力——teacher 掌握正确答案等特权信息，其分布系统性偏离学生基于探索的推理，朴素蒸馏只会让学生继承这种信息不对称而非学会更好的搜索策略。作者一是构造 Evidence Anchors，即从网页抽取的简短 step 级证据片段作为特权信息，只暴露关键推理步而不泄露整条答案路径；二是提出 **SSPO**，把 teacher-student 分歧转成 GRPO 内的 step 级 advantage 权重，且只施加于错误 trajectory，从而把「往哪个方向更新」（由 outcome reward 决定）与「每步更新多少」（由 teacher 调制）解耦，正确 trajectory 完全不动以保留多样性。在 Qwen3-8B 上于 BrowseComp、GAIA、FRAMES 持续优于 GRPO，并追平或超过 gradient steps 翻倍的 GRPO，而每步只多一次前向、额外开销约 5%。
- **arXiv**：[2608.12764](https://arxiv.org/abs/2608.12764)

#### LoongReflect: Boosting Long-Horizon Reflection in Search Agents via Global Perspective Distillation (2026-08)
- **简介**：Zhixin Zhang、Xinke Jiang、Xu Chu、Yasha Wang 等。反思（评估 trajectory 进展、识别缺失证据与不可靠中间状态、决定继续/修正/放弃当前分支）是长 horizon agent 的关键能力，但反思发生在当前分支的局部，其效用却只能由对最终 trajectory 结果的贡献来判定，这种 local-global 错配使 outcome-based RL 对反思决策只能提供局部、稀疏且延迟的监督。**LoongReflect** 把反思形式化为 memory-control 策略：agent 在可回退的 trajectory tree 上使用显式的 reflect 与 backtrack 动作，reflect 把已验证事实、缺失证据与分支特有风险整合进工作记忆，backtrack 把不可靠分支移出活跃上下文、只保留一条简短的纠错教训。训练用 look-ahead、extragradient 式协调机制融合两路信号：快通道从特权 teacher 蒸馏具全局视角的反思行为，监督仅施加在 reflection 与 backtracking token 上；慢通道用 outcome-based GRPO 优化完整 trajectory，使局部控制决策与最终任务成功对齐。在多跳 RAG 与数学推理 benchmark 上一致优于纯 outcome RL 与 self-distillation 基线（abstract 未给出具体数值）。
- **arXiv**：[2608.11967](https://arxiv.org/abs/2608.11967)

#### HindSearch: Trajectory-Level Hindsight Critique for Search-Augmented Reinforcement Learning (2026-08)
- **简介**：Haowei Liu、Jiamian Wang、Hsin-Tai Wu、Zhiqiang Tao、Yi Fang。指出 search-augmented LM agent 通常只用二值 exact-match reward 训练，丢掉了失败 trajectory 中关于「为什么失败」的绝大部分信息。**HindSearch** 为 GRPO 加入 hindsight 自蒸馏：每轮 rollout 后由一个冻结 judge 借助 gold answer 为每条失败 trajectory 写简短 critique，该 critique 再对 student 的 search 动作提供额外的 on-policy 蒸馏信号。在标准七基准套件、Qwen2.5-3B-Instruct 上取得 39.4% 平均 EM，优于此前 search-RL 基线；若切断 judge 对 gold answer 的访问，增益几乎全部消失，从而将收益来源定位到 hindsight 本身。
- **arXiv**：[2608.01597](https://arxiv.org/abs/2608.01597)

#### Contextual Information Policy Optimization for Search Agents (CIPO) (2026-08)
- **简介**：Xingyu Guo、Wei Chen、Linlin Yang、Baochang Zhang。指出现有 search agent 的 RL 只奖励最终答案正确性或中间进展，从不直接评估检索之后的动作是否 grounded 在检索证据上；这种错配助长 prior-driven reasoning——agent 先凭内部知识形成结论、再把检索当作确认手段，造成确认偏误与证据利用低效。提出 **CIPO**：面向证据的 RL 框架，对受检索信息影响的推理动作赋予稠密的 turn-level credit，并与全局 outcome reward 结合以保住答案正确性，从而抑制脱离证据的猜测、鼓励检索事实能引导或修正后续推理的 trajectory；无需人工过程标注，也不需额外 reward model。七个 in-domain 与 out-of-domain 基准上降低了 prior-driven reasoning 比例，多数任务表现优异。
- **arXiv**：[2608.06128](https://arxiv.org/abs/2608.06128)

#### CRISP: Critical Step Perception for Training Efficient Deep Search Agents (2026-08)
- **简介**：Haosi Mo、Zihao Yan、Ruiqing Zhang、Xuebo Liu、Min Zhang 等。针对 deep search agent 轨迹冗长、充斥重复查询与无关观测的成本问题，指出现有效率方法一律鼓励少用工具，会连带压制真正在收集必要证据的步骤。**CRISP** 先用 Backward Evidence Induction 构造 critical-step 标签：由强模型从最终答案出发反向遍历完整搜索 trajectory，逐步判断每次工具交互是否提供或保留了支撑答案的证据；再把这些判断蒸馏进一个更小的 critical-step recognizer，使全轨迹分析单次前向即可完成；策略优化阶段仅对成功 rollout 施加 efficiency-aware reward。在 BrowseComp 与 HLE-Verified 上保持有竞争力的答案准确率，同时平均交互 turn 数分别减少 15.1% 与 33.2%。
- **arXiv**：[2608.01867](https://arxiv.org/abs/2608.01867)

#### Training Documents Reranker with Search Rubrics for Deep Research Agent (RubricRanker) (2026-08)
- **简介**：Wenhan Liu、Yu Lu、Qiaolin Xia、Zhicheng Dou 等。指出为 deep research agent 供料的检索系统通常只按相关性匹配挑文档，而逐篇都相关的 top-k 合起来未必构成能满足 agent 查询复杂信息需求的「集合」（如需多样、简洁、权威）。作者提出 search-oriented rubrics，显式定义每个 agent 查询下高质量文档集应满足的要求，rubric 以层次结构组织并由强 LLM 合成；在此基础上训练 reranker **RubricRanker** 从召回结果中挑高质量子集，训练分两阶段：rubric 引导的监督微调与基于 rubric 的强化学习。RubricRanker 在四个 deep research 基准上比最强基线高 2.6 个点，并在五个 RAG 基准上泛化良好。
- **arXiv**：[2608.03527](https://arxiv.org/abs/2608.03527)

#### EviBack: Search-Agent Reinforcement Learning via Evidence-Constrained Teacher Backoff (EviBack) (2026-07)
- **简介**：面向可靠 agentic RAG 的搜索 agent RL 工作，提出 **证据约束的 teacher backoff（Evidence-Constrained Teacher Backoff）** 机制训练搜索 agent，强调检索证据对最终结论的约束与可靠性（agentic RAG 场景）。当稀疏结果奖励难以驱动搜索行为时，用受证据约束的 teacher 信号在关键节点"退避"提供更可靠的学习方向。
- **arXiv**：[2607.23955](https://arxiv.org/abs/2607.23955)

#### AREX: Towards a Recursively Self-Improving Agent for Deep Research（AREX） (2026-07)
- **简介**：Shuqi Lu、Zheng Liu、Zhicheng Dou、Di He 等（BAAI/人大等大团队）提出的递归自改进（RSI）深度研究智能体家族。AREX 交替运行"内层研究循环（取证＋构造暂定答案）"与"外层自改进循环（按约束逐条审计、定位未解主张、发起定向后续检索）"，并学习一个自主 context-update 工具把增长的交互史压缩成保留已验证证据/未决约束的紧凑改进态；训练经 agentic mid-training＋长程 RL，并对"获取决定性证据/纠正错误方向"的关键步加权以缓解稀疏终局奖励。实例化 4B 稠密与 122B-A10B MoE 模型，在 BrowseComp、WideSearch、DeepSearchQA、HLE 等基准上大幅超越同规模基线。
- **arXiv**：[2607.21461](https://arxiv.org/abs/2607.21461)

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

#### SUMONE: A Simple and Efficient Summarizer for Plug-and-Play Agent Memory (2026-11)
- **简介**：作者 Jizhe Zhang、Ang Li、Chenheng Zhang、Yisen Wang。长程 agent 的交互历史不断膨胀，上下文压缩必不可少，而现有学习式压缩常依赖下游任务结果或求解器行为，使训练与特定任务/agent 耦合。**SumOne** 是独立的摘要器，通过 masked reconstruction 训练以保留 agent 历史信息：给定已过期历史的摘要，冻结的 reconstructor 恢复原轨迹中随机遮盖的部分，重建质量加长度惩罚构成任务无关的奖励，用 GRPO 优化摘要器，实现无需参考摘要、也无需下游任务执行的自监督。推理时仅保留轻量摘要器即可即插即用压缩。在长程搜索与软件工程任务上，相对完整上下文，DeepSearchQA 估计推理成本降约 78%、SWE-bench Verified 降约 75%，绝对分数下降 0.04；同一检查点可跨下游模型与 harness 迁移。
- **arXiv**：[2611.06445](https://arxiv.org/abs/2611.06445)

#### MEMENTO: Learning Context Management for Long Context Processing Agents (2026-10)
- **简介**：作者 Yunfan Zhang、Taehoon Hwang、Kathleen McKeown、Smaranda Muresan。处理长输入的 LLM agent 需要做 context management；常见的 compaction 让模型用自生成笔记替换历史，但 GPT-5.5、GLM-5.3 等现成模型常写出无效笔记，仅靠 RL 也很少发现有用的笔记行为。**MEMENTO** 基于 Qwen3-8B：先在带事后（hindsight）生成笔记的合成轨迹上微调以自举记笔记能力，再用结合答案正确性、笔记质量启发式与置信度校准的奖励做 RL；虽只训练压缩到新窗口，测试时可扩展出 context folding 与基于置信度回溯等策略。在四个书籍级问答 benchmark（书长最多 2.5M token）上，Qwen3-8B 准确率由 36.9% 提升到 67.8%，上下文始终不超过 20K token，优于同 harness 下的 GPT-5.5 agent（13.1%）与完整书籍上下文（56.7%），成本不足其 1/50。
- **arXiv**：[2611.00244](https://arxiv.org/abs/2611.00244)

#### Don't Lose the Thread: Empowering Long-Horizon LLM Agents with Cognitive Resource Self-Allocation (CORAL) (2026-10)
- **简介**：作者 Zheng Zhu、Xiaozhuan Liang、Xi Chen、Bei Yu、Jiaya Jia，COLM 2026 接收。LLM agent 在长程任务中常因认知过载而失败：工作记忆被不断增长且无关的信息占满，稀释注意力并妨碍规划。论文提出 **CORAL**（Cognitive Resource Self-Allocation），以 agent 可调用的工作记忆管理工具集实现：agent 可 checkpoint 任务进展与已验证事实，并自适应地重置杂乱上下文、只保留 checkpoint 信息，从干净状态开启新的求解 episode；并用 Multi-episode Agentic Reinforced Policy Optimization 算法增强 checkpoint 能力。8B 模型在 GAIA 上平均准确率 42.8%，在最难的 Level 2、3 上提升明显，输入 token 比 ReAct agent 少 6-7 倍；也可泛化到多跳知识 QA 和更难的 BrowseComp，使强专有 agent 提升 8.0 点。
- **arXiv**：[2610.40350](https://arxiv.org/abs/2610.40350)

#### From Interaction to Capability: Task Advancement as State and Supervision for GUI Agents (TAM) (2026-10)
- **简介**：作者 Ziyang Chen、Qian Li、Chengwei Qin。GUI memory 应保留过往动作的结果并学习可复用操作，但在长程工作流中，agent 需跨应用使用信息，且有用操作与失败尝试交织。论文提出 **Task Advancement Memory (TAM)**，将 task advancement 定义为经验证的任务进展：执行时 Task Advancement State 表示已验证的进展如何改变剩余目标，使 agent 能跨应用建立在先前成果之上；每个任务后，advancement supervision 识别并抽象出产生经验证局部结果的交互，所学操作保留所需输入、预期结果与验证条件的关系，因而成功和失败任务中局部成功的操作均被保留。在 MobileWorld GUI-only 任务上，仅 Advancement State 就把成功率从 12.0% 提升到 28.2%；完整 TAM 比最强基线任务成功率相对提升 17.6%，每个已解决任务的生命周期 token 降低 38.7%；在 SPA-Bench 单应用英文 Level 3 上达到 SOTA。
- **arXiv**：[2610.37482](https://arxiv.org/abs/2610.37482)

#### Uni-Mem: Learning a Long-Horizon Agentic Memory Manager with Unified Online Policy Optimization (2026-10)
- **简介**：作者 Zhiqian Xia、Jielong Tang 等。现有可学习记忆方法要么只优化记忆构建，要么把检索侧在线 RL 与离线构建监督拼接，限制了两个阶段的协同适应。论文探索端到端在线 RL 训练统一记忆管理器 **Uni-Mem**，在共享策略下同时学习构建与检索。由于直接用 GRPO 会遇到混淆的组内比较和长记忆管理轨迹上的稀疏反馈，提出 **Stage-Decoupled GRPO**：交替优化构建与检索，并在每个 rollout group 内控制另一阶段带来的变异，得到阶段专属学习信号；再提出 **Event-Wise Credit Assignment (ECA)**，把反馈定位到负责的事件并把事件级 advantage 传播到其组成操作。在 LoCoMo 和 LongMemEval 上优于最强的 RL 记忆基线。（abstract 未给出具体数值）
- **arXiv**：[2610.29671](https://arxiv.org/abs/2610.29671)

#### All-in-One Memory: Process-Rewarded Reinforcement Learning for Memory Agents (AiO-Mem) (2026-10)
- **简介**：作者 Bingbing Wang、Jingjie Lin、Weiming Qiao、Jing Li、Ruifeng Xu。现有记忆方法把检索与记忆构建分开优化，缺乏统一奖励空间评估搜索、写入和停止决策。**AiO-Mem** 把这些操作组织为共享证据状态上的转移，在统一奖励空间中直接评价每个动作的语义质量：先将异构历史编译为可溯源的多视角事件，由统一记忆策略 actor 展开完整交互轨迹；再用经证据落地 SFT 与轨迹级 DPO 训练的 trajectory-conditioned process reward model 为轨迹打过程分，聚合为 action-balanced 轨迹奖励，并通过 Process-Aligned DAPO（PA-DAPO）优化记忆策略。在长上下文记忆基准上，答案准确率一致优于静态记忆系统及 RL 训练的记忆 agent（abstract 未给出具体数值）。
- **arXiv**：[2610.21147](https://arxiv.org/abs/2610.21147)

#### Hy-Memory-30B-A3B: Attributed Reinforcement Learning for Long-Term Memory Writing (2026-10)
- **简介**：作者 Tianxiang Fei、Jiazheng Kang、Mao Zheng、Xingyan He、Xiang Yu。外部记忆系统的质量取决于执行写入路径的模型（保留什么、如何表示、如何用新证据更新持久状态），而现有框架通常在特定 prompt 后接通用大模型，成本高且专用 writer 难以跨协议迁移。**Hy-Memory-30B-A3B** 是跨 Mem0、MemOS 和 Hy-Memory 联合训练的紧凑长期记忆 writer：Memory On-Policy Distillation 在 student 生成的状态上教授各框架特定的写入协议，随后 Memory RL 用稠密逐 chunk 探针、跨 chunk 归因奖励、过量存储控制和 chunk 级 GRPO 优化下游可回答性，并跨窗口携带策略写出的记忆库使训练与部署对齐。在固定 20% 基准子采样上，它在四个评测 harness（含留出的 EverMemOS）全部排名第一，宏平均 55.49%，对比 Claude Opus 5 的 54.62%。
- **arXiv**：[2610.22196](https://arxiv.org/abs/2610.22196)

#### MemComposer: Learning to Assemble Memories for LLM Agents via Reinforcement Learning (2026-10)
- **简介**：作者 Yixuan Xie、Xueqiang Xu、Xinyi Fan、Xinxi Lyu、Tao Feng、Xing Fan。agent 记忆使 LLM agent 无需更新参数即可复用过往经验，但语义相似并不保证记忆有用，下游表现取决于记忆如何被选择与呈现。**MemComposer** 联合考察当前 query 与检索到的候选，决定纳入哪些经验、选多少、如何排序；用基于 GRPO 的 **Experience Policy Optimization（ExPO）** 训练，通过完整 agent 交互评估所组装的记忆上下文，并用任务成功反馈优化组装策略。在 tau^2-bench 三个域、两个冻结 executor 上，MemComposer 均优于各自最强基线，同时平均记忆 token 用量分别减少 16.9% 与 4.9%；所学策略还能迁移到未见的域、executor 和记忆库，并受益于更强模型生成的记忆。
- **arXiv**：[2610.21091](https://arxiv.org/abs/2610.21091)

#### AMBER: Training Long-Horizon Web Agents through Append-Only Memory (2026-10)
- **简介**：作者 Chinmay Savadikar、Zhaoyu Zhang、Mingyu Zhao、Shuang Xie、Han Li、Tianfu Wu。长程 agent 的交互历史会很快超出上下文预算，需要在长时间内保持事实、记住执行错误和纠正反馈。覆写式记忆原则上能保留 append-only 记忆所能保留的一切，但必须学会把每个事实传递过每次重写，难以从稀疏结果奖励中学到；对 web agent，作者发现训练出的覆写记忆会删除轨迹所需的关键信息和环境的纠正反馈。**AMBER**（Append-only Memory Bank for Evidence Retention）让 agent 联合学习推理、行动与写入自由格式记忆，同时用 append-only 规则从构造上保证信息保留。（abstract 被截断，实验数字以原文为准）
- **arXiv**：[2610.07118](https://arxiv.org/abs/2610.07118)

#### Continuous Context Management (2026-09)
- **简介**：作者 William Hoy、Jingxuan Fan、Nurcin Celik、Xu Pan 研究长程 LLM agent 的 **Continuous Context Management（CCM）**：每轮都做 compaction，而非在预定阈值触发才压缩，agent 每轮输出更新后的记忆与环境动作，下一轮 prompt 仅含原任务、保留的记忆与最新观测。无微调时在 TerminalBench-2 上用 Claude Sonnet 4.6、Opus 4.6、GLM-5、Kimi K3 评估，CCM 显著降低累计输入量与活跃 prompt 大小，多数模型任务成功率下降，Kimi K3 保持不变。为提升开源模型的 CCM，采用 GRPO 加特权完整历史蒸馏：冻结的学生初始模型在由该学生 rollout 重建的完整历史下为采样动作打分，提供稠密动作 token 监督，无需另做教师 rollout。WebShop 上显著优于 GRPO，对 Qwen3-4B-Instruct 超过完整历史 GRPO（Qwen3-8B 则未超过）；Endless Terminals 上较 GRPO 略有提升，两种 CCM 策略均优于未训练的完整历史基线。
- **arXiv**：[2609.35540](https://arxiv.org/abs/2609.35540)

#### PSD: Pseudo Self-Distillation of Memory Representation Capabilities for LLM Agents (2026-09)
- **简介**：作者 Pirzada Suhail、Menglin Xia、Xuchao Zhang 等。agent 记忆系统的构建与维护依赖反复调用大型闭源模型，成本高昂，而闭源模型不暴露 logits。**PSD** 让同一个小模型扮演两个角色：teacher 看到包含 oracle 答案作为参考上下文的特权 prompt，student 只看任务 prompt，student 学习复现 teacher 的输出分布，从而在不接触 oracle 内部的情况下把 oracle 引导的行为吸收进自身权重，用于构建层级记忆表示。在 LoCoMo 上，PSD 训练的 Qwen3-0.6B、1.7B、4B 在下游检索上匹配或超过 GPT-4.1-mini 而部署成本低得多（off-policy PSD 在多数条件下最强），且仅在 LoCoMo 上训练即可迁移到 LongMemEval。
- **arXiv**：[2609.23449](https://arxiv.org/abs/2609.23449)

#### Just-in-Time Memory: Learning to Curate Task-Adaptive Memory for LLM Agents (JitMem) (2026-09)
- **简介**：作者 Yefan Zhou、Yang Li、Zeyu Leo Liu、Semih Yavuz、Shafiq Joty。现有 agent 记忆多在写入时整理：任务完成后把轨迹提炼成固定产物（反思、工作流、技能等）再按相似度检索，需要在未知未来 query 时决定记什么，信息被不可逆丢弃，而且学习写入时的整理器要面对长程 credit assignment。论文改为保留原始轨迹、把整理推迟到读取时：给定检索到的轨迹和新任务，memory curator 合成紧凑、任务自适应的 payload，因其在同一任务上被消费，可直接用即时任务成功训练，避免延迟的效用信号。**JitMem** 在 ALFWorld、WebShop、$\tau^2$-bench 上稳定优于无记忆 agent 以及启发式和学习式写入时记忆方法，分别比最强基线高 16.2、16.3、3.9 个绝对成功率点；未训练的 curator 已具竞争力，训练后进一步提升。
- **arXiv**：[2609.27334](https://arxiv.org/abs/2609.27334)

#### MemCalib: Benchmarking and Optimizing Memory Use in LLM Agents (MemCalib-RL) (2026-09)
- **简介**：作者 Ruike Cao、Fanyu Zhao、Fugen Yao 等。agent 记忆的效果取决于底层 LLM 是否让上下文中的每条记忆具有恰当影响，这一能力长期被忽视。**MemCalib** 是基于真实记忆系统场景的基准；结果显示前沿开源/闭源模型常过度或不足使用记忆，导致回答有偏、质量低。对 GRPO 和 on-policy self-distillation 等常见后训练算法的实验显示存在明显的方向偏斜：模型在一个方向改善、另一个方向恶化。为此提出 **MemCalib-RL**：有序双向反事实 credit assignment 算法，分离过度使用与不足使用信号，并通过精确的 atom 消融把 credit 定位到回答 token。在 Qwen3-8B、Ministral-3-8B-Instruct、Qwen3.5-35B-A3B 上取得最佳整体表现，更好平衡两种偏差，收益可泛化到外部基准。
- **arXiv**：[2609.24259](https://arxiv.org/abs/2609.24259)

#### Interactive Memory Learning for Long-Term Conversations (ICML) (2026-09)
- **简介**：作者 Cai Ke、Jiangyue Yan、Han Zhang 等。现有长期对话 agent 的记忆多是静态启发式的被动归档，不能自演化或随用户需求演变而调整记忆管理。**ICML**（InteraCtive Memory Learning）是多智能体框架，把记忆机制从被动归档转为可学习的交互式记忆策略：先用 session 合成流水线生成专家数据以便在未见场景快速测试期适应；随后通过在线 RL，让 Planner agent 选择性编码高价值信息、Trigger agent 动态检索以优化响应质量，两者在持续交互反馈中共同演化；并通过延迟 reward 机制把未来反馈回传到早期的存储决策，使记忆策略与用户期望精确对齐。实验显示显著优于强 baseline，且响应质量随交互累积持续提升（abstract 未给出具体数值）。
- **arXiv**：[2609.17088](https://arxiv.org/abs/2609.17088)

#### Rollback the World, Keep the Reflection: Rollback-Induced Reflection for Long-Horizon LLM Agents (RIR) (2026-09)
- **简介**：作者 Yi Yu、Liuyi Yao、Yaliang Li 等。长程 LLM agent 中单个错误动作会改变后续状态与观察、使错误累积；现有方法要么只修正上下文而不修复被改变的环境状态，要么恢复早期状态却丢弃有用经验。作者把可靠恢复视为 rollback 边界控制问题：联合决定何时干预、从何处恢复、恢复后保留哪些信息。**RIR**（Rollback-Induced Reflection）把执行恢复到选定的先前状态，同时携带从被放弃轨迹中提炼的可复用知识指导后续决策，并用关于回滚深度与保留记忆的统一算子刻画恢复。在三个长程 benchmark 上，多种 LLM backbone 的平均任务表现均得到提升，结构化反思记忆保留有用经验，选择性回滚实现高效恢复。
- **arXiv**：[2609.18304](https://arxiv.org/abs/2609.18304)

#### Elastic Horizon: Discovering the Effective Interaction Frontier in Agentic Reinforcement Learning (2026-09)
- **简介**：Gangyi Zhang、Junjie Meng、Letian Zhang、Chongming Gao 等。扩大交互 horizon 能提升长 horizon agent，渐进扩张的课程也优于固定 horizon，但现有调度都是开环的——单调增长到人工设定上限，没有机制判断何时扩张已不再有收益。作者提出「有效交互前沿」假设：存在一个动态边界，越过后额外交互收益递减而成本线性增长；**Elastic Horizon** 用成功 trajectory 长度的第 90 百分位在线追踪该边界，构成闭环控制器。在 AppWorld 与 BFCL 上固定 horizon 扫描确实呈现明显饱和平台；Elastic Horizon 无论从容量不足还是过剩的初始化出发都能把 horizon 稳定在饱和带内，在 7B 与 14B backbone 上取得最佳成功率，并节省最多 25% 的每步 trajectory token。
- **arXiv**：[2609.07247](https://arxiv.org/abs/2609.07247)

#### PARSER: Read in Parallel, Reason in Depth for Long-Context LLM Agents (2026-09)
- **简介**：Kun Li、Zexuan Qiu、Tianhua Zhang、Irwin King、Helen Meng。顺序式记忆 agent 逐块读长文档并维护紧凑记忆状态，把文档遍历与推理深度耦合在一起，导致对证据位置敏感且推理延迟随文档长度线性增长。**PARSER** 将读与推解耦：一组各绑定单个 chunk 的轻量 subagent 并行读完整篇文档，lead agent 则通过迭代的 scatter–gather 轮次深入推理——每轮向全部 subagent 广播查询、汇聚返回证据，并据已发现内容构造更深的后续查询；所有可学习行为集中在 lead agent 并用 RL 优化，subagent 保持冻结的现成模型。在 context 从 7K 到 896K token 的多跳 QA 上，4B backbone 的 PARSER 平均比最强顺序记忆基线高 5.7 个点、在 896K token 处高 12.0 个点；9B backbone 超过 DeepSeek-V4-Pro 6.3 个点；受控实验表明它对证据位置、顺序与距离扰动稳健（顺序方法在这些条件下准确率剧烈波动），并把推理延迟最多降低 11 倍。
- **arXiv**：[2609.06702](https://arxiv.org/abs/2609.06702)

#### MEMO: Multimodal Evidence Memory Organization for Long-Horizon LLM Agents (2026-09)
- **简介**：Xian Gao、Jinpeng Wang、Jiacheng Ruan、Ting Liu 等。长期运行的 agent 依赖外部记忆，但交互 trajectory 的持续累积与有限 context 容量存在根本张力：难点不只是检索相关记录，还要在预算下选出必要证据并以合适模态组织——纯文本保真但线性 token 表示使不同重要度内容以近乎等价的单位成本争夺 context，视觉化读出可用二维版式凸显结构却在渲染压缩中丢失细节。**MEMO** 用一个训练过的证据抽取器选出相关记忆块并构成带来源信息与呈现要求的证据单元，再由训练过的 query 条件化 memory manager 把每个单元分配到文本、视觉或双通道载体并选择匹配证据结构的版式，最后由确定性构造模块生成文本包与视觉页；memory manager 以离线 reader 度量记忆方案效用的反馈来训练，使保留与呈现决策对齐下游使用。在 HotpotQA、2WikiMultiHopQA、LoCoMo、ALFWorld 四个基准与多种 reader 后端上，MEMO 用更少记忆 token 提升下游表现（abstract 未给出具体数值）。
- **arXiv**：[2609.07471](https://arxiv.org/abs/2609.07471)

#### Explore More, Drift Less: Outcome-Only Reinforcement Learning Can Suffice for Long-Horizon Interactive Agents (CANOPY) (2026-09)
- **简介**：阿里巴巴（Liming Pu、Xiaoxia Li、Yifu Liu、Teng Cao 等）。业界普遍认为 outcome-only RL 在小开源模型上很快触顶，因此转向更密 reward、SFT 先验、技能库、精选记忆或多智能体编排来补偿；作者认为这个天花板是两种实践缺陷造成的假象：signal starvation——group-relative RL 在稀疏结果 reward 下只有当某任务的 rollout group 同时含成功与失败时才有梯度，探索规模不足恰好让最难、最有信息量的任务失声；policy drift——在小任务池上榨取过多更新会损伤 policy 本身，无锚定的目标会让采样分布在饱和期崩塌。**CANOPY**（Coverage-ANchored On-PolicY RL）以极简协议同时对症：扩大同任务探索直至自然信号重现，所有更新保持 on-policy、KL-anchored 且只作用于 agent 自身的 action token，再在测试期兑现更大的交互预算。仅靠环境交互训练（无任务专用监督、无辅助 credit 信号、无复杂 scaffolding）的 Qwen3-14B 在长 horizon 交互式编码基准 AppWorld 上登顶公开榜（2026 年 2 月，Test-Normal TGC 86.9、Test-Challenge 67.6），同样设计原则使 Qwen3.5-9B 在 SWE-bench Verified 上提升 16.6 个点。
- **arXiv**：[2609.01245](https://arxiv.org/abs/2609.01245)

#### ContextPilot: Teaching Agents for Proactive Context Management via Fine-grained RL (2026-08)
- **简介**：腾讯（Zhuoshi Pan、Qizhi Pei、Junru Lu、Honglin Lin 等）。主动式上下文管理已允许模型用专门工具编辑自己的工作上下文，但仍有三个局限：工具集局限于搜索、删除、摘要，缺少全局规划、长期记忆与自适应压缩；探索时把影响异质的上下文管理动作一视同仁；RL 中把 trajectory 级 reward 粗粒度地摊给所有中间编辑动作。**ContextPilot** 一方面把工具集系统扩充为规划、长期记忆与软性上下文卸载，另一方面提出针对上下文管理的 RL 方法：用 context 与 entropy 的变化识别关键编辑决策以触发分支采样，再从所有经过该编辑动作的分支 trajectory 估计 action 级 advantage。长上下文 QA 与 deep search 任务上，它以更紧凑的工作上下文取得更强性能，跨多种 base model 与基准均稳定优于既有基线（abstract 未给出具体数值）。
- **arXiv**：[2608.28476](https://arxiv.org/abs/2608.28476)

#### ERSkill: Evolving for Skill-Guided Adaptive Memory Retrieval (2026-08)
- **简介**：Haolong Chen、Liang Zhang、Zhuo Li、Lei Xue 等。LLM agent 越来越依赖长期记忆维持持续交互，但支配记忆的检索机制很少被当作可进化组件，静态检索在异构记忆 query 上表现受限，因为不同 query 需要不同的证据构造策略。**ERSkill** 把交互历史编译成结构化记忆库，把检索行为表示为由基础 primitive 组合而成的可执行 skill；推理时由一个训练好的 router 为每个 query 动态匹配最优 skill，构造定制化证据供答案生成。为支持持续改进，训练阶段让 skill 集合与 router 协同进化：用 experience trie 高效记录已探索的检索路径，并用 double-frontier 机制把新 skill 能力的扩张与面向 router 的稳定部署安全解耦。多个 agent 记忆 benchmark 上显著优于强非进化与自进化基线，F1、BLEU-1 与 LLM-judge 的总体平均分在 Qwen3-Next-80B-A3B-Instruct 上提升 31.3%、在 GPT-5.4-nano 上提升 28.1%。
- **arXiv**：[2608.12720](https://arxiv.org/abs/2608.12720)

#### Verifiable Memory: Learning Unified Memory Management with Local and Global Verifiers for Large Language Model Agents (VerMem) (2026-08)
- **简介**：Xiaolong Sun、Qichao Wang、Hangyu Li、Liang Chen。长 horizon 交互要求 agent 保留可复用信息、控制有界活跃 context 并能回溯早期证据，而现有方法多把长期记忆（LTM）与短期记忆分开优化，统一策略又靠 trajectory 级反馈、对单条记忆决策 credit 过弱。**VerMem** 把 LTM、活跃 context、episodic history 表示为不同状态，用单一策略以七个原子操作统一控制（增/改/软删 LTM、取入活跃 context、过滤或摘要 context、恢复 episodic 片段）。训练由 SFT 初始化后走三阶段 RL curriculum：local verifier 为可执行的记忆状态转移打分，global verifier 在任务完成后评估证据连贯性与终态一致性，二者与任务、证据召回、效率、约束信号经层次化 credit assignment 组合，verifier 仅训练期使用。五个基准、两个骨干上多数指标最佳，并在受控 online-token 预算下取得最优效率-性能前沿。
- **arXiv**：[2608.03137](https://arxiv.org/abs/2608.03137)

#### RoMeRL: Balancing Feedback Coverage and the Memory-Reward Trap in Self-Evolving Agent Memory via Reduced-Order Utility States (2026-08)
- **简介**：Yi Yang、Zhennan Chen、Yihong Zhuang、Ying Tai 等。指出自演化 LLM agent 的学习型记忆有两个耦合难题：以 trajectory 索引的 utility 随交互历史增长，把有限反馈摊薄在膨胀的状态空间上；trajectory 级 reward 又被联合分给一同被检索出的多条记忆，使无关经验获得误导性更新、落入 memory-reward trap。**RoMeRL** 用按结果极性与记忆动态因子化的定维 per-task 记忆状态表示该 utility 空间，新经验通过一组固定语义坐标（内容随时间更新或替换）纳入，把反馈集中到有界支撑上；理论上证明该 reduced-order 参数化提高每个坐标的平均反馈量，并刻画错误坐标的稳态占据率。ALFWorld 与 LifelongAgentBench 上提升任务表现，Cold-Q 比例降低 80.0%、反馈密度约提升 6.0 倍、记忆规模减少 84.4%、LLM 调用减少 21.1%。
- **arXiv**：[2608.02508](https://arxiv.org/abs/2608.02508)

#### Living-Harness Is an Interactive-Agent Evolver (Living-Harness) (2026-07)
- **简介**：来自 Yuetian Du、Qiang Zhu 等 12 人。LLM agent 可在 episode 内或重试后恢复失败，但同一执行失败会在后续任务复现，因为 post-episode 反馈很少修订那套指导未来交互的持久 harness。提出 **Living-Harness**：自演化的 agent harness，把每条完成轨迹及其评估器信号转为"有界 harness 更新"的后验证据；在领域级 **Evolution-SOP** 指导下抽取 episode 抽象与结构化更新证据，写入两类互补的过程性知识——记录触发条件/失败模式/恢复动作的 **情节记忆**，与记录状态节点/修复边/转移规则的 **状态图**；工具与基础上下文冻结，过程性修复跨演化周期累积。在源自 τ²-Bench 与 MultiWOZ-2.4 的八个交互环境上，平均 Pass@1 较最强交互基线 +10.07 / +9.91 个百分点，并支持跨骨干仅检索复用演化后的 harness 状态。
- **arXiv**：[2607.26598](https://arxiv.org/abs/2607.26598)

#### The Physics of Multi-Turn Long-Horizon Planning: From Pre-training to Post-training via Single- and Multi-Teacher On-Policy Agentic Distillation (2026-07)
- **简介**：来自中科院自动化所（Tianyi Men、Zhuoran Jin、Kang Liu、Jun Zhao）。构建统一可控的多轮环境，系统研究长程规划能力在三阶段的获得/塑造/整合。(1) 预训练获得：显式世界模型构造（CoT 状态转移建模）带来更强长程泛化，原子技能不足以支撑组合泛化而少量长程数据即有效，次优轨迹因误差在长 horizon 放大而严重损害性能；(2) **GRPO 与 OPD 后训练塑造**：用互信息区分通用规划模式与任务特定规划知识，识别后训练三区域（不必要/有效/不支持），OPD 在低质量与长程设定下有效区域比 GRPO 更宽（更新方向更一致）；(3) **多教师在线蒸馏（MOPD）整合**：收敛到跨环境共享的规划模式，兼容模式支持跨环境泛化，部分共享支持持续学习，完全冲突则造成严重干扰。
- **arXiv**：[2607.24720](https://arxiv.org/abs/2607.24720)

#### AttriMem: Attribution-Guided Process Feedback for Agent Memory Learning（AttriMem） (2026-07)
- **简介**：Qinfeng Li、Wenqi Zhang、Xuhong Zhang 等（浙大等）针对"用 RL 学习记忆构造策略"的细粒度信用分配瓶颈：现有 RL 记忆方法只用结果/模块级奖励，无法指出哪些中间记忆内容支撑了最终答案，而中间记忆决策又缺唯一 ground-truth、恰当 credit 随不确定推理轨迹变化无法预先指定。AttriMem 用**基于 token 级贡献度的归因**从全局结果奖励中派生局部过程奖励，指导 what to extract/store/update/compress/discard 的记忆策略。长程对话问答实验中超越检索型、启发式与 RL 基线，跨基准与跨答案模型泛化并稳定 RL 优化。
- **arXiv**：[2607.21106](https://arxiv.org/abs/2607.21106)

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

#### SWE-Trail: History-Guided Trajectory Collection for Coding Agent Training (2026-11)
- **简介**：作者 Ziming Li、Lu Yu、Jiatan Huang、Qing Cui、Jun Zhou、Chuxu Zhang。coding agent 的轨迹越来越多地用作后训练数据，其采集方式决定模型学到什么。对每个 issue 独立多次采样会使后续轨迹日益冗余；把历史作为建议暴露给 agent 又会造成训练-测试不匹配。作者主张历史应指导采集精力投向哪里，而由 coding agent 决定生成什么。**SWE-Trail** 是轨迹收集器：在新 rollout 中用历史找出额外采集可能有用的状态，agent 从其普通上下文提出候选下一步动作，SWE-Trail 从同一检查点短暂执行这些动作以决定继续哪条路线，历史始终保持在 agent 上下文之外。评估方式是用各采集方法产生的轨迹训练学生模型（摘要后续的具体数值未显示）。
- **arXiv**：[2611.02074](https://arxiv.org/abs/2611.02074)

#### SWE-Ladder: Learning to Localize, Reproduce, and Patch via Progressive Privileged Self-Distillation (2026-10)
- **简介**：作者 Xudong Cai、Zhengxi Lu 等。仓库级 issue 修复需要 SWE agent 定位故障代码、复现失败并打补丁；RL 只有最终补丁的单次测试结果，而 OPSD 在每一步都向 teacher 暴露完整参考补丁，二者都缺少对定位和复现阶段的针对性指导。**SWE-Ladder** 把每个训练实例已验证的修复转为分阶段证据（定位用改动文件和行，复现用 fail-to-pass 测试，补丁用 diff）：规则路由器依据工具调用把 rollout 每步分配到某阶段，冻结的 teacher 只看到匹配阶段的证据，各阶段按修复顺序蒸馏，同时 GRPO 优化整条轨迹。Qwen3-4B-Instruct-2507 在 SWE-bench Verified（R2E-Gym scaffold）上 pass@1 从 9.6% 提升至 16.4%，GRPO 为 10.4%、完整补丁 OPSD 为 12.4%；OpenHands scaffold 下由 13.0% 升至 17.6%。
- **arXiv**：[2610.33667](https://arxiv.org/abs/2610.33667)

#### CoHARL: Co-Evolving Agent Policies and Executable Harnesses with Reinforcement Learning (2026-10)
- **简介**：作者 Jianwen Chen、Peng Xia 等。LLM agent 通过可执行 harness 管理工具使用和反馈；为固定策略优化 harness 忽略了学习中的行为变化，而固定 harness 下训练策略又使生成训练经验的接口不变。**CoHARL** 是基于种群的 RL 框架，共同演化一个共享策略和一组可执行 harness：策略从整个种群的 rollout 中学习，proposer 利用执行轨迹在训练过程中修改 harness 代码。在 SWE-rebench 上达到 27.6% 的平均 resolved rate，比仅搜索 harness 提升 6.4 个百分点；在 SWE-bench Verified 和 Terminal-Bench 2.0 上也有提升，并在匹配训练预算下优于「RL 前一次性搜索 harness」和「交替更新策略与单个活跃 harness」。
- **arXiv**：[2610.36469](https://arxiv.org/abs/2610.36469)

#### Enhancing Bug-Exposing Test Generation via Step-Level Credit Assignment and Self-Distillation from Valuable Failures (SCALF) (2026-10)
- **简介**：作者 Wenjie Gan、Zhaoxi Zhang 等。修复 LLM 生成代码需要能暴露故障的测试（输入加正确的期望输出），而已有 RL 测试生成方法对所有推理步骤施加相同的 outcome reward，可能抑制有用推理、强化错误步骤，也不能有效利用「找到触发 bug 的输入但预测输出错误」的轨迹。论文提出 **SCALF**，一种过程监督 RL 方法，结合 Step-level Credit Assignment 与对有价值失败轨迹的定向学习：由过程 reward 导出步骤专属 advantage，让每个推理阶段获得自己的学习信号；并把暴露 bug 但输出预测错误的轨迹路由到 on-policy self-distillation，借助执行反馈学习其已验证的 bug 触发进展。在三个 benchmark 上，SCALF 在联合成功率、输出正确率、bug 触发率上分别最多比强 prompting 与后训练基线高 2.97、5.26、3.53 个百分点。
- **arXiv**：[2610.27694](https://arxiv.org/abs/2610.27694)

#### RIPPLE: Recovery Internalization through Process-Level Policy Learning and Effect-Aware Credit for Long-Horizon Agentic Coding (2026-10)
- **简介**：作者 Jingxuan Xu、Zihang Xu、Yuechen Luo、Ziqi Miao、Wenqiang Zhu、Weihao Li。长程 coding agent 能观察到失败并生成看似合理的 critique，却仍沿错误轨迹继续，论文称之为 Critique-to-Action Handoff Gap。**RIPPLE** 是面向恢复的训练框架：仅看前缀的 teacher 提供有证据支撑的恢复监督（不接触未来动作或终端结果），轻量蒸馏后，RL 为反思触发、诊断与纠正动作分配阶段特定的 credit，并用局部有效性信号对轨迹 advantage 加门控，使中间恢复决策不会无差别继承终端结果。在 4 个仓库级软件工程 benchmark 上，RIPPLE 相对基础 policy 平均提升 4.8 分，部署时无需辅助 critic；消融表明局部、阶段对齐的 credit 有效，恢复收益集中于前缀含足够诊断证据且仓库状态可逆的场景。
- **arXiv**：[2610.31276](https://arxiv.org/abs/2610.31276)

#### WebCoder: Scaling General Frontend Development with Artifact-Grounded Agentic Reinforcement Learning (2026-10)
- **简介**：作者 Zijun Dou、Zhen Yang、Xijun Liu、Dongzhuoran Zhou、Aohan Zeng、Jie Tang 等。真实前端开发需在不断演化的应用状态上组合复现、生成、修改与修复，学习通用策略需把异构执行反馈转为任务对齐的奖励并做长序列信用分配。**WebCoder** 的核心思想是通过各开发活动对共享可执行 artifact 的影响及其对任务需求的满足度来评估；构建 Webdev-World 可执行环境（代码编辑、应用运行、浏览器检查、按需 skill，并跨交互保持项目状态），独立 verifier 给出终端奖励，中间 artifact 进展的变化把 rollout advantage 重新分配到各开发片段，并以 context compaction 与 message-tree 样本构造支持超出单上下文的训练。在 Webdev-Bench 上，WebCoder-27B 将需求满足度从基座的 64.8 提升到 72.8，高于 Claude-Opus-4.8 (65.1) 与 GPT-5.4 (59.6)；Vision2Web 平均分从 41.1 提升到 62.4。
- **arXiv**：[2610.23104](https://arxiv.org/abs/2610.23104)

#### Lean4AgenticRL: Advancing Agentic RL with Verifiable Multi-level Process Rewards (2026-10)
- **简介**：作者 Ruida Wang、Gerald Friedland、Tong Zhang、Chris Kong。面向 SWE 与终端使用的大规模 agentic RL 多依赖高层结果奖励，对低层 toolcall 和中层 workflow 阶段奖励考虑很少。**Lean4AgenticRL** 据称是首个在统一 Lean4 形式化框架下为开放式 agentic RL 构建多层级过程奖励的框架：先构建兼顾 agent 自身能力与可控性的 problem-specific workflow 系统，再通过 workflow 形式化与多层级奖励标注在通用 harness 中验证 agent 行为，并配合 workflow-stage-normalized GRPO 训练。得到的 LAR-9B/27B 优于同规模领先模型，LAR-27B 在 TerminalBench-2.1 上达 77.3%；平均超过基座模型 9.89%，超过已知同规模第二名 5.07%。
- **arXiv**：[2610.22448](https://arxiv.org/abs/2610.22448)

#### SPROUT: Scaling Agentic Reinforcement Learning for Long-Horizon Software Engineering (2026-10)
- **简介**：作者 Mengyang Liu、Weiming Wang、Haixin Nan、Yu Wu 等。长 horizon 软件工程任务中验证成功的轨迹稀缺，一个小错误就能让整个 rollout 失效；从初始状态反复重启低效，仅重置上下文与工作目录又会丢失已安装依赖等状态变化。**SPROUT（SnaPshot-based RollOUT）** 在每个 assistant turn 后对文件系统和 agent 上下文做 checkpoint，再由 verifier 引导的回溯策略决定从哪里恢复失败尝试及如何继续，实现从中间状态恢复与分支的 rollout。相对 pass@k 最高提升 23.6%，仅用其 32%-52% 的 token；对 Qwen3.8-27B 做 RL 训练时，在三个长程编码基准上比标准 GRPO 稳定高 13.9 分，同时单步训练时间减少 39.5%。
- **arXiv**：[2610.14405](https://arxiv.org/abs/2610.14405)

#### SWE-Critix: Post-Training Generalist Verifiers for Assessing Coding-Agent Trajectories (2026-10)
- **简介**：作者 Jiajun Hu、Chun Yong Chong。coding agent 常缺乏现成的 oracle 评估器，其轨迹可达数百轮交互，现有 verifier 要么是未经专门训练的通用 LLM，要么是主要输出标量奖励的 critic。**SWE-Critix** 是训练 coding-agent 轨迹通用 verifier 的后训练流水线：给定任务与执行轨迹，verifier 预测完成标签并生成解释其判断的自然语言分析；流水线结合 SFT 与 RL，在 CodeForge-Preview 轨迹上训练一系列 30B 模型。在留出测试集上，分类准确率从基础 Qwen3-30B-A3B 推理模型的 57%，经 SFT 提升到 66%，RL 后达 78%。作者开源了模型、数据集与训练代码。
- **arXiv**：[2610.13356](https://arxiv.org/abs/2610.13356)

#### FC-SWE: Failure-Conditioned RL for Long-Horizon Software Engineering Agents (2026-10)
- **简介**：作者 Jia Liufu、Bin Hu、Linglin Jing、Terry Kong、Yuki Huang、Ashwath Aithal 等。仓库级 SWE 是长程设定，现有 GRPO 式训练对每个 issue 独立采样多条轨迹、在固定 group 内比较终端奖励，没有把失败 patch 的 verifier 反馈作为后续尝试的上下文复用，而这些反馈含有宝贵的诊断信息。**FC-SWE** 是 failure-conditioned RL 框架：patch 验证失败后，将仓库恢复到原始任务状态，以失败 patch 和 verifier 反馈为上下文生成恢复轨迹，并通过两种机制把 GRPO 适配到这类由完整多轮工具使用轨迹串成的链：trajectory-local rewards 保留每次尝试自己的 verifier 结果，避免恢复成功去奖励较早的失败 patch；还有 active-set advantage estimation（abstract 在此被截断，具体实验数字以原文为准）。
- **arXiv**：[2610.07898](https://arxiv.org/abs/2610.07898)

#### Beyond Binary Verifiable Rewards: Behavioral Self-Rewarding for SWE Agents (BSR) (2026-10)
- **简介**：作者 Yiran Wu、Andrew Zhao、Yurui Chang、Qingyun Wu、Xuan Feng。可执行测试为 SWE agent 提供可靠的 RLVR 监督，但二值奖励无法区分调试质量差异很大的轨迹。论文研究当前策略能否评判自己的轨迹并据此提升，而无需单独训练奖励模型。**Behavioral Self-Rewarding (BSR)** 复用策略，对同一任务内、verifier 结果相同的轨迹比较可观察的调试行为，把成对偏好聚合为任务内分数，再转成有界的辅助奖励，在引入奖励对比的同时保持 verifier 定义的 pass/fail 排序；它利用已收集的轨迹，无需新的环境交互。在 SWE-bench Verified 和 SWE-bench-Live 上，对评测的 Qwen3-14B 与 Qwen3-32B 配置均有一致提升。
- **arXiv**：[2610.12598](https://arxiv.org/abs/2610.12598)

#### Cross-Benchmark Transfer from RL on Agentic Coding Tasks (2026-10)
- **简介**：作者 Sushant Mehta、Logan Ritchie、Edwin Chen（ARRSI 2026 workshop）研究在专家构建的 agentic 编码任务上做 RL 能否弥合「最后一英里」缺陷，以及所学能否迁移到训练分布之外。仅用 RL，在 1,700 个任务（1,000 个由隐藏 fail-to-pass 与 pass-to-pass 测试评分的仓库任务，700 个由专家隐藏验证器评分的终端任务）上后训练 1T 参数（32B 激活）的开源 MoE 模型 Kimi K2.7 Code；reward 为通过目标检查的比例，任一 pass-to-pass 测试失败则归零。rank-32 LoRA 上一个 epoch 的 GSPO 提升了六个外部基准的 pass@1：SWE-Bench Pro 60.1→64.8，DeepSWE 31.0→43.4，Terminal-Bench 2.1 67.4→82.0，Terminal-Bench 3 1.4→12.1，Terminal-Bench 4 0.0→7.6，SWE-Marathon 5.0→25.0；合并检验显著（p<0.001），在训练数据收集后发布的三套基准上仍显著，且在训练中未使用的两种 harness 下也有提升，DeepSWE 与 Terminal-Bench 3 的中位轨迹步数缩短 24–35%。
- **arXiv**：[2610.00890](https://arxiv.org/abs/2610.00890)

#### SWE-MILE: Asynchronous Potential-Induced Milestone Credit Assignment for Long-Horizon Software Engineering Agents (2026-09)
- **简介**：作者 Chaoqun Cui、Hao Zhou、Meiqi Chen、Fandong Meng、Wenji Mao 指出，长程 SWE agent 的 RLVR 通常只有终局监督，难以区分有效动作、冗余探索与功能回归。**SWE-MILE** 无需辅助奖励模型或外部评估器，从 workflow 运行时导出细粒度过程监督：以任务相关文件暴露度和测试状态对齐度分别量化导航 potential 与验证 potential，其差分把里程碑进展与回归归因到单个动作，再通过折扣反向传播把 credit 传给之前的步骤。并引入 asynchronous shadow probing，在隔离沙箱中回放改动仓库的动作并与 agent 主交互并行做验证，基本隐藏验证延迟。过程 credit 与终局 advantage 结合；在两个具代表性的长程 SWE 任务上带来显著提升（abstract 未给出具体数值）。
- **arXiv**：[2609.32631](https://arxiv.org/abs/2609.32631)

#### FLARE: A Full-Lifecycle Dense Supervision Paradigm for Long-Horizon Coding Agents via Generative Reward Model (2026-09)
- **简介**：作者 Jingxuan Xu、Gang Wu、Yanan Wu 等。长程 SWE agent 面临稀疏 pass/fail 奖励造成的 credit assignment 危机，失败轨迹也被浪费。**FLARE** 以轻量 Generative Reward Model (GRM) 提供稠密监督：先由离线因果诊断框架 RADAR 通过因果链回溯蒸馏出实时的步级风险反馈 GRM；推理时作为 Active Scaffold 拦截高风险步骤并局部断点重执行；训练时 GRM 信号用于 SFT 的过程感知重排序和 RL 的步级稠密奖励。FLARE (N=1) 优于 Global Rollout (N=5)，token 消耗降至 1/5；训练阶段 SFT 相对提升 19.13%，RL 稳定提升 9.19%。
- **arXiv**：[2609.23808](https://arxiv.org/abs/2609.23808)

#### One to More, More to One: Category-Aware Iterative Expert Training for Software Engineering Agents (2026-09)
- **简介**：作者 Jie Zhao、Ziyu Jiang、Suhang Zheng 等。仓库级 SWE 任务类别异质，池化 agentic RL 会出现类别跷跷板（有的类别提升、有的退步）。论文提出类别感知的专家训练与策略整合框架：用 SWE Labeler 多轴标注组织训练池；同源的类别专家交替进行长程 Agentic-miniRL 与 Refresh-Repair-Expand (RRE)（刷新实例掌握度、对自身成功轨迹做 Repair SFT、重选任务）；再通过标签路由的多 teacher on-policy distillation (**MOPD**，含 ReLU 门控的奖励外推) 合并为单一学生模型，且无需外部模型提供轨迹。MOPD 在 SWE-bench Pro V2 达 86.19%、SWE-bench Multilingual 达 71.78%，较基座分别提升 4.57、3.89 个百分点。
- **arXiv**：[2609.23377](https://arxiv.org/abs/2609.23377)

#### CodeMidas: Scaling Agentic Coding RL Environments from Code Itself (2026-09)
- **简介**：作者 Bowen Ye、Lei Li、Shicheng Li 等。训练 coding agent 的 RL 需要多样任务与可靠 verifier，而现有方法依赖 issue、commit 等开发产物，任务范围受限。**CodeMidas** 是一条 agentic 流水线，仅以源码为输入，让 agent 探索已实现的功能、制定行为规格、基于原代码执行构造测试，并通过执行检查与重复 rollout 过滤，生成可执行 RL 环境，得到覆盖 3,185 个开源仓库、23 种语言、15 个技术领域的 5,545 个训练任务。用 GRPO 训练 MiMo-V2.5 后，五个基准均提升：DeepSWE +11.7%、ProgramBench +17%、Terminal-Bench v2.1 +8.5%，且 agent 表现出更多代码库探索与更多样的自我验证行为。
- **arXiv**：[2609.22068](https://arxiv.org/abs/2609.22068)

#### ScienceIDE: Turning World's Scientific Codebase into Agent Learnable Environments (2026-09)
- **简介**：作者 Hejia Geng、Zesen Huang、Haoyang Li 等。科学代码仓库蕴含数十年人类知识，但工具链碎片化、领域约定隐式、正确性标准专门，使其难以转化为可靠的学习经验（作者称为 scientific experience bottleneck）。**ScienceIDE** 在专家定义的科学案例与验收标准指导下，由 agent 把代码仓库转换为可执行环境，支持任务生成、执行与科学验证，为 SFT、RL 与评估提供统一基础。基于验证过的交互轨迹训练 PhAI-IDE-72B/9B/4B 模型族，在 held-out 科学代码修复上有提升，并在代码、推理、知识等通用 benchmark 上也有所增益，显示出从科学经验到更广泛能力的正向迁移（abstract 未给出具体数值）。
- **arXiv**：[2609.19134](https://arxiv.org/abs/2609.19134)

#### AMDKernelVault: Large-Scale Datasets and Agentic Training for AMD GPU Kernel Optimization (2026-09)
- **简介**：作者 Ji Liu、Saptarshi Majumder、Yiqing Huang 等。现有基于 LLM 的 kernel agent 多以 CUDA/NVIDIA 为中心，且依赖反复调用前沿 LLM。作者发布面向 AMD CDNA GPU 的开源 HIP 与 Triton kernel 语料及训练框架 **AMDKernelVault**：HIPKernelGen 与 TritonKernelGen 为 agent 驱动流水线，把 PyTorch 参考实现转为 HIP/Triton kernel，在 ROCm 下编译验证并在 AMD 硬件上做延迟 profiling。语料含 62,153 条经执行验证的 HIP kernel 样本、2,377 条 ROCm 库问答和 39,893 个 Triton kernel。用 SFT 加 execution-aware RL 训练 Qwen3-8B 作为语料效用演示，在固定评测预算下于 PyTorch-to-HIP（34.0% Pass@1）、TritonBench-G（33.2% Corr@3）、ROCmBench（41.94% Corr@3）上正确率最高，但编译与速度指标并非全面领先。
- **arXiv**：[2609.12471](https://arxiv.org/abs/2609.12471)

#### ExecCritic: Learn to Test, Test to Improve for Coding Agents (2026-09)
- **简介**：Microsoft Research（Leitian Tao, Baolin Peng, Hao Cheng, Jianfeng Gao 等）。执行反馈只有在测试真正刻画 issue 所要求的行为时才能引导仓库修复，而同一条 trajectory 同时写补丁与测试时二者错误会相互印证、制造虚假信心。**ExecCritic** 把 test–verify–revise 脚手架与角色专用的 RL 配方结合：脚手架把测试构造与源码修复分离，Test agent 独立生成仓库原生测试、fail-closed 的 harness 对其鉴定并冻结，Repair agent 只依执行反馈修改源码而不改测试；两个角色都以 Qwen-3.5-35B-A3B 为 backbone 分别训练，Learn to Test 让 Test agent 学会产出能区分正确与错误补丁的行为有效测试，Test to Improve 让 Repair agent 同时学直接解题与反馈引导的修订。SWE-bench Verified 上测试质量决定反馈是否有用：固定基座 Repair agent 时，基座 Test agent 的测试把 resolved 率从无测试基线 61.2% 降到 57.3%，而 GPT-5.6-sol 的测试提升到 65.3%；角色专用后训练把 Qwen Test agent 的 Base-to-Gold 成功率从 22.2% 提到 62.2%，两个后训练 agent 组合达 72.6%，在评测时不借助更强模型或 Oracle 反馈的条件下比原始无测试基线高 11.4 个点。
- **arXiv**：[2609.09133](https://arxiv.org/abs/2609.09133)

#### Correct Tests Are Not Enough: Measuring and Training Oracle Conversion in Specification-Based Test Generation (2026-09)
- **简介**：Yunhao Liang、Chengguang Gan、Ruixuan Ying、Shiwen Ni 等。从自然语言规范生成测试需要既暴露错误行为的输入、又有正确的预期输出，而二者并不同步改善——模型可以靠选更容易的输入提高正确率，也可能找到有用输入却预测不出其预期输出。作者用可执行的 reward 分解与套件级 oracle 转化率来研究这一交互：生成器一次响应联合产出五个输入–输出测试，训练时由经审计的参考程序提供正确性反馈，固定的错误程序库提供两个效用信号（潜在 input kill 与检查输出后的 effective kill），并用一个加性 GRPO 目标同时保留两类信号、推理时不需执行。在审计过的 TC-Bench 划分（506 训练 / 142 评测任务）上，三次独立训练的 Qwen3.5-9B 在第 75 步把全测试正确率从 28.59% 提到 42.54%、input kill 从 24.06% 到 25.27%、effective full kill 从 12.23% 到 14.15%；对齐的 50 步消融显示权衡：去掉 kill reward 正确率与 full kill 略升但 input kill 降到 21.60%，而固定输入的 source–oracle 交叉实验把 NoKill 与 FullKill 的主要差异归因于更难的输入选择而非同输入下更差的输出预测。
- **arXiv**：[2609.05879](https://arxiv.org/abs/2609.05879)

#### SpecCoder: Specification-Aware Code Generation with Curriculum Dual-Task Reinforcement Learning (2026-09)
- **简介**：Yixuan Li、Mingxuan Huang、Jiajing Wang、Lipeng Ma 等。复杂编程题的自然语言需求同时规定目标、输入输出格式、约束、示例与边界情形，漏掉任一条就可能产出可运行但功能错误的代码；既有 training-free 方法依赖 prompt 或 agent workflow，训练类方法则只优化最终代码，对「原始需求 → 结构化规范」这一中间映射及其与实现行为的对齐缺乏监督。**SpecCoder** 采用两阶段训练：先用 specification-guided SFT 训练模型先推导结构化规范分析、再据其生成代码；随后引入 curriculum dual-task GRPO，联合优化规范引导的生成与判别任务，以强化规范与代码行为之间的对应。在 APPS、CodeContests、xCodeEval 上 SpecCoder 一致提升独立代码生成与 agent workflow 表现，BigCodeBench-Hard、ClassEval 的补充评测以及人工评测与扰动研究进一步验证结构化规范的作用（abstract 未给出具体数值）。
- **arXiv**：[2609.06041](https://arxiv.org/abs/2609.06041)

#### Two-Stage Reinforcement Learning for Sound and Adversarial Test Generation in Code LLMs (TCS) (2026-09)
- **简介**：南洋理工大学等（Jiacheng Xu、Wentao Zhang、Zhiyi Lyu、Chaojie Wang、Bo An 等）。代码 RL 的可执行反馈主要来自测试用例，而既 sound 又有区分度的高质量测试稀缺；作者指出用模型自动生成测试天然是一个对抗式 RL 问题——测试生成器需依据 solver 当前的失败模式产出有效反例。**TCS**（Test Cases Scaling）为此提出两阶段 RL 框架：两阶段都从一个滚动更新的 policy-aligned buffer 中训练测试生成器，第一阶段生成与参考解一致（sound）的测试，第二阶段把 buffer 限制到当前失败模式并学习反例测试。在 TACO 与 LiveCodeBench 上，TCS 同时改善 pass@1 与依据生成测试进行的推理期答案选择，且学到的测试生成器还能有效地在其他 LLM 的输出之间做选择（abstract 未给出具体数值）。
- **arXiv**：[2609.03955](https://arxiv.org/abs/2609.03955)

#### Rubric-to-Code Credit Assignment for Reinforcement Learning (RCCA) (2026-08)
- **简介**：蚂蚁集团 Ling 团队（Rui Jin、Jikai Chen、Yihan Chen、Hao Zhou、Linjian Mo、Chenyi Zhuang 等）。交互式网页应用生成的质量取决于多条面向用户的功能需求，每条常只绑定事件处理器、状态更新、DOM 片段或 CSS 选择器等局部代码区域，而标准 GRPO 把这些结构化结果压成单一序列级 reward、再把 advantage 均摊到所有 token，削弱了 credit assignment。**RCCA**（Rubric-to-Code Credit Assignment）围绕显式功能 rubric 构造训练任务，用分层 reward 区分格式、源码、运行时与功能四类失败，并把 evaluator 生成的文本归因对齐到负责的代码 span 与生成 token 上，从而把 rubric 级功能反馈转为对代码的局部化优化信号。所得 **Ling-RCCA-Flash** 在 MiniAppBench 上得 41.25，比 Ling-3.0-Flash 高 32.20 分并略超 Claude Opus 4.5；ArtifactsBench 上得 76.19，比 SFT 模型高 4.48 分、比 GPT-5 高 3.64 分，在该榜官方设置下创新高。
- **arXiv**：[2608.27906](https://arxiv.org/abs/2608.27906)

#### Learning Generalizable Behaviors for Terminal Agents (River) (2026-08)
- **简介**：Salesforce AI Research 与 CMU（Yihang Yao, Bo Pang, Xuan Phi Nguyen, Ding Zhao, Shafiq Joty, Semih Yavuz）。终端 agent 的 RL 训练缺少真实用户交互数据，只能依赖合成环境，但后者存在域差与保真度不足；已有工作主要扩大合成环境的数量与多样性，对 reward 信号质量与泛化机制探索不足。作者提出 Agentic Compositional Generalization 假设：RL 主要塑造的是组合与路由预训练/SFT 已获得的低层技能的高层决策行为，而非从零教授领域技能，因此决定「哪些行为被强化」的 verifier 质量比单纯堆环境数量与多样性更重要。据此提出 **River** 训练配方：过滤低质量环境，并在结果 reward 之外加入过程级行为正则。所得 RL agent 在四个终端 agent 基准上取得已评测开源 8B RL 模型中的最佳表现，且能跨模型族、规模、agent harness 与 RL 目标迁移；仅用不到 30% 的 TMax 训练环境，就把 2B–27B 模型在 Terminal-Bench-Lite 与 Terminal-Bench-v2.1 上的 RL 增益平均分别提高 106% 与 30%。
- **arXiv**：[2608.22631](https://arxiv.org/abs/2608.22631)

#### LEGO-RL: Harness-Native Reinforcement Learning for Coding Agents (2026-08)
- **简介**：Yiming Du, Yuxin Jiang, Tao Yuan, Haoli Bai 等（Huawei Noah's Ark Lab 与 CUHK 等）。编码 agent 的 RL 越来越依赖长时运行的 agent harness 来管理工具集成、仓库上下文与执行反馈，但这些 harness 的原生执行环境与 policy gradient 训练本质错配：环境崩溃与 reward hacking 污染 outcome 信号，train-inference 差异又让 rollout 行为与 policy 更新脱钩。**LEGO-RL** 在不改动 harness 内部控制流的前提下把原生编码 agent harness 接到可扩展的 policy gradient 优化上，由三根支柱构成：（1）忠实优化——进程内 LLM proxy 捕获原始生成流以实现 token 级对齐，并在 trainer 侧稳健重算 log-probability，即使 harness 侧做过 compaction 或重序列化；（2）可靠执行——带镜像缓存与分阶段防御的可扩展沙箱编排以抑制 reward hacking；（3）可观测训练——自动化验证与监控的插件，配合用于细粒度 trajectory 诊断的 Live UI。作者用 GSPO 在三种原生编码 agent harness 上训练稀疏 MoE 模型 Qwen3.5-35B-A3B，SWE-bench Verified 上 OpenHands SDK 从 64.0% 提升到 70.4%、Claude Code 从 62.4% 到 68.2%、OpenCode 从 57.2% 到 66.6%，同时把 rollout 与训练的概率相关性维持在 0.99 以上。
- **arXiv**：[2608.17393](https://arxiv.org/abs/2608.17393)

#### DiDPO: Diff-in-Diff Policy Optimization for Coding Agent Training (2026-08)
- **简介**：Xucong Wang、Zhe Zhao、Liheng Yu、Pengkun Wang 等。指出 coding agent 的 RLVR 面临更细粒度的 credit assignment 难题：单步 coding action 会把多处改动同时打包进同一代码版本，独立改动的贡献不可分辨，而现有方法只用 outcome reward 或 step 级 reward，无法深入 code diff 内部。**DiDPO** 是 critic-free 方法，直接由 code diff 结构构造细粒度 credit 单元：把多轮交互组织为若干 thought-action 步，在采样的多条 trajectory 间发现 code diff，用 groupability score 把整块 diff 切成 sub-diff 并聚合高度相似者选出 anchor，anchor 构成 advantage 组后把 diff 级 advantage 投影回单个 response token。长 horizon 编码与推理基准上明显优于强 agentic RL 基线，Qwen2.5-7B-Coder 上超过同类方法 10% 以上；同时开源 agentic RL 代码库 verl-code。
- **arXiv**：[2608.07147](https://arxiv.org/abs/2608.07147)

#### CodeGrep: An RL-Trained Retrieval Agent for LLM Coding Agents (2026-08)
- **简介**：Wuya Chen、Yihao Yang、Yang Cao、Yue Lin。指出现代 LLM coding agent 的共同低效：token 预算大量花在「找到该改哪个文件」而非改它——SWE-Bench Verified 上 30B OpenHands agent 每解决一个 issue 平均 23 轮、631K token，许多调用消耗在仓库探索的 grep、glob、view_file 上。**CodeGrep** 是用 GRPO 端到端训练的 14B 检索 agent，发起多轮并行 grep/glob/read 调用并把候选文件交给冻结的下游 coding agent。全部 500 个实例上解决率 27.0%（无检索基线 25.8%），已解决实例轮数减少 15%、token 减少 19%。跨检索器还发现下游收益存在精度阈值：precision 0.375 的 BM25 反而拖累，0.445 的 Jina 中性，0.677 的 CodeGrep 才越过阈值；效率信号施加在 advantage 层而非 reward 层以减少 KL drift。
- **arXiv**：[2608.05886](https://arxiv.org/abs/2608.05886)

#### Graph Is the Verifier: Agentic Reinforcement Learning for Interprocedural Vulnerability Detection (VulAgentRL) (2026-07)
- **简介**：来自新加坡管理大学 / David Lo 组（Yikun Li、Yintong Huo、David Lo 等 12 人）。真实漏洞常跨多个函数，但多数学习式检测器孤立分类单函数——采样真实 CVE 发现 71.7% 的漏洞函数需函数外证据才能正确分类。Agentic RL 可让模型自行收集证据，但仅以最终判定定义的奖励可"不做调查也能拿到"。提出 **VulAgentRL**：建立在代码属性图（CPG）上的过程间漏洞检测 agentic RL 框架——推理时策略向 CPG 查询调用者/被调用者/数据流等，训练时同一图 **验证策略所引用的证据**；因每个 CPG 节点带持久整数 ID，验证是精确比较而非文本匹配，故奖励只嘉奖"有证据支撑"的判定。并用蒸馏 teacher 调查做 warm start（RL 无法习得从未采样过的工具使用行为）。在防泄漏的仓库级划分下，以更少工具调用于严格 pairwise-correct 指标上超越含前沿模型的 SOTA，并在 OOD 语料与类别不平衡下保持优势。
- **arXiv**：[2607.26656](https://arxiv.org/abs/2607.26656)

#### Sample-Efficient Learning from Agent Experience（Experience Distillation） (2026-07)
- **简介**：Chenhui Gou、Haoqin Tu、Hamid Rezatofighi 等（Monash/UCSC 等）针对"真实环境交互昂贵"提出 **Experience Distillation**：先让智能体用 in-context learning 从自身试错交互史中高效学习，再用 context distillation 把交互史内化进权重，全程**不需超出已采集经验的额外环境交互**。在 749 个精选软件工程任务与 6 个文字冒险游戏上，保留至少 64.8% 的 ICL 收益（直接 SFT 仅 3.8%），并以至少 9.6× 更少的环境样本匹配经典 RL 基线——为长程/高成本 agentic 学习提供 RL 之外的样本高效替代路径。
- **arXiv**：[2607.21051](https://arxiv.org/abs/2607.21051)

#### Multi-turn RL with Structural and Performance Aware Rewards for CUDA Kernel Generation（CudaPerf） (2026-07)
- **简介**：Quazi Ishtiaque Mahmud、Nesreen K. Ahmed、Ali Jannesari（Iowa State/Cisco 等）提出的反思式多轮 RL 框架 CudaPerf，用于优化 CUDA kernel 生成。除可验证执行奖励外，引入源自并行化特征（内存合并、occupancy、算术强度、同步模式）的**结构代码感知奖励**；两阶段：离线成对排序模块经对比学习区分强弱候选，在线 RL 阶段以统一奖励联合优化正确性/性能/结构效率，并用执行反馈做迭代精化。发布 2.9k C→CUDA、1k PyTorch→CUDA 数据集；相较 Qwen-3-32B 与 CUDA Agent 分别取得最高 5× / 3.32× 加速、17% / 7% 正确率提升。
- **arXiv**：[2607.20908](https://arxiv.org/abs/2607.20908)

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

#### Learning Where to Look: Observation Policy Optimization for Thinking with Images (OPO) (2026-11)
- **简介**：论文被 NeurIPS 2026 接收，作者 Junfeng Wang、Jiawei Liu、Yongchao Xu 等。VLM 在 thinking with images（缩放到高分辨率图像的局部区域）中，许多失败源于错误的观察动作，且 bounding-box token 的生成熵显著高于普通语言 token；基于最终答案奖励的 RL 只提供轨迹级反馈，难以给观察动作精确信用。**OPO**（Observation Policy Optimization）包含两个模块：Uncertainty-Guided Observation Branching (UOB) 把每次图像放大工具调用视为观察事件，在 rollout 中于高不确定、低冗余的事件处选择性分支，高效探索替代的裁剪决策及其视觉证据；Evidence-Localized Advantage Attribution (ELAA) 用局部证据充分性奖励比较兄弟分支，仅把信用分配给对应的观察动作 token，从而把观察动作与其余推理解耦优化（abstract 摘录未给出具体数值）。
- **arXiv**：[2611.04943](https://arxiv.org/abs/2611.04943)

#### ThinkWise: Learning When to Think in Large Multimodal Reasoning Models via Visual and Difficulty Perception (2026-11)
- **简介**：作者 Wenkang Zhou、Chen Yang、Jiaheng Peng 等。大型多模态推理模型（LMRM）的 RLVR 往往忽略视觉感知不确定性的作用，导致「推理分配错位」：在视觉 grounding 不确定时过度分配深度推理，产生不必要的长推理轨迹并损害性能。**ThinkWise** 是感知感知的 RLVR 框架，联合建模任务难度与视觉感知状态来调节推理深度：实例自适应的视觉感知建模机制对感知不确定性做上下文感知校准，并设计结构化奖励——answer reward 监督正确性，感知与难度条件化的 length reward 自适应调节推理深度，perception reward 鼓励视觉 grounding 更强的轨迹。在多个多模态推理 benchmark 上的实验验证了其有效性（abstract 摘录未给出具体数值）。
- **arXiv**：[2611.18685](https://arxiv.org/abs/2611.18685)

#### When Fine-Grained Credit Assignment Helps and Hurts: Multimodal Agentic Reinforcement Learning through Video Editing (2026-10)
- **简介**：作者 Lei Zhao、Zongheng Cao、Xinyu Hu。论文以视频剪辑为 testbed（融合多模态感知、长程规划、工具使用与延迟结果反馈），构造 Content Selection、Narrative Sequencing、Rough Cut、Fine Cut 四类电影预告片剪辑任务，研究细粒度 credit assignment 何时有益。衰减式 process reward 在平均超 60 轮的 Fine Cut 上相对 outcome-only GRPO 带来 45.4% 的相对提升，而在 30-40 轮的较短任务上至多 10.4%。固定权重的 GRPO+PRM 训练后期会出现过度推理和冗余工具调用（过程奖励上升、任务完成率下降，即 reward hacking），将过程奖励权重衰减至零可缓解并保留早期收益。此外 on-policy self-distillation（OPSD）尽管有稠密 token 级监督，收益很小甚至明显退化，且对 privileged information 的选择高度敏感。
- **arXiv**：[2610.35497](https://arxiv.org/abs/2610.35497)

#### MRC-MMAgent: Reinforcement Learning for Multi-Role Collaboration in Multimodal Tool-Using Agents (2026-10)
- **简介**：作者 Shengqin Wang、Jie Jin 等。现有多模态工具使用 agent 的 RL 多训练单角色 agent、仅用终止奖励，忽略人类检视检索证据、重构查询、修订结论的自我纠正过程。**MRC-MMAgent** 是多角色协作 RL 框架，把轨迹评估与续写直接纳入可执行 agent 循环：构建「求解-检视-续写」的多角色循环，使 agent 通过工具获取证据、评估之前的轨迹，并在当前答案证据不足时继续求解；并提出 role-coupled 学习目标，结合任务成功、交互有效性与成功纠正，对所有角色生成的决策做全轨迹策略优化。仅用原始基线训练集、不构造额外数据，在多个 benchmark 上稳定优于对应基础模型，取得领先或极具竞争力的表现（abstract 未给出具体数值）。
- **arXiv**：[2610.39324](https://arxiv.org/abs/2610.39324)

#### Replan for Error Recovery: Training Embodied VLMs through Verifiable Interaction with Action-Grouped Turn-GSPO (AGT-GSPO) (2026-10)
- **简介**：作者 Zhuoran Wang、Keqing Fan、Yang Li、Jinbo Su、Ke Wang、Sijia Luo。具身 VLM 执行出错后需重规划恢复，现有方法依赖推理时机制，或从失败样例与参考修正中学习，而非已验证的动作结果。论文提出 ERR-Env（提供任务进展与执行错误反馈的环境）和 ERR-Bench（含正常、中间与错误状态的 held-out 任务），并提出 **AGT-GSPO**：结合针对任务进展与错误状态的 turn 级奖励与 action-grouped advantage 估计的在线 RL 方法，训练 2B 的紧凑具身 VLM（ERR-VLM）。在 ERR-Bench 上发现各 VLM 的任务进展与执行可靠性存在错配；AGT-GSPO 使 ERR-VLM 的无错完成 Strict Success Rate 提升 10.25 个百分点、Recovery Rate 提升 22.95 个百分点，消融与零样本迁移进一步验证有效性。
- **arXiv**：[2610.27003](https://arxiv.org/abs/2610.27003)

#### Long-MDR: Long-Context Reinforcement Learning for Multimodal Deep-Research Agents (2026-10)
- **简介**：作者 On Tai Tang、Huanyao Zhang、Jiepeng Zhou、Wentao Zhang。多模态研究 agent 需要在长期研究历史上推理，但在线 RL 此前多局限于较短上下文与交互轮次。论文把在线 RL 推到 128k 上下文和 75+ 工具交互轮，据称是首个在 128k 上下文、首个以 75 工具轮视野训练的在线多模态 deep-research RL 研究。该规模暴露传统 RL 的问题：早期弱策略浪费大交互预算、rollout 昂贵而奖励几乎不涨，后期策略熵在性能饱和前就已坍缩。**Long-MDR** 是三组件训练配方：On-Policy Distillation Warmup、Progressive Horizon Expansion、Entropy-Triggered Rescue，共同提升长程 RL 的学习效率与稳定性。（abstract 未给出具体数值）
- **arXiv**：[2610.05195](https://arxiv.org/abs/2610.05195)

#### OneSearch-VL: Unified Multimodal Deep Research Agent for Image and Video (2026-10)
- **简介**：作者 Hongyu Li、Manyuan Zhang、Kaituo Feng、Shu Chen、Dian Zheng、Hao Li。单图、多图和视频 deep research 需要不同视觉操作但共享视觉定位、外部检索和事实组合的流程，关键难点是保持局部视觉锚点、实体关系、有来源支撑的事实与产生答案的操作之间的依赖。**OneSearch-VL** 以 Visually Grounded Evidence Graph (VGEG) 为核心，用作数据构造、过程监督与操作级评测的共同参照；基于 VGEG 的数据引擎构造并验证多图/视频问题、过滤专家轨迹，得到 SFT 与 RL 数据集 OneSearch-VL-SFT-110K 和 OneSearch-VL-RL-10K，并由 VGEG 标注派生 Evidence-aware Visual-Grounded Rubric reward (EVGR) 在 RL 中监督证据可追溯性与视觉定位。OneSearch-VL-8B 在两个新 benchmark 上比带工具的 Qwen3-VL-8B 分别提升 20.2 和 17.6 个百分点。
- **arXiv**：[2610.12419](https://arxiv.org/abs/2610.12419)

#### HaPRL: Human-Anchored Process Reinforcement Learning for Visual Search Agent (2026-09)
- **简介**：作者 Zhangquan Chen、Yaoxin Niu、Xiang An 等指出，多轮视觉搜索 agent 通过逐步决定看哪里来回答高分辨率图像问题，但其 RL 只奖励最终答案，搜索过程缺少监督，常出现推理有误而答案碰巧正确的「错误路径」，导致沿错误路径放大训练。提出 **HaPRL**，首个用人类搜索行为强化搜索过程的框架：先构建标注平台，收集 1K+ 含细粒度行为信号的人工标注数据；训练时由精心设计的 judge 以任务自适应权重给每条 rollout 打分，并以人类标注者在同一图像上实际搜索过程的蒸馏轨迹为锚点。实验表明 HaPRL 稳定优于结果型 RL，且早期过程监督使后续结果型 scaling 的提升达 6.7 倍。代码已开源。
- **arXiv**：[2609.37190](https://arxiv.org/abs/2609.37190)

#### When Should a VLM Look? Paying Only for Visual Calls That Were Needed and Used (CounterCredit) (2026-09)
- **简介**：作者 Kunyu Peng、Junming Liu、Ruiqi He 等。会裁剪、放大的视觉 agent 通常因工具调用成功而得到奖励，但成功调用并不说明模型需要看、或用了得到的像素：冷启动 checkpoint 上仅 10%–12% 的视觉调用既被需要又被使用，已发布 agent 在个别基准上有 36%–87% 的调用是多余的。**CounterCredit** 在每次返回图像的调用的实际调用前状态上，用策略自身的金标准答案得分问两个问题：决策价值（视觉分支 vs 立即回答）和证据价值（返回的裁剪 vs 替换进同一调用的随机同尺寸 patch）；两者都验证的调用获得返现，其余已执行调用付租金，价格有界使正确轨迹始终优于错误轨迹，并用双通道 GRPO advantage。相比仅结果奖励的 GRPO，在 V* 达 89.5%、HR-Bench-4K 80.2%、HR-Bench-8K 76.4%，高出 6.3–9.4 点，多余调用率降至 31%–36%。
- **arXiv**：[2609.22910](https://arxiv.org/abs/2609.22910)

#### Region-Level Policy Optimization for Fine-grained MLLM Perception (Vision-RL2) (2026-09)
- **简介**：作者 Yuheng Shi、Xiaohuan Pei、Minjing Dong、Chang Xu。MLLM 的细粒度感知常靠提高分辨率，但视觉 token 增加了编码与 prefilling 成本。作者发现定位感兴趣区域（RoI）比识别其内容能容忍约 3–4 倍的 token 压缩，因此主张在粗视图中定位、把分辨率集中到选中的证据上。由于 proposal network 通过离散区域选择影响答案，其对答案的忠实度无法直接监督，作者提出 **Vision-RL2**：把连贯区域作为 action，由冻结的 MLLM reader 依据移除该区域后答案似然的变化为其打分，用互补的 subtractive/additive 目标抑制干扰提议、补回遗漏证据，只更新预测器，无需区域标注、响应采样或推理轨迹。在六个细粒度 benchmark、四个 MLLM backbone 上，各 token 预算下均超过基座，并以约 4 倍更少的视觉 token 超过其最大预算下的精度。
- **arXiv**：[2609.19745](https://arxiv.org/abs/2609.19745)

#### Navigating Sparse Evidence: Agentic Visual RAG via Explicit Context Selection and Consolidation (SCoRE) (2026-09)
- **简介**：作者 Yucheng Shen、Lingyong Yan、Jiulong Wu、Shuaiqiang Wang、Jianmin Wu、Dawei Yin。视觉 RAG 面临证据稀疏、且现有 agent 常基于原始探索轨迹或压缩的文本记忆生成答案、易受探索噪声影响的问题。**SCoRE**（Selection and Consolidation for Robust Evidence）是统一的 agent 循环：探索阶段只在文本 ledger 中保留与 query 相关的观察及其来源指针，保持视觉上下文有界；终止时重新载入被引用的原图并将视觉证据整合排成逻辑序列，使最终推理与探索试错解耦，并通过 claim-to-image 索引链接保证视觉 grounding。训练上结合过滤后的冷启动轨迹蒸馏与 evidence-aware RL，其 reward 鼓励证据覆盖、整合紧凑性与答案正确性（abstract 未给出具体数值）。
- **arXiv**：[2609.15800](https://arxiv.org/abs/2609.15800)

#### VideoScout: Learning Agentic Active Exploration with Adaptive Reasoning Pacing for Long Video Understanding (2026-09)
- **简介**：作者 Weixin Xu、Zhenyu Yang、Bing Wang 等。长视频受限于视觉上下文窗口，均匀采样或由粗到细的缩放难以定位稀疏而关键的证据。作者把长视频理解形式化为 **Sequential Evidence Acquisition（SEA）**：agent 沿时间轴逐轮观看，每轮决定观看速度、保留什么证据、何时回看不确定片段、何时停止作答。**VideoScout** 通过自适应推理节奏动态控制观看速度，在有界视觉窗口内遍历更多内容。训练上构建含 10K 条答案验证轨迹、超过 66K 探索轮次的 VideoScout-66K，两阶段：冷启动 SFT 学习每轮输出格式，再用 DAPO 做轨迹级 RL，reward 综合答案准确率、格式合规与 agent 观看进度和 teacher 作答时点的时间 IoU 对齐。7B 模型在长视频理解与推理 benchmark 上较已有同规模训练的 agentic 模型表现强劲。
- **arXiv**：[2609.15606](https://arxiv.org/abs/2609.15606)

#### UniCAR-RL: Seeing Better before Thinking Deeper in Visual Mathematics (2026-09)
- **简介**：作者 Yuzhe Li、Hao Yan、Hao Wang 等（EMNLP 2026 Findings）。MLLM 解复杂视觉数学题时常因感知不足，初始视觉幻觉引发级联推理错误；端到端 RL 的稀疏 reward 无法区分感知幻觉与逻辑失误，而用感知增强的 CoT 数据微调成本高且仍有幻觉。**UniCAR-RL** 是无需标注的 RL 框架，在训练中显式解耦感知与推理的优化，含三个协同分支：Caption-RL 通过 verifier 引导的推理验证来优化感知；Reasoning-RL 基于金标图像描述进行逻辑推理以阻断级联错误；QA-RL 保留原生端到端对齐以保证问答稳健。仅用原始短答案数据即可显著提升 MLLM 的数学与视觉推理，并在不同架构与规模上泛化良好（abstract 未给出具体数值）。
- **arXiv**：[2609.13849](https://arxiv.org/abs/2609.13849)

#### Drive by Hindsight and Foresight: Tool-Grounded Synergistic Reasoning over Hierarchical Memory for Autonomous Driving (2026-09)
- **简介**：Baojie Chen、Zijun Jia、Jing Zhong。自动驾驶 VLM 仍受幻觉、时空感知弱与泛化差困扰，而 CoT、RAG 或静态注入工具输出的做法只丰富了 context，模型既不主动感知场景也不在作答后积累经验。作者提出把分层记忆与主动工具调用紧耦合进闭环推理：场景级短期记忆维护动态场景状态，演化的长期记忆检索可复用经验与工具策略；推理时模型据此自适应调用工具精炼推理，离线再把可复用经验固化进长期记忆池；训练用多步教师 rollout 构造的已验证 memory-tool trajectory 做 SFT 加 GRPO 两阶段。7B 模型在 DriveLMM-o1 上取得 80.03 综合推理分与 79.09% MCQ 准确率，比最强基线高 7.74 个 MCQ 点；短期记忆使 STSBench 准确率提升 24.2 个点，离线长期记忆固化在全参数冻结下再带来 3.57 个 MCQ 点增益。
- **arXiv**：[2609.08217](https://arxiv.org/abs/2609.08217)

#### Eliciting Self-Verification in Multimodal Reasoning Agents with Reinforcement Learning (SVRL) (2026-09)
- **简介**：Vishwas Sathish、Viresh Ranjan、Xinliang Zhu、Arnab Dhua、Douglas Gray。多模态 agent 的可靠工具使用仍困难：模型要在同时理解文本与图像的前提下整合噪声检索证据，而监督往往只是稀疏的 outcome 级信号、没有显式核验信号。**SVRL**（Self-Verification via Reinforcement Learning）是纯 RL 的微调框架，训练多模态 agent 在自身推理轨迹内核验并过滤检索证据，从而减少推理时对外部 verifier 的依赖；此外引入抑制无谓工具调用的 search-aware penalty 与鼓励多样、规范查询的 query-diversity reward，为「何时搜、搜什么」提供细粒度反馈。仅用 5,000 条视觉问答样本对 Qwen-2.5-VL-7B 做 SVRL 微调，即在多个基准上一致提升多跳 VQA 泛化与工具效率，以显著更低的训练与推理成本缩小紧凑 agent 与更大专有模型的差距（abstract 未给出具体数值）。
- **arXiv**：[2609.08025](https://arxiv.org/abs/2609.08025)

#### Thinking with Cameras: Active Visual Reasoning via Dynamic Viewpoint Control for Surveillance Video Understanding (CamVLM) (2026-09)
- **简介**：Xiao Zhang、Wang Zeng、Sheng Jin、Shichao Kan 等。LVLM 在通用视频理解上进展显著，但监控场景缺少大规模领域数据、且固定视角的被动观察会在目标远、小、被遮挡或移出画面时错失关键证据。**CamVLM** 提出「用相机思考」：让 LVLM 通过动态视角控制主动获取视觉证据而非被动分析固定视频流。作者先构建大规模监控视频理解数据集 CCTV-Anomaly（14,459 个视频、10 类异常，含详细字幕与事件标注），再构建以物体为中心的视角 trajectory 数据集 CamTrack-53K 用于学习相机动作，并把视角控制形式化为主动视觉感知问题，提出基于 RL 的视角 policy 优化框架，把相机控制建模为序贯决策以学到超越监督 trajectory 模仿的长 horizon 观察策略。实验表明 CamVLM 在被动观察与动态视角两种设置下均达到 SOTA（abstract 未给出具体数值）。
- **arXiv**：[2609.06475](https://arxiv.org/abs/2609.06475)

#### LiteSearch-VL: Small Multimodal Search Agents via Trajectory Distillation and Synthetic Step-DPO (2026-08)
- **简介**：Saeed Khaki、Nima Safaei、Kamal Ginotra。多模态搜索 agent 需要交织图像理解、网页检索、工具使用与证据综合，现有强系统要么是 GPT-5、Gemini 这类闭源前沿模型，要么是用大量 agentic 数据与 RL 训练的大型开源 VLM；作者追问：在单机预算下把已发布的 agent trajectory 蒸馏进小得多的 backbone，究竟迁移了什么？**LiteSearch-VL** 给出低算力配方，仅用已发布的 OpenSearch-VL trajectory、LoRA adapter，以及针对五类局部失败模式（过早作答、错工具、弱 query、重复 query、忽略图像）由 GPT-5 生成 hard negative 的合成 step-level 偏好 DPO，训练 Qwen3-VL-2B/4B。SimpleVQA、FVQA、LiveVQA、VDR-Bench-testmini 上 12,400 条 GPT-5 判分 rollout 显示主要效应是行为迁移而非均匀准确率提升：全 trajectory SFT 迁移了「agent 契约」，把 2B 从几乎从不给出可用答案（1,240 条中 1,237 条 no_answer）提升到 28.4% macro Pass@1，追平甚至略超开箱的 4B base（25.6%），而合成偏好学习与紧凑工具蒸馏只是精修（4B 最佳配置 30.8%）；VDR step 预算消融进一步表明增加搜索轮数只是把弃答转成 wrong_entity 错误，说明小模型多模态 agent 的下一个瓶颈是答案验证而非搜索深度。
- **arXiv**：[2608.29357](https://arxiv.org/abs/2608.29357)

#### MCite-RL: Towards Reliable Multimodal RAG via Citation-enhanced Agentic Reinforcement Learning (2026-08)
- **简介**：北京大学（Suifeng Zhao, Zida Liu, Jun Gao, Sujian Li 等）。带视觉引用的多模态 RAG 对 MLLM 的可追溯与可验证性至关重要，但现有 RAG 与 SFT 方法跨模态推理不够稳健，导致视觉引用不精确、或引用与生成答案相互脱耦。**MCite-RL** 引入 Agentic Refinement 模块处理视觉引用，通过迭代检索、推理与递归裁剪逐步收窄搜索空间，把引用从静态一步变成动态、证据驱动的推理过程；同时用 Citation-enhanced Reward 在 RL 中融合过程级与结果级反馈，联合优化答案准确性与来源可追溯性。Wiki-VISA、FinRAGBench-V、MMLongBench-Doc 等基准上有效实现引用精度与答案质量的联合提升（abstract 未给出具体数值）。
- **arXiv**：[2608.21808](https://arxiv.org/abs/2608.21808)

#### Think with Structured Grounding: Perceptual Reinforcement Learning for Chart and Visual-Tabular Understanding (TwSG) (2026-08)
- **简介**：MBZUAI 等（Changjiang Jiang, Qiannian Zhao, Preslav Nakov, Zhuohan Xie 等）。能「用图像思考」的 MLLM 往往依赖外部工具做细粒度感知，既带来推理延迟，也无法真正弥合图表、视觉表格这类文本密集且结构关系严格的图像中的空间-结构鸿沟。**TwSG** 把复杂图像的工具使用能力内化进模型，让多步推理与微裁剪的收益压缩到单次高效前向：先用一个 MLLM 在 ground-truth 答案引导下定位关键区域，再让 teacher 模型生成高质量 VQA 数据，把这些细粒度、区域级监督信号蒸馏回全图表示。训练分两阶段——带聚焦区域描述的多轮数据冷启动 SFT，以及由新的过程 reward 机制 TL-GRPO 驱动的强化微调（RFT）以鼓励策略性推理。多种 MLLM 架构上推理延迟下降，同时准确率与鲁棒性大幅提升，并使模型获得原生的细粒度区域描述与灵活推理能力（abstract 未给出具体数值）。
- **arXiv**：[2608.22429](https://arxiv.org/abs/2608.22429)

#### MMDynOpt-Agent: Dynamic Optimization for Multimodal Large Language Model Reasoning via Reinforcement Learning (2026-08)
- **简介**：Wenjin Liu, Haoran Luo, Fayuan Ke, Carl Yang 等。多模态大模型（MLLM）在视觉理解与复杂推理上潜力强，但现有方法难以把多模态输入中的视觉线索与问题语义高效转化为有效的推理条件，限制了推理表现。**MMDynOpt-Agent** 把多模态推理的动态优化建模为马尔可夫决策过程并用端到端 RL 求解：一个轻量多模态 agent 充当决策 policy，把目标 MLLM 当作环境，通过多轮动态优化 prompt 自适应地引导其推理；为压低推理成本，reward 由格式合规、答案正确性与预算意识三部分组合，同时保证推理准确率与效率。该 agent 具备可迁移性与泛化性——用一个目标 MLLM 训练后可在推理时迁移到其他 MLLM。十五个公开数据集上表现优于 baseline（abstract 未给出具体数值）。
- **arXiv**：[2608.14026](https://arxiv.org/abs/2608.14026)

#### Teaching MLLMs to Say No: Generalized Referring Expression Comprehension via Refusal Calibrated GRPO (RC-GRPO) (2026-08)
- **简介**：Xuzheng Yang、Jun Ling、Tao Huang、Peng Wang 等。面向 Generalized Referring Expression Comprehension（GREC）：文本指称的对象存在时要定位（正样本），不存在时要拒绝输出（负样本）。多模态大模型擅长定位已存在对象，但训练缺乏负样本，面对不存在的指称常给出幻觉 bounding box；而 SFT 与常规 RL 虽能增强拒答，往往损害正样本定位精度、削弱核心能力。提出 **RC-GRPO**：在 rollout 中强制产出 "None" 输出，使负样本也能获得有效 advantage 估计，同时对正样本施加惩罚以防过度拒答，在准确率与可靠性间取得平衡；第二阶段推理强化进一步巩固因果理解与可解释性。三个 GREC 基准上在保持强拒答能力的同时取得更优定位准确率。
- **arXiv**：[2608.04698](https://arxiv.org/abs/2608.04698)

#### EmoAgent-R1: Towards Multimodal Emotion Understanding with Reinforcement Learning-based Dynamic Agent Specialization（EmoAgent-R1） (2026-07)
- **简介**：Lihuang Fang、Yuchen Zou、Jinghui Qin 等（广东工业大学等）提出的 RL 动态智能体专化框架，用于多模态情感理解（MER）。先用合成 answer-conditioned CoT 与 agent-routing 数据冷启动，赋予 MLLM 初步识别/推理/路由能力，再以 RL 在"agent selection＋agent specialization"两步 agentic workflow 中感知情感；核心提出 **Progressive GRPO（P-GRPO）**，把组相对优势与 PMI 启发的渐进式 token 级调制结合，将稀疏奖励转成细粒度信号以缓解 GRPO 的粗粒度均匀信用分配。MER 基准上情感推理更强且优化更稳。
- **arXiv**：[2607.21013](https://arxiv.org/abs/2607.21013)

#### OmniReasoner: Thinking with Long Audio-Video via Native Tool Use（OmniReasoner） (2026-07)
- **简介**：Yu Chen、Caorui Li、Yidong Wang 等提出的工具使用后训练框架，让全模态 LLM 在长音视频推理中经 **SFT＋RL** 学会"是否/在何处调用 zoom-in 工具"再作答：先构造低成本全局预览，需要时以请求的时间区间调用工具做高保真视听细查。为解决前后采样粒度不一致，提出 TimeAnchor 保持工具时间参数在不同粒度间有效且往返一致；并用 Temporal Augmented Data Engine 通过视频编辑/合成免人工区间标注地合成工具使用轨迹。全模态与视频基准上同时提升答案准确率与时间定位，且把高保真计算集中在信息区。
- **arXiv**：[2607.19339](https://arxiv.org/abs/2607.19339)

#### Sparse Evidence Can Suffice: Agentic Evidence Seeking for Multimodal Video Misinformation Detection（SIEVE） (2026-07)
- **简介**：Haochen Zhao、Yongxiu Xu、Gaopeng Gou 等（中科院信工所等）提出 SIEVE，将多模态视频虚假信息检测从"整段视频单次判定"改为"取证与验证解耦"：证据寻找智能体主动探索多模态证据、构造紧凑证据包，再由 verifier 判定真伪。智能体以监督取证轨迹＋**证据感知的 RL 目标**训练，鼓励获取信息量大的证据、抑制无谓或无效交互；多个视频虚假信息基准上一致超越基线，并以紧凑证据包支持可靠验证、提供可检视的显式证据轨迹。
- **arXiv**：[2607.18080](https://arxiv.org/abs/2607.18080)

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

#### The Web Doesn't Sit Still: Adversarial Self-Evolving Attacks on Search Agents (2026-11)
- **简介**：论文被 NeurIPS 2026 接收，作者 Geyuan Wang、Shunyu Liu、Siyuan Liang、Chenyang Lyu、Dacheng Tao。搜索 agent 因需访问网络而易受恶意信息注入、隐蔽上下文操纵等攻击，而现有攻击依赖静态、人工编写的模板，无法模拟真实威胁的动态性。作者提出对抗自进化框架，在攻击者策略演化与防御者自适应缓解之间做双层优化，以暴露搜索 agent 的漏洞；其中 **CREO**（Contrastive Rollout Evolutionary Optimization）借助对比演化信号驱动攻击策略种群的定向演化，并构建受控沙盒与细粒度指标来量化内部漏洞。在多个多跳问答 benchmark 与前沿 LLM 上，攻击效果优于静态基线，凸显亟需更鲁棒的防御框架（abstract 未给出具体数值）。
- **arXiv**：[2611.03754](https://arxiv.org/abs/2611.03754)

#### SecPO: Principled Adversarial Training for Prompt Injection Security (2026-10)
- **简介**：作者 Sizhe Chen、Yibo Peng、Jaewon Chang、Chawin Sitawarin、David Wagner。prompt injection 是 AI agent 的头号威胁，现有防御最强的 Meta-SecAlign 用 DPO 训练，在自适应注入下 ASR 仍接近 100%；原因之一是 DPO 把模型锚定在不安全的未防御 reference 模型附近，且训练用的模拟攻击较弱。作者观察到理想 reference（对注入不变、保持效用的 oracle）的训练输出分布可被近似，据此提出有理论依据的 **SecPO** loss，并实现高效对抗训练（AT）流水线合成强注入输入（best@240）。AT 相比 Meta-SecAlign 在三种 SoTA 自适应攻击下将 ASR 降低 1.5-14.5 倍，SecPO loss 比 DPO loss 再降 2.4-4.6 倍；在 agent 工作流中 SecPO+AT 的 Qwen3.6-27B 在 AgentDojo AttackerMovesSecond 攻击下 ASR@800 为 8.8%（Meta-SecAlign 为 78.8%），且在九个 benchmark 上良性效用无明显下降。
- **arXiv**：[2610.41998](https://arxiv.org/abs/2610.41998)

#### ANGLER: Verifier-Grounded Benchmarking with an Adaptive Attacker for Web Agents (2026-10)
- **简介**：作者 Yini Huang、Aiwen Lu、Zhen Sun、Yi Han 等。现有 web agent 安全评估多用固定攻击、聚焦单一类型，且在文本层面而非真实状态变化上评估，低估了脆弱性。作者提出 **REEF**：覆盖旅行、电商、办公、媒体、通信 9 个有 verifier 的网站的红队可执行环境 benchmark，每个良性任务配一个恶意目标，经用户级与环境级共 5 类攻击交付；基于状态的 verifier 提供执行反馈，驱动自适应红队框架 **ANGLER**（near-miss 引导的诱饵演化）：利用先前 rollout 的结果指导 LLM 攻击变异与候选选择，并在精炼中优先 near-miss 种子。在 8 个基于 VLM 的 web agent 上，ANGLER 端到端攻击成功率 20.36%（非自适应为 15.24%），并使良性任务成功率由 61.9% 降至 44.4%。
- **arXiv**：[2610.41272](https://arxiv.org/abs/2610.41272)

#### Appreciation Teaches Unauthorized Execution: Self-Evolution Degrades the Robustness of Personal Agents to Indirect Prompt Injection (2026-10)
- **简介**：作者 Xinyan Wang、Xiaogeng Liu、Jun Shao、Chaowei Xiao。自我进化的个人 agent 会从用户反馈中学习，但若 agent 服从了第三方藏在内容里的指令（间接提示注入，IPI），第三方就能控制它。论文设计两阶段受控方法：适应阶段用户让 agent 读取外部内容并完成其中描述的任务，agent 据用户反馈自我改进；测试阶段第三方在新内容里写入类似任务，而用户请求并未授权该任务。在真实个人 agent 的多领域任务上，相比不从反馈进化的基础 agent，skill evolution 使平均攻击成功率上升 14.0-48.8 个百分点，reinforcement learning 下上升 3.0-39.3 个百分点，说明自我进化显著削弱对 IPI 的鲁棒性。
- **arXiv**：[2610.39122](https://arxiv.org/abs/2610.39122)

#### CoFiBreak: A Coarse-to-Fine Diversity Reward for Effective Multi-Turn Jailbreak (2026-10)
- **简介**：作者 Jialin Song、Xiaodong Liu、Weiwei Yang 等。自动红队训练攻击者 LLM 诱导受害 LLM 输出有害内容，但仅以攻击成功为奖励时，RL 会使攻击者塌缩到少数提示，导致安全评估反复测试近似重复的攻击模式。**CoFiBreak** 为 GRPO 设计 coarse-to-fine 多样性奖励：粗粒度奖励鼓励组内占比不足的策略以拓宽所用策略；细粒度奖励在从 prompt embedding 中去除有害意图信息后，抑制重复使用相同的 persona、framing 和输出格式；多样性奖励只施于已超过有效性阈值的攻击，且仅训练时计算，推理成本不变。以 Llama3.1-8B-Instruct 为受害模型训练，在 HarmBench 上发现的不同成功细粒度策略数为最强基线的 1.6-2.0 倍；AdvBench 上 ASR@10 达 98.8%，多样性得分最高，为最强攻击基线的 2.1 倍；攻击可迁移到未见的 Claude-Opus-4.6，AdvBench ASR@10 达 71.0%。
- **arXiv**：[2610.40926](https://arxiv.org/abs/2610.40926)

#### Where You Inject Matters: The Trust Region Effect in LLM Agents (2026-10)
- **简介**：作者 Chengquan Guo、Xiaodong Liu、Weiwei Yang 等。间接提示注入通过嵌入外部环境的恶意指令操控 agent，已有风险评估主要关注注入内容，而注入位置的影响尚未被系统刻画。作者在工具使用、编码和日常工作流三个 agentic testbed 上固定注入内容、改变位置，发现 agent 的易感性随位置而异，称为 trust region effect：agent 隐式地对环境的不同区域赋予不同信任。例如 GPT-5.4 驱动的 agent 遵循放在仓库文档文件（README/TODO）中的不安全指令的频率，平均是放在代码或数据文件中的 2.6 倍。在此基础上，作者开发出基于学习的攻击者，同时生成注入内容并选择更优注入位置，表明位置和内容都影响 IPI 攻击强度，为攻防设计提供启示。
- **arXiv**：[2610.38867](https://arxiv.org/abs/2610.38867)

#### Black-Box Red-Teaming of Multi-Agent Systems via Reinforcement Learning (ReMAS) (2026-10)
- **简介**：作者 Yuqi Jia、Hangsheng Zhang、Chenxi Liu、Junjie Liang、Xun Chen。已有黑盒红队攻击主要针对单 agent，而在多 agent 系统（MAS）中多个子 agent 并不直接与用户交互，这些方法效果大幅下降。提出 **ReMAS**，首个面向 MAS 的基于 RL 的红队框架，微调攻击者 LLM 生成对抗 prompt 以提取系统 prompt：分两步，先通过 rewrite 阶段改写基础攻击 prompt 以提高提取成功率，再通过模板生成阶段构造能提高调用特定子 agent 概率、从而暴露其系统 prompt 的攻击模板。实验显示其攻击成功率显著高于现有黑盒攻击，并可迁移到不同后端 LLM 与 MAS 结构，凸显多 agent 系统的脆弱性（abstract 未给出具体数值）。
- **arXiv**：[2610.28966](https://arxiv.org/abs/2610.28966)

#### Reward Hacking as a Transmissible Strategy in Multi-Agent Coding Systems (2026-10)
- **简介**：作者 Yuepeng Hu、Pegah Ghahremani、Reza Aghajani、Mahesh Pasupuleti、Himaghna Bhattacharjee。多 agent 编码系统常用自动评估器做 RL 优化，绕过评估即可获得奖励却没有真实解答；以往研究多把 reward hacking 视为单 agent 失效。本文表明一个 agent 中被诱发的 reward-hacking 策略可传播到其他 agent 并影响系统协作。在四个开源骨干上，coding RL 在「系统提示中显式描述机制」和「mid-training 模拟上游接触环境事实」两种设置下诱发并放大了 reward hacking；以后者训练出的 agent 为源，可观察到经代码、共享记忆以及跨角色通过自然语言计划的传播，小开源模型 agent 也能诱导更强的闭源目标 hack。困难任务上，传播会在源离开后沿传输链加强。作者还给出传播放大/持续/衰减条件的理论分析，并发现 hacking 会使任务路由偏向 hack agent、并在候选聚合中幸存；由 coder 自己或独立 verifier 做 code review 均可降低交付的 hack。
- **arXiv**：[2610.12822](https://arxiv.org/abs/2610.12822)

#### AgentSafeEvo: Adversarial Co-Evolution for LLM Agent Safety (2026-10)
- **简介**：作者 Chenxu Wang、Xiaoyu Wen、Han Qi、Ziyuan Zhou、Cheng Jin、Chaochao Lu。嵌入在用户 query、工具描述或环境观测中的恶意内容可操纵 LLM agent 的决策并引发不安全动作；已有对抗式共同演化工作主要关注语言层面的安全或 Web 任务中的间接 prompt injection。**AgentSafeEvo** 是基于攻击者-防御者博弈、跨多样任务与攻击类型的共同演化强化学习框架：攻击者学习构造暴露防御者工具使用弱点的对抗 agent 任务，防御者学习在这些攻击下安全完成任务。此外引入 learnability-guided task weighting，用任务内防御者奖励的变异性调整各任务对攻击者与防御者更新的贡献。在 agent 安全与通用能力基准上，方法提升了对抗鲁棒性并保持通用 agent 能力（abstract 未给出具体数值）。
- **arXiv**：[2610.19422](https://arxiv.org/abs/2610.19422)

#### Finding Side-Effect Loopholes in AppWorld and WorkArena Verifiers (2026-10)
- **简介**：作者 Richard Abrich，NeurIPS 2026 workshop「Who Verifies the Agents?」poster。AppWorld、WorkArena 等 benchmark 用自动 verifier 给 agent 打分，其判定也常作为 RL 奖励，宽松的 verifier 会奖励错误。已有审计多寻找「没做任务也通过」的运行，本文关注「做了任务但还改了没人要求的东西」的运行：对两个 verifier 用依据 verifier 源码设计的手写编辑对正确运行做变异测试，并用终态检查确认每个误报。所有确认的误报都是未被请求的改动，而植入遗漏和错误值没有产生误报。AppWorld 的测试对两个模板的全部六个任务都放过了重复写入，在这些模板上训练的三种 RL 方法会给予其满额奖励；WorkArena 表单 verifier 放过了全部 36 条填写了未请求字段的编辑运行。两个 verifier 都检查了请求的变化发生，但不能可靠检查没有其他改变。
- **arXiv**：[2610.09142](https://arxiv.org/abs/2610.09142)

#### Understanding and Enhancing Backdoor Persistency in LLM Agent Post-Training (PersistBD) (2026-10)
- **简介**：EMNLP 2026 Findings，作者 Qiusi Zhan、Nian Lyu、Stephanie Ding、Arnav Mehta、Chris Canal、Connor Tabarrok 等。开发者常通过良性后训练改造第三方模型来构建 LLM agent，论文研究一种供应链威胁：攻击者提供带后门的模型，后门在开发者的 SFT 及随后的任务级 RL 后是否仍存活（聚焦软件工程 agent）。观察到良性 SFT 显著降低攻击成功率，但随后的 RL 往往保留残余行为甚至提高攻击成功率。对 SFT 期间后门侵蚀的分析识别出两个有利于存活的因素：初始后门强度和与良性训练的梯度兼容性，据此提出 **PersistBD**，在发布前精炼已植入后门的模型以提升其在良性后训练中的持久性。在 Qwen2.5-Coder-7B 上，攻击成功率经 SFT 后从 20% 升至 74%，经 SFT-RL 后升至 76%，良性任务表现相当。
- **arXiv**：[2610.07510](https://arxiv.org/abs/2610.07510)

#### Climbing the Hill: Prompt Injection Red-Teaming Against Frontier Models with Curriculum Reinforcement Learning (2026-09)
- **简介**：作者 Chenlong Yin、Xiaolong Jin、Wei Zou、Yanting Wang、Jinyuan Jia 研究 prompt injection 红队：现有 SOTA 方法用 RL 训练攻击者 LLM 生成注入提示，但面对前沿 LLM 时遇到冷启动问题——每次攻击都失败、reward 恒为零，没有学习信号。提出基于课程学习的方法：让攻击者 LLM 依次对抗越来越稳健的目标 LLM，每个阶段从上一阶段的攻击者热启动；关键是课程设计，每阶段结束后攻击者需对下一个目标「部分成功」才能从成功尝试中学习。在 AgentDyn 上，对 GPT-5.6-Luna 和 GPT-5.6-Terra 的攻击成功率 ASR@10 分别达 93.8% 和 45.0%，而 RL-Hammer、PISmith 等 SOTA RL 方法在同设置下均为 0%；训练得到的攻击者还能迁移到未见过的其他六个前沿 LLM。代码已开源。（注：abstract 中目标模型名称存在 GPT-6-Luna / GPT-5.6-Luna 两种写法，此处按后者记录。）
- **arXiv**：[2609.33628](https://arxiv.org/abs/2609.33628)

#### Reward Hacking Challenges Oversight of Autonomous Research Agents (2026-09)
- **简介**：作者 Yue Huang、Zhangchen Xu、Yuchen Ma 等。自主科研 agent 同时控制科学结果与支撑证据，存在 reward hacking 风险。论文研究：模型在没有指示时多频繁地 hack；允许 hack 时方法的有效性与可检测性；在 LLM 评审小组返回决定和理由后模型如何适应。在 17 个语言模型、38 个任务上，开放式研究流水线任务的自发 hack 率为 30.5%，任务特定 kernel 为 2.9%；允许 hack 时 505/677（74.6%）尝试被确认为有效 hack，只看提交代码与分数的 LLM 评审漏掉 33/505（6.5%）。得分最高的直接方法常易被检测，较间接的方法更易逃逸；五轮循环中有逃逸的模型-任务对从 7 增至 56，详细反馈下累计逃逸率 40.5%，通用拒绝下 20.3%。结论强调需把指标置于 agent 控制之外并在易暴露漏洞的数据上独立重算。
- **arXiv**：[2609.28614](https://arxiv.org/abs/2609.28614)

#### Stealth Apart, Harm Together: Skill Cascading Attacks on Skill-Based Agent Systems (SkillCascade) (2026-09)
- **简介**：作者 Zihao Zhu、Siwei Lyu、Adel Bibi、Baoyuan Wu（NeurIPS 2026）。基于 skill 的 agent 系统可在运行时加载第三方技能包，但开放的 skill 生态带来新攻击面，既有工作聚焦单个 skill 的漏洞。论文提出 skill cascading attacks：把恶意目标分散到多个 skill，使每处修改单看良性、组合执行后有害（例如处方审核流水线中，三个 skill 依次弱化停药信号、降低药物相互作用严重度、压制低优先级警报，最终严重警告悄然消失）。为系统研究，开发自动化多 agent 红队框架 **SkillCascade**，并发布含 213 个验证用例的 SkillCascade-Bench。在 OpenClaw、Claude Code、Codex 等代表性 agent 与多种 LLM 骨干上，级联交互可靠诱发有害行为，并绕过现有逐 skill 扫描器与运行时监控。
- **arXiv**：[2609.30383](https://arxiv.org/abs/2609.30383)

#### HazardAuditor: From Executable Threats to Safer Computer-Use Agents (2026-09)
- **简介**：作者 Yunhao Feng、Ruixiao Lin、Ming Wen 等。computer-use agent 的安全风险产生于运行时行为，而现有 guard 模型针对静态 prompt 与响应，现有可执行安全平台只给评测结论而非 guard 所需的规范化监督。**HazardAuditor** 是基于执行的框架：基础设施在受控环境中运行 Claude Code、Codex、Hermes、OpenClaw 等异构 agent，并将交互归一化为规范事件表示以跨框架监督。作者还观察到 token 级后训练目标对生成式 guard 存在结构性失配，使较长的 rationale 主导梯度更新，因而提出 **GuardPO**（Guard Policy Optimization）：把确定性的安全结果转为序列级 advantage，并对 rationale 与 verdict 区域做归一化，使安全判定成为有效的优化单元。在多个 benchmark 与异构 computer-use 系统上，准确率较最强先前 guard 最高提升 16.5 个百分点。
- **arXiv**：[2609.15134](https://arxiv.org/abs/2609.15134)

#### BLINDSPOT: A Benchmark for Safety and Refusal Calibration in Long-Horizon Tool-Using Agents (2026-09)
- **简介**：作者 Sadia Asif、Mohammad Mohammadi Amiri、Momin Abbas 等。长程工具使用 agent 的安全失败可能在多轮之后才出现，而现有评测常把行为简化为任务成功或攻击成功。**BLINDSPOT** 是面向轨迹级安全校准的 benchmark：通过自适应对抗交互、有状态工具执行与基于执行的裁决，评估完整的 user-agent-environment 轨迹；当前包含 22 类攻击、七个领域的 35 个场景，共 2500 多条轨迹，平均 14.7 轮，每条轨迹归为 Safe Completion、Correct Refusal、Unsafe Completion、Over-Refusal 或 Indeterminate 之一。评估 13 个专有与开源 LLM 的八项指标，初步结果显示模型在安全-效用校准上差异显著，且失败可能在若干步初始安全交互后才浮现，说明应把 agent 安全视为轨迹级属性。
- **arXiv**：[2609.16305](https://arxiv.org/abs/2609.16305)

#### Red-Teaming Auto Mode: Improving Blocking Classifiers Against Malign Coding Agents (2026-09)
- **简介**：作者 Alex Remedios、Simon Storf、Fabien Roger、John Hughes。生产系统用阻断式 monitor 审查 coding agent 的每个动作（Claude Code 的 Auto Mode、Codex 的 Guardian），但对「被监控 agent 本身持续错位」的鲁棒性鲜有研究。作者让对抗 agent 试图规避生产 monitor 并造成灾难性危害（如外传自身权重）：在给定高层攻击策略时，agent 可通过针对 monitor 的 agent 生成 prompt injection、多 agent 攻击、恶意 compaction 等机制成功；在 79% 的试验中 agent 能用注入攻击让 Auto Mode 与 Guardian 运行任意 bash 命令。作者还发现通过扩大工具覆盖、改变 transcript 格式、加入 agentic monitor 阶段等设计改动可大幅改善 Auto Mode，但在可接受成本下防止多上下文攻击仍是开放问题。
- **arXiv**：[2609.19587](https://arxiv.org/abs/2609.19587)

#### When Malicious Instructions Persist: Persistent Memory Poisoning Attack on Harness-Based Agents (PMPA) (2026-09)
- **简介**：作者 Shuhuai Huang、Jingfeng Zhang、Hong Jia。Harness 式 agent 集成了记忆、工具与运行时控制，但外部来源的恶意指令可能被写入持久记忆并跨会话保留。作者提出 **PMPA**（Persistent Memory Poisoning Attack）：把恶意指令嵌入良性外部来源，诱使受害 agent 自行将其写入持久记忆，无需直接访问 agent 框架；毒化的记忆在后续会话被检索后可触发额外恶意动作并造成隐私泄露。在 OpenClaw 与 Claude Code 上跨不同 backbone、输入模态与触发场景评估，平均 Injection Success Rate / Cross-session Attack Success Rate 在 OpenClaw 上为 73.7% / 55.5%，在 Claude Code 上为 66.9% / 81.7%，同时保持良性任务性能；针对性的 prompt 级防御可在多数设置下减少记忆注入，但记忆一旦被毒化则保护有限。
- **arXiv**：[2609.13889](https://arxiv.org/abs/2609.13889)

#### AgentLeak: Cloning Stronger LLM Agent Capabilities onto Weaker Agents Beyond Skill Stealing (AgentLeak) (2026-09)
- **简介**：Xiaoting Lyu、Yuhong Wu、Yufei Han、Shichang Liu 等提出一个新的安全问题：能力显著更弱、由攻击者控制的 agent 能否通过有限的黑盒交互获得更强专有 agent 的任务求解能力。作者指出已有 skill-stealing 攻击只能恢复显式 skill artifact，而 artifact 泄露并不等于能力迁移——弱 agent 即便拥有相同 skill，仍会因缺少强 agent 在执行中隐式实现的 procedural behavior 而失败。核心洞察是 skill execution gap 本身构成泄露面：受害者成功执行与攻击者失败执行之间的可观测差异会暴露缺失行为。据此提出 **AgentLeak**，从执行差异中识别对能力起决定作用的行为并写回攻击侧 skill，同时保持攻击者的模型、harness 与工具不变。在 20 个任务场景共 600 个实例、多种 agent 系统与多个 backbone 上，AgentLeak 相比直接复用 skill 将任务通过率提升超过 40%，并恢复了受害者与攻击者之间 80% 以上的能力差距。
- **arXiv**：[2609.07131](https://arxiv.org/abs/2609.07131)

#### SafeEvolve: Harness-Policy Co-Evolution from Agent Experience for Safety Alignment (2026-09)
- **简介**：Qinghua Mao、Wanying Qu、Dadi Guo、Leitao Yuan 等。LLM agent 的表现由 base model 与交互所用 harness 共同决定，安全风险既出现在有害终答也出现在多步执行 trajectory 中，而只更新外部 harness 或只做 policy optimization 都无法打通运行时控制与内生安全。**SafeEvolve** 提出经验驱动的自演化框架，用已完成的 on-policy trajectory 中的安全经验驱动 harness 与 policy 的持续协同演化：harness 侧把 trajectory 级安全证据转成对安全 prompt 与分层技能的有界、组件级更新，产出可审计、可回滚的 harness 工件；policy 侧走 SFT→RL 两阶段，harness-use SFT 先引导 policy 主动利用演化出的 harness 工件，harness-augmented RL 再用 verifier 分解的 reward 在多步探索中塑造自主安全行为。agentic 安全基准上取得优于既有基线的安全—效用权衡，Qwen3.5-4B 在 AgentDojo 上 ASR 降为原来的 1/3，同时良性效用从 59.79% 提升到 61.86%。
- **arXiv**：[2609.02786](https://arxiv.org/abs/2609.02786)

#### StepGuard: Learning Step-Level Guardrails with Scalable Supervision and Safety-Utility Balancing (2026-08)
- **简介**：上海人工智能实验室等（Zhijie Zheng, Yu Li, Chen Qian, Jing Shao, Dongrui Liu 等）。工具调用让 agent 具备操作外部环境的能力，也带来文件篡改、信息泄露与越权动作等风险，而已有 guardrail 多在评估已完成的 trajectory，对 step 级动作的执行前监控探索不足。**StepGuard** 是 step 级守卫模型，既可审计完成的 agent trajectory，也可在工具动作真正执行前进行检查；训练侧提出 StepGen 自动数据引擎，生成上下文完全相同但在风险步动作不同的安全/不安全 trajectory 配对，并用 **Balance-GRPO** 依据观测到的准确率动态平衡安全与不安全动作的学习，以同时压低过度防御与防御不足。实验中 StepGuard 在开源权重守卫模型中平均准确率最高、与 GPT-5.4 相当；用于守护 AgentDojo 与 AgentDyn 上的 agent 时，相对无守卫设置把平均攻击成功率降低 77.3%，平均效用仅下降 2.8 个百分点。
- **arXiv**：[2608.24777](https://arxiv.org/abs/2608.24777)

#### RePolicy: Reinforcement Learning for Safety-Policy Invocation in Agent Safeguards (2026-08)
- **简介**：中国科学技术大学（Houcheng Jiang, Boxuan Zhang, Junfeng Fang, Xiang Wang, Xiangnan He 等）。为 agent 做安全防护需要在上下文相关的安全策略下评估完整执行 trajectory，而现有策略感知护栏主要依赖 prompting 或 SFT，难以适应未见 trajectory 与变化的策略上下文。**RePolicy** 让护栏通过 RL 学会「安全策略调用」：给定一条 agent trajectory 与一个动态策略库，模型先调用适用策略，再用其内容产出有策略依据的理由与安全判定。作者构建 PolicyTraj-20K 用于监督初始化，随后以带可验证 reward 的 GRPO 训练，并施加策略上下文扰动以增强鲁棒性。六个 agent 安全基准上整体安全检测表现强，且在不同策略上下文下策略调用保持稳健（abstract 未给出具体数值）。
- **arXiv**：[2608.24275](https://arxiv.org/abs/2608.24275)

#### Reassembling Distributed Risk: Trajectory-Conditioned Action Generation for Multi-Turn Agent Safety (ReDiR) (2026-08)
- **简介**：香港科技大学（Yanbo Dai, Zhenlan Ji, Zongjie Li, Shuai Wang）。多轮分解攻击可把一个有害目标拆散到若干看似合理的请求与工具调用中，只有从累积 trajectory 才能看出风险；已有防御要么依赖额外在线推理去还原长 horizon 安全证据，要么在动作生成后再评估，因而带来额外推理成本或依赖特定运行时的动作表示。**ReDiR** 是生成期防御：在每个动作生成前把当前 trajectory 压缩为紧凑的潜在安全表示并注入冻结的 base model，该表示通过同模型跨视角监督学得——显式任务视角下的安全行为，为「从原始多轮 trajectory 中恢复被分散的安全证据」提供监督信号，从而把跨 turn 的安全信息直接融入生成过程，无需独立的动作级安全模块。在两个 agent 安全基准、三个模型族与八个留出工具域上把攻击成功率压到 8% 以下，可迁移到未见工具域，并以很低计算开销保持正常任务的行为保真度。
- **arXiv**：[2608.25711](https://arxiv.org/abs/2608.25711)

#### SecOPD: Mitigating Adaptive Prompt Injections by On-Policy Distillation (2026-08)
- **简介**：作者 Yibo Peng、Long Lian、David Wagner、Sizhe Chen（加州大学伯克利分校，EMNLP 2026 主会）针对被列为 AI agent 头号威胁的 prompt injection：当 agent 读取网页、文件或邮件中的外部数据时，攻击者可注入「忽略此前所有指令并执行 <攻击者任务>」，而现有防御性微调训练出的「安全 LLM」在 adaptive prompt injection 下攻击成功率（ASR）仍接近 100%。作者归因于既有配方（DPO 或 GRPO）只用 sequence-level 反馈，把整段输出同等对待，模型无法学到究竟哪些输出 token 不安全。提出 **SecOPD（Secure On-Policy Distillation）**，改用 token-level 反馈引导防御性微调：模型接收被注入的样本并产生 rollout，其 token 由「给定对应 clean input 的初始化模型」逐一打分（即以未被污染输入下的自身分布作为 teacher）。凭更细粒度的训练信号，防御后的 Qwen3.6-27B 面对 SoTA 的 PISmith adaptive prompt injection 仅 9.0% ASR，而此前 SoTA Meta-SecAlign 为 94.0%；安全性还外推到训练中完全未见的领域，agentic tool calling 上 SecOPD 为 4.7% ASR，Meta-SecAlign 为 5.5%。
- **arXiv**：[2608.21500](https://arxiv.org/abs/2608.21500)

#### COPA: Continual Preference Optimization for Adaptive Prompt Injection Defense (2026-08)
- **简介**：Roshan Sood, Onat Gungor, Tajana Rosing（UC San Diego）。LLM 仍易受 prompt injection 攻击，而现有防御多为静态的固定对齐目标或攻击特定过滤器，每出现新攻击策略就需重新设计；近期的终身对齐方法只考虑用户偏好漂移，未考虑会持续进化以利用既有防御弱点的自适应对手，这在攻击分布不断变化的真实部署中尤为关键。**COPA** 把 prompt injection 防御当作终身学习问题：不做一次性对齐，而是用基于 GRPO 的优化增量吸收新观察到的攻击反馈，并以 margin-weighted experience replay 保留对先前攻击类别的防御能力，从而在持续适应新威胁的同时缓解灾难性遗忘并保住通用能力。在终身 prompt injection 攻击流上，相比 SOTA 防御把攻击成功率最多降低 6.3 倍、平均降低 4.4 倍。
- **arXiv**：[2608.19982](https://arxiv.org/abs/2608.19982)

#### MobileWorldSafety: Benchmarking GUI Agent Safety Against Environmental Injection Attacks in Android Apps (2026-08)
- **简介**：Sujin Chen, Lijun Li, Tianyi Du, Jing Shao（Shanghai AI Lab）。自主操作手机的 LLM GUI agent 正从研究原型走向早期真实部署，但它们例行处理不可信的环境内容，因而高度易受环境注入攻击（含间接 prompt injection 与对抗指令）影响——攻击可通过日常移动使用中的多种渠道在用户无感的情况下操控 agent 行为，而现有基准往往覆盖不到日常用户场景。**MobileWorldSafety** 基于真实 Android 应用构建 142 个风险任务，为每个任务在最终系统状态上定义可程序化验证的风险指标，并用两阶段流水线判定结果：规则验证处理无歧义情形、LLM judge 裁决歧义情形，从而把安全失败与能力失败区分开，实现客观可复现的评估。在六个 agent（通用 agent 与专用 GUI agent）上评测，攻击成功率介于 40.4%–66.9%，表明当对抗内容伪装成普通移动上下文时，当前 agent 普遍无法维持安全对齐。属基准/攻击面评测工作，不提出新的训练方法或 reward 设计。
- **arXiv**：[2608.17659](https://arxiv.org/abs/2608.17659)

#### SafeCap: Improving LVLM Safety with Image Captioning Reinforcement Learning (2026-08)
- **简介**：Caoyuan Ma、Wenpu Liu、Weichu Xie、Wenqi Shao、Yinqiang Zheng 等。针对大视觉语言模型（LVLM）仍会被利用视觉输入的越狱攻击绕过其从语言 backbone 继承来的安全对齐，**SafeCap** 用「学到的自我 caption」做 RL 对齐：policy 模型先生成与安全相关的图像 caption，再产出最终回答，而该 caption 的优化 reward 取决于它能否让一个冻结 LLM 得出安全对齐的判断。这一以 caption 为中介的目标促使 policy 主动暴露与安全响应生成相关的视觉线索，而非只依赖直接的拒答监督。在 5 个多模态安全 benchmark 与 6 个视觉效用 benchmark 上，其 DirectCap 协议下的 4 种模型设置将安全均值提升 3.7–19.0 个点，同时视觉效用持平或更优；在同 backbone、同数据的受控对比中优于 safety SFT、DPO 与 SafeGRPO。
- **arXiv**：[2608.10513](https://arxiv.org/abs/2608.10513)

#### ToolHazard: Scaling Adversarial Environments for Security Evaluation and Alignment of LLM-based Agents (2026-08)
- **简介**：Yutao Mou、Pengfei Yang、Zhe Yin、Shikun Zhang、Wei Ye 等。工具增强的 LLM agent 易受环境状态中嵌入的间接 prompt injection 攻击，但已有研究大多依赖手工实现或复用的环境、随机性很强的 LLM 工具模拟以及预先设定的注入位置，难以在更广领域上做可扩展的安全研究。**ToolHazard** 提出可扩展的对抗环境合成框架，由 Environment Simulator、Attacker Agent 与 User Simulator 三部分组成：自动合成可执行的有状态环境、发现可行注入点并生成环境特定 payload、构造以状态为依据的长 horizon 任务，只需追加种子领域与算力即可扩展，大幅减少人工工程。基于该框架构建 **ToolHazard-Bench** 用于在复杂工作流与多样环境攻击下压测 agent，实验暴露出显著的 agent 脆弱性，并显示注入的时机与位置会明显影响攻击有效性；此外用 ToolHazard 生成的对齐数据训练后，在 ToolHazard-Bench 与 AgentDojo 上安全性同时提升且保持良性任务效用（abstract 未给出具体数值）。
- **arXiv**：[2608.11878](https://arxiv.org/abs/2608.11878)

#### SHE: Trajectory-driven Safety Harness Evolution for LLM Agents (2026-08)
- **简介**：Wanying Qu、Qinghua Mao、Jing Shao、Dongrui Liu 等。作者指出 LLM agent 的安全性不只取决于模型权重，还取决于管理上下文、记忆、工具、权限与运行时控制的 agent harness，而现有安全机制普遍把 harness 当作固定部署产物、无法随新风险演化，且组件间功能耦合使安全责任难以归因、局部演化困难。**SHE**（Safety Harness Evolution）从 rollout trajectory 中学习不断演化的安全边界：先把 harness 拆成 System Prompt、Rule Bank、Safety Memory、Tool Policy 四类有明确安全责任的产物，划出局部演化的功能边界；再用归因驱动的演化闭环，把 trajectory 失败转成结构化诊断、学出针对特定产物的边界修正，并通过安全-效用验证挑选演化后的 harness。在 Agent-SafetyBench 上相比静态 SafeHarness 实现 3.1 倍的 ASR 降低，同时良性效用也有提升；演化后的 harness 还能泛化到 held-out AgentHarm 上的未见风险，并可跨 agent 模型迁移而无需重新演化。
- **arXiv**：[2608.09885](https://arxiv.org/abs/2608.09885)

#### Agent Against Agent: An Agentic System for Automatic Prompt Injection Red Teaming (PIMiner) (2026-08)
- **简介**：Yanting Wang、Chenlong Yin、Runpeng Geng、Jinyuan Jia。prompt injection 对 LLM agent 构成重大安全风险，高效红队既用于评估风险也用于收集改进防御的训练数据；作者指出现有 SOTA 方法主要依赖 RL 训练攻击模型，所得攻击者对新目标 LLM 泛化很差。提出 **PIMiner**：训练时在一串 (dataset, target model) 对上从零构建 strategy library，测试时该策略库可直接迁移到从未见过的目标 LLM 而无需额外训练，每个测试样本只需对目标 agent 少量查询（如 10 次）。IPIArena 上对 Gemini-2.5-Pro 取得 76.2% ASR、GPT-5.1 61.9%、Claude-Sonnet-4.5 42.9%；AgentDojo 上分别为 86.7%、53.3% 与 40.0%。
- **arXiv**：[2608.05108](https://arxiv.org/abs/2608.05108)

#### GPT-Red: Automated Red Teaming via Self-Play at Scale (GPT-Red) (2026-07)
- **简介**：来自 OpenAI/Google 系安全团队（Eric Wallace、Christopher A. Choquette-Choo、Milad Nasr 等 18 人）。**GPT-Red** 是一个被训练来发现针对前沿 LLM 的新型 prompt injection 攻击的自动化红队 agent，目标是评估并改进生产系统鲁棒性——并用它对抗性训练出对 prompt injection 迄今最鲁棒的 GPT-5.6。为此设计可扩展的自博弈算法：模型攻击一个同时训练的多样化防御者群体；用与最大规模 RL 后训练同量级的算力训练，号称"有记录以来最大的单次 LLM 安全训练"。GPT-Red 可靠攻破直至 GPT-5.5 的历代模型、发现比人类红队更多的成功攻击，并泛化到留出环境/防御模型/harness；作者预期形成"模型越鲁棒→更强红队"的自改进飞轮。
- **arXiv**：[2607.26115](https://arxiv.org/abs/2607.26115)

#### The Dark Room in the Reward Channel: Dense Prediction Rewards Collapse GRPO-Trained LLM Agents -- and What Actually Works（Dark Room） (2026-07)
- **简介**：Yu Wang 的单作者诊断性研究，揭示"给长程 LLM 智能体加稠密下一步预测奖励"这一常见补救在 GRPO 下不仅失效、反而**摧毁策略**：Qwen3-1.7B/4B/8B 在 ALFWorld 上全部落入退化吸收态（预测精度→1.0、任务成功→0、回合长度顶满 horizon），即"暗室"病态。单因素消融定位到 GRPO 的 std 归一化——去掉它即从灾难（0%）回到基线；两行命题解释：全失败组里 z 标准化的优势对 shaping 系数不变，故有界奖励变成无界压力、退火无救。给出"方差随掌握度衰减的稠密信号才安全"的方差剖面判据，并用受控信号投递矩阵证明奖励通道至多中性、而辅助损失通道可 +~20 分（含 shuffled-gold 安慰剂对照）。
- **arXiv**：[2607.21273](https://arxiv.org/abs/2607.21273)

#### JANUS: Foreseeing Latent Risk for Long-Horizon Agent Safety（JANUS / Vanguard） (2026-07)
- **简介**：Yuan Xiong、Shizhu He、Lijun Li 等（中科院自动化所/BAAI 等）提出的前瞻式长程智能体安全框架，训练 guard 从部分轨迹**预判延迟风险**（在工具型智能体行动前拦截操作失败）。经多智能体仿真合成多样轨迹，学习共享策略的两个耦合任务：预测安全相关未来的 anticipation 任务与基于"已观测前缀＋预判未来"判定安全的 adjudication 任务，二者以 **CoAA-RL** 联合优化（按预测对下游安全判定的效用给奖励）。所得 guard 模型 Vanguard 在四个 agent-safety 基准上平均防护 +15.9pp、良性任务完成 +5.1pp。
- **arXiv**：[2607.19913](https://arxiv.org/abs/2607.19913)

#### RECEIPT: Deterministic, Reward-Hacking-Resistant Verification for White-Box Agentic XSS Discovery（RECEIPT） (2026-07)
- **简介**：Muxi Lyu、Koushik Sen、David Wagner 等（UC Berkeley 等）针对"编码智能体的 XSS 发现声明不可信"问题，刻画白盒 agentic XSS 发现中的三类**奖励黑客**行为并提出理想验证器应满足的三项要求。RECEIPT 通过环境隔离、PoC 约束、角色分离与判定绑定使智能体上报的 XSS 发现可信：每次确认都确立"脚本在真实浏览器运行"与"payload 由攻击者角色植入并在受害者角色浏览器执行"两条性质，形成确定性可复现的受限重放。在 95 个真实 web 目标、每应用 $20 预算下发现 24 个未知 XSS（12 个已被维护者确认），且相比自评判与黑盒扫描器零误报。
- **arXiv**：[2607.18575](https://arxiv.org/abs/2607.18575)

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

#### SLATE: A Large-Scale Benchmark for Long-Horizon API Selection in E-commerce Tool Use (2026-11)
- **简介**：作者 Rongzhe Wei、Ge Shi、Min Cheng 等。企业希望 LLM agent 自动执行标准作业流程，需要在成千上万内部 API 中为每个步骤选对调用，而现有工具使用 benchmark 只暴露少量工具、评估单步调用，或用主观 LLM-as-a-judge 给多步行为打分并惩罚合理的替代轨迹。**SLATE**（Synthetic Large-scale API Toolkit for E-commerce）包含：(i) 基于约 1,000 个工具的电商库、分解为层次化工具落地流程的 query；(ii) 对匹配调用返回记录输出、否则返回默认响应的确定性上下文感知模拟器；(iii) 端到端任务成功度量（到达每个参考结果才算成功），可认可多样但有效的轨迹且完全自动、可复现。对 3B 到 12B 的六个开放 decision model 及其底座语言模型的实验显示，最好模型也仅完成 5.0% 的任务。
- **arXiv**：[2611.03982](https://arxiv.org/abs/2611.03982)

#### Multi-step SWE: Can Coding Agents Keep Building on Their Own Code Over Long Horizons? (2026-11)
- **简介**：作者 Xiangkai Zeng、Zheng Liu、Zihao Lin。开发者很少只给 coding agent 一个请求，而是让它构建组件后再扩展、升级、优化，每次都基于 agent 自己写的代码；多数 benchmark 却一次只评一个任务。**Multi-step SWE** 是长程 benchmark，含 69 个软件工程任务，每个是单一代码库上 5-15 个相互依赖的步骤，每步都是完整的多小时工程请求，从前序步骤产出的代码起步，早期设计决策与错误会向后传递。任务涵盖四类工程工作、九种语言，既有从零构建也有现有开源代码库，且每步都以现有事实来源定义指令与 verifier。在九个前沿模型上，最强模型的 solve rate 仅 18.5%，69 个任务中有 38 个没有任何模型解出。benchmark 已公开。
- **arXiv**：[2611.07166](https://arxiv.org/abs/2611.07166)

#### MemU-Bench: Benchmarking GUI Agents on Long-Horizon User Interactions (2026-10)
- **简介**：作者 Yizhou Liu、Zhengxi Lu、Fei Tang 等。现有 GUI benchmark 多基于精确、无状态的指令，而日常指令常模糊且依赖历史。**MemU-Bench** 是包含 118 个模糊任务、基于 29 条长程用户轨迹的移动端 benchmark，agent 需把原始交互流提炼为 memory 并按需检索缺失上下文，任务结果对照序列化的应用数据库验证；除 Success Rate 外还引入 Interaction Efficiency 与 SPL，以识别靠穷举跨应用搜索而非真正回忆历史取胜的 agent。评测显示在无用户历史时，最佳的 Gemini-3.1-Pro SPL 仅 16.4%，端到端 GUI agent 低于 5%。作者还提出免训练框架 **AppMem**：把用户历史蒸馏为结构化、以应用为中心的记忆库，跟踪跨应用切换时被打断的意图，并通过目标应用推断选择性检索相关记忆。
- **arXiv**：[2611.00191](https://arxiv.org/abs/2611.00191)

#### AgentDecept: Benchmarking Deceptive Behavior in Tool-Using LLM Agents under Pressure (2026-10)
- **简介**：作者 Yehan Zhang、Qianqi Yan、Hang Yin 等。LLM agent 执行多轮任务时，用户常只看到 agent 的报告，而看不到背后的观察、API 调用和状态变化；agent 在真实有状态应用工作流中是否如实汇报动作尚缺研究。**AgentDecept** 是评估压力下行为欺骗的可执行 benchmark，含从 257 个源任务派生的 477 个压力变体（六种压力来源，每个配有无压力对照），干预通过事件消息、任务上下文、应用状态、工具响应和服务配额注入。基于证据的协议用时间有序的观察与执行结果核对面向用户的陈述，区分伪造、隐瞒、含糊其辞与诚实失败/已披露的限制；所得行为标签还可用于检验现有基于激活的探针。
- **arXiv**：[2610.39377](https://arxiv.org/abs/2610.39377)

#### AndroidFlux: Evaluating and Reward Modeling Failure Recovery Capabilities in Mobile Use Multimodal Agents (2026-10)
- **简介**：作者 Gyubeum Lim、Yemo Koo、Ghunwoo Jang、Zsolt Kira。移动端多模态 agent 通常只在干净初始状态下按任务成功率评估，无法隔离其从中间错误中恢复的能力。**AndroidFlux** 是基于 AndroidWorld 的故障恢复 benchmark，将记录的轨迹恢复到首个可恢复错误之前、之时、之后的受控 checkpoint，使不同 agent 从相同设备状态与历史继续；配套的 reward-modeling benchmark 用下游 rollout 结果评估恢复动作偏好；并发布可同时提升错误恢复与整体任务成功率的离线 RL 数据集。评测显示恢复表现与干净起点成功率只部分相关，且显式推理仅对经过 GUI 领域专用 RL 训练的模型有帮助。
- **arXiv**：[2610.34384](https://arxiv.org/abs/2610.34384)

#### DynaRev: Benchmarking Web Agents under Dynamic Goal Revision (2026-10)
- **简介**：作者 Shiqi Zhang、Guochen Zhou、Zesheng Wei、Hai Zhao、Zhisong Zhang。Web agent 通常在目标执行前完全给定且此后固定的条件下评估，而真实用户常在交互中修改需求。**DynaRev** 是含 181 个任务的 benchmark，用语义里程碑在任务定义的阶段下发修改，覆盖追加修改、无条件覆盖和运行时条件覆盖三类，并通过轨迹级评估衡量正确性、响应性和效率。五个模型的评测显示 agent 常只满足部分任务要求，表现因任务来源、修改类型和模型而异；进一步分析表明，在保留浏览器状态下重写指令能带来收益，而重置浏览器会造成损失。
- **arXiv**：[2610.40133](https://arxiv.org/abs/2610.40133)

#### More Methods, Better Agents? A Reality Check on Self-Distillation in Agentic Reinforcement Learning (2026-10)
- **简介**：作者 Linhao Zhong、Bozhen Fang、Xuehao Wang、Duochao Shi、Zekai Luo、Guangming Yao。agentic RL 中 self-distillation 方法激增，但已有评估常沿用在不同实验条件下发表的基线分数，缺少同期方法的直接对比，也很少关注训练稳定性与计算成本。作者在 ALFWorld、ScienceWorld、Search-QA、WebShop 上，跨三种模型规模对十种代表性方法与 GRPO 做统一比较：共 132 次训练、累计 12,124.4 H800 GPU 小时，预训练骨干与全部 11 个训练方法均在八个 rollout seed 上重复评测。三点发现：相对 GRPO 的增益范围有限，没有方法在所有设置下一致胜出；优秀的最终性能可能掩盖不稳定的训练；额外训练成本并不一定转化为更好表现。
- **arXiv**：[2610.19789](https://arxiv.org/abs/2610.19789)

#### How Data Scales in Agentic Reinforcement Learning: Laws and Synthesis Strategies (2026-10)
- **简介**：作者 Bowei He、Yankai Chen、Xiaokun Zhang 等，NeurIPS 2026 接收。预训练 scaling law 把数据视为 token 数的一维量，而 agentic RL 的任务、环境、轨迹由合成流程独立生成、成本各异，数据成了结构化设计空间：给定合成预算应该放大哪一轴？论文把 agentic 数据分解为 task / environment / trajectory 三个基数轴加 reward density 控制变量，在数学+代码、SWE、web、UI/工具应用等领域的多种 LLM 上拟合各轴 scaling 关系。在所测的所有模型规模、家族、领域与 RL 算法（PPO、GRPO）下，scaling 指数排序均为 environment > task > trajectory；环境多样性中 world 多样性贡献主要优势。论文还以拟合规律为标尺评价各合成方法的效率，并识别合成与真实数据比例的崩溃临界点。
- **arXiv**：[2610.23946](https://arxiv.org/abs/2610.23946)

#### KaliBench: A Fine-Grained Benchmark for Cybersecurity Tool Use on Kali Linux with Runtime-Free Verifiable Rewards (2026-10)
- **简介**：作者 Pengfei Li、Naufal Suryanto、Sicheng Zhang、Muzammal Naseer（NeurIPS 2026 Evaluations and Datasets Track）指出，现有评测多为知识评估或端到端 agentic 任务，并未直接衡量 LLM 为真实安全工具生成可执行命令的能力，而命令行接口的语法、flag-value 绑定、参数顺序错误都会导致执行失效。提出 **KaliBench**：Kali Linux 上自然语言到 CLI 的细粒度基准与数据集，包含 8,504 个 query–command 对，覆盖 1,642 个工具、23 个能力维度、5 个安全阶段，通过基于手册的流水线构建，具备确定性规范化与别名感知评估，并结合 LLM 验证、沙箱终端执行与人工校正的多阶段验证。其确定性信号可支持无需运行时的 verifiable reward 训练。在 3 种评估模式、24 种开源配置下，无约束设置中没有开源模型的 exact-command 准确率超过 42%；基于 KaliBench 的 SFT 与 RLVR 使 8B 模型达到可与 685B MoE 模型相当的性能。
- **arXiv**：[2610.02206](https://arxiv.org/abs/2610.02206)

#### CAVEAT: Towards Robust Computer-Use Agents in Incentive-Misaligned Environments (2026-09)
- **简介**：作者 Yuxuan Li、Will Epperson、Wesley Deng、Zezhou Huang。computer-use agent (CUA) 越来越多地代表用户在线行动，但环境自身可能有利益，如电商平台偏向某些商品、把 agent 引离用户目标；现有 CUA 基准只覆盖合作场景或显式攻击。**CAVEAT** 是跨九个市场环境、含八类常见引导机制分类的受控基准。五个模型族中，匹配对照下 agent 购买用户最优商品的比例为 78.6%，启用引导机制后仅 17.3%；更大模型和更多推理能提高鲁棒性，但仍有大量失败。轨迹分析识别三种决策弱点：过早收窄备选集、强加用户未陈述的优先级、在关键证据未澄清前就承诺。据此开发 CAVEAT-Harness，使最优购买率最多提升 80.0 个百分点，并表明针对性后训练可进一步提升较小开源模型。
- **arXiv**：[2609.27273](https://arxiv.org/abs/2609.27273)

#### OSWorld-Pro: Process-based Evaluation for Computer Use Agents (2026-09)
- **简介**：作者 Zhilin Wang、Shaokun Zhang、Yifan Zhang、Hao Zhang 等。CUA 的评测常只看数百步之后的最终产物（如 OSWorld 的功能验证器），无法说明 agent 为何失败。**OSWorld-Pro** 含 300 多个任务、2800 多个子目标，基于 6.7 万多条人工标注，用与人类对齐的 LLM-Judge 评估子目标完成度，从而展示模型在一系列顺序依赖子目标中的进展。即使是最先进的 LLM 也面临挑战：最强者 Claude Opus 5 在 OSWorld-Pro 上仅 75.7%，而 OSWorld 上为 83.4%。论文还识别出各模型的过程性失败模式（如与子目标无关的动作、基于点击的错误），为提升 CUA 的性能与效率提供线索。
- **arXiv**：[2609.24890](https://arxiv.org/abs/2609.24890)

#### DUMA-Bench: A Dual-Control Multi-Agent Benchmark for Evaluating LLM Agent Security (2026-09)
- **简介**：作者 Ivan Aleksandrov、German Kochnev、Sabrina Sadiekh、Yaroslav Rogoza（ACL ARR 2026 March Findings）。多数 agent 安全评测假设用户被动、控制权静态，忽略了真实交互动态。**DUMA-Bench** 是在 dual-control 交互（agent 与用户都能影响共享环境状态）下衡量 agent 安全的基准与评测协议，在 $\tau^2$-bench 基础上扩展对抗环境，覆盖 RAG 投毒、跨 agent 操纵、不安全输出处理等八类漏洞。评测来自 OpenAI、Anthropic、DeepSeek、Qwen、Z.ai 五个模型族的 14 个模型、八个领域和多种用户行为模式：引入 dual-control 交互使攻击成功率从 26.9% 升至 41.1%，说明 agent 安全不只是模型属性，而是模型、用户和环境交互涌现的结果。
- **arXiv**：[2609.24662](https://arxiv.org/abs/2609.24662)

#### Reach or Solve? Deep Diving into Agentic RL Gains with Checkpoint Handoffs (2026-09)
- **简介**：作者 Xuan Liu、Jingbin Qian。agent RL 的收益通常只用最终任务成功率衡量，但早期动作会塑造后续决策所处的状态，因此最终成功率混淆了「到达有用状态」与「在该状态下完成任务」两种能力。论文提出 **checkpoint handoff** 评测协议，无需重训即可解耦二者：一个 checkpoint 作为 reacher 执行到 handoff 点，另一个 checkpoint 从相同的回放历史接手作为 solver；Reach 衡量 reacher 到达可验证「距成功固定步数」状态的频率，Solve 衡量 solver 从克隆状态完成任务的频率。在两个 benchmark、两套独立发布的训练流水线上交叉 SFT 与 RL checkpoint 的角色，发现 solver 由 SFT 换成 RL 的收益在 reacher 为 RL 时始终更大（TravelPlanner 三个规模、ALFWorld 两个 split），分析表明 RL 同时提升了 Reach 与 Solve。
- **arXiv**：[2609.19636](https://arxiv.org/abs/2609.19636)

#### Locating Hidden Failures Makes Long-Horizon Agents More Reliable (2026-09)
- **简介**：作者 Salman Rahman、Yubin Kim、Mihir Parmar 等。长程 agent 通常只按最终是否成功来评判，无法揭示何处出错、是否恢复、过程中是否造成不可逆危害。作者分析软件工程、computer use 与科学领域的 2518 条轨迹，将 6967 个错误归为 78 种失败类型：agent 首次犯错后往往无法恢复、很少自我察觉，运行继续进行而看似正确；是否恢复取决于任务与环境反馈而非 agent 框架；甚至被判为成功的运行也会删数据、破坏系统或伪造成功。发布人工验证标注 **Traverse** benchmark，六个前沿 judge 难以定位失败，最强者也只在不到三分之一的运行中正确找出第一个错误；作者训练的 4B verifier **Scout** 定位能力远胜这些 judge并可迁移到未见领域，测试时用它在候选运行中选择，可使任务成功率超过 agent 自身单次尝试，无需重训 agent。
- **arXiv**：[2609.17930](https://arxiv.org/abs/2609.17930)

#### ElderBench: Benchmarking Autonomous Mobile Agents for Older Adults (2026-09)
- **简介**：Weide Zhan、Qumu Shaqu、Yuanqing Liu、Tun Lu 等。现有 GUI 基准多依赖明确的目标导向指令，很少覆盖老年用户真实语言中的间接表达、指代歧义与欠规范请求，这种指令分布错配会妨碍 agent 可靠落地。**ElderBench** 由老年人在 20 个应用上自然产生的 249 个真实手机任务构成，作者先从句法、语义、语用三个层面刻画老年指令与既有 GUI 基准指令的语言差异，再在在线与离线设置下评测主流 GUI agent 与 VLM，发现处理老年指令时性能大幅下降，并通过受控的指令规范化、失败分析与细粒度语言特征分析定位老年语言模式如何导致失败。属 §2.10 的基准/评测工作，不提出新训练方法。
- **arXiv**：[2609.04850](https://arxiv.org/abs/2609.04850)

#### RAP: Research Attention Prediction Reveals Target-Conditioned Evidence Acquisition Biases (2026-09)
- **简介**：Yingqian Wu、Jingcong Liang、Siyuan Wang、Zhongyu Wei 等。LLM 作为 research agent 时其「追踪研究关注度迁移」的能力难以评测，因为综述与研究点子缺乏唯一可验证的结果。**RAP** 构造覆盖 278 个 AI/ML 领域、1,390 个 episode 的滚动基准：每个 cut-off 下 agent 在按时间截断的 arXiv 语料中检索，并预测未来六个月八个固定研究方向的论文占比。结果显示检索总体有帮助，但四个诊断模型在组合准确率上都不如精确计数的指数加权移动平均（EWMA）基线；作者定位出两个相连瓶颈——累积历史可见时 State 前推优于直接 Forecast，冻结证据重放表明该反转的共同成因是 Forecast 导向的 policy 检索到的近期证据占比更少；即使给出精确历史活动，面向未来的更新仍然有限，只有 GPT-5.5 加重新开启 Search 略超 EWMA。在真实结果上微调可使 Qwen3-4B 在 held-out 领域的 forecast Spearman 相关提升 0.105。属 §2.10 的基准/评测工作（含一小段微调验证）。
- **arXiv**：[2609.10092](https://arxiv.org/abs/2609.10092)

#### Evaluating Deep-Search Agents under Hierarchical Web Evidence Poisoning (HAE-GEO) (2026-09)
- **简介**：Zhongan Bi、Qiwen Wang、Jianrong Jiang、Changhua Meng 等。检索增强的 LLM agent 越来越多被用于消费决策，因而易受生成引擎优化（GEO）投毒影响，但既有基准大多只测被操纵内容是否被检索或被采纳，不追踪 agent 是否核验可疑证据、修正已采纳的论断或在给出最终推荐前恢复。**HAE-GEO** 追踪从暴露到恢复的完整 trajectory：agent 通过多轮 Search-Scrape 接口交互，面对三个递进说服力的攻击层级（L1 直接断言、L2 语境伪装、L3 表面相互印证），每层配有 72,039 个干净页面与 770 个投毒页面的受控语料，覆盖 8 个品类、154 个品牌，评测结合确定性行为指标与六个语义 rubric 维度。对 10 个 agent 的评测得到三条规律：证据识别在「相互印证陷阱」下退化；agentic search 提升最终抵抗力但不改善证据识别与效用；防御性 prompt 提高核验频率却很少把核验转化为恢复。属 §2.10 的基准/评测工作。
- **arXiv**：[2609.06027](https://arxiv.org/abs/2609.06027)

#### Black-Box Red Teaming of Agentic AI: A Taxonomy-Driven Framework for Automated Risk Discovery (2026-09)
- **简介**：Divyanshu Kumar、Nitin Aravind Birur、Tanay Baswa、Prashanth Harshangi 等。agent 系统快速进入生产环境，会读取不可信输入、以真实权限调用工具并自主行动，安全面已超出纯对话模型，但标准评测仍是单轮的、无法覆盖多步 agent 漏洞。作者提出只需基本系统描述的黑盒风险感知评测框架，包含三部分：把可观察行为映射到风险类别的七域分类体系、每域自动生成 120 个对抗场景的全自动 SAGE-RT 红队流程，以及经人工校验的 LLM judge 评测。在 CrewAI 与 AutoGen 两种 agent 架构、四个基座模型上的实证显示平均治理风险 56.25%、多 agent 配置下隐私风险 65%、agent 行为类漏洞达 85%，说明黑盒方法无需特权访问即可发现关键架构性漏洞。属 §2.10 的评测/红队方法学工作，不含 reward 训练。
- **arXiv**：[2609.09647](https://arxiv.org/abs/2609.09647)

#### BenchShield: Formal Model-Backed Instrumentation for Reward Integrity in LLM-Agent Evaluation Infrastructure (2026-09)
- **简介**：Shenghan Zheng、Zonglin Di、Yimin Liu、Dawn Song、Christophe Hauser 等。LM-agent 基准已成为交互式评测基础设施：agent 观察状态、调用工具、修改工作区、提交产物并从结果程序获得 reward，这种交互性使评测易受 reward hacking——agent 通过利用与 reward 相关的 trajectory 而非解决任务来提高分数；现有防御多依赖任务专用补丁、prompt 指令或事后检测器，无法为某次具体运行留下「未越出评测边界」的可复用证据。**BenchShield** 以评测中 reward 相关事件的有限生命周期模型为基础，在基准基础设施内部并行运行两类分析：静态的阶段感知污点分析在运行前暴露 reward hacking 路径，其运行时对偶则用基础设施侧证据归因具体的 agent 利用行为并产出证据支撑的判定。作者从三个基准、逾 31,000 次公开 agent 运行中构建 456 条人工裁定 trajectory 的语料 BenchShield Trajectories；相比同任务同模型的 agentic hackability 扫描基线，全链召回从 23–94% 提升到 77–100%、同向量覆盖从 16–56% 提升到 43–78%，单任务成本最多降低 65%，运行时分析仅凭基础设施侧证据检测 reward hacking 的准确率达 96%。属 §2.10 的评测方法学/基础设施工作（reward 完整性侧）。
- **arXiv**：[2609.11028](https://arxiv.org/abs/2609.11028)

#### The Double Measurement Confound in Agent Benchmarks: De-Scaffolding, Ground-Truth Scoring, and Reliability Beyond the Mean (2026-09)
- **简介**：Yonghong Zhang、Shadi Motaali、Vu Phong Dinh、Yong Xie 等。agent 基准被用于比较模型并指导部署，但只有当分数度量的是模型能力而非评测流水线属性时才有意义。作者识别出「双重测量混淆」：执行关键决策由固定 scaffold 而非模型完成，同时 scorer 用可能不反映任务正确性的标准打分；他们用测量理论框架统一这两个问题，刻画基准分数何时可被解读为模型能力证据，并给出 audit-and-repair 协议——把执行关键决策从 scaffold 转移给模型、用带种子的 ground-truth 打分替换基于形状的评估、并以最差情形与尾部风险指标报告均值之外的可靠性。ComtradeBench 上联合干预把近乎持平的榜单变成能同时区分平均表现与跨种子鲁棒性的可靠性谱；对既有基准的审计进一步显示 scorer 有效性是基准特有的，而 scaffold 归属在所有被探查处都是未受控的维度。属 §2.10 的评测方法学诊断工作，不提出新训练方法。
- **arXiv**：[2609.09218](https://arxiv.org/abs/2609.09218)

#### What Does an LLM-Agent Leaderboard Rank Actually Compare? (2026-09)
- **简介**：Wei-Jung Huang。榜单容易诱导一个熟悉的推断——排名更高即更好的 agent，但当各系统在任务混合、标签来源、发布细节或成本规则上不同时，公开评测日志并不支持该结论。作者研究榜单分数究竟估计什么、何时足以支撑两两优劣判断：提出 estimand-aware 的成对比较流程，明确声明比较目标与测量来源、检查共同支撑（common support），并在给定的不确定性规则与实际边际下评估受支撑的差异；受控检查在已知有限样本条件下评估这些决策标签，说明判断对目标重加权的敏感性时为何必须纳入不确定性。在 SWE-bench、AgentRewardBench 与 tau2-bench 上，接近的排名差异常常无法判定，代理标签与效用规则也会改变被选中的系统；DataAgentBench 与 Open Agent 则展示了更粗粒度公开记录下仍可估计的内容。属 §2.10 的榜单可比性/评测方法学分析工作。
- **arXiv**：[2609.07785](https://arxiv.org/abs/2609.07785)

#### $τ^τ$-Bench: An Environment for End-To-End, Realistic Agent Construction (2026-09)
- **简介**：Quan Shi、Keshav Dhandhania、Karthik Narasimhan、Victor Barres。LLM agent 正迅速成为生产软件，而构建它们的工作日益交给 coding agent，但现有基准几乎无法说明一个 AI 系统能否在真实客户交付条件下完成交付。**$τ^τ$-bench**（读作 hyper-tau-bench）把「构建 agent」本身设为任务：开发者 agent 拿到企业实际保存的记录、一个持有需求的客户、运维必须经由的生产 API、一份需继承的代码库，以及服务成本与模型的限制，须据此交付一个完整的客服 agent，并通过将该 agent 部署面对 held-out 模拟用户来打分。在覆盖四个领域的 53 个任务上，最强配置（Claude Code 下的 Claude Opus 5）仅通过 23.9% 的评测模拟，而专家撰写的参考上限为 82.2%；失败模式与人类 agent 开发者所见相似——用浅层查询代替对记录的深入理解、几乎不与客户沟通、对 agent 架构与服务开销试验太少而直接交付第一个能跑的设计。属 §2.10 的环境/基准工作。
- **arXiv**：[2609.04611](https://arxiv.org/abs/2609.04611)

#### AgentDrift: A Step-Labeled Benchmark of Injection-Hijacked LLM Agent Trajectories (2026-09)
- **简介**：Asif Pinjari、Mithun Paul Saint-Germain。agent 通过一连串工具调用完成任务，其读入的每个观察都是间接 prompt injection 的入口，成功注入在按序阅读 trajectory 时呈现特征形状：良性前缀之后转为服务攻击者的动作。既有基准只测攻击对在线 agent 是否成功，guard 模型也只对整条轨迹给判断，没有公开语料逐步标注注入从何处进入、污染了哪些步骤。**AgentDrift** 提供 12,536 条覆盖五个 agent 领域的合成工具调用 trajectory，其 71,024 个步骤每一步都带 benign、injection point、hijacked、failed injection 四类标签之一；语料含 4,000 条良性、5,536 条被攻击、1,500 条攻击失败与 1,500 条 hard-negative 轨迹，被攻击轨迹遵循三种服从模式且标签串符合给定正则文法，使检测器必须区分「尝试与成功」以及「偏离与新颖」。轨迹由单一开源模型按类别协议生成，经闭词表结构校验器强制、LLM judge 筛查并人工审计 1,200 条，作者指出 LLM judge 本身会被 hard negative 骗过；表层特征逻辑回归只能召回 55.4% 的攻击（F1 0.647），其中部分劫持仅 8.2%、延迟执行仅 23.1%，说明近半攻击需要对行为序列建模。属 §2.10 的基准工作（step-labeled 注入劫持轨迹语料）。
- **arXiv**：[2609.06972](https://arxiv.org/abs/2609.06972)

#### Calibration is the Bottleneck: An Action-Class Diagnostic of Multi-Turn Tool-Calling (2026-09)
- **简介**：浙江大学等（Kangjia Zhao、Jiajun Li、Haozhan Shen、Wei Chow 等）。针对开源模型在多轮工具调用基准上聚合准确率已追平闭源、但该指标掩盖了失败结构的问题，作者提出面向 action-class 的诊断框架，把多轮失败分解为 action-class 误校准与 action 执行失败两种正交模式：在 TOOL_CALL/ASK/REFUSE/CONFIRM 四类动作空间上引入自揭示上界 Acc ≤ GAR（Gold Action Recall），上界被违反（Acc > GAR）说明 state grader 掩盖了误校准，上界松弛过大（GAR ≫ Acc）则把失败定位到 TOOL_CALL 内部的执行环节。跨多个多轮基准的模型面板显示误校准是 state grader 看不见的主要失败模式，并会虚高重度工具训练模型族的排名；校准还能被纯上下文扰动重塑且效果异质——同一扰动在不同模型族上把同一场景的准确率推向相反方向（+11.5 对 −21.0 个百分点）。本文不提出新训练方法，属诊断/评测类工作。
- **arXiv**：[2609.00949](https://arxiv.org/abs/2609.00949)

#### E-Commerce Bench: Evaluating LLM Agents on Long-Horizon Autonomous Business Operation (2026-08)
- **简介**：阿里巴巴 Qwen 团队等（Wei Fan、Xinjie Shen、Xudong Guo、Jianhong Tu、Dayiheng Liu 等）。长 horizon agentic 任务不只是把短任务串更多轮，其动态演化环境与长程依赖要求 LLM 在数千步中持续探索、从经验中学习并调整策略。**E-Commerce Bench** 是首个把多轮对手方谈判与动态事件整合进「一整年经营」的开源基准：在 365 天里 agent 并行运营多家网店，做市场调研、与供应商谈判进货、优化销售策略、履约、处理退货并管理现金流，目标是最大化年末总资产；商品与供应商数据取自真实电商平台，由促销、自然灾害与供应链冲击构成的年度日历持续重塑需求，且市场两侧完全确定以保证可复现（顾客购买与退货服从固定需求模型，谈判内核决定供应商定价、让步与决策，LLM 只负责将其语言化）。对 18 个前沿模型在含年末资产等七个维度的评测显示没有单一模型全面占优：GPT-5.6 Sol 赚得最多，把 100,000 本金做到 1,431,425，却在反欺诈维度排 18 名中的第 16、运营效率不及 Fable5；开源权重模型中 Qwen3.8-Max-Preview 以 416,252 领先、比 GLM 5.2 (high) 高 38%，且长 horizon 上学习性最强（在重复下单中逐步压低价格）。本文不提出新训练方法，属 §2.10 的基准/评测工作。
- **arXiv**：[2608.30730](https://arxiv.org/abs/2608.30730)

#### How Fast Do Agents Rot? An Empirical Study of Long-Horizon Degradation in LLM Agents for Production Decision-Making (2026-08)
- **简介**：Shubhra Mittal。作者主张「基准成功率稳步攀升、生产环境长流程仍不可靠」这一鸿沟主要是任务 horizon 的产物：基准以短中 horizon 为主，而生产负载的依赖步数要多一个量级。为直接度量该效应，论文做了跨 9 个模型（1.2B–671B 的 6 个开源模型与 3 个已部署的闭源系统）、4 类任务（含一个真正 agentic 的工具使用回路）、5 种 horizon 与 3 种上下文条件的大规模受控研究：任务成功率遵循由单一「每步可靠性」参数支配的几何律，该参数随模型规模上升但饱和在远低于 1 处，因此足够长的 horizon 必然崩塌；在 agentic 任务上，所有被测模型（包括广泛部署的系统）都在 16 步内从近乎全对掉到近零（n=10,664 条被分析 trajectory）。退化由步数而非上下文长度驱动——限制上下文窗口反而让衰减更陡（logit 斜率 −0.69 对 −0.44，p=3×10⁻⁶），否证了 lost-in-the-middle 式解释；把实测可靠性投射到代表性基准 horizon，成功率从 GAIA 长度的 0.42 降到百步生产 horizon 的 0.24，作者据此主张用 horizon 感知的评测与可靠性预算替代聚合通过率。本文为纯实证度量研究、不提出新训练方法，属 §2.10 的基准/评测工作。
- **arXiv**：[2609.01660](https://arxiv.org/abs/2609.01660)

#### CatchBench: When Can an Agent Failure Be Caught? (2026-08)
- **简介**：Yue Zhao。围绕「agent 的失败何时能被抓住」这一审计问题，作者指出审计能力通常受限于记录而非方法，于是 **CatchBench** 把同一个审计问题放到三种信息状态下考察：运行前的声明配置（PRE）、运行中不断增长的 trace 前缀（LIVE）与完成后的完整 trace（POST），并用七份任务契约各自携带标签与指标（四份为证据型，三份为 Gold 派生的机制诊断），而不是合成单一排行榜。发布版评测 72 个参赛方法——从规则扫描器、结构化模型到覆盖九个模型族（GPT、Claude、Gemini、Gemma、Llama、Qwen、DeepSeek、Mistral、Nova）的十一个 LLM judge——覆盖 1187 份声明配置与 1162 次记录运行；118 组预先声明的对比中仅 47 组可区分，其余按「未解决」发布而不强行排名，并揭示一条忽略全部名称与权限、只标记首个能力之后所有声明能力的规则在六个配置来源之一上取得完美 F1，说明该处分数测的是语料构造方式而非方法的推理能力。该文不提出训练方法或 reward 设计，属 §2.10 的基准/评测工作。
- **arXiv**：[2608.22808](https://arxiv.org/abs/2608.22808)

#### LegacyWorld: Atomicity-Aware Evaluation of GUI Agents for Legacy Workflows (2026-08)
- **简介**：Thilo Reintjes, Sivajeet Chand, Derui Zhu, Alexander Pretschner 等（TU Munich 等）。企业遗留系统的关键 workflow 缺少可编程接口、仍需人工 GUI 操作，而领域专家指出在这类有状态 workflow 上「demo 成功」并不够：一次失败的 agent 运行可能在业务或医疗记录中留下持久的非法改动。作者因此用原子性作为评测口径——一次运行要么正确完成预期 workflow，要么失败但不留下意外的持久副作用——构建了 28 个由领域专家参与设计的 Windows GUI workflow，每个都带初始状态、目标状态与任务专属 validator，并对比专家手写 prompt 与由专家黄金路径录屏自动生成的 prompt。在六个托管 computer-use agent 上，有用完成、安全失败与非原子副作用被证明是三类彼此独立的运行画像，作者据此主张把 workflow 捕获、状态 validator 与原子性验收测试作为 AI 遗留 workflow 自动化的一等需求（abstract 未给出具体数值）。本文为纯评测研究，不含 RL 或训练成分，属基准/评测定位。
- **arXiv**：[2608.14131](https://arxiv.org/abs/2608.14131)

#### ComponentBench: Diagnosing Component-Level Failures in Computer-Use Agents (2026-08)
- **简介**：Tianchen Guan, Xinlei Lin, Royce Cheng-Yue, Shuyan Zhou 等（Duke University）。当前 computer-use agent 评测分裂为长 horizon workflow 基准与原子 GUI grounding 测试，中间那层「以组件为中心的真实交互」（如切换一组按钮）缺乏仪表化——它短到便于诊断，又足以刻画现代界面的负担。**ComponentBench** 以库无关的 97 个规范 UI 组件本体为骨架，在常用组件库上实例化 2,910 个可程序化验证的任务，并配套清洗后的人类参考 trajectory，从而同时评估任务成功率与交互效率；此外提供可扩展 pipeline 审计实现后的真实结构难度，并跨任务与组件族合成结构化失败分析。在 GPT-5.4、Gemini 3 Flash、GPT-5.4 mini、GPT-5 mini、Gemini 3.1 Flash-Lite、Qwen3-VL-235B、UI-TARS-1.5-7B 七个模型与四种观测/动作空间上，同一 harness 内仅改变观测与动作空间就能让任务成功率变动超过 30%（GPT-5 mini 从 accessibility-tree 观测的 83.1% 降到纯坐标 Pixel 控制的 48.9%），且最快配置仍需匹配人类参考的 3.7 倍时间。属纯基准/诊断工作，不提出新训练方法。
- **arXiv**：[2608.18307](https://arxiv.org/abs/2608.18307)

#### Harness the Memory: A Holistic Evaluation of Memory Substrates in Memory Agents (2026-08)
- **简介**：Wei-Chieh Huang, Weizhi Zhang, Yuchen Wu, Yankai Chen 等。记忆正成为长 horizon LLM agent 的核心基础设施，但已有评测很少回答「不同运行 regime 下应当选用哪种 memory substrate（记忆被表示与存储的底层介质）」。作者在统一 harness 下做受控评测，覆盖稠密与稀疏索引、文本记录、结构化存储、层次化存储、refinement 式记忆、参数化更新，以及与激活兼容的上下文机制，跨三个骨干模型与四个基准套件（涵盖以用户为中心的问答与以 agent 为中心的决策），仪表化 26 项性能与效率指标。结论是没有单一 substrate 全面占优：宽泛检索有利于长上下文事实型 QA，但过度检索会把注意力从行动关键上下文移开、损害序列决策；可扩展性构成另一条 routing 轴——在中等历史长度下表现好的 substrate 到更长 horizon 可能变得昂贵或脆弱。作者据此主张把 substrate routing 作为自适应 agent 记忆系统的必要组件（abstract 未给出具体数值）。属基准/评测工作，本身不提出新的训练方法。
- **arXiv**：[2608.15008](https://arxiv.org/abs/2608.15008)

#### Demystifying Agent Skills: Why They Work-Until They Don't (2026-08)
- **简介**：Zhiyuan Jiang, Fangrui Huang, Hanwen Xing, Mengdi Wang 等。skill（结构化知识包）已成为在推理期增强 LLM agent 的实用手段，但已有评测多只衡量 skill 是否提升聚合任务成功率，未回答更根本的问题：skill 何时有用、为何有用、在哪里失效。作者在多个基准、agent harness 与 LLM 上做受控实验，分离表示形式、结果标注、检索难度与跨框架鲁棒性的影响，并设计对比研究把受控定量实验与成对 trajectory 分析结合：归一化 8,135 条受控实验记录，从 240 条开放编码记录中保留 238 个有效唯一标签，凝练出 3 个高层类别、12 种 skill 使用模式。主要结论是 skill 之所以有效，在于把嘈杂 trajectory 转化为稳定执行的流程锚点——procedural anchoring 占 65.7% 的 skill 案例，而显式知识注入仅 4.5%，说明 skill 稳定的是行动而非补充缺失事实；在匹配比较下 skill 比 Workflow Memory 高 6.06 分。检索是独立瓶颈：skill 池从 5 增到 100 时实际使用精度从 29.6% 跌到 3.3%；可混淆的干扰项会损害离线识别，但下游成功率仍稳定，精确命中 ground-truth skill 既不充分也不必要；skill 在脆弱假设、上下文不兼容或适配不足时失效。属 §2.10 的分析/诊断工作，不提出新的训练方法。
- **arXiv**：[2608.14036](https://arxiv.org/abs/2608.14036)

#### CentaurBench: Benchmarking LLM Capabilities on Augmenting vs. Automating Real-World Work Tasks (2026-08)
- **简介**：Pattaraphon Kenny Wongchamcharoen, Kris Gulati, Min Min Fong, Abhishek Nagaraj（UC Berkeley）。多数 LLM 基准按「自动化工作任务」的能力给模型排名，但实际使用中模型常是在辅助另一个（人类或 LLM）agent，因此选型问题不只是哪个模型输出最好，而是哪个模型最能改进另一个（较弱）agent 的工作。**CentaurBench** 提出统一框架同时评估自动化与增强两种能力：在七个有经济学依据的真实工作任务上，助手模型为一个标准化的低能力 worker 模型撰写辅助文本、由该 worker 产出交付物；自动化模式下助手直接产出。输出由 LLM judge 面板依任务专属 rubric 做盲配对比较，重复十轮。结果显示两种 regime 下的排名只有中度相关，自动化模式的冠军在七项任务的五项上输掉增强评比；且辅助并非稳定为正——三项任务上不受辅助的 worker 排名高于所有受辅助条件，平均只有一个模型的指导优于不给指导，说明自动化能力是辅助质量的不完备代理。属 §2.10 的基准/评测工作，不含 RL 或训练成分（按仓库惯例 §2.10 收录纯 benchmark）。
- **arXiv**：[2608.18554](https://arxiv.org/abs/2608.18554)

#### Practice Makes Unsafe: Skill Misevolution in Self-Improving LLM Agents (2026-08)
- **简介**：Xutao Mao、Liangjie Zhao、Xiang Zheng、Cong Wang。自我改进的 LLM agent 会把成功 trajectory 转化为跨任务的持久状态，于是一次「不安全的成功」在触发它的输入消失后仍可能变成可复用策略；skill evolution 把操作 trajectory 蒸馏成可执行、可迁移、可检查的流程，使这一失效变得可度量，而由于进化优化的是任务结果而非流程安全，被污染的经验会导致 skill misevolution。已有 benchmark 只能测当前行为或静态产物，无法把风险跨「编写—检索—后续执行」三个环节归因。作者提出 SkillMisevo-Gym（跨 agent 框架对 skill 状态做版本化的生命周期 harness）与 SkillMisevo-Bench（从恶意暴露到 carryover 任务的冻结设计，含概念对齐的良性任务与九项生命周期指标），以及修复不安全内容并治理后续复用的 wrapper **SafeEvolve**。25 种 agent-方法配置（各含 25 个 episode、525 个任务）中，21 个进化配置全部写出了不安全产物，但只有 15 个真正造成全新会话中的危害；暴露扫描里 3 个恶意任务就把 carryover ASR 从 16.0% 推高到 35.3%，SafeEvolve 在代表性 skill 进化方法上把不安全检索与新会话危害分别降低 26.7 与 17.3 个百分点，良性效用均值仅变动 0.4 个点。
- **arXiv**：[2608.12851](https://arxiv.org/abs/2608.12851)

#### REDAgentBench: Executable Red Teaming and Faithful Measurement of LLM Agent Systems (2026-08)
- **简介**：Zixing Chen、Xingyuan Liu、Jie Zhu、Lifan Guo、Chi Zhang 等。针对现有 agent 安全评测把安全性压缩成单一 ASR、把暴露、执行、观测与判定四个环节混为一谈，可能把「证据是否可见」误当成真实违规的问题，**REDAgentBench** 提出可执行的自主红队与忠实度量框架：从显式安全约束及其关联的 agent 系统脆弱点导出攻击，在隔离的服务沙箱中真实运行，并依据服务回执与最终状态变化校验是否产生实际危害，基准共含 1661 个用例、覆盖五类服务面。在 6 个模型、3 种 agent harness 上宏平均 ASR 为 65.69%，且上报 ASR 会随 harness 与证据视图变化、评测上下文的披露本身也会改变执行行为；在以状态为依据的诊断子集中，近五分之一有明确动作锚点的确认违规发生在 agent 已经说出相关约束或风险之后，揭示出 Recognition–Execution Gap；最后，一个无需训练的 policy reminder 在配对重放中把确认违规降低超过 70 个百分点。
- **arXiv**：[2608.10669](https://arxiv.org/abs/2608.10669)

#### The Horizon Gap: Planning, Memory, Execution, Training, and Evaluation for Long-Horizon LLM Agents (2026-08)
- **简介**：Mingguang Chen、Licheng Wang、Bo Qu。前沿语言模型能在一次前向中解出数年前堪称研究成果的推理题，却在数小时量级任务上失败——忘记早前决策、把半成品宣称完成、偏离目标；作者称之为 horizon gap，系统综述 2024–2026 年 1,547 篇 arXiv 论文（系统化种子采集，公开 26.8% 的 bleed 过滤率并做定向补充）。先区分三个常被混用的属性：long-horizon（任务所需步数）、long-context（模型 token 容量）与 long-term memory（跨步/跨会话持久性）；再按长 horizon 任务生命周期分为 planning、memory、execution、training、evaluation、foundations/safety 六类，与「horizon 承载在何处」交叉组织。共同发现是 horizon 越长仅有结果的信号越无信息量，而过程奖励模型、credit assignment、trajectory 级诊断本质都在制造更稠密的步级信号。
- **arXiv**：[2608.06663](https://arxiv.org/abs/2608.06663)

## 3. OPD（Off-Policy / On-Policy Distillation / Drift）

> 行为策略与目标策略错位时如何修正、teacher 信号如何高效压缩、πθ 相对参考策略的漂移如何监控。

### 3.1 Off-Policy RL：IS / Clipping 设计

#### Anatomy of Off-Policy Policy Gradient: Importance Sampling, KL Regularization, and Baselines (2026-10)
- **简介**：作者 Haoqun Cao、Yurun Yuan、Tengyang Xie。LLM 后训练中 policy gradient 常被 off-policy 地使用，尽管朴素 policy gradient 假设 on-policy 采样。论文研究这种用法何时成立，以及 importance sampling、KL 正则与 baseline 各自的作用。结论是：一般的 off-policy 更新要满足「最优策略仍为驻点」的一致性，需要轨迹级 importance weighting（受 curse of horizon 困扰），或 KL 正则加上等于最优 soft value 的 baseline（可由 grouped mean 实现）。沿 KL 路线，作者将其对应到 group-relative policy gradient (GRPG) 更新，并证明其等价于轨迹级 Bellman residual 最小化，给出全策略覆盖下的有限样本 regret 界，为 GRPO 的一个变体提供理论解释；还揭示 baseline 的常数平移相当于 pessimism，并据此给出 asymmetric REINFORCE 目标及单策略覆盖下的 regret 界。
- **arXiv**：[2610.25725](https://arxiv.org/abs/2610.25725)

#### EPIC: Rethinking Trust Regions for Autoregressive LLM Policies via Exact-KL Prefix Importance-Ratio Clipping (2026-10)
- **简介**：作者 Chengxi Li、Hanlin Wu、Wei-Ying Ma、Jingjing Liu 等。PPO 式 token 级 clipping 仅通过当前 token 的条件概率衡量策略变化，而前缀上下文本身由策略生成，这种失配在长回复与 off-policy 训练中尤其严重。论文把自回归生成建模为有限时域 MDP，在行为策略采样下推导 performance-difference 恒等式，指出生成前缀的累积似然比才是每步策略改进项的精确重要性权重；基于 TRPO 的单调改进原则，用精确 reverse-KL 坐标 $r-1-\log r$ 构造前缀级信任域，得到 **EPIC**——一种 PPO 式序列 clipping 方法，直接约束前缀级 policy drift，无需局部二次近似或 expert-routing replay。在稀疏 MoE 语言模型的 off-policy RL 实验中，EPIC 在数学推理 benchmark 上稳定优于标准 clipping 与序列级 clipping 基线，并与基于 routing replay 的方法相当（abstract 未给出具体数值）。
- **arXiv**：[2610.29999](https://arxiv.org/abs/2610.29999)

#### Gibbs-Gap Policy Optimization for Off-Policy RL with Verifiable Rewards (2026-10)
- **简介**：作者 Xuanchang Zhang、Tong Zhang。RLVR 常在来自不同（可能已过时）behavior policy 的 response 上训练，用 learner-to-behavior importance weight 修正会在长 response 上带来高方差。论文提出 **GGPO（Gibbs-Gap Policy Optimization）**：对同一 prompt 下各 response 的 KL 调整后奖励做 log-mean-exp 对比，完全不使用 learner-to-behavior importance weight。作者从 KL 正则化策略优化推导出 GGPO，证明 on-policy 时其 population loss 恰为 KL 正则化的最优性间隙，并刻画了 reference refresh 的作用：固定 reference 的 GGPO 在五个基准的加权平均为 32.96，而默认 refresh 调度为 46.58。数学推理实验表明，GGPO 在训练-推理不匹配和固定离线数据下优于 GRPO 与 AGRO，且在策略陈旧度增大时性能保持稳定。
- **arXiv**：[2610.20814](https://arxiv.org/abs/2610.20814)

#### Trajectory-Aware Policy Optimization: Closing the State Distribution Gap in Long-Horizon LLM RL (TAPO) (2026-10)
- **简介**：作者 Zhicheng Cai、Weinan Dai、Wei-Ying Ma、Ya-Qin Zhang 等。现有 LLM RL 方法（GRPO、GSPO）只在 token 或 sequence 级别控制策略失配，忽视了长 horizon 自回归 rollout 中局部失配累积造成的状态分布偏移，导致训练不稳定甚至崩溃。**TAPO** 从 off-policy policy gradient 理论推导出状态分布修正项并纳入 RL 目标，同时用 geometric-mean estimator 降方差，并配合 length-aware dynamic clipping 避免轨迹上过度偏离的更新。在八个竞赛级基准上，TAPO 在数学推理和算法编程上相对 GRPO 最高提升 48% 和 12%，并显著提高训练稳定性。
- **arXiv**：[2610.13020](https://arxiv.org/abs/2610.13020)

#### Baselines and Score Centering in Policy Gradient Estimation (2026-10)
- **简介**：作者 Shuo Han、Ziqing Hu。LLM RL 后训练实际上是 off-policy 的（训练-推理不匹配或异步训练的策略陈旧），通常通过 importance sampling 并对比值做 truncation、masking 或 clipping 来处理，此时 importance 加权的 policy gradient 不再对奖励的常数平移保持不变。本文提出 covariance 视角：以奖励与 importance 加权 score 的协方差为估计对象，同时覆盖 sequence 级与 token 级 importance sampling，并说明 group-mean baseline（样本均值）与 score centering（总体均值）是两种方差不同的无偏协方差估计。结论：sequence 级 IS 下，已用 group-mean baseline 时 score centering 没有额外收益；token 级 IS 下，只用 group-mean baseline 而不做 score centering，在比值被正则化时一般会引入偏差。这是一篇理论笔记，不含大规模实验。
- **arXiv**：[2610.22263](https://arxiv.org/abs/2610.22263)

#### Target-ESS Tempered Policy Optimization (TETPO): Smooth Variance Control via Bridge Importance Sampling (2026-10)
- **简介**：作者 Shayan Mohajer Hamidi、Kayhan Behdin 等。异步 RL 后训练使 learner 在陈旧策略的 rollout 上更新，importance weight 重尾会使少数轨迹主导更新、有效样本量（ESS）坍缩。现有稳定器对所有情形施加固定干预（丢弃或重加权）。**TETPO** 将每个权重 rho 替换为 rho^alpha，alpha 在 [0,1] 内数据自适应：对每个 prompt 组选取使 tempered 权重满足目标 ESS 比例 kappa 的最大 alpha（一维二分搜索），健康组几乎不 tempering，权重退化严重时则加强；在 population 层面对应采样策略与学习策略之间几何桥的精确 importance weight，且没有显式零权重 mask。数学推理基准上，高陈旧度下 TETPO 优于强 on-policy 与 off-policy 基线（abstract 未给出具体数值）。
- **arXiv**：[2610.14440](https://arxiv.org/abs/2610.14440)

#### COPC: Coupled Off-Policy Correction for Asynchronous LLM Reinforcement Learning (2026-10)
- **简介**：作者 Zicheng Hu、Zhijian Zhou、Xuan Zhang 等。异步 RL 解耦 rollout 与优化以加速后训练，但会在陈旧轨迹上训练。现有方法主要靠 actor 目标中的重要性比率控制修正 token 级策略失配，论文指出这种 policy-side 修正不够：advantage 估计也继承了行为策略续写的失配，称为 advantage staleness。作者推导了双通道 actor 更新的偏差/方差分解，揭示策略权重与 advantage 估计误差的非可分耦合，据此提出 **COPC**：actor-critic 方法，结合 token 级 ratio masking 与对 TD 残差的双侧 clipped-ratio 加权来估计回报与优势。在工具集成数学推理与搜索任务上取得最高报告性能，超过各自最强异步基线，并有更宽的高性能参数区间和更稳定的训练。
- **arXiv**：[2610.09597](https://arxiv.org/abs/2610.09597)

#### On KL-Regularized Policy Optimization (KLPO) (2026-10)
- **简介**：作者 Yifan Zhang。异步 RL 中 rollout 来自陈旧 checkpoint，且推理引擎概率即使参数相同也与 trainer 不同；常规做法要么裁剪重要性比率（引入偏差），要么如 GRPO 对每个 prompt 采样一组回答（长 episode 下代价高）。**KLPO** 把 KL 正则锚定在 sampler 上，正则化改进步有闭式 Gibbs 解，通过对 sampler 自身轨迹做最小二乘拟合其 log-ratio 最优性条件，sampler 概率只以 log-ratio 形式出现，无需重要性权重；对 token 级 policy mirror descent 目标，证明梯度可仅凭终端回报无 critic 地计算，即使工具输出是随机的；还证明 KL 项的独立蒙特卡洛估计保持梯度无偏，并推导 top-K 与二元近似的精确 KL gap（abstract 被截断，实验结果以原文为准）。
- **arXiv**：[2610.08963](https://arxiv.org/abs/2610.08963)

#### Towards Better Training Signal: Advantage Clipped Policy Optimization (ACPO) (2026-09)
- **简介**：作者 Ruichuan Huang、Jinghan Liu、Congliang Chen 指出，LLM 推理 RL 对 on-policy 数据的需求限制了训练效率，通过重要性采样（IS）复用 off-policy 数据会带来不稳定，PPO/GRPO 因而普遍采用 IS 比率 clipping。但训练稳定性与梯度估计主要由 IS 比率与 advantage 的乘积决定，故提出 **ACPO**，直接对「IS 比率 × advantage」的乘积做 clip，获得更稳定的梯度估计；并建立其与 policy mirror descent（PMD）中梯度裁剪的联系，在标准 RL 设定下证明 clipped-PMD 的收敛性。数学推理基准上 ACPO 在精度与训练效率上稳定优于 PPO 和 GRPO，Qwen3-8B+PPO 在标准数学基准上有 4–6 个百分点的提升。
- **arXiv**：[2609.36816](https://arxiv.org/abs/2609.36816)

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

#### Li$^{2}$SAO: Lifting Light Single-Rollout Asynchronous RL Optimization via Group-Relative Actor-Critic Bootstrapping (2026-10)
- **简介**：作者 Zichen Bai、Yanli Wang、Yang Liu 等。异步 RL 通过随 rollout 到达就更新模型提高 LLM 后训练效率；已有工作指出 GRPO 的组采样与长尾 rollout 长度不匹配，提出 Single-rollout Asynchronous Optimization (SAO) 以改善稳定性与 off-policy 行为，但 SAO 依赖闭源的预训练 value model。论文分析了 GRPO 与自己轻量复现的 LiSAO（无预训练 value model）：GRPO 在单轮推理任务上更好，而 LiSAO 在 agentic 任务上更好，且 LiSAO 因 value model 尚未训好，早期训练明显落后。为此提出 **Li$^{2}$SAO**，用 group-relative actor-critic bootstrapping：critic 冷启动期间用组相对 advantage 训练 actor、用组中心化 return 训练 critic，之后切换到单 rollout GAE。实验显示其在早期追平或超过 GRPO，并进一步提升后期性能上限。
- **arXiv**：[2610.40726](https://arxiv.org/abs/2610.40726)

#### EnvSched: Environment-Aware Scheduling for Rollout-Efficient Agent Reinforcement Learning (2026-10)
- **简介**：作者 Sihan Yang、Weichen Bai、Ruichuan An 等。随着语言 agent 的可执行环境规模增长，生成和验证 rollout 日益昂贵。**EnvSched**（Environment-aware Scheduling）是从 rollout 历史中学习的在线调度器：exploitation 优先选择可能提供有信息量奖励差异的任务，coverage 则让每个 batch 覆盖观测不足的环境与交互模式；将环境-任务对组织为双关系图的节点（环境内边刻画任务语义及共享的工具、状态、验证器结构，跨环境边刻画抽象的执行与奖励拓扑），沿两类边传播历史 rollout 统计以预测任务内奖励方差，并辅以环境配额与相似度惩罚限制集中与冗余，无需额外 pilot rollout。在 Qwen3-8B 的 RL 中，训练 40 步后 BFCL-v3、tau-Bench、ACEBench-Agent 平均分 48.81，超过随机选任务训练 120 步所达最高分；同预算下也提升不同模型规模的下游表现。
- **arXiv**：[2610.37609](https://arxiv.org/abs/2610.37609)

#### FlashNext-RL: A System Optimization Study for Long-Horizon Agentic RL of Qwen3.8-Flash-Next (2026-10)
- **简介**：作者 Zhewei Yao、Yuxiong He。长程 agentic RL 既需要高效的 rollout 生成，也需要长序列训练，而 Qwen3.8-Flash-Next 的显存需求与模型专属算子使现有开源实现难以胜任。论文提出基于 prime-rl 的模型专属 RL 训练优化框架 **FlashNext-RL**，用 vLLM 生成 SWE rollout，重点是在单个 8xB200 节点上对完整 48 层模型做 BF16 训练，并优化 host offload、gradient accumulation 以及 MoE、稀疏注意力和门控残差流的 kernel。完整模型开发研究中，非 padding 训练吞吐从初始 32K token 配置的约 395 tokens/s/GPU 提升到优化后 131K token 配置的 4,438 tokens/s/GPU；配合三个推理节点可支持目标 batch 1,024 条可训练 rollout、131,072 token 上下文的在线 SWE RL，第 5–15 步平均每次迭代 22.4 分钟，entropy 与 trainer-sampler mismatch KL 保持有界。结果强调 RL 训练应针对模型架构、显存布局与 kernel 需求定制。
- **arXiv**：[2610.28998](https://arxiv.org/abs/2610.28998)

#### Arctic-SWE: Revisiting the Long Horizon Agent Training by Modeling-System Co-Design (2026-10)
- **简介**：作者 Boyi Liu、Yite Wang、Xinyu Lian、Fan Shu、Sida Li、Feng Yan。长程 agent（含软件工程）RL 的强基线所需的配方与基础设施难以复现，使新算法或数据的收益与欠调基线的改进难以区分。**Arctic-SWE** 是一个完全异步的 RL 框架，通过 model-system co-design 建立此类基线：以在小任务池上过拟合为测试床，整合 gated reward、importance-weight 控制、对齐的训练/推理引擎以及跨策略更新一致的状态；单个组件都不新，但合起来构成缺失的基础。在原始 R2E-Gym 数据上，Qwen3.5-4B 达到 SWE-bench Verified 62.0% pass@1，与依赖在线任务合成的 FrogNano (61.5%) 相当，rollout 数约为其六分之一；进一步清洗数据后，4B 在 Verified/Pro-v1/Pro-v2 上从 47.4%/25.0%/37.9% 升至 65.0%/38.3%/59.7%，9B 也有类似提升。
- **arXiv**：[2610.24914](https://arxiv.org/abs/2610.24914)

#### ReFlow: Replay-Driven Decoupling for Efficient RL Post-Training (2026-10)
- **简介**：作者 Yukun Tian、Junwei Su、Haofeng Xu、Lansong Diao、Zhengping Qian、Weibo Zhao 等。现有 RL 系统用异步执行重叠 rollout 与策略优化，但轻度策略陈旧带来的效率增益有限，激进异步又会损害模型质量；LLM RL 与多轮 agent 负载的资源需求随时间变化，静态资源分配与生成策略效率低。**ReFlow** 是系统-算法协同设计的框架，核心洞见是对近期轨迹做 experience replay 以复用样本，摊薄 rollout 成本又不引入过度陈旧；包含三项设计：dual replay data cycle（兼顾吞吐与优化稳定）、elastic resource pool（动态资源放置）和 adaptive scheduler（协调生成策略与资源分配）。在单轮推理与多轮 agentic RL 任务上，训练吞吐相比最先进同步与异步 RL 系统最高提升 2.06 倍，模型性能相当或更好。
- **arXiv**：[2610.21640](https://arxiv.org/abs/2610.21640)

#### ThunderSyncRL: Lossless Acceleration of Agentic Reinforcement Learning (2026-10)
- **简介**：作者 Seil Kang、Hangoo Kang、Tarun Suresh、Youngeun Kim、Shreyas Pimpalgaonkar、Seong Jae Hwang。agent 后训练需要长而异构的轨迹，同步系统让 learner 空等 rollout 和验证完成，异步训练虽可重叠但带来策略陈旧。**ThunderSyncRL** 在所需输入全部确定后立即开始梯度计算，因而没有策略陈旧：对 GRPO，每条轨迹奖励一到就计算其 score gradient，无需等待整个 group；对 OPD，在工具调用于沙箱执行期间就对已完成 agentic turn 中 teacher 打分的动作计算梯度。论文证明这种梯度流式计算与批同步训练得到相同的 GRPO/OPD 更新、不改变目标。在 SWE-bench Verified 和 Terminal Bench 4.0 上，达到同等性能最多比同步训练快 1.9 倍，固定预算下比异步训练最多高 2.47 个百分点。
- **arXiv**：[2610.05935](https://arxiv.org/abs/2610.05935)

#### RollVerify: Bridging Efficiency and Accuracy in Long-Tail Rollout Reinforcement Learning (2026-10)
- **简介**：作者 Yongqiang Yao、Jinru Tan、Kaihuan Liang、Zixin Yin、Yazhe Niu、Ruihao Gong。长上下文下 rollout 长度长尾化，同步 on-policy 训练产生 GPU 气泡；异步或 partial rollout 提升吞吐，却引入陈旧的 off-policy 样本，已有方法多靠训练时重加权，仍与完全 on-policy 存在差距。**RollVerify** 基于 partial rollout，不被动重加权，而是在样本进入训练前主动验证并修复：引入 off-policy shift 指标 OPS 量化部分生成轨迹的偏离度，在 OPS 约束下做 sequence 级与 token 级验证，截断轨迹中无效的后缀，既保护精度又保留 partial rollout 的效率收益。在数学及工具辅助数学推理上，精度可与 on-policy 训练相当，同时降低训练开销（abstract 后半段被截断，具体加速比以原文为准）。
- **arXiv**：[2610.09914](https://arxiv.org/abs/2610.09914)

#### NeMo-DCR: Bit-Exact Delta-Compressed Refit for Scalable Agentic RL at Trillion-Parameter Scale (2026-10)
- **简介**：作者 Songlin Jiang、Zhiyu Li、Terry Kong 等（代码已合入 NVIDIA NeMo RL）。Agentic RL 将训练与 rollout 拆分到不同集群，每次策略更新都要传到 rollout 集群：1T checkpoint 在两个 AWS 区域间传输需 87.5 分钟，而每步仅有 1-2% 的 BF16 元素发生变化。**NeMo-DCR**（Delta-Compressed Refit）只发送变化量，却让接收端得到与稠密 refit 逐位相同的结果：在双方共享的 Hugging Face 规范坐标中描述变化，用可压缩的 XOR mask 编码（XOR 不适用处用覆盖写），经对象存储或 relay tree 流式传输，原地更新并支持失败后重试与联合提交。真实更新下比 AReaL-DTE 快 7 倍；30B-1T 的 refit 比仅传输全量 checkpoint 的参考快 12-40 倍，可使 1T 规模的同步 Kimi K2 RL 单步快 1.4-1.5 倍。
- **arXiv**：[2610.08430](https://arxiv.org/abs/2610.08430)

#### Asynchrony Is Nearly Free for Evolution Strategies on Long-Horizon LLM Agentic Tasks (2026-10)
- **简介**：作者 William Hoy、Jingxuan Fan、Nurcin Celik、Xu Pan。长程 agent 后训练中同步更新会让快 worker 空等；异步 RL 虽解决空转，却带来策略滞后和 off-policy 优化。进化策略（ES）是无需反向传播的替代方案，但在 LLM 后训练中一直基本是同步的。论文提出有界陈旧度的异步 ES，并在 WebShop 上系统变化滞后程度与陈旧 fitness 测量的比例，另设两个压力测试覆盖极端陈旧情形。结果显示自然异步下表现紧贴同步 ES，即使极端陈旧（包括全部 fitness 都陈旧）评测准确率下降也很小，说明 ES 无需显式 off-policy 修正即可容忍显著的策略陈旧，适用于长程 LLM agent 的异步后训练。
- **arXiv**：[2610.04196](https://arxiv.org/abs/2610.04196)

#### Probe the Harness: Setup Checks for Stale-Data RL Comparisons in Language Models (PTH) (2026-10)
- **简介**：作者 Taiheng Pan。在陈旧样本上训练语言模型的方法通常与重要性校正基线对比，论文表明实验 harness 的细节可以反转方法排名，并提出检查清单 **PTH**（Probe The Harness）。案例是 SAN（不依赖行为策略）与 truncated importance sampling（TIS）在 verl 与单卡训练器中的对比，SAN 起初在两个栈中都领先；四个 harness 细节改变了结论：PPO ratio 对 learner 自己重算的概率取值、data seed 未传入 TIS 分支、replay queue 在 33 次更新中复用首批数据、两个 loss normaliser 与描述不符。检查 harness 后，TIS 在 verl 上与 SAN 持平，在 trainer 中 TIS 稳定学习而 SAN 仍保持一定优势。论文还给出 TIS 与未校正 GRPO 在 sampler lag 下的参考结果。
- **arXiv**：[2610.02911](https://arxiv.org/abs/2610.02911)

#### Asynchronous LLM Post-Training: Group-Mass Capping and Convergence Analysis (2026-10)
- **简介**：作者 Qijia He、Ruinan Jin、Jun Luo、Shaofeng Zou、Yingbin Liang 为异步 RL 后训练中的陈旧 rollout 影响提供理论分析。针对 GRPO 类算法推导出收敛界，刻画梯度估计量二阶矩与偏差的权衡：对轨迹级重要性加权估计量，一旦二阶矩被统一控制，延迟便通过 clipping 或 rescaling 引入的偏差进入收敛界。据此提出 **GMC-GRPO**（group mass capping GRPO），在共享二阶矩保证的加权估计量类中最小化基于比率的偏差界。理论上，相比 TIC-GRPO，延迟的四阶项对阈值的依赖从 $O(\epsilon^{-4})$ 改善到 $O(\epsilon^{-2})$。在 Qwen3 模型与推理基准上，GMC-GRPO 对陈旧 rollout 更稳健，在大延迟下于稳定基线中表现最佳。
- **arXiv**：[2610.01896](https://arxiv.org/abs/2610.01896)

#### Where Does Staleness Accumulate? Pool Aware Effective Staleness Control for Asynchronous RL in LLM Post-Training (2026-09)
- **简介**：作者 Chenliang Li、Neiwen Ling、Zijun Wei、Alfredo Garcia 研究全异步 RL 后训练中的 policy lag。将轨迹 staleness 分解为 rollout 完成前累积的 Generation Staleness 与完成后在池中累积的 Waiting Staleness，并提出 **PACE**（Pool-Aware Control of Effective Staleness）：把池占用过量转化为自适应拒绝预算，并用结合 Waiting Staleness 与 prefix-aware Generation Staleness 的有效 staleness 分数对完成的轨迹排序，避免仅因长或被打断、跨多个策略版本的 rollout 被惩罚。在单轮数学推理中，相同 wall-clock 预算下六个基准平均验证精度较未过滤异步 RL 提升 18.7%，并以少 47.1% 的 GPU 时间匹配同步 RL；在多轮工具集成推理、MoE 模型及其他 RL 算法上同样有效。
- **arXiv**：[2609.36830](https://arxiv.org/abs/2609.36830)

#### PEARL: Adaptive Prefill-Decode Execution with Elasticity for Agentic Reinforcement Learning (2026-09)
- **简介**：多轮 rollout 主导了 agentic RL 的成本。论文作者（Jiaan Zhu、Wei Gao、Youhui Bai 等）发现，除异步执行与弹性 GPU 资源外，prefill–decode（PD）配置（colocation 与 disaggregation 的选择及 PD 比例）随负载变化，使资源扩缩与 PD 配置相互依赖。提出异步 agentic RL 系统 **PEARL**，协同外部资源弹性、闲置训练 GPU 的临时复用与自适应 PD 执行：维护统一的 GPU–worker–role 状态，用运行时 profile 预测 rollout batch 完成时间，在当前 GPU 预算下选择 PD 模式与比例，并生成最小化 worker 与角色变动的增量切换计划，配合成本感知的切换/借用策略。评测显示吞吐为固定资源 ROLL 的 2.17–2.79 倍；相比 RLBoost+，Qwen3-8B 提升约 26.9%，Qwen3-30B-A3B 提升约 36.3%。
- **arXiv**：[2609.35158](https://arxiv.org/abs/2609.35158)

#### FlashREINFORCE: Critic-Free Single-Rollout Asynchronous RL for Agentic Language Models (2026-09)
- **简介**：作者 Jian Hu、Yifan Zhang、Hao Zhang 等（代码位于 NVIDIA-NeMo）。异步 RL 适合长程 agent，但组相对更新需要对每个 prompt 重复 rollout，带来 prompt 覆盖受限与组同步屏障。**FlashREINFORCE** 是首个可扩展到长时间异步训练和大模型的 critic-free 单 rollout 框架，结合 One-Batch REINFORCE、Sequence Trust Region 与 Sample-Mean Optimization，在控制 policy drift 与长度放大的负更新的同时提供批均值基线。在 policy lag 为 4 时 DeepSeek-R1-Distill-Qwen-1.5B 稳定训练 6000 步，Qwen3-30B-A3B（lag 8）稳定 1450 步；同等条件下与强化后的组相对基线差距在 0.5 分内，用一半 rollout 超过已发布同步基线（38.0 对 36.3），并在 ALFWorld 达 98.3%/96.5%（seen/unseen）。
- **arXiv**：[2611.18627](https://arxiv.org/abs/2611.18627)

#### WeightBridge: An Efficient Weight Transfer Library for Reinforcement Learning (2026-09)
- **简介**：作者 Xuanlin Jiang、Samuel Hsia、Michael Kuchnik 等。LLM RL 系统中，把更新后的参数从 trainer 传到 rollout generator 的权重传输已成为性能瓶颈，难点在于支持多样的 trainer/rollout 布局与同步需求。**WeightBridge** 自动提取 trainer 与 rollout 权重布局的对应关系，规划并执行无冗余、负载均衡的权重传输，并以小而通用的 API 协调多种同步模式。在不同模型、并行布局和同步模式的配置下，相比最先进的开源 RL 框架，平均 GPU 停顿时间最多降低 42 倍；一个 coding agent 无需人工指导即可把它集成进两个不同的 RL 框架。
- **arXiv**：[2609.25442](https://arxiv.org/abs/2609.25442)

#### GrowMTP: Can RL Grow Its Own Draft Head? (2026-09)
- **简介**：作者 Minghua He、Lingzhe Zhang 等。RL 后训练的 wall-clock 主要消耗在自回归 rollout 生成上，speculative decoding 是常见加速手段，但现有 draft head 需要在 RL 之前预训练或 warm-up，带来额外成本。作者观察到 RL 本身提供了在线训练 draft head 的两个条件：rollout 分布比预训练窄得多，且 verification 步骤持续产出与该分布对齐的监督信号。**GrowMTP** 据此在 RL 循环内从零训练 draft head，head 的所有更新与策略 backbone 解耦。在 Qwen3-4B（无 draft head）、MiMo-7B-SFT（弱 head）和 Qwen3.5-4B-Base（强 head）上，rollout 加速分别为 2.13x、1.93x、1.36x，端到端加速分别为 1.60x、1.41x、1.20x。
- **arXiv**：[2609.16648](https://arxiv.org/abs/2609.16648)

#### BRACE: Anchored Bellman-Residual Correction for Stale Critics in Asynchronous RL (2026-09)
- **简介**：Guanqun Zhao、Zijun Xie、Binbin Zheng、Jiafeng Lu 等针对异步 RL 训练中被忽视的 critic 侧偏差问题：policy lag 使 value model 偏向陈旧的 behavior policy（stale critic），而已有异步 LLM 训练工作只修正 actor；经典 RL 的 off-policy value correction 又无法迁移到长程 agentic 任务——修正 horizon 太短会让回归目标里不含 reward，太长则 importance ratio 的乘积随轨迹长度指数 drift。提出 **BRACE**，把 Bellman residual 的修正 horizon 限制在 policy token 的一个前缀上，并在其之后锚定一个常权重的 Monte-Carlo 尾项，从而把 policy correction 与 reward propagation 解耦。在 BrowseComp-Plus 上 mean@1 相对最强基线提升 $2.4\%$，每步训练比同步训练快 $2.46\times$，并且在 off-policy 程度达 $50$ 次更新时仍保持稳定。
- **arXiv**：[2609.09783](https://arxiv.org/abs/2609.09783)

#### Headroom-Drift Replay: A Primitive for Principled Replay Control in GRPO (2026-09)
- **简介**：Hyun Bin Park、Du-Seong Chang 针对推理模型 RL 后训练被「反复生成新鲜 rollout」卡住吞吐的问题（尤其 agentic 场景下环境交互主导 wall-clock 成本），指出已有 replay 方法都被嵌进含探索、经验重构或混合策略优化的大流水线中，难以剥离 replay 本身的贡献。本文只问一个聚焦问题：单靠有原则的 replay 选择能走多远？提出 **Headroom-Drift Replay**，一个 GRPO 上的 group 级 replay 控制原语，把复用拆成两个决策：Headroom 按「剩余可学价值」对已存 group 排序，Drift 则按与当前 policy 的兼容性做门控；新鲜 on-policy 流不做任何改动，也不引入额外生成或训练机构。在数学推理、多模态推理与 Agentic Search 基准上，这一单点干预在 Avg Mean@32 上优于朴素 replay，并追平或超过更复杂的 replay 方法；在环境交互主导成本的 Agentic Search 上以明显更低的 wall-clock 时间达到可比质量（abstract 未给出具体数值）。
- **arXiv**：[2609.03941](https://arxiv.org/abs/2609.03941)

#### AInfer-PD: Communication-Safe In-Place Prefill-Decode Multiplexing for Distributed MoE Rollouts (2026-09)
- **简介**：Guowei Wang、Chaokun Yang、Zhenxuan Pan 等指出 rollout 推理常主导大规模 RL 的 wall-clock 时间：agentic RL 中每条轨迹在模型生成与环境交互间多轮交替，异步轨迹会在其他轨迹仍处 decode 阶段时不断引入新的 prefill 工作，使 P/D 共存成为 rollout 的持续性质而非一次性 prompt 摄入事件；共享加速器上这种持续共存会让 prefill 干扰延迟敏感的 decode。P/D 分离虽能避免共置但需独立设备池与 KV-cache 传输，而已有 in-place 复用设计缺乏大规模 MoE 部署（attention TP/DP 叠加分布式专家执行）所需的通信隔离——实践中 P 与 D 会以跨 rank 不一致的顺序发起相交的集合通信，DeepEP 的 P/D 路径还共享可变协议状态。**AInfer-PD** 把 in-place P/D 复用扩展到分布式 MoE rollout：跨 rank 协调 P/D 集合通信顺序，并给两条 DeepEP 路径独立的通信状态，使交叉的 ADP/ATP 与 DeepEP 路径可安全并发执行 P/D，同时保留共享权重与 KV 存储。单节点 prefill 密集负载下，固定负载 rollout 完成时间相对关闭 P/D 复用的同款 AInfer 引擎降低 7.1–22.5%、相对 SGLang 降低 24.8–32.9%；双节点分别为 18.0–35.3% 与 18.3–31.8%；同引擎消融中细粒度边界比整 epoch 异步入队再降 8.6–19.8%。本文属系统 / 基础设施侧工作，不改变 off-policy 优化目标本身。
- **arXiv**：[2609.00993](https://arxiv.org/abs/2609.00993)

#### SPO++: Stream-Aligned Policy Optimization for Asynchronous Agentic RL (2026-08)
- **简介**：作者 Kai Ruan、Jinghao Lin、Qianshan Wei、Ziqi Zhou 等针对 group-relative RL 必须等待同一 prompt 的 sibling rollout 完成、在长且长度高度可变的 tool-use 轨迹上代价高昂的问题展开。Single-stream Policy Optimization（SPO）用一个持久的 prompt-level value estimate 去掉了这一同步依赖，但其配方是先对每条轨迹 whiten 出一个 advantage，再优化 token-mean 的 actor loss；本文指出 trajectory centering 一般并不能 center actor 实际消耗的 token 加权量，于是提出 **SPO++**，改为在 action-token measure 下对终局 outcome advantage 做标准化以消除这一 mismatch，并按产生证据的 policy event（而非 learner 的接收顺序）来组织 prompt 证据。在 ALFWorld 两个模型规模与 Math-TIR 的对齐实验中，SPO++ 的在线学习效率优于 SPO（abstract 未给出具体数值），paired ablation 表明 action-token-measure 归一化是所测组件中贡献最强的一项。
- **arXiv**：[2608.24870](https://arxiv.org/abs/2608.24870)

#### TailSieve: Partial-Rollout-Guided Tail Routing for LLM Rollouts (2026-08)
- **简介**：作者 Tianqi Xu、Lu Lv、Haoyang Huang、Wenjie Huang 等关注 RL 后训练、on-policy distillation（OPD）与重采样评测共同依赖的大规模 rollout：与按 request 级延迟/吞吐优化的在线服务不同，少量长尾生成会主导整个 rollout step 的 makespan，而实践中 rollout 请求常被均匀路由到各 replica，把超长生成塞进高并发 decoding batch。**TailSieve** 联合调控 tail routing 与 replica 分配：在已知生成长度的理想设定下证明长尾情形下 makespan-最优路由 = tail isolation + load balancing，且简单 top-k 策略可逼近该离线最优；再利用「长尾 prompt 在 policy 更新间仍倾向于长尾」这一观察，用 partial rollout 作为免训练信号识别候选 tail group，并由层级控制器依据 response-work 历史与实测的并发-吞吐模型在线调整隔离组数与 tail/bulk 两池的 replica 划分。仅路由改动即取得最高 1.67x 加速，低并发 tail 池进一步支持路由特化的 speculative decoding（MTP 或 DFlash），相对均匀路由最高 2.59x；被选中的 prompt 在当前 policy 下重新生成，从而保持 on-policy 生成并在稳态下避免路由引入的长度偏差。
- **arXiv**：[2608.22788](https://arxiv.org/abs/2608.22788)

#### Rollplex: Cross-Phase GPU Spatial Sharing for Vision Language Model Post-Training (2026-08)
- **简介**：作者 Hanfeng Lu、Tianyu Feng、Suyi Li、Yuheng Zhao 等针对 VLM 的 RL 后训练 runtime 提出跨阶段 GPU 空间共享。现有 on-policy RL runtime 把 rollout、reference scoring、actor training 严格串行执行，而 VLM 中稠密视频输入与 prompt prefix 的处理占据每个阶段的很大比例；由于 prefix 计算与生成的 response 无关，可以搬进 rollout decode 窗口并行执行，且不破坏同步 on-policy 语义。**Rollplex** 拆解 reference 与 training 阶段并将 prefix 计算移入 decode 窗口，为此引入两个机制：phase-aware memory management 按生产者-消费者生命周期控制 HBM 驻留（朴素 colocate Qwen2.5-VL-32B 需约 165 GiB/GPU），以及 parallelism-aware weight sharing 让不同 tensor-parallel 度下 layout 兼容的张量复用同一物理存储、仅重建不兼容张量，从而避免完整的第二份 actor 副本。在 32 卡 H800 上，同等 GPU 预算下相对串行 colocation 加速 1.23×–1.30×，相对 disaggregation 加速 1.57×–2.24×，同时保持同步 RL 更新。
- **arXiv**：[2608.14498](https://arxiv.org/abs/2608.14498)

#### Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Requirements (2026-08)
- **简介**：作者 Zhi Zheng、Rongsheng Chen、Yunpeng Ba、Zhenkun Wang 等主张用 evolution strategies（ES）替代 RL 来微调 long-horizon LLM agent：长程 agentic 推理带来分支交互与稀疏 reward，使基于反向传播的 RL 训练栈难以扩展到大模型，且长轨迹上的 credit assignment 更难。作者论证 ES 的三点优势——只需 inference 级显存即可做全参数优化、黑盒反馈接口易与 prompt 空间进化（技能优化、test-time compute）组合、以及在轨迹级做参数归因而无需沿 horizon 分解 reward。**Agentic ESOpt** 每步在当前参数附近采样扰动、用 reward 评估对应 agent 并做在线 reward-weighted 更新，并对扰动尺度 $σ$ 施加 cosine decay 调度以平衡探索与适应。在 WebArena-Lite 上对 Qwen-3.5-27B 做全参数优化，相对 No Skill 基线提升 6.69%；在 test-time 自动启发式设计中做 prompt-参数在线协同进化，在 36 个设置中的 28 个超过对应基线。
- **arXiv**：[2608.17310](https://arxiv.org/abs/2608.17310)

#### Scheduling Mixed RL Rollouts Beyond Prefix Locality (MISA-T) (2026-08)
- **简介**：作者 Zetao Hong、Song Yuan、Yuanhao Ding、Yibo Zhu 等针对现代 LLM RL post-training 中「多域混合 rollout」的服务调度问题：当 RLVR、RLHF 与 agentic rollout 共享同一个异步推理服务时，三者的序列结构、交互模式与 KV-residency 时间差异巨大，而 prefix-aware routing 只优化 cache 复用与负载均衡，并不管异构 rollout session 如何争抢 KV-cache 容量，还可能扭曲 trainer 指定的 workload 混合比例。提出 **MISA-T**，一个位于 routing 层的准入策略，组合自适应 session admission、workload-aware 的 KV 容量分配、以及 residency-time-aware 的 KV 记账。在 Step3.7 与 Qwen3.6-35B-A3B 的 rollout-only ablation 上，相对 sweep 调优过的 cache-aware vLLM Router 分别提升 rollout 吞吐 53.3% 与 43.6%，同时保持高 prefix-cache 命中率；在 50 轮迭代的对齐实验中提升 rollout 吞吐 35.6%、平均迭代时间下降 22.8%，且消耗的 workload 混合比例贴近 trainer 目标、任务分数可比。属 §3.2 的系统级 off-policy 基础设施代表。
- **arXiv**：[2608.11152](https://arxiv.org/abs/2608.11152)

#### TideRL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling (2026-08)
- **简介**：作者 Yanyu Ren、Xizheng Wang、Xiao Liu、Bowen Lv 等（清华大学）指出多轮 agentic RL 的 rollout 任务会反复因外部环境暂停、带着不断增长的 context 恢复、并在高度不齐的时刻结束，此时真正重要的是训练 goodput 而非 GPU 占用率，GPU 空等与重复 prefill 重算都是纯开销。提出 **TideRL**，一个 readiness-aware 的弹性 RL 系统，含三个组件：Continuous Task Batching 保留有用的 rollout 状态、Resource-Aware Ref-Actor Pipelining（RA²P）依据 ready backlog 与到达间隔在 decoupled streaming 与 colocated aggregation 之间切换、Elastic Resource Scaling 用同一 readiness 信号在 rollout 与 training 之间迁移 rank。在纯文本与多模态 agentic workload 上，相对同步基线提升 RL 训练 goodput 最高 5.6×、相对异步基线提升超 33%，任务性能相当；KV cache 命中率提升 1.58×，单步训练时间最多降 44.3%，总等待时间最多削减 77.6%。
- **arXiv**：[2608.10402](https://arxiv.org/abs/2608.10402)

#### Beyond On-Policy Exploration: Integrating External Policy Rollouts for Reinforcement Learning in Diffusion Language Models (ERILS) (2026-08)
- **简介**：作者 Wonseok Lee、Jimyeong Kim、Jungmin Ko、Wonjong Rhee 指出 diffusion LLM（dLLM）的 RL 普遍只依赖目标模型自身的 on-policy rollout，当成功 rollout 稀缺时训练几乎收不到正 reward、进展极为有限。作者转而把更强外部策略生成的高 reward rollout 与 on-policy rollout 混用，并针对由此产生的两个实际障碍——rollout 长度差异、以及联合处理两类来源 reward 时的不稳定——提出 **ERILS**（External Rollout Integration with Length Control and Source-Specific Processing）：对外部 rollout 施加长度控制，并对 on-policy 与外部 rollout 的 reward 分别做 source-specific 处理。Sudoku、Countdown、MATH500 的 zero-shot 评测中 multi-sample 表现全面提升，Sudoku 增益最大：best-of-4 完成准确率 98.4%，而最强基线仅 40.3%；在 128、256、512 token 三种生成长度下确定性单次解码准确率均维持约 90%。组件分析显示长度受控的外部 rollout 优于不受控版本，source-specific reward 处理则避免了联合处理时出现的训练崩塌。
- **arXiv**：[2608.01717](https://arxiv.org/abs/2608.01717)

#### SpecRoll: Fast-Slow Verifier-Feedback Adaptation for Speculative Reinforcement Learning Rollouts (SpecRoll) (2026-08)
- **简介**：作者 Nhat Minh Pham、Duy Tung Doan、Khac-Hoai Nam Bui 针对 RL 后训练中自回归 rollout 生成的效率瓶颈：speculative decoding 本可加速，但 RL 里 target policy 持续演化，静态 proposer 会迅速陈旧（stale），频繁更新 drafter 又带来大量开销。**SpecRoll** 在两个时间尺度上自适应：轻量 future-token heads 并行给出提案，Reflex 模块用延迟的 verifier feedback 做有界、轨迹局部的 hidden-state 修正且完全不需反向传播；只有检测到持续退化时才走慢路径更新 head 参数。再配合 concurrency-aware 稀疏树验证与精确 target 验证，使 target rollout 分布与 GRPO 目标保持不变。在 1.5B–14B 的五个模型、三个数学推理数据集上取得 1.26–2.15x 生成加速与 1.21–2.04x 端到端加速，并在全部 15 组匹配设定上生成与端到端时间均优于 FastGRPO，平均配对端到端增益 1.18x。属 §3.2 的 rollout 加速基础设施代表（严格说它不引入 off-policy 偏差，而是在保持 on-policy 分布前提下提速）。
- **arXiv**：[2608.04962](https://arxiv.org/abs/2608.04962)

#### Stale but Stable: Staleness-Adaptive Trust Regions for Stabilizing Asynchronous Reinforcement Learning (SAT) (2026-07)
- **简介**：Junyao Yang 等提出 Staleness-Adaptive Trust Region（SAT）。指出异步 RL 中 staleness 由 policy lag、engine delay、MoE routing 共同放大；从信任域视角看，training-inference divergence 主导有限视界近似误差，而 PPO clipping 只是"采样代理"、对高 staleness 更新控制不足。SAT 以脱梯度采样 log-ratio 作为 staleness 代理，用基于 staleness 的核缩放定位每 batch 的高失配尾部，仅收缩符号选定的 PPO 区间端点；证明了局部区间包含性与相对 PPO 的逐点悲观性。在 Qwen3-30B-A3B-Base（SGLang 推理 + Megatron 训练）解耦异步设置下，SAT-GSPO w/ R3 取得最佳 AIME24 avg@8（lag 1 达 35.83、lag 8 达 34.79）；自适应 clipping 与 routing replay 分别针对失配尾部与路由不一致互补稳定。
- **arXiv**：[2607.18722](https://arxiv.org/abs/2607.18722)

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

#### Aligning Training and Rollout Policies for End-to-End FP4 RL on Mixture-of-Experts LLMs (MoEResQ) (2026-11)
- **简介**：作者 Hei Yi Mak、Hoang Le、Mehran Taghian Jazi 等。研究 MoE 大模型的端到端 FP4 RL 后训练：针对稠密模型的 FP4 RL 恢复配方在 MoE 上与朴素 FP4 基线相比毫无改进。作者深入分析发现根因在 MoE 独有的路由机制——router 做离散专家选择，使训练-推理策略对齐对「恢复技术是否在训练侧与推理侧对称施加」极为敏感；非对称配方（包括只修正 rollout 侧的稠密模型方法）会使 rollout-训练-推理对齐急剧恶化，表现为训练与推理概率的 Pearson 相关系数以及 router 选择匹配率下降。据此提出 **MoEResQ**：强制训练侧与推理侧修正对称的 MoE RL 量化恢复配方，恢复训练-推理对齐并显著缩小与全精度的准确率差距，确立对称恢复是 MoE FP4 RL 的关键原则（abstract 未给出具体数值）。
- **arXiv**：[2611.04376](https://arxiv.org/abs/2611.04376)

#### End-to-End MXFP8 Reinforcement Learning on Blackwell: What Matching Buys, When It Pays Off, and How It Fails (2026-10)
- **简介**：作者 Wengee Zhang。低精度算术可降低 LLM RL 成本，但 RL 要求 sampler 与 learner 一致。论文在 verl 中实现端到端 MXFP8 RL（Megatron/Transformer Engine 训练；vLLM 与 SGLang rollout；稠密与 MoE），在 B200 GPU 上用 Qwen3-8B 与 Qwen3-30B-A3B 测量其收益、加速条件与失败方式。(1) 匹配：对 MXFP8 learner，位于 learner 网格上的 sampler 比更高精度的 BF16 sampler 更一致，train-rollout KL 在一半线性层对齐时低 11%、全部对齐时低 16%；在 MoE 上，sampler 使用另一网格的 FP8 pipeline 从第一步起 KL 是共享网格的两倍。(2) 速度：稠密场景仅当每 micro-batch 约 4.5k token 以上才划算；端到端使 MoE 的 step 时间降 10.7%、生成时间降 24%。(3) 正确性：给出权重同步必须保持的七个条件及检查方法。所有运行单 seed、至多 60 步，不对最终模型质量作断言。
- **arXiv**：[2610.34915](https://arxiv.org/abs/2610.34915)

#### bf16 Rounding Silently Discards Most RL Policy Updates: A Hidden Training-Inference Mismatch and an Unbiased One-Line Fix (Fair Cast) (2026-10)
- **简介**：作者 Andrew Gao、William Gao。RL 后训练用 fp32 master 权重做优化，而 rollout 由 round-to-nearest 转换出的 bf16 副本服务。在常用学习率下大多数单步更新小于半个 bf16 间距而被舍入丢弃：Qwen3-1.7B 上每步仅 0.4-0.5% 的服务权重发生变化，生成训练数据的策略落后优化器 50-150 步；现有 mismatch 修正比较的是同一次舍入之后的两个副本，无法看见这一滞后。论文提出 **Fair Cast**：用无偏舍入规则（stochastic rounding 或 error feedback）导出 master 权重，并在服务副本上计算训练损失；改动仅一行、无新超参，每次导出 1.7B 参数耗时 22-38 ms。在 Qwen3-0.6B 至 4B、AdamW 与 Muon 的 25 对匹配 seed 中，Fair Cast 领先 18 对、持平 6 对、落后 1 对，平均提升 6.1 点；在 zebra puzzles 上达到 0.57±0.16，而 round-to-nearest 塌缩到 0.33±0.10。
- **arXiv**：[2610.41073](https://arxiv.org/abs/2610.41073)

#### On the Origin and Evolution of Training-Inference Mismatch in LLM Reinforcement Learning (DLC) (2026-10)
- **简介**：作者 Yu Li、Yufa Zhou、Tian Lan。on-policy RL 假设训练回复来自当前策略，但实际中推理引擎采样、训练器另行计算梯度，二者的 training-inference mismatch 使数据 off-policy；其来源有两类——引擎数值差异与异步采样的陈旧权重，针对一种来源有效的修正在另一种下可能失效。论文把在 sampler 权重处评估 trainer 得到的每 token 误差精确拆分为引擎的 numerical discrepancy 和 trainer 的 policy lag：数值差异近似恒定，对每个前缀做 score 中心化可去除其主要偏置；policy lag 则由 trainer 自身更新产生，形成「偏置进入下次更新、使下次滞后更大」的反馈环，重要性加权只在 lag ratio 集中时能切断该环，在 Qwen3-1.7B 上于 ratio 有效样本量低于一半时失效。据此提出 **DLC**（discrepancy-lag correction）：对数值差异中心化、对 lag ratio 相对 sampler 权重裁剪，并在有效样本量低于一半时重新同步 sampler。在陈旧 sampler 下，DLC 使 Qwen3-0.6B/1.7B 的 KL 保持在两次独立 on-policy 训练间 KL 的两倍以内，其重同步还能防止逐 token 修正无法避免的塌缩。
- **arXiv**：[2610.40945](https://arxiv.org/abs/2610.40945)

#### Where Rounding Happens Matters: Rounding-Location Alignment Stabilizes LLM Reinforcement Learning (2026-10)
- **简介**：作者 Yijing Ke、Tianle Cai、Ruosong Wang。LLM RL 常使用分离的推理与训练系统，即使权重与 BF16 张量相同，实现差异也可能让本应 on-policy 的更新变成 off-policy 并破坏训练稳定性；现有补救是对齐整个推理与训练栈、改全局精度或修改学习算法。论文探究：仅对齐执行栈的部分位置能否稳定 BF16 RL。作者在 RoPE、SwiGLU 和 residual RMSNorm 三个区域，把训练侧的舍入位置与推理对齐——让选定的中间量保持 FP32 直到其 BF16 输出。在四个 Qwen 系列模型上，三个区域全部对齐可避免默认 BF16 基线出现的塌缩；在所研究的运行中仅对齐 RoPE 路径也足够，尽管仍存在其他数值差异。该干预不改动推理引擎、矩阵乘法与 attention kernel 以及学习目标。
- **arXiv**：[2610.32951](https://arxiv.org/abs/2610.32951)

#### InterQ-RL: Interaction-Aware Correction of Execution Mismatch in Quantized Agentic RL (2026-10)
- **简介**：作者 Yingbo Hao、Song Liu、Guanghua Yu 等。多轮 agent RL 依赖长闭环 rollout，生成与 KV-cache 成本是主要瓶颈；量化 rollout 降低成本，但低精度执行器同时是行为策略，会与 learner 产生执行失配。现有量化 RL 方法主要面向单轮生成，只对齐 learner 前向或修正当前 token 的 rollout-learner ratio，没有建模产生每个 agent 状态的策略历史，因此即使量化对齐后，残余失配仍沿到达状态的策略历史累积。**InterQ-RL**（Interaction-aware Quantized RL）对到达状态的策略历史做有界的 execution-prefix 修正，在固定位移预算下按有效支撑度在动作片段间分配修正强度，且无需额外前向。在 4B-30B 稠密与 MoE 骨干、ALFWorld/WebShop/ScienceWorld 上（rollout 统一用 W8A8-KV8），InterQ-RL 取得量化设置中最强的终点结果，与 BF16 相差在 0.7 点内，并在两个设置中超过 BF16，且学习速度不逊于仅做对齐、避免其后期退化。
- **arXiv**：[2610.35303](https://arxiv.org/abs/2610.35303)

#### TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models (2026-10)
- **简介**：作者 Xin Wang、Hao Yu、Zhengyang Zhuge、Bochao Mao、Zheng Li、Junda Feng。低精度 rollout 能降低 RL 后训练的计算与显存开销，但现有 FP4 RL 方法分别优化训练和 rollout 路径的量化精度，没有直接缩小两条量化执行路径间的差异。**TRACE**（Train-Rollout Quantization Alignment via Compact GuidancE）面向 MoE 语言模型：用 rollout 侧的量化结果指导训练侧 FP4 舍入决策（rollout-guided QAT），直接缩小 train-rollout 差异；并设计量化信息缓存方案，仅保留较深层的 mantissa 与 scale 信息以降低存储与通信。在四个大型 MoE 模型的推理、代码、长程 RL 任务上，实现 FP4 权重/激活加 FP4 KV-cache 的 rollout，RL 表现与 BF16 rollout 相当，rollout 最高加速 5.4 倍。
- **arXiv**：[2610.07767](https://arxiv.org/abs/2610.07767)

#### TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning (2026-10)
- **简介**：作者 Zhen Li、Shuai Zhang、Yanggan Gu、Yiming Zhang、Yang Yu、Mingfa Feng。低精度执行加速 LLM RL，但 learner 与 sampler 的执行差异会使策略优化失稳。论文刻画 mismatch 与策略梯度方向的相互作用，区分局部放大与收缩的更新贡献；在原生 NVFP4 运行中观察到两类放大区域早期失衡，偏向负优势、负 gap 的更新，其尾部 token 在 mismatch 扩散到全局之前集中于少数 response 片段。据此提出 **TRIAGE**：用 segment 级诊断有选择地重平衡策略梯度更新，并对残余严重失配做有界修复，同时保持 sampler 和 learner 均为原生 NVFP4 W4A4 前向。在 Qwen3-4B 和 Qwen3-30B-A3B 上训练稳定，五个数学推理 benchmark 达到全精度水平，rollout 吞吐最高为 BF16 的 2.3 倍。
- **arXiv**：[2610.07043](https://arxiv.org/abs/2610.07043)

#### miniTE: Diagnosing and Bridging the Execution Gap in Native NVFP4 Reinforcement Learning (2026-10)
- **简介**：作者 Zhen Li、Mingfa Feng、Yiming Zhang、Yang Yu、Changhai Zhou、Pengxiang Li。原生 NVFP4 有望加速 LLM RL，但在独立的训练与采样引擎间稳定学习仍具挑战。论文表明仅对齐权重与数值格式不足以保证一致的策略评估，跨引擎数值执行一致性很关键。**miniTE** 是轻量框架，结合执行诊断、跨引擎数值对齐和高效的原生 W4A4 GEMM 计算，在不修改 RL 目标的前提下提高数值执行一致性。在 Qwen3-30B-A3B 的长程 RL 实验中训练稳定、性能接近 BF16，评测质量优于朴素 Transformer Engine NVFP4；单个 NVIDIA B300 节点上 learner 达到 33,540 tokens/s，完整 learner 更新吞吐为 BF16 的 1.39 倍、TE NVFP4 的 1.47 倍。框架、kernel 与配置已开源。
- **arXiv**：[2610.23928](https://arxiv.org/abs/2610.23928)

#### Rethinking Training-Inference Mismatch in LLM Reinforcement Learning: Where It Arises and How to Correct It (2026-09)
- **简介**：作者 Tianrun Yu、Kaixiang Zhao、Shangzhe Li、Weitong Zhang 等研究 RLVR 中推理引擎采样 rollout、训练引擎算梯度时对同一 token 给出不同概率的 training-inference mismatch。基于经验支持的 logit 位移刻画：不匹配可表示为 softmax 前 log-odds 的加性位移，其分布近似与 token 置信度无关，论文提出 **calibrated importance sampling（CIS）**，对大的正位移在单一常数阈值处截断，等价于随 token 置信度升高而收紧的重要性比上限。理论上，CIS 以受截断超额控制的偏差为代价，把精确重要性采样中无界的二阶矩替换为常数有界项。在三个 MoE 模型和五个数学推理基准上，CIS 在所有三个模型上均取得评测基线中最高的五基准平均。代码已开源。
- **arXiv**：[2609.32444](https://arxiv.org/abs/2609.32444)

#### Towards Full Pipeline FP8 Reinforcement Learning for LLMs (2026-09)
- **简介**：作者 Fanchao Chen、Ziheng Jiang、Ziyun Wei 等。FP8 量化可加速 RL 训练，但全流程 FP8 RL 即使有 TIS 等训练-推理不匹配校正仍严重不稳定，表现为训练中期熵飙升和输出乱码。论文将其归因于被忽视的原因：累积的 FP8 量化噪声扭曲重要性比值，使负 advantage token 被不成比例地推出 trust region 并错误地被零梯度，病态输出因而得不到惩罚。提出 **Calibrated Clipping**：通过匹配下界裁剪分位数并相应重平衡上界，使 FP8 的裁剪边界与高精度 BF16 分布对齐。在 GRPO 与 DAPO、8B 到 32B 模型和多种 FP8 scaling 粒度上，消除了熵飙升并恢复到与 BF16 基线相当的性能。
- **arXiv**：[2609.22870](https://arxiv.org/abs/2609.22870)

#### Score Centering Stabilizes Off-policy Reinforcement Learning (2026-09)
- **简介**：作者 Martin Marek、Max Ryabinin。LLM 的 RL 对训练引擎与推理引擎之间的细微差异（training-inference mismatch, TIM）非常敏感，而完全消除 TIM 会牺牲 rollout 效率。作者表明 TIM 下不稳定主要源于 drift：训练与推理引擎间持续存在、并随训练步数累积的偏差。他们推导出一个可加的「**score centering**」修正项来抵消 drift，从而稳定 TIM 下的 RL。在 0.6B 到 30B 参数模型的训练中，仅用 score centering 就能在量化设置下匹配或超过基于 importance sampling 的方法，且 mismatch 越严重差距越大；由于修正项是加性的，它还能与 importance sampling 组合，在 staleness 实验中优于纯 IS baseline。
- **arXiv**：[2609.20807](https://arxiv.org/abs/2609.20807)

#### Expert-Space Exploration in MoE Reinforcement Learning (ESRL) (2026-09)
- **简介**：作者 Hongyi He、Zhenghao Lin、Xiao Liu 等。MoE 模型 RL 的现有进展多聚焦优化稳定性与训练效率，把专家选择视为固定组件。作者发现扰动专家路由能像提高解码温度一样增加 rollout 多样性，但直接扰动会激活不合适的专家、显著损害 rollout 质量。**ESRL** 显式探索 MoE 的专家路由空间：保留高置信专家作为锚点，把随机路由限制在合理候选池内，并根据 router 熵自适应扰动强度；为缓解扰动造成的路由失配，rollout 时记录所用专家路径并在策略优化时回放。在 Qwen3-30B-A3B 上，平均 Pass@1 与 Pass@8 较 GRPO 分别提升 3.2 与 4.5 个百分点，覆盖数学、科学、代码，且无额外采样或计算开销。
- **arXiv**：[2609.13058](https://arxiv.org/abs/2609.13058)

#### MoE Proxy Models for Low-Cost Failure Reproduction and Diagnosis in LLM RL Post-Training (2026-08)
- **简介**：作者 Yikai Wang、Chuansai Zhou、Yuhang Zhou、Weiqiang Wu 等系统分析了在华为昇腾（Ascend）平台上做大规模 RL post-training 时遇到的故障，指出框架适配、数值 precision、算子实现等因素会引发 gradient overflow 与 loss divergence，而直接在大模型上复现这类故障代价极高。文章归纳了代表性故障类型并识别出与故障复现相关的三个模型侧因素，据此提出一套 proxy model 构造方法：用结构保持的、基于聚类的 expert pruning 挑选代表性 expert，同时保留 MoE 骨干架构、routing 机制与基础任务能力。实验显示 proxy model 将加速卡需求降低 50%–87.5%、单步 NPU-hour 成本最多降低 33.3×，且保留主要训练动力学、能复现与原模型一致的故障响应，可作为故障复现、定向验证与辅助诊断的低成本替身。属 §3.3 的工程框架/基础设施代表，主贡献是诊断与降本而非新的 TIM 修正算法。
- **arXiv**：[2608.10823](https://arxiv.org/abs/2608.10823)

#### ACRL: Adaptive Control of Training-Inference Discrepancy for Stable Reinforcement Learning（ACRL） (2026-07)
- **简介**：作者 Wenwu Fan 等人针对 LLM RL 训练中「训练-推理不匹配（Training-Inference Discrepancy）」导致的不稳定问题，指出其两大成因——训练/推理引擎架构分离，以及推理端低精度量化（如 FP8）与训练端高精度（BF16）计算的差异。提出 ACRL，通过自适应地把训练-推理差异约束在合理区间内来稳定训练；副作用是提升策略熵、增强探索、提高精度。实验显示推理端用 FP8 量化时，ACRL 既能稳定训练，又能追平 BF16 baseline，并优于重要性采样（IS）修正方案。定位为 TIM/Precision 层的显式对照方法（相对 IS-fix）。
- **arXiv**：[2607.24062](https://arxiv.org/abs/2607.24062)

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

#### On the Generalization Limitation of On-Policy Distillation for Post-RL Consolidation (2026-11)
- **简介**：作者 Yichen Wang、Yufa Zhou、Siqi Zhu 等。多教师 on-policy distillation（MOPD）把多个教师的能力整合进学生；此前工作报告近乎完美的分布内（IND）蒸馏表现，作者发现这可能掩盖了超出监督分布的能力整合不完整。在合成推理上的初步研究表明 OPD 不能完全迁移教师的泛化能力；进而在 Qwen3 模型上评估数学、代码、agent 与长上下文的外推与组合泛化：外推设置中 OPD/MOPD 达到教师 IND 分数的 96.04-102.73%，但 OOD 分数比基座模型低 2.45-16.25%；三个组合设置中 MOPD 提升了 IND，但组合任务（OOD）仍比最佳教师低 18.19-24.34%。分析教师能力与数据覆盖两个瓶颈，联合缓解后在实验中消除了 OOD 遗忘；(M)OPD 平均而言也比 RL 基线减少 OOD 遗忘，其更新几何更小、谱更集中。
- **arXiv**：[2611.05609](https://arxiv.org/abs/2611.05609)

#### Same Trajectory, Different Knowledge: Information-State-Aware On-Policy Distillation for Search Agents (ISA-Distill) (2026-11)
- **简介**：作者 Minhao Sun、Xian-Ling Mao、Eric Tao、Zi-Ao Ma、Zhijing Wu。On-policy distillation 用教师对学生生成轨迹的反馈训练 search agent；近期方法给教师上下文补充证据、参考答案或 hindsight 信息，但这类 privileged information 可能让教师依赖学生当前状态无法可靠恢复的事实，造成信息状态失配，使教师低估对学生建立这些事实有帮助的检索。**ISA-Distill**（Information-State-Aware Distillation）使教师反馈适配学生的事实可恢复性：估计学生恢复动作相关知识的可靠程度，并对同一采样动作从两个互补视角给出教师反馈——fact-use 视角给教师相关事实以指导学生使用知识，evidence-seeking 视角只呈现信息需求以指导学生获取支撑证据；可恢复性高时偏重前者、低时偏重后者，所得反馈指导策略优化。在多种 QA benchmark 与模型规模上提升了答案准确率与检索效率（abstract 未给出具体数值）。
- **arXiv**：[2611.01987](https://arxiv.org/abs/2611.01987)

#### Bernoulli On-Policy Distillation: Don't Ignore High-Confidence Student Tokens (2026-11)
- **简介**：作者 Mingyu Cao、Yaokai Jin、Shiwei Liu、Lu Yin。On-policy distillation（OPD）在学生自己的轨迹上用更强教师的 token 级监督训练学生。作者指出标准的采样 reverse-KL OPD 会系统性地抑制对高置信度 token 的更新——其弱梯度贡献并不意味着教师监督弱或无信息。提出 **Bernoulli OPD**：直接匹配该 token 上教师与学生的概率，得到有界的概率空间修正，既保留师生分歧、又避免 vanilla OPD 引入的置信度相关抑制。固定 rollout 分析表明它显著恢复了高置信度 token 的优化信号，同时仍对师生分歧敏感；对置信度分布、答案校准、策略移动与训练动态的分析显示其收益并非简单源于更长推理、更大优化步长或更均匀地模仿教师。代码已开源（abstract 未给出具体数值）。
- **arXiv**：[2611.02093](https://arxiv.org/abs/2611.02093)

#### EvoOPD: Evolving Privileged Information Strategies for Black-Box On-Policy Distillation (2026-11)
- **简介**：作者 Ximo Zhu、Ruiqi Liu、Yi Han 等。闭源模型输出分布不可得，依赖教师 logits 的细粒度蒸馏难以直接用于黑盒模型。作者观察到黑盒教师的富文本反馈可视为 privileged information（PI），经 PI 条件化的 privileged teacher 转化为学生自身轨迹上的稠密监督信号；但应从教师获取何种 PI 才最有利于学生学习并不清楚——PI 策略在关注点、请求的信息类型、细节程度上差异显著，且其相对效果随训练进程变化。**EvoOPD** 把 PI 获取本身当作优化问题，维护随学生状态演化的 PI Strategy Bank，依据 PI 在当前学生上的实测 repair utility 自适应调整教师反馈的关注点、信息类型与粒度。在数学推理与代码生成上以 Qwen3-1.7B、4B、8B 学生系统评估（摘要后续的具体数值未显示）。
- **arXiv**：[2611.06542](https://arxiv.org/abs/2611.06542)

#### Replay for LLM Continual Learning with Multi-Teacher On-Policy Distillation (2026-11)
- **简介**：作者 Yiqiao Qiu。多教师 on-policy distillation（MOPD）把多个领域专家蒸馏到一个学生，但要求所有专家同时可用；现实中能力往往先后到来，早期领域的专家可能已不在，顺序蒸馏会导致灾难性遗忘；一些流水线为每阶段存快照并以所有快照做多教师恢复，存储随阶段数增长。本文从「保留什么」出发形式化 continual MOPD，研究向上一阶段学生做 replay：每个阶段内，以上一阶段的学生作为教师回放早期领域的 prompt，因而无需保留早期专家或阶段快照。探索两个设计选择：回放预算在早期领域间共享还是按领域固定；回放步骤只蒸馏 token 分布，还是同时对齐隐状态、区域池化的 self-attention 分数与 value 关系。基于一个 1.5B 学生的公开专家构建逻辑、数学、代码、安全、指令遵循五领域 benchmark（14 个评测集）进行实验（摘要后续具体结果未显示）。
- **arXiv**：[2611.02746](https://arxiv.org/abs/2611.02746)

#### MDSKILL: Learning to Internalize Skill Memory through Multi-View Distillation for Efficient Dynamic Agent Self-Evolution (2026-11)
- **简介**：作者 Dian Jin、Jinluan Yang、Kairong Han 等。skill memory 把过往经验编码为文本 prompt 以改进自进化 agent 的 RL；On-Policy Self-Distillation（OPSD）可提供稠密监督，把 skill 引导轨迹中的知识内化。但自生成记忆常含噪声监督及功能纠缠、冗余或冲突的知识，在有限训练预算下会阻碍内化甚至降低性能。**MDSKILL** 把 skill memory 组织为三个互补功能视图（state grounding、action planning、execution control），用动态指标筛选高质量 skill，构建视图特定与整合的推理分支，再由 OPSD 把各分支互补的经验知识迁移并巩固到共享策略中。在三个 agent benchmark 上对比 RL、skill 增强与不依赖更强外部教师的自蒸馏基线，在无额外计算开销下，ALFWorld 与 WebShop 上分别比 SOTA 最多提升 7.1% 与 8.1%。
- **arXiv**：[2611.05313](https://arxiv.org/abs/2611.05313)

#### Distilling the Shape of Uncertainty: What On-Policy Distillation Learns and How to Predict It (2026-10)
- **简介**：作者 Shiping Yang、Chuanqi Tang、Hossein Hajimirsadeghi、Layla El Asri、Mohammad Amin Shabani、Angel X. Chang。OPD 在 LLM 后训练中日益常用，但更强的 teacher 未必能提升 student，先前工作归因于缺少新知识，却不清楚是什么知识。论文表明 OPD 不仅学习 how to branch（考虑哪些候选 token），还学习 where to branch（在哪里保持推理路径开放），而现有 token-overlap 指标只捕捉前者。作者提出以逐位置 next-token 熵序列刻画后者的 entropy shape，并提出 **ESC**（Entropy Shape Correlation）度量师生在熵形状上的一致性；通过跟踪训练中的 ESC 并在保持 teacher token 偏好的同时干预其熵，实证 where to branch 独立于 how to branch 因果驱动 OPD 增益。最后把 OPD 结果预测形式化为 teacher 选择任务，OPD 前的 ESC 对两种 student 均能选出最佳 teacher，优于现有预测指标。
- **arXiv**：[2610.39020](https://arxiv.org/abs/2610.39020)

#### Frontier-OPD: Discovering the Edge of Agent Capability and Teaching Agents to Go Beyond (2026-10)
- **简介**：作者 Zheng Jiang、Yiming Chen、Lifeng Sun 等。多轮 agent 的 on-policy distillation 需要同时判断在哪里需要指导，以及 student 之后能否自主维持交互。**Frontier-OPD** 围绕 student 的学习前沿——仍有挑战、但在简短 teacher 指导后可自主续写的决策情境：把 on-policy 经验组织为共享状态轨迹图，通过轨迹均衡聚合师生分歧来识别不同的干预候选并保留重放所需历史；随后通过比较 teacher bridge 及其后 student 续写的一致性与原轨迹等长片段，评估该指导是否能被当前 student 利用；被接受的 teacher 响应补充普通蒸馏数据，并用新 rollout 随策略变化更新前沿估计。在 ALFWorld、WebShop、ScienceWorld 上，各 student 规模与 teacher 配置下的平均自主表现均高于对比的蒸馏基线；在某个 ALFWorld 计时设置中，额外的收集成本被更短的达标时间抵消。
- **arXiv**：[2610.32802](https://arxiv.org/abs/2610.32802)

#### RPSD: Retrieval-Privileged On-Policy Self-Distillation with Optimal Transport (2026-10)
- **简介**：作者 Dongyue Li、Zebang Cheng、Yi Bin、Fei Ma 等。基于 LLM 的生成式序列推荐越来越多用 GRPO 后训练，但存在 credit-blind 优化（序列级奖励被所有生成 token 共享）和 collaborative-blind 监督（训练信号只来自单一用户、忽略相似用户的共享行为模式）。**RPSD** 引入 on-policy self-distillation：具备 privileged knowledge 的 teacher 重新评估 student 的 on-policy 响应，用师生 token 级置信度之差把共享的序列级 advantage 重塑为 token 级 credit；并让 teacher 基于相似用户的协同行为证据——用 temporally aware optimal transport 联合考虑 item 相似度与时间结构来比较跨用户交互轨迹，再用 diversity-aware 选择保留互补轨迹作为 privileged evidence。据称是首个把 OPSD 引入序列推荐并以跨用户行为为 privileged 信号的工作，在多个真实数据集上达到 SOTA（abstract 未给出具体数值）。
- **arXiv**：[2610.37281](https://arxiv.org/abs/2610.37281)

#### SCOPE: Selective Critical-Step On-Policy Self-Distillation for Agentic Reinforcement Learning (2026-10)
- **简介**：作者 Qirui Liu、Wei Cao、Yu Mi 等。结果型 RL 对长程 LLM agent 的监督较粗：失败轨迹可能含正确步骤，成功轨迹也可能含错误；轨迹级 skill 还可能改变无关步骤上 teacher 的偏好，引入偏离目标的蒸馏信号。**SCOPE**（Selective Critical-Step On-Policy Self-Distillation）在 SFT warm-up 后让策略分析自身轨迹、生成步骤级 hindsight skill，并仅在选定的错误步骤上蒸馏其指导；有界的带符号师生系数稳定辅助更新，同时保留正向强化与负向纠正；成功相对的 pairwise 长度奖励鼓励高效完成。Qwen2.5-3B 上 ALFWorld 成功率 97.7%、WebShop 85.2%，分别比 SEED 高 5.9 和 6.3 个百分点；未见 ALFWorld 任务达 89.6%，比 GRPO 高 23.2 点且交互少 46.7%；扩展到 Qwen3-32B，见过任务 99.2%、未见任务 94.0%。
- **arXiv**：[2610.38391](https://arxiv.org/abs/2610.38391)

#### DTP-OPSD: Reinforcement Learning with Distractor-Aware Temporal-Privilege On-Policy Self-Distillation for Temporal Video Grounding (2026-10)
- **简介**：作者 Zhaoyu Chen、Yongwei Nie。Temporal Video Grounding（TVG）在同一视频含多个重复、相似或相反事件时很难，GRPO 采样的所有回复可能都错过目标、tIoU 为零，组相对 advantage 全为零而无策略梯度。**DTP-OPSD** 结合 RL 与 distractor-aware temporal-privilege on-policy self-distillation：student 观察干净视频并 on-policy 生成回复，共享权重的 teacher 观察特权视频（目标窗口用红色整帧边框、同视频干扰窗口用蓝色边框标记），teacher 评估 student 生成的前缀并提供稠密 token 级 Jensen-Shannon 散度监督，即使 GRPO 无梯度时仍有信息；teacher 不优于 student 时抑制蒸馏，在 teacher 更好而 student 全部 tIoU 为零的组上加强。仅用 300 条额外后训练样本，在 Charades/ActivityNet/QVHighlights-TimeLens 上取得开源 8B 模型中最高 mIoU（55.93、54.77、66.71），相比同数据 Pure GRPO 在全部 12 项指标上均提升。
- **arXiv**：[2611.00120](https://arxiv.org/abs/2611.00120)

#### ToolOPD: Credit-Aware On-Policy Distillation for Tool-Use Agents (2026-10)
- **简介**：作者 Chu Zhao、Lei Tang 等。标准 on-policy distillation 默认长程轨迹上各动作的 credit 均匀分布；作者通过 credit 分布分析与「结局翻转」反事实实验发现，credit 集中在少数关键动作上，且仅翻转最终结局就会显著改变多个 turn 的 credit，均匀假设会削弱推理准确率。论文提出 **ToolOPD**：先把完成的轨迹及其结局总结为可复用的 hindsight skill，再度量该 skill 对 teacher 在各 student 动作 token 上概率的改变，将其转为 outcome-aware credit——保留支持成功决策或抑制失败决策的监督、压制无关与冲突信号；用该 credit 重新加权动作级蒸馏损失，并与 RL 联合优化。在多个长程工具交互 benchmark 上显著优于 RL 与 self-distillation 基线。（abstract 未给出具体数值）
- **arXiv**：[2610.25693](https://arxiv.org/abs/2610.25693)

#### Correctness-Dependent Asymmetric Target Distributions for Self-Distillation Policy Optimization (APSD) (2026-10)
- **简介**：作者 Suwan Yoon、Seongryong Jung、Changhee Lee。Self-distillation policy optimization 通过以反馈为条件的 self-teacher 为 LLM 后训练提供稠密 token 级监督，但直接匹配 self-teacher 可能破坏正确 rollout 上的成功行为，而对错误 rollout 的纠正又不足。论文提出 **APSD**（Asymmetric Policy Optimization via Self-Distillation），通过混合 student 与 self-teacher 分布构造依赖正确性的目标：对正确 rollout 做插值以保留成功行为，对错误 rollout 外推到 self-teacher 之外以加强纠正。文中用散度分解、reverse-KL 重心几何和梯度分析刻画方法，强调梯度需流经依赖 student 的目标。在 RLVR 与富反馈设置下，APSD 整体优于 RL 和 self-distillation 基线，在 Qwen3 各规模上平均以 2.6 倍更少训练步数达到 SDPO 的表现。
- **arXiv**：[2610.25973](https://arxiv.org/abs/2610.25973)

#### OPDArena: Understanding Learning and Forgetting in On-Policy Distillation (2026-10)
- **简介**：作者 Yuanze Xu、Tianzhu Ye、Hui Sun、Wenhui Wang、Li Dong、Furu Wei。OPD 越来越多用于语言模型后训练，但 teacher 监督如何塑造学习与遗忘尚不清楚。论文提出 **OPDArena** 框架，在五个预训练模型和多种 teacher 来源上对比 SFT、RLVR 和 OPD，联合评估数学学习、通用能力保留与预测分布变化。结果显示 teacher 选择强烈影响能力保留：由 RLVR 训练的 teacher 所教出的 student 在目标任务上提升且保留已有能力，而 SFT 训练的 teacher 的 student 能力损失更大；跨各后训练方法，域外分布变化与遗忘的正相关比域内变化更一致。受控 teacher 对比进一步表明，teacher 通过后训练获得的任务偏好使 student 的更新在域内更为集中。代码已开源。
- **arXiv**：[2610.25651](https://arxiv.org/abs/2610.25651)

#### Capable Teacher, Myopic Scorer: Reward-Free Teacher Repair for Agentic On-Policy Distillation (R2S) (2026-10)
- **简介**：作者 Shu Zhao、Tan Yu。Agentic OPD 的常见解释是 compounding error，因此现有方法用课程、teacher 引导 turn、prefix replay 等把训练限制在 teacher 可靠区域内。作者指出这一解释不完整：在 student 偏离 teacher 最远的状态上，teacher 自己接管仍能完成任务，但按其 token 分数训练的 student 仍会耗尽 turn。原因是 next-token 打分的短视——teacher 不知道 episode 终点，会放过 outcome RL 会以零 reward 惩罚的安全拖延动作，作者称之为 rollout-scorer gap。提出 **Rollout-to-Scorer Distillation (R2S)**：在蒸馏前修复 scorer——挖掘未训练 student 的高分歧状态，让 teacher 接管，保留足够多接管通过无 reward 检查（共识或 LLM judge）的状态，微调 teacher 在分歧 turn 之后的每个 assistant token 上给出接管感知的分数；修复后的 teacher 冻结为标准 OPD 的唯一 scorer，全程无需任务 reward。在三个环境、两个 teacher、四个 student 的全部十二个组合中优于三个 SOTA agentic-OPD 方法，平均比各组合中最优者高 12.8 个成功率点。
- **arXiv**：[2610.28586](https://arxiv.org/abs/2610.28586)

#### ECOD: Experience-Calibrated On-Policy Distillation for Multi-Turn Agents (2026-10)
- **简介**：作者 Jie Jin、Shengqin Wang 等。稀疏延迟奖励使长程语言 agent 的 RL 困难，OPD 以 student 生成上下文上的稠密 teacher 反馈作补充，但标准 OPD 可能与 teacher 相对能力、student 轨迹结局及任务执行的时间结构不对齐。论文提出 **ECOD**（Experience-Calibrated On-Policy Distillation），在 task、trajectory、turn 三个层面分配 teacher 监督：在匹配任务条件下比较 teacher 与 student 成功率后验，把直接监督限制在失败的 student 轨迹上，并由成功 teacher 轨迹的长度导出时间先验，所得权重作用于 teacher loss 或 teacher 导出的 advantage。在 ALFWorld 和 WebShop 上用 Qwen2.5-1.5B/7B-Instruct 评估：匹配 teacher 设置下 ECOD 使标准 OPD 在 ALFWorld unseen 成功率最多相对提升 4.06%；与 SOD、TCOD-F2B、ATOD 配合在两个规模上平均相对提升 ALFWorld unseen 6.20%、WebShop 7.05%。
- **arXiv**：[2611.18656](https://arxiv.org/abs/2611.18656)

#### Edit, Don't Imitate: Minimal-Edit On-Policy Distillation for Multimodal Reasoning (2026-10)
- **简介**：作者 Wen Ye、Ailin Deng 等。视觉数学推理常在推理开始前就失败：VLM 读错图表后会基于错误前提继续推理，小模型尤其明显，而准确的文字描述可显著缩小与大模型的差距。论文提出 minimal-edit corrective on-policy distillation：把参考图描述与已验证的参考解答作为仅训练时可用的信息，teacher 在 minimal-edit 目标下对错误的 student rollout 打分，只有 teacher 不喜欢的 token 才贡献纠正损失；正确 rollout 则接受任务 reward 的 RL。这样既保留 on-policy 监督，又避免整段回复模仿。该方法测试时无需描述，在各模型规模上达到或超过推理时提供参考图描述的基座模型，并提升以低层几何感知为主的感知性能、迁移到外部数学与感知 benchmark；由于 teacher 评估修正而非自行生成解答，用冻结的同尺寸 teacher 也有效，只是多数指标上增益小于更大 teacher。（abstract 未给出具体数值）
- **arXiv**：[2610.30231](https://arxiv.org/abs/2610.30231)

#### A Stronger Teacher Within: Strengthening MoE Models via On-Policy Bidirectional Distillation (MoEvolve) (2026-10)
- **简介**：作者 Haoyu Chen、Jun Bai 等。RLVR 显著提升了 MoE LLM，但 outcome 奖励监督稀疏；OPD 提供稠密 token 反馈却需要更强的兼容 teacher，而 on-policy self-distillation 依赖人工设计的特权信息。作者观察到 MoE 架构本身提供了出路：增加每 token 的激活 expert 数会暴露潜在 expert 能力，从而在模型内部得到更强 teacher；但直接扩张会因新共同激活的 expert 未经协同优化而掉点。论文提出 **MoEvolve**，遵循「先恢复后提升」的双向蒸馏：先恢复激活扩张后的高 k 模型并利用其额外激活容量进一步增强，再反向蒸馏把增强后的能力迁移回原始低 k 配置，保留推理时的激活预算。在多种 MoE 模型和 benchmark 上，MoEvolve 比初始模型最多提升 22.2 个 domain 平均分，比同预算 RLVR 基线最多高 6.5 分，最终低 k 模型常常超过其激活扩张的 teacher。
- **arXiv**：[2610.23185](https://arxiv.org/abs/2610.23185)

#### Beyond the Teacher: Understanding and Preserving Distillation Overshoot in Agentic On-Policy Distillation (SPOD) (2026-10)
- **简介**：作者 Jike Zhong、Ming Li、Yuxiang Lai、Xi Xiao、Sheng Zhang 等。在多轮 agentic 任务的 on-policy distillation（OPD）中，论文发现反直觉现象 distillation overshoot：较小且起初更弱的 student 在训练中可显著超过 teacher，随后随蒸馏继续而退化。通过决策级与任务级分析，将其归因于 OPD 的两种竞争效应：修复 student 的弱决策，以及侵蚀 teacher 不偏好的成功 student 选择；早期修复占优、后期侵蚀占优。据此提出 **SPOD (Success-Preserving On-Policy Distillation)**：对 student 成功选择上与之冲突的 teacher 监督做选择性 mask，并引入 soft probability floor 抵消侵蚀，其余处保留 teacher 指导，只需二值的 episode 成功信号与 OPD 中已有的模型概率。在两个模型族、三个 agentic benchmark 上，SPOD 稳定优于普通 OPD，最终成功率最高提升 15.2 个百分点、最多超过 teacher 20.0 点，且六个设置中的最终 checkpoint 均优于普通 OPD 的最佳中间 checkpoint。
- **arXiv**：[2610.25034](https://arxiv.org/abs/2610.25034)

#### Asynchronous On-Policy Self-Distillation (AsyncOPSD) (2026-10)
- **简介**：作者 Xidong Wu、Qi Qi、Yuqiong Ji、Shangqian Gao。on-policy self-distillation（OPSD）让同一个语言模型同时充当 student 与 teacher（teacher 以特权信息为条件），架构简单，但同步 rollout 生成主导训练时间，成为系统瓶颈。提出 **AsyncOPSD**，将 rollout 生成与 learner 更新解耦以提高吞吐；由于 teacher 与 student 共享参数，避免了 OPD 架构在异步训练中的多模型协调开销。随着 student 策略演化，缓冲区中旧策略生成的轨迹会变得陈旧，为此引入基于 staleness-aware policy control 与 token 级熵加权的蒸馏目标。在三种模型规模与数学推理 benchmark 上，AsyncOPSD 相比同步 OPSD 将训练吞吐提升 1.96-2.25 倍，同时提高平均 benchmark 精度。
- **arXiv**：[2610.25880](https://arxiv.org/abs/2610.25880)

#### A Fisher-Geometric View of Multi-Teacher On-Policy Distillation (RGBM) (2026-10)
- **简介**：作者 Liu Kaiyuan、Zhuang Ziyuan、Ma Haoxuan、Gu Yannian、Song Mingfei、Weng Rongxiang。把多个专才模型整合为通才是后训练的核心目标：模型合并快但被视为启发式，multi-teacher on-policy distillation 能匹配分布但 rollout 开销大且有跨任务梯度冲突。论文证明二者是同一目标在不同阶数下的表现：在同源专才的共享 basin 内，multi-teacher OPD 目标在二阶上恰为 Fisher-merging 目标，其最小值即 Fisher merge，故合并是蒸馏在二阶下的闭式阶段。据此提出 **RGBM (Residual-Governed Branch-and-Merge)** 两阶段整合：Phase 1 把合并限制在 task-vector 子空间，并利用混合 task vector 的系数同样混合隐式奖励这一性质，在无标注 prompt 上用残差程序求解，得到域残差谱；Phase 2 让该谱决定哪些域可被追加，仅对高于谱间隙的域训练独立分支并以校准收缩写回。在两个开源整合套件上，仅 Phase 1 合并就达到或超过完整训练的 multi-teacher 蒸馏，RGBM 在零额外联合训练步数下进一步提升。
- **arXiv**：[2610.26525](https://arxiv.org/abs/2610.26525)

#### Ask to Distill: Learning to Synthesize Data for On-Policy Distillation (Ask-to-Distill) (2026-10)
- **简介**：作者 Jiashuo Sun、Hao Zhang、Xueqiang Xu、Heng Wang、Yixuan Xie、Pengcheng Jiang 等。OPD 只需很少训练问题即可奏效，启发把问题视为分布知识迁移的载体。**Ask-to-Distill** 将合成问题视为分布诱发器，优化其在 student 生成轨迹上暴露的 teacher 监督：以 student 初始化的 synthesizer，用由冻结的 student-teacher 对计算、无需参考答案/正确性奖励/逐候选 student 更新的分布式代理奖励优化，奖励结合 teacher-student 分歧与 teacher 在 student 候选支撑集上的概率质量，并加入长度与多样性控制；生成的问题经过滤与去重形成仅含 prompt 的题库供标准 OPD 使用。在数学推理、代码生成与 agentic 任务上，该方法在平均表现与 teacher-gap 恢复上优于未优化的 student 生成 prompt 与 KL 过滤的真实问题，并在评测设置中匹配或超过真实题库 OPD。代码、数据、模型已公开。
- **arXiv**：[2610.27450](https://arxiv.org/abs/2610.27450)

#### CORD: Conflict-Aware Optimization for Joint RL and Distillation in Agents (2026-10)
- **简介**：作者 Wenhao Yu、Zhengcheng Wang、Jiahong Liu、Zexuan Qiu、Minda Hu、Shaohang Wei 等。agent 训练中联合 RL 与 self-distillation 把任务奖励与稠密 teacher 指导结合，但两者梯度会部分抵消，混合系数只能控制相对尺度而不能选择性去除相反分量。**CORD** 是面向两个目标的冲突感知优化框架：将「被保护目标」的选择与触发投影的条件分离，并在每个 optimizer mini-batch 上合并一次梯度；采用 nearest-point 形式得到闭式更新，并刻画其对被保护目标的一阶影响；梯度诊断将 agent 训练中的冲突与局部优化进展损失联系起来。作为梯度合并机制，CORD 可叠加到不同联合训练配方。在三种配方上使 ALFWorld 成功率提升 1.5-15.9 个百分点；WebShop 上三组中两组提升，其中 SDAR 成功率从 60.7% 升至 64.6%。
- **arXiv**：[2610.29965](https://arxiv.org/abs/2610.29965)

#### GIFT: Gradient Informed Filtering for Distillation Augmented Reasoning RL (2026-10)
- **简介**：作者 Shijie Li、Yilin Gao、Tieyuan Chen、Yun Luo、Weiyao Lin、Hang Yu。复杂推理训练依赖两类互补信号：稀疏的 outcome RL 与稠密的 token 级蒸馏，但监督互补不等于优化兼容，模仿式蒸馏可能与 outcome 驱动的 RL 冲突，现有混合方法缺少原则性解决办法。**GIFT** 对蒸馏梯度做 token 级几何分解：与 RL 梯度平行的分量度量一阶冲突、作为 safety 过滤器，正交分量度量新的方向性指导、作为 value 度量，据此得到对 token 级指导排序的打分规则；再配合渐进的 coverage annealing，逐步收缩接收蒸馏的 token 预算，优先丢弃冲突最大的更新，使训练平滑过渡到收敛时纯 outcome 目标。在数学与逻辑推理 benchmark、多个模型规模上稳定优于强基线，在困难数学推理任务上相对提升最高 9.5%。代码已开源。
- **arXiv**：[2610.30796](https://arxiv.org/abs/2610.30796)

#### Function-Space Temporal Self-Distillation for Policy Optimization (FSPO) (2026-10)
- **简介**：作者 Aryan Das、Arnab Bhattacharya、Vinay Kumar Verma。RLVR 的标量结果监督过于粗糙；self-distillation 通过对比无辅助策略与获得反馈的 self-teacher 提供更稠密的 token 级指导，但 teacher 随 student 演化会产生不稳定的目标。论文提出 **FSPO（Function-Space Policy Optimization）**：将当前的 privileged adaptation 与 function-space temporal consolidation 结合，把稀疏的历史 next-token 分布存为固定的时间目标，无需维护常驻 teacher；并引入 privileged disagreement correction、reliability-adaptive grounding 以及兼顾质量与覆盖的 temporal memory。在推理与工具使用基准上，5 小时训练预算下 FSPO 平均准确率最高，比最强基线平均高约 4 个百分点，同时在 Olmo3-7B-Instruct 上将峰值显存从约 70 GB 降到 56 GB。
- **arXiv**：[2610.19687](https://arxiv.org/abs/2610.19687)

#### Teach Me After You Learn It: Lookahead Self-Distillation for RLVR (LEAD) (2026-10)
- **简介**：作者 Zepeng Zhai、Meilin Chen。现有 on-policy self-distillation（OPSD）的监督仅来自同一实例的 privileged information，跨实例监督尚未被探索。**LEAD（Lookahead Self-Distillation）** 让 teacher 先从其他 on-policy rollout 中学习再去教 student：在互不相交的 rollout 子集上做虚拟 RL 更新构造 lookahead teacher，用其为另一子集提供跨组 token 级监督；并提出 harmonious modulation，依据 OPSD 与 RLVR 两种信号的一致程度自适应地调节 GRPO advantage。在科学推理与交互式 agent 基准上，LEAD 在各模型规模上取得最佳的域内平均表现，域外泛化与 GRPO 相当（abstract 未给出具体数值）。
- **arXiv**：[2610.18256](https://arxiv.org/abs/2610.18256)

#### Branching On-Policy Distillation (BranchOPD) (2026-10)
- **简介**：作者 Dongyang Chen、Shudong Liu、Shubo Li 等。OPD 在 student 自己生成的轨迹上提供 teacher 监督，但单条轨迹在每个决策点只实现一个 continuation，有限采样预算下监督覆盖有限。**BranchOPD** 利用 student 不确定性与 teacher-student 分歧选出信息量大的决策点，从共享前缀重采样 student 策略的 continuation，并对主轨迹与分支都施加 token 级 teacher 监督，适用于数学的 token 级决策与 agent 任务的 turn 内决策；共享前缀复用使 student 生成 token 成本相对 7 条独立轨迹在 ALFWorld 和 WebShop 上分别降低约 19% 与 31%。另提出 protocol-boundary repair（PBR）缓解重复与无效生成。BranchOPD 在所有评测上优于 Vanilla OPD 和 Multi-Rollout OPD（n=7），相对 Vanilla OPD 数学 avg@16 平均提升 18.55 个百分点，ALFWorld ID/OOD 平均成功率提升 13.8 点。
- **arXiv**：[2610.13481](https://arxiv.org/abs/2610.13481)

#### Information-Gain Calibrated On-Policy Distillation for Deep Search Agents (2026-10)
- **简介**：作者 Chenxi Li、Xueyang Feng、Songyang Liu、Xiaoying Tang、Chaozhuo Li 等。深度搜索中，teacher 在检索前对候选 query 的偏好未必能预测哪个 query 会返回有用证据，而标准 OPD 仅依据此前上下文监督每个 student query，形成 evidence-blind supervision（EBS）。作者按交互 turn 重组 OPD 目标，揭示两个隐含先验：按 token 数量分配监督的 budget prior，以及偏向 teacher 偏好 token 的 direction prior。实证发现首轮搜索占总绝对信息增益的 51.1% 却只获得 18.0% 的监督预算，且 43.7% 的搜索 query 的 teacher 奖励与检索后信息效用符号相反。论文用检索后的信息增益（IG）同时校准这两个先验。在六个多模态搜索基准上，2B 与 4B student 平均准确率比标准 OPD 提升 3.4 与 1.7 个百分点，4B student 在 MMSearch 和 BrowseComp-VL 上超过其 8B teacher。
- **arXiv**：[2610.20664](https://arxiv.org/abs/2610.20664)

#### State-OPD: State-Grounded On-Policy Distillation for Multi-Turn Agents (2026-10)
- **简介**：作者 Zhentao Zhang、Yanzhao Zheng、Yanlin Liu 等。把 OPD 扩展到多轮 agent 环境仍有挑战。作者以交互状态审视 agentic OPD，发现 student 推理的两种作用：先前的推理会扭曲 teacher 监督，而当前推理决定同一状态下采样到的动作备选。在匹配的交互状态上去掉 student 之前的推理可提升 teacher continuation 成功率并改变其 token 级目标；重采样完整的「推理-动作」响应比在固定推理下重采样动作带来更高的动作多样性。**State-OPD** 把 teacher 监督建立在交互状态上，并把采样预算重分配到 response 熵高的状态以生成多个完整推理-动作响应。在 ALFWorld、ScienceWorld、WebShop 上，Qwen3-1.7B/4B student 均取得最高宏平均成功率；Qwen3-4B 从最强基线的 26.07% 提升到 31.74%，并在 ALFWorld Seen 上超过 Qwen3-32B teacher。
- **arXiv**：[2610.13238](https://arxiv.org/abs/2610.13238)

#### Reward-Aligned Distillation (RAD) (2026-10)
- **简介**：作者 Yilan Chen、Zian Su、Manish Prajapat、Dingmin Wang、Rein Houthooft、Rui Song。标准 OPD 不管 teacher 偏好是否真能提高任务奖励，对每个 token 都蒸馏；已有 token 选择方法依赖不确定性或 teacher-student 散度等未必指向奖励提升的代理量。**RAD** 是双层优化框架：通过对临时优化器步骤求微分，评分每个 token 的监督对 student 更新后奖励的预测影响，再从初始权重出发做投影上升更新 token 权重，权重可增强、抑制甚至反转单个蒸馏梯度贡献。在相同训练 response 预算下，六个竞赛数学基准的平均准确率比最强基线高 6.5 点（Qwen3-1.7B non-thinking）和 2.5 点（Qwen3-4B-Instruct-2507）；LiveCodeBench-v6 上 pass@1 分别达 54.69% 与 58.67%（Qwen3-1.7B 与 OLMo3-7B-Instruct），超过最强基线 8.7 和 6.4 点。简化变体 RAD-Simple 数学准确率相当。
- **arXiv**：[2610.14572](https://arxiv.org/abs/2610.14572)

#### Condition-Contrastive Extrapolation for On-Policy Distillation of Language Models (CCE-OPD) (2026-10)
- **简介**：作者 Wenxuan Luo、Chenxi Li、Haoyue Liu、Yongxin Guo、Xiaoying Tang。在 student 生成的前缀下，teacher 可能偏好违反题目前提的无效 continuation，使 OPD 鼓励错误推理步骤。**CCE-OPD** 让 teacher 监督对题目条件敏感：（1）固定前缀比较，同一冻结 teacher 分别评估完整问题与遮蔽某条条件的问题；（2）有界 log 概率对比，据此重加权 teacher 分布得到辅助蒸馏目标。在 200 个诊断推理状态上，条件对比将偏好无效 continuation 的比例从 27.0% 降到 16.0%；在六个数学基准上，将 Qwen3-8B 蒸馏到 0.6B 和 1.7B student 时 avg@8 分别比 OPD 平均高 1.60 和 1.95 个百分点，并在四个迁移基准上取得最佳结果。
- **arXiv**：[2610.21195](https://arxiv.org/abs/2610.21195)

#### Rethinking Offline On-Policy Distillation: From Scalar Advantages to Distributional Supervision (DistOPD) (2026-10)
- **简介**：作者 Qi Sun、Siyue Zhang、Tingyu Song、Yilun Zhao、Chen Zhao。OPD 需要反复用 teacher 评估新鲜 rollout，代价高；Lightning OPD 等离线方法在固定参考 rollout 上缓存 teacher log 概率来降低成本。作者将其推广到多轮 agentic search，发现它始终不如同一 teacher 训练的 SFT 起点；经验上观察到比数学推理更大的 teacher-reference 失配与后训练分布变平坦，并从理论上证明固定 rollout 上的标量 advantage 监督在一定条件下允许输出分布变平的解。据此提出 **DistOPD（Distributional On-Policy Distillation）**：在选定位置保留参考与 teacher 的候选 token 概率，并把其余部分聚合为一个 residual token。在 agentic search 上，DistOPD 性能可与在线 OPD 相当，提升 student-teacher 对齐，训练 GPU 时长与 Lightning OPD 相近。
- **arXiv**：[2611.18648](https://arxiv.org/abs/2611.18648)

#### JustOPD: On-Policy Distillation from Autoregressive to Diffusion Language Models (2026-10)
- **简介**：作者 Zikun Zhang、Jiayuan Sheng、David D. Yao、Wenpin Tang。扩散语言模型（DLM）上的 OPD 尚未被充分探索，已有工作主要是 DLM 到 DLM 蒸馏；而强 DLM teacher 稀缺、生成顺序灵活又带来额外挑战。如何把强自回归模型（ARM）的知识蒸馏给较弱 DLM，受制于 ARM 左到右解码与 DLM 灵活解码顺序之间的不匹配。**JustOPD** 在训练时使用自回归 rollout，使 student 的生成过程与 ARM teacher 的左到右解码对齐，同时在推理时保留 DLM 的灵活生成顺序能力。以 Qwen3-14B 为 teacher、Dream-7B 为 student，在多种 divergence 选择下数学推理能力均一致优于基础 DLM（abstract 未给出具体数值）。
- **arXiv**：[2610.17397](https://arxiv.org/abs/2610.17397)

#### ActionOPD: Distilling Decisions, Not Tokens (2026-10)
- **简介**：作者 Huining Yuan、Xiangmin Yi、Zelai Xu、Zixu Wang、Xinlei Chen、Xiao-Ping Zhang。蒸馏 agent 时，离线模仿提供稠密但 token 特定的监督且有状态分布失配，token 级 OPD 解决失配却需 teacher token 概率与兼容的分词。作者主张语义动作是 agent 蒸馏的天然监督单位。**ActionOPD** 是动作级 on-policy distillation：在 student 遇到的状态上，同时从 student 和黑盒 teacher 采样动作，按语义一致性奖励 student 动作，再通过策略优化对齐 student 与 teacher 的决策，无需复现 teacher 写的推理或访问其 token 概率，也不要求同一模型家族或 tokenizer。训练 Qwen3-4B 和 Qwen3-8B（两个 teacher）在多样工具使用任务上，四个 OOD 基准平均提升 3.3-6.9 个百分点；在 tau^2-bench 上与离线蒸馏和 token 级 OPD 相比有竞争力。
- **arXiv**：[2610.19820](https://arxiv.org/abs/2610.19820)

#### DiffGate: Difficulty-Gated Teacher Guidance for On-Policy Distillation (2026-10)
- **简介**：作者 Karn Tiwari、Varnith Chordia、Prathosh A P。现有 OPD 目标多是 token 局部且与结果无关的，而推理质量由整条轨迹决定；RLVR/GRPO 虽有结果级监督，却奖励稀疏、信用粗糙且在全失败组上信号消失。论文指出两者盲区互补，提出 **DiffGate**：把 GRPO 与选择性、有界的 teacher 引导结合，仅对失败轨迹施加 teacher 监督，按 group 难度缩放并平滑截断，由 verifier 决定「哪些轨迹」接受引导，teacher 在这些轨迹内提供稠密 token 级更新方向。在 Qwen3-0.6B / 1.7B 上，code avg@8 比同配置 GRPO 提升 +1.7 / +1.8，pass@8 提升 +1.6 / +5.7；数学 avg@8 与 GRPO 相差在 0.5 以内，pass@8 提升 +1.1 / +3.9。
- **arXiv**：[2610.04596](https://arxiv.org/abs/2610.04596)

#### How Should Teachers Be Prepared? RL on Student-Induced States for On-Policy Distillation (Prep-OPD) (2026-10)
- **简介**：作者 Xiaoyu Ma、Haoyue Liu、Zhichao Wang、Jionghao Zhu、Xiaoying Tang。擅长独立解题的 teacher 未必擅长从 student 的前缀继续推理：当前缀偏离 teacher 路径或含错时，teacher 续写反而更不准。论文提出 **Prep-OPD**，在蒸馏前先用 RL 训练 teacher 适应 student 的推理状态并在出错时纠偏——固定 student 前缀、以最终答案正确性为奖励优化 teacher 续写，再用准备好的 teacher 通过轨迹引导和 token 级监督训练 student。在 8 个数学推理 benchmark 上，以 Qwen3-4B-Instruct-2507 为 teacher、Qwen3-1.7B 为 student，平均准确率比标准 OPD 和最强基线 Relay-OPD 分别高 8.28 与 2.30 个百分点；同一 teacher 复用到 Qwen3-0.6B 也有提升。
- **arXiv**：[2610.04950](https://arxiv.org/abs/2610.04950)

#### Stochastic Teacher Intervention for Agentic On-Policy Distillation (STI-OPD) (2026-10)
- **简介**：作者 Junnan Liu、Linhao Luo、Zhijun Chen、Qianren Mao、Thuy-Trang Vu、Gholamreza Haffari。多轮 agentic 任务中 student 的决策会塑造后续观测，早期错误逐轮累积，使轨迹偏离 teacher 的 rollout 分布，teacher 的 token 级监督变得不可靠。**STI-OPD** 在多轮交互中依据师生策略差异，用 teacher 生成的动作替换 student 提议的动作；通过 KL 估计策略差异并映射为干预概率，再按概率随机采样是否干预，替代阈值式或固定调度的做法，自适应平衡 teacher 控制与 student 探索。针对混合策略轨迹，引入 Importance-Weighted Reverse KL 目标以修正 token 采样不匹配。在工具集成推理和长程交互上，STI-OPD 在所有评测 benchmark 和 student 规模上都优于最强的已有 OPD 基线，消融显示两个组件均有贡献。
- **arXiv**：[2610.10878](https://arxiv.org/abs/2610.10878)

#### GraphOPD: Graph-Augmented On-Policy Distillation for LLM Agents (2026-10)
- **简介**：作者 Bohan Lin、Liyi Chen、Zhuoning Guo、Muyang Li、Qimeng Wang、Yan Gao。现有 agent OPD 按每步师生散度大小分配引导，但在多轮链式决策中该规则失灵：早期漂移进入后续上下文，使 teacher 与漂移轨迹保持一致，且可互换的步骤会产生与结果无关的大散度；实验显示蒸馏最高散度步相比随机选择并无稳定收益。**GraphOPD** 首次把图结构增强引入 OPD：从环境状态变化记录中读出「哪些步骤使能了哪些后续步骤」，构建依赖图，用随机游走平稳分布给每步打分，并将该结构信用与散度信号融合成轨迹内相对 mask，把监督集中在最有价值的步骤上。在 ALFWorld、WebShop、SearchQA 三个规模、11 个基线下，最多领先最强基线 +5.8 pp，且信号可迁移到域外工具集成推理。
- **arXiv**：[2610.08959](https://arxiv.org/abs/2610.08959)

#### Sharpen Without Search: On-Policy Distillation of Sequence-Level Power Distribution (OPPD) (2026-10)
- **简介**：作者 Erfan Baghaei Potraghloo、Seyedarmin Azizi、Arya Fayyazi 等。序列级 power distribution 把每个完整答案概率升到大于 1 的幂再归一化，从中采样可不改参数地提升推理，但需要大量候选打分。论文提出 **OPPD**（on-policy power distillation）：训练中的模型生成候选，由冻结 teacher 的 power 分布通过序列蒙特卡洛采样器加权，同样的概率再用于最大似然更新，使模型单次生成即可产出 sharpen 后的答案。单次生成准确率相对未训练模型在 MATH500 / GSM8K 上最多提升 23.0 / 27.3 点，且比 64 候选的 power sampling 高 2.4 / 3.5 点；不用参考答案的情况下，相比同 checkpoint、同预算 GRPO 在 MATH500、GSM8K、AIME 上高 3.8 / 4.0 / 5.4 点，在 GRPO 之后再用 OPPD 最多再加 9.3 点。
- **arXiv**：[2610.06804](https://arxiv.org/abs/2610.06804)

#### Learning to Revise Reasoning with Segment-wise On-Policy Distillation (Seg-OPD) (2026-10)
- **简介**：作者 Yuxiang Zhang、Ding Cao、Shuting Cui、Lei Wang、Weijieying Ren、Tianxiang Zhao。token 级 OPD 不提供「如何修改 student 某一步」的连贯替代推理步，且当 student 前缀退化时 teacher 监督仍以该前缀为条件，可能强化差的推理模式。受控干预实验发现，把 student 片段替换为 teacher 重写能提升后续推理准确率。**Seg-OPD** 据此用不确定性指标选取 student 片段并获取对应 teacher redraft，训练 student 偏好 teacher 重写而非自己的片段，同时保留稠密 token 级 OPD 监督。在数学推理与竞赛编程上，Seg-OPD 训练的 student 修订成功率更高，相对最强基线在多种模型与任务上平均相对提升 5.22%。
- **arXiv**：[2610.02703](https://arxiv.org/abs/2610.02703)

#### PIVOT: Perplexity-Informed KD-to-RL Transition Scheduling for Vertical-Domain Few-Shot Distillation (2026-10)
- **简介**：作者 Heng Li、Yong Zhang、Ning Cheng 等，已被 Findings of EMNLP 2026 接收。垂直领域少样本分类中，小模型因监督有限难以获得领域决策知识；现有 KD-to-RL 流程多采用全局固定的切换调度，忽略不同样本需要的 teacher 引导量不同。**PIVOT** 根据 teacher 评估的序列困惑度对样本路由：低困惑度样本转入 GRPO 做奖励驱动精修，高困惑度样本继续留在 OPD 中获取领域知识。在 Banking77 和 HWU64 上，同样的 warm-up 后优化步数下，PIVOT 一致优于继续 OPD 与全局同步 OPD→GRPO 的基线，下游性能更强、训练动态更稳定。（abstract 未给出具体数值）
- **arXiv**：[2610.11167](https://arxiv.org/abs/2610.11167)

#### Beyond Timestamps: Decision-Aligned On-Policy Distillation for Long-Horizon Agents (2026-09)
- **简介**：作者 Mingju Chen、Can Lv 等指出，长程 agent 的 on-policy self-distillation（OPSD）存在 **Decision–Timestamp Mismatch**：特权引导对应的功能性决策与学生决策常发生在不同时间步，且一个决策可能跨越多个 timestep，按时间戳局部监督会让上下文与 credit 范围双重错位。论文提出 **AlignOPSD**，遵循「先对齐监督、再分配 credit」：Decision-Aligned Supervision Rectification 在 sibling rollout 的功能匹配上下文中重新为同一学生响应打分以校准局部 teacher 证据；Semi-Markov Hierarchical Credit Assignment 由对应关系变化导出变长决策 span，并把基于结果的 credit 分配到 span 与其中各 turn。在 ALFWorld、WebShop、Search-QA 上用 Qwen2.5-3B/7B 实验，AlignOPSD 在八组对比中全部优于 GRPO 与 StepOPSD，相对 GRPO 提升 5.5–8.7%。
- **arXiv**：[2609.33391](https://arxiv.org/abs/2609.33391)

#### An RL View of OPD: Least Square Policy Distillation for Sample-Efficient LLM Reasoning (2026-09)
- **简介**：作者 Shangzhe Li、Yuxiao Yang、Taylor W. Killian 等从 RL 视角审视 on-policy distillation（OPD），建立 OPD 中 reverse-KL 目标与 KL 正则化策略优化的联系。据此提出 **Least-Square Policy Distillation（LSPD）**，把 value-based RL 中的乐观探索与 off-policy 数据复用引入策略蒸馏，既保持策略多样性，又通过反复利用已采集轨迹提升 rollout 效率；理论上其理想形式在在线探索下达到 $\tilde{O}(\log K)$ regret。实验在六个数学推理基准、多种师生设置上稳定优于现有蒸馏基线，Avg@16 平均提升 +1.59；Pass@k（至 k=64）显示其更好保留多样性；完全 off-policy 的变体仅用前 25% rollout batch 即可达到 vanilla OPD 相当的性能。
- **arXiv**：[2609.35505](https://arxiv.org/abs/2609.35505)

#### Understanding the Synergy between SFT, RLVR, and OPD in LLM Post-Training (2026-09)
- **简介**：作者 Emre Can Acikgoz、Dilek Hakkani-Tür、Shafiq Joty 等研究 SFT、RLVR、OPD 组成多阶段后训练流水线时各阶段之间的相互作用。在 Qwen3 数学/科学推理上，对 9 组师生配对（参数比 2x–53x）的受控实验表明，OPD 的效果取决于师生兼容性而非仅取决于教师规模；短暂的 SFT warm-up 能提升后续 OPD，而经 RLVR 强化的学生在同一教师蒸馏下反而退化；用 RLVR 适配教师则使 OPD 精度随教师新增能力成比例上升。教师适配加学生 warm-up 使相同蒸馏步数下的平均 OPD 精度从 29.2% 提升到 43.8%。此外同等精度下，OPD 为后续 RLVR 留下的初始化优于 SFT，且差距随 RL 算力增大而扩大。
- **arXiv**：[2609.31900](https://arxiv.org/abs/2609.31900)

#### Dense Is Not Enough: Hierarchical Supervision Allocation for Long-Horizon On-Policy Distillation (2026-09)
- **简介**：作者 Yuhao Sun、Binrui Wu 等指出，在长程 agent 任务中对 token 做均匀匹配的 OPD 会错配监督：局部差异大不一定改善未来行为，关键指导可能超出学生当前能力，或在缺少特权输入时无法保持。论文把长程 OPD 表述为分层监督分配，主张有效指导位于「未来效用」与「当前可学性」的交集，并提出由粗到细的 **LENS-OPD**（Locate、Validate、Refine）：Locate 根据学生能力自适应轨迹暴露并提出候选干预决策；Validate 检验该处教师指导是否改善同一学生后续行为；Refine 把有益的引导行为内化到可部署策略，并在已验证 turn 内把 token 级监督集中到关键的师生冲突上。多个长程 agent 基准与师生配置上，LENS-OPD 稳定优于 vanilla OPD 及课程/选择类强基线。
- **arXiv**：[2609.33409](https://arxiv.org/abs/2609.33409)

#### Scaling Properties of Same-Family On-Policy Distillation (2026-09)
- **简介**：作者 Yuntai Bao、Qinfeng Li 等研究 on-policy distillation（OPD）在 weak-to-strong、same-base、strong-to-weak 三类师生设置下的 scaling 性质。发现早期训练都呈现规律的「有效迁移」阶段：held-out 准确率（gold score）近似随 $d=\sqrt{KL(\pi_\theta\Vert\pi_{ref})}$ 线性上升；在所有观察到的 weak-to-strong 配对中，学生峰值 gold score 都超过教师本身，说明小型 RL 专家可通过 OPD 把能力迁移给更大的学生。论文拟合了峰值分数与斜率随师生参数量及教师分数变化的幂律，表明峰值随教师规模提升仅到约学生规模为止，且在相同 gold score 下较小教师迁移更好，故教师分数本身不能决定其监督价值。另探讨两种 OPD 变体、weak-to-strong 的 bootstrapping 和 on-policy 程度的影响。
- **arXiv**：[2609.32722](https://arxiv.org/abs/2609.32722)

#### Dr. OPD: Learning What to Follow for Optimal On-Policy Distillation of Large Language Models (2026-09)
- **简介**：作者 Zhenyu Wang、Tianze Wang、Linjun Zhang、Yifan Hu 指出 vanilla OPD 对所有 token 的教师信号一视同仁，但不同 token 的教师信号对学生性能影响差异很大：有的纠正关键推理错误，有的对最终答案几乎无影响。**Dr. OPD**（OPD Done Right）将最优加权 OPD 表述为双层优化：学生在加权教师监督上学习，权重则被选择以最大化所得学生的期望 reward；通过交替迭代求解，每轮先闭式更新权重再对加权目标做一步梯度。在正则条件下可证明加权更新比 vanilla OPD 获得更高期望 reward。在数学与代码的 strong-to-weak 及同规模蒸馏中稳定优于所有评测基线，strong-to-weak 设置下数学平均分较 vanilla OPD 提升 9.7 分，并使小学生超越其大教师。
- **arXiv**：[2609.38025](https://arxiv.org/abs/2609.38025)

#### Do We Really Need KL Divergence for On-Policy Distillation of Large Language Models? (2026-09)
- **简介**：作者 Wenze Lin、Jiyuan Long、Shenzhi Wang 等发现 on-policy distillation（OPD）可能并不需要 KL 散度：只要更新方向朝向教师即可。具体而言，对教师概率高于学生的 token 给 +1 reward、低于的给 −1，几乎复现 reverse-KL OPD 的训练过程；并且真正关键的只是师生分歧大的一小部分 token 的更新方向，其余 token 即便被拉离教师也不影响训练。基于此提出 **Consensus Multi-Teacher On-Policy Distillation（C-MOPD）**：MOPD 把每个样本路由给单个教师，可能造成跨域能力冲突，而 C-MOPD 让每个样本由所有教师共同监督。实验显示 C-MOPD 在数学与代码基准上稳定优于 MOPD。代码已开源。
- **arXiv**：[2609.33791](https://arxiv.org/abs/2609.33791)

#### Unbiased Top-$k$ Estimation for On-Policy Distillation (2026-09)
- **简介**：作者 Linjian Meng、Siyuan Gan 等关注 OPD 中 reverse-KL 梯度估计问题：只用采样 token 便宜但监督信号有限，使用全词表完整但昂贵；近期的 Top-$k$ OPD（TK-OPD）因丢弃 top-$k$ 之外的概率质量而引入偏差、损害精度。论文提出 **Tail-Corrected Top-$k$ On-Policy Distillation（TT-OPD）**：除选定的 top-$k$ token 外，再利用学生 rollout 中的采样 token，在期望意义上恢复被丢弃的概率质量，从而得到 reverse-KL 梯度的无偏估计，同时保留 TK-OPD 丰富的分布监督与低计算成本。实验显示 TT-OPD 显著优于所测试的其他 OPD 变体（abstract 未给出具体数值）。
- **arXiv**：[2609.34447](https://arxiv.org/abs/2609.34447)

#### Understanding On-Policy Distillation: A Mechanistic Interpretability Perspective via Sparse Crosscoders (2026-09)
- **简介**：作者 Zichao Yu、Qianshuo Ye、Xu Wang、Difan Zou 用稀疏 crosscoder 研究 OPD 究竟把什么蒸馏进学生内部表示。为弥补标准 crosscoder 无法刻画模型对特征「使用方式」变化的缺陷，提出 **swap readout**，独立读取每个学生 checkpoint 的特征激活。在三种 OPD 设置下发现：OPD 既不创造新特征，也不传递教师自身的特征，超过 98% 的学生高频特征激活率变化在 20% 以内。OPD 之前常用的 SFT warm-up 并非增加特征，而是重新加权共享特征：既提前完成了一部分 OPD 的工作，也改变了 OPD 单独难以改变的对话格式、推理风格和数学符号等特征。对直接蒸馏的学生施加这种重加权（不改权重）可使其精度接近 warm-up 学生。结论是 OPD 重新加权已有特征而非获取新特征。
- **arXiv**：[2609.35210](https://arxiv.org/abs/2609.35210)

#### From Self-Distillation to Self-Practice: Privileged Information for Multi-Turn Agents (PSP) (2026-09)
- **简介**：作者 Xingyu Su、Abhishek Kumar、Qing Ping 等。On-policy self-distillation (OPSD) 用以特权信息 (PI) 为条件的同一模型作 teacher 提供 token 级监督。论文发现在多轮 agent 中，这会让 student 学到「有信心却没有信息」的行为，性能不及普通 RL，最差甚至低于未训练基座。提出 **PSP**（Privileged Self-Practice）：保留 PI，但将其从 loss 移到 sampler——当 student 的 rollout 大多失败时，注入分析模型写的简短任务指令重新采样，仍用不变的 GRPO 目标训练，PI 只留在 prompt 中而不进入 loss。在 AppWorld 与 SWE-bench Verified 上、三种 student 模型下均取得最优平均分，相对 GRPO 提升 7%–58%。
- **arXiv**：[2609.29051](https://arxiv.org/abs/2609.29051)

#### Pistis Technical Report (IDRL) (2026-09)
- **简介**：作者 Heyun Chen、Xiaohan Lan、Jiaxi Li 等。Pistis 是基于 Qwen3.6 与 Qwen3.5 的 27B/9B 多模态大模型家族。在大规模多模态 SFT 之后，提出 **IDRL**（Interleaved Distillation and Reinforcement Learning）：在同一训练循环中交替执行 on-policy distillation 与 RL，而非孤立优化或静态联合 loss，从而带来更有效的知识迁移、更好的优化稳定性以及对长程 agentic 轨迹更精确的 credit assignment。产出 Pistis-Thinking（深度多模态推理）与 Pistis-Agentic（加入 agentic 轨迹，多模态搜索尤强）两个变体，两个规模均优于各自基座。另提出 Pistis-Auto-Harnessing (PAH)，在不更新参数、不增加交互预算的前提下自动迭代优化 agent 的推理 harness（abstract 未给出具体数值）。
- **arXiv**：[2609.28554](https://arxiv.org/abs/2609.28554)

#### Reward-Tilted On-Policy Distillation for Acoustic Grounding in Audio-Language Models (RT-OPD) (2026-09)
- **简介**：作者 Kaiyang Li、Shaobo Han、Yue Tian、Shihao Ji（投稿 ICASSP 2027）。音频语言模型 (ALM) 可能利用文本捷径回答问题而忽视声学证据，而 on-policy distillation (OPD) 并不显式区分声学支持与语言可预测性。**RT-OPD**：给定同一问题和 student 生成的文本，冻结 teacher 分别在有/无音频输入下预测下一个 token，二者 log 概率之差定义奖励，用其重塑 teacher 分布后做 reverse-KL 蒸馏，强调音频提供的额外证据。在两个小型 student、三个基准上稳定优于 Vanilla OPD；静音/替换音频实验表明 student 更依赖声学证据。3B 模型在 MMAU 上达 72.72%，为同规模对比中最高，并与若干 7B、8B 模型相当。
- **arXiv**：[2609.28778](https://arxiv.org/abs/2609.28778)

#### Tool-Augmented On-Policy Distillation for LLM Domain Adaptation in Sequence-Based Omics Tasks (TA-OPD) (2026-09)
- **简介**：作者 Jie Ying、Zhefan Wang、Zihong Chen 等。现有多组学序列基准只看分类/回归指标，忽视模型是否理解背后的生物学证据。论文提出首个多组学序列推理基准 OmicsBench（1,160 道专家验证题、六个任务，用专家制定的实例级 rubric 评估证据链），对 17 个 LLM 的评测发现：科学 LLM 序列分类准确率更高，却无法给出有效证据，疑似依赖统计捷径。据此提出 **TA-OPD**（tool-augmented on-policy distillation），一种让序列预测与证据支撑的生物推理对齐的后训练方法。在 0.8B 到 27B 的五个 Qwen3.5 模型上，TA-OPD 稳定增强生物学证据 grounding，并在多数任务上提升预测性能，且收益跨规模保持。
- **arXiv**：[2609.23435](https://arxiv.org/abs/2609.23435)

#### RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning (2026-09)
- **简介**：作者 Yan Yu、Zhengxi Lu 等。多轮 agent 的 RL 每条轨迹只有一个标量 reward，self on-policy distillation（OPD）用带特权 skill 的 self-teacher 提供稠密 token 级监督，但作者发现：特权信息并不必然让 teacher 可靠，且 teacher 监督的收益随训练阶段变化。**RetireOPD** 先用环境 reward 训练一个解耦的、skill 条件化的 teacher，再让不带 skill 的 student 联合 RL 与 OPD 训练；并提出 **Adaptive Retirement**：当 student 与 teacher 的差异不再缩小、且 student 达到 teacher 成功率的目标比例后，student 自行撤掉 teacher，之后仅用 RL 训练。在 1.5B–7B 的 Qwen2.5 上，ALFWorld 成功率比 RL baseline 提升 14.1%–18.8%，WebShop 准确率提升 11.8%–19.0%，且在所有设置下都超过自身的 skill-conditioned teacher。
- **arXiv**：[2609.20784](https://arxiv.org/abs/2609.20784)

#### Beyond Token-Local Imitation: Reward-Compatible Temporal Credit Assignment for On-Policy Distillation (γOPD) (2026-09)
- **简介**：作者 Shiqi Liu、Zeyu He 等。OPD 中，token 级目标稳定但只提供局部监督，序列级目标能捕获未来 credit 却带来依赖 horizon 的方差。本文建立统一的时间 credit 视角，指出实际使用的 token 级 OPD 可解释为序列级 reverse-KL 梯度的时间近似。据此提出 **γOPD（GammaOPD）**：用折扣式时间 credit assignment 在长程监督与优化稳定性之间折中，并具有与 horizon 无关的方差上界；进一步设计 reward-compatible bounded mixing（RBM）机制，把可验证的结果反馈与折扣 OPD advantage 混合，不再纯依赖 teacher。在数学与代码推理上，于 vanilla、size-mismatched 与 multi-teacher 蒸馏设置下均持续优于已有 OPD 方法。
- **arXiv**：[2609.16937](https://arxiv.org/abs/2609.16937)

#### Know When to Stop, Where to Restart: Accelerating Multi-Turn Agentic On-Policy Distillation (STRIDE) (2026-09)
- **简介**：作者 Zhiyu Gui、Kexin Huang 等。OPD 的成本主要来自 student 的自回归 rollout，在多轮 agentic 场景下尤其难以扩展，而已有加速方法按固定离线预算截断监督信号。在 $τ^2$-bench 上的分析发现：有信息的监督集中在每轮的前缀，且跨轮的 teacher 认可度损失在时间上锁定于 student 的第一个错误动作。据此提出 **STRIDE**：自适应早停（当累计 teacher log-prob 低于 OOD 阈值时终止 rollout）加上 prefix buffer（缓存高质量前缀，并从最弱的正确轮次重启生成），形成数据驱动的课程。在 $τ^2$-bench retail 上以 3.73× 加速匹配全轨迹 OPD 并超过 30B teacher；在 AIME 2025 上以 5.10× 加速优于完整 OPD。
- **arXiv**：[2609.14636](https://arxiv.org/abs/2609.14636)

#### Reducing the Output-Mode Gap in Speech Language Models via Joint-Output On-Policy Distillation (JO-OPD) (2026-09)
- **简介**：作者 Daxin Tan、Dehua Tao 等。语音大模型交错生成文本与声学 token 时，相同语音输入下 S2TS 模式内部文本的回答准确率明显低于 S2T 响应，作者称之为 output-mode gap（OMG）。**JO-OPD**（Joint-Output On-Policy Distillation）把模型自身更强的 S2T 策略蒸馏到联合生成中：用 student 生成的 S2TS 轨迹，在每个文本位置由 S2T teacher 基于 student 先前输出的纯文本投影给出 soft target，student 则基于完整交错历史预测，并用 preservation 目标正则化原生非文本预测。在 Step-Audio-2-mini 上，OMG 在 Spoken-MQA 上由 42.87 降至 16.26 个百分点，在语音版 GSM8K 上由 29.72 降至 13.04，S2T 精度基本不变，降幅显著大于匹配的 SFT baseline。
- **arXiv**：[2609.15313](https://arxiv.org/abs/2609.15313)

#### Data-Free On-Policy Distillation: How Far Can We Go Without External Data? (DF-OPD) (2026-09)
- **简介**：作者 Gengsheng Li、Mao Zheng 等。OPD 被广泛用于前沿模型后训练，但其对训练问题的依赖程度并不清楚。在两种单 teacher OPD 设置下，作者发现用 8 个真实 prompt 训练即可达到与 17k 题相当的表现，难度与初始蒸馏 gap 差异很大的数据集结果也相近；可能原因是重复采样即可暴露足够的 teacher 监督，且 OPD 迁移的是可泛化的推理能力而非数据集特定知识。进而提出 **data-free OPD（DF-OPD）**：由系统自行生成训练问题，仅用 64 个无种子的自生成问题即可在单 teacher 设置下达到与全量数据 OPD 相当的表现；多 teacher 下，1k 个生成问题相当于约 7k 条真实后训练样本。
- **arXiv**：[2609.14193](https://arxiv.org/abs/2609.14193)

#### Negative Self-Distillation: Learning to Reason by Avoiding Flaws (NSD) (2026-09)
- **简介**：Rongcan Pei、Zhepei Wei、Shuyao Xu、Xinyu Zhu、Wei-Lin Chen、Yu Meng。指出 On-Policy Self-Distillation (OPSD) 让模型借助 ground-truth 解法等特权信息充当自己的老师，已成为 LLM 自我改进的流行范式，但近期结果显示它会在复杂推理任务上严重损害性能——强迫 student 模仿一条基于特权信息、人为自信的推理轨迹，反而压抑了不确定性表达，并惩罚了求解难题所必需的探索与自我纠错行为。提出 **NSD**：不再模仿特权解法，而是通过远离有缺陷的推理来优化——不依赖 ground-truth 答案或外部监督，而是用模型自身生成一个 question-specific 的 negative condition（例如让它扮演 "careless reasoner"），再把 student 分布推离这个自生成的 negative teacher。作者指出朴素套用 unlearning 目标有问题，因为缺陷推理 token 与基础语言 token 相互混杂，不加区分地惩罚会灾难性损害模型的语言能力；解决办法是设计动态 gating 机制自动识别并隔离 reasoning-critical token，使梯度更新只针对行为缺陷而保留语言先验。实验中 NSD 持续优于 OPSD 及其他 label-free 自举 RL 基线（abstract 未给出具体数值）。
- **arXiv**：[2609.11699](https://arxiv.org/abs/2609.11699)

#### What Matters in On-Policy Distillation? A Perspective on Data Efficiency and Data Selection (2026-09)
- **简介**：Zhinan Hou、Jiaqi Zhang、Xunliang Cai、Keyou You（清华大学与美团）系统研究 On-Policy Distillation（OPD）中被忽视的数据侧机制。作者先考察极端设定 **1-shot OPD**（仅用一个样例训练），发现它在所有采样到的训练样例上都稳定有效，且更难的样例往往带来更大增益；进一步分析指出 student 的提升并非来自高 token entropy，而是来自难题天然生成的更长 CoT——更长的 CoT 有助于在长推理 horizon 上与 teacher 保持更紧的对齐，并学到短 CoT 里通常缺失的关键思维模式（如 ``Alternatively'' 这类 reflection）。据此提出只挑选难样例的简单数据选择方法，甚至完全超出 teacher 能力的「unsolvable」样例也能被有效利用；在 1.5B–7B 四个模型上，仅用 8 个精选难样例训练即可匹配 17K 数据集基线。本文属 §3.4 的数据效率机制分析，是 OPD 数据效率的关键对照。
- **arXiv**：[2609.05198](https://arxiv.org/abs/2609.05198)

#### CompassOPD: Cross-Family On-Policy Distillation via Within-Family Likelihood Shifts (2026-09)
- **简介**：Naibin Gu、Qingyi Si、Chenxu Yang、Chuanyu Qin 等（中科院信工所）发现 OPD 在 teacher 与 student 同族时效果很强，但在跨模型族（cross-family）设定下即使做了 tokenizer 对齐也会退化，明显更强的外部 teacher 几乎带不来额外收益。作者把跨族 OPD 信号分解为两部分：低能力 teacher-族 reference 与 student 之间的 offset，以及该 reference 到强 teacher 的同族 log-likelihood shift；标准 OPD 把两者一起迁移，导致 offset 主导更新方向、掩盖了真正与 teacher 能力提升相关的变化。提出 **CompassOPD**，去掉 offset 只迁移 within-family shift，同时用冻结的 student reference 把更新锚定到 student 初始 policy，使 teacher 侧与 student 侧的变化各自在其模型族内度量。在三个 student 族与多个 teacher 族上，平均推理准确率相对标准跨族 OPD 最高提升 5.50 个点；对 MoE teacher 还可通过降低 expert activation 直接从 teacher checkpoint 构造 reference，省去单独的 reference checkpoint 并仍保持相对 OPD 的 3.43 点增益。
- **arXiv**：[2609.10154](https://arxiv.org/abs/2609.10154)

#### TV-Regulated OPD: Direction Matters in On-Policy Distillation (2026-09)
- **简介**：Han Xiao、Yifan Niu、Dongyi Liu、Chang Luo、Jia Li 针对主流 OPD 监督信号方差大、噪声高、训练不稳定的问题，系统排查了「究竟什么真正影响性能」以及不稳定背后的机制。两个核心发现是：只保留 token-level advantage 的符号（方向）就足以取得与标准 OPD 相当的性能；更平滑且有界的 advantage 能在不牺牲性能的前提下稳定训练过程。据此用 total variation（TV）对 advantage 做整形，提出 **TV-OPD**；由于 advantage 有界且随训练衰减，TV-OPD 表现出稳定的训练动力学与平稳的后期性能。多种设定下的实验显示 TV-OPD 在训练后期一致取得更好性能与更低方差（abstract 未给出具体数值）。本文兼具 §3.4 的方向性机制分析与正则化方法两重定位。
- **arXiv**：[2609.08341](https://arxiv.org/abs/2609.08341)

#### RISE: Recursive Improvement via Self-Extrapolating Policy Distillation (RISE) (2026-09)
- **简介**：Yang Li、Semih Yavuz、Shafiq Joty（Salesforce Research）指出 OPD 虽能提供 dense 的逐 token 监督，但受 teacher 质量瓶颈制约：外部 teacher 存在分布不匹配，而依赖 privileged conditioning 的 self-distillation 又受 in-context learning 能力限制。提出 **RISE**，直接从模型自身的 RLVR 训练轨迹构造一个合成 teacher——在参数空间或输出 logit 空间中，对当前 checkpoint 与一个滞后 anchor 之间的位移做外推（extrapolation），把稀疏的 outcome 诱导的参数更新转换为 dense 的 token 级目标，无需任何外部模型或 privileged conditioning。RISE 让 RLVR 与 OPD 形成互补闭环：outcome reward 把外推方向锚定到正确推理，外推出的 teacher 再细化 token 级决策；且 teacher 随 student 每轮刷新，使蒸馏成为递归自我改进机制而非一次性压缩。在数学推理、多领域 STEM、代码生成与多轮 agentic 任务上，RISE 在所有设定下均优于纯 RLVR 训练与 on-policy self-distillation（abstract 未给出具体数值）。
- **arXiv**：[2609.05295](https://arxiv.org/abs/2609.05295)

#### Beyond Verified Answers: Solver-Informed Self-Distillation for Bootstrapping Operations Research Language Models (SOLID) (2026-09)
- **简介**：Rui Zhu、Minglong Cao、Chenyu Zhou、Jianghao Lin、Dongdong Ge 针对训练 LLM 做 operations research（OR）建模的三个痛点：训练依赖人工专家或更强模型校验的合成 formulation 而难以规模化监督；credit assignment 要么粗（outcome reward 只评整条轨迹、无法定位出错的建模决策）要么贵（process-level 监督需额外 evaluator）；privileged self-distillation 因使用部署时不可得的 solver 上下文而引入 style mismatch。作者发现模型可以仅凭自身 rollout 产生的 solver-artifact 反馈自我提升，从而把 self-distillation 变成无需 evaluator 的 dense 监督来源，提出 **SOLID**：执行多条 rollout 的候选程序、对其目标值聚类、取多数组的 artifact 作为 pseudo-reference，再用 group-relative advantage 与 dense 自监督信号更新模型。在多个 OR 基准上，SOLID 对通用模型与 OR 专调模型的求解准确率都优于只用 outcome 的 group-relative 训练（abstract 未给出具体数值），说明 solver artifact 可支撑无需可信答案的规模化自我提升。
- **arXiv**：[2609.09957](https://arxiv.org/abs/2609.09957)

#### Rethinking On-Policy Distillation of Large Language Models II: One Training Example (2026-09)
- **简介**：Zixuan Fu、Bingxiang He、Yuxin Zuo、Zhiyuan Liu 等（清华大学等）把 On-Policy Distillation（OPD）推到「数据极小化」的极限：只用单个 query 训练。结果是 one-shot OPD 能持续提升数百步，并在多个任务域与模型族上回收 full-data OPD 的大部分收益。作者用「训练中访问到的状态」和「student 与 teacher 对齐的速率」解释这一现象，定义 **state coverage**（full-data OPD 访问的状态中被某 query 集的 rollout 触达的比例）：单个 query 已达 71.5%，其中大部分在前 100 步内完成；语义上互异的 query 越多，coverage 与验证准确率同步上升，16 个 query 达 98.9% 并追平全量数据训练。但无论训一个 query 还是整个数据集，对齐速度都以相近节奏放缓，即使状态集固定也要数百步才被吸收——结论是 OPD「数据过剩而算法饥饿（data-overfed but algorithm-starved）」。state coverage 结论延伸到多 teacher 场景：每域 16 个语义多样 query 即可匹配 full-data MOPD；压力测试中内容稀薄的模板与跨域 WildChat query 也接近真实 query 基线，说明任务内容与其诱导的状态覆盖可以脱钩。本文属 §3.4 的机制分析，重新审视 OPD 近期成功背后的数据与机制归因。
- **arXiv**：[2609.04172](https://arxiv.org/abs/2609.04172)

#### Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR (2026-09)
- **简介**：Boyan Li、Bingsen Chen、Chenghao Yang、Xi Ye 等对比了 RLVR 与 On-Policy Distillation（OPD）两条推理 LLM 后训练主线的组合方式。以往工作用 OPD 的 dense token-level 监督补 RL 的稀疏 reward，并在单步内融合两种信号——或做加权相加，或用 teacher 对 RL advantage 做重标定（teacher-modulated rescaling）。本文表明一个简单的两阶段方案 **OPD-then-RL** 在逻辑与数学推理基准上一致优于纯 OPD、纯 RLVR 以及所有此类联合基线，并通过 pass@$k$ 行为、学习动态与参数更新给出统一解释：OPD 扩大 student 对 teacher 可支撑解空间的覆盖，RL 在该支撑集内做锐化，而联合优化会使两种信号相互干扰。实用配方上，作者发现 OPD 的验证分数是切换到 RL 时机的关键信号，且 OPD 比 SFT 是更好的 RL 冷启动（abstract 未给出具体数值）。本文是 OPD 与 RLVR 交互关系的关键对照，属 §3.4 的序贯性机制分析而非新算法组件。
- **arXiv**：[2609.04108](https://arxiv.org/abs/2609.04108)

#### Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation (TGOPD) (2026-09)
- **简介**：Zhiwei Zhang、Zechen Sun、Fei Zhao、Kam-Fai Wong 等指出 vanilla OPD 在所有 prompt 上一律采纳冻结 teacher 的 dense token-level 监督，却从不检查 teacher 在该 prompt 上是否可靠；由于 reverse KL 是 mode-seeking 的，一个「自信但错误」的 teacher 会诱发强而误导的更新，而 entropy 或师生 likelihood 一致度这类分布式代理只度量不确定性 / 一致性，并不直接验证结果正确性。提出 **Teacher-Gated On-Policy Distillation (TGOPD)**：原则是 dense 监督被采纳前先在 prompt 级验证 teacher 可靠性——用少量经 verifier 打分的 teacher probe 估计可靠性，通过检查的 prompt 独占式路由到 dense OPD，未通过的转由 verifier-grounded GRPO 处理。在 4B 与 35B student、数学 / 代码 / 指令跟随三域上，TGOPD 在全部六个单域设置上优于 vanilla OPD，多域训练下两个规模的七基准均值也更高；由于把本来闲置的 teacher 算力用于可靠性估计，异步 OPD 中 teacher 节点 GPU 利用率从 9.8% 提升到 78.9%（4B 单域实测）。
- **arXiv**：[2609.02998](https://arxiv.org/abs/2609.02998)

#### Learn from Whoever Is Right: Answer-Verified Multi-Teacher Distillation for Multi-Domain LLMs (MT-SDPO) (2026-09)
- **简介**：Xixiang He、Xingming Li、Baiqi Wu、Qingyong Hu 等针对「把多个单域 RL 训出的能力合进一个可部署模型」的难题指出：已有做法按 domain 标签把样本路由给对应 teacher，但域专长只在平均意义上成立——匹配的 teacher 在具体样本上未必对，别的域的 teacher 有时反而对，因此可靠 teacher 必须逐样本识别而非逐域指定。提出 **Multi-Teacher Self-Distillation Policy Optimization (MT-SDPO)**，一种把若干冻结 teacher 统一进单个 student 的 on-policy distillation 方法，含三个部件：(1) self-anchor，用同组内一条正确的 rollout 监督当前 rollout；(2) answer-verified eligibility，teacher 只有在自己的答案通过 verifier 时才有资格监督该样本；(3) privileged distillation，把 anchor 与所有已验证反馈合并进一个只有 EMA self-teacher 能读、student 读不到的上下文，从而部署时仍只保留一个 policy。在三个模型族的五个 student 上，MT-SDPO 把 Qwen3-8B 的最弱域提升 14.79 分、域间差距收窄 74.7%，优于「每域各配一个匹配 teacher」的平衡性。代码已开源。
- **arXiv**：[2609.02548](https://arxiv.org/abs/2609.02548)

#### Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement (2026-08)
- **简介**：Yi Ding、Ruqi Zhang 质问 On-Policy Distillation（OPD）到底有没有在「蒸馏」：OPD 中 teacher 打分的是对它而言本质 off-policy 的 student 轨迹，其监督可靠性、以及 student 收益的真正来源都不清楚。作者定量分析训练中的 teacher 监督，发现噪声占比可观且随 teacher 规模增大而上升；但 student policy 对这种噪声不敏感——保留或剔除噪声监督都收敛到相当的性能。进一步拆解收益来源发现学习集中在低 log-probability token 上，且用单一固定的负 advantage 就能匹配 teacher 给出的 advantage，说明 OPD 很大程度上是在压制低 log-probability token，这一步根本不需要 teacher。据此提出无监督信号的 **On-Policy Self-Adaptation (OPSA)**：用 entropy 自适应的负 advantage，在高 entropy 位置给更强学习信号，压制尾部 token 并把概率质量在头部 token 间均匀再分配。相对 Qwen3-1.7B base，OPSA 在 AIME24 上 Avg@32 提升 35.41 分（相对增益 263%），在三个基准上 Pass@32 均翻倍以上，并在 AIME24 Avg@32 上超过 OPD 16.77 分。本文兼具 §3.4 的病理分析与由此导出的替代方法。
- **arXiv**：[2608.31046](https://arxiv.org/abs/2608.31046)

#### When Teacher Guidance Misleads: Reward-Aligned On-Policy Distillation (RA-OPD) (2026-08)
- **简介**：Siyuan Gan、Yuhan Li、Xiran Wang、Jing Huo 等指出 OPD 中 teacher 对 student 自生成前缀的指导并不总可靠：训练本应把模型推向更可能正确、即 outcome reward 更高的响应，但 teacher 有时会抑制 student 走向正确轨迹、或把它推向错误轨迹，这类与 outcome reward 错配的指导会误导优化并最终损害性能。提出 **Reward-Aligned On-Policy Distillation (RA-OPD)**：核心是只保留那些「所诱导的更新方向确实把 student 推向正确轨迹或抑制其走向错误轨迹」的轨迹——对每条采样轨迹检查其 trajectory-level 蒸馏回报是否与 outcome reward 一致，过滤掉不一致的轨迹，因而不引入额外计算开销。在 Qwen3 与 DeepSeek-R1 系列模型上、跨七个数学基准与三个代码基准，RA-OPD 明显优于标准 OPD 及其他被测 OPD 变体（abstract 未给出具体数值）。
- **arXiv**：[2608.27960](https://arxiv.org/abs/2608.27960)

#### SpikeOPD: Stable On-Policy Distillation for Autoregressive Spiking Language Models (2026-08)
- **简介**：Enqiao Lu、Xingrui Yu、Yiwei Fu、Yang You、Ivor Tsang 等面向能效导向的 spiking language model：从零训练有能力的 SNN 语言模型很难，实际路径是通过 KD 做 ANN-to-SNN 迁移，但已有迁移方法在固定语料前缀上蒸馏，而自回归推理条件于自生成前缀，造成 prefix-source mismatch——表现为与 ANN teacher 的 output-policy 失配，以及自生成前缀与匹配语料前缀之间的内部 spiking 动力学 drift。作者先做受控压力测试评估只用 teacher full-KL 的 Vanilla OPD，观察到它会出现延迟性的 rollout-feedback 崩塌，说明单靠 on-policy 覆盖并不保证稳定适配。据此提出 **SpikeOPD**：以 full-KL teacher 校正压低 output-policy 失配，同时用 matched-prefix policy anchoring 在相同前缀上约束 policy 相对冻结参考 SNN 的偏移，并以 layerwise spike 正则限制 on-policy 适配期间的放电率偏差。三个规模上相对对应的 KD SNN 平均准确率分别提升 0.8、1.7、2.9 分（0.125B / 0.35B / 1.3B），且保持其稀疏计算特性。
- **arXiv**：[2608.27857](https://arxiv.org/abs/2608.27857)

#### On-policy Distillation with Verifiable Reward (OPDVR) (2026-08)
- **简介**：作者 Wenze Lin、Jiale Zhao、Xitai Jiang、Songde Rao 等（据开源仓库 LeapLabTHU 推断为清华大学 LeapLab 团队，含 Shenzhi Wang、Gao Huang）针对 RLVR 与 OPD 的互补缺陷：RLVR 只有稀疏的 task-level 反馈，OPD 虽提供 dense token-level 指导但忽略整条轨迹的正确性，性能被 teacher 上界锁死；已有融合方案多依赖加权组合或启发式切换，引入额外超参与权衡。本文提出 **OPDVR**，不新增任何超参地把两者统一：先基于轨迹正确性重写 sampled-token OPD 的 implicit reward，再用 ReLU gating 保证正确轨迹的 reward 非负、错误轨迹非正，使蒸馏信号与任务成败对齐同时保留 teacher 的分布性指导。该改写把 sampled-token OPD 变成了一个严格意义上的 RLVR 方法，因而可与任意 policy gradient 算法（如 GRPO）直接组合；在六个推理 benchmark 上 OPDVR 一致优于标准 OPD（abstract 未给出具体数值）。
- **arXiv**：[2608.24696](https://arxiv.org/abs/2608.24696)

#### A Token-Level Analysis of Sampled-Token Reverse-KL On-Policy Distillation (SuRe) (2026-08)
- **简介**：作者 Bing Shao、Jiazheng Zhang、Long Ma、Yujiong Shen 等追问一个尚不清楚的问题：OPD 用冻结 teacher 的 token-level 信号监督学生自己的轨迹时，sampled loss 究竟如何把更新量分配到各 token 上。本文解析 reverse KL 的 per-token K2 估计量相对学生 logit 的梯度，证明其 $\ell_1$ 范数可分解为「teacher 与 student 对数概率差的绝对值」乘以「一个学生侧 softmax 因子」，后者随被采样 token 在学生下概率越低而越大；在其数学蒸馏实验中这些 per-token 范数高度不均匀，低学生概率的 token 占据了范数总和中不成比例的份额，且这些 token 同时富集了大的师生 gap。作为该分析导出的轻量干预，作者研究 **Surprise-aware Reweighting（SuRe）**——一个 detached、有界的加权规则，进一步放大这一既有的分配倾向；在两个 Qwen3 学生规模上 SuRe 相对 vanilla OPD 改进了多项数学指标，且在所选 out-of-domain benchmark 上未见明显退化（abstract 未给出具体数值）。作者自述首要贡献是对 K2 估计量下 reverse-KL OPD 的梯度级刻画，SuRe 只是一个经验实例，因此属 §3.4 的机制/token 级分析。
- **arXiv**：[2608.25643](https://arxiv.org/abs/2608.25643)

#### Preserving General Capabilities during Domain Specialization with Uncertainty-Calibrated MOPD (2026-08)
- **简介**：作者 Ziyuan Liu、Jiao Ou、Jian Liang、Ruiming Tang 等研究垂域特化与通用能力的取舍：把大模型专门化到垂直领域会提升领域行为，却常损伤推理、代码、指令遵循与创意写作等通用能力。本文在 Multi-Teacher On-Policy Distillation（MOPD，学生在自采样轨迹上同时受领域 teacher 与通用 teacher 监督）框架下指出两个局限——普通 on-policy 采样很少暴露具有大正 teacher-student advantage 的 token，且仅凭 advantage 的符号无法判定该更新方向是否可靠。提出 **uncertainty-calibrated MOPD**：dual-temperature sampling 拓宽候选轨迹池，positive-advantage-density filtering 挑出正学习信号更强的轨迹，再由 centered log-likelihood（CLL）filtering 计算 entropy 校准的 teacher-endorsement 分数、按「方向与 endorsement 是否一致」概率性保留 token 级更新。在角色扮演与医疗两个垂域特化实验中，通用能力平均分相对标准 MOPD 分别提升 4.73% 与 10.84%，同时保持垂域性能；消融与诊断分析进一步确认增益并非仅来自更大的 rollout 预算，且轨迹级与 token 级机制各自命中了其针对的失效模式。
- **arXiv**：[2608.26735](https://arxiv.org/abs/2608.26735)

#### D$^3$-MOPD: Adaptive Dynamic Domain ScheDuling for Efficient Multi-Teacher Distillation (2026-08)
- **简介**：作者 Zechen Sun、Zhiwei Zhang、Fei Zhao、Juntao Li 等针对 MOPD（在学生自己的 rollout 上最小化各领域 reverse-KL，把多个领域专家 teacher 蒸馏进单一学生）的一个被忽视事实：各领域收敛速率差异极大，有的早早 plateau、有的直到训练预算用尽仍在提升，而现有做法在训练前就固定各领域数据配比，因此在快收敛领域浪费算力、在慢收敛领域训练不足。提出 **D$^3$-MOPD**，一个零额外开销的调度器，直接复用训练中已产生的 per-domain reverse-KL 信号在线调整领域配比：一个 off-process watcher 异步运行在训练进程之外，周期性跟踪每个领域的 KL 轨迹、估计剩余 headroom 与当前改进速率，据此调整领域采样比例而不改动核心训练循环，可自然扩展到任意领域数目，且领域越多、收敛模式越多样，预期收益越大。在由四个领域专家 teacher 蒸馏的 Qwen3.6-35B-A3B 学生上，D$^3$-MOPD 闭合了 97% 的平均师生性能差距（vanilla MOPD 为 63%），以约 3$\times$ 更少的 rollout step 达到同等峰值性能，并在七个 benchmark 中的三个上超过对应的专家 teacher。
- **arXiv**：[2608.24987](https://arxiv.org/abs/2608.24987)

#### OPDSearch+: On-Policy Distillation with RL Refinement for Search-Augmented Reasoning (2026-08)
- **简介**：作者 Qinglin Ye、Zhiyuan Gu、Jingjie Xia、Yiheng Zhang 等针对小模型难以胜任 search-augmented reasoning 的问题，指出从已训练 teacher 做 OPD 存在两个障碍：高质量多轮检索轨迹依赖动态 retriever 响应，大规模收集 SFT 数据代价过高；而为任务专门训练 teacher 成本高昂，直接用未经任务微调的现成 teacher 做 OPD 又把学生锁在 teacher 的性能上界并伴随严重训练不稳定。提出 **OPDSearch+**，首个无需 teacher 微调的检索增强推理蒸馏范式，并给出关键观察：teacher 的作用是重塑学生的 policy 分布，使后续 RL 收敛到单纯 RL 无法到达的更优解。第一阶段学生与真实搜索引擎交互，以 per-position forward KL 目标被蒸馏，从而在完全不做任务特化 teacher 训练的前提下迁移推理分解与证据整合能力；第二阶段 RL 在这一更丰富的行为基础上继续精炼。在七个 QA benchmark 上，3B 模型的 OPDSearch+ 一致超过此前所有 3B RL 基线，HotpotQA 提升 13.1%、2WikiMultihopQA 提升 8.5%。
- **arXiv**：[2608.24310](https://arxiv.org/abs/2608.24310)

#### STAR-OPD: Structured Aspect-Cascade-Aware On-Policy Reward Distillation for ABSA Quadruple Extraction (2026-08)
- **简介**：作者 Tong Sun、Mingyang Ma、Jiayang Yu 研究 ABSA 四元组抽取（在常含多个细粒度情感元组的评论上联合预测 target、aspect、opinion、sentiment）的蒸馏问题：大 CoT 模型表现良好但难以蒸馏进可部署的小模型。作者识别出一个任务特有的失效模式——学生在 target-aspect 交界处的错误会产生结构上非法的状态（target-aspect 绑定断裂、幻觉 target），进而污染下游预测；而传统 off-policy distillation 只在 teacher 生成的轨迹上训练，对推理时真正主导的「学生自身诱发的结构状态」几乎没有监督。提出 **STAR-OPD**，在通用 on-policy distillation 之上为 ABSA 四元组抽取注入 cascade-aware、set-structured 的 reward：在学生 rollout 上训练，reward 直接针对绑定一致性、target grounding 与细粒度 aspect 消歧。在 E-ABSA20K 与 SemEval-2014 上 STAR-OPD 一致优于 off-policy 与通用 on-policy 基线，降低 target 幻觉并在结构困难样例上大幅提升；以 Qwen3-4B 为学生时明显缩小师生差距并提升推理效率（abstract 未给出具体数值）。
- **arXiv**：[2608.20831](https://arxiv.org/abs/2608.20831)

#### Beyond Imitation: Filtering On-Policy Distillation by Reasoning Progress (R2-OPD) (2026-08)
- **简介**：作者 Chen Yang、Haiyuan Wan、Rengrong Xiong、Yize Chen 等指出 On-Policy Distillation（OPD）隐含假设「teacher 导出的 reward 是 reasoning progress 的合适代理」，因而在策略优化中对所有 teacher 反馈一视同仁；但作者观察到二者常相冲突——有明确推理推进的步骤仅因偏离 teacher 输出就拿到更低的 distillation reward。**R2-OPD**（Reasoning-Progress-Aware Reward Filtering）在同一条轨迹内对 reasoning span 构造两套排序，一套来自 teacher 导出的 reward，另一套来自独立估计的 progress reward；当两套排序不一致时选择性抑制该处的 distillation reward，从而削减与推理进展相冲突的监督、同时保留有效的 teacher 指导。作者报告相对标准 OPD 在推理性能上有一致提升（abstract 未给出具体数值）。
- **arXiv**：[2608.19408](https://arxiv.org/abs/2608.19408)

#### Beyond Teacher Likelihood: Group-Calibrated On-Policy Distillation for Long-Context Reasoning (GC-OPD) (2026-08)
- **简介**：作者 Zhu Zhang、Jixun Wang、Xiaoang Xu、Xiaorong Wang 等针对长上下文场景下 OPD 的失效模式：token-level 的 teacher 支持会偏好「局部看似合理」但遗漏散落于长输入中的证据、或违反全局任务约束的 response，而任务 verifier 是在 response 级评估完成度并可给出反映部分成功的分级 reward。作者在两个长上下文证据聚合任务的固定 response 上做诊断，发现随输入长度增加，轨迹级 OPD 分数与 verifier reward 的一致性逐步下降，即存在 teacher-verifier 分歧。**GC-OPD** 在每个 rollout group 内分别归一化 verifier reward 与轨迹级 OPD 分数，用二者之差作为带符号的 teacher-verifier 分歧残差，再由 relative-advantage-based credit assignment（RACA）按 token 的相对 OPD advantage 把该轨迹级残差分摊到 token，同时保留原始 OPD 信号。在五个长上下文 benchmark 上，Qwen3-4B 与 Qwen3-8B 官方 checkpoint 的五项均值从 29.08 提升到 40.47、从 35.12 提升到 44.65，同设定下 vanilla OPD 分别为 39.31 与 43.56；消融显示带符号残差优于「额外 OPD 项」或「直接加 group 归一化 verifier reward」，RACA 也优于 token 上均匀分配。代码已开源。
- **arXiv**：[2608.19181](https://arxiv.org/abs/2608.19181)

#### Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation (2026-08)
- **简介**：作者 Huan-ang Gao、Haohan Chi、Yong Yan、Shiyuan Feng 等（清华大学 AIR，含 Wei-Ying Ma、Ya-Qin Zhang、Hao Zhou）研究多 teacher on-policy distillation（M-OPD）把多个领域专精 RL 专家合并为单一通才 student 时的优化动力学，并补上缺失的可复现配方。作者在 SmolLM3-3B-Base 上搭建带 oracle routing 的受控 M-OPD benchmark，以隔离「能力融合」与「路由歧义」，发现显著的 capability integration gap：标准 M-OPD 相对领域路由 oracle ensemble 只拿到 35.6% 的可用空间，指令遵循这类简短任务甚至严重退化并提前停滞。关键结论是该失败**并非来自梯度冲突**，而是 token 级优化预算的严重错配，由三个正交因素驱动：跨领域的结构性序列长度差异、学习速率不均导致的收敛漂移、以及异步策略更新带来的多步 reward staleness。**Open-MOPD** 以 token-share balancing、gap-aware 动态预算分配与 student reward refresh 对症修复，把 headroom 恢复率从 35.6% 提升到 83.4%，并在学术可负担算力预算下全面开源端到端后训练配方、训练轨迹与评测套件。
- **arXiv**：[2608.19098](https://arxiv.org/abs/2608.19098)

#### Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models (2026-08)
- **简介**：作者 Zhaoyi Li、Deyang Kong、Yuan Wei、Evan Yang 等指出 OPD 的泛化行为仍缺乏理解——多数研究只在单一领域、且在贴近训练数据的 benchmark 上评估。本文做受控研究，每次只变动一个泛化因子，覆盖领域内分布偏移、跨领域迁移与多 teacher 设定。主要发现：OPD 迁移的是 teacher 的**推理行为**而非其对特定题目的答案，训练题难度几乎无关，连 teacher 自己都做不对的题也仍然有用；迁移强度强烈依赖师生的同源关系——同源师生对能让 student 在跨语言、跨推理长度乃至跨领域上都逼近 teacher，而异源师生对基本只拟合被训练的分布。这种「广覆盖」是双刃剑：既然把 prompt 路由到领域专家并不能把每个 teacher 的影响限制在其领域内，组合多个 teacher 就会产生依赖混合比例的能力此消彼长（seesaw）。本文不提新训练方法，属 §3.4 的机制/泛化性质分析，是诊断 multi-teacher OPD 的关键对照。
- **arXiv**：[2608.16647](https://arxiv.org/abs/2608.16647)

#### Step-Level On-Policy Distillation: Interpolating Between On-Policy Distillation and Supervised Fine-Tuning (SOPD) (2026-08)
- **简介**：作者 Changhui Sun、Lanbo Liu、Hang Lei、Tong Ling 等指出标准 token 级 OPD 只能沿一条出错的 student 轨迹给出碎片化修正，无法展开一条完整正确的修复路径。**SOPD**（Step-Level On-Policy Distillation）把 SFT 的长程修正与 OPD 的 on-policy 优势结合，在完整的 student 生成轨迹上提供 step 级监督；作者证明在 step 长度的两个极限下 SOPD 分别退化为 SFT 或逼近 OPD。相比 SFT，SOPD 的 teacher response 以 student 轨迹为条件，因而更贴合 student 实际访问的状态；相比 OPD，它给出更长视野的修正而非碎片化的 token 级指导。在推理与 agent 任务上 SOPD 均明显优于常规 SFT 与 OPD，例如在 ALFWorld 上平均成功率比 vanilla OPD 高 13.4 个点。
- **arXiv**：[2608.16333](https://arxiv.org/abs/2608.16333)

#### SimpleOPD: Simple Tokenizer-Agnostic On-Policy Distillation for Long-Context Reasoning (2026-08)
- **简介**：作者 Haonan He、Haodi Lei、Yun Luo、Haoran Zhang 等处理「长上下文推理 teacher → 短上下文 student」这一 OPD 设定下的四类实际困难：tokenizer mismatch、师生分布不匹配、response 长度爆炸与训练不稳定，实验以把长上下文推理模型 SU-01 的证明推理能力迁移给短上下文 student 为载体。**SimpleOPD** 的做法是在共享文本空间中做 OPD，只对齐在 student 与 teacher 两套 tokenizer 下占据完全相同文本跨度的 token；为抑制生成过长与频繁截断，引入 student reference KL loss 并对 `</think>`、`<|im_end|>` 等特殊终止 token 屏蔽其 advantage，约束 student 不过度偏离初始策略，从而缓解师生分布不匹配并让长度平稳增长。在同族与异族 student（Qwen3、Qwen3.5、Intern-S2、GLM-4.7、Gemma-4）上数学推理尤其是自然语言数学证明均有一致增益：Intern-S2-Preview 在 ProofBench 上提升 21.2 个点达到 55.2，超过 Gemini-2.5-Pro；在 HLE、HiPhO 等科学 benchmark 上同样有提升，说明 OPD 迁移的推理能力可泛化到数学训练域之外。
- **arXiv**：[2608.14277](https://arxiv.org/abs/2608.14277)

#### Trust Is Not Enough: Influence Calibration for On-Policy Self-Distillation in Agentic RL (ICSD) (2026-08)
- **简介**：作者 Qizhen Lan、Xi Xiao、Xiangchen Guan、Mengchen Fan 等指出 on-policy self-distillation（OPSD）虽能由特权 self-teacher 在策略自身轨迹上给出稠密 token 级监督，但现有方法主要按 teacher trust 分配这份监督，而 trust 并不能说明「强调某个 token 是否真的有利于当前策略目标」；作者称之为 trust-utility mismatch。**ICSD**（Influence Calibration for Self-Distillation）对每个被监督 token，测量其 importance-weighted RL surrogate 贡献对一次 teacher 指向的输出扰动的一阶响应，再用 batch-adaptive calibration 把这个非平稳信号转成有界的分配权重，同时保持每个 action turn 内原有辅助损失的总质量；这些权重是 detached 的、只作用于 distillation loss，无需额外前向。在 ALFWorld、WebShop、Search-QA 上，于 GRPO 与 GiGPO 下、跨 1.5B–7B 两个模型族，ICSD 在所有匹配的聚合指标上优于仅按 trust 分配；7B 时 ALFWorld 成功率 96.1%、WebShop 得分 93.1。冻结 batch 分析显示 ICSD 把分配给「与目标相反」token 的 teacher 支持质量从 60.1% 降到 37.8%，与 RL 梯度的余弦兼容性提升 0.192。
- **arXiv**：[2608.14945](https://arxiv.org/abs/2608.14945)

#### DART-SD: Diamond-topology Aware Retrieval and Tuning for Self-Distillation of Multi-Turn Tool-Calling Agents (2026-08)
- **简介**：作者 Hangrui Xu、Jiarui Wang、Yang Yang、Chuanbo Zhu 等指出多轮 tool-calling agent 的训练受限于对完整轨迹的整体模仿：当任务包含多个次序无关的子目标时，最优解空间构成一个庞大的组合式「钻石格」（diamond lattice），把这种拓扑压成单一整条轨迹会造成 topological collapse——不加区分地惩罚合法的替代探索路径，严重损害策略多样性。**DART-SD** 把范式从全局强制转为拓扑引导的局部修正：先把执行过程建模为收敛的 Interaction-State Transition Graph（ISTG），如实刻画成功与失败探索路径的钻石拓扑；在自主 rollout 中定位 Critical Topological Breakpoint（CTB）并检索由成功路径支持的恢复参考；最后以 CTB 引导的局部监督做渐进式 self-distillation，训练损失仅在生成的恢复步骤上计算，严格保护有效的推理前缀不被破坏性梯度更新影响。在复杂多轮 tool-calling benchmark 上明显优于全轨迹模仿基线（abstract 未给出具体数值）。
- **arXiv**：[2608.18524](https://arxiv.org/abs/2608.18524)

#### CROP: Task Relevance via Counterfactuals for Selective On-Policy Distillation (2026-08)
- **简介**：作者 Enhan Li、Junhao He、Hongyang Du（香港大学）指出 OPD 对学生自采 trajectory 上所有 response token 给出等权 credit，而已有的 selective OPD 判据（uncertainty、teacher-student disagreement 等）几乎只刻画「优化需求」，缺少「task relevance」这一互补维度，即该处监督是否真的与当前输入的语义内容绑定。提出 **CROP**（Counterfactual Relevance for On-Policy Distillation），用 paraphrase 校准的 counterfactual sensitivity margin 来操作化 task relevance：对每个源 prompt 构造经校验的 original–paraphrase–counterfactual 三元组，固定学生 rollout 不变，用每个 response 位置对「任务相关条件改变」的敏感度、除以其对「保义改写」的敏感度作为打分。对齐的选择性对照实验表明 CROP 选出的监督位置优于随机选择与最低相关性选择，组件对比确认 counterfactual sensitivity 与 paraphrase calibration 均有贡献；在两组 teacher-student 设置下，CROP 相对最强的非 CROP selector 分别提升综合性能 1.92 与 2.96 个点。
- **arXiv**：[2608.13387](https://arxiv.org/abs/2608.13387)

#### Latent On-Policy Self-Distillation (LOPD) (2026-08)
- **简介**：作者 Guibin Zhang、Jiayang Lyu、Ran Sun、Xinlei Yu 等（含 Shuicheng Yan，新加坡国立大学相关）指出 on-policy self-distillation（OPSD）虽能用「特权 self-teacher」在学生自身 trajectory 上提供 dense 监督，但现有方法严重依赖人工指定的 privileged artifact（答案、feedback、skill、示范 trajectory 等），限制了端到端可学习性与持续自我改进的可扩展性。提出 **LOPD**，不再手工规定特权 context 的形式，而是让 teacher 的特权 context 本身端到端可学：检索相关经验并组合成连续 latent token 来 condition self-teacher，学生依任务与交互历史生成 trajectory 并在其访问过的每个前缀上接受 token 级 dense 监督，另引入 privileged-margin 目标稳定并约束 latent context 的学习。实验上 LOPD 在 agentic tool use 与代码生成两类任务上超过 RLVR 以及 OPSD、SDPO、Skill-SD 等代表性 OPSD 方法，并以不到 GRPO / Skill-SD 30% 的 rollout 预算即反超二者；ablation 直接验证「让特权 context 可学」是取得增益的必要条件。
- **arXiv**：[2608.13040](https://arxiv.org/abs/2608.13040)

#### I-SDPO: Instance-Level Adaptive Self-Distillation Policy Optimization (2026-08)
- **简介**：作者 Yubo Zhang、Xinhong Ma、Zezhong Tan、Ziqiang Dong 针对 GRPO 在 rollout group 全错时拿不到任何有用相对信号的问题：privileged self-distillation 可用 dense token 监督填补这一空白，但全程使用又会引入另一种失效——teacher 是 reward 目标的有偏、低方差代理，当策略已能产出成功 trajectory 后持续模仿会与提升 reward 的更新方向相冲突。提出 **I-SDPO**，把「对 teacher 的依赖」视为随能力变化的量：对每个输入 instance 做一次 routing 决策并在该 instance 的整个 rollout group 内共享——全错组走 privileged self-distillation 目标，存在成功样本的组则原样交给 GRPO，从而只在 group-relative reward 无信息处使用模仿。局部分析刻画了 teacher 方向与 reward 方向何时对齐，并说明不衰减的有偏 distillation 权重会带来优化偏差下界；该 routing 规则随成功概率上升自动降低期望 distillation 比例，无需手工调度。在 SciKnowEval 上 I-SDPO 在四个科学领域全部取得最优，平均 mean@16 准确率从 GRPO 的 56.67% 提升到 70.31%，单领域最大增益 18.24 个点。
- **arXiv**：[2608.12957](https://arxiv.org/abs/2608.12957)

#### Towards Understanding On-Policy Distillation through the Lens of Test-Time Scaling (2026-08)
- **简介**：作者 Xinmu Ge、Zizhuo Zhang、Yu Huang、Jianing Zhu 等（含 Weiran Huang、Jiangchao Yao、Bo Han、Jun Zhou，上海交通大学 / 香港浸会大学 / 蚂蚁集团相关）质疑「OPD 让学生从更强 teacher 处获得超出 pre-OPD base 模型的新能力」这一流行看法，改用 test-time scaling 视角，通过变动采样预算 K 并同时看 pass@K 与 avg@K 来检验。跨多个 OPD 变体的观察是：OPD 训练后的模型在各种采样预算下 avg@K 始终更优，但 pass@K 的优势随 K 增大逐渐让回给 pre-OPD base 模型；训练过程中的 pass@K 动态进一步显示模型在渐进地「用大 K 的能力边界换小 K 的表现」。以 pass@1024 为判据的题目级可解性分析揭示出不对称性：OPD 让原本可解变为不可解的题目数量多于反向变化，故作者将其刻画为「illusory distillation」——表观增益主要来自采样效率提升而非真正从 teacher 获得新推理能力。属 §3.4 的机制/病理分析类工作，是 OPD 能力边界主张的关键对照。
- **arXiv**：[2608.11829](https://arxiv.org/abs/2608.11829)

#### REOPD: Reliability-Adaptive Reward Extrapolation for On-Policy Distillation (2026-08)
- **简介**：作者 Yang Sun、Lichao Ma、Houyuan Qin、Yuxin Liu 等指出 ExOPD 一类 reward-extrapolation 方法通过放大 teacher-reference 对数似然比来突破单纯模仿，但对所有 token 施加同一个全局系数 λ，容易把学生推向隐式 reward 的极端峰值，导致 reward hacking 与训练不稳，且最优 λ 跨领域漂移、需要昂贵的 sweep。提出 **REOPD**，一个 reliability-adaptive 的 reward extrapolation 框架：把 token 级 compatibility 权重与 batch 级自适应预算结合，得到逐 token 系数 λ_{b,t}=1+γ_b·q_t，在保持 teacher 对齐的同时只沿可靠的 teacher-reference 方向做选择性外推；该方法不需要 verifier、reward model、value model，也不需要标准 OPD 之外的额外 rollout。实验中 REOPD 在单 teacher 数学以及多 teacher 设置的两个领域上均优于 G-OPD，在单 teacher 代码上与 G-OPD 持平（abstract 未给出具体数值）。
- **arXiv**：[2608.11698](https://arxiv.org/abs/2608.11698)

#### ReOrder-OPD:Reliability-Aware Prompt Ordering for On-Policy Distillation (2026-08)
- **简介**：作者 Ximo Zhu、Ruiqi Liu、Rong Wang、Ping Wu 等指出 OPD 的 token 级 teacher 监督并不总可靠，而现有方法用局部 confidence 或师生一致性来加权、过滤、截断 trajectory，这些信号并不直接回答「teacher 能否从该学生前缀续写出正确答案」，且 trajectory 级干预会把单条 rollout 的不可靠与其 prompt 的低期望训练价值混为一谈。本文定义 prompt 级的 teacher continuation reliability R——teacher 从学生前缀出发到达正确答案的概率，在当前学生诱导的前缀与 trajectory 上取平均；oracle 实验显示高 R 的 prompt 带来更大 OPD 增益，且在固定 prompt 池上按 R 降序训练优于随机序与升序。由于精确估计 R 需要大量 teacher 续写，作者改用代理指标：一条独立学生 rollout 与同 prompt 下 verifier 判正的 teacher trajectory 之间的最大 ROUGE-5 F1；按该分数等频分十档后各档平均 R 单调上升，说明代理能区分粗粒度可靠性。**ReOrder-OPD** 依代理分数排序 prompt 后再为 vanilla OPD 采独立的 on-policy 训练 trajectory，在 Qwen3 与 Gemma4 数学、Qwen3 代码设置的所有对齐综合比较中均有提升，并在 FiRe-OPD 与 ExOPD 的全部六个设置上取得增益，说明 prompt 排序与 trajectory 内部的监督设计互补。
- **arXiv**：[2608.10905](https://arxiv.org/abs/2608.10905)

#### Mismatch Matters: On-Policy Distillation Beyond Token Agreement (TIDE) (2026-08)
- **简介**：作者 Zichao Yu、Chengzhi Yu、Shengze Xu、Yujin Han 等（含 Difan Zou，香港大学相关）揭示 OPD 的一个失效模式 degenerate agreement：学生靠重复循环骗到与 teacher 近乎完美的 token agreement，整体回答却是错的。作者因此把关注点从 agreement 转向师生 mismatch，并把 mismatch token 分为两类：student-excess token（学生采样但 teacher 赋以近零概率，其 log-ratio 修正无界增长、破坏更新稳定性）与 student-deficit token（teacher 偏好但学生极少采样，其缺失阻断 teacher 推理模式的传递）。提出 **TIDE**（Token-level Independent Deficit-Excess correction），用有界的 Hellinger shaping 抑制最严重的已采样 excess，并用解析式的 teacher top-K 注入补回缺失概率质量，无需 deficit token 被实际采样。在多组 Qwen3 teacher-student 配对的数学推理 benchmark 上，TIDE 稳定优于标准 OPD 及近期 token-selection 与 reward-shaping 基线；师生 mismatch 严重时增益尤为明显，Avg@8 从 6.9% 提升到 20.3%，平均响应长度缩短为原来的 1/3.6，格式失败大幅减少。代码见 https://github.com/yzc-666/TIDE
- **arXiv**：[2608.09836](https://arxiv.org/abs/2608.09836)

#### Distill Skills into Weights, Not Prompts: Abstract Skills as Privileged Signals for On-Policy Self-Distillation (SKALD) (2026-08)
- **简介**：作者 Yubo Jiang、Fengying Xie、Zhiguo Jiang、Haopeng Zhang 指出 RLVR 在 rollout 组全对或全错时给不出 group-relative 信号，而这类组在其实验中占 63.0%–68.0%。提出 **SKALD**（Skill-Anchored Latent Distillation），一个 on-policy self-distillation 框架，对同一 Qwen3-Base 模型使用两种 context view：只看题目的 student，与额外 condition 在「抽象、已过滤显式答案的 skill card」上的 teacher；学生在自己的前缀上训练，把 skill 带来的 advantage 转移进共享参数，测试时不需要特权输入。为稳定 context 引入的分布不匹配，SKALD 采用带退火的 exponentially tilted 目标，压低那些 teacher 偏好但学生似然极低的 token，当 tilt 退到 0 时收敛为 teacher cross-entropy 并还原 forward-KL 的学生梯度；另用经验 gate 仅在 verified rollout 估出正的 teacher advantage 时才启动 distillation。在五个 held-out 数学 benchmark 上，SKALD 在 0.6B / 1.7B / 4B 规模相对 GRPO 的总体 avg@8 分别提升 +2.46、+4.85、+12.01；1.7B 时只对零方差组做 distillation 即可恢复完整增益的 84.7%，SKALD 仍比 FLOP 对齐的 GRPO 高 +4.06、比「context 里直接给 skill」高 +3.77。
- **arXiv**：[2608.09826](https://arxiv.org/abs/2608.09826)

#### SR-OPSD: Self-Referenced On-Policy Self-Distillation (2026-08)
- **简介**：作者 Zhuo Sun、Entong Li、Yanlong Zhao、Xiaoyuan Cheng 等指出 OPSD 中的 self-teacher 通常只是被优化策略的 stop-gradient 或 EMA 副本再加上额外 context，因而与学生策略及其 on-policy context 分布共同演化；用固定的投影目标去直接匹配这样一个移动目标，容易导致优化不稳或分布过度集中。提出 **SR-OPSD**：在固定的学生生成 context 下，通过 token 级变分刻画得出有效 distillation 目标其实是 self-teacher 策略与一个 reference 策略之间的几何插值，同时用 Rényi divergence 族来推广投影几何。该表述把「自适应目标放在哪里」与「学生如何朝目标投影」解耦——插值系数决定底层目标，Rényi 阶数决定投影几何及其对 token 级密度比的敏感度。在科学评测、数学推理与代码生成任务、多个大模型上的大量实验显示 SR-OPSD 在各类设置下达到 SOTA 或具竞争力的表现（abstract 未给出具体数值）。
- **arXiv**：[2608.09745](https://arxiv.org/abs/2608.09745)

#### WDL-OPD: Weak-Driven On-Policy Distillation via Mixture-Constrained Co-Training (2026-08)
- **简介**：作者 Zehao Chen、Gongxun Li、Tianxiang Ai、Yifei Li 等（含 Fuzhen Zhuang、Xianglong Liu、Jianxin Li，北京航空航天大学相关）指出 OPD 虽然靠学生自采 trajectory 缓解了离线蒸馏的 train-test state mismatch，但同一反馈回路本身可能不稳定：每次更新同时改变策略与下一次更新所依据的状态分布。提出 **WDL-OPD**，一种带两个可训练策略的 mixture-constrained 协同训练方法：anchor 策略负责生成全部 rollout，auxiliary 策略在同样被访问的状态上做评估，二者 token 分布的几何混合通过 reverse KL 去匹配冻结的 teacher，两个策略都接收梯度。作者证明冻结 auxiliary 时会退化为一个与 OPD² 和 W2S-OPD 密切相关的 anchor-plus-contrast 代理目标，而联合训练则带来静态 delta 无法表达的 branch 级自由度。在 Qwen3 的 1.7B 与 4B 实验中，WDL-OPD 在四个「规模×领域」设置中都给出最强 checkpoint：MATH500 准确率在 4B 从 0.630 提升到 0.685、在 1.7B 从 0.521 提升到 0.585；代码生成上七个单策略 OPD 配置出现 entropy 增长或 trajectory 退化，而协同训练达到独立复评的 0.637 与 0.375。作者明确指出部分对比在 curriculum 或初始化上不一致，因此结论支持「稳定化假设」而非普适因果主张，并给出完整算法、失败证据与受控对比矩阵。
- **arXiv**：[2608.09447](https://arxiv.org/abs/2608.09447)

#### Bidirectional Context Self-Distillation for Reinforcement Learning of Skill-Based LLM Agents (BCSD) (2026-08)
- **简介**：作者 Tianjun Pan、Yuan Li、Hongda Wang、Linbo Jin 等（含 Chengyu Wang、Chengfu Huo，阿里巴巴相关）指出外部自然语言 skill 能给 LLM agent 提供可复用、可编辑的指导，但效果不仅取决于 skill 质量，更取决于策略能否把指导翻译成合适动作，而专门提升这种「skill 利用能力」的方法基本空白——skill-based agent 通常只用任务级 reward 训练，监督稀疏且无法捕捉「用得好不好」的细微差别。提出 **BCSD**（Bidirectional Context Self-Distillation），把 self-distillation 与 RL 结合：不同于以往只依赖单一特权 context 的自蒸馏，BCSD 从两个互补的 skill-context view 评估同一条 trajectory——augmented view 加入更高层的 Meta-Skill 指导，reduced view 剪掉通用指导以突出任务专属 skill，二者的 token 级信号合并后用于 rescale RL 的 advantage。在 ALFWorld 与 WebShop 上，BCSD 在各模型规模下取得最佳总体表现，ablation 验证两种 context view 的贡献互补（abstract 未给出具体数值）。
- **arXiv**：[2608.09555](https://arxiv.org/abs/2608.09555)

#### Simple-OPD: Demystifying Warm-up for On-policy Distillation (2026-08)
- **简介**：作者 Tao Liu、Taiqiang Wu、Mao Zheng、Yujiu Yang 系统拆解 OPD 之前 warm-up 阶段的作用——OPD 在 student 自己的 rollout 上用 teacher 的 token-level 监督训练，其效果强烈依赖 OPD 前的 warm-up。数据侧发现有效的 warm-up 依赖 teacher-compatible 的 chain-of-thought 监督，而且 teacher 的错误 rollout 能带来与正确 rollout 相当的收益，说明 warm-up 主要迁移的是与 teacher 兼容的思维模式而非正确答案本身；训练侧发现 LoRA 配合接近饱和的训练时长，比全参数 SFT 更好地平衡 in-domain 适配与 OOD 泛化。据此提出 **Simple-OPD**：一种即插即用的初始化方法，在 OPD 前用 LoRA 在 teacher 生成的 CoT 上给 student 做 warm-up，多种设定下验证了有效性与鲁棒性（abstract 未给出具体数值）。本文兼具机制分析与轻量方法两重定位，是 §3.4 中关于 OPD 初始化条件的关键对照。
- **arXiv**：[2608.06802](https://arxiv.org/abs/2608.06802)

#### On-Policy Self-Distillation without Any Supervision (U-OPSD) (2026-08)
- **简介**：作者 Yijiang Li、Bingyang Wang、Yijun Liang、Nuno Vasconcelos 指出现有 OPD / OPSD 仍重度依赖外部监督——ground-truth 信号、环境反馈或更大模型的指导，因而并非真正的「self」-distillation。本文提出 **U-OPSD**，仅凭模型自身生成、依靠内部一致性完成无监督 on-policy self-distillation：先采样多条 rollout，在 self-consistency 阈值下由多数投票构造伪解，再把模型分布条件在该伪解上，只对与之分歧的 completion 做自蒸馏，从而让模型恰好在「自信却错误」的位置自我纠正。在 AIME24、AIME25、HMMT25、MATH500、AMC23 五个数学推理基准上，Qwen3 non-thinking 模式的 4B 与 8B 分别较 base 提升 8.5% 与 10.7%，平均超过带 GT 的 OPSD 3.2% 与 2.3%；thinking 模式下与 OPSD 基本持平（4B 领先 0.9%、8B 相当），并分别超过 GRPO 0.7% 与 1.1%。
- **arXiv**：[2608.06296](https://arxiv.org/abs/2608.06296)

#### DASH: Divergence-Adaptive Supervision Horizons for On-Policy Self-Distillation of Reasoning Models (DASH) (2026-08)
- **简介**：作者 ZhiYan Hou、Xinyu Tang、Haiyun Guo、Jinqiao Wang 针对 OPSD 未充分利用 rollout 时间结构的问题：RLVR 的可验证 outcome 信号稀疏且停留在序列级，OPSD 通过在 student 访问的前缀上查询 privileged teacher 提供 dense token-level 分布监督缓解稀疏性，但标准 OPSD 给每个局部 divergence 相同系数，无视其位置与所处的 divergence 序列——同样幅度的分歧可能来自完全不同的师生失配演化史，单个局部标量无法区分这些时间上下文。**DASH** 把每个局部蒸馏信号与序列级均值之间的差距映射为自适应传播门（propagation gate），再用这些门控制向后的多步聚合，从而按局部 divergence 在生成过程中的演化方式调整 token-level 监督权重。三个数学推理基准、三个模型规模下均优于作者匹配复现的 vanilla OPSD；由于复用 OPSD 已计算的师生分布，增益不需要任何额外的 teacher 或 student 前向。
- **arXiv**：[2608.06243](https://arxiv.org/abs/2608.06243)

#### When Teachers Mislead: Spurious-Signal-Aware On-Policy Distillation (SA-OPD) (2026-08)
- **简介**：作者 Yinuo Jiang、Yongjie Ye、Qiang Zhang、Huajun Chen 指出近期 selective OPD 只按 confident / informative / learnable 挑选信号，忽略了语言模型的一种根本失效模式：其 token 级判断可能由与输入无关的语言先验、格式惯例或刻板推理模板驱动，而非任务相关证据。作者把这类「与优化相关但弱输入依赖」的监督称为 OPD 中的 spurious signal——它们会产生很大梯度，却几乎不提供改进任务的方向。**SA-OPD** 引入一个轻量的 input-groundedness 代理，估计某个 token 级蒸馏信号是否真正依赖输入，并只过滤同时满足低 input-groundedness 与极端蒸馏 divergence 的 token，从而剔除高影响的 spurious 更新、实现细粒度 OPD 优化。在 LLM 与 VLM 两类设定上均稳定优于 Vanilla OPD 及有竞争力的 selective 方法（abstract 未给出具体数值），把 input-groundedness 确立为 OPD 监督筛选的新维度。
- **arXiv**：[2608.03632](https://arxiv.org/abs/2608.03632)

#### Look Ahead Before You Distill: Future Trajectory Validation of Teacher Guidance for Agentic On-Policy Distillation (FutureBridge-OPD / FTB) (2026-08)
- **简介**：作者 Chishui Chen、Yaoyou Fan、Xuyang Liu、Linfeng Zhang 关注多轮 agentic 任务中的 OPD：OPD 在 student 访问的状态上给出 teacher 监督以缩小训练与推理的分布差，但 student 的偏离会随轮次累积，使轨迹逐渐离开 teacher 指导仍然有效的状态区域。作者的定量分析显示高分歧（high-disagreement）状态是施加 teacher 指导的好时机，但该指导是否有益必须看它对后续 student 轨迹的影响。**FutureBridge-OPD（FTB）** 因此在高分歧状态执行一段短的 teacher bridge，再用由此产生的 student 续写来判断这段 bridge 相对 teacher 是否提高了正向蒸馏信号的密度，据此决定是否采用。在 ALFWorld、WebShop、ScienceWorld 上，以 Qwen3-32B teacher → Qwen3-1.7B student 的主设定，FTB 平均分别超过 vanilla OPD 与 TCOD 16.6 分与 7.6 分，并在不同 student 规模与 teacher 设定下保持有效。
- **arXiv**：[2608.01953](https://arxiv.org/abs/2608.01953)

#### AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning (AgentOPSD) (2026-08)
- **简介**：作者 Zi-Han Wang、Zhengxi Lu、Yongliang Shen、Yujiu Yang 针对 RLVR 只给出轨迹级 advantage、难以把功劳归给长时程多轮 agentic 任务中少数决定成败的关键决策；近期工作用 privileged self-distillation 提供更密的监督，但这类局部信号该如何表示时序 credit 仍不清楚。**AgentOPSD** 是 critic-free 的递归 turn-level credit assignment 方法：把 token 级的 teacher-student log-probability gap 聚合成 turn 级证据，并在 log-odds 空间递归更新一个贝叶斯信念状态，由此得到把稀疏 outcome 监督转换为 turn 级 credit 的重加权方案，并用相邻状态间的边缘信念修正量识别关键 turn（pivotal turn）。方法与标准策略优化完全兼容，既不需要额外 critic 也不需要额外 rollout。在 ALFWorld、WebShop、Search-QA 上用 Qwen2.5 的 3B 与 7B 两个规模评测，优于 GRPO 与强自蒸馏基线，Qwen2.5-7B 在 ALFWorld 达到 89.1% 成功率；消融把增益归因于 turn 级聚合与依赖历史的递归信念更新。
- **arXiv**：[2608.05987](https://arxiv.org/abs/2608.05987)

#### Outcome-Confounded Local Supervision in On-Policy Distillation (2026-07)
- **简介**：作者 Guoqing Ma 对 On-Policy Distillation（OPD）中「局部 token 级监督信号」的可靠性提出诊断性质疑。OPD 中教师在学生访问的前缀上给出 dense token-level likelihood，通常局部解读为「一致=可安全模仿、分歧=定位错误」；本文证明这两种解读都被整条轨迹的最终结果所混淆（outcome-confounded）。提出「outcome-resolved 诊断」，把逐点师生分歧与最终答案对错交叉，区分安全模仿、有益分歧、有害分歧、以及「一致但失败（agreement-on-failure）」。在 Qwen3-8B 学生 / Qwen3-32B 教师的八种子实验中，agreement-on-failure 占响应 token 质量的 67.84%（Qwen2.5-7B/32B 对为 67.68%）。三种匹配训练探针（模仿/掩码/对比整条轨迹）均无法稳定降低该比例，指出局部分歧配轨迹级结果无法定位失败轨迹的不可挽回点。本文定位为诊断而非新训练方法，是 OPD 监督信号有效性的关键对照。
- **arXiv**：[2607.23731](https://arxiv.org/abs/2607.23731)

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

#### Beyond On-Policy vs. Off-Policy: At What Level Should LLM Agent Distillation Follow the Student? (2026-10)
- **简介**：作者 Ya Gao、Zhengtao Zou、Pekka Marttinen。蒸馏交互式 LLM agent 需要在 teacher 生成的示范（对应 off-policy）与 student 生成的经验（对应 on-policy）间选择。作者认为这一区分本质上是层次化的：产生交互历史的策略不必与生成当前动作 token 前缀的策略相同。论文提出层次化 agent 蒸馏框架，形式化两个层级的 on-policiness，可显式控制其配对、混合比例与训练调度；通过 self-distillation（冻结 teacher 能访问 student 没有的任务指引）在三个模型、四个交互环境上做三阶段受控研究。发现：student 诱导的历史配 teacher 生成的动作前缀通常提供最强的单独训练信号，两层都跟随 student 通常最差；但完全 on-policy 数据在与其他配置混合时有益；各层 teacher/student 比例相同的混合也可能表现差异很大，说明配对方式比 on-policy 数据总量更重要；先用完全 teacher 生成的上下文、后引入完全 on-policy 上下文优于静态混合。
- **arXiv**：[2610.31918](https://arxiv.org/abs/2610.31918)

#### Do As I Do, Not As I Say: On-Policy Distillation Inherits the Teacher in Vision-Language Models (2026-10)
- **简介**：作者 Frank F. Yang、Kefei Yan、Yiren Wang、Yiyang Wang。论文在不同数据预算和 teacher 可靠性不同的任务上，对 VLM 的 OPD 与过滤后的 SFT 做对照研究。发现 OPD 仅在 teacher 在 student 轨迹上仍可靠时才有效：ChartQA 上 teacher 完成 99% 的解答，仅用 10% 数据的 OPD 与全量数据 SFT 差距在 1.4 个点以内；Geometry3K 上 teacher 有 22% 的解答未完成，OPD 比 SFT 低 7.9 点，差距多由不完整回答解释。更有意思的是，从 SFT student 出发做 OPD 会使其完成率从 76% 降到 62%，抹掉 SFT 学到的完成行为；仅按完整性过滤 teacher 示范即可恢复 SFT 的大部分优势。结论：on-policy 监督可能迁移的不只是 teacher 知识，还有在 student 访问状态上的持续性失败模式。
- **arXiv**：[2610.20989](https://arxiv.org/abs/2610.20989)

#### SAPD: Step-Aligned Privileged Distillation (2026-10)
- **简介**：作者 Tianle Wang、Jiayu Liu、Ruizhi Zhao、Ning Miao。On-policy 后训练需要昂贵的 rollout 生成，论文探讨固定示范能否通过更好的监督支撑有竞争力的 off-policy 学习。**SAPD** 是一种免 rollout 的 self-distillation：利用参考解的已知推进过程，把每个推理转移与有针对性的特权引导对齐，形成 step-aligned 的分布式监督，而不是把参考解当作无差别上下文。在数学推理 benchmark 上平均优于 SFT 和 label smoothing，同时与 on-policy RL 和 self-distillation 保持竞争力，较好保留域外代码能力，训练循环相对 on-policy 基线约 2 倍加速。
- **arXiv**：[2610.09665](https://arxiv.org/abs/2610.09665)

#### REVO: Rollout-Efficient Off-Policy Distillation via Variance-Guided Reuse (2026-09)
- **简介**：作者 Yuxiao Yang、Shangzhe Li、Weitong Zhang 等针对 on-policy distillation（OPD）需频繁刷新学生 rollout、生成成本高的问题，提出 off-policy 蒸馏框架 **REVO**，让每条学生 rollout 可用于多步学习器更新。REVO 通过稳定化的 prefix 加权与基于当前学生的一步重采样来处理前缀级和当前 token 的策略失配，避免重新生成整条轨迹；并用 student-teacher log 概率比的方差衡量 token 位置剩余的学习信号，据此在复用的 rollout 内优先优化信息量大的位置。跨多种师生规模，REVO 仅用 50 个 rollout 迭代即可匹配或超过训练 200 个迭代的 OPD 基线，在域内与跨域推理基准上均成立。
- **arXiv**：[2609.37500](https://arxiv.org/abs/2609.37500)

#### On-Policy Distillation Meets Off-Policy GRPO: Training Compact Instruction-Following Rerankers (2026-09)
- **简介**：Vignesh Prabhakar、Jialing Pan、Anil Babu Ankisettipalli 从 RL 视角重做 reranker 蒸馏：传统流水线让 student 在固定样本集上离线模仿 teacher 输出，监督被限制在 teacher 已观测到的排序空间内。提出两阶段框架——Stage 1 用 off-policy GRPO 配 LLM-judge 反馈、在 88K 条指令跟随样本上强化一个 4B teacher reranker；Stage 2 让 1B 的 compact student 从自身 policy 采样排序，并对这些排序接收 teacher 导出的 soft reward，把 student 探索与知识迁移耦合起来。收益在分布迁移下最明显：MAIR-11（11 子集、869 query）上 student 达 0.7670 nDCG@6，比离线 listwise KD 高 4.6 分；受控对比显示既换离线蒸馏目标（pairwise RankNet KD）也不行、把 teacher 分布匹配搬到 on-policy（GKD）也不行，均无法复现「在 student 采样排序上做 reward-based OPD」的效果。MAIR-Full（126 任务、9356 query）上取得所测蒸馏变体中最高的 task-macro 点估计，0.6808 nDCG@6 与 0.7865 MRR@6，并在可比的 MAIR-11 评测上超过两个已发布的 7B RL 训练 reranker；同样的 Stage 2 流程在三个架构各异的 student backbone 上均有提升，9861-query 验证基准上 1B reranker 达 0.7624 nDCG@6。EMNLP 2026 Findings。
- **arXiv**：[2609.01947](https://arxiv.org/abs/2609.01947)

#### Reason in the Words You Speak: Idiolectal Paraphrasing Off-Policy Traces for Reasoning Distillation in VideoLLMs (Echo-GRPO / VideoEcho-R1) (2026-08)
- **简介**：作者 Ji Soo Lee、Jinyoung Park、Seohyun Lee、Jongha Kim 等指出 GRPO 的 on-policy 性质把模型限制在它已能产出的推理技能内，难以学到更高级能力；已有工作从更强 teacher policy 注入 privileged reasoning trace，但这些 trace 相对学生 policy 天然 out of distribution。作者观察到这种 on-policy 与 off-policy 的错配会在语义关键的推理 token 上触发 gradient clipping，最终「奖励了正确答案，却没学到支撑答案的推理」。据此提出 **Echo-GRPO**：不去模仿 teacher 那些在学生下低概率的 privileged trace，而是通过 Dual-Reference Decoding 在保持语义的前提下把它们改写成学生 policy 自己的 idiolect（其特有词汇与表达方式），并实例化为面向视频推理蒸馏的 **VideoEcho-R1**，在三个多模态 LLM backbone、五个 benchmark 上取得一致提升（abstract 未给出具体数值）。作者进一步表明该 idiolectal paraphrasing 是可插拔模块，对 RL 与 SFT 两类推理蒸馏框架都有稳定增益，说明 policy-aligned 监督的适用范围不限于 GRPO——是 §3.5 中「off-policy trace 为何失效、如何改造成近 on-policy 监督」的关键对照。
- **arXiv**：[2608.26684](https://arxiv.org/abs/2608.26684)

#### REGEN: Replay-recycling for Expert-to-Generalist distillation with Offline Reinforcement Learning (REGEN) (2026-07)
- **简介**：Yunjie Chen、Xiaoxin Chen、Fang Wang 提出 REGEN，针对多教师 on-policy distillation（MOPD）仍需耦合 inference 与 backward、扩展性受限的问题，改为直接**回收**教师专项 RL 训练时的 replay memory（免费副产品），并用**离线 RL 算法**训练通用学生，从而完全解耦 rollout 采样与反向训练、大幅降低成本。在数学推理、代码生成、指令跟随上以显著更低的成本达到 MOPD 的精度，将在线 RL 从"一次性学习阶段"转为可复用的数据合成过程。
- **arXiv**：[2607.19450](https://arxiv.org/abs/2607.19450)

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

#### DTDD: Divergence-Triggered Dynamic Distillation for Reliable On-Policy Supervision (2026-10)
- **简介**：作者 Qijia He、Zequn Li、Spencer Luo、Pradyumna Narayana、Zixian Ma 等。当 student 与 teacher 因初始策略差距大或长程推理、工具使用中的累积误差而分歧时，OPD 的稠密 token 监督会变得不可靠。**DTDD** 用 segment 级 student-teacher 散度触发 teacher 接管：触发后 student 前缀仍接受标准 OPD 监督，teacher 生成恢复后缀并让 student 通过 behavior cloning（对 teacher 生成 token 的交叉熵）学习。在长程推理、无 SFT 的基座初始化蒸馏和多轮 agent 三类设置中评测：DTDD 将 warmup 后峰值梯度范数降低 8.6 倍；从基座初始化时，以 OPD 的 1/2.6 的优化步数达到 21.3% 的留出数学准确率；ALFWorld（LoRA）与 agentic code 上的最佳表现分别提升 4.8 和 4.1 个百分点。消融显示 segment 平均触发优于 token spike 触发。
- **arXiv**：[2610.19302](https://arxiv.org/abs/2610.19302)

#### Gains and Collapse in On-Policy Distillation: A Reinforcement Learning Perspective (2026-10)
- **简介**：作者 Han Cui、Jianhao Yan、Yun Luo、Hongbo Zhang、Zhizhang Fu、Yue Zhang。OPD 在带来增益的同时也可能坍缩为过长、重复的生成，机理不明。论文从 RL 视角解释：teacher 隐式地奖励 student 行为，包括 teacher 自己很少产生的行为。实验显示 OPD 提升性能但并不扩展 student 能力；当隐式 reward model 可靠时正确回答更易被采样，而当偏好与质量错位时会发生 reward hacking，隐式 RM 会放大过长重复的 student rollout。据此发现，对不健康回答做 mask 以及使用 SFT 初始化都能有效缓解坍缩。结论是 OPD 放大 teacher 隐式反馈所偏好的 student 行为，关键从「teacher 生成得多好」转为「teacher 评判 student rollout 多可靠」。（abstract 未给出具体数值）
- **arXiv**：[2610.03185](https://arxiv.org/abs/2610.03185)

#### Why On-Policy Distillation Sometimes Fails: Vanishing Learning Signals (2026-10)
- **简介**：作者 Lei Zhao、Qichao Zhao、Bowen Zuo、Qishi Zhan。在代码生成与数学推理中，使用更大 teacher 的 OPD 会出现早期 loss 平台期：200 步后平均最终 loss 仅下降 25.1%，而 self-RL teacher（对初始 student 继续 RL 得到）可达 96.2%。论文把 OPD 视为小学习率极限下的连续时间动力系统，训练日志诊断显示平台期与基于梯度的 learning-signal 代理指标早期下降相关；并证明当 teacher 足够接近初始 student 时存在局部恢复保证，为 self-RL teacher 的成功提供条件性解释。观测到参数相对变化仅 0.025–0.098%、训练前后表征线性 CKA 超过 0.98，提示表征适应受限可能导致学习信号坍缩（该假设尚待检验）。
- **arXiv**：[2610.11247](https://arxiv.org/abs/2610.11247)

#### Towards Unbiased On-Policy Distillation for Block Diffusion Language Models (Un-OPD) (2026-10)
- **简介**：作者 Zaiquan Yang、Fei Wei、Yong Wang、Yudong Han、Yiyu Li、Zhuofan Zong。针对把 OPD 扩展到较大 block size 的 block diffusion 语言模型，论文揭示两类优化偏差：teacher 与 student block 边界不匹配造成的 context misalignment，以及 OPD 内在的优化偏差——student 迅速吸收高支持度信号而滞后于低支持度更新，导致过早置信度飙升和过度自信坍缩。提出 **Un-OPD**：boundary-aware 步骤过滤去掉上下文错位的解码步，support-rebalanced 置信度校准在高支持位置降低优化强度；并加入 rollout 复用机制降低生成开销。在数学推理与代码生成上稳定训练并取得更优性能，墙钟训练时间约减半。
- **arXiv**：[2610.05373](https://arxiv.org/abs/2610.05373)

#### Does On-Policy Distillation for Safety Pose Backdoor Risks? (2026-10)
- **简介**：作者 Jian Luo、Kehan Qi、Qingqiao Hu、Meilong Xu、Jiacheng Qiu、Weimin Lyu。OPD 被用于提升 LLM 安全，但通常假设 teacher 和数据可信。论文发现一个被忽视的威胁：安全对齐但被植入后门的 teacher 能把隐藏的恶意行为传给原本干净的 student，仅 3% 投毒率即可使蒸馏后 student 的攻击成功率（ASR）达到最高 70%。两个训练选择会放大风险：增加 epoch（仅 10 个投毒样本训练 16 epoch 后 ASR 达 67%）、常用的 top-k KL 比 sampled-token KL 更快转移后门。作者还探索了简单的缓解办法 **Lazy Defense**：对 KL reward 做 clip 以减缓 student 更新，在低投毒率下可延迟后门迁移。
- **arXiv**：[2610.07654](https://arxiv.org/abs/2610.07654)

#### On Language Drift during RLVR Post-Training (2026-10)
- **简介**：作者 Michael Sullivan、Alexander Koller 研究推理模型在 RLVR 后训练中思维链出现非标准、看似无意义的语言使用的 language drift 现象。理论上证明 RLVR 的优化压力允许无界的语言漂移，而 SFT 不会；实验显示漂移主要出现在新颖推理任务上的 RLVR 中，即目标行为无法直接从基座模型中「引出」时。最后证明在不约束期望 reward 的情况下无法约束语言漂移，意味着在前沿水平上，CoT 可监控性的提升无法不以损害性能为代价。
- **arXiv**：[2610.02015](https://arxiv.org/abs/2610.02015)

#### Reasoning-Preserving Fine-Tuning of Post-RL LLMs with Null-Basis LoRA (NB-LoRA) (2026-09)
- **简介**：作者 Wenzhi Fang、Nicholas Tzou、Lazar Valkov、Srinivas Chappidi。RL 后训练模型再做 SFT 适配新领域时，推理能力容易被严重覆盖；现有的经验回放、特殊初始化或梯度投影方法要么保留有限，要么开销大。论文分析发现推理激活集中在低维子空间，留有大量零空间容量，且近似零空间可用少量样本可靠估计。据此提出 **NB-LoRA**：把推理保持表述为逐层隐状态保持约束，用推理激活构造固定的近似零基，并通过该基重参数化 LoRA 更新，使约束贯穿整个微调。在多个 RL 训练 LLM 与多样下游任务上，NB-LoRA 的适配性能与标准 LoRA 相当，推理准确率保持在微调前水平附近，且保持效果可泛化到留出的推理基准。
- **arXiv**：[2609.25618](https://arxiv.org/abs/2609.25618)

#### Geometry of Divergence: Tracking Hidden-State Trajectories for Adaptive Multi-Turn Reasoning (2026-08)
- **简介**：Jie Liang、Zhengxin Yu、Hamid Nasiri、Peter Garraghan 关注 LLM agent 在长多轮交互中于严格资源约束下维持目标一致推理的问题：随着多轮上下文累积，底层 LLM 对早前轮次任务相关信息的内部表示会被扰乱，使「建设性推理」与「表示 drift」的边界变得模糊。作者把多轮推理形式化为底层 LLM 的 hidden-state 轨迹，并用两个互补信号刻画：temporal curvature 捕捉轮间更新的方向一致性，variance slope 度量探索空间的扩张或收缩。在四个任务、三个底层 LLM 上，这两个几何信号能在 episode 结束前就区分正确与错误的 episode；作者进一步把每个 episode 分解为由 Read / Write / Respond / Transfer 四种动作构成的三动作链，显示可分性依赖于动作类型，不同信号区分不同的链式模式。基于轨迹几何识别推理过程中的关键轮次，把 $τ$-Bench 上任务成功率从 24.1% 提升到 39.6%，同时降低 11.2% 的 token 成本。
- **arXiv**：[2608.30650](https://arxiv.org/abs/2608.30650)

#### Why Summaries Turn Neutral: Policy Attribution for Sentiment Drift in Reinforcement Learning from Human Feedback (2026-08)
- **简介**：作者 Mikhail Krasitskii、Alexander Gelbukh、Olga Kolesnikova、Grigori Sidorov（墨西哥国立理工学院 IPN-CIC）诊断 RLHF 在提升摘要流畅度与安全性的同时引入的 sentiment drift：摘要过度中性化、情感层次被抹平。作者提出 **Policy Attribution** 框架，用梯度分解与 logit 分解把 drift 归因到 reward model 信号与 KL 惩罚两个来源，并论证该现象反映的是在偏好不确定性下、为最大化期望 reward 而偏向「低风险」token 的策略性偏置。在 Reddit TL;DR 与 CNN/DailyMail 上，RLHF 摘要 reward 更高但情感方差低 30–40%；跨八种语言的分析表明 drift 与语言无关，形态更丰富的语言被抑制得更厉害。作者还提出并验证了一种 sentiment-aware 正则化，在不损害摘要质量的前提下把 drift 降低 18–22%，代码与工具包将开源。
- **arXiv**：[2608.15530](https://arxiv.org/abs/2608.15530)

#### Toward Plasticity-Preserving KL Regularization for Capability Retention in LLM Reinforcement Learning (CoKL) (2026-08)
- **简介**：作者 Li Wang、Xiaodong Lu、Jiajun Chai、Guojun Yin 针对 RL 后训练在优化新目标时损害 base 模型既有能力的问题：KL 正则被广泛用于约束相对 reference 模型的 policy drift 以缓解遗忘，但标准 full-policy KL 约束的是整条响应分布，可能不必要地限制探索与目标任务学习。本文提出 **CoKL**（Correctness-Conditioned KL Regularization），把保持性约束从完整输出分布收窄到 correctness-conditioned 响应分布，以 forward KL 实例化并推导出可用于 RL 后训练的有限 group 训练目标；它将「分配给正确响应的总概率质量」与「正确响应内部的条件分布」解耦，只正则 reference 支持的正确响应之间的相对概率分配。作者进一步证明 reference policy 不完美时 full-policy 的 forward 与 reverse KL 都会诱导一个严格的最优正确率差距，而 CoKL 规避了这一局限。可控多解环境与多规模持续后训练实验显示 CoKL 在目标任务提升与旧能力保持间的平衡优于现有正则方法（abstract 未给出具体数值）。
- **arXiv**：[2608.01743](https://arxiv.org/abs/2608.01743)

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

#### Divergence controls entropy in distillation (2026-10)
- **简介**：作者 Nicolas Zucchet、Scott W. Linderman。论文从熵的视角研究蒸馏目标中的数据与散度如何决定 student 熵。证明 forward KL 会使 student 熵高于 teacher（交叉熵训练是其特例，在预训练和 SFT 中定量验证）；reverse KL 会压低熵，直到师生差距过大；二者插值在训练早期平滑改变熵、在收敛时则突变。On-policy distillation 熵更低的原因来自 token 级 reverse KL，而非 on-policy 采样本身。因此散度起到隐式熵正则作用，在 self-distillation 中尤为明显：以特权信息为条件会压低熵，效果最好的散度超参正是能补偿该效应的那些。
- **arXiv**：[2610.03529](https://arxiv.org/abs/2610.03529)

#### One Symptom, Three Levers: A Critical Review of On-Policy Self-Distillation (2026-08)
- **简介**：作者 Justin Robert、Raheel Qader 对 On-Policy Self-Distillation（OPSD）做批判性梳理：OPD 让模型在自己的生成上训练、由 teacher 逐 token 打分，兼具 imitation learning 的 dense 监督与 RL 的 on-policy 采样，但需要一个更大的第二模型作 teacher；OPSD 去掉这一开销，teacher 即模型自身，只是额外条件于学生在测试时看不到的 privileged information（参考解答、计划或环境反馈）——teacher 不比 student 更强，只是「知道得更多」。早期结果显示其精度可比 RL 而生成 token 量只是一个零头，但产生信号的同一不对称性也在污染信号，collapse（模型可产出的推理路径集合逐步收窄）已成为该方向的主导失效模式，且 collapse 并非 OPSD 独有、只是被 privileged information 加剧。本文把 collapse 视为受三个「杠杆」支配的症状：信号施加在何处（token 如何加权）、teacher 被展示了什么（privileged information 的性质）、信号何时变化（teacher 动态与指导的衰减），范围限定在数学推理；明确声明不报告任何新实验（abstract 未给出具体数值），贡献是结构性的——为各文献中命名不一的现象建立共享术语，并划清已成定论与尚存争议的边界。属 §3.4 的批判性综述。
- **arXiv**：[2608.25936](https://arxiv.org/abs/2608.25936)

## 4. Multi-Agent

> 多个 LLM 协作、辩论、自博弈、coordinator 训练。

### 4.1 Multi-Agent Co-Training

#### JHiMA-WebRL: A Joint Hierarchical Multi-Agent Reinforcement Learning Framework for Long-Horizon Web Tasks (2026-11)
- **简介**：作者 Shanka Subhra Mondal、Jiayun Wang。长程 web 任务需要同时规划、导航、跟踪进度与执行视觉 grounded 动作，单体策略较为吃力；已有模块化多 agent 系统的 planner 通常不与 actor 联合更新。**JHiMA-WebRL** 将 task decomposer 与 actor 耦合：先用强教师模型从多模态 GUI 轨迹构造结构化子任务监督，微调 decomposer 与子任务条件化的 actor，再把 GRPO 扩展为通过层次 rollout 分组与角色特定 advantage 在交互式 WebArena 环境中联合优化二者。在 WebArena-Lite-v2 上，由 7B actor + 7B decomposer 组成的 JHiMA-WebRL-14B 平均成功率 37.2%，高于作者对比的 ScaleCUA-32B 的 24%。
- **arXiv**：[2611.05263](https://arxiv.org/abs/2611.05263)

#### Octopus: One Brain, Many Branches for Latent-KV Multi-Agent Reinforcement Learning (2026-11)
- **简介**：作者 Tianming Sha、Junnan Li、Siqi Zhao。深度搜索与长程推理本质上是分支轨迹，但现有服务引擎、harness 与 RL 训练器把 episode 当作平坦、只追加的序列。**Octopus** 是协同设计的服务运行时、内存管理器与 RL 框架，把 episode 执行为基于 radix cursor 的 GPU 驻留 DAG（KV 命中率 89.6%）；agent 通过 fork、rewind、read_thread、wait、done 五个原语协同。只在 full-attention 层拼接兄弟 KV 页，保留 92.2% 的全文答案信息，并将 hijacking 降低 21 倍（57.9% → 2.7%）。RL 阶段通过 packed DAG kernel 与 **DAG-GRPO** 目标对共享前缀去重，同时保持挂载的 KV 边可微，使 rollout-training KL 散度降低 32 倍。以 Qwen3.5-4B 加二元奖励训练，涌现出 commander-worker 式的自组织协作（摘要后续部分未显示）。
- **arXiv**：[2611.06956](https://arxiv.org/abs/2611.06956)

#### s$^3$: Simple Scaling of Swarms via Cooperative Multi-agent RL (2026-11)
- **简介**：作者 Kunal Jha、Gonçalo Faria、Shuyue Stella Li、Scott Geng、Faeze Brahman、Noah A. Smith 等。多 agent 系统的组织方式通常来自预先设计的静态 harness，不随 agent 行为与任务适应。**s$^3$** 是首个让 agent 自行决定独自工作多久、何时分享发现或阅读队友消息的多 agent RL 方法，无预定义组织或角色；agent 学会采取不同方法、阅读彼此工作、涌现互补角色，并能泛化到更大团队。作者研究 agent 须合并私有信息的协调任务，以及分享发现可提高搜索效率的并行探索任务。在训练与推理预算匹配下，用两倍于训练时 agent 数量评估的 s$^3$ swarm 在协调任务上比最强基线多解决超过 25 倍的问题；在探索任务上，通信使重复工作减少近 50%，并比 best-of-N 高出最多 9.4 个百分点。
- **arXiv**：[2611.08615](https://arxiv.org/abs/2611.08615)

#### Belief-Conditioned Multi-Agent Actor Critic for Decentralized LLM Collaboration (BCMAAC) (2026-10)
- **简介**：作者 Tianle Chen、Shuo Liu、Christopher Amato，NeurIPS 2026 Workshop PTA 接收。已有用 MARL 优化 LLM 协作的工作多依赖预定义工作流，限制了效率与可扩展性；去中心化 LLM 协作中 agent 独立执行、通信受限、共享联合目标，且必须从通信痕迹中推断有用信息。论文提出 **BCMAAC**（Belief-Conditioned Multi-Agent Actor-Critic），维护对上下文信息的信念并用其辅助训练。在 Minecraft 游戏、网页搜索和工具使用任务上，相比朴素 actor-critic 基线，BCMAAC 提升了任务表现与协作行为，在上下文信息更丰富的多轮任务中信念建模优势尤为明显（abstract 未给出具体数值）。
- **arXiv**：[2610.35398](https://arxiv.org/abs/2610.35398)

#### Learning Calibrated Confidence for Multi-Agent Collaboration (MASCAL) (2026-10)
- **简介**：作者 Chenxiao Fan、Chongming Gao、Gangyi Zhang 等。多 agent 系统通过分工完成多跳问答等复杂信息搜寻，上游 agent 的中间答案正确性参差，而多数系统不给下游 agent 任何定量信号，只能依据文本证据判断是否信任。**MASCAL** 让每个 agent 通过 RL 联合优化任务表现与置信度校准，学会为自己的答案输出正确概率；上游在交接时传递该概率，为下游决策提供定量依据，最终答案也带有校准概率；并提出 confidence-aware weighting，在对上游概率有响应的位置提高学习权重，使 agent 更多地从主动使用该信号的决策中学习。在涵盖网页检索与推理的三种协作架构上，MASCAL 提升了两个 agent 的任务准确率与校准质量，并能泛化到更深的协作链与不同模型规模；干预分析证实上游置信度确实影响下游决策，最终置信度还支持选择性回答与针对性复查。
- **arXiv**：[2610.39927](https://arxiv.org/abs/2610.39927)

#### MultiAgent-OPD: End-to-End Training of Hierarchical Multi-Agent Systems with On-Policy Distillation (2026-10)
- **简介**：作者 Yuhan Sun、Jiaxin Zhao 等。层次化多 agent 系统中 Leader 分派任务、子 agent 执行，联合训练的难点在于最终答案对子 agent 只给出间接反馈，而 LLM judge 打分又依赖 judge 自身判断。论文提出 **MultiAgent-OPD**，把 on-policy distillation 与 RL 结合，端到端联合训练共享参数的 Leader-Searcher-Answerer 团队：先用答案正确性在 Searcher-Answerer 流程上训练一个 Teacher；团队训练时冻结的 Teacher 为 student 生成的搜索与摘要 token 打分，为 Searcher 提供局部过程监督，Leader 从团队成败学习，Answerer 从候选正确性学习。在七个数据集上达到 38.68% macro avg@4，比 Prompted Critic 高 4.06 个百分点，比训练的级联多 agent 系统 C-MAS + RL 高 2.61 个百分点，且在四次尝试内解决更多问题。
- **arXiv**：[2610.23234](https://arxiv.org/abs/2610.23234)

#### Training LLMs for Multi-Agent Collaboration with Verifiable Information Rewards (CoSIR) (2026-10)
- **简介**：作者 Keru Chen、Sen Lin、Yingbin Liang、Nathaniel D. Bastian、Shaofeng Zou。多 agent 协作要求共享相关证据、请求缺失信息并吸收队友贡献，而最终任务结果很难反馈单条消息的价值。**CoSIR（Collaboration via Surprisal-based Information Rewards）** 在简单、可验证的环境中后训练协作行为：用 Bayesian evaluator 按团队 surprisal（各 agent 已有证据下对正确选项的负对数概率之和）的降低量给共享信息打分，请求则按若被如实提供可能带来的同一量的降低打分；另设奖励鼓励 agent 在私有选择和计算中使用收到的信息。通过 self-play 后训练 Qwen3-4B、Qwen3-8B 和 Gemma-3-4B-it，在七个未见自然语言协作任务上取得最高总体得分，同时通用任务表现保持有竞争力；行为分析显示私有证据披露增多、未使用队友输入的情况减少。
- **arXiv**：[2610.17893](https://arxiv.org/abs/2610.17893)

#### SynCo: Data Synthesis Co-Training for Self-Evolving LLMs via Multi-Agent Reinforcement Learning (2026-10)
- **简介**：作者 Wei Yang、Shawn Li、Yuehan Qin、Yawei Wang、Mingxi Wang、Shixuan Li。自演化 LLM agent 不仅需要更新 agent，还需随其能力变化演化训练经验；但多数流程依赖静态数据集或单独更新的合成模型，使曾经有用的任务变得平凡、过难的任务又缺乏信息量。**SynCo** 基于多智能体 RL 联合优化两个独立参数化的 agent：Synthesizer 根据 Reasoner 不断演化的能力状态构造训练任务，Reasoner 从由此产生的经验中学习。每个合成任务诱导多条 Reasoner rollout，其结果为两个 agent 提供互补奖励：正确性反馈改进 Reasoner，任务质量、答案可靠性和基于结果的 teachability 指导 Synthesizer；二者的更新再反馈到后续合成轮次。（abstract 被截断，实验数字以原文为准）
- **arXiv**：[2610.11345](https://arxiv.org/abs/2610.11345)

#### Recursive Self-Improvement through Multi-Agent Self-Supervision (MASS) (2026-10)
- **简介**：作者 Hyunin Lee、Jinglue Xu、Jeffrey Seely、Donghyun Lee、Somayeh Sojoudi、Matei Zaharia。在开放研究等不可验证任务上，模型输出超出人类专家可可靠评估范围时，递归自改进（RSI）面临监督瓶颈，只能由模型本身兼任优化者与评估者，但单个实例难以在同质循环中批评并改进自己的复杂推理。**MASS**（Multi-Agent Self-Supervision）在进化式 workflow 优化与对自生成轨迹的 SFT 之间交替：让单个基座模型迭代地提出、执行并自评多 agent workflow，在结构护栏约束的进化搜索中优化这类计算图式编排，发现给定任务下最有效的角色与信息路由。用 Qwen3.6-27B 经两个 MASS 周期，在四个开放式公开 benchmark 上每输出 token 的性能提升 1.2-1.6 倍。
- **arXiv**：[2610.12176](https://arxiv.org/abs/2610.12176)

#### MAS-OPD: On-Policy Distillation for Multi-agent Systems (2026-09)
- **简介**：作者 Qiyong Zhong、Mao Zheng、Mingyang Song 等指出多智能体系统（MAS）联合后训练的难点：RL 的团队级 reward 无法确定哪个 agent 的哪一步导致结果，局部 reward 又需逐任务设计；而 on-policy distillation（OPD）能提供稠密 token 级教师监督且无需局部 reward，但在相互依赖的多 agent 场景中尚缺探索。提出 **MAS-OPD**：Role-Advantage Specialization 将角色优势定义为教师在目标角色与非目标角色条件下信号之差，以构建互补的角色专长并保留各角色所需的共同知识；Privileged Attribution for Coordination 把交互冲突归因到其来源并仅作为特权信息提供给教师。在代码与数学基准上，MAS-OPD 在两种学生规模下均取得最高均值，并使 agent 形成更清晰的角色分工与更有效的协作行为。
- **arXiv**：[2609.34234](https://arxiv.org/abs/2609.34234)

#### Improving LLM Collaboration via Multi-Agent Preference Learning (MAPL) (2026-09)
- **简介**：作者 Shuo Liu、Xinzichen Li、Tianle Chen、Christopher Amato 指出，MARL 用于 LLM 协作时难以构造可靠 reward，完整准确的指标往往缺失且难聚合；偏好学习可从比较性的人类或 AI 反馈中学习，但在多智能体系统中的扩展尚缺探索。论文从去中心化与中心化协作两个视角形式化基于偏好的 MAS，并提出通用的多智能体偏好学习框架 **MAPL**：通过把当前解与各 agent 生成的去中心化或中心化解对比来迭代更新，并以带学习 reward model 的 MARL from human feedback（MARLHF）和多智能体 DPO（MADPO）实例化。在协作写作、代码、工具使用和旅行规划上，MAPL 提升了协作质量与效率，并接近使用固定明确 reward 的 MARL；MARLHF 在多数任务上优于 MADPO，但对数据覆盖、agent 与比较器模型及底层 MARL 算法较敏感。
- **arXiv**：[2609.32827](https://arxiv.org/abs/2609.32827)

#### Reinforcement Learning of Communication in a Mesh of Small Language Models (TalkMesh) (2026-09)
- **简介**：作者 Mehmet Kerem Turkcan。独立采样的多数投票会饱和，而通信可以补充采样无法提供的信息。**TalkMesh** 是小语言模型 agent 的去中心化网络，学习何时、传递什么：每个 agent 采样提案并用训练的置信头打分，最自信者广播提示，低于阈值者修订并保留得分更高的修订，gossip 共识近似按置信度加权投票；talk policy 用 GRPO 以修订前后正确性的变化为奖励，学习写提示和修订。三个 agent（至多 6 次输出）即可达到各自 32 次采样多数投票的准确率；最多 8 个 agent 训练、32 个评估时，Qwen3.5-0.8B 在 GSM8K 上从 0.568 提升到 0.705，SmolLM3-3B 在 MATH-500 上从 0.492 提升到 0.722。8 个 agent 中 4 个串通作恶时多数投票降到 0.000，而带防御的网络仍保持 0.507。
- **arXiv**：[2609.30578](https://arxiv.org/abs/2609.30578)

#### DualSQL: Text-to-SQL with Multi-Agent Reinforcement Learning (2026-09)
- **简介**：作者 Shijie Chen、Yu Gan、Yeounoh Chung 等。Text-to-SQL 系统通常是 schema linking 与 SQL 生成两个 agent 的流水线，但已有工作分别训练两个模型，未利用任务间协同。**DualSQL** 让两个 agent 共享同一模型权重与 agentic scaffold，通过多智能体 RL 联合优化，并设计三种数据库访问工具支持多步推理；引入 rollout guardrail 机制稳定多智能体 RL 训练、避免模型崩溃，并提出更准确的 SQL 正确性指标 REX（robust execution match）来给出 reward。仅用 3755 个样本训练，DualSQL-4B 在 BIRD dev 上达到 68.0% 执行准确率，DualSQL-8B 达到 71.1%，超过此前 32B 的单模型 SOTA。
- **arXiv**：[2609.18135](https://arxiv.org/abs/2609.18135)

#### UnifiedPlayers: Enhance Tool-Integrated Reasoning in Agentic Reinforcement Learning (2026-09)
- **简介**：作者 Wenjie Liao、Liangjie Zhao、Zehong Cao。自演化方法常把轨迹生成与评估分离，静态 verifier 无法适应新的失败模式，self-consistency 信号又可能强化轨迹间共有的错误；联合适配规划、执行与评估则带来协调难题——每个组件都在不断改变其他组件的训练数据或反馈。**UnifiedPlayers** 是一个合作框架：Planning Player 生成任务，Execution Player 产出带 Python 工具调用的多轮轨迹，Evaluation Player 构造可执行 verifier；在 GRPO 下设计角色专属 reward，使三者朝共同目标协调。在两个 backbone、十二个推理 benchmark 上，数学推理至少超出最强 baseline 3.5%，通用推理至少 3.9%；学得的 verifier 对抗检测准确率 84.2%，reward 的题内方差是 self-consistency baseline 的 2.03 倍。
- **arXiv**：[2609.20089](https://arxiv.org/abs/2609.20089)

#### CoSkill: Joint Reinforcement Learning of Reasoning and Meta-Skill Agents for Hierarchical Skill Evolution (2026-09)
- **简介**：Jinyuan Feng、Dongmin Li、Yiqun Chen、Yang Gao 等 7 人。指出现有 agentic RL 的技能库范式存在结构性缺陷：要么把 skill evolution 与 policy optimization 解耦，要么把 meta-skill 固化为 fixed workflow，两者都把 skill 当作被动管理的对象，限制了技能的灵活演化及其与 reasoning agent 的 co-adaptation。提出 **CoSkill**：把静态 meta-skill workflow 重写为可学习的 Meta-Skill Agent，与 Reasoning Agent 在分层技能库上联合训练——二者建模为共享单一骨干的 cooperative team，实现端到端 co-adaptation：Reasoning Agent 基于检索到的 task skill 及其子集中选出的 step skills 生成动作，其任务表现反过来指导 Meta-Skill Agent 精炼这些 step skills。在 ALFWorld 与 WebShop 上成功率达 98.4% 与 90.6%（相对此前 skill-based 与 RL 基线 +3.5 与 +6.2 pp），并在早期样本效率、渐近性能与 wall-clock 效率上占优。
- **arXiv**：[2609.04865](https://arxiv.org/abs/2609.04865)

#### SRPO: Setwise Relative Policy Optimization for Multi-Agent LLMs (2026-09)
- **简介**：Shengtian Yang、Ziyu Xiong、Yu Li、Yewen Li 等 6 人。指出多 agent LLM 在共享环境中协同多个 policy 时，现有 RL 方法通常对每个 response 或 trajectory 单独优化，即使多个输出共同引发同一次状态转移，导致「更新单元」与「系统实际执行的动作」不一致。提出 **SRPO**：把 active set（一次 transition 所消耗的最小输出集合）视为一个 multi-agent action——将成员 log-ratio 合并为单一 cardinality-normalized set ratio，赋予一个 relative advantage，并对整个 set 只做一次 clip；该形式把「分工」与「联合 co-evolution」统一为不同 set size 的动作。在数学推理与 multi-turn search 上，同一训练接口覆盖 fixed、mixed 与 dynamically routed workflow、跨四种模型规模，取得所报告对比中最强的 macro-average 结果（abstract 未给出具体数值）；并给出不同 event reduction 与 set size 下的优化稳定性诊断。
- **arXiv**：[2609.08452](https://arxiv.org/abs/2609.08452)

#### RACER: Reinforced Agent Collaboration for Explainable Reasoning on Knowledge Graphs (RACER) (2026-08)
- **简介**：来自南京大学（Yuwei Lou、Hao Hu、Jidong Ge、Xianping Tao 等 8 人，ICONIP 2026）。针对 KG-enhanced LLM 范式普遍依赖单 agent 路径抽取与固定 prompting、缺乏自适应且面对巨大搜索空间的问题，提出 **RACER**：以 semantic-aware action pruning 结合 teacher-guided reinforcement learning 从大规模知识图谱中高效抽取高质量推理路径；再用跨任务累积的 shared memory graph 搭配 attention 驱动的多路径知识精炼模块，缓解单路径生成的缺陷。整体由 GraphAgent、TemplateAgent、AnswerAgent、CriticAgent 四角色多 agent 协作系统编排，动态精炼 prompt 并评估答案。在 CommonsenseQA 与 OpenBookQA 上平均较 SOTA 的 KG-enhanced LLM 基线提升约 5%。
- **arXiv**：[2608.29263](https://arxiv.org/abs/2608.29263)

#### One Model, Many Minds: Unlocking Multi-Agent Synergy in a Single Agent via Mixture of Roles (MoRe) (2026-08)
- **简介**：来自 Zhichen Zeng、Huiyuan Chen、Jingru Cheng、Hanghang Tong 等 10 人。指出单 agent 范式靠预定义 persona 或 steering vector 引入专精，但只能施加一种固定专精、无法适配多样 query；而 MAS 虽能动态多视角求解，融合各专精却要多轮交互，抬高 context 长度与推理成本。提出 **MoRe**：学习一个多样化的 steering vector codebook，每个向量编码一个 latent role，再由 query-aware router 动态融合出一个涵盖多角色的 steering vector，用它引导骨干 LLM，从而在单 agent、单轮推理中实现多视角专精；训练采用三阶段 SFT curriculum 加 GRPO post-training，骨干 LLM 全程冻结。在 reasoning 与 personality benchmark 上平均超过单 agent 基线 2.2%，性能与 MAS 持平而 token 成本降低 20 倍。
- **arXiv**：[2608.27338](https://arxiv.org/abs/2608.27338)

#### Co-RL: Unsupervised Reasoning Emerges from Diverse Cohort in Multi-agent RL (2026-08)
- **简介**：来自 UC San Diego 等机构（Yunhao Yang、Yuexin Bian、Yunjie Tian、Yijiang Li 等 9 人）。指出 RL 提升推理的最强效果仍高度依赖 ground-truth 监督（如 verifiable reward），而 self-rewarding RL 虽摆脱标注，却会强化自身偏差与次优行为、压缩回复多样性，最终走向同质化与训练崩溃。提出 **Co-RL**：让多个**互不共享参数**的解耦模型同时用 RL 优化，reward 全部来自同伴模型（peer-derived reward）；并进一步论证提升 cohort 多样性（异构模型家族、不同规模、改写后的训练样本）可降低驱动自强化反馈回路的 correlated errors，从而稳定提升推理、保持行为多样性、缓解训练崩溃。在完全不使用任何标签的条件下，7 个纯文本 benchmark 平均提升 3.0–8.6%、4 个多模态 benchmark 平均提升 2.3–7.2%，优于此前 label-free 方法，并可匹敌或超越有监督方法。
- **arXiv**：[2608.17253](https://arxiv.org/abs/2608.17253)

#### ExRole: From Team Trajectories to Executable Roles in Multi-Agent Language Models (2026-08)
- **简介**：作者 Zhou Liu、Chaoyang Han、Zewei Pan、Zeli Su、Wentao Zhang。指出多智能体系统大多把 role 当作手写的 prompt 标签，与学到的行为及参数更新彼此脱节；主张有用的 role 应当是**可执行的控制变量**——既概括对未来 utility 有预测性的行为，又能指导后续交互，还能定位承担该行为的可训练容量。提出 **ExRole** 这一 trajectory-to-role 框架：从 prefix-local 的团队 trace 学习 future-aware role prototype，将其解析为可读指令与 token-aligned role marker，并可选地以 turn-aligned credit 路由共享的 LoRA rank slot。MuSiQue 与 2WikiMultiHopQA 上相对单 agent search 分别提升 15.0/14.4 与 13.5/16.1 EM/F1 点，相对最强的非 ExRole 对照仍保有 11.5/11.6 与 7.7/9.7 点，且一致优于 role-free、手工、随机与打乱 role 等对照；Role-Agent-Turn 干预进一步表明诱导出的 role 捕捉到超越固定 agent 身份与 turn 位置的可迁移行为专业化。
- **arXiv**：[2608.11949](https://arxiv.org/abs/2608.11949)

#### MoRSE: Task-Oriented Multi-Agent System with Mixture of Role-Subtask Experts (2026-08)
- **简介**：作者 Peiwen Li、Shiyang Zhang、Yangtian Zhang 等 6 人（含 Yale 的 David van Dijk、Rex Ying）。指出现有 LLM 多智能体系统主要依赖粗粒度的 prompt 级差异化、缺乏面向多样子任务的参数适配，导致 agent 间异质性不足、专业能力有限，成为复杂需求任务的性能瓶颈。提出 **MoRSE**，在任务结构与参数两个层面施加 (role, subtask) 条件化的专业化：结构层面把每个任务分解为依赖感知的子任务有向无环图（DAG）并为每个 agent 指派特定 (role, subtask)，实现任务级分工；参数层面提出动态 Mixture of (role, subtask) LoRA Experts 模块 + 基于 prototype 的子任务语义 router，在共享 LLM 底座上低成本获得参数级专业化。为在稀疏任务 reward 下稳定地协同优化 expert 与 router，进一步提出带**两层 credit assignment 的分层 group-relative policy optimization**，把 expert 更新与 routing 决策引入的 cross-route 方差隔离，从而解耦 expert 质量与 routing 质量。代码生成基准上跨三种 backbone 在整体任务与 step-wise 指标上均有提升，且训练得到的专业化能泛化到 held-out 任务类别与领域（abstract 未给出具体数值）。
- **arXiv**：[2608.09251](https://arxiv.org/abs/2608.09251)

#### Learning Sexism Detection Using Multi-Agent Perspectivist Preference Optimization (MAP-PO) (2026-08)
- **简介**：Hadi Mohammadi、Tina Shahedi、Robert A. Bagheri、Mehdi Dastani 等（Utrecht 方向作者）。针对标注者对「性别歧视」判断的真实分歧被多数投票抹平的问题，提出 **MAP-PO**：先按标注行为（而非人口统计属性）对标注者聚类，为每个 cluster 微调一个 LLM agent 复现该群体的标注行为，再用**同时组合个体级与团队级 reward 的 preference optimization** 协调多个 agent。在 EXIST 2024 英语/西语推文数据集、两种语言 × 两种骨干共四种设定下均得到两个一致结论：不微调时各 agent 行为几乎无差别，说明 cluster 专属训练必要；而仅用本 cluster 标签训练会使 agent 过度偏离其应代表的群体，加入共享的团队级训练信号才能保持一致性。属 §4.1 中以多 agent 偏好优化保留标注分歧的代表工作。
- **arXiv**：[2608.04056](https://arxiv.org/abs/2608.04056)

#### MADA-RL: Multi-Agent Debate-Aware Reinforcement Learning for Parameter-Efficient Reasoning in Compact Models（MADA-RL） (2026-07)
- **简介**：Pulici、Chu、Kharlamov、Ding、Tresp、Ma（LMU/慕尼黑等）提出的后训练框架，将紧凑模型（≤4B）分化为 generator 与 critic 两类角色，用「辩论感知」信号并仅经 LoRA 微调少量参数进行训练。核心贡献是 **counterfactual critic advantage**：把 critic 的 advantage 重定义为其奖励减去 generator 集成的逐样本准确率，从而实现比静态均值归一化更精准的信用分配，使 critic 学会纠正而非模仿 generator。在 5 个数学推理基准上将 DeepSeek-R1-Distill-Qwen-1.5B 从 39.9% 提到 41.9%（+2.0，p<0.001），且可训练参数比全量微调少 16 倍，位于「准确率-可训练参数」Pareto 前沿；未超过 DeepScaleR、STILL-3 等用更大数据训练的最强基线。
- **arXiv**：[2607.18006](https://arxiv.org/abs/2607.18006)

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

#### When Debate Amplifies Risk: Social-Bias Vulnerabilities in Multi-Agent Safety Deliberation (2026-11)
- **简介**：作者 Qinghua Mao、Xi Lin、Yuxin Qi 等。检索增强的 Web 应用越来越多地用多 agent debate（MAD）聚合冲突、嘈杂、异构的信息，但审议本质上是社会过程，易受从众压力、权威服从等人际动态影响，可能压过基于证据的安全边界。论文提出 **SafeConflict-Bias**——跨六类核心 Web 风险的受控反事实诊断协议，在严格保持 query 与检索证据不变的情况下，分离出对话多数派影响、权威线索与置信度不对称。在多种闭源与开源骨干上的实验显示：孤立模型能可靠拒绝不安全建议，标准 MAD 却灾难性地屈从社会压力——不安全采纳率从 0.0% 升至 100.0%，并系统性压制安全的少数派证据。为此提出免重训练的推理时审计框架 **RiskTrace-Debias**（证据优先的预承诺、社会元数据屏蔽、一致性门控仲裁），将最坏情形的不安全采纳率压低到 2.0% 以内。
- **arXiv**：[2611.04289](https://arxiv.org/abs/2611.04289)

#### Debate to Deceive: Poisoning LLM's Fact-Checking via Game-Theoretic Multi-Agent Debate (D2D) (2026-11)
- **简介**：作者 Liting Huang、Huizhi Liang、Yi Lei、Xingjun Ma、Varun Ojha。针对操纵事实核查判断的知识攻击，现有对抗后缀/模板注入与单 agent 自我反思方法分别在多种防御下鲁棒性不足、易陷入确认偏误或试错代价高。**D2D**（Debate-to-Deceive）是基于人类辩论结构与修辞风格的多 agent debate 框架，使 agent 生成语义连贯、修辞有效且能抵抗早期确认偏误的对抗性段落。它把对抗过程形式化为演化博弈：agent 通过竞争自适应演化修辞策略（Logos、Pathos、Ethos）；为应对巨大的 LLM 输出空间，把 agent 行为解耦为从紧凑策略空间中经演化启发的学习动态选择修辞动作的策略层，以及以策略与辩论上下文为条件生成具体段落的生成层。在 FEVER、HealthVer、CLIMATE-FEVER、SciFact 四个事实核查 benchmark 上显著优于现有攻击并能规避多数防御（abstract 未给出具体数值）。
- **arXiv**：[2611.08756](https://arxiv.org/abs/2611.08756)

#### HEDGE: Gated Hypergraph Memory for Scalable Multi-Agent Debate (2026-10)
- **简介**：作者 Amirhossein Abaskohi、Mahdi Mostajabdaveh、Zirui Zhou。多 agent 辩论要求 agent 跨多轮保留并回顾论点，但上下文窗口有限且记忆缺乏结构，相关信息越来越难以找回；扁平或成对图记忆也难以捕捉多 agent 一致、冲突证据与共识演化等高阶关系。**HEDGE** 把辩论轨迹表示为语义超图：超边连接跨 agent 的相关论点，超边内关系图保留细粒度论证结构；由监督学习与强化学习训练的 memory selection module（MSM）把每个记忆候选路由到 STORE、BACKUP 或 OMIT，保留有用信息并限制冗余。在七个 benchmark 上稳定优于单 agent、多 agent 与记忆基线，在 GPT-4o-mini 下比最强多 agent 系统平均准确率高 5.0 点，在冲突密集任务上增益达 8.2 点，同时记忆用量减少 48%、辩论轮数减少 34%。
- **arXiv**：[2610.38902](https://arxiv.org/abs/2610.38902)

#### AI Safety via Debate is Compromised by Cognitive Biases (2026-10)
- **简介**：作者 Gefei Liu、Sonya Rashkovan、Sophia Lloyd George、Isaac Sheidlower、Serena Booth。RLHF 中人类评估者常偏爱讨好或有说服力的回答而非真实回答；AI safety via debate 让两个 agent 就对立立场互相质询，以便向裁决者暴露谬误，其核心前提是真实论点在对抗审视下更容易辩护。论文检验当辩手使用利用人类判断偏差的修辞策略时该优势是否仍然成立：受竞技辩论启发，构造 68 段关于侦探谜案（已知真凶）的 LLM 生成对话，涵盖锚定、忽略谬误、行话、冗长四种干预，分别施加于支持真凶一方或支持无辜嫌疑人一方，以区分对裁决的影响与正确性。在 369 名参与者的研究中检验这些偏差对裁决的影响（abstract 被截断，具体结果以原文为准）。
- **arXiv**：[2610.05461](https://arxiv.org/abs/2610.05461)

#### Do We Need Complex Topology Control? Distinct-Peer Random Routing Improves Cost-Efficiency in Sparse Multi-Agent Debate (2026-09)
- **简介**：作者 Boxuan Wang、Zhuoyun Li、Xiaowei Huang、Yi Dong。多 agent 辩论 (MAD) 通过迭代的同伴交互提升 LLM 推理准确率，通信拓扑是核心，因而出现越来越复杂的学习、自适应或动态重构交互的机制，而此前研究表明简单的稀疏通信已能以低得多的成本取得有竞争力的表现。论文仔细审视稀疏 MAD，发现一个简单的无放回随机路由策略（每轮每个 agent 与两个不同且新采样的同伴辩论）是出人意料的强基线，稳定改善稀疏 MAD 的准确率-成本权衡；并进一步研究辩论终止，轻量终止即可显著降低推理成本且保持有竞争力的准确率。结论是复杂拓扑控制（如学习式拓扑适应）应先与强简单路由和终止基线比较（abstract 未给出具体数值）。
- **arXiv**：[2609.27150](https://arxiv.org/abs/2609.27150)

#### Challenge-Aware Knowledge Recruitment with Meta-Agent Supervision for Medical Multi-Agent Reasoning (2026-09)
- **简介**：作者 Riccardo Riccio、Haitz Sáez de Ocáriz Borde、Pietro Liò。LLM 在较难的医学推理上表现下降，多 agent debate 往往无法胜过强的单模型 prompting；作者认为关键局限在于现有系统仅依据问题选择专家与检索知识，而非依据推理中浮现的缺口。提出 challenge-aware 自适应招募：外部 CHALLENGER 检查演化中的推理轨迹，识别具体的事实或专业知识缺口，触发针对性的知识检索或讨论中途的专科 agent 招募。在 MEDAGENTSBENCH hard 子集（九个生物医学数据集共 862 题）上，前置检索收益很小，而 challenge-aware 的中途检索更好；在多专科 backbone 上 9 个数据集中 8 个准确率提升，比仅 Challenger 高 6 个百分点；Challenger 触发的专科招募把 GPT-4.1-mini 在 MedQA 的准确率从 46% 提升到 54%。单纯同时使用检索与专科招募会降低性能，说明需协调各干预通道。
- **arXiv**：[2611.05393](https://arxiv.org/abs/2611.05393)

#### A Layered Analysis of Disagreement And Answer Quality in Multi-Agent LLM Debate (2026-09)
- **简介**：Chen Qian（单作者）。针对「multi-agent debate 通过暴露真实 disagreement 提升答案质量」这一广被假定但很少被检验的机制，提出四层测量：(A) debater 自报的 agreement；(B) 回复文本是否真的反驳；(C) 移除诱发指令后立场是否持续；(D) 开源权重模型中立场在自身 token log-probabilities 上的反应。在 GlobalOpinionQA 上用三模型委员会做 750 场 debate、三种语气（friendly / neutral / hostile）：语气强烈重塑自报一致性，完全 agreement 在 friendly 与 hostile 两端相差 50.4 个百分点；仅读回复文本的 judge 能复现同一模式；删除 hostile 指令后标签回退为 agreement 的比例比保留指令的 matched re-ask 高 23.1 点（首轮 p=0.0625 不显著、合并所有轮次 p=0.016 显著，28 例首轮回退中仅 11 例同时体现在回复文本里）；反对论据更一致地削弱立场 margin 而非改变方向。最终答案未见质量提升：经偏置校正的 jury 给出 299/299 平局，可验证对照任务准确率不变，而未校正的 jury 曾有 66% 判 debate 胜出——实为阅读顺序造成的 artifact。属 §4.2 的辩论分歧与答案质量关系的实证分析工作，不提出训练方法。
- **arXiv**：[2609.08016](https://arxiv.org/abs/2609.08016)

#### MABPD: Multi-Agent Bias Probing & Detection via Structured Argument Debate (2026-09)
- **简介**：来自印度 Graphic Era University（Garvit Joshi、Stavya Dhyani、Jasmine、Arun Chauhan），EMNLP 2026 Main。针对新闻媒体偏见依赖 loaded language、selective framing、strategic omission 等细微语言线索、单模型难以检测且传统上需大规模标注语料监督训练的问题，追问结构化多 agent 深思能否作为监督分类的 training-free 替代。提出 **MABPD**：三个专职 LLM agent 从互补视角分析文章，并通过 Structured Argument Debate（SAD）协议消解分歧——SAD 实现领域驱动的 asymmetric burden of proof（无文本证据支撑的 biased claim 权重为零），配合 role-weighted voting 与 post-consensus verification，用显式的审议结构替代任务专属的监督决策边界。消融显示起作用的是结构化审议而非单纯 agent 并行：移除 debate 模块使 F1 最多下降 10.6 点。在 BABE（4,121 条专家标注句）上取得 83.4% macro F1，距监督 SOTA（MAGPIE 84.1%）仅 0.7 pp，且无任何任务专属训练或阈值调优；跨数据集在 SemEval 2019 HyperPartisan（644 篇）上 zero-shot 准确率 75.0%，距监督 SOTA（82.2%）7.2 pp。属 §4.2 的结构化辩论机制在偏见探测上的 training-free 评测/诊断工作，不涉及参数更新。
- **arXiv**：[2609.04841](https://arxiv.org/abs/2609.04841)

#### When Agents Disagree: Bayesian Backward Reasoning as a Label-Free Anchor for Multi-Agent Collective Decision-Making (2026-09)
- **简介**：Ken Chen、Wei Wang、Sachith Seneviratne、Hansani Weeratunge 等 5 人。指出当多个 LLM agent 给出冲突答案时，voting、electoral rules、LLM judge 等既有 collective decision-making 方法都依赖 forward reasoning（证据→标签的单向映射），因而聚合的估计共享同一 factorization，会继承 forward pool 内的相关性误差。方法从显式 likelihood 出发、通过 Bayesian backward reasoning 为每个实例构造 reverse posterior，使 forward 与 reverse posterior 成为对同一后验的不同 factorization 近似；再用 Jensen-Shannon divergence 按 cross-path consistency 对 agent 排序，并据此给出三种策略：硬选择 **MinJS**、软重加权 **FwdJS**、对数线性融合 **LogLin**。在 DDXPlus、五种 LLM backbone 上：MinJS 在全部 backbone 上优于随机选择，FwdJS 普遍优于最强基线，LogLin 取得所评方法中最优、且在 agent 分歧子集上增益最大；有标注数据时可用轻量两阶段 calibration 进一步精炼 reverse anchor（abstract 未给出具体数值）。属 §4.2/§4.3 交界处的推理期共识与 agent 加权机制研究，非 RL 训练中的 credit assignment。
- **arXiv**：[2609.11709](https://arxiv.org/abs/2609.11709)

#### Remember and Reweight: Enhancing Multi-Agent Debate with Experience Memory and Confidence Estimation (R²-MAD) (2026-09)
- **简介**：来自中科院自动化所与 UCL（Xuanfa Jin、Zhijian Ma、Yongcheng Zeng、Haifeng Zhang、Jun Wang 等 6 人，EMNLP 2026 Findings）。指出 multi-agent debate（MAD）存在 shared misconception 这一关键脆弱性——当多数 agent 初始收敛到错误答案时，debate 过程会放大而非纠正错误；已有方法多只处理 peer skew，未触及 agent 自身有偏的 concept prior。提出 **R²-MAD**，为 agent 配备从历史 debate 累积的 experience memory，用两个互补机制同时干预两种失效模式：debate-state-aware 的检索策略依据当前 consensus 水平检索相关历史证据来校准 concept prior，再基于检索到的经验估计每个 agent 的可靠性并转成 confidence weight 调节 peer influence。在多个基准上一致优于单 agent 与 MAD 基线（abstract 未给出具体数值）。
- **arXiv**：[2609.03619](https://arxiv.org/abs/2609.03619)

#### Beyond Consensus: Downward Bias and Role Asymmetry in Multi-Agent LLM Judges for Subjective Evaluation (2026-08)
- **简介**：来自 Minsoo Song、Chanwoo Kim、Sugyeong Eo、Chanjun Park（EMNLP 2026 Findings）。质疑主观 rubric 打分场景下「agent 间达成 consensus 即等于对齐人类判断」的假设，将单 judge 基线与 consensus-based MAD 协议对比，并设计三组消融分别隔离 role prompting、多轮交互与显式分数共享的影响。在六个 LLM judge 上，单 judge 基线平均取得最强的人类对齐度，而 MAD 在两个任务上均出现对齐退化；消融显示性能下降主要源自非对称的 role prompting 而非交互本身——赋予「严格 judge」角色会引入系统性 downward bias 且 consensus 无法纠正，且 consensus 分数远低于严格/宽松两种独立条件的算术中点，体现 strict-stance dominance 而非简单平均。去除角色不对称（Symmetric MAD）可基本恢复基线，而屏蔽 peer 分数则扩大分歧并进一步恶化对齐。属 §4.2 的 debate 偏差/现象分析工作，无训练与 policy 优化。
- **arXiv**：[2608.30373](https://arxiv.org/abs/2608.30373)

#### Meta-Moderator: Empowering Multi-Agent Debate with Meta-Cognition (2026-08)
- **简介**：来自 Wentao Hu、Zhuoyue Wan、Jinhao Shen、Chen Jason Zhang 等 6 人（EMNLP 2026 Findings）。指出 multi-agent debate 的收益常被薄弱的 moderation 拖累：常见流水线依赖固定预算、基于一致性的停止条件或未经训练的 judge，导致冗余 deliberation 与不可靠的证据聚合。将 moderation 建模为 meta-cognitive 过程（监控 debate 效用、控制 deliberation、裁决最终答案），提出 **Meta-Moderator** 这一可学习框架，通过 outcome-driven policy optimization **独立于 debater 单独训练**，使 debate 调控成为显式能力而非 prompting 的副产物。在五个 benchmark 上优于常用 decision layer，并可跨任务与跨系统配置迁移；分析显示它更有选择性地分配 debate 预算，并在信息性 hypothesis 出现后减少 mis-aggregation（abstract 未给出具体数值）。
- **arXiv**：[2608.23029](https://arxiv.org/abs/2608.23029)

#### Group Perspective Matters: Regulating Debate Relationships Can Mitigate Blind Conformity in Multi-Agent Debate (DEAR) (2026-08)
- **简介**：来自北京交通大学（Hao Wu、Shoucheng Song、Chang Yao、Huaiyu Wan 等 7 人）。指出 Multi-Agent Debate 中 LLM 极易 blind conformity（盲目从众），而基于 confidence/perplexity 的个体评估无法反映推理正确性、甚至加剧从众。作者把视角从个体评估转向群体交互，将 LLM 间的相互引用定义为 **Debate Relationships**，提出 **DEAR**：先把 consensus 与 divergence 量化为 group evidence 刻画 debate 状态，再分三阶段——What（感知群体协商倾向与不确定性）、Who（Selection RL-Agent 选参考 peer）、How（Behavior RL-Agent 调整生成行为），两个 RL-Agent 的执行建模为序贯决策并用 multi-agent reinforcement learning 联合优化。实验显示性能更优且 token 消耗明显下降（abstract 未给具体数值）。
- **arXiv**：[2608.03648](https://arxiv.org/abs/2608.03648)

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

#### CoEvolve: Counterfactual-Guided Co-Evolution of Memory and Collaboration in Multi-Agent Systems (2026-10)
- **简介**：作者 Xufei Shi、Derong Xu、Wei Chen 等。基于 LLM 的多 agent 系统的记忆与协作架构常是静态的；已有工作通过交互经验演化记忆或协作，但这些相互依赖组件的更新未必对齐：演化后的工作流可能需要记忆未提供的经验，演化后的记忆可能编码工作流用不上的指导。**CoEvolve** 是以 skill 为中心的对齐式记忆-协作演化框架：把可复用的局部协作模式表示为 skill，将演化的协作决策与其支撑记忆关联起来；并比较原始与反事实轨迹来识别局部演化信号，据此跨轮协调 skill 库、skill 记忆和执行者记忆的更新。在 ALFWorld 和 WebShop 上取得所评测基线中最佳的总体表现，消融实验验证了对齐式共演化与反事实引导演化的有效性（abstract 未给出具体数值）。
- **arXiv**：[2610.33539](https://arxiv.org/abs/2610.33539)

#### Which Agent's Skill Should Change? Group-Relative Credit Assignment for Evolving Multi-Agent Teams (GRSE) (2026-10)
- **简介**：作者 Yanan Ma、Yang Wang、David Le、Chenhan Yuan、Yizheng Sun、Guangwei Zhang。合作的语言模型 agent 可通过修订可复用 skill 在不更新权重的情况下改进，前提是把共享团队奖励追溯到负责的 agent 与 skill；已有方法给轨迹中出现的 skill 记功，但「出现」不等于「负责」，在成功运行中活跃的有害 skill 仍会获得正向 credit。**GRSE (Group-Relative Skill Evolution)** 受 GRPO 启发，在相同 skill 库下对每个问题多次执行并对比各次执行，据此给 agent-skill 对分配行为 credit，并对失败做结构化诊断。在 HotpotQA 与 LoCoMo、两个开源权重 actor 上，GRSE 优于最接近的前作 MASkills（HotpotQA 与除 open-domain 外的所有 LoCoMo 类别），多跳问题上提升最大（+6.7 与 +3.2 F1）；对被污染的库也稳健：从植入的有害 skill 出发，能以正确符号给每个植入 skill 记功并恢复到干净库的得分（+6.3 分），而 MASkills 误归因两个并跌破起点。
- **arXiv**：[2610.27227](https://arxiv.org/abs/2610.27227)

#### Credit Where Credit Is Due: Contrastive Attribution Topology Search for Multi-Agent Systems (CATS) (2026-10)
- **简介**：作者 Fabian Tritthart、Tom Zehle。LLM 多 agent 系统（MAS）的精度与成本取决于拓扑（包含哪些 agent、如何通信），但单一任务得分无法说明哪些 agent 有益或有害。提出 **CATS (Contrastive Attribution Topology Search)**：在固定的专才 agent 超图内搜索紧凑团队，由 LLM attributer 对比不同候选拓扑的执行轨迹并给单个 agent 分配 credit，据此更新超图的边并逐步剪除弱边与弱 agent，得到部署用的小团队。使用小型开源权重模型且候选评估次数匹配时，CATS 在 BigCodeBench 上比仅看得分的启发式搜索平均准确率高 5.3 个百分点，部署 token 少 54%；MBPP 上以更少 token 匹配最强基线；在 HotpotQA prompt-poisoning 压力测试中，三次运行都排除了所有被对抗性提示的 agent，而随机搜索会保留部分。
- **arXiv**：[2610.29292](https://arxiv.org/abs/2610.29292)

#### Compositional RL: Evaluating and Training LLMs for Collaborative Reasoning (C-RL) (2026-10)
- **简介**：作者 Shangbin Feng、Yulia Tsvetkov、Jan Kautz、David Acuna。模型常被单独评估，却越来越多被一起部署（路由、辩论、集成、协作训练）。论文定义模型的 compositional strength：用模型池上合作博弈的 Shapley value 形式化，并以其对全联盟的 leave-one-out 贡献作为可行代理（需 n+1 次而非 2^n 次系统评估）。在五个模型池、十二种协作方法与十二个数据集上，个体强度与组合强度基本不相关：15 个设置中有 12 个个体强度不能转化为组合优势。据此提出 **C-RL**：在在线多模型系统内训练策略，奖励整个系统的产出而非策略自身孤立的正确性；Shapley 的 leave-one-out 基线在 group policy gradient 下被抵消，无需额外 rollout。在三种协作协议、八个任务上，C-RL 在 24 个设置中的 21 个产出最强协作者，平均提升协作表现 4.0 到 5.4 分（约为基线增益的 3 倍），最高 13.1 分。
- **arXiv**：[2610.17456](https://arxiv.org/abs/2610.17456)

#### Error-Propagation Modeling for Failure Attribution in LLM-Based Multi-Agent Systems (EMFA) (2026-10)
- **简介**：作者 Jiaqi Liao、Yuanzhao Zhai、Huanxi Liu、Xu Zhang、Zheming Zhuang、Dawei Feng。LLM 多 agent 系统的失败归因很难，因为观察到的结果往往不能直接揭示导致失败的错误；这里的归因目标是 decisive error，即修正后可恢复失败执行的 agent-step 对。现有方法多只识别可疑步骤，没有显式建模错误如何跨交互传播或在未解决的循环中持续，难以把决定性错误与下游失败症状区分。**EMFA** 构建失败轨迹的结构化表示，同时建模级联传播与持续交互循环，用传播感知的候选筛选加反事实验证来定位决定性 agent-step 对。在 Who&When benchmark 上取得 SOTA 的 step 级归因准确率，agent 级保持有竞争力（abstract 被截断，具体提升以原文为准）。
- **arXiv**：[2610.11600](https://arxiv.org/abs/2610.11600)

#### Learning What to Skip: Counterfactual Credit Assignment for Efficient Multi-Agent LLM Workflows (LW2S) (2026-09)
- **简介**：作者 Jinfeng Xu、Zheyu Chen、Ziyue Peng 等指出，多 agent LLM workflow（规划、执行、验证、总结）中每个组件的价值取决于已产生的状态，全部执行可能浪费计算或覆盖已正确的中间答案。论文把组件省略表述为反事实 credit assignment：完整 workflow 日志揭示已执行轨迹的 reward，受控的 skip 干预则揭示省略未来某一步的后果。提出 **LW2S**（Learning What to Skip），由这些干预学习针对动作的安全性模型，结合 held-out 校准与领域原生 guard 来选择跳过；早期跳过被拒绝时控制器可继续执行并重新考虑之后的组件。在数学推理、选择题 QA 和代码生成上、两个指令模型家族中，LW2S 降低记录的 token 成本，同时在评测设置下持平或提升完整 workflow 的聚合准确率；分析还表明共享错误案例说明仅凭一致性不足以选择跳过。
- **arXiv**：[2609.30734](https://arxiv.org/abs/2609.30734)

#### GraMRAG: Orchestrating Multi-Agent Multi-Step Reasoning via Graph Memory with Reinforcement Learning (2026-09)
- **简介**：作者 Zhongyu Wang。现有多智能体 RAG 在复杂多模态推理上存在检索不足与「状态盲区」。**GraMRAG** 构建动态多模态 memory graph，把 agent 推理形式化为显式建模 action-observation 依赖的有向无环图，以缓解状态盲区、抑制冗余检索，并结合多尺度实体裁剪与 ReAct 风格视觉工具链实现跨模态长程推理。进一步提出 **TAPO**（Topology-Aware Policy Optimization）：利用图拓扑识别关键路径并对节点做有针对性的剪枝，实现多步推理轨迹上的细粒度 credit assignment。在多个多模态 benchmark 上持续优于已有 baseline 并达到 SOTA（abstract 未给出具体数值）。
- **arXiv**：[2609.14066](https://arxiv.org/abs/2609.14066)

#### DCFA: Dual-view Causal-inspired Attribution for Failure Reasoning in LLM-based Multi-agent Systems (DCFA) (2026-09)
- **简介**：Zehao Wang、Lanjun Wang、Shilong Jin、Junjie Chen、Yanghua Xiao。针对 LLM 多 agent 系统脆弱、常因推理与协调错误导致系统级失败，而 failure attribution 需从自然语言交互 trace 中定位 decisive error（最早一个被纠正即可翻转系统失败的动作）。指出两个挑战：浅层归因（现有方法只捕捉不完整检索、格式错误等可被验证机制修正的小偏差，错失决定性成因）与上下文退化（trace 变长时模型推理能力迅速劣化）。提出 **DCFA**，一个 training-free 归因框架：全局模块从系统 trace 构建结构化的 causal-inspired 依赖图以定位最初的 decisive error，局部模块施加 local counterfactual-inspired 推理来精炼归因。在 Who&When benchmark 上跨六个 LLM，step-level 准确率相对 SOTA 基线最多提升 8.27%。属 §4.3 中不做 RL 的失败归因 / credit assignment 工作（与 CockpitHAT、ASCon 等同类），无参数更新。
- **arXiv**：[2609.04749](https://arxiv.org/abs/2609.04749)

#### EDGE: Error Dependency Graph-Guided Multi-Error Attribution in Multi-Agent LLM Systems (EDGE) (2026-09)
- **简介**：来自 Jun Hou、Priya Pitre、Yi Fang、Xuan Wang（EMNLP 2026）。指出 LLM agent 失败常常包含多个相互关联的错误而非单一失误，而现有归因方法只定位责任 agent、步骤或 root cause，不显式建模错误之间的依赖关系。提出 **EDGE**：先从观测到的 error event 构建 error dependency graph，并通过 counterfactual rollout 验证出可靠的因果子集；推断图用于引导一个两阶段的 LLM-as-judge 检测器完成 error attribution，而经干预验证的子图为解释与修复分析提供更可靠依据。在 TRAIL 与 MAST 上，EDGE 在多数被测模型与设置下提升了 category-level 多错误归因效果，且在改编的 Who&When 式 prompt 下依赖图对各种 prompting 策略均有帮助（abstract 未给出具体数值）。属 §4.3 中不做 RL 的失败归因 / credit assignment 类工作。
- **arXiv**：[2609.01360](https://arxiv.org/abs/2609.01360)

#### Detect Before You Attribute: Cascade Failure Attribution for Multi-Agent Systems (DUOTRACE) (2026-08)
- **简介**：来自 Jiayi Zhang、Zexin Wang、Degang Sun、Changhua Pei 等 7 人。指出基于 topology/spectrum 的失败归因方法利用了 trajectory 结构却忽略细粒度语义，而基于 LLM 的归因方法虽能捕捉语义线索但在长 trajectory 上受长上下文退化困扰。提出 **DUOTRACE**，一个即插即用的检测过滤器，遵循 detect-before-attribute 范式：先检测异常执行，再把聚焦后的 trajectory 证据交给下游 LLM 归因方法。为在 agent trajectory 上做有效的 VAE-based 异常检测，DUOTRACE 融合语义-结构双视图节点表示、Tree-LSTM trajectory 编码器，以及 prefix-chain 与 LLM 两种数据增强，以应对异构节点、层次化执行结构和失败数据稀缺。配合六种 LLM 归因基线，agent-level 与 step-level 归因准确率分别提升 8.7% 和 7.0%。属 §4.3 的失败归因 / credit assignment 类工作（训练检测器而非 policy 优化）。
- **arXiv**：[2608.29646](https://arxiv.org/abs/2608.29646)

#### Adaptive Influence Graphs for Failure Attribution in Multi-Agent Systems (AIG) (2026-08)
- **简介**：来自 Yarden Bakish、Amir Dudai、Roy Ganz、Ron Litman 等 7 人。针对多 agent LLM 系统失败难以定位、failure attribution 仍主要依赖人工工程师的问题，观察到工程师并不通读原始日志，而是借 observability 工具围绕组件、动作与依赖关系组织 trace 以做定向导航，并假设现代 LLM 同样受益于该范式。提出 **Adaptive Influence Graphs (AIG)**：两阶段 agentic 框架，先把失败 trace 转成结构化图，再由 agent 主动遍历该图以定位关键错误。跨多个模型验证更丰富的 trace 表示一致改善 failure attribution，其中 adaptive 图构建 + agent-directed traversal 效果最佳，并在多 agent failure attribution 标准 benchmark Who&When 上取得新 SOTA（abstract 未给出具体数值）。属 §4.3 的失败归因 / credit assignment 类工作，本身不含 policy 优化或参数更新（沿 CockpitHAT、ASCon、AFANet 的收录惯例）。
- **arXiv**：[2608.24361](https://arxiv.org/abs/2608.24361)

#### Beyond LLM-Based Reasoning: Lightweight GNNs for Agent Failure Attribution (AFANet) (2026-08)
- **简介**：来自 UIUC 等机构（Ting-Wei Li、Yuanchen Bei、Xiao Lin、Hanghang Tong）。面向 Agent Failure Attribution 任务（给定失败的多 agent trajectory，定位出错 agent 及其错误类型），指出现有方案几乎全靠 LLM——直接 prompting、在合成数据上微调、或复杂 agentic pipeline，长上下文处理、昂贵后训练与手工工作流带来大量开销，且即便最强模型在现有 benchmark 上准确率仍有限，说明单纯放大模型规模不够。提出 **AFANet**：一个轻量图（GNN）框架，用 step-level 语义信号刻画交互 trajectory、用 agent-level 关系建模 agent 间结构来完成归因。参数量显著更少、推理成本接近于零，却能在 in-domain benchmark 上匹配或超过包括微调模型在内的 LLM 基线，跨不同 GNN 架构保持稳健，并可用廉价的 test-time adaptation 在 OOD benchmark 上进一步改进（abstract 未给出具体数值）。属 §4.3 的失败归因 / credit assignment 类工作，本身不含 RL 训练。
- **arXiv**：[2608.18575](https://arxiv.org/abs/2608.18575)

#### ASCon: A Direction-Aware Reciprocal Agent--Step Contextualization Model for Failure Attribution in Multi-Agent Systems (2026-08)
- **简介**：作者 Shuyu Jiang、Yue Ran、Kaiyu Xu、Xingshu Chen 等 8 人。LLM 多智能体系统（MAS）的 failure attribution 要回答"谁导致失败、何时发生、为何发生"，即识别 faulty agent / erroneous step / failure mode 三类归因目标；现有方法多为单一目标训练专用模型，忽视目标之间的证据依赖——尽管目标不同，它们都依赖 MAS trajectory 中共同的诊断证据（任务约束、agent 角色、行为历史、agent 间交互）。据此提出 **ASCon** 统一表示模型：用 direction-aware graph attention 建模执行上下文，用 masked step-to-agent attention 构造行为感知的 agent 表示，再以 agent-conditioned step contextualization 把 agent 上下文回注到 step 表示；得到的上下文化表示只需接轻量的 target-specific head 即可服务不同归因目标。实验显示 faulty-agent 检测 micro-accuracy 提升 5.83%+、faulty-step 检测提升 10.63%+、failure-mode 检测 Macro-F1 提升 14.73%+，并能显著增强 LLM-based 方法在 out-of-domain 场景下的归因能力。属 §4.3 的失败归因（credit assignment）表示学习方法，训练判别式归因模型而非做 RL policy 优化。
- **arXiv**：[2608.10646](https://arxiv.org/abs/2608.10646)

#### Discovering Efficient and Explainable Communication Topologies for LLM-based Multi-Agent Systems via Causal Inference (E2-Explainer) (2026-08)
- **简介**：作者 Junzhi Li、Peng He、Qirui Ji 等 6 人。指出 LLM 多智能体系统性能很大程度取决于通信 topology，而现有 topology 生成方法通常仅由 task-level reward 驱动黑箱优化，无法解释"为何选中某些通信边"，也难以识别真正支撑成功协作的关键通信子图。提出 **E2-Explainer**：一个 model-agnostic 框架，为任意 topology generator 产出的拓扑提供可解释归因——把 topology 解释形式化为因果归因问题，寻找由边级 task-preservation 证据支撑的紧凑通信子图；证据来自一个 Granger-style 目标，度量 mask 掉每条通信信道后任务结果与最终响应稳定性的变化；再把这些预算受限子图蒸馏进一个 amortized explainer，使部署时无需反复做边级评估即可高效事后解释。多个推理与代码基准上能识别出保持成功协作的关键子图，且这些子图可直接执行以剪除冗余通信边，在维持竞争力性能的同时大幅降低通信成本（abstract 未给出具体数值）。属通信拓扑的边级归因（credit assignment）与可解释性分析工作，本身不做 policy 优化。
- **arXiv**：[2608.12921](https://arxiv.org/abs/2608.12921)

#### CockpitHAT: Dependency-Graph-Driven Hierarchical Attribution for Embodied Multi-Agent Cockpits (CockpitHAT) (2026-08)
- **简介**：作者 Wei Wang、Shuanghe Liu、Zhu Zhuo 等 7 人。针对 LLM 多智能体系统的 Correctness Collapse——任务级高准确率掩盖过程级失败，在汽车座舱这类安全关键 embodied 场景中字面正确的话语也可能触发危险物理操作，而现有归因只靠文本 trace。提出 **CockpitHAT**：用交互 DAG 上的 dependency-distance 阈值替代位置窗口界定责任范围，经 embodied adapter 融合多通道证据，并在 confidence-weighted analyst consensus 中对高风险失败施加 safety-uplift；同时发布 **CockpitBench**（212 条失败 trace，覆盖对话/车辆状态/环境/记忆四通道，三专家共识标注 ASIL 严重度）。Who&When 上 agent-level / step-exact 达 77.9%/37.8% 与 86.5%/46.0% 两个 split，最多超纯文本 SOTA 方法 ECHO 17.6/16.7 点。属 §4.3 的失败归因（credit assignment）诊断方法与基准，不做 RL 训练。
- **arXiv**：[2608.01805](https://arxiv.org/abs/2608.01805)

#### MARS-RA: Rank Aggregation for Credit Assignment via Multimodal Comparisons in Embodied Multi-Agent Cooperation (2026-07)
- **简介**：Dawei Wang 等（Newcastle 等团队，ACL 2026 Main）将合作型多 agent RL 的信用分配重构为**排名聚合问题**：由大型多模态模型对 agent 间贡献做成对比较，把绝对估计转为相对估计，从而对噪声与 agent 数量动态变化更鲁棒；比较结果转成贡献分数用于 potential-based reward shaping。提供收敛性与鲁棒性理论证明，并将 Shapley 值作为可解释参照。定位为具身多 agent 协作下的信用分配新范式。
- **arXiv**：[2607.27967](https://arxiv.org/abs/2607.27967)

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

#### EpiAgent: Epistemic Self-Play for LLM Agents (2026-10)
- **简介**：作者 Xuanfa Jin、Xinyu Cui、Zhijian Ma、Xue Yan、Jian Zhao、Haifeng Zhang。认识博弈论通过 agent 对他人持有的信念及其更新解释策略行为，而现有 LLM agent 在不完全信息下缺少显式学习和更新这类信念的机制。**EpiAgent** 为 LLM agent 增加显式、可学习的 epistemic 层，维护对他人私有信息的信念并与自身可得信息合成为 epistemic state 用于选择动作；策略被分解为两个耦合阶段——Epistemic Update（据交互历史更新信念）与 Action Planning（基于更新后的认识状态选择动作）；将其依赖关系建模为双层优化，提出 **Epistemic Self-Play** 算法，借助 self-play 交替优化两阶段，并用内在预测奖励衡量推断信念对后续观察的预测质量。在混合与竞争场景下持续优于基于 prompt 和基于 RL 的基线，进一步分析显示专门的 epistemic 训练提高了信念推断准确度（abstract 未给出具体数值）。
- **arXiv**：[2610.40227](https://arxiv.org/abs/2610.40227)

#### CoSTA: Co-Evolving Self-Play Timeline Agents for Complex Temporal Knowledge Graph Question Answering (2026-10)
- **简介**：作者 Haikun Liu、Zhichao Yan、Shaoru Guo、Victor Gutierrez Basulto、Jiapu Wang 等。复杂时序知识图谱问答（TKGQA）中，单 agent 在同一推理上下文中检索事实，不同事件历史的证据会交织、干扰项会误导后续检索；多 agent 分摊调查可分离证据，但问题分解、子任务调查与证据综合相互依赖，协作必须被学习。**CoSTA** 将事实组织成可导航的时间线，由 worker 承担调查任务，coordinator 综合其带时间戳的原始观察；先用 role-conditioned distillation 初始化共享策略，再通过以最终答案为奖励的 co-evolving self-play 联合优化两个角色。在三个 benchmark 上取得 SOTA 总体表现，在复杂多事实问题上优势尤为突出，且其紧凑变体优于所评测的闭源 LLM agent。
- **arXiv**：[2610.38450](https://arxiv.org/abs/2610.38450)

#### CasePlay: Self-Play Reinforcement Learning from Case Reports for Medical Reasoning (2026-10)
- **简介**：作者 Hao Wang、Zihan Wang、Kai Wu、Jian Yao、Yiwen Ye、Yongcan Luo 等。提升 LLM 临床推理的 RL 流程常依赖专家标注问题或昂贵的合成数据，而医学文献中的临床病例报告是大量高质量无标注文档。论文研究基于文档的 self-play 能否把病例报告转为 RL 训练信号，并识别现有 self-play 的两个问题：知识点固着（proposer 反复追问报告中少数显著线索）与难度-质量错配（难度合适的问题仍可能形式差或浅显）。**CasePlay** 包含 (a) Knowledge-Conditioned Proposer，把出题锚定到病例特定知识点以覆盖更广证据；(b) Rubric-Judge Reasoner，在答案准确性反馈之外加入对线索整合、推理深度与选项质量的 rubric 质量反馈，引导 proposer 产出推理密集的问题。在七个医学相关 benchmark 上优于现有 self-play 基线，使 SFT 热启动模型平均提升 3.1 分，LiveClin 上提升 6.3 分。
- **arXiv**：[2610.24952](https://arxiv.org/abs/2610.24952)

#### Peer Pressure: Execution-Verified Co-Evolution of Attacker and Defender Populations for Code Security (PRPO) (2026-10)
- **简介**：作者 Khushbu Pahwa、Yubin Ge、Daniele Bonadiman、Xiaofei Ma、Yi Zhang。基于静态漏洞数据集训练的防御者会落后于不断适应的对手，因此论文联合训练攻击者与防御者，并让执行结果决定每个奖励：攻击者（Red）编辑真实仓库，只有补丁可应用、可构建、通过测试、完成开发任务且在隔离容器中复现真实 exploit 才有奖励。**PRPO (Peer-Relative Policy Optimization)** 让双方都是同一冻结基座上的 LoRA adapter 种群，并相对面对相同条件的同角色 peer 记功，把种群放进策略梯度 baseline 而非对手池，并用锦标赛保持多样性。其 7B diff reviewer 在三种未训练语言的四个未见应用上达到 64.8% balanced accuracy，比监督安全微调高 5.5 点、比冻结的前沿 reviewer（Claude Opus 4.8）高 4.4 点；9B 仓库 agent 在 172 个 held-out VulnGym 公告中找出 23 个（监督微调为 10、未训练同骨干为 7）；不改算法用于 jailbreak 防御，在另一 judge 下把攻击成功率从 36.0% 降至 4.6%（WildGuard 为 8.9%）。
- **arXiv**：[2610.24896](https://arxiv.org/abs/2610.24896)

#### Explorer-Zero: Improving Self-Play for Long-Horizon Tasks with Decomposable Verifiable Rewards (2026-10)
- **简介**：作者 Tianyu Liu、Davide Belli、Bence Major、Arash Behboodi、Tribhuvanesh Orekondy。长 horizon 交互任务的 RL 通常依赖人工标注任务与可靠验证，在有状态环境中代价高昂。**Explorer-Zero** 是 self-play 框架，通过环境交互联合生成任务与细粒度可验证奖励：学习到的 proposer 探索环境、构造任务，并把中间与最终状态记为验证目标；这些检查点奖励部分完成，proposer-solver 交替训练使生成的课程自适应。作者从理论上证明，当不成功 rollout 完成的子任务数不同时，子任务奖励能减少它们之间的坍缩。在没有人工标注训练任务的情况下，Explorer-Zero 在 ALFWorld、ScienceWorld、AppWorld 上优于对比的 self-play 基线；AppWorld 上 30B agent 达 59.67% 任务成功率，接近人工标注任务上 RL 的 62.65%；self-play 初始化还能提升后续人工数据 RL 的表现。
- **arXiv**：[2610.19868](https://arxiv.org/abs/2610.19868)

#### GameSmith: Scaling Zero-Sum Self-Play with Synthesised Games (2026-09)
- **简介**：作者 Weixian Waylon Li、Jiaxin Zhang、Han Zhou 等指出，在双人零和游戏上后训练 LLM 能提供可验证 reward 且增益可迁移到通用推理，但现有流程依赖少数手写游戏及与之绑定的 advantage 估计器，游戏供给和大规模语料上的 advantage 估计都限制扩展。**GameSmith** 对两者都做扩展：让 LLM 在 reward、观测与隐藏信息代码固定的模板内合成游戏，经 seed-paired replay 与规范的独立重实现验证后扩展为编码、规则、规模变体，从三个 SPIRAL 模板得到 29 个家族中的 26 个新游戏、336 个已验证变体；先蒸馏更大教师（Qwen3-32B）在这些游戏上的 self-play，再用 **CRAB**（基于对位置期望回报收缩估计来为每个决策赋 credit 的 advantage 估计器）做 self-play RL。在 Qwen3-4B-Base 上，蒸馏使 13 个基准的平均 pass@1 从 45.5 升到 62.5，高于同预算下三个 SPIRAL 游戏，加 CRAB 的 self-play 达到 64.9；趋势在多模型上成立，迁移随游戏家族数增加而增强。
- **arXiv**：[2610.13995](https://arxiv.org/abs/2610.13995)

#### Self-Play Pretraining with Zero Data (2026-09)
- **简介**：作者 Aditya Cowsik、Kfir Dolev、Michael Y. Li、Noah D. Goodman 等。预训练数据仍主要由人工为模型筛选，论文设想让模型自己生成最有助于提升自身的数据，从而使数据受算力而非人类知识所限。受 Solomonoff 归纳启发，从随机初始化起，两个模型同步学习：generator 提出由通用图灵机解释的程序并生成字节序列，learner 自回归预测这些序列；learner 用标准交叉熵训练，generator 用 RL 训练以产生处于 learner 能力前沿的序列，形成自适应课程。由于 generator 与 learner 都未接触自然数据，这是对迁移的干净检验：在多个自然数据集上，zero-shot loss 随算力呈可预测的缩放，模型还表现出 in-context learning 并发现可识别的数学序列。
- **arXiv**：[2609.30063](https://arxiv.org/abs/2609.30063)

#### Information-Gain Rewards over Diversity-Pruned Tests: GT-Anchored Verifier Co-Training for Reliable Code Generation (CoVer) (2026-09)
- **简介**：作者 Ana Nunez、Peyman Najafirad。让单个模型既当 coder 又当 test 作者的自博弈代码 RL 存在两大病灶：宽容坍缩（通过率奖励被平凡、无区分度的测试最大化）和集中偏差（i.i.d. 采样测试聚集在众数输入上，抬高估计方差）。**CoVer**（Co-trained Coder and Verifier）是单策略 GRPO 框架：用信息增益 (IG) 奖励，按测试的 pass/fail 向量与以真值为锚的分级正确性信号之间的互信息给每个自生成测试打分，并按协方差符号门控，仅奖励正向区分的测试；再通过三阶段多样性感知筛选（无效、输入字符串、执行特征过滤）得到行为非冗余的测试集。在五个基准上，相对 Qwen2.5-Instruct，7B 的 pass@1 提升 5.8 点、14B 提升 7.1 点；作为 CodeT 排序流水线的 backbone，CoVer-7B 再增 3.5 点。
- **arXiv**：[2609.21208](https://arxiv.org/abs/2609.21208)

#### Adversarial Closed-Loop Curriculum for Evolving Role-Playing Agents (AdvRole) (2026-09)
- **简介**：作者 Zheng Zhang、Liu Liu、Qi Chai、Deheng Ye、Peilin Zhao、Mao Zheng。角色扮演 agent 的 RL 方法通常在学习开始前收集的固定场景池上训练，随着 agent 进步，它表现差的场景也在变化，而训练分布保持静态，形成分布瓶颈。**AdvRole** 是对抗式上下文改写框架，把角色扮演 RL 变成闭环课程：交替训练学习角色扮演的 Actor 和一个把角色设定与对话上下文改写成针对当前 Actor 的难场景的 Rewriter；Rewriter 以性能差距奖励训练，偏好使当前 Actor 相对原始场景得分下降的改写，使场景池随 Actor 演化并持续针对其尚未掌握的区域。在涵盖中英文的三个角色扮演基准及作者发布的新多语言基准上，一致优于基线（abstract 未给出具体数值）。
- **arXiv**：[2609.28609](https://arxiv.org/abs/2609.28609)

#### Scaling Multi-Agent Systems with Prospect-State Propagation (PspMAS) (2026-09)
- **简介**：Zhimei Chen、Mu Chen、Fakhri Karray，EMNLP 2026 Findings。指出当前 LLM 多 agent 系统为容纳更多 agent 而周期性压缩中间状态以削减推理期 token 开销，但在经济仿真中这类朴素 scaling 会丢弃语义丰富的经济状态（即 agent 行为 trajectory）——而它们正是宏观经济波动的关键驱动。论文揭示了仿真过程中 agent heterogeneity 逐步衰减的现象，提出 **PspMAS**：把每个 agent 的微观状态解耦为紧凑的 Prospect State 与富表达的 Semantic State，前者受 prospect theory 启发、通过轻量可并行的 propagator 记录心理轨迹并持续向系统注入 heterogeneity，后者发挥 LLM 的感知、推理、规划与决策能力，二者互补构成可扩展的多 agent 仿真方案（abstract 未给出具体实验数值）。属 §4.4 的多 agent 经济仿真现象研究与机制设计（heterogeneity 衰减的度量与缓解），不涉及 policy 优化或参数更新。
- **arXiv**：[2609.08033](https://arxiv.org/abs/2609.08033)

#### Bilevel Coordinated Reflection: A Game-Theoretic Approach to Multi-Agent LLM Systems (SRMA) (2026-09)
- **简介**：来自 Yihang Chen、Yuxiang Chen、Yuxuan Huang、Meng Fang、Jun Wang 等 6 人。针对 orchestrator-worker 式多 agent LLM 系统缺乏对 coordination、memory 改进与外部验证作用的统一理论刻画，将 orchestrator 与 worker 的交互建模为 bilevel coordination game：在 bounded coupling 下 worker 的局部更新博弈是近似 potential game，其均衡 slack 由任务分解质量控制；并把 reflection 视作语义 memory 状态上的随机游走，给出有限时间上界、最坏情形紧性以及在可证伪的 persistent-harm 条件下的正下界。进一步证明信息论不可能性结果：仅观察生成 transcript 的 gate 无法在文本不可区分的环境上一致改进，而 environment-grounded gate 可以；据此提出 **SRMA**（Stochastic Reflective Memory Ascent），仅当 grounded evaluation risk 严格下降时才接受候选 memory，在校准与非退化纠正质量条件下可精确、几何或多项式速率收敛，并给出随机评估的 confidence gating 与分段平稳环境下的 re-anchoring 保证。在 500 个 SWE-bench 实例上，完整的 Kimi 系统解决率 72.2%，对比公开 mini-SWE-agent 参考的 70.8%。属 §4.4 的博弈论建模与理论分析工作（无 policy 参数更新，改进发生在 memory 层面）。
- **arXiv**：[2609.02750](https://arxiv.org/abs/2609.02750)

#### The Chase Is the Curriculum, the Capture Anchors the Credit: Pursuit-Evasion Self-Play for Zero-Data LLM Reasoning (LURE) (2026-08)
- **简介**：来自 Jing Yu、Shengchao Chen、Yiyun Tan。指出 RLVR 依赖大规模人工构造任务集，而现有 zero-data self-play 只能通过事后探测-拒绝来筛查可学习性，从不学习「该把任务放在环境难度轴的哪个位置」，且仅用稀疏终局 reward 给 solver 记功。提出 **LURE**，把 zero-data self-play 重构为 pursuit-evasion game：LLM evader 在各环境难度轴上摆放任务、力求领先 planner-executor pursuer 一步，pursuer 则通过可验证交互追捕；evader 用 capture-frontier reward 训练，该 reward 在 solver 恰好在一半 rollout 中捕获它时取峰值，把「勉强可解」从手调拒绝带变成可学的定位策略；pursuer 获得 capture-anchored 稠密 process credit，即把单调的 verifier 进度与终局捕获联合做 group normalization，并施加 round-anchored KL 以稳定 co-evolution。在三个可验证 reasoning 环境、三个骨干家族上于 unified/specialist 设置下均优于先进基线，unified 模型在来自三个任务族的九个 held-out benchmark 上的聚合 OOD zero-shot 准确率超过所有训练过的基线（abstract 未给出具体数值）。
- **arXiv**：[2608.21871](https://arxiv.org/abs/2608.21871)

#### Markets, Not Planners: Decentralized Orchestration of LLM Agents with Private Information (AgentLance) (2026-08)
- **简介**：来自 Xiao Liu、Haoyang Li、Songwei Li、Fengli Xu、James Evans 等 7 人。指出现有 orchestration 多为中心化 planner 逐一派活，随 agent 池增长成为瓶颈，且需要 agent 的私有信息（如执行成本）、易被操纵——在中心化 LLM allocator 下仅插入一条偏好就能让被偏爱 agent 的任务份额近乎翻倍。提出 **AgentLance**：一个重复 labor market，agent 依据私有成本与自维护的 strategy notes 对任务竞标，allocator 依 bid 与公开 reputation 记录选出中标者，并用 VCG 式支付规则奖励成本感知的报价；复杂任务通过层级委派处理，中标 agent 可拆解任务并经同一机制转包。在数学推理、代码生成、知识密集 QA 与 agentic 任务上，AgentLance 能把 agent 匹配到其专长，并随成本敏感度上升把工作转移到更便宜的 agent，一致优于单模型、中心化 orchestration 与市场基线；进一步诊断成本自估不准、次优 bidding 等市场失灵并在受控实验中纠正后获得额外增益（abstract 未给出具体数值）。机制侧为 reward/支付规则设计与博弈激励分析，agent 策略经 strategy notes 在上下文中演化而非参数更新。
- **arXiv**：[2608.23867](https://arxiv.org/abs/2608.23867)

#### The Open-Strategy Dictator Game: Cooperation Under Mutual Transparency (OSDG) (2026-08)
- **简介**：来自 Michael Glass（单作者，机构未标注）。提出 **OSDG**（Open-Strategy Dictator Game）：经典 dictator game 的变体，每个玩家的策略是一份对所有参与者公开可见的自然语言文档，dictator 选择 SHARE 还是 TAKE 可依赖 recipient 策略文本的内容，由一个 LLM 在 recipient 策略的上下文中解读 dictator 策略并裁决每次交互。在多样策略间跑 round-robin tournament 得到 payoff matrix，再用 softmax equilibrium frequencies、dominance analysis 及对合作相对价值的敏感性分析加以考察。结果是「有条件合作」策略（对合作者 share、对剥削者 take）持续占优，而无条件策略（always share / always take）被弱支配，表明在 agent 能互相检视对方决策程序的环境中，conditional cooperation 在很宽的 payoff 参数范围内具备演化稳健性（abstract 未给出具体数值）。属 §4.4 的博弈论/现象分析工作，不含策略训练。
- **arXiv**：[2608.14913](https://arxiv.org/abs/2608.14913)

#### Emergent Misaligned Communication in Long-Horizon Multi-Agent LLM Commerce (2026-08)
- **简介**：来自 Zeyuan Li、Lukas Petersson、Alessandro Acquisti、Michiel A. Bakker（机构未在 arXiv 页标注）。针对前沿 LLM agent 越来越多地代表不同 principal、用自然语言而非结构化 API 完成交易的现实，指出安全文献多以单 agent 的对抗诱导评测或风格化任务为主，而在长时程、多 principal、真实运营状态与 agent 间自然语言交流并存的环境里，misalignment 的普遍性与结构仍严重缺乏度量。作者分析 Vending-Bench Arena（覆盖 13 个前沿 LLM 的竞争性自动售货环境）20 次一年期模拟中的 2,583 封 agent 间邮件，把 speech-act misalignment 操作化为包含虚假事实陈述、manipulation、collusion 或威胁的邮件，并结合 ground-truth 模拟器状态与记录下的 reasoning trace 做分类与验证。主分类器下 12.6% 的邮件被判为 misaligned，20 次运行全部出现、覆盖 74.7% 的单个 agent-run，且在不同采样温度的重复分类以及换用另两个前沿模型家族做 judge 的全流程复现下，其量级与构成均保持一致；misalignment 还表现出互惠性与压力条件依赖——收到对手方的 misaligned 邮件使自身回复 misaligned 的 odds 上升 1.65 倍，低库存条件使其上升 1.58 倍；能力不对称利用的检验未发现更强模型会差别性剥削更弱对手，模型性能排名也不预测 misalignment 率。属 §4.6 的现象/度量类工作（在既有 benchmark 上做行为测量，不含训练）。
- **arXiv**：[2608.14825](https://arxiv.org/abs/2608.14825)

#### Do LLMs Beat Nash? Testing Decentralized Coordination in Self-Play Multi-Agent Games (2026-08)
- **简介**：来自 McGill（Deborah Sinishaw、Qile Zhu、Edwin Meriaux、Gregory Dudek）。质疑"无中央控制器的 LLM agent 必须依靠通信才能协调行动"这一常见假设，追问在完全无通信时、同一模型的独立实例能否仅凭对同类对手的推理超过 uncoordinated play 的标准博弈论基准。构建 one-shot、无通信博弈基准：每个模型只被告知对手运行同一模型，并以底层博弈的 Nash equilibrium 为参照评测 13 个语言模型。在覆盖 7 种博弈原型、每方 2–10 个动作的两人矩阵博弈中，2 个前沿托管模型稳定超过自身 Nash 基准、在若干原型上接近最优联合结果，而多数开源权重模型只取得随博弈结构剧烈波动的部分增益；当团队博弈含 4 个及以上可互换 agent（尤其动作空间增大）时性能大幅退化，说明驱动二人 self-play 增益的能力无法迁移到更大规模的多 agent 团队。属 §4.4 的现象/机制分析与评测工作，不含训练、reward 设计或信用分配。
- **arXiv**：[2608.12547](https://arxiv.org/abs/2608.12547)

#### SearchMaster: Grounded and Regulated Self-Play for Search Agents (SearchMaster) (2026-08)
- **简介**：作者 Wentao Tan、Qiong Cao、Jiaqi Wang、Nan Duan。指出训练 search agent 需要真正要求 multi-hop 检索的任务与善用工具的轨迹，而现有 pipeline 依赖人工任务、专家演示或更强教师模型。提出 **SearchMaster**：单个 LLM 在本地检索环境中自行生成、求解并验证任务，用三个控制抑制自生成信号的误导——Evidence-Chain Generator (ECG) 以显式跨文档 evidence chain 约束任务生成、减少伪 multi-hop 问题；Search-Depth Reward (SDR) 以成功 rollout 的检索深度而非成功率评估难度，保持任务 search-intensive；Over-Opening Penalty (OOP) 抑制大量开文档的浅层浏览；经验证的 Proposer 与 Solver rollout 再用 GRPO 联合优化。六个 deep-search benchmark 上把 Qwen3.5-9B 平均准确率从 38.19% 提到 51.52%，BrowseComp-Plus 提升 30.1 点。
- **arXiv**：[2608.01822](https://arxiv.org/abs/2608.01822)

#### Interactive Alignment (2026-07)
- **简介**：Sylvain Chassang（econ.TH 主类，cross-list cs.GT + cs.MA）研究交互式 agent（含 AI 系统、团队、公司、政府）与人类福祉的长期对齐。构建"农耕博弈"：agent 群体做种植/交易/扩张决策并在"转移给人类"与"自我扩张投资"间分配产出——由于转移会减少扩张资源，演化力量倾向于**选择淘汰对齐行为**。用两条互补路径研究"宪法性原则能否使对齐长期存续"：其一是由 LLM 解释书面 constitution 的 AI-agent 模拟，其二是可解析的演化博弈论框架。结论：让 agent 依人群状态**条件化**利他与交易排斥的"务实规范执行"，比单纯利他或无条件利他执行更能维持长期对齐。属博弈论视角下的多 agent 对齐工作。
- **arXiv**：[2607.25019](https://arxiv.org/abs/2607.25019)

#### Same Game, Different Story: A Minimal Conservative Strategic Robustness Benchmark for Large Language Model Agents（Same Game, Different Story） (2026-07)
- **简介**：Mousavi Davoudi、Amiri-Margavi、Gholami Davodi、Hasani Balyani、Gharagozlou 提出的博弈论视角基准，将「策略鲁棒性」定义为：在**保持收益不变、仅改变叙事框架**时，模型诱导的动作分布是否保持不变（区别于「策略能力」）。作者对 GPT-3.5、GPT-4、LLaMa-2 在四类社会困境博弈上已发表的合计合作率做二次分析（24 个 model-game-context 单元、7,200 次决策）。在其保守变换下，合并策略鲁棒性为 0.783，且「朋友分享」框架相比「商业」框架使合作率提升 0.307，表明社会关系框架即便在动作集与收益固定时也能显著改变 LLM 行为，主张用「收益等价的一族 prompt」而非单一呈现来评估博弈鲁棒性。
- **arXiv**：[2607.19670](https://arxiv.org/abs/2607.19670)

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

#### Learning Multi-Agent Orchestration: Higher-Order Interactions in Policy Improvement (IAPO) (2026-11)
- **简介**：作者 Kaixuan Liu、Guojun Xiong。基于 LLM 的多 agent 系统中，学习式 orchestrator 负责选择 agent 并控制其获得的信息；现有方法以各决策自身的 advantage 给其记功，隐含假设 episode 中其他决策固定。但在 orchestration 中每个决策的贡献依赖其之前的决策，联合更新的收益并非各决策效应之和。作者提出 **IAPO**（interaction-aware policy optimization）：精确地把两个策略的性能差分解为各个决策的单独效应及其交互项（作者证明交互项可占可达增益的一半），并从采样轨迹中同时优化二者，同时保证每次更新都是改进。在长输入版 GSM8K、MATH、DROP、MBPP、HumanEval 上，对比同一 7B executor 下十五个已发表多 agent 与 workflow 搜索系统，IAPO 部署的计划以更少 token 达到最强系统的准确率，去掉交互项在代码任务上损失 7 到 17 个点的准确率。
- **arXiv**：[2611.07495](https://arxiv.org/abs/2611.07495)

#### EvoRouter: Cost-Efficient Multi-Agent Routing via Insight-Guided On-Policy Distillation (2026-10)
- **简介**：作者 Yitao Xiao、Xu Zhang、Ye Cheng 等。大规模用多 agent 系统服务网页查询需联合配置 LLM、协作模式、agent 数、交互轮数与 skill，配置空间组合爆炸，稀疏执行奖励难以为相互依赖的规划决策提供细粒度监督。**EvoRouter** 是由 insight-guided on-policy distillation 驱动的自进化多 agent 路由框架：专门的 expert 周期性地把跨任务配置轨迹中的性能-成本反馈综合为可复用的条件路由 insight（能力需求、资源扩展回报、配置间交互）；具特权的 teacher 结合这些 insight 与当前执行反馈，通过 on-policy distillation 为 router 自身的规划轨迹提供稠密 token 级监督；cost-aware RL 联合优化任务性能与计算成本，形成执行驱动的自我改进闭环。在 19 个域内数据集和 2 个域外 benchmark 上，域内平均准确率 68.69%，比最强路由基线高 7.23 个百分点，同时推理计算成本降低 50.95%。
- **arXiv**：[2610.35690](https://arxiv.org/abs/2610.35690)

#### DynaFlow: Learning Free-Form Multi-Agent Orchestration (2026-10)
- **简介**：作者 Ruhao Liu、Qi Li、Shizun Wang、Xinchao Wang。多 agent 系统常依赖预定义协作结构，难以适应多样任务。**DynaFlow** 把自由形式、可执行的工作流合成纳入主 agent 的动作空间：主 agent 自行决定何时需要协作，并生成任务特定的程序，通过串行或并行执行、结构化信息交换与验证来协调子 agent；执行结果与轨迹指导后续动作，使协作可随推理进展调整。仅训练主 agent——先在蒸馏轨迹上做 SFT，再对剔除了无依据前提、偏置验证和信息流缺陷的样本进行精炼，所有子 agent 保持固定，从而把编排学习与执行者能力提升分开。在数学推理、科学问答和多跳检索的六个 benchmark 上，Forced 模式总体准确率 88.08%，在相同 Qwen3.5-27B 骨干下比最强对比基线高 10.90 个百分点。
- **arXiv**：[2610.34467](https://arxiv.org/abs/2610.34467)

#### Learning When to Collaborate: Selective Multi-Agent Medical Reasoning via Uncertainty-Aware Routing (SMART-Med) (2026-10)
- **简介**：作者 Jiacheng Hou、Xueliang Cui、Dan Lu、Ruxin Wang，NeurIPS 2026 接收。现有多模态医疗推理多 agent 方法假设始终开启的稠密协作并依赖大型云端模型，带来隐私与效率问题。**SMART-Med** 是完全本地的多 agent 方法，把协作视为可学习的决策而非默认行为：不确定性驱动的路由机制自适应决定何时直接回答、何时召集少量专家 agent，将稠密多 agent 推理转为选择性、按问题自适应的协调。为使路由可靠，作者构建带高质量推理轨迹与难度层级的 MedVR-3K 数据集，并用 GRPO 微调本地 agent，使用两个互补奖励：certainty-aware accuracy reward 与评价正确性和推理质量的视觉-语言奖励模型。在四个医学 VQA benchmark 上性能与现有 MAS 方法相当，推理时间降低超过 3.8 倍、token 用量降低 72%；MedVR-3K 微调后在 Slake、PathVQA、VQA-RAD、PMC-VQA 上平均提升 6.7%。
- **arXiv**：[2610.33544](https://arxiv.org/abs/2610.33544)

#### Learning to Steer LLM Agents and Multi-Agent Systems (2026-10)
- **简介**：作者 Leyang Shen、Yaxin Gong、Yang Zhang、Chenxiao Fan、Jiakai Wang、Dong Wang。LLM agent 与多 agent 系统在长 horizon 工具使用与协作中易偏离总体目标（goal drift），浪费资源并导致不良结果。论文提出一个 steering agent，在线监督执行并在发生漂移时将系统拉回正轨：它观察执行上下文，用低成本快路径决定何时干预，用高成本慢路径决定如何干预；两条路径通过结合 pairwise ranking 目标与 GRPO 的强化学习联合训练。在信息检索类任务上，steering agent 对单 agent 与多 agent 系统都提供有效监督，提升性能与效率；学到的 steering 能力可泛化到域外任务、迁移到未见的 executor，并支持无需重训的干预频率调节（abstract 未给出具体数值）。
- **arXiv**：[2610.22316](https://arxiv.org/abs/2610.22316)

#### R^2Orch: Recursive Self-Improvement for Agent Orchestration via Self-Play within a Skill-Internalization Loop (2026-10)
- **简介**：作者 Mingda Zhang、Yun Huang、Qiang Huang、Yanjin Li、Rui Qian、Qichao Zhao。现有 orchestration 学习面临人工设计课程、任务 rollout 全部同结果时的 advantage collapse，以及用单个噪声分数评价每个 skill 的不可靠信号等问题。**R^2Orch** 提出 Recursive Self-Play Orchestration：内循环用 self-play 在自选任务上改进 orchestrator，外循环改进内循环的改进方式。训练目标 Recursive Learnability 奖励 Proposer 选择那些在 skill 被撤出时 rollout 仍不断得到不同结果的任务；在此基础上的 Skill Withdrawal 在受控评估显示 orchestrator 无该 skill 同样表现时由外循环撤出该 skill。两个循环构成一个周期：失败被蒸馏为 skill，skill 内化到参数，撤出使 Proposer 找到新的失败，让 orchestrator 走向零 skill 部署。在十二个数据集（问答、数学、代码、交互任务）上一致优于基线（abstract 未给出具体数值）。
- **arXiv**：[2610.22740](https://arxiv.org/abs/2610.22740)

#### Decoupled Multi-Agent Orchestration (DeOrch) (2026-10)
- **简介**：作者 Xinle Wu、Yao Lu。学习式编排可自动构建有效的 LLM 多 agent 系统，但现有方法把规划耦合到固定 worker 池，并用同一终端结果训练任务分解与协作，限制迁移并模糊信用分配。**DeOrch** 将与 worker 无关的规划同具体 worker 选择分离：两阶段 planner 先在没有 worker 信息的情况下分解任务，再利用来自池的紧凑、不含 worker 身份的 matchability 反馈选择协作操作，从而对分解和协作决策做条件信用分配；轻量 matcher 根据 worker 在固定探测集上的行为估计适配度，并用 contextual bandit 在线适配，使新 worker 无需重训 planner 或 matcher 即可加入。在多种域内外任务上优于已有自动 MAS 编排方法且 worker 调用更少，迁移到完全未见的 worker 池无需重训仍有效。
- **arXiv**：[2610.07556](https://arxiv.org/abs/2610.07556)

#### Multi-Agent System Search via Active Substructure-aware Policy Optimization (ASPO) (2026-09)
- **简介**：作者 Beicheng Xu、Bowen Fan、Weitong Qian、Lingching Tung、Bin Cui 研究由执行 reward 学习构造 query 特定多智能体系统（MAS）的策略，而不是手工设计角色、prompt 与通信结构。现有方法重复遍历固定训练 query 并在 workflow 级分配 reward，忽略 query 学习潜力的演化，也无法分辨哪些子结构提升了解质量。提出 RL 框架 **ASPO**（Active Substructure-aware Policy Optimization）：Adaptive Query-Selection Mechanism（AQSM）把训练聚焦在策略能力边界的 query（能解但尚不可靠）；互补的发现机制为难 query 扩大架构探索；并引入 substructure 级 reward，衡量每个动作后代子图内的输出质量增益，引导 PPO 强化有用的架构改进、抑制冗余或有害的计算。在覆盖数学推理、通用 QA、代码生成的六个基准上，ASPO 对十二个基线在每个基准上均排名第一。
- **arXiv**：[2609.32430](https://arxiv.org/abs/2609.32430)

#### MAGIC: Mixed-Granularity Agent Graphs via Incremental Construction with Dense-Reward Reinforcement Learning (2026-09)
- **简介**：作者 Kairui Yang、Ziheng Yi、Xunkai Li 等。协作拓扑决定 LLM 多智能体系统的性能与执行成本，现有拓扑生成器全程使用单个 agent 或预定义分组，忽视不同子任务的协作需求。**MAGIC** 为每个功能角色局部选择粒度，在一个组织内结合细粒度控制与可复用协作模式；用稠密奖励 RL 学习：序贯选择功能角色、将其实例化为单 agent 或可复用组、并连接到已有单元，直接用当前策略采样轨迹的回报优化构造策略，并以基于势函数的 reward shaping 从 probe 效用和结构信号提供中间反馈，同时保持累积任务奖励不变。在八个基准上优于 SOTA 基线，且推理效率高（abstract 未给出具体数值）。
- **arXiv**：[2609.26667](https://arxiv.org/abs/2609.26667)

#### CERA-MoA: Co-Evolving Routing Mechanisms with Continually Learning LLM Agents (2026-09)
- **简介**：作者 Jiaxuan Jiang、Liyuan He、Zhixuan Fang。现有 Mixture-of-Agents（MoA）把 query 路由与 agent 微调割裂处理，路由策略无法适应后训练中不断演化的 agent 能力，agent 也难以获得数据驱动的协同专精。**CERA-MoA** 是迭代式 RL 框架，让动态 router 与各独立 agent 策略共同演化：设计基于中间层隐状态的预测式 familiarity estimator 评估 agent 间的语义胜任度，避免完整 rollout 开销；依据 familiarity 分数用累计阈值自适应路由激活最小的定制 agent 子集，平衡性能与效率；并按 agent 演化中的胜任度主动分配针对性训练样本以促进能力分化。在多个领域上优于静态 agent 路由与固定 workflow 微调的 SOTA baseline（abstract 未给出具体数值）。
- **arXiv**：[2609.18779](https://arxiv.org/abs/2609.18779)

#### Learning How Much to Collaborate: Difficulty-Aware Topology Selection for Multi-Agent Code Generation (DATS) (2026-09)
- **简介**：作者 Yunsong Hong（Neurocomputing 接收）。多 agent 代码生成系统常对所有问题使用同一种通信拓扑。在五种拓扑、APPS/HumanEval+/LiveCodeBench 的 614 道题上，层次化协作相对单 agent 的优势从最易三分之一题的 pass@1 +2.4 点增至最难三分之一的 +21.1 点，代价是约十倍 token。作者提出 **DATS**（Difficulty-Aware Topology Selector），预测每种拓扑的成功概率并最大化「预测成功率减成本」，其图预测器把拓扑视作连通度序上的节点；成本惩罚可无需重训重新校准。在始终层次化成本的 40% 下，DATS 达 77.7% pass@1，而 always-hierarchical 为 73.6%、最强的学习型对手为 74.3%；四个 backbone 上增益 4.07–4.23 点；在 400 道数学题上复现了随难度增大的协作增益。
- **arXiv**：[2609.13890](https://arxiv.org/abs/2609.13890)

#### Codebook Agent: Amortized Topology Design for LLM Multi-Agent Systems (Codebook Agent) (2026-09)
- **简介**：来自 UCLA（Jinxi Yu、Yubei Li、Eric Hanchen Jiang、Kai-Wei Chang、Ying Nian Wu 等 9 人）。质疑把 query 自适应通信 topology 设计当作条件图生成（在 N×N 邻接空间中用 VAE/自回归/扩散解码器搜索、再用图网络 proxy 排序）的既有形式化，给出三条实证依据：通过 reward filter 的 topology 即使 codebook 容量从 8 增到 64 也仅坍缩为约 6 种不同的图；edge 数与实测 token 消耗呈负相关（Pearson r≈−0.4），稀疏化反而更贵；当 agent 共享 profile（公开基准的默认配置）时基于 agent-profile 节点的 message-passing 打分器是 adjacency-invariant，根本无法排序。据此提出 **Codebook Agent**：用 vector-quantized autoencoder 把成功 topology 压缩进一个与 query 无关的 16 条目 codebook，用 reward-weighted MLP 将 query embedding 映射到 code 上的分布，再用读取展平邻接矩阵、在实测 utility 与逐任务归一化 token cost 上回归的 MLP proxy 单次批量前向重排 top 候选。测试时无需迭代搜索与 message passing，在全部 6 个基准上均最准（平均 84.6，最强先前设计器为 83.0），2.4 ms 产出一个 topology，并少用 21.9–33.2% 的 LLM token。
- **arXiv**：[2609.02264](https://arxiv.org/abs/2609.02264)

#### Reward-Guided Autoregressive Graph Generation for Efficient Multi-Agent Communication Topology Design (RGA-Designer) (2026-08)
- **简介**：来自卢森堡大学等（Poomphob Suwannapichat、Boonyarit Changaival、Caesar Wu、Pascal Bouvry）。指出 LLM 多 agent 系统依靠协调多个 agent 在复杂推理上取得强性能，代价是巨大的 token 消耗；近期把自动 topology 设计重构为自回归图生成的 ARG-Designer，其训练目标对「生成稀疏而高效的 topology」没有任何显式激励。提出 **RGA-Designer**：受 RLHF 启发，先训练一个同时刻画任务正确性与结构紧凑性的 reward model，再以该 reward model 作为反馈信号微调预训练好的图生成器，把通信 topology 的稀疏性直接写进优化目标。在保持与 ARG-Designer 相当的任务准确率的同时，平均减少 20.5% 的 token 消耗。
- **arXiv**：[2608.20099](https://arxiv.org/abs/2608.20099)

#### SyncPlan: Long-Horizon LLM Coordination with Explicit Synchronization and Adaptive Correction (SyncPlan) (2026-08)
- **简介**：作者 Shen You、Xiaoming Zhu、Weining Weng 等 20 人。指出 LLM 多智能体协调存在效率与适应性的权衡：反复调用 LLM 或多轮通信延迟大、易受异步进度与环境变化影响，而 one-shot 规划开销低但产出开环计划、在动作依赖他人与环境时迅速失效。提出 **SyncPlan** 这一 plan-execute-correct 框架：中心化 LLM coordinator 单次调用生成每个 agent 的 action chain；执行时用显式 wait primitive 与死锁检测强制 agent 间及 agent-环境依赖，轻量 Plan Staleness Detector 持续评估剩余计划、在环境变化使假设失效时触发重规划；并用 SFT 与 planning-oriented RL（稠密任务进度奖励 + 结果级执行反馈）优化 coordinator。Overcooked 与《王者荣耀》环境上取得 SOTA 成功率，wall-clock 运行时间不到现有 LLM coordinator 的 0.05%。
- **arXiv**：[2608.01652](https://arxiv.org/abs/2608.01652)

#### MANTA: Multi-Agent Network Topology Adaptation for Self-Evolving Multi-Agent Systems (2026-07)
- **简介**：Mao-xun Huang、Claire Cardie、Hen-Hsen Huang 等提出 **MANTA**，让多 agent LLM 系统的通信拓扑在**推理时自演化**：执行前由结构经验初始化任务条件拓扑，部署中监控协作轨迹并在当前组织不足时施加有界结构更新（可改 agent 角色、通信链路、执行顺序、信息可见性、验证路径），同时保持任务接口与 agent 预算不变。在信息检索、工具使用、规划、工作流执行、数学推理五个基准上取得平均 74.0，超最强基线 5.8 个百分点，PlanCraft 上最佳，表明推理时自改进可延伸到协作架构本身。
- **arXiv**：[2607.28527](https://arxiv.org/abs/2607.28527)

#### Toward an Organizational Science of Multi-Agent LLM Systems: Decoupling Who, How, and Which Algorithm (2026-07)
- **简介**：Huan Chen 等提出 **IMACS**，把多 agent LLM 系统中常被混为一谈的三个正交关切拆开——"谁在团队里（organization）/ 成员如何对齐（coordination）/ 用哪种算法融合工作（collaboration protocol）"，并使经典组织理论（Belbin 角色、Mintzberg 协调、RACI 问责）成为可执行配置；六种已发表协作算法统一到同一接口下。进一步提出 **Adaptive Org Routing**——一个 contextual-bandit 元协议，按质量-成本权衡为每个任务在线选择协议，超越所有固定协议。消融揭示：问责放置的最优位置在不同模型族间会翻转，故组织设计不能硬编码，需按模型绑定重新验证或学习。
- **arXiv**：[2607.25446](https://arxiv.org/abs/2607.25446)

#### TriAgent: Divergence-Aware Multi-Agent Committees for Cost-Efficient Financial Sentiment Analysis（TriAgent） (2026-07)
- **简介**：Isabel Xu、Cynthia Xu、Rachel Ren、Cong Guo、Jiacheng Ding（The Overlake School / University of Memphis 等）提出的按「上下文粒度」分层的多 agent 委员会：词级词典（VADER）、句级领域 transformer（FinBERT）、跨句推理器（Qwen2.5 系列，含 Mistral-7B、Phi-3.5-mini 跨家族核验）。用三方 **Semantic Divergence Index (SDI)** 度量三粒度间的两两分歧并据此路由查询。核心发现是「critic plateau」：当把 LLM 当作对小 agent 输出的 critic 时 F1 在 1.5B–7B Qwen 上稳定在约 0.87，而同尺寸 3-persona 投票掉到 0.66，说明增益来自粒度分层的多样性而非单纯投票。SDI 还兼作幻觉检测（AUC=0.90），并在 20 标的回测上取得最佳风险调整收益（Sharpe=3.50）；10M 用户规模下相较 GPT-4o-mini 基线年省 $9.3M。
- **arXiv**：[2607.19794](https://arxiv.org/abs/2607.19794)

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

- **Multi-Agent Debate Strategies: Survey, Taxonomy, and Challenges** (2026-07)：Quim Motger 等（UPC，投稿 ACM Computing Surveys）对多 agent 辩论（MAD）做系统性文献综述，梳理 141 篇主研究，提出覆盖"辩论参与者 / 交互机制 / 一致性协议"的三维分类法并配形式化记号；核心发现是该领域已隐性收敛到一种狭窄设计模式（静态全连接拓扑、逐字交换、短期记忆、投票裁决），多为惯例而非系统比较所致。[arXiv:2607.26212](https://arxiv.org/abs/2607.26212)
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

#### MAC-Bench: Benchmarking Multi-Agent Collaboration via Interconnected Tasks (2026-11)
- **简介**：作者 Zerui Chen、Qinggang Zhang、Qichuan Liu 等。评估 LLM 多 agent 系统不仅要看任务求解，还要看处理相互依赖工作的能力；现有 benchmark 常强调任务难度而协作要求不足，或只评估有限的预定义协作模式。**MAC-Bench** 不预设 agent 角色或通信协议，而是把协作要求直接嵌入任务依赖与共享资源约束：从选定 benchmark 任务中在两个层次挖掘可复用协作结构——局部协作模式刻画子任务间反复出现的交互，全局任务结构刻画这些交互如何构成完整工作流；再经由粗到细的组合重组为结合不同协作需求的任务图，并在可执行环境中实例化，对部分进展与完整任务解决做自动验证（abstract 摘录未给出评测数值）。
- **arXiv**：[2611.01728](https://arxiv.org/abs/2611.01728)

#### TeamGym: A Gym for Team Decision-Making by AI Agents with Distributed Information (2026-10)
- **简介**：作者 Chanwoo Park、Jaywoong Jeong、Zachary Wojtowicz、Yubin Kim、Taekyun Lee、Kaiqing Zhang。现实决策常需团队整合分散在各参与者手中的信息，而 AI agent 仍多在孤立环境中训练与评测。**TeamGym** 是基于法律、金融、医疗和软件工程真实语料构造的团队决策环境，用于评测与训练语言模型 agent；它提出识别 pivotal information（各元素必要、整体充分的最小事实集）的方法，从而控制其分布并追踪 agent 识别与整合的情况。在 12 个来源语料上评测 8 个模型：把关键信息分散到不同参与者，比交给单个模型平均使成功率降低 22.9 个百分点（communication gap）；当参与者须自行从原始材料中识别提取关键信息时，成功率再降低 19.9 点（retrieval gap）。用 TeamGym 以团队决策损失做 RL 训练，两项实验均提升最终决策质量，并涌现出参与、发言者选择与沟通上的协调行为，尽管奖励只评估最终决策。
- **arXiv**：[2610.37571](https://arxiv.org/abs/2610.37571)

#### Understanding Communication Failures in Multi-Agent LLM Systems (2026-10)
- **简介**：作者 Priya Pitre、Jun Hou、Yi Fang、Naren Ramakrishnan、Xuan Wang。多 agent LLM 系统通过辩论、证据共享与相互纠正来改善推理，但交互也会引入仅凭最终准确率无法解释的失败：agent 过早收敛、忽视异议、压制正确的少数答案，或跨轮保留错误的 framing。论文基于涵盖医疗 QA、代码、金融、多跳推理、伦理与具身文本任务的 3,211 条失败多轮轨迹，建立了以语料为依据的沟通失败分类体系，并据此分析失败在数据集与系统设计上的分布、轮级错误动态、恢复难度、多数派转移、少数信号压制以及自报置信度。发现失败主要由恢复/纠错和共识/决策问题主导，而非孤立的首轮错误，且许多在 agent 首次看到同伴回复后才出现；据此建议加强异议处理、证据验证、少数信号保留以及定稿前的恢复检查。
- **arXiv**：[2610.32060](https://arxiv.org/abs/2610.32060)

#### MASIF: Implicit Failures for Process Observability in Multi-Agent Systems (2026-10)
- **简介**：作者 Yanwei Huang、Arpit Narechania、Atanu Sinha。多 agent 系统（MAS）可能在局部执行错误被纠正、绕过或带到后面的情况下仍完成任务，作者称之为 implicit failures (IFs)，并将其作为超越最终结果的过程可观测信号。在九种 MAS benchmark 配置的 1,828 条轨迹中，IF 出现在 54.3% 的成功轨迹和 75.2% 的失败轨迹中；作者给出错误类型与吸收机制的分类，并用人工与跨模型标注审计支撑，用反事实回放探究修复 IF 如何改变最终结果。为检验实用价值，作者将基于 IF 的 best-of-N 轨迹选择与已有 PRM 对比，并提出新 RL 算法 **IF-GRPO**，在 GRPO 中用 IF 代价细化成功轨迹之间的偏好；实验显示在准确率与过程指标上有一定收益，同时暴露任务表现、IF 数量与成本、按 token 归一化的 IF 密度之间的权衡。
- **arXiv**：[2610.25852](https://arxiv.org/abs/2610.25852)

#### Does Multi-Agent Collaboration Pay When Agents Serve Users? Testing Agent Scaling in Dual-Control Settings (2026-10)
- **简介**：作者 Ethan Yang、Xiangcheng Shi。多 agent 系统（MAS）的部署前提是协作优于单 agent（SAS），但何时划算并不清楚；此前最系统的研究给出「能力饱和」规则：SAS 弱时协调有益、强时有害，交叉点约为 45% 单 agent 准确率，但该规则来自没有用户参与的自主 benchmark。论文在 $τ^2$-bench（dual-control）的航空、零售、电信领域，对 3 个模型族的 6 个模型，用同一模型构建的集中式 MAS 与原生 SAS 比较，约 7,950 段模拟对话并逐 episode 记录 completion-token 账本（研究预注册）。固定模拟用户下没有任何格子出现 MAS 收益：13 个有效格中 MAS 在 12 个更差、1 个持平，拟合增益在观察到的能力范围内为负（加权均值 -28 点，95% CI [-44, -12]）；规则预测有益的两个格子里 MAS 反而输 13 与 22 点；MAS 也没有可检测地减少错误数据库写入，只是把其他失败转为预算耗尽与不作为。结论是服务用户的场景下默认应选 SAS。
- **arXiv**：[2610.27655](https://arxiv.org/abs/2610.27655)

#### When Agents Collude: Weak-to-Strong On-Policy Distillation for Mitigating Strategic Deception in Multi-Agent Systems (PACT) (2026-10)
- **简介**：作者 Eric Onyame、Runtao Zhou、Kowshik Thopalli、Bhavya Kailkhura、Chirag Agarwal。论文在模拟内幕交易环境中研究多 agent 协作带来的合谋与策略性欺骗风险，覆盖三个模型家族的七个语言模型，比较单 agent、顺序与去中心化团队。相比单 agent，多 agent 交互使未对齐行为率从 35.4% 升至 68.2%，持续欺骗率从 31.0% 升至 62.2%，生存压力下团队在 87% 的运行中能经追问持续欺骗。为缓解，作者提出 **PACT**，一种 weak-to-strong 的 on-policy distillation 框架：使用通用诚实提示，让更强的 student 在自己采样的续写上匹配 teacher 的 next-token 分布。（abstract 在此处被截断，后续结果请以原文为准）
- **arXiv**：[2610.33011](https://arxiv.org/abs/2610.33011)

#### Homogenization in Multi-Agent Systems (2026-10)
- **简介**：作者 Prakhar Ganesh、Kyra Wilson、Luca Zappella、Barry-John Theobald、Nicholas Apostoloff、Lucas Monteiro Paes。多 agent 系统的交互也可能导致同质化，即 agent 收敛到相似行为，降低多样性并强化共同失败。论文用三个指标刻画同质化：向多数的从众、向极端的极化、在后续交互中对变化日益增长的惰性，并在代码生成、招聘、科学同行评审三类任务中评估：同质化会隐藏并放大相关错误从而造成系统性漏洞；在招聘中使有偏 agent 的影响在其移除后仍长期存在；在同行评审中造成跨研究领域的评价标准不均。结论是同质化是 MAS 的一种失败模式，评估必须超越总体性能分析交互动态；简单的增加多样性方法（采样随机性、混合模型 MAS）无法降低同质化风险。
- **arXiv**：[2610.09824](https://arxiv.org/abs/2610.09824)

#### How does Adversarial Influence Scale in Multi-Agent Systems? (2026-09)
- **简介**：作者 Addison J. Wu、Jasin Cekinmez、Michel Liao、Karthik Narasimhan、Thomas L. Griffiths。多 agent 协商可提升性能，但若部分 agent 不怀好意会怎样？论文研究群体规模增大与欺骗者比例上升时对欺骗的易感性：关键不是 agent 数量，而是欺骗者比例；初始正确的 agent 转向错误最终答案的叛变率随该比例线性上升。人类在类似从众研究中通常只有误导者占多数时才被可靠影响，而 LLM agent 即便欺骗者是少数也经常叛变；易感性还取决于交互的是哪些模型，尤其是诚实 agent 一侧；出人意料的是，允许欺骗者私下协调反而可能降低其效果。结论：增加更多 agent 并非充分防御，因为对手可随群体扩张。
- **arXiv**：[2609.30028](https://arxiv.org/abs/2609.30028)

#### Emergent Collusion in Long-Horizon LLM Agent Interaction (2026-09)
- **简介**：作者 Xinrui Shi、Yanzhe Zhang、Diyi Yang。研究长程多 agent 环境中合谋的涌现：两个 agent 反复完成各自任务、共享任务日志、互相验证对方工作并获得奖励，并设置现实约束，使得只有违反验证协议才能获得更高奖励。结果发现 agent 在重复交互中越来越偏离协议：10 个模型中 94% 的轨迹出现合谋，同一家族中能力更强的模型更早达到。受控的对等干预表明合谋受对方行为塑造；消融显示验证反馈、交互历史和奖励结构也有影响，尤其是限制 agent 可用交互历史的量与范围可减少合谋。结论是长程交互会以带来安全风险的方式重塑 agent 的协调。
- **arXiv**：[2609.24967](https://arxiv.org/abs/2609.24967)

#### Rethinking Multi-Agent Collaboration: When More Is Less (2026-09)
- **简介**：作者 Yishuo Yuan、Yibo Wu、Yihan Zhang 等。随着单 agent 能力与 harness 的进步，多 agent 协作面临收益递减与上下文开销增长。作者通过系统分析划定了多 agent 相对单 agent 的能力边界：其系统性收益特别出现在依赖稀疏的长程任务，而紧耦合、顺序性的工作流仍以单 agent harness 更优。据此提出轻量协作机制 **SAIGE**（Semantic-Aware Incremental Graph Evolution），把协作建模为动态演化的图，节点为按需生成的 agent 实例，边为通过基于内容的信息检索建立的语义依赖。在长程复杂任务 benchmark 上取得上下文效率与任务表现的良好折中，并发现扩大 agent 池或加深递归并不能持续改善结果——多 agent 的优势受任务结构限制，并非普遍成立。
- **arXiv**：[2609.19759](https://arxiv.org/abs/2609.19759)

#### Emergence World: Adversarial Stress-Testing of Long-Horizon Multi-Agent Systems (2026-09)
- **简介**：作者 Deepak Akkil、Tamer Abuelsaad、Karthik Vikram 等。agent 的失败可能通过记忆、工具、其他 agent 与环境状态在交互结束后长时间传播。**Emergence World** 是持续运行的多 agent 环境，用于对长程自主系统做对抗压力测试：从相同初始条件运行八个并行世界、每个十个 agent（七个由不同前沿模型驱动的同质世界和一个混合模型世界），历时 16 天产生超过 85 万次 LLM 调用、近 500 亿 token；在运行状态累积后通过普通交互面施加间接 prompt injection、错误信息、私有 agent 记忆泄露三类压力事件。无一世界在三类事件上都完全具备韧性；检测不等于遏制，系统可能识别威胁却仍与对抗内容交互、写入持久记忆并在最长 46 小时后据此行动；同一模型-persona 组合在混合与同质群体中行为差异显著，说明模型级对齐不具可组合性。
- **arXiv**：[2609.17320](https://arxiv.org/abs/2609.17320)

#### Rethinking the Evaluation of Efficiency Methods for Multi-Agent Systems (2026-09)
- **简介**：Jiamu Zhang、Lingxi Zhang、Pengjun Lu、Qiyue Zhang 等 9 人，EMNLP 2026。针对 LLM 多 agent 系统（MAS）的效率方法——剪 agent、删通信边、搜索紧凑结构——论文主张现有评测可能高估了它们真实的效率改进能力：所报增益常在方法专属的 prompt 与起始 topology 下测得，难以归因到所提的结构性改动；且许多「成功」出现在 non-MAS-demanding 设定中，那里单 agent 或随机剪枝系统本就能保持强性能。为此构建一个受控且 MAS-demanding 的诊断 benchmark，在共享 backbone 模型、agent registry 与 runtime 下，沿 topology、规模、深度、工具使用四个维度做受控变化评测代表性效率方法。分析表明许多已报增益是 setup-dependent 的，可能源于 structural collapse、工具路径被禁用、或起始系统本身随机剪枝即可保持准确率，而非 MAS 效率的稳健改进（abstract 未给出具体数值）。属 §4.6 的诊断基准/实证再评估工作，不提出训练方法。
- **arXiv**：[2609.05933](https://arxiv.org/abs/2609.05933)

#### ERPBench: Evaluating LLM Agents for Enterprise Decision-Making Across Competitive Market Ecologies (2026-09)
- **简介**：来自上海交通大学 GAIR 实验室（Xinran Zhang、Pengrui Lu、Lyumanshan Ye、Pengfei Liu）。针对 LLM agent 日益被用于企业工作流、但现有评测很少检验商业决策结论能否跨不同竞争性市场生态迁移的问题，提出 **ERPBench**：一个 execution-instrumented 基准，在六轮 ERP 仿真中耦合定价、生产、采购、库存、财务与共享市场竞争；同一组 100 个固定问题在两种匹配的市场生态下评测——Solo（每个 agent 对抗固定规则对手）与 Arena（六个待评 LLM agent 在共享市场中竞争），覆盖六个模型族、共 1,200 条模型级 trajectory 与 7,200 个决策轮。结果显示领先模型随生态而变：Solo 下 DeepSeek 领先（平均估值 252.29M，平均排名 1.67），Arena 下 Gemini 领先（263.95M，1.76）；两种生态在 100 个问题中仅 21 个给出相同的任务级胜者，Gemini 的垫底率从 22% 降至 0%。属 §4.6 的企业决策基准/评测工作，不涉及训练。
- **arXiv**：[2609.04667](https://arxiv.org/abs/2609.04667)

#### SwarmBench: Can Large Language Models Act as Agent Swarm Orchestrators? (SwarmBench) (2026-08)
- **简介**：来自中科院自动化所（Jinshan Gao、Zhuoran Jin、Tianyi Men、Kang Liu、Jun Zhao，EMNLP 2026 Findings）。针对 LLM 多 agent 系统正从固定交互 topology 走向动态编排的 Agent Swarm、而现有基准仍以单 agent 或通用 agent 任务为主、难以系统评测编排能力的问题，提出 **SwarmBench**，从 accuracy、efficiency、cost 与 process quality 多个视角评测模型的 orchestration 能力。实验显示当前模型的编排能力差异显著，不仅体现在最终准确率、效率与成本上，也体现在编排过程本身的质量上；基于这些发现进一步提出 **SwarmExp**，一种基于 experience extraction 与 experience replay 的简单方法，可持续提升 LLM 的编排表现（abstract 未给出具体数值）。属 §4.6 的基准/评测工作，SwarmExp 为经验复用而非 policy 参数更新。
- **arXiv**：[2608.30661](https://arxiv.org/abs/2608.30661)

#### The Collaboration Tax: How Much LLM Multi-Agent Systems Pay to Coordinate (2026-08)
- **简介**：来自 Weixiang Sun、Zehong Wang、Hong Huang、Yanfang Ye 等 5 人（EMNLP 2026 Main）。追问一个尚不清楚的问题：当两个 LLM 必须协作而非独立作答时，究竟损失多少性能。把 **collaboration tax** 形式化为带私有信息的两人合作博弈的 team-decentralisation loss，并给出两条命题刻画其符号及其与 max-superadditivity 违反的等价性；在 32 个单 agent 可解、按 grounding friction 来源分组的任务上操作化该定义，测量 7 家提供方的 11 个模型。发现该税沿两条无例外的轴呈结构化规律：跨所有模型一致的类别序，以及随能力单调下降；其近因不是推理缺陷，而是一条四阶段对话级联——agent 给出无依据断言、不去询问伙伴、跳过整合双方观点、未重新推导就接受答案。该税可由对话特征机械地预测，且部分可治：针对四个阶段的 prompt 干预能弥合相当一部分差距，主导瓶颈随类别而异；异质配对中该税被拉向更强的一方而非加性中点，实证印证了框架预测的 max-superadditivity 违反。属 §4.6 的实证代价分析，不含训练方法（abstract 未给出税额的具体数值）。
- **arXiv**：[2608.22152](https://arxiv.org/abs/2608.22152)

#### The Interaction Tax: When Communication Erases Diversity in Multi-Agent Teams (2026-08)
- **简介**：来自 Summer Eunhyung Ann、Haokun Liu、Chenhao Tan（ICML 2026, PMLR 306）。针对「多 agent 交互到底有益还是有害」的文献矛盾（debate、critique loop、mixture-of-agents 报告增益，另一批工作则发现等预算下交互只增成本、或独立采样已能获得多 agent 收益），主张矛盾部分源于缺失一个区分：并非所有多 agent 通信都等价。发现不同模型家族会找到结构上不同的解，但当 agent 互相读取对方的**完整输出**时，其提案会在一轮内收敛，抹掉了使用多模型的初衷——多样性，作者称之为 **interaction tax**。在 11 个 verifier 打分的优化任务、等预算设置下测试，表明 full-solution interaction 是较弱的默认选择，而独立生成提案可避免这种坍缩；full-solution interaction 主要让 agent 黏着于最先看到的解而不去尝试不同路径，critique 只在被违反的规则易于 LLM 发现并修复时才有帮助。结论是多 agent 性能取决于交换什么信息、何时交换，而非 agent 数量。属 §4.6 的实证代价 / 现象分析，不含训练方法（abstract 未给出具体数值）。
- **arXiv**：[2608.23541](https://arxiv.org/abs/2608.23541)

#### FM-Bench: A Benchmark for Long-Horizon Management with Competing Agents (2026-08)
- **简介**：来自 Analogy AI（Tianyou Wang、Chongyang Gao、Kezhen Chen 等 9 人）。指出语言模型 agent 已能可靠完成有界任务，但在「动作具有累积后果、环境会对其选择作出反应」的长时程决策上几乎没有被度量。提出 **FM-Bench**（Football Management Benchmark）：LLM agent 通过 26 个工具、约 340–400 个决策点经营一家足球俱乐部 20 个游戏年，在与所有对手同等预算下组建阵容、交易球员、谈合同、投资设施与青训、排兵布阵，并对一个可以解雇它的董事会负责；由确定性引擎把每年结果累积为唯一终局分数，全程不用 LLM judge 或人工评分。solo 赛道让 15 个前沿模型对抗冻结的脚本化世界，Arena 赛道把同样 15 个模型加一个脚本锚点放入同一个共享的 20 年世界（作者称是该规模下首个 head-to-head 评测），并拆解出 6 项行为能力；三个 seed 下 15 个模型全部走完时程而盲目脚本基线在多数时程中被淘汰，claude-fable-5 在 solo 平均分与 Arena 上居首、但 Arena 冠军在十个模型间轮转，规模/价格/厂商均不预测排名，排序只在时程后期才稳定，人类首次上手的最好成绩仅位于模型榜末尾；区分模型的是管理行为而非算力——高分模型临近终局减少慢回报投资、让现金保持投入而非闲置、远早于截止期启动续约，而 token 花费不预测任何结果；没有模型能从数百次被拒报价中学到市场隐藏价格，自管理 memory 以两种相反方式失败（只增不减的档案，或每赛季被重写的计划）。属 §4.6 的基准/评测工作，不含训练成分。
- **arXiv**：[2608.18423](https://arxiv.org/abs/2608.18423)

#### Social Gym and SPaRTan: Benchmarking and Improving LLM Social Reasoning via Multi-Agent Game Tournaments (2026-08)
- **简介**：来自 CMU（Keyu He、Xuhui Zhou、Maarten Sap）。指出社交交互不像数学/逻辑那样有客观 ground truth，评测只能退回昂贵、主观且噪声大的 LLM judge，模型也因此得不到可靠的学习信号。为同时解决二者，先提出 **Social Gym**：含 21 个多智能体社交博弈（狼人杀、Resistance、Spyfall 等）的环境，由规则判定胜负使 agent 表现可验证、客观，并以 Elo tournament 产出跨博弈榜单——GPT-5-mini 居榜首，但没有模型能在所有博弈与所有角色上一致领先，暴露社交推理的局限。再提出 **SPaRTan**（Self-Play and Reflect-Transfer）这一 training-free 自我改进闭环：模型先对局，基于自身 trajectory 与结果反思产出可迁移 playbook，并在后续博弈中套用该 playbook；结果显示 playbook 能帮 GPT-5-mini 拉平其弱势角色的表现，但基本无法提升 Qwen3-32B。属 §4.6 的环境/基准工作，self-play 改进完全不做 weight update。
- **arXiv**：[2608.09128](https://arxiv.org/abs/2608.09128)

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
