# Contributing to Awesome-Agentic

感谢你愿意为本仓库贡献！

## 贡献规则

### 范围
- 只收 **2023.01 起** 的工作；2022 及之前作品不收（仅极少数作经典思想锚点提及）
- 范围限定 **大模型 / Agent / 多模态推理大模型**
- **不收** 纯 CV / T2I 生成评测 / 经典 RL（无 LLM）工作

### 论文条目格式

每篇论文必须采用以下模板：

```markdown
#### 论文标题 (YYYY-MM)
- **简介**：一句话概括论文核心贡献与机制（不超过 80 字，必要时可加机制 / 数字）。
- **arXiv**：[XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
```

- **YYYY-MM** 取 arXiv 首次提交的年月（不是 v2/v3 更新月）
- 简介要包含核心机制或关键数字，避免空泛描述
- 必须给可访问链接（arXiv / OpenReview / 官方博客均可）

### 排序规则
- 每个小节内按发表时间**倒序**排列（最新置顶）
- 同月论文可任意顺序，建议按重要性

### 章节归属
四大方向定义：
- **Reasoning RL**：单轮长 CoT 推理，可验证或半可验证任务
- **Agentic RL**：多轮、长 horizon、部分可观测；工具 / GUI / Web / Code / Memory / 安全
- **OPD**：Off-Policy RL + On-Policy Distillation + Policy Drift
- **Multi-Agent**：多 LLM 协作 / 辩论 / 自博弈 / coordinator

不确定时优先放在最贴近的子方向；可在 PR description 中说明。

### PR Checklist

- [ ] arXiv ID 准确、链接可访问
- [ ] 时间格式 `(YYYY-MM)`
- [ ] 简介 ≤ 80 字
- [ ] 章节内时间倒序
- [ ] 不重复（先 grep 现有论文）

## 反馈
有疑问请开 Issue。
