# How Does Chain-of-Thought Help Language Models Count?

这是面向 arXiv 阅读、后续可迁移到 ICLR 2027 模板的机制分析草稿。正文为英文；本说明和证据索引为中文。

## 论文主线

论文比较两类 counting computation，而不把“能 probe 出数字”直接等同于“模型在用 counter”：

1. **Behavior**：在相同 passage 上比较 Direct、Native CoT、Index enumeration 和 Bullet enumeration；报告 exact accuracy、绝对误差和可解析性随 needle 数量 \(N\) 与 context length \(L\) 的变化。
2. **Direct 的竞争假设**：
   - \(H_{\mathrm{run}}\)：模型在读 passage 时逐记录执行 \(+1/0\) 更新；
   - \(H_{\mathrm{agg}}\)：模型主要在答案位置广泛读取多个记录，再形成 terminal total。
3. **CoT 的顺序机制**：targeted retrieval \(\rightarrow\) 写出一条证据 \(\rightarrow\) semantic progress transition \(\rightarrow\) 检索下一来源或停止 \(\rightarrow\) terminal readout。
4. **因果桥**：只有当局部 ablation/patching 的效应依次传播到 source identity、unique coverage、progress/stop、terminal state 和完整答案时，才把相关表示或 attention component 命名为机制的一部分。

主机制模型为 **Qwen3-8B** 和 **Gemma4-E4B**。更大的模型集合只用于行为 response surface；它不承担主要的因果机制结论。

## 文件结构

- `main.tex`：论文入口、摘要、引言、四联主图和总论证。
- `sections/experimental_design.tex`：数据生成、prompt/policy 控制、token landmark、hidden-state 抽取和聚合规则。
- `sections/behavior.tex`：行为结果、模型集合、误差分解和 empirical-law TODO。
- `sections/representations.tex`：\(H_{\mathrm{run}}\) 与 \(H_{\mathrm{agg}}\)、CoT progress、PCA/probe、steering 和 patching。
- `sections/routing.tex`：broad aggregation、targeted retrieval、progress transition 的分数与功能验证。
- `sections/synthesis.tex`：time-indexed causal chain、noise 定义、factorial intervention 和跨模型复现。
- `sections/synthetic_discussion.tex`：v10/v15/v20 的证据边界与 training-dynamics 扩展。
- `sections/appendices.tex`：完整行为表、排除模型、TODO ledger、prompt/activation manifest 和统计协议。
- `figures/mechanism_overview.tex`：可编辑的 standalone TikZ 四联主图。
- `figures/mechanism_overview.pdf`：主文实际引用的预编译图。
- `figures/mechanism_overview_legacy_two_lane.tex`：合并前的两泳道图，保留用于追溯。
- `SOURCE_MAP.md`：本地报告、表格与草稿中每项结论的证据边界。
- `references.bib`：公开文献。
- `counting_mechanism_draft.pdf`：交付 PDF。

## 固定的测量约定

- passage 内 record state：在每个 `[END-RECORD]` 的闭括号 token 后读取 `resid_post`。
- Direct terminal state：在 teacher-forced `Total:` 的冒号 token 后读取。
- CoT retrieval query：在不含可见序号的 `Item:` 冒号 token 后读取。
- CoT progress state：在第 \(k\) 条已写来源的闭括号 token 后读取。
- 不在 PCA、probe 或 patch 前跨 prompt/样本平均 raw hidden state；原子观测是 family × variant × landmark × layer。
- PCA 仅作可视化；probe、几何、patch 和统计均使用完整 residual space，并按 semantic family 等权汇总。
- 只有 steering direction 可以由训练 split 上的跨 prompt 条件均值定义；主 patch 使用单个自然 donor。
- 若 passage 前缀逐 token 与 prefix hash 完全相同，Direct 与 CoT 在 passage 内的 hidden states 理论上相同；不能把共享 passage state 写成 mode-specific counter。

## TODO 实验顺序

1. `B0–B1`：配置/失败审计与 paired behavior grid。
2. `D1–D2`：冻结 controlled mechanism corpus、prompt 与 reasoning-policy factorial。
3. `R1–R3`：registered-state decoding；Direct 的 prefix/total shift；CoT 的 anti-leakage 与 progress intervention。
4. `H1–H2`：在 discovery split 找候选，在独立 split 做 selectivity、query-local ablation、clean patching 和 free-running 验证。
5. `M1`：在 Qwen3-8B 上完成 route × state × terminal 的整链 factorial。
6. `M2`：null scratchpad、semantic corruption、duplicate/false-stop/common-evidence controls，并在 Gemma4-E4B 复现功能角色。
7. `B2`：仅在 untouched grid 有预测力时报告 empirical law，否则保留为定性 response surface。
8. `S1`：v20-style synthetic setting 的多 seed、objective-switch 与 all-sequence-throughout controls。
9. `R0`：为每个公开结果冻结数据、token、activation、head-selection 和 intervention manifest。

其中，Qwen 机制主张至少需要 `B0–B1, D1–D2, R1–R3, H1–H2, M1, R0`；Qwen–Gemma 共性主张还需要 `M2`。`B2` 与 `S1` 是可选扩展，不应阻塞主机制论文。

## 编译

先编译 standalone 主图，再编译正文：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error figures\mechanism_overview.tex
python C:\Users\HP\.codex\plugins\cache\openai-bundled\latex\0.2.4\scripts\compile_latex.py main.tex
Copy-Item -LiteralPath main.pdf -Destination counting_mechanism_draft.pdf -Force
```

主文引用 `figures/mechanism_overview.pdf`，不要用 `\input` 嵌入 standalone 图源。
