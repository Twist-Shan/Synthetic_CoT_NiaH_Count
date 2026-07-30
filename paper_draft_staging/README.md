# Counting mechanism paper draft

这是面向 arXiv 阅读和后续 ICLR 2027 投稿的普通 LaTeX 草稿。正文为英文；本说明和证据索引为中文。

## 当前主线

论文围绕一个问题展开：**How does chain-of-thought help language models count?**

正文的四层证据链为：

1. **Behavior**：确定 CoT、Index、Bullet 相对 Direct 的准确率优势，以及准确率和绝对误差随 needle 数量 \(N\) 与 context length \(L\) 的变化。
2. **Representation**：在 matched semantic sites 上寻找 Direct final total、CoT progress 和 CoT final total；用 held-out PCA/centroid curve 描述，用 geometric steering 和 natural-state patching 验证因果可执行性。
3. **Routing circuit**：只保留三类核心 head role：Direct broad-source、CoT matched-source、CoT next-source/continue-stop。Attention 只用于候选发现，功能命名必须通过 query-local ablation、clean restoration 和 free-running validation。
4. **Mechanism synthesis**：干预 targeted retrieval，测 source identity、unique coverage、final-state SNR 和 exact count；在上游 route 仍被破坏时恢复 clean identity state 或 \(Z_C\)，检验完整的 retrieval \(\rightarrow\) state quality \(\rightarrow\) accuracy 链条。

主机制模型为 **Qwen3-8B** 和 **Gemma4-E4B**。Gemma4-E4B 按模型报告为约 4.5B effective/non-embedding parameters、8B including embeddings；正文同时说明两种口径。

## 文件

- `main.tex`：入口文件。
- `sections/behavior.tex`：行为结果、模型集合和 empirical-law TODO。
- `sections/representations.tex`：两种最小机制、counter state、PCA/geometry、steering/patching。
- `sections/routing.tex`：三类 routing head 分数、heatmap 规划及因果实验。
- `sections/synthesis.tex`：retrieval noise、state noise、固定 coverage 和 gap-removal 实验。
- `sections/synthetic_discussion.tex`：v10/v15/v20、training dynamics、threats、related work。
- `sections/appendices.tex`：完整行为表、排除模型、实验 ledger、controls 和统计细节。
- `figures/mechanism_overview.tex`：可编辑 TikZ 主架构图。
- `figures/synthetic_v10_attention_signatures.png`：v10 preliminary panel。
- `figures/synthetic_v20_mechanism_timeline.png`：v20 one-seed timeline。
- `references.bib`：公开文献。
- `SOURCE_MAP.md`：草稿中的证据和本地报告/表格映射。
- `archive/main_pre_four_layer_restructure.tex`：本轮四层重构前的旧正文。
- `counting_mechanism_draft.pdf`：交付 PDF。

## 行为模型集合

主文 response-surface panel：

- Qwen3-4B、8B、14B、32B；
- Gemma4-E4B；
- GLM-4-9B 与 GLM-Z1-9B；
- Cogito-v1-Preview-8B；
- Nemotron-Nano-v2-9B；
- Nemotron-3-Nano-4B。

当前 Qwen3-14B 四种模式全部缺失，已登记为 2,000-request held-out TODO。GLM 当前是 GLM-4 的 Direct/Index/Bullet 与 GLM-Z1 的 CoT 拼接，只能作 descriptive slot，不能作 same-weight thinking toggle。

其余已审计模型的 numerical exact accuracy 和排除原因都放在附录，包括 Qwen3-1.7B、Gemma4-12B、DeepSeek-R1-Qwen3-8B、Granite、Ministral 和 OLMo3。

## 编译

在本目录运行：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

本轮已用 TeX Live 2025 编译，并将全部 29 页渲染成 PNG 逐页检查；最终日志中没有 overfull box 或 unresolved reference。

## 首要实验顺序

1. `B0`：Qwen3-8B 与 Gemma4-E4B 的 500-context paired behavior rerun。
2. `B1`：冻结 Qwen 4B/8B/32B surface 后，生成 Qwen3-14B 四模式 held-out grid。
3. `R1–R3`：counter curve、geometric/natural transport、iterative-counter event test。
4. `H1-D/H1-C/H2`：三类 routing role 的 discovery、ablation、patching 和 free-running validation。
5. `M1–M3`：targeted retrieval \(\rightarrow\) unique coverage \(\rightarrow Z_C\) quality \(\rightarrow\) exact count，以及 fixed-coverage、null-compute 和跨模型复现。
6. `S1–S2`：synthetic multi-seed dynamics 和 objective/exposure controls。
