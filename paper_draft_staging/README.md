# Counting mechanism paper draft

这是面向 arXiv 阅读与内部迭代的普通 LaTeX 预印本草稿；当前没有绑定 ICLR 模板。正文为英文，工作说明与证据索引为中文。

## 文件

- `main.tex`：当前编译为 22 页的论文正文与附录。
- `references.bib`：公开论文引用。
- `figures/mechanism_overview.tex`：可编辑的 TikZ 主机制图。
- `figures/synthetic_v10_attention_signatures.png`：v10 的 preliminary attention panel。
- `figures/synthetic_v20_mechanism_timeline.png`：v20 的 one-seed training timeline；当前沿用内部报告标签，正文已经标明应替换。
- `SOURCE_MAP.md`：草稿中的数字、机制证据与本地报告/表格的对应关系。
- `counting_mechanism_draft.pdf`：经过编译和逐页版式检查的交付 PDF。

## 当前论文主线

最稳妥的叙述不是“non-thinking 检索不到、thinking 才能检索”，而是：

1. Direct/non-thinking 往往能够定位 targets；候选机制是分布式 tagging、answer-query 处的 broad parallel retrieval、压缩的 global total state 与离散数字读出。
2. Thinking 的候选机制是逐项 targeted retrieval、progress/termination control，以及与进度状态可分的 final-total state。
3. “可线性解码的 count code”不等于“algorithmic counter”。只有跨位置/词面/未见 count 泛化的稳定更新律，并且通过 state/head intervention，才能称为 counter。
4. “count 越大越 noisy”目前是待检验假设，而不是既有结论；正文预注册了 within-label covariance、local SNR、decoder residual variance 和行为条件方差。
5. 当前 law 只能称为 count--length response surface。Direct 的 linear 与 log-length 形式近似并列且系数不稳定，不宜宣称统一 scaling law。

## 建议模型集合

- 行为与 response-surface：Qwen3 1.7B/4B/8B/32B；只有冻结模型能预测 held-out size 时才补 0.6B/14B。Gemma4-E4B/12B 用作跨 family 验证。
- 主机制：Qwen3-8B。
- failure/capacity control：Qwen3-4B。
- 跨架构复现：优先评估 Gemma4-12B 的 activation 成本与 matched-decoding numerical accuracy；资源不足则使用 E4B。
- 高容量 spot check：Qwen3-32B，只做预注册的稀疏层/头集合。

## 编译

在本目录运行：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

如果只想清理辅助文件，可使用 `latexmk -c main.tex`；不要删除 `main.tex`、`references.bib` 或 `figures/`。

## 首要 TODO

1. 同权重、同输入、匹配 decoding/token budget 的 Qwen3 与 Gemma4 paired rerun。
2. semantic-anchor representation map，并严格区分 candidate-validity tag、source ordinal/target rank、prompt prefix、trace semantic progress 与 final total。
3. Qwen3-8B 的 state transport、pre-`o_proj` head-slice patch、query-local ablation 与 mediation。
4. answer position × trace extent 因子控制以及 free-running trace 验证。
5. 至少五个 synthetic seeds、objective-switch controls、capability matching，并修复 v20 causal stage。
6. 用冻结表格重绘 publication-quality English figures，并把所有 preliminary 数字审计到 request-level export。
