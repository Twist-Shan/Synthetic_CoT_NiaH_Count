# ICLR 2027 论文 Roadmap 评估与执行建议

## 结论

这条路线有 ICLR 主会潜力，但论文主张必须从“CoT 提高 counting accuracy”升级为：

> **在稀疏长上下文 counting 中，显式推理使模型从一次性的 broad normalized aggregation / global readout，切换到 sequential targeted retrieval + trace-local accumulation；该切换通过表示、head-level routing 与干预被因果连接到 count error。**

如果最后只有“thinking 更准、attention 更集中、PCA 有一条曲线”，在 2026 年已有 counting-mechanism 文献背景下，novelty 和 technical soundness 都不够。真正有竞争力的是 **long-context sparse semantic retrieval 场景中的 algorithmic path switch**，以及对 direct path 的反证式分析：direct 未必使用一个干净、因果生效的 running counter。

## 1. 当前 roadmap 最重要的修正

### 1.1 不要预设 non-thinking 有 noisy running counter

现有 Qwen pilot 更支持：needle 信息可以被识别，局部位置也可能含有 ordinal information，但 direct answer 的任务相关计算主要在后部形成，表现为 softmax-normalized broad aggregation、density/proportion-like code 或 noisy global readout。  
因此正文应并列检验：

- \(H_{\mathrm{run}}\)：每经过一个 relevant record，存在随 \(+1\) 更新且会影响后续输出的 counter state；
- \(H_{\mathrm{agg}}\)：局部有可解码信息，但真正用于输出的 count 主要由 answer query 附近的 delayed global aggregation/readout 形成。

只有通过 clean/corrupt state patching 后，后续 state 和最终答案都按 \(\Delta\) 移动，才能把某个 state 称为 counter。

### 1.2 “broad retrieval”建议改成“broad aggregation”

Direct mode 可能已经能定位 needles，但无法稳定把多个 needle 聚合成精确整数。继续叫 broad retrieval 会把 retrieval failure 和 aggregation failure 混为一谈。建议术语：

- **broad aggregation component**：final query 对许多 gold records 分散取信息，形成全局 numerosity/density readout；
- **targeted retrieval component**：trace step \(k\) 选择下一个未访问 gold record；
- **progress-transition / update component**：推动 pointer 或 accumulator 更新。

“successor-like head”不要直接当正式名称。只有证明其 OV circuit 在换标签后仍实现抽象 \(k\mapsto k+1\)，才叫 successor head；否则用 progress-transition head。

### 1.3 PCA 是展示，不是机制证据

证据层级必须写死：

1. **Decodable**：held-out probe 能解码 count/progress；
2. **Dynamic**：needle 后 \(+1\)，distractor 后近似不变；
3. **Causal**：patching/steering 改变后续 progress 与 final count。

只有三层同时满足，才叫 internal counter。否则用 “count-decodable representation” 或 “ordinal information”。

## 2. non-thinking 到底在哪里看 counter

主分析位置应是 **同一个 forward pass 中每个原文 needle record 的终点 landmark**，再配 matched distractor landmarks。具体为：

- record 最后一个内容 token；
- 或 record 后稳定 delimiter；
- 多 token record 必须固定 landmark 规则。

同时分析两类位置：

1. 原文 needle/distractor landmarks：检验 running counter；
2. final query / answer-prefix state：检验 global count readout。

只在不同 prompt 的 final hidden state 上预测 \(N\)，最多说明“最终状态包含 count 信息”，不能说明模型沿原文逐步计数。

Probe 需要同时控制或联合预测：

\[
c(t),\quad N,\quad N-c(t),\quad t,\quad t/L,\quad N/L,\quad L.
\]

否则 count probe 很可能只是 position、density 或 total-count leakage。

## 3. thinking counter 的关键反泄漏设计

Thinking/Index-Enumeration 最大风险是 index token 自带答案。机制主实验必须包括：

- 无数字 enumeration；
- 随机非数字标签；
- visible index permutation；
- 在数字生成前的 neutral separator 上取 hidden state；
- 用“已成功检索的 distinct gold records 数量”定义 semantic progress，而不是文本中的 \(k\)。

若 counter geometry 在这些条件下仍存在，且 patching 能改变下一次 retrieval 和最终 count，才可以说是 trace-local accumulator。

## 4. 模型集合如何组织

### Behavior / law panel

保留你列出的较大集合：

- Qwen family: 4B / 8B / 14B / 32B；
- Gemma4-E4B；
- GLM-4-9B 与 GLM-Z1-9B；
- Cogito-v1-8B；
- Nemotron-Nano-v2-9B；
- Nemotron-3-Nano-4B。

但不要把它们视为一个干净的 parameter scaling sequence。dense/MoE、tokenizer、post-training、native reasoning policy 都不同。推荐：

- 先做 **within-family** Qwen curves；
- 再做 family-specific hierarchical response surfaces；
- pooled cross-family slope 只作为 secondary analysis。

你目前 law 波动大、linear 与 log-linear 差别小、系数不稳定，这意味着 law 不应成为论文主贡献。预注册候选形式，用 grouped held-out log loss/Brier/RMSE 选择；不能优于 condition-only baseline 就定性报告。

### Mechanism panel

Qwen-8B 与 Gemma4-E4B 足够，前提是两者都能稳定跑四种 mode，并能获取全部 hidden states/head outputs。  
跨模型不要求相同 head index，而要求相同 **functional decomposition**。若 Gemma 的具体实现不同，也可以写成“functional convergence with implementation heterogeneity”。

## 5. 主实验的正确顺序

### Figure 1：行为与 failure budget

同一 stimulus 配对比较 Direct / Enum / Index-Enum / Native-CoT，展示：

- exact accuracy；
- MAE 与 signed error；
- parse/truncation/format；
- enumeration unique recall、FP、duplicate、aggregation error；
- mode × query-order interaction。

当前结果已经说明 query-order effect 不是统一主效应，因此正文不能把 query order 合并掉。

### Figure 2：representation evidence ladder

每个模型分别展示：

- PCA/3D trajectory；
- held-out count/progress decoding；
- within-count dispersion；
- adjacent-count separation；
- correct vs wrong；
- direct needle landmarks vs CoT neutral landmarks；
- steering/patching effect。

### Figure 3：head taxonomy

Attention heatmap 只用于候选发现。每类 head 同时展示：

- observational metric；
- frozen held-out head set；
- head-output patching；
- resample/mean ablation；
- random-head、same-layer、attention-mass matched control。

### Figure 4：causal bridge

核心不是“ablate head accuracy 掉了”，而是：

\[
\text{intervention}
\to \text{retrieval/update metric}
\to \text{progress state}
\to \text{final count error}.
\]

这是整篇 paper 最能区别于已有 counting MI 工作的部分。

### Figure 5：synthetic training dynamics

在最终 v15/v20-style setting 上跟踪：

- accuracy；
- decodability；
- dynamic consistency；
- causal patch restoration；
- broad aggregation score；
- targeted retrieval score；
- emergence time 与 seed variability。

v10 可以作为完整 causal suite 的来源，但不能让 v10、v15、v20 的 task format、loss mask 或 architecture 混在主结论里。当前可访问材料中未定位到 v20 报告，因此草稿中已显式保留 TODO。

## 6. Noise 应如何定义

不要用一个模糊的 “noise” 覆盖所有现象。至少拆成：

- **representation noise**：固定 count 下 hidden-state dispersion；
- **decoder noise**：held-out count decoder 的条件 MSE；
- **retrieval noise**：miss / false positive / duplicate；
- **attention routing noise**：target mass、duplicate mass、entropy；
- **protocol noise**：format、stop、aggregation arithmetic。

Direct 的一个可检验模型是 normalized density code：

\[
\rho(N,L)=\frac{N}{N+\alpha(L-N)}+\eta,
\qquad
N=\frac{\alpha L\rho}{1-(1-\alpha)\rho}.
\]

逆映射对 \(\rho\) 的导数随 \(L\) 放大，因此小的 representation error 可转化为随上下文增长的 count error。  
CoT 则把一次 ill-conditioned global inversion 换成多个局部、可检查的 retrieval/update step。该解释只有在 intervention 改变 routing noise 后 count error 同步变化时才成立。

## 7. Novelty 风险与论文定位

截至 2026 年已有三类非常接近的工作：

1. repeated-item counting 中的 internal counter / CountScope；
2. 通过输入 partition + intermediate counts 做 System-2 large-scale counting；
3. count representation 与 output digit direction 的 geometric readout bottleneck。

因此不能把贡献写成：

- “首次发现 LLM 有 counter”；
- “首次发现 CoT 可以帮助 counting”；
- “首次发现有 attention head 传递 count”。

可 defend 的定位是：

> **我们研究 sparse semantic NIAH counting，而不是 homogeneous repeated lists；输入不做外部分块；在同一 unstructured context 上，直接比较 direct 与显式推理，并因果识别 retrieval-and-aggregation algorithm 的切换。**

如果 P0 实验全部成立，这一定位足以形成强 ICLR story。若只成立 targeted retrieval，而 accumulator 或 direct global-aggregation 证据不足，则收缩为“CoT improves counting by sequentializing semantic retrieval”，仍然可以成文，但贡献会弱一档。

## 8. Go / No-Go 标准

### 可进入 ICLR 主线

- 两个 mechanism models 中至少一个有完整因果链，另一个有 functional replication；
- direct path 的 running-counter 与 delayed-aggregation 假设被清楚区分；
- CoT accumulator 通过 no-index controls；
- 至少一类 targeted head 在 held-out prompts 上同时有 selectivity、necessity、partial sufficiency；
- routing/state intervention 对 count error 有可重复的方向性作用；
- 行为结果在 matched stimuli 与 failure-budget 分解后仍成立。

### 应收缩主张

- probe 很好但 patching 无效；
- index controls 后 counter 消失；
- attention heatmap 清晰但 head-output intervention 不选择性；
- Qwen/Gemma 只在模板特定条件成立；
- law 不能 held-out 泛化。

这种情况下仍可写，但题目和摘要应聚焦“behavioral decomposition + retrieval sequentialization”，不要写完整 circuit 或 internal counter。
