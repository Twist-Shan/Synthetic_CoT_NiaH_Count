# 合并后的论文大纲与机制分析方案

## 1. 核心结论

论文不应预设：

> non-thinking 使用一个 noisy counter，而 CoT 使用一个更干净的 counter。

更严谨、也更有 novelty 的工作假设是：

> 在 sparse semantic long-context counting 中，Direct/non-thinking 可能主要依赖 answer-query 附近的 broad multi-record aggregation 与 terminal total readout；CoT 则可能把计算 sequentialize 为 matched-source retrieval、trace writing、progress update 和 next-source/stop control。论文的目标是用干预区分这些 computation，而不是把它们画成既定事实。

这条主张需要形成一条闭合证据链：

\[
\text{behavioral advantage}
\rightarrow
\text{count/progress state}
\rightarrow
\text{routing and update component}
\rightarrow
\text{retrieval/coverage and state quality}
\rightarrow
\text{exact count}.
\]

只有最后一层成立，摘要才能使用 “causally identify an algorithmic path switch”。如果只发现 targeted retrieval，应收缩为 “CoT sequentializes semantic retrieval”；如果只有 probe/PCA，应收缩为 “count-decodable geometry”。

## 2. 对 pro 分析的取舍

### 保留并提升为主线

1. Direct 的 \(H_{\mathrm{run}}\) 与 \(H_{\mathrm{agg}}\) 竞争检验。
2. Counter 的三级证据：decodable、dynamic、causal。
3. CoT semantic progress 和 index-leakage controls。
4. Broad aggregation、targeted retrieval、progress-transition/update 三种功能角色。
5. Attention 只做 discovery，必须配 held-out ablation、clean patch 和 matched controls。
6. 机制主结果必须是：

   \[
   \text{head/state intervention}
   \rightarrow
   \text{source identity / unique coverage}
   \rightarrow
   \text{progress/final-state quality}
   \rightarrow
   \text{complete parsed count}.
   \]

7. Behavior panel 与 mechanism panel 分层；Qwen3-8B 和 Gemma4-E4B 为 co-primary mechanism models。

### 修正

1. **Query-last 的因果掩码问题。**  
   当 query 位于 context 之后时，原文 record endpoint 在生成时尚未看到 query，不能编码 query-specific \(c_q(t)\)。因此：

   - running-counter landmark test 只用于 query-first 或 query-before-context 的条件；
   - query-last 只能在 final query/answer-prefix 分析对 context 的回读与聚合；
   - query order 可以进入 behavior factorial，但不能把两种条件混进同一个 local-counter estimand。

2. **Direct 术语。**  
   观察阶段称 broad source-attending / broad aggregation candidate。只有 head-output intervention 改变 terminal total state 和完整答案后，才称 causal broad-aggregation component。避免把 retrieval failure 与 aggregation failure混为一谈。

3. **CoT 的 next target。**  
   Native trace 不一定存在唯一的“下一个未访问 source”。主 next-source score 应在 canonical order 中定义；native free-running trace 使用实际 aligned source 作执行验证。

4. **Noise。**  
   正文只保留两个与机制直接相关的对象：

   - retrieval/update error：miss、wrong source、duplicate、false positive、stop；
   - conditional state dispersion：固定 semantic count/progress 下的 hidden-state dispersion，相对于相邻 count separation 归一化。

   Parse、format、truncation 放在 behavior failure budget；attention entropy 只是 descriptive routing statistic；probe MSE 是 decodability，不等于 representation noise。

### 降级或删除

1. Normalized-density 具体逆映射移到附录，作为 \(H_{\mathrm{agg}}\) 的 auxiliary falsifiable model，不作为 Direct 的默认机制。
2. 删除 “successor-like” 正式命名。除非 relabeling 后 OV circuit 仍实现抽象 \(k\mapsto k+1\)，否则称 progress-transition 或 next-source control。
3. PCA/3D trajectory、attention heatmap 和模拟 causal curve 不放进 overview；分别作为真实结果图。
4. Empirical law 不是主贡献。Qwen family 做 held-out within-family validation；失败则只报告定性 response surfaces。
5. Synthetic dynamics 只跟踪与 LLM 相同的 causal metrics；否则移入附录。
6. 删除正文中的内部 roadmap 语言，如 “paper succeeds only if” 和大段 go/no-go 元叙事；claim gates 放附录。

## 3. 两种模式的分析设计

### 3.1 Direct / non-thinking

#### 最小 computation

\[
w_i^D=\mathcal R_D(q,R_i),\qquad
Z_D=\mathcal G_D(\{w_i^D V_i\}_{i=1}^R),\qquad
\widehat N_D=\mathcal Q_D(Z_D).
\]

该表达只规定多个 record contribution 进入 terminal state，不预设 \(\mathcal G_D\) 是 running counter、parallel sum、normalized density 还是 multi-hop aggregation。

#### 竞争假设

- \(H_{\mathrm{run}}\)：在 query-first 条件下，每个 valid record 后存在 \(+1\)、distractor/duplicate 后存在 \(0\) 的 causally used running state。
- \(H_{\mathrm{agg}}\)：local states 可能含 salience/ordinal information，但 task-relevant total 主要在 final query 附近形成。

#### 注册位置

- query-first：每个 record 的固定 record-end/delimiter landmark \(h_{a_i}^{\ell}\)；
- 所有 query orders：final query / pre-answer state \(Z_D^\ell=h^\ell(q_D)\)；
- query-last 原文 endpoint 只可分析 query-independent content encoding，不进入 query-specific counter test。

#### 判别实验

1. Joint/residualized probe 同时控制 \(c(t),N,N-c(t),t,t/L,N/L,L\)。
2. Valid、invalid、duplicate record 的 \(+1/0\) update test。
3. Same-position natural donor patch：局部 state 是否改变后续 landmarks 与 final answer。
4. Final-query source add/delete、head ablation 和 \(Z_D\) restoration。
5. 若 local test 失败但 terminal state 可运输，结论为 delayed terminal aggregation，而不是“Direct 没有任何 count information”。

### 3.2 CoT / explicit enumeration

#### 最小 computation

\[
\pi_k=\Pi_C(q,\Gamma_{k-1}),\quad
e_k=\mathcal W_C(R_{\pi_k}),\quad
\Gamma_k=\mathcal F_C(\Gamma_{k-1},e_k),\quad
\widehat N_C=\mathcal Q_C(\Gamma_K).
\]

\(\Gamma_k\) 允许包含 internal residual state、visible trace 和 KV-cache ledger；不预设它是一个 scalar neural counter。

#### Semantic progress

\[
p_k=
\left|
\bigcup_{u\le k}\operatorname{uniq}(\widehat G_u)\cap G_q
\right|.
\]

标签是已成功检索的 distinct valid sources 数量，不是 emitted step \(k\) 或可见数字。

#### 注册位置

- retrieval query \(q_k^{\mathrm{ret}}\)：source-bearing content 生成前；
- progress landmark \(q_k^{\mathrm{end}}\)：当前 item 结束、下一 index/number 生成前的 neutral separator；
- final state \(Z_C^\ell=h^\ell(q_C)\)：最终答案第一位之前。

#### Anti-leakage controls

1. unindexed enumeration；
2. random nonnumeric labels；
3. permuted visible indices；
4. pre-number neutral landmarks；
5. fixed emitted step with different semantic coverage；
6. same semantic progress with different trace extent。

#### 判别实验

1. Matched-source head patch 是否改变 emitted source identity。
2. New-valid item 是否 \(+1\)，invalid/duplicate 是否 no-op。
3. Progress-state patch 是否改变 next source、continue/stop 和 final count。
4. Residual patch 与 trace-token/KV patch 分离：
   - residual transport 支持 internal progress register；
   - trace/KV-only transport 支持 external ledger。

## 4. 合并后的正文结构

### 1. Introduction

- sparse semantic counting 为什么同时测试 retrieval、coverage、aggregation 和 readout；
- 核心问题：CoT 是否改变 retrieval-and-aggregation computation；
- behavioral fact、mechanism comparison、integrated causal test 三项贡献；
- novelty 不写成“首次发现 counter/head/CoT counting”。

### 2. Behavioral Phenomenon

1. Task、四种 mode 和 metrics；
2. broad behavior panel 与 mechanism panel；
3. matched-stimulus accuracy、MAE、signed error；
4. parse/format/truncation 与 enumeration failure budget；
5. core regime 的 trace advantage；
6. \(N,L\) partial trends；
7. Qwen within-family response surfaces，empirical law 作为 secondary held-out test。

Direct 与 Native-CoT 为主比较；unindexed/indexed enumeration 为 structured process controls。

### 3. Competing Computations and Registered States

1. Direct 与 CoT 两条最小计算式；
2. \(H_{\mathrm{run}}\) 与 \(H_{\mathrm{agg}}\)；
3. query-first/query-last 可识别性的区别；
4. Direct record landmarks、\(Z_D\)、CoT \(q_k^{\mathrm{ret}}\)、\(P_k\)、\(Z_C\)；
5. counter 的 decodable → dynamic → causal 命名门槛。

### 4. Counting Representations and Causal State Control

1. held-out decoder 与 2D visualization；
2. full-space counter/progress centroids；
3. adjacent separation 与 conditional state dispersion；
4. Direct valid/distractor update；
5. CoT semantic-progress/no-index tests；
6. residual-preserving steering；
7. same-position natural-state transplant。

### 5. Routing and Update Components

1. broad aggregation candidate；
2. matched-source targeted retrieval candidate；
3. next-source/progress-transition candidate；
4. discovery → freeze → held-out selectivity → ablation → clean patch rescue；
5. attention span mass、raw mass 和 conditional selectivity同时报告；
6. head/attention/MLP/residual 分层 localization。

### 6. Why CoT Helps: Integrated Causal Chain

1. retrieval/update error 与 conditional state dispersion；
2. targeted-route intervention 是否改变 source identity、unique recall、duplicate/FP；
3. 是否继而改变 progress/final-state SNR 和 true-count margin；
4. clean identity/progress/final-state restoration 是否恢复完整 parsed count；
5. fixed-coverage control：区分 coverage rescue 与 beyond-coverage aggregation improvement；
6. equal-length null scratchpad 和 semantic trace corruption；
7. Qwen3-8B 完整链条与 Gemma4-E4B functional replication。

### 7. Cross-Model Replication and Synthetic Emergence

1. 两个 mechanism model 以 semantic role 而非 head index 对齐；
2. synthetic v15/v20 只跟踪同一套 decodability、dynamic、causal 和 routing metrics；
3. multi-seed emergence；
4. objective switch/exposure controls；
5. 不可比时降为 appendix。

### 8. Related Work, Limitations, and Conclusion

定位为 sparse semantic NIAH 上的 direct/CoT algorithmic-path comparison。

### Appendix

- full model/failure table 和排除原因；
- density-code auxiliary hypothesis；
- query-order controls；
- TODO ledger；
- claim gates；
- synthetic additional results；
- reproducibility checklist。

## 5. Claim ladder

- **C0 Behavioral**：matched stimuli 上 trace-generating modes 在 core regime 优于 Direct。
- **C1 Process**：CoT 更稳定地逐步检索 distinct valid sources；Direct 的 task-relevant readout 更集中于 final query。
- **C2 Representation**：CoT semantic progress 满足 decodable 和 dynamic；Direct local state 与 terminal total 被区分。
- **C3 Component causality**：targeted retrieval、progress transition 和 broad aggregation candidates 对相应 semantic endpoint 具有 necessity 与 partial sufficiency。
- **C4 Integrated causality**：routing intervention 经 retrieval/update error 和 state quality 改变完整 count；clean state restoration 部分恢复性能。
- **C5 Generality**：Qwen3-8B 完整链条、Gemma4-E4B functional replication，并在 synthetic training 中出现兼容的 formation order。

## 6. 统一 TODO 编号

- `B0`：checkpoint/template/failure audit；
- `B1`：paired behavioral grid 与 Qwen3-14B held-out extension；
- `B2`：response-surface held-out validation；
- `R1`：counter/progress curve、decoder、state dispersion；
- `R2`：Direct \(+1/0\) dynamics 与 CoT anti-leakage；
- `R3`：steering 与 natural-state patching；
- `H1`：三类 role score 的 discovery/test screen；
- `H2`：head-output ablation、patching 与 semantic endpoints；
- `M1`：targeted route → retrieval/coverage → state → answer factorial intervention；
- `M2`：fixed coverage、null scratchpad 与跨模型 replication；
- `S1`：v15/v20-aligned multi-seed synthetic dynamics。

## 7. 新主图设计

Figure 1 是 graphical abstract，不混入伪 PCA、伪 heatmap 或预期结果曲线。

### 上半部分：同一输入上的两条 computation swimlane

- 横向列：shared context → routing → state computation → answer；
- Direct：
  - final-query broad multi-record access；
  - terminal \(Z_D\)；
  - 用虚线标出待检验的 \(H_{\mathrm{run}}\)，不写 “noisy”；
- CoT：
  - \(q_k\) matched-source retrieval；
  - visible item \(e_k\)；
  - internal/external carrier \(\Gamma_k\)；
  - continue/stop loop；
  - final \(Z_C\)。

### 下半部分：统一因果证据链

\[
\text{routing/state intervention}
\rightarrow
\text{source identity / unique coverage}
\rightarrow
\text{progress/final-state quality}
\rightarrow
\text{complete count}.
\]

在 routing state、progress state 和 final state 标出三个可 patch/rescue 的干预点。实线表示计算依赖，虚线表示竞争假设。图中不提前写 “Direct is noisy” 或 “CoT is precise”。

后续真实结果图顺序：

1. Figure 2：behavior surfaces + failure budget；
2. Figure 3：representation evidence ladder；
3. Figure 4：head roles + held-out causal effects；
4. Figure 5：integrated causal restoration；
5. Figure 6：synthetic emergence（满足 comparability gate 后）。
