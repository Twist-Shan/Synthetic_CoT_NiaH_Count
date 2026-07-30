# Claim-to-source map

本文档限定草稿的证据边界：

- **Observed**：可由冻结的 request-level 数据或表格复算。
- **Preliminary**：已有报告支持，但尚未在目标模型、配对数据或独立 split 上复现。
- **Hypothesis**：机制解释，必须通过正文登记的因果实验。
- **TODO**：尚不能写成结果或结论。

## Realistic NIAH behavior

基础目录：

`C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Realistic_CoT_NiaH_Count\reports\v2_realistic_niah_analysis_20260727`

补充报告：

- `Realistic_CoT_NiaH_Count\reports\NIAH-counting.html`
- `Realistic_CoT_NiaH_Count\reports\NIAH-4K-report-standalone.html`

| 草稿内容 | 主要来源 | 当前边界 |
|---|---|---|
| 26,500 requests、53 个 model–mode cells、\(N/L\)/seed 设计 | `README.md`, `analysis_manifest.json` | Observed；需要 B0 检查配置与 provenance |
| core-regime 与 full-grid accuracy | `tables/prompt/model_mode_summary.csv` 和 request-level compact table | Observed；exact、parse、format、truncation 必须分开 |
| accuracy 随 \(N,L\) 的总体下降与 absolute error 的总体上升 | `accuracy_by_N.csv`, `accuracy_by_L.csv` | Preliminary；只能说总体趋势，不能声称每个 cell 单调 |
| focused response surfaces | `tables/focused_empirical_law/*` | Exploratory；Qwen3-14B 尚未形成 untouched confirmatory grid |
| GLM logical slot | focused mapping 与 model summary | Descriptive；GLM-4/GLM-Z1 不是同权重 thinking toggle |
| Cogito/Nemotron 名称与 revision | reasoning-model extension 的 `family_manifest.json` | Observed metadata；正文用冻结名称 |
| 旧 realistic thinking mechanism | `NIAH-counting.html` | Preliminary motivation；不能替代 D1–M2 |
| 旧 4K Direct broad retrieval / final count code | `NIAH-4K-report-standalone.html` | Preliminary motivation；不能替代 Qwen/Gemma 因果实验 |

当前 focused behavior panel 包括 Qwen3-4B/8B/32B、Gemma4-E4B、GLM logical slot、Cogito-v1-Preview-8B、Nemotron-Nano-v2-9B、Nemotron-3-Nano-4B。Qwen3-14B 是 `B2` 的 confirmatory TODO。其余已审计模型保留在附录并报告 exclusion reason。

## Formal V2 generation facts

以下事实来自 V2 manifest/config 及生成脚本，应由 `B0/R0` 再冻结一次：

| 项目 | 固定值或规则 |
|---|---|
| source pool | 去重后的 218 篇 Paul Graham 文本 |
| target lengths | \(L\in\{2000,3000,5000,10000,20000\}\) Qwen tokens |
| target counts | \(N\in\{1,2,3,4,5,6,8,10,20,30\}\) |
| seeds | 1234–1243 |
| passages | 500 |
| target predicate | 固定年份、审计类型、城市和分数条件 |
| insertion | legal sentence boundaries；插入后按 Qwen tokenizer 验证 exact length；禁止 truncation |
| identity | occurrence ID、sentence span、token span、rendered-prefix hash |

V2 的 predicate 位于 passage 前，response suffix 位于 passage 后。若两种 mode 的 rendered prefix 在某 passage landmark 前逐 token 相同，则该处所有 deterministic hidden states 必须相同。这是分析约束，不是待验证的经验现象。

## Synthetic bridge

| 草稿内容 | 主要来源 | 当前边界 |
|---|---|---|
| v10 representation、routing、patching、ablation、steering | `v10_main_seed1234_20260712_172332/syn_v10_report.html` | Preliminary；单 seed、toy architecture、两种 mode 为独立模型 |
| v10 centroid geometry | `analysis/report_stratified/tables/centroid_mean_geometry.csv` | class means 不能单独证明 within-class noise |
| v10 local ablation / head patching | `position_local_ablation_by_bin.csv`, `nested_head_patching_regression_by_bin.csv` | 需要 matched random control；局部 transport 不等于完整算法 |
| v10 curved steering | `analysis/geometry_path_steering/tables/geometry_path_steering_regression.csv` | 支持 local path；不支持统一 global \(+1\) direction |
| v15 confounds | `v15_main_all_sequence_seed1234_20260718_171459/{syn_v15_report.html,config.json,manifest.json}` | 多因素同时变化；用作 anti-leakage/control 设计，不作最终机制 setting |
| v20 free-running accuracy | `v20_main_RoPE_count1-30_seed1234/tables/final_autoregressive_summary.csv` | Preliminary one-seed result；尚未 capability-match |
| v20 timeline | `analysis/mechanism_report_assets/mechanism_timeline.png` 与 phase tables | objective switch 是混杂；无 multi-seed/no-switch control 前不称 phase transition |
| v20 causal completeness | `manifest.json`, `analysis/v10_port/manifest.json` | 部分 stage 失败或为 repair artifact；`S1` 必须重跑并记录 provenance |

Synthetic 结果只用于提出测量方法和 training-dynamics 假设，不能写成 pretrained LLM 已验证的机制。

## 核心主张与证据门槛

| 主张 | 当前状态 | 升级为正文结论所需证据 |
|---|---|---|
| CoT/enumeration 在短上下文、\(N\leq10\) 时通常优于 Direct | Preliminary behavior | `B0–B1` 的同 passage 配对效应、bootstrap CI 与失败分解 |
| Direct 实现 passage-time running update \(H_{\mathrm{run}}\) | Hypothesis | `R1–R2`：record-end hidden state 的 sibling-controlled \(+1/0\) dynamics、prefix-shift localization、自然 state patch 对最终总数的定向影响 |
| Direct 实现 answer-time broad aggregation \(H_{\mathrm{agg}}\) | Hypothesis | `H1–H2`：terminal query 的多记录 fan-in、valid-vs-negative selectivity、局部损伤与 clean restoration、最终答案效应 |
| CoT 实现 targeted retrieval | Hypothesis | `H1–H2`：gold-source selectivity、source-identity retargeting、unique-coverage 与 free-running count 效应 |
| CoT 具有 semantic progress transition | Hypothesis | `R1,R3,H1–H2`：去除 visible index 后仍解码 semantic progress；new/repeated/non-target 控制；next-source/stop 的双向 patch |
| 表示是模型真正使用的 count/progress state | TODO | 同时满足 held-out decoding、正确更新动力学和 causal use；PCA 不能单独满足 |
| CoT 通过降低 functional retrieval/update noise 提高计数 | Hypothesis | `M1`：干预效应依次传播到 identity、coverage、progress/stop、terminal state、answer |
| 改善不是额外 token/共同证据/readout 造成 | TODO | `M2`：null scratchpad、semantic corruption、duplicate/false-stop、common-evidence readout |
| 机制在 Qwen3-8B 与 Gemma4-E4B 间功能复现 | TODO | `M2`：重新 discovery，不要求相同 head index；要求相同 proximal causal role |
| \(N,L\) 存在稳定 empirical law | Exploratory | `B2` 必须预测 untouched grid；若失败，只报告定性 response surface |

## 实验与分析审计规则

1. 每个精确数字必须回溯到冻结表格、manifest 和 configuration hash，而不只回溯到 HTML prose。
2. 数据 split 按 semantic family 分组，禁止同一家族的 prompt 变体跨 discovery/validation/test。
3. PCA 只可视化单个 state；不得先按 prompt 或 count 平均再 PCA。
4. probe 必须与 nuisance-only baseline 比较，并报告 family-held-out 增量。
5. attention score 只用于候选发现。功能命名要求独立 split 上的 query-local damage、clean restoration 和语义 retargeting。
6. head role 是功能角色，不是假设跨模型共享相同 layer/head index。
7. teacher-forced 实验只证明 local competence；自然执行结论必须在 free-running 的 first-error 前复现。
8. primary patch 使用一个自然 donor；跨 prompt 均值只用于预注册的 steering direction。
9. noise 只指可归因的 retrieval/update error 或 nuisance-conditioned state dispersion；attention entropy 仅作描述量。
10. 对多个候选 head/bundle 的确认采用预注册上限和 family-wise 校正；不得在 test split 重新挑选。
