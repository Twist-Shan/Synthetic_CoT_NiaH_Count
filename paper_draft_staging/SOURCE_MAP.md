# Claim-to-source map

本文档限制草稿的证据边界。`Preliminary` 表示可报告但尚未在目标设置复现；`TODO` 表示不能写成已证实结论。

## Realistic NIAH behavior

基础目录：

`C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Realistic_CoT_NiaH_Count\reports\v2_realistic_niah_analysis_20260727`

| 草稿内容 | 主要来源 | 边界 |
|---|---|---|
| 26,500 requests、53 个 model–mode cells、\(N/L\)/seed 设计 | `README.md`、`analysis_manifest.json` | 完成的行为审计；不是 matched-policy mechanism study |
| 主文 core-regime 和附录 full-grid accuracy | `tables/prompt/model_mode_summary.csv` 及 request-level compact table | numerical exact、registered success、parse、format、truncation 必须分开 |
| \(N\) 与 \(L\) 的 accuracy 趋势 | `tables/prompt/accuracy_by_N.csv`、`accuracy_by_L.csv` | “generally decreases”，不能宣称每模型逐点单调 |
| focused 8-slot response surfaces | `tables/focused_empirical_law/model_mode_mapping.csv`、`selected_mode_laws.csv`、`selected_model_fit_metrics.csv`、`selected_model_coefficients.csv` | 只能称 exploratory response surface；Qwen3-14B 未运行 |
| GLM logical slot | 同上 mapping 和 model summary | GLM-4 提供 Direct/Index/Bullet，GLM-Z1 提供 CoT；不能作 same-checkpoint causal effect |
| Cogito/Nemotron 精确模型名与 revision | reasoning-model extension 下各模型 `family_manifest.json` | Cogito 的冻结名称为 `Cogito-v1-Preview-8B` |
| 旧 realistic thinking mechanism | `reports/NIAH-counting.html` | 仅作动机；需要在 V2 paired setting 复现 |
| 旧 4K non-thinking broad retrieval / final count code | `reports/NIAH-4K-report-standalone.html` | preliminary mechanism evidence；不能替代 Qwen3-8B/Gemma4-E4B 因果实验 |

## Synthetic bridge

| 草稿内容 | 主要来源 | 边界 |
|---|---|---|
| v10 representation、routing、patching、ablation、steering | `colab_results/v10_main_seed1234_20260712_172332/syn_v10_report.html` | 单 seed、toy architecture、两种模式为独立模型 |
| v10 centroid geometry | v10 `analysis/report_stratified/tables/centroid_mean_geometry.csv` | class means 不能单独证明 within-class noise |
| v10 local ablation / head patching | v10 `position_local_ablation_by_bin.csv`、`nested_head_patching_regression_by_bin.csv` | 需要 matched random control；局部 transport 不等于完整算法 |
| v10 curved steering | v10 `analysis/geometry_path_steering/tables/geometry_path_steering_regression.csv` | 支持 local path，不支持统一 global \(+1\) direction |
| v15 position/trace confounds | `colab_results/v15_main_all_sequence_seed1234_20260718_171459/syn_v15_report.html`、`config.json`、`manifest.json` | 多因素同时变化，不能作为最终机制 setting |
| v20 final free-running accuracy | `colab_results/v20_main_RoPE_count1-30_seed1234/tables/final_autoregressive_summary.csv` | one seed；thinking .912、non-thinking .335；未 capability-match |
| v20 timeline | v20 `analysis/mechanism_report_assets/mechanism_timeline.png`、phase-transition tables | step 1500 objective switch；无 multi-seed/no-switch control 前不称 phase transition |
| v20 causal-stage completeness | v20 `manifest.json`、`analysis/v10_port/manifest.json` | 部分 stage 失败或为 repair artifact，必须完整重跑并记录 provenance |

## Public references

公开论文元数据在 `references.bib`。Qwen3 和 Gemma4 的模型描述分别使用其 technical report。Gemma4-E4B 的参数口径应始终写为约 4.5B effective/non-embedding、8B including embeddings，避免把它错误放入 dense 4B scaling 点。

## 核心因果链的证据门槛

| 结论 | 当前状态 | 升级所需证据 |
|---|---|---|
| trace-generating modes 在 core regime 通常明显优于 Direct | Preliminary behavior | `B0` paired matched-policy rerun |
| Direct 使用 broad-source route 并形成 terminal total \(Z_D\) | Hypothesis + synthetic/older realistic motivation | `H1-D` 多 source causal fan-in、\(Z_D\) damage/recovery、complete-count recovery |
| CoT 使用 matched-source retrieval 和 next-source/stop control | Synthetic motivation | `H1-C/H2` identity retargeting、bidirectional continue/stop patching、free-running effect |
| 模型有可执行 counter state | 未在目标 LLM 设置建立 | `R1–R3` grouped holdout、local/natural transport、new-valid \(+1\)、duplicate/invalid no-op |
| targeted retrieval 通过降低 functional noise 提高准确率 | 未建立 | `M1` route→identity/coverage→\(Z_C\)→answer，clean state restoration |
| targeted retrieval 在 coverage 之外改善 aggregation noise | 未建立 | `M2` fixed-\(U\) 条件下 \(Z_C\) SNR 仍受 route 干预且可恢复 |
| 上述链条解释 CoT–Direct gap | 未建立 | `M3` gap removal、null-compute controls、Qwen3-8B/Gemma4-E4B replication |

## 审计规则

1. 正文每个精确数字必须回溯到冻结表格和 manifest，而不只回溯到 HTML prose。
2. PCA 只作可视化；noise 结论依赖 full-space spacing、conditional covariance、decoder error 和 causal transport。
3. Attention score 只用于 discovery；功能命名要求独立 reporting split 的 query-local damage、clean restoration 和 semantic retargeting。
4. Teacher-forced 结果只说明 local competence；自然执行必须在 free-running first-error 之前复现方向。
5. Synthetic 结果不写成 pretrained LLM 已证实机制。
6. empirical law 必须预测 untouched Qwen3-14B；失败时明确保留为 qualitative response surface。
