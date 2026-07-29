# Claim-to-source map

本文件用于限制证据边界。`Preliminary` 表示可以在草稿中报告但尚未完成目标设置复现；`Hypothesis/TODO` 表示不能写成已证实结论。

## Realistic NIAH

| 草稿内容 | 主要本地来源 | 状态与注意事项 |
|---|---|---|
| 26,500 requests；53 个 model--mode cells；$N$、$L$、seed 设计 | `C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Realistic_CoT_NiaH_Count\reports\v2_realistic_niah_analysis_20260727\README.md`；同目录 `analysis_manifest.json` | Completed behavior audit；机制比较仍受 decoding/checkpoint 差异影响 |
| Table 3 的 Qwen/Gemma registered success 与 Gemma numerical exact | `...\v2_realistic_niah_analysis_20260727\tables\prompt\model_mode_summary.csv` | Preliminary behavior；strict success 必须与 numerical exact、format、parse、truncation 分开 |
| Index vs Bullet：13 个 paired models 中 10 个 Index 更高 | `...\tables\prompt\paired_index_bullet.csv` | Descriptive paired result；不能由此单独推出内部机制 |
| Native trace style；50.3% 无 city-bearing list | `...\tables\prompt\native_list_style_exclusive_summary.csv`；`native_list_style_inclusive_summary.csv`；`native_no_full_style_summary.csv` | Post-treatment/self-selected trace style，仅作描述 |
| Direct/Native count--length response surfaces | `...\tables\focused_empirical_law\selected_mode_laws.csv`；`mode_candidate_summary.csv`；`...\tables\bias_multivariable\consensus_formula_by_mode_target.csv` | Exploratory；不称为 universal scaling law，加入模型前先冻结 basis/split |
| 较早 realistic thinking probes/patching | `C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Realistic_CoT_NiaH_Count\reports\NIAH-counting.html` | Motivating only；样本小，且 HTML 正文/表格有需复核之处 |
| 较早 4K non-thinking：高 recall、broad heads、final count code、steering、compressed manifold | `C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Realistic_CoT_NiaH_Count\reports\NIAH-4K-report-standalone.html` | Preliminary mechanism evidence；必须在 V2 Qwen/Gemma matched setting 复现 |

上表中的 `...` 均指第一行所列 `v2_realistic_niah_analysis_20260727` 目录。

## Synthetic bridge

| 草稿内容 | 主要本地来源 | 状态与注意事项 |
|---|---|---|
| v10 representation、routing、patching、ablation、steering、transplant | `C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Synthetic_NiaH_like_Count\colab_results\v10_main_seed1234_20260712_172332\syn_v10_report.html` | Strong controlled motivation，但仅单 seed、toy architecture、两种模式为独立模型 |
| v10 centroid geometry | `...\analysis\report_stratified\tables\centroid_mean_geometry.csv` | 只描述 class means；不能据此宣称 within-class noise |
| v10 query-local ablation | `...\analysis\report_stratified\tables\position_local_ablation_by_bin.csv` | 必须与 matched random heads 一起解释 |
| v10 nested head-output patch | `...\analysis\report_stratified\tables\nested_head_patching_regression_by_bin.csv` | 支持局部 transport；不等于完整算法证明 |
| v10 curved-manifold steering | `...\analysis\geometry_path_steering\tables\geometry_path_steering_regression.csv` | 支持 local path，不支持统一 global $+1$ direction |
| v10 progress/total dissociation | `...\analysis\hidden_state_patching\tables\misaligned_trace_rollout_factor_summary.csv`；`...\analysis\head_state_bidirectional\tables\*.csv` | Teacher-forced grid 与稀疏 free rollout 必须分开报告 |
| v15 all-sequence/RoPE/RPE candidate setting | `C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Synthetic_NiaH_like_Count\colab_results\v15_main_all_sequence_seed1234_20260718_171459\syn_v15_report.html`；同目录 `config.json`、`manifest.json` | 多因素同时改变；state probe 受 position/$k$ confounding，不能作最终机制结论 |
| v20 final free-running accuracy | `C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Synthetic_NiaH_like_Count\colab_results\v20_main_RoPE_count1-30_seed1234\tables\final_autoregressive_summary.csv` | One seed；thinking .912、non-thinking .335；尚未 capability matched |
| v20 training timeline | `...\analysis\mechanism_report_assets\mechanism_timeline.png`；`...\analysis\phase_transition_audit\tables\high_power_ar_summary.csv` | Objective 在 step 1500 切换；未经 no-switch/multi-seed control 不称为 phase transition |
| v20 causal-stage completeness | `...\manifest.json`；`...\analysis\v10_port\manifest.json` | Partial/failed stages 需要修复与完整重跑 |

Synthetic 表中的 `...` 指该行最近列出的 run 目录。

## Public references

公开论文元数据集中在 `references.bib`。模型能力陈述以 Qwen3 technical report（arXiv:2505.09388）与 Gemma 4 technical report（arXiv:2607.02770）为准。所有机制性结论仍以本地实验及其可复现实验表为依据，而不是由 technical report 推断。

## 审计规则

1. 正文每个精确数字应能回溯到 frozen CSV，而不是只回溯到 HTML prose。
2. Attention score 只用于发现候选；功能命名还要求跨 split 复现和 query-local causal specificity。
3. Probe 结论必须注明 anchor、label、held-out groups、position/token controls 与是否包含错误样本。
4. Synthetic 证据不得写成 pretrained LLM 已证实机制。
5. `count code`、`prefix counter`、`progress register` 与 `final-total register` 不得互换使用。

