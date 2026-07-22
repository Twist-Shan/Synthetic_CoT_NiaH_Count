# Claim-to-source map

This file records where the paper's quantitative claims came from. Paths are absolute so the draft can be audited independently of the LaTeX folder.

## Primary narrative reports

- Synthetic v10 narrative and two-route development:  
  `C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Synthetic_NiaH_like_Count\colab_results\v10_main_seed1234_20260712_172332\syn_v10_report.html`
- Synthetic v16.2 RoPE causal report:  
  `C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Synthetic_NiaH_like_Count\colab_results\v16_2_main_rope_seed1234\v16_2_full_causal_report.html`
- Synthetic v16.3 query-position comparison:  
  `C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Synthetic_NiaH_like_Count\colab_results\v16_3_main_data-query_seed1234_20260721\v16_2_vs_v16_3_query_order_report.html`
- Realistic 4K report:  
  `C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Realistic_CoT_NiaH_Count\reports\NIAH-4K-report-standalone.html`
- Earlier realistic CoT mechanism report:  
  `C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Realistic_CoT_NiaH_Count\reports\NIAH-counting.html`

## Main figure

`figures/mechanism_routes.png` is the first embedded figure in the v16.2 causal report. It is the requested two-hypothesis route map and is used as Figure 1 without altering its scientific content.

## Controlled synthetic claims

- Architecture, data, training, and loss schedule: `v16_2_main_rope_seed1234/config.json`, `tables/model_specifications.csv`.
- Final independent behavior and permutation robustness: v16.2 report Section 4.1; detailed rows in `tables/autoregressive_detail.csv` and `tables/prefix_permutation_consistency.csv`.
- Learning dynamics: `tables/checkpoint_dynamics_autoregressive.csv`, `tables/mechanism_emergence_milestones.csv`, and `figures/checkpoint_mechanism_overview.png`.
- Count-state geometry: `tables/checkpoint_state_probe_summary.csv`, `analysis/v10_port/tables/representation_geometry.csv`, and `analysis/v10_port/tables/residual_count_transport.csv`.
- Query-local necessity: `analysis/v10_port/tables/position_local_head_ablation.csv`.
- Targeted retrieval sufficiency: `analysis/v10_port/tables/retrieval_head_patching.csv`.
- Successor/stop and MLP conversion: `successor_head_patching.csv`, `successor_mlp_feature_concentration.csv`, `successor_mlp_feature_patching.csv`.
- Final-query donor transport: `analysis/v10_port/tables/final_query_head_transport.csv`.
- Trace-span conflict and final bridge: `length_preserving_trace_conflicts.csv`, `final_bridge_component_patching.csv`.
- Position-matched early stop: `trace_early_stop_patching.csv`.
- Head-state bidirectionality: `head_to_state_geometry.csv`, `state_to_head_routing.csv`.

## Query-position claims

All compact comparison tables are under:

`C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Synthetic_NiaH_like_Count\colab_results\v16_3_main_data-query_seed1234_20260721\analysis\v16_2_vs_v16_3\tables`

Specific mappings:

- 84% -> 96% paired AR gain and CI: `paired_final_behavior.csv`.
- Accuracy AULC and persistent thresholds: `checkpoint_learning_summary.csv`.
- 81% -> 98% exact trace: `thinking_trace_final_metrics.csv`.
- L2 `R^2` 0.550 -> 0.978: `representation_summary.csv`.
- Direct coverage 0.842 -> 0.074 and query mass 0.963: `attention_route_summary.csv`.
- Matched data/initialization audit: `setting_and_identity_audit.csv`.

## Realistic claims

The exact run-level data are summarized and linked from the 4K standalone report. Key sections are:

- Retrieval/count dissociation: Sections 2--3.
- Enumeration effects and numbered enumeration: Section 4.
- Direct parallel proportion code, dilution, head ablation, and steering: Section 7.
- Earlier broad-versus-targeted CoT attention and probe results: `NIAH-counting.html`, sections "Counting without CoT" and "Counting with CoT".

Because the realistic standalone report aggregates many run folders, any camera-ready table should be regenerated from the report's linked CSV/JSON artifacts rather than manually transcribed from plotted pixels.
