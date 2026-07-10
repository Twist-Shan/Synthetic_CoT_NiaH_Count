# Experiment Specifications

These documents record the design requirements that produced the executable notebooks.
The notebooks remain the canonical runnable entry points; the specifications explain
the intended controls, metrics, and mechanistic questions.

| Version | Specification | Main question |
| --- | --- | --- |
| v0 | `synthetic_experiments_pipeline.md` | Loss-mask effects in trace-enumeration counting |
| v2 | `pipeline_v2_codex_prompt.md` | Controlled thinking vs non-thinking counting |
| v3 | `pipeline_v3_codex_prompt.md` | Attention-head ablation and total-token attention |
| v3.2 | `pipeline_v3_2_causal_tests_codex_prompt.md` | Causal tests of retrieval heads and residual paths |
| v4 | `pipeline_v4_steering_codex_prompt.md` | Count directions, steering, and activation patching |
| v5 | `pipeline_v5_mixed_thinking_toggle_codex_prompt.md` | One model with explicit thinking-mode control |
| v6 | `pipeline_v6_separator_trace_codex_prompt.md` | Separator traces without numeric index leakage |

Later experiments are documented directly in their notebook headings and configuration
cells because they are compact extensions of the v2 setup.
