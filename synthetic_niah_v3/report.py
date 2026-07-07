from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd


def _image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _fig(path: Path, title: str, caption: str) -> str:
    if not path.exists():
        return ""
    return f"""
    <figure>
      <img src="{_image_uri(path)}" alt="{html.escape(title)}">
      <figcaption><b>{html.escape(title)}</b><br>{html.escape(caption)}</figcaption>
    </figure>
    """


def _table(path: Path, max_rows: int = 16) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "<p class='muted'>No table generated.</p>"
    return pd.read_csv(path).head(max_rows).to_html(index=False, escape=True)


def write_summary(run_dir: Path, cfg: dict, eval_df: pd.DataFrame, corrupt_df: pd.DataFrame, attention_df: pd.DataFrame) -> dict:
    generated = eval_df[eval_df["eval_mode"].isin(["direct", "generated_trace"])] if not eval_df.empty else pd.DataFrame()

    def acc_by_len(model_type: str) -> dict:
        if generated.empty:
            return {}
        sub = generated[generated["model_type"].eq(model_type)]
        if sub.empty:
            return {}
        return {str(k): float(v) for k, v in sub.groupby("seq_len_eval")["final_accuracy"].mean().to_dict().items()}

    trace = eval_df[(eval_df["model_type"].eq("thinking")) & (eval_df["eval_mode"].eq("generated_trace"))] if not eval_df.empty else pd.DataFrame()
    summary = {
        "run_name": run_dir.name,
        "preset": cfg["preset"],
        "train_seq_len": cfg["train_seq_len"],
        "seq_lens_eval": cfg["seq_lens_eval"],
        "count_range": [1, 10],
        "seeds": cfg["seeds"],
        "non_thinking_final_accuracy_by_len": acc_by_len("non_thinking"),
        "thinking_final_accuracy_by_len": acc_by_len("thinking"),
        "thinking_trace_exact_by_len": {
            str(k): float(v) for k, v in trace.groupby("seq_len_eval")["trace_exact_rate"].mean().to_dict().items()
        }
        if not trace.empty
        else {},
        "round1_main_takeaway": "Round 1 isolates length/noise generalization at fixed count range 1..10.",
        "round2_main_takeaway": "Round 2 checks whether thinking final answers follow prompt count or corrupted trace-derived shortcuts.",
        "round3_main_takeaway": "Round 3 separates probe/attention diagnostics from causal single-head ablation evidence.",
        "limitations": [
            "All data are symbolic.",
            "Counts are limited to 1..10.",
            "The trace exposes count length, so final readout may exploit trace length or last-index shortcuts.",
            "Probe decodability is not causal evidence.",
            "Attention patterns are not causal unless ablation or masking changes behavior.",
            "There is no loss-mask ablation in this version by design.",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_report(run_dir: Path, cfg: dict, summary: dict) -> Path:
    figures = run_dir / "figures"
    tables = run_dir / "tables"
    css = """
    body { margin:0; background:#f5f7fb; color:#172033; font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; line-height:1.55; }
    .shell { max-width:1180px; margin:0 auto; padding:24px 20px 56px; }
    header { background:linear-gradient(135deg,#0f172a,#2563eb); color:#fff; border-radius:14px; padding:28px 32px; }
    header h1 { margin:0 0 8px; font-size:2.25rem; }
    section { background:#fff; border:1px solid #dbe3ef; border-radius:12px; padding:20px; margin-top:20px; box-shadow:0 8px 22px rgba(15,23,42,.05); }
    h2 { margin-top:0; }
    h3 { margin-bottom:4px; }
    code { background:#eef2ff; color:#1e3a8a; border-radius:5px; padding:1px 5px; }
    table { border-collapse:collapse; width:100%; font-size:.86rem; display:block; overflow:auto; }
    th,td { border-bottom:1px solid #dbe3ef; padding:6px 7px; text-align:left; white-space:nowrap; }
    th { background:#f8fafc; }
    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; align-items:start; }
    figure { margin:0; border:1px solid #dbe3ef; border-radius:10px; overflow:hidden; background:#fff; }
    img { width:100%; max-height:390px; object-fit:contain; display:block; background:white; }
    figcaption { border-top:1px solid #dbe3ef; padding:9px 11px; color:#475569; font-size:.92rem; }
    .muted { color:#64748b; }
    @media (max-width:850px) { .grid { grid-template-columns:1fr; } header h1 { font-size:1.8rem; } }
    """
    fig_specs = {
        "round1": [
            ("round1_train_loss_by_step.png", "Training loss", "x=training step; y=masked next-token CE. Only the fixed objective for each model is trained."),
            ("round1_final_accuracy_by_step_and_seq_len.png", "Hard length eval", "x=checkpoint step; y=final count accuracy. Lines compare model type and eval sequence length."),
            ("round1_accuracy_by_count_final.png", "Final accuracy by exact count", "x=gold count 1..10; y=accuracy at the final checkpoint."),
            ("round1_accuracy_heatmap_count_x_seq_len.png", "Length generalization heatmap", "Rows are model types; columns are eval lengths."),
            ("round1_trace_metrics_by_seq_len.png", "Thinking trace quality", "Trace exact, marker recall, and invalid-generation rate are separate from final count accuracy."),
            ("round1_oracle_vs_generated_trace_accuracy.png", "Oracle vs generated trace", "Oracle trace tests whether final readout works when retrieval trace is supplied correctly."),
        ],
        "round2": [
            ("round2_corruption_accuracy_by_type.png", "Corruption accuracy", "x=corruption type; y=whether answer still equals prompt count."),
            ("round2_follow_rule_breakdown.png", "Follow-rule breakdown", "Shows if predictions follow prompt count, trace pair count, last index, max index, or marker count."),
            ("round2_confusion_pred_vs_prompt_count.png", "Predicted vs prompt count", "Rows are true prompt count; columns are predicted count."),
            ("round2_confusion_pred_vs_trace_pair_count.png", "Predicted vs trace pair count", "Detects trace-length shortcuts."),
            ("round2_confusion_pred_vs_last_index.png", "Predicted vs last index", "Detects last-index shortcuts."),
            ("round2_corruption_by_seq_len.png", "Corruption robustness by length", "x=eval sequence length; y=prompt-count accuracy under each corruption."),
        ],
        "round3": [
            ("round3_probe_accuracy_layer_by_anchor.png", "Probe accuracy", "x=anchor; y=count probe accuracy. Index-token anchors are marked leakage-prone in the CSV."),
            ("round3_probe_r2_layer_by_anchor.png", "Probe R2", "Ridge regression R2 for numeric count decoding."),
            ("round3_probe_vs_position_baseline.png", "Probe vs position baseline", "Points above diagonal exceed a position-only baseline."),
            ("round3_attention_head_leaderboard.png", "Attention leaderboard", "Ranks heads by retrieval-like metrics; this is diagnostic only."),
            ("round3_thinking_trace_to_prompt_heatmap_best_head.png", "Thinking retrieval heatmap", "Layer/head matrix of trace-to-prompt correct top-1 retrieval."),
            ("round3_nonthinking_ans_to_prompt_attention.png", "Non-thinking retrieval heatmap", "Layer/head matrix of <Ans>-to-prompt top-n retrieval."),
            ("round3_attention_metrics_by_count_bin.png", "Attention by count bin", "Compares attention mass to prompt needles for low/mid/high counts."),
            ("round3_attention_metrics_by_seq_len.png", "Attention by length", "Checks whether retrieval geometry degrades at 512 or 1024."),
            ("round3_head_ablation_effects.png", "Head ablation: final answer", "Negative delta means the ablated head hurt final count accuracy."),
            ("round3_attention_masking_effects.png", "Head ablation: trace", "Negative delta means the ablated head hurt trace exactness. Targeted masking is TODO unless implemented later."),
        ],
    }

    def fig_grid(name: str) -> str:
        return "<div class='grid'>" + "".join(_fig(figures / f, title, cap) for f, title, cap in fig_specs[name]) + "</div>"

    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic NIAH Counting v3 No-Loss Report</title><style>{css}</style></head>
<body><div class="shell">
<header><h1>Synthetic NIAH Counting v3: No Loss Ablation</h1>
<p>Run <code>{html.escape(run_dir.name)}</code>; preset <code>{html.escape(cfg['preset'])}</code>; train length <code>{cfg['train_seq_len']}</code>; eval lengths <code>{cfg['seq_lens_eval']}</code>; seeds <code>{cfg['seeds']}</code>.</p></header>
<section><h2>Configuration</h2><p>v3 trains exactly two model types per seed: <code>non_thinking</code> and <code>thinking</code>. It does not run full-LM/final-heavy/trace-only loss sweeps.</p>{pd.DataFrame([{'field': k, 'value': str(v)} for k, v in cfg.items()]).to_html(index=False, escape=True)}</section>
<section><h2>Interpretation Summary</h2>
<h3>Behavioral evidence</h3><p>{html.escape(summary['round1_main_takeaway'])}</p>
<h3>Trace-generation evidence</h3><p>For thinking, generated-trace accuracy and trace exactness are reported separately from oracle-trace final readout.</p>
<h3>Corrupted-trace evidence</h3><p>{html.escape(summary['round2_main_takeaway'])}</p>
<h3>Probe evidence</h3><p>Probe results should be read against position-only and trace-length-only baselines; probe decodability is not causal evidence.</p>
<h3>Attention evidence</h3><p>Attention retrieval metrics can suggest targeted retrieval, but they are diagnostic until paired with interventions.</p>
<h3>Causal-intervention evidence</h3><p>{html.escape(summary['round3_main_takeaway'])}</p></section>
<section><h2>Round 1: Hard Length/Noise Evaluation</h2>{_table(tables / 'round1_step_to_thresholds.csv')}{fig_grid('round1')}</section>
<section><h2>Round 2: Corrupted-Trace Diagnostics</h2>{_table(tables / 'round2_follow_rule_summary.csv')}{fig_grid('round2')}</section>
<section><h2>Round 3: Probes, Attention, and Causal Tests</h2>{_table(tables / 'round3_attention_head_metrics.csv')}{fig_grid('round3')}</section>
<section><h2>Limitations</h2><ul>{''.join(f'<li>{html.escape(item)}</li>' for item in summary['limitations'])}</ul></section>
</div></body></html>"""
    out = run_dir / "syn_v3_no_loss_report.html"
    out.write_text(html_text, encoding="utf-8")
    return out
