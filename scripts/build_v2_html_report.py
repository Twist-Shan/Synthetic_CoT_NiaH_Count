from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def fmt(value: object, digits: int = 3) -> str:
    x = to_float(value)
    if math.isnan(x):
        return ""
    if abs(x) < 0.0005:
        return "0"
    return f"{x:.{digits}f}"


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def table(rows: list[dict], columns: list[tuple[str, str]], max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    head = "".join(f"<th>{esc(label)}</th>" for _, label in columns)
    body_rows = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = fmt(value)
            cells.append(f"<td>{esc(value)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    if not body_rows:
        body_rows.append(f"<tr><td colspan='{len(columns)}'>No rows found.</td></tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"


def group_mean(rows: list[dict[str, str]], keys: list[str], value_keys: list[str]) -> list[dict]:
    acc: dict[tuple, dict[str, float]] = {}
    counts: dict[tuple, int] = defaultdict(int)
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        if key not in acc:
            acc[key] = {k: 0.0 for k in value_keys}
        counts[key] += 1
        for value_key in value_keys:
            acc[key][value_key] += to_float(row.get(value_key), 0.0)
    out = []
    for key, sums in acc.items():
        item = {k: v for k, v in zip(keys, key)}
        n = counts[key]
        for value_key, total in sums.items():
            item[value_key] = total / max(n, 1)
        out.append(item)
    return out


def image_card(path: Path, title: str, caption: str) -> str:
    if not path.exists():
        return ""
    return f"""
    <figure class="figure-card">
      <img src="{image_data_uri(path)}" alt="{esc(title)}">
      <figcaption><strong>{esc(title)}</strong><br>{esc(caption)}</figcaption>
    </figure>
    """


def build_report(result_dir: Path, out_html: Path) -> None:
    run_dir = result_dir / "run"
    if not run_dir.exists():
        raise FileNotFoundError(f"Expected run directory under {result_dir}")

    manifest = read_json(result_dir / "manifest.json")
    config = read_json(run_dir / "config.json") or manifest.get("config", {})
    preset = manifest.get("preset", "unknown")

    by_count = read_csv(run_dir / "metrics_eval_by_count.csv")
    by_bin = read_csv(run_dir / "metrics_eval_by_bin.csv")
    train = read_csv(run_dir / "metrics_train.csv")
    probes = read_csv(run_dir / "probes" / "probe_metrics.csv")
    non_attn = read_csv(run_dir / "attention" / "attention_nonthinking_metrics.csv")
    think_attn = read_csv(run_dir / "attention" / "attention_thinking_metrics.csv")
    dynamics = read_csv(run_dir / "targeted_retrieval_deep_dive" / "h3_dynamics_pre_index_k_L2H0.csv")
    ablation = read_csv(run_dir / "targeted_retrieval_deep_dive" / "h3_thinking_head_ablation.csv")

    final_step = max((to_int(r.get("step")) for r in by_count), default=0)
    final_count = [r for r in by_count if to_int(r.get("step")) == final_step]
    final_bin = [r for r in by_bin if to_int(r.get("step")) == final_step]

    final_summary = []
    for model_type in sorted({r.get("model_type", "") for r in final_count}):
        rows = [r for r in final_count if r.get("model_type") == model_type]
        final_summary.append(
            {
                "model_type": model_type,
                "mean_accuracy": fmt(mean(to_float(r.get("accuracy"), 0.0) for r in rows)),
                "min_count_accuracy": fmt(min(to_float(r.get("accuracy"), 0.0) for r in rows)),
                "mean_final_loss": fmt(mean(to_float(r.get("eval_final_answer_loss"), 0.0) for r in rows), 4),
            }
        )

    count_rows = []
    for row in sorted(final_count, key=lambda r: (r.get("model_type", ""), to_int(r.get("count")))):
        count_rows.append(
            {
                "model_type": row.get("model_type"),
                "count": row.get("count"),
                "accuracy": fmt(row.get("accuracy")),
                "mae": fmt(row.get("mae")),
                "final_answer_loss": fmt(row.get("eval_final_answer_loss"), 4),
            }
        )

    bin_rows = []
    for row in sorted(final_bin, key=lambda r: (r.get("model_type", ""), r.get("count_bin", ""))):
        bin_rows.append(
            {
                "model_type": row.get("model_type"),
                "count_bin": row.get("count_bin"),
                "accuracy": fmt(row.get("accuracy")),
                "mae": fmt(row.get("mae")),
                "final_answer_loss": fmt(row.get("eval_final_answer_loss"), 4),
            }
        )

    last_train_step = max((to_int(r.get("step")) for r in train), default=0)
    train_rows = []
    for row in [r for r in train if to_int(r.get("step")) == last_train_step]:
        train_rows.append(
            {
                "model_type": row.get("model_type"),
                "train_loss": fmt(row.get("train_loss"), 4),
                "train_completion_loss": fmt(row.get("train_completion_loss"), 4),
                "train_final_answer_loss": fmt(row.get("train_final_answer_loss"), 4),
                "learning_rate": fmt(row.get("learning_rate"), 6),
            }
        )

    probe_best = sorted(probes, key=lambda r: to_float(r.get("probe_accuracy"), -1.0), reverse=True)[:12]
    probe_rows = [
        {
            "model_type": r.get("model_type"),
            "label_type": r.get("label_type"),
            "anchor_type": r.get("anchor_type"),
            "layer": r.get("layer"),
            "probe_accuracy": fmt(r.get("probe_accuracy")),
            "probe_mae": fmt(r.get("probe_mae")),
            "probe_r2": fmt(r.get("probe_r2")),
        }
        for r in probe_best
    ]

    thinking_head_rank = group_mean(
        think_attn,
        ["query_anchor", "layer", "head"],
        ["diagonal_dominance", "correct_top1_rate", "needle_attention_mass", "noise_attention_mass", "needle_vs_noise_ratio"],
    )
    thinking_head_rank.sort(
        key=lambda r: (
            to_float(r.get("correct_top1_rate"), -1.0),
            to_float(r.get("diagonal_dominance"), -1.0),
            to_float(r.get("needle_attention_mass"), -1.0),
        ),
        reverse=True,
    )
    thinking_head_rows = [
        {
            "query_anchor": r.get("query_anchor"),
            "layer": r.get("layer"),
            "head": r.get("head"),
            "correct_top1_rate": fmt(r.get("correct_top1_rate")),
            "diagonal_dominance": fmt(r.get("diagonal_dominance")),
            "needle_attention_mass": fmt(r.get("needle_attention_mass")),
            "noise_attention_mass": fmt(r.get("noise_attention_mass")),
        }
        for r in thinking_head_rank[:12]
    ]
    best_think = thinking_head_rank[0] if thinking_head_rank else {}

    non_head_rank = group_mean(
        non_attn,
        ["layer", "head"],
        ["top_n_retrieval_recall", "ans_to_all_needles_mass", "ans_to_noise_mass", "needle_vs_noise_ratio", "attention_entropy_over_prompt_body"],
    )
    non_head_rank.sort(
        key=lambda r: (
            to_float(r.get("top_n_retrieval_recall"), -1.0),
            to_float(r.get("ans_to_all_needles_mass"), -1.0),
        ),
        reverse=True,
    )
    non_head_rows = [
        {
            "layer": r.get("layer"),
            "head": r.get("head"),
            "top_n_retrieval_recall": fmt(r.get("top_n_retrieval_recall")),
            "ans_to_all_needles_mass": fmt(r.get("ans_to_all_needles_mass")),
            "ans_to_noise_mass": fmt(r.get("ans_to_noise_mass")),
            "attention_entropy": fmt(r.get("attention_entropy_over_prompt_body")),
        }
        for r in non_head_rank[:12]
    ]
    best_non = non_head_rank[0] if non_head_rank else {}

    ablation_rows = [
        {
            "condition": r.get("condition"),
            "accuracy": fmt(r.get("accuracy")),
            "invalid_rate": fmt(r.get("invalid_rate")),
            "trace_exact_match_rate": fmt(r.get("trace_exact_match_rate")),
            "trace_marker_recall": fmt(r.get("trace_marker_recall")),
            "trace_index_accuracy": fmt(r.get("trace_index_accuracy")),
        }
        for r in ablation
    ]

    dynamics_rows = [
        {
            "step": r.get("step"),
            "correct_top1_rate": fmt(r.get("correct_top1_rate")),
            "diagonal_dominance": fmt(r.get("diagonal_dominance")),
            "needle_attention_mass": fmt(r.get("needle_attention_mass")),
            "noise_attention_mass": fmt(r.get("noise_attention_mass")),
        }
        for r in dynamics
    ]

    setup_rows = [
        {"field": "preset", "value": preset},
        {"field": "saved_at", "value": manifest.get("saved_at", "")},
        {"field": "seq_len", "value": config.get("seq_len")},
        {"field": "train_steps", "value": config.get("train_steps")},
        {"field": "batch_size", "value": config.get("batch_size")},
        {"field": "eval_every", "value": config.get("eval_every")},
        {"field": "test_examples_per_count", "value": config.get("test_examples_per_count")},
        {"field": "attention_examples_per_count", "value": config.get("attention_examples_per_count")},
        {"field": "count_range", "value": f"{config.get('min_count')}..{config.get('max_count')}"},
        {"field": "noise_vocab_size", "value": config.get("noise_vocab_size")},
        {"field": "marker_vocab_size", "value": config.get("marker_vocab_size")},
        {"field": "model", "value": f"{config.get('n_layer')} layers, {config.get('n_head')} heads, d_model={config.get('n_embd')}"},
    ]

    plot_dir = run_dir / "plots"
    probe_dir = run_dir / "probes"
    attn_dir = run_dir / "attention"
    deep_dir = run_dir / "targeted_retrieval_deep_dive"

    figures_main = [
        (plot_dir / "train_loss_vs_step.png", "Training losses", "两个模型的训练 loss。debug run 只有 200 steps，所以主要看流程是否跑通。"),
        (plot_dir / "eval_final_answer_loss_vs_step.png", "Final-answer loss over steps", "测试集最后答案 token 的 cross-entropy。"),
        (plot_dir / "eval_accuracy_by_bin_vs_step.png", "Accuracy by count bin", "低/中/高 count bin 的 exact-count accuracy。"),
        (plot_dir / "final_accuracy_by_count.png", "Final accuracy by exact count", "最后 checkpoint 在 count=1..10 上的最终计数准确率。"),
        (plot_dir / "accuracy_heatmap_by_count_and_step_non_thinking.png", "Non-thinking accuracy heatmap", "横轴训练 step，纵轴 gold count，颜色为 accuracy。"),
        (plot_dir / "accuracy_heatmap_by_count_and_step_thinking.png", "Thinking accuracy heatmap", "横轴训练 step，纵轴 gold count，颜色为 accuracy。"),
    ]
    figures_probe = [
        (probe_dir / "probe_final_count_accuracy_heatmap_non_thinking.png", "Probe: non-thinking final count", "不同 layer/anchor 的 final-count probe accuracy。"),
        (probe_dir / "probe_final_count_accuracy_heatmap_thinking.png", "Probe: thinking final count", "注意 trace 结构会带来位置/格式泄漏，embedding 层 100% 不一定是机制证据。"),
        (probe_dir / "probe_prefix_count_accuracy_heatmap_thinking.png", "Probe: thinking prefix count accuracy", "trace prefix 的 count probe。"),
        (probe_dir / "probe_prefix_count_mae_heatmap_thinking.png", "Probe: thinking prefix count MAE", "trace prefix 的 count probe 绝对误差。"),
    ]
    figures_attention = [
        (attn_dir / "attention_thinking_correct_top1_by_layer_head.png", "Thinking attention correct top-1", "trace item k 是否最关注 prompt needle k。"),
        (attn_dir / "attention_thinking_diagonal_dominance_by_layer_head.png", "Thinking attention diagonal dominance", "trace-to-needle 注意力矩阵的对角占优程度。"),
        (attn_dir / "attention_matrix_thinking_best_head_low.png", "Thinking best head matrix: low", "低 count 样本平均 attention matrix。"),
        (attn_dir / "attention_matrix_thinking_best_head_mid.png", "Thinking best head matrix: mid", "中 count 样本平均 attention matrix。"),
        (attn_dir / "attention_matrix_thinking_best_head_high.png", "Thinking best head matrix: high", "高 count 样本平均 attention matrix。"),
        (attn_dir / "attention_nonthinking_topn_recall_by_layer_head.png", "Non-thinking top-n recall", "non-thinking <Ans> 的 top-n prompt positions 中有多少是真 needle。"),
        (attn_dir / "attention_nonthinking_ans_needle_mass_by_layer_head.png", "Non-thinking needle mass", "non-thinking <Ans> 对所有 needle 的 attention mass。"),
    ]
    figures_deep = [
        (deep_dir / "h1_thinking_correct_top1_by_head.png", "H1 correct top-1 by head", "targeted-retrieval head ranking。"),
        (deep_dir / "h1_thinking_diagonal_dominance_by_head.png", "H1 diagonal dominance by head", "对角占优越高，越像第 k 个 trace item 对准第 k 个 needle。"),
        (deep_dir / "h1_thinking_needle_mass_by_head.png", "H1 needle mass by head", "raw attention mass to true prompt needles。"),
        (deep_dir / "h1_matrix_pre_index_k_L2H0_low.png", "H1 matrix low", "debug run 中最强 head 的 low-count matrix。"),
        (deep_dir / "h1_matrix_pre_index_k_L2H0_mid.png", "H1 matrix mid", "debug run 中最强 head 的 mid-count matrix。"),
        (deep_dir / "h1_matrix_pre_index_k_L2H0_high.png", "H1 matrix high", "debug run 中最强 head 的 high-count matrix。"),
        (deep_dir / "h2_broad_vs_targeted_summary.png", "H2 broad vs targeted", "non-thinking broad aggregation 与 thinking targeted retrieval 的 summary comparison。"),
        (deep_dir / "h2_nonthinking_topn_recall.png", "H2 non-thinking top-n recall", "non-thinking <Ans> retrieval heads。"),
        (deep_dir / "h2_nonthinking_entropy.png", "H2 non-thinking entropy", "prompt-body attention entropy。"),
        (deep_dir / "h3_targeted_head_training_dynamics.png", "H3 training dynamics", "targeted head 指标随 checkpoint 变化。"),
        (deep_dir / "h3_thinking_head_ablation.png", "H3 head ablation", "final thinking model 中 ablate target/control head 后的行为变化。"),
    ]

    caveat = ""
    if str(preset).lower() == "debug":
        caveat = """
        <div class="callout warn">
          <strong>重要 caveat:</strong> 这是一份 <code>debug</code> run：
          seq_len=64、train_steps=200、每个 count 的测试样本较少。它适合检查 pipeline 和机制分析是否能跑通，
          但不适合作为 paper-quality 结论。真正结论应该优先使用 <code>main</code> run。
        </div>
        """

    best_think_text = ""
    if best_think:
        best_think_text = (
            f"当前 debug run 中，thinking 最强 targeted-retrieval candidate 是 "
            f"<code>{esc(best_think.get('query_anchor'))}</code>, layer <code>{esc(best_think.get('layer'))}</code>, "
            f"head <code>{esc(best_think.get('head'))}</code>；"
            f"correct-top1={fmt(best_think.get('correct_top1_rate'))}, "
            f"diagonal-dominance={fmt(best_think.get('diagonal_dominance'))}, "
            f"needle-mass={fmt(best_think.get('needle_attention_mass'))}。"
        )
    best_non_text = ""
    if best_non:
        best_non_text = (
            f"non-thinking 最强 <code>&lt;Ans&gt;</code> retrieval head 是 layer "
            f"<code>{esc(best_non.get('layer'))}</code>, head <code>{esc(best_non.get('head'))}</code>；"
            f"top-n recall={fmt(best_non.get('top_n_retrieval_recall'))}, "
            f"needle-mass={fmt(best_non.get('ans_to_all_needles_mass'))}, "
            f"entropy={fmt(best_non.get('attention_entropy_over_prompt_body'))}。"
        )

    css = """
    :root { --bg:#f6f7fb; --paper:#fff; --ink:#172033; --muted:#64748b; --line:#dbe3ef; --blue:#2563eb; --green:#16a34a; --amber:#d97706; --red:#dc2626; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height:1.6; }
    .shell { max-width: 1180px; margin: 0 auto; padding: 32px 24px 56px; }
    .hero { background: linear-gradient(135deg, #111827, #1d4ed8); color:white; border-radius: 18px; padding: 34px 38px; box-shadow: 0 18px 60px rgba(15,23,42,.16); }
    .hero h1 { margin:0 0 8px; font-size: clamp(2rem, 4vw, 3.5rem); line-height:1.05; letter-spacing:-.03em; }
    .hero p { margin:0; max-width:850px; color:rgba(255,255,255,.85); }
    .meta { display:flex; flex-wrap:wrap; gap:10px; margin-top:18px; }
    .pill { display:inline-flex; align-items:center; gap:6px; padding:5px 10px; border-radius:999px; background:rgba(255,255,255,.12); color:white; font-size:.9rem; }
    section { background:var(--paper); border:1px solid var(--line); border-radius:14px; padding:24px; margin-top:22px; box-shadow: 0 10px 30px rgba(15,23,42,.06); }
    h2 { margin:0 0 12px; font-size:1.55rem; letter-spacing:-.01em; }
    h3 { margin:20px 0 8px; font-size:1.1rem; }
    p { margin: 8px 0 14px; }
    code { background:#eef2ff; color:#1e3a8a; border-radius:5px; padding:1px 5px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap:16px; }
    .kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin-top:16px; }
    .kpi { border:1px solid var(--line); border-radius:12px; padding:14px; background:#f8fafc; }
    .kpi .label { color:var(--muted); font-size:.82rem; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
    .kpi .value { font-size:1.65rem; font-weight:800; margin-top:4px; }
    .callout { border-left:5px solid var(--blue); background:#eff6ff; padding:13px 15px; border-radius:10px; margin:14px 0; }
    .callout.warn { border-left-color:var(--amber); background:#fffbeb; }
    .callout.good { border-left-color:var(--green); background:#f0fdf4; }
    .callout.risk { border-left-color:var(--red); background:#fef2f2; }
    .table-wrap { overflow:auto; border:1px solid var(--line); border-radius:10px; margin:12px 0 18px; }
    table { width:100%; border-collapse:collapse; font-size:.92rem; }
    th, td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
    th { background:#f8fafc; color:#334155; font-weight:800; }
    tr:last-child td { border-bottom:0; }
    .figure-card { margin:0; border:1px solid var(--line); border-radius:12px; overflow:hidden; background:white; }
    .figure-card img { display:block; width:100%; height:auto; background:white; }
    .figure-card figcaption { padding:11px 13px; color:#475569; font-size:.9rem; border-top:1px solid var(--line); }
    .small { color:var(--muted); font-size:.9rem; }
    .toc { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
    .toc a { color:white; text-decoration:none; background:rgba(255,255,255,.13); border-radius:999px; padding:6px 10px; font-size:.9rem; }
    ul { padding-left: 20px; }
    """

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthetic NIAH Counting v2 Report</title>
  <style>{css}</style>
</head>
<body>
<div class="shell">
  <header class="hero">
    <h1>Synthetic NIAH Counting v2 Report</h1>
    <p>固定长度 symbolic NIAH counting 实验报告：两个模型（non-thinking / thinking）、final-count accuracy、probe、attention、targeted retrieval deep dive。</p>
    <div class="meta">
      <span class="pill">preset: {esc(preset)}</span>
      <span class="pill">seq_len: {esc(config.get("seq_len"))}</span>
      <span class="pill">steps: {esc(config.get("train_steps"))}</span>
      <span class="pill">seed: {esc(config.get("seed"))}</span>
      <span class="pill">saved: {esc(manifest.get("saved_at", ""))}</span>
    </div>
    <nav class="toc">
      <a href="#summary">Summary</a>
      <a href="#setup">Setup</a>
      <a href="#behavior">Behavior</a>
      <a href="#probe">Probe</a>
      <a href="#attention">Attention</a>
      <a href="#deep">Deep Dive</a>
      <a href="#takeaways">Takeaways</a>
    </nav>
  </header>

  {caveat}

  <section id="summary">
    <h2>Executive Summary</h2>
    <div class="callout good">
      <strong>行为结果：</strong> final step = <code>{final_step}</code>。
      两个模型在 debug setting 下整体已经基本学会 final count；高 count 仍有少量误差，因此比 main run 更能看到未完全饱和时的差异。
    </div>
    <div class="callout">
      <strong>机制亮点：</strong> {best_think_text}
      这与 NIAH 文档里的 “CoT token targeted retrieval 到对应 prompt needle” 假设相当接近。
    </div>
    <div class="callout">
      <strong>对照：</strong> {best_non_text}
      non-thinking 也有 retrieval-like head，但它发生在 final-answer query 上，更像一次性聚合而非 trace-indexed sequential retrieval。
    </div>
    <div class="kpi-grid">
      {''.join(f"<div class='kpi'><div class='label'>{esc(r['model_type'])} mean accuracy</div><div class='value'>{esc(r['mean_accuracy'])}</div><div class='small'>min by count: {esc(r['min_count_accuracy'])}</div></div>" for r in final_summary)}
    </div>
  </section>

  <section id="setup">
    <h2>Experiment Setting</h2>
    <p>模型输入为 symbolic haystack：noise tokens 与 countable marker tokens 混合。non-thinking 直接输出 <code>&lt;Ans&gt; count</code>；thinking 先生成显式 trace，再输出最终 count。</p>
    {table(setup_rows, [("field", "field"), ("value", "value")])}
  </section>

  <section id="behavior">
    <h2>Behavior: Final Count Accuracy</h2>
    <p>准确率只看最终 count 是否匹配 gold count；thinking 的 trace exact/marker 指标另算，不混进 final count accuracy。</p>
    <h3>Final bin summary</h3>
    {table(bin_rows, [("model_type", "model"), ("count_bin", "bin"), ("accuracy", "accuracy"), ("mae", "MAE"), ("final_answer_loss", "final-answer loss")])}
    <h3>Final exact-count summary</h3>
    {table(count_rows, [("model_type", "model"), ("count", "gold count"), ("accuracy", "accuracy"), ("mae", "MAE"), ("final_answer_loss", "final-answer loss")])}
    <h3>Last training rows</h3>
    {table(train_rows, [("model_type", "model"), ("train_loss", "train loss"), ("train_completion_loss", "completion loss"), ("train_final_answer_loss", "final-answer loss"), ("learning_rate", "lr")])}
    <div class="grid">
      {''.join(image_card(p, t, c) for p, t, c in figures_main)}
    </div>
  </section>

  <section id="probe">
    <h2>Probe Results</h2>
    <p>Probe 衡量 hidden state 是否线性可读出 count。需要注意：thinking trace 的某些位置天然携带 count/position 信息，所以 embedding-level 完美 probe 不能直接等同于 causal counter。</p>
    {table(probe_rows, [("model_type", "model"), ("label_type", "label"), ("anchor_type", "anchor"), ("layer", "layer"), ("probe_accuracy", "probe acc"), ("probe_mae", "MAE"), ("probe_r2", "R2")])}
    <div class="grid">
      {''.join(image_card(p, t, c) for p, t, c in figures_probe)}
    </div>
  </section>

  <section id="attention">
    <h2>Attention and Retrieval</h2>
    <p>Attention 是 diagnostic evidence，不是单独的 causal proof。这里最关心的是 thinking trace item <code>k</code> 是否对齐到 prompt needle <code>k</code>。</p>
    <h3>Thinking head ranking</h3>
    {table(thinking_head_rows, [("query_anchor", "query anchor"), ("layer", "layer"), ("head", "head"), ("correct_top1_rate", "correct top-1"), ("diagonal_dominance", "diagonal dominance"), ("needle_attention_mass", "needle mass"), ("noise_attention_mass", "noise mass")])}
    <h3>Non-thinking head ranking</h3>
    {table(non_head_rows, [("layer", "layer"), ("head", "head"), ("top_n_retrieval_recall", "top-n recall"), ("ans_to_all_needles_mass", "needle mass"), ("ans_to_noise_mass", "noise mass"), ("attention_entropy", "entropy")])}
    <div class="grid">
      {''.join(image_card(p, t, c) for p, t, c in figures_attention)}
    </div>
  </section>

  <section id="deep">
    <h2>Targeted Retrieval Deep Dive</h2>
    <p>这部分对应三个机制假设：H1 targeted head 是否存在；H2 non-thinking 是否更 broad；H3 target head 是否随训练形成并对行为有影响。</p>
    <h3>H3 dynamics</h3>
    {table(dynamics_rows, [("step", "step"), ("correct_top1_rate", "correct top-1"), ("diagonal_dominance", "diagonal dominance"), ("needle_attention_mass", "needle mass"), ("noise_attention_mass", "noise mass")])}
    <h3>H3 head ablation</h3>
    {table(ablation_rows, [("condition", "condition"), ("accuracy", "accuracy"), ("invalid_rate", "invalid"), ("trace_exact_match_rate", "trace exact"), ("trace_marker_recall", "marker recall"), ("trace_index_accuracy", "index acc")])}
    <div class="grid">
      {''.join(image_card(p, t, c) for p, t, c in figures_deep)}
    </div>
  </section>

  <section id="takeaways">
    <h2>Takeaways and Next Steps</h2>
    <ul>
      <li><strong>当前 debug 结果：</strong> 两个模型都基本学会 final counting；这说明 pipeline 跑通，但不能作为 “thinking 提升最终准确率” 的强证据。</li>
      <li><strong>最有意思的发现：</strong> thinking 出现 trace-indexed targeted retrieval head；这更接近 NIAH 文档里 CoT 逐个检索 needle 的机制假设。</li>
      <li><strong>probe caveat：</strong> thinking trace 的位置结构可能泄漏 count，后续 probe 应控制位置/trace length，或固定 anchor 位置。</li>
      <li><strong>下一步：</strong> 用 main run 复现同样报告；再加更难泛化设置，例如更长 seq_len、更低 needle density、count extrapolation 或 distractor markers。</li>
      <li><strong>因果性：</strong> 如果 ablation 没有明显降低 accuracy，说明该 head 可能是 diagnostic/redundant；需要 activation patching 或 targeted attention masking 来做更强 causal test。</li>
    </ul>
  </section>
</div>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained HTML report for Trace Count v2 results.")
    parser.add_argument("--result_dir", type=Path, required=True, help="Bundle directory, e.g. colab_results/v2_marker_trace_debug_seed1234_...")
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path. Defaults to RESULT_DIR/report.html")
    args = parser.parse_args()
    result_dir = args.result_dir
    out_html = args.out or (result_dir / "report.html")
    build_report(result_dir, out_html)
    print(out_html)


if __name__ == "__main__":
    main()
