from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: object) -> float | None:
    if value in (None, "", "nan", "None"):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def fmt(value: object, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return ""
    if abs(number) >= 1000 and number.is_integer():
        return f"{int(number)}"
    if abs(number) < 1e-4 and number != 0:
        return f"{number:.2e}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def table(headers: list[str], rows: list[list[object]], cls: str = "") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap {cls}"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def kpi(label: str, value: object, sub: str = "", tone: str = "") -> str:
    return (
        f'<div class="kpi {tone}"><div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{esc(value)}</div>'
        f'<div class="kpi-sub">{esc(sub)}</div></div>'
    )


def figure(src: str, title: str, caption: str, wide: bool = False) -> str:
    klass = "figure-card wide" if wide else "figure-card"
    return (
        f'<figure class="{klass}">'
        f'<img src="{esc(src)}" alt="{esc(title)}">'
        f'<figcaption><strong>{esc(title)}</strong><br>{esc(caption)}</figcaption>'
        f"</figure>"
    )


def group_mean(rows: list[dict[str, str]], keys: list[str], cols: list[str]) -> dict[tuple[str, ...], dict[str, float | None]]:
    buckets: dict[tuple[str, ...], dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        for col in cols:
            buckets[key][col].append(as_float(row.get(col)))
    return {key: {col: mean(vals) for col, vals in vals_by_col.items()} for key, vals_by_col in buckets.items()}


def final_behavior_table(rows: list[dict[str, str]]) -> list[list[object]]:
    grouped = group_mean(
        rows,
        ["model_type", "eval_mode", "seq_len_eval"],
        ["final_accuracy", "final_mae", "trace_exact_rate", "trace_marker_recall"],
    )
    order_mode = {"direct": 0, "generated_trace": 1, "oracle_trace": 2}
    out: list[list[object]] = []
    for (model_type, eval_mode, seq_len), vals in sorted(
        grouped.items(), key=lambda item: (item[0][0], int(item[0][2]), order_mode.get(item[0][1], 9))
    ):
        out.append(
            [
                model_type,
                eval_mode,
                seq_len,
                fmt(vals.get("final_accuracy"), 4),
                fmt(vals.get("final_mae"), 3),
                fmt(vals.get("trace_exact_rate"), 4),
                fmt(vals.get("trace_marker_recall"), 4),
            ]
        )
    return out


def prediction_collapse_rows(result_dir: Path, final_step: int) -> list[list[object]]:
    path = result_dir / "metrics" / "eval_by_step.csv"
    if not path.exists():
        return []
    stats: dict[tuple[str, str, str], dict[str, object]] = defaultdict(
        lambda: {"n": 0, "acc": 0, "mae": 0.0, "pred": Counter()}
    )
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("checkpoint_step") != str(final_step):
                continue
            mode = row.get("eval_mode", "")
            if mode not in {"direct", "generated_trace", "oracle_trace"}:
                continue
            key = (row.get("model_type", ""), mode, row.get("seq_len_eval", ""))
            gold = as_float(row.get("count"))
            pred = as_float(row.get("pred_count"))
            if gold is None or pred is None:
                continue
            pred_int = int(round(pred))
            state = stats[key]
            state["n"] = int(state["n"]) + 1
            state["acc"] = int(state["acc"]) + int(pred_int == int(round(gold)))
            state["mae"] = float(state["mae"]) + abs(pred_int - gold)
            cast_counter: Counter[int] = state["pred"]  # type: ignore[assignment]
            cast_counter[pred_int] += 1

    rows: list[list[object]] = []
    for (model_type, mode, seq_len), state in sorted(
        stats.items(), key=lambda item: (item[0][0], int(item[0][2]), item[0][1])
    ):
        n = int(state["n"])
        if n == 0:
            continue
        pred_counter: Counter[int] = state["pred"]  # type: ignore[assignment]
        top = ", ".join(f"{pred}:{count / n:.1%}" for pred, count in pred_counter.most_common(4))
        rows.append(
            [
                model_type,
                mode,
                seq_len,
                n,
                fmt(int(state["acc"]) / n, 4),
                fmt(float(state["mae"]) / n, 2),
                top,
            ]
        )
    return rows


def train_final_rows(result_dir: Path) -> list[list[object]]:
    out: list[list[object]] = []
    for path in sorted((result_dir / "metrics").glob("train_log_*seed*.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        row = rows[-1]
        out.append(
            [
                row.get("model_type", path.stem),
                row.get("seed", ""),
                row.get("step", ""),
                fmt(row.get("train_total_loss"), 4),
                fmt(row.get("train_final_count_ce"), 4),
                fmt(row.get("train_trace_ce"), 4),
                fmt(row.get("learning_rate"), 3),
            ]
        )
    return out


def threshold_rows(rows: list[dict[str, str]]) -> list[list[object]]:
    out: list[list[object]] = []
    for row in rows:
        if row.get("threshold") != "0.99":
            continue
        if row.get("seq_len_eval") not in {"256", "512", "1024"}:
            continue
        out.append(
            [
                row.get("model_type", ""),
                row.get("eval_mode", ""),
                row.get("seq_len_eval", ""),
                row.get("count_bin", ""),
                row.get("step_to_threshold") or "not reached",
                fmt(row.get("auc_accuracy_over_training"), 1),
            ]
        )
    return out


def round2_rows(rows: list[dict[str, str]]) -> list[list[object]]:
    selected = {
        "oracle_trace",
        "empty_trace",
        "deleted_one_item",
        "duplicated_one_item",
        "extra_random_item",
        "last_index_replaced",
        "correct_indices_wrong_markers",
        "wrong_indices_correct_markers",
    }
    grouped = group_mean(
        [r for r in rows if r.get("corruption_type") in selected],
        ["seq_len_eval", "corruption_type"],
        [
            "correct_prompt_count",
            "follows_prompt_count",
            "follows_trace_pair_count",
            "follows_last_index",
            "follows_marker_count",
        ],
    )
    out: list[list[object]] = []
    for (seq_len, corruption), vals in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        out.append(
            [
                seq_len,
                corruption,
                fmt(vals.get("correct_prompt_count"), 3),
                fmt(vals.get("follows_trace_pair_count"), 3),
                fmt(vals.get("follows_last_index"), 3),
                fmt(vals.get("follows_marker_count"), 3),
            ]
        )
    return out


def attention_top_id_rows(rows: list[dict[str, str]]) -> list[list[object]]:
    target = [
        r
        for r in rows
        if r.get("model_type") == "thinking"
        and r.get("seq_len_eval") == "256"
        and r.get("query_anchor") == "index_k_pos"
    ]
    grouped = group_mean(
        target,
        ["layer", "head", "query_anchor"],
        ["correct_top1_rate", "diagonal_dominance", "needle_mass", "needle_to_noise_ratio", "entropy"],
    )
    scored = []
    for key, vals in grouped.items():
        score = sum(vals.get(c) or 0.0 for c in ["correct_top1_rate", "diagonal_dominance", "needle_mass"])
        scored.append((score, key, vals))
    out: list[list[object]] = []
    for _, (layer, head, query_anchor), vals in sorted(scored, reverse=True)[:8]:
        out.append(
            [
                layer,
                head,
                query_anchor,
                fmt(vals.get("correct_top1_rate"), 3),
                fmt(vals.get("diagonal_dominance"), 3),
                fmt(vals.get("needle_mass"), 3),
                fmt(vals.get("needle_to_noise_ratio"), 3),
                fmt(vals.get("entropy"), 3),
            ]
        )
    return out


def attention_len_rows(rows: list[dict[str, str]]) -> list[list[object]]:
    grouped = group_mean(
        [r for r in rows if r.get("model_type") == "thinking" and r.get("query_anchor") == "index_k_pos"],
        ["seq_len_eval"],
        ["correct_top1_rate", "diagonal_dominance", "needle_mass", "needle_to_noise_ratio"],
    )
    out = []
    for (seq_len,), vals in sorted(grouped.items(), key=lambda item: int(item[0][0])):
        out.append(
            [
                seq_len,
                fmt(vals.get("correct_top1_rate"), 3),
                fmt(vals.get("diagonal_dominance"), 3),
                fmt(vals.get("needle_mass"), 3),
                fmt(vals.get("needle_to_noise_ratio"), 3),
            ]
        )
    return out


def probe_rows(rows: list[dict[str, str]]) -> list[list[object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model_type", ""), row.get("anchor_type", ""), row.get("target_type", ""))].append(row)
    important = {
        "ans_pos",
        "pre_ans_pos",
        "think_open_pos",
        "think_close_pos",
        "index_k_pos",
        "pre_index_k",
        "marker_k_pos",
        "post_marker_k",
    }
    out: list[list[object]] = []
    for (model_type, anchor, target), bucket in sorted(grouped.items()):
        if anchor not in important:
            continue
        bucket.sort(
            key=lambda r: (
                as_float(r.get("test_accuracy")) or -1,
                as_float(r.get("r2")) or -1,
                -(as_float(r.get("mae")) or 999),
            ),
            reverse=True,
        )
        row = bucket[0]
        out.append(
            [
                model_type,
                anchor,
                target,
                row.get("layer", ""),
                fmt(row.get("test_accuracy"), 3),
                fmt(row.get("r2"), 3),
                fmt(row.get("mae"), 3),
                fmt(row.get("position_only_accuracy"), 3),
                fmt(row.get("trace_length_only_accuracy"), 3),
                row.get("leakage_prone", ""),
            ]
        )
    return out


def ablation_rows(rows: list[dict[str, str]], col: str, limit: int = 8) -> list[list[object]]:
    scored = []
    for idx, row in enumerate(rows):
        val = as_float(row.get(col))
        if val is not None:
            scored.append((abs(val), idx, row))
    out: list[list[object]] = []
    for _, _, row in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]:
        out.append(
            [
                row.get("model_type", ""),
                row.get("eval_mode", ""),
                row.get("seq_len_eval", ""),
                row.get("count_bin", ""),
                f"L{row.get('layer', '')}H{row.get('head', '')}",
                fmt(row.get("baseline_final_accuracy"), 3),
                fmt(row.get("intervened_final_accuracy"), 3),
                fmt(row.get("delta_final_accuracy"), 3),
                fmt(row.get("baseline_trace_exact"), 3),
                fmt(row.get("intervened_trace_exact"), 3),
                fmt(row.get("delta_trace_exact"), 3),
            ]
        )
    return out


def build_html(result_dir: Path) -> str:
    config = read_json(result_dir / "config.json")
    summary = read_json(result_dir / "summary.json")
    tables_dir = result_dir / "tables"
    figures_dir = result_dir / "figures"

    final_rows = read_csv(tables_dir / "round1_final_checkpoint_by_count.csv")
    threshold = read_csv(tables_dir / "round1_step_to_thresholds.csv")
    round2 = read_csv(tables_dir / "round2_follow_rule_summary.csv")
    attention = read_csv(tables_dir / "round3_attention_head_metrics.csv")
    probe = read_csv(tables_dir / "round3_probe_results.csv")
    ablation = read_csv(tables_dir / "round3_head_ablation_results.csv")

    final_step = int(config.get("train_steps", summary.get("train_steps", 10000)))
    non_id = summary.get("non_thinking_final_accuracy_by_len", {}).get("256")
    think_id = summary.get("thinking_final_accuracy_by_len", {}).get("256")
    non_512 = summary.get("non_thinking_final_accuracy_by_len", {}).get("512")
    think_512 = summary.get("thinking_final_accuracy_by_len", {}).get("512")
    think_trace_256 = summary.get("thinking_trace_exact_by_len", {}).get("256")
    think_trace_1024 = summary.get("thinking_trace_exact_by_len", {}).get("1024")

    setup_rows = [
        ["preset", config.get("preset", summary.get("preset"))],
        ["run_name", summary.get("run_name", result_dir.name)],
        ["seed(s)", ", ".join(map(str, config.get("seeds", summary.get("seeds", []))))],
        ["model types", "non_thinking, thinking"],
        ["train seq_len", config.get("train_seq_len")],
        ["eval seq_len", ", ".join(map(str, config.get("seq_lens_eval", [])))],
        ["count range", f"{config.get('count_min', 1)}..{config.get('count_max', 10)}"],
        ["train steps", config.get("train_steps")],
        ["batch size", config.get("batch_size")],
        ["eval every", config.get("eval_every")],
        ["test examples/count", config.get("test_examples_per_count")],
        ["probe examples/count", config.get("probe_examples_per_count")],
        ["attention examples/count", config.get("attention_examples_per_count")],
        ["vocab size", config.get("vocab_size")],
        ["transformer", f"{config.get('n_layers')} layers, {config.get('n_heads')} heads, d_model={config.get('d_model')}"],
        ["position encoding", "RoPE, context_len=" + str(config.get("context_len"))],
        ["objective", "next-token prediction on completion tokens; no loss-mask ablation"],
    ]

    figure_specs = {
        "round1": [
            ("figures/round1_train_loss_by_step.png", "训练 loss", "横轴是训练 step，纵轴是 completion loss。它显示两个模型都已经在训练分布上充分收敛，因此 OOD collapse 不是简单欠训练。"),
            ("figures/round1_final_accuracy_by_step_and_seq_len.png", "不同长度上的 final accuracy", "横轴是训练 step，纵轴是最终答案准确率。256 是训练长度；512/1024 是长度 OOD。长长度长期停在约 0.1，说明这个 OOD 没有被训练过程解决。"),
            ("figures/round1_accuracy_by_count_final.png", "最终 checkpoint 按 gold count 的准确率", "横轴是 gold count，纵轴是 final-answer accuracy。ID 长度接近完美；长长度下多数 count 失败。"),
            ("figures/round1_oracle_vs_generated_trace_accuracy.png", "thinking: generated trace vs oracle trace", "横轴是 seq_len，纵轴是 final-answer accuracy。oracle trace 仍在长长度崩掉，说明失败不只是 trace generation，而是长位置 answer readout 也有问题。"),
            ("figures/round1_trace_metrics_by_seq_len.png", "thinking trace 质量", "横轴是 seq_len，纵轴是 trace exact / marker recall 等 trace 指标。256 有 targeted trace；长长度 trace exact 急剧下降。"),
            ("figures/round1_accuracy_heatmap_count_x_seq_len.png", "长度与 count 的准确率热图", "横轴是 seq_len，纵轴是模型或 count 维度聚合，颜色是 accuracy。它强调 512/1024 的问题是整体性塌缩。"),
        ],
        "round2": [
            ("figures/round2_corruption_accuracy_by_type.png", "corrupted trace 下的正确率", "横轴是 trace corruption 类型，纵轴是是否输出 prompt 的真实 count。该实验只看 thinking 模型在给定 trace 时依赖什么信息。"),
            ("figures/round2_follow_rule_breakdown.png", "corrupted trace follow-rule 分解", "横轴是 corruption 类型，纵轴是跟随 prompt count、trace pair count、last index、marker count 等规则的比例。"),
            ("figures/round2_corruption_by_seq_len.png", "corrupted trace 的长度分解", "横轴是 seq_len，纵轴是 correct prompt count。长长度下 oracle_trace 也接近 0.1，因此 round2 的长长度结论被 readout collapse 污染。"),
            ("figures/round2_confusion_pred_vs_prompt_count.png", "预测 count vs prompt count", "横纵轴分别是 gold/pred count 的混淆关系。用于看模型是随机错还是系统性偏小。"),
            ("figures/round2_confusion_pred_vs_trace_pair_count.png", "预测 count vs trace-pair count", "如果模型跟 trace 项数走，这张图会接近对角线。"),
            ("figures/round2_confusion_pred_vs_last_index.png", "预测 count vs last index", "如果模型直接读最后一个 index token，这张图会出现明显对角关系。"),
        ],
        "round3": [
            ("figures/round3_probe_vs_position_baseline.png", "probe vs position baseline", "横轴是只用绝对位置的 baseline，纵轴是 hidden-state probe accuracy。靠近对角或 position baseline 很高的点不能当作干净 counter evidence。"),
            ("figures/round3_probe_accuracy_layer_by_anchor.png", "probe accuracy by layer/anchor", "横轴是 layer，纵轴是 probe accuracy，不同线/面板对应不同 anchor。注意 index_k_pos 与 trace length 有泄漏风险。"),
            ("figures/round3_probe_r2_layer_by_anchor.png", "ridge probe R2 by layer/anchor", "横轴是 layer，纵轴是 ridge R2。R2 高说明可线性读出，但不等于模型因果使用该方向。"),
            ("figures/round3_attention_head_leaderboard.png", "attention head leaderboard", "按 targeted retrieval 指标排序的 attention head。ID 长度下 layer 3 的 heads 是最值得关注的 candidate。"),
            ("figures/round3_thinking_trace_to_prompt_heatmap_best_head.png", "thinking 最强 trace-to-prompt head 热图", "横轴是 trace item index，纵轴是 prompt needle index，颜色是 attention mass。ID 中近对角说明 CoT trace token 定向检索对应 needle。"),
            ("figures/round3_attention_metrics_by_seq_len.png", "attention metrics by seq_len", "横轴是 seq_len，纵轴是 needle mass / ratio。长长度下 top1 指标可能被低 needle mass 误导，因此要同时看 mass。"),
            ("figures/round3_head_ablation_effects.png", "single-head ablation effects", "横轴是 count bin 或条件，纵轴是 ablation 前后差值。ID 上 ablate targeted head 会显著破坏 generated trace。"),
            ("figures/round3_attention_masking_effects.png", "attention masking status", "当前 targeted attention masking 还未实现；已有的是 single-head zero ablation。"),
            ("figures/round3_nonthinking_ans_to_prompt_attention.png", "non-thinking answer-to-prompt attention", "non-thinking 也可能有一次性 retrieval head，但不是 sequential trace-index retrieval。"),
        ],
    }

    def fig_grid(items: list[tuple[str, str, str]]) -> str:
        cards = []
        for src, title, cap in items:
            if (result_dir / src).exists():
                cards.append(figure(src, title, cap))
        return '<div class="figure-grid">' + "\n".join(cards) + "</div>"

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthetic NIAH Counting v3 Report</title>
  <style>
    :root {{
      --bg: #f5f7fb; --paper: #ffffff; --ink: #172033; --muted: #64748b;
      --line: #d9e1ee; --blue: #2563eb; --green: #16a34a; --amber: #d97706;
      --red: #dc2626; --violet: #7c3aed;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.58; }}
    .shell {{ max-width: 1220px; margin: 0 auto; padding: 28px 22px 56px; }}
    .hero {{ background: linear-gradient(135deg, #111827, #1d4ed8); color: white;
      border-radius: 16px; padding: 30px 34px; box-shadow: 0 18px 50px rgba(15,23,42,.16); }}
    .hero h1 {{ margin: 0 0 10px; font-size: clamp(2rem, 4vw, 3.3rem); line-height: 1.06; letter-spacing: 0; }}
    .hero p {{ margin: 0; max-width: 950px; color: rgba(255,255,255,.86); }}
    .meta, .toc {{ display: flex; flex-wrap: wrap; gap: 9px; margin-top: 16px; }}
    .pill, .toc a {{ display: inline-flex; align-items: center; padding: 5px 10px; border-radius: 999px;
      background: rgba(255,255,255,.13); color: white; text-decoration: none; font-size: .9rem; }}
    section {{ background: var(--paper); border: 1px solid var(--line); border-radius: 14px; padding: 22px;
      margin-top: 20px; box-shadow: 0 10px 26px rgba(15,23,42,.055); }}
    h2 {{ margin: 0 0 10px; font-size: 1.55rem; letter-spacing: 0; }}
    h3 {{ margin: 20px 0 8px; font-size: 1.08rem; }}
    p {{ margin: 8px 0 13px; }}
    code {{ background: #eef2ff; color: #1e3a8a; border-radius: 5px; padding: 1px 5px; }}
    ul {{ margin: 8px 0 12px 22px; padding: 0; }}
    .callout {{ border-left: 5px solid var(--blue); background: #eff6ff; padding: 13px 15px; border-radius: 10px; margin: 13px 0; }}
    .callout.warn {{ border-left-color: var(--amber); background: #fffbeb; }}
    .callout.good {{ border-left-color: var(--green); background: #f0fdf4; }}
    .callout.risk {{ border-left-color: var(--red); background: #fef2f2; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-top: 14px; }}
    .kpi {{ border: 1px solid var(--line); border-radius: 12px; padding: 14px; background: #f8fafc; }}
    .kpi.risk {{ background: #fff7ed; }}
    .kpi.good {{ background: #f0fdf4; }}
    .kpi-label {{ color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 800; }}
    .kpi-value {{ font-size: 1.65rem; font-weight: 850; margin-top: 4px; }}
    .kpi-sub {{ color: var(--muted); font-size: .88rem; margin-top: 2px; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 10px; margin: 12px 0 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
    th {{ background: #f8fafc; color: #334155; font-weight: 800; }}
    tr:last-child td {{ border-bottom: 0; }}
    .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 16px; align-items: start; margin-top: 14px; }}
    .figure-card {{ margin: 0; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: white; }}
    .figure-card img {{ display: block; width: 100%; height: auto; max-height: 520px; object-fit: contain; background: white; }}
    .figure-card figcaption {{ padding: 11px 13px; color: #475569; font-size: .9rem; border-top: 1px solid var(--line); }}
    .note {{ color: var(--muted); font-size: .92rem; }}
    details {{ border: 1px solid var(--line); border-radius: 10px; padding: 10px 13px; background: #fbfdff; margin-top: 12px; }}
    summary {{ cursor: pointer; font-weight: 800; }}
    @media (max-width: 760px) {{
      .shell {{ padding: 16px 12px 42px; }}
      .hero {{ padding: 24px 22px; border-radius: 12px; }}
      section {{ padding: 18px; }}
      .figure-grid {{ grid-template-columns: 1fr; }}
      .figure-card img {{ max-height: none; }}
    }}
  </style>
</head>
<body>
<div class="shell">
  <header class="hero">
    <h1>Synthetic NIAH Counting v3 Report</h1>
    <p>这份报告整理 v3 main run：两个模型（non-thinking / thinking）、固定 count range 1..10、训练长度 256、评估长度 256/512/1024，以及 corrupted trace、probe、attention、single-head ablation 的诊断结果。</p>
    <div class="meta">
      <span class="pill">run: {esc(summary.get("run_name", result_dir.name))}</span>
      <span class="pill">preset: {esc(config.get("preset"))}</span>
      <span class="pill">seed: {esc(config.get("seeds"))}</span>
      <span class="pill">train seq_len: {esc(config.get("train_seq_len"))}</span>
      <span class="pill">steps: {esc(config.get("train_steps"))}</span>
    </div>
    <nav class="toc">
      <a href="#summary">Summary</a>
      <a href="#setup">Setup</a>
      <a href="#round1">Round 1</a>
      <a href="#diagnosis">OOD Diagnosis</a>
      <a href="#round2">Round 2</a>
      <a href="#round3">Round 3</a>
      <a href="#next">Next</a>
    </nav>
  </header>

  <section id="summary">
    <h2>Executive Summary</h2>
    <div class="callout risk">
      <strong>最重要结论：</strong> 这次 OOD 结果确实“不太合理”。512/1024 的低准确率主要测到了长序列位置/上下文外推失败，而不是一个干净的 counting OOD。尤其是 thinking 模型在 <code>oracle_trace</code> 下长长度仍然约 0.1，说明给了正确 trace 之后 final answer readout 也崩了。
    </div>
    <div class="callout good">
      <strong>仍然可信的发现：</strong> ID 长度 256 下，thinking 模型确实形成了 trace-index 到 prompt needle 的 targeted retrieval head；single-head ablation 会显著破坏 generated trace。这部分更接近我们想要的 mechanistic evidence。
    </div>
    <div class="kpi-grid">
      {kpi("non-thinking ID accuracy", fmt(non_id, 4), "direct, seq_len=256", "good")}
      {kpi("thinking ID accuracy", fmt(think_id, 4), "generated trace, seq_len=256", "good")}
      {kpi("thinking ID trace exact", fmt(think_trace_256, 4), "generated trace exact, seq_len=256", "good")}
      {kpi("non-thinking L512 accuracy", fmt(non_512, 4), "length OOD", "risk")}
      {kpi("thinking L512 accuracy", fmt(think_512, 4), "length OOD", "risk")}
      {kpi("thinking L1024 trace exact", fmt(think_trace_1024, 4), "generated trace exact, seq_len=1024", "risk")}
    </div>
  </section>

  <section id="setup">
    <h2>Experiment Setup</h2>
    <p><strong>任务定义。</strong> 输入是 symbolic haystack，里面混有普通 noise tokens 和 needle/marker tokens。non-thinking 模型直接输出 <code>&lt;Ans&gt; count</code>；thinking 模型先输出显式 trace（第 k 个 needle 的 index token 和 marker token），再输出最终 count。</p>
    <p><strong>本次 v3 的关键限制。</strong> 训练只用 <code>seq_len=256</code>，但评估包含 <code>512/1024</code>。因此 OOD 同时改变了 answer token 位置、needle density、retrieval distance、softmax key 数量，不能单独解释为 counting OOD。</p>
    {table(["field", "value"], setup_rows)}
  </section>

  <section id="round1">
    <h2>Round 1: Behavior And Length OOD</h2>
    <p><strong>横轴/纵轴定义。</strong> 这一组图主要看训练步数、序列长度、gold count 与最终 count accuracy 的关系。accuracy 只看最终答案 token 是否等于真实 count；thinking 的 trace exact / marker recall 是单独指标。</p>
    {table(["model", "eval mode", "seq_len", "final acc", "final MAE", "trace exact", "marker recall"], final_behavior_table(final_rows))}
    <h3>Step To Threshold</h3>
    <p>下面只列 threshold=0.99：ID 长度达到阈值很快；512/1024 基本未达到阈值。</p>
    {table(["model", "eval mode", "seq_len", "count bin", "step to 0.99", "accuracy AUC"], threshold_rows(threshold))}
    {fig_grid(figure_specs["round1"])}
  </section>

  <section id="diagnosis">
    <h2>Why The OOD Looks Strange</h2>
    <p>我额外从 <code>eval_by_step.csv</code> 里检查了 final checkpoint 的预测分布。长长度不是“随机 chance”，而是系统性偏向很小的数字，尤其是 <code>1</code>。这说明模型在长位置上进入了一个稳定但错误的 readout regime。</p>
    {table(["model", "eval mode", "seq_len", "n", "acc", "MAE", "top predictions"], prediction_collapse_rows(result_dir, final_step))}
    <div class="callout warn">
      <strong>解释。</strong> 如果 thinking 的 <code>generated_trace</code> 失败，还可以说 retrieval/trace generation 失败；但 <code>oracle_trace</code> 也失败，就说明最终答案模块没有学会在 512/1024 的 answer position 上利用 trace。当前 OOD 把“能不能数”和“能不能在没见过的位置输出”绑在一起了。
    </div>
    {table(["model", "seed", "final step", "train total loss", "final-count CE", "trace CE", "lr"], train_final_rows(result_dir))}
  </section>

  <section id="round2">
    <h2>Round 2: Corrupted Trace Diagnostics</h2>
    <p><strong>图表定义。</strong> 这里只针对 thinking 模型：给模型不同形式的 trace（正确、空 trace、删除一项、复制一项、错误 marker、错误 index 等），再看它最后输出是否跟 prompt count、trace pair count、last index 或 marker count 走。</p>
    <p><strong>读法。</strong> ID 长度 256 下，这组实验可以帮助判断模型是不是依赖 trace 的项数、最后 index 或 marker 数量；但 512/1024 下因为 oracle trace 也 readout collapse，所以长长度 corrupted-trace 结论不应过度解读。</p>
    {table(["seq_len", "corruption", "correct prompt count", "follows trace pairs", "follows last index", "follows marker count"], round2_rows(round2))}
    {fig_grid(figure_specs["round2"])}
  </section>

  <section id="round3">
    <h2>Round 3: Probe, Attention, And Ablation</h2>
    <h3>Probe</h3>
    <p><strong>定义。</strong> probe 从指定 anchor 的 hidden state 线性预测 count 或 prefix count。这里必须同时看 position-only / trace-length-only baseline；如果 baseline 已经很高，hidden probe 的高分不能直接解释为干净的 counter direction。</p>
    {table(["model", "anchor", "target", "best layer", "test acc", "R2", "MAE", "position-only", "trace-len-only", "leakage-prone"], probe_rows(probe))}
    <h3>Attention</h3>
    <p><strong>定义。</strong> thinking 的核心 attention 指标看 trace index token 是否 attend 到对应 prompt needle。<code>correct_top1</code> 是 top attention needle 是否为第 k 个 needle；<code>diagonal_dominance</code> 看对角项占 needle mass 的比例；<code>needle_mass</code> 是 query 放到全部 prompt needles 的总 attention mass。</p>
    {table(["layer", "head", "query", "correct top1", "diagonal", "needle mass", "needle/noise", "entropy"], attention_top_id_rows(attention))}
    <p>下面是按长度聚合后的 thinking index-token attention。长长度里 top1 可能看似还高，但 needle mass 很低时不能说明模型真的在强检索对应 needle。</p>
    {table(["seq_len", "correct top1", "diagonal", "needle mass", "needle/noise"], attention_len_rows(attention))}
    <h3>Single-Head Ablation</h3>
    <p><strong>定义。</strong> single-head zero ablation 把某个 head 的输出置零，比较 intervention 前后的 final accuracy / trace exact。ID 上 layer 3 head 2 对 generated trace 最关键，说明 targeted retrieval head 至少有部分因果作用。</p>
    {table(["model", "mode", "seq_len", "bin", "head", "base acc", "abl acc", "delta acc", "base trace", "abl trace", "delta trace"], ablation_rows(ablation, "delta_trace_exact"))}
    {fig_grid(figure_specs["round3"])}
    <details>
      <summary>原始表文件</summary>
      <ul>
        <li><code>tables/round3_probe_results.csv</code></li>
        <li><code>tables/round3_attention_head_metrics.csv</code></li>
        <li><code>tables/round3_head_ablation_results.csv</code></li>
      </ul>
    </details>
  </section>

  <section id="next">
    <h2>Interpretation And Next Experiments</h2>
    <div class="callout">
      <strong>可以写进当前实验总结的版本：</strong> v3 在 ID 长度下复现了 thinking trace 的 targeted retrieval 机制；但当前 length OOD 不是干净的 counting OOD，因为 oracle trace 也无法救回 final count，表明 answer readout / position extrapolation 已经失败。
    </div>
    <h3>下一版建议</h3>
    <ul>
      <li><strong>不要把 256→512/1024 当主 OOD。</strong> 它适合作为附录压力测试，不适合作为核心 counting OOD。</li>
      <li><strong>用“所有数字都见过”的组合 OOD。</strong> 例如 easy calibration 里出现 count 1..10，main hard setting 训练 count 1..5，测试 hard count 6..10。</li>
      <li><strong>保留 oracle-trace sanity check。</strong> 如果 oracle trace 都失败，说明 readout 或位置泛化出了问题，不能归因于 retrieval/counter。</li>
      <li><strong>probe 必须报告 baseline。</strong> position-only 和 trace-length-only baseline 高时，counter direction 结论要降级。</li>
      <li><strong>attention 结论以 ID ablation 为主。</strong> 目前最可信的是 ID 上的 targeted head + single-head ablation，而不是长长度的 top1 attention。</li>
    </ul>
  </section>
</div>
</body>
</html>
"""
    return html_doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_dir",
        nargs="?",
        default="colab_results/v3_20260707_194756_main",
        help="Path to a v3 result directory containing config.json, summary.json, tables/, figures/.",
    )
    parser.add_argument("--out", default=None, help="Output HTML path. Defaults to result_dir/report.html.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(result_dir)
    out = Path(args.out) if args.out else result_dir / "report.html"
    out.write_text(build_html(result_dir), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
