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
    shown = rows[:max_rows] if max_rows is not None else rows
    head = "".join(f"<th>{esc(label)}</th>" for _, label in columns)
    body_rows = []
    for row in shown:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = fmt(value)
            cells.append(f"<td>{esc(value)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    if not body_rows:
        body_rows.append(f"<tr><td colspan='{len(columns)}'>No rows found.</td></tr>")
    more = ""
    if max_rows is not None and len(rows) > max_rows:
        more = f"<p class='small'>Showing {max_rows} of {len(rows)} rows.</p>"
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>{more}"


def group_mean(rows: list[dict[str, str]], keys: list[str], value_keys: list[str]) -> list[dict]:
    sums: dict[tuple, dict[str, float]] = {}
    counts: dict[tuple, int] = defaultdict(int)
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        if key not in sums:
            sums[key] = {k: 0.0 for k in value_keys}
        counts[key] += 1
        for value_key in value_keys:
            sums[key][value_key] += to_float(row.get(value_key), 0.0)
    out = []
    for key, values in sums.items():
        item = {k: v for k, v in zip(keys, key)}
        n = max(counts[key], 1)
        for value_key, total in values.items():
            item[value_key] = total / n
        out.append(item)
    return out


def mean_for(rows: list[dict], key: str, default: float = math.nan) -> float:
    values = [to_float(r.get(key)) for r in rows if not math.isnan(to_float(r.get(key)))]
    return mean(values) if values else default


def find_one(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def figure_card(
    path: Path | None,
    title: str,
    *,
    purpose: str,
    axes: str,
    groups: str,
    result: str,
    interpretation: str,
) -> str:
    if path is None or not path.exists():
        return ""
    return f"""
    <figure class="figure-card">
      <div class="figure-image">
        <img src="{image_data_uri(path)}" alt="{esc(title)}">
      </div>
      <figcaption>
        <h4>{esc(title)}</h4>
        <div class="figure-notes">
          <div><span>这张图在做什么</span>{esc(purpose)}</div>
          <div><span>横纵轴</span>{esc(axes)}</div>
          <div><span>颜色/分组</span>{esc(groups)}</div>
          <div><span>本次结果</span>{esc(result)}</div>
          <div class="wide"><span>说明了什么</span>{esc(interpretation)}</div>
        </div>
      </figcaption>
    </figure>
    """


def figure_grid(cards: list[str], columns: str = "two") -> str:
    visible = [card for card in cards if card.strip()]
    if not visible:
        return ""
    return f"<div class='figure-grid {esc(columns)}'>{''.join(visible)}</div>"


def rows_at_step(rows: list[dict[str, str]], step: int) -> list[dict[str, str]]:
    return [r for r in rows if to_int(r.get("step")) == step]


def saturation_step(by_count: list[dict[str, str]], model_type: str, threshold: float = 0.999) -> int | None:
    steps = sorted({to_int(r.get("step")) for r in by_count if r.get("model_type") == model_type})
    for step in steps:
        rows = [r for r in by_count if r.get("model_type") == model_type and to_int(r.get("step")) == step]
        if rows and min(to_float(r.get("accuracy"), 0.0) for r in rows) >= threshold:
            return step
    return None


def count_bin_definition() -> str:
    return "low=1-3, mid=4-6, high=7-10。本 v2 包没有单独的 ID/OOD split；它用同一组 marker token，在不同 count 数量上比较模型行为。"


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
    ablation = read_csv(run_dir / "targeted_retrieval_deep_dive" / "h3_thinking_head_ablation.csv")

    deep_dir = run_dir / "targeted_retrieval_deep_dive"
    dynamics_path = find_one(deep_dir, "h3_dynamics_*.csv")
    dynamics = read_csv(dynamics_path) if dynamics_path else []

    final_step = max((to_int(r.get("step")) for r in by_count), default=0)
    final_count = rows_at_step(by_count, final_step)
    final_bin = rows_at_step(by_bin, final_step)
    final_train_step = max((to_int(r.get("step")) for r in train), default=0)

    final_summary = []
    for model_type in sorted({r.get("model_type", "") for r in final_count}):
        rows = [r for r in final_count if r.get("model_type") == model_type]
        final_summary.append(
            {
                "model_type": model_type,
                "mean_accuracy": fmt(mean(to_float(r.get("accuracy"), 0.0) for r in rows)),
                "min_accuracy": fmt(min(to_float(r.get("accuracy"), 0.0) for r in rows)),
                "mean_mae": fmt(mean(to_float(r.get("mae"), 0.0) for r in rows)),
                "saturation_step": saturation_step(by_count, model_type),
            }
        )

    count_rows = [
        {
            "model_type": r.get("model_type"),
            "count": r.get("count"),
            "accuracy": fmt(r.get("accuracy")),
            "mae": fmt(r.get("mae")),
            "under_rate": fmt(r.get("under_rate")),
            "over_rate": fmt(r.get("over_rate")),
            "final_answer_loss": fmt(r.get("eval_final_answer_loss"), 4),
        }
        for r in sorted(final_count, key=lambda x: (x.get("model_type", ""), to_int(x.get("count"))))
    ]

    bin_rows = [
        {
            "model_type": r.get("model_type"),
            "count_bin": r.get("count_bin"),
            "accuracy": fmt(r.get("accuracy")),
            "mae": fmt(r.get("mae")),
            "final_answer_loss": fmt(r.get("eval_final_answer_loss"), 4),
        }
        for r in sorted(final_bin, key=lambda x: (x.get("model_type", ""), x.get("count_bin", "")))
    ]

    last_train_rows = [
        {
            "model_type": r.get("model_type"),
            "train_loss": fmt(r.get("train_loss"), 4),
            "completion_loss": fmt(r.get("train_completion_loss"), 4),
            "final_answer_loss": fmt(r.get("train_final_answer_loss"), 4),
            "learning_rate": fmt(r.get("learning_rate"), 6),
        }
        for r in train
        if to_int(r.get("step")) == final_train_step
    ]

    probe_rows_sorted = sorted(
        probes,
        key=lambda r: (
            to_float(r.get("ridge_rounded_accuracy"), -1.0),
            to_float(r.get("probe_accuracy"), -1.0),
            to_float(r.get("probe_r2"), -1.0),
        ),
        reverse=True,
    )
    probe_rows = [
        {
            "model_type": r.get("model_type"),
            "label_type": r.get("label_type"),
            "anchor_type": r.get("anchor_type"),
            "layer": r.get("layer"),
            "probe_accuracy": fmt(r.get("probe_accuracy")),
            "ridge_rounded_accuracy": fmt(r.get("ridge_rounded_accuracy")),
            "probe_mae": fmt(r.get("probe_mae")),
            "probe_r2": fmt(r.get("probe_r2")),
        }
        for r in probe_rows_sorted[:16]
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
    best_think = thinking_head_rank[0] if thinking_head_rank else {}
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
    best_non = non_head_rank[0] if non_head_rank else {}
    non_head_rows = [
        {
            "layer": r.get("layer"),
            "head": r.get("head"),
            "top_n_retrieval_recall": fmt(r.get("top_n_retrieval_recall")),
            "needle_mass": fmt(r.get("ans_to_all_needles_mass")),
            "noise_mass": fmt(r.get("ans_to_noise_mass")),
            "entropy": fmt(r.get("attention_entropy_over_prompt_body")),
        }
        for r in non_head_rank[:12]
    ]

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
            "thinking_accuracy": fmt(r.get("thinking_accuracy")),
            "high_count_accuracy": fmt(r.get("high_count_accuracy")),
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
        {"field": "probe_train/test_examples_per_count", "value": f"{config.get('probe_train_examples_per_count')} / {config.get('probe_test_examples_per_count')}"},
        {"field": "attention_examples_per_count", "value": config.get("attention_examples_per_count")},
        {"field": "count_range", "value": f"{config.get('min_count')}..{config.get('max_count')}"},
        {"field": "count_bins", "value": count_bin_definition()},
        {"field": "noise_vocab_size", "value": config.get("noise_vocab_size")},
        {"field": "marker_vocab_size", "value": config.get("marker_vocab_size")},
        {"field": "model", "value": f"{config.get('n_layer')} layers, {config.get('n_head')} heads, d_model={config.get('n_embd')}, n_positions={config.get('n_positions')}"},
        {"field": "training objective", "value": "all-token next-token prediction; no final-count reweighting"},
    ]

    final_acc_text = "; ".join(
        f"{r['model_type']}: mean={r['mean_accuracy']}, min={r['min_accuracy']}, first all-count >=0.999 step={r['saturation_step']}"
        for r in final_summary
    )
    high_bin_text = "; ".join(
        f"{r['model_type']} {r['count_bin']}={r['accuracy']}"
        for r in bin_rows
        if r["count_bin"] == "high"
    )
    best_think_text = (
        f"{best_think.get('query_anchor')} L{best_think.get('layer')}H{best_think.get('head')}: "
        f"correct top-1={fmt(best_think.get('correct_top1_rate'))}, "
        f"diagonal dominance={fmt(best_think.get('diagonal_dominance'))}, "
        f"needle mass={fmt(best_think.get('needle_attention_mass'))}"
        if best_think
        else "no thinking attention metrics"
    )
    best_non_text = (
        f"L{best_non.get('layer')}H{best_non.get('head')}: "
        f"top-n recall={fmt(best_non.get('top_n_retrieval_recall'))}, "
        f"needle mass={fmt(best_non.get('ans_to_all_needles_mass'))}, "
        f"entropy={fmt(best_non.get('attention_entropy_over_prompt_body'))}"
        if best_non
        else "no non-thinking attention metrics"
    )

    baseline = next((r for r in ablation if r.get("condition") == "baseline_no_ablation"), {})
    target_ablation = next((r for r in ablation if "target" in r.get("condition", "")), {})
    ablation_text = "没有找到 ablation 表。"
    if baseline and target_ablation:
        acc_delta = to_float(baseline.get("accuracy"), 0.0) - to_float(target_ablation.get("accuracy"), 0.0)
        trace_delta = to_float(baseline.get("trace_exact_match_rate"), 0.0) - to_float(target_ablation.get("trace_exact_match_rate"), 0.0)
        ablation_text = (
            f"target-head ablation 后 accuracy 从 {fmt(baseline.get('accuracy'))} 到 {fmt(target_ablation.get('accuracy'))} "
            f"(下降 {fmt(acc_delta)}), trace exact 从 {fmt(baseline.get('trace_exact_match_rate'))} 到 "
            f"{fmt(target_ablation.get('trace_exact_match_rate'))} (下降 {fmt(trace_delta)})。"
        )

    plot_dir = run_dir / "plots"
    probe_dir = run_dir / "probes"
    attn_dir = run_dir / "attention"
    h1_low = find_one(deep_dir, "h1_matrix_*_low.png")
    h1_mid = find_one(deep_dir, "h1_matrix_*_mid.png")
    h1_high = find_one(deep_dir, "h1_matrix_*_high.png")

    main_figures = [
        figure_card(
            plot_dir / "train_loss_vs_step.png",
            "训练损失曲线：两个模型是否学会训练分布",
            purpose="展示 non-thinking 与 thinking 在训练过程中的 token-level loss。实线是 completion loss，虚线是 final-answer loss。",
            axes="横轴是 training step；纵轴是 cross-entropy loss，越低代表模型越能预测对应 token。",
            groups="颜色区分 model_type：non_thinking 直接回答，thinking 先生成 trace 再回答。",
            result=f"最终 step={final_train_step} 时两个模型的 loss 都接近 0。",
            interpretation="这说明两个模型都已经把当前 synthetic task 学到接近饱和。注意 thinking 的 completion 更长，包含 trace token，因此 completion loss 不能和 non-thinking 做完全公平的一对一比较。",
        ),
        figure_card(
            plot_dir / "eval_final_answer_loss_vs_step.png",
            "测试集 final-answer loss：最后计数答案是否稳定",
            purpose="只看最后 count token 的 cross-entropy，不把 thinking 的 trace token 混进来。",
            axes="横轴是 training step；纵轴是 final-answer cross-entropy，越低越好。",
            groups="颜色区分 non_thinking 与 thinking。",
            result="main run 后期两条曲线都贴近 0。",
            interpretation="最后答案层面已经不是主要瓶颈；如果还要区分机制，需要看 probe、attention 和 ablation，而不是只看 final loss。",
        ),
        figure_card(
            plot_dir / "eval_accuracy_by_bin_vs_step.png",
            "按 count bin 的 exact-count accuracy：低/中/高数量是否都学会",
            purpose="把 count 分成 low/mid/high 三组，观察训练过程中不同数量区间的准确率。",
            axes="横轴是 training step；纵轴是 final count exact-match accuracy。",
            groups=f"颜色是 model_type；线型是 count_bin。{count_bin_definition()}",
            result=f"最终 high bin 也达到：{high_bin_text}。",
            interpretation="这说明在本设置下，数量从 1 到 10 都被模型解决了；它不是早期 v1 那种 length/count OOD 崩溃设置。",
        ),
        figure_card(
            plot_dir / "final_accuracy_by_count.png",
            "最终 checkpoint 按精确 count 的准确率",
            purpose="检查 count=1 到 count=10 是否存在某些具体数字特别难。",
            axes="横轴是 gold count；纵轴是最终答案 exact-match accuracy。",
            groups="颜色区分 non_thinking 与 thinking。",
            result=final_acc_text,
            interpretation="main run 中每个 count 都达到 1.0，说明行为结果已经饱和。后续的科学问题应转向：两个模型是否用同一种机制解决，而不是谁更准。",
        ),
        figure_card(
            plot_dir / "accuracy_heatmap_by_count_and_step_non_thinking.png",
            "non-thinking 学习动态 heatmap",
            purpose="展示 non-thinking 在不同 step、不同 count 上何时学会。",
            axes="横轴是 training step；纵轴是 gold count；颜色是 accuracy，黄色/亮色表示更高。",
            groups="单独展示 non-thinking；每一行对应一个 count。",
            result=f"first all-count >=0.999 step={next((r['saturation_step'] for r in final_summary if r['model_type']=='non_thinking'), None)}。",
            interpretation="如果某些高 count 行更晚变亮，说明直接回答模型先学低 count，再学更复杂数量；最终全亮表示 main 训练预算足够。",
        ),
        figure_card(
            plot_dir / "accuracy_heatmap_by_count_and_step_thinking.png",
            "thinking 学习动态 heatmap",
            purpose="展示 thinking trace-supervised 模型在不同 step、不同 count 上何时学会。",
            axes="横轴是 training step；纵轴是 gold count；颜色是 accuracy。",
            groups="单独展示 thinking；每一行对应一个 count。",
            result=f"first all-count >=0.999 step={next((r['saturation_step'] for r in final_summary if r['model_type']=='thinking'), None)}。",
            interpretation="和 non-thinking 对照可看出 trace supervision 是否让高 count 更早稳定。最终同样饱和，因此机制差异主要要看 attention/probe。",
        ),
    ]

    probe_figures = [
        figure_card(
            probe_dir / "probe_final_count_accuracy_heatmap_non_thinking.png",
            "Probe heatmap：non-thinking hidden state 中能否线性读出 final count",
            purpose="对不同 layer 与 anchor position 的 hidden state 训练 count probe。",
            axes="横轴是 layer/embedding；纵轴是 anchor_type，例如 ans_token 或 last_prompt_token；颜色是 probe accuracy。",
            groups="每个格子代表一个 hidden-state 位置的线性可读性。",
            result="non-thinking 的 embedding/早期位置通常不含 final count；后层或答案附近若变高，说明答案相关表征被形成。",
            interpretation="probe 高说明 count 信息线性可读，但不自动说明模型正在因果使用这一路径。",
        ),
        figure_card(
            probe_dir / "probe_final_count_accuracy_heatmap_thinking.png",
            "Probe heatmap：thinking hidden state 中能否线性读出 final count",
            purpose="同样读 final count，但 thinking 有显式 trace token，因此某些位置天然含有 count/position 信息。",
            axes="横轴是 layer/embedding；纵轴是 anchor_type，例如 think_start、think_end、ans_token。",
            groups="颜色是 final-count probe accuracy。",
            result="main run 中 thinking 多个位置可被 probe 近乎完美读出 final count。",
            interpretation="这支持 thinking 表征中 count 更显式；但需要小心 trace length/position 泄漏，最好结合 prefix-count probe 和 attention deep dive 解读。",
        ),
        figure_card(
            probe_dir / "probe_prefix_count_accuracy_heatmap_thinking.png",
            "Probe heatmap：thinking trace 前缀是否编码 running count",
            purpose="在 trace 的第 k 项附近读出 prefix_count，也就是已经处理到第几个 needle。",
            axes="横轴是 layer；纵轴是 trace anchor，例如 post_marker_k；颜色是 prefix-count probe accuracy。",
            groups="只针对 thinking，因为 non-thinking 没有逐项 trace。",
            result="高 accuracy 表示 trace 过程中存在可线性读出的 running-count / index 表征。",
            interpretation="如果 prefix-count probe 高，同时 attention 呈现第 k 个 trace item 对齐第 k 个 needle，就更接近“显式 sequential counter/retrieval”的证据链。",
        ),
        figure_card(
            probe_dir / "probe_prefix_count_mae_heatmap_thinking.png",
            "Probe heatmap：thinking prefix-count 的 MAE",
            purpose="用误差大小补充 accuracy，避免只看 exact match 掩盖接近程度。",
            axes="横轴是 layer；纵轴是 trace anchor；颜色是 MAE，越低越好。",
            groups="只针对 thinking prefix-count probe。",
            result="低 MAE 的位置说明 running count 不仅可分类，而且数值上接近正确。",
            interpretation="MAE 与 accuracy 一起看，可以区分“完全读出”和“只差一两步”的中间状态。",
        ),
        figure_card(
            probe_dir / "probe_accuracy_vs_training_step_ans_token.png",
            "关键 final-count probe 排名",
            purpose="把最相关的 anchor/layer probe 做成条形图，快速比较哪些位置最能读出 final count。",
            axes="横轴是 model:anchor:layer 组合；纵轴是 probe accuracy。",
            groups="每根柱子是一个 probe setting。",
            result="thinking 的 trace/answer 附近 probe 通常排在最前。",
            interpretation="这说明显式 trace 让 count 信息更容易被线性 probe 捕捉；但仍需和行为及 ablation 一起判断因果性。",
        ),
    ]

    attention_figures = [
        figure_card(
            attn_dir / "attention_thinking_correct_top1_by_layer_head.png",
            "Thinking attention：trace item k 是否最关注 prompt needle k",
            purpose="检验 targeted retrieval：第 k 个 trace step 的 query 是否把最大注意力给第 k 个 needle。",
            axes="横轴是 attention head；纵轴是 layer；颜色是 correct top-1 rate。",
            groups="只展示平均表现最好的 query_anchor。",
            result=f"最强 thinking head 是 {best_think_text}。",
            interpretation="correct top-1 接近 1 表示 trace step 与 prompt needle 形成近乎一一对应的 retrieval，这很像 NIAH CoT 中逐项检索 needle 的模式。",
        ),
        figure_card(
            attn_dir / "attention_thinking_diagonal_dominance_by_layer_head.png",
            "Thinking attention：trace-to-needle 矩阵的对角占优",
            purpose="不仅看 top-1，还看整个 trace item k 到 needle j 的矩阵是否沿对角线集中。",
            axes="横轴是 attention head；纵轴是 layer；颜色是 diagonal dominance，越高越像 k 对 k。",
            groups="只展示最佳 query_anchor。",
            result=f"最佳 head 的 diagonal dominance={fmt(best_think.get('diagonal_dominance'))}。",
            interpretation="高 diagonal dominance 说明不是泛泛看所有 needle，而是结构化地把第 k 个 trace token 对齐到第 k 个 prompt needle。",
        ),
        figure_card(
            attn_dir / "attention_matrix_thinking_best_head_low.png",
            "Thinking 最佳 head 的 attention matrix：low count",
            purpose="直接展示最佳 targeted head 在 low count 样本上的平均 trace-to-needle 矩阵。",
            axes="横轴是 prompt needle index j；纵轴是 trace item index k；颜色是 attention mass。",
            groups="low count = 1-3。",
            result="若亮点集中在对角线，代表第 k 个 trace item 检索第 k 个 needle。",
            interpretation="这是 targeted retrieval 最直观的图：它把抽象指标变成了可检查的矩阵结构。",
        ),
        figure_card(
            attn_dir / "attention_matrix_thinking_best_head_mid.png",
            "Thinking 最佳 head 的 attention matrix：mid count",
            purpose="查看中等 count 时 targeted retrieval 是否仍保持。",
            axes="横轴是 prompt needle index j；纵轴是 trace item index k；颜色是 attention mass。",
            groups="mid count = 4-6。",
            result="main run 中 best head 在 mid count 也保持明显对角结构。",
            interpretation="这说明该机制不是只在很短 trace 上出现，而能延伸到更多 needle。",
        ),
        figure_card(
            attn_dir / "attention_matrix_thinking_best_head_high.png",
            "Thinking 最佳 head 的 attention matrix：high count",
            purpose="检查 count=7-10 时 targeted retrieval 是否还能维持。",
            axes="横轴是 prompt needle index j；纵轴是 trace item index k；颜色是 attention mass。",
            groups="high count = 7-10。",
            result="主实验里 high count 仍有强对角结构。",
            interpretation="这是最关键的 evidence：thinking trace 在高 count 也能逐项对齐 prompt needle，和 NIAH targeted retrieval 类比最强。",
        ),
        figure_card(
            attn_dir / "attention_nonthinking_topn_recall_by_layer_head.png",
            "Non-thinking attention：最终 <Ans> 是否一次性覆盖所有 needles",
            purpose="对 non-thinking 的 final answer query，看 top-n prompt positions 中有多少是真 needle。",
            axes="横轴是 attention head；纵轴是 layer；颜色是 top-n retrieval recall。",
            groups="n 等于该样本 gold count；每格是一个 layer/head 的平均。",
            result=f"最强 non-thinking head 是 {best_non_text}。",
            interpretation="top-n recall 高说明 non-thinking 也能检索 needle，但它是最终答案 token 的一次性聚合，不是 trace step k 到 needle k 的逐项机制。",
        ),
        figure_card(
            attn_dir / "attention_nonthinking_ans_needle_mass_by_layer_head.png",
            "Non-thinking attention：最终 <Ans> 分给 needles 的总注意力",
            purpose="补充 top-n recall：即使 top positions 找到 needle，总 attention mass 是否真的集中在 needles 上。",
            axes="横轴是 attention head；纵轴是 layer；颜色是 ans_to_all_needles_mass。",
            groups="每格是一个 layer/head 的平均。",
            result=f"best non-thinking needle mass={fmt(best_non.get('ans_to_all_needles_mass'))}。",
            interpretation="如果 recall 高但 mass 不高，说明它能把 needle 排在前面，但注意力仍较分散；这和 thinking 的高 needle mass targeted head 不同。",
        ),
    ]

    deep_figures = [
        figure_card(
            deep_dir / "h1_thinking_correct_top1_by_head.png",
            "H1 deep dive：thinking targeted head 的 correct top-1 排名",
            purpose="系统枚举 thinking 的 query_anchor/layer/head，寻找是否存在强 targeted retrieval head。",
            axes="横轴是 head；纵轴是 layer；颜色是 correct top-1 rate。",
            groups="固定为最佳 query_anchor；不同格子是不同 attention head。",
            result=f"最佳 candidate：{best_think_text}。",
            interpretation="这个图回答 H1：是否真的有一个 head 在做 k-to-k retrieval。main run 的结果非常强，支持 targeted retrieval 假设。",
        ),
        figure_card(
            deep_dir / "h1_thinking_diagonal_dominance_by_head.png",
            "H1 deep dive：targeted head 的 diagonal dominance",
            purpose="检验最佳 retrieval 是否是整体矩阵对角化，而不只是 top-1 巧合。",
            axes="横轴是 head；纵轴是 layer；颜色是 diagonal dominance。",
            groups="固定为最佳 query_anchor。",
            result=f"最佳 candidate diagonal dominance={fmt(best_think.get('diagonal_dominance'))}。",
            interpretation="对角占优高说明 trace 序号和 prompt needle 序号之间有稳定排列关系。",
        ),
        figure_card(
            deep_dir / "h1_thinking_needle_mass_by_head.png",
            "H1 deep dive：targeted head 分给真实 needles 的注意力质量",
            purpose="检查 attention 是否真的落在 needle token 上，而不是仅在矩阵中相对对角。",
            axes="横轴是 head；纵轴是 layer；颜色是 needle_attention_mass。",
            groups="固定为最佳 query_anchor。",
            result=f"最佳 candidate needle mass={fmt(best_think.get('needle_attention_mass'))}。",
            interpretation="needle mass 高让 targeted retrieval 更可信：它不是只在若干非 needle 位置形成对角，而是真正看 prompt 里的目标 needle。",
        ),
        figure_card(
            h1_low,
            "H1 matrix：最佳 targeted head 在 low count 的对齐矩阵",
            purpose="直接看最佳 head 在 low count 上的 trace item k 到 prompt needle j 的平均 attention。",
            axes="横轴 prompt needle index j；纵轴 trace item index k；颜色 attention mass。",
            groups="low count = 1-3。",
            result="low count 上应出现短对角线。",
            interpretation="对角线说明模型不是只知道总数，而是按 trace 序号逐个 retrieve。",
        ),
        figure_card(
            h1_mid,
            "H1 matrix：最佳 targeted head 在 mid count 的对齐矩阵",
            purpose="验证中等数量时 targeted retrieval 是否仍然稳定。",
            axes="横轴 prompt needle index j；纵轴 trace item index k；颜色 attention mass。",
            groups="mid count = 4-6。",
            result="mid count 中对角结构仍明显。",
            interpretation="机制随 count 增大仍存在，说明它更像可扩展算法的一部分，而不是少量样本特例。",
        ),
        figure_card(
            h1_high,
            "H1 matrix：最佳 targeted head 在 high count 的对齐矩阵",
            purpose="验证高数量时是否保持逐项 retrieval。",
            axes="横轴 prompt needle index j；纵轴 trace item index k；颜色 attention mass。",
            groups="high count = 7-10。",
            result="high count 上仍出现强对角，这是最接近 NIAH-CoT targeted retrieval 的证据。",
            interpretation="在 behavior 已饱和的情况下，这张图提供了机制层面的区分：thinking 不是只输出正确答案，而是在 trace 中形成明确检索路径。",
        ),
        figure_card(
            deep_dir / "h2_broad_vs_targeted_summary.png",
            "H2：non-thinking broad retrieval vs thinking targeted retrieval",
            purpose="把两个模型最强 retrieval 指标放在同一视角下比较。",
            axes="横轴是 retrieval route/metric；纵轴是对应分数。",
            groups="non-thinking 使用 final <Ans> query；thinking 使用 trace item query。",
            result=f"thinking best={best_think_text}; non-thinking best={best_non_text}。",
            interpretation="如果 non-thinking recall 高但 needle mass/diagonal 结构弱，而 thinking diagonal/needle mass 很强，就支持二者机制不同：一个偏聚合，一个偏逐项检索。",
        ),
        figure_card(
            deep_dir / "h2_nonthinking_topn_recall.png",
            "H2：non-thinking top-n recall 的 head 分布",
            purpose="寻找 non-thinking 是否有能一次性找齐 needles 的 head。",
            axes="横轴/纵轴是 head/layer；颜色是 top-n recall。",
            groups="每格一个 attention head。",
            result=f"最佳 non-thinking head：{best_non_text}。",
            interpretation="这说明 non-thinking 也不是完全无检索；差别在于它没有显式 trace item k 对应 needle k 的结构。",
        ),
        figure_card(
            deep_dir / "h2_nonthinking_needle_mass.png",
            "H2：non-thinking 对 needle 的注意力质量",
            purpose="看 final <Ans> query 是否把大量 attention mass 分给所有 true needles。",
            axes="横轴/纵轴是 head/layer；颜色是 needle mass。",
            groups="每格一个 attention head。",
            result=f"best non-thinking needle mass={fmt(best_non.get('ans_to_all_needles_mass'))}。",
            interpretation="和 top-n recall 一起看，可以判断 non-thinking 是强聚焦 retrieve，还是只是把 needle 放进 top positions 但分布仍散。",
        ),
        figure_card(
            deep_dir / "h2_nonthinking_entropy.png",
            "H2：non-thinking prompt-body attention entropy",
            purpose="衡量 final <Ans> query 的 attention 分布有多分散。",
            axes="横轴/纵轴是 head/layer；颜色是 entropy，越高越分散。",
            groups="每格一个 attention head。",
            result=f"best non-thinking entropy={fmt(best_non.get('attention_entropy_over_prompt_body'))}。",
            interpretation="entropy 高表示 broad aggregation；entropy 低且 needle mass 高表示更尖锐的 retrieval。",
        ),
        figure_card(
            deep_dir / "h3_targeted_head_training_dynamics.png",
            "H3：targeted head 随训练形成的动态",
            purpose="跟踪最佳 thinking targeted head 在不同 checkpoint 的 correct top-1、diagonal dominance、needle mass，以及行为准确率。",
            axes="横轴是 checkpoint step；纵轴是不同 metric 的值。",
            groups="不同曲线对应 retrieval 指标和 behavior accuracy。",
            result="main run 中 retrieval 指标随训练上升，并在行为准确率饱和附近稳定。",
            interpretation="这支持 targeted retrieval 不是随机后验现象，而是在训练过程中逐步形成；但因果性仍需 ablation/patching 检验。",
        ),
        figure_card(
            deep_dir / "h3_thinking_head_ablation.png",
            "H3：ablation targeted/control heads 后的行为变化",
            purpose="在最终 thinking model 中屏蔽目标 head 或 control head，观察 final accuracy 与 trace 指标变化。",
            axes="横轴是 ablation condition；纵轴是 accuracy / trace metrics。",
            groups="条件包括 baseline、target head、same-layer control、early-layer control。",
            result=ablation_text,
            interpretation="如果 target ablation 只轻微影响 accuracy，说明该 head 对机制诊断很强，但 final answer 可能有冗余路径；trace exact 的下降更直接说明它参与 trace retrieval。",
        ),
    ]

    main_figures_html = (
        figure_grid(main_figures[0:2], "two")
        + figure_grid(main_figures[2:4], "two")
        + figure_grid(main_figures[4:6], "two")
    )
    probe_figures_html = (
        figure_grid(probe_figures[0:2], "two")
        + figure_grid(probe_figures[2:4], "two")
        + figure_grid(probe_figures[4:5], "single")
    )
    attention_figures_html = (
        figure_grid(attention_figures[0:2], "two")
        + figure_grid(attention_figures[2:5], "three")
        + figure_grid(attention_figures[5:7], "two")
    )
    deep_figures_html = (
        figure_grid(deep_figures[0:3], "three")
        + figure_grid(deep_figures[3:6], "three")
        + figure_grid(deep_figures[6:7], "single")
        + figure_grid(deep_figures[7:10], "three")
        + figure_grid(deep_figures[10:12], "two")
    )

    css = """
    :root {
      --bg:#f5f7fb; --paper:#ffffff; --ink:#111827; --muted:#64748b; --line:#d6deea;
      --blue:#2563eb; --green:#16a34a; --amber:#d97706; --red:#dc2626; --soft:#f8fafc;
    }
    * { box-sizing: border-box; }
    body {
      margin:0; background:var(--bg); color:var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
      line-height:1.65;
    }
    .shell { max-width: 1500px; margin: 0 auto; padding: 30px 26px 64px; }
    .hero {
      background: linear-gradient(135deg, #0f172a, #1d4ed8);
      color:white; border-radius: 16px; padding: 34px 40px;
      box-shadow: 0 20px 70px rgba(15,23,42,.18);
    }
    .hero h1 { margin:0 0 10px; font-size: clamp(2.2rem, 4vw, 4rem); line-height:1.05; }
    .hero p { margin:0; max-width:1050px; color:rgba(255,255,255,.86); font-size:1.05rem; }
    .meta, .toc { display:flex; flex-wrap:wrap; gap:9px; margin-top:18px; }
    .pill, .toc a {
      display:inline-flex; align-items:center; padding:6px 11px; border-radius:999px;
      background:rgba(255,255,255,.13); color:white; text-decoration:none; font-size:.92rem;
    }
    section {
      background:var(--paper); border:1px solid var(--line); border-radius:14px;
      padding:26px; margin-top:24px; box-shadow: 0 10px 30px rgba(15,23,42,.055);
    }
    h2 { margin:0 0 12px; font-size:1.65rem; }
    h3 { margin:22px 0 10px; font-size:1.15rem; }
    code { background:#eef2ff; color:#1e3a8a; border-radius:5px; padding:1px 5px; }
    .callout { border-left:5px solid var(--blue); background:#eff6ff; padding:14px 16px; border-radius:10px; margin:14px 0; }
    .callout.good { border-left-color:var(--green); background:#f0fdf4; }
    .callout.warn { border-left-color:var(--amber); background:#fffbeb; }
    .kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; margin-top:16px; }
    .kpi { border:1px solid var(--line); border-radius:12px; padding:15px; background:var(--soft); }
    .kpi .label { color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; font-weight:800; }
    .kpi .value { font-size:1.75rem; font-weight:850; margin-top:4px; }
    .small { color:var(--muted); font-size:.92rem; }
    .table-wrap { overflow:auto; border:1px solid var(--line); border-radius:10px; margin:12px 0 18px; }
    table { width:100%; border-collapse:collapse; font-size:.94rem; }
    th, td { padding:9px 11px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
    th { background:var(--soft); color:#334155; font-weight:850; }
    tr:last-child td { border-bottom:0; }
    .figure-grid { display:grid; gap:16px; align-items:start; margin:18px 0 28px; }
    .figure-grid.single { grid-template-columns: minmax(0, 1fr); }
    .figure-grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .figure-grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .figure-card {
      margin:0; border:1px solid var(--line); border-radius:14px; overflow:hidden; background:white;
      box-shadow: 0 8px 24px rgba(15,23,42,.055);
    }
    .figure-image {
      padding:10px; background:white; height:min(34vw, 520px);
      display:flex; align-items:center; justify-content:center; overflow:hidden;
    }
    .figure-grid.two .figure-image { height:min(26vw, 380px); }
    .figure-grid.three .figure-image { height:min(20vw, 300px); }
    .figure-card img {
      display:block; width:auto; height:auto; max-width:100%; max-height:100%;
      object-fit:contain; margin:0 auto;
    }
    .figure-card figcaption { border-top:1px solid var(--line); padding:13px 14px 15px; background:#fbfdff; }
    .figure-card h4 { margin:0 0 10px; font-size:1.02rem; line-height:1.35; }
    .figure-notes { display:grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap:8px; }
    .figure-grid.two .figure-notes, .figure-grid.three .figure-notes { grid-template-columns: 1fr; }
    .figure-notes div { border:1px solid #e4ebf5; background:white; border-radius:9px; padding:8px 10px; font-size:.91rem; }
    .figure-notes .wide { grid-column: 1 / -1; }
    .figure-notes span { display:block; color:#1d4ed8; font-weight:850; margin-bottom:3px; font-size:.8rem; }
    ul { padding-left: 20px; }
    @media (max-width: 820px) {
      .shell { padding: 18px 12px 42px; }
      section, .hero { padding:20px; }
      .figure-grid.single, .figure-grid.two, .figure-grid.three { grid-template-columns: 1fr; }
      .figure-grid.two .figure-image, .figure-grid.three .figure-image, .figure-image { height:auto; max-height: none; }
      .figure-card img { width:100%; max-height:none; }
      .figure-notes { grid-template-columns: 1fr; }
    }
    """

    caveat = ""
    if str(preset).lower() == "debug":
        caveat = """
        <div class="callout warn">
          <strong>重要 caveat:</strong> 这个结果包是 <code>debug</code> preset。它适合检查 pipeline，不适合作为正式机制结论。
        </div>
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
    <p>这个报告把 v2 主实验整理成可读版：先看最终行为是否解决 counting，再看 probe 是否能读出 count，最后看 attention/deep dive 是否出现类似 NIAH-CoT 的 targeted retrieval。</p>
    <div class="meta">
      <span class="pill">preset: {esc(preset)}</span>
      <span class="pill">seq_len: {esc(config.get("seq_len"))}</span>
      <span class="pill">steps: {esc(config.get("train_steps"))}</span>
      <span class="pill">seed: {esc(config.get("seed"))}</span>
      <span class="pill">saved: {esc(manifest.get("saved_at", ""))}</span>
    </div>
    <nav class="toc">
      <a href="#summary">Summary</a>
      <a href="#definitions">Definitions</a>
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
      <strong>行为层面：</strong> final step=<code>{final_step}</code>，最终 exact-count accuracy 已经饱和。{final_acc_text}。
    </div>
    <div class="callout">
      <strong>机制层面最有意思的发现：</strong> thinking 中出现非常强的 targeted retrieval head：{esc(best_think_text)}。
      这表示第 k 个 trace token 几乎总是 retrieve 第 k 个 prompt needle，和 NIAH 里 CoT token targeted retrieval 的现象非常接近。
    </div>
    <div class="callout">
      <strong>对照：</strong> non-thinking 也有能找 needle 的 head：{esc(best_non_text)}。
      但它发生在最终 <code>&lt;Ans&gt;</code> query 上，更像一次性 broad aggregation，而不是 trace-indexed sequential retrieval。
    </div>
    <div class="kpi-grid">
      {''.join(f"<div class='kpi'><div class='label'>{esc(r['model_type'])} mean accuracy</div><div class='value'>{esc(r['mean_accuracy'])}</div><div class='small'>min by count: {esc(r['min_accuracy'])}; saturation step: {esc(r['saturation_step'])}</div></div>" for r in final_summary)}
    </div>
  </section>

  <section id="definitions">
    <h2>关键定义</h2>
    <ul>
      <li><strong>final count accuracy</strong>：只看最终答案 count 是否等于 gold count。thinking 模型的 trace 是否完全正确不混入这个指标。</li>
      <li><strong>trace exact / marker recall / index accuracy</strong>：只对 thinking 有意义，用来衡量显式 trace 是否逐项正确。</li>
      <li><strong>correct top-1 retrieval</strong>：对 thinking，第 k 个 trace query 的最高 attention 是否落在第 k 个 prompt needle 上。</li>
      <li><strong>diagonal dominance</strong>：trace-to-needle attention matrix 是否沿对角线集中；越接近 1，越像 k-to-k retrieval。</li>
      <li><strong>needle attention mass</strong>：attention 总量有多少给了真正的 needle token；它补充 top-1，避免只看排名不看质量。</li>
      <li><strong>count bins</strong>：{esc(count_bin_definition())}</li>
    </ul>
  </section>

  <section id="setup">
    <h2>Experiment Setting</h2>
    <p>两个模型在同一类 synthetic prompt 上训练。<code>non_thinking</code> 直接输出最终 count；<code>thinking</code> 先输出 indexed trace，再输出最终 count。训练目标是 all-token next-token prediction，没有 final answer 加权。</p>
    {table(setup_rows, [("field", "field"), ("value", "value")])}
  </section>

  <section id="behavior">
    <h2>Behavior: Final Count Accuracy</h2>
    <p>这一节回答：两个模型有没有学会计数？是否在低/中/高 count 上都能正确？注意这里的准确率只看最后答案。</p>
    <h3>Final bin summary</h3>
    {table(bin_rows, [("model_type", "model"), ("count_bin", "bin"), ("accuracy", "accuracy"), ("mae", "MAE"), ("final_answer_loss", "final-answer loss")])}
    <h3>Final exact-count summary</h3>
    {table(count_rows, [("model_type", "model"), ("count", "gold count"), ("accuracy", "accuracy"), ("mae", "MAE"), ("under_rate", "under"), ("over_rate", "over"), ("final_answer_loss", "final-answer loss")])}
    <h3>Last training rows</h3>
    {table(last_train_rows, [("model_type", "model"), ("train_loss", "train loss"), ("completion_loss", "completion loss"), ("final_answer_loss", "final-answer loss"), ("learning_rate", "lr")])}
    {main_figures_html}
  </section>

  <section id="probe">
    <h2>Probe: Count Information in Hidden States</h2>
    <p>这一节回答：模型内部 hidden state 是否线性包含 count/running count 信息。Probe 是诊断工具，不等价于因果证明。</p>
    {table(probe_rows, [("model_type", "model"), ("label_type", "label"), ("anchor_type", "anchor"), ("layer", "layer"), ("probe_accuracy", "clf acc"), ("ridge_rounded_accuracy", "ridge rounded acc"), ("probe_mae", "MAE"), ("probe_r2", "R2")])}
    {probe_figures_html}
  </section>

  <section id="attention">
    <h2>Attention and Retrieval</h2>
    <p>这一节回答：模型是否真的把注意力放到 prompt needles 上；thinking 是否形成 trace item k 到 needle k 的 targeted retrieval。</p>
    <h3>Thinking head ranking</h3>
    {table(thinking_head_rows, [("query_anchor", "query anchor"), ("layer", "layer"), ("head", "head"), ("correct_top1_rate", "correct top-1"), ("diagonal_dominance", "diagonal dominance"), ("needle_attention_mass", "needle mass"), ("noise_attention_mass", "noise mass")])}
    <h3>Non-thinking head ranking</h3>
    {table(non_head_rows, [("layer", "layer"), ("head", "head"), ("top_n_retrieval_recall", "top-n recall"), ("needle_mass", "needle mass"), ("noise_mass", "noise mass"), ("entropy", "entropy")])}
    {attention_figures_html}
  </section>

  <section id="deep">
    <h2>Targeted Retrieval Deep Dive</h2>
    <p>这一节对应三个机制假设：H1 thinking 是否存在 targeted head；H2 non-thinking 是否更像 broad aggregation；H3 targeted head 是否随训练形成、ablation 后是否影响行为。</p>
    <h3>H3 dynamics table</h3>
    {table(dynamics_rows, [("step", "step"), ("correct_top1_rate", "correct top-1"), ("diagonal_dominance", "diagonal dominance"), ("needle_attention_mass", "needle mass"), ("noise_attention_mass", "noise mass"), ("thinking_accuracy", "thinking acc"), ("high_count_accuracy", "high-count acc")], max_rows=24)}
    <h3>H3 head ablation table</h3>
    {table(ablation_rows, [("condition", "condition"), ("accuracy", "accuracy"), ("invalid_rate", "invalid"), ("trace_exact_match_rate", "trace exact"), ("trace_marker_recall", "marker recall"), ("trace_index_accuracy", "index acc")])}
    {deep_figures_html}
  </section>

  <section id="takeaways">
    <h2>Takeaways</h2>
    <ul>
      <li><strong>最终行为已饱和：</strong> main run 中 non-thinking 与 thinking 都能在 count=1..10 上达到完美或近完美 final count accuracy。</li>
      <li><strong>thinking 的机制证据更清楚：</strong> 它出现了强 targeted retrieval head，尤其是 {esc(best_think_text)}，这和 NIAH CoT token 对应检索 prompt needle 的现象相似。</li>
      <li><strong>non-thinking 不是没有 retrieval：</strong> 它也有 top-n recall 高的 head，但更像 final answer token 的一次性聚合，缺少 trace-indexed k-to-k 矩阵结构。</li>
      <li><strong>ablation 暗示冗余：</strong> {esc(ablation_text)} 因此这个 head 很可能是强诊断信号，但 final answer 可能还有备份路径。</li>
      <li><strong>下一步实验：</strong> 如果要证明 thinking 对 OOD 有帮助，需要加更严格的 held-out count 或更长 seq_len OOD；如果要证明机制因果，需要做 activation patching / targeted attention masking / multi-head ablation。</li>
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
    parser.add_argument("--result_dir", type=Path, required=True, help="Bundle directory under colab_results.")
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path. Defaults to RESULT_DIR/report.html")
    args = parser.parse_args()
    out_html = args.out or (args.result_dir / "report.html")
    build_report(args.result_dir, out_html)
    print(out_html)


if __name__ == "__main__":
    main()
