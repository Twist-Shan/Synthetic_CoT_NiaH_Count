from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Iterable


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: object) -> float:
    try:
        if value in (None, ""):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values: Iterable[object]) -> float:
    values = [number(value) for value in values]
    values = [value for value in values if math.isfinite(value)]
    return fmean(values) if values else math.nan


def weighted_mean(rows: list[dict[str, str]], metric: str, weight: str = "n_examples") -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = number(row.get(metric))
        row_weight = number(row.get(weight))
        if not math.isfinite(value):
            continue
        if not math.isfinite(row_weight) or row_weight <= 0:
            row_weight = 1.0
        numerator += value * row_weight
        denominator += row_weight
    return numerator / denominator if denominator else math.nan


def grouped_weighted(
    rows: list[dict[str, str]], keys: list[str], metrics: list[str]
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    output: list[dict[str, object]] = []
    for group_key, group_rows in groups.items():
        item: dict[str, object] = dict(zip(keys, group_key))
        item["n_examples"] = sum(
            number(row.get("n_examples"))
            if math.isfinite(number(row.get("n_examples")))
            else 1.0
            for row in group_rows
        )
        for metric in metrics:
            item[metric] = weighted_mean(group_rows, metric)
        output.append(item)
    return output


def fmt(value: object, digits: int = 3) -> str:
    value = number(value)
    if not math.isfinite(value):
        return "n/a"
    if value != 0 and abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    value = number(value)
    return "n/a" if not math.isfinite(value) else f"{100 * value:.{digits}f}%"


def code(value: object) -> str:
    return f"<code>{html.escape(str(value))}</code>"


def image_data(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def figure(path: Path, title: str, caption: str, *, wide: bool = False) -> str:
    if not path.exists():
        return ""
    classes = "figure wide" if wide else "figure"
    return f"""
    <figure class="{classes}">
      <h3>{html.escape(title)}</h3>
      <img src="{image_data(path)}" alt="{html.escape(title)}">
      <figcaption>{caption}</figcaption>
    </figure>
    """


def table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    headers = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body: list[str] = []
    for row in rows:
        cells = "".join(f"<td>{row.get(key, '')}</td>" for key, _ in columns)
        body.append(f"<tr>{cells}</tr>")
    if not body:
        body.append(f'<tr><td colspan="{len(columns)}">No data</td></tr>')
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def first_perfect_step(eval_rows: list[dict[str, str]], mode: str) -> int | None:
    steps = sorted({int(number(row["step"])) for row in eval_rows})
    for step in steps:
        rows = [
            row
            for row in eval_rows
            if int(number(row["step"])) == step and row.get("mode") == mode
        ]
        if not rows or weighted_mean(rows, "final_accuracy") < 1.0:
            continue
        if mode == "thinking" and weighted_mean(rows, "trace_exact") < 1.0:
            continue
        return step
    return None


def best_probe(
    probe_rows: list[dict[str, str]], mode: str, anchor: str, target: str = "final_count"
) -> dict[str, str] | None:
    rows = [
        row
        for row in probe_rows
        if row.get("mode") == mode
        and row.get("anchor_name") == anchor
        and row.get("target") == target
    ]
    return max(rows, key=lambda row: number(row.get("accuracy"))) if rows else None


def build_report(run_dir: Path) -> str:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    tables = run_dir / "tables"
    figures = run_dir / "figures"
    train_rows = read_rows(tables / "train_log.csv")
    eval_rows = read_rows(tables / "eval_by_step.csv")
    switch_rows = read_rows(tables / "mode_switch.csv")
    attention_rows = read_rows(tables / "attention_metrics.csv")
    similarity_rows = read_rows(tables / "mode_hidden_similarity.csv")
    probe_rows = read_rows(tables / "probe_results.csv")

    final_step = max(int(number(row["step"])) for row in eval_rows)
    final_eval = [row for row in eval_rows if int(number(row["step"])) == final_step]
    final_switch = [row for row in switch_rows if int(number(row["step"])) == final_step]
    step_500_eval = [row for row in eval_rows if int(number(row["step"])) == 500]
    step_500_switch = [row for row in switch_rows if int(number(row["step"])) == 500]

    eval_metrics = [
        "final_accuracy",
        "final_mae",
        "trace_exact",
        "trace_marker_precision",
        "trace_marker_recall",
        "premature_close_rate",
        "missing_close_rate",
        "invalid_count_rate",
        "first_token_switch_accuracy",
        "empty_trace_rate",
    ]
    final_by_mode = {
        str(row["mode"]): row
        for row in grouped_weighted(final_eval, ["mode"], eval_metrics)
    }
    final_by_bin = sorted(
        grouped_weighted(
            final_eval,
            ["mode", "count_bin"],
            ["final_accuracy", "trace_exact", "trace_marker_recall"],
        ),
        key=lambda row: (
            str(row["mode"]),
            {"low": 0, "mid": 1, "high": 2}.get(str(row["count_bin"]), 9),
        ),
    )
    switch_metrics = [
        "p_close_after_think",
        "p_any_marker_after_think",
        "p_gold_first_marker_after_think",
        "p_desired_next_token",
        "argmax_is_close",
        "argmax_is_gold_first_marker",
        "argmax_is_desired",
    ]
    final_switch_by_mode = {
        str(row["mode"]): row
        for row in grouped_weighted(final_switch, ["mode"], switch_metrics)
    }

    thinking_attention = sorted(
        [row for row in attention_rows if row.get("mode") == "thinking"],
        key=lambda row: number(row.get("correct_top1")),
        reverse=True,
    )
    nonthinking_attention = sorted(
        [row for row in attention_rows if row.get("mode") == "nonthinking"],
        key=lambda row: number(row.get("needle_mass")),
        reverse=True,
    )
    best_think = thinking_attention[0]
    best_non = nonthinking_attention[0]
    random_top1 = 10.0 / sum(range(1, 11))

    similarity = []
    for layer in sorted({int(number(row["layer"])) for row in similarity_rows}):
        rows = [row for row in similarity_rows if int(number(row["layer"])) == layer]
        similarity.append({"layer": layer, "cosine": mean(row["cosine_similarity"] for row in rows)})

    probes = [
        best_probe(probe_rows, "nonthinking", "mode_pos"),
        best_probe(probe_rows, "nonthinking", "think_open_pos"),
        best_probe(probe_rows, "nonthinking", "pre_count_pos"),
        best_probe(probe_rows, "thinking", "mode_pos"),
        best_probe(probe_rows, "thinking", "think_open_pos"),
        best_probe(probe_rows, "thinking", "pre_count_pos"),
    ]
    probes = [row for row in probes if row is not None]

    non_final = final_by_mode["nonthinking"]
    think_final = final_by_mode["thinking"]
    non_switch = final_switch_by_mode["nonthinking"]
    think_switch = final_switch_by_mode["thinking"]
    first_non = first_perfect_step(eval_rows, "nonthinking")
    first_think = first_perfect_step(eval_rows, "thinking")
    train = config["train"]
    model = config["model"]
    first_train = train_rows[0]
    last_train = train_rows[-1]

    summary_rows = [
        {
            "mode": "non-thinking / THINK_OFF",
            "examples": f"{int(number(non_final['n_examples'])):,}",
            "final": pct(non_final["final_accuracy"]),
            "trace": "empty by design",
            "switch": pct(non_switch["argmax_is_desired"]),
            "probability": fmt(non_switch["p_desired_next_token"], 6),
        },
        {
            "mode": "thinking / THINK_ON",
            "examples": f"{int(number(think_final['n_examples'])):,}",
            "final": pct(think_final["final_accuracy"]),
            "trace": pct(think_final["trace_exact"]),
            "switch": pct(think_switch["argmax_is_desired"]),
            "probability": fmt(think_switch["p_desired_next_token"], 6),
        },
    ]
    bin_rows = [
        {
            "mode": html.escape(str(row["mode"])),
            "bin": html.escape(str(row["count_bin"])),
            "range": {"low": "1-3", "mid": "4-6", "high": "7-10"}.get(str(row["count_bin"]), ""),
            "examples": f"{int(number(row['n_examples'])):,}",
            "accuracy": pct(row["final_accuracy"]),
            "trace": pct(row["trace_exact"]) if row["mode"] == "thinking" else "n/a",
        }
        for row in final_by_bin
    ]
    attention_table = [
        {
            "head": f"L{int(number(row['layer']))}H{int(number(row['head']))}",
            "top1": fmt(row["correct_top1"]),
            "dominance": fmt(row["diagonal_dominance"]),
            "needle": fmt(row["needle_mass"]),
            "entropy": fmt(row["entropy"]),
        }
        for row in thinking_attention[:8]
    ]
    probe_table = [
        {
            "mode": html.escape(row["mode"]),
            "anchor": code(row["anchor_name"]),
            "layer": row["layer"],
            "accuracy": fmt(row["accuracy"]),
            "r2": fmt(row["r2"]),
            "mae": fmt(row["mae"]),
            "position": fmt(row["position_baseline_acc"]),
            "trace_len": fmt(row["trace_len_baseline_acc"]),
        }
        for row in probes
    ]

    styles = """
    :root{--ink:#172033;--muted:#59657a;--line:#dbe2ed;--soft:#f5f7fb;--blue:#2563eb;--green:#15803d;--amber:#a16207}
    *{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","Microsoft YaHei",Arial,sans-serif;line-height:1.68}
    main{max-width:1220px;margin:0 auto;padding:38px 28px 80px}h1{font-size:34px;margin:0 0 6px}h2{font-size:25px;margin:44px 0 18px;padding-top:20px;border-top:1px solid var(--line)}h3{font-size:17px;margin:0 0 10px}p,li{font-size:15.5px}.subtitle{color:var(--muted);margin-bottom:24px}
    code{background:#edf1f7;border-radius:4px;padding:1px 5px;font-family:Consolas,"SFMono-Regular",monospace}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.card{border:1px solid var(--line);border-radius:8px;padding:14px;background:var(--soft)}.card .label{font-size:12px;color:var(--muted)}.card .value{font-size:21px;font-weight:750;margin-top:3px}
    .callout{border-left:5px solid var(--blue);background:#eef4ff;padding:14px 18px;border-radius:6px;margin:20px 0}.good{border-left-color:var(--green);background:#edf9f0}.warn{border-left-color:var(--amber);background:#fff7e6}
    .format-grid,.figure-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.format{border:1px solid var(--line);border-radius:8px;padding:15px;background:#fbfcfe}.sequence{font-family:Consolas,monospace;font-size:14px;overflow-wrap:anywhere}.on{color:#08783e;font-weight:700}.off{color:#b45309;font-weight:700}.target{color:#b42318;font-weight:700}
    .figure{border:1px solid var(--line);border-radius:8px;padding:14px;margin:0 0 18px;background:#fff}.figure img{display:block;width:100%;height:360px;object-fit:contain;margin:auto}.figure.wide{grid-column:1/-1}.figure.wide img{height:470px}figcaption{font-size:13.5px;color:var(--muted);margin-top:10px}
    .table-wrap{overflow-x:auto;margin:14px 0 24px}table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}th{background:#eef2f8}.formula{background:#f7f8fb;border:1px solid var(--line);border-radius:7px;padding:12px 14px;margin:10px 0;font-family:Consolas,monospace;font-size:13.5px}.small{font-size:13px;color:var(--muted)}
    @media(max-width:900px){main{padding:24px 14px 60px}.cards,.format-grid,.figure-grid{grid-template-columns:1fr}.figure.wide{grid-column:auto}.figure img,.figure.wide img{height:auto}}
    """

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v5 Explicit Switch Report</title><style>{styles}</style></head>
<body><main>
  <h1>v5 Explicit Thinking Switch：结果与机制诊断</h1>
  <div class="subtitle">单模型、显式 <code>&lt;THINK_ON&gt;</code>/<code>&lt;THINK_OFF&gt;</code> 开关；结果目录：{code(run_dir)}</div>
  <div class="cards">
    <div class="card"><div class="label">THINK_OFF final accuracy</div><div class="value">{pct(non_final['final_accuracy'])}</div></div>
    <div class="card"><div class="label">THINK_ON final accuracy</div><div class="value">{pct(think_final['final_accuracy'])}</div></div>
    <div class="card"><div class="label">THINK_ON trace exact</div><div class="value">{pct(think_final['trace_exact'])}</div></div>
    <div class="card"><div class="label">Final switch accuracy</div><div class="value">{pct(mean([non_switch['argmax_is_desired'], think_switch['argmax_is_desired']]))}</div></div>
  </div>
  <div class="callout good"><b>主结论。</b>显式开关训练成功：同一个模型在 <code>THINK_OFF</code> 下立即关闭 thinking block 并直接输出 count，在 <code>THINK_ON</code> 下生成完整 marker trace 后再输出 count。最终两条路径的 ID final-count accuracy 都是 100%，thinking trace exact 也是 100%。但这批结果不支持“thinking 比 non-thinking 更准”；它支持的是“同一模型能被显式 token 稳定路由到两种计算/输出格式”。</div>

  <h2>1. 实验设定</h2>
  {table([
      {'item':'模型','setting':f"随机初始化 GPT-2；{model['n_layer']} layers × {model['n_head']} heads；d_model={model['n_embd']}；MLP={model['n_inner']}；learned absolute position embeddings；context={model['n_positions']}"},
      {'item':'数据','setting':f"prompt body 长度 {train['seq_len']}；needle count 均匀采样于 {train['count_min']}–{train['count_max']}；needle 位置无放回采样；marker identity 从 10 类中有放回采样；64 类 noise token"},
      {'item':'词表','setting':"90 tokens：6 special + 64 noise + 10 marker + 10 count"},
      {'item':'训练','setting':f"一个模型混合训练两种格式；thinking_fraction={train['thinking_fraction']}；{train['train_steps']} steps；batch={train['batch_size']}；AdamW lr={train['lr']}；warmup={train['warmup_steps']}；weight decay={train['weight_decay']}；seed={train['seed']}"},
      {'item':'trace','setting':"trace_indices=false：thinking trace 只复制按 prompt 位置排序的 marker 序列 M1…Mn，不包含显式数字 index"},
      {'item':'评估','setting':f"每个 count {train['eval_examples_per_count']} 个样本，因此每种模式共 10,000 个；greedy autoregressive generation；所有结果都是 count 1–10、长度 256 的 ID 结果"},
  ], [('item','项目'),('setting','具体设定')])}
  <div class="format-grid">
    <div class="format"><b>Thinking / THINK_ON</b><div class="sequence">&lt;BOS&gt; <span class="on">&lt;THINK_ON&gt;</span> prompt &lt;Think/&gt; <span class="target">M1 … Mn &lt;/Think&gt; &lt;Cn&gt; &lt;EOS&gt;</span></div><p>监督 trace markers、关闭 token、最终 count 和 EOS。</p></div>
    <div class="format"><b>Non-thinking / THINK_OFF</b><div class="sequence">&lt;BOS&gt; <span class="off">&lt;THINK_OFF&gt;</span> prompt &lt;Think/&gt; <span class="target">&lt;/Think&gt; &lt;Cn&gt; &lt;EOS&gt;</span></div><p>监督立即关闭 thinking block、最终 count 和 EOS。</p></div>
  </div>
  <p>这里没有旧 v5 的 ambiguous-prefix mask 技巧：两个样本在 prompt 前已经分别带有不同的显式开关 token，因此相同的 <code>&lt;Think/&gt;</code> 表面前缀在完整上下文中并不相同。</p>

  <h2>2. 指标如何计算</h2>
  <div class="formula">final_accuracy = mean[ 1(predicted_count == gold_count) ]</div>
  <div class="formula">trace_exact = mean[ 1(generated_marker_sequence == gold_marker_sequence) ]</div>
  <p><b>Trace precision/recall：</b>先计算生成 marker 序列与 gold marker 序列的最长公共子序列长度 LCS；precision = LCS / generated length，recall = LCS / gold length。<b>Switch accuracy：</b>在完整 prompt 加 <code>&lt;Think/&gt;</code> 后只看下一个 token；THINK_ON 的目标是第一个 gold marker，THINK_OFF 的目标是 <code>&lt;/Think&gt;</code>。</p>
  {table(summary_rows,[('mode','模式'),('examples','final eval 样本'),('final','final accuracy'),('trace','trace exact'),('switch','switch argmax accuracy'),('probability','P(desired next token)')])}

  <h2>3. 学习过程：先学会开关，再学会稳定计数</h2>
  <div class="callout"><b>关键时间顺序。</b>step 500 时，THINK_OFF/THINK_ON 的 desired-next-token argmax accuracy 已分别为 {pct(weighted_mean([r for r in step_500_switch if r['mode']=='nonthinking'],'argmax_is_desired'))} 和 {pct(weighted_mean([r for r in step_500_switch if r['mode']=='thinking'],'argmax_is_desired'))}，但 final-count accuracy 仅为 {pct(weighted_mean([r for r in step_500_eval if r['mode']=='nonthinking'],'final_accuracy'))} 和 {pct(weighted_mean([r for r in step_500_eval if r['mode']=='thinking'],'final_accuracy'))}。因此“识别模式开关”明显早于“完成计数”。首次在一个 checkpoint 上达到全 count 100% 的时间为 THINK_OFF step {first_non}；THINK_ON 同时达到 final=100% 且 trace=100% 是 step {first_think}。</div>
  <div class="figure-grid">
    {figure(figures/'train_loss_by_step_and_mode.png','Figure 1. 训练 loss 分解','横轴是 training step；纵轴是对应监督 token 上的 next-token cross-entropy。total loss 按所有有效 label token 平均；四条 component loss 分别只在 thinking trace+close、thinking count、non-thinking close、non-thinking count 上平均。不同 component 的 token 数不同，因此绝对值主要用于看各自收敛，不宜直接解释为任务权重。',wide=True)}
    {figure(figures/'final_accuracy_by_step_mode.png','Figure 2. Final-count accuracy 随训练变化','横轴是 checkpoint step；纵轴是每种模式跨 count 1–10、每个 count 1000 个样本的 autoregressive final-count exact accuracy。蓝色 THINK_OFF，橙色 THINK_ON。中期存在明显波动，说明单个早期 checkpoint 不足以代表稳定性能。')}
    {figure(figures/'mode_switch_accuracy_by_step.png','Figure 3. 显式开关的首 token 路由准确率','横轴是 checkpoint step；纵轴是在 <Think/> 后 desired token 是否为 argmax 的比例。THINK_OFF 的 desired token 是 </Think>；THINK_ON 是 gold 第一个 marker。两条曲线几乎从 step 500 起就在 1.0，说明开关学习并非性能瓶颈。')}
  </div>
  <p>最终 training total loss 从 {fmt(first_train['loss_total'])} 降至 {fmt(last_train['loss_total'])}；non-thinking close loss 最终为 {fmt(last_train['loss_nonthinking_close'])}，thinking trace loss 为 {fmt(last_train['loss_thinking_trace'])}。最终 switch 概率非常尖锐：THINK_OFF 的 P(close)={fmt(non_switch['p_close_after_think'],6)}；THINK_ON 的 P(gold first marker)={fmt(think_switch['p_gold_first_marker_after_think'],6)}。</p>

  <h2>4. 最终行为：两种模式都饱和</h2>
  <div class="figure-grid">
    {figure(figures/'final_accuracy_by_count_mode.png','Figure 4. 按 gold count 分解的 final accuracy','横轴是 gold needle count 1–10；纵轴是 autoregressive final-count exact accuracy；颜色区分 THINK_OFF 与 THINK_ON。两条曲线在所有 count 上均为 1.0，因此当前 ID 设定没有产生 thinking 优势。')}
    {figure(figures/'trace_metrics_by_count.png','Figure 5. THINK_ON trace 质量','横轴是 gold count；纵轴是 trace 指标比例。trace_exact 检查完整 marker 序列逐 token 相等；precision/recall 基于 LCS；premature/missing close 是格式错误率。最终所有 count 的 trace exact、precision、recall 均为 1，格式错误率为 0。')}
  </div>
  {table(bin_rows,[('mode','模式'),('bin','count bin'),('range','count 范围'),('examples','样本数'),('accuracy','final accuracy'),('trace','trace exact')])}
  <div class="figure-grid">
    {figure(figures/'confusion_matrix_nonthinking.png','Figure 6a. THINK_OFF confusion matrix','横轴 predicted count，纵轴 gold count；每一行归一化为 1。最终矩阵完全位于对角线，说明没有系统性 under-count 或 over-count。')}
    {figure(figures/'confusion_matrix_thinking.png','Figure 6b. THINK_ON confusion matrix','横轴 predicted count，纵轴 gold count；每一行归一化为 1。最终矩阵完全位于对角线。注意这只说明最终答案正确，trace 正确性由 Figure 5 单独确认。')}
  </div>

  <h2>5. Attention：存在中等 retrieval 信号，但没有 v2 那种单一强 targeted head</h2>
  <p>对第 k 个 thinking trace marker query，定义：</p>
  <div class="formula">correct_top1 = 1[ prompt needle k 在所有 prompt needles 中获得最大 attention ]</div>
  <div class="formula">diagonal_dominance = A(query_k, needle_k) / sum_j A(query_k, needle_j)</div>
  <div class="formula">needle_mass = sum_j A(query_k, needle_j)</div>
  <p>这些值先在 trace step 与样本上平均。由于 count 1–10 平衡、每个样本贡献 count 个 trace query，随机地在本样本 needles 中选一个的加权 top-1 基线约为 {fmt(random_top1)}。最强 head {code(f"L{best_think['layer']}H{best_think['head']}")} 的 correct_top1={fmt(best_think['correct_top1'])}、diagonal dominance={fmt(best_think['diagonal_dominance'])}，高于随机基线但远低于 v2 中接近 1.0 的 index-token retrieval head；而它的 raw needle mass 只有 {fmt(best_think['needle_mass'])}，说明“在 needle 子集中会排序”并不等于“大部分注意力都投向正确 needle”。</p>
  <div class="figure-grid">
    {figure(figures/'attention_trace_to_prompt_best_head.png','Figure 7. 16 个 head 的 diagonal dominance','横轴 head 0–3，纵轴 layer 0–3；颜色和数字是 diagonal_dominance。它是“投向全部 prompt needles 的 attention 中，正确第 k 个 needle 占多少”的条件比例，不是 raw attention mass。L3 整体较高，但最高也只有约 0.39。')}
    {figure(figures/'mode_hidden_similarity.png','Figure 8. 两种模式在 close 位置的 hidden-state cosine','横轴是 hidden-state index：0 为 embedding 输出，1–4 为四层 Transformer 输出；纵轴是在各自 </Think> 位置上 THINK_ON 与 THINK_OFF hidden vector 的 cosine similarity，再对 5000 个相同 base examples 平均。中层约 0.55、末层升至 0.895，符合“中间路径分化、最终 readout 再汇合”的描述；但两种 close token 处于不同绝对位置，因此该指标也混合了 learned position embedding 差异。')}
  </div>
  {table(attention_table,[('head','thinking head'),('top1','correct top-1'),('dominance','diagonal dominance'),('needle','raw needle mass'),('entropy','prompt entropy')])}
  <div class="callout warn"><b>关于 needle/noise ratio。</b>部分 head 对 noise 的总 attention 接近 0，使 needle/noise ratio 达到极大值；这类比值数值不稳定，报告不把它当主证据。THINK_OFF 中 raw needle mass 最高的是 {code(f"L{best_non['layer']}H{best_non['head']}")}（{fmt(best_non['needle_mass'])}），更像对 needle 集合的 broad aggregation，而非 k-to-k retrieval。</div>

  <h2>6. Probe：count 可读出，但多数结果不是机制证明</h2>
  <p>每个 anchor/layer 上用 70% hidden vectors 训练、30% 测试。分类使用标准化 hidden state 上的 nearest-centroid classifier；回归使用 ridge，报告 accuracy、R² 与 MAE。position baseline 和 trace-length lookup baseline 使用同样 70/30 切分。</p>
  {table(probe_table,[('mode','模式'),('anchor','anchor'),('layer','最佳 layer'),('accuracy','probe acc'),('r2','ridge R²'),('mae','ridge MAE'),('position','position baseline'),('trace_len','trace-length baseline')])}
  <p><code>mode_pos</code> 处 final count 基本不可读（accuracy 约 0.114），说明开关 token 本身没有泄漏 count。到 prompt 结束的 <code>think_open_pos</code>，THINK_OFF 最佳分类 accuracy={fmt(best_probe(probe_rows,'nonthinking','think_open_pos')['accuracy'])}，THINK_ON={fmt(best_probe(probe_rows,'thinking','think_open_pos')['accuracy'])}，说明 prompt processing 已形成可读 count 信息。到 <code>pre_count_pos</code> 两者均可达到 1.0。</p>
  <div class="callout warn"><b>限制。</b>thinking 的 close/pre-count 绝对位置等于 prompt length + trace length，而 trace length 又等于 count，因此 position/trace-length baseline 可直接达到 1.0；<code>count_pos</code> 还已经看到了答案 token。故这些高 probe 数字只能说明“信息可读”，不能定位 causal counter，也不能证明某条 direction 可用于 steering。</div>

  <h2>7. 目前支持的解释与缺失证据</h2>
  <div class="callout good"><b>结果支持：</b><ul><li>一个 Transformer 可以通过显式 THINK_ON/OFF token 学会两种稳定、可自由生成的输出路径。</li><li>开关路由非常容易，早于计数能力成熟；旧 v5 retrieval 不显著不能归因于“开关没学会”。</li><li>THINK_ON 的中间表示与 THINK_OFF 明显不同，但最终层重新接近，符合模式特异 computation 接到共享 count readout 的图景。</li><li>THINK_ON attention 有弱到中等 k-to-k retrieval 结构；THINK_OFF 有较强的 broad needle-set attention。</li></ul></div>
  <div class="callout warn"><b>结果尚不支持：</b><ul><li>不支持 thinking 提高 ID accuracy：两种模式都饱和为 100%。</li><li>不支持单一 targeted head 构成完整 counting circuit：最强 correct_top1 只有 {fmt(best_think['correct_top1'])}，且 raw needle mass 很低。</li><li>不支持 OOD generalization；没有测试更长序列、更多 needles、未见 count 或开关扰动。</li><li>attention 与 probe 都不是因果证据。下一步应对 THINK_ON/OFF embedding、L3 heads、broad-aggregation heads 做 ablation/patching，并测试交换开关 token 后行为是否相应翻转。</li></ul></div>

  <h2>8. 复现文件</h2>
  {table([
      {'item':'config','path':code(run_dir/'config.json')},
      {'item':'final checkpoint','path':code(run_dir/'checkpoints'/'final.pt')},
      {'item':'tables','path':code(tables)},
      {'item':'figures','path':code(figures)},
      {'item':'report builder','path':code('scripts/build_v5_explicit_switch_report.py')},
  ],[('item','文件'),('path','路径')])}
  <p class="small">本 HTML 中的全部 PNG 均以 base64 内嵌，可单文件发送和离线查看。报告由原始 CSV 聚合生成，没有把 teacher-forced accuracy 混入主行为结论。</p>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    output = args.out or args.run_dir / "report.html"
    output.write_text(build_report(args.run_dir), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
