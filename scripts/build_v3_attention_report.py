from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


KEY_CATEGORY_COLUMNS = [
    "correct_prompt_needle",
    "other_prompt_needles",
    "prompt_noise",
    "previous_index_token",
    "previous_marker_token",
    "earlier_trace_tokens",
    "think_open",
    "bos",
    "current_index_self",
    "current_marker_self",
    "other_context",
]


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    try:
        parsed = float(str(value))
        return f"{parsed:.{digits}f}"
    except ValueError:
        return str(value)


def html_table(rows: list[dict[str, object]], columns: list[tuple[str, str]], *, max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _label in columns:
            value = row.get(key, "")
            cells.append(f"<td>{html.escape(fmt(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def image_block(path: Path, title: str, caption: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"""
    <figure>
      <h3>{html.escape(title)}</h3>
      <img src="data:image/png;base64,{data}" alt="{html.escape(title)}">
      <figcaption>{caption}</figcaption>
    </figure>
    """


def summarize(result_dir: Path) -> dict[str, object]:
    tables = result_dir / "analysis" / "tables"
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    head_summary = read_rows(tables / "head_summary.csv")
    last_index = read_rows(tables / "last_index_head_summary.csv")
    ablation = read_rows(tables / "head_ablation_results.csv")

    token_path = tables / "token_attention_rows.csv"
    n_token_rows = 0
    counts = Counter()
    trace_items = Counter()
    key_heads = {("3", "3"), ("3", "1"), ("2", "3"), ("4", "2")}
    category_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_bin: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    top_roles: dict[str, Counter[str]] = defaultdict(Counter)
    seen_examples = set()

    with token_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_token_rows += 1
            if row["query_anchor"] == "index_token_k" and row["layer"] == "3" and row["head"] == "3":
                trace_items[int(row["count"])] += 1
                if row["example_id"] not in seen_examples:
                    counts[int(row["count"])] += 1
                    seen_examples.add(row["example_id"])
            if row["query_anchor"] != "index_token_k" or row["is_last_index"] != "True":
                continue
            lh = (row["layer"], row["head"])
            if lh not in key_heads:
                continue
            label = f"L{row['layer']}H{row['head']}"
            for col in KEY_CATEGORY_COLUMNS:
                value = _float(row[col])
                category_values[label][col].append(value)
                by_bin[(label, row["count_bin"])][col].append(value)
            top_roles[label][row["top_role"]] += 1

    top_last_retrieval = sorted(
        last_index,
        key=lambda r: _float(r["correct_prompt_needle_mass"]),
        reverse=True,
    )[:8]
    top_last_plus = sorted(
        last_index,
        key=lambda r: _float(r["plus_one_score"]),
        reverse=True,
    )[:8]

    category_rows = []
    for label in ["L3H3", "L3H1", "L2H3", "L4H2"]:
        values = {
            col: fmean(v)
            for col, v in category_values[label].items()
            if v
        }
        category_rows.append({"head": label, **values})

    bin_rows = []
    for label in ["L3H3", "L2H3"]:
        for count_bin in ["low", "mid", "high"]:
            values = by_bin[(label, count_bin)]
            bin_rows.append(
                {
                    "head": label,
                    "count_bin": count_bin,
                    "correct_prompt_needle": fmean(values["correct_prompt_needle"]),
                    "previous_marker_token": fmean(values["previous_marker_token"]),
                    "previous_index_token": fmean(values["previous_index_token"]),
                    "prompt_noise": fmean(values["prompt_noise"]),
                }
            )

    for row in head_summary:
        row["head_id"] = f"L{row['layer']}H{row['head']}"
    for row in last_index:
        row["head_id"] = f"L{row['layer']}H{row['head']}"

    return {
        "manifest": manifest,
        "n_token_rows": n_token_rows,
        "n_attention_examples": sum(counts.values()),
        "counts": dict(sorted(counts.items())),
        "trace_items": dict(sorted(trace_items.items())),
        "head_summary": head_summary,
        "last_index": last_index,
        "ablation": ablation,
        "top_last_retrieval": top_last_retrieval,
        "top_last_plus": top_last_plus,
        "category_rows": category_rows,
        "bin_rows": bin_rows,
        "top_roles": {
            key: ", ".join(f"{role}: {count}" for role, count in counter.most_common(5))
            for key, counter in top_roles.items()
        },
    }


def build_report(result_dir: Path) -> str:
    summary = summarize(result_dir)
    manifest = summary["manifest"]
    figs = result_dir / "analysis" / "figures"

    top_retrieval = summary["top_last_retrieval"][0]
    top_plus = summary["top_last_plus"][0]
    ablation = summary["ablation"]
    baseline = next(row for row in ablation if row["condition"] == "baseline_no_ablation")
    l3h3_ablation = next(row for row in ablation if row["condition"] == "best_retrieval_L3H3")
    top2_ablation = next(row for row in ablation if row["condition"] == "top2_retrieval_L3H3_L3H1")
    plus_ablation = next(row for row in ablation if row["condition"] == "best_plus_one_L2H3")

    top_retrieval_table = [
        {
            "head": f"L{r['layer']}H{r['head']}",
            "correct_prompt_needle_mass": r["correct_prompt_needle_mass"],
            "all_prompt_needles_mass": r["all_prompt_needles_mass"],
            "correct_prompt_needle_top1": r["correct_prompt_needle_top1"],
            "previous_prompt_needle_mass": r["previous_prompt_needle_mass"],
            "plus_one_score": r["plus_one_score"],
        }
        for r in summary["top_last_retrieval"]
    ]
    top_plus_table = [
        {
            "head": f"L{r['layer']}H{r['head']}",
            "plus_one_score": r["plus_one_score"],
            "previous_index_token_mass": r["previous_index_token_mass"],
            "previous_marker_token_mass": r["previous_marker_token_mass"],
            "correct_prompt_needle_mass": r["correct_prompt_needle_mass"],
            "correct_prompt_needle_top1": r["correct_prompt_needle_top1"],
        }
        for r in summary["top_last_plus"]
    ]
    category_rows = summary["category_rows"]
    for row in category_rows:
        row["top_roles"] = summary["top_roles"].get(row["head"], "")

    ablation_rows = [
        {
            "condition": row["condition"],
            "heads": row["heads"],
            "accuracy": row["accuracy"],
            "invalid_rate": row["invalid_rate"],
            "trace_exact_match_rate": row["trace_exact_match_rate"],
            "trace_marker_recall": row["trace_marker_recall"],
            "trace_index_accuracy": row["trace_index_accuracy"],
            "n_examples": row["n_examples"],
        }
        for row in ablation
    ]

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trace Count v3 Attention Deep-Dive Report</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5b6578;
      --line: #d9deea;
      --band: #f5f7fb;
      --accent: #2454d6;
      --accent-soft: #e8eeff;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif;
      color: var(--ink);
      background: #fff;
      line-height: 1.62;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 36px 28px 72px;
    }}
    h1 {{
      font-size: 34px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    h2 {{
      margin-top: 42px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
      font-size: 24px;
    }}
    h3 {{
      font-size: 18px;
      margin: 0 0 12px;
    }}
    p, li {{
      font-size: 16px;
    }}
    code {{
      background: #eef1f7;
      padding: 1px 5px;
      border-radius: 4px;
      font-family: "SFMono-Regular", Consolas, monospace;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 16px;
      margin-bottom: 22px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 22px 0 28px;
    }}
    .card {{
      background: var(--band);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px 16px;
    }}
    .card .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .card .value {{
      font-weight: 700;
      font-size: 20px;
    }}
    .callout {{
      background: var(--accent-soft);
      border-left: 5px solid var(--accent);
      padding: 14px 18px;
      border-radius: 8px;
      margin: 20px 0;
    }}
    .grid2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    figure {{
      margin: 22px 0;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
    }}
    img {{
      width: 100%;
      max-height: 760px;
      object-fit: contain;
      display: block;
      margin: 0 auto;
    }}
    figcaption {{
      color: var(--muted);
      font-size: 14px;
      margin-top: 12px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 14px 0 24px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      vertical-align: top;
    }}
    th {{
      background: var(--band);
      text-align: left;
    }}
    .small {{
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 860px) {{
      .summary, .grid2 {{
        grid-template-columns: 1fr;
      }}
      main {{
        padding: 24px 16px 56px;
      }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Trace Count v3: v2 Attention Head Deep-Dive</h1>
  <div class="subtitle">问题：v2 indexed-CoT 中，最后一个 trace 数字 token 是在检索最后一个 prompt needle，还是主要依赖上一个 trace 数字/marker 做局部 +1？</div>

  <div class="summary">
    <div class="card"><div class="label">模型来源</div><div class="value">v2 thinking</div></div>
    <div class="card"><div class="label">attention examples</div><div class="value">{summary["n_attention_examples"]}</div></div>
    <div class="card"><div class="label">每个 count</div><div class="value">{manifest.get("attention_examples_per_count")}</div></div>
    <div class="card"><div class="label">token-level rows</div><div class="value">{summary["n_token_rows"]:,}</div></div>
  </div>

  <div class="callout">
    <b>核心结论。</b>
    对最后一个 trace index token 而言，最强 head 是 <b>L3H3</b>：
    它平均把 <b>{fmt(top_retrieval["correct_prompt_needle_mass"])}</b> 的 attention 放在最后一个 prompt needle 上，
    top-1 retrieval rate 为 <b>{fmt(top_retrieval["correct_prompt_needle_top1"])}</b>。
    它对 previous index + previous marker 的局部 trace attention 只有 <b>{fmt(_float(top_retrieval["plus_one_score"]))}</b>。
    因此，这个结果不支持“最后数字只是在看上一个数字然后 +1”的简单解释；更合理的描述是：
    <b>v2 thinking trace 至少包含一个非常强的 targeted retrieval head，最后一个 trace 数字会直接指向最后一个 prompt needle。</b>
  </div>

  <h2>1. 实验设置与度量定义</h2>
  <p>
    本报告分析的是已经训练好的 v2 marker-trace thinking model。v2 的 thinking 格式是：
    <code>&lt;Think/&gt; &lt;1&gt; marker_1 &lt;2&gt; marker_2 ... &lt;n&gt; marker_n &lt;/Think&gt; &lt;Ans&gt; &lt;n&gt;</code>。
    本次 v3 deep-dive 不重新训练模型，而是在 teacher-forced gold trace 上抽取 attention。
  </p>
  <p>
    数据来自独立 attention pool：count = 1..10，每个 count 有 {manifest.get("attention_examples_per_count")} 个样本，总计 {summary["n_attention_examples"]} 个样本。
    对每个样本、每个 trace item <code>k</code>、每个 query anchor、每个 layer/head 记录一行 attention 统计，所以总共有 {summary["n_token_rows"]:,} 行 token-level 明细。
    本报告尤其关注 <code>query_anchor = index_token_k</code> 且 <code>k = n</code> 的最后一个 trace index token。
  </p>
  <p>
    关键度量如下。
    <code>correct_prompt_needle_mass</code> 是 query token 对对应 prompt needle 的 attention mass；在最后 index 的分析里，它就是对最后一个 prompt needle 的 mass。
    <code>correct_prompt_needle_top1</code> 表示在所有 prompt needles 中，attention 最大的 needle 是否就是正确 needle。
    <code>plus_one_score</code> 定义为 attention 到 previous index token 与 previous marker token 的和，用来近似“局部 trace / +1”路径。
    注意：attention 是诊断信号，不等同于因果机制；因果解释需要结合 ablation/path patching。
  </p>

  <h2>2. 最后 trace 数字：retrieval head vs local +1 head</h2>
  {image_block(
        figs / "last_index_retrieval_vs_plus_one_heatmaps.png",
        "Figure 1. 最后 trace 数字的 retrieval mass 与 plus-one/local-trace mass",
        "左图横轴是 head，纵轴是 layer，颜色是最后一个 trace index token 对最后一个 prompt needle 的平均 attention mass。右图颜色是同一 query 对 previous index + previous marker 的 attention mass。L3H3 在左图显著最高，L2H3 在右图显著最高，说明 retrieval 与局部 trace 不是同一个 head。"
    )}
  <p>
    最后一位 index token 上，retrieval 最强的是 L3H3，平均对最后 prompt needle 的 mass 为 {fmt(top_retrieval["correct_prompt_needle_mass"])}，
    对所有 prompt needles 的总 mass 为 {fmt(top_retrieval["all_prompt_needles_mass"])}，并且 top-1 retrieval rate = {fmt(top_retrieval["correct_prompt_needle_top1"])}。
    第二强 retrieval head 是 L3H1，correct prompt needle mass = {fmt(summary["top_last_retrieval"][1]["correct_prompt_needle_mass"])}，top-1 = {fmt(summary["top_last_retrieval"][1]["correct_prompt_needle_top1"])}。
  </p>
  <p>
    相比之下，最强 plus-one/local-trace head 是 L{top_plus["layer"]}H{top_plus["head"]}，
    plus-one score = {fmt(top_plus["plus_one_score"])}，主要来自 previous marker mass = {fmt(top_plus["previous_marker_token_mass"])}。
    这个 head 同时也有一定 retrieval 成分，但对最后 prompt needle 的 mass 只有 {fmt(top_plus["correct_prompt_needle_mass"])}。
  </p>

  <div class="grid2">
    <section>
      <h3>Top last-index retrieval heads</h3>
      {html_table(top_retrieval_table, [
        ("head", "head"),
        ("correct_prompt_needle_mass", "correct needle mass"),
        ("all_prompt_needles_mass", "all needle mass"),
        ("correct_prompt_needle_top1", "top-1"),
        ("previous_prompt_needle_mass", "previous prompt needle"),
        ("plus_one_score", "plus-one score"),
      ])}
    </section>
    <section>
      <h3>Top last-index plus-one/local-trace heads</h3>
      {html_table(top_plus_table, [
        ("head", "head"),
        ("plus_one_score", "plus-one score"),
        ("previous_index_token_mass", "prev index"),
        ("previous_marker_token_mass", "prev marker"),
        ("correct_prompt_needle_mass", "correct needle mass"),
        ("correct_prompt_needle_top1", "top-1"),
      ])}
    </section>
  </div>

  {image_block(
        figs / "last_index_retrieval_vs_plus_one_scatter.png",
        "Figure 2. 每个 head 的 retrieval score 与 plus-one score",
        "每个点是一个 layer/head。横轴是 previous index + previous marker attention，纵轴是最后 prompt needle attention。L3H3 与 L3H1 位于高 retrieval、低 plus-one 区域；L2H3 位于高 plus-one、较低 retrieval 区域。该图说明两种注意力模式可以分离。"
    )}

  <h2>3. L3H3 到底在看什么 token 类别？</h2>
  {image_block(
        figs / "last_index_category_mass_bars.png",
        "Figure 3. 最后 index token 的 attention mass 按 token 类别分解",
        "左图只看最佳 retrieval head L3H3，右图对所有 layer/head 平均。L3H3 的 mass 高度集中在 correct_prompt_needle；但所有 head 平均主要分散在 prompt_noise，因此不能把整个模型的 attention 简化为单一 retrieval 机制。"
    )}
  <p>
    对 L3H3 来说，最后 index token 的 attention 类别分布非常集中：
    correct prompt needle = {fmt(next(r for r in category_rows if r["head"] == "L3H3")["correct_prompt_needle"])},
    other prompt needles = {fmt(next(r for r in category_rows if r["head"] == "L3H3")["other_prompt_needles"])},
    prompt noise = {fmt(next(r for r in category_rows if r["head"] == "L3H3")["prompt_noise"])},
    previous marker = {fmt(next(r for r in category_rows if r["head"] == "L3H3")["previous_marker_token"])},
    previous index = {fmt(next(r for r in category_rows if r["head"] == "L3H3")["previous_index_token"])}。
    这基本排除了 L3H3 主要做 local +1 的解释。
  </p>
  {html_table(category_rows, [
    ("head", "head"),
    ("correct_prompt_needle", "correct needle"),
    ("other_prompt_needles", "other needles"),
    ("prompt_noise", "prompt noise"),
    ("previous_index_token", "prev index"),
    ("previous_marker_token", "prev marker"),
    ("earlier_trace_tokens", "earlier trace"),
    ("top_roles", "top role examples"),
  ])}

  <h3>按 count-bin 看 L3H3 与 L2H3</h3>
  <p>
    L3H3 在 low/mid/high counts 上都保持高 retrieval mass；L2H3 的 previous-marker mass 随 count 增大而明显增强。
    这说明长 trace 中确实有更强的 local trace tracking，但它与 L3H3 的 prompt retrieval 机制并存。
  </p>
  {html_table(summary["bin_rows"], [
    ("head", "head"),
    ("count_bin", "count bin"),
    ("correct_prompt_needle", "correct needle"),
    ("previous_marker_token", "prev marker"),
    ("previous_index_token", "prev index"),
    ("prompt_noise", "prompt noise"),
  ])}

  <h2>4. Head ablation：这些 head 是否是单点因果瓶颈？</h2>
  {image_block(
        figs / "head_ablation_results.png",
        "Figure 4. Autoregressive thinking under head ablation",
        "横轴是 ablation condition，纵轴是行为指标。baseline、ablate L3H3、ablate L2H3/control 都几乎不掉；同时 ablate L3H3+L3H1 会让 trace exact 从 1.000 降到 0.955，accuracy 仍为 0.995。"
    )}
  {html_table(ablation_rows, [
    ("condition", "condition"),
    ("heads", "ablated heads"),
    ("accuracy", "answer acc"),
    ("invalid_rate", "invalid"),
    ("trace_exact_match_rate", "trace exact"),
    ("trace_marker_recall", "marker recall"),
    ("trace_index_accuracy", "index acc"),
    ("n_examples", "n"),
  ])}
  <p>
    Ablation 的结论要谨慎：L3H3 是一个非常清晰的 retrieval-like attention head，但单独消掉它后，
    final accuracy 仍为 {fmt(l3h3_ablation["accuracy"])}，trace exact 只是从 baseline 的 {fmt(baseline["trace_exact_match_rate"])}
    轻微降到 {fmt(l3h3_ablation["trace_exact_match_rate"])}。同时消掉 L3H3 和 L3H1 后，
    trace exact 降到 {fmt(top2_ablation["trace_exact_match_rate"])}，final accuracy 为 {fmt(top2_ablation["accuracy"])}。
    这说明 retrieval heads 可能是冗余的、可替代的，或者模型也能通过 MLP/residual/其他 heads 补偿。
  </p>
  <p>
    消掉 plus-one 最强 head L2H3 后，accuracy 和 trace exact 都维持在 {fmt(plus_ablation["accuracy"])} / {fmt(plus_ablation["trace_exact_match_rate"])}。
    因此，目前证据不支持“L2H3 这样的 previous-token head 是最终计数输出的必要因果通路”。
  </p>

  <h2>5. 对原问题的回答</h2>
  <div class="callout">
    <p><b>最后一个计数 token 的 attention 主要看哪里？</b></p>
    <p>
      在最强且最清晰的 head L3H3 中，它主要看最后一个 prompt needle，而不是看上一个 trace 数字或 marker。
      具体数值是：correct final prompt needle mass {fmt(top_retrieval["correct_prompt_needle_mass"])};
      previous index mass {fmt(top_retrieval["previous_index_token_mass"])};
      previous marker mass {fmt(top_retrieval["previous_marker_token_mass"])}。
    </p>
    <p>
      但模型不是只有这一条路径。L2H3/L4H2 等 head 显示了局部 trace attention，尤其 L2H3 会看 previous marker。
      所以更精确的说法是：<b>v2 thinking model 同时有 targeted retrieval heads 和 local trace heads；最后一个 index 的最显著 attention 证据偏向 targeted retrieval，而不是纯 +1。</b>
    </p>
  </div>

  <h2>6. 局限与下一步实验</h2>
  <ul>
    <li><b>Attention 不等于因果。</b> L3H3 的 attention pattern 很强，但单 head ablation 影响小，说明不能只凭 attention claim “模型必须依赖它”。</li>
    <li><b>需要更强的 causal tests。</b> 下一步可以做 multi-head retrieval ablation、path patching、只 patch L3H3 output/value、或在最后 index token 位置 patch attention output，看 final answer 和 trace 是否系统变化。</li>
    <li><b>需要与 v6 separator trace 对比。</b> v6 去掉 index token 后，如果仍出现类似 sep-token-to-prompt-needle 的 diagonal retrieval，那么说明 v2 的 retrieval 不只是 numeric index leakage。</li>
    <li><b>需要 counterfactual prompt。</b> 例如交换最后两个 prompt needles 的 marker/type/position，检查 L3H3 是否跟位置上的最后 needle 走，而不是跟 trace token identity 走。</li>
  </ul>

  <h2>7. 文件与复现信息</h2>
  <table>
    <tbody>
      <tr><th>result_dir</th><td><code>{html.escape(str(result_dir))}</code></td></tr>
      <tr><th>source_v2_run_dir</th><td><code>{html.escape(str(manifest.get("source_v2_run_dir", "")))}</code></td></tr>
      <tr><th>thinking_model_dir</th><td><code>{html.escape(str(manifest.get("thinking_model_dir", "")))}</code></td></tr>
      <tr><th>analysis_dir</th><td><code>{html.escape(str(manifest.get("analysis_dir", "")))}</code></td></tr>
      <tr><th>saved_at</th><td><code>{html.escape(str(manifest.get("saved_at", "")))}</code></td></tr>
    </tbody>
  </table>
  <p class="small">
    Generated from CSV tables in <code>analysis/tables/</code>. This report intentionally uses cautious language:
    the attention patterns are strong mechanistic diagnostics, while ablation results suggest redundancy rather than a single-head causal bottleneck.
  </p>
</main>
</body>
</html>
"""
    return html_doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HTML report for v3 v2 attention deep-dive results.")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result_dir = args.result_dir
    out = args.out or result_dir / "report.html"
    out.write_text(build_report(result_dir), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
