from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if abs(number) < 1e-4 and number != 0:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def image_data(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def table(headers: list[str], rows: Iterable[Iterable[Any]], classes: str = "") -> str:
    header = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table class="{classes}"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def head_name(row: dict[str, str]) -> str:
    return f"L{int(float(row['layer']))}H{int(float(row['head']))}"


def find_source_run(result_dir: Path, explicit: str | None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            result_dir.parent / "v5_synthetic_niah_v5" / "v5",
            result_dir.parent / "v5",
            result_dir.parent,
        ]
    )
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "vocab.json").is_file():
            return candidate
    return None


def aggregate_by_count(
    path: Path,
    selected: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int, int], dict[str, float]], int, float]:
    sums: dict[tuple[int, int, int], dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "top1": 0.0, "mass": 0.0, "all_mass": 0.0, "noise": 0.0}
    )
    total_rows = 0
    chance_sum = 0.0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            count = int(row["count"])
            chance_sum += 1.0 / count
            key2 = (int(row["layer"]), int(row["head"]))
            if key2 not in selected:
                continue
            item = sums[(key2[0], key2[1], count)]
            item["n"] += 1
            item["top1"] += float(row["correct_top1"])
            item["mass"] += float(row["correct_prompt_needle_mass"])
            item["all_mass"] += float(row["all_prompt_needles_mass"])
            item["noise"] += float(row["prompt_noise_mass"])
    return sums, total_rows, chance_sum / max(total_rows, 1)


def mean_item(item: dict[str, float], name: str) -> float:
    return item[name] / max(item["n"], 1.0)


def build_report(result_dir: Path, source_run: Path | None, output: Path) -> None:
    tables = result_dir / "tables"
    figures = result_dir / "figures"
    switch = read_csv(tables / "switch_summary.csv")
    pred = read_csv(tables / "prediction_query_head_summary.csv")
    post = read_csv(tables / "post_marker_head_summary.csv")
    hidden = read_csv(tables / "hidden_norm_summary.csv")

    config: dict[str, Any] = {}
    final_eval: list[dict[str, str]] = []
    if source_run:
        config = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
        eval_path = source_run / "tables" / "eval_by_step.csv"
        if eval_path.is_file():
            eval_rows = read_csv(eval_path)
            max_step = max(int(row["step"]) for row in eval_rows)
            final_eval = [row for row in eval_rows if int(row["step"]) == max_step]

    train = config.get("train", {})
    model = config.get("model", {})
    selected = {(2, 0), (2, 2), (3, 2)}
    by_count, retrieval_row_count, weighted_chance = aggregate_by_count(
        tables / "prediction_query_rows.csv", selected
    )
    post_row_count = sum(1 for _ in (tables / "post_marker_rows.csv").open("r", encoding="utf-8")) - 1
    switch_example_rows = sum(1 for _ in (tables / "switch_examples.csv").open("r", encoding="utf-8")) - 1

    pred_sorted = sorted(pred, key=lambda row: float(row["correct_prompt_needle_mass"]), reverse=True)
    pred_lookup = {(int(row["layer"]), int(row["head"])): row for row in pred}
    post_lookup = {(int(row["layer"]), int(row["head"])): row for row in post}
    switch_lookup = {row["variant"]: row for row in switch}

    head_rows = []
    interpretations = {
        (2, 2): "最纯的 targeted retrieval：几乎全部 attention 直接落在第 k 个 prompt needle。",
        (2, 0): "第二个强 targeted head；count=2 时略弱，其余 count 接近确定性。",
        (3, 2): "跨 count 稳定的 targeted head；仍有少量质量分配到其他位置。",
        (3, 1): "needle-set routing：几乎全部看 needles，但只约一半落在当前第 k 个。",
        (3, 3): "较宽的 needle attention；更像集合级聚合/路由，而非单点检索。",
        (3, 0): "top-1 看似较高但总 needle mass 仅 0.007；不能据此称为 retrieval head。",
    }
    for key in [(2, 2), (2, 0), (3, 2), (3, 1), (3, 3), (3, 0)]:
        row = pred_lookup[key]
        head_rows.append(
            [
                f"<strong>{head_name(row)}</strong>",
                f(row["correct_top1"]),
                f(row["correct_prompt_needle_mass"]),
                f(row["all_prompt_needles_mass"]),
                f(row["prompt_noise_mass"]),
                f(row["bos_mass"]),
                html.escape(interpretations[key]),
            ]
        )

    count_rows = []
    for count in range(1, 11):
        cells: list[Any] = [str(count), f(1.0 / count)]
        for layer, head in [(2, 2), (2, 0), (3, 2)]:
            item = by_count[(layer, head, count)]
            cells.extend([f(mean_item(item, "top1")), f(mean_item(item, "mass"))])
        count_rows.append(cells)

    behavior_rows = []
    if final_eval:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in final_eval:
            grouped[row["mode"]].append(row)
        for mode_name in ["nonthinking", "thinking"]:
            rows = grouped.get(mode_name, [])
            if rows:
                behavior_rows.append(
                    [
                        mode_name,
                        str(sum(int(row["n_examples"]) for row in rows)),
                        f(sum(float(row["final_accuracy"]) for row in rows) / len(rows)),
                        f(sum(float(row["final_mae"]) for row in rows) / len(rows)),
                        f(sum(float(row["trace_exact"]) for row in rows) / len(rows)) if mode_name == "thinking" else "N/A",
                        html.escape("强制给定 </Think> 后读 count")
                        if mode_name == "nonthinking"
                        else html.escape("从 <Think/> 后自回归生成 trace"),
                    ]
                )

    image_names = [
        "switch_probability_summary.png",
        "prediction_query_correct_top1.png",
        "prediction_query_correct_mass.png",
        "post_marker_correct_top1.png",
        "prediction_query_marker_margin.png",
    ]
    images = {name: image_data(figures / name) for name in image_names}

    think_switch = switch_lookup.get("thinking", switch[0])
    non_switch = switch_lookup.get("nonthinking", switch[0])
    l2h2 = pred_lookup[(2, 2)]
    l2h0 = pred_lookup[(2, 0)]
    l3h2 = pred_lookup[(3, 2)]
    l2h2_post = post_lookup[(2, 2)]
    l3h2_post = post_lookup[(3, 2)]
    hidden_rows = [
        [row["variant"], row["anchor"], f(row["hidden_norm"])] for row in hidden
    ]

    setting_rows = [
        ["模型", f"随机初始化 GPT-2；{model.get('n_layer', 4)} layers × {model.get('n_head', 4)} heads；d_model={model.get('n_embd', 256)}；MLP={model.get('n_inner', 1024)}；learned absolute position embeddings"],
        ["上下文与词表", f"prompt body 长度 {train.get('seq_len', 256)}；context window {model.get('n_positions', 384)}；vocab=88（64 noise、10 marker、10 count、4 special）"],
        ["数据", f"每个 prompt 含 1–10 个 needle；位置均匀无放回采样；marker identity 从 10 类中有放回采样，因此不同 needle 可同 token"],
        ["单模型混合训练", f"thinking_fraction={train.get('thinking_fraction', 0.5)}；每个样本随机选择 thinking 或 non-thinking 格式；不是两个模型"],
        ["训练", f"{train.get('train_steps', 10000)} steps；batch={train.get('batch_size', 128)}；AdamW lr={train.get('lr', 0.0003)}；warmup={train.get('warmup_steps', 500)}；weight decay={train.get('weight_decay', 0.01)}；seed={train.get('seed', 1234)}"],
        ["trace 格式", "trace_indices=False：thinking trace 只有 marker 序列 M1…Mn，没有显式 <I_k> index token"],
        ["诊断集", f"100 examples/count，共 1000 prompts；5500 trace prediction queries；16 heads，因此 prediction-query={retrieval_row_count:,} 行、post-marker={post_row_count:,} 行；switch={switch_example_rows:,} 行"],
    ]

    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v5.2 Mixed Thinking Toggle：Switch 与 Retrieval 诊断</title>
<style>
:root{{--ink:#172033;--muted:#536176;--line:#dce3ec;--soft:#f5f7fa;--blue:#2563eb;--green:#16834a;--red:#c83b3b;--amber:#b4690e}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);font-family:"Segoe UI","Microsoft YaHei",Arial,sans-serif;line-height:1.62;background:white}}
main{{max-width:1240px;margin:0 auto;padding:38px 32px 80px}} h1{{font-size:34px;line-height:1.18;margin:0 0 10px;letter-spacing:0}} h2{{font-size:25px;margin:48px 0 14px;padding-top:16px;border-top:1px solid var(--line);letter-spacing:0}} h3{{font-size:19px;margin:24px 0 8px;letter-spacing:0}} p{{margin:9px 0}} code{{background:#edf1f5;padding:2px 5px;border-radius:4px;font-family:Consolas,monospace}} .lede{{font-size:17px;color:var(--muted);max-width:960px}}
.meta{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0 26px}} .pill{{font-size:13px;border:1px solid var(--line);padding:5px 9px;border-radius:6px;background:var(--soft)}}
.callout{{border-left:5px solid var(--blue);padding:14px 17px;margin:18px 0;background:#f1f6ff}} .callout.good{{border-color:var(--green);background:#effaf3}} .callout.warn{{border-color:var(--amber);background:#fff8e8}} .callout.bad{{border-color:var(--red);background:#fff3f2}}
.summary-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}} .summary{{border:1px solid var(--line);border-radius:6px;padding:16px;background:white}} .summary strong{{display:block;font-size:28px;color:var(--blue);margin-bottom:3px}} .summary span{{color:var(--muted);font-size:14px}}
.sequence{{font-family:Consolas,monospace;background:#f7f9fc;border:1px solid var(--line);padding:12px 14px;margin:8px 0;overflow-x:auto;white-space:nowrap}} .supervised{{color:#a32020;font-weight:700}} .unsupervised{{color:#687386}}
.figure{{margin:20px 0 30px}} .figure img{{display:block;width:100%;height:auto;max-height:610px;object-fit:contain;border:1px solid var(--line);background:white}} .figure figcaption{{font-size:14px;color:var(--muted);margin-top:9px}} .figure-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}} .figure-grid .figure{{margin:8px 0 22px}}
.table-wrap{{overflow-x:auto;margin:12px 0 20px}} table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}} th{{background:#eef2f7;white-space:nowrap}} tbody tr:nth-child(even){{background:#fafbfd}} .compact td,.compact th{{padding:6px 8px}} .num{{font-variant-numeric:tabular-nums}}
.formula{{border:1px solid var(--line);padding:10px 12px;margin:8px 0;background:#fafbfd}} .small{{font-size:13px;color:var(--muted)}} ul{{padding-left:22px}} li{{margin:6px 0}} footer{{margin-top:52px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}}
@media(max-width:850px){{main{{padding:26px 16px 60px}} .summary-grid,.figure-grid{{grid-template-columns:1fr}} h1{{font-size:28px}}}}
@media print{{main{{max-width:none;padding:18px}} .figure-grid{{grid-template-columns:1fr 1fr}} .figure img{{max-height:450px}}}}
</style></head><body><main>
<h1>v5.2：Mixed Thinking Toggle 的 switch 与 targeted retrieval 诊断</h1>
<p class="lede">这份报告回答两个容易混淆的问题：同一个 Transformer 是否真的学会了在 thinking / non-thinking 之间主动切换；以及 marker-only CoT 是否形成了针对第 k 个 prompt needle 的检索头。全部图像以内嵌方式保存，报告可离线单文件转发。</p>
<div class="meta"><span class="pill">checkpoint step 10,000</span><span class="pill">seed 1234</span><span class="pill">100 examples/count</span><span class="pill">4 layers × 4 heads</span><span class="pill">trace_indices=False</span></div>

<div class="summary-grid">
  <div class="summary"><strong>{f(think_switch['p_any_trace_start_after_think_open'],6)}</strong><span><code>&lt;Think/&gt;</code> 后生成任一 marker 的概率</span></div>
  <div class="summary"><strong>{f(l2h2['correct_top1'])}</strong><span>L2H2 在正确 prediction query 的 top-1 retrieval</span></div>
  <div class="summary"><strong>{f(l2h2['correct_prompt_needle_mass'])}</strong><span>L2H2 对正确第 k 个 prompt needle 的原始 attention mass</span></div>
</div>

<div class="callout bad"><strong>结论 1：当前 v5 没有真正学到“开关”。</strong> thinking 与 non-thinking 在 <code>&lt;Think/&gt;</code> 之前具有完全相同的 prefix；默认 loss mask 又不监督 non-thinking 的 <code>&lt;/Think&gt;</code>。因此模型在自由生成时几乎总是进入 trace，而不是根据一个可见 mode signal 切换。</div>
<div class="callout good"><strong>结论 2：marker-only trace 的 targeted retrieval 非常显著。</strong> 把 attention 测在“预测第 k 个 marker 的 query token”上后，L2H2、L2H0、L3H2 都近乎确定性地定位第 k 个 prompt needle。此前“不显著”的印象主要来自测量 anchor 放在 marker 已经出现之后。</div>
<div class="callout warn"><strong>证据边界：</strong>这些是 attention 关联证据，不是 causal sufficiency / necessity。尚未做 head ablation、activation patching 或 mode-token intervention，因此不能声称这些 heads 单独构成完整计数电路。</div>

<h2>1. 实验设定</h2>
{table(['项目','具体设置'], setting_rows)}

<h3>两种训练格式与 loss mask</h3>
<div class="sequence">thinking：&lt;BOS&gt; prompt &lt;Think/&gt; <span class="supervised">M1 M2 … Mn &lt;/Think&gt; &lt;Cn&gt; &lt;EOS&gt;</span></div>
<div class="sequence">non-thinking：&lt;BOS&gt; prompt &lt;Think/&gt; <span class="unsupervised">&lt;/Think&gt;</span> <span class="supervised">&lt;Cn&gt; &lt;EOS&gt;</span></div>
<p>红色部分进入 next-token cross-entropy；灰色 <code>&lt;/Think&gt;</code> 在默认 non-thinking 样本中标签为 <code>IGNORE_INDEX</code>。这意味着 50% non-thinking 样本并没有教模型在 <code>&lt;Think/&gt;</code> 后关闭思考；它们只教模型在外部已经给出 <code>&lt;/Think&gt;</code> 后输出正确 count。</p>

<h2>2. 指标与计算方法</h2>
<p>对 count 为 n 的 prompt，将按位置排序的 needle token 位置记为 <code>P={{p1,…,pn}}</code>。第 k 个 trace marker 的正确来源是 <code>pk</code>。</p>
<div class="formula"><strong>Prediction query：</strong>k=1 时 query 是 <code>&lt;Think/&gt;</code>；k&gt;1 时 query 是前一个 trace marker <code>M(k−1)</code>。它的 hidden state 正在预测当前 marker <code>Mk</code>。</div>
<div class="formula"><strong>Post-marker control：</strong>query 改为已经出现的 <code>Mk</code> token。本报告仍检查它对当前 <code>pk</code> 的 attention；这不是“预测 Mk 时的检索”，也不是严格的“检索下一个 p(k+1)”指标。</div>
<div class="formula"><strong>correct top-1：</strong><code>1[argmax(j∈P) A(query,j) = pk]</code>，只在 n 个 prompt needle 之间比较。随机 baseline 对单个 count 是 <code>1/n</code>；按本实验全部 query 加权后为 {f(weighted_chance)}。</div>
<div class="formula"><strong>correct needle mass：</strong><code>A(query,pk)</code>，即对正确 needle 位置的原始 attention 概率。它比 top-1 更能防止“在极小总质量中勉强排第一”的假阳性。</div>
<div class="formula"><strong>all-needle mass：</strong><code>Σ(j∈P) A(query,j)</code>；<strong>diagonal share：</strong><code>A(query,pk) / Σ(j∈P)A(query,j)</code>；<strong>noise mass：</strong>attention 对 256-token prompt body 中所有非-needle token 的总和。</div>
<div class="formula"><strong>marker logit margin：</strong><code>logit(gold marker identity) − max logit(other marker identities)</code>。这是整模型 logits 的属性，不属于某一个 head，因此按 head 复制后会得到 16 个相同数值。</div>

<h2>3. 行为结果：条件任务满分，但自由 mode selection 不是二选一</h2>
{table(['评估模式','样本数','final accuracy','MAE','trace exact','实际给模型的 prefix'], behavior_rows or [['未找到原 v5 eval 表','','','','','']])}
<p>这里的 non-thinking accuracy 是条件准确率：评估器先把 <code>&lt;Think/&gt; &lt;/Think&gt;</code> 都放进 prefix，再在 count tokens 中读出预测。thinking 则从 <code>&lt;Think/&gt;</code> 后贪心生成完整 trace。因此二者都达到 1.0，不等价于模型能从相同 prefix 主动选择两种模式。</p>

<figure class="figure"><img src="{images['switch_probability_summary.png']}" alt="switch probability summary"><figcaption><strong>Figure 1. Switch 与 endpoint 概率。</strong>横轴依次为：在 <code>&lt;Think/&gt;</code> 后关闭的概率、开始任一 trace marker 的概率、在 teacher-forced <code>&lt;/Think&gt;</code> 后输出任一 count 的概率、输出 gold count 的概率；纵轴为 softmax 概率。蓝/橙表示用 non-thinking / thinking 完整序列做 forward。前两项的 prefix 完全相同，所以两色重合是因果必然，而非“两个开关都学得一样好”。模型给 marker 的概率约 {f(think_switch['p_any_trace_start_after_think_open'],8)}，给 close 的概率仅 {f(think_switch['p_close_after_think_open'],2)}。</figcaption></figure>
<p>对应 logit margin：<code>close − best trace-start</code> 为 {f(think_switch['margin_close_vs_trace_start_after_open'])}；teacher-forced close 后，gold count 相对其他 count 的 margin 为 thinking {f(think_switch['margin_gold_count_vs_other_counts_after_close'])}、non-thinking {f(non_switch['margin_gold_count_vs_other_counts_after_close'])}。</p>

<h2>4. 正确 prediction query 上存在清晰的 k-to-k retrieval</h2>
<div class="figure-grid">
<figure class="figure"><img src="{images['prediction_query_correct_top1.png']}" alt="prediction query top-1 heatmap"><figcaption><strong>Figure 2a. correct top-1 retrieval。</strong>横轴 head 0–3，纵轴 Transformer layer 1–4；每格为 5500 个 query 上的 top-1 命中率。颜色范围 0–1。L2H2={f(l2h2['correct_top1'])}、L2H0={f(l2h0['correct_top1'])}、L3H2={f(l3h2['correct_top1'])}。</figcaption></figure>
<figure class="figure"><img src="{images['prediction_query_correct_mass.png']}" alt="prediction query correct needle mass heatmap"><figcaption><strong>Figure 2b. correct prompt-needle mass。</strong>坐标与 2a 相同；每格是对正确第 k 个 prompt needle 的原始 attention mass 均值。L2H2={f(l2h2['correct_prompt_needle_mass'])}，说明它不是仅在很小 needle mass 内“相对排第一”，而是把几乎全部 attention 直接送到正确位置。</figcaption></figure>
</div>

{table(['Head','top-1','correct mass','all-needle mass','noise mass','BOS mass','解释'], head_rows)}

<h3>强 retrieval heads 是否只在少数 count 上有效？</h3>
{table(['count','random 1/n','L2H2 top-1','L2H2 mass','L2H0 top-1','L2H0 mass','L3H2 top-1','L3H2 mass'], count_rows, 'compact num')}
<p>L2H2 和 L3H2 在 count 1–10 全范围保持接近 1.0；L2H0 主要在 count=2 时下降到 0.825，其他 count 基本稳定。因此 targeted retrieval 不是由低 count 平均值制造的假象。</p>

<h2>5. 为什么旧 anchor 会让 retrieval 看起来不显著</h2>
<figure class="figure"><img src="{images['post_marker_correct_top1.png']}" alt="post marker top-1 heatmap"><figcaption><strong>Figure 3. Post-marker control。</strong>横轴 head、纵轴 layer；每格在 marker <code>Mk</code> 已经出现在 trace 后，检查该 token 对当前第 k 个 prompt needle 的 top-1。L2H2 从正确 prediction query 的 {f(l2h2['correct_top1'])} 降到 {f(l2h2_post['correct_top1'])}，L3H2 从 {f(l3h2['correct_top1'])} 降到 {f(l3h2_post['correct_top1'])}。这说明 retrieval 是 query-position specific：生成 marker 之前强，marker 已出现后其路由功能发生变化。</figcaption></figure>
<div class="callout warn"><strong>不要把 Figure 3 解读成“模型不会 retrieval”。</strong>它测的是错误时间点，而且 target 仍是当前 pk，不是下一个 p(k+1)。要研究 successor transition，需要单独把 post-Mk query 与 p(k+1) 对齐并做 causal ablation。</div>

<h2>6. Logits 与 hidden-state sanity checks</h2>
<figure class="figure"><img src="{images['prediction_query_marker_margin.png']}" alt="marker logit margin repeated heatmap"><figcaption><strong>Figure 4. Gold marker logit margin。</strong>横轴 head、纵轴 layer，但所有格均为 {f(l2h2['target_marker_logit_margin_vs_markers'])}，因为该值直接由同一个 prediction-query logits 计算，再附加到每条 head 记录；它只证明模型在该位置非常确信 gold marker identity，不提供 head localization 证据。</figcaption></figure>
{table(['variant','anchor','final hidden norm'], hidden_rows, 'compact')}
<p>final hidden norm 在两种 teacher-forced 序列间非常接近；norm 本身不包含方向信息，不能用来判断 mode representation 是否相同。若要比较 mode，应使用 matched-prompt hidden-state cosine、linear probe 或 activation patching。</p>

<h2>7. 严谨结论与下一步</h2>
<ol>
<li><strong>“开关学好了”不成立。</strong>当前输入没有提供可辨识的 mode cue，且 non-thinking close 未被监督。模型自由运行时选择 trace；non-thinking 只是在外部强制 close 后能正确读 count。</li>
<li><strong>“marker-only trace 没有 targeted retrieval”也不成立。</strong>L2H2、L2H0、L3H2 在真正预测 Mk 的 query 上提供强且跨 count 稳定的 k-to-k attention。</li>
<li><strong>可能存在 retrieval + set-level routing 的分工。</strong>L3H1/L3H3 几乎把全部质量给 needle 集合，但不总集中于当前 pk；这与聚合/路由角色相容，不过尚无因果证据。</li>
<li><strong>attention 不是充分机制证明。</strong>下一步应：(a) 加一个在 prompt 前可见且受监督的 mode token；(b) 分别 mask L2H2/L2H0/L3H2 及其组合；(c) patch clean/corrupt prompt 的 head output；(d) 用 p(k+1) 重定义 post-marker successor metric。</li>
</ol>

<footer>Generated from <code>{html.escape(str(result_dir))}</code>. Source checkpoint: <code>{html.escape(str(source_run) if source_run else 'not found')}</code>. All figures are embedded as base64; no external files or network access are required.</footer>
</main></body></html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="colab_results/v5_2_switch_diagnostics")
    parser.add_argument("--source-run", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    source_run = find_source_run(result_dir, args.source_run or None)
    output = Path(args.output) if args.output else result_dir / "syn_v5_2_report.html"
    build_report(result_dir, source_run, output)


if __name__ == "__main__":
    main()
