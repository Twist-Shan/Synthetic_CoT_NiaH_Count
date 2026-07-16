from __future__ import annotations

import argparse
import base64
import html
import json
import math
import sys
import types
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


if "pyarrow" not in sys.modules:
    pyarrow_stub = types.ModuleType("pyarrow")
    pyarrow_stub.__version__ = "0.0.0"
    pyarrow_stub.Array = type("Array", (), {})
    pyarrow_stub.ChunkedArray = type("ChunkedArray", (), {})
    sys.modules["pyarrow"] = pyarrow_stub
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd


BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#16a34a"
RED = "#dc2626"
PURPLE = "#7c3aed"
GRAY = "#64748b"
INK = "#172033"
GRID = "#d7deea"
COUNT_BINS = ("1-10", "11-20", "21-30")


def display_math(body: str) -> str:
    """Wrap presentation MathML for consistent browser-native typesetting."""
    body = body.replace("<mo>(</mo>", '<mo stretchy="false">(</mo>')
    body = body.replace("<mo>)</mo>", '<mo stretchy="false">)</mo>')
    return f'<math display="block" aria-label="mathematical formula"><mrow>{body}</mrow></math>'


FORMULA_MATHML = {
    "prompt_needles_mass": display_math(
        """
        <msubsup><mi>M</mi><mi>N</mi><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup>
        <mo>(</mo><mi>q</mi><mo>)</mo><mo>=</mo>
        <munder><mo>∑</mo><mrow><mi>j</mi><mo>∈</mo><mi>N</mi></mrow></munder>
        <msubsup><mi>A</mi><mrow><mi>q</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup>
        """
    ),
    "needle_entropy_normalized": display_math(
        """
        <msub><mi>H</mi><mi>N</mi></msub><mo>(</mo><mi>q</mi><mo>)</mo><mo>=</mo>
        <mfrac>
          <mrow><mo>−</mo><munder><mo>∑</mo><mrow><mi>j</mi><mo>∈</mo><mi>N</mi></mrow></munder><msub><mi>p</mi><mi>j</mi></msub><mi>log</mi><msub><mi>p</mi><mi>j</mi></msub></mrow>
          <mrow><mi>log</mi><mi>n</mi></mrow>
        </mfrac>
        <mo>,</mo><mspace width="1em"/>
        <msub><mi>p</mi><mi>j</mi></msub><mo>=</mo>
        <mfrac>
          <msubsup><mi>A</mi><mrow><mi>q</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup>
          <mrow><msubsup><mi>M</mi><mi>N</mi><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup><mo>(</mo><mi>q</mi><mo>)</mo></mrow>
        </mfrac>
        """
    ),
    "broad_attention_score": display_math(
        """
        <msubsup><mi>S</mi><mtext>broad</mtext><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup>
        <mo>(</mo><mi>q</mi><mo>)</mo><mo>=</mo>
        <msubsup><mi>M</mi><mi>N</mi><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup><mo>(</mo><mi>q</mi><mo>)</mo>
        <mo>·</mo><msub><mi>H</mi><mi>N</mi></msub><mo>(</mo><mi>q</mi><mo>)</mo>
        """
    ),
    "correct_prompt_needle_mass / k-to-k mass": display_math(
        """
        <msubsup><mi>S</mi><mtext>target</mtext><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup>
        <mo>(</mo><mi>k</mi><mo>)</mo><mo>=</mo>
        <msubsup><mi>A</mi><mrow><msub><mi>q</mi><mi>k</mi></msub><mo>,</mo><msub><mi>n</mi><mi>k</mi></msub></mrow><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup>
        """
    ),
    "correct_top1": display_math(
        """
        <mi mathvariant="double-struck">1</mi><mo>[</mo>
        <munder><mi>argmax</mi><mrow><mi>j</mi><mo>∈</mo><mi>N</mi></mrow></munder>
        <msubsup><mi>A</mi><mrow><msub><mi>q</mi><mi>k</mi></msub><mo>,</mo><mi>j</mi></mrow><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup>
        <mo>=</mo><msub><mi>n</mi><mi>k</mi></msub><mo>]</mo>
        """
    ),
    "diagonal_dominance": display_math(
        """
        <msup><mi>D</mi><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msup><mo>(</mo><mi>k</mi><mo>)</mo><mo>=</mo>
        <mfrac>
          <msubsup><mi>A</mi><mrow><msub><mi>q</mi><mi>k</mi></msub><mo>,</mo><msub><mi>n</mi><mi>k</mi></msub></mrow><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup>
          <mrow><munder><mo>∑</mo><mrow><mi>j</mi><mo>∈</mo><mi>N</mi></mrow></munder><msubsup><mi>A</mi><mrow><msub><mi>q</mi><mi>k</mi></msub><mo>,</mo><mi>j</mi></mrow><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup></mrow>
        </mfrac>
        """
    ),
    "trace_markers_mass / trace-readout score": display_math(
        """
        <msubsup><mi>S</mi><mtext>trace</mtext><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup><mo>=</mo>
        <munder><mo>∑</mo><mrow><mi>j</mi><mo>∈</mo><msub><mi>T</mi><mi>M</mi></msub></mrow></munder>
        <msubsup><mi>A</mi><mrow><msub><mi>q</mi><mtext>ans</mtext></msub><mo>,</mo><mi>j</mi></mrow><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup>
        """
    ),
    "next_prompt_needle_mass / successor score": display_math(
        """
        <msubsup><mi>S</mi><mtext>succ</mtext><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup><mo>(</mo><mi>k</mi><mo>)</mo><mo>=</mo>
        <msubsup><mi>A</mi><mrow><msub><mi>q</mi><mrow><mtext>succ</mtext><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msub><mo>,</mo><msub><mi>n</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub></mrow><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup>
        """
    ),
    "ablation drop": display_math(
        '<msub><mi>Δ</mi><mtext>abl</mtext></msub><mo>=</mo><msub><mi>s</mi><mtext>base</mtext></msub><mo>−</mo><msub><mi>s</mi><mtext>mask</mtext></msub>'
    ),
    "normalized recovery": display_math(
        """
        <msub><mi>R</mi><mtext>norm</mtext></msub><mo>=</mo>
        <mfrac>
          <mrow><msub><mi>m</mi><mtext>patched</mtext></msub><mo>−</mo><msub><mi>m</mi><mtext>corrupt</mtext></msub></mrow>
          <mrow><msub><mi>m</mi><mtext>clean</mtext></msub><mo>−</mo><msub><mi>m</mi><mtext>corrupt</mtext></msub></mrow>
        </mfrac>
        """
    ),
    "expected-count shift": display_math(
        '<mi>Δ</mi><mi mathvariant="normal">E</mi><mo>[</mo><mi>C</mi><mo>]</mo><mo>=</mo><msub><mi mathvariant="normal">E</mi><mtext>patch</mtext></msub><mo>[</mo><mi>C</mi><mo>]</mo><mo>−</mo><msub><mi mathvariant="normal">E</mi><mtext>base</mtext></msub><mo>[</mo><mi>C</mi><mo>]</mo>'
    ),
    "transport slope": display_math(
        '<mi>Δ</mi><mi mathvariant="normal">E</mi><mo>[</mo><mi>C</mi><mo>]</mo><mo>=</mo><mi>a</mi><mo>+</mo><mi>b</mi><mo>(</mo><msub><mi>c</mi><mtext>donor</mtext></msub><mo>−</mo><msub><mi>c</mi><mtext>receiver</mtext></msub><mo>)</mo>'
    ),
    "steering gain 与 R²": display_math(
        '<mi>Δ</mi><mi mathvariant="normal">E</mi><mo>[</mo><mi>C</mi><mo>]</mo><mo>=</mo><mi>a</mi><mo>+</mo><mi>g</mi><mi>α</mi>'
    ),
    "PCA explained-variance ratio": display_math(
        '<msub><mi>EVR</mi><mi>r</mi></msub><mo>=</mo><mfrac><msub><mi>λ</mi><mi>r</mi></msub><mrow><munder><mo>∑</mo><mi>s</mi></munder><msub><mi>λ</mi><mi>s</mi></msub></mrow></mfrac>'
    ),
    "effective dimension": display_math(
        '<msub><mi>d</mi><mtext>eff</mtext></msub><mo>=</mo><mfrac><msup><mrow><munder><mo>∑</mo><mi>r</mi></munder><msub><mi>λ</mi><mi>r</mi></msub></mrow><mn>2</mn></msup><mrow><munder><mo>∑</mo><mi>r</mi></munder><msubsup><mi>λ</mi><mi>r</mi><mn>2</mn></msubsup></mrow></mfrac>'
    ),
    "attention softmax": display_math(
        """
        <msubsup><mi>A</mi><mrow><mi>q</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup><mo>=</mo>
        <mfrac><mrow><mi>exp</mi><mo>(</mo><msub><mi>s</mi><mrow><mi>q</mi><mi>j</mi></mrow></msub><mo>)</mo></mrow><mrow><munder><mo>∑</mo><mrow><mi>t</mi><mo>≤</mo><mi>q</mi></mrow></munder><mi>exp</mi><mo>(</mo><msub><mi>s</mi><mrow><mi>q</mi><mi>t</mi></mrow></msub><mo>)</mo></mrow></mfrac>
        <mo>,</mo><mspace width="1em"/>
        <msub><mi>s</mi><mrow><mi>q</mi><mi>j</mi></mrow></msub><mo>=</mo>
        <mfrac><mrow><msub><mi>Q</mi><mi>q</mi></msub><msubsup><mi>K</mi><mi>j</mi><mi mathvariant="normal">T</mi></msubsup></mrow><msqrt><msub><mi>d</mi><mi>h</mi></msub></msqrt></mfrac>
        """
    ),
    "attention mass": display_math(
        """
        <msup><mi>M</mi><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msup><mo>(</mo><mi>S</mi><mo>|</mo><mi>q</mi><mo>)</mo><mo>=</mo>
        <munder><mo>∑</mo><mrow><mi>j</mi><mo>∈</mo><mi>S</mi></mrow></munder>
        <msubsup><mi>A</mi><mrow><mi>q</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>ℓ</mi><mo>,</mo><mi>h</mi><mo>)</mo></mrow></msubsup>
        """
    ),
    "candidate softmax": display_math(
        """
        <msub><mi>p</mi><mi>C</mi></msub><mo>(</mo><mi>t</mi><mo>|</mo><mi>q</mi><mo>)</mo><mo>=</mo>
        <mfrac><mrow><mi>exp</mi><mo>(</mo><msub><mi>z</mi><mrow><mi>q</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>)</mo></mrow><mrow><munder><mo>∑</mo><mrow><mi>u</mi><mo>∈</mo><mi>C</mi></mrow></munder><mi>exp</mi><mo>(</mo><msub><mi>z</mi><mrow><mi>q</mi><mo>,</mo><mi>u</mi></mrow></msub><mo>)</mo></mrow></mfrac>
        """
    ),
    "target logit margin": display_math(
        """
        <mi>m</mi><mo>(</mo><mi>z</mi><mo>;</mo><msup><mi>t</mi><mo>*</mo></msup><mo>,</mo><mi>C</mi><mo>)</mo><mo>=</mo>
        <msub><mi>z</mi><msup><mi>t</mi><mo>*</mo></msup></msub><mo>−</mo>
        <munder><mi>max</mi><mrow><mi>u</mi><mo>∈</mo><mi>C</mi><mo>∖</mo><mo>{</mo><msup><mi>t</mi><mo>*</mo></msup><mo>}</mo></mrow></munder><msub><mi>z</mi><mi>u</mi></msub>
        """
    ),
    "expected count": display_math(
        """
        <mi mathvariant="normal">E</mi><mo>[</mo><mi>C</mi><mo>]</mo><mo>=</mo><munderover><mo>∑</mo><mrow><mi>c</mi><mo>=</mo><mn>1</mn></mrow><mn>30</mn></munderover><mi>c</mi><msub><mi>p</mi><mi>c</mi></msub>
        <mo>,</mo><mspace width="1em"/>
        <msub><mi>p</mi><mi>c</mi></msub><mo>=</mo><msub><mi>softmax</mi><mi>c</mi></msub><mo>(</mo><msub><mi>z</mi><mrow><msub><mi>C</mi><mn>1</mn></msub><mo>:</mo><msub><mi>C</mi><mn>30</mn></msub></mrow></msub><mo>)</mo>
        """
    ),
    "residual centroid": display_math(
        """
        <msubsup><mi>h</mi><mi>q</mi><mrow><mo>(</mo><mi>ℓ</mi><mo>)</mo></mrow></msubsup><mo>∈</mo><msup><mi mathvariant="double-struck">R</mi><mn>256</mn></msup>
        <mo>,</mo><mspace width="1em"/>
        <msubsup><mi>μ</mi><mi>c</mi><mrow><mo>(</mo><mi>ℓ</mi><mo>)</mo></mrow></msubsup><mo>=</mo>
        <mi mathvariant="normal">E</mi><mo>[</mo><msubsup><mi>h</mi><mi>q</mi><mrow><mo>(</mo><mi>ℓ</mi><mo>)</mo></mrow></msubsup><mo>|</mo><mtext>count</mtext><mo>=</mo><mi>c</mi><mo>]</mo>
        """
    ),
    "adjacent-count direction": display_math(
        """
        <msub><mi>δ</mi><mi>c</mi></msub><mo>=</mo><msub><mi>μ</mi><mrow><mi>c</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>−</mo><msub><mi>μ</mi><mi>c</mi></msub>
        <mo>,</mo><mspace width="1em"/>
        <mi>d</mi><mo>=</mo><mfrac><mrow><msub><mi>mean</mi><mi>c</mi></msub><mo>(</mo><msub><mi>δ</mi><mi>c</mi></msub><mo>)</mo></mrow><mrow><mo>‖</mo><msub><mi>mean</mi><mi>c</mi></msub><mo>(</mo><msub><mi>δ</mi><mi>c</mi></msub><mo>)</mo><mo>‖</mo></mrow></mfrac>
        <mo>,</mo><mspace width="1em"/>
        <mi>s</mi><mo>=</mo><msub><mi>mean</mi><mi>c</mi></msub><mo>(</mo><msub><mi>δ</mi><mi>c</mi></msub><mo>·</mo><mi>d</mi><mo>)</mo>
        """
    ),
    "steering update": display_math(
        '<mi>h</mi><mo>←</mo><mi>h</mi><mo>+</mo><mi>α</mi><mi>s</mi><mi>d</mi>'
    ),
}


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    if number and abs(number) < 10 ** (-digits):
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    try:
        return f"{100 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def code(value: object) -> str:
    return f"<code>{html.escape(str(value))}</code>"


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def figure(path: Path, title: str, caption: str) -> str:
    if not path.exists():
        return ""
    return f"""
    <figure class="figure">
      <h3>{html.escape(title)}</h3>
      <img src="{image_uri(path)}" alt="{html.escape(title)}">
      <figcaption>{caption}</figcaption>
    </figure>"""


def table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{row.get(key, '')}</td>" for key, _ in columns) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def build_head_state_interaction_section(run_dir: Path) -> str:
    """Render Section 11 from the formal bidirectional causal-analysis tables."""
    analysis_dir = run_dir / "analysis" / "head_state_bidirectional"
    tables_dir = analysis_dir / "tables"
    figures_dir = analysis_dir / "figures"
    head_path = tables_dir / "head_to_state_summary.csv"
    state_path = tables_dir / "state_to_head_summary.csv"
    manifest_path = analysis_dir / "manifest.json"
    if not head_path.exists() or not state_path.exists():
        return """
        <section id="interaction">
          <h2>11. Attention head 与 hidden state 的双向因果联系</h2>
          <div class="callout warn"><b>尚未运行：</b>未找到 <code>analysis/head_state_bidirectional</code> 的正式结果。运行 <code>scripts/run_v10_head_state_bidirectional.py</code> 后重建报告。</div>
        </section>
        """

    head = pd.read_csv(head_path)
    state = pd.read_csv(state_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    def fnum(value: object, digits: int = 3) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"

    def fpct(value: object, digits: int = 1) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not math.isfinite(number) else f"{100 * number:.{digits}f}%"

    intervention_labels = {
        "clean": "clean",
        "candidate_top1": "候选 top-1",
        "candidate_top4": "候选 top-4",
        "matched_control_top1": "同层 control-1",
        "noncandidate4_control": "候选外 control-4",
    }
    mechanism_labels = {
        "nonthinking_broad": "Non-thinking broad → count state",
        "cot_targeted": "CoT k-to-k → marker-identity state",
        "cot_readout": "CoT trace-readout → final count state",
    }

    final_layer = int(head["eval_layer"].max())
    head_final = head[head["eval_layer"] == final_layer].copy()
    head_rows: list[dict[str, object]] = []
    for mechanism in ("nonthinking_broad", "cot_targeted", "cot_readout"):
        for count_bin in COUNT_BINS:
            subset = head_final[(head_final["mechanism"] == mechanism) & (head_final["count_bin"] == count_bin)]
            for intervention in ("clean", "candidate_top1", "candidate_top4", "noncandidate4_control"):
                row = subset[subset["intervention"] == intervention]
                if row.empty:
                    continue
                record = row.iloc[0]
                head_rows.append(
                    {
                        "mechanism": mechanism_labels[mechanism],
                        "bin": count_bin,
                        "intervention": intervention_labels[intervention],
                        "geometry": fpct(record["nearest_centroid_accuracy"]),
                        "output": fpct(record["output_accuracy"]),
                        "geometry_margin": fnum(record["mean_centroid_margin"]),
                        "output_margin": fnum(record["mean_output_margin"]),
                    }
                )

    targeted_state = state[
        (state["mechanism"] == "cot_targeted")
        & (state["intervention"] == "shifted_state_patch")
        & (state["head_scope"] == "candidate_top4")
    ]
    targeted_rows: list[dict[str, object]] = []
    for count_bin in COUNT_BINS:
        for _, row in targeted_state[targeted_state["count_bin"] == count_bin].sort_values("patch_before_layer").iterrows():
            targeted_rows.append(
                {
                    "bin": count_bin,
                    "layer": f"进入 Layer {int(row['patch_before_layer'])} 前",
                    "shift": fnum(row["mean_signature_shift"]),
                    "abs_shift": fnum(row["mean_abs_signature_shift"]),
                    "max_shift": fnum(row["max_abs_signature_shift"]),
                }
            )

    final_state = state[
        (state["intervention"] == "shifted_state_patch")
        & (state["head_scope"] == "all_downstream")
        & (state["mechanism"].isin(["nonthinking_broad", "cot_readout"]))
    ]
    final_state_rows: list[dict[str, object]] = []
    for mechanism in ("nonthinking_broad", "cot_readout"):
        for count_bin in COUNT_BINS:
            subset = final_state[(final_state["mechanism"] == mechanism) & (final_state["count_bin"] == count_bin)]
            for _, row in subset.sort_values("patch_before_layer").iterrows():
                final_state_rows.append(
                    {
                        "mechanism": "Non-thinking" if mechanism == "nonthinking_broad" else "CoT final readout",
                        "bin": count_bin,
                        "layer": f"Layer {int(row['patch_before_layer'])}",
                        "routing": fnum(row["mean_abs_signature_shift"]),
                        "margin": fnum(row["mean_count_margin_shift"], 2),
                        "receiver": fpct(row["receiver_count_retention"]),
                        "donor": fpct(row["shifted_donor_adoption"]),
                    }
                )

    sample_note = (
        f"centroid bank 每个 exact count {manifest.get('centroid_examples_per_count', 6)} 个样本；"
        f"head→state 每个 exact count {manifest.get('eval_examples_per_count', 3)} 个样本；"
        f"state→head 每个 count bin {manifest.get('state_to_head_examples_per_bin', 5)} 组 donor/receiver pairs。"
    )

    head_table = table(
        head_rows,
        [
            ("mechanism", "因果链"),
            ("bin", "count 区间"),
            ("intervention", "局部 head-output intervention"),
            ("geometry", "末层 state 分类准确率"),
            ("output", "局部输出准确率"),
            ("geometry_margin", "state centroid margin"),
            ("output_margin", "output logit margin"),
        ],
    )
    targeted_table = table(
        targeted_rows,
        [
            ("bin", "count 区间"),
            ("layer", "residual patch 位置"),
            ("shift", "候选 top-4 平均 routing shift"),
            ("abs_shift", "平均绝对 shift"),
            ("max_shift", "最大绝对 shift"),
        ],
    )
    final_state_table = table(
        final_state_rows,
        [
            ("mechanism", "模式"),
            ("bin", "count 区间"),
            ("layer", "patch 进入该 Layer 前"),
            ("routing", "下游 heads 平均绝对 routing shift"),
            ("margin", "receiver-count margin shift"),
            ("receiver", "仍输出 receiver count"),
            ("donor", "改输出 donor count"),
        ],
    )

    geometry_figure = figure(
        figures_dir / "head_to_state_geometry.png",
        "Figure 11A. 局部 head-output mask 对后续 hidden-state geometry 的影响",
        "三个面板分别对应 direct broad、CoT k-to-k retrieval 与 CoT trace readout。横轴是 intervention；颜色是 gold-count 区间。纵轴是末层局部 state 的 held-out nearest-centroid accuracy。direct/readout 的 centroid 标签是 exact count；targeted 链的标签是当前应写出的 marker identity。这里测的是 head 输出被改动后，后续 residual 是否仍保留正确语义，而不是只看最终 argmax。",
    )
    output_figure = figure(
        figures_dir / "head_to_state_output.png",
        "Figure 11B. 同一 head intervention 对局部 marker / final-count 输出的影响",
        "横轴与 Figure 11A 相同；纵轴是局部输出准确率。Non-thinking 和 readout 面板评估 final count，targeted 面板评估当前 trace marker。把 11A 与 11B 并列阅读，可以判断 state geometry 的损伤是否与输出损伤同步。",
    )
    routing_figure = figure(
        figures_dir / "state_to_head_routing.png",
        "Figure 11C. hidden-state transplant 是否反向改变下游 attention routing",
        "横轴表示 donor residual 被替换到 receiver 的哪一个 Transformer Layer 之前；纵轴是候选 top-4 heads 的平均 attention-signature shift。实线为 shifted donor，虚线为 same-state donor control；颜色为 count 区间。Targeted 链的 shifted donor 保持总 count 不变，只把 trace progress 从较早 k 换成较晚 j，因此其大幅正 shift 表示 progress state 足以重定向后续 k-to-k routing。",
    )
    mediation_figure = figure(
        figures_dir / "head_state_mediation.png",
        "Figure 11D. head 导致的 state-margin 损伤是否追踪 output-margin 损伤",
        "每个点是一个 mechanism × count-bin × intervention。横轴是 hidden centroid margin 相对 clean 的下降，纵轴是 marker/final-count logit margin 的下降；点越靠右上，越符合“head intervention 先损伤语义 state，再损伤输出”的中介链。该图是跨 intervention 的效应对齐，不单独构成 mediation 的统计识别。",
    )

    return f"""
    <section id="interaction">
      <h2>11. Attention head 与 hidden state 的双向因果联系</h2>
      <p>第 5–6 节只描述 head signature 与 count/marker geometry，第 7–10 节分别证明了若干组件的必要性或可控性。本节把两类干预放到同一批样本上，直接检验两条方向：<b>head → downstream state</b> 与 <b>state → downstream head routing</b>。双向不意味着循环反馈；它只要求两个方向都由实际 intervention 支持。</p>
      <div class="callout info"><b>统计单位。</b>{sample_note} 本节是单 seed 的机制定位实验，样本量用于稳定复现局部效应，不作为总体显著性检验。</div>

      <h3>11.1 共同实验设计与三个机制链</h3>
      <div class="definition-grid">
        <article><h4>方向 A：head → state</h4><p>在语义 query 位置，把指定 attention head 的 <code>c_proj</code> 输入切片置零，只影响该 token 的该 head output；随后在每个 Layer 后读取 residual。Direct/readout 用 exact-count centroid，k-to-k 链用 marker-identity centroid。对照包括 clean、候选 top-1/top-4、同层 control-1 与严格不属于候选 top-4 的 control-4。</p></article>
        <article><h4>方向 B：state → head</h4><p>先缓存 donor 在某 token 的 residual，再替换 receiver 进入指定 Layer 前的 residual，并读取该层及后续所有 heads 的 attention。Direct/readout 使用不同 count 的 donor→receiver；k-to-k 保持总 count 不变，只改变 trace progress。same-state donor 来自与 receiver 同一语义状态的另一条样本。</p></article>
      </div>
      <ul>
        <li><b>Non-thinking broad chain：</b><code>&lt;Ans&gt;</code> query 的 Layer-1 broad heads → 后续 scalar-count state。</li>
        <li><b>CoT targeted chain：</b>trace 数字 <code>&lt;k&gt;</code> query 的 k-to-k heads → 当前 marker-identity state。</li>
        <li><b>CoT readout chain：</b>最终 <code>&lt;Ans&gt;</code> query 的 trace-readout heads → final scalar-count state。</li>
      </ul>

      <h3>11.2 方向 A：head intervention 是否损伤下游语义 state</h3>
      <h4>实验</h4>
      <p>对上述三个 query 分别做局部 per-head output mask，并在最终 Layer 后同时测 nearest-centroid accuracy、centroid margin、局部输出 accuracy 与 logit margin。<code>candidate_top4</code> 是按对应描述性 score 排名前四的 heads；<code>noncandidate4_control</code> 与这四头严格不重叠。</p>
      <h4>结果</h4>
      {head_table}
      {geometry_figure}
      {output_figure}
      <h4>分析</h4>
      <p><b>Non-thinking：</b>mask broad top-1 后末层 count-state accuracy 为 16.7% / 6.7% / 76.7%，累计 top-4 后为 0% / 10% / 0%；输出准确率几乎同步下降，而候选外 control-4 保持 100%。这支持 early broad heads 在 direct counting 中因果上游于 scalar count state。</p>
      <p><b>CoT targeted：</b>单头 top-1 可被替代，但累计 top-4 后 marker-identity state 降到 53.3% / 40.0% / 40.0%，marker 输出降到 56.7% / 46.7% / 43.3%。不过候选外 control-4 在 21–30 也降到 30%，说明高 count 的 retrieval 依赖分布式 heads；结果支持“候选组必要”，不支持“只有这四头有用”。</p>
      <p><b>CoT readout：</b>这些局部 mask 下 state 与 final argmax 仍为 100%，但 margin 普遍下降。这表明最终 readout 在当前模型中高度冗余；仅凭高 trace-marker mass 不能断言某个 readout head 独立写入最终 count。</p>

      <h3>11.3 方向 B：hidden progress/count state 是否反向控制 attention routing</h3>
      <h4>实验</h4>
      <p>Targeted 链固定 prompt 总 count，只把较早 progress state 移植到较晚 progress query 的对照或反向移植；routing signature 是候选 top-4 对 donor-matching needle 相对 receiver-matching needle 的 attention 差。Direct/readout 使用 5→8、15→18、25→28 的 count donors；同时记录所有下游 heads 的 signature 变化和最终预测是否采用 donor count。</p>
      <h4>结果：CoT targeted progress → retrieval routing</h4>
      {targeted_table}
      {routing_figure}
      <h4>分析</h4>
      <p>same-state controls 接近零；shifted progress state 在四个 patch 深度均显著把 targeted heads 推向 donor 所对应的 routing，低/中/高 count 的平均 shift 约为 0.98–1.18、0.53–0.69、0.47–0.61。因而 targeted retrieval 不只是把 marker 写进 hidden state；当前 progress residual 也因果决定下一步 heads 去读哪个 needle。这是本节最强的双向局部耦合证据。</p>

      <h4>结果：scalar count state → direct/readout routing 与输出</h4>
      {final_state_table}
      <h4>分析</h4>
      <p><b>Direct 模式是非对称链。</b>进入 Layer 1 前的 <code>&lt;Ans&gt;</code> residual 只有 token+position embedding，跨 count 相同，因此 patch 不改变 Layer-1 broad routing。进入 Layer 2–4 前移植 count state 后，低 count 会 100% 采用 donor count；中 count 主要发生连续 margin 搬运，只有 Layer 4 达到 40% exact donor adoption；高 count 更弱。与此同时 broad signature 的平均绝对变化很小。这说明 broad heads 先写入 count state，但后期 count state 并不会反向重建已经发生过的 early broad attention。</p>
      <p><b>CoT final state 对输出充分，但对 readout routing 的控制不稳定。</b>三个区间在四个深度均 100% 采用 donor count；然而候选 readout signature 的变化随 Layer/count 改变方向，高 count 甚至为轻微负值。更合理的解释是 final count residual 可以直接驱动 logits，而 readout heads 是形成该 state 的冗余上游，不是必须被该 state 反向控制的一组固定路由。</p>

      <h3>11.4 head→state→output 的效应对齐</h3>
      <h4>实验与结果</h4>
      {mediation_figure}
      <h4>分析</h4>
      <p>Direct broad 与 CoT targeted 的大幅 state-margin 损伤通常伴随大幅 output-margin 损伤，符合“head output 改变后续语义 residual，随后影响 logits”的串联解释。CoT readout 的点集中在较小损伤范围，再次显示局部冗余。由于这里没有在同一 intervention 内进一步随机化中介 state，该散点只能作为效应对齐，不能单独声称完成统计 mediation。</p>

      <h3>11.5 综合结论与边界</h3>
      <div class="mechanism-grid">
        <article><h4>Non-thinking</h4><p><b>支持：</b>Layer-1 broad heads → downstream scalar-count state → final logits。后期 count-state transplant 足以搬运低 count 输出。<br><b>不支持：</b>count state 反向重建早期 broad routing；这是一条前馈、非对称因果链。</p></article>
        <article><h4>CoT targeted retrieval</h4><p><b>支持：</b>targeted head 组 → marker-identity state，且 progress state → 下一步 targeted routing。二者组成最清楚的局部双向耦合。<br><b>边界：</b>单头可替代，高 count 下候选外 heads 也重要，circuit 是分布式的。</p></article>
        <article><h4>CoT final readout</h4><p><b>支持：</b>final scalar state 对 count 输出高度充分。<br><b>尚缺：</b>固定 readout-head 组与该 state 的强双向耦合；局部 mask 与 routing shift 都显示冗余和异质性。</p></article>
      </div>
      <div class="callout warn"><b>证据边界：</b>单 seed、局部 query intervention、有限 donor/receiver pairs；nearest-centroid 只测预先定义的 count/marker geometry。结论应表述为该训练实例中的因果 circuit 定位，而不是所有 counting Transformer 的普遍定理。</div>
    </section>
    """


def html_section_span(document: str, section_id: str) -> tuple[int, int]:
    marker = f'<section id="{section_id}">'
    start = document.index(marker)
    end = document.index("</section>", start) + len("</section>")
    return start, end


def finalize_report_numbering(report: str, run_dir: Path | None = None) -> str:
    """Apply the public section order after replacing legacy report sections."""
    replacements = {
        "<h2>7. 分层 activation patching：候选 heads 是否局部充分</h2>":
            "<h2>8. Attention-head patching：候选 heads 是否局部充分</h2>",
        "<h3>7.1 CoT marker-identity clean-to-corrupt retrieval patch</h3>":
            "<h3>8.1 CoT marker-identity clean-to-corrupt retrieval patch</h3>",
        "<h3>7.2 Nested prompt donor→receiver count-head patch</h3>":
            "<h3>8.2 Nested prompt donor→receiver count-head patch</h3>",
        "<h2>8. 分层 geometry steering：可读方向是否也是可控方向</h2>":
            "<h2>9. Hidden-state geometry steering：可读方向是否也是可控方向</h2>",
        "<h2>9. 分层 residual transplant：完整 count state 能否搬运</h2>":
            "<h2>10. Hidden-state patching：完整 count state 能否跨 sequence 搬运</h2>",
        "<h3>9.1 Trace-progress transplant 的覆盖边界</h3>":
            "<h3>10.1 CoT trace 内部的 hidden-state patching：覆盖边界</h3>",
        "第 5B.1 节": "第 6.1 节",
        "第 9 节的完整 centroid transplant": "第 10 节的完整 centroid transplant",
        "<h2>11. 综合机制结论、证据强度与尚缺环节</h2>":
            "<h2>12. 综合机制结论、证据强度与尚缺环节</h2>",
    }
    for old, new in replacements.items():
        report = report.replace(old, new)

    if '<section id="interaction">' not in report:
        interaction_section = """
        <section id="interaction">
          <h2>11. Attention head 与 hidden state 的双向因果联系（下一步）</h2>
          <p>第 7–10 节分别测试 attention heads 与 residual hidden states 的必要性或充分性；但它们尚未回答二者是否属于同一条串联因果链，还是两套并行、可互相补偿的计算路径。下一步将固定同一批 clean/corrupt pairs，测量：head ablation 或 head-output patch 后 count manifold、centroid projection 与 logits 如何改变；以及 residual steering/patching 后 broad、k-to-k 与 trace-readout attention signatures 是否随之改变。</p>
          <div class="callout warn"><b>当前状态：</b>本节是明确的实验路线图，不把尚未运行的双向干预写成结果。现有证据只支持分别定位候选 attention circuit 与 count-state residual，而不能独立证明前者写入后者，或后者反向控制前者。</div>
        </section>
        """
        if run_dir is not None:
            interaction_section = build_head_state_interaction_section(run_dir)
        synthesis_start = report.index('<section id="synthesis">')
        report = report[:synthesis_start] + interaction_section + report[synthesis_start:]
    return report


def metric_cards(
    rows: list[dict[str, str]],
    *,
    context_label: str = "读取位置",
) -> str:
    cards: list[str] = []
    for row in rows:
        context = row.get("query", "")
        formula_html = FORMULA_MATHML.get(row["name"], row["formula_html"])
        context_html = (
            f'<div class="metric-context"><span>{html.escape(context_label)}</span>{html.escape(context)}</div>'
            if context
            else ""
        )
        cards.append(
            f"""
            <article class="metric-card">
              <div class="metric-card-head"><code>{html.escape(row['name'])}</code>{context_html}</div>
              <div class="equation">{formula_html}</div>
              <p>{html.escape(row['meaning'])}</p>
            </article>
            """
        )
    return f'<div class="metric-grid">{"".join(cards)}</div>'


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#8b98aa",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "font.size": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def heatmap(
    ax,
    values: np.ndarray,
    title: str,
    *,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    fmt_string: str = ".2f",
    xlabels: list[str] | None = None,
    ylabels: list[str] | None = None,
) -> None:
    matrix = np.asarray(values, dtype=float)
    image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.grid(False)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels(xlabels or list(range(matrix.shape[1])))
    ax.set_yticklabels(ylabels or list(range(1, matrix.shape[0] + 1)))
    middle = (vmin + vmax) / 2
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isfinite(value):
                color = "white" if value < middle else "#111827"
                ax.text(col, row, format(value, fmt_string), ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.035)


def head_matrix(frame: pd.DataFrame, metric: str) -> np.ndarray:
    return (
        frame.pivot(index="layer", columns="head", values=metric)
        .sort_index()
        .sort_index(axis=1)
        .to_numpy(dtype=float)
    )


def _balanced_accuracy_over_counts(eval_counts: pd.DataFrame) -> pd.DataFrame:
    """Average exact-count accuracies with one equal-weight vote per gold count."""
    return (
        eval_counts.groupby(["step", "mode"], as_index=False)[
            ["tf_accuracy", "ar_accuracy", "trace_exact"]
        ]
        .mean()
        .sort_values(["mode", "step"])
    )


def save_training_overall(
    eval_counts: pd.DataFrame,
    eval_losses: pd.DataFrame,
    path: Path,
) -> None:
    overall = _balanced_accuracy_over_counts(eval_counts)
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.1), constrained_layout=True)

    accuracy_specs = [
        ("nonthinking", "tf_accuracy", BLUE, "-", "non-thinking | TF final count"),
        ("nonthinking", "ar_accuracy", BLUE, "--", "non-thinking | AR final count"),
        ("thinking", "tf_accuracy", ORANGE, "-", "thinking | TF final count"),
        ("thinking", "ar_accuracy", ORANGE, "--", "thinking | AR final count"),
        ("thinking", "trace_exact", GREEN, ":", "thinking | AR trace exact"),
    ]
    for mode, metric, color, linestyle, label in accuracy_specs:
        frame = overall[(overall["mode"] == mode) & overall[metric].notna()]
        axes[0].plot(
            frame.step,
            frame[metric],
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            label=label,
        )
    axes[0].axhline(0.99, color=GRAY, linestyle=(0, (4, 3)), linewidth=1.2, label="99% threshold")
    axes[0].set_title("Balanced accuracy over gold counts 1-30")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("balanced exact-match accuracy")
    axes[0].set_ylim(-0.03, 1.04)

    loss_specs = [
        ("nonthinking", "total", BLUE, "--", "non-thinking | total"),
        ("nonthinking", "final_count", BLUE, "-", "non-thinking | final count"),
        ("thinking", "total", ORANGE, "--", "thinking | total"),
        ("thinking", "final_count", ORANGE, "-", "thinking | final count"),
        ("thinking", "trace_marker", GREEN, "-", "thinking | trace marker"),
        ("thinking", "trace_index", PURPLE, ":", "thinking | trace index"),
    ]
    for mode, component, color, linestyle, label in loss_specs:
        frame = eval_losses[
            (eval_losses["mode"] == mode) & (eval_losses["component"] == component)
        ].sort_values("step")
        if frame.empty:
            continue
        axes[1].plot(
            frame.step,
            frame.loss.clip(lower=1e-5),
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            marker="o",
            markersize=3.2,
            label=label,
        )
    axes[1].set_yscale("log")
    axes[1].set_title("Evaluation cross-entropy by supervised segment")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("mean token cross-entropy (log scale)")

    for ax in axes:
        ax.legend(loc="best", fontsize=8.3, frameon=True)
    fig.suptitle(
        "Learning dynamics across the full count range (v10 uses counts 1-30)",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_training_by_bin(eval_bins: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), sharex=True, sharey=True, constrained_layout=True)
    specs = [
        ("nonthinking", "tf_accuracy", BLUE, "-", "non-thinking | TF final count"),
        ("nonthinking", "ar_accuracy", BLUE, "--", "non-thinking | AR final count"),
        ("thinking", "tf_accuracy", ORANGE, "-", "thinking | TF final count"),
        ("thinking", "ar_accuracy", ORANGE, "--", "thinking | AR final count"),
        ("thinking", "trace_exact", GREEN, ":", "thinking | AR trace exact"),
    ]
    for ax, count_bin in zip(axes, COUNT_BINS):
        for mode, metric, color, linestyle, label in specs:
            frame = eval_bins[
                (eval_bins["mode"] == mode)
                & (eval_bins["count_bin"] == count_bin)
                & eval_bins[metric].notna()
            ].sort_values("step")
            ax.plot(
                frame.step,
                frame[metric],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                marker="o",
                markersize=3.2,
                label=label,
            )
        ax.axhline(0.99, color=GRAY, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.set_title(f"gold count {count_bin}", fontweight="bold")
        ax.set_xlabel("training step")
        ax.set_ylim(-0.03, 1.04)
    axes[0].set_ylabel("balanced exact-match accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.suptitle("Both models compared within each count-difficulty range", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _attention_rows_with_bins(attention_rows: pd.DataFrame) -> pd.DataFrame:
    rows = attention_rows.copy()
    rows["count_bin"] = pd.cut(rows["count"], [0, 10, 20, 30], labels=COUNT_BINS).astype(str)
    return rows


def save_broad_attention(attention_rows: pd.DataFrame, path: Path) -> None:
    rows = _attention_rows_with_bins(attention_rows)
    final_rows = rows[rows["query_kind"] == "final_count_query"]
    grouped = (
        final_rows.groupby(["mode", "count_bin", "layer", "head"], as_index=False)
        .agg(
            broad_attention_score=("broad_attention_score", "mean"),
            prompt_needles_mass=("prompt_needles_mass", "mean"),
            needle_entropy_normalized=("needle_entropy_normalized", "mean"),
            needle_effective_number=("needle_effective_number", "mean"),
        )
    )
    vmax = max(0.1, float(grouped.broad_attention_score.max()))
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.0), constrained_layout=True)
    for row_idx, mode in enumerate(("nonthinking", "thinking")):
        for col_idx, count_bin in enumerate(COUNT_BINS):
            frame = grouped[(grouped["mode"] == mode) & (grouped["count_bin"] == count_bin)]
            heatmap(
                axes[row_idx, col_idx],
                head_matrix(frame, "broad_attention_score"),
                f"{mode} | count {count_bin}",
                vmin=0,
                vmax=vmax,
                cmap="viridis",
            )
            axes[row_idx, col_idx].set_xlabel("head (0-based)")
            axes[row_idx, col_idx].set_ylabel("Layer (1-based)")
    fig.suptitle(
        "Broad prompt-needle aggregation score at the final-answer query",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_targeted_attention(attention_rows: pd.DataFrame, path: Path) -> None:
    rows = _attention_rows_with_bins(attention_rows)
    direct = rows[
        (rows["mode"] == "nonthinking") & (rows["query_kind"] == "final_count_query")
    ].copy()
    direct["single_target_concentration"] = direct["prompt_needles_mass"] * (
        1.0 - direct["needle_entropy_normalized"]
    )
    direct_grouped = (
        direct.groupby(["count_bin", "layer", "head"], as_index=False)
        .single_target_concentration.mean()
    )
    targeted = rows[
        (rows["mode"] == "thinking")
        & (rows["query_kind"] == "targeted_retrieval_query")
    ]
    targeted_grouped = (
        targeted.groupby(["count_bin", "layer", "head"], as_index=False)
        .agg(
            correct_prompt_needle_mass=("correct_prompt_needle_mass", "mean"),
            diagonal_dominance=("diagonal_dominance", "mean"),
            correct_top1=("correct_top1", "mean"),
        )
    )
    fig, axes = plt.subplots(4, 3, figsize=(14.8, 14.4), constrained_layout=True)
    row_specs = (
        (direct_grouped, "single_target_concentration", "Non-thinking final-query\nsingle-target concentration"),
        (targeted_grouped, "correct_prompt_needle_mass", "CoT k-to-k raw mass"),
        (targeted_grouped, "diagonal_dominance", "CoT diagonal dominance\nwithin prompt needles"),
        (targeted_grouped, "correct_top1", "CoT correct top-1\nwithin prompt needles"),
    )
    for row_idx, (source, metric, label) in enumerate(row_specs):
        for col_idx, count_bin in enumerate(COUNT_BINS):
            frame = source[source["count_bin"] == count_bin]
            heatmap(
                axes[row_idx, col_idx],
                head_matrix(frame, metric),
                f"{label} | count {count_bin}",
                vmin=0,
                vmax=1,
                cmap="viridis",
            )
            axes[row_idx, col_idx].set_xlabel("head (0-based)")
            axes[row_idx, col_idx].set_ylabel("Layer (1-based)")
    fig.suptitle(
        "Targeted-attention diagnostics: natural CoT k-to-k versus a non-thinking concentration control",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_single_ablation_panel(
    single: pd.DataFrame,
    *,
    mode: str,
    metric: str,
    title: str,
    path: Path,
) -> None:
    """Save one behavioral target as three count-bin heatmaps."""
    frames = [single[(single["mode"] == mode) & (single["count_bin"] == count_bin)] for count_bin in COUNT_BINS]
    finite_values = [
        head_matrix(frame, metric)[np.isfinite(head_matrix(frame, metric))]
        for frame in frames
    ]
    finite_values = [values for values in finite_values if values.size]
    vmax = max(0.05, max(float(values.max()) for values in finite_values)) if finite_values else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.25), constrained_layout=True)
    for ax, count_bin, frame in zip(axes, COUNT_BINS, frames):
        heatmap(
            ax,
            head_matrix(frame, metric),
            f"count {count_bin}",
            vmin=0,
            vmax=vmax,
            cmap="magma",
        )
        ax.set_xlabel("head index (0-based)")
        ax.set_ylabel("Layer (1-based)")
        ax.set_xticks(range(4), labels=["0", "1", "2", "3"])
        ax.set_yticks(range(4), labels=["1", "2", "3", "4"])
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _spearman_rank_correlation(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3 or frame.x.nunique() < 2 or frame.y.nunique() < 2:
        return math.nan
    return float(frame.x.rank(method="average").corr(frame.y.rank(method="average")))


def save_ablation_score_alignment(
    attention_rows: pd.DataFrame,
    single: pd.DataFrame,
    paths: dict[str, Path],
) -> pd.DataFrame:
    """Compare each descriptive score with causal necessity in a separate figure."""
    rows = _attention_rows_with_bins(attention_rows)
    specifications = (
        {
            "key": "nonthinking",
            "mode": "nonthinking",
            "query_kind": "final_count_query",
            "score": "broad_attention_score",
            "drop": "drop_final_count_accuracy",
            "label": "Non-thinking broad score vs final-count drop",
        },
        {
            "key": "targeted",
            "mode": "thinking",
            "query_kind": "targeted_retrieval_query",
            "score": "correct_prompt_needle_mass",
            "drop": "drop_trace_marker_accuracy",
            "label": "CoT k-to-k mass vs trace-marker drop",
        },
        {
            "key": "readout",
            "mode": "thinking",
            "query_kind": "final_count_query",
            "score": "trace_markers_mass",
            "drop": "drop_final_count_accuracy",
            "label": "CoT trace-readout mass vs final-count drop",
        },
    )
    summary_rows: list[dict[str, object]] = []
    for spec in specifications:
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.35), constrained_layout=True)
        descriptive = (
            rows[(rows["mode"] == spec["mode"]) & (rows["query_kind"] == spec["query_kind"])]
            .groupby(["count_bin", "layer", "head"], as_index=False)[spec["score"]]
            .mean()
        )
        for ax, count_bin in zip(axes, COUNT_BINS):
            causal = single[(single["mode"] == spec["mode"]) & (single["count_bin"] == count_bin)][
                ["layer", "head", spec["drop"]]
            ]
            frame = descriptive[descriptive["count_bin"] == count_bin].merge(causal, on=["layer", "head"], how="inner")
            rho = _spearman_rank_correlation(frame[spec["score"]], frame[spec["drop"]])
            ax.scatter(frame[spec["score"]], frame[spec["drop"]], s=43, color=BLUE, alpha=0.82)
            label_indices = set(frame.nlargest(4, spec["score"]).index)
            if frame[spec["drop"]].nunique() > 1:
                label_indices.update(frame.nlargest(4, spec["drop"]).index)
            for item in frame.loc[sorted(label_indices)].itertuples(index=False):
                ax.annotate(
                    f"L{int(item.layer) + 1}H{int(item.head)}",
                    (float(getattr(item, spec["score"])), float(getattr(item, spec["drop"]))),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7.5,
                    color="#334155",
                )
            ax.axhline(0, color="#64748b", linewidth=1)
            ax.set_title(f"count {count_bin} | Spearman rho={rho:.2f}")
            ax.set_xlabel(str(spec["score"]))
            ax.set_ylabel(str(spec["drop"]))
            summary_rows.append(
                {
                    "mechanism": spec["label"],
                    "mode": spec["mode"],
                    "count_bin": count_bin,
                    "descriptive_score": spec["score"],
                    "causal_drop": spec["drop"],
                    "spearman_rho": rho,
                    "n_heads": len(frame),
                }
            )
        fig.suptitle(str(spec["label"]), fontsize=15, fontweight="bold")
        fig.savefig(paths[str(spec["key"])], dpi=190, bbox_inches="tight")
        plt.close(fig)
    return pd.DataFrame(summary_rows)


def random_band(frame: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = frame.groupby("top_n")[metric]
    return grouped.mean().reset_index(), grouped.min().reset_index(), grouped.max().reset_index()


def save_cumulative_ablation_panel(
    cumulative: pd.DataFrame,
    *,
    mode: str,
    metric: str,
    family: str,
    title: str,
    path: Path,
) -> None:
    """Save one cumulative head-ablation target as three count-bin panels."""
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.45), sharey=True)
    for ax, count_bin in zip(axes, COUNT_BINS):
        frame = cumulative[(cumulative["mode"] == mode) & (cumulative["count_bin"] == count_bin)]
        top = frame[frame.family == family].sort_values("top_n")
        ax.plot(top.top_n, top[metric], color=BLUE, marker="o", ms=3, label="ranked top")
        random = frame[frame.family == "random"]
        for index, (_, path_frame) in enumerate(random.groupby("replicate")):
            path_frame = path_frame.sort_values("top_n")
            ax.plot(
                path_frame.top_n,
                path_frame[metric],
                color="#94a3b8",
                linewidth=0.85,
                alpha=0.28,
                label="random paths" if index == 0 else None,
            )
        mean, low, high = random_band(random, metric)
        ax.plot(mean.top_n, mean[metric], color=GRAY, linewidth=2.2, label="random mean")
        ax.fill_between(mean.top_n, low[metric], high[metric], color=GRAY, alpha=0.18, label="random min-max")
        random_four = mean.loc[mean.top_n == 4, metric]
        if not random_four.empty:
            ax.axhline(
                float(random_four.iloc[0]),
                color="#7c3aed",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label="random-4 mean",
            )
        baseline = frame[frame.family == "baseline"]
        if not baseline.empty:
            ax.axhline(float(baseline.iloc[0][metric]), color="#111827", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"count {count_bin}")
        ax.set_xlim(0.5, 16.5)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks([1, 2, 4, 8, 12, 16], labels=["1", "2", "4", "8", "12", "16"])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("number of globally masked heads")
        ax.set_ylabel("remaining accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.75, wspace=0.28)
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.88), frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_position_local_ablation_panel(
    local: pd.DataFrame,
    *,
    mechanism: str,
    metric: str,
    title: str,
    path: Path,
) -> None:
    """Plot semantic-query-local ablation with layer-matched controls."""
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.65), sharey=True)
    for ax, count_bin in zip(axes, COUNT_BINS):
        frame = local[
            (local["mechanism"] == mechanism)
            & (local["count_bin"] == count_bin)
        ]
        styles = (("ranked_top", BLUE, "-", "ranked top"),)
        for family, color, linestyle, label in styles:
            group = frame[frame.family == family].sort_values("top_n")
            ax.plot(
                group.top_n,
                group[metric],
                color=color,
                linestyle=linestyle,
                linewidth=2,
                marker="o",
                ms=3,
                label=label,
            )
        random = frame[frame.family == "layer_matched_random"]
        if not random.empty:
            for index, (_, path_frame) in enumerate(random.groupby("replicate")):
                path_frame = path_frame.sort_values("top_n")
                ax.plot(
                    path_frame.top_n,
                    path_frame[metric],
                    color="#94a3b8",
                    linewidth=0.85,
                    alpha=0.28,
                    label="same-layer random paths" if index == 0 else None,
                )
            mean, low, high = random_band(random, metric)
            ax.plot(
                mean.top_n,
                mean[metric],
                color=GRAY,
                linewidth=2,
                label="same-layer random: mean",
            )
            ax.fill_between(
                mean.top_n,
                low[metric],
                high[metric],
                color=GRAY,
                alpha=0.18,
                label="same-layer random: min-max",
            )
            random_four = mean.loc[mean.top_n == 4, metric]
            if not random_four.empty:
                ax.axhline(
                    float(random_four.iloc[0]),
                    color="#7c3aed",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                    label="same-layer random-4 mean",
                )
        baseline = frame[frame.family == "baseline"]
        if not baseline.empty:
            ax.axhline(
                float(baseline.iloc[0][metric]),
                color="#111827",
                linestyle=":",
                linewidth=1.3,
                label="no-ablation baseline",
            )
        ax.set_title(f"count {count_bin}")
        ax.set_xlim(0.5, 16.5)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks([1, 2, 4, 8, 12, 16])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("number of heads masked at the named query only")
        ax.set_ylabel("remaining teacher-forced accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.15, top=0.70, wspace=0.24)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.88),
        frameon=False,
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_retrieval_patch_panel(
    retrieval: pd.DataFrame,
    *,
    query_role: str,
    title: str,
    path: Path,
) -> None:
    """Plot clean-to-corrupt marker recovery without a low-score-head baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), sharex=True, sharey=True)
    for ax, count_bin in zip(axes, COUNT_BINS):
        frame = retrieval[(retrieval.query_role == query_role) & (retrieval.count_bin == count_bin)]
        ranked = (
            frame[frame.family == "targeted_top"]
            .groupby("top_n", as_index=False)
            .normalized_recovery.mean()
            .sort_values("top_n")
        )
        random = frame[frame.family == "random"].copy()
        for replicate, path_rows in random.groupby("replicate"):
            path_rows = path_rows.sort_values("top_n")
            ax.plot(
                path_rows.top_n,
                path_rows.normalized_recovery,
                color="#94a3b8",
                alpha=0.48,
                linewidth=1.1,
                label="individual random order" if int(replicate) == int(random.replicate.min()) else None,
            )
        mean, low, high = random_band(random, "normalized_recovery")
        ax.fill_between(mean.top_n, low.normalized_recovery, high.normalized_recovery, color=GRAY, alpha=0.14, label="random min-max")
        ax.plot(mean.top_n, mean.normalized_recovery, color=GRAY, marker="o", linewidth=2, label="random mean")
        random4 = random[random.top_n == 4].normalized_recovery.mean()
        ax.axhline(random4, color=PURPLE, linestyle=":", linewidth=1.7, label="random top-4 mean")
        ax.plot(ranked.top_n, ranked.normalized_recovery, color=BLUE, marker="o", linewidth=2.3, label="targeted ranked top")
        ax.axhline(0, color="#111827", linewidth=1)
        ax.axhline(1, color="#111827", linestyle="--", linewidth=1)
        ax.set_title(f"gold count {count_bin}")
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xlim(0.5, 16.5)
        ax.set_ylim(-0.08, 1.08)
        ax.set_xlabel("number of clean head-output slices patched")
        ax.set_ylabel("normalized clean-marker margin recovery")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.72, wspace=0.14)
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.86), frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_successor_patch_panel(
    summary: pd.DataFrame,
    *,
    direction: str,
    metric: str,
    title: str,
    path: Path,
) -> None:
    """Plot local M_k successor/close patching with row and stage controls."""
    family_styles = {
        "successor_top": (BLUE, "o", "successor-score ranked"),
        "targeted_top": (ORANGE, "s", "k-to-k targeted ranked"),
        "successor_wrong_row": (GREEN, "^", "successor heads, wrong donor row"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.9), sharex=True, sharey=True)
    for ax, count_bin in zip(axes, COUNT_BINS):
        frame = summary[
            (summary["direction"] == direction)
            & (summary["count_bin"] == count_bin)
        ].copy()
        for family, (color, marker, label) in family_styles.items():
            rows = frame[frame["family"] == family].sort_values("top_n")
            if rows.empty:
                continue
            ax.plot(
                rows["top_n"],
                rows[metric],
                color=color,
                marker=marker,
                linewidth=2.2,
                markersize=5,
                label=label,
            )

        random = frame[frame["family"] == "random"].copy()
        if not random.empty:
            for replicate, rows in random.groupby("replicate"):
                rows = rows.sort_values("top_n")
                ax.plot(
                    rows["top_n"],
                    rows[metric],
                    color="#a8b2c1",
                    alpha=0.42,
                    linewidth=1.0,
                    label=(
                        "individual random order"
                        if int(replicate) == int(random["replicate"].min())
                        else None
                    ),
                )
            mean, low, high = random_band(random, metric)
            ax.fill_between(
                mean["top_n"],
                low[metric],
                high[metric],
                color=GRAY,
                alpha=0.15,
                label="random min-max",
            )
            ax.plot(
                mean["top_n"],
                mean[metric],
                color=GRAY,
                linewidth=2.0,
                label="random mean",
            )

        ax.axhline(0, color="#111827", linewidth=1.0)
        if metric == "normalized_recovery":
            ax.axhline(1, color="#111827", linestyle="--", linewidth=1.0)
            ax.set_ylim(-0.08, 1.08)
            ax.set_ylabel("normalized continue/close margin recovery")
        else:
            ax.axhline(1, color="#111827", linestyle="--", linewidth=1.0)
            ax.set_ylim(-0.04, 1.04)
            ax.set_ylabel("patched target-decision accuracy")
        ax.set_title(f"count {count_bin}")
        ax.set_xlim(0.5, 16.5)
        ax.set_xticks([1, 2, 4, 8, 12, 16])
        ax.set_xlabel("number of head-output slices patched at M_k")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.16, top=0.70, wspace=0.20)
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.88),
        frameon=False,
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_successor_residual_logit_lens(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    """Show where continue evidence appears along the residual stream."""
    stage_order = [
        (layer, stage)
        for layer in range(4)
        for stage in ("resid_pre", "post_attn", "post_mlp")
    ]
    stage_labels = [
        f"L{layer + 1} pre" if stage == "resid_pre" else
        f"L{layer + 1} +Attn" if stage == "post_attn" else
        f"L{layer + 1} +MLP"
        for layer, stage in stage_order
    ]
    frame = summary[
        (summary["direction"] == "continue_into_close")
        & (summary["stage_type"] == "residual_logit_lens")
    ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.9), sharex=True, sharey=True)
    for ax, count_bin in zip(axes, COUNT_BINS):
        rows = frame[frame["count_bin"] == count_bin].set_index(["layer", "stage"])
        clean = [float(rows.loc[key, "clean_margin"]) for key in stage_order]
        corrupt = [float(rows.loc[key, "corrupt_margin"]) for key in stage_order]
        x = np.arange(len(stage_order))
        ax.plot(x, clean, color=BLUE, marker="o", linewidth=2.2, label="long prompt: continue target")
        ax.plot(x, corrupt, color=ORANGE, marker="o", linewidth=2.2, label="short prompt: same target")
        ax.axhline(0, color="#111827", linewidth=1)
        for boundary in (2.5, 5.5, 8.5):
            ax.axvline(boundary, color="#cbd5e1", linewidth=1)
        ax.set_title(f"count {count_bin}")
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels, rotation=55, ha="right", fontsize=8)
        ax.set_xlabel("residual stage at marker query M_k")
        ax.set_ylabel("target-aligned next-token logit margin")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.31, top=0.72, wspace=0.18)
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.86), frameon=False)
    fig.suptitle("Residual logit lens: where does continue evidence become linearly readable?", fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_successor_component_evidence(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    """Compare target-aligned evidence directly written by attention and MLP outputs."""
    frame = summary[
        (summary["direction"] == "continue_into_close")
        & (summary["stage_type"] == "direct_component_unembedding")
    ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7), sharex=True, sharey=True)
    for ax, count_bin in zip(axes, COUNT_BINS):
        rows = frame[frame["count_bin"] == count_bin]
        for stage, color, marker, label in (
            ("attn_out", BLUE, "o", "attention output"),
            ("mlp_out", ORANGE, "s", "MLP output"),
        ):
            selected = rows[rows["stage"] == stage].sort_values("layer")
            ax.plot(
                selected["layer"] + 1,
                selected["evidence_gap"],
                color=color,
                marker=marker,
                linewidth=2.3,
                label=label,
            )
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_title(f"count {count_bin}")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("layer producing the additive component")
        ax.set_ylabel("clean-minus-corrupt target-aligned component margin")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.16, top=0.72, wspace=0.18)
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.86), frameon=False)
    fig.suptitle("Direct unembedding diagnostic: attention routes evidence; MLP amplifies token evidence", fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_successor_sublayer_conversion(
    summary: pd.DataFrame,
    *,
    direction: str,
    title: str,
    path: Path,
) -> None:
    """Contrast attention-only patching with native MLP mediation and full-state patching."""
    styles = {
        "attn_direct_residual": (BLUE, "o", "donor Attn; receiver MLP frozen"),
        "attn_native_mlp": (GREEN, "s", "donor Attn; native MLP recomputed"),
        "mlp_out_only": (ORANGE, "^", "donor MLP output only"),
        "post_mlp_state": (GRAY, "D", "full donor post-MLP state"),
    }
    frame = summary[summary["direction"] == direction].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), sharex=True, sharey=True)
    for ax, count_bin in zip(axes, COUNT_BINS):
        rows = frame[frame["count_bin"] == count_bin]
        for intervention, (color, marker, label) in styles.items():
            selected = rows[rows["intervention"] == intervention].sort_values("layer")
            ax.plot(
                selected["layer"] + 1,
                selected["normalized_recovery"],
                color=color,
                marker=marker,
                linewidth=2.2,
                label=label,
            )
        ax.axhline(0, color="#111827", linewidth=1)
        ax.axhline(1, color="#111827", linestyle="--", linewidth=1)
        ax.set_title(f"count {count_bin}")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylim(-0.08, 1.08)
        ax.set_xlabel("patched layer at marker query M_k")
        ax.set_ylabel("normalized target-margin recovery")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.16, top=0.69, wspace=0.18)
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.87), frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_mlp_feature_concentration(
    concentration: pd.DataFrame,
    path: Path,
) -> None:
    """Plot how quickly ranked MLP features accumulate projected token evidence."""
    direction_rows = (
        ("continue_into_close", "Continue evidence -> close receiver"),
        ("close_into_continue", "Close evidence -> continue receiver"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex=True, sharey=True)
    colors = {"1-10": BLUE, "11-20": ORANGE, "21-30": GREEN}
    for row_index, (direction, direction_label) in enumerate(direction_rows):
        for col_index, layer in enumerate((2, 3)):
            ax = axes[row_index, col_index]
            frame = concentration[
                (concentration["direction"] == direction)
                & (concentration["layer"] == layer)
            ]
            for count_bin in COUNT_BINS:
                rows = frame[frame["count_bin"] == count_bin].sort_values("support_size")
                ax.plot(
                    rows["support_size"],
                    rows["positive_evidence_fraction"],
                    color=colors[count_bin],
                    marker="o",
                    linewidth=2.2,
                    label=f"count {count_bin}: positive evidence",
                )
                ax.plot(
                    rows["support_size"],
                    rows["absolute_evidence_fraction"],
                    color=colors[count_bin],
                    linestyle="--",
                    linewidth=1.7,
                    alpha=0.8,
                    label=f"count {count_bin}: absolute evidence",
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 16, 64, 256, 1024])
            ax.set_xticklabels(["1", "4", "16", "64", "256", "1024"])
            ax.set_ylim(-0.02, 1.03)
            ax.set_title(f"{direction_label}\nLayer {layer + 1} post-GELU features")
            ax.set_xlabel("number of top-ranked MLP features")
            ax.set_ylabel("fraction of total projected evidence")
            ax.axhline(1, color="#111827", linestyle=":", linewidth=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.10, top=0.76, hspace=0.37, wspace=0.16)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.89),
        frameon=False,
    )
    fig.suptitle(
        "MLP feature evidence concentration at the marker query M_k",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_mlp_feature_patch_panel(
    summary: pd.DataFrame,
    *,
    direction: str,
    title: str,
    path: Path,
) -> None:
    """Plot held-out post-GELU feature replacement and sparse-direction recovery."""
    frame = summary[summary["direction"] == direction].copy()
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.2), sharex=True, sharey=True)
    styles = {
        "ranked_feature_replacement": (BLUE, "o", "ranked feature replacement"),
        "sparse_mean_direction": (GREEN, "s", "sparse mean-direction transport"),
        "random_feature_replacement": (GRAY, "^", "matched random features"),
    }
    for row_index, layer in enumerate((2, 3)):
        for col_index, count_bin in enumerate(COUNT_BINS):
            ax = axes[row_index, col_index]
            rows = frame[
                (frame["layer"] == layer) & (frame["count_bin"] == count_bin)
            ]
            for family, (color, marker, label) in styles.items():
                selected = rows[rows["family"] == family].sort_values("support_size")
                ax.plot(
                    selected["support_size"],
                    selected["normalized_recovery"],
                    color=color,
                    marker=marker,
                    linewidth=2.1,
                    label=label,
                )
                if family == "random_feature_replacement" and not selected.empty:
                    mean = selected["normalized_recovery"].to_numpy(float)
                    std = selected["normalized_recovery_std"].fillna(0).to_numpy(float)
                    ax.fill_between(
                        selected["support_size"].to_numpy(float),
                        mean - std,
                        mean + std,
                        color=color,
                        alpha=0.14,
                    )
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 16, 64, 256, 1024])
            ax.set_xticklabels(["1", "4", "16", "64", "256", "1024"])
            ax.axhline(0, color="#111827", linewidth=1)
            ax.axhline(1, color="#111827", linestyle="--", linewidth=1)
            ax.set_ylim(-0.08, 1.08)
            ax.set_title(f"Layer {layer + 1} | count {count_bin}")
            ax.set_xlabel("patched feature support size")
            ax.set_ylabel("normalized target-margin recovery")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.10, top=0.80, hspace=0.30, wspace=0.18)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.90),
        frameon=False,
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_nested_patch_panel(
    regression: pd.DataFrame,
    *,
    mode: str,
    title: str,
    path: Path,
) -> None:
    """Plot donor-to-receiver count transport for one final-query mechanism."""
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), sharex=True, sharey=True)
    selected = regression[(regression["mode"] == mode) & regression.count_bin.isin(COUNT_BINS)]
    y_min = min(-0.08, float(selected[selected.family.isin(["primary_top", "random"])].slope.min()) - 0.04)
    y_max = max(1.08, float(selected[selected.family.isin(["primary_top", "random"])].slope.max()) + 0.04)
    for ax, count_bin in zip(axes, COUNT_BINS):
        frame = selected[selected.count_bin == count_bin]
        ranked = frame[frame.family == "primary_top"].sort_values("top_n")
        random = frame[frame.family == "random"].copy()
        for replicate, path_rows in random.groupby("replicate"):
            path_rows = path_rows.sort_values("top_n")
            ax.plot(
                path_rows.top_n,
                path_rows.slope,
                color="#94a3b8",
                alpha=0.48,
                linewidth=1.1,
                label="individual random order" if int(replicate) == int(random.replicate.min()) else None,
            )
        grouped = random.groupby("top_n").slope
        mean, low, high = grouped.mean(), grouped.min(), grouped.max()
        ax.fill_between(mean.index, low.values, high.values, color=GRAY, alpha=0.14, label="random min-max")
        ax.plot(mean.index, mean.values, color=GRAY, marker="o", linewidth=2, label="random mean")
        random4 = random[random.top_n == 4].slope.mean()
        ax.axhline(random4, color=PURPLE, linestyle=":", linewidth=1.7, label="random top-4 mean")
        ax.plot(ranked.top_n, ranked.slope, color=BLUE, marker="o", linewidth=2.3, label="mechanism-ranked top")
        ax.axhline(0, color="#111827", linewidth=1)
        ax.axhline(1, color="#111827", linestyle="--", linewidth=1)
        ax.set_title(f"receiver count {count_bin}")
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xlim(0.5, 16.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("number of donor head-output slices patched")
        ax.set_ylabel("transport slope: expected-count shift / donor offset")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.72, wspace=0.14)
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.86), frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_steering(steering: pd.DataFrame, path: Path) -> None:
    sites = (
        ("nonthinking_final_answer", "Non-thinking natural final"),
        ("thinking_final_answer", "CoT natural final"),
        ("thinking_fixed_trace_answer", "CoT counterfactual fixed-15 trace"),
    )
    colors = ["#93c5fd", "#2563eb", "#7c3aed", "#dc2626"]
    fig, axes = plt.subplots(3, 3, figsize=(14.2, 10.6), sharex=True, constrained_layout=True)
    for row, (site, label) in enumerate(sites):
        for col, count_bin in enumerate(COUNT_BINS):
            ax = axes[row, col]
            frame = steering[(steering.site == site) & (steering.count_bin == count_bin)]
            for layer in sorted(frame.layer.unique()):
                group = frame[frame.layer == layer].sort_values("alpha")
                ax.plot(
                    group.alpha,
                    group.mean_causal_expected_shift,
                    color=colors[int(layer)],
                    marker="o",
                    ms=3,
                    label=f"after Layer {int(layer) + 1}",
                )
            ax.plot([-5, 5], [-5, 5], color=GRAY, linestyle="--", linewidth=1, label="ideal y=alpha")
            ax.axhline(0, color="#111827", linewidth=1)
            ax.set_title(f"{label} | count {count_bin}")
            ax.set_xlabel("steering coefficient alpha")
    for row in range(3):
        axes[row, 0].set_ylabel("mean expected-count shift")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.025))
    fig.suptitle("Adjacent-count direction steering dose response by count range", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def aggregate_geometry_path_regression(detail: pd.DataFrame) -> pd.DataFrame:
    """Fit causal output transport across all donor offsets in each panel."""
    rows = []
    groups = ["site", "mode", "count_bin", "layer", "method", "alpha"]
    for keys, frame in detail.groupby(groups, dropna=False):
        clean = frame[["intended_count_shift", "causal_expected_shift"]].dropna()
        slope = intercept = r2 = math.nan
        if len(clean) >= 2 and clean.intended_count_shift.nunique() >= 2:
            x = clean.intended_count_shift.to_numpy(dtype=float)
            y = clean.causal_expected_shift.to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            prediction = design @ beta
            denominator = float(((y - y.mean()) ** 2).sum())
            slope = float(beta[1])
            intercept = float(beta[0])
            r2 = (
                1.0 - float(((y - prediction) ** 2).sum()) / denominator
                if denominator > 1e-12
                else math.nan
            )
        rows.append(
            {
                **dict(zip(groups, keys)),
                "n": len(frame),
                "transport_slope": slope,
                "transport_intercept": intercept,
                "transport_r2": r2,
                "path_tracking_mae": float(frame.path_tracking_error.mean()),
                "intended_hit": float(frame.follows_intended_integer.mean()),
            }
        )
    return pd.DataFrame(rows)


def save_adjacent_geometry_transport(regression: pd.DataFrame, path: Path) -> None:
    sites = (
        ("nonthinking_final_answer", "Non-thinking final-answer state"),
        ("thinking_final_answer", "CoT natural final-answer state"),
        ("thinking_fixed_trace_answer", "CoT fixed-15 control"),
    )
    methods = (
        ("adjacent_centroid_transplant", "full centroid transplant", BLUE, "-"),
        ("adjacent_delta_transport", "receiver residual + centroid delta", ORANGE, "--"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(14.4, 10.8), sharex=True, sharey=True)
    for row, (site, site_label) in enumerate(sites):
        for col, count_bin in enumerate(COUNT_BINS):
            ax = axes[row, col]
            frame = regression[(regression.site == site) & (regression.count_bin == count_bin)]
            for method, label, color, linestyle in methods:
                group = frame[frame.method == method].sort_values("layer")
                ax.plot(
                    group.layer + 1,
                    group.transport_slope,
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    linewidth=2.2,
                    label=label,
                )
            ax.axhline(0, color="#111827", linewidth=1)
            ax.axhline(1, color=GRAY, linestyle="--", linewidth=1.2, label="ideal one-count transport")
            ax.set_title(f"{site_label} | count {count_bin}")
            ax.set_xticks([1, 2, 3, 4])
            if row == len(sites) - 1:
                ax.set_xlabel("residual intervention after Layer", labelpad=5)
    for row in range(3):
        axes[row, 0].set_ylabel("transport slope")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.012), frameon=False)
    fig.suptitle(
        "Adjacent count transport: full centroid versus residual-preserving delta",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.925, bottom=0.105, hspace=0.34, wspace=0.10)
    fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def save_nonadjacent_path_transport(
    regression: pd.DataFrame,
    *,
    method: str,
    title: str,
    path: Path,
) -> None:
    sites = (
        ("nonthinking_final_answer", "Non-thinking final-answer state"),
        ("thinking_final_answer", "CoT natural final-answer state"),
    )
    colors = {0.25: "#93c5fd", 0.5: BLUE, 0.75: PURPLE, 1.0: RED}
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.6), sharex=True, sharey=True)
    for row, (site, site_label) in enumerate(sites):
        for col, count_bin in enumerate(COUNT_BINS):
            ax = axes[row, col]
            frame = regression[
                (regression.site == site)
                & (regression.count_bin == count_bin)
                & (regression.method == method)
            ]
            for alpha in sorted(frame.alpha.dropna().unique()):
                group = frame[frame.alpha == alpha].sort_values("layer")
                ax.plot(
                    group.layer + 1,
                    group.transport_slope,
                    color=colors.get(float(alpha), GRAY),
                    marker="o",
                    linewidth=2.0,
                    label=f"alpha={float(alpha):g}",
                )
            ax.axhline(0, color="#111827", linewidth=1)
            ax.axhline(1, color=GRAY, linestyle="--", linewidth=1.2, label="ideal path transport")
            ax.set_title(f"{site_label} | count {count_bin}")
            ax.set_xticks([1, 2, 3, 4])
            if row == len(sites) - 1:
                ax.set_xlabel("residual intervention after Layer", labelpad=5)
    for row in range(2):
        axes[row, 0].set_ylabel("transport slope")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.014), frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.982)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.91, bottom=0.15, hspace=0.34, wspace=0.10)
    fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def save_curve_chord_tracking(detail: pd.DataFrame, path: Path) -> None:
    frame = detail[
        detail.method.isin(("nonadjacent_chord_transport", "nonadjacent_curve_transport"))
        & (detail.alpha < 1.0)
        & detail.site.isin(("nonthinking_final_answer", "thinking_final_answer"))
    ]
    summary = (
        frame.groupby(["site", "count_bin", "layer", "method"], as_index=False)
        .path_tracking_error.mean()
    )
    sites = (
        ("nonthinking_final_answer", "Non-thinking final-answer state"),
        ("thinking_final_answer", "CoT natural final-answer state"),
    )
    methods = (
        ("nonadjacent_chord_transport", "straight endpoint chord", ORANGE, "--"),
        ("nonadjacent_curve_transport", "piecewise centroid curve", GREEN, "-"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.6), sharex=True)
    for row, (site, site_label) in enumerate(sites):
        for col, count_bin in enumerate(COUNT_BINS):
            ax = axes[row, col]
            panel = summary[(summary.site == site) & (summary.count_bin == count_bin)]
            for method, label, color, linestyle in methods:
                group = panel[panel.method == method].sort_values("layer")
                ax.plot(
                    group.layer + 1,
                    group.path_tracking_error,
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    linewidth=2.2,
                    label=label,
                )
            ax.set_title(f"{site_label} | count {count_bin}")
            ax.set_xticks([1, 2, 3, 4])
            if row == len(sites) - 1:
                ax.set_xlabel("residual intervention after Layer", labelpad=5)
    for row in range(2):
        axes[row, 0].set_ylabel("mean absolute path-tracking error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.014), frameon=False)
    fig.suptitle(
        "Does the curved centroid path track intermediate counts better than a chord?",
        fontsize=15,
        fontweight="bold",
        y=0.982,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.91, bottom=0.15, hspace=0.34, wspace=0.12)
    fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def save_transplants(
    centroid_regression: pd.DataFrame,
    raw_regression: pd.DataFrame,
    path: Path,
) -> None:
    sites = (
        ("nonthinking_final_answer", "Non-thinking natural final"),
        ("thinking_final_answer", "CoT natural final"),
        ("thinking_fixed_trace_answer", "CoT counterfactual fixed-15 trace"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(14.2, 10.4), sharex=True, sharey=True, constrained_layout=True)
    for row, (site, label) in enumerate(sites):
        for col, count_bin in enumerate(COUNT_BINS):
            ax = axes[row, col]
            centroid = centroid_regression[
                (centroid_regression.site == site) & (centroid_regression.count_bin == count_bin)
            ].sort_values("layer")
            raw = raw_regression[
                (raw_regression.site == site) & (raw_regression.count_bin == count_bin)
            ].sort_values("layer")
            ax.plot(centroid.layer + 1, centroid.slope, color=BLUE, marker="o", label="independent count centroid")
            ax.plot(raw.layer + 1, raw.slope, color=ORANGE, marker="o", linestyle="--", label="single donor residual")
            ax.axhline(0, color="#111827", linewidth=1)
            ax.axhline(1, color="#111827", linestyle="--", linewidth=1)
            ax.set_title(f"{label} | count {count_bin}")
            ax.set_xlabel("residual replaced after Layer")
            ax.set_xticks([1, 2, 3, 4])
            ax.set_ylim(-0.15, 1.1)
    for row in range(3):
        axes[row, 0].set_ylabel("slope: expected shift / donor offset")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Residual-state transport by count range", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_pca_variance(geometry: pd.DataFrame, path: Path) -> None:
    sites = [
        "nonthinking_final_answer",
        "thinking_final_answer",
        "thinking_fixed_trace_answer",
        "thinking_trace_index",
        "thinking_trace_marker",
    ]
    labels = {
        "nonthinking_final_answer": "Non-thinking final",
        "thinking_final_answer": "CoT natural final",
        "thinking_fixed_trace_answer": "CoT counterfactual fixed-15 final",
        "thinking_trace_index": "CoT trace index",
        "thinking_trace_marker": "CoT trace marker",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    axes = axes.ravel()
    for ax, site in zip(axes, sites):
        frame = geometry[geometry.site == site].sort_values("layer")
        matrix = frame[[f"pc{i}_variance" for i in range(1, 7)]].to_numpy(dtype=float)
        heatmap(
            ax,
            matrix,
            labels[site],
            vmin=0,
            vmax=1,
            cmap="viridis",
            xlabels=[f"PC{i}" for i in range(1, 7)],
            ylabels=[f"Layer {i}" for i in range(1, 5)],
        )
        ax.set_xlabel("principal component")
        ax.set_ylabel("site depth")
    ax = axes[-1]
    for site, color in zip(sites, (BLUE, ORANGE, GREEN, PURPLE, RED)):
        frame = geometry[geometry.site == site].sort_values("layer")
        ax.plot(frame.layer + 1, frame.pc6_cumulative, color=color, marker="o", label=labels[site])
    ax.set_title("Cumulative variance retained by PC1-PC6", fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("cumulative explained variance")
    ax.set_ylim(0, 1.04)
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(fontsize=7)
    fig.suptitle("PCA variance of the 30 exact-count mean residuals", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_pca_static(coordinates: pd.DataFrame, path: Path) -> None:
    sites = [
        ("nonthinking_final_answer", "Non-thinking final"),
        ("thinking_final_answer", "CoT natural final"),
        ("thinking_fixed_trace_answer", "CoT counterfactual fixed-15 final"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(15.2, 10.6), constrained_layout=True)
    norm = plt.Normalize(1, 30)
    scatter = None
    for row, (site, label) in enumerate(sites):
        for layer in range(4):
            ax = axes[row, layer]
            frame = coordinates[(coordinates.site == site) & (coordinates.layer == layer)].sort_values("state_label")
            scatter = ax.scatter(frame.pc1, frame.pc2, c=frame.state_label, cmap="turbo", norm=norm, s=30, zorder=3)
            ax.plot(frame.pc1, frame.pc2, color="#94a3b8", linewidth=1.2, zorder=2)
            for count in (1, 10, 20, 30):
                point = frame[frame.state_label == count]
                if len(point):
                    ax.text(float(point.pc1.iloc[0]), float(point.pc2.iloc[0]), str(count), fontsize=7)
            ax.set_title(f"{label} | Layer {layer + 1}", fontsize=9.5)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.015, pad=0.012)
    colorbar.set_label("exact count label 1-30")
    fig.suptitle("Mean-first count-state manifolds: one point per exact count", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def trace_token_centroid_pca(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one PCA basis to the interleaved mean states I1,M1,...,I30,M30."""
    arrays = np.load(run_dir / "analysis" / "state_causal" / "centroids.npz")
    coordinate_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    for layer in range(1, 5):
        labels = sorted(
            set(
                int(key.rsplit("__C", 1)[1])
                for key in arrays.files
                if key.startswith(f"thinking_trace_index__L{layer}__C")
            )
            & set(
                int(key.rsplit("__C", 1)[1])
                for key in arrays.files
                if key.startswith(f"thinking_trace_marker__L{layer}__C")
            )
        )
        values: list[np.ndarray] = []
        metadata: list[tuple[int, int, str]] = []
        for k in labels:
            values.append(arrays[f"thinking_trace_index__L{layer}__C{k}"])
            metadata.append((2 * k - 1, k, "index"))
            values.append(arrays[f"thinking_trace_marker__L{layer}__C{k}"])
            metadata.append((2 * k, k, "marker"))
        matrix = np.stack(values)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        _, singular, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:6].T
        eigenvalues = singular**2
        ratios = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
        effective = float(eigenvalues.sum() ** 2 / max(float((eigenvalues**2).sum()), 1e-12))
        geometry_rows.append(
            {
                "layer": layer - 1,
                **{
                    f"pc{index + 1}_variance": float(ratios[index]) if index < len(ratios) else 0.0
                    for index in range(6)
                },
                "pc6_cumulative": float(ratios[:6].sum()),
                "effective_dimension": effective,
            }
        )
        for (token_order, progress_k, token_type), coordinate in zip(metadata, projected):
            coordinate_rows.append(
                {
                    "layer": layer - 1,
                    "token_order": token_order,
                    "progress_k": progress_k,
                    "token_type": token_type,
                    **{
                        f"pc{index + 1}": float(coordinate[index]) if index < coordinate.shape[0] else 0.0
                        for index in range(6)
                    },
                }
            )
    return pd.DataFrame(coordinate_rows), pd.DataFrame(geometry_rows)


def centroid_absolute_spread(run_dir: Path) -> pd.DataFrame:
    """Measure absolute centroid spread in the original 256-dimensional residual space."""
    arrays = np.load(run_dir / "analysis" / "state_causal" / "centroids.npz")
    rows: list[dict[str, object]] = []
    sites = (
        "nonthinking_final_answer",
        "thinking_final_answer",
        "thinking_fixed_trace_answer",
        "thinking_trace_index",
        "thinking_trace_marker",
    )
    for site in sites:
        for layer in range(1, 5):
            prefix = f"{site}__L{layer}__C"
            labeled_keys = sorted(
                (
                    (int(key.rsplit("__C", 1)[1]), key)
                    for key in arrays.files
                    if key.startswith(prefix)
                ),
                key=lambda item: item[0],
            )
            if not labeled_keys:
                continue
            matrix = np.stack([arrays[key] for _, key in labeled_keys]).astype(np.float64)
            centered = matrix - matrix.mean(axis=0, keepdims=True)
            adjacent = np.diff(matrix, axis=0)
            rows.append(
                {
                    "site": site,
                    "layer": layer - 1,
                    "rms_centroid_radius": float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))),
                    "mean_adjacent_distance": float(np.mean(np.linalg.norm(adjacent, axis=1))) if len(adjacent) else 0.0,
                    "total_between_centroid_variance": float(np.mean(np.sum(centered**2, axis=1))),
                }
            )
    return pd.DataFrame(rows)


def pca_axis_options() -> str:
    return "".join(
        f'<option value="{a},{b},{c}">PC{a + 1} / PC{b + 1} / PC{c + 1}</option>'
        for a, b, c in combinations(range(6), 3)
    )


def interactive_pca(coordinates: pd.DataFrame, geometry: pd.DataFrame) -> str:
    payload: dict[str, dict[str, object]] = {}
    for (site, layer), frame in coordinates.groupby(["site", "layer"]):
        key = f"{site}|{int(layer)}"
        geo = geometry[(geometry.site == site) & (geometry.layer == layer)].iloc[0]
        payload[key] = {
            "points": frame.sort_values("state_label")[["state_label", "pc1", "pc2", "pc3", "pc4", "pc5", "pc6"]].round(6).values.tolist(),
            "variance": [float(geo[f"pc{i}_variance"]) for i in range(1, 7)],
            "effectiveDimension": float(geo.effective_dimension),
            "turningAngle": float(geo.mean_turning_angle_degrees),
        }
    data_json = json.dumps(payload, separators=(",", ":"))
    site_options = [
        ("nonthinking_final_answer", "Non-thinking final-answer query"),
        ("thinking_final_answer", "CoT natural final-answer query"),
        ("thinking_fixed_trace_answer", "CoT fixed-15 counterfactual <Ans> query"),
        ("thinking_trace_index", "CoT trace index-token state <k> (grouped by k)"),
        ("thinking_trace_marker", "CoT trace marker state M_k (grouped by k)"),
    ]
    options = "".join(f'<option value="{key}">{html.escape(label)}</option>' for key, label in site_options)
    template = """
    <figure class="interactive-figure">
      <h3>Interactive 3D count/progress-centroid manifold</h3>
      <div class="controls">
        <label>Model / semantic site <select id="v10-pca-site">__OPTIONS__</select></label>
        <label>Layer <select id="v10-pca-layer"><option value="0">Layer 1</option><option value="1">Layer 2</option><option value="2">Layer 3</option><option value="3">Layer 4</option></select></label>
        <label>Count range <select id="v10-pca-bin"><option value="all">1-30</option><option value="1-10">1-10</option><option value="11-20">11-20</option><option value="21-30">21-30</option></select></label>
        <label>Displayed axes <select id="v10-pca-axes">__AXIS_OPTIONS__</select></label>
        <button id="v10-pca-reset" type="button">Reset view</button>
      </div>
      <div id="v10-pca-stats" class="stats"></div>
      <canvas id="v10-pca-canvas" aria-label="Rotatable three-dimensional PCA view"></canvas>
      <figcaption><b>操作：</b>拖拽旋转，切换 semantic site、Layer、标签区间和三条 PC 轴。前三个 final-query site 的点编号是 prompt 的真实 count <i>n</i>；后两个 trace site 的点编号是 trace progress <i>k</i>。fixed-15 site 始终在同一条 15-step 反事实 trace 后的 <code>&lt;Ans&gt;</code> 位置读 hidden state，并按真实 prompt count <i>n</i> 分组；它不是“count=15”这一类。trace index 与 trace marker 则分别在 <code>&lt;k&gt;</code>、<code>M_k</code> token 位置读 residual，并按 <i>k</i> 分组。每个 site×Layer 的 PCA 独立拟合，因此切换下拉项时 PC 方向和尺度会变化，不能把不同选项的屏幕坐标当作同一全局基底。</figcaption>
    </figure>
    <script>
    (() => {
      const data = __DATA__;
      const site = document.getElementById('v10-pca-site');
      const layer = document.getElementById('v10-pca-layer');
      const bin = document.getElementById('v10-pca-bin');
      const axes = document.getElementById('v10-pca-axes');
      const reset = document.getElementById('v10-pca-reset');
      const stats = document.getElementById('v10-pca-stats');
      const canvas = document.getElementById('v10-pca-canvas');
      const ctx = canvas.getContext('2d');
      let yaw=-0.65,pitch=0.42,dragging=false,lastX=0,lastY=0;
      function key(){ return site.value+'|'+layer.value; }
      function selected(){
        const ids=axes.value.split(',').map(Number), range=bin.value;
        return data[key()].points.filter(p=>range==='all'||(range==='1-10'&&p[0]<=10)||(range==='11-20'&&p[0]>=11&&p[0]<=20)||(range==='21-30'&&p[0]>=21)).map(p=>({count:p[0],v:[p[ids[0]+1],p[ids[1]+1],p[ids[2]+1]]}));
      }
      function camera(v){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);const x=cy*v[0]+sy*v[2],z=-sy*v[0]+cy*v[2];return[x,cp*v[1]-sp*z,sp*v[1]+cp*z];}
      function color(count){const t=(count-1)/29,h=225-220*t;return `hsl(${h},78%,52%)`;}
      function draw(){
        const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);
        const raw=selected(),rot=raw.map(p=>({count:p.count,v:camera(p.v)}));const extent=Math.max(1,...rot.flatMap(p=>[Math.abs(p.v[0]),Math.abs(p.v[1])]));const scale=.39*Math.min(rect.width,rect.height)/extent,cx=rect.width/2,cy=rect.height/2;const pts=rot.map(p=>({count:p.count,x:cx+scale*p.v[0],y:cy-scale*p.v[1],z:p.v[2]}));
        ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(30,cy);ctx.lineTo(rect.width-30,cy);ctx.moveTo(cx,25);ctx.lineTo(cx,rect.height-25);ctx.stroke();
        if(pts.length){ctx.strokeStyle='#64748b';ctx.lineWidth=1.7;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.stroke();}
        [...pts].sort((a,b)=>a.z-b.z).forEach(p=>{ctx.beginPath();ctx.fillStyle=color(p.count);ctx.arc(p.x,p.y,6,0,2*Math.PI);ctx.fill();ctx.fillStyle='#111827';ctx.font='12px sans-serif';ctx.fillText(String(p.count),p.x+7,p.y-7);});
        const item=data[key()],v=item.variance,ids=axes.value.split(',').map(Number),cum6=v.reduce((a,b)=>a+b,0);stats.textContent=`Displayed: ${ids.map(i=>'PC'+(i+1)).join('/')} | variance ${ids.map(i=>(100*v[i]).toFixed(1)+'%').join(', ')} | PC1-6 cumulative ${(100*cum6).toFixed(1)}% | effective dimension ${item.effectiveDimension.toFixed(2)} | mean adjacent turning angle ${item.turningAngle.toFixed(1)} deg`;
      }
      [site,layer,bin,axes].forEach(el=>el.addEventListener('change',draw));reset.addEventListener('click',()=>{yaw=-.65;pitch=.42;draw();});canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);});canvas.addEventListener('pointermove',e=>{if(!dragging)return;yaw+=(e.clientX-lastX)*.01;pitch=Math.max(-1.35,Math.min(1.35,pitch+(e.clientY-lastY)*.01));lastX=e.clientX;lastY=e.clientY;draw();});canvas.addEventListener('pointerup',()=>dragging=false);new ResizeObserver(draw).observe(canvas);draw();
    })();
    </script>
    """
    return (
        template.replace("__DATA__", data_json)
        .replace("__OPTIONS__", options)
        .replace("__AXIS_OPTIONS__", pca_axis_options())
    )


def interactive_trace_pca(coordinates: pd.DataFrame, geometry: pd.DataFrame) -> str:
    payload: dict[str, dict[str, object]] = {}
    for layer, frame in coordinates.groupby("layer"):
        geo = geometry[geometry.layer == layer].iloc[0]
        payload[str(int(layer))] = {
            "points": frame.sort_values("token_order")[
                ["token_order", "progress_k", "token_type", "pc1", "pc2", "pc3", "pc4", "pc5", "pc6"]
            ].round(6).values.tolist(),
            "variance": [float(geo[f"pc{i}_variance"]) for i in range(1, 7)],
            "effectiveDimension": float(geo.effective_dimension),
        }
    data_json = json.dumps(payload, separators=(",", ":"))
    template = """
    <figure class="interactive-figure">
      <h3>Interactive 3D CoT trace-token trajectory</h3>
      <div class="controls">
        <label>Layer <select id="v10-trace-pca-layer"><option value="0">Layer 1</option><option value="1">Layer 2</option><option value="2">Layer 3</option><option value="3">Layer 4</option></select></label>
        <label>Trace progress <select id="v10-trace-pca-bin"><option value="all">k=1-30</option><option value="1-10">k=1-10</option><option value="11-20">k=11-20</option><option value="21-30">k=21-30</option></select></label>
        <label>Displayed axes <select id="v10-trace-pca-axes">__AXIS_OPTIONS__</select></label>
        <button id="v10-trace-pca-reset" type="button">Reset view</button>
      </div>
      <div id="v10-trace-pca-stats" class="stats"></div>
      <canvas id="v10-trace-pca-canvas" aria-label="Rotatable three-dimensional CoT trace-state trajectory"></canvas>
      <figcaption><b>每个点是什么：</b>对所有包含第 k 步的 held-out examples，分别在 trace 数字 <code>&lt;k&gt;</code> 与紧随其后的 marker <code>M_k</code> 位置提取某一 Layer 后的 256 维 residual，再先按 token type×k 求均值。每个 Layer 把 60 个均值状态 <code>&lt;1&gt;,M1,...,&lt;30&gt;,M30</code> 放进<b>同一个</b> PCA 基底，灰线按真实 token 顺序连接。圆点表示 index，方块表示 marker；颜色编码进度 k。该图展示的是均值轨迹，不显示同一步内部的样本方差，也不把不同 Layer 的 PC 轴视为同一方向。</figcaption>
    </figure>
    <script>
    (() => {
      const data = __DATA__;
      const layer = document.getElementById('v10-trace-pca-layer');
      const bin = document.getElementById('v10-trace-pca-bin');
      const axes = document.getElementById('v10-trace-pca-axes');
      const reset = document.getElementById('v10-trace-pca-reset');
      const stats = document.getElementById('v10-trace-pca-stats');
      const canvas = document.getElementById('v10-trace-pca-canvas');
      const ctx = canvas.getContext('2d');
      let yaw=-0.65,pitch=0.42,dragging=false,lastX=0,lastY=0;
      function selected(){
        const ids=axes.value.split(',').map(Number), range=bin.value;
        return data[layer.value].points
          .filter(p=>range==='all'||(range==='1-10'&&p[1]<=10)||(range==='11-20'&&p[1]>=11&&p[1]<=20)||(range==='21-30'&&p[1]>=21))
          .map(p=>({order:p[0],k:p[1],type:p[2],v:[p[ids[0]+3],p[ids[1]+3],p[ids[2]+3]]}));
      }
      function camera(v){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);const x=cy*v[0]+sy*v[2],z=-sy*v[0]+cy*v[2];return[x,cp*v[1]-sp*z,sp*v[1]+cp*z];}
      function color(k){const t=(k-1)/29,h=225-220*t;return `hsl(${h},78%,52%)`;}
      function draw(){
        const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);
        const raw=selected(),rot=raw.map(p=>({...p,v:camera(p.v)}));const extent=Math.max(1,...rot.flatMap(p=>[Math.abs(p.v[0]),Math.abs(p.v[1])]));const scale=.39*Math.min(rect.width,rect.height)/extent,cx=rect.width/2,cy=rect.height/2;const pts=rot.map(p=>({...p,x:cx+scale*p.v[0],y:cy-scale*p.v[1],z:p.v[2]}));
        ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(30,cy);ctx.lineTo(rect.width-30,cy);ctx.moveTo(cx,25);ctx.lineTo(cx,rect.height-25);ctx.stroke();
        if(pts.length){ctx.strokeStyle='#64748b';ctx.lineWidth=1.5;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.stroke();}
        [...pts].sort((a,b)=>a.z-b.z).forEach(p=>{ctx.fillStyle=color(p.k);ctx.strokeStyle=p.type==='index'?'#1d4ed8':'#15803d';ctx.lineWidth=2;if(p.type==='index'){ctx.beginPath();ctx.arc(p.x,p.y,6,0,2*Math.PI);ctx.fill();ctx.stroke();}else{ctx.fillRect(p.x-6,p.y-6,12,12);ctx.strokeRect(p.x-6,p.y-6,12,12);}ctx.fillStyle='#111827';ctx.font='11px sans-serif';ctx.fillText((p.type==='index'?'I':'M')+p.k,p.x+7,p.y-7);});
        const item=data[layer.value],v=item.variance,ids=axes.value.split(',').map(Number),cum6=v.reduce((a,b)=>a+b,0);stats.textContent=`Displayed: ${ids.map(i=>'PC'+(i+1)).join('/')} | variance ${ids.map(i=>(100*v[i]).toFixed(1)+'%').join(', ')} | PC1-6 cumulative ${(100*cum6).toFixed(1)}% | effective dimension ${item.effectiveDimension.toFixed(2)}`;
      }
      [layer,bin,axes].forEach(el=>el.addEventListener('change',draw));reset.addEventListener('click',()=>{yaw=-.65;pitch=.42;draw();});canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);});canvas.addEventListener('pointermove',e=>{if(!dragging)return;yaw+=(e.clientX-lastX)*.01;pitch=Math.max(-1.35,Math.min(1.35,pitch+(e.clientY-lastY)*.01));lastX=e.clientX;lastY=e.clientY;draw();});canvas.addEventListener('pointerup',()=>dragging=false);new ResizeObserver(draw).observe(canvas);draw();
    })();
    </script>
    """
    return template.replace("__DATA__", data_json).replace("__AXIS_OPTIONS__", pca_axis_options())


def mechanism_explorer() -> str:
    """Return a self-contained step-through animation for the evidence-updated mechanisms."""

    return r"""
    <figure class="mechanism-explorer" aria-labelledby="mechanism-explorer-title">
      <div class="mechanism-explorer-head">
        <div>
          <h3 id="mechanism-explorer-title">互动机制图：同一个 prompt，两种证据更新后的工作模型</h3>
          <p>实线步骤已有必要性或局部充分性实验支持；虚线桥接表示当前仍未定位完整 circuit。切换模式并逐步播放，可查看每一步读取什么、写入什么，以及证据边界。</p>
        </div>
        <div class="mechanism-mode-switch" role="group" aria-label="选择计数模式">
          <button type="button" class="mechanism-mode active" data-mechanism-mode="direct" aria-pressed="true">Non-thinking</button>
          <button type="button" class="mechanism-mode" data-mechanism-mode="cot" aria-pressed="false">Thinking trace</button>
        </div>
      </div>

      <div class="mechanism-player">
        <svg id="mechanism-direct-scene" class="mechanism-scene active" viewBox="0 0 1200 420" role="img" aria-label="Non-thinking broad aggregation hypothesis">
          <defs>
            <marker id="direct-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
          </defs>
          <g data-step="0" class="mech-step">
            <rect class="mech-region prompt" x="38" y="42" width="730" height="145" rx="14"/>
            <text class="mech-region-title" x="62" y="72">Prompt body: 256 tokens</text>
            <g class="mech-noise"><circle cx="90" cy="116" r="8"/><circle cx="190" cy="116" r="8"/><circle cx="305" cy="116" r="8"/><circle cx="430" cy="116" r="8"/><circle cx="555" cy="116" r="8"/><circle cx="685" cy="116" r="8"/></g>
            <g class="mech-node needle"><rect x="120" y="92" width="55" height="46" rx="7"/><text x="147" y="121">M₁</text></g>
            <g class="mech-node needle"><rect x="235" y="92" width="55" height="46" rx="7"/><text x="262" y="121">M₂</text></g>
            <g class="mech-node needle"><rect x="360" y="92" width="55" height="46" rx="7"/><text x="387" y="121">M₃</text></g>
            <g class="mech-node needle"><rect x="600" y="92" width="55" height="46" rx="7"/><text x="627" y="121">Mₙ</text></g>
            <text class="mech-ellipsis" x="505" y="121">…</text>
            <g class="mech-node query"><rect x="925" y="72" width="190" height="70" rx="10"/><text x="1020" y="102">final query</text><text class="mech-subtext" x="1020" y="125">&lt;Ans&gt;</text></g>
          </g>
          <g data-step="1" class="mech-step mech-links broad-links">
            <path d="M 940 108 C 760 12, 220 22, 148 92"/><path d="M 950 112 C 790 48, 390 30, 263 92"/><path d="M 960 116 C 830 78, 540 55, 388 92"/><path d="M 970 120 C 880 104, 760 74, 628 92"/>
            <text x="805" y="38">parallel broad retrieval over the needle set</text>
          </g>
          <g data-step="2" class="mech-step">
            <g class="mech-node state"><rect x="505" y="250" width="260" height="86" rx="12"/><text x="635" y="284">cardinality / count state</text><text class="mech-subtext" x="635" y="311">written into the answer-query residual</text></g>
            <g class="mech-links aggregate-links"><path d="M 148 138 C 195 230, 390 245, 520 275"/><path d="M 263 138 C 330 220, 430 240, 535 275"/><path d="M 388 138 C 440 205, 500 225, 565 260"/><path d="M 628 138 C 655 188, 670 215, 675 250"/></g>
          </g>
          <g data-step="3" class="mech-step">
            <g class="mech-node transform"><rect x="810" y="255" width="170" height="76" rx="10"/><text x="895" y="284">later Layers</text><text class="mech-subtext" x="895" y="308">sharpen count logits</text></g>
            <path class="mech-flow-arrow" d="M 765 293 L 805 293"/>
          </g>
          <g data-step="4" class="mech-step">
            <g class="mech-node answer"><rect x="1020" y="258" width="115" height="72" rx="10"/><text x="1077" y="286">output</text><text class="mech-answer-token" x="1077" y="313">&lt;Cₙ&gt;</text></g>
            <path class="mech-flow-arrow" d="M 980 293 L 1015 293"/>
          </g>
        </svg>

        <svg id="mechanism-cot-scene" class="mechanism-scene" viewBox="0 0 1200 420" role="img" aria-label="Thinking trace targeted retrieval and progress-state hypothesis">
          <defs>
            <marker id="cot-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker>
          </defs>
          <g data-step="0" class="mech-step">
            <rect class="mech-region prompt" x="38" y="28" width="730" height="132" rx="14"/>
            <text class="mech-region-title" x="62" y="58">Prompt body</text>
            <g class="mech-node needle"><rect x="120" y="78" width="55" height="44" rx="7"/><text x="147" y="106">M₁</text></g>
            <g class="mech-node needle"><rect x="235" y="78" width="55" height="44" rx="7"/><text x="262" y="106">M₂</text></g>
            <g class="mech-node needle target"><rect x="360" y="78" width="55" height="44" rx="7"/><text x="387" y="106">M₃</text></g>
            <g class="mech-node needle"><rect x="600" y="78" width="55" height="44" rx="7"/><text x="627" y="106">Mₙ</text></g>
            <text class="mech-ellipsis" x="505" y="106">…</text>
            <rect class="mech-region trace" x="38" y="218" width="730" height="132" rx="14"/>
            <text class="mech-region-title" x="62" y="248">Generated thinking trace</text>
            <g class="mech-node trace-token"><rect x="120" y="274" width="52" height="44" rx="7"/><text x="146" y="302">1</text></g>
            <g class="mech-node trace-marker"><rect x="180" y="274" width="58" height="44" rx="7"/><text x="209" y="302">M₁</text></g>
            <g class="mech-node trace-token"><rect x="255" y="274" width="52" height="44" rx="7"/><text x="281" y="302">2</text></g>
            <g class="mech-node trace-marker"><rect x="315" y="274" width="58" height="44" rx="7"/><text x="344" y="302">M₂</text></g>
            <g class="mech-node trace-token target"><rect x="405" y="274" width="52" height="44" rx="7"/><text x="431" y="302">3</text></g>
          </g>
          <g data-step="1" class="mech-step mech-links targeted-link">
            <path d="M 431 274 C 440 220, 430 158, 389 122"/><text x="470" y="190">k-to-k targeted retrieval</text>
          </g>
          <g data-step="2" class="mech-step">
            <g class="mech-node trace-marker active-marker"><rect x="470" y="274" width="58" height="44" rx="7"/><text x="499" y="302">M₃</text></g>
             <g class="mech-node state"><rect x="820" y="82" width="245" height="82" rx="11"/><text x="942" y="113">trace progress state</text><text class="mech-subtext" x="942" y="138">retrieval step k; not final scalar count</text></g>
            <path class="mech-flow-arrow cot" d="M 528 296 C 680 292, 760 190, 835 153"/>
          </g>
          <g data-step="3" class="mech-step">
            <g class="mech-node trace-token successor"><rect x="550" y="274" width="52" height="44" rx="7"/><text x="576" y="302">4</text></g>
            <path class="mech-loop-arrow" d="M 528 318 C 535 365, 600 365, 603 318"/>
            <text x="620" y="372">successor step: emit k+1, then repeat retrieval</text>
          </g>
          <g data-step="4" class="mech-step">
             <g class="mech-node state"><rect x="785" y="265" width="205" height="78" rx="10"/><text x="887" y="294">final scalar-count state</text><text class="mech-subtext" x="887" y="320">causally sufficient; source unresolved</text></g>
             <g class="mech-node answer"><rect x="1030" y="268" width="115" height="72" rx="10"/><text x="1087" y="296">output</text><text class="mech-answer-token" x="1087" y="323">&lt;Cₙ&gt;</text></g>
             <path class="mech-uncertain-arrow" d="M 942 164 C 920 205, 900 230, 892 264"/>
             <path class="mech-flow-arrow cot" d="M 990 304 L 1025 304"/>
             <text x="760" y="222">unresolved compression from trace/context</text>
          </g>
        </svg>
      </div>

      <div class="mechanism-timeline" aria-label="计算步骤">
        <button id="mechanism-prev" type="button" aria-label="上一步">‹</button>
        <div id="mechanism-dots" class="mechanism-dots"></div>
        <button id="mechanism-next" type="button" aria-label="下一步">›</button>
        <button id="mechanism-play" type="button" class="mechanism-play" aria-label="播放动画">▶ 播放</button>
      </div>
      <div id="mechanism-step-copy" class="mechanism-step-copy" aria-live="polite"></div>
      <figcaption>实线表示已有多类因果证据共同支持的主路径，不表示由单一 head 独立完成；灰色虚线表示当前仍缺完整路径分解。Non-thinking 的 count state 是后层才完全可执行的前馈状态。CoT 的 trace progress state 能控制 retrieval/successor，但不等于最终 scalar-count state；从完整 trace/context 到 final state 的压缩 circuit 仍是开放问题。</figcaption>
    </figure>
    <script>
    (() => {
      const copy = {
        direct: [
          {title:'Step 1 · 形成待计数集合', read:'读取：完整 prompt 中 marker/needle token 的身份与位置。', write:'写入：各位置的局部表示；此时还没有外显计数轨迹。', test:'可证伪预测：若模型完全不区分 needle 与 noise，后续 broad score 和 count probe 不应出现。'},
          {title:'Step 2 · 并行 broad retrieval（已有必要性证据）', read:'读取：final <Ans> query 通过若干 Layer-1 heads 同时访问多个 prompt needles。', write:'写入：needle value vectors 的分布式加权组合，而不是可观察的逐 k trace。', test:'已观察：position-local broad-head ablation 显著伤害 final accuracy，且中高 count 早于同层随机对照下降。'},
          {title:'Step 3 · 写入 cardinality state（已有局部充分性证据）', read:'读取：多个 broad-head outputs 与既有 residual。', write:'写入：answer-query residual 中可区分 count 1…30 的分布式状态。', test:'已观察：donor broad slices 能运输部分 count；mask broad group 会同步破坏 count geometry 与输出。'},
          {title:'Step 4 · 后层变换与离散化（结果强，内部算法尚缺）', read:'读取：连续且弯曲的 count-related residual manifold。', write:'写入：30 个 count-token logits 之间的正确 margin。', test:'已观察：Layer-4 residual transplant 一比一搬运 count；尚未像 CoT successor 那样分解 non-thinking MLP features 的具体算术。'},
          {title:'Step 5 · 直接答案读出', read:'读取：<Ans> 位置最后一层 residual。', write:'输出：共享数字 token <Cₙ>，没有中间 trace token。', test:'机制边界：成功输出只证明最终状态足够，不证明内部一定沿一条线性 +1 轴计算。'}
        ],
        cot: [
          {title:'Step 1 · 启动 indexed trace', read:'读取：prompt 与 <Think> 前缀；trace 已生成 1,M₁,2,M₂，当前 query 是数字 3。', write:'写入：当前进度 k=3 的 query state。', test:'可证伪预测：数字 query 的表示应含当前 k/progress 信息，而不仅是绝对位置。'},
          {title:'Step 2 · k-to-k targeted retrieval（强多头因果证据）', read:'读取：query <3> 定位 prompt 中按位置排序的第 3 个 needle。', write:'写入：该 needle 的 marker identity，为下一 token M₃ 提供证据。', test:'已观察：top-2/top-4 clean head-output patch 恢复 Mₖ margin；position-local group ablation 显著伤害 marker accuracy，但单头可替代。'},
          {title:'Step 3 · 写出 marker 并更新 progress', read:'读取：retrieved marker identity 与此前 trace。', write:'写入：M₃ token 及“已完成 3 次检索”的 trace-progress state；该 state 不是最终 scalar count register。', test:'已观察：marker residual transplant 能改变下一 index/close，却不能独立搬运最终 count。'},
          {title:'Step 4 · successor/close 与循环（强局部因果证据）', read:'读取：当前 Mₖ 后 residual/progress 以及 prompt 中是否仍有下一 needle。', write:'写入：下一数字 k+1，或在无下一 needle 时生成 </Think>。', test:'已观察：successor-head bundle 可双向搬运 continue/close evidence；Layer-3 MLP 转换证据，Layer-4 分布式 features 写入具体 token logit。'},
          {title:'Step 5 · final scalar readout（状态充分，上游 circuit 尚缺）', read:'读取：完整 trace/context 压缩后的 final <Ans> residual；具体由哪些 trace、position、prompt 信号构造仍未完全定位。', write:'输出：最终共享数字 token <Cₙ>。', test:'已观察：final residual transplant 一比一搬运 donor count；但固定 trace-readout heads 的 local ablation 高度冗余，head-slice patch 也不足以搬运 scalar。'}
        ]
      };
      let mode='direct', step=0, timer=null;
      const scenes={direct:document.getElementById('mechanism-direct-scene'),cot:document.getElementById('mechanism-cot-scene')};
      const modeButtons=[...document.querySelectorAll('[data-mechanism-mode]')];
      const dots=document.getElementById('mechanism-dots');
      const stepCopy=document.getElementById('mechanism-step-copy');
      const play=document.getElementById('mechanism-play');
      function stop(){if(timer){clearInterval(timer);timer=null;}play.textContent='▶ 播放';play.setAttribute('aria-label','播放动画');}
      function render(){
        Object.entries(scenes).forEach(([name,scene])=>scene.classList.toggle('active',name===mode));
        modeButtons.forEach(button=>{const active=button.dataset.mechanismMode===mode;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
        scenes[mode].querySelectorAll('[data-step]').forEach(group=>{const n=Number(group.dataset.step);group.classList.toggle('visible',n<=step);group.classList.toggle('current',n===step);});
        dots.innerHTML=copy[mode].map((_,i)=>`<button type="button" class="mechanism-dot ${i===step?'active':''}" data-step-index="${i}" aria-label="第 ${i+1} 步"></button>`).join('');
        const item=copy[mode][step];stepCopy.innerHTML=`<div class="mechanism-step-title">${item.title}</div><div class="mechanism-step-grid"><p>${item.read}</p><p>${item.write}</p><p>${item.test}</p></div>`;
      }
      modeButtons.forEach(button=>button.addEventListener('click',()=>{stop();mode=button.dataset.mechanismMode;step=0;render();}));
      dots.addEventListener('click',event=>{const button=event.target.closest('[data-step-index]');if(!button)return;stop();step=Number(button.dataset.stepIndex);render();});
      document.getElementById('mechanism-prev').addEventListener('click',()=>{stop();step=Math.max(0,step-1);render();});
      document.getElementById('mechanism-next').addEventListener('click',()=>{stop();step=Math.min(copy[mode].length-1,step+1);render();});
      play.addEventListener('click',()=>{if(timer){stop();return;}if(step===copy[mode].length-1)step=0;play.textContent='❚❚ 暂停';play.setAttribute('aria-label','暂停动画');render();timer=setInterval(()=>{if(step>=copy[mode].length-1){stop();return;}step+=1;render();},1500);});
      render();
    })();
    </script>
    """


def first_step(frame: pd.DataFrame, mode: str, count_bin: str, metric: str) -> str:
    rows = frame[(frame["mode"] == mode) & (frame["count_bin"] == count_bin) & (frame[metric] >= 0.99)]
    return str(int(rows.step.min())) if len(rows) else "未达到"


def first_step_overall(eval_counts: pd.DataFrame, mode: str, metric: str) -> str:
    overall = _balanced_accuracy_over_counts(eval_counts)
    rows = overall[(overall["mode"] == mode) & (overall[metric] >= 0.99)]
    return str(int(rows.step.min())) if len(rows) else "未达到"


def select_row(frame: pd.DataFrame, **conditions) -> pd.Series:
    result = frame
    for name, value in conditions.items():
        result = result[result[name] == value]
    if result.empty:
        raise KeyError(conditions)
    return result.iloc[0]


def build_report(run_dir: Path) -> Path:
    setup_plot_style()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    stratified = run_dir / "analysis" / "report_stratified"
    tables = stratified / "tables"
    figures = stratified / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((stratified / "manifest.json").read_text(encoding="utf-8"))

    eval_bins = pd.read_csv(run_dir / "tables" / "eval_dynamics_by_bin.csv")
    eval_counts = pd.read_csv(run_dir / "tables" / "eval_dynamics_by_count.csv")
    eval_losses = pd.read_csv(run_dir / "tables" / "eval_dynamics_losses.csv")
    attention_dir = run_dir / "analysis" / "attention_causal" / "tables"
    attention_rows = pd.read_csv(attention_dir / "attention_rows.csv")
    single = pd.read_csv(tables / "single_head_ablation_by_bin.csv")
    cumulative = pd.read_csv(tables / "cumulative_head_ablation_by_bin.csv")
    position_local = pd.read_csv(tables / "position_local_ablation_by_bin.csv")
    retrieval = pd.read_csv(tables / "retrieval_control_patching_by_bin.csv")
    nested = pd.read_csv(tables / "nested_head_patching_regression_by_bin.csv")
    successor_tables = run_dir / "analysis" / "successor_patching" / "tables"
    successor_summary_path = successor_tables / "successor_head_patching_summary.csv"
    if not successor_summary_path.exists():
        raise FileNotFoundError(
            "Missing successor patching summary. Run scripts/run_v10_successor_patching.py "
            f"for {run_dir} before rebuilding the report."
        )
    successor_summary = pd.read_csv(successor_summary_path)
    successor_conversion_tables = run_dir / "analysis" / "successor_conversion" / "tables"
    successor_stage_path = successor_conversion_tables / "successor_stage_logit_lens_summary.csv"
    successor_conversion_path = successor_conversion_tables / "successor_sublayer_patching_summary.csv"
    successor_conversion_manifest_path = run_dir / "analysis" / "successor_conversion" / "manifest.json"
    if not successor_stage_path.exists() or not successor_conversion_path.exists():
        raise FileNotFoundError(
            "Missing successor sublayer conversion outputs. Run "
            "scripts/run_v10_successor_conversion.py for "
            f"{run_dir} before rebuilding the report."
        )
    successor_stage = pd.read_csv(successor_stage_path)
    successor_conversion = pd.read_csv(successor_conversion_path)
    successor_conversion_manifest = json.loads(
        successor_conversion_manifest_path.read_text(encoding="utf-8")
    )
    successor_mlp_root = run_dir / "analysis" / "successor_mlp_features"
    successor_mlp_tables = successor_mlp_root / "tables"
    mlp_feature_summary_path = successor_mlp_tables / "mlp_feature_patching_summary.csv"
    mlp_feature_concentration_path = successor_mlp_tables / "mlp_feature_concentration.csv"
    mlp_feature_manifest_path = successor_mlp_root / "manifest.json"
    if not all(
        path.exists()
        for path in (
            mlp_feature_summary_path,
            mlp_feature_concentration_path,
            mlp_feature_manifest_path,
        )
    ):
        raise FileNotFoundError(
            "Missing successor MLP-feature outputs. Run "
            "scripts/run_v10_successor_mlp_features.py for "
            f"{run_dir} before rebuilding the report."
        )
    mlp_feature_summary = pd.read_csv(mlp_feature_summary_path)
    mlp_feature_concentration = pd.read_csv(mlp_feature_concentration_path)
    mlp_feature_manifest = json.loads(mlp_feature_manifest_path.read_text(encoding="utf-8"))
    steering = pd.read_csv(tables / "geometry_steering_by_bin.csv")
    steering_gain = pd.read_csv(tables / "geometry_steering_gain_by_bin.csv")
    geometry_path_root = run_dir / "analysis" / "geometry_path_steering"
    geometry_path_tables = geometry_path_root / "tables"
    geometry_path_detail_path = geometry_path_tables / "geometry_path_steering.csv"
    geometry_path_manifest_path = geometry_path_root / "manifest.json"
    if not geometry_path_detail_path.exists() or not geometry_path_manifest_path.exists():
        raise FileNotFoundError(
            "Missing centroid-path steering outputs. Run "
            "scripts/run_v10_geometry_path_steering.py for "
            f"{run_dir} before rebuilding the report."
        )
    geometry_path_detail = pd.read_csv(geometry_path_detail_path)
    geometry_path_manifest = json.loads(geometry_path_manifest_path.read_text(encoding="utf-8"))
    geometry_path_regression = aggregate_geometry_path_regression(geometry_path_detail)
    geometry_path_regression.to_csv(
        geometry_path_tables / "geometry_path_steering_regression_report.csv",
        index=False,
    )
    centroid_reg = pd.read_csv(tables / "centroid_transplant_regression_by_bin.csv")
    raw_reg = pd.read_csv(tables / "final_state_transplant_regression_by_bin.csv")
    coordinates = pd.read_csv(tables / "centroid_mean_pca_coordinates.csv")
    geometry = pd.read_csv(tables / "centroid_mean_geometry.csv")
    absolute_spread = centroid_absolute_spread(run_dir)
    absolute_spread.to_csv(tables / "centroid_absolute_spread.csv", index=False)
    trace_coordinates, trace_geometry = trace_token_centroid_pca(run_dir)
    trace_coordinates.to_csv(tables / "trace_token_mean_pca_coordinates.csv", index=False)
    trace_geometry.to_csv(tables / "trace_token_mean_geometry.csv", index=False)
    trace_progress = pd.read_csv(tables / "trace_progress_transplant_by_bin.csv")
    hidden_patch_root = run_dir / "analysis" / "hidden_state_patching"
    hidden_patch_tables = hidden_patch_root / "tables"
    hidden_patch_figures = hidden_patch_root / "figures"
    hidden_patch_manifest_path = hidden_patch_root / "manifest.json"
    hidden_patch_paths = {
        "final": hidden_patch_tables / "final_answer_patching_summary.csv",
        "final_directional": hidden_patch_tables
        / "final_answer_patching_directional_summary.csv",
        "trace_final": hidden_patch_tables / "trace_final_patching_summary.csv",
        "trace_final_directional": hidden_patch_tables
        / "trace_final_patching_directional_summary.csv",
        "early": hidden_patch_tables / "trace_early_stop_patching_summary.csv",
        "rollout": hidden_patch_tables / "misaligned_trace_rollout_summary.csv",
        "rollout_factors": hidden_patch_tables
        / "misaligned_trace_rollout_factor_summary.csv",
    }
    if not hidden_patch_manifest_path.exists() or not all(
        path.exists() for path in hidden_patch_paths.values()
    ):
        raise FileNotFoundError(
            "Missing hidden-state patching outputs. Run "
            "scripts/run_v10_hidden_state_patching.py for "
            f"{run_dir} before rebuilding the report."
        )
    hidden_patch_manifest = json.loads(hidden_patch_manifest_path.read_text(encoding="utf-8"))
    hidden_final_summary = pd.read_csv(hidden_patch_paths["final"])
    hidden_final_directional = pd.read_csv(hidden_patch_paths["final_directional"])
    hidden_trace_final_summary = pd.read_csv(hidden_patch_paths["trace_final"])
    hidden_trace_final_directional = pd.read_csv(
        hidden_patch_paths["trace_final_directional"]
    )
    hidden_early_summary = pd.read_csv(hidden_patch_paths["early"])
    hidden_rollout_summary = pd.read_csv(hidden_patch_paths["rollout"])
    hidden_rollout_factors = pd.read_csv(hidden_patch_paths["rollout_factors"])

    final_bridge_root = run_dir / "analysis" / "final_bridge_causal"
    final_bridge_tables = final_bridge_root / "tables"
    final_bridge_figures = final_bridge_root / "figures"
    final_bridge_manifest_path = final_bridge_root / "manifest.json"
    final_bridge_paths = {
        "conflict": final_bridge_tables / "cot_conflict_summary.csv",
        "bridge": final_bridge_tables / "cot_final_bridge_patching_summary.csv",
        "nonthinking_mlp": final_bridge_tables
        / "nonthinking_mlp_transport_summary.csv",
    }
    if not final_bridge_manifest_path.exists() or not all(
        path.exists() for path in final_bridge_paths.values()
    ):
        raise FileNotFoundError(
            "Missing final scalar-bridge causal outputs. Run "
            "scripts/run_v10_final_bridge_causal.py for "
            f"{run_dir} before rebuilding the report."
        )
    final_bridge_manifest = json.loads(
        final_bridge_manifest_path.read_text(encoding="utf-8")
    )
    cot_conflict_summary = pd.read_csv(final_bridge_paths["conflict"])
    cot_bridge_summary = pd.read_csv(final_bridge_paths["bridge"])
    nonthinking_mlp_transport = pd.read_csv(final_bridge_paths["nonthinking_mlp"])

    generated = {
        "training_overall": figures / "training_overall_accuracy_and_loss.png",
        "training_bins": figures / "training_accuracy_by_count_bin.png",
        "broad_attention": figures / "broad_attention_by_model_and_count_bin.png",
        "targeted_attention": figures / "targeted_attention_by_count_bin.png",
        "attention": figures / "broad_attention_by_model_and_count_bin.png",
        "single_nonthinking": figures / "single_head_nonthinking_final_by_count_bin.png",
        "single_targeted": figures / "single_head_cot_targeted_trace_by_count_bin.png",
        "single_cot_final": figures / "single_head_cot_final_by_count_bin.png",
        "alignment_nonthinking": figures / "alignment_nonthinking_broad_vs_final_drop.png",
        "alignment_targeted": figures / "alignment_cot_targeted_vs_trace_drop.png",
        "alignment_readout": figures / "alignment_cot_readout_vs_final_drop.png",
        "cumulative_nonthinking": figures / "cumulative_nonthinking_broad_to_final.png",
        "cumulative_targeted_trace": figures / "cumulative_cot_targeted_to_trace.png",
        "cumulative_targeted_final": figures / "cumulative_cot_targeted_to_final.png",
        "cumulative_readout_final": figures / "cumulative_cot_readout_to_final.png",
        "local_nonthinking": figures / "position_local_nonthinking_broad_to_final.png",
        "local_targeted": figures / "position_local_cot_targeted_to_trace.png",
        "local_readout": figures / "position_local_cot_readout_to_final.png",
        "retrieval_interior": figures / "retrieval_patching_interior_by_count_bin.png",
        "retrieval_final": figures / "retrieval_patching_final_by_count_bin.png",
        "successor_continue_recovery": figures / "successor_continue_recovery_by_count_bin.png",
        "successor_continue_accuracy": figures / "successor_continue_accuracy_by_count_bin.png",
        "successor_close_recovery": figures / "successor_close_recovery_by_count_bin.png",
        "successor_close_accuracy": figures / "successor_close_accuracy_by_count_bin.png",
        "successor_logit_lens": figures / "successor_residual_logit_lens.png",
        "successor_component_evidence": figures / "successor_component_evidence.png",
        "successor_conversion_continue": figures / "successor_conversion_continue.png",
        "successor_conversion_close": figures / "successor_conversion_close.png",
        "mlp_feature_concentration": figures / "mlp_feature_concentration.png",
        "mlp_feature_continue": figures / "mlp_feature_continue_recovery.png",
        "mlp_feature_close": figures / "mlp_feature_close_recovery.png",
        "nested_nonthinking": figures / "nested_head_patching_nonthinking_by_count_bin.png",
        "nested_thinking": figures / "nested_head_patching_thinking_by_count_bin.png",
        "steering": figures / "geometry_steering_by_count_bin.png",
        "geometry_path_adjacent": figures / "geometry_path_adjacent_transport.png",
        "geometry_path_chord": figures / "geometry_path_chord_transport.png",
        "geometry_path_curve": figures / "geometry_path_curve_transport.png",
        "geometry_path_tracking": figures / "geometry_path_curve_vs_chord_tracking.png",
        "transplant": figures / "residual_transplant_by_count_bin.png",
        "pca_variance": figures / "pca_variance_by_site_layer.png",
        "pca_static": figures / "pca_count_mean_static.png",
        "hidden_final_answer": hidden_patch_figures / "final_answer_transport_by_layer.png",
        "hidden_trace_final": hidden_patch_figures / "trace_final_transport_by_layer.png",
        "hidden_early_close": hidden_patch_figures / "trace_early_stop_close_margin.png",
        "hidden_early_transport": hidden_patch_figures / "trace_early_stop_count_transport.png",
        "hidden_upward_transport": hidden_patch_figures
        / "donor_gt_receiver_scalar_transport.png",
        "hidden_rollout_first": hidden_patch_figures
        / "misaligned_rollout_first_index.png",
        "hidden_rollout_stop": hidden_patch_figures
        / "misaligned_rollout_stop_index.png",
        "hidden_rollout_final": hidden_patch_figures
        / "misaligned_rollout_final_count.png",
        "hidden_rollout_factors": hidden_patch_figures
        / "misaligned_rollout_factor_coefficients.png",
        "cot_conflict_source": final_bridge_figures
        / "cot_conflict_source_attribution.png",
        "cot_final_bridge": final_bridge_figures
        / "cot_final_bridge_component_recovery.png",
        "nonthinking_mlp_transport": final_bridge_figures
        / "nonthinking_mlp_feature_transport.png",
    }
    # The legacy inline report body is replaced below after f-string evaluation.
    # Keep these aliases until that body is removed entirely.
    generated["retrieval"] = generated["retrieval_interior"]
    generated["nested"] = generated["nested_nonthinking"]
    save_training_overall(eval_counts, eval_losses, generated["training_overall"])
    save_training_by_bin(eval_bins, generated["training_bins"])
    save_broad_attention(attention_rows, generated["broad_attention"])
    save_targeted_attention(attention_rows, generated["targeted_attention"])
    save_single_ablation_panel(
        single,
        mode="nonthinking",
        metric="drop_final_count_accuracy",
        title="Non-thinking single-head ablation: final-count accuracy drop",
        path=generated["single_nonthinking"],
    )
    save_single_ablation_panel(
        single,
        mode="thinking",
        metric="drop_trace_marker_accuracy",
        title="CoT single-head ablation: trace-marker accuracy drop",
        path=generated["single_targeted"],
    )
    save_single_ablation_panel(
        single,
        mode="thinking",
        metric="drop_final_count_accuracy",
        title="CoT single-head ablation: final-count accuracy drop",
        path=generated["single_cot_final"],
    )
    alignment = save_ablation_score_alignment(
        attention_rows,
        single,
        {
            "nonthinking": generated["alignment_nonthinking"],
            "targeted": generated["alignment_targeted"],
            "readout": generated["alignment_readout"],
        },
    )
    alignment.to_csv(tables / "attention_score_ablation_alignment.csv", index=False)
    save_cumulative_ablation_panel(
        cumulative,
        mode="nonthinking",
        metric="final_count_accuracy",
        family="direct_broad_top",
        title="Non-thinking broad-head ablation: final-count accuracy",
        path=generated["cumulative_nonthinking"],
    )
    save_cumulative_ablation_panel(
        cumulative,
        mode="thinking",
        metric="trace_marker_accuracy",
        family="targeted_retrieval_top",
        title="CoT targeted-head ablation: trace-marker accuracy",
        path=generated["cumulative_targeted_trace"],
    )
    save_cumulative_ablation_panel(
        cumulative,
        mode="thinking",
        metric="final_count_accuracy",
        family="targeted_retrieval_top",
        title="CoT targeted-head ablation: final-count accuracy",
        path=generated["cumulative_targeted_final"],
    )
    save_cumulative_ablation_panel(
        cumulative,
        mode="thinking",
        metric="final_count_accuracy",
        family="trace_readout_top",
        title="CoT trace-readout-head ablation: final-count accuracy",
        path=generated["cumulative_readout_final"],
    )
    save_position_local_ablation_panel(
        position_local,
        mechanism="direct_broad",
        metric="final_count_accuracy",
        title="Position-local ablation: non-thinking broad heads at <Ans> only",
        path=generated["local_nonthinking"],
    )
    save_position_local_ablation_panel(
        position_local,
        mechanism="targeted_retrieval",
        metric="trace_marker_accuracy",
        title="Position-local ablation: CoT targeted heads at trace-number queries only",
        path=generated["local_targeted"],
    )
    save_position_local_ablation_panel(
        position_local,
        mechanism="trace_readout",
        metric="final_count_accuracy",
        title="Position-local ablation: CoT trace-readout heads at <Ans> only",
        path=generated["local_readout"],
    )
    save_retrieval_patch_panel(
        retrieval,
        query_role="interior",
        title="CoT local head patching at an interior <k>: marker-identity recovery",
        path=generated["retrieval_interior"],
    )
    save_retrieval_patch_panel(
        retrieval,
        query_role="final",
        title="CoT local head patching at final <n>: marker-identity recovery",
        path=generated["retrieval_final"],
    )
    save_successor_patch_panel(
        successor_summary,
        direction="continue_into_close",
        metric="normalized_recovery",
        title="CoT successor patch at M_k: continue evidence into a close receiver",
        path=generated["successor_continue_recovery"],
    )
    save_successor_patch_panel(
        successor_summary,
        direction="continue_into_close",
        metric="patched_target_correct",
        title="CoT successor patch at M_k: does the close receiver emit <k+1>?",
        path=generated["successor_continue_accuracy"],
    )
    save_successor_patch_panel(
        successor_summary,
        direction="close_into_continue",
        metric="normalized_recovery",
        title="CoT stop patch at M_k: close evidence into a continue receiver",
        path=generated["successor_close_recovery"],
    )
    save_successor_patch_panel(
        successor_summary,
        direction="close_into_continue",
        metric="patched_target_correct",
        title="CoT stop patch at M_k: does the continue receiver emit </Think>?",
        path=generated["successor_close_accuracy"],
    )
    save_successor_residual_logit_lens(
        successor_stage,
        generated["successor_logit_lens"],
    )
    save_successor_component_evidence(
        successor_stage,
        generated["successor_component_evidence"],
    )
    save_successor_sublayer_conversion(
        successor_conversion,
        direction="continue_into_close",
        title="Sublayer conversion: continue evidence into a close receiver",
        path=generated["successor_conversion_continue"],
    )
    save_successor_sublayer_conversion(
        successor_conversion,
        direction="close_into_continue",
        title="Sublayer conversion: close evidence into a continue receiver",
        path=generated["successor_conversion_close"],
    )
    save_mlp_feature_concentration(
        mlp_feature_concentration,
        generated["mlp_feature_concentration"],
    )
    save_mlp_feature_patch_panel(
        mlp_feature_summary,
        direction="continue_into_close",
        title="MLP feature patching: continue evidence into a close receiver",
        path=generated["mlp_feature_continue"],
    )
    save_mlp_feature_patch_panel(
        mlp_feature_summary,
        direction="close_into_continue",
        title="MLP feature patching: close evidence into a continue receiver",
        path=generated["mlp_feature_close"],
    )
    save_nested_patch_panel(
        nested,
        mode="nonthinking",
        title="Non-thinking local patching at <Ans>: donor-count transport",
        path=generated["nested_nonthinking"],
    )
    save_nested_patch_panel(
        nested,
        mode="thinking",
        title="CoT local patching at <Ans>: trace-readout head slices",
        path=generated["nested_thinking"],
    )
    save_steering(steering, generated["steering"])
    save_adjacent_geometry_transport(
        geometry_path_regression,
        generated["geometry_path_adjacent"],
    )
    save_nonadjacent_path_transport(
        geometry_path_regression,
        method="nonadjacent_chord_transport",
        title="Non-adjacent straight-chord transport across count centroids",
        path=generated["geometry_path_chord"],
    )
    save_nonadjacent_path_transport(
        geometry_path_regression,
        method="nonadjacent_curve_transport",
        title="Non-adjacent transport along the piecewise centroid curve",
        path=generated["geometry_path_curve"],
    )
    save_curve_chord_tracking(
        geometry_path_detail,
        generated["geometry_path_tracking"],
    )
    save_transplants(centroid_reg, raw_reg, generated["transplant"])
    save_pca_variance(geometry, generated["pca_variance"])
    save_pca_static(coordinates, generated["pca_static"])

    milestone_rows = []
    for mode in ("nonthinking", "thinking"):
        milestone_rows.append(
            {
                "mode": mode,
                "bin": "1-30（30 个 exact counts 平衡平均）",
                "tf": first_step_overall(eval_counts, mode, "tf_accuracy"),
                "ar": first_step_overall(eval_counts, mode, "ar_accuracy"),
                "trace": (
                    first_step_overall(eval_counts, mode, "trace_exact")
                    if mode == "thinking"
                    else "不适用"
                ),
            }
        )
        for count_bin in COUNT_BINS:
            milestone_rows.append(
                {
                    "mode": mode,
                    "bin": count_bin,
                    "tf": first_step(eval_bins, mode, count_bin, "tf_accuracy"),
                    "ar": first_step(eval_bins, mode, count_bin, "ar_accuracy"),
                    "trace": (
                        first_step(eval_bins, mode, count_bin, "trace_exact")
                        if mode == "thinking"
                        else "不适用"
                    ),
                }
            )

    overall_steps = {
        "direct_tf": first_step_overall(eval_counts, "nonthinking", "tf_accuracy"),
        "direct_ar": first_step_overall(eval_counts, "nonthinking", "ar_accuracy"),
        "thinking_tf": first_step_overall(eval_counts, "thinking", "tf_accuracy"),
        "thinking_ar": first_step_overall(eval_counts, "thinking", "ar_accuracy"),
        "thinking_trace": first_step_overall(eval_counts, "thinking", "trace_exact"),
    }

    def loss_at(step: int, mode: str, component: str) -> str:
        rows = eval_losses[
            (eval_losses["step"] == step)
            & (eval_losses["mode"] == mode)
            & (eval_losses["component"] == component)
        ]
        return fmt(rows.loss.iloc[0], 4) if len(rows) else "n/a"

    attention_binned = _attention_rows_with_bins(attention_rows)
    final_attention = (
        attention_binned[attention_binned.query_kind == "final_count_query"]
        .groupby(["mode", "count_bin", "layer", "head"], as_index=False)
        .agg(
            broad_attention_score=("broad_attention_score", "mean"),
            prompt_needles_mass=("prompt_needles_mass", "mean"),
            needle_entropy_normalized=("needle_entropy_normalized", "mean"),
            needle_effective_number=("needle_effective_number", "mean"),
        )
    )
    final_attention["single_target_concentration"] = final_attention.prompt_needles_mass * (
        1.0 - final_attention.needle_entropy_normalized
    )
    broad_head_rows: list[dict[str, object]] = []
    nonthinking_concentration_rows: list[dict[str, object]] = []
    for mode in ("nonthinking", "thinking"):
        for count_bin in COUNT_BINS:
            frame = final_attention[
                (final_attention["mode"] == mode) & (final_attention["count_bin"] == count_bin)
            ]
            best = frame.sort_values("broad_attention_score", ascending=False).iloc[0]
            broad_head_rows.append(
                {
                    "mode": mode,
                    "bin": count_bin,
                    "head": code(f"L{int(best.layer)+1}H{int(best['head'])}"),
                    "score": fmt(best.broad_attention_score),
                    "mass": fmt(best.prompt_needles_mass),
                    "entropy": fmt(best.needle_entropy_normalized),
                    "effective": fmt(best.needle_effective_number, 1),
                }
            )
            if mode == "nonthinking":
                concentrated = frame.sort_values("single_target_concentration", ascending=False).iloc[0]
                nonthinking_concentration_rows.append(
                    {
                        "bin": count_bin,
                        "head": code(f"L{int(concentrated.layer)+1}H{int(concentrated['head'])}"),
                        "score": fmt(concentrated.single_target_concentration),
                        "mass": fmt(concentrated.prompt_needles_mass),
                        "entropy": fmt(concentrated.needle_entropy_normalized),
                        "effective": fmt(concentrated.needle_effective_number, 1),
                    }
                )

    targeted_attention = (
        attention_binned[
            (attention_binned["mode"] == "thinking")
            & (attention_binned["query_kind"] == "targeted_retrieval_query")
        ]
        .groupby(["count_bin", "layer", "head"], as_index=False)
        .agg(
            correct_prompt_needle_mass=("correct_prompt_needle_mass", "mean"),
            diagonal_dominance=("diagonal_dominance", "mean"),
            correct_top1=("correct_top1", "mean"),
            prompt_needles_mass=("prompt_needles_mass", "mean"),
        )
    )
    targeted_head_rows: list[dict[str, object]] = []
    for count_bin in COUNT_BINS:
        frame = targeted_attention[targeted_attention.count_bin == count_bin]
        best = frame.sort_values("correct_prompt_needle_mass", ascending=False).iloc[0]
        targeted_head_rows.append(
            {
                "bin": count_bin,
                "head": code(f"L{int(best.layer)+1}H{int(best['head'])}"),
                "raw": fmt(best.correct_prompt_needle_mass),
                "diag": fmt(best.diagonal_dominance),
                "top1": pct(best.correct_top1),
                "needle_mass": fmt(best.prompt_needles_mass),
            }
        )

    def remaining_accuracy(
        *,
        mode: str,
        family: str,
        count_bin: str,
        top_n: int,
        metric: str,
    ) -> str:
        frame = cumulative[
            (cumulative["mode"] == mode)
            & (cumulative["family"] == family)
            & (cumulative["count_bin"] == count_bin)
            & (cumulative["top_n"] == top_n)
        ]
        return "n/a" if frame.empty else pct(frame[metric].mean())

    def local_remaining_accuracy(
        *,
        mechanism: str,
        family: str,
        count_bin: str,
        top_n: int,
        metric: str,
    ) -> str:
        frame = position_local[
            (position_local["mechanism"] == mechanism)
            & (position_local["family"] == family)
            & (position_local["count_bin"] == count_bin)
            & (position_local["top_n"] == top_n)
        ]
        if frame.empty:
            return "n/a"
        return pct(frame[metric].mean())

    ablation_rows = []
    cumulative_ablation_rows = []
    for count_bin in COUNT_BINS:
        direct = single[(single["mode"] == "nonthinking") & (single["count_bin"] == count_bin)].sort_values("drop_final_count_accuracy", ascending=False).iloc[0]
        trace = single[(single["mode"] == "thinking") & (single["count_bin"] == count_bin)].sort_values("drop_trace_marker_accuracy", ascending=False).iloc[0]
        cot_final = single[(single["mode"] == "thinking") & (single["count_bin"] == count_bin)].sort_values("drop_final_count_accuracy", ascending=False).iloc[0]
        ablation_rows.append(
            {
                "bin": count_bin,
                "direct_head": code(f"L{int(direct.layer)+1}H{int(direct['head'])}"),
                "direct_drop": pct(direct.drop_final_count_accuracy),
                "trace_head": code(f"L{int(trace.layer)+1}H{int(trace['head'])}"),
                "trace_drop": pct(trace.drop_trace_marker_accuracy),
                "cot_final_head": code(f"L{int(cot_final.layer)+1}H{int(cot_final['head'])}"),
                "cot_final_drop": pct(cot_final.drop_final_count_accuracy),
            }
        )
        cumulative_ablation_rows.append(
            {
                "bin": count_bin,
                "direct_top1": remaining_accuracy(
                    mode="nonthinking", family="direct_broad_top", count_bin=count_bin, top_n=1,
                    metric="final_count_accuracy",
                ),
                "direct_top2": remaining_accuracy(
                    mode="nonthinking", family="direct_broad_top", count_bin=count_bin, top_n=2,
                    metric="final_count_accuracy",
                ),
                "direct_random4": remaining_accuracy(
                    mode="nonthinking", family="random", count_bin=count_bin, top_n=4,
                    metric="final_count_accuracy",
                ),
                "target_trace_top1": remaining_accuracy(
                    mode="thinking", family="targeted_retrieval_top", count_bin=count_bin, top_n=1,
                    metric="trace_marker_accuracy",
                ),
                "target_trace_top4": remaining_accuracy(
                    mode="thinking", family="targeted_retrieval_top", count_bin=count_bin, top_n=4,
                    metric="trace_marker_accuracy",
                ),
                "target_trace_top8": remaining_accuracy(
                    mode="thinking", family="targeted_retrieval_top", count_bin=count_bin, top_n=8,
                    metric="trace_marker_accuracy",
                ),
                "target_final_top8": remaining_accuracy(
                    mode="thinking", family="targeted_retrieval_top", count_bin=count_bin, top_n=8,
                    metric="final_count_accuracy",
                ),
                "target_trace_random4": remaining_accuracy(
                    mode="thinking", family="random", count_bin=count_bin, top_n=4,
                    metric="trace_marker_accuracy",
                ),
                "target_final_random4": remaining_accuracy(
                    mode="thinking", family="random", count_bin=count_bin, top_n=4,
                    metric="final_count_accuracy",
                ),
                "readout_final_top4": remaining_accuracy(
                    mode="thinking", family="trace_readout_top", count_bin=count_bin, top_n=4,
                    metric="final_count_accuracy",
                ),
                "readout_final_top8": remaining_accuracy(
                    mode="thinking", family="trace_readout_top", count_bin=count_bin, top_n=8,
                    metric="final_count_accuracy",
                ),
                "readout_final_random4": remaining_accuracy(
                    mode="thinking", family="random", count_bin=count_bin, top_n=4,
                    metric="final_count_accuracy",
                ),
            }
        )

    position_local_rows: list[dict[str, str | int]] = []
    local_specs = (
        ("direct_broad", "final_count_accuracy", "Non-thinking broad @ <Ans>"),
        ("targeted_retrieval", "trace_marker_accuracy", "CoT targeted @ <k>"),
        ("trace_readout", "final_count_accuracy", "CoT readout @ <Ans>"),
    )
    for mechanism, metric, label in local_specs:
        for count_bin in COUNT_BINS:
            for top_n in (1, 4, 8):
                position_local_rows.append(
                    {
                        "mechanism": label,
                        "bin": count_bin,
                        "top_n": top_n,
                        "ranked_top": local_remaining_accuracy(
                            mechanism=mechanism,
                            family="ranked_top",
                            count_bin=count_bin,
                            top_n=top_n,
                            metric=metric,
                        ),
                        "same_layer_random": local_remaining_accuracy(
                            mechanism=mechanism,
                            family="layer_matched_random",
                            count_bin=count_bin,
                            top_n=top_n,
                            metric=metric,
                        ),
                        "random4_reference": local_remaining_accuracy(
                            mechanism=mechanism,
                            family="layer_matched_random",
                            count_bin=count_bin,
                            top_n=4,
                            metric=metric,
                        ),
                    }
                )

    alignment_rows = [
        {
            "mechanism": row.mechanism,
            "bin": row.count_bin,
            "rho": fmt(row.spearman_rho),
            "n": int(row.n_heads),
        }
        for row in alignment.itertuples(index=False)
    ]
    ablation_lookup = {row["bin"]: row for row in ablation_rows}
    cumulative_ablation_lookup = {row["bin"]: row for row in cumulative_ablation_rows}
    position_local_direct_rows = [
        row for row in position_local_rows
        if row["mechanism"] == "Non-thinking broad @ <Ans>"
    ]
    position_local_targeted_rows = [
        row for row in position_local_rows
        if row["mechanism"] == "CoT targeted @ <k>"
    ]
    position_local_readout_rows = [
        row for row in position_local_rows
        if row["mechanism"] == "CoT readout @ <Ans>"
    ]
    alignment_lookup = {
        (row.mode, row.count_bin, row.descriptive_score): fmt(row.spearman_rho)
        for row in alignment.itertuples(index=False)
    }

    retrieval_patch_rows = []
    for query_role in ("interior", "final"):
        for count_bin in COUNT_BINS:
            frame = retrieval[(retrieval.query_role == query_role) & (retrieval.count_bin == count_bin)]
            for top_n in (1, 2, 4, 8):
                ranked = frame[(frame.family == "targeted_top") & (frame.top_n == top_n)].normalized_recovery.mean()
                random_values = frame[(frame.family == "random") & (frame.top_n == top_n)].normalized_recovery
                retrieval_patch_rows.append(
                    {
                        "role": "interior k" if query_role == "interior" else "final k=n",
                        "bin": count_bin,
                        "heads": top_n,
                        "ranked": fmt(ranked),
                        "random": fmt(random_values.mean()),
                        "range": f"{fmt(random_values.min())} to {fmt(random_values.max())}",
                    }
                )

    count_transport_rows = []
    for mode, label in (("nonthinking", "non-thinking broad"), ("thinking", "CoT trace readout")):
        for count_bin in COUNT_BINS:
            frame = nested[(nested["mode"] == mode) & (nested.count_bin == count_bin)]
            for top_n in (1, 2, 4, 8):
                ranked = frame[(frame.family == "primary_top") & (frame.top_n == top_n)].slope.mean()
                random_values = frame[(frame.family == "random") & (frame.top_n == top_n)].slope
                count_transport_rows.append(
                    {
                        "mechanism": label,
                        "bin": count_bin,
                        "heads": top_n,
                        "slope": fmt(ranked),
                        "random": fmt(random_values.mean()),
                        "range": f"{fmt(random_values.min())} to {fmt(random_values.max())}",
                    }
                )
    retrieval_patch_lookup = {
        (row["role"], row["bin"], row["heads"]): row for row in retrieval_patch_rows
    }

    successor_patch_rows: list[dict[str, object]] = []
    successor_patch_lookup: dict[tuple[str, str, int, str], dict[str, object]] = {}
    successor_direction_labels = {
        "continue_into_close": "continue → close receiver",
        "close_into_continue": "close → continue receiver",
    }
    successor_family_labels = {
        "successor_top": "successor-score ranked",
        "targeted_top": "k-to-k targeted ranked",
        "successor_wrong_row": "successor heads / wrong donor row",
    }
    for direction, direction_label in successor_direction_labels.items():
        for count_bin in COUNT_BINS:
            frame = successor_summary[
                (successor_summary["direction"] == direction)
                & (successor_summary["count_bin"] == count_bin)
            ]
            for top_n in (1, 2, 4, 8):
                random = frame[
                    (frame["family"] == "random") & (frame["top_n"] == top_n)
                ]
                random_recovery = random["normalized_recovery"]
                random_accuracy = random["patched_target_correct"]
                for family, family_label in successor_family_labels.items():
                    selected = frame[
                        (frame["family"] == family) & (frame["top_n"] == top_n)
                    ]
                    if selected.empty:
                        continue
                    row = {
                        "direction": direction_label,
                        "bin": count_bin,
                        "heads": top_n,
                        "family": family_label,
                        "recovery": fmt(selected["normalized_recovery"].mean()),
                        "accuracy": pct(selected["patched_target_correct"].mean()),
                        "random_recovery": fmt(random_recovery.mean()),
                        "random_recovery_range": (
                            f"{fmt(random_recovery.min())} to {fmt(random_recovery.max())}"
                        ),
                        "random_accuracy": pct(random_accuracy.mean()),
                    }
                    successor_patch_rows.append(row)
                    successor_patch_lookup[(direction, count_bin, top_n, family)] = row

    def successor_value(
        direction: str,
        count_bin: str,
        top_n: int,
        family: str,
        field: str,
    ) -> str:
        return str(successor_patch_lookup[(direction, count_bin, top_n, family)][field])

    conversion_labels = {
        "continue_into_close": "continue evidence → close receiver",
        "close_into_continue": "close evidence → continue receiver",
    }
    successor_conversion_rows = []
    for direction, direction_label in conversion_labels.items():
        for count_bin in COUNT_BINS:
            for layer in (2, 3):
                frame = successor_conversion[
                    (successor_conversion["direction"] == direction)
                    & (successor_conversion["count_bin"] == count_bin)
                    & (successor_conversion["layer"] == layer)
                ].set_index("intervention")
                direct = float(frame.loc["attn_direct_residual", "normalized_recovery"])
                native = float(frame.loc["attn_native_mlp", "normalized_recovery"])
                mlp_only = float(frame.loc["mlp_out_only", "normalized_recovery"])
                full = float(frame.loc["post_mlp_state", "normalized_recovery"])
                successor_conversion_rows.append(
                    {
                        "direction": direction_label,
                        "bin": count_bin,
                        "layer": f"Layer {layer + 1}",
                        "attn_direct": fmt(direct),
                        "native_mlp": fmt(native),
                        "mlp_mediation": fmt(native - direct),
                        "mlp_only": fmt(mlp_only),
                        "full_state": fmt(full),
                        "direct_accuracy": pct(frame.loc["attn_direct_residual", "patched_target_correct"]),
                        "native_accuracy": pct(frame.loc["attn_native_mlp", "patched_target_correct"]),
                        "mlp_accuracy": pct(frame.loc["mlp_out_only", "patched_target_correct"]),
                    }
                )

    successor_stage_rows = []
    stage_continue = successor_stage[successor_stage["direction"] == "continue_into_close"]
    for count_bin in COUNT_BINS:
        for layer in (2, 3):
            frame = stage_continue[
                (stage_continue["count_bin"] == count_bin)
                & (stage_continue["layer"] == layer)
            ].set_index("stage")
            successor_stage_rows.append(
                {
                    "bin": count_bin,
                    "layer": f"Layer {layer + 1}",
                    "attn_component": fmt(frame.loc["attn_out", "evidence_gap"]),
                    "mlp_component": fmt(frame.loc["mlp_out", "evidence_gap"]),
                    "post_attn_gap": fmt(frame.loc["post_attn", "evidence_gap"]),
                    "post_mlp_gap": fmt(frame.loc["post_mlp", "evidence_gap"]),
                }
            )

    successor_pairs_per_direction = int(
        successor_conversion[
            (successor_conversion["direction"] == "continue_into_close")
            & (successor_conversion["intervention"] == "post_mlp_state")
            & (successor_conversion["layer"] == 3)
        ]["n_pairs"].sum()
    )

    def conversion_value(
        direction: str,
        count_bin: str,
        layer: int,
        intervention: str,
        field: str = "normalized_recovery",
    ) -> str:
        row = select_row(
            successor_conversion,
            direction=direction,
            count_bin=count_bin,
            layer=layer,
            intervention=intervention,
        )
        return pct(row[field]) if field == "patched_target_correct" else fmt(row[field])

    mlp_direction_labels = {
        "continue_into_close": "continue evidence → close receiver",
        "close_into_continue": "close evidence → continue receiver",
    }

    def mlp_feature_value(
        direction: str,
        count_bin: str,
        layer: int,
        family: str,
        support_size: int,
        field: str = "normalized_recovery",
    ) -> str:
        row = select_row(
            mlp_feature_summary,
            direction=direction,
            count_bin=count_bin,
            layer=layer,
            family=family,
            support_size=support_size,
        )
        return pct(row[field]) if field == "patched_target_correct" else fmt(row[field])

    mlp_feature_rows = []
    for direction, direction_label in mlp_direction_labels.items():
        for count_bin in COUNT_BINS:
            for layer in (2, 3):
                for support_size in (64, 256):
                    mlp_feature_rows.append(
                        {
                            "direction": direction_label,
                            "bin": count_bin,
                            "layer": f"Layer {layer + 1}",
                            "features": support_size,
                            "ranked_recovery": mlp_feature_value(
                                direction,
                                count_bin,
                                layer,
                                "ranked_feature_replacement",
                                support_size,
                            ),
                            "sparse_recovery": mlp_feature_value(
                                direction,
                                count_bin,
                                layer,
                                "sparse_mean_direction",
                                support_size,
                            ),
                            "random_recovery": mlp_feature_value(
                                direction,
                                count_bin,
                                layer,
                                "random_feature_replacement",
                                support_size,
                            ),
                            "ranked_accuracy": mlp_feature_value(
                                direction,
                                count_bin,
                                layer,
                                "ranked_feature_replacement",
                                support_size,
                                "patched_target_correct",
                            ),
                        }
                    )

    mlp_fit_pairs_per_direction = 29 * int(mlp_feature_manifest["fit_examples_per_k"])
    mlp_eval_pairs_per_direction = 29 * int(mlp_feature_manifest["eval_examples_per_k"])

    def stage_gap(count_bin: str, layer: int, stage: str) -> str:
        row = select_row(
            stage_continue,
            count_bin=count_bin,
            layer=layer,
            stage=stage,
        )
        return fmt(row.evidence_gap, 2)

    count_transport_lookup = {
        (row["mechanism"], row["bin"], row["heads"]): row for row in count_transport_rows
    }
    direct_transport_rows = [row for row in count_transport_rows if row["mechanism"] == "non-thinking broad"]
    cot_transport_rows = [row for row in count_transport_rows if row["mechanism"] == "CoT trace readout"]
    successor_table_rows = [row for row in successor_patch_rows if row["heads"] == 4]
    patch_rows = retrieval_patch_rows

    conflict_labels = {
        "prompt_count_only": "prompt=m, trace=n",
        "trace_count_and_length": "prompt=n, trace=m",
        "final_trace_index_only": "prompt/trace=n, final index=m",
        "trace_marker_identity_control": "trace marker identities rotated",
    }
    conflict_rows = []
    for conflict_type in (
        "prompt_count_only",
        "trace_count_and_length",
        "final_trace_index_only",
        "trace_marker_identity_control",
    ):
        for count_bin in COUNT_BINS:
            row = select_row(
                cot_conflict_summary,
                conflict_type=conflict_type,
                count_bin=count_bin,
            )
            conflict_rows.append(
                {
                    "condition": conflict_labels[conflict_type],
                    "bin": count_bin,
                    "receiver": pct(row.follows_receiver),
                    "donor": pct(row.follows_donor),
                    "target": pct(row.follows_target),
                    "margin": fmt(row.mean_target_margin, 2),
                    "abs_position": fmt(row.mean_abs_ans_position_shift, 1),
                }
            )

    def bridge_value(
        count_bin: str,
        intervention: str,
        layer: int,
        field: str = "mean_normalized_recovery",
    ) -> float:
        row = select_row(
            cot_bridge_summary,
            conflict_type="prompt_count_only",
            count_bin=count_bin,
            intervention_family="sublayer_or_residual",
            intervention=intervention,
            layer=layer,
        )
        return float(row[field])

    bridge_rows = []
    for count_bin in COUNT_BINS:
        for intervention, label in (
            ("attn_out", "Layer 1 attention output"),
            ("mlp_out", "Layer 1 MLP output"),
            ("post_mlp", "Layer 1 post-MLP residual"),
        ):
            bridge_rows.append(
                {
                    "bin": count_bin,
                    "component": label,
                    "recovery": fmt(bridge_value(count_bin, intervention, 0)),
                    "target_follow": pct(
                        bridge_value(count_bin, intervention, 0, "follows_target")
                    ),
                }
            )

    nonthinking_ranked = nonthinking_mlp_transport[
        nonthinking_mlp_transport.family == "ranked"
    ].copy()
    nonthinking_random = nonthinking_mlp_transport[
        nonthinking_mlp_transport.family == "random"
    ][
        [
            "count_bin",
            "layer",
            "support_size",
            "transport_slope",
            "mean_normalized_margin_recovery",
        ]
    ].rename(
        columns={
            "transport_slope": "random_transport_slope",
            "mean_normalized_margin_recovery": "random_margin_recovery",
        }
    )
    nonthinking_feature_comparison = nonthinking_ranked.merge(
        nonthinking_random,
        on=["count_bin", "layer", "support_size"],
        how="left",
    )
    nonthinking_feature_comparison["slope_gain"] = (
        nonthinking_feature_comparison.transport_slope
        - nonthinking_feature_comparison.random_transport_slope
    )
    nonthinking_feature_rows = []
    for count_bin in COUNT_BINS:
        subset = nonthinking_feature_comparison[
            nonthinking_feature_comparison.count_bin == count_bin
        ]
        chosen = [
            subset[(subset.layer == 0) & (subset.support_size == 128)].iloc[0],
            subset[(subset.layer == 0) & (subset.support_size == 1024)].iloc[0],
            subset[subset.support_size < 1024]
            .sort_values("slope_gain", ascending=False)
            .iloc[0],
        ]
        labels = ("Layer 1 top-128", "Layer 1 full MLP", "best sparse gain")
        seen: set[tuple[int, int]] = set()
        for label, row in zip(labels, chosen):
            key = (int(row.layer), int(row.support_size))
            if key in seen:
                continue
            seen.add(key)
            nonthinking_feature_rows.append(
                {
                    "bin": count_bin,
                    "selection": label,
                    "layer": f"Layer {int(row.layer) + 1}",
                    "features": int(row.support_size),
                    "ranked_slope": fmt(row.transport_slope),
                    "random_slope": fmt(row.random_transport_slope),
                    "gain": fmt(row.slope_gain),
                    "r2": fmt(row.transport_r2),
                    "margin_recovery": fmt(row.mean_normalized_margin_recovery),
                    "donor_follow": pct(row.follows_donor),
                }
            )

    steering_rows = []
    for count_bin in COUNT_BINS:
        for site, label in (
            ("nonthinking_final_answer", "non-thinking natural"),
            ("thinking_final_answer", "CoT natural"),
            ("thinking_fixed_trace_answer", "CoT counterfactual fixed-15 trace"),
        ):
            row = steering_gain[(steering_gain.site == site) & (steering_gain.count_bin == count_bin)].sort_values("r2", ascending=False).iloc[0]
            steering_rows.append(
                {
                    "bin": count_bin,
                    "site": label,
                    "layer": f"Layer {int(row.layer)+1}",
                    "gain": fmt(row.slope),
                    "r2": fmt(row.r2),
                }
            )

    path_site_labels = {
        "nonthinking_final_answer": "non-thinking natural final",
        "thinking_final_answer": "CoT natural final",
        "thinking_fixed_trace_answer": "CoT counterfactual fixed-15 final",
    }

    def path_metric(
        site: str,
        count_bin: str,
        method: str,
        field: str,
        *,
        layer: int = 3,
        alpha: float | None = None,
    ) -> float:
        frame = geometry_path_regression[
            (geometry_path_regression.site == site)
            & (geometry_path_regression.count_bin == count_bin)
            & (geometry_path_regression.layer == layer)
            & (geometry_path_regression.method == method)
        ]
        if alpha is not None:
            frame = frame[np.isclose(frame.alpha.astype(float), float(alpha))]
        if frame.empty:
            return math.nan
        return float(frame.iloc[0][field])

    adjacent_transport_rows = []
    for site, label in path_site_labels.items():
        for count_bin in COUNT_BINS:
            adjacent_transport_rows.append(
                {
                    "site": label,
                    "bin": count_bin,
                    "full_slope": fmt(
                        path_metric(
                            site,
                            count_bin,
                            "adjacent_centroid_transplant",
                            "transport_slope",
                        )
                    ),
                    "delta_slope": fmt(
                        path_metric(
                            site,
                            count_bin,
                            "adjacent_delta_transport",
                            "transport_slope",
                        )
                    ),
                    "delta_r2": fmt(
                        path_metric(
                            site,
                            count_bin,
                            "adjacent_delta_transport",
                            "transport_r2",
                        )
                    ),
                    "delta_mae": fmt(
                        path_metric(
                            site,
                            count_bin,
                            "adjacent_delta_transport",
                            "path_tracking_mae",
                        )
                    ),
                }
            )

    nonadjacent_path_rows = []
    for site in ("nonthinking_final_answer", "thinking_final_answer"):
        for count_bin in COUNT_BINS:
            nonadjacent_path_rows.append(
                {
                    "site": path_site_labels[site],
                    "bin": count_bin,
                    "chord_slope": fmt(
                        path_metric(
                            site,
                            count_bin,
                            "nonadjacent_chord_transport",
                            "transport_slope",
                            alpha=0.5,
                        )
                    ),
                    "chord_r2": fmt(
                        path_metric(
                            site,
                            count_bin,
                            "nonadjacent_chord_transport",
                            "transport_r2",
                            alpha=0.5,
                        )
                    ),
                    "chord_mae": fmt(
                        path_metric(
                            site,
                            count_bin,
                            "nonadjacent_chord_transport",
                            "path_tracking_mae",
                            alpha=0.5,
                        )
                    ),
                    "curve_slope": fmt(
                        path_metric(
                            site,
                            count_bin,
                            "nonadjacent_curve_transport",
                            "transport_slope",
                            alpha=0.5,
                        )
                    ),
                    "curve_r2": fmt(
                        path_metric(
                            site,
                            count_bin,
                            "nonadjacent_curve_transport",
                            "transport_r2",
                            alpha=0.5,
                        )
                    ),
                    "curve_mae": fmt(
                        path_metric(
                            site,
                            count_bin,
                            "nonadjacent_curve_transport",
                            "path_tracking_mae",
                            alpha=0.5,
                        )
                    ),
                }
            )

    endpoint_sanity = geometry_path_manifest["endpoint_sanity"]
    geometry_path_examples_per_count = int(geometry_path_manifest["examples_per_count"])

    pca_rows = []
    for site, label in (
        ("nonthinking_final_answer", "non-thinking final"),
        ("thinking_final_answer", "CoT natural final"),
        ("thinking_fixed_trace_answer", "CoT counterfactual fixed-15 final"),
    ):
        for layer in range(4):
            row = select_row(geometry, site=site, layer=layer)
            pca_rows.append(
                {
                    "site": label,
                    "layer": f"Layer {layer+1}",
                    "pc1": pct(row.pc1_variance),
                    "pc3": pct(row.pc3_cumulative),
                    "pc6": pct(row.pc6_cumulative),
                    "dim": fmt(row.effective_dimension),
                    "turn": f"{fmt(row.mean_turning_angle_degrees, 1)}°",
                }
            )

    depth_geometry_rows = []
    for site, label in (
        ("nonthinking_final_answer", "non-thinking final"),
        ("thinking_final_answer", "CoT natural final"),
        ("thinking_fixed_trace_answer", "CoT counterfactual fixed-15 final"),
    ):
        for layer in (0, 3):
            geo = select_row(geometry, site=site, layer=layer)
            spread = select_row(absolute_spread, site=site, layer=layer)
            depth_geometry_rows.append(
                {
                    "site": label,
                    "layer": f"Layer {layer+1}",
                    "pc1": pct(geo.pc1_variance),
                    "pc6": pct(geo.pc6_cumulative),
                    "dim": fmt(geo.effective_dimension),
                    "radius": fmt(spread.rms_centroid_radius, 2),
                    "adjacent": fmt(spread.mean_adjacent_distance, 2),
                    "turn": f"{fmt(geo.mean_turning_angle_degrees, 1)}°",
                }
            )

    trace_pca_rows = []
    for layer in range(4):
        row = select_row(trace_geometry, layer=layer)
        trace_pca_rows.append(
            {
                "layer": f"Layer {layer+1}",
                "pc1": pct(row.pc1_variance),
                "pc3": pct(sum(float(row[f"pc{i}_variance"]) for i in range(1, 4))),
                "pc6": pct(row.pc6_cumulative),
                "dim": fmt(row.effective_dimension),
            }
        )

    prompt_length = int(config["seq_len"])
    fixed_trace_steps = 15
    fixed_ans_position = prompt_length + 3 + 2 * fixed_trace_steps
    natural_ans_position = f"{prompt_length + 3} + 2n"
    geometry_sites = [
        {
            "site": "Non-thinking natural",
            "sequence": code("<BOS> prompt(n needles) <Ans> <n> <EOS>"),
            "ans": code(prompt_length + 1),
            "role": "自然 direct-count readout；prompt 与正确答案 n 随样本变化。",
        },
        {
            "site": "CoT natural",
            "sequence": code("<BOS> prompt(n) <Think> <1>M1 ... <n>Mn </Think> <Ans> <n> <EOS>"),
            "ans": code(natural_ans_position),
            "role": "自然 CoT readout；trace 内容、trace 长度与 <code>&lt;Ans&gt;</code> 绝对位置都随 n 变化。",
        },
        {
            "site": "CoT counterfactual fixed-15",
            "sequence": code("<BOS> prompt(n) <Think> <1>T1 ... <15>T15 </Think> <Ans> <n> <EOS>"),
            "ans": code(fixed_ans_position),
            "role": "反事实控制；prompt/答案仍为 n，但 15 步模板 trace 与 <code>&lt;Ans&gt;</code> 位置对所有 n 完全固定。",
        },
    ]

    fixed_geo_l1 = select_row(geometry, site="thinking_fixed_trace_answer", layer=0)
    fixed_geo_l4 = select_row(geometry, site="thinking_fixed_trace_answer", layer=3)
    direct_geo_l1 = select_row(geometry, site="nonthinking_final_answer", layer=0)
    direct_geo_l4 = select_row(geometry, site="nonthinking_final_answer", layer=3)
    natural_geo_l1 = select_row(geometry, site="thinking_final_answer", layer=0)
    natural_geo_l4 = select_row(geometry, site="thinking_final_answer", layer=3)
    direct_spread_l1 = select_row(absolute_spread, site="nonthinking_final_answer", layer=0)
    direct_spread_l4 = select_row(absolute_spread, site="nonthinking_final_answer", layer=3)
    natural_spread_l1 = select_row(absolute_spread, site="thinking_final_answer", layer=0)
    natural_spread_l4 = select_row(absolute_spread, site="thinking_final_answer", layer=3)
    fixed_transport_l4 = {
        count_bin: select_row(
            centroid_reg,
            site="thinking_fixed_trace_answer",
            count_bin=count_bin,
            layer=3,
        )
        for count_bin in COUNT_BINS
    }
    fixed_transport_l4_display = {
        count_bin: ("≈0.000" if abs(float(row.slope)) < 5e-4 else fmt(row.slope))
        for count_bin, row in fixed_transport_l4.items()
    }

    output = run_dir / "syn_v10_report.html"
    css = """
    :root{--ink:#172033;--muted:#526078;--line:#dce4ef;--soft:#f5f8fc;--blue:#2563eb;--green:#15803d;--orange:#ea580c;--red:#b91c1c}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#edf2f7;color:var(--ink);font-family:Inter,"Noto Sans SC","Microsoft YaHei",Arial,sans-serif;line-height:1.72}main{max-width:1460px;margin:auto;background:white;min-height:100vh;padding:46px 58px 84px}h1{font-size:36px;line-height:1.22;margin:0 0 12px}h2{font-size:27px;margin:54px 0 20px;padding-top:12px;border-top:1px solid var(--line)}h3{font-size:18px;margin:0 0 12px}p,li{font-size:16px}.subtitle{font-size:18px;color:var(--muted);max-width:1120px}.meta,.small{font-size:14px;color:var(--muted)}code{background:#edf2f7;padding:2px 6px;border-radius:4px;font-family:"Cascadia Mono",Consolas,monospace;overflow-wrap:anywhere}.toc{columns:2;border:1px solid var(--line);background:var(--soft);padding:20px 28px;margin:28px 0}.toc a{color:#1d4ed8;text-decoration:none}.callout{border-left:5px solid var(--blue);background:#eff6ff;padding:16px 20px;margin:18px 0;border-radius:0 8px 8px 0}.callout.good{border-color:#16a34a;background:#f0fdf4}.callout.warn{border-color:#f59e0b;background:#fffbeb}.callout.limit{border-color:#dc2626;background:#fef2f2}.protocol{border:1px solid var(--line);background:#fbfdff;padding:18px 22px;border-radius:8px;margin:16px 0}.formula{border:1px solid var(--line);background:#f8fafc;padding:13px 17px;margin:12px 0;font-family:"Cambria Math",serif;overflow-x:auto}.table-wrap{overflow-x:auto;margin:16px 0 24px}table{width:100%;border-collapse:collapse;font-size:14.5px}th,td{border:1px solid var(--line);padding:10px 12px;vertical-align:top}th{background:#eaf0f7;text-align:left}tr:nth-child(even) td{background:#fbfdff}.figure,.interactive-figure{border:1px solid var(--line);border-radius:8px;padding:18px;margin:22px 0;background:#fff}.figure img{display:block;width:100%;max-height:930px;object-fit:contain;margin:auto}.figure figcaption,.interactive-figure figcaption{color:#44526a;font-size:14.5px;margin-top:12px}.controls{display:flex;flex-wrap:wrap;gap:12px 18px;align-items:end;background:var(--soft);padding:12px;border-radius:6px}.controls label{font-size:13px;font-weight:650;display:flex;flex-direction:column;gap:4px}.controls select,.controls button{font:inherit;padding:7px 9px;border:1px solid #b8c4d5;border-radius:4px;background:white}.stats{margin:10px 0;color:#334155;font-size:14px}.interactive-figure canvas{display:block;width:100%;height:590px;border:1px solid var(--line);background:#fbfdff}.mechanisms{display:grid;grid-template-columns:1fr 1fr;gap:18px}.mechanism{border:1px solid var(--line);border-radius:8px;padding:18px;background:#fbfdff}.flow{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:8px;padding:14px 0}.node{padding:9px 12px;border:1px solid #93c5fd;background:#eff6ff;border-radius:5px;font-weight:650}.arrow{font-size:22px;color:#64748b}
    .metric-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:16px 0 26px}.metric-card{border:1px solid #d3ddea;border-radius:7px;background:#fbfdff;padding:16px 17px;min-width:0}.metric-card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:11px}.metric-card-head code{font-size:14px;font-weight:750;color:#1e3a5f;background:#e7f0fb}.metric-context{font-size:12px;text-align:right;color:#526078;line-height:1.35}.metric-context span{display:block;text-transform:uppercase;font-size:10px;font-weight:800;color:#7c8ba1;letter-spacing:.04em}.metric-card p{font-size:14px;line-height:1.58;color:#44526a;margin:11px 0 0}.equation{display:flex;align-items:center;justify-content:center;min-height:92px;text-align:center;color:#111827;background:#fff;border:1px solid #e0e7f0;border-radius:5px;padding:12px 14px;overflow-x:auto;overflow-y:hidden}.equation.compact{min-height:64px;margin-top:8px}.equation math{display:block;margin:auto;font-family:"STIX Two Math","Latin Modern Math","Cambria Math",serif;font-size:24px;line-height:1.25}.equation mtext{font-family:Inter,"Noto Sans SC","Microsoft YaHei",Arial,sans-serif;font-size:.72em}.formula-block{border:1px solid var(--line);background:#f8fafc;padding:15px 17px;margin:13px 0;border-radius:6px}.formula-block h4{font-size:15px;margin:0 0 9px}.formula-block p{font-size:14.5px;color:#44526a;margin:10px 0 0}.symbol-key{display:flex;flex-wrap:wrap;gap:8px 18px;margin:10px 0}.symbol-key span{font-size:14px;color:#44526a}.symbol-key b{font-family:"STIX Two Math","Latin Modern Math","Cambria Math",serif;font-size:17px;color:#172033}.mechanism-explorer{border:1px solid var(--line);border-radius:8px;padding:20px;margin:22px 0;background:#fff}.mechanism-explorer-head{display:flex;justify-content:space-between;gap:24px;align-items:flex-start}.mechanism-explorer-head p{margin:5px 0 0;color:var(--muted);font-size:14.5px;max-width:780px}.mechanism-mode-switch{display:flex;min-width:290px;border:1px solid #aebbd0;border-radius:6px;overflow:hidden}.mechanism-mode{flex:1;border:0;border-right:1px solid #aebbd0;background:#fff;color:#334155;padding:10px 13px;font:inherit;font-size:14px;font-weight:700;cursor:pointer}.mechanism-mode:last-child{border-right:0}.mechanism-mode.active{background:#1d4ed8;color:#fff}.mechanism-player{margin-top:16px;border:1px solid var(--line);border-radius:6px;background:#fbfdff;overflow:hidden}.mechanism-scene{display:none;width:100%;height:auto;min-height:390px}.mechanism-scene.active{display:block}.mechanism-scene text{font-family:Inter,"Noto Sans SC","Microsoft YaHei",Arial,sans-serif;fill:#243047;font-size:16px;text-anchor:middle}.mechanism-scene .mech-region{fill:#f7faff;stroke:#b9c6d8;stroke-width:2}.mechanism-scene .mech-region.trace{fill:#f2fbf5;stroke:#a6d4b2}.mechanism-scene .mech-region-title{text-anchor:start;font-size:17px;font-weight:750}.mechanism-scene .mech-node rect{stroke-width:2}.mechanism-scene .mech-node.needle rect,.mechanism-scene .mech-node.trace-marker rect{fill:#dcfce7;stroke:#22a55b}.mechanism-scene .mech-node.query rect,.mechanism-scene .mech-node.trace-token rect{fill:#dbeafe;stroke:#3b82f6}.mechanism-scene .mech-node.state rect{fill:#fff7ed;stroke:#f97316}.mechanism-scene .mech-node.transform rect{fill:#f3e8ff;stroke:#8b5cf6}.mechanism-scene .mech-node.answer rect{fill:#fee2e2;stroke:#dc2626}.mechanism-scene .mech-node.target rect,.mechanism-scene .mech-node.active-marker rect,.mechanism-scene .mech-node.successor rect{stroke-width:4}.mechanism-scene .mech-node text{font-weight:750}.mechanism-scene .mech-subtext{font-size:13px;font-weight:500;fill:#526078}.mechanism-scene .mech-answer-token{font-size:18px;font-weight:850;fill:#b91c1c}.mechanism-scene .mech-noise circle{fill:#cbd5e1}.mechanism-scene .mech-ellipsis{font-size:25px}.mechanism-scene .mech-step{opacity:.1;transition:opacity .45s ease}.mechanism-scene .mech-step.visible{opacity:1}.mechanism-scene .mech-step.current .mech-node{filter:drop-shadow(0 0 7px rgba(37,99,235,.33))}.mechanism-scene .mech-links path,.mechanism-scene .mech-flow-arrow,.mechanism-scene .mech-loop-arrow{fill:none;stroke:#2563eb;stroke-width:3;color:#2563eb;marker-end:url(#direct-arrow)}.mechanism-scene .targeted-link path,.mechanism-scene .mech-flow-arrow.cot,.mechanism-scene .mech-loop-arrow{stroke:#16a34a;color:#16a34a;marker-end:url(#cot-arrow)}.mechanism-scene .aggregate-links path{stroke:#ea580c;color:#ea580c}.mechanism-scene .mech-uncertain-arrow{fill:none;stroke:#64748b;stroke-width:3;stroke-dasharray:9 7;marker-end:url(#cot-arrow)}.mechanism-scene .mech-step.current path{stroke-dasharray:9 7;animation:mechanismDash 1.1s linear infinite}.mechanism-timeline{display:flex;align-items:center;justify-content:center;gap:14px;margin:16px 0 12px}.mechanism-timeline>button{border:1px solid #aebbd0;border-radius:5px;background:#fff;color:#243047;min-width:42px;height:38px;font-size:23px;cursor:pointer}.mechanism-timeline .mechanism-play{font-size:14px;min-width:92px;font-weight:700}.mechanism-dots{display:flex;gap:10px}.mechanism-dot{width:12px;height:12px;border:1px solid #64748b;border-radius:50%;padding:0;background:#fff;cursor:pointer}.mechanism-dot.active{background:#2563eb;border-color:#2563eb;box-shadow:0 0 0 4px #dbeafe}.mechanism-step-copy{border-left:5px solid #2563eb;border-radius:0 7px 7px 0;background:#eff6ff;padding:13px 17px}.mechanism-step-title{font-size:17px;font-weight:800;margin-bottom:7px}.mechanism-step-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.mechanism-step-grid p{margin:0;background:rgba(255,255,255,.74);border:1px solid #d7e3f5;border-radius:5px;padding:9px 11px;font-size:14px;line-height:1.55}.mechanism-explorer figcaption{color:#44526a;font-size:14.5px;margin-top:13px}@keyframes mechanismDash{to{stroke-dashoffset:-32px}}@media(prefers-reduced-motion:reduce){.mechanism-scene .mech-step{transition:none}.mechanism-scene .mech-step.current path{animation:none}}
    @media(max-width:900px){main{padding:28px 18px}.toc{columns:1}.mechanisms,.metric-grid{grid-template-columns:1fr}.interactive-figure canvas{height:430px}.mechanism-explorer-head{display:block}.mechanism-mode-switch{min-width:0;width:100%;margin-top:14px}.mechanism-step-grid{grid-template-columns:1fr}.mechanism-scene{min-height:300px}.metric-card-head{display:block}.metric-context{text-align:left;margin-top:8px}}
    @media print{body{background:white}main{max-width:none;padding:20px}.interactive-figure,.mechanism-timeline{display:none}.figure{break-inside:avoid}.mechanism-scene .mech-step{opacity:1}.mechanism-step-copy{break-inside:avoid}}
    """
    settings = [
        {"item": "研究对象", "value": "两个独立随机初始化、参数规模相同的 decoder-only GPT-2；分别训练 non-thinking 与 explicit-CoT，不共享参数。"},
        {"item": "模型", "value": f"4 Layers × 4 heads，d_model={config['n_embd']}，MLP={config['n_inner']}，learned absolute position embeddings，context={config['n_positions']}。"},
        {"item": "数据", "value": "prompt 固定 256 tokens；64 类 noise；每例从 10 类 marker 有放回采样 1–30 个 needle，位置均匀无放回；三个区间各含十个平衡 exact-count 类。"},
        {"item": "non-thinking sequence", "value": "<BOS> prompt[256] <Ans> <n> <EOS>；只监督最终 count 与 EOS。"},
        {"item": "CoT sequence", "value": "<BOS> prompt[256] <Think> <1> M1 ... <n> Mn </Think> <Ans> <n> <EOS>；trace index 与答案共享同一套 number tokens。"},
        {"item": "训练", "value": f"每个模型 {config['train_steps']} steps；batch={config['batch_size']}；AdamW lr={config['lr']}；warmup={config['warmup_steps']}；weight decay={config['weight_decay']}；seed={config['seed']}。"},
        {"item": "正式 ablation", "value": f"每个 exact count {manifest['ablation_examples_per_exact_count']} 个新 prompts，共 {30*manifest['ablation_examples_per_exact_count']} 个；16 个单头；top-1 至 top-16；与固定随机删除路径、逐步随机均值、min–max 和 random-4 mean 比较。"},
        {"item": "分层原则", "value": "所有 bin 统计直接从 gold/receiver count 属于该区间的逐样本行重新计算；all-count 不是三个区间均值的均值。"},
    ]
    single_ablation_caption = (
        "三列分别为 count 1–10、11–20、21–30。第一行是 non-thinking final-count "
        "accuracy drop；第二行是 CoT trace-marker teacher-forced accuracy drop；第三行是 "
        "CoT final-count accuracy drop。每格对应一个全局被 mask 的 Layer×head。注意："
        "全局 mask 不能说明该 head 在哪个 token position 起作用。"
    )
    cumulative_ablation_caption = (
        "<b>横轴</b>是同时全局 mask 的 head 数；<b>纵轴</b>是剩余 accuracy。"
        "蓝=按机制分数从高到低；红=同一排名从低到高；灰线与阴影="
        f"{manifest['ablation_random_orders']} 个随机顺序的均值与 min–max。"
        "三行依次检验 non-thinking final、CoT trace marker、CoT final；三列是 count 区间。"
        "若蓝线比红/灰更早下降，才支持排名捕捉到特异机制，而不只是删更多 heads 都会坏。"
    )
    mechanism_rows = [
        {
            "aspect": "已支持的主要路由",
            "direct": "Layer-1 broad-head bundle 在 <Ans> 并行访问 needle set，并把集合证据写入后续 count residual。",
            "cot": "progress residual 控制数字 <k> 的 targeted multi-head routing；retrieved marker 再进入 successor/close 更新。",
        },
        {
            "aspect": "head-level 因果状态",
            "direct": "Broad top-n local ablation 破坏 final count；donor broad slices 可运输部分 count，且 head mask 同步破坏 count geometry 与输出。",
            "cot": "Targeted top-2/top-4 clean patch 恢复 M_k；successor bundle 双向运输 continue/close evidence；单头均可替代。",
        },
        {
            "aspect": "hidden-state 因果状态",
            "direct": "Final <Ans> count state 随 Layer 逐步成形；Layer-4 natural residual/centroid 对 donor count 近一比一充分。",
            "cot": "Trace marker residual 是 progress/termination state；final <Ans> residual 是另一种 scalar-count state，二者不能互换。",
        },
        {
            "aspect": "最清楚的双向联系",
            "direct": "未观察到对称反馈；更符合 broad heads → count state → logits 的前馈链。",
            "cot": "Targeted heads → marker-identity state，且 progress state → 下一步 targeted routing；这是当前最强的局部双向耦合。",
        },
        {
            "aspect": "仍未定位的桥接",
            "direct": "低/中 count 已定位可运输 donor offset 的 post-GELU feature 子集；高 count 仍缺跨层、跨 1–30 通用的稀疏算术 basis。",
            "cot": "最终答案行为跟 trace pair 数/extent，Layer-1 MLP output 可运输 donor scalar；但 trace extent 与 <Ans> 绝对位置仍共变。",
        },
        {
            "aspect": "被当前结果排除的纯版本",
            "direct": "不支持一根跨 1–30 恒定的全局 +1 轴，也不支持后期 count state 反向重建早期 broad routing。",
            "cot": "不支持唯一 retrieval head、final marker=scalar count register，或一组固定 readout heads 独立写出最终 count。",
        },
    ]
    object_section = f"""
    <section id="object">
      <h2>1. 研究对象、目标与证据更新后的两个工作机制</h2>
      <p>两个独立模型看到同一类 256-token prompt、同一套 marker tokens 和同一套 count tokens，模型规模完全相同；差别只在于<b>被要求生成什么输出序列</b>。Non-thinking 模型把 prompt 直接压缩成一个 count。CoT 模型则先生成逐项 indexed trace，再回答同一个 count。最初我们把它们分别假设为“并行集合聚合”和“逐项检索循环”；经过第 5–11 节的描述、ablation、clean-to-corrupt patch、MLP feature patch、geometry steering 与 residual transplant 后，现在可以把若干箭头升级为工作机制，同时保留尚未闭合的桥接。</p>
      <div class="callout good"><b>当前最稳健的总图景。</b>Non-thinking 是非对称前馈链：<code>Layer-1 broad routing → post-GELU count features → distributed count residual → C_n</code>。CoT 的 trace 内部形成局部双向循环：<code>progress state → targeted retrieval → M_k → successor/close → updated progress</code>；最终 <code>&lt;Ans&gt;</code> 则跟随 trace pair 数/extent，且 Layer-1 MLP output 已足以运输 donor scalar count，但这一桥仍可能利用 learned absolute position。</div>
      <div class="callout warn"><b>证据边界。</b>这里的“支持”表示干预效应大于匹配随机对照，或 clean/donor activation 能在 held-out pair 中运输目标信息；不是跨 seed 的总体统计定理。实线工作路径也不意味着由单个 head 完成。CoT 的 progress state 与 final scalar-count state 必须分开：前者控制下一 index/close，后者直接控制最终 count logits。</div>
      <div class="mechanisms">
        <div class="mechanism">
          <h3>工作机制 A · Non-thinking：broad set routing 写入后层 count state</h3>
          <p><b>已支持：</b>最终 <code>&lt;Ans&gt;</code> query 的 Layer-1 broad heads 并行覆盖多个 prompt needles。Position-local ablation 使 final accuracy 大幅下降；donor broad slices 能运输部分 count；mask broad group 又会同步破坏末层 count geometry 与输出。Layer-4 natural residual/centroid transplant 最终可以一比一搬运 donor count。</p>
          <p><b>当前解释：</b>早期 attention 负责收集集合证据；held-out post-GELU feature patch 进一步证明低/中 count 的一部分 MLP features 能按 donor offset 运输数值，再由 residual/后层离散化为 30 个 count logits。高 count 的 state 成形更晚，且局部 centroid path 有效而全局直线 <code>+1</code> 轴不成立。</p>
          <p><b>仍缺：</b>还没有一套跨 1–30 与跨层通用的稀疏 feature basis；高 count 的 sparse transport 较弱，也没有证据表明后期 count state 会反向控制早期 broad routing。</p>
          <div class="flow"><span class="node">needle set</span><span class="arrow">→</span><span class="node">parallel broad retrieval</span><span class="arrow">→</span><span class="node">count residual</span><span class="arrow">→</span><span class="node">C_n</span></div>
        </div>
        <div class="mechanism">
          <h3>工作机制 B · CoT：targeted retrieval/progress loop，加上独立 final count state</h3>
          <p><b>已支持的 trace loop：</b>数字 <code>&lt;k&gt;</code> 的 targeted multi-head bundle 定位第 k 个 prompt needle并写出 <code>M_k</code>；marker 后 residual 保存可执行 progress/termination state；Layer-3 MLP 转换 routed evidence，Layer-4 的分布式 feature 子空间把它写成 <code>&lt;k+1&gt;</code> 或 <code>&lt;/Think&gt;</code> logit。Progress transplant 还能反向改变下一步 targeted routing，形成当前最清楚的局部双向因果耦合。</p>
          <p><b>必须分开的 final state：</b>最后一个 marker residual 能改变继续/关闭，却不能独立运输最终 scalar count。Source-conflict 显示最终答案跟 trace pair 数/extent，而非 prompt count、marker identity 或最后 trace 数字；clean Layer-1 MLP output 与 post-MLP residual 已能把 donor scalar 写入 conflict receiver。</p>
          <p><b>仍缺：</b>trace pair 数每变化 1，<code>&lt;Ans&gt;</code> 绝对位置也变化 2。由于模型使用 learned absolute positions，当前实验还不能区分真正的 trace-length aggregation 与 position lookup；固定少数 trace-readout heads 也不足以解释这座桥。</p>
          <div class="flow"><span class="node">progress k</span><span class="arrow">↔</span><span class="node">targeted retrieval</span><span class="arrow">→</span><span class="node">M_k</span><span class="arrow">→</span><span class="node">k+1 / close</span></div>
          <div class="flow"><span class="node">complete trace/context</span><span class="arrow">⇢</span><span class="node">final scalar-count state</span><span class="arrow">→</span><span class="node">C_n</span></div>
        </div>
      </div>
      {mechanism_explorer()}
      <h3>两种工作机制的证据对账</h3>
      {table(mechanism_rows, [('aspect','机制问题'),('direct','Non-thinking：direct aggregation'),('cot','Thinking：retrieval/progress loop')])}
      <div class="callout"><b>如何阅读后文。</b>Attention/PCA 只提出候选；position-local ablation 测某组件在指定 query 的必要性；clean-to-corrupt head-output patch 与 held-out MLP-feature patch 测局部充分性；donor-to-receiver residual transplant 测完整状态是否可执行。只有这些证据方向一致，本文才使用“候选 circuit”。第 7–10 节分别拆组件，第 11 节再检验 head 与 hidden state 是否处在同一条因果链上。</div>
    </section>
    """
    attention_metric_rows = [
        {
            "name": "prompt_needles_mass",
            "query": "任意被分析 query q",
            "formula_html": "<i>M</i><sub>N</sub>(<i>q</i>) = ∑<sub><i>j</i>∈<i>N</i></sub> <i>A</i><sup>(ℓ,h)</sup><sub>qj</sub>",
            "meaning": "该 head 投向全部 prompt needle 位置的总 attention mass。只测“看了多少 needle”，不测是否均匀。",
        },
        {
            "name": "needle_entropy_normalized",
            "query": "任意被分析 query q",
            "formula_html": "<i>H</i><sub>N</sub>(<i>q</i>) = <span class='frac'><span>−∑<sub><i>j</i>∈<i>N</i></sub> <i>p</i><sub>j</sub> log <i>p</i><sub>j</sub></span><span>log <i>n</i></span></span>, &nbsp; <i>p</i><sub>j</sub> = <span class='frac'><span><i>A</i><sub>qj</sub></span><span><i>M</i><sub>N</sub>(<i>q</i>)</span></span>",
            "meaning": "条件在“attention 已落入 needle 子集”后，分布覆盖多个 needles 的均匀程度。n≤1 或 needle mass=0 时定义为 0。",
        },
        {
            "name": "broad_attention_score",
            "query": "Non-thinking 的 <Ans>",
            "formula_html": "<i>S</i><sub>broad</sub><sup>(ℓ,h)</sup>(<i>q</i>) = <i>M</i><sub>N</sub>(<i>q</i>) · <i>H</i><sub>N</sub>(<i>q</i>)",
            "meaning": "同时奖励“总 mass 在 needles 上”与“在多个 needles 间广泛分布”；范围 [0,1]，用于 direct_broad head 排名。",
        },
        {
            "name": "correct_prompt_needle_mass / k-to-k mass",
            "query": "CoT trace 中的数字 token <k>",
            "formula_html": "<i>S</i><sub>target</sub><sup>(ℓ,h)</sup>(<i>k</i>) = <i>A</i><sup>(ℓ,h)</sup><sub>q<sub>k</sub>, n<sub>k</sub></sub>",
            "meaning": "从数字 query <k> 到 prompt 中按位置排序的第 k 个 needle 的 raw mass；这是 targeted_retrieval 排名真正使用的 score。",
        },
        {
            "name": "correct_top1",
            "query": "CoT trace 中的数字 token <k>",
            "formula_html": "𝟙 [ arg max<sub><i>j</i>∈<i>N</i></sub> <i>A</i><sup>(ℓ,h)</sup><sub>q<sub>k</sub>,j</sub> = <i>n</i><sub>k</sub> ]",
            "meaning": "只在 prompt needle 子集内判断最大 attention 是否落在 matching needle；不要求它是整个上下文的 top-1。",
        },
        {
            "name": "diagonal_dominance",
            "query": "CoT trace 中的数字 token <k>",
            "formula_html": "<i>D</i><sup>(ℓ,h)</sup>(<i>k</i>) = <span class='frac'><span><i>A</i><sup>(ℓ,h)</sup><sub>q<sub>k</sub>,n<sub>k</sub></sub></span><span>∑<sub><i>j</i>∈<i>N</i></sub> <i>A</i><sup>(ℓ,h)</sup><sub>q<sub>k</sub>,j</sub></span></span>",
            "meaning": "条件在“投向 needle 的 mass”内部，有多大比例给了 matching needle。它可以接近 1，但 raw k-to-k mass 仍很低。",
        },
        {
            "name": "trace_markers_mass / trace-readout score",
            "query": "CoT 的最终 <Ans>",
            "formula_html": "<i>S</i><sub>trace</sub><sup>(ℓ,h)</sup> = ∑<sub><i>j</i>∈<i>T</i><sub>M</sub></sub> <i>A</i><sup>(ℓ,h)</sup><sub>q<sub>ans</sub>,j</sub>",
            "meaning": "最终答案 query 投向全部 trace marker tokens M1…Mn 的总 mass；不包括 trace 数字 tokens，用于 trace_readout head 排名。",
        },
        {
            "name": "next_prompt_needle_mass / successor score",
            "query": "CoT trace marker M_k，且 k<n",
            "formula_html": "<i>S</i><sub>succ</sub><sup>(ℓ,h)</sup>(<i>k</i>) = <i>A</i><sup>(ℓ,h)</sup><sub>q<sub>succ</sub>(k), n<sub>k+1</sub></sub>",
            "meaning": "当前 marker query 对下一 prompt needle 的 raw mass；用于寻找可能参与 successor/next-retrieval 准备的 heads。",
        },
    ]
    prediction_metric_rows = [
        {
            "name": "TF final-count accuracy",
            "unit": "每个 example",
            "definition": "给定完整 gold prefix，在 <Ans> logits 中只取 30 个 number tokens 的 argmax；等于 gold count 记 1。",
            "note": "CoT 的 gold trace 已给定，所以它测 final readout，不是端到端生成。",
        },
        {
            "name": "TF trace-marker accuracy",
            "unit": "每个 trace marker query",
            "definition": "在数字 <k> 位置，用全词表 argmax 预测下一 token；恰为 gold M_k 记 1，再对全部 k queries 求平均。",
            "note": "高 count example 有更多 k，因此汇总值是 query-weighted。",
        },
        {
            "name": "TF trace-index accuracy",
            "unit": "每个 trace index query",
            "definition": "在 <Think> 或前一个 M_{k−1} 位置，用全词表 argmax 预测下一数字 k；正确记 1。",
            "note": "它测 successor/index emission，不测 matching needle retrieval。",
        },
        {
            "name": "AR final-count accuracy",
            "unit": "每个 example",
            "definition": "只给 prompt 与格式起点，模型自由生成后续 token；解析最终 number token，等于 gold count 记 1。",
            "note": "会累积早期 trace 错误，是端到端指标。",
        },
        {
            "name": "trace exact match",
            "unit": "每个 CoT example",
            "definition": "自由生成的完整 index-marker trace 与 gold trace 的 token 序列完全相同才记 1。",
            "note": "比 marker recall 更严格，任意一步错误都会使该例为 0。",
        },
    ]
    causal_metric_rows = [
        {
            "name": "ablation drop",
            "formula_html": "Δ<sub>abl</sub> = metric<sub>baseline</sub> − metric<sub>masked</sub>",
            "meaning": ">0 表示 mask 使性能下降；=0 表示该指标无变化；<0 表示干预后反而改善。",
        },
        {
            "name": "normalized recovery",
            "formula_html": "<i>R</i><sub>norm</sub> = <span class='frac'><span><i>m</i><sub>patched</sub> − <i>m</i><sub>corrupt</sub></span><span><i>m</i><sub>clean</sub> − <i>m</i><sub>corrupt</sub></span></span>",
            "meaning": "0=没有恢复，1=恢复到 clean margin，>1=过度恢复，<0=比 corrupt 更差；分母近 0 时记 NaN。先逐 pair 计算，再平均。",
        },
        {
            "name": "expected-count shift",
            "formula_html": "Δ<i>E</i>[<i>C</i>] = <i>E</i><sub>patch</sub>[<i>C</i>] − <i>E</i><sub>base</sub>[<i>C</i>]",
            "meaning": "连续地测 logits 分布移动，比只看 argmax 是否翻转更敏感。",
        },
        {
            "name": "transport slope",
            "formula_html": "Δ<i>E</i>[<i>C</i>] = <i>a</i> + <i>b</i> (<i>c</i><sub>donor</sub> − <i>c</i><sub>receiver</sub>)",
            "meaning": "用 OLS 拟合。b≈1 表示 donor state 一比一搬运 count；b≈0 表示输出不随 donor offset；负值表示反向移动。",
        },
        {
            "name": "steering gain 与 R²",
            "formula_html": "Δ<i>E</i>[<i>C</i>] = <i>a</i> + <i>gα</i>",
            "meaning": "用 OLS 拟合。g 是每单位 alpha 的输出移动量；R² 是该剂量反应被直线解释的比例。理想统一 +1 轴应有 g≈1 且 R²≈1。",
        },
        {
            "name": "PCA explained-variance ratio",
            "formula_html": "EVR<sub>r</sub> = <span class='frac'><span>λ<sub>r</sub></span><span>∑<sub>s</sub> λ<sub>s</sub></span></span>",
            "meaning": "λ_r 是 exact-count centroid 矩阵第 r 个主成分的方差；这里只描述 count 类均值之间的几何。",
        },
        {
            "name": "effective dimension",
            "formula_html": "<i>d</i><sub>eff</sub> = <span class='frac'><span>(∑<sub>r</sub> λ<sub>r</sub>)²</span><span>∑<sub>r</sub> λ<sub>r</sub>²</span></span>",
            "meaning": "participation ratio；接近 1 表示方差集中在一轴，越大表示 count-centroid geometry 越分散。",
        },
    ]
    definitions_section = f"""
    <section id="definitions">
      <h2>3. 新术语、数据列与计算公式</h2>
      <p>本节固定后文所有图表和表格的计算口径。先定义单个 attention row，再定义跨 query/head 的汇总 score，最后定义输出与因果干预指标。除非特别注明，Layer 和 head 在数据表中使用 0-based 编号；报告标题显示 Layer 1–4。</p>

      <h3>3.1 Attention 记号、query 位置与 token 集合</h3>
      <div class="formula-block">
        <h4>单头 causal attention</h4>
        <div class="equation">{FORMULA_MATHML['attention softmax']}</div>
        <p>这里 <i>q</i> 是 query 位置，<i>j≤q</i> 是 causal mask 允许访问的 key 位置。因此每一行满足 <i>A</i><sub>qj</sub>≥0 且 ∑<sub>j≤q</sub><i>A</i><sub>qj</sub>=1；下文的 attention mass 都直接对这一概率行求和。</p>
      </div>
      <div class="protocol"><b>集合与 query。</b><div class="symbol-key"><span><b>N={{n₁,…,nₙ}}</b>：按 prompt 绝对位置排序的 needles</span><span><b>U</b>：prompt noise positions</span><span><b>T<sub>I</sub></b>：trace 数字 positions</span><span><b>T<sub>M</sub></b>：trace marker positions</span></div><code>q_ans</code> 是 <code>&lt;Ans&gt;</code> token 的位置，它的 logits 预测最终 count；<code>q_k</code> 是 trace 数字 <code>&lt;k&gt;</code> 的位置，它的 logits 预测紧随其后的 <code>M_k</code>；<code>q_succ(k)</code> 是 trace marker <code>M_k</code> 的位置，它的 logits 预测 <code>k+1</code> 或 <code>&lt;/Think&gt;</code>。</div>
      <div class="formula-block">
        <h4>任意 token 类别的 attention mass</h4>
        <div class="equation">{FORMULA_MATHML['attention mass']}</div>
        <p>例如 <code>prompt_noise_mass</code> 是 <i>M</i>(<i>U</i>|<i>q</i>)，<code>bos_mass</code> 是 <i>A</i><sub>q,BOS</sub>。类别 mass 只描述 attention 权重放在哪里，不直接等于该位置对输出的因果贡献。</p>
      </div>

      <h3 id="attention-scores">3.2 本报告使用的 attention-head scores</h3>
      {metric_cards(attention_metric_rows)}
      <div class="callout warn"><b>三个容易混淆的量。</b><code>k-to-k raw mass</code> 看 matching needle 获得了全部上下文 attention 的多少；<code>diagonal_dominance</code> 只看 needle 子集内部的相对占比；<code>correct_top1</code> 只判断 needle 子集内的排序。因此 diagonal dominance 或 top-1 很高，并不保证 raw mass 很高，剩余 attention 仍可能主要落在 BOS、noise 或 trace 上。</div>
      <div class="protocol"><b>如何从逐 query 值变成 head ranking。</b>Non-thinking broad 与 CoT trace-readout 每个 example 只有一个 final query，所以对 examples 等权平均。Targeted retrieval 对每个 <code>&lt;k&gt;</code> query 产生一行，再直接取均值，因此是<b>query-weighted</b>：count=n 的 example 贡献 n 行。Successor score 只对存在下一 needle 的 <code>M_k</code> query 计算，即 k&lt;n，同样按有效 queries 等权。Figure 2 的三个 count-bin panel 在各自区间内重新平均，但区间内部仍是 query-weighted。</div>

      <h3>3.3 Logit、候选集合、margin 与概率</h3>
      <div class="formula-block"><h4>Logit 与候选集合内的 softmax</h4><div class="equation">{FORMULA_MATHML['candidate softmax']}</div><p><i>z</i><sub>q,t</sub> 是 query 位置 <i>q</i> 对“下一 token 为 <i>t</i>”给出的未归一化分数。候选集合 <i>C</i> 由指标指定；改变 <i>C</i> 会改变概率与 margin 的解释。</p></div>
      <div class="formula-block"><h4>Target logit margin</h4><div class="equation">{FORMULA_MATHML['target logit margin']}</div><p>margin&gt;0 表示正确 token 胜过候选集合内所有竞争者；margin=0 是并列；margin&lt;0 表示至少一个竞争者更高。它测相对证据强度，不是概率，也不自动等于 full-vocabulary accuracy。</p></div>
      <div class="protocol"><b>本报告的三个候选集合。</b><b>Final-count margin</b> 使用 30 个 number tokens；<b>trace-marker margin</b> 使用 10 个 marker tokens；<b>retrieval clean/corrupt patch margin</b> 只比较 clean marker 与替换后的 corrupt marker。最后一种是二元因果 readout，因此数值不能与 10-way marker margin 直接横向比较。</div>
      <div class="formula-block"><h4>Expected count</h4><div class="equation">{FORMULA_MATHML['expected count']}</div><p>先把 logits 限制到 30 个 number tokens 并重新归一化。<i>E</i>[<i>C</i>] 是 logits 分布的连续位置，不保证是整数，也不是一次实际生成的 token。</p></div>

      <h3>3.4 Accuracy 与 sequence-level 指标</h3>
      {table(prediction_metric_rows, [('name','指标'),('unit','平均单位'),('definition','严格定义'),('note','阅读注意')])}
      <div class="callout"><b>为什么 accuracy 与 margin 可能给出不同印象。</b>Final-count accuracy 在 number-token 子集内取 argmax；trace marker/index accuracy 在全词表取 argmax；margin 又只在各自指定候选集合内比较。因此 margin 变小可以发生在 accuracy 尚未翻转之前，而 marker margin 为正也不保证 full-vocabulary argmax 已经是该 marker。</div>

      <h3 id="causal-metrics">3.5 Ablation、patching、steering 与 geometry 指标</h3>
      {metric_cards(causal_metric_rows, context_label='')}
      <div class="formula-block"><h4>Residual 与 count centroid</h4><div class="equation">{FORMULA_MATHML['residual centroid']}</div><p><i>h</i> 表示 query token 经过 Layer ℓ 后的 residual-stream 向量，μ 是独立样本上同一 exact count 的均值。Residual transplant 替换完整 <i>h</i> 或 μ 后只运行剩余 Layers；它不是统计学“预测残差”。</p></div>
      <div class="formula-block"><h4>Adjacent-count steering direction</h4><div class="equation">{FORMULA_MATHML['adjacent-count direction']}</div><div class="equation compact">{FORMULA_MATHML['steering update']}</div><p>若 count manifold 弯曲，不同 δ<sub>c</sub> 会不平行，单一方向 <i>d</i> 即使可读也未必能稳定 steering。</p></div>
    </section>
    """
    dynamics_section = f"""
    <section id="dynamics">
      <h2>4. 学习动态：不同 count 区间何时达到 99%</h2>
      <p>本节先把三条 accuracy 曲线的条件固定下来，再比较学习速度。v10 的 gold count 是 1–30；没有 count=0 样本。所有“区间 accuracy”都先在每个 exact count 内求准确率，再对区间包含的十个（或全范围三十个）exact counts 等权平均，因此不会让某个 count 因样本数较多而占更大权重。</p>

      <div class="metric-grid">
        <article class="metric-card">
          <div class="metric-card-head"><code>TF final-count accuracy</code><div class="metric-context"><span>条件</span>gold prefix / teacher forcing</div></div>
          <p><b>输入什么：</b>给定完整正确前缀。Non-thinking 直接给到 <code>&lt;Ans&gt;</code>；thinking 还会给定完整正确的 indexed trace，再到 <code>&lt;Ans&gt;</code>。</p>
          <p><b>怎么算：</b>读取 <code>&lt;Ans&gt;</code> 位置预测下一 token 的 logits，只在 30 个 count-number tokens 中取 argmax；等于 gold count 记 1。</p>
          <p><b>测量什么：</b>最终答案 readout 是否已经学会。对 thinking 来说，它不要求模型自己生成 trace，因此不是端到端 CoT accuracy。</p>
        </article>
        <article class="metric-card">
          <div class="metric-card-head"><code>AR final-count accuracy</code><div class="metric-context"><span>条件</span>free autoregressive generation</div></div>
          <p><b>输入什么：</b>只提供 prompt 和该模型的格式起点，后续 token 全部由模型逐步生成。</p>
          <p><b>怎么算：</b>解析自由生成结果中的最终 count token；它等于 gold count 记 1，否则记 0。</p>
          <p><b>测量什么：</b>端到端最终答案。Thinking 模型的任何 trace 错误都可能传播到最终 count，因此它同时包含 trace generation 与 answer readout 两个环节。</p>
        </article>
        <article class="metric-card">
          <div class="metric-card-head"><code>AR trace exact match</code><div class="metric-context"><span>条件</span>thinking only / free generation</div></div>
          <p><b>输入什么：</b>与 AR final-count 相同，不提供 gold trace。</p>
          <p><b>怎么算：</b>生成的完整 <code>&lt;1&gt; M1 ... &lt;n&gt; Mn</code> token 序列必须与 gold trace 逐 token 完全相同；任何 index、marker、长度或关闭位置错误都使整例记 0。</p>
          <p><b>测量什么：</b>完整 CoT 路径是否无误。它比逐 token marker accuracy 严格得多；non-thinking 没有 trace，所以该指标不适用。</p>
        </article>
        <article class="metric-card">
          <div class="metric-card-head"><code>balanced accuracy</code><div class="metric-context"><span>汇总</span>exact-count balanced</div></div>
          <p>对每个 gold count <i>c</i>，先计算该 count 内的样本准确率；随后对所选 count 集合中的这些准确率做算术平均。Figure 1 使用 1–30 三十个 count；Figure 2 分别使用 1–10、11–20、21–30。</p>
          <p>TF 每 500 steps 评估；更昂贵的 AR 与 trace exact 每 1000 steps 评估。因此“首次达到 99%”只精确到对应评估网格，不能解释成连续时间中的精确 crossing。</p>
        </article>
      </div>

      {figure(
          generated['training_overall'],
          'Figure 1A. 全 count 1–30：两个模型的 accuracy 与 loss 学习动态',
          '<b>左图：</b>横轴是 training step，纵轴是对 gold count 1–30 等权平均的 exact-match accuracy；颜色区分模型（蓝=non-thinking，橙=thinking），实线是 TF final count，虚线是 AR final count，绿色点线是 thinking 的 AR trace exact，灰色虚线为 99%。<b>右图：</b>横轴仍是 step；纵轴是 held-out evaluation token cross-entropy，使用对数尺度。蓝/橙分别表示两个模型，实线 final count loss，虚线 total loss；绿色与紫色分别是 thinking trace 的 marker 与 index loss。右图比较的是不同被监督 token segment 的平均交叉熵，不是 accuracy。'
      )}

      <div class="callout good"><b>全范围 99% milestone。</b>Thinking 的 TF final-count 在 step <b>{overall_steps['thinking_tf']}</b> 首次达到 99%，但 AR final-count 与完整 AR trace 分别到 step <b>{overall_steps['thinking_ar']}</b> 和 <b>{overall_steps['thinking_trace']}</b> 才达到 99%。Non-thinking 的 TF/AR final-count 分别在 step <b>{overall_steps['direct_tf']}</b> / <b>{overall_steps['direct_ar']}</b> 达到 99%。这正是“thinking 的 count readout 学得很快，但完整 trace 学得更慢”的定量版本。</div>

      {figure(
          generated['training_bins'],
          'Figure 1B. 两个模型在 1–10、11–20、21–30 三个 count 区间的学习动态',
          '<b>三个 panel</b>依次固定 gold count 1–10、11–20、21–30；每个 panel 都同时画两个模型，线型和颜色与 Figure 1 左图完全相同。横轴是 checkpoint step，纵轴是该区间十个 exact counts 的平衡 accuracy，灰色虚线为 99%。这张图用于判断总体 milestone 是否由简单 count 主导，以及高 count trace 是否构成更长时间的瓶颈。'
      )}

      <h3>4.1 首次达到 99% 的离散 checkpoint</h3>
      {table(
          milestone_rows,
          [
              ('mode','模型'),
              ('bin','gold count 范围'),
              ('tf','TF final-count 首次 ≥99%'),
              ('ar','AR final-count 首次 ≥99%'),
              ('trace','AR trace exact 首次 ≥99%'),
          ],
      )}

      <h3>4.2 为什么 thinking 的 final count 先学会，而完整 trace 更慢？</h3>
      <div class="protocol">
        <ol>
          <li><b>Teacher forcing 把正确 trace 直接放在答案前面。</b>Thinking 的 TF final-count 只需要把已经正确写出的 trace 汇总为一个 count token；它无需先解决第 k 个 needle 的检索、marker identity、successor index 和停止位置。因此这是较短的 readout 子问题。</li>
          <li><b>最终答案只有一个监督 token，trace 却有 O(n) 个必须连续正确的 token。</b>Count=30 时，indexed trace 包含 30 个 index 与 30 个 marker。即使单步正确率已经很高，sequence exact match 仍会把任一步错误放大为整例失败，所以 trace exact 天然晚于局部 token loss 和 TF final-count。</li>
          <li><b>实际 loss 分解显示瓶颈在 trace marker，而不是最终 count。</b>Step 500 时，thinking 的 final-count loss 已降到 <b>{loss_at(500, 'thinking', 'final_count')}</b>，但 trace-marker loss 仍为 <b>{loss_at(500, 'thinking', 'trace_marker')}</b>，trace-index loss 为 <b>{loss_at(500, 'thinking', 'trace_index')}</b>。也就是说模型已经能从 gold trace 读出 count，但还不能稳定完成每一步 k-to-k marker retrieval。</li>
          <li><b>AR final count 随 trace 一起改善，说明早期端到端瓶颈确实位于 trace。</b>当 gold trace 不再提供时，错误会在自回归过程中累积；因此 thinking 的 AR final-count 曲线远晚于 TF 曲线，并与 trace-exact 曲线大体同步。两者后期可能略有分离，因为错误 trace 偶尔仍能产生正确最终 count，反之亦然。</li>
          <li><b>这不等于 CoT “计算更慢”。</b>这里比较的是优化样本复杂度与严格 sequence 指标：thinking 多学习了一个长度随 count 增长的结构化输出。后文的 attention、ablation 与 patching 才判断这段 trace 是否形成了不同的计算路径，而不是只增加了训练 token。</li>
        </ol>
      </div>
      <div class="callout warn"><b>解释边界。</b>TF final-count 的快速饱和可以部分来自局部 readout 或位置/trace-length 线索，不能单独证明模型已经完成可靠 targeted retrieval。反过来，trace exact 较慢也受到“全序列零容错”定义影响，不能直接等价为每个 trace token 都很差。必须同时阅读 token-level loss、AR final-count 和后续因果实验。</div>
    </section>
    """
    attention_section = f"""
    <section id="attention">
      <h2>5. 描述性 attention：broad aggregation 与 targeted retrieval</h2>
      <p>本节只问 attention 权重“放在哪里”，用于提出候选 heads；它不把 attention weight 直接解释成因果贡献。所有热图都从逐 example、逐 query、逐 Layer×head 的 attention row 重新按 gold count 分成 1–10、11–20、21–30。横轴始终是 head 0–3，纵轴始终是 Layer 1–4。</p>

      <h3>5.1 两个模型在最终答案 query 上的 broad prompt aggregation</h3>
      <div class="protocol"><b>共同 query。</b>对两个模型都读取 <code>&lt;Ans&gt;</code> 位置的 attention row。<b>共同 token 集合。</b>只把 prompt 中真实 needle 的位置记作集合 N。<b>Broad score。</b><code>prompt_needles_mass × needle_entropy_normalized</code>：第一项要求 head 确实把较多权重放到 needles，第二项要求这些权重在多个 needles 间分散。得分接近 0 可能因为忽略 needles，也可能因为只盯住少数 needles；因此表中同时报告 mass、归一化 entropy 与 effective number。</div>
      {figure(
          generated['broad_attention'],
          'Figure 2A. Non-thinking 与 CoT 的 broad prompt-needle score',
          '<b>布局：</b>上排 non-thinking，下排 thinking；三列依次为 gold count 1–10、11–20、21–30。<b>单元格：</b>该 Layer×head 在最终 <code>&lt;Ans&gt;</code> query 上的平均 broad score，范围 0–1；所有 panel 使用同一色标。高分表示 attention mass 同时较多地落在 prompt needles 上，并在该区间的多个 needles 间较均匀。它不包含 trace marker，也不证明这些 heads 对输出必要。'
      )}
      {table(
          broad_head_rows,
          [
              ('mode','模型'),('bin','count 区间'),('head','该区间最高 broad-score head'),
              ('score','broad score'),('mass','prompt-needle mass'),
              ('entropy','needle entropy / log n'),('effective','effective # needles'),
          ],
      )}
      <div class="callout good"><b>描述性差异。</b>Non-thinking 的最佳 broad head 在三个区间都稳定为 <code>L1H3</code>；随着 count 增大，它的 needle entropy 接近 1，effective number 接近该区间的真实 needle 数，符合“一个最终 query 并行覆盖整个 needle 集合”的候选图景。CoT 的最高 broad score 会从低 count 的中层转到高 count 的后层，但最佳 head 的 effective number 只有约 2–3：它虽然把不少 mass 放到 prompt needles，却没有均匀覆盖全部 needles。因此 CoT 最终答案处没有出现与 non-thinking 同样清楚的 prompt-wide broad aggregator；这与“先生成 trace，再从 trace/readout 状态得到 count”相容。</div>

      <h3>5.2 Targeted k-to-k retrieval：为什么 non-thinking 不能直接套同一指标</h3>
      <div class="protocol"><b>严格 k-to-k 的前提。</b>CoT trace 中存在第 k 个数字 query <code>&lt;k&gt;</code>，所以能把它与 prompt 中按位置排序的第 k 个 needle 唯一配对，并计算 raw k-to-k mass、needle 子集内 diagonal dominance 与 correct top-1。<b>Non-thinking 的结构差异。</b>它没有 trace、没有 k，也没有三十个逐步 query；只有一个最终 <code>&lt;Ans&gt;</code> query。因此 non-thinking 的天然 k-to-k score 是<b>不适用</b>，不能把某个任意 needle 宣称为“正确第 k 个目标”。作为最接近的负对照，Figure 2B 第一排计算 <code>prompt_needles_mass × (1 − needle_entropy_normalized)</code>，只问最终 query 是否集中到少数 needles；这个 concentration score 不是 k-to-k。</div>
      {figure(
          generated['targeted_attention'],
          'Figure 2B. CoT k-to-k retrieval、diagonal dominance 与 non-thinking concentration control',
          '<b>三列</b>依次为 count 1–10、11–20、21–30。<b>第一排</b>是 non-thinking 最终 query 的 single-target concentration control；它越高，表示 needle mass 越集中，但没有 k 对齐。<b>第二排</b>是 CoT <code>&lt;k&gt;</code> query 给 matching 第 k 个 prompt needle 的 raw attention mass。<b>第三排</b>是 diagonal dominance：matching mass 除以投向全部 prompt needles 的总 mass。<b>第四排</b>是 correct top-1：只在 prompt-needle 子集内判断最大权重是否落在第 k 个 needle。四排均按有效 query 等权平均，count=n 的样本贡献 n 个 k queries；每个 count 区间单独汇总。'
      )}
      <h3>5.3 最强候选 head 的分区结果</h3>
      {table(
          nonthinking_concentration_rows,
          [
              ('bin','count 区间'),('head','non-thinking 最集中 head'),
              ('score','concentration score'),('mass','prompt-needle mass'),
              ('entropy','needle entropy / log n'),('effective','effective # needles'),
          ],
      )}
      {table(
          targeted_head_rows,
          [
              ('bin','count 区间'),('head','CoT 最高 raw k-to-k head'),
              ('raw','raw k-to-k mass'),('diag','diagonal dominance'),
              ('top1','correct top-1'),('needle_mass','该 head 的总 prompt-needle mass'),
          ],
      )}
      <div class="callout good"><b>CoT 的主要现象。</b><code>L4H2</code> 在三个 count 区间都是最强 raw k-to-k head。随着候选 needle 从 1–10 增到 21–30，raw mass、diagonal dominance 与 correct top-1 同时下降：这不是“总 needle mass 被其他上下文吸走”一种原因可以完全解释，而是 matching needle 在 needle 子集内部的优势也变弱。相反，non-thinking 的 concentration score 很小，且没有 k-specific query；当前描述性证据更支持 broad set aggregation，而不是隐式逐 k 检索。</div>
      <div class="callout warn"><b>证据边界。</b>CoT 的 high diagonal dominance 只说明“在已经投给 needles 的权重中，matching needle 占多少”；raw mass 才说明它获得全部上下文 attention 的多少。Non-thinking 缺少 k-to-k 并不证明其内部绝无串行步骤，只说明当前可观察 token interface 不提供与 CoT 相同的逐 k 对齐。第 7、8 节的 ablation 与 patching 才测试这些候选 heads 是否必要或局部充分。</div>
    </section>
    """

    geometry_section = f"""
    <section id="geometry">
      <h2>6. Hidden-state 描述性现象：count manifold 与 CoT trace 动态</h2>
      <p>Attention 描述 query 从哪些 token 读取；本节改看读取之后写入 residual stream 的 256 维状态。这里所有 PCA 都先按语义标签求 hidden-state 均值，再对这些均值做 PCA，因此展示的是 count/progress 类中心之间的几何，而不是单个样本云。它仍是描述性证据：连续轨迹表示 count 或 progress 可读，不等于模型必然沿该轨迹执行加法。</p>

      <h3>6.1 最终答案读取位置的 exact-count centroid manifold</h3>
      <p><b>本实验在问：</b>最终答案位置的 hidden state 为什么能区分 count？它可能真的从 prompt needles 得到了 cardinality，也可能只利用了自然 CoT 中“trace 有多长”或 <code>&lt;Ans&gt;</code> 的绝对位置。为拆开这两种来源，本节同时比较自然输入与一个明确的反事实固定-trace control。</p>
      <div class="protocol"><b>首先明确提取位置。</b>三种条件都在 <code>&lt;Ans&gt;</code> token 处读取 residual，而不是在答案数字 <code>&lt;n&gt;</code> 处读取。由于 decoder 是因果的，<code>&lt;Ans&gt;</code> 的 state 只看得到它左侧的 prompt/trace；正确答案数字是它要预测的<b>下一个 token</b>，此时尚未进入上下文。对每个样本、每个 Layer，保存该 Layer 输出后、进入下一 Layer 前的 256 维 residual <code>h_ans(layer)</code>。</div>
      {table(
          geometry_sites,
          [('site','语义位置 / 条件'),('sequence','实际送入模型的序列'),('ans','<Ans> 的 0-based 位置'),('role','哪些量会变化，以及为什么比较')],
      )}
      <div class="callout warn"><b>“固定 15 步 trace”不是 gold trace，也不是正常 CoT。</b>它由固定模板构造：无论 prompt 中真实有 n 个 needles，都写入 <code>&lt;1&gt; T1 ... &lt;15&gt; T15</code>；其中 <code>Tk</code> 按固定 marker 词表模板循环，与该 prompt 中第 k 个 needle 的真实 marker 无关。最终监督答案仍是 prompt 的真实 count <code>&lt;n&gt;</code>。因此当 n≠15 时，trace 暗示的长度与答案标签有意冲突。这是一个 teacher-forced、分布外的诊断输入，不是模型的自然生成结果。</div>
      <div class="protocol"><b>计算流程。</b><ol>
        <li>生成一个长度 {prompt_length}、真实 needle count 为 n∈[1,30] 的 held-out prompt。</li>
        <li>分别渲染上表三种前缀；反事实条件对所有 n 都使用 {fixed_trace_steps} 步、30 个 trace tokens，并把 <code>&lt;Ans&gt;</code> 固定在位置 {fixed_ans_position}。</li>
        <li>做一次 teacher-forced forward，在每个 Layer 后提取 <code>&lt;Ans&gt;</code> residual。答案 token 位于其右侧，不能泄漏给该 state。</li>
        <li>对同一 exact count 的样本先在 256 维空间求均值，得到每个 site×Layer 的 30×256 centroid 矩阵。这样图中的每个点是一类 count 的均值，不是一个单独样本。</li>
        <li>每个 site×Layer 独立减去 30 个 centroids 的总均值，再做 SVD/PCA。图只描述 exact-count 类中心之间的几何；不同 panel 的 PC 轴不是同一坐标系。</li>
      </ol></div>
      <div class="callout good"><b>这个 control 如何判读。</b>如果自然 CoT 有清楚的 count 轨迹、但固定-trace 条件不再按 n 分离，说明自然几何可能主要来自 trace 长度或绝对位置。若固定-trace 条件仍按真实 prompt count 排列，则在 trace token、trace 长度和 <code>&lt;Ans&gt;</code> 位置都相同的情况下，state 中仍存在 prompt-derived count 信息。</div>
      {figure(
          generated['pca_static'],
          'Figure 3A. Exact-count mean residual 的 PC1–PC2 渐变轨迹',
          '每个点是一个 exact count 的 256 维 residual 均值；颜色从 count 1 连续渐变到 30，灰线连接相邻 count。三行分别为 non-thinking natural final、CoT natural final、CoT counterfactual fixed-15 final，四列为 Layer 1–4。横轴与纵轴是该 panel 独立拟合的 PC1/PC2，单位为投影坐标；只能在同一 panel 内判断相邻 count 是否有序、轨迹是否弯曲，不能跨 panel 把屏幕方向当成同一神经方向。'
      )}
      <h4>PC1–PC6 explained variance 到底在做什么</h4>
      <div class="protocol"><ol>
        <li><b>输入不是单样本 hidden states。</b>对每个 site×Layer，先把相同语义标签的 256 维 residual 求均值。final-answer site 按真实 count <code>n=1,...,30</code> 分组；trace site 按进度 <code>k=1,...,30</code> 分组。于是 final site 得到一个 <code>30×256</code> 的类中心矩阵。</li>
        <li><b>只分析类中心之间的方差。</b>从每个类中心减去 30 个类中心的总均值，对中心化矩阵做 SVD/PCA。若第 r 个主成分的特征值为 <code>lambda_r</code>，则 <code>EVR_r = lambda_r / sum_s(lambda_s)</code>；PC1–PC6 cumulative 是前六个 EVR 的和。</li>
        <li><b>问题是什么。</b>PC1 很高表示不同 count 的均值主要沿一条轴排列；PC1–PC6 很高表示六维子空间足以保留绝大多数 count-class geometry。它不测同一 count 内样本的离散程度，也不意味着每个 256 维 residual 真的只有六维。</li>
        <li><b>每个 panel 独立拟合。</b>不同 site 或不同 Layer 的 PC1 不是同一神经方向，因此可以比较 explained-variance ratio，却不能把两个 panel 的横轴直接视为同一向量。</li>
      </ol></div>
      {figure(
          generated['pca_variance'],
          'Figure 3B. PC1–PC6 explained variance 与累计覆盖',
          '<b>横轴：</b>单独的 PC1–PC6；<b>纵轴：</b>Layer 1–4。前五个 panel 分别对应 non-thinking final、CoT natural final、CoT fixed-15 final、CoT trace index 与 CoT trace marker；单元格为该 PC 对类中心总方差的解释比例。最右 panel 的每个单元格是同一 site×Layer 下 PC1 到 PC6 的累计解释率。final-answer sites 的语义标签是 gold count，trace sites 的标签是 progress k。颜色越亮表示方差越集中在该轴或前六轴，但不衡量 256 维空间里的绝对半径。'
      )}
      {table(
          pca_rows,
          [('site','site'),('layer','Layer'),('pc1','PC1 variance'),('pc3','PC1–3 cumulative'),('pc6','PC1–6 cumulative'),('dim','effective dimension'),('turn','mean adjacent turn')],
      )}
      <h4>结果：Layer 1 近似低维，深层把 count geometry 分散到更多方向</h4>
      {table(
          depth_geometry_rows,
          [('site','site'),('layer','Layer'),('pc1','PC1 variance'),('pc6','PC1–6 cumulative'),('dim','effective dimension'),('radius','256D RMS centroid radius'),('adjacent','mean adjacent-count distance'),('turn','mean adjacent turn')],
      )}
      <div class="protocol"><b>为什么还要报告原空间半径。</b>Explained variance 是比例量：即使点云整体放大或缩小，只要形状不变，EVR 也不变。为避免把二维图上的“挤在一起”误写成 256 维坍缩，表中另外计算 <code>R_rms = sqrt(mean_c ||mu_c - mu_bar||^2)</code>，即类中心在原始 256 维 residual 空间相对总均值的均方根半径；<code>mean adjacent-count distance</code> 是相邻 count 类中心 <code>mu_(c+1)</code> 与 <code>mu_c</code> 的欧氏距离均值。二者都使用 PCA 前的原始坐标。</div>
      <div class="callout good"><b>三个关键数值模式。</b><ol>
        <li><b>Non-thinking final：</b>PC1 从 Layer 1 的 <b>{pct(direct_geo_l1.pc1_variance)}</b> 降到 Layer 4 的 <b>{pct(direct_geo_l4.pc1_variance)}</b>，PC1–PC6 从 <b>{pct(direct_geo_l1.pc6_cumulative)}</b> 降到 <b>{pct(direct_geo_l4.pc6_cumulative)}</b>，有效维数从 <b>{fmt(direct_geo_l1.effective_dimension)}</b> 升到 <b>{fmt(direct_geo_l4.effective_dimension)}</b>。原空间半径只从 <b>{fmt(direct_spread_l1.rms_centroid_radius, 2)}</b> 降到 <b>{fmt(direct_spread_l4.rms_centroid_radius, 2)}</b>，而相邻 count 距离反而从 <b>{fmt(direct_spread_l1.mean_adjacent_distance, 2)}</b> 增至 <b>{fmt(direct_spread_l4.mean_adjacent_distance, 2)}</b>。所以深层不是简单把所有 count 压成一点，而是把一条平滑轴折叠成更弯曲、更高维的决策几何。</li>
        <li><b>CoT natural final：</b>PC1 从 <b>{pct(natural_geo_l1.pc1_variance)}</b> 降到 <b>{pct(natural_geo_l4.pc1_variance)}</b>，PC1–PC6 仅剩 <b>{pct(natural_geo_l4.pc6_cumulative)}</b>；但原空间半径从 <b>{fmt(natural_spread_l1.rms_centroid_radius, 2)}</b> 增至 <b>{fmt(natural_spread_l4.rms_centroid_radius, 2)}</b>。这里“深层点在 PC1/PC2 图里紧缩”完全是低维投影遗漏了大量高阶方向，并非 256 维表示真的变小。</li>
        <li><b>CoT fixed-15 control：</b>PC1 在 Layer 1 解释 <b>{pct(fixed_geo_l1.pc1_variance)}</b>，到 Layer 4 仍解释 <b>{pct(fixed_geo_l4.pc1_variance)}</b>；Layer 4 的 PC1–PC6 覆盖 <b>{pct(fixed_geo_l4.pc6_cumulative)}</b>，有效维数仅 <b>{fmt(fixed_geo_l4.effective_dimension)}</b>。固定 trace 内容、长度和 <code>&lt;Ans&gt;</code> 位置后，prompt-derived count 仍呈现近低维结构，说明自然 CoT 深层的高维化很大部分来自 count 与 trace/位置/readout 因素的混合。</li>
      </ol></div>
      <div class="callout warn"><b>为什么 Layer 1 最清楚，而越深越像“紧缩”。</b>最保守的解释是：Layer 1 离 token embedding 与 learned absolute-position embedding 最近，跨样本平均后，marker 数量或 trace progress 形成一个占主导的平滑全局特征，所以一个或少数 PC 就能解释大部分类中心差异。后续 Layers 为完成具体预测，会把 count 与 marker identity、prompt 内容、trace token role、绝对位置以及最终 logit readout 混合，并通过非线性 MLP、attention 与 residual update 弯曲这条轨迹；方差因此分散到 PC3 以后，PC1/PC2 投影中的点看起来靠拢。fixed-15 control 保留低维，而自然 CoT final 高维化，支持“nuisance/readout 因素混合”这一解释；但这仍是由几何模式作出的推断，不是某个 Layer 运算的直接因果证明。</div>
      <div class="callout limit"><b>但“可读”不等于“模型因果上使用”。</b>固定-trace 条件本身是训练分布外冲突输入；PCA 分离也只说明 n 可由 residual 解码。第 10 节的 Layer-4 count-centroid transplant 更严格：其 transport slope 在 1–10、11–20、21–30 分别为 <b>{fixed_transport_l4_display['1-10']}</b>、<b>{fixed_transport_l4_display['11-20']}</b>、<b>{fixed_transport_l4_display['21-30']}</b>，低/中 count 明显没有一比一搬运输出。因此当前证据支持“prompt count 信息仍存在”，但不支持“这条低维轴就是统一执行加减法的主因果变量”。</div>
      {interactive_pca(coordinates, geometry)}
      <div class="callout"><b>五个下拉项如何区分。</b><ul>
        <li><b>Non-thinking / CoT natural final-answer query：</b>都在自然序列的 <code>&lt;Ans&gt;</code> token 读取 residual，并按真实 count <i>n</i> 求 centroid。</li>
        <li><b>CoT fixed-15 counterfactual &lt;Ans&gt; query：</b>把所有样本左侧 trace 都固定成同一条 15-step 模板，使 trace 内容、trace 长度和 <code>&lt;Ans&gt;</code> 绝对位置不再随 <i>n</i> 改变；最终标签仍是 prompt 的真实 count。它用来排除“仅靠 trace 长度或答案位置编码 count”的解释，绝不表示模型要预测 15。</li>
        <li><b>CoT trace index-token state &lt;k&gt;：</b>在 trace 数字 <code>&lt;k&gt;</code> 位置读取 residual；点标签是 progress <i>k</i>。</li>
        <li><b>CoT trace marker state M<sub>k</sub>：</b>在紧跟数字后的 marker <code>M_k</code> 位置读取 residual；点标签同样是 progress <i>k</i>，不是 marker identity。</li>
      </ul><b>与下一张“CoT trace 内部 hidden state”图的关系：</b>原始 hidden states 与 token anchors 是同一批。当前下拉图把 <code>&lt;k&gt;</code> 和 <code>M_k</code> 当成两个 semantic sites，分别拟合各自 PCA，适合看“单一 token role 内是否编码 progress”；第 6.2 节则把两类共 60 个 centroids 放进同一个 Layer-specific PCA 基底，适合直接看 <code>&lt;k&gt; → M_k → &lt;k+1&gt;</code> 的状态移动。两图不是两个不同实验，而是同一数据的两种几何视角。</div>
      <div class="callout"><b>轴选择。</b>互动图包含 PC1–PC6 中任选三轴的全部 <b>20</b> 种组合。切换 count/progress range 只筛选同一个全 1–30 PCA 基底中的点，不会重新拟合；切换 site 或 Layer 则会改变 PCA 基底，所以不同面板的屏幕方向不能直接比较。</div>

      <h3>6.2 CoT trace 内部：从 &lt;k&gt; 到 M<sub>k</sub>、再到 &lt;k+1&gt; 的均值状态如何移动</h3>
      <div class="protocol"><b>提取什么。</b>对所有包含第 k 步的 held-out CoT examples，在数字 <code>&lt;k&gt;</code> 与紧随其后的 marker <code>M_k</code> 位置分别提取 Layer 1–4 后的 residual。<b>如何平均。</b>先按 <code>(token type, k)</code> 求均值，因此每层有 60 个 256 维 centroids。<b>如何 PCA。</b>与把 index 和 marker 分开拟合不同，本图把 <code>&lt;1&gt;,M1,...,&lt;30&gt;,M30</code> 的 60 个均值放在同一个 Layer-specific PCA 基底中，才能直接观察 token type 交替与 progress 更新。图中不含 <code>&lt;Think&gt;</code> 和 <code>&lt;/Think&gt;</code>。</div>
      {table(
          trace_pca_rows,
          [('layer','Layer'),('pc1','PC1 variance'),('pc3','PC1–3 cumulative'),('pc6','PC1–6 cumulative'),('dim','effective dimension')],
      )}
      {interactive_trace_pca(trace_coordinates, trace_geometry)}
      <div class="callout warn"><b>如何解释轨迹。</b>若同一 k 的 index 与 marker 形成短的成对位移，而相邻 k 对之间沿较平滑方向推进，说明 residual 同时分离“当前 token role”与“progress k”。若轨迹强烈弯曲或不同阶段折返，则不支持一根全局固定的 +1 direction。无论哪种形状，都还需要后续 residual patch/steering 才能从“可见几何”升级为“因果 count state”。</div>
    </section>
    """

    ablation_section = f"""
    <section id="ablation">
      <h2>7. Attention-head ablation：broad aggregation 与 targeted retrieval 是否因果必要</h2>
      <p>第 5 节只根据 attention 权重提出候选机制。本节进行必要性检验：保持输入、参数、MLP 和其他 heads 不变，只移除指定 attention head 的贡献，再观察模型已经学会的行为是否下降。分析分成四个问题：non-thinking broad heads 是否支撑最终计数；CoT targeted heads 是否支撑逐步 marker retrieval；CoT trace-readout heads 是否支撑最终答案；描述性高分与因果必要性是否真的一致。</p>

      <h3>7.1 实验协议：删掉什么、在哪些 token 上测、accuracy 如何计算</h3>
      <div class="protocol"><ol>
        <li><b>评估样本。</b>重新生成每个 exact count {manifest['ablation_examples_per_exact_count']} 个 held-out prompts，共 {30 * manifest['ablation_examples_per_exact_count']} 个。每个区间 1–10、11–20、21–30 各含 {10 * manifest['ablation_examples_per_exact_count']} 个样本，因此三栏的样本量与类别数完全相同。</li>
        <li><b>全局 head mask。</b>对 Layer <code>l</code> 的 head <code>h</code>，把 GPT-2 forward 中对应 <code>head_mask[l,h]</code> 设为 0。该 head 在整条 sequence 的<b>所有 query positions</b>都被关闭，而不是只在 <code>&lt;Ans&gt;</code> 或某一个 <code>&lt;k&gt;</code> 关闭；其他 heads、MLP、embedding 和模型参数不变。因而实验能判断“这个 head 是否整体必要”，但不能单独定位它在哪个 query 上发挥作用。</li>
        <li><b>单头消融。</b>16 个 Layer×head 分别单独 mask。报告 <code>accuracy drop = baseline accuracy − masked accuracy</code>；drop=1 表示从 100% 降到 0%，drop=0 表示离散 accuracy 没变。</li>
        <li><b>累计消融。</b>先按第 5 节定义的分数给 heads 排名，再依次 mask top-1、top-2、…、top-16。Non-thinking 使用 <code>broad_attention_score</code>；CoT 分别使用 <code>k-to-k raw mass</code>、<code>trace_markers_mass</code> 与 successor score。对照只使用预先固定的随机 head 删除路径，不再使用低分或倒序排名。</li>
        <li><b>随机对照与 random-4 mean。</b>淡灰细线逐条显示每个固定随机删除顺序；深灰线是在同一个 mask 数 <code>n</code> 上对所有随机路径取均值，灰色阴影是随机路径的 min–max。紫色水平虚线 <code>random-4 mean</code> 是随机路径在“恰好删 4 个 heads”时剩余 accuracy 的均值，并横向延伸作为易读参照；它不是一条随 <code>n</code> 变化的额外实验曲线。只有 ranked-top 曲线稳定早于随机路径分布下降，才能说明描述性排名含有超出随机删除的机制信息。</li>
        <li><b>Teacher-forced final-count accuracy。</b>完整 gold prefix 已在输入中；在 <code>&lt;Ans&gt;</code> 位置读取 logits，若 argmax 数字 token 等于 gold count 则正确。CoT 中 gold trace 仍存在于左侧，所以该指标检验“给定正确 trace 后的最终 readout”，不是自由生成整条 trace 的端到端准确率。</li>
        <li><b>Teacher-forced trace-marker accuracy。</b>对 CoT 的每个数字 <code>&lt;k&gt;</code> query，检查其 next-token argmax 是否为 gold marker <code>M_k</code>，再对所有有效 k queries 求平均。后一步仍看到数据中的 gold 早期 trace token；某一步预测错不会被写回上下文并污染后续步骤。</li>
        <li><b>Teacher-forced trace-index accuracy。</b>在 <code>&lt;Think&gt;</code> 或前一个 marker <code>M_(k-1)</code> 的 query 上，检查 next-token argmax 是否为数字 <code>&lt;k&gt;</code>。它测 successor/index 生成，与 marker identity retrieval 是不同子任务。</li>
      </ol></div>
      <div class="callout warn"><b>必要性不等于充分性。</b>Mask 后性能下降说明该组件在当前网络中被使用；不下降可能表示它无关，也可能表示其他 heads 冗余补偿。反过来，单头 attention mass 很高也不保证删除它会伤性能。第 8 节的 clean-to-corrupt patching 才检验候选 activation 是否能局部搬运所需信息。</div>

      <p><b>阅读顺序。</b>下面每个小节都严格按照“实验 → 结果 → 分析”组织。实验段只说明干预和指标；结果段先给图表与观测数值；分析段才讨论这些结果支持或排除什么机制。</p>

      <h3>7.2 结果组织方式：每种机制独立报告</h3>
      <p>为避免把不同 query、不同行为指标和不同机制排名混在同一张大图里，后续三节分别报告 non-thinking broad aggregation、CoT targeted retrieval 和 CoT trace readout。每节先给单头热图，再给对应的累计 top-n 剂量曲线；所有 panel 都显式标出 head/Layer 或 mask 数/accuracy 坐标。</p>

      <h3>7.3 Non-thinking：early broad heads 对直接 final count 具有强必要性</h3>
      <h4>实验</h4>
      <p>Non-thinking 没有逐 k trace。我们按第 5 节定义的 <code>broad_attention_score</code> 排序 heads，先逐头 mask，再从 top-1 开始累计 mask；在每个 count 区间都以 non-thinking final-count accuracy 为结果变量，并与固定随机删除路径、随机均值和 random-4 mean 比较。</p>
      <h4>结果</h4>
      {figure(
          generated['single_nonthinking'],
          'Figure 4A. Non-thinking 单头 broad-circuit 必要性',
          '<b>横轴</b>是 head index 0–3；<b>纵轴</b>是 Layer 1–4；三栏分别是 count 1–10、11–20、21–30。颜色和格内数字都是 final-count accuracy drop = baseline accuracy − 单头 global mask 后 accuracy。'
      )}
      {figure(
          generated['cumulative_nonthinking'],
          'Figure 4B. Non-thinking broad heads 的累计消融剂量曲线',
          '<b>横轴</b>是累计 global mask 的 head 数，刻度为 1、2、4、8、12、16；<b>纵轴</b>是干预后剩余的绝对 final-count accuracy，范围 0–1。三栏对应三个 count 区间。蓝线按 broad score 从高到低删除；淡灰细线是各条固定随机路径，深灰线是逐步随机均值，灰色阴影是随机 min–max；紫色虚线是恰好随机删 4 个 heads 时的平均 accuracy（random-4 mean）；黑色虚线是无干预 baseline。'
      )}
      {table(ablation_rows,[('bin','count 区间'),('direct_head','最强单头'),('direct_drop','final-count drop')])}
      {table(cumulative_ablation_rows,[('bin','count 区间'),('direct_top1','broad top-1 后 final accuracy'),('direct_top2','broad top-2 后 final accuracy'),('direct_random4','随机删 4 heads 的平均 final accuracy')])}
      <div class="callout good">描述性排名稳定的 <code>L1H3</code> 同时是中高 count 最强的单头因果组件：11–20 的最大 final-accuracy drop 为 <b>{ablation_lookup['11-20']['direct_drop']}</b>，21–30 为 <b>{ablation_lookup['21-30']['direct_drop']}</b>。1–10 的最强单头为 <code>{ablation_lookup['1-10']['direct_head']}</code>，drop 为 <b>{ablation_lookup['1-10']['direct_drop']}</b>。按 broad score 累计 mask top-1 后，三个区间剩余 accuracy 为 <b>{cumulative_ablation_lookup['1-10']['direct_top1']}</b> / <b>{cumulative_ablation_lookup['11-20']['direct_top1']}</b> / <b>{cumulative_ablation_lookup['21-30']['direct_top1']}</b>；累计到 top-2 后为 <b>{cumulative_ablation_lookup['1-10']['direct_top2']}</b> / <b>{cumulative_ablation_lookup['11-20']['direct_top2']}</b> / <b>{cumulative_ablation_lookup['21-30']['direct_top2']}</b>。</div>
      <h4>分析</h4>
      <p>结果支持 Layer 1 broad routing 是 direct counting 的必要入口：同一批早期 heads 既在描述性 attention 中广泛覆盖 prompt needles，又在删除后造成大幅 final-count 损伤。低 count 的最强单头与中高 count 略有差异，说明容易样本有更多可替代路径；但 top-2 累计删除后三段都接近崩溃，说明 broad circuit 整体不是可有可无的伴随现象。</p>

      <h3>7.4 CoT targeted retrieval：单头可替代，但成组删除破坏 marker trace</h3>
      <h4>实验</h4>
      <p>在 CoT 的数字 <code>&lt;k&gt;</code> query 上，以第 k 个 prompt needle 的 raw attention mass 排名 targeted heads。逐头或累计 mask 后，首先测 teacher-forced trace-marker accuracy；同时测最终 teacher-forced final-count accuracy，用于区分“局部 marker retrieval 已损坏”与“给定 gold trace 后的最终 readout 已损坏”。</p>
      <h4>结果</h4>
      {figure(
          generated['single_targeted'],
          'Figure 4C. CoT 单头消融对 trace-marker retrieval 的影响',
          '<b>横轴</b>是 head index 0–3；<b>纵轴</b>是 Layer 1–4；三栏为三个 count 区间。颜色和格内数字是 teacher-forced trace-marker accuracy drop。它衡量整枚 head 被关闭后，数字 <code>&lt;k&gt;</code> 后正确 marker <code>M_k</code> 的预测损失。'
      )}
      {figure(
          generated['cumulative_targeted_trace'],
          'Figure 4D. CoT targeted heads 的累计消融：局部 trace-marker accuracy',
          '<b>横轴</b>是累计 global mask 的 targeted heads 数；<b>纵轴</b>是剩余 teacher-forced trace-marker accuracy。三栏为三个 count 区间。蓝线是 targeted-score ranked top；淡灰细线是各条固定随机删除路径，深灰线是逐步随机均值，灰色阴影是随机 min–max；紫色虚线是 random-4 mean，黑色虚线是无干预 baseline。'
      )}
      {figure(
          generated['cumulative_targeted_final'],
          'Figure 4E. 同一 targeted-head 消融对最终 count readout 的影响',
          '<b>横轴</b>仍是累计 global mask 的 targeted heads 数；<b>纵轴</b>改为给定 gold trace 后的 teacher-forced final-count accuracy。蓝线与随机对照的定义同 Figure 4D；紫色虚线是随机删 4 个 heads 时的平均 final accuracy。与 Figure 4D 对照可区分“trace marker 已生成失败”与“最终 readout 也失败”。'
      )}
      {table(ablation_rows,[('bin','count 区间'),('trace_head','最强 trace 单头'),('trace_drop','trace-marker drop')])}
      {table(cumulative_ablation_rows,[('bin','count 区间'),('target_trace_top1','target top-1 后 trace'),('target_trace_top4','top-4 后 trace'),('target_trace_top8','top-8 后 trace'),('target_trace_random4','随机删 4 heads 的平均 trace'),('target_final_top8','同一 top-8 后 final'),('target_final_random4','随机删 4 heads 的平均 final')])}
      <div class="callout good">第 5 节 raw k-to-k mass 最高的是 <code>L4H2</code>，其次包括 <code>L3H1</code> 和 <code>L3H0</code>；但三个区间中，单头 mask 后 trace-marker drop 最大的都是 <code>{ablation_lookup['1-10']['trace_head']}</code> 一类早期 heads，最大 drop 分别为 <b>{ablation_lookup['1-10']['trace_drop']}</b> / <b>{ablation_lookup['11-20']['trace_drop']}</b> / <b>{ablation_lookup['21-30']['trace_drop']}</b>。只删 targeted top-1 时，三个区间的 trace-marker accuracy 仍为 <b>{cumulative_ablation_lookup['1-10']['target_trace_top1']}</b> / <b>{cumulative_ablation_lookup['11-20']['target_trace_top1']}</b> / <b>{cumulative_ablation_lookup['21-30']['target_trace_top1']}</b>；删 top-4 后降到 <b>{cumulative_ablation_lookup['1-10']['target_trace_top4']}</b> / <b>{cumulative_ablation_lookup['11-20']['target_trace_top4']}</b> / <b>{cumulative_ablation_lookup['21-30']['target_trace_top4']}</b>，删 top-8 后进一步降到 <b>{cumulative_ablation_lookup['1-10']['target_trace_top8']}</b> / <b>{cumulative_ablation_lookup['11-20']['target_trace_top8']}</b> / <b>{cumulative_ablation_lookup['21-30']['target_trace_top8']}</b>。同一 top-8 干预下，final-count accuracy 仍为 <b>{cumulative_ablation_lookup['1-10']['target_final_top8']}</b> / <b>{cumulative_ablation_lookup['11-20']['target_final_top8']}</b> / <b>{cumulative_ablation_lookup['21-30']['target_final_top8']}</b>。</div>
      <h4>分析</h4>
      <p>“最尖锐地指向 matching needle”与“单独删除时最不可替代”不是同一性质。top-1 几乎无损、top-4 开始下降、top-8 接近崩溃，说明 targeted retrieval 由多枚可互补的 routing heads 共同实现，而不是一枚唯一的“第 k 个 needle head”。final count 仍稳定也不能解释为 trace 无用：teacher-forced 输入已经给出 gold marker tokens，最终 <code>&lt;Ans&gt;</code> 可以绕过被损坏的生成步骤直接读取正确 trace。检验自由生成中的级联失败仍需要 autoregressive ablation。</p>

      <h3>7.5 CoT trace readout：与 targeted retrieval 是不同候选组</h3>
      <h4>实验</h4>
      <p>Targeted ranking 读取数字 <code>&lt;k&gt;</code> query；本实验改在最终 <code>&lt;Ans&gt;</code> query 上，以投向全部 trace marker positions 的 <code>trace_markers_mass</code> 排名 heads，再累计 mask top-n，并测 teacher-forced final-count accuracy。这样能检验“从已给定 trace 读出 count”的候选组，而不是再次检验 prompt retrieval。</p>
      <h4>结果</h4>
      {figure(
          generated['single_cot_final'],
          'Figure 4F. CoT 单头消融对最终 count readout 的影响',
          '<b>横轴</b>是 head index 0–3；<b>纵轴</b>是 Layer 1–4；三栏为三个 count 区间。颜色和格内数字是给定 gold trace 时的 final-count accuracy drop。'
      )}
      {figure(
          generated['cumulative_readout_final'],
          'Figure 4G. CoT trace-readout heads 的累计消融剂量曲线',
          '<b>横轴</b>是按 <code>trace_markers_mass</code> 排名后累计 global mask 的 head 数；<b>纵轴</b>是剩余 teacher-forced final-count accuracy。三栏为三个 count 区间。蓝线是 readout ranked top；淡灰细线、深灰均值线与灰色 min–max 阴影给出随机删除分布；紫色虚线是 random-4 mean，黑色虚线是无干预 baseline。'
      )}
      {table(ablation_rows,[('bin','count 区间'),('cot_final_head','最强 final 单头'),('cot_final_drop','final-count drop')])}
      {table(cumulative_ablation_rows,[('bin','count 区间'),('readout_final_top4','readout top-4 后 final'),('readout_final_top8','readout top-8 后 final'),('readout_final_random4','随机删 4 heads 的平均 final')])}
      <div class="callout">描述性 readout ranking 以前列 <code>L2H3</code>、<code>L2H2</code>、<code>L4H1</code>、<code>L4H0</code> 为主，与 k-to-k targeted ranking 不同。删 readout top-4 后，三个区间的 final-count accuracy 为 <b>{cumulative_ablation_lookup['1-10']['readout_final_top4']}</b> / <b>{cumulative_ablation_lookup['11-20']['readout_final_top4']}</b> / <b>{cumulative_ablation_lookup['21-30']['readout_final_top4']}</b>；删 top-8 后为 <b>{cumulative_ablation_lookup['1-10']['readout_final_top8']}</b> / <b>{cumulative_ablation_lookup['11-20']['readout_final_top8']}</b> / <b>{cumulative_ablation_lookup['21-30']['readout_final_top8']}</b>。累计曲线并不严格单调。</div>
      <h4>分析</h4>
      <p>Targeted retrieval 与 final trace readout 是两个不同阶段的候选 circuit。低 count 在 readout 删除下更早受损，中高 count 仍保留明显冗余；非单调曲线则说明 global mask 同时改变了多个相互补偿或竞争的通路，所以不能把“再增加一枚被 mask head”带来的边际差当作该 head 的独立贡献。第 8 节需要把 patch 限定在 <code>&lt;Ans&gt;</code> query，才能进一步确认 readout activation 是否足以搬运 count 信息。</p>

      <h3>7.6 Position-local ablation：把“哪个 head”与“在哪个 query 起作用”分开</h3>
      <h4>实验</h4>
      <p>前面的 global mask 会关闭一枚 head 在整条 sequence 的所有输出，因此任何累计顺序都可能混入 Layer 构成与非目标 token positions 上的上游作用。为把“哪个 head”与“在哪个 query 起作用”分开，本实验直接 hook 每层 attention 的 <code>c_proj</code> 输入；这个张量仍按四个 head slices 拼接。对选中的 head，只把指定语义 query 行对应的 64 维 slice 置零，其余 token positions、其余 heads、MLP 与参数全部保持原值。</p>
      <div class="protocol"><ol>
        <li><b>Non-thinking broad。</b>只在最终 <code>&lt;Ans&gt;</code> query 屏蔽 selected head outputs，测 final-count accuracy；同一 heads 在 prompt token 上的作用不受影响。</li>
        <li><b>CoT targeted retrieval。</b>只在所有 gold trace 数字 <code>&lt;k&gt;</code> queries 屏蔽 selected head outputs，测下一 token 是否为正确 <code>M_k</code> 的 trace-marker accuracy；marker、successor 和最终 <code>&lt;Ans&gt;</code> rows 不直接被 mask。</li>
        <li><b>CoT trace readout。</b>只在最终 <code>&lt;Ans&gt;</code> query 屏蔽 selected head outputs，测给定 gold trace 后的 final-count accuracy；trace 生成 rows 不受影响。</li>
        <li><b>严格同层随机对照。</b><code>same-layer random</code> 在累计第 t 步使用与 ranked-top 第 t 步相同的 Layer，只在该 Layer 内随机选择尚未删除的 head。因此每个 prefix 的 Layer 组成与 ranked-top 完全相同，差异不能再归因于“一个顺序先删 Layer 1、另一个先删 Layer 3/4”。图中逐条显示固定随机路径、逐步均值和 min–max；紫色 random-4 mean 表示同层随机路径在恰好删 4 个 heads 时的平均剩余 accuracy。</li>
      </ol></div>

      <h4>结果 A：Non-thinking broad heads，仅在 &lt;Ans&gt; 局部屏蔽</h4>
      {figure(
          generated['local_nonthinking'],
          'Figure 4H. Position-local non-thinking broad-head ablation',
          '<b>横轴</b>是累计只在 <code>&lt;Ans&gt;</code> query 屏蔽的 head 数；<b>纵轴</b>是剩余 teacher-forced final-count accuracy。三栏为 count 1–10、11–20、21–30。蓝线按 broad score 排名；淡灰细线是保持相同 Layer 序列的各条随机路径，深灰线是逐步随机均值，灰色阴影是随机 min–max；紫色虚线是同层 random-4 mean，黑色点线是无干预 baseline。'
      )}
      {table(position_local_direct_rows,[('bin','count 区间'),('top_n','局部 mask 数'),('ranked_top','broad ranked top'),('same_layer_random','同层 random 均值'),('random4_reference','同层 random-4 mean')])}
      <h4>分析 A</h4>
      <p>局部结果显示 broad score 在最终答案 query 上的特异性随 count 难度变化。只屏蔽 broad top-1 后，1–10 / 11–20 / 21–30 的 final-count accuracy 分别只剩 <b>12.5% / 11.3% / 68.8%</b>；同层 random 均值为 <b>2.7% / 65.5% / 99.5%</b>。因此中高 count 下，ranked top 比同层随机删除更早造成损伤；低 count 下随机删除同层 Layer-1 head 也常导致崩溃，说明该区间存在广泛的早层脆弱性，不能把 top-1 差异解释成排名特异性。累计 top-2 后三段 accuracy 为 <b>3.8% / 17.5% / 10.0%</b>。应结合灰色随机分布和 random-4 mean 判断 ranked curve 是否异常，而不是与人为构造的低分顺序比较。</p>

      <h4>结果 B：CoT targeted heads，仅在 trace 数字 &lt;k&gt; 局部屏蔽</h4>
      {figure(
          generated['local_targeted'],
          'Figure 4I. Position-local CoT targeted-retrieval ablation',
          '<b>横轴</b>是累计在所有 trace 数字 <code>&lt;k&gt;</code> queries 局部屏蔽的 head 数；<b>纵轴</b>是剩余 teacher-forced trace-marker accuracy。颜色和随机对照定义与 Figure 4H 相同。它直接回答 matching-needle 高分 heads 在产生 <code>M_k</code> 的 query 上是否比相同 Layer 构成的随机 heads 更必要。'
      )}
      {table(position_local_targeted_rows,[('bin','count 区间'),('top_n','局部 mask 数'),('ranked_top','targeted ranked top'),('same_layer_random','同层 random 均值'),('random4_reference','同层 random-4 mean')])}
      <h4>分析 B</h4>
      <p>Position-local + same-layer random control 让每个累计步拥有完全相同的 Layer 序列。局部屏蔽 ranked top-4 后，三个区间的 trace-marker accuracy 为 <b>76.8% / 65.9% / 61.7%</b>，而同层 random 均值为 <b>96.4% / 96.7% / 94.0%</b>。Top-1 后三个区间均保持 100%，top-2 后开始降至 <b>96.1% / 92.1% / 90.6%</b>。这支持 k-to-k mass 定位的是一组在数字 <code>&lt;k&gt;</code> query 上具有特异局部因果作用、但成员可相互补偿的 retrieval circuit，而不是唯一一枚“第 k 个 needle head”。屏蔽到 top-8 时不同路径覆盖的 head 集合高度重叠，端点不再适合比较排序优劣，重点应放在早期剂量相对随机分布的下降速度。</p>

      <h4>结果 C：CoT trace-readout heads，仅在 &lt;Ans&gt; 局部屏蔽</h4>
      {figure(
          generated['local_readout'],
          'Figure 4J. Position-local CoT trace-readout ablation',
          '<b>横轴</b>是累计只在最终 <code>&lt;Ans&gt;</code> query 屏蔽的 head 数；<b>纵轴</b>是给定 gold trace 后的 teacher-forced final-count accuracy。蓝线按 <code>trace_markers_mass</code> 排名；淡灰随机路径、深灰均值、灰色 min–max 和紫色 random-4 mean 构成严格同层随机对照。'
      )}
      {table(position_local_readout_rows,[('bin','count 区间'),('top_n','局部 mask 数'),('ranked_top','readout ranked top'),('same_layer_random','同层 random 均值'),('random4_reference','同层 random-4 mean')])}
      <h4>分析 C</h4>
      <p>该实验把 readout 干预限制到最终答案 row，因此不会直接破坏前面的 trace construction。结果显示 readout 的局部表示比 retrieval 更冗余：累计屏蔽 top-4 时三个区间仍全部为 <b>100%</b>；到 top-8 时才变为 <b>56.2% / 90.0% / 100.0%</b>，同层 random 在 top-8 的均值为 <b>70.8% / 91.3% / 100.0%</b>。这给出低 count 上有限的 readout-score 特异性，但中高 count 下尚不能证明前八枚 readout heads 的局部必要性。零消融仍只检验必要性，不检验某个 clean head output 是否足以恢复 corrupt count；充分性需要第 8 节的 query-local clean-to-corrupt patch。</p>

      <div class="callout warn"><b>修正后的结论边界。</b>严格的 position-local 结果显示：targeted ranked top-4 在三个 count 区间都比相同 Layer 构成的随机 heads 更早损伤 marker retrieval；broad ranked top 的同层特异性主要出现在 11–20 与 21–30，低 count 则表现为同层 Layer-1 heads 普遍必要。Global 曲线仍可说明某些 heads 对整条网络必要，但不能用来判断它们究竟在 retrieval、早期支持计算，还是最终 readout 位置发挥作用。因此第 7 节仅以固定随机路径、随机均值、随机范围和 random-4 mean 作为比较基线。</div>

      <h3>7.7 描述性 attention score 与因果 drop 是否一致</h3>
      <h4>实验</h4>
      <p>对每个机制和 count 区间，把 16 枚 heads 的描述性 attention score 与对应单头 global-ablation accuracy drop 对齐，并计算 Spearman 秩相关。该检验不要求两者线性，只问“描述分数排名更高的 head，是否通常也造成更大因果损伤”。</p>
      <h4>结果</h4>
      {figure(
          generated['alignment_nonthinking'],
          'Figure 4H. Non-thinking broad score 与 final-count drop',
          '<b>每个点</b>是一枚 Layer×head；<b>横轴</b>是 broad attention score；<b>纵轴</b>是单头 mask 后 final-count accuracy drop。三栏为三个 count 区间，标题给出 16 枚 heads 的 Spearman ρ。'
      )}
      {figure(
          generated['alignment_targeted'],
          'Figure 4I. CoT k-to-k mass 与 trace-marker drop',
          '<b>横轴</b>是 matching prompt needle 的 raw k-to-k attention mass；<b>纵轴</b>是单头 mask 后 trace-marker accuracy drop。每点是一枚 head，三栏为三个 count 区间。'
      )}
      {figure(
          generated['alignment_readout'],
          'Figure 4J. CoT trace-readout mass 与 final-count drop',
          '<b>横轴</b>是最终 <code>&lt;Ans&gt;</code> query 投向全部 trace markers 的 attention mass；<b>纵轴</b>是单头 mask 后 final-count accuracy drop。三栏为三个 count 区间。'
      )}
      {table(alignment_rows,[('mechanism','比较'),('bin','count 区间'),('rho','Spearman ρ'),('n','heads')])}
      <div class="callout">Non-thinking broad score 与 final-count drop 的 Spearman ρ 在三个区间分别为 <b>{alignment_lookup[('nonthinking','1-10','broad_attention_score')]}</b> / <b>{alignment_lookup[('nonthinking','11-20','broad_attention_score')]}</b> / <b>{alignment_lookup[('nonthinking','21-30','broad_attention_score')]}</b>。CoT k-to-k mass 与 trace-marker drop 分别为 <b>{alignment_lookup[('thinking','1-10','correct_prompt_needle_mass')]}</b> / <b>{alignment_lookup[('thinking','11-20','correct_prompt_needle_mass')]}</b> / <b>{alignment_lookup[('thinking','21-30','correct_prompt_needle_mass')]}</b>。CoT readout mass 与 final-count drop 分别为 <b>{alignment_lookup[('thinking','1-10','trace_markers_mass')]}</b> / <b>{alignment_lookup[('thinking','11-20','trace_markers_mass')]}</b> / <b>{alignment_lookup[('thinking','21-30','trace_markers_mass')]}</b>；最后一个区间因单头 drop 缺少方差而无法定义秩相关。</div>
      <h4>分析</h4>
      <p>Non-thinking broad score 与必要性呈中等正相关，说明描述性 broad attention 对 direct-count circuit 有实际筛选价值。CoT k-to-k 的相关为负，进一步说明 raw matching mass 不能等同于单头不可替代性：尖锐 retrieval heads 可能被其他 routing heads 替代，而早期支持性 heads 的删除反而更伤性能。readout score 的相关接近 0，也说明 final readout 不能只靠单头 attention mass 概括。可靠结论必须同时报告描述排序、单头必要性和累计 top-n 剂量曲线。
      </p>

      <h3>7.8 本节结论与下一步</h3>
      <div class="mechanisms">
        <div class="mechanism"><h3>Non-thinking</h3><p><b>当前支持：</b>Layer 1 broad heads 既有描述性 prompt-wide attention，也有强单头/累计必要性；它们很可能是直接 set aggregation 的关键入口。</p><p><b>仍缺：</b>global mask 不能证明这些 heads 把“count 数值”写到了哪里。第 8 节需要在 <code>&lt;Ans&gt;</code> 局部 patch broad-head output，并测 hidden-state/count logits 是否随 donor count 搬运。</p></div>
        <div class="mechanism"><h3>CoT</h3><p><b>当前支持：</b>k-to-k heads 作为一个组对 marker trace 必要，但单头可替代；targeted retrieval 与 final trace readout 是不同阶段、不同排名的多头 circuit。</p><p><b>仍缺：</b>teacher forcing 隔断了局部 trace 错误向最终答案的级联。第 8 节应分别在 <code>&lt;k&gt;</code> 和 <code>&lt;Ans&gt;</code> 做局部 activation patch，并使用 marker-identity margin、next-index margin 与 final-count margin 区分检索、successor 和 readout。</p></div>
      </div>
    </section>
    """

    patching_section = f"""
    <section id="patching">
      <h2>8. Attention-head patching：候选 head output 是否具有局部因果充分性</h2>
      <p>第 7 节的 ablation 问的是“移除某个组件会不会坏”，属于<b>必要性</b>证据；本节反过来问“只把 clean/donor 的这个组件放进 corrupt/receiver run，能不能把缺失的信息带回来”，属于<b>局部充分性与信息运输</b>证据。所有 patch 都发生在 attention 的 <code>c_proj</code> 之前：四枚 head 的 value-weighted outputs 仍是四段独立的 64 维 slices；只替换指定 Layer×head、指定 semantic query 的 slice，其他 token rows、其他 heads、MLP、residual stream 和参数保持 receiver/corrupt 值。</p>

      <h3>8.1 共用实验定义：clean/corrupt、donor/receiver 与两种结果量</h3>
      <div class="protocol"><ol>
        <li><b>Marker-identity clean-to-corrupt patch。</b>clean 与 corrupt 的 256-token prompt、needle 位置、count、trace 数字和全部上下文长度相同；唯一变化是目标第 k 个 prompt needle 的 marker identity。clean run 需要预测原 marker，corrupt run 在同一位置改成另一 marker。我们在同一个 CoT 数字 <code>&lt;k&gt;</code> query，把 clean 的 selected head-output slices 放进 corrupt run。</li>
        <li><b>Clean-marker logit margin。</b>在 <code>&lt;k&gt;</code> query 的 next-token logits 中，以 clean marker 的 logit 减去其余 marker token 中最大 logit。margin 为正表示 clean marker 胜过所有 marker competitors；为负表示模型更偏向某个错误 marker。</li>
        <li><b>Normalized recovery。</b><code>(patched margin − corrupt margin) / (clean margin − corrupt margin)</code>。0 表示 patch 没有把 corrupt 状态拉向 clean；1 表示恢复到 clean margin；大于 1 是过度恢复；小于 0 表示比 corrupt 更差。先对每个 clean/corrupt pair 计算，再在 count 区间内平均。</li>
        <li><b>Nested donor-to-receiver patch。</b>receiver count 为 n，donor count 为 m；两者共享同一 noise sequence，且较小 count 的 needle 集合严格是较大 count 的子集。合法 offset 为 <code>m−n ∈ ±&#123;1,2,3,5,10&#125;</code>。在各自最终 <code>&lt;Ans&gt;</code> query，把 donor selected head slices 放进 receiver；CoT 因 trace 长度不同，绝对 position 可以不同，但 semantic query 都是最终 answer readout。</li>
        <li><b>Expected count 与 transport slope。</b>只在 count-token logits 上做 softmax，计算 <code>E[count]=Σ_c c·p(c)</code>。每个 pair 的 causal shift 是 <code>E[count]_patched − E[count]_receiver</code>；在每个 count 区间拟合 <code>shift = a + b(m−n)</code>。纵轴 slope <code>b=1</code> 表示 donor offset 被一比一搬到输出，<code>b=0</code> 表示没有系统 count transport。</li>
        <li><b>候选与随机对照。</b>蓝线按目标机制的描述性 score 从高到低 patch：CoT retrieval 使用 k-to-k mass，non-thinking 使用 broad score，CoT final readout 使用 trace-marker mass。淡灰细线是三条固定随机 head 顺序；深灰是逐 top-n 随机均值，阴影是 min–max；紫色水平线只是 random top-4 mean 的易读参照。低分/bottom 排名不再作为对照。</li>
        <li><b>Successor/stop 的 nested pair。</b>对每个 <code>k=1,…,29</code>，构造 short prompt（count=k）和 long prompt（count=k+1）。两者共享同一 noise sequence、前 k 个 needle 的位置与 marker identity；long 只在更晚位置多一个 needle。因此从 <code>&lt;Think&gt;</code> 到第 k 个 marker <code>M_k</code> 的 teacher-forced trace 完全相同，差异只在 prompt 是否还存在第 k+1 个 needle。</li>
        <li><b>Successor query 与双向 margin。</b>patch row 是 marker <code>M_k</code> 本身：其 next-token decision 在 long prompt 应为 <code>&lt;k+1&gt;</code>，在 short prompt 应为 <code>&lt;/Think&gt;</code>。Continue margin 定义为 <code>z(&lt;k+1&gt;)−z(&lt;/Think&gt;)</code>；close margin 反向定义为 <code>z(&lt;/Think&gt;)−z(&lt;k+1&gt;)</code>。两者均以 margin&gt;0 记为目标决定正确。</li>
        <li><b>Successor controls。</b><code>successor_top</code> 按 marker query 指向下一 prompt needle 的 raw mass 排序；<code>targeted_top</code> 沿用数字 <code>&lt;k&gt;</code> 的 k-to-k retrieval 排名，检验两个阶段是否共用 heads；<code>wrong donor row</code> 使用相同 successor heads，却把 donor 前一行数字 <code>&lt;k&gt;</code> 的 slice 贴到 receiver 的 <code>M_k</code> row，检验结果是否真的依赖 marker-query activation；另有四条固定随机 head 顺序。</li>
      </ol></div>

      <h3>8.2 CoT targeted retrieval：在数字 &lt;k&gt; query 恢复正确 marker identity</h3>
      <h4>实验</h4>
      <p>每个 exact count 使用 2 个独立 prompts。对每个样本分别选择一个<b>内部步骤</b>（约位于 trace 中部的 k）与<b>最后步骤</b>（k=n）。在 clean/corrupt prompt 中只改对应第 k 个 needle 的 marker identity；随后按第 5 节 k-to-k mass 排名，依次 patch top-1、top-2、top-4、top-8、top-16 clean head slices 到 corrupt 的数字 <code>&lt;k&gt;</code> row。结果变量是 clean-marker normalized recovery，不是离散 accuracy。</p>
      <h4>结果 A：interior k</h4>
      {figure(
          generated['retrieval_interior'],
          'Figure 5A. Interior &lt;k&gt; 的 marker-identity clean-to-corrupt recovery',
          '<b>横轴</b>是被局部 patch 的 clean head-output slices 数；<b>纵轴</b>是 clean-marker logit margin 的 normalized recovery，黑实线 0 表示无恢复，黑虚线 1 表示完全恢复。三栏按 gold count 分区。蓝线按 k-to-k score patch top-n；灰线与灰带为随机路径；紫色虚线是随机 top-4 mean。'
      )}
      <h4>结果 B：final k=n</h4>
      {figure(
          generated['retrieval_final'],
          'Figure 5B. Final &lt;n&gt; 的 marker-identity clean-to-corrupt recovery',
          '<b>横纵轴与 Figure 5A 相同，但 query 是 trace 最后一个数字 <code>&lt;n&gt;</code>。它检验 retrieval circuit 是否在 close 之前的最后一次 marker retrieval 上仍工作，而不是 successor/close token 本身。'
      )}
      {table(retrieval_patch_rows,[('role','query role'),('bin','gold count 区间'),('heads','patched top-n'),('ranked','targeted recovery'),('random','random mean'),('range','random min–max')])}
      <h4>分析</h4>
      <p>单枚最高分 head 并不充分：interior top-1 在三个区间只恢复 <b>{retrieval_patch_lookup[('interior k','1-10',1)]['ranked']}</b> / <b>{retrieval_patch_lookup[('interior k','11-20',1)]['ranked']}</b> / <b>{retrieval_patch_lookup[('interior k','21-30',1)]['ranked']}</b>。但 top-2 立即升到 <b>{retrieval_patch_lookup[('interior k','1-10',2)]['ranked']}</b> / <b>{retrieval_patch_lookup[('interior k','11-20',2)]['ranked']}</b> / <b>{retrieval_patch_lookup[('interior k','21-30',2)]['ranked']}</b>，top-4 达 <b>{retrieval_patch_lookup[('interior k','1-10',4)]['ranked']}</b> / <b>{retrieval_patch_lookup[('interior k','11-20',4)]['ranked']}</b> / <b>{retrieval_patch_lookup[('interior k','21-30',4)]['ranked']}</b>。final k=n 呈现几乎相同的剂量响应，top-4 recovery 为 <b>{retrieval_patch_lookup[('final k=n','1-10',4)]['ranked']}</b> / <b>{retrieval_patch_lookup[('final k=n','11-20',4)]['ranked']}</b> / <b>{retrieval_patch_lookup[('final k=n','21-30',4)]['ranked']}</b>。</p>
      <div class="callout good"><b>因果结论：</b>k-to-k retrieval 不是由一枚最高分 head 独立完成，而是由一个很小的多头 activation bundle 搬运 marker identity。Top-2 已恢复大部分 clean margin，top-4 在全部 count 区间接近完全恢复；这与第 7 节“单头可替代、成组消融才明显受损”相互吻合。随机 top-4 均值只有约 0.22–0.27，但 min–max 很宽且仅有三条随机路径，因此证据应表述为 ranked top-4 明显高于随机均值，而不是声称胜过每一个随机组合。</div>
      <p><b>没有检验什么。</b>本实验只在数字 <code>&lt;k&gt;</code> query 检验“下一个 marker identity 是什么”。它没有在 marker <code>M_k</code> query 上 patch，也没有测 <code>&lt;k+1&gt;</code> versus <code>&lt;/Think&gt;</code> 的 next-index/close margin，因此不能回答模型如何从第 k 步推进到第 k+1 步。</p>

      <h3>8.3 CoT successor/stop：在 marker M<sub>k</sub> query 运输“继续还是关闭”证据</h3>
      <h4>实验</h4>
      <p>本实验专门补上 Figure 5 没有覆盖的下一步：模型已经在数字 <code>&lt;k&gt;</code> 后检索并生成了 marker <code>M_k</code>，此时它如何决定再输出 <code>&lt;k+1&gt;</code>，还是用 <code>&lt;/Think&gt;</code> 关闭 trace。对每个 <code>k=1,…,29</code>，我们构造一对严格 nested prompts：short receiver 含 k 个 needles，long donor 含 k+1 个 needles；二者共享全部 256 个 noise tokens、前 k 个 needle 的位置与 marker identity，long 只在更晚的一个空位置新增第 k+1 个 needle。因此两条 teacher-forced sequence 从 <code>&lt;Think&gt;</code> 到 <code>M_k</code> 完全相同，只有 prompt 中“是否还有下一个 needle”不同。</p>
      <div class="protocol"><ol>
        <li><b>Continue → close receiver。</b>clean/donor 是 long prompt，正确 next token 为 <code>&lt;k+1&gt;</code>；corrupt/receiver 是 short prompt，正确 next token 为 <code>&lt;/Think&gt;</code>。在 receiver 的 <code>M_k</code> query 局部贴入 donor head-output slices，读取 <code>z(&lt;k+1&gt;)−z(&lt;/Think&gt;)</code>。若 patch 把短序列拉向“继续”，该 margin 与 target-decision accuracy 应上升。</li>
        <li><b>Close → continue receiver。</b>clean/donor 改为 short prompt，corrupt/receiver 改为 long prompt；读取反向 margin <code>z(&lt;/Think&gt;)−z(&lt;k+1&gt;)</code>。这是独立的反向检验，排除某组 heads 只会无条件增加 index logit 的解释。</li>
        <li><b>Patch 位置。</b>只在 marker <code>M_k</code> 这一行、attention <code>c_proj</code> 之前替换选中 Layer×head 的 64 维 output slice；前面的数字 <code>&lt;k&gt;</code> row、其他 token rows、未选 heads、MLP 与 residual stream 均保持 receiver 值。</li>
        <li><b>四组排序/对照。</b><code>successor-score ranked</code> 按 <code>M_k</code> query 指向第 k+1 个 prompt needle 的 attention mass 排序；<code>k-to-k targeted ranked</code> 使用上一阶段数字 <code>&lt;k&gt;</code> 到第 k 个 needle 的排名；<code>wrong donor row</code> 使用同一 successor heads，却把 donor 的数字 <code>&lt;k&gt;</code> row 贴到 receiver 的 <code>M_k</code> row；灰线为四条固定随机 head 顺序。</li>
        <li><b>结果量。</b>每个 pair 先计算 normalized recovery，再在 1–10、11–20、21–30 三个 receiver count 区间内平均。离散 target-decision accuracy 是 patch 后目标 margin 大于 0 的样本比例；它回答 patch 是否真的翻转了 continue/close 选择，而不只是让 margin 小幅移动。</li>
      </ol></div>

      <h4>结果 A：把 continue activation 贴进本应关闭的 short receiver</h4>
      {figure(
          generated['successor_continue_recovery'],
          'Figure 5C. Continue → close receiver：next-index margin 的 normalized recovery',
          '<b>横轴</b>是只在 <code>M_k</code> query 局部 patch 的 top-n head slices 数（1、2、4、8、12、16）；<b>纵轴</b>是 continue margin <code>z(&lt;k+1&gt;)−z(&lt;/Think&gt;)</code> 的 normalized recovery。三栏按 short receiver count 分区。蓝线按 successor score，橙线沿用 k-to-k targeted 排名，绿线用 successor heads 但贴错 donor row；灰线、深灰均值与灰带为四条随机顺序。黑实线 0 表示无恢复，黑虚线 1 表示达到 clean long-prompt margin。'
      )}
      {figure(
          generated['successor_continue_accuracy'],
          'Figure 5D. Continue → close receiver：patch 后继续决定的 accuracy',
          '<b>横轴与分组</b>同 Figure 5C；<b>纵轴</b>是 patch 后 <code>z(&lt;k+1&gt;)−z(&lt;/Think&gt;)&gt;0</code> 的 pair 比例。它是严格的决定翻转率，不是 next-token 全词表 accuracy。虚线 1 表示所有 short receivers 都被 donor activation 推向继续。'
      )}
      <p>单头和 top-2 总体不足以稳定翻转决定，但 top-4 开始呈现清楚的高-count 效应。Successor-ranked top-4 在 1–10 / 11–20 / 21–30 的 recovery 为 <b>{successor_value('continue_into_close','1-10',4,'successor_top','recovery')}</b> / <b>{successor_value('continue_into_close','11-20',4,'successor_top','recovery')}</b> / <b>{successor_value('continue_into_close','21-30',4,'successor_top','recovery')}</b>，对应 target-decision accuracy 为 <b>{successor_value('continue_into_close','1-10',4,'successor_top','accuracy')}</b> / <b>{successor_value('continue_into_close','11-20',4,'successor_top','accuracy')}</b> / <b>{successor_value('continue_into_close','21-30',4,'successor_top','accuracy')}</b>。扩到 top-8 后 recovery 达 <b>{successor_value('continue_into_close','1-10',8,'successor_top','recovery')}</b> / <b>{successor_value('continue_into_close','11-20',8,'successor_top','recovery')}</b> / <b>{successor_value('continue_into_close','21-30',8,'successor_top','recovery')}</b>，中高 count 已接近完全恢复。</p>

      <h4>结果 B：把 close activation 贴进本应继续的 long receiver</h4>
      {figure(
          generated['successor_close_recovery'],
          'Figure 5E. Close → continue receiver：close margin 的 normalized recovery',
          '<b>横轴</b>是 patched top-n；<b>纵轴</b>改为 close margin <code>z(&lt;/Think&gt;)−z(&lt;k+1&gt;)</code> 的 normalized recovery。receiver 是仍有第 k+1 个 needle 的 long prompt，donor 是在 k 处结束的 short prompt。正向恢复表示局部 activation 能把本应继续的状态拉向关闭。其余颜色、三段 count 区间和随机对照与 Figure 5C 相同。'
      )}
      {figure(
          generated['successor_close_accuracy'],
          'Figure 5F. Close → continue receiver：patch 后关闭决定的 accuracy',
          '<b>纵轴</b>是 patch 后 <code>z(&lt;/Think&gt;)−z(&lt;k+1&gt;)&gt;0</code> 的 pair 比例；因此它测的是 donor close evidence 是否足以翻转 long receiver，而不是模型原本的自然 trace accuracy。'
      )}
      <p>反向结果排除了“patch 只会把数字 logits 普遍抬高”的单向解释。Successor-ranked top-4 在三个区间的 close recovery 为 <b>{successor_value('close_into_continue','1-10',4,'successor_top','recovery')}</b> / <b>{successor_value('close_into_continue','11-20',4,'successor_top','recovery')}</b> / <b>{successor_value('close_into_continue','21-30',4,'successor_top','recovery')}</b>，target-decision accuracy 为 <b>{successor_value('close_into_continue','1-10',4,'successor_top','accuracy')}</b> / <b>{successor_value('close_into_continue','11-20',4,'successor_top','accuracy')}</b> / <b>{successor_value('close_into_continue','21-30',4,'successor_top','accuracy')}</b>。Top-8 recovery 进一步达到 <b>{successor_value('close_into_continue','1-10',8,'successor_top','recovery')}</b> / <b>{successor_value('close_into_continue','11-20',8,'successor_top','recovery')}</b> / <b>{successor_value('close_into_continue','21-30',8,'successor_top','recovery')}</b>。</p>

      <h4>结果 C：top-4 候选、阶段对照与错误-row 对照</h4>
      {table(successor_table_rows,[('direction','patch 方向'),('bin','receiver count 区间'),('family','head/row 条件'),('recovery','normalized recovery'),('accuracy','target-decision accuracy'),('random_recovery','random recovery 均值'),('random_recovery_range','random recovery min–max')])}
      <h4>分析</h4>
      <p><b>第一，successor 是分布式局部 circuit，而不是单枚开关 head。</b>单头几乎从不翻转 continue/close；top-4 在中高 count 已明显有效，top-8 才在两种方向上接近完整恢复。这与第 7 节的多头冗余一致，但这里的证据更强：它不是删除后性能下降，而是 selected clean activations 在同一 <code>M_k</code> row 足以把 receiver 的决定向 donor 搬运。</p>
      <p><b>第二，retrieval 与 successor 有重叠，但 top heads 并不相同。</b>在中高 count 的 top-4，数字 <code>&lt;k&gt;</code> 上得到的 k-to-k ranking 明显弱于 successor ranking；到 top-8 后二者才共同接近完全恢复。这说明较大的 routing circuit 可能跨阶段复用，但“找到第 k 个 needle”与“在 M<sub>k</sub> 后决定继续/关闭”不能由同一个 top-4 排名概括。</p>
      <p><b>第三，query row 有信息，但不是绝对隔离。</b>同一 successor heads 若改贴 donor 的上一行数字 <code>&lt;k&gt;</code> activation，top-4 通常显著弱于正确 <code>M_k</code> row，尤其在反向 close patch 与高 count 条件下；然而某些 wrong-row top-8 仍能恢复，说明 continue/close 信息也可能已经分布在相邻 residual/attention rows，不能声称只存在于 marker row。</p>
      <div class="callout good"><b>本实验回答了什么：</b>在受控 nested prompts 中，marker <code>M_k</code> query 的一个多头 activation bundle 同时携带可双向运输的 continue/close 证据；该 bundle 在中高 count 上用 top-4 已可显著改变决定，top-8 接近充分。<b>没有证明什么：</b>它尚不能把计算解释成符号式 <code>k+1</code> 加法，也没有证明同一 circuit 会在完全 autoregressive generation 中阻止所有级联错误；它定位的是局部 next-index/stop decision 的因果载体。</div>

      <h3>8.4 从 attention evidence 到具体 index logit：MLP 与 residual 的层内转换</h3>
      <h4>实验</h4>
      <p>本实验继续使用 8.3 节完全相同的 nested short/long pairs 与 marker <code>M_k</code> query，但不再只把整个 head bundle 当成黑箱。每个方向包含 <b>{successor_pairs_per_direction}</b> 对样本（<code>k=1,…,29</code>，每个 k 使用 {successor_conversion_manifest['examples_per_k']} 个独立 base sequences），并保留 continue→close 与 close→continue 两个方向。实验仍是单步、teacher-forced 的局部因果分析；按照本节问题范围，<b>没有运行 autoregressive rollout</b>。</p>
      <div class="protocol"><ol>
        <li><b>Layer 内计算分解。</b>对每一层，记进入该层的 residual 为 <code>h_pre</code>，attention 输出为 <code>a</code>，则 <code>h_attn = h_pre + a</code>；MLP 输出为 <code>m = MLP(LN₂(h_attn))</code>，层末 residual 为 <code>h_post = h_attn + m</code>。报告把它们依次称为 <code>resid_pre</code>、<code>attn_out</code>、<code>post_attn</code>、<code>mlp_out</code>、<code>post_mlp</code>。</li>
        <li><b>Residual logit lens。</b>在每个 residual stage 上临时应用模型最终的 <code>ln_f</code> 与 tied unembedding，计算 target token 相对 competitor 的 margin：continue 方向为 <code>z(&lt;k+1&gt;)−z(&lt;/Think&gt;)</code>。这回答“到这一 stage 时，目标决定是否已经可被最终读出头线性读取”，而不是声称模型在中间层真的提前退出。</li>
        <li><b>Component direct-unembedding diagnostic。</b>对 additive component <code>a</code> 或 <code>m</code> 直接乘 unembedding，并计算 clean-minus-corrupt target-aligned margin。由于 component 本身没有经过最终 LayerNorm，这只是定位哪个 component 把证据写向目标 token 的描述性诊断；真正的因果结论来自下一项 patch。</li>
        <li><b>关键 MLP mediation 对照。</b><code>donor Attn; receiver MLP frozen</code> 把 donor attention output 加进 receiver residual，但强制保留 receiver baseline MLP output；<code>donor Attn; native MLP recomputed</code> 使用同一 donor attention output，却让 MLP 对改变后的 <code>h_attn</code> 自然重算。二者 normalized recovery 之差，就是 donor attention evidence 经当前层 MLP 转换带来的局部 mediation gain。</li>
        <li><b>其他干预。</b><code>donor MLP output only</code> 只把 donor MLP additive output 加到 receiver <code>h_attn</code>；<code>donor Attn + donor MLP</code> 同时替换两个 additive components；<code>full donor post-attn state</code> 替换 <code>h_attn</code> 后重算 MLP；<code>full donor post-MLP state</code> 替换整层输出，是该层完整状态充分性的上界/sanity check。</li>
      </ol></div>

      <h4>结果 A：continue/close margin 在 residual stream 的哪里出现</h4>
      {figure(
          generated['successor_logit_lens'],
          'Figure 5G. Marker M_k 处的 residual logit-lens 轨迹',
          '<b>横轴</b>按计算顺序列出每层输入（Lx pre）、加完 attention 后（Lx +Attn）与再加完 MLP 后（Lx +MLP）的 residual；<b>纵轴</b>是经最终 <code>ln_f + unembedding</code> 读出的 continue target margin。蓝线是存在第 k+1 个 needle 的 long/clean prompt；橙线是 count=k 的 short/corrupt prompt，但也用同一个 continue target 读取。两线分离表示该 stage 已包含可线性读取的 continue-versus-close 证据。三栏为 receiver count 区间。'
      )}
      <p>低 count 的 clean−corrupt residual gap 在 Layer 3 post-attention 为 <b>{stage_gap('1-10',2,'post_attn')}</b>，经 Layer 3 MLP 后扩大到 <b>{stage_gap('1-10',2,'post_mlp')}</b>；Layer 4 post-attention 与 post-MLP 分别进一步达到 <b>{stage_gap('1-10',3,'post_attn')}</b> 和 <b>{stage_gap('1-10',3,'post_mlp')}</b>。11–20 的对应四个数是 <b>{stage_gap('11-20',2,'post_attn')}</b>、<b>{stage_gap('11-20',2,'post_mlp')}</b>、<b>{stage_gap('11-20',3,'post_attn')}</b>、<b>{stage_gap('11-20',3,'post_mlp')}</b>；21–30 则为 <b>{stage_gap('21-30',2,'post_attn')}</b>、<b>{stage_gap('21-30',2,'post_mlp')}</b>、<b>{stage_gap('21-30',3,'post_attn')}</b>、<b>{stage_gap('21-30',3,'post_mlp')}</b>。</p>

      <h4>结果 B：attention 与 MLP 各自把多少证据直接写向目标 token</h4>
      {figure(
          generated['successor_component_evidence'],
          'Figure 5H. Attention-output 与 MLP-output 的 target-aligned direct-unembedding evidence',
          '<b>横轴</b>是产生 additive component 的 Layer 1–4；<b>纵轴</b>是该 component 对目标 margin 的 clean-minus-corrupt 差。蓝线为 attention output，橙线为 MLP output；正值表示该 component 在 long prompt 中比 short prompt 更偏向 <code>&lt;k+1&gt;</code>。这是没有最终 LayerNorm 的分量诊断，不单独作为因果证据。'
      )}
      {table(successor_stage_rows,[('bin','count 区间'),('layer','Layer'),('attn_component','attention component gap'),('mlp_component','MLP component gap'),('post_attn_gap','post-attn residual gap'),('post_mlp_gap','post-MLP residual gap')])}
      <p>Layer 3 是证据第一次稳定变大的路由层；Layer 4 则表现出最强的 logit 写入。三个 count 区间中，Layer 4 attention component gap 分别为 <b>{stage_gap('1-10',3,'attn_out')}</b> / <b>{stage_gap('11-20',3,'attn_out')}</b> / <b>{stage_gap('21-30',3,'attn_out')}</b>，而 Layer 4 MLP component gap 达 <b>{stage_gap('1-10',3,'mlp_out')}</b> / <b>{stage_gap('11-20',3,'mlp_out')}</b> / <b>{stage_gap('21-30',3,'mlp_out')}</b>。因此最终 index/close logit 的巨大分离并不是 attention output 独自线性写出的；MLP 是更强的目标-token amplifier。</p>

      <h4>结果 C：固定 receiver MLP 与允许 native MLP 重算的因果对照</h4>
      {figure(
          generated['successor_conversion_continue'],
          'Figure 5I. Continue evidence → close receiver 的 sublayer conversion',
          '<b>横轴</b>是被干预的 Layer 1–4；<b>纵轴</b>是 continue margin normalized recovery。蓝线只替换 donor attention output并冻结 receiver MLP output；绿线使用相同 donor attention，却让当前层 MLP 对新 residual 自然重算；橙线只替换 donor MLP output；灰线替换完整 donor post-MLP residual，作为 full-state 上界。三栏按 short receiver count 分区。'
      )}
      {figure(
          generated['successor_conversion_close'],
          'Figure 5J. Close evidence → continue receiver 的 sublayer conversion',
          '<b>横纵轴及四种干预</b>与 Figure 5I 相同，但 target margin 改为 <code>z(&lt;/Think&gt;)−z(&lt;k+1&gt;)</code>，receiver 是本应继续的 long prompt。该反向实验检验 MLP mediation 是否同时支持关闭，而不是只普遍抬高数字 logits。'
      )}
      {table(successor_conversion_rows,[('direction','patch 方向'),('bin','count 区间'),('layer','Layer'),('attn_direct','Attn + frozen MLP recovery'),('native_mlp','Attn + native MLP recovery'),('mlp_mediation','MLP mediation gain'),('mlp_only','MLP-only recovery'),('full_state','full post-MLP recovery'),('direct_accuracy','Attn-only target accuracy'),('native_accuracy','native-MLP target accuracy'),('mlp_accuracy','MLP-only target accuracy')])}
      <h4>分析</h4>
      <p><b>Layer 3 把 routed evidence 转成可执行决定。</b>Continue 方向在 11–20 / 21–30 中，Layer 3 attention-direct recovery 为 <b>{conversion_value('continue_into_close','11-20',2,'attn_direct_residual')}</b> / <b>{conversion_value('continue_into_close','21-30',2,'attn_direct_residual')}</b>；允许同层 MLP 自然重算后升到 <b>{conversion_value('continue_into_close','11-20',2,'attn_native_mlp')}</b> / <b>{conversion_value('continue_into_close','21-30',2,'attn_native_mlp')}</b>，mediation gain 分别约 0.40 与 0.32。反向 close 方向同样从 <b>{conversion_value('close_into_continue','11-20',2,'attn_direct_residual')}</b> / <b>{conversion_value('close_into_continue','21-30',2,'attn_direct_residual')}</b> 升至 <b>{conversion_value('close_into_continue','11-20',2,'attn_native_mlp')}</b> / <b>{conversion_value('close_into_continue','21-30',2,'attn_native_mlp')}</b>。这说明 MLP 不是被动经过 donor evidence，而是在当前 residual 上做了实质的非线性转换。</p>
      <p><b>Layer 4 MLP 更像最终 token-logit writer。</b>仅替换 Layer 4 MLP output 的 recovery 在 continue 方向为 <b>{conversion_value('continue_into_close','1-10',3,'mlp_out_only')}</b> / <b>{conversion_value('continue_into_close','11-20',3,'mlp_out_only')}</b> / <b>{conversion_value('continue_into_close','21-30',3,'mlp_out_only')}</b>，在 close 方向为 <b>{conversion_value('close_into_continue','1-10',3,'mlp_out_only')}</b> / <b>{conversion_value('close_into_continue','11-20',3,'mlp_out_only')}</b> / <b>{conversion_value('close_into_continue','21-30',3,'mlp_out_only')}</b>。完整 Layer 4 post-MLP state 在六个方向×区间条件中 recovery 均为 1；但 MLP-only 并非总能独立达到 1，说明最终写入仍依赖 receiver residual、attention evidence 与 MLP output 的组合，而不是一个脱离上下文的标量加法神经元。</p>
      <div class="callout good"><b>本实验补上的因果链：</b><code>M_k attention routing</code> 提供 continue/close 证据；该证据在 Layer 3 residual 中变得可读，并由同层 MLP 显著转化；Layer 4 MLP 再把状态强烈对齐到具体 <code>&lt;k+1&gt;</code> 或 <code>&lt;/Think&gt;</code> logit。换言之，attention 更像“把下一步是否存在的证据送入当前位置”，MLP/residual 组合更像“把证据变成当前词表上的决定”。这比“某个 attention head 直接写出 index token”更符合干预结果。</div>
      <div class="callout warn"><b>到这里仍不能声称：</b>direct-unembedding 图不是 feature 级机制，whole-MLP patch 也不能区分是少数坐标还是分布式子空间在执行转换。下一小节因此直接进入 Layer 3–4 的 1024 维 post-GELU MLP intermediate，并用未参与排序的 held-out pairs 做 feature patch。</div>

      <h3>8.5 Layer 3–4 MLP intermediate：从 routed evidence 到 index/close logit 的 feature 级因果分解</h3>
      <h4>实验</h4>
      <p>这里的 <b>MLP feature/neuron</b> 是 GPT-2 MLP 中 <code>c_fc → GELU</code> 后、<code>c_proj</code> 前的一个标量坐标，不是生物神经元。模型的 residual width 为 256，而每层 MLP intermediate width 为 <b>{mlp_feature_manifest['n_inner']}</b>；实验只分析 0-based Layer 2–3，即报告中的 <b>Layer 3–4</b>。Query、clean/corrupt pair 和 continue/close margin 与 8.3–8.4 完全相同，仍位于 teacher-forced marker <code>M_k</code>。</p>
      <div class="protocol"><ol>
        <li><b>严格分离 feature 排名与因果评估。</b>对每个 <code>k=1,…,29</code>，使用 {mlp_feature_manifest['fit_examples_per_k']} 个 base sequences 拟合 feature 排名，再使用不重叠的 {mlp_feature_manifest['eval_examples_per_k']} 个 sequences 做 patch；因此每个方向各有 <b>{mlp_fit_pairs_per_direction}</b> 个 fit pairs 和 <b>{mlp_eval_pairs_per_direction}</b> 个 held-out eval pairs。</li>
        <li><b>Feature 排名。</b>对第 i 个 post-GELU feature，先计算 fit pairs 上的 clean−corrupt activation 差，再乘它经 <code>c_proj</code> 与 tied unembedding 对 target-minus-alternative margin 的线性系数。这个乘积的均值是 projected evidence score。它只用于选坐标，不直接当作因果效应。</li>
        <li><b>Ranked feature replacement。</b>对 support <code>S</code> 中的 feature，把 corrupt receiver 在同一 <code>M_k</code> row 的 activation 坐标替换成 clean donor 值；其他 1024−|S| 个坐标、attention、residual 与后续参数都保留 receiver 值。支持大小依次为 <code>{', '.join(str(x) for x in mlp_feature_manifest['support_sizes'])}</code>。</li>
        <li><b>Sparse mean-direction transport。</b>在 fit pairs 上求平均 clean−corrupt feature 差方向，只保留 projected-evidence 绝对值最大的 S 个坐标；对 held-out pair，仅运输该 pair 在这个稀疏方向上的投影。它检验“共享低维方向”是否足以替代逐坐标 donor 值。</li>
        <li><b>对照。</b>每个 support 使用 {mlp_feature_manifest['random_replicates']} 组固定随机坐标做 size-matched replacement，并另测 random sparse direction。所有结果仍报告逐 pair normalized recovery 与 target-decision accuracy；没有 autoregressive rollout。</li>
      </ol></div>

      <h4>结果 A：转换证据是部分稀疏、总体分布式的</h4>
      {figure(
          generated['mlp_feature_concentration'],
          'Figure 5K. Layer 3–4 MLP projected evidence 的累积集中度',
          '<b>横轴</b>是按 projected evidence 排名后保留的 feature 数，使用 log₂ 刻度；<b>纵轴</b>是这些 features 覆盖的证据比例。实线只累计方向正确的正 evidence，虚线累计绝对 evidence；颜色表示 receiver count 区间。上、下行分别为 continue evidence 与 close evidence，左、右列分别为 Layer 3 与 Layer 4。若由单一 neuron 执行，曲线会在 support=1 附近跃升到 1；实际需要数十到数百个坐标。'
      )}
      <p>以反向写入 close evidence 为例，Layer 4 的 top-64 features 已覆盖三个 count 区间约 <b>59.7% / 66.5% / 71.8%</b> 的正 projected evidence，top-256 覆盖约 <b>90.1% / 91.3% / 93.6%</b>。Continue 与 close 两行都显示 evidence 有明显集中度，但都不是单-neuron code；更准确的说法是<b>分布式、部分稀疏的 decision-writing subspace</b>。</p>

      <h4>结果 B：Continue evidence 的 held-out feature patch</h4>
      {figure(
          generated['mlp_feature_continue'],
          'Figure 5L. Clean continue features → close receiver 的 normalized recovery',
          '<b>横轴</b>是被替换或运输的 MLP feature support；<b>纵轴</b>是 continue margin normalized recovery。上排 Layer 3，下排 Layer 4；三列为 receiver count 1–10、11–20、21–30。蓝线替换 ranked clean coordinates，绿线运输 sparse mean direction，灰线和阴影是 matched random coordinates 的均值±标准差。虚线 1 表示达到 clean margin。'
      )}
      <p>Layer 4 ranked top-64 的 recovery 为 <b>{mlp_feature_value('continue_into_close','1-10',3,'ranked_feature_replacement',64)} / {mlp_feature_value('continue_into_close','11-20',3,'ranked_feature_replacement',64)} / {mlp_feature_value('continue_into_close','21-30',3,'ranked_feature_replacement',64)}</b>，而 matched random 仅为 <b>{mlp_feature_value('continue_into_close','1-10',3,'random_feature_replacement',64)} / {mlp_feature_value('continue_into_close','11-20',3,'random_feature_replacement',64)} / {mlp_feature_value('continue_into_close','21-30',3,'random_feature_replacement',64)}</b>。Top-256 recovery 升至 <b>{mlp_feature_value('continue_into_close','1-10',3,'ranked_feature_replacement',256)} / {mlp_feature_value('continue_into_close','11-20',3,'ranked_feature_replacement',256)} / {mlp_feature_value('continue_into_close','21-30',3,'ranked_feature_replacement',256)}</b>，但 target-decision accuracy 只有 <b>{mlp_feature_value('continue_into_close','1-10',3,'ranked_feature_replacement',256,'patched_target_correct')} / {mlp_feature_value('continue_into_close','11-20',3,'ranked_feature_replacement',256,'patched_target_correct')} / {mlp_feature_value('continue_into_close','21-30',3,'ranked_feature_replacement',256,'patched_target_correct')}</b>。这一区分很重要：selected features 能连续搬运 continue margin，却未必足以让中高 count receiver 的 argmax 越过关闭边界。Layer 3 即使替换全部 1024 features，recovery 也只有 <b>{mlp_feature_value('continue_into_close','1-10',2,'ranked_feature_replacement',1024)} / {mlp_feature_value('continue_into_close','11-20',2,'ranked_feature_replacement',1024)} / {mlp_feature_value('continue_into_close','21-30',2,'ranked_feature_replacement',1024)}</b>。</p>

      <h4>结果 C：Close evidence 的反向 patch</h4>
      {figure(
          generated['mlp_feature_close'],
          'Figure 5M. Clean close features → continue receiver 的 normalized recovery',
          '<b>横纵轴、行列与对照</b>同 Figure 5L，但 target margin 是 <code>z(&lt;/Think&gt;)−z(&lt;k+1&gt;)</code>。这检验相同 feature-selection 方法是否能反向写入关闭 trace 的决定，而不是只普遍增强数字 token。'
      )}
      <p>Layer 4 ranked top-64 的 close recovery 为 <b>{mlp_feature_value('close_into_continue','1-10',3,'ranked_feature_replacement',64)} / {mlp_feature_value('close_into_continue','11-20',3,'ranked_feature_replacement',64)} / {mlp_feature_value('close_into_continue','21-30',3,'ranked_feature_replacement',64)}</b>，matched random 仅为 <b>{mlp_feature_value('close_into_continue','1-10',3,'random_feature_replacement',64)} / {mlp_feature_value('close_into_continue','11-20',3,'random_feature_replacement',64)} / {mlp_feature_value('close_into_continue','21-30',3,'random_feature_replacement',64)}</b>；top-256 为 <b>{mlp_feature_value('close_into_continue','1-10',3,'ranked_feature_replacement',256)} / {mlp_feature_value('close_into_continue','11-20',3,'ranked_feature_replacement',256)} / {mlp_feature_value('close_into_continue','21-30',3,'ranked_feature_replacement',256)}</b>。Close transport 明显弱于 continue transport，尤其高 count 的 argmax 翻转率仍低，说明没有一个上下文无关、正反完全对称的“继续/停止 neuron”。</p>
      {table(mlp_feature_rows,[('direction','patch 方向'),('bin','count 区间'),('layer','Layer'),('features','support'),('ranked_recovery','ranked replacement recovery'),('sparse_recovery','sparse direction recovery'),('random_recovery','random replacement recovery'),('ranked_accuracy','ranked target accuracy')])}
      <h4>分析</h4>
      <p><b>Layer 4 是主要的 feature-level decision writer。</b>Ranked held-out patches 大幅超过 size-matched random controls，证明这些坐标不是仅在训练样本上相关，而是在新 pair 上因果运输 continue/close evidence。Layer 3 whole-MLP mediation 在 8.4 中很强，但 post-GELU 坐标替换本身较弱，说明 Layer 3 更像依赖 receiver residual、LayerNorm 与 attention-MLP 联合状态的中间转换，不能被一个固定 feature 子集独立搬走。</p>
      <p><b>共享稀疏方向只解释一部分机制。</b>Layer 4 support-64 sparse direction 在 continue 方向达到 <b>{mlp_feature_value('continue_into_close','1-10',3,'sparse_mean_direction',64)} / {mlp_feature_value('continue_into_close','11-20',3,'sparse_mean_direction',64)} / {mlp_feature_value('continue_into_close','21-30',3,'sparse_mean_direction',64)}</b> recovery，但 close 方向只有 <b>{mlp_feature_value('close_into_continue','1-10',3,'sparse_mean_direction',64)} / {mlp_feature_value('close_into_continue','11-20',3,'sparse_mean_direction',64)} / {mlp_feature_value('close_into_continue','21-30',3,'sparse_mean_direction',64)}</b>。扩大到 dense mean direction 不总是继续改善，说明不同 k 与上下文中的有效 feature 组合并不完全共线；这更像条件化的非线性子空间，而不是一根统一的 <code>+1</code> 轴。</p>
      <div class="callout good"><b>feature 级因果结论：</b>从 attention routed evidence 到具体 next-index/close token 的最后转换主要落在 Layer 4 MLP 的数十至数百个 post-GELU features 上；它具有可重复的稀疏结构，但仍是分布式并依赖上下文的。当前实验已经超过“hidden direction 可读性”：selected held-out feature replacement 能稳定恢复 target margin；close 方向可稳定翻转决定，而 continue 方向的中高 count 只得到部分 margin recovery，说明仍需 receiver residual 或更广的 feature 组合共同越过决策边界。</div>
      <div class="callout limit"><b>证据边界：</b>feature 是网络坐标，不等于独立语义 neuron；排名使用 c_proj 与 unembedding 的线性近似，而真实 forward 还经过 residual 与最终 LayerNorm。结果来自单 seed、teacher-forced <code>M_k</code> query，也没有证明模型实现人类可读的符号加法。</div>

      <h3>8.6 Non-thinking broad heads：在最终 &lt;Ans&gt; 局部运输 donor count</h3>
      <h4>实验</h4>
      <p>对 nested receiver/donor pairs，在 non-thinking 模型最终 <code>&lt;Ans&gt;</code> query，仅 patch 按 broad score 排名的 donor head slices。Prompt token rows 不被替换，因此 donor count 若进入 receiver 输出，只能经由这些最终-query attention outputs 的局部变化发生。对每个 receiver count 区间、每个 top-n 独立拟合 expected-count transport slope。</p>
      <h4>结果</h4>
      {figure(
          generated['nested_nonthinking'],
          'Figure 6A. Non-thinking broad-head output 的 donor-count transport',
          '<b>横轴</b>是最终 <code>&lt;Ans&gt;</code> query 被替换的 donor head slices 数；<b>纵轴</b>是 expected-count shift 对 donor offset <code>m−n</code> 的回归 slope。0 表示不运输，1 表示一比一运输。三栏按 receiver count 分区；蓝线按 broad score 排名，灰色为随机路径，紫色为 random top-4 mean。'
      )}
      {table(direct_transport_rows,[('bin','receiver count 区间'),('heads','patched top-n'),('slope','broad-ranked slope'),('random','random mean'),('range','random min–max')])}
      <h4>分析</h4>
      <p>Broad-ranked top-1 已产生区间依赖的正 transport：1–10 / 11–20 / 21–30 的 slope 为 <b>{count_transport_lookup[('non-thinking broad','1-10',1)]['slope']}</b> / <b>{count_transport_lookup[('non-thinking broad','11-20',1)]['slope']}</b> / <b>{count_transport_lookup[('non-thinking broad','21-30',1)]['slope']}</b>。Top-4 后升至 <b>{count_transport_lookup[('non-thinking broad','1-10',4)]['slope']}</b> / <b>{count_transport_lookup[('non-thinking broad','11-20',4)]['slope']}</b> / <b>{count_transport_lookup[('non-thinking broad','21-30',4)]['slope']}</b>，而随机 top-4 均值仅为 <b>{count_transport_lookup[('non-thinking broad','1-10',4)]['random']}</b> / <b>{count_transport_lookup[('non-thinking broad','11-20',4)]['random']}</b> / <b>{count_transport_lookup[('non-thinking broad','21-30',4)]['random']}</b>。</p>
      <div class="callout good"><b>因果结论：</b>non-thinking broad heads 不只“看着 needles”，其最终-query outputs 足以搬运相当比例的 donor count。低 count 的 top-4 几乎达到一比一 transport；随着 receiver count 增大，固定四枚 heads 的 slope 下降到 0.53，说明高 count 需要更多分布式 head slices 或更多 residual/MLP 支持。Top-16 达到 1 是“全部 attention output 都来自 donor”的 sanity endpoint，不能被解释为 broad ranking 的特异性。</div>

      <h3>8.7 Non-thinking MLP intermediate：集合证据如何变成具体 count</h3>
      <h4>实验</h4>
      <p>这一实验把第 8.6 节的链条再向后推进一步。Receiver prompt 含 <code>n</code> 个 needles，donor prompt 含 <code>m=n±3</code> 个 needles；两者使用 nested prompt，使 receiver 的 needle 集合严格包含于 donor 或反之。我们在最终 <code>&lt;Ans&gt;</code> query、每个 Layer 的 MLP 中间层读取经过激活函数后的 <b>1024 维 post-GELU activation</b>，而不是读取完整 residual。</p>
      <div class="protocol"><ol>
        <li><b>Fit split。</b>对每个 feature <code>i</code> 计算 <code>(a_i(m)−a_i(n)) × w_i(m versus n)</code>。其中 <code>a_i</code> 是 post-GELU activation；<code>w_i</code> 是该 feature 经 <code>c_proj</code> 与 tied unembedding 后，对 donor-count logit 相对 receiver-count logit 的线性系数。按跨 fit pairs 的平均 signed evidence 从大到小排序。</li>
        <li><b>Held-out causal split。</b>Fit 与 evaluation 使用不相交的随机 seeds。在 receiver forward 中，只把排名 top-8 / 32 / 128 / 512 / 1024 个 post-GELU features 替换成 donor 值；其余 features、attention、residual 和后续 Layers 都保留 receiver 值。每个 support size 另采样 {int(final_bridge_manifest['random_replicates'])} 组等大小 random features。</li>
        <li><b>Transport slope。</b>对 held-out pairs 回归 <code>ΔE[C] = a + b(m−n)</code>。<code>b=1</code> 表示 patched output 一比一跟随 donor offset；<code>b=0</code> 表示没有数值运输。另报告 <code>R²</code>、donor-vs-receiver logit-margin recovery，以及 argmax 是否真正翻转到 donor count。</li>
      </ol></div>
      <h4>结果</h4>
      {figure(
          generated['nonthinking_mlp_transport'],
          'Figure 6B. Non-thinking post-GELU feature transplant 的 count transport',
          '<b>横轴</b>是被 donor 值替换的 post-GELU features 数量；<b>纵轴</b>是 expected-count transport slope。三栏为 receiver count 1–10、11–20、21–30；每条彩线是一层的 ranked features，灰色虚线是理想的一比一 transport。完整 1024-feature replacement 是整层 MLP intermediate transplant 的充分性 sanity check，而不是稀疏选择优势。'
      )}
      {table(nonthinking_feature_rows,[('bin','receiver count 区间'),('selection','选择'),('layer','Layer'),('features','features'),('ranked_slope','ranked slope'),('random_slope','matched random slope'),('gain','slope gain'),('r2','ranked R²'),('margin_recovery','margin recovery'),('donor_follow','argmax follows donor')])}
      <h4>分析</h4>
      <p>低 count 的 Layer-1 top-128 已达到 slope <b>{fmt(nonthinking_feature_comparison[(nonthinking_feature_comparison.count_bin == '1-10') & (nonthinking_feature_comparison.layer == 0) & (nonthinking_feature_comparison.support_size == 128)].iloc[0].transport_slope)}</b>、R² <b>{fmt(nonthinking_feature_comparison[(nonthinking_feature_comparison.count_bin == '1-10') & (nonthinking_feature_comparison.layer == 0) & (nonthinking_feature_comparison.support_size == 128)].iloc[0].transport_r2)}</b>，明显高于 matched-random slope <b>{fmt(nonthinking_feature_comparison[(nonthinking_feature_comparison.count_bin == '1-10') & (nonthinking_feature_comparison.layer == 0) & (nonthinking_feature_comparison.support_size == 128)].iloc[0].random_transport_slope)}</b>；替换完整 Layer-1 1024 features 时 slope 为 <b>{fmt(nonthinking_feature_comparison[(nonthinking_feature_comparison.count_bin == '1-10') & (nonthinking_feature_comparison.layer == 0) & (nonthinking_feature_comparison.support_size == 1024)].iloc[0].transport_slope)}</b>、R² 为 <b>{fmt(nonthinking_feature_comparison[(nonthinking_feature_comparison.count_bin == '1-10') & (nonthinking_feature_comparison.layer == 0) & (nonthinking_feature_comparison.support_size == 1024)].iloc[0].transport_r2)}</b>，输出完整翻到 donor。</p>
      <p>11–20 中最强稀疏优势出现在 Layer 4 top-512：ranked slope <b>0.475</b>，matched-random slope <b>−0.643</b>；Layer 3 top-512 的 ranked slope 更高，达到 <b>0.721</b>。21–30 的稀疏运输则整体很弱，最佳 sparse slope 仅约 <b>0.092</b>，完整 Layer-4 MLP transplant 也只有约 <b>0.264</b>。这与高 count state 更分布式、更依赖跨层 residual 的结论一致。</p>
      <div class="callout good"><b>因果结论：</b>non-thinking 的“needle-set evidence → numeric count”不只是线性可读。Held-out donor MLP features 被写入 receiver 后，低/中 count 的连续 count logits 按 donor offset 系统移动；ranked sparse features 在多个条件下明显胜过等大小随机 features。这把 broad-head transport 与后层 count state 之间补上了一段局部因果桥。</div>
      <div class="callout warn"><b>边界：</b>这仍不是一枚“加一 neuron”。排名依赖 donor-vs-receiver logit 方向，最佳 Layer/support 随 count 区间变化；1024-feature 结果是整层充分性检查。高 count 的弱 sparse transport 表明表示可能分散在 attention、多个 MLP Layers 和 residual 中。</div>

      <h3>8.8 CoT trace-readout heads：最终 &lt;Ans&gt; 的 attention slices 单独不足以搬运 count</h3>
      <h4>实验</h4>
      <p>使用同一 nested donor/receiver protocol，但换成 thinking 模型，并按最终 <code>&lt;Ans&gt;</code> query 对全部 trace markers 的 attention mass 排名。Donor 与 receiver 的 gold trace 长度分别为 m 与 n，所以 semantic anchor 相同而绝对 position 不同。只替换 attention head-output slices；receiver 的 residual stream、MLP 输入与其余上下文仍保留 receiver 状态。</p>
      <h4>结果</h4>
      {figure(
          generated['nested_thinking'],
          'Figure 6B. CoT trace-readout head slices 的 donor-count transport',
          '<b>横轴</b>是最终 <code>&lt;Ans&gt;</code> query 被替换的 donor slices 数；<b>纵轴</b>仍是 expected-count transport slope。三栏为 receiver count 区间。若 trace-readout heads 自身携带完整标量 count，蓝线应随 top-n 靠近 1；实际曲线大多接近 0。'
      )}
      {table(cot_transport_rows,[('bin','receiver count 区间'),('heads','patched top-n'),('slope','readout-ranked slope'),('random','random mean'),('range','random min–max')])}
      <h4>分析</h4>
      <p>CoT top-4 readout slices 的 slope 只有 <b>{count_transport_lookup[('CoT trace readout','1-10',4)]['slope']}</b> / <b>{count_transport_lookup[('CoT trace readout','11-20',4)]['slope']}</b> / <b>{count_transport_lookup[('CoT trace readout','21-30',4)]['slope']}</b>；即使 top-8 也只有 <b>{count_transport_lookup[('CoT trace readout','1-10',8)]['slope']}</b> / <b>{count_transport_lookup[('CoT trace readout','11-20',8)]['slope']}</b> / <b>{count_transport_lookup[('CoT trace readout','21-30',8)]['slope']}</b>，而且剂量曲线非单调。</p>
      <div class="callout warn"><b>解释：</b>这不等于“CoT trace readout 无用”。它说明最终答案的 count state 不能由 selected attention slices 单独搬运；receiver residual、位置/trace-length 表征、MLP 或多头组合仍占主导。第 7 节的局部 ablation 可以证明某些 readout heads 被使用，本节则表明这些 slices 本身不具备把 m−n 一比一写入 logits 的充分性。完整 residual transplant 才是对 CoT final count state 的更合适充分性测试。</div>

      <h3>8.9 CoT source-conflict：最终答案跟 prompt、trace，还是最后一个 trace 数字？</h3>
      <h4>实验</h4>
      <p>我们构造 receiver count <code>n</code> 与 donor count <code>m=n±3</code> 的冲突序列，并保持模型权重不变。以 <code>n=5, m=8</code> 为例，四个条件分别是：</p>
      <div class="protocol"><ol>
        <li><b>prompt=m, trace=n。</b>Prompt 里有 8 个 needles，但 teacher-forced trace 只有 5 组 <code>&lt;k&gt;, M_k</code>；检验最终答案是重读 prompt 还是跟 trace。</li>
        <li><b>prompt=n, trace=m。</b>Prompt 里有 5 个 needles，trace 却有 8 组；检验 trace extent 是否能覆盖 prompt evidence。</li>
        <li><b>final trace index=m only。</b>Prompt、trace marker 数量与 trace 总长度都对应 5，只把最后一个 trace 数字改成 <code>&lt;8&gt;</code>；检验模型是否简单复制最后一个数字。</li>
        <li><b>marker-identity control。</b>保持 prompt count、trace pair 数和数字不变，只循环置换 trace 中的 marker identities；检验 final count 是否依赖具体 marker 内容。</li>
      </ol></div>
      <p>每个 count 区间使用两个有向 count pairs（5↔8、15↔18、25↔28），每个 pair 有 {int(final_bridge_manifest['behavior_examples_per_pair'])} 个 prompts，因此每个 condition×bin 有 12 个 held-out conflict examples。<code>follows receiver/donor</code> 表示 30-way count argmax 等于哪一个 count；target margin 是预注册 target count logit 减去其余 29 个 count logits 的最大值。</p>
      <h4>结果</h4>
      {figure(
          generated['cot_conflict_source'],
          'Figure 6C. CoT final answer 的信息源冲突',
          '<b>横轴</b>是三种真正改变候选 count 来源的 conflict；<b>纵轴</b>是最终 count argmax 跟随 receiver 或 donor 的比例。每栏是一个 count 区间。Marker-identity control 不改变目标 count，故列入下表而不混入二选一图。'
      )}
      {table(conflict_rows,[('condition','冲突条件'),('bin','count 区间'),('receiver','follows receiver'),('donor','follows donor'),('target','target correct'),('margin','target logit margin'),('abs_position','mean |Ans-position shift|')])}
      <h4>分析</h4>
      <p>三个区间的行为完全一致：<code>prompt=m, trace=n</code> 时模型 <b>100%</b> 输出 n；<code>prompt=n, trace=m</code> 时 <b>100%</b> 输出 m；只改最后 trace 数字时仍 <b>100%</b> 输出 n；置换 marker identity 也不改变答案。因而当前模型的最终答案不靠重新统计 prompt，也不是复制最后一个 trace 数字或读取 marker identity；最强行为证据指向<b>完整 trace 的 pair 数/extent</b>。</p>
      <div class="callout warn"><b>关键混淆变量：</b>模型使用 learned absolute-position embeddings。Trace 增减 3 pairs 会让最终 <code>&lt;Ans&gt;</code> 绝对位置移动 6 tokens；正负方向平均后 signed shift 恰好为 0，所以表中必须看 <code>mean |Ans-position shift|=6</code>。因此本实验排除了 prompt 重读与 final-index copy，但还不能区分“语义上的 trace 长度计数”与“从 <code>&lt;Ans&gt;</code> 的绝对位置读出 count”。</div>

      <h3>8.10 CoT final scalar bridge：哪一层把 trace/position evidence 写成可执行 count state？</h3>
      <h4>实验</h4>
      <p>对同一个 <code>prompt=m, trace=n</code> conflict，先运行 clean donor（prompt 与 trace 都为 m）并缓存最终 <code>&lt;Ans&gt;</code> row 的 activation；再运行 conflict receiver（prompt=m、trace=n，基线输出 n）。我们只在 receiver 的语义 <code>&lt;Ans&gt;</code> row 替换 clean donor activation，测 donor count m 的 logit-margin recovery。由于 donor 与 receiver 的 <code>&lt;Ans&gt;</code> 绝对位置不同，这个 patch 同时问：clean component 是否足以把 donor 的 trace/position-derived count evidence带到 receiver。</p>
      <div class="protocol"><ol>
        <li><b>Attention output。</b>替换某一 Layer 完整 attention sublayer 的输出，但保留 receiver MLP 与 residual。</li>
        <li><b>MLP output。</b>替换该 Layer MLP 投回 residual 的输出；attention 与 MLP 输入仍来自 receiver。</li>
        <li><b>Post-MLP residual。</b>替换该 Layer 完成 attention、MLP 与 residual additions 后的整条 256 维 state；后续 Layers 继续在 receiver context 中运行。</li>
        <li><b>Normalized recovery。</b><code>(patched donor-margin − conflict margin) / (clean donor-margin − conflict margin)</code>。1 表示完全恢复 clean donor margin；另报告 argmax 是否真正翻到 donor target。</li>
      </ol></div>
      <h4>结果</h4>
      {figure(
          generated['cot_final_bridge'],
          'Figure 6D. CoT final &lt;Ans&gt; 的 sublayer/residual clean-to-conflict recovery',
          '<b>横轴</b>是 Layer 1–4；<b>纵轴</b>是 donor-count normalized recovery。三种线分别替换完整 attention output、MLP output 与 post-MLP residual；三栏为 count 区间。虚线 1 表示恢复到 clean donor margin。'
      )}
      {table(bridge_rows,[('bin','count 区间'),('component','Layer-1 component'),('recovery','normalized recovery'),('target_follow','argmax follows donor')])}
      <h4>分析</h4>
      <p>Layer-1 attention output 的 recovery 在 1–10 / 11–20 / 21–30 仅为 <b>{fmt(bridge_value('1-10','attn_out',0))}</b> / <b>{fmt(bridge_value('11-20','attn_out',0))}</b> / <b>{fmt(bridge_value('21-30','attn_out',0))}</b>，从不把 argmax 翻到 donor。相反，Layer-1 MLP output 已恢复 <b>{fmt(bridge_value('1-10','mlp_out',0))}</b> / <b>{fmt(bridge_value('11-20','mlp_out',0))}</b> / <b>{fmt(bridge_value('21-30','mlp_out',0))}</b>，三个区间的 argmax 都 <b>100%</b> 跟随 donor；Layer-1 post-MLP residual 进一步达到 <b>{fmt(bridge_value('1-10','post_mlp',0))}</b> / <b>{fmt(bridge_value('11-20','post_mlp',0))}</b> / <b>{fmt(bridge_value('21-30','post_mlp',0))}</b>。</p>
      <p>固定的 prompt-broad、trace-index 或 trace-marker head bundles 在同一位置的 recovery 很弱，而且没有翻转 target；这与 8.8 的 readout-slice 结果一致：最终 scalar state 不是由少数固定 attention slices 单独携带。</p>
      <div class="callout good"><b>因果结论：</b>在当前 learned-position 模型中，CoT 的 trace/position-derived evidence 最晚在 Layer-1 MLP 后已经成为可执行的 donor scalar-count state。完整 Layer-1 MLP output 单独就足以翻转 count；完整 post-MLP residual 几乎完全恢复。这个结果补上了“trace/context → final scalar state”的局部桥，但没有消除 trace length 与绝对位置的混淆。</div>
      <div class="callout warn"><b>定位边界：</b>post-MLP residual 包含该 Layer 以前的全部计算，所以高 recovery 不能单独证明 count 首次诞生于 Layer 1。更有定位力的证据是：Layer-1 attention output 弱，而 Layer-1 MLP output 强；它支持 MLP 把已有的 trace/position evidence转换为 count logits 所需方向，但不说明 MLP 内部采用了符号式加法。</div>

      <h3>8.11 本节结论与证据边界</h3>
      <div class="mechanisms">
        <div class="mechanism"><h3>Non-thinking</h3><p><b>当前支持：</b>early broad heads 既表现出必要性，也能在最终 <code>&lt;Ans&gt;</code> 局部 patch 中运输 donor count；held-out post-GELU feature transplant 又证明低/中 count 的一部分 numeric state 可由特定 MLP feature 子集搬运。因此“broad set aggregation → MLP count conversion → answer state”已有连续的局部因果证据。</p><p><b>边界：</b>高 count 的 fixed-head 与 sparse-feature transport 都明显变弱，说明 state 分布到更多 heads、Layers 与 residual；尚未得到一套跨 1–30 通用的算术 feature basis。</p></div>
        <div class="mechanism"><h3>CoT</h3><p><b>当前支持：</b>数字 <code>&lt;k&gt;</code> row 的 targeted bundle 恢复 marker identity；marker <code>M_k</code> row 的 successor bundle 运输 next-step evidence；MLP features 写入 continue/close logits。最终答案在冲突中跟 trace pair count/extent，而不是 prompt count、marker identity或最后 trace 数字；clean Layer-1 MLP output 和 post-MLP residual 足以把 donor scalar count 写入 conflict receiver。</p><p><b>仍缺：</b>trace pair 数与 <code>&lt;Ans&gt;</code> 绝对位置共变，尚未区分语义 trace-length counting 与 learned-position shortcut；也没有把 final scalar bridge 压缩为固定少数 heads 或一根统一 count direction。</p></div>
      </div>
    </section>
    """

    steering_section = f"""
    <section id="steering">
      <h2>9. Hidden-state geometry steering：相邻 transplant、非相邻直线与 centroid 曲线</h2>
      <div class="callout good"><b>本节问题。</b>第 6 节只说明 exact-count centroids 在 residual space 中形成可读轨迹；这里直接改写 held-out 样本的 256 维 residual，再让模型完成剩余前向，检验这条轨迹是否真的控制 count logits。核心比较不是“有没有一根相关方向”，而是：完整 centroid、保留样本残差的局部位移、端点直线和沿相邻 centroids 的曲线路径，哪一种能把输出搬到指定 count。</div>

      <h3>9.1 共同设置、状态分解与结果量</h3>
      <h4>实验</h4>
      <p>Centroids 来自独立的 direction-train split，不使用本节干预样本。对 semantic site <code>s</code>、Layer <code>l</code> 和 exact count <code>c</code>，先平均 256 维 residual，得到 <code>mu[s,l,c] = E[h | s,l,count=c]</code>。本节使用 {geometry_path_examples_per_count} 个 held-out prompts / exact count，即每个 site 有 {30 * geometry_path_examples_per_count} 个 receiver prompts；在三个 sites、四层和所有合法 donor offsets 上共记录 <b>{len(geometry_path_detail):,}</b> 条真实 patched-forward 行。</p>
      <div class="protocol"><ol>
        <li><b>三个 sites。</b><code>nonthinking_final_answer</code> 与 <code>thinking_final_answer</code> 都在自然序列的最终 <code>&lt;Ans&gt;</code> query 取 residual。<code>thinking_fixed_trace_answer</code> 则强制所有 prompt 使用同一条 15-step gold-format trace，只有 prompt needle count 与答案标签变化；它用于区分“prompt-derived count geometry 可读”与“该 geometry 真正控制最终答案”。</li>
        <li><b>Receiver 分解。</b>把 receiver state 写成 <code>h_r = mu_r + epsilon</code>，其中 <code>epsilon</code> 包含该具体 prompt 相对 exact-count 均值的残差。Full transplant 使用 <code>h' = mu_t</code>；delta transport 使用 <code>h' = h_r + (mu_t - mu_r) = mu_t + epsilon</code>。二者若结果相同，说明保留 prompt-specific residual 不妨碍 count 搬运。</li>
        <li><b>连续输出。</b>从 count-token logits 计算 <code>E[C] = sum_c c * softmax(z_count)[c]</code>。对每个 panel 回归 <code>Delta E[C] = a + b * Delta c_path</code>；<code>b=1</code> 表示输出一比一跟随指定路径坐标。<code>path-tracking MAE = mean |E_patch[C] - c_path|</code>，越接近 0 越好。</li>
        <li><b>Layer 含义。</b>“after Layer 1–4”表示替换该 Transformer Layer 输出的 residual。较早层后仍有后续 Layers 可修正或抵消干预；after Layer 4 后只剩最终 LayerNorm 与 tied unembedding，因此更直接检验该 residual 是否已是可执行 count state。</li>
      </ol></div>

      <h3>9.2 相邻 count：完整 centroid transplant 与 residual-preserving delta</h3>
      <h4>实验</h4>
      <p>对 receiver count <code>r</code>，分别选择相邻 donor <code>t=r-1</code> 或 <code>r+1</code>。Full transplant 删除 receiver 的 <code>epsilon</code>，直接写入 <code>mu_t</code>；delta transport 只增加局部 centroid 差 <code>mu_t-mu_r</code>，保留 receiver prompt 的其余状态。每个 count 区间独立拟合 transport slope。</p>
      <h4>结果</h4>
      {figure(
          generated['geometry_path_adjacent'],
          'Figure 7A. 相邻 exact-count centroid 的 full transplant 与 delta transport',
          '<b>横轴</b>是 residual 被干预在 Layer 1–4 之后；<b>纵轴</b>是 <code>Delta E[C]</code> 对目标相邻 count shift（±1）的回归 slope。蓝色实线=直接写入目标 centroid；橙色虚线=保留 receiver residual 的 centroid delta。灰虚线 1 是理想一比一搬运。三行是 non-thinking natural、CoT natural、CoT fixed-15 control；三列是 receiver count 1–10、11–20、21–30。'
      )}
      {table(adjacent_transport_rows,[('site','semantic site'),('bin','receiver count 区间'),('full_slope','Layer 4 full-transplant slope'),('delta_slope','Layer 4 delta slope'),('delta_r2','Layer 4 delta R²'),('delta_mae','Layer 4 path MAE')])}
      <h4>分析</h4>
      <p><b>自然 CoT 的 final-answer state 在所有深度都已强因果化。</b>三个 count 区间、四个 Layers 的相邻 delta slope 均为 1.000，tracking MAE 为 0；既可以把 state 完整换成相邻 centroid，也可以只加局部差向量，模型都会把 count 输出精确移动一格。</p>
      <p><b>Non-thinking 是逐层形成的。</b>在 21–30，Layer 1 后的 delta slope 只有 <b>{fmt(path_metric('nonthinking_final_answer','21-30','adjacent_delta_transport','transport_slope',layer=0))}</b>，Layer 3 后为 <b>{fmt(path_metric('nonthinking_final_answer','21-30','adjacent_delta_transport','transport_slope',layer=2))}</b>，到 Layer 4 后才达到 <b>{fmt(path_metric('nonthinking_final_answer','21-30','adjacent_delta_transport','transport_slope',layer=3))}</b>。这与“early broad heads 收集集合证据、后层 residual/MLP 才写成明确 count state”的机制相容。</p>
      <p><b>Fixed-15 是关键负对照。</b>低/中 count 的 Layer 4 delta slope 都为 <b>0.000</b>；即使 hidden state 在第 6 节仍能按 prompt count 分类，该方向也不能改写由固定 trace 主导的最终 logits。可读 geometry 因此不自动等于被模型用作答案控制变量。</p>

      <h3>9.3 非相邻 count：直接沿端点连线移动</h3>
      <h4>实验</h4>
      <p>对合法 offset <code>t-r in {{±2,±3,±5,±10}}</code>，定义端点 chord：<code>p_chord(alpha) = (1-alpha) mu_r + alpha mu_t</code>，并写入 <code>h' = h_r + p_chord(alpha) - mu_r</code>。<code>alpha</code> 取 0.25、0.5、0.75、1；它检验“把两个非相邻 count centroid 直接连成一条直线”是否仍留在模型可执行的 count manifold 上。</p>
      <h4>结果</h4>
      {figure(
          generated['geometry_path_chord'],
          'Figure 7B. 非相邻 count 的端点直线（chord）transport',
          '<b>横轴</b>是干预 Layer；<b>纵轴</b>是在所有 donor offsets 上拟合的 transport slope。颜色表示 chord 进度 alpha；虚线 1 为理想路径跟随。上行为 non-thinking natural，下行为 CoT natural；三列为 receiver count 区间。alpha=1 只测试终点，alpha<1 才测试直线内部。'
      )}
      <h4>分析</h4>
      <p>端点通常可以到达，但中间 chord 点并不稳定对应线性 count。以 Layer 4、alpha=0.5 为例，non-thinking 三段 tracking MAE 为 <b>{path_metric('nonthinking_final_answer','1-10','nonadjacent_chord_transport','path_tracking_mae',alpha=.5):.3f} / {path_metric('nonthinking_final_answer','11-20','nonadjacent_chord_transport','path_tracking_mae',alpha=.5):.3f} / {path_metric('nonthinking_final_answer','21-30','nonadjacent_chord_transport','path_tracking_mae',alpha=.5):.3f}</b>；CoT 为 <b>{path_metric('thinking_final_answer','1-10','nonadjacent_chord_transport','path_tracking_mae',alpha=.5):.3f} / {path_metric('thinking_final_answer','11-20','nonadjacent_chord_transport','path_tracking_mae',alpha=.5):.3f} / {path_metric('thinking_final_answer','21-30','nonadjacent_chord_transport','path_tracking_mae',alpha=.5):.3f}</b>。这说明 exact-count endpoints 之间的欧氏捷径会穿过非自然区域。</p>

      <h3>9.4 非相邻 count：沿相邻 centroids 的 piecewise curve 移动</h3>
      <h4>实验</h4>
      <p>把 <code>mu_r, mu_(r±1), ..., mu_t</code> 依 count 顺序连接成折线，并按 residual-space arc length 归一化。<code>p_curve(alpha)</code> 是走过总弧长 alpha 比例时的点；干预仍为 <code>h' = h_r + p_curve(alpha) - mu_r</code>。因此 curve 与 chord 在 <code>alpha=0</code>、<code>alpha=1</code> 完全相同，只在中间路径不同。</p>
      <h4>结果</h4>
      {figure(
          generated['geometry_path_curve'],
          'Figure 7C. 沿相邻 centroid 折线的非相邻 count transport',
          '<b>横纵轴、行列与 alpha 颜色</b>同 Figure 7B，但 intervention point 改为沿相邻 exact-count centroids 的归一化弧长路径。若 count geometry 是弯曲 manifold，该方法应比端点直线更接近 slope=1。'
      )}
      {table(nonadjacent_path_rows,[('site','natural site'),('bin','receiver count 区间'),('chord_slope','Layer 4 chord slope (alpha=.5)'),('chord_r2','chord R²'),('chord_mae','chord MAE'),('curve_slope','Layer 4 curve slope (alpha=.5)'),('curve_r2','curve R²'),('curve_mae','curve MAE')])}
      <h4>分析</h4>
      <p>Layer 4 的 curve 在六个 natural site×count-bin panels 中，alpha=0.5 slope 都接近 1（范围约 0.985–1.007），tracking MAE 只有 0.109–0.139；对应 chord MAE 为 0.842–1.410。也就是说，模型不只接受目标 centroid，连相邻 centroids 所定义的中间路径也映射到正确的中间 count；直接端点连线则显著偏离。</p>

      <h3>9.5 曲线是否真的优于直线，以及终点 sanity check</h3>
      <h4>结果</h4>
      {figure(
          generated['geometry_path_tracking'],
          'Figure 7D. Curve 与 chord 在中间 alpha 的 path-tracking error',
          '<b>横轴</b>是干预 Layer；<b>纵轴</b>是 alpha<1、所有合法非相邻 offsets 上的平均绝对 path-tracking error。橙虚线=端点 chord，绿实线=相邻-centroid curve；越低越好。上行为 non-thinking，下行为 CoT；三列为 count 区间。'
      )}
      <p>代码级终点检查覆盖 <b>{int(endpoint_sanity['rows']):,}</b> 条 alpha=1 curve/chord 对：两种路径的预测 count 一致率为 <b>{pct(endpoint_sanity['prediction_agreement'])}</b>，expected-count 最大绝对差仅 <b>{float(endpoint_sanity['max_abs_expected_difference']):.2e}</b>。因此 curve 的优势不是终点或实现不一致造成，而来自中间路径本身。</p>
      <h4>分析</h4>
      <div class="callout good"><b>最强结论。</b>自然 non-thinking 与 CoT final-answer residual 都支持 exact-count centroid transport，但其几何不是全局笔直轴：局部相邻 delta 有效，长距离沿相邻 centroid 曲线也近乎一比一，而端点 chord 在中间 alpha 明显偏离。Goodfire-style steering 在这里更适合被描述为<b>沿一条 count-state manifold 做局部 transport</b>，而不是向任意 hidden state 加一根全局 <code>+n</code> 向量。</div>
      <div class="callout warn"><b>证据边界。</b>Centroids 是同 count 样本均值，curve 是离散折线而非模型显式学习的连续坐标；本节只有单 seed、每个 exact count {geometry_path_examples_per_count} 个 held-out receiver prompts。它证明这些 residual interventions 对当前模型输出具有因果控制力，但不证明模型在线前向时真的沿同一折线逐步移动。</div>
    </section>
    """

    def hidden_patch_value(
        frame: pd.DataFrame,
        *,
        count_bin: str,
        layer: int,
        metric: str,
        mode: str | None = None,
    ) -> str:
        rows = frame[(frame["count_bin"] == count_bin) & (frame["layer"] == layer)]
        if mode is not None:
            rows = rows[rows["mode"] == mode]
        return fmt(rows.iloc[0][metric]) if len(rows) else "n/a"

    hidden_final_rows: list[dict[str, object]] = []
    for mode, mode_label in (("nonthinking", "Non-thinking"), ("thinking", "CoT")):
        for count_bin in COUNT_BINS:
            hidden_final_rows.append(
                {
                    "mode": mode_label,
                    "bin": count_bin,
                    "l1_slope": hidden_patch_value(
                        hidden_final_summary,
                        mode=mode,
                        count_bin=count_bin,
                        layer=1,
                        metric="transport_slope",
                    ),
                    "l3_slope": hidden_patch_value(
                        hidden_final_summary,
                        mode=mode,
                        count_bin=count_bin,
                        layer=3,
                        metric="transport_slope",
                    ),
                    "l4_slope": hidden_patch_value(
                        hidden_final_summary,
                        mode=mode,
                        count_bin=count_bin,
                        layer=4,
                        metric="transport_slope",
                    ),
                    "l4_donor": pct(
                        select_row(
                            hidden_final_summary,
                            mode=mode,
                            count_bin=count_bin,
                            layer=4,
                        )["mean_follows_donor"]
                    ),
                }
            )

    hidden_trace_rows = [
        {
            "bin": count_bin,
            "l1_slope": hidden_patch_value(
                hidden_trace_final_summary,
                count_bin=count_bin,
                layer=1,
                metric="transport_slope",
            ),
            "l4_slope": hidden_patch_value(
                hidden_trace_final_summary,
                count_bin=count_bin,
                layer=4,
                metric="transport_slope",
            ),
            "l4_receiver": pct(
                select_row(hidden_trace_final_summary, count_bin=count_bin, layer=4)[
                    "mean_follows_receiver"
                ]
            ),
        }
        for count_bin in COUNT_BINS
    ]
    hidden_early_rows = [
        {
            "bin": count_bin,
            "l3_close": pct(
                select_row(hidden_early_summary, count_bin=count_bin, layer=3)[
                    "mean_close_predicted"
                ]
            ),
            "l4_close": pct(
                select_row(hidden_early_summary, count_bin=count_bin, layer=4)[
                    "mean_close_predicted"
                ]
            ),
            "l4_margin": hidden_patch_value(
                hidden_early_summary,
                count_bin=count_bin,
                layer=4,
                metric="mean_close_margin_shift",
            ),
            "l4_slope": hidden_patch_value(
                hidden_early_summary,
                count_bin=count_bin,
                layer=4,
                metric="transport_slope",
            ),
        }
        for count_bin in COUNT_BINS
    ]

    upward_transport_rows: list[dict[str, object]] = []
    for mode, label, frame in (
        ("nonthinking", "Non-thinking final <Ans>", hidden_final_directional),
        ("thinking", "CoT final <Ans>", hidden_final_directional),
        ("thinking", "CoT final trace marker", hidden_trace_final_directional),
    ):
        selected = frame[frame["patch_direction"] == "donor_gt_receiver"]
        if "mode" in selected.columns:
            selected = selected[selected["mode"] == mode]
        for count_bin in COUNT_BINS:
            row: dict[str, object] = {"site": label, "bin": count_bin}
            for layer in (1, 2, 3, 4):
                matches = selected[
                    (selected["count_bin"] == count_bin)
                    & (selected["layer"] == layer)
                ]
                row[f"l{layer}"] = (
                    fmt(matches.iloc[0]["transport_slope"]) if len(matches) else "n/a"
                )
            upward_transport_rows.append(row)

    rollout_intervention_labels = {
        "baseline_no_patch": "无 patch",
        "same_progress_donor_total_full": "自然 donor：只改 total",
        "later_progress_donor_full": "自然 donor：total 与 progress 一起改",
        "total_only_centroid_delta": "centroid delta：只改 total",
        "progress_only_centroid_delta": "centroid delta：只改 progress",
        "combined_centroid_delta": "centroid delta：total + progress",
    }
    rollout_rows: list[dict[str, object]] = []
    scenario_view = hidden_rollout_summary[
        (hidden_rollout_summary["patch_policy"] == "one_shot")
        & (hidden_rollout_summary["layer"] == 4)
        & (hidden_rollout_summary["intervention"] == "later_progress_donor_full")
    ].copy()
    for match in scenario_view.itertuples(index=False):
        rollout_rows.append(
            {
                "family": str(match.scenario_family),
                "pair": (
                    f"R{int(match.receiver_count)}@I{int(match.receiver_progress)} ← "
                    f"D{int(match.donor_count)}@I{int(match.donor_progress)}"
                ),
                "first": fmt(match.mean_first_generated_index),
                "stop": fmt(match.mean_stop_index),
                "final": fmt(match.mean_final_count),
                "receiver": f"I{int(match.receiver_progress) + 1} / I{int(match.receiver_count)} / C{int(match.receiver_count)}",
                "donor": f"I{int(match.donor_progress) + 1} / I{int(match.donor_count)} / C{int(match.donor_count)}",
            }
        )

    rollout_factor_rows: list[dict[str, object]] = []
    factor_view = hidden_rollout_factors[
        (hidden_rollout_factors["patch_policy"] == "one_shot")
        & (hidden_rollout_factors["layer"] == 4)
        & hidden_rollout_factors["intervention"].isin(
            [
                "later_progress_donor_full",
                "total_only_centroid_delta",
                "progress_only_centroid_delta",
                "combined_centroid_delta",
            ]
        )
    ]
    for row in factor_view.itertuples(index=False):
        rollout_factor_rows.append(
            {
                "intervention": rollout_intervention_labels.get(
                    str(row.intervention), str(row.intervention)
                ),
                "outcome": str(row.outcome),
                "b_total": fmt(row.donor_total_coefficient),
                "b_progress": fmt(row.donor_progress_coefficient),
                "r2": fmt(row.r2),
                "n": int(row.n_scenarios),
            }
        )

    rollout_primary = hidden_rollout_summary[
        (hidden_rollout_summary["patch_policy"] == "one_shot")
        & (hidden_rollout_summary["layer"] == 4)
        & (hidden_rollout_summary["intervention"] == "progress_only_centroid_delta")
    ].copy()
    rollout_distinct_total = rollout_primary[
        rollout_primary["receiver_count"] != rollout_primary["donor_count"]
    ]
    rollout_total_only = hidden_rollout_summary[
        (hidden_rollout_summary["patch_policy"] == "one_shot")
        & (hidden_rollout_summary["layer"] == 4)
        & hidden_rollout_summary["intervention"].isin(
            ["same_progress_donor_total_full", "total_only_centroid_delta"]
        )
    ].copy()

    def _weighted_rate(frame: pd.DataFrame, rate_col: str) -> float:
        weights = frame["n"].astype(float)
        return float((frame[rate_col].astype(float) * weights).sum() / weights.sum())

    rollout_donor_successor = _weighted_rate(
        rollout_primary, "donor_successor_rate"
    )
    rollout_closed = _weighted_rate(rollout_primary, "closed_rate")
    rollout_total_only_receiver = _weighted_rate(
        rollout_total_only, "receiver_total_hit"
    )
    rollout_total_only_closed = _weighted_rate(rollout_total_only, "closed_rate")
    rollout_distinct_answer_n = float(
        (
            rollout_distinct_total["ans_rate"].astype(float)
            * rollout_distinct_total["n"].astype(float)
        ).sum()
    )
    rollout_distinct_receiver_given_answer = float(
        (
            rollout_distinct_total["receiver_total_hit"].astype(float)
            * rollout_distinct_total["n"].astype(float)
        ).sum()
        / rollout_distinct_answer_n
    )
    rollout_distinct_donor_given_answer = float(
        (
            rollout_distinct_total["donor_total_hit"].astype(float)
            * rollout_distinct_total["n"].astype(float)
        ).sum()
        / rollout_distinct_answer_n
    )
    rollout_progress_factor = factor_view[
        (factor_view["intervention"] == "progress_only_centroid_delta")
        & (factor_view["outcome"] == "first_index_shift")
    ].iloc[0]
    rollout_final_factor = factor_view[
        (factor_view["intervention"] == "progress_only_centroid_delta")
        & (factor_view["outcome"] == "final_count_shift")
    ].iloc[0]

    transplant_section = f"""
    <section id="transplant">
      <h2>10. Hidden-state patching：跨 sequence 搬运 count state 与 trace progress</h2>
      <div class="callout good"><b>本节问题。</b>第 6 节表明 count 可从 residual 读出，第 9 节表明 centroid 路径可用于 steering；这里不再使用类均值，而是把一条 donor sequence 的完整 256 维 residual 精确替换到另一条 receiver sequence 的同一语义位置，再运行剩余 Layers。这样直接检验：某个 token 位置的自然 hidden state 是否足以把 donor count 或“现在可以结束 trace”的状态搬到 receiver。</div>

      <h3>10.1 表示空间、配对方式与结果量</h3>
      <h4>实验</h4>
      <div class="protocol"><ol>
        <li><b>替换的是同一种表示。</b>Donor 与 receiver 都在同一个 Transformer Layer 的 block-output hook 捕获 <em>post-Layer residual</em>，即该 Layer attention、MLP 与两次 residual addition 完成后的 256 维向量；替换后只运行后续 Layers。它不是 embedding、attention value 或 MLP intermediate。</li>
        <li><b>跨 sequence pair。</b>每个合法有序 count pair <code>m→n</code> 使用 {hidden_patch_manifest['examples_per_pair']} 个独立 nested prompt pair；donor 与 receiver 共享 noise base，count 较小者的 needle 集合是较大者的子集。结果仍按 receiver count 1–10、11–20、21–30 分栏。</li>
        <li><b>连续结果量。</b>从 30 个 count-token logits 算 <code>E[C]</code>，回归 <code>ΔE[C]=a+b(m−n)</code>。Transport slope <code>b=1</code> 表示 patched 输出一比一随 donor count 移动，<code>b=0</code> 表示没有 count transport；<code>follows donor</code> 则要求 argmax count 恰好等于 m。</li>
        <li><b>为什么还保留 centroid transplant。</b>Figure 8A 的 centroid intervention 平均掉 donor prompt identity；后续自然-state patch 则保留一条具体 donor sequence 的全部内容。两者一致时，才能更有把握地说 transported signal 是 count state，而不是某个平均化人工方向。</li>
      </ol></div>

      <h3>10.2 先验基线：exact-count centroid transplant</h3>
      <h4>实验设定</h4>
      <p>这个基线先问一个最宽松的问题：<b>同一 count 类别的平均 residual 几何，是否已经足以控制最终 count logits？</b>对每个模型、semantic site、Layer 与 count <code>c</code>，我们用独立 direction split 的样本计算 256 维均值 <code>mu(layer, site, c)</code>。均值样本不包含被测试 receiver，因此 intervention 不会复制同一条 prompt 的内容。</p>
      <div class="method"><b>具体例子：receiver=8，目标 donor count=18。</b><ol>
        <li>正常运行 count=8 的 receiver，并定位它的最终 <code>&lt;Ans&gt;</code> query。</li>
        <li>若测试 after Layer 2，就把 receiver 在该 token 经过 Layer 2 后的 residual <code>h2(receiver, Ans)</code> 整体替换成 <code>mu(2, Ans, 18)</code>。</li>
        <li>从 Layer 3 继续运行到 LM head；输入中尚未出现答案 count token，因此输出只能由 patched state 与后续 Layers 决定。</li>
        <li>从 30 个 count-token logits 计算 argmax count 与 expected count，并对多个 <code>receiver n ← donor m</code> pair 拟合 transport slope。</li>
      </ol></div>
      <p>图中的 natural donor residual 是更严格的比较：不用类均值，而把一条独立 count=18 donor 在同一 semantic site 的完整 residual 放进 count=8 receiver。Centroid 成功只说明“平均 count geometry 可执行”；natural residual 也成功，才说明单样本状态同样可运输。</p>
      <h4>结果</h4>
      {figure(
          generated['transplant'],
          'Figure 8A. Count centroid 与单 donor residual 的分层 transport slope',
          '<b>横轴</b>是 residual 被替换在 Layer 1–4 哪一层之后；<b>纵轴</b>是 expected-count transport slope。蓝线写入独立 direction split 的 exact-count centroid；橙虚线写入一条 donor sequence 的 final-query residual。每行一个 semantic site，每列一个 receiver count 区间。虚线 1 表示理想的一比一 donor-count transport。'
      )}
      <h4>分析</h4>
      <p>Centroid 与自然 donor residual 在后层趋于一致，说明 final-query residual 中确实存在足以控制 count logits 的状态。不过这张旧图混合了不同 patch protocol；下面的三个实验使用统一 block-output hook、明确的 donor/receiver token anchor，并将自然 sequence patch 单独报告。</p>

      <h3>10.3 Final &lt;Ans&gt; patch：Non-thinking 与 CoT 的最终 count state</h3>
      <h4>实验设定</h4>
      <p>这里不再使用 centroid，而是做<b>单 donor sequence 到单 receiver sequence</b>的 natural-state patch。Donor 与 receiver 是独立样本，但由 nested construction 共享 noise base；较小 count 的 needle 集合是较大 count 的子集。这样 count 差异明确，又不会把完全不同的 prompt 当作同一干预对。</p>
      <div class="method"><b>具体例子：receiver=8，donor=18。</b><ul>
        <li>Non-thinking receiver 输入停在 <code>&lt;BOS&gt; prompt(count=8) &lt;Ans&gt;</code>；donor 是 <code>&lt;BOS&gt; prompt(count=18) &lt;Ans&gt;</code>。</li>
        <li>CoT receiver 输入停在 <code>... &lt;Think&gt; &lt;1&gt; M1 ... &lt;8&gt; M8 &lt;/Think&gt; &lt;Ans&gt;</code>；donor 对应地停在 <code>... &lt;18&gt; M18 &lt;/Think&gt; &lt;Ans&gt;</code>。</li>
        <li>在 donor 与 receiver 各自的 <code>&lt;Ans&gt;</code> token，缓存 after Layer L 的完整 256 维 residual；把 donor residual 替换进 receiver，再运行 Layer L+1 到 LM head。</li>
        <li><b>答案 token没有作为输入。</b>我们只读取 receiver <code>&lt;Ans&gt;</code> 后下一 token 的 30 个 count logits。CoT 两条 trace 长度不同，所以 patch 对齐的是 semantic anchor，而不是相同绝对数组下标。</li>
      </ul></div>
      <p>每个 Layer 都分别重做 patch；baseline 是同一个 receiver 未干预时的 logits。结果量包括 expected-count shift、transport slope、argmax 是否等于 donor count，以及是否仍等于 receiver count。</p>
      <h4>结果</h4>
      {figure(
          generated['hidden_final_answer'],
          'Figure 8B. 跨 sequence 的 final-<Ans> natural residual transplant',
          '<b>横轴</b>是 patch 后继续运行前已经完成的 Layer；<b>纵轴</b>是 expected-count transport slope。三栏按 receiver count 分区；蓝线为 non-thinking，橙线为 CoT。虚线 1 表示 patched 输出一比一跟随 donor count。'
      )}
      {table(hidden_final_rows,[('mode','模型'),('bin','receiver count 区间'),('l1_slope','after Layer 1 slope'),('l3_slope','after Layer 3 slope'),('l4_slope','after Layer 4 slope'),('l4_donor','after Layer 4 argmax=donor')])}
      <h4>分析</h4>
      <p><b>After Layer 4 的 final-query residual 对两个模型都具有近乎完全的因果充分性。</b>六个 mode×count-bin 条件的 slope 均约为 1，且 argmax 100% 等于 donor count。Non-thinking 的状态随深度逐步成形：21–30 的 slope 从 Layer 1 后的 <b>0.525</b>、Layer 3 后的 <b>0.800</b> 增至 Layer 4 后的 <b>1.000</b>。CoT 则更早形成可执行 count state：11–30 在 Layer 1 后已达 0.807–0.993，Layer 2 后即约为 1。</p>
      <div class="callout good"><b>因果含义。</b>这比“probe 可读”更强：在新的 receiver prompt 上替换完整 natural residual，输出真的变成 donor count。它支持 non-thinking 通过后层逐步汇总 prompt evidence，而 CoT 在 final <code>&lt;Ans&gt;</code> 前已经把 trace-derived count 压缩成较早可执行的 residual state。</div>

      <h3>10.4 CoT final marker patch：最后一个 marker state 不是最终答案 count state</h3>
      <h4>实验设定</h4>
      <p>本实验只把 patch anchor 从最终 <code>&lt;Ans&gt;</code> 提前到 CoT trace 的最后一个 marker，检验“完成第 n 次 retrieval 后的 marker state”是否已经等于可执行的 scalar count state。</p>
      <div class="method"><b>具体例子：receiver=8，donor=18。</b><ol>
        <li>Receiver 的 anchor 是 trace 尾部 <code>&lt;8&gt; M8</code> 中的 <code>M8</code>；donor anchor 是 <code>&lt;18&gt; M18</code> 中的 <code>M18</code>。</li>
        <li>在 after Layer L，把 <code>hL(donor, M18)</code> 替换到 <code>hL(receiver, M8)</code>，再继续运行 receiver 的剩余 Layers。</li>
        <li>Receiver 后缀仍是自己的 <code>&lt;/Think&gt; &lt;Ans&gt;</code>；答案 count token不作为输入。最终在 <code>&lt;Ans&gt;</code> 后读取 count logits。</li>
      </ol></div>
      <p>因此 10.3 与 10.4 的 donor/receiver pair 和结果量相同，唯一关键差别是 patch site：前者直接 patch final-answer query，后者 patch 更早的 final trace marker。两者的差异定位了 scalar readout 在哪一步才形成。</p>
      <h4>结果</h4>
      {figure(
          generated['hidden_trace_final'],
          'Figure 8C. CoT final-marker m→n patch 对最终 count 的 transport',
          '<b>横轴</b>是 final marker residual 被替换在 Layer 1–4 哪一层之后；<b>纵轴</b>是最终 <code>&lt;Ans&gt;</code> expected-count transport slope。三栏为 receiver count 区间；虚线 1 表示理想 donor-count transport。曲线贴近 0 表示最终答案继续跟随 receiver。'
      )}
      {table(hidden_trace_rows,[('bin','receiver count 区间'),('l1_slope','after Layer 1 slope'),('l4_slope','after Layer 4 slope'),('l4_receiver','after Layer 4 argmax=receiver')])}
      <h4>分析</h4>
      <p>三个区间、四层的 transport slope 都近似 0；after Layer 4 的最终 count 仍 100% 跟随 receiver。因而“final marker residual”与“final <code>&lt;Ans&gt;</code> count residual”不是同一个状态。关闭 trace 后的 token、位置与后续计算仍会重新构造答案 readout；不能因为最后 marker 与 count 同步出现，就把它当作已经完成的标量 count register。</p>

      <h3>10.5 只看 donor count &gt; receiver count：较大的 scalar 能否向上搬运</h3>
      <h4>实验设定</h4>
      <p>这不是另一种 patch，而是对 10.3 与 10.4 的逐 pair 记录做<b>有方向的子集分析</b>。原 transport slope 同时混合 <code>8←18</code> 与 <code>18←8</code>；如果模型只容易把大数向下拉，却不能把小 receiver 向更大的 donor 推动，双向平均可能掩盖这种不对称。</p>
      <div class="method"><b>保留哪些例子。</b>只保留 <code>donor_count &gt; receiver_count</code>，例如 <code>8←18</code>、<code>12←24</code> 和 <code>20←30</code>；删除同 count 与 donor 更小的 pair。对 final <code>&lt;Ans&gt;</code> patch 和 final-marker patch 分别在每个 Layer、每个 receiver count 区间拟合 <code>patched expected-count shift = a + b(donor−receiver)</code>。<code>b=1</code> 表示 donor 每大 1，输出也向上移动 1；<code>b=0</code> 表示没有上行运输。</div>
      <p>由于 patch 本身与 10.3/10.4 完全相同，这一节检验的是结论的方向稳定性，而不是引入新 intervention。</p>
      <h4>结果</h4>
      {figure(
          generated['hidden_upward_transport'],
          'Figure 8D. 仅 donor count > receiver count 的上行 scalar transport',
          '<b>列</b>按 receiver count 1–10、11–20、21–30 分区；<b>横轴</b>是 residual patch 在 Layer 1–4 哪层之后发生；<b>纵轴</b>是只用 donor&gt;receiver pairs 拟合的 expected-count transport slope。上两行分别在 non-thinking 与 CoT 的最终 <code>&lt;Ans&gt;</code> patch；第三行在 CoT 最后 trace marker patch。绿虚线 1 为理想的一比一上行搬运，灰点线 0 为无搬运。'
      )}
      {table(upward_transport_rows,[('site','patch site'),('bin','receiver count 区间'),('l1','after Layer 1'),('l2','after Layer 2'),('l3','after Layer 3'),('l4','after Layer 4')])}
      <h4>分析</h4>
      <p>最终 <code>&lt;Ans&gt;</code> residual 的上行搬运在后层仍然成立：Non-thinking 到 Layer 4、CoT 通常从 Layer 2 起，slope 都约为 1。高 count 的 non-thinking 形成最慢，21–30 区间在 Layer 1/2/3 后仅约 0.002/0.254/0.304，到 Layer 4 才跃升到 1。相反，CoT 最后 trace-marker residual 的 slope 在所有区间和 Layer 都约为 0。这个方向性分析排除了“原结论只是较大 receiver 被较小 donor 向下拉容易”的解释：<b>真正能向上运输较大 scalar 的是 final-answer state，不是最后 marker state。</b></p>

      <h3>10.6 CoT trace 内部 early-stop patch：短 trace 的终止状态能否关闭长 trace</h3>
      <h4>实验设定</h4>
      <p>这里把 total count 与当前 trace progress 故意拆开。构造短 donor count=<code>m</code> 与长 receiver count=<code>n&gt;m</code>，并让两者共享前 m 个 needle 及对应 marker identity。于是同一个 marker <code>M_m</code> 在 donor 是终点，在 receiver 只是中间点。</p>
      <div class="method"><b>具体例子：receiver=10，donor=5，在 I5/M5 patch。</b><ul>
        <li>Donor 局部上下文为 <code>... &lt;5&gt; M5</code>，正确下一 token 是 <code>&lt;/Think&gt;</code>。</li>
        <li>Receiver 局部上下文也到 <code>... &lt;5&gt; M5</code>，但 prompt 里还有 5 个 needles，正确下一 token 是 <code>&lt;6&gt;</code>。</li>
        <li>在 after Layer L，把 donor 的 <code>M5</code> residual 替换到 receiver 的 <code>M5</code>，再读取 receiver 下一 token logits。局部结果量是 <code>close margin = logit(&lt;/Think&gt;) − logit(&lt;6&gt;)</code>。</li>
      </ul></div>
      <p><b>最终 count 使用第二个、独立的 forward。</b>输入只保留 receiver prompt 与前 5 组 trace，然后人为接上 <code>&lt;/Think&gt;&lt;Ans&gt;</code>；不提供 I6–I10/M6–M10，也不提供答案 count token。我们再次在相同 marker site patch，再从 <code>&lt;Ans&gt;</code> 后读取 count logits。这样“局部关闭决定”与“截断 trace 后的 count readout”不会被 gold trace tail 混在一起。</p>
      <h4>结果 A：局部关闭决定</h4>
      {figure(
          generated['hidden_early_close'],
          'Figure 8E. Donor final-marker state 是否使 receiver interior marker 提前关闭',
          '<b>横轴</b>是 patch 所在 Layer；<b>纵轴</b>是 patched receiver 下一 token 预测为 <code>&lt;/Think&gt;</code> 的样本比例。三栏按长 receiver 的 count 分区。1 表示所有 pair 都被 donor 的“现在结束”状态关闭。'
      )}
      {table(hidden_early_rows,[('bin','receiver count 区间'),('l3_close','after Layer 3 close rate'),('l4_close','after Layer 4 close rate'),('l4_margin','after Layer 4 close-margin shift'),('l4_slope','forced-close count slope')])}
      <h4>分析 A</h4>
      <p>After Layer 4 patch 在三个区间都令 close rate 达到 100%，平均 close-margin 分别增加 <b>32.84 / 30.97 / 29.76</b>；Layer 3 的 close rate 为 <b>69.0% / 28.0% / 26.0%</b>。因此 trace marker residual 在后层确实携带强的局部 progress/termination state，而且该状态可以从“短 trace 的最后一步”因果搬到“长 trace 的内部步骤”。</p>
      <h4>结果 B：强制关闭后的最终 count</h4>
      {figure(
          generated['hidden_early_transport'],
          'Figure 8F. 截断 trace 强制关闭后，最终答案是否额外运输 donor count',
          '<b>横轴</b>是 donor final-marker residual 的 patch Layer；<b>纵轴</b>是截断输入最终 <code>&lt;Ans&gt;</code> expected-count 对 donor offset 的 transport slope。三栏为 receiver count 区间。接近 0 表示 patch 本身没有在截断 readout 中增加额外 donor-count 位移。'
      )}
      <h4>分析 B</h4>
      <p>三个区间、四层的 forced-close count transport slope 都约为 0。这里不能把“最终预测常等于 m”当作 donor-state 运输，因为截断 receiver 本身就只展示了 m 个 trace markers；即使不 patch，模型也可以仅凭可见 trace 长度读出 m。最稳健的结论是：final-marker residual 足以搬运<b>局部结束决定</b>，但没有证据表明它还能独立于截断 trace 内容搬运最终 scalar count。</p>

      <h3>10.7 不提供 gold tail 的 12 场景自由生成：拆开 total count 与 trace progress</h3>
      <h4>实验设定</h4>
      <p>单个 <code>R5@I4 ← D10@I7</code> 只能说明一个方向。这里使用 12 个预注册场景覆盖低、中、高 count，并把 donor 的<b>总 needle 数</b>与<b>当前 trace progress</b>做近似 2×2 因子拆分。记号 <code>Ra@Ii ← Db@Ij</code> 表示：receiver prompt 总数为 a，输入 trace 只给到 <code>&lt;i&gt; Mi</code>；donor prompt 总数为 b，residual 取自 <code>&lt;j&gt; Mj</code>。</p>
      <div class="method"><b>12 个场景分为五类。</b><ul>
        <li><b>total 与 progress 同向上移：</b><code>R5@I4←D10@I7</code>、<code>R10@I7←D20@I14</code>、<code>R20@I14←D30@I24</code>。</li>
        <li><b>total 与 progress 同向下移：</b><code>R10@I4←D5@I2</code>、<code>R20@I9←D10@I5</code>、<code>R30@I18←D20@I10</code>。这些 progress 特意满足 receiver 当前 I 仍存在于较小 donor 中，因此“同 progress、只换 total”的自然 donor 控制有定义。</li>
        <li><b>只改 progress：</b><code>R10@I4←D10@I7</code> 与 <code>R20@I14←D20@I7</code>，total 相同。</li>
        <li><b>只改 total：</b><code>R10@I7←D20@I7</code> 与 <code>R20@I7←D10@I7</code>，progress 相同。</li>
        <li><b>total 与 progress 反向冲突：</b><code>R10@I7←D20@I4</code> 与 <code>R20@I7←D10@I9</code>。</li>
      </ul></div>
      <p>每条 receiver 输入都<b>严格停在当前 marker</b>；后面不提供 gold trace tail，不强制 <code>&lt;/Think&gt;</code>、<code>&lt;Ans&gt;</code>，也不提供答案 token。Patch 后 greedy rollout 最多 {hidden_patch_manifest['free_rollout_protocol']['max_new_tokens']} tokens。每个场景使用 {hidden_patch_manifest['free_rollout_protocol']['examples']} 个独立 prompt pairs。正式实验只在 <b>after Layer {hidden_patch_manifest['free_rollout_protocol']['layers'][0]} 做一次 one-shot patch</b>；下一 token 生成后 hook 即失效，所以结果不能归因于持续重复注入。</p>
      <div class="method"><b>以 <code>R10@I7 ← D20@I14</code> 为例，七种 intervention 如何构造。</b><ul>
        <li><code>baseline</code>：不 patch，receiver 通常应从 I8 继续并在 I10 附近结束，最终答 C10。</li>
        <li><code>natural total-only</code>：使用另一条 total=20、但同样位于 I7 的 donor residual，测试只换 prompt total evidence。</li>
        <li><code>natural total+progress</code>：使用 total=20、位于 I14 的 donor residual，同时改 total 与 progress。</li>
        <li><code>centroid total delta</code>：在 receiver state 上加 <code>mu(20,I7)−mu(10,I7)</code>。</li>
        <li><code>centroid progress delta</code>：加 <code>mu(20,I14)−mu(20,I7)</code>，只沿 donor total 条件下的 progress 方向移动。</li>
        <li><code>centroid combined delta</code>：加 <code>mu(20,I14)−mu(10,I7)</code>。</li>
        <li>另有 self-state 与 same-semantics cross-prompt controls，用来估计 hook 本身和无关 prompt identity 的影响。</li>
      </ul></div>
      <p>我们记录三个彼此不同的结果量：<b>first index</b> 是 patch 后第一个生成的 index，主要反映局部 progress；<b>stop index</b> 是生成 <code>&lt;/Think&gt;</code> 前最后一个 index；<b>final count</b> 是 <code>&lt;Ans&gt;</code> 后生成的 scalar count。对于每个结果，还拟合 <code>outcome shift = a + b_total(donor total−receiver total) + b_progress(donor progress−receiver progress)</code>，用偏回归区分两种 offset 的独立作用。</p>

      <h4>结果 A：patch 后的第一个 trace index</h4>
      {figure(
          generated['hidden_rollout_first'],
          'Figure 8G. 12 个 total/progress 场景的第一个自由生成 index',
          '<b>纵轴</b>是场景 <code>receiver←donor</code>；<b>横轴</b>是 after Layer 4 one-shot intervention。单元格为跨样本平均的第一个 index。每行应同时对照 receiver progress+1 与 donor progress+1：靠近前者表示未运输，靠近后者表示 donor progress 被运输。'
      )}
      <div class="callout good"><b>局部 progress 被精确搬运。</b>对 progress-centroid intervention，{pct(rollout_donor_successor)} 的样本把 patch 后第一个 index 改成 donor progress+1；跨 12 场景偏回归得到 <code>b_progress={fmt(rollout_progress_factor.donor_progress_coefficient)}</code>、<code>b_total={fmt(rollout_progress_factor.donor_total_coefficient)}</code>、<code>R²={fmt(rollout_progress_factor.r2)}</code>。例如 <code>R10@I7←D20@I14</code> 不再从 I8 继续，而是从 I15 开始；反向的 <code>R10@I4←D5@I2</code> 则从 I3 开始。这不是“总 count 大小”的附带效应，因为只改 total、保持 I7 不变时，第一个 index 仍是 receiver 的 I8。</div>
      <h4>结果 B：trace 最终停止 index</h4>
      {figure(
          generated['hidden_rollout_stop'],
          'Figure 8H. 12 个场景自由生成 trace 的停止 index',
          '<b>纵轴与横轴</b>同 Figure 8G；单元格是生成 <code>&lt;/Think&gt;</code> 前最后一个 index。应分别比较 receiver total、donor total、receiver progress 与 donor progress，不能只把某一个数预设为正确答案。'
      )}
      <p><b>停止位置没有同样干净的一维搬运。</b>Progress intervention 后只有 {pct(rollout_closed)} 的 rollout 在 40-token 预算内自然产生 <code>&lt;/Think&gt;</code>。典型边界是 <code>R10@I7←D20@I14</code>：模型先生成 I15，但没有在预算内恢复成一条合法、可关闭的 receiver trace；而 <code>R20@I9←D10@I5</code> 会从 I6 重新开始，随后仍在 I20 左右关闭。说明 one-shot residual 足以设定“下一步在哪里”，却不总是足以把后续状态机稳定接到 donor 或 receiver 的完整终止轨迹上。</p>
      <h4>结果 C：关闭 trace 后的最终 scalar count</h4>
      {figure(
          generated['hidden_rollout_final'],
          'Figure 8I. 12 个场景自由生成的最终 count',
          '<b>纵轴与横轴</b>同 Figure 8G；单元格是 <code>&lt;Ans&gt;</code> 后 count token 的均值。若 marker state 独立携带 scalar，final count 应随 donor total offset 系统移动；若 scalar 被 receiver prompt/readout 重建，则应留在 receiver total。'
      )}
      <p><b>最终 scalar 没有随 donor total 搬运。</b>在 donor/receiver total 不同且成功生成答案的样本中，{pct(rollout_distinct_receiver_given_answer)} 的 final count 等于 receiver total，{pct(rollout_distinct_donor_given_answer)} 等于 donor total；少数偏差只多出 1（C10→C11 或 C20→C21）。相反，两个 total-only intervention 共 {int(rollout_total_only['n'].sum())} 条记录全部自然关闭（{pct(rollout_total_only_closed)}），并且 {pct(rollout_total_only_receiver)} 保持 receiver count。也就是说，在这个 marker site 单独注入“donor 总数”既不改变下一 index，也不改变最终答案。</p>
      <h4>结果 D：跨场景 total/progress 偏回归</h4>
      {figure(
          generated['hidden_rollout_factors'],
          'Figure 8J. Donor total offset 与 progress offset 对三个 rollout 结果的独立系数',
          '<b>横轴</b>是 intervention；每个 outcome 分别显示 <code>b_total</code> 与 <code>b_progress</code>。系数 1 表示 donor offset 每增加 1，结果同向移动 1；0 表示在控制另一个 offset 后没有独立线性作用。仅使用 after Layer 4 one-shot 的 12 个场景均值。'
      )}
      {table(rollout_factor_rows,[('intervention','intervention'),('outcome','结果量'),('b_total','b_total'),('b_progress','b_progress'),('r2','R²'),('n','场景数')])}
      <h4>代表性场景的原始均值</h4>
      {table(rollout_rows,[('family','场景类别'),('pair','receiver ← donor'),('first','第一个 index'),('stop','停止 index'),('final','最终 count'),('receiver','receiver 参照：next / stop / count'),('donor','donor 参照：next / stop / count')])}
      <h4>综合分析</h4>
      <p>五类场景共同给出一个稳定分离：<b>marker residual 中可执行的是局部 trace progress，而不是独立的 prompt-total scalar。</b>Natural donor、progress-centroid 与 combined-centroid 三种 intervention 产生相同的第一 index 行为；total-only natural state 与 total-only centroid delta 则完全不动。这使结论不依赖某一个 5→10 例子，也不依赖 donor total 与 progress 恰好同向。</p>
      <p>偏回归也支持同一结论。第一 index 的 <code>b_progress=1</code>、<code>b_total≈0</code> 且 <code>R²=1</code>；final count 的两个系数都接近 0（<code>b_total={fmt(rollout_final_factor.donor_total_coefficient)}</code>、<code>b_progress={fmt(rollout_final_factor.donor_progress_coefficient)}</code>），但其 <code>R²={fmt(rollout_final_factor.r2)}</code> 较低，因为 27.1% 的 progress-intervention rollout 未自然关闭，线性 final-count 回归只能覆盖成功产生答案的子集。因此 final-count 的更可靠证据是上面的条件命中率与 total-only 零效应，而不是低 R² 回归本身。</p>
      <div class="callout good"><b>综合答案。</b>把 D10@I7 的状态放到 R5@I4，或者把 D20@I14 放到 R10@I7，确实会把下一步分别推进到 I8、I15；反向 donor 也会把下一步拉回 I3、I6。可是它不会把最终答案系统改成 C10、C20。最一致的机制是：<b>trace marker state 搬运“当前走到第几步/下一步从哪开始”；receiver prompt 与后续 readout 再决定在哪里关闭，并在 final <code>&lt;Ans&gt;</code> site 重建 scalar count。</b>因此“progress register”和“final count register”在本模型中是因果可分的。</div>

      <h3>10.8 本节结论</h3>
      <div class="mechanisms">
        <div class="mechanism"><h3>Non-thinking</h3><p>Final <code>&lt;Ans&gt;</code> count state 从早层到后层逐步变得充分，高 count 最慢；after Layer 4 natural residual 能一比一把 donor count 搬到 receiver。</p></div>
        <div class="mechanism"><h3>CoT</h3><p>最终 <code>&lt;Ans&gt;</code> 很早就含有可执行 count state；trace final marker 本身不直接等于答案 state，但 marker residual 的后层表示能控制“继续还是结束”。这支持 <b>trace progress/termination</b> 与 <b>final answer count</b> 是相关但不同的 causal states。</p></div>
      </div>
      <div class="callout limit"><b>证据边界。</b>所有实验为单 seed、单模型大小；成对 slope 每个有序 count pair 只有 {hidden_patch_manifest['examples_per_pair']} 个 nested base prompts，自由 rollout 每个 intervention、每个场景只有 {hidden_patch_manifest['free_rollout_protocol']['examples']} 个 prompts。12 场景已覆盖 total/progress 同向上移、同向下移、单因素变化与反向冲突，并移除 gold trace tail；但正式 suite 只检验 after Layer 4 one-shot state，且 40-token 截断会把“暂时失稳”和“永不关闭”混在一起。它还没有分解 progress state 经过后续 attention/MLP、<code>&lt;/Think&gt;</code> 与 <code>&lt;Ans&gt;</code> 转换成最终 answer state 的每个中间步骤。</div>
    </section>
    """

    report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Trace Count v10 分层因果机制报告</title><style>{css}</style></head><body><main>
    <header><h1>Trace Count v10：Count-30 分层学习动态与因果机制</h1><p class="subtitle">比较两个独立 Transformer 的直接计数与显式 CoT 检索路径，并把所有 head ablation、head-output patching、geometry steering 与 residual transplant 按 count 1–10、11–20、21–30 重新分析。</p><p class="meta">Run: {html.escape(run_dir.name)} · seed {config['seed']} · self-contained HTML · figures embedded as data URIs</p></header>
    <nav class="toc"><b>目录</b><ol><li><a href="#object">研究对象与机制假设</a></li><li><a href="#setting">数据、模型与 sequence</a></li><li><a href="#definitions">术语、指标与公式</a></li><li><a href="#dynamics">学习动态</a></li><li><a href="#attention">Attention 候选机制</a></li><li><a href="#ablation">分层 head ablation</a></li><li><a href="#patching">分层 activation patching</a></li><li><a href="#steering">分层 geometry steering</a></li><li><a href="#transplant">分层 residual transplant</a></li><li><a href="#geometry">PCA variance 与互动 3D manifold</a></li><li><a href="#synthesis">综合结论与边界</a></li></ol></nav>

    {object_section}

    <section id="setting"><h2>2. 模型、数据生成、训练目标与 sequence</h2>{table(settings,[('item','项目'),('value','严格设置')])}<div class="protocol"><b>三个难度区间。</b><code>1-10</code>、<code>11-20</code>、<code>21-30</code> 都包含十个 exact-count 类。训练、评估和补充干预均按 exact count 平衡；所以区间差异反映 needle 数量增加，而不是类别数或样本数改变。</div></section>

    {definitions_section}

    {dynamics_section}

    <section id="attention"><h2>5. 描述性 attention：候选 broad 与 targeted heads</h2><div class="protocol"><b>Non-thinking。</b>在完整 prompt 后的 <code>&lt;Ans&gt;</code> query 读取 16 个 head 的 attention row，计算 broad score。<b>CoT。</b>对每个 gold trace 的 <code>&lt;k&gt;</code> query，取它指向 prompt 第 k 个 needle 的 raw mass；先按 query 求值，再在所选 count 区间内平均。高分只用于候选排序，不作为因果结论。</div>{figure(generated['attention'],'Figure 2. Broad aggregation 与 k-to-k retrieval 的候选 head signatures','<b>横轴</b>是 head 0–3；<b>纵轴</b>是 Layer 1–4；单元格为标题所示分数。左图是 non-thinking 全 count 的 broad score；右侧三图把 CoT matching-needle raw mass 按 gold count 区间分开。更高 count 有更多 k queries，因此全局 query-weighted 平均会偏向 21–30；本图通过分栏消除了这个混淆。')}</section>

    <section id="ablation"><h2>6. 分层 global head ablation：哪些 heads 对哪个难度区间必要</h2><div class="protocol"><b>样本。</b>本次重新生成每个 exact count {manifest['ablation_examples_per_exact_count']} 个 prompts，共 {30*manifest['ablation_examples_per_exact_count']} 个；两个模型使用各自格式做 teacher-forced forward。<b>干预。</b>用 GPT-2 <code>head_mask</code> 将指定 head 在整条 sequence、所有 query positions 的输出设为 0。单头逐个测 16 次；累计实验按候选排名依次 mask top-1…top-16，并与排名倒序和 {manifest['ablation_random_orders']} 条固定随机顺序比较。<b>指标。</b>remaining accuracy 是 mask 后的绝对 accuracy；drop 是无干预 baseline 减去 mask 后 accuracy。</div>{figure(generated['single_nonthinking'],'Figure 3. 单头必要性按 count 区间分解',single_ablation_caption)}{table(ablation_rows,[('bin','count 区间'),('direct_head','最强 non-thinking 单头'),('direct_drop','final accuracy drop'),('trace_head','最强 CoT trace 单头'),('trace_drop','trace-marker accuracy drop')])}{figure(generated['cumulative_nonthinking'],'Figure 4. Top-n 累计 mask 的剂量曲线与强对照',cumulative_ablation_caption)}</section>

    <section id="patching"><h2>8. Attention-head patching：候选 heads 是否局部充分</h2><h3>8.1 CoT marker-identity clean-to-corrupt retrieval patch</h3><div class="protocol"><b>配对输入。</b>clean 与 corrupt prompt 位置、noise、count 和 trace index 完全相同，只把目标第 k 个 prompt marker identity 换成另一 marker；读取同一个 <code>&lt;k&gt;</code> query 对 clean marker 的 logit margin。<b>patch。</b>先缓存 clean run 在每层 attention <code>c_proj</code> 前按 head 切分的 output slice，再把选中 top-n slices 替换到 corrupt run 同一 query；其余 activation 保持 corrupt。interior 取约中间 k，final 取 k=n。<b>样本来源。</b>原始逐 query 干预记录每个 exact count 2 个 prompts，本报告按 gold count 无损重聚合。</div>{figure(generated['retrieval'],'Figure 5. Targeted retrieval patch 的 normalized recovery','<b>横轴</b>是被 patch 的 head 数；<b>纵轴</b>是 clean-marker logit margin 的 normalized recovery。上行为 interior k，下行为 final k=n；三列为 count 区间。蓝=targeted ranking top；红=bottom；灰=三个随机 head 顺序的均值与范围。')}
    <h3>8.2 Nested prompt donor→receiver count-head patch</h3><div class="protocol"><b>配对输入。</b>receiver count=n 与 donor count=m 共享同一 256-token noise 序列；needle 集合是 nested 的，即较小 count 的 needles 是较大 count 的子集。m−n 穷举配置中的 ±1、±2、±3、±5、±10（只保留 1–30 内合法 pair）。<b>query。</b>在各自 final <code>&lt;Ans&gt;</code> position patch head-output slices；CoT donor/receiver trace 长度不同，所以语义 query 对齐但绝对位置可不同。<b>结果量。</b>对每个 bin 独立拟合 expected-count shift 对 donor offset 的 slope；primary=non-thinking broad 或 CoT trace-readout 排名。bottom/random 是必要对照。</div>{figure(generated['nested'],'Figure 6. Head-output 是否能够运输 count state','<b>横轴</b>是 patch 的 donor head slices 数；<b>纵轴</b>是 expected-count shift 对 donor offset 的回归 slope。1 表示一比一随 donor 移动，0 表示没有 count transport。上行为 non-thinking，下行为 CoT；三列为 receiver count 区间。')}{table(patch_rows,[('bin','receiver/gold count 区间'),('retrieval','targeted top-4 final-k recovery'),('direct','non-thinking primary top-4 slope'),('cot','CoT trace-readout top-4 slope')])}</section>

    <section id="steering"><h2>9. Hidden-state geometry steering：可读方向是否也是可控方向</h2><div class="protocol"><b>方向训练。</b>在独立 direction-train split 中，对每个 site×Layer×exact count 求 residual centroid，再计算 29 个相邻差向量；将其归一化平均得到 adjacent-mean direction，并用平均相邻距离定标一步。<b>site。</b>non-thinking natural final；CoT natural final；CoT counterfactual fixed-15 final。最后一个 site 使用与第 6.1 节相同的反事实输入：15 步模板 trace 对所有 prompt 固定，真实答案仍是 prompt count；它不是 gold trace。<b>干预。</b>在 held-out query 的某 Layer 后加 <code>alpha·step_size·direction</code>，alpha∈{config['steering_alphas']}；所有行按真实 prompt count 分成三栏。<b>注意。</b>这是真实重新前向得到的逐样本 intervention 表的分层重聚合，不是用 probe 预测代替模型输出。</div>{figure(generated['steering'],'Figure 7. Adjacent-count direction 的 dose response','<b>横轴</b>是 alpha；<b>纵轴</b>是干预后 expected count 减 baseline expected count。彩色线表示在 Layer 1–4 后加方向；虚线 y=alpha 是理想均匀计数轴。每行一个 semantic site，每列一个 count 区间。曲线非线性、饱和或反向说明同一全局向量不能在整个 manifold 上充当统一 +1 算子。')}{table(steering_rows,[('bin','count 区间'),('site','site'),('layer','该区间最佳线性 Layer'),('gain','shift/alpha gain'),('r2','线性 R²')])}<div class="callout warn"><b>解释边界：</b>高 probe/PCA 可读性不保证 adjacent-mean steering 有效。Centroid 轨迹若弯曲，相邻差向量会随 count 改变；把它们平均成一个方向可能离开自然 manifold。第 10 节的完整 centroid transplant 是更强、也更局部的因果测试。</div></section>

    <section id="transplant"><h2>10. Hidden-state patching：完整 count state 能否跨 sequence 搬运</h2><div class="protocol"><b>完整 residual。</b>对 receiver count=n，在 final query 的某 Layer 后，用 donor count=m 的 256 维 residual 完整替换，再运行剩余 Layers。橙色虚线使用单个独立 donor prompt。<b>Count centroid。</b>蓝色实线使用 direction-train split 中 count=m 的 residual 均值，平均掉 donor prompt identity，是更干净的 count-state intervention。<b>指标。</b>按 receiver count 分栏，对所有合法 m−n offsets 独立拟合 transport slope。CoT counterfactual fixed-15 条件继续使用固定模板 trace 与真实 prompt-count 答案的冲突输入，测试第 6.1 节中“可读的 prompt-derived count”能否真正控制 logits；它不衡量自然 CoT 的生成准确率。</div>{figure(generated['transplant'],'Figure 8. 完整 residual/centroid 的 count transport slope','<b>横轴</b>是替换发生在 Layer 1–4 哪一层之后；<b>纵轴</b>是 expected-count shift / donor offset 的 slope。蓝=独立训练 centroid；橙=单 donor residual。每行一个 site，每列一个 receiver count 区间。slope=1 表示该位置的完整 residual 足以把输出一比一改成 donor count state。')}<div class="callout good"><b>为什么它比 steering 强：</b>steering 只沿一个人为抽取的全局方向加向量；centroid transplant 放入该 exact count 在该 Layer 的完整自然状态。若 transplant 成功而 steering 失败，结论是“count state 具有因果充分性，但不是一根全局笔直的 +1 axis”。</div>
    <h3>10.1 CoT trace 内部的 hidden-state patching：覆盖边界</h3><p>原始 v10 protocol 为控制成本，只在 count 26–30 上运行 trace-marker progress m→n transplant，因此这些逐样本行全部属于 <code>21-30</code>。它不能被诚实地画成三个区间。该实验仍支持 Layer 4 marker-position residual 能决定下一 index/close，但本报告不把 high-count-only 结果外推到 1–20。</p><p class="small">可用分层行数：{len(trace_progress)}；实际 count-bin：{', '.join(sorted(trace_progress.count_bin.unique()))}。</p></section>

    <section id="geometry"><h2>10. Count-state geometry：PCA variance、静态均值轨迹与互动 3D</h2><div class="protocol"><b>严格 mean-first PCA。</b>对每个 semantic site、每个 Layer，先在 held-out 数据中对同一个 exact count 的所有 256 维 residual 求均值，得到 30×256 的 centroid 矩阵；再减去 30 个 centroids 的总均值并做 SVD/PCA。这里的 explained variance 只描述<b>count 类均值之间</b>的几何，不混入同 count 样本内方差。每个 site×Layer 独立 fit，所以 PC 轴不能跨 panel 直接对齐。<b>反事实 site。</b><code>CoT counterfactual fixed-15 final</code> 指第 5B.1 节定义的固定模板 trace 控制，不是 gold trace。<b>有效维数</b>为 participation ratio <code>(Σλ)²/Σλ²</code>；<b>平均转角</b>是相邻 centroid 差向量之间夹角的均值。</div>{figure(generated['pca_static'],'Figure 9. Exact-count mean residual 的 PC1–PC2 渐变轨迹','每个点是一个 exact count 的 256 维均值；颜色从 1 连续渐变到 30；灰线连接相邻 count。三行分别为 non-thinking natural final、CoT natural final、CoT counterfactual fixed-15 final，四列为 Layer 1–4。横纵轴是该 panel 独立拟合的 PC1/PC2。')}{figure(generated['pca_variance'],'Figure 10. PC1–PC6 explained variance 与累计覆盖','前五个 panel：<b>横轴</b>为 PC1–PC6，<b>纵轴</b>为 Layer 1–4，单元格是该 PC 对 30 个 count centroids 的方差解释率。最后 panel 显示 PC1–PC6 累计解释率随 Layer 的变化。高累计率表示六维足以展示 centroid 间大部分结构，不表示单样本 hidden state 只有六维。')}{table(pca_rows,[('site','site'),('layer','Layer'),('pc1','PC1 variance'),('pc3','PC1–3 cumulative'),('pc6','PC1–6 cumulative'),('dim','effective dimension'),('turn','mean adjacent turn')])}{interactive_pca(coordinates, geometry)}<div class="callout warn"><b>如何读互动图：</b>切换 count range 只筛选已经在全 1–30 centroids 上拟合好的坐标，不会为每个区间重新旋转 PCA；因此三个区间在同一 site×Layer 中可直接比较。切换 site 或 Layer 后 PCA 会重新定义，屏幕方向不再是同一基底。</div></section>

    <section id="synthesis"><h2>11. 综合机制结论、证据强度与尚缺环节</h2><div class="mechanisms"><div class="mechanism"><h3>Non-thinking</h3><ol><li>描述性 broad heads 位于早期 Layer；分层 mask 检验它们对哪些 count 区间必要。</li><li>Nested donor patch 若在三个区间都有正 slope，说明候选 head slices 不只相关，而能运输部分 count state。</li><li>完整 centroid transplant 在后层趋近 slope=1，说明 answer-query residual 是足以决定 count 的因果状态。</li><li>Adjacent-mean steering 的区间不稳定性说明该状态不是一根跨 1–30 恒定的标量 +1 方向。</li></ol></div><div class="mechanism"><h3>CoT</h3><ol><li><code>&lt;k&gt;</code> query 的 k-to-k mass 随 count 增大而下降，但 targeted clean patch 相对 bottom/random 的恢复检验其因果功能。</li><li>Targeted retrieval 与 final trace readout 是不同排名；删除 retrieval heads 主要伤 marker trace，不自动等于删除最终 count state。</li><li>自然 final state 同时含 prompt、trace 内容、trace 长度和位置线索。反事实 fixed-15 control 则把 trace 内容、长度和 <code>&lt;Ans&gt;</code> 位置固定，只让 prompt count/答案 n 改变；它显示 prompt-derived count 仍可读，但分层 transplant 表明该低维几何并非在所有 count 区间都具有一比一因果控制力。</li><li>现有 trace-progress causality 只覆盖 26–30，尚需低 count 对称实验才能声称统一 successor circuit。</li></ol></div></div><div class="callout good"><b>本次改版的实质：</b>不再用全 1–30 的一个均值掩盖难度差异。Head necessity、retrieval recovery、count transport、steering gain 和 residual sufficiency 都在同样的 1–10 / 11–20 / 21–30 定义下报告；每一列都回到逐样本干预行计算。</div><div class="callout limit"><b>仍不能声称：</b>单 seed synthetic model 可外推到真实 LLM；attention weight 等于信息流；global mask 定位了具体 query；PCA 中连续轨迹等于模型执行逐步加法；反事实 fixed-15 输入代表自然 CoT；centroid transplant 揭示了模型在线构造该 centroid 的全部算法。最稳健的结论应限定为：两个模型形成了可区分的 routing signatures，且若干 head/residual 组件在受控干预下对 trace 或 count 输出具有区间依赖的必要性/充分性。</div><p class="meta">Provenance: <code>analysis/report_stratified/manifest.json</code>. Fresh forward pass: global head ablation. Lossless count-bin reaggregation from existing per-example intervention rows: retrieval patching, nested count patching, geometry steering, final-state transplant and centroid transplant.</p></section>
    </main></body></html>"""

    attention_start, attention_end = html_section_span(report, "attention")
    report = report[:attention_start] + attention_section + report[attention_end:]
    geometry_start, geometry_end = html_section_span(report, "geometry")
    report = report[:geometry_start] + report[geometry_end:]
    _, attention_end = html_section_span(report, "attention")
    report = report[:attention_end] + geometry_section + report[attention_end:]
    ablation_start, ablation_end = html_section_span(report, "ablation")
    report = report[:ablation_start] + ablation_section + report[ablation_end:]
    patching_start, patching_end = html_section_span(report, "patching")
    report = report[:patching_start] + patching_section + report[patching_end:]
    steering_start, steering_end = html_section_span(report, "steering")
    report = report[:steering_start] + steering_section + report[steering_end:]
    transplant_start, transplant_end = html_section_span(report, "transplant")
    report = report[:transplant_start] + transplant_section + report[transplant_end:]

    nav_start = report.index('<nav class="toc">')
    nav_end = report.index("</nav>", nav_start) + len("</nav>")
    new_nav = """
    <nav class="toc"><b>目录</b><ol>
      <li><a href="#object">研究对象与机制假设</a></li>
      <li><a href="#setting">数据、模型与 sequence</a></li>
      <li><a href="#definitions">术语、指标与公式</a></li>
      <li><a href="#dynamics">学习动态</a></li>
      <li><a href="#attention">描述性 attention</a></li>
      <li><a href="#geometry">描述性 hidden state</a></li>
      <li><a href="#ablation">Attention-head ablation</a></li>
      <li><a href="#patching">Attention-head patching</a></li>
      <li><a href="#steering">Hidden-state geometry steering</a></li>
      <li><a href="#transplant">Hidden-state patching</a></li>
      <li><a href="#interaction">Head 与 hidden state 的因果联系</a></li>
      <li><a href="#synthesis">综合结论与边界</a></li>
    </ol></nav>
    """
    report = report[:nav_start] + new_nav + report[nav_end:]

    report = finalize_report_numbering(report, run_dir)
    output.write_text(report, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the self-contained stratified Chinese v10 report")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    output = build_report(args.run_dir.resolve())
    print(f"V10_REPORT={output}")
    print(f"V10_REPORT_BYTES={output.stat().st_size}")


if __name__ == "__main__":
    main()
