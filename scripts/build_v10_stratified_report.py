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


def html_section_span(document: str, section_id: str) -> tuple[int, int]:
    marker = f'<section id="{section_id}">'
    start = document.index(marker)
    end = document.index("</section>", start) + len("</section>")
    return start, end


def finalize_report_numbering(report: str) -> str:
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
    show_bottom: bool,
    title: str,
    path: Path,
) -> None:
    """Save one cumulative head-ablation target as three count-bin panels."""
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.45), sharey=True, constrained_layout=True)
    for ax, count_bin in zip(axes, COUNT_BINS):
        frame = cumulative[(cumulative["mode"] == mode) & (cumulative["count_bin"] == count_bin)]
        top = frame[frame.family == family].sort_values("top_n")
        ax.plot(top.top_n, top[metric], color=BLUE, marker="o", ms=3, label="ranked top")
        if show_bottom:
            bottom = frame[frame.family == "primary_bottom"].sort_values("top_n")
            ax.plot(bottom.top_n, bottom[metric], color=RED, marker="o", ms=3, label="matched ranking bottom")
        random = frame[frame.family == "random"]
        mean, low, high = random_band(random, metric)
        ax.plot(mean.top_n, mean[metric], color=GRAY, linewidth=2, label="8 random orders: mean")
        ax.fill_between(mean.top_n, low[metric], high[metric], color=GRAY, alpha=0.18, label="random min-max")
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
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_retrieval_patch(retrieval: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.7, 7.8), sharex=True, sharey=True, constrained_layout=True)
    for row, role in enumerate(("interior", "final")):
        for col, count_bin in enumerate(COUNT_BINS):
            ax = axes[row, col]
            frame = retrieval[(retrieval.query_role == role) & (retrieval.count_bin == count_bin)]
            for family, color, label in (
                ("targeted_top", BLUE, "targeted top"),
                ("targeted_bottom", RED, "targeted bottom"),
            ):
                group = frame[frame.family == family].groupby("top_n", as_index=False).normalized_recovery.mean()
                ax.plot(group.top_n, group.normalized_recovery, color=color, marker="o", label=label)
            random = frame[frame.family == "random"]
            mean, low, high = random_band(random, "normalized_recovery")
            ax.plot(mean.top_n, mean.normalized_recovery, color=GRAY, marker="o", label="random mean")
            ax.fill_between(mean.top_n, low.normalized_recovery, high.normalized_recovery, color=GRAY, alpha=0.18)
            ax.axhline(0, color="#111827", linewidth=1)
            ax.axhline(1, color="#111827", linestyle="--", linewidth=1)
            ax.set_title(f"{role} trace index | count {count_bin}")
            ax.set_xticks([1, 2, 4, 8, 16])
            ax.set_xlabel("number of patched heads")
    axes[0, 0].set_ylabel("normalized clean-margin recovery")
    axes[1, 0].set_ylabel("normalized clean-margin recovery")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Marker-identity clean-to-corrupt patching, stratified by gold count", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_nested_patch(regression: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.7, 7.8), sharex=True, constrained_layout=True)
    for row, mode in enumerate(("nonthinking", "thinking")):
        for col, count_bin in enumerate(COUNT_BINS):
            ax = axes[row, col]
            frame = regression[(regression["mode"] == mode) & (regression["count_bin"] == count_bin)]
            for family, color, label in (
                ("primary_top", BLUE, "ranked top"),
                ("primary_bottom", RED, "ranked bottom"),
            ):
                group = frame[frame.family == family].sort_values("top_n")
                ax.plot(group.top_n, group.slope, color=color, marker="o", label=label)
            random = frame[frame.family == "random"]
            grouped = random.groupby("top_n").slope
            mean, low, high = grouped.mean(), grouped.min(), grouped.max()
            ax.plot(mean.index, mean.values, color=GRAY, marker="o", label="random mean")
            ax.fill_between(mean.index, low.values, high.values, color=GRAY, alpha=0.18)
            ax.axhline(0, color="#111827", linewidth=1)
            ax.axhline(1, color="#111827", linestyle="--", linewidth=1)
            ax.set_title(f"{mode} final query | count {count_bin}")
            ax.set_xticks([1, 2, 4, 8, 16])
            ax.set_xlabel("number of donor head slices patched")
    axes[0, 0].set_ylabel("slope: expected shift / donor offset")
    axes[1, 0].set_ylabel("slope: expected shift / donor offset")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Nested-prompt head-output patching: transport of count across all offset sizes", fontsize=15, fontweight="bold")
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
    """Return a self-contained step-through animation for the two hypotheses."""

    return r"""
    <figure class="mechanism-explorer" aria-labelledby="mechanism-explorer-title">
      <div class="mechanism-explorer-head">
        <div>
          <h3 id="mechanism-explorer-title">互动机制图：同一个 prompt，两种候选计算路径</h3>
          <p>这是待检验的计算图，不是把 attention 图直接画成结论。切换模式并逐步播放，观察每一步读取什么、把信息写到哪里，以及对应的因果预测。</p>
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
            <g class="mech-node state"><rect x="820" y="82" width="245" height="82" rx="11"/><text x="942" y="113">progress / count state</text><text class="mech-subtext" x="942" y="138">three successful retrievals</text></g>
            <path class="mech-flow-arrow cot" d="M 528 296 C 680 292, 760 190, 835 153"/>
          </g>
          <g data-step="3" class="mech-step">
            <g class="mech-node trace-token successor"><rect x="550" y="274" width="52" height="44" rx="7"/><text x="576" y="302">4</text></g>
            <path class="mech-loop-arrow" d="M 528 318 C 535 365, 600 365, 603 318"/>
            <text x="620" y="372">successor step: emit k+1, then repeat retrieval</text>
          </g>
          <g data-step="4" class="mech-step">
            <g class="mech-node answer"><rect x="940" y="270" width="150" height="72" rx="10"/><text x="1015" y="298">final readout</text><text class="mech-answer-token" x="1015" y="325">&lt;Cₙ&gt;</text></g>
            <path class="mech-flow-arrow cot" d="M 942 164 C 925 205, 940 236, 970 269"/>
            <text x="795" y="230">after trace close</text>
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
      <figcaption>箭头表示本报告要检验的信息路由方向，不等同于单张 attention map。Non-thinking 图中的并行聚合不要求单个 head 独自完成求和；CoT 图中的循环也不预设模型真的执行符号加法，successor 与 progress state 都需要后续 patching、ablation 和 residual transplant 来验证。</figcaption>
    </figure>
    <script>
    (() => {
      const copy = {
        direct: [
          {title:'Step 1 · 形成待计数集合', read:'读取：完整 prompt 中 marker/needle token 的身份与位置。', write:'写入：各位置的局部表示；此时还没有外显计数轨迹。', test:'可证伪预测：若模型完全不区分 needle 与 noise，后续 broad score 和 count probe 不应出现。'},
          {title:'Step 2 · 并行 broad retrieval', read:'读取：final <Ans> query 通过若干 attention heads 同时访问多个 prompt needles。', write:'写入：needle value vectors 的加权组合，而不是按 k=1,2,… 逐个输出。', test:'可证伪预测：高 broad-score heads 的全局 mask 应比 bottom/random heads 更早破坏 final accuracy。'},
          {title:'Step 3 · 聚合为 cardinality state', read:'读取：多个 head outputs 与既有 residual。', write:'写入：answer-query residual 中可区分 count 1…30 的分布式状态。', test:'可证伪预测：把 donor count 的 head slices 或 residual 搬给 receiver，输出应朝 donor count 移动。'},
          {title:'Step 4 · 后层变换与离散化', read:'读取：连续的 count-related residual geometry。', write:'写入：30 个 number-token logits 之间更大的正确 margin。', test:'可证伪预测：早层可能负责集合聚合，后层 residual transplant 应更接近一比一决定最终 count。'},
          {title:'Step 5 · 直接答案读出', read:'读取：<Ans> 位置最后一层 residual。', write:'输出：共享数字 token <Cₙ>，没有中间 trace token。', test:'机制边界：成功输出只证明最终状态足够，不证明内部一定沿一条线性 +1 轴计算。'}
        ],
        cot: [
          {title:'Step 1 · 启动 indexed trace', read:'读取：prompt 与 <Think> 前缀；trace 已生成 1,M₁,2,M₂，当前 query 是数字 3。', write:'写入：当前进度 k=3 的 query state。', test:'可证伪预测：数字 query 的表示应含当前 k/progress 信息，而不仅是绝对位置。'},
          {title:'Step 2 · k-to-k targeted retrieval', read:'读取：query <3> 定位 prompt 中按位置排序的第 3 个 needle。', write:'写入：该 needle 的 marker identity，为下一 token M₃ 提供证据。', test:'可证伪预测：clean 第 k 个 marker 的 retrieval-head activation patch 应恢复 corrupt run 的 Mₖ logit margin。'},
          {title:'Step 3 · 写出 marker 并更新进度', read:'读取：retrieved marker identity 与此前 trace。', write:'写入：M₃ token 及“已经完成 3 次有效检索”的 progress/count state。', test:'可证伪预测：trace-marker residual transplant 应改变下一 index/close 决策，而不只是复制 marker identity。'},
          {title:'Step 4 · successor 与循环', read:'读取：当前 marker 后 residual/progress。', write:'写入：下一数字 4，再以 <4> 查询第 4 个 prompt needle；无下一 needle 时生成 </Think>。', test:'可证伪预测：若只是看前一数字做 +1，移除 prompt retrieval 不应破坏 marker；若是纯检索，successor/close 又不应依赖 progress state。'},
          {title:'Step 5 · trace-mediated final readout', read:'读取：完整 trace、trace 长度/进度 residual，以及可能仍可访问的 prompt。', write:'输出：最终共享数字 token <Cₙ>。', test:'可证伪预测：固定或冲突 trace、final residual patching 可区分答案究竟跟随 prompt count、trace count，还是二者的混合。'}
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
    retrieval = pd.read_csv(tables / "retrieval_control_patching_by_bin.csv")
    nested = pd.read_csv(tables / "nested_head_patching_regression_by_bin.csv")
    steering = pd.read_csv(tables / "geometry_steering_by_bin.csv")
    steering_gain = pd.read_csv(tables / "geometry_steering_gain_by_bin.csv")
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
        "retrieval": figures / "retrieval_patching_by_count_bin.png",
        "nested": figures / "nested_head_patching_by_count_bin.png",
        "steering": figures / "geometry_steering_by_count_bin.png",
        "transplant": figures / "residual_transplant_by_count_bin.png",
        "pca_variance": figures / "pca_variance_by_site_layer.png",
        "pca_static": figures / "pca_count_mean_static.png",
    }
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
        show_bottom=True,
        title="Non-thinking broad-head ablation: final-count accuracy",
        path=generated["cumulative_nonthinking"],
    )
    save_cumulative_ablation_panel(
        cumulative,
        mode="thinking",
        metric="trace_marker_accuracy",
        family="targeted_retrieval_top",
        show_bottom=True,
        title="CoT targeted-head ablation: trace-marker accuracy",
        path=generated["cumulative_targeted_trace"],
    )
    save_cumulative_ablation_panel(
        cumulative,
        mode="thinking",
        metric="final_count_accuracy",
        family="targeted_retrieval_top",
        show_bottom=True,
        title="CoT targeted-head ablation: final-count accuracy",
        path=generated["cumulative_targeted_final"],
    )
    save_cumulative_ablation_panel(
        cumulative,
        mode="thinking",
        metric="final_count_accuracy",
        family="trace_readout_top",
        show_bottom=False,
        title="CoT trace-readout-head ablation: final-count accuracy",
        path=generated["cumulative_readout_final"],
    )
    save_retrieval_patch(retrieval, generated["retrieval"])
    save_nested_patch(nested, generated["nested"])
    save_steering(steering, generated["steering"])
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
            & (cumulative["replicate"] == 0)
            & (cumulative["count_bin"] == count_bin)
            & (cumulative["top_n"] == top_n)
        ]
        return "n/a" if frame.empty else pct(frame.iloc[0][metric])

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
                "readout_final_top4": remaining_accuracy(
                    mode="thinking", family="trace_readout_top", count_bin=count_bin, top_n=4,
                    metric="final_count_accuracy",
                ),
                "readout_final_top8": remaining_accuracy(
                    mode="thinking", family="trace_readout_top", count_bin=count_bin, top_n=8,
                    metric="final_count_accuracy",
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
    alignment_lookup = {
        (row.mode, row.count_bin, row.descriptive_score): fmt(row.spearman_rho)
        for row in alignment.itertuples(index=False)
    }

    patch_rows = []
    for count_bin in COUNT_BINS:
        retrieval_top4 = retrieval[
            (retrieval.family == "targeted_top")
            & (retrieval.query_role == "final")
            & (retrieval.count_bin == count_bin)
            & (retrieval.top_n == 4)
        ].normalized_recovery.mean()
        direct_top4 = select_row(
            nested,
            mode="nonthinking",
            family="primary_top",
            replicate=0,
            count_bin=count_bin,
            top_n=4,
        )
        cot_top4 = select_row(
            nested,
            mode="thinking",
            family="primary_top",
            replicate=0,
            count_bin=count_bin,
            top_n=4,
        )
        patch_rows.append(
            {
                "bin": count_bin,
                "retrieval": fmt(retrieval_top4),
                "direct": fmt(direct_top4.slope),
                "cot": fmt(cot_top4.slope),
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
    .metric-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:16px 0 26px}.metric-card{border:1px solid #d3ddea;border-radius:7px;background:#fbfdff;padding:16px 17px;min-width:0}.metric-card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:11px}.metric-card-head code{font-size:14px;font-weight:750;color:#1e3a5f;background:#e7f0fb}.metric-context{font-size:12px;text-align:right;color:#526078;line-height:1.35}.metric-context span{display:block;text-transform:uppercase;font-size:10px;font-weight:800;color:#7c8ba1;letter-spacing:.04em}.metric-card p{font-size:14px;line-height:1.58;color:#44526a;margin:11px 0 0}.equation{display:flex;align-items:center;justify-content:center;min-height:92px;text-align:center;color:#111827;background:#fff;border:1px solid #e0e7f0;border-radius:5px;padding:12px 14px;overflow-x:auto;overflow-y:hidden}.equation.compact{min-height:64px;margin-top:8px}.equation math{display:block;margin:auto;font-family:"STIX Two Math","Latin Modern Math","Cambria Math",serif;font-size:24px;line-height:1.25}.equation mtext{font-family:Inter,"Noto Sans SC","Microsoft YaHei",Arial,sans-serif;font-size:.72em}.formula-block{border:1px solid var(--line);background:#f8fafc;padding:15px 17px;margin:13px 0;border-radius:6px}.formula-block h4{font-size:15px;margin:0 0 9px}.formula-block p{font-size:14.5px;color:#44526a;margin:10px 0 0}.symbol-key{display:flex;flex-wrap:wrap;gap:8px 18px;margin:10px 0}.symbol-key span{font-size:14px;color:#44526a}.symbol-key b{font-family:"STIX Two Math","Latin Modern Math","Cambria Math",serif;font-size:17px;color:#172033}.mechanism-explorer{border:1px solid var(--line);border-radius:8px;padding:20px;margin:22px 0;background:#fff}.mechanism-explorer-head{display:flex;justify-content:space-between;gap:24px;align-items:flex-start}.mechanism-explorer-head p{margin:5px 0 0;color:var(--muted);font-size:14.5px;max-width:780px}.mechanism-mode-switch{display:flex;min-width:290px;border:1px solid #aebbd0;border-radius:6px;overflow:hidden}.mechanism-mode{flex:1;border:0;border-right:1px solid #aebbd0;background:#fff;color:#334155;padding:10px 13px;font:inherit;font-size:14px;font-weight:700;cursor:pointer}.mechanism-mode:last-child{border-right:0}.mechanism-mode.active{background:#1d4ed8;color:#fff}.mechanism-player{margin-top:16px;border:1px solid var(--line);border-radius:6px;background:#fbfdff;overflow:hidden}.mechanism-scene{display:none;width:100%;height:auto;min-height:390px}.mechanism-scene.active{display:block}.mechanism-scene text{font-family:Inter,"Noto Sans SC","Microsoft YaHei",Arial,sans-serif;fill:#243047;font-size:16px;text-anchor:middle}.mechanism-scene .mech-region{fill:#f7faff;stroke:#b9c6d8;stroke-width:2}.mechanism-scene .mech-region.trace{fill:#f2fbf5;stroke:#a6d4b2}.mechanism-scene .mech-region-title{text-anchor:start;font-size:17px;font-weight:750}.mechanism-scene .mech-node rect{stroke-width:2}.mechanism-scene .mech-node.needle rect,.mechanism-scene .mech-node.trace-marker rect{fill:#dcfce7;stroke:#22a55b}.mechanism-scene .mech-node.query rect,.mechanism-scene .mech-node.trace-token rect{fill:#dbeafe;stroke:#3b82f6}.mechanism-scene .mech-node.state rect{fill:#fff7ed;stroke:#f97316}.mechanism-scene .mech-node.transform rect{fill:#f3e8ff;stroke:#8b5cf6}.mechanism-scene .mech-node.answer rect{fill:#fee2e2;stroke:#dc2626}.mechanism-scene .mech-node.target rect,.mechanism-scene .mech-node.active-marker rect,.mechanism-scene .mech-node.successor rect{stroke-width:4}.mechanism-scene .mech-node text{font-weight:750}.mechanism-scene .mech-subtext{font-size:13px;font-weight:500;fill:#526078}.mechanism-scene .mech-answer-token{font-size:18px;font-weight:850;fill:#b91c1c}.mechanism-scene .mech-noise circle{fill:#cbd5e1}.mechanism-scene .mech-ellipsis{font-size:25px}.mechanism-scene .mech-step{opacity:.1;transition:opacity .45s ease}.mechanism-scene .mech-step.visible{opacity:1}.mechanism-scene .mech-step.current .mech-node{filter:drop-shadow(0 0 7px rgba(37,99,235,.33))}.mechanism-scene .mech-links path,.mechanism-scene .mech-flow-arrow,.mechanism-scene .mech-loop-arrow{fill:none;stroke:#2563eb;stroke-width:3;color:#2563eb;marker-end:url(#direct-arrow)}.mechanism-scene .targeted-link path,.mechanism-scene .mech-flow-arrow.cot,.mechanism-scene .mech-loop-arrow{stroke:#16a34a;color:#16a34a;marker-end:url(#cot-arrow)}.mechanism-scene .aggregate-links path{stroke:#ea580c;color:#ea580c}.mechanism-scene .mech-step.current path{stroke-dasharray:9 7;animation:mechanismDash 1.1s linear infinite}.mechanism-timeline{display:flex;align-items:center;justify-content:center;gap:14px;margin:16px 0 12px}.mechanism-timeline>button{border:1px solid #aebbd0;border-radius:5px;background:#fff;color:#243047;min-width:42px;height:38px;font-size:23px;cursor:pointer}.mechanism-timeline .mechanism-play{font-size:14px;min-width:92px;font-weight:700}.mechanism-dots{display:flex;gap:10px}.mechanism-dot{width:12px;height:12px;border:1px solid #64748b;border-radius:50%;padding:0;background:#fff;cursor:pointer}.mechanism-dot.active{background:#2563eb;border-color:#2563eb;box-shadow:0 0 0 4px #dbeafe}.mechanism-step-copy{border-left:5px solid #2563eb;border-radius:0 7px 7px 0;background:#eff6ff;padding:13px 17px}.mechanism-step-title{font-size:17px;font-weight:800;margin-bottom:7px}.mechanism-step-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.mechanism-step-grid p{margin:0;background:rgba(255,255,255,.74);border:1px solid #d7e3f5;border-radius:5px;padding:9px 11px;font-size:14px;line-height:1.55}.mechanism-explorer figcaption{color:#44526a;font-size:14.5px;margin-top:13px}@keyframes mechanismDash{to{stroke-dashoffset:-32px}}@media(prefers-reduced-motion:reduce){.mechanism-scene .mech-step{transition:none}.mechanism-scene .mech-step.current path{animation:none}}
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
        {"item": "正式 ablation", "value": f"每个 exact count {manifest['ablation_examples_per_exact_count']} 个新 prompts，共 {30*manifest['ablation_examples_per_exact_count']} 个；16 个单头；top-1 至 top-16；bottom 与 {manifest['ablation_random_orders']} 个随机顺序。"},
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
            "aspect": "直接优化的输出",
            "direct": "给定 prompt，紧接 <Ans> 预测唯一的 count token C_n；没有可观察的中间步骤。",
            "cot": "先生成 1,M1,2,M2,…,n,Mn 的 indexed trace，再在 </Think><Ans> 后预测同一套数字 token C_n。",
        },
        {
            "aspect": "候选 attention 路由",
            "direct": "final query 并行访问多个 prompt needles；候选 broad heads 同时满足较高 needle mass 与较高 needle-subset entropy。",
            "cot": "每个数字 query <k> 定向访问排序后的第 k 个 prompt needle；候选 targeted heads 具有较高 k-to-k mass。",
        },
        {
            "aspect": "候选计数状态",
            "direct": "多个 needle value 的聚合结果被写入 <Ans> residual；后层把分布式 cardinality state 变为 count logits。",
            "cot": "每次成功检索写出 M_k，并在 trace token residual 中更新 progress/count state；successor 决定 k+1 或关闭 trace。",
        },
        {
            "aspect": "最终答案从哪里读",
            "direct": "直接从 natural final-answer query 的最后 residual 读出。",
            "cot": "可能从完整 trace、最终 progress state、prompt，或这些来源的混合中读出；固定/冲突 trace 干预用于拆分来源。",
        },
        {
            "aspect": "关键因果预测",
            "direct": "mask broad heads 应降低 final accuracy；donor head/residual patch 应让预测随 donor count 移动。",
            "cot": "targeted patch 应恢复 M_k；progress transplant 应改变 next-index/close；final-state patch 应搬运最终 count。",
        },
        {
            "aspect": "什么结果会推翻纯版本",
            "direct": "若有效 head 只盯单个位置，或 count residual 无法被搬运，则“纯 broad aggregation”不足。",
            "cot": "若 marker 预测不依赖 matching needle，或最终答案完全忽略 trace/progress，则“纯 retrieval loop”不足。",
        },
    ]
    object_section = f"""
    <section id="object">
      <h2>1. 研究对象、目标与两个机制假设</h2>
      <p>两个模型看到同一类 256-token prompt、同一套 marker tokens 和同一套 count tokens，模型规模也完全相同；差别只在于<b>被要求生成什么输出序列</b>。Non-thinking 模型必须把 prompt 直接压缩成一个 count，CoT 模型必须先把每个被计数对象以 indexed trace 外显出来，再回答 count。我们关心的不是给这两种格式贴一个名字，而是检验它们是否诱导了不同的信息路由、状态表示与最终读出路径。</p>
      <div class="callout warn"><b>假设边界。</b>下面是两个可检验的机制模型，不是先验结论。Non-thinking 仍可能在内部形成多阶段计算；CoT 也可能同时使用 targeted retrieval、broad aggregation 和位置捷径。真实机制可以是两者的混合，后文的 mask、patch、steering 与 transplant 用来判断每个环节的必要性和局部充分性。</div>
      <div class="mechanisms">
        <div class="mechanism">
          <h3>假设 A · Non-thinking：并行集合聚合后直接读出</h3>
          <p><b>任务分解：</b>把 prompt 中散布的 needle token 看成一个无序集合，计算其基数 <code>n</code>，然后在唯一的 <code>&lt;Ans&gt;</code> query 输出 <code>&lt;C_n&gt;</code>。</p>
          <p><b>候选算法：</b>早期 broad heads 从 final query 并行读取多个 needles 的 value；这些 head outputs 与 MLP 在 residual stream 中形成分布式 cardinality state；后层把连续状态离散化为 30 个 count-token logits。它不需要在输出序列中保存“现在数到第几个”。</p>
          <div class="flow"><span class="node">needle set</span><span class="arrow">→</span><span class="node">parallel broad retrieval</span><span class="arrow">→</span><span class="node">count residual</span><span class="arrow">→</span><span class="node">C_n</span></div>
        </div>
        <div class="mechanism">
          <h3>假设 B · Thinking：逐项检索、进度更新、trace-mediated 读出</h3>
          <p><b>任务分解：</b>对 <code>k=1,2,…</code>，用数字 query <code>&lt;k&gt;</code> 找到按位置排序的第 k 个 prompt needle，写出其 marker <code>M_k</code>；随后生成 <code>k+1</code>，重复直到关闭 trace，最后输出 <code>&lt;C_n&gt;</code>。</p>
          <p><b>候选算法：</b>targeted heads 负责 k-to-k retrieval；marker 后的 residual 保存已完成检索次数或 progress；successor/close circuit 决定继续还是停止；final query 再从 trace/progress state 读出 count。这一假设不等同于“模型只看上一个数字做 +1”。</p>
          <div class="flow"><span class="node">index k</span><span class="arrow">→</span><span class="node">needle k</span><span class="arrow">→</span><span class="node">M_k + progress</span><span class="arrow">→</span><span class="node">k+1 / close</span><span class="arrow">→</span><span class="node">C_n</span></div>
        </div>
      </div>
      {mechanism_explorer()}
      <h3>两种机制的可区分预测</h3>
      {table(mechanism_rows, [('aspect','比较维度'),('direct','Non-thinking：direct aggregation'),('cot','Thinking：indexed retrieval loop')])}
      <div class="callout"><b>证据等级。</b>attention 与 PCA 只能给出候选结构；probe 只说明状态可被读取；global head mask 测全局必要性；clean-to-corrupt head-output patch 测某个 query 上的局部充分性；donor-to-receiver residual transplant 测完整状态是否足以搬运 count。只有多种证据方向一致时，才把某个组件称为候选 circuit。</div>
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
        <li><b>累计消融。</b>先按第 5 节定义的分数给 heads 排名，再依次 mask top-1、top-2、…、top-16。Non-thinking 使用 <code>broad_attention_score</code>；CoT 分别使用 <code>k-to-k raw mass</code>、<code>trace_markers_mass</code> 与 successor score。曲线同时与倒序/bottom heads和 {manifest['ablation_random_orders']} 个固定随机顺序比较；只有“高分 top 曲线比 bottom/random 更早下降”才说明该排名捕捉了特异机制。</li>
        <li><b>Teacher-forced final-count accuracy。</b>完整 gold prefix 已在输入中；在 <code>&lt;Ans&gt;</code> 位置读取 logits，若 argmax 数字 token 等于 gold count 则正确。CoT 中 gold trace 仍存在于左侧，所以该指标检验“给定正确 trace 后的最终 readout”，不是自由生成整条 trace 的端到端准确率。</li>
        <li><b>Teacher-forced trace-marker accuracy。</b>对 CoT 的每个数字 <code>&lt;k&gt;</code> query，检查其 next-token argmax 是否为 gold marker <code>M_k</code>，再对所有有效 k queries 求平均。后一步仍看到数据中的 gold 早期 trace token；某一步预测错不会被写回上下文并污染后续步骤。</li>
        <li><b>Teacher-forced trace-index accuracy。</b>在 <code>&lt;Think&gt;</code> 或前一个 marker <code>M_(k-1)</code> 的 query 上，检查 next-token argmax 是否为数字 <code>&lt;k&gt;</code>。它测 successor/index 生成，与 marker identity retrieval 是不同子任务。</li>
      </ol></div>
      <div class="callout warn"><b>必要性不等于充分性。</b>Mask 后性能下降说明该组件在当前网络中被使用；不下降可能表示它无关，也可能表示其他 heads 冗余补偿。反过来，单头 attention mass 很高也不保证删除它会伤性能。第 8 节的 clean-to-corrupt patching 才检验候选 activation 是否能局部搬运所需信息。</div>

      <p><b>阅读顺序。</b>下面每个小节都严格按照“实验 → 结果 → 分析”组织。实验段只说明干预和指标；结果段先给图表与观测数值；分析段才讨论这些结果支持或排除什么机制。</p>

      <h3>7.2 消融结果总览：单头必要性与 top-n 剂量曲线</h3>
      <h4>实验</h4>
      <p>先对 16 枚 Layer×head 做逐头 global mask，再分别按 non-thinking <code>broad_attention_score</code>、CoT <code>k-to-k raw mass</code> 与 CoT <code>trace_markers_mass</code> 从高到低累计 mask top-n。单头实验回答“哪一枚 head 单独不可替代”；累计实验回答“同类候选 heads 是否存在冗余，以及删到多少枚时行为开始崩溃”。</p>
      <h4>结果</h4>
      {figure(
          generated['single'],
          'Figure 4A. 单头 global ablation：每枚 head 被删除后的性能下降',
          '<b>横轴</b>为 head 0–3，<b>纵轴</b>为 Layer 1–4。每个单元格是 baseline accuracy 减去单头 mask 后 accuracy；数值越大，说明该 head 单独删除造成的损伤越大。三列为 count 1–10、11–20、21–30。第一行测 non-thinking final-count，第二行测 CoT trace-marker，第三行测 CoT teacher-forced final-count。mask 覆盖整条 sequence，因此本图定位的是“整枚 head 的必要性”，还不能确定它究竟在哪个 token query 上起作用。'
      )}
      {table(
          ablation_rows,
          [('bin','count 区间'),('direct_head','最强 non-thinking final 单头'),('direct_drop','final drop'),('trace_head','最强 CoT trace 单头'),('trace_drop','trace-marker drop'),('cot_final_head','最强 CoT final 单头'),('cot_final_drop','final drop')],
      )}
      {figure(
          generated['cumulative'],
          'Figure 4B. 按机制分数累计 mask：从单头冗余到成组失效',
          '<b>横轴</b>为同时 global mask 的 head 数 top-n，<b>纵轴</b>为干预后剩余的绝对 accuracy；虚线 1 是未干预 baseline。四行依次为 non-thinking broad→final count、CoT targeted→trace marker、CoT targeted→final count、CoT trace-readout→final count；三列为 count 区间。蓝线按相应机制分数从高到低删除，红线为可用时的 bottom 排序，灰线与阴影为固定随机顺序的均值和 min–max。高分曲线若比 bottom/random 更早下降，才说明描述性排序捕捉到了特异机制。'
      )}
      {table(
          cumulative_ablation_rows,
          [('bin','count 区间'),('direct_top1','NT broad top-1 后 final'),('direct_top2','NT broad top-2 后 final'),('target_trace_top1','CoT target top-1 后 trace'),('target_trace_top4','top-4 后 trace'),('target_trace_top8','top-8 后 trace'),('target_final_top8','target top-8 后 final'),('readout_final_top4','readout top-4 后 final'),('readout_final_top8','readout top-8 后 final')],
      )}
      <h4>分析</h4>
      <p>两张图要结合阅读：单头 drop 高说明该 head 在当前网络中较难被替代；单头 drop 低但累计 top-n 很快下降，则说明同类 heads 彼此补偿。后续 7.3–7.5 分别把这两类证据用于 non-thinking broad aggregation、CoT targeted retrieval 和 CoT trace readout。</p>

      <h3>7.3 Non-thinking：early broad heads 对直接 final count 具有强必要性</h3>
      <h4>实验</h4>
      <p>Non-thinking 没有逐 k trace。我们按第 5 节定义的 <code>broad_attention_score</code> 排序 heads，先逐头 mask，再从 top-1 开始累计 mask；在每个 count 区间都以 non-thinking final-count accuracy 为结果变量，并与 bottom/random 删除顺序比较。</p>
      <h4>结果</h4>
      <div class="callout good">描述性排名稳定的 <code>L1H3</code> 同时是中高 count 最强的单头因果组件：11–20 的最大 final-accuracy drop 为 <b>{ablation_lookup['11-20']['direct_drop']}</b>，21–30 为 <b>{ablation_lookup['21-30']['direct_drop']}</b>。1–10 的最强单头为 <code>{ablation_lookup['1-10']['direct_head']}</code>，drop 为 <b>{ablation_lookup['1-10']['direct_drop']}</b>。按 broad score 累计 mask top-1 后，三个区间剩余 accuracy 为 <b>{cumulative_ablation_lookup['1-10']['direct_top1']}</b> / <b>{cumulative_ablation_lookup['11-20']['direct_top1']}</b> / <b>{cumulative_ablation_lookup['21-30']['direct_top1']}</b>；累计到 top-2 后为 <b>{cumulative_ablation_lookup['1-10']['direct_top2']}</b> / <b>{cumulative_ablation_lookup['11-20']['direct_top2']}</b> / <b>{cumulative_ablation_lookup['21-30']['direct_top2']}</b>。</div>
      <h4>分析</h4>
      <p>结果支持 Layer 1 broad routing 是 direct counting 的必要入口：同一批早期 heads 既在描述性 attention 中广泛覆盖 prompt needles，又在删除后造成大幅 final-count 损伤。低 count 的最强单头与中高 count 略有差异，说明容易样本有更多可替代路径；但 top-2 累计删除后三段都接近崩溃，说明 broad circuit 整体不是可有可无的伴随现象。</p>

      <h3>7.4 CoT targeted retrieval：单头可替代，但成组删除破坏 marker trace</h3>
      <h4>实验</h4>
      <p>在 CoT 的数字 <code>&lt;k&gt;</code> query 上，以第 k 个 prompt needle 的 raw attention mass 排名 targeted heads。逐头或累计 mask 后，首先测 teacher-forced trace-marker accuracy；同时测最终 teacher-forced final-count accuracy，用于区分“局部 marker retrieval 已损坏”与“给定 gold trace 后的最终 readout 已损坏”。</p>
      <h4>结果</h4>
      <div class="callout good">第 5 节 raw k-to-k mass 最高的是 <code>L4H2</code>，其次包括 <code>L3H1</code> 和 <code>L3H0</code>；但三个区间中，单头 mask 后 trace-marker drop 最大的都是 <code>{ablation_lookup['1-10']['trace_head']}</code> 一类早期 heads，最大 drop 分别为 <b>{ablation_lookup['1-10']['trace_drop']}</b> / <b>{ablation_lookup['11-20']['trace_drop']}</b> / <b>{ablation_lookup['21-30']['trace_drop']}</b>。只删 targeted top-1 时，三个区间的 trace-marker accuracy 仍为 <b>{cumulative_ablation_lookup['1-10']['target_trace_top1']}</b> / <b>{cumulative_ablation_lookup['11-20']['target_trace_top1']}</b> / <b>{cumulative_ablation_lookup['21-30']['target_trace_top1']}</b>；删 top-4 后降到 <b>{cumulative_ablation_lookup['1-10']['target_trace_top4']}</b> / <b>{cumulative_ablation_lookup['11-20']['target_trace_top4']}</b> / <b>{cumulative_ablation_lookup['21-30']['target_trace_top4']}</b>，删 top-8 后进一步降到 <b>{cumulative_ablation_lookup['1-10']['target_trace_top8']}</b> / <b>{cumulative_ablation_lookup['11-20']['target_trace_top8']}</b> / <b>{cumulative_ablation_lookup['21-30']['target_trace_top8']}</b>。同一 top-8 干预下，final-count accuracy 仍为 <b>{cumulative_ablation_lookup['1-10']['target_final_top8']}</b> / <b>{cumulative_ablation_lookup['11-20']['target_final_top8']}</b> / <b>{cumulative_ablation_lookup['21-30']['target_final_top8']}</b>。</div>
      <h4>分析</h4>
      <p>“最尖锐地指向 matching needle”与“单独删除时最不可替代”不是同一性质。top-1 几乎无损、top-4 开始下降、top-8 接近崩溃，说明 targeted retrieval 由多枚可互补的 routing heads 共同实现，而不是一枚唯一的“第 k 个 needle head”。final count 仍稳定也不能解释为 trace 无用：teacher-forced 输入已经给出 gold marker tokens，最终 <code>&lt;Ans&gt;</code> 可以绕过被损坏的生成步骤直接读取正确 trace。检验自由生成中的级联失败仍需要 autoregressive ablation。</p>

      <h3>7.5 CoT trace readout：与 targeted retrieval 是不同候选组</h3>
      <h4>实验</h4>
      <p>Targeted ranking 读取数字 <code>&lt;k&gt;</code> query；本实验改在最终 <code>&lt;Ans&gt;</code> query 上，以投向全部 trace marker positions 的 <code>trace_markers_mass</code> 排名 heads，再累计 mask top-n，并测 teacher-forced final-count accuracy。这样能检验“从已给定 trace 读出 count”的候选组，而不是再次检验 prompt retrieval。</p>
      <h4>结果</h4>
      <div class="callout">描述性 readout ranking 以前列 <code>L2H3</code>、<code>L2H2</code>、<code>L4H1</code>、<code>L4H0</code> 为主，与 k-to-k targeted ranking 不同。删 readout top-4 后，三个区间的 final-count accuracy 为 <b>{cumulative_ablation_lookup['1-10']['readout_final_top4']}</b> / <b>{cumulative_ablation_lookup['11-20']['readout_final_top4']}</b> / <b>{cumulative_ablation_lookup['21-30']['readout_final_top4']}</b>；删 top-8 后为 <b>{cumulative_ablation_lookup['1-10']['readout_final_top8']}</b> / <b>{cumulative_ablation_lookup['11-20']['readout_final_top8']}</b> / <b>{cumulative_ablation_lookup['21-30']['readout_final_top8']}</b>。累计曲线并不严格单调。</div>
      <h4>分析</h4>
      <p>Targeted retrieval 与 final trace readout 是两个不同阶段的候选 circuit。低 count 在 readout 删除下更早受损，中高 count 仍保留明显冗余；非单调曲线则说明 global mask 同时改变了多个相互补偿或竞争的通路，所以不能把“再增加一枚被 mask head”带来的边际差当作该 head 的独立贡献。第 8 节需要把 patch 限定在 <code>&lt;Ans&gt;</code> query，才能进一步确认 readout activation 是否足以搬运 count 信息。</p>

      <h3>7.6 描述性 attention score 与因果 drop 是否一致</h3>
      <h4>实验</h4>
      <p>对每个机制和 count 区间，把 16 枚 heads 的描述性 attention score 与对应单头 global-ablation accuracy drop 对齐，并计算 Spearman 秩相关。该检验不要求两者线性，只问“描述分数排名更高的 head，是否通常也造成更大因果损伤”。</p>
      <h4>结果</h4>
      {figure(
          generated['ablation_alignment'],
          'Figure 4C. Attention score 与单头 ablation drop 的逐 head 对齐',
          '<b>每个点</b>是一枚 Layer×head；只标注描述分数最高 4 个与因果 drop 最高 4 个 head 的并集，标签使用 1-based Layer 与 0-based head。<b>三行</b>依次比较 non-thinking broad score→final-count drop、CoT raw k-to-k mass→trace-marker drop、CoT trace-readout mass→final-count drop；<b>三列</b>为 count 1–10、11–20、21–30。横轴是描述性 score，纵轴是 mask 后 accuracy drop；标题给出全部 16 枚 heads 的 Spearman ρ。'
      )}
      {table(alignment_rows,[('mechanism','比较'),('bin','count 区间'),('rho','Spearman ρ'),('n','heads')])}
      <div class="callout">Non-thinking broad score 与 final-count drop 的 Spearman ρ 在三个区间分别为 <b>{alignment_lookup[('nonthinking','1-10','broad_attention_score')]}</b> / <b>{alignment_lookup[('nonthinking','11-20','broad_attention_score')]}</b> / <b>{alignment_lookup[('nonthinking','21-30','broad_attention_score')]}</b>。CoT k-to-k mass 与 trace-marker drop 分别为 <b>{alignment_lookup[('thinking','1-10','correct_prompt_needle_mass')]}</b> / <b>{alignment_lookup[('thinking','11-20','correct_prompt_needle_mass')]}</b> / <b>{alignment_lookup[('thinking','21-30','correct_prompt_needle_mass')]}</b>。CoT readout mass 与 final-count drop 分别为 <b>{alignment_lookup[('thinking','1-10','trace_markers_mass')]}</b> / <b>{alignment_lookup[('thinking','11-20','trace_markers_mass')]}</b> / <b>{alignment_lookup[('thinking','21-30','trace_markers_mass')]}</b>；最后一个区间因单头 drop 缺少方差而无法定义秩相关。</div>
      <h4>分析</h4>
      <p>Non-thinking broad score 与必要性呈中等正相关，说明描述性 broad attention 对 direct-count circuit 有实际筛选价值。CoT k-to-k 的相关为负，进一步说明 raw matching mass 不能等同于单头不可替代性：尖锐 retrieval heads 可能被其他 routing heads 替代，而早期支持性 heads 的删除反而更伤性能。readout score 的相关接近 0，也说明 final readout 不能只靠单头 attention mass 概括。可靠结论必须同时报告描述排序、单头必要性和累计 top-n 剂量曲线。
      </p>

      <h3>7.7 本节结论与下一步</h3>
      <div class="mechanisms">
        <div class="mechanism"><h3>Non-thinking</h3><p><b>当前支持：</b>Layer 1 broad heads 既有描述性 prompt-wide attention，也有强单头/累计必要性；它们很可能是直接 set aggregation 的关键入口。</p><p><b>仍缺：</b>global mask 不能证明这些 heads 把“count 数值”写到了哪里。第 8 节需要在 <code>&lt;Ans&gt;</code> 局部 patch broad-head output，并测 hidden-state/count logits 是否随 donor count 搬运。</p></div>
        <div class="mechanism"><h3>CoT</h3><p><b>当前支持：</b>k-to-k heads 作为一个组对 marker trace 必要，但单头可替代；targeted retrieval 与 final trace readout 是不同阶段、不同排名的多头 circuit。</p><p><b>仍缺：</b>teacher forcing 隔断了局部 trace 错误向最终答案的级联。第 8 节应分别在 <code>&lt;k&gt;</code> 和 <code>&lt;Ans&gt;</code> 做局部 activation patch，并使用 marker-identity margin、next-index margin 与 final-count margin 区分检索、successor 和 readout。</p></div>
      </div>
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

    <section id="ablation"><h2>6. 分层 global head ablation：哪些 heads 对哪个难度区间必要</h2><div class="protocol"><b>样本。</b>本次重新生成每个 exact count {manifest['ablation_examples_per_exact_count']} 个 prompts，共 {30*manifest['ablation_examples_per_exact_count']} 个；两个模型使用各自格式做 teacher-forced forward。<b>干预。</b>用 GPT-2 <code>head_mask</code> 将指定 head 在整条 sequence、所有 query positions 的输出设为 0。单头逐个测 16 次；累计实验按候选排名依次 mask top-1…top-16，并与排名倒序和 {manifest['ablation_random_orders']} 条固定随机顺序比较。<b>指标。</b>remaining accuracy 是 mask 后的绝对 accuracy；drop 是无干预 baseline 减去 mask 后 accuracy。</div>{figure(generated['single'],'Figure 3. 单头必要性按 count 区间分解',single_ablation_caption)}{table(ablation_rows,[('bin','count 区间'),('direct_head','最强 non-thinking 单头'),('direct_drop','final accuracy drop'),('trace_head','最强 CoT trace 单头'),('trace_drop','trace-marker accuracy drop')])}{figure(generated['cumulative'],'Figure 4. Top-n 累计 mask 的剂量曲线与强对照',cumulative_ablation_caption)}</section>

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

    nav_start = report.index('<nav class="toc">')
    nav_end = report.index("</nav>", nav_start) + len("</nav>")
    new_nav = """
    <nav class="toc"><b>目录</b><ol>
      <li><a href="#object">研究对象与机制假设</a></li>
      <li><a href="#setting">数据、模型与 sequence</a></li>
      <li><a href="#definitions">术语、指标与公式</a></li>
      <li><a href="#dynamics">学习动态</a></li>
      <li><a href="#attention">5. 描述性 attention</a></li>
      <li><a href="#geometry">6. 描述性 hidden state</a></li>
      <li><a href="#ablation">7. Attention-head ablation</a></li>
      <li><a href="#patching">8. Attention-head patching</a></li>
      <li><a href="#steering">9. Hidden-state geometry steering</a></li>
      <li><a href="#transplant">10. Hidden-state patching</a></li>
      <li><a href="#interaction">11. Head 与 hidden state 的因果联系</a></li>
      <li><a href="#synthesis">12. 综合结论与边界</a></li>
    </ol></nav>
    """
    report = report[:nav_start] + new_nav + report[nav_end:]

    report = finalize_report_numbering(report)
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
