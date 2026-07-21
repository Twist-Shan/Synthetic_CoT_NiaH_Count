#!/usr/bin/env python3
"""Build the complete Chinese v16.2 report with the full v10 analysis port."""

from __future__ import annotations

import sys

for _optional in ("pyarrow", "numexpr", "bottleneck"):
    sys.modules.setdefault(_optional, None)

import argparse
import base64
import html
import json
import re
import shutil
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

pd.options.mode.string_storage = "python"
pd.options.future.infer_string = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from synthetic_counting_v16_2.interactive_geometry import write_interactive_geometry_table
from synthetic_counting_v16_2.report_readability import (
    REPORT_NAME,
    cleanup_legacy_reports,
    polish_report_html,
)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    try:
        temporary.replace(path)
    except PermissionError:
        # Windows browsers can briefly hold a read handle that permits writes but
        # denies rename-over-existing. Preserve rebuildability for an open report.
        shutil.copyfile(temporary, path)
        temporary.unlink()


def _image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _refresh_embedded_image(document: str, needle: str, image_path: Path) -> str:
    matches = [
        match
        for match in re.finditer(r"<figure\b.*?</figure>", document, flags=re.S)
        if needle in match.group(0)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one report figure for {needle!r}, found {len(matches)}")
    match = matches[0]
    block = match.group(0)
    refreshed, replacements = re.subn(
        r'src="data:image/png;base64,[^"]+"',
        f'src="{_image_uri(image_path)}"',
        block,
        count=1,
        flags=re.S,
    )
    if replacements != 1:
        raise ValueError(f"figure {needle!r} does not contain one embedded PNG")
    return document[: match.start()] + refreshed + document[match.end() :]


def _refresh_core_plot_images(document: str, run_dir: Path) -> str:
    sources = {
        '<span class="figure-tag">图 1。</span>': "checkpoint_mechanism_overview.png",
        '<span class="figure-tag">图 3。</span>': "checkpoint_attention_retrieval_emergence.png",
        '<span class="figure-tag">图 5。</span>': "checkpoint_answer_routing.png",
        '<span class="figure-tag">图 6。</span>': "checkpoint_ordered_trace_retrieval.png",
        '<span class="figure-tag">图 9。</span>': "checkpoint_cross_site_counter_transfer.png",
        '<span class="figure-tag">图 10。</span>': "checkpoint_state_geometry_emergence.png",
        '<span class="figure-tag">图 11。</span>': "checkpoint_representation_stability.png",
    }
    for needle, filename in sources.items():
        document = _refresh_embedded_image(document, needle, run_dir / "figures" / filename)
    return document


def _figure(path: Path, tag: str, title: str, caption: str, alt: str) -> str:
    return f"""
    <figure class="report-figure">
      <h4>{html.escape(title)}</h4>
      <img src="{_image_uri(path)}" alt="{html.escape(alt)}" loading="lazy">
      <figcaption><span class="figure-tag">{html.escape(tag)}</span>{caption}</figcaption>
    </figure>
    """


def _interactive_hidden_state(frame: pd.DataFrame) -> str:
    """Render a V10-style rotatable centroid manifold with checkpoint controls."""

    payload: dict[str, dict[str, object]] = {}
    keys = ["mode", "site", "step", "layer"]
    for group_key, group in frame.groupby(keys, sort=True):
        mode, site, step, layer = group_key
        ordered = group.sort_values("label")
        first = ordered.iloc[0]
        payload[f"{mode}|{site}|{int(step)}|{int(layer)}"] = {
            "p": [
                [int(row.label)] + [round(float(getattr(row, f"pc{axis}")), 6) for axis in range(1, 7)]
                for row in ordered.itertuples()
            ],
            "v": [round(float(first[f"pc{axis}_variance"]), 7) for axis in range(1, 7)],
            "e": round(float(first["effective_dimension"]), 5),
            "a": round(float(first["adjacent_cosine"]), 5),
            "s": round(float(first["straightness"]), 5),
        }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    step_options = "".join(
        f'<option value="{int(step)}"{" selected" if int(step) == int(frame.step.max()) else ""}>'
        f'{int(step):,}</option>'
        for step in sorted(frame.step.unique(), reverse=True)
    )
    axis_options = "".join(
        f'<option value="{left},{middle},{right}">PC{left + 1} / PC{middle + 1} / PC{right + 1}</option>'
        for left, middle, right in combinations(range(6), 3)
    )
    fragment = """
      <figure class="interactive-hidden-state" id="v162-hs3d-root">
        <h4>可交互 3D hidden-state centroid manifold</h4>
        <div class="hs3d-controls">
          <label>Model / semantic site
            <select id="v162-hs3d-site">
              <option value="nonthinking|final_answer">NT · final &lt;Ans&gt;</option>
              <option value="thinking|final_answer">Thinking · final &lt;Ans&gt;</option>
              <option value="thinking|trace_index" selected>Thinking · index &lt;k&gt;</option>
              <option value="thinking|trace_marker">Thinking · marker M_k</option>
            </select>
          </label>
          <label>Checkpoint step<select id="v162-hs3d-step">__STEP_OPTIONS__</select></label>
          <label>Hidden-state depth
            <select id="v162-hs3d-layer">
              <option value="0">Embedding output</option>
              <option value="1">Layer 1 output</option>
              <option value="2">Layer 2 output</option>
              <option value="3" selected>Layer 3 output</option>
              <option value="4">Layer 4 output</option>
            </select>
          </label>
          <label>Displayed axes<select id="v162-hs3d-axes">__AXIS_OPTIONS__</select></label>
          <label>Label range
            <select id="v162-hs3d-range">
              <option value="1,10">1–10</option>
              <option value="1,3">1–3</option>
              <option value="4,7">4–7</option>
              <option value="8,10">8–10</option>
            </select>
          </label>
          <label class="hs3d-check"><input id="v162-hs3d-labels" type="checkbox" checked>显示 count / progress 编号</label>
        </div>
        <div id="v162-hs3d-stats" class="hs3d-stats" aria-live="polite"></div>
        <canvas id="v162-hs3d-canvas" aria-label="可旋转的三维 hidden-state centroid PCA 图"></canvas>
        <div class="hs3d-view-controls" aria-label="三维视角控制">
          <button type="button" data-hs3d-yaw="-0.20">向左旋转</button>
          <button type="button" data-hs3d-yaw="0.20">向右旋转</button>
          <button type="button" data-hs3d-pitch="-0.16">向上旋转</button>
          <button type="button" data-hs3d-pitch="0.16">向下旋转</button>
          <button type="button" id="v162-hs3d-reset">重置视角</button>
        </div>
        <figcaption><span class="figure-tag">图 18A。</span><strong>交互与坐标定义。</strong>拖拽画布或使用按钮旋转；下拉框可选择 model/site、21 个 checkpoint、Embedding/Layer 1–4、PC1–PC6 中任意三轴，以及标签区间。点是 held-out hidden states 按 count <i>n</i> 或 progress <i>k</i> 求得的 256 维类中心，灰线按标签顺序连接。每个 checkpoint×site×Layer 都独立执行 mean-first PCA，因此切换选项后的 PC 方向与尺度不能被当作同一全局坐标系；跨 checkpoint 的定量比较应使用紧邻本图的 variance、effective dimension 与 adjacent cosine，而不是屏幕上的绝对旋转方向。</figcaption>
        <p class="metric-definition"><strong>派生量的计算。</strong>设十个中心为 μ<sub>1</sub>,…,μ<sub>10</sub>，PCA 方差占比为 r<sub>j</sub>。Effective dimension = 1/Σ<sub>j</sub>r<sub>j</sub><sup>2</sup>；adjacent cosine 是相邻位移 Δ<sub>i</sub>=μ<sub>i+1</sub>−μ<sub>i</sub> 之间余弦的均值；path straightness = ‖μ<sub>10</sub>−μ<sub>1</sub>‖/Σ<sub>i</sub>‖Δ<sub>i</sub>‖。若初始化时所有中心完全重合，则这些退化量按 0 报告。</p>
      </figure>
      <script>
      (() => {
        const root = document.getElementById('v162-hs3d-root');
        if (!root) return;
        const data = __DATA_JSON__;
        const site = document.getElementById('v162-hs3d-site');
        const step = document.getElementById('v162-hs3d-step');
        const layer = document.getElementById('v162-hs3d-layer');
        const axes = document.getElementById('v162-hs3d-axes');
        const range = document.getElementById('v162-hs3d-range');
        const labels = document.getElementById('v162-hs3d-labels');
        const stats = document.getElementById('v162-hs3d-stats');
        const canvas = document.getElementById('v162-hs3d-canvas');
        const context = canvas.getContext('2d');
        let yaw = -0.68;
        let pitch = 0.42;
        let dragging = false;
        let lastX = 0;
        let lastY = 0;
        const key = () => `${site.value}|${step.value}|${layer.value}`;
        const rotate = vector => {
          const cy = Math.cos(yaw), sy = Math.sin(yaw);
          const cp = Math.cos(pitch), sp = Math.sin(pitch);
          const x = cy * vector[0] + sy * vector[2];
          const z = -sy * vector[0] + cy * vector[2];
          return [x, cp * vector[1] - sp * z, sp * vector[1] + cp * z];
        };
        const color = value => `hsl(${220 - 210 * (value - 1) / 9},72%,46%)`;
        const drawText = (text, x, y, align = 'left') => {
          context.textAlign = align;
          context.lineWidth = 3;
          context.strokeStyle = 'rgba(255,255,255,.92)';
          context.strokeText(text, x, y);
          context.fillStyle = '#172235';
          context.fillText(text, x, y);
        };
        function draw() {
          const rect = canvas.getBoundingClientRect();
          const dpr = window.devicePixelRatio || 1;
          canvas.width = Math.max(1, Math.round(rect.width * dpr));
          canvas.height = Math.max(1, Math.round(rect.height * dpr));
          context.setTransform(dpr, 0, 0, dpr, 0, 0);
          context.clearRect(0, 0, rect.width, rect.height);
          context.fillStyle = '#fbfdff';
          context.fillRect(0, 0, rect.width, rect.height);
          const item = data[key()];
          if (!item) {
            stats.textContent = '当前组合没有保存的 hidden states。';
            return;
          }
          const ids = axes.value.split(',').map(Number);
          const limits = range.value.split(',').map(Number);
          const raw = item.p
            .filter(point => point[0] >= limits[0] && point[0] <= limits[1])
            .map(point => ({label: point[0], value: ids.map(axis => point[axis + 1])}));
          const rotated = raw.map(point => ({label: point.label, value: rotate(point.value)}));
          const xs = rotated.map(point => point.value[0]);
          const ys = rotated.map(point => point.value[1]);
          const minX = Math.min(...xs), maxX = Math.max(...xs);
          const minY = Math.min(...ys), maxY = Math.max(...ys);
          const padding = 76;
          const spanX = Math.max(maxX - minX, 1e-6);
          const spanY = Math.max(maxY - minY, 1e-6);
          const scale = Math.min((rect.width - 2 * padding) / spanX, (rect.height - 2 * padding) / spanY) * 0.86;
          const centerX = rect.width / 2 - scale * (minX + maxX) / 2;
          const centerY = rect.height / 2 + scale * (minY + maxY) / 2;
          const points = rotated.map(point => ({
            label: point.label,
            x: centerX + scale * point.value[0],
            y: centerY - scale * point.value[1],
            z: point.value[2],
          }));
          context.strokeStyle = '#7b8794';
          context.lineWidth = 1.6;
          context.beginPath();
          points.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
          context.stroke();
          const depthSorted = [...points].sort((left, right) => left.z - right.z);
          depthSorted.forEach(point => {
            context.beginPath();
            context.arc(point.x, point.y, 7, 0, 2 * Math.PI);
            context.fillStyle = color(point.label);
            context.fill();
            context.strokeStyle = '#ffffff';
            context.lineWidth = 1.5;
            context.stroke();
          });
          if (labels.checked) {
            context.font = '12px Inter, Segoe UI, sans-serif';
            const occupied = points.map(point => ({
              left: point.x - 9, right: point.x + 9,
              top: point.y - 9, bottom: point.y + 9,
            }));
            const candidates = [
              {dx: 11, dy: -9, align: 'left'}, {dx: 11, dy: 18, align: 'left'},
              {dx: -11, dy: -9, align: 'right'}, {dx: -11, dy: 18, align: 'right'},
              {dx: 0, dy: -15, align: 'center'}, {dx: 0, dy: 25, align: 'center'},
            ];
            const intersects = (left, right) => !(
              left.right + 3 < right.left || right.right + 3 < left.left ||
              left.bottom + 3 < right.top || right.bottom + 3 < left.top
            );
            [...points].sort((left, right) => left.label - right.label).forEach(point => {
              const text = String(point.label);
              const width = context.measureText(text).width;
              let selected = candidates[candidates.length - 1];
              let box = null;
              for (const candidate of candidates) {
                const x = point.x + candidate.dx;
                const y = point.y + candidate.dy;
                const left = candidate.align === 'left' ? x : candidate.align === 'right' ? x - width : x - width / 2;
                const proposal = {left, right: left + width, top: y - 12, bottom: y + 3};
                const inside = proposal.left >= 4 && proposal.right <= rect.width - 4 && proposal.top >= 4 && proposal.bottom <= rect.height - 4;
                if (inside && !occupied.some(other => intersects(proposal, other))) {
                  selected = candidate;
                  box = proposal;
                  break;
                }
              }
              const labelX = point.x + selected.dx;
              const labelY = point.y + selected.dy;
              if (!box) {
                const left = selected.align === 'left' ? labelX : selected.align === 'right' ? labelX - width : labelX - width / 2;
                box = {left, right: left + width, top: labelY - 12, bottom: labelY + 3};
              }
              occupied.push(box);
              drawText(text, labelX, labelY, selected.align);
            });
          }
          const triadOrigin = [64, rect.height - 52];
          const triadVectors = [[1,0,0], [0,1,0], [0,0,1]].map(rotate);
          context.font = '12px Inter, Segoe UI, sans-serif';
          triadVectors.forEach((vector, index) => {
            const endX = triadOrigin[0] + 34 * vector[0];
            const endY = triadOrigin[1] - 34 * vector[1];
            context.beginPath();
            context.moveTo(triadOrigin[0], triadOrigin[1]);
            context.lineTo(endX, endY);
            context.strokeStyle = ['#2f6f9f', '#d97a3a', '#2f8f6b'][index];
            context.lineWidth = 2;
            context.stroke();
            drawText(`PC${ids[index] + 1}`, endX + 5, endY - 4);
          });
          const axisVariance = ids.map(axis => `${(100 * item.v[axis]).toFixed(1)}%`).join(' / ');
          const cumulative = 100 * item.v.reduce((sum, value) => sum + value, 0);
          stats.textContent = `显示 ${ids.map(axis => `PC${axis + 1}`).join('/')}：解释率 ${axisVariance}；PC1–PC6 累计 ${cumulative.toFixed(1)}%；有效维度 ${item.e.toFixed(2)}；相邻方向余弦 ${item.a.toFixed(3)}；路径直度 ${item.s.toFixed(3)}`;
        }
        [site, step, layer, axes, range, labels].forEach(control => control.addEventListener('change', draw));
        root.querySelectorAll('[data-hs3d-yaw]').forEach(button => button.addEventListener('click', () => {
          yaw += Number(button.dataset.hs3dYaw);
          draw();
        }));
        root.querySelectorAll('[data-hs3d-pitch]').forEach(button => button.addEventListener('click', () => {
          pitch = Math.max(-1.35, Math.min(1.35, pitch + Number(button.dataset.hs3dPitch)));
          draw();
        }));
        document.getElementById('v162-hs3d-reset').addEventListener('click', () => {
          yaw = -0.68;
          pitch = 0.42;
          draw();
        });
        canvas.addEventListener('pointerdown', event => {
          dragging = true;
          lastX = event.clientX;
          lastY = event.clientY;
          canvas.setPointerCapture(event.pointerId);
        });
        canvas.addEventListener('pointermove', event => {
          if (!dragging) return;
          yaw += (event.clientX - lastX) * 0.01;
          pitch = Math.max(-1.35, Math.min(1.35, pitch + (event.clientY - lastY) * 0.01));
          lastX = event.clientX;
          lastY = event.clientY;
          draw();
        });
        const stopDragging = () => { dragging = false; };
        canvas.addEventListener('pointerup', stopDragging);
        canvas.addEventListener('pointercancel', stopDragging);
        new ResizeObserver(draw).observe(canvas);
        draw();
      })();
      </script>
    """
    return (
        fragment.replace("__STEP_OPTIONS__", step_options)
        .replace("__AXIS_OPTIONS__", axis_options)
        .replace("__DATA_JSON__", data_json)
    )


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _pct(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{100 * float(value):.{digits}f}%"


def _table(headers: Iterable[str], rows: Iterable[Iterable[object]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(
        f'<th class="num">{html.escape(str(value))}</th>' if index in numeric else f"<th>{html.escape(str(value))}</th>"
        for index, value in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="num">{value}</td>' if index in numeric else f"<td>{value}</td>"
            for index, value in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return '<div class="table-wrap"><table><thead><tr>' + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def _origin_slope(frame: pd.DataFrame) -> float:
    x = frame["offset"].to_numpy(float)
    y = frame["expected_count_shift"].to_numpy(float)
    return float(x @ y / max(x @ x, 1e-12))


def _section(document: str, section_id: str) -> tuple[int, int, str]:
    match = re.search(rf'<section id="{re.escape(section_id)}">.*?</section>', document, flags=re.S)
    if not match:
        raise ValueError(f"section {section_id!r} not found in base report")
    return match.start(), match.end(), match.group(0)


def _insert_before_section_end(document: str, section_id: str, fragment: str) -> str:
    start, end, block = _section(document, section_id)
    updated = block.rsplit("</section>", 1)[0] + fragment + "\n    </section>"
    return document[:start] + updated + document[end:]


def _summary_values(table_dir: Path) -> dict[str, float]:
    local = pd.read_csv(table_dir / "position_local_head_ablation.csv")
    retrieval = pd.read_csv(table_dir / "retrieval_head_patching.csv")
    successor = pd.read_csv(table_dir / "successor_head_patching.csv")
    head_transport = pd.read_csv(table_dir / "final_query_head_transport.csv")
    residual = pd.read_csv(table_dir / "residual_count_transport.csv")
    early = pd.read_csv(table_dir / "trace_early_stop_patching.csv")
    bridge = pd.read_csv(table_dir / "final_bridge_component_patching.csv")
    state_to_head = pd.read_csv(table_dir / "state_to_head_routing.csv")
    head_to_state = pd.read_csv(table_dir / "head_to_state_geometry.csv")
    feature = pd.read_csv(table_dir / "successor_mlp_feature_patching.csv")

    def mean(frame: pd.DataFrame, column: str, **conditions) -> float:
        subset = frame
        for key, value in conditions.items():
            subset = subset[subset[key] == value]
        return float(subset[column].mean())

    def transport(mode: str, path_kind: str, top_n: int) -> float:
        subset = head_transport[
            (head_transport["mode"] == mode)
            & (head_transport["path_kind"] == path_kind)
            & (head_transport["top_n"] == top_n)
        ]
        return _origin_slope(subset)

    def residual_slope(mode: str, layer: int, intervention: str) -> float:
        subset = residual[
            (residual["mode"] == mode)
            & (residual["layer"] == layer)
            & (residual["intervention"] == intervention)
        ]
        return _origin_slope(subset)

    return {
        "nt_local_top1_accuracy": mean(local, "accuracy", role="nonthinking_broad", path_kind="ranked", top_n=1),
        "nt_local_top1_random": mean(local, "accuracy", role="nonthinking_broad", path_kind="random", top_n=1),
        "target_local_top4_accuracy": mean(local, "accuracy", role="thinking_targeted", path_kind="ranked", top_n=4),
        "target_local_top4_random": mean(local, "accuracy", role="thinking_targeted", path_kind="random", top_n=4),
        "readout_local_top2_accuracy": mean(local, "accuracy", role="thinking_readout", path_kind="ranked", top_n=2),
        "readout_local_top2_random": mean(local, "accuracy", role="thinking_readout", path_kind="random", top_n=2),
        "retrieval_top4": mean(retrieval, "normalized_recovery", path_kind="ranked", top_n=4),
        "retrieval_top4_random": mean(retrieval, "normalized_recovery", path_kind="random", top_n=4),
        "continue_top1": mean(successor, "normalized_recovery", direction="continue_to_close", path_kind="ranked", top_n=1),
        "continue_top4": mean(successor, "normalized_recovery", direction="continue_to_close", path_kind="ranked", top_n=4),
        "close_top1": mean(successor, "normalized_recovery", direction="close_to_continue", path_kind="ranked", top_n=1),
        "close_top4": mean(successor, "normalized_recovery", direction="close_to_continue", path_kind="ranked", top_n=4),
        "nt_head_top4": transport("nonthinking", "ranked", 4),
        "nt_head_top4_random": transport("nonthinking", "random", 4),
        "thinking_head_top2": transport("thinking", "ranked", 2),
        "thinking_head_top1": transport("thinking", "ranked", 1),
        "thinking_head_top2_random": transport("thinking", "random", 2),
        "nt_residual_l4": residual_slope("nonthinking", 4, "centroid_delta_alpha_1"),
        "thinking_residual_l2": residual_slope("thinking", 2, "centroid_delta_alpha_1"),
        "early_l4_shift": mean(early, "close_margin_shift", layer=4),
        "early_l4_flip": mean(early, "patched_close_decision", layer=4),
        "bridge_l2_attn": mean(bridge, "normalized_recovery", layer=2, component="attention_output"),
        "bridge_l4_mlp": mean(bridge, "normalized_recovery", layer=4, component="mlp_output"),
        "state_to_head_l3h3": mean(state_to_head, "routing_shift", layer=3, head=3),
        "nt_state_top1": mean(head_to_state, "state_accuracy", mode="nonthinking", top_n=1),
        "thinking_state_top2": mean(head_to_state, "state_accuracy", mode="thinking", top_n=2),
        "feature_l4_256_continue": mean(feature, "normalized_recovery", layer=4, direction="continue_to_close", path_kind="ranked", support=256),
        "feature_l4_256_close": mean(feature, "normalized_recovery", layer=4, direction="close_to_continue", path_kind="ranked", support=256),
    }


def build(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    output = run_dir / REPORT_NAME
    core = run_dir / "v16_2_rope_core_report_zh.html"
    current = run_dir / "v16_2_rope_complete_report_zh.html"
    if not core.exists():
        if not current.exists():
            if not output.exists():
                raise FileNotFoundError(
                    "the audited Chinese report core or the assembled single report is required"
                )
            document = polish_report_html(output.read_text(encoding="utf-8"))
            document = _refresh_core_plot_images(document, run_dir)
            _atomic_text(output, document)
            cleanup_legacy_reports(run_dir, output)
            return output
        shutil.copy2(current, core)
    document = core.read_text(encoding="utf-8")
    if "<title>" not in document or '<section id="causal">' not in document:
        raise ValueError("base report does not have the expected audited structure")

    table_dir = run_dir / "analysis" / "v10_port" / "tables"
    figure_dir = run_dir / "analysis" / "v10_port" / "figures"
    interactive_geometry = pd.read_csv(write_interactive_geometry_table(run_dir))
    interactive_3d = _interactive_hidden_state(interactive_geometry)
    manifest = json.loads((run_dir / "analysis" / "v10_port" / "manifest.json").read_text(encoding="utf-8"))
    values = _summary_values(table_dir)
    geometry = pd.read_csv(table_dir / "representation_geometry.csv")
    crosswalk = pd.read_csv(table_dir / "analysis_crosswalk.csv")
    conflicts = pd.read_csv(table_dir / "length_preserving_trace_conflicts.csv")
    successor_rank = pd.read_csv(table_dir / "successor_head_rankings.csv").sort_values("rank")
    feature_concentration = pd.read_csv(table_dir / "successor_mlp_feature_concentration.csv")

    document = re.sub(r"<title>.*?</title>", "<title>v16.2 RoPE：完整 v10-style representation 与因果机制报告</title>", document, count=1, flags=re.S)
    document = re.sub(
        r"<h1>.*?</h1>",
        "<h1>两种表示，两条计数路径：v16.2 的完整 representation 与因果机制</h1>",
        document,
        count=1,
        flags=re.S,
    )
    extra_css = """
    .evidence-note{border-left:4px solid #6750a4;background:#f6f3fb;padding:14px 16px;margin:18px 0;border-radius:0 8px 8px 0}
    .formula-list code{white-space:normal}.compact td,.compact th{padding-top:7px;padding-bottom:7px}
    .causal-strong{color:#146c43;font-weight:700}.causal-limited{color:#9a6700;font-weight:700}
    .interactive-hidden-state{border:1px solid var(--line);border-radius:10px;padding:18px;margin:24px 0;background:#fff}
    .interactive-hidden-state h4{margin:0 0 14px}.hs3d-controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px 16px;align-items:end;background:#f5f8fc;padding:14px;border-radius:7px}
    .hs3d-controls label{display:flex;flex-direction:column;gap:5px;font-size:13px;font-weight:650;color:#334155;min-width:0}.hs3d-controls select{width:100%;min-width:0;padding:7px 8px;border:1px solid #b8c4d5;border-radius:5px;background:#fff;color:#172235;font:inherit}
    .hs3d-controls .hs3d-check{flex-direction:row;grid-column:1/-1;align-items:center;gap:8px;padding-bottom:8px;white-space:nowrap}.hs3d-stats{min-height:28px;margin:11px 2px 7px;color:#44526a;font-size:14px;line-height:1.5}
    #v162-hs3d-canvas{display:block;width:100%;height:620px;border:1px solid var(--line);background:#fbfdff;touch-action:none}
    .hs3d-view-controls{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.hs3d-view-controls button{padding:7px 10px;border:1px solid #aebbd0;border-radius:5px;background:#fff;color:#243047;font:inherit;cursor:pointer}.hs3d-view-controls button:focus-visible{outline:3px solid rgba(37,99,235,.28);outline-offset:2px}
    .interactive-hidden-state figcaption{color:#44526a;font-size:14.5px;margin-top:12px}.metric-definition{color:#526077;font-size:13.5px;line-height:1.55;background:#f5f8fc;border-left:3px solid #7d8da8;padding:9px 12px;margin:10px 0 0}@media(max-width:760px){#v162-hs3d-canvas{height:440px}.hs3d-controls{grid-template-columns:1fr}}
    """
    document = document.replace("</style>", extra_css + "\n</style>", 1)

    hypothesis = _figure(
        figure_dir / "hypothesis_causal_map.png",
        "假设图 H2。",
        "v16.2 的两条候选机制与干预位置",
        "上行是 nonthinking 的 prompt 广聚合→后层 count state→答案；下行是 thinking 的 k-to-k 检索→显式 progress→trace readout→答案。实线箭头均由 query-local ablation、head/output patching 或 residual transplant 直接测试；灰色跨路线表示两种位置间是否共享可迁移的 count 方向。右侧列出必要性、充分性和几何因果三类测试。",
        "Nonthinking direct aggregation and thinking serial retrieval hypotheses with causal intervention points",
    )
    crosswalk_html = _table(
        ["v10 部分", "原分析", "v16.2 实现", "适配状态"],
        [
            [html.escape(str(row.v10_section)), html.escape(str(row.v10_analysis)), html.escape(str(row.v16_2_implementation)), html.escape(str(row.adaptation_status))]
            for row in crosswalk.itertuples()
        ],
    )
    intro = f"""
      <h3>1.1 v10→v16.2 的完整迁移范围</h3>
      <p>本版不是把若干 PCA 图附在旧报告末尾，而是把 v10 第 4–11 节的证据结构完整迁移到 v16.2：学习时序、attention role、2D/3D representation、global 与 position-local ablation、clean-to-corrupt head patching、successor/stop、MLP feature、final bridge、geometry steering、hidden-state patching，以及 head↔state 双向关系。由于 v16.2 只有 count 1–10，描述性分箱改为 1–3、4–7、8–10；所有主效果仍按 exact count 平衡汇总。</p>
      {hypothesis}
      {crosswalk_html}
      <div class="evidence-note"><strong>核心更新。</strong> 新的同长度干预否定了“最后一个 marker/index 本身携带最终标量 count”的简单解释：替换最后 index、marker、整对 token，甚至把最后一对换成两个 neutral token，答案仍 100% 跟随原 count；只有真正删去最后一对、从而把 trace span 与 <code>&lt;Ans&gt;</code> 相对位置缩短两格时，答案才 100% 转向 n−1。v16.2 的 thinking readout 因而主要读取 <strong>trace span/boundary geometry</strong>，而不是最后 token identity。</div>
    """
    document = _insert_before_section_end(document, "questions", intro)

    nt_l4 = geometry[(geometry["mode"] == "nonthinking") & (geometry["site"] == "final_answer") & (geometry["layer"] == 4)].iloc[0]
    th_l2 = geometry[(geometry["mode"] == "thinking") & (geometry["site"] == "final_answer") & (geometry["layer"] == 2)].iloc[0]
    trace_i = geometry[(geometry["mode"] == "thinking") & (geometry["site"] == "trace_index") & (geometry["layer"] == 3)].iloc[0]
    trace_m = geometry[(geometry["mode"] == "thinking") & (geometry["site"] == "trace_marker") & (geometry["layer"] == 3)].iloc[0]
    representation_fragment = f"""
      <h3>6.4 完整 2D/3D centroid manifold 与样本云</h3>
      <div class="formula-list">
        <p><strong>Mean-first PCA：</strong>先在 256 维 residual 中对同一标签求均值 <code>μ_c = mean_i h_i | y_i=c</code>，再对十个 <code>μ_c</code> 去中心化并做 SVD。图中的深色编号点是 centroid；淡色点是 held-out 单样本投影，且使用同一 centroid PCA 基底。这样 PC 不会被类内 noise 主导。</p>
        <p><strong>有效维度：</strong><code>d_eff = 1 / Σ_j r_j²</code>，其中 <code>r_j</code> 是全部 centroid-PC 的解释方差比例。<strong>相邻更新余弦：</strong><code>cos Δ_c,Δ_{{c+1}}</code>，<code>Δ_c=μ_{{c+1}}−μ_c</code>。<strong>路径直度：</strong><code>||μ_10−μ_1|| / Σ_c ||μ_{{c+1}}−μ_c||</code>；1 表示直线，越小表示越弯折。</p>
      </div>
      {_figure(figure_dir / 'representation_manifolds_2d.png', '图 17。', '2D mean-first manifold 与类内样本云', '每个 panel 横纵轴分别是该 site×Layer 独立拟合的 centroid PC1/PC2；括号给出对 centroid 总方差的解释比例。颜色和数字编码 count 或 progress k，灰线连接相邻标签。淡点只显示类内散布，不能跨 panel 比较屏幕方向。', 'Two-dimensional mean-first residual manifolds with held-out sample clouds')}
      {_figure(figure_dir / 'representation_manifolds_3d.png', '图 18。', '3D count/progress representation', '前三个 panel 对比 nonthinking/ thinking 最终答案 state，后两个是 trace index/marker；右下在同一 PCA 基底中同时放入 20 个 index/marker centroids，圆点与方块表示 token role。三个坐标轴分别是 PC1–PC3，百分比是 centroid variance explained。3D 图用于观察弯曲、分叉与 role separation，不把投影距离直接解释为因果强度。', 'Three-dimensional count and trace-progress residual representations')}
      {interactive_3d}
      <p>数值上，nonthinking L4 的 PC1–PC3 覆盖 {_pct(nt_l4.pc1_to_pc3_variance)} centroid 方差、有效维度 {_fmt(nt_l4.effective_dimension,2)}、相邻方向余弦 {_fmt(nt_l4.mean_adjacent_displacement_cosine)}；thinking L2 answer 的对应值为 {_pct(th_l2.pc1_to_pc3_variance)}、{_fmt(th_l2.effective_dimension,2)}、{_fmt(th_l2.mean_adjacent_displacement_cosine)}。Thinking 虽可被 ridge 几乎完美解码，却不是单轴直线：L2 路径直度仅 {_fmt(th_l2.path_straightness_chord_over_arc)}。Trace L3 index 的 PC1–PC3 覆盖 {_pct(trace_i.pc1_to_pc3_variance)}、marker 为 {_pct(trace_m.pc1_to_pc3_variance)}；二者在 joint 3D 中形成分离但相互对应的两条轨迹。</p>
      {_figure(figure_dir / 'representation_geometry_summary.png', '图 19。', '各层几何摘要', '四个 panel 横轴都是 Layer output 1–4；纵轴依次为 PC1–PC3 累计解释率、有效维度、相邻 step 余弦与 chord/arc 路径直度。Answer 与 trace 使用各自标签；曲线的层间变化表示 representation 重组，而不是简单“信息多少”。', 'Layer-wise residual geometry summary')}
      {_figure(figure_dir / 'representation_learning_dynamics.png', '图 20。', 'Representation 如何在训练中涌现与重组', '横轴是 0–10,000 training step，虚线是 1,500 步 loss-scope 切换；四个 panel 的纵轴分别是 held-out ridge R²、centroid PC1 方差、有效维度和相邻方向一致性。曲线显示 thinking 的可解码 count/progress 更早稳定，而 nonthinking 后期从近单轴过渡为更高维、但更可执行的 L4 state。', 'Representation learning dynamics across 21 checkpoints')}
    """
    document = document.replace("<h3>6.4 Trace-to-answer", representation_fragment + "\n      <h3>6.5 Trace-to-answer", 1)
    document = document.replace("<h3>6.5 表示何时稳定", "<h3>6.6 表示何时稳定", 1)

    top_successor = successor_rank.iloc[0]
    l4_conc = feature_concentration[(feature_concentration["layer"] == 4) & (feature_concentration["support"] == 16)].iloc[0]
    fig22 = _figure(
        figure_dir / "retrieval_head_patching.png",
        "图 22。",
        "k-to-k head output 是否足以恢复字符身份",
        f"横轴是累计 patch 的 clean head slices；左轴是 normalized marker-margin recovery，右轴是 margin&gt;0 的样本比例。彩线按独立 selection split 的 targeted score 排序，灰色为随机路径。Top-4 达到 {_fmt(values['retrieval_top4'])} recovery，随机 top-4 仅 {_fmt(values['retrieval_top4_random'])}。",
        "Count-preserving character corruption and k-to-k head-output patching",
    )
    fig23 = _figure(
        figure_dir / "successor_head_patching.png",
        "图 23。",
        "Successor/stop 决策的双向 head sufficiency",
        f"左图把 clean continue activation 写入本应 close 的 receiver；右图反向把 close activation 写入本应 continue 的 receiver。横轴为 patch slices 数，纵轴为对应二元 margin 的 normalized recovery；虚线 1 是完全恢复。Top-1 分别恢复 {_fmt(values['continue_top1'])}/{_fmt(values['close_top1'])}，top-4 为 {_fmt(values['continue_top4'])}/{_fmt(values['close_top4'])}。",
        "Bidirectional successor and stop head-output patching",
    )
    fig28 = _figure(
        figure_dir / "final_bridge_component_recovery.png",
        "图 28。",
        "Clean component 对 shortened-trace receiver 的恢复",
        f"横轴是 patch Layer，纵轴是 clean n-vs-(n−1) margin 的 normalized recovery；三组柱分别替换该层完整 attention output、MLP output 或 post-layer residual。误差条是跨样本 SEM。L2 attention output recovery={_fmt(values['bridge_l2_attn'])}，L4 MLP output={_fmt(values['bridge_l4_mlp'])}。",
        "Final-answer component patching from clean to shortened trace",
    )
    fig29 = _figure(
        figure_dir / "residual_count_transport.png",
        "图 29。",
        "Residual count state 的因果可执行性",
        f"横轴是 residual 在哪一层后被 patch，纵轴是 expected-count transport slope；蓝线是 natural donor state，橙/绿是 α=.5/1 的 train-centroid delta，虚线 1 为理想一比一搬运。Nonthinking 的 α=1 到 L4 才达 {_fmt(values['nt_residual_l4'])}；thinking 在 L2 已达 {_fmt(values['thinking_residual_l2'])}。",
        "Natural residual transplant and train-centroid count steering",
    )
    fig30 = _figure(
        figure_dir / "trace_early_stop_patching.png",
        "图 30。",
        "Final-marker state 是否足以提前关闭 trace",
        f"横轴是 donor final-marker residual 在 Layer 1–4 哪一层后写入；左轴是 close margin shift，右轴是 patched receiver 选择 close 的比例。L4 平均把 close margin 推高 {_fmt(values['early_l4_shift'],2)}，并使 {_pct(values['early_l4_flip'])} 样本翻转为 close。",
        "Position-matched final-marker residual patch inducing early stop",
    )
    causal_sections = f"""
    <section id="causal-heads">
      <p class="section-kicker">v10 §7 · Necessity</p>
      <h2>7. Attention heads 的必要性：global 与 query-local ablation</h2>
      <p>每个 role 的 16 个 heads 先在独立的 <code>head_selection</code> prompts 上排序，再在 heldout-reporting prompts 上评估。Global mask 把某 head 的所有 query 权重清零；query-local mask 只在机制定义的位置删除：nonthinking/readout 只删 <code>&lt;Ans&gt;</code>，targeted 只删全部 trace index <code>&lt;k&gt;</code>。因此 local 结果能排除“该 head 在别处被破坏”的解释。随机对照是 {manifest['options']['random_paths']} 条确定性随机顺序；灰带是这些路径的 min–max，不是置信区间。</p>
      <div class="equation">Accuracy drop = baseline accuracy − intervened accuracy；margin drop 同理。Head mask 在 softmax 后把选中 head 的 attention weights 乘 0，因此同时移除该 head 的 value contribution。</div>
      {_figure(figure_dir / 'head_ablation_global_local.png', '图 21。', '三条机制的累计删头曲线', '三行依次是 nonthinking broad→final count、thinking targeted→trace marker、thinking trace-readout→final count；左列 global，右列 query-local。横轴是累计删去的 head 数，纵轴是干预后的绝对准确率。彩线为机制排序，灰线/灰带为随机均值/min–max。', 'Global and query-local cumulative attention-head ablation')}
      <p><strong>Nonthinking：</strong>只在 <code>&lt;Ans&gt;</code> 删除第一 broad head，准确率从 82.5% 降至 {_pct(values['nt_local_top1_accuracy'])}，而随机单头均值仍为 {_pct(values['nt_local_top1_random'])}；说明最终广聚合不是可有可无的伴随信号。<strong>Thinking targeted：</strong>局部删 top-4 后 marker accuracy 为 {_pct(values['target_local_top4_accuracy'])}，随机 top-4 为 {_pct(values['target_local_top4_random'])}。<strong>Thinking readout：</strong>单头删除只显著压低 margin、未翻转 argmax；联合删 top-2 后 final accuracy 从 100% 降至 {_pct(values['readout_local_top2_accuracy'])}，随机 top-2 仍为 {_pct(values['readout_local_top2_random'])}，显示两个 readout heads 的冗余/协同。</p>
      <div class="callout caution"><strong>边界。</strong>删头证明必要性，不证明该 head 单独携带完整算法；全部 16 heads 删除后的 chance-level 结果也只是 sanity check。Nonthinking 的 global 随机删头本身破坏较大，所以机制特异性主要由 query-local 对照和下一节 donor transport 支撑。</div>
    </section>

    <section id="causal-retrieval-conversion">
      <p class="section-kicker">v10 §8.2–8.5 · Sufficiency and conversion</p>
      <h2>8. Retrieval、successor/stop 与 MLP conversion</h2>
      <h3>8.1 保持 count 不变的 k-to-k clean→corrupt patch</h3>
      <p>对第 k 个 prompt occurrence，把原目标字符 A 换成同一目标集合中的另一字符 B，并同步把 gold trace marker 改成 B。目标位置、总 count、prefix set、序列长度与 query 位置全部不变；只有“第 k 个字符身份”改变。随后只把 clean head 的 pre-output slice 写回 corrupt 的同一 <code>&lt;k&gt;</code> query。</p>
      <div class="equation">Normalized recovery = (patched margin − corrupt margin) / (clean margin − corrupt margin)，margin = z(A) − z(B)。0=没有恢复；1=恢复 clean margin；该量可超出 [0,1]，报告均值时不裁剪。</div>
      {fig22}
      <h3>8.2 同一 marker query 的 continue↔close 双向 patch</h3>
      <p>对一个 count=n 的 long prompt，选内部 k；把 k 之后所有目标 occurrence 改成非目标字符，得到在 k 处应关闭的 short receiver。两条序列在 M<sub>k</sub> query 之前长度完全相同，且 query 绝对位置相同。16 heads 用 10 个 selection examples 的双向 recovery 排序，最佳为 L{int(top_successor['layer'])}H{int(top_successor['head'])}（selection bidirectional score={_fmt(top_successor['bidirectional_score'])}），再在 24 个不重叠 examples 上累计 patch。</p>
      {fig23}
      <h3>8.3 Residual logit lens 与 additive component</h3>
      {_figure(figure_dir / 'successor_logit_lens_components.png', '图 24。', 'Continue/close evidence 在哪里写入 residual', '左图横轴按每层 pre→+attention→+MLP 展开，纵轴是 clean−short 的 continue logit-lens margin；右图横轴是 Layer，纵轴是 attention/MLP additive component 对目标 unembedding 方向的 clean−short 投影。Component 诊断忽略最终 LayerNorm，只用于定位证据转换，不单独构成因果证明。', 'Successor residual logit lens and additive component evidence')}
      <h3>8.4 Layer-3/4 MLP feature：分布式而非单 neuron</h3>
      <p>在 selection split 上定义每个 post-GELU feature 的直接证据 <code>e_j=(f_clean,j−f_short,j)·[W₂ᵀ(u_next−u_close)]_j</code>，按平均正证据排序；再在 reporting split 只替换这些 feature。L4 前 16 个 feature 只覆盖 {_pct(l4_conc.positive_evidence_fraction)} 正证据；需要 256 个 feature 才能在双向干预中分别恢复 {_fmt(values['feature_l4_256_continue'])}/{_fmt(values['feature_l4_256_close'])}。全 1,024 feature 不一定优于前 256，因为负证据也一并被写入。</p>
      {_figure(figure_dir / 'successor_mlp_features.png', '图 25。', 'MLP evidence concentration 与 feature patching', '左上横轴为 ranked feature support（log₂），纵轴为累计正/绝对证据比例；其余 panel 横轴为 patch feature 数（为显示 support=0，绘图时横坐标加 1），纵轴为 normalized recovery 或决定翻转率。实线是 ranked，淡虚线是 matched random。', 'MLP feature evidence concentration and causal feature patching')}
    </section>

    <section id="causal-final-readout">
      <p class="section-kicker">v10 §8.6–8.10 · Count transport and source conflict</p>
      <h2>9. 最终 count 从哪里来：head transport、同长度冲突与 final bridge</h2>
      <h3>9.1 Final-query head slices 能否搬运 donor count</h3>
      <p>对 receiver count=n 与 donor count=m，把 donor 在各层 <code>&lt;Ans&gt;</code> query 的 head slice 写入 receiver；用 restricted count softmax 的期望值 <code>E[C]</code> 计算搬运。</p>
      <div class="equation">Transport slope = Σ(m−n)·[E(C_patched)−E(C_base)] / Σ(m−n)²。0=不随 donor 变；1=一比一搬运 donor offset。回归过原点，因为 no-op patch 的理论位移为 0。</div>
      {_figure(figure_dir / 'final_query_head_transport.png', '图 26。', 'Final-query head output 的 donor-count transport', '横轴是累计 patch 的 donor head slices；纵轴是 expected-count transport slope。左为 nonthinking broad ranking，右为 thinking trace-readout ranking；彩线是机制排序，灰色为随机路径。', 'Donor count transport through final-query attention-head slices')}
      <p>Nonthinking top-4 broad heads 的 slope={_fmt(values['nt_head_top4'])}，随机 top-4={_fmt(values['nt_head_top4_random'])}；Thinking 单个 readout head 仅 {_fmt(values['thinking_head_top1'])}，但 top-2 立即达到 {_fmt(values['thinking_head_top2'])}，随机 top-2={_fmt(values['thinking_head_top2_random'])}。这与 top-2 local ablation 的 90 pp drop 互相闭合：两个 readout heads 合起来既必要，又足以搬运完整 donor count。</p>
      <h3>9.2 同长度 conflict：token identity 还是 trace span？</h3>
      {_figure(figure_dir / 'length_preserving_trace_conflicts.png', '图 27。', 'Trace 信息源冲突与位置控制', '左图纵轴是 argmax 跟随原 n 或 n−1 的比例；右图纵轴是 z(n)−z(n−1) 相对 clean 的变化。前五项保持总长度和 <Ans> 位置，最后一项真实删除 final pair、使 <Ans> 左移两格。所有同长度 token 内容干预都保留原 n；只有位置移动的删除使 100% 样本转向 n−1。', 'Length-preserving trace conflicts compared with position-shifting deletion')}
      <p>这组结果改变了对旧 trace-deletion 图的解释。Prompt 中删一个真实 occurrence、最后 index 改成 n−1、复制上一对、marker 换成非目标字符、甚至把最后一对变成两个 <code>&lt;Sep&gt;</code>，都不改变答案；真实缩短 trace 才使 margin 平均下降 22.94、期望 count 精确下降 1。因而最直接的机制是：<strong>模型在 <code>&lt;Ans&gt;</code> 读取由 trace span/边界位置编码的 count</strong>。这仍是因果发现，但因果变量是结构长度/相对位置，不是最后 marker 身份。</p>
      <h3>9.3 从缩短 trace 恢复 clean answer：final bridge 在哪一层</h3>
      {fig28}
      <p>L2 attention output 几乎一次性恢复 clean margin，直接对应早期形成的 L2 trace-readout scaffold；后续 L2 residual、L3 residual 维持该 count state，L4 MLP 再把它转成更可执行的 count logits。该实验的 receiver 仍有位置移动，所以回答的是“哪一层能补回被缩短 trace 破坏的证据”，不能单独区分 trace span 与 RoPE 相对位置。</p>
    </section>

    <section id="causal-state">
      <p class="section-kicker">v10 §9–11 · Steering, patching, bidirectionality</p>
      <h2>10. Hidden-state causality：centroid steering、early stop 与 head↔state</h2>
      <h3>10.1 独立 train centroid 的 count steering</h3>
      <p>每个 mode×Layer 的 count centroid 只用 train-region 每个 count 10 个 examples 拟合；held-out receiver 从未参与方向估计。对 receiver n 与相邻 donor m，在 Layer ℓ 后写入 <code>h′=h+α(μ_m−μ_n)</code>，α∈{{0.5,1}}；另以完整 natural donor residual transplant 为上界。</p>
      {fig29}
      <p>Representation 的差异不是“thinking probe 更高”这么简单：thinking 的 L2 state 已经<strong>因果可执行</strong>，而 nonthinking 的早中层 centroid direction 即使可读，也不能稳定控制答案，直到 L4 才接近一比一。这与 learning dynamics 中 thinking 早形成 readout、nonthinking 晚形成后层聚合完全一致。</p>
      <h3>10.2 Position-matched early stop</h3>
      <p>选 donor total=k 与 receiver total=k+2；两者 M<sub>k</sub> query 的绝对位置完全相同。把 donor 的“这是 final marker”residual 写入 receiver 的同一个内部 M<sub>k</sub>，比较 <code>z(&lt;/Think&gt;)−z(&lt;k+1&gt;)</code>。</p>
      {fig30}
      <h3>10.3 Head→state 与 state→head 的双向关系</h3>
      {_figure(figure_dir / 'head_state_bidirectional.png', '图 31。', 'Attention head 与 hidden state 的双向因果联系', '左图横轴是 query-local 删除的 role heads 数，纵轴同时画后续 L4 nearest-centroid state accuracy 与输出 accuracy；右图把 total count 固定为 10，将 progress j=8 的 pre-L3 residual 写入 k=3 query，纵轴是 top L3 heads 对 donor occurrence 相对 receiver occurrence 的 attention-mass shift。', 'Bidirectional causal relationship between attention heads and residual states')}
      <p>Nonthinking 删除首个 broad head 后，L4 state accuracy 从 75% 降至 {_pct(values['nt_state_top1'])}，与输出下降同步；thinking 联合删两个 readout heads 后 state accuracy 降至 {_pct(values['thinking_state_top2'])}。反向地，progress state transplant 使 L3H3 对 donor-k 相对 receiver-k 的 routing 增加 {_fmt(values['state_to_head_l3h3'])} attention mass。它支持“head 写 state、state 再控制后续 routing”的循环依赖，而非 attention 与 representation 两条互不相干的相关性曲线。</p>
      <div class="callout warning"><strong>不是统计 mediation 识别。</strong>双向干预建立了可操纵的方向关系，但没有满足自然直接/间接效应所需的无混杂假设；因此报告使用“causal link / sufficiency”而不声称估计了严格 mediation proportion。</div>
    </section>
    """
    start, end, _ = _section(document, "causal")
    document = document[:start] + causal_sections + document[end:]

    document = document.replace("<h2>8. 训练数据结构与 noise 的影响</h2>", "<h2>11. 训练数据结构与 noise 的影响</h2>", 1)
    document = document.replace("<h3>8.1 接受过滤造成的 count 不均衡</h3>", "<h3>11.1 接受过滤造成的 count 不均衡</h3>", 1)
    document = document.replace("<h3>8.2 Noise/structure 特征如何定义</h3>", "<h3>11.2 Noise/structure 特征如何定义</h3>", 1)
    document = document.replace("<h2>9. 运行成本、产物与复现审计</h2>", "<h2>12. 运行成本、产物与复现审计</h2>", 1)
    document = document.replace("<h3>9.1 Runtime</h3>", "<h3>12.1 Runtime</h3>", 1)
    document = document.replace("<h3>9.2 Artifact 完整性</h3>", "<h3>12.2 Artifact 完整性</h3>", 1)
    document = document.replace("<h3>9.3 关键产物</h3>", "<h3>12.3 关键产物</h3>", 1)

    evidence_rows = [
        ["Nonthinking broad aggregation", "L4 broad score 晚期涌现", "local top-1: 82.5%→42.5%", "top-4 donor slope 0.680", '<span class="causal-strong">强支持，后层分布式</span>'],
        ["Thinking k-to-k retrieval", "correct-k/diagonal dominance", "local top-4 marker acc. 42.3%", "top-4 recovery 0.909", '<span class="causal-strong">必要且充分</span>'],
        ["Successor/stop", "continue/close state 分离", "双向局部 patch", "top-4 recovery 0.930/0.911", '<span class="causal-strong">双向充分</span>'],
        ["Thinking trace readout", "L2 trace mass 很早出现", "local top-2 final acc. 10%", "top-2 donor slope 1.000", '<span class="causal-strong">双头协同</span>'],
        ["Final trace token identity", "最后 pair 与答案相关", "同长度 5 类替换均无效", "仅 span 缩短改变答案", '<span class="causal-limited">简单 token-state 假设被否定</span>'],
        ["Executable count state", "ridge/centroid 可解码", "state/head 干预同步", "Thinking L2、NT L4 slope≈1", '<span class="causal-strong">层级差异明确</span>'],
    ]
    full_limits = f"""
    <section id="limits">
      <p class="section-kicker">Synthesis and evidence boundary</p>
      <h2>13. 综合机制结论、证据强度与仍然开放的问题</h2>
      <h3>13.1 证据矩阵</h3>
      {_table(['机制命题','描述性证据','必要性','充分性/transport','判断'], evidence_rows)}
      <h3>13.2 最简机制解释</h3>
      <ol>
        <li><strong>Thinking 先搭 readout，后学检索。</strong>L2 answer-to-trace routing 在早期已稳定；中后期 L3/L4 的 k-to-k heads 才把真实字符按顺序写入 trace。显式 trace 同时提供 progress token、marker 内容和一个可被 RoPE/边界读取的 span。</li>
        <li><strong>Progress control 与 scalar count readout 不是同一件事。</strong>内部 marker residual 在 L4 足以触发 close；最终答案则主要由两个 L2 readout heads 联合读取 trace span，之后经 L4 MLP 形成可执行 count logits。</li>
        <li><strong>Nonthinking 走更晚的直接路径。</strong>它没有显式 k interface；L4 broad heads 在 <code>&lt;Ans&gt;</code> 聚合多个 occurrence，并把 count 写入 L4 residual。该 state 到 L4 才能用 centroid delta 或 natural donor 近一比一控制答案。</li>
        <li><strong>两种 representation 都可解码，但几何不同。</strong>Thinking 更早、维度更高、轨迹更弯；nonthinking 晚期 L4 才出现大尺度可执行 count manifold。高 ridge R² 不等于一条直的“number line”。</li>
      </ol>
      <h3>13.3 仍然不能声称什么</h3>
      <ul>
        <li>所有结果来自单 seed、单 corpus、单 4×4 模型；role heads 可能在不同 seed 置换或分裂。</li>
        <li>随机对照只有 {manifest['options']['random_paths']} 条固定路径，灰带不是置信区间；它用于机制特异性 sanity check，不用于总体显著性推断。</li>
        <li>Shortened-trace 与 component recovery 改变了 trace span/相对位置；它证明模型使用该结构，但不能在这一个模型里把“抽象长度”与 RoPE relative-position signal 完全分离。</li>
        <li>Successor short prompt 由字符替换构造，虽然 count 与 query 位置严格配对，但不是天然 Shakespeare window；结论是局部电路 sufficiency，不是自然分布效应量。</li>
        <li>Head mask 同时删除 attention weight 与 value contribution；若要进一步分解，应比较 weight-only rerouting、V-slice patch 与 W<sub>O</sub> contribution patch。</li>
        <li>3D PCA 是 mean-first 可视化，每个 panel 轴独立；不能把图上方向跨 site/Layer 直接解释成同一个神经方向。</li>
      </ul>
      <h3>13.4 信息增益最高的下一步</h3>
      <ol>
        <li>至少 5 个 seeds 重复完整 role ranking、top-2 readout synergy 和 L2/L4 transport onset；报告 head permutation-invariant 的子空间重合。</li>
        <li>构造固定 <code>&lt;Ans&gt;</code> 相对位置但系统改变有效 trace pair 数的专门训练分布，例如显式 padding mask/token，从训练阶段解耦 count 与 span。</li>
        <li>对 L2 readout heads 分别 patch Q/K、V 与 pre-W<sub>O</sub> slice，区分“找到 boundary”与“搬运 count state”。</li>
        <li>在自然 held-out test AR rollouts 上重复 intervention，避免 teacher forcing 把错误 trace 强行修正。</li>
        <li>随机化 target span、noise run、三字符 balance 与 set frequency，做真正的结构因果 sweep，而不是只看观察相关。</li>
      </ol>
      <div class="callout success"><strong>最终结论。</strong>v16.2 的 CoT 优势不是“同一个内部计数器更强”，而是把算法拆成了可监督的接口：ordered character retrieval、progress/stop control、以及以 trace span 为核心的 final readout。Nonthinking 则在最后一层把多目标 broad aggregation 压缩成 count。新增的 ablation、clean-to-corrupt patch、donor transport、centroid steering 和 position-matched early-stop 干预，使这一区分从 representation 相关性升级为一组互相闭合的因果证据；同时，同长度 conflict 也修正了原先可能把最后 marker 误认为 scalar-count carrier 的解释。</div>
    </section>
    """
    start, end, _ = _section(document, "limits")
    document = document[:start] + full_limits + document[end:]

    artifacts = """
      <h3>12.4 本次 v10-port 新增产物</h3>
      <ul class="artifact-list">
        <li><a href="analysis/v10_port/manifest.json">analysis/v10_port/manifest.json</a>：采样、身份与全部表格 SHA256。</li>
        <li><a href="analysis/v10_port/tables/interactive_hidden_state_pca.csv">interactive_hidden_state_pca.csv</a>：21 个 checkpoint × 4 个语义位置 × 5 个深度的 mean-first PCA 坐标与几何统计。</li>
        <li><a href="analysis/v10_port/tables/analysis_crosswalk.csv">analysis_crosswalk.csv</a>：v10 第 4–11 节逐项适配表。</li>
        <li><a href="analysis/v10_port/tables/global_head_ablation.csv">global_head_ablation.csv</a>；<a href="analysis/v10_port/tables/position_local_head_ablation.csv">position_local_head_ablation.csv</a>。</li>
        <li><a href="analysis/v10_port/tables/retrieval_head_patching.csv">retrieval_head_patching.csv</a>；<a href="analysis/v10_port/tables/successor_head_patching.csv">successor_head_patching.csv</a>；<a href="analysis/v10_port/tables/successor_mlp_feature_patching.csv">successor_mlp_feature_patching.csv</a>。</li>
        <li><a href="analysis/v10_port/tables/final_query_head_transport.csv">final_query_head_transport.csv</a>；<a href="analysis/v10_port/tables/residual_count_transport.csv">residual_count_transport.csv</a>；<a href="analysis/v10_port/tables/trace_early_stop_patching.csv">trace_early_stop_patching.csv</a>。</li>
        <li><a href="analysis/v10_port/tables/length_preserving_trace_conflicts.csv">length_preserving_trace_conflicts.csv</a>；<a href="analysis/v10_port/tables/final_bridge_component_patching.csv">final_bridge_component_patching.csv</a>。</li>
        <li>复现入口：<a href="../../scripts/run_v16_2_v10_port_analysis.py">run_v16_2_v10_port_analysis.py</a>、<a href="../../scripts/plot_v16_2_v10_port_analysis.py">plot_v16_2_v10_port_analysis.py</a>、<a href="../../scripts/build_v16_2_interactive_geometry.py">build_v16_2_interactive_geometry.py</a>、<a href="../../scripts/build_v16_2_v10_full_report.py">build_v16_2_v10_full_report.py</a>、<a href="../../scripts/audit_v16_2_report_figures.py">audit_v16_2_report_figures.py</a>。</li>
      </ul>
    """
    document = _insert_before_section_end(document, "runtime-repro", artifacts)
    document = re.sub(
        r"<footer>.*?</footer>",
        f"<footer>完整 v10-style 迁移报告。数据身份：{html.escape(str(manifest['run_id']))}；RoPE-only；checkpoint step {manifest['checkpoint_step']}。所有新增因果表位于 <code>analysis/v10_port/tables/</code>，所有新增图位于 <code>analysis/v10_port/figures/</code>。</footer>",
        document,
        count=1,
        flags=re.S,
    )

    document = polish_report_html(document)
    document = _refresh_core_plot_images(document, run_dir)
    _atomic_text(output, document)
    cleanup_legacy_reports(run_dir, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "colab_results" / "v16_2_main_rope_seed1234",
    )
    args = parser.parse_args()
    print(build(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
