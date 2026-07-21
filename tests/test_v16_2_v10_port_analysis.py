from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from synthetic_counting_v16_2.config import preset_config
from synthetic_counting_v16_2.data import (
    V16_2Vocab,
    balanced_v16_2_examples,
    build_corpus_split,
    load_corpus_text,
    render_v16_2,
)
from synthetic_counting_v16_2.interactive_geometry import mean_first_pca
from synthetic_counting_v16_2.needle_pool import build_needle_pool
from synthetic_counting_v16_2.report_readability import FIGURE_SPECS, polish_report_html
from synthetic_counting_v16_2.training import paired_v16_2_model
from synthetic_counting_v16_2.v10_port_analysis import (
    _attention_vectors,
    _capture_attention_inputs,
    _capture_hidden,
    _forward,
    _local_attention_edit,
    _normalized_recovery,
    _residual_patch,
    _retrieval_corruption,
    _successor_pair,
    analysis_crosswalk,
)


@pytest.fixture(scope="module")
def prepared():
    cfg = preset_config("debug", device="cpu")
    text = load_corpus_text()
    split = build_corpus_split(cfg, text)
    vocab = V16_2Vocab.build(cfg, text)
    pool = build_needle_pool(cfg, text, split, vocab.fingerprint)
    examples = balanced_v16_2_examples(
        cfg, vocab, text, split, pool, 1, 8162, region_name="validation"
    )
    model = paired_v16_2_model(cfg, vocab, "rope").eval()
    return cfg, vocab, examples, model


def test_normalized_recovery_has_unclipped_intervention_semantics():
    clean = np.asarray([4.0, 2.0, 1.0])
    corrupt = np.asarray([0.0, 0.0, 1.0])
    patched = np.asarray([2.0, 3.0, 9.0])
    recovery = _normalized_recovery(clean, corrupt, patched)
    assert recovery[0] == pytest.approx(0.5)
    assert recovery[1] == pytest.approx(1.5)
    assert np.isnan(recovery[2])


def test_character_retrieval_corruption_preserves_count_length_and_positions(prepared):
    _, vocab, examples, _ = prepared
    example = next(item for item in examples if int(item.count or 0) >= 3)
    k = 2
    clean, corrupt, clean_id, corrupt_id = _retrieval_corruption(example, vocab, k)
    assert clean_id != corrupt_id
    assert len(clean.input_ids) == len(corrupt.input_ids)
    assert clean.spans == corrupt.spans
    assert clean.count == corrupt.count == example.count
    differences = [
        index for index, (left, right) in enumerate(zip(clean.input_ids, corrupt.input_ids, strict=True))
        if left != right
    ]
    assert differences == [
        clean.prompt_needle_positions[k - 1],
        clean.spans.trace_marker_positions[k - 1],
    ]


def test_successor_pair_has_position_matched_marker_query(prepared):
    _, vocab, examples, _ = prepared
    example = next(item for item in examples if int(item.count or 0) >= 4)
    k = 2
    clean, short = _successor_pair(example, vocab, k)
    assert len(clean.input_ids) == len(short.input_ids)
    assert clean.spans.trace_marker_positions[k - 1] == short.spans.trace_marker_positions[k - 1]
    query = clean.spans.trace_marker_positions[k - 1]
    assert clean.input_ids[query] == short.input_ids[query]
    changed = [
        position
        for position in clean.prompt_needle_positions[k:]
        if clean.input_ids[position] != short.input_ids[position]
    ]
    assert len(changed) == int(example.count) - k


def test_same_attention_slice_patch_and_same_residual_patch_are_noops(prepared):
    cfg, vocab, examples, model = prepared
    example = next(item for item in examples if int(item.count or 0) >= 2)
    item = render_v16_2(example, vocab, "thinking")
    baseline, attention_inputs = _capture_attention_inputs(model, [item], vocab, cfg.device)
    answer_position = item.spans.ans_pos
    donor = _attention_vectors(attention_inputs, [answer_position])
    all_heads = [(layer, head) for layer in range(1, 5) for head in range(4)]
    with _local_attention_edit(model, all_heads, [[answer_position]], donor):
        patched = _forward(model, [item], vocab, cfg.device)
    assert torch.allclose(baseline.logits, patched.logits, atol=1e-6, rtol=1e-6)

    baseline_hidden_output, hidden = _capture_hidden(
        type("Context", (), {"models": {"thinking": model}, "vocab": vocab, "device": cfg.device})(),
        "thinking",
        [item],
    )
    same_vector = hidden[2][:, answer_position].detach().clone()
    with _residual_patch(model, 2, [answer_position], same_vector):
        residual_patched = _forward(model, [item], vocab, cfg.device)
    assert torch.allclose(
        baseline_hidden_output.logits, residual_patched.logits, atol=1e-6, rtol=1e-6
    )


def test_v10_crosswalk_covers_all_mechanism_sections():
    frame = analysis_crosswalk()
    assert set(frame["v10_section"]) == {
        "v10 §4",
        "v10 §5",
        "v10 §6",
        "v10 §7",
        "v10 §8.2",
        "v10 §8.3-8.5",
        "v10 §8.6-8.10",
        "v10 §9",
        "v10 §10",
        "v10 §11",
    }
    assert frame["v16_2_implementation"].str.len().min() > 10


def test_interactive_mean_first_pca_uses_centroids_and_orients_pc1_by_label():
    rng = np.random.default_rng(162)
    labels = np.repeat(np.arange(1, 11), 5)
    signal = labels[:, None] * np.asarray([[2.0, -1.0, 0.5]])
    values = signal + rng.normal(scale=0.02, size=signal.shape)
    result = mean_first_pca(values, labels)
    coordinates = result["coordinates"]
    variance = result["variance"]
    assert isinstance(coordinates, np.ndarray)
    assert isinstance(variance, np.ndarray)
    assert coordinates.shape == (10, 6)
    assert np.corrcoef(np.arange(1, 11), coordinates[:, 0])[0, 1] > 0.999
    assert variance[0] > 0.999
    assert 1.0 <= float(result["effective_dimension"]) < 1.01


def test_interactive_mean_first_pca_reports_zero_for_degenerate_initialization():
    labels = np.repeat(np.arange(1, 11), 2)
    result = mean_first_pca(np.zeros((20, 8)), labels)
    assert np.count_nonzero(result["coordinates"]) == 0
    assert np.count_nonzero(result["variance"]) == 0
    assert result["effective_dimension"] == 0.0
    assert result["adjacent_cosine"] == 0.0
    assert result["straightness"] == 0.0


def test_interactive_effective_dimension_uses_all_centroid_components():
    labels = np.repeat(np.arange(1, 11), 2)
    values = np.repeat(np.eye(10), 2, axis=0)
    result = mean_first_pca(values, labels, components=6)
    assert np.isclose(result["effective_dimension"], 9.0)
    assert np.isclose(np.sum(result["variance"]), 6.0 / 9.0)


def test_report_readability_is_idempotent_and_defines_every_figure_first():
    figures = []
    for _, needle, _ in FIGURE_SPECS:
        if needle.startswith("<span"):
            figures.append(f"<figure><figcaption>{needle} original</figcaption></figure>")
        elif needle.startswith('id="'):
            figures.append(f"<figure {needle}><figcaption>interactive</figcaption></figure>")
        else:
            figures.append(f"<figure><h4>{needle}</h4><figcaption>original</figcaption></figure>")
    sections = [
        '<section id="questions"><h2>1. 研究问题与核心结论</h2>'
        + "".join(figures)
        + "</section>",
        '<section id="setup"><h2>2. setup</h2></section>',
        '<section id="definitions"><h2>3. definitions</h2></section>',
        '<section id="learning"><h2>4. 行为表现与 learning dynamics</h2></section>',
        '<section id="attention-representation"><h2>5. attention</h2></section>',
        '<section id="residual-representation"><h2>6. residual</h2></section>',
        '<section id="causal-heads"><h2>7. heads</h2></section>',
        '<section id="causal-retrieval-conversion"><h2>8. retrieval</h2></section>',
        '<section id="causal-final-readout"><h2>9. readout</h2></section>',
        '<section id="causal-state"><h2>10. Hidden-state causality：centroid steering、early stop 与 head↔state</h2>'
        '<h3>10.1 steering</h3><h3>10.2 early stop</h3>'
        '<h3>10.3 Head→state 与 state→head 的双向关系</h3><p>bidirectional</p></section>',
        '<section id="data-noise"><h2>11. data</h2><h3>11.1 bins</h3></section>',
        '<section id="runtime-repro"><h2>12. 运行成本、产物与复现审计</h2><h3>12.1 Runtime</h3></section>',
        '<section id="limits"><h2>13. limits</h2></section>',
    ]
    document = (
        '<html><head><style></style></head><body>'
        '<div class="callout success preserved">legacy links</div>'
        '<nav class="toc">legacy navigation</nav><main>'
        + "".join(sections)
        + "</main></body></html>"
    )
    polished = polish_report_html(document)
    assert polished.count('class="figure-reading-guide"') == len(FIGURE_SPECS)
    assert polished.index('id="reading-primer"') < polished.index('class="figure-reading-guide"')
    assert polished.index('data-figure-key="fig-01"') > polished.index('<section id="learning">')
    assert polished.index('<section id="limits">') < polished.index('<section id="runtime-repro">')
    assert '<section id="causal-bidirectional">' in polished
    repolished = polish_report_html(polished)
    for version in (polished, repolished):
        assert version.count('id="reading-primer"') == 1
        assert version.count('class="figure-reading-guide"') == len(FIGURE_SPECS)
        for key, needle, _ in FIGURE_SPECS:
            assert version.index(f'data-figure-key="{key}"') < version.index(needle)
