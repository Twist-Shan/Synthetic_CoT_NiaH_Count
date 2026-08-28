from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from synthetic_counting_v20.config import config_from_dict, preset_config as preset_v20
from synthetic_counting_v20.data import (
    V20Example,
    V20Vocab,
    character_token,
    component_target_positions,
    render_v20,
)
from synthetic_counting_v20.model import build_model
from synthetic_counting_v20.phase_transition import (
    _target_positions_and_ids,
    build_training_token_exposure,
)
from synthetic_counting_v20.training import _parse_generation
from synthetic_counting_v22.config import preset_config as preset_v22


ROOT = Path(__file__).resolve().parents[1]


def _example(count: int = 3) -> V20Example:
    markers = tuple(character_token("a") for _ in range(count))
    return V20Example(
        example_kind="counting_task",
        seq_tokens=list(markers),
        corpus_region="validation",
        corpus_start=0,
        corpus_end=count,
        prompt_sha256="v22-test",
        set_id="set",
        needle_characters=("a", "b", "c"),
        rendered_set_order=("a", "b", "c"),
        needle_positions=tuple(range(count)),
        needle_markers=markers,
        count=count,
        set_frequency_sum=0.1,
        set_frequency_bin=1,
        per_character_counts=(count, 0, 0),
    )


def test_v22_is_a_thinking_only_v20_matched_separator_config():
    v20 = preset_v20("main", device="cpu")
    v22 = preset_v22("main", device="cpu")
    assert (v22.version, v22.count_tokenization, v22.trace_format) == (
        "v22",
        "atomic",
        "separator",
    )
    assert v22.enabled_model_variants == ("rope/thinking",)
    assert v22.count_max_threshold == v20.count_max_threshold == 30
    assert v22.seq_len == v20.seq_len == 256
    assert v22.max_render_len == v20.max_render_len == 327
    assert config_from_dict(v22.to_dict()) == v22
    assert "(<Sep> marker)*n" in v22.to_dict()["sequence_templates"]["thinking"]


def test_separator_rendering_replaces_only_trace_indices():
    corpus = "abc xyz\n"
    example = _example(3)
    cfg20 = preset_v20("main", device="cpu")
    cfg22 = preset_v22("main", device="cpu")
    vocab20 = V20Vocab.build(cfg20, corpus)
    vocab22 = V20Vocab.build(cfg22, corpus)
    indexed = render_v20(example, vocab20, "thinking")
    separator = render_v20(example, vocab22, "thinking")
    assert indexed.spans is not None and separator.spans is not None
    assert vocab20.id_to_token == vocab22.id_to_token
    assert vocab20.fingerprint != vocab22.fingerprint
    assert len(indexed.tokens) == len(separator.tokens)
    assert indexed.spans.think_pos == separator.spans.think_pos
    assert indexed.spans.ans_pos == separator.spans.ans_pos
    assert [separator.tokens[pos] for pos in separator.spans.trace_query_positions] == [
        "<Sep>",
        "<Sep>",
        "<Sep>",
    ]
    trace = separator.tokens[
        separator.spans.think_pos + 1 : separator.spans.think_close_pos
    ]
    assert trace == [
        "<Sep>",
        character_token("a"),
        "<Sep>",
        character_token("a"),
        "<Sep>",
        character_token("a"),
    ]
    assert not any(token in vocab22.numbers for token in trace)
    assert separator.tokens[separator.spans.count_pos] == "<3>"
    components = component_target_positions(separator)
    assert "trace_delimiter" in components and "trace_index" not in components
    assert components["trace_delimiter"] == separator.spans.trace_query_positions


def test_separator_generation_parser_reports_order_stopping_and_format():
    cfg = preset_v22("main", device="cpu")
    vocab = V20Vocab.build(cfg, "abc xyz\n")
    example = _example(3)
    rendered = render_v20(example, vocab, "thinking")
    metrics = _parse_generation(rendered.tokens, vocab, example, "thinking")
    assert metrics["ar_accuracy"] == 1.0
    assert metrics["trace_exact"] == 1.0
    assert metrics["trace_ordered_marker_accuracy"] == 1.0
    assert metrics["trace_marker_count_accuracy"] == 1.0
    assert metrics["trace_delimiter_count_accuracy"] == 1.0
    assert metrics["trace_format_valid"] == 1.0
    assert metrics["trace_closed"] == 1.0

    malformed = list(rendered.tokens)
    assert rendered.spans is not None
    malformed[rendered.spans.trace_query_positions[-1]] = "<1>"
    bad = _parse_generation(malformed, vocab, example, "thinking")
    assert bad["trace_exact"] == 0.0
    assert bad["trace_format_valid"] == 0.0
    assert bad["trace_delimiter_count_accuracy"] == 0.0


def test_separator_continue_targets_and_exposure_are_not_called_indices(tmp_path):
    cfg = preset_v22("main", device="cpu")
    vocab = V20Vocab.build(cfg, "abc xyz\n")
    item = render_v20(_example(3), vocab, "thinking")
    _, targets, ks = _target_positions_and_ids(item, vocab, "marker_successor")
    assert ks == [1, 2, 3]
    assert [vocab.id_to_token[target] for target in targets] == [
        "<Sep>",
        "<Sep>",
        "</Think>",
    ]

    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    state = {"accepted_counts": {"1": 2, "2": 3, "3": 5}}
    pd.DataFrame(
        [
            {
                "position_encoding": "rope",
                "mode": "thinking",
                "step": 10,
                "cumulative_sampling_json": json.dumps(state),
            }
        ]
    ).to_csv(table_dir / "train_metrics.csv", index=False)
    exposure = build_training_token_exposure(cfg, tmp_path)
    k2 = exposure[exposure.k == 2].iloc[0]
    assert k2.trace_query_token_exposure == 8
    assert k2.trace_delimiter_token_exposure == 8
    assert k2.trace_index_token_exposure == 0


def test_v20_and_v22_shared_parameters_start_identically():
    corpus = "abc xyz\n"
    cfg20 = preset_v20("debug", device="cpu", enabled_model_variants=("rope/thinking",))
    cfg22 = preset_v22("debug", device="cpu")
    vocab20 = V20Vocab.build(cfg20, corpus)
    vocab22 = V20Vocab.build(cfg22, corpus)
    model20 = build_model(cfg20, vocab20, "rope", "cpu")
    model22 = build_model(cfg22, vocab22, "rope", "cpu")
    assert model20.parameter_count() == model22.parameter_count()
    for left, right in zip(model20.parameters(), model22.parameters(), strict=True):
        assert torch.equal(left, right)


def test_v22_colab_notebook_is_clean_and_auditable():
    path = ROOT / "notebooks" / "Trace_Count_v22_NoIndex_Colab.ipynb"
    assert path.exists()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(
        cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'VERSION = "v22"' in source
    assert 'TRACE_FORMAT = "separator"' in source
    assert '"rope/thinking"' in source
    assert '"rope/nonthinking"' not in source
    assert "COUNT_MAX_THRESHOLD = 30" in source
    assert "CHECKPOINT_EVERY_STEPS = 100" in source
    assert "RECOVERY_EVERY_STEPS = 500" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
