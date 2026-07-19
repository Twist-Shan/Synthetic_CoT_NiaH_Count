from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from synthetic_counting_v19.config import canonical_run_specs, preset_config
from synthetic_counting_v19.data import (
    ANSWER,
    COUNT,
    DIGIT_OFFSET,
    END,
    IGNORE_INDEX,
    INDEX,
    MARKER_OFFSET,
    NUM_END,
    START,
    ReferenceExample,
    count_probabilities,
    decode_number,
    digit_trace_layout,
    encode_number,
    render,
    sample_example,
)
from synthetic_counting_v19.model import ReferenceTransformer
from synthetic_counting_v19.training import parse_generated_suffix


ROOT = Path(__file__).resolve().parents[1]


def test_v19_suite_matches_v18_except_for_digit_number_grammar():
    specs = canonical_run_specs()
    assert len(specs) == 4
    assert len({spec.name for spec in specs}) == 4
    assert Counter(spec.distribution for spec in specs) == {"power": 2, "uniform": 2}
    assert Counter(spec.mode for spec in specs) == {"direct": 2, "cot": 2}
    assert {spec.context_length for spec in specs} == {1024}
    assert {spec.train_max_count for spec in specs} == {128}
    assert {spec.alpha for spec in specs if spec.distribution == "power"} == {1.5}

    cfg = preset_config("debug", device="cpu")
    assert (cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.n_inner) == (4, 4, 256, 1024)
    assert cfg.vocab_size == 283


def test_v19_decimal_codec_round_trips_boundaries_and_rejects_ambiguous_forms():
    for value in (1, 9, 10, 12, 99, 100, 128):
        encoded = encode_number(value)
        assert all(DIGIT_OFFSET <= token < DIGIT_OFFSET + 10 for token in encoded)
        assert decode_number(encoded) == value

    assert decode_number(()) is None
    assert decode_number((DIGIT_OFFSET, DIGIT_OFFSET + 1)) is None
    assert decode_number(encode_number(128), maximum=127) is None
    assert decode_number((MARKER_OFFSET,)) is None


def test_v19_direct_and_cot_render_share_decimal_digits_but_not_roles():
    marker_a, marker_b = MARKER_OFFSET + 2, MARKER_OFFSET + 7
    example = ReferenceExample((5, marker_a, 99, marker_b), 2, (1, 3), (marker_a, marker_b))

    direct = render(example, "direct")
    direct_completion = (COUNT, *encode_number(2), NUM_END)
    assert direct.tokens == (*example.context, ANSWER, *direct_completion)
    assert direct.labels == (IGNORE_INDEX,) * 5 + direct_completion

    cot = render(example, "cot")
    cot_completion = (
        INDEX,
        *encode_number(1),
        marker_a,
        INDEX,
        *encode_number(2),
        marker_b,
        END,
        COUNT,
        *encode_number(2),
        NUM_END,
    )
    assert cot.tokens == (*example.context, START, *cot_completion)
    assert cot.labels == (IGNORE_INDEX,) * 5 + cot_completion
    assert INDEX != COUNT
    assert encode_number(2)[0] in direct.tokens
    assert encode_number(2)[0] in cot.tokens


def test_v19_multidigit_layout_anchors_ktok_query_on_last_index_digit():
    context_length = 32
    layout = digit_trace_layout(context_length, 12)
    assert len(layout.index_role_positions) == 12
    assert len(layout.index_digit_positions[8]) == 1
    assert len(layout.index_digit_positions[9]) == 2
    assert len(layout.index_digit_positions[11]) == 2
    assert layout.index_query_positions[9] == layout.index_digit_positions[9][-1]
    assert layout.marker_positions[9] == layout.index_query_positions[9] + 1
    assert len(layout.count_digit_positions) == 2
    assert layout.num_end_position == layout.count_digit_positions[-1] + 1


def test_v19_free_running_parser_handles_multidigit_trace_and_final_count():
    markers = [MARKER_OFFSET + (index % 10) for index in range(1, 13)]
    cot_suffix: list[int] = []
    for index, marker in enumerate(markers, 1):
        cot_suffix.extend((INDEX, *encode_number(index), marker))
    cot_suffix.extend((END, COUNT, *encode_number(12), NUM_END))
    enumeration, count, parsed_markers = parse_generated_suffix(cot_suffix, "cot", 128)
    assert enumeration == count == 12
    assert parsed_markers == markers

    _, direct_count, direct_markers = parse_generated_suffix(
        [COUNT, *encode_number(128), NUM_END],
        "direct",
        128,
    )
    assert direct_count == 128
    assert direct_markers == []

    malformed = cot_suffix.copy()
    malformed[malformed.index(INDEX, 1) + 1] = DIGIT_OFFSET + 9
    assert parse_generated_suffix(malformed, "cot", 128)[0] is None


def test_v19_sampling_keeps_v18_noise_marker_and_count_distributions():
    cfg = preset_config("debug", device="cpu")
    spec = next(spec for spec in canonical_run_specs() if spec.mode == "cot" and spec.distribution == "power")
    example = sample_example(cfg, spec, random.Random(9), count=12)
    assert len(example.context) == 1024
    assert len(example.needle_positions) == len(example.needle_markers) == 12
    assert all(MARKER_OFFSET <= token < MARKER_OFFSET + 10 for token in example.needle_markers)
    probabilities = count_probabilities(spec)
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[0] > probabilities[1] > probabilities[-1]


def test_v19_model_forward_uses_reference_rope_scale_and_new_vocab():
    cfg = preset_config("debug", device="cpu")
    model = ReferenceTransformer(cfg).eval()
    ids = torch.tensor(
        [[5, MARKER_OFFSET, START, INDEX, *encode_number(1), MARKER_OFFSET, END]],
        dtype=torch.long,
    )
    with torch.no_grad():
        output = model(ids, output_attentions=True, output_hidden_states=True)
    assert output.logits.shape == (1, ids.shape[1], cfg.vocab_size)
    assert output.attentions is not None and len(output.attentions) == cfg.n_layer
    assert output.hidden_states is not None and len(output.hidden_states) == cfg.n_layer + 1
    assert model.parameter_count() > 3_000_000
    assert model.parameter_count() < 4_500_000


def test_v19_notebook_mounts_drive_first_compiles_and_documents_digit_anchors():
    path = ROOT / "notebooks" / "Trace_Count_v19_Colab.ipynb"
    assert path.exists()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert "drive.mount" in "".join(code_cells[0].get("source", []))
    contents = path.read_text(encoding="utf-8")
    contents_lower = contents.lower()
    for expected in (
        "synthetic_counting_v19.run_v19",
        "shared decimal digit",
        "final decimal digit",
        "attention_summary.csv",
        "state_pca_variance.csv",
        "Interactive PC1-PC6",
    ):
        assert expected.lower() in contents_lower
    for cell in code_cells:
        source = "".join(cell.get("source", []))
        compile(source, f"{path.name}:{cell.get('id', 'code-cell')}", "exec")
