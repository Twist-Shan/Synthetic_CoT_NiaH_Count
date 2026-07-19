from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from synthetic_counting_v11.config import preset_config as legacy_preset_config
from synthetic_counting_v11.data import (
    Vocab,
    WindowWithoutReplacementSampler,
    corpus_split_bounds,
    split_target_window_starts,
)
from synthetic_counting_v18.config import canonical_run_specs, preset_config
from synthetic_counting_v18.data import (
    ANSWER,
    COUNT_OFFSET,
    END,
    INDEX_OFFSET,
    IGNORE_INDEX,
    MARKER_OFFSET,
    PAD,
    START,
    ReferenceExample,
    count_probabilities,
    count_token,
    index_token,
    render,
    sample_example,
)
from synthetic_counting_v18.model import ReferenceTransformer


ROOT = Path(__file__).resolve().parents[1]


def test_v16_2_uses_split_local_indexed_windows_without_replacement():
    cfg = legacy_preset_config("v16_2", "debug", device="cpu")
    assert cfg.training_data_mode == "split_window_index"
    assert cfg.window_sampling == "without_replacement"
    assert cfg.min_candidate_windows == 2
    assert np.isclose(
        cfg.corpus_train_fraction + cfg.corpus_validation_fraction,
        0.9,
    )

    bounds = corpus_split_bounds(cfg)
    assert bounds["train"][1] == bounds["validation"][0]
    assert bounds["validation"][1] == bounds["test"][0]

    index = split_target_window_starts(cfg, "train")
    eligible = {key: starts for key, starts in index.items() if len(starts) >= cfg.min_candidate_windows}
    assert eligible
    assert all(len(starts) >= cfg.min_candidate_windows for starts in eligible.values())


def test_v16_2_sampler_has_no_repeat_before_epoch_end_and_resumes_exactly():
    cfg = legacy_preset_config("v16_2", "debug", device="cpu")
    vocab = Vocab.build(cfg)
    sampler = WindowWithoutReplacementSampler(cfg, vocab, split="train", seed=73)
    draws = min(256, sampler.epoch_size)
    examples = sampler.sample(draws)
    identities = [(example.target_token, example.seed) for example in examples]
    assert len(identities) == len(set(identities))

    state = sampler.state_dict()
    expected = [
        (example.target_token, example.count, example.seed)
        for example in sampler.sample(20)
    ]
    restored = WindowWithoutReplacementSampler(cfg, vocab, split="train", seed=73)
    restored.load_state_dict(state)
    actual = [
        (example.target_token, example.count, example.seed)
        for example in restored.sample(20)
    ]
    assert actual == expected

    train_lo, train_hi = corpus_split_bounds(cfg)["train"]
    assert all(train_lo <= example.seed and example.seed + cfg.seq_len <= train_hi for example in examples)


def test_v16_2_sampling_is_natural_not_forced_uniform():
    cfg = legacy_preset_config("v16_2", "debug", device="cpu")
    vocab = Vocab.build(cfg)
    sampler = WindowWithoutReplacementSampler(cfg, vocab, split="train", seed=91)
    counts = Counter(example.count for example in sampler.sample(min(4_000, sampler.epoch_size)))
    assert len(counts) >= 2
    assert len(set(counts.values())) > 1


def test_v18_focused_suite_has_four_uniform_power_direct_cot_runs():
    specs = canonical_run_specs()
    assert len(specs) == 4
    assert len({spec.name for spec in specs}) == 4
    assert Counter(spec.distribution for spec in specs) == {"power": 2, "uniform": 2}
    assert Counter(spec.mode for spec in specs) == {"direct": 2, "cot": 2}
    assert {spec.context_length for spec in specs} == {1024}
    assert {spec.train_max_count for spec in specs} == {128}
    assert {spec.alpha for spec in specs if spec.distribution == "power"} == {1.5}


def test_v18_token_layout_separates_noise_marker_index_and_count_families():
    marker_a, marker_b = MARKER_OFFSET + 2, MARKER_OFFSET + 7
    example = ReferenceExample((5, marker_a, 99, marker_b), 2, (1, 3), (marker_a, marker_b))
    direct = render(example, "direct")
    assert direct.tokens == (*example.context, ANSWER, count_token(2))
    assert direct.labels == (IGNORE_INDEX,) * 5 + (count_token(2),)

    cot = render(example, "cot")
    completion = (index_token(1), marker_a, index_token(2), marker_b, END, count_token(2))
    assert cot.tokens == (*example.context, START, *completion)
    assert cot.labels == (IGNORE_INDEX,) * 5 + completion
    assert index_token(2) != count_token(2)
    assert INDEX_OFFSET > MARKER_OFFSET + 9
    assert COUNT_OFFSET > INDEX_OFFSET + 127
    assert PAD not in cot.tokens

    power = next(spec for spec in canonical_run_specs() if spec.name == "power_L1024_train128_eval128_a1.5_direct")
    probabilities = count_probabilities(power)
    assert len(probabilities) == 128
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[0] > probabilities[1] > probabilities[-1]


def test_v18_sampling_uses_256_noise_types_and_10_disjoint_marker_types():
    cfg = preset_config("debug", device="cpu")
    spec = next(spec for spec in canonical_run_specs() if spec.mode == "cot")
    example = sample_example(cfg, spec, random.Random(9), count=10)
    assert len(example.context) == 1024
    assert len(example.needle_positions) == len(example.needle_markers) == 10
    assert all(MARKER_OFFSET <= token < MARKER_OFFSET + 10 for token in example.needle_markers)
    marker_positions = set(example.needle_positions)
    assert all(
        (MARKER_OFFSET <= token < MARKER_OFFSET + 10) if index in marker_positions else (0 <= token < 256)
        for index, token in enumerate(example.context)
    )


def test_v18_model_is_reference_scale_rope_and_tied_unembedding():
    cfg = preset_config("debug", device="cpu")
    model = ReferenceTransformer(cfg).eval()
    ids = torch.tensor([[5, 6, START, index_token(1), MARKER_OFFSET, END]], dtype=torch.long)
    with torch.no_grad():
        output = model(ids)
    assert output.logits.shape == (1, 6, cfg.vocab_size)
    assert (cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.n_inner) == (4, 4, 256, 1024)
    assert cfg.n_embd // cfg.n_head == 64
    assert model.parameter_count() > 3_000_000
    assert model.parameter_count() < 4_500_000


def test_v18_model_exposes_causal_attention_and_layerwise_hidden_states():
    cfg = preset_config("debug", device="cpu")
    model = ReferenceTransformer(cfg).eval()
    ids = torch.tensor(
        [[5, MARKER_OFFSET, START, index_token(1), MARKER_OFFSET, END]],
        dtype=torch.long,
    )
    with torch.no_grad():
        output = model(ids, output_attentions=True, output_hidden_states=True)

    assert output.attentions is not None
    assert output.hidden_states is not None
    assert len(output.attentions) == cfg.n_layer
    assert len(output.hidden_states) == cfg.n_layer + 1
    assert output.attentions[0].shape == (1, cfg.n_head, ids.shape[1], ids.shape[1])
    assert output.hidden_states[-1].shape == (1, ids.shape[1], cfg.n_embd)

    weights = output.attentions[0]
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)
    future = torch.triu(torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(weights[..., future]) == 0


def test_v16_2_v18_notebooks_mount_drive_first_and_compile():
    for version in ("v16_2", "v18"):
        path = ROOT / "notebooks" / f"Trace_Count_{version}_Colab.ipynb"
        assert path.exists()
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        first = "".join(code_cells[0].get("source", []))
        assert "drive.mount" in first
        assert f"synthetic_counting_{version}.run_{version}" in path.read_text(encoding="utf-8")
        for cell in code_cells:
            source = "".join(cell.get("source", []))
            compile(source, f"{path.name}:{cell.get('id', 'code-cell')}", "exec")


def test_v18_notebook_exposes_learning_attention_and_state_outputs():
    path = ROOT / "notebooks" / "Trace_Count_v18_Colab.ipynb"
    contents = path.read_text(encoding="utf-8")
    for expected in (
        "dynamics_by_band.csv",
        "attention_summary.csv",
        "state_probe_summary.csv",
        "state_pca_variance.csv",
        "state_centroids_pca.csv",
        "correct_prompt_needle_mass",
        "broad_attention_score",
        "Interactive PC1-PC6",
    ):
        assert expected in contents
