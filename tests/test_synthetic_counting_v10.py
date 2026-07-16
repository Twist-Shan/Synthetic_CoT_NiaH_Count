from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch

from synthetic_counting_v10.attention_causal import (
    _head_mask,
    attention_categories,
    collect_attention,
)
from synthetic_counting_v10.config import preset_config
from synthetic_counting_v10.core import (
    Vocab,
    build_model,
    component_target_positions,
    make_example,
    render,
)
from synthetic_counting_v10.state_causal import (
    SITE_NONTHINKING_FINAL,
    _fit_pca,
    fit_count_directions,
)
from synthetic_counting_v10.report_followups import (
    _forward_with_position_local_head_ablation,
    _layer_matched_order,
    _local_query_positions,
)
from synthetic_counting_v10.successor_patching import (
    DIRECTION_CLOSE,
    DIRECTION_CONTINUE,
    run_successor_patching_rows,
    successor_nested_example_pair,
)
from synthetic_counting_v10.successor_conversion import (
    INTERVENTIONS,
    run_successor_conversion_rows,
    summarize_successor_conversion,
    summarize_successor_logit_lens,
)
from synthetic_counting_v10.successor_mlp_features import (
    _analysis_layers,
    evaluate_mlp_feature_patching,
    fit_mlp_feature_statistics,
    summarize_feature_concentration,
    summarize_mlp_feature_patching,
)
from synthetic_counting_v10.geometry_path_steering import (
    centroid_chord_point,
    centroid_polyline_point,
)
from synthetic_counting_v10.hidden_state_patching import (
    _decode_free_rollout,
    _directional_regression_summary,
    _free_rollout_with_residual_patch,
    _patched_forward,
    _post_block_states,
    _prefix_nested_pair,
    _rollout_factor_summary,
    _thinking_prefix_ids,
    _truncated_thinking_ids,
    default_rollout_scenarios,
    run_final_answer_patching,
    run_misaligned_trace_rollout_patching,
    run_trace_early_stop_patching,
    run_trace_final_patching,
)
from synthetic_counting_v10.head_state_bidirectional import (
    _forward_with_residual_patch_and_attention,
    _pre_block_states,
    run_head_to_state,
    run_state_to_head,
    summarize_head_to_state,
    summarize_state_to_head,
)
from synthetic_counting_v10.final_bridge_causal import (
    CONFLICT_FINAL_INDEX,
    CONFLICT_MARKERS,
    CONFLICT_PROMPT,
    CONFLICT_TRACE,
    _feature_replacements,
    _normalized_recovery,
    build_conflict_case,
)


def tiny_setup():
    cfg = preset_config("debug", count_max=6)
    vocab = Vocab.build(cfg)
    model = build_model(cfg, vocab, "cpu").eval()
    return cfg, vocab, model


def test_v10_uses_separate_v2_formats_and_shared_numeric_tokens():
    cfg, vocab, _ = tiny_setup()
    example = make_example(cfg, vocab, random.Random(1), count=6)
    direct = render(example, vocab, "nonthinking")
    thinking = render(example, vocab, "thinking")

    assert direct.tokens[-3:] == ["<Ans>", "<6>", "<EOS>"]
    assert thinking.tokens[-4:] == ["</Think>", "<Ans>", "<6>", "<EOS>"]
    assert [thinking.tokens[pos] for pos in thinking.spans.trace_index_positions] == [
        f"<{value}>" for value in range(1, 7)
    ]
    assert thinking.tokens[thinking.spans.count_pos] == thinking.tokens[thinking.spans.trace_index_positions[-1]]
    assert len(thinking.tokens) <= cfg.n_positions
    assert component_target_positions(direct)["final_count"] == [direct.spans.count_pos]
    assert len(component_target_positions(thinking)["trace_marker"]) == 6


def test_attention_categories_partition_the_causal_row():
    cfg, vocab, _ = tiny_setup()
    example = make_example(cfg, vocab, random.Random(2), count=4)
    item = render(example, vocab, "thinking")
    row = np.zeros(len(item.tokens), dtype=float)
    row[: item.spans.ans_pos + 1] = 1.0 / (item.spans.ans_pos + 1)
    categories = attention_categories(item, row)
    assert abs(categories["classified_mass"] + categories["other_or_query_self_mass"] - 1.0) < 1e-6
    assert categories["prompt_needles_mass"] > 0
    assert 0 <= categories["broad_attention_score"] <= categories["prompt_needles_mass"]


def test_tiny_attention_collection_and_head_mask():
    cfg, vocab, model = tiny_setup()
    examples = [make_example(cfg, vocab, random.Random(3), count=3)]
    detail, summary = collect_attention(model, vocab, examples, "thinking", "cpu")
    assert not detail.empty
    assert {"final_count_query", "targeted_retrieval_query", "successor_query"}.issubset(
        set(summary.query_kind)
    )
    mask = _head_mask(model, [(0, 1), (1, 0)], "cpu")
    assert mask.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_pca_and_direction_geometry_recover_a_known_count_axis():
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(1, 7), 8)
    values = rng.normal(scale=0.01, size=(len(labels), 12))
    values[:, 0] += labels * 2.0
    _, coordinates, variance = _fit_pca(values, 6)
    assert coordinates.shape == (len(labels), 6)
    assert variance[:6].sum() <= 1.0 + 1e-8

    states = {SITE_NONTHINKING_FINAL: [values, values + 0.1]}
    eval_values = values + rng.normal(scale=0.005, size=values.shape)
    directions, geometry, centroids = fit_count_directions(
        states,
        {SITE_NONTHINKING_FINAL: labels},
        {SITE_NONTHINKING_FINAL: [eval_values, eval_values + 0.1]},
        {SITE_NONTHINKING_FINAL: labels},
    )
    assert (SITE_NONTHINKING_FINAL, 0, "adjacent_mean") in directions
    assert geometry.projection_r2_heldout.max() > 0.99
    assert (SITE_NONTHINKING_FINAL, 0, 6) in centroids


def test_v10_main_matches_requested_v2_architecture_and_count_range():
    cfg = preset_config("main")
    assert (cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.n_inner) == (4, 4, 256, 1024)
    assert (cfg.seq_len, cfg.count_min, cfg.count_max) == (256, 1, 30)
    assert cfg.early_stop_patience == 0
    assert cfg.max_render_len == 322
    assert cfg.n_positions >= cfg.max_render_len


def test_final_bridge_conflicts_isolate_prompt_trace_and_last_index_sources():
    cfg, vocab, _ = tiny_setup()
    receiver_count, donor_count, seed = 3, 5, 917

    prompt = build_conflict_case(
        cfg, vocab, receiver_count, donor_count, seed, CONFLICT_PROMPT
    )
    assert len(prompt.conflict.prompt_needle_positions) == donor_count
    assert len(prompt.conflict.spans.trace_index_positions) == receiver_count
    assert prompt.conflict.spans.ans_pos == 1 + cfg.seq_len + 1 + 2 * receiver_count + 1
    assert (prompt.target_count, prompt.alternative_count) == (donor_count, receiver_count)

    trace = build_conflict_case(
        cfg, vocab, receiver_count, donor_count, seed, CONFLICT_TRACE
    )
    assert len(trace.conflict.prompt_needle_positions) == receiver_count
    assert len(trace.conflict.spans.trace_index_positions) == donor_count
    assert trace.conflict.spans.ans_pos == trace.clean.spans.ans_pos

    last_index = build_conflict_case(
        cfg, vocab, receiver_count, donor_count, seed, CONFLICT_FINAL_INDEX
    )
    assert len(last_index.conflict.spans.trace_index_positions) == receiver_count
    assert last_index.conflict.spans.ans_pos == 1 + cfg.seq_len + 1 + 2 * receiver_count + 1
    assert last_index.conflict.tokens[last_index.conflict.spans.trace_index_positions[-1]] == "<5>"
    assert last_index.conflict.tokens[last_index.conflict.spans.trace_index_positions[-2]] == "<2>"

    marker_control = build_conflict_case(
        cfg, vocab, receiver_count, donor_count, seed, CONFLICT_MARKERS
    )
    assert marker_control.target_count == receiver_count
    assert marker_control.clean.spans.ans_pos == marker_control.conflict.spans.ans_pos
    assert marker_control.clean.tokens != marker_control.conflict.tokens


def test_final_bridge_recovery_and_feature_support_controls_are_well_defined():
    assert _normalized_recovery(4.0, -2.0, 1.0) == 0.5
    assert np.isnan(_normalized_recovery(1.0, 1.0, 1.0))

    donor = torch.arange(12, dtype=torch.float32)
    receiver = torch.zeros(12, dtype=torch.float32)
    ranking = torch.arange(11, -1, -1)
    conditions, replacements = _feature_replacements(
        donor,
        receiver,
        ranking,
        support_sizes=(2, 12),
        random_replicates=1,
        seed=29,
    )
    assert len(conditions) == len(replacements) == 4
    full = [
        replacement
        for condition, replacement in zip(conditions, replacements)
        if condition["family"] == "ranked" and condition["support_size"] == 12
    ]
    assert len(full) == 1
    assert torch.equal(full[0], donor)


def test_bidirectional_residual_patch_is_pre_layer_and_hook_safe():
    cfg, vocab, model = tiny_setup()
    item = render(make_example(cfg, vocab, random.Random(45), count=5), vocab, "thinking")
    ids = torch.tensor([item.input_ids], dtype=torch.long)
    before = sum(len(module._forward_pre_hooks) for module in model.modules())
    states = _pre_block_states(model, ids, item.spans.ans_pos)
    logits, attentions = _forward_with_residual_patch_and_attention(
        model, ids, 0, item.spans.ans_pos, states[0]
    )
    after = sum(len(module._forward_pre_hooks) for module in model.modules())
    assert len(states) == cfg.n_layer
    assert logits.shape[0] == ids.shape[1]
    assert logits.shape[1] == len(vocab.id_to_token)
    assert len(attentions) == cfg.n_layer
    assert before == after


def test_bidirectional_analysis_tables_cover_both_causal_directions():
    cfg = preset_config("debug", device="cpu")
    vocab = Vocab.build(cfg)
    models = {mode: build_model(cfg, vocab, "cpu").eval() for mode in cfg.modes}
    all_heads = [(layer, head) for layer in range(cfg.n_layer) for head in range(cfg.n_head)]
    rankings = {
        "direct_broad": all_heads,
        "targeted_retrieval": list(reversed(all_heads)),
        "trace_readout": all_heads,
        "random": list(reversed(all_heads)),
    }
    head_detail = run_head_to_state(
        models,
        cfg,
        vocab,
        rankings,
        centroid_examples_per_count=1,
        eval_examples_per_count=1,
        seed=46,
    )
    state_detail = run_state_to_head(
        models, cfg, vocab, rankings, examples_per_bin=1, seed=47
    )
    assert set(head_detail.mechanism) == {
        "nonthinking_broad", "cot_targeted", "cot_readout"
    }
    assert {"clean", "candidate_top4", "noncandidate4_control"}.issubset(
        set(head_detail.intervention)
    )
    state_summary = summarize_state_to_head(state_detail)
    assert {"candidate_top4", "all_downstream"} == set(state_summary.head_scope)
    assert {"clean", "same_state_control", "shifted_state_patch"} == set(
        state_detail.intervention
    )
    assert state_detail.patch_before_layer.min() == 1
    assert not summarize_head_to_state(head_detail).empty
    assert not summarize_state_to_head(state_detail).empty


def test_layer_matched_controls_preserve_the_ranked_layer_sequence():
    ranking = [(2, 1), (3, 0), (2, 3), (0, 2), (3, 2), (0, 1)]
    low = _layer_matched_order(ranking, reverse_within_layer=True)
    random_order = _layer_matched_order(ranking, seed=19)
    assert [layer for layer, _ in low] == [layer for layer, _ in ranking]
    assert [layer for layer, _ in random_order] == [layer for layer, _ in ranking]
    assert sorted(low) == sorted(ranking)
    assert sorted(random_order) == sorted(ranking)


def test_position_local_ablation_runs_at_trace_and_answer_queries_without_leaking_hooks():
    cfg, vocab, model = tiny_setup()
    example = make_example(cfg, vocab, random.Random(7), count=4)
    item = render(example, vocab, "thinking")
    ids = torch.tensor([item.input_ids], dtype=torch.long)
    mask = torch.ones_like(ids)
    baseline = model(input_ids=ids, attention_mask=mask).logits
    local = _forward_with_position_local_head_ablation(
        model,
        ids,
        mask,
        [(0, 0)],
        [_local_query_positions(item, "trace_index_tokens")],
    ).logits
    restored = model(input_ids=ids, attention_mask=mask).logits
    assert local.shape == baseline.shape
    assert not torch.allclose(local, baseline)
    assert torch.allclose(restored, baseline)
    assert _local_query_positions(item, "ans_token") == [item.spans.ans_pos]


def test_successor_pair_is_identical_through_marker_k_and_then_changes_target():
    cfg, vocab, _ = tiny_setup()
    short, long = successor_nested_example_pair(cfg, vocab, k=4, seed=81)
    assert short.count == 4
    assert long.count == 5
    assert short.needle_positions == long.needle_positions[:4]
    assert short.needle_markers == long.needle_markers[:4]
    assert long.needle_positions[-1] > short.needle_positions[-1]

    short_item = render(short, vocab, "thinking")
    long_item = render(long, vocab, "thinking")
    short_query = short_item.spans.trace_marker_positions[3]
    long_query = long_item.spans.trace_marker_positions[3]
    assert short_query == long_query
    assert (
        short_item.tokens[short_item.spans.think_pos : short_query + 1]
        == long_item.tokens[long_item.spans.think_pos : long_query + 1]
    )
    assert short_item.tokens[short_query + 1] == "</Think>"
    assert long_item.tokens[long_query + 1] == "<5>"


def test_successor_patching_smoke_records_both_directions_and_controls():
    cfg, vocab, model = tiny_setup()
    rankings = {
        "successor": [(0, 0), (1, 0), (0, 1), (1, 1)],
        "targeted_retrieval": [(1, 1), (0, 1), (1, 0), (0, 0)],
    }
    detail = run_successor_patching_rows(
        model,
        cfg,
        vocab,
        rankings,
        examples_per_k=1,
        random_replicates=1,
        top_ns=(1, 2),
    )
    assert set(detail.direction) == {DIRECTION_CONTINUE, DIRECTION_CLOSE}
    assert set(detail.family) == {
        "successor_top",
        "targeted_top",
        "successor_wrong_row",
        "random",
    }
    assert set(detail.donor_row) == {"marker_query", "index_query"}
    assert detail.normalized_recovery.notna().any()


def test_successor_conversion_smoke_records_all_sublayers_and_interventions():
    cfg, vocab, model = tiny_setup()
    logit_detail, patch_detail = run_successor_conversion_rows(
        model,
        cfg,
        vocab,
        examples_per_k=1,
    )
    assert not logit_detail.empty
    assert not patch_detail.empty
    assert set(logit_detail.stage) == {
        "resid_pre",
        "attn_out",
        "post_attn",
        "mlp_out",
        "post_mlp",
    }
    assert set(patch_detail.intervention) == set(INTERVENTIONS)
    assert patch_detail.patched_margin.notna().all()
    assert not summarize_successor_logit_lens(logit_detail).empty
    assert not summarize_successor_conversion(patch_detail).empty


def test_successor_mlp_feature_fit_and_held_out_patch_smoke():
    cfg, vocab, model = tiny_setup()
    layers = _analysis_layers(cfg)
    assert layers == (0, 1)
    statistics, fitted = fit_mlp_feature_statistics(
        model,
        cfg,
        vocab,
        fit_examples_per_k=1,
        layers=layers,
    )
    assert not statistics.empty
    assert set(statistics.layer) == set(layers)
    assert statistics.groupby(["direction", "count_bin", "layer"]).size().min() == cfg.n_inner
    concentration = summarize_feature_concentration(statistics, (1, 4, cfg.n_inner))
    assert set(concentration.support_size) == {1, 4, cfg.n_inner}

    detail = evaluate_mlp_feature_patching(
        model,
        cfg,
        vocab,
        fitted,
        eval_examples_per_k=1,
        fit_examples_per_k=1,
        support_sizes=(1, 4, cfg.n_inner),
        random_replicates=1,
        layers=layers,
        patch_batch_size=8,
    )
    assert not detail.empty
    assert set(detail.family) == {
        "ranked_feature_replacement",
        "random_feature_replacement",
        "sparse_mean_direction",
        "random_sparse_mean_direction",
    }
    assert detail.patched_margin.notna().all()
    full = detail[
        (detail.family == "ranked_feature_replacement")
        & (detail.support_size == cfg.n_inner)
    ]
    assert not full.empty
    assert np.isfinite(full.normalized_recovery).all()
    assert not summarize_mlp_feature_patching(detail).empty


def test_centroid_curve_differs_midway_but_matches_chord_endpoint():
    centroids = {
        1: np.asarray([0.0, 0.0], dtype=np.float32),
        2: np.asarray([1.0, 0.0], dtype=np.float32),
        3: np.asarray([1.0, 2.0], dtype=np.float32),
    }
    chord_mid, chord_count = centroid_chord_point(centroids, 1, 3, 0.5)
    curve_mid, curve_count = centroid_polyline_point(centroids, 1, 3, 0.5)
    assert np.allclose(chord_mid, [0.5, 1.0])
    assert np.allclose(curve_mid, [1.0, 0.5])
    assert chord_count == 2.0
    assert curve_count == 2.25
    chord_end, chord_end_count = centroid_chord_point(centroids, 1, 3, 1.0)
    curve_end, curve_end_count = centroid_polyline_point(centroids, 1, 3, 1.0)
    assert np.allclose(chord_end, centroids[3])
    assert np.allclose(curve_end, centroids[3])
    assert chord_end_count == curve_end_count == 3.0


def test_centroid_curve_supports_reverse_count_transport():
    centroids = {
        1: np.asarray([0.0, 0.0], dtype=np.float32),
        2: np.asarray([1.0, 0.0], dtype=np.float32),
        3: np.asarray([1.0, 2.0], dtype=np.float32),
    }
    start, start_count = centroid_polyline_point(centroids, 3, 1, 0.0)
    end, end_count = centroid_polyline_point(centroids, 3, 1, 1.0)
    assert np.allclose(start, centroids[3])
    assert np.allclose(end, centroids[1])
    assert start_count == 3.0
    assert end_count == 1.0


def test_hidden_state_patch_uses_matched_post_block_space_and_removes_hook():
    cfg, vocab, model = tiny_setup()
    receiver = make_example(cfg, vocab, random.Random(91), count=3)
    donor = make_example(cfg, vocab, random.Random(92), count=4)
    receiver_item = render(receiver, vocab, "nonthinking")
    donor_item = render(donor, vocab, "nonthinking")
    receiver_ids = torch.tensor([receiver_item.input_ids], dtype=torch.long)
    donor_ids = torch.tensor([donor_item.input_ids], dtype=torch.long)
    baseline, receiver_states = _post_block_states(
        model, receiver_ids, receiver_item.spans.ans_pos
    )
    _, donor_states = _post_block_states(model, donor_ids, donor_item.spans.ans_pos)
    patched = _patched_forward(
        model,
        receiver_ids,
        0,
        receiver_item.spans.ans_pos,
        donor_states[0],
    )
    restored = model(input_ids=receiver_ids).logits[0]
    assert len(receiver_states) == cfg.n_layer
    assert receiver_states[0].shape == donor_states[0].shape == (1, cfg.n_embd)
    assert patched.shape == baseline.shape
    assert not torch.allclose(patched, baseline)
    assert torch.allclose(restored, baseline)


def test_prefix_pair_and_truncated_trace_preserve_Mm_position_without_gold_tail():
    cfg, vocab, _ = tiny_setup()
    short, long = _prefix_nested_pair(cfg, vocab, 3, 5, seed=93)
    assert short.needle_positions == long.needle_positions[:3]
    assert short.needle_markers == long.needle_markers[:3]
    short_item = render(short, vocab, "thinking")
    long_item = render(long, vocab, "thinking")
    truncated_ids, marker_pos, ans_pos = _truncated_thinking_ids(long, vocab, 3)
    assert marker_pos == short_item.spans.trace_marker_positions[-1]
    assert marker_pos == long_item.spans.trace_marker_positions[2]
    assert vocab.decode(truncated_ids[-3:]) == [short.needle_markers[-1], "</Think>", "<Ans>"]
    assert ans_pos == len(truncated_ids) - 1
    assert vocab.number_id(5) not in truncated_ids[ans_pos + 1 :]


def test_free_rollout_prefix_and_parser_do_not_supply_gold_trace_tail():
    cfg, vocab, model = tiny_setup()
    example = make_example(cfg, vocab, random.Random(94), count=5)
    prefix, marker_pos = _thinking_prefix_ids(example, vocab, 3)
    tokens = vocab.decode(prefix)
    assert marker_pos == len(prefix) - 1
    assert tokens[-2:] == ["<3>", example.needle_markers[2]]
    assert "</Think>" not in tokens
    assert "<Ans>" not in tokens
    assert "<4>" not in tokens[marker_pos + 1 :]

    suffix = vocab.encode(["<4>", example.needle_markers[3], "</Think>", "<Ans>", "<5>", "<EOS>"])
    parsed = _decode_free_rollout(suffix, vocab, receiver_progress=3)
    assert parsed["closed_trace"] == 1.0
    assert parsed["first_generated_index"] == 4
    assert parsed["inferred_stop_index"] == 4
    assert parsed["final_count"] == 5

    generated = _free_rollout_with_residual_patch(
        model, prefix, vocab, max_new_tokens=2
    )
    assert len(generated) <= 2


def test_hidden_state_patching_smoke_covers_final_trace_and_early_stop_protocols():
    cfg = preset_config("debug", count_max=4, patch_offsets=(-1, 1))
    vocab = Vocab.build(cfg)
    nonthinking = build_model(cfg, vocab, "cpu").eval()
    thinking = build_model(cfg, vocab, "cpu").eval()
    final_detail, final_summary = run_final_answer_patching(
        {"nonthinking": nonthinking, "thinking": thinking},
        cfg,
        vocab,
        examples_per_pair=1,
    )
    trace_detail, trace_summary = run_trace_final_patching(
        thinking, cfg, vocab, examples_per_pair=1
    )
    early_detail, early_summary = run_trace_early_stop_patching(
        thinking, cfg, vocab, examples_per_pair=1
    )
    assert set(final_detail["mode"]) == {"nonthinking", "thinking"}
    assert set(final_detail["patch_direction"]) == {
        "donor_gt_receiver",
        "donor_lt_receiver",
        "same_count",
    }
    assert (
        final_detail.groupby(["mode", "receiver_count", "donor_count"])["layer"]
        .nunique()
        .min()
        == cfg.n_layer
    )
    assert set(trace_detail["site"]) == {"trace_final_marker_to_final_marker"}
    assert set(trace_detail["patch_direction"]) == {
        "donor_gt_receiver",
        "donor_lt_receiver",
    }
    assert set(early_detail["site"]) == {"trace_prefix_early_stop"}
    assert not early_detail["uses_gold_trace_tail"].any()
    assert early_detail["forced_close_and_ans_tokens"].all()
    assert {"transport_slope", "transport_r2"}.issubset(final_summary.columns)
    assert {"transport_slope", "mean_close_margin_shift"}.issubset(early_summary.columns)
    assert not trace_summary.empty
    directional = _directional_regression_summary(
        final_detail,
        ["mode", "site", "count_bin", "layer"],
    )
    assert set(directional.patch_direction) == {
        "donor_gt_receiver",
        "donor_lt_receiver",
    }
    assert (directional.n_rows > 0).all()


def test_misaligned_later_progress_rollout_patching_smoke():
    cfg = preset_config("debug", count_max=6)
    vocab = Vocab.build(cfg)
    thinking = build_model(cfg, vocab, "cpu").eval()
    detail, summary = run_misaligned_trace_rollout_patching(
        thinking,
        cfg,
        vocab,
        receiver_count=3,
        receiver_progress=2,
        donor_count=5,
        donor_progress=4,
        examples=1,
        centroid_examples=1,
        max_new_tokens=4,
    )
    assert not detail.empty and not summary.empty
    assert set(detail.patch_policy) == {"none", "one_shot", "persistent"}
    assert not detail.uses_gold_trace_tail.any()
    assert not detail.forced_close_or_ans.any()
    assert (detail.donor_count > detail.receiver_count).all()
    assert (detail.donor_progress > detail.receiver_progress).all()
    patched = detail[detail.patch_policy != "none"]
    assert patched.layer.nunique() == cfg.n_layer
    assert {
        "later_progress_donor_full",
        "total_only_centroid_delta",
        "progress_only_centroid_delta",
        "combined_centroid_delta",
    }.issubset(set(patched.intervention))
    assert {
        "receiver_count",
        "receiver_progress",
        "donor_count",
        "donor_progress",
    }.issubset(summary.columns)


def test_misaligned_rollout_can_select_final_layer_and_one_shot_only():
    cfg = preset_config("debug", count_max=6)
    vocab = Vocab.build(cfg)
    thinking = build_model(cfg, vocab, "cpu").eval()
    detail, _ = run_misaligned_trace_rollout_patching(
        thinking,
        cfg,
        vocab,
        receiver_count=3,
        receiver_progress=2,
        donor_count=5,
        donor_progress=4,
        examples=1,
        centroid_examples=1,
        max_new_tokens=2,
        layers=[cfg.n_layer],
        patch_policies=["one_shot"],
    )
    patched = detail[detail.patch_policy != "none"]
    assert set(patched.layer) == {cfg.n_layer}
    assert set(patched.patch_policy) == {"one_shot"}
    assert len(detail) == 8


def test_default_rollout_scenarios_cover_total_progress_factorial_design():
    scenarios = default_rollout_scenarios(30)
    assert len(scenarios) == 12
    assert {scenario["scenario_family"] for scenario in scenarios} == {
        "both_up",
        "both_down",
        "progress_only",
        "total_only",
        "opposed",
    }
    scenario_ids = {scenario["scenario_id"] for scenario in scenarios}
    assert len(scenario_ids) == len(scenarios)
    assert all(
        scenario["receiver_progress"] < scenario["donor_count"]
        and scenario["donor_progress"] <= scenario["donor_count"]
        for scenario in scenarios
    )
    assert any(
        scenario["donor_count"] > scenario["receiver_count"]
        and scenario["donor_progress"] > scenario["receiver_progress"]
        for scenario in scenarios
    )
    assert any(
        scenario["donor_count"] < scenario["receiver_count"]
        and scenario["donor_progress"] < scenario["receiver_progress"]
        for scenario in scenarios
    )
    assert any(
        scenario["donor_count"] == scenario["receiver_count"]
        and scenario["donor_progress"] != scenario["receiver_progress"]
        for scenario in scenarios
    )
    assert any(
        scenario["donor_count"] != scenario["receiver_count"]
        and scenario["donor_progress"] == scenario["receiver_progress"]
        for scenario in scenarios
    )


def test_rollout_factor_summary_separates_total_and_progress_offsets():
    rows = []
    for total_offset, progress_offset in [
        (-10, -7),
        (-10, 0),
        (0, -7),
        (0, 3),
        (10, 0),
        (10, 7),
        (20, -3),
        (-10, 5),
    ]:
        receiver_count = 15
        receiver_progress = 8
        rows.append(
            {
                "scenario_id": f"t{total_offset}_p{progress_offset}",
                "scenario_family": "synthetic",
                "intervention": "combined_centroid_delta",
                "patch_policy": "one_shot",
                "layer": 4,
                "receiver_count": receiver_count,
                "receiver_progress": receiver_progress,
                "donor_count": receiver_count + total_offset,
                "donor_progress": receiver_progress + progress_offset,
                "first_generated_index": receiver_progress + 1 + 2.0 * progress_offset,
                "inferred_stop_index": receiver_count + 1.5 * progress_offset,
                "final_count": receiver_count + 0.75 * total_offset,
            }
        )
    summary = _rollout_factor_summary(pd.DataFrame(rows))
    coefficients = {
        row.outcome: (
            row.donor_total_coefficient,
            row.donor_progress_coefficient,
        )
        for row in summary.itertuples(index=False)
    }
    assert np.allclose(coefficients["first_index_shift"], (0.0, 2.0))
    assert np.allclose(coefficients["stop_index_shift"], (0.0, 1.5))
    assert np.allclose(coefficients["final_count_shift"], (0.75, 0.0))
