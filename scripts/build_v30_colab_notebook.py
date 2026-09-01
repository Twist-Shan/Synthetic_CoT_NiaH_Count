from __future__ import annotations

import json
from pathlib import Path

from build_v29_colab_notebook import build as build_v29_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Trace_Count_v30_Depth6_Multiseed_Colab.ipynb"


def _cell(notebook: dict, cell_id: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)


def _set_source(notebook: dict, cell_id: str, source: str) -> None:
    _cell(notebook, cell_id)["source"] = source.splitlines(keepends=True)


def build() -> Path:
    source_path = build_v29_notebook()
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("synthetic_counting_v29", "synthetic_counting_v30")
        source = source.replace("run_v29", "run_v30")
        source = source.replace('VERSION = "v29"', 'VERSION = "v30"')
        source = source.replace('manifest["version"] == "v29"', 'manifest["version"] == "v30"')
        source = source.replace(
            "v29_countweight4_fixed_partial_readout_L256_pool100_seed",
            "v30_depth6_countweight4_partial_readout_L256_pool100_seed",
        )
        cell["source"] = source.splitlines(keepends=True)

    _set_source(
        notebook,
        "title",
        """# Trace Count v30: depth-only capacity control

This is a pre-registered three-seed paired experiment. It keeps v29's
256-character prompt, count support 1--10, 100 marker sets, uniform semantic
count sampler, separator/no-index trace, four attention heads, width 256,
partial count-only output untying, component-normalized count coefficient 4,
optimizer, 1,500-step language phase, and 10,000-step schedule.

The sole change from v29 is transformer depth: 4 layers become 6. The trace
tokens and targets are unchanged. There is no auxiliary objective, post-hoc
decoder, frozen phase, test-time update, or inference-time rule.

Primary gate, fixed before opening the three final test sets: in every seed,
Thinking accuracy >= 0.90, minimum per-count accuracy >= 0.80, trace exact >=
0.90, count spread <= 0.20, and Thinking-minus-Non-thinking accuracy >= 0.10.
All completed seeds are retained regardless of outcome.
""",
    )

    runtime = "".join(_cell(notebook, "runtime-settings")["source"])
    runtime = runtime.replace(
        "from synthetic_counting_v28.config import preset_config as v28_preset_config",
        "from synthetic_counting_v29.config import preset_config as v29_preset_config",
    )
    runtime = runtime.replace(
        "baseline = v28_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)",
        "baseline = v29_preset_config(PRESET, seed=SEEDS[0], device=DEVICE)",
    )
    runtime = runtime.replace(
        'assert changed_fields == {"version", "task_output_count_weight"}, changed_fields',
        'assert changed_fields == {"version", "n_layer"}, changed_fields',
    )
    runtime = runtime.replace(
        "assert planned.n_positions == 384\n",
        "assert planned.n_positions == 384\n"
        "assert baseline.n_layer == 4\n"
        "assert planned.n_layer == 6\n",
    )
    old_audit = '''# Step-zero audit: the scalar loss change cannot alter parameters or logits.
text = load_corpus_text()
baseline_vocab = V20Vocab.build(baseline, text)
planned_vocab = V20Vocab.build(planned, text)
baseline_model = build_model(baseline, baseline_vocab, device="cpu").eval()
planned_model = build_model(planned, planned_vocab, device="cpu").eval()
probe_ids = torch.tensor([[baseline_vocab.token_to_id["<BOS>"], baseline_vocab.token_to_id["<Ans>"]]])
with torch.no_grad():
    max_step0_logit_diff = float(
        (baseline_model(probe_ids).logits - planned_model(probe_ids).logits).abs().max()
    )
assert max_step0_logit_diff == 0.0
parameter_delta = planned_model.parameter_count() - baseline_model.parameter_count()
assert parameter_delta == 0
del baseline_model, planned_model'''
    new_audit = '''# Architecture audit: vocabulary and width are fixed; only two blocks are added.
text = load_corpus_text()
baseline_vocab = V20Vocab.build(baseline, text)
planned_vocab = V20Vocab.build(planned, text)
assert baseline_vocab == planned_vocab
baseline_model = build_model(baseline, baseline_vocab, device="cpu").eval()
planned_model = build_model(planned, planned_vocab, device="cpu").eval()
assert len(baseline_model.layers) == 4
assert len(planned_model.layers) == 6
parameter_delta = planned_model.parameter_count() - baseline_model.parameter_count()
assert parameter_delta > 0
del baseline_model, planned_model'''
    if old_audit not in runtime:
        raise RuntimeError("v29 architecture audit block not found")
    runtime = runtime.replace(old_audit, new_audit)
    runtime = runtime.replace(
        '"controlled_changes_from_v28": sorted(changed_fields),',
        '"controlled_changes_from_v29": sorted(changed_fields),',
    )
    runtime = runtime.replace(
        '"loss": "v28 component-normalized CE with count-region coefficient 4",',
        '"loss": "unchanged v29 component-normalized CE with count-region coefficient 4",',
    )
    runtime = runtime.replace(
        '"parameter_delta_from_v28": parameter_delta,',
        '"parameter_delta_from_v29": parameter_delta,',
    )
    runtime = runtime.replace(
        '"step0_max_logit_difference": max_step0_logit_diff,\n',
        "",
    )
    _set_source(notebook, "runtime-settings", runtime)

    notebook["metadata"]["colab"]["name"] = TARGET.name
    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return TARGET


if __name__ == "__main__":
    print(build())
