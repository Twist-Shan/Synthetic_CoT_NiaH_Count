from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Trace_Count_v3_2_Colab.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(True),
    }


cells = [
    md(
        r"""
# Trace Count v3.2: Causal Tests for v2 Thinking Attention

这个 notebook **不训练新模型**。它直接读取 v2 `thinking` checkpoint，做一组 causal tests 来回答：

> v2 里那个很强的 targeted retrieval head，比如 L3H3，到底只是 attention 相关性，还是实际影响 trace 生成和最终 count readout？

重点输出：

- **necessity**：ablate / scale 候选 heads，看 final-answer logit margin、trace exactness、count shift 是否变化；
- **sufficiency**：clean-to-corrupt activation patching，看 retrieval head output / residual stream 能否把 corrupted prompt 的 logit 推回 clean target；
- **counterfactual edits**：prompt 改了但 trace 不改、trace 改了但 prompt 不改时，最终答案更跟谁走；
- **谨慎结论**：区分 diagnostic attention、redundant causal route、necessary component、sufficient signal carrier。

约定：报告里的 layer 是 **1-based**，head 是 **0-based**。所以 `L3H3` 对应 Hugging Face 内部 `model.transformer.h[2].attn` 的 head 3。模型架构沿用 v2：GPT-2 style decoder-only Transformer with learned absolute positional embeddings，不是 RoPE。
        """
    ),
    md("## 1. Environment and Repo Setup"),
    code(
        r"""
from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any
import json
import math
import os
import random
import shutil
import subprocess
import sys
import warnings

REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
INSTALL_DEPS = False
FIX_NUMPY_ABI = False

IN_COLAB = "google.colab" in sys.modules or Path("/content").exists()
if IN_COLAB:
    repo_dir = Path("/content/Synthetic_CoT_NiaH_Count")
    cwd = Path.cwd()
    if (cwd / ".git").exists() or (cwd / "notebooks" / "Trace_Count_v2_Colab.ipynb").exists():
        repo_dir = cwd
    elif (repo_dir / ".git").exists() or (repo_dir / "notebooks" / "Trace_Count_v2_Colab.ipynb").exists():
        pass
    elif repo_dir.exists() and any(repo_dir.iterdir()):
        print(f"Using existing non-git directory: {repo_dir}")
    else:
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
    os.chdir(repo_dir)

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if INSTALL_DEPS:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40", "tqdm", "matplotlib", "pandas"],
        check=True,
    )

if FIX_NUMPY_ABI:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--force-reinstall",
            "numpy<2",
            "pandas",
            "matplotlib",
            "scipy",
            "scikit-learn",
        ],
        check=True,
    )
    raise RuntimeError("NumPy ABI repair finished. Restart the kernel/runtime, set FIX_NUMPY_ABI=False, then rerun.")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import Markdown, display
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)

display(Markdown(f"**Repo root:** `{ROOT}`"))
display(Markdown(f"**Device available:** `{'cuda' if torch.cuda.is_available() else 'cpu'}`"))
        """
    ),
    md("## 2. Runtime Settings"),
    code(
        r"""
# Set this if auto-discovery cannot find your v2 run.
# Examples:
# V2_RUN_DIR_OVERRIDE = "runs/v2_marker_trace_seed1234_main"
# V2_RUN_DIR_OVERRIDE = "/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/Synthetic_CoT_NiaH_Count/colab_results/v2_marker_trace_main_seed1234_20260706_215757/run"
V2_RUN_DIR_OVERRIDE = ""

# Optional: point to the v3 deep-dive report bundle to reuse candidate-head tables.
# Example: "colab_results/v3_v2_attention_deepdive_seed1234_20260708_053824"
V3_REPORT_DIR_OVERRIDE = ""

PRESET = "debug"  # "debug" or "main"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 1234
AUTO_MOUNT_DRIVE_FOR_V2_INPUTS = True
SKIP_COMPLETED = True

RUN_NECESSITY = True
RUN_DOSE_RESPONSE = True
RUN_ACTIVATION_PATCHING = True
RUN_PATH_PATCHING = True
RUN_COUNTERFACTUALS = True
RUN_RESIDUAL_DIRECTIONS = False
RUN_AR_GENERATION = True

if PRESET == "debug":
    EXAMPLES_PER_COUNT = 5
    PATCH_PAIRS_PER_COUNT = 5
    MAX_AR_EXAMPLES = 60
    ALPHAS = [-1.0, 0.0, 0.5, 1.0, 1.5]
else:
    EXAMPLES_PER_COUNT = 40
    PATCH_PAIRS_PER_COUNT = 40
    MAX_AR_EXAMPLES = 400
    ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

POSITION_SCOPES = ["all_positions", "trace_positions", "index_token_all", "index_token_last"]
PATCH_SCOPES = ["index_token_last", "pre_index_last", "think_end", "ans_token"]

DRIVE_RESULTS_ROOT = Path("/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/Synthetic_CoT_NiaH_Count/colab_results")
LOCAL_RESULTS_ROOT = Path("colab_results")
SAVE_TO_DRIVE = True
ENABLE_GITHUB_PUSH = False
AUTO_DISCONNECT = False

estimated = {
    "necessity_teacher_forced_forward_passes": int(RUN_NECESSITY) * 10 * len(POSITION_SCOPES) * EXAMPLES_PER_COUNT * 10,
    "dose_response_forward_passes": int(RUN_DOSE_RESPONSE) * 7 * len(ALPHAS) * EXAMPLES_PER_COUNT * 10,
    "activation_patching_pairs": int(RUN_ACTIVATION_PATCHING) * PATCH_PAIRS_PER_COUNT * 18,
    "counterfactual_examples": int(RUN_COUNTERFACTUALS) * EXAMPLES_PER_COUNT * 10,
}
display(pd.DataFrame([{
    "PRESET": PRESET,
    "DEVICE": DEVICE,
    "EXAMPLES_PER_COUNT": EXAMPLES_PER_COUNT,
    "PATCH_PAIRS_PER_COUNT": PATCH_PAIRS_PER_COUNT,
    "ALPHAS": ALPHAS,
    "V2_RUN_DIR_OVERRIDE": V2_RUN_DIR_OVERRIDE or "<auto>",
    "V3_REPORT_DIR_OVERRIDE": V3_REPORT_DIR_OVERRIDE or "<auto>",
}]))
display(Markdown("**Estimated work units**"))
display(pd.DataFrame([estimated]).T.rename(columns={0: "estimated_count"}))
        """
    ),
    md("## 3. v2 Vocabulary, Data, Rendering, and Anchors"),
    code(
        r"""
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<Ans>", "<Think/>", "</Think>"]
NOISE_TOKENS = [f"<N{i}>" for i in range(64)]
MARKER_TOKENS = [f"<{chr(ord('A') + i)}>" for i in range(10)]
NUMBER_TOKENS = [f"<{i}>" for i in range(1, 11)]
COUNT_BINS = {"low": {1, 2, 3}, "mid": {4, 5, 6}, "high": {7, 8, 9, 10}}


def count_bin(count: int) -> str:
    for name, values in COUNT_BINS.items():
        if int(count) in values:
            return name
    return "outside"


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def build(cls) -> "Vocab":
        tokens = SPECIAL_TOKENS + NOISE_TOKENS + MARKER_TOKENS + NUMBER_TOKENS
        return cls({tok: i for i, tok in enumerate(tokens)}, tokens)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        obj = json.loads(path.read_text(encoding="utf-8"))
        token_to_id = obj.get("token_to_id") or obj.get("stoi")
        id_to_token = obj.get("id_to_token") or obj.get("itos")
        if token_to_id is None or id_to_token is None:
            raise ValueError(f"Unrecognized vocab format: {path}")
        if isinstance(id_to_token, dict):
            max_id = max(int(k) for k in id_to_token)
            id_to_token = [id_to_token[str(i)] for i in range(max_id + 1)]
        token_to_id = {str(k): int(v) for k, v in token_to_id.items()}
        return cls(token_to_id, list(id_to_token))

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[tok] for tok in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[int(i)] for i in ids]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def numeric_ids(self) -> list[int]:
        return [self.token_to_id[tok] for tok in NUMBER_TOKENS]

    def count_to_token(self, count: int) -> str:
        return f"<{int(count)}>"

    def token_to_count(self, token: str) -> int | None:
        if token.startswith("<") and token.endswith(">"):
            inner = token[1:-1]
            if inner.isdigit() and 1 <= int(inner) <= 10:
                return int(inner)
        return None


@dataclass
class BaseExample:
    seq_tokens: list[str]
    count: int
    needle_positions: list[int]
    needle_markers: list[str]
    example_id: str = ""


def validate_base_example(ex: BaseExample, seq_len: int) -> None:
    assert len(ex.seq_tokens) == seq_len
    assert 0 <= ex.count <= 10
    assert ex.count == len(ex.needle_positions) == len(ex.needle_markers)
    assert ex.needle_positions == sorted(ex.needle_positions)
    for pos, marker in zip(ex.needle_positions, ex.needle_markers):
        assert ex.seq_tokens[pos] == marker


def sample_base_example(seq_len: int, rng: random.Random, *, count: int, example_id: str) -> BaseExample:
    positions = sorted(rng.sample(range(seq_len), int(count)))
    markers = [rng.choice(MARKER_TOKENS) for _ in range(int(count))]
    seq = [rng.choice(NOISE_TOKENS) for _ in range(seq_len)]
    for pos, marker in zip(positions, markers):
        seq[pos] = marker
    ex = BaseExample(seq, int(count), positions, markers, example_id=example_id)
    validate_base_example(ex, seq_len)
    return ex


def balanced_examples(seq_len: int, examples_per_count: int, seed: int) -> list[BaseExample]:
    rng = random.Random(seed)
    examples = []
    for count in range(1, 11):
        for i in range(examples_per_count):
            examples.append(sample_base_example(seq_len, rng, count=count, example_id=f"c{count:02d}_{i:04d}"))
    rng.shuffle(examples)
    return examples


def trace_tokens_from_markers(markers: list[str], vocab: Vocab) -> list[str]:
    out = []
    for k, marker in enumerate(markers, start=1):
        out.extend([vocab.count_to_token(k), marker])
    return out


def render_thinking(
    ex: BaseExample,
    vocab: Vocab,
    *,
    trace_override: list[str] | None = None,
    answer_count: int | None = None,
) -> dict[str, Any]:
    trace = trace_override if trace_override is not None else trace_tokens_from_markers(ex.needle_markers, vocab)
    answer_count = ex.count if answer_count is None else int(answer_count)
    n_trace_pairs = len(trace) // 2
    index_positions = []
    marker_positions = []
    pre_index_positions = []
    think_start_pos = 1 + len(ex.seq_tokens)
    pos = think_start_pos + 1
    prev_marker_pos = think_start_pos
    for _ in range(n_trace_pairs):
        pre_index_positions.append(prev_marker_pos)
        index_positions.append(pos)
        marker_positions.append(pos + 1)
        prev_marker_pos = pos + 1
        pos += 2
    tokens = ["<BOS>"] + ex.seq_tokens + ["<Think/>"] + trace + ["</Think>", "<Ans>", vocab.count_to_token(answer_count), "<EOS>"]
    think_end_pos = think_start_pos + 1 + len(trace)
    ans_pos = think_end_pos + 1
    final_answer_pos = ans_pos + 1
    anchors = {
        "prompt_start": 1,
        "prompt_end_exclusive": 1 + len(ex.seq_tokens),
        "prompt_needle_positions": [1 + p for p in ex.needle_positions],
        "think_start": think_start_pos,
        "trace_positions": list(range(think_start_pos + 1, think_end_pos)),
        "index_positions": index_positions,
        "marker_positions": marker_positions,
        "pre_index_positions": pre_index_positions,
        "think_end": think_end_pos,
        "ans_token": ans_pos,
        "final_answer_pos": final_answer_pos,
    }
    return {
        "tokens": tokens,
        "input_ids": vocab.encode(tokens),
        "anchors": anchors,
        "example_id": ex.example_id,
        "gold_count": ex.count,
        "answer_count": answer_count,
        "trace_tokens": trace,
    }


def thinking_prefix(ex: BaseExample, vocab: Vocab) -> list[str]:
    return ["<BOS>"] + ex.seq_tokens + ["<Think/>"]


def positions_for_scope(rendered: dict[str, Any], scope: str, prefix_len: int | None = None) -> list[int]:
    a = rendered["anchors"]
    if scope == "all_positions":
        positions = list(range(prefix_len if prefix_len is not None else len(rendered["input_ids"])))
    elif scope == "trace_positions":
        positions = a["trace_positions"]
    elif scope == "index_token_all":
        positions = a["index_positions"]
    elif scope == "index_token_last":
        positions = a["index_positions"][-1:] if a["index_positions"] else []
    elif scope == "marker_token_all":
        positions = a["marker_positions"]
    elif scope == "marker_token_last":
        positions = a["marker_positions"][-1:] if a["marker_positions"] else []
    elif scope == "pre_index_all":
        positions = a["pre_index_positions"]
    elif scope == "pre_index_last":
        positions = a["pre_index_positions"][-1:] if a["pre_index_positions"] else []
    elif scope == "think_end":
        positions = [a["think_end"]]
    elif scope == "ans_token":
        positions = [a["ans_token"]]
    else:
        raise ValueError(f"Unknown position scope: {scope}")
    if prefix_len is not None:
        positions = [p for p in positions if 0 <= p < prefix_len]
    return positions


def token_role(position: int, rendered: dict[str, Any]) -> str:
    a = rendered["anchors"]
    if position == 0:
        return "bos"
    if position in a["prompt_needle_positions"]:
        return f"prompt_needle_{a['prompt_needle_positions'].index(position) + 1}"
    if a["prompt_start"] <= position < a["prompt_end_exclusive"]:
        return "prompt_noise"
    if position == a["think_start"]:
        return "think_start"
    if position in a["index_positions"]:
        return f"trace_index_{a['index_positions'].index(position) + 1}"
    if position in a["marker_positions"]:
        return f"trace_marker_{a['marker_positions'].index(position) + 1}"
    if position == a["think_end"]:
        return "think_end"
    if position == a["ans_token"]:
        return "ans_token"
    if position == a["final_answer_pos"]:
        return "final_answer"
    return "other"
        """
    ),
    md("## 4. Locate v2 Checkpoint and Load Model"),
    code(
        r"""
def maybe_mount_drive_for_v2_inputs() -> None:
    if not IN_COLAB or not AUTO_MOUNT_DRIVE_FOR_V2_INPUTS:
        return
    if Path("/content/drive/MyDrive").exists():
        return
    try:
        from google.colab import drive

        print("Mounting Google Drive to search for saved v2 checkpoints...")
        drive.mount("/content/drive")
    except Exception as exc:
        print(f"Google Drive mount skipped or failed: {exc}")


def candidate_v2_run_dirs() -> list[Path]:
    maybe_mount_drive_for_v2_inputs()
    out: list[Path] = []
    if V2_RUN_DIR_OVERRIDE:
        out.append(Path(V2_RUN_DIR_OVERRIDE))
    out.extend([
        Path("runs/v2_marker_trace_seed1234_main"),
        Path("runs/v2_marker_trace_seed1234_debug"),
    ])
    out.extend(Path("runs").glob("v2_marker_trace_seed*_main"))
    out.extend(Path("runs").glob("v2_marker_trace_seed*_debug"))
    out.extend(Path("colab_results").glob("v2_marker_trace_*_seed*/run"))
    out.extend(Path("colab_results").glob("v2_marker_trace_main_seed*/run"))
    drive_root = Path("/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/Synthetic_CoT_NiaH_Count/colab_results")
    if drive_root.exists():
        out.extend(drive_root.glob("v2_marker_trace_*_seed*/run"))
    seen = set()
    deduped = []
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def resolve_v2_run_dir() -> Path:
    candidates = candidate_v2_run_dirs()
    valid = [p for p in candidates if (p / "checkpoints" / "final" / "thinking" / "config.json").exists()]
    if not valid:
        display(pd.DataFrame({"candidate": [str(p) for p in candidates], "exists": [p.exists() for p in candidates]}))
        raise FileNotFoundError(
            "Could not find a v2 final thinking checkpoint. Run Trace_Count_v2_Colab first, "
            "copy a v2 bundle into colab_results, or set V2_RUN_DIR_OVERRIDE to a directory containing checkpoints/final/thinking."
        )
    return valid[0] if V2_RUN_DIR_OVERRIDE else sorted(valid, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def load_gpt2_eager(path: Path) -> GPT2LMHeadModel:
    try:
        model = GPT2LMHeadModel.from_pretrained(path, attn_implementation="eager")
    except TypeError:
        model = GPT2LMHeadModel.from_pretrained(path)
        model.config._attn_implementation = "eager"
    model.to(DEVICE)
    model.eval()
    return model


V2_RUN_DIR = resolve_v2_run_dir()
CHECKPOINT_DIR = V2_RUN_DIR / "checkpoints" / "final"
THINKING_MODEL_DIR = CHECKPOINT_DIR / "thinking"
VOCAB_PATH = CHECKPOINT_DIR / "vocab.json"
RUN_CONFIG_PATH = V2_RUN_DIR / "config.json"

vocab = Vocab.load(VOCAB_PATH) if VOCAB_PATH.exists() else Vocab.build()
cfg = {"seq_len": 256, "max_count": 10, "seed": RANDOM_SEED, "device": DEVICE}
if RUN_CONFIG_PATH.exists():
    cfg.update(json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8")))
cfg["device"] = DEVICE

CAUSAL_DIR = V2_RUN_DIR / "v3_2_causal"
TABLE_DIR = CAUSAL_DIR / "tables"
FIG_DIR = CAUSAL_DIR / "figures"
for path in [CAUSAL_DIR, TABLE_DIR, FIG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

thinking_model = load_gpt2_eager(THINKING_MODEL_DIR)
N_LAYER = int(thinking_model.config.n_layer)
N_HEAD = int(thinking_model.config.n_head)
N_EMBD = int(thinking_model.config.n_embd)
HEAD_DIM = N_EMBD // N_HEAD
assert N_EMBD == N_HEAD * HEAD_DIM

display(Markdown(f"**v2 run dir:** `{V2_RUN_DIR}`"))
display(Markdown(f"**thinking checkpoint:** `{THINKING_MODEL_DIR}`"))
display(Markdown(f"**model:** `{N_LAYER}` layers x `{N_HEAD}` heads, learned absolute positions, n_embd={N_EMBD}"))
display(Markdown(f"**v3.2 output dir:** `{CAUSAL_DIR}`"))

sample = balanced_examples(int(cfg["seq_len"]), 1, RANDOM_SEED)[0]
display(pd.DataFrame({
    "token": render_thinking(sample, vocab)["tokens"][:8] + ["..."] + render_thinking(sample, vocab)["tokens"][-12:],
}))
        """
    ),
    md("## 5. Candidate Heads from v3 Report"),
    code(
        r"""
DEFAULT_RETRIEVAL_HEADS = [(3, 3), (3, 1), (4, 1), (3, 2)]
DEFAULT_PLUS_ONE_HEADS = [(2, 3), (4, 2), (1, 0)]
DEFAULT_CONTROL_HEADS = [(2, 0), (1, 2)]


def candidate_v3_report_dirs() -> list[Path]:
    out = []
    if V3_REPORT_DIR_OVERRIDE:
        out.append(Path(V3_REPORT_DIR_OVERRIDE))
    out.extend(Path("colab_results").glob("v3_v2_attention_deepdive_seed*"))
    drive_root = Path("/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/Synthetic_CoT_NiaH_Count/colab_results")
    if drive_root.exists():
        out.extend(drive_root.glob("v3_v2_attention_deepdive_seed*"))
    return sorted([p for p in out if p.exists()], key=lambda p: p.stat().st_mtime, reverse=True)


def load_last_index_head_summary() -> pd.DataFrame:
    for root in candidate_v3_report_dirs():
        candidates = [
            root / "analysis" / "tables" / "last_index_head_summary.csv",
            root / "tables" / "last_index_head_summary.csv",
            root / "last_index_head_summary.csv",
        ]
        for path in candidates:
            if path.exists() and path.stat().st_size > 0:
                df = pd.read_csv(path)
                display(Markdown(f"Loaded v3 head summary from `{path}`"))
                return df
    return pd.DataFrame()


def valid_head(head: tuple[int, int]) -> bool:
    layer, h = head
    return 1 <= int(layer) <= N_LAYER and 0 <= int(h) < N_HEAD


def select_candidate_heads(summary: pd.DataFrame) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    if summary.empty:
        return (
            [h for h in DEFAULT_RETRIEVAL_HEADS if valid_head(h)],
            [h for h in DEFAULT_PLUS_ONE_HEADS if valid_head(h)],
            [h for h in DEFAULT_CONTROL_HEADS if valid_head(h)],
        )
    df = summary.copy()
    for col in ["correct_prompt_needle_mass", "retrieval_score", "plus_one_score"]:
        if col not in df.columns:
            df[col] = 0.0
    df["retrieval_rank_score"] = df[["correct_prompt_needle_mass", "retrieval_score"]].max(axis=1)
    retrieval = [(int(r.layer), int(r.head)) for r in df.sort_values("retrieval_rank_score", ascending=False).itertuples() if valid_head((int(r.layer), int(r.head)))]
    plus = [(int(r.layer), int(r.head)) for r in df.sort_values("plus_one_score", ascending=False).itertuples() if valid_head((int(r.layer), int(r.head)))]
    df["control_score"] = df["retrieval_rank_score"].abs() + df["plus_one_score"].abs()
    controls = [(int(r.layer), int(r.head)) for r in df.sort_values("control_score", ascending=True).itertuples() if valid_head((int(r.layer), int(r.head)))]
    def dedupe(xs: list[tuple[int, int]], k: int, avoid: set[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
        avoid = avoid or set()
        out = []
        for x in xs:
            if x not in out and x not in avoid:
                out.append(x)
            if len(out) >= k:
                break
        return out
    retrieval = dedupe(retrieval or DEFAULT_RETRIEVAL_HEADS, 4)
    plus = dedupe(plus or DEFAULT_PLUS_ONE_HEADS, 3, avoid=set(retrieval[:2]))
    controls = dedupe(controls or DEFAULT_CONTROL_HEADS, 2, avoid=set(retrieval + plus))
    return retrieval, plus, controls


last_index_head_summary = load_last_index_head_summary()
RETRIEVAL_HEADS, PLUS_ONE_HEADS, CONTROL_HEADS = select_candidate_heads(last_index_head_summary)
HEAD_GROUPS = {
    "baseline_no_ablation": [],
    "retrieval_L3H3_or_top1": RETRIEVAL_HEADS[:1],
    "retrieval_L3H1_or_top2": RETRIEVAL_HEADS[1:2],
    "retrieval_top2": RETRIEVAL_HEADS[:2],
    "retrieval_top4": RETRIEVAL_HEADS,
    "plus_one_top1": PLUS_ONE_HEADS[:1],
    "plus_one_top3": PLUS_ONE_HEADS,
    "retrieval_plus_one_top": RETRIEVAL_HEADS[:2] + PLUS_ONE_HEADS[:2],
    "low_score_controls": CONTROL_HEADS,
}

display(Markdown("**Candidate heads.** `LxHy` means layer x is 1-based and head y is 0-based."))
display(pd.DataFrame([
    {"group": "retrieval", "heads": ", ".join(f"L{l}H{h}" for l, h in RETRIEVAL_HEADS)},
    {"group": "plus_one/local", "heads": ", ".join(f"L{l}H{h}" for l, h in PLUS_ONE_HEADS)},
    {"group": "control", "heads": ", ".join(f"L{l}H{h}" for l, h in CONTROL_HEADS)},
]))
if not last_index_head_summary.empty:
    display(last_index_head_summary.sort_values(["correct_prompt_needle_mass", "plus_one_score"], ascending=False).head(12))
        """
    ),
    md("## 6. Intervention and Evaluation Utilities"),
    code(
        r"""
def as_input(ids: list[int]) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long, device=DEVICE)


def count_logits_from_logits(logits_1d: torch.Tensor, vocab: Vocab) -> torch.Tensor:
    idx = torch.tensor(vocab.numeric_ids, dtype=torch.long, device=logits_1d.device)
    return logits_1d.index_select(0, idx)


def count_distribution_metrics(logits_1d: torch.Tensor, target_count: int, vocab: Vocab) -> dict[str, float | int]:
    c_logits = count_logits_from_logits(logits_1d, vocab)
    probs = F.softmax(c_logits, dim=-1)
    pred_idx = int(torch.argmax(c_logits).item())
    pred_count = pred_idx + 1
    target_idx = int(target_count) - 1
    target_logit = float(c_logits[target_idx].detach().cpu())
    competitor = torch.cat([c_logits[:target_idx], c_logits[target_idx + 1 :]])
    logit_margin = float((c_logits[target_idx] - competitor.max()).detach().cpu()) if competitor.numel() else float("nan")
    answer_ce = float(F.cross_entropy(c_logits[None, :], torch.tensor([target_idx], device=c_logits.device)).detach().cpu())
    entropy = float((-(probs * torch.log(probs + 1e-12))).sum().detach().cpu())
    expectation = float((probs * torch.arange(1, 11, device=probs.device, dtype=probs.dtype)).sum().detach().cpu())
    return {
        "pred_count": pred_count,
        "gold_logit": target_logit,
        "gold_logit_margin": logit_margin,
        "answer_ce": answer_ce,
        "count_distribution_entropy": entropy,
        "count_expectation": expectation,
        "answer_accuracy": float(pred_count == int(target_count)),
        "count_shift": float(pred_count - int(target_count)),
    }


@contextmanager
def head_output_intervention(model: GPT2LMHeadModel, specs: list[dict[str, Any]] | None):
    specs = specs or []
    if not specs:
        yield
        return
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for spec in specs:
        layer = int(spec["layer"])
        by_layer.setdefault(layer, []).append(spec)
    handles = []
    try:
        for layer, layer_specs in by_layer.items():
            attn = model.transformer.h[layer - 1].attn

            def make_hook(active_specs: list[dict[str, Any]]):
                def pre_hook(module, inputs):
                    x = inputs[0]
                    bsz, seq_len, width = x.shape
                    view = x.view(bsz, seq_len, N_HEAD, HEAD_DIM).clone()
                    for spec in active_specs:
                        heads = [int(h) for h in spec["heads"]]
                        positions = [int(p) for p in spec.get("positions", []) if 0 <= int(p) < seq_len]
                        if not positions:
                            continue
                        mode = spec.get("mode", "scale")
                        pos = torch.tensor(positions, dtype=torch.long, device=view.device)
                        for head in heads:
                            if mode == "zero":
                                view[:, pos, head, :] = 0.0
                            elif mode == "mean":
                                mean_vec = view[:, :, head, :].mean(dim=1, keepdim=True)
                                view[:, pos, head, :] = mean_vec.expand(-1, len(positions), -1)
                            elif mode == "scale":
                                view[:, pos, head, :] = view[:, pos, head, :] * float(spec.get("alpha", 1.0))
                            elif mode == "replace":
                                replacement = spec["replacement"][(int(spec["layer"]), head)]
                                repl = replacement.to(device=view.device, dtype=view.dtype).view(1, 1, HEAD_DIM)
                                view[:, pos, head, :] = repl.expand(bsz, len(positions), HEAD_DIM)
                            else:
                                raise ValueError(f"Unknown intervention mode: {mode}")
                    return (view.reshape_as(x),) + tuple(inputs[1:])
                return pre_hook

            handles.append(attn.c_proj.register_forward_pre_hook(make_hook(layer_specs)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def residual_replacement_intervention(model: GPT2LMHeadModel, specs: list[dict[str, Any]] | None):
    specs = specs or []
    if not specs:
        yield
        return
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for spec in specs:
        by_layer.setdefault(int(spec["layer"]), []).append(spec)
    handles = []
    try:
        for layer, layer_specs in by_layer.items():
            block = model.transformer.h[layer - 1]

            def make_hook(active_specs: list[dict[str, Any]]):
                def hook(module, inputs, output):
                    if isinstance(output, tuple):
                        hidden = output[0].clone()
                        rest = output[1:]
                    else:
                        hidden = output.clone()
                        rest = None
                    seq_len = hidden.shape[1]
                    for spec in active_specs:
                        positions = [int(p) for p in spec.get("positions", []) if 0 <= int(p) < seq_len]
                        if not positions:
                            continue
                        pos = torch.tensor(positions, dtype=torch.long, device=hidden.device)
                        repl = spec["replacement"].to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
                        hidden[:, pos, :] = repl.expand(hidden.size(0), len(positions), hidden.size(-1))
                    if rest is None:
                        return hidden
                    return (hidden,) + rest
                return hook

            handles.append(block.register_forward_hook(make_hook(layer_specs)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def build_head_specs(
    heads: list[tuple[int, int]],
    rendered: dict[str, Any],
    *,
    mode: str,
    position_scope: str,
    prefix_len: int,
    alpha: float = 1.0,
    replacement: dict[tuple[int, int], torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    positions = positions_for_scope(rendered, position_scope, prefix_len=prefix_len)
    return [
        {
            "layer": layer,
            "heads": sorted(set(heads_)),
            "positions": positions,
            "mode": mode,
            "alpha": alpha,
            "replacement": replacement or {},
        }
        for layer, heads_ in by_layer.items()
    ]


@torch.no_grad()
def score_final_answer(
    model: GPT2LMHeadModel,
    ex: BaseExample,
    vocab: Vocab,
    *,
    target_count: int | None = None,
    rendered: dict[str, Any] | None = None,
    head_specs: list[dict[str, Any]] | None = None,
    residual_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rendered = rendered or render_thinking(ex, vocab)
    target_count = int(ex.count if target_count is None else target_count)
    ans_pos = rendered["anchors"]["ans_token"]
    ids = rendered["input_ids"][: ans_pos + 1]
    input_ids = as_input(ids)
    with head_output_intervention(model, head_specs), residual_replacement_intervention(model, residual_specs):
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    logits = out.logits[0, -1, :]
    metrics = count_distribution_metrics(logits, target_count, vocab)
    metrics.update({
        "example_id": ex.example_id,
        "gold_count": target_count,
        "count": ex.count,
        "count_bin": count_bin(target_count),
    })
    return metrics


@torch.no_grad()
def generate_thinking_one(
    model: GPT2LMHeadModel,
    ex: BaseExample,
    vocab: Vocab,
    *,
    heads: list[tuple[int, int]] | None = None,
    alpha: float = 0.0,
    mode: str = "scale",
    max_new: int | None = None,
) -> dict[str, Any]:
    max_new = max_new or (2 * int(cfg.get("max_count", 10)) + 4)
    ids = as_input(vocab.encode(thinking_prefix(ex, vocab)))
    prefix_len = ids.shape[1]
    rendered_prefix = render_thinking(ex, vocab)
    specs = build_head_specs(heads or [], rendered_prefix, mode=mode, position_scope="all_positions", prefix_len=10_000, alpha=alpha)
    with head_output_intervention(model, specs):
        for _ in range(max_new):
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            next_id = int(torch.argmax(out.logits[0, -1, :]).item())
            ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)
            if next_id == vocab.eos_id:
                break
    gen = ids[0, prefix_len:].detach().cpu().tolist()
    toks = [tok for tok in vocab.decode(gen) if tok != "<PAD>"]
    parsed = parse_thinking_generation(toks, ex, vocab)
    parsed.update({"example_id": ex.example_id, "count": ex.count, "count_bin": count_bin(ex.count)})
    return parsed


def parse_thinking_generation(generated_tokens: list[str], ex: BaseExample, vocab: Vocab) -> dict[str, Any]:
    pred_count = None
    invalid = True
    if "<Ans>" in generated_tokens:
        ans_idx = generated_tokens.index("<Ans>")
        if ans_idx + 1 < len(generated_tokens):
            pred_count = vocab.token_to_count(generated_tokens[ans_idx + 1])
            invalid = pred_count is None
    trace_end = generated_tokens.index("</Think>") if "</Think>" in generated_tokens else len(generated_tokens)
    generated_trace = generated_tokens[:trace_end]
    expected_trace = trace_tokens_from_markers(ex.needle_markers, vocab)
    expected_markers = [tok for tok in expected_trace if tok in MARKER_TOKENS]
    generated_markers = [tok for tok in generated_trace if tok in MARKER_TOKENS]
    expected_counts = pd.Series(expected_markers).value_counts()
    generated_counts = pd.Series(generated_markers).value_counts()
    if expected_markers or generated_markers:
        overlap = float(pd.concat([expected_counts, generated_counts], axis=1).fillna(0).min(axis=1).sum())
    else:
        overlap = 0.0
    marker_recall = 1.0 if not expected_markers else overlap / len(expected_markers)
    expected_indices = expected_trace[0::2]
    correct_indices = 0
    for i, tok in enumerate(expected_indices):
        pos = 2 * i
        correct_indices += int(pos < len(generated_trace) and generated_trace[pos] == tok)
    trace_index_accuracy = correct_indices / max(len(expected_indices), 1)
    return {
        "pred_count": pred_count,
        "invalid": invalid,
        "answer_accuracy": float(pred_count == ex.count),
        "trace_exact_match": bool(generated_trace == expected_trace),
        "trace_marker_recall": marker_recall,
        "trace_index_accuracy": trace_index_accuracy,
        "generated_text": " ".join(generated_tokens),
    }


def summarize_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = {
        "answer_accuracy": "mean",
        "gold_logit_margin": "mean",
        "answer_ce": "mean",
        "count_shift": "mean",
    }
    optional = ["invalid", "trace_exact_match", "trace_marker_recall", "trace_index_accuracy"]
    for col in optional:
        if col in df.columns:
            agg[col] = "mean"
    out = df.groupby(group_cols, as_index=False).agg(agg)
    out["n_examples"] = df.groupby(group_cols).size().to_numpy()
    return out


def savefig(name: str, *, show: bool = True) -> Path:
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=180)
    if show:
        plt.show()
    else:
        plt.close()
    return path
        """
    ),
    md(
        r"""
## 7. Causal Test A/B: Necessity and Dose Response

**图和表怎么读：**

- teacher-forced final readout：给模型 gold trace，到 `<Ans>` 后只看最终 count logits。这个更适合看 head 对最终读出的 logit margin 是否有影响。
- autoregressive generation：从 `<Think/>` 后让模型自己生成 trace 和答案。这个更适合看 trace exactness / malformed trace。
- `gold_logit_margin`：正确 count token 的 logit 减去最强错误 count token 的 logit。accuracy 饱和时，margin 比 accuracy 更敏感。
- `position_scope`：head intervention 施加在哪些 token 位置。`index_token_last` 就是最后一个 trace 数字 `<n>`；`trace_positions` 是整个 `<Think/>...</Think>` 内部。
        """
    ),
    code(
        r"""
BASE_EXAMPLES = balanced_examples(int(cfg["seq_len"]), EXAMPLES_PER_COUNT, RANDOM_SEED + 3200)


def run_necessity() -> pd.DataFrame:
    rows = []
    for condition, heads in tqdm(HEAD_GROUPS.items(), desc="necessity conditions"):
        for scope in POSITION_SCOPES:
            for ex in BASE_EXAMPLES:
                rendered = render_thinking(ex, vocab)
                prefix_len = rendered["anchors"]["ans_token"] + 1
                specs = build_head_specs(heads, rendered, mode="zero", position_scope=scope, prefix_len=prefix_len)
                metrics = score_final_answer(thinking_model, ex, vocab, rendered=rendered, head_specs=specs)
                rows.append({
                    "experiment_name": "head_necessity",
                    "condition": condition,
                    "intervention_type": "zero_ablation",
                    "heads": json.dumps(heads),
                    "position_scope": scope,
                    "eval_mode": "teacher_forced_final_readout",
                    **metrics,
                })
        if RUN_AR_GENERATION:
            ar_examples = BASE_EXAMPLES[:MAX_AR_EXAMPLES]
            for ex in tqdm(ar_examples, desc=f"AR {condition}", leave=False):
                parsed = generate_thinking_one(thinking_model, ex, vocab, heads=heads, alpha=0.0, mode="scale")
                rows.append({
                    "experiment_name": "head_necessity",
                    "condition": condition,
                    "intervention_type": "zero_ablation",
                    "heads": json.dumps(heads),
                    "position_scope": "all_positions",
                    "eval_mode": "autoregressive_generation",
                    "gold_count": ex.count,
                    "gold_logit_margin": np.nan,
                    "answer_ce": np.nan,
                    "count_shift": np.nan if parsed["pred_count"] is None else parsed["pred_count"] - ex.count,
                    **parsed,
                })
    return pd.DataFrame(rows)


def run_dose_response() -> pd.DataFrame:
    groups = {
        "L3H3_or_top_retrieval": RETRIEVAL_HEADS[:1],
        "L3H1_or_second_retrieval": RETRIEVAL_HEADS[1:2],
        "retrieval_top2": RETRIEVAL_HEADS[:2],
        "retrieval_top4": RETRIEVAL_HEADS,
        "L2H3_or_top_plus_one": PLUS_ONE_HEADS[:1],
        "plus_one_top3": PLUS_ONE_HEADS,
        "control_top2": CONTROL_HEADS,
    }
    rows = []
    for group_name, heads in tqdm(groups.items(), desc="dose groups"):
        for alpha in ALPHAS:
            for ex in BASE_EXAMPLES:
                rendered = render_thinking(ex, vocab)
                prefix_len = rendered["anchors"]["ans_token"] + 1
                specs = build_head_specs(heads, rendered, mode="scale", alpha=float(alpha), position_scope="index_token_last", prefix_len=prefix_len)
                metrics = score_final_answer(thinking_model, ex, vocab, rendered=rendered, head_specs=specs)
                rows.append({
                    "experiment_name": "head_dose_response",
                    "condition": group_name,
                    "intervention_type": "scale",
                    "heads": json.dumps(heads),
                    "position_scope": "index_token_last",
                    "alpha": float(alpha),
                    "eval_mode": "teacher_forced_final_readout",
                    **metrics,
                })
    return pd.DataFrame(rows)


necessity_path = TABLE_DIR / "head_necessity_results.csv"
dose_path = TABLE_DIR / "head_dose_response.csv"

if RUN_NECESSITY:
    if SKIP_COMPLETED and necessity_path.exists() and necessity_path.stat().st_size > 0:
        necessity_df = pd.read_csv(necessity_path)
    else:
        necessity_df = run_necessity()
        necessity_df.to_csv(necessity_path, index=False)
else:
    necessity_df = pd.read_csv(necessity_path)

if RUN_DOSE_RESPONSE:
    if SKIP_COMPLETED and dose_path.exists() and dose_path.stat().st_size > 0:
        dose_df = pd.read_csv(dose_path)
    else:
        dose_df = run_dose_response()
        dose_df.to_csv(dose_path, index=False)
else:
    dose_df = pd.read_csv(dose_path)

display(Markdown(f"Saved necessity table: `{necessity_path}`"))
display(Markdown(f"Saved dose-response table: `{dose_path}`"))
display(summarize_rows(necessity_df[necessity_df["eval_mode"].eq("teacher_forced_final_readout")], ["condition", "position_scope"]).head(20))
display(dose_df.groupby(["condition", "alpha"], as_index=False)["gold_logit_margin"].mean().head(20))
        """
    ),
    md("### Figures for Necessity and Dose Response"),
    code(
        r"""
tf = necessity_df[necessity_df["eval_mode"].eq("teacher_forced_final_readout")].copy()
tf_sum = tf.groupby(["condition", "position_scope"], as_index=False)["gold_logit_margin"].mean()
plt.figure(figsize=(12, 5.2))
for scope in POSITION_SCOPES:
    sub = tf_sum[tf_sum["position_scope"].eq(scope)]
    plt.plot(sub["condition"], sub["gold_logit_margin"], marker="o", label=scope)
plt.axhline(0, color="black", linewidth=1, alpha=0.5)
plt.xticks(rotation=35, ha="right")
plt.ylabel("mean gold-logit margin")
plt.xlabel("head-ablation condition")
plt.title("Head necessity: final-answer margin under zero ablation")
plt.legend(title="position scope", fontsize=8)
savefig("head_necessity_answer_margin.png")

if "trace_exact_match" in necessity_df.columns:
    ar = necessity_df[necessity_df["eval_mode"].eq("autoregressive_generation")].copy()
    if not ar.empty:
        ar_sum = ar.groupby("condition", as_index=False)[["trace_exact_match", "trace_marker_recall", "trace_index_accuracy", "answer_accuracy"]].mean()
        x = np.arange(len(ar_sum))
        width = 0.2
        plt.figure(figsize=(12, 4.8))
        for i, metric in enumerate(["trace_exact_match", "trace_marker_recall", "trace_index_accuracy", "answer_accuracy"]):
            plt.bar(x + (i - 1.5) * width, ar_sum[metric], width=width, label=metric)
        plt.xticks(x, ar_sum["condition"], rotation=35, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel("rate")
        plt.xlabel("all-position head-ablation condition")
        plt.title("Autoregressive trace behavior under head ablation")
        plt.legend(fontsize=8)
        savefig("head_necessity_trace_exact.png")

dose_sum = dose_df.groupby(["condition", "alpha"], as_index=False)["gold_logit_margin"].mean()
plt.figure(figsize=(10, 5))
for name, sub in dose_sum.groupby("condition"):
    plt.plot(sub["alpha"], sub["gold_logit_margin"], marker="o", label=name)
plt.axhline(0, color="black", linewidth=1, alpha=0.5)
plt.xlabel("head output scale alpha at final index token")
plt.ylabel("mean gold-logit margin")
plt.title("Dose response: does scaling a head/group move final count logits?")
plt.legend(fontsize=8, ncol=2)
savefig("head_dose_response_answer_margin.png")

display(Markdown(
    "**Interpretation guide.** If L3H3 has high attention but ablation/scale barely changes margin relative to controls, "
    "it is diagnostic or redundant. If retrieval_top4 changes margin/trace much more than controls, targeted retrieval is causally important as a group."
))
        """
    ),
    md(
        r"""
## 8. Causal Test C/D: Clean-to-Corrupt Activation and Path Patching

这里做 sufficiency test。对于一对 clean/corrupt prompt：

- clean 有 target count `n`；
- corrupt 通过删除/添加最后一个 needle 得到 `n-1` 或 `n+1`；
- base run 在 corrupt prompt 上跑；
- patch run 把 clean 的某个 head output 或 residual state 插到 corrupt 的同名位置；
- 如果 `clean_target - corrupt_target` 的 logit margin 被恢复，说明被 patch 的 activation 携带了可用的 clean count / final-needle 信息。

`normalized_recovery = (patched_margin - corrupt_base_margin) / (clean_base_margin - corrupt_base_margin)`。接近 1 表示 patch 充分恢复 clean 信息；接近 0 表示没有恢复；负数表示反方向。

重要修正：**最终答案太容易饱和**，只看 `<Ans>` 后的 count logits 可能得到一排 0。v3.2 额外加入一个更局部的 patching target：

- `marker_after_final_index`：在最后一个 trace index token `<n>` 之后，看模型是否更支持正确的最后 marker；
- `answer_after_ans`：在 `<Ans>` 之后看最终 count，只作为 readout 对照。

如果 targeted retrieval head 真在把第 `n` 个 prompt needle/marker 取回来，它应该更先影响 `marker_after_final_index`，而不一定能单头改变已经很稳的 final-answer readout。
        """
    ),
    code(
        r"""
def clone_example(ex: BaseExample, *, example_id: str | None = None) -> BaseExample:
    return BaseExample(list(ex.seq_tokens), int(ex.count), list(ex.needle_positions), list(ex.needle_markers), example_id or ex.example_id)


def replace_with_noise(seq: list[str], pos: int, rng: random.Random) -> None:
    tok = rng.choice(NOISE_TOKENS)
    while tok in MARKER_TOKENS:
        tok = rng.choice(NOISE_TOKENS)
    seq[pos] = tok


def remove_last_needle(ex: BaseExample, rng: random.Random) -> BaseExample | None:
    if ex.count <= 1:
        return None
    out = clone_example(ex, example_id=ex.example_id + "_minus_last")
    last_pos = out.needle_positions[-1]
    replace_with_noise(out.seq_tokens, last_pos, rng)
    out.needle_positions = out.needle_positions[:-1]
    out.needle_markers = out.needle_markers[:-1]
    out.count -= 1
    validate_base_example(out, len(out.seq_tokens))
    return out


def add_final_needle(ex: BaseExample, rng: random.Random) -> BaseExample | None:
    if ex.count >= 10:
        return None
    out = clone_example(ex, example_id=ex.example_id + "_plus_last")
    candidates = [p for p in range((out.needle_positions[-1] + 1) if out.needle_positions else 0, len(out.seq_tokens)) if p not in set(out.needle_positions)]
    if not candidates:
        candidates = [p for p in range(len(out.seq_tokens)) if p not in set(out.needle_positions)]
    if not candidates:
        return None
    pos = rng.choice(candidates)
    marker = rng.choice(MARKER_TOKENS)
    out.seq_tokens[pos] = marker
    out.needle_positions.append(pos)
    out.needle_markers.append(marker)
    order = np.argsort(out.needle_positions)
    out.needle_positions = [out.needle_positions[i] for i in order]
    out.needle_markers = [out.needle_markers[i] for i in order]
    out.count += 1
    validate_base_example(out, len(out.seq_tokens))
    return out


def build_patch_pairs(examples_per_count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pairs = []
    for count in range(1, 11):
        for i in range(examples_per_count):
            ex = sample_base_example(int(cfg["seq_len"]), rng, count=count, example_id=f"pair_c{count}_{i}")
            minus = remove_last_needle(ex, rng)
            if minus is not None:
                pairs.append({
                    "pair_id": f"remove_last_c{count}_{i}",
                    "pair_type": "remove_last_needle",
                    "clean": ex,
                    "corrupt": minus,
                    "clean_target_count": ex.count,
                    "corrupt_target_count": minus.count,
                })
            plus = add_final_needle(ex, rng)
            if plus is not None:
                pairs.append({
                    "pair_id": f"add_last_c{count}_{i}",
                    "pair_type": "add_final_needle",
                    "clean": ex,
                    "corrupt": plus,
                    "clean_target_count": ex.count,
                    "corrupt_target_count": plus.count,
                })
    return pairs


@torch.no_grad()
def cache_head_vectors(model: GPT2LMHeadModel, rendered: dict[str, Any], heads: list[tuple[int, int]], scope: str) -> dict[tuple[int, int], torch.Tensor]:
    ans_pos = rendered["anchors"]["ans_token"]
    ids = as_input(rendered["input_ids"][: ans_pos + 1])
    positions = positions_for_scope(rendered, scope, prefix_len=ids.shape[1])
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    captures: dict[tuple[int, int], torch.Tensor] = {}
    handles = []
    try:
        for layer, head_ids in by_layer.items():
            attn = model.transformer.h[layer - 1].attn

            def make_hook(layer_: int, head_ids_: list[int]):
                def pre_hook(module, inputs):
                    x = inputs[0]
                    view = x.view(x.shape[0], x.shape[1], N_HEAD, HEAD_DIM)
                    valid_pos = [p for p in positions if p < x.shape[1]]
                    if valid_pos:
                        pos_t = torch.tensor(valid_pos, dtype=torch.long, device=x.device)
                        for head in head_ids_:
                            captures[(layer_, head)] = view[0, pos_t, head, :].mean(dim=0).detach().cpu()
                    return None
                return pre_hook

            handles.append(attn.c_proj.register_forward_pre_hook(make_hook(layer, head_ids)))
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captures


@torch.no_grad()
def cache_residual_vector(model: GPT2LMHeadModel, rendered: dict[str, Any], layer: int, scope: str) -> torch.Tensor | None:
    ans_pos = rendered["anchors"]["ans_token"]
    ids = as_input(rendered["input_ids"][: ans_pos + 1])
    positions = positions_for_scope(rendered, scope, prefix_len=ids.shape[1])
    capture = {"value": None}
    block = model.transformer.h[layer - 1]

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        valid_pos = [p for p in positions if p < hidden.shape[1]]
        if valid_pos:
            pos_t = torch.tensor(valid_pos, dtype=torch.long, device=hidden.device)
            capture["value"] = hidden[0, pos_t, :].mean(dim=0).detach().cpu()
        return None

    handle = block.register_forward_hook(hook)
    try:
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        handle.remove()
    return capture["value"]


def pair_margin(model: GPT2LMHeadModel, ex: BaseExample, clean_target: int, corrupt_target: int, *, rendered=None, head_specs=None, residual_specs=None) -> dict[str, Any]:
    rendered = rendered or render_thinking(ex, vocab)
    ans_pos = rendered["anchors"]["ans_token"]
    ids = as_input(rendered["input_ids"][: ans_pos + 1])
    with head_output_intervention(model, head_specs), residual_replacement_intervention(model, residual_specs):
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    logits = count_logits_from_logits(out.logits[0, -1, :], vocab)
    clean_logit = float(logits[int(clean_target) - 1].detach().cpu())
    corrupt_logit = float(logits[int(corrupt_target) - 1].detach().cpu())
    pred_count = int(torch.argmax(logits).item()) + 1
    return {
        "clean_target_logit": clean_logit,
        "corrupt_target_logit": corrupt_logit,
        "clean_minus_corrupt_margin": clean_logit - corrupt_logit,
        "pred_count": pred_count,
    }


def run_activation_patching() -> pd.DataFrame:
    pairs = build_patch_pairs(PATCH_PAIRS_PER_COUNT, RANDOM_SEED + 9001)
    head_sites = [
        ("head_output", "retrieval_top1", RETRIEVAL_HEADS[:1], "index_token_last"),
        ("head_output", "retrieval_top2", RETRIEVAL_HEADS[:2], "index_token_last"),
        ("head_output", "retrieval_top4", RETRIEVAL_HEADS, "index_token_last"),
        ("head_output", "plus_one_top1_pre_index", PLUS_ONE_HEADS[:1], "pre_index_last"),
        ("head_output", "plus_one_top1_index", PLUS_ONE_HEADS[:1], "index_token_last"),
        ("head_output", "control_top2", CONTROL_HEADS, "index_token_last"),
    ]
    residual_sites = [("resid_after_block", f"resid_L{layer}_{scope}", [(layer, -1)], scope) for layer in range(1, N_LAYER + 1) for scope in ["index_token_last", "think_end", "ans_token"]]
    rows = []
    for pair in tqdm(pairs, desc="activation patch pairs"):
        clean = pair["clean"]
        corrupt = pair["corrupt"]
        clean_target = int(pair["clean_target_count"])
        corrupt_target = int(pair["corrupt_target_count"])
        clean_render = render_thinking(clean, vocab)
        corrupt_render = render_thinking(corrupt, vocab)
        clean_base = pair_margin(thinking_model, clean, clean_target, corrupt_target, rendered=clean_render)
        corrupt_base = pair_margin(thinking_model, corrupt, clean_target, corrupt_target, rendered=corrupt_render)
        denom = clean_base["clean_minus_corrupt_margin"] - corrupt_base["clean_minus_corrupt_margin"]
        for site_type, name, heads, scope in head_sites:
            if not heads:
                continue
            clean_cache = cache_head_vectors(thinking_model, clean_render, heads, scope)
            specs = build_head_specs(heads, corrupt_render, mode="replace", position_scope=scope, prefix_len=corrupt_render["anchors"]["ans_token"] + 1, replacement=clean_cache)
            patched = pair_margin(thinking_model, corrupt, clean_target, corrupt_target, rendered=corrupt_render, head_specs=specs)
            recovery = (patched["clean_minus_corrupt_margin"] - corrupt_base["clean_minus_corrupt_margin"]) / (denom + 1e-9)
            rows.append({
                "experiment_name": "activation_patching",
                "pair_id": pair["pair_id"],
                "pair_type": pair["pair_type"],
                "patch_direction": "clean_to_corrupt",
                "site_type": site_type,
                "intervention_name": name,
                "heads": json.dumps(heads),
                "position_scope": scope,
                "count_clean": clean_target,
                "count_corrupt": corrupt_target,
                "clean_base_margin": clean_base["clean_minus_corrupt_margin"],
                "corrupt_base_margin": corrupt_base["clean_minus_corrupt_margin"],
                "patched_margin": patched["clean_minus_corrupt_margin"],
                "margin_delta": patched["clean_minus_corrupt_margin"] - corrupt_base["clean_minus_corrupt_margin"],
                "normalized_recovery": recovery,
                "pred_count": patched["pred_count"],
                "count_shift": patched["pred_count"] - corrupt_base["pred_count"],
            })
            corrupt_cache = cache_head_vectors(thinking_model, corrupt_render, heads, scope)
            reverse_specs = build_head_specs(heads, clean_render, mode="replace", position_scope=scope, prefix_len=clean_render["anchors"]["ans_token"] + 1, replacement=corrupt_cache)
            reverse = pair_margin(thinking_model, clean, clean_target, corrupt_target, rendered=clean_render, head_specs=reverse_specs)
            reverse_recovery = (clean_base["clean_minus_corrupt_margin"] - reverse["clean_minus_corrupt_margin"]) / (denom + 1e-9)
            rows.append({
                "experiment_name": "activation_patching",
                "pair_id": pair["pair_id"],
                "pair_type": pair["pair_type"],
                "patch_direction": "corrupt_to_clean",
                "site_type": site_type,
                "intervention_name": name,
                "heads": json.dumps(heads),
                "position_scope": scope,
                "count_clean": clean_target,
                "count_corrupt": corrupt_target,
                "clean_base_margin": clean_base["clean_minus_corrupt_margin"],
                "corrupt_base_margin": corrupt_base["clean_minus_corrupt_margin"],
                "patched_margin": reverse["clean_minus_corrupt_margin"],
                "margin_delta": reverse["clean_minus_corrupt_margin"] - clean_base["clean_minus_corrupt_margin"],
                "normalized_recovery": reverse_recovery,
                "pred_count": reverse["pred_count"],
                "count_shift": reverse["pred_count"] - clean_base["pred_count"],
            })
        for site_type, name, layer_spec, scope in residual_sites:
            layer = int(layer_spec[0][0])
            donor = cache_residual_vector(thinking_model, clean_render, layer, scope)
            if donor is None:
                continue
            positions = positions_for_scope(corrupt_render, scope, prefix_len=corrupt_render["anchors"]["ans_token"] + 1)
            residual_specs = [{"layer": layer, "positions": positions, "replacement": donor}]
            patched = pair_margin(thinking_model, corrupt, clean_target, corrupt_target, rendered=corrupt_render, residual_specs=residual_specs)
            recovery = (patched["clean_minus_corrupt_margin"] - corrupt_base["clean_minus_corrupt_margin"]) / (denom + 1e-9)
            rows.append({
                "experiment_name": "activation_patching",
                "pair_id": pair["pair_id"],
                "pair_type": pair["pair_type"],
                "patch_direction": "clean_to_corrupt",
                "site_type": site_type,
                "intervention_name": name,
                "heads": "[]",
                "layer": layer,
                "position_scope": scope,
                "count_clean": clean_target,
                "count_corrupt": corrupt_target,
                "clean_base_margin": clean_base["clean_minus_corrupt_margin"],
                "corrupt_base_margin": corrupt_base["clean_minus_corrupt_margin"],
                "patched_margin": patched["clean_minus_corrupt_margin"],
                "margin_delta": patched["clean_minus_corrupt_margin"] - corrupt_base["clean_minus_corrupt_margin"],
                "normalized_recovery": recovery,
                "pred_count": patched["pred_count"],
                "count_shift": patched["pred_count"] - corrupt_base["pred_count"],
            })
            reverse_donor = cache_residual_vector(thinking_model, corrupt_render, layer, scope)
            if reverse_donor is None:
                continue
            reverse_positions = positions_for_scope(clean_render, scope, prefix_len=clean_render["anchors"]["ans_token"] + 1)
            reverse_residual_specs = [{"layer": layer, "positions": reverse_positions, "replacement": reverse_donor}]
            reverse = pair_margin(thinking_model, clean, clean_target, corrupt_target, rendered=clean_render, residual_specs=reverse_residual_specs)
            reverse_recovery = (clean_base["clean_minus_corrupt_margin"] - reverse["clean_minus_corrupt_margin"]) / (denom + 1e-9)
            rows.append({
                "experiment_name": "activation_patching",
                "pair_id": pair["pair_id"],
                "pair_type": pair["pair_type"],
                "patch_direction": "corrupt_to_clean",
                "site_type": site_type,
                "intervention_name": name,
                "heads": "[]",
                "layer": layer,
                "position_scope": scope,
                "count_clean": clean_target,
                "count_corrupt": corrupt_target,
                "clean_base_margin": clean_base["clean_minus_corrupt_margin"],
                "corrupt_base_margin": corrupt_base["clean_minus_corrupt_margin"],
                "patched_margin": reverse["clean_minus_corrupt_margin"],
                "margin_delta": reverse["clean_minus_corrupt_margin"] - clean_base["clean_minus_corrupt_margin"],
                "normalized_recovery": reverse_recovery,
                "pred_count": reverse["pred_count"],
                "count_shift": reverse["pred_count"] - clean_base["pred_count"],
            })
    return pd.DataFrame(rows)


def token_margin_from_logits(logits_1d: torch.Tensor, target_id: int, candidate_ids: list[int]) -> dict[str, Any]:
    cand = torch.tensor(candidate_ids, dtype=torch.long, device=logits_1d.device)
    cand_logits = logits_1d.index_select(0, cand)
    target_pos = candidate_ids.index(int(target_id))
    target_logit = cand_logits[target_pos]
    other = torch.cat([cand_logits[:target_pos], cand_logits[target_pos + 1 :]])
    margin = target_logit - other.max() if other.numel() else torch.tensor(float("nan"), device=logits_1d.device)
    pred_id = int(cand[int(torch.argmax(cand_logits).item())].item())
    return {
        "target_logit": float(target_logit.detach().cpu()),
        "target_margin": float(margin.detach().cpu()),
        "pred_token_id": pred_id,
    }


def score_local_target(
    model: GPT2LMHeadModel,
    rendered: dict[str, Any],
    *,
    target_type: str,
    target_token: str,
    head_specs: list[dict[str, Any]] | None = None,
    residual_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if target_type == "marker_after_final_index":
        q_pos = rendered["anchors"]["index_positions"][-1]
        candidate_tokens = MARKER_TOKENS
    elif target_type == "answer_after_ans":
        q_pos = rendered["anchors"]["ans_token"]
        candidate_tokens = NUMBER_TOKENS
    else:
        raise ValueError(f"Unknown local target type: {target_type}")
    ids = as_input(rendered["input_ids"][: q_pos + 1])
    with head_output_intervention(model, head_specs), residual_replacement_intervention(model, residual_specs):
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    logits = out.logits[0, -1, :]
    candidate_ids = [vocab.token_to_id[tok] for tok in candidate_tokens]
    metrics = token_margin_from_logits(logits, vocab.token_to_id[target_token], candidate_ids)
    metrics["pred_token"] = vocab.id_to_token[metrics["pred_token_id"]]
    metrics["target_token"] = target_token
    return metrics


def build_prompt_corrupt_keep_trace_pairs(examples_per_count: int, seed: int) -> list[dict[str, Any]]:
    # Same trace length, corrupted prompt evidence. These pairs isolate the
    # targeted-retrieval question better than count-changing pairs whose traces
    # have different lengths. The corrupt prompt removes the final prompt needle,
    # but the teacher-forced trace remains the clean trace.
    rng = random.Random(seed)
    pairs = []
    for count in range(2, 11):
        for i in range(examples_per_count):
            clean = sample_base_example(int(cfg["seq_len"]), rng, count=count, example_id=f"local_c{count}_{i}")
            corrupt = remove_last_needle(clean, rng)
            if corrupt is None:
                continue
            pairs.append({
                "pair_id": f"prompt_delete_keep_trace_c{count}_{i}",
                "pair_type": "prompt_delete_last_keep_clean_trace",
                "clean": clean,
                "corrupt": corrupt,
                "clean_trace": trace_tokens_from_markers(clean.needle_markers, vocab),
                "clean_target_count": clean.count,
                "corrupt_prompt_count": corrupt.count,
                "target_marker": clean.needle_markers[-1],
            })
    return pairs


def run_local_causal_patching() -> pd.DataFrame:
    pairs = build_prompt_corrupt_keep_trace_pairs(PATCH_PAIRS_PER_COUNT, RANDOM_SEED + 9101)
    head_sites = [
        ("head_output", "retrieval_top1", RETRIEVAL_HEADS[:1], "index_token_last"),
        ("head_output", "retrieval_top2", RETRIEVAL_HEADS[:2], "index_token_last"),
        ("head_output", "retrieval_top4", RETRIEVAL_HEADS, "index_token_last"),
        ("head_output", "plus_one_top1_pre_index", PLUS_ONE_HEADS[:1], "pre_index_last"),
        ("head_output", "plus_one_top1_index", PLUS_ONE_HEADS[:1], "index_token_last"),
        ("head_output", "control_top2", CONTROL_HEADS, "index_token_last"),
    ]
    residual_sites = [
        ("resid_after_block", f"resid_L{layer}_index_last", layer, "index_token_last", "marker_after_final_index")
        for layer in range(1, N_LAYER + 1)
    ] + [
        ("resid_after_block", f"resid_L{layer}_ans", layer, "ans_token", "answer_after_ans")
        for layer in range(1, N_LAYER + 1)
    ]
    rows = []
    for pair in tqdm(pairs, desc="local causal patching"):
        clean = pair["clean"]
        corrupt = pair["corrupt"]
        clean_trace = pair["clean_trace"]
        clean_render = render_thinking(clean, vocab)
        corrupt_render = render_thinking(corrupt, vocab, trace_override=clean_trace, answer_count=clean.count)
        targets = [
            ("marker_after_final_index", pair["target_marker"]),
            ("answer_after_ans", vocab.count_to_token(clean.count)),
        ]
        base_scores: dict[str, dict[str, Any]] = {}
        for target_type, target_token in targets:
            clean_base = score_local_target(thinking_model, clean_render, target_type=target_type, target_token=target_token)
            corrupt_base = score_local_target(thinking_model, corrupt_render, target_type=target_type, target_token=target_token)
            base_scores[target_type] = {"clean": clean_base, "corrupt": corrupt_base}
        for site_type, name, heads, scope in head_sites:
            if not heads:
                continue
            clean_cache = cache_head_vectors(thinking_model, clean_render, heads, scope)
            specs = build_head_specs(heads, corrupt_render, mode="replace", position_scope=scope, prefix_len=corrupt_render["anchors"]["ans_token"] + 1, replacement=clean_cache)
            for target_type, target_token in targets:
                patched = score_local_target(thinking_model, corrupt_render, target_type=target_type, target_token=target_token, head_specs=specs)
                clean_margin = base_scores[target_type]["clean"]["target_margin"]
                corrupt_margin = base_scores[target_type]["corrupt"]["target_margin"]
                recovery = (patched["target_margin"] - corrupt_margin) / (clean_margin - corrupt_margin + 1e-9)
                rows.append({
                    "experiment_name": "local_causal_patching",
                    "pair_id": pair["pair_id"],
                    "pair_type": pair["pair_type"],
                    "site_type": site_type,
                    "intervention_name": name,
                    "heads": json.dumps(heads),
                    "position_scope": scope,
                    "target_type": target_type,
                    "target_token": target_token,
                    "count_clean": pair["clean_target_count"],
                    "count_corrupt": pair["corrupt_prompt_count"],
                    "clean_base_margin": clean_margin,
                    "corrupt_base_margin": corrupt_margin,
                    "patched_margin": patched["target_margin"],
                    "margin_delta": patched["target_margin"] - corrupt_margin,
                    "normalized_recovery": recovery,
                    "pred_token": patched["pred_token"],
                })
        for site_type, name, layer, scope, target_type in residual_sites:
            target_token = pair["target_marker"] if target_type == "marker_after_final_index" else vocab.count_to_token(clean.count)
            donor = cache_residual_vector(thinking_model, clean_render, layer, scope)
            if donor is None:
                continue
            positions = positions_for_scope(corrupt_render, scope, prefix_len=corrupt_render["anchors"]["ans_token"] + 1)
            residual_specs = [{"layer": layer, "positions": positions, "replacement": donor}]
            patched = score_local_target(thinking_model, corrupt_render, target_type=target_type, target_token=target_token, residual_specs=residual_specs)
            clean_margin = base_scores[target_type]["clean"]["target_margin"]
            corrupt_margin = base_scores[target_type]["corrupt"]["target_margin"]
            recovery = (patched["target_margin"] - corrupt_margin) / (clean_margin - corrupt_margin + 1e-9)
            rows.append({
                "experiment_name": "local_causal_patching",
                "pair_id": pair["pair_id"],
                "pair_type": pair["pair_type"],
                "site_type": site_type,
                "intervention_name": name,
                "heads": "[]",
                "layer": layer,
                "position_scope": scope,
                "target_type": target_type,
                "target_token": target_token,
                "count_clean": pair["clean_target_count"],
                "count_corrupt": pair["corrupt_prompt_count"],
                "clean_base_margin": clean_margin,
                "corrupt_base_margin": corrupt_margin,
                "patched_margin": patched["target_margin"],
                "margin_delta": patched["target_margin"] - corrupt_margin,
                "normalized_recovery": recovery,
                "pred_token": patched["pred_token"],
            })
    return pd.DataFrame(rows)


activation_path = TABLE_DIR / "activation_patching_results.csv"
local_patch_path = TABLE_DIR / "local_causal_patching_results.csv"
path_path = TABLE_DIR / "path_patching_results.csv"
if RUN_ACTIVATION_PATCHING:
    if SKIP_COMPLETED and activation_path.exists() and activation_path.stat().st_size > 0:
        activation_df = pd.read_csv(activation_path)
        required_directions = {"clean_to_corrupt", "corrupt_to_clean"}
        observed_directions = set(activation_df.get("patch_direction", pd.Series(dtype=str)).dropna().astype(str))
        if not required_directions.issubset(observed_directions):
            print("Existing activation_patching_results.csv is incomplete; rerunning activation patching.")
            activation_df = run_activation_patching()
            activation_df.to_csv(activation_path, index=False)
    else:
        activation_df = run_activation_patching()
        activation_df.to_csv(activation_path, index=False)
    if SKIP_COMPLETED and local_patch_path.exists() and local_patch_path.stat().st_size > 0:
        local_patch_df = pd.read_csv(local_patch_path)
    else:
        local_patch_df = run_local_causal_patching()
        local_patch_df.to_csv(local_patch_path, index=False)
else:
    activation_df = pd.read_csv(activation_path)
    local_patch_df = pd.read_csv(local_patch_path)

if RUN_PATH_PATCHING:
    # Minimal path patching is implemented as named head-output patches.
    path_df = local_patch_df[local_patch_df["site_type"].eq("head_output")].copy()
    path_df["path_name"] = np.where(
        path_df["intervention_name"].str.contains("retrieval"),
        "final_prompt_needle_to_retrieval_head_to_marker_or_answer",
        np.where(path_df["intervention_name"].str.contains("plus_one"), "local_trace_to_plus_one_head_to_marker_or_answer", "control_path"),
    )
    path_df.to_csv(path_path, index=False)
else:
    path_df = pd.read_csv(path_path)

display(Markdown(f"Saved activation patching table: `{activation_path}`"))
display(Markdown(f"Saved local causal patching table: `{local_patch_path}`"))
display(Markdown(f"Saved path patching table: `{path_path}`"))
display(activation_df.groupby(["site_type", "intervention_name", "position_scope"], as_index=False)[["normalized_recovery", "margin_delta", "count_shift"]].mean().sort_values("normalized_recovery", ascending=False).head(20))
display(local_patch_df.groupby(["target_type", "site_type", "intervention_name", "position_scope"], as_index=False)[["normalized_recovery", "margin_delta"]].mean().sort_values("normalized_recovery", ascending=False).head(20))
        """
    ),
    md("### Figures for Activation and Path Patching"),
    code(
        r"""
if not activation_df.empty:
    head_patch = activation_df[activation_df["site_type"].eq("head_output")].copy()
    if not head_patch.empty:
        hp = head_patch.groupby(["intervention_name", "position_scope"], as_index=False)["normalized_recovery"].mean()
        labels = hp["intervention_name"] + "\n" + hp["position_scope"]
        plt.figure(figsize=(11, 4.8))
        plt.bar(labels, hp["normalized_recovery"], color="#2f6fed")
        plt.axhline(0, color="black", linewidth=1)
        plt.axhline(1, color="gray", linewidth=1, linestyle="--")
        if float(hp["normalized_recovery"].abs().max()) < 1e-3:
            plt.ylim(-0.05, 1.05)
            plt.text(
                0.5,
                0.55,
                "No measurable final-answer recovery in this setting.\n"
                "This usually means final count readout is saturated/redundant;\n"
                "inspect the local marker/readout patching plot below.",
                transform=plt.gca().transAxes,
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
            )
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("mean normalized recovery")
        plt.xlabel("patched head group and token position")
        plt.title("Final-answer head-output patching: often insensitive when readout is saturated")
        savefig("patching_recovery_by_head_group.png")

if not local_patch_df.empty:
    local_head = local_patch_df[local_patch_df["site_type"].eq("head_output")].copy()
    if not local_head.empty:
        hp = local_head.groupby(["target_type", "intervention_name", "position_scope"], as_index=False)["normalized_recovery"].mean()
        target_types = list(hp["target_type"].drop_duplicates())
        fig, axes = plt.subplots(1, len(target_types), figsize=(max(8, 6 * len(target_types)), 4.8), squeeze=False)
        for ax, target_type in zip(axes[0], target_types):
            sub = hp[hp["target_type"].eq(target_type)].copy()
            labels = sub["intervention_name"] + "\n" + sub["position_scope"]
            ax.bar(labels, sub["normalized_recovery"], color="#2f6fed")
            ax.axhline(0, color="black", linewidth=1)
            ax.axhline(1, color="gray", linewidth=1, linestyle="--")
            ax.set_title(target_type)
            ax.set_ylabel("mean normalized recovery")
            ax.set_xlabel("patched head group and token position")
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle("Local causal patching: marker-after-index vs final-answer readout")
        savefig("local_patching_marker_and_answer_recovery_by_head_group.png")

    local_resid = local_patch_df[local_patch_df["site_type"].eq("resid_after_block")].copy()
    if not local_resid.empty:
        target_types = list(local_resid["target_type"].drop_duplicates())
        pretty_target = {
            "marker_after_final_index": "Marker after final trace index",
            "answer_after_ans": "Final answer after <Ans>",
        }
        pretty_position = {
            "index_token_last": "last trace\nindex token",
            "ans_token": "<Ans>\ntoken",
            "think_end": "</think>\ntoken",
        }
        fig, axes = plt.subplots(1, len(target_types), figsize=(max(8, 4.8 * len(target_types)), 3.9), squeeze=False)
        last_im = None
        for ax, target_type in zip(axes[0], target_types):
            sub = local_resid[local_resid["target_type"].eq(target_type)]
            mat = sub.groupby(["position_scope", "layer"], as_index=False)["normalized_recovery"].mean().pivot(index="position_scope", columns="layer", values="normalized_recovery")
            last_im = ax.imshow(mat.fillna(0).values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_title(pretty_target.get(target_type, target_type), fontsize=11, pad=8)
            ax.set_yticks(range(len(mat.index)), [pretty_position.get(x, x.replace("_", "\n")) for x in mat.index])
            ax.set_xticks(range(len(mat.columns)), mat.columns)
            ax.set_xlabel("residual stream after layer")
            ax.set_ylabel("patched token position")
            ax.tick_params(axis="both", labelsize=9)
        fig.subplots_adjust(left=0.11, right=0.88, top=0.80, bottom=0.20, wspace=0.38)
        cax = fig.add_axes([0.90, 0.24, 0.015, 0.50])
        fig.colorbar(last_im, cax=cax, label="normalized recovery")
        fig.suptitle("Local residual patching recovery by target, layer, and position", y=0.95)
        savefig("patching_recovery_by_layer_position.png")

if not path_df.empty:
    target_types = list(path_df["target_type"].drop_duplicates()) if "target_type" in path_df.columns else ["all"]
    pretty_target = {
        "marker_after_final_index": "Marker after final trace index",
        "answer_after_ans": "Final answer after <Ans>",
    }
    pretty_component = {
        "control_top2": "control\ntop-2",
        "plus_one_top1_index": "+1 head\nlast index",
        "plus_one_top1_pre_index": "+1 head\npre-index",
        "retrieval_top1": "retrieval\ntop-1",
        "retrieval_top2": "retrieval\ntop-2",
        "retrieval_top4": "retrieval\ntop-4",
    }
    pretty_path = {
        "control_path": "control",
        "final_prompt_needle_to_retrieval_head_to_marker_or_answer": "prompt-final-needle\nvia retrieval head",
        "local_trace_to_plus_one_head_to_marker_or_answer": "local trace\nvia +1 head",
    }
    fig_h = max(3.2, 0.45 * path_df["path_name"].nunique() + 2.0)
    fig, axes = plt.subplots(1, len(target_types), figsize=(max(8, 5.2 * len(target_types)), fig_h), squeeze=False)
    last_im = None
    for ax, target_type in zip(axes[0], target_types):
        sub = path_df[path_df["target_type"].eq(target_type)] if target_type != "all" else path_df
        path_sum = sub.groupby(["path_name", "intervention_name"], as_index=False)["normalized_recovery"].mean()
        mat = path_sum.pivot(index="path_name", columns="intervention_name", values="normalized_recovery").fillna(0)
        last_im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_yticks(range(len(mat.index)), [pretty_path.get(x, x.replace("_", "\n")) for x in mat.index])
        ax.set_xticks(range(len(mat.columns)), [pretty_component.get(x, x.replace("_", "\n")) for x in mat.columns], rotation=0, ha="center")
        ax.set_xlabel("patched component")
        ax.set_ylabel("causal path hypothesis")
        ax.set_title(pretty_target.get(target_type, target_type), fontsize=11, pad=8)
        ax.tick_params(axis="both", labelsize=8)
    fig.subplots_adjust(left=0.18, right=0.88, top=0.80, bottom=0.28, wspace=0.52)
    cax = fig.add_axes([0.90, 0.29, 0.015, 0.42])
    fig.colorbar(last_im, cax=cax, label="normalized recovery")
    fig.suptitle("Minimal path patching recovery heatmap", y=0.95)
    savefig("path_patching_recovery_heatmap.png")
        """
    ),
    md(
        r"""
## 9. Causal Test E: Counterfactual Prompt and Trace Edits

这一节专门分开 prompt count 和 trace count：

- `prompt_delete_last_keep_trace`：prompt 少一个 needle，但 teacher-forced trace 仍然是旧的完整 trace；
- `prompt_insert_last_keep_trace`：prompt 多一个 needle，但 trace 仍然没有这个新 needle；
- `wrong_final_index_keep_prompt`：prompt 正确，但 trace 最后一个 index token 被改错；
- `wrong_previous_marker_keep_prompt`：prompt 正确，但 trace 中倒数第二个 marker 被改错。

如果最终答案更跟 trace 走，`trace_count_logit - prompt_count_logit` 会偏正；如果更跟 prompt 走，会偏负。
        """
    ),
    code(
        r"""
def render_attention_mass(model: GPT2LMHeadModel, rendered: dict[str, Any], layer: int, head: int, q_scope: str) -> dict[str, float]:
    q_positions = positions_for_scope(rendered, q_scope, prefix_len=rendered["anchors"]["ans_token"] + 1)
    if not q_positions:
        return {"all_prompt_needles_mass": np.nan, "last_prompt_needle_mass": np.nan, "prompt_noise_mass": np.nan}
    ids = as_input(rendered["input_ids"][: rendered["anchors"]["ans_token"] + 1])
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_attentions=True, use_cache=False)
    weights = out.attentions[layer - 1][0, head].detach().float().cpu().numpy()
    q = q_positions[-1]
    row = weights[q]
    needles = rendered["anchors"]["prompt_needle_positions"]
    prompt_positions = list(range(rendered["anchors"]["prompt_start"], rendered["anchors"]["prompt_end_exclusive"]))
    noise = [p for p in prompt_positions if p not in set(needles)]
    return {
        "all_prompt_needles_mass": float(row[needles].sum()) if needles else 0.0,
        "last_prompt_needle_mass": float(row[needles[-1]]) if needles else 0.0,
        "prompt_noise_mass": float(row[noise].sum()) if noise else 0.0,
    }


def wrong_final_index_trace(ex: BaseExample, vocab: Vocab) -> list[str]:
    trace = trace_tokens_from_markers(ex.needle_markers, vocab)
    if ex.count >= 2:
        trace[-2] = vocab.count_to_token(ex.count - 1)
    else:
        trace[-2] = vocab.count_to_token(2)
    return trace


def wrong_previous_marker_trace(ex: BaseExample, vocab: Vocab, rng: random.Random) -> list[str]:
    trace = trace_tokens_from_markers(ex.needle_markers, vocab)
    if ex.count >= 2:
        idx = -3
    else:
        idx = -1
    old = trace[idx]
    choices = [m for m in MARKER_TOKENS if m != old]
    trace[idx] = rng.choice(choices)
    return trace


def run_counterfactuals() -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED + 11001)
    examples = balanced_examples(int(cfg["seq_len"]), EXAMPLES_PER_COUNT, RANDOM_SEED + 11002)
    rows = []
    l_retr, h_retr = RETRIEVAL_HEADS[0]
    for ex in tqdm(examples, desc="counterfactual edits"):
        base_trace = trace_tokens_from_markers(ex.needle_markers, vocab)
        variants = []
        minus = remove_last_needle(ex, rng)
        if minus is not None:
            variants.append(("prompt_delete_last_keep_trace", minus, base_trace, ex.count, minus.count))
        plus = add_final_needle(ex, rng)
        if plus is not None:
            variants.append(("prompt_insert_last_keep_trace", plus, base_trace, ex.count, plus.count))
        variants.append(("wrong_final_index_keep_prompt", ex, wrong_final_index_trace(ex, vocab), ex.count, max(1, ex.count - 1)))
        variants.append(("wrong_previous_marker_keep_prompt", ex, wrong_previous_marker_trace(ex, vocab, rng), ex.count, ex.count))
        for edit_type, prompt_ex, trace, trace_count, prompt_count in variants:
            rendered = render_thinking(prompt_ex, vocab, trace_override=trace, answer_count=trace_count)
            ans_pos = rendered["anchors"]["ans_token"]
            ids = as_input(rendered["input_ids"][: ans_pos + 1])
            with torch.no_grad():
                out = thinking_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            logits = count_logits_from_logits(out.logits[0, -1, :], vocab)
            trace_logit = float(logits[int(trace_count) - 1].detach().cpu())
            prompt_logit = float(logits[int(prompt_count) - 1].detach().cpu()) if 1 <= int(prompt_count) <= 10 else np.nan
            pred_count = int(torch.argmax(logits).item()) + 1
            attn = render_attention_mass(thinking_model, rendered, l_retr, h_retr, "index_token_last")
            rows.append({
                "experiment_name": "counterfactual_edits",
                "example_id": ex.example_id,
                "edit_type": edit_type,
                "prompt_count": int(prompt_count),
                "trace_count": int(trace_count),
                "pred_count": pred_count,
                "trace_count_logit": trace_logit,
                "prompt_count_logit": prompt_logit,
                "trace_minus_prompt_logit": trace_logit - prompt_logit if np.isfinite(prompt_logit) else np.nan,
                "retrieval_head": f"L{l_retr}H{h_retr}",
                **attn,
            })
    return pd.DataFrame(rows)


counterfactual_path = TABLE_DIR / "counterfactual_edit_results.csv"
if RUN_COUNTERFACTUALS:
    if SKIP_COMPLETED and counterfactual_path.exists() and counterfactual_path.stat().st_size > 0:
        counterfactual_df = pd.read_csv(counterfactual_path)
    else:
        counterfactual_df = run_counterfactuals()
        counterfactual_df.to_csv(counterfactual_path, index=False)
else:
    counterfactual_df = pd.read_csv(counterfactual_path)

display(Markdown(f"Saved counterfactual edit table: `{counterfactual_path}`"))
display(counterfactual_df.groupby("edit_type", as_index=False)[["trace_minus_prompt_logit", "last_prompt_needle_mass", "all_prompt_needles_mass"]].mean())
        """
    ),
    md("### Figures for Counterfactuals"),
    code(
        r"""
if not counterfactual_df.empty:
    cf = counterfactual_df.groupby("edit_type", as_index=False)["trace_minus_prompt_logit"].mean()
    plt.figure(figsize=(9, 4.6))
    plt.bar(cf["edit_type"], cf["trace_minus_prompt_logit"], color="#2f6fed")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("mean logit(trace count) - logit(prompt count)")
    plt.xlabel("counterfactual edit")
    plt.title("Does final answer follow teacher-forced trace or edited prompt?")
    savefig("counterfactual_prompt_vs_trace_logits.png")

    attn_cf = counterfactual_df.groupby("edit_type", as_index=False)[["last_prompt_needle_mass", "all_prompt_needles_mass", "prompt_noise_mass"]].mean()
    x = np.arange(len(attn_cf))
    width = 0.25
    plt.figure(figsize=(9.5, 4.6))
    for i, metric in enumerate(["last_prompt_needle_mass", "all_prompt_needles_mass", "prompt_noise_mass"]):
        plt.bar(x + (i - 1) * width, attn_cf[metric], width=width, label=metric)
    plt.xticks(x, attn_cf["edit_type"], rotation=25, ha="right")
    plt.ylabel("attention mass")
    plt.xlabel("counterfactual edit")
    plt.title(f"Retrieval head attention shift under counterfactual edits ({counterfactual_df['retrieval_head'].iloc[0]})")
    plt.legend(fontsize=8)
    savefig("counterfactual_l3h3_attention_shift.png")
        """
    ),
    md("## 10. Optional Residual Count Direction"),
    code(
        r"""
residual_direction_path = TABLE_DIR / "residual_direction_results.csv"
residual_steering_path = TABLE_DIR / "residual_steering_results.csv"

if RUN_RESIDUAL_DIRECTIONS:
    # Lightweight linear readout without sklearn: fit y from final hidden state at <Ans>.
    examples = balanced_examples(int(cfg["seq_len"]), max(10, EXAMPLES_PER_COUNT), RANDOM_SEED + 13001)
    X, y = [], []
    for ex in tqdm(examples, desc="collect residual states"):
        r = render_thinking(ex, vocab)
        ans_pos = r["anchors"]["ans_token"]
        ids = as_input(r["input_ids"][: ans_pos + 1])
        with torch.no_grad():
            out = thinking_model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False)
        X.append(out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy())
        y.append(float(ex.count))
    X = np.asarray(X)
    y = np.asarray(y)
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    ridge = 1e-3
    direction = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(Xc.shape[1]), Xc.T @ yc)
    pred = Xc @ direction + y.mean()
    ss_res = float(((pred - y) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / (ss_tot + 1e-9)
    residual_direction_df = pd.DataFrame([{"direction_type": "ridge_ans_final_hidden", "projection_r2": r2, "n_examples": len(examples)}])
    residual_direction_df.to_csv(residual_direction_path, index=False)
    residual_steering_df = pd.DataFrame([{"status": "not_run", "reason": "steering is intentionally secondary in v3.2"}])
    residual_steering_df.to_csv(residual_steering_path, index=False)
else:
    residual_direction_df = pd.DataFrame([{"status": "skipped", "reason": "RUN_RESIDUAL_DIRECTIONS=False"}])
    residual_steering_df = pd.DataFrame([{"status": "skipped", "reason": "RUN_RESIDUAL_DIRECTIONS=False"}])
    residual_direction_df.to_csv(residual_direction_path, index=False)
    residual_steering_df.to_csv(residual_steering_path, index=False)

display(residual_direction_df)
        """
    ),
    md("## 11. Synthesis"),
    code(
        r"""
def safe_mean(df: pd.DataFrame, filt: pd.Series, col: str) -> float:
    sub = df[filt]
    if sub.empty or col not in sub:
        return float("nan")
    return float(sub[col].mean())


baseline_margin = safe_mean(
    necessity_df,
    necessity_df["condition"].eq("baseline_no_ablation")
    & necessity_df["position_scope"].eq("index_token_last")
    & necessity_df["eval_mode"].eq("teacher_forced_final_readout"),
    "gold_logit_margin",
)
top_retr_margin = safe_mean(
    necessity_df,
    necessity_df["condition"].eq("retrieval_top4")
    & necessity_df["position_scope"].eq("index_token_last")
    & necessity_df["eval_mode"].eq("teacher_forced_final_readout"),
    "gold_logit_margin",
)
control_margin = safe_mean(
    necessity_df,
    necessity_df["condition"].eq("low_score_controls")
    & necessity_df["position_scope"].eq("index_token_last")
    & necessity_df["eval_mode"].eq("teacher_forced_final_readout"),
    "gold_logit_margin",
)
best_final_patch = activation_df.sort_values("normalized_recovery", ascending=False).head(1) if "activation_df" in globals() and not activation_df.empty else pd.DataFrame()
best_local_patch = local_patch_df.sort_values("normalized_recovery", ascending=False).head(1) if "local_patch_df" in globals() and not local_patch_df.empty else pd.DataFrame()
cf_summary = counterfactual_df.groupby("edit_type", as_index=False)["trace_minus_prompt_logit"].mean() if "counterfactual_df" in globals() and not counterfactual_df.empty else pd.DataFrame()

lines = []
lines.append("### v3.2 当前结论草稿")
lines.append("")
lines.append(f"- **实验对象**：v2 thinking model，checkpoint `{THINKING_MODEL_DIR}`；架构是 GPT-2 style learned absolute positional embeddings。")
lines.append(f"- **候选 retrieval heads**：{', '.join(f'L{l}H{h}' for l, h in RETRIEVAL_HEADS)}。候选 local/plus-one heads：{', '.join(f'L{l}H{h}' for l, h in PLUS_ONE_HEADS)}。")
lines.append(f"- **Necessity 初读**：baseline final-answer margin = `{baseline_margin:.3f}`；retrieval_top4 at final index margin = `{top_retr_margin:.3f}`；control margin = `{control_margin:.3f}`。如果 retrieval_top4 和 control 接近，说明这些 attention heads 更像 diagnostic/redundant；如果 retrieval_top4 明显更低，说明 retrieval group 有必要性。")
if not best_local_patch.empty:
    r = best_local_patch.iloc[0]
    lines.append(f"- **Local sufficiency 初读**：局部 target `{r['target_type']}` 上最高 normalized recovery 来自 `{r['intervention_name']}` at `{r['position_scope']}`，recovery = `{r['normalized_recovery']:.3f}`。这比 final-answer patching 更直接检验 targeted retrieval 是否携带最后 marker/needle 信息。")
if not best_final_patch.empty:
    r = best_final_patch.iloc[0]
    lines.append(f"- **Final-answer patching 初读**：最高 final-answer recovery 来自 `{r['intervention_name']}` at `{r['position_scope']}`，recovery = `{r['normalized_recovery']:.3f}`。如果这里接近 0 但 local patching 有效果，说明最终读出可能被 trace/readout 冗余保护。")
if not cf_summary.empty:
    lines.append("- **Prompt vs trace 初读**：`trace_minus_prompt_logit > 0` 表示最终答案更支持 teacher-forced trace count；`< 0` 表示更支持 edited prompt count。")
    for _, r in cf_summary.iterrows():
        lines.append(f"  - `{r['edit_type']}`: `{r['trace_minus_prompt_logit']:.3f}`")
lines.append("")
lines.append("### 需要用谨慎语言解释")
lines.append("")
lines.append("- Attention mass 高本身不是因果证据；只有 ablation/scale/patching 相对 control 改变 logits 或行为时，才说支持 causal role。")
lines.append("- 如果 ablation 小但 patching 有 recovery，比较合理的说法是：该 head 携带 sufficient information，但模型有 redundant routes。")
lines.append("- 如果 ablation 和 patching 都弱，比较合理的说法是：该 attention pattern 更像 diagnostic，而不是当前实验下可检测到的必要/充分路径。")
display(Markdown("\n".join(lines)))
        """
    ),
    md("## 12. Save Results to Google Drive"),
    code(
        r"""
DRIVE_SAVE_COMPLETED = False
SAVE_RESULTS = True

if SAVE_RESULTS:
    if IN_COLAB and SAVE_TO_DRIVE:
        from google.colab import drive

        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        results_root = DRIVE_RESULTS_ROOT
    else:
        results_root = LOCAL_RESULTS_ROOT
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = results_root / f"v3_2_causal_seed{RANDOM_SEED}_{timestamp}_{PRESET}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CAUSAL_DIR, bundle_dir / "v3_2_causal", dirs_exist_ok=True)
    nb_src = Path("notebooks/Trace_Count_v3_2_Colab.ipynb")
    if nb_src.exists():
        (bundle_dir / "notebooks").mkdir(exist_ok=True)
        shutil.copy2(nb_src, bundle_dir / "notebooks" / nb_src.name)
    manifest = {
        "source_v2_run_dir": str(V2_RUN_DIR),
        "thinking_model_dir": str(THINKING_MODEL_DIR),
        "causal_dir": str(CAUSAL_DIR),
        "preset": PRESET,
        "examples_per_count": EXAMPLES_PER_COUNT,
        "patch_pairs_per_count": PATCH_PAIRS_PER_COUNT,
        "retrieval_heads": RETRIEVAL_HEADS,
        "plus_one_heads": PLUS_ONE_HEADS,
        "control_heads": CONTROL_HEADS,
        "saved_at": timestamp,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if IN_COLAB and SAVE_TO_DRIVE:
        DRIVE_SAVE_COMPLETED = True
    print("Saved bundle:", bundle_dir)
        """
    ),
    md("## 13. Optional GitHub Push"),
    code(
        r"""
if ENABLE_GITHUB_PUSH:
    subprocess.run(["git", "status", "--short"], check=False)
    print("GitHub push is intentionally disabled by default. Add/commit/push manually after reviewing outputs.")
else:
    print("ENABLE_GITHUB_PUSH=False; skipping GitHub push.")
        """
    ),
    md("## 14. Optional Runtime Disconnect"),
    code(
        r"""
if AUTO_DISCONNECT and globals().get("DRIVE_SAVE_COMPLETED", False):
    import time

    print("Drive save completed. Disconnecting Colab runtime in 3 seconds...")
    time.sleep(3)
    try:
        from google.colab import drive, runtime

        try:
            drive.flush_and_unmount()
            print("Google Drive flushed and unmounted.")
        except Exception as exc:
            print(f"Drive flush/unmount skipped: {exc}")
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect unavailable: {exc}")
else:
    print("AUTO_DISCONNECT is False or Drive save did not run; kernel stays alive.")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {OUT}")
