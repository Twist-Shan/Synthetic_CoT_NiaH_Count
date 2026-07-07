from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Trace_Count_v3_Colab.ipynb"


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
# Trace Count v3: Symbolic NIAH Counting

This notebook implements the v3 synthetic NIAH-style counting suite from
`notebooks/pipeline_v3_codex_prompt.md`.

The core change from v2 is that the task should no longer stop at saturated
`seq_len=256` behavior. v3 asks whether explicit indexed thinking traces create
a different and more robust counting route under:

1. longer evaluation sequences (`256`, `512`, `1024`);
2. different loss-mask policies;
3. corrupted-trace readout tests;
4. probe, attention, and simple causal head-ablation diagnostics.

Default mode is `debug`. Switch `PRESET = "main"` when you want the full suite.
        """
    ),
    code(
        r"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import base64
import csv
import html
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from typing import Any

REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
# v3 is self-contained. Keeping this False avoids Colab numpy/pandas ABI
# breakage from changing package versions inside an already-running kernel.
INSTALL_PACKAGE = False

IN_COLAB = "google.colab" in sys.modules or Path("/content").exists()
if IN_COLAB:
    repo_dir = Path("/content/Synthetic_CoT_NiaH_Count")
    cwd = Path.cwd()
    if (cwd / ".git").exists() or (cwd / "pyproject.toml").exists():
        repo_dir = cwd
    elif (repo_dir / ".git").exists():
        pass
    elif repo_dir.exists() and any(repo_dir.iterdir()):
        print(f"Using existing non-git directory: {repo_dir}")
    else:
        if repo_dir.exists():
            repo_dir.rmdir()
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
    os.chdir(repo_dir)

ROOT = Path.cwd()
if INSTALL_PACKAGE and (ROOT / "pyproject.toml").exists():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
elif INSTALL_PACKAGE:
    print("pyproject.toml not found; skipped editable install.")

import matplotlib.pyplot as plt
import numpy as np
try:
    import pandas as pd
except Exception as exc:
    raise RuntimeError(
        "Pandas failed to import, usually because a previous pip install changed NumPy "
        "inside this Colab kernel. Restart the runtime, keep INSTALL_PACKAGE=False, "
        "and rerun from the first cell."
    ) from exc
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import Markdown, display
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from tqdm.auto import tqdm

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 130

print("cwd:", ROOT)
print("python:", sys.executable)
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
        """
    ),
    md(
        r"""
## Configuration

`debug` runs all code paths with tiny data and short training. `main` is the
paper-style preset requested in the v3 prompt. Round 2 main is expensive because
it trains nine loss-policy conditions for five seeds.
        """
    ),
    code(
        r"""
PRESET = "debug"  # "debug" or "main"
RUN_ROUNDS = {"round1": True, "round2": True, "round3": True, "round4": True}
RUN_TRAINING = True
SKIP_COMPLETED = True
RUN_TESTS_FIRST = True
SAVE_TO_DRIVE_AT_END = True

CONFIGS = {
    "debug": {
        "train_seq_len": 256,
        "seq_lens_eval": [256, 512],
        "train_steps": 200,
        "batch_size": 32,
        "eval_every": 50,
        "log_every": 10,
        "checkpoint_every": 100,
        "test_examples_per_count": 20,
        "probe_examples_per_count": 20,
        "attention_examples_per_count": 10,
        "corrupt_examples_per_count": 20,
        "seeds": [1234],
    },
    "main": {
        "train_seq_len": 256,
        "seq_lens_eval": [256, 512, 1024],
        "train_steps": 10000,
        "batch_size": 128,
        "eval_every": 500,
        "log_every": 50,
        "checkpoint_every": 1000,
        "test_examples_per_count": 1000,
        "probe_examples_per_count": 500,
        "attention_examples_per_count": 100,
        "corrupt_examples_per_count": 200,
        "seeds": [1234, 1235, 1236, 1237, 1238],
    },
}

cfg = dict(CONFIGS[PRESET])
cfg.update(
    {
        "count_min": 1,
        "count_max": 10,
        "noise_vocab_size": 64,
        "marker_vocab_size": 10,
        "vocab_size": 90,
        "n_layers": 4,
        "n_heads": 4,
        "d_model": 256,
        "d_mlp": 1024,
        "dropout": 0.0,
        "context_len": 1152,
        "learning_rate": 3e-4,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "warmup_steps": 500 if PRESET == "main" else 20,
        "grad_clip_norm": 1.0,
        "final_weight": 10.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("runs") / "syn_v3" / f"{TIMESTAMP}_{PRESET}"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
METRICS_DIR = RUN_DIR / "metrics"
TABLES_DIR = RUN_DIR / "tables"
FIGURES_DIR = RUN_DIR / "figures"
for p in [RUN_DIR, CHECKPOINT_DIR, METRICS_DIR, TABLES_DIR, FIGURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

with (RUN_DIR / "config.json").open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)

display(Markdown(f"**Run directory:** `{RUN_DIR}`"))
display(pd.DataFrame([{"field": k, "value": v} for k, v in cfg.items()]))
        """
    ),
    md(
        r"""
## Vocabulary, Data, Rendering, and Loss Masks

Vocabulary size should be exactly `90`: six special tokens, 64 noise tokens,
10 marker tokens, and 10 numeric tokens. `<10>` is a single token.
        """
    ),
    code(
        r"""
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<Ans>", "<Think/>", "</Think>"]
NOISE_TOKENS = [f"<N{i}>" for i in range(cfg["noise_vocab_size"])]
MARKER_TOKENS = [f"<{chr(ord('A') + i)}>" for i in range(cfg["marker_vocab_size"])]
NUMBER_TOKENS = [f"<{i}>" for i in range(1, cfg["count_max"] + 1)]


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def build(cls) -> "Vocab":
        toks = SPECIAL_TOKENS + NOISE_TOKENS + MARKER_TOKENS + NUMBER_TOKENS
        assert len(toks) == len(set(toks)) == 90
        return cls({tok: i for i, tok in enumerate(toks)}, toks)

    def encode(self, toks: list[str]) -> list[int]:
        return [self.token_to_id[t] for t in toks]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[int(i)] for i in ids]

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump({"token_to_id": self.token_to_id, "id_to_token": self.id_to_token}, f, indent=2)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def number_ids(self) -> list[int]:
        return [self.token_to_id[t] for t in NUMBER_TOKENS]


vocab = Vocab.build()
vocab.save(RUN_DIR / "vocab.json")


@dataclass
class BaseExample:
    seq_len: int
    seq_tokens: list[str]
    count: int
    needle_positions: list[int]
    needle_markers: list[str]
    seed: int | None = None


@dataclass
class RenderSpans:
    bos_pos: int
    seq_start: int
    seq_end_exclusive: int
    think_open_pos: int | None
    trace_token_positions: list[int]
    trace_index_positions: list[int]
    trace_marker_positions: list[int]
    think_close_pos: int | None
    ans_pos: int
    final_count_pos: int
    eos_pos: int


@dataclass
class Rendered:
    tokens: list[int]
    token_strs: list[str]
    spans: RenderSpans


def count_bin(count: int) -> str:
    if count <= 3:
        return "low"
    if count <= 6:
        return "mid"
    return "high"


def make_example(seq_len: int, count: int, rng: random.Random, seed: int | None = None) -> BaseExample:
    positions = sorted(rng.sample(range(seq_len), count))
    markers = [rng.choice(MARKER_TOKENS) for _ in range(count)]
    seq_tokens = [rng.choice(NOISE_TOKENS) for _ in range(seq_len)]
    for pos, marker in zip(positions, markers):
        seq_tokens[pos] = marker
    ex = BaseExample(seq_len, seq_tokens, count, positions, markers, seed)
    validate_example(ex)
    return ex


def sample_example(seq_len: int, rng: random.Random, seed: int | None = None) -> BaseExample:
    count = rng.randint(cfg["count_min"], cfg["count_max"])
    return make_example(seq_len, count, rng, seed=seed)


def balanced_examples(seq_len: int, examples_per_count: int, seed: int, counts: list[int] | None = None) -> list[BaseExample]:
    rng = random.Random(seed)
    counts = counts or list(range(cfg["count_min"], cfg["count_max"] + 1))
    out = []
    for count in counts:
        for i in range(examples_per_count):
            out.append(make_example(seq_len, count, rng, seed=seed * 1_000_000 + count * 10_000 + i))
    rng.shuffle(out)
    return out


def validate_example(ex: BaseExample) -> None:
    assert len(ex.seq_tokens) == ex.seq_len
    assert ex.count == len(ex.needle_positions) == len(ex.needle_markers)
    assert ex.needle_positions == sorted(ex.needle_positions)
    assert len(set(ex.needle_positions)) == len(ex.needle_positions)
    for pos, marker in zip(ex.needle_positions, ex.needle_markers):
        assert ex.seq_tokens[pos] == marker
    for tok in ex.seq_tokens:
        assert tok in NOISE_TOKENS or tok in MARKER_TOKENS


def number_token(n: int) -> str:
    assert 1 <= n <= cfg["count_max"]
    return f"<{n}>"


def render_non_thinking(ex: BaseExample, vocab: Vocab) -> Rendered:
    toks = ["<BOS>"] + ex.seq_tokens + ["<Ans>", number_token(ex.count), "<EOS>"]
    ans_pos = 1 + ex.seq_len
    spans = RenderSpans(
        bos_pos=0,
        seq_start=1,
        seq_end_exclusive=1 + ex.seq_len,
        think_open_pos=None,
        trace_token_positions=[],
        trace_index_positions=[],
        trace_marker_positions=[],
        think_close_pos=None,
        ans_pos=ans_pos,
        final_count_pos=ans_pos + 1,
        eos_pos=ans_pos + 2,
    )
    return Rendered(vocab.encode(toks), toks, spans)


def trace_tokens_for_example(ex: BaseExample) -> list[str]:
    out = []
    for i, marker in enumerate(ex.needle_markers, start=1):
        out.extend([number_token(i), marker])
    return out


def render_thinking(ex: BaseExample, vocab: Vocab) -> Rendered:
    trace = trace_tokens_for_example(ex)
    toks = ["<BOS>"] + ex.seq_tokens + ["<Think/>"] + trace + ["</Think>", "<Ans>", number_token(ex.count), "<EOS>"]
    think_open = 1 + ex.seq_len
    trace_start = think_open + 1
    trace_positions = list(range(trace_start, trace_start + len(trace)))
    trace_index_positions = trace_positions[0::2]
    trace_marker_positions = trace_positions[1::2]
    think_close = trace_start + len(trace)
    ans_pos = think_close + 1
    spans = RenderSpans(
        bos_pos=0,
        seq_start=1,
        seq_end_exclusive=1 + ex.seq_len,
        think_open_pos=think_open,
        trace_token_positions=trace_positions,
        trace_index_positions=trace_index_positions,
        trace_marker_positions=trace_marker_positions,
        think_close_pos=think_close,
        ans_pos=ans_pos,
        final_count_pos=ans_pos + 1,
        eos_pos=ans_pos + 2,
    )
    return Rendered(vocab.encode(toks), toks, spans)


def build_loss_weights(tokens: list[int], spans: RenderSpans, policy: str, model_type: str, final_weight: float = 10.0) -> torch.Tensor:
    weights = torch.zeros(len(tokens), dtype=torch.float32)

    def include_predicting_token(token_pos: int, weight: float = 1.0) -> None:
        pred_pos = token_pos - 1
        if 0 <= pred_pos < len(weights) - 1:
            weights[pred_pos] = weight

    if policy.endswith("full_lm"):
        weights[: len(tokens) - 1] = 1.0
        return weights

    if model_type == "non_thinking":
        if policy in {"non_completion_equal", "non_final_heavy"}:
            include_predicting_token(spans.ans_pos, 1.0)
            include_predicting_token(spans.final_count_pos, final_weight if policy == "non_final_heavy" else 1.0)
            include_predicting_token(spans.eos_pos, 1.0)
        elif policy == "non_final_only":
            include_predicting_token(spans.final_count_pos, 1.0)
        else:
            raise ValueError(f"Unknown non-thinking policy: {policy}")
        return weights

    if model_type == "thinking":
        trace_positions = spans.trace_token_positions
        if policy in {"think_trace_and_final", "think_final_heavy"}:
            if spans.think_open_pos is not None:
                include_predicting_token(spans.think_open_pos, 1.0)
            for pos in trace_positions:
                include_predicting_token(pos, 1.0)
            include_predicting_token(spans.think_close_pos, 1.0)
            include_predicting_token(spans.ans_pos, 1.0)
            include_predicting_token(spans.final_count_pos, final_weight if policy == "think_final_heavy" else 1.0)
            include_predicting_token(spans.eos_pos, 1.0)
        elif policy == "think_final_only":
            include_predicting_token(spans.final_count_pos, 1.0)
        elif policy == "think_trace_only":
            for pos in trace_positions:
                include_predicting_token(pos, 1.0)
            include_predicting_token(spans.think_close_pos, 1.0)
            include_predicting_token(spans.ans_pos, 1.0)
        else:
            raise ValueError(f"Unknown thinking policy: {policy}")
        return weights

    raise ValueError(f"Unknown model_type={model_type}")


def run_sanity_tests() -> None:
    rng = random.Random(0)
    ex = make_example(32, 10, rng, seed=1)
    assert "<10>" in NUMBER_TOKENS and len(vocab.encode(["<10>"])) == 1
    non = render_non_thinking(ex, vocab)
    think = render_thinking(ex, vocab)
    assert non.token_strs.count("<Ans>") == 1
    assert non.token_strs[non.spans.final_count_pos] == "<10>"
    assert think.token_strs[think.spans.think_open_pos] == "<Think/>"
    assert think.token_strs[think.spans.think_close_pos] == "</Think>"
    assert think.token_strs[think.spans.ans_pos] == "<Ans>"
    assert think.token_strs[think.spans.trace_token_positions[0] : think.spans.trace_token_positions[-1] + 1] == trace_tokens_for_example(ex)
    w = build_loss_weights(non.tokens, non.spans, "non_final_only", "non_thinking")
    assert torch.nonzero(w).flatten().tolist() == [non.spans.ans_pos]
    w = build_loss_weights(think.tokens, think.spans, "think_trace_only", "thinking")
    assert w[think.spans.ans_pos].item() == 0.0
    w = build_loss_weights(think.tokens, think.spans, "think_final_heavy", "thinking", final_weight=10.0)
    assert w[think.spans.ans_pos].item() == 10.0
    print("Sanity tests passed.")


if RUN_TESTS_FIRST:
    run_sanity_tests()
        """
    ),
    md(
        r"""
## RoPE Decoder-Only Transformer

v3 uses RoPE instead of learned absolute position embeddings so that evaluation
at `512` and `1024` is not blocked by learned position IDs beyond training.
        """
    ),
    code(
        r"""
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos[:, :, : x.size(2), :]) + (rotate_half(x) * sin[:, :, : x.size(2), :])


class RoPECausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        output_attentions: bool = False,
        ablate_heads: set[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = build_rope_cache(t, self.head_dim, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None, :, :], torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        y = probs @ v
        if ablate_heads:
            for li, hi in ablate_heads:
                if int(li) == layer_idx and 0 <= int(hi) < self.n_heads:
                    y[:, int(hi), :, :] = 0.0
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y), probs if output_attentions else None


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPECausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, layer_idx: int, output_attentions: bool = False, ablate_heads: set[tuple[int, int]] | None = None):
        attn_out, probs = self.attn(self.ln1(x), layer_idx, output_attentions, ablate_heads)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, probs


class TinyRoPETransformer(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg["d_model"], cfg["n_heads"], cfg["d_mlp"], cfg["dropout"]) for _ in range(cfg["n_layers"])]
        )
        self.ln_f = nn.LayerNorm(cfg["d_model"])
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.token_embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        ablate_heads: set[tuple[int, int]] | None = None,
    ) -> dict[str, Any]:
        if input_ids.size(1) > self.cfg["context_len"]:
            raise ValueError(f"Sequence length {input_ids.size(1)} exceeds context_len={self.cfg['context_len']}")
        x = self.token_embed(input_ids)
        hidden_states = []
        attentions = []
        for layer_idx, block in enumerate(self.blocks, start=1):
            x, probs = block(x, layer_idx, output_attentions, ablate_heads)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(probs)
        logits = self.lm_head(self.ln_f(x))
        return {"logits": logits, "hidden_states": hidden_states, "attentions": attentions}


def make_model(seed: int) -> TinyRoPETransformer:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return TinyRoPETransformer(cfg).to(cfg["device"])


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


tmp_model = make_model(0)
print("params:", parameter_count(tmp_model))
del tmp_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
        """
    ),
    md(
        r"""
## Training and Evaluation Utilities
        """
    ),
    code(
        r"""
ROUND1_CONDITIONS = [
    ("non_thinking", "non_final_only"),
    ("thinking", "think_trace_and_final"),
]
ROUND2_CONDITIONS = [
    ("non_thinking", "non_full_lm"),
    ("non_thinking", "non_completion_equal"),
    ("non_thinking", "non_final_heavy"),
    ("non_thinking", "non_final_only"),
    ("thinking", "think_full_lm"),
    ("thinking", "think_trace_and_final"),
    ("thinking", "think_final_heavy"),
    ("thinking", "think_final_only"),
    ("thinking", "think_trace_only"),
]
ROUND3_CONDITIONS = [
    ("thinking", "think_trace_and_final"),
    ("thinking", "think_final_heavy"),
    ("thinking", "think_final_only"),
    ("thinking", "think_trace_only"),
]
ROUND4_CONDITIONS = [
    ("non_thinking", "non_final_only"),
    ("thinking", "think_trace_and_final"),
    ("thinking", "think_final_heavy"),
    ("thinking", "think_final_only"),
]


def condition_name(model_type: str, loss_policy: str, seed: int) -> str:
    return f"{model_type}__{loss_policy}__seed{seed}"


def stable_condition_offset(model_type: str, loss_policy: str) -> int:
    text = f"{model_type}|{loss_policy}"
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 1_000_000


def render_for_model(ex: BaseExample, model_type: str, vocab: Vocab) -> Rendered:
    return render_non_thinking(ex, vocab) if model_type == "non_thinking" else render_thinking(ex, vocab)


def make_batch(model_type: str, loss_policy: str, rng: random.Random, batch_size: int, vocab: Vocab) -> dict[str, torch.Tensor]:
    rendered = []
    weights = []
    for _ in range(batch_size):
        ex = sample_example(cfg["train_seq_len"], rng)
        r = render_for_model(ex, model_type, vocab)
        rendered.append(r.tokens)
        weights.append(build_loss_weights(r.tokens, r.spans, loss_policy, model_type, cfg["final_weight"]))
    max_len = max(len(x) for x in rendered)
    input_ids = torch.full((batch_size, max_len), vocab.pad_id, dtype=torch.long)
    weight_t = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for i, (ids, w) in enumerate(zip(rendered, weights)):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        weight_t[i, : len(ids)] = w
    return {"input_ids": input_ids.to(cfg["device"]), "loss_weights": weight_t.to(cfg["device"])}


def weighted_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor, loss_weights: torch.Tensor) -> torch.Tensor:
    logits_shift = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    weights = loss_weights[:, :-1].contiguous()
    ce = F.cross_entropy(logits_shift.view(-1, logits_shift.size(-1)), labels.view(-1), reduction="none").view_as(labels)
    denom = weights.sum().clamp_min(1.0)
    return (ce * weights).sum() / denom


def diagnostic_losses(model: nn.Module, model_type: str, examples: list[BaseExample], vocab: Vocab, max_examples: int = 64) -> dict[str, float]:
    model.eval()
    rows = []
    with torch.no_grad():
        for ex in examples[:max_examples]:
            r = render_for_model(ex, model_type, vocab)
            ids = torch.tensor(r.tokens, dtype=torch.long, device=cfg["device"])[None, :]
            logits = model(ids)["logits"]
            ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, cfg["vocab_size"]), ids[:, 1:].reshape(-1), reduction="none")
            ce = ce.view(1, -1)[0]
            spans = r.spans
            prompt_positions = list(range(spans.seq_start - 1, spans.seq_end_exclusive - 1))
            trace_positions = []
            if model_type == "thinking":
                trace_positions = [p - 1 for p in spans.trace_token_positions + [spans.think_close_pos, spans.ans_pos] if p is not None and p > 0]
            final_positions = [spans.ans_pos]
            rows.append(
                {
                    "prompt": float(ce[prompt_positions].mean().item()) if prompt_positions else math.nan,
                    "trace": float(ce[trace_positions].mean().item()) if trace_positions else math.nan,
                    "final": float(ce[final_positions].mean().item()),
                }
            )
    model.train()
    return {
        "train_prompt_loss": float(np.nanmean([r["prompt"] for r in rows])),
        "train_trace_loss": float(np.nanmean([r["trace"] for r in rows])) if any(not math.isnan(r["trace"]) for r in rows) else math.nan,
        "train_final_answer_loss": float(np.nanmean([r["final"] for r in rows])),
    }


def lr_for_step(step: int) -> float:
    warmup = max(cfg["warmup_steps"], 1)
    if step <= warmup:
        return cfg["learning_rate"] * step / warmup
    progress = (step - warmup) / max(cfg["train_steps"] - warmup, 1)
    return cfg["learning_rate"] * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(model: nn.Module, model_type: str, loss_policy: str, seed: int, step: int) -> Path:
    ckpt = CHECKPOINT_DIR / condition_name(model_type, loss_policy, seed) / f"step_{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg}, ckpt / "model.pt")
    vocab.save(ckpt / "vocab.json")
    return ckpt


def final_checkpoint_path(model_type: str, loss_policy: str, seed: int) -> Path:
    return CHECKPOINT_DIR / condition_name(model_type, loss_policy, seed) / "final" / "model.pt"


def load_condition_model(model_type: str, loss_policy: str, seed: int, step: str | int = "final") -> TinyRoPETransformer:
    model = make_model(seed)
    if step == "final":
        path = final_checkpoint_path(model_type, loss_policy, seed)
    else:
        path = CHECKPOINT_DIR / condition_name(model_type, loss_policy, seed) / f"step_{int(step):06d}" / "model.pt"
    state = torch.load(path, map_location=cfg["device"])
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def train_condition(model_type: str, loss_policy: str, seed: int) -> list[dict[str, Any]]:
    final_path = final_checkpoint_path(model_type, loss_policy, seed)
    if SKIP_COMPLETED and final_path.exists():
        print(f"[skip] {condition_name(model_type, loss_policy, seed)}")
        return []

    print(f"[train] {condition_name(model_type, loss_policy, seed)}")
    rng = random.Random(seed + stable_condition_offset(model_type, loss_policy))
    torch.manual_seed(seed)
    model = make_model(seed)
    opt = AdamW(model.parameters(), lr=cfg["learning_rate"], betas=cfg["betas"], weight_decay=cfg["weight_decay"])
    log_examples = balanced_examples(cfg["train_seq_len"], 8 if PRESET == "debug" else 32, seed + 17)
    rows = []
    pbar = tqdm(range(1, cfg["train_steps"] + 1), desc=condition_name(model_type, loss_policy, seed), leave=False)
    for step in pbar:
        lr = lr_for_step(step)
        for group in opt.param_groups:
            group["lr"] = lr
        batch = make_batch(model_type, loss_policy, rng, cfg["batch_size"], vocab)
        out = model(batch["input_ids"])
        loss = weighted_lm_loss(out["logits"], batch["input_ids"], batch["loss_weights"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
        opt.step()
        if step == 1 or step % cfg["log_every"] == 0 or step == cfg["train_steps"]:
            diag = diagnostic_losses(model, model_type, log_examples, vocab)
            row = {
                "step": step,
                "model_type": model_type,
                "loss_policy": loss_policy,
                "seed": seed,
                "train_total_loss": float(loss.item()),
                "learning_rate": lr,
                **diag,
            }
            rows.append(row)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
        if step % cfg["checkpoint_every"] == 0 or step == cfg["train_steps"]:
            save_checkpoint(model, model_type, loss_policy, seed, step)
    final_dir = CHECKPOINT_DIR / condition_name(model_type, loss_policy, seed) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg}, final_dir / "model.pt")
    vocab.save(final_dir / "vocab.json")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows
        """
    ),
    md(
        r"""
## Generation, Parsing, and Evaluation
        """
    ),
    code(
        r"""
def restrict_numeric_prediction(logits: torch.Tensor, vocab: Vocab) -> tuple[int, float, dict[int, float]]:
    number_ids = torch.tensor(vocab.number_ids, device=logits.device)
    sub = logits[number_ids]
    pred_idx = int(torch.argmax(sub).item())
    pred_count = pred_idx + 1
    probs = torch.softmax(sub, dim=-1).detach().cpu().numpy()
    return pred_count, float(-torch.log_softmax(sub, dim=-1)[pred_idx].item()), {i + 1: float(p) for i, p in enumerate(probs)}


@torch.no_grad()
def greedy_generate(
    model: nn.Module,
    prefix_ids: list[int],
    max_new_tokens: int,
    stop_id: int | None = None,
    ablate_heads: set[tuple[int, int]] | None = None,
) -> list[int]:
    ids = torch.tensor(prefix_ids, dtype=torch.long, device=cfg["device"])[None, :]
    generated = []
    for _ in range(max_new_tokens):
        logits = model(ids, ablate_heads=ablate_heads)["logits"][0, -1]
        nxt = int(torch.argmax(logits).item())
        generated.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=cfg["device"])], dim=1)
        if stop_id is not None and nxt == stop_id:
            break
        if ids.size(1) >= cfg["context_len"]:
            break
    return generated


def parse_trace_tokens(token_strs: list[str]) -> dict[str, Any]:
    pairs = []
    malformed = False
    i = 0
    while i < len(token_strs):
        if i + 1 >= len(token_strs):
            malformed = True
            break
        idx_tok, marker_tok = token_strs[i], token_strs[i + 1]
        if idx_tok not in NUMBER_TOKENS or marker_tok not in MARKER_TOKENS:
            malformed = True
            i += 1
            continue
        pairs.append((int(idx_tok.strip("<>")), marker_tok))
        i += 2
    return {"pairs": pairs, "malformed": malformed}


def parse_thinking_generation(full_generated: list[str]) -> dict[str, Any]:
    invalid = False
    if "</Think>" not in full_generated:
        return {"invalid": True, "reason": "missing_think_close", "trace_tokens": full_generated, "pred_count": None}
    close_idx = full_generated.index("</Think>")
    trace_tokens = full_generated[:close_idx]
    after_close = full_generated[close_idx + 1 :]
    if "<Ans>" not in after_close:
        return {"invalid": True, "reason": "missing_ans", "trace_tokens": trace_tokens, "pred_count": None}
    ans_rel = after_close.index("<Ans>")
    after_ans = after_close[ans_rel + 1 :]
    pred_count = None
    for tok in after_ans:
        if tok in NUMBER_TOKENS:
            pred_count = int(tok.strip("<>"))
            break
    if pred_count is None:
        invalid = True
    parsed = parse_trace_tokens(trace_tokens)
    return {
        "invalid": invalid,
        "reason": "no_numeric_after_ans" if invalid else "",
        "trace_tokens": trace_tokens,
        "trace_pairs": parsed["pairs"],
        "trace_malformed": parsed["malformed"],
        "pred_count": pred_count,
    }


def trace_metrics(ex: BaseExample, parsed: dict[str, Any]) -> dict[str, float]:
    expected_tokens = trace_tokens_for_example(ex)
    gen_tokens = parsed.get("trace_tokens", [])
    pairs = parsed.get("trace_pairs", [])
    expected_pairs = [(i, m) for i, m in enumerate(ex.needle_markers, start=1)]
    marker_counter = Counter([m for _, m in pairs])
    expected_counter = Counter(ex.needle_markers)
    overlap = sum((marker_counter & expected_counter).values())
    precision = overlap / max(len(pairs), 1)
    recall = overlap / max(len(expected_pairs), 1)
    index_matches = sum(1 for (a, _), (b, _) in zip(pairs, expected_pairs) if a == b)
    index_acc = index_matches / max(len(expected_pairs), 1)
    return {
        "trace_exact_rate": float(gen_tokens == expected_tokens),
        "trace_marker_recall": float(recall),
        "trace_marker_precision": float(precision),
        "trace_index_accuracy": float(index_acc),
        "duplicate_marker_position_rate": float(max(len(pairs) - len(set(pairs)), 0) / max(len(pairs), 1)),
        "missing_trace_item_rate": float(max(len(expected_pairs) - len(pairs), 0) > 0),
        "extra_trace_item_rate": float(max(len(pairs) - len(expected_pairs), 0) > 0),
    }


@torch.no_grad()
def eval_non_thinking(model: nn.Module, examples: list[BaseExample], vocab: Vocab) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for ex in examples:
        r = render_non_thinking(ex, vocab)
        prefix = r.tokens[: r.spans.ans_pos + 1]
        ids = torch.tensor(prefix, dtype=torch.long, device=cfg["device"])[None, :]
        logits = model(ids)["logits"][0, -1]
        pred, ce, _ = restrict_numeric_prediction(logits, vocab)
        rows.append(
            {
                "count": ex.count,
                "count_bin": count_bin(ex.count),
                "pred_count": pred,
                "correct": pred == ex.count,
                "mae": abs(pred - ex.count),
                "undercount": pred < ex.count,
                "overcount": pred > ex.count,
                "invalid": False,
                "final_answer_ce": ce,
            }
        )
    return rows


@torch.no_grad()
def eval_thinking_generated(model: nn.Module, examples: list[BaseExample], vocab: Vocab, ablate_heads: set[tuple[int, int]] | None = None) -> list[dict[str, Any]]:
    rows = []
    stop_id = vocab.token_to_id["<EOS>"]
    max_new = 2 * cfg["count_max"] + 8
    model.eval()
    for ex in examples:
        r = render_thinking(ex, vocab)
        prefix = r.tokens[: r.spans.think_open_pos + 1]
        gen_ids = greedy_generate(model, prefix, max_new, stop_id=stop_id, ablate_heads=ablate_heads)
        gen_toks = vocab.decode(gen_ids)
        parsed = parse_thinking_generation(gen_toks)
        pred = parsed.get("pred_count")
        invalid = bool(parsed.get("invalid")) or pred is None
        metrics = trace_metrics(ex, parsed)
        rows.append(
            {
                "count": ex.count,
                "count_bin": count_bin(ex.count),
                "pred_count": pred if pred is not None else -1,
                "correct": (pred == ex.count) if pred is not None else False,
                "mae": abs(pred - ex.count) if pred is not None else math.nan,
                "undercount": (pred < ex.count) if pred is not None else False,
                "overcount": (pred > ex.count) if pred is not None else False,
                "invalid": invalid,
                **metrics,
            }
        )
    return rows


@torch.no_grad()
def eval_thinking_oracle(model: nn.Module, examples: list[BaseExample], vocab: Vocab) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for ex in examples:
        r = render_thinking(ex, vocab)
        prefix = r.tokens[: r.spans.ans_pos + 1]
        ids = torch.tensor(prefix, dtype=torch.long, device=cfg["device"])[None, :]
        logits = model(ids)["logits"][0, -1]
        pred, ce, _ = restrict_numeric_prediction(logits, vocab)
        rows.append(
            {
                "count": ex.count,
                "count_bin": count_bin(ex.count),
                "pred_count": pred,
                "correct": pred == ex.count,
                "mae": abs(pred - ex.count),
                "undercount": pred < ex.count,
                "overcount": pred > ex.count,
                "invalid": False,
                "final_answer_ce": ce,
            }
        )
    return rows


def summarize_eval_rows(rows: list[dict[str, Any]], group_keys: list[str], meta: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    df = pd.DataFrame(rows)
    if df.empty:
        return out
    for keys, sub in df.groupby(group_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = dict(zip(group_keys, keys))
        item.update(meta)
        item.update(
            {
                "n_examples": len(sub),
                "final_accuracy": float(sub["correct"].mean()),
                "final_mae": float(sub["mae"].dropna().mean()) if sub["mae"].notna().any() else math.nan,
                "undercount_rate": float(sub["undercount"].mean()),
                "overcount_rate": float(sub["overcount"].mean()),
                "invalid_generation_rate": float(sub["invalid"].mean()) if "invalid" in sub else 0.0,
                "final_answer_ce": float(sub["final_answer_ce"].mean()) if "final_answer_ce" in sub else math.nan,
            }
        )
        for col in ["trace_exact_rate", "trace_marker_recall", "trace_marker_precision", "trace_index_accuracy", "duplicate_marker_position_rate", "missing_trace_item_rate", "extra_trace_item_rate"]:
            item[col] = float(sub[col].mean()) if col in sub else math.nan
        out.append(item)
    return out


def evaluate_condition(model_type: str, loss_policy: str, seed: int, step: int | str = "final") -> dict[str, pd.DataFrame]:
    model = load_condition_model(model_type, loss_policy, seed, step=step)
    rows_all = []
    for seq_len in cfg["seq_lens_eval"]:
        examples = balanced_examples(seq_len, cfg["test_examples_per_count"], seed + 1000 + seq_len)
        if model_type == "non_thinking":
            rows = eval_non_thinking(model, examples, vocab)
            mode = "direct"
        else:
            rows = eval_thinking_generated(model, examples, vocab)
            mode = "generated_trace"
        for r in rows:
            r.update({"seq_len_eval": seq_len, "eval_mode": mode})
        rows_all.extend(rows)
        if model_type == "thinking":
            oracle_rows = eval_thinking_oracle(model, examples, vocab)
            for r in oracle_rows:
                r.update({"seq_len_eval": seq_len, "eval_mode": "oracle_trace"})
            rows_all.extend(oracle_rows)
    meta = {"model_type": model_type, "loss_policy": loss_policy, "seed": seed, "checkpoint_step": step}
    by_count = summarize_eval_rows(rows_all, ["eval_mode", "seq_len_eval", "count"], meta)
    by_bin = summarize_eval_rows(rows_all, ["eval_mode", "seq_len_eval", "count_bin"], meta)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"raw": pd.DataFrame(rows_all).assign(**meta), "by_count": pd.DataFrame(by_count), "by_bin": pd.DataFrame(by_bin)}
        """
    ),
    md(
        r"""
## Round 1 and Round 2: Training + Hard Evaluation
        """
    ),
    code(
        r"""
def train_requested_conditions() -> pd.DataFrame:
    conditions = []
    if RUN_ROUNDS["round1"]:
        conditions.extend(ROUND1_CONDITIONS)
    if RUN_ROUNDS["round2"]:
        conditions.extend(ROUND2_CONDITIONS)
    # Preserve order while deduplicating.
    conditions = list(dict.fromkeys(conditions))
    all_logs = []
    for seed in cfg["seeds"]:
        for model_type, loss_policy in conditions:
            if RUN_TRAINING:
                all_logs.extend(train_condition(model_type, loss_policy, seed))
    df = pd.DataFrame(all_logs)
    out = METRICS_DIR / "train_log.csv"
    if not df.empty:
        df.to_csv(out, index=False)
    elif out.exists():
        df = pd.read_csv(out)
    return df


train_log = train_requested_conditions()
display(Markdown("**Training log preview**"))
display(train_log.head() if not train_log.empty else "No new training logs; existing checkpoints may have been skipped.")


def evaluate_round1_round2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conditions = []
    if RUN_ROUNDS["round1"]:
        conditions.extend(ROUND1_CONDITIONS)
    if RUN_ROUNDS["round2"]:
        conditions.extend(ROUND2_CONDITIONS)
    conditions = list(dict.fromkeys(conditions))
    raw_parts, count_parts, bin_parts = [], [], []
    for seed in cfg["seeds"]:
        for model_type, loss_policy in tqdm(conditions, desc="eval conditions"):
            final_path = final_checkpoint_path(model_type, loss_policy, seed)
            if not final_path.exists():
                print("[missing checkpoint]", condition_name(model_type, loss_policy, seed))
                continue
            result = evaluate_condition(model_type, loss_policy, seed, step="final")
            raw_parts.append(result["raw"])
            count_parts.append(result["by_count"])
            bin_parts.append(result["by_bin"])
    raw = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
    by_count = pd.concat(count_parts, ignore_index=True) if count_parts else pd.DataFrame()
    by_bin = pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame()
    raw.to_csv(METRICS_DIR / "eval_raw_final.csv", index=False)
    by_count.to_csv(METRICS_DIR / "eval_by_count.csv", index=False)
    by_bin.to_csv(METRICS_DIR / "eval_by_bin.csv", index=False)
    return raw, by_count, by_bin


eval_raw, eval_by_count, eval_by_bin = evaluate_round1_round2()
display(Markdown("**Final eval by bin**"))
display(eval_by_bin.head(20))
        """
    ),
    md(
        r"""
## Round 3: Corrupted-Trace Diagnostics

These tests ask whether the final answer follows the prompt count, trace pair
count, last index token, max index token, or marker count when the provided
trace conflicts with the prompt.
        """
    ),
    code(
        r"""
def corrupt_trace(ex: BaseExample, corruption_type: str, rng: random.Random) -> list[str] | None:
    pairs = [(i, m) for i, m in enumerate(ex.needle_markers, start=1)]
    if corruption_type == "wrong_indices_correct_markers":
        return [tok for _, m in pairs for tok in ["<1>", m]]
    if corruption_type == "correct_indices_wrong_markers":
        return [tok for i, _ in pairs for tok in [number_token(i), rng.choice(MARKER_TOKENS)]]
    if corruption_type == "shuffled_trace_order":
        p = pairs[:]
        rng.shuffle(p)
        return [tok for i, m in p for tok in [number_token(i), m]]
    if corruption_type == "deleted_one_item":
        if len(pairs) <= 1:
            return None
        p = pairs[:]
        del p[rng.randrange(len(p))]
        return [tok for i, m in p for tok in [number_token(i), m]]
    if corruption_type == "duplicated_one_item":
        if len(pairs) >= cfg["count_max"]:
            return None
        p = pairs[:]
        p.insert(rng.randrange(len(p) + 1), rng.choice(pairs))
        return [tok for i, m in p for tok in [number_token(i), m]]
    if corruption_type == "extra_random_item":
        if ex.count >= cfg["count_max"]:
            return None
        p = pairs + [(ex.count + 1, rng.choice(MARKER_TOKENS))]
        return [tok for i, m in p for tok in [number_token(i), m]]
    if corruption_type == "last_index_replaced":
        if not pairs:
            return None
        p = pairs[:]
        last_marker = p[-1][1]
        wrong = rng.choice([x for x in range(1, cfg["count_max"] + 1) if x != p[-1][0]])
        p[-1] = (wrong, last_marker)
        return [tok for i, m in p for tok in [number_token(i), m]]
    if corruption_type == "indices_removed":
        return list(ex.needle_markers)
    if corruption_type == "markers_removed":
        return [number_token(i) for i, _ in pairs]
    raise ValueError(corruption_type)


CORRUPTION_TYPES = [
    "wrong_indices_correct_markers",
    "correct_indices_wrong_markers",
    "shuffled_trace_order",
    "deleted_one_item",
    "duplicated_one_item",
    "extra_random_item",
    "last_index_replaced",
    "indices_removed",
    "markers_removed",
]


def corrupted_trace_labels(trace_tokens: list[str], prompt_count: int) -> dict[str, Any]:
    parsed = parse_trace_tokens(trace_tokens)
    pairs = parsed["pairs"]
    idxs = [i for i, _ in pairs]
    markers = [tok for tok in trace_tokens if tok in MARKER_TOKENS]
    return {
        "prompt_count": prompt_count,
        "trace_pair_count": len(pairs),
        "last_index_value": idxs[-1] if idxs else None,
        "max_index_value": max(idxs) if idxs else None,
        "marker_count_in_trace": len(markers),
    }


@torch.no_grad()
def eval_corrupted_trace_condition(model_type: str, loss_policy: str, seed: int) -> pd.DataFrame:
    if model_type != "thinking":
        return pd.DataFrame()
    model = load_condition_model(model_type, loss_policy, seed)
    rows = []
    rng = random.Random(seed + 3030)
    for seq_len in cfg["seq_lens_eval"]:
        examples = balanced_examples(seq_len, cfg["corrupt_examples_per_count"], seed + 3000 + seq_len)
        for ex in tqdm(examples, desc=f"corrupt {loss_policy} L{seq_len}", leave=False):
            for corruption_type in CORRUPTION_TYPES:
                trace = corrupt_trace(ex, corruption_type, rng)
                if trace is None:
                    continue
                prefix_toks = ["<BOS>"] + ex.seq_tokens + ["<Think/>"] + trace + ["</Think>", "<Ans>"]
                ids = torch.tensor(vocab.encode(prefix_toks), dtype=torch.long, device=cfg["device"])[None, :]
                logits = model(ids)["logits"][0, -1]
                pred, ce, _ = restrict_numeric_prediction(logits, vocab)
                labels = corrupted_trace_labels(trace, ex.count)
                row = {
                    "model_type": model_type,
                    "loss_policy": loss_policy,
                    "seed": seed,
                    "checkpoint_step": "final",
                    "seq_len_eval": seq_len,
                    "count": ex.count,
                    "corruption_type": corruption_type,
                    "pred_count": pred,
                    "correct_prompt_count": pred == ex.count,
                    "invalid": False,
                    "final_answer_ce": ce,
                    **labels,
                }
                for key in ["prompt_count", "trace_pair_count", "last_index_value", "max_index_value", "marker_count_in_trace"]:
                    val = row[key]
                    row[f"follows_{key}"] = bool(val is not None and pred == val)
                rows.append(row)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


if RUN_ROUNDS["round3"]:
    corrupt_parts = []
    for seed in cfg["seeds"]:
        for model_type, loss_policy in ROUND3_CONDITIONS:
            if final_checkpoint_path(model_type, loss_policy, seed).exists():
                corrupt_parts.append(eval_corrupted_trace_condition(model_type, loss_policy, seed))
    corrupt_df = pd.concat(corrupt_parts, ignore_index=True) if corrupt_parts else pd.DataFrame()
    corrupt_df.to_csv(TABLES_DIR / "round3_corrupted_trace_results.csv", index=False)
    if not corrupt_df.empty:
        follow_cols = [c for c in corrupt_df.columns if c.startswith("follows_")]
        follow = corrupt_df.groupby(["loss_policy", "seq_len_eval", "corruption_type"], as_index=False)[follow_cols + ["correct_prompt_count"]].mean()
        follow.to_csv(TABLES_DIR / "round3_follow_rule_summary.csv", index=False)
        display(follow.head(20))
else:
    corrupt_df = pd.DataFrame()
        """
    ),
    md(
        r"""
## Round 4: Probes, Attention, and Head Ablation

Probe and attention are diagnostic. The report should only call a head causal if
an intervention changes final accuracy, trace quality, or logit margin.
        """
    ),
    code(
        r"""
def collect_hidden_records(model: nn.Module, model_type: str, loss_policy: str, seed: int, seq_len: int, examples: list[BaseExample]) -> list[dict[str, Any]]:
    records = []
    model.eval()
    with torch.no_grad():
        for ex_id, ex in enumerate(tqdm(examples, desc=f"cache {model_type} {loss_policy}", leave=False)):
            r = render_for_model(ex, model_type, vocab)
            ids = torch.tensor(r.tokens, dtype=torch.long, device=cfg["device"])[None, :]
            out = model(ids, output_hidden_states=True)
            hs = [model.token_embed(ids).detach().cpu()[0]] + [h.detach().cpu()[0] for h in out["hidden_states"]]
            anchors = []
            s = r.spans
            if model_type == "non_thinking":
                anchors.extend([("ans_pos", s.ans_pos, ex.count), ("pre_ans_pos", s.ans_pos - 1, ex.count)])
                for pos in ex.needle_positions[: min(ex.count, 3)]:
                    anchors.append(("needle_prompt_positions", s.seq_start + pos, 1))
                sampled_noise = [i for i, tok in enumerate(ex.seq_tokens) if tok in NOISE_TOKENS][:3]
                for pos in sampled_noise:
                    anchors.append(("noise_prompt_positions", s.seq_start + pos, 0))
            else:
                anchors.extend(
                    [
                        ("think_open_pos", s.think_open_pos, ex.count),
                        ("think_close_pos", s.think_close_pos, ex.count),
                        ("ans_pos", s.ans_pos, ex.count),
                        ("pre_ans_pos", s.ans_pos - 1, ex.count),
                    ]
                )
                for k, pos in enumerate(s.trace_index_positions, start=1):
                    anchors.append(("pre_index_k", pos - 1, k))
                    anchors.append(("index_k_pos", pos, k))
                for k, pos in enumerate(s.trace_marker_positions, start=1):
                    anchors.append(("marker_k_pos", pos, k))
                    anchors.append(("post_marker_k", min(pos + 1, len(r.tokens) - 1), k))
            for layer_idx, h in enumerate(hs):
                layer_name = "embeddings" if layer_idx == 0 else f"layer_{layer_idx}"
                for anchor_type, pos, target in anchors:
                    if pos is None or pos < 0 or pos >= h.size(0):
                        continue
                    target_type = "prefix_count" if anchor_type in {"pre_index_k", "index_k_pos", "marker_k_pos", "post_marker_k"} else "final_count"
                    if anchor_type in {"needle_prompt_positions", "noise_prompt_positions"}:
                        target_type = "is_needle"
                    records.append(
                        {
                            "model_type": model_type,
                            "loss_policy": loss_policy,
                            "seed": seed,
                            "seq_len_eval": seq_len,
                            "example_id": ex_id,
                            "layer": layer_name,
                            "anchor_type": anchor_type,
                            "target_type": target_type,
                            "target": int(target),
                            "position": int(pos),
                            "feature": h[pos].numpy().astype(np.float32),
                        }
                    )
    return records


def fit_probe(train_records: list[dict[str, Any]], test_records: list[dict[str, Any]]) -> dict[str, Any]:
    X_train = np.stack([r["feature"] for r in train_records])
    y_train = np.array([r["target"] for r in train_records])
    X_test = np.stack([r["feature"] for r in test_records])
    y_test = np.array([r["target"] for r in test_records])
    pos_train = np.array([[r["position"]] for r in train_records], dtype=float)
    pos_test = np.array([[r["position"]] for r in test_records], dtype=float)
    classes = sorted(set(y_train.tolist()))
    if len(classes) <= 1:
        return {}
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, multi_class="auto"))
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    pos_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, multi_class="auto"))
    pos_clf.fit(pos_train, y_train)
    pos_pred = pos_clf.predict(pos_test)
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 17)))
    ridge.fit(X_train, y_train.astype(float))
    rpred = ridge.predict(X_test)
    rounded = np.clip(np.rint(rpred), min(classes), max(classes)).astype(int)
    return {
        "probe_type": "logreg+ridge",
        "train_accuracy": float(accuracy_score(y_train, clf.predict(X_train))),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "ridge_rounded_accuracy": float(accuracy_score(y_test, rounded)),
        "r2": float(r2_score(y_test, rpred)),
        "mae": float(mean_absolute_error(y_test, rpred)),
        "position_only_accuracy": float(accuracy_score(y_test, pos_pred)),
        "trace_length_only_accuracy": math.nan,
        "embedding_only_accuracy": math.nan,
    }


def run_probes() -> pd.DataFrame:
    rows = []
    for seed in cfg["seeds"]:
        for model_type, loss_policy in ROUND4_CONDITIONS:
            if not final_checkpoint_path(model_type, loss_policy, seed).exists():
                continue
            model = load_condition_model(model_type, loss_policy, seed)
            seq_len = cfg["train_seq_len"]
            train_examples = balanced_examples(seq_len, cfg["probe_examples_per_count"], seed + 4001)
            test_examples = balanced_examples(seq_len, cfg["probe_examples_per_count"], seed + 4002)
            train_records = collect_hidden_records(model, model_type, loss_policy, seed, seq_len, train_examples)
            test_records = collect_hidden_records(model, model_type, loss_policy, seed, seq_len, test_examples)
            keys = ["model_type", "loss_policy", "seed", "seq_len_eval", "layer", "anchor_type", "target_type"]
            buckets = defaultdict(list)
            for rec in train_records:
                buckets[tuple(rec[k] for k in keys)].append(rec)
            test_buckets = defaultdict(list)
            for rec in test_records:
                test_buckets[tuple(rec[k] for k in keys)].append(rec)
            for key, tr in buckets.items():
                te = test_buckets.get(key, [])
                if len(tr) < 20 or len(te) < 20:
                    continue
                result = fit_probe(tr, te)
                if result:
                    rows.append(dict(zip(keys, key)) | {"checkpoint_step": "final", **result})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "round4_probe_results.csv", index=False)
    return df


@torch.no_grad()
def attention_metrics_for_condition(model_type: str, loss_policy: str, seed: int) -> pd.DataFrame:
    model = load_condition_model(model_type, loss_policy, seed)
    rows = []
    for seq_len in cfg["seq_lens_eval"]:
        examples = balanced_examples(seq_len, cfg["attention_examples_per_count"], seed + 5000 + seq_len)
        for ex in tqdm(examples, desc=f"attn {model_type} {loss_policy}", leave=False):
            r = render_for_model(ex, model_type, vocab)
            ids = torch.tensor(r.tokens, dtype=torch.long, device=cfg["device"])[None, :]
            out = model(ids, output_attentions=True)
            attentions = [a.detach().cpu()[0] for a in out["attentions"]]
            prompt_needle_positions = [r.spans.seq_start + p for p in ex.needle_positions]
            prompt_noise_positions = [r.spans.seq_start + i for i, tok in enumerate(ex.seq_tokens) if tok in NOISE_TOKENS][: max(10, ex.count)]
            if model_type == "thinking":
                anchors = {
                    "index_k_pos": r.spans.trace_index_positions,
                    "marker_k_pos": r.spans.trace_marker_positions,
                    "post_marker_k": [min(p + 1, len(r.tokens) - 1) for p in r.spans.trace_marker_positions],
                }
                for query_anchor, q_positions in anchors.items():
                    for layer_idx, att in enumerate(attentions, start=1):
                        for head in range(att.size(0)):
                            matrix = []
                            top1 = []
                            needle_mass, noise_mass, entropy = 0.0, 0.0, 0.0
                            for k, qpos in enumerate(q_positions[: ex.count]):
                                vals = att[head, qpos, prompt_needle_positions].numpy()
                                matrix.append(vals.tolist())
                                if len(vals):
                                    top1.append(int(np.argmax(vals)) == k)
                                needle_mass += float(att[head, qpos, prompt_needle_positions].sum().item())
                                if prompt_noise_positions:
                                    noise_mass += float(att[head, qpos, prompt_noise_positions].sum().item())
                                prompt_vals = att[head, qpos, r.spans.seq_start : r.spans.seq_end_exclusive].numpy()
                                prompt_vals = prompt_vals / max(prompt_vals.sum(), 1e-12)
                                entropy += float(-(prompt_vals * np.log(prompt_vals + 1e-12)).sum())
                            A = np.array(matrix, dtype=float) if matrix else np.zeros((0, 0))
                            diag = float(np.trace(A) / max(A.sum(), 1e-12)) if A.size else math.nan
                            rows.append(
                                {
                                    "model_type": model_type,
                                    "loss_policy": loss_policy,
                                    "seed": seed,
                                    "checkpoint_step": "final",
                                    "seq_len_eval": seq_len,
                                    "count": ex.count,
                                    "count_bin": count_bin(ex.count),
                                    "layer": layer_idx,
                                    "head": head,
                                    "query_anchor": query_anchor,
                                    "correct_top1_rate": float(np.mean(top1)) if top1 else math.nan,
                                    "diagonal_dominance": diag,
                                    "needle_mass": needle_mass / max(ex.count, 1),
                                    "noise_mass": noise_mass / max(ex.count, 1),
                                    "needle_to_noise_ratio": needle_mass / max(noise_mass, 1e-12),
                                    "entropy": entropy / max(ex.count, 1),
                                    "top_n_recall": math.nan,
                                }
                            )
            else:
                qpos = r.spans.ans_pos
                for layer_idx, att in enumerate(attentions, start=1):
                    for head in range(att.size(0)):
                        prompt_positions = list(range(r.spans.seq_start, r.spans.seq_end_exclusive))
                        prompt_att = att[head, qpos, prompt_positions].numpy()
                        top_idx = np.argsort(prompt_att)[-ex.count :]
                        top_prompt_positions = {prompt_positions[i] for i in top_idx}
                        recall = len(top_prompt_positions & set(prompt_needle_positions)) / max(ex.count, 1)
                        needle_mass = float(att[head, qpos, prompt_needle_positions].sum().item())
                        noise_mass = float(att[head, qpos, prompt_noise_positions].sum().item()) if prompt_noise_positions else 0.0
                        p = prompt_att / max(prompt_att.sum(), 1e-12)
                        entropy = float(-(p * np.log(p + 1e-12)).sum())
                        rows.append(
                            {
                                "model_type": model_type,
                                "loss_policy": loss_policy,
                                "seed": seed,
                                "checkpoint_step": "final",
                                "seq_len_eval": seq_len,
                                "count": ex.count,
                                "count_bin": count_bin(ex.count),
                                "layer": layer_idx,
                                "head": head,
                                "query_anchor": "ans_pos",
                                "correct_top1_rate": math.nan,
                                "diagonal_dominance": math.nan,
                                "needle_mass": needle_mass,
                                "noise_mass": noise_mass,
                                "needle_to_noise_ratio": needle_mass / max(noise_mass, 1e-12),
                                "entropy": entropy,
                                "top_n_recall": recall,
                            }
                        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def run_attention() -> pd.DataFrame:
    parts = []
    for seed in cfg["seeds"]:
        for model_type, loss_policy in ROUND4_CONDITIONS:
            if final_checkpoint_path(model_type, loss_policy, seed).exists():
                parts.append(attention_metrics_for_condition(model_type, loss_policy, seed))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.to_csv(TABLES_DIR / "round4_attention_head_metrics.csv", index=False)
    return df


def choose_top_thinking_head(attn_df: pd.DataFrame) -> dict[str, Any] | None:
    if attn_df.empty:
        return None
    sub = attn_df[attn_df["model_type"].eq("thinking") & attn_df["correct_top1_rate"].notna()]
    if sub.empty:
        return None
    agg = (
        sub.groupby(["loss_policy", "seed", "layer", "head", "query_anchor"], as_index=False)
        .agg(correct_top1_rate=("correct_top1_rate", "mean"), diagonal_dominance=("diagonal_dominance", "mean"), needle_mass=("needle_mass", "mean"))
        .sort_values(["correct_top1_rate", "diagonal_dominance", "needle_mass"], ascending=False)
    )
    return agg.iloc[0].to_dict()


def run_head_ablation(attn_df: pd.DataFrame) -> pd.DataFrame:
    top = choose_top_thinking_head(attn_df)
    if top is None:
        return pd.DataFrame()
    seed = int(top["seed"])
    loss_policy = top["loss_policy"]
    layer = int(top["layer"])
    head = int(top["head"])
    model = load_condition_model("thinking", loss_policy, seed)
    rows = []
    for seq_len in cfg["seq_lens_eval"]:
        examples = balanced_examples(seq_len, min(cfg["attention_examples_per_count"], cfg["test_examples_per_count"]), seed + 6000 + seq_len)
        base = pd.DataFrame(eval_thinking_generated(model, examples, vocab))
        abl = pd.DataFrame(eval_thinking_generated(model, examples, vocab, ablate_heads={(layer, head)}))
        for bin_name in ["low", "mid", "high"]:
            b = base[base["count_bin"] == bin_name]
            a = abl[abl["count_bin"] == bin_name]
            if b.empty or a.empty:
                continue
            rows.append(
                {
                    "model_type": "thinking",
                    "loss_policy": loss_policy,
                    "seed": seed,
                    "checkpoint_step": "final",
                    "seq_len_eval": seq_len,
                    "intervention_type": "single_head_ablation",
                    "layer": layer,
                    "head": head,
                    "query_anchor": top["query_anchor"],
                    "mask_type": "zero_head_output",
                    "count_bin": bin_name,
                    "baseline_final_accuracy": float(b["correct"].mean()),
                    "intervened_final_accuracy": float(a["correct"].mean()),
                    "delta_final_accuracy": float(a["correct"].mean() - b["correct"].mean()),
                    "baseline_trace_exact": float(b["trace_exact_rate"].mean()),
                    "intervened_trace_exact": float(a["trace_exact_rate"].mean()),
                    "delta_trace_exact": float(a["trace_exact_rate"].mean() - b["trace_exact_rate"].mean()),
                    "baseline_logit_margin": math.nan,
                    "intervened_logit_margin": math.nan,
                    "delta_logit_margin": math.nan,
                }
            )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "round4_head_ablation_results.csv", index=False)
    # attention masking remains a TODO; write an empty table with the requested schema.
    pd.DataFrame(columns=df.columns).to_csv(TABLES_DIR / "round4_attention_masking_results.csv", index=False)
    return df


if RUN_ROUNDS["round4"]:
    probe_df = run_probes()
    attn_df = run_attention()
    ablation_df = run_head_ablation(attn_df)
    display(Markdown("**Probe preview**"))
    display(probe_df.head() if not probe_df.empty else "No probe rows.")
    display(Markdown("**Attention preview**"))
    display(attn_df.head() if not attn_df.empty else "No attention rows.")
    display(Markdown("**Ablation preview**"))
    display(ablation_df.head() if not ablation_df.empty else "No ablation rows.")
else:
    probe_df = pd.DataFrame()
    attn_df = pd.DataFrame()
    ablation_df = pd.DataFrame()
        """
    ),
    md(
        r"""
## Plots and Summary Tables
        """
    ),
    code(
        r"""
def step_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    # This notebook evaluates final checkpoints by default. If checkpoint evals are
    # added later, this function will compute threshold steps from those rows.
    if df.empty or "checkpoint_step" not in df:
        return pd.DataFrame()
    rows = []
    group_cols = ["model_type", "loss_policy", "seed", "seq_len_eval", "count_bin", "eval_mode"]
    for keys, sub in df.groupby(group_cols):
        item = dict(zip(group_cols, keys))
        sub = sub.copy()
        sub["step_num"] = sub["checkpoint_step"].replace({"final": cfg["train_steps"]}).astype(int)
        sub = sub.sort_values("step_num")
        for threshold in [0.90, 0.95, 0.99]:
            hit = sub[sub["final_accuracy"] >= threshold]
            item[f"step_to_{int(threshold * 100)}"] = int(hit["step_num"].iloc[0]) if not hit.empty else math.nan
        item["auc_accuracy"] = float(sub["final_accuracy"].mean())
        rows.append(item)
    return pd.DataFrame(rows)


def write_round2_tables(eval_by_bin: pd.DataFrame) -> None:
    if eval_by_bin.empty:
        return
    summary = eval_by_bin.copy()
    thresholds = step_thresholds(summary)
    merged = summary.merge(thresholds, on=["model_type", "loss_policy", "seed", "seq_len_eval", "count_bin", "eval_mode"], how="left")
    merged.rename(
        columns={
            "final_accuracy": "final_accuracy_last",
            "final_answer_ce": "final_answer_loss_last",
            "trace_exact_rate": "trace_exact_last",
            "invalid_generation_rate": "invalid_generation_rate_last",
        },
        inplace=True,
    )
    keep = [
        "model_type",
        "loss_policy",
        "seed",
        "seq_len_eval",
        "count_bin",
        "eval_mode",
        "final_accuracy_last",
        "final_answer_loss_last",
        "trace_exact_last",
        "invalid_generation_rate_last",
        "step_to_90",
        "step_to_95",
        "step_to_99",
        "auc_accuracy",
    ]
    for col in keep:
        if col not in merged:
            merged[col] = math.nan
    merged[keep].to_csv(TABLES_DIR / "round2_summary_by_policy.csv", index=False)
    eval_by_count.to_csv(TABLES_DIR / "round2_final_checkpoint_by_count.csv", index=False)
    thresholds.to_csv(TABLES_DIR / "round2_step_to_thresholds.csv", index=False)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()


def make_plots() -> None:
    if not train_log.empty:
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=train_log, x="step", y="train_total_loss", hue="loss_policy", style="model_type", errorbar=None)
        plt.title("Round 1/2 training objective loss")
        plt.ylabel("weighted train_total_loss")
        save_fig(FIGURES_DIR / "round1_train_loss_by_step.png")
        save_fig(FIGURES_DIR / "round2_loss_policy_train_losses.png")

    if not eval_by_bin.empty:
        generated = eval_by_bin[eval_by_bin["eval_mode"].isin(["direct", "generated_trace"])]
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=generated, x="seq_len_eval", y="final_accuracy", hue="loss_policy", style="count_bin", markers=True, errorbar=None)
        plt.title("Round 1/2 final accuracy by eval length and count bin")
        plt.ylim(-0.03, 1.03)
        save_fig(FIGURES_DIR / "round1_final_accuracy_by_step_and_seq_len.png")
        save_fig(FIGURES_DIR / "round2_loss_policy_accuracy_by_step.png")

        plt.figure(figsize=(10, 5))
        sns.barplot(data=generated, x="loss_policy", y="final_accuracy", hue="seq_len_eval", errorbar=None)
        plt.xticks(rotation=35, ha="right")
        plt.ylim(0, 1)
        plt.title("Final length generalization by loss policy")
        save_fig(FIGURES_DIR / "round2_loss_policy_final_length_generalization.png")

        plt.figure(figsize=(8, 5))
        pivot = generated.pivot_table(index="count_bin", columns="seq_len_eval", values="final_accuracy", aggfunc="mean")
        sns.heatmap(pivot, vmin=0, vmax=1, annot=True, cmap="viridis")
        plt.title("Round 1 accuracy heatmap: count bin x seq_len")
        save_fig(FIGURES_DIR / "round1_accuracy_heatmap_count_x_seq_len.png")

        trace = generated[generated["model_type"].eq("thinking")]
        if not trace.empty:
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=trace, x="seq_len_eval", y="trace_exact_rate", hue="loss_policy", style="count_bin", markers=True, errorbar=None)
            plt.ylim(-0.03, 1.03)
            plt.title("Round 1/2 trace exact by eval length")
            save_fig(FIGURES_DIR / "round1_trace_metrics_by_seq_len.png")
            save_fig(FIGURES_DIR / "round2_loss_policy_trace_quality.png")

        thresh = pd.read_csv(TABLES_DIR / "round2_summary_by_policy.csv") if (TABLES_DIR / "round2_summary_by_policy.csv").exists() else pd.DataFrame()
        if not thresh.empty:
            plt.figure(figsize=(10, 4))
            sns.barplot(data=thresh, x="loss_policy", y="step_to_95", hue="seq_len_eval", errorbar=None)
            plt.xticks(rotation=35, ha="right")
            plt.title("Round 2 step to 95% accuracy")
            save_fig(FIGURES_DIR / "round2_loss_policy_step_to_95.png")

            plt.figure(figsize=(10, 4))
            sns.barplot(data=thresh, x="loss_policy", y="auc_accuracy", hue="seq_len_eval", errorbar=None)
            plt.xticks(rotation=35, ha="right")
            plt.title("Round 2 AUC accuracy over training/eval snapshots")
            save_fig(FIGURES_DIR / "round2_loss_policy_auc.png")

    if "eval_by_count" in globals() and not eval_by_count.empty:
        final_modes = eval_by_count[eval_by_count["eval_mode"].isin(["direct", "generated_trace"])]
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=final_modes, x="count", y="final_accuracy", hue="loss_policy", style="seq_len_eval", markers=True, errorbar=None)
        plt.ylim(-0.03, 1.03)
        plt.title("Round 1 accuracy by exact count")
        save_fig(FIGURES_DIR / "round1_accuracy_by_count_final.png")

    if "corrupt_df" in globals() and not corrupt_df.empty:
        plt.figure(figsize=(12, 5))
        sns.barplot(data=corrupt_df, x="corruption_type", y="correct_prompt_count", hue="loss_policy", errorbar=None)
        plt.xticks(rotation=35, ha="right")
        plt.ylim(0, 1)
        plt.title("Round 3 corrupted trace: prompt-count accuracy")
        save_fig(FIGURES_DIR / "round3_corruption_accuracy_by_type.png")

        follow_cols = [c for c in corrupt_df.columns if c.startswith("follows_")]
        follow_long = corrupt_df.melt(id_vars=["loss_policy", "corruption_type"], value_vars=follow_cols, var_name="rule", value_name="follows")
        plt.figure(figsize=(12, 5))
        sns.barplot(data=follow_long, x="corruption_type", y="follows", hue="rule", errorbar=None)
        plt.xticks(rotation=35, ha="right")
        plt.ylim(0, 1)
        plt.title("Round 3 follow-rule breakdown")
        save_fig(FIGURES_DIR / "round3_follow_rule_breakdown.png")
        for target, filename in [
            ("prompt_count", "round3_confusion_pred_vs_prompt_count.png"),
            ("trace_pair_count", "round3_confusion_pred_vs_trace_pair_count.png"),
            ("last_index_value", "round3_confusion_pred_vs_last_index.png"),
        ]:
            sub = corrupt_df.dropna(subset=[target])
            if not sub.empty:
                mat = pd.crosstab(sub[target], sub["pred_count"], normalize="index")
                plt.figure(figsize=(7, 5))
                sns.heatmap(mat, vmin=0, vmax=1, cmap="viridis", annot=PRESET == "debug", fmt=".2f")
                plt.title(f"Prediction vs {target}")
                save_fig(FIGURES_DIR / filename)

    if "probe_df" in globals() and not probe_df.empty:
        sub = probe_df[probe_df["target_type"].isin(["final_count", "prefix_count"])]
        plt.figure(figsize=(12, 5))
        sns.barplot(data=sub, x="anchor_type", y="test_accuracy", hue="layer", errorbar=None)
        plt.xticks(rotation=35, ha="right")
        plt.ylim(0, 1)
        plt.title("Round 4 probe accuracy by layer and anchor")
        save_fig(FIGURES_DIR / "round4_probe_accuracy_layer_by_anchor.png")

        plt.figure(figsize=(12, 5))
        sns.barplot(data=sub, x="anchor_type", y="r2", hue="layer", errorbar=None)
        plt.xticks(rotation=35, ha="right")
        plt.title("Round 4 probe R2 by layer and anchor")
        save_fig(FIGURES_DIR / "round4_probe_r2_layer_by_anchor.png")

        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=sub, x="position_only_accuracy", y="test_accuracy", hue="model_type", style="target_type")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.title("Probe vs position-only baseline")
        save_fig(FIGURES_DIR / "round4_probe_vs_position_baseline.png")

    if "attn_df" in globals() and not attn_df.empty:
        attn_summary = (
            attn_df.groupby(["model_type", "loss_policy", "layer", "head", "query_anchor"], as_index=False)
            .agg(correct_top1_rate=("correct_top1_rate", "mean"), diagonal_dominance=("diagonal_dominance", "mean"), needle_mass=("needle_mass", "mean"), top_n_recall=("top_n_recall", "mean"))
        )
        plt.figure(figsize=(12, 5))
        plot_col = "correct_top1_rate"
        sns.barplot(data=attn_summary.sort_values(plot_col, ascending=False).head(24), x="query_anchor", y=plot_col, hue="loss_policy", errorbar=None)
        plt.xticks(rotation=35, ha="right")
        plt.ylim(0, 1)
        plt.title("Round 4 attention head leaderboard")
        save_fig(FIGURES_DIR / "round4_attention_head_leaderboard.png")

        thinking = attn_df[attn_df["model_type"].eq("thinking")]
        if not thinking.empty:
            mat = thinking.pivot_table(index="layer", columns="head", values="correct_top1_rate", aggfunc="mean")
            plt.figure(figsize=(6, 4.5))
            sns.heatmap(mat, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="viridis")
            plt.title("Thinking trace-to-prompt retrieval")
            save_fig(FIGURES_DIR / "round4_thinking_trace_to_prompt_heatmap_best_head.png")

        non = attn_df[attn_df["model_type"].eq("non_thinking")]
        if not non.empty:
            mat = non.pivot_table(index="layer", columns="head", values="top_n_recall", aggfunc="mean")
            plt.figure(figsize=(6, 4.5))
            sns.heatmap(mat, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="viridis")
            plt.title("Non-thinking <Ans> top-n retrieval")
            save_fig(FIGURES_DIR / "round4_nonthinking_ans_to_prompt_attention.png")

        plt.figure(figsize=(10, 5))
        sns.barplot(data=attn_df, x="count_bin", y="needle_mass", hue="loss_policy", errorbar=None)
        plt.title("Round 4 attention needle mass by count bin")
        save_fig(FIGURES_DIR / "round4_attention_metrics_by_count_bin.png")

    if "ablation_df" in globals() and not ablation_df.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=ablation_df, x="count_bin", y="delta_final_accuracy", hue="loss_policy", errorbar=None)
        plt.title("Round 4 head ablation effects on final accuracy")
        save_fig(FIGURES_DIR / "round4_head_ablation_effects.png")
        plt.figure(figsize=(8, 4))
        sns.barplot(data=ablation_df, x="count_bin", y="delta_trace_exact", hue="loss_policy", errorbar=None)
        plt.title("Round 4 head ablation effects on trace exact")
        save_fig(FIGURES_DIR / "round4_attention_masking_effects.png")


write_round2_tables(eval_by_bin)
make_plots()
        """
    ),
    md(
        r"""
## Self-contained HTML Report and Summary JSON
        """
    ),
    code(
        r"""
def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def html_table(path: Path, max_rows: int = 20) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "<p>No table found.</p>"
    df = pd.read_csv(path)
    return df.head(max_rows).to_html(index=False, escape=True)


def fig_block(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    return f'''
    <figure>
      <img src="{image_uri(path)}" alt="{html.escape(path.name)}">
      <figcaption><strong>{html.escape(path.name)}</strong><br>{html.escape(caption)}</figcaption>
    </figure>
    '''


def best_condition_from_eval(eval_by_bin: pd.DataFrame, model_type: str) -> str:
    if eval_by_bin.empty:
        return ""
    sub = eval_by_bin[eval_by_bin["model_type"].eq(model_type)]
    if sub.empty:
        return ""
    agg = sub.groupby(["loss_policy"], as_index=False)["final_accuracy"].mean().sort_values("final_accuracy", ascending=False)
    return str(agg.iloc[0]["loss_policy"])


summary = {
    "run_name": RUN_DIR.name,
    "preset": PRESET,
    "train_seq_len": cfg["train_seq_len"],
    "seq_lens_eval": cfg["seq_lens_eval"],
    "count_range": [cfg["count_min"], cfg["count_max"]],
    "seeds": cfg["seeds"],
    "best_nonthinking_condition": best_condition_from_eval(eval_by_bin, "non_thinking") if "eval_by_bin" in globals() else "",
    "best_thinking_condition": best_condition_from_eval(eval_by_bin, "thinking") if "eval_by_bin" in globals() else "",
    "round1_main_takeaway": "Compare non_final_only vs think_trace_and_final across longer eval lengths.",
    "round2_main_takeaway": "Loss masks trade off final readout, trace generation, and length generalization.",
    "round3_main_takeaway": "Corrupted-trace modes diagnose whether final answer follows prompt count, trace length, or index shortcuts.",
    "round4_main_takeaway": "Probe/attention are diagnostic; head ablation is the causal evidence available in this notebook.",
    "limitations": [
        "All data are symbolic.",
        "Counts are limited to 1..10.",
        "The trace exposes count length, so final readout may exploit trace length or last index.",
        "Probe decodability is not causal evidence.",
        "Attention masking is written as a TODO table unless extended beyond single-head ablation.",
    ],
}
with (RUN_DIR / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)


def build_report() -> Path:
    css = '''
    body { margin:0; background:#f5f7fb; color:#172033; font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; line-height:1.6; }
    .shell { max-width:1400px; margin:0 auto; padding:28px 22px 60px; }
    header { background:linear-gradient(135deg,#0f172a,#1d4ed8); color:white; border-radius:16px; padding:30px 34px; }
    header h1 { margin:0 0 8px; font-size:2.6rem; }
    section { background:white; border:1px solid #dbe3ef; border-radius:14px; padding:22px; margin-top:22px; box-shadow:0 8px 24px rgba(15,23,42,.05); }
    h2 { margin-top:0; }
    code { background:#eef2ff; color:#1e3a8a; border-radius:5px; padding:1px 5px; }
    table { border-collapse:collapse; width:100%; font-size:.9rem; }
    th,td { border-bottom:1px solid #dbe3ef; padding:7px 8px; text-align:left; white-space:nowrap; }
    th { background:#f8fafc; }
    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
    figure { margin:0 0 18px; border:1px solid #dbe3ef; border-radius:12px; background:#fff; overflow:hidden; }
    img { display:block; width:100%; max-height:420px; object-fit:contain; background:white; }
    figcaption { border-top:1px solid #dbe3ef; padding:10px 12px; color:#475569; }
    @media (max-width:900px) { .grid { grid-template-columns:1fr; } header h1 { font-size:2rem; } }
    '''
    figs = {p.name: p for p in sorted(FIGURES_DIR.glob("*.png"))}
    round1 = [
        ("round1_train_loss_by_step.png", "Training objective loss by step; compare model type and loss policy."),
        ("round1_final_accuracy_by_step_and_seq_len.png", "Hard eval: final count accuracy across longer sequence lengths and count bins."),
        ("round1_accuracy_by_count_final.png", "Final checkpoint accuracy by exact count."),
        ("round1_accuracy_heatmap_count_x_seq_len.png", "Accuracy heatmap by count bin and eval sequence length."),
        ("round1_trace_metrics_by_seq_len.png", "Thinking trace quality across eval sequence length."),
    ]
    round2 = [
        ("round2_loss_policy_train_losses.png", "Loss-mask ablation training losses."),
        ("round2_loss_policy_accuracy_by_step.png", "Loss policy comparison for final accuracy."),
        ("round2_loss_policy_step_to_95.png", "Steps required to reach 95% accuracy where available."),
        ("round2_loss_policy_auc.png", "AUC-style summary of accuracy over available checkpoints."),
        ("round2_loss_policy_trace_quality.png", "Trace quality by loss policy."),
        ("round2_loss_policy_final_length_generalization.png", "Final length generalization by policy."),
    ]
    round3 = [
        ("round3_corruption_accuracy_by_type.png", "Whether the model keeps answering the prompt count under corrupted traces."),
        ("round3_follow_rule_breakdown.png", "Which rule the final answer follows: prompt, trace length, last index, max index, or marker count."),
        ("round3_confusion_pred_vs_prompt_count.png", "Predicted count vs true prompt count."),
        ("round3_confusion_pred_vs_trace_pair_count.png", "Predicted count vs corrupted trace pair count."),
        ("round3_confusion_pred_vs_last_index.png", "Predicted count vs last index token."),
    ]
    round4 = [
        ("round4_probe_accuracy_layer_by_anchor.png", "Hidden-state probe accuracy by layer and anchor."),
        ("round4_probe_r2_layer_by_anchor.png", "Ridge R2 for numeric count probes."),
        ("round4_probe_vs_position_baseline.png", "Probe accuracy compared with position-only baseline."),
        ("round4_attention_head_leaderboard.png", "Attention head leaderboard."),
        ("round4_thinking_trace_to_prompt_heatmap_best_head.png", "Thinking trace-to-prompt retrieval head map."),
        ("round4_nonthinking_ans_to_prompt_attention.png", "Non-thinking final-answer retrieval map."),
        ("round4_attention_metrics_by_count_bin.png", "Attention metrics by count bin."),
        ("round4_head_ablation_effects.png", "Single-head ablation effect on final accuracy."),
        ("round4_attention_masking_effects.png", "Single-head ablation effect on trace exact; attention masking is TODO unless extended."),
    ]

    def blocks(specs):
        return "".join(fig_block(figs.get(name, Path("__missing__")), cap) for name, cap in specs)

    html_text = f'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Trace Count v3 Report</title><style>{css}</style></head>
<body><div class="shell">
<header><h1>Trace Count v3 Report</h1><p>Run: <code>{html.escape(RUN_DIR.name)}</code> · preset: <code>{html.escape(PRESET)}</code> · train seq len: <code>{cfg['train_seq_len']}</code> · eval lengths: <code>{cfg['seq_lens_eval']}</code></p></header>
<section><h2>Interpretation Summary</h2>
<h3>Behavior</h3><p>Use Round 1/2 figures to judge whether thinking outperforms non-thinking at longer eval lengths and which count bin breaks first.</p>
<h3>Trace</h3><p>Trace exact, marker recall, and index accuracy are reported separately from final count accuracy.</p>
<h3>Corrupted Trace</h3><p>Round 3 classifies whether predictions follow prompt count, trace pair count, last index, max index, or marker count.</p>
<h3>Hidden States</h3><p>Probe accuracy must be compared against position-only and trace-length baselines; it is diagnostic, not causal.</p>
<h3>Attention</h3><p>Thinking retrieval heads are evaluated by correct top-1, diagonal dominance, needle mass, and noise mass. Non-thinking uses final <code>&lt;Ans&gt;</code> retrieval.</p>
<h3>Causality</h3><p>Only head ablation or masking can support causal claims. This notebook implements single-head ablation and writes attention masking as TODO unless extended.</p>
</section>
<section><h2>Config</h2>{pd.DataFrame([{'field': k, 'value': str(v)} for k, v in cfg.items()]).to_html(index=False, escape=True)}</section>
<section><h2>Round 1: Hard Evaluation</h2><div class="grid">{blocks(round1)}</div></section>
<section><h2>Round 2: Loss-Mask Ablation</h2>{html_table(TABLES_DIR / 'round2_summary_by_policy.csv')}<div class="grid">{blocks(round2)}</div></section>
<section><h2>Round 3: Corrupted Trace</h2>{html_table(TABLES_DIR / 'round3_follow_rule_summary.csv')}<div class="grid">{blocks(round3)}</div></section>
<section><h2>Round 4: Probe, Attention, Causality</h2>{html_table(TABLES_DIR / 'round4_probe_results.csv')}<div class="grid">{blocks(round4)}</div></section>
<section><h2>Limitations</h2><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in summary['limitations'])}</ul></section>
</div></body></html>'''
    out = RUN_DIR / "syn_v3_report.html"
    out.write_text(html_text, encoding="utf-8")
    return out


report_path = build_report()
display(Markdown(f"**Report:** `{report_path}`"))
display(Markdown(f"**Summary JSON:** `{RUN_DIR / 'summary.json'}`"))
print("FINAL_REPORT", report_path)
        """
    ),
    md(
        r"""
## Save Results to Google Drive

This cell copies the run directory to your preferred Drive location. It is safe
to rerun after the notebook completes.
        """
    ),
    code(
        r"""
DRIVE_RESULTS_ROOT = Path("/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/Synthetic_CoT_NiaH_Count/colab_results")

if IN_COLAB and SAVE_TO_DRIVE_AT_END:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    target = DRIVE_RESULTS_ROOT / RUN_DIR.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(RUN_DIR, target)
    nb_src = Path("notebooks/Trace_Count_v3_Colab.ipynb")
    if nb_src.exists():
        (target / "notebooks").mkdir(exist_ok=True)
        shutil.copy2(nb_src, target / "notebooks" / nb_src.name)
    print("Saved to Drive:", target)
else:
    print("Not in Colab or SAVE_TO_DRIVE_AT_END=False; skipped Drive copy.")
        """
    ),
    md(
        r"""
## Optional: Commit and Push to GitHub

Set `PUSH_TO_GITHUB = True` only after reviewing the generated files. This cell
uses the repo remote and the current branch.
        """
    ),
    code(
        r"""
PUSH_TO_GITHUB = False
GIT_COMMIT_MESSAGE = "Add Trace Count v3 Colab notebook"

if PUSH_TO_GITHUB:
    subprocess.run(["git", "status", "--short"], check=False)
    subprocess.run(["git", "add", "notebooks/Trace_Count_v3_Colab.ipynb", "notebooks/pipeline_v3_codex_prompt.md", "scripts/build_v3_notebook.py"], check=True)
    subprocess.run(["git", "commit", "-m", GIT_COMMIT_MESSAGE], check=True)
    subprocess.run(["git", "push"], check=True)
else:
    print("PUSH_TO_GITHUB=False; skipped git commit/push.")
        """
    ),
]


def main() -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "colab": {"name": "Trace_Count_v3_Colab.ipynb", "provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
