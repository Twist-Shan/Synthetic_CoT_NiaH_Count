from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch


SUPPORTED_POSITION_ENCODINGS = ("rope",)
SUPPORTED_MODES = ("nonthinking", "thinking")
ALL_MODEL_VARIANTS = tuple(
    f"{position}/{mode}"
    for position in SUPPORTED_POSITION_ENCODINGS
    for mode in SUPPORTED_MODES
)
REFERENCE_MODEL_VARIANTS = ("rope/nonthinking", "rope/thinking")
SUPPORTED_COUNT_TOKENIZATIONS = ("atomic", "digitwise")
SUPPORTED_VERSIONS = ("v20", "v21")
SUPPORTED_TRAINING_COUNT_DISTRIBUTIONS = ("natural", "uniform")


def _float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class V20Config:
    """Shared v20/v21 configuration.

    v20 and v21 are deliberately paired.  The only task-grammar difference is
    ``count_tokenization``: v20 uses one atomic token per integer, whereas v21
    renders every trace index and final answer with shared decimal digit tokens.
    """

    version: str = "v20"
    preset: str = "debug"
    seed: int = 1234
    seq_len: int = 256
    needle_set_size: int = 3
    needle_pool_size: int = 100
    needle_pool_frequency_threshold: float = 0.12
    needle_pool_frequency_bins: int = 20
    needle_pool_seed: int | None = None
    count_max_threshold: int = 30
    task_occurrence_ratio: float = 1.0
    training_count_distribution: str = "natural"
    corpus_train_fraction: float = 0.80
    corpus_validation_fraction: float = 0.10
    candidate_filter_max_attempts: int = 100_000
    shuffle_needle_set_order: bool = True
    # v20/v21 are intentionally RoPE-only so the comparison isolates output
    # tokenization rather than position encoding.
    position_encodings: tuple[str, ...] = ("rope",)
    enabled_model_variants: tuple[str, ...] = REFERENCE_MODEL_VARIANTS

    train_steps: int = 10_000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    warmup_steps: int = 500
    grad_clip: float = 1.0
    precision: str = "bf16"
    log_every: int = 50
    eval_every: int = 500
    ar_eval_every: int = 1_000
    # Dense model-only snapshots are the scientific checkpoints used for phase
    # transition analysis.  Full optimizer/RNG recovery checkpoints are less
    # frequent and are overwritten except for the objective boundary and final.
    checkpoint_every: int = 100
    recovery_every: int = 500
    snapshot_shard_every: int = 500
    snapshot_dtype: str = "float16"
    eval_examples_per_count: int = 10
    final_examples_per_count: int = 50
    ar_examples_per_count: int = 2
    permutation_examples_per_count: int = 1
    max_steps_for_language_pred: int = 1_500
    final_count_loss_weight: float = 1.0
    cot_trace_loss_weight: float = 1.0

    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    n_inner: int = 1024
    n_positions: int = 384
    max_relative_distance: int = 256
    rpe_max_update: bool = False
    rope_base: float = 10_000.0

    attention_examples_per_count: int = 10
    state_train_examples_per_count: int = 20
    state_eval_examples_per_count: int = 10
    phase_examples_per_count: int = 1
    phase_head_selection_examples_per_count: int = 2
    phase_cloud_steps: tuple[int, ...] = (
        0, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 5_000, 7_000, 10_000
    )
    analysis_batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model/data interface metadata. These are deliberately immutable in v20.
    noise_source: str = "shakespeare_char"
    task_type: str = "target_character_set"
    loss_scope: str = "all_sequence"
    query_layout: str = "query_first"
    count_tokenization: str = "atomic"
    use_sdpa: bool = True

    @property
    def count_min(self) -> int:
        return 1

    @property
    def count_max(self) -> int:
        """Compatibility alias; count_max_threshold is the only stored setting."""

        return int(self.count_max_threshold)

    @property
    def effective_needle_pool_seed(self) -> int:
        return int(self.seed + 20_000 if self.needle_pool_seed is None else self.needle_pool_seed)

    @property
    def corpus_test_fraction(self) -> float:
        return 1.0 - float(self.corpus_train_fraction) - float(self.corpus_validation_fraction)

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(mode for _, mode in self.model_variants))

    @property
    def model_variants(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (value.split("/", 1)[0], value.split("/", 1)[1])
            for value in self.enabled_model_variants
        )

    @property
    def max_render_len(self) -> int:
        # BOS + query + prompt + Think + (index digits + marker)*n +
        # close + Ans + final-count digits + EOS.  Query-first is fixed in both
        # versions, and digitwise v21 remains within the 384-token context.
        def width(value: int) -> int:
            return 1 if self.count_tokenization == "atomic" else len(str(int(value)))
        prefix = 1 + (2 + self.needle_set_size) + self.seq_len
        direct = prefix + 1 + width(self.count_max_threshold) + 1
        trace = sum(width(index) + 1 for index in range(1, self.count_max_threshold + 1))
        thinking = prefix + 1 + trace + 1 + 1 + width(self.count_max_threshold) + 1
        return max(direct, thinking)

    @property
    def count_bins(self) -> tuple[tuple[int, int], ...]:
        return ((1, self.count_max_threshold),)

    def count_bin(self, count: int) -> str:
        value = int(count)
        if not 1 <= value <= self.count_max_threshold:
            raise ValueError(f"count {value} is outside 1..{self.count_max_threshold}")
        return f"1-{self.count_max_threshold}"

    def validate(self) -> None:
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(f"V20Config.version must be one of {SUPPORTED_VERSIONS}")
        expected_tokenization = "atomic" if self.version == "v20" else "digitwise"
        if self.count_tokenization != expected_tokenization:
            raise ValueError(
                f"{self.version} requires count_tokenization={expected_tokenization!r}"
            )
        if self.query_layout != "query_first":
            raise ValueError("v20/v21 require query-first sequence construction")
        if self.needle_set_size != 3:
            raise ValueError("v20/v21 require exactly three distinct characters per needle set")
        if self.needle_pool_size <= 0 or self.needle_pool_frequency_bins <= 0:
            raise ValueError("needle pool size and number of bins must be positive")
        if not 0.0 < self.needle_pool_frequency_threshold <= 1.0:
            raise ValueError("needle_pool_frequency_threshold must be in (0, 1]")
        if not 1 <= self.count_max_threshold <= self.seq_len:
            raise ValueError("count_max_threshold must satisfy 1 <= threshold <= seq_len")
        if not 0.0 <= self.task_occurrence_ratio <= 1.0:
            raise ValueError("task_occurrence_ratio must be in [0, 1]")
        if self.training_count_distribution not in SUPPORTED_TRAINING_COUNT_DISTRIBUTIONS:
            raise ValueError(
                "training_count_distribution must be one of "
                f"{SUPPORTED_TRAINING_COUNT_DISTRIBUTIONS}"
            )
        if self.training_count_distribution != "natural" and self.task_occurrence_ratio != 1.0:
            raise ValueError(
                "controlled count distributions require task_occurrence_ratio=1 so the "
                "requested example distribution is unambiguous"
            )
        if self.corpus_train_fraction <= 0 or self.corpus_validation_fraction <= 0:
            raise ValueError("corpus train and validation fractions must be positive")
        if self.corpus_train_fraction + self.corpus_validation_fraction >= 1:
            raise ValueError("train + validation fractions must be less than one")
        if self.candidate_filter_max_attempts <= 0:
            raise ValueError("candidate_filter_max_attempts must be positive")
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least two")
        if (self.n_layer, self.n_head, self.n_embd, self.n_inner) != (4, 4, 256, 1024):
            raise ValueError("v20/v21 require 4 layers, 4 heads, d_model=256, MLP=1024")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.max_render_len > self.n_positions:
            raise ValueError(
                f"max rendered length {self.max_render_len} exceeds n_positions={self.n_positions}"
            )
        if type(self.rpe_max_update) is not bool:
            raise ValueError("rpe_max_update must be a boolean")
        if self.rpe_max_update:
            raise ValueError("rpe_max_update is inapplicable because v20/v21 are RoPE-only")
        if type(self.max_relative_distance) is not int or self.max_relative_distance <= 0:
            raise ValueError("max_relative_distance must be a positive integer")
        if self.rpe_max_update and self.max_relative_distance != self.max_render_len - 1:
            raise ValueError(
                "rpe_max_update requires max_relative_distance == max_render_len - 1"
            )
        if not self.position_encodings:
            raise ValueError("at least one position encoding is required")
        invalid = sorted(set(self.position_encodings) - set(SUPPORTED_POSITION_ENCODINGS))
        if invalid:
            raise ValueError(f"unsupported position encodings: {invalid}")
        if not self.enabled_model_variants:
            raise ValueError("enabled_model_variants must contain at least one model")
        if len(set(self.enabled_model_variants)) != len(self.enabled_model_variants):
            raise ValueError("enabled_model_variants must not contain duplicates")
        invalid_variants = sorted(set(self.enabled_model_variants) - set(ALL_MODEL_VARIANTS))
        if invalid_variants:
            raise ValueError(f"unsupported model variants: {invalid_variants}")
        active_positions = tuple(
            position
            for position in SUPPORTED_POSITION_ENCODINGS
            if any(value.startswith(f"{position}/") for value in self.enabled_model_variants)
        )
        if self.position_encodings != active_positions:
            raise ValueError(
                "position_encodings must equal the position encodings used by "
                "enabled_model_variants"
            )
        if self.noise_source != "shakespeare_char" or self.task_type != "target_character_set":
            raise ValueError("v20/v21 require the Shakespeare target-character-set task")
        if self.loss_scope != "all_sequence":
            raise ValueError("v20/v21 require all-sequence next-token loss metadata")
        if self.precision not in {"float32", "bf16"}:
            raise ValueError("precision must be float32 or bf16")
        if self.snapshot_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("snapshot_dtype must be float16, bfloat16, or float32")
        if type(self.use_sdpa) is not bool:
            raise ValueError("use_sdpa must be a boolean")
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        for name in ("final_count_loss_weight", "cot_trace_loss_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if type(self.max_steps_for_language_pred) is not int or self.max_steps_for_language_pred < 0:
            raise ValueError("max_steps_for_language_pred must be a nonnegative integer")
        if self.max_steps_for_language_pred < self.train_steps and self.task_occurrence_ratio == 0:
            raise ValueError(
                "task_occurrence_ratio must be positive when task-output-only training is scheduled"
            )
        if type(self.checkpoint_every) is not int or self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be a positive integer")
        if type(self.recovery_every) is not int or self.recovery_every <= 0:
            raise ValueError("recovery_every must be a positive integer")
        if type(self.snapshot_shard_every) is not int or self.snapshot_shard_every <= 0:
            raise ValueError("snapshot_shard_every must be a positive integer")
        if self.snapshot_shard_every % self.checkpoint_every:
            raise ValueError("snapshot_shard_every must be divisible by checkpoint_every")
        if self.recovery_every % self.checkpoint_every:
            raise ValueError("recovery_every must be divisible by checkpoint_every")
        if not (0 <= self.adam_beta1 < 1 and 0 <= self.adam_beta2 < 1):
            raise ValueError("Adam betas must be in [0, 1)")
        for name in (
            "train_steps",
            "batch_size",
            "log_every",
            "eval_every",
            "ar_eval_every",
            "eval_examples_per_count",
            "final_examples_per_count",
            "phase_examples_per_count",
            "phase_head_selection_examples_per_count",
            "permutation_examples_per_count",
            "analysis_batch_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["position_encodings"] = list(self.position_encodings)
        result["enabled_model_variants"] = list(self.enabled_model_variants)
        result["count_max"] = self.count_max
        result["count_max_alias"] = "read-only alias of count_max_threshold"
        result["effective_needle_pool_seed"] = self.effective_needle_pool_seed
        result["corpus_test_fraction"] = self.corpus_test_fraction
        if self.max_steps_for_language_pred < self.train_steps:
            result["training_objective"] = (
                "teacher-forced weighted next-token cross-entropy over every non-padding "
                f"token through step {self.max_steps_for_language_pred}; from step "
                f"{self.max_steps_for_language_pred + 1}, task-output targets only, "
                "starting inclusively at <Ans> for nonthinking and <Think> for thinking"
            )
            result["training_loss_schedule"] = (
                f"steps 1-{self.max_steps_for_language_pred}: all_sequence; steps "
                f"{self.max_steps_for_language_pred + 1}-{self.train_steps}: task_output"
            )
        else:
            result["training_objective"] = (
                "teacher-forced weighted next-token cross-entropy over every non-padding "
                "token for all configured training steps"
            )
            result["training_loss_schedule"] = (
                f"steps 1-{self.train_steps}: all_sequence; task-output switch is after training"
            )
        result["task_output_scope"] = {
            "nonthinking": "<Ans> through <EOS>, inclusive",
            "thinking": "<Think> through <EOS>, inclusive",
        }
        result["task_occurrence_ratio_definition"] = (
            "example-level probability of formatting a training corpus window as a counting task"
        )
        result["sequence_layout"] = "query_first"
        result["sequence_templates"] = {
            "nonthinking": "<BOS> query[5] data[256] <Ans> number <EOS>",
            "thinking": "<BOS> query[5] data[256] <Think> (number marker)*n </Think> <Ans> number <EOS>",
        }
        result["checkpoint_policy"] = {
            "analysis_snapshot_every": self.checkpoint_every,
            "optimizer_recovery_every": self.recovery_every,
            "snapshot_shard_every": self.snapshot_shard_every,
            "snapshot_dtype": self.snapshot_dtype,
        }
        return result


def preset_config(preset: str = "debug", **overrides: Any) -> V20Config:
    cfg = V20Config(preset="main")
    if preset == "debug":
        cfg = replace(
            cfg,
            preset="debug",
            seq_len=48,
            count_max_threshold=4,
            n_positions=96,
            max_relative_distance=96,
            train_steps=6,
            batch_size=4,
            warmup_steps=2,
            log_every=1,
            eval_every=3,
            ar_eval_every=3,
            checkpoint_every=3,
            recovery_every=3,
            snapshot_shard_every=3,
            precision="float32",
            eval_examples_per_count=2,
            ar_examples_per_count=1,
            permutation_examples_per_count=1,
            attention_examples_per_count=1,
            state_train_examples_per_count=2,
            state_eval_examples_per_count=1,
            analysis_batch_size=8,
            phase_examples_per_count=1,
            phase_head_selection_examples_per_count=1,
            phase_cloud_steps=(0, 3, 6),
        )
    elif preset != "main":
        raise ValueError(f"unknown preset: {preset}")
    unknown = sorted(set(overrides) - set(cfg.__dataclass_fields__))
    if unknown:
        raise TypeError(f"unknown V20Config overrides: {unknown}")
    if "enabled_model_variants" in overrides:
        overrides["enabled_model_variants"] = tuple(overrides["enabled_model_variants"])
        if "position_encodings" in overrides:
            overrides["position_encodings"] = tuple(overrides["position_encodings"])
        if "position_encodings" not in overrides:
            overrides["position_encodings"] = tuple(
                position
                for position in SUPPORTED_POSITION_ENCODINGS
                if any(
                    value.startswith(f"{position}/")
                    for value in overrides["enabled_model_variants"]
                )
            )
    elif "position_encodings" in overrides:
        overrides["position_encodings"] = tuple(overrides["position_encodings"])
        overrides["enabled_model_variants"] = tuple(
            f"{position}/{mode}"
            for position in overrides["position_encodings"]
            for mode in SUPPORTED_MODES
        )
    cfg = replace(cfg, **overrides)
    if cfg.rpe_max_update:
        cfg = replace(cfg, max_relative_distance=cfg.max_render_len - 1)
    cfg.validate()
    return cfg


def config_from_dict(values: dict[str, Any]) -> V20Config:
    data = dict(values)
    legacy_loss_schedule = "max_steps_for_language_pred" not in data
    alias = data.pop("count_max", None)
    threshold = int(data["count_max_threshold"])
    if alias is not None and int(alias) != threshold:
        raise ValueError("serialized count_max alias disagrees with count_max_threshold")
    for derived in (
        "count_max_alias",
        "effective_needle_pool_seed",
        "corpus_test_fraction",
        "training_objective",
        "training_loss_schedule",
        "task_output_scope",
        "task_occurrence_ratio_definition",
        "sequence_layout",
        "sequence_templates",
        "checkpoint_policy",
    ):
        data.pop(derived, None)
    data["position_encodings"] = tuple(data["position_encodings"])
    if "enabled_model_variants" in data:
        data["enabled_model_variants"] = tuple(data["enabled_model_variants"])
    else:
        data["enabled_model_variants"] = tuple(
            f"{position}/{mode}"
            for position in data["position_encodings"]
            for mode in SUPPORTED_MODES
        )
    data.setdefault("final_count_loss_weight", 1.0)
    data.setdefault("cot_trace_loss_weight", 1.0)
    data.setdefault("weight_decay", 0.01)
    # Retained only to reject accidental RPE-era configs with a clear message.
    data.setdefault("rpe_max_update", False)
    # Before revision 5, the main cadence was 1,000 steps. Preserve that value
    # when loading a rare hand-written legacy config that omitted the field.
    data.setdefault("checkpoint_every", 100)
    data.setdefault("recovery_every", 500)
    data.setdefault("snapshot_shard_every", 500)
    data.setdefault("snapshot_dtype", "float16")
    data.setdefault("final_examples_per_count", 50)
    data.setdefault("ar_examples_per_count", 2)
    data.setdefault("permutation_examples_per_count", 1)
    data.setdefault("phase_examples_per_count", 1)
    data.setdefault("phase_head_selection_examples_per_count", 2)
    data.setdefault("phase_cloud_steps", (0, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 5_000, 7_000, 10_000))
    data["phase_cloud_steps"] = tuple(data["phase_cloud_steps"])
    data.setdefault("query_layout", "query_first")
    data.setdefault("count_tokenization", "atomic" if data.get("version", "v20") == "v20" else "digitwise")
    data.setdefault("use_sdpa", True)
    data.setdefault("training_count_distribution", "natural")
    if legacy_loss_schedule:
        data["max_steps_for_language_pred"] = int(data["train_steps"])
    cfg = V20Config(**data)
    if cfg.rpe_max_update:
        cfg = replace(cfg, max_relative_distance=cfg.max_render_len - 1)
    cfg.validate()
    return cfg


def default_run_name(cfg: V20Config) -> str:
    variants = "-".join(value.replace("nonthinking", "nt").replace("thinking", "t") for value in cfg.enabled_model_variants)
    eval_size = cfg.eval_examples_per_count * cfg.count_max_threshold
    rpe_distance_tag = f"_rped{cfg.max_relative_distance}" if cfg.rpe_max_update else ""
    schedule_tag = (
        "allseq-taskout"
        if cfg.max_steps_for_language_pred < cfg.train_steps
        else "all_sequence"
    )
    return (
        f"{cfg.version}_{cfg.preset}_L{cfg.seq_len}_pool{cfg.needle_pool_size}x{cfg.needle_set_size}_"
        f"pf{_float_tag(cfg.needle_pool_frequency_threshold)}_count1-{cfg.count_max_threshold}{rpe_distance_tag}_"
        f"taskr{_float_tag(cfg.task_occurrence_ratio)}_wd{_float_tag(cfg.weight_decay)}_"
        f"countdist-{cfg.training_count_distribution}_"
        f"fcw{_float_tag(cfg.final_count_loss_weight)}_"
        f"cotw{_float_tag(cfg.cot_trace_loss_weight)}_langsteps{cfg.max_steps_for_language_pred}_"
        f"steps{cfg.train_steps}_snap{cfg.checkpoint_every}_recover{cfg.recovery_every}_"
        f"evaln{eval_size}_{variants.replace('/', '-')}_{cfg.count_tokenization}_"
        f"query-first_{schedule_tag}_seed{cfg.seed}"
    )


def prepare_run_dir(out_root: str | Path, cfg: V20Config, run_name: str | None = None) -> Path:
    path = Path(out_root) / (run_name or default_run_name(cfg))
    for subdir in ("tables", "figures", "checkpoints", "analysis", "logs", "data"):
        (path / subdir).mkdir(parents=True, exist_ok=True)
    return path
