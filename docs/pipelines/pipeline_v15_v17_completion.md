# v15-v17: v10-style completion-only counting experiments

## Shared controlled setting

v15-v17 use the v10-capacity decoder core: four causal Transformer layers, four
attention heads per layer, hidden width 256, MLP width 1024, tied token
embedding/unembedding, prompt length 256, and exact counts 1-30. Every
position-encoding x output-mode combination is initialized and trained as an
independent model. Non-thinking and thinking therefore never share weights.

The two output formats are:

```text
non-thinking: <BOS> [task prefix] prompt <Ans> <n> <EOS>
thinking:     <BOS> [task prefix] prompt <Think> <1> M1 ... <n> Mn </Think> <Ans> <n> <EOS>
```

Trace indices and final answers share the same numeric tokens `<1>...<30>`.

### v10-style completion-only causal objective

The complete gold sequence is present as teacher-forced causal context, but the
task prefix and prompt/haystack do not contribute to the loss.

For non-thinking, only the final answer and EOS are supervised:

```text
L_non = -1/2 [log p(<n> | prefix through <Ans>)
              + log p(<EOS> | prefix through <n>)].
```

For thinking, supervision starts at the first trace number and continues through
the trace, closing delimiter, answer, and EOS. If this supervised suffix has
length `K`, then:

```text
L_think = -(1/K) sum_{t in completion suffix}
          log p_theta(x_t | x_0,...,x_{t-1}).
```

This is causal next-token training with teacher forcing, not free-running rollout
during training. Evaluation separately reports teacher-forced component metrics
and free-running autoregressive final-count accuracy.

## v15: Shakespeare haystack with inserted marker needles

- Haystack: contiguous length-256 windows from the standard Tiny Shakespeare
  corpus (`karpathy/char-rnn`, `data/tinyshakespeare/input.txt`).
- Needles: choose `n` distinct positions and replace the original characters at
  those positions with independently sampled marker tokens from ten marker types.
- Count: `n` is uniform on 1-30 during training and validation.
- Position encodings: RoPE and learned relative-position bias (RPE).
- Models: `2 position encodings x 2 modes = 4` independent Transformers.

## v16: native target-letter counting in Shakespeare

v16 does not insert or overwrite markers. It samples a contiguous length-256
Tiny Shakespeare window whose exact number of occurrences of the target character
is `n`. The deduplicated target set is:

```text
S H A K E R s h a k e r
```

Every prompt starts with an explicit task prefix:

```text
<BOS> <CountChar> target-character <Sep> 256-character-window ...
```

In thinking mode, the trace alternates shared number tokens and the matched native
target character. The kth trace number can therefore be evaluated against the kth
occurrence of the named character without a separate marker vocabulary.

- Count: balanced 1-30.
- Position encodings: RoPE and RPE.
- Models: `2 position encodings x 2 modes = 4` independent Transformers.

## v17: decreasing long-tail synthetic training distribution

v17 keeps v10's synthetic length-256 haystack, inserted marker task, learned
absolute position embedding (APE), and separate non-thinking/thinking models.
Only the training distribution over exact count changes. Both exposed samplers
assign less probability to examples with more needles:

```text
power:       p(n) = n^(-alpha) / sum_{j=1..30} j^(-alpha)
exponential: p(n) = exp(-beta*(n-1)) / sum_{j=1..30} exp(-beta*(j-1))
```

Defaults are `alpha=1.0` and `beta=0.15`. Validation remains balanced with equal
examples for each count 1-30, so performance across count bins is comparable even
though the training stream is imbalanced. The theoretical probabilities and the
realized batch statistics are both saved.

- Position encoding: APE only.
- Models: `1 position encoding x 2 modes = 2` independent Transformers.

## Training and outputs

Main runs use 10,000 optimizer steps, batch size 128, AdamW learning rate `3e-4`,
500 warmup steps, evaluation every 500 steps, autoregressive evaluation and
checkpointing every 1,000 steps. Colab notebooks mount Drive before setup and can
sync checkpoints during training.

Each run records:

- completion-only total loss plus trace-index, trace-marker, final-answer, and EOS
  component losses where applicable;
- teacher-forced final-count and trace-marker accuracy;
- free-running autoregressive final-count accuracy;
- balanced exact-count and 1-10 / 11-20 / 21-30 summaries;
- descriptive broad-attention, k-to-k retrieval, trace readout, count probes, and
  PCA centroid geometry.

Entry points:

```bash
python -m synthetic_counting_v15.run_v15 --preset main --stage all --device cuda
python -m synthetic_counting_v16.run_v16 --preset main --stage all --device cuda
python -m synthetic_counting_v17.run_v17 --preset main --stage all --device cuda
```

For v17, choose the alternative sampler with:

```bash
python -m synthetic_counting_v17.run_v17 --preset main --stage all \
  --count-sampling exponential --exponential-beta 0.15
```
