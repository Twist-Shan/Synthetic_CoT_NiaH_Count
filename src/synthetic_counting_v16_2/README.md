# v16.2 RoPE：自然字符窗口中的三字符计数

v16.2 将 v10 的机制问题迁移到 Tiny Shakespeare 的原生字符窗口。仓库中的核心 v16.2 是 **RoPE、query-first、count 1–10** 这一套逻辑；它不是 query-last，也不是同目录中另存的 RPE/count 1–30 探索性运行。

## 1. 科学问题

v16.2 研究：

1. 在不人工插入 needle 的自然字符序列中，CoT 是否仍能形成 targeted retrieval？
2. Thinking 与 Nonthinking 是否形成不同的 attention routing 和 hidden-state geometry？
3. 这些机制在 10,000 个训练 step 中何时出现，是否先于或后于最终行为准确率？
4. v10 的 ablation、patching、state transplant 和 representation 分析能否迁移到 RoPE 与自然字符数据？

## 2. 输入、query 顺序与输出

每个样本首先给出三个互不相同的 query 字符，然后给出一段未经改写的 256 字符 Tiny Shakespeare 窗口：

```text
<CountChar> A B C <Sep> data[256]
```

目标 count 是三个 query 字符在 data 中的出现次数之和。这里明确是 **query first / data second**。

Nonthinking 直接输出 count。Thinking 则按这些目标字符在 data 中从左到右出现的顺序，交替输出 occurrence index 与实际字符 marker，再输出最终 count。因而 Thinking trace 提供可严格对齐的 `k-to-k retrieval`，Nonthinking 没有语义明确的第 `k` 次 retrieval query，不能直接套用同一个指标。

## 3. 实验设定

| 项目 | 设定 |
|---|---|
| 文本 | Tiny Shakespeare，连续 256 字符窗口，不插入、不替换字符 |
| query | 3 个不同字符；100 个 query sets，按字符频率分层 |
| count | 三字符 union count，1–10 |
| 顺序 | query first，data second |
| 模型 | 4 层、4 heads、`d_model=256`、MLP 1024、context 384、无 dropout |
| 位置编码 | RoPE，base 10,000 |
| 优化 | AdamW，batch 128，学习率 `3e-4`，weight decay 0.01 |
| 训练 | 10,000 steps，seed 1234 |
| objective | steps 1–1500 为 all-sequence；step 1501 起只监督 task output |
| checkpoint | 每 500 steps |
| 数据控制 | 固定 pool/split/evaluation manifest；selection 与 reporting split 分离 |

Nonthinking 与 Thinking 使用相同 seed、相同架构与匹配的数据顺序，但仍是两次独立训练。

## 4. 数据与复现逻辑

[`needle_pool.py`](needle_pool.py) 预先索引满足 count 条件的窗口，避免训练时反复扫描整份语料。[`data.py`](data.py) 负责 query-first 序列和两种 target representation。[`training.py`](training.py) 保存 optimizer/RNG/sampler 状态，支持从 checkpoint 确定性恢复。

主入口：

```powershell
python -m synthetic_counting_v16_2.run_v16_2 `
  --preset main `
  --stage all `
  --device cuda `
  --seed 1234 `
  --out-root runs/synthetic_counting_v16_2 `
  --run-name v16_2_main_rope_seed1234
```

Colab 入口为 [`Trace_Count_v16_2_Colab.ipynb`](../../notebooks/Trace_Count_v16_2_Colab.ipynb)。它负责挂载 Google Drive、在 Colab 本地运行代码、恢复/同步 checkpoint、流式显示训练进度、验证产物，并可在成功后断开 runtime。

完整说明见 [`pipeline_v16_2_character_sets.md`](../../docs/pipelines/pipeline_v16_2_character_sets.md)。

## 5. 分析系统

| 模块 | 作用 |
|---|---|
| [`checkpoint_dynamics.py`](checkpoint_dynamics.py) | 跨 checkpoint 的行为、attention 与 representation dynamics |
| [`analysis.py`](analysis.py) | 最终行为、trace、attention、hidden state 与 intervention |
| [`v10_port_analysis.py`](v10_port_analysis.py) | 将 v10 的完整机制分析迁移到 v16.2 |
| [`interactive_geometry.py`](interactive_geometry.py) | 可选择 mode/site/layer/step 的 3D geometry |
| [`report_readability.py`](report_readability.py) | 报告定义、图注与可读性检查 |
| [`timing.py`](timing.py) | 训练、评估、checkpoint 和同步耗时记录 |

分析中特别区分：

- `attention mass`：某个 query 位置分给一组目标 key 位置的 attention 概率之和；
- `top-1 retrieval accuracy`：最大 attention 的 key 是否正好是目标 occurrence；
- `enrichment`：实际目标 mass 除以在可见位置中均匀分配时的 chance mass；
- `effective dimension`：hidden-state 协方差特征值的 participation ratio，\((\sum_i \lambda_i)^2/\sum_i\lambda_i^2\)；
- `causal recovery`：corruption 后通过 patch 恢复的目标 logit/accuracy，占 clean 与 corrupted 差距的比例。

## 6. 已得到的结果

核心结果在 [`v16_2_main_rope_seed1234`](../../colab_results/v16_2_main_rope_seed1234)，完整中文因果报告是 [`v16_2_full_causal_report.html`](../../colab_results/v16_2_main_rope_seed1234/v16_2_full_causal_report.html)。

最终 step 10,000 的 autoregressive exact accuracy：

| 模型 | 样本数 | 最终 AR accuracy | 其他 trace 指标 |
|---|---:|---:|---|
| Nonthinking | 100 | 0.71 | 无显式 trace |
| Thinking | 100 | 0.99 | trace exact 0.89；marker recall 约 0.972 |

主要机制观察：

- Thinking 的行为学习明显快于 Nonthinking，说明显式 occurrence-level supervision 在自然字符数据中仍有优势。
- Thinking 出现稳定的 ordered targeted retrieval 与 final-answer trace readout；最终固定 head 指标中，ordered correct-top1-minus-chance 约 0.467，trace readout mass 约 0.873。
- Nonthinking 也形成 broad needle aggregation。一个最终关键 head 的 top-n needle recall 约 0.842，enrichment 约 21.1，但其整体 AR 仍只有 0.71，说明“看到了目标字符”不等于已经精确整合总数。
- hidden states 不是一条简单线性 count axis。示例 geometry summary 中，Nonthinking `<Ans>` 后层 effective dimension 约 3.0；Thinking `<Ans>` 中层约 5.5；Thinking index 与 marker manifold 的维度和结构不同。报告因此更强调 manifold geometry，而不是单独把线性 probe 的 \(R^2\) 当成机制证据。
- v10 causal port 提供 retrieval corruption/patching、successor/stop、MLP、residual transport、trace conflict 和 head↔state 分析，用来区分定位、identity transport、状态更新与最终 readout。

## 7. 当前可支持的机制图景

Thinking 不是简单地让 `<Ans>` 直接 attention 到最后一个 index。更符合现有证据的描述是：

1. trace index query 通过 targeted head 从 prompt 定位对应 occurrence；
2. value 与 residual 路径传输该 occurrence 的字符/身份信息；
3. marker/index 位置形成结构化但多维的 hidden-state manifold；
4. successor/stop 与 trace readout 路径决定继续检索、结束 trace以及将累计状态送到答案位置；
5. 最终 count 可从 residual state 读出，attention 到最后 index 只是候选信息路径之一。

## 8. 局限与相邻运行

- 当前核心结果只有一个 seed；head 编号与 transition 时间需要多 seed 稳定性验证。
- periodic/final AR 样本数较少，尤其不适合精确判断局部 accuracy 跳变。
- teacher-forced trace 分析可能高估自由生成时机制的稳定性。
- attention 指标需要与 pattern-only、value-only、residual patch 和位置局部 ablation 一起解释。
- [`v16_2_main_rpe_seed1234`](../../colab_results/v16_2_main_rpe_seed1234) 是另一套 RPE/count 1–30 运行，不应与本 README 的 canonical RoPE/count 1–10 结果混写。
- v16.3 将输入改为 data-first/query-last，是 query 顺序的直接对照；v20 则把 RoPE/query-first 扩展到 count 1–30，并将 checkpoint 密度提高到每 100 steps。

