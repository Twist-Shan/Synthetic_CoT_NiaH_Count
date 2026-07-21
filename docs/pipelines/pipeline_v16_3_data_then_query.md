# v16.3：data-first / query-last 计数实验

`synthetic_counting_v16_3` 是 v16.2 RoPE 实验的独立受控复现版本。除任务序列中
data 与 query 的先后顺序外，模型、数据分布、训练预算、loss schedule、checkpoint
协议、评估套件和机制分析均从 v16.2 复制。

## 唯一计划内变量

v16.2 是 **query-first**：

```text
nonthinking: <BOS> query data <Ans> count <EOS>
thinking:    <BOS> query data <Think> trace </Think> <Ans> count <EOS>
```

v16.3 固定为 **data-first / query-last**：

```text
nonthinking:
<BOS> data[256] <CountChar> q1 q2 q3 <Sep> <Ans> count <EOS>

thinking:
<BOS> data[256] <CountChar> q1 q2 q3 <Sep>
<Think> <1> marker1 ... <n> markern </Think> <Ans> count <EOS>
```

这里的 query 是 `<CountChar>`、三个待计数字符和 `<Sep>` 组成的五个 token；
`query[3]` 只表示其中有三个目标字符，不表示 query 只有三个 token。两种输出
representation 都先看到完整数据，再看到完全相同的任务 query。

配置中写入不可变字段：

```text
version = "v16_3"
sequence_layout = "data_then_query"
```

默认 run name 也包含 `data-query`。配置、vocabulary fingerprint、run directory 和
checkpoint 恢复均会核对该布局，因此 v16.2 的 query-first checkpoint 不能被静默加载
到 v16.3。

## 数据与固定评估套件

- 噪声源：Tiny Shakespeare 字符流。
- 数据窗口：连续 256 个字符，不改写原始字符。
- 切分：80% train、10% validation、10% test；相邻区域间留 `seq_len - 1` 个字符的
  guard，避免窗口跨区。
- Query：从训练区构造的 100 个互异三字符集合中采样；三字符总训练频率不超过
  0.04，分成 20 个频率 bin。
- 接受条件：三个 query 字符在窗口中的 union count 为 1–10。
- Thinking trace：按目标字符在 data 中从左到右的出现顺序输出
  `occurrence index, actual marker`。
- 固定套件：train 和 validation 各自保存 raw、task、ratio-matched mixture；test 只在
  最终 checkpoint 评估。

所有 split、needle pool 和 suite manifest 均写入 `RUN_DIR/data/` 并带 fingerprint。

## 模型与训练

主实验默认与 v16.2 相同：

- RoPE nonthinking 与 RoPE thinking 两个模型；
- 4 layers、4 heads、`d_model=256`、MLP=1024、context=384；
- AdamW，learning rate `3e-4`，warmup 500，weight decay 0.01；
- batch size 128，10,000 optimizer steps；
- 相同 seed、相同初始化和相同训练样本抽样顺序；
- step 1–1,500：all-sequence next-token loss；
- step 1,501–10,000：task-output-only loss；
- nonthinking 的 task-output span 从 `<Ans>` 开始；
- thinking 的 task-output span 从 `<Think>` 开始；
- 每 500 步保存 checkpoint，并额外保存 step 0、loss 边界、最终 numeric checkpoint
  和 `final` alias。

`final_count_loss_weight` 与 `cot_trace_loss_weight` 默认均为 1.0。训练 loss 使用当前
active target 上的加权平均；固定评估套件仍报告未加权 full-sequence loss，保证跨
checkpoint 可比。

## 完整实验与机制分析

Pipeline stages：

```text
prepare -> train -> attention -> state -> plots
```

训练后 notebook 默认运行 checkpoint dynamics，覆盖：

- teacher-forced 与 autoregressive accuracy；
- final-answer attention、prompt needle enrichment 和 top-n coverage；
- thinking 的 strict k-to-k trace retrieval；
- final count 与 trace progress 的 hidden-state decoding；
- representation geometry、cross-site transfer 与 checkpoint CKA；
- generated-prefix states；
- shortened-trace counterfactual readout；
- runtime breakdown。

完整 v10-style 因果实验入口也复制到 v16.3：

```bash
python scripts/run_v16_3_v10_port_analysis.py RUN_DIR --device cuda
python scripts/plot_v16_3_v10_port_analysis.py RUN_DIR
python scripts/build_v16_3_interactive_geometry.py RUN_DIR
```

这些分析通过 `spans.prompt_*`、`spans.query_*`、`spans.trace_*` 和 `spans.ans_pos`
定位 token，不依赖 v16.2 的绝对位置。

## Colab notebook 生命周期

Notebook：`notebooks/Trace_Count_v16_3_Colab.ipynb`。

它按以下顺序运行：

1. 挂载 Google Drive；Drive 仅保存 live checkpoints 与最终结果，不要求其中存在源码 repo；
2. 与原始 v16/v16.1 一致，从 GitHub clone 或 fast-forward pull 到
   `/content/Synthetic_CoT_NiaH_Count`，在 Colab 本地磁盘训练以避免 Drive I/O；
3. 以 editable/no-deps 方式安装，并验证 kernel 与 subprocess 都导入本地 v16.3；
4. 准备数据与固定 suites；
5. 训练两种 RoPE representation；
6. 每个 checkpoint 同步到
   `colab_results/v16_3_live_checkpoints/<run-name>/`；
7. 重跑时先从该目录自动恢复；已完成模型在 `SKIP_COMPLETED=True` 时跳过；
8. 运行 checkpoint dynamics；
9. 将完整 run bundle 复制到带时间戳的 `colab_results` 目录；
10. 复制成功后调用 `drive.flush_and_unmount()`；若 `AUTO_DISCONNECT=True`，等待
    `DISCONNECT_DELAY_SECONDS` 后调用 `runtime.unassign()` 自动断开。

自动断开只发生在最终完整结果复制成功之后。若训练或保存抛出异常，final cell 不会
执行断开，便于检查错误；已同步的 live checkpoints 仍可用于下一次恢复。

## 本地 smoke test

```bash
python -m synthetic_counting_v16_3.run_v16_3 \
  --preset debug --stage all --device cpu
```

主实验命令由 notebook 自动构造，也可本地运行：

```bash
python -m synthetic_counting_v16_3.run_v16_3 \
  --preset main --stage all --device cuda \
  --model-variant rope/nonthinking \
  --model-variant rope/thinking \
  --task-occurrence-ratio 1.0 \
  --count-max-threshold 10 \
  --train-steps 10000 \
  --max-steps-for-language-pred 1500 \
  --checkpoint-every 500 \
  --eval-examples-per-count 50 \
  --checkpoint-sync-root /path/to/v16_3_live_checkpoints \
  --skip-completed
```

Checkpoint dynamics：

```bash
python scripts/analyze_v16_3_checkpoint_dynamics.py RUN_DIR --device cuda
```
