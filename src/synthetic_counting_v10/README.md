# v10：符号化计数与完整因果分析基线

v10 是本仓库第一套“行为结果—attention routing—hidden-state representation—因果干预”完整闭环的实验系统。它使用完全可控的合成序列，把 Nonthinking 与 Thinking 训练成两个独立模型，并把 count 范围扩展到 1–30。后续 v16.2、v20 的机制分析，大量复用了这里建立的分析接口。

## 1. 科学问题

v10 主要回答：

1. 显式 CoT trace 是否让模型学会逐个定位 needle，而不是从整个 prompt 直接估计总数？
2. attention head 分别承担 broad aggregation、targeted retrieval、successor/stop 等什么角色？
3. count、occurrence index 和 marker identity 在 residual stream 中形成了怎样的几何结构？
4. attention pattern、value transport、MLP 更新和 residual state 中哪些成分对答案具有因果作用？

## 2. 任务与两种输出 representation

输入是长度为 256 的合成 prompt，包含噪声 token 和出现 1–30 次的目标 marker。

Nonthinking 直接输出最终答案：

```text
<BOS> prompt <Ans> count <EOS>
```

Thinking 输出一条与真实 occurrence 对齐的 trace：

```text
<BOS> prompt <Think>
<1> marker_1 <2> marker_2 ... <n> marker_n
</Think> <Ans> <n> <EOS>
```

这里的 `<k>` 是第 `k` 次 occurrence 的显式 index token，`marker_k` 是 prompt 中对应位置的实际 marker token。因此可以定义严格的 `k-to-k retrieval`：第 `k` 个 trace index 位置是否定位到 prompt 中第 `k` 个 needle。

## 3. 实验设定

| 项目 | 设定 |
|---|---|
| 数据 | 固定长度 256；count 在 1–30 间均匀采样 |
| 词表 | 64 个 noise token，10 个 marker token，另含控制/index/count token |
| 模型 | 4 层、4 heads、`d_model=256`、MLP 维度 1024、无 dropout |
| 位置编码 | learned absolute positional embedding（APE） |
| 对照 | Nonthinking 与 Thinking 分别初始化、分别训练 |
| 优化 | AdamW，batch 128，学习率 `3e-4`，warmup 500 steps |
| 训练 | 10,000 steps，completion-only loss，seed 1234 |
| 主要评估 | teacher-forced accuracy、autoregressive exact accuracy、trace exact accuracy |

两个模型不是同一网络的两个 decoding mode；比较时应理解为“输出监督结构不同的两次独立训练”。

## 4. 运行与分析流程

主入口是 [`run_v10.py`](run_v10.py)。`--stage all` 依次完成训练、行为评估、attention/hidden-state 分析、绘图与表格导出。

```powershell
python -m synthetic_counting_v10.run_v10 `
  --preset main `
  --stage all `
  --device cuda `
  --out-root runs/synthetic_counting_v10 `
  --run-name v10_main_seed1234 `
  --skip-completed
```

主要模块：

| 文件 | 作用 |
|---|---|
| [`core.py`](core.py) | 数据生成、tokenization、模型与基础评估 |
| [`training.py`](training.py) | 两个模型的训练、checkpoint 与 dynamics |
| [`attention_causal.py`](attention_causal.py) | attention role、head ablation 与 retrieval patching |
| [`state_causal.py`](state_causal.py) | residual-state intervention |
| [`hidden_state_patching.py`](hidden_state_patching.py) | hidden-state patching |
| [`successor_patching.py`](successor_patching.py) | successor/stop 路径因果分析 |
| [`successor_mlp_features.py`](successor_mlp_features.py) | successor 相关 MLP feature |
| [`final_bridge_causal.py`](final_bridge_causal.py) | trace 到最终答案的因果桥接 |
| [`geometry_path_steering.py`](geometry_path_steering.py) | representation geometry 与 steering |
| [`head_state_bidirectional.py`](head_state_bidirectional.py) | head 与 state 的双向干预 |

Colab 入口是 [`Trace_Count_v10_Colab.ipynb`](../../notebooks/Trace_Count_v10_Colab.ipynb)，完整 pipeline 说明见 [`pipeline_v10_two_model_count30_causal.md`](../../docs/pipelines/pipeline_v10_two_model_count30_causal.md)。

## 5. 已得到的结果

保存结果位于 [`v10_main_seed1234_20260712_172332`](../../colab_results/v10_main_seed1234_20260712_172332)，主报告是 [`syn_v10_report.html`](../../colab_results/v10_main_seed1234_20260712_172332/syn_v10_report.html)。

最终 step 10,000 时，两个模型在 count 1–10、11–20、21–30 三个区间的 teacher-forced 与 autoregressive exact accuracy 均达到 100%；Thinking 的整条 trace exact accuracy 也达到 100%。因此 v10 的重点不在“哪个模型最终更准”，而在两个模型通过什么内部路径得到同一个正确答案。

机制结果可概括为：

- Nonthinking 形成了对整个 prompt 广泛聚合的直接计数路径。对最关键 broad head 做单头 ablation 时，准确率由 1.000 降到约 0.096，说明这条直接路径具有强因果作用。
- Thinking 中可观测到与 occurrence 对齐的 targeted retrieval、successor/stop 和 trace-to-answer readout 角色，但作用分布在多个 head/层，而不是一颗“万能计数 head”。
- retrieval patching 中，单个最高排名 head 的 normalized recovery 只有约 0.028；加入第二个关键 head 后恢复到约 0.800，前四个 head 合计约 0.968。这说明定位与 identity transport 存在冗余和协作。
- residual-state transplant 能在后层使最终预测跟随 donor count，支持“答案依赖 residual stream 中形成的 count state”；但 learned absolute position 会与 trace 位置纠缠，因此跨 count 的 state transplant 必须谨慎解释。
- attention map 只能说明信息流候选路径。v10 通过 ablation、activation patching、state transplant 和 steering 将相关性证据与因果证据分开。

## 6. 如何理解 v10 的机制结论

当前证据支持以下工作模型：

1. Thinking 使用显式 `<k>`/marker trace，把一次全局计数拆成可监督的 occurrence-level retrieval。
2. 部分 head 负责选择正确 prompt needle，value/residual 路径负责传输 marker identity 或计数相关状态。
3. successor/stop 组件帮助决定继续到 `<k+1>` 还是结束 trace。
4. 最终 `<Ans>` 并非只依靠 attention 指向最后一个 index；答案也能从后层 residual stream 的 count representation 中被读出。

这仍不是“一个 head 实现递归公式”的证明。更准确的表述是：v10 找到了一个分布式、可因果干预的 retrieval—state update—readout 计算图。

## 7. 局限

- 这是完全符号化的数据，token identity、位置和 occurrence index 比自然文本更干净。
- APE 带来绝对位置混淆，尤其影响跨位置 hidden-state patching。
- 单个 seed 不能确定 head specialization 是否跨初始化稳定。
- teacher-forced trace 证据不等价于自由生成时逐步执行同一算法。
- 完美最终准确率会产生 ceiling effect，不适合研究行为能力如何缓慢形成；这也是 v16.2 和 v20 引入自然字符窗口、密集 checkpoint 与更难 count 范围的原因。

