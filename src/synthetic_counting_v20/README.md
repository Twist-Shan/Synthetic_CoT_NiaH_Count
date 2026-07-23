# v20：RoPE count 1–30 的 counting mechanism 与功能级 phase audit

v20 以 v16.2 的 query-first、RoPE、自然字符窗口为基础，把 count 扩展到 1–30，使用 atomic count/index tokens，并将 scientific checkpoint 加密到每 100 steps。它的核心目标不是只比较最终准确率，而是追踪 targeted retrieval、successor、value transport、residual geometry、causal dependence 和最终行为分别如何形成。

## 1. 科学问题

v20 主要研究：

1. CoT counting 的具体计算图是什么：如何定位第 `k` 个 needle、传输 identity、形成下一个状态并输出 count？
2. Thinking 相比 Nonthinking 的优势来自更好的 retrieval、可迭代 trace、representation geometry，还是最终 readout？
3. 每一种功能是突然出现还是平滑形成？若有行为准确率陡升，它是否只是多个局部误差沿长 trace 相乘后的放大？
4. 机制变化能否由 objective switch、每个 `k` 的 token exposure、count curriculum 或序列长度解释？

这里的“phase transition”按功能分别判断。targeted retrieval、successor、value transport、residual geometry、causal effect 和行为准确率可以有不同的转折中心和宽度，不要求它们同时出现。

## 2. 实验设定

| 项目 | 设定 |
|---|---|
| 数据 | Tiny Shakespeare 连续 256 字符窗口 |
| query | 3 个不同字符，query first |
| count | 三个 query 字符 union count，1–30 |
| 数字表示 | 每个 count/index 是一个独立 atomic token |
| 模型 | 4 层、4 heads、`d_model=256`、MLP 1024、context 384 |
| 位置编码 | RoPE |
| 训练 | 10,000 steps，batch 128，AdamW，lr `3e-4`，wd 0.01，bf16 |
| objective | steps 1–1500 all-sequence；1501 起 task-output-only |
| 对照 | 匹配的 Nonthinking 与 Thinking 独立模型 |
| periodic AR | 2 examples/count，用于低成本轨迹 |
| final AR | 50 examples/count，共 1,500 个样本/模型 |
| scientific snapshots | 每 100 steps 保存 model-only FP16 |
| recovery checkpoints | 每 500 steps，另保留 objective boundary/final |

v21 只把这里的 atomic count/index tokens 改成共享 digit-wise tokenization，其余逻辑尽量保持一致，是数字表示方式的直接对照。

## 3. I/O 与稳定性设计

密集 checkpoint 如果保存 optimizer、RNG 和原始 attention tensor，会迅速占满 Drive 并拖垮 Colab。v20 将“科学分析快照”和“训练恢复 checkpoint”分开：

- 每 100 steps 保存 model-only FP16 scientific snapshot；
- 每 500 steps 保存可恢复训练的 full checkpoint，并使用 rolling/pinned/final 保留策略；
- 五个 scientific snapshots 打包成一个 shard；
- 分析时按 shard 流式读取、聚合后释放；
- 每个 checkpoint 只写 sufficient-statistics CSV；
- 3D sample cloud、局部因果干预等大产物只在 milestone 保存；
- 禁止持久化 raw attention tensor 或巨型逐 token CSV；
- Drive 同步使用临时文件原子替换，并检查 size 与 mtime。

这些设计在保留 100-step dynamics 的同时，把主要 checkpoint 存储控制在可管理范围。

## 4. 运行入口

```powershell
python -m synthetic_counting_v20.run_v20 `
  --preset main `
  --stage all `
  --device cuda `
  --seed 1234 `
  --out-root runs/synthetic_counting_v20 `
  --run-name v20_main_RoPE_count1-30_seed1234
```

Colab 入口是 [`Trace_Count_v20_Colab.ipynb`](../../notebooks/Trace_Count_v20_Colab.ipynb)，实验设计见 [`v20_v21_experiment_design.md`](../../docs/v20_v21_experiment_design.md)。

主要模块：

| 文件 | 作用 |
|---|---|
| [`data.py`](data.py) / [`needle_pool.py`](needle_pool.py) | query-first 样本、平衡 count pool 与 exposure 统计 |
| [`training.py`](training.py) | 训练、密集 scientific snapshots 与恢复 checkpoint |
| [`analysis.py`](analysis.py) | 基础行为、attention、state 与 causal analysis |
| [`extended_analysis.py`](extended_analysis.py) | 交互式 attention dynamics、高样本 AR 与扩展机制指标 |
| [`phase_transition.py`](phase_transition.py) | 逐功能 transition 拟合、per-k/exposure 与 causal audit |
| [`v10_port_analysis.py`](v10_port_analysis.py) | v10 causal suite 的迁移 |
| [`interactive_geometry.py`](interactive_geometry.py) | hidden-state manifold 的可交互 3D 展示 |

## 5. 核心指标如何计算

### 5.1 Targeted retrieval

对 Thinking 的第 `k` 个 trace-index query，目标 key 是 prompt 中第 `k` 个目标字符位置。

- targeted mass：该 query 分给正确 key 的 attention 概率；
- correct-vs-best-wrong QK margin：正确 key 的 pre-softmax QK score 减去所有错误候选中的最大值；
- correct-occurrence top-1：最大 attention key 是否为正确 occurrence；
- per-k 曲线：分别对每个语义 `k` 聚合，而不是只看所有 `k` 的总体均值。

### 5.2 Marker successor

在 marker query 位置，考察它对“继续所需位置/下一 index token”或结束 trace 所需信息的 attention/readout 倾向。role score 是正确 successor 目标相对 control/wrong 目标的归一化优势。

### 5.3 定位与 identity transport

- attention-pattern-only patch：只替换 attention pattern，测试路由本身；
- value-only patch：保留目标 attention pattern，只替换 value/output 内容，测试 identity transport；
- residual-stream patch：替换指定层与位置的 residual state，测试整合后的状态是否足以恢复输出；
- normalized recovery：\((m_{\text{patched}}-m_{\text{corrupt}})/(m_{\text{clean}}-m_{\text{corrupt}})\)，其中 \(m\) 是目标 logit margin 或其他预注册结果指标。

### 5.4 每个 k 的 exposure

若 \(N_n(t)\) 是 step \(t\) 前 count 为 \(n\) 的训练样本数，则 Thinking 中语义 index \(k\) 的累计 exposure 为

\[
E_k(t)=\sum_{n\ge k}N_n(t).
\]

高 `k` 只在更长 trace 中出现，因此 exposure 更少。将横轴从 training step 换成 \(E_k(t)\)，可以区分“晚形成”究竟来自样本暴露不足，还是来自需要前置机制/curriculum。

### 5.5 突然还是平滑

每个功能单独比较平滑 sigmoid/连续增长模型和 changepoint 模型，并报告：

- transition center：曲线达到中点的 step；
- 10–90% width：从总变化量 10% 到 90% 所需的 step 宽度；
- model evidence：如 BIC 差值；
- per-k 与 exposure 对齐后的稳定性；
- causal effect 是否在相似窗口形成。

窄 width 支持快速形成，宽 width 支持渐进 specialization；单 seed 下应使用“快速/平滑形成”，而非宣称普适相变。

## 6. 最终行为结果

结果位于 [`v20_main_RoPE_count1-30_seed1234`](../../colab_results/v20_main_RoPE_count1-30_seed1234)，主报告是 [`v20_counting_mechanism_report.html`](../../colab_results/v20_main_RoPE_count1-30_seed1234/v20_counting_mechanism_report.html)。

最终 50 examples/count 的 autoregressive exact accuracy：

| 模型 | 正确率 | 解释 |
|---|---:|---|
| Nonthinking | 0.3347 | 能形成 broad retrieval，但直接从自然字符窗口精确整合 1–30 的总数仍很困难 |
| Thinking | 0.9120 | occurrence-level trace 显著改善长 count；剩余错误主要集中在更长 trace |

Nonthinking 差并不意味着它完全没有看到目标字符。它更可能学到了 noisy broad aggregation，但缺少 Thinking 提供的逐 occurrence 对齐、局部校验与可迭代状态路径；小的局部计数误差会直接落到最终答案。

## 7. 当前最可信的 counting mechanism

现有结果支持一个分布式计算图，而不是“单一 induction/successor head 完成递归公式”：

1. **定位**：Thinking 的 trace-index query 使用 targeted retrieval head 为正确 occurrence 分配更高 QK score/attention。
2. **传输**：value 路径把被定位字符的 identity 带回 trace；单一 head 的 value recovery 很低，而多个 head 合并后可高恢复，说明传输是分布式的。
3. **状态更新**：marker/index 位置的 residual stream 进入结构化的 count/occurrence manifold；MLP 和多个 attention outputs 共同更新状态。目前不能把它简化为已证明的单方向“向量 +1”。
4. **继续或停止**：marker-successor/stop 角色帮助决定下一个 index 或 `</Think>`。
5. **最终 readout**：答案依赖后层 residual count state。`<Ans>` attention 指向最后 index 可能参与传输，但不是“只读取最后 index”这一单一路径。

## 8. Training dynamics 与逐功能 phase audit

当前单 seed 的 100-step audit 给出：

| 功能 | 形成方式 | 估计中心 | 10–90% width | 当前解释 |
|---|---|---:|---:|---|
| marker-successor role | 快速 | 约 186 | 约 220 | 早期迅速形成局部顺序/继续规则 |
| targeted retrieval mass | 平滑 | 约 4,864 | 约 3,854 | routing specialization 长时间增强 |
| correct-vs-wrong QK margin | 平滑 | 约 4,635 | 约 3,656 | 正确 occurrence 的 score margin 渐进拉开 |
| top-1 retrieval | 平滑 | 约 4,393 | 约 5,435 | 聚合后看不到窄窗口突变 |
| value transport top-1 | 平滑、弱且分布式 | — | 约 4,505 | 单 head 不是 identity transport 的充分载体 |
| value transport top-2 | 平滑、协作 | — | 约 9,279 | 最终 normalized recovery 约 0.952 |
| targeted causal dependence | 平滑 | 约 5,423 | 约 4,106 | 因果作用随 routing 一起逐步增强 |
| Thinking final AR | 中后期加速 | 约 5k–6k 区间明显 | 拟合不确定 | 更像长 trace 对局部改进的乘法放大 |

marker-successor 可以称为“快速涌现的候选功能”，因为它在约 100–300 steps 内由接近零升到很高；targeted retrieval 则更适合称为渐进 specialization。二者不需要同步，也不应被强行归为同一个全局 phase transition。

Thinking AR 在 5,500→6,000 steps 由约 0.359 升至 0.532，但 targeted mass、QK margin、value transport 和 causal damage 均没有同样窄的共同跳变。一个合理解释是：若长 trace 的每一步局部成功率为 \(p\)，全局 exact success 近似含有 \(p^n\) 项；当 \(p\) 平滑提高时，长序列的 exact accuracy 可以显得很陡。

## 9. 仍欠缺的证据

- 多 seed：确认快速 successor 与平滑 targeted retrieval 是否跨初始化复现。
- 更密集且高样本的 AR：目前 100-step 模型快照很密，但高功效 AR 仍只在部分 step 运行。
- objective-switch、count distribution、sequence length 的系统干预：区分 exposure、curriculum 和优化动力学。
- 更完整的 pattern-only/value-only/residual patch 跨 step 曲线：确认定位、identity transport 与 state update 的因果时间顺序。
- generated-prefix 状态：检验 teacher-forced manifold 是否真正被自由生成过程使用。
- per-k changepoint：聚合曲线可能把不同 `k` 的快速但错位形成平均成平滑曲线。

