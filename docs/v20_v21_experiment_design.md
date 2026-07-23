# v20 / v21 实验设计：targeted retrieval、marker successor 与 phase transition

## 1. 受控问题

v20 和 v21 都使用 Tiny Shakespeare 的 256 字符窗口、三个字符组成的 query、query-first、RoPE、count 1–30、seed 1234，以及配对初始化的 nonthinking / thinking 模型。两者唯一有意改变的因素是数字表示：

- v20（atomic）：整数 1–30 各自是一个独立 token；
- v21（digit-wise）：所有整数由共享的 `<D0>`–`<D9>` 拼写，例如 12 是 `<D1><D2>`。

因此，v20 是与 v10 因果实验接口兼容的主实验；v21 是检验“独立 count token 是否促成机制涌现”的 tokenization control。

序列均为：

```text
nonthinking: <BOS> query[5] data[256] <Ans> number <EOS>
thinking:    <BOS> query[5] data[256] <Think> (number_k marker_k)*n </Think> <Ans> number_n <EOS>
```

## 2. 三层证据与操作定义

不把线性 probe 的 R² 当成主要机制证据。报告应按以下顺序组织：行为转折 → attention role → manifold geometry → 因果干预。

### 2.1 Targeted retrieval head

在 trace 中数字 (k) 的最后一个 token 位置作为 query；正确 key 是 prompt 中第 (k) 个 needle。对 head (h) 定义：

\[
T_h = \mathbb{E}_{x,k}\big[A_h(q=\mathrm{index}_k,\; key=\mathrm{needle}_k)\big].
\]

它是“定位并提取第 (k) 个 needle”的首要候选机制。最终 checkpoint 只在独立的 head-selection split 上选择 (T_h) 最大的 head；训练动态和干预在不重叠的 reporting split 上进行。

### 2.2 Marker successor head

在 marker (k) 的位置，模型下一步需要输出 (k+1)（若 (k=n)，则输出 `</Think>`）。候选 successor head 的描述性分数是 marker (k) 对紧邻它之前的 index-token group (k) 的绝对 attention mass：

\[
S_h = \mathbb{E}_{x,k}\big[\sum_{j\in\mathrm{tokens}(k)} A_h(q=\mathrm{marker}_k,key=j)\big].
\]

对 v21，marker 位置只预测 (k+1) 的第一个 digit，因此另行计算完整 digit sequence 的 teacher-forced exact 指标；不能把“首 digit 正确”误写成“语义整数 (k+1) 正确”。

### 2.3 Manifold geometry

对每层、每个位置类型（trace index、trace marker、final answer）先计算每个 (k) 的 hidden-state centroid (c_k)，再报告：

- adjacent distance：(lVert c_{k+1}-c_k\rVert_2) 的均值；
- adjacent step cosine：相邻位移向量的 cosine；
- straightness：(lVert c_K-c_1\rVert_2 / \sum_k\lVert c_{k+1}-c_k\rVert_2)；
- effective dimension：centroid PCA 方差比例 (lambda_i) 的 participation ratio (1/\sum_i\lambda_i^2)；
- adjacent-between / within：相邻 centroid 的均方距离除以同一 (k) 内的均方散度。

这些量描述的是轨迹是否形成、是否平滑、是否接近一维以及类间/类内分离；R² 不是主结论。

### 2.4 因果性质

在预先指定的 milestone checkpoint，仅把某个 head 在相应 query 位置的 pre-output head slice 置零：

- targeted head：只干预 trace-index query，观察下一 marker 的正确-token logit margin；
- successor head：只干预 marker query，观察下一 index token / `</Think>` 的 margin；
- 同层另一个 head 作为 control。

这比全局 head ablation 更局部，减少“整条序列都被破坏”的混淆。v20 另外运行完整的 v10 causal port，包括 retrieval corruption/patching、successor/stop、MLP、residual transport、trace conflict 和 head↔state 分析。

## 3. 功能级 phase transition 的判据

模型快照每 100 step 保存一次。这里不要求 targeted retrieval、successor、value transport、residual geometry、causal effect 与最终 accuracy 同时变化；每一种功能都单独判断是快速出现还是平滑形成。对每个功能分别查看：

1. 每个 true count 和每个 (k) 的 dense teacher-forced / autoregressive accuracy；
2. 固定 targeted 或 successor head 的 mass、QK margin、top-1 accuracy 和 role score；
3. marker/index centroid manifold 的 separation、step cosine 和 effective dimension；
4. pattern-only、value-only、residual-stream patch recovery，以及位置局部 head ablation 的 margin drop；
5. 变化是否可以被每个 (k) 的训练 exposure、objective switch 或 count curriculum 解释。

每条曲线分别比较连续增长（如 sigmoid）与 changepoint 模型，并报告 transition center、10–90% width 和模型证据。窄窗口支持该功能快速形成，宽窗口支持渐进 specialization；行为 accuracy 的陡升也可以与内部功能的平滑改善并存，例如长 trace 将逐步误差以乘法形式放大。单 seed 结果只称为“快速形成候选”或“平滑形成”，是否为稳定 phase transition 需要多 seed 验证。

## 4. 每个 k 的训练 token exposure

令 (N_n(t)) 为 step (t) 前训练中 accepted total count 为 (n) 的样本数。thinking trace 中语义 index (k) 的出现次数为：

\[
E_k^{\mathrm{semantic}}(t)=\sum_{n\geq k}N_n(t).
\]

v20 中 index-token exposure 就是 (E_k^{\mathrm{semantic}})。v21 中数字 (k) 有 (d(k)) 个 digit，所以：

\[
E_k^{\mathrm{index-token}}(t)=d(k)\sum_{n\geq k}N_n(t).
\]

此外分别记录 marker (k) exposure、从 marker (k) 输出 continue 的 exposure (sum_{n>k}N_n(t))、输出 close 的 exposure (N_k(t))，以及 final answer (k) 的 example/digit exposure。nonthinking 没有 trace，相关字段为 0，而 final-answer exposure 仍然计算。

## 5. I/O、存储和运行时间策略

- 每 100 step：model-only FP16 scientific snapshot；
- 每 500 step：包含 optimizer 与 RNG 的 full recovery checkpoint；
- 五个 scientific snapshot 打包到一个 shard；step 0 单独一 shard；
- recovery 只保留 rolling `latest.pt`，另固定保留 objective boundary 与 final；
- checkpoint analysis 一次流式读取一个 shard，立即聚合并释放；
- 周期 AR 每个 count 只用 2 个固定样本；query-order permutation 每个 count 用 1 个样本并展开六种顺序，避免该诊断反过来主导训练总时长；
- 每个 checkpoint 只写 sufficient-statistics CSV；3D sample cloud 和位置局部因果只在 milestone 保存；
- 禁止写 raw attention tensor 或巨型逐-token checkpoint CSV；
- milestone 的样本只保留 3 个 centroid-PCA 坐标，并生成可选择 mode/site/layer/step 的 `interactive_manifold_3d.html`；
- Drive 同步使用临时文件原子替换，并同时检查 size 与 mtime，避免同大小的新版 `latest.pt` 被错误跳过。

以约 3.18M 参数估算，FP16 scientific snapshots 约 6.4 MB/step。101 个快照 × 两个模型约 1.3 GB；加 rolling/pinned/final full checkpoints 后，整套实验预计约 1.5 GB 加表格和图片。若每 100 step 都保存 optimizer 的 full checkpoint，则会接近 7–8 GB，且产生大量 Drive I/O。

## 6. 主要入口

- `python -m synthetic_counting_v20.run_v20 --preset main --stage all ...`
- `python -m synthetic_counting_v21.run_v21 --preset main --stage all ...`
- `notebooks/Trace_Count_v20_Colab.ipynb`
- `notebooks/Trace_Count_v21_Colab.ipynb`

两份 notebook 都会挂载 Drive、在 Colab 本地 clone/update repo、流式显示训练输出、增量保存到 `colab_results`、验证关键产物存在，并在验证成功后可自动断开 runtime。
