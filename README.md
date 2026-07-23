# Synthetic CoT NiaH Counting

这个仓库研究 Transformer 在 needle-in-a-haystack 风格计数任务中，Nonthinking（直接回答）与 Thinking（显式 occurrence trace）如何学习不同的 counting mechanism。项目从完全可控的符号序列逐步扩展到 Tiny Shakespeare 原生字符窗口，并通过 attention、hidden-state manifold、ablation、activation patching、state transplant 和 dense checkpoint dynamics 区分“模型答对了”与“模型如何答对”。

## 从这里开始

目前最重要的三套实验系统是：

| 系统 | 核心问题 | 主要设定 | 当前结果 | 入口 |
|---|---|---|---|---|
| v10 | 建立完整的计数机制与因果分析基线 | 合成 token、APE、count 1–30、独立 Nonthinking/Thinking | 两者最终均 100%；找到 broad、targeted、successor、residual/readout 等分布式路径 | [v10 README](src/synthetic_counting_v10/README.md) · [报告](colab_results/v10_main_seed1234_20260712_172332/syn_v10_report.html) |
| v16.2 RoPE | 在自然字符窗口中检验 v10 机制能否复现 | Tiny Shakespeare、query first、3 字符 union count 1–10、RoPE | Nonthinking AR 0.71；Thinking AR 0.99；出现 ordered retrieval、trace readout 与结构化 manifold | [v16.2 README](src/synthetic_counting_v16_2/README.md) · [报告](colab_results/v16_2_main_rope_seed1234/v16_2_full_causal_report.html) |
| v20 | 追踪 counting functions 如何随训练形成 | v16.2 基础上扩到 count 1–30、atomic count token、100-step snapshots | Nonthinking AR 0.3347；Thinking AR 0.912；successor 快速形成，targeted routing/value/causal dependence 更平滑 | [v20 README](src/synthetic_counting_v20/README.md) · [报告](colab_results/v20_main_RoPE_count1-30_seed1234/v20_counting_mechanism_report.html) |

如果只想理解当前最可信的科学结论，建议依次阅读 v10 → v16.2 → v20。它们分别对应“可控基线”“自然字符复现”“训练动力学与功能级 phase audit”。

## 当前机制结论

现有结果最支持的是一个分布式 retrieval—transport—state update—readout 计算图：

1. Thinking 的 trace-index query 通过 targeted routing 定位 prompt 中第 `k` 个目标 occurrence。
2. attention value 与 residual 路径传输被定位字符的 identity；多个 heads 协作通常比单个最高分 head 更重要。
3. marker/index 位置的 residual stream 形成结构化的 count/occurrence manifold，但尚不能简化为已证明的单一“count vector +1”。
4. successor/stop 相关功能帮助模型继续到下一个 index 或结束 trace。
5. 最终答案可从后层 residual count state 读出；`<Ans>` attention 指向最后一个 index 可能参与信息传输，但不是唯一解释。
6. Nonthinking 能形成 broad aggregation，却缺少 Thinking 的逐 occurrence 对齐与局部可监督状态路径，因此在更自然、更长的 count 任务上明显更差。

v20 进一步表明，不同功能可以以不同速度形成：marker-successor role 在早期快速出现，targeted retrieval mass、QK margin、value transport 与 targeted causal dependence则跨数千 steps 平滑增强。这里按功能分别判断“快速形成还是渐进形成”，不要求所有机制同步变化。

## Version map

下表记录每个 version 的主要改动、想回答的问题和本地结果状态。早期版本通常以 notebook 和 pipeline 文档为主；“有结果”表示当前仓库的 `colab_results/` 或正式 `output/` 中保留了对应报告/产物。

| Version | 做了什么 | 主要结果或用途 | 本地状态 |
|---|---|---|---|
| v0 | 建立 trace-enumeration counting 基础代码、loss-mask regime 与一键 Colab | 奠定数据生成、训练、probe 和 attention 分析框架 | 代码 + [notebook](notebooks/Trace_Count_v0_Colab.ipynb) |
| v1 | NIAH-like count，比较 think 与 no-think | 首次把显式 trace 用于 needle counting | [notebook](notebooks/Trace_Count_v1_Colab.ipynb) |
| v2 | 固定长度、count 1–10 的严格合成对照；两模型、indexed trace | 后续 v3–v4 机制分析的训练基座 | [notebook](notebooks/Trace_Count_v2_Colab.ipynb) · [pipeline](docs/pipelines/pipeline_v2_codex_prompt.md) |
| v2.2 | 在 v2 checkpoint 上深入分析 attention 分布 | 细化 needle、local successor 与 position pattern | [notebook](notebooks/Trace_Count_v2_2_Colab.ipynb) |
| v3 | 不重训，区分 final-needle retrieval 与局部 `+1` attention | 确立不同候选 head role 的诊断方式 | [notebook](notebooks/Trace_Count_v3_Colab.ipynb) · [pipeline](docs/pipelines/pipeline_v3_codex_prompt.md) |
| v3.2 | 对 v2 候选 heads 做 causal ablation/patching | 找到强 final-needle attention；单 head ablation 也显示冗余 | [notebook](notebooks/Trace_Count_v3_2_Colab.ipynb) · [pipeline](docs/pipelines/pipeline_v3_2_causal_tests_codex_prompt.md) |
| v4 | hidden-state probe、matched directions、steering 与 patching | 把 attention 分析扩展到 count representation 和因果 state intervention | [notebook](notebooks/Trace_Count_v4_Colab.ipynb) · [pipeline](docs/pipelines/pipeline_v4_steering_codex_prompt.md) |
| v5 | 单个共享模型，用 `<THINK_ON>/<THINK_OFF>` 切换输出方式 | 检验同一参数内是否共享或切换计算路径 | [notebook](notebooks/Trace_Count_v5_Colab.ipynb) · [pipeline](docs/pipelines/pipeline_v5_mixed_thinking_toggle_codex_prompt.md) |
| v5.2 | 不重训，对 switch token 和 mode routing 做诊断 | 检查切换信号如何影响后续状态 | [notebook](notebooks/Trace_Count_v5_2_Colab.ipynb) |
| v5.3 | v5 的 attention、ablation 与 causal patching | 定位 shared model 的 mode-specific mechanism | [notebook](notebooks/Trace_Count_v5_3_Mechanism_Colab.ipynb) · [pipeline](docs/pipelines/pipeline_v5_3_mechanism_causal.md) |
| v5.4 | v5 的 count-state causal direction 分析 | 检验 representation direction 是否被模型因果使用 | [notebook](notebooks/Trace_Count_v5_4_Count_State_Causal_Colab.ipynb) · [pipeline](docs/pipelines/pipeline_v5_4_count_state_causal.md) |
| v6 | trace 去掉显式数字 index，改用 separator | 检验 targeted retrieval 是否依赖显式 index token | [notebook](notebooks/Trace_Count_v6_Colab.ipynb) · [pipeline](docs/pipelines/pipeline_v6_separator_trace_codex_prompt.md) |
| v7 | 在 v2 基础上只增加 context length | 长序列控制实验 | [notebook](notebooks/Trace_Count_v7_Colab.ipynb) |
| v8 | 在 v2 基础上只扩展 needle count | count-range 控制实验 | [notebook](notebooks/Trace_Count_v8_Colab.ipynb) |
| v9 | query-conditioned pair counting，并做 reduced-capacity/shortcut audit | 引入 query-conditioned counting，检查 shortcut | [notebook](notebooks/Trace_Count_v9_Colab.ipynb) |
| **v10** | 合成 count 1–30、APE、两模型；完整行为/attention/state/causal suite | 两模型最终 100%；建立后续机制分析标准 | **核心系统** · [README](src/synthetic_counting_v10/README.md) · [报告](colab_results/v10_main_seed1234_20260712_172332/syn_v10_report.html) |
| v11 | 小模型 `d_model=64`，比较 APE/RoPE/RPE | 位置编码学习动力学对照 | [aggregate report](colab_results/cot_learning_dynamics_v11_v14/syn_cot_learning_stages_report.html) |
| v12 | 小模型 APE，长度 512、count 1–50 | 更长 context 和更大 count | [aggregate report](colab_results/cot_learning_dynamics_v11_v14/syn_cot_learning_stages_report.html) |
| v13 | 固定、有限、平衡的数据集 | 区分 memorization 与 generalization | [aggregate report](colab_results/cot_learning_dynamics_v11_v14/syn_cot_learning_stages_report.html) |
| v14 | 小模型 Tiny Shakespeare 字符 haystack，并插入 markers | 从符号噪声过渡到自然字符背景 | [aggregate report](colab_results/cot_learning_dynamics_v11_v14/syn_cot_learning_stages_report.html) |
| v15 | Tiny Shakespeare 连续窗口中插入 needle；RoPE/RPE；all-sequence objective | 自然文本背景下的插入式计数 | [报告](colab_results/v15_main_all_sequence_seed1234_20260718_171459/syn_v15_report.html) |
| v16 | 不插入 marker，直接计数原生目标字符；RoPE/RPE | 从人工插入转为 native character count | [报告](colab_results/v16_main_all_sequence_seed1234_20260718_171835/syn_v16_report.html) |
| v16.1 | split-local indexed window sampling、无跨 split window、确定性 resume | 修复数据泄漏、采样效率与恢复问题 | [notebook](notebooks/Trace_Count_v16_1_Colab.ipynb) · [pipeline](docs/pipelines/pipeline_v16_1_split_window_sampling.md) |
| **v16.2** | 三字符 query set；canonical 为 query-first、RoPE、count 1–10；objective switch；完整 v10 port | Thinking AR 0.99 vs Nonthinking 0.71；自然字符中复现 targeted routing 与 manifold | **核心系统** · [README](src/synthetic_counting_v16_2/README.md) · [报告](colab_results/v16_2_main_rope_seed1234/v16_2_full_causal_report.html) |
| v16.2 RPE run | 与 canonical v16.2 不同的 RPE/count 1–30 探索运行 | 保留作位置编码/范围对照，不与 RoPE 主报告混写 | [结果目录](colab_results/v16_2_main_rpe_seed1234) |
| v16.3 | 在 v16.2 基础上改为 data-first/query-last | query 顺序的直接控制实验 | [对比报告](colab_results/v16_3_main_data-query_seed1234_20260721/v16_2_vs_v16_3_query_order_report.html) |
| v17 | v10 合成任务改用 RoPE 与 decreasing long-tail count distribution | 检验位置编码和不平衡 count exposure | [报告](output/reports/v16_v17/v17/syn_v17_report.html) |
| v18 | prompt 1024、count 1–128、index/count token 完全分离；uniform/power × direct/CoT 四模型 | 长上下文、大 count 与两种数据分布 | [报告](colab_results/v18_main_all_seed1234_20260719_191912/syn_v18_report.html) |
| v19 | v18 改为 index/final count 共享 decimal digit tokenization | 检验 compositional number representation | [报告](colab_results/v19_main_all_seed1234_20260719_205527/syn_v19_report.html) |
| **v20** | query-first RoPE、count 1–30、atomic tokens、100-step dynamics、功能级 causal/phase audit | Thinking 0.912 vs Nonthinking 0.3347；successor 快速、targeted/value/causal 路径更平滑 | **核心系统** · [README](src/synthetic_counting_v20/README.md) · [报告](colab_results/v20_main_RoPE_count1-30_seed1234/v20_counting_mechanism_report.html) |
| v21 | v20 的 atomic count/index token 改为共享 digit-wise tokenization | 与 v20 构成数字 tokenization 控制 | [报告](colab_results/v21_main_RoPE_count1-30_digit_seed1234/v21_counting_mechanism_report.html) |

## 仓库结构

```text
.
├── src/                   # 各 version 的可安装 Python package
├── notebooks/             # Colab 入口；Drive mount、恢复、同步、分析
├── docs/pipelines/        # 各实验的设计与复现说明
├── tests/                 # 数据、训练恢复、分析与报告回归测试
├── data/                  # 本地语料/索引数据
├── colab_results/         # checkpoint、表格、图片与 HTML 报告（Git ignore）
├── output/                # 保留的正式报告产物
├── external_snapshots/    # 外部参考代码/快照
└── pyproject.toml         # 依赖和 package 列表
```

目录约定：

- `src/synthetic_counting_vX/`：该 version 的实现；重要 version 在包内放独立 README。
- `notebooks/Trace_Count_vX_Colab.ipynb`：Colab 调度器，不应成为唯一代码来源。
- `colab_results/<run_name>/config.json`：该次运行的真实设定，以它为准，而不是只凭目录名推断。
- `tables/`：报告所用的聚合统计；`figures/`：静态图；`analysis/`：扩展与因果分析。
- `checkpoints/` 或 snapshot shards：模型权重和恢复状态，体积大且不纳入 Git。

## 安装

需要 Python 3.10+。本地开发推荐：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

如果只运行实验而不开发：

```powershell
python -m pip install -e .
```

检查 CUDA：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 运行原则

每个主实验都支持 `python -m <package>.run_<version>` 入口，具体参数以包内 README、`config.py` 与对应 notebook 为准。不要在不同 version 间仅凭相似参数复制 checkpoint：vocabulary、query 顺序、位置编码、数字 tokenization 或 objective schedule 不同都会使 checkpoint 不兼容。

典型流程：

1. 固定 seed、data split/pool 和 evaluation manifest。
2. 先运行 train，确认 checkpoint/recovery 正常。
3. 再运行 attention/state/causal analysis，避免训练时持久化原始大 tensor。
4. 报告只引用可追溯到 CSV/JSON 的聚合值。
5. 对“phase transition”逐功能报告 center、10–90% width、模型证据与 per-k exposure，不把单条 accuracy 陡升当作充分证据。

## 测试

运行全部测试：

```powershell
pytest
```

核心系统的定向测试：

```powershell
pytest tests/test_synthetic_counting_v10.py
pytest tests/test_synthetic_counting_v16_2.py tests/test_synthetic_counting_v16_2_dynamics.py
pytest tests/test_synthetic_counting_v20_v21.py
```

代码检查：

```powershell
ruff check src tests
```

## 文件与存储管理

以下内容属于可再生成或机器相关文件，已在 `.gitignore` 中排除：

- `.venv/`、`__pycache__/`、`.pytest_cache/`、`.ruff_cache/`
- `tmp/`、`runs/`、`outputs/`
- `colab_results/` 与大 checkpoint
- 本地 `paper_draft/`

清理仓库时不要按“体积大”直接删除 `colab_results/`：其中包含唯一的 checkpoint、分析表和正式 HTML。安全清理对象应限于明确的 cache、临时渲染目录、可再生成的 debug run 和 Python bytecode。正式结果如需迁移，应先核对 `config.json`、manifest、final checkpoint、tables 和 HTML 报告是否齐全。

## 解释结果时的证据等级

由弱到强建议区分：

1. 线性 probe / \(R^2\)：说明信息可读，不说明模型使用它。
2. attention mass / top-1：说明候选 routing，不说明 value 内容或因果必要性。
3. hidden-state manifold：说明 representation geometry，但仍可能是旁观变量。
4. ablation：说明组件的必要性，但可能混有全局破坏。
5. pattern-only、value-only、residual patch：区分定位、内容传输和整合状态。
6. localized causal intervention、counterfactual transplant、steering：更直接检验具体计算假设。
7. 多 seed、per-k、exposure/curriculum 控制：判断机制与训练动力学是否稳定。

本项目因此不把高 \(R^2\)、漂亮 attention map 或单 seed 的一条陡峭 accuracy 曲线单独称为 counting algorithm 或 phase transition。
