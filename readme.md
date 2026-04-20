# DML_fresh_start — 选矿厂因果推断全流程项目

> 东鞍山烧结厂选矿车间：从 DCS 原始数据到因果发现、双重机器学习 (DML) 因果效应估计、反驳性实验的完整管线。

本项目包含**两条独立主线**，共享合成 DAG 生成器但数据流互不耦合：

| 主线 | 目标 | 数据来源 |
|------|------|----------|
| **真实数据线** | 在东鞍山选矿厂真实 DCS+化验数据上发现因果结构并估计因果效应 | DCS xlsx + 化验 Excel |
| **合成数据线** | 在已知真实因果效应的合成数据上验证算法正确性和鲁棒性 | `SyntheticDAGGenerator` 程序生成 |

---

## 目录结构总览

```
DML_fresh_start/
├── data_processing/              # [真实] 步骤 1: 原始数据预处理
├── 数据预处理/                   # [真实] 步骤 2: 变量筛选与注释
├── 多种方法因果发现/             # [真实+合成] 步骤 3: 因果发现（DAG 学习）
├── 因果的发现算法理论验证/       # [合成] 合成 DAG 生成器 + 蒙特卡洛基准测试
├── DAG图分析/                    # [真实] 步骤 4: DAG 因果角色解析
├── 双重机器学习/                 # [真实] 步骤 5: DML 因果效应估计（已弃用，见反驳性实验）
├── 反驳性实验/                   # [真实+合成] 步骤 6: DML 反驳实验 + 理论验证
├── data/                         # 数据目录（不在版本控制中）
├── 东鞍山烧结厂选矿专家知识.txt  # 领域知识文档
└── readme.md                     # 本文件
```

---

# 一、真实数据管线

完整流程：`原始数据 → 预处理 → 变量筛选 → 因果发现(DAG) → DAG角色解析 → DML因果效应估计 → 反驳性验证`

## 步骤 1：数据预处理 (`data_processing/`)

> 详细说明见 [`data_processing/README.md`](data_processing/README.md)

步骤 1.1 ~ 1.3 可并行，步骤 1.4 需在前三步完成后运行。

### 1.1 `preprocess_X.py` — X 特征主处理

| 项 | 说明 |
|----|------|
| **输入** | `data/大量长时间数据/` 下的 DCS 批次 xlsx 文件 + `data/操作变量和混杂变量/*.csv` 白名单 |
| **输出** | `data/X_features_final.parquet` — 1min 频率 X 特征宽表 (index=time) |
| **功能** | 从 DCS 批次 xlsx 读取传感器数据，按白名单筛选变量，物理硬阈值裁剪，exception-based ffill（有变化才记录），多批次合并 |
| **运行** | `python data_processing/preprocess_X.py` |

### 1.2 `preprocess_Y.py` — Y 品位目标变量处理

| 项 | 说明 |
|----|------|
| **输入** | `data/化验数据/*.xlsx` 化验记录 Excel |
| **输出** | `data/y_target_final.parquet` — 稀疏时间戳 Y 目标变量 (columns: y_fx_xin1, y_fx_xin2) |
| **功能** | 自动识别"新1#/新2#"精矿品位列，时间戳对齐到分钟精度，物理范围裁剪 (0~100%) |
| **运行** | `python data_processing/preprocess_Y.py` |

### 1.3 `preprocess_indicators.py` — 化验指标处理

| 项 | 说明 |
|----|------|
| **输入** | `data/化验数据/*.xlsx` 化验记录 Excel |
| **输出** | `data/indicators_final.parquet` — 1min 频率化验指标宽表 (ffill 后) |
| **功能** | 从化验 Excel 中提取非品位类化验指标（粒度、pH、矿浆浓度等），8:30 单点指标 ffill 至次日，多点指标 ffill 至下一观测 |
| **运行** | `python data_processing/preprocess_indicators.py` |

### 1.4 `merge_final.py` — X/Y/指标对齐合并

| 项 | 说明 |
|----|------|
| **输入** | `data/X_features_final.parquet` + `data/y_target_final.parquet` + `data/indicators_final.parquet` |
| **输出** | `data/timeseries_dataset_{line}_final.parquet` — 时序宽表（128K 行，1min 频率，Y 稀疏，供因果发现）<br>`data/modeling_dataset_{line}_final.parquet` — 监督宽表（仅 Y 非 NaN 行，供 DML） |
| **功能** | 以 X 全时间轴为锚点，±1min 容差对齐 Y 和指标，按产线拆分输出两类宽表 |
| **运行** | `python data_processing/merge_final.py` 或 `python data_processing/merge_final.py --line xin2` |

---

## 步骤 2：变量筛选与注释 (`数据预处理/`)

> 本步骤生成去共线性后的变量清单及元数据注释，供后续因果发现和 DML 使用。

### 2.1 `run_collinearity_detection.py` — 共线性检测与去冗余

| 项 | 说明 |
|----|------|
| **输入** | `data/X_features_final.parquet` + `data/操作变量和混杂变量/*.csv` |
| **输出** | `数据预处理/结果/non_collinear_representative_vars.csv` — 去共线性后的代表变量清单<br>`数据预处理/结果/Collinearity_Analysis_Report.md` — 共线性分析报告 |
| **功能** | Spearman 秩相关 + 层次凝聚聚类 (HCA)，自动识别高能共线组并挑选代表变量 |
| **运行** | `python 数据预处理/run_collinearity_detection.py` |

### 2.2 `annotate_variables.py` — 变量自动注释

| 项 | 说明 |
|----|------|
| **输入** | 去共线性后的变量 CSV + 工艺阶段分类 CSV + 变量描述 Excel |
| **输出** | `*_annotated.csv` / `*_annotated.md` — 带阶段、描述、单位注释的变量名册 |
| **功能** | 自动匹配每个变量的工艺阶段 (Stage 0~8)、中文描述、物理单位 |
| **运行** | `python 数据预处理/annotate_variables.py <输入CSV>` |

### 2.3 `classify_operability.py` — 可操作性分类 (LLM 辅助)

| 项 | 说明 |
|----|------|
| **输入** | 带注释的变量 CSV + `东鞍山烧结厂选矿专家知识.txt` |
| **输出** | `*_operability.csv` — 带 operable/observable 标签的变量清单 |
| **功能** | 调用 DeepSeek API，结合硬性可控设备清单和专家知识，批量分类每个变量的可操作性 |
| **运行** | `python 数据预处理/classify_operability.py` |

### 2.4 `sync_group_to_annotated.py` — Group 同步合并

| 项 | 说明 |
|----|------|
| **输入** | ABC 分类合集 CSV + 带注释变量 CSV |
| **输出** | 更新后的 annotated CSV（新增 Group A/B/C 列） |
| **功能** | 将产线归属标签 (A=新1专线, B=新2专线, C=公用) 同步到变量名册 |
| **运行** | `python 数据预处理/sync_group_to_annotated.py` |

---

## 步骤 3：因果发现 (`多种方法因果发现/`)

> 在真实时序数据上运行多种因果发现算法，输出 GraphML 格式的 DAG。
> 所有脚本共享 `causal_discovery_config.py` 作为数据加载和物理拓扑约束接口。

### 3.0 `causal_discovery_config.py` — 公共配置与数据加载（模块，非独立脚本）

| 项 | 说明 |
|----|------|
| **输入** | `data/timeseries_dataset_{line}_final.parquet` + 变量注释 Markdown |
| **输出** | 提供 `prepare_data(line)` 函数：返回 (df, X_cols, var_to_stage, var_to_group) |
| **功能** | 按产线 (xin1/xin2) 加载时序宽表，X 列 ffill，Y 列 ffill（零阶保持）后重采样至 10min，物理拓扑约束 `can_cause()` |

### 3.1 `run_tcdf_space_time_dag.py` — TCDF 因果发现

| 项 | 说明 |
|----|------|
| **输入** | `prepare_data(line)` 输出的时序 DataFrame |
| **输出** | `多种方法因果发现/因果发现结果/tcdf_real_dag_{line}.graphml` |
| **功能** | 注意力卷积神经网络 (TCDF)，基于时间和空间拓扑限制构建因果 DAG |
| **运行** | `python 多种方法因果发现/run_tcdf_space_time_dag.py --line xin2` |

### 3.2 `run_dynotears_dag.py` — DYNOTEARS 因果发现

| 项 | 说明 |
|----|------|
| **输入** | `prepare_data(line)` 输出的时序 DataFrame |
| **输出** | `多种方法因果发现/因果发现结果/dynotears_real_dag_{line}.graphml` |
| **功能** | 动态 NOTEARS，联合学习同期邻接矩阵 W + 滞后邻接矩阵 A_k，含物理拓扑掩码惩罚 |
| **运行** | `python 多种方法因果发现/run_dynotears_dag.py --line xin2 --lags 3` |

### 3.3 `run_granger_dag.py` — Granger 因果检验

| 项 | 说明 |
|----|------|
| **输入** | `prepare_data(line)` 输出的时序 DataFrame |
| **输出** | `多种方法因果发现/因果发现结果/granger_real_dag_{line}.graphml` |
| **功能** | 经典 Granger 因果检验 + BH-FDR 多重检验校正，作为深度学习方法的互补基线 |
| **运行** | `python 多种方法因果发现/run_granger_dag.py --line xin2 --lags 5` |

### 3.4 `run_innovation_real_data.py` — 创新因果发现算法

| 项 | 说明 |
|----|------|
| **输入** | `prepare_data(line)` 输出的时序 DataFrame |
| **输出** | `多种方法因果发现/因果发现结果/{algo}_real_dag_{line}.graphml`（biattn_cuts / multiscale_nts / mb_cuts） |
| **功能** | 运行三种创新因果发现算法：BiAttn-CUTS、MultiScale-NTS、MB-CUTS，全部施加物理拓扑掩码 |
| **运行** | `python 多种方法因果发现/run_innovation_real_data.py --line xin2` |

### 辅助脚本

| 脚本 | 功能 |
|------|------|
| `dml_causal_metrics.py` | DML 控制质量得分 (DML-CQS) 指标模块：CIS + BCES + IVP |
| `annotate_variables.py` | 与 `数据预处理/` 中同名脚本功能一致 |
| `test_config.py` | 配置验证脚本，检查 `prepare_data()` 对两条产线的输出 |

---

## 步骤 4：DAG 因果角色解析 (`DAG图分析/`)

### 4.1 `analyze_dag_causal_roles_v4_1.py` — DAG 角色解析器

| 项 | 说明 |
|----|------|
| **输入** | 步骤 3 输出的 `.graphml` 文件 + 变量注释 CSV（含 Group/Operability） |
| **输出** | `DAG图分析/DAG解析结果/causal_roles_{algo}_{line}.csv` — 每个操作变量的因果角色明细 |
| **功能** | 对每个可操作变量 (Treatment T) 识别：混杂因子 (Confounder)、中介变量 (Mediator)、碰撞节点 (Collider)、工具变量 (IV)，基于 Pearl SCM + 路径分析 |
| **运行** | `python DAG图分析/DAG解析结果/analyze_dag_causal_roles_v4_1.py --algo biattn_cuts --line xin2` |

> **关键接口**：输出 CSV 的格式为 `{Treatment_T, Role, Node_Name}`，由下游 DML 反驳实验的 `load_dag_roles()` 解析为 `{treatment: {confounders, mediators, colliders, instruments}}` 字典。

---

## 步骤 5：DML 因果效应估计 (`双重机器学习/`)

### 5.1 `compare_xin1_xin2.py` — 双产线 DML 效应对比

| 项 | 说明 |
|----|------|
| **输入** | 新1#/新2# 的 DML 结果 CSV（由反驳实验脚本产出） |
| **输出** | 终端打印对比表 |
| **功能** | 对比两条产线共享设备（MC 系列）和独立设备（FX 系列）的因果效应一致性 |
| **运行** | `python 双重机器学习/compare_xin1_xin2.py` |

---

## 步骤 6：DML 反驳性实验 (`反驳性实验/`)

> 核心脚本：在真实数据上运行 LSTM-VAE + DML 估计因果效应 θ，并通过安慰剂/随机混杂/数据子集等反驳实验验证稳健性。

### 6.1 `run_refutation_xin2_v3.py` — v3 两阶段解耦 VAE-DML

| 项 | 说明 |
|----|------|
| **输入** | `data/modeling_dataset_xin2_final.parquet` + DAG 角色 CSV |
| **输出** | `反驳性实验/{安慰剂实验,随机混杂变量实验,数据子集实验,稳定性诊断}/*.csv` |
| **功能** | Stage 1: 训练 VAE 编码器 → Stage 2: 冻结编码器，独立训练 Y/D 预测头 → DML 残差回归 → Bootstrap θ 聚合 |
| **运行** | `python 反驳性实验/run_refutation_xin2_v3.py --mode all` |

### 6.2 `run_refutation_xin2_v4.py` — v4 交叉拟合策略改进

| 项 | 说明 |
|----|------|
| **输入** | 同 v3 |
| **输出** | 同 v3（加 `_v4` 后缀），额外输出 `refutation_cf_compare_v4.csv` |
| **功能** | 在 v3 基础上新增：滑动窗口交叉拟合、分层折检查、折边随机化、嵌套学习率选择、策略对比实验 |
| **运行** | `python 反驳性实验/run_refutation_xin2_v4.py --mode all` 或 `--mode cf_compare` |

### 6.3 `run_refutation_xin2_v5.py` — v5 四项微创新

| 项 | 说明 |
|----|------|
| **输入** | 同 v3 |
| **输出** | 同 v3（加 `_v5` 后缀），额外输出消融实验结果 |
| **功能** | 在 v3/v4 基础上新增四项微创新：A. 因果优先梯度投影、B. 双流潜变量、C. 课程式训练、D. 不确定性加权 DML 残差，含消融实验 |
| **运行** | `python 反驳性实验/run_refutation_xin2_v5.py --mode all` 或 `--mode ablation` |

### 6.5 `tune_v5_hyperparameters.py` — V5 超参数调优

| 项 | 说明 |
|----|------|
| **输入** | 真实管线：`data/modeling_dataset_xin2_final.parquet`；模拟管线：`SyntheticDAGGenerator` 合成数据 |
| **输出** | `反驳性实验/超参数调优/` 下：最佳参数 JSON + 调优历史 CSV + 调优报告 TXT |
| **功能** | 使用贝叶斯优化（Optuna TPE）或准随机搜索，对 v5 四项微创新的核心超参数进行自动调优。支持真实/模拟双管线 |
| **运行** | `python 反驳性实验/tune_v5_hyperparameters.py --pipeline real --n_trials 30`<br>`python 反驳性实验/tune_v5_hyperparameters.py --pipeline simulation --n_trials 50`<br>`python 反驳性实验/tune_v5_hyperparameters.py --pipeline both --n_trials 30` |

> **调优方法选择**：优先使用 Optuna TPE 贝叶斯优化（`pip install optuna`），因为 v5 目标函数是计算昂贵的黑盒函数，TPE 比网格/随机搜索效率高 3-10 倍。若 Optuna 未安装，自动回退到准随机搜索。

### 辅助脚本

| 脚本 | 功能 |
|------|------|
| `check_progress.py` | 检查 DML 理论验证实验进度（读取 `DML理论验证/` 下的 CSV/JSON） |
| `evaluate_highdim_results.py` | **统一评估与横向对比**：读取 v1~v5 蒙特卡洛结果 CSV，生成跨方法对比报告 |

### 6.4 `evaluate_highdim_results.py` — 统一评估脚本

| 项 | 说明 |
|----|------|
| **输入** | `反驳性实验/DML理论验证/` 下所有蒙特卡洛结果 CSV（highdim_*.csv / monte_carlo_*.csv） |
| **输出** | `反驳性实验/评估报告/` 下：method_summary.csv + pairwise_tests.csv + consistency_comparison.csv + evaluation_report.txt |
| **功能** | 1. 跨方法横向对比（RMSE/Coverage/SE校准/正态性检验）<br>2. 成对统计检验（paired Wilcoxon signed-rank test，检验 v3 vs v4 差异显著性）<br>3. √n 一致性收敛斜率横向对比<br>4. 自动标记最优方法和 SE 校准异常 |
| **运行** | `python 反驳性实验/evaluate_highdim_results.py` |

> 使用方式：先运行各 `run_dml_theory_validation_highdim_*.py --mode full`，再执行评估脚本。支持 `--results_dir` 和 `--output_dir` 参数自定义路径。

---

# 二、合成数据管线

完整流程：`生成合成 DAG + 数据 → 蒙特卡洛基准测试 → 超参调优 → DML 理论验证`

> 本管线使用程序生成的已知因果结构数据，验证因果发现算法的 SHD/F1 和 DML 的无偏性/覆盖率等理论性质。

## 核心模块：`synthetic_dag_generator.py` (`因果的发现算法理论验证/`)

| 项 | 说明 |
|----|------|
| **输入** | 参数：n_nodes, graph_type (ER/SF/Layered), edge_funcs |
| **输出** | 邻接矩阵 `adj`、生成数据 `data`、因果角色字典 |
| **功能** | 生成 ER/Scale-Free/分层工业 DAG，支持线性/饱和/阈值/倒U/多项式/交互等非线性关系，提供 `identify_causal_roles()` / `find_adjustment_set()` / `generate_data(do_interventions=...)` |
| **类** | `SyntheticDAGGenerator` |

---

## S-1：因果发现算法基准测试 (`因果的发现算法理论验证/`)

### S-1.1 `run_monte_carlo_benchmark_fixed.py` — 蒙特卡洛基准测试

| 项 | 说明 |
|----|------|
| **输入** | `SyntheticDAGGenerator` 生成的随机 DAG + 数据（每次实验独立随机） |
| **输出** | `因果的发现算法理论验证/因果发现结果/monte_carlo_benchmark/` 下：各算法的详细结果 CSV + 统计汇总 CSV + Markdown 报告 + 算法对比表 |
| **功能** | N 次独立实验（默认 50 次），评估 CUTS+、NTS-NOTEARS、Coupled、MultiScale-NTS 四种算法的 SHD/TPR/FDR/F1 鲁棒性 |
| **运行** | `python 因果的发现算法理论验证/run_monte_carlo_benchmark_fixed.py --n_experiments 50` |

### S-1.2 `tune_multiscale_nts.py` — MultiScale-NTS 超参调优

| 项 | 说明 |
|----|------|
| **输入** | `SyntheticDAGGenerator` 生成的随机 DAG + 数据 |
| **输出** | `因果的发现算法理论验证/因果发现结果/hyperparameter_tuning/` 下：最佳参数 JSON + 调优报告 + 结果 CSV |
| **功能** | 网格搜索 window_size / kernel_sizes / hidden_mult / lr / epochs 等超参，以多次实验平均 F1 为评估指标 |
| **运行** | `python 因果的发现算法理论验证/tune_multiscale_nts.py --n_trials 5` |

---

## S-2：DML-CQS 增强版蒙特卡洛基准 (`多种方法因果发现/`)

### S-2.1 `run_monte_carlo_benchmark (2).py` — DML 控制质量评估

| 项 | 说明 |
|----|------|
| **输入** | `SyntheticDAGGenerator` 生成的随机 DAG + 数据 |
| **输出** | 蒙特卡洛结果 CSV（含 DML-CQS 三子指标及综合得分） |
| **功能** | 在基准测试基础上引入 `dml_causal_metrics.py` 的 DML-CQS 指标（CIS/BCES/IVP），评估因果发现结果对下游 DML 控制变量选择的质量 |
| **运行** | `python "多种方法因果发现/run_monte_carlo_benchmark (2).py"` |

---

## S-3：DML 理论验证 (`反驳性实验/`)

### S-3.1 `run_dml_theory_validation.py` — DML 理论性质验证

| 项 | 说明 |
|----|------|
| **输入** | `SyntheticDAGGenerator` 生成的已知因果结构合成数据 |
| **输出** | `反驳性实验/DML理论验证/` 下：蒙特卡洛结果 CSV + 汇总 JSON |
| **功能** | 验证 DML 五大理论性质：无偏性 E[θ̂]≈θ_true、√n-一致性、95% CI 覆盖率、渐近正态性、混杂控制有效性 |
| **运行** | `python 反驳性实验/run_dml_theory_validation.py --mode all` |

> 支持模式：`quick` (快速验证) / `full` (200 次蒙特卡洛) / `consistency` (不同 n) / `confounding` (混杂对比) / `all`

### S-3.2 `evaluate_highdim_results.py` — 统一评估与横向对比

| 项 | 说明 |
|----|------|
| **输入** | `反驳性实验/DML理论验证/` 下蒙特卡洛结果 CSV（v1~v5 全部） |
| **输出** | `反驳性实验/评估报告/` 下：method_summary.csv、pairwise_tests.csv、consistency_comparison.csv、evaluation_report.txt |
| **功能** | 跨方法横向对比 + paired Wilcoxon 检验 + SE 校准诊断 + 正态性检验 + √n 一致性对比 |
| **运行** | `python 反驳性实验/evaluate_highdim_results.py` |

---

# 三、数据流总览

```
┌─────────────────────── 真实数据管线 ───────────────────────┐
│                                                             │
│  DCS xlsx ──→ preprocess_X ──→ X_features_final.parquet     │
│  化验 xlsx ──→ preprocess_Y ──→ y_target_final.parquet      │
│  化验 xlsx ──→ preprocess_indicators ──→ indicators.parquet  │
│                        │                                    │
│                   merge_final                               │
│                   ╱          ╲                               │
│     timeseries_dataset       modeling_dataset                │
│     (128K 行, 1min)          (仅 Y 非 NaN 行)              │
│           │                        │                        │
│     causal_discovery_config        │                        │
│     prepare_data(line)             │                        │
│     (ffill X+Y, 重采样→12K行)     │                        │
│           │                        │                        │
│     ┌─────┴─────┐                  │                        │
│     │ TCDF      │                  │                        │
│     │ DYNOTEARS │─→ .graphml ──→ analyze_dag ──→ roles.csv  │
│     │ Granger   │                                │          │
│     │ Innovation│                                │          │
│     └───────────┘                                │          │
│                                                  ↓          │
│                              run_refutation_xin2_v3/v4/v5   │
│                              (LSTM-VAE + DML + 反驳实验)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────── 合成数据管线 ──────────────────────┐
│                                                        │
│  SyntheticDAGGenerator                                 │
│  (ER / Scale-Free / Layered)                           │
│           │                                            │
│     ┌─────┴─────────────────────────────┐              │
│     │                                   │              │
│     ↓                                   ↓              │
│  run_monte_carlo_benchmark       run_dml_theory_       │
│  (SHD/F1/DML-CQS 评估)         validation             │
│                                  (无偏性/覆盖率/       │
│  tune_multiscale_nts             √n-一致性)            │
│  (超参网格搜索)                                        │
│                                  tune_v5_hyper-        │
│                                  parameters            │
│                                  (贝叶斯调优)          │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

# 四、快速开始

## 环境准备

```bash
pip install pandas pyarrow numpy openpyxl python-calamine torch scipy networkx scikit-learn tqdm
```

## 真实数据管线（需准备原始数据文件）

```bash
# 1. 数据预处理（步骤 1.1~1.3 可并行）
python data_processing/preprocess_X.py
python data_processing/preprocess_Y.py
python data_processing/preprocess_indicators.py

# 2. 合并宽表
python data_processing/merge_final.py

# 3. 变量筛选（按需）
python 数据预处理/run_collinearity_detection.py

# 4. 因果发现
python 多种方法因果发现/run_tcdf_space_time_dag.py --line xin2
python 多种方法因果发现/run_innovation_real_data.py --line xin2

# 5. DAG 角色解析
python DAG图分析/DAG解析结果/analyze_dag_causal_roles_v4_1.py --algo biattn_cuts --line xin2

# 6. DML 反驳实验
python 反驳性实验/run_refutation_xin2_v5.py --mode all

# 7. V5 超参数调优（可选，优化 v5 四项微创新的超参）
python 反驳性实验/tune_v5_hyperparameters.py --pipeline real --n_trials 30
```

## 合成数据管线（无需外部数据）

```bash
# 蒙特卡洛基准测试
python 因果的发现算法理论验证/run_monte_carlo_benchmark_fixed.py --n_experiments 50

# 超参调优（因果发现算法）
python 因果的发现算法理论验证/tune_multiscale_nts.py --n_trials 5

# V5 超参调优（DML 创新方法，在模拟数据上调优）
python 反驳性实验/tune_v5_hyperparameters.py --pipeline simulation --n_trials 30

# DML 理论验证
python 反驳性实验/run_dml_theory_validation.py --mode all

# 统一评估（跨方法横向对比 + 统计检验）
python 反驳性实验/evaluate_highdim_results.py
```

---

# 五、关键设计决策

| 决策 | 说明 |
|------|------|
| **X 列只 ffill，禁止 bfill** | DCS exception-based recording：有变化才记录，稳定段不记录 ≠ 缺失。bfill = 用未来值回填 = 时间泄露 (lookahead leak) |
| **ffill 在完整 128K 行时序上执行** | 必须在 dropna(y) / 重采样之前执行，否则化验行间隔数小时，ffill 无法正确填充传感器稳定段 |
| **因果发现对 Y 做 ffill（零阶保持）** | 精矿品位在相邻化验间连续变化，ffill 不改变 X→Y 因果方向，保留完整时序结构（~12K 行 vs ~1K 稀疏行） |
| **DML 只用真实 Y 行** | 因果效应估计需要精确 Y 值，使用 `modeling_dataset`（仅含真实化验行），不受因果发现阶段的 Y ffill 影响 |
| **物理拓扑掩码** | `can_cause()` 基于 Stage 前馈流向和产线 Group 归属，硬约束不可行的因果方向 |
| **双产线分离** | xin1 使用 Group A+C 变量，xin2 使用 Group B+C 变量，互不干扰 |
