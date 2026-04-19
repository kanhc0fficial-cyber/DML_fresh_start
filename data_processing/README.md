# data_processing/ 操作指南

本目录包含4个针对选矿厂 DCS + 化验场景的数据预处理脚本，专为双重机器学习 (DML) / 因果发现场景设计。

---

## 目录结构

```
data_processing/
├── preprocess_X.py          # X 特征主处理脚本（DCS 传感器数据）
├── preprocess_indicators.py # 化验指标预处理脚本（非品位类化验结果）
├── preprocess_Y.py          # Y（精矿品位）变量处理脚本
├── merge_final.py           # 最终 X/Y/指标对齐宽表生成脚本
└── README.md                # 本操作指南
```

---

## 数据放置要求

在运行脚本之前，请将本地数据按以下结构放置到仓库根目录的 `data/` 下：

```
data/
├── 大量长时间数据/                        # DCS 批次目录
│   ├── 10.13/
│   │   ├── xxx.xlsx                      # xlsx 格式，每个 sheet=变量名，列含 time/value/quality
│   │   └── ...
│   ├── 10.22/
│   ├── 11.4~10.22/
│   ├── 12.12/
│   ├── 12.25数据/
│   ├── 1.9-1.29数据/
│   └── 3.12plc数据/
│
├── 操作变量和混杂变量/                    # 白名单来源
│   ├── output_MC_可用版.csv               # 磨矿/磁选操作变量（需含 NAME 列）
│   └── output_最终可用变量合集_无digital版.csv  # 浮选变量无数字型（需含 NAME 列）
│
├── 化验数据/                              # 化验记录 Excel（供 preprocess_indicators 和 preprocess_Y 使用）
│   ├── 化验记录_2023.xlsx
│   ├── 化验记录_2024.xlsx
│   └── ...
│
└── .cache/                               # 自动生成，断点续传缓存，无需手动创建
```

> **说明**：`data/` 目录不在版本控制中，脚本会自动创建缺少的子目录（如 `.cache/`）。

---

## 脚本说明与运行顺序

请按以下顺序运行（步骤 1~3 可并行，步骤 4 需在 1~3 完成后运行）：

### 步骤 1：运行 preprocess_X.py（约需数分钟 ~ 数小时，取决于数据量）

**功能**：从 DCS 批次目录读取所有 xlsx 文件，按白名单筛选变量，合并多批次数据，生成 1min 频率 X 特征宽表。

**关键设计**：
- **物理硬阈值裁剪**：超范围值直接置 NaN，不做插值平滑
- **短缺失 ffill**：连续缺失 ≤ 60min 的段做前向填充（模拟 DCS 断点重连）
- **大缺失保留 NaN**：不用全局中位数/噪声填充，不做 EWMA 平滑（保护因果时滞结构）
- **Hash 刷新机制**：白名单文件 md5 变更时自动失效所有批次缓存

```bash
python data_processing/preprocess_X.py
```

**输出**：`data/X_features_final.parquet`

---

### 步骤 2：运行 preprocess_Y.py

**功能**：从化验记录 Excel 中自动识别"新1#"和"新2#"精矿品位列，生成稀疏时间戳的 Y 目标变量。

**关键设计**：
- 自动匹配列名/sheet名中含"新1"/"新2"/"品位"等关键词的数据
- 时间戳对齐到分钟精度（floor）
- 物理范围裁剪（0~100%），超范围赋 NaN，不做任何插值

```bash
python data_processing/preprocess_Y.py
```

**输出**：`data/y_target_final.parquet`（含列 `y_fx_xin1`、`y_fx_xin2`）

> **若自动识别失败**：请查看日志 `data/preprocess_Y_log.txt`，根据实际列名修改脚本顶部的 `_PATTERN_XIN1` / `_PATTERN_XIN2` 正则表达式。

---

### 步骤 3：运行 preprocess_indicators.py

**功能**：从化验记录 Excel 中提取非品位类化验指标（如粒度、pH、矿浆浓度等），生成 1min 频率的指标宽表。

**关键设计**：
- 自动扫描 Excel 中各小表块，不依赖固定行列索引
- **8:30 单点指标**（如班次品位）：ffill 到次日同时刻（最多 1440min）
- **其余多点指标**：ffill 到下一个真实观测点

```bash
python data_processing/preprocess_indicators.py
```

**输出**：`data/indicators_final.parquet`

---

### 步骤 4：运行 merge_final.py（需先完成步骤 1~3）

**功能**：以 Y 的化验时间点为锚，用 ±1min 容差将 X 特征和化验指标对齐，剔除 X 缺失率 > 50% 的行，生成最终建模宽表。

```bash
# 生成完整宽表（包含两条产线所有 Y 点）
python data_processing/merge_final.py

# 仅生成新1#产线的建模数据集
python data_processing/merge_final.py --line xin1

# 仅生成新2#产线的建模数据集
python data_processing/merge_final.py --line xin2

# 自定义输出路径
python data_processing/merge_final.py --output /path/to/my_dataset.parquet
```

**输出**：
- `data/modeling_dataset_final.parquet`（完整宽表）
- `data/modeling_dataset_xin1_final.parquet`（新1#子集，当 SAVE_PER_LINE=True 时）
- `data/modeling_dataset_xin2_final.parquet`（新2#子集，当 SAVE_PER_LINE=True 时）

---

## 输出文件格式

| 文件 | index | 频率 | 说明 |
|------|-------|------|------|
| `X_features_final.parquet` | DatetimeIndex (time) | 1min | X 特征宽表，列=传感器 TAG 名（大写） |
| `y_target_final.parquet` | DatetimeIndex (time) | 不规则（化验频率） | Y 目标，列=y_fx_xin1/y_fx_xin2 |
| `indicators_final.parquet` | DatetimeIndex (time) | 1min（ffill 后） | 化验指标宽表 |
| `modeling_dataset_final.parquet` | DatetimeIndex (time) | 不规则（Y 锚点） | 建模用宽表，含全部 X + 指标 + Y 列 |

---

## 常见问题

**Q: preprocess_X.py 运行很慢？**
- 安装 `pip install python-calamine` 可获得约 25~200 倍的 xlsx 读取加速
- 第二次运行会利用批次缓存，速度大幅提升

**Q: preprocess_Y.py 提示未找到新1#/新2#数据？**
- 检查日志 `data/preprocess_Y_log.txt`，查看实际解析到的列名
- 如列名不匹配，修改脚本顶部 `_PATTERN_XIN1` / `_PATTERN_XIN2` 正则表达式

**Q: 修改了白名单后，想强制重新处理所有批次？**
- 删除 `data/.cache/` 目录，或修改任意白名单 CSV 内容（hash 会变化，缓存自动失效）

**Q: 如何添加新批次数据？**
- 将新批次目录路径追加到 `preprocess_X.py` 的 `DATA_BATCH_DIRS` 列表末尾，重新运行即可
- 已有批次缓存不受影响，只有新批次会被重新处理

**Q: 如何配置物理阈值？**
- 在 `preprocess_X.py` 的 `PHYSICAL_BOUNDS` 字典中按 TAG 名添加 (lo, hi) 元组
- 在 `preprocess_Y.py` 中修改 `Y_STRICT_LO` / `Y_STRICT_HI` 可收紧品位范围

---

## 依赖安装

```bash
pip install pandas pyarrow numpy openpyxl python-calamine
```

---

## 日志文件

每个脚本运行后在 `data/` 下生成对应日志：

- `data/preprocess_X_log.txt`
- `data/preprocess_indicators_log.txt`
- `data/preprocess_Y_log.txt`
- `data/merge_final_log.txt`
