# FNO-CBAM SST 项目 — 数据说明文档

## 1. 数据总览

本项目使用两种 SST 数据源：

| 数据源 | 用途 | 空间分辨率 | 时间分辨率 | 覆盖区域 | 时间跨度 |
|--------|------|-----------|-----------|---------|---------|
| **OSTIA** | Stage 1 预训练 | ~0.02° (~2km) | 日 (daily) | 南海北部 15°N-24°N, 111°E-118°E | 1991-01 ~ 2021-12 |
| **JAXA L3** | Stage 2 微调 + 最终推理 | 0.02° (~2km) | 时 (hourly) | 同上 | 2015-07 ~ 2025-03 |

两者共享同一空间网格：**451×351 像素**（纬度×经度），覆盖中国南海北部海域（含广东沿岸、海南岛东侧、台湾海峡南端）。

---

## 2. JAXA L3 SST 数据

### 2.1 数据来源与存储

- **来源**: JAXA (日本宇宙航空研究开发机构) Himawari-8/9 卫星红外 SST L3 产品
- **原始路径**: `/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3/`
- **总文件数**: ~83,000 个 NC 文件
- **时间范围**: 2015-07 至 2025-03（约117个月）

### 2.2 目录结构

```
jaxa_extract_L3/
├── 201507/          ← YYYYMM 月份目录
│   ├── 07/          ← DD 日目录
│   │   ├── 20150707000000.nc    ← UTC 00:00
│   │   ├── 20150707010000.nc    ← UTC 01:00
│   │   ├── ...
│   │   └── 20150707230000.nc    ← UTC 23:00
│   ├── 08/
│   └── ...
├── 201508/
├── ...
├── 202503/
├── JAXA_extract.m           ← MATLAB 提取脚本
└── missing_date.txt         ← 缺失时次记录
```

每天最多 **24 个小时文件**，文件名格式为 `YYYYMMDDHHmmss.nc`。

### 2.3 NC 文件结构

```
维度:
  time: 1
  lat:  451
  lon:  351

变量:
  time                        int64   (time,)
    ├── units = "seconds since 1981-01-01 00:00:00"
  lat                         float32 (lat,)
    ├── units = "degrees_north"
    ├── 范围: 24.0 → 15.0（递减，row 0 = 24°N 北端）
    ├── 步长: -0.02°
  lon                         float32 (lon,)
    ├── units = "degrees_east"
    ├── 范围: 111.0 → 118.0（递增）
    ├── 步长: +0.02°
  sea_surface_temperature     float32 (time, lat, lon)
    ├── units = "kelvin"
    ├── 有效值范围: ~285K ~ 310K (约12°C ~ 37°C)
    ├── 缺失值: NaN（云遮挡或卫星无覆盖）
```

**注意**: lat 轴是**从北到南递减**的（24°N → 15°N），使用 `imshow` 时需要 `origin='upper'`，使用 `pcolormesh` 则会自动处理。

### 2.4 陆地掩码 (Land Mask)

JAXA NC 文件中**没有单独的陆地掩码变量**。陆地像素在所有时次的 SST 中都是 NaN。

获取陆地掩码的方法：
1. **多帧叠加法**（推荐）: 叠加多天多小时的数据，从未出现过有效SST值的像素 = 陆地
2. **使用OSTIA的land_mask**: OSTIA HDF5 中包含 `land_mask` 变量（1=陆地, 0=海洋），陆地像素约占 **15.3%**（24,235/158,301）

```python
# 方法1: 从JAXA数据构建 land mask
import netCDF4 as nc
import numpy as np
from pathlib import Path

JAXA_ROOT = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')
valid_count = np.zeros((451, 351), dtype=np.int32)

# 叠加多个晴天帧
for month in ['202303', '202304', '202404']:
    month_dir = JAXA_ROOT / month
    for day_name in sorted([d.name for d in month_dir.iterdir() if d.is_dir()])[:10]:
        for hour in [0, 6, 12, 18]:
            fpath = month_dir / day_name / f"{month}{day_name}{hour:02d}0000.nc"
            if fpath.exists():
                with nc.Dataset(fpath, 'r') as ds:
                    sst = ds.variables['sea_surface_temperature'][:].squeeze()
                valid_count += (~np.isnan(sst)).astype(np.int32)

land_mask = (valid_count == 0)  # 从未有有效值的像素 = 陆地
# land_mask: True=陆地, False=海洋
```

```python
# 方法2: 直接使用OSTIA的land_mask
import h5py

with h5py.File('/data/sst_data/sst_missing_value_imputation/processed_data/processed_sst_train.h5', 'r') as f:
    land_mask = f['land_mask'][:].astype(bool)  # (451, 351), True=陆地
```

**陆地位置**: 主要在网格的**上方**（北侧，23°N-24°N 广东沿海）和**右上角**（东北侧，台湾海峡方向）。

### 2.5 缺失率分析

JAXA SST 的缺失主要由**云层遮挡**导致。红外卫星无法穿透云层获取 SST。

#### 2.5.1 季节特征（最主要影响因素）

| 季节 | 统计天数 | 缺失率均值 | 缺失率中位数 | <40%天数（数据好） | >90%天数（基本全缺） |
|------|---------|-----------|-------------|------------------|-------------------|
| **春 (3-5月)** | 853 | **55.0%** | **51.3%** | **240天 (28%)** | 73天 (9%) |
| 夏 (6-8月) | 873 | 70.7% | 73.4% | 113天 (13%) | **250天 (29%)** |
| 秋 (9-11月) | 907 | 67.0% | 69.1% | 124天 (14%) | 156天 (17%) |
| 冬 (12-2月) | 901 | 70.5% | 73.7% | 88天 (10%) | 179天 (20%) |

- **春季数据质量最好**: 超过1/4的天缺失率<40%，平均缺失率仅55%
- **夏季最差**: 梅雨和台风季，近30%的天基本全缺失
- **冬季次差**: 冬季季风云系影响

#### 2.5.2 缺失率最低的月份（精选）

| 月份 | 缺失率均值 | <40%天数 | 说明 |
|------|-----------|---------|------|
| 2023-03 | 36.8% | 19/31天 | 数据最好的月份 |
| 2024-04 | 37.8% | 19/30天 | |
| 2023-05 | 40.4% | 18/31天 | |
| 2024-02 | 41.2% | 15/29天 | |
| 2021-05 | 42.1% | 12/31天 | |

#### 2.5.3 缺失率最低的日期 (Top 10)

| 日期 | 缺失率 | 季节 |
|------|--------|------|
| 2023-06-20 | 17.9% | 夏 |
| 2023-06-21 | 17.9% | 夏 |
| 2023-07-09 | 18.0% | 夏 |
| 2023-07-10 | 18.2% | 夏 |
| 2023-09-20 | 18.5% | 秋 |
| 2023-09-21 | 18.8% | 秋 |
| 2023-07-11 | 19.0% | 夏 |
| 2023-03-08 | 19.2% | 春 |
| 2024-08-02 | 19.5% | 夏 |
| 2025-03-23 | 19.6% | 春 |

最优情况下缺失率也有约18%，说明即使晴天仍有部分区域无法获取SST（海岸线附近、卫星覆盖边缘等）。

#### 2.5.4 完全缺失的日期 (100%)

台风过境等极端天气会导致整天所有小时全部缺失：

| 日期 | 季节 | 可能原因 |
|------|------|---------|
| 2016-08-02 | 夏 | 台风 |
| 2017-07-17 | 夏 | 台风 |
| 2018-06-13 | 夏 | 梅雨 |
| 2020-08-01 | 夏 | 台风 |
| ... | | |

全缺失天主要集中在**夏季和秋季**（台风活跃期）。

#### 2.5.5 白天 vs 夜间

| 指标 | 白天 (JST 06-17) | 夜间 (JST 18-05) | 差异 |
|------|------------------|------------------|------|
| 缺失率均值 | 69.6% | 70.9% | -1.3% |
| <50%帧占比 | 24.1% | 23.3% | +0.8% |
| >90%帧占比 | 23.5% | 28.8% | -5.4% |

日夜差异**非常小**。夜间数据**并非全缺失**，与白天质量基本相当。这符合红外SST的物理特性——缺失由云遮挡而非日照决定。

### 2.6 数据加载方式

```python
# 加载单帧JAXA数据
import netCDF4 as nc
import numpy as np

fpath = '/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3/202303/04/20230304030000.nc'
with nc.Dataset(fpath, 'r') as ds:
    sst = ds.variables['sea_surface_temperature'][:].squeeze()  # (451, 351), Kelvin, NaN=缺失
    lat = ds.variables['lat'][:]   # (451,) 24.0→15.0 递减
    lon = ds.variables['lon'][:]   # (351,) 111.0→118.0 递增

sst_celsius = sst - 273.15  # 转换为摄氏度
missing_mask = np.isnan(sst).astype(np.float32)  # 1=缺失, 0=有效
```

---

## 3. OSTIA SST 数据

### 3.1 数据来源与存储

- **来源**: OSTIA (Operational Sea Surface Temperature and Sea Ice Analysis)，英国气象局多源融合 SST 分析产品
- **特点**: 融合了多颗卫星和浮标数据，**无缺失值**（完整的全球SST分析场）
- **路径**: `/data/sst_data/sst_missing_value_imputation/processed_data/`
- **格式**: HDF5，已处理为训练/验证/测试集

### 3.2 数据集划分

| 集合 | 文件 | 样本数 | 时间范围 |
|------|------|--------|---------|
| **Train** | `processed_sst_train.h5` | 7,882 | 1991-01-01 ~ 2012-07-30 |
| **Valid** | `processed_sst_valid.h5` | 1,689 | 2012-07-31 ~ 2017-05-16 |
| **Test** | `processed_sst_test.h5` | 1,690 | 2017-05-17 ~ 2021-12-31 |

总计 **11,261 天** (约30.8年) 的每日 SST 数据。

### 3.3 HDF5 文件结构

```
数据集:
  ground_truth_sst        float32  (N, 451, 351)    完整的SST真值 (Kelvin)
  input_sst               float32  (N, 451, 351)    人工制造缺失后的输入SST
  missing_mask            uint8    (N, 451, 351)    缺失掩码 (1=缺失, 0=有效)
  effective_cloud_mask    uint8    (N, 451, 351)    云掩码 (1=云, 0=晴)
  land_mask               uint8    (451, 351)       陆地掩码 (1=陆地, 0=海洋)，所有帧共享
  latitude                float32  (451,)           15.02 → 23.98 (递增，与JAXA方向相反!)
  longitude               float32  (351,)           111.03 → 117.97 (递增)
  time_index              int64    (N,)             时间索引
  time_iso                object   (N,)             ISO时间字符串
```

**关键区别**: OSTIA的 lat 是 **15°→24° 递增**（从南到北），而 JAXA 的 lat 是 **24°→15° 递减**（从北到南）。两者的 `land_mask` 和 SST 数据阵列的行方向相反。

### 3.4 数据字段说明

| 字段 | 说明 |
|------|------|
| `ground_truth_sst` | OSTIA 完整的 SST 分析场（无缺失），单位 Kelvin。训练时作为标签(target)。陆地区域为 NaN。 |
| `input_sst` | 人工制造缺失后的输入。缺失区域填充为 **0**（非NaN）。模型看到的输入。 |
| `missing_mask` | 1=该像素被人工遮蔽（缺失），0=保留的有效观测。海洋区域缺失率约 **37%**（中位40%）。 |
| `effective_cloud_mask` | 用于生成缺失的云掩码模板。来源于真实的卫星云图，保证缺失形态逼真。 |
| `land_mask` | 陆地掩码，1=陆地，0=海洋。陆地像素占 15.3%（24,235 个）。所有帧共享同一份。 |

### 3.5 数据统计

| 指标 | 值 |
|------|------|
| SST 范围 | 约 287K ~ 304K (14°C ~ 31°C) |
| SST 均值 | ~297.4K (24.2°C) |
| SST 标准差 | ~1.85K |
| 海洋区域缺失率 | 均值37.2%, 中位40.0%, 范围7.5%~55.7% |
| 陆地占比 | 15.3% |

### 3.6 数据加载方式

```python
import h5py
import numpy as np

path = '/data/sst_data/sst_missing_value_imputation/processed_data/processed_sst_train.h5'
with h5py.File(path, 'r') as f:
    # 单帧加载
    gt_sst = f['ground_truth_sst'][idx]      # (451,351) Kelvin, NaN=陆地
    input_sst = f['input_sst'][idx]           # (451,351) Kelvin, 0=缺失
    mask = f['missing_mask'][idx]             # (451,351) 1=缺失
    land_mask = f['land_mask'][:]             # (451,351) 1=陆地

    # 坐标
    lat = f['latitude'][:]                    # (451,) 15→24 递增
    lon = f['longitude'][:]                   # (351,) 111→118 递增

    # 时间
    time_str = f['time_iso'][idx]             # ISO时间字符串

# 注意: 归一化参数 (OSTIA训练集统计)
# mean = 299.92 K, std = 2.69 K
# JAXA微调也使用相同归一化参数
```

---

## 4. 预处理中间数据

JAXA 原始数据经过三步预处理后送入模型训练：

### 4.1 预处理Pipeline

```
JAXA hourly NC → [Step1] 时间加权填充(48h lookback) → [Step2] 低通滤波 → [Step3] 3D时空KNN填充(120h causal) → 模型输入
```

| 步骤 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| **Step 1** | `preprocessing/temporal_weighted_fill.py` | JAXA hourly NC | `jaxa_weighted_aligned/jaxa_weighted_series_XX.h5` |
| **Step 2** | `preprocessing/lowpass_filter.py` | `jaxa_weighted_series_XX.h5` | `jaxa_filtered/jaxa_filtered_XX.h5` |
| **Step 3** | `preprocessing/knn_fill_3d.py` | `jaxa_filtered_XX.h5` | `jaxa_knn_filled/jaxa_knn_filled_XX.h5` |

### 4.2 9个年度序列划分

数据按年度划分为9个序列（series 0-8），每个序列约1年：

| Series | 起止时间 | 用途 |
|--------|---------|------|
| 0 | 2017-07-06 ~ 2018-07-05 | 训练 |
| 1 | 2016-07-06 ~ 2017-07-05 | 训练 |
| 2 | 2021-07-05 ~ 2022-07-04 | 训练 |
| 3 | 2018-07-06 ~ 2019-07-04 | 训练 |
| 4 | 2019-07-07 ~ 2020-07-04 | 训练 |
| 5 | 2020-07-05 ~ 2021-07-04 | 训练 |
| 6 | 2022-07-05 ~ 2023-07-04 | 训练 |
| 7 | 2023-07-05 ~ 2024-07-03 | 训练 |
| 8 | 2024-07-04 ~ 2025-03-30 | **验证** |

### 4.3 中间数据路径

| 目录 | 内容 | 文件数 |
|------|------|--------|
| `jaxa_weighted_aligned/` | Step1 输出 (时间加权填充后) | 12 (含统计JSON) |
| `jaxa_filtered/` | Step2 输出 (低通滤波后) | 11 |
| `jaxa_knn_filled/` | Step3 输出 (KNN填充后, 模型输入) | 10 (9序列 + 统计) |
| `sst_knn_npy_cache/` | Step3 的 npy 缓存 (用于共享内存加载) | 36 |

### 4.4 npy 缓存说明

训练时通过 `preload_shared_data()` 加载 npy 文件到 PyTorch 共享内存，避免磁盘 IO 瓶颈：

```
sst_knn_npy_cache/
├── sst_00.npy ~ sst_08.npy       # SST数据 (float32)
├── obs_00.npy ~ obs_08.npy       # 原始观测掩码 (float32)
├── miss_00.npy ~ miss_08.npy     # 原始缺失掩码 (float32)
└── land_00.npy ~ land_08.npy     # 陆地掩码 (float32)
```

---

## 5. 关键脚本索引

### 5.1 数据预处理

| 脚本 | 说明 |
|------|------|
| `preprocessing/temporal_weighted_fill.py` | JAXA hourly数据时间加权填充，48h causal lookback，日周期感知（同时刻观测权重更高） |
| `preprocessing/lowpass_filter.py` | 低通滤波去噪 |
| `preprocessing/knn_fill.py` | 渐进式2D KNN空间填充（旧版，已弃用） |
| `preprocessing/knn_fill_3d.py` | 3D时空KNN填充，因果性（过去120h），渐进密度分带，216核并行 |
| `preprocessing/post_knn_filter.py` | KNN填充后的后处理滤波 |
| `scripts/run_preprocessing.sh` | 预处理总启动脚本 |

### 5.2 数据集定义

| 脚本 | 说明 |
|------|------|
| `datasets/ostia_dataset.py` | OSTIA预训练数据集 (`SSTDatasetTemporal`)，30天窗口，HDF5按需读取 |
| `datasets/ostia_dataset_filled.py` | OSTIA预填充加速版数据集 |
| `datasets/jaxa_dataset.py` | JAXA数据集（早期版本） |
| `inference/jaxa_inference_dataset.py` | JAXA微调数据集 (`JAXAFinetuneDataset`)，共享内存版，含人工挖空 |

### 5.3 训练与推理

| 脚本 | 说明 |
|------|------|
| `training/train_ostia.py` | OSTIA预训练 (8 GPU DDP, Output Composition) |
| `training/train_jaxa.py` | JAXA微调 (4-8 GPU DDP, 共享内存, 实验管理) |
| `training/resume_train_jaxa.py` | 断点续训 |
| `inference/fill_jaxa.py` | JAXA数据推理填充 |
| `inference/evaluate.py` | 模型评估 (VRMSE, MAE, RMSE) |
| `sst_pipeline/pipeline.py` | 统一推理Pipeline |

### 5.4 陆地掩码获取

| 方式 | 位置 |
|------|------|
| OSTIA land_mask | `processed_sst_train.h5` → `land_mask` 字段 (451×351, uint8) |
| JAXA npy缓存 | `sst_knn_npy_cache/land_XX.npy` (float32) |
| 多帧叠加构建 | 见 2.4 节代码示例 |

---

## 6. 归一化参数

训练时使用的归一化参数（OSTIA训练集统计）:

```python
mean = 299.92   # Kelvin (约26.8°C)
std  = 2.69     # Kelvin

# JAXA微调使用的参数（OSTIA预训练阶段确定）
ostia_mean = 299.9221  # Kelvin
ostia_std  = 2.6919    # Kelvin

# 归一化:  sst_norm = (sst - mean) / std
# 反归一化: sst = sst_norm * std + mean
```

**重要**: JAXA微调阶段沿用 OSTIA 的归一化参数，不重新计算。

---

## 7. 坐标系注意事项

| 属性 | JAXA NC | OSTIA HDF5 |
|------|---------|------------|
| lat 方向 | **24°→15° 递减** (row 0 = 北端) | **15°→24° 递增** (row 0 = 南端) |
| lon 方向 | 111°→118° 递增 | 111°→118° 递增 |
| lat 步长 | -0.02° | +0.02° |
| 缺失表示 | NaN | missing_mask=1, input_sst=0 |
| 陆地表示 | NaN (无单独字段) | land_mask=1 |

如果需要混用两种数据，注意 **lat 方向相反**，需要 `np.flip(data, axis=0)` 统一方向。

---

## 8. OSTIA 推理流程

### 8.1 推理脚本

**脚本**: `inference/fill_ostia.py`

**用途**: 在OSTIA验证集/测试集上评估预训练模型的填充效果。由于OSTIA有完整的ground truth，可以直接计算误差指标。

### 8.2 推理逻辑

```
1. 加载 OSTIA HDF5 数据集 (SSTDatasetTemporal)
   └─ 每个样本返回 30天 input_sst_seq + mask_seq + ground_truth

2. 逐样本推理:
   ├─ 输入: sst_seq [30, 451, 351] (归一化, 缺失区域已最近邻插值)
   ├─       mask_seq [30, 451, 351] (1=缺失, 0=观测)
   ├─ 模型前向: model(sst_tensor, mask_tensor) → pred [1, 451, 351]
   └─ Output Composition:
       ├─ 观测区域 (mask=0): 保留 input_sst_seq[29] (第30天输入)
       └─ 缺失区域 (mask=1): 使用 pred (模型预测)

3. 评估 (仅在海洋缺失区域):
   ├─ MAE  = mean(|pred - gt|) × norm_std → Kelvin
   ├─ RMSE = sqrt(mean((pred - gt)²)) × norm_std → Kelvin
   └─ VRMSE = RMSE / std(gt)

4. 输出:
   ├─ 评估指标文件: ostia_filled_output/evaluation_results.txt
   └─ 4连图可视化: ostia_filled_visualization/ (每10个样本一张)
       ├─ Panel 1: Input SST (Day 30, 含插值填充)
       ├─ Panel 2: Ground Truth SST
       ├─ Panel 3: FNO-CBAM Output
       └─ Panel 4: Error Map (仅缺失区域)
```

### 8.3 运行命令

```bash
# 默认参数运行
python inference/fill_ostia.py

# 自定义参数
python inference/fill_ostia.py \
    --val_data_path /path/to/processed_sst_valid.h5 \
    --model_path /path/to/best_model.pth \
    --output_dir ./ostia_filled_output \
    --vis_dir ./ostia_filled_visualization \
    --gpu_id 0 \
    --num_samples 100 \
    --vis_interval 10
```

### 8.4 关键配置 (脚本内 DEFAULT_CONFIG)

```python
val_data_path = '.../processed_data/processed_sst_valid.h5'
model_path    = '.../experiments/ostia_pretrain/best_model.pth'
output_dir    = '.../ostia_filled_output'
vis_dir       = '.../ostia_filled_visualization'
gpu_id        = 0
vis_interval  = 10   # 每10个样本生成一张4连图
```

### 8.5 数据流 (OSTIA)

```
processed_sst_valid.h5
    ↓
SSTDatasetTemporal (datasets/ostia_dataset.py)
    ├─ 构建30天滑动窗口
    ├─ 缺失区域最近邻插值填充 → input_sst_seq
    ├─ 归一化: (sst - mean) / std
    └─ 前29天不足时用首帧padding
    ↓
model(sst_seq, mask_seq)
    ↓
Output Composition: final = input*(1-mask) + pred*mask
    ↓
反归一化: sst = final * std + mean → Kelvin
    ↓
与 ground_truth 比较 → MAE / RMSE / VRMSE
```

---

## 9. JAXA 推理流程

### 9.1 推理脚本

| 脚本 | 说明 |
|------|------|
| `inference/fill_jaxa.py` | JAXA SST填充 — 混合输入方案，逐帧推理并保存NC |
| `inference/run_inference_vis.py` | JAXA推理+5连图可视化（含人工挖空评估） |
| `inference/evaluate.py` | JAXA模型定量评估（在观测区域人工挖空后比较） |
| `inference/vis_5panel_postfiltered.py` | 5面板可视化（含高斯滤波后效果） |
| `sst_pipeline/pipeline.py` | 统一Pipeline API（一键处理单日期） |

### 9.2 核心推理逻辑 (fill_jaxa.py — 混合输入方案)

JAXA推理与OSTIA不同，因为JAXA没有"无缺失的ground truth"——数据本身就有真实云缺失，目标是填补这些真实缺失。

```
输入构建 (混合方案):
    前29天: KNN填充后的完整数据 (jaxa_knn_filled_XX.h5)
            → 提供丰富的历史时序信息，无NaN
    第30天: 低通滤波数据 (jaxa_filtered_XX.h5)
            → 保留真实观测，缺失区域用全局均值(299.92K)填充

推理:
    1. 构建30天序列: sst_seq [30, 451, 351]
    2. 归一化: (sst - 299.92) / 2.69
    3. 模型推理: model(sst_tensor, mask_tensor) → pred
    4. 反归一化: pred * 2.69 + 299.92 → Kelvin
    5. Output Composition:
       ├─ 观测区域: 保留滤波数据 (真实观测)
       └─ 缺失区域: 使用模型预测
    6. 陆地区域设为 NaN

输出:
    ├─ NC文件 (每帧一个): jaxa_filled_output/series_XX/jaxa_filled_YYYYMMDD.nc
    │   包含: sst_filled, sst_knn, sst_filtered, original_missing_mask
    ├─ 4连图可视化: jaxa_filled_visualization/
    │   ├─ Panel 1: Filtered JAXA SST (真实缺失显示)
    │   ├─ Panel 2: KNN Filled SST (粗填充)
    │   ├─ Panel 3: FNO-CBAM Filled SST (模型填充)
    │   └─ Panel 4: Model-KNN差异图 (仅缺失区域)
    └─ GIF动画: jaxa_filled_animation.gif
```

### 9.3 评估逻辑 (evaluate.py)

由于JAXA缺失区域没有ground truth，评估使用**人工挖空**策略：

```
1. 在原始观测区域(已知真值)内随机挖空 20% 的方形区域
2. 挖空位置用均值填充作为模型输入
3. 模型预测挖空区域
4. 比较预测值 vs 原始KNN值(=真实观测) → 得到误差指标
5. 可选: 对预测结果做高斯滤波(σ=1.0)后再评估
```

### 9.4 统一Pipeline (sst_pipeline/)

提供Python API和CLI两种方式：

```python
# Python API
from sst_pipeline import Pipeline
pipeline = Pipeline()
result = pipeline.process(
    date="2017-08-08",
    apply_gaussian=True, sigma=1.0,
    visualize=True, save_nc=True
)
```

```bash
# CLI
python -m sst_pipeline.run --date 2017-08-08 --sigma 1.0
python -m sst_pipeline.run --start-date 2017-08-01 --end-date 2017-08-10
python -m sst_pipeline.run --list-dates
```

Pipeline 内部流程:

```
sst_pipeline/
├── config.py           # PathConfig / ModelConfig / PostprocessConfig 等
├── data/loader.py      # 从 jaxa_knn_filled 加载30天窗口
├── model/wrapper.py    # 加载 FNO_CBAM_SST_Temporal 模型
├── inference/predictor.py  # 推理 + Output Composition
├── postprocess/gaussian.py # 高斯滤波 (σ=1.0)
└── visualization/plotter.py # 结果可视化

流程:
load_window(date) → prepare_model_input() → predict_with_composition()
    → GaussianPostProcessor(σ=1.0) → save_nc() + plot_reconstruction()
```

### 9.5 运行命令

```bash
# JAXA填充推理 (逐帧, 保存NC)
python inference/fill_jaxa.py

# JAXA评估 (人工挖空, 计算VRMSE)
python inference/evaluate.py

# JAXA推理+5连图可视化
python inference/run_inference_vis.py

# 统一Pipeline
python -m sst_pipeline.run --date 2017-08-08 --sigma 1.0
```

### 9.6 关键配置 (fill_jaxa.py 脚本内)

```python
KNN_FILLED_DIR = '.../jaxa_knn_filled'        # KNN填充数据 (前29天)
FILTERED_DIR   = '.../jaxa_filtered'           # 滤波数据 (第30天)
MODEL_PATH     = '.../experiments/jaxa_finetune_8years/best_model.pth'
OUTPUT_DIR     = '.../jaxa_filled_output'
VIS_DIR        = '.../jaxa_filled_visualization'
WINDOW_SIZE    = 30
GPU_ID         = 4
SERIES_IDS     = [0]   # 要处理的序列ID
```

### 9.7 OSTIA vs JAXA 推理对比

| 维度 | OSTIA 推理 | JAXA 推理 |
|------|-----------|-----------|
| **脚本** | `inference/fill_ostia.py` | `inference/fill_jaxa.py` |
| **目的** | 评估预训练模型效果 | 填补真实云缺失 |
| **输入数据** | OSTIA HDF5 (SSTDatasetTemporal) | JAXA KNN_filled + filtered H5 |
| **30天输入** | 30天都走同一个dataset | 前29天KNN + 第30天滤波 |
| **缺失处理** | 最近邻插值填充 | 缺失区域用全局均值(299.92K) |
| **有无GT** | 有完整ground truth | 无 (真实缺失) |
| **评估方式** | 直接与GT比较 | 人工挖空后比较 (evaluate.py) |
| **输出格式** | 可视化PNG + 指标TXT | NC文件 + PNG + GIF |
| **后处理** | 无 | 可选高斯滤波(σ=1.0) |
| **模型权重** | `ostia_pretrain/best_model.pth` | `jaxa_finetune_8years/best_model.pth` |

---

## 10. 模型权重路径索引

| 模型 | 路径 | 说明 |
|------|------|------|
| OSTIA预训练 | `experiments/ostia_pretrain/best_model.pth` | Stage 1 |
| JAXA微调 (8年) | `experiments/jaxa_finetune_8years/best_model.pth` | Stage 2 基线 |
| JAXA微调 (最新) | `experiments/run004_jaxa_3dknn_progressive_stride1_lr0.0005/best_model.pth` | 3D KNN + stride=1 |
| JAXA微调 (各实验) | `experiments/run004_*/best_model.pth` | 消融实验 |
