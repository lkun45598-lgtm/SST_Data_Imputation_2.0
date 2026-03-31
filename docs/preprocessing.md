# 数据预处理流程

## 概述

JAXA卫星SST数据预处理Pipeline，将原始hourly NC文件转换为模型可用的30天时间序列H5数据。

**注意**: 移除了低通滤波步骤，因为会抹平真实的高频洋流细节（锋面），让FNO自己学习如何过滤噪声。

## 预处理流程图

```
JAXA原始数据 (hourly NC)
        │
        ▼
┌─────────────────────────────┐
│  Step 1: 时间加权填充        │
│  temporal_weighted_fill.py  │
│  hourly → daily            │
│  逆时间距离加权填充缺失值    │
└─────────────────────────────┘
        │
        ▼
    jaxa_weighted_aligned/*.h5
        │
        ▼
┌─────────────────────────────┐
│  Step 2: 3D时空KNN填充      │
│  knn_fill_3d.py             │
│  渐进密度分带, 因果性(过去   │
│  120h), 216核并行           │
└─────────────────────────────┘
        │
        ▼
    jaxa_knn_filled/*.h5  ← 模型输入数据
```

---

## Step 1: 时间加权填充

**脚本**: `preprocessing/temporal_weighted_fill.py`

### 算法原理

对于目标时刻 `t` 的每个缺失像素，在过去48小时内同一像素位置的历史观测中做加权平均：

```
hour_diff = min(|Δh| % 24, 24 - |Δh| % 24)   # 环形小时距离 (0~12)
weight = (1 / dt) × exp(-0.5 × hour_diff)      # 逆时间距离 × 日周期衰减
filled_value = Σ(w_i × v_i) / Σ(w_i)
```

### 核心特性

- **逆时间距离加权**: 越近的观测权重越大
- **日周期感知**: 同一时刻（如昨天14:00）的观测比不同时刻（如3小时前的11:00）权重更高，减少日变化（~2.3K）引起的偏差
- **48小时回看窗口**: 滑动缓存，超过48h的旧数据自动清除
- **纯时间维度**: 只做同一像素位置的时间插值，不考虑空间邻居
- **并行处理**: 9个年度序列可并行（216核）

### 输入输出

| 项目 | 说明 |
|------|------|
| 输入 | `/data/sst_data/.../jaxa_extract_L3/YYYYMM/DD/YYYYMMDDHHMMSS.nc` |
| 输出 | `jaxa_weighted_aligned/series_XX/*.h5` |
| 时间范围 | 2015-2025 (10年) |

### 使用方法

```bash
cd /path/to/project
python preprocessing/temporal_weighted_fill.py --mode full --workers 64
```

---

## Step 2: 3D时空KNN填充（因果性）

**脚本**: `preprocessing/knn_fill_3d.py`

### 算法原理

渐进密度分带3D时空KNN填充，**仅使用过去数据**（因果性，不含未来）。

把每个像素视为三维点 `(t, y, x)`，在过去120小时（5天）的时空邻域中找K个最近邻，反距离加权插值（IDW）。

```
对每一帧 t：
1. 计算每个缺失像素的2D缺失密度（半径20像素内缺失点数量）
2. 按密度排序，分成16个band（低密度=边缘优先）
3. 用 [t-120h, t] 范围内的已知像素构建3D KDTree（不含未来数据）
4. 逐band填充：
   - Band 0（边缘）：查base tree，IDW插值
   - Band 1~15（向内）：查base tree + 前序band补充tree，合并取K近邻
5. 两轮填充 + 全局均值兜底，确保海洋区域零NaN
```

时间与空间尺度统一：`TIME_SCALE = 5.0/24 ≈ 0.208`（1小时 ≈ 0.208个空间像素距离），使空间邻居比远时间点有更高优先级。

### 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| LOOKBACK_WINDOW | 120 | 回看窗口（小时），仅过去数据 |
| TIME_SCALE | 5.0/24 | 时间→空间缩放因子 |
| K | 30 | KNN邻居数 |
| POWER | 2 | IDW距离权重指数 |
| NUM_BANDS | 16 | 渐进密度分带数 |
| DENSITY_RADIUS | 20 | 缺失密度计算半径（像素） |

### 因果性设计

时间窗口只看过去，不使用未来数据，与Step 1时序加权填充（48h lookback）保持一致：
- Step 1 时序加权填充：过去48小时，纯时间维度
- Step 2 3D KNN填充：过去120小时，时空联合

这确保了训练时模型输入的构建方式与推理时一致，不存在信息泄露。

### 输入输出

| 项目 | 说明 |
|------|------|
| 输入 | `jaxa_filtered/*.h5` |
| 输出 | `jaxa_knn_filled/*.h5` |

### 使用方法

```bash
python preprocessing/knn_fill_3d.py
```

---

## H5文件格式

### 输出数据结构

```python
with h5py.File('jaxa_knn_filled_YYYYMMDD.h5', 'r') as f:
    sst_knn_filled = f['sst_knn_filled'][:]  # [H, W] float32, 单位: Kelvin
    missing_mask = f['missing_mask'][:]       # [H, W] uint8, 1=原始缺失
    lat = f['lat'][:]                         # [H] float32
    lon = f['lon'][:]                         # [W] float32
```

### 数据规格

| 属性 | 值 |
|------|-----|
| 空间分辨率 | 0.05° × 0.05° |
| 空间范围 | 15°N-24°N, 111°E-118°E (南海) |
| 网格尺寸 | 451 × 351 (lat × lon) |
| 数据类型 | float32 (SST), uint8 (mask) |
| 单位 | Kelvin |

---

## 数据统计

### 各阶段缺失率变化

| 阶段 | 平均缺失率 | 说明 |
|------|-----------|------|
| 原始数据 | 40-80% | 云层遮挡严重 |
| 时间加权填充后 | 20-40% | 利用时间信息 |
| KNN填充后 | 0% | 完全填充 |

### 数据集划分

| 数据集 | Series ID | 年份 | 样本数 |
|--------|-----------|------|--------|
| 训练集 | 0-7 | 8年 | ~2900 |
| 验证集 | 8 | 1年 | ~365 |
| 测试集 | 9 | 1年 | ~365 |

---

## 运行完整预处理

```bash
# 1. 时间加权填充 (最耗时)
python preprocessing/temporal_weighted_fill.py --mode full --workers 64

# 2. KNN填充
python preprocessing/knn_fill.py

# 检查结果
ls -lh jaxa_knn_filled/
```

**预计总时间**: 约12-24小时 (取决于CPU核心数)

---

## 为什么移除低通滤波？

原有的预处理流程包含低通滤波步骤，但存在以下问题：

1. **抹平真实细节**: 低通滤波会把真实的高频洋流细节（锋面）给抹平
2. **保留KNN伪影**: 而KNN填充的大色块反而留下了
3. **误导模型**: 导致模型认为"平滑的模糊图像"才是真理，难以生成锐利的流体细节

**解决方案**: 让FNO自己学习如何过滤噪声，通过降低FNO modes实现天然的频域低通滤波。
