# FNO-CBAM SST Missing Value Imputation 2.0

基于 Fourier Neural Operator (FNO) 和 Convolutional Block Attention Module (CBAM) 的海表温度 (SST) 缺失值重建系统。

## 相比 1.0 的主要改进

- **昼夜感知时序填充**：权重 `w = (1/dt) * exp(-α * hour_diff)` 避免昼夜温差混叠
- **三步预处理流水线**：时序填充 → 高斯低通滤波 → 3D时空KNN
- **小时级数据处理**：从日尺度升级到小时尺度，更充分利用卫星观测
- **3D时空KNN**：渐进式密度分带策略，边缘优先填充

## 目录结构

```
SST_Data_Imputation_2.0/
├── preprocessing/                  # 数据预处理 Pipeline
│   ├── temporal_weighted_fill.py   # Step1: 昼夜感知时序加权填充
│   ├── lowpass_filter.py           # Step2: 高斯低通滤波 (σ=1.5)
│   ├── knn_fill_3d.py              # Step3: 3D时空KNN填充
│   └── knn_fill.py                 # 2D KNN填充 (备用)
│
├── models/                         # 模型定义
│   └── fno_cbam_temporal.py        # FNO-CBAM时序模型
│
├── losses/                         # 损失函数
│   └── temporal_loss.py            # 组合损失 (MSE + 梯度 + 时间连续性)
│
├── datasets/                       # 数据集定义
│   ├── jaxa_dataset.py             # JAXA微调数据集
│   ├── ostia_dataset.py            # OSTIA预训练数据集
│   └── ostia_dataset_filled.py     # OSTIA预填充数据集
│
├── training/                       # 训练脚本
│   ├── train_jaxa.py               # JAXA微调 (8 GPU DDP)
│   ├── train_ostia.py              # OSTIA预训练 (8 GPU DDP)
│   └── resume_train_jaxa.py        # 断点续训
│
├── inference/                      # 推理与评估
│   ├── fill_jaxa.py                # JAXA数据填充
│   ├── evaluate.py                 # 模型评估
│   ├── jaxa_inference_dataset.py   # 推理数据集
│   └── run_inference_vis.py        # 推理+可视化
│
├── postprocessing/                 # 后处理
│   └── gaussian_filter.py          # 高斯滤波平滑
│
├── visualization/                  # 可视化工具
│   ├── plot_reconstruction.py      # 重建结果可视化
│   ├── daily_missing_rate_analysis.py  # 缺失率分析
│   └── compare_sigma.py            # 滤波参数对比
│
├── scripts/                        # 运行脚本
│   └── run_preprocessing.sh        # 预处理全流程
│
└── docs/                           # 文档
```

## 数据预处理流程

```
JAXA hourly NC files (每小时卫星观测，平均缺失率 ~58%)
        ↓
[Step 1: temporal_weighted_fill.py]
  昼夜感知时序加权填充
  w = (1/dt) * exp(-α * hour_diff), α=0.5
  48h回溯窗口，同时刻观测权重最高
  → 缺失率降至 ~13.5%
        ↓
[Step 2: lowpass_filter.py]
  高斯低通滤波 (σ=1.5)
  仅对时序填充像素滤波，真实观测不动
  → 消除昼夜温差引起的伪影
        ↓
[Step 3: knn_fill_3d.py]
  3D时空KNN填充
  渐进式密度分带 (16 bands)，边缘优先
  ±60h时间窗口，K=30
  → 缺失率降至 0%
        ↓
模型输入数据 (sst_knn_filled/*.h5)
```

## 模型架构

### FNO-CBAM-Temporal

```
输入: [B, 60, H, W]
      ├─ 30 channels: SST序列 (30天)
      └─ 30 channels: Mask序列 (30天)

架构:
    ├─ Lifting: Conv2d (60 → 64)
    ├─ FNO Blocks × 6
    │   ├─ Spectral Conv (傅里叶域卷积)
    │   ├─ Local Conv (空间域卷积)
    │   ├─ CBAM Attention
    │   └─ GELU + Residual
    └─ Projection: Conv2d (64 → 1)

输出: [B, 1, H, W] (第30天SST预测)
```

### 损失函数

```
L_total = α₁·L_mse + α₂·L_gradient + α₃·L_temporal
```

## 快速开始

### 运行预处理

```bash
# 全流程 (Step1 → Step2 → Step3)
bash scripts/run_preprocessing.sh

# 单步运行
bash scripts/run_preprocessing.sh --step 2        # 仅低通滤波
bash scripts/run_preprocessing.sh --step 2 3      # 滤波+KNN
bash scripts/run_preprocessing.sh --series 0 1     # 指定系列
```

### 训练

```bash
# JAXA微调 (8 GPU DDP)
bash scripts/run_training.sh
```

## 依赖环境

```
torch >= 1.10
numpy
scipy
h5py
netCDF4
matplotlib
tqdm
```
