#!/usr/bin/env python3
"""
FNO-CBAM SST 标准指标评估

指标:
  - RMSE  (Root Mean Square Error, °C)
  - MAE   (Mean Absolute Error, °C)
  - Bias  (Signed Mean Error, °C, 正=模型偏暖 负=模型偏冷)
  - R     (Pearson Correlation Coefficient)

评估方式: 人工挖空 → 模型预测 → 与KNN真值对比 (仅在挖空∩观测∩海洋区域)
输出: 每样本逐行表 + 汇总统计 + CSV文件
"""

import os
import sys
import torch
import numpy as np
import h5py
import csv
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from training.train_jaxa import output_composition

# ============================================================================
# Configuration
# ============================================================================
NPY_DIR = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/sst_knn_npy_cache'
KNN_H5_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled')
MODEL_PATH = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/run004_jaxa_3dknn_progressive_stride1_lr0.0005/best_model.pth'
OUTPUT_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/evaluation_results')

SERIES_ID = 8          # 验证集
WINDOW_SIZE = 30
STRIDE = 24
NORM_MEAN = 299.9221
NORM_STD = 2.6919
GPU_ID = 7
GAUSSIAN_SIGMA = 1.0
NUM_SAMPLES = 50       # 更多样本以获得稳定统计
MASK_RATIO = 0.2
SEED = 42


# ============================================================================
# Data Loading
# ============================================================================
def load_npy_data(npy_dir, series_id):
    sst = np.load(f'{npy_dir}/sst_{series_id:02d}.npy', mmap_mode='r')
    obs = np.load(f'{npy_dir}/obs_{series_id:02d}.npy', mmap_mode='r')
    miss = np.load(f'{npy_dir}/miss_{series_id:02d}.npy', mmap_mode='r')
    land = np.load(f'{npy_dir}/land_{series_id:02d}.npy', mmap_mode='r')
    return sst, obs, miss, land


def load_h5_metadata(h5_dir, series_id):
    h5_path = h5_dir / f'jaxa_knn_filled_{series_id:02d}.h5'
    with h5py.File(h5_path, 'r') as f:
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        timestamps = [ts.decode('utf-8') if isinstance(ts, bytes) else ts
                      for ts in f['timestamps'][:]]
        land_mask = f['land_mask'][:]
    return lat, lon, timestamps, land_mask


# ============================================================================
# Mask & Filter
# ============================================================================
def generate_block_mask(obs_mask, land_mask, ratio=0.2, min_size=10, max_size=50, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    valid = (obs_mask * (1 - land_mask)).astype(bool)
    target = int(valid.sum() * ratio)
    if target < 10:
        return np.zeros_like(obs_mask, dtype=np.float32)
    H, W = obs_mask.shape
    mask = np.zeros((H, W), dtype=np.float32)
    count = 0
    for _ in range(500):
        if count >= target:
            break
        bh = rng.randint(min_size, max_size + 1)
        bw = rng.randint(min_size, max_size + 1)
        y, x = rng.randint(0, H - bh + 1), rng.randint(0, W - bw + 1)
        if valid[y:y+bh, x:x+bw].sum() > 0:
            mask[y:y+bh, x:x+bw] = 1.0
            count = int((mask * valid).sum())
    return mask


def apply_gaussian_filter_ocean(sst, land_mask, sigma=1.0, missing_mask=None):
    result = sst.copy()
    valid = ~np.isnan(result) & (land_mask == 0)
    if valid.sum() == 0:
        return result
    mean_val = np.nanmean(result[valid])
    tmp = result.copy()
    tmp[~valid] = mean_val
    filtered = gaussian_filter(tmp, sigma=sigma)
    if missing_mask is not None:
        result = np.where((missing_mask > 0) & (land_mask == 0), filtered, result)
    else:
        result = np.where(land_mask == 0, filtered, np.nan)
    return result


# ============================================================================
# Inference
# ============================================================================
def run_inference(model, sst_seq, mask_seq, device):
    sst_norm = (sst_seq - NORM_MEAN) / NORM_STD
    sst_norm = np.nan_to_num(sst_norm, nan=0.0)
    sst_t = torch.from_numpy(sst_norm).unsqueeze(0).float().to(device)
    mask_t = torch.from_numpy(mask_seq.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(sst_t, mask_t)
    pred_composed = output_composition(pred, sst_t, mask_t)
    return pred_composed.squeeze().cpu().numpy() * NORM_STD + NORM_MEAN


# ============================================================================
# Metrics (all in °C, on masked region)
# ============================================================================
def compute_metrics(pred_kelvin, gt_kelvin, mask):
    """
    Args:
        pred_kelvin: (H, W) model prediction in Kelvin
        gt_kelvin:   (H, W) ground truth in Kelvin
        mask:        (H, W) 1=evaluate, 0=ignore

    Returns:
        dict with RMSE, MAE, Bias, R, num_pixels
    """
    valid = mask > 0
    n = int(valid.sum())
    if n < 2:
        return {'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 'r': np.nan, 'n': n}

    pred_c = pred_kelvin[valid] - 273.15
    gt_c = gt_kelvin[valid] - 273.15

    diff = pred_c - gt_c

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))  # positive = model warmer than GT
    r, _ = pearsonr(pred_c, gt_c)
    r = float(r)

    return {'rmse': rmse, 'mae': mae, 'bias': bias, 'r': r, 'n': n}


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 80)
    print("FNO-CBAM SST Evaluation — RMSE / MAE / Bias / R")
    print("=" * 80)

    device = torch.device(f'cuda:{GPU_ID}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nModel: {MODEL_PATH}")
    model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64,
                                   width=64, depth=6).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    epoch = ckpt['epoch'] + 1
    print(f"  Epoch {epoch}, val_mae={ckpt.get('val_mae', 'N/A')}")

    # Load data
    print(f"\nSeries {SERIES_ID} data...")
    sst, obs, miss, land = load_npy_data(NPY_DIR, SERIES_ID)
    lat, lon, timestamps, land_mask = load_h5_metadata(KNN_H5_DIR, SERIES_ID)
    n_days = len(timestamps)
    print(f"  {sst.shape[0]} hourly frames, {n_days} days")
    print(f"  {timestamps[0][:10]} -> {timestamps[-1][:10]}")

    # Sample indices
    day_indices = np.linspace(WINDOW_SIZE, n_days - 1, NUM_SAMPLES, dtype=int)

    # Run evaluation
    print(f"\nEvaluating {NUM_SAMPLES} samples (mask_ratio={MASK_RATIO}, sigma={GAUSSIAN_SIGMA})...\n")

    header = f"{'#':>3}  {'Date':>12}  {'RMSE':>8}  {'MAE':>8}  {'Bias':>8}  {'R':>8}  {'Pixels':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    rng = np.random.RandomState(SEED)
    all_metrics = []

    # Also collect all pred/gt pairs for global metrics
    all_pred_c = []
    all_gt_c = []

    for i, day_idx in enumerate(day_indices):
        date_str = timestamps[day_idx][:10]
        hourly_idx = day_idx * STRIDE

        # Build 30-day window
        frame_indices = [max(0, hourly_idx - t * STRIDE)
                         for t in range(WINDOW_SIZE - 1, -1, -1)]
        sst_seq = sst[frame_indices].copy()
        mask_seq = miss[frame_indices].copy()

        gt_sst = sst[hourly_idx].copy()
        obs_mask = obs[hourly_idx].copy()

        # Artificial masking
        artificial_mask = generate_block_mask(obs_mask, land_mask, MASK_RATIO, rng=rng)
        loss_mask = (artificial_mask * obs_mask * (1 - land_mask)).astype(np.float32)

        # Prepare input
        sst_seq_input = sst_seq.copy()
        sst_seq_input[-1] = np.where(artificial_mask == 1, NORM_MEAN, sst_seq[-1])
        mask_seq_input = mask_seq.copy()
        mask_seq_input[-1] = artificial_mask

        # Inference + output composition + gaussian
        pred_sst = run_inference(model, sst_seq_input, mask_seq_input, device)
        fno_composed = np.where(artificial_mask > 0, pred_sst, gt_sst)
        fno_gaussian = apply_gaussian_filter_ocean(fno_composed, land_mask, sigma=GAUSSIAN_SIGMA, missing_mask=artificial_mask)

        # Compute metrics
        m = compute_metrics(fno_gaussian, gt_sst, loss_mask)
        m['date'] = date_str
        all_metrics.append(m)

        # Collect for global metrics
        valid = loss_mask > 0
        if valid.sum() > 1:
            all_pred_c.append(fno_gaussian[valid] - 273.15)
            all_gt_c.append(gt_sst[valid] - 273.15)

        print(f"{i+1:3d}  {date_str:>12}  {m['rmse']:8.4f}  {m['mae']:8.4f}  "
              f"{m['bias']:+8.4f}  {m['r']:8.4f}  {m['n']:8d}")

    # ========================================================================
    # Summary statistics
    # ========================================================================
    print(sep)

    valid_metrics = [m for m in all_metrics if not np.isnan(m['rmse'])]
    n_valid = len(valid_metrics)

    rmses = [m['rmse'] for m in valid_metrics]
    maes = [m['mae'] for m in valid_metrics]
    biases = [m['bias'] for m in valid_metrics]
    rs = [m['r'] for m in valid_metrics]

    # Per-sample averages
    avg_rmse = np.mean(rmses)
    avg_mae = np.mean(maes)
    avg_bias = np.mean(biases)
    avg_r = np.mean(rs)

    print(f"{'AVG':>3}  {'(per-sample)':>12}  {avg_rmse:8.4f}  {avg_mae:8.4f}  "
          f"{avg_bias:+8.4f}  {avg_r:8.4f}  {'':>8}")

    # Global metrics (pooled all pixels)
    all_pred_c = np.concatenate(all_pred_c)
    all_gt_c = np.concatenate(all_gt_c)
    global_diff = all_pred_c - all_gt_c
    global_rmse = float(np.sqrt(np.mean(global_diff ** 2)))
    global_mae = float(np.mean(np.abs(global_diff)))
    global_bias = float(np.mean(global_diff))
    global_r, _ = pearsonr(all_pred_c, all_gt_c)
    global_r = float(global_r)
    total_pixels = len(all_pred_c)

    print(f"{'ALL':>3}  {'(pooled)':>12}  {global_rmse:8.4f}  {global_mae:8.4f}  "
          f"{global_bias:+8.4f}  {global_r:8.4f}  {total_pixels:8d}")
    print(sep)

    # ========================================================================
    # Print clean summary table
    # ========================================================================
    print(f"\n{'=' * 60}")
    print(f"  FNO-CBAM SST Reconstruction — Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"  Model:     {Path(MODEL_PATH).parent.name}")
    print(f"  Data:      Series {SERIES_ID} ({timestamps[0][:10]} ~ {timestamps[-1][:10]})")
    print(f"  Samples:   {n_valid}")
    print(f"  Total px:  {total_pixels:,}")
    print(f"{'=' * 60}")
    print(f"")
    print(f"  {'Metric':<12} {'Per-sample Avg':>16} {'Global (pooled)':>16}")
    print(f"  {'-'*12} {'-'*16} {'-'*16}")
    print(f"  {'RMSE (°C)':<12} {avg_rmse:>16.4f} {global_rmse:>16.4f}")
    print(f"  {'MAE (°C)':<12} {avg_mae:>16.4f} {global_mae:>16.4f}")
    print(f"  {'Bias (°C)':<12} {avg_bias:>+16.4f} {global_bias:>+16.4f}")
    print(f"  {'R':<12} {avg_r:>16.4f} {global_r:>16.4f}")
    print(f"")
    print(f"  Bias > 0 => model warmer than GT")
    print(f"  Bias < 0 => model cooler than GT")
    print(f"{'=' * 60}")

    # ========================================================================
    # Save CSV
    # ========================================================================
    csv_path = OUTPUT_DIR / 'evaluation_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'rmse_c', 'mae_c', 'bias_c', 'pearson_r', 'num_pixels'])
        for m in all_metrics:
            writer.writerow([m['date'], f"{m['rmse']:.6f}", f"{m['mae']:.6f}",
                             f"{m['bias']:.6f}", f"{m['r']:.6f}", m['n']])
        # Summary rows
        writer.writerow([])
        writer.writerow(['SUMMARY', 'RMSE', 'MAE', 'Bias', 'R', 'N'])
        writer.writerow(['per_sample_avg', f"{avg_rmse:.6f}", f"{avg_mae:.6f}",
                         f"{avg_bias:.6f}", f"{avg_r:.6f}", n_valid])
        writer.writerow(['global_pooled', f"{global_rmse:.6f}", f"{global_mae:.6f}",
                         f"{global_bias:.6f}", f"{global_r:.6f}", total_pixels])

    print(f"\nCSV saved: {csv_path}")
    print("Done.")


if __name__ == '__main__':
    main()
