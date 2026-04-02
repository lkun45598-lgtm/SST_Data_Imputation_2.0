#!/usr/bin/env python3
"""
FNO-CBAM SST 全量评估 — Series 0-8 全部天数

指标: RMSE / MAE / Bias / R
输出: 逐series汇总 + 全局汇总 + CSV
"""

import os
import sys
import torch
import numpy as np
import h5py
import csv
import time
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

SERIES_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
WINDOW_SIZE = 30
STRIDE = 24
NORM_MEAN = 299.9221
NORM_STD = 2.6919
GPU_ID = 3
GAUSSIAN_SIGMA = 1.0
MASK_RATIO = 0.2
SEED = 42
BATCH_SIZE = 8


# ============================================================================
# Helpers
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


def run_inference_batch(model, sst_list, mask_list, device):
    """批量推理，输入为 list of (30,H,W) arrays，返回 list of (H,W) arrays"""
    B = len(sst_list)
    sst_batch = np.stack([(s - NORM_MEAN) / NORM_STD for s in sst_list])
    sst_batch = np.nan_to_num(sst_batch, nan=0.0)
    mask_batch = np.stack([m.astype(np.float32) for m in mask_list])
    sst_t = torch.from_numpy(sst_batch).float().to(device)
    mask_t = torch.from_numpy(mask_batch).to(device)
    with torch.no_grad():
        pred = model(sst_t, mask_t)
    pred_composed = output_composition(pred, sst_t, mask_t)
    results = pred_composed.cpu().numpy() * NORM_STD + NORM_MEAN
    return [results[i, 0] for i in range(B)]


def compute_metrics(pred_k, gt_k, mask):
    valid = mask > 0
    n = int(valid.sum())
    if n < 2:
        return None
    pred_c = pred_k[valid] - 273.15
    gt_c = gt_k[valid] - 273.15
    diff = pred_c - gt_c
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    r, _ = pearsonr(pred_c, gt_c)
    return {'rmse': rmse, 'mae': mae, 'bias': bias, 'r': float(r), 'n': n,
            'pred_c': pred_c, 'gt_c': gt_c}


# ============================================================================
# Main
# ============================================================================
def main():
    t_start = time.time()
    print("=" * 80)
    print("FNO-CBAM Full Evaluation — Series 0-8, All Days")
    print("=" * 80)

    device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else f'cuda:{GPU_ID}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nModel: {Path(MODEL_PATH).parent.name}")
    model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64,
                                   width=64, depth=6).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Epoch {ckpt['epoch']+1}, val_mae={ckpt.get('val_mae','N/A')}")

    # Per-series and global accumulators
    all_pred_global = []
    all_gt_global = []
    series_summaries = []
    all_rows = []  # for CSV

    for sid in SERIES_IDS:
        print(f"\n{'='*80}")
        print(f"Series {sid}")
        print(f"{'='*80}")

        sst, obs, miss, land = load_npy_data(NPY_DIR, sid)
        _, _, timestamps, land_mask = load_h5_metadata(KNN_H5_DIR, sid)
        n_days = len(timestamps)
        print(f"  {n_days} days, {timestamps[0][:10]} ~ {timestamps[-1][:10]}")

        rng = np.random.RandomState(SEED + sid)
        series_pred = []
        series_gt = []
        series_count = 0
        t_series = time.time()

        # 收集待推理的批次
        batch_sst, batch_mask = [], []
        batch_meta = []  # (day_idx, date_str, gt_sst, artificial_mask, loss_mask)

        def flush_batch():
            """批量推理并计算指标"""
            nonlocal series_count
            if not batch_sst:
                return
            pred_list = run_inference_batch(model, batch_sst, batch_mask, device)
            for pred_sst, (di, ds, gt, amask, lmask) in zip(pred_list, batch_meta):
                fno_composed = np.where(amask > 0, pred_sst, gt)
                fno_gaussian = apply_gaussian_filter_ocean(fno_composed, land_mask, sigma=GAUSSIAN_SIGMA, missing_mask=amask)
                m = compute_metrics(fno_gaussian, gt, lmask)
                if m is None:
                    continue
                series_pred.append(m['pred_c'])
                series_gt.append(m['gt_c'])
                series_count += 1
                all_rows.append([sid, ds, f"{m['rmse']:.6f}", f"{m['mae']:.6f}",
                                 f"{m['bias']:.6f}", f"{m['r']:.6f}", m['n']])
                if series_count % 50 == 0:
                    elapsed = time.time() - t_series
                    print(f"  [{series_count}/{n_days-WINDOW_SIZE}] {ds} "
                          f"({elapsed:.0f}s elapsed)", flush=True)
            batch_sst.clear()
            batch_mask.clear()
            batch_meta.clear()

        for day_idx in range(WINDOW_SIZE, n_days):
            date_str = timestamps[day_idx][:10]
            hourly_idx = day_idx * STRIDE

            if hourly_idx >= sst.shape[0]:
                continue

            # Build 30-day window
            frame_indices = [max(0, hourly_idx - t * STRIDE)
                             for t in range(WINDOW_SIZE - 1, -1, -1)]
            sst_seq = sst[frame_indices].copy()
            mask_seq = miss[frame_indices].copy()

            gt_sst = sst[hourly_idx].copy()
            obs_mask = obs[hourly_idx].copy()

            # Artificial mask
            artificial_mask = generate_block_mask(obs_mask, land_mask, MASK_RATIO, rng=rng)
            loss_mask = (artificial_mask * obs_mask * (1 - land_mask)).astype(np.float32)

            if loss_mask.sum() < 2:
                continue

            # Input prep
            sst_seq_input = sst_seq.copy()
            sst_seq_input[-1] = np.where(artificial_mask == 1, NORM_MEAN, sst_seq[-1])
            mask_seq_input = mask_seq.copy()
            mask_seq_input[-1] = artificial_mask

            batch_sst.append(sst_seq_input)
            batch_mask.append(mask_seq_input)
            batch_meta.append((day_idx, date_str, gt_sst, artificial_mask, loss_mask))

            if len(batch_sst) >= BATCH_SIZE:
                flush_batch()

        # 处理最后不满一个 batch 的
        flush_batch()

        # Series summary
        if series_pred:
            sp = np.concatenate(series_pred)
            sg = np.concatenate(series_gt)
            diff = sp - sg
            s_rmse = float(np.sqrt(np.mean(diff**2)))
            s_mae = float(np.mean(np.abs(diff)))
            s_bias = float(np.mean(diff))
            s_r, _ = pearsonr(sp, sg)
            s_r = float(s_r)
            s_n = len(sp)

            all_pred_global.append(sp)
            all_gt_global.append(sg)

            series_summaries.append({
                'series': sid, 'days': series_count, 'pixels': s_n,
                'rmse': s_rmse, 'mae': s_mae, 'bias': s_bias, 'r': s_r
            })

            elapsed = time.time() - t_series
            print(f"\n  Series {sid} done: {series_count} days, {s_n:,} pixels, {elapsed:.0f}s")
            print(f"    RMSE={s_rmse:.4f}  MAE={s_mae:.4f}  Bias={s_bias:+.4f}  R={s_r:.4f}")

    # ========================================================================
    # Global summary
    # ========================================================================
    all_pred = np.concatenate(all_pred_global)
    all_gt = np.concatenate(all_gt_global)
    diff = all_pred - all_gt
    g_rmse = float(np.sqrt(np.mean(diff**2)))
    g_mae = float(np.mean(np.abs(diff)))
    g_bias = float(np.mean(diff))
    g_r, _ = pearsonr(all_pred, all_gt)
    g_r = float(g_r)
    total_px = len(all_pred)
    total_days = sum(s['days'] for s in series_summaries)
    total_time = time.time() - t_start

    print(f"\n\n{'='*80}")
    print(f"  FNO-CBAM Full Evaluation Summary")
    print(f"{'='*80}")
    print(f"  Model:      {Path(MODEL_PATH).parent.name}")
    print(f"  Series:     {SERIES_IDS}")
    print(f"  Total days: {total_days}")
    print(f"  Total px:   {total_px:,}")
    print(f"  Time:       {total_time/60:.1f} min")
    print(f"{'='*80}\n")

    # Per-series table
    hdr = f"  {'Series':>6}  {'Days':>5}  {'Pixels':>10}  {'RMSE':>8}  {'MAE':>8}  {'Bias':>8}  {'R':>8}"
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    for s in series_summaries:
        tag = "val" if s['series'] == 8 else "train"
        print(f"  {s['series']:>4} ({tag})  {s['days']:>5}  {s['pixels']:>10,}  "
              f"{s['rmse']:8.4f}  {s['mae']:8.4f}  {s['bias']:+8.4f}  {s['r']:8.4f}")
    print(sep)
    print(f"  {'GLOBAL':>8}  {total_days:>5}  {total_px:>10,}  "
          f"{g_rmse:8.4f}  {g_mae:8.4f}  {g_bias:+8.4f}  {g_r:8.4f}")
    print(sep)

    # Clean summary
    print(f"\n  {'Metric':<12} {'Global':>12}")
    print(f"  {'-'*12} {'-'*12}")
    print(f"  {'RMSE (°C)':<12} {g_rmse:>12.4f}")
    print(f"  {'MAE (°C)':<12} {g_mae:>12.4f}")
    print(f"  {'Bias (°C)':<12} {g_bias:>+12.4f}")
    print(f"  {'R':<12} {g_r:>12.4f}")
    print(f"\n  Bias > 0 => model warmer | Bias < 0 => model cooler")
    print(f"{'='*80}")

    # Save CSV
    csv_path = OUTPUT_DIR / 'evaluation_full_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['series', 'date', 'rmse_c', 'mae_c', 'bias_c', 'pearson_r', 'num_pixels'])
        for row in all_rows:
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(['SERIES_SUMMARY', '', 'RMSE', 'MAE', 'Bias', 'R', 'Pixels'])
        for s in series_summaries:
            writer.writerow([f"series_{s['series']}", f"{s['days']}days",
                             f"{s['rmse']:.6f}", f"{s['mae']:.6f}",
                             f"{s['bias']:.6f}", f"{s['r']:.6f}", s['pixels']])
        writer.writerow([])
        writer.writerow(['GLOBAL', f"{total_days}days",
                         f"{g_rmse:.6f}", f"{g_mae:.6f}",
                         f"{g_bias:.6f}", f"{g_r:.6f}", total_px])

    print(f"\nCSV saved: {csv_path}")
    print("Done.")


if __name__ == '__main__':
    main()
