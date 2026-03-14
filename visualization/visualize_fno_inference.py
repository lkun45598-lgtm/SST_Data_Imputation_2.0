#!/usr/bin/env python3
"""
用最优模型推理并可视化：JAXA原始 → 预处理填充 → FNO预测 → 对比
"""

import numpy as np
import torch
import h5py
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

# ============================================================================
# 配置
# ============================================================================
MODEL_PATH = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/jaxa_finetune_corrected/best_model.pth'
DATA_DIR = Path('/data/chla_data_imputation_data_260125/sst_post_filtered')
JAXA_RAW_DIR = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')
OUTPUT_DIR = Path('/home/lz/Data_Imputation/visualization/output')
OUTPUT_DIR.mkdir(exist_ok=True)

# 选择观测率较高的时间点
TARGET_FRAMES = [72, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500]


def load_model(device):
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    norm_mean = checkpoint.get('norm_mean', 299.9221)
    norm_std = checkpoint.get('norm_std', 2.6919)

    print(f"Model loaded: MAE={checkpoint.get('val_mae', 'N/A')}")
    print(f"  norm_mean={norm_mean:.4f}, norm_std={norm_std:.4f}")

    return model, norm_mean, norm_std


def load_jaxa_raw(timestamp_str):
    """加载JAXA原始数据"""
    dt = datetime.strptime(timestamp_str[:19], '%Y-%m-%dT%H:%M:%S')
    raw_file = JAXA_RAW_DIR / dt.strftime('%Y%m') / dt.strftime('%d') / dt.strftime('%Y%m%d%H%M%S.nc')

    if not raw_file.exists():
        return None

    with nc.Dataset(raw_file) as f:
        sst = f.variables['sea_surface_temperature'][:].squeeze()
        sst = np.ma.filled(sst, np.nan)
    return sst


def run_inference(model, sst_data, obs_mask, land_mask, target_idx, norm_mean, norm_std, device):
    """对单帧进行推理"""
    # 组装30天序列（每24帧取1帧）
    # 注意：sst_data已经是KNN+后处理填充后的完整数据
    # 训练时：前29天SST是完整的，只有第30天被人工挖空填均值
    # 推理时：前29天SST直接用填充值，第30天缺失区域填均值
    sst_seq = []
    mask_seq = []

    for day in range(29, -1, -1):
        frame_idx = target_idx - day * 24
        if frame_idx < 0:
            frame_idx = 0

        sst_frame = sst_data[frame_idx].copy()
        obs = obs_mask[frame_idx].copy()
        missing = 1.0 - obs

        if day == 0:
            # 第30天（目标帧）：缺失区域填均值（和训练时人工挖空一致）
            sst_filled = np.where(missing > 0, norm_mean, sst_frame)
        else:
            # 前29天：直接用KNN填充后的完整数据（和训练一致）
            sst_filled = sst_frame.copy()

        sst_filled = np.nan_to_num(sst_filled, nan=norm_mean)
        sst_seq.append(sst_filled)
        mask_seq.append(missing)

    sst_seq = np.stack(sst_seq)  # [30, H, W]
    mask_seq = np.stack(mask_seq)  # [30, H, W]

    # 归一化
    sst_norm = (sst_seq - norm_mean) / norm_std

    # 转tensor
    sst_tensor = torch.from_numpy(sst_norm).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask_seq.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(sst_tensor, mask_tensor)

        # Output Composition（和训练一致）
        # mask=1的区域用预测，mask=0的区域用输入
        last_input = sst_tensor[:, -1:, :, :]  # [1, 1, H, W]
        last_mask = mask_tensor[:, -1:, :, :]   # [1, 1, H, W]
        composed = last_input * (1 - last_mask) + pred * last_mask

    # 反归一化 - 纯模型预测
    pred_kelvin = pred.squeeze().cpu().numpy() * norm_std + norm_mean
    # 反归一化 - composition后的结果
    composed_kelvin = composed.squeeze().cpu().numpy() * norm_std + norm_mean

    # 最终组合：观测区域用原值，缺失区域用composition结果
    target_sst = sst_data[target_idx].copy()
    target_obs = obs_mask[target_idx].copy()
    ocean = (land_mask == 0)

    result = target_sst.copy()
    fill_positions = (target_obs == 0) & ocean
    result[fill_positions] = composed_kelvin[fill_positions]
    result[~ocean] = np.nan

    return composed_kelvin, result


def plot_comparison(jaxa_raw, preprocessed, fno_composed, obs_mask_frame, land_mask, lat, lon, timestamp, output_path):
    """5面板对比可视化: JAXA原始 | 预处理填充 | FNO重建 | 观测掩码 | FNO-预处理差异"""
    fig = plt.figure(figsize=(28, 5))
    gs = gridspec.GridSpec(1, 7, figure=fig, width_ratios=[1, 1, 1, 0.05, 1, 1, 0.05], wspace=0.15)

    ocean = (land_mask == 0)

    # 统一色标（SST）
    all_data = []
    for d in [jaxa_raw, preprocessed, fno_composed]:
        if d is not None:
            valid = d[ocean & ~np.isnan(d)]
            if len(valid) > 0:
                all_data.append(valid)

    if all_data:
        all_valid = np.concatenate(all_data)
        vmin = np.percentile(all_valid, 1)
        vmax = np.percentile(all_valid, 99)
    else:
        vmin, vmax = 295, 305

    # Panel 1: JAXA Raw
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor('#D2B48C')
    if jaxa_raw is not None:
        d = jaxa_raw.copy(); d[~ocean] = np.nan
        ax0.pcolormesh(lon, lat, d, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
    ax0.set_title('JAXA Raw', fontsize=11, fontweight='bold')
    ax0.set_ylabel('Latitude (°N)', fontsize=9)
    ax0.set_aspect('equal'); ax0.tick_params(labelsize=8)

    # Panel 2: Preprocessed
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor('#D2B48C')
    d = preprocessed.copy(); d[~ocean] = np.nan
    ax1.pcolormesh(lon, lat, d, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
    ax1.set_title('Preprocessed (KNN+Filter)', fontsize=11, fontweight='bold')
    ax1.set_aspect('equal'); ax1.tick_params(labelsize=8)

    # Panel 3: FNO Composed
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#D2B48C')
    d = fno_composed.copy(); d[~ocean] = np.nan
    im_sst = ax2.pcolormesh(lon, lat, d, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
    ax2.set_title('FNO Reconstruction', fontsize=11, fontweight='bold')
    ax2.set_aspect('equal'); ax2.tick_params(labelsize=8)

    # SST Colorbar (for panels 1-3)
    cbar_sst_ax = fig.add_subplot(gs[0, 3])
    fig.colorbar(im_sst, cax=cbar_sst_ax, label='SST (K)')

    # Panel 4: Observation Mask
    ax3 = fig.add_subplot(gs[0, 4])
    ax3.set_facecolor('#D2B48C')
    mask_display = np.full_like(land_mask, np.nan, dtype=float)
    mask_display[ocean] = obs_mask_frame[ocean]
    ax3.pcolormesh(lon, lat, mask_display, cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    ax3.set_title('Obs Mask (green=obs)', fontsize=11, fontweight='bold')
    ax3.set_aspect('equal'); ax3.tick_params(labelsize=8)

    # Panel 5: Difference (FNO - Preprocessed)
    ax4 = fig.add_subplot(gs[0, 5])
    ax4.set_facecolor('#D2B48C')
    diff = fno_composed - preprocessed
    diff[~ocean] = np.nan
    diff_abs = np.abs(diff[ocean & ~np.isnan(diff)])
    if len(diff_abs) > 0:
        dmax = np.percentile(diff_abs, 99)
    else:
        dmax = 1.0
    im_err = ax4.pcolormesh(lon, lat, diff, cmap='bwr', vmin=-dmax, vmax=dmax, shading='auto')
    ax4.set_title(f'FNO - Preproc (max±{dmax:.2f}K)', fontsize=11, fontweight='bold')
    ax4.set_aspect('equal'); ax4.tick_params(labelsize=8)

    # Error Colorbar (for panel 5)
    cbar_err_ax = fig.add_subplot(gs[0, 6])
    fig.colorbar(im_err, cax=cbar_err_ax, label='Error (K)')

    for ax in [ax0, ax1, ax2, ax3, ax4]:
        ax.set_xlim(lon.min(), lon.max())
        ax.set_ylim(lat.min(), lat.max())
        ax.set_xlabel('Longitude (°E)', fontsize=9)

    obs_rate = obs_mask_frame[ocean].mean() * 100
    fig.suptitle(f'SST Reconstruction: {timestamp}  |  Obs rate: {obs_rate:.1f}%', fontsize=13, fontweight='bold', y=1.02)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载模型
    model, norm_mean, norm_std = load_model(device)

    # 加载数据（第一个序列）
    h5_path = DATA_DIR / 'jaxa_filtered_00.h5'
    print(f"\nLoading data: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        sst_data = f['sst_data'][:]
        obs_mask = f['original_obs_mask'][:]
        land_mask = f['land_mask'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        timestamps = [t.decode() if isinstance(t, bytes) else t for t in f['timestamps'][:]]

    print(f"Data shape: {sst_data.shape}, frames: {len(timestamps)}")

    # 对每个目标帧进行推理和可视化
    for target_idx in TARGET_FRAMES:
        if target_idx >= len(timestamps):
            print(f"Skip frame {target_idx}: out of range")
            continue

        ts = timestamps[target_idx]
        print(f"\nProcessing frame {target_idx}: {ts}")

        # 加载JAXA原始数据
        jaxa_raw = load_jaxa_raw(ts)
        if jaxa_raw is not None:
            valid_rate = np.sum(~np.isnan(jaxa_raw)) / jaxa_raw.size * 100
            print(f"  JAXA raw valid rate: {valid_rate:.1f}%")

        # 预处理后的数据
        preprocessed = sst_data[target_idx].copy()
        preprocessed[land_mask > 0] = np.nan

        # FNO推理
        fno_pred, fno_composed = run_inference(
            model, sst_data, obs_mask, land_mask, target_idx,
            norm_mean, norm_std, device
        )
        fno_pred[land_mask > 0] = np.nan
        fno_composed[land_mask > 0] = np.nan

        # 可视化
        output_path = OUTPUT_DIR / f'fno_inference_frame{target_idx}.png'
        plot_comparison(jaxa_raw, preprocessed, fno_composed,
                       obs_mask[target_idx],
                       land_mask, lat, lon, ts, output_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
