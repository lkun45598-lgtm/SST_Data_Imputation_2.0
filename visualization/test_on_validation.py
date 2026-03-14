#!/usr/bin/env python3
"""
在验证集上推理测试：人工挖空20%，看FNO预测效果
和训练场景完全一致
"""

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from inference.jaxa_inference_dataset import JAXAFinetuneDataset

MODEL_PATH = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/jaxa_finetune_corrected/best_model.pth'
DATA_DIR = '/data/chla_data_imputation_data_260125/sst_post_filtered'
OUTPUT_DIR = Path('/home/lz/Data_Imputation/visualization/output')
OUTPUT_DIR.mkdir(exist_ok=True)


def load_model(device):
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6
    ).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    norm_mean = checkpoint.get('norm_mean', 299.9221)
    norm_std = checkpoint.get('norm_std', 2.6919)
    print(f"Model loaded: MAE={checkpoint.get('val_mae', 'N/A'):.4f}K")
    return model, norm_mean, norm_std


def output_composition(pred, sst_seq, mask_seq):
    """和训练一致的output composition"""
    last_input = sst_seq[:, -1:, :, :]
    last_mask = mask_seq[:, -1:, :, :]
    composed = last_input * (1 - last_mask) + pred * last_mask
    return composed


def plot_validation(gt_sst, input_sst, pred_composed, loss_mask, obs_mask, land_mask,
                    lat, lon, norm_mean, norm_std, sample_idx, output_path):
    """
    5面板：GT真值 | 挖空后输入 | FNO预测 | Loss区域 | 误差图
    """
    fig = plt.figure(figsize=(24, 5))
    gs = gridspec.GridSpec(1, 6, figure=fig, width_ratios=[1, 1, 1, 0.05, 1, 0.05], wspace=0.2)

    ocean = (land_mask == 0)

    # 反归一化
    gt_kelvin = gt_sst * norm_std + norm_mean
    input_kelvin = input_sst * norm_std + norm_mean
    pred_kelvin = pred_composed * norm_std + norm_mean

    gt_kelvin[~ocean] = np.nan
    input_kelvin[~ocean] = np.nan
    pred_kelvin[~ocean] = np.nan

    # 色标
    valid = gt_kelvin[ocean & ~np.isnan(gt_kelvin)]
    vmin, vmax = np.percentile(valid, 1), np.percentile(valid, 99)

    # Panel 1: Ground Truth
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor('#D2B48C')
    ax0.pcolormesh(lon, lat, gt_kelvin, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
    ax0.set_title('Ground Truth', fontsize=11, fontweight='bold')
    ax0.set_ylabel('Latitude (°N)', fontsize=9)

    # Panel 2: Input (挖空后)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor('#D2B48C')
    ax1.pcolormesh(lon, lat, input_kelvin, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
    ax1.set_title('Input (20% masked)', fontsize=11, fontweight='bold')

    # Panel 3: FNO Prediction (composed)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#D2B48C')
    im_sst = ax2.pcolormesh(lon, lat, pred_kelvin, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
    ax2.set_title('FNO Prediction', fontsize=11, fontweight='bold')

    # SST Colorbar
    cbar_sst_ax = fig.add_subplot(gs[0, 3])
    fig.colorbar(im_sst, cax=cbar_sst_ax, label='SST (K)')

    # Panel 4: Error (pred - gt) 只在loss_mask区域
    ax4 = fig.add_subplot(gs[0, 4])
    ax4.set_facecolor('#D2B48C')
    error = (pred_kelvin - gt_kelvin)
    error_masked = np.where(loss_mask > 0, error, np.nan)
    error_masked[~ocean] = np.nan
    valid_err = error_masked[~np.isnan(error_masked)]
    if len(valid_err) > 0:
        mae = np.mean(np.abs(valid_err))
        rmse = np.sqrt(np.mean(valid_err**2))
        dmax = max(np.percentile(np.abs(valid_err), 99), 0.1)
    else:
        mae, rmse, dmax = 0, 0, 1
    im_err = ax4.pcolormesh(lon, lat, error_masked, cmap='bwr', vmin=-dmax, vmax=dmax, shading='auto')
    ax4.set_title(f'Error (MAE={mae:.3f}K)', fontsize=11, fontweight='bold')

    # Error Colorbar
    cbar_err_ax = fig.add_subplot(gs[0, 5])
    fig.colorbar(im_err, cax=cbar_err_ax, label='Error (K)')

    for ax in [ax0, ax1, ax2, ax4]:
        ax.set_xlim(lon.min(), lon.max())
        ax.set_ylim(lat.min(), lat.max())
        ax.set_aspect('equal')
        ax.set_xlabel('Longitude (°E)', fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(f'Validation Sample #{sample_idx}  |  MAE={mae:.3f}K  RMSE={rmse:.3f}K',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}  MAE={mae:.3f}K  RMSE={rmse:.3f}K")


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, norm_mean, norm_std = load_model(device)

    # 加载验证集（和训练一致）
    valid_dataset = JAXAFinetuneDataset(
        data_dir=DATA_DIR,
        series_ids=[8],
        window_size=30,
        mask_ratio=0.2,
        min_mask_size=10,
        max_mask_size=50,
        normalize=True,
        mean=299.9221,
        std=2.6919,
        cache_size=50,
        seed=123,
        hour_offset=0
    )

    # 加载坐标和land_mask
    with h5py.File(f'{DATA_DIR}/jaxa_filtered_08.h5', 'r') as f:
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        land_mask = f['land_mask'][:]

    print(f"\nValidation samples: {len(valid_dataset)}")

    # 选几个样本推理
    test_indices = [0, 50, 100, 150, 200]
    all_mae = []

    for idx in test_indices:
        if idx >= len(valid_dataset):
            continue

        print(f"\nSample {idx}:")
        sample = valid_dataset[idx]

        sst_seq = torch.from_numpy(sample['input_sst_seq']).unsqueeze(0).float().to(device)
        mask_seq = torch.from_numpy(sample['mask_seq']).unsqueeze(0).float().to(device)
        gt_sst = sample['ground_truth_sst']
        loss_mask = sample['loss_mask']
        obs_mask = sample['original_obs_mask']

        with torch.no_grad():
            pred = model(sst_seq, mask_seq)
            composed = output_composition(pred, sst_seq, mask_seq)

        composed_np = composed.squeeze().cpu().numpy()

        # 最后一帧输入（挖空后）
        input_last = sample['input_sst_seq'][-1]

        output_path = OUTPUT_DIR / f'valid_test_sample{idx}.png'
        plot_validation(gt_sst, input_last, composed_np, loss_mask, obs_mask, land_mask,
                       lat, lon, norm_mean, norm_std, idx, output_path)

        # 计算MAE
        mask_bool = loss_mask > 0
        if mask_bool.sum() > 0:
            pred_k = composed_np * norm_std + norm_mean
            gt_k = gt_sst * norm_std + norm_mean
            mae = np.mean(np.abs(pred_k[mask_bool] - gt_k[mask_bool]))
            all_mae.append(mae)

    if all_mae:
        print(f"\n{'='*50}")
        print(f"Average MAE over {len(all_mae)} samples: {np.mean(all_mae):.3f}K")
        print(f"{'='*50}")


if __name__ == '__main__':
    main()
