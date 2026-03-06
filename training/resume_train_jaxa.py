#!/usr/bin/env python3
"""
JAXA Fine-tuning 恢复训练脚本

从 run004_jaxa_3dknn_progressive_stride1_lr0.0005 的 best_model.pth (epoch 26) 恢复训练。
所有超参数与原训练完全一致:
  - lr=5e-4, CosineAnnealingLR(T_max=300, eta_min=1e-6)
  - weight_decay=5e-4
  - batch_size=2/GPU, world_size=2
  - sample_stride=1, stride=24
  - alpha_mse=1.0, alpha_grad=0.02, alpha_temporal=0.1
  - early_stop_patience=40

日期: 2026-02-28
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from inference.jaxa_inference_dataset import JAXAFinetuneDataset, preload_shared_data
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

# 复用 train_jaxa.py 中的 loss 函数
from training.train_jaxa import (
    masked_mse_loss, masked_gradient_loss, output_composition,
    jaxa_combined_loss, train_epoch, valid_epoch
)


def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29510'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


def resume_worker(rank, world_size, config, shared_data):
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    # 配置
    save_dir = config['save_dir']
    resume_path = config['resume_checkpoint']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    lr = config['lr']
    start_epoch = config['start_epoch']

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("JAXA Fine-tuning 恢复训练 (DDP, Shared Memory)")
        print("=" * 80)
        print(f"\n配置:")
        print(f"  - 恢复自: {resume_path}")
        print(f"  - 保存目录: {save_dir}")
        print(f"  - Batch size: {batch_size} per GPU x {world_size} GPUs = {batch_size * world_size}")
        print(f"  - Learning rate: {lr} (CosineAnnealing, T_max={num_epochs})")
        print(f"  - 恢复epoch: {start_epoch} (从 epoch {start_epoch + 1}/{num_epochs} 开始)")
        print(f"  - Weight decay: {config['weight_decay']}")
        print(f"  - Loss: alpha_mse={config['alpha_mse']}, alpha_grad={config['alpha_grad']}, alpha_temporal={config['alpha_temporal']}")
        print(f"  - Early stop patience: {config['early_stop_patience']}")
        print(f"  - Sample stride: {config['sample_stride']}")
        print()

    # ---- 数据集 ----
    ostia_mean = 299.9221
    ostia_std = 2.6919

    train_dataset = JAXAFinetuneDataset(
        shared_data=shared_data,
        series_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        window_size=30,
        stride=24,
        sample_stride=config['sample_stride'],
        mask_ratio=0.2,
        min_mask_size=10,
        max_mask_size=50,
        normalize=True,
        mean=ostia_mean,
        std=ostia_std,
        seed=42 + rank
    )

    valid_dataset = JAXAFinetuneDataset(
        shared_data=shared_data,
        series_ids=[8],
        window_size=30,
        stride=24,
        sample_stride=config['sample_stride'],
        mask_ratio=0.2,
        min_mask_size=10,
        max_mask_size=50,
        normalize=True,
        mean=ostia_mean,
        std=ostia_std,
        seed=123
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=1, pin_memory=True, prefetch_factor=2, persistent_workers=True
    )

    if rank == 0:
        print(f"数据:")
        print(f"  - Train: {len(train_dataset)} samples (序列 0-7, 8年)")
        print(f"  - Valid: {len(valid_dataset)} samples (序列 8, 1年)")

    # ---- 模型 ----
    model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6).to(device)

    # ---- 加载 checkpoint ----
    if rank == 0:
        print(f"\n加载checkpoint: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if rank == 0:
        print(f"  模型权重加载成功 (来自 epoch {checkpoint['epoch'] + 1})")
        if 'val_mae' in checkpoint:
            print(f"  Checkpoint val_mae: {checkpoint['val_mae']:.4f}K")

    model = DDP(model, device_ids=[rank])

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数量: {total_params:,}")

    # ---- 优化器 ----
    weight_decay = config['weight_decay']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 恢复优化器状态 (包含 Adam 的动量/方差缓存)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if rank == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  优化器状态恢复成功 (当前LR: {current_lr:.10f})")

    # ---- 学习率调度器 ----
    # CosineAnnealingLR 是确定性的，通过 step 到正确位置来恢复
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 快进 scheduler 到 start_epoch 的位置
    # 训练代码中 scheduler.step() 在每个 epoch 结束后调用
    # best_model 保存在 epoch 26 之后 (scheduler 已 step 到 27)
    # 所以需要 step scheduler start_epoch 次
    for i in range(start_epoch):
        scheduler.step()

    if rank == 0:
        resumed_lr = optimizer.param_groups[0]['lr']
        print(f"  Scheduler 恢复到 epoch {start_epoch} (LR: {resumed_lr:.10f})")

    # ---- 训练状态 ----
    best_val_mae = checkpoint.get('val_mae', float('inf'))
    patience_counter = 0
    early_stop_patience = config['early_stop_patience']
    norm_mean, norm_std = ostia_mean, ostia_std

    # 加载已有的训练历史（如果存在）
    history_path = Path(save_dir) / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        if rank == 0:
            print(f"  已有训练历史: {len(history.get('train', []))} epochs")
    else:
        history = {'train': [], 'valid': []}

    if rank == 0:
        print(f"\n  最佳 val_mae: {best_val_mae:.4f}K (来自 epoch {checkpoint['epoch'] + 1})")
        print(f"\n开始恢复训练 (epoch {start_epoch + 1} → {num_epochs}, early_stop={early_stop_patience})...\n")

    # ---- 训练循环 ----
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)

        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, rank, norm_mean, norm_std,
            alpha_mse=config['alpha_mse'],
            alpha_grad=config['alpha_grad'],
            alpha_temporal=config['alpha_temporal']
        )
        valid_metrics = valid_epoch(model, valid_loader, device, epoch, rank, norm_mean, norm_std)

        scheduler.step()

        # Early stopping 逻辑
        improved = valid_metrics['mae'] < best_val_mae
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        if rank == 0:
            history['train'].append(train_metrics)
            history['valid'].append(valid_metrics)

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f} | MAE: {train_metrics['mae']:.3f}K")
            print(f"  Valid - Loss: {valid_metrics['loss']:.4f} | MAE: {valid_metrics['mae']:.3f}K | RMSE: {valid_metrics['rmse']:.3f}K")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f} | Patience: {patience_counter}/{early_stop_patience}")

            if improved:
                best_val_mae = valid_metrics['mae']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_mae': best_val_mae,
                    'valid_metrics': valid_metrics,
                    'norm_mean': norm_mean,
                    'norm_std': norm_std
                }, f'{save_dir}/best_model.pth')
                print(f"  最优模型已保存! MAE: {best_val_mae:.3f}K")

            # 定期保存 checkpoint (每10个epoch)
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_mae': valid_metrics['mae'],
                    'norm_mean': norm_mean,
                    'norm_std': norm_std
                }, f'{save_dir}/checkpoint_epoch_{epoch + 1}.pth')

            # 每个epoch都保存训练历史 (防止中断丢失)
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)

        # Early stopping: 所有rank同步退出
        should_stop = torch.tensor([1 if patience_counter >= early_stop_patience else 0], device=device)
        dist.broadcast(should_stop, src=0)
        if should_stop.item() == 1:
            if rank == 0:
                print(f"\nEarly stopping! 验证集MAE连续{early_stop_patience}个epoch未改善。")
            break

    if rank == 0:
        # 保存最终模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'norm_mean': norm_mean,
            'norm_std': norm_std
        }, f'{save_dir}/final_model.pth')

        # 保存训练历史
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print("\n" + "=" * 80)
        print(f"训练完成! 最优MAE: {best_val_mae:.3f}K")
        print(f"模型保存位置: {save_dir}")
        print("=" * 80)

    cleanup_distributed()


def main():
    # ---- 恢复配置 ----
    # 完全对应原训练的超参数
    npy_dir = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/sst_knn_npy_cache'
    exp_dir = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/run004_jaxa_3dknn_progressive_stride1_lr0.0005'
    resume_ckpt = f'{exp_dir}/best_model.pth'

    config = {
        'npy_dir': npy_dir,
        'resume_checkpoint': resume_ckpt,
        'save_dir': exp_dir,          # 保存到同一目录
        'start_epoch': 27,            # best_model 在 epoch 26 (0-indexed), 从 27 开始

        # 与原训练完全一致的超参数
        'batch_size': 2,              # per GPU
        'num_epochs': 300,            # 总 epochs (T_max)
        'lr': 5e-4,                   # 初始学习率
        'stride': 24,                 # hourly → daily
        'sample_stride': 1,           # 每帧都作为训练样本
        'early_stop_patience': 40,    # early stopping 耐心
        'weight_decay': 5e-4,         # AdamW weight decay
        'alpha_mse': 1.0,             # MSE loss 权重
        'alpha_grad': 0.02,           # 梯度 loss 权重
        'alpha_temporal': 0.1,        # 时序连续性权重
        'world_size': 2,              # 使用 2 块 GPU
    }

    print("=" * 80)
    print("恢复训练配置:")
    print("=" * 80)
    for k, v in config.items():
        if k != 'npy_dir':
            print(f"  {k}: {v}")
    print()

    world_size = config['world_size']

    # Step 1: 预加载数据到共享内存
    all_series = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    shared_data = preload_shared_data(npy_dir, all_series)

    # Step 2: 启动DDP恢复训练
    mp.spawn(resume_worker, args=(world_size, config, shared_data), nprocs=world_size, join=True)

    print("\n恢复训练完成!")


if __name__ == '__main__':
    main()
