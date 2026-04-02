#!/usr/bin/env python3
"""
JAXA Fine-tuning Training Script - 8 GPU DDP (共享内存版)
从OSTIA预训练模型继续微调，使用30天输入序列

特点:
1. 加载OSTIA预训练权重
2. 双输入: sst_seq [30, H, W] + mask_seq [30, H, W]
3. Output Composition: 非挖空区域直接用输入，挖空区域用模型预测
4. Loss只在 loss_mask = artificial_mask ∩ original_obs_mask 区域计算
5. 梯度安全的masked loss（乘法而非索引）
6. 共享内存: 主进程预加载~67GB到/dev/shm，8个DDP进程共享同一份物理内存

作者: Claude Code
日期: 2026-02-17
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ============================================================================
# Loss Functions for JAXA Fine-tuning
# ============================================================================

def masked_mse_loss(pred, target, loss_mask):
    """
    梯度安全的Masked MSE Loss

    只在loss_mask=1的区域计算loss，使用乘法而非索引以保证梯度流

    Args:
        pred: (B, 1, H, W) 预测值
        target: (B, 1, H, W) 真值
        loss_mask: (B, H, W) Loss区域 (1=计算loss, 0=忽略)

    Returns:
        loss: scalar
    """
    mask = loss_mask.unsqueeze(1)  # (B, 1, H, W)
    diff_squared = (pred - target) ** 2
    masked_diff = diff_squared * mask
    loss = masked_diff.sum() / (mask.sum() + 1e-8)
    return loss


def masked_gradient_loss(pred, target, loss_mask):
    """
    梯度匹配损失（在loss_mask区域）

    Args:
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        loss_mask: (B, H, W)

    Returns:
        loss: scalar
    """
    mask = loss_mask.unsqueeze(1)

    # Y方向梯度
    pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_grad_y = mask[:, :, 1:, :]

    # X方向梯度
    pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    mask_grad_x = mask[:, :, :, 1:]

    # 计算梯度误差
    grad_loss_y = torch.abs(pred_grad_y - target_grad_y) * mask_grad_y
    grad_loss_x = torch.abs(pred_grad_x - target_grad_x) * mask_grad_x

    total_grad_loss = (grad_loss_y.sum() / (mask_grad_y.sum() + 1e-8) +
                       grad_loss_x.sum() / (mask_grad_x.sum() + 1e-8)) / 2

    return total_grad_loss


def output_composition(pred, sst_seq, mask_seq, land_mask=None):
    """
    输出组合：非挖空区域用输入值，挖空区域用模型预测，陆地区域保持输入

    Args:
        pred: (B, 1, H, W) 模型预测
        sst_seq: (B, 30, H, W) 输入SST序列
        mask_seq: (B, 30, H, W) mask序列
        land_mask: (B, H, W) 陆地掩码，1=陆地，0=海洋

    Returns:
        composed: (B, 1, H, W) 组合后的输出
    """
    last_input = sst_seq[:, -1:, :, :]  # (B, 1, H, W)
    last_mask = mask_seq[:, -1:, :, :]  # (B, 1, H, W)
    # mask=1: 挖空区域用预测, mask=0: 非挖空区域用输入
    composed = last_input * (1 - last_mask) + pred * last_mask

    # 陆地区域强制使用输入值（保持不变）
    if land_mask is not None:
        land_mask_expanded = land_mask.unsqueeze(1)  # (B, 1, H, W)
        composed = composed * (1 - land_mask_expanded) + last_input * land_mask_expanded

    return composed


def jaxa_combined_loss(pred, target, loss_mask, sst_seq=None,
                       alpha_mse=1.0, alpha_grad=0.2, alpha_temporal=0.1):
    """
    JAXA微调的组合损失（使用output composition后简化版）

    Args:
        pred: (B, 1, H, W) 预测值（已经过output composition）
        target: (B, 1, H, W) 真值（归一化后）
        loss_mask: (B, H, W) Loss区域 = artificial_mask ∩ original_obs_mask
        sst_seq: (B, 30, H, W) 输入SST序列（用于时间连续性）
        alpha_mse: MSE权重（挖空区域重建）
        alpha_grad: 梯度loss权重
        alpha_temporal: 时间连续性权重

    Returns:
        total_loss, loss_mse, loss_grad
    """
    # MSE Loss (在挖空区域 - 重建缺失数据)
    loss_mse = masked_mse_loss(pred, target, loss_mask)

    # 梯度Loss (在挖空区域)
    if alpha_grad > 0:
        loss_grad = masked_gradient_loss(pred, target, loss_mask)
    else:
        loss_grad = torch.tensor(0.0, device=pred.device)

    # 时间连续性约束 (可选)
    if sst_seq is not None and alpha_temporal > 0:
        prev_day = sst_seq[:, -2:-1, :, :]
        expected_change = target - prev_day
        predicted_change = pred - prev_day
        temporal_penalty = F.relu(torch.abs(predicted_change - expected_change) - 0.5)
        loss_temporal = (temporal_penalty * loss_mask.unsqueeze(1)).sum() / (loss_mask.sum() + 1e-8)
    else:
        loss_temporal = torch.tensor(0.0, device=pred.device)

    total_loss = alpha_mse * loss_mse + alpha_grad * loss_grad + alpha_temporal * loss_temporal

    return total_loss, loss_mse, loss_grad


# ============================================================================
# DDP Setup
# ============================================================================

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29510'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


# ============================================================================
# Training & Validation
# ============================================================================

def train_epoch(model, train_loader, optimizer, device, epoch, rank, norm_mean, norm_std,
                alpha_mse=1.0, alpha_grad=0.02, alpha_temporal=0.1):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_grad = 0.0
    total_mae = 0.0
    n_samples = 0

    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    else:
        pbar = train_loader

    for batch in pbar:
        sst_seq = batch['input_sst_seq'].to(device).float()  # [B, 30, H, W]
        mask_seq = batch['mask_seq'].to(device).float()  # [B, 30, H, W]
        gt_sst = batch['ground_truth_sst'].to(device).unsqueeze(1).float()  # [B, 1, H, W]
        loss_mask = batch['loss_mask'].to(device).float()  # [B, H, W]
        land_mask = batch['land_mask'].to(device).float()  # [B, H, W]

        # Forward
        pred = model(sst_seq, mask_seq)

        # Output Composition: 非挖空区域用输入，挖空区域用预测，陆地保持不变
        pred = output_composition(pred, sst_seq, mask_seq, land_mask)

        # Loss (只在挖空区域计算)
        loss, loss_mse, loss_grad = jaxa_combined_loss(
            pred, gt_sst, loss_mask,
            sst_seq=sst_seq,
            alpha_mse=alpha_mse,
            alpha_grad=alpha_grad,
            alpha_temporal=alpha_temporal
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            # 反归一化计算MAE (只在挖空区域)
            pred_kelvin = pred * norm_std + norm_mean
            gt_kelvin = gt_sst * norm_std + norm_mean
            mask_expanded = loss_mask.unsqueeze(1)
            mae = (torch.abs(pred_kelvin - gt_kelvin) * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)

            bs = sst_seq.size(0)
            total_loss += loss.item() * bs
            total_mse += loss_mse.item() * bs
            total_grad += loss_grad.item() * bs
            total_mae += mae.item() * bs
            n_samples += bs

            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/n_samples:.4f}',
                    'MAE': f'{total_mae/n_samples:.3f}K'
                })

    # 同步所有进程
    metrics_tensor = torch.tensor([total_loss, total_mse, total_grad, total_mae, n_samples], device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    total_loss, total_mse, total_grad, total_mae, n_samples = metrics_tensor.tolist()

    return {
        'loss': total_loss / n_samples,
        'mse': total_mse / n_samples,
        'grad': total_grad / n_samples,
        'mae': total_mae / n_samples
    }


def valid_epoch(model, valid_loader, device, epoch, rank, norm_mean, norm_std):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    n_samples = 0

    if rank == 0:
        pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1} [Valid]")
    else:
        pbar = valid_loader

    with torch.no_grad():
        for batch in pbar:
            sst_seq = batch['input_sst_seq'].to(device).float()
            mask_seq = batch['mask_seq'].to(device).float()
            gt_sst = batch['ground_truth_sst'].to(device).unsqueeze(1).float()
            loss_mask = batch['loss_mask'].to(device).float()
            land_mask = batch['land_mask'].to(device).float()

            pred = model(sst_seq, mask_seq)

            # Output Composition: 非挖空区域用输入，挖空区域用预测，陆地保持不变
            pred = output_composition(pred, sst_seq, mask_seq, land_mask)

            # Loss (只在挖空区域计算)
            loss, loss_mse, _ = jaxa_combined_loss(
                pred, gt_sst, loss_mask,
                sst_seq=sst_seq,
                alpha_grad=0  # 验证时不计算梯度loss
            )

            # 反归一化
            pred_kelvin = pred * norm_std + norm_mean
            gt_kelvin = gt_sst * norm_std + norm_mean
            mask_expanded = loss_mask.unsqueeze(1)

            error = torch.abs(pred_kelvin - gt_kelvin) * mask_expanded
            mae = error.sum() / (mask_expanded.sum() + 1e-8)
            rmse = torch.sqrt(((pred_kelvin - gt_kelvin)**2 * mask_expanded).sum() / (mask_expanded.sum() + 1e-8))

            bs = sst_seq.size(0)
            total_loss += loss.item() * bs
            total_mse += loss_mse.item() * bs
            total_mae += mae.item() * bs
            total_rmse += rmse.item() * bs
            n_samples += bs

    metrics_tensor = torch.tensor([total_loss, total_mse, total_mae, total_rmse, n_samples], device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    total_loss, total_mse, total_mae, total_rmse, n_samples = metrics_tensor.tolist()

    return {
        'loss': total_loss / n_samples,
        'mse': total_mse / n_samples,
        'mae': total_mae / n_samples,
        'rmse': total_rmse / n_samples
    }


# ============================================================================
# Main Training Worker
# ============================================================================

def train_worker(rank, world_size, config, shared_data):
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    # 配置
    save_dir = config['save_dir']
    pretrained_path = config['pretrained_path']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    lr = config['lr']

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print("\n" + "="*80)
        print("JAXA Fine-tuning Training (8 GPU DDP, Shared Memory)")
        print("="*80)
        print(f"\n配置:")
        print(f"  - 预训练模型: {pretrained_path}")
        print(f"  - 保存目录: {save_dir}")
        print(f"  - Batch size: {batch_size} per GPU x {world_size} GPUs = {batch_size*world_size}")
        print(f"  - Learning rate: {lr}")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - 输入: 30天序列 (60通道: 30xSST + 30xmask)")
        print(f"  - Loss: 只在 artificial_mask ∩ original_obs_mask 区域计算")
        print(f"  - IO: 共享内存 (零磁盘IO)")
        print()

    # 数据集 - 使用共享内存张量
    ostia_mean = 299.9221  # OSTIA训练时的mean (Kelvin)
    ostia_std = 2.6919     # OSTIA训练时的std (Kelvin)

    train_dataset = JAXAFinetuneDataset(
        shared_data=shared_data,
        series_ids=[0, 1, 2, 3, 4, 5, 6, 7],  # 8年训练数据
        window_size=30,
        stride=24,  # hourly数据每隔24帧取一帧
        sample_stride=config.get('sample_stride', 24),  # 训练样本间隔，消除hourly重复
        mask_ratio=0.2,
        min_mask_size=10,
        max_mask_size=50,
        normalize=True,
        mean=ostia_mean,
        std=ostia_std,
        seed=42 + rank  # 每个rank不同seed，增加挖空多样性
    )

    valid_dataset = JAXAFinetuneDataset(
        shared_data=shared_data,
        series_ids=[8],  # 1年验证数据
        window_size=30,
        stride=24,
        sample_stride=config.get('sample_stride', 24),  # 验证集也用相同采样
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

    # 共享内存版: 数据在RAM中，IO不再是瓶颈
    # num_workers=2 用于并行做mask生成等CPU计算
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
        print(f"  - Normalize: mean={train_dataset.mean:.2f}K, std={train_dataset.std:.2f}K")

    # 模型
    model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64, width=64, depth=6).to(device)

    # 加载预训练权重
    if pretrained_path and os.path.exists(pretrained_path):
        if rank == 0:
            print(f"\n加载预训练模型: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        if rank == 0:
            print("  预训练权重加载成功")
    else:
        if rank == 0:
            print("\n未找到预训练模型，从头开始训练")

    model = DDP(model, device_ids=[rank])

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n模型参数量: {total_params:,}")

    # 优化器
    weight_decay = config.get('weight_decay', 5e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 训练
    best_val_mae = float('inf')
    patience_counter = 0
    early_stop_patience = config.get('early_stop_patience', 20)
    norm_mean, norm_std = train_dataset.mean, train_dataset.std
    history = {'train': [], 'valid': []}

    if rank == 0:
        print(f"\n开始训练 ({num_epochs} epochs, early_stop={early_stop_patience})...\n")

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)  # Shift sample offset so each epoch covers different hours

        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, rank, norm_mean, norm_std,
                                     alpha_mse=config.get('alpha_mse', 1.0),
                                     alpha_grad=config.get('alpha_grad', 0.02),
                                     alpha_temporal=config.get('alpha_temporal', 0.1))
        valid_metrics = valid_epoch(model, valid_loader, device, epoch, rank, norm_mean, norm_std)

        scheduler.step()

        # Early stopping 逻辑 (所有rank同步判断)
        improved = valid_metrics['mae'] < best_val_mae
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        if rank == 0:
            history['train'].append(train_metrics)
            history['valid'].append(valid_metrics)

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f} | MAE: {train_metrics['mae']:.3f}K")
            print(f"  Valid - Loss: {valid_metrics['loss']:.4f} | MAE: {valid_metrics['mae']:.3f}K | RMSE: {valid_metrics['rmse']:.3f}K")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f} | Patience: {patience_counter}/{early_stop_patience}")

            if improved:
                best_val_mae = valid_metrics['mae']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': best_val_mae,
                    'valid_metrics': valid_metrics,
                    'norm_mean': norm_mean,
                    'norm_std': norm_std
                }, f'{save_dir}/best_model.pth')
                print(f"  最优模型已保存! MAE: {best_val_mae:.3f}K")

            # 定期保存checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_mae': valid_metrics['mae'],
                    'norm_mean': norm_mean,
                    'norm_std': norm_std
                }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pth')

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
            'epoch': num_epochs - 1,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'norm_mean': norm_mean,
            'norm_std': norm_std
        }, f'{save_dir}/final_model.pth')

        # 保存训练历史
        with open(f'{save_dir}/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # 保存配置
        config_save = {k: v for k, v in config.items() if k != 'shared_data'}
        with open(f'{save_dir}/config.json', 'w') as f:
            json.dump(config_save, f, indent=2)

        print("\n" + "="*80)
        print(f"训练完成! 最优MAE: {best_val_mae:.3f}K")
        print(f"模型保存位置: {save_dir}")
        print("="*80)

    cleanup_distributed()


# ============================================================================
# Experiment Management
# ============================================================================

EXPERIMENTS_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments')
EXPERIMENT_LOG = EXPERIMENTS_DIR / 'experiment_log.jsonl'


def get_experiment_history():
    """读取所有历史实验记录，返回列表（按时间排序）"""
    if not EXPERIMENT_LOG.exists():
        return []
    records = []
    with open(EXPERIMENT_LOG, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_experiment_history():
    """打印历史实验表格，方便调参参考"""
    records = get_experiment_history()
    if not records:
        print("暂无历史实验记录。")
        return

    print(f"\n{'='*100}")
    print(f"历史实验记录 ({len(records)} 次)")
    print(f"{'='*100}")
    print(f"{'ID':<6} {'名称':<30} {'KNN':<8} {'LR':<10} {'BS':<5} "
          f"{'Epochs':<8} {'BestMAE':<10} {'BestRMSE':<10} {'日期':<12}")
    print("-" * 100)
    for r in records:
        cfg = r.get('config', {})
        res = r.get('results', {})
        print(f"{r.get('id','?'):<6} {r.get('name','?'):<30} "
              f"{cfg.get('knn_method','?'):<8} "
              f"{cfg.get('lr','?'):<10} {cfg.get('batch_size','?'):<5} "
              f"{res.get('actual_epochs','?'):<8} "
              f"{res.get('best_val_mae','?'):<10} "
              f"{res.get('best_val_rmse','?'):<10} "
              f"{r.get('date','?'):<12}")
    print(f"{'='*100}\n")


def create_experiment_dir(experiment_name):
    """创建带自增编号的实验目录，返回路径"""
    records = get_experiment_history()
    next_id = max([r.get('id', 0) for r in records], default=0) + 1
    dir_name = f"run{next_id:03d}_{experiment_name}"
    exp_dir = EXPERIMENTS_DIR / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir, next_id, dir_name


def log_experiment(exp_id, name, config, results, save_dir):
    """追加一条实验记录到 experiment_log.jsonl"""
    import time as _time
    record = {
        'id': exp_id,
        'name': name,
        'date': _time.strftime('%Y-%m-%d'),
        'save_dir': str(save_dir),
        'config': {
            'knn_method': config.get('knn_method', '3d_spatiotemporal'),
            'pretrained_path': config.get('pretrained_path', ''),
            'lr': config.get('lr'),
            'batch_size': config.get('batch_size'),
            'num_epochs': config.get('num_epochs'),
            'early_stop_patience': config.get('early_stop_patience'),
            'stride': config.get('stride'),
            'sample_stride': config.get('sample_stride'),
            'weight_decay': config.get('weight_decay', 5e-4),
            'alpha_mse': config.get('alpha_mse', 1.0),
            'alpha_grad': config.get('alpha_grad', 0.02),
            'alpha_temporal': config.get('alpha_temporal', 0.1),
            'world_size': config.get('world_size', 8),
        },
        'results': results,
    }
    with open(EXPERIMENT_LOG, 'a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 同时保存一份到实验目录
    with open(Path(save_dir) / 'experiment_record.json', 'w') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def main():
    # ---- 打印历史实验，方便调参参考 ----
    print_experiment_history()

    # ---- 配置 ----
    npy_dir = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/sst_knn_npy_cache'

    config = {
        'npy_dir': npy_dir,
        'pretrained_path': '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/jaxa_finetune_8years/best_model.pth',
        'batch_size': 2,       # per GPU
        'num_epochs': 300,
        'lr': 5e-4,
        'stride': 24,
        'sample_stride': 1,
        'early_stop_patience': 40,
        'weight_decay': 5e-4,
        'alpha_mse': 1.0,
        'alpha_grad': 0.02,
        'alpha_temporal': 0.1,
        'knn_method': '3d_progressive',
        'world_size': 4,
    }

    # ---- 创建实验目录（自动递增编号） ----
    exp_name = f"jaxa_3dknn_progressive_stride{config['sample_stride']}_lr{config['lr']}"
    exp_dir, exp_id, dir_name = create_experiment_dir(exp_name)
    config['save_dir'] = str(exp_dir)

    print(f"本次实验: #{exp_id} - {dir_name}")
    print(f"保存目录: {exp_dir}")

    world_size = config['world_size']

    # Step 1: 预加载所有数据到共享内存（在mp.spawn之前）
    all_series = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    shared_data = preload_shared_data(npy_dir, all_series)

    # Step 2: 启动DDP训练
    mp.spawn(train_worker, args=(world_size, config, shared_data), nprocs=world_size, join=True)

    # Step 3: 训练结束后，读取结果并记录
    history_path = exp_dir / 'training_history.json'
    results = {}
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        if history.get('valid'):
            val_maes = [e['mae'] for e in history['valid']]
            val_rmses = [e['rmse'] for e in history['valid']]
            best_idx = int(np.argmin(val_maes))
            results = {
                'best_val_mae': round(val_maes[best_idx], 4),
                'best_val_rmse': round(val_rmses[best_idx], 4),
                'best_epoch': best_idx + 1,
                'actual_epochs': len(val_maes),
                'final_train_mae': round(history['train'][-1]['mae'], 4),
                'final_val_mae': round(val_maes[-1], 4),
            }
    log_experiment(exp_id, dir_name, config, results, exp_dir)
    print(f"\n实验记录已保存到: {EXPERIMENT_LOG}")


if __name__ == '__main__':
    main()
