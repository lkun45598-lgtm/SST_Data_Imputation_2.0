#!/usr/bin/env python3
"""
JAXA Fine-tuning Dataset with Artificial Masking
用于JAXA微调的数据集，支持30天输入序列和人工挖空

特点:
1. 双输入: sst_seq [30, H, W] + mask_seq [30, H, W]
2. 人工方形挖空: 只在原始观测区域 (original_obs_mask) 上生成
3. Loss区域: artificial_mask ∩ original_obs_mask
4. 共享内存方案: 主进程预加载数据 → share_memory_() → DDP进程共享
   一份物理内存(~67GB)被所有进程共享，零磁盘IO

作者: Claude Code
日期: 2026-02-17
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time


# ============================================================================
# H5 → npy 转换（训练前一次性执行）
# ============================================================================

def _convert_single_series(args):
    """转换单个序列的H5到npy（子进程函数）"""
    sid, h5_path, npy_dir = args
    sst_npy = Path(npy_dir) / f'sst_{sid:02d}.npy'
    obs_npy = Path(npy_dir) / f'obs_{sid:02d}.npy'
    miss_npy = Path(npy_dir) / f'miss_{sid:02d}.npy'
    land_npy = Path(npy_dir) / f'land_{sid:02d}.npy'

    if sst_npy.exists() and obs_npy.exists() and miss_npy.exists() and land_npy.exists():
        return sid, 0, "skip"

    t0 = time.time()
    with h5py.File(h5_path, 'r') as f:
        np.save(str(sst_npy), f['sst_data'][:])
        np.save(str(obs_npy), f['original_obs_mask'][:])
        np.save(str(miss_npy), f['original_missing_mask'][:])
        if 'land_mask' in f:
            np.save(str(land_npy), f['land_mask'][:])
        else:
            np.save(str(land_npy), np.isnan(f['sst_data'][0]).astype(np.float32))

    elapsed = time.time() - t0
    size_gb = sum(p.stat().st_size for p in [sst_npy, obs_npy, miss_npy, land_npy]) / 1e9
    return sid, elapsed, f"{size_gb:.1f}GB"


def prepare_npy_cache(data_dir, series_ids, num_workers=216):
    """
    并行将H5转换为npy文件（训练前调用，DDP spawn之前）

    Args:
        data_dir: H5数据目录
        series_ids: 需要转换的序列ID列表
        num_workers: 并行进程数
    """
    data_dir = Path(data_dir)
    npy_dir = data_dir / 'npy_cache'
    npy_dir.mkdir(parents=True, exist_ok=True)

    # 检查哪些需要转换
    tasks = []
    for sid in series_ids:
        h5_path = data_dir / f'jaxa_knn_filled_{sid:02d}.h5'
        sst_npy = npy_dir / f'sst_{sid:02d}.npy'
        if h5_path.exists() and not sst_npy.exists():
            tasks.append((sid, str(h5_path), str(npy_dir)))

    if not tasks:
        print(f"npy缓存已就绪 ({npy_dir})")
        return

    print(f"\n{'='*60}")
    print(f"H5 → npy 转换 ({len(tasks)}个序列, {num_workers}核心)")
    print(f"{'='*60}")

    actual_workers = min(num_workers, len(tasks))
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        for sid, elapsed, info in executor.map(_convert_single_series, tasks):
            print(f"  Series {sid}: {elapsed:.1f}s, {info}", flush=True)

    print(f"转换完成! 总耗时: {time.time()-t0:.1f}s")
    print(f"{'='*60}\n")


# ============================================================================
# Shared Memory Preloading
# ============================================================================

def preload_shared_data(npy_dir, series_ids):
    """
    预加载所有npy数据到PyTorch共享内存

    在DDP mp.spawn之前调用。数据通过/dev/shm共享，
    所有DDP进程 + DataLoader workers共享同一份物理内存。

    Args:
        npy_dir: npy缓存目录
        series_ids: 序列ID列表

    Returns:
        shared_data: dict, {series_id: {'sst': Tensor, 'obs': Tensor, 'miss': Tensor, 'land': Tensor}}
    """
    npy_dir = Path(npy_dir)
    shared_data = {}
    total_bytes = 0

    print(f"\n{'='*60}")
    print(f"预加载数据到共享内存 ({len(series_ids)}个序列)")
    print(f"{'='*60}")

    t0 = time.time()

    for sid in series_ids:
        t1 = time.time()

        sst_path = npy_dir / f'sst_{sid:02d}.npy'
        obs_path = npy_dir / f'obs_{sid:02d}.npy'
        miss_path = npy_dir / f'miss_{sid:02d}.npy'
        land_path = npy_dir / f'land_{sid:02d}.npy'

        if not sst_path.exists():
            print(f"  Warning: {sst_path} not found, skipping series {sid}")
            continue

        # 逐个加载到共享内存，每次只有一个npy在普通内存中
        sst_np = np.load(str(sst_path)).astype(np.float32)
        sst = torch.from_numpy(sst_np).share_memory_()
        del sst_np

        obs_np = np.load(str(obs_path)).astype(np.float32)
        obs = torch.from_numpy(obs_np).share_memory_()
        del obs_np

        miss_np = np.load(str(miss_path)).astype(np.float32)
        miss = torch.from_numpy(miss_np).share_memory_()
        del miss_np

        land_np = np.load(str(land_path)).astype(np.float32)
        land = torch.from_numpy(land_np).share_memory_()
        del land_np

        series_bytes = (sst.nelement() + obs.nelement() + miss.nelement() + land.nelement()) * 4
        total_bytes += series_bytes

        shared_data[sid] = {
            'sst': sst,
            'obs': obs,
            'miss': miss,
            'land': land
        }

        elapsed = time.time() - t1
        print(f"  Series {sid}: {sst.shape[0]} frames, "
              f"{series_bytes/1e9:.1f}GB, {elapsed:.1f}s", flush=True)

    total_time = time.time() - t0
    print(f"\n共享内存加载完成!")
    print(f"  总数据量: {total_bytes/1e9:.1f}GB")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  所有DDP进程将共享同一份物理内存")
    print(f"{'='*60}\n")

    return shared_data


# ============================================================================
# Mask Generator
# ============================================================================

class SquareMaskGenerator:
    """方形挖空生成器"""

    def __init__(self, mask_ratio=0.2, min_size=10, max_size=50, seed=None):
        """
        Args:
            mask_ratio: 目标挖空比例（相对于有效区域）
            min_size: 方形最小边长
            max_size: 方形最大边长
            seed: 随机种子
        """
        self.mask_ratio = mask_ratio
        self.min_size = min_size
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)

    def generate(self, valid_mask, target_ratio=None):
        """
        在valid_mask区域内生成方形挖空

        Args:
            valid_mask: 有效区域掩码 (H, W), 1=可挖空, 0=不可挖空
            target_ratio: 目标挖空比例，None则使用默认值

        Returns:
            artificial_mask: 挖空掩码 (H, W), 1=挖空, 0=保留
        """
        if target_ratio is None:
            target_ratio = self.mask_ratio

        H, W = valid_mask.shape
        artificial_mask = np.zeros((H, W), dtype=np.float32)

        valid_count = valid_mask.sum()
        if valid_count == 0:
            return artificial_mask

        target_masked = int(valid_count * target_ratio)
        current_masked = 0

        # 获取有效区域的边界
        valid_y, valid_x = np.where(valid_mask == 1)
        if len(valid_y) == 0:
            return artificial_mask

        y_min, y_max = valid_y.min(), valid_y.max()
        x_min, x_max = valid_x.min(), valid_x.max()

        max_attempts = 1000
        attempts = 0

        while current_masked < target_masked and attempts < max_attempts:
            # 随机方形大小
            size = self.rng.integers(self.min_size, self.max_size + 1)

            # 随机位置（在有效区域范围内）
            if y_max - size < y_min or x_max - size < x_min:
                attempts += 1
                continue

            y_start = self.rng.integers(y_min, max(y_min + 1, y_max - size + 1))
            x_start = self.rng.integers(x_min, max(x_min + 1, x_max - size + 1))

            y_end = min(y_start + size, H)
            x_end = min(x_start + size, W)

            # 只挖空valid_mask区域内的像素
            region = valid_mask[y_start:y_end, x_start:x_end].copy()
            new_masked = region.sum() - (artificial_mask[y_start:y_end, x_start:x_end] * region).sum()

            if new_masked > 0:
                artificial_mask[y_start:y_end, x_start:x_end] = np.where(
                    region == 1, 1.0, artificial_mask[y_start:y_end, x_start:x_end]
                )
                current_masked = (artificial_mask * valid_mask).sum()

            attempts += 1

        return artificial_mask


# ============================================================================
# Dataset
# ============================================================================

class JAXAFinetuneDataset(Dataset):
    """
    JAXA微调数据集 - 共享内存版

    使用预加载的共享内存张量，零磁盘IO。
    所有DDP进程共享同一份物理内存（~67GB）。
    支持stride参数：hourly数据stride=24，每隔24帧取一帧构建30天窗口。
    """

    def __init__(self, shared_data, series_ids=None, window_size=30, stride=1,
                 sample_stride=1,
                 mask_ratio=0.2, min_mask_size=10, max_mask_size=50,
                 normalize=True, mean=None, std=None, seed=42):
        """
        Args:
            shared_data: dict, {series_id: {'sst': Tensor, 'obs': Tensor, 'miss': Tensor, 'land': Tensor}}
            series_ids: 序列ID列表
            window_size: 时间窗口大小
            stride: 帧间隔 (hourly数据用24，构建30天窗口时的步长)
            sample_stride: 训练样本间隔 (24=每天1个样本，消除hourly重复)
            mask_ratio: 人工挖空比例
            min_mask_size: 方形最小边长
            max_mask_size: 方形最大边长
            normalize: 是否归一化
            mean: 归一化均值
            std: 归一化标准差
            seed: 随机种子
        """
        self.window_size = window_size
        self.stride = stride
        self.sample_stride = sample_stride
        self.normalize = normalize
        self._epoch_offset = 0

        if series_ids is None:
            series_ids = [0, 1, 2]

        self.series_info = []
        self.cumulative_frames = [0]
        self._sst_data = []
        self._obs_data = []
        self._miss_data = []
        self._land_masks = []

        total_frames = 0

        for sid in series_ids:
            if sid not in shared_data:
                print(f"  Warning: series {sid} not in shared_data, skipping")
                continue

            sd = shared_data[sid]
            sst = sd['sst']
            obs = sd['obs']
            miss = sd['miss']
            land = sd['land']

            num_frames = sst.shape[0]
            shape = tuple(sst.shape[1:])

            self.series_info.append({
                'series_id': sid,
                'num_frames': num_frames,
                'shape': shape
            })

            self._sst_data.append(sst)
            self._obs_data.append(obs)
            self._miss_data.append(miss)
            self._land_masks.append(land)

            total_frames += num_frames
            self.cumulative_frames.append(total_frames)

        self.total_frames = total_frames
        self.shape = shape if self.series_info else (451, 351)

        # Build sampled index list
        self._build_sample_indices()

        if mean is None or std is None:
            self.mean = 300.0
            self.std = 5.0
        else:
            self.mean = mean
            self.std = std

        self.mask_generator = SquareMaskGenerator(
            mask_ratio=mask_ratio, min_size=min_mask_size,
            max_size=max_mask_size, seed=seed
        )

        print(f"\nJAXA Finetune Dataset initialized:")
        print(f"  Series: {[s['series_id'] for s in self.series_info]}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Sample stride: {sample_stride} (samples per epoch: {len(self._sample_indices)})")
        print(f"  Window size: {window_size}")
        print(f"  Stride: {stride} ({'hourly->daily' if stride == 24 else 'consecutive'})")
        print(f"  Mask ratio: {mask_ratio*100:.0f}%")
        print(f"  Normalize: {normalize} (mean={self.mean:.2f}, std={self.std:.2f})")
        print(f"  Storage: shared memory (zero-copy)")

    def _build_sample_indices(self):
        """Build sampled frame indices based on sample_stride and current epoch offset."""
        self._sample_indices = []
        for global_idx in range(self._epoch_offset, self.total_frames, self.sample_stride):
            self._sample_indices.append(global_idx)

    def set_epoch(self, epoch):
        """Update epoch offset so each epoch samples different hours.

        With sample_stride=24, epoch 0 uses hour 0, epoch 1 uses hour 1, etc.
        After 24 epochs, the cycle repeats. This ensures all 24 hours are covered.
        """
        self._epoch_offset = epoch % self.sample_stride
        self._build_sample_indices()

    def __len__(self):
        return len(self._sample_indices)

    def _get_series_and_local_idx(self, global_idx):
        for i, info in enumerate(self.series_info):
            if global_idx < self.cumulative_frames[i + 1]:
                return i, global_idx - self.cumulative_frames[i]
        raise IndexError(f"Index {global_idx} out of range")

    def __getitem__(self, idx):
        # Map sampled index to actual global frame index
        actual_idx = self._sample_indices[idx]
        series_idx, local_idx = self._get_series_and_local_idx(actual_idx)

        sst_t = self._sst_data[series_idx]
        obs_t = self._obs_data[series_idx]
        miss_t = self._miss_data[series_idx]
        land_t = self._land_masks[series_idx]

        # 构建30帧索引（stride间隔），负索引clamp到0
        frame_indices = [max(0, local_idx - t * self.stride)
                         for t in range(self.window_size - 1, -1, -1)]

        # 从共享内存读取（纯内存操作，零磁盘IO）
        sst_seq = sst_t[frame_indices].numpy().copy()  # (30, H, W)
        mask_seq = miss_t[frame_indices].numpy().copy()  # (30, H, W)

        # target帧
        ground_truth_sst = sst_t[local_idx].numpy().copy()  # (H, W)
        original_obs_mask = obs_t[local_idx].numpy().copy()  # (H, W)
        land_mask = land_t.numpy().copy()  # (H, W)

        # 人工挖空
        valid_for_mask = original_obs_mask * (1 - land_mask)
        artificial_mask = self.mask_generator.generate(valid_for_mask)

        loss_mask = (artificial_mask * original_obs_mask * (1 - land_mask)).astype(np.float32)

        mask_seq[-1] = artificial_mask

        sst_seq_input = sst_seq.copy()
        sst_seq_input[-1] = np.where(artificial_mask == 1, self.mean, sst_seq[-1])

        if self.normalize:
            sst_seq_input = (sst_seq_input - self.mean) / self.std
            ground_truth_sst = (ground_truth_sst - self.mean) / self.std

        sst_seq_input = np.nan_to_num(sst_seq_input, nan=0.0)
        ground_truth_sst = np.nan_to_num(ground_truth_sst, nan=0.0)

        return {
            'input_sst_seq': sst_seq_input.astype(np.float32),
            'mask_seq': mask_seq.astype(np.float32),
            'ground_truth_sst': ground_truth_sst.astype(np.float32),
            'loss_mask': loss_mask.astype(np.float32),
            'original_obs_mask': original_obs_mask.astype(np.float32),
            'land_mask': land_mask.astype(np.float32),
            'time_index': actual_idx
        }


def test_dataset():
    """测试共享内存数据集"""
    print("=" * 70)
    print("Testing JAXAFinetuneDataset (shared memory)")
    print("=" * 70)

    npy_dir = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/sst_knn_npy_cache'

    # 预加载到共享内存
    shared_data = preload_shared_data(npy_dir, [0])

    dataset = JAXAFinetuneDataset(
        shared_data=shared_data,
        series_ids=[0],
        window_size=30,
        stride=24,
        mask_ratio=0.2,
        normalize=True,
        mean=299.9221,
        std=2.6919,
        seed=42
    )

    print(f"\nDataset length: {len(dataset)}")

    # 速度测试
    t0 = time.time()
    for i in range(100):
        sample = dataset[i * 80]
    elapsed = time.time() - t0
    print(f"\n100个样本: {elapsed:.3f}s, 平均 {elapsed/100*1000:.1f}ms/样本")

    sample = dataset[500]
    print(f"\nSample shapes:")
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")

    print("\nOK!")


if __name__ == '__main__':
    test_dataset()
