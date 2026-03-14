#!/usr/bin/env python3
"""
JAXA Fine-tuning Training Script - Updated for New Data
使用修复后的完整数据（74873帧，8.5年）

更新内容：
1. 数据路径：/data/chla_data_imputation_data_260125/sst_post_filtered/
2. 使用完整74873帧数据
3. 训练集：序列0-7（8年）
4. 验证集：序列8（1年）

作者: Claude Code
日期: 2026-03-09
"""

import os
import sys

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入原始训练脚本的所有内容
from training.train_jaxa import *


def main():
    # 更新配置
    config = {
        'data_dir': '/data/chla_data_imputation_data_260125/sst_post_filtered',  # 新数据路径
        'save_dir': '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/jaxa_finetune_corrected',
        'pretrained_path': '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/run004_jaxa_3dknn_progressive_stride1_lr0.0005/best_model.pth',
        'batch_size': 2,  # per GPU
        'num_epochs': 100,
        'lr': 5e-4,
    }

    world_size = 4  # 4卡DDP
    mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
