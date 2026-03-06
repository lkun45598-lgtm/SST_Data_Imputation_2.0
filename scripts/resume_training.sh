#!/bin/bash
# 恢复训练脚本
# 从 run004_jaxa_3dknn_progressive_stride1_lr0.0005 的 best_model (epoch 26) 恢复
# 使用 GPU 0,1 (2卡 DDP)
#
# 用法:
#   bash scripts/resume_training.sh           # 前台运行
#   nohup bash scripts/resume_training.sh &   # 后台运行 (日志写入 nohup.out)

set -e

cd /data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM

export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "恢复训练 - $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

/home/lz/miniconda3/envs/pytorch/bin/python training/resume_train_jaxa.py \
    2>&1 | tee experiments/resume_run004_stride1.log
