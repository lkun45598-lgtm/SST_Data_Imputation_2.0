#!/bin/bash
# =============================================================
# SST Preprocessing Pipeline: temporal_fill → lowpass_filter → knn_fill → post_filter
# =============================================================
# Pipeline:
#   1. Temporal weighted fill (diurnal-aware): raw JAXA → sst_temperal_data/
#   2. Gaussian low-pass filter (σ=1.5):      sst_temperal_data/ → sst_filtered/
#   3. 3D spatiotemporal KNN fill:             sst_filtered/ → sst_knn_filled/
#   4. Post-KNN Gaussian filter (σ=1.5):      sst_knn_filled/ → sst_post_filtered/
#
# Usage:
#   bash run_preprocessing.sh              # Run all 4 steps, all series
#   bash run_preprocessing.sh --step 2     # Run only step 2 (filter)
#   bash run_preprocessing.sh --step 2 3   # Run steps 2 and 3
#   bash run_preprocessing.sh --series 0 1 # Only process series 0 and 1
# =============================================================

set -e

cd "$(dirname "$0")/.."

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
PREPROCESSING_DIR="preprocessing"

# Parse arguments
STEPS=()
SERIES_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                STEPS+=("$1")
                shift
            done
            ;;
        --series)
            shift
            SERIES_ARGS="--series"
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SERIES_ARGS="$SERIES_ARGS $1"
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Default: run all steps
if [ ${#STEPS[@]} -eq 0 ]; then
    STEPS=(1 2 3 4)
fi

echo "============================================================="
echo "SST Preprocessing Pipeline"
echo "============================================================="
echo "Steps: ${STEPS[*]}"
echo "Series: ${SERIES_ARGS:-all}"
echo ""

for step in "${STEPS[@]}"; do
    case $step in
        1)
            echo "============================================================="
            echo "Step 1: Temporal Weighted Fill (diurnal-aware, α=0.5)"
            echo "============================================================="
            if [ -n "$SERIES_ARGS" ]; then
                python $PREPROCESSING_DIR/temporal_weighted_fill.py --mode single $SERIES_ARGS --workers 216
            else
                python $PREPROCESSING_DIR/temporal_weighted_fill.py --mode full --workers 216
            fi
            echo ""
            ;;
        2)
            echo "============================================================="
            echo "Step 2: Gaussian Low-Pass Filter (σ=1.5)"
            echo "============================================================="
            python $PREPROCESSING_DIR/lowpass_filter.py $SERIES_ARGS --sigma 1.5 --workers 216
            echo ""
            ;;
        3)
            echo "============================================================="
            echo "Step 3: 3D Spatiotemporal KNN Fill"
            echo "============================================================="
            if [ -n "$SERIES_ARGS" ]; then
                python $PREPROCESSING_DIR/knn_fill_3d.py $SERIES_ARGS
            else
                python $PREPROCESSING_DIR/knn_fill_3d.py
            fi
            echo ""
            ;;
        4)
            echo "============================================================="
            echo "Step 4: Post-KNN Gaussian Filter (σ=1.5)"
            echo "============================================================="
            python $PREPROCESSING_DIR/post_knn_filter.py $SERIES_ARGS --sigma 1.5 --workers 216
            echo ""
            ;;
        *)
            echo "Unknown step: $step (valid: 1, 2, 3, 4)"
            exit 1
            ;;
    esac
done

echo "============================================================="
echo "Pipeline complete!"
echo "============================================================="
