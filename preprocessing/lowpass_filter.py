#!/usr/bin/env python3
"""
JAXA SST Low-Pass Filter — Gaussian smoothing to reduce temporal aliasing artifacts

Processing pipeline position:
    temporal_weighted_fill → [THIS SCRIPT: Gaussian filter] → knn_fill_3d

Problem:
    Temporal weighted filling combines observations from different hours of the day.
    Due to the diurnal SST cycle (~2.3K variation), this creates speckle/aliasing
    artifacts at boundaries between observed and temporally-filled pixels.

Solution:
    Apply a Gaussian low-pass filter (σ=1.5) to each frame BEFORE KNN filling.
    Only valid (non-NaN) pixels are filtered; NaN areas are preserved for KNN.

Input:  sst_temperal_data/jaxa_weighted_series_XX.h5
Output: sst_filtered/jaxa_filtered_XX.h5

Usage:
    python lowpass_filter.py                    # Process all 9 series
    python lowpass_filter.py --series 0 1 2     # Process specific series
    python lowpass_filter.py --sigma 2.0        # Custom sigma
"""

import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from scipy import ndimage

# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path('/data/chla_data_imputation_data_260125/sst_temperal_data')
OUTPUT_DIR = Path('/data/chla_data_imputation_data_260125/sst_filtered')

DEFAULT_SIGMA = 1.5
NUM_WORKERS = 216


# ============================================================
# Filter Functions
# ============================================================

def gaussian_filter_frame(args):
    """
    Apply Gaussian low-pass filter ONLY to temporally-filled pixels.
    Real satellite observations are preserved untouched.

    Strategy:
    1. Temporarily fill NaN with local mean (to avoid edge artifacts)
    2. Apply Gaussian filter to the full image
    3. Only replace temporally-filled pixels (fill_mask=1) with filtered values
    4. Real observations (fill_mask=0, not NaN) are kept as-is

    Args:
        args: (frame_idx, sst_frame, missing_mask_frame, fill_mask_frame, sigma)

    Returns:
        (frame_idx, filtered_frame)
    """
    frame_idx, sst_frame, missing_mask_frame, fill_mask_frame, sigma = args

    filtered = sst_frame.copy()

    # Pixels that are temporally filled: fill_mask=1 and not still missing
    temporal_filled = (fill_mask_frame == 1) & (missing_mask_frame == 0) & (~np.isnan(sst_frame))

    if temporal_filled.sum() == 0:
        return frame_idx, filtered

    # Valid pixels = observed + temporally filled (exclude still-missing NaN)
    valid_mask = (missing_mask_frame == 0) & (~np.isnan(sst_frame))

    # Temporarily fill NaN with valid area mean (to avoid edge artifacts in filter)
    temp_data = sst_frame.copy()
    mean_val = np.nanmean(sst_frame[valid_mask])
    temp_data[~valid_mask] = mean_val

    # Apply Gaussian filter
    filtered_temp = ndimage.gaussian_filter(temp_data, sigma=sigma, mode='reflect')

    # ONLY replace temporally-filled pixels, keep real observations untouched
    filtered[temporal_filled] = filtered_temp[temporal_filled]

    return frame_idx, filtered


# ============================================================
# Series Processing
# ============================================================

def process_series(series_id, sigma=DEFAULT_SIGMA, num_workers=NUM_WORKERS):
    """Process a single JAXA series with Gaussian low-pass filter."""

    input_path = INPUT_DIR / f'jaxa_weighted_series_{series_id:02d}.h5'
    output_path = OUTPUT_DIR / f'jaxa_filtered_{series_id:02d}.h5'

    if not input_path.exists():
        print(f"  Series {series_id}: input not found ({input_path}), skipping")
        return None

    print(f"\n{'='*70}")
    print(f"Series {series_id}: Gaussian filter (sigma={sigma})")
    print(f"{'='*70}")

    start_time = time.time()

    # Load data
    print(f"  Loading {input_path.name}...", flush=True)
    with h5py.File(str(input_path), 'r') as f:
        sst_data = f['sst_data'][:]          # (T, H, W)
        missing_mask = f['missing_mask'][:]   # (T, H, W)
        fill_mask = f['fill_mask'][:]         # (T, H, W)
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        timestamps = f['timestamps'][:]
        attrs = dict(f.attrs)

    T, H, W = sst_data.shape
    print(f"  Shape: ({T}, {H}, {W})")

    # Pre-filter statistics (valid pixels only)
    valid_data = sst_data[missing_mask == 0]
    before_mean = float(np.nanmean(valid_data))
    before_std = float(np.nanstd(valid_data))
    print(f"  Before: mean={before_mean:.2f}K, std={before_std:.3f}K")

    # Prepare parallel tasks (include fill_mask to distinguish obs vs filled)
    tasks = [(t, sst_data[t], missing_mask[t], fill_mask[t], sigma) for t in range(T)]

    # Parallel filtering
    filtered_sst = np.zeros_like(sst_data)
    print(f"  Filtering {T} frames with {num_workers} workers...", flush=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(gaussian_filter_frame, task): task[0]
                   for task in tasks}

        done = 0
        for future in as_completed(futures):
            frame_idx, filtered_frame = future.result()
            filtered_sst[frame_idx] = filtered_frame
            done += 1
            if done % 2000 == 0:
                elapsed = time.time() - start_time
                rate = done / elapsed
                eta = (T - done) / rate if rate > 0 else 0
                print(f"    {done}/{T} frames ({done/T*100:.0f}%, "
                      f"{rate:.0f} fr/s, ETA {eta:.0f}s)", flush=True)

    # Post-filter statistics
    valid_filtered = filtered_sst[missing_mask == 0]
    after_mean = float(np.nanmean(valid_filtered))
    after_std = float(np.nanstd(valid_filtered))
    print(f"  After:  mean={after_mean:.2f}K, std={after_std:.3f}K")

    # Compute filter effect
    diff = filtered_sst - sst_data
    valid_diff = diff[missing_mask == 0]
    mae = float(np.mean(np.abs(valid_diff)))
    rmse = float(np.sqrt(np.mean(valid_diff ** 2)))
    print(f"  Filter effect: MAE={mae:.4f}K, RMSE={rmse:.4f}K")

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Saving {output_path.name}...", flush=True)

    with h5py.File(str(output_path), 'w') as f:
        f.create_dataset('sst_data', data=filtered_sst.astype(np.float32),
                         compression='gzip', compression_opts=4)
        f.create_dataset('missing_mask', data=missing_mask,
                         compression='gzip', compression_opts=4)
        f.create_dataset('fill_mask', data=fill_mask,
                         compression='gzip', compression_opts=4)
        f.create_dataset('latitude', data=latitude)
        f.create_dataset('longitude', data=longitude)
        f.create_dataset('timestamps', data=timestamps)

        # Copy original attributes
        for key, value in attrs.items():
            f.attrs[key] = value

        # Add filter attributes
        f.attrs['filter_method'] = 'gaussian'
        f.attrs['filter_sigma'] = sigma
        f.attrs['filter_mae'] = mae
        f.attrs['filter_rmse'] = rmse
        f.attrs['filter_date'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    elapsed = time.time() - start_time
    file_size = output_path.stat().st_size / 1024 / 1024

    print(f"  Done! Time: {elapsed:.1f}s, Size: {file_size:.0f}MB")

    return {
        'series_id': series_id,
        'elapsed': elapsed,
        'file_size_mb': file_size,
        'mae': mae,
        'rmse': rmse,
    }


def main():
    parser = argparse.ArgumentParser(
        description='JAXA SST Gaussian Low-Pass Filter')
    parser.add_argument('--series', type=int, nargs='+', default=None,
                        help='Series IDs to process (default: all 0-8)')
    parser.add_argument('--sigma', type=float, default=DEFAULT_SIGMA,
                        help=f'Gaussian sigma (default: {DEFAULT_SIGMA})')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help=f'Number of workers (default: {NUM_WORKERS})')
    args = parser.parse_args()

    series_list = args.series if args.series is not None else list(range(9))

    print("=" * 70)
    print("JAXA SST Gaussian Low-Pass Filter")
    print("=" * 70)
    print(f"  Input:   {INPUT_DIR}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  Sigma:   {args.sigma}")
    print(f"  Workers: {args.workers}")
    print(f"  Series:  {series_list}")

    total_start = time.time()
    results = []

    for sid in series_list:
        result = process_series(sid, sigma=args.sigma, num_workers=args.workers)
        if result is not None:
            results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'Series':<10} {'Time(s)':<10} {'Size(MB)':<10} {'MAE(K)':<10} {'RMSE(K)':<10}")
    print("-" * 50)

    for r in results:
        print(f"  #{r['series_id']:<7} {r['elapsed']:<10.0f} {r['file_size_mb']:<10.0f} "
              f"{r['mae']:<10.4f} {r['rmse']:<10.4f}")

    print(f"\nTotal time: {time.time()-total_start:.0f}s")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
