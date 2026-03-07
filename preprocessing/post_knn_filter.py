#!/usr/bin/env python3
"""
Post-KNN Gaussian Filter — Smooth KNN-filled pixels to reduce residual noise

Processing pipeline position:
    temporal_weighted_fill → lowpass_filter → knn_fill_3d → [THIS SCRIPT]

Problem:
    After 3D KNN filling, there are still visible noise/speckle artifacts,
    especially in regions with low observation rates.

Solution:
    Apply Gaussian low-pass filter (σ=1.5) ONLY to non-observation ocean pixels.
    Real satellite observations are preserved untouched.

Input:  sst_knn_filled/jaxa_knn_filled_XX.h5
Output: sst_knn_filled/jaxa_knn_filled_XX.h5 (in-place update)
        Also regenerates npy cache.

Usage:
    python post_knn_filter.py                    # Process all 9 series
    python post_knn_filter.py --series 0 1 2     # Process specific series
    python post_knn_filter.py --sigma 1.5        # Custom sigma
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

INPUT_DIR = Path('/data/chla_data_imputation_data_260125/sst_knn_filled')
OUTPUT_DIR = Path('/data/chla_data_imputation_data_260125/sst_post_filtered')
NPY_CACHE_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/sst_knn_npy_cache')

DEFAULT_SIGMA = 1.5
NUM_WORKERS = 216


# ============================================================
# Filter Functions
# ============================================================

def filter_frame(args):
    """
    Apply Gaussian filter ONLY to non-observation ocean pixels in one frame.

    Args:
        args: (frame_idx, sst_frame, land_mask, obs_mask_frame, sigma)

    Returns:
        (frame_idx, filtered_frame)
    """
    frame_idx, sst_frame, land_mask, obs_mask_frame, sigma = args

    result = sst_frame.copy()
    ocean = (land_mask == 0)

    # Only filter: ocean pixels that are NOT original observations
    to_filter = ocean & (obs_mask_frame == 0)

    if to_filter.sum() == 0:
        return frame_idx, result

    # Fill land with ocean mean for filter boundary handling
    temp = sst_frame.copy()
    ocean_mean = np.nanmean(sst_frame[ocean])
    temp[land_mask == 1] = ocean_mean

    # Apply Gaussian filter
    filtered = ndimage.gaussian_filter(temp, sigma=sigma, mode='reflect')

    # Only replace non-observation ocean pixels
    result[to_filter] = filtered[to_filter]

    # Land stays NaN
    result[land_mask == 1] = np.nan

    return frame_idx, result


# ============================================================
# Series Processing
# ============================================================

def process_series(series_id, sigma=DEFAULT_SIGMA, num_workers=NUM_WORKERS):
    """Process a single series with post-KNN Gaussian filter."""

    input_path = INPUT_DIR / f'jaxa_knn_filled_{series_id:02d}.h5'
    output_path = OUTPUT_DIR / f'jaxa_knn_filled_{series_id:02d}.h5'

    if not input_path.exists():
        print(f"  Series {series_id}: input not found ({input_path}), skipping")
        return None

    print(f"\n{'='*70}")
    print(f"Series {series_id}: Post-KNN Gaussian filter (sigma={sigma})")
    print(f"{'='*70}")

    start_time = time.time()

    # Load data
    print(f"  Loading {input_path.name}...", flush=True)
    with h5py.File(str(input_path), 'r') as f:
        sst_data = f['sst_data'][:]            # (T, H, W)
        land_mask = f['land_mask'][:]           # (H, W)
        obs_mask = f['original_obs_mask'][:]    # (T, H, W)
        missing_mask = f['original_missing_mask'][:]
        temporal_fill_mask = f['temporal_fill_mask'][:]
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        timestamps = f['timestamps'][:]
        attrs = dict(f.attrs)

    T, H, W = sst_data.shape
    ocean = (land_mask == 0)
    print(f"  Shape: ({T}, {H}, {W}), Ocean: {ocean.sum()} px")

    # Pre-filter gradient stats
    sample_frames = [0, T//4, T//2, 3*T//4, T-1]
    pre_grads = []
    for t in sample_frames:
        temp_g = np.where(ocean, np.nan_to_num(sst_data[t], nan=0), 0)
        gy, gx = np.gradient(temp_g)
        pre_grads.append(np.mean(np.sqrt(gy**2 + gx**2)[ocean]))
    print(f"  Before filter: mean gradient = {np.mean(pre_grads):.4f} K/px")

    # Prepare parallel tasks
    tasks = [(t, sst_data[t], land_mask, obs_mask[t], sigma) for t in range(T)]

    # Parallel filtering
    filtered_sst = np.zeros_like(sst_data)
    print(f"  Filtering {T} frames with {num_workers} workers...", flush=True)

    filter_start = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(filter_frame, task): task[0]
                   for task in tasks}

        done = 0
        for future in as_completed(futures):
            frame_idx, filtered_frame = future.result()
            filtered_sst[frame_idx] = filtered_frame
            done += 1
            if done % 2000 == 0:
                elapsed = time.time() - filter_start
                rate = done / elapsed
                eta = (T - done) / rate if rate > 0 else 0
                print(f"    {done}/{T} frames ({done/T*100:.0f}%, "
                      f"{rate:.0f} fr/s, ETA {eta:.0f}s)", flush=True)

    print(f"  Filtering done in {time.time()-filter_start:.1f}s")

    # Post-filter gradient stats
    post_grads = []
    for t in sample_frames:
        temp_g = np.where(ocean, np.nan_to_num(filtered_sst[t], nan=0), 0)
        gy, gx = np.gradient(temp_g)
        post_grads.append(np.mean(np.sqrt(gy**2 + gx**2)[ocean]))
    print(f"  After filter:  mean gradient = {np.mean(post_grads):.4f} K/px")

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Saving {output_path.name}...", flush=True)

    save_start = time.time()
    with h5py.File(str(output_path), 'w') as f:
        f.create_dataset('sst_data', data=filtered_sst.astype(np.float32),
                         compression='gzip', compression_opts=4)
        f.create_dataset('land_mask', data=land_mask,
                         compression='gzip', compression_opts=4)
        f.create_dataset('original_obs_mask', data=obs_mask,
                         compression='gzip', compression_opts=4)
        f.create_dataset('temporal_fill_mask', data=temporal_fill_mask,
                         compression='gzip', compression_opts=4)
        f.create_dataset('original_missing_mask', data=missing_mask,
                         compression='gzip', compression_opts=4)
        f.create_dataset('latitude', data=latitude)
        f.create_dataset('longitude', data=longitude)
        f.create_dataset('timestamps', data=timestamps)

        # Copy original attributes
        for key, value in attrs.items():
            f.attrs[key] = value

        # Add post-filter attributes
        f.attrs['post_filter_method'] = 'gaussian'
        f.attrs['post_filter_sigma'] = sigma
        f.attrs['post_filter_date'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    elapsed = time.time() - start_time
    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"  Saved in {time.time()-save_start:.1f}s")
    print(f"  Series {series_id} done! Total: {elapsed:.1f}s, Size: {file_size:.0f}MB")

    return {
        'series_id': series_id,
        'elapsed': elapsed,
        'file_size_mb': file_size,
    }


def generate_npy_cache():
    """Regenerate npy cache from post-filtered H5 files."""
    print(f"\n{'='*70}")
    print(f"Regenerating npy cache: {NPY_CACHE_DIR}")
    print(f"{'='*70}")

    NPY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for sid in range(9):
        h5_path = OUTPUT_DIR / f'jaxa_knn_filled_{sid:02d}.h5'
        if not h5_path.exists():
            print(f"  Series {sid}: H5 not found, skipping")
            continue

        sst_npy = NPY_CACHE_DIR / f'sst_{sid:02d}.npy'
        obs_npy = NPY_CACHE_DIR / f'obs_{sid:02d}.npy'
        miss_npy = NPY_CACHE_DIR / f'miss_{sid:02d}.npy'
        land_npy = NPY_CACHE_DIR / f'land_{sid:02d}.npy'

        t0 = time.time()
        with h5py.File(str(h5_path), 'r') as f:
            np.save(str(sst_npy), f['sst_data'][:])
            np.save(str(obs_npy), f['original_obs_mask'][:])
            np.save(str(miss_npy), f['original_missing_mask'][:])
            np.save(str(land_npy), f['land_mask'][:])

        elapsed = time.time() - t0
        size_gb = sum(p.stat().st_size for p in [sst_npy, obs_npy, miss_npy, land_npy]) / 1e9
        print(f"  Series {sid}: {elapsed:.1f}s, {size_gb:.1f}GB")

    print(f"  npy cache regeneration complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Post-KNN Gaussian Filter for SST data')
    parser.add_argument('--series', type=int, nargs='+', default=None,
                        help='Series IDs to process (default: all 0-8)')
    parser.add_argument('--sigma', type=float, default=DEFAULT_SIGMA,
                        help=f'Gaussian sigma (default: {DEFAULT_SIGMA})')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help=f'Number of workers (default: {NUM_WORKERS})')
    parser.add_argument('--no-cache', action='store_true',
                        help='Skip npy cache regeneration')
    args = parser.parse_args()

    series_list = args.series if args.series is not None else list(range(9))

    print("=" * 70)
    print("Post-KNN Gaussian Filter")
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
    for r in results:
        print(f"  Series #{r['series_id']}: {r['elapsed']:.0f}s, {r['file_size_mb']:.0f}MB")

    # Regenerate npy cache
    if not args.no_cache:
        generate_npy_cache()

    print(f"\nAll done! Total time: {time.time()-total_start:.0f}s")


if __name__ == '__main__':
    main()
