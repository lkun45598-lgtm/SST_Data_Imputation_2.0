"""JAXA SST Temporal Weighted Filling Module (Hourly Version)

This module implements temporal weighted filling algorithm for JAXA satellite SST data
to reduce missing values caused by cloud occlusion.

Key change from previous version:
    - INTERVAL changed from 24 to 1: now processes EVERY HOUR instead of daily
    - Vectorized filling (numpy) replaces per-pixel multiprocessing
    - Sliding window cache avoids redundant file I/O
    - 9 series processed in parallel using multiprocessing (216 cores)

Algorithm:
    For each missing pixel at target time t, fill using weighted average of historical
    observations within lookback window:

    weight(t_history) = 1 / (t - t_history)
    filled_value = sum(w_i * v_i) / sum(w_i)

Typical usage:

    # Generate all 9 series (hourly, 216 cores)
    python temporal_weighted_fill.py --mode full --workers 216

    # Generate single series
    python temporal_weighted_fill.py --mode single --series 0

    # Quick test (48 hours of series 0)
    python temporal_weighted_fill.py --mode test

Author: Claude Code
Date: 2026-02-17
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm


# ============================================================
# Configuration Constants
# ============================================================

JAXA_ROOT = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')
OUTPUT_ROOT = Path('/data/chla_data_imputation_data_260125/sst_temperal_data')

INTERVAL = 1           # 1 hour interval (hourly processing)
LOOKBACK_WINDOW = 48   # Maximum lookback window in hours
DEFAULT_WORKERS = 216  # Default number of CPU cores

# 9 yearly series configurations
# (series_id, start_date, end_date, start_year_label)
YEAR_CONFIGS = [
    (0, datetime(2017, 7, 6, 0), datetime(2018, 7, 5, 23), 2),
    (1, datetime(2016, 7, 6, 0), datetime(2017, 7, 5, 23), 1),
    (2, datetime(2021, 7, 5, 0), datetime(2022, 7, 4, 23), 6),
    (3, datetime(2018, 7, 6, 0), datetime(2019, 7, 4, 23), 3),
    (4, datetime(2019, 7, 7, 0), datetime(2020, 7, 4, 23), 4),
    (5, datetime(2020, 7, 5, 0), datetime(2021, 7, 4, 23), 5),
    (6, datetime(2022, 7, 5, 0), datetime(2023, 7, 4, 23), 7),
    (7, datetime(2023, 7, 5, 0), datetime(2024, 7, 3, 23), 8),
    (8, datetime(2024, 7, 4, 0), datetime(2025, 3, 30, 23), 9),
]

NUM_SERIES = len(YEAR_CONFIGS)


# ============================================================
# Core Data Loading
# ============================================================

def load_jaxa_frame(target_time: datetime) -> Tuple[Optional[np.ndarray],
                                                      Optional[np.ndarray],
                                                      Optional[np.ndarray]]:
    """Load JAXA SST data for specified time."""
    date_str = target_time.strftime('%Y%m')
    day_str = target_time.strftime('%d')
    file_str = target_time.strftime('%Y%m%d%H%M%S')
    file_path = JAXA_ROOT / date_str / day_str / f'{file_str}.nc'

    if not file_path.exists():
        return None, None, None

    try:
        ds = xr.open_dataset(file_path)
        sst = ds.sea_surface_temperature.values
        if len(sst.shape) == 3:
            sst = sst[0]
        lat = ds.lat.values
        lon = ds.lon.values
        ds.close()
        return sst, lat, lon
    except Exception as e:
        return None, None, None


# ============================================================
# Vectorized Temporal Weighted Filling
# ============================================================

def fill_frame_vectorized(target_sst: np.ndarray,
                          target_hour_offset: int,
                          history_cache: Dict[int, np.ndarray]
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized temporal weighted filling for one frame.

    For each missing pixel, computes inverse-time-distance weighted average
    of valid observations from history_cache.
    """
    missing_mask = np.isnan(target_sst)

    if not missing_mask.any():
        return target_sst.copy(), np.zeros_like(target_sst, dtype=np.uint8)

    filled = target_sst.copy()
    weight_sum = np.zeros_like(target_sst, dtype=np.float64)
    value_sum = np.zeros_like(target_sst, dtype=np.float64)
    count = np.zeros_like(target_sst, dtype=np.int32)

    for t, t_sst in history_cache.items():
        time_distance = target_hour_offset - t
        if time_distance <= 0:
            continue

        valid = (~np.isnan(t_sst)) & missing_mask
        if not valid.any():
            continue

        weight = 1.0 / time_distance
        weight_sum[valid] += weight
        value_sum[valid] += weight * t_sst[valid]
        count[valid] += 1

    fillable = weight_sum > 0
    filled[fillable] = (value_sum[fillable] / weight_sum[fillable]).astype(np.float32)
    fill_info = np.clip(count, 0, 255).astype(np.uint8)

    return filled, fill_info


# ============================================================
# Series Generation (Hourly) - Single Series Worker
# ============================================================

def generate_time_series(series_id: int, num_threads: int = 24) -> Dict:
    """Generate one yearly time series with hourly temporal weighted filling.

    Args:
        series_id: Series index (0-8).
        num_threads: Number of numpy threads per process.

    Returns:
        Dictionary with statistics.
    """
    # Limit numpy threads for this process
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)

    config = YEAR_CONFIGS[series_id]
    sid, start_date, end_date, start_year = config

    total_hours = int((end_date - start_date).total_seconds() / 3600) + 1

    print(f"[Series #{sid:02d}] {start_date.strftime('%Y-%m-%d')} -> "
          f"{end_date.strftime('%Y-%m-%d')} | {total_hours} hours | "
          f"{num_threads} threads", flush=True)

    # Initialize storage
    sst_frames = []
    missing_masks = []
    fill_masks = []
    timestamps = []
    original_missing_rates = []
    filled_missing_rates = []

    lat = None
    lon = None
    cache = {}
    skipped = 0

    for hour_offset in range(total_hours):
        target_time = start_date + timedelta(hours=hour_offset)
        target_sst, frame_lat, frame_lon = load_jaxa_frame(target_time)

        if target_sst is None:
            skipped += 1
            continue

        if lat is None:
            lat = frame_lat
            lon = frame_lon

        # Fill missing pixels
        if hour_offset == 0 or len(cache) == 0:
            filled_sst = target_sst.copy()
            fill_info = np.zeros_like(target_sst, dtype=np.uint8)
        else:
            filled_sst, fill_info = fill_frame_vectorized(
                target_sst, hour_offset, cache
            )

        # Cache: store raw observation
        cache[hour_offset] = target_sst

        # Evict old entries
        evict_threshold = hour_offset - LOOKBACK_WINDOW
        for h in list(cache.keys()):
            if h < evict_threshold:
                del cache[h]

        # Statistics
        original_rate = np.isnan(target_sst).sum() / target_sst.size * 100
        filled_rate = np.isnan(filled_sst).sum() / filled_sst.size * 100
        original_missing_rates.append(original_rate)
        filled_missing_rates.append(filled_rate)

        # Store frame
        sst_frames.append(filled_sst)
        missing_masks.append(np.isnan(filled_sst).astype(np.uint8))
        fill_masks.append((fill_info > 0).astype(np.uint8))
        timestamps.append(target_time.isoformat())

        # Progress every 1000 hours
        if (hour_offset + 1) % 1000 == 0:
            print(f"  [Series #{sid:02d}] {hour_offset+1}/{total_hours} hours done "
                  f"({(hour_offset+1)/total_hours*100:.1f}%)", flush=True)

    if len(sst_frames) == 0:
        print(f"  [Series #{sid:02d}] No valid frames!", flush=True)
        return {'series_id': sid, 'num_frames': 0}

    # Convert to numpy arrays
    sst_data = np.array(sst_frames, dtype=np.float32)
    missing_mask_data = np.array(missing_masks, dtype=np.uint8)
    fill_mask_data = np.array(fill_masks, dtype=np.uint8)
    del sst_frames, missing_masks, fill_masks

    # Save to HDF5
    OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
    output_file = OUTPUT_ROOT / f'jaxa_weighted_series_{sid:02d}.h5'

    print(f"  [Series #{sid:02d}] Saving {output_file.name} "
          f"({sst_data.shape[0]} frames) ...", flush=True)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('sst_data', data=sst_data,
                         compression='gzip', compression_opts=4)
        f.create_dataset('missing_mask', data=missing_mask_data,
                         compression='gzip', compression_opts=4)
        f.create_dataset('fill_mask', data=fill_mask_data,
                         compression='gzip', compression_opts=4)
        f.create_dataset('latitude', data=lat, dtype=np.float32)
        f.create_dataset('longitude', data=lon, dtype=np.float32)
        f.create_dataset('timestamps', data=np.array(timestamps, dtype='S32'))

        f.attrs['series_id'] = sid
        f.attrs['start_year'] = start_year
        f.attrs['num_frames'] = len(sst_data)
        f.attrs['interval'] = INTERVAL
        f.attrs['shape'] = sst_data.shape
        f.attrs['actual_start_date'] = timestamps[0]
        f.attrs['actual_end_date'] = timestamps[-1]
        f.attrs['lookback_window'] = LOOKBACK_WINDOW
        f.attrs['skipped_hours'] = skipped
        f.attrs['creation_date'] = datetime.now().isoformat()

    file_size_mb = output_file.stat().st_size / 1024 / 1024

    stats = {
        'series_id': sid,
        'start_year': start_year,
        'num_frames': len(sst_data),
        'skipped_hours': skipped,
        'original_missing_rate_avg': float(np.mean(original_missing_rates)),
        'filled_missing_rate_avg': float(np.mean(filled_missing_rates)),
        'improvement_avg': float(np.mean(
            np.array(original_missing_rates) - np.array(filled_missing_rates)
        )),
        'file_path': str(output_file),
        'file_size_mb': file_size_mb
    }

    print(f"  [Series #{sid:02d}] DONE: {stats['num_frames']} frames, "
          f"missing {stats['original_missing_rate_avg']:.1f}% -> "
          f"{stats['filled_missing_rate_avg']:.1f}%, "
          f"size {file_size_mb:.0f} MB", flush=True)

    return stats


def _worker_wrapper(args):
    """Wrapper for multiprocessing."""
    series_id, num_threads = args
    return generate_time_series(series_id, num_threads)


# ============================================================
# Full Dataset Generation (Parallel)
# ============================================================

def generate_full_dataset(series_list: Optional[List[int]] = None,
                          num_workers: int = DEFAULT_WORKERS):
    """Generate JAXA hourly weighted dataset.

    Processes multiple series in parallel using multiprocessing.
    Each series gets num_workers/num_series threads for numpy operations.

    Args:
        series_list: Series indices to process. None = all 9.
        num_workers: Total CPU cores available.
    """
    if series_list is None:
        series_list = list(range(NUM_SERIES))

    num_parallel = len(series_list)
    threads_per_series = max(1, num_workers // num_parallel)

    print("=" * 70)
    print("JAXA Hourly Dataset Generation - Temporal Weighted Filling")
    print("=" * 70)
    print(f"  Data source:  {JAXA_ROOT}")
    print(f"  Output dir:   {OUTPUT_ROOT}")
    print(f"  Interval:     {INTERVAL}h (hourly)")
    print(f"  Lookback:     {LOOKBACK_WINDOW}h")
    print(f"  Series:       {series_list} ({num_parallel} series)")
    print(f"  Total cores:  {num_workers}")
    print(f"  Parallel:     {num_parallel} series x {threads_per_series} threads each")
    print(flush=True)

    # Process series in parallel
    tasks = [(idx, threads_per_series) for idx in series_list]

    all_stats = []

    if num_parallel == 1:
        # Single series: run directly (better progress output)
        stats = generate_time_series(series_list[0], threads_per_series)
        all_stats.append(stats)
    else:
        # Multiple series: parallel processing
        with ProcessPoolExecutor(max_workers=num_parallel) as executor:
            future_to_sid = {}
            for idx, nthreads in tasks:
                future = executor.submit(_worker_wrapper, (idx, nthreads))
                future_to_sid[future] = idx

            for future in as_completed(future_to_sid):
                sid = future_to_sid[future]
                try:
                    stats = future.result()
                    all_stats.append(stats)
                    print(f"\n>>> Series #{sid:02d} completed! <<<\n", flush=True)
                except Exception as e:
                    print(f"\n>>> Series #{sid:02d} FAILED: {e} <<<\n", flush=True)
                    import traceback
                    traceback.print_exc()

    # Sort by series_id
    all_stats.sort(key=lambda s: s['series_id'])

    # Save statistics
    stats_file = OUTPUT_ROOT / 'dataset_statistics_hourly.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'num_series': len(all_stats),
            'total_frames': sum(s['num_frames'] for s in all_stats),
            'series': all_stats,
            'creation_date': datetime.now().isoformat(),
            'parameters': {
                'interval': INTERVAL,
                'lookback_window': LOOKBACK_WINDOW,
                'num_workers': num_workers,
                'year_configs': [
                    {
                        'series_id': c[0],
                        'start': c[1].isoformat(),
                        'end': c[2].isoformat(),
                        'start_year': c[3]
                    }
                    for c in YEAR_CONFIGS
                ]
            }
        }, f, indent=2)

    print(f"\nStatistics saved: {stats_file}")

    # Plot
    if len(all_stats) > 1:
        plot_overall_statistics(all_stats)

    # Summary
    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)
    total_frames = sum(s['num_frames'] for s in all_stats)
    total_size = sum(s['file_size_mb'] for s in all_stats)
    print(f"  Total series: {len(all_stats)}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Total size:   {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    if all_stats:
        avg_imp = np.mean([s['improvement_avg'] for s in all_stats])
        print(f"  Avg improvement: {avg_imp:.2f}%")
    print()


# ============================================================
# Visualization
# ============================================================

def plot_overall_statistics(all_stats: List[Dict]):
    """Plot overall statistics for all time series."""
    print("\nGenerating statistics plot...")

    series_ids = [s['series_id'] for s in all_stats]
    original_rates = [s['original_missing_rate_avg'] for s in all_stats]
    filled_rates = [s['filled_missing_rate_avg'] for s in all_stats]
    improvements = [s['improvement_avg'] for s in all_stats]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Missing rate comparison
    ax = axes[0, 0]
    x = np.arange(len(series_ids))
    width = 0.35
    ax.bar(x - width/2, original_rates, width, label='Original', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, filled_rates, width, label='After Filling', color='#27ae60', alpha=0.8)
    ax.set_xlabel('Series ID', fontsize=12)
    ax.set_ylabel('Missing Rate (%)', fontsize=12)
    ax.set_title('Missing Rate by Series (Hourly)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'#{i}' for i in series_ids])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Improvement
    ax = axes[0, 1]
    ax.bar(x, improvements, color='#3498db', alpha=0.8)
    ax.set_xlabel('Series ID', fontsize=12)
    ax.set_ylabel('Missing Rate Reduction (%)', fontsize=12)
    ax.set_title('Improvement by Series', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'#{i}' for i in series_ids])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=np.mean(improvements), color='red', linestyle='--',
               label=f'Avg: {np.mean(improvements):.2f}%')
    ax.legend()

    # 3. Frame counts & file size
    ax = axes[1, 0]
    frame_counts = [s['num_frames'] for s in all_stats]
    file_sizes = [s['file_size_mb'] for s in all_stats]
    ax2 = ax.twinx()
    ax.bar(x - 0.2, frame_counts, 0.4, label='Frames', color='#2ecc71', alpha=0.8)
    ax2.bar(x + 0.2, file_sizes, 0.4, label='File Size (MB)', color='#9b59b6', alpha=0.8)
    ax.set_xlabel('Series ID', fontsize=12)
    ax.set_ylabel('Frame Count', fontsize=12, color='#2ecc71')
    ax2.set_ylabel('File Size (MB)', fontsize=12, color='#9b59b6')
    ax.set_title('Frames & File Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'#{i}' for i in series_ids])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 4. Summary
    ax = axes[1, 1]
    ax.axis('off')
    total_frames = sum(s['num_frames'] for s in all_stats)
    total_size = sum(file_sizes)
    summary = f"""
JAXA Hourly Weighted Dataset Summary
{'='*50}

Total Series:   {len(all_stats)}
Total Frames:   {total_frames:,}
Interval:       {INTERVAL}h (hourly)
Lookback:       {LOOKBACK_WINDOW}h

Missing Rate:
  Original:  {np.mean(original_rates):.2f}%
  Filled:    {np.mean(filled_rates):.2f}%
  Improve:   {np.mean(improvements):.2f}%

File Size:
  Total:     {total_size:.1f} MB ({total_size/1024:.2f} GB)
  Avg/Series: {np.mean(file_sizes):.1f} MB
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    output_path = OUTPUT_ROOT / 'dataset_statistics_hourly.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Statistics plot saved: {output_path}")


# ============================================================
# Test Mode
# ============================================================

def run_test():
    """Quick test: process first 48 hours of series 0."""
    print("=" * 70)
    print("JAXA Temporal Weighted Fill - TEST MODE (48 hours)")
    print("=" * 70)

    config = YEAR_CONFIGS[0]
    sid, start_date, end_date, start_year = config
    test_hours = 48

    print(f"  Series: #{sid}")
    print(f"  Start: {start_date}")
    print(f"  Test hours: {test_hours}")
    print()

    cache = {}

    for hour_offset in range(test_hours):
        target_time = start_date + timedelta(hours=hour_offset)
        target_sst, _, _ = load_jaxa_frame(target_time)

        if target_sst is None:
            print(f"  Hour {hour_offset:3d} ({target_time.strftime('%Y-%m-%d %H:%M')}): "
                  f"FILE NOT FOUND")
            continue

        original_missing = np.isnan(target_sst).sum() / target_sst.size * 100

        if hour_offset == 0 or len(cache) == 0:
            filled_sst = target_sst.copy()
            filled_missing = np.isnan(filled_sst).sum() / filled_sst.size * 100
            improvement = 0.0
        else:
            filled_sst, _ = fill_frame_vectorized(target_sst, hour_offset, cache)
            filled_missing = np.isnan(filled_sst).sum() / filled_sst.size * 100
            improvement = original_missing - filled_missing

        cache[hour_offset] = target_sst
        for h in list(cache.keys()):
            if h < hour_offset - LOOKBACK_WINDOW:
                del cache[h]

        print(f"  Hour {hour_offset:3d} ({target_time.strftime('%Y-%m-%d %H:%M')}): "
              f"missing {original_missing:.1f}% -> {filled_missing:.1f}% "
              f"(+{improvement:.1f}%, cache={len(cache)})")

    print(f"\nTest complete!")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='JAXA SST Temporal Weighted Filling (Hourly)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all 9 series (216 cores, 9 parallel)
  python temporal_weighted_fill.py --mode full --workers 216

  # Generate single series
  python temporal_weighted_fill.py --mode single --series 0 --workers 216

  # Quick test (48 hours)
  python temporal_weighted_fill.py --mode test
"""
    )

    parser.add_argument('--mode', type=str, choices=['full', 'test', 'single'],
                        default='full',
                        help='full: all 9 series, test: 48h test, single: specific series')
    parser.add_argument('--series', type=int, nargs='+', default=[0],
                        help='Series ID(s) for single mode (0-8)')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Total CPU cores (default: {DEFAULT_WORKERS})')

    args = parser.parse_args()

    if args.mode == 'full':
        generate_full_dataset(num_workers=args.workers)
    elif args.mode == 'test':
        run_test()
    elif args.mode == 'single':
        for s in args.series:
            if not (0 <= s < NUM_SERIES):
                print(f"Error: Series ID must be 0-{NUM_SERIES-1}, got {s}")
                sys.exit(1)
        generate_full_dataset(series_list=args.series, num_workers=args.workers)


if __name__ == '__main__':
    main()
