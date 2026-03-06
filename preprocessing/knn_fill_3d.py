#!/usr/bin/env python3
"""
3D Spatiotemporal KNN Fill for Hourly SST Data — Progressive Band Strategy

Replaces batch 3D KNN with progressive density-banded 3D spatiotemporal KNN.
For each frame t, considers spatial + temporal neighbors within +/-60 hours.

Algorithm (Progressive):
  For each frame:
    1. Compute 2D missing density for each NaN ocean pixel
    2. Sort by density, split into NUM_BANDS bands (low density = edge first)
    3. Build base 3D cKDTree from all known pixels in [t-60h, t+60h]
    4. Fill band-by-band: each band queries base tree + supplementary tree
       (built from previously filled bands), merges K nearest, IDW interpolation
  Pass 2: Re-run on remaining NaN using Pass 1 results
  Fallback: Global mean for any residual NaN

Parallelism: Linux fork (COW sharing), 216 workers
"""

import numpy as np
from scipy.spatial import cKDTree
import h5py
import time
from pathlib import Path
import multiprocessing as mp

# ============================================================================
# Parameters (aligned with daily pipeline: time_scale=5.0, k=30, window=30days)
# ============================================================================
HALF_WINDOW = 60           # +/-60 hours (total 120h = 5 days)
TIME_SCALE = 5.0 / 24     # ~0.2083: 1 hour = 0.208 spatial pixels
K = 30
POWER = 2
NUM_WORKERS = 216
NUM_BANDS = 16             # density bands for progressive filling
DENSITY_RADIUS = 20        # radius for 2D missing density computation

# Paths
INPUT_DIR = Path('/data/chla_data_imputation_data_260125/sst_filtered')
OUTPUT_DIR = Path('/data/chla_data_imputation_data_260125/sst_knn_filled')
NPY_CACHE_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/sst_knn_npy_cache')

# ============================================================================
# Module-level globals for COW sharing via fork
# ============================================================================
_g_sst = None       # (T, H, W) float32
_g_ocean = None      # (H, W) bool


def _progressive_fill_frame(t):
    """
    Worker function: fill NaN ocean pixels in frame t using progressive
    density-banded 3D spatiotemporal KNN.

    1. Compute 2D missing density for each NaN pixel
    2. Split into NUM_BANDS bands by density percentile (low=edge first)
    3. Build base 3D KDTree from known pixels in temporal window
    4. Fill band-by-band: base tree + supplementary tree from prior bands

    Reads from module-level globals (_g_sst, _g_ocean) shared via fork COW.

    Returns:
        (frame_index, filled_frame, num_filled)
    """
    T, H, W = _g_sst.shape
    frame = _g_sst[t].copy()

    # Find NaN ocean pixels in this frame
    nan_ocean = np.isnan(frame) & _g_ocean
    my, mx = np.where(nan_ocean)
    n_miss = len(my)

    if n_miss == 0:
        return t, frame, 0

    # Temporal window bounds
    t0 = max(0, t - HALF_WINDOW)
    t1 = min(T, t + HALF_WINDOW + 1)

    # Extract window (view, no copy)
    window = _g_sst[t0:t1]

    # --- Step 1: Compute 2D missing density ---
    miss_coords_2d = np.column_stack([my.astype(np.float64),
                                      mx.astype(np.float64)])
    miss_tree_2d = cKDTree(miss_coords_2d)
    neighbor_counts = miss_tree_2d.query_ball_point(miss_coords_2d,
                                                     r=DENSITY_RADIUS,
                                                     return_length=True)
    density = np.asarray(neighbor_counts, dtype=np.float64)

    # --- Step 2: Sort by density, split into bands ---
    sort_idx = np.argsort(density)  # low density (edge) first
    band_size = max(1, n_miss // NUM_BANDS)
    bands = []
    for b in range(NUM_BANDS):
        start = b * band_size
        end = start + band_size if b < NUM_BANDS - 1 else n_miss
        band_indices = sort_idx[start:end]
        if len(band_indices) > 0:
            bands.append(band_indices)

    # --- Step 3: Build base 3D KDTree (all known pixels in window) ---
    known_3d = (~np.isnan(window)) & _g_ocean[np.newaxis, :, :]
    dt_idx, ky, kx = np.where(known_3d)
    del known_3d

    if len(dt_idx) == 0:
        return t, frame, 0

    time_offsets = (dt_idx + t0 - t) * TIME_SCALE
    base_coords = np.column_stack([
        time_offsets.astype(np.float64),
        ky.astype(np.float64),
        kx.astype(np.float64)
    ])
    base_vals = window[dt_idx, ky, kx].astype(np.float64)
    del dt_idx, ky, kx, time_offsets

    base_tree = cKDTree(base_coords)
    del base_coords  # tree stores its own copy

    # --- Step 4: Progressive band-by-band filling ---
    eps = 1e-10
    # Pre-allocate supplementary pool arrays
    sup_coords_list = np.empty((n_miss, 3), dtype=np.float64)
    sup_vals_list = np.empty(n_miss, dtype=np.float64)
    sup_count = 0

    for band_idx in bands:
        band_my = my[band_idx]
        band_mx = mx[band_idx]
        n_band = len(band_idx)

        # Query points for this band (time_offset = 0 for current frame)
        q = np.column_stack([
            np.zeros(n_band, dtype=np.float64),
            band_my.astype(np.float64),
            band_mx.astype(np.float64)
        ])

        k_base = min(K, len(base_vals))
        base_dists, base_idxs = base_tree.query(q, k=k_base)
        if k_base == 1:
            base_dists = base_dists.reshape(-1, 1)
            base_idxs = base_idxs.reshape(-1, 1)

        if sup_count > 0:
            # Build supplementary tree from previously filled bands
            sup_tree = cKDTree(sup_coords_list[:sup_count])
            k_sup = min(K, sup_count)
            sup_dists, sup_idxs = sup_tree.query(q, k=k_sup)
            if k_sup == 1:
                sup_dists = sup_dists.reshape(-1, 1)
                sup_idxs = sup_idxs.reshape(-1, 1)

            # Merge: for each query point, take K nearest from both sets
            all_filled = np.empty(n_band, dtype=np.float64)
            for i in range(n_band):
                # Combine distances and values from both trees
                d_b = base_dists[i]
                v_b = base_vals[base_idxs[i]]
                d_s = sup_dists[i]
                v_s = sup_vals_list[sup_idxs[i]]

                d_all = np.concatenate([d_b, d_s])
                v_all = np.concatenate([v_b, v_s])

                # Take K nearest
                topk = np.argsort(d_all)[:K]
                d_k = d_all[topk]
                v_k = v_all[topk]

                w = 1.0 / (d_k ** POWER + eps)
                all_filled[i] = np.sum(w * v_k) / np.sum(w)
        else:
            # First band: only base tree
            w = 1.0 / (base_dists ** POWER + eps)
            nv = base_vals[base_idxs]
            all_filled = np.sum(w * nv, axis=1) / np.sum(w, axis=1)

        # Write filled values to frame
        frame[band_my, band_mx] = all_filled.astype(np.float32)

        # Add to supplementary pool (time_offset=0 for current frame)
        sup_coords_list[sup_count:sup_count + n_band, 0] = 0.0
        sup_coords_list[sup_count:sup_count + n_band, 1] = band_my.astype(np.float64)
        sup_coords_list[sup_count:sup_count + n_band, 2] = band_mx.astype(np.float64)
        sup_vals_list[sup_count:sup_count + n_band] = all_filled
        sup_count += n_band

    return t, frame, n_miss


def process_series(series_id):
    """Process a single JAXA series with 3D spatiotemporal KNN filling."""
    global _g_sst, _g_ocean

    input_path = INPUT_DIR / f'jaxa_filtered_{series_id:02d}.h5'
    output_path = OUTPUT_DIR / f'jaxa_knn_filled_{series_id:02d}.h5'

    print(f"\n{'='*70}")
    print(f"Series {series_id}: {input_path.name}")
    print(f"{'='*70}")

    series_start = time.time()

    # Load data
    print(f"  Loading H5...", flush=True)
    load_start = time.time()
    with h5py.File(str(input_path), 'r') as f:
        sst_data = f['sst_data'][:]         # (T, H, W) float32
        missing_mask = f['missing_mask'][:]  # (T, H, W) uint8
        fill_mask = f['fill_mask'][:]        # (T, H, W) uint8
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        timestamps = f['timestamps'][:]
        series_id_attr = int(f.attrs.get('series_id', series_id))
        start_year = int(f.attrs.get('start_year', 0))

    T, H, W = sst_data.shape
    land_mask = np.all(np.isnan(sst_data), axis=0).astype(np.uint8)
    ocean_mask = (land_mask == 0)

    initial_nan = int(np.sum(np.isnan(sst_data) & ocean_mask[np.newaxis, :, :]))
    print(f"  Loaded in {time.time()-load_start:.1f}s")
    print(f"  Shape: ({T}, {H}, {W}), Land: {land_mask.sum()} px ({land_mask.sum()/(H*W)*100:.1f}%)")
    print(f"  Ocean NaN before KNN: {initial_nan:,}")

    # Set globals for COW sharing
    _g_sst = sst_data
    _g_ocean = ocean_mask

    # ==== Pass 1 ====
    print(f"\n  Pass 1: 3D progressive KNN ({NUM_BANDS} bands, {NUM_WORKERS} workers)...", flush=True)
    p1_start = time.time()

    ctx = mp.get_context('fork')
    with ctx.Pool(NUM_WORKERS) as pool:
        results = []
        for r in pool.imap_unordered(_progressive_fill_frame, range(T), chunksize=8):
            results.append(r)
            if len(results) % 1000 == 0:
                elapsed = time.time() - p1_start
                rate = len(results) / elapsed
                eta = (T - len(results)) / rate if rate > 0 else 0
                print(f"    Pass 1: {len(results)}/{T} frames "
                      f"({len(results)/T*100:.0f}%, {rate:.1f} fr/s, ETA {eta:.0f}s)",
                      flush=True)

    total_filled_p1 = 0
    for idx, frame, count in results:
        sst_data[idx] = frame
        total_filled_p1 += count

    remaining_nan = int(np.sum(np.isnan(sst_data) & ocean_mask[np.newaxis, :, :]))
    print(f"  Pass 1 done: filled {total_filled_p1:,} px, "
          f"remaining NaN: {remaining_nan:,}, time: {time.time()-p1_start:.1f}s")

    # ==== Pass 2 ====
    total_filled_p2 = 0
    if remaining_nan > 0:
        print(f"\n  Pass 2: filling remaining NaN...", flush=True)
        _g_sst = sst_data  # updated with Pass 1 results

        frames_with_nan = [t for t in range(T)
                           if np.any(np.isnan(sst_data[t]) & ocean_mask)]
        print(f"    {len(frames_with_nan)} frames still have NaN")

        p2_start = time.time()
        with ctx.Pool(NUM_WORKERS) as pool:
            results2 = list(pool.imap_unordered(_progressive_fill_frame, frames_with_nan, chunksize=4))

        for idx, frame, count in results2:
            sst_data[idx] = frame
            total_filled_p2 += count

        remaining_nan = int(np.sum(np.isnan(sst_data) & ocean_mask[np.newaxis, :, :]))
        print(f"  Pass 2 done: filled {total_filled_p2:,} px, "
              f"remaining NaN: {remaining_nan:,}, time: {time.time()-p2_start:.1f}s")

    # ==== Fallback: global mean ====
    if remaining_nan > 0:
        print(f"  Fallback: filling {remaining_nan:,} residual NaN with global mean...")
        global_mean = np.nanmean(sst_data)
        nan_ocean_3d = np.isnan(sst_data) & ocean_mask[np.newaxis, :, :]
        sst_data[nan_ocean_3d] = global_mean

    # Land stays NaN
    sst_data[:, land_mask == 1] = np.nan

    # Verify: no NaN in ocean
    final_ocean_nan = int(np.sum(np.isnan(sst_data) & ocean_mask[np.newaxis, :, :]))
    assert final_ocean_nan == 0, f"Bug: {final_ocean_nan} ocean NaN remain after all passes!"

    # Compute original_obs_mask
    original_obs_mask = ((fill_mask == 0) & (missing_mask == 0)).astype(np.uint8)
    original_missing_rate = float(np.sum(missing_mask)) / (T * H * W) * 100

    # ==== Save output H5 ====
    print(f"\n  Saving: {output_path}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_start = time.time()
    with h5py.File(str(output_path), 'w') as f:
        f.create_dataset('sst_data', data=sst_data.astype(np.float32),
                         compression='gzip', compression_opts=4)
        f.create_dataset('land_mask', data=land_mask,
                         compression='gzip', compression_opts=4)
        f.create_dataset('original_obs_mask', data=original_obs_mask,
                         compression='gzip', compression_opts=4)
        f.create_dataset('temporal_fill_mask', data=fill_mask,
                         compression='gzip', compression_opts=4)
        f.create_dataset('original_missing_mask', data=missing_mask,
                         compression='gzip', compression_opts=4)
        f.create_dataset('latitude', data=latitude)
        f.create_dataset('longitude', data=longitude)
        f.create_dataset('timestamps', data=timestamps)

        f.attrs['series_id'] = series_id_attr
        f.attrs['start_year'] = start_year
        f.attrs['num_frames'] = T
        f.attrs['knn_k'] = K
        f.attrs['knn_time_scale'] = float(TIME_SCALE)
        f.attrs['knn_temporal_window'] = HALF_WINDOW * 2
        f.attrs['knn_power'] = POWER
        f.attrs['knn_method'] = '3d_progressive'
        f.attrs['knn_num_bands'] = NUM_BANDS
        f.attrs['knn_density_radius'] = DENSITY_RADIUS
        f.attrs['original_missing_rate'] = original_missing_rate
        f.attrs['pixels_filled_by_knn'] = int(total_filled_p1 + total_filled_p2)
        f.attrs['land_pixel_count'] = int(land_mask.sum())
        f.attrs['creation_date'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    elapsed = time.time() - series_start
    file_size = output_path.stat().st_size / 1024 / 1024

    print(f"  Saved in {time.time()-save_start:.1f}s")
    print(f"  Series {series_id} done! Total time: {elapsed:.1f}s, Size: {file_size:.0f}MB")
    print(f"  KNN filled: {total_filled_p1+total_filled_p2:,} px "
          f"(P1: {total_filled_p1:,}, P2: {total_filled_p2:,})")

    return {
        'series_id': series_id,
        'elapsed': elapsed,
        'file_size_mb': file_size,
        'filled_p1': total_filled_p1,
        'filled_p2': total_filled_p2,
    }


def generate_npy_cache():
    """Convert output H5 files to npy cache for training."""
    print(f"\n{'='*70}")
    print(f"Generating npy cache: {NPY_CACHE_DIR}")
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

    print(f"  npy cache generation complete!")


def main():
    print("=" * 70)
    print("3D Progressive Density-Banded KNN Fill for Hourly SST Data")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Temporal window: +/-{HALF_WINDOW}h ({HALF_WINDOW*2}h total, = {HALF_WINDOW*2/24:.0f} days)")
    print(f"  Time scale: {TIME_SCALE:.4f} (1h ~ {TIME_SCALE:.3f} spatial px)")
    print(f"  K: {K}, Power: {POWER}")
    print(f"  Progressive bands: {NUM_BANDS}, Density radius: {DENSITY_RADIUS}")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"\nInput:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Cache:  {NPY_CACHE_DIR}")

    total_start = time.time()
    results = []

    for sid in range(9):
        input_path = INPUT_DIR / f'jaxa_weighted_series_{sid:02d}.h5'
        if input_path.exists():
            result = process_series(sid)
            results.append(result)
        else:
            print(f"\nWarning: {input_path} not found, skipping")

    # Summary
    print(f"\n{'='*70}")
    print(f"3D KNN Fill Summary")
    print(f"{'='*70}")
    print(f"{'Series':<10} {'Time(s)':<10} {'Size(MB)':<10} {'Filled(P1)':<15} {'Filled(P2)':<15}")
    print("-" * 60)
    for r in results:
        print(f"  #{r['series_id']:<7} {r['elapsed']:<10.0f} {r['file_size_mb']:<10.0f} "
              f"{r['filled_p1']:<15,} {r['filled_p2']:<15,}")
    print(f"\n  Total KNN fill time: {time.time()-total_start:.0f}s")

    # Generate npy cache
    generate_npy_cache()

    print(f"\nAll done! Total time: {time.time()-total_start:.0f}s")


if __name__ == '__main__':
    main()
