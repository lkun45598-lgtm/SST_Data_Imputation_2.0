#!/usr/bin/env python3
"""
Post-filter for KNN filled data
Only applies Gaussian filter to filled regions, preserving original observations
"""
import numpy as np
import h5py
from scipy import ndimage
from pathlib import Path
import time

INPUT_DIR = Path('/data/chla_data_imputation_data_260125/sst_knn_filled_optimized')
OUTPUT_DIR = Path('/data/chla_data_imputation_data_260125/sst_post_filtered')
OUTPUT_DIR.mkdir(exist_ok=True)

SIGMA = 1.5

for i in range(9):
    input_file = INPUT_DIR / f'jaxa_knn_filled_0{i}.h5'
    output_file = OUTPUT_DIR / f'jaxa_filtered_0{i}.h5'

    print(f'\n{"="*70}')
    print(f'Processing Series {i}: {input_file.name}')
    print(f'{"="*70}')

    start_time = time.time()

    with h5py.File(input_file, 'r') as f:
        sst_data = f['sst_data'][:]
        obs_mask = f['original_obs_mask'][:]
        land_mask = f['land_mask'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        timestamps = f['timestamps'][:]

    T = len(sst_data)
    print(f'  Loaded {T} frames')

    ocean = (land_mask == 0)
    filtered_data = sst_data.copy()

    for t in range(T):
        frame = sst_data[t].copy()
        obs = (obs_mask[t] == 1) & ocean
        filled = (obs_mask[t] == 0) & ocean

        if filled.sum() > 0:
            temp = frame.copy()
            temp[~ocean] = np.nanmean(frame[ocean])
            smoothed = ndimage.gaussian_filter(temp, sigma=SIGMA, mode='reflect')
            frame[filled] = smoothed[filled]

        filtered_data[t] = frame

        if (t+1) % 1000 == 0:
            print(f'    Filtered {t+1}/{T} frames')

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('sst_data', data=filtered_data, compression='gzip')
        f.create_dataset('original_obs_mask', data=obs_mask)
        f.create_dataset('land_mask', data=land_mask)
        f.create_dataset('latitude', data=lat)
        f.create_dataset('longitude', data=lon)
        f.create_dataset('timestamps', data=timestamps)

    elapsed = time.time() - start_time
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f'  Saved to: {output_file}')
    print(f'  Time: {elapsed:.1f}s, Size: {size_mb:.1f}MB')

print(f'\n{"="*70}')
print('All files processed!')
print(f'{"="*70}')
