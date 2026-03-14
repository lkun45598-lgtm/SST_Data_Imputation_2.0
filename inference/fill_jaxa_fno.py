"""
JAXA SST FNO推理填充脚本

输入: 时间加权填充数据 (jaxa_weighted_aligned)
输出: FNO填充后的NC文件 (按 YYYYMM/YYYYMMDD.nc 格式保存)

处理流程:
1. 按时间顺序排列所有h5文件
2. 跨文件连接，确保每天都有30天历史数据
3. FNO推理填充缺失区域
4. 高斯滤波 (sigma=1.0)
5. 保存为NC文件
"""

import numpy as np
import torch
import h5py
import netCDF4 as nc
from pathlib import Path
from datetime import datetime
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import argparse
import sys

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal


# ============================================================================
# Configuration
# ============================================================================

# 数据路径
INPUT_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_weighted_aligned')
OUTPUT_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_fno_filled')
MODEL_PATH = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/jaxa_finetune_8years/best_model.pth')
LAND_MASK_FILE = Path('/home/lz/Data_Imputation/visualization/jaxa_land_mask.npz')

# 模型参数
MODEL_CONFIG = {
    'out_size': (451, 351),
    'modes1': 80,
    'modes2': 64,
    'width': 64,
    'depth': 6,
    'cbam_reduction_ratio': 16
}

# 处理参数
SEQ_LEN = 30  # 输入序列长度
GAUSSIAN_SIGMA = 1.0  # 高斯滤波sigma

# NC文件参考时间
NC_TIME_UNITS = 'seconds since 1981-01-01 00:00:00'
NC_TIME_CALENDAR = 'standard'


# ============================================================================
# Helper Functions
# ============================================================================

def load_model(model_path, device):
    """加载FNO-CBAM模型"""
    model = FNO_CBAM_SST_Temporal(**MODEL_CONFIG).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    norm_mean = checkpoint.get('norm_mean', 299.9221)
    norm_std = checkpoint.get('norm_std', 2.6919)

    return model, norm_mean, norm_std


def load_land_mask():
    """加载陆地掩码"""
    if LAND_MASK_FILE.exists():
        return np.load(LAND_MASK_FILE)['land_mask']
    return None


def get_sorted_series_info(input_dir):
    """获取按时间排序的series信息"""
    series_info = []

    for h5_file in sorted(input_dir.glob('jaxa_weighted_series_*.h5')):
        with h5py.File(h5_file, 'r') as f:
            timestamps = [ts.decode('utf-8') if isinstance(ts, bytes) else ts
                         for ts in f['timestamps'][:]]
            start_date = datetime.strptime(timestamps[0][:10], '%Y-%m-%d')
            end_date = datetime.strptime(timestamps[-1][:10], '%Y-%m-%d')

            series_info.append({
                'file': h5_file,
                'start_date': start_date,
                'end_date': end_date,
                'timestamps': timestamps,
                'num_days': len(timestamps)
            })

    # 按开始时间排序
    series_info.sort(key=lambda x: x['start_date'])

    return series_info


def apply_gaussian_filter(sst_data, land_mask, sigma=1.0):
    """对SST数据应用高斯滤波"""
    sst = sst_data.copy()

    valid_mask = ~np.isnan(sst) & (land_mask == 0)
    if valid_mask.sum() == 0:
        return sst

    mean_val = np.nanmean(sst[valid_mask])
    sst_for_filter = sst.copy()
    sst_for_filter[~valid_mask] = mean_val

    filtered = gaussian_filter(sst_for_filter, sigma=sigma)
    result = np.where(land_mask == 0, filtered, np.nan)

    return result


def run_fno_inference(model, sst_seq, mask_seq, norm_mean, norm_std, device):
    """
    运行FNO模型推理

    Args:
        model: FNO-CBAM模型
        sst_seq: [30, H, W] SST序列 (Kelvin), 缺失区域已填充均值
        mask_seq: [30, H, W] mask序列 (1=缺失, 0=有效)
        norm_mean, norm_std: 归一化参数
        device: 设备

    Returns:
        pred: [H, W] 预测结果 (Kelvin)
    """
    # 归一化
    sst_norm = (sst_seq - norm_mean) / norm_std
    sst_norm = np.nan_to_num(sst_norm, nan=0.0)

    # 转tensor
    sst_tensor = torch.from_numpy(sst_norm).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask_seq.astype(np.float32)).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        pred = model(sst_tensor, mask_tensor)

    # 反归一化
    pred_kelvin = pred.squeeze().cpu().numpy() * norm_std + norm_mean

    return pred_kelvin


def datetime_to_nc_time(dt):
    """将datetime转换为NC时间（seconds since 1981-01-01）"""
    ref_date = datetime(1981, 1, 1, 0, 0, 0)
    delta = dt - ref_date
    return int(delta.total_seconds())


def save_nc_file(output_path, sst_data, lat, lon, timestamp_str):
    """
    保存为NC文件

    Args:
        output_path: 输出路径
        sst_data: [H, W] SST数据 (Kelvin)
        lat, lon: 坐标数组
        timestamp_str: 时间戳字符串 (如 '2017-08-17T00:00:00')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 解析时间
    dt = datetime.strptime(timestamp_str[:19], '%Y-%m-%dT%H:%M:%S')
    time_val = datetime_to_nc_time(dt)

    # 创建NC文件
    with nc.Dataset(output_path, 'w', format='NETCDF4') as f:
        # 创建维度
        f.createDimension('time', 1)
        f.createDimension('lat', len(lat))
        f.createDimension('lon', len(lon))

        # 创建变量 - time
        time_var = f.createVariable('time', 'i8', ('time',))
        time_var.units = NC_TIME_UNITS
        time_var.calendar = NC_TIME_CALENDAR
        time_var.long_name = 'reference time of sst file'
        time_var[:] = [time_val]

        # 创建变量 - lat
        lat_var = f.createVariable('lat', 'f4', ('lat',))
        lat_var.units = 'degrees_north'
        lat_var.long_name = 'latitude'
        lat_var[:] = lat

        # 创建变量 - lon
        lon_var = f.createVariable('lon', 'f4', ('lon',))
        lon_var.units = 'degrees_east'
        lon_var.long_name = 'longitude'
        lon_var[:] = lon

        # 创建变量 - sst
        sst_var = f.createVariable('sea_surface_temperature', 'f4', ('time', 'lat', 'lon'),
                                   fill_value=np.nan)
        sst_var.units = 'kelvin'
        sst_var.long_name = 'sea surface skin temperature (FNO filled)'
        sst_var[0, :, :] = sst_data

        # 全局属性
        f.title = 'JAXA SST FNO Filled'
        f.institution = 'FNO-CBAM Model'
        f.source = 'Temporal weighted data + FNO inference + Gaussian filter'
        f.history = f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'


# ============================================================================
# Main Processing
# ============================================================================

class JAXAFNOFiller:
    """JAXA SST FNO填充器"""

    def __init__(self, device, model_path=MODEL_PATH):
        self.device = device

        # 加载模型
        print("Loading FNO-CBAM model...")
        self.model, self.norm_mean, self.norm_std = load_model(model_path, device)
        print(f"  norm_mean={self.norm_mean:.4f}K, norm_std={self.norm_std:.4f}K")

        # 加载陆地掩码
        self.land_mask = load_land_mask()
        if self.land_mask is not None:
            print(f"  Land mask loaded: {self.land_mask.sum()} land pixels")

        # 获取series信息
        print("\nScanning input files...")
        self.series_info = get_sorted_series_info(INPUT_DIR)
        for i, info in enumerate(self.series_info):
            print(f"  [{i}] {info['file'].name}: {info['start_date'].date()} to {info['end_date'].date()} ({info['num_days']} days)")

        # 数据缓存
        self.data_cache = {}
        self.lat = None
        self.lon = None

    def load_series_data(self, series_idx):
        """加载series数据到缓存"""
        if series_idx in self.data_cache:
            return self.data_cache[series_idx]

        info = self.series_info[series_idx]
        with h5py.File(info['file'], 'r') as f:
            data = {
                'sst': f['sst_data'][:],
                'mask': f['missing_mask'][:],
                'timestamps': info['timestamps']
            }

            if self.lat is None:
                self.lat = f['latitude'][:]
                self.lon = f['longitude'][:]

        # 只保留最近2个series的缓存，避免内存溢出
        if len(self.data_cache) >= 2:
            oldest_key = min(self.data_cache.keys())
            del self.data_cache[oldest_key]

        self.data_cache[series_idx] = data
        return data

    def get_30day_sequence(self, series_idx, day_idx):
        """
        获取30天序列，支持跨文件

        Args:
            series_idx: 当前series索引
            day_idx: 当前series中的天索引

        Returns:
            sst_seq: [30, H, W] SST序列
            mask_seq: [30, H, W] mask序列
            或 None 如果数据不足
        """
        current_data = self.load_series_data(series_idx)

        if day_idx >= SEQ_LEN - 1:
            # 当前文件内有足够数据
            start_idx = day_idx - SEQ_LEN + 1
            sst_seq = current_data['sst'][start_idx:day_idx+1]
            mask_seq = current_data['mask'][start_idx:day_idx+1]
            return sst_seq, mask_seq

        elif series_idx > 0:
            # 需要从前一个文件获取数据
            prev_data = self.load_series_data(series_idx - 1)

            # 需要从前一个文件取的天数
            need_from_prev = SEQ_LEN - 1 - day_idx

            if len(prev_data['sst']) >= need_from_prev:
                # 前一个文件有足够数据
                prev_sst = prev_data['sst'][-need_from_prev:]
                prev_mask = prev_data['mask'][-need_from_prev:]

                curr_sst = current_data['sst'][:day_idx+1]
                curr_mask = current_data['mask'][:day_idx+1]

                sst_seq = np.concatenate([prev_sst, curr_sst], axis=0)
                mask_seq = np.concatenate([prev_mask, curr_mask], axis=0)

                return sst_seq, mask_seq

        # 数据不足
        return None, None

    def process_single_day(self, sst_seq, mask_seq, target_sst, target_mask):
        """
        处理单天数据

        Args:
            sst_seq: [30, H, W] 输入SST序列
            mask_seq: [30, H, W] 输入mask序列
            target_sst: [H, W] 目标天的原始SST
            target_mask: [H, W] 目标天的mask

        Returns:
            filled_sst: [H, W] 填充后的SST (经过高斯滤波)
        """
        # 准备推理输入：缺失区域填充均值
        sst_for_inference = sst_seq.copy()
        for t in range(SEQ_LEN):
            missing_t = (mask_seq[t] > 0) | np.isnan(sst_seq[t])
            sst_for_inference[t] = np.where(missing_t, self.norm_mean, sst_seq[t])

        # FNO推理
        pred = run_fno_inference(
            self.model, sst_for_inference, mask_seq,
            self.norm_mean, self.norm_std, self.device
        )

        # 组合结果：观测区域用原值，缺失区域用预测
        ocean_mask = (self.land_mask == 0)
        target_missing = (target_mask > 0) | np.isnan(target_sst)

        result = np.where(target_missing & ocean_mask, pred, target_sst)
        result = np.where(self.land_mask > 0, np.nan, result)

        # 高斯滤波
        result = apply_gaussian_filter(result, self.land_mask, sigma=GAUSSIAN_SIGMA)

        return result

    def run(self, start_series=None, end_series=None):
        """
        运行完整填充流程

        Args:
            start_series: 起始series索引 (默认0)
            end_series: 结束series索引 (默认全部)
        """
        if start_series is None:
            start_series = 0
        if end_series is None:
            end_series = len(self.series_info)

        print(f"\n{'='*70}")
        print("Starting FNO filling process")
        print(f"{'='*70}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Gaussian sigma: {GAUSSIAN_SIGMA}")

        total_processed = 0
        total_skipped = 0

        for series_idx in range(start_series, end_series):
            info = self.series_info[series_idx]
            print(f"\n--- Processing Series {series_idx}: {info['file'].name} ---")
            print(f"    Date range: {info['start_date'].date()} to {info['end_date'].date()}")

            current_data = self.load_series_data(series_idx)
            num_days = len(current_data['timestamps'])

            # 确定起始天
            if series_idx == 0:
                # 第一个文件从第30天开始
                start_day = SEQ_LEN - 1
            else:
                # 后续文件从第1天开始（使用前一个文件的数据）
                start_day = 0

            for day_idx in tqdm(range(start_day, num_days), desc=f"Series {series_idx}"):
                timestamp = current_data['timestamps'][day_idx]

                # 获取30天序列
                sst_seq, mask_seq = self.get_30day_sequence(series_idx, day_idx)

                if sst_seq is None:
                    total_skipped += 1
                    continue

                # 处理当天
                target_sst = current_data['sst'][day_idx]
                target_mask = current_data['mask'][day_idx]

                filled_sst = self.process_single_day(sst_seq, mask_seq, target_sst, target_mask)

                # 生成输出路径: YYYYMM/YYYYMMDD.nc
                dt = datetime.strptime(timestamp[:10], '%Y-%m-%d')
                year_month = dt.strftime('%Y%m')
                date_str = dt.strftime('%Y%m%d')
                output_path = OUTPUT_DIR / year_month / f'{date_str}.nc'

                # 保存NC文件
                save_nc_file(output_path, filled_sst, self.lat, self.lon, timestamp)

                total_processed += 1

        print(f"\n{'='*70}")
        print(f"FNO filling completed!")
        print(f"  Total processed: {total_processed} days")
        print(f"  Total skipped: {total_skipped} days")
        print(f"  Output directory: {OUTPUT_DIR}")
        print(f"{'='*70}")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='JAXA SST FNO Filling')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--start-series', type=int, default=None, help='Start series index')
    parser.add_argument('--end-series', type=int, default=None, help='End series index')

    args = parser.parse_args()

    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # 创建填充器并运行
    filler = JAXAFNOFiller(device)
    filler.run(start_series=args.start_series, end_series=args.end_series)


if __name__ == '__main__':
    main()
