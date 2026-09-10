import os
import csv
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
from natsort import natsorted
from matplotlib import pyplot as plt
import pymzml
from bintrees import FastAVLTree
from ultralytics import YOLO
import bisect
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")


# =========================
# ROI 数据结构与构建
# =========================

class ROIObj:
    def __init__(self, start_scan_idx, start_time_sec, init_mz, init_intensity):
        self.scan_start = start_scan_idx
        self.scan_end = start_scan_idx
        self.rt_start_sec = start_time_sec
        self.rt_end_sec = start_time_sec

        self.mean_mz = float(init_mz)
        self.points = 1
        self.mzmin = float(init_mz)
        self.mzmax = float(init_mz)

        self.times_sec = [float(start_time_sec)]
        self.intensities = [float(init_intensity)]

        self._touched_in_this_scan = True
        self._missing_streak = 0

    def touch_same_scan(self, mz, intensity):
        # 同一扫描中多点合并（强度累加；均值 m/z 更新）
        self.mean_mz = (self.mean_mz * self.points + mz) / (self.points + 1)
        self.points += 1
        self.intensities[-1] = self.intensities[-1] + float(intensity)

        self.mzmin = min(self.mzmin, mz)
        self.mzmax = max(self.mzmax, mz)

        self._touched_in_this_scan = True

    def extend_new_scan(self, scan_idx, time_sec, mz, intensity):
        self.mean_mz = (self.mean_mz * self.points + mz) / (self.points + 1)
        self.points += 1
        self.scan_end = scan_idx
        self.rt_end_sec = time_sec
        self.times_sec.append(float(time_sec))
        self.intensities.append(float(intensity))
        self.mzmin = min(self.mzmin, mz)
        self.mzmax = max(self.mzmax, mz)
        self._touched_in_this_scan = True
        self._missing_streak = 0

    def append_missing(self, scan_idx, time_sec):
        self.scan_end = scan_idx
        self.rt_end_sec = time_sec
        self.times_sec.append(float(time_sec))
        self.intensities.append(0.0)
        self._missing_streak += 1
        self._touched_in_this_scan = False

    def apex_time_min(self):
        if not self.intensities:
            return (self.rt_start_sec + self.rt_end_sec) / 120.0
        k = int(np.argmax(self.intensities))
        return float(self.times_sec[k] / 60.0)

    def rt_window_min(self):
        return float(self.rt_start_sec / 60.0), float(self.rt_end_sec / 60.0)


def _spec_time_in_seconds(spec):
    if hasattr(spec, "ms_level") and spec.ms_level != 1:
        return None
    if hasattr(spec, "scan_time") and spec.scan_time is not None:
        val, unit = spec.scan_time
        if unit == "second":
            return float(val)
        elif unit == "minute":
            return float(val) * 60.0
    if hasattr(spec, "scan_time_in_minutes"):
        return float(spec.scan_time_in_minutes()) * 60.0
    return None

def get_all_eics_via_roi(datadir, ppm=10, roi_delta_mz=0.005, required_points=8, dropped_points=3,
                         min_nonzero_points=3, min_height=0.0):
    """
    对 datadir 下所有 mzML：构建 ROI -> 导出 EIC 列表。
    返回: list_per_file，其中每个元素为该文件的 eics 列表（list[dict]）
    """
    p = Path(datadir).resolve()
    paths = [path for path in p.glob("*.mzML")]
    paths = natsorted(paths)
    if not paths:
        raise FileNotFoundError(f"No mzML found under: {datadir}")

    all_eics = []
    for path in paths:
        rois, _ = build_rois_for_file(
            path, ppm=ppm, roi_delta_mz=roi_delta_mz,
            required_points=required_points, dropped_points=dropped_points
        )
        eics = make_eics_from_rois(
            rois,path, min_nonzero_points=min_nonzero_points, min_height=min_height
        )
        all_eics.append(eics)
    return all_eics


def get_eics_for_single_file(path, ppm=10, roi_delta_mz=0.005, required_points=8, dropped_points=3,
                             min_nonzero_points=3, min_height=0.0):
    """
    对单个 mzML 文件：构建 ROI -> 导出 EIC 列表。
    返回: eics 列表（list[dict]）
    """
    rois, _ = build_rois_for_file(
        path, ppm=ppm, roi_delta_mz=roi_delta_mz,
        required_points=required_points, dropped_points=dropped_points
    )
    eics = make_eics_from_rois(
        rois, path, min_nonzero_points=min_nonzero_points, min_height=min_height
    )
    return eics

def build_rois_for_file(path, ppm=10, roi_delta_mz=0.02, required_points=8, dropped_points=3):
    """
    从单个 mzML 构建 ROI 列表（仅 MS1）。
    - ppm: m/z 相对容差（自适应）
    - roi_delta_mz: 绝对容差下限（Da）
    - required_points: ROI 至少扫描点数（建议 <= 峰上点数的 0.5~0.7 倍）
    - dropped_points: 允许连续缺失点数
    返回:
      rois_finished: list[ROIObj]
      run_times_min: list[float]（整场 MS1 扫描时间，分钟）
    """
    run = pymzml.run.Reader(path)
    scans = []
    times_sec = []
    for spec in run:
        t_sec = _spec_time_in_seconds(spec)
        if t_sec is None:
            continue
        scans.append(spec)
        times_sec.append(t_sec)

    if len(scans) == 0:
        return [], []

    active = FastAVLTree()
    rois_finished = []

    min_key = float("inf")
    max_key = 0.0

    # 初始化（第 0 扫）
    s0 = scans[0]
    t0 = times_sec[0]
    for mz, I in zip(s0.mz, s0.i):
        if I > 0:
            roi = ROIObj(0, t0, mz, I)
            active[mz] = roi
            min_key = min(min_key, mz)
            max_key = max(max_key, mz)

    for scan_idx in tqdm(range(len(scans)), desc=f"ROIs @ {Path(path).name}"):
        if scan_idx == 0:
            continue
        spec = scans[scan_idx]
        t_sec = times_sec[scan_idx]

        # 标记本扫描尚未触达
        for _, roi in active.items():
            roi._touched_in_this_scan = False

        for mz, I in zip(spec.mz, spec.i):
            if I <= 0:
                continue

            ceiling_item = None
            floor_item = None
            if mz < max_key:
                _, ceiling_item = active.ceiling_item(mz)
            if mz > min_key:
                _, floor_item = active.floor_item(mz)

            if ceiling_item is None and floor_item is None:
                roi = ROIObj(scan_idx, t_sec, mz, I)
                active[mz] = roi
                min_key = min(min_key, mz)
                max_key = max(max_key, mz)
                continue

            if ceiling_item is None:
                closest_roi = floor_item
            elif floor_item is None:
                closest_roi = ceiling_item
            else:
                if abs(ceiling_item.mean_mz - mz) < abs(mz - floor_item.mean_mz):
                    closest_roi = ceiling_item
                else:
                    closest_roi = floor_item

            tol = max(roi_delta_mz, ppm * 1e-6 * max(closest_roi.mean_mz, mz))
            if abs(closest_roi.mean_mz - mz) <= tol:
                if closest_roi.scan_end == scan_idx:
                    closest_roi.touch_same_scan(mz, I)
                else:
                    closest_roi.extend_new_scan(scan_idx, t_sec, mz, I)
            else:
                roi = ROIObj(scan_idx, t_sec, mz, I)
                active[mz] = roi
                min_key = min(min_key, mz)
                max_key = max(max_key, mz)

        # 未触达 ROI：补 0 或关闭
        to_remove = []
        for key, roi in active.items():
            if roi.scan_end != scan_idx:
                roi.append_missing(scan_idx, t_sec)
                if roi._missing_streak > dropped_points:
                    if roi.points >= required_points:
                        rois_finished.append(roi)
                    to_remove.append(key)
        for key in to_remove:
            try:
                del active[key]
            except KeyError:
                pass

        try:
            min_key, _ = active.min_item()
            max_key, _ = active.max_item()
        except ValueError:
            min_key = float("inf")
            max_key = 0.0

    # 收尾
    for _, roi in active.items():
        if roi.points >= required_points:
            rois_finished.append(roi)

    run_times_min = [t / 60.0 for t in times_sec]
    return rois_finished, run_times_min

def make_eics_from_rois(rois,path, min_nonzero_points=3, min_height=0.0):
    """
    将 ROI 转为 EIC（时间–强度序列）。
    过滤：至少 min_nonzero_points 非零，且最大强度 >= min_height。
    返回: list[dict] with keys:
      mz, rt_min (np.ndarray), int (np.ndarray), rt_start, rt_end
    """
    eics = []
    for roi in rois:
        x = np.asarray(roi.times_sec, dtype=np.float64) / 60.0  # 分钟
        y = np.asarray(roi.intensities, dtype=np.float64)
        if np.count_nonzero(y) < min_nonzero_points:
            continue
        if y.max() < min_height:
            continue
        rt_start, rt_end = roi.rt_window_min()
        eics.append({
            "mz": float(roi.mean_mz),
            "mzmin": float(roi.mzmin),
            "mzmax": float(roi.mzmax),
            "rt_min": x,
            "int": y,
            "rt_start": float(rt_start),
            "rt_end": float(rt_end),
            "sample":path
        })
    return eics

def inference(eics_per_file, args):
    """
    基于 ROI 导出的 EIC 做推断。
    - eics_per_file: list[file_idx] -> list[eic dict]
    返回: final_results = list[file_idx] -> list[eic_peaks]；
         eic_peaks = list of peaks for that EIC,
         每个 peak: [x1, x2, y1, y2, conf, peak_rt, mz, peak_area_sum, mzmin, mzmax]
    """
    save_path = args.img_tmp_path
    model_path = args.model
    model = YOLO(model_path)
    noise_thresold = float(args.noise_thresold)

    final_results = []
    for file_eics in eics_per_file:
        eic_results = []
        for eic in tqdm(file_eics, desc="Inference on EICs"):
            x_abs = eic["rt_min"]           # 绝对时间（分钟）
            y = eic["int"]
            
            mz = [eic["mz"], eic["mzmin"], eic["mzmax"]]#[mz,mzmin,mzmax]
            sample = eic["sample"]
            peak_list = []

            if len(x_abs) < 2 or y.max() <= noise_thresold:
                eic_results.append([])
                continue
            
            windows = select_window_size(x_abs, args)
            

            for window in windows:
                mask = (x_abs >= window[0]) & (x_abs <= window[1])
                sub_x = x_abs[mask]
                sub_y = y[mask]

                if len(sub_x) == 0:
                    continue

                max_sub = float(sub_y.max())
                if(max_sub < noise_thresold):
                    continue

                if args.smooth_mode:
                    none_zero_mask = sub_y > 0
                    sub_x = sub_x[none_zero_mask]
                    sub_y = sub_y[none_zero_mask]
                    intensity_smooth = sub_y
                    # y_sub_nan = replace_zeros(sub_y,max_sub)
                    # y_interp = pd.Series(y_sub_nan).interpolate(limit_direction='both').to_numpy()
                    # intensity_smooth = savgol_filter(y_interp, 5, 2)
                    plt.plot(sub_x, intensity_smooth, linewidth=1.0, color='black')
                    plt.xlim(sub_x[0], sub_x[-1])
                elif args.down_sample:
                    
                    sub_x, sub_y = max_pooling_dowmsample(sub_x, sub_y, factor=3)
                    
                    plt.plot(sub_x, sub_y, linewidth=1.0, color='black')
                    if args.use_min_windowsize:
                        if window[1] - window[0] < args.min_window_size:
                            oriWindowSize = sub_x[-1] - sub_x[0]
                            left = max(sub_x[0] - (args.min_window_size - oriWindowSize) / 2, x_abs[0])
                            right = min(sub_x[-1] + (args.min_window_size - oriWindowSize) / 2, x_abs[-1])
                            plt.xlim(left, right)
                        else:
                            left = sub_x[0]
                            right = sub_x[-1]
                            plt.xlim(left, right)
                    else:
                        left = sub_x[0]
                        right = sub_x[-1]
                        plt.xlim(left, right)   
                else:
                    plt.plot(sub_x, sub_y, linewidth=1.0, color='black')

                    if args.use_min_windowsize:
                        if window[1] - window[0] < args.min_window_size:
                            oriWindowSize = sub_x[-1] - sub_x[0]
                            left = max(sub_x[0] - (args.min_window_size - oriWindowSize) / 2, x_abs[0])
                            right = min(sub_x[-1] + (args.min_window_size - oriWindowSize) / 2, x_abs[-1])
                            plt.xlim(left, right)
                        else:
                            left = sub_x[0]
                            right = sub_x[-1]
                            plt.xlim(left, right)
                    else:
                        left = sub_x[0]
                        right = sub_x[-1]
                        plt.xlim(left, right)

                plt.ylim(0, max_sub * 1.1)
                plt.axis('off')
                #plt.axis('on')
                #save_path1 = f"{save_path}/mz{mz[0]:.4f}-win{window[0]:.2f}-{window[1]:.2f}.jpg"
                #plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
                plt.savefig(save_path,bbox_inches='tight', pad_inches=0, dpi=100)
                if args.isDebug:
                    #debug_img_path = os.path.join(args.debugPlotImgPath, f"mz{mz[0]:.4f}-win{window[0]:.2f}-{window[1]:.2f}.jpg")
                    mz_name = mz[0]
                    rt_name = (left + right)/2
                    fname = os.path.splitext(os.path.basename(sample))[0]

                    dst_dir = os.path.join(args.debugPlotImgPath, fname)
                    pred_dst_dir = os.path.join(args.debugPlotImgPath, "pred_plots")
                    os.makedirs(dst_dir, exist_ok=True)
                    os.makedirs(pred_dst_dir, exist_ok=True)

                    new_name = f"{mz_name:.4f}_{rt_name:.2f}.jpg"
                    dst_path = os.path.join(dst_dir, new_name)
                    

                    import shutil
                    shutil.copy(save_path, dst_path)
                    
                plt.close()
                if args.isDebug:
                    peak = get_img_result_debug(model, dst_path, pred_dst_dir, fname, conf=0.5)
                else:
                    peak = get_img_result(model, save_path, conf=0.5)

                # if os.path.exists(save_path):
                #     os.unlink(save_path)

                for xy in peak:
                    # xy[0] = (sub_x[-1] - sub_x[0]) * xy[0] + sub_x[0]
                    # xy[1] = (sub_x[-1] - sub_x[0]) * xy[1] + sub_x[0]
                    xy[0] = xy[0] * (right - left) + left
                    xy[1] = xy[1] * (right - left) + left
                    xy[2] = (1 - xy[2]) * max_sub * 1.1
                    xy[3] = (1 - xy[3]) * max_sub * 1.1
                    #print("max_sub:", max_sub, "xy[2]:", xy[2], "xy[3]:", xy[3])
                    peak_list.append(xy)
                
            keep = peak_nms(peak_list)
            keep_lists = [peak_list[i] for i in keep if (peak_list[i][1] - peak_list[i][0])>0.05]

            peak_lists = allign_peaks(keep_lists,x_abs,y,mz,sample)
            eic_results.append(peak_lists)
        final_results.append(eic_results)
    return final_results

def select_window_size(rt_list , args):
    total_rt = rt_list[-1] - rt_list[0]
    offset = rt_list[0]
    window_counts = args.window_counts#EIC对应的窗口数量
    window_count_thresholds = args.window_count_thresholds#决定窗口数量的RT阈值

    if len(window_count_thresholds) + 1 != len(window_counts):
        raise ValueError("window_counts长度必须大于window_count_thresholds")

    for i, t in enumerate(window_count_thresholds):
        if total_rt < t:
            n_windows = window_counts[i]
            break
    else:
        n_windows = window_counts[-1]

    window_size = 2 * total_rt / (n_windows + 1)

    # if args.use_min_windowsize:
    #     window_size = max(window_size, args.min_window_size)
    
    step = window_size / 2

    def nearest(value):
        pos = bisect.bisect_left(rt_list, value)
        if pos == 0:
            return rt_list[0]
        if pos == len(rt_list):
            return rt_list[-1]
        before, after = rt_list[pos - 1], rt_list[pos]
        return before if abs(value - before) <= abs(value - after) else after


    windows = []
    for i in range(n_windows):
        start = i * step + offset
        end = start + window_size
        if end > rt_list[-1]:  # 防止超过 total_rt
            end = rt_list[-1]
        windows.append((nearest(start), nearest(end)))
    
    return windows

def get_img_result(model, img_path, conf):
    result = model.predict(img_path, conf, verbose=False)
    peak = []
    for item in result:
        if len(item.boxes) > 0:
            for i, box in enumerate(item.boxes):
                x1, y1, x2, y2 = box.xyxyn.cpu().flatten().tolist()
                conf_score = box.conf.cpu().item()
                peak.append([x1, x2, y1, y2, conf_score])
    return peak

def get_img_result_debug(model, img_path, pred_path, file_name, conf):
    result = model.predict(img_path, conf, verbose=False, save=True, project=pred_path, name=file_name, exist_ok=True)
    peak = []
    for item in result:
        if len(item.boxes) > 0:
            for i, box in enumerate(item.boxes):
                x1, y1, x2, y2 = box.xyxyn.cpu().flatten().tolist()
                conf_score = box.conf.cpu().item()
                peak.append([x1, x2, y1, y2, conf_score])
    return peak

def peak_nms(peak_list):
    iou_threshold = 0.15
    if len(peak_list) == 0:
        return peak_list
    peaks = np.array(peak_list)
    x1 = peaks[:, 0]
    x2 = peaks[:, 1]
    y1 = peaks[:, 2]
    y2 = peaks[:, 3]
    scores = peaks[:, 4]
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_others = (x2[indices[1:]] - x1[indices[1:]]) * (y2[indices[1:]] - y1[indices[1:]])
        union = area_i + area_others - inter
        iou = inter / (union + 1e-6)
        indices = indices[1:][iou < iou_threshold]
    return keep

def allign_peaks(peak_list,x,y,mz,sample):
    if not peak_list:
        return []
    
    peaks = sorted(peak_list,key=lambda x:x[0])
    merged = []

    # 初始化第一个峰
    cur_peak = peaks[0][:]

    for current in peaks[1:]:
        # 判断是否有交集 (包括重叠和包含)
        if current[0] <= cur_peak[1]:
            # 左边界：取更左的
            if current[0] < cur_peak[0]:
                cur_peak[0], cur_peak[2] = current[0], current[2]
            # 右边界：取更右的
            if current[1] > cur_peak[1]:
                cur_peak[1], cur_peak[3] = current[1], current[3]
        else:
            merged.append(cur_peak)
            cur_peak = current[:]

    # 加入最后一个峰
    merged.append(cur_peak)

    for peak in merged:
        #mask = (x >= peak[0]) & (x <= peak[1]) #原有的直接选取两个判定点中间部分的方法
        mask_start = chooseMask(peak[0],x,y, prefer_left=True)
        mask_end = chooseMask(peak[1],x,y, prefer_left=False)
        mask = (x >= mask_start) & (x <= mask_end)
        idx_max = np.argmax(y[mask])
        peak_rt = x[mask][idx_max]
        peak_area = float(np.trapz(y[mask], x[mask])) * 60
        peak.append(peak_rt)
        peak.append(mz[0])
        peak.append(peak_area)
        peak.append(mz[1])
        peak.append(mz[2])
        peak.append(sample)
    return merged

def chooseMask(point, x, y, prefer_left=False):
    """
    根据给定 point，在 x 中找 point 左右最近的两个点，
    按 y 值较小的规则选一个作为边界点。
    
    prefer_left:
        True  → 当 y 相同时选左点
        False → 当 y 相同时选右点
    """
    # 找到 point 在 x 中应插入的位置
    idx = np.searchsorted(x, point)

    # 找左右点（考虑越界）
    x_left = x[idx - 1] if idx - 1 >= 0 else None
    x_right = x[idx] if idx < len(x) else None

    # 同时取 y
    y_left = y[idx - 1] if idx - 1 >= 0 else None
    y_right = y[idx] if idx < len(y) else None

    # 处理越界情况
    if x_left is None:
        return x_right    # 只能用右侧
    if x_right is None:
        return x_left     # 只能用左侧

    # 两侧都存在 → 比较 y
    if y_left < y_right:
        return x_left
    elif y_right < y_left:
        return x_right
    else:
        # y 相同时的偏向规则
        return x_left if prefer_left else x_right


def save_result(results, eics_per_file, args):
    """批量保存（兼容旧调用），内部逐文件保存。"""
    p = Path(args.datadir).resolve()
    paths = [path for path in p.glob("*.mzML")]
    paths = sorted(paths, key=lambda x: x.as_posix())

    for index in range(len(paths)):
        save_single_file_result(
            results=[results[index]],
            eics_per_file=[eics_per_file[index]],
            mzml_paths=[paths[index]],
            args=args
        )

def save_single_file_result(results, eics_per_file, mzml_paths, args):
    """
    保存单个文件的结果（CSV 和/或图像）。
    - results: list[list[eic_peaks]]，长度为1
    - eics_per_file: list[list[eic_dict]]，长度为1
    - mzml_paths: list[Path]，长度为1
    """
    save_csv = bool(args.save_csv)
    save_images = bool(args.save_images)
    images_dir = Path(args.images_dir) if hasattr(args, "images_dir") and args.images_dir else None
    window_size = float(getattr(args, "images_window_size", 2.0))

    # 1) 保存 CSV
    if save_csv:
        title = ['mz','mzmin','mzmax','rt', 'rtmin', 'rtmax', 'into','maxo', 'sample', 'conf']

        if hasattr(args, "csv_save_dir") and args.csv_save_dir:
            csv_root = Path(args.csv_save_dir).resolve()
        else:
            p = Path(args.datadir).resolve()
            csv_root = p.joinpath("ours/csv")
        csv_root.mkdir(parents=True, exist_ok=True)

        for index in range(len(mzml_paths)):
            result = results[index]
            path = mzml_paths[index]
            csv_path = csv_root / f"{path.stem}.csv"
            with open(csv_path, "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(title)
                if len(result) > 0:
                    for peaks in result:
                        for peak in peaks:
                            row = [peak[6], peak[8], peak[9], peak[5], peak[0], peak[1], peak[7], peak[2], peak[10], peak[4]]
                            writer.writerow(row)
            df = pd.read_csv(csv_path)
            os.unlink(csv_path)
            if len(df) == 0:
                pd.DataFrame(columns=title).to_csv(csv_path, index=False)
            else:
                df_sort = df.sort_values(by=['mz', 'rtmin'])
                grouped = df_sort.groupby(['mz', 'rt'], as_index=False).agg({
                    'mzmin': 'min',
                    'mzmax': 'max',
                    'rtmin': 'min',
                    'rtmax': 'max',
                    'into': 'max',
                    'maxo': 'max',
                    'sample': 'first',
                    'conf':'max'
                })
                grouped.columns = [str(col) for col in grouped.columns]
                grouped.to_csv(csv_path, index=False)
            print(f"[CSV] Saved: {csv_path}")

    # 2) 保存 EIC 局部图像（rt vs intensity）
    if save_images:
        if images_dir is None:
            images_dir = Path(args.datadir) / "peak_images"
        images_dir.mkdir(parents=True, exist_ok=True)

        skyblue_hex = "#87CEEB"

        for file_idx in range(len(mzml_paths)):
            result = results[file_idx]
            file_eics = eics_per_file[file_idx]
            mzml_path = mzml_paths[file_idx]
            file_out_dir = images_dir / mzml_path.stem
            file_out_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for eic_idx, (peaks, eic) in enumerate(zip(result, file_eics)):
                if len(peaks) == 0:
                    continue
                x = np.asarray(eic["rt_min"], dtype=float)
                y = np.asarray(eic["int"], dtype=float)
                mz = float(eic["mz"])
                if len(x) == 0:
                    continue

                eic_start, eic_end = float(x[0]), float(x[-1])
                eic_span = eic_end - eic_start

                if eic_span <= window_size:
                    windows = [(eic_start, eic_end, peaks)]
                else:
                    windows = group_peaks_into_windows(peaks, window_size, eic_start, eic_end)

                for (win_start, win_end, group_peaks) in windows:
                    mask = (x >= win_start) & (x <= win_end)
                    if not np.any(mask):
                        continue
                    x_sub = x[mask]
                    y_sub = y[mask]

                    plt.figure(figsize=(6, 4), dpi=150)
                    plt.plot(x_sub, y_sub, color='black', linewidth=1.0)
                    plt.xlabel("RT (min)")
                    plt.ylabel("Intensity")
                    plt.ylim(0, max(1.0, float(y_sub.max()) * 1.1))
                    plt.xlim(x_sub[0], x_sub[-1])

                    for peak in group_peaks:
                        rt_left = float(peak[0])
                        rt_right = float(peak[1])
                        mask_fill = (x_sub >= rt_left) & (x_sub <= rt_right)
                        if np.any(mask_fill):
                            plt.fill_between(x_sub, 0, y_sub, where=mask_fill,
                                             color=skyblue_hex, alpha=0.4)

                    fname = f"{mzml_path.stem}_eic{eic_idx}_mz{mz:.5f}_win{win_start:.2f}-{win_end:.2f}.png"
                    out_path = file_out_dir / fname
                    plt.tight_layout()
                    plt.savefig(out_path)
                    plt.close()
                    saved += 1

            print(f"[Images] Saved {saved} EIC peak images to: {file_out_dir}")

def group_peaks_into_windows(peaks, window_size, eic_start, eic_end):
    """
    peaks: list of peak records for one EIC; each peak: [rt_left(0), rt_right(1), ..., rt_med(6), mz(7), ...]
    window_size: float, minutes
    eic_start/eic_end: EIC 时间范围（分钟）
    返回: list of (win_start, win_end, group_peaks)
    """
    if not peaks:
        return []

    # 先按 rt_left 排序，便于分组
    peaks_sorted = sorted(peaks, key=lambda p: float(p[0]))

    # 整体能否放入一个窗口
    overall_minL = min(float(p[0]) for p in peaks_sorted)
    overall_maxR = max(float(p[1]) for p in peaks_sorted)
    if (overall_maxR - overall_minL) <= window_size:
        # 将窗口居中到峰群中点，并裁剪到 EIC 范围内
        mid = 0.5 * (overall_minL + overall_maxR)
        win_start = max(eic_start, min(mid - 0.5 * window_size, eic_end - window_size))
        win_end = min(eic_end, win_start + window_size)
        return [(win_start, win_end, peaks_sorted)]

    # 否则，使用贪心分组：当前组的 [minL, maxR] 跨度不能超过窗口大小
    groups = []
    cur = [peaks_sorted[0]]
    cur_minL = float(peaks_sorted[0][0])
    cur_maxR = float(peaks_sorted[0][1])

    for p in peaks_sorted[1:]:
        L, R = float(p[0]), float(p[1])
        new_minL = min(cur_minL, L)
        new_maxR = max(cur_maxR, R)
        if (new_maxR - new_minL) <= window_size:
            cur.append(p)
            cur_minL, cur_maxR = new_minL, new_maxR
        else:
            groups.append((cur_minL, cur_maxR, cur))
            cur = [p]
            cur_minL, cur_maxR = L, R
    groups.append((cur_minL, cur_maxR, cur))

    # 为每组确定窗口起止（同样居中并裁剪）
    windows = []
    for g_minL, g_maxR, g_peaks in groups:
        mid = 0.5 * (g_minL + g_maxR)
        win_start = max(eic_start, min(mid - 0.5 * window_size, eic_end - window_size))
        win_end = min(eic_end, win_start + window_size)
        windows.append((win_start, win_end, g_peaks))
    return windows

def replace_zeros(intensity,max_intensity):
    result = intensity.copy()
    threshold = max_intensity * 0.3
    n = len(intensity)

    for i in range(1, n - 1):  # 避免越界
        if intensity[i] == 0 and intensity[i - 1] >= threshold and intensity[i + 1] >= threshold:
            result[i] = np.nan

    return result

def max_pooling_dowmsample(x, y, factor):
    if factor <= 1:
        return x, y
    n = len(x)
    
    new_x = []
    new_y = []
    for i in range(0, n, factor):
        end = min(i + factor, n)
        max_idx = np.argmax(y[i:end]) + i
        new_x.append(x[max_idx])
        new_y.append(y[max_idx])
    return np.array(new_x), np.array(new_y)

def get_args():
    parser = argparse.ArgumentParser()

    # ROI 构建参数
    parser.add_argument('--ppm', type=int, default=15, help='ppm tolerance for ROI dynamic m/z matching')
    parser.add_argument('--roi_delta_mz', type=float, default=0.01, help='absolute m/z tol lower bound (Da)')
    parser.add_argument('--roi_required_points', type=int, default=15, help='min points per ROI')
    parser.add_argument('--roi_dropped_points', type=int, default=8, help='allowed consecutive missing points')

    # EIC 过滤
    parser.add_argument('--min_nonzero_points', type=int, default=5, help='min nonzero points in EIC to keep')
    parser.add_argument('--min_height', type=float, default=5000, help='min peak height to keep EIC')

    # 滑窗参数（分钟）
    parser.add_argument('--window_counts',type=int, nargs='+',default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14],help='num of windows in one EIC')
    parser.add_argument('--window_count_thresholds',type=float, nargs='+',default=[1,2,3,4,5,6,7,8,9,10,11,12,13],help='the thresholds define num of windows')
    # parser.add_argument('--window_size', type=float, default=1.6, help='sliding window size (min)')
    # parser.add_argument('--slide_step', type=float, default=0.8, help='sliding step (min)')

    # YOLO & IO
    # parser.add_argument('--model',
    #                     default='/data1/zhaohaowei/peak_detection/yolo/ultralytics/runs/semi_segment/train235/weights/best.pt',
    #                     help='path to peak detection model')
    parser.add_argument('--model',
                        default='/data1/zhaohaowei/peak_detection/pltest/mytest_pl/runs/semi_detect/train6/weights/best.pt',
                        help='path to peak detection model')
    parser.add_argument('--datadir',
                        default='/data1/zhaohaowei/peak_detection/mzml_dataset/experiment/exp2/QEHF/')
    # parser.add_argument('--datadir',
    #                     default='/data1/zhaohaowei/peak_detection/mzml_dataset/experiment/exp3_st004098/')
    parser.add_argument('--img_tmp_path',
                        default='/data1/zhaohaowei/peak_detection/mzml_dataset/experiment/exp2/QEHF/ours_20260905/tmp.jpg')
    parser.add_argument('--save_csv', action='store_true', default=True)
    parser.add_argument('--csv_save_dir', type=str, default='/data1/zhaohaowei/peak_detection/mzml_dataset/experiment/exp2/QEHF/ours_20260906/csv/',
                        help='directory to save csv files; default datadir/ours/csv')
    parser.add_argument('--noise_thresold', type=float, default=5000)


    parser.add_argument('--save_images', action='store_true', default=False,
                        help='save EIC local plots (rt vs intensity) around predicted peaks')
    parser.add_argument('--images_dir', type=str, default='/data1/zhaohaowei/peak_detection/mzml_dataset/experiment/exp2/QEHF/ours_20260905/img',
                        help='directory to save peak images; default datadir/peak_images')
    parser.add_argument('--images_rt_margin', type=float, default=0.1,
                        help='margin (min) added around [rt_left, rt_right] for plotting window')
    parser.add_argument('--images_window_size', type=float, default=1,
                    help='EIC plotting window size in minutes (default 2.0)')

    #平滑设置和下采样设置
    parser.add_argument('--smooth_mode',type=bool, default=False,
                        help='whether smooth the RT_Intensity pairs when plotting or not')
    parser.add_argument('--down_sample',type=bool, default=False, help='whether down sample the RT_Intensity pairs in windows or not')
    parser.add_argument("--use_min_windowsize", type=bool, default=False, help="whether use the min window size between the predefined window size and the actual RT span of EIC")
    parser.add_argument('--min_window_size', type=float, default=0.8, help='the min window size when use_min_windowsize is True')
    
    #degug设置
    parser.add_argument('--isDebug',type=bool, default=False, help='enable debug mode with verbose output')
    parser.add_argument('--debugPlotImgPath', type=str, default='/data1/zhaohaowei/peak_detection/mzml_dataset/experiment/exp3_st004098/ours_260611_test/debug_plot/')
    return parser

def main(args):
    # 0) 获取所有 mzML 文件列表
    p = Path(args.datadir).resolve()
    all_paths = [path for path in p.glob("*.mzML")]
    all_paths = natsorted(all_paths)
    if not all_paths:
        raise FileNotFoundError(f"No mzML found under: {args.datadir}")

    # 1) 幂等检查：确定 CSV 输出目录，检查已完成的文件
    if hasattr(args, "csv_save_dir") and args.csv_save_dir:
        csv_root = Path(args.csv_save_dir).resolve()
    else:
        csv_root = p.joinpath("ours/csv")

    existing_csvs = set()
    if csv_root.exists():
        for csv_file in csv_root.glob("*.csv"):
            existing_csvs.add(csv_file.stem)

    # 筛选出尚未处理的 mzML 文件
    pending_paths = [path for path in all_paths if path.stem not in existing_csvs]

    total = len(all_paths)
    done = total - len(pending_paths)
    print(f"[Idempotent] Total mzML files: {total}, already completed: {done}, pending: {len(pending_paths)}")

    if not pending_paths:
        print("[Idempotent] All files already processed. Skipping entire pipeline.")
        return

    # 2) 逐文件处理：读取 -> ROI -> EIC -> 推断 -> 保存
    for file_idx, mzml_path in enumerate(pending_paths):
        print(f"\n[Processing] ({file_idx + 1}/{len(pending_paths)}) {mzml_path.name}")

        # 2a) 单文件 ROI + EIC
        eics = get_eics_for_single_file(
            mzml_path,
            ppm=int(args.ppm),
            roi_delta_mz=float(args.roi_delta_mz),
            required_points=int(args.roi_required_points),
            dropped_points=int(args.roi_dropped_points),
            min_nonzero_points=int(args.min_nonzero_points),
            min_height=float(args.min_height)
        )

        # 2b) 单文件推断（inference 接受 list[list[eic]] 格式）
        results = inference(eics_per_file=[eics], args=args)

        # 2c) 单文件保存
        save_single_file_result(
            results=results,
            eics_per_file=[eics],
            mzml_paths=[mzml_path],
            args=args
        )

        print(f"[Completed] {mzml_path.name} saved successfully.")

    print(f"\n[Done] Processed {len(pending_paths)} file(s). Total completed: {total}/{total}")


if __name__ == "__main__":
    args = get_args().parse_args()
    
    main(args)

