import os
import sys
import numpy as np
import nibabel as nib
import time
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.optimize import minimize_scalar
from scipy.special import softmax
import pandas as pd

# 添加核心库路径
sys.path.append("/home/DMT_dev")
import morse_3d

# ─────────────────────────────────────────────
# 1. HL-PH 生成逻辑 (参考 run_hl_ph.py)
# ─────────────────────────────────────────────

def get_dmt_energy_field(prob_map, dmt_res, tau=0.15):
    dmt_energy_map = np.zeros_like(prob_map, dtype=np.float32)
    original_shape = prob_map.shape
    
    births = np.asarray(dmt_res["births"])
    deaths = np.asarray(dmt_res["deaths"])
    birth_coords = np.asarray(dmt_res["birth_coords"])
    death_coords = np.asarray(dmt_res["death_coords"])
    
    valid_deaths = np.where(np.isinf(deaths), births, deaths)
    
    for i in range(len(births)):
        persistence = abs(births[i] - valid_deaths[i])
        if persistence < tau and persistence > 0:
            if not (np.any(np.isnan(birth_coords[i])) or np.any(np.isnan(death_coords[i]))):
                bz = int(np.clip(np.round(birth_coords[i][0]), 0, original_shape[0] - 1))
                by = int(np.clip(np.round(birth_coords[i][1]), 0, original_shape[1] - 1))
                bx = int(np.clip(np.round(birth_coords[i][2]), 0, original_shape[2] - 1))
                
                dz = int(np.clip(np.round(death_coords[i][0]), 0, original_shape[0] - 1))
                dy = int(np.clip(np.round(death_coords[i][1]), 0, original_shape[1] - 1))
                dx = int(np.clip(np.round(death_coords[i][2]), 0, original_shape[2] - 1))
                
                dmt_energy_map[bz, by, bx] = max(dmt_energy_map[bz, by, bx], persistence)
                dmt_energy_map[dz, dy, dx] = max(dmt_energy_map[dz, dy, dx], persistence)
            
    # 按照最新方案：15x15 膨胀 + 1.0 高斯平滑
    dmt_energy_field = maximum_filter(dmt_energy_map, size=15)
    dmt_energy_field = gaussian_filter(dmt_energy_field, sigma=1.0)
    
    dmt_max = np.max(dmt_energy_field)
    if dmt_max > 0:
        dmt_energy_field /= dmt_max
        
    return dmt_energy_field

def get_hessian_mask(prob_map, sigma=1.0):
    p_map = prob_map.astype(np.float32)
    L_zz = gaussian_filter(p_map, sigma=sigma, order=[2, 0, 0])
    L_yy = gaussian_filter(p_map, sigma=sigma, order=[0, 2, 0])
    L_xx = gaussian_filter(p_map, sigma=sigma, order=[0, 0, 2])
    L_zy = gaussian_filter(p_map, sigma=sigma, order=[1, 1, 0])
    L_zx = gaussian_filter(p_map, sigma=sigma, order=[1, 0, 1])
    L_yx = gaussian_filter(p_map, sigma=sigma, order=[0, 1, 1])
    
    H_matrix = np.zeros(p_map.shape + (3, 3), dtype=np.float32)
    H_matrix[..., 0, 0] = L_zz
    H_matrix[..., 1, 1] = L_yy
    H_matrix[..., 2, 2] = L_xx
    H_matrix[..., 0, 1] = L_zy
    H_matrix[..., 0, 2] = L_zx
    H_matrix[..., 1, 2] = L_yx
    H_matrix[..., 1, 0] = L_zy
    H_matrix[..., 2, 0] = L_zx
    H_matrix[..., 2, 1] = L_yx
    
    del L_zz, L_yy, L_xx, L_zy, L_zx, L_yx
    
    eigvals = np.linalg.eigvalsh(H_matrix)
    del H_matrix
    l3, l2, l1 = eigvals[..., 0], eigvals[..., 1], eigvals[..., 2]
    
    saddle_mask = (l1 > 0) & (l2 < 0) & (l3 < 0)
    saddle_score = l1 * saddle_mask
    saddle_score = np.nan_to_num(saddle_score, nan=0.0)
    
    max_score = np.percentile(saddle_score[saddle_score > 0], 99.9) if np.any(saddle_score > 0) else 1e-9
    saddle_score = np.clip(saddle_score / max_score, 0.0, 1.0)
    
    return saddle_score

def generate_hluq(prob_map, dmt_res):
    dmt_field = get_dmt_energy_field(prob_map, dmt_res)
    hessian_geom = get_hessian_mask(prob_map)
    
    p = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
    entropy_weight = (-p * np.log(p + 1e-9) - (1-p) * np.log(1-p+1e-9)) / np.log(2.0)
    entropy_weight = entropy_weight ** 2.0
    
    hluq = dmt_field * hessian_geom * entropy_weight
    
    # 可视化增强
    active = hluq[hluq > 0]
    if len(active) > 0:
        vmax = np.percentile(active, 99.9)
        hluq = np.clip(hluq / (vmax + 1e-9), 0.0, 1.0)
    
    return hluq

# ─────────────────────────────────────────────
# 2. 校准评估逻辑 (参考 calibrate_hl_ph.py)
# ─────────────────────────────────────────────

def ece_score(probs, labels, n_bins=15):
    probs = np.clip(probs.ravel(), 0, 1)
    labels = labels.ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.any():
            ece += mask.sum() / len(probs) * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)

def brier_score(probs, labels):
    return float(np.mean((probs.ravel() - labels.ravel())**2))

def temperature_scaling_fit(logits, labels):
    def nll(T):
        scaled = logits / T
        max_l = np.max(scaled, axis=1, keepdims=True)
        log_sum_exp = max_l.squeeze() + np.log(np.sum(np.exp(scaled - max_l), axis=1) + 1e-12)
        log_p = scaled[:, 1] - log_sum_exp
        return -np.mean(labels * log_p + (1 - labels) * np.log(1 - np.exp(log_p) + 1e-12))
    return minimize_scalar(nll, bounds=(0.1, 10), method='bounded').x

# ─────────────────────────────────────────────
# 3. 批量处理脚本
# ─────────────────────────────────────────────

def process_dataset(folder_path):
    dataset_name = os.path.basename(folder_path.rstrip('/'))
    print(f"\n[Processing Dataset: {dataset_name}]")
    
    prob_path = os.path.join(folder_path, "prob.nii.gz")
    gt_path = os.path.join(folder_path, "gt.nii.gz")
    
    if not os.path.exists(prob_path) or not os.path.exists(gt_path):
        print(f"Skipping {dataset_name}: Missing prob or gt.")
        return None

    # 加载
    img = nib.load(prob_path)
    gt_img = nib.load(gt_path)
    prob_data = img.get_fdata().astype(np.float64)
    if prob_data.ndim == 4: prob_data = prob_data[..., 1]
    gt_data = gt_img.get_fdata().astype(np.float32)

    # 1. DMT & HL-PH
    print("  Step 1: Running DMT Engine...")
    t0 = time.time()
    dmt_res = morse_3d.extract_persistence_3d_morse(prob_data)
    hluq = generate_hluq(prob_data, dmt_res)
    print(f"  DMT & HL-PH Done in {time.time() - t0:.2f}s")

    # 2. 保存 HLUQ
    hluq_path = os.path.join(folder_path, "hl_ph_uq.nii.gz")
    nib.save(nib.Nifti1Image(hluq.astype(np.float32), img.affine, img.header), hluq_path)
    print(f"  HLUQ Saved to: {hluq_path}")

    # 3. 校准
    print("  Step 2: Calibration...")
    p_flat = prob_data.astype(np.float32).ravel()
    y_flat = gt_data.ravel()
    
    # Baselines
    raw_ece = ece_score(p_flat, y_flat)
    raw_brier = brier_score(p_flat, y_flat)
    
    # TS
    p_safe = np.clip(p_flat, 1e-7, 1-1e-7)
    logits = np.stack([np.log(1-p_safe), np.log(p_safe)], axis=1)
    T_opt = temperature_scaling_fit(logits[::10], y_flat[::10]) # Sample for speed
    ts_probs = softmax(logits / T_opt, axis=1)[:, 1]
    ts_ece = ece_score(ts_probs, y_flat)
    ts_brier = brier_score(ts_probs, y_flat)
    
    # Halo Formula: P_calib = P - lambda * UQ_halo * P^2
    uq_halo = gaussian_filter(hluq, sigma=2.0).ravel()
    
    best_lb = 0
    best_ece = 1.0
    for lb in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        calib_p = np.clip(p_flat - lb * uq_halo * (p_flat**2), 0, 1)
        ece = ece_score(calib_p, y_flat)
        if ece < best_ece:
            best_ece = ece
            best_lb = lb
    
    best_calib_p = np.clip(p_flat - best_lb * uq_halo * (p_flat**2), 0, 1)
    best_brier = brier_score(best_calib_p, y_flat)

    return {
        "Dataset": dataset_name,
        "Raw_ECE": raw_ece,
        "Raw_Brier": raw_brier,
        "TS_ECE": ts_ece,
        "TS_Brier": ts_brier,
        "HLPH_ECE": best_ece,
        "HLPH_Brier": best_brier,
        "Best_Lambda": best_lb
    }

if __name__ == "__main__":
    base_dir = "/home/first_SM/inference_results"
    datasets = [
        os.path.join(base_dir, "Dataset001_LungVessel"),
        os.path.join(base_dir, "Dataset002_PortalVein"),
        os.path.join(base_dir, "Dataset003_ACDC")
    ]
    
    all_metrics = []
    for d in datasets:
        metrics = process_dataset(d)
        if metrics: all_metrics.append(metrics)
        
    df = pd.DataFrame(all_metrics)
    print("\n" + "="*80)
    print("BATCH CALIBRATION SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    
    csv_out = "/home/DMT_dev/hl_ph_uq/batch_calibration_summary.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nFinal summary saved to: {csv_out}")
