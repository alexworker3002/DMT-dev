import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.ndimage import gaussian_filter
import time

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)

def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs  = np.clip(probs.ravel(), 0.0, 1.0)
    labels = labels.ravel().astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0: continue
        bin_conf = probs[mask].mean()
        bin_acc  = labels[mask].mean()
        ece += mask.sum() / N * abs(bin_conf - bin_acc)
    return float(ece)

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs  = np.clip(probs.ravel(), 0.0, 1.0)
    labels = labels.ravel().astype(np.float32)
    return float(np.mean((probs - labels) ** 2))

def temperature_scaling_fit(logits: np.ndarray, labels: np.ndarray):
    def nll(T):
        scaled = logits / T
        # 对数似然计算
        # exp_scaled = np.exp(scaled)
        # sum_exp = np.sum(exp_scaled, axis=1)
        # log_p = scaled[:, 1] - np.log(sum_exp + 1e-12)
        # return -np.mean(labels * log_p + (1 - labels) * np.log(1 - np.exp(log_p) + 1e-12))
        
        # 稳定的 NLL 实现
        max_logits = np.max(scaled, axis=1, keepdims=True)
        log_sum_exp = max_logits.squeeze() + np.log(np.sum(np.exp(scaled - max_logits), axis=1) + 1e-12)
        log_p = scaled[:, 1] - log_sum_exp
        return -np.mean(labels * log_p + (1 - labels) * np.log(1 - np.exp(log_p) + 1e-12))
    
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return result.x

# ─────────────────────────────────────────────
# 实验主函数
# ─────────────────────────────────────────────

def evaluate_calibration(case_name, prob_path, gt_path, uq_path):
    print(f"\nEvaluating: {case_name}")
    print(f"  - Prob: {prob_path}")
    print(f"  - UQ: {uq_path}")
    
    # 1. 加载
    prob_4d = load_nifti(prob_path)
    gt_3d   = load_nifti(gt_path)
    uq_data = load_nifti(uq_path)
    
    if prob_4d.ndim == 4:
        prob_fg = prob_4d[..., 1]
        p_safe  = np.clip(prob_4d.reshape(-1, 2), 1e-7, 1 - 1e-7)
        logits  = np.log(p_safe)
    else:
        prob_fg = prob_4d
        p1 = np.clip(prob_fg.ravel(), 1e-7, 1 - 1e-7)
        logits = np.stack([np.log(1-p1), np.log(p1)], axis=1)

    p_flat = prob_fg.ravel()
    y_flat = gt_3d.ravel()
    uq_flat = uq_data.ravel()
    
    # 2. TS 拟合
    print("  Fitting Temperature Scaling...")
    N_SAMPLE = 1_000_000
    if len(p_flat) > N_SAMPLE:
        idx = np.random.choice(len(p_flat), N_SAMPLE, replace=False)
        T_opt = temperature_scaling_fit(logits[idx], y_flat[idx])
    else:
        T_opt = temperature_scaling_fit(logits, y_flat)
    
    ts_probs = softmax(logits / T_opt, axis=1)[:, 1]

    # 3. 计算指标
    results = []
    
    # Raw
    results.append({
        "Case": case_name,
        "Method": "Raw",
        "ECE": ece_score(p_flat, y_flat),
        "Brier": brier_score(p_flat, y_flat),
        "Lambda": "-"
    })
    
    # TS
    results.append({
        "Case": case_name,
        "Method": f"TS (T={T_opt:.2f})",
        "ECE": ece_score(ts_probs, y_flat),
        "Brier": brier_score(ts_probs, y_flat),
        "Lambda": "-"
    })
    
    # HL-PH Halo Calib (NEW FORMULA)
    print("  Applying Gaussian Diffusion (sigma=2.0) and Halo Penalty...")
    uq_halo = gaussian_filter(uq_data, sigma=2.0)
    uq_halo_flat = uq_halo.reshape(-1)
    
    LAMBDA_LIST = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for lb in LAMBDA_LIST:
        # P_calib = P_orig - lb * UQ_halo * (P_orig)^2
        calib_p = np.clip(p_flat - lb * uq_halo_flat * (p_flat ** 2), 0.0, 1.0)
        results.append({
            "Case": case_name,
            "Method": f"HL-PH Halo",
            "ECE": ece_score(calib_p, y_flat),
            "Brier": brier_score(calib_p, y_flat),
            "Lambda": lb
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    configs = [
        {
            "name": "PortalVein_001",
            "prob": "/home/DMT_dev/data/PortalVein/PortalVein_001_prob.nii.gz",
            "gt": "/home/DMT_dev/data/PortalVein/PortalVein_001_gt.nii.gz",
            "uq": "/home/DMT_dev/data/PortalVein/PortalVein_001_hl_ph_uq.nii.gz"
        },
        {
            "name": "Vessel12_01",
            "prob": "/home/DMT_dev/data/vessel12_01_3rd/vessel12_01_prob.nii.gz",
            "gt": "/home/DMT_dev/data/vessel12_01_3rd/vessel12_01_gt.nii.gz",
            "uq": "/home/DMT_dev/data/vessel12_01_3rd/vessel12_01_hl_ph_uq.nii.gz"
        }
    ]
    
    all_res = []
    for cfg in configs:
        all_res.append(evaluate_calibration(cfg['name'], cfg['prob'], cfg['gt'], cfg['uq']))
        
    final_df = pd.concat(all_res, ignore_index=True)
    print("\n" + "="*80)
    print("CALIBRATION BENCHMARK RESULTS (HL-PH)")
    print("="*80)
    print(final_df.to_string(index=False))
    
    # 保存结果
    out_file = "/home/DMT_dev/hl_ph_uq/calibration_results_hlph.csv"
    final_df.to_csv(out_file, index=False)
    print(f"\nFinal report saved to: {out_file}")
