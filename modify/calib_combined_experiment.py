"""
calib_combined_experiment.py
============================
公式验证: P_calibrated = TS(P_raw) * (1 - lambda * UQ)
同时验证 Vessel12 (肺) 和 PortalVein (血管)
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_nifti(path: str) -> np.ndarray:
    data = nib.load(path).get_fdata()
    return np.nan_to_num(data).astype(np.float32)

def ece_score(probs, labels, n_bins=15):
    probs = np.clip(probs.ravel(), 0.0, 1.0)
    labels = labels.ravel().astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0: continue
        ece += mask.sum() / len(probs) * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)

def brier_score(probs, labels):
    return float(np.mean((np.clip(probs, 0.0, 1.0).ravel() - labels.ravel().astype(np.float32)) ** 2))

def fit_ts(prob_4d, labels_3d):
    """拟合最优温度 T"""
    if prob_4d.ndim == 4:
        p_safe = np.clip(prob_4d.reshape(-1, 2), 1e-7, 1 - 1e-7)
        logits = np.log(p_safe)
    else:
        p1 = np.clip(prob_4d.ravel(), 1e-7, 1 - 1e-7)
        logits = np.stack([np.log(1-p1), np.log(p1)], axis=1)
    
    y = labels_3d.ravel()
    # 抽样加速
    if len(y) > 1_000_000:
        idx = np.random.choice(len(y), 1_000_000, replace=False)
        logits_sub, y_sub = logits[idx], y[idx]
    else:
        logits_sub, y_sub = logits, y

    def nll(T):
        scaled = logits_sub / T
        log_p = scaled[:, 1] - np.log(np.exp(scaled[:, 0]) + np.exp(scaled[:, 1]) + 1e-12)
        return -np.mean(y_sub * log_p + (1 - y_sub) * np.log(1 - np.exp(log_p) + 1e-12))
    
    res = minimize_scalar(nll, bounds=(0.1, 5.0), method="bounded")
    return res.x, logits

# ─────────────────────────────────────────────
# 核心评测逻辑
# ─────────────────────────────────────────────

def run_eval(name, prob_path, gt_path, uq_path):
    print(f"\n评价任务: {name}")
    prob_data = load_nifti(prob_path)
    gt_data   = load_nifti(gt_path)
    uq_data   = load_nifti(uq_path)
    
    # 1. 拟合 TS
    t_opt, logits = fit_ts(prob_data, gt_data)
    p_ts = softmax(logits / t_opt, axis=1)[:, 1]
    p_raw = (prob_data[..., 1] if prob_data.ndim==4 else prob_data).ravel()
    y = gt_data.ravel()

    # 2. 计算基准 (Raw 和 TS)
    results = []
    for m, p in [("Raw", p_raw), (f"TS (T={t_opt:.2f})", p_ts)]:
        results.append({"Dataset": name, "Method": m, "ECE": ece_score(p, y), "Brier": brier_score(p, y), "Lambda": "-"})

    # 3. 组合实验: TS * (1 - lambda * UQ)
    for lb in [0.5, 1.0, 2.0, 5.0]:
        p_combined = np.clip(p_ts * (1.0 - lb * uq_data.ravel()), 0.0, 1.0)
        results.append({
            "Dataset": name,
            "Method": "TS + UQ Combined",
            "ECE": ece_score(p_combined, y),
            "Brier": brier_score(p_combined, y),
            "Lambda": lb
        })
    
    return results

# ─────────────────────────────────────────────
# 运行
# ─────────────────────────────────────────────

def main():
    all_res = []
    
    # 配置
    tasks = [
        ("Lung (Vessel12)", 
         "/home/DMT_dev/data/vessel12_01_3rd/vessel12_01_prob.nii.gz",
         "/home/DMT_dev/data/vessel12_01_3rd/vessel12_01_gt.nii.gz",
         "/home/DMT_dev/data/vessel12_01_3rd/vessel12_01_uq_exp_tau0.1.nii.gz"),
        
        ("Vessel (PortalVein)",
         "/home/DMT_dev/data/PortalVein/PortalVein_001_prob.nii.gz",
         "/home/DMT_dev/data/PortalVein/PortalVein_001_gt.nii.gz",
         "/home/DMT_dev/data/PortalVein/PortalVein_001_uq_4th.nii.gz")
    ]

    for name, p, g, u in tasks:
        all_res.extend(run_eval(name, p, g, u))

    df = pd.DataFrame(all_res)
    print("\n" + "="*80)
    print("双重校准 (TS + UQ) 实验汇总")
    print("="*80)
    print(df.to_string(index=False))
    df.to_csv("/home/DMT_dev/modify/calib_combined_final_results.csv", index=False)

if __name__ == "__main__":
    main()
