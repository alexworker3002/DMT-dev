"""
calib_portal_vein_eval.py
=========================
实验目标:
  在 PortalVein 数据集上重复校准对比实验。
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
DATA_DIR = "/home/DMT_dev/data/PortalVein"

PROB_FILE = os.path.join(DATA_DIR, "PortalVein_001_prob.nii.gz")
GT_FILE   = os.path.join(DATA_DIR, "PortalVein_001_gt.nii.gz")
UQ_FILES  = ["PortalVein_001_uq_4th.nii.gz"]

LAMBDA = 0.5
N_BINS = 15
OUT_CSV = os.path.join("/home/DMT_dev/modify", "calib_portalvein_results.csv")

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
        log_p = scaled[:, 1] - np.log(np.exp(scaled[:, 0]) + np.exp(scaled[:, 1]) + 1e-12)
        return -np.mean(labels * log_p + (1 - labels) * np.log(1 - np.exp(log_p) + 1e-12))
    
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return result.x

# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PortalVein UQ Calibration Evaluation")
    print("=" * 60)

    # 1. 加载
    print("[1/4] Loading PortalVein data...")
    prob_4d = load_nifti(PROB_FILE)
    gt_3d   = load_nifti(GT_FILE)
    
    # 兼容性检查：如果是 3D prob 则直接用，如果是 4D 则取通道 1
    if prob_4d.ndim == 4:
        prob_fg = prob_4d[..., 1]
        p_safe  = np.clip(prob_4d.reshape(-1, 2), 1e-7, 1 - 1e-7)
        logits  = np.log(p_safe)
    else:
        prob_fg = prob_4d
        # 模拟 logits
        p1 = np.clip(prob_fg.ravel(), 1e-7, 1 - 1e-7)
        logits = np.stack([np.log(1-p1), np.log(p1)], axis=1)

    p_flat = prob_fg.ravel()
    y_flat = gt_3d.ravel()

    # 2. TS 拟合
    print("[2/4] Fitting Temperature Scaling...")
    N_SAMPLE = 1_000_000
    if len(p_flat) > N_SAMPLE:
        idx = np.random.choice(len(p_flat), N_SAMPLE, replace=False)
        T_opt = temperature_scaling_fit(logits[idx], y_flat[idx])
    else:
        T_opt = temperature_scaling_fit(logits, y_flat)
    
    ts_probs = softmax(logits / T_opt, axis=1)[:, 1]

    # 3. 评测
    print("[3/4] Computing Metrics...")
    results = []

    # Raw
    results.append({
        "Method": "Raw",
        "ECE": ece_score(p_flat, y_flat, N_BINS),
        "Brier": brier_score(p_flat, y_flat),
        "T": 1.0
    })

    # TS
    results.append({
        "Method": f"TempScaling (T={T_opt:.3f})",
        "ECE": ece_score(ts_probs, y_flat, N_BINS),
        "Brier": brier_score(ts_probs, y_flat),
        "T": T_opt
    })

    # UQ Calib (遍历多个 lambda)
    LAMBDA_LIST = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for uq_fname in UQ_FILES:
        uq_path = os.path.join(DATA_DIR, uq_fname)
        if not os.path.exists(uq_path): continue
        
        uq_data = load_nifti(uq_path)
        uq_data = np.nan_to_num(uq_data, nan=0.0) 
        uq_flat = uq_data.ravel()
        
        for lb in LAMBDA_LIST:
            calib_p = np.clip(p_flat * (1.0 - lb * uq_flat), 0.0, 1.0)
            
            ece_val = ece_score(calib_p, y_flat, N_BINS)
            brier_val = brier_score(calib_p, y_flat)
            
            results.append({
                "Method": f"UQ-Calib (λ={lb})",
                "ECE": ece_val,
                "Brier": brier_val,
                "T": "-"
            })
            print(f"  [λ={lb:4.1f}] ECE={ece_val:.6f}  Brier={brier_val:.6f}")

    # 4. 输出
    print("\n[4/4] Summary Results")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}")

if __name__ == "__main__":
    main()
