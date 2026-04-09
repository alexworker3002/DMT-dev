"""
calib_uq_eval.py
=================
实验目标:
  1. 读取 prob_map（原始 softmax 概率图）和多个 uq_map
  2. 计算 calibrated_prob = prob_map * (1 - λ * uq_map)，λ = 0.5
  3. 对 calibrated_prob / 原始 prob_map / Temperature Scaling 处理后的 prob_map
     分别计算 ECE（Expected Calibration Error）和 Brier Score
  4. 打印对比表，并保存结果到 CSV
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
DATA_DIR = "/home/DMT_dev/data/vessel12_01_3rd"

PROB_FILE = os.path.join(DATA_DIR, "vessel12_01_prob.nii.gz")
GT_FILE   = os.path.join(DATA_DIR, "vessel12_01_gt.nii.gz")

# 所有需要评测的 UQ 文件名
UQ_FILES = [
    "vessel12_01_uq_4th.nii.gz",
    "vessel12_01_uq_exp_tau0.1.nii.gz",
]

LAMBDA = 0.5         # UQ 抑制系数
N_BINS = 15          # ECE 分箱数
OUT_CSV = os.path.join(os.path.dirname(__file__), "calib_uq_results.csv")


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_nifti(path: str) -> np.ndarray:
    """读取 NIfTI 文件，返回 float32 数组。"""
    return nib.load(path).get_fdata().astype(np.float32)


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error（等频宽分箱）。
    probs:  shape (N,) 预测正类概率，范围 [0, 1]
    labels: shape (N,) 二值真实标签 {0, 1}
    """
    probs  = np.clip(probs.ravel(),  0.0, 1.0)
    labels = labels.ravel().astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    N    = len(probs)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc  = labels[mask].mean()
        ece += mask.sum() / N * abs(bin_conf - bin_acc)

    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier Score（均方误差），越小越好。"""
    probs  = np.clip(probs.ravel(),  0.0, 1.0)
    labels = labels.ravel().astype(np.float32)
    return float(np.mean((probs - labels) ** 2))


def temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Temperature Scaling 校准。
    logits: shape (N, 2) 原始 logit（从 softmax 逆推）
    labels: shape (N,)   二值真实标签
    返回校准后的正类概率，shape (N,)
    """
    # 用 NLL 作为优化目标，在 [0.1, 10] 范围内做 1D 标量搜索
    def nll(T):
        scaled = logits / T
        # log_softmax for class 1
        log_p = scaled[:, 1] - np.log(np.exp(scaled[:, 0]) + np.exp(scaled[:, 1]))
        return -np.mean(labels * log_p + (1 - labels) * np.log(
            1 - np.exp(log_p) + 1e-12))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T_opt  = result.x
    print(f"  → Temperature Scaling: T* = {T_opt:.4f}")

    scaled_logits = logits / T_opt
    probs = softmax(scaled_logits, axis=1)[:, 1]
    return probs


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("UQ Calibration Evaluation Experiment")
    print("=" * 60)

    # ── 1. 加载数据 ──────────────────────────────
    print("\n[1/4] Loading data ...")
    prob_4d = load_nifti(PROB_FILE)   # (H, W, D, 2)
    gt_3d   = load_nifti(GT_FILE)     # (H, W, D)

    # 取前景通道（class 1）
    prob_fg = prob_4d[..., 1]         # (H, W, D)，范围 [0, 1]

    # 展平为 1D
    p_flat = prob_fg.ravel()
    y_flat = gt_3d.ravel()

    print(f"  Volume  : {prob_fg.shape}")
    print(f"  Voxels  : {len(p_flat):,}")
    print(f"  GT ratio: {y_flat.mean():.4%}")

    # ── 2. 计算 logits（Temperature Scaling 需要）──
    # logit = log(p / (1-p))；为数值稳定，先 clip
    p_safe = np.clip(prob_4d.reshape(-1, 2), 1e-7, 1 - 1e-7)
    logits = np.log(p_safe)           # 近似 logits（差一个归一化常数，TS 不影响）

    y_flat_ts = y_flat  # 同一 flatten 顺序

    # ── 3. Temperature Scaling ───────────────────
    print("\n[2/4] Fitting Temperature Scaling ...")
    # 为加速，使用随机子采样（最多 1M 个体素）
    N_SAMPLE = 1_000_000
    if len(p_flat) > N_SAMPLE:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(p_flat), N_SAMPLE, replace=False)
        logits_sub = logits[idx]
        y_sub      = y_flat[idx]
    else:
        logits_sub = logits
        y_sub      = y_flat

    ts_probs_sub = temperature_scaling(logits_sub, y_sub)

    # 全量 TS 概率（用最优 T 直接缩放全量 logits）
    # 重新拿到 T*
    def nll_full(T):
        scaled = logits_sub / T
        log_p  = scaled[:, 1] - np.log(np.exp(scaled[:, 0]) + np.exp(scaled[:, 1]))
        return -np.mean(y_sub * log_p + (1 - y_sub) * np.log(
            1 - np.exp(log_p) + 1e-12))

    res   = minimize_scalar(nll_full, bounds=(0.1, 10.0), method="bounded")
    T_opt = res.x
    ts_probs_full = softmax(logits / T_opt, axis=1)[:, 1]

    # ── 4. 评测各方案 ────────────────────────────
    print("\n[3/4] Computing ECE & Brier Score ...")

    results = []

    # --- 基线：原始 prob_map ---
    ece_raw    = ece_score(p_flat, y_flat, N_BINS)
    brier_raw  = brier_score(p_flat, y_flat)
    results.append({
        "Method": "Raw prob_map",
        "ECE": ece_raw,
        "Brier": brier_raw,
        "Lambda": "-",
        "UQ_file": "-",
    })
    print(f"  [Raw]            ECE={ece_raw:.6f}  Brier={brier_raw:.6f}")

    # --- Temperature Scaling ---
    ece_ts    = ece_score(ts_probs_full, y_flat, N_BINS)
    brier_ts  = brier_score(ts_probs_full, y_flat)
    results.append({
        "Method": f"Temperature Scaling (T={T_opt:.3f})",
        "ECE": ece_ts,
        "Brier": brier_ts,
        "Lambda": "-",
        "UQ_file": "-",
    })
    print(f"  [Temp Scaling]   ECE={ece_ts:.6f}  Brier={brier_ts:.6f}  T={T_opt:.4f}")

    # --- UQ 校准 (遍历 lambda) ---
    LAMBDA_LIST = [0.1, 0.5, 1.0, 2.0, 5.0]

    for uq_fname in UQ_FILES:
        uq_path = os.path.join(DATA_DIR, uq_fname)
        if not os.path.exists(uq_path):
            print(f"  [SKIP] {uq_fname} not found.")
            continue

        uq_3d   = load_nifti(uq_path) 
        uq_flat = uq_3d.ravel()

        for lb in LAMBDA_LIST:
            # calibrated_prob = prob * (1 - λ * uq)
            calib_p = np.clip(p_flat * (1.0 - lb * uq_flat), 0.0, 1.0)

            ece_c   = ece_score(calib_p, y_flat, N_BINS)
            brier_c = brier_score(calib_p, y_flat)

            label = f"UQ-Calib (λ={lb}, {uq_fname.replace('vessel12_01_', '')})"
            results.append({
                "Method": label,
                "ECE": ece_c,
                "Brier": brier_c,
                "Lambda": lb,
                "UQ_file": uq_fname,
            })
            print(f"  [λ={lb:3.1f} | {uq_fname[:20]}...]  ECE={ece_c:.6f}  Brier={brier_c:.6f}")

    # ── 5. 输出汇总 ──────────────────────────────
    print("\n[4/4] Summary")
    print("-" * 60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    df.to_csv(OUT_CSV, index=False)
    print(f"\nResults saved to: {OUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
