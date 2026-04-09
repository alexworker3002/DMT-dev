import numpy as np
from scipy.ndimage import gaussian_filter
import nibabel as nib
import time

def generate_hessian_uq_3d(prob_map, sigma=1.0):
    # Step 1: 高效计算 6 个独立二阶导数
    L_zz = gaussian_filter(prob_map, sigma=sigma, order=[2, 0, 0])
    L_yy = gaussian_filter(prob_map, sigma=sigma, order=[0, 2, 0])
    L_xx = gaussian_filter(prob_map, sigma=sigma, order=[0, 0, 2])
    L_zy = gaussian_filter(prob_map, sigma=sigma, order=[1, 1, 0])
    L_zx = gaussian_filter(prob_map, sigma=sigma, order=[1, 0, 1])
    L_yx = gaussian_filter(prob_map, sigma=sigma, order=[0, 1, 1])

    # Step 2: 提取背景掩码，仅对有效区域计算（极大加速）
    geometry_mask = prob_map > 0.05
    
    # 提取有效的一维像素数组
    a = L_zz[geometry_mask]
    b = L_yy[geometry_mask]
    c = L_xx[geometry_mask]
    d = L_zy[geometry_mask]
    e = L_zx[geometry_mask]
    f = L_yx[geometry_mask]
    
    del L_zz, L_yy, L_xx, L_zy, L_zx, L_yx
    
    # Step 3: 使用解析法直接计算 3x3 对称矩阵特征值，强制 float32
    p1 = a + b + c
    p2 = a * b + b * c + c * a - d**2 - f**2 - e**2
    p3 = a * b * c + 2 * d * e * f - a * f**2 - b * e**2 - c * d**2
    
    del a, b, c, d, e, f
    
    q = np.float32((p1**2 - 3 * p2) / 9.0)
    q = np.clip(q, 0.0, None)
    r = np.float32((2 * p1**3 - 9 * p1 * p2 + 27 * p3) / 54.0)
    
    q_sqrt = np.sqrt(q, dtype=np.float32)
    q32 = q * q_sqrt
    
    mask = q32 > 1e-7
    ratio = np.zeros_like(r)
    ratio[mask] = r[mask] / q32[mask]
    ratio = np.clip(ratio, -1.0, 1.0)
    
    theta = np.arccos(ratio, dtype=np.float32)
    
    l1 = p1 / 3.0 + 2.0 * q_sqrt * np.cos(theta / 3.0, dtype=np.float32)
    l2 = p1 / 3.0 + 2.0 * q_sqrt * np.cos((theta - 2 * np.pi) / 3.0, dtype=np.float32)
    l3 = p1 / 3.0 + 2.0 * q_sqrt * np.cos((theta + 2 * np.pi) / 3.0, dtype=np.float32)
    
    del p1, p2, p3, q, r, q_sqrt, q32, mask, ratio, theta
    
    # Step 4: 提取鞍点分数
    saddle_mask = (l1 > 0) & (l2 < 0) & (l3 < 0)
    saddle_score_1d = l1 * saddle_mask
    saddle_score_1d = np.nan_to_num(saddle_score_1d, nan=0.0)
    
    if np.any(saddle_score_1d > 0):
        max_score = np.percentile(saddle_score_1d[saddle_score_1d > 0], 99)
    else:
        max_score = 1e-9
        
    saddle_score_1d = np.clip(saddle_score_1d / max_score, 0.0, 1.0)

    # Step 5: 熵调制与掩码融合
    p = prob_map[geometry_mask]
    raw_entropy = -p * np.log(p + 1e-9, dtype=np.float32) - (1 - p) * np.log(1 - p + 1e-9, dtype=np.float32)
    entropy_weight = (raw_entropy / np.float32(np.log(2.0))) ** 2.0

    uq_1d = saddle_score_1d * entropy_weight

    # 写回 3D 结果图
    uq_map = np.zeros_like(prob_map)
    uq_map[geometry_mask] = uq_1d

    return uq_map

if __name__ == "__main__":
    file_path = "/home/DMT_dev/data/PortalVein/PortalVein_001_prob.nii.gz"
    print(f"Loading data from {file_path}...")
    try:
        img = nib.load(file_path)
        prob_map = img.get_fdata().astype(np.float32)
        if prob_map.ndim == 4:
            prob_map = prob_map[..., 1]
        print(f"Loaded prob_map with shape: {prob_map.shape}")
        
        start_time = time.time()
        uq_map = generate_hessian_uq_3d(prob_map, sigma=1.0)
        end_time = time.time()
        
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        print(f"Output UQ map shape: {uq_map.shape}, min: {uq_map.min():.4f}, max: {uq_map.max():.4f}")
        
        # 保存 UQ map
        import os
        output_path = file_path.replace("_prob.nii.gz", "_uq_map.nii.gz")
        out_img = nib.Nifti1Image(uq_map.astype(np.float32), img.affine, img.header)
        nib.save(out_img, output_path)
        print(f"Saved UQ map to: {output_path}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
