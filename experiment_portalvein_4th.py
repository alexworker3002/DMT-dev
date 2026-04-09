import numpy as np
import nibabel as nib
import morse_3d
import time
import os
import sys

def generate_topological_uq_map(dmt_res: dict, prob_map: np.ndarray, tau: float = 0.15, sigma: float = 2.0, gamma: float = 2.0) -> np.ndarray:
    """
    基于离散莫尔斯理论、L1持久度能量和宽带概率门控生成 3D 拓扑不确定性 (Topological UQ) 热力图。
    集成 4th 版本的优化：Sigma 瘦身、几何掩码、熵调制。
    """
    print("  提取并过滤幻觉特征...")
    original_shape = prob_map.shape
    births = np.nan_to_num(np.asarray(dmt_res["births"]))
    deaths = np.nan_to_num(np.asarray(dmt_res["deaths"]))
    birth_coords = np.nan_to_num(np.asarray(dmt_res["birth_coords"]))
    death_coords = np.nan_to_num(np.asarray(dmt_res["death_coords"]))
    
    valid_deaths = np.where(np.isinf(deaths), births, deaths)
    
    # 1. 过滤拓扑幻觉 (持久度 < tau)
    persistence = births - valid_deaths
    hallucination_mask = persistence < tau
    
    h_birth_coords = birth_coords[hallucination_mask]
    h_death_coords = death_coords[hallucination_mask]
    h_births = births[hallucination_mask]
    h_deaths = valid_deaths[hallucination_mask]
    
    if len(h_birth_coords) == 0:
        print("  幻觉特征数量为 0。")
        return np.zeros(original_shape, dtype=np.float32)
        
    # 2A. 计算 Wasserstein 拓扑势能
    l1_persistence = (h_births - h_deaths)
    spatial_dist = np.linalg.norm(h_birth_coords - h_death_coords, axis=1)
    geometric_scale = np.clip(spatial_dist, 1.0, None) 
    energy = l1_persistence * geometric_scale
    
    # 2B. 指数级熵调制
    cz = np.clip(np.floor(h_birth_coords[:, 0]).astype(int), 0, original_shape[0] - 1)
    cy = np.clip(np.floor(h_birth_coords[:, 1]).astype(int), 0, original_shape[1] - 1)
    cx = np.clip(np.floor(h_birth_coords[:, 2]).astype(int), 0, original_shape[2] - 1)
    p = prob_map[cz, cy, cx]
    
    raw_entropy = -p * np.log(p + 1e-9) - (1 - p) * np.log(1 - p + 1e-9)
    # 确保底数为非负数，防止非整数次幂产生 NaN
    entropy_weight = (np.maximum(raw_entropy, 0.0) / (np.log(2.0) + 1e-12)) ** gamma
    
    weights = energy * entropy_weight
    all_weights = np.concatenate((weights, weights))

    all_coords = np.vstack((h_birth_coords, h_death_coords))
    all_coords = all_coords - 1.0 
    
    print(f"  判定为幻觉的特征数: {len(h_births)} (展开为 {len(all_coords)} 个物理映射点)")
    print(f"  生成高斯扩散热力图 (sigma={sigma})...")
    
    # 3. 极速局部 3D 高斯扩散
    uq_map = np.zeros(original_shape, dtype=np.float32)
    Z_max, Y_max, X_max = original_shape
    radius = int(3 * sigma)
    var_term = 2.0 * (sigma ** 2)
    
    for i, coord in enumerate(all_coords):
        cz_coord, cy_coord, cx_coord = coord
        w_i = all_weights[i]
        if w_i < 1e-6:
            continue
            
        z_min = max(0, int(np.floor(cz_coord - radius)))
        z_max_idx = min(Z_max, int(np.ceil(cz_coord + radius)) + 1)
        y_min = max(0, int(np.floor(cy_coord - radius)))
        y_max_idx = min(Y_max, int(np.ceil(cy_coord + radius)) + 1)
        x_min = max(0, int(np.floor(cx_coord - radius)))
        x_max_idx = min(X_max, int(np.ceil(cx_coord + radius)) + 1)
        
        grid_z, grid_y, grid_x = np.meshgrid(
            np.arange(z_min, z_max_idx),
            np.arange(y_min, y_max_idx),
            np.arange(x_min, x_max_idx),
            indexing='ij'
        )
        dist_sq = (grid_z - cz_coord)**2 + (grid_y - cy_coord)**2 + (grid_x - cx_coord)**2
        uq_map[z_min:z_max_idx, y_min:y_max_idx, x_min:x_max_idx] += w_i * np.exp(-dist_sq / var_term)
    
    # 4. 百分位拉伸 (使用 nanpercentile 以防万一)
    active_pixels = uq_map[uq_map > 1e-9]
    if len(active_pixels) > 0:
        vmax = np.nanpercentile(active_pixels, 99)
        if np.isnan(vmax) or vmax <= 0:
            vmax = np.max(uq_map) + 1e-9
        uq_map = np.clip(uq_map / (vmax + 1e-9), 0.0, 1.0)

    # 5. 空间几何掩码 (Geometry Masking)
    geometry_mask = (prob_map > 0.05).astype(np.float32)
    uq_map = np.nan_to_num(uq_map) * geometry_mask

    return uq_map

def main():
    root_dir = "/home/DMT_dev/data/PortalVein"
    input_path = os.path.join(root_dir, "PortalVein_001_prob.nii.gz")
    
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return
        
    print(f"1. 加载概率图数据: {input_path}")
    img = nib.load(input_path)
    data = img.get_fdata()
    
    # 自动处理多通道情况 (通常背景在0，前景在1)
    if len(data.shape) == 4 and data.shape[-1] >= 2:
        prob_map = data[..., 1]
    else:
        prob_map = data

    # 模拟 3rd 步骤：量化到 5 位小数以加速 DMT 并提升鲁棒性
    print("  正在执行量化 (5位小数)...")
    prob_map_q = np.round(prob_map, 5).astype(np.float64)
    
    # 保存量化后的中间文件
    quantized_path = os.path.join(root_dir, "PortalVein_001_prob_quantized.nii.gz")
    nib.save(nib.Nifti1Image(prob_map_q, img.affine, img.header), quantized_path)
    print(f"  量化概率图已保存: {quantized_path}")

    # 2. 运行 DMT 引擎
    print("\n2. 运行 C++ DMT 引擎计算持久同调...")
    t0 = time.time()
    dmt_res = morse_3d.extract_persistence_3d_morse(prob_map_q)
    t1 = time.time()
    print(f"  DMT 引擎耗时: {t1 - t0:.2f} 秒, 特征数: {len(dmt_res['births'])}")
    
    # 3. 生成 4th 逻辑 UQ 热力图
    print("\n3. 使用 4th 逻辑生成针对性 UQ 热力图 (Sigma=1.0, Gamma=2.0, Geometry Masking)...")
    t2 = time.time()
    uq_heatmap = generate_topological_uq_map(
        dmt_res=dmt_res,
        prob_map=prob_map_q,
        tau=0.10, # 采用 3rd 实验中效果较好的宽容度
        sigma=1.0,
        gamma=1.2
    )
    t3 = time.time()
    print(f"  THE生成耗时: {t3 - t2:.2f} 秒")
    
    # 4. 保存最终 NIfTI 结果
    out_path = os.path.join(root_dir, "PortalVein_001_uq_4th.nii.gz")
    print(f"\n4. 保存最终 NIfTI 结果到: {out_path}")
    out_img = nib.Nifti1Image(uq_heatmap.astype(np.float32), img.affine, img.header)
    nib.save(out_img, out_path)
    print("  复现完成！")

if __name__ == "__main__":
    main()
