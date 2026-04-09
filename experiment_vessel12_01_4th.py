import numpy as np
import nibabel as nib
import morse_3d
import time
import os

def generate_topological_uq_map(dmt_res: dict, prob_map: np.ndarray, tau: float = 0.15, sigma: float = 2.0) -> np.ndarray:
    """
    基于离散莫尔斯理论、L1持久度能量和宽带概率门控生成 3D 拓扑不确定性 (Topological UQ) 热力图。
    """
    print("  提取并过滤幻觉特征...")
    original_shape = prob_map.shape
    births = np.asarray(dmt_res["births"])
    deaths = np.asarray(dmt_res["deaths"])
    birth_coords = np.asarray(dmt_res["birth_coords"])
    death_coords = np.asarray(dmt_res["death_coords"])
    
    # 避免 deaths 中存在极大的负无穷/正无穷导致计算异常，安全替换
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
        
    # ================= 核心升级：熵调制双重加权 =================
    
    # 2A. 计算 Wasserstein 拓扑势能 (代数寿命 × 物理几何跨度)
    l1_persistence = (h_births - h_deaths)
    
    # 计算降生点与死亡点在 3D 物理空间中的欧氏距离 (即断裂/孔洞的解剖学跨度)
    # 注意：需确保坐标是物理正确的，不含 padding 偏移
    spatial_dist = np.linalg.norm(h_birth_coords - h_death_coords, axis=1)
    
    # 距离保底：防止同位像素的微小抖动距离为 0，基础跨度设为 1.0 个体素
    geometric_scale = np.clip(spatial_dist, 1.0, None) 
    
    # 真正的物理拓扑能量：概率落差越大、断裂跨度越宽，能量呈几何级暴增！
    energy = l1_persistence * geometric_scale
    
    # 2B. 指数级熵调制 (Exponential Entropy Modulation)
    # 引入 Gamma 指数来执行非线性的噪声衰减。建议值 2.0。
    gamma = 2.0
    
    # 获取对应坐标处的概率 p (安全取整并限制越界)
    cz = np.clip(np.floor(h_birth_coords[:, 0]).astype(int), 0, original_shape[0] - 1)
    cy = np.clip(np.floor(h_birth_coords[:, 1]).astype(int), 0, original_shape[1] - 1)
    cx = np.clip(np.floor(h_birth_coords[:, 2]).astype(int), 0, original_shape[2] - 1)
    p = prob_map[cz, cy, cx]
    
    # 计算原始香农熵
    raw_entropy = -p * np.log(p + 1e-9) - (1 - p) * np.log(1 - p + 1e-9)
    
    # 核心升级：先将熵归一化到 [0, 1] 区间 (除以 ln(2))，然后求 gamma 次幂。
    # 物理意义：p=0.5 时的断裂信号几乎无损 (1.0^gamma = 1.0)
    # p=0.05 时的背景噪点遭受降维打击 (例如 0.28^2 = 0.07)
    entropy_weight = (raw_entropy / np.log(2.0)) ** gamma
    
    # 2C. 终极融合权重：Wasserstein 能量 × 锐化后的信息熵
    weights = energy * entropy_weight
    
    # 因为下面将 Birth 和 Death 坐标 vstack 在了一起，权重数组也需要复制拼接到同等长度
    all_weights = np.concatenate((weights, weights))
    # ==========================================================

    # 合并生灭坐标，并减去 1.0 (修正 C++ 端 padding=1 的空间偏移)
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
        
        # 极小噪点或被门控拦截的特征：权重趋近于 0，直接跳过省算力
        if w_i < 1e-6:
            continue
            
        z_min = max(0, int(np.floor(cz_coord - radius)))
        z_max_idx = min(Z_max, int(np.ceil(cz_coord + radius)) + 1)
        
        y_min = max(0, int(np.floor(cy_coord - radius)))
        y_max_idx = min(Y_max, int(np.ceil(cy_coord + radius)) + 1)
        
        x_min = max(0, int(np.floor(cx_coord - radius)))
        x_max_idx = min(X_max, int(np.ceil(cx_coord + radius)) + 1)
        
        if z_min >= Z_max or z_max_idx <= 0 or \
           y_min >= Y_max or y_max_idx <= 0 or \
           x_min >= X_max or x_max_idx <= 0:
            continue
            
        grid_z, grid_y, grid_x = np.meshgrid(
            np.arange(z_min, z_max_idx),
            np.arange(y_min, y_max_idx),
            np.arange(x_min, x_max_idx),
            indexing='ij'
        )
        
        dist_sq = (grid_z - cz_coord)**2 + (grid_y - cy_coord)**2 + (grid_x - cx_coord)**2
        # 应用我们精心计算的权重 w_i
        uq_map[z_min:z_max_idx, y_min:y_max_idx, x_min:x_max_idx] += w_i * np.exp(-dist_sq / var_term)
    
    # ================= 核心升级：百分位可视化拉伸 =================
    # 提取所有有不确定性（大于0）的像素
    active_pixels = uq_map[uq_map > 0]
    if len(active_pixels) > 0:
        # 取 99% 百分位作为上限，防止极个别极值压垮全局亮度，挽救细微特征
        vmax = np.percentile(active_pixels, 99)
        # 裁剪并归一化到 [0, 1] 空间
        uq_map = np.clip(uq_map / (vmax + 1e-9), 0.0, 1.0)
    # ==========================================================

    # ================= 空间校准：几何掩码 (Geometry Masking) =================
    # 切除悬浮在绝对背景 (p <= 0.05) 中的高斯拖尾，强制高光贴合解剖学结构
    geometry_mask = (prob_map > 0.05).astype(np.float32)
    uq_map = uq_map * geometry_mask
    # =========================================================================

    return uq_map

def main():
    data_path = "/home/DMT_dev/data/vessel12_01_3rd/vessel12_01_prob_quantized.nii.gz"
    if not os.path.exists(data_path):
        print(f"错误: 找不到文件 {data_path}")
        return
        
    print(f"1. 加载 NIfTI 概率图数据: {data_path}")
    img = nib.load(data_path)
    data = img.get_fdata()
    
    # 如果原始文件带通道（前向实验），确保兼容提取通道逻辑
    # 因为上一轮我们量化的文件已经被提取过了单通道，所以可以直接作为 prob_map
    if len(data.shape) == 4 and data.shape[-1] >= 2:
        prob_map = data[..., 1]
    else:
        prob_map = data
    
    if not np.issubdtype(prob_map.dtype, np.floating):
        prob_map = prob_map.astype(np.float64)
    elif prob_map.dtype != np.float64:
        prob_map = prob_map.astype(np.float64)
        
    original_shape = prob_map.shape
    print(f"  Shape: {original_shape}, Dtype: {prob_map.dtype}")
    print(f"  数值范围: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
    
    print("\n2. 运行 C++ DMT 引擎计算持久同调...")
    t0 = time.time()
    dmt_res = morse_3d.extract_persistence_3d_morse(prob_map)
    t1 = time.time()
    
    num_pairs = len(dmt_res['births'])
    print(f"  DMT 引擎耗时: {t1 - t0:.2f} 秒")
    print(f"  提取出的总特征数: {num_pairs}")
    
    print("\n3. 生成拓扑幻觉能量 (THE / UQ) 热力图...")
    t2 = time.time()
    uq_heatmap = generate_topological_uq_map(
        dmt_res=dmt_res,
        prob_map=prob_map,
        tau=0.10, # 在上一轮中已经使用过 tau=0.10 获取更多细节，此处保持一致或依据您需要调整。
        sigma=1.0  # 将高斯核标准差减半，让高光更锐利
    )
    t3 = time.time()
    
    print(f"  THE生成耗时: {t3 - t2:.2f} 秒")
    print(f"  THE Heatmap 数值范围: [{uq_heatmap.min():.4f}, {uq_heatmap.max():.4f}]")
    
    # 保存结果为主流 NIfTI 格式
    out_path = "/home/DMT_dev/data/vessel12_01_3rd/vessel12_01_uq_4th.nii.gz"
    print(f"\n4. 保存 THE 结果为 NIfTI 格式到: {out_path}")
    
    out_img = nib.Nifti1Image(uq_heatmap.astype(np.float32), affine=img.affine, header=img.header)
    nib.save(out_img, out_path)
    
    # 如果还需要顺手保存一个 .npz 文件，可解除注释：
    # npz_out_path = "/home/DMT_dev/data/vessel12_01_3rd/vessel12_01_uq_4th.npz"
    # np.savez_compressed(npz_out_path, uq_map=uq_heatmap)
    
    print("  完成！")

if __name__ == "__main__":
    main()
