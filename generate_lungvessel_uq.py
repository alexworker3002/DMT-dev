import numpy as np
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
    
    # 2A. 计算持久度能量 (采用 L1 线性惩罚，保留所有真实存在的断裂权重)
    energy = (h_births - h_deaths)
    
    # 2B. 熵调制 (Entropy Modulation) — 废弃宽带门控，改用连续的香农熵作为权重调节因子
    # 安全取整并限制越界以获取对应坐标的概率 p
    cz = np.clip(np.floor(h_birth_coords[:, 0]).astype(int), 0, original_shape[0] - 1)
    cy = np.clip(np.floor(h_birth_coords[:, 1]).astype(int), 0, original_shape[1] - 1)
    cx = np.clip(np.floor(h_birth_coords[:, 2]).astype(int), 0, original_shape[2] - 1)
    
    p = prob_map[cz, cy, cx]
    
    # 香农熵：H(p) = -p*log(p) - (1-p)*log(1-p)
    # • 当 p→0 或 p→1（绝对背景/前景）时，H(p)→0，权重被自然压制
    # • 当 p≈0.5（决策边界）时，H(p)→ln(2)≈0.693，权重被最大化激活
    # • 相比宽带门控，背景处 p=0.05 的权重为 H(0.05)≈0.286，而非1.0
    entropy = -p * np.log(p + 1e-9) - (1 - p) * np.log(1 - p + 1e-9)
    
    # 2C. 融合权重：将代数拓扑势能与连续信息熵绑定
    weights = energy * entropy
    
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
        
    return uq_map

def main():
    data_path = "/home/DMT_dev/data/vessel12_01/vessel12_01_prob.npz"
    if not os.path.exists(data_path):
        print(f"错误: 找不到文件 {data_path}")
        return
        
    print(f"1. 加载概率图数据: {data_path}")
    data = np.load(data_path)
    
    if 'probabilities' not in data:
        print("错误: NPZ 文件不包含 'probabilities' 键")
        return
        
    # 获取前景概率 (假设索引 1 是血管前景)
    prob_map = data['probabilities'][1]
    
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
        tau=0.15,
        sigma=2.0
    )
    t3 = time.time()
    
    print(f"  THE生成耗时: {t3 - t2:.2f} 秒")
    print(f"  THE Heatmap 数值范围: [{uq_heatmap.min():.4f}, {uq_heatmap.max():.4f}]")
    
    # 保存结果
    out_path = "/home/DMT_dev/data/vessel12_01/vessel12_01_uq.npz"
    print(f"\n4. 保存THE结果到: {out_path}")
    np.savez_compressed(out_path, uq_map=uq_heatmap)
    print("  完成！")

if __name__ == "__main__":
    main()