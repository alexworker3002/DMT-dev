import numpy as np
import morse_3d
import time
import sys
import os
from scipy.ndimage import gaussian_filter

def generate_topological_uq_map(dmt_res: dict, original_shape: tuple, tau: float = 0.15, sigma: float = 2.0) -> np.ndarray:
    """
    根据给定的离散莫尔斯(DMT)计算结果，生成3D拓扑不确定性(Topological UQ)热力图。
    """
    print("  提取并过滤幻觉特征...")
    births = np.asarray(dmt_res["births"])
    deaths = np.asarray(dmt_res["deaths"])
    birth_coords = np.asarray(dmt_res["birth_coords"])
    death_coords = np.asarray(dmt_res["death_coords"])
    
    persistence = births - deaths
    hallucination_mask = persistence < tau
    
    h_birth_coords = birth_coords[hallucination_mask]
    h_death_coords = death_coords[hallucination_mask]
    
    all_coords = np.vstack((h_birth_coords, h_death_coords))
    all_coords = all_coords - 1.0
    
    print(f"  幻觉点总数 (Birth + Death): {len(all_coords)}")
    print(f"  生成高斯扩散热力图 (sigma={sigma})...")
    
    uq_map = np.zeros(original_shape, dtype=np.float32)
    Z_max, Y_max, X_max = original_shape
    
    radius = int(3 * sigma)
    var_term = 2.0 * (sigma ** 2)
    
    for coord in all_coords:
        cz, cy, cx = coord
        
        z_min = max(0, int(np.floor(cz - radius)))
        z_max_idx = min(Z_max, int(np.ceil(cz + radius)) + 1)
        
        y_min = max(0, int(np.floor(cy - radius)))
        y_max_idx = min(Y_max, int(np.ceil(cy + radius)) + 1)
        
        x_min = max(0, int(np.floor(cx - radius)))
        x_max_idx = min(X_max, int(np.ceil(cx + radius)) + 1)
        
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
        
        dist_sq = (grid_z - cz)**2 + (grid_y - cy)**2 + (grid_x - cx)**2
        uq_map[z_min:z_max_idx, y_min:y_max_idx, x_min:x_max_idx] += np.exp(-dist_sq / var_term)
        
    return uq_map

def main():
    data_path = "data/vein_mask/patient-id-840.npz"
    if not os.path.exists(data_path):
        print(f"错误: 找不到文件 {data_path}")
        return
        
    print(f"1. 加载概率图数据: {data_path}")
    data = np.load(data_path)
    
    # 根据常见的 npz 键名提取数组
    # 我们假设数组可能在 'arr_0', 'array', 'pred', 'vol' 等键下面
    key = 'pred' if 'pred' in data else ('array' if 'array' in data else list(data.keys())[0])
    prob_map = data[key]
    
    if not np.issubdtype(prob_map.dtype, np.floating):
        prob_map = prob_map.astype(np.float64)
    elif prob_map.dtype != np.float64:
        prob_map = prob_map.astype(np.float64)
        
    print("  为防内存溢出(OOM)，裁剪为局部 128x128x128 进行引擎与热力图代码正确性验证...")
    prob_map = prob_map[192:320, 192:320, 192:320]
    
    # 核心测试：边界模糊化 (生成 0-1 连续数据)
    print("  执行边界模糊化 (sigma=1.5) 生成连续 'fake_mask'...")
    fake_mask = gaussian_filter(prob_map.astype(np.float64), sigma=1.5)
    prob_map = fake_mask # 使用模糊后的数据进行 DMT 分析
        
    original_shape = prob_map.shape
    print(f"  Shape: {original_shape}, Dtype: {prob_map.dtype}, Key: {key}")
    print(f"  数值范围: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
    
    print("\n2. 运行 C++ DMT 引擎...")
    t0 = time.time()
    # 根据 test_morse_on_mask.py，可以直接向 morse_3d.extract_persistence_3d_morse 传入 numpy array
    dmt_res = morse_3d.extract_persistence_3d_morse(prob_map)
    t1 = time.time()
    num_pairs = len(dmt_res['births'])
    print(f"  DMT 引擎耗时: {t1 - t0:.2f} 秒")
    print(f"  提取出的总特征数: {num_pairs}")
    
    print("\n3. 生成拓扑不确定性(UQ)热力图...")
    t2 = time.time()
    uq_heatmap = generate_topological_uq_map(
        dmt_res=dmt_res,
        original_shape=original_shape,
        tau=0.1,  # 调低 tau 以捕获模糊边界产生的小尺度特征
        sigma=2.0
    )
    t3 = time.time()
    
    print(f"  UQ 生成耗时: {t3 - t2:.2f} 秒")
    print(f"  UQ Heatmap 数值范围: [{uq_heatmap.min():.4f}, {uq_heatmap.max():.4f}]")
    
    # 保存结果
    out_path = "data/patient-id-840-uq-heatmap.npz"
    print(f"\n4. 保存结果到: {out_path} (包含 uq_map 和 fake_mask)")
    np.savez_compressed(out_path, uq_map=uq_heatmap, fake_mask=fake_mask)
    print("  完成！")

if __name__ == "__main__":
    main()
