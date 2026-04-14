import os
import sys
import numpy as np
import nibabel as nib
import time
from scipy.ndimage import maximum_filter, gaussian_filter

# 添加核心库路径
sys.path.append("/home/DMT_dev")
import morse_3d

# ----------------- 阶段一：重构 DMT 引擎 (绝对物理坐标映射) -----------------
def get_dmt_sparse_energy_map(prob_map, dmt_res, tau=0.15):
    """
    生成纯粹的 DMT 全局能量点阵 (无空间扩散)
    """
    dmt_energy_map = np.zeros_like(prob_map, dtype=np.float32)
    original_shape = prob_map.shape
    
    births = np.asarray(dmt_res["births"])
    deaths = np.asarray(dmt_res["deaths"])
    birth_coords = np.asarray(dmt_res["birth_coords"])
    death_coords = np.asarray(dmt_res["death_coords"])
    
    # 将本质特征 (Essential Classes) 的 death 替换为 birth，使其 persistence 为 0
    valid_deaths = np.where(np.isinf(deaths), births, deaths)
    
    for i in range(len(births)):
        # 强制绝对值，确保能量始终为正
        persistence = abs(births[i] - valid_deaths[i])
        
        if persistence < tau and persistence > 0:
            # 过滤掉 C++ 可能因边界截断传来的 NaN 脏坐标
            if not (np.any(np.isnan(birth_coords[i])) or np.any(np.isnan(death_coords[i]))):
                
                # 【极其关键的修复】：C++ 返回的是 bi/2.0，已经是原图尺度，千万不要减 1！
                # 使用 np.round 处理半整数坐标 (面/边)，然后强转 int 锁定体素
                bz = int(np.clip(np.round(birth_coords[i][0]), 0, original_shape[0] - 1))
                by = int(np.clip(np.round(birth_coords[i][1]), 0, original_shape[1] - 1))
                bx = int(np.clip(np.round(birth_coords[i][2]), 0, original_shape[2] - 1))
                
                dz = int(np.clip(np.round(death_coords[i][0]), 0, original_shape[0] - 1))
                dy = int(np.clip(np.round(death_coords[i][1]), 0, original_shape[1] - 1))
                dx = int(np.clip(np.round(death_coords[i][2]), 0, original_shape[2] - 1))
                
                # 在产生拓扑变化的两个关键咽喉点 (Birth & Death) 注入持久度能量
                dmt_energy_map[bz, by, bx] = max(dmt_energy_map[bz, by, bx], persistence)
                dmt_energy_map[dz, dy, dx] = max(dmt_energy_map[dz, dy, dx], persistence)
            
    dmt_energy_field = maximum_filter(dmt_energy_map, size=15)
    
    # 增加一层极其轻微的高斯平滑，让力场的边缘柔和过渡，防止出现硬切边
    dmt_energy_field = gaussian_filter(dmt_energy_field, sigma=1.0)
    
    # 局部能量归一化
    dmt_max = np.max(dmt_energy_field)
    if dmt_max > 0:
        dmt_energy_field /= dmt_max
        
    return dmt_energy_field

# ----------------- 阶段二：生成 Hessian 几何掩码 (全像素并行) -----------------
def get_hessian_geometry_mask(prob_map, sigma=1.0):
    """
    生成零偏移的 Hessian 鞍点几何掩码 (附带极致内存管理)
    """
    p_map = prob_map.astype(np.float32)

    # 1. 计算 6 个独立高斯二阶导数
    L_zz = gaussian_filter(p_map, sigma=sigma, order=[2, 0, 0])
    L_yy = gaussian_filter(p_map, sigma=sigma, order=[0, 2, 0])
    L_xx = gaussian_filter(p_map, sigma=sigma, order=[0, 0, 2])
    L_zy = gaussian_filter(p_map, sigma=sigma, order=[1, 1, 0])
    L_zx = gaussian_filter(p_map, sigma=sigma, order=[1, 0, 1])
    L_yx = gaussian_filter(p_map, sigma=sigma, order=[0, 1, 1])
    
    # 2. 构造全图 3x3 Hessian 矩阵
    H_matrix = np.zeros(p_map.shape + (3, 3), dtype=np.float32)
    H_matrix[..., 0, 0] = L_zz
    H_matrix[..., 1, 1] = L_yy
    H_matrix[..., 2, 2] = L_xx
    H_matrix[..., 0, 1] = H_matrix[..., 1, 0] = L_zy
    H_matrix[..., 0, 2] = H_matrix[..., 2, 0] = L_zx
    H_matrix[..., 1, 2] = H_matrix[..., 2, 1] = L_yx
    
    # 【爆显存防御】：矩阵已组装，立即抹杀中间体数组释放数 GB 内存
    del L_zz, L_yy, L_xx, L_zy, L_zx, L_yx
    
    # 3. 并行特征值分解 (注意 numpy 返回的是升序：l3是最小值，l1是最大值)
    eigvals = np.linalg.eigvalsh(H_matrix)
    del H_matrix
    
    l3, l2, l1 = eigvals[..., 0], eigvals[..., 1], eigvals[..., 2]
    
    # 4. 提取鞍点分数：一正两负 (马鞍面：沿着一个方向是谷，另外两个方向是峰)
    saddle_mask = (l1 > 0) & (l2 < 0) & (l3 < 0)
    saddle_score = l1 * saddle_mask
    saddle_score = np.nan_to_num(saddle_score, nan=0.0)
    
    # 鞍点强度归一化
    max_score = np.percentile(saddle_score[saddle_score > 0], 99) if np.any(saddle_score > 0) else 1e-9
    saddle_score = np.clip(saddle_score / max_score, 0.0, 1.0)
    
    return saddle_score

# ----------------- 阶段三：终极张量融合 -----------------
def generate_hybrid_topological_uq(prob_map, dmt_res, tau=0.15, hessian_sigma=1.0):
    """
    微分-代数混合拓扑 UQ 主管线
    """
    print("    [1/3] 计算 DMT 能量屏障 (代数全局指导)...")
    dmt_energy = get_dmt_sparse_energy_map(prob_map, dmt_res, tau)
    
    print("    [2/3] 计算 Hessian 几何掩码 (微积分局部手术刀)...")
    hessian_geometry = get_hessian_geometry_mask(prob_map, sigma=hessian_sigma)
    
    print("    [3/3] 计算 结构敏感熵 (基础概率犹豫度)...")
    p = np.clip(prob_map.astype(np.float32), 0.0, 1.0) # 严防插值越界导致 log(NaN)
    raw_entropy = -p * np.log(p + 1e-9) - (1 - p) * np.log(1 - p + 1e-9)
    # 引入 Gamma=2.0，对熵进行非线性锐化
    entropy_weight = (raw_entropy / np.log(2.0)) ** 1.0 
    
    print("    [+] 张量融合: Energy × Geometry × Entropy...")
    hybrid_uq = dmt_energy * hessian_geometry * entropy_weight
    
    # 保底几何掩码：完全切除绝对背景中的任何微弱共振
    geometry_mask = (p > 0.05).astype(np.float32)
    hybrid_uq = hybrid_uq * geometry_mask
    
    # 最终可视化拉伸 (让医生肉眼可见的强高光)
    active_pixels = hybrid_uq[hybrid_uq > 0]
    if len(active_pixels) > 0:
        vmax = np.percentile(active_pixels, 99)
        hybrid_uq = np.clip(hybrid_uq / (vmax + 1e-9), 0.0, 1.0)
        
    return hybrid_uq

def run_hl_ph_pipeline(file_path):
    print(f"\n==================================================")
    print(f"🚀 开始处理: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    print("=> 正在加载 NIfTI 数据...")
    img = nib.load(file_path)
    prob_map = img.get_fdata()
    
    # 自动识别通道 (如果是 HxWxDx2 的 One-hot 预测，取前景通道)
    if prob_map.ndim == 4:
        prob_map = prob_map[..., 1]
    
    # morse_3d 强依赖于 float64 类型
    prob_map_f64 = prob_map.astype(np.float64)
    print(f"✅ 数据加载完成, Shape: {prob_map_f64.shape}")

    print("=> 步骤 1: 运行 DMT 引擎...")
    t0 = time.time()
    dmt_res = morse_3d.extract_persistence_3d_morse(prob_map_f64)
    print(f"✅ DMT 耗时: {time.time() - t0:.2f} 秒, 提取特征数: {len(dmt_res['births'])}")

    print("=> 步骤 2: 运行 HL-PH (Hessian-Localized Persistent Homology)...")
    t1 = time.time()
    final_uq_map = generate_hybrid_topological_uq(prob_map_f64, dmt_res)
    print(f"✅ HL-PH 融合耗时: {time.time() - t1:.2f} 秒")

    # 输出路径设置
    out_path = file_path.replace("_prob.nii.gz", "_hl_ph_uq.nii.gz")
    if out_path == file_path:
        out_path = file_path + "_hl_ph_uq.nii.gz"
        
    print(f"=> 保存结果到: {out_path}")
    out_img = nib.Nifti1Image(final_uq_map.astype(np.float32), img.affine, img.header)
    nib.save(out_img, out_path)
    print("🎉 处理完成!")

if __name__ == "__main__":
    test_files = [
        "/home/DMT_dev/data/PortalVein/PortalVein_001_prob.nii.gz",
        "/home/DMT_dev/data/vessel12_01_3rd/vessel12_01_prob.nii.gz"
    ]
    for f in test_files:
        run_hl_ph_pipeline(f)