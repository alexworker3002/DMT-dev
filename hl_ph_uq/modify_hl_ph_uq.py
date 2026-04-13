import os
import nibabel as nib
import numpy as np
import time
from scipy.ndimage import gaussian_filter, label, find_objects
from skimage.morphology import remove_small_objects

def post_process_hl_ph_uq(uq_map, prob_map, min_size=50, gamma=1.2, top_p=99.9):
    """
    对 HL-PH UQ Map 进行后处理优化：
    1. 连通域降噪：移除过小的孤立不确定性簇
    2. 非线性对比度增强：利用 Gamma 变换拉伸高不确定性区域
    3. 结构一致性约束：结合概率图梯度，将 UQ 限制在解剖结构的拓扑关键位置
    """
    print(f"  [+] 原始 UQ 统计: Max={uq_map.max():.4f}, Mean={uq_map.mean():.4f}, Non-zero pixels={np.sum(uq_map > 0)}")
    
    # 1. 连通域降噪 (移除细碎噪点)
    # 利用 skimage 的 remove_small_objects 需要布尔掩码
    binary_mask = uq_map > 0.05
    cleaned_mask = remove_small_objects(binary_mask, min_size=min_size)
    processed_uq = uq_map * cleaned_mask.astype(np.float32)
    
    # 2. 非线性对比度拉伸 (Gamma 变换)
    # 物理意义：让“确定”的幻觉信号更强，模糊的过渡地带更温和
    processed_uq = np.power(processed_uq, 1.0 / gamma) 
    
    # 3. 归一化与百分位亮度拉伸
    active_mask = processed_uq > 1e-6
    if np.any(active_mask):
        v_max = np.percentile(processed_uq[active_mask], top_p)
        processed_uq = np.clip(processed_uq / (v_max + 1e-9), 0.0, 1.0)
    
    # 4. (可选) 边缘锐化：结合 Hessian 马鞍面的锐利特性
    # 如果 UQ 在概率图变化剧烈的边界处，则保留，在内部则轻微削弱
    # 这里简单使用概率图的高斯梯度作为权重
    # prob_grad = np.abs(prob_map - gaussian_filter(prob_map, sigma=1.0))
    # processed_uq *= (1.0 + 2.0 * prob_grad) # 增强边界处的 UQ 显著性
    # processed_uq = np.clip(processed_uq, 0.0, 1.0)

    print(f"  [+] 处理后 UQ 统计: Max={processed_uq.max():.4f}, Mean={processed_uq.mean():.4f}, Non-zero pixels={np.sum(processed_uq > 0)}")
    return processed_uq

def modify_operations(file_path):
    print(f"\nProcessing: {file_path}")
    if not os.path.exists(file_path):
        print(f"Skip: File not found")
        return

    # 加载 UQ Map
    uq_img = nib.load(file_path)
    uq_data = uq_img.get_fdata().astype(np.float32)
    
    # 加载对应的 Prob Map 以进行参考
    prob_path = file_path.replace("_hl_ph_uq.nii.gz", "_prob.nii.gz")
    if not os.path.exists(prob_path):
        # 尝试不同目录或前缀
        prob_path = file_path.replace("_hl_ph_uq", "_prob")
    
    if os.path.exists(prob_path):
        prob_img = nib.load(prob_path)
        prob_data = prob_img.get_fdata()
        if prob_data.ndim == 4: prob_data = prob_data[..., 1]
    else:
        print(f"Warning: Reference Prob Map not found at {prob_path}, using dummy.")
        prob_data = np.zeros_like(uq_data)

    # 执行后处理
    start_t = time.time()
    refined_uq = post_process_hl_ph_uq(uq_data, prob_data, min_size=64, gamma=1.5)
    end_t = time.time()
    print(f"  [+] Post-processing took {end_t - start_t:.2f}s")

    # 保存结果
    out_path = file_path.replace("_hl_ph_uq.nii.gz", "_hl_ph_uq_refined.nii.gz")
    out_img = nib.Nifti1Image(refined_uq, uq_img.affine, uq_img.header)
    nib.save(out_img, out_path)
    print(f"  [+] Saved to: {out_path}")

if __name__ == "__main__":
    test_files = [
        "/home/DMT_dev/data/PortalVein/PortalVein_001_hl_ph_uq.nii.gz",
        "/home/DMT_dev/data/vessel12_01_3rd/vessel12_01_hl_ph_uq.nii.gz"
    ]
    for f in test_files:
        modify_operations(f)
