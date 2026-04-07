import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

def compute_uq_maps(pred_probs: np.ndarray, gt_labels: np.ndarray, uq_topological: np.ndarray) -> tuple:
    """
    计算并归一化用于对比的 UQ 热力图。
    输入必须是 1D 数组。
    """
    eps = 1e-9
    
    # 基础预测绝对误差 (金标准 Error Map)
    error_map = np.abs(pred_probs - gt_labels)
    
    # 1. Oracle (完美物理指导 - 上帝视角)
    uq_oracle = error_map.copy()
    
    # 2. Predictive Entropy (零样本预测熵 - 传统最强基线)
    uq_entropy = -pred_probs * np.log(pred_probs + eps) - (1 - pred_probs) * np.log(1 - pred_probs + eps)
    
    # 3. Topological UQ (我们的方法，直接接收传入的数据)
    uq_topo = uq_topological.copy()
    
    # 局部极值归一化函数
    def normalize(x):
        max_val = x.max()
        return x / max_val if max_val > 0 else x

    # 踢除违规的 Cross-Entropy，保留纯粹的 Zero-Shot 对比
    uq_maps = {
        'Oracle': normalize(uq_oracle),
        'Predictive Entropy': normalize(uq_entropy),
        'Topological UQ': normalize(uq_topo)
    }
    
    return error_map, uq_maps

def compute_sparsification_curve(error_map: np.ndarray, uq_map: np.ndarray, num_fractions: int = 100) -> tuple:
    """
    极速计算稀疏化误差曲线 (Cumulative Sum 优化版)。
    """
    N = len(error_map)
    fractions = np.linspace(0.0, 1.0, num_fractions)
    
    # 根据 UQ 的不确定性值从高到低排序 (argsort 默认升序，加负号变降序)
    sort_idx = np.argsort(-uq_map)
    sorted_errors = error_map[sort_idx]
    
    # 计算误差的累积和，避免 for 循环里重复求和
    cumsum_errors = np.cumsum(sorted_errors)
    total_sum = sorted_errors.sum()
    
    errors_list = np.zeros(num_fractions, dtype=np.float32)
    
    for i, frac in enumerate(fractions):
        num_remove = int(np.round(frac * N))
        
        if num_remove == N:
            errors_list[i] = 0.0
        elif num_remove == 0:
            errors_list[i] = total_sum / N
        else:
            remaining_sum = total_sum - cumsum_errors[num_remove - 1]
            remaining_count = N - num_remove
            errors_list[i] = remaining_sum / remaining_count
            
    return fractions, errors_list

def plot_ause(results: dict, save_path: str = "vessel12_ause_curve.png"):
    """
    绘制前 20% 剔除比例的 Sparsification Error 曲线。
    """
    plt.figure(figsize=(10, 8))
    
    styles = {
        'Oracle': {'color': 'black', 'linestyle': '--', 'linewidth': 2},
        'Predictive Entropy': {'color': 'blue', 'linestyle': '-', 'linewidth': 2},
        'Topological UQ': {'color': 'red', 'linestyle': '-', 'linewidth': 2.5}
    }
    
    # 计算 Oracle 的曲线面积 (用于基准相减)
    area_oracle = np.trapezoid(results['Oracle']['errors'], results['Oracle']['fractions'])
    
    for name, res in results.items():
        fracs = res['fractions']
        errs = res['errors']
        
        # 计算当前方法的绝对面积
        area_method = np.trapezoid(errs, fracs)
        
        # AUSE 核心指标：与上帝视角的差距
        ause = area_method - area_oracle
        
        if name == 'Oracle':
            label = f"{name} (Area: {area_method:.4f})"
        else:
            label = f"{name} (AUSE: {ause:.4f})"
            
        plt.plot(fracs, errs, label=label, **styles[name])
        
    # 降维打击视角：医疗图像 99% 是背景，决战全在前 20%！
    plt.xlim(-0.01, 0.20) 
    
    plt.xlabel("Fraction of removed pixels", fontsize=14)
    plt.ylabel("Remaining Mean Absolute Error", fontsize=14)
    plt.title("Sparsification Error Curve (Zoomed to Top 20%)", fontsize=16)
    
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] AUSE 曲线已保存至: {save_path}")
    plt.close()

def main():
    # 1. 配置文件路径
    prob_path = "/home/DMT_dev/data/vessel12_01/vessel12_01_prob.npz"
    gt_path   = "/home/DMT_dev/data/vessel12_01/vessel12_01_gt.nii.gz"  # NIfTI 格式
    uq_path   = "/home/DMT_dev/data/vessel12_01/vessel12_01_uq.npz"
    
    for p in [prob_path, gt_path, uq_path]:
        if not os.path.exists(p):
            print(f"[Error] 找不到文件: {p}")
            return
            
    print("1. 加载真实测评数据...")
    prob_data = np.load(prob_path)
    uq_data   = np.load(uq_path)
    
    # GT 为 NIfTI 格式，需要 nibabel 读取
    # nibabel 返回 shape (X, Y, Z)，需要转置为 (Z, Y, X) 以与 prob_map / uq_map 对齐
    gt_nib    = nib.load(gt_path)
    gt_labels = np.transpose(gt_nib.get_fdata(), (2, 1, 0)).astype(np.float64)
    
    # 提取前景概率 (index=1)，键名为 'probabilities'
    pred_probs    = prob_data['probabilities'][1].astype(np.float64)  # shape: (Z, Y, X)
    uq_topological = uq_data['uq_map'].astype(np.float64)             # shape: (Z, Y, X)
    
    print(f"   pred_probs shape: {pred_probs.shape}")
    print(f"   gt_labels  shape: {gt_labels.shape}")
    print(f"   uq_map     shape: {uq_topological.shape}")
    
    print("2. 展平 3D 体积至 1D 向量 (加速计算)...")
    pred_probs_flat = pred_probs.flatten()
    gt_labels_flat = gt_labels.flatten()
    uq_topological_flat = uq_topological.flatten()
    
    print("3. 计算 Baseline UQ (预测熵)...")
    error_map, uq_maps = compute_uq_maps(pred_probs_flat, gt_labels_flat, uq_topological_flat)
    
    print("4. 计算 Sparsification Error 曲线...")
    results = {}
    for name, uq_map in uq_maps.items():
        print(f"   正在处理: {name}")
        # 取 200 个采样点，让曲线更平滑
        fracs, errs = compute_sparsification_curve(error_map, uq_map, num_fractions=200)
        results[name] = {'fractions': fracs, 'errors': errs}
        
    print("5. 渲染高精度学术图表...")
    plot_ause(results, "vessel12_ause_curve_real.png")

if __name__ == "__main__":
    main()