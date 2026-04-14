import os
import sys
import time
import numpy as np
import pandas as pd
import nibabel as nib
from skimage.morphology import skeletonize

# Ensure DMT_dev and its submodules can be imported
sys.path.append("/workspace/repositories/DMT_dev")
sys.path.append("/home/DMT_dev")

try:
    import morse_3d
    from hl_ph_uq.run_hl_ph import generate_hybrid_topological_uq
except ImportError as e:
    print(f"Warning: Module import failed. Dependencies may not be reachable: {e}")

def compute_dicemetric(v_pred, v_gt):
    """
    Standard Volume Dice Score
    """
    v_pred = (v_pred > 0.5).astype(np.uint8)
    v_gt = (v_gt > 0.5).astype(np.uint8)
    
    intersection = np.sum(v_pred * v_gt)
    sum_volumes = np.sum(v_pred) + np.sum(v_gt)
    
    if sum_volumes == 0:
        return 1.0
    return 2.0 * intersection / sum_volumes

def compute_cl_score(volume, skeleton):
    """
    Computes T_prec or T_sens topology score.
    Returns the ratio of the skeleton that is inside the volume.
    Safely handles empty skeletons.
    """
    skel_sum = np.sum(skeleton)
    if skel_sum == 0:
        return 0.0
    
    # Calculate skeleton inside the volume
    intersection = np.logical_and(volume, skeleton)
    return np.sum(intersection) / skel_sum

def evaluate_cldice(v_pred, v_gt):
    """
    Computes standard clDice score.
    Includes hardware safety interceptions (force binarization) 
    and handles empty edge cases.
    """
    # [极度重要：安全拦截机制] - 强制浮点截断以防 skeletonize C++ 层崩溃
    v_pred = (v_pred > 0.5).astype(np.uint8)
    v_gt = (v_gt > 0.5).astype(np.uint8)
    
    # 空集处理
    sum_pred = np.sum(v_pred)
    sum_gt = np.sum(v_gt)
    
    if sum_pred == 0 and sum_gt == 0:
        return 1.0, 1.0, 1.0 # Both empty -> perfect alignment
    if sum_pred == 0 or sum_gt == 0:
        return 0.0, 0.0, 0.0 # One is empty -> zero score
    
    # 提取 3D 骨架 (耗时操作)
    s_pred = skeletonize(v_pred).astype(np.uint8)
    s_gt = skeletonize(v_gt).astype(np.uint8)
    
    # 计算拓扑精确率和敏感度
    t_prec = compute_cl_score(v_gt, s_pred)
    t_sens = compute_cl_score(v_pred, s_gt)
    
    if t_prec + t_sens == 0:
        return 0.0, t_prec, t_sens
        
    cldice = 2.0 * t_prec * t_sens / (t_prec + t_sens)
    
    return cldice, t_prec, t_sens

def test_portalvein_pipeline():
    """
    测试步骤：
    1. 使用 prob（修改精度为3位小数）
    2. 计算 hl_ph_uq
    3. 根据 uq 计算 modify 二值 refine 的 nii.gz
       (规则 A: refined_prob = prob * (1.0 - lambda * uq), 然后 threshold > 0.5)
    4. 对比 pred 与 gt 的 clDice, 以及 refine 与 gt 的 clDice
    """
    print("\n" + "="*50)
    print("🚀 Starting clDice Topology Testing Pipeline on PortalVein")
    print("="*50)
    
    data_dir = "/workspace/repositories/DMT_dev/data/PotalVein"
    if not os.path.exists(data_dir):
        # Fallback to absolute mounted path just in case
        data_dir = "/home/DMT_dev/data/PotalVein"
        
    prob_path = os.path.join(data_dir, "PortalVein_001_prob.nii.gz")
    pred_path = os.path.join(data_dir, "PortalVein_001_pred.nii.gz")
    gt_path = os.path.join(data_dir, "PortalVein_001_gt.nii.gz")
    
    if not os.path.exists(prob_path):
        print(f"❌ Cannot find prob map at {prob_path}")
        return
        
    # ------------- 1. 加载并处理 Prob ------------- #
    print("\n=> [Step 1] Loading Prob Map & Adjusting Precision...")
    prob_img = nib.load(prob_path)
    prob_map = prob_img.get_fdata().astype(np.float64)
    
    # 取前景通道 (如果维数是 4)
    if prob_map.ndim == 4:
        prob_map = prob_map[..., 1]
    
    # prob_map = np.round(prob_map, 3) (已删除根据用户要求)
    print(f"   Max Prob: {prob_map.max()}, Min Prob: {prob_map.min()}")
    
    # 加载 GT 和 Pred 以供对比
    gt_img = nib.load(gt_path)
    gt_map = gt_img.get_fdata()
    
    pred_img = nib.load(pred_path)
    pred_map = pred_img.get_fdata()
    
    # ------------- 2. 计算 hl_ph_uq ------------- #
    print("\n=> [Step 2] Computing HL-PH UQ Map...")
    t0 = time.time()
    # 抽取拓扑对
    dmt_res = morse_3d.extract_persistence_3d_morse(prob_map)
    # 融合生成张量 UQ
    uq_map = generate_hybrid_topological_uq(prob_map, dmt_res)
    print(f"   UQ Map computed in {time.time() - t0:.2f}s, Max UQ: {uq_map.max():.4f}")
    
    # 保存 UQ Map 以供人工查看
    uq_out_path = os.path.join(data_dir, "PortalVein_001_hl_ph_uq.nii.gz")
    uq_img = nib.Nifti1Image(uq_map.astype(np.float32), prob_img.affine, prob_img.header)
    nib.save(uq_img, uq_out_path)
    print(f"   Saved HL-PH UQ map to: {uq_out_path}")
    
    # ------------- 3. 根据 UQ 计算二值化 refine ------------- #
    print("\n=> [Step 3] Applying Topology-guided Refinement (Rule A)...")
    LMBDA = 0.5  # 可以调节惩罚系数
    # Refined probability: P_calib = P - lambda * UQ * P^2
    refined_prob = prob_map - LMBDA * np.clip(uq_map, 0.0, 1.0) * (prob_map ** 2)
    # Thresholding to get binary refine map
    refine_map = (refined_prob > 0.5).astype(np.uint8)
    
    # 保存 Refine Map 以供人工查看
    refine_out_path = os.path.join(data_dir, "PortalVein_001_refine.nii.gz")
    refine_img = nib.Nifti1Image(refine_map.astype(np.float32), prob_img.affine, prob_img.header)
    nib.save(refine_img, refine_out_path)
    print(f"   Saved refined mask to: {refine_out_path}")
    
    # ------------- 4. 执行 clDice 评估对比 ------------- #
    print("\n=> [Step 4] Running clDice Topological Evaluation Series...")
    
    results = []
    
    # 4.1 Original Pred vs GT
    start_time = time.time()
    print("   -> Evaluating Base Prediction vs GT...")
    dice_pred = compute_dicemetric(pred_map, gt_map)
    cldice_pred, t_prec_pred, t_sens_pred = evaluate_cldice(pred_map, gt_map)
    time_pred = time.time() - start_time
    
    results.append({
        "Model Type": "Base Prediction",
        "Dice": dice_pred,
        "clDice": cldice_pred,
        "T_Prec": t_prec_pred,
        "T_Sens": t_sens_pred,
        "Eval Time (s)": round(time_pred, 2)
    })
    
    # 4.2 Refined Pred vs GT
    start_time = time.time()
    print("   -> Evaluating Refined Mask vs GT...")
    dice_ref = compute_dicemetric(refine_map, gt_map)
    cldice_ref, t_prec_ref, t_sens_ref = evaluate_cldice(refine_map, gt_map)
    time_ref = time.time() - start_time
    
    results.append({
        "Model Type": "HL-PH Refined (λ=0.5)",
        "Dice": dice_ref,
        "clDice": cldice_ref,
        "T_Prec": t_prec_ref,
        "T_Sens": t_sens_ref,
        "Eval Time (s)": round(time_ref, 2)
    })
    
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("📊 topology_metrics_report")
    print("="*50)
    print(df.to_string(index=False))
    
    report_path = os.path.join(os.path.dirname(__file__), "topology_metrics_report.csv")
    df.to_csv(report_path, index=False)
    print(f"\n✅ Report saved to {report_path}")


def test_vessel12_pipeline():
    """
    Vessel12 dataset evaluation pipeline
    """
    print("\n" + "="*50)
    print("🚀 Starting clDice Topology Testing Pipeline on Vessel12")
    print("="*50)
    
    data_dir = "/workspace/repositories/DMT_dev/data/vessel12"
    if not os.path.exists(data_dir):
        data_dir = "/home/DMT_dev/data/vessel12"
        
    prob_path = os.path.join(data_dir, "prob.nii.gz")
    pred_path = os.path.join(data_dir, "pred.nii.gz")
    gt_path = os.path.join(data_dir, "gt.nii.gz")
    
    if not os.path.exists(prob_path):
        print(f"❌ Cannot find prob map at {prob_path}")
        return
        
    prob_img = nib.load(prob_path)
    prob_map = prob_img.get_fdata().astype(np.float64)
    if prob_map.ndim == 4: prob_map = prob_map[..., 1]
    
    # [IMPORTANT] Round to 3 decimal places to reduce critical cells and avoid OOM
    prob_map = np.round(prob_map, 3)
    
    gt_img = nib.load(gt_path)
    gt_map = gt_img.get_fdata()
    pred_img = nib.load(pred_path)
    pred_map = pred_img.get_fdata()
    
    print("\n=> [Step 2] Computing HL-PH UQ Map...")
    t0 = time.time()
    dmt_res = morse_3d.extract_persistence_3d_morse(prob_map)
    uq_map = generate_hybrid_topological_uq(prob_map, dmt_res)
    
    uq_out_path = os.path.join(data_dir, "vessel12_hl_ph_uq.nii.gz")
    nib.save(nib.Nifti1Image(uq_map.astype(np.float32), prob_img.affine, prob_img.header), uq_out_path)
    
    print("\n=> [Step 3] Applying Topology-guided Refinement...")
    LMBDA = 0.5
    refined_prob = prob_map - LMBDA * np.clip(uq_map, 0.0, 1.0) * (prob_map ** 2)
    refine_map = (refined_prob > 0.5).astype(np.uint8)
    
    refine_out_path = os.path.join(data_dir, "vessel12_refine.nii.gz")
    nib.save(nib.Nifti1Image(refine_map.astype(np.float32), prob_img.affine, prob_img.header), refine_out_path)
    
    print("\n=> [Step 4] Running clDice Evaluation...")
    results = []
    
    # Base
    dice_pred = compute_dicemetric(pred_map, gt_map)
    cldice_pred, t_prec_pred, t_sens_pred = evaluate_cldice(pred_map, gt_map)
    results.append({"Dataset": "Vessel12", "Type": "Base", "Dice": dice_pred, "clDice": cldice_pred})
    
    # Refined
    dice_ref = compute_dicemetric(refine_map, gt_map)
    cldice_ref, t_prec_ref, t_sens_ref = evaluate_cldice(refine_map, gt_map)
    results.append({"Dataset": "Vessel12", "Type": "HL-PH Refined", "Dice": dice_ref, "clDice": cldice_ref})
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    # test_portalvein_pipeline()
    test_vessel12_pipeline()
