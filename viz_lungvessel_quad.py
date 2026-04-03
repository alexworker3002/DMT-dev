import numpy as np
import plotly.graph_objects as go
import gradio as gr
import os
import nibabel as nib

def run_quad_viz():
    ct_path = '/home/DMT_dev/data/vessel12_01/vessel12_01_ct.nii.gz'
    label_path = '/home/DMT_dev/data/vessel12_01/vessel12_01_gt.nii.gz'
    pred_path = '/home/DMT_dev/data/vessel12_01/vessel12_01_pred.nii.gz'
    prob_path = '/home/DMT_dev/data/vessel12_01/vessel12_01_prob.npz'
    uq_path = '/home/DMT_dev/data/vessel12_01/vessel12_01_uq.npz'
    
    if not all(os.path.exists(p) for p in [ct_path, label_path, uq_path]):
        return "Missing data files. Ensure DMT THE has been calculated."

    print("Loading Original CT...")
    # shape: (X, Y, Z) -> transpose to (Z, Y, X)
    full_ct = np.transpose(nib.load(ct_path).get_fdata(), (2, 1, 0)).astype(np.float32)
    # 限制肺部窗以增强对比度，一般肺窗是 W: 1500, L: -500 ([-1250, 250])，这里用常见值
    ct_min, ct_max = -1200, 400
    
    print("Loading Label Mask...")
    full_label = np.transpose(nib.load(label_path).get_fdata(), (2, 1, 0)).astype(np.float32)

    has_pred = os.path.exists(pred_path)
    if has_pred:
        print("Loading Predicted Mask...")
        full_pred = np.transpose(nib.load(pred_path).get_fdata(), (2, 1, 0)).astype(np.float32)

    print("Loading Probability Map...")
    full_prob = np.load(prob_path)['probabilities'][1].astype(np.float32)

    print("Loading UQ Heatmap...")
    uq_data = np.load(uq_path)['uq_map']
    
    Z_dim = full_ct.shape[0]

    def get_comparison(z):
        z_idx = int(z)
        c_slice = full_ct[z_idx, :, :]
        # Clip CT slice for visualization normalization
        c_slice = np.clip(c_slice, ct_min, ct_max)
        
        l_slice = full_label[z_idx, :, :]
        p_slice = full_pred[z_idx, :, :] if has_pred else np.zeros_like(l_slice)
        prob_slice = full_prob[z_idx, :, :]
        u_slice = uq_data[z_idx, :, :]
        
        def create_fig(z_data, colorscale, title, show_scale=True, zmin=None, zmax=None):
            f = go.Figure(data=go.Heatmap(z=z_data, colorscale=colorscale, showscale=show_scale, zmin=zmin, zmax=zmax))
            f.update_layout(title=title, width=400, height=400, margin=dict(l=0,r=0,t=40,b=0), template='plotly_dark')
            # 翻转 y 轴，使得医学图像正常显示
            f.update_yaxes(autorange="reversed")
            return f

        # 图1: 原始 CT 图像 
        fig_ct = create_fig(c_slice, 'gray', '1. Original CT', show_scale=False, zmin=ct_min, zmax=ct_max)
        
        # 图2: 对应 label 掩码
        fig_label = create_fig(l_slice, 'gray', '2. Ground Truth Label', show_scale=False)
        
        # 新增图: 模型输出的二值化预测掩码
        fig_pred = create_fig(p_slice, 'gray', '3. Predicted Mask (Binary)', show_scale=False)
        
        # 新增图: 模型原始概率场 (Softmax Probability)
        fig_prob = create_fig(prob_slice, 'magma', '4. Softmax Probability Map', show_scale=True, zmin=0.0, zmax=1.0)
        
        # 图5: morse_3d 计算结果 (THE)
        fig_uq = create_fig(u_slice, 'jet', '5. Topological THE (UQ)', show_scale=True, zmin=0.0)
        
        # 图6: 仅仅叠加模型输出二值掩码和 UQ 值
        fig_overlay = go.Figure()
        # 底部: 使用模型输出的二值掩码 (灰度显示)
        fig_overlay.add_trace(go.Heatmap(z=p_slice, colorscale='gray', opacity=0.8, showscale=False))
        # 叠加: UQ 热力图 (Jet 色调)
        # 动态适应大幅降低的新算法能量域
        threshold = uq_data.max() * 0.01 if uq_data.max() > 0 else 1e-4
        uq_overlay = np.where(u_slice > threshold, u_slice, np.nan)
        fig_overlay.add_trace(go.Heatmap(z=uq_overlay, colorscale='jet', opacity=0.7, showscale=True))
        
        fig_overlay.update_layout(title='6. Predicted Mask + UQ Overlay', width=450, height=450, margin=dict(l=0,r=0,t=40,b=0), template='plotly_dark')
        fig_overlay.update_yaxes(autorange="reversed")
        
        return fig_ct, fig_label, fig_pred, fig_prob, fig_uq, fig_overlay

    print("Launching Dashboard...")
    with gr.Blocks(theme='ocean') as demo:
        gr.Markdown('### 🔬 Lung Vessel Segmentation & THE Topological Analysis')
        gr.Markdown('Sample: **Dataset001_LungVessel (vessel12_01)**')
        
        z_slider = gr.Slider(0, Z_dim - 1, value=Z_dim // 2, label='Z-Level Index (Axial View)')
        
        with gr.Row():
            p1 = gr.Plot(label="Original CT")
            p2 = gr.Plot(label="Label Mask")
            p_pred = gr.Plot(label="Predicted Mask (Binary)")
            
        with gr.Row():
            p_prob = gr.Plot(label="Softmax Probability")
            p3 = gr.Plot(label="THE Heatmap")
            p4 = gr.Plot(label="Combined Overlay View")
        
        z_slider.change(get_comparison, inputs=z_slider, outputs=[p1, p2, p_pred, p_prob, p3, p4])

    # Find an open port starting from 7860
    import socket
    def find_free_port(start_port):
        for port in range(start_port, start_port + 100):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            res = sock.connect_ex(('0.0.0.0', port))
            sock.close()
            if res != 0:
                return port
        return start_port

    port = find_free_port(7865)
    demo.launch(server_name='0.0.0.0', server_port=port)

if __name__ == "__main__":
    run_quad_viz()
