import numpy as np
import plotly.graph_objects as go
import gradio as gr
import os

def run_viz():
    # 1. 加载数据
    uq_path = 'data/patient-id-840-uq-heatmap.npz'
    mask_path = 'data/vein_mask/patient-id-840.npz'
    
    if not os.path.exists(uq_path) or not os.path.exists(mask_path):
        return "Missing data files."

    uq_data = np.load(uq_path)['uq_map']
    full_mask = np.load(mask_path)['array']

    # 2. 执行与 generate_uq_heatmap.py 一一对应的裁剪逻辑 [192:320]
    mask_crop = full_mask[192:320, 192:320, 192:320]

    def get_slice(z):
        # z 现在是局部 128 范围内的索引
        m_slice = mask_crop[z, :, :]
        u_slice = uq_data[z, :, :]
        
        # 创建 Plotly 叠加图
        fig = go.Figure()
        # 底部: 裁剪后的局部红色/灰色 Mask
        fig.add_trace(go.Heatmap(z=m_slice, colorscale='gray', opacity=0.8, showscale=False))
        # 顶部: UQ 热力图叠加 (使用 jet 颜色)
        fig.add_trace(go.Heatmap(z=u_slice, colorscale='jet', opacity=0.4, showscale=True))
        
        fig.update_layout(
            title=f'Local ROI Comparison (Z-offset: 192+{z})',
            width=600, height=600, 
            margin=dict(l=0,r=0,t=40,b=0), 
            template='plotly_dark'
        )
        return fig

    with gr.Blocks(theme='ocean') as demo:
        gr.Markdown('### 🔍 DMT-UQ 局部裁剪对比器 (128x128x128)')
        gr.Markdown('视角锁定在原始空间的 **[192:320, 192:320, 192:320]** 核心区域。')
        z_slider = gr.Slider(0, 127, value=64, label='Local Z-Slice')
        out_plot = gr.Plot()
        z_slider.change(get_slice, inputs=z_slider, outputs=out_plot)

    demo.launch(server_name='0.0.0.0', server_port=7863)

if __name__ == "__main__":
    run_viz()
