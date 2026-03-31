import numpy as np
import morse_3d
import time
import os
import gc
from scipy.ndimage import zoom
import plotly.graph_objects as go
import gradio as gr

def process_64_data(file_path):
    print(f"Loading {file_path}...")
    data = np.load(file_path)['array'].astype(np.float64)
    
    # 1. Downsample from 512 -> 64 (Ratio 1/8)
    print("Downsampling 512^3 -> 64^3...")
    t_ds = time.time()
    data_64 = zoom(data, 0.125, order=1) # Bilinear interpolation for speed + smoothing
    ds_time = time.time() - t_ds
    
    # 2. Standardize
    min_val, max_val = np.min(data_64), np.max(data_64)
    data_64 = (data_64 - min_val) / (max_val - min_val)
    print(f"Data standardized: [{min_val:.2f}, {max_val:.2f}] -> [0, 1]")
    
    # 3. Run Morse 3D
    print("Running morse_3d (v3.1) on 64^3 grid...")
    t0 = time.time()
    results = morse_3d.extract_persistence_3d_morse(data_64)
    compute_time = time.time() - t0
    
    return {
        'results': results,
        'ds_time': ds_time,
        'compute_time': compute_time,
        'slice': data_64[data_64.shape[0]//2, :, :],
        'shape': data_64.shape
    }

def create_plots(cache, top_n=1000):
    res = cache['results']
    b, d, dims, bc = res['births'], res['deaths'], res['dims'], res['birth_coords']
    
    finite_mask = np.isfinite(d)
    pers = np.zeros_like(b)
    pers[finite_mask] = b[finite_mask] - d[finite_mask]
    pers[~finite_mask] = 1.1 # Essential
    
    # --- Persistence Diagram ---
    fig_pd = go.Figure()
    colors = ['#EF553B', '#00CC96', '#636EFA']
    for h in [0, 1, 2]:
        mask = (dims == h)
        if not np.any(mask): continue
        bh, dh = b[mask], d[mask]
        dh_plot = np.where(np.isfinite(dh), dh, -0.05)
        fig_pd.add_trace(go.Scatter(x=bh, y=dh_plot, mode='markers', name=f'H{h}',
                                    marker=dict(size=5, opacity=0.7, color=colors[h])))
    fig_pd.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"))
    fig_pd.update_layout(title="Persistence Diagram (64^3 Grid)", template="plotly_dark", width=700)
    
    # --- 3D Scatter ---
    top_indices = np.argsort(pers)[-top_n:]
    top_coords, top_dims, top_pers = bc[top_indices], dims[top_indices], pers[top_indices]
    
    fig_3d = go.Figure()
    for h in [0, 1, 2]:
        mask = (top_dims == h)
        if not np.any(mask): continue
        fig_3d.add_trace(go.Scatter3d(
            x=top_coords[mask, 0], y=top_coords[mask, 1], z=top_coords[mask, 2],
            mode='markers', name=f'Top H{h}',
            marker=dict(size=np.power(top_pers[mask], 0.4) * 12, color=colors[h], opacity=0.8)
        ))
    fig_3d.update_layout(title=f"3D Distribution (Top {top_n} Features)",
                         scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                         template="plotly_dark", width=800)
    
    return fig_pd, fig_3d

def run_64_analysis():
    file_path = "/home/DMT_dev/data/patient-id-840.npz"
    cache = process_64_data(file_path)
    
    summary = f"Process Completed (Downsampled to {cache['shape']}):\n- Downsample Time: {cache['ds_time']:.2f}s\n- Compute Time: {cache['compute_time']:.2f}s\n- Total Pairs: {len(cache['results']['births'])}"
    
    fig_pd, fig_3d = create_plots(cache)
    slice_fig = go.Figure(data=go.Heatmap(z=cache['slice'], colorscale='Viridis'))
    slice_fig.update_layout(title="Central Slice (64^3)", width=500, height=500, template="plotly_dark")
    
    return summary, fig_pd, fig_3d, slice_fig

with gr.Blocks(theme='ocean') as demo:
    gr.Markdown("# DMT-dev v3.1: 64³ Downsampled Topological Rapid Analysis")
    with gr.Column():
        btn = gr.Button("Analyze 64³ Downsample", variant="primary")
        output_txt = gr.Textbox(label="Benchmark Summary")
        
        with gr.Row():
            pd_out = gr.Plot()
            slice_out = gr.Plot()
            
        viz_3d_out = gr.Plot()

    btn.click(run_64_analysis, outputs=[output_txt, pd_out, viz_3d_out, slice_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
