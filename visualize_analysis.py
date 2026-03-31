import numpy as np
import morse_3d
import time
import os
import gc
import plotly.graph_objects as go
import gradio as gr

def process_data(file_path):
    print(f"Loading {file_path}...")
    start = time.time()
    data = np.load(file_path)['array'].astype(np.float64)
    load_time = time.time() - start
    
    # 1. Standardize (Normalization)
    min_val, max_val = np.min(data), np.max(data)
    data = (data - min_val) / (max_val - min_val)
    print(f"Data standardized: [{min_val}, {max_val}] -> [0, 1]")
    
    # 2. Run Morse 3D
    print("Running morse_3d (v3.1)...")
    t0 = time.time()
    results = morse_3d.extract_persistence_3d_morse(data)
    compute_time = time.time() - t0
    
    # 3. Handle Persistence (Super-level)
    b = results['births']
    d = results['deaths']
    dims = results['dims']
    bc = results['birth_coords']
    
    # Finite filtering
    finite_mask = np.isfinite(d)
    pers = np.zeros_like(b)
    pers[finite_mask] = b[finite_mask] - d[finite_mask]
    # For essential features, set a large persistence for ranking
    pers[~finite_mask] = 1.1 # Since data is [0, 1], max finite pers is 1.0

    return {
        'b': b, 'd': d, 'dims': dims, 'pers': pers, 'coords': bc,
        'load_time': load_time, 'compute_time': compute_time,
        'slice': data[data.shape[0]//2, :, :]  # Keep a slice for visualization
    }

def create_plots(results, top_n=2000):
    b, d, dims, pers, coords = results['b'], results['d'], results['dims'], results['pers'], results['coords']
    
    # --- Persistence Diagram ---
    fig_pd = go.Figure()
    colors = ['#EF553B', '#00CC96', '#636EFA'] # H0, H1, H2 colors
    
    for h in [0, 1, 2]:
        mask = (dims == h)
        if not np.any(mask): continue
        bh = b[mask]
        dh = d[mask]
        # Replace -inf with -0.05 for plotting
        dh_plot = np.where(np.isfinite(dh), dh, -0.05)
        
        fig_pd.add_trace(go.Scatter(
            x=bh, y=dh_plot, mode='markers', name=f'H{h}',
            marker=dict(size=4, opacity=0.6, color=colors[h])
        ))
    
    # Diagonal line
    fig_pd.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"))
    fig_pd.update_layout(title="Persistence Diagram (Super-level Selection)", xaxis_title="Birth", yaxis_title="Death", 
                         width=800, height=600, template="plotly_dark")
    
    # --- 3D Visualization of Persistent Features ---
    # Sort by persistence
    top_indices = np.argsort(pers)[-top_n:]
    top_coords = coords[top_indices]
    top_dims = dims[top_indices]
    top_pers = pers[top_indices]
    
    fig_3d = go.Figure()
    for h in [0, 1, 2]:
        mask = (top_dims == h)
        if not np.any(mask): continue
        fig_3d.add_trace(go.Scatter3d(
            x=top_coords[mask, 0], y=top_coords[mask, 1], z=top_coords[mask, 2],
            mode='markers', name=f'Top {h}',
            marker=dict(size=np.power(top_pers[mask], 0.5) * 10, color=colors[h], opacity=0.8)
        ))
        
    fig_3d.update_layout(title=f"3D Distribution of Top {top_n} Persistent Features",
                         scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                         width=800, height=800, template="plotly_dark")
    
    return fig_pd, fig_3d

# Start the application
cache = None

def run_analysis():
    global cache
    if cache is None:
        file_path = "/home/DMT_dev/data/patient-id-840.npz"
        cache = process_data(file_path)
    
    summary = f"Process Completed:\n- Load Time: {cache['load_time']:.2f}s\n- Compute Time: {cache['compute_time']:.2f}s\n- Total Pairs: {len(cache['b'])}"
    
    fig_pd, fig_3d = create_plots(cache)
    
    # Slice as sanity check
    slice_fig = go.Figure(data=go.Heatmap(z=cache['slice'], colorscale='Viridis'))
    slice_fig.update_layout(title="Central Slice (Z-Axis)", width=600, height=600, template="plotly_dark")
    
    return summary, fig_pd, fig_3d, slice_fig

with gr.Blocks(theme='ocean') as demo:
    gr.Markdown("# DMT-dev v3.1: 3D Topological Feature Analysis")
    with gr.Column():
        btn = gr.Button("Analyze Patient-840 (512³ Grid)", variant="primary")
        output_txt = gr.Textbox(label="Benchmark Summary")
        
        with gr.Row():
            pd_out = gr.Plot(label="Persistence Diagram")
            slice_out = gr.Plot(label="Original Data Slice")
            
        viz_3d_out = gr.Plot(label="Interactive 3D Persistence Scatter")

    btn.click(run_analysis, outputs=[output_txt, pd_out, viz_3d_out, slice_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
