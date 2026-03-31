#!/usr/bin/env python3
import os
import glob
import numpy as np
import gradio as gr
from PIL import Image
import threading

# Configuration
LUNG_COLOR = [80, 180, 255]   # Soft Blue
VEIN_COLOR = [255, 80, 80]   # Bright Red
OVERLAP_COLOR = [255, 200, 80] # Golden highlight for overlap might be better to see? 
                              # Actually Purple is more standard but Gold pops.
                              # Let's go with Purple for medical accuracy.
OVERLAP_COLOR = [255, 80, 255] 

class MaskExplorer:
    def __init__(self):
        self.cache = {}
        self.patients = self._find_patients()
        self.current_patient = None
        self.lung = None
        self.vein = None

    def _find_patients(self):
        files = glob.glob("lung_mask/patient-id-*.npz")
        ids = [os.path.basename(f).replace(".npz", "") for f in files]
        return sorted(ids)

    def load_data(self, patient_id):
        if self.current_patient == patient_id:
            return self.lung, self.vein
        
        if patient_id in self.cache:
            self.lung, self.vein = self.cache[patient_id]
        else:
            lung_path = f"lung_mask/{patient_id}.npz"
            vein_path = f"vein_mask/{patient_id}.npz"
            
            self.lung = np.load(lung_path)['array'].astype(np.uint8)
            self.vein = np.load(vein_path)['array'].astype(np.uint8)
            
            # Simple LRU-like cache (just keep last 3 patients)
            if len(self.cache) > 3:
                self.cache.pop(next(iter(self.cache)))
            self.cache[patient_id] = (self.lung, self.vein)
            
        self.current_patient = patient_id
        return self.lung, self.vein

    def get_stats(self, patient_id):
        lung, vein = self.load_data(patient_id)
        v_lung = np.sum(lung)
        v_vein = np.sum(vein)
        v_overlap = np.sum((lung > 0) & (vein > 0))
        
        return {
            "Lung Volume (voxels)": f"{v_lung:,}",
            "Vein Volume (voxels)": f"{v_vein:,}",
            "Overlap Volume (voxels)": f"{v_overlap:,}",
            "Vein/Lung Ratio": f"{(v_vein/v_lung)*100:.2f}%" if v_lung > 0 else "0%"
        }

    def render(self, patient_id, axis, slice_idx):
        lung, vein = self.load_data(patient_id)
        
        if axis == "Axial (Z)":
            l_slice = lung[slice_idx, :, :]
            v_slice = vein[slice_idx, :, :]
        elif axis == "Coronal (Y)":
            l_slice = lung[:, slice_idx, :]
            v_slice = vein[:, slice_idx, :]
        else: # Sagittal (X)
            l_slice = lung[:, :, slice_idx]
            v_slice = vein[:, :, slice_idx]

        h, w = l_slice.shape
        # Create base image
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply colors
        img[l_slice > 0] = LUNG_COLOR
        img[v_slice > 0] = VEIN_COLOR
        img[(l_slice > 0) & (v_slice > 0)] = OVERLAP_COLOR
        
        # Add a subtle grid/background instead of pure black for "WOW"
        # Not really needed for medical, but good for aesthetics.
        
        return Image.fromarray(img)

explorer = MaskExplorer()

def update_ui(patient_id, axis, slice_idx):
    img = explorer.render(patient_id, axis, slice_idx)
    stats = explorer.get_stats(patient_id)
    stats_md = f"### 📊 Patient {patient_id} Stats\n"
    for k, v in stats.items():
        stats_md += f"- **{k}**: `{v}`\n"
    return img, stats_md

# custom CSS for premium look
css = """
#slice-img { border: 2px solid #555; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
.gradio-container { background-color: #121212 !important; color: #eee !important; }
"""

with gr.Blocks() as demo:
    gr.Markdown("""
    # 🫁 Pulmonary Mask Visualizer
    ### Highly Interactive 3D Visualization of Lung and Vein Masks
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            patient_select = gr.Dropdown(explorer.patients, label="Select Patient", value=explorer.patients[0] if explorer.patients else None)
            axis_select = gr.Radio(["Axial (Z)", "Coronal (Y)", "Sagittal (X)"], label="View Axis", value="Axial (Z)")
            slice_slider = gr.Slider(0, 511, step=1, label="Slice Index", value=256)
            
            stats_output = gr.Markdown()
            
            with gr.Accordion("Advanced Settings", open=False):
                gr.Markdown("Mask processing controls...")
        
        with gr.Column(scale=2):
            output_image = gr.Image(label="Slice Visualization", elem_id="slice-img", interactive=False)
            
    inputs = [patient_select, axis_select, slice_slider]
    outputs = [output_image, stats_output]
    
    patient_select.change(update_ui, inputs, outputs)
    axis_select.change(update_ui, inputs, outputs)
    slice_slider.change(update_ui, inputs, outputs)
    
    demo.load(update_ui, inputs, outputs)

if __name__ == "__main__":
    print("Launching Mask Visualizer...")
    # Fix for Gradio 6.0: move theme and css to launch()
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Ocean(), css=css, share=False)
