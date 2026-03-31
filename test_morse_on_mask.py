import numpy as np
import morse_3d
import time
import psutil
import os
import gc

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)  # GB

def test_morse_on_mask(mask_path):
    print(f"\n[DMT Test] Mask: {mask_path}")
    
    # 1. Load data
    print("  Loading mask...")
    start_load = time.time()
    mask = np.load(mask_path)['array'].astype(np.float64) # Must be float64 for morse_3d
    end_load = time.time()
    print(f"  Load time: {end_load - start_load:.2f}s")
    
    # 2. Performance Measurement
    start_mem = get_memory_usage()
    print(f"  Base Memory: {start_mem:.2f} GB")
    
    print("  Running extract_persistence_3d_morse...")
    start_time = time.time()
    
    try:
        results = morse_3d.extract_persistence_3d_morse(mask)
        end_time = time.time()
        end_mem = get_memory_usage()
        
        # Calculate stats
        print(f"  \n--- Compute Resource Analysis ---")
        print(f"  Total Duration: {end_time - start_time:.4f}s")
        print(f"  Memory Increase: {end_mem - start_mem:.4f} GB")
        print(f"  Final Memory: {end_mem:.2f} GB")
        
        print(f"  \n--- Result Statistics ---")
        print(f"  Number of persistence pairs: {len(results['births'])}")
        
        # Superlevel: Persistence = Birth - Death (since Birth > Death)
        # Note: Essential features have Death = -inf
        b = results['births']
        d = results['deaths']
        m_dim = results['dims']
        
        # Filter for finite pairs to calculate statistics
        finite_mask = np.isfinite(d)
        if np.any(finite_mask):
            pers = b[finite_mask] - d[finite_mask]
            print(f"  Max Persistence (Finite): {np.max(pers):.4f}")
            print(f"  Avg Persistence (Finite): {np.mean(pers):.4f}")
        
        # Dimension-wise Analysis
        print("\n--- Homology Dimension Alignment (v3.0) ---")
        for dim in [0, 1, 2]:
            count = np.sum(m_dim == dim)
            essential = np.sum((m_dim == dim) & (~np.isfinite(d)))
            print(f"  H{dim} (Dim {dim} Birth): {count} total ({essential} essential)")
        
        # Save sample results to CSV? No, just print.
        
    except Exception as e:
        print(f"  DMT Execution FAILED: {e}")
    
    # Clean up
    del mask
    if 'results' in locals(): del results
    gc.collect()

if __name__ == "__main__":
    import sys
    # Select file from command line or default
    mask_file = sys.argv[1] if len(sys.argv) > 1 else "lung_mask/patient-id-017.npz"
    
    if os.path.exists(mask_file):
        test_morse_on_mask(mask_file)
    else:
        print(f"File {mask_file} not found.")
