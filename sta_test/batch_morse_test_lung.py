import os
import glob
import time
import psutil
import pandas as pd
import numpy as np
import nibabel as nib
import gc
import sys

# Ensure morse_3d from parent directory is loadable
sys.path.append('/home/DMT_dev')
import morse_3d

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)  # GB

def main():
    data_dir = '/home/first_SM/nnUNet_raw/Dataset001_LungVessel/labelsTr'
    output_dir = '/home/DMT_dev/sta_test'
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, 'morse_processing_log_lung.txt')
    report_csv = os.path.join(output_dir, 'morse_statistics_report_lung.csv')
    
    files = sorted(glob.glob(os.path.join(data_dir, '*.nii.gz')))
    if not files:
        print(f"No files found in {data_dir}")
        return
        
    print(f"Found {len(files)} files to process in LungVessel dataset.")
    
    stats_list = []
    
    # Check if we should resume
    if os.path.exists(report_csv):
        try:
            old_df = pd.read_csv(report_csv)
            stats_list = old_df.to_dict('records')
            processed_files = set(old_df['Filename'].tolist())
            print(f"Resuming: {len(processed_files)} files already processed.")
        except:
            processed_files = set()
    else:
        processed_files = set()

    f_log = open(log_file, 'a')
    f_log.write(f"\n\n--- NEW RUN START: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    f_log.flush()
        
    for idx, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        if filename in processed_files:
            continue
            
        print(f"[{idx+1}/{len(files)}] Processing {filename}...")
        f_log.write(f"\n--- Processing: {filename} ---\n")
        f_log.flush()
        
        try:
            start_mem = get_memory_usage()
            
            # Load NIfTI
            load_start = time.time()
            img = nib.load(file_path)
            # Lung vessel labels often have 1 or 2. Ensure we treat as floating point for the solver
            data = img.get_fdata().astype(np.float64)
            load_time = time.time() - load_start
            f_log.write(f"Load Time: {load_time:.2f}s, Shape: {data.shape}\n")
            f_log.flush()
            
            # Run Morse 3D
            morse_start = time.time()
            results = morse_3d.extract_persistence_3d_morse(data)
            morse_time = time.time() - morse_start
            end_mem = get_memory_usage()
            mem_inc = end_mem - start_mem
            
            f_log.write(f"Morse Duration: {morse_time:.4f}s\n")
            f_log.write(f"Memory Diff: {mem_inc:.4f} GB (Current: {end_mem:.2f} GB)\n")
            f_log.flush()
            
            # Analyze Results
            b = results['births']
            d = results['deaths']
            m_dim = results['dims']
            
            total_pairs = len(b)
            f_log.write(f"Total Persistence Pairs: {total_pairs}\n")
            
            file_stats = {
                'Filename': filename,
                'Shape': str(data.shape),
                'Load_Time_sec': load_time,
                'Morse_Time_sec': morse_time,
                'Memory_Inc_GB': mem_inc,
                'Total_Pairs': total_pairs
            }
            
            # Stats per dimension (0, 1, 2)
            for dim in [0, 1, 2]:
                mask_dim = (m_dim == dim)
                count = np.sum(mask_dim)
                essential = np.sum(mask_dim & (~np.isfinite(d)))
                finite_mask = mask_dim & np.isfinite(d)
                
                max_pers = 0.0
                avg_pers = 0.0
                median_pers = 0.0
                if np.any(finite_mask):
                    pers = b[finite_mask] - d[finite_mask]
                    max_pers = float(np.max(pers))
                    avg_pers = float(np.mean(pers))
                    median_pers = float(np.median(pers))
                    
                f_log.write(f"  H{dim} Features: {count} (Essential: {essential})\n")
                if count > essential:
                    f_log.write(f"    Max Persistence: {max_pers:.4f}\n")
                    f_log.write(f"    Avg Persistence: {avg_pers:.4f}\n")
                    f_log.write(f"    Median Persistence: {median_pers:.4f}\n")
                    
                file_stats[f'H{dim}_Total'] = count
                file_stats[f'H{dim}_Essential'] = essential
                file_stats[f'H{dim}_Max_Pers'] = max_pers
                file_stats[f'H{dim}_Avg_Pers'] = avg_pers
                file_stats[f'H{dim}_Median_Pers'] = median_pers
            
            f_log.flush()
            stats_list.append(file_stats)
            
            # Save CSV Report Incrementally
            df = pd.DataFrame(stats_list)
            df.to_csv(report_csv, index=False)
            
            # Cleanup
            del data
            del img
            del results
            b = d = m_dim = None
            gc.collect()
            
            # EXIT AFTER ONE FILE TO PREVENT MEMORY ACCUMULATION
            # A shell loop will restart this script to continue
            break
            
        except Exception as e:
            err_msg = f"ERROR processing {filename}: {str(e)}"
            print(f"  {err_msg}")
            f_log.write(f"  {err_msg}\n")
            f_log.flush()
                
    f_log.write("\n" + "="*50 + "\n")
    f_log.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f_log.close()
    
    # Save Final CSV Report
    if stats_list:
        df = pd.DataFrame(stats_list)
        df.to_csv(report_csv, index=False)
        print(f"\nProcessing complete for LungVessel!")
        
        # Print summary
        print("\n--- Summary Report (LungVessel) ---")
        print(f"Total processed files: {len(df)}")
        print(f"Average processing time: {df['Morse_Time_sec'].mean():.2f}s")
        print(f"Average H0 count: {df['H0_Total'].mean():.2f}")
        print(f"Average H1 count: {df['H1_Total'].mean():.2f}")
        print(f"Average H2 count: {df['H2_Total'].mean():.2f}")
            
if __name__ == '__main__':
    main()
