import os
import numpy as np

root_dir = "../raw_results"
save_dir = "../processed_results"

# --------------------------------------------------
# 1️⃣ Collect all sequence lengths first
# --------------------------------------------------
seq_lens = set()
for model_name in os.listdir(root_dir):
    model_dir = os.path.join(root_dir, model_name)
    if not os.path.isdir(model_dir):  # ✅ Added safety check
        continue
    for seq_len in os.listdir(model_dir):
        seq_lens.add(seq_len)

# --------------------------------------------------
# 2️⃣ Loop over sequence lengths FIRST
# --------------------------------------------------
for seq_len in seq_lens:
    print(f"\nProcessing seq_len = {seq_len}")
    seq_length = int(seq_len)
    
    # Window boundaries
    gift_window = np.load(f"../dataset/windows/gift_{seq_len}.npy").tolist()
    utsd_window = np.load(f"../dataset/windows/utsd_{seq_len}.npy").tolist()
    artificial_window = np.load(f"../dataset/windows/artificial_{seq_len}.npy").tolist()
    
    # --------------------------------------------------
    # 3️⃣ Now loop over models
    # --------------------------------------------------
    for model_name in os.listdir(root_dir):
        if model_name!="NoRevIN_False":
            continue

        seq_len_dir = os.path.join(root_dir, model_name, seq_len)
        if not os.path.exists(seq_len_dir):
            continue
            
        for dataset_results in os.listdir(seq_len_dir):
            if not dataset_results.endswith(".npz"):
                continue
                            
            try:
                results = np.load(os.path.join(seq_len_dir, dataset_results))
            except Exception as e:
                print(f"Error loading {os.path.join(seq_len_dir, dataset_results)}: {e}")
                continue
            
            # Select correct window list
            if "gift" in dataset_results.lower():
                windows = [0] + gift_window
            elif "utsd" in dataset_results.lower():
                windows = [0] + utsd_window
            else:
                windows = artificial_window
            
            # ✅ Added safety check for empty slices
            if len(windows) < 2:
                print(f"⚠️  Skipping {dataset_results}: not enough windows")
                continue
            
            aggregated_results = {}
            for key in results.keys():
                values = results[key]
                aggregated_means = []
                
                for i in range(len(windows) - 1):
                    slice_data = values[windows[i]:windows[i+1]]
                    # ✅ Check for empty slice BEFORE taking mean
                    if len(slice_data) == 0:
                        print(len(values), windows[-1])
                        print(f"⚠️  Empty slice at window {i} for {key}")
                        aggregated_means.append(np.nan)  # or 0, or skip

                    elif np.all(np.isnan(slice_data)):
                        pass # do not append anything if all values are NaN (happened for MASE when context is constant)
                    
                    else:
                        aggregated_means.append(np.nanmean(slice_data))
                
                aggregated_results[key] = np.array(aggregated_means)
            
            # Save
            save_path = os.path.join(save_dir, model_name, seq_len)
            os.makedirs(save_path, exist_ok=True)
            np.savez_compressed(
                os.path.join(save_path, dataset_results),
                **aggregated_results
            )
            print(f"Saved → {os.path.join(save_path, dataset_results)}")