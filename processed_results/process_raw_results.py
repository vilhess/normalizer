import os
import numpy as np 

root_dir="../raw_results"
save_dir="../processed_results"

for model_name in os.listdir(root_dir):
    model_dir = os.path.join(root_dir, model_name)
    for seq_len in os.listdir(model_dir):
        seq_len_dir = os.path.join(model_dir, seq_len)
        for dataset_results in os.listdir(seq_len_dir):
            if dataset_results.endswith(".npz"):
                print(f"Loading results from {dataset_results}...")
                results = np.load(os.path.join(seq_len_dir, dataset_results))
                aggregated_results = {}
                for key in results.keys():
                    aggregated_results[key] = np.mean(results[key])  
                save_path = os.path.join(save_dir, model_name, seq_len)
                os.makedirs(save_path, exist_ok=True)
                np.savez_compressed(os.path.join(save_path, dataset_results), **aggregated_results)
                print(f"Saved aggregated results to {os.path.join(save_path, dataset_results)}")