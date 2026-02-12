import torch
import numpy as np

class MetricScorer:
    def __init__(self, max_pred_len=256, patch_len=8):
        self.max_pred_len = max_pred_len
        self.patch_len = patch_len

        self.predictions = []
        self.targets = []
        self.contexts = []

    def update(self, preds, targets, contexts):
        self.predictions.append(preds)
        self.targets.append(targets)
        self.contexts.append(contexts)

    def compute(self):
        
        final_results = {}

        preds = torch.cat(self.predictions, dim=0).detach().cpu()
        targets = torch.cat(self.targets, dim=0).detach().cpu()
        contexts = torch.cat(self.contexts, dim=0).detach().cpu()
        
        for end in range(self.patch_len, self.max_pred_len + 1, self.patch_len):
            pred_slice = preds[:, :end]
            target_slice = targets[:, :end]
            mae = torch.mean(torch.abs(pred_slice - target_slice), dim=-1)
            rmse = torch.sqrt(torch.mean((pred_slice - target_slice) ** 2, dim=-1))
            mase_score = mase(pred_slice, contexts, target_slice)

            final_results[f"MAE_{end}"] = mae.numpy().astype(np.float32) 
            final_results[f"RMSE_{end}"] = rmse.numpy().astype(np.float32) 
            final_results[f"MASE_{end}"] = mase_score.numpy().astype(np.float32)
        return final_results

    def reset(self):
        self.predictions = []
        self.targets = []
        self.contexts = []


def mase(forecast, context, ground_truth):
    numerator = torch.mean(torch.abs(forecast - ground_truth), dim=-1)
    divider = torch.mean(torch.abs(context[:, :-1] - context[:, 1:]), dim=-1)

    remove_idx = divider < 1e-8
    divider = divider[~remove_idx]
    numerator = numerator[~remove_idx]

    print(f"prc removed: {100 * remove_idx.sum().item() / len(remove_idx):.2f}%")
    
    score_mase = numerator / divider
    return score_mase

def save_results_npz(results, filename):
    np.savez_compressed(filename, **results)