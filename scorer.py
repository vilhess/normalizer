import numpy as np
import torch


class MetricScorer:
    def __init__(self, max_pred_len=256, patch_len=8):
        self.max_pred_len = max_pred_len
        self.patch_len = patch_len

        self.predictions = []
        self.quantiles_predictions = []
        self.targets = []
        self.contexts = []

    def update(self, preds, targets, contexts):

        self.predictions.append(preds[0])
        self.quantiles_predictions.append(preds[1])

        self.targets.append(targets)
        self.contexts.append(contexts)

    def compute(self):

        final_results = {}

        preds = torch.cat(self.predictions, dim=0).detach().cpu()
        quantiles_preds = torch.cat(self.quantiles_predictions, dim=0).detach().cpu()
        targets = torch.cat(self.targets, dim=0).detach().cpu()
        contexts = torch.cat(self.contexts, dim=0).detach().cpu()

        for end in range(self.patch_len, self.max_pred_len + 1, self.patch_len):
            pred_slice = preds[:, :end]
            target_slice = targets[:, :end]
            mae = torch.mean(torch.abs(pred_slice - target_slice), dim=-1)
            rmse = torch.sqrt(torch.mean((pred_slice - target_slice) ** 2, dim=-1))
            mase_score = mase(pred_slice, contexts, target_slice)

            quant_slice = quantiles_preds[:, :end]
            sql_score = sql(quant_slice, contexts, target_slice)

            final_results[f"MAE_{end}"] = mae.numpy().astype(np.float32)
            final_results[f"RMSE_{end}"] = rmse.numpy().astype(np.float32)
            final_results[f"MASE_{end}"] = mase_score.numpy().astype(np.float32)
            final_results[f"SQL_{end}"] = sql_score.numpy().astype(
                np.float32
            )  # new metric
        return final_results

    def reset(self):
        self.predictions = []
        self.quantiles_predictions = []
        self.targets = []
        self.contexts = []


def mase(forecast, context, ground_truth):
    numerator = torch.mean(torch.abs(forecast - ground_truth), dim=-1)
    divider = torch.mean(torch.abs(context[:, :-1] - context[:, 1:]), dim=-1)
    # possibly nan if context has constant values, which would lead to division by zero. In that case, we set MASE to NaN
    score_mase = numerator / divider
    score_mase = torch.where(divider == 0, torch.nan, score_mase)
    return score_mase


def save_results_npz(results, filename):
    np.savez_compressed(filename, **results)


# Scaled Quantile Loss (SQL) implementation.
def sql(
    forecast,
    context,
    ground_truth,
    quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
):
    quantiles = torch.tensor(quantiles, device=forecast.device, dtype=forecast.dtype)
    assert (
        forecast[:, :, 0].shape == ground_truth.shape
    ), "Forecast and ground truth must have the same shape"
    assert forecast.shape[-1] == len(
        quantiles
    ), "The last dimension of forecast must match the number of quantiles"

    tar_quantiles = ground_truth.unsqueeze(-1)
    errors = tar_quantiles - forecast
    numerator = torch.max(quantiles * errors, (quantiles - 1) * errors)

    divider = (
        torch.mean(torch.abs(context[:, :-1] - context[:, 1:]), dim=-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    # possibly nan if context has constant values, which would lead to division by zero. In that case, we set SQL to NaN
    score_mase = numerator / divider
    score_mase = torch.where(divider == 0, torch.nan, score_mase)
    score_mase = score_mase.mean(dim=(1, 2))  # average over patches and quantiles
    return score_mase
