import torch
import torch.nn as nn

class CausalRevIN(nn.Module):
    def __init__(self, eps=1e-5, use_asinh=True):
        super().__init__()
        self.eps = eps
        self.use_asinh = use_asinh
        self.cached_mean = None
        self.cached_std = None

        self.cached_cumsum_x = None
        self.cached_cumsum_x2 = None
        self.cached_counts = None

    def forward(self, x, mode):
        assert x.dim() == 3, "Input tensor must be (batch, n_patches, patch_len)"

        x64 = x.double()

        if mode == "norm":
            mean, std = self._get_statistics(x64)
            self.cached_mean, self.cached_std = mean[:, -1:].detach(), std[:, -1:].detach()
            out = (x64 - mean) / std
            if self.use_asinh:
                out = torch.asinh(out)

        elif mode == "denorm":
            assert self.cached_mean is not None and self.cached_std is not None, \
                "Call forward(..., 'norm') before 'denorm'"
            if self.use_asinh:
                x64 = torch.sinh(x64)
            out = x64 * self.cached_std + self.cached_mean

        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented.")

        return out.float()

    def _get_statistics(self, x):
        """
        Numerically stable mean and variance computation using 
        incremental mean and variance along the patch dimension.
        x: (B, P, L) float64
        Returns: mean, std (both (B, P, 1))
        """
        B, P, L = x.shape

        nan_counts = torch.isnan(x).sum(-1, keepdim=True)
        nan_counts = torch.cumsum(nan_counts, dim=1)

        counts = torch.arange(1, P+1, device=x.device).view(1, P, 1).repeat(B, 1, 1) * L
        counts = counts - nan_counts
    
        if self.cached_counts is not None:
            factor = B//self.cached_counts.size(0)
            self.cached_counts = self.cached_counts.repeat_interleave(factor, dim=0)
            counts += self.cached_counts
        self.cached_counts = counts[:, -1:, :]

        cumsum_x = torch.cumsum(x.nansum(dim=-1, keepdim=True), dim=1)
        if self.cached_cumsum_x is not None:
            self.cached_cumsum_x = self.cached_cumsum_x.repeat_interleave(factor, dim=0)
            cumsum_x += self.cached_cumsum_x
        self.cached_cumsum_x = cumsum_x[:, -1:, :]

        mean = cumsum_x / counts


        cumsum_x2 = torch.cumsum((x**2).nansum(dim=-1, keepdim=True), dim=1)
        if self.cached_cumsum_x2 is not None:
            self.cached_cumsum_x2 = self.cached_cumsum_x2.repeat_interleave(factor, dim=0)
            cumsum_x2 += self.cached_cumsum_x2
        self.cached_cumsum_x2 = cumsum_x2[:, -1:, :]

        var = (cumsum_x2 - 2 * mean * cumsum_x + counts * mean**2) / counts
        std = torch.sqrt(var + 1e-5)

        return mean, std
    
    def clear_cache(self):
        self.cached_cumsum_x = None
        self.cached_cumsum_x2 = None
        self.cached_counts = None
        self.cached_mean = None
        self.cached_std = None

class PrefixRevIN(nn.Module):
    def __init__(self, eps=1e-5, use_asinh=True, prefix_tokens=8):
        super().__init__()
        self.eps = eps
        self.cached_mean = None
        self.cached_std = None
        self.asinh = use_asinh
        self.prefix_tokens = prefix_tokens

    def forward(self, x, mode: str):
        assert x.dim() == 3, "Input tensor must be (batch, n_patches, patch_len)"

        if mode == "norm":

            if self.cached_mean is not None and self.cached_std is not None:
                factor = x.size(0) // self.cached_mean.size(0)
                self.cached_mean = self.cached_mean.repeat_interleave(factor, dim=0)
                self.cached_std = self.cached_std.repeat_interleave(factor, dim=0)
                mean, std = self.cached_mean, self.cached_std
            else:
                mean, std = self._get_statistics(x)
                self.cached_mean, self.cached_std = mean.detach(), std.detach()

            out = (x - mean) / std
            if self.asinh:
                out = torch.asinh(out)

        elif mode == "denorm":
            assert self.cached_mean is not None and self.cached_std is not None, \
                "Call forward(..., 'norm') before 'denorm'"
            if self.asinh:
                x = torch.sinh(x)
            out = x * self.cached_std + self.cached_mean
            
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented.")
        return out

    def _get_statistics(self, x):
        mean = x[:, :self.prefix_tokens, :].mean(dim=(-1, -2), keepdim=True)
        std = x[:, :self.prefix_tokens, :].std(dim=(-1, -2), keepdim=True) + self.eps
        return mean, std
    
    def clear_cache(self):
        self.cached_mean = None
        self.cached_std = None

class NoRevIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mode):
        return x

    def clear_cache(self):
        pass