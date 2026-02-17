import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

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
    
    def _get_attn_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask[:, :self.prefix_tokens] = 1
        return mask
    
    def clear_cache(self):
        self.cached_mean = None
        self.cached_std = None


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, hid_dim)
        self.output_layer = nn.Linear(hid_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        hid = self.act(self.hidden_layer(x))
        out = self.output_layer(hid)
        res = self.residual_layer(x)
        out = out+res
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, last=False):
        super().__init__()
        assert d_model%n_heads==0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.head_dim = d_model//n_heads
        self.n_heads = n_heads

        self.rope = RotaryEmbedding(dim=self.head_dim//2)
        self.prefix_tokens = 8

        self.k_cache = None
        self.v_cache = None

        self.last = last
    
    def forward(self, q):
        bs, context, dim = q.size()
        offset = 0
        mask = torch.tril(torch.ones(q.size(-2), q.size(-2)))
        mask[:, :self.prefix_tokens] = 1
        mask = mask.bool().to(q.device)

        k = q
        v = q

        if self.last:
            q = q[:, -1:, :]
            mask=None
            offset += (context - 1)

        q = self.WQ(q).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.WK(k).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.WV(v).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)

        if self.k_cache is not None and self.v_cache is not None:
            offset += self.k_cache.size(2)
            mask = None
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)

        self.k_cache = k
        self.v_cache = v

        q = self.rope.rotate_queries_or_keys(q, offset=offset)
        k = self.rope.rotate_queries_or_keys(k)

        values = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=mask)

        values = values.transpose(1, 2).reshape(bs, -1, dim)
        values = self.out_proj(values)
        return values
    
    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None
    
class FeedForward(nn.Module):
    def __init__(self, d_model, multiple_of=256):
        super().__init__()
        hidden_dim = d_model*4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.w2(self.act(self.w1(x)) * self.w3(x))
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, last):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, last=last)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model)
    
    def forward(self, x):
        out_attn = self.attn(self.ln1((x)))
        x = x + out_attn
        out = x + self.ff(self.ln2(x))
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, last=False)
                for _ in range(n_layers-1)
            ]
        )
        self.layers.append(TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, last=True))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
class PatchFM(nn.Module, PyTorchModelHubMixin): 
    def __init__(self, patch_len, d_model, n_heads, n_layers_encoder, revin_config_name, use_asinh, 
                quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dropout=0.):
        super().__init__()
        
        self.patch_len = patch_len
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.n_quantiles = len(self.quantiles)

        if revin_config_name == "PrefixRevIN":
            self.revin = PrefixRevIN(use_asinh=use_asinh)
        else:
            raise NotImplementedError(f"RevIN config '{revin_config_name}' not implemented.")
        
        self.proj_embedding = ResidualBlock(in_dim=patch_len, hid_dim=2*patch_len, out_dim=d_model)
        self.transformer_encoder = TransformerEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers_encoder)
        self.proj_output = ResidualBlock(in_dim=d_model, hid_dim=2*d_model, out_dim=patch_len*self.n_quantiles)

    @torch.inference_mode()
    def forecast(self, x, target_len=None):
        if target_len is None:
            target_len=self.patch_len
        x = rearrange(x, "b (pn pl) -> b pn pl", pl=self.patch_len)

        rollouts = -(-target_len // self.patch_len)  # ceil division
        predictions = []
        for _ in range(rollouts):
                
            # Forward pass
            x = self.revin(x, mode="norm")
            x = self.proj_embedding(x)
            x = self.transformer_encoder(x)
            x = x[:, -1:, :]  # Keep only the last patch for autoregressive forecasting
            forecasting = self.proj_output(x)
            forecasting = self.revin(forecasting, mode="denorm")

            # Reshape to (bs, patch_num, patch_len, n_quantiles)
            forecasting = rearrange(
                forecasting, "b 1 (pl q) -> b 1 pl q", 
                pl=self.patch_len, q=self.n_quantiles
            )
            
            # Take median quantile (index 4)
            patch_median = forecasting[:, -1:, :, 4].detach()
            predictions.append(patch_median[:, 0, :])

            # Append median patch for next rollout
            x = patch_median.clone()
        
        predictions = torch.cat(predictions, dim=1)
        predictions = predictions[:, :target_len]

        self.clear_cache()
        return predictions
        
    def clear_cache(self):
        self.revin.clear_cache()    
        for layer in self.transformer_encoder.layers:
            layer.attn.clear_cache()  
    
def get_prefix_kv_model(device="cpu", use_asinh=True):

    print(
        f"This configuration is supported only for PrefixRevIN with "
        f"context length > 256 (prefix × patch_len). "
        f"Current setting: {'asinh' if use_asinh else 'no asinh'}."
    )    
    
    model = PatchFM.from_pretrained(f"vilhess/PatchFM-PrefixRevIN-{'asinh' if use_asinh else 'noasinh'}").eval()  
    model.to(device)
    return model