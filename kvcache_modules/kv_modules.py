import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

from kvcache_modules.kv_mha import PrefixMultiHeadAttention, CausalMultiHeadAttention
from kvcache_modules.kv_normalizer import PrefixRevIN, CausalRevIN, NoRevIN

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
    def __init__(self, d_model, n_heads, last, model_type="causal"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)

        if model_type == "causal":
            self.attn = CausalMultiHeadAttention(d_model=d_model, n_heads=n_heads, last=last)
        elif model_type == "prefix":
            self.attn = PrefixMultiHeadAttention(d_model=d_model, n_heads=n_heads, last=last)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model)
    
    def forward(self, x):
        out_attn = self.attn(self.ln1((x)))
        x = x + out_attn
        out = x + self.ff(self.ln2(x))
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, model_type="causal"):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, last=False, model_type=model_type)
                for _ in range(n_layers-1)
            ]
        )
        self.layers.append(TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, last=True, model_type=model_type))
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

        if revin_config_name == "CausalRevIN":
            model_type="causal"
            self.revin = CausalRevIN(use_asinh=use_asinh)

        elif revin_config_name == "NoRevIN":
            model_type="causal"
            if use_asinh:
                print("Sinh^(-1) won't be considered with this normalization")
            self.revin = NoRevIN()

        elif revin_config_name == "PrefixRevIN":
            print(f"Warining: PrefixRevIN KV Cache currently works only for context lengths longer or equal to 256 values.")
            model_type="prefix"
            self.revin = PrefixRevIN(use_asinh=use_asinh)

        else:
            raise NotImplementedError(f"RevIN config '{revin_config_name}' not implemented.")
        
        self.proj_embedding = ResidualBlock(in_dim=patch_len, hid_dim=2*patch_len, out_dim=d_model)
        self.transformer_encoder = TransformerEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers_encoder, model_type=model_type)
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
    
def get_kv_model(revin_strategy, use_asinh):
    if revin_strategy=="NoRevIN":
        model_name = f"vilhess/PatchFM-NoRevIN"
    else:
        model_name = f"vilhess/PatchFM-{revin_strategy}-{'asinh' if use_asinh else 'noasinh'}"
    model = PatchFM.from_pretrained(model_name).eval()  
    return model