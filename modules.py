import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from normalizer import RevIN, CausalRevIN, WURevIN

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
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
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model%n_heads==0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.head_dim = d_model//n_heads
        self.n_heads = n_heads
        self.rope = RotaryEmbedding(dim=self.head_dim//2)
    
    def forward(self, q, wu_tokens=None):
        bs, context, dim = q.size()
        k = q
        v = q
        q = self.WQ(q).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.WK(k).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.WV(v).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        q  = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)
        if (wu_tokens is None) or (wu_tokens==0):
            values = nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        elif wu_tokens:
            mask = torch.tril(torch.ones(q.size(-2), q.size(-2)))
            mask[:, :wu_tokens] = 1
            mask = mask.bool().to(q.device)
            values = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
            )
        values = values.transpose(1, 2).reshape(bs, -1, dim)
        values = self.out_proj(values)
        return values
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1, multiple_of=256):
        super().__init__()
        hidden_dim = d_model*4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w2(self.act(self.w1(x)) * self.w3(x))
        return self.dp(x)
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, dropout=dropout)
    
    def forward(self, x, wu_tokens=None):
        out_attn = self.attn(self.ln1((x)), wu_tokens=wu_tokens)
        x = x + out_attn
        out = x + self.ff(self.ln2(x))
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, wu_tokens=None):
        for layer in self.layers:
            x = layer(x, wu_tokens=wu_tokens)
        return self.norm(x)
    
class PatchFM(nn.Module, PyTorchModelHubMixin): 
    def __init__(self, patch_len, d_model, n_heads, n_layers_encoder, revin_config_name, use_asinh, 
                quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dropout=0.):
        super().__init__()
        
        self.patch_len = patch_len
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.n_quantiles = len(self.quantiles)

        if revin_config_name == "CausalRevIN":
            self.revin = CausalRevIN(asinh=use_asinh)
            self.wu_tokens = None
        elif revin_config_name == "RevIN":
            self.revin = RevIN(asinh=use_asinh)
            self.wu_tokens = None
        elif revin_config_name == "WURevIN":
            self.wu_tokens = 8
            self.revin = WURevIN(asinh=use_asinh, wu_tokens=self.wu_tokens)
        else:
            raise NotImplementedError(f"RevIN config '{revin_config_name}' not implemented.")
        
        self.proj_embedding = ResidualBlock(in_dim=patch_len, hid_dim=2*patch_len, out_dim=d_model, dropout=dropout)
        self.dp = nn.Dropout(dropout)
        self.transformer_encoder = TransformerEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers_encoder, dropout=dropout)
        self.proj_output = ResidualBlock(in_dim=d_model, hid_dim=2*d_model, out_dim=patch_len*self.n_quantiles, dropout=dropout)

    def forward(self, x): 

        bs, ws = x.size()
        x = rearrange(x, "b (pn pl) -> b pn pl", pl=self.patch_len)  # Reshape to (bs, patch_num, patch_len)
        x = self.revin(x, mode="norm")
        x = self.proj_embedding(x) # bs, pn, d_model
        x = self.dp(x)
        x = self.transformer_encoder(x, wu_tokens=self.wu_tokens) # bs, pn, d_model
        forecasting = self.proj_output(x)  # bs, pn, patch_len  
        forecasting = self.revin(forecasting, mode="denorm")
        forecasting = rearrange(forecasting, "b pn (pl q) -> b pn pl q", pl=self.patch_len, q=self.n_quantiles)

        return forecasting[:, :, :, self.quantiles.index(0.5)]  # Return median predictions only here
        
    @torch.inference_mode()
    def forecast(self, x, target_len=None):
        if target_len is None:
            target_len = self.patch_len
        n_patches = max(target_len // self.patch_len, 1)
        preds = []
        x_input = x
        for _ in range(n_patches):
            out = self.forward(x_input)
            next_patch = out[:, -1:, :]  # Get the last predicted patch
            preds.append(next_patch)
            x_input = torch.cat([x_input, rearrange(next_patch, "b 1 n -> b n")], dim=1)
        preds = torch.cat(preds, dim=1)
        preds = preds.flatten(1, -1)
        return preds
    
    @torch.inference_mode()
    def forecast_causal(self, x):
        out = self.forward(x)
        return out
    
def get_model(revin_strategy, use_asinh, device='cpu'):
    model = PatchFM.from_pretrained(f"vilhess/PatchFM-{revin_strategy}-{'asinh' if use_asinh else 'noasinh'}").eval()
    model.to(device)
    return model