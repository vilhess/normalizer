import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from rotary_embedding_torch import RotaryEmbedding

from kvcache_modules import get_kv_model
from normalizer import CausalRevIN, NoRevIN, PrefixRevIN, RevIN


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.0):
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
        out = out + res
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.rope = RotaryEmbedding(dim=self.head_dim // 2)

    def forward(self, q, prefix_tokens=None):
        bs, context, dim = q.size()
        k = q
        v = q
        q = self.WQ(q).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.WK(k).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.WV(v).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)
        if (prefix_tokens is None) or (prefix_tokens == 0):
            values = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        elif prefix_tokens:
            mask = torch.tril(torch.ones(q.size(-2), q.size(-2)))
            mask[:, :prefix_tokens] = 1
            mask = mask.bool().to(q.device)
            values = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        values = values.transpose(1, 2).reshape(bs, -1, dim)
        values = self.out_proj(values)
        return values


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1, multiple_of=256):
        super().__init__()
        hidden_dim = d_model * 4
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
        self.attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, dropout=dropout)

    def forward(self, x, prefix_tokens=None):
        out_attn = self.attn(self.ln1((x)), prefix_tokens=prefix_tokens)
        x = x + out_attn
        out = x + self.ff(self.ln2(x))
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model, n_heads=n_heads, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, prefix_tokens=None):
        for layer in self.layers:
            x = layer(x, prefix_tokens=prefix_tokens)
        return self.norm(x)


class PatchFM(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        patch_len,
        d_model,
        n_heads,
        n_layers_encoder,
        revin_config_name,
        use_asinh,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        dropout=0.0,
    ):
        super().__init__()

        self.patch_len = patch_len
        self.quantiles = (
            quantiles
            if quantiles is not None
            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        self.n_quantiles = len(self.quantiles)

        if revin_config_name == "CausalRevIN":
            self.revin = CausalRevIN(asinh=use_asinh)
            self.prefix_tokens = None
        elif revin_config_name == "RevIN" or revin_config_name == "OptimalRevIN":
            self.revin = RevIN(asinh=use_asinh)
            self.prefix_tokens = None
        elif revin_config_name == "PrefixRevIN":
            self.prefix_tokens = 8
            self.revin = PrefixRevIN(asinh=use_asinh, prefix_tokens=self.prefix_tokens)
        elif revin_config_name == "NoRevIN":
            if use_asinh:
                print(
                    "Warning: asinh transformation is not applied when using NoRevIN."
                )
            self.prefix_tokens = None
            self.revin = NoRevIN()
        else:
            raise NotImplementedError(
                f"RevIN config '{revin_config_name}' not implemented."
            )

        self.proj_embedding = ResidualBlock(
            in_dim=patch_len, hid_dim=2 * patch_len, out_dim=d_model, dropout=dropout
        )
        self.dp = nn.Dropout(dropout)
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers_encoder, dropout=dropout
        )
        self.proj_output = ResidualBlock(
            in_dim=d_model,
            hid_dim=2 * d_model,
            out_dim=patch_len * self.n_quantiles,
            dropout=dropout,
        )

    def forward(self, x, quantiles=False):

        bs, ws = x.size()
        x = rearrange(
            x, "b (pn pl) -> b pn pl", pl=self.patch_len
        )  # Reshape to (bs, patch_num, patch_len)
        x = self.revin(x, mode="norm")
        x = self.proj_embedding(x)  # bs, pn, d_model
        x = self.dp(x)
        x = self.transformer_encoder(
            x, prefix_tokens=self.prefix_tokens
        )  # bs, pn, d_model
        forecasting = self.proj_output(x)  # bs, pn, patch_len
        forecasting = self.revin(forecasting, mode="denorm")
        forecasting = rearrange(
            forecasting,
            "b pn (pl q) -> b pn pl q",
            pl=self.patch_len,
            q=self.n_quantiles,
        )

        if quantiles:
            return forecasting  # Return all quantiles
        else:
            return forecasting[
                :, :, :, self.quantiles.index(0.5)
            ]  # Return median predictions only here

    @torch.inference_mode()
    def forecast_causal(self, x):
        out = self.forward(x, quantiles=False)
        return out

    @torch.inference_mode()
    def forecast(self, x, target_len=None):

        if target_len is None:
            target_len = self.patch_len

        assert x.ndim in (
            1,
            2,
        ), f"Input dimension must be 1D (time) or 2D (batch, time), got {x.ndim}D."
        bs, ws = x.size()

        context = x.clone()

        rollouts = -(-target_len // self.patch_len)  # ceil division
        predictions = []

        forecasting = self.forward(
            x, quantiles=True
        )  # Get all quantiles for the initial context
        forecasting = forecasting[
            :, -1, :, :
        ]  # Keep only the last patch for autoregressive forecasting

        context_expanded = torch.repeat_interleave(
            context.unsqueeze(-1), repeats=self.n_quantiles, dim=-1
        )  # batch x ws x n_quantiles
        base_context_expanded = torch.cat(
            (context_expanded, forecasting), dim=1
        )  # batch x ws+patch_size x n_quantiles
        context_expanded = base_context_expanded.permute(0, 2, 1).reshape(
            bs * self.n_quantiles, base_context_expanded.size(1)
        )

        x = context_expanded
        q = torch.tensor(self.quantiles, device=x.device)

        predictions.append(forecasting)

        for _ in range(rollouts - 1):

            # Forward pass
            forecasting = self.forward(
                x, quantiles=True
            )  # batch*n_quantiles x patch_num x patch_len x n_quantiles
            forecasting = forecasting[
                :, -1, :, :
            ]  # batch*n_quantiles x patch_len x n_quantiles

            forecasting = rearrange(
                forecasting, "(b q) pl h -> b q pl h", q=self.n_quantiles
            )
            forecasting = forecasting.permute(0, 2, 1, 3).flatten(
                start_dim=-2
            )  # batch x patch_len x n_quantiles**2
            forecasting = torch.quantile(
                forecasting, q, dim=-1
            )  # n_quantiles x batch x patch_len
            forecasting = forecasting.permute(
                1, 2, 0
            )  # batch x patch_len x n_quantiles

            base_context_expanded = torch.cat(
                (base_context_expanded, forecasting), dim=1
            )  # # batch x ws+iter*patch_size x n_quantiles
            context_expanded = base_context_expanded.permute(0, 2, 1).reshape(
                bs * self.n_quantiles, base_context_expanded.size(1)
            )

            x = context_expanded
            predictions.append(forecasting)

        pred_quantiles = torch.cat(predictions, dim=1)
        pred_quantiles = pred_quantiles[:, :target_len, :]
        pred_median = pred_quantiles[:, :, 4]

        return pred_median, pred_quantiles


def get_model(
    revin_strategy, use_asinh, seq_len, kv_cache_if_possible=True, device="cpu"
):

    if (
        revin_strategy == "PrefixRevIN2"
    ):  # ablation study for prefix strategy replaced by naive during inference
        print(
            "Using PrefixRevIN2 strategy for ablation study: prefix replaced by naive (optimal) during inference."
        )
        revin_strategy = "PrefixRevIN"
        model = PatchFM.from_pretrained(
            f"vilhess/PatchFM-{revin_strategy}-{'asinh' if use_asinh else 'noasinh'}"
        ).eval()
        model.revin = RevIN(asinh=use_asinh)

    elif revin_strategy == "RevIN":
        model = PatchFM.from_pretrained(
            f"vilhess/PatchFM-{revin_strategy}-{'asinh' if use_asinh else 'noasinh'}"
        ).eval()

    elif revin_strategy == "CausalRevIN":
        if kv_cache_if_possible:
            print("Using CausalRevIN with KV caching for inference.")
            model = get_kv_model(revin_strategy=revin_strategy, use_asinh=use_asinh)
        else:
            model = PatchFM.from_pretrained(
                f"vilhess/PatchFM-{revin_strategy}-{'asinh' if use_asinh else 'noasinh'}"
            ).eval()

    elif revin_strategy == "PrefixRevIN":
        if kv_cache_if_possible and seq_len >= 256:
            print("Using PrefixRevIN with KV caching for inference.")
            model = get_kv_model(revin_strategy=revin_strategy, use_asinh=use_asinh)
        else:
            model = PatchFM.from_pretrained(
                f"vilhess/PatchFM-{revin_strategy}-{'asinh' if use_asinh else 'noasinh'}"
            ).eval()

    elif revin_strategy == "NoRevIN":
        assert not use_asinh, "NoRevIN strategy does not use asinh transformation."
        if kv_cache_if_possible:
            print("Using NoRevIN with KV caching for inference.")
            model = get_kv_model(revin_strategy=revin_strategy, use_asinh=use_asinh)
        else:
            model = PatchFM.from_pretrained(f"vilhess/PatchFM-{revin_strategy}").eval()

    elif revin_strategy == "OptimalRevIN":
        assert not use_asinh, "OptimalRevIN strategy does not use asinh transformation."
        model = PatchFM.from_pretrained(
            f"vilhess/PatchFM-{revin_strategy}").eval()
        
    else:
        raise NotImplementedError(f"RevIN strategy '{revin_strategy}' not implemented.")

    model.to(device)
    return model
