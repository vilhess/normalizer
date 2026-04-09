import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, last=False):
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.head_dim = d_model // n_heads
        self.n_heads = n_heads

        self.rope = RotaryEmbedding(dim=self.head_dim // 2)

        self.k_cache = None
        self.v_cache = None

        self.last = last

    def forward(self, q):
        bs, context, dim = q.size()
        offset = 0
        is_causal = True

        k = q
        v = q

        if self.last:
            q = q[:, -1:, :]
            is_causal = False
            offset += context - 1

        q = self.WQ(q).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.WK(k).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.WV(v).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)

        if self.k_cache is not None and self.v_cache is not None:
            offset += self.k_cache.size(2)
            is_causal = False
            factor = q.size(0) // self.k_cache.size(0)
            self.k_cache = torch.repeat_interleave(self.k_cache, repeats=factor, dim=0)
            self.v_cache = torch.repeat_interleave(self.v_cache, repeats=factor, dim=0)
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)

        self.k_cache = k
        self.v_cache = v

        q = self.rope.rotate_queries_or_keys(q, offset=offset)
        k = self.rope.rotate_queries_or_keys(k)

        values = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal
        )

        values = values.transpose(1, 2).reshape(bs, -1, dim)
        values = self.out_proj(values)
        return values

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None


class PrefixMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, last=False, prefix_tokens=8):
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.head_dim = d_model // n_heads
        self.n_heads = n_heads

        self.rope = RotaryEmbedding(dim=self.head_dim // 2)
        self.prefix_tokens = prefix_tokens

        self.k_cache = None
        self.v_cache = None

        self.last = last

    def forward(self, q):
        bs, context, dim = q.size()
        offset = 0
        mask = torch.tril(torch.ones(q.size(-2), q.size(-2)))
        mask[:, : self.prefix_tokens] = 1
        mask = mask.bool().to(q.device)

        k = q
        v = q

        if self.last:
            q = q[:, -1:, :]
            mask = None
            offset += context - 1

        q = self.WQ(q).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.WK(k).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.WV(v).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)

        if self.k_cache is not None and self.v_cache is not None:
            offset += self.k_cache.size(2)
            mask = None
            factor = q.size(0) // self.k_cache.size(0)
            self.k_cache = torch.repeat_interleave(self.k_cache, repeats=factor, dim=0)
            self.v_cache = torch.repeat_interleave(self.v_cache, repeats=factor, dim=0)
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)

        self.k_cache = k
        self.v_cache = v

        q = self.rope.rotate_queries_or_keys(q, offset=offset)
        k = self.rope.rotate_queries_or_keys(k)

        values = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False, attn_mask=mask
        )
        values = values.transpose(1, 2).reshape(bs, -1, dim)
        values = self.out_proj(values)
        return values

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None
