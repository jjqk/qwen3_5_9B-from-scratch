import torch
from torch import nn
from einops import rearrange
from torch import einsum
from modules.rmsnorm import RMSNorm
from modules.pos_enc import apply_rope


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention with partial RoPE and optional sigmoid output gate.
    Used as the 'full_attention' layer in Qwen3.5 (every 4th layer).
    Qwen3.5-9B: n_heads=16, num_kv_groups=4, head_dim=256.
    """

    def __init__(self, idim, n_heads, num_kv_groups, head_dim, dtype):
        super().__init__()
        self.idim = idim
        self.n_heads = n_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.group_size = n_heads // num_kv_groups
        self.odim = n_heads * head_dim
        self.scale = head_dim ** -0.5

        # Q and output gate are tied into a single 2× projection (HF convention).
        self.W_query = nn.Linear(idim, self.odim * 2, dtype=dtype, bias=False)
        self.k_proj = nn.Linear(idim, head_dim * num_kv_groups, dtype=dtype, bias=False)
        self.v_proj = nn.Linear(idim, head_dim * num_kv_groups, dtype=dtype, bias=False)
        self.o_proj = nn.Linear(self.odim, idim, dtype=dtype, bias=False)

        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, x, cos, sin, mask=None):
        b, L, _ = x.shape

        # Combined Q + gate projection (2× linear), then split
        q_raw = self.W_query(x)                              # (B, L, odim * 2)
        q_raw = q_raw.view(b, L, self.n_heads, self.head_dim * 2)
        q, gate = torch.chunk(q_raw, 2, dim=-1)              # each (B, L, H, head_dim)
        gate = gate.reshape(b, L, self.odim)                 # (B, L, odim)
        q = q.transpose(1, 2)                                # (B, H, L, head_dim)

        k = self.k_proj(x)   # (B, L, head_dim * num_kv_groups)
        v = self.v_proj(x)   # (B, L, head_dim * num_kv_groups)
        k = rearrange(k, 'b l (g d) -> b g l d', g=self.num_kv_groups)
        v = rearrange(v, 'b l (g d) -> b g l d', g=self.num_kv_groups)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots.masked_fill(mask, -torch.inf)
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)')

        # Qwen3.5 full-attention uses a gated Q projection
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)