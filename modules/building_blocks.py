import torch.nn.functional as F
from torch import nn
from modules.full_attention import GroupedQueryAttention
from modules.linear_attention import GatedDeltaNetAttention
from modules.rmsnorm import RMSNorm


class GatedFeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dtype):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = F.silu(x_fc1) * x_fc2
        return self.fc3(x)

class GatedAttentionBlock(nn.Module):
    """
    Full (quadratic-softmax) GQA block with sigmoid output gate.
    Instantiated every 4th layer (full_attention_interval=4).
    Qwen3.5-9B: n_heads=16, num_kv_groups=4, head_dim=256。
    """
    def __init__(self, dim, n_heads, num_kv_groups, head_dim, mlp_dim, dtype):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = GroupedQueryAttention(dim, n_heads=n_heads, num_kv_groups=num_kv_groups,
                                           head_dim=head_dim, dtype=dtype)
        self.norm2 = RMSNorm(dim)
        self.ff    = GatedFeedForward(dim, mlp_dim, dtype)

    def forward(self, x, cos, sin, mask=None):
        x = self.attn(self.norm1(x), cos, sin, mask) + x
        x = self.ff(self.norm2(x)) + x
        return x

class GatedDeltaNetBlock(nn.Module):
    """
    Linear-attention DeltaNet block with SiLU output gate.
    Occupies the 3 layers preceding each GatedAttentionBlock.
    Qwen3.5-9B: num_key_heads=16, num_value_heads=32,
                key_head_dim=128, value_head_dim=128, conv_kernel_dim=4.
    """
    def __init__(self, dim, num_key_heads, num_value_heads, key_head_dim, value_head_dim,
                 conv_kernel_dim, mlp_dim, dtype):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = GatedDeltaNetAttention(dim, 
                                            num_key_heads=num_key_heads,
                                            num_value_heads=num_value_heads,
                                            key_head_dim=key_head_dim,
                                            value_head_dim=value_head_dim,
                                            conv_kernel_dim=conv_kernel_dim,
                                            dtype=dtype)
        self.norm2 = RMSNorm(dim)
        self.ff    = GatedFeedForward(dim, mlp_dim, dtype)

    def forward(self, x, cos=None, sin=None, mask=None):
        # DeltaNet does not use positional encoding; cos/sin kept for a
        # uniform block interface so the model loop can call all blocks alike.
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x