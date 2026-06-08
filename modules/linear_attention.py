import torch
import torch.nn.functional as F
from torch import nn
from modules.rmsnorm import RMSNormGated


def l2norm(x, dim=-1, eps=1e-6):
    """Unit L2 normalisation without a learnable scale (matches HF convention)."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
):
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out

class GatedDeltaNetAttention(nn.Module):
    """
    Gated DeltaNet linear attention, tailored from HF Qwen3_5GatedDeltaNet.
    Qwen3.5-9B: num_key_heads=16, num_value_heads=32,
                key_head_dim=128, value_head_dim=128, conv_kernel_dim=4.
    """
    def __init__(self, idim, num_key_heads, num_value_heads, key_head_dim, value_head_dim, conv_kernel_dim, dtype):
        super().__init__()
        self.num_k_heads = num_key_heads
        self.num_v_heads = num_value_heads
        self.head_k_dim  = key_head_dim
        self.head_v_dim  = value_head_dim
        self.key_dim     = key_head_dim * num_key_heads
        self.value_dim   = value_head_dim * num_value_heads

        self.in_proj_qkv = nn.Linear(idim, self.key_dim * 2 + self.value_dim, dtype=dtype, bias=False)
        self.in_proj_z   = nn.Linear(idim, self.value_dim, dtype=dtype, bias=False)
        self.in_proj_b   = nn.Linear(idim, num_value_heads, dtype=dtype, bias=False)
        self.in_proj_a   = nn.Linear(idim, num_value_heads, dtype=dtype, bias=False)

        # QKV
        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=conv_kernel_dim,
                                padding=conv_kernel_dim - 1, groups=conv_dim,
                                dtype=dtype, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(num_value_heads, dtype=dtype))

        A = torch.empty(num_value_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # RMSNorm gated by silu(z), normalises per value-head dimension
        self.norm = RMSNormGated(value_head_dim, eps=1e-6)

        self.out_proj = nn.Linear(self.value_dim, idim, dtype=dtype, bias=False)

    def forward(self, hidden_states):
        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out = torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
        )

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output