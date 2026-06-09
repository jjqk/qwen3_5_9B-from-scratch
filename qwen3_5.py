import torch
from torch import nn
from modules.building_blocks import GatedAttentionBlock, GatedDeltaNetBlock
from modules.llm_mem import calc_model_memory_size
from modules.pos_enc import compute_rope_params
from modules.rmsnorm import RMSNorm


class Qwen3_5Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        layer_types = cfg.get("layer_types", ["full_attention"] * cfg["n_layers"])
        if len(layer_types) != cfg["n_layers"]:
            raise ValueError("len(layer_types) must equal n_layers")

        self.trf_blocks = nn.ModuleList()
        for ltype in layer_types:
            if ltype == 'full_attention':
                self.trf_blocks.append(
                    GatedAttentionBlock(cfg["emb_dim"], cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"],
                                        cfg["hidden_dim"], cfg["dtype"]))
            else:  # 'linear_attention'
                self.trf_blocks.append(
                    GatedDeltaNetBlock(cfg["emb_dim"], 
                                       cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], 
                                       cfg["linear_key_head_dim"], cfg["linear_value_head_dim"], 
                                       cfg["linear_conv_kernel_dim"],
                                       cfg["hidden_dim"], cfg["dtype"]))

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        head_dim = cfg["emb_dim"] // cfg["n_heads"] if cfg["head_dim"] is None else cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            partial_rotary_factor=cfg.get("partial_rotary_factor", 1.0),
            dtype=torch.float32,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.dtype = cfg["dtype"]

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)

        num_tokens = x.shape[1]
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.trf_blocks:
            x = block(x, self.cos, self.sin, mask)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.dtype))
        return logits


if __name__ == "__main__":

    # Qwen3.5-9B text configuration
    QWEN3_5_CONFIG = {
        # General
        "vocab_size": 248_320,
        "context_length": 1_010_000,
        "emb_dim": 4_096,
        "rms_norm_eps": 1e-6,
        # Full attention (GroupedQueryAttention)
        "n_heads": 16,
        "n_kv_groups": 4,
        "head_dim": 256,
        "rope_base": 10_000_000.0,
        "partial_rotary_factor": 0.25,
        # Linear attention (GatedDeltaNetAttention)
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        # Common
        "n_layers": 32,
        "hidden_dim": 12_288,
        "layer_types": [
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
        ],
        "dtype": torch.bfloat16,
    }

    model = Qwen3_5Model(QWEN3_5_CONFIG)
    print("\nModel : \n", model)

    # A quick check that the forward pass works before continuing:
    test = model(torch.tensor([1, 2, 3]).unsqueeze(0))
    print("Test model output shape : ", test.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Account for weight tying
    total_params_normalized = total_params - model.tok_emb.weight.numel()
    print(f"\nTotal number of unique parameters: {total_params_normalized:,}")

    print(f"float32 (PyTorch default): {calc_model_memory_size(model, input_dtype=torch.float32):.2f} GB")
    print(f"bfloat16: {calc_model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")
