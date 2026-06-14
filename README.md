# Build Qwen3.5-9B From Scratch

PyTorch implementation of the LLM Qwen3.5-9B aligned to the official [HuggingFace checkpoint](https://huggingface.co/Qwen/Qwen3.5-9B).

## Architecture

                                  [Inputs]
                                      │
                                      ▼
                             ┌──────────────────┐
                             │  Embedding Layer │
                             └──────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ 8× Repeating Macro-Blocks (32 Layers Total)                 │
        │                                                             │
        │  ┌───────────────────────────────────────────────────────┐  │
        │  │ 3× Linear Attention Layers                            │  │
        │  │    [Gated DeltaNet Block] ──► [SiLU-Gated FFN Block]  │  │
        │  └───────────────────────────────────────────────────────┘  │
        │                             │                               │
        │                             ▼                               │
        │  ┌───────────────────────────────────────────────────────┐  │
        │  │ 1× Full Attention Layer                               │  │
        │  │    [Gated Attention (GQA)] ──► [SiLU-Gated FFN Block] │  │
        │  └───────────────────────────────────────────────────────┘  │
        └─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                             ┌──────────────────┐
                             │     RMSNorm      │
                             └──────────────────┘
                                      │
                                      ▼
                            [Next tokens / Outputs]

## Language Model Configuration/Parameters
```
Number of Parameters: 9B
Embedding Dimension: 4096
Token Embedding: 248320
Context Length: 1010000
Number of Layers: 32
Hidden Layout: 8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
Gated DeltaNet:
  Number of Linear Attention Heads: 32 for V and 16 for QK
  Head Dimension: 128
Gated Attention:
  Number of Attention Heads: 16 for Q and 4 for KV
  Head Dimension: 256
  Rotary Position Embedding Dimension: 64
Feed Forward Network:
  Intermediate Dimension: 12288
```

## Project Structure
```
qwen3_5.py              # Model definition (Qwen3_5Model)
test.py                 # Download weights + run inference
modules/
  full_attention.py     # Full attention, GroupedQueryAttention
  linear_attention.py   # Linear attention, GatedDeltaNetAttention
  building_blocks.py    # Attention blocks used to build the model
  generate_tokens.py    # Generate next tokens
  llm_mem.py            # Calculate model memory size
  loading_weights.py    # Load model weights from HuggingFace
  pos_enc.py            # Rotary position encoding
  rmsnorm.py            # RMSNorm and RMSNormGated
  tokenizer.py          # Qwen3.5 tokenizer wrapper
```

## Usage
```
- method 1: play for fun by running test.py in command line to perform inference
  python3 test.py

- method 2: configure and define the model in python file
  import torch
  from qwen3_5 import Qwen3_5Model

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
```

## Key Implementation Details
```
- The core hybrid architecture
  Instead of utilizing standard quadratic attention for every layer, Qwen3.5 uses a 3:1 hybrid ratio:
  - 75% of the layers utilize Gated DeltaNet (Linear Attention) to achieve highly compressed, O(1) per-token memory scaling.
  - 25% of the layers interleave standard Gated Attention (Full Quadratic Attention) to preserve precise, global token-level retrieval for the most complex reasoning steps.
- GatedDeltaNetAttention
  Linear attention tailored from [HuggingFace Qwen3_5GatedDeltaNet](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py), function torch_chunk_gated_delta_rule() is used to leverage hardware parallelism.
- GroupedQueryAttention
  The full attention layer employs Grouped Query Attention (GQA).
- Weight loading
  The weights of the model Qwen3_5Model is loaded from HuggingFace, refer to `modules/loading_weights.py`.
```

## Note
The code is expected to work for all models Qwen3.5 0.8B - 9B by corresponding configurations, but only "Qwen3.5 0.8B" and "Qwen3.5 9B" are checked, and other models are to be checked.

## References
- [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Qwen3.5 0.8B From Scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/16_qwen3.5)
- [Qwen3.5 Playground](https://github.com/rishikksh20/qwen3-5-playground)

## License

See [LICENSE](LICENSE).
