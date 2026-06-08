import os, json
import time
import torch
from datetime import timedelta
from huggingface_hub import hf_hub_download, snapshot_download
from modules.generate_tokens import generate_tokens
from modules.loading_weights import load_weights_into_qwen3_5
from modules.tokenizer import Qwen3_5Tokenizer
from pathlib import Path
from qwen3_5 import Qwen3_5Model
from safetensors.torch import load_file


def test_qwen3_5_0_8B(prompt, config):
    model = Qwen3_5Model(config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load pretrained weights
    repo_id = "Qwen/Qwen3.5-9B"
    local_dir = Path(repo_id).parts[-1]

    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    weights_dict = {}
    for filename in sorted(set(index["weight_map"].values())):
        shard_path = os.path.join(repo_dir, filename)
        shard = load_file(shard_path)
        weights_dict.update(shard)

    load_weights_into_qwen3_5(model, config, weights_dict)
    model.to(device)
    del weights_dict

    # Load tokenizer
    tokenizer_file_path = "Qwen3.5-9B/tokenizer.json"

    hf_hub_download(
        repo_id=repo_id,
        filename="tokenizer.json",
        local_dir=local_dir,
    )

    tokenizer = Qwen3_5Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        repo_id=repo_id,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )

    # Generate text
    prompt = "What is world order and what is the trend?"
    input_token_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)
    prompt_text = tokenizer.decode(input_token_ids)
    print(f"\nPrompt Text: {prompt_text}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    generated_tokens = 0

    for token in generate_tokens(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=8000,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.2,
        window_size=1024
    ):
        generated_tokens += 1
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True
        )

    elapsed = time.perf_counter() - start_time
    # Convert and print as hour:minute:second
    formatted_time = str(timedelta(seconds=elapsed))
    print(f"\n\n\nTotal time: {formatted_time}")

    tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0
    print(f"Generation speed: {tokens_per_sec:.2f} tokens/sec")

    if torch.cuda.is_available():
        def calc_gpu_gb(x):
            return f"{x / 1024 / 1024 / 1024:.2f} GB"

        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory used: {calc_gpu_gb(torch.cuda.max_memory_allocated())}")
    else:
        print("No GPU available.")


if __name__ == "__main__":
    prompt = "What is world order and what is the trend?"

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

    test_qwen3_5_0_8B(prompt, QWEN3_5_CONFIG)