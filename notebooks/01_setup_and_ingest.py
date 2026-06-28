# NOTE (2026-06-27): Notebooks 02–05 (tool-category definition, dataset generation,
# MoE upcycling, expert fine-tuning) have been archived to
# hivemind-archive/domain-moe-sdk/. The domain-MoE premise — one expert = one tool
# category — is empirically broken (learned routers route by token syntax, not domain).
# Domain specialisation is now handled by Branch-Train-Merge (Axis 3).
# See hivemind-archive/README.md for full rationale.

# %% [markdown]
# # HiveMind Pipeline: Setup & Model Ingestion
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Escherbridge/laxame-hivemind/blob/main/hivemind-models/notebooks/01_setup_and_ingest.py)
#
# This notebook handles the first stage of the HiveMind BitNet MoE pipeline:
# 1. Install all required dependencies
# 2. Clone / mount the hivemind-models repository
# 3. Load a model from HuggingFace (TinyLlama 1.1B for testing, BitNet 3B for production)
# 4. Inspect the model architecture and print a summary table
# 5. Save the loaded model to a checkpoint directory for downstream notebooks

# %% [markdown]
# ## 1. Environment Detection & Setup

# %%
import sys

# Detect if running in Google Colab
COLAB = "google.colab" in str(get_ipython()) if hasattr(__builtins__, "__IPYTHON__") else False
print(f"Running in Colab: {COLAB}")
print(f"Python version: {sys.version}")

# %%
# Install dependencies
if COLAB:
    print("Installing dependencies for Colab...")
    get_ipython().system('pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
    get_ipython().system('pip install -q transformers>=4.40.0 safetensors>=0.4.0 accelerate>=0.27.0 peft>=0.10.0')
    get_ipython().system('pip install -q anthropic pyyaml tqdm tabulate')
    print("Dependencies installed.")
else:
    print("Not in Colab. Ensure dependencies are installed:")
    print("  pip install torch transformers safetensors accelerate peft anthropic pyyaml tqdm tabulate")

# %%
import torch
import os
from pathlib import Path

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Repository Setup

# %%
# Configure paths
if COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    REPO_ROOT = Path("/content/laxame-hivemind")
    DRIVE_ROOT = Path("/content/drive/MyDrive/hivemind-pipeline")
    CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent if "__file__" in dir() else Path.cwd().parent.parent
    CHECKPOINT_DIR = REPO_ROOT / "hivemind-models" / "output" / "checkpoints"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Repo root: {REPO_ROOT}")
print(f"Checkpoint dir: {CHECKPOINT_DIR}")

# %%
# Clone the repo if in Colab and not already present
if COLAB and not REPO_ROOT.exists():
    print("Cloning hivemind-models repository...")
    get_ipython().system('git clone https://github.com/Escherbridge/laxame-hivemind.git /content/laxame-hivemind')
    print("Repository cloned.")
elif COLAB:
    print("Repository already exists, pulling latest...")
    get_ipython().system('cd /content/laxame-hivemind && git pull')

# Add src to path
MODELS_SRC = REPO_ROOT / "hivemind-models"
if str(MODELS_SRC) not in sys.path:
    sys.path.insert(0, str(MODELS_SRC))
    print(f"Added {MODELS_SRC} to sys.path")

# %% [markdown]
# ## 3. Model Configuration

# %%
# Model selection -- use TinyLlama for testing on free Colab (fits in ~2GB RAM)
# Switch to BitNet 3B for production runs with more VRAM

MODEL_CONFIGS = {
    "tinyllama": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "num_layers": 22,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "description": "1.1B params, good for testing on free Colab T4",
    },
    "bitnet_3b": {
        "model_id": "1bitLLM/bitnet_b1_58-3B",
        "num_layers": 26,
        "hidden_size": 3200,
        "intermediate_size": 8640,
        "description": "3B params, ternary weights, needs more VRAM",
    },
}

# Choose which model to use
USE_MODEL = "tinyllama"  # Change to "bitnet_3b" for production
model_cfg = MODEL_CONFIGS[USE_MODEL]

print(f"Selected model: {model_cfg['model_id']}")
print(f"Description: {model_cfg['description']}")
print(f"Expected layers: {model_cfg['num_layers']}")

# %% [markdown]
# ## 4. Load Model from HuggingFace

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time

model_id = model_cfg["model_id"]
print(f"Loading model: {model_id}")
print("This may take a few minutes on first run (downloading weights)...\n")

# Load configuration first
hf_config = AutoConfig.from_pretrained(model_id)
print(f"Model type: {hf_config.model_type}")
print(f"Hidden size: {hf_config.hidden_size}")
print(f"Num layers: {hf_config.num_hidden_layers}")
print(f"Num attention heads: {hf_config.num_attention_heads}")
if hasattr(hf_config, "num_key_value_heads"):
    print(f"Num KV heads: {hf_config.num_key_value_heads}")
print(f"Intermediate size: {hf_config.intermediate_size}")
print(f"Vocab size: {hf_config.vocab_size}")

# %%
# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"Tokenizer type: {type(tokenizer).__name__}")
print(f"Vocab size: {tokenizer.vocab_size}")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: '{tokenizer.eos_token}'")

# %%
# Load the model
print("\nLoading model weights...")
start_time = time.time()

# Use float16 to reduce memory; use device_map="auto" for multi-GPU
load_kwargs = {
    "torch_dtype": torch.float16,
    "low_cpu_mem_usage": True,
}

if torch.cuda.is_available():
    load_kwargs["device_map"] = "auto"
    print("Using device_map='auto' for GPU placement")
else:
    print("No GPU detected, loading on CPU (will be slow for inference)")

model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
elapsed = time.time() - start_time
print(f"\nModel loaded in {elapsed:.1f}s")

# %% [markdown]
# ## 5. Inspect Model Architecture

# %%
# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Estimated fp16 size: {total_params * 2 / 1e9:.2f} GB")

# %%
# Detailed layer inspection
from tabulate import tabulate

state_dict = model.state_dict()
print(f"\nTotal tensors in state_dict: {len(state_dict)}")

# Group tensors by component
component_stats = {}
for name, tensor in state_dict.items():
    # Determine component from name
    if "embed_tokens" in name:
        component = "embedding"
    elif "lm_head" in name:
        component = "lm_head"
    elif "model.norm" in name and "layers" not in name:
        component = "final_norm"
    elif "self_attn" in name:
        component = "attention"
    elif "mlp" in name:
        component = "mlp"
    elif "layernorm" in name or "input_layernorm" in name or "post_attention_layernorm" in name:
        component = "layer_norm"
    else:
        component = "other"

    if component not in component_stats:
        component_stats[component] = {"count": 0, "params": 0, "size_mb": 0}
    component_stats[component]["count"] += 1
    component_stats[component]["params"] += tensor.numel()
    component_stats[component]["size_mb"] += tensor.numel() * tensor.element_size() / (1024 ** 2)

table_data = []
for comp, stats in sorted(component_stats.items()):
    table_data.append([
        comp,
        stats["count"],
        f"{stats['params']:,}",
        f"{stats['size_mb']:.2f} MB",
        f"{stats['params'] / total_params * 100:.1f}%"
    ])

print("\n" + tabulate(
    table_data,
    headers=["Component", "Tensors", "Parameters", "Size (fp16)", "% of Total"],
    tablefmt="grid"
))

# %%
# Per-layer breakdown
print("\nPer-Layer Analysis:")
print("=" * 80)

layer_pattern = "model.layers.{idx}"
num_layers = hf_config.num_hidden_layers

layer_table = []
for layer_idx in range(num_layers):
    prefix = f"model.layers.{layer_idx}."
    layer_tensors = {k: v for k, v in state_dict.items() if k.startswith(prefix)}

    attn_params = sum(
        t.numel() for k, t in layer_tensors.items() if "self_attn" in k
    )
    mlp_params = sum(
        t.numel() for k, t in layer_tensors.items() if "mlp" in k
    )
    norm_params = sum(
        t.numel() for k, t in layer_tensors.items()
        if "layernorm" in k or "input_layernorm" in k or "post_attention_layernorm" in k
    )
    total_layer = sum(t.numel() for t in layer_tensors.values())

    layer_table.append([
        f"Layer {layer_idx}",
        len(layer_tensors),
        f"{attn_params:,}",
        f"{mlp_params:,}",
        f"{norm_params:,}",
        f"{total_layer:,}",
        f"{total_layer * 2 / 1e6:.1f} MB",
    ])

print(tabulate(
    layer_table,
    headers=["Layer", "Tensors", "Attn Params", "MLP Params", "Norm Params", "Total Params", "Size (fp16)"],
    tablefmt="grid"
))

# %%
# Identify MLP blocks for MoE conversion
print("\nMLP Block Structure (candidates for MoE conversion):")
print("=" * 60)

sample_layer = 0
prefix = f"model.layers.{sample_layer}.mlp."
mlp_tensors = {k: v for k, v in state_dict.items() if k.startswith(prefix)}

mlp_table = []
for name, tensor in sorted(mlp_tensors.items()):
    short_name = name.replace(f"model.layers.{sample_layer}.", "")
    mlp_table.append([
        short_name,
        list(tensor.shape),
        str(tensor.dtype),
        f"{tensor.numel():,}",
        f"{tensor.numel() * tensor.element_size() / 1e6:.2f} MB",
    ])

print(tabulate(
    mlp_table,
    headers=["Tensor", "Shape", "Dtype", "Elements", "Size"],
    tablefmt="grid"
))
print(f"\nMLP structure identical across all {num_layers} layers.")
print(f"Each MLP block: {sum(t.numel() for t in mlp_tensors.values()):,} params")
print(f"Total MLP params in model: {sum(t.numel() for t in mlp_tensors.values()) * num_layers:,}")

# %%
# Quick sanity check: forward pass
print("\nSanity check: running a forward pass...")
test_input = tokenizer("The capital of France is", return_tensors="pt")
if torch.cuda.is_available():
    test_input = {k: v.to(model.device) for k, v in test_input.items()}

with torch.no_grad():
    output = model(**test_input)

print(f"Input shape: {test_input['input_ids'].shape}")
print(f"Output logits shape: {output.logits.shape}")
print(f"Output dtype: {output.logits.dtype}")

# Greedy decode a few tokens
generated = model.generate(
    **test_input,
    max_new_tokens=20,
    do_sample=False,
)
decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"\nGenerated: {decoded}")
print("Forward pass OK.")

# %% [markdown]
# ## 6. Save Checkpoint

# %%
# Save model checkpoint for downstream notebooks
checkpoint_path = CHECKPOINT_DIR / USE_MODEL
checkpoint_path.mkdir(parents=True, exist_ok=True)

print(f"Saving checkpoint to: {checkpoint_path}")
start_time = time.time()

# Save model
model.save_pretrained(str(checkpoint_path), safe_serialization=True)
# Save tokenizer
tokenizer.save_pretrained(str(checkpoint_path))
# Save config
hf_config.save_pretrained(str(checkpoint_path))

elapsed = time.time() - start_time
print(f"Checkpoint saved in {elapsed:.1f}s")

# List saved files
saved_files = list(checkpoint_path.iterdir())
print(f"\nSaved files ({len(saved_files)}):")
total_size = 0
for f in sorted(saved_files):
    size = f.stat().st_size if f.is_file() else 0
    total_size += size
    print(f"  {f.name:40s} {size / 1e6:>8.2f} MB")

print(f"\nTotal checkpoint size: {total_size / 1e9:.2f} GB")

# %% [markdown]
# ## Summary
#
# This notebook has:
# 1. Installed all dependencies for the HiveMind pipeline
# 2. Loaded the **{model_id}** model from HuggingFace
# 3. Inspected the architecture: **{num_layers}** transformer layers, **{total_params:,}** parameters
# 4. Identified MLP blocks for MoE conversion
# 5. Saved the checkpoint to `{checkpoint_path}`
#
# **Next step**: Run `02_define_tool_categories.py` to define the 8 tool categories.

# %%
print("\n" + "=" * 60)
print("NOTEBOOK 01 COMPLETE: Setup & Model Ingestion")
print("=" * 60)
print(f"\nModel: {model_id}")
print(f"Parameters: {total_params:,}")
print(f"Layers: {num_layers}")
print(f"Checkpoint: {checkpoint_path}")
print(f"\nReady for notebook 02: Define Tool Categories")
