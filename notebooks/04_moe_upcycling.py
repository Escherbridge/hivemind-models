# %% [markdown]
# # HiveMind Pipeline: MoE Upcycling
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Escherbridge/laxame-hivemind/blob/main/hivemind-models/notebooks/04_moe_upcycling.py)
#
# This notebook converts a dense transformer into a Mixture-of-Experts (MoE) model:
# 1. Load the base model checkpoint from notebook 01
# 2. Configure MoE parameters (num_experts, top_k, target layers)
# 3. Replace selected dense MLP blocks with MoE layers
# 4. Verify the upcycled model with a forward pass
# 5. Compare parameter counts (dense vs MoE, total vs active)
# 6. Analyze memory requirements per expert
# 7. Save the upcycled model checkpoint

# %% [markdown]
# ## 1. Environment Setup

# %%
import sys
import json
import copy
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

COLAB = "google.colab" in str(get_ipython()) if hasattr(__builtins__, "__IPYTHON__") else False

if COLAB:
    get_ipython().system('pip install -q torch transformers safetensors accelerate tabulate tqdm')
    REPO_ROOT = Path("/content/laxame-hivemind")
    DRIVE_ROOT = Path("/content/drive/MyDrive/hivemind-pipeline")
    CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent if "__file__" in dir() else Path.cwd().parent.parent
    CHECKPOINT_DIR = REPO_ROOT / "hivemind-models" / "output" / "checkpoints"

MODELS_SRC = REPO_ROOT / "hivemind-models"
if str(MODELS_SRC) not in sys.path:
    sys.path.insert(0, str(MODELS_SRC))

print(f"Colab: {COLAB}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Load Base Model

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Load from checkpoint (saved by notebook 01) or directly from HF
USE_MODEL = "tinyllama"  # Match notebook 01
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

checkpoint_path = CHECKPOINT_DIR / USE_MODEL
if checkpoint_path.exists():
    print(f"Loading from checkpoint: {checkpoint_path}")
    source = str(checkpoint_path)
else:
    print(f"Checkpoint not found at {checkpoint_path}")
    print(f"Loading directly from HuggingFace: {MODEL_ID}")
    source = MODEL_ID

hf_config = AutoConfig.from_pretrained(source)
tokenizer = AutoTokenizer.from_pretrained(source)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"\nModel config:")
print(f"  Model type: {hf_config.model_type}")
print(f"  Hidden size: {hf_config.hidden_size}")
print(f"  Intermediate size: {hf_config.intermediate_size}")
print(f"  Num layers: {hf_config.num_hidden_layers}")
print(f"  Num attention heads: {hf_config.num_attention_heads}")

# Load model in fp16
model = AutoModelForCausalLM.from_pretrained(
    source,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
print(f"\nDense model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## 3. MoE Configuration

# %%
@dataclass
class MoEConfig:
    """Configuration for MoE upcycling."""
    num_experts: int = 8           # Number of experts (matches 8 tool categories)
    top_k: int = 2                 # Number of experts activated per token
    moe_layers: list[int] = None   # Which layers to convert to MoE (None = auto)
    moe_frequency: int = 2         # Convert every Nth layer (when moe_layers is None)
    router_jitter: float = 0.01    # Jitter noise for load balancing during training
    router_z_loss_coeff: float = 0.001  # Auxiliary loss to prevent router collapse
    capacity_factor: float = 1.25  # Expert capacity factor for token dropping

    def __post_init__(self):
        if self.moe_layers is None:
            # Default: convert every other layer starting from layer 1
            num_layers = hf_config.num_hidden_layers
            self.moe_layers = list(range(self.moe_frequency - 1, num_layers, self.moe_frequency))


moe_config = MoEConfig(
    num_experts=8,
    top_k=2,
    moe_frequency=2,  # Every other layer
)

print("MoE Configuration:")
print(f"  Num experts: {moe_config.num_experts}")
print(f"  Top-K: {moe_config.top_k}")
print(f"  MoE layers: {moe_config.moe_layers}")
print(f"  Total layers: {hf_config.num_hidden_layers}")
print(f"  Dense layers: {hf_config.num_hidden_layers - len(moe_config.moe_layers)}")
print(f"  MoE layers: {len(moe_config.moe_layers)}")

# %% [markdown]
# ## 4. MoE Module Definitions

# %%
class TopKRouter(nn.Module):
    """
    Top-K routing module for MoE.

    Routes each token to the top_k experts with the highest gating scores.
    Includes auxiliary losses for load balancing.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int,
                 jitter: float = 0.01, z_loss_coeff: float = 0.001):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter = jitter
        self.z_loss_coeff = z_loss_coeff

        # Linear projection to get expert scores
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """
        Route tokens to experts.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            router_weights: (batch_size, seq_len, top_k) -- weights for selected experts
            selected_experts: (batch_size, seq_len, top_k) -- indices of selected experts
            router_logits: (batch_size, seq_len, num_experts) -- raw logits for aux loss
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Flatten for routing
        flat_hidden = hidden_states.view(-1, hidden_size)

        # Add jitter during training for load balancing
        if self.training and self.jitter > 0:
            flat_hidden = flat_hidden * (1.0 + torch.randn_like(flat_hidden) * self.jitter)

        # Compute router logits
        router_logits = self.gate(flat_hidden)  # (batch*seq, num_experts)

        # Top-K selection
        router_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )

        # Normalize weights via softmax over selected experts
        router_weights = F.softmax(router_weights, dim=-1)

        # Reshape back
        router_weights = router_weights.view(batch_size, seq_len, self.top_k)
        selected_experts = selected_experts.view(batch_size, seq_len, self.top_k)
        router_logits = router_logits.view(batch_size, seq_len, self.num_experts)

        return router_weights, selected_experts, router_logits


class MoEExpertMLP(nn.Module):
    """
    A single MLP expert, identical in architecture to the original dense MLP.

    For Llama-style models: gate_proj, up_proj, down_proj with SiLU activation.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer replacing a dense MLP.

    Contains N expert MLPs and a router. Each token is routed to top_k experts.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.hidden_size = hidden_size

        # Router
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            jitter=config.router_jitter,
            z_loss_coeff=config.router_z_loss_coeff,
        )

        # Expert MLPs
        self.experts = nn.ModuleList([
            MoEExpertMLP(hidden_size, intermediate_size)
            for _ in range(config.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE layer.

        Uses a loop over experts (simple but not fastest). For production,
        consider using megablocks or scattermoe for parallel dispatch.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Route tokens
        router_weights, selected_experts, router_logits = self.router(hidden_states)

        # Initialize output
        final_output = torch.zeros_like(hidden_states)

        # Flatten for processing
        flat_hidden = hidden_states.view(-1, hidden_size)
        flat_output = final_output.view(-1, hidden_size)
        flat_weights = router_weights.view(-1, self.top_k)
        flat_experts = selected_experts.view(-1, self.top_k)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which tokens are routed to this expert (across all top_k slots)
            expert_mask = (flat_experts == expert_idx)  # (num_tokens, top_k)

            if not expert_mask.any():
                continue

            # Get token indices and their corresponding slot weights
            token_indices, slot_indices = expert_mask.nonzero(as_tuple=True)

            if len(token_indices) == 0:
                continue

            # Gather tokens for this expert
            expert_input = flat_hidden[token_indices]

            # Forward through expert
            expert_output = self.experts[expert_idx](expert_input)

            # Weight by router score
            weights = flat_weights[token_indices, slot_indices].unsqueeze(-1)
            weighted_output = expert_output * weights.to(expert_output.dtype)

            # Scatter-add back to output
            flat_output.index_add_(0, token_indices, weighted_output)

        return flat_output.view(batch_size, seq_len, hidden_size)


print("MoE modules defined: TopKRouter, MoEExpertMLP, MoELayer")

# %% [markdown]
# ## 5. Upcycle: Dense to MoE

# %%
def upcycle_model_to_moe(model, config: MoEConfig, hf_config) -> nn.Module:
    """
    Convert selected dense MLP layers to MoE layers.

    Each expert is initialized as a copy of the original dense MLP weights.
    This ensures the model starts from a good initialization.
    """
    hidden_size = hf_config.hidden_size
    intermediate_size = hf_config.intermediate_size
    layers_converted = []

    for layer_idx in config.moe_layers:
        # Access the transformer layer
        layer = model.model.layers[layer_idx]
        original_mlp = layer.mlp

        # Create MoE layer
        moe_layer = MoELayer(hidden_size, intermediate_size, config)

        # Initialize all experts with the original MLP weights
        for expert in moe_layer.experts:
            expert.gate_proj.weight.data.copy_(original_mlp.gate_proj.weight.data)
            expert.up_proj.weight.data.copy_(original_mlp.up_proj.weight.data)
            expert.down_proj.weight.data.copy_(original_mlp.down_proj.weight.data)

        # Initialize router with small random weights
        nn.init.kaiming_uniform_(moe_layer.router.gate.weight, a=math.sqrt(5))
        moe_layer.router.gate.weight.data *= 0.01  # Small init for balanced routing

        # Replace the MLP with MoE layer
        layer.mlp = moe_layer

        layers_converted.append(layer_idx)

    print(f"Converted {len(layers_converted)} layers to MoE: {layers_converted}")
    return model


print("Running MoE upcycling...")
start_time = time.time()
model = upcycle_model_to_moe(model, moe_config, hf_config)
elapsed = time.time() - start_time
print(f"Upcycling complete in {elapsed:.1f}s")

# %% [markdown]
# ## 6. Verify Forward Pass

# %%
print("Verifying forward pass with MoE model...")

device = next(model.parameters()).device
test_input = tokenizer("The capital of France is", return_tensors="pt").to(device)

with torch.no_grad():
    output = model(**test_input)

print(f"Input shape: {test_input['input_ids'].shape}")
print(f"Output logits shape: {output.logits.shape}")
print(f"Output dtype: {output.logits.dtype}")
print(f"Logits range: [{output.logits.min().item():.4f}, {output.logits.max().item():.4f}]")

# Check for NaN/Inf
has_nan = torch.isnan(output.logits).any().item()
has_inf = torch.isinf(output.logits).any().item()
print(f"Has NaN: {has_nan}")
print(f"Has Inf: {has_inf}")

if not has_nan and not has_inf:
    print("\nForward pass OK.")

    # Try generating a few tokens
    generated = model.generate(
        **test_input,
        max_new_tokens=20,
        do_sample=False,
    )
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated: {decoded}")
else:
    print("\nWARNING: NaN or Inf detected in output. Check initialization.")

# %% [markdown]
# ## 7. Parameter Count Comparison

# %%
from tabulate import tabulate

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
dense_params_original = hf_config.hidden_size * hf_config.intermediate_size * 3  # gate + up + down per layer
dense_total_original = dense_params_original * hf_config.num_hidden_layers

# Count by type
moe_params = 0
dense_mlp_params = 0
attention_params = 0
other_params = 0
router_params = 0

for name, param in model.named_parameters():
    if "experts" in name:
        moe_params += param.numel()
    elif "router" in name:
        router_params += param.numel()
    elif "mlp" in name:
        dense_mlp_params += param.numel()
    elif "self_attn" in name:
        attention_params += param.numel()
    else:
        other_params += param.numel()

# Active parameters per forward pass (only top_k experts per MoE layer)
moe_active_per_layer = dense_params_original * moe_config.top_k  # top_k experts active
moe_total_per_layer = dense_params_original * moe_config.num_experts
dense_remaining_layers = hf_config.num_hidden_layers - len(moe_config.moe_layers)
active_moe_params = moe_active_per_layer * len(moe_config.moe_layers)
active_dense_mlp = dense_params_original * dense_remaining_layers
active_total = active_moe_params + active_dense_mlp + attention_params + other_params + router_params

# Original dense model params
original_total = sum(p.numel() for p in AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, low_cpu_mem_usage=True
).parameters()) if False else dense_total_original + attention_params + other_params  # Estimate to avoid re-loading

comparison_table = [
    ["Total Parameters (MoE)", f"{total_params:,}", f"{total_params * 2 / 1e9:.2f} GB"],
    ["Active Params per Token", f"{active_total:,}", f"{active_total * 2 / 1e9:.2f} GB"],
    ["MoE Expert Params", f"{moe_params:,}", f"{moe_params * 2 / 1e9:.2f} GB"],
    ["Router Params", f"{router_params:,}", f"{router_params * 2 / 1e6:.2f} MB"],
    ["Dense MLP Params", f"{dense_mlp_params:,}", f"{dense_mlp_params * 2 / 1e6:.2f} MB"],
    ["Attention Params", f"{attention_params:,}", f"{attention_params * 2 / 1e9:.2f} GB"],
    ["Other Params", f"{other_params:,}", f"{other_params * 2 / 1e6:.2f} MB"],
]

print("Parameter Count Comparison")
print("=" * 70)
print(tabulate(comparison_table, headers=["Component", "Count", "Size (fp16)"], tablefmt="grid"))

print(f"\nExpansion ratio: {total_params / active_total:.2f}x (total / active)")
print(f"MoE layers use {moe_config.top_k}/{moe_config.num_experts} experts per token")

# %% [markdown]
# ## 8. Per-Layer Analysis

# %%
layer_analysis = []

for layer_idx in range(hf_config.num_hidden_layers):
    layer = model.model.layers[layer_idx]
    is_moe = layer_idx in moe_config.moe_layers

    layer_params = sum(p.numel() for p in layer.parameters())
    mlp_params = sum(p.numel() for p in layer.mlp.parameters())

    if is_moe:
        # Count per-expert params
        expert_params = sum(p.numel() for p in layer.mlp.experts[0].parameters())
        router_p = sum(p.numel() for p in layer.mlp.router.parameters())
        active_params = expert_params * moe_config.top_k + router_p
        layer_type = f"MoE ({moe_config.num_experts} experts)"
    else:
        expert_params = mlp_params
        active_params = layer_params
        layer_type = "Dense"

    layer_analysis.append([
        f"Layer {layer_idx}",
        layer_type,
        f"{layer_params:,}",
        f"{active_params:,}",
        f"{layer_params * 2 / 1e6:.1f} MB",
        f"{active_params * 2 / 1e6:.1f} MB",
    ])

print("Per-Layer Analysis")
print("=" * 80)
print(tabulate(
    layer_analysis,
    headers=["Layer", "Type", "Total Params", "Active Params", "Total Size", "Active Size"],
    tablefmt="grid",
))

# %% [markdown]
# ## 9. Memory Analysis

# %%
print("Memory Analysis: VRAM Requirements Per Expert")
print("=" * 60)

# Sample MoE layer
sample_moe_layer_idx = moe_config.moe_layers[0]
sample_moe = model.model.layers[sample_moe_layer_idx].mlp

expert_param_count = sum(p.numel() for p in sample_moe.experts[0].parameters())
expert_size_fp16 = expert_param_count * 2  # bytes
expert_size_fp32 = expert_param_count * 4  # bytes for optimizer states

memory_table = [
    ["Single Expert (fp16)", f"{expert_param_count:,}", f"{expert_size_fp16 / 1e6:.2f} MB"],
    ["Single Expert (fp32 optim)", f"{expert_param_count:,}", f"{expert_size_fp32 / 1e6:.2f} MB"],
    ["All Experts per MoE Layer", f"{expert_param_count * moe_config.num_experts:,}",
     f"{expert_size_fp16 * moe_config.num_experts / 1e6:.2f} MB"],
    ["Active Experts per Token", f"{expert_param_count * moe_config.top_k:,}",
     f"{expert_size_fp16 * moe_config.top_k / 1e6:.2f} MB"],
    ["Router (per MoE layer)", f"{sum(p.numel() for p in sample_moe.router.parameters()):,}",
     f"{sum(p.numel() for p in sample_moe.router.parameters()) * 2 / 1e3:.2f} KB"],
]

print(tabulate(memory_table, headers=["Component", "Params", "Size"], tablefmt="grid"))

# Total VRAM estimate
total_model_size = total_params * 2  # fp16
kv_cache_per_token = (
    hf_config.hidden_size // hf_config.num_attention_heads *
    (hf_config.num_key_value_heads or hf_config.num_attention_heads) *
    2 *  # K and V
    2 *  # fp16 bytes
    hf_config.num_hidden_layers
)

print(f"\nTotal Model VRAM (fp16): {total_model_size / 1e9:.2f} GB")
print(f"KV Cache per token: {kv_cache_per_token / 1e3:.2f} KB")
print(f"KV Cache for 2048 tokens: {kv_cache_per_token * 2048 / 1e6:.2f} MB")

# For HiveMind distributed inference: each node only needs its assigned experts
experts_per_node = moe_config.num_experts  # In P2P, experts can be split across nodes
print(f"\nFor P2P distribution:")
for n_nodes in [1, 2, 4, 8]:
    experts_per = moe_config.num_experts // n_nodes
    expert_vram = expert_size_fp16 * experts_per * len(moe_config.moe_layers)
    print(f"  {n_nodes} nodes: {experts_per} experts/node, ~{expert_vram / 1e6:.1f} MB expert VRAM each")

# %% [markdown]
# ## 10. Save Upcycled Model

# %%
# Save the MoE model checkpoint
moe_checkpoint_path = CHECKPOINT_DIR / f"{USE_MODEL}_moe"
moe_checkpoint_path.mkdir(parents=True, exist_ok=True)

print(f"Saving MoE model to: {moe_checkpoint_path}")
start_time = time.time()

# Save model state dict (custom save since architecture changed)
torch.save({
    "model_state_dict": model.state_dict(),
    "moe_config": {
        "num_experts": moe_config.num_experts,
        "top_k": moe_config.top_k,
        "moe_layers": moe_config.moe_layers,
        "moe_frequency": moe_config.moe_frequency,
        "router_jitter": moe_config.router_jitter,
        "router_z_loss_coeff": moe_config.router_z_loss_coeff,
        "capacity_factor": moe_config.capacity_factor,
    },
    "base_model_id": MODEL_ID,
    "hf_config": hf_config.to_dict(),
}, str(moe_checkpoint_path / "moe_model.pt"))

# Save tokenizer alongside
tokenizer.save_pretrained(str(moe_checkpoint_path))

# Save the MoE config as JSON for easier loading
moe_meta = {
    "base_model_id": MODEL_ID,
    "model_type": hf_config.model_type,
    "num_experts": moe_config.num_experts,
    "top_k": moe_config.top_k,
    "moe_layers": moe_config.moe_layers,
    "total_params": total_params,
    "active_params_per_token": active_total,
    "hidden_size": hf_config.hidden_size,
    "intermediate_size": hf_config.intermediate_size,
    "num_hidden_layers": hf_config.num_hidden_layers,
}
with open(moe_checkpoint_path / "moe_config.json", "w") as f:
    json.dump(moe_meta, f, indent=2)

elapsed = time.time() - start_time
print(f"Saved in {elapsed:.1f}s")

# List files
for f in sorted(moe_checkpoint_path.iterdir()):
    size = f.stat().st_size if f.is_file() else 0
    print(f"  {f.name:40s} {size / 1e6:>8.2f} MB")

# %% [markdown]
# ## Summary
#
# This notebook has:
# 1. Loaded the dense base model (**TinyLlama 1.1B**)
# 2. Configured MoE: **{num_experts}** experts, top-**{top_k}** routing
# 3. Converted **{len(moe_layers)}** dense MLP layers to MoE layers
# 4. Verified the forward pass produces valid outputs
# 5. Compared parameters: total vs active per token
# 6. Analyzed memory requirements for distributed inference
# 7. Saved the MoE checkpoint
#
# **Next step**: Run `05_expert_finetuning.py` to train each expert on its category data.

# %%
print("\n" + "=" * 60)
print("NOTEBOOK 04 COMPLETE: MoE Upcycling")
print("=" * 60)
print(f"\nBase model: {MODEL_ID}")
print(f"MoE config: {moe_config.num_experts} experts, top-{moe_config.top_k}")
print(f"MoE layers: {moe_config.moe_layers}")
print(f"Total params: {total_params:,}")
print(f"Active params per token: {active_total:,}")
print(f"Checkpoint: {moe_checkpoint_path}")
print(f"\nReady for notebook 05: Expert Fine-Tuning")
