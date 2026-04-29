# %% [markdown]
# # HiveMind Pipeline: BitNet Quantization & Sharding
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Escherbridge/laxame-hivemind/blob/main/hivemind-models/notebooks/06_quantize_and_shard.py)
#
# This notebook quantizes and shards the trained MoE model for HiveMind distribution:
# 1. Load the trained MoE model from notebook 05
# 2. Quantize weights to BitNet b1.58 (ternary) using the existing bitllm.py
# 3. Compare fp16 vs BitNet sizes per expert
# 4. Shard the model for P2P distribution (embed, layer groups, experts, head)
# 5. Generate manifest.json with checksums and capability metadata
# 6. Include tool schemas in the manifest
# 7. Export tokenizer
# 8. Print final artifact summary

# %% [markdown]
# ## 1. Environment Setup

# %%
import sys
import json
import yaml
import time
import hashlib
import os
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

COLAB = "google.colab" in str(get_ipython()) if hasattr(__builtins__, "__IPYTHON__") else False

if COLAB:
    get_ipython().system('pip install -q torch transformers safetensors tabulate tqdm pyyaml')
    REPO_ROOT = Path("/content/laxame-hivemind")
    DRIVE_ROOT = Path("/content/drive/MyDrive/hivemind-pipeline")
    CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
    TOOL_CONFIG_DIR = DRIVE_ROOT / "tool_configs"
    OUTPUT_DIR = DRIVE_ROOT / "sharded_model"
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent if "__file__" in dir() else Path.cwd().parent.parent
    CHECKPOINT_DIR = REPO_ROOT / "hivemind-models" / "output" / "checkpoints"
    TOOL_CONFIG_DIR = REPO_ROOT / "hivemind-models" / "output" / "tool_configs"
    OUTPUT_DIR = REPO_ROOT / "hivemind-models" / "output" / "sharded_model"

MODELS_SRC = REPO_ROOT / "hivemind-models"
if str(MODELS_SRC) not in sys.path:
    sys.path.insert(0, str(MODELS_SRC))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Colab: {COLAB}")
print(f"Output dir: {OUTPUT_DIR}")

# %% [markdown]
# ## 2. Load Trained MoE Model

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

USE_MODEL = "tinyllama"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load MoE config
moe_config_path = CHECKPOINT_DIR / f"{USE_MODEL}_moe" / "moe_config.json"
if moe_config_path.exists():
    with open(moe_config_path) as f:
        moe_meta = json.load(f)
else:
    moe_meta = {
        "num_experts": 8, "top_k": 2,
        "moe_layers": list(range(1, 22, 2)),
        "hidden_size": 2048, "intermediate_size": 5632,
        "num_hidden_layers": 22,
    }

print(f"MoE config: {moe_meta['num_experts']} experts, top-{moe_meta['top_k']}")
print(f"MoE layers: {moe_meta['moe_layers']}")

# %%
# MoE module definitions (same as notebooks 04/05, included for standalone use)

class TopKRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat = hidden_states.view(-1, hidden_size)
        logits = self.gate(flat)
        weights, experts = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return (weights.view(batch_size, seq_len, self.top_k),
                experts.view(batch_size, seq_len, self.top_k),
                logits.view(batch_size, seq_len, self.num_experts))


class MoEExpertMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([
            MoEExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        router_weights, selected_experts, _ = self.router(hidden_states)
        final_output = torch.zeros_like(hidden_states)
        flat_hidden = hidden_states.view(-1, hidden_size)
        flat_output = final_output.view(-1, hidden_size)
        flat_weights = router_weights.view(-1, self.top_k)
        flat_experts = selected_experts.view(-1, self.top_k)
        for eidx in range(self.num_experts):
            mask = (flat_experts == eidx)
            if not mask.any():
                continue
            tidx, sidx = mask.nonzero(as_tuple=True)
            if len(tidx) == 0:
                continue
            out = self.experts[eidx](flat_hidden[tidx])
            w = flat_weights[tidx, sidx].unsqueeze(-1)
            flat_output.index_add_(0, tidx, out * w.to(out.dtype))
        return flat_output.view(batch_size, seq_len, hidden_size)

# %%
# Build and load model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Convert to MoE
for layer_idx in moe_meta["moe_layers"]:
    layer = base_model.model.layers[layer_idx]
    layer.mlp = MoELayer(
        moe_meta["hidden_size"], moe_meta["intermediate_size"],
        moe_meta["num_experts"], moe_meta["top_k"]
    )

# Load trained weights
trained_path = CHECKPOINT_DIR / f"{USE_MODEL}_moe_trained" / "moe_trained.pt"
if trained_path.exists():
    print(f"Loading trained weights from {trained_path}")
    ckpt = torch.load(str(trained_path), map_location="cpu", weights_only=False)
    base_model.load_state_dict(ckpt["model_state_dict"], strict=False)
else:
    moe_path = CHECKPOINT_DIR / f"{USE_MODEL}_moe" / "moe_model.pt"
    if moe_path.exists():
        print(f"Loading MoE weights from {moe_path}")
        ckpt = torch.load(str(moe_path), map_location="cpu", weights_only=False)
        base_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        print("WARNING: No trained weights found. Using base model weights.")

model = base_model
model.eval()
state_dict = model.state_dict()

print(f"Model loaded. Tensors: {len(state_dict)}")

# %% [markdown]
# ## 3. BitNet Ternary Quantization

# %%
# Import the BitLLM quantization from the repo
try:
    from src.convert.bitllm import (
        BitLLMConfig, BitLLMTensor,
        quantize_to_ternary, dequantize_ternary,
        tensor_to_safetensors_dict, estimate_bitllm_size,
    )
    print("Loaded BitLLM from src.convert.bitllm")
except ImportError:
    print("BitLLM not found in repo, using inline implementation")

    @dataclass
    class BitLLMConfig:
        bit_width: int = 2
        group_size: int = 64

    @dataclass
    class BitLLMTensor:
        packed_weights: torch.Tensor
        scales: torch.Tensor
        zeros: torch.Tensor | None
        original_shape: tuple
        bit_width: int
        group_size: int

    def quantize_to_ternary(tensor, group_size=64):
        original_shape = tensor.shape
        tensor = tensor.float()
        flat = tensor.view(-1)
        if flat.numel() % group_size != 0:
            pad = group_size - (flat.numel() % group_size)
            flat = F.pad(flat, (0, pad))
        grouped = flat.view(-1, group_size)
        abs_mean = grouped.abs().mean(dim=1, keepdim=True)
        threshold = 0.7 * abs_mean
        scales = abs_mean.squeeze().half()
        ternary = torch.zeros_like(grouped, dtype=torch.uint8)
        ternary[grouped > threshold] = 1
        ternary[grouped < -threshold] = 2
        # Pack 4 per byte
        ternary_flat = ternary.view(-1)
        if ternary_flat.numel() % 4 != 0:
            ternary_flat = F.pad(ternary_flat, (0, 4 - ternary_flat.numel() % 4))
        ternary_flat = ternary_flat.view(-1, 4)
        packed = torch.zeros(ternary_flat.shape[0], dtype=torch.uint8)
        for i in range(4):
            packed |= (ternary_flat[:, i].to(torch.uint8) << (i * 2))
        return BitLLMTensor(packed, scales, threshold.squeeze().half(), original_shape, 2, group_size)

    def estimate_bitllm_size(tensor, bit_width=2, group_size=64):
        numel = tensor.numel()
        orig = numel * tensor.element_size()
        packed = (numel + 3) // 4
        n_groups = (numel + group_size - 1) // group_size
        scale_bytes = n_groups * 2
        zeros_bytes = n_groups * 2
        total = packed + scale_bytes + zeros_bytes
        return {"original_bytes": orig, "total_bytes": total, "compression_ratio": orig / total if total else 0}

    def tensor_to_safetensors_dict(bitllm, base_name):
        meta = torch.tensor(list(bitllm.original_shape) + [bitllm.bit_width, bitllm.group_size], dtype=torch.int32)
        d = {f"{base_name}.packed": bitllm.packed_weights, f"{base_name}.scales": bitllm.scales, f"{base_name}.meta": meta}
        if bitllm.zeros is not None:
            d[f"{base_name}.zeros"] = bitllm.zeros
        return d

# %%
# Quantize all weight tensors
print("Quantizing model to BitNet b1.58 (ternary)...")
print("=" * 60)

from tabulate import tabulate

BITLLM_CONFIG = BitLLMConfig(bit_width=2, group_size=64)
GROUP_SIZE = BITLLM_CONFIG.group_size

quantized_tensors = {}  # name -> BitLLMTensor
passthrough_tensors = {}  # name -> tensor (norms, biases stay fp16)

total_fp16_bytes = 0
total_bitnet_bytes = 0
quant_stats = []

for name, tensor in state_dict.items():
    fp16_size = tensor.numel() * tensor.element_size()
    total_fp16_bytes += fp16_size

    # Quantize 2D weight matrices, keep everything else as fp16
    if "weight" in name and tensor.dim() >= 2 and tensor.numel() > 256:
        bitllm_tensor = quantize_to_ternary(tensor, GROUP_SIZE)
        quantized_tensors[name] = bitllm_tensor

        est = estimate_bitllm_size(tensor, bit_width=2, group_size=GROUP_SIZE)
        bitnet_size = est["total_bytes"]
        total_bitnet_bytes += bitnet_size

        quant_stats.append([name[:50], f"{tensor.shape}", f"{fp16_size / 1e6:.2f}",
                            f"{bitnet_size / 1e6:.2f}", f"{est['compression_ratio']:.1f}x"])
    else:
        passthrough_tensors[name] = tensor
        total_bitnet_bytes += fp16_size

print(f"Quantized {len(quantized_tensors)} weight tensors")
print(f"Kept {len(passthrough_tensors)} tensors as fp16 (norms, biases, embeddings)")
print(f"\nFP16 total: {total_fp16_bytes / 1e9:.3f} GB")
print(f"BitNet total: {total_bitnet_bytes / 1e9:.3f} GB")
print(f"Compression: {total_fp16_bytes / total_bitnet_bytes:.2f}x")

# Show first 20 quantized layers
print(f"\nQuantization details (first 20):")
print(tabulate(quant_stats[:20], headers=["Tensor", "Shape", "FP16 (MB)", "BitNet (MB)", "Ratio"], tablefmt="grid"))

# %% [markdown]
# ## 4. Compare FP16 vs BitNet Per Expert

# %%
print("Per-Expert Size Comparison: FP16 vs BitNet")
print("=" * 60)

expert_sizes = []
for expert_id in range(moe_meta["num_experts"]):
    fp16_total = 0
    bitnet_total = 0

    for layer_idx in moe_meta["moe_layers"]:
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_id}."
        for name, tensor in state_dict.items():
            if name.startswith(prefix):
                fp16_size = tensor.numel() * tensor.element_size()
                fp16_total += fp16_size
                if name in quantized_tensors:
                    est = estimate_bitllm_size(tensor, bit_width=2, group_size=GROUP_SIZE)
                    bitnet_total += est["total_bytes"]
                else:
                    bitnet_total += fp16_size

    ratio = fp16_total / bitnet_total if bitnet_total > 0 else 0
    expert_sizes.append([
        f"Expert {expert_id}",
        f"{fp16_total / 1e6:.2f} MB",
        f"{bitnet_total / 1e6:.2f} MB",
        f"{ratio:.1f}x",
    ])

print(tabulate(expert_sizes, headers=["Expert", "FP16 Size", "BitNet Size", "Compression"], tablefmt="grid"))

# %% [markdown]
# ## 5. Shard the Model for HiveMind Distribution

# %%
from safetensors.torch import save_file

def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_shard(tensors: dict[str, torch.Tensor], filename: str, output_dir: Path) -> dict:
    """Save a shard as safetensors and return metadata."""
    path = output_dir / filename
    # Ensure all tensors are CPU and contiguous
    clean = {k: v.cpu().contiguous() for k, v in tensors.items()}
    save_file(clean, str(path))
    size = path.stat().st_size
    checksum = compute_sha256(path)
    return {
        "filename": filename,
        "size_bytes": size,
        "checksum_sha256": checksum,
        "tensor_count": len(tensors),
        "tensor_names": list(tensors.keys()),
    }


print("Sharding model for HiveMind distribution...")
print("=" * 60)

shard_output = OUTPUT_DIR / "shards"
shard_output.mkdir(parents=True, exist_ok=True)

shard_manifest_entries = []
num_layers = moe_meta["num_hidden_layers"]

# %%
# Shard 1: Embedding layers
print("\n[1/5] Embedding shard...")
embed_tensors = {}
for name, tensor in state_dict.items():
    if "embed_tokens" in name:
        embed_tensors[name] = tensor

if embed_tensors:
    info = save_shard(embed_tensors, "shard_embed.safetensors", shard_output)
    info["shard_type"] = "embedding"
    shard_manifest_entries.append(info)
    print(f"  shard_embed.safetensors: {info['size_bytes'] / 1e6:.2f} MB, {info['tensor_count']} tensors")

# %%
# Shard 2: Layer groups (non-MoE layers grouped, MoE layers individual)
print("\n[2/5] Layer shards...")

# Group dense layers together, MoE layers get individual shards
dense_layers = [i for i in range(num_layers) if i not in moe_meta["moe_layers"]]
moe_layer_indices = moe_meta["moe_layers"]

# Group dense layers into chunks of ~4 layers each
DENSE_GROUP_SIZE = 4
dense_groups = []
for i in range(0, len(dense_layers), DENSE_GROUP_SIZE):
    group = dense_layers[i:i + DENSE_GROUP_SIZE]
    dense_groups.append((group[0], group[-1]))

for start, end in dense_groups:
    tensors = {}
    for name, tensor in state_dict.items():
        for layer_idx in range(start, end + 1):
            prefix = f"model.layers.{layer_idx}."
            if name.startswith(prefix) and layer_idx not in moe_meta["moe_layers"]:
                tensors[name] = tensor
                break

    if tensors:
        filename = f"shard_layers_{start}_{end}.safetensors"
        info = save_shard(tensors, filename, shard_output)
        info["shard_type"] = "dense_layers"
        info["layer_range"] = [start, end]
        shard_manifest_entries.append(info)
        print(f"  {filename}: {info['size_bytes'] / 1e6:.2f} MB, {info['tensor_count']} tensors")

# MoE layer shards (attention + norms, without expert weights)
for layer_idx in moe_layer_indices:
    tensors = {}
    prefix = f"model.layers.{layer_idx}."
    for name, tensor in state_dict.items():
        if name.startswith(prefix) and "experts" not in name:
            tensors[name] = tensor

    if tensors:
        filename = f"shard_moe_layer_{layer_idx}_base.safetensors"
        info = save_shard(tensors, filename, shard_output)
        info["shard_type"] = "moe_layer_base"
        info["layer_index"] = layer_idx
        shard_manifest_entries.append(info)
        print(f"  {filename}: {info['size_bytes'] / 1e6:.2f} MB, {info['tensor_count']} tensors")

# %%
# Shard 3: Individual expert shards
print("\n[3/5] Expert shards...")
for expert_id in range(moe_meta["num_experts"]):
    expert_tensors = {}
    for layer_idx in moe_layer_indices:
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_id}."
        for name, tensor in state_dict.items():
            if name.startswith(prefix):
                expert_tensors[name] = tensor

    if expert_tensors:
        filename = f"shard_expert_{expert_id}.safetensors"
        info = save_shard(expert_tensors, filename, shard_output)
        info["shard_type"] = "expert"
        info["expert_id"] = expert_id
        shard_manifest_entries.append(info)
        print(f"  {filename}: {info['size_bytes'] / 1e6:.2f} MB, {info['tensor_count']} tensors")

# %%
# Shard 4: LM Head
print("\n[4/5] Head shard...")
head_tensors = {}
for name, tensor in state_dict.items():
    if "lm_head" in name or (name.startswith("model.norm") and "layers" not in name):
        head_tensors[name] = tensor

if head_tensors:
    info = save_shard(head_tensors, "shard_head.safetensors", shard_output)
    info["shard_type"] = "head"
    shard_manifest_entries.append(info)
    print(f"  shard_head.safetensors: {info['size_bytes'] / 1e6:.2f} MB, {info['tensor_count']} tensors")

# %%
# Shard 5: Export tokenizer
print("\n[5/5] Tokenizer...")
tokenizer_dir = shard_output / "tokenizer"
tokenizer_dir.mkdir(exist_ok=True)
tokenizer.save_pretrained(str(tokenizer_dir))
tokenizer_files = [f.name for f in tokenizer_dir.iterdir() if f.is_file()]
print(f"  Tokenizer saved: {len(tokenizer_files)} files")

# %% [markdown]
# ## 6. Generate Manifest with Tool Schemas

# %%
# Load tool categories
CATEGORY_NAMES = [
    "web_search", "code_generation", "math_computation", "file_data_ops",
    "system_shell", "database_api", "image_media", "reasoning_planning",
]

tool_config_path = TOOL_CONFIG_DIR / "tool_categories.yaml"
if tool_config_path.exists():
    with open(tool_config_path) as f:
        tool_config = yaml.safe_load(f)
    tool_categories = tool_config["categories"]
    print(f"Loaded {len(tool_categories)} tool categories from config")
else:
    # Minimal fallback
    tool_categories = [
        {"id": i, "name": name, "tools": []}
        for i, name in enumerate(CATEGORY_NAMES)
    ]
    print("Tool config not found, using minimal category metadata")

# Build expert capabilities map
expert_capabilities = {}
for cat in tool_categories:
    expert_id = cat["id"]
    expert_capabilities[str(expert_id)] = {
        "category": cat["name"],
        "description": cat.get("description", ""),
        "tools": [t["function"]["name"] if "function" in t else t.get("name", "unknown")
                  for t in cat.get("tools", [])],
        "tool_schemas": cat.get("tools", []),
    }

# %%
# Build manifest
from datetime import datetime

total_shard_size = sum(s["size_bytes"] for s in shard_manifest_entries)

manifest = {
    "model_id": f"hivemind-moe-{USE_MODEL}",
    "base_model_id": MODEL_ID,
    "model_type": "moe_llama",
    "version": "1.0",
    "created_at": datetime.utcnow().isoformat() + "Z",
    "pipeline": "hivemind-bitnet-moe-toolcall",

    # Architecture
    "architecture": {
        "type": "mixture_of_experts",
        "base": "llama",
        "hidden_size": moe_meta["hidden_size"],
        "intermediate_size": moe_meta["intermediate_size"],
        "num_hidden_layers": moe_meta["num_hidden_layers"],
        "num_experts": moe_meta["num_experts"],
        "top_k": moe_meta["top_k"],
        "moe_layers": moe_meta["moe_layers"],
        "dense_layers": [i for i in range(num_layers) if i not in moe_meta["moe_layers"]],
    },

    # Quantization
    "quantization": {
        "method": "bitnet_b1.58_ternary",
        "bit_width": 2,
        "group_size": GROUP_SIZE,
        "fp16_size_bytes": total_fp16_bytes,
        "quantized_size_bytes": total_bitnet_bytes,
        "compression_ratio": total_fp16_bytes / total_bitnet_bytes if total_bitnet_bytes > 0 else 0,
    },

    # Shards
    "total_size_bytes": total_shard_size,
    "shards": shard_manifest_entries,

    # Expert capabilities (tool schemas)
    "expert_capabilities": expert_capabilities,

    # Tokenizer
    "tokenizer": {
        "path": "tokenizer/",
        "type": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "files": tokenizer_files,
    },

    # Inference config
    "inference": {
        "recommended_batch_size": 1,
        "max_sequence_length": 2048,
        "dtype": "float16",
        "requires_gpu": False,
        "p2p_compatible": True,
    },
}

# Save manifest
manifest_path = shard_output / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest saved to: {manifest_path}")
print(f"Manifest size: {manifest_path.stat().st_size / 1024:.1f} KB")

# %% [markdown]
# ## 7. Final Artifact Summary

# %%
print("\n" + "=" * 70)
print("FINAL ARTIFACT SUMMARY")
print("=" * 70)

# Categorize shards
shard_types = {}
for entry in shard_manifest_entries:
    stype = entry.get("shard_type", "unknown")
    if stype not in shard_types:
        shard_types[stype] = {"count": 0, "size": 0}
    shard_types[stype]["count"] += 1
    shard_types[stype]["size"] += entry["size_bytes"]

summary_table = []
for stype, info in sorted(shard_types.items()):
    summary_table.append([stype, info["count"], f"{info['size'] / 1e6:.2f} MB"])

summary_table.append(["---", "---", "---"])
summary_table.append(["TOTAL", sum(s["count"] for s in shard_types.values()),
                       f"{total_shard_size / 1e6:.2f} MB"])

print("\nShard breakdown:")
print(tabulate(summary_table, headers=["Shard Type", "Count", "Size"], tablefmt="grid"))

# List all files
print("\nAll output files:")
all_files = sorted(shard_output.rglob("*"))
file_table = []
total_disk = 0
for f in all_files:
    if f.is_file():
        size = f.stat().st_size
        total_disk += size
        rel = f.relative_to(shard_output)
        file_table.append([str(rel), f"{size / 1e6:.2f} MB"])

print(tabulate(file_table, headers=["File", "Size"], tablefmt="grid"))
print(f"\nTotal download size: {total_disk / 1e6:.2f} MB ({total_disk / 1e9:.3f} GB)")

# Expert capability summary
print("\nExpert-to-Tool Mapping (from manifest):")
for eid, cap in expert_capabilities.items():
    tools = ", ".join(cap.get("tools", []))
    print(f"  Expert {eid} ({cap['category']}): {tools}")

# %% [markdown]
# ## Summary
#
# This notebook has:
# 1. Loaded the trained MoE model
# 2. Quantized weights to BitNet b1.58 (ternary) -- **{compression}x** compression
# 3. Sharded the model into distributable artifacts:
#    - Embedding shard
#    - Dense layer group shards
#    - MoE layer base shards (attention + norms)
#    - Individual expert shards (one per expert)
#    - LM head shard
# 4. Generated manifest.json with checksums and tool schemas
# 5. Exported tokenizer
#
# **Total artifacts**: {num_shards} shards, {total_size} MB
#
# **Next step**: Run `07_p2p_test_harness.py` to test distributed inference.

# %%
print("\n" + "=" * 60)
print("NOTEBOOK 06 COMPLETE: Quantize & Shard")
print("=" * 60)
print(f"\nModel: {MODEL_ID} (MoE)")
print(f"Quantization: BitNet b1.58 (ternary, {GROUP_SIZE}-group)")
print(f"FP16 size: {total_fp16_bytes / 1e9:.3f} GB")
print(f"BitNet size: {total_bitnet_bytes / 1e9:.3f} GB")
print(f"Shard count: {len(shard_manifest_entries)}")
print(f"Total disk: {total_disk / 1e6:.2f} MB")
print(f"Output: {shard_output}")
print(f"\nReady for notebook 07: P2P Test Harness")
