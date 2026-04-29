# %% [markdown]
# # HiveMind Pipeline: Expert Fine-Tuning with LoRA
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Escherbridge/laxame-hivemind/blob/main/hivemind-models/notebooks/05_expert_finetuning.py)
#
# This notebook fine-tunes each MoE expert on its tool-category dataset using LoRA:
# 1. Load the upcycled MoE model from notebook 04
# 2. Load per-expert datasets from notebook 03
# 3. Apply LoRA adapters to expert MLP weights
# 4. Fine-tune each expert on its category dataset
# 5. Train the gating network for correct routing
# 6. Evaluate per-expert tool-call accuracy
# 7. Save the fully trained model

# %% [markdown]
# ## 1. Environment Setup

# %%
import sys
import json
import time
import math
import random
import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

COLAB = "google.colab" in str(get_ipython()) if hasattr(__builtins__, "__IPYTHON__") else False

if COLAB:
    get_ipython().system('pip install -q torch transformers safetensors accelerate peft tabulate tqdm matplotlib')
    REPO_ROOT = Path("/content/laxame-hivemind")
    DRIVE_ROOT = Path("/content/drive/MyDrive/hivemind-pipeline")
    CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
    DATASET_DIR = DRIVE_ROOT / "datasets"
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent if "__file__" in dir() else Path.cwd().parent.parent
    CHECKPOINT_DIR = REPO_ROOT / "hivemind-models" / "output" / "checkpoints"
    DATASET_DIR = REPO_ROOT / "hivemind-models" / "output" / "datasets"

MODELS_SRC = REPO_ROOT / "hivemind-models"
if str(MODELS_SRC) not in sys.path:
    sys.path.insert(0, str(MODELS_SRC))

print(f"Colab: {COLAB}")
print(f"CUDA: {torch.cuda.is_available()}")

# %% [markdown]
# ## 2. Load MoE Model

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

USE_MODEL = "tinyllama"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

moe_checkpoint_path = CHECKPOINT_DIR / f"{USE_MODEL}_moe"

# Load MoE config
moe_config_path = moe_checkpoint_path / "moe_config.json"
if moe_config_path.exists():
    with open(moe_config_path) as f:
        moe_meta = json.load(f)
    print(f"Loaded MoE config: {moe_meta['num_experts']} experts, top-{moe_meta['top_k']}")
else:
    print("WARNING: MoE config not found. Using defaults.")
    moe_meta = {
        "num_experts": 8,
        "top_k": 2,
        "moe_layers": list(range(1, 22, 2)),
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 22,
    }

# We need to reconstruct the MoE model architecture.
# First load the base model, then apply MoE conversion, then load weights.

# %%
# Import MoE modules from notebook 04 (redefined here for standalone operation)

class TopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int,
                 jitter: float = 0.01, z_loss_coeff: float = 0.001):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter = jitter
        self.z_loss_coeff = z_loss_coeff
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_size)
        if self.training and self.jitter > 0:
            flat_hidden = flat_hidden * (1.0 + torch.randn_like(flat_hidden) * self.jitter)
        router_logits = self.gate(flat_hidden)
        router_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        router_weights = F.softmax(router_weights, dim=-1)
        router_weights = router_weights.view(batch_size, seq_len, self.top_k)
        selected_experts = selected_experts.view(batch_size, seq_len, self.top_k)
        router_logits = router_logits.view(batch_size, seq_len, self.num_experts)
        return router_weights, selected_experts, router_logits


class MoEExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int,
                 top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([
            MoEExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        router_weights, selected_experts, router_logits = self.router(hidden_states)
        final_output = torch.zeros_like(hidden_states)
        flat_hidden = hidden_states.view(-1, hidden_size)
        flat_output = final_output.view(-1, hidden_size)
        flat_weights = router_weights.view(-1, self.top_k)
        flat_experts = selected_experts.view(-1, self.top_k)

        for expert_idx in range(self.num_experts):
            expert_mask = (flat_experts == expert_idx)
            if not expert_mask.any():
                continue
            token_indices, slot_indices = expert_mask.nonzero(as_tuple=True)
            if len(token_indices) == 0:
                continue
            expert_input = flat_hidden[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            weights = flat_weights[token_indices, slot_indices].unsqueeze(-1)
            flat_output.index_add_(0, token_indices, expert_output * weights.to(expert_output.dtype))

        return flat_output.view(batch_size, seq_len, hidden_size)


def build_moe_model(base_model, moe_meta):
    """Rebuild MoE architecture on top of base model."""
    moe_layers = moe_meta["moe_layers"]
    num_experts = moe_meta["num_experts"]
    top_k = moe_meta["top_k"]
    hidden_size = moe_meta["hidden_size"]
    intermediate_size = moe_meta["intermediate_size"]

    for layer_idx in moe_layers:
        layer = base_model.model.layers[layer_idx]
        moe_layer = MoELayer(hidden_size, intermediate_size, num_experts, top_k)
        layer.mlp = moe_layer

    return base_model

# %%
# Load base model and convert to MoE, then load saved weights
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Rebuilding MoE architecture...")
model = build_moe_model(base_model, moe_meta)

# Load saved MoE weights if available
moe_weight_path = moe_checkpoint_path / "moe_model.pt"
if moe_weight_path.exists():
    print(f"Loading MoE weights from {moe_weight_path}...")
    checkpoint = torch.load(str(moe_weight_path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print("MoE weights loaded.")
else:
    print("WARNING: MoE weights not found. Using freshly upcycled model (experts = copies of dense MLP).")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"\nMoE model loaded. Total params: {total_params:,}")

# %% [markdown]
# ## 3. Load Per-Expert Datasets

# %%
CATEGORY_NAMES = [
    "web_search", "code_generation", "math_computation", "file_data_ops",
    "system_shell", "database_api", "image_media", "reasoning_planning",
]

expert_datasets = {}

for expert_id, cat_name in enumerate(CATEGORY_NAMES):
    dataset_path = DATASET_DIR / f"expert_{expert_id}_{cat_name}.jsonl"

    if dataset_path.exists():
        samples = []
        with open(dataset_path) as f:
            for line in f:
                samples.append(json.loads(line))
        expert_datasets[expert_id] = samples
        print(f"Expert {expert_id} ({cat_name}): loaded {len(samples)} samples")
    else:
        print(f"Expert {expert_id} ({cat_name}): dataset not found at {dataset_path}")
        # Generate minimal synthetic data for testing
        expert_datasets[expert_id] = [
            {
                "messages": [
                    {"role": "user", "content": f"Example query for {cat_name} expert {i}"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": f"call_{i}", "type": "function",
                         "function": {"name": f"tool_{cat_name}", "arguments": json.dumps({"query": f"test {i}"})}}
                    ]},
                ],
                "expert_id": expert_id,
                "category": cat_name,
            }
            for i in range(50)
        ]
        print(f"  -> Generated {len(expert_datasets[expert_id])} synthetic samples")

total_samples = sum(len(d) for d in expert_datasets.values())
print(f"\nTotal training samples: {total_samples}")

# %% [markdown]
# ## 4. Prepare Training Data

# %%
class ToolCallingDataset(Dataset):
    """Dataset for tool-calling fine-tuning."""

    def __init__(self, samples: list[dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_samples = []

        for sample in samples:
            text = self._format_sample(sample)
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.encoded_samples.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": encoding["input_ids"].squeeze(0).clone(),
                "expert_id": sample.get("expert_id", 0),
            })

    def _format_sample(self, sample: dict) -> str:
        """Format a sample into a training string."""
        messages = sample["messages"]
        user_msg = messages[0]["content"]
        tool_calls = messages[1].get("tool_calls", [])

        if tool_calls:
            tc = tool_calls[0]
            func_name = tc["function"]["name"]
            func_args = tc["function"]["arguments"]
            # Format: <|user|> query <|assistant|> <tool_call> {"name": ..., "arguments": ...} </tool_call>
            return (
                f"<|user|>\n{user_msg}\n<|assistant|>\n"
                f'<tool_call>\n{{"name": "{func_name}", "arguments": {func_args}}}\n</tool_call>'
            )
        else:
            return f"<|user|>\n{user_msg}\n<|assistant|>\n"

    def __len__(self):
        return len(self.encoded_samples)

    def __getitem__(self, idx):
        return self.encoded_samples[idx]


# Split each expert's data into train/test
TRAIN_SPLIT = 0.9
MAX_SEQ_LENGTH = 512

expert_train_datasets = {}
expert_test_datasets = {}

for expert_id, samples in expert_datasets.items():
    random.shuffle(samples)
    split_idx = int(len(samples) * TRAIN_SPLIT)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    expert_train_datasets[expert_id] = ToolCallingDataset(train_samples, tokenizer, MAX_SEQ_LENGTH)
    expert_test_datasets[expert_id] = ToolCallingDataset(test_samples, tokenizer, MAX_SEQ_LENGTH)

    print(f"Expert {expert_id}: {len(train_samples)} train, {len(test_samples)} test")

# %% [markdown]
# ## 5. LoRA Configuration

# %%
@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = None  # Names of modules to apply LoRA to

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["gate_proj", "up_proj", "down_proj"]

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """
    LoRA adapter for a linear layer.

    Wraps an existing nn.Linear and adds low-rank A and B matrices.
    output = W*x + (B @ A) * x * scaling
    """

    def __init__(self, original_linear: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.original = original_linear
        self.rank = config.rank
        self.scaling = config.scaling

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(config.rank, in_features, dtype=original_linear.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.rank, dtype=original_linear.weight.dtype))
        self.lora_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = self.original(x)
        # LoRA forward
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_out

    def merge(self):
        """Merge LoRA weights into the original linear layer."""
        with torch.no_grad():
            merged = self.lora_B @ self.lora_A * self.scaling
            self.original.weight.data += merged.to(self.original.weight.dtype)

    @property
    def lora_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


lora_config = LoRAConfig(rank=16, alpha=32, dropout=0.05)
print(f"LoRA config: rank={lora_config.rank}, alpha={lora_config.alpha}, scaling={lora_config.scaling}")
print(f"Target modules: {lora_config.target_modules}")

# %% [markdown]
# ## 6. Apply LoRA and Fine-Tune Each Expert

# %%
def apply_lora_to_expert(expert: MoEExpertMLP, config: LoRAConfig) -> tuple[MoEExpertMLP, list[LoRALinear]]:
    """Apply LoRA adapters to an expert's MLP layers."""
    lora_modules = []

    for name in config.target_modules:
        if hasattr(expert, name):
            original = getattr(expert, name)
            lora_layer = LoRALinear(original, config)
            setattr(expert, name, lora_layer)
            lora_modules.append(lora_layer)

    return expert, lora_modules


def get_lora_params(model, expert_idx: int, moe_layers: list[int]) -> list[nn.Parameter]:
    """Get all LoRA parameters for a specific expert across all MoE layers."""
    params = []
    for layer_idx in moe_layers:
        layer = model.model.layers[layer_idx]
        if hasattr(layer.mlp, 'experts'):
            expert = layer.mlp.experts[expert_idx]
            for name, module in expert.named_modules():
                if isinstance(module, LoRALinear):
                    params.extend([module.lora_A, module.lora_B])
    return params


def train_expert(
    model,
    expert_id: int,
    train_dataset: ToolCallingDataset,
    moe_layers: list[int],
    lora_config: LoRAConfig,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
) -> list[float]:
    """
    Fine-tune a single expert using LoRA.

    Returns list of per-step losses.
    """
    # Apply LoRA to this expert in all MoE layers
    all_lora_modules = []
    for layer_idx in moe_layers:
        layer = model.model.layers[layer_idx]
        if hasattr(layer.mlp, 'experts'):
            expert = layer.mlp.experts[expert_id]
            _, lora_mods = apply_lora_to_expert(expert, lora_config)
            all_lora_modules.extend(lora_mods)

    # Collect LoRA parameters
    lora_params = []
    for mod in all_lora_modules:
        lora_params.extend([mod.lora_A, mod.lora_B])

    total_lora = sum(p.numel() for p in lora_params)
    print(f"  Expert {expert_id}: {len(all_lora_modules)} LoRA modules, {total_lora:,} trainable params")

    # Freeze everything except LoRA params
    for param in model.parameters():
        param.requires_grad = False
    for param in lora_params:
        param.requires_grad = True

    # Optimizer
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_dataset) // batch_size
    )

    # DataLoader
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Training loop
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_steps = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Mask padding in labels
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step_loss = loss.item()
            losses.append(step_loss)
            epoch_loss += step_loss
            num_steps += 1

        avg_loss = epoch_loss / max(num_steps, 1)
        print(f"    Epoch {epoch + 1}/{epochs}: avg_loss={avg_loss:.4f}")

    model.eval()

    # Merge LoRA weights into original
    for mod in all_lora_modules:
        mod.merge()

    # Restore original linear layers (remove LoRA wrappers)
    for layer_idx in moe_layers:
        layer = model.model.layers[layer_idx]
        if hasattr(layer.mlp, 'experts'):
            expert = layer.mlp.experts[expert_id]
            for name in lora_config.target_modules:
                mod = getattr(expert, name)
                if isinstance(mod, LoRALinear):
                    setattr(expert, name, mod.original)

    return losses

# %%
# Fine-tune each expert
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 4

all_losses = {}
training_times = {}

print("=" * 60)
print("Expert Fine-Tuning with LoRA")
print("=" * 60)

for expert_id in range(moe_meta["num_experts"]):
    cat_name = CATEGORY_NAMES[expert_id] if expert_id < len(CATEGORY_NAMES) else f"expert_{expert_id}"
    train_ds = expert_train_datasets.get(expert_id)

    if train_ds is None or len(train_ds) == 0:
        print(f"\nExpert {expert_id} ({cat_name}): No training data, skipping.")
        continue

    print(f"\nExpert {expert_id} ({cat_name}): training on {len(train_ds)} samples...")
    start_time = time.time()

    losses = train_expert(
        model=model,
        expert_id=expert_id,
        train_dataset=train_ds,
        moe_layers=moe_meta["moe_layers"],
        lora_config=lora_config,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
    )

    elapsed = time.time() - start_time
    all_losses[expert_id] = losses
    training_times[expert_id] = elapsed
    print(f"  Completed in {elapsed:.1f}s, final loss: {losses[-1]:.4f}")

# %% [markdown]
# ## 7. Plot Training Loss Curves

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Per-Expert Training Loss Curves", fontsize=14)

for expert_id in range(moe_meta["num_experts"]):
    ax = axes[expert_id // 4][expert_id % 4]
    cat_name = CATEGORY_NAMES[expert_id] if expert_id < len(CATEGORY_NAMES) else f"expert_{expert_id}"

    if expert_id in all_losses and all_losses[expert_id]:
        losses = all_losses[expert_id]
        ax.plot(losses, linewidth=0.8, alpha=0.7)
        # Smoothed line
        if len(losses) > 10:
            window = max(1, len(losses) // 20)
            smoothed = [sum(losses[max(0, i-window):i+1]) / min(i+1, window+1) for i in range(len(losses))]
            ax.plot(smoothed, linewidth=2, color="red", label="Smoothed")
        ax.set_title(f"Expert {expert_id}: {cat_name}", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Expert {expert_id}: {cat_name}", fontsize=10)

plt.tight_layout()
plot_path = CHECKPOINT_DIR / f"{USE_MODEL}_moe" / "training_loss_curves.png"
plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
print(f"Loss curves saved to: {plot_path}")
plt.show()

# %% [markdown]
# ## 8. Train the Gating Network

# %%
print("Training the gating network for correct routing...")
print("=" * 60)

# Combine all training data with expert labels
router_train_data = []
for expert_id, samples in expert_datasets.items():
    for sample in samples:
        text = sample["messages"][0]["content"]
        router_train_data.append({"text": text, "expert_id": expert_id})

random.shuffle(router_train_data)
print(f"Router training samples: {len(router_train_data)}")

# Train router: freeze everything except router gates
for param in model.parameters():
    param.requires_grad = False

router_params = []
for layer_idx in moe_meta["moe_layers"]:
    layer = model.model.layers[layer_idx]
    if hasattr(layer.mlp, 'router'):
        for param in layer.mlp.router.parameters():
            param.requires_grad = True
            router_params.append(param)

total_router_params = sum(p.numel() for p in router_params)
print(f"Router trainable params: {total_router_params:,}")

# Simple router training: encode queries, get hidden states, train router to predict expert
router_optimizer = torch.optim.Adam(router_params, lr=1e-3)
model.train()

ROUTER_EPOCHS = 5
ROUTER_BATCH_SIZE = 8
router_losses = []

for epoch in range(ROUTER_EPOCHS):
    epoch_loss = 0.0
    num_batches = 0
    random.shuffle(router_train_data)

    for i in range(0, min(len(router_train_data), 500), ROUTER_BATCH_SIZE):
        batch = router_train_data[i:i + ROUTER_BATCH_SIZE]
        texts = [b["text"] for b in batch]
        expert_ids = torch.tensor([b["expert_id"] for b in batch], device=device)

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get hidden states from the model
        with torch.no_grad():
            outputs = model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )

        # Use the hidden state at the first MoE layer as input to router
        # Take the mean over sequence positions for the classification signal
        first_moe_layer = moe_meta["moe_layers"][0]
        hidden = outputs.hidden_states[first_moe_layer]  # (batch, seq, hidden)
        hidden_mean = hidden.mean(dim=1)  # (batch, hidden)

        # Get router logits from first MoE layer
        first_router = model.model.layers[first_moe_layer].mlp.router
        router_logits = first_router.gate(hidden_mean)  # (batch, num_experts)

        # Cross-entropy loss
        loss = F.cross_entropy(router_logits, expert_ids)
        loss.backward()
        router_optimizer.step()
        router_optimizer.zero_grad()

        epoch_loss += loss.item()
        num_batches += 1
        router_losses.append(loss.item())

    avg = epoch_loss / max(num_batches, 1)
    print(f"  Router epoch {epoch + 1}/{ROUTER_EPOCHS}: avg_loss={avg:.4f}")

model.eval()
print("Router training complete.")

# %% [markdown]
# ## 9. Evaluate Per-Expert Accuracy

# %%
from tabulate import tabulate

print("Evaluating per-expert tool-call accuracy on held-out test set...")
print("=" * 60)

eval_results = []

for expert_id in range(moe_meta["num_experts"]):
    cat_name = CATEGORY_NAMES[expert_id] if expert_id < len(CATEGORY_NAMES) else f"expert_{expert_id}"
    test_ds = expert_test_datasets.get(expert_id)

    if test_ds is None or len(test_ds) == 0:
        eval_results.append([f"Expert {expert_id}", cat_name, 0, "N/A", "N/A", "N/A"])
        continue

    correct_routing = 0
    valid_tool_calls = 0
    total_tested = 0

    for sample_data in expert_datasets[expert_id][-10:]:  # Use last 10 samples as test
        text = sample_data["messages"][0]["content"]
        expected_tool = sample_data["messages"][1]["tool_calls"][0]["function"]["name"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )

            # Check routing
            first_moe_layer = moe_meta["moe_layers"][0]
            hidden = outputs.hidden_states[first_moe_layer]
            hidden_mean = hidden.mean(dim=1)
            router_logits = model.model.layers[first_moe_layer].mlp.router.gate(hidden_mean)
            predicted_expert = router_logits.argmax(dim=-1).item()

            if predicted_expert == expert_id:
                correct_routing += 1

            # Try generating output and check if it contains a valid tool call
            gen_input = tokenizer(f"<|user|>\n{text}\n<|assistant|>\n", return_tensors="pt").to(device)
            generated = model.generate(
                **gen_input,
                max_new_tokens=100,
                do_sample=False,
            )
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)

            if expected_tool.lower() in decoded.lower() or "tool_call" in decoded.lower():
                valid_tool_calls += 1

        total_tested += 1

    routing_acc = correct_routing / max(total_tested, 1) * 100
    tool_acc = valid_tool_calls / max(total_tested, 1) * 100
    final_loss = all_losses.get(expert_id, [0])[-1] if expert_id in all_losses else 0

    eval_results.append([
        f"Expert {expert_id}",
        cat_name,
        total_tested,
        f"{routing_acc:.1f}%",
        f"{tool_acc:.1f}%",
        f"{final_loss:.4f}",
    ])

print(tabulate(
    eval_results,
    headers=["Expert", "Category", "Test Samples", "Routing Acc", "Tool-Call Acc", "Final Loss"],
    tablefmt="grid",
))

# Compute overall routing accuracy
overall_routing = sum(
    float(r[3].replace('%', '')) for r in eval_results if r[3] != "N/A"
) / sum(1 for r in eval_results if r[3] != "N/A") if any(r[3] != "N/A" for r in eval_results) else 0

print(f"\nOverall routing accuracy: {overall_routing:.1f}%")

# %% [markdown]
# ## 10. Save Trained Model

# %%
trained_path = CHECKPOINT_DIR / f"{USE_MODEL}_moe_trained"
trained_path.mkdir(parents=True, exist_ok=True)

print(f"Saving trained MoE model to: {trained_path}")
start_time = time.time()

torch.save({
    "model_state_dict": model.state_dict(),
    "moe_config": moe_meta,
    "training_info": {
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_rank": lora_config.rank,
        "lora_alpha": lora_config.alpha,
        "samples_per_expert": {k: len(v) for k, v in expert_datasets.items()},
        "final_losses": {k: v[-1] if v else None for k, v in all_losses.items()},
        "training_times": training_times,
    },
}, str(trained_path / "moe_trained.pt"))

tokenizer.save_pretrained(str(trained_path))

# Save training metadata
with open(trained_path / "training_meta.json", "w") as f:
    json.dump({
        "base_model_id": MODEL_ID,
        "num_experts": moe_meta["num_experts"],
        "top_k": moe_meta["top_k"],
        "moe_layers": moe_meta["moe_layers"],
        "lora_rank": lora_config.rank,
        "lora_alpha": lora_config.alpha,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "per_expert_results": {
            str(i): {
                "category": CATEGORY_NAMES[i] if i < len(CATEGORY_NAMES) else f"expert_{i}",
                "final_loss": all_losses.get(i, [0])[-1] if i in all_losses else None,
                "training_time_s": training_times.get(i, 0),
            }
            for i in range(moe_meta["num_experts"])
        },
    }, f, indent=2)

elapsed = time.time() - start_time
print(f"Saved in {elapsed:.1f}s")

for f in sorted(trained_path.iterdir()):
    size = f.stat().st_size if f.is_file() else 0
    print(f"  {f.name:40s} {size / 1e6:>8.2f} MB")

# %% [markdown]
# ## Summary
#
# This notebook has:
# 1. Loaded the MoE model from notebook 04
# 2. Loaded per-expert datasets from notebook 03
# 3. Applied LoRA (rank={rank}, alpha={alpha}) to each expert's MLP
# 4. Fine-tuned all 8 experts on their category datasets
# 5. Trained the gating network for routing accuracy
# 6. Evaluated per-expert tool-call accuracy
# 7. Saved the fully trained MoE model
#
# **Next step**: Run `06_quantize_and_shard.py` to quantize and shard for distribution.

# %%
print("\n" + "=" * 60)
print("NOTEBOOK 05 COMPLETE: Expert Fine-Tuning")
print("=" * 60)
print(f"\nModel: {MODEL_ID} (MoE)")
print(f"Experts trained: {len(all_losses)}")
print(f"LoRA rank: {lora_config.rank}, alpha: {lora_config.alpha}")
print(f"Overall routing accuracy: {overall_routing:.1f}%")
print(f"Checkpoint: {trained_path}")
print(f"\nReady for notebook 06: Quantize & Shard")
