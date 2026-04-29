"""
Per-expert fine-tuning with LoRA adapters.

``ExpertTrainer`` attaches lightweight LoRA adapters to each expert's MLP
weights, trains each expert on its designated tool-category dataset, and
then merges the adapters back into the base weights.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.moe.upcycle import MoELayer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class ExpertTrainingConfig:
    """Per-expert or global training hyper-parameters."""

    learning_rate: float = 2e-4
    num_steps: int = 500
    batch_size: int = 4
    max_seq_len: int = 512
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    warmup_steps: int = 50
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    log_every: int = 25

    # Per-expert overrides: expert_index -> partial config dict
    per_expert: dict[int, dict[str, Any]] = field(default_factory=dict)

    def for_expert(self, expert_idx: int) -> "ExpertTrainingConfig":
        """Return a copy with per-expert overrides applied."""
        overrides = self.per_expert.get(expert_idx, {})
        if not overrides:
            return self
        vals = {
            f.name: getattr(self, f.name)
            for f in self.__dataclass_fields__.values()
            if f.name != "per_expert"
        }
        vals.update(overrides)
        return ExpertTrainingConfig(**vals)


# ---------------------------------------------------------------------------
# Minimal LoRA adapter
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Drop-in LoRA wrapper around an existing ``nn.Linear``.

    ``output = original(x) + (x @ A^T @ B^T) * (alpha / rank)``
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def merge_and_unload(self) -> nn.Linear:
        """Merge LoRA weights back into the original linear and return it."""
        with torch.no_grad():
            self.original.weight.add_(
                (self.lora_B @ self.lora_A) * self.scaling
            )
        return self.original


# ---------------------------------------------------------------------------
# Simple JSONL dataset
# ---------------------------------------------------------------------------

class ToolCallingDataset(Dataset):
    """
    Loads a JSONL file produced by ``ToolCallingDatasetGenerator`` and
    tokenises messages into input_ids / labels tensors.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Any,
        max_seq_len: int = 512,
    ) -> None:
        self.samples: list[dict] = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        logger.info("Loaded %d samples from %s", len(self.samples), jsonl_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        # Concatenate all message contents
        text_parts: list[str] = []
        for msg in sample.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            text_parts.append(f"<|{role}|>\n{content}")
        text = "\n".join(text_parts)

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Causal LM: labels = input_ids, ignore padding
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Expert trainer
# ---------------------------------------------------------------------------

class ExpertTrainer:
    """
    Fine-tune each expert in an MoE model on its designated dataset.

    Workflow:
        1. Attach LoRA adapters to each expert's linear layers.
        2. Train each expert individually using its tool-category dataset.
        3. Merge LoRA weights back into the base expert weights.
        4. Log per-expert loss curves.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: ExpertTrainingConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._loss_curves: dict[int, list[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_all_experts(
        self,
        expert_datasets: dict[int, str | Path],
    ) -> dict[int, list[float]]:
        """
        Train every expert whose index appears in *expert_datasets*.

        Args:
            expert_datasets: Mapping from expert index to JSONL file path.

        Returns:
            Per-expert loss curves (expert_idx -> list of loss values).
        """
        moe_layers = self._find_moe_layers()
        logger.info(
            "Found %d MoE layers in the model. Training %d experts.",
            len(moe_layers),
            len(expert_datasets),
        )

        for expert_idx, dataset_path in expert_datasets.items():
            logger.info("=== Training expert %d ===", expert_idx)
            cfg = self.config.for_expert(expert_idx)
            dataset = ToolCallingDataset(
                dataset_path, self.tokenizer, max_seq_len=cfg.max_seq_len
            )
            losses = self._train_single_expert(
                expert_idx, moe_layers, dataset, cfg
            )
            self._loss_curves[expert_idx] = losses
            logger.info(
                "Expert %d training complete. Final loss: %.4f",
                expert_idx,
                losses[-1] if losses else float("nan"),
            )

        return self._loss_curves

    @property
    def loss_curves(self) -> dict[int, list[float]]:
        return dict(self._loss_curves)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_moe_layers(self) -> list[MoELayer]:
        """Walk the model and collect all MoELayer instances."""
        moe_layers: list[MoELayer] = []
        for module in self.model.modules():
            if isinstance(module, MoELayer):
                moe_layers.append(module)
        return moe_layers

    def _train_single_expert(
        self,
        expert_idx: int,
        moe_layers: list[MoELayer],
        dataset: ToolCallingDataset,
        cfg: ExpertTrainingConfig,
    ) -> list[float]:
        """Apply LoRA, train, merge, return losses."""
        # 1. Attach LoRA to every linear in this expert across all MoE layers
        lora_modules: list[tuple[nn.Module, str, LoRALinear]] = []
        for moe_layer in moe_layers:
            if expert_idx >= len(moe_layer.experts):
                continue
            expert = moe_layer.experts[expert_idx]
            for name, child in list(expert.named_modules()):
                if isinstance(child, nn.Linear):
                    lora = LoRALinear(
                        child,
                        rank=cfg.lora_rank,
                        alpha=cfg.lora_alpha,
                        dropout=cfg.lora_dropout,
                    )
                    # Replace in parent
                    parts = name.split(".")
                    parent = expert
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], lora)
                    lora_modules.append((parent, parts[-1], lora))

        logger.info(
            "Expert %d: attached %d LoRA adapters", expert_idx, len(lora_modules)
        )

        # 2. Build optimiser — only LoRA params are trainable
        lora_params = []
        for _, _, lora in lora_modules:
            lora_params.extend([lora.lora_A, lora.lora_B])

        if not lora_params:
            logger.warning("Expert %d: no trainable params found.", expert_idx)
            return []

        optimizer = torch.optim.AdamW(
            lora_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Simple linear warmup + cosine decay
        def lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return float(step + 1) / float(max(1, cfg.warmup_steps))
            progress = (step - cfg.warmup_steps) / max(
                1, cfg.num_steps - cfg.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # 3. Training loop
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )
        data_iter = iter(dataloader)

        device = next(self.model.parameters()).device
        scaler = torch.amp.GradScaler("cuda") if cfg.fp16 and device.type == "cuda" else None

        self.model.train()
        losses: list[float] = []

        for step in tqdm(range(cfg.num_steps), desc=f"Expert {expert_idx}"):
            # Get batch (cycle through data)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / cfg.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / cfg.gradient_accumulation_steps
                loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            loss_val = loss.item() * cfg.gradient_accumulation_steps
            losses.append(loss_val)

            if (step + 1) % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                logger.info(
                    "Expert %d | step %d/%d | loss %.4f | lr %.2e",
                    expert_idx,
                    step + 1,
                    cfg.num_steps,
                    loss_val,
                    lr_now,
                )

        # 4. Merge LoRA weights back
        for parent, attr_name, lora in lora_modules:
            merged = lora.merge_and_unload()
            setattr(parent, attr_name, merged)

        logger.info("Expert %d: LoRA merged back into base weights.", expert_idx)
        return losses

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_loss_curves(self, path: str | Path) -> None:
        """Save loss curves as JSON for later plotting."""
        data = {str(k): v for k, v in self._loss_curves.items()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Loss curves saved to %s", path)
