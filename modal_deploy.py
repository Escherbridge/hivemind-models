"""
Modal deployment for GPU-accelerated model conversion.

Deploy model sharding to Modal's GPU infrastructure for fast conversion.

Usage:
    # Install modal first: pip install modal
    # Authenticate: modal token new

    # Run conversion on Modal
    modal run modal_deploy.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0

    # Deploy as persistent function
    modal deploy modal_deploy.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# Create Modal app
app = modal.App("hivemind-models")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "safetensors>=0.4.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "boto3>=1.28.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    )
    .run_commands("pip install huggingface_hub")
)

# Volume for storing converted models
volume = modal.Volume.from_name("hivemind-models-volume", create_if_missing=True)
VOLUME_PATH = "/models"


@app.function(
    image=image,
    gpu="A10G",  # Good balance of cost and performance
    timeout=3600,  # 1 hour timeout for large models
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("r2-credentials", required=False)],
)
def convert_model(
    model_id: str,
    layer_groups: list[tuple[int, int]] | None = None,
    quantize: bool = True,
    quant_bits: int = 4,
    upload_to_r2: bool = False,
) -> dict:
    """
    Convert a HuggingFace model to sharded format on Modal GPU.

    Args:
        model_id: HuggingFace model ID
        layer_groups: Custom layer groupings (auto-detected if None)
        quantize: Whether to quantize the model
        quant_bits: Quantization bits (4 or 8)
        upload_to_r2: Whether to upload to R2 after conversion

    Returns:
        Dictionary with conversion results
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # Auto-detect layer groups if not provided
    if layer_groups is None:
        config = AutoConfig.from_pretrained(model_id)
        num_layers = config.num_hidden_layers

        # Create balanced groups
        layers_per_group = num_layers // 4
        layer_groups = []
        for i in range(4):
            start = i * layers_per_group
            end = (i + 1) * layers_per_group - 1 if i < 3 else num_layers - 1
            layer_groups.append((start, end))

    # Create output directory
    model_name = model_id.replace("/", "_")
    output_dir = Path(VOLUME_PATH) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {model_id}")
    print(f"Layer groups: {layer_groups}")
    print(f"Quantize: {quantize} ({quant_bits}-bit)")
    print(f"Output: {output_dir}")

    # Import sharding module
    # Since we're running in Modal, we need to define the sharding logic inline
    # or copy the relevant code

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    state_dict = model.state_dict()
    print(f"Model loaded with {len(state_dict)} tensors")

    # Helper functions for sharding
    def extract_embedding_tensors(sd):
        return {k: v for k, v in sd.items() if "embed_tokens" in k}

    def extract_layer_tensors(sd, start, end):
        tensors = {}
        for layer_idx in range(start, end + 1):
            prefix = f"model.layers.{layer_idx}"
            for k, v in sd.items():
                if k.startswith(prefix):
                    tensors[k] = v
        return tensors

    def extract_head_tensors(sd):
        return {k: v for k, v in sd.items() if "lm_head" in k or k == "model.norm.weight"}

    def save_shard(tensors, filename):
        from safetensors.torch import save_file
        import hashlib

        # Simple quantization (dequantized float16)
        if quantize:
            qmax = 2 ** (quant_bits - 1) - 1
            quantized = {}
            for name, tensor in tensors.items():
                if "weight" in name and tensor.dim() >= 2:
                    absmax = tensor.abs().max()
                    if absmax > 0:
                        scale = absmax / qmax
                        q = torch.round(tensor / scale).clamp(-qmax - 1, qmax)
                        tensor = (q * scale).to(torch.float16)
                quantized[name] = tensor.cpu().contiguous()
            tensors = quantized

        cpu_tensors = {k: v.cpu().contiguous() for k, v in tensors.items()}
        path = output_dir / filename
        save_file(cpu_tensors, str(path))

        # Compute checksum
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        return {
            "filename": filename,
            "size_bytes": path.stat().st_size,
            "checksum": hasher.hexdigest(),
            "tensor_count": len(tensors),
        }

    shards = []
    total_size = 0

    # 1. Embedding shard
    print("Creating embedding shard...")
    embed_tensors = extract_embedding_tensors(state_dict)
    if embed_tensors:
        info = save_shard(embed_tensors, "shard_embed.safetensors")
        shards.append(info)
        total_size += info["size_bytes"]

    # 2. Layer shards
    for start, end in layer_groups:
        print(f"Creating shard for layers {start}-{end}...")
        layer_tensors = extract_layer_tensors(state_dict, start, end)
        if layer_tensors:
            info = save_shard(layer_tensors, f"shard_layers_{start}_{end}.safetensors")
            info["layer_range"] = [start, end]
            shards.append(info)
            total_size += info["size_bytes"]

    # 3. Head shard
    print("Creating head shard...")
    head_tensors = extract_head_tensors(state_dict)
    if head_tensors:
        info = save_shard(head_tensors, "shard_head.safetensors")
        shards.append(info)
        total_size += info["size_bytes"]

    # 4. Export tokenizer
    print("Exporting tokenizer...")
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))

    # 5. Create manifest
    manifest = {
        "model_id": model_id,
        "model_type": "llama",
        "version": "1.0",
        "total_size_bytes": total_size,
        "quantized": quantize,
        "quant_bits": quant_bits if quantize else None,
        "dtype": "float16",
        "layer_groups": layer_groups,
        "shards": shards,
        "tokenizer": {
            "path": "tokenizer/",
            "type": type(tokenizer).__name__,
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nConversion complete!")
    print(f"Total size: {total_size / (1024**2):.2f} MB")
    print(f"Shards: {len(shards)}")

    # Upload to R2 if requested
    if upload_to_r2:
        print("\nUploading to R2...")
        r2_url = upload_to_r2_storage(output_dir, model_name)
        manifest["cdn_base_url"] = r2_url
        print(f"Uploaded to: {r2_url}")

    # Commit volume changes
    volume.commit()

    return {
        "model_id": model_id,
        "output_dir": str(output_dir),
        "total_size_mb": total_size / (1024**2),
        "shards": len(shards),
        "manifest": manifest,
    }


def upload_to_r2_storage(output_dir: Path, model_name: str) -> str | None:
    """Upload converted model to R2."""
    import boto3

    # Get credentials from environment
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY")
    secret_key = os.environ.get("R2_SECRET_KEY")
    bucket_name = os.environ.get("R2_BUCKET_NAME")
    public_url = os.environ.get("R2_PUBLIC_URL")

    if not all([account_id, access_key, secret_key, bucket_name]):
        print("R2 credentials not configured, skipping upload")
        return None

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    # Upload all files
    for file_path in output_dir.rglob("*"):
        if file_path.is_file():
            key = f"models/{model_name}/{file_path.relative_to(output_dir)}"
            key = key.replace("\\", "/")

            print(f"  Uploading {file_path.name}...")
            client.upload_file(str(file_path), bucket_name, key)

    if public_url:
        return f"{public_url.rstrip('/')}/models/{model_name}"
    return f"https://{bucket_name}.r2.cloudflarestorage.com/models/{model_name}"


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def list_converted_models() -> list[str]:
    """List all converted models in the volume."""
    models = []
    models_path = Path(VOLUME_PATH)

    if models_path.exists():
        for item in models_path.iterdir():
            if item.is_dir() and (item / "manifest.json").exists():
                models.append(item.name)

    return models


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def get_model_manifest(model_name: str) -> dict | None:
    """Get the manifest for a converted model."""
    manifest_path = Path(VOLUME_PATH) / model_name / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)

    return None


@app.local_entrypoint()
def main(
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantize: bool = True,
    quant_bits: int = 4,
    upload: bool = False,
):
    """
    Convert a model using Modal GPU.

    Example:
        modal run modal_deploy.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0
    """
    print(f"Starting conversion of {model_id}")

    result = convert_model.remote(
        model_id=model_id,
        quantize=quantize,
        quant_bits=quant_bits,
        upload_to_r2=upload,
    )

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Model: {result['model_id']}")
    print(f"Output: {result['output_dir']}")
    print(f"Size: {result['total_size_mb']:.2f} MB")
    print(f"Shards: {result['shards']}")

    if result['manifest'].get('cdn_base_url'):
        print(f"CDN URL: {result['manifest']['cdn_base_url']}")

    return result
