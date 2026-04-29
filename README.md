# Hivemind Models

Model sharding and CDN upload pipeline for Hivemind distributed WebGPU inference.

## Overview

This package converts HuggingFace models into sharded safetensors files optimized for distributed inference across browser clients. The sharding strategy allows different parts of the model to be loaded and executed on different peer nodes in the Hivemind network.

## Features

- **Model Sharding**: Split large models into layer groups for distributed inference
- **Quantization**: INT4/INT8 quantization for reduced model size
- **CDN Upload**: Upload shards to Cloudflare R2 for fast global distribution
- **Validation**: Verify shard integrity and tensor shapes
- **Modal Deploy**: GPU-accelerated conversion on Modal cloud

## Installation

```bash
# Clone the repository
cd hivemind-models

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### 1. Convert a Model (Local)

```bash
# Using the CLI
python -m src.cli.main convert --config configs/tinyllama-1b.yaml

# Or using the script
python scripts/convert_model.py --config configs/tinyllama-1b.yaml

# Dry run to see what would be created
python -m src.cli.main convert --config configs/tinyllama-1b.yaml --dry-run
```

### 2. Convert a Model (Modal GPU)

For faster conversion of large models, use Modal's GPU infrastructure:

```bash
# Install Modal
pip install modal

# Authenticate
modal token new

# Run conversion on GPU
modal run modal_deploy.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### 3. Validate Shards

```bash
python -m src.cli.main validate ./output/tinyllama-1b-q4
```

### 4. Upload to CDN

```bash
# Set up environment variables (see .env.example)
cp .env.example .env
# Edit .env with your R2 credentials

# Upload
python -m src.cli.main upload ./output/tinyllama-1b-q4 --name tinyllama-1b-q4
```

## Configuration

Model configurations are YAML files in the `configs/` directory:

```yaml
model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir: "./output/tinyllama-1b-q4"

# Layer grouping for distributed inference
layer_groups:
  - [0, 3]    # Client layers
  - [4, 10]   # Pool A
  - [11, 17]  # Pool B
  - [18, 21]  # Pool C + Head

quantize: true
quant_bits: 4
dtype: float16
```

### Available Configurations

| Config | Model | Size | Layers |
|--------|-------|------|--------|
| `tinyllama-1b.yaml` | TinyLlama 1.1B | ~600MB Q4 | 22 |
| `llama-2-7b.yaml` | Llama 2 7B | ~3.5GB Q4 | 32 |
| `mistral-7b.yaml` | Mistral 7B | ~3.5GB Q4 | 32 |
| `phi-2.yaml` | Phi-2 | ~1.4GB Q4 | 32 |

## Output Structure

After conversion, the output directory contains:

```
output/tinyllama-1b-q4/
  shard_embed.safetensors       # Embedding layer
  shard_layers_0_3.safetensors  # Layers 0-3
  shard_layers_4_10.safetensors # Layers 4-10
  shard_layers_11_17.safetensors # Layers 11-17
  shard_layers_18_21.safetensors # Layers 18-21
  shard_head.safetensors        # LM head + final norm
  manifest.json                 # Model manifest for client
  tokenizer/                    # Tokenizer files
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
```

## Manifest Format

The `manifest.json` file describes the sharded model:

```json
{
  "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "model_type": "llama",
  "version": "1.0",
  "total_size_bytes": 629145600,
  "quantized": true,
  "quant_bits": 4,
  "dtype": "float16",
  "layer_groups": [[0, 3], [4, 10], [11, 17], [18, 21]],
  "shards": [
    {
      "filename": "shard_embed.safetensors",
      "size_bytes": 16777216,
      "checksum_sha256": "abc123...",
      "tensor_count": 1
    }
    // ... more shards
  ],
  "tokenizer": {
    "path": "tokenizer/",
    "type": "LlamaTokenizerFast"
  },
  "cdn_base_url": "https://models.example.com/tinyllama-1b-q4"
}
```

## Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Required for R2 upload
R2_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY=your_access_key
R2_SECRET_KEY=your_secret_key
R2_BUCKET_NAME=hivemind-models
R2_PUBLIC_URL=https://models.your-domain.com

# Optional: CDN cache management
CF_ZONE_ID=your_zone_id
CF_API_TOKEN=your_api_token
```

## CLI Reference

```bash
# Convert a model
python -m src.cli.main convert --config <config.yaml> [--output <dir>] [--dry-run]

# Upload to R2
python -m src.cli.main upload <model_dir> --name <model_name>

# Validate shards
python -m src.cli.main validate <model_dir> [--no-checksums] [--no-tensors]

# Show config info
python -m src.cli.main info --config <config.yaml>

# List models in R2
python -m src.cli.main list-models
```

## Python API

```python
from src.convert.sharder import ModelSharder, ShardConfig
from src.upload.r2 import R2Uploader, R2Config

# Load config and convert
config = ShardConfig.from_yaml("configs/tinyllama-1b.yaml")
sharder = ModelSharder(config)
result = sharder.shard()

print(f"Created {len(result.shards)} shards, {result.total_size_bytes / 1e6:.1f} MB")

# Upload to R2
r2_config = R2Config.from_env()
uploader = R2Uploader(r2_config)
upload_result = uploader.upload_model("./output/tinyllama-1b-q4", "tinyllama-1b-q4")
```

## Sharding Strategy

The layer grouping strategy optimizes for distributed inference:

1. **Client Layers (0-3)**: Run locally on the user's browser for fast initial processing
2. **Pool A, B, C**: Distributed across peer nodes in the network
3. **Head + Final Layers**: Can run on client or specialized nodes

This allows:
- Parallel processing across multiple peers
- Load balancing based on peer capabilities
- Graceful degradation if peers disconnect

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.35+
- safetensors 0.4+
- GPU recommended for large models (or use Modal)

## License

MIT
