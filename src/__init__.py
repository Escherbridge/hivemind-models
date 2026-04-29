"""
Hivemind Models - Model sharding and CDN upload pipeline.

This package provides tools for:
- Converting HuggingFace models to sharded safetensors format
- Quantization (INT8, INT4)
- Uploading shards to Cloudflare R2 CDN
- Generating manifests for browser-based inference
"""

__version__ = "0.1.0"
__author__ = "Hivemind Team"

from src.convert.sharder import ModelSharder
from src.upload.r2 import R2Uploader
from src.upload.manifest import ManifestGenerator

__all__ = [
    "ModelSharder",
    "R2Uploader",
    "ManifestGenerator",
]
