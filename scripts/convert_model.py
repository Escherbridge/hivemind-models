#!/usr/bin/env python3
"""
Convert a single HuggingFace model to sharded safetensors format.

Usage:
    python scripts/convert_model.py --config configs/tinyllama-1b.yaml
    python scripts/convert_model.py --config configs/tinyllama-1b.yaml --output ./my_output
    python scripts/convert_model.py --config configs/tinyllama-1b.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.convert.sharder import ShardConfig, ModelSharder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to sharded safetensors"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without converting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1

    logger.info(f"Loading configuration from {args.config}")

    try:
        config = ShardConfig.from_yaml(args.config)

        if args.output:
            config.output_dir = args.output

        logger.info(f"Model: {config.model_id}")
        logger.info(f"Output: {config.output_dir}")
        logger.info(f"Layer groups: {config.layer_groups}")
        logger.info(f"Quantize: {config.quantize} ({config.quant_bits}-bit)")

        sharder = ModelSharder(config)

        if args.dry_run:
            logger.info("DRY RUN - showing planned shards")
            plan = sharder.dry_run()

            print("\nPlanned shards:")
            for shard in plan["planned_shards"]:
                print(f"  - {shard['filename']}: {shard['description']}")

            return 0

        logger.info("Starting conversion...")
        result = sharder.shard()

        logger.info(f"Conversion complete!")
        logger.info(f"Total shards: {len(result.shards)}")
        logger.info(f"Total size: {result.total_size_bytes / (1024**2):.2f} MB")
        logger.info(f"Manifest: {result.manifest_path}")

        print("\nGenerated shards:")
        for shard in result.shards:
            print(f"  - {shard.filename}: {shard.size_bytes / (1024**2):.2f} MB ({shard.tensor_count} tensors)")

        return 0

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
