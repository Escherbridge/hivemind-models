#!/usr/bin/env python3
"""
Batch convert multiple HuggingFace models to sharded format.

Usage:
    python scripts/batch_convert.py --configs configs/*.yaml
    python scripts/batch_convert.py --configs configs/tinyllama-1b.yaml configs/mistral-7b.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from glob import glob
from pathlib import Path
from typing import Any

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
        description="Batch convert HuggingFace models to sharded safetensors"
    )
    parser.add_argument(
        "--configs",
        "-c",
        nargs="+",
        required=True,
        help="Paths to YAML configuration files (supports glob patterns)",
    )
    parser.add_argument(
        "--output-base",
        "-o",
        type=Path,
        default=None,
        help="Base output directory (model name will be appended)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without converting",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing if a model fails",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def expand_config_paths(patterns: list[str]) -> list[Path]:
    """Expand glob patterns to actual paths."""
    paths: list[Path] = []
    for pattern in patterns:
        expanded = glob(pattern, recursive=True)
        paths.extend(Path(p) for p in expanded)
    return sorted(set(paths))


def convert_single_model(
    config_path: Path,
    output_base: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Convert a single model and return result info."""
    result = {
        "config": str(config_path),
        "success": False,
        "error": None,
        "shards": 0,
        "size_mb": 0,
    }

    try:
        config = ShardConfig.from_yaml(config_path)

        if output_base:
            # Use model name as subdirectory
            model_name = config.model_id.replace("/", "_")
            config.output_dir = output_base / model_name

        result["model_id"] = config.model_id
        result["output_dir"] = str(config.output_dir)

        sharder = ModelSharder(config)

        if dry_run:
            plan = sharder.dry_run()
            result["success"] = True
            result["shards"] = len(plan["planned_shards"])
            result["dry_run"] = True
            return result

        conversion_result = sharder.shard()
        result["success"] = True
        result["shards"] = len(conversion_result.shards)
        result["size_mb"] = conversion_result.total_size_bytes / (1024**2)

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Failed to convert {config_path}: {e}")

    return result


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Expand config paths
    config_paths = expand_config_paths(args.configs)

    if not config_paths:
        logger.error("No configuration files found")
        return 1

    logger.info(f"Found {len(config_paths)} configuration files")

    results: list[dict[str, Any]] = []
    failed_count = 0

    for i, config_path in enumerate(config_paths, 1):
        logger.info(f"\n[{i}/{len(config_paths)}] Processing {config_path}")

        result = convert_single_model(
            config_path,
            args.output_base,
            args.dry_run,
        )
        results.append(result)

        if not result["success"]:
            failed_count += 1
            if not args.continue_on_error:
                logger.error("Stopping due to error (use --continue-on-error to continue)")
                break

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH CONVERSION SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nTotal: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\nSuccessful conversions:")
        for r in successful:
            if r.get("dry_run"):
                print(f"  - {r.get('model_id', r['config'])}: {r['shards']} shards (dry run)")
            else:
                print(f"  - {r.get('model_id', r['config'])}: {r['shards']} shards, {r['size_mb']:.2f} MB")

    if failed:
        print("\nFailed conversions:")
        for r in failed:
            print(f"  - {r['config']}: {r['error']}")

    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
