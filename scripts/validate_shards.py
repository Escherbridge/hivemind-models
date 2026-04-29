#!/usr/bin/env python3
"""
Validate sharded model files.

Usage:
    python scripts/validate_shards.py ./output
    python scripts/validate_shards.py ./output --no-checksums
    python scripts/validate_shards.py ./output --reconstruction-test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.convert.validation import (
    validate_shards,
    test_model_reconstruction,
    print_validation_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate sharded model files"
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing sharded model",
    )
    parser.add_argument(
        "--no-checksums",
        action="store_true",
        help="Skip checksum verification (faster)",
    )
    parser.add_argument(
        "--no-tensors",
        action="store_true",
        help="Skip tensor verification (faster)",
    )
    parser.add_argument(
        "--reconstruction-test",
        action="store_true",
        help="Test model reconstruction from shards",
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

    if not args.model_dir.exists():
        logger.error(f"Model directory not found: {args.model_dir}")
        return 1

    logger.info(f"Validating model at {args.model_dir}")

    # Run main validation
    report = validate_shards(
        args.model_dir,
        verify_checksums=not args.no_checksums,
        verify_tensors=not args.no_tensors,
    )

    print_validation_report(report)

    # Run reconstruction test if requested
    if args.reconstruction_test:
        logger.info("\nRunning reconstruction test...")
        recon_result = test_model_reconstruction(args.model_dir)

        if recon_result.passed:
            print(f"[OK] Reconstruction test: {recon_result.message}")
        else:
            print(f"[FAIL] Reconstruction test: {recon_result.message}")
            return 1

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
