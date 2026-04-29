#!/usr/bin/env python3
"""
Upload sharded model to Cloudflare R2.

Usage:
    python scripts/upload_model.py ./output --name tinyllama-1b-q4
    python scripts/upload_model.py ./output --name tinyllama-1b-q4 --purge-cache
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload sharded model to Cloudflare R2"
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing sharded model",
    )
    parser.add_argument(
        "--name",
        "-n",
        required=True,
        help="Name for the model in CDN",
    )
    parser.add_argument(
        "--purge-cache",
        action="store_true",
        help="Purge CDN cache after upload",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=".env",
        help="Path to .env file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
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

    # Load environment variables
    if args.env_file.exists():
        load_dotenv(args.env_file)
        logger.info(f"Loaded environment from {args.env_file}")
    else:
        load_dotenv()

    if not args.model_dir.exists():
        logger.error(f"Model directory not found: {args.model_dir}")
        return 1

    # Import after loading env
    from src.upload.r2 import create_uploader_from_env

    try:
        uploader = create_uploader_from_env()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("\nSet the following environment variables:")
        logger.error("  - R2_ACCOUNT_ID")
        logger.error("  - R2_ACCESS_KEY")
        logger.error("  - R2_SECRET_KEY")
        logger.error("  - R2_BUCKET_NAME")
        logger.error("  - R2_PUBLIC_URL (optional)")
        return 1

    if args.dry_run:
        logger.info("DRY RUN - showing files that would be uploaded")

        files = list(args.model_dir.rglob("*"))
        files = [f for f in files if f.is_file()]

        total_size = sum(f.stat().st_size for f in files)

        print(f"\nWould upload {len(files)} files to models/{args.name}/")
        print(f"Total size: {total_size / (1024**2):.2f} MB")
        print("\nFiles:")
        for f in sorted(files):
            rel = f.relative_to(args.model_dir)
            size = f.stat().st_size / (1024**2)
            print(f"  - {rel}: {size:.2f} MB")

        return 0

    logger.info(f"Uploading model to R2")
    logger.info(f"Source: {args.model_dir}")
    logger.info(f"Destination: models/{args.name}")

    result = uploader.upload_model(args.model_dir, args.name)

    logger.info(f"Upload complete!")
    logger.info(f"Files uploaded: {result.successful}/{result.total_files}")
    logger.info(f"Total size: {result.total_bytes / (1024**2):.2f} MB")

    if result.cdn_base_url:
        logger.info(f"CDN URL: {result.cdn_base_url}")

    if result.failed > 0:
        logger.warning(f"{result.failed} files failed to upload")
        for r in result.results:
            if not r.success:
                logger.warning(f"  - {r.key}: {r.message}")

    # Purge cache if requested
    if args.purge_cache and result.successful > 0:
        logger.info("\nPurging CDN cache...")

        try:
            from src.upload.cdn import create_cdn_manager_from_env
            cdn = create_cdn_manager_from_env()
            purge_result = cdn.purge_model(args.name)

            if purge_result.success:
                logger.info(f"Cache purged: {purge_result.message}")
            else:
                logger.warning(f"Cache purge failed: {purge_result.message}")

        except ValueError as e:
            logger.warning(f"Could not purge cache (CDN not configured): {e}")

    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
