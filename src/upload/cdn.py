"""
CDN cache management utilities.

Manage Cloudflare CDN cache for model files.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass
class CDNConfig:
    """Configuration for Cloudflare CDN."""

    zone_id: str
    api_token: str
    base_url: str

    @classmethod
    def from_env(cls) -> "CDNConfig":
        """Create config from environment variables."""
        zone_id = os.environ.get("CF_ZONE_ID", "")
        api_token = os.environ.get("CF_API_TOKEN", "")
        base_url = os.environ.get("CDN_BASE_URL", "")

        if not all([zone_id, api_token, base_url]):
            raise ValueError(
                "Missing required CDN environment variables. "
                "Set CF_ZONE_ID, CF_API_TOKEN, CDN_BASE_URL"
            )

        return cls(
            zone_id=zone_id,
            api_token=api_token,
            base_url=base_url,
        )


@dataclass
class PurgeResult:
    """Result of a cache purge operation."""

    success: bool
    purged_urls: list[str]
    message: str


class CDNManager:
    """
    Manage Cloudflare CDN cache.

    Provides utilities for cache purging and management.
    """

    CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"

    def __init__(self, config: CDNConfig):
        """
        Initialize the CDN manager.

        Args:
            config: CDN configuration
        """
        self.config = config

    def _get_headers(self) -> dict[str, str]:
        """Get API headers."""
        return {
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json",
        }

    def purge_urls(self, urls: list[str]) -> PurgeResult:
        """
        Purge specific URLs from the cache.

        Args:
            urls: List of URLs to purge

        Returns:
            PurgeResult
        """
        if not urls:
            return PurgeResult(
                success=True,
                purged_urls=[],
                message="No URLs to purge",
            )

        endpoint = f"{self.CLOUDFLARE_API_BASE}/zones/{self.config.zone_id}/purge_cache"

        # Cloudflare limits to 30 URLs per request
        batch_size = 30
        all_purged: list[str] = []
        errors: list[str] = []

        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]

            try:
                response = requests.post(
                    endpoint,
                    headers=self._get_headers(),
                    json={"files": batch},
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        all_purged.extend(batch)
                        logger.info(f"Purged {len(batch)} URLs")
                    else:
                        errors.extend(result.get("errors", []))
                else:
                    errors.append(f"HTTP {response.status_code}: {response.text}")

            except Exception as e:
                errors.append(str(e))

        if errors:
            return PurgeResult(
                success=False,
                purged_urls=all_purged,
                message=f"Partial purge with errors: {errors}",
            )

        return PurgeResult(
            success=True,
            purged_urls=all_purged,
            message=f"Successfully purged {len(all_purged)} URLs",
        )

    def purge_model(self, model_name: str) -> PurgeResult:
        """
        Purge all cached files for a model.

        Args:
            model_name: Name of the model

        Returns:
            PurgeResult
        """
        # Construct URLs for common model files
        base = self.config.base_url.rstrip("/")
        urls = [
            f"{base}/models/{model_name}/manifest.json",
            # Purge the model directory itself
            f"{base}/models/{model_name}/",
        ]

        logger.info(f"Purging cache for model: {model_name}")
        return self.purge_urls(urls)

    def purge_prefix(self, prefix: str) -> PurgeResult:
        """
        Purge all URLs with a given prefix.

        Note: This uses Cloudflare's prefix purge which requires an Enterprise plan.
        For non-Enterprise, use purge_urls with specific URLs.

        Args:
            prefix: URL prefix to purge

        Returns:
            PurgeResult
        """
        endpoint = f"{self.CLOUDFLARE_API_BASE}/zones/{self.config.zone_id}/purge_cache"

        full_prefix = f"{self.config.base_url.rstrip('/')}/{prefix.lstrip('/')}"

        try:
            response = requests.post(
                endpoint,
                headers=self._get_headers(),
                json={"prefixes": [full_prefix]},
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    return PurgeResult(
                        success=True,
                        purged_urls=[full_prefix],
                        message=f"Purged prefix: {full_prefix}",
                    )
                else:
                    return PurgeResult(
                        success=False,
                        purged_urls=[],
                        message=f"Purge failed: {result.get('errors')}",
                    )
            else:
                return PurgeResult(
                    success=False,
                    purged_urls=[],
                    message=f"HTTP {response.status_code}: {response.text}",
                )

        except Exception as e:
            return PurgeResult(
                success=False,
                purged_urls=[],
                message=f"Request failed: {e}",
            )

    def purge_everything(self) -> PurgeResult:
        """
        Purge the entire cache for the zone.

        Use with caution!

        Returns:
            PurgeResult
        """
        endpoint = f"{self.CLOUDFLARE_API_BASE}/zones/{self.config.zone_id}/purge_cache"

        logger.warning("Purging entire cache!")

        try:
            response = requests.post(
                endpoint,
                headers=self._get_headers(),
                json={"purge_everything": True},
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    return PurgeResult(
                        success=True,
                        purged_urls=["*"],
                        message="Entire cache purged",
                    )

            return PurgeResult(
                success=False,
                purged_urls=[],
                message=f"Purge failed: {response.text}",
            )

        except Exception as e:
            return PurgeResult(
                success=False,
                purged_urls=[],
                message=f"Request failed: {e}",
            )

    def get_cache_status(self, url: str) -> dict[str, Any]:
        """
        Check the cache status of a URL by making a HEAD request.

        Args:
            url: URL to check

        Returns:
            Dictionary with cache headers
        """
        try:
            response = requests.head(url, timeout=10)

            return {
                "url": url,
                "status_code": response.status_code,
                "cf_cache_status": response.headers.get("CF-Cache-Status"),
                "cache_control": response.headers.get("Cache-Control"),
                "age": response.headers.get("Age"),
                "content_length": response.headers.get("Content-Length"),
            }

        except Exception as e:
            return {
                "url": url,
                "error": str(e),
            }

    def warmup_model(self, model_name: str, shard_files: list[str]) -> list[dict[str, Any]]:
        """
        Warm up the cache for a model by requesting all files.

        Args:
            model_name: Name of the model
            shard_files: List of shard filenames

        Returns:
            List of cache status results
        """
        base = self.config.base_url.rstrip("/")
        results = []

        # Request manifest first
        manifest_url = f"{base}/models/{model_name}/manifest.json"
        results.append(self.get_cache_status(manifest_url))

        # Request each shard
        for shard_file in shard_files:
            url = f"{base}/models/{model_name}/{shard_file}"
            # Just do a HEAD request to trigger caching
            status = self.get_cache_status(url)
            results.append(status)
            logger.debug(f"Warmed up {shard_file}: {status.get('cf_cache_status')}")

        cached_count = sum(
            1 for r in results
            if r.get("cf_cache_status") in ("HIT", "STALE", "REVALIDATED")
        )
        logger.info(f"Cache warmup complete: {cached_count}/{len(results)} cached")

        return results


def create_cdn_manager_from_env() -> CDNManager:
    """Create a CDNManager from environment variables."""
    config = CDNConfig.from_env()
    return CDNManager(config)
