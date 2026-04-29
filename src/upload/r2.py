"""
Cloudflare R2 upload utilities.

S3-compatible upload using boto3 for Cloudflare R2 storage.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class R2Config:
    """Configuration for Cloudflare R2."""

    account_id: str
    access_key: str
    secret_key: str
    bucket_name: str
    public_url: str | None = None

    @classmethod
    def from_env(cls) -> "R2Config":
        """Create config from environment variables."""
        account_id = os.environ.get("R2_ACCOUNT_ID", "")
        access_key = os.environ.get("R2_ACCESS_KEY", "")
        secret_key = os.environ.get("R2_SECRET_KEY", "")
        bucket_name = os.environ.get("R2_BUCKET_NAME", "")
        public_url = os.environ.get("R2_PUBLIC_URL")

        if not all([account_id, access_key, secret_key, bucket_name]):
            raise ValueError(
                "Missing required R2 environment variables. "
                "Set R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET_NAME"
            )

        return cls(
            account_id=account_id,
            access_key=access_key,
            secret_key=secret_key,
            bucket_name=bucket_name,
            public_url=public_url,
        )

    @property
    def endpoint_url(self) -> str:
        """Get the R2 endpoint URL."""
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


@dataclass
class UploadResult:
    """Result of a file upload."""

    success: bool
    key: str
    size_bytes: int
    checksum_md5: str
    cdn_url: str | None
    message: str
    etag: str | None = None


@dataclass
class BatchUploadResult:
    """Result of batch upload operation."""

    total_files: int
    successful: int
    failed: int
    total_bytes: int
    results: list[UploadResult]
    cdn_base_url: str | None


class R2Uploader:
    """
    Upload files to Cloudflare R2.

    Uses boto3 with S3-compatible API.
    """

    # Cache headers for different file types
    CACHE_HEADERS = {
        ".safetensors": {"CacheControl": "public, max-age=31536000, immutable"},
        ".json": {"CacheControl": "public, max-age=3600"},
        ".txt": {"CacheControl": "public, max-age=3600"},
        ".model": {"CacheControl": "public, max-age=31536000, immutable"},
    }

    # Content types
    CONTENT_TYPES = {
        ".safetensors": "application/octet-stream",
        ".json": "application/json",
        ".txt": "text/plain",
        ".model": "application/octet-stream",
    }

    def __init__(self, config: R2Config):
        """
        Initialize the R2 uploader.

        Args:
            config: R2 configuration
        """
        self.config = config
        self._client = None

    @property
    def client(self) -> Any:
        """Get or create boto3 S3 client."""
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 required. Install with: pip install boto3")

            self._client = boto3.client(
                "s3",
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name="auto",
            )
        return self._client

    def _compute_md5(self, file_path: Path) -> str:
        """Compute MD5 checksum of a file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_cache_headers(self, file_path: Path) -> dict[str, str]:
        """Get cache headers for a file type."""
        suffix = file_path.suffix.lower()
        return self.CACHE_HEADERS.get(suffix, {"CacheControl": "public, max-age=86400"})

    def _get_content_type(self, file_path: Path) -> str:
        """Get content type for a file."""
        suffix = file_path.suffix.lower()
        return self.CONTENT_TYPES.get(suffix, "application/octet-stream")

    def upload_file(
        self,
        file_path: str | Path,
        key: str | None = None,
        prefix: str = "",
        progress_callback: Callable[[int], None] | None = None,
    ) -> UploadResult:
        """
        Upload a single file to R2.

        Args:
            file_path: Path to the file to upload
            key: S3 key (defaults to filename)
            prefix: Key prefix (e.g., "models/tinyllama/")
            progress_callback: Optional callback for progress updates

        Returns:
            UploadResult
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return UploadResult(
                success=False,
                key="",
                size_bytes=0,
                checksum_md5="",
                cdn_url=None,
                message=f"File not found: {file_path}",
            )

        # Determine key
        if key is None:
            key = file_path.name
        if prefix:
            key = f"{prefix.rstrip('/')}/{key}"

        file_size = file_path.stat().st_size
        checksum = self._compute_md5(file_path)
        content_type = self._get_content_type(file_path)
        cache_headers = self._get_cache_headers(file_path)

        logger.info(f"Uploading {file_path.name} ({file_size / (1024*1024):.2f} MB) to {key}")

        try:
            extra_args = {
                "ContentType": content_type,
                **cache_headers,
            }

            # Use multipart upload for large files
            if file_size > 100 * 1024 * 1024:  # 100 MB
                self._multipart_upload(file_path, key, extra_args, progress_callback)
            else:
                # Simple upload
                with open(file_path, "rb") as f:
                    self.client.put_object(
                        Bucket=self.config.bucket_name,
                        Key=key,
                        Body=f,
                        **extra_args,
                    )

            # Get CDN URL
            cdn_url = None
            if self.config.public_url:
                cdn_url = f"{self.config.public_url.rstrip('/')}/{key}"

            return UploadResult(
                success=True,
                key=key,
                size_bytes=file_size,
                checksum_md5=checksum,
                cdn_url=cdn_url,
                message="Upload successful",
            )

        except Exception as e:
            logger.error(f"Upload failed for {file_path.name}: {e}")
            return UploadResult(
                success=False,
                key=key,
                size_bytes=file_size,
                checksum_md5=checksum,
                cdn_url=None,
                message=f"Upload failed: {e}",
            )

    def _multipart_upload(
        self,
        file_path: Path,
        key: str,
        extra_args: dict[str, str],
        progress_callback: Callable[[int], None] | None,
    ) -> None:
        """Perform multipart upload for large files."""
        try:
            from boto3.s3.transfer import TransferConfig
        except ImportError:
            raise ImportError("boto3 required for multipart upload")

        config = TransferConfig(
            multipart_threshold=100 * 1024 * 1024,
            max_concurrency=10,
            multipart_chunksize=100 * 1024 * 1024,
        )

        callback = None
        if progress_callback:
            class ProgressCallback:
                def __init__(self, cb: Callable[[int], None]):
                    self.cb = cb
                    self.total = 0

                def __call__(self, bytes_transferred: int):
                    self.total += bytes_transferred
                    self.cb(self.total)

            callback = ProgressCallback(progress_callback)

        self.client.upload_file(
            str(file_path),
            self.config.bucket_name,
            key,
            ExtraArgs=extra_args,
            Config=config,
            Callback=callback,
        )

    def upload_directory(
        self,
        directory: str | Path,
        prefix: str = "",
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> BatchUploadResult:
        """
        Upload all files in a directory.

        Args:
            directory: Directory to upload
            prefix: Key prefix for all files
            include_patterns: Glob patterns to include (e.g., ["*.safetensors"])
            exclude_patterns: Glob patterns to exclude

        Returns:
            BatchUploadResult
        """
        directory = Path(directory)

        if not directory.exists():
            return BatchUploadResult(
                total_files=0,
                successful=0,
                failed=0,
                total_bytes=0,
                results=[],
                cdn_base_url=None,
            )

        # Collect files to upload
        files_to_upload: list[Path] = []

        if include_patterns:
            for pattern in include_patterns:
                files_to_upload.extend(directory.glob(pattern))
        else:
            files_to_upload = list(directory.rglob("*"))

        # Filter out directories and excluded patterns
        files_to_upload = [f for f in files_to_upload if f.is_file()]

        if exclude_patterns:
            for pattern in exclude_patterns:
                excluded = set(directory.glob(pattern))
                files_to_upload = [f for f in files_to_upload if f not in excluded]

        results: list[UploadResult] = []
        total_bytes = 0

        for file_path in tqdm(files_to_upload, desc="Uploading"):
            # Compute relative key
            relative_path = file_path.relative_to(directory)
            key = str(relative_path).replace("\\", "/")

            result = self.upload_file(file_path, key=key, prefix=prefix)
            results.append(result)

            if result.success:
                total_bytes += result.size_bytes

        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        cdn_base_url = None
        if self.config.public_url and prefix:
            cdn_base_url = f"{self.config.public_url.rstrip('/')}/{prefix.rstrip('/')}"

        return BatchUploadResult(
            total_files=len(results),
            successful=successful,
            failed=failed,
            total_bytes=total_bytes,
            results=results,
            cdn_base_url=cdn_base_url,
        )

    def upload_model(
        self,
        model_dir: str | Path,
        model_name: str,
    ) -> BatchUploadResult:
        """
        Upload a complete sharded model.

        Args:
            model_dir: Directory containing shards and manifest
            model_name: Name for the model in the CDN

        Returns:
            BatchUploadResult
        """
        return self.upload_directory(
            model_dir,
            prefix=f"models/{model_name}",
            include_patterns=["*.safetensors", "*.json", "tokenizer/*"],
        )

    def list_objects(self, prefix: str = "") -> list[dict[str, Any]]:
        """
        List objects in the bucket.

        Args:
            prefix: Key prefix to filter by

        Returns:
            List of object metadata
        """
        objects = []
        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.config.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                objects.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                    "etag": obj["ETag"],
                })

        return objects

    def delete_object(self, key: str) -> bool:
        """
        Delete an object from the bucket.

        Args:
            key: Object key

        Returns:
            True if successful
        """
        try:
            self.client.delete_object(Bucket=self.config.bucket_name, Key=key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            return False

    def get_object_url(self, key: str) -> str | None:
        """
        Get the public URL for an object.

        Args:
            key: Object key

        Returns:
            Public URL or None if no public URL configured
        """
        if self.config.public_url:
            return f"{self.config.public_url.rstrip('/')}/{key}"
        return None


def create_uploader_from_env() -> R2Uploader:
    """Create an R2Uploader from environment variables."""
    config = R2Config.from_env()
    return R2Uploader(config)
