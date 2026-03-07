"""Local file-backed cache implementation with TTL expiry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import time
from typing import Callable, cast

from model_serving_platform.infrastructure.cache.base import CacheEntry, CacheStore


class LocalFileCacheStore(CacheStore):
    """Store cache entries as JSON files under configured directory path.

    The backend is intentionally simple for v1 because local-first operation
    requires deterministic behaviour and minimal moving infrastructure parts.
    Parameters: cache path and TTL values are provided by settings.
    """

    def __init__(
        self,
        cache_directory_path: Path,
        ttl_seconds: float,
        current_time_provider: Callable[[], float] | None = None,
    ) -> None:
        """Initialise local file cache backend and expiry configuration.

        Cache directory is created eagerly to avoid runtime write surprises,
        and optional clock injection is used for deterministic expiry tests.
        Parameters: paths and TTL values define file cache behaviour.
        """

        self._cache_directory_path = cache_directory_path
        self._ttl_seconds = ttl_seconds
        self._current_time_provider = current_time_provider or time
        self._cache_directory_path.mkdir(parents=True, exist_ok=True)

    def get(self, cache_key: str) -> CacheEntry | None:
        """Return cached payload when present and still within TTL window.

        Expired entries are removed immediately so later reads do not process
        stale data and cache behaviour remains deterministic across requests.
        Parameters: cache_key identifies one cached record on disk.
        """

        cache_file_path = self._get_cache_file_path(cache_key=cache_key)
        if not cache_file_path.exists():
            return None
        cached_entry_payload = json.loads(cache_file_path.read_text(encoding="utf-8"))
        expires_at_unix_seconds = float(cached_entry_payload["expires_at_unix_seconds"])
        if self._current_time_provider() >= expires_at_unix_seconds:
            cache_file_path.unlink(missing_ok=True)
            return None
        cached_value_payload = cast(dict[str, object], cached_entry_payload["payload"])
        return CacheEntry(
            payload=cached_value_payload,
            expires_at_unix_seconds=expires_at_unix_seconds,
        )

    def set(self, cache_key: str, payload: dict[str, object]) -> None:
        """Write payload to deterministic file path with expiry timestamp.

        Cache writes always overwrite existing records to ensure latest value
        and expiry are applied after each successful external lookup call.
        Parameters: cache_key and payload define one cache record write.
        """

        cache_file_path = self._get_cache_file_path(cache_key=cache_key)
        cache_file_path.write_text(
            json.dumps(
                {
                    "expires_at_unix_seconds": self._current_time_provider()
                    + self._ttl_seconds,
                    "payload": payload,
                }
            ),
            encoding="utf-8",
        )

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Build deterministic file path for one cache key string.

        Hashing is used to keep filenames filesystem-safe and consistent while
        preserving deterministic mapping from logical key to storage location.
        Parameters: cache_key is the logical identifier for cached value.
        """

        cache_file_name = (
            hashlib.sha256(cache_key.encode("utf-8")).hexdigest() + ".json"
        )
        return self._cache_directory_path / cache_file_name
