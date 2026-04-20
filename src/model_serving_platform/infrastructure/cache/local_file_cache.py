"""Disk-backed cache: one JSON file per logical key, with expiry timestamps.

`LocalFileCacheStore` implements the `CacheStore` protocol from `base.py`. Each
`set` writes a small JSON object to a folder configured at startup (for example
the service cache path from settings). The file holds the caller's `payload`
dict plus `expires_at_unix_seconds`, computed as "now plus configured lifetime".

`get` reads that file, parses JSON, and compares current time to the stored
expiry. If the file is missing, or time is past expiry, `get` returns `None`.
Expired files are deleted on read so stale entries do not accumulate silently.

Keys are turned into filenames with SHA-256 so arbitrary string keys stay safe
for the filesystem (no slashes or odd characters in the name). The mapping
from key to filename is fixed: the same key always maps to the same file.

`current_time_provider` exists so tests can freeze or advance a fake clock without
waiting for real time to pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import time
from typing import Callable, cast

from model_serving_platform.infrastructure.cache.base import CacheEntry, CacheStore


class LocalFileCacheStore(CacheStore):
    """Concrete `CacheStore` that persists each entry as one JSON file on disk.

    Suitable for single-process or single-node setups where a shared folder is
    enough. Not a distributed cache: other machines do not see these files unless
    they share the same filesystem.
    """

    def __init__(
        self,
        cache_directory_path: Path,
        ttl_seconds: float,
        current_time_provider: Callable[[], float] | None = None,
    ) -> None:
        """Remember cache folder, entry lifetime in seconds, and optional clock.

        The directory is created immediately so later `set` calls do not fail
        only because the folder was never created.
        """

        self._cache_directory_path = cache_directory_path
        self._ttl_seconds = ttl_seconds
        self._current_time_provider = current_time_provider or time
        # Ensure writes succeed without a separate mkdir step on first use.
        self._cache_directory_path.mkdir(parents=True, exist_ok=True)

    def get(self, cache_key: str) -> CacheEntry | None:
        """Return a live `CacheEntry` for `cache_key`, or None if absent or expired."""

        cache_file_path = self._get_cache_file_path(cache_key=cache_key)
        if not cache_file_path.exists():
            return None
        cached_entry_payload = json.loads(cache_file_path.read_text(encoding="utf-8"))
        expires_at_unix_seconds = float(cached_entry_payload["expires_at_unix_seconds"])
        # Drop expired files on read so disk state matches logical cache state.
        if self._current_time_provider() >= expires_at_unix_seconds:
            cache_file_path.unlink(missing_ok=True)
            return None
        cached_value_payload = cast(dict[str, object], cached_entry_payload["payload"])
        return CacheEntry(
            payload=cached_value_payload,
            expires_at_unix_seconds=expires_at_unix_seconds,
        )

    def set(self, cache_key: str, payload: dict[str, object]) -> None:
        """Write `payload` to disk and set expiry to now plus configured lifetime."""

        cache_file_path = self._get_cache_file_path(cache_key=cache_key)
        # Overwrite whole file so each successful lookup refreshes expiry and value.
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
        """Map string key to a single file path under the cache directory."""

        # Hash avoids illegal filename characters and keeps paths predictable.
        cache_file_name = (
            hashlib.sha256(cache_key.encode("utf-8")).hexdigest() + ".json"
        )
        return self._cache_directory_path / cache_file_name
