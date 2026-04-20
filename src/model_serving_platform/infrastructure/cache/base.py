"""Shared cache types for code that calls external enrichment services.

Enrichment clients (HTTP lookups, description resolution, and similar) can avoid
repeated network calls by storing previous responses in a cache. This module does
not implement disk or memory storage. It only defines:

- `CacheEntry`: one cached record after read (payload plus when it expires).
- `CacheStore`: the minimal `get` and `set` operations any backend must provide.

Wrappers depend on `CacheStore` as a protocol, so the real storage can be a local
JSON file directory (`local_file_cache.py`) or another implementation later
without rewriting the enrichment client logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One item returned from a cache read, ready for the caller to interpret.

    `payload` is a plain dictionary (already parsed from storage). Callers map
    it into their own result types. `expires_at_unix_seconds` is a Unix timestamp
    after which the entry must be treated as invalid, even if still on disk.
    """

    # Serializable blob as produced by the client that wrote the cache entry.
    payload: dict[str, object]
    # Wall-clock expiry as seconds since the Unix epoch; compared at read time.
    expires_at_unix_seconds: float


class CacheStore(Protocol):
    """Contract for anything that can store and retrieve cache records by string key.

    A protocol in Python means: any class with matching `get` and `set` methods
    satisfies this type. Enrichment code depends on behaviour, not on one concrete
    class name.
    """

    def get(self, cache_key: str) -> CacheEntry | None:
        """Load one entry for `cache_key`, or return None if missing or expired.

        The key must be stable: the same logical request must always produce the
        same key string so a later `set` and `get` pair match.
        """

    def set(self, cache_key: str, payload: dict[str, object]) -> None:
        """Write `payload` under `cache_key` and store when this entry expires.

        Each backend has a configured lifetime for entries (time to live). After
        that duration, `get` must treat the record as absent. Callers only pass
        a key and a dictionary; persistence format is implementation-specific.
        """
