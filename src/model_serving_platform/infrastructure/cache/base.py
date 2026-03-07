"""Cache abstractions used by infrastructure client integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Represent one deserialised cache entry payload and expiry timestamp.

    Cache clients use this value to pass stored payloads through a typed
    boundary before converting them into domain-specific result structures.
    Parameters: payload and expires_at_unix_seconds describe one cache record.
    """

    payload: dict[str, object]
    expires_at_unix_seconds: float


class CacheStore(Protocol):
    """Define cache operations required by enrichment client wrappers.

    The protocol keeps cache backend details out of enrichment logic so file
    and alternative backends can be swapped without changing client code.
    Parameters: implementations follow these method signatures exactly.
    """

    def get(self, cache_key: str) -> CacheEntry | None:
        """Return cache entry for key when present and not expired.

        Implementations should return None when key is absent or record has
        expired according to configured cache retention policy.
        Parameters: cache_key is deterministic key from client logic.
        """

    def set(self, cache_key: str, payload: dict[str, object]) -> None:
        """Persist one cache payload with backend-specific expiry metadata.

        Implementations apply configured TTL and serialise payload safely for
        subsequent deterministic lookup by identical cache keys.
        Parameters: cache_key identifies record and payload stores value.
        """
