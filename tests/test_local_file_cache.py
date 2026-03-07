"""Unit tests for local file cache backend behaviour."""

from __future__ import annotations

from pathlib import Path

from model_serving_platform.infrastructure.cache.local_file_cache import (
    LocalFileCacheStore,
)


class MutableClock:
    """Provide mutable time source for deterministic TTL expiry tests.

    File cache expiry logic depends on current time, so tests use this helper
    to advance time without real sleeps or non-deterministic wall clock usage.
    Parameters: initial_unix_seconds is the initial time value.
    """

    def __init__(self, initial_unix_seconds: float) -> None:
        """Initialise mutable clock with starting unix timestamp value.

        Tests call `advance` to simulate passing time and verify cache expiry
        behaviour at specific deterministic points around TTL boundaries.
        Parameters: initial_unix_seconds defines first returned value.
        """

        self._current_unix_seconds = initial_unix_seconds

    def now(self) -> float:
        """Return current mutable unix timestamp for cache operations.

        Cache backend calls this value for read and write decisions, allowing
        tests to assert miss, hit, and expiry behaviour deterministically.
        Parameters: none.
        """

        return self._current_unix_seconds

    def advance(self, seconds: float) -> None:
        """Move mutable clock forward by provided number of seconds.

        Advancing time allows tests to force cache entry expiry without delay
        and verify that expired files are removed on subsequent lookups.
        Parameters: seconds is added to current unix timestamp value.
        """

        self._current_unix_seconds += seconds


def test_local_file_cache_reports_miss_for_unknown_key(tmp_path: Path) -> None:
    """Verify cache get returns None when key has no stored entry file.

    This test covers miss behaviour for fresh cache directories and confirms
    get operations are safe for keys that have never been written.
    Parameters: tmp_path is provided by pytest.
    """

    cache_store = LocalFileCacheStore(
        cache_directory_path=tmp_path / "cache",
        ttl_seconds=60.0,
    )

    cache_entry = cache_store.get(cache_key="missing-key")

    assert cache_entry is None


def test_local_file_cache_returns_hit_before_expiry(tmp_path: Path) -> None:
    """Verify cache set then get returns stored payload within TTL window.

    This test confirms basic hit behaviour and validates payload round-trip
    serialisation for deterministic local cache read and write operations.
    Parameters: tmp_path is provided by pytest.
    """

    mutable_clock = MutableClock(initial_unix_seconds=100.0)
    cache_store = LocalFileCacheStore(
        cache_directory_path=tmp_path / "cache",
        ttl_seconds=60.0,
        current_time_provider=mutable_clock.now,
    )
    cache_store.set(cache_key="description-key", payload={"description": "text"})

    cache_entry = cache_store.get(cache_key="description-key")

    assert cache_entry is not None
    assert cache_entry.payload == {"description": "text"}


def test_local_file_cache_expires_entries_and_deletes_file(tmp_path: Path) -> None:
    """Verify expired entries return miss and are removed from disk.

    Cache expiry is required to prevent stale enrichment values persisting
    indefinitely, so this test checks both miss return and file cleanup.
    Parameters: tmp_path is provided by pytest.
    """

    mutable_clock = MutableClock(initial_unix_seconds=100.0)
    cache_directory_path = tmp_path / "cache"
    cache_store = LocalFileCacheStore(
        cache_directory_path=cache_directory_path,
        ttl_seconds=10.0,
        current_time_provider=mutable_clock.now,
    )
    cache_store.set(cache_key="interaction-key", payload={"partners": ["A"]})
    mutable_clock.advance(seconds=11.0)

    cache_entry = cache_store.get(cache_key="interaction-key")

    assert cache_entry is None
    assert list(cache_directory_path.glob("*.json")) == []
