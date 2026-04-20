"""Clients that fetch extra entity data from outside this service process.

"In enrichment" here means: look up text descriptions or lists of interaction
partners for an entity name, using HTTP services configured at startup (or report
that those services are not configured). The GraphSAGE runtime calls these
clients when it needs information that is not already in the loaded bundle.

This file defines:

- Result dataclasses (`EntityDescriptionLookupResult`, `InteractionPartnerLookupResult`)
  so every lookup returns both data and an explicit outcome (`success`,
  `not_found`, `failed`, `unavailable`).
- `ExternalEnrichmentClient`, a protocol listing the two lookup methods plus
  `supports_interaction_strategy`, so callers depend on behaviour, not one class.
- `HttpExternalEnrichmentClient`, which performs real GET requests with timeout
  and linear backoff retries, and records metrics when enabled.
- `NoopExternalEnrichmentClient`, which never calls the network and always
  reports `unavailable`, for tests or runs without external URLs.
- `CachingExternalEnrichmentClient`, which wraps another client and stores
  lookup results in a `CacheStore` (for example local JSON files) to avoid
  repeat HTTP calls for the same entity name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha256
from time import sleep
from typing import Literal, Protocol, cast

import httpx

from model_serving_platform.infrastructure.cache.base import CacheStore
from model_serving_platform.infrastructure.metrics import ServiceMetrics

external_enrichment_client_logger = logging.getLogger(
    "model_serving_platform.external_enrichment_client"
)


@dataclass(frozen=True, slots=True)
class EntityDescriptionLookupResult:
    """Outcome of asking an external system for one entity's text description.

    `description` is the text when found; it is None when not found or when
    the call did not succeed. `outcome` tells the runtime which case occurred
    so it can branch without guessing from None alone.
    """

    description: str | None
    outcome: Literal["success", "not_found", "failed", "unavailable"]


@dataclass(frozen=True, slots=True)
class InteractionPartnerLookupResult:
    """Outcome of asking an external system for entities linked by interactions.

    `partner_entity_names` is the list returned when the call succeeds; it is
    empty for not found, failure, or when interaction lookup is not configured.
    `outcome` records which situation applies for the runtime.
    """

    partner_entity_names: list[str]
    outcome: Literal["success", "not_found", "failed", "unavailable"]


class ExternalEnrichmentClient(Protocol):
    """Minimum surface the inference runtime needs for external lookups.

    Any class that implements these three members can be passed in. That allows
    HTTP, no-op, caching wrapper, or test doubles without changing runtime code.
    """

    def lookup_entity_description(
        self, entity_name: str
    ) -> EntityDescriptionLookupResult:
        """Return description plus outcome for one `entity_name` string."""

    def lookup_interaction_partners(
        self, entity_name: str
    ) -> InteractionPartnerLookupResult:
        """Return partner names plus outcome for one source `entity_name`."""

    def supports_interaction_strategy(self) -> bool:
        """True when this client can meaningfully support interaction attachment."""


class HttpExternalEnrichmentClient(ExternalEnrichmentClient):
    """Calls configured HTTP endpoints for description and interaction lookups.

    URLs may be None: then lookups return `unavailable` without sending requests.
    Retries use `sleep` with increasing delay; final failure returns payload None
    and callers map that to `failed` outcomes. Optional `transport` is for tests.
    """

    def __init__(
        self,
        description_lookup_url: str | None,
        interaction_lookup_url: str | None,
        timeout_seconds: float,
        retry_count: int,
        retry_backoff_seconds: float,
        transport: httpx.BaseTransport | None = None,
        service_metrics: ServiceMetrics | None = None,
    ) -> None:
        """Store URLs, timing, retry policy, HTTP client, and optional metrics."""

        self._description_lookup_url = description_lookup_url
        self._interaction_lookup_url = interaction_lookup_url
        self._timeout_seconds = timeout_seconds
        self._retry_count = retry_count
        self._retry_backoff_seconds = retry_backoff_seconds
        self._http_client = httpx.Client(transport=transport)
        self._service_metrics = service_metrics

    def lookup_entity_description(
        self, entity_name: str
    ) -> EntityDescriptionLookupResult:
        """GET description endpoint with `entity_name` query param, or unavailable."""

        # No URL in settings: treat as disabled, not as a network failure.
        if self._description_lookup_url is None:
            self._record_external_lookup(
                operation="entity_description_lookup",
                outcome="unavailable",
            )
            return EntityDescriptionLookupResult(
                description=None,
                outcome="unavailable",
            )
        response_payload = self._request_json_with_retries(
            request_url=self._description_lookup_url,
            query_params={"entity_name": entity_name},
            operation_name="entity_description_lookup",
        )
        if response_payload is None:
            self._record_external_lookup(
                operation="entity_description_lookup",
                outcome="failed",
            )
            return EntityDescriptionLookupResult(
                description=None,
                outcome="failed",
            )
        resolved_description = response_payload.get("description")
        if isinstance(resolved_description, str) and resolved_description != "":
            self._record_external_lookup(
                operation="entity_description_lookup",
                outcome="success",
            )
            return EntityDescriptionLookupResult(
                description=resolved_description,
                outcome="success",
            )
        self._record_external_lookup(
            operation="entity_description_lookup",
            outcome="not_found",
        )
        return EntityDescriptionLookupResult(
            description=None,
            outcome="not_found",
        )

    def lookup_interaction_partners(
        self, entity_name: str
    ) -> InteractionPartnerLookupResult:
        """GET interaction endpoint with `entity_name` query param, or unavailable."""

        if self._interaction_lookup_url is None:
            self._record_external_lookup(
                operation="interaction_partner_lookup",
                outcome="unavailable",
            )
            return InteractionPartnerLookupResult(
                partner_entity_names=[],
                outcome="unavailable",
            )
        response_payload = self._request_json_with_retries(
            request_url=self._interaction_lookup_url,
            query_params={"entity_name": entity_name},
            operation_name="interaction_partner_lookup",
        )
        if response_payload is None:
            self._record_external_lookup(
                operation="interaction_partner_lookup",
                outcome="failed",
            )
            return InteractionPartnerLookupResult(
                partner_entity_names=[],
                outcome="failed",
            )
        raw_partner_names = response_payload.get("partners")
        if not isinstance(raw_partner_names, list):
            self._record_external_lookup(
                operation="interaction_partner_lookup",
                outcome="not_found",
            )
            return InteractionPartnerLookupResult(
                partner_entity_names=[],
                outcome="not_found",
            )
        partner_entity_names = [
            partner_name
            for partner_name in raw_partner_names
            if isinstance(partner_name, str)
        ]
        if len(partner_entity_names) == 0:
            self._record_external_lookup(
                operation="interaction_partner_lookup",
                outcome="not_found",
            )
            return InteractionPartnerLookupResult(
                partner_entity_names=[],
                outcome="not_found",
            )
        self._record_external_lookup(
            operation="interaction_partner_lookup",
            outcome="success",
        )
        return InteractionPartnerLookupResult(
            partner_entity_names=partner_entity_names,
            outcome="success",
        )

    def supports_interaction_strategy(self) -> bool:
        """True when an interaction lookup base URL is set (HTTP client may still fail per call)."""

        return self._interaction_lookup_url is not None

    def _record_external_lookup(self, operation: str, outcome: str) -> None:
        """Emit one external lookup counter when `service_metrics` is not None."""

        if self._service_metrics is None:
            return
        self._service_metrics.increment_external_lookup(
            operation=operation,
            outcome=outcome,
        )

    def _request_json_with_retries(
        self,
        request_url: str,
        query_params: dict[str, str],
        operation_name: str,
    ) -> dict[str, object] | None:
        """GET JSON body or None after all attempts exhausted."""

        # First attempt plus `retry_count` extra tries; backoff grows with attempt.
        total_attempt_count = self._retry_count + 1
        for attempt_index in range(total_attempt_count):
            try:
                response = self._http_client.get(
                    request_url,
                    params=query_params,
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                external_enrichment_client_logger.info(
                    "external_enrichment_call_success",
                    extra={
                        "operation_name": operation_name,
                        "request_url": request_url,
                        "attempt_index": attempt_index + 1,
                    },
                )
                return cast(dict[str, object], response.json())
            except (httpx.HTTPError, ValueError) as request_error:
                external_enrichment_client_logger.warning(
                    "external_enrichment_call_failed",
                    extra={
                        "operation_name": operation_name,
                        "request_url": request_url,
                        "attempt_index": attempt_index + 1,
                        "error_message": str(request_error),
                    },
                )
                if attempt_index == total_attempt_count - 1:
                    break
                backoff_seconds = self._retry_backoff_seconds * (attempt_index + 1)
                sleep(backoff_seconds)
        return None


class NoopExternalEnrichmentClient(ExternalEnrichmentClient):
    """Stub client: no HTTP; every lookup returns `unavailable` and empty lists."""

    def lookup_entity_description(
        self, entity_name: str
    ) -> EntityDescriptionLookupResult:
        """Ignore `entity_name`; always return no description and unavailable."""

        _ = entity_name
        return EntityDescriptionLookupResult(description=None, outcome="unavailable")

    def lookup_interaction_partners(
        self, entity_name: str
    ) -> InteractionPartnerLookupResult:
        """Ignore `entity_name`; always return no partners and unavailable."""

        _ = entity_name
        return InteractionPartnerLookupResult(
            partner_entity_names=[],
            outcome="unavailable",
        )

    def supports_interaction_strategy(self) -> bool:
        """Always false: this stub never enables interaction attachment."""

        return False


class CachingExternalEnrichmentClient(ExternalEnrichmentClient):
    """Delegates to an inner client on cache miss; stores JSON-serialisable dicts on hit write.

    Keys are built from normalised entity name plus operation prefix so description
    and interaction caches never collide. Inner client answers `supports_interaction_strategy`.
    """

    def __init__(
        self,
        wrapped_external_enrichment_client: ExternalEnrichmentClient,
        cache_store: CacheStore,
        service_metrics: ServiceMetrics | None = None,
    ) -> None:
        """Remember inner client, disk or other cache backend, and optional metrics."""

        self._wrapped_external_enrichment_client = wrapped_external_enrichment_client
        self._cache_store = cache_store
        self._service_metrics = service_metrics

    def lookup_entity_description(
        self, entity_name: str
    ) -> EntityDescriptionLookupResult:
        """Try cache; on valid hit return stored result; else call inner client then `set`."""

        cache_key = _build_description_cache_key(entity_name=entity_name)
        cached_entry = self._cache_store.get(cache_key=cache_key)
        if cached_entry is not None:
            cached_description = cached_entry.payload.get("description")
            cached_outcome = cached_entry.payload.get("outcome")
            if isinstance(cached_outcome, str) and (
                isinstance(cached_description, str) or cached_description is None
            ):
                external_enrichment_client_logger.info(
                    "cache_lookup_result",
                    extra={"cache_key": cache_key, "cache_outcome": "hit"},
                )
                self._record_cache_event(
                    cache_name="entity_description_lookup",
                    outcome="hit",
                )
                return EntityDescriptionLookupResult(
                    description=cached_description,
                    outcome=cast(
                        Literal["success", "not_found", "failed", "unavailable"],
                        cached_outcome,
                    ),
                )
        external_enrichment_client_logger.info(
            "cache_lookup_result",
            extra={"cache_key": cache_key, "cache_outcome": "miss"},
        )
        self._record_cache_event(
            cache_name="entity_description_lookup",
            outcome="miss",
        )
        lookup_result = (
            self._wrapped_external_enrichment_client.lookup_entity_description(
                entity_name=entity_name
            )
        )
        self._cache_store.set(
            cache_key=cache_key,
            payload={
                "description": lookup_result.description,
                "outcome": lookup_result.outcome,
            },
        )
        external_enrichment_client_logger.info(
            "cache_lookup_result",
            extra={"cache_key": cache_key, "cache_outcome": "write"},
        )
        self._record_cache_event(
            cache_name="entity_description_lookup",
            outcome="write",
        )
        return lookup_result

    def lookup_interaction_partners(
        self, entity_name: str
    ) -> InteractionPartnerLookupResult:
        """Same pattern as description lookup: cache read, inner client on miss, then write."""

        cache_key = _build_interaction_cache_key(entity_name=entity_name)
        cached_entry = self._cache_store.get(cache_key=cache_key)
        if cached_entry is not None:
            cached_partner_entity_names = cached_entry.payload.get(
                "partner_entity_names"
            )
            cached_outcome = cached_entry.payload.get("outcome")
            if isinstance(cached_partner_entity_names, list) and isinstance(
                cached_outcome, str
            ):
                partner_entity_names = [
                    partner_name
                    for partner_name in cached_partner_entity_names
                    if isinstance(partner_name, str)
                ]
                external_enrichment_client_logger.info(
                    "cache_lookup_result",
                    extra={"cache_key": cache_key, "cache_outcome": "hit"},
                )
                self._record_cache_event(
                    cache_name="interaction_partner_lookup",
                    outcome="hit",
                )
                return InteractionPartnerLookupResult(
                    partner_entity_names=partner_entity_names,
                    outcome=cast(
                        Literal["success", "not_found", "failed", "unavailable"],
                        cached_outcome,
                    ),
                )
        external_enrichment_client_logger.info(
            "cache_lookup_result",
            extra={"cache_key": cache_key, "cache_outcome": "miss"},
        )
        self._record_cache_event(
            cache_name="interaction_partner_lookup",
            outcome="miss",
        )
        lookup_result = (
            self._wrapped_external_enrichment_client.lookup_interaction_partners(
                entity_name=entity_name
            )
        )
        self._cache_store.set(
            cache_key=cache_key,
            payload={
                "partner_entity_names": lookup_result.partner_entity_names,
                "outcome": lookup_result.outcome,
            },
        )
        external_enrichment_client_logger.info(
            "cache_lookup_result",
            extra={"cache_key": cache_key, "cache_outcome": "write"},
        )
        self._record_cache_event(
            cache_name="interaction_partner_lookup",
            outcome="write",
        )
        return lookup_result

    def supports_interaction_strategy(self) -> bool:
        """Pass through to wrapped client; cache layer does not change capability."""

        return self._wrapped_external_enrichment_client.supports_interaction_strategy()

    def _record_cache_event(self, cache_name: str, outcome: str) -> None:
        """Emit hit, miss, or write counter when `service_metrics` is not None."""

        if self._service_metrics is None:
            return
        self._service_metrics.increment_cache_event(
            cache_name=cache_name,
            outcome=outcome,
        )


def _build_description_cache_key(entity_name: str) -> str:
    """Stable string key: prefix `description:` plus hash of lowercased trimmed name."""

    normalised_entity_name = entity_name.strip().lower()
    return "description:" + sha256(normalised_entity_name.encode("utf-8")).hexdigest()


def _build_interaction_cache_key(entity_name: str) -> str:
    """Same as description key but prefix `interaction:` so namespaces do not overlap."""

    normalised_entity_name = entity_name.strip().lower()
    return "interaction:" + sha256(normalised_entity_name.encode("utf-8")).hexdigest()
