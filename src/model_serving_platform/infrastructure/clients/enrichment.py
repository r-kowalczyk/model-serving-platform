"""External enrichment client abstractions and HTTP implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import sleep
from typing import Literal, Protocol, cast

import httpx

external_enrichment_client_logger = logging.getLogger(
    "model_serving_platform.external_enrichment_client"
)


@dataclass(frozen=True, slots=True)
class EntityDescriptionLookupResult:
    """Represent the result of external description lookup for one entity.

    The service uses this result to decide whether unseen entity handling can
    proceed with external data or should take explicit degraded paths.
    Parameters: description and outcome describe lookup result state.
    """

    description: str | None
    outcome: Literal["success", "not_found", "failed", "unavailable"]


@dataclass(frozen=True, slots=True)
class InteractionPartnerLookupResult:
    """Represent the result of external interaction partner name lookup.

    Interaction lookups are optional in v1 and may fail, so this structure
    makes success and degraded outcomes explicit for runtime decisions.
    Parameters: partner names and outcome describe lookup result state.
    """

    partner_entity_names: list[str]
    outcome: Literal["success", "not_found", "failed", "unavailable"]


class ExternalEnrichmentClient(Protocol):
    """Define client operations used by GraphSAGE unseen entity handling.

    Runtime code depends on this protocol to avoid route-level HTTP logic and
    to keep enrichment concerns testable and replaceable by deterministic fakes.
    Parameters: implementations follow these method signatures exactly.
    """

    def lookup_entity_description(
        self, entity_name: str
    ) -> EntityDescriptionLookupResult:
        """Fetch one entity description from external enrichment provider.

        Implementations may return unavailable or failed outcomes when network
        access is disabled or downstream requests do not complete successfully.
        Parameters: entity_name identifies the unresolved endpoint entity.
        """

    def lookup_interaction_partners(
        self, entity_name: str
    ) -> InteractionPartnerLookupResult:
        """Fetch candidate interaction partner names for one source entity.

        Implementations should return partner names when available, otherwise
        return explicit degraded outcomes instead of raising route-level errors.
        Parameters: entity_name identifies source entity for interaction lookup.
        """

    def supports_interaction_strategy(self) -> bool:
        """Return whether interaction enrichment path is currently available.

        Service orchestration uses this check to enforce explicit strategy
        degradation when interaction lookups are unavailable by configuration.
        Parameters: none.
        """


class HttpExternalEnrichmentClient(ExternalEnrichmentClient):
    """HTTP enrichment client with timeout, retries, and bounded backoff.

    The implementation keeps external dependency handling inside infrastructure
    code so route and application layers stay focused on serving behaviour.
    Parameters: endpoint URLs and timeout/retry values are startup settings.
    """

    def __init__(
        self,
        description_lookup_url: str | None,
        interaction_lookup_url: str | None,
        timeout_seconds: float,
        retry_count: int,
        retry_backoff_seconds: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Initialise HTTP client configuration for enrichment operations.

        Optional transport injection exists for deterministic tests that avoid
        live network calls while covering timeout and retry control flow.
        Parameters: values are passed from service settings or test fixtures.
        """

        self._description_lookup_url = description_lookup_url
        self._interaction_lookup_url = interaction_lookup_url
        self._timeout_seconds = timeout_seconds
        self._retry_count = retry_count
        self._retry_backoff_seconds = retry_backoff_seconds
        self._http_client = httpx.Client(transport=transport)

    def lookup_entity_description(
        self, entity_name: str
    ) -> EntityDescriptionLookupResult:
        """Lookup one entity description using configured HTTP endpoint.

        This method returns explicit outcomes so runtime code can apply clear
        fallback behaviour without exposing raw transport-layer exceptions.
        Parameters: entity_name is sent as query parameter to lookup endpoint.
        """

        if self._description_lookup_url is None:
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
            return EntityDescriptionLookupResult(
                description=None,
                outcome="failed",
            )
        resolved_description = response_payload.get("description")
        if isinstance(resolved_description, str) and resolved_description != "":
            return EntityDescriptionLookupResult(
                description=resolved_description,
                outcome="success",
            )
        return EntityDescriptionLookupResult(
            description=None,
            outcome="not_found",
        )

    def lookup_interaction_partners(
        self, entity_name: str
    ) -> InteractionPartnerLookupResult:
        """Lookup external interaction partner names for one source entity.

        The runtime uses this optional signal for interaction attachment mode
        and falls back explicitly when partner data is unavailable.
        Parameters: entity_name is sent as query parameter to lookup endpoint.
        """

        if self._interaction_lookup_url is None:
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
            return InteractionPartnerLookupResult(
                partner_entity_names=[],
                outcome="failed",
            )
        raw_partner_names = response_payload.get("partners")
        if not isinstance(raw_partner_names, list):
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
            return InteractionPartnerLookupResult(
                partner_entity_names=[],
                outcome="not_found",
            )
        return InteractionPartnerLookupResult(
            partner_entity_names=partner_entity_names,
            outcome="success",
        )

    def supports_interaction_strategy(self) -> bool:
        """Return whether interaction lookup endpoint URL is configured.

        This simple check is used by service logic to degrade unsupported
        interaction attachment requests toward cosine strategy explicitly.
        Parameters: none.
        """

        return self._interaction_lookup_url is not None

    def _request_json_with_retries(
        self,
        request_url: str,
        query_params: dict[str, str],
        operation_name: str,
    ) -> dict[str, object] | None:
        """Execute one HTTP GET with timeout and bounded retry behaviour.

        This method retries transient transport failures and HTTP status errors
        with deterministic linear backoff then returns None on final failure.
        Parameters: request_url and query_params define one external request.
        """

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
    """External enrichment client that always reports unavailable outcomes.

    This implementation is useful for tests and local runs where no external
    enrichment dependency is configured but explicit degraded handling is needed.
    Parameters: none.
    """

    def lookup_entity_description(
        self, entity_name: str
    ) -> EntityDescriptionLookupResult:
        """Return unavailable outcome for description lookups.

        This method intentionally avoids network calls and communicates that
        external enrichment description lookup is not configured for this run.
        Parameters: entity_name is accepted for protocol compatibility.
        """

        _ = entity_name
        return EntityDescriptionLookupResult(description=None, outcome="unavailable")

    def lookup_interaction_partners(
        self, entity_name: str
    ) -> InteractionPartnerLookupResult:
        """Return unavailable outcome for interaction partner lookups.

        This method intentionally avoids network calls and communicates that
        interaction enrichment is not configured for this runtime environment.
        Parameters: entity_name is accepted for protocol compatibility.
        """

        _ = entity_name
        return InteractionPartnerLookupResult(
            partner_entity_names=[],
            outcome="unavailable",
        )

    def supports_interaction_strategy(self) -> bool:
        """Return false because interaction endpoint is not configured.

        Service-level strategy resolution reads this capability to downgrade
        unsupported interaction requests to cosine strategy explicitly.
        Parameters: none.
        """

        return False
