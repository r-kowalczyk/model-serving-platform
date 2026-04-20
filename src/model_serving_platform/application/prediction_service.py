"""Prediction coordinator used by API routes.

This file contains the `PredictionService` class, which is the layer between
HTTP route handlers and the lower-level inference runtime.
Route handlers pass validated request objects into this service.
The service applies API business rules, calls runtime scoring methods, records
metrics and logs, and returns response objects expected by the API layer.
In plain terms, this is the place where request policy is enforced, while
model scoring remains in runtime code.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import cast
from uuid import uuid4

from model_serving_platform.application.inference_runtime import InferenceRuntime
from model_serving_platform.domain.prediction import (
    AttachmentStrategy,
    PredictLinkRequest,
    PredictLinkResponse,
    PredictLinksRequest,
    PredictLinksResponse,
    PredictionItem,
)
from model_serving_platform.infrastructure.metrics import ServiceMetrics

prediction_service_logger = logging.getLogger(
    "model_serving_platform.prediction_service"
)


class TwoUnseenEndpointsError(ValueError):
    """Error for pair requests where both entities are unknown.

    In v1 behaviour, a pair request may include at most one unseen entity.
    If both are unseen, the service rejects the request deterministically.
    """


class TopKLimitExceededError(ValueError):
    """Error for ranking requests that ask for too many results.

    The service has a configured `max_top_k` limit.
    If a request exceeds that limit, the service rejects it.
    This keeps response sizes and runtime cost within expected bounds.
    """


class MissingDescriptionForRestrictedNetworkError(ValueError):
    """Error for missing descriptions in restricted network mode.

    In restricted mode, the service cannot assume it can fetch missing text
    descriptions from external systems.
    For unseen entities, the caller must provide descriptions directly.
    """


class PredictionService:
    """Run request-level prediction policy and runtime calls.

    This class is intentionally separate from route handlers.
    Routes should focus on HTTP transport concerns, while this class focuses on
    prediction rules and response composition.
    The class receives long-lived dependencies at startup, then reuses them for
    every request.

    Examples of HTTP transport concerns (kept in routes):
    - Read HTTP body JSON into typed request models.
    - Convert service exceptions into HTTP status codes such as 422.
    - Read request-scoped values from middleware state.

    Examples of prediction rules (kept in this class):
    - Reject pair requests where both endpoints are unseen.
    - Enforce configured `max_top_k` limits for ranking requests.
    - Require caller descriptions for unseen entities in restricted mode.
    - Apply explicit strategy fallback when interaction strategy is unavailable.

    Examples of response composition (kept in this class):
    - Build `PredictLinkResponse` and `PredictLinksResponse` objects.
    - Copy metadata fields such as model version and bundle version.
    - Attach latency and request identifier values to each response.
    """

    def __init__(
        self,
        inference_runtime: InferenceRuntime,
        service_version: str,
        bundle_version: str | None,
        max_top_k: int,
        default_attachment_strategy: AttachmentStrategy,
        restricted_network_mode: bool = False,
        service_metrics: ServiceMetrics | None = None,
    ) -> None:
        """Store startup dependencies and configuration for later requests.

        Parameters:
        - `inference_runtime`: object that performs actual scoring operations.
        - `service_version` and `bundle_version`: returned in API responses.
        - `max_top_k`: upper bound for ranking request size.
        - `default_attachment_strategy`: strategy used when client omits one.
        - `restricted_network_mode`: enables stricter input requirements.
        - `service_metrics`: optional metrics collector for counters.
        """

        self._inference_runtime = inference_runtime
        self._service_version = service_version
        self._bundle_version = bundle_version
        self._max_top_k = max_top_k
        self._default_attachment_strategy = default_attachment_strategy
        self._restricted_network_mode = restricted_network_mode
        self._service_metrics = service_metrics

    def predict_link(
        self, predict_link_request: PredictLinkRequest
    ) -> PredictLinkResponse:
        """Handle one pairwise prediction request from start to finish.

        Steps performed by this method:
        1. Start a latency timer.
        2. Check whether each requested entity is known to the runtime.
           Here "runtime" means the in-memory inference engine loaded at startup.
        3. Apply request rules such as "both entities cannot be unseen".
        4. In restricted mode, require descriptions for unseen entities.
        5. Resolve the attachment strategy, including controlled fallback.
        6. Call runtime scoring for one entity pair.
        7. Build and return the API response payload.
        """

        request_start_time = perf_counter()
        source_entity_is_known = self._inference_runtime.has_entity_name(
            predict_link_request.entity_a_name
        )
        target_entity_is_known = self._inference_runtime.has_entity_name(
            predict_link_request.entity_b_name
        )
        # v1 policy rejects pair requests when both entities are unseen because
        # this path is intentionally constrained to at most one unseen endpoint.
        if not source_entity_is_known and not target_entity_is_known:
            raise TwoUnseenEndpointsError(
                "Pair predictions do not support two unseen endpoints in v1."
            )
        # Restricted mode assumes no external enrichment source is available, so
        # caller-provided descriptions are required for unseen entities.
        if self._restricted_network_mode:
            if (
                not source_entity_is_known
                and predict_link_request.entity_a_description is None
            ):
                raise MissingDescriptionForRestrictedNetworkError(
                    "Restricted network mode requires entity_a_description for unseen entities."
                )
            if (
                not target_entity_is_known
                and predict_link_request.entity_b_description is None
            ):
                raise MissingDescriptionForRestrictedNetworkError(
                    "Restricted network mode requires entity_b_description for unseen entities."
                )

        (
            resolved_attachment_strategy,
            strategy_fallback_was_used,
        ) = self._resolve_attachment_strategy(
            requested_attachment_strategy=predict_link_request.attachment_strategy
        )
        # Runtime call performs actual scoring. Service layer keeps policy and
        # response shaping, while runtime layer keeps model-specific computation.
        runtime_prediction_result = self._inference_runtime.score_entity_pair(
            source_entity_name=predict_link_request.entity_a_name,
            target_entity_name=predict_link_request.entity_b_name,
            attachment_strategy=resolved_attachment_strategy,
            source_entity_description=predict_link_request.entity_a_description,
            target_entity_description=predict_link_request.entity_b_description,
        )
        request_latency_milliseconds = (perf_counter() - request_start_time) * 1000
        resolved_request_id = predict_link_request.request_id or str(uuid4())
        if self._service_metrics is not None:
            self._service_metrics.increment_prediction_count(
                endpoint="/v1/predict-link"
            )
        prediction_service_logger.info(
            "inference_complete",
            extra={
                "endpoint": "/v1/predict-link",
                "latency_ms": request_latency_milliseconds,
                "attachment_strategy_used": runtime_prediction_result.attachment_strategy_used,
                "enrichment_status": runtime_prediction_result.enrichment_status,
                "bundle_version": self._bundle_version,
                "service_version": self._service_version,
                "request_id": resolved_request_id,
                "fallback_used": strategy_fallback_was_used,
            },
        )

        if strategy_fallback_was_used:
            response_enrichment_status = "interaction_unavailable_fallback_to_cosine"
        else:
            response_enrichment_status = runtime_prediction_result.enrichment_status

        return PredictLinkResponse(
            score=runtime_prediction_result.score,
            predicted_label=None,
            model_version=self._service_version,
            bundle_version=self._bundle_version,
            attachment_strategy_used=self._normalise_attachment_strategy(
                attachment_strategy=runtime_prediction_result.attachment_strategy_used
            ),
            enrichment_status=response_enrichment_status,
            latency_ms=request_latency_milliseconds,
            request_id=resolved_request_id,
        )

    def predict_links(
        self, predict_links_request: PredictLinksRequest
    ) -> PredictLinksResponse:
        """Handle one ranking prediction request from start to finish.

        Steps performed by this method:
        1. Start a latency timer.
        2. Enforce `top_k` upper bound to control request size.
        3. Apply restricted-mode description requirement for unseen source.
        4. Resolve the attachment strategy, including controlled fallback.
        5. Build candidate list from known entities, excluding source itself.
        6. Call runtime scoring against candidate entities.
        7. Build and return ranked API response payload.
        """

        request_start_time = perf_counter()
        if predict_links_request.top_k > self._max_top_k:
            raise TopKLimitExceededError(
                f"Requested top_k exceeds configured max_top_k={self._max_top_k}."
            )
        source_entity_is_known = self._inference_runtime.has_entity_name(
            predict_links_request.entity_name
        )
        if (
            self._restricted_network_mode
            and not source_entity_is_known
            and predict_links_request.entity_description is None
        ):
            raise MissingDescriptionForRestrictedNetworkError(
                "Restricted network mode requires entity_description for unseen entities."
            )

        (
            resolved_attachment_strategy,
            strategy_fallback_was_used,
        ) = self._resolve_attachment_strategy(
            requested_attachment_strategy=predict_links_request.attachment_strategy
        )
        known_entity_names = self._inference_runtime.get_known_entity_names()
        # Remove the source entity from candidate list so self-links are not
        # returned as recommendations unless a future API version allows them.
        candidate_entity_names = [
            candidate_entity_name
            for candidate_entity_name in known_entity_names
            if candidate_entity_name != predict_links_request.entity_name
        ]
        ranked_prediction_results = (
            self._inference_runtime.score_entity_against_candidates(
                source_entity_name=predict_links_request.entity_name,
                candidate_entity_names=candidate_entity_names,
                top_k=predict_links_request.top_k,
                attachment_strategy=resolved_attachment_strategy,
                source_entity_description=predict_links_request.entity_description,
            )
        )
        request_latency_milliseconds = (perf_counter() - request_start_time) * 1000
        resolved_request_id = predict_links_request.request_id or str(uuid4())
        if self._service_metrics is not None:
            self._service_metrics.increment_prediction_count(
                endpoint="/v1/predict-links"
            )

        if ranked_prediction_results:
            response_enrichment_status = ranked_prediction_results[0].enrichment_status
            attachment_strategy_used = self._normalise_attachment_strategy(
                attachment_strategy=ranked_prediction_results[
                    0
                ].attachment_strategy_used
            )
        else:
            response_enrichment_status = "not_required"
            attachment_strategy_used = resolved_attachment_strategy
        if strategy_fallback_was_used:
            response_enrichment_status = "interaction_unavailable_fallback_to_cosine"

        prediction_service_logger.info(
            "inference_complete",
            extra={
                "endpoint": "/v1/predict-links",
                "latency_ms": request_latency_milliseconds,
                "prediction_count": len(ranked_prediction_results),
                "attachment_strategy_used": attachment_strategy_used,
                "enrichment_status": response_enrichment_status,
                "bundle_version": self._bundle_version,
                "service_version": self._service_version,
                "request_id": resolved_request_id,
                "fallback_used": strategy_fallback_was_used,
            },
        )

        return PredictLinksResponse(
            predictions=[
                PredictionItem(
                    entity_name=prediction_result.target_entity_name,
                    score=prediction_result.score,
                )
                for prediction_result in ranked_prediction_results
            ],
            predicted_label=None,
            model_version=self._service_version,
            bundle_version=self._bundle_version,
            attachment_strategy_used=attachment_strategy_used,
            enrichment_status=response_enrichment_status,
            latency_ms=request_latency_milliseconds,
            request_id=resolved_request_id,
        )

    def _resolve_attachment_strategy(
        self, requested_attachment_strategy: AttachmentStrategy | None
    ) -> tuple[AttachmentStrategy, bool]:
        """Choose which attachment strategy the runtime should use.

        Behaviour:
        - If the client provided a strategy, use it.
        - Otherwise, use the configured default strategy.
        - If strategy is `interaction` but runtime does not support it, switch
          to `cosine`, emit fallback metrics/logs, and return `True` fallback flag.
        """

        if requested_attachment_strategy is not None:
            resolved_attachment_strategy = requested_attachment_strategy
        else:
            resolved_attachment_strategy = self._default_attachment_strategy
        if (
            resolved_attachment_strategy == "interaction"
            and not self._inference_runtime.supports_interaction_strategy()
        ):
            # Fallback is explicit and observable so operators can track when
            # requested strategy and runtime capability do not match.
            if self._service_metrics is not None:
                self._service_metrics.increment_fallback_usage(
                    reason="interaction_strategy_unavailable"
                )
            prediction_service_logger.info(
                "fallback_used",
                extra={
                    "fallback_reason": "interaction_strategy_unavailable",
                    "fallback_from_strategy": "interaction",
                    "fallback_to_strategy": "cosine",
                },
            )
            return "cosine", True
        return resolved_attachment_strategy, False

    def _normalise_attachment_strategy(
        self, attachment_strategy: str
    ) -> AttachmentStrategy:
        """Map runtime strategy text to API response literal values.

        The public response model allows only `interaction` or `cosine`.
        This method enforces that contract before response construction.
        """

        if attachment_strategy in ("interaction", "cosine"):
            return cast(AttachmentStrategy, attachment_strategy)
        raise ValueError("Runtime produced an unsupported attachment strategy.")
