"""Application service for prediction endpoint orchestration."""

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
    """Raise when both endpoints are unseen in a pair prediction request.

    Stage 4 keeps this as an explicit contract because v1 allows at most one
    unseen endpoint for pairwise predictions and must reject two unseen nodes.
    Parameters: this exception uses a fixed human-readable message.
    """


class TopKLimitExceededError(ValueError):
    """Raise when requested top-k exceeds configured API upper bound.

    The service enforces this to prevent oversized ranking requests from
    creating unpredictable latency and excessive response payload size.
    Parameters: this exception uses a fixed human-readable message.
    """


class MissingDescriptionForRestrictedNetworkError(ValueError):
    """Raise when unseen entities omit descriptions in restricted mode.

    Restricted-network mode disables external enrichment assumptions, so
    unseen entities must include caller-provided descriptions by contract.
    Parameters: this exception uses a fixed human-readable message.
    """


class PredictionService:
    """Coordinate request-level prediction behaviour outside route handlers.

    Route handlers delegate to this service so business rules and runtime
    interactions stay testable and do not leak into transport-layer code.
    Parameters: runtime and startup metadata are provided at app creation.
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
        """Initialise the prediction orchestration service.

        Startup wiring passes configuration and metadata once so each request
        can reuse stable values without recomputing service-level settings.
        Parameters: fields are persisted for request orchestration methods.
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
        """Predict a link score for one entity pair using runtime boundary.

        This method applies request rules, resolves attachment strategy, and
        returns a typed response with latency and request correlation fields.
        Parameters: predict_link_request contains one pair prediction request.
        """

        request_start_time = perf_counter()
        source_entity_is_known = self._inference_runtime.has_entity_name(
            predict_link_request.entity_a_name
        )
        target_entity_is_known = self._inference_runtime.has_entity_name(
            predict_link_request.entity_b_name
        )
        if not source_entity_is_known and not target_entity_is_known:
            raise TwoUnseenEndpointsError(
                "Pair predictions do not support two unseen endpoints in v1."
            )
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
        """Predict ranked link scores for one source against known candidates.

        This method enforces top-k constraints and delegates ranking to the
        runtime so endpoint logic remains thin and configuration-aware.
        Parameters: predict_links_request contains one ranking request.
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
        """Resolve request attachment strategy with configuration defaults.

        This helper keeps defaulting behaviour consistent across endpoints and
        centralises strategy resolution in one explicit service function.
        Parameters: requested_attachment_strategy may be omitted by clients.
        """

        if requested_attachment_strategy is not None:
            resolved_attachment_strategy = requested_attachment_strategy
        else:
            resolved_attachment_strategy = self._default_attachment_strategy
        if (
            resolved_attachment_strategy == "interaction"
            and not self._inference_runtime.supports_interaction_strategy()
        ):
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
        """Validate runtime attachment strategy values for typed responses.

        Response models expose a strict literal strategy value, so this helper
        enforces that runtime values remain within the public API contract.
        Parameters: attachment_strategy is provided by runtime results.
        """

        if attachment_strategy in ("interaction", "cosine"):
            return cast(AttachmentStrategy, attachment_strategy)
        raise ValueError("Runtime produced an unsupported attachment strategy.")
