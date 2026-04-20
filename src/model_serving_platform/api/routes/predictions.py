"""Routes that expose GraphSAGE link prediction operations over HTTP.

This module defines two POST endpoints: one for a single pair score and one
for ranked candidate links from one source entity to many possible targets.
The route functions are transport adapters: they read request bodies, obtain
the shared `PredictionService` from `request.app.state`, and call service logic.
Each request is given a request identifier by using the body value when present
or falling back to the middleware-provided `request.state.request_id` value.
Domain validation errors are translated into HTTP 422 responses with clear text
so API clients get deterministic failure behaviour for invalid input requests.
Structured warning logs are emitted for rejected requests to support operations.
"""

import logging

from fastapi import APIRouter, HTTPException, Request, status

from model_serving_platform.application.prediction_service import (
    MissingDescriptionForRestrictedNetworkError,
    PredictionService,
    TopKLimitExceededError,
    TwoUnseenEndpointsError,
)
from model_serving_platform.domain.prediction import (
    PredictLinkRequest,
    PredictLinkResponse,
    PredictLinksRequest,
    PredictLinksResponse,
)

prediction_router = APIRouter(prefix="/v1")
prediction_router_logger = logging.getLogger("model_serving_platform.api.predictions")


@prediction_router.post("/predict-link", response_model=PredictLinkResponse)
def predict_link(
    request: Request, predict_link_request: PredictLinkRequest
) -> PredictLinkResponse:
    """Predict one link score between two provided entities.

    This endpoint delegates all business logic to application service code
    so transport handling stays clear, typed, and easy to maintain.
    Parameters: request carries app state and body contains prediction data.
    """

    # `request.app.state` is an attribute bag on the FastAPI app object.
    # During startup, the application attaches long-lived objects to it, for
    # example `prediction_service`, `service_metrics`, and service settings.
    # They are stored there so every request handler can reuse the same loaded
    # objects instead of rebuilding them for each incoming API request.
    prediction_service: PredictionService = request.app.state.prediction_service

    # Use the client-provided request identifier when present, otherwise use
    # the middleware identifier stored on this specific request object.
    resolved_request_id = predict_link_request.request_id or request.state.request_id
    resolved_predict_link_request = predict_link_request.model_copy(
        update={"request_id": resolved_request_id}
    )
    try:
        return prediction_service.predict_link(
            predict_link_request=resolved_predict_link_request
        )
    except TwoUnseenEndpointsError as two_unseen_endpoints_error:
        prediction_router_logger.warning(
            "prediction_request_rejected",
            extra={
                "endpoint": "/v1/predict-link",
                "error_type": "two_unseen_endpoints",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(two_unseen_endpoints_error),
        ) from two_unseen_endpoints_error
    except MissingDescriptionForRestrictedNetworkError as missing_description_error:
        prediction_router_logger.warning(
            "prediction_request_rejected",
            extra={
                "endpoint": "/v1/predict-link",
                "error_type": "missing_description_for_restricted_network",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(missing_description_error),
        ) from missing_description_error


@prediction_router.post("/predict-links", response_model=PredictLinksResponse)
def predict_links(
    request: Request, predict_links_request: PredictLinksRequest
) -> PredictLinksResponse:
    """Predict ranked candidate link scores for one source entity.

    This endpoint uses application service orchestration to enforce API-level
    constraints while leaving route code as a thin transport boundary.
    Parameters: request carries app state and body contains ranking input.
    """

    # `request.app.state` stores shared objects created once at startup.
    prediction_service: PredictionService = request.app.state.prediction_service

    # Use the client-provided request identifier when present, otherwise use
    # the middleware identifier stored on this specific request object.
    resolved_request_id = predict_links_request.request_id or request.state.request_id
    resolved_predict_links_request = predict_links_request.model_copy(
        update={"request_id": resolved_request_id}
    )
    try:
        return prediction_service.predict_links(
            predict_links_request=resolved_predict_links_request
        )
    except TopKLimitExceededError as top_k_limit_exceeded_error:
        prediction_router_logger.warning(
            "prediction_request_rejected",
            extra={
                "endpoint": "/v1/predict-links",
                "error_type": "top_k_limit_exceeded",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(top_k_limit_exceeded_error),
        ) from top_k_limit_exceeded_error
    except MissingDescriptionForRestrictedNetworkError as missing_description_error:
        prediction_router_logger.warning(
            "prediction_request_rejected",
            extra={
                "endpoint": "/v1/predict-links",
                "error_type": "missing_description_for_restricted_network",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(missing_description_error),
        ) from missing_description_error
