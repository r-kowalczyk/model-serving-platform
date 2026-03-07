"""Prediction endpoints for GraphSAGE service inference APIs."""

from fastapi import APIRouter, HTTPException, Request, status

from model_serving_platform.application.prediction_service import (
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


@prediction_router.post("/predict-link", response_model=PredictLinkResponse)
def predict_link(
    request: Request, predict_link_request: PredictLinkRequest
) -> PredictLinkResponse:
    """Predict one link score between two provided entities.

    This endpoint delegates all business logic to application service code
    so transport handling stays clear, typed, and easy to maintain.
    Parameters: request carries app state and body contains prediction data.
    """

    prediction_service: PredictionService = request.app.state.prediction_service
    try:
        return prediction_service.predict_link(
            predict_link_request=predict_link_request
        )
    except TwoUnseenEndpointsError as two_unseen_endpoints_error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(two_unseen_endpoints_error),
        ) from two_unseen_endpoints_error


@prediction_router.post("/predict-links", response_model=PredictLinksResponse)
def predict_links(
    request: Request, predict_links_request: PredictLinksRequest
) -> PredictLinksResponse:
    """Predict ranked candidate link scores for one source entity.

    This endpoint uses application service orchestration to enforce API-level
    constraints while leaving route code as a thin transport boundary.
    Parameters: request carries app state and body contains ranking input.
    """

    prediction_service: PredictionService = request.app.state.prediction_service
    try:
        return prediction_service.predict_links(
            predict_links_request=predict_links_request
        )
    except TopKLimitExceededError as top_k_limit_exceeded_error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(top_k_limit_exceeded_error),
        ) from top_k_limit_exceeded_error
