"""External client implementations for enrichment dependencies."""

from model_serving_platform.infrastructure.clients.enrichment import (
    EntityDescriptionLookupResult,
    ExternalEnrichmentClient,
    HttpExternalEnrichmentClient,
    InteractionPartnerLookupResult,
    NoopExternalEnrichmentClient,
)

__all__ = [
    "ExternalEnrichmentClient",
    "HttpExternalEnrichmentClient",
    "NoopExternalEnrichmentClient",
    "EntityDescriptionLookupResult",
    "InteractionPartnerLookupResult",
]
