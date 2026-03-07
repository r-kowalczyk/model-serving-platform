"""External client implementations for enrichment dependencies."""

from model_serving_platform.infrastructure.clients.enrichment import (
    CachingExternalEnrichmentClient,
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
    "CachingExternalEnrichmentClient",
    "EntityDescriptionLookupResult",
    "InteractionPartnerLookupResult",
]
