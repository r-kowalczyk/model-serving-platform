"""Prometheus metrics collector for service runtime observability."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


class ServiceMetrics:
    """Collect and expose Prometheus metrics for service operations.

    A dedicated collector class keeps metrics logic out of route and runtime
    code while providing one shared place for counters and histograms.
    Parameters: enabled controls whether metrics recording is active.
    """

    def __init__(self, enabled: bool) -> None:
        """Initialise metric registry and metric instruments when enabled.

        A per-instance registry avoids global duplicate metric registration
        during repeated app factory use in automated test execution.
        Parameters: enabled toggles recording and rendering behaviour.
        """

        self._enabled = enabled
        self._registry = CollectorRegistry()
        self._http_request_count = Counter(
            "model_serving_http_request_total",
            "Total HTTP requests handled by endpoint, method, and status code.",
            ["endpoint", "method", "status_code"],
            registry=self._registry,
        )
        self._http_request_latency_seconds = Histogram(
            "model_serving_http_request_latency_seconds",
            "HTTP request latency in seconds by endpoint and method.",
            ["endpoint", "method"],
            registry=self._registry,
        )
        self._prediction_count = Counter(
            "model_serving_prediction_total",
            "Total prediction operations by endpoint.",
            ["endpoint"],
            registry=self._registry,
        )
        self._external_lookup_count = Counter(
            "model_serving_external_lookup_total",
            "Total external enrichment lookups by operation and outcome.",
            ["operation", "outcome"],
            registry=self._registry,
        )
        self._cache_event_count = Counter(
            "model_serving_cache_event_total",
            "Total cache events by cache name and outcome.",
            ["cache_name", "outcome"],
            registry=self._registry,
        )
        self._fallback_usage_count = Counter(
            "model_serving_fallback_total",
            "Total fallback decisions by reason.",
            ["reason"],
            registry=self._registry,
        )

    @property
    def enabled(self) -> bool:
        """Return whether metrics collection and rendering are enabled.

        Routes and instrumentation call this property to skip recording and
        avoid exposing metric payloads when metrics are disabled in settings.
        Parameters: none.
        """

        return self._enabled

    def observe_http_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_seconds: float,
    ) -> None:
        """Record one HTTP request count and latency observation.

        This method tracks service-level traffic by endpoint, method, and
        response status to support latency and reliability observability.
        Parameters: values are emitted from request middleware lifecycle.
        """

        if not self._enabled:
            return
        self._http_request_count.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(status_code),
        ).inc()
        self._http_request_latency_seconds.labels(
            endpoint=endpoint,
            method=method,
        ).observe(latency_seconds)

    def increment_prediction_count(self, endpoint: str) -> None:
        """Increment prediction operation counter for one endpoint label.

        Prediction counts provide a simple high-level throughput signal for
        pair and ranked scoring endpoints used by production-like monitoring.
        Parameters: endpoint is API path label for prediction operation.
        """

        if not self._enabled:
            return
        self._prediction_count.labels(endpoint=endpoint).inc()

    def increment_external_lookup(self, operation: str, outcome: str) -> None:
        """Increment external lookup metric for operation and outcome labels.

        Tracking lookup outcomes helps identify dependency instability and
        fallback pressure during external enrichment request execution.
        Parameters: operation and outcome identify lookup metric dimensions.
        """

        if not self._enabled:
            return
        self._external_lookup_count.labels(
            operation=operation,
            outcome=outcome,
        ).inc()

    def increment_cache_event(self, cache_name: str, outcome: str) -> None:
        """Increment cache event metric for cache area and outcome labels.

        Cache hit, miss, and write counts support verification that caching
        behaviour is functioning and reducing repeated external lookups.
        Parameters: cache_name and outcome identify cache event dimensions.
        """

        if not self._enabled:
            return
        self._cache_event_count.labels(
            cache_name=cache_name,
            outcome=outcome,
        ).inc()

    def increment_fallback_usage(self, reason: str) -> None:
        """Increment fallback metric for one fallback reason label value.

        Fallback counters expose degraded-path usage and help operators track
        when dependency or capability limits force alternate request handling.
        Parameters: reason describes why fallback behaviour was used.
        """

        if not self._enabled:
            return
        self._fallback_usage_count.labels(reason=reason).inc()

    def render_prometheus_text(self) -> str:
        """Render metrics payload in Prometheus text exposition format.

        Metrics endpoint calls this method to expose collector registry state
        to scrape clients and local observability checks.
        Parameters: none.
        """

        if not self._enabled:
            return ""
        return generate_latest(self._registry).decode("utf-8")
