"""Shared runtime interface used by the rest of the application.

This file does not run a model by itself. It defines the shape of the runtime
that other parts of the service are allowed to call.
In this project, "service code" means the code that handles API requests, checks
input rules, and builds API responses. It is the coordinator layer.
The model engine itself lives elsewhere and can be swapped or changed.
The phrase "inference boundary" means the hand-off line between those two parts:
service code calls runtime methods, and runtime code returns plain results.
This boundary is useful because each side can change internally as long as the
shared method signatures stay the same. A method signature is the exact method
name, input fields, and output type. If that signature changes, callers break.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RuntimePredictionResult:
    """One prediction result returned by a runtime method.

    Think of this as one row in a results table:
    source entity, target entity, predicted score, and context fields that
    explain which strategy was used and whether enrichment was needed.
    """

    # These fields are the minimum prediction output values that service code
    # needs in order to build domain responses without runtime-specific types.
    source_entity_name: str
    target_entity_name: str
    score: float
    attachment_strategy_used: str
    enrichment_status: str


@dataclass(frozen=True, slots=True)
class RuntimeInitialisationSummary:
    """Startup summary describing whether the runtime is ready to serve traffic.

    The application stores this object so health and metadata endpoints can
    explain startup status in plain terms, for example "ready" or "not ready",
    together with the reason and key model details.
    """

    # These startup fields are surfaced through readiness and metadata routes so
    # operators can verify what runtime path was initialised and why.
    runtime_name: str
    model_num_layers: int
    base_embedding_count: int
    is_ready: bool
    readiness_reason: str


class InferenceRuntime(Protocol):
    """Runtime interface that service code depends on.

    A `Protocol` is a Python typing feature that defines required attributes
    and methods without providing implementation.
    Any class that provides these members can be used as an `InferenceRuntime`.
    This lets application code depend on behaviour, not on one concrete class.
    In tests, you can provide a simple fake runtime with the same signatures.
    """

    # Runtime implementations expose startup status as a data attribute so
    # application code can read readiness and metadata without calling methods.
    initialisation_summary: RuntimeInitialisationSummary

    def has_entity_name(self, entity_name: str) -> bool:
        """Check whether this entity is already known by the loaded runtime data.

        In simple terms, this asks: "Did this name exist in the model bundle
        when the service started?"
        The service layer uses this to decide whether a request is valid.
        """

    def get_known_entity_names(self) -> list[str]:
        """Return all entity names that are currently known to the runtime.

        The ranking endpoint uses this list as the default candidate pool when
        it needs to score one source entity against many possible targets.
        """

    def supports_interaction_strategy(self) -> bool:
        """Return whether the runtime can use the interaction attachment strategy.

        The service layer checks this before running predictions so it can
        reject unsupported requests clearly instead of failing later.
        """

    def score_entity_pair(
        self,
        source_entity_name: str,
        target_entity_name: str,
        attachment_strategy: str,
        source_entity_description: str | None = None,
        target_entity_description: str | None = None,
    ) -> RuntimePredictionResult:
        """Predict one link score between two entities.

        Inputs identify the source and target entities, plus the requested
        attachment strategy and optional text descriptions for unseen entities.
        Output is one `RuntimePredictionResult`.
        """

    def score_entity_against_candidates(
        self,
        source_entity_name: str,
        candidate_entity_names: list[str],
        top_k: int,
        attachment_strategy: str,
        source_entity_description: str | None = None,
    ) -> list[RuntimePredictionResult]:
        """Predict scores from one source entity to many candidate entities.

        The runtime returns multiple `RuntimePredictionResult` values, typically
        sorted so the highest scoring candidates are first, up to `top_k`.
        """
