"""Runtime boundary between service logic and GraphSAGE infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RuntimePredictionResult:
    """Represent a single runtime score result used by service logic.

    This structure keeps transport and runtime concerns separate because
    route handlers should depend on plain application data types only.
    Parameters: fields identify entities and the computed score value.
    """

    source_entity_name: str
    target_entity_name: str
    score: float
    attachment_strategy_used: str
    enrichment_status: str


@dataclass(frozen=True, slots=True)
class RuntimeInitialisationSummary:
    """Describe runtime startup details used by readiness and metadata.

    The service stores this summary in app state so operators can verify
    which runtime path initialised and whether base embeddings were prepared.
    Parameters: each field is generated during runtime initialisation.
    """

    runtime_name: str
    model_num_layers: int
    base_embedding_count: int
    is_ready: bool
    readiness_reason: str


class InferenceRuntime(Protocol):
    """Define the minimum inference boundary required by service code.

    This protocol keeps route handlers independent from concrete GraphSAGE
    classes and enables deterministic fake runtimes in service-level tests.
    Parameters: implementations follow these method signatures exactly.
    """

    initialisation_summary: RuntimeInitialisationSummary

    def score_entity_pair(
        self,
        source_entity_name: str,
        target_entity_name: str,
        attachment_strategy: str,
    ) -> RuntimePredictionResult:
        """Score a single entity pair for potential link likelihood.

        Implementations may use precomputed embeddings or unseen-node flow
        depending on endpoint membership and selected attachment strategy.
        Parameters: entity names and strategy identify one prediction request.
        """

    def score_entity_against_candidates(
        self,
        source_entity_name: str,
        candidate_entity_names: list[str],
        top_k: int,
        attachment_strategy: str,
    ) -> list[RuntimePredictionResult]:
        """Score one source entity against many candidate entity names.

        This method supports top-k style link recommendation use cases while
        keeping ranking logic within infrastructure-specific runtime code.
        Parameters: names, top_k, and strategy define ranking behaviour.
        """
