"""GraphSAGE bundle loader and startup contract validator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model_serving_platform.infrastructure.bundles.errors import (
    GraphSageBundleValidationError,
)
from model_serving_platform.infrastructure.bundles.manifest import (
    GraphSageBundleManifest,
)

REQUIRED_BUNDLE_FILE_NAMES = (
    "model_state.pt",
    "manifest.json",
    "node_features.npy",
    "edge_index.npy",
)


@dataclass(frozen=True, slots=True)
class LoadedGraphSageBundleMetadata:
    """Represent validated bundle metadata stored in application state.

    This value object is the service-facing result of startup validation.
    It keeps route code simple by exposing only data that is safe to return
    through metadata endpoints and operational logging.
    Parameters: all fields originate from validated bundle files.
    """

    bundle_path: str
    manifest_path: str
    model_state_path: str
    node_features_path: str
    edge_index_path: str
    node_count: int
    feature_dimension: int
    edge_count: int
    semantic_model_name: str
    semantic_max_length: int
    is_undirected: bool
    attachment_seed: int
    attachment_top_k: int
    model_architecture: dict[str, int | float | str]
    bundle_version: str | None


class GraphSageBundleLoader:
    """Load and validate a GraphSAGE serving bundle from a directory path.

    The loader performs contract checks before the API marks readiness true.
    This protects serving reliability by rejecting malformed bundle inputs
    before any request handling can begin.
    Parameters: none.
    """

    def load_and_validate_bundle(
        self, bundle_directory_path: Path
    ) -> LoadedGraphSageBundleMetadata:
        """Load bundle files and enforce the Stage 2 startup contract.

        Validation checks required files, manifest schema and tensor shapes.
        A failure raises a structured exception so startup stops immediately.
        Parameters: bundle_directory_path points to an extracted bundle.
        """

        self._validate_required_files(bundle_directory_path=bundle_directory_path)
        parsed_manifest = self._load_manifest(
            bundle_directory_path=bundle_directory_path
        )
        node_features_array, edge_index_array = self._load_graph_arrays(
            bundle_directory_path=bundle_directory_path
        )
        self._validate_graph_shapes(
            node_features_array=node_features_array,
            edge_index_array=edge_index_array,
            parsed_manifest=parsed_manifest,
        )
        return LoadedGraphSageBundleMetadata(
            bundle_path=str(bundle_directory_path.resolve()),
            manifest_path=str((bundle_directory_path / "manifest.json").resolve()),
            model_state_path=str((bundle_directory_path / "model_state.pt").resolve()),
            node_features_path=str(
                (bundle_directory_path / "node_features.npy").resolve()
            ),
            edge_index_path=str((bundle_directory_path / "edge_index.npy").resolve()),
            node_count=int(node_features_array.shape[0]),
            feature_dimension=int(node_features_array.shape[1]),
            edge_count=int(edge_index_array.shape[1]),
            semantic_model_name=parsed_manifest.semantic_model_name,
            semantic_max_length=parsed_manifest.semantic_max_length,
            is_undirected=parsed_manifest.is_undirected,
            attachment_seed=parsed_manifest.attachment_seed,
            attachment_top_k=parsed_manifest.attachment_top_k,
            model_architecture=parsed_manifest.model.model_dump(),
            bundle_version=parsed_manifest.bundle_version,
        )

    def _validate_required_files(self, bundle_directory_path: Path) -> None:
        """Ensure required bundle artefacts are present before loading data.

        This check exists to avoid partial startup where core model files are
        missing, because serving cannot recover from absent core artefacts.
        Parameters: bundle_directory_path points to the bundle root directory.
        """

        missing_file_names = [
            file_name
            for file_name in REQUIRED_BUNDLE_FILE_NAMES
            if not (bundle_directory_path / file_name).exists()
        ]
        if missing_file_names:
            raise GraphSageBundleValidationError(
                error_code="missing_required_files",
                message="Bundle is missing one or more required files.",
                details={
                    "bundle_directory_path": str(bundle_directory_path),
                    "missing_file_names": missing_file_names,
                },
            )

    def _load_manifest(self, bundle_directory_path: Path) -> GraphSageBundleManifest:
        """Read manifest JSON and validate it against the strict schema.

        Parsing into a typed model guarantees required keys are present before
        any tensor checks run, which keeps failures clear and deterministic.
        Parameters: bundle_directory_path points to the bundle root directory.
        """

        manifest_path = bundle_directory_path / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            return GraphSageBundleManifest.model_validate(manifest_data)
        except Exception as validation_error:
            raise GraphSageBundleValidationError(
                error_code="invalid_manifest_schema",
                message="Manifest schema validation failed.",
                details={
                    "manifest_path": str(manifest_path),
                    "validation_error": str(validation_error),
                },
            ) from validation_error

    def _load_graph_arrays(
        self, bundle_directory_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load graph arrays required for shape validation at startup.

        The service reads these arrays during startup because their shapes
        determine whether runtime reconstruction can succeed consistently.
        Parameters: bundle_directory_path points to the bundle root directory.
        """

        node_features_array = np.load(bundle_directory_path / "node_features.npy")
        edge_index_array = np.load(bundle_directory_path / "edge_index.npy")
        return node_features_array, edge_index_array

    def _validate_graph_shapes(
        self,
        node_features_array: np.ndarray,
        edge_index_array: np.ndarray,
        parsed_manifest: GraphSageBundleManifest,
    ) -> None:
        """Validate graph tensor dimensions against manifest architecture.

        Stage 2 requires input feature width and edge index layout checks so
        malformed bundles are rejected before GraphSAGE runtime wiring starts.
        Parameters: arrays and manifest come from validated bundle files.
        """

        if node_features_array.ndim != 2:
            raise GraphSageBundleValidationError(
                error_code="invalid_node_features_rank",
                message="node_features.npy must be two dimensional.",
                details={"actual_rank": int(node_features_array.ndim)},
            )
        if edge_index_array.ndim != 2:
            raise GraphSageBundleValidationError(
                error_code="invalid_edge_index_rank",
                message="edge_index.npy must be two dimensional.",
                details={"actual_rank": int(edge_index_array.ndim)},
            )
        if int(node_features_array.shape[1]) != parsed_manifest.model.input_dim:
            raise GraphSageBundleValidationError(
                error_code="feature_dimension_mismatch",
                message="node_features.npy feature dimension does not match manifest model.input_dim.",
                details={
                    "node_features_dimension": int(node_features_array.shape[1]),
                    "manifest_input_dim": parsed_manifest.model.input_dim,
                },
            )
        if int(edge_index_array.shape[0]) != 2:
            raise GraphSageBundleValidationError(
                error_code="invalid_edge_index_shape",
                message="edge_index.npy first dimension must equal 2.",
                details={"edge_index_shape": list(edge_index_array.shape)},
            )
