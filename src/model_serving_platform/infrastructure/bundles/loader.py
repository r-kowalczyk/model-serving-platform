"""GraphSAGE bundle loader and startup contract validator."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
OPTIONAL_CACHE_FILE_NAMES = ("resolver_cache.json", "interaction_cache.json")

bundle_loader_logger = logging.getLogger(
    "model_serving_platform.infrastructure.bundles.loader"
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

        resolved_bundle_directory_path = bundle_directory_path.resolve()
        bundle_directory_diagnostics = self._collect_bundle_directory_diagnostics(
            bundle_directory_path=resolved_bundle_directory_path
        )
        bundle_loader_logger.info(
            "bundle_validation_started",
            extra={
                "bundle_directory_path": str(resolved_bundle_directory_path),
                "bundle_directory_exists": bundle_directory_diagnostics[
                    "bundle_directory_exists"
                ],
                "bundle_directory_is_dir": bundle_directory_diagnostics[
                    "bundle_directory_is_dir"
                ],
                "bundle_directory_is_readable": bundle_directory_diagnostics[
                    "bundle_directory_is_readable"
                ],
                "discovered_bundle_file_names": bundle_directory_diagnostics[
                    "discovered_bundle_file_names"
                ],
                "list_directory_error": bundle_directory_diagnostics[
                    "list_directory_error"
                ],
            },
        )
        self._validate_bundle_directory_access(
            bundle_directory_path=resolved_bundle_directory_path,
            bundle_directory_diagnostics=bundle_directory_diagnostics,
        )
        self._validate_required_files(
            bundle_directory_path=bundle_directory_path,
            bundle_directory_diagnostics=bundle_directory_diagnostics,
        )
        self._initialise_optional_cache_files(
            bundle_directory_path=resolved_bundle_directory_path
        )
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

    def _validate_required_files(
        self,
        bundle_directory_path: Path,
        bundle_directory_diagnostics: dict[str, object],
    ) -> None:
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
            discovered_bundle_file_names = cast(
                list[str], bundle_directory_diagnostics["discovered_bundle_file_names"]
            )
            discovered_file_names_display = (
                ", ".join(discovered_bundle_file_names)
                if discovered_bundle_file_names
                else "<none>"
            )
            bundle_loader_logger.error(
                "bundle_required_files_missing",
                extra={
                    "bundle_directory_path": str(bundle_directory_path),
                    "missing_file_names": missing_file_names,
                    "discovered_bundle_file_names": discovered_bundle_file_names,
                },
            )
            raise GraphSageBundleValidationError(
                error_code="missing_required_files",
                message="Bundle is missing required files: "
                + ", ".join(missing_file_names)
                + ". Discovered files: "
                + discovered_file_names_display,
                details={
                    "bundle_directory_path": str(bundle_directory_path),
                    "missing_file_names": missing_file_names,
                    "bundle_directory_exists": bundle_directory_diagnostics[
                        "bundle_directory_exists"
                    ],
                    "bundle_directory_is_dir": bundle_directory_diagnostics[
                        "bundle_directory_is_dir"
                    ],
                    "bundle_directory_is_readable": bundle_directory_diagnostics[
                        "bundle_directory_is_readable"
                    ],
                    "discovered_bundle_file_names": discovered_bundle_file_names,
                    "list_directory_error": bundle_directory_diagnostics[
                        "list_directory_error"
                    ],
                },
            )

    def _validate_bundle_directory_access(
        self,
        bundle_directory_path: Path,
        bundle_directory_diagnostics: dict[str, object],
    ) -> None:
        """Ensure bundle directory exists, is a directory, and is readable.

        Deployment mount issues can present as missing required files even when
        artefacts exist in storage, so this preflight check fails with explicit
        diagnostics before required file validation is attempted.
        Parameters: bundle_directory_path points to the expected bundle root.
        """

        bundle_directory_exists = bool(
            bundle_directory_diagnostics["bundle_directory_exists"]
        )
        bundle_directory_is_dir = bool(
            bundle_directory_diagnostics["bundle_directory_is_dir"]
        )
        bundle_directory_is_readable = bool(
            bundle_directory_diagnostics["bundle_directory_is_readable"]
        )
        if (
            bundle_directory_exists
            and bundle_directory_is_dir
            and bundle_directory_is_readable
        ):
            return
        raise GraphSageBundleValidationError(
            error_code="invalid_bundle_directory",
            message=(
                "Bundle directory is not accessible. "
                f"path={bundle_directory_path}; "
                f"exists={bundle_directory_exists}; "
                f"is_dir={bundle_directory_is_dir}; "
                f"is_readable={bundle_directory_is_readable}; "
                "list_directory_error="
                f"{bundle_directory_diagnostics['list_directory_error']}"
            ),
            details={
                "bundle_directory_path": str(bundle_directory_path),
                "bundle_directory_exists": bundle_directory_exists,
                "bundle_directory_is_dir": bundle_directory_is_dir,
                "bundle_directory_is_readable": bundle_directory_is_readable,
                "list_directory_error": bundle_directory_diagnostics[
                    "list_directory_error"
                ],
                "discovered_bundle_file_names": bundle_directory_diagnostics[
                    "discovered_bundle_file_names"
                ],
            },
        )

    def _collect_bundle_directory_diagnostics(
        self, bundle_directory_path: Path
    ) -> dict[str, object]:
        """Collect visibility and listing diagnostics for bundle directory.

        Startup diagnostics should include concrete path state because mount
        and permission issues can hide files without changing configuration.
        Parameters: bundle_directory_path points to the expected bundle root.
        """

        bundle_directory_exists = bundle_directory_path.exists()
        bundle_directory_is_dir = (
            bundle_directory_path.is_dir() if bundle_directory_exists else False
        )
        bundle_directory_is_readable = (
            os.access(bundle_directory_path, os.R_OK | os.X_OK)
            if bundle_directory_exists and bundle_directory_is_dir
            else False
        )
        discovered_bundle_file_names: list[str] = []
        list_directory_error: str | None = None
        if bundle_directory_exists and bundle_directory_is_dir:
            try:
                discovered_bundle_file_names = sorted(
                    file_path.name
                    for file_path in bundle_directory_path.iterdir()
                    if file_path.is_file()
                )
            except Exception as directory_listing_error:
                list_directory_error = str(directory_listing_error)
        return {
            "bundle_directory_exists": bundle_directory_exists,
            "bundle_directory_is_dir": bundle_directory_is_dir,
            "bundle_directory_is_readable": bundle_directory_is_readable,
            "discovered_bundle_file_names": discovered_bundle_file_names,
            "list_directory_error": list_directory_error,
        }

    def _initialise_optional_cache_files(self, bundle_directory_path: Path) -> None:
        """Create optional cache files when they are absent at startup.

        Cache files are treated as optional contract artefacts because serving
        can create an empty cache and continue with deterministic behaviour.
        Parameters: bundle_directory_path points to the expected bundle root.
        """

        for optional_cache_file_name in OPTIONAL_CACHE_FILE_NAMES:
            optional_cache_file_path = bundle_directory_path / optional_cache_file_name
            if optional_cache_file_path.exists():
                continue
            # Writing empty JSON objects here keeps startup deterministic when
            # bundles are uploaded without optional cache artefacts.
            try:
                optional_cache_file_path.write_text("{}", encoding="utf-8")
            except Exception as cache_initialisation_error:
                raise GraphSageBundleValidationError(
                    error_code="optional_cache_initialisation_failed",
                    message="Unable to initialise optional cache file: "
                    + optional_cache_file_name,
                    details={
                        "bundle_directory_path": str(bundle_directory_path),
                        "optional_cache_file_name": optional_cache_file_name,
                        "cache_initialisation_error": str(cache_initialisation_error),
                    },
                ) from cache_initialisation_error
            bundle_loader_logger.info(
                "bundle_optional_cache_file_created",
                extra={
                    "bundle_directory_path": str(bundle_directory_path),
                    "optional_cache_file_name": optional_cache_file_name,
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
