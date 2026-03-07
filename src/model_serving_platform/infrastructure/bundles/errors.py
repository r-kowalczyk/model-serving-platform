"""Custom exceptions for GraphSAGE bundle validation."""

from typing import Any


class GraphSageBundleValidationError(ValueError):
    """Represent a structured validation failure for bundle startup checks.

    The service raises this exception when the bundle contract is not met.
    Keeping a typed exception makes startup failure messages precise and
    testable while allowing API boot logic to fail fast and stop safely.
    Parameters: error_code identifies the failure and details holds context.
    """

    def __init__(self, error_code: str, message: str, details: dict[str, Any]) -> None:
        """Initialise the validation error with structured detail fields.

        These values are used in tests and logs so operators can understand
        exactly which contract requirement failed during startup checks.
        Parameters: error_code is a machine key, message is plain text detail.
        """

        super().__init__(message)
        self.error_code = error_code
        self.details = details
