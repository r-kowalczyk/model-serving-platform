"""Application entry point for the model serving platform service."""

from model_serving_platform.api.app import create_app
from model_serving_platform.config.settings import ServiceSettings

app = create_app()


def run() -> None:
    """Run the HTTP service with process-level settings.

    This function is the local entrypoint used by developers and container
    runtime commands. It loads settings from environment variables and starts
    a single Uvicorn process that serves the FastAPI application.
    Parameters: none.
    """
    import uvicorn

    service_settings = ServiceSettings()
    uvicorn.run(
        "model_serving_platform.main:app",
        host=service_settings.host,
        port=service_settings.port,
        reload=service_settings.reload,
    )
