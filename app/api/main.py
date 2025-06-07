# app/api/main.py

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import time # For X-Process-Time header

# Import settings and helper functions from the core module
from app.core.config import settings, ensure_directories
from app.core.embedding_manager import get_embedding_manager # Import getter for embedding manager

# Import API routers from app.api.v1.endpoints
from app.api.v1.endpoints import chat as chat_router_v1
from app.api.v1.endpoints import documents as documents_router_v1

# --- Application Metadata ---
TITLE = settings.APP_NAME
VERSION = settings.APP_VERSION
DESCRIPTION = f"""
{settings.APP_NAME} - API for interacting with a RAG (Retrieval Augmented Generation) system.

This API allows you to:
- Upload and manage documents for the knowledge base.
- Ask questions and get answers based on the ingested documents.
"""

# Configure logging
logging.basicConfig(level=logging.INFO if settings.ENVIRONMENT != "development" else logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Application Lifecycle Events ---
async def startup_event():
    """
    Actions to perform when the application starts.
    """
    logger.info("Application startup...")
    logger.info(f"Running in {settings.ENVIRONMENT} mode.")
    logger.info(f"Debug mode: {settings.DEBUG}")

    try:
        ensure_directories()
        logger.info("Checked/Ensured necessary data directories exist.")
    except Exception as e:
        logger.error(f"Error ensuring directories: {e}")

    # Explicitly load the embedding model on startup
    try:
        embedding_manager = get_embedding_manager()
        embedding_manager.load_model() # This will log success or failure
        logger.info(f"Embedding model '{embedding_manager._model_name}' loaded with dimension {embedding_manager.get_embedding_dimension()}.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load embedding model on startup: {e}", exc_info=True)
        # Depending on your app's requirements, you might want to exit if the embedding model fails to load.
        # For now, we'll just log the error. Consider raising an exception to stop startup.
        # raise RuntimeError("Failed to initialize embedding model, application cannot start.") from e


    # Placeholder: Initialize LLM client (if not handled by RAGPipeline's lazy loading)
    logger.info("LLM client (placeholder for explicit startup init) initialized.")

    # Placeholder: Initialize Vector Store client (if not handled by RAGPipeline's lazy loading)
    # e.g., vector_store = get_vector_store_client(); await vector_store.initialize()
    logger.info("Vector Store client (placeholder for explicit startup init) initialized.")
    logger.info("Application startup complete.")

async def shutdown_event():
    """
    Actions to perform when the application shuts down.
    """
    logger.info("Application shutdown...")
    # Placeholder: Cleanup resources
    logger.info("Resources (placeholder) cleaned up.")
    logger.info("Application shutdown complete.")

# --- FastAPI Application Instance ---
app = FastAPI(
    title=TITLE,
    version=VERSION,
    description=DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    on_startup=[startup_event],
    on_shutdown=[shutdown_event]
)

# --- Middleware ---
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin).strip() for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS enabled for origins: {settings.BACKEND_CORS_ORIGINS}")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# --- Custom Exception Handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

# --- API Routers ---
app.include_router(
    chat_router_v1.router,
    prefix=f"{settings.API_V1_STR}/chat",
    tags=["Chat (V1)"]
)
logger.info(f"Included Chat router at prefix: {settings.API_V1_STR}/chat")

app.include_router(
    documents_router_v1.router,
    prefix=f"{settings.API_V1_STR}/documents",
    tags=["Documents (V1)"]
)
logger.info(f"Included Documents router at prefix: {settings.API_V1_STR}/documents")


# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "documentation_urls": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": app.openapi_url
        }
    }

# --- Main execution for local development (using uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly from main.py (for development/debugging)...")
    # Ensure .env is loaded if running directly (though config.py should handle it)
    from dotenv import load_dotenv
    from pathlib import Path
    project_root_main = Path(__file__).resolve().parent.parent.parent # Adjust if main.py moves
    env_path_main = project_root_main / ".env"
    if env_path_main.exists():
        load_dotenv(dotenv_path=env_path_main)
        logger.info(f".env file loaded from {env_path_main} for direct run.")

    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
