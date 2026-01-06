"""
FastAPI Application - Main entry point for the API.

Endpoints:
- POST /query - Ask the AI agent a question
- GET /queries - View query history
- POST /feedback - Submit feedback on a response
- GET /health - Check system status

Run with:
    uv run uvicorn src.application.api.main:app --reload
"""

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.domain.logger import configure_logging, get_logger
from src.application.api.routes import query_router, history_router, feedback_router, health_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on app startup and shutdown.
    
    Startup: Configure logging, log connection info
    Shutdown: Log shutdown
    """
    configure_logging()
    
    logger.info(
        "api_startup",
        milvus_db=settings.milvus_db_path,
        collection=settings.milvus_collection_name,
    )
    
    yield 
    
    logger.info("api_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Product Research Assistant API",
        description=(
            "AI-powered product research assistant for e-commerce. "
            "Uses RAG for product catalog search, web search for market trends, "
            "and deterministic price analysis."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",      # Swagger UI
        redoc_url="/redoc",    # ReDoc UI
    )
    
    # Allow cross-origin requests (configure for production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register route handlers
    app.include_router(query_router, tags=["Query"])
    app.include_router(history_router, tags=["History"])
    app.include_router(feedback_router, tags=["Feedback"])
    app.include_router(health_router, tags=["Health"])
    
    return app


# Create the app instance
app = create_app()


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - shows available endpoints."""
    return {
        "message": "Product Research Assistant API",
        "docs": "/docs",
        "health": "/health",
    }
