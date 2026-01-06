"""API routes module."""

from src.application.api.routes.query import router as query_router
from src.application.api.routes.history import router as history_router
from src.application.api.routes.feedback import router as feedback_router
from src.application.api.routes.health import router as health_router

__all__ = [
    "query_router",
    "history_router",
    "feedback_router",
    "health_router",
]
