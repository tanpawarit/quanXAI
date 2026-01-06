"""Health endpoint - GET /health."""

from fastapi import APIRouter
from sqlalchemy import text

from src.config import settings
from src.application.api.schemas import HealthResponse
from src.application.api.dependencies import MilvusClientDep, DbSessionDep
from src.infrastructure.milvus import ProductCollectionSchema

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API",
)
async def health_check(
    milvus: MilvusClientDep,
    db: DbSessionDep,
) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the status of all components:
    - Milvus connection (checks for collection existence)
    - Database connection (executes SELECT 1)
    - OpenAI API configuration
    - Tavily API configuration
    """
    # Check Milvus
    try:
        milvus.connect()
        has_collection = milvus.has_collection(ProductCollectionSchema.COLLECTION_NAME)
        milvus_status = "connected" if has_collection else "connected_empty"
    except Exception:
        milvus_status = "error"
    
    # Check Database
    try:
        with db.get_session() as session:
            session.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "error"

    components = {
        "milvus": milvus_status,
        "database": db_status,
        "openai": "configured" if settings.openai_api_key else "not_configured",
        "tavily": "configured" if settings.tavily_api_key else "not_configured",
    }
    
    # Check if all critical components are working
    critical_components = ["milvus", "database", "openai"]
    all_healthy = all(
        components[k].startswith("connected") or components[k] == "configured"
        for k in critical_components
    )
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="0.1.0",
        components=components,
    )
