"""History endpoint - GET /queries."""

from fastapi import APIRouter, Query

from src.application.api.schemas import HistoryResponse, HistoryItemResponse
from src.application.api.dependencies import QueryHistoryRepoDep

router = APIRouter()


@router.get(
    "/queries",
    response_model=HistoryResponse,
    summary="Get query history",
    description="Retrieve history of past queries",
)
async def get_query_history(
    history_repo: QueryHistoryRepoDep,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
) -> HistoryResponse:
    """
    Get paginated query history from database.
    """
    # Get from database
    items, total = history_repo.get_all(page=page, page_size=page_size)
    
    # Convert to response model
    history_items = [
        HistoryItemResponse(
            id=item.id,
            query=item.query,
            response=item.response,
            agents_used=item.agents_used or [],
            created_at=item.created_at,
        )
        for item in items
    ]
    
    return HistoryResponse(
        items=history_items,
        total=total,
        page=page,
        page_size=page_size,
    )
