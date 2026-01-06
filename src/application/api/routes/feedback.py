"""Feedback endpoint - POST /feedback."""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from src.application.api.schemas import FeedbackRequest, FeedbackResponse
from src.application.api.dependencies import FeedbackRepoDep

router = APIRouter()


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback",
    description="Submit feedback for a query response",
)
async def submit_feedback(
    request: FeedbackRequest,
    feedback_repo: FeedbackRepoDep,
) -> FeedbackResponse:
    """
    Submit feedback for a query response.
    """
    try:
        feedback = feedback_repo.create(
            query_id=request.query_id,
            rating=request.rating,
            comment=request.comment,
        )
        
        return FeedbackResponse(
            id=feedback.id,
            query_id=feedback.query_id,
            rating=feedback.rating,
            comment=feedback.comment,
            created_at=feedback.created_at,
            message="Feedback submitted successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}",
        )
