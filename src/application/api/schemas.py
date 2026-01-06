"""
API Schemas - Request and response models using Pydantic.

Pydantic validates data automatically and generates OpenAPI docs.

Request models: QueryRequest, FeedbackRequest
Response models: QueryResponse, ProductResponse, HealthResponse, etc.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# REQUEST MODELS
# ============================================================

class QueryRequest(BaseModel):
    """Request body for POST /query"""
    query: str = Field(
        ...,
        description="User query to process",
        min_length=1,
        max_length=1000
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "What wireless headphones do we have in stock?"},
                {"query": "Which products have the lowest profit margins?"},
            ]
        }
    }


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback"""
    query_id: int = Field(..., description="ID of the query to rate")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = Field(None, max_length=500, description="Optional comment")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query_id": 1, "rating": 5, "comment": "Very helpful response!"}
            ]
        }
    }


# ============================================================
# RESPONSE MODELS
# ============================================================

class ProductResponse(BaseModel):
    """Product data in responses."""
    product_id: str
    name: str
    category: str
    brand: str
    price: float
    stock_quantity: Optional[int] = None
    average_rating: Optional[float] = None
    description: Optional[str] = None
    margin_percent: Optional[float] = None


class SourceResponse(BaseModel):
    """Web source data in responses."""
    title: str
    url: str
    snippet: Optional[str] = None


class QueryResponse(BaseModel):
    """Response for POST /query"""
    answer: str = Field(..., description="AI-generated answer")
    reasoning: str = Field("", description="Why tools were selected")
    agents_used: list[str] = Field(default_factory=list, description="Agents used")
    products: list[ProductResponse] = Field(default_factory=list)
    sources: list[SourceResponse] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    query_id: Optional[int] = Field(None, description="ID for feedback")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "We have 3 wireless headphones in stock...",
                    "reasoning": "User asked about products in our catalog",
                    "agents_used": ["product_qa"],
                    "products": [{"product_id": "PROD-001", "name": "Wireless Bluetooth Headphones Pro"}],
                    "sources": [],
                    "confidence": 0.92,
                    "query_id": 1,
                }
            ]
        }
    }


class HistoryItemResponse(BaseModel):
    """Single query in history."""
    id: int
    query: str
    response: str
    agents_used: list[str]
    created_at: datetime


class HistoryResponse(BaseModel):
    """Response for GET /queries"""
    items: list[HistoryItemResponse]
    total: int
    page: int
    page_size: int


class FeedbackResponse(BaseModel):
    """Response for POST /feedback"""
    id: int
    query_id: int
    rating: int
    comment: Optional[str]
    created_at: datetime
    message: str = "Feedback submitted successfully"


class HealthResponse(BaseModel):
    """Response for GET /health"""
    status: str = "healthy"
    version: str = "0.1.0"
    components: dict = Field(default_factory=dict)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "0.1.0",
                    "components": {
                        "milvus": "connected",
                        "openai": "configured",
                    }
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    status_code: int
