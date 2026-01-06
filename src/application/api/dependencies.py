"""
Dependency Injection - Shared instances for API routes.

FastAPI's Depends() system allows routes to receive pre-configured
components without creating them manually.

How it works:
1. Functions like get_llm() create and cache instances
2. Routes use Annotated[Type, Depends(get_*)] to receive them
3. @lru_cache ensures only one instance is created

Usage in routes:
    @router.post("/query")
    async def query(
        request: QueryRequest,
        agent: AgentDep,  # Automatically injected
    ):
        result = await agent.query(request.query)
        return result
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.config import settings
from src.infrastructure.llm import OpenAILLM
from src.infrastructure.milvus import MilvusClient
from src.infrastructure.openai import OpenAIEmbedder
from src.infrastructure.search import TavilySearchClient
from src.application.agent import ProductResearchAgent
from src.application.pipeline import ProductEmbedder
from src.application.vector_store import MilvusProductRepository
from src.application.database import DatabaseSession, QueryHistoryRepository, FeedbackRepository


# ============================================================
# CACHED COMPONENT FACTORIES
# ============================================================

@lru_cache
def get_milvus_client() -> MilvusClient:
    """Get cached Milvus client (vector database)."""
    return MilvusClient()


@lru_cache
def get_embedder() -> OpenAIEmbedder:
    """Get cached embedder (text → vector)."""
    return OpenAIEmbedder()


@lru_cache
def get_llm() -> OpenAILLM:
    """Get cached LLM client (OpenAI GPT)."""
    return OpenAILLM()


@lru_cache
def get_search_client() -> TavilySearchClient:
    """Get cached web search client."""
    return TavilySearchClient()


# ============================================================
# COMPONENT FACTORIES WITH DEPENDENCIES
# ============================================================

def get_repository(
    milvus: Annotated[MilvusClient, Depends(get_milvus_client)],
) -> MilvusProductRepository:
    """Get product repository with Milvus client."""
    return MilvusProductRepository(milvus_client=milvus)


@lru_cache
def get_db_session() -> DatabaseSession:
    """Get cached database session manager."""
    db = DatabaseSession()
    db.create_tables()  # Create tables if not exist
    return db


def get_query_history_repository(
    db: Annotated[DatabaseSession, Depends(get_db_session)],
) -> QueryHistoryRepository:
    """Get query history repository with database session."""
    return QueryHistoryRepository(session=db)


def get_feedback_repository(
    db: Annotated[DatabaseSession, Depends(get_db_session)],
) -> FeedbackRepository:
    """Get feedback repository with database session."""
    return FeedbackRepository(session=db)


@lru_cache
def get_agent() -> ProductResearchAgent:
    """Get AI agent (LangGraph-based Planner + Workers)."""
    return ProductResearchAgent()


# ============================================================
# TYPE ALIASES FOR ROUTES
# Use these in route function parameters
# ============================================================

MilvusClientDep = Annotated[MilvusClient, Depends(get_milvus_client)]
EmbedderDep = Annotated[OpenAIEmbedder, Depends(get_embedder)]
LLMDep = Annotated[OpenAILLM, Depends(get_llm)]
SearchClientDep = Annotated[TavilySearchClient, Depends(get_search_client)]
RepositoryDep = Annotated[MilvusProductRepository, Depends(get_repository)]
AgentDep = Annotated[ProductResearchAgent, Depends(get_agent)]
DbSessionDep = Annotated[DatabaseSession, Depends(get_db_session)]
QueryHistoryRepoDep = Annotated[QueryHistoryRepository, Depends(get_query_history_repository)]
FeedbackRepoDep = Annotated[FeedbackRepository, Depends(get_feedback_repository)]
