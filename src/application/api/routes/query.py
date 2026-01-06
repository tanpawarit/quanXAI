"""Query endpoint - POST /query."""

from fastapi import APIRouter, HTTPException

from src.application.api.schemas import QueryRequest, QueryResponse, ProductResponse, SourceResponse
from src.application.api.dependencies import AgentDep, QueryHistoryRepoDep

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Process a query",
    description="Send a query to the AI agent for processing",
)
async def process_query(
    request: QueryRequest,
    agent: AgentDep,
    history_repo: QueryHistoryRepoDep,
) -> QueryResponse:
    """
    Process a user query through the AI agent.
    
    The agent will:
    1. Analyze the query
    2. Select appropriate tool(s)
    3. Execute tools
    4. Synthesize a response
    5. Save to query history
    """
    try:
        # Fetch recent history for context
        history = history_repo.get_recent(limit=5)
        
        # Format history for agent
        chat_history = []
        # History is returned latest first, so we reverse it to be chronological
        for entry in reversed(history):
            chat_history.append({"role": "user", "content": entry.query})
            chat_history.append({"role": "assistant", "content": entry.response})
            
        result = await agent.query(request.query, chat_history)
        
        # Convert products to response model
        products = [
            ProductResponse(
                product_id=p.get("product_id", ""),
                name=p.get("name", ""),
                category=p.get("category", ""),
                brand=p.get("brand", ""),
                price=p.get("price", 0.0),
                stock_quantity=p.get("stock_quantity"),
                average_rating=p.get("average_rating"),
                description=p.get("description"),
                margin_percent=p.get("margin_percent"),
            )
            for p in result.get("products", [])
        ]
        
        # Convert sources to response model
        sources = [
            SourceResponse(
                title=s.get("title", ""),
                url=s.get("url", ""),
                snippet=s.get("snippet"),
            )
            for s in result.get("sources", [])
        ]
        
        answer = result["answer"]
        reasoning = result.get("reasoning", "")
        agents_used = result.get("agents_used", [])
        confidence = result.get("confidence", 0.0)
        
        # Save to database
        query_entry = history_repo.create(
            query=request.query,
            response=answer,
            reasoning=reasoning,
            agents_used=agents_used,
            confidence=confidence,
        )
        
        return QueryResponse(
            answer=answer,
            reasoning=reasoning,
            agents_used=agents_used,
            products=products,
            sources=sources,
            confidence=confidence,
            query_id=query_entry.id,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}",
        )
