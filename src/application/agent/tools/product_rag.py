"""Product RAG Tool - Search products using vector similarity."""

from typing import Optional
from src.application.pipeline.embedder import ProductEmbedder
from src.application.vector_store.repository import MilvusProductRepository
from src.application.agent.tools.base import ToolResult


class ProductRAGTool:
    """Search products in the catalog using semantic search."""
    
    def __init__(self, repository: MilvusProductRepository = None, embedder: ProductEmbedder = None):
        self._repository = repository or MilvusProductRepository()
        self._embedder = embedder or ProductEmbedder()
    
    async def execute(self, query: str, category: Optional[str] = None, min_price: Optional[float] = None, max_price: Optional[float] = None, limit: int = 10, **kwargs) -> ToolResult:
        """Search products in the catalog."""
        try:
            query_embedding = await self._embedder.embed_query(query)
            products = await self._repository.search(
                query_embedding=query_embedding, query_text=query, limit=limit,
                category=category, min_price=min_price, max_price=max_price,
            )
            
            product_dicts = []
            for p in products:
                desc = p.description[:200] + "..." if len(p.description) > 200 else p.description
                product_dicts.append({
                    "product_id": p.product_id, "name": p.name, "category": p.category,
                    "brand": p.brand, "description": desc, "price": p.price,
                    "stock_quantity": p.stock_quantity, "average_rating": p.average_rating,
                    "in_stock": p.is_in_stock(),
                })
            
            answer = f"Found {len(products)} products matching your query." if products else "No products found."
            return ToolResult(answer=answer, products=product_dicts, confidence=0.9 if products else 0.3)
        except Exception as e:
            return ToolResult(answer="", error=f"Search failed: {str(e)}", confidence=0.0)
