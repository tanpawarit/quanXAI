"""Price Analysis Tool - Margin calculations."""

from typing import Optional
from src.application.pipeline.embedder import ProductEmbedder
from src.application.vector_store.repository import MilvusProductRepository
from src.application.agent.tools.base import ToolResult


class PriceAnalysisTool:
    """Analyze product pricing and profit margins."""
    
    def __init__(self, repository: MilvusProductRepository = None, embedder: ProductEmbedder = None):
        self._repository = repository or MilvusProductRepository()
        self._embedder = embedder or ProductEmbedder()
    
    async def execute(self, category: Optional[str] = None, margin_threshold: Optional[float] = None, limit: int = 10, **kwargs) -> ToolResult:
        """Analyze pricing and margins."""
        try:
            # Use generic query to get products
            query = category or "products"
            query_embedding = await self._embedder.embed_query(query)
            products = await self._repository.search(
                query_embedding=query_embedding, query_text=query,
                limit=limit * 2, category=category,
            )
            
            analyzed = []
            total_margin = 0.0
            
            for p in products:
                margin = p.calculate_margin()
                if margin_threshold is not None and margin >= margin_threshold:
                    continue
                analyzed.append({
                    "product_id": p.product_id, "name": p.name, "category": p.category,
                    "brand": p.brand, "price": p.price, "cost": p.cost,
                    "margin_percent": round(margin, 2), "is_low_margin": p.is_low_margin(),
                })
                total_margin += margin
            
            analyzed.sort(key=lambda x: x["margin_percent"])
            analyzed = analyzed[:limit]
            
            avg_margin = total_margin / len(products) if products else 0.0
            low_margin_count = sum(1 for a in analyzed if a["is_low_margin"])
            
            answer = f"Analyzed {len(analyzed)} products. Avg margin: {avg_margin:.1f}%. Low margin (<40%): {low_margin_count}." if analyzed else "No products found."
            return ToolResult(answer=answer, products=analyzed, confidence=0.95)
        except Exception as e:
            return ToolResult(answer="", error=f"Price analysis failed: {str(e)}", confidence=0.0)
