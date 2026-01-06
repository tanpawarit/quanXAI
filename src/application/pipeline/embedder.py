"""
Product Embedder - Convert text to vectors for search.

What are embeddings?
- Text is converted to a list of numbers (vector)
- Similar texts have similar vectors
- This enables semantic search (find by meaning, not just keywords)

Example:
- "wireless headphones" and "bluetooth earbuds" will have similar vectors
- even though they share no common words

Default model: OpenAI text-embedding-3-small (1536 dimensions)
"""

import asyncio
from typing import Optional

from src.domain.entities import Product
from src.domain.exceptions import EmbeddingError
from src.domain.logger import get_logger
from src.domain.ports import IEmbedder
from src.infrastructure.openai import OpenAIEmbedder

logger = get_logger(__name__)


class ProductEmbedder:
    """Generate embeddings for products and queries."""
    
    def __init__(self, embedder: IEmbedder = None):
        """Initialize with embedding provider (default: OpenAI)."""
        self._embedder = embedder or OpenAIEmbedder()
    
    async def embed_products(
        self,
        products: list[Product],
    ) -> list[Product]:
        """
        Add embeddings to products.
        
        Process:
        1. Extract text from each product (name + description + brand + category)
        2. Send all texts to OpenAI in batch
        3. Assign resulting vectors to each product
        
        Returns: Same products with embedding field filled in
        """
        if not products:
            return products
        
        # Step 1: Extract text for embedding
        texts = [p.to_search_text() for p in products]
        
        logger.debug("generating_embeddings", count=len(products))
        
        try:
            # Step 2: Generate embeddings (batch API call)
            embeddings = await self._embedder.embed(texts)
            
            # Step 3: Assign embeddings to products
            for product, embedding in zip(products, embeddings):
                product.embedding = embedding
            
            logger.debug("embeddings_generated", count=len(embeddings))
            
            return products
            
        except Exception as e:
            logger.error("embedding_failed", error=str(e))
            raise EmbeddingError(f"Failed to embed products: {e}")
    
    def embed_products_sync(
        self,
        products: list[Product],
    ) -> list[Product]:
        """Synchronous version of embed_products."""
        return asyncio.run(self.embed_products(products))
    
    async def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.
        
        Returns: Vector that can be compared with product embeddings
        """
        return await self._embedder.embed_single(query)
    
    @property
    def dimension(self) -> int:
        """Vector size (e.g., 1536 for OpenAI)."""
        return self._embedder.dimension
