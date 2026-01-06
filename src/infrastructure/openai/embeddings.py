"""OpenAI embedding service using text-embedding-3-small."""

import asyncio
from typing import Optional

from openai import AsyncOpenAI

from src.config import settings
from src.domain.exceptions import EmbeddingError
from src.domain.ports import IEmbedder


class OpenAIEmbedder(IEmbedder):
    """
    OpenAI embedding service implementation.
    
    Uses text-embedding-3-small (1536 dimensions) for generating embeddings.
    Supports batch processing with configurable batch size.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        batch_size: int = None,
    ):
        """
        Initialize OpenAI embedder.
        
        Args:
            api_key: OpenAI API key (default from settings)
            model: Embedding model name (default text-embedding-3-small)
            batch_size: Batch size for bulk embedding (default 100)
        """
        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.openai_embedding_model
        self._batch_size = batch_size or settings.embedding_batch_size_int
        self._client: Optional[AsyncOpenAI] = None
    
    @property
    def client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension (1536 for text-embedding-3-small)."""
        return settings.embedding_dimension_int
    
    async def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (1536 dimensions)
            
        Raises:
            EmbeddingError: If embedding fails
        """
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self._model,
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"Failed to embed text: {e}")
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding fails
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            
            try:
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self._model,
                )
                
                # Sort by index to maintain order
                batch_embeddings = [None] * len(batch)
                for item in response.data:
                    batch_embeddings[item.index] = item.embedding
                
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                raise EmbeddingError(f"Failed to embed batch {i//self._batch_size}: {e}")
            
            # Small delay between batches to respect rate limits
            if i + self._batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return all_embeddings
    
    async def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
