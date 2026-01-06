"""
Ports (Interfaces) - Abstract contracts for external services.

This module defines interfaces using Python's ABC (Abstract Base Class).
Other modules implement these interfaces (adapters).

Why use interfaces?
- Decouples business logic from external services
- Makes testing easier (can mock interfaces)
- Allows swapping implementations without changing core code

Interface → Implementation mapping:
- IVectorStore  → vector_store/repository.py (Milvus)
- IEmbedder     → pkg/openai/embeddings.py
- ISearchClient → pkg/search/tavily.py
- ILLMProvider  → pkg/llm/openai.py
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.domain.entities import Product, SearchQuery, SearchResult


class IVectorStore(ABC):
    """
    Interface for vector database operations.
    
    Implemented by: MilvusProductRepository
    """
    
    @abstractmethod
    async def insert(self, products: list[Product]) -> int:
        """
        Insert products into the vector store.
        
        Returns: Number of products inserted
        """
        ...
    
    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        query_text: str,
        limit: int = 10,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> list[Product]:
        """
        Search products using hybrid search (vector + keyword).
        
        Args:
            query_embedding: Vector from embedding model
            query_text: Original text for BM25 search
            limit: Max results
            category/min_price/max_price: Filters
            
        Returns: Matching products
        """
        ...
    
    @abstractmethod
    async def upsert(self, products: list[Product]) -> int:
        """Insert or update existing products."""
        ...
    
    @abstractmethod
    async def delete(self, product_ids: list[str]) -> int:
        """Delete products by ID."""
        ...
    
    @abstractmethod
    async def get_all_ids(self) -> list[str]:
        """Get all product IDs in the store."""
        ...


class IEmbedder(ABC):
    """
    Interface for text embedding generation.
    
    Implemented by: OpenAIEmbedder
    """
    
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...
    
    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for one text."""
        ...
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector size (e.g., 1536 for OpenAI)."""
        ...


class ISearchClient(ABC):
    """
    Interface for web search.
    
    Implemented by: TavilySearchClient
    """
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search the web.
        
        Returns: List of results with title, url, content
        """
        ...


class ILLMProvider(ABC):
    """
    Interface for LLM (Large Language Model) interactions.
    
    Implemented by: OpenAILLM
    """
    
    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
    ) -> dict:
        """
        Send a chat completion request.
        
        Args:
            messages: Chat history [{"role": "user", "content": "..."}]
            tools: Optional function definitions for tool calling
            temperature: Randomness (0=deterministic, 1=creative)
            
        Returns: Response with content and optional tool_calls
        """
        ...
