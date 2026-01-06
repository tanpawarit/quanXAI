"""Milvus vector database client and schema."""

from src.infrastructure.milvus.client import MilvusClient
from src.infrastructure.milvus.schema import ProductCollectionSchema

__all__ = ["MilvusClient", "ProductCollectionSchema"]
