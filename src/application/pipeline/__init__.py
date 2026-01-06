"""Data pipeline module for CSV loading, embedding, and ingestion."""

from src.application.pipeline.ingester import DataIngester
from src.application.pipeline.loader import CSVProductLoader
from src.application.pipeline.embedder import ProductEmbedder

__all__ = ["DataIngester", "CSVProductLoader", "ProductEmbedder"]
