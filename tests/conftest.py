"""
Shared fixtures for agent tests.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.application.vector_store.repository import MilvusProductRepository
from src.application.pipeline.embedder import ProductEmbedder
from src.domain.entities import Product

@pytest.fixture
def sample_product():
    """Create a sample product entity."""
    return Product(
        product_id="PROD-001",
        name="Test Wireless Headphones",
        description="High quality wireless headphones with noise cancelling.",
        category="Electronics",
        brand="TestBrand",
        price=99.99,
        cost=50.0,
        stock_quantity=10,
        monthly_sales=100,
        average_rating=4.5
    )

@pytest.fixture
def mock_repository(sample_product):
    """Mock Milvus repository."""
    repo = MagicMock(spec=MilvusProductRepository)
    # Default behavior: return sample product
    repo.search = AsyncMock(return_value=[sample_product])
    return repo

@pytest.fixture
def mock_embedder():
    """Mock product embedder."""
    embedder = MagicMock(spec=ProductEmbedder)
    # Default behavior: return dummy embedding
    embedder.embed_query = AsyncMock(return_value=[0.1] * 384)
    return embedder
