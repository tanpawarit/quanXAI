"""
Domain Entities - Core business objects for the Product Research Assistant.

This module contains the main data structures used throughout the application.
These are pure Python dataclasses with no external dependencies.

Main entities:
- Product: Represents a product in the catalog
- SearchQuery: A search request with filters
- SearchResult: Search response with matching products
- QueryHistory: Tracks user queries
- Feedback: User feedback on responses
"""

import hashlib

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class ProductCategory(str, Enum):
    """Available product categories in the catalog."""
    ELECTRONICS = "Electronics"
    SPORTS_FITNESS = "Sports & Fitness"
    HOME_GARDEN = "Home & Garden"
    FASHION = "Fashion"
    HEALTH_WELLNESS = "Health & Wellness"
    ACCESSORIES = "Accessories"
    KITCHEN_DINING = "Kitchen & Dining"


@dataclass
class Product:
    """
    A product in the catalog.
    
    This is the main entity that stores all product information
    including pricing, stock, and computed embeddings for search.
    
    Example:
        product = Product(
            product_id="PROD-001",
            name="Wireless Headphones",
            category="Electronics",
            brand="AudioMax",
            description="High quality wireless headphones",
            price=99.99,
            cost=45.00,
            stock_quantity=50,
            monthly_sales=120
        )
        margin = product.calculate_margin()  # Returns: 54.95
    """
    # Required fields
    product_id: str
    name: str
    category: str
    brand: str
    description: str
    price: float          # Selling price to customer
    cost: float           # Cost from supplier
    stock_quantity: int   # Current stock level
    monthly_sales: int    # Units sold per month
    
    # Optional fields
    average_rating: Optional[float] = None  # 1-5 star rating
    review_count: int = 0
    supplier: str = ""
    last_updated: Optional[date] = None
    
    # Vector embedding for semantic search
    embedding: Optional[list[float]] = field(default=None, repr=False)
    
    def calculate_margin(self) -> float:
        """
        Calculate profit margin as a percentage.
        
        Formula: ((price - cost) / price) * 100
        
        Returns:
            Margin percentage (e.g., 50.0 means 50% margin)
        """
        if self.price <= 0:
            return 0.0
        return ((self.price - self.cost) / self.price) * 100
    
    def is_low_margin(self, threshold: float = 40.0) -> bool:
        """Check if margin is below threshold (default 40%)."""
        return self.calculate_margin() < threshold
    
    def is_in_stock(self) -> bool:
        """Check if product has stock available."""
        return self.stock_quantity > 0
    
    def is_bestseller(self, sales_threshold: int = 100) -> bool:
        """Check if monthly sales exceed threshold."""
        return self.monthly_sales >= sales_threshold
    
    def to_search_text(self) -> str:
        """
        Create text for embedding generation and BM25 search.
        
        Includes product_id so hybrid search can match exact IDs.
        """
        return f"{self.product_id}. {self.name}. {self.description}. Brand: {self.brand}. Category: {self.category}."
    
    def content_hash(self) -> str:
        """
        Generate MD5 hash of searchable content.
        
        Used for incremental updates to detect actual content changes.
        Only re-embeds products when this hash changes.
        
        Returns:
            32-character MD5 hex string
        """
        return hashlib.md5(self.to_search_text().encode()).hexdigest()


@dataclass
class SearchQuery:
    """
    A search request with optional filters.
    
    Example:
        query = SearchQuery(
            text="wireless headphones",
            category_filter="Electronics",
            max_price=100.0
        )
    """
    text: str                                  # Search text
    category_filter: Optional[str] = None      # Filter by category
    brand_filter: Optional[str] = None         # Filter by brand
    min_price: Optional[float] = None          # Minimum price
    max_price: Optional[float] = None          # Maximum price
    in_stock_only: bool = False                # Only show in-stock items
    limit: int = 10                            # Max results to return


@dataclass
class SearchResult:
    """Search response containing matching products."""
    products: list[Product]    # Matching products
    query: str                 # Original query text
    total_found: int           # Total matches found
    confidence: float = 0.0    # Search confidence (0-1)


@dataclass
class QueryHistory:
    """Record of a user query and response."""
    id: Optional[int] = None
    query: str = ""
    response: str = ""
    agents_used: list[str] = field(default_factory=list)
    reasoning: str = ""
    created_at: Optional[str] = None


@dataclass
class Feedback:
    """User feedback for a query response."""
    id: Optional[int] = None
    query_id: int = 0
    rating: int = 0          # 1-5 stars
    comment: str = ""
    created_at: Optional[str] = None
