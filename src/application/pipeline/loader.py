"""
CSV Product Loader - Load products from CSV file.

Input: data/products_catalog.csv
Output: List of Product entities

CSV columns:
- product_id, product_name, category, brand
- description, current_price, cost
- stock_quantity, monthly_sales
- average_rating, review_count
- supplier, last_updated

Uses LangChain's CSVLoader for parsing.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import CSVLoader

from src.config import settings
from src.domain.entities import Product
from src.domain.exceptions import PipelineError
from src.domain.logger import get_logger

logger = get_logger(__name__)


class CSVProductLoader:
    """Load products from CSV file into Product entities."""
    
    def __init__(self, file_path: str = None):
        """
        Initialize loader.
        
        Args:
            file_path: Path to CSV (default: from settings)
        """
        self.file_path = file_path or settings.products_csv_path
    
    def load(self) -> list[Product]:
        """
        Load all products from CSV.
        
        Returns: List of Product entities
        """
        # Check file exists
        if not Path(self.file_path).exists():
            raise PipelineError(f"CSV file not found: {self.file_path}", stage="load")
        
        try:
            # Use LangChain CSVLoader
            loader = CSVLoader(
                file_path=self.file_path,
                encoding="utf-8",
            )
            documents = loader.load()
            
            # Convert each row to Product
            products = []
            for doc in documents:
                product = self._document_to_product(doc)
                if product:
                    products.append(product)
            
            return products
            
        except Exception as e:
            raise PipelineError(f"Failed to load CSV: {e}", stage="load")
    
    def _document_to_product(self, doc) -> Optional[Product]:
        """
        Convert LangChain Document to Product.
        
        CSVLoader puts data in page_content as "key: value" format.
        """
        try:
            # Parse page_content (format: "key: value\nkey: value\n...")
            content = doc.page_content
            data = {}
            for line in content.split("\n"):
                if ": " in line:
                    key, value = line.split(": ", 1)
                    data[key.strip()] = value.strip()
            
            # Skip rows with empty product_id
            product_id = data.get("product_id", "").strip()
            if not product_id:
                return None
            
            # Parse date field
            last_updated = None
            if data.get("last_updated"):
                try:
                    last_updated = datetime.strptime(
                        data["last_updated"], "%Y-%m-%d"
                    ).date()
                except ValueError:
                    pass  # Skip invalid dates
            
            # Map CSV columns to Product fields
            return Product(
                product_id=product_id,
                name=data.get("product_name", ""),
                category=data.get("category", ""),
                brand=data.get("brand", ""),
                description=data.get("description", ""),
                price=float(data.get("current_price", 0) or 0),
                cost=float(data.get("cost", 0) or 0),
                stock_quantity=int(float(data.get("stock_quantity", 0) or 0)),
                monthly_sales=int(float(data.get("monthly_sales", 0) or 0)),
                average_rating=float(data.get("average_rating", 0)) if data.get("average_rating") else None,
                review_count=int(float(data.get("review_count", 0) or 0)),
                supplier=data.get("supplier", ""),
                last_updated=last_updated,
            )
            
        except Exception as e:
            logger.warning("skipping_invalid_row", error=str(e))
            return None


def load_products(file_path: str = None) -> list[Product]:
    """Quick function to load products."""
    loader = CSVProductLoader(file_path)
    return loader.load()
