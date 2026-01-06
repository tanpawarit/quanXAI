"""Milvus product repository implementing IVectorStore."""

from datetime import datetime
from typing import Optional

from src.config import settings
from src.domain.entities import Product
from src.domain.exceptions import VectorStoreError
from src.domain.ports import IVectorStore
from src.infrastructure.milvus import MilvusClient, ProductCollectionSchema


class MilvusProductRepository(IVectorStore):
    """
    Product repository using Milvus vector database.
    
    Implements IVectorStore interface for:
    - Hybrid search (dense + sparse)
    - CRUD operations
    - Metadata filtering
    """
    
    def __init__(self, milvus_client: MilvusClient = None):
        """
        Initialize repository.
        
        Args:
            milvus_client: Milvus client (default: MilvusClient)
        """
        self._milvus = milvus_client or MilvusClient()
        self._collection_name = ProductCollectionSchema.COLLECTION_NAME
    
    async def insert(self, products: list[Product]) -> int:
        """
        Insert products into Milvus.
        
        Args:
            products: List of products with embeddings
            
        Returns:
            Number of products inserted
        """
        if not products:
            return 0
        
        try:
            self._milvus.connect()
            
            data = [
                ProductCollectionSchema.product_to_dict(p)
                for p in products
            ]
            
            result = self._milvus.insert(self._collection_name, data)
            return len(products)
            
        except Exception as e:
            raise VectorStoreError(str(e), operation="insert")
        
        finally:
            self._milvus.disconnect()
    
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
        Hybrid search for products.
        
        Combines dense vector search with BM25 sparse search.
        
        Args:
            query_embedding: Dense vector from embedding model
            query_text: Original text for BM25 sparse search
            limit: Max results to return
            category: Optional category filter
            min_price: Optional min price filter
            max_price: Optional max price filter
            
        Returns:
            List of matching products
        """
        try:
            self._milvus.connect()
            
            # Build filter expression
            filters = []
            if category:
                filters.append(f'category == "{category}"')
            if min_price is not None:
                filters.append(f"price >= {min_price}")
            if max_price is not None:
                filters.append(f"price <= {max_price}")
            
            filter_expr = " and ".join(filters) if filters else None
            
            # Output fields (all metadata)
            output_fields = [
                "product_id", "name", "category", "brand", "description",
                "price", "cost", "stock_quantity", "monthly_sales",
                "average_rating", "review_count", "supplier", "last_updated",
            ]
            
            # Hybrid search combining dense vectors with BM25 sparse vectors
            # Uses RRF (Reciprocal Rank Fusion) ranker for combining results
            try:
                results = self._milvus.hybrid_search(
                    collection_name=self._collection_name,
                    dense_vector=query_embedding,
                    query_text=query_text,  # BM25 tokenizes text for sparse search automatically
                    limit=limit,
                    output_fields=output_fields,
                    filter_expr=filter_expr,
                )
            except Exception:
                # Fallback to dense-only search if hybrid search fails
                results = self._milvus.search(
                    collection_name=self._collection_name,
                    data=[query_embedding],
                    anns_field="dense_embedding",
                    limit=limit,
                    output_fields=output_fields,
                    filter_expr=filter_expr,
                )
            
            # Convert results to Product entities
            products = []
            if results and len(results) > 0:
                for hit in results[0]:
                    product = self._hit_to_product(hit)
                    if product:
                        products.append(product)
            
            return products
            
        except Exception as e:
            raise VectorStoreError(str(e), operation="search")
        
        finally:
            self._milvus.disconnect()
    
    async def upsert(self, products: list[Product]) -> int:
        """
        Upsert products (insert or update if exists).
        
        Args:
            products: List of products with embeddings
            
        Returns:
            Number of products upserted
        """
        if not products:
            return 0
        
        try:
            self._milvus.connect()
            
            data = [
                ProductCollectionSchema.product_to_dict(p)
                for p in products
            ]
            
            result = self._milvus.upsert(self._collection_name, data)
            return len(products)
            
        except Exception as e:
            raise VectorStoreError(str(e), operation="upsert")
        
        finally:
            self._milvus.disconnect()
    
    async def delete(self, product_ids: list[str]) -> int:
        """
        Delete products by IDs.
        
        Args:
            product_ids: List of product IDs to delete
            
        Returns:
            Number of products deleted
        """
        if not product_ids:
            return 0
        
        try:
            self._milvus.connect()
            
            ids_str = ", ".join(f'"{id}"' for id in product_ids)
            filter_expr = f"product_id in [{ids_str}]"
            
            result = self._milvus.delete(self._collection_name, filter_expr)
            return len(product_ids)
            
        except Exception as e:
            raise VectorStoreError(str(e), operation="delete")
        
        finally:
            self._milvus.disconnect()
    
    async def get_all_ids(self) -> list[str]:
        """Get all product IDs in the store."""
        try:
            self._milvus.connect()
            
            results = self._milvus.query(
                collection_name=self._collection_name,
                filter_expr="product_id != ''",
                output_fields=["product_id"],
                limit=10000,
            )
            
            return [r["product_id"] for r in results]
            
        except Exception as e:
            raise VectorStoreError(str(e), operation="get_all_ids")
        
        finally:
            self._milvus.disconnect()
    
    def _hit_to_product(self, hit) -> Optional[Product]:
        """
        Convert Milvus search hit to Product entity.
        
        Args:
            hit: Milvus search hit
            
        Returns:
            Product entity or None
        """
        try:
            entity = hit.get("entity", hit)
            
            # Parse date
            last_updated = None
            if entity.get("last_updated"):
                try:
                    last_updated = datetime.strptime(
                        entity["last_updated"], "%Y-%m-%d"
                    ).date()
                except ValueError:
                    pass
            
            return Product(
                product_id=entity.get("product_id", ""),
                name=entity.get("name", ""),
                category=entity.get("category", ""),
                brand=entity.get("brand", ""),
                description=entity.get("description", ""),
                price=float(entity.get("price", 0)),
                cost=float(entity.get("cost", 0)),
                stock_quantity=int(entity.get("stock_quantity", 0)),
                monthly_sales=int(entity.get("monthly_sales", 0)),
                average_rating=float(entity.get("average_rating", 0)) if entity.get("average_rating") else None,
                review_count=int(entity.get("review_count", 0)),
                supplier=entity.get("supplier", ""),
                last_updated=last_updated,
            )
        except Exception:
            return None
