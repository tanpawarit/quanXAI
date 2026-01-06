"""
Data Ingester - Main data pipeline for loading products into Milvus.

Pipeline Flow:
    CSV File
        ↓
    [1] Load products from CSV
        ↓
    [2] Generate embeddings (OpenAI)
        ↓
    [3] Store in Milvus (vector database)
        ↓
    Ready for search!

Two modes:
- ingest_all(): Full reload - clears and reloads everything
- incremental_update(): Smart update - only changes what's different
"""

import asyncio
from typing import Optional

from src.config import settings
from src.domain.entities import Product
from src.domain.exceptions import PipelineError
from src.domain.logger import get_logger
from src.domain.ports import IVectorStore
from src.infrastructure.milvus import MilvusClient, ProductCollectionSchema
from src.application.pipeline.loader import CSVProductLoader
from src.application.pipeline.embedder import ProductEmbedder

logger = get_logger(__name__)


class DataIngester:
    """
    Orchestrates the data ingestion pipeline.
    
    Usage:
        ingester = DataIngester()
        count = await ingester.ingest_all(drop_existing=True)
        print(f"Loaded {count} products")
    """
    
    def __init__(
        self,
        loader: CSVProductLoader = None,
        embedder: ProductEmbedder = None,
        milvus_client: MilvusClient = None,
    ):
        """Initialize with optional custom components."""
        self._loader = loader or CSVProductLoader()
        self._embedder = embedder or ProductEmbedder()
        self._milvus = milvus_client or MilvusClient()
    
    async def ingest_all(
        self,
        drop_existing: bool = False,
    ) -> int:
        """
        Full ingestion: load CSV → embed → store in Milvus.
        
        Args:
            drop_existing: Delete existing data first?
            
        Returns: Number of products loaded
        """
        collection_name = ProductCollectionSchema.COLLECTION_NAME
        
        try:
            # Step 1: Load products from CSV
            logger.debug("loading_products", file_path=self._loader.file_path)
            products = self._loader.load()
            logger.debug("products_loaded", count=len(products))
            
            if not products:
                raise PipelineError("No products loaded", stage="load")
            
            # Step 2: Generate embeddings
            products = await self._embedder.embed_products(products)
            
            # Step 3: Setup Milvus
            logger.debug("setting_up_milvus", collection=collection_name)
            self._milvus.connect()
            
            # Drop existing if requested
            if drop_existing and self._milvus.has_collection(collection_name):
                logger.debug("dropping_collection", collection=collection_name)
                self._milvus.drop_collection(collection_name)
            
            # Create collection if needed
            if not self._milvus.has_collection(collection_name):
                schema = ProductCollectionSchema.get_schema()
                
                # Create collection
                self._milvus.client.create_collection(
                    collection_name=collection_name,
                    schema=schema,
                )
                
                # Create indexes using prepare_index_params (returns proper IndexParams object)
                index_params = self._milvus.client.prepare_index_params()
                
                # Dense vector index (AUTOINDEX for compatibility with Milvus Lite)
                index_params.add_index(
                    field_name="dense_embedding",
                    index_type="AUTOINDEX",  # Milvus Lite only supports FLAT, IVF_FLAT, AUTOINDEX
                    metric_type="COSINE",
                )
                
                # Sparse vector index (BM25 for keyword search)
                index_params.add_index(
                    field_name="sparse_embedding",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="BM25",
                )
                
                self._milvus.client.create_index(
                    collection_name=collection_name,
                    index_params=index_params,
                )
                
                logger.debug("collection_created", collection=collection_name)
            
            # Step 4: Insert products
            logger.debug("inserting_products", count=len(products))
            data = [ProductCollectionSchema.product_to_dict(p) for p in products]
            result = self._milvus.insert(collection_name, data)
            
            logger.info("ingestion_complete", count=len(products), collection=collection_name)
            return len(products)
            
        except Exception as e:
            error_msg = str(e)
            if "Open local milvus failed" in error_msg or "code=1" in error_msg:
                logger.error("milvus_connection_failed", error=error_msg)

            logger.error("ingestion_failed", error=error_msg, stage="ingest_all")
            raise PipelineError(f"Ingestion failed: {e}", stage="ingest_all")
        
        finally:
            self._milvus.disconnect()
    
    async def incremental_update(
        self,
        new_csv_path: str = None,
    ) -> dict:
        """
        Smart update: compare CSV with database, update only what changed.
        
        Process:
        1. Load new CSV
        2. Get existing IDs and content_hashes from Milvus
        3. Calculate: new, updated (hash changed), unchanged, deleted
        4. Apply changes (only re-embed changed products)
        
        Returns: {'inserted': N, 'updated': N, 'deleted': N, 'unchanged': N}
        """
        collection_name = ProductCollectionSchema.COLLECTION_NAME
        
        try:
            # Load new products
            if new_csv_path:
                self._loader = CSVProductLoader(new_csv_path)
            
            new_products = self._loader.load()
            new_products_map = {p.product_id: p for p in new_products}
            new_ids = set(new_products_map.keys())
            
            logger.debug("new_products_loaded", count=len(new_products))
            
            # Get existing IDs and hashes
            self._milvus.connect()
            
            existing_data = {}  # {product_id: content_hash}
            if self._milvus.has_collection(collection_name):
                results = self._milvus.query(
                    collection_name=collection_name,
                    filter_expr="product_id != ''",
                    output_fields=["product_id", "content_hash"],
                    limit=10000,
                )
                existing_data = {r["product_id"]: r.get("content_hash", "") for r in results}
            
            existing_ids = set(existing_data.keys())
            logger.debug("existing_products_found", count=len(existing_ids))
            
            # Calculate changes
            to_insert_ids = new_ids - existing_ids  # In new, not in DB
            to_delete_ids = existing_ids - new_ids  # In DB, not in new
            common_ids = new_ids & existing_ids     # In both
            
            # Compare hashes for common products
            to_update_ids = set()
            unchanged_ids = set()
            
            for pid in common_ids:
                new_hash = new_products_map[pid].content_hash()
                old_hash = existing_data.get(pid, "")
                
                if new_hash != old_hash:
                    to_update_ids.add(pid)
                else:
                    unchanged_ids.add(pid)
            
            logger.debug(
                "changes_calculated",
                to_insert=len(to_insert_ids),
                to_update=len(to_update_ids),
                unchanged=len(unchanged_ids),
                to_delete=len(to_delete_ids),
            )
            
            # Process inserts and updates (only changed products)
            products_to_upsert = [
                new_products_map[pid] for pid in (to_insert_ids | to_update_ids)
            ]
            
            if products_to_upsert:
                products_to_upsert = await self._embedder.embed_products(products_to_upsert)
                data = [ProductCollectionSchema.product_to_dict(p) for p in products_to_upsert]
                self._milvus.upsert(collection_name, data)
            
            # Process deletes
            if to_delete_ids:
                ids_str = ", ".join(f'"{id}"' for id in to_delete_ids)
                self._milvus.delete(
                    collection_name,
                    filter_expr=f"product_id in [{ids_str}]",
                )
            
            result = {
                "inserted": len(to_insert_ids),
                "updated": len(to_update_ids),
                "unchanged": len(unchanged_ids),
                "deleted": len(to_delete_ids),
            }
            
            logger.info("incremental_update_complete", **result)
            return result
            
        except Exception as e:
            logger.error("incremental_update_failed", error=str(e))
            raise PipelineError(f"Incremental update failed: {e}", stage="update")
        
        finally:
            self._milvus.disconnect()
    
    # Synchronous wrappers
    def ingest_all_sync(self, drop_existing: bool = False) -> int:
        """Sync version of ingest_all."""
        return asyncio.run(self.ingest_all(drop_existing))
    
    def incremental_update_sync(self, new_csv_path: str = None) -> dict:
        """Sync version of incremental_update."""
        return asyncio.run(self.incremental_update(new_csv_path))
