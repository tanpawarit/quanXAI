"""Milvus client wrapper for connection management."""

from pymilvus import MilvusClient as PyMilvusClient, connections

from src.config import settings


class MilvusClient:
    """
    Milvus client wrapper.
    
    Provides connection management and collection operations.
    Uses pymilvus MilvusClient for simplified API.
    Supports Milvus Lite (file-based, no Docker needed).
    """
    
    # Shared connection registry to avoid file lock errors
    _connections: dict[str, PyMilvusClient] = {}

    def __init__(
        self,
        db_path: str = None,
    ):
        """
        Initialize Milvus client.
        
        Args:
            db_path: Path to Milvus Lite database file (default from settings)
        """
        self._db_path = db_path or settings.milvus_db_path
        self._client: PyMilvusClient | None = None
    
    @property
    def client(self) -> PyMilvusClient:
        """Get or create Milvus client connection."""
        if self._client is None:
            # Check if we already have a connection for this path
            if self._db_path in self._connections:
                self._client = self._connections[self._db_path]
            else:
                # Milvus Lite uses file path as URI
                self._client = PyMilvusClient(uri=self._db_path)
                self._connections[self._db_path] = self._client
                
        return self._client
    
    def connect(self) -> "MilvusClient":
        """
        Establish connection to Milvus.
        
        Returns:
            Self for chaining
        """
        _ = self.client  # Triggers connection
        return self
    
    def disconnect(self) -> None:
        """
        Close connection to Milvus.
        
        Note: For Milvus Lite with shared connections, we generally don't want
        to close the underlying connection as other components might be using it.
        We just clear our local reference.
        """
        self._client = None
    
    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists."""
        return self.client.has_collection(collection_name)
    
    def create_collection(
        self,
        collection_name: str,
        schema: dict,
        index_params: dict = None,
    ) -> None:
        """
        Create collection with schema.
        
        Args:
            collection_name: Name of collection
            schema: Collection schema dict
            index_params: Optional index parameters
        """
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
    
    def drop_collection(self, collection_name: str) -> None:
        """Drop collection if exists."""
        if self.has_collection(collection_name):
            self.client.drop_collection(collection_name)
    
    def insert(self, collection_name: str, data: list[dict]) -> dict:
        """
        Insert data into collection.
        
        Args:
            collection_name: Target collection
            data: List of dicts with field values
            
        Returns:
            Insert result
        """
        return self.client.insert(
            collection_name=collection_name,
            data=data,
        )
    
    def upsert(self, collection_name: str, data: list[dict]) -> dict:
        """
        Upsert data into collection.
        
        Args:
            collection_name: Target collection
            data: List of dicts with field values
            
        Returns:
            Upsert result
        """
        return self.client.upsert(
            collection_name=collection_name,
            data=data,
        )
    
    def delete(self, collection_name: str, filter_expr: str) -> dict:
        """
        Delete data from collection.
        
        Args:
            collection_name: Target collection
            filter_expr: Filter expression (e.g., 'product_id in ["P001", "P002"]')
            
        Returns:
            Delete result
        """
        return self.client.delete(
            collection_name=collection_name,
            filter=filter_expr,
        )
    
    def search(
        self,
        collection_name: str,
        data: list[list[float]],
        anns_field: str,
        limit: int = 10,
        output_fields: list[str] = None,
        filter_expr: str = None,
    ) -> list:
        """
        Vector similarity search.
        
        Args:
            collection_name: Target collection
            data: Query vectors
            anns_field: Vector field to search
            limit: Max results per query
            output_fields: Fields to return
            filter_expr: Optional filter expression
            
        Returns:
            Search results
        """
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        return self.client.search(
            collection_name=collection_name,
            data=data,
            anns_field=anns_field,
            limit=limit,
            output_fields=output_fields,
            filter=filter_expr,
            search_params=search_params,
        )
    
    def hybrid_search(
        self,
        collection_name: str,
        dense_vector: list[float],
        query_text: str = None,
        sparse_vector: dict = None,
        limit: int = 10,
        output_fields: list[str] = None,
        filter_expr: str = None,
    ) -> list:
        """
        Hybrid search combining dense and sparse vectors.
        
        Args:
            collection_name: Target collection
            dense_vector: Dense embedding vector
            query_text: Query text for BM25 sparse search (preferred)
            sparse_vector: Pre-computed sparse vector (alternative to query_text)
            limit: Max results
            output_fields: Fields to return
            filter_expr: Optional filter
            
        Returns:
            Combined search results
        """
        from pymilvus import AnnSearchRequest, RRFRanker
        
        # Use a larger candidate pool for RRF ranking
        candidate_limit = limit * 3
        
        # Dense search request
        dense_search = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=candidate_limit,
        )
        
        # Sparse search request - use query_text for BM25 or pre-computed sparse_vector
        sparse_data = query_text if query_text else sparse_vector
        sparse_search = AnnSearchRequest(
            data=[sparse_data],
            anns_field="sparse_embedding",
            param={"metric_type": "BM25"},  # Use BM25 metric for text-based sparse search
            limit=candidate_limit,
        )
        
        # Hybrid search with RRF ranker
        results = self.client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_search, sparse_search],
            ranker=RRFRanker(),
            limit=limit,
            output_fields=output_fields,
        )
        
        return results
    
    def query(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: list[str] = None,
        limit: int = 100,
    ) -> list:
        """
        Query collection with filter.
        
        Args:
            collection_name: Target collection
            filter_expr: Filter expression
            output_fields: Fields to return
            limit: Max results
            
        Returns:
            Query results
        """
        return self.client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=output_fields,
            limit=limit,
        )
    
    def __enter__(self) -> "MilvusClient":
        """Context manager enter."""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()
