"""Milvus collection schema for products with hybrid search support."""

from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType

from src.config import settings


class ProductCollectionSchema:
    """
    Schema definition for products collection.
    
    Supports hybrid search with:
    - Dense vectors: OpenAI text-embedding-3-small (1536 dim)
    - Sparse vectors: BM25 for keyword matching
    """
    
    COLLECTION_NAME = settings.milvus_collection_name
    DENSE_DIM = settings.embedding_dimension_int  # 1536 for text-embedding-3-small
    
    @classmethod
    def get_schema(cls) -> CollectionSchema:
        """
        Create collection schema with hybrid search fields.
        
        Returns:
            CollectionSchema for products
        """
        fields = [
            # Primary key
            FieldSchema(
                name="product_id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=50,
            ),
            # Searchable text (for BM25 sparse)
            FieldSchema(
                name="search_text",
                dtype=DataType.VARCHAR,
                max_length=4096,
                enable_analyzer=True,  # Enable tokenization for BM25
                analyzer_params={"type": "standard"},
            ),
            # Dense embedding vector
            FieldSchema(
                name="dense_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=cls.DENSE_DIM,
            ),
            # Sparse embedding (BM25) - auto-generated from search_text
            FieldSchema(
                name="sparse_embedding",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
            ),
            # Metadata fields
            FieldSchema(
                name="name",
                dtype=DataType.VARCHAR,
                max_length=256,
            ),
            FieldSchema(
                name="category",
                dtype=DataType.VARCHAR,
                max_length=100,
            ),
            FieldSchema(
                name="brand",
                dtype=DataType.VARCHAR,
                max_length=100,
            ),
            FieldSchema(
                name="description",
                dtype=DataType.VARCHAR,
                max_length=4096,
            ),
            FieldSchema(
                name="price",
                dtype=DataType.FLOAT,
            ),
            FieldSchema(
                name="cost",
                dtype=DataType.FLOAT,
            ),
            FieldSchema(
                name="stock_quantity",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="monthly_sales",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="average_rating",
                dtype=DataType.FLOAT,
            ),
            FieldSchema(
                name="review_count",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="supplier",
                dtype=DataType.VARCHAR,
                max_length=100,
            ),
            FieldSchema(
                name="last_updated",
                dtype=DataType.VARCHAR,
                max_length=20,
            ),
            # Hash of search_text for change detection
            FieldSchema(
                name="content_hash",
                dtype=DataType.VARCHAR,
                max_length=32,  # MD5 hex length
            ),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Product catalog with hybrid search support",
            enable_dynamic_field=False,
        )
        
        # Add BM25 function to auto-generate sparse embeddings
        bm25_function = Function(
            name="bm25_function",
            input_field_names=["search_text"],
            output_field_names=["sparse_embedding"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)
        
        return schema
    
    @classmethod
    def get_index_params(cls) -> list[dict]:
        """
        Get index parameters for both dense and sparse fields.
        
        Returns:
            List of index param dicts
        """
        index_params = [
            # Dense vector index (HNSW for better recall)
            {
                "field_name": "dense_embedding",
                "index_type": settings.milvus_index_type,
                "metric_type": "COSINE",
                "params": {
                    "M": 16,
                    "efConstruction": 256,
                } if settings.milvus_index_type == "HNSW" else {
                    "nlist": 128,
                },
            },
            # Sparse vector index (inverted index for BM25)
            {
                "field_name": "sparse_embedding",
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
            },
        ]
        return index_params
    
    @classmethod
    def product_to_dict(cls, product) -> dict:
        """
        Convert Product entity to Milvus insert dict.
        
        Args:
            product: Product entity with embedding
            
        Returns:
            Dict ready for Milvus insert
        """
        return {
            "product_id": product.product_id,
            "search_text": product.to_search_text(),
            "dense_embedding": product.embedding,
            "name": product.name,
            "category": product.category,
            "brand": product.brand,
            "description": product.description,
            "price": product.price,
            "cost": product.cost,
            "stock_quantity": product.stock_quantity,
            "monthly_sales": product.monthly_sales,
            "average_rating": product.average_rating or 0.0,
            "review_count": product.review_count,
            "supplier": product.supplier,
            "last_updated": str(product.last_updated) if product.last_updated else "",
            "content_hash": product.content_hash(),
        }
