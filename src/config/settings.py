"""Application settings using Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Usage:
        from src.config import settings
        print(settings.openai_api_key)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # OpenAI
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    
    # Milvus Lite (file-based, no Docker needed)
    milvus_db_path: str = "./data/milvus_lite.db"
    milvus_collection_name: str = "products"
    milvus_index_type: str = "HNSW"  # IVF_FLAT or HNSW
    
    # Tavily Web Search
    tavily_api_key: str = ""
    
    # Database (SQLite - no Docker needed)
    database_url: str = "sqlite:///./data/app.db"
    
    # Data
    products_csv_path: str = "products_catalog.csv"
    
    # Embedding
    embedding_dimension: str = "1536"  # text-embedding-3-small
    embedding_batch_size: str = "100"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: str = "8000"
    
    # Environment: "development" or "production"
    environment: str = "development"
    
    # Helper properties to convert string to int
    
    @property
    def embedding_dimension_int(self) -> int:
        return int(self.embedding_dimension)
    
    @property
    def embedding_batch_size_int(self) -> int:
        return int(self.embedding_batch_size)
    
    @property
    def api_port_int(self) -> int:
        return int(self.api_port)
    
    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Singleton instance
settings = get_settings()
