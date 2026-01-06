"""
Domain Exceptions - Custom error types for the application.

All custom exceptions inherit from DomainException.
Use these instead of generic exceptions for better error handling.

Exception hierarchy:
- DomainException (base)
  ├── ProductNotFoundError
  ├── InvalidProductDataError
  ├── EmbeddingError
  ├── VectorStoreError
  ├── SearchError
  ├── PipelineError
  ├── LLMError
  └── ToolError
"""


class DomainException(Exception):
    """Base exception for all domain errors."""
    pass


class ProductNotFoundError(DomainException):
    """Product was not found in the catalog."""
    
    def __init__(self, product_id: str):
        self.product_id = product_id
        super().__init__(f"Product not found: {product_id}")


class InvalidProductDataError(DomainException):
    """Product data failed validation."""
    
    def __init__(self, message: str, field: str = None):
        self.field = field
        super().__init__(f"Invalid product data: {message}")


class EmbeddingError(DomainException):
    """Failed to generate text embeddings."""
    
    def __init__(self, message: str):
        super().__init__(f"Embedding error: {message}")


class VectorStoreError(DomainException):
    """Vector database operation failed."""
    
    def __init__(self, message: str, operation: str = None):
        self.operation = operation
        super().__init__(f"Vector store error ({operation}): {message}")


class SearchError(DomainException):
    """Search operation failed."""
    
    def __init__(self, message: str):
        super().__init__(f"Search error: {message}")


class PipelineError(DomainException):
    """Data pipeline operation failed."""
    
    def __init__(self, message: str, stage: str = None):
        self.stage = stage
        super().__init__(f"Pipeline error ({stage}): {message}")


class LLMError(DomainException):
    """LLM API call failed."""
    
    def __init__(self, message: str):
        super().__init__(f"LLM error: {message}")


class ToolError(DomainException):
    """Agent tool execution failed."""
    
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' error: {message}")
