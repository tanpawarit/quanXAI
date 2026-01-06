"""
Base Tool - Common types for AI tools.

ToolResult: Standard result format returned by all tools.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolResult:
    """
    Standard result format returned by all tools.
    
    Fields:
    - answer: Human-readable response
    - products: List of products (for catalog tools)
    - sources: Web sources (for search tools)
    - confidence: 0-1 score (1 = deterministic/certain)
    - metadata: Extra info for debugging
    - error: Error message if failed
    """
    answer: str
    products: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None
