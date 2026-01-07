"""
Unit tests for agent tools using Table-Driven tests.
"""
import pytest
from unittest.mock import AsyncMock

from src.application.agent.tools.product_rag import ProductRAGTool

class TestProductRAGTool:
    """Test suite for ProductRAGTool."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("test_case", [
        {
            "name": "basic_search",
            "query": "headphones",
            "kwargs": {},
            "repo_response": "sample_product", # utilizing fixture logic
            "expected_count": 1,
            "expected_confidence": 0.9,
        },
        {
            "name": "search_with_category",
            "query": "headphones",
            "kwargs": {"category": "Electronics"},
            "repo_response": "sample_product",
            "expected_count": 1,
            "expected_confidence": 0.9,
        },
        {
            "name": "empty_result",
            "query": "nonexistent",
            "kwargs": {},
            "repo_response": [], # Empty list
            "expected_count": 0,
            "expected_confidence": 0.3, # Low confidence when no products found
        },
    ])
    async def test_execute(self, test_case, mock_repository, mock_embedder, sample_product):
        """
        Table-driven test for execute method.
        """
        # Arrange
        if test_case["repo_response"] == "sample_product":
            search_result = [sample_product]
        else:
            search_result = test_case["repo_response"]
            
        mock_repository.search = AsyncMock(return_value=search_result)
        
        tool = ProductRAGTool(repository=mock_repository, embedder=mock_embedder)

        # Act
        result = await tool.execute(test_case["query"], **test_case["kwargs"])

        # Assert
        assert len(result.products) == test_case["expected_count"]
        assert result.confidence == test_case["expected_confidence"]
        
        # Verify repository was called with correct args
        mock_repository.search.assert_called_once()
        call_kwargs = mock_repository.search.call_args.kwargs
        for k, v in test_case["kwargs"].items():
            assert call_kwargs[k] == v

    @pytest.mark.asyncio
    async def test_execute_error(self, mock_repository, mock_embedder):
        """Test error handling (separate case as it acts differently)."""
        # Arrange
        mock_repository.search.side_effect = Exception("DB Connection Failed")
        tool = ProductRAGTool(repository=mock_repository, embedder=mock_embedder)

        # Act
        result = await tool.execute("anything")

        # Assert
        assert result.error is not None
        assert "Search failed" in result.error
        assert result.confidence == 0.0
