"""
Integration tests for Agent Graph flow using Table-Driven tests.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.application.agent.graph import ProductResearchAgent
from src.application.agent.state import AgentState

class TestAgentFlow:
    """Test the orchestration flow of ProductResearchAgent."""

    @pytest.fixture
    def mock_agent_deps(self):
        """Mock internal agents and LLM."""
        with patch('src.application.agent.graph.PlannerAgent') as MockPlanner, \
             patch('src.application.agent.graph.ProductQAAgent') as MockProductQA, \
             patch('src.application.agent.graph.MarketAnalysisAgent') as MockMarketAnalysis, \
             patch('src.application.agent.graph.ChatOpenAI') as MockLLM:
            
            # Setup instances
            planner_instance = MockPlanner.return_value
            product_qa_instance = MockProductQA.return_value
            market_instance = MockMarketAnalysis.return_value
            llm_instance = MockLLM.return_value
            
            # Default success responses
            planner_instance.plan = AsyncMock()
            product_qa_instance.invoke = AsyncMock(return_value={
                "answer": "Product Info", "products": [{"id": 1}], "success": True
            })
            market_instance.invoke = AsyncMock(return_value={
                "answer": "Market Info", "sources": ["google.com"], "success": True
            })
            llm_instance.ainvoke = AsyncMock(return_value=MagicMock(content="Final Answer"))
            
            yield {
                "planner": planner_instance,
                "product_qa": product_qa_instance,
                "market_analysis": market_instance,
                "llm": llm_instance
            }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("test_case", [
        {
            "name": "single_step_product_qa",
            "query": "search headphones",
            "plan": [
                {"agent": "product_qa", "action": "search headphones"}
            ],
            "expected_agents_used": ["product_qa"],
            "expected_call_counts": {"product_qa": 1, "market_analysis": 0}
        },
        {
            "name": "single_step_market_analysis",
            "query": "market price",
            "plan": [
                {"agent": "market_analysis", "action": "check price"}
            ],
            "expected_agents_used": ["market_analysis"],
            "expected_call_counts": {"product_qa": 0, "market_analysis": 1}
        },
        {
            "name": "multi_step_workflow",
            "query": "compare price",
            "plan": [
                {"agent": "product_qa", "action": "get products"},
                {"agent": "market_analysis", "action": "check competitors"}
            ],
            "expected_agents_used": ["product_qa", "market_analysis"],
            "expected_call_counts": {"product_qa": 1, "market_analysis": 1}
        },
    ])
    async def test_agent_flow_execution(self, test_case, mock_agent_deps):
        """
        Table-driven test for agent flow execution.
        Verifies that graph executes correct nodes based on the plan.
        """
        # Arrange
        mock_agent_deps["planner"].plan.return_value = {
            "plan": test_case["plan"],
            "reasoning": "Test plan"
        }
        
        agent = ProductResearchAgent()
        
        # Act
        result = await agent.query(test_case["query"])
        
        # Assert
        # Check what agents ended up in the result "agents_used" list
        # Note: The implementation in graph.py appends to agents_used in each node
        # So we check if the expected agents are present.
        
        actual_pool = set(result["agents_used"])
        expected_pool = set(test_case["expected_agents_used"])
        assert expected_pool.issubset(actual_pool), f"Expected {expected_pool} but got {actual_pool}"

        # Verify call counts
        assert mock_agent_deps["product_qa"].invoke.call_count == test_case["expected_call_counts"]["product_qa"]
        assert mock_agent_deps["market_analysis"].invoke.call_count == test_case["expected_call_counts"]["market_analysis"]
