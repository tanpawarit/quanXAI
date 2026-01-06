"""
AI Agent module - LangGraph-based Planner + Worker Agents.

Main Components:
- ProductResearchAgent: Main entry point (graph.py)
- PlannerAgent: Creates execution plans (agents/planner.py)
- ProductQAAgent: Product catalog questions (agents/product_qa.py)
- MarketAnalysisAgent: Market research (agents/market_analysis.py)

Tools (tools/):
- ProductRAGTool: Search product catalog
- WebSearchTool: Search the web
- PriceAnalysisTool: Calculate margins
- CalculatorTool: Math calculations
"""

# Main agent
from src.application.agent.graph import ProductResearchAgent

# Worker agents
from src.application.agent.agents import (
    PlannerAgent,
    ProductQAAgent,
    MarketAnalysisAgent,
)

# Tools
from src.application.agent.tools import (
    ToolResult,
    ProductRAGTool,
    WebSearchTool,
    PriceAnalysisTool,
    CalculatorTool,
)

# State
from src.application.agent.state import AgentState

__all__ = [
    # Main agent
    "ProductResearchAgent",
    "AgentState",
    # Worker agents
    "PlannerAgent",
    "ProductQAAgent",
    "MarketAnalysisAgent",
    # Tools
    "ToolResult",
    "ProductRAGTool",
    "WebSearchTool",
    "PriceAnalysisTool",
    "CalculatorTool",
]

