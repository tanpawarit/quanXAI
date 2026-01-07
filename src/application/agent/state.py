"""
Agent State - Shared state that flows through the LangGraph.

This TypedDict defines all data that passes between nodes in the agent graph.
"""

from typing import TypedDict, Annotated
from operator import add


class PlanStep(TypedDict):
    """A single step in the execution plan."""
    step: int
    action: str
    agent: str  # "product_qa" or "market_analysis"


class StepResult(TypedDict):
    """Result from executing a single step."""
    step: int
    agent: str
    answer: str
    products: list[dict]
    sources: list[dict]
    confidence: float
    success: bool


class AgentState(TypedDict):
    """
    Shared state for the Planner + Worker Agents architecture.
    
    Flow:
    1. Planner creates `plan` and `reasoning`
    2. Workers execute steps, appending to `step_results`
    3. Synthesizer creates final `answer`
    """
    # Input
    query: str
    chat_history: list[dict]  # List of {"role": "user"|"assistant", "content": str}
    
    # Planner output
    plan: list[PlanStep]
    reasoning: str
    
    # Execution tracking
    current_step: int
    step_results: Annotated[list[StepResult], add]  # Accumulates results
    
    # Collected data from tools
    products: Annotated[list[dict], add]
    sources: Annotated[list[dict], add]
    
    # Final output
    answer: str
    confidence: float
    agents_used: Annotated[list[str], add]
