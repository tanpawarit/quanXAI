"""
Planner Agent - Analyzes queries and creates execution plans.

This agent doesn't use tools - it uses the LLM to create a structured
plan that determines which worker agents to call and in what order.
"""

import json
from typing import Any

from langchain_openai import ChatOpenAI

from src.application.agent.prompts.planner import PLANNER_SYSTEM_PROMPT
from src.application.agent.state import AgentState, PlanStep
from src.config.settings import settings


class PlannerAgent:
    """
    Creates execution plans for user queries.
    
    Example:
        planner = PlannerAgent()
        result = await planner.plan("Compare our prices with competitors")
        # result = {"reasoning": "...", "plan": [...]}
    """
    
    def __init__(self):
        """Initialize with ChatOpenAI LLM."""
        self._llm = ChatOpenAI(
            model=settings.openai_chat_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
    
    async def plan(self, query: str, chat_history: list[dict] = None) -> dict[str, Any]:
        """
        Analyze query and create execution plan.
        
        Args:
            query: User's question
            chat_history: Previous conversation history
            
        Returns:
            Dict with 'reasoning' and 'plan' (list of PlanStep)
        """
        messages = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}]
        
        # Add history if available
        if chat_history:
            messages.extend(chat_history)
            
        messages.append({"role": "user", "content": query})
        
        response = await self._llm.ainvoke(messages)
        content = response.content
        
        # Parse JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            return {
                "reasoning": result.get("reasoning", ""),
                "plan": result.get("plan", []),
            }
        except json.JSONDecodeError:
            # Fallback: single step using product_qa
            return {
                "reasoning": "Could not parse plan, defaulting to product search.",
                "plan": [{"step": 1, "action": query, "agent": "product_qa"}],
            }
    
    async def __call__(self, state: AgentState) -> dict:
        """
        LangGraph node function.
        
        Updates state with plan and reasoning.
        """
        result = await self.plan(state["query"], state.get("chat_history", []))
        
        return {
            "reasoning": result["reasoning"],
            "plan": result["plan"],
            "current_step": 0,
        }
