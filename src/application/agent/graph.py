"""
Agent Graph - LangGraph StateGraph connecting Planner → Worker Agents → Synthesize.

This is the main entry point for the agent system.

Flow:
1. Planner analyzes query and creates execution plan
2. Worker agents execute steps sequentially
3. Synthesize node creates final answer

Example:
    from src.application.agent.graph import ProductResearchAgent
    
    agent = ProductResearchAgent()
    result = await agent.query("What headphones do we have?")
"""

from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.config.settings import settings
from src.application.agent.state import AgentState
from src.application.agent.agents.planner import PlannerAgent
from src.application.agent.agents.product_qa import ProductQAAgent
from src.application.agent.agents.market_analysis import MarketAnalysisAgent


class ProductResearchAgent:
    """
    Main AI Agent using LangGraph with Planner + Worker architecture.
    
    Architecture:
        Query → Planner → [ProductQA, MarketAnalysis] → Synthesize → Answer
    
    Usage:
        agent = ProductResearchAgent()
        result = await agent.query("What headphones do we have?")
        print(result["answer"])
    """
    
    def __init__(self):
        """Initialize all agent components and build the graph."""
        self._llm = ChatOpenAI(
            model=settings.openai_chat_model,
            temperature=0.7,
            api_key=settings.openai_api_key,
        )
        
        # Initialize agents
        self._planner = PlannerAgent()
        self._product_qa = ProductQAAgent()
        self._market_analysis = MarketAnalysisAgent()
        
        # Build the graph
        self._graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph."""
        
        # Create graph with AgentState
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("planner", self._planner_node)
        graph.add_node("product_qa", self._product_qa_node)
        graph.add_node("market_analysis", self._market_analysis_node)
        graph.add_node("synthesize", self._synthesize_node)
        
        # Set entry point
        graph.set_entry_point("planner")
        
        # Add conditional edges from planner
        graph.add_conditional_edges(
            "planner",
            self._route_to_agent,
            {
                "product_qa": "product_qa",
                "market_analysis": "market_analysis",
                "synthesize": "synthesize",
            }
        )
        
        # Add edges from worker agents
        graph.add_conditional_edges(
            "product_qa",
            self._check_next_step,
            {
                "product_qa": "product_qa",
                "market_analysis": "market_analysis",
                "synthesize": "synthesize",
            }
        )
        
        graph.add_conditional_edges(
            "market_analysis",
            self._check_next_step,
            {
                "product_qa": "product_qa",
                "market_analysis": "market_analysis",
                "synthesize": "synthesize",
            }
        )
        
        # End after synthesize
        graph.add_edge("synthesize", END)
        
        return graph.compile()
    
    async def _planner_node(self, state: AgentState) -> dict:
        """Planner node: create execution plan."""
        result = await self._planner.plan(state["query"], state.get("chat_history", []))
        
        return {
            "reasoning": result["reasoning"],
            "plan": result["plan"],
            "current_step": 0,
            "step_results": [],
            "products": [],
            "sources": [],
            "agents_used": [],
        }
    
    async def _product_qa_node(self, state: AgentState) -> dict:
        """ProductQA worker node."""
        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])
        
        if current_step >= len(plan):
            return {}
        
        step = plan[current_step]
        action = step.get("action", state["query"])
        
        # Get context from previous results
        context = self._build_context(state.get("step_results", []))
        
        result = await self._product_qa.invoke(action, context)
        
        return {
            "step_results": [{
                "step": current_step + 1,
                "agent": "product_qa",
                "answer": result["answer"],
                "products": result.get("products", []),
                "sources": [],
                "success": result.get("success", True),
                "confidence": result.get("confidence", 0.5),
            }],
            "products": result.get("products", []),
            "current_step": current_step + 1,
            "agents_used": ["product_qa"],
        }
    
    async def _market_analysis_node(self, state: AgentState) -> dict:
        """MarketAnalysis worker node."""
        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])
        
        if current_step >= len(plan):
            return {}
        
        step = plan[current_step]
        action = step.get("action", state["query"])
        
        # Get context from previous results
        context = self._build_context(state.get("step_results", []))
        
        result = await self._market_analysis.invoke(action, context)
        
        return {
            "step_results": [{
                "step": current_step + 1,
                "agent": "market_analysis",
                "answer": result["answer"],
                "products": [],
                "sources": result.get("sources", []),
                "success": result.get("success", True),
                "confidence": result.get("confidence", 0.5),
            }],
            "sources": result.get("sources", []),
            "current_step": current_step + 1,
            "agents_used": ["market_analysis"],
        }
    
    async def _synthesize_node(self, state: AgentState) -> dict:
        """Synthesize final answer from all step results."""
        step_results = state.get("step_results", [])
        
        # Build synthesis prompt
        results_text = ""
        for sr in step_results:
            results_text += f"\n[Step {sr['step']} - {sr['agent']}]\n"
            results_text += sr["answer"] + "\n"
        
        prompt = f"""Based on the research results below, provide a comprehensive answer to the user's query.

            User Query: {state["query"]}

            Research Results:
            {results_text}

            Provide a clear, concise answer that:
            1. Directly addresses the user's question
            2. Synthesizes findings from all research steps
            3. Provides actionable recommendations if applicable

            Do not mention internal tools or agents. Write naturally."""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._llm.ainvoke(messages)
        
        # Calculate confidence from step results
        if step_results:
            confidences = [sr.get("confidence", 0.5) for sr in step_results]
            confidence = sum(confidences) / len(confidences)
        else:
            confidence = 0.5
        
        return {
            "answer": response.content,
            "confidence": confidence,
        }
    
    def _route_to_agent(self, state: AgentState) -> Literal["product_qa", "market_analysis", "synthesize"]:
        """Route from planner to first agent in plan."""
        plan = state.get("plan", [])
        
        if not plan:
            return "synthesize"
        
        first_agent = plan[0].get("agent", "product_qa")
        return first_agent
    
    def _check_next_step(self, state: AgentState) -> Literal["product_qa", "market_analysis", "synthesize"]:
        """Check if there are more steps or go to synthesize."""
        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])
        
        if current_step >= len(plan):
            return "synthesize"
        
        next_agent = plan[current_step].get("agent", "product_qa")
        return next_agent
    
    def _build_context(self, step_results: list) -> str:
        """Build context string from previous step results."""
        if not step_results:
            return ""
        
        context_parts = []
        for sr in step_results:
            context_parts.append(f"Previous analysis ({sr['agent']}): {sr['answer']}")
        
        return "\n".join(context_parts)
    
    async def query(self, query: str, chat_history: list[dict] = None) -> dict:
        """
        Process a user query and return the result.
        
        Args:
            query: User's question
            chat_history: Previous conversation history
            
        Returns:
            Dict with: answer, reasoning, agents_used, products, sources, confidence
        """
        initial_state: AgentState = {
            "query": query,
            "chat_history": chat_history or [],
            "plan": [],
            "reasoning": "",
            "current_step": 0,
            "step_results": [],
            "products": [],
            "sources": [],
            "answer": "",
            "confidence": 0.0,
            "agents_used": [],
        }
        
        result = await self._graph.ainvoke(initial_state)
        
        return {
            "answer": result.get("answer", ""),
            "reasoning": result.get("reasoning", ""),
            "agents_used": result.get("agents_used", []),
            "products": result.get("products", []),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
        }
    
    async def chat(self, query: str) -> str:
        """Simple interface - returns just the answer string."""
        result = await self.query(query)
        return result["answer"]
