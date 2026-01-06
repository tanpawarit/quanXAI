"""
MarketAnalysis Agent - ReAct agent for market research.

Uses LangChain 1.x's create_agent with tools:
- web_search: Search the web
- price_analysis: Calculate margins
"""

from langchain.agents import create_agent
from langchain_core.tools import tool

from src.config.settings import settings
from src.application.agent.prompts.market_analysis import MARKET_ANALYSIS_SYSTEM_PROMPT
from src.application.agent.tools import WebSearchTool, PriceAnalysisTool


# Initialize tool instances
_web_search = WebSearchTool()
_price_analysis = PriceAnalysisTool()


@tool("web_search", description="Search web for market info, competitor prices, reviews. Use for external data.")
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for market information."""
    result = await _web_search.execute(query=query, max_results=max_results)
    if result.error:
        return f"Error: {result.error}"
    output = result.answer + "\n\nSources:\n"
    for s in result.sources[:3]:
        output += f"- {s['title']}: {s['url']}\n"
    
    # Append confidence
    output += f"\nConfidence Score: {result.confidence}"
    return output


@tool("price_analysis", description="Analyze product margins. Use for pricing comparisons and metrics.")
async def price_analysis(category: str = None, margin_threshold: float = None, limit: int = 10) -> str:
    """Analyze product pricing and margins."""
    result = await _price_analysis.execute(category=category, margin_threshold=margin_threshold, limit=limit)
    if result.error:
        return f"Error: {result.error}"
    return result.answer


class MarketAnalysisAgent:
    """
    ReAct agent for market research and competitor analysis.
    
    Example:
        agent = MarketAnalysisAgent()
        result = await agent.invoke("What are competitor prices for headphones?")
    """
    
    def __init__(self):
        """Initialize the ReAct agent with tools."""
        self._agent = create_agent(
            model=settings.openai_chat_model,
            tools=[web_search, price_analysis],
            system_prompt=MARKET_ANALYSIS_SYSTEM_PROMPT,
        )
    
    async def invoke(self, query: str, context: str = "") -> dict:
        """
        Process a query and return the result.
        
        Args:
            query: User's question
            context: Optional context from previous steps
            
        Returns:
            Dict with 'answer', 'sources', 'success', 'confidence'
        """
        full_query = query
        if context:
            full_query = f"Context: {context}\n\nQuestion: {query}"
        
        result = await self._agent.ainvoke({
            "messages": [{"role": "user", "content": full_query}]
        })
        
        # Extract final message and sources from tool calls
        messages = result.get("messages", [])
        sources = []
        confidences = []
        answer = "No response generated."
        
        for msg in messages:
            # Extract sources and confidence from tool messages
            if hasattr(msg, 'type') and msg.type == 'tool':
                content = msg.content if hasattr(msg, 'content') else str(msg)
                
                # Extract confidence if present
                if "Confidence Score: " in content:
                    try:
                        # Find the line with confidence
                        for line in content.split("\n"):
                            if line.strip().startswith("Confidence Score: "):
                                conf_val = float(line.strip().replace("Confidence Score: ", ""))
                                confidences.append(conf_val)
                    except ValueError:
                        pass
                
                # Parse sources from web_search output format
                if "Sources:" in content:
                    lines = content.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line.startswith("- ") and ": http" in line:
                            # Format: "- Title: URL"
                            parts = line[2:].split(": ", 1)
                            if len(parts) == 2:
                                sources.append({
                                    "title": parts[0],
                                    "url": parts[1]
                                })
        
        # Get the final AI response
        if messages:
            last_message = messages[-1]
            answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Calculate aggregated confidence
        if confidences:
            confidence = sum(confidences) / len(confidences)
        else:
            confidence = 0.5
            
        return {
            "answer": answer,
            "sources": sources,
            "success": True,
            "confidence": confidence,
        }
