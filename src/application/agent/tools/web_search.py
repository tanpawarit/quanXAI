"""Web Search Tool - Search the web using Tavily API."""

from src.infrastructure.search import TavilySearchClient
from src.application.agent.tools.base import ToolResult


class WebSearchTool:
    """Search the web for market information using Tavily."""
    
    def __init__(self, search_client: TavilySearchClient = None):
        self._client = search_client or TavilySearchClient()
    
    async def execute(self, query: str, max_results: int = 5, **kwargs) -> ToolResult:
        """Search the web."""
        try:
            context = await self._client.search_with_context(query=query, max_results=max_results)
            
            sources = [{"title": s["title"], "url": s["url"], "snippet": s["snippet"]} for s in context.get("sources", [])]
            answer = context.get("answer", "") or (f"Found {len(sources)} relevant sources." if sources else "No results found.")
            
            return ToolResult(answer=answer, sources=sources, confidence=0.8 if sources else 0.2)
        except Exception as e:
            return ToolResult(answer="", error=f"Web search failed: {str(e)}", confidence=0.0)
