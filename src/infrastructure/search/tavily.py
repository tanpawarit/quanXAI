"""Tavily web search client using LangChain integration."""

from typing import Optional

from langchain_tavily import TavilySearch

from src.config import settings
from src.domain.exceptions import SearchError
from src.domain.ports import ISearchClient


class TavilySearchClient(ISearchClient):
    """
    Tavily web search implementation using LangChain Tavily integration.
    
    Provides web search for market trends and competitor information.
    """
    
    def __init__(self, api_key: str = None, max_results: int = 5):
        """
        Initialize Tavily client.
        
        Args:
            api_key: Tavily API key (default from settings)
            max_results: Maximum results to return
        """
        self._api_key = api_key or settings.tavily_api_key
        self._max_results = max_results
        self._tool: Optional[TavilySearch] = None
    
    @property
    def tool(self) -> TavilySearch:
        """Get or create Tavily search tool."""
        if self._tool is None:
            self._tool = TavilySearch(
                max_results=self._max_results,
                topic="general",
                include_answer=True,
                search_depth="basic",
            )
        return self._tool
    
    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of search results with title, url, content
            
        Raises:
            SearchError: If search fails
        """
        try:
            # Use ainvoke for async operation
            response = await self.tool.ainvoke({"query": query})
            
            # Parse response (it's a JSON string or dict)
            import json
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response
            
            results = []
            
            # Add AI-generated answer if available
            if data.get("answer"):
                results.append({
                    "type": "answer",
                    "content": data["answer"],
                })
            
            # Add search results
            for item in data.get("results", [])[:max_results]:
                results.append({
                    "type": "result",
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0),
                })
            
            return results
            
        except Exception as e:
            raise SearchError(f"Web search failed: {e}")
    
    async def search_with_context(
        self,
        query: str,
        max_results: int = 5,
    ) -> dict:
        """
        Search and return structured context.
        
        Args:
            query: Search query
            max_results: Max results
            
        Returns:
            Dict with 'answer', 'sources', 'raw_results'
        """
        results = await self.search(query, max_results)
        
        answer = ""
        sources = []
        
        for r in results:
            if r.get("type") == "answer":
                answer = r["content"]
            elif r.get("type") == "result":
                sources.append({
                    "title": r["title"],
                    "url": r["url"],
                    "snippet": r["content"][:200] + "..." if len(r.get("content", "")) > 200 else r.get("content", ""),
                })
        
        return {
            "answer": answer,
            "sources": sources,
            "raw_results": results,
        }
