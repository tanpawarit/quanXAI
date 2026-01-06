"""OpenAI LLM provider for chat and function calling."""

from typing import Optional

from openai import AsyncOpenAI

from src.config import settings
from src.domain.exceptions import LLMError
from src.domain.ports import ILLMProvider


class OpenAILLM(ILLMProvider):
    """
    OpenAI LLM provider implementation.
    
    Uses GPT-4o-mini for chat completions with function calling support.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
    ):
        """
        Initialize OpenAI LLM.
        
        Args:
            api_key: OpenAI API key (default from settings)
            model: Model name (default gpt-4o-mini)
        """
        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.openai_chat_model
        self._client: Optional[AsyncOpenAI] = None
    
    @property
    def client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client
    
    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
    ) -> dict:
        """
        Send chat completion request.
        
        Args:
            messages: Chat messages [{"role": "user", "content": "..."}]
            tools: Optional tool definitions for function calling
            temperature: Sampling temperature
            
        Returns:
            Response dict with 'content' and optional 'tool_calls'
            
        Raises:
            LLMError: If request fails
        """
        try:
            kwargs = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = await self.client.chat.completions.create(**kwargs)
            
            message = response.choices[0].message
            
            result = {
                "content": message.content or "",
                "role": message.role,
            }
            
            # Handle tool calls
            if message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in message.tool_calls
                ]
            
            return result
            
        except Exception as e:
            raise LLMError(f"Chat completion failed: {e}")
    
    async def chat_with_tools(
        self,
        query: str,
        tools: list[dict],
        system_prompt: str = None,
    ) -> dict:
        """
        Convenience method for single query with tools.
        
        Args:
            query: User query
            tools: Tool definitions
            system_prompt: Optional system prompt
            
        Returns:
            Response with content and tool_calls
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": query})
        
        return await self.chat(messages, tools=tools)
    
    async def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
