"""
ProductQA Agent - ReAct agent for product catalog questions.

Uses LangChain 1.x's create_agent with tools:
- product_catalog: Search products
- calculator: Do math
- price_analysis: Calculate margins
"""

from langchain.agents import create_agent
from langchain_core.tools import tool

from src.config.settings import settings
from src.application.agent.prompts.product_qa import PRODUCT_QA_SYSTEM_PROMPT
from src.application.agent.tools import ProductRAGTool, CalculatorTool, PriceAnalysisTool


# Initialize tool instances
_product_rag = ProductRAGTool()
_calculator = CalculatorTool()
_price_analysis = PriceAnalysisTool()


@tool("product_catalog", description="Search products by query, category, price range. Use for product/stock/price questions.")
async def product_rag(query: str, category: str = None, min_price: float = None, max_price: float = None, limit: int = 10) -> str:
    """Search the product catalog."""
    result = await _product_rag.execute(query=query, category=category, min_price=min_price, max_price=max_price, limit=limit)
    if result.error:
        return f"Error: {result.error}"
    output = result.answer + "\n\n"
    for p in result.products[:5]:
        output += f"- [{p['product_id']}] {p['name']} ({p['brand']}) - {p['category']}: ${p['price']:.2f} | Stock: {p['stock_quantity']} | Rating: {p['average_rating']}\n"
    
    # Append confidence for the agent to parse
    output += f"\nConfidence Score: {result.confidence}"
    return output


@tool("calculator", description="Evaluate math expressions. Use for margins, percentages, averages.")
async def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    result = await _calculator.execute(expression=expression)
    if result.error:
        return f"Error: {result.error}"
    return result.answer


@tool("price_analysis", description="Analyze product margins. Use for low-margin products, pricing metrics.")
async def price_analysis(category: str = None, margin_threshold: float = None, limit: int = 10) -> str:
    """Analyze product pricing and margins."""
    result = await _price_analysis.execute(category=category, margin_threshold=margin_threshold, limit=limit)
    if result.error:
        return f"Error: {result.error}"
    return result.answer


class ProductQAAgent:
    """
    ReAct agent for product catalog questions.
    
    Example:
        agent = ProductQAAgent()
        result = await agent.invoke("What headphones do we have?")
    """
    
    def __init__(self):
        """Initialize the ReAct agent with tools."""
        self._agent = create_agent(
            model=settings.openai_chat_model,
            tools=[product_rag, calculator, price_analysis],
            system_prompt=PRODUCT_QA_SYSTEM_PROMPT,
        )
    
    async def invoke(self, query: str, context: str = "") -> dict:
        """
        Process a query and return the result.
        
        Args:
            query: User's question
            context: Optional context from previous steps
            
        Returns:
            Dict with 'answer', 'products', 'success', 'confidence'
        """
        full_query = query
        if context:
            full_query = f"Context: {context}\n\nQuestion: {query}"
        
        result = await self._agent.ainvoke({
            "messages": [{"role": "user", "content": full_query}]
        })
        
        # Extract final message and products from tool calls
        messages = result.get("messages", [])
        products = []
        confidences = []
        answer = "No response generated."
        
        for msg in messages:
            # Extract products from tool messages (product_rag results)
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
                
                # Parse products from product_rag output format
                # Format: "- Name (Brand): $Price | Stock: N | Rating: R"
                lines = content.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("- [") and "Stock:" in line and "Rating:" in line:
                        try:
                            # Parse: "- [PROD-001] Name (Brand) - Category: $Price | Stock: N | Rating: R"
                            main_part = line[2:]  # Remove "- "
                            product_id_part, rest = main_part.split("] ", 1)
                            product_id = product_id_part[1:]  # Remove "["
                            
                            # Handle cases with or without category (backward compatibility)
                            if " - " in rest and ": $" in rest:
                                # New format with category
                                name_brand, price_rest = rest.split(": $", 1)  # Split at price
                                name_brand_part, category = name_brand.rsplit(" - ", 1)
                                name, brand = name_brand_part.rsplit(" (", 1)
                            else:
                                # Old format without category
                                name_brand, price_rest = rest.split("): $", 1)
                                name, brand = name_brand.rsplit(" (", 1)
                                category = ""

                            price_str, stock_rating = price_rest.split(" | Stock: ", 1)
                            stock_str, rating_str = stock_rating.split(" | Rating: ", 1)
                            
                            products.append({
                                "product_id": product_id,
                                "name": name,
                                "category": category.replace(")", ""), # Clean up in case of parsing errors
                                "brand": brand.replace(")", ""),
                                "price": float(price_str),
                                "stock_quantity": int(stock_str),
                                "average_rating": float(rating_str),
                            })
                        except (ValueError, IndexError):
                            # Skip malformed lines
                            continue
        
        # Get the final AI response
        if messages:
            last_message = messages[-1]
            answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Calculate aggregated confidence
        # If no tools used (direct answer), assume 0.5 or high confidence based on answer
        # Usually direct answer for product QA might mean general knowledge or "I don't know".
        # Let's default to 0.7 if no tools, or average of tools if used.
        if confidences:
            confidence = sum(confidences) / len(confidences)
        else:
            confidence = 0.5
            
        return {
            "answer": answer,
            "products": products,
            "success": True,
            "confidence": confidence,
        }
