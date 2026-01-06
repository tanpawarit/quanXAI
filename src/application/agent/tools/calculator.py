"""Calculator Tool - Safe math expression evaluation."""

import math
from src.application.agent.tools.base import ToolResult


class CalculatorTool:
    """Calculator for accurate math calculations."""
    
    async def execute(self, expression: str = None, **kwargs) -> ToolResult:
        """Evaluate a math expression safely."""
        try:
            if not expression:
                return ToolResult(answer="No expression provided", error="Missing expression", confidence=0.0)
            
            result = self._safe_eval(expression.strip())
            return ToolResult(answer=f"{expression} = {result}", confidence=1.0, metadata={"result": result})
        except Exception as e:
            return ToolResult(answer="", error=f"Calculation failed: {str(e)}", confidence=0.0)
    
    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate a math expression."""
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "ceil": math.ceil, "floor": math.floor,
        }
        return eval(expression, {"__builtins__": {}}, allowed_names)
