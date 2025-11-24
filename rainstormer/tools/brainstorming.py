"""Brainstorming tools for MCTS simulations."""

from __future__ import annotations

import ast
import operator
import math
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from ..schema.llm.message import Tool
from ..utils.logger import logger


# =============================================================================
# Tool Definitions
# =============================================================================


class WebSearchTool(BaseModel):
    """Web search tool for grounded brainstorming."""

    name: str = "web_search"
    description: str = (
        "Search the web for relevant information to support brainstorming. "
        "Use this to find facts, statistics, examples, or prior art."
    )

    class Parameters(BaseModel):
        query: str = Field(description="Search query (be specific)")
        max_results: int = Field(default=3, ge=1, le=5, description="Number of results")

    async def execute(self, query: str, max_results: int = 3) -> dict[str, Any]:
        """
        Execute web search.

        Note: This is a placeholder implementation. In production, integrate
        with a real search API (Tavily, SerpAPI, Brave Search, etc.)
        """
        logger.info(f"Web search: '{query}' (max_results={max_results})")

        # Placeholder - returns mock results for now
        # TODO: Integrate with actual search API
        return {
            "success": True,
            "query": query,
            "results": [
                {
                    "title": f"Result {i + 1} for: {query}",
                    "snippet": f"This is a placeholder result for '{query}'. "
                    "Integrate with a real search API for actual results.",
                    "url": f"https://example.com/result{i + 1}",
                }
                for i in range(max_results)
            ],
            "note": "Placeholder results - integrate with real search API",
        }


class CalculateTool(BaseModel):
    """Calculator tool for numerical reasoning in brainstorming."""

    name: str = "calculate"
    description: str = (
        "Perform mathematical calculations. Use for estimates, projections, "
        "ROI calculations, market sizing, or any numerical reasoning."
    )

    class Parameters(BaseModel):
        expression: str = Field(
            description="Mathematical expression (e.g., '1000 * 0.15 * 12')"
        )
        context: str = Field(default="", description="What this calculation represents")

    # Safe operators for eval
    SAFE_OPERATORS: ClassVar[dict] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Safe functions
    SAFE_FUNCTIONS: ClassVar[dict] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pow": pow,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    # Safe constants
    SAFE_CONSTANTS: ClassVar[dict] = {
        "pi": math.pi,
        "e": math.e,
    }

    def _safe_eval(self, node: ast.AST) -> float:
        """Safely evaluate an AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        if isinstance(node, ast.Name):
            name = node.id.lower()
            if name in self.SAFE_CONSTANTS:
                return self.SAFE_CONSTANTS[name]
            raise ValueError(f"Unknown constant: {node.id}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.SAFE_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            return self.SAFE_OPERATORS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.SAFE_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._safe_eval(node.operand)
            return self.SAFE_OPERATORS[op_type](operand)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are supported")
            func_name = node.func.id.lower()
            if func_name not in self.SAFE_FUNCTIONS:
                raise ValueError(f"Unknown function: {func_name}")
            args = [self._safe_eval(arg) for arg in node.args]
            return self.SAFE_FUNCTIONS[func_name](*args)

        raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    async def execute(self, expression: str, context: str = "") -> dict[str, Any]:
        """
        Execute a mathematical calculation safely.

        Uses AST parsing to avoid arbitrary code execution.
        """
        logger.info(f"Calculate: '{expression}' (context: {context})")

        try:
            # Parse and evaluate safely
            tree = ast.parse(expression, mode="eval")
            result = self._safe_eval(tree.body)

            return {
                "success": True,
                "expression": expression,
                "result": result,
                "formatted": f"{result:,.2f}"
                if isinstance(result, float)
                else str(result),
                "context": context,
            }
        except Exception as e:
            logger.warning(f"Calculation failed: {e}")
            return {
                "success": False,
                "expression": expression,
                "error": str(e),
                "context": context,
            }


# =============================================================================
# Tool Registry
# =============================================================================

# Instantiate default tools
_web_search = WebSearchTool()
_calculate = CalculateTool()

# Tool registry as list of Tool schemas for LLM
BRAINSTORMING_TOOLS: list[Tool] = [
    Tool(
        name="web_search",
        description=_web_search.description,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (be specific)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (1-5)",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["query"],
        },
        handler=_web_search.execute,
    ),
    Tool(
        name="calculate",
        description=_calculate.description,
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression (e.g., '1000 * 0.15 * 12')",
                },
                "context": {
                    "type": "string",
                    "description": "What this calculation represents",
                    "default": "",
                },
            },
            "required": ["expression"],
        },
        handler=_calculate.execute,
    ),
]

# Tool name to executor mapping
_TOOL_EXECUTORS = {
    "web_search": _web_search.execute,
    "calculate": _calculate.execute,
}


# =============================================================================
# Execution Helpers
# =============================================================================


async def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a tool by name with given arguments.

    Args:
        tool_name: Name of the tool to execute.
        arguments: Tool arguments as a dictionary.

    Returns:
        Tool execution result.
    """
    executor = _TOOL_EXECUTORS.get(tool_name)
    if not executor:
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}",
        }

    try:
        return await executor(**arguments)
    except Exception as e:
        logger.error(f"Tool execution failed: {tool_name} - {e}")
        return {
            "success": False,
            "tool": tool_name,
            "error": str(e),
        }


def summarize_tool_result(
    tool_name: str,
    result: dict[str, Any],
    max_tokens: int = 200,
) -> str:
    """
    Summarize a tool result for inclusion in context.

    Args:
        tool_name: Name of the tool that was executed.
        result: Tool execution result.
        max_tokens: Maximum tokens for summary (approximate).

    Returns:
        Summarized result string.
    """
    if not result.get("success", False):
        error = result.get("error", "Unknown error")
        return f"[{tool_name} failed: {error}]"

    if tool_name == "web_search":
        query = result.get("query", "")
        results = result.get("results", [])
        if not results:
            return f"[web_search '{query}': No results found]"

        # Summarize top results
        summaries = []
        for r in results[:3]:
            title = r.get("title", "")[:50]
            snippet = r.get("snippet", "")[:100]
            summaries.append(f"- {title}: {snippet}")

        summary = f"[web_search '{query}':\n" + "\n".join(summaries) + "]"
        # Rough token limit (4 chars per token)
        if len(summary) > max_tokens * 4:
            summary = summary[: max_tokens * 4] + "...]"
        return summary

    if tool_name == "calculate":
        expr = result.get("expression", "")
        formatted = result.get("formatted", str(result.get("result", "?")))
        context = result.get("context", "")
        if context:
            return f"[calculate: {expr} = {formatted} ({context})]"
        return f"[calculate: {expr} = {formatted}]"

    # Generic fallback
    return f"[{tool_name}: {str(result)[: max_tokens * 4]}]"
