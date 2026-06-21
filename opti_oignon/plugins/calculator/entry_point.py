"""
Calculator plugin for Opti-Oignon.

Provides safe mathematical expression evaluation via hook_tool_call
and hook_post_inference. Uses Python's ast module for safe parsing
(no eval/exec of arbitrary code).
"""

import ast
import math
import operator
import re
from typing import Any

# Plugin metadata (injected by loader)
__plugin_name__: str = "calculator"
__plugin_version__: str = "1.0.0"

# Configuration defaults
_MAX_EXPR_LENGTH = 500
_PRECISION = 10

# Safe operators for AST evaluation
_SAFE_OPERATORS = {
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

# Safe math functions and constants
_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "radians": math.radians,
    "degrees": math.degrees,
    "hypot": math.hypot,
}

_SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# Pattern to detect calc requests in text
_CALC_PATTERN = re.compile(
    r"\bcalc(?:ulate)?\s*[:\(]\s*(.+?)\s*[\)\n]?"
    r"|"
    r"\beval(?:uate)?\s*[:\(]\s*(.+?)\s*[\)\n]?",
    re.IGNORECASE,
)


class CalculatorError(Exception):
    """Raised when expression evaluation fails."""


def _safe_eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node with safety restrictions."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise CalculatorError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise CalculatorError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval_node(node.operand))

    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise CalculatorError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        # Guard against huge exponents
        if isinstance(node.op, ast.Pow):
            if isinstance(right, (int, float)) and abs(right) > 10000:
                raise CalculatorError("Exponent too large (max 10000)")
        return op_fn(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("Only named function calls are allowed")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise CalculatorError(f"Unknown function: {func_name}")
        args = [_safe_eval_node(arg) for arg in node.args]
        return _SAFE_FUNCTIONS[func_name](*args)

    if isinstance(node, ast.Name):
        name = node.id
        if name in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[name]
        if name in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[name]
        raise CalculatorError(f"Unknown variable: {name}")

    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_node(elt) for elt in node.elts)

    if isinstance(node, ast.List):
        return [_safe_eval_node(elt) for elt in node.elts]

    raise CalculatorError(f"Unsupported expression: {type(node).__name__}")


def evaluate(expression: str) -> float | int | complex:
    """Safely evaluate a mathematical expression.

    Parameters
    ----------
    expression : str
        Mathematical expression to evaluate.

    Returns
    -------
    float, int, or complex
        The result of the evaluation.

    Raises
    ------
    CalculatorError
        If the expression is invalid or unsafe.
    """
    expr = expression.strip()
    if not expr:
        raise CalculatorError("Empty expression")
    if len(expr) > _MAX_EXPR_LENGTH:
        raise CalculatorError(
            f"Expression too long ({len(expr)} > {_MAX_EXPR_LENGTH} chars)"
        )

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Syntax error: {exc}") from exc

    return _safe_eval_node(tree)


def format_result(value: Any) -> str:
    """Format a numeric result for display."""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == int(value) and not math.isinf(value):
            return str(int(value))
        return f"{value:.{_PRECISION}g}"
    if isinstance(value, complex):
        return str(value)
    return str(value)


# =========================================================================
# Hook implementations
# =========================================================================

def hook_tool_call(ctx: Any) -> dict[str, Any] | None:
    """Handle tool_call hook: evaluate math expressions.

    Expects ctx.data to contain:
        tool_name: str — if "calculator" or "calc", evaluate expression
        expression: str — the math expression
    """
    tool_name = ctx.data.get("tool_name", "")
    if tool_name not in ("calculator", "calc", "math"):
        return None

    expression = ctx.data.get("expression", "")
    if not expression:
        return {"result": None, "error": "No expression provided"}

    try:
        result = evaluate(expression)
        formatted = format_result(result)
        return {
            "result": formatted,
            "raw_result": result,
            "expression": expression,
            "error": None,
        }
    except CalculatorError as exc:
        return {
            "result": None,
            "expression": expression,
            "error": str(exc),
        }


def hook_post_inference(ctx: Any) -> dict[str, Any] | None:
    """Handle post_inference hook: detect and evaluate calc() in responses.

    Scans the LLM response text for calc(...) or calculate(...) patterns
    and appends computed results.
    """
    response_text = ctx.data.get("response", "")
    if not response_text:
        return None

    matches = _CALC_PATTERN.findall(response_text)
    if not matches:
        return None

    results: list[dict[str, str]] = []
    for groups in matches:
        expr = groups[0] or groups[1]
        if not expr.strip():
            continue
        try:
            result = evaluate(expr.strip())
            results.append({
                "expression": expr.strip(),
                "result": format_result(result),
            })
        except CalculatorError:
            pass

    if results:
        return {"calculator_results": results}
    return None


# Hook registry dict (alternative to hook_ prefix functions)
HOOKS = {
    "tool_call": hook_tool_call,
    "post_inference": hook_post_inference,
}


def init() -> None:
    """Plugin initialization (called by PluginLoader)."""
    pass


def shutdown() -> None:
    """Plugin shutdown (called by PluginLoader)."""
    pass
