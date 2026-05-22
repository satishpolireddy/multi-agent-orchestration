"""CalculatorTool — safely evaluates mathematical expressions via AST (no eval())."""
from __future__ import annotations
import ast, logging, math, operator as op
from langchain_core.tools import BaseTool
logger = logging.getLogger(__name__)
ALLOWED_OPS = {ast.Add:op.add,ast.Sub:op.sub,ast.Mult:op.mul,ast.Div:op.truediv,ast.Pow:op.pow,ast.Mod:op.mod,ast.FloorDiv:op.floordiv,ast.USub:op.neg,ast.UAdd:op.pos}
ALLOWED_FUNCS = {"abs":abs,"round":round,"sqrt":math.sqrt,"log":math.log,"sin":math.sin,"cos":math.cos,"tan":math.tan,"pi":math.pi,"e":math.e}
def _safe_eval(node):
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.Name) and node.id in ALLOWED_FUNCS: return ALLOWED_FUNCS[node.id]
    if isinstance(node, ast.BinOp):
        f = ALLOWED_OPS.get(type(node.op))
        if not f: raise ValueError(f"Unsupported op: {type(node.op).__name__}")
        return f(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        f = ALLOWED_OPS.get(type(node.op))
        if not f: raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return f(_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        f = ALLOWED_FUNCS.get(node.func.id if isinstance(node.func, ast.Name) else "")
        if not f: raise ValueError("Unsupported function")
        return f(*[_safe_eval(a) for a in node.args])
    raise ValueError(f"Unsupported: {type(node).__name__}")
class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Evaluates arithmetic expressions safely."
    def _run(self, expr):
        try: return str(_safe_eval(ast.parse(expr.strip(), mode="eval").body))
        except Exception as e: return f"Error: {e}"
    async def _arun(self, expr): return self._run(expr)
