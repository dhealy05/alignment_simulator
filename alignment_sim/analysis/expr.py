from __future__ import annotations

import ast
from typing import Any, Dict, Iterable, Set


class ExprError(ValueError):
    pass


def eval_arith(expr: str, context: Dict[str, Any]) -> float:
    node = _parse_arith(expr, set(context.keys()))
    env = {"__builtins__": {}}
    env.update(context)
    return float(eval(compile(node, "<arith>", "eval"), env, {}))


def eval_bool(expr: str, context: Dict[str, Any]) -> bool:
    node = _parse_bool(expr, set(context.keys()))
    env = {"__builtins__": {}}
    env.update(context)
    return bool(eval(compile(node, "<bool>", "eval"), env, {}))


def validate_arith(expr: str, allowed_names: Iterable[str]) -> None:
    _parse_arith(expr, set(allowed_names))


def validate_bool(expr: str, allowed_names: Iterable[str]) -> None:
    _parse_bool(expr, set(allowed_names))


def _parse_arith(expr: str, allowed_names: Set[str]) -> ast.Expression:
    node = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )

    for n in ast.walk(node):
        if not isinstance(n, allowed_nodes):
            raise ExprError(f"Unsupported arithmetic expression: {expr!r}")
        if isinstance(n, ast.Name) and n.id not in allowed_names:
            raise ExprError(f"Unknown name {n.id!r} in arithmetic expression: {expr!r}")
    return node


def _parse_bool(expr: str, allowed_names: Set[str]) -> ast.Expression:
    node = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.UnaryOp,
        ast.BinOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.USub,
        ast.UAdd,
    )

    for n in ast.walk(node):
        if not isinstance(n, allowed_nodes):
            raise ExprError(f"Unsupported boolean expression: {expr!r}")
        if isinstance(n, ast.Name) and n.id not in allowed_names:
            raise ExprError(f"Unknown name {n.id!r} in boolean expression: {expr!r}")
    return node
