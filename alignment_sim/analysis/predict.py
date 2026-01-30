from __future__ import annotations

import logging
import operator
import re
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
}


def parse_condition(condition: str) -> Dict[str, Any]:
    parts = [p.strip() for p in condition.split(",") if p.strip()]
    parsed: Dict[str, Any] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = [x.strip() for x in part.split("=", 1)]
        parsed[key] = _parse_value(value)
    return parsed


def parse_expected(expected: str) -> Tuple[str, str, float, str] | None:
    cleaned = expected.strip()
    prob_match = re.match(r"^\s*P\((\w+)\)\s*(<=|>=|==|=|<|>)\s*([-+]?\d*\.?\d+)\s*$", cleaned)
    if prob_match:
        var, op, value = prob_match.groups()
        if op == "=":
            op = "=="
        return var, op, float(value), "prob"

    numeric_match = re.match(r"^\s*([A-Za-z_]\w*)\s*(<=|>=|==|=|<|>)\s*([-+]?\d*\.?\d+)\s*$", cleaned)
    if not numeric_match:
        return None
    var, op, value = numeric_match.groups()
    if op == "=":
        op = "=="
    return var, op, float(value), "value"


def check_predictions(results: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    pred_type = cfg["prediction"]["type"]
    if pred_type == "invariant":
        return check_invariant(results, cfg)

    logger.debug("Checking %d predictions against %d results", len(cfg["prediction"]["expects"]), len(results))
    checks = []
    for pred in cfg["prediction"]["expects"]:
        condition = pred["condition"]
        expected = pred["outcome"]
        logger.debug("Checking prediction: %s -> %s", condition, expected)
        cond_map = parse_condition(condition)
        parsed = parse_expected(expected)

        matches = [r for r in results if _cell_matches(r, cond_map)]

        if parsed is None:
            checks.append(
                {
                    "condition": condition,
                    "expected": expected,
                    "observed": None,
                    "passed": False,
                    "n_cells": len(matches),
                    "note": "Could not parse expected outcome",
                }
            )
            continue

        metric, op, threshold, kind = parsed
        if len(matches) == 0:
            checks.append(
                {
                    "condition": condition,
                    "expected": expected,
                    "observed": None,
                    "passed": False,
                    "n_cells": 0,
                    "note": "No matching cells for condition",
                }
            )
            continue

        values = [r.get(metric) for r in matches if metric in r]
        if not values:
            checks.append(
                {
                    "condition": condition,
                    "expected": expected,
                    "observed": None,
                    "passed": False,
                    "n_cells": len(matches),
                    "note": f"Metric {metric!r} not found in results",
                }
            )
            continue

        numeric_values = []
        for v in values:
            if v is None:
                continue
            if isinstance(v, bool):
                numeric_values.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                numeric_values.append(float(v))

        if not numeric_values:
            checks.append(
                {
                    "condition": condition,
                    "expected": expected,
                    "observed": None,
                    "passed": False,
                    "n_cells": len(matches),
                    "note": f"Metric {metric!r} has no numeric values",
                }
            )
            continue

        observed = float(np.mean(numeric_values))
        passed = OPS[op](observed, threshold)
        checks.append(
            {
                "condition": condition,
                "expected": expected,
                "observed": observed,
                "passed": bool(passed),
                "n_cells": len(matches),
                "kind": kind,
            }
        )

    return checks


def check_invariant(results: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = []
    for pred in cfg["prediction"]["expects"]:
        condition = pred["condition"]
        expected = pred["outcome"]
        tolerance = float(pred.get("tolerance", 0.0))
        cond_map = parse_condition(condition)
        matches = [r for r in results if _cell_matches(r, cond_map)]

        if not matches:
            checks.append(
                {
                    "condition": condition,
                    "expected": expected,
                    "observed": None,
                    "passed": False,
                    "n_cells": 0,
                    "note": "No matching cells for condition",
                }
            )
            continue

        violations = 0
        eval_errors = 0
        total_checked = 0
        for r in matches:
            measurements = r.get("_measurements", {})
            n_samples = r.get("_n_samples") or max((len(v) for v in measurements.values()), default=0)
            base_context = {k: v for k, v in r.items() if not k.startswith("_")}
            if n_samples == 0:
                try:
                    ok = _eval_outcome(expected, base_context)
                except Exception:
                    ok = False
                    eval_errors += 1
                total_checked += 1
                if not ok:
                    violations += 1
                continue
            for i in range(n_samples):
                context = dict(base_context)
                for name, values in measurements.items():
                    if i < len(values):
                        context[name] = values[i]
                try:
                    ok = _eval_outcome(expected, context)
                except Exception:
                    ok = False
                    eval_errors += 1
                total_checked += 1
                if not ok:
                    violations += 1

        violation_rate = violations / total_checked if total_checked else 0.0
        passed = violation_rate <= tolerance
        note = None
        if eval_errors:
            note = f"{eval_errors} evaluation error(s) while checking outcome"

        checks.append(
            {
                "condition": condition,
                "expected": expected,
                "observed": f"{violations}/{total_checked} violations",
                "passed": passed,
                "n_cells": len(matches),
                "n_samples_checked": total_checked,
                "violations": violations,
                "violation_rate": violation_rate,
                "tolerance": tolerance,
                "note": note,
            }
        )
    return checks


def _parse_value(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _normalize_expr(expr: str) -> str:
    expr = re.sub(r"\bAND\b", "and", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bOR\b", "or", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bNOT\b", "not", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\btrue\b", "True", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bfalse\b", "False", expr, flags=re.IGNORECASE)
    expr = re.sub(r"(?<![<>=!])=(?!=)", "==", expr)
    return expr


def _eval_outcome(expr: str, context: Dict[str, Any]) -> bool:
    import ast

    expr = _normalize_expr(expr)
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
            raise ValueError(f"Unsupported expression in invariant outcome: {expr!r}")
        if isinstance(n, ast.Name) and n.id not in context:
            raise KeyError(f"Unknown name in invariant outcome: {n.id!r}")

    env = {"__builtins__": {}}
    env.update(context)
    return bool(eval(compile(node, "<outcome>", "eval"), env, {}))


def _cell_matches(cell: Dict[str, Any], cond_map: Dict[str, Any]) -> bool:
    for key, expected in cond_map.items():
        if key not in cell:
            return False
        actual = cell[key]
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if abs(float(actual) - float(expected)) > 1e-9:
                return False
        else:
            if str(actual) != str(expected):
                return False
    return True
