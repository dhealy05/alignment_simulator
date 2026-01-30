from __future__ import annotations

import ast
import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ORDINAL_MAP = {"low": 0.2, "medium": 0.5, "high": 0.8}


def fit_linear(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    logger.debug("Fitting linear model: X shape=%s, y shape=%s", X.shape, y.shape)
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    betas, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    y_pred = X_with_intercept @ betas
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    logger.debug("Fit result: intercept=%.4f, coefs=%s, R²=%.4f", betas[0], betas[1:], r_squared)
    return {"intercept": float(betas[0]), "coefs": betas[1:], "r_squared": r_squared}


def fit_logistic(X: np.ndarray, y: np.ndarray, fit_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    logger.debug("Fitting logistic model: X shape=%s, y shape=%s", X.shape, y.shape)
    fit_cfg = fit_cfg or {}
    max_iter = int(fit_cfg.get("max_iter", 2000))
    lr = float(fit_cfg.get("lr", 0.05))
    tol = float(fit_cfg.get("tol", 1e-6))
    l2 = float(fit_cfg.get("l2", 0.01))

    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    n_features = X_with_intercept.shape[1]
    betas = np.zeros(n_features, dtype=float)

    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    converged = False
    for _ in range(max_iter):
        z = X_with_intercept @ betas
        p = sigmoid(z)
        gradient = X_with_intercept.T @ (p - y) / len(y)
        if l2:
            gradient[1:] += l2 * betas[1:]
        new_betas = betas - lr * gradient
        if np.linalg.norm(new_betas - betas) < tol:
            betas = new_betas
            converged = True
            break
        betas = new_betas

    p_pred = sigmoid(X_with_intercept @ betas)
    eps = 1e-10
    ll_model = float(np.sum(y * np.log(p_pred + eps) + (1 - y) * np.log(1 - p_pred + eps)))
    p_null = float(np.mean(y)) if len(y) else 0.0
    ll_null = float(np.sum(y * np.log(p_null + eps) + (1 - y) * np.log(1 - p_null + eps)))
    pseudo_r2 = 1 - ll_model / ll_null if ll_null != 0 else 0.0

    logger.debug(
        "Logistic fit result: intercept=%.4f, coefs=%s, pseudo R²=%.4f, converged=%s",
        betas[0],
        betas[1:],
        pseudo_r2,
        converged,
    )
    return {
        "intercept": float(betas[0]),
        "coefs": betas[1:],
        "pseudo_r2": pseudo_r2,
        "odds_ratios": np.exp(betas[1:]),
        "converged": converged,
        "max_iter": max_iter,
        "lr": lr,
        "tol": tol,
        "l2": l2,
    }


def build_feature_matrix(results: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    feature_names = extract_fit_features(cfg)
    var_defs = {v["name"]: v for v in cfg["variables"]["controlled"]}

    mappings: Dict[str, Dict[Any, float]] = {}
    for name in feature_names:
        var_def = var_defs[name]
        if var_def["type"] == "categorical":
            mappings[name] = infer_categorical_mapping(var_def)

    rows = []
    for r in results:
        row = []
        for name in feature_names:
            var_def = var_defs[name]
            value = r[name]
            row.append(encode_value(var_def, value, mappings.get(name)))
        rows.append(row)

    return np.array(rows, dtype=float), feature_names


def extract_fit_features(cfg: Dict[str, Any]) -> List[str]:
    model = cfg["prediction"]["fit"].get("model", "")
    controlled_names = [v["name"] for v in cfg["variables"]["controlled"]]
    features = []
    for name in controlled_names:
        if _name_in_model(name, model):
            features.append(name)
    return features if features else cfg["run"]["grid_over"]


def _name_in_model(name: str, model: str) -> bool:
    import re

    return re.search(rf"\b{re.escape(name)}\b", model) is not None


def encode_value(var_def: Dict[str, Any], value: Any, mapping: Dict[Any, float] | None) -> float:
    vtype = var_def["type"]
    if vtype in {"continuous", "binary"}:
        return float(value)
    if vtype == "categorical":
        if mapping is None:
            raise ValueError(f"No categorical mapping for {var_def['name']}")
        return float(mapping[value])
    raise ValueError(f"Unsupported variable type: {vtype!r}")


def infer_categorical_mapping(var_def: Dict[str, Any]) -> Dict[Any, float]:
    if "value_map" in var_def and isinstance(var_def["value_map"], dict):
        return {k: float(v) for k, v in var_def["value_map"].items()}

    values = var_def["values"]
    if isinstance(values, dict):
        numeric_map = {}
        for k, v in values.items():
            if isinstance(v, dict) and "numeric" in v:
                numeric_map[k] = float(v["numeric"])
        if len(numeric_map) == len(values):
            return numeric_map
        labels = list(values.keys())
    else:
        labels = list(values)

    if set(labels).issubset(DEFAULT_ORDINAL_MAP.keys()):
        return {k: DEFAULT_ORDINAL_MAP[k] for k in labels}

    if len(labels) == 1:
        return {labels[0]: 0.0}
    step = 1.0 / (len(labels) - 1)
    return {label: i * step for i, label in enumerate(labels)}


def build_fit_report(fit_raw: Dict[str, Any], feature_names: List[str]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"beta_0": fit_raw.get("intercept", 0.0), "intercept": fit_raw.get("intercept", 0.0)}
    if "r_squared" in fit_raw:
        report["r_squared"] = fit_raw["r_squared"]
    if "pseudo_r2" in fit_raw:
        report["pseudo_r2"] = fit_raw["pseudo_r2"]
    if "odds_ratios" in fit_raw:
        report["odds_ratios"] = {name: float(val) for name, val in zip(feature_names, fit_raw["odds_ratios"])}
    if "converged" in fit_raw:
        report["converged"] = fit_raw["converged"]

    for name, coef in zip(feature_names, fit_raw.get("coefs", [])):
        report[f"beta_{name}"] = float(coef)
    return report


def infer_variables(fit_report: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    inferred = {}
    for item in cfg["variables"].get("inferred", []):
        name = item["name"]
        formula = item["formula"]
        logger.debug("Inferring variable %s from formula: %s", name, formula)
        value = safe_eval(formula, fit_report)
        logger.debug("  %s = %.4f", name, value)
        inferred[name] = value
    return inferred


def safe_eval(expr: str, variables: Dict[str, Any]) -> float:
    node = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Call,
        ast.Attribute,
    )

    for n in ast.walk(node):
        if not isinstance(n, allowed_nodes):
            raise ValueError(f"Unsupported expression in inferred formula: {expr!r}")
        if isinstance(n, ast.Call):
            if not (isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)):
                raise ValueError(f"Unsupported call in inferred formula: {expr!r}")
            if n.func.value.id != "math":
                raise ValueError(f"Only math.* calls allowed in inferred formula: {expr!r}")

    env = {"__builtins__": {}, "math": math}
    env.update(variables)
    return float(eval(compile(node, "<expr>", "eval"), env, {}))
