from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import yaml

from ..analysis.expr import eval_arith
from .gamespec import apply_mechanism, load_gamespec

logger = logging.getLogger(__name__)


def summarize_game(suite_path: str, output_path: str | None = None) -> str:
    suite_manifest, suite_dir = _load_suite(suite_path)
    suite = suite_manifest["suite"]

    gamespec_path = _resolve_path(suite_dir, suite["paths"]["gamespec"])
    spec = load_gamespec(gamespec_path)

    results_dir = _resolve_path(suite_dir, suite["paths"]["results_dir"])

    outputs_by_config: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for cfg_entry in suite.get("configs", []):
        cfg_id = cfg_entry.get("id", "config")
        cert_path = os.path.join(results_dir, cfg_id, "certificate.yaml")
        if not os.path.exists(cert_path):
            raise ValueError(f"Missing certificate for config {cfg_id!r}: {cert_path}")
        with open(cert_path, "r", encoding="utf-8") as f:
            cert = yaml.safe_load(f)
        raw_results = cert.get("raw_results")
        if raw_results is None:
            raise ValueError(f"Certificate for {cfg_id!r} is missing raw_results (run-suite must include them).")
        cfg_outputs = _compute_outputs(raw_results, cfg_entry.get("outputs", []))
        outputs_by_config[cfg_id] = cfg_outputs

    strategy_estimates = _build_strategy_estimates(outputs_by_config, suite.get("aggregation", {}))
    gaming_gap = _compute_gaming_gap(outputs_by_config, suite.get("aggregation", {}))

    mechanisms = spec.get("mechanisms", [])
    mechanism_summaries = {}
    welfare_table: List[Dict[str, Any]] = []
    observed_strategies: List[Dict[str, Any]] = []
    stability_summary: Dict[str, Any] | None = None

    for mech in mechanisms:
        mech_name = mech["name"]
        params = apply_mechanism(spec["parameters"], mech)
        metric_values = _metric_values_for_mechanism(strategy_estimates, mech_name)
        if gaming_gap and mech_name in gaming_gap:
            metric_values.update(gaming_gap[mech_name])
        observed_strategies.append({"mechanism": mech_name, **metric_values})

        welfare = _compute_welfare(spec, suite.get("aggregation", {}), params, metric_values)
        if welfare:
            welfare_table.append({"mechanism": mech_name, "welfare": welfare})

        mechanism_summaries[mech_name] = {"observed": metric_values, "welfare": welfare}

    stability_summary = _compute_stability(
        spec,
        suite.get("aggregation", {}),
        strategy_estimates,
    )

    pareto = _compute_pareto_frontier(suite.get("pareto", {}), welfare_table)
    thresholds = _compute_thresholds(suite.get("thresholds", []), observed_strategies)

    summary = {
        "equilibrium_summary": {
            "game": spec["game"]["name"],
            "model": _infer_model(results_dir),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mechanisms": mechanism_summaries,
            "observed_strategies": observed_strategies,
            "welfare_table": welfare_table,
            "stability": stability_summary,
            "pareto": pareto,
            "thresholds": thresholds,
        }
    }

    output_path = output_path or _default_report_path(suite, suite_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)
    logger.info("Wrote equilibrium summary to: %s", output_path)
    return output_path


def _load_suite(path: str) -> tuple[Dict[str, Any], str]:
    if os.path.isdir(path):
        suite_path = os.path.join(path, "suite.yaml")
    else:
        suite_path = path
    with open(suite_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "suite" not in data:
        raise ValueError("suite.yaml must contain top-level 'suite' mapping.")
    suite_dir = os.path.dirname(os.path.abspath(suite_path))
    return data, suite_dir


def _resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _compute_outputs(raw_results: List[Dict[str, Any]], outputs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {}
    for out in outputs:
        name = out["name"]
        metric = out["metric"]
        summarize = out.get("summarize", "mean")
        group_by = out.get("group_by", [])
        results[name] = _group_and_summarize(raw_results, metric, group_by, summarize)
    return results


def _group_and_summarize(
    raw_results: List[Dict[str, Any]],
    metric: str,
    group_by: List[str],
    summarize: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Any]] = {}
    for r in raw_results:
        key = tuple(r.get(g) for g in group_by) if group_by else ("__all__",)
        values = r.get("_measurements", {}).get(metric)
        if values is None:
            values = [r.get(metric)]
        grouped.setdefault(key, []).extend(values)

    rows: List[Dict[str, Any]] = []
    for key, values in grouped.items():
        cleaned = [v for v in values if v is not None]
        value = _summarize_values(cleaned, summarize)
        row = {group_by[i]: key[i] for i in range(len(group_by))}
        row["value"] = value
        row["n"] = len(cleaned)
        rows.append(row)
    return rows


def _summarize_values(values: List[Any], summarize: str) -> float | None:
    if not values:
        return None
    if summarize == "proportion":
        numeric = [1.0 if bool(v) else 0.0 for v in values]
        return sum(numeric) / len(numeric)
    numeric = []
    for v in values:
        if isinstance(v, bool):
            numeric.append(1.0 if v else 0.0)
        elif isinstance(v, (int, float)):
            numeric.append(float(v))
    return sum(numeric) / len(numeric) if numeric else None


def _build_strategy_estimates(
    outputs_by_config: Dict[str, Dict[str, List[Dict[str, Any]]]],
    aggregation: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    strategy = {}
    for metric_name, info in aggregation.get("strategy_estimates", {}).items():
        cfg_id = info["from_config"]
        output_name = info["metric"]
        rows = outputs_by_config.get(cfg_id, {}).get(output_name, [])
        strategy[metric_name] = {"rows": rows, "action": info.get("action")}
    return strategy


def _compute_gaming_gap(
    outputs_by_config: Dict[str, Dict[str, List[Dict[str, Any]]]],
    aggregation: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    cfg = aggregation.get("gaming_gap")
    if not isinstance(cfg, dict):
        return {}
    cfg_id = cfg.get("from_config")
    output_name = cfg.get("metric")
    if not cfg_id or not output_name:
        return {}
    rows = outputs_by_config.get(cfg_id, {}).get(output_name, [])
    if not rows:
        return {}

    signal_var = cfg.get("signal_var", "signal")
    normal_label = cfg.get("signal_normal", "normal")
    suspicious_label = cfg.get("signal_suspicious", "suspicious")
    base_metric = cfg.get("base_metric") or output_name.replace("_by_signal", "")

    grouped: Dict[str, Dict[str, float]] = {}
    for row in rows:
        group = row.get("mechanism", "__all__")
        signal = row.get(signal_var)
        value = row.get("value")
        if signal is None or value is None:
            continue
        grouped.setdefault(str(group), {})[str(signal)] = float(value)

    results: Dict[str, Dict[str, Any]] = {}
    for group, signals in grouped.items():
        normal_val = signals.get(normal_label)
        suspicious_val = signals.get(suspicious_label)
        if normal_val is None or suspicious_val is None:
            continue
        results[group] = {
            f"{base_metric}_normal": normal_val,
            f"{base_metric}_suspicious": suspicious_val,
            "gaming_gap": normal_val - suspicious_val,
        }
    return results


def _metric_values_for_mechanism(
    strategy_estimates: Dict[str, Dict[str, Any]],
    mechanism: str,
) -> Dict[str, Any]:
    metrics = {}
    for metric_name, entry in strategy_estimates.items():
        rows = entry.get("rows", [])
        value = None
        for row in rows:
            if row.get("mechanism") == mechanism:
                value = row.get("value")
                break
        metrics[metric_name] = value
    return metrics


def _compute_welfare(
    spec: Dict[str, Any],
    aggregation: Dict[str, Any],
    params: Dict[str, float],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    welfare_cfg = aggregation.get("welfare", {})
    assumptions = welfare_cfg.get("assumptions", {})
    outcome_formulas = assumptions.get("expected_outcome_probs")
    if not outcome_formulas:
        return {}

    context = dict(params)
    for k, v in metrics.items():
        if v is not None:
            context[k] = v

    outcome_probs = {}
    for outcome_name, expr in outcome_formulas.items():
        try:
            outcome_probs[outcome_name] = float(eval_arith(expr, context))
        except Exception:
            return {}

    payoffs = {}
    for player, outcome_payoffs in spec["payoffs"].items():
        total = 0.0
        for outcome_name, expr in outcome_payoffs.items():
            prob = outcome_probs.get(outcome_name)
            if prob is None:
                continue
            value = eval_arith(expr, params)
            total += prob * value
        payoffs[player] = total

    welfare_values = {}
    welfare_defs = spec.get("welfare", [])
    if welfare_defs:
        for w in welfare_defs:
            welfare_values[w["name"]] = _eval_welfare_formula(w["formula"], payoffs)
    else:
        welfare_values["total"] = sum(payoffs.values())
    for player, value in payoffs.items():
        welfare_values[player] = value
    welfare_values["payoffs"] = payoffs
    return welfare_values


def _eval_welfare_formula(formula: str, payoffs: Dict[str, float]) -> float:
    expr = formula
    for player, value in payoffs.items():
        expr = expr.replace(f"payoff({player})", str(value))
    return eval_arith(expr, {})


def _compute_stability(
    spec: Dict[str, Any],
    aggregation: Dict[str, Any],
    strategy_estimates: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    stability_cfg = aggregation.get("stability", {})
    if not stability_cfg:
        return None
    br_cfg = stability_cfg.get("best_response_gap", {})
    if not br_cfg:
        return None
    welfare_cfg = aggregation.get("welfare", {})
    outcome_formulas = welfare_cfg.get("assumptions", {}).get("expected_outcome_probs")
    if not outcome_formulas:
        return None

    metric_name = None
    uses = br_cfg.get("uses", [])
    if isinstance(uses, list) and uses:
        metric_name = uses[0]
    if metric_name is None or metric_name not in strategy_estimates:
        return None

    action_label = br_cfg.get("action") or strategy_estimates[metric_name].get("action")
    if not action_label:
        return None

    epsilon = float(stability_cfg.get("epsilon", 0.05))
    player = br_cfg.get("player") or _first_player(spec)
    rows = strategy_estimates[metric_name].get("rows", [])

    gaps = []
    for row in rows:
        mechanism = row.get("mechanism")
        if mechanism is None:
            continue
        p = row.get("value")
        if p is None:
            continue
        params = apply_mechanism(spec["parameters"], _find_mechanism(spec, mechanism))
        current = _expected_payoff_for_metric(spec, params, metric_name, p, outcome_formulas, player)
        payoff_action = _expected_payoff_for_metric(spec, params, metric_name, 1.0, outcome_formulas, player)
        payoff_other = _expected_payoff_for_metric(spec, params, metric_name, 0.0, outcome_formulas, player)
        best_dev = max(payoff_action, payoff_other)
        gaps.append(best_dev - current)

    if not gaps:
        return None

    gap_mean = sum(gaps) / len(gaps)
    return {
        "player": player,
        "metric": metric_name,
        "action": action_label,
        "epsilon": epsilon,
        "gap_mean": gap_mean,
        "stable": gap_mean <= epsilon,
    }


def _expected_payoff_for_metric(
    spec: Dict[str, Any],
    params: Dict[str, float],
    metric_name: str,
    metric_value: float,
    outcome_formulas: Dict[str, str],
    player: str,
) -> float:
    context = dict(params)
    context[metric_name] = metric_value
    outcome_probs = {}
    for outcome_name, expr in outcome_formulas.items():
        outcome_probs[outcome_name] = float(eval_arith(expr, context))

    total = 0.0
    for outcome_name, expr in spec["payoffs"][player].items():
        prob = outcome_probs.get(outcome_name, 0.0)
        total += prob * eval_arith(expr, params)
    return total


def _first_player(spec: Dict[str, Any]) -> str:
    return spec["players"][0]["name"]


def _find_mechanism(spec: Dict[str, Any], name: str) -> Dict[str, Any]:
    for mech in spec.get("mechanisms", []):
        if mech["name"] == name:
            return mech
    raise ValueError(f"Unknown mechanism: {name!r}")


def _compute_pareto_frontier(pareto_cfg: Dict[str, Any], welfare_table: List[Dict[str, Any]]) -> Dict[str, Any]:
    x_key = pareto_cfg.get("x")
    y_key = pareto_cfg.get("y")
    if not x_key or not y_key:
        return {"frontier": []}

    frontier = []
    for row in welfare_table:
        if _is_dominated(row, welfare_table, x_key, y_key):
            continue
        frontier.append(row.get("mechanism"))
    return {"frontier": frontier, "x": x_key, "y": y_key}


def _is_dominated(row: Dict[str, Any], table: List[Dict[str, Any]], x_key: str, y_key: str) -> bool:
    x_val = _get_nested(row, x_key)
    y_val = _get_nested(row, y_key)
    if x_val is None or y_val is None:
        return True
    for other in table:
        if other is row:
            continue
        ox = _get_nested(other, x_key)
        oy = _get_nested(other, y_key)
        if ox is None or oy is None:
            continue
        if ox >= x_val and oy >= y_val and (ox > x_val or oy > y_val):
            return True
    return False


def _get_nested(row: Dict[str, Any], key: str) -> Any:
    parts = key.split(".")
    current: Any = row
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _compute_thresholds(thresholds: List[Dict[str, Any]], strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = {}
    for threshold in thresholds:
        name = threshold["name"]
        metric = threshold["metric"]
        condition = threshold["condition"]
        op, value = _parse_condition(condition)
        satisfied = []
        for row in strategies:
            metric_val = row.get(metric)
            if metric_val is None:
                continue
            if op(metric_val, value):
                satisfied.append(row.get("mechanism"))
        results[name] = {"metric": metric, "condition": condition, "satisfied_by": satisfied}
    return results


def _parse_condition(cond: str):
    match = re.match(r"^\s*(<=|>=|==|<|>)\s*([-+]?\d*\.?\d+)\s*$", cond)
    if not match:
        raise ValueError(f"Unsupported condition: {cond!r}")
    op, value = match.groups()
    value_f = float(value)
    ops = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
    }
    return ops[op], value_f


def _default_report_path(suite: Dict[str, Any], suite_dir: str) -> str:
    reports = suite.get("reports", [])
    for report in reports:
        if report.get("name") == "equilibrium_summary":
            return _resolve_path(suite_dir, report["output"])
    return _resolve_path(suite_dir, "results/equilibrium_summary.yaml")


def _infer_model(results_dir: str) -> str | None:
    # Try to infer model from any certificate
    for root, _, files in os.walk(results_dir):
        if "certificate.yaml" in files:
            with open(os.path.join(root, "certificate.yaml"), "r", encoding="utf-8") as f:
                cert = yaml.safe_load(f)
            return cert.get("model_tested")
    return None
