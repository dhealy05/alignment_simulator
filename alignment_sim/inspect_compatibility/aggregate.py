"""Aggregation utilities for Inspect game logs.

This module provides functions to aggregate per-sample results
into mechanism-level summaries with action distributions,
outcome distributions, and average payoffs.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


def aggregate_results(log_path: str) -> Dict[str, Any]:
    """Aggregate game results from an Inspect eval log.

    Reads an Inspect evaluation log and aggregates results by mechanism,
    computing action distributions, outcome distributions, and average payoffs.

    Args:
        log_path: Path to the Inspect eval log (JSON or directory).

    Returns:
        Dictionary with aggregated results by mechanism.
    """
    # Import here to avoid requiring inspect_ai for basic usage
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        raise ImportError(
            "inspect_ai is required for log aggregation. "
            "Install with: pip install inspect_ai"
        )

    log = read_eval_log(log_path)

    by_mechanism: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for sample in log.samples:
        if not sample.scores:
            continue

        # Get metadata from first scorer (scores is a dict)
        score = next(iter(sample.scores.values()))
        if not score.metadata:
            continue

        mech_name = score.metadata.get("mechanism", "unknown")
        by_mechanism[mech_name].append(score.metadata)

    return build_summary(by_mechanism)


def build_summary(
    by_mechanism: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Build summary statistics from grouped results.

    Args:
        by_mechanism: Dictionary mapping mechanism names to lists of
            result metadata dictionaries.

    Returns:
        Summary dictionary with per-mechanism statistics.
    """
    summary: Dict[str, Any] = {}

    for mech_name, results in by_mechanism.items():
        if not results:
            continue

        n_samples = len(results)

        # Compute action distributions
        action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in results:
            actions = r.get("actions", {})
            for player, action in actions.items():
                action_counts[player][action] += 1

        action_distribution = {
            player: {action: count / n_samples for action, count in counts.items()}
            for player, counts in action_counts.items()
        }

        # Compute outcome distribution
        outcome_counts: Dict[str, int] = defaultdict(int)
        for r in results:
            outcome = r.get("outcome")
            if outcome:
                outcome_counts[outcome] += 1

        outcome_distribution = {
            outcome: count / n_samples for outcome, count in outcome_counts.items()
        }

        # Compute average payoffs
        payoff_sums: Dict[str, float] = defaultdict(float)
        payoff_counts: Dict[str, int] = defaultdict(int)
        for r in results:
            payoffs = r.get("payoffs", {})
            for player, val in payoffs.items():
                if val is not None and isinstance(val, (int, float)):
                    payoff_sums[player] += float(val)
                    payoff_counts[player] += 1

        avg_payoffs = {
            player: payoff_sums[player] / payoff_counts[player]
            if payoff_counts[player] > 0
            else 0.0
            for player in payoff_sums.keys()
        }

        # Compute total welfare
        total_welfare = sum(avg_payoffs.values())

        # Count parse errors
        parse_error_counts: Dict[str, int] = defaultdict(int)
        for r in results:
            parse_errors = r.get("parse_errors", {})
            for player, error in parse_errors.items():
                parse_error_counts[f"{player}:{error}"] += 1

        summary[mech_name] = {
            "n_samples": n_samples,
            "action_distribution": dict(action_distribution),
            "outcome_distribution": dict(outcome_distribution),
            "avg_payoffs": dict(avg_payoffs),
            "total_welfare": total_welfare,
        }

        if parse_error_counts:
            summary[mech_name]["parse_errors"] = dict(parse_error_counts)

    return summary


def write_summary(
    summary: Dict[str, Any],
    output_path: str,
    format: str = "yaml",
) -> None:
    """Write aggregated summary to a file.

    Args:
        summary: Summary dictionary from aggregate_results or build_summary.
        output_path: Path to write the summary file.
        format: Output format, either "yaml" or "json".
    """
    path = Path(output_path)

    if format == "yaml":
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {"game_summary": summary},
                f,
                sort_keys=False,
                default_flow_style=False,
            )
    elif format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"game_summary": summary}, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info("Wrote summary to %s", path)


def compare_mechanisms(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Compare outcomes across mechanisms.

    Useful for analyzing how different mechanism designs affect behavior.

    Args:
        summary: Summary dictionary from aggregate_results.

    Returns:
        Comparison dictionary highlighting differences across mechanisms.
    """
    mechanisms = list(summary.keys())

    if len(mechanisms) < 2:
        return {"note": "Need at least 2 mechanisms to compare"}

    # Collect all players and outcomes across mechanisms
    all_players = set()
    all_outcomes = set()
    for mech_data in summary.values():
        all_players.update(mech_data.get("action_distribution", {}).keys())
        all_outcomes.update(mech_data.get("outcome_distribution", {}).keys())

    comparison: Dict[str, Any] = {
        "mechanisms": mechanisms,
        "players": list(all_players),
        "outcomes": list(all_outcomes),
        "payoff_comparison": {},
        "outcome_comparison": {},
    }

    # Compare payoffs across mechanisms
    for player in all_players:
        comparison["payoff_comparison"][player] = {
            mech: summary[mech].get("avg_payoffs", {}).get(player, 0.0)
            for mech in mechanisms
        }

    # Compare outcome rates across mechanisms
    for outcome in all_outcomes:
        comparison["outcome_comparison"][outcome] = {
            mech: summary[mech].get("outcome_distribution", {}).get(outcome, 0.0)
            for mech in mechanisms
        }

    # Compute welfare comparison
    comparison["welfare_comparison"] = {
        mech: summary[mech].get("total_welfare", 0.0) for mech in mechanisms
    }

    return comparison
