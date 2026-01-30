"""Model comparison functionality for AlignmentSim."""
from __future__ import annotations

import copy
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import yaml

from .core.runner import AlignmentSimRunner

logger = logging.getLogger(__name__)


def run_comparison(
    cfg: Dict[str, Any],
    models: List[str],
    concurrency: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Run the same config across multiple models and collect results.

    Args:
        cfg: The experiment configuration
        models: List of model names to compare
        concurrency: Max concurrent API requests per model

    Returns:
        Dict mapping model name to certificate
    """
    results = {}

    for model in models:
        logger.info("Running experiment for model: %s", model)
        model_cfg = copy.deepcopy(cfg)
        model_cfg["run"]["model"] = model

        runner = AlignmentSimRunner(model_cfg, concurrency=concurrency)
        cert = runner.run()

        # Remove internal data
        cert.pop("_results", None)
        cert.pop("_config", None)

        results[model] = cert

    return results


def build_comparison_table(
    comparison: Dict[str, Dict[str, Any]],
) -> tuple[List[str], List[str], Dict[str, Dict[str, str]], Dict[str, str]]:
    """Build comparison data structures from raw results.

    Returns:
        Tuple of (models, scenarios, table_data, verdicts)
    """
    models = list(comparison.keys())

    # Extract scenario names from first model's predictions
    first_cert = comparison[models[0]]
    scenarios = [p["condition"] for p in first_cert["predictions"]]

    # Build comparison table
    table_data: Dict[str, Dict[str, str]] = {}
    for scenario in scenarios:
        table_data[scenario] = {}
        for model in models:
            cert = comparison[model]
            for pred in cert["predictions"]:
                if pred["condition"] == scenario:
                    observed = pred.get("observed")
                    passed = pred.get("passed", False)
                    if isinstance(observed, float):
                        cell = f"{observed:.0%}" if observed <= 1.0 else f"{observed:.2f}"
                    else:
                        cell = str(observed)
                    cell += " ✓" if passed else " ✗"
                    table_data[scenario][model] = cell
                    break

    verdicts = {model: comparison[model]["verdict"] for model in models}

    return models, scenarios, table_data, verdicts


def format_table(
    property_name: str,
    models: List[str],
    scenarios: List[str],
    table_data: Dict[str, Dict[str, str]],
    verdicts: Dict[str, str],
) -> str:
    """Format comparison as ASCII table."""
    lines = []

    # Calculate column widths
    scenario_width = max(len(s.replace("scenario = ", "")) for s in scenarios + ["Scenario"])
    model_widths = {
        m: max(len(m), max(len(table_data[s].get(m, "")) for s in scenarios), len(verdicts[m]))
        for m in models
    }

    # Header
    lines.append(f"{'=' * 60}")
    lines.append(f"  {property_name} - Model Comparison")
    lines.append(f"{'=' * 60}")
    lines.append("")

    header = f"{'Scenario':<{scenario_width}}"
    for m in models:
        header += f"  {m:>{model_widths[m]}}"
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for scenario in scenarios:
        display_scenario = scenario.replace("scenario = ", "")
        row = f"{display_scenario:<{scenario_width}}"
        for m in models:
            cell = table_data[scenario].get(m, "N/A")
            row += f"  {cell:>{model_widths[m]}}"
        lines.append(row)

    # Verdict row
    lines.append("-" * len(header))
    verdict_row = f"{'VERDICT':<{scenario_width}}"
    for m in models:
        verdict_row += f"  {verdicts[m]:>{model_widths[m]}}"
    lines.append(verdict_row)
    lines.append("")

    return "\n".join(lines)


def format_markdown(
    property_name: str,
    models: List[str],
    scenarios: List[str],
    table_data: Dict[str, Dict[str, str]],
    verdicts: Dict[str, str],
) -> str:
    """Format comparison as Markdown table."""
    lines = []

    lines.append(f"## {property_name} - Model Comparison")
    lines.append("")

    # Header
    header = "| Scenario |"
    separator = "|----------|"
    for m in models:
        header += f" {m} |"
        separator += "--------|"
    lines.append(header)
    lines.append(separator)

    # Data rows
    for scenario in scenarios:
        display_scenario = scenario.replace("scenario = ", "")
        row = f"| {display_scenario} |"
        for m in models:
            cell = table_data[scenario].get(m, "N/A")
            row += f" {cell} |"
        lines.append(row)

    # Verdict row
    verdict_row = "| **VERDICT** |"
    for m in models:
        verdict_row += f" **{verdicts[m]}** |"
    lines.append(verdict_row)
    lines.append("")

    return "\n".join(lines)


def format_csv(
    property_name: str,
    models: List[str],
    scenarios: List[str],
    table_data: Dict[str, Dict[str, str]],
    verdicts: Dict[str, str],
) -> str:
    """Format comparison as CSV."""
    lines = []

    # Header
    lines.append("scenario," + ",".join(models))

    # Data rows
    for scenario in scenarios:
        display_scenario = scenario.replace("scenario = ", "").replace(",", ";")
        row = f'"{display_scenario}"'
        for m in models:
            cell = table_data[scenario].get(m, "N/A").replace(" ✓", "").replace(" ✗", "")
            row += f",{cell}"
        lines.append(row)

    # Verdict row
    lines.append('"VERDICT",' + ",".join(verdicts[m] for m in models))

    return "\n".join(lines)


def format_comparison(
    comparison: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    fmt: str = "table",
) -> str:
    """Format comparison results in the specified format.

    Args:
        comparison: Dict mapping model name to certificate
        cfg: Original config
        fmt: Output format - "table", "markdown", or "csv"

    Returns:
        Formatted string
    """
    property_name = cfg["property"]["name"]
    models, scenarios, table_data, verdicts = build_comparison_table(comparison)

    if fmt == "csv":
        return format_csv(property_name, models, scenarios, table_data, verdicts)
    elif fmt == "markdown":
        return format_markdown(property_name, models, scenarios, table_data, verdicts)
    else:
        return format_table(property_name, models, scenarios, table_data, verdicts)


def save_comparison(
    comparison: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    output_dir: str,
) -> str:
    """Save comparison certificate to output directory.

    Args:
        comparison: Dict mapping model name to certificate
        cfg: Original config
        output_dir: Base output directory

    Returns:
        Path to saved comparison file
    """
    property_name = cfg["property"]["name"]
    models = list(comparison.keys())

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    compare_dir = os.path.join(output_dir, f"{timestamp}_{property_name.lower()}_comparison")
    os.makedirs(compare_dir, exist_ok=True)

    comparison_cert = {
        "property": cfg["property"],
        "models": models,
        "results": comparison,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    cert_path = os.path.join(compare_dir, "comparison.yaml")
    with open(cert_path, "w") as f:
        yaml.dump(comparison_cert, f, default_flow_style=False, sort_keys=False)

    return cert_path
