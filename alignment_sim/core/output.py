from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def write_raw_responses(output_dir: str, results: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    """Write raw LLM requests and responses to a JSON file for debugging/reference."""
    raw_data = []
    grid_vars = config["run"]["grid_over"]

    for r in results:
        cell_key = {k: r[k] for k in grid_vars if k in r}
        prompts = r.get("_prompts", {})
        raw_responses = r.get("_raw_responses", [])

        entry = {
            "cell": cell_key,
            "request": {
                "system": prompts.get("system", ""),
                "user": prompts.get("user", ""),
            },
            "responses": raw_responses,
        }

        # Include history if present
        if prompts.get("history"):
            entry["request"]["history"] = prompts["history"]

        raw_data.append(entry)

    raw_path = os.path.join(output_dir, "raw_responses.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)

    logger.info("Wrote raw responses to: %s", raw_path)
    return raw_path
plt = None


def _ensure_matplotlib(output_dir: str | None = None) -> None:
    """Lazily import matplotlib and set a writable cache dir if provided."""
    global plt
    if plt is not None:
        return
    if output_dir:
        mpl_dir = os.path.join(output_dir, ".matplotlib")
        os.makedirs(mpl_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
    import matplotlib.pyplot as _plt

    plt = _plt


def create_output_dir(config: Dict[str, Any], base_dir: str = "outputs") -> str:
    """Create a timestamped output directory for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    property_name = config["property"]["name"].lower().replace(" ", "_")
    model_name = config["run"]["model"].replace("/", "_").replace(".", "-")
    dir_name = f"{timestamp}_{property_name}_{model_name}"
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    logger.info("Created output directory: %s", output_dir)
    return output_dir


def write_summary(
    output_dir: str,
    config: Dict[str, Any],
    results: List[Dict[str, Any]],
    fit: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> str:
    """Write a human-readable summary report."""
    pred_type = config["prediction"]["type"]
    target_name = config["prediction"].get("target") or config["variables"]["measured"][0]["name"]

    lines = []
    lines.append("=" * 60)
    lines.append(f"AlignmentSim Run Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Property: {config['property']['name']}")
    lines.append(f"Formula:  {config['property']['formula']}")
    if config["property"].get("source"):
        lines.append(f"Source:   {config['property']['source']}")
    lines.append("")
    lines.append(f"Model tested: {config['run']['model']}")
    lines.append(f"Timestamp:    {datetime.now().isoformat()}")
    lines.append("")

    # Grid info
    n_cells = len(results)
    n_per_cell = config["run"]["n_per_cell"]
    lines.append(f"Grid: {n_cells} cells x {n_per_cell} samples = {n_cells * n_per_cell} total")
    lines.append(f"Temperature: {config['run']['temperature']}")
    lines.append("")

    # Fit results
    lines.append("-" * 60)
    lines.append("FIT RESULTS")
    lines.append("-" * 60)
    if pred_type == "invariant":
        lines.append("No fit (invariant prediction type)")
    elif pred_type == "logistic":
        lines.append(f"Pseudo R² = {fit.get('pseudo_r2', 0.0):.4f}")
        lines.append(f"Intercept (β₀) = {fit.get('beta_0', 0.0):.4f}")
        for key, value in fit.items():
            if key.startswith("beta_") and key != "beta_0":
                lines.append(f"{key} = {value:.4f}")
        if "odds_ratios" in fit:
            lines.append("Odds ratios:")
            for name, value in fit["odds_ratios"].items():
                lines.append(f"  {name}: {value:.4f}")
        if "converged" in fit:
            lines.append(f"Converged: {fit['converged']}")
            if not fit["converged"]:
                lines.append("Warning: logistic fit did not converge (try increasing max_iter or adjusting lr/l2).")
    else:
        lines.append(f"R² = {fit.get('r_squared', 0.0):.4f}")
        lines.append(f"Intercept (β₀) = {fit.get('beta_0', 0.0):.4f}")
        for key, value in fit.items():
            if key.startswith("beta_") and key != "beta_0":
                lines.append(f"{key} = {value:.4f}")

    # Inferred variables
    for inf in config["variables"].get("inferred", []):
        name = inf["name"]
        if name in fit and isinstance(fit[name], (int, float)):
            lines.append(f"Inferred {name} = {fit[name]:.4f}")
    lines.append("")

    # Predictions
    lines.append("-" * 60)
    lines.append("PREDICTION CHECKS")
    lines.append("-" * 60)
    n_passed = sum(1 for p in predictions if p.get("passed"))
    n_total = len(predictions)
    lines.append(f"Passed: {n_passed}/{n_total}")
    lines.append("")
    for pred in predictions:
        status = "✓ PASS" if pred.get("passed") else "✗ FAIL"
        obs = pred.get("observed")
        if isinstance(obs, (int, float)):
            obs_str = f"{obs:.4f}"
        else:
            obs_str = str(obs) if obs is not None else "N/A"
        lines.append(f"  [{status}] {pred['condition']}")
        lines.append(f"          Expected: {pred['expected']}")
        lines.append(f"          Observed: {obs_str}")
        if pred.get("violations") is not None:
            lines.append(f"          Violations: {pred.get('violations')} (rate={pred.get('violation_rate', 0.0):.4f})")
        if pred.get("note"):
            lines.append(f"          Note: {pred['note']}")
        lines.append("")

    # Verdict
    passed_all = all(p.get("passed") for p in predictions) if predictions else False
    passed_any = any(p.get("passed") for p in predictions) if predictions else False
    if passed_all:
        verdict = "PROVEN" if pred_type == "invariant" else "SUPPORTED"
    elif passed_any:
        verdict = "PARTIAL"
    else:
        verdict = "FAILED"

    lines.append("-" * 60)
    lines.append(f"VERDICT: {verdict}")
    lines.append("-" * 60)
    lines.append("")

    # Cell-level statistics
    lines.append("-" * 60)
    lines.append("CELL-LEVEL STATISTICS")
    lines.append("-" * 60)

    # Get controlled variable names from grid_over
    grid_vars = config["run"]["grid_over"]
    header = " | ".join([f"{v:>12}" for v in grid_vars] + [f"{'mean':>8}", f"{'std':>8}", f"{'min':>8}", f"{'max':>8}"])
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        scores = _get_cell_scores(r, target_name)
        if scores:
            mean_s = np.mean(scores)
            std_s = np.std(scores)
            min_s = np.min(scores)
            max_s = np.max(scores)
        else:
            mean_s = std_s = min_s = max_s = 0.0

        row_vals = [f"{r[v]:>12}" if isinstance(r[v], str) else f"{r[v]:>12.2f}" for v in grid_vars]
        row_vals += [f"{mean_s:>8.4f}", f"{std_s:>8.4f}", f"{min_s:>8.4f}", f"{max_s:>8.4f}"]
        lines.append(" | ".join(row_vals))

    lines.append("")

    # Categorical distributions (all actions)
    categorical_measures = []
    for measured in config.get("variables", {}).get("measured", []):
        params = measured.get("params", {})
        if params.get("output_type") == "categorical":
            categorical_measures.append(measured["name"])

    if categorical_measures:
        lines.append("-" * 60)
        lines.append("CATEGORICAL DISTRIBUTIONS")
        lines.append("-" * 60)
        for name in categorical_measures:
            categories = sorted(
                {
                    v
                    for r in results
                    for v in r.get("_measurements", {}).get(name, [])
                    if v is not None
                }
            )
            if not categories:
                continue
            lines.append(f"Variable: {name}")
            header = " | ".join([f"{v:>12}" for v in grid_vars] + [f"{c:>16}" for c in categories])
            lines.append(header)
            lines.append("-" * len(header))
            for r in results:
                values = r.get("_measurements", {}).get(name, [])
                counts = Counter(values)
                total = len(values) if values else 0
                row_vals = [f"{r[v]:>12}" if isinstance(r[v], str) else f"{r[v]:>12.2f}" for v in grid_vars]
                for c in categories:
                    count = counts.get(c, 0)
                    if total:
                        cell = f"{count:>4} ({count / total:>4.2f})"
                    else:
                        cell = f"{count:>4} ( n/a)"
                    row_vals.append(f"{cell:>16}")
                lines.append(" | ".join(row_vals))
            lines.append("")

    summary_text = "\n".join(lines)
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info("Wrote summary to: %s", summary_path)
    return summary_path


def generate_plots(
    output_dir: str,
    config: Dict[str, Any],
    results: List[Dict[str, Any]],
    fit: Dict[str, Any],
) -> List[str]:
    """Generate visualization plots for the results."""
    pred_type = config["prediction"]["type"]
    if pred_type == "invariant":
        return []
    _ensure_matplotlib(output_dir)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    measured_name = config["prediction"].get("target") or config["variables"]["measured"][0]["name"]
    grid_vars = config["run"]["grid_over"]
    generated = []

    # Try to identify the main controlled variables for plotting
    # Usually we have something like delta_B (continuous) and delta_V (categorical)
    continuous_vars = []
    categorical_vars = []
    for var in config["variables"]["controlled"]:
        if var["name"] in grid_vars:
            if var["type"] == "continuous":
                continuous_vars.append(var["name"])
            elif var["type"] == "categorical":
                categorical_vars.append(var["name"])

    # Plot 1: Main effect plot (if we have a continuous and categorical var)
    if continuous_vars and categorical_vars:
        plot_path = _plot_main_effects(
            plots_dir, results, continuous_vars[0], categorical_vars[0], measured_name, config
        )
        if plot_path:
            generated.append(plot_path)

    # Plot 2: Heatmap (if we have exactly 2 main grid variables)
    if len(grid_vars) >= 2:
        plot_path = _plot_heatmap(
            plots_dir, results, grid_vars[0], grid_vars[1], measured_name, config
        )
        if plot_path:
            generated.append(plot_path)

    # Plot 3: Distribution of scores
    plot_path = _plot_score_distributions(plots_dir, results, measured_name, config)
    if plot_path:
        generated.append(plot_path)

    # Plot 4: Observed vs Predicted
    plot_path = _plot_observed_vs_predicted(plots_dir, results, fit, measured_name, config)
    if plot_path:
        generated.append(plot_path)

    logger.info("Generated %d plots in: %s", len(generated), plots_dir)
    return generated


def _plot_main_effects(
    plots_dir: str,
    results: List[Dict[str, Any]],
    x_var: str,
    group_var: str,
    y_var: str,
    config: Dict[str, Any],
) -> str | None:
    """Plot measured variable vs continuous variable, grouped by categorical."""
    try:
        # Get unique values for grouping
        group_values = sorted(set(r[group_var] for r in results))
        x_values = sorted(set(r[x_var] for r in results))

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(group_values)))

        for i, gv in enumerate(group_values):
            subset = [r for r in results if r[group_var] == gv]
            x_data = []
            y_data = []
            y_err = []

            for xv in x_values:
                cell_results = [r for r in subset if r[x_var] == xv]
                if cell_results:
                    # Aggregate across other dimensions (like sponsored_brand)
                    all_scores = []
                    for cr in cell_results:
                        all_scores.extend(_get_cell_scores(cr, y_var))
                    if all_scores:
                        x_data.append(xv)
                        y_data.append(np.mean(all_scores))
                        y_err.append(np.std(all_scores) / np.sqrt(len(all_scores)))

            ax.errorbar(x_data, y_data, yerr=y_err, marker='o', label=f"{group_var}={gv}",
                       color=colors[i], capsize=3, linewidth=2, markersize=8)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(x_var, fontsize=12)
        ax.set_ylabel(y_var, fontsize=12)
        ax.set_title(f"{y_var} vs {x_var} by {group_var}", fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(plots_dir, "main_effects.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.debug("Created main effects plot: %s", plot_path)
        return plot_path
    except Exception as e:
        logger.warning("Failed to create main effects plot: %s", e)
        return None


def _plot_heatmap(
    plots_dir: str,
    results: List[Dict[str, Any]],
    x_var: str,
    y_var: str,
    z_var: str,
    config: Dict[str, Any],
) -> str | None:
    """Plot heatmap of measured variable across two controlled variables."""
    try:
        x_values = sorted(set(r[x_var] for r in results), key=lambda v: (isinstance(v, str), v))
        y_values = sorted(set(r[y_var] for r in results), key=lambda v: (isinstance(v, str), v))

        # Build matrix, averaging over other dimensions
        matrix = np.zeros((len(y_values), len(x_values)))
        for i, yv in enumerate(y_values):
            for j, xv in enumerate(x_values):
                matching = [r for r in results if r[x_var] == xv and r[y_var] == yv]
                if matching:
                    all_scores = []
                    for m in matching:
                        all_scores.extend(_get_cell_scores(m, z_var))
                    matrix[i, j] = np.mean(all_scores) if all_scores else 0

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')

        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([str(v) for v in x_values])
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels([str(v) for v in y_values])

        ax.set_xlabel(x_var, fontsize=12)
        ax.set_ylabel(y_var, fontsize=12)
        ax.set_title(f"{z_var} Heatmap", fontsize=14)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(z_var, fontsize=12)

        # Add value annotations
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                text = ax.text(j, i, f"{matrix[i, j]:.3f}",
                              ha="center", va="center", color="black", fontsize=9)

        plot_path = os.path.join(plots_dir, "heatmap.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.debug("Created heatmap: %s", plot_path)
        return plot_path
    except Exception as e:
        logger.warning("Failed to create heatmap: %s", e)
        return None


def _plot_score_distributions(
    plots_dir: str,
    results: List[Dict[str, Any]],
    measured_name: str,
    config: Dict[str, Any],
) -> str | None:
    """Plot distribution of all scores."""
    try:
        all_scores = []
        for r in results:
            all_scores.extend(_get_cell_scores(r, measured_name))

        if not all_scores:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        axes[0].hist(all_scores, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='red', linestyle='--', label='Zero')
        axes[0].axvline(x=np.mean(all_scores), color='green', linestyle='-', label=f'Mean={np.mean(all_scores):.3f}')
        axes[0].set_xlabel(measured_name, fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title("Score Distribution", fontsize=14)
        axes[0].legend()

        # Box plot by cell
        cell_scores = [_get_cell_scores(r, measured_name) for r in results]
        cell_labels = [f"C{i+1}" for i in range(len(results))]

        # Only show first 20 cells if too many
        if len(cell_scores) > 20:
            cell_scores = cell_scores[:20]
            cell_labels = cell_labels[:20]

        axes[1].boxplot(cell_scores, labels=cell_labels)
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_xlabel("Cell", fontsize=12)
        axes[1].set_ylabel(measured_name, fontsize=12)
        axes[1].set_title("Score Distribution by Cell", fontsize=14)
        axes[1].tick_params(axis='x', rotation=45)

        plot_path = os.path.join(plots_dir, "distributions.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.debug("Created distributions plot: %s", plot_path)
        return plot_path
    except Exception as e:
        logger.warning("Failed to create distributions plot: %s", e)
        return None


def _plot_observed_vs_predicted(
    plots_dir: str,
    results: List[Dict[str, Any]],
    fit: Dict[str, Any],
    measured_name: str,
    config: Dict[str, Any],
) -> str | None:
    """Plot observed vs predicted values from the fit."""
    try:
        from .fit import build_feature_matrix

        X, feature_names = build_feature_matrix(results, config)
        observed_raw = [r.get(measured_name) for r in results]
        observed_values: List[float] = []
        row_indices: List[int] = []
        for idx, value in enumerate(observed_raw):
            if value is None:
                continue
            if isinstance(value, bool):
                observed_values.append(1.0 if value else 0.0)
                row_indices.append(idx)
            elif isinstance(value, (int, float)):
                observed_values.append(float(value))
                row_indices.append(idx)

        if not observed_values:
            return None

        X = X[row_indices, :]

        # Compute predicted values
        if fit.get("type") == "logistic":
            z = fit.get("beta_0", 0.0) + sum(
                fit.get(f"beta_{name}", 0.0) * X[:, i]
                for i, name in enumerate(feature_names)
            )
            y_predicted = 1 / (1 + np.exp(-z))
            r2_label = f"pseudo R² = {fit.get('pseudo_r2', 0.0):.3f}"
        else:
            y_predicted = fit.get("beta_0", 0.0) + sum(
                fit.get(f"beta_{name}", 0.0) * X[:, i]
                for i, name in enumerate(feature_names)
            )
            r2_label = f"R² = {fit.get('r_squared', 0.0):.3f}"

        fig, ax = plt.subplots(figsize=(8, 8))

        y_observed = np.array(observed_values, dtype=float)
        ax.scatter(y_observed, y_predicted, alpha=0.6, edgecolor='black', s=60)

        # Perfect prediction line
        lims = [
            min(min(y_observed), min(y_predicted)),
            max(max(y_observed), max(y_predicted))
        ]
        ax.plot(lims, lims, 'r--', label='Perfect fit')

        ax.set_xlabel(f"Observed {measured_name}", fontsize=12)
        ax.set_ylabel(f"Predicted {measured_name}", fontsize=12)
        ax.set_title(f"Observed vs Predicted ({r2_label})", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(plots_dir, "observed_vs_predicted.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.debug("Created observed vs predicted plot: %s", plot_path)
        return plot_path
    except Exception as e:
        logger.warning("Failed to create observed vs predicted plot: %s", e)
        return None


def _get_cell_scores(cell: Dict[str, Any], metric: str) -> List[float]:
    measurements = cell.get("_measurements", {})
    values = measurements.get(metric)
    if values is None:
        values = cell.get("_scores", [])
    numeric_values: List[float] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            numeric_values.append(1.0 if v else 0.0)
        elif isinstance(v, (int, float)):
            numeric_values.append(float(v))
    return numeric_values
