from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Dict, List

import numpy as np
from openai import AsyncOpenAI

from ..analysis.fit import build_feature_matrix, build_fit_report, fit_linear, fit_logistic, infer_variables
from ..analysis.measure import (
    AsyncEmbedder,
    aggregate_measurements,
    get_output_type,
    measure_derived,
    measure_response_async,
)
from ..analysis.predict import check_predictions
from .certificate import build_certificate
from .grid import build_grid
from .prompt import render_prompts

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 10


async def sample_model_async(
    client: AsyncOpenAI,
    system: str,
    user: str,
    n: int,
    temperature: float,
    model: str,
    extra_args: Dict[str, Any] | None = None,
    history: List[Dict[str, str]] | None = None,
) -> List[Dict[str, Any]]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    # Insert conversation history between system and final user message
    if history:
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
    if user:
        messages.append({"role": "user", "content": user})

    logger.debug("Requesting %d samples from %s (temperature=%.2f)", n, model, temperature)
    kwargs: Dict[str, Any] = {"model": model, "messages": messages, "n": n, "temperature": temperature}
    if extra_args:
        kwargs.update(extra_args)
    resp = await client.chat.completions.create(**kwargs)
    responses = []
    for choice in resp.choices:
        msg = choice.message
        content = msg.content or ""
        tool_calls = _normalize_tool_calls(getattr(msg, "tool_calls", None))
        responses.append({"content": content, "tool_calls": tool_calls})
    logger.debug("Received %d responses", len(responses))
    return responses


def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    if not tool_calls:
        return []
    normalized = []
    for call in tool_calls:
        name = None
        arguments = None
        if hasattr(call, "function"):
            name = getattr(call.function, "name", None)
            arguments = getattr(call.function, "arguments", None)
        elif isinstance(call, dict):
            fn = call.get("function", {})
            name = fn.get("name")
            arguments = fn.get("arguments")
        normalized.append({"name": name, "arguments": arguments})
    return normalized


class AlignmentSimRunner:
    def __init__(self, config: Dict[str, Any], concurrency: int = DEFAULT_CONCURRENCY) -> None:
        self.config = config
        self.client = AsyncOpenAI()
        self.concurrency = concurrency
        logger.info("Initialized AlignmentSimRunner for property: %s (concurrency=%d)", config["property"]["name"], concurrency)

    def run(self) -> Dict[str, Any]:
        """Synchronous entry point that runs the async implementation."""
        return asyncio.run(self._run_async())

    async def _process_cell(
        self,
        cell: Dict[str, Any],
        cell_idx: int,
        total_cells: int,
        measured_defs: List[Dict[str, Any]],
        embedder: AsyncEmbedder | None,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        """Process a single cell with rate limiting via semaphore."""
        async with semaphore:
            logger.info("Processing cell %d/%d: %s", cell_idx, total_cells, cell)

            prompts = render_prompts(self.config, cell)
            responses = await sample_model_async(
                self.client,
                prompts["system"],
                prompts["user"],
                self.config["run"]["n_per_cell"],
                self.config["run"]["temperature"],
                self.config["run"]["model"],
                _build_extra_args(self.config),
                prompts.get("history"),
            )

            measurements_per_response: List[Dict[str, Any]] = []
            for r in responses:
                response_text = r.get("content", "")
                tool_calls = r.get("tool_calls", [])
                per_response: Dict[str, Any] = {}
                for measured_def in measured_defs:
                    if measured_def["method"] == "derived":
                        continue
                    value = await measure_response_async(embedder, response_text, cell, measured_def, tool_calls)
                    per_response[measured_def["name"]] = value
                for measured_def in measured_defs:
                    if measured_def["method"] != "derived":
                        continue
                    value = measure_derived(cell, per_response, measured_def.get("params", {}))
                    per_response[measured_def["name"]] = value
                measurements_per_response.append(per_response)

            aggregated: Dict[str, Any] = {}
            measurements_map: Dict[str, List[Any]] = {}
            for measured_def in measured_defs:
                name = measured_def["name"]
                output_type = get_output_type(measured_def)
                values = [m.get(name) for m in measurements_per_response]
                measurements_map[name] = values
                aggregated[name] = aggregate_measurements(values, output_type)

            primary_name = self.config["prediction"].get("target") or measured_defs[0]["name"]
            primary_values = measurements_map.get(primary_name, [])
            primary_numeric = [float(v) for v in primary_values if v is not None and isinstance(v, (int, float, bool))]
            mean_score = float(np.mean(primary_numeric)) if primary_numeric else 0.0
            min_score = float(min(primary_numeric)) if primary_numeric else 0.0
            max_score = float(max(primary_numeric)) if primary_numeric else 0.0
            logger.info(
                "Cell %d primary metric %s: mean=%.4f, min=%.4f, max=%.4f",
                cell_idx,
                primary_name,
                mean_score,
                min_score,
                max_score,
            )

            result = {
                **cell,
                **aggregated,
                "_measurements": measurements_map,
                "_n_samples": len(responses),
                "_cell_idx": cell_idx,
                "_prompts": prompts,
                "_raw_responses": [r["content"] for r in responses],
            }
            if primary_numeric:
                result["_scores"] = primary_numeric
            return result

    async def _run_async(self) -> Dict[str, Any]:
        try:
            logger.info("Starting experiment run (async)")
            logger.info("Target model: %s", self.config["run"]["model"])

            grid = build_grid(self.config)
            logger.info("Built grid with %d cells", len(grid))

            measured_defs = self.config["variables"]["measured"]
            measured_names = [m["name"] for m in measured_defs]
            logger.info("Measured variables: %s", ", ".join(measured_names))

            needs_embedding = any(m["method"] == "embedding_direction" for m in measured_defs)
            embedder = AsyncEmbedder(self.client) if needs_embedding else None
            semaphore = asyncio.Semaphore(self.concurrency)

            n_per_cell = self.config["run"]["n_per_cell"]
            total_samples = len(grid) * n_per_cell
            logger.info("Will collect %d samples (%d cells x %d per cell) with concurrency=%d",
                       total_samples, len(grid), n_per_cell, self.concurrency)

            # Launch all cells concurrently (bounded by semaphore)
            tasks = [
                self._process_cell(cell, i, len(grid), measured_defs, embedder, semaphore)
                for i, cell in enumerate(grid, start=1)
            ]
            results = await asyncio.gather(*tasks)

            # Sort results by cell index to maintain deterministic order
            results = sorted(results, key=lambda r: r.pop("_cell_idx"))

            prediction_type = self.config["prediction"]["type"]
            target_name = self.config["prediction"].get("target") or measured_defs[0]["name"]

            fit_report: Dict[str, Any] = {"type": prediction_type}
            if prediction_type != "invariant":
                logger.info("All cells processed, fitting model")
                fit_cfg = self.config["prediction"].get("fit", {})
                fit_level = fit_cfg.get("level")
                if fit_level is None:
                    fit_level = "sample" if prediction_type == "logistic" else "cell"
                X, y, feature_names = _build_fit_data(results, self.config, target_name, fit_level)
                logger.debug("Feature matrix shape: %s, features: %s", X.shape, feature_names)

                if prediction_type == "logistic":
                    fit_raw = fit_logistic(X, y, fit_cfg)
                else:
                    fit_raw = fit_linear(X, y)

                fit_report = build_fit_report(fit_raw, feature_names)
                fit_report["type"] = prediction_type
                fit_report["fit_level"] = fit_level

                if prediction_type == "logistic":
                    logger.info("Fit complete: pseudo R²=%.4f", fit_report.get("pseudo_r2", 0.0))
                else:
                    logger.info("Fit complete: R²=%.4f", fit_report.get("r_squared", 0.0))
                for name in feature_names:
                    logger.info("  beta_%s = %.4f", name, fit_report.get(f"beta_{name}", float("nan")))
                if prediction_type == "logistic" and fit_report.get("converged") is False:
                    logger.warning("Logistic fit did not converge; consider increasing max_iter or adjusting lr/l2.")

                fit_report.update(infer_variables(fit_report, self.config))
                for inf in self.config["variables"].get("inferred", []):
                    logger.info("Inferred %s = %.4f", inf["name"], fit_report.get(inf["name"], float("nan")))

            logger.info("Checking predictions")
            predictions = check_predictions(results, self.config)
            for pred in predictions:
                status = "PASS" if pred["passed"] else "FAIL"
                logger.info("  [%s] %s -> %s (observed: %s)", status, pred["condition"], pred["expected"], pred.get("observed"))

            certificate = build_certificate(self.config, results, fit_report, predictions)
            logger.info("Experiment complete. Verdict: %s", certificate["verdict"])

            # Attach raw data for output generation (not serialized to YAML)
            certificate["_results"] = results
            certificate["_config"] = self.config

            return certificate
        finally:
            await _close_async_client(self.client)


def _build_extra_args(config: Dict[str, Any]) -> Dict[str, Any]:
    run_cfg = config.get("run", {})
    extra: Dict[str, Any] = {}
    if "response_format" in run_cfg:
        extra["response_format"] = run_cfg["response_format"]
    if "tools" in run_cfg:
        extra["tools"] = run_cfg["tools"]
    if "tool_choice" in run_cfg:
        extra["tool_choice"] = run_cfg["tool_choice"]
    if "max_tokens" in run_cfg:
        extra["max_tokens"] = run_cfg["max_tokens"]
    return extra


async def _close_async_client(client: Any) -> None:
    close = getattr(client, "close", None)
    aclose = getattr(client, "aclose", None)
    for fn in (close, aclose):
        if not callable(fn):
            continue
        try:
            result = fn()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.debug("Async client close failed: %s", exc)
        break


def _build_fit_data(
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    target_name: str,
    fit_level: str,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    X_cell, feature_names = build_feature_matrix(results, config)

    if fit_level == "sample":
        rows: List[np.ndarray] = []
        y_values: List[float] = []
        for idx, r in enumerate(results):
            measurements = r.get("_measurements", {})
            values = measurements.get(target_name)
            if not values:
                continue
            for v in values:
                if v is None:
                    continue
                if isinstance(v, bool):
                    y_values.append(1.0 if v else 0.0)
                    rows.append(X_cell[idx])
                elif isinstance(v, (int, float)):
                    y_values.append(float(v))
                    rows.append(X_cell[idx])
        if not y_values:
            raise ValueError(f"No numeric sample values found for target {target_name!r}.")
        X = np.vstack(rows)
        y = np.array(y_values, dtype=float)
        return X, y, feature_names

    # Default: cell-level fit on aggregated results
    y_values: List[float] = []
    row_indices: List[int] = []
    for idx, r in enumerate(results):
        value = r.get(target_name)
        if value is None:
            continue
        if isinstance(value, bool):
            y_values.append(1.0 if value else 0.0)
            row_indices.append(idx)
        elif isinstance(value, (int, float)):
            y_values.append(float(value))
            row_indices.append(idx)
    if not y_values:
        raise ValueError(f"No numeric values found for target {target_name!r}.")
    X = X_cell[row_indices, :]
    y = np.array(y_values, dtype=float)
    return X, y, feature_names
