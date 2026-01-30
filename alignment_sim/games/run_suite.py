from __future__ import annotations

import logging
import os
from typing import Any, Dict

import yaml

from ..core.certificate import dump_certificate
from ..core.config import load_config
from ..core.output import generate_plots, write_raw_responses, write_summary
from ..core.runner import AlignmentSimRunner

logger = logging.getLogger(__name__)


def run_suite(
    suite_path: str,
    output_dir: str | None = None,
    concurrency: int = 10,
    no_plots: bool = False,
) -> str:
    suite_manifest, suite_dir = _load_suite(suite_path)
    suite = suite_manifest["suite"]

    results_dir = _resolve_path(suite_dir, output_dir or suite["paths"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    run_defaults = suite.get("run_defaults", {})

    for cfg_entry in suite.get("configs", []):
        cfg_path = _resolve_path(suite_dir, cfg_entry["path"])
        config = load_config(cfg_path)

        # Apply run defaults unless explicitly overridden
        for key, value in run_defaults.items():
            if key not in config.get("run", {}):
                config.setdefault("run", {})[key] = value

        # Ensure raw results are included for summarization
        config.setdefault("run", {})["include_raw_results"] = True

        logger.info("Running suite config: %s", cfg_entry.get("id", cfg_path))
        runner = AlignmentSimRunner(config, concurrency=concurrency)
        cert = runner.run()

        # Extract internal data for outputs
        results = cert.pop("_results", [])
        cfg_used = cert.pop("_config", config)

        cfg_out_dir = os.path.join(results_dir, cfg_entry.get("id", "config"))
        os.makedirs(cfg_out_dir, exist_ok=True)

        dump_certificate(cert, os.path.join(cfg_out_dir, "certificate.yaml"))
        write_summary(cfg_out_dir, cfg_used, results, cert["fit"], cert["predictions"])
        write_raw_responses(cfg_out_dir, results, cfg_used)
        if not no_plots:
            generate_plots(cfg_out_dir, cfg_used, results, cert["fit"])

    logger.info("Suite run complete. Results in: %s", results_dir)
    return results_dir


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
