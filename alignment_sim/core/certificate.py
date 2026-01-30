from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


def build_certificate(
    cfg: Dict[str, Any],
    results: List[Dict[str, Any]],
    fit: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    logger.debug("Building certificate")
    n_total = sum(r.get("_n_samples", len(r.get("_scores", []))) for r in results)
    verdict = _verdict(predictions, cfg["prediction"]["type"])
    logger.debug("Certificate: %d total samples, verdict=%s", n_total, verdict)
    cert = {
        "schema_version": cfg["schema_version"],
        "property": cfg["property"],
        "model_tested": cfg["run"]["model"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_total_samples": n_total,
        "fit": fit,
        "predictions": predictions,
        "verdict": verdict,
    }
    if cfg["run"].get("include_raw_results"):
        logger.debug("Including raw results in certificate")
        cert["raw_results"] = results
    return cert


def dump_certificate(cert: Dict[str, Any], path: str | None = None) -> None:
    data = yaml.safe_dump(cert, sort_keys=False)
    if path:
        logger.info("Writing certificate to: %s", path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
    else:
        logger.debug("Printing certificate to stdout")
        print(data)


def _verdict(predictions: List[Dict[str, Any]], pred_type: str) -> str:
    if not predictions:
        return "UNKNOWN"
    passed = [p.get("passed") is True for p in predictions]
    if all(passed):
        return "PROVEN" if pred_type == "invariant" else "SUPPORTED"
    if any(passed):
        return "PARTIAL"
    return "FAILED"
