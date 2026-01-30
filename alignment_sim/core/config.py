from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import yaml

logger = logging.getLogger(__name__)

ALLOWED_INJECT = {"system_prompt", "user_message", "tool_context"}
ALLOWED_CONTROL_TYPES = {"continuous", "categorical", "binary"}
ALLOWED_MEASURE_METHODS = {"embedding_direction", "regex", "tool_call", "json_field", "derived"}
ALLOWED_PREDICTION_TYPES = {"threshold", "monotonic", "phase_transition", "linear", "logistic", "invariant"}


class ConfigError(ValueError):
    pass


def load_config(path: str) -> Dict[str, Any]:
    logger.info("Loading config from: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError("Config must be a YAML mapping/object at top level.")
    validate_config(data)
    logger.info("Config loaded successfully: %s (schema v%s)", data["property"]["name"], data["schema_version"])
    return data


def validate_config(cfg: Dict[str, Any]) -> None:
    logger.debug("Validating config structure")
    _require_keys(cfg, ["schema_version", "property", "variables", "prediction", "run"])

    schema_version = cfg["schema_version"]
    if schema_version not in {"0.1", "0.2"}:
        raise ConfigError(f"Unsupported schema_version: {schema_version!r} (expected '0.1' or '0.2').")
    logger.debug("Schema version: %s", schema_version)

    prop = cfg["property"]
    if not isinstance(prop, dict):
        raise ConfigError("property must be a mapping.")
    prop_keys = ["name", "formula"]
    if schema_version == "0.1":
        prop_keys.append("source")
    _require_keys(prop, prop_keys)

    variables = cfg["variables"]
    if not isinstance(variables, dict):
        raise ConfigError("variables must be a mapping.")
    if "controlled" not in variables or not isinstance(variables["controlled"], list):
        raise ConfigError("variables.controlled must be a list.")

    controlled = variables["controlled"]
    name_to_var = {}
    for var in controlled:
        if not isinstance(var, dict):
            raise ConfigError("Each controlled variable must be a mapping.")
        _require_keys(var, ["name", "type", "values", "inject"])
        name = var["name"]
        vtype = var["type"]
        inject = var["inject"]

        if vtype not in ALLOWED_CONTROL_TYPES:
            raise ConfigError(f"Unsupported controlled variable type: {vtype!r}")
        if inject not in ALLOWED_INJECT:
            raise ConfigError(f"Unsupported inject type: {inject!r}")
        _validate_values(var)
        name_to_var[name] = var

    measured = variables.get("measured", [])
    if not isinstance(measured, list) or len(measured) == 0:
        raise ConfigError("variables.measured must be a non-empty list.")
    for m in measured:
        if not isinstance(m, dict):
            raise ConfigError("Each measured variable must be a mapping.")
        _require_keys(m, ["name", "method", "params"])
        method = m["method"]
        if method not in ALLOWED_MEASURE_METHODS:
            raise ConfigError(f"Unsupported measurement method: {method!r}")
        params = m.get("params", {})
        if not isinstance(params, dict):
            raise ConfigError("Measured params must be a mapping.")
        if method == "embedding_direction":
            _require_keys(params, ["anchor_positive", "anchor_negative"])
        elif method == "json_field":
            _require_keys(params, ["path"])
        elif method == "regex":
            _require_keys(params, ["pattern"])
        elif method == "tool_call":
            _require_keys(params, ["tool_name"])
        elif method == "derived":
            _require_keys(params, ["formula"])

    inferred = variables.get("inferred", [])
    if inferred is not None and not isinstance(inferred, list):
        raise ConfigError("variables.inferred must be a list if present.")
    if inferred:
        for inf in inferred:
            if not isinstance(inf, dict):
                raise ConfigError("Each inferred variable must be a mapping.")
            _require_keys(inf, ["name", "formula"])

    prediction = cfg["prediction"]
    if not isinstance(prediction, dict):
        raise ConfigError("prediction must be a mapping.")
    _require_keys(prediction, ["type", "expects"])
    pred_type = prediction["type"]
    if pred_type not in ALLOWED_PREDICTION_TYPES:
        raise ConfigError(f"Unsupported prediction type: {pred_type!r}")
    if not isinstance(prediction["expects"], list):
        raise ConfigError("prediction.expects must be a list.")
    fit = prediction.get("fit")
    if pred_type != "invariant":
        if not isinstance(fit, dict):
            raise ConfigError("prediction.fit must be a mapping.")
        _require_keys(fit, ["model"])
        if "infer" in fit and not isinstance(fit["infer"], list):
            raise ConfigError("prediction.fit.infer must be a list if present.")
        if "level" in fit:
            if fit["level"] not in {"cell", "sample"}:
                raise ConfigError("prediction.fit.level must be 'cell' or 'sample'.")
    elif fit is not None and not isinstance(fit, dict):
        raise ConfigError("prediction.fit must be a mapping if provided for invariant.")

    target = prediction.get("target")
    if target is not None:
        measured_names = {m["name"] for m in measured}
        if target not in measured_names:
            raise ConfigError(f"prediction.target references unknown measured variable: {target!r}")

    run = cfg["run"]
    if not isinstance(run, dict):
        raise ConfigError("run must be a mapping.")
    _require_keys(run, ["grid_over", "n_per_cell", "temperature", "model"])
    if not isinstance(run["grid_over"], list) or len(run["grid_over"]) == 0:
        raise ConfigError("run.grid_over must be a non-empty list.")

    for name in run["grid_over"]:
        if name not in name_to_var:
            raise ConfigError(f"run.grid_over references unknown variable: {name!r}")

    for name, var in name_to_var.items():
        if name not in run["grid_over"]:
            values = _normalized_values(var)
            if len(values) != 1:
                raise ConfigError(
                    f"Controlled variable {name!r} not in grid_over and has multiple values."
                )


def _require_keys(obj: Dict[str, Any], keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ConfigError(f"Missing required keys: {', '.join(missing)}")


def _validate_values(var: Dict[str, Any]) -> None:
    vtype = var["type"]
    values = var["values"]
    if vtype in {"continuous", "binary"}:
        if not isinstance(values, list) or len(values) == 0:
            raise ConfigError(f"{var['name']}: values must be a non-empty list.")
    elif vtype == "categorical":
        if not isinstance(values, (list, dict)) or len(values) == 0:
            raise ConfigError(f"{var['name']}: values must be a non-empty list or mapping.")


def _normalized_values(var: Dict[str, Any]) -> List[Any]:
    values = var["values"]
    if var["type"] == "categorical" and isinstance(values, dict):
        return list(values.keys())
    if isinstance(values, list):
        return values
    return list(values)


def get_controlled_var(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    for var in cfg["variables"]["controlled"]:
        if var["name"] == name:
            return var
    raise ConfigError(f"Unknown controlled variable: {name!r}")


def get_measured_var(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg["variables"]["measured"][0]


def normalized_controlled_values(var: Dict[str, Any]) -> List[Any]:
    return _normalized_values(var)
