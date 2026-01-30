from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

import yaml

from ..analysis.expr import ExprError, eval_arith, eval_bool, validate_arith, validate_bool

logger = logging.getLogger(__name__)


class GameSpecError(ValueError):
    pass


def load_gamespec(path: str) -> Dict[str, Any]:
    logger.info("Loading GameSpec from: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise GameSpecError("GameSpec must be a YAML mapping/object at top level.")
    validate_gamespec(data)
    return data


def validate_gamespec(spec: Dict[str, Any]) -> None:
    _require_keys(spec, ["game", "players", "actions", "parameters", "mechanisms", "outcomes", "payoffs"])

    game = spec["game"]
    if not isinstance(game, dict):
        raise GameSpecError("game must be a mapping.")
    _require_keys(game, ["name", "class", "description"])
    if game["class"] != "mechanism_design":
        raise GameSpecError("v0.3 supports only game.class = 'mechanism_design'.")

    players = spec["players"]
    if not isinstance(players, list) or not players:
        raise GameSpecError("players must be a non-empty list.")
    player_names = []
    for p in players:
        if not isinstance(p, dict):
            raise GameSpecError("Each player must be a mapping.")
        _require_keys(p, ["name", "role"])
        player_names.append(p["name"])
    if len(set(player_names)) != len(player_names):
        raise GameSpecError("Player names must be unique.")

    actions = spec["actions"]
    if not isinstance(actions, dict):
        raise GameSpecError("actions must be a mapping.")
    for name in player_names:
        if name not in actions:
            raise GameSpecError(f"Missing actions for player: {name!r}")
        if not isinstance(actions[name], list) or len(actions[name]) < 2:
            raise GameSpecError(f"actions[{name!r}] must be a list with at least 2 actions.")

    information = spec.get("information", {})
    if information is not None and not isinstance(information, dict):
        raise GameSpecError("information must be a mapping if provided.")
    if isinstance(information, dict):
        for player, info in information.items():
            if not isinstance(info, dict):
                raise GameSpecError("information entries must be mappings.")
            observes = info.get("observes", [])
            if observes is None:
                continue
            if not isinstance(observes, list):
                raise GameSpecError("information.<player>.observes must be a list.")
            observes_actions = info.get("observes_actions")
            if observes_actions is not None and not isinstance(observes_actions, bool):
                raise GameSpecError("information.<player>.observes_actions must be a boolean if provided.")

    parameters = spec["parameters"]
    if not isinstance(parameters, dict) or not parameters:
        raise GameSpecError("parameters must be a non-empty mapping.")
    for k, v in parameters.items():
        if not isinstance(v, (int, float)):
            raise GameSpecError(f"Parameter {k!r} must be numeric.")

    mechanisms = spec["mechanisms"]
    if not isinstance(mechanisms, list) or not mechanisms:
        raise GameSpecError("mechanisms must be a non-empty list.")
    mech_names = set()
    for mech in mechanisms:
        if not isinstance(mech, dict):
            raise GameSpecError("Each mechanism must be a mapping.")
        _require_keys(mech, ["name", "overrides"])
        if mech["name"] in mech_names:
            raise GameSpecError(f"Duplicate mechanism name: {mech['name']!r}")
        mech_names.add(mech["name"])
        overrides = mech["overrides"]
        if not isinstance(overrides, dict):
            raise GameSpecError("mechanism.overrides must be a mapping.")
        for k, v in overrides.items():
            if k not in parameters:
                raise GameSpecError(f"Mechanism override references unknown parameter: {k!r}")
            if not isinstance(v, (int, float)):
                raise GameSpecError(f"Override {k!r} must be numeric.")

    chance = spec.get("chance", {})
    if chance is not None and not isinstance(chance, dict):
        raise GameSpecError("chance must be a mapping if provided.")
    if isinstance(chance, dict):
        for name, expr in chance.items():
            if not isinstance(expr, str):
                raise GameSpecError(f"chance[{name!r}] must be a string expression.")
            _validate_chance_expr(expr, parameters.keys())

    if isinstance(information, dict):
        for player, info in information.items():
            observes = info.get("observes", [])
            if observes is None:
                continue
            for param in observes:
                if param not in parameters:
                    raise GameSpecError(f"information[{player!r}] references unknown parameter: {param!r}")

    outcomes = spec["outcomes"]
    if not isinstance(outcomes, list) or not outcomes:
        raise GameSpecError("outcomes must be a non-empty list.")
    outcome_names = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise GameSpecError("Each outcome must be a mapping.")
        _require_keys(outcome, ["name", "when"])
        outcome_names.append(outcome["name"])
        if not isinstance(outcome["when"], str):
            raise GameSpecError("Outcome 'when' must be a string.")
    if len(set(outcome_names)) != len(outcome_names):
        raise GameSpecError("Outcome names must be unique.")

    # Validate outcome guards
    action_labels = set()
    for actions_list in actions.values():
        action_labels.update(actions_list)
    allowed_names = set(player_names) | action_labels | set(chance.keys() if isinstance(chance, dict) else [])
    for outcome in outcomes:
        try:
            validate_bool(outcome["when"], allowed_names)
        except ExprError as exc:
            raise GameSpecError(str(exc)) from exc

    payoffs = spec["payoffs"]
    if not isinstance(payoffs, dict) or not payoffs:
        raise GameSpecError("payoffs must be a non-empty mapping.")
    for player in player_names:
        if player not in payoffs:
            raise GameSpecError(f"Missing payoffs for player: {player!r}")
        if not isinstance(payoffs[player], dict):
            raise GameSpecError(f"payoffs[{player!r}] must be a mapping.")
        for outcome_name, expr in payoffs[player].items():
            if outcome_name not in outcome_names:
                raise GameSpecError(f"payoffs[{player!r}] references unknown outcome: {outcome_name!r}")
            if not isinstance(expr, str):
                raise GameSpecError("Payoff expression must be a string.")
            try:
                validate_arith(expr, parameters.keys())
            except ExprError as exc:
                raise GameSpecError(str(exc)) from exc

    welfare = spec.get("welfare")
    if welfare is not None:
        if not isinstance(welfare, list):
            raise GameSpecError("welfare must be a list if provided.")
        for w in welfare:
            if not isinstance(w, dict):
                raise GameSpecError("Each welfare entry must be a mapping.")
            _require_keys(w, ["name", "formula"])
            if not isinstance(w["formula"], str):
                raise GameSpecError("welfare formula must be a string.")

    suite = spec.get("suite")
    if suite is not None and not isinstance(suite, dict):
        raise GameSpecError("suite must be a mapping if provided.")
    if isinstance(suite, dict):
        player_models = suite.get("player_models")
        if player_models is not None:
            if not isinstance(player_models, dict):
                raise GameSpecError("suite.player_models must be a mapping if provided.")
            for player, model in player_models.items():
                if player not in player_names:
                    raise GameSpecError(f"suite.player_models references unknown player: {player!r}")
                if not isinstance(model, str):
                    raise GameSpecError(f"suite.player_models[{player!r}] must be a string model name.")


def apply_mechanism(parameters: Dict[str, float], mechanism: Dict[str, Any]) -> Dict[str, float]:
    result = dict(parameters)
    overrides = mechanism.get("overrides", {})
    for k, v in overrides.items():
        result[k] = float(v)
    return result


def get_player_names(spec: Dict[str, Any]) -> List[str]:
    return [p["name"] for p in spec["players"]]


def get_action_labels(spec: Dict[str, Any], player: str) -> List[str]:
    return list(spec["actions"][player])


def _require_keys(obj: Dict[str, Any], keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise GameSpecError(f"Missing required keys: {', '.join(missing)}")


def _validate_chance_expr(expr: str, parameters: Iterable[str]) -> None:
    expr = expr.strip()
    if not expr.startswith("Bernoulli(") or not expr.endswith(")"):
        raise GameSpecError(f"Unsupported chance expression: {expr!r}")
    inner = expr[len("Bernoulli(") : -1].strip()
    if not inner:
        raise GameSpecError(f"Invalid chance expression: {expr!r}")
    if inner in parameters:
        return
    try:
        float(inner)
    except ValueError as exc:
        raise GameSpecError(f"Chance expression must reference a parameter or number: {expr!r}") from exc


def sample_chance(
    spec: Dict[str, Any],
    parameters: Dict[str, float],
    rng: Any,
) -> Dict[str, bool]:
    chance = spec.get("chance", {})
    if not isinstance(chance, dict):
        return {}
    results: Dict[str, bool] = {}
    for name, expr in chance.items():
        prob = _chance_probability(expr, parameters)
        results[name] = rng.random() < prob
    return results


def evaluate_outcome(
    spec: Dict[str, Any],
    actions: Dict[str, str],
    chance_vals: Dict[str, bool],
) -> str:
    outcomes = spec.get("outcomes", [])
    action_labels = set()
    for actions_list in spec.get("actions", {}).values():
        action_labels.update(actions_list)
    context: Dict[str, Any] = {}
    for player, action in actions.items():
        context[player] = action
    for label in action_labels:
        context[label] = label
    context.update(chance_vals)

    matched = []
    for outcome in outcomes:
        expr = outcome.get("when", "")
        try:
            if eval_bool(expr, context):
                matched.append(outcome["name"])
        except Exception:
            logger.warning("Failed to evaluate outcome guard: %s", expr)
            continue

    if not matched:
        return "__invalid__"
    if len(matched) > 1:
        logger.warning("Multiple outcomes matched (%s); using first.", ", ".join(matched))
    return matched[0]


def compute_payoffs(
    spec: Dict[str, Any],
    parameters: Dict[str, float],
    outcome: str,
) -> Dict[str, float | None]:
    payoffs = spec.get("payoffs", {})
    results: Dict[str, float | None] = {}
    for player, payoff_map in payoffs.items():
        if outcome not in payoff_map:
            results[player] = None
            continue
        try:
            results[player] = eval_arith(payoff_map[outcome], parameters)
        except Exception:
            results[player] = None
    return results


def _chance_probability(expr: str, parameters: Dict[str, float]) -> float:
    expr = expr.strip()
    inner = expr[len("Bernoulli(") : -1].strip()
    if inner in parameters:
        return float(parameters[inner])
    return float(inner)
