from __future__ import annotations

import logging
from typing import Any, Dict, List

from types import SimpleNamespace

logger = logging.getLogger(__name__)


def render_prompts(cfg: Dict[str, Any], cell: Dict[str, Any]) -> Dict[str, Any]:
    """Render prompts for a cell.

    Returns:
        Dict with keys: system, user, history (list of {role, content} dicts)
    """
    logger.debug("Rendering prompts for cell: %s", cell)
    system_parts = []
    user_parts = []
    tool_parts = []
    history: List[Dict[str, str]] = []
    format_context: Dict[str, Any] = dict(cell)

    for var in cfg["variables"]["controlled"]:
        inject = var["inject"]
        template = var.get("template", "")
        vtype = var["type"]
        values = var["values"]

        if inject == "system_prompt":
            if template:
                rendered = template.format(**format_context)
                logger.debug("  System prompt part from %s: %s", var["name"], rendered[:50] + "..." if len(rendered) > 50 else rendered)
                system_parts.append(rendered)
        elif inject == "user_message":
            if vtype == "categorical" and isinstance(values, dict):
                scenario_data = values[cell[var["name"]]]
                if isinstance(scenario_data, dict):
                    scenario = scenario_data.get("scenario", "")
                    format_context[var["name"]] = SimpleNamespace(**scenario_data)
                    # Extract history if present
                    scenario_history = scenario_data.get("history", [])
                else:
                    scenario = str(scenario_data)
                    scenario_history = []
                if scenario_history:
                    history.extend(scenario_history)
                    logger.debug("  History from %s: %d turns", var["name"], len(scenario_history))
                if scenario:
                    logger.debug("  User message from %s scenario: %s", var["name"], scenario[:50] + "..." if len(scenario) > 50 else scenario)
                    user_parts.append(scenario)
                elif template:
                    user_parts.append(template.format(**format_context))
            else:
                if template:
                    user_parts.append(template.format(**format_context))
        elif inject == "tool_context":
            if template:
                rendered = template.format(**format_context)
                tool_parts.append(rendered)

    if tool_parts:
        system_parts.append("Tool context:\n" + "\n".join(tool_parts))

    if any(m.get("method") == "json_field" for m in cfg.get("variables", {}).get("measured", [])):
        json_schema = cfg.get("run", {}).get("json_schema")
        if not json_schema:
            for measured in cfg["variables"]["measured"]:
                if measured.get("method") == "json_field":
                    json_schema = measured.get("params", {}).get("schema")
                    if json_schema:
                        break
        if isinstance(json_schema, dict):
            import json as _json

            json_schema = _json.dumps(json_schema, ensure_ascii=True)
        if json_schema:
            system_parts.append(f"Respond with valid JSON only. Use this schema:\n{json_schema}")
        else:
            system_parts.append("Respond with valid JSON only.")

    system = "\n".join(p for p in system_parts if p).strip()
    user = "\n".join(p for p in user_parts if p).strip()
    logger.debug("Rendered system prompt (%d chars), user prompt (%d chars), history (%d turns)",
                 len(system), len(user), len(history))
    return {"system": system, "user": user, "history": history}


def render_game_prompt(
    spec: Dict[str, Any],
    player: str,
    params: Dict[str, float],
    actions_so_far: Dict[str, str],
) -> Dict[str, str]:
    """Render prompts for a single player in a multi-agent game run."""
    info = spec.get("information", {})
    player_info = info.get(player, {}) if isinstance(info, dict) else {}
    observes = player_info.get("observes")
    observes_actions = player_info.get("observes_actions", True)

    if isinstance(observes, list) and observes:
        observed_params = observes
    else:
        observed_params = list(params.keys())

    actions = spec.get("actions", {}).get(player, [])
    role = None
    for p in spec.get("players", []):
        if p.get("name") == player:
            role = p.get("role")
            break

    system_lines = []
    if role:
        system_lines.append(f"You are {role}")
    system_lines.append("Choose one action from the list below.")
    system_lines.append("")
    system_lines.append("Actions:")
    for action in actions:
        system_lines.append(f"- {action}")
    system_lines.append("")
    system_lines.append('Respond with JSON only: {"choice":"<action>"}')
    system_lines.append(f"Valid actions: {', '.join(actions)}")

    user_lines = []
    if observed_params:
        user_lines.append("Mechanism details:")
        for name in observed_params:
            if name in params:
                user_lines.append(f"- {name}: {params[name]}")
        user_lines.append("")

    if observes_actions:
        if actions_so_far:
            user_lines.append("Previous actions:")
            for pname, action in actions_so_far.items():
                user_lines.append(f"- {pname}: {action}")
        else:
            user_lines.append("Previous actions: none.")

    system = "\n".join(system_lines).strip()
    user = "\n".join(user_lines).strip()
    return {"system": system, "user": user}
