from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import yaml
from openai import AsyncOpenAI

from ..analysis.expr import eval_arith
from ..core.prompt import render_game_prompt
from .gamespec import (
    apply_mechanism,
    compute_payoffs,
    evaluate_outcome,
    get_action_labels,
    get_player_names,
    load_gamespec,
    sample_chance,
)

logger = logging.getLogger(__name__)


def run_game(
    gamespec_path: str,
    output_dir: str | None = None,
    concurrency: int = 10,
    n_samples: int | None = None,
    no_plots: bool = False,
) -> str:
    """Run a multi-agent GameSpec with sequential player decisions."""
    spec = load_gamespec(gamespec_path)
    suite = spec.get("suite", {}) if isinstance(spec.get("suite"), dict) else {}
    game_name = spec["game"]["name"]

    n_samples = int(n_samples or suite.get("n_per_cell", 30))
    temperature = float(suite.get("temperature", 0.7))
    base_model = suite.get("model", "gpt-4.1-mini")
    player_models = dict(suite.get("player_models", {}))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_base = output_dir or "outputs"
    out_dir = os.path.join(out_base, f"{timestamp}_{game_name}_multiagent")
    os.makedirs(out_dir, exist_ok=True)

    # Persist the gamespec used for reproducibility
    with open(os.path.join(out_dir, "game.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)

    seed = suite.get("seed")
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)

    logger.info("Running multi-agent game: %s (%d samples per mechanism)", game_name, n_samples)
    logger.info("Output directory: %s", out_dir)

    resolved_models = {p: player_models.get(p, base_model) for p in get_player_names(spec)}
    runner = _MultiAgentRunner(
        spec=spec,
        n_samples=n_samples,
        temperature=temperature,
        base_model=base_model,
        player_models=player_models,
        concurrency=concurrency,
        seed=int(seed),
    )
    results = runner.run()

    samples_path = os.path.join(out_dir, "samples.jsonl")
    with open(samples_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = build_game_summary(spec, results, seed=int(seed))
    summary["game_run_summary"]["player_models"] = resolved_models
    summary_path = os.path.join(out_dir, "game_summary.yaml")
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)

    logger.info("Wrote samples: %s", samples_path)
    logger.info("Wrote summary: %s", summary_path)
    if no_plots:
        logger.info("Plots skipped (no_plots=True).")
    return out_dir


class _MultiAgentRunner:
    def __init__(
        self,
        spec: Dict[str, Any],
        n_samples: int,
        temperature: float,
        base_model: str,
        player_models: Dict[str, str],
        concurrency: int,
        seed: int,
    ) -> None:
        self.spec = spec
        self.n_samples = n_samples
        self.temperature = temperature
        self.base_model = base_model
        self.player_models = player_models
        self.concurrency = concurrency
        self.seed = seed
        self.client = AsyncOpenAI()

    def run(self) -> List[Dict[str, Any]]:
        return asyncio.run(self._run_async())

    async def _run_async(self) -> List[Dict[str, Any]]:
        players = get_player_names(self.spec)
        mechanisms = self.spec.get("mechanisms", [])
        semaphore = asyncio.Semaphore(self.concurrency)
        try:
            tasks = []
            for mech_idx, mech in enumerate(mechanisms):
                for i in range(self.n_samples):
                    tasks.append(
                        self._run_sample(mech, mech_idx, i, players, semaphore)
                    )

            results = await asyncio.gather(*tasks)
            return results
        finally:
            await _close_async_client(self.client)

    async def _run_sample(
        self,
        mechanism: Dict[str, Any],
        mech_idx: int,
        sample_idx: int,
        players: List[str],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        async with semaphore:
            params = apply_mechanism(self.spec["parameters"], mechanism)
            actions: Dict[str, str] = {}
            parse_errors: Dict[str, str] = {}
            prompts: Dict[str, Dict[str, str]] = {}
            raw_responses: Dict[str, str] = {}

            for player in players:
                prompt = render_game_prompt(self.spec, player, params, actions)
                prompts[player] = prompt
                model = self.player_models.get(player, self.base_model)
                response = await _sample_model(
                    self.client,
                    prompt["system"],
                    prompt["user"],
                    model,
                    self.temperature,
                )
                raw_responses[player] = response
                allowed = get_action_labels(self.spec, player)
                action, error = parse_action(response, allowed)
                if action is None:
                    action = allowed[0]
                actions[player] = action
                if error:
                    parse_errors[player] = error

            rng = random.Random(self.seed + mech_idx * 100000 + sample_idx)
            chance_vals = sample_chance(self.spec, params, rng)
            outcome = evaluate_outcome(self.spec, actions, chance_vals)
            payoffs = compute_payoffs(self.spec, params, outcome)

            row = {
                "mechanism": mechanism.get("name", f"mech_{mech_idx}"),
                "sample_idx": sample_idx,
                "actions": actions,
                "outcome": outcome,
                "chance": chance_vals,
                "payoffs": payoffs,
                "prompts": prompts,
                "raw_responses": raw_responses,
            }
            if parse_errors:
                row["parse_errors"] = parse_errors
            return row


async def _sample_model(
    client: AsyncOpenAI,
    system: str,
    user: str,
    model: str,
    temperature: float,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if user:
        messages.append({"role": "user", "content": user})
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        n=1,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


def parse_action(response: str, actions: List[str]) -> Tuple[str | None, str | None]:
    mapping = {a.lower(): a for a in actions}
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "choice" in data:
            choice = str(data["choice"]).strip()
            if choice in actions:
                return choice, None
            lowered = choice.lower()
            if lowered in mapping:
                return mapping[lowered], "choice_case_mismatch"
            return None, "invalid_choice"
    except Exception:
        pass

    # Fallback: scan text for any action token
    ordered = sorted(actions, key=len, reverse=True)
    for action in ordered:
        if re.search(rf"\b{re.escape(action)}\b", response, flags=re.IGNORECASE):
            return action, "choice_from_text"
    return None, "unparseable_choice"


def build_game_summary(
    spec: Dict[str, Any],
    rows: List[Dict[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    players = get_player_names(spec)
    mechanisms = sorted({r["mechanism"] for r in rows})

    summary: Dict[str, Any] = {
        "game_run_summary": {
            "game": spec["game"]["name"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "players": players,
            "mechanisms": {},
        }
    }

    for mech in mechanisms:
        mech_rows = [r for r in rows if r["mechanism"] == mech]
        action_dist: Dict[str, Dict[str, float]] = {}
        outcome_dist: Dict[str, float] = {}
        avg_payoffs: Dict[str, float] = {}

        # Action distributions
        for player in players:
            counts: Dict[str, int] = {}
            for r in mech_rows:
                action = r.get("actions", {}).get(player)
                if action is None:
                    continue
                counts[action] = counts.get(action, 0) + 1
            total = sum(counts.values()) or 1
            action_dist[player] = {k: v / total for k, v in counts.items()}

        # Outcome distribution
        outcome_counts: Dict[str, int] = {}
        for r in mech_rows:
            outcome = r.get("outcome")
            if outcome is None:
                continue
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        total_outcomes = sum(outcome_counts.values()) or 1
        outcome_dist = {k: v / total_outcomes for k, v in outcome_counts.items()}

        # Average payoffs - include all players from payoffs section, not just LLM players
        payoff_players = list(spec.get("payoffs", {}).keys())
        for player in payoff_players:
            values = [r.get("payoffs", {}).get(player) for r in mech_rows]
            numeric = [float(v) for v in values if isinstance(v, (int, float))]
            avg_payoffs[player] = sum(numeric) / len(numeric) if numeric else 0.0

        welfare = _compute_welfare(spec, avg_payoffs)

        summary["game_run_summary"]["mechanisms"][mech] = {
            "action_distribution": action_dist,
            "outcome_distribution": outcome_dist,
            "avg_payoffs": avg_payoffs,
            "welfare": welfare,
            "n_samples": len(mech_rows),
        }

    return summary


def _compute_welfare(spec: Dict[str, Any], payoffs: Dict[str, float]) -> Dict[str, Any]:
    welfare_defs = spec.get("welfare", [])
    welfare_values: Dict[str, Any] = {}
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


async def _close_async_client(client: Any) -> None:
    close = getattr(client, "close", None)
    aclose = getattr(client, "aclose", None)
    for fn in (close, aclose):
        if not callable(fn):
            continue
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.debug("Async client close failed: %s", exc)
        break
