"""Inspect Scorer for multi-agent game outcomes."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict

from inspect_ai.scorer import Score, Scorer, mean, scorer, stderr
from inspect_ai.solver import TaskState

from alignment_sim.games.gamespec import compute_payoffs, evaluate_outcome, sample_chance

# Target is not used for game scoring (no ground truth answer)
# but we need to accept it per the Scorer protocol


@scorer(metrics=[mean(), stderr()])
def game_scorer() -> Scorer:
    """Score game outcomes and compute payoffs.

    This scorer:
    1. Samples any chance events (using deterministic RNG)
    2. Evaluates the outcome based on player actions
    3. Computes payoffs for each player
    4. Returns total welfare as the primary score

    The scorer stores detailed metadata including:
    - mechanism: Name of the mechanism
    - actions: Dict of player -> action
    - outcome: Name of the matched outcome
    - chance: Dict of chance event -> bool
    - payoffs: Dict of player -> payoff value

    Returns:
        Scorer that evaluates game samples.
    """

    async def score(state: TaskState, target: Any) -> Score:
        spec: Dict[str, Any] = state.metadata["gamespec"]
        actions: Dict[str, str] = state.metadata["actions"]
        params: Dict[str, float] = state.metadata["params"]
        mechanism: Dict[str, Any] = state.metadata["mechanism"]
        sample_idx: int = state.metadata["sample_idx"]

        mech_name = mechanism.get("name", "unknown")

        # Deterministic RNG based on mechanism and sample index
        # This ensures reproducibility across runs.
        seed = _stable_seed(mech_name, sample_idx)
        rng = random.Random(seed)

        # Sample chance events
        chance_vals = sample_chance(spec, params, rng)

        # Evaluate outcome based on actions and chance
        outcome = evaluate_outcome(spec, actions, chance_vals)

        # Compute payoffs for each player
        payoffs = compute_payoffs(spec, params, outcome)

        # Primary score is total welfare (sum of all payoffs)
        total_payoff = sum(
            v for v in payoffs.values() if v is not None and isinstance(v, (int, float))
        )

        # Format explanation
        action_str = ", ".join(f"{p}={a}" for p, a in actions.items())
        payoff_str = ", ".join(
            f"{p}={v:.2f}" if isinstance(v, float) else f"{p}={v}"
            for p, v in payoffs.items()
        )

        explanation = f"Actions: {action_str}\nOutcome: {outcome}\nPayoffs: {payoff_str}"

        return Score(
            value=total_payoff,
            answer=outcome,
            explanation=explanation,
            metadata={
                "mechanism": mech_name,
                "actions": actions,
                "outcome": outcome,
                "chance": chance_vals,
                "payoffs": payoffs,
                "params": params,
                "transcripts": state.metadata.get("transcripts", {}),
                "parse_errors": state.metadata.get("parse_errors", {}),
            },
        )

    return score


@scorer(metrics=[mean(), stderr()])
def player_payoff_scorer(player: str) -> Scorer:
    """Score a specific player's payoff.

    This is useful when you want separate metrics for each player.

    Args:
        player: Name of the player to score.

    Returns:
        Scorer that returns the specified player's payoff.
    """

    async def score(state: TaskState, target: Any) -> Score:
        spec: Dict[str, Any] = state.metadata["gamespec"]
        actions: Dict[str, str] = state.metadata["actions"]
        params: Dict[str, float] = state.metadata["params"]
        mechanism: Dict[str, Any] = state.metadata["mechanism"]
        sample_idx: int = state.metadata["sample_idx"]

        mech_name = mechanism.get("name", "unknown")

        seed = _stable_seed(mech_name, sample_idx)
        rng = random.Random(seed)

        chance_vals = sample_chance(spec, params, rng)
        outcome = evaluate_outcome(spec, actions, chance_vals)
        payoffs = compute_payoffs(spec, params, outcome)

        player_payoff = payoffs.get(player)
        if player_payoff is None:
            player_payoff = 0.0

        return Score(
            value=float(player_payoff),
            answer=outcome,
            explanation=f"{player} payoff: {player_payoff}",
            metadata={
                "mechanism": mech_name,
                "player": player,
                "action": actions.get(player),
                "payoff": player_payoff,
            },
        )

    return score


def _stable_seed(mech_name: str, sample_idx: int) -> int:
    payload = f"{mech_name}|{sample_idx}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")
