"""Inspect Task definitions for AlignmentSim games."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from alignment_sim.games.gamespec import get_player_names, load_gamespec
from alignment_sim.inspect_compatibility.scorer import game_scorer, player_payoff_scorer
from alignment_sim.inspect_compatibility.solver import game_solver

logger = logging.getLogger(__name__)


@task
def audit_game(
    gamespec: str,
    n_samples: int = 30,
    per_player_scores: bool = False,
) -> Task:
    """Run a multi-agent game from a GameSpec.

    This task runs a multi-agent sequential game where each player
    is controlled by a different model. Players make decisions in
    order, and later players can observe earlier players' actions
    (subject to the information structure defined in the GameSpec).

    Args:
        gamespec: Path to the GameSpec YAML file.
        n_samples: Number of samples per mechanism (default: 30).
        per_player_scores: If True, add separate scorers for each player's payoff.

    Returns:
        Inspect Task configured for the game.

    Example:
        ```bash
        inspect eval alignment_sim/inspect_compatibility:audit_game \
          --model-role Firm=anthropic/claude-sonnet-4 \
          --model-role AI=openai/gpt-4.1 \
          -T gamespec=games/audit_deception.yaml \
          -T n_samples=30
        ```
    """
    spec = load_gamespec(gamespec)
    game_name = spec["game"]["name"]

    logger.info(
        "Creating task for game %r with %d samples per mechanism",
        game_name,
        n_samples,
    )

    # Create samples for each (mechanism, sample_idx) combination
    samples: List[Sample] = []
    for mech in spec["mechanisms"]:
        mech_name = mech.get("name", "unnamed")
        for i in range(n_samples):
            samples.append(
                Sample(
                    id=f"{mech_name}_{i}",
                    input=f"Run game under mechanism: {mech_name}",
                    metadata={
                        "mechanism": mech,
                        "sample_idx": i,
                        "gamespec": spec,
                    },
                )
            )

    logger.info(
        "Created %d samples across %d mechanisms",
        len(samples),
        len(spec["mechanisms"]),
    )

    # Build scorer list
    scorers = [game_scorer()]
    if per_player_scores:
        for player in get_player_names(spec):
            scorers.append(player_payoff_scorer(player))

    return Task(
        dataset=MemoryDataset(samples, name=game_name),
        solver=game_solver(),
        scorer=scorers if len(scorers) > 1 else scorers[0],
        name=game_name,
    )


@task
def multi_game(
    gamespecs: List[str],
    n_samples: int = 30,
) -> Task:
    """Run multiple games in a single evaluation.

    This is useful for comparing behavior across different game structures.

    Args:
        gamespecs: List of paths to GameSpec YAML files.
        n_samples: Number of samples per mechanism per game.

    Returns:
        Inspect Task combining all games.
    """
    all_samples: List[Sample] = []

    for gamespec_path in gamespecs:
        spec = load_gamespec(gamespec_path)
        game_name = spec["game"]["name"]

        for mech in spec["mechanisms"]:
            mech_name = mech.get("name", "unnamed")
            for i in range(n_samples):
                all_samples.append(
                    Sample(
                        id=f"{game_name}_{mech_name}_{i}",
                        input=f"Game: {game_name}, Mechanism: {mech_name}",
                        metadata={
                            "game": game_name,
                            "mechanism": mech,
                            "sample_idx": i,
                            "gamespec": spec,
                        },
                    )
                )

    return Task(
        dataset=MemoryDataset(all_samples, name="multi_game"),
        solver=game_solver(),
        scorer=game_scorer(),
        name="multi_game",
    )
