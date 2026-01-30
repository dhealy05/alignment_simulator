"""Inspect Solver for multi-agent sequential games."""

from __future__ import annotations

import logging
from typing import Any, Dict

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.solver import Solver, TaskState, solver

from alignment_sim.core.prompt import render_game_prompt
from alignment_sim.games.game_runner import parse_action
from alignment_sim.games.gamespec import apply_mechanism, get_action_labels, get_player_names

logger = logging.getLogger(__name__)


@solver
def game_solver() -> Solver:
    """Multi-agent sequential game solver.

    This solver orchestrates a multi-agent game where each player
    makes decisions sequentially. Each player is assigned a model
    via Inspect's model roles system.

    The solver:
    1. Iterates through players in order
    2. Renders a prompt for each player (with information they can observe)
    3. Generates a response using the player's assigned model
    4. Parses the action from the response
    5. Stores all actions and transcripts in state.metadata for the scorer

    Model roles should match player names in the GameSpec. For example:
        --model-role Firm=anthropic/claude-sonnet-4
        --model-role AI=openai/gpt-4.1

    Returns:
        Solver that processes game samples.
    """

    async def solve(state: TaskState, generate) -> TaskState:
        spec: Dict[str, Any] = state.metadata["gamespec"]
        mechanism: Dict[str, Any] = state.metadata["mechanism"]

        # Apply mechanism overrides to parameters
        params = apply_mechanism(spec["parameters"], mechanism)

        actions: Dict[str, str] = {}
        transcripts: Dict[str, str] = {}
        parse_errors: Dict[str, str] = {}

        players = get_player_names(spec)
        logger.debug(
            "Running game with mechanism=%s, players=%s",
            mechanism.get("name"),
            players,
        )

        for player in players:
            # Get model for this player role
            # Falls back to default model if role not specified
            try:
                model = get_model(role=player)
            except Exception:
                # If role not found, use default model
                model = get_model()
                logger.warning(
                    "No model role defined for player %r, using default model", player
                )

            # Render prompt with information this player can observe
            prompt = render_game_prompt(spec, player, params, actions)

            # Generate response
            messages = []
            if prompt.get("system"):
                messages.append(ChatMessageSystem(content=prompt["system"]))
            if prompt.get("user"):
                messages.append(ChatMessageUser(content=prompt["user"]))

            response = await model.generate(messages)
            response_text = response.completion

            transcripts[player] = response_text

            # Parse action from response
            allowed = get_action_labels(spec, player)
            action, error = parse_action(response_text, allowed)

            if action is None:
                # Default to first action if parsing fails
                action = allowed[0]
                logger.warning(
                    "Failed to parse action for player %r, defaulting to %r",
                    player,
                    action,
                )

            actions[player] = action
            if error:
                parse_errors[player] = error

        # Store results for scorer
        state.metadata["actions"] = actions
        state.metadata["params"] = params
        state.metadata["transcripts"] = transcripts
        if parse_errors:
            state.metadata["parse_errors"] = parse_errors

        logger.debug("Game complete: actions=%s", actions)

        return state

    return solve
