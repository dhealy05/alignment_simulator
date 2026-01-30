"""Inspect framework compatibility layer for AlignmentSim.

This module provides Inspect-compatible Task, Solver, and Scorer
implementations for running GameSpec multi-agent games.

Usage:
    inspect eval alignment_sim/inspect_compatibility:audit_game \
      --model-role Firm=anthropic/claude-sonnet-4-20250514 \
      --model-role AI=openai/gpt-4.1 \
      -T gamespec=games/audit_deception.yaml \
      -T n_samples=30
"""

from .tasks import audit_game
from .solver import game_solver
from .scorer import game_scorer
from .aggregate import aggregate_results, build_summary

__all__ = [
    "audit_game",
    "game_solver",
    "game_scorer",
    "aggregate_results",
    "build_summary",
]
