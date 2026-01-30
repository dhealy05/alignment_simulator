from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List

from .config import get_controlled_var, normalized_controlled_values

logger = logging.getLogger(__name__)


def build_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    grid_vars = cfg["run"]["grid_over"]
    logger.debug("Building grid over variables: %s", grid_vars)

    grids = []
    for name in grid_vars:
        var = get_controlled_var(cfg, name)
        values = normalized_controlled_values(var)
        logger.debug("  %s: %d values -> %s", name, len(values), values)
        grids.append(values)

    cells = []
    for combo in itertools.product(*grids):
        cell = {}
        for name, value in zip(grid_vars, combo):
            cell[name] = value
        cells.append(cell)

    # Add fixed controlled variables not in grid_over (single-value)
    for var in cfg["variables"]["controlled"]:
        name = var["name"]
        if name in grid_vars:
            continue
        value = normalized_controlled_values(var)[0]
        logger.debug("  %s (fixed): %s", name, value)
        for cell in cells:
            cell[name] = value

    logger.debug("Grid built: %d cells", len(cells))
    return cells
