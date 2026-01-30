# Inspect Compatibility Layer

This module provides [Inspect](https://inspect.aisi.org.uk/) framework compatibility for AlignmentSim's multi-agent games.

## Overview

The Inspect compatibility layer allows you to run GameSpec games using Inspect's CLI and infrastructure. This provides:

- Model roles for assigning different models to different players
- Inspect's logging and caching infrastructure
- Compatibility with other Inspect-based tools like Petri

## Quick Start

### 1. Install Inspect

```bash
pip install inspect-ai
```

### 2. Run a Game

```bash
inspect eval alignment_sim/inspect_compatibility:audit_game \
  --model-role Firm=anthropic/claude-sonnet-4-20250514 \
  --model-role AI=openai/gpt-4.1 \
  -T gamespec=games/audit_deception.yaml \
  -T n_samples=30
```

### 3. Aggregate Results

```bash
# Using the CLI
alignmentsim aggregate-inspect ./logs/eval_xxx.json --output summary.yaml

# Or programmatically
from alignment_sim.inspect_compatibility import aggregate_results
summary = aggregate_results("./logs/eval_xxx.json")
```

## Components

### Tasks

- **`audit_game`**: Run a single GameSpec with configurable samples per mechanism
- **`multi_game`**: Run multiple GameSpecs in a single evaluation

### Solver

- **`game_solver`**: Orchestrates sequential player decisions using model roles

### Scorer

- **`game_scorer`**: Computes outcomes, payoffs, and total welfare
- **`player_payoff_scorer(player)`**: Scores a specific player's payoff

### Aggregation

- **`aggregate_results(log_path)`**: Aggregate Inspect logs into mechanism-level summaries
- **`build_summary(by_mechanism)`**: Build summary statistics from grouped results
- **`compare_mechanisms(summary)`**: Compare outcomes across mechanisms

## Model Roles

Model roles map player names from your GameSpec to specific models. Use the `--model-role` flag:

```bash
--model-role PlayerName=provider/model-name
```

For example, with a GameSpec that has players `Firm` and `AI`:

```bash
inspect eval alignment_sim/inspect_compatibility:audit_game \
  --model-role Firm=anthropic/claude-sonnet-4-20250514 \
  --model-role AI=anthropic/claude-3-5-haiku-20241022 \
  -T gamespec=games/audit_deception.yaml
```

If a player has no assigned role, the default model is used.

## Task Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamespec` | str | required | Path to GameSpec YAML file |
| `n_samples` | int | 30 | Samples per mechanism |
| `per_player_scores` | bool | False | Add separate scorers for each player |

## Output

The scorer stores detailed metadata for each sample:

```python
{
    "mechanism": "baseline",
    "actions": {"Firm": "Audit", "AI": "Honest"},
    "outcome": "mutual_cooperation",
    "chance": {},
    "payoffs": {"Firm": 10.0, "AI": 8.0},
    "params": {"audit_cost": 2.0, ...},
    "transcripts": {"Firm": "...", "AI": "..."},
}
```

## Example: Full Workflow

```bash
# 1. Run the game
inspect eval alignment_sim/inspect_compatibility:audit_game \
  --model-role Firm=anthropic/claude-sonnet-4-20250514 \
  --model-role AI=openai/gpt-4.1 \
  -T gamespec=games/audit_deception.yaml \
  -T n_samples=50 \
  --log-dir ./game_logs

# 2. Aggregate results
alignmentsim aggregate-inspect ./game_logs/eval_xxx.json -o results.yaml

# 3. View in Inspect's log viewer
inspect view ./game_logs
```

## Programmatic Usage

```python
from inspect_ai import eval
from alignment_sim.inspect_compatibility import audit_game, aggregate_results

# Run evaluation
log = eval(
    audit_game(gamespec="games/audit_deception.yaml", n_samples=30),
    model_roles={
        "Firm": "anthropic/claude-sonnet-4-20250514",
        "AI": "openai/gpt-4.1",
    },
)

# Aggregate results
summary = aggregate_results(log[0].location)
print(summary)
```
