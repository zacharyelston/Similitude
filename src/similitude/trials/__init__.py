"""Design trials: search a design's values against the project's own checks.

    from similitude import trials

    result = trials.run("trials/lighter-can.toml", evaluate)
    result.best["genes"]       # the best candidate's values
    result.best_toml           # a design file: the trial's design plus [overrides]

- `engine`: a trial file, the seeded genetic algorithm and the ranking.
- `evaluate`: what a project's `evaluate(design)` returns, and how one candidate is scored.
"""

from similitude.trials.engine import (
    Constraint,
    Gene,
    Objective,
    Result,
    Trial,
    TrialError,
    best_toml,
    load,
    rank_key,
    run,
    score,
    violation,
)
from similitude.trials.evaluate import Evaluate, Evaluation, evaluate_candidate

__all__ = [
    "Constraint",
    "Evaluate",
    "Evaluation",
    "Gene",
    "Objective",
    "Result",
    "Trial",
    "TrialError",
    "best_toml",
    "evaluate_candidate",
    "load",
    "rank_key",
    "run",
    "score",
    "violation",
]
