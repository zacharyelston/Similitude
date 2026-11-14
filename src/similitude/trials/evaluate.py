"""Score one candidate of a trial: load the design with its genes, then call the project's own function.

The project supplies `evaluate(design) -> Evaluation`. It gets the design with the candidate's values in place,
runs the project's real calculations and checks, and returns:

- `metrics`: named numbers, which a trial's objectives and constraints refer to;
- `checks`: the `similitude.checks.Check` results, all of them, so a trial cannot score a design differently
  from the way the project checks it.

The function must give the same answer for the same design every time, and must not depend on anything a
candidate before it left behind: candidates run in separate processes, in no particular order.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from similitude import design
from similitude.checks import Check, Status


@dataclass(frozen=True)
class Evaluation:
    """What a project's `evaluate` returns for one design."""

    metrics: Mapping[str, float]
    checks: Sequence[Check] = ()


Evaluate = Callable[[design.Design], Evaluation]


def evaluate_candidate(design_path: str | Path, overrides: Mapping[str, Any], evaluate: Evaluate) -> dict:
    """One candidate's result, as plain data that goes into the history.

    A candidate that cannot be loaded or evaluated comes back as `{"error": ...}`, and loses to every other.
    An `evaluate` that breaks its contract (the wrong return type, a metric that is not a number) is a bug in
    the project, not a bad candidate, so that raises instead."""
    try:
        result = evaluate(design.load(design_path, overrides))
    except Exception as e:  # a combination of values that cannot be built or checked
        return {"error": f"{type(e).__name__}: {e}"}
    if not isinstance(result, Evaluation):
        raise TypeError(f"evaluate must return a similitude.trials.Evaluation, not {type(result).__name__}")

    metrics = {}
    for name, value in result.metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"metric {name!r} must be a plain number, not {type(value).__name__}; "
                "turn a quantity into one with .to(unit).magnitude"
            )
        if not math.isfinite(value):
            return {"error": f"metric {name!r} is {value}"}
        metrics[name] = float(value)

    checks = list(result.checks)
    for check in checks:
        if not isinstance(check, Check):
            raise TypeError(f"checks must be similitude.checks.Check records, not {type(check).__name__}")
    return {
        "metrics": metrics,
        "fails": sum(c.status is Status.FAIL for c in checks),
        "warns": sum(c.status is Status.WARN for c in checks),
        "checks": [c.to_dict() for c in checks],
    }
