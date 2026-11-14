"""Search a design's values: a seeded genetic algorithm over a trial's genes.

A trial is a TOML file. It names the design, the genes (design values allowed to change, with their ranges or
choices), the objectives (metrics to push up or down) and the constraints (metrics that must stay in bounds):

    [trial]
    name = "lighter-can"
    design = "can.toml"            # relative to this file
    question = "How light can the can be and still hold a litre?"
    population = 12
    generations = 8
    seed = 1

    [[trial.gene]]
    key = "body.radius"            # a dotted key, as in a design's [overrides]
    min = 30
    max = 80
    step = 1
    unit = "mm"                    # needed when the design value has a unit
    why = "the can's proportions"

    [[trial.gene]]
    key = "body.lid"
    choices = ["pressed", "seamed"]

    [[trial.objective]]
    metric = "mass_g"
    goal = "min"                   # or "max"
    weight = 1                     # optional
    scale = 1                      # optional: the metric's "one unit of caring"

    [[trial.constraint]]
    metric = "volume_cm3"
    min = 1000                     # and/or max

`[trial]` also takes `elite`, `mutation`, `mutation_rate` and `crossover`.

`run(trial, evaluate)` scores every candidate with the project's own `evaluate` (see `similitude.trials.evaluate`)
and writes two files:

- `history.jsonl`: the design as it stands, then every candidate of every generation, best first, with its genes,
  the overrides it was loaded with, its metrics and every check;
- `best.toml`: the best candidate as a design file, which includes the trial's design and sets `[overrides]`.
  The include is a relative path, and is only written inside the design's own project (the nearest folder above
  it with a `.git` or a `pyproject.toml`); anywhere else the file holds the `[overrides]` alone.

How candidates are ranked, best first:

1. a candidate that could be evaluated, before one that could not;
2. fewer FAIL checks, so a candidate with a FAIL loses to every candidate without one;
3. the smaller constraint violation, so one that breaks a constraint loses to every one that breaks none;
4. the lower score: each objective adds `weight * metric / scale`, negated for "max".

Each generation keeps the best few candidates unchanged (elitism) and breeds the rest by tournament selection,
uniform crossover and mutation.

A run is deterministic for a seed. Every random draw is made here, in one process and in a fixed order, and
candidates are scored in worker processes that draw nothing, so the result does not depend on how many workers
there are. Nothing is reused from an earlier run: each run scores its candidates afresh.

A gene's range is where the search may look. It is not a statement about uncertainty: a design value that is a
`{ref, range}` keeps its bounds, and a gene on it names its `.ref`.

The algorithm, `Gene`, `Objective`, `Constraint` and the ranking are ported from eOx's `src/eox/evolve/engine.py`
(MIT); see NOTICE.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import pickle
import random
import tomllib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pint

from similitude import design
from similitude.trials.evaluate import Evaluate, evaluate_candidate

_TRIAL_KEYS = {
    "name", "design", "question", "population", "generations", "seed", "elite", "mutation", "mutation_rate",
    "crossover", "gene", "objective", "constraint",
}  # fmt: skip
_GENE_KEYS = {"key", "min", "max", "step", "choices", "unit", "why"}
_OBJECTIVE_KEYS = {"metric", "goal", "weight", "scale"}
_CONSTRAINT_KEYS = {"metric", "min", "max"}


class TrialError(ValueError):
    """A trial file is wrong, or does not fit its design or its `evaluate`."""


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _same(a: Any, b: Any) -> bool:
    """Equal, and of one type: True is not 1, and 1 is not 1.0."""
    return type(a) is type(b) and a == b


@dataclass(frozen=True)
class Gene:
    """A design value the trial may change: a range `lo..hi` (in steps of `step`, if given), or `choices`."""

    key: str
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    choices: tuple | None = None
    unit: str = ""
    why: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise TrialError("a gene needs a key: the dotted name of a design value")
        if self.choices is not None:
            object.__setattr__(self, "choices", tuple(self.choices))
            if self.lo is not None or self.hi is not None or self.step is not None:
                raise TrialError(f"gene {self.key}: give min and max, or choices, not both")
            if not self.choices:
                raise TrialError(f"gene {self.key}: choices is empty")
            if not all(isinstance(c, (str, bool)) or _is_number(c) for c in self.choices):
                raise TrialError(f"gene {self.key}: each choice must be a number, a string or true/false")
            if len(set(map(repr, self.choices))) < len(self.choices):
                raise TrialError(f"gene {self.key}: a choice is listed twice")
            if self.unit and not all(_is_number(c) for c in self.choices):
                raise TrialError(f"gene {self.key}: with a unit, the choices must be numbers")
        else:
            if not (_is_number(self.lo) and _is_number(self.hi)):
                raise TrialError(f"gene {self.key}: give min and max, or choices")
            if not self.lo < self.hi:
                raise TrialError(f"gene {self.key}: min must be below max")
            if self.step is not None and not (_is_number(self.step) and self.step > 0):
                raise TrialError(f"gene {self.key}: step must be a number above zero")
        if self.unit:
            try:
                design.ureg.Unit(self.unit)
            except Exception:
                raise TrialError(f"gene {self.key}: unknown unit {self.unit!r}") from None

    @property
    def whole(self) -> bool:
        """Whether the values are whole numbers: min and step are, so every step from min is."""
        return self.step is not None and isinstance(self.lo, int) and isinstance(self.step, int)

    def sample(self, rng: _Rng) -> Any:
        if self.choices is not None:
            return self.choices[rng.below(len(self.choices))]
        return self._snap(rng.uniform(self.lo, self.hi))

    def contains(self, value: Any) -> bool:
        """Whether the search could produce this value: one of the choices, or in range and on a step."""
        if self.choices is not None:
            return any(_same(c, value) for c in self.choices)
        if not (_is_number(value) and self.lo - 1e-9 <= value <= self.hi + 1e-9):
            return False
        if not self.step:
            return True
        steps = (value - self.lo) / self.step
        return abs(steps - round(steps)) <= 1e-9 and round(steps) <= self._last_step()

    def mutate(self, value: Any, rng: _Rng, sigma: float) -> Any:
        """A nearby value: another choice, or a step drawn from a normal `sigma` of the range wide."""
        if self.choices is not None:
            others = [c for c in self.choices if not _same(c, value)]
            return others[rng.below(len(others))] if others else value
        return self._snap(value + rng.normal(sigma * (self.hi - self.lo)))

    def override(self, value: Any) -> Any:
        """The value as a design override: with its unit, when the gene has one."""
        return f"{value!r} {self.unit}" if self.unit else value

    def ends(self) -> tuple:
        """The values that must fit the design for every value to fit: both ends, or every choice."""
        return self.choices if self.choices is not None else (self._snap(self.lo), self._snap(self.hi))

    def _last_step(self) -> int:
        """How many steps from min the last value at or below max is."""
        return math.floor((self.hi - self.lo) / self.step + 1e-9)

    def _snap(self, x: float) -> float | int:
        x = min(max(x, self.lo), self.hi)
        if not self.step:
            return float(x)
        x = self.lo + min(round((x - self.lo) / self.step), self._last_step()) * self.step
        return x if self.whole else float(f"{x:.12g}")  # 12 digits: drops the float noise of lo + n * step


@dataclass(frozen=True)
class Objective:
    """A metric to push down ("min") or up ("max"). It adds `weight * metric / scale` to the score."""

    metric: str
    goal: str
    weight: float = 1.0
    scale: float = 1.0  # the metric's "one unit of caring"

    def __post_init__(self) -> None:
        if self.goal not in ("min", "max"):
            raise TrialError(f'objective {self.metric}: goal must be "min" or "max", not {self.goal!r}')
        if not _is_number(self.weight) or not _is_number(self.scale) or self.scale <= 0:
            raise TrialError(f"objective {self.metric}: weight must be a number, and scale a number above zero")


@dataclass(frozen=True)
class Constraint:
    """A metric that must stay at or above `lo`, at or below `hi`, or both."""

    metric: str
    lo: float | None = None
    hi: float | None = None

    def __post_init__(self) -> None:
        if self.lo is None and self.hi is None:
            raise TrialError(f"constraint {self.metric}: give min, max or both")
        if not all(x is None or _is_number(x) for x in (self.lo, self.hi)):
            raise TrialError(f"constraint {self.metric}: min and max must be numbers")
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise TrialError(f"constraint {self.metric}: min is above max")


@dataclass
class Trial:
    name: str
    design: Path
    genes: list[Gene]
    objectives: list[Objective] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    question: str = ""
    population: int = 12
    generations: int = 8
    elite: int = 2
    seed: int = 1
    mutation: float = 0.2  # sigma as a fraction of each gene's range (it halves by the last generation)
    mutation_rate: float = 0.35  # the chance each gene mutates in a child
    crossover: float = 0.7  # the chance a child is a mix of two parents, not a copy of one
    path: Path | None = None  # the trial file, if it came from one

    def __post_init__(self) -> None:
        self.design = Path(self.design)
        if not isinstance(self.name, str) or not self.name:
            raise TrialError("a trial needs a name")
        if not self.genes:
            raise TrialError(f"trial {self.name}: no genes, so nothing to search")
        keys = [g.key for g in self.genes]
        if len(set(keys)) < len(keys):
            raise TrialError(f"trial {self.name}: a gene key is listed twice")
        for what in ("population", "generations", "elite", "seed"):
            if isinstance(getattr(self, what), bool) or not isinstance(getattr(self, what), int):
                raise TrialError(f"trial {self.name}: {what} must be a whole number")
        if self.population < 2 or self.generations < 1:
            raise TrialError(f"trial {self.name}: population must be at least 2, and generations at least 1")
        if not 0 <= self.elite < self.population:
            raise TrialError(f"trial {self.name}: elite must be from 0 to population - 1")
        for what in ("mutation_rate", "crossover"):
            if not (_is_number(getattr(self, what)) and 0 <= getattr(self, what) <= 1):
                raise TrialError(f"trial {self.name}: {what} is a chance, from 0 to 1")
        if not (_is_number(self.mutation) and self.mutation >= 0):
            raise TrialError(f"trial {self.name}: mutation must be a number, zero or above")

    def metrics(self) -> list[str]:
        """The metrics the objectives and constraints name."""
        return list(dict.fromkeys(x.metric for x in (*self.objectives, *self.constraints)))


@dataclass
class Result:
    """What `run` found, and where it wrote it."""

    trial: Trial
    individuals: list[dict]  # one record per candidate per generation it was in, each generation best first
    reference: dict  # the design as it stands
    best: dict
    history: Path
    best_toml: Path


def load(path: str | os.PathLike) -> Trial:
    """Read a trial file, and check that its genes fit its design."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise TrialError(f"{path}: {e}") from None
    try:
        if set(raw) != {"trial"} or not isinstance(raw["trial"], dict):
            raise TrialError("a trial file holds one [trial] table")
        d = raw["trial"]
        _known(d, _TRIAL_KEYS, "[trial]")
        for needed in ("name", "design"):
            if not isinstance(d.get(needed), str):
                raise TrialError(f"[trial] needs a {needed}")
        genes = []
        for g in _tables(d, "gene"):
            _known(g, _GENE_KEYS, "[[trial.gene]]")
            genes.append(Gene(g.get("key"), g.get("min"), g.get("max"), g.get("step"), g.get("choices"),
                              g.get("unit", ""), g.get("why", "")))  # fmt: skip
        objectives = []
        for o in _tables(d, "objective"):
            _known(o, _OBJECTIVE_KEYS, "[[trial.objective]]")
            if not isinstance(o.get("metric"), str) or "goal" not in o:
                raise TrialError("[[trial.objective]] needs a metric and a goal")
            objectives.append(Objective(o["metric"], o["goal"], o.get("weight", 1.0), o.get("scale", 1.0)))
        constraints = []
        for c in _tables(d, "constraint"):
            _known(c, _CONSTRAINT_KEYS, "[[trial.constraint]]")
            if not isinstance(c.get("metric"), str):
                raise TrialError("[[trial.constraint]] needs a metric")
            constraints.append(Constraint(c["metric"], c.get("min"), c.get("max")))
        settings = ("question", "population", "generations", "elite", "seed", "mutation", "mutation_rate", "crossover")
        trial = Trial(d["name"], path.parent / d["design"], genes, objectives, constraints, path=path,
                      **{k: d[k] for k in settings if k in d})  # fmt: skip
        _fit_to_design(trial)
    except TrialError as e:
        raise TrialError(f"{path}: {e}") from None
    return trial


def _known(table: dict, keys: set, where: str) -> None:
    unknown = sorted(set(table) - keys)
    if unknown:
        raise TrialError(f"{where} has no setting {', '.join(unknown)} (it takes: {', '.join(sorted(keys))})")


def _tables(d: dict, key: str) -> list[dict]:
    tables = d.get(key, [])
    if not isinstance(tables, list) or not all(isinstance(t, dict) for t in tables):
        raise TrialError(f"write each {key} as a [[trial.{key}]] table")
    return tables


# --- ranking ----------------------------------------------------------------------------------------------------


def violation(trial: Trial, rec: dict) -> float:
    """How far outside the constraints a candidate is; 0 when it meets them all."""
    if "error" in rec:
        return math.inf
    m, v = rec["metrics"], 0.0
    for c in trial.constraints:
        x = m[c.metric]
        if c.hi is not None and x > c.hi:
            v += x - c.hi
        if c.lo is not None and x < c.lo:
            v += c.lo - x
    return v


def score(trial: Trial, rec: dict) -> float:
    """The weighted sum of the objectives, each in units of its scale. Lower is better."""
    if "error" in rec:
        return math.inf
    m = rec["metrics"]
    return sum(o.weight * (1 if o.goal == "min" else -1) * m[o.metric] / o.scale for o in trial.objectives)


def rank_key(trial: Trial, rec: dict) -> tuple:
    """Sorts candidates best first: evaluated, then fewest FAILs, then least violation, then lowest score."""
    if "error" in rec:
        return (1, 0, 0.0, 0.0)
    return (0, rec.get("fails", 0), violation(trial, rec), score(trial, rec))


# --- the run ----------------------------------------------------------------------------------------------------


class _Rng:
    """Random draws built on `random.Random.random()` alone.

    That is the one stream Python promises to keep the same from version to version for a given seed; `choice`,
    `sample` and the rest may change. So a trial's seed means the same thing after an upgrade."""

    def __init__(self, seed: int) -> None:
        self.random = random.Random(seed).random

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.random()

    def below(self, n: int) -> int:
        """A whole number from 0 to n - 1."""
        return min(int(self.random() * n), n - 1)

    def normal(self, sigma: float) -> float:
        """A draw from a normal distribution about zero (Box–Muller)."""
        u, v = 1.0 - self.random(), self.random()  # u in (0, 1], so its log is finite
        return sigma * math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)

    def distinct(self, n: int, k: int) -> list[int]:
        """k different whole numbers from 0 to n - 1."""
        pool = list(range(n))
        for i in range(k):
            j = i + self.below(n - i)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]


def _key(genes: dict) -> str:
    return json.dumps(genes, sort_keys=True)


def _job(job: tuple) -> dict:
    return evaluate_candidate(*job)


def run(
    trial: Trial | str | os.PathLike,
    evaluate: Evaluate,
    *,
    workers: int | None = None,
    out: str | os.PathLike | None = None,
    log: Callable[[str], None] | None = None,
) -> Result:
    """Evolve the trial, and write `history.jsonl` and `best.toml` into `out` (`out/trials/<name>` by default).

    `evaluate(design) -> Evaluation` scores one candidate. With more than one worker it runs in other processes,
    so it must be a function at the top level of a module they can import. `workers=1` runs it in this process.
    The answer is the same either way."""
    if not isinstance(trial, Trial):
        trial = load(trial)
    baseline = _fit_to_design(trial)
    log = log or (lambda message: None)
    workers = max(1, (os.cpu_count() or 2) - 1) if workers is None else workers
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise TrialError(f"workers must be a whole number, 1 or more, not {workers!r}")
    if workers > 1:
        try:
            pickle.dumps(evaluate)
        except Exception:
            raise TrialError(
                "evaluate must be a function at the top level of a module, so worker processes can import it; "
                "or pass workers=1 to run it in this process"
            ) from None
    out_dir = Path(out) if out is not None else Path("out") / "trials" / trial.name
    out_dir.mkdir(parents=True, exist_ok=True)
    history = out_dir / "history.jsonl"

    rng = _Rng(trial.seed)
    cache: dict[str, dict] = {}
    seen: set[str] = set()
    # "spawn" everywhere: a worker starts clean and imports what it needs, whatever the platform's default is
    pool = (
        ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 1
        else None
    )

    def overrides(genes: dict) -> dict:
        return {g.key: g.override(genes[g.key]) for g in trial.genes}

    def evaluate_all(candidates: list[tuple[dict, dict]]) -> None:
        """Score the (genes, overrides) pairs not scored yet, each once, in the order given."""
        todo = [c for c in {_key(c[0]): c for c in candidates}.values() if _key(c[0]) not in cache]
        if not todo:
            return
        log(f"  evaluating {len(todo)} new candidate(s) on {workers} worker(s)")
        jobs = [(str(trial.design), given, evaluate) for _, given in todo]
        for (genes, given), res in zip(todo, pool.map(_job, jobs) if pool else map(_job, jobs)):
            if "error" not in res:
                missing = [m for m in trial.metrics() if m not in res["metrics"]]
                if missing:
                    raise TrialError(
                        f"trial {trial.name}: evaluate returned no metric {', '.join(missing)} "
                        f"(it returned: {', '.join(res['metrics']) or 'none'})"
                    )
            rec = {"genes": genes, "overrides": given, **res}
            judged = "error" not in rec
            rec["violation"] = violation(trial, rec) if judged else None
            rec["score"] = score(trial, rec) if judged else None
            cache[_key(genes)] = rec

    def child(a: dict, b: dict, gen: int, made: set[str]) -> dict:
        sigma = trial.mutation * (1 - 0.5 * gen / max(1, trial.generations - 1))
        for _ in range(8):
            mix = rng.random() < trial.crossover
            c = {g.key: b[g.key] if mix and rng.random() < 0.5 else a[g.key] for g in trial.genes}
            for g in trial.genes:
                if rng.random() < trial.mutation_rate:
                    c[g.key] = g.mutate(c[g.key], rng, sigma)
            if _key(c) not in seen and _key(c) not in made:
                break
            g = trial.genes[rng.below(len(trial.genes))]  # a repeat: nudge one gene and try again
            a = {**c, g.key: g.mutate(c[g.key], rng, sigma)}
        return c

    try:
        with open(history, "w", encoding="utf-8") as f:

            def record(rec: dict) -> None:
                f.write(json.dumps(rec, allow_nan=False) + "\n")
                f.flush()

            # The design as it stands starts the population when every gene could produce its value: in range,
            # and on a step. A trial that looks elsewhere still scores it, for comparison.
            in_range = all(g.contains(baseline[g.key]) for g in trial.genes)
            log("reference: the design as it stands" + ("" if in_range else " (outside the genes: compared, not bred)"))
            evaluate_all([(baseline, overrides(baseline) if in_range else {})])
            reference = {"generation": None, "rank": None, **cache[_key(baseline)]}
            record(reference)

            pop = [baseline] if in_range else []
            pop += [{g.key: g.sample(rng) for g in trial.genes} for _ in range(trial.population - len(pop))]
            individuals: list[dict] = []
            for gen in range(trial.generations):
                log(f"generation {gen + 1}/{trial.generations}")
                evaluate_all([(genes, overrides(genes)) for genes in pop])
                recs = sorted((cache[_key(genes)] for genes in pop), key=lambda r: rank_key(trial, r))
                recs = [{"generation": gen, "rank": i, **r} for i, r in enumerate(recs)]
                seen.update(_key(genes) for genes in pop)
                for rec in recs:
                    record(rec)
                individuals += recs
                top = recs[0]
                log(
                    f"  best: score {top['score']}, {top.get('fails', 0)} FAIL, violation {top['violation']}: "
                    + ", ".join(f"{k}={v}" for k, v in top["genes"].items())
                )
                if gen == trial.generations - 1:
                    break

                def pick() -> dict:  # a tournament of three; recs are sorted, so the lowest index is the best
                    return recs[min(rng.distinct(len(recs), min(3, len(recs))))]["genes"]

                nxt = [r["genes"] for r in recs[: trial.elite]]
                made = {_key(genes) for genes in nxt}
                while len(nxt) < trial.population:
                    nxt.append(child(pick(), pick(), gen + 1, made))
                    made.add(_key(nxt[-1]))
                pop = nxt
    finally:
        if pool:
            pool.shutdown()

    best = min(individuals, key=lambda r: rank_key(trial, r))  # the first of equals: the earliest found
    best_path = out_dir / "best.toml"
    best_path.write_text(best_toml(trial, best, reference, out_dir), encoding="utf-8")
    return Result(trial, individuals, reference, best, history, best_path)


# --- the design the trial searches ------------------------------------------------------------------------------


def _fit_to_design(trial: Trial) -> dict:
    """Check that every gene names a design value it can set, and return the design's own value for each gene.

    A value the gene cannot express (a quantity where the gene has no unit, say) comes back as None."""
    try:
        base = design.load(trial.design)
    except (OSError, design.DesignError) as e:
        raise TrialError(f"trial {trial.name}: design {trial.design.name}: {e}") from None
    baseline = {}
    for g in trial.genes:
        try:
            current = _at(base, g.key)
            loaded = [_at(design.load(trial.design, {g.key: g.override(v)}), g.key) for v in g.ends()]
        except design.DesignError as e:
            raise TrialError(f"trial {trial.name}: gene {g.key}: {e}") from None
        if g.choices is not None:
            same = [c for c, value in zip(g.choices, loaded) if type(value) is type(current) and value == current]
            baseline[g.key] = same[0] if same else None
            continue
        if isinstance(current, pint.Quantity):
            # 12 digits: "0.05 m" is 50 mm, not the 50.00000000000001 the conversion gives
            current = float(f"{current.to(g.unit).magnitude:.12g}") if g.unit else None
        if not _is_number(current):
            baseline[g.key] = None
        elif g.contains(current):
            baseline[g.key] = g._snap(current)  # exactly one of the gene's own values
        else:
            baseline[g.key] = current  # out of range or off a step: `run` compares it, and does not breed it
    return baseline


def _at(tree: Any, dotted: str) -> Any:
    """The value at a dotted key of a loaded design."""
    node = tree
    try:
        for part in dotted.split("."):
            if isinstance(node, design.Design):
                node = node[part]
            elif isinstance(node, list):
                node = node[int(part)]
            elif isinstance(node, design.Uncertain) and part in ("ref", "lo", "hi"):
                node = getattr(node, part)
            else:
                raise KeyError(part)
    except (LookupError, ValueError):
        raise design.UnknownKeyError(f"{dotted}: no such design value") from None
    return node


# --- the best design --------------------------------------------------------------------------------------------


def best_toml(trial: Trial, best: dict, reference: dict, out_dir: Path) -> str:
    """The best candidate as a design file: the trial's design, included, plus the genes as `[overrides]`.

    The design is included by a relative path when this file is written inside the design's own project, so
    the file can be loaded as it is and the path never leaves the project. Written anywhere else, a path to the
    design would spell out where it lives on this machine, so the file then holds the `[overrides]` alone, and
    says which design they are for."""
    include = _relative(trial.design, out_dir)
    name, design_name = _one_line(trial.name), _one_line(trial.design.name)
    lines = [f"# Trial {name}: the best candidate, as [overrides] of {design_name}."]
    if include:
        lines.append("# Load it with similitude.design.load: it includes that design.")
    else:
        lines.append(f"# It does not include {design_name}, which is not in a project this file is in: load that")
        lines.append("# design with these overrides, or copy them into a design file that includes it.")
    lines.append("# Review before adopting: a trial is evidence, not a decision.")
    include = [f"include = {json.dumps(include)}", ""] if include else []
    if "error" in best:
        lines += ["# No candidate could be evaluated, so nothing is overridden.", *_comment(best["error"]), ""]
        return "\n".join(lines + include).rstrip() + "\n"
    was = "it could not be evaluated" if "error" in reference else f"{reference['score']:.6g}"
    lines.append(f"# Score {best['score']:.6g} (the design as it stands: {was}; lower is better).")
    if best["fails"]:
        lines.append(f"# It FAILS {best['fails']} check(s): no candidate without a FAIL was found.")
    if best["violation"]:
        lines.append("# It breaks a constraint: no candidate that meets them all was found.")
    lines += ["", *include, "[overrides]"]
    rows = [f"{json.dumps(key)} = {_toml(value)}" for key, value in best["overrides"].items()]
    width = max(len(row) for row in rows)
    for row, key in zip(rows, best["genes"]):
        was = reference["genes"][key]
        was = "another value" if was is None else _toml(was)  # a string is quoted, so it stays on one line
        note = "" if reference["genes"][key] == best["genes"][key] else f"  # the design has {was}"
        lines.append(f"{row:<{width}}{note}".rstrip())
    return "\n".join(lines) + "\n"


_PROJECT_MARKERS = (".git", "pyproject.toml")


def _project(path: Path) -> Path | None:
    """The project a file belongs to: its nearest folder, going up, that holds a `.git` or a `pyproject.toml`."""
    for folder in path.resolve().parents:
        if any((folder / marker).exists() for marker in _PROJECT_MARKERS):
            return folder
    return None


def _relative(target: Path, start: Path) -> str:
    """The path to the file `target` from the folder `start`, when `start` is inside `target`'s project.

    Otherwise "": two folders that merely share an ancestor, such as a home or a temporary directory, are not
    one project, and a path between them would name folders that say whose machine this is."""
    target, start = target.resolve(), start.resolve()
    project = _project(target)
    if project is None or not start.is_relative_to(project):
        return ""
    return Path(os.path.relpath(target, start)).as_posix()


def _one_line(text: str) -> str:
    """Free text made safe for one comment line."""
    return _printable(" ".join(str(text).split()))


def _comment(text: str) -> list[str]:
    """Free text as TOML comment lines, however many lines it has."""
    return [f"# {_printable(line)}".rstrip() for line in str(text).splitlines()] or ["#"]


def _printable(line: str) -> str:
    """A line with its control characters, which TOML does not allow in a comment, written out as \\x1b."""
    return "".join(c if c == "\t" or (c >= " " and c != "\x7f") else f"\\x{ord(c):02x}" for c in line)


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a valid TOML basic string
    return repr(value)
