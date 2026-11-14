"""The command line: a thin layer over the core, so CI and agents can drive it.

    similitude check designs/can.toml --set body.wall="0.25 mm"
    similitude check designs/can.toml --set body.wall="0.25 mm" --out out/can
    similitude keep out/can thicker-wall --shows "a 0.25 mm wall passes"
    similitude trial trials/lighter-can.toml
    similitude keep out/trials/lighter-can lighter-can

**A project supplies the model.** It registers a function in its own `pyproject.toml`:

    [project.entry-points."similitude.models"]
    can = "cannery.model:evaluate"

- The function is `evaluate(design) -> similitude.trials.Evaluation`: the design's named metrics and every check
  result. `check` and `trial` call the same one, so a trial cannot score a design differently from the way it is
  checked.
- The project must be installed (`uv sync`, or `pip install -e .`) for the entry point to be found.
- With one model registered, the commands use it. With several, name one with `--model`.

**Exit codes**, the same for every command:

- 0: the command ran, and no check is a FAIL;
- 1: the command ran, and the design fails. For `check`, a check is a FAIL (`similitude.checks.exit_code`). For
  `trial`, the best candidate found has a FAIL, breaks a constraint, or could not be evaluated;
- 2: the command could not run: a bad argument, an unreadable design or trial, no model, a model that raised, a
  run that may not be kept.

**`--set key=value`** overrides one design value, by the dotted key a design's `[overrides]` uses. The value is
read as TOML when it is TOML (`3`, `0.25`, `true`, `[1.0, 2.0]`, `{ref = 9.0, range = [7.0, 12.0]}`), and
otherwise as text, so `body.wall="0.25 mm"` reaches the design as `0.25 mm`. To pass text that is also TOML,
quote it twice: `--set 'label="3"'`.

**Output directories.** `check --out DIR` and `trial` write a run into a directory, in the form `keep` reads:

- `START`: the commit, branch and command line, recorded with `similitude.runs.start` before the model runs, so
  `keep` can say what made the run. Outside a git checkout nothing can be recorded; the command says so and
  carries on.
- `design-files/`: the design file and every file it includes, copied as they were before the model runs. The
  command stops unless the copy loads as the same design as the files it came from.
- `design.toml`: the design the run used. It includes the copy in `design-files/` and sets the `--set` values,
  or the trial's best candidate, as `[overrides]`. The command stops unless it loads as the copy with those
  values. `check` evaluates the design loaded from it, and a trial searches the copied files.
- `checks.json` and `checks.md`: the checks. For a trial they are the best candidate's, followed by one check
  per trial constraint, so the trial's exit code is that of the checks it wrote. If no candidate could be
  evaluated, the one check written is a FAIL that says so, and `design.toml` is the best-ranked candidate as it
  was tried, which may not load.

Because the run holds its own design, `keep` takes no `--design` for it, and refuses one. A trial's best
candidate with a WARN is kept only once `checks.json` records its disposition.

- A directory that already has a `START`, a `checks.json` or a `design.toml` is refused: remove it, or name
  another.
- A design that includes a file by an absolute path cannot be copied, so it cannot go into a run.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import shlex
import shutil
import sys
import tomllib
import traceback
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from similitude import __version__, checks, design, runs, trials

GROUP = "similitude.models"
OK, FAILED, ERROR = 0, 1, 2
NOT_RECORDED = "(not recorded)"  # the command that made a run with no START
FILES = "design-files"  # in an output directory: the design file and its includes, as they were
CHECKS_MD = "checks.md"

_REGISTER = (
    f'no model is registered; a project names one in its pyproject.toml under [project.entry-points."{GROUP}"], '
    'as `name = "package.module:evaluate"`, and must be installed'
)


class CliError(Exception):
    """The command could not run. The message is for whoever typed it."""


# --- models -----------------------------------------------------------------------------------------------------


def models() -> dict[str, metadata.EntryPoint]:
    """The registered models, by name."""
    found: dict[str, metadata.EntryPoint] = {}
    for entry in metadata.entry_points(group=GROUP):
        if entry.name in found and found[entry.name].value != entry.value:
            raise CliError(f"model {entry.name!r} is registered twice: {found[entry.name].value} and {entry.value}")
        found[entry.name] = entry
    return dict(sorted(found.items()))


def load_model(name: str | None = None) -> trials.Evaluate:
    """A registered model's `evaluate`: the one named, or the only one there is."""
    registered = models()
    if name is None:
        if not registered:
            raise CliError(_REGISTER)
        if len(registered) > 1:
            raise CliError(f"{len(registered)} models are registered; name one with --model: {', '.join(registered)}")
        name = next(iter(registered))
    elif name not in registered:
        known = f"the registered models are: {', '.join(registered)}" if registered else _REGISTER
        raise CliError(f"no model named {name!r}; {known}")
    entry = registered[name]
    try:
        model = entry.load()
    except Exception as e:
        raise CliError(f"model {name} ({entry.value}) could not be loaded: {type(e).__name__}: {e}") from e
    if not callable(model):
        raise CliError(f"model {name} ({entry.value}) is a {type(model).__name__}, not a function")
    return model


# --- arguments --------------------------------------------------------------------------------------------------


def parse_sets(pairs: Sequence[str]) -> dict[str, Any]:
    """`--set key=value` arguments as design overrides."""
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key, equals, text = pair.partition("=")
        key = key.strip()
        if not equals or not key:
            raise CliError(f"--set takes key=value, not {pair!r}")
        if key in overrides:
            raise CliError(f"--set gives {key} twice")
        overrides[key] = _value(text)
    return overrides


def _value(text: str) -> Any:
    """A TOML value if `text` is one, and otherwise the text itself."""
    try:
        parsed = tomllib.loads(f"v = {text}")
    except tomllib.TOMLDecodeError:
        return text
    return parsed["v"] if set(parsed) == {"v"} else text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="similitude",
        description="Check a design, run a trial, or keep a run.",
        epilog="Exit codes: 0, ran with no FAIL; 1, the design fails; 2, the command could not run.",
    )
    parser.add_argument("--version", action="version", version=f"similitude {__version__}")
    commands = parser.add_subparsers(dest="subcommand", required=True, metavar="command")
    model = {"metavar": "NAME", "help": "the registered model to use; needed when there is more than one"}

    check = commands.add_parser(
        "check",
        help="load a design, run the model's checks on it, and exit 1 if any is a FAIL",
        description="Load a design, run the project's model on it and print its checks. Exits 1 if any is a FAIL.",
    )
    check.add_argument("design", help="the design file (TOML)")
    check.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help='override one design value by its dotted key, as in body.wall="0.25 mm"; may be repeated',
    )
    check.add_argument("--model", **model)
    check.add_argument(
        "--out", metavar="DIR", help="also write the run into this fresh directory: START, the design and the checks"
    )
    check.add_argument("--json", action="store_true", help="print the checks as JSON, not as a Markdown table")
    check.set_defaults(run=_check)

    trial = commands.add_parser(
        "trial",
        help="run a trial: search a design's values, and exit 1 if the best one found fails",
        description=(
            "Run a trial file's search with the project's model. Writes the search's history and the best "
            "candidate's design and checks, and exits 1 if the best candidate has a FAIL, breaks a constraint or "
            "could not be evaluated."
        ),
    )
    trial.add_argument("trial", help="the trial file (TOML)")
    trial.add_argument("--model", **model)
    trial.add_argument("--out", metavar="DIR", help="the fresh directory to write into; out/trials/<name> by default")
    trial.add_argument(
        "--workers", type=int, metavar="N", help="processes to score candidates in; one fewer than the CPUs by default"
    )
    trial.set_defaults(run=_trial)

    keep = commands.add_parser(
        "keep",
        help="copy an output directory to runs/<date>_<name>/ with where it came from",
        description="Keep a run: copy its output directory to runs/<date>_<name>/ and record its source.",
    )
    keep.add_argument("dir", help="the run's output directory")
    keep.add_argument("name", help="the kept run's name: letters, digits, '.', '_' and '-'")
    keep.add_argument("--design", metavar="FILE", help="the design the run used, for a directory with no design.toml")
    keep.add_argument("--shows", default="", metavar="TEXT", help="one line for runs/README.md: what the run shows")
    keep.add_argument(
        "--command", metavar="TEXT", help="for a run with no START: the command line that made it, in quotes"
    )
    keep.add_argument(
        "--allow-dirty",
        action="store_true",
        help="keep a run made with uncommitted changes, and record that it was",
    )
    keep.set_defaults(run=_keep)
    return parser


# --- commands ---------------------------------------------------------------------------------------------------


def _start(out: Path, command: str) -> None:
    """Record where a run starts from. Outside a git checkout there is nothing to record: say so and carry on."""
    for holds in (runs.DESIGN, FILES):
        if (out / holds).exists():
            raise CliError(f"{out} already has a {holds}; start a run in a fresh output directory")
    try:
        runs.start(out, command=command)
    except RuntimeError as e:
        print(f"similitude: no {runs.START} recorded, so keep cannot say what made this run: {e}", file=sys.stderr)


def _design_files(path: Path) -> list[Path]:
    """A design file and every file it includes, the design file first."""
    files: list[Path] = []
    todo = [path.resolve()]
    while todo:
        file = todo.pop()
        if file in files:
            continue
        files.append(file)
        with open(file, "rb") as f:
            includes = tomllib.load(f).get("include", [])
        for name in [includes] if isinstance(includes, str) else includes:
            if Path(name).is_absolute():
                raise CliError(
                    f"{file.name} includes {name} by an absolute path, so the design cannot be copied into a run; "
                    "include it by a relative path"
                )
            todo.append((file.parent / name).resolve())
    return files


def _copy_design(path: Path, out: Path) -> Path:
    """Copy a design file and its includes into `out/design-files/`, laid out as they are, and return the copy
    of the design file. The copies include one another by the same relative paths as the files they came from."""
    files = _design_files(path)
    above = Path(os.path.commonpath([file.parent for file in files]))
    for file in files:
        copy = out / FILES / file.relative_to(above)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, copy)
    copy = out / FILES / files[0].relative_to(above)
    if design.load(copy) != design.load(path):
        raise CliError(
            f"{copy} does not load as the same design as {path}, so the run stops here. "
            "A design with a NaN in it never compares equal to itself"
        )
    return copy


def _write_design(out: Path, copy: Path, overrides: Mapping[str, Any], *, loads: bool = True) -> design.Design | None:
    """Write `out/design.toml`, the design a run used, and return it as loaded from there.

    The file includes the copied design and sets `overrides`. It must load as the copy does with those overrides:
    a run is not given a design.toml that is some other design. `loads=False` is for a candidate that could not
    be evaluated, which is written as it was tried and not loaded."""
    lines = [
        "# The design this run used: the design file as it was, in design-files/, with the values the run set.",
        "# Written by the similitude command; never edit it.",
        f"include = {_toml(copy.relative_to(out).as_posix())}",
    ]
    if overrides:
        lines += ["", "[overrides]", *(f"{_toml(str(key))} = {_toml(value)}" for key, value in overrides.items())]
    file = out / runs.DESIGN
    file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not loads:
        return None
    written = design.load(file)
    if written != design.load(copy, overrides):
        raise CliError(f"{file} does not load as the design this run was asked for, so the run stops here")
    return written


def _toml(value: Any) -> str:
    """One override's value, as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_toml(str(key))} = {_toml(item)}" for key, item in value.items()) + "}"
    raise CliError(f"an override of type {type(value).__name__} cannot be written into a run's design.toml")


def _table(found: Sequence[checks.Check]) -> str:
    return checks.to_markdown(found) + "\n" + checks.summary(found) + "\n"


def _write_checks(out: Path, found: Sequence[checks.Check]) -> None:
    """Write a run's `checks.json` and `checks.md`."""
    (out / runs.RESULTS).write_text(checks.to_json(found), encoding="utf-8")
    (out / CHECKS_MD).write_text(_table(found), encoding="utf-8")


def _check(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    overrides = parse_sets(args.set)
    loaded = design.load(args.design, overrides)
    out = None if args.out is None else Path(args.out)
    if out is not None:
        _start(out, args.invoked)
        loaded = _write_design(out, _copy_design(Path(args.design), out), overrides)
    try:
        result = model(loaded)
    except Exception as e:
        traceback.print_exc()
        raise CliError(f"the model raised on this design: {type(e).__name__}: {e}") from e
    if not isinstance(result, trials.Evaluation):
        raise CliError(f"the model must return a similitude.trials.Evaluation, not {type(result).__name__}")
    found = list(result.checks)
    for check in found:
        if not isinstance(check, checks.Check):
            raise CliError(f"the model's checks must be similitude.checks.Check records, not {type(check).__name__}")

    table = _table(found)
    if out is not None:
        _write_checks(out, found)
    sys.stdout.write(checks.to_json(found) if args.json else table)
    return checks.exit_code(found)


def _trial_checks(trial: trials.Trial, best: Mapping[str, Any]) -> list[checks.Check]:
    """A trial's result as checks: the best candidate's own, then one for each bound of each constraint.

    A FAIL among them is exactly what makes the trial fail: a FAIL of the candidate's, a constraint it breaks,
    or, when no candidate could be evaluated, the one check that says so."""
    if "error" in best:
        reason = " ".join(str(best["error"]).split())
        return [checks.Check("Trial", "the best candidate could be evaluated", 0, "", limit=1, op=">=", reason=reason)]
    found = [checks.Check.from_dict(check) for check in best["checks"]]
    for constraint in trial.constraints:
        value = best["metrics"][constraint.metric]
        for bound, op, which in ((constraint.lo, ">=", "min"), (constraint.hi, "<=", "max")):
            if bound is not None:
                found.append(
                    checks.Check("Trial constraint", f"{constraint.metric} ({which})", value, "", limit=bound, op=op)
                )
    return found


def _trial(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    trial = trials.load(args.trial)
    out = Path(args.out) if args.out is not None else Path("out") / "trials" / trial.name
    _start(out, args.invoked)
    trial = dataclasses.replace(trial, design=_copy_design(trial.design, out))  # search the copy the run keeps
    result = trials.run(
        trial, model, workers=args.workers, out=out, log=lambda message: print(message, file=sys.stderr)
    )
    best = result.best
    _write_design(out, trial.design, best["overrides"], loads="error" not in best)
    found = _trial_checks(trial, best)
    _write_checks(out, found)

    print("best: " + ", ".join(f"{key}={value}" for key, value in best["genes"].items()))
    if "error" in best:
        print(f"  could not be evaluated: {best['error']}")
    else:
        for name, value in best["metrics"].items():
            print(f"  {name} = {value:g}")
        print(f"  {best['fails']} FAIL, {best['warns']} WARN, constraint violation {best['violation']:g}")
    print(f"history: {result.history}")
    print(f"best design: {result.best_toml}")
    return checks.exit_code(found)


def _keep(args: argparse.Namespace) -> int:
    src = Path(args.dir)
    command = args.command
    if command is None and src.is_dir() and not (src / runs.START).exists():
        command = NOT_RECORDED  # runs.keep would otherwise record this keep command as the one that made the run
    try:
        kept = runs.keep(
            src, args.name, design=args.design, command=command, shows=args.shows, allow_dirty=args.allow_dirty
        )
    except (ValueError, RuntimeError, OSError) as e:
        raise CliError(_as_flags(str(e))) from e
    print(os.path.relpath(kept))
    return OK


def _as_flags(message: str) -> str:
    """A message from `similitude.runs`, with its keyword arguments named as this command's options."""
    for argument, flag in (("allow_dirty=True", "--allow-dirty"), ("command=", "--command"), ("design=", "--design")):
        message = message.replace(argument, flag)
    return message


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command, and return its exit code. A bad command line exits 2 through argparse."""
    args = _parser().parse_args(argv)
    # The command line as typed, for START: without the path this program was found at.
    args.invoked = shlex.join(["similitude", *(sys.argv[1:] if argv is None else argv)])
    try:
        return args.run(args)
    except (CliError, design.DesignError, trials.TrialError, OSError) as e:
        print(f"similitude: error: {e}", file=sys.stderr)
    except Exception as e:  # a bug, here or in the model: show where, and still exit 2, never 1
        traceback.print_exc()
        print(f"similitude: error: {type(e).__name__}: {e}", file=sys.stderr)
    return ERROR


if __name__ == "__main__":
    sys.exit(main())
