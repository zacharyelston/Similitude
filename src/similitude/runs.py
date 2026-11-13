"""Kept runs: an output directory copied to `runs/<date>_<name>/` with where it came from.

`out/` is scratch. A run worth keeping is copied with `keep`, which adds:

- `SOURCE`: the commit, the branch and the command line (TOML; read it with `read_source`);
- `design.toml`: the resolved design the run used;
- `KEPT.md`: a report of the source and of each check declared before the run beside its result;
- a row in `runs/README.md`.

A kept run is never edited or overwritten; make a new one.

A run declares its pass/fail checks before it runs, in `expect.toml` in its output directory:

    [[check]]
    group = "Float"          # optional
    name = "Freeboard"
    op = ">="                # or "<="
    limit = 0.15
    unit = "m"               # optional
    reason = "deck stays dry in a 0.1 m chop"   # optional

The results are the run's `checks.json`, as `similitude.checks.to_json` writes it. Each declared check is judged
against the limit written down first, whatever limit the run itself used.

Git is called as a separate program.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

RUNS_DIR = "runs"
SOURCE = "SOURCE"
DESIGN = "design.toml"
EXPECT = "expect.toml"
RESULTS = "checks.json"
REPORT = "KEPT.md"
README = "README.md"

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_OPS = ("<=", ">=")
_EXPECT_KEYS = {"group", "name", "op", "limit", "unit", "reason"}
_README_HEAD = """\
# Kept runs

Each directory is a copy of an output directory worth keeping, named `<date>_<name>`. Its `SOURCE` file records
the commit, branch and command it came from, and `KEPT.md` shows the checks declared before the run beside
their results. Never edit a kept run; make a new one.

| Run | Commit | Branch | Expected checks | What it shows |
|---|---|---|---|---|
"""


class DirtyTreeError(RuntimeError):
    """The working tree has changes that no commit records."""


class UndisposedWarningError(RuntimeError):
    """The run has a WARN with no disposition recorded."""


@dataclass(frozen=True)
class Expectation:
    """A check declared before the run: the result's value must satisfy `value <op> limit`."""

    name: str
    op: str
    limit: float
    group: str = ""
    unit: str = ""
    reason: str = ""

    @property
    def label(self) -> str:
        return f"{self.group}: {self.name}" if self.group else self.name


@dataclass(frozen=True)
class Outcome:
    """A declared check beside its result. The verdict is PASS, FAIL or MISSING (the run reported no such check)."""

    expected: Expectation
    verdict: str
    value: float | str | None = None
    unit: str = ""
    note: str = ""


def keep(
    src_dir: str | os.PathLike,
    name: str,
    *,
    repo: str | os.PathLike | None = None,
    design: str | os.PathLike | None = None,
    command: str | None = None,
    shows: str = "",
    allow_dirty: bool = False,
    date: datetime.date | None = None,
) -> Path:
    """Copy the output directory `src_dir` to `runs/<date>_<name>/` and return the new directory.

    - `repo`: the git repository that made the run and holds `runs/`; by default the one around the current
      directory.
    - `design`: the resolved design file, copied in as `design.toml`. Leave it out when the run already wrote
      `design.toml` into `src_dir`. A run with no design is refused.
    - `command`: the command line that made the run; by default this process's own. The repository's path is
      written as `.` and the home directory as `~`.
    - `shows`: one line for the `runs/README.md` row, saying what the run shows.
    - `allow_dirty`: keep the run although the working tree has uncommitted changes, and record that it did.
      Untracked files count, so keep output directories ignored, as `out/` is. Changes under `runs/` never count.
    - `date`: the date in the directory's name; by default today.

    Raises FileExistsError rather than overwrite a kept run, DirtyTreeError for a dirty tree, and
    UndisposedWarningError if `checks.json` holds a WARN with no disposition. Nothing is written unless the whole
    keep succeeds.
    """
    src = Path(src_dir).resolve()
    if not src.is_dir():
        raise NotADirectoryError(f"{src_dir} is not an output directory")
    if not _NAME.fullmatch(name):
        raise ValueError(f"a run's name is letters, digits, '.', '_' and '-', not {name!r}")
    root = Path(_git(Path(repo) if repo is not None else Path.cwd(), "rev-parse", "--show-toplevel")).resolve()
    run = f"{(date or datetime.date.today()).isoformat()}_{name}"
    dest = root / RUNS_DIR / run
    if dest.exists():
        raise FileExistsError(f"{RUNS_DIR}/{run} is already kept; a kept run is never overwritten")

    for reserved in (SOURCE, REPORT):
        if (src / reserved).exists():
            raise ValueError(f"{src_dir} already has a {reserved}; is it a kept run?")
    design_path = _find_design(src, design)
    declared = _declared(src)
    results = _results(src)
    undisposed = [r for r in results if r.get("status") == "WARN" and not str(r.get("disposition", "")).strip()]
    if undisposed:
        names = ", ".join(_label(r) for r in undisposed)
        raise UndisposedWarningError(f"record a disposition in {RESULTS} for each WARN before keeping: {names}")
    outcomes = [_judge(e, results) for e in declared]

    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "symbolic-ref", "--short", "-q", "HEAD", check=False) or "(detached)"
    dirty_files = _git(root, "status", "--porcelain", "--untracked-files=all", "--", ".", f":(exclude){RUNS_DIR}")
    dirty_files = [line.strip() for line in dirty_files.splitlines()]
    if dirty_files and not allow_dirty:
        raise DirtyTreeError(
            f"the working tree has uncommitted changes ({len(dirty_files)} paths, first: {dirty_files[0]}); "
            "commit them, or pass allow_dirty=True to keep the run and record that it was dirty"
        )
    source = {
        "run": run,
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty_files),
        "command": _scrub(command if command is not None else shlex.join(sys.argv), root),
    }
    if dirty_files:
        source["dirty_files"] = dirty_files

    tmp = dest.with_name(f".{run}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        shutil.copytree(src, tmp)
        if design_path != src / DESIGN:
            shutil.copyfile(design_path, tmp / DESIGN)
        (tmp / SOURCE).write_text(_source_text(source), encoding="utf-8")
        (tmp / REPORT).write_text(_report(source, outcomes), encoding="utf-8")
        os.rename(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _add_readme_row(root / RUNS_DIR / README, source, outcomes, shows)
    return dest


def read_source(run_dir: str | os.PathLike) -> dict:
    """A kept run's `SOURCE`: `run`, `commit`, `branch`, `dirty`, `command`, and `dirty_files` if it was dirty."""
    return tomllib.loads((Path(run_dir) / SOURCE).read_text(encoding="utf-8"))


def expectations(run_dir: str | os.PathLike) -> list[Outcome]:
    """Each check declared in the run's `expect.toml` beside its result from `checks.json`, in declared order."""
    run_dir = Path(run_dir)
    results = _results(run_dir)
    return [_judge(e, results) for e in _declared(run_dir)]


def report(run_dir: str | os.PathLike) -> str:
    """A kept run's report, as Markdown: its source, and each expected check beside its result."""
    run_dir = Path(run_dir)
    return _report(read_source(run_dir), expectations(run_dir))


# --- git


def _git(cwd: Path, *args: str, check: bool = True) -> str:
    try:
        done = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("git is needed to keep a run, and isn't on the PATH") from None
    if check and done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}: {done.stderr.strip()}")
    return done.stdout.rstrip("\n")


# --- the design


def _find_design(src: Path, design: str | os.PathLike | None) -> Path:
    in_run = src / DESIGN
    if design is None:
        if not in_run.is_file():
            raise FileNotFoundError(
                f"a kept run needs its resolved design: write {DESIGN} into {src.name}/ or pass design="
            )
        path = in_run
    else:
        path = Path(design).resolve()
        if in_run.exists() and path != in_run:
            raise ValueError(f"{src.name}/ already has a {DESIGN}, and design= names another file")
        if not path.is_file():
            raise FileNotFoundError(f"no design file at {design}")
    try:
        tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as err:
        raise ValueError(f"the design {path.name} is not TOML: {err}") from None
    return path


# --- expectations and results


def _declared(run_dir: Path) -> list[Expectation]:
    path = run_dir / EXPECT
    if not path.exists():
        return []
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as err:
        raise ValueError(f"{EXPECT} is not TOML: {err}") from None
    if set(doc) - {"check"}:
        raise ValueError(f"{EXPECT} holds [[check]] tables only, not {sorted(set(doc) - {'check'})}")
    declared = []
    for i, entry in enumerate(doc.get("check", []), start=1):
        where = f"{EXPECT} check {i}"
        if set(entry) - _EXPECT_KEYS:
            raise ValueError(f"{where}: unknown keys {sorted(set(entry) - _EXPECT_KEYS)}")
        for key in ("name", "op", "limit"):
            if key not in entry:
                raise ValueError(f"{where}: needs a {key}")
        if entry["op"] not in _OPS:
            raise ValueError(f"{where}: op must be one of {_OPS}, not {entry['op']!r}")
        if isinstance(entry["limit"], bool) or not isinstance(entry["limit"], (int, float)):
            raise ValueError(f"{where}: the limit must be a number, not {entry['limit']!r}")
        for key in ("name", "group", "unit", "reason"):
            if not isinstance(entry.get(key, ""), str):
                raise ValueError(f"{where}: {key} must be text")
        declared.append(Expectation(**entry))
    labels = [e.label for e in declared]
    repeated = sorted({x for x in labels if labels.count(x) > 1})
    if repeated:
        raise ValueError(f"{EXPECT} declares these more than once: {', '.join(repeated)}")
    return declared


def _results(run_dir: Path) -> list[dict]:
    path = run_dir / RESULTS
    if not path.exists():
        return []
    try:
        results = json.loads(path.read_text(encoding="utf-8"))["checks"]
        if not all(isinstance(r, dict) and isinstance(r["name"], str) for r in results):
            raise TypeError
    except (json.JSONDecodeError, KeyError, TypeError):
        raise ValueError(f"{RESULTS} is not the JSON that similitude.checks.to_json writes") from None
    return results


def _judge(expected: Expectation, results: list[dict]) -> Outcome:
    matches = [
        r for r in results if r["name"] == expected.name and (not expected.group or r.get("group") == expected.group)
    ]
    if not matches:
        return Outcome(expected, "MISSING", note="the run reported no check with this name")
    if len(matches) > 1:
        groups = ", ".join(str(r.get("group", "")) for r in matches)
        raise ValueError(f"{EXPECT}: {expected.name!r} matches results in several groups ({groups}); give its group")
    result = matches[0]
    unit = str(result.get("unit", ""))
    value = _number(result.get("value"))
    if value is None:
        return Outcome(expected, "FAIL", result.get("value"), unit, "the result is not a number")
    if expected.unit and unit and expected.unit != unit:
        return Outcome(expected, "FAIL", value, unit, f"declared in {expected.unit} but reported in {unit}")
    holds = value <= expected.limit if expected.op == "<=" else value >= expected.limit
    note = ""
    own_limit = _number(result.get("limit"))
    if own_limit is not None and (result.get("op"), own_limit) != (expected.op, float(expected.limit)):
        note = f"the run's own check used {result.get('op')} {own_limit:g}"
    return Outcome(expected, "PASS" if holds else "FAIL", value, unit, note)


def _number(x) -> float | None:
    """A number from `checks.json`, where NaN and the infinities are written as strings."""
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": float("-inf")}.get(x)


def _label(result: dict) -> str:
    group = result.get("group", "")
    return f"{group}: {result['name']}" if group else result["name"]


def _tally(outcomes: list[Outcome]) -> str:
    if not outcomes:
        return "none declared"
    return f"{sum(o.verdict == 'PASS' for o in outcomes)} of {len(outcomes)} PASS"


# --- what gets written


def _scrub(command: str, root: Path) -> str:
    """The command line without the repository's or the home directory's absolute path."""
    command = command.replace(str(root), ".")
    home = str(Path.home())
    return command.replace(home, "~") if len(home) > 1 else command


def _toml_string(text: str) -> str:
    # A JSON string is a valid TOML basic string, but for DEL, which TOML wants escaped too.
    return json.dumps(text, ensure_ascii=False).replace("\x7f", "\\u007f")


def _source_text(source: dict) -> str:
    lines = ["# Where this kept run came from. Written by similitude.runs.keep; never edit a kept run."]
    for key, value in source.items():
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, list):
            text = "[" + "".join(f"\n    {_toml_string(v)}," for v in value) + "\n]"
        else:
            text = _toml_string(value)
        lines.append(f"{key} = {text}")
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _code(text: str) -> str:
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def _report(source: dict, outcomes: list[Outcome]) -> str:
    if source["dirty"]:
        tree = "dirty when kept: " + ", ".join(_code(f) for f in source.get("dirty_files", []))
    else:
        tree = "clean"
    lines = [
        f"# Kept run {source['run']}",
        "",
        f"- **Commit:** `{source['commit']}` on `{source['branch']}`",
        f"- **Working tree:** {tree}",
        f"- **Command:** {_code(source['command'])}",
        f"- **Design:** [{DESIGN}]({DESIGN})",
        "",
        "## Expected checks",
        "",
    ]
    if not outcomes:
        lines.append("No checks were declared before this run.")
        return "\n".join(lines) + "\n"
    lines += [
        f"Declared in [{EXPECT}]({EXPECT}) before the run, and judged against the limits written there. "
        f"Results are from [{RESULTS}]({RESULTS}).",
        "",
        "| Verdict | Check | Declared | Result | Note |",
        "|---|---|---|---|---|",
    ]
    for o in outcomes:
        e = o.expected
        verdict = o.verdict if o.verdict == "PASS" else f"**{o.verdict}**"
        declared = f"{e.op} {e.limit:g} {e.unit}".strip()
        if o.value is None:
            result = ""
        else:
            value = f"{o.value:g}" if isinstance(o.value, float) else str(o.value)
            result = f"{value} {o.unit}".strip()
        cells = (verdict, e.label, declared, result, o.note)
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    lines += ["", f"**Result:** {_tally(outcomes)}."]
    return "\n".join(lines) + "\n"


def _add_readme_row(readme: Path, source: dict, outcomes: list[Outcome], shows: str) -> None:
    text = readme.read_text(encoding="utf-8") if readme.exists() else _README_HEAD
    if not text.endswith("\n"):
        text += "\n"
    commit = f"`{source['commit'][:7]}`" + (" (dirty)" if source["dirty"] else "")
    cells = (f"[{source['run']}]({source['run']}/)", commit, source["branch"], _tally(outcomes), shows)
    readme.write_text(text + "| " + " | ".join(_cell(c) for c in cells) + " |\n", encoding="utf-8")
