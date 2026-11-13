"""Kept runs: an output directory copied to `runs/<date>_<name>/` with where it came from.

`out/` is scratch. A run has two moments:

1. **Before it produces anything, `start(out_dir)`** writes `START` into the output directory: the commit, the
   branch, whether the working tree was dirty, the command line, and the SHA-256 of `expect.toml`.
2. **When it is worth keeping, `keep(out_dir, name)`** copies the output directory and adds:
   - `SOURCE`: the commit, the branch and the command line (TOML; read it with `read_source`);
   - `design.toml`: the resolved design the run used;
   - `KEPT.md`: a report of the source and of each expected check beside its result;
   - a row in `runs/README.md`.

`keep` only knows what the run looked like when it started if `start` recorded it. So `SOURCE` says which it is:

- `recorded = "at run start"`: the commit, branch, dirty state and command are those `start` wrote down, and
  `expect.toml` is known to be unchanged since then.
- `recorded = "when kept"`: the run made no `START`. The commit is the checkout's when `keep` ran, which may not be
  the source that made the results, and nothing shows when `expect.toml` was written. The report says so.

A kept run is never edited or overwritten; make a new one.

A run declares its pass/fail checks before it starts, in `expect.toml` in its output directory:

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
import hashlib
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
START = "START"
SOURCE = "SOURCE"
DESIGN = "design.toml"
EXPECT = "expect.toml"
RESULTS = "checks.json"
REPORT = "KEPT.md"
README = "README.md"

AT_START = "at run start"
WHEN_KEPT = "when kept"
DECLARED_BEFORE = "declared before the run"
FOUND_WHEN_KEPT = "found when kept"
NONE_DECLARED = "none"

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_COMMIT = re.compile(r"[0-9a-f]{40}([0-9a-f]{24})?")
_NON_FINITE = {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": float("-inf")}
_OPS = ("<=", ">=")
_EXPECT_KEYS = {"group", "name", "op", "limit", "unit", "reason"}
_README_HEAD = """\
# Kept runs

Each directory is a copy of an output directory worth keeping, named `<date>_<name>`. Its `SOURCE` file records
the commit, branch and command, and whether they were recorded when the run started or only when it was kept.
`KEPT.md` shows the run's expected checks beside their results. Never edit a kept run; make a new one.

| Run | Commit | Branch | Expected checks | What it shows |
|---|---|---|---|---|
"""


class DirtyTreeError(RuntimeError):
    """The working tree has changes that no commit records."""


class UndisposedWarningError(RuntimeError):
    """The run has a WARN with no disposition recorded."""


class ExpectationsChangedError(RuntimeError):
    """`expect.toml` is not the file the run started with."""


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


def start(
    out_dir: str | os.PathLike,
    *,
    repo: str | os.PathLike | None = None,
    command: str | None = None,
) -> Path:
    """Record where a run starts from, in `START` in its output directory, and return that file.

    Call it before the run produces anything, and after writing `expect.toml` if the run declares checks. It
    records the commit, the branch, the working tree's uncommitted changes, the command line (by default this
    process's own) and the SHA-256 of `expect.toml`. `keep` uses the record in place of what it would otherwise
    read from the checkout at keep time.

    The output directory is created if need be. One that already has a `START` or results is refused: a run
    starts once, in a fresh directory. A dirty tree is recorded, not refused; `keep` decides.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for existing in (SOURCE, START, RESULTS):
        if (out / existing).exists():
            raise FileExistsError(f"{out_dir} already has a {existing}; start a run in a fresh output directory")
    root = _root(repo)
    _declared(out)  # a malformed expect.toml stops the run here, not after it
    record = {"started": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}
    record |= _checkout_state(root)
    record["command"] = _scrub(command if command is not None else shlex.join(sys.argv), root)
    if (out / EXPECT).exists():
        record["expect_sha256"] = _sha256(out / EXPECT)
    path = out / START
    path.write_text(
        _toml_text("Where this run started from. Written by similitude.runs.start; never edit it.", record),
        encoding="utf-8",
    )
    return path


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
    - `command`: for a run with no `START`, the command line that made it; by default this process's own. A run
      with a `START` recorded its own, and passing one is an error. The repository's path is written as `.` and
      the home directory as `~`.
    - `shows`: one line for the `runs/README.md` row, saying what the run shows.
    - `allow_dirty`: keep the run although the working tree had uncommitted changes, and record that it did. The
      changed files are not kept, so such a run can't be reproduced from its commit. Untracked files count, so
      keep output directories ignored, as `out/` is. Changes under `runs/` never count.
    - `date`: the date in the directory's name; by default today.

    The commit, branch, dirty state and command come from the run's `START` if it has one, and otherwise from the
    checkout as it is now; `SOURCE` and the report say which.

    Raises:

    - FileExistsError rather than overwrite a kept run;
    - DirtyTreeError for a dirty tree;
    - UndisposedWarningError if `checks.json` holds a WARN with no disposition;
    - ExpectationsChangedError if `expect.toml` was added, edited or removed after the run started.

    A keep that fails leaves `runs/` as it was: no directory and no README row.
    """
    src = Path(src_dir).resolve()
    if not src.is_dir():
        raise NotADirectoryError(f"{src_dir} is not an output directory")
    if not _NAME.fullmatch(name):
        raise ValueError(f"a run's name is letters, digits, '.', '_' and '-', not {name!r}")
    root = _root(repo)
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

    started = _read_start(src)
    if started is None:
        state = _checkout_state(root)
        state["command"] = _scrub(command if command is not None else shlex.join(sys.argv), root)
        when = "has"
    else:
        if command is not None:
            raise ValueError(f"the run recorded its own command in {START}; don't pass command=")
        if not _git(root, "rev-parse", "--verify", "-q", started["commit"] + "^{commit}", check=False):
            raise ValueError(f"the run started at commit {started['commit'][:7]}, which is not in this repository")
        _check_expectations_unchanged(src, started)
        state = {key: started[key] for key in ("commit", "branch", "dirty", "dirty_files", "command")}
        when = "had"
    dirty_files = state.pop("dirty_files")
    if dirty_files and not allow_dirty:
        raise DirtyTreeError(
            f"the working tree {when} uncommitted changes ({len(dirty_files)} paths, first: {dirty_files[0]}); "
            "commit them and run again, or pass allow_dirty=True to keep the run and record that it was dirty"
        )
    source = {"run": run} | state
    source["recorded"] = WHEN_KEPT if started is None else AT_START
    if started is not None:
        source["started"] = started["started"]
    if not declared:
        source["expectations"] = NONE_DECLARED
    else:
        source["expectations"] = FOUND_WHEN_KEPT if started is None else DECLARED_BEFORE
    if dirty_files:
        source["dirty_files"] = dirty_files

    readme = root / RUNS_DIR / README
    tmp = dest.with_name(f".{run}.tmp")
    readme_tmp = dest.with_name(f".{run}.{README}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        shutil.copytree(src, tmp)
        if design_path != src / DESIGN:
            shutil.copyfile(design_path, tmp / DESIGN)
        header = "Where this kept run came from. Written by similitude.runs.keep; never edit a kept run."
        (tmp / SOURCE).write_text(_toml_text(header, source), encoding="utf-8")
        (tmp / REPORT).write_text(_report(source, outcomes), encoding="utf-8")
        readme_tmp.write_text(_readme_with_row(readme, source, outcomes, shows), encoding="utf-8")
        os.rename(tmp, dest)
        try:
            os.replace(readme_tmp, readme)
        except BaseException:
            os.rename(dest, tmp)  # the run is not kept until its README row is
            raise
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        readme_tmp.unlink(missing_ok=True)
        raise
    return dest


def read_source(run_dir: str | os.PathLike) -> dict:
    """A kept run's `SOURCE`.

    `run`, `commit`, `branch`, `dirty`, `command`; `recorded`, which is "at run start" or "when kept";
    `expectations`, which is "declared before the run", "found when kept" or "none"; `started` if the run made a
    start record; and `dirty_files` if it was dirty.
    """
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


def _root(repo: str | os.PathLike | None) -> Path:
    return Path(_git(Path(repo) if repo is not None else Path.cwd(), "rev-parse", "--show-toplevel")).resolve()


def _checkout_state(root: Path) -> dict:
    """The checkout as it is now: its commit, branch and uncommitted changes outside `runs/`."""
    status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", ".", f":(exclude){RUNS_DIR}")
    dirty_files = [line.strip() for line in status.splitlines()]
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "symbolic-ref", "--short", "-q", "HEAD", check=False) or "(detached)",
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
    }


# --- the start record


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_start(run_dir: Path) -> dict | None:
    """The run's `START`, or None if it made none."""
    path = run_dir / START
    if not path.exists():
        return None
    try:
        record = tomllib.loads(path.read_text(encoding="utf-8"))
        ok = (
            isinstance(record["commit"], str)
            and _COMMIT.fullmatch(record["commit"])
            and isinstance(record["dirty"], bool)
            and all(isinstance(record[key], str) for key in ("started", "branch", "command"))
            and isinstance(record.get("expect_sha256", ""), str)
            and all(isinstance(f, str) for f in record.setdefault("dirty_files", []))
            and record["dirty"] == bool(record["dirty_files"])
        )
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, KeyError, TypeError):
        ok = False
    if not ok:
        raise ValueError(f"{START} is not a record that similitude.runs.start wrote")
    return record


def _check_expectations_unchanged(run_dir: Path, started: dict) -> None:
    recorded = started.get("expect_sha256")
    now = _sha256(run_dir / EXPECT) if (run_dir / EXPECT).exists() else None
    if now == recorded:
        return
    if recorded is None:
        what = "was added after the run started; the run declared no checks, so remove it"
    elif now is None:
        what = "was there when the run started, and is gone; put it back"
    else:
        what = "was edited after the run started; restore the file the run started with"
    raise ExpectationsChangedError(f"{EXPECT} {what}")


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
    unit = str(result.get("unit") or "")
    value = _number(result.get("value"))
    if value is None:
        return Outcome(expected, "FAIL", result.get("value"), unit, "the result is not a number")
    if expected.unit and expected.unit != unit:
        reported = f"reported in {unit}" if unit else "the result has no unit"
        return Outcome(expected, "FAIL", value, unit, f"declared in {expected.unit} but {reported}")
    holds = value <= expected.limit if expected.op == "<=" else value >= expected.limit
    note = ""
    own_limit = _number(result.get("limit"))
    if own_limit is not None and (result.get("op"), own_limit) != (expected.op, float(expected.limit)):
        note = f"the run's own check used {result.get('op')} {own_limit:g}"
    return Outcome(expected, "PASS" if holds else "FAIL", value, unit, note)


def _number(x) -> float | None:
    """A number from `checks.json`, where a NaN or an infinity is written as `{"nonfinite": "NaN"}`."""
    if isinstance(x, dict) and set(x) == {"nonfinite"} and x["nonfinite"] in _NON_FINITE:
        return _NON_FINITE[x["nonfinite"]]
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x)


def _label(result: dict) -> str:
    group = result.get("group", "")
    return f"{group}: {result['name']}" if group else result["name"]


def _tally(outcomes: list[Outcome]) -> str:
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


def _toml_text(header: str, mapping: dict) -> str:
    lines = [f"# {header}"]
    for key, value in mapping.items():
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, list):
            text = "[" + "".join(f"\n    {_toml_string(v)}," for v in value) + ("\n]" if value else "]")
        else:
            text = _toml_string(value)
        lines.append(f"{key} = {text}")
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _code(text: str) -> str:
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def _report(source: dict, outcomes: list[Outcome]) -> str:
    at_start = source["recorded"] == AT_START
    if at_start:
        recorded = f"when the run started ({source['started']}), by `similitude.runs.start`"
    else:
        recorded = (
            "when the run was kept, not when it ran. The run made no start record, so the commit, working tree "
            "and command below are those of the keep, and may not be what produced these results"
        )
    if source["dirty"]:
        files = ", ".join(_code(f) for f in source.get("dirty_files", []))
        tree = f"dirty: {files}. The changed files are not kept, so the commit alone does not reproduce this run"
    else:
        tree = "clean"
    lines = [
        f"# Kept run {source['run']}",
        "",
        f"- **Recorded:** {recorded}.",
        f"- **Commit:** `{source['commit']}` on `{source['branch']}`",
        f"- **Working tree:** {tree}",
        f"- **Command:** {_code(source['command'])}",
        f"- **Design:** [{DESIGN}]({DESIGN})",
        "",
        "## Expected checks",
        "",
    ]
    if not outcomes:
        if at_start:
            lines.append(f"The run started with no {EXPECT}: no checks were declared before it.")
        else:
            lines.append(f"The run has no {EXPECT}: no checks were declared for it.")
        return "\n".join(lines) + "\n"
    if source["expectations"] == DECLARED_BEFORE:
        timing = (
            f"Declared in [{EXPECT}]({EXPECT}) before the run: the file matches the SHA-256 recorded when the "
            "run started."
        )
    else:
        timing = (
            f"From [{EXPECT}]({EXPECT}) as it was when the run was kept. The run made no start record, so nothing "
            "shows these were written before the run."
        )
    lines += [
        f"{timing} Each check is judged against the limit written there. Results are from [{RESULTS}]({RESULTS}).",
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


def _readme_with_row(readme: Path, source: dict, outcomes: list[Outcome], shows: str) -> str:
    """The README's text with this run's row added."""
    text = readme.read_text(encoding="utf-8") if readme.exists() else _README_HEAD
    if not text.endswith("\n"):
        text += "\n"
    commit = f"`{source['commit'][:7]}`"
    if source["dirty"]:
        commit += " (dirty)"
    if source["recorded"] == WHEN_KEPT:
        commit += " (when kept)"
    if not outcomes:
        expected = "none declared"
    else:
        expected = _tally(outcomes) + ("" if source["expectations"] == DECLARED_BEFORE else " (timing unverified)")
    cells = (f"[{source['run']}]({source['run']}/)", commit, source["branch"], expected, shows)
    return text + "| " + " | ".join(_cell(c) for c in cells) + " |\n"
