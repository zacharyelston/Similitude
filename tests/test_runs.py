"""Kept runs, in a temporary git repo: SOURCE contents, refusal to overwrite, dirty-tree handling, expectations."""

import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from similitude import runs

DAY = datetime.date(2026, 11, 13)
RUN = "2026-11-13_sweep"

EXPECT = """\
# Written before the run.
[[check]]
group = "Float"
name = "Freeboard"
op = ">="
limit = 0.15
unit = "m"
reason = "deck stays dry in a 0.1 m chop"

[[check]]
name = "Cable temperature"
op = "<="
limit = 60
unit = "°C"

[[check]]
name = "Range"
op = ">="
limit = 20
unit = "km"
"""
FREEBOARD = '[[check]]\nname = "Freeboard"\nop = ">="\nlimit = 0.15\nunit = "m"\n'


def git(repo, *args):
    env = os.environ | {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    cmd = ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com", *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout.strip()


def result(group, name, value, unit, limit=None, op="<=", warn_limit=None, status="PASS", disposition=""):
    """One check as `similitude.checks.to_json` writes it."""
    return {
        "group": group,
        "name": name,
        "status": status,
        "value": value,
        "unit": unit,
        "limit": limit,
        "op": op,
        "warn_limit": warn_limit,
        "warn_within": 0.0,
        "reason": "",
        "disposition": disposition,
    }


def write_results(out, *results):
    (out / "checks.json").write_text(json.dumps({"checks": list(results)}), encoding="utf-8")


def write_outputs(d):
    (d / "figures").mkdir(parents=True)
    (d / "summary.md").write_text("# Sweep\n")
    (d / "figures" / "polar.svg").write_text("<svg/>\n")
    (d / "design.toml").write_text('[hull]\nlength = "4.57 m"\n')


def begin(repo, name="sweep", expect=None, command="similitude sweep"):
    """A run that declares its checks, records its start, then writes its outputs."""
    d = repo / "out" / name
    d.mkdir(parents=True)
    if expect is not None:
        (d / "expect.toml").write_text(expect, encoding="utf-8")
    runs.start(d, repo=repo, command=command)
    write_outputs(d)
    return d


def commit_a_change(repo, text="thrust = 2\n"):
    (repo / "model.py").write_text(text)
    git(repo, "commit", "-q", "-am", "Change the model")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    """A git repo with one commit on `main`, and `out/` ignored."""
    root = tmp_path / "project"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / ".gitignore").write_text("out/\n")
    (root / "model.py").write_text("thrust = 1\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "A model")
    return root.resolve()


@pytest.fixture
def out(repo):
    """A finished run's output directory. The run recorded its start, and declared no checks."""
    return begin(repo)


@pytest.fixture
def unrecorded(repo):
    """A finished run's output directory with no start record."""
    d = repo / "out" / "sweep"
    write_outputs(d)
    return d


def keep(out, repo, name="sweep", **kwargs):
    return runs.keep(out, name, repo=repo, date=DAY, **kwargs)


def tree(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def kept_names(repo):
    return sorted(p.name for p in (repo / "runs").iterdir()) if (repo / "runs").exists() else []


# --- the start record


def test_start_records_where_the_run_starts_from(repo):
    d = repo / "out" / "sweep"
    d.mkdir(parents=True)
    (d / "expect.toml").write_text(EXPECT, encoding="utf-8")
    path = runs.start(d, repo=repo, command="similitude sweep design/skiff.toml")
    assert path == d / "START"
    record = tomllib.loads(path.read_text(encoding="utf-8"))
    started = datetime.datetime.fromisoformat(record.pop("started"))
    assert abs(datetime.datetime.now(datetime.UTC) - started) < datetime.timedelta(minutes=1)
    assert record == {
        "commit": git(repo, "rev-parse", "HEAD"),
        "branch": "main",
        "dirty": False,
        "dirty_files": [],
        "command": "similitude sweep design/skiff.toml",
        "expect_sha256": hashlib.sha256(EXPECT.encode()).hexdigest(),
    }


def test_start_makes_the_output_directory_and_defaults_to_this_process(repo):
    record = tomllib.loads(runs.start(repo / "out" / "new" / "case", repo=repo).read_text())
    assert shlex.split(record["command"])[1:] == sys.argv[1:]
    assert "expect_sha256" not in record  # the run declared no checks


def test_start_records_a_dirty_tree_without_refusing(repo):
    (repo / "model.py").write_text("thrust = 2\n")
    (repo / "new_model.py").write_text("drag = 1\n")
    record = tomllib.loads(runs.start(repo / "out" / "sweep", repo=repo, command="x").read_text())
    assert record["dirty"] is True
    assert record["dirty_files"] == ["M model.py", "?? new_model.py"]


def test_a_run_starts_once_and_before_its_results(repo, out):
    with pytest.raises(FileExistsError, match="already has a START"):
        runs.start(out, repo=repo)
    late = repo / "out" / "late"
    late.mkdir()
    write_results(late, result("Float", "Freeboard", 0.21, "m"))
    with pytest.raises(FileExistsError, match="already has a checks.json"):
        runs.start(late, repo=repo)
    with pytest.raises(FileExistsError, match="already has a SOURCE"):
        runs.start(keep(out, repo), repo=repo)


def test_a_malformed_expect_toml_stops_the_run_at_its_start(repo):
    d = repo / "out" / "sweep"
    d.mkdir(parents=True)
    (d / "expect.toml").write_text('[[check]]\nname = "x"\nop = "<"\nlimit = 1\n')
    with pytest.raises(ValueError, match="check 1: op must be one of"):
        runs.start(d, repo=repo)
    assert not (d / "START").exists()


# --- the copy and SOURCE


def test_keep_copies_the_output_directory_to_runs(repo, out):
    before = tree(out)
    kept = keep(out, repo)
    assert kept == repo / "runs" / RUN
    assert tree(out) == before
    copied = tree(kept)
    assert {name: copied[name] for name in before} == before
    assert set(copied) - set(before) == {"SOURCE", "KEPT.md"}


def test_the_date_defaults_to_today(repo, out):
    kept = runs.keep(out, "sweep", repo=repo)
    assert kept.name == f"{datetime.date.today().isoformat()}_sweep"


def test_source_records_the_commit_branch_and_command_the_run_started_with(repo):
    out = begin(repo, command="similitude sweep design/skiff.toml --set hull.length='5 m'")
    started = tomllib.loads((out / "START").read_text())["started"]
    kept = keep(out, repo)
    assert runs.read_source(kept) == {
        "run": RUN,
        "commit": git(repo, "rev-parse", "HEAD"),
        "branch": "main",
        "dirty": False,
        "command": "similitude sweep design/skiff.toml --set hull.length='5 m'",
        "recorded": "at run start",
        "started": started,
        "expectations": "none",
    }
    text = (kept / "SOURCE").read_text()
    assert f'commit = "{git(repo, "rev-parse", "HEAD")}"' in text
    assert 'branch = "main"' in text


def test_source_names_the_commit_that_made_the_run_not_the_one_checked_out_when_kept(repo, out):
    made_at = git(repo, "rev-parse", "HEAD")
    git(repo, "switch", "-q", "-c", "feat/more-thrust")
    later = commit_a_change(repo)
    kept = keep(out, repo)
    source = runs.read_source(kept)
    assert later != made_at
    assert (source["commit"], source["branch"], source["recorded"]) == (made_at, "main", "at run start")
    report = (kept / "KEPT.md").read_text()
    assert f"- **Commit:** `{made_at}` on `main`\n" in report
    assert f"- **Recorded:** when the run started ({source['started']}), by `similitude.runs.start`.\n" in report
    assert f"`{made_at[:7]}` | main |" in (repo / "runs" / "README.md").read_text().splitlines()[-1]


def test_a_run_with_no_start_record_is_kept_but_its_source_is_labelled_as_read_when_kept(repo, unrecorded):
    later = commit_a_change(repo)  # the outputs were made before this commit; nothing recorded that
    kept = keep(unrecorded, repo, command="similitude sweep")
    assert runs.read_source(kept) == {
        "run": RUN,
        "commit": later,
        "branch": "main",
        "dirty": False,
        "command": "similitude sweep",
        "recorded": "when kept",
        "expectations": "none",
    }
    report = (kept / "KEPT.md").read_text()
    assert (
        "- **Recorded:** when the run was kept, not when it ran. The run made no start record, so the commit, "
        "working tree and command below are those of the keep, and may not be what produced these results.\n"
    ) in report
    assert report.endswith("The run has no expect.toml: no checks were declared for it.\n")
    assert f"| `{later[:7]}` (when kept) | main |" in (repo / "runs" / "README.md").read_text().splitlines()[-1]


def test_source_records_another_branch_and_a_detached_head(repo):
    git(repo, "switch", "-q", "-c", "feat/longer-hull")
    assert runs.read_source(keep(begin(repo, "branch"), repo, name="branch"))["branch"] == "feat/longer-hull"
    git(repo, "switch", "-q", "--detach")
    assert runs.read_source(keep(begin(repo, "detached"), repo, name="detached"))["branch"] == "(detached)"


def test_a_start_commit_this_repository_doesnt_have_is_refused(repo, out):
    start = out / "START"
    start.write_text(start.read_text().replace(git(repo, "rev-parse", "HEAD"), "0" * 40))
    with pytest.raises(ValueError, match="started at commit 0000000, which is not in this repository"):
        keep(out, repo)
    assert kept_names(repo) == []


@pytest.mark.parametrize(
    "damage",
    [
        lambda text: "not toml [",
        lambda text: text.replace("commit = ", "comit = "),
        lambda text: text.replace("dirty = false", 'dirty = "no"'),
        lambda text: text.replace("dirty = false", "dirty = true"),  # dirty, but with no files named
        lambda text: text.replace('branch = "main"', "branch = 1"),
    ],
)
def test_a_start_record_start_didnt_write_is_refused(repo, out, damage):
    (out / "START").write_text(damage((out / "START").read_text()))
    with pytest.raises(ValueError, match="START is not a record that similitude.runs.start wrote"):
        keep(out, repo)


def test_the_command_comes_from_the_start_record_or_from_keep_never_both(repo, out, unrecorded_elsewhere):
    with pytest.raises(ValueError, match="recorded its own command in START"):
        keep(out, repo, command="another command")
    kept = runs.keep(unrecorded_elsewhere, "plain", repo=repo, date=DAY)
    assert shlex.split(runs.read_source(kept)["command"])[1:] == sys.argv[1:]


@pytest.fixture
def unrecorded_elsewhere(repo):
    d = repo / "out" / "plain"
    write_outputs(d)
    return d


def test_the_command_is_recorded_without_private_paths(repo, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home" / "somebody"))
    command = f'{tmp_path}/home/somebody/bin/tool --design {repo}/design/skiff.toml --note "a \\\\ b"'
    scrubbed = '~/bin/tool --design ./design/skiff.toml --note "a \\\\ b"'
    assert runs.read_source(keep(begin(repo, command=command), repo))["command"] == scrubbed
    plain = repo / "out" / "plain"
    write_outputs(plain)
    assert runs.read_source(keep(plain, repo, name="plain", command=command))["command"] == scrubbed


def test_the_repo_defaults_to_the_one_around_the_current_directory(repo, monkeypatch):
    (repo / "out" / "elsewhere").mkdir(parents=True)
    monkeypatch.chdir(repo / "out" / "elsewhere")
    runs.start(repo / "out" / "sweep", command="similitude sweep")
    write_outputs(repo / "out" / "sweep")
    assert runs.keep(repo / "out" / "sweep", "sweep", date=DAY) == repo / "runs" / RUN


def test_keeping_outside_a_git_repo_is_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "design.toml").write_text("")
    with pytest.raises(RuntimeError, match="git rev-parse"):
        runs.keep(out, "sweep", repo=tmp_path)
    with pytest.raises(RuntimeError, match="git rev-parse"):
        runs.start(tmp_path / "out2", repo=tmp_path)


@pytest.mark.parametrize("name", ["", "../escape", "a/b", ".hidden", "two words"])
def test_a_name_that_isnt_a_plain_slug_is_refused(repo, out, name):
    with pytest.raises(ValueError, match="a run's name"):
        keep(out, repo, name=name)
    assert not (repo / "runs").exists()


def test_a_missing_output_directory_is_refused(repo):
    with pytest.raises(NotADirectoryError):
        keep(repo / "out" / "nothing", repo)


# --- the resolved design


def test_the_runs_own_design_is_kept(repo, out):
    assert (keep(out, repo) / "design.toml").read_text() == '[hull]\nlength = "4.57 m"\n'


def test_a_design_passed_in_is_copied_as_design_toml(repo, out, tmp_path):
    (out / "design.toml").unlink()
    resolved = tmp_path / "skiff.resolved.toml"
    resolved.write_text('[hull]\nlength = "5 m"\n')
    kept = keep(out, repo, design=resolved)
    assert (kept / "design.toml").read_text() == '[hull]\nlength = "5 m"\n'
    assert not (out / "design.toml").exists()


def test_a_run_with_no_design_is_refused(repo, out):
    (out / "design.toml").unlink()
    with pytest.raises(FileNotFoundError, match="needs its resolved design"):
        keep(out, repo)
    with pytest.raises(FileNotFoundError, match="no design file"):
        keep(out, repo, design=repo / "missing.toml")
    assert not (repo / "runs").exists()


def test_two_designs_for_one_run_are_refused(repo, out, tmp_path):
    other = tmp_path / "other.toml"
    other.write_text("[hull]\n")
    with pytest.raises(ValueError, match="already has a design.toml"):
        keep(out, repo, design=other)
    assert keep(out, repo, design=out / "design.toml").is_dir()  # naming the run's own design is fine


def test_a_design_that_isnt_toml_is_refused(repo, out):
    (out / "design.toml").write_text("{not toml")
    with pytest.raises(ValueError, match="is not TOML"):
        keep(out, repo)


# --- refusal to overwrite, and all or nothing


def test_keeping_refuses_to_overwrite_a_kept_run(repo, out):
    kept = keep(out, repo, shows="the first sweep")
    before = tree(kept)
    readme = (repo / "runs" / "README.md").read_text()
    (out / "summary.md").write_text("# A different sweep\n")
    with pytest.raises(FileExistsError, match="never overwritten"):
        keep(out, repo, shows="a different sweep")
    assert tree(kept) == before
    assert (repo / "runs" / "README.md").read_text() == readme


def test_the_same_name_on_another_day_is_a_new_run(repo, out):
    keep(out, repo)
    later = runs.keep(out, "sweep", repo=repo, date=datetime.date(2026, 11, 14))
    assert later.name == "2026-11-14_sweep"


def test_a_kept_run_isnt_kept_again(repo, out):
    kept = keep(out, repo)
    with pytest.raises(ValueError, match="is it a kept run"):
        keep(kept, repo, name="again")


def test_a_keep_that_fails_while_copying_leaves_nothing_behind(repo, out, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(runs.shutil, "copyfile", fail)
    resolved = repo / "out" / "resolved.toml"
    (out / "design.toml").rename(resolved)
    with pytest.raises(OSError, match="disk full"):
        keep(out, repo, design=resolved)
    assert kept_names(repo) == []
    monkeypatch.undo()
    assert keep(out, repo, design=resolved).is_dir()


def test_a_keep_that_fails_writing_the_readme_row_keeps_nothing_and_can_be_retried(repo, out, monkeypatch):
    first = keep(begin(repo, "first"), repo, name="first", shows="the first")
    readme = (repo / "runs" / "README.md").read_text()
    before = tree(first)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(runs.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        keep(out, repo)
    assert kept_names(repo) == ["2026-11-13_first", "README.md"]  # no run directory, no temporary files
    assert (repo / "runs" / "README.md").read_text() == readme
    assert tree(first) == before
    monkeypatch.undo()
    assert keep(out, repo).is_dir()  # the same keep now succeeds
    assert len((repo / "runs" / "README.md").read_text().splitlines()) == len(readme.splitlines()) + 1


def test_a_keep_that_cant_read_the_readme_keeps_nothing(repo, out):
    (repo / "runs" / "README.md").mkdir(parents=True)  # not a file: reading it fails
    with pytest.raises(OSError):
        keep(out, repo)
    assert kept_names(repo) == ["README.md"]
    (repo / "runs" / "README.md").rmdir()
    assert keep(out, repo).is_dir()


# --- a dirty working tree


def test_a_run_started_on_a_changed_tracked_file_is_refused(repo):
    (repo / "model.py").write_text("thrust = 2\n")
    out = begin(repo)
    with pytest.raises(runs.DirtyTreeError, match="had uncommitted changes.*M model.py"):
        keep(out, repo)
    assert not (repo / "runs").exists()


def test_a_run_started_beside_an_untracked_file_is_refused(repo):
    (repo / "new_model.py").write_text("drag = 1\n")
    with pytest.raises(runs.DirtyTreeError, match=r"\?\? new_model.py"):
        keep(begin(repo), repo)


def test_a_run_started_on_a_staged_change_is_refused(repo):
    (repo / "model.py").write_text("thrust = 2\n")
    git(repo, "add", "model.py")
    with pytest.raises(runs.DirtyTreeError):
        keep(begin(repo), repo)


def test_the_tree_that_counts_is_the_one_the_run_started_on(repo):
    (repo / "model.py").write_text("thrust = 2\n")
    dirty_start = begin(repo, "dirty-start")
    git(repo, "commit", "-q", "-am", "Change the model")  # committing afterwards doesn't clean the run
    with pytest.raises(runs.DirtyTreeError):
        keep(dirty_start, repo, name="dirty-start")

    clean_start = begin(repo, "clean-start")
    (repo / "model.py").write_text("thrust = 3\n")  # editing afterwards doesn't dirty the run
    assert runs.read_source(keep(clean_start, repo, name="clean-start"))["dirty"] is False


def test_a_run_with_no_start_record_is_refused_on_a_tree_dirty_when_kept(repo, unrecorded):
    (repo / "model.py").write_text("thrust = 2\n")
    with pytest.raises(runs.DirtyTreeError, match="has uncommitted changes.*M model.py"):
        keep(unrecorded, repo, command="similitude sweep")
    assert runs.read_source(keep(unrecorded, repo, command="similitude sweep", allow_dirty=True))["dirty"] is True


def test_allow_dirty_keeps_the_run_and_records_that_it_was_dirty(repo):
    (repo / "model.py").write_text("thrust = 2\n")
    (repo / "new_model.py").write_text("drag = 1\n")
    kept = keep(begin(repo), repo, allow_dirty=True)
    source = runs.read_source(kept)
    assert source["dirty"] is True
    assert source["dirty_files"] == ["M model.py", "?? new_model.py"]
    assert source["commit"] == git(repo, "rev-parse", "HEAD")
    assert (
        "- **Working tree:** dirty: `M model.py`, `?? new_model.py`. The changed files are not kept, so the commit "
        "alone does not reproduce this run\n"
    ) in (kept / "KEPT.md").read_text()
    assert "(dirty)" in (repo / "runs" / "README.md").read_text().splitlines()[-1]


def test_allow_dirty_on_a_clean_tree_records_clean(repo, out):
    kept = keep(out, repo, allow_dirty=True)
    assert runs.read_source(kept)["dirty"] is False
    assert "dirty_files" not in runs.read_source(kept)
    assert "- **Working tree:** clean" in (kept / "KEPT.md").read_text()


def test_kept_runs_not_yet_committed_dont_make_the_tree_dirty(repo):
    keep(begin(repo, "first"), repo, name="first")
    kept = keep(begin(repo, "second"), repo, name="second")
    assert runs.read_source(kept)["dirty"] is False


# --- expectations


def test_the_report_shows_each_expected_check_beside_its_result(repo):
    out = begin(repo, expect=EXPECT)
    write_results(
        out,
        result("Float", "Freeboard", 0.21, "m", 0.15, ">="),
        result("Bus", "Cable temperature", 68.0, "°C", 90.0, "<=", 60.0, "WARN", "Peak lasts under a minute."),
        result("Mass", "All-up mass", 412.0, "kg", status="INFO"),
    )
    kept = keep(out, repo)
    assert (kept / "expect.toml").read_text(encoding="utf-8") == EXPECT
    source = runs.read_source(kept)
    assert source["expectations"] == "declared before the run"
    report = (kept / "KEPT.md").read_text(encoding="utf-8")
    assert report == runs.report(kept)
    assert report == (
        f"# Kept run {RUN}\n"
        "\n"
        f"- **Recorded:** when the run started ({source['started']}), by `similitude.runs.start`.\n"
        f"- **Commit:** `{git(repo, 'rev-parse', 'HEAD')}` on `main`\n"
        "- **Working tree:** clean\n"
        "- **Command:** `similitude sweep`\n"
        "- **Design:** [design.toml](design.toml)\n"
        "\n"
        "## Expected checks\n"
        "\n"
        "Declared in [expect.toml](expect.toml) before the run: the file matches the SHA-256 recorded when the run "
        "started. Each check is judged against the limit written there. Results are from [checks.json](checks.json).\n"
        "\n"
        "| Verdict | Check | Declared | Result | Note |\n"
        "|---|---|---|---|---|\n"
        "| PASS | Float: Freeboard | >= 0.15 m | 0.21 m |  |\n"
        "| **FAIL** | Cable temperature | <= 60 °C | 68 °C | the run's own check used <= 90 |\n"
        "| **MISSING** | Range | >= 20 km |  | the run reported no check with this name |\n"
        "\n"
        "**Result:** 1 of 3 PASS.\n"
    )


def test_expectations_pair_each_declared_check_with_its_result(repo):
    out = begin(repo, expect=EXPECT)
    write_results(out, result("Float", "Freeboard", 0.15, "m", 0.15, ">="), result("Trip", "Range", 19.9, "km"))
    outcomes = runs.expectations(keep(out, repo))
    assert [(o.expected.name, o.verdict, o.value) for o in outcomes] == [
        ("Freeboard", "PASS", 0.15),  # the limit itself holds
        ("Cable temperature", "MISSING", None),
        ("Range", "FAIL", 19.9),
    ]
    assert outcomes[0].expected == runs.Expectation(
        "Freeboard", ">=", 0.15, group="Float", unit="m", reason="deck stays dry in a 0.1 m chop"
    )


def test_a_result_is_judged_against_the_declared_limit_not_the_runs_own(repo):
    out = begin(repo, expect='[[check]]\nname = "Draft"\nop = "<="\nlimit = 0.30\n')
    write_results(out, result("Float", "Draft", 0.35, "m", 0.40, "<="))  # the run moved the limit, and passed itself
    (outcome,) = runs.expectations(keep(out, repo))
    assert outcome.verdict == "FAIL"
    assert outcome.note == "the run's own check used <= 0.4"


def test_an_expectation_added_after_the_run_started_is_refused(repo, out):
    write_results(out, result("Float", "Freeboard", 0.21, "m"))
    (out / "expect.toml").write_text(FREEBOARD)  # written with the result in hand
    with pytest.raises(runs.ExpectationsChangedError, match="expect.toml was added after the run started"):
        keep(out, repo)
    assert kept_names(repo) == []
    (out / "expect.toml").unlink()
    assert runs.read_source(keep(out, repo))["expectations"] == "none"


def test_an_expectation_edited_after_the_run_started_is_refused(repo):
    out = begin(repo, expect=FREEBOARD)
    write_results(out, result("Float", "Freeboard", 0.12, "m"))
    (out / "expect.toml").write_text(FREEBOARD.replace("0.15", "0.10"))  # relaxed until the result passes
    with pytest.raises(runs.ExpectationsChangedError, match="expect.toml was edited after the run started"):
        keep(out, repo)
    assert kept_names(repo) == []
    (out / "expect.toml").write_text(FREEBOARD)
    (outcome,) = runs.expectations(keep(out, repo))
    assert outcome.verdict == "FAIL"


def test_an_expectation_removed_after_the_run_started_is_refused(repo):
    out = begin(repo, expect=FREEBOARD)
    (out / "expect.toml").unlink()
    with pytest.raises(runs.ExpectationsChangedError, match="was there when the run started, and is gone"):
        keep(out, repo)


def test_expectations_with_no_start_record_are_reported_as_unverified(repo, unrecorded):
    write_results(unrecorded, result("Float", "Freeboard", 0.21, "m"))
    (unrecorded / "expect.toml").write_text(FREEBOARD)  # nothing can show when this was written
    kept = keep(unrecorded, repo, command="similitude sweep")
    assert runs.read_source(kept)["expectations"] == "found when kept"
    report = (kept / "KEPT.md").read_text()
    assert (
        "From [expect.toml](expect.toml) as it was when the run was kept. The run made no start record, so nothing "
        "shows these were written before the run."
    ) in report
    assert "before the run:" not in report
    assert "| 1 of 1 PASS (timing unverified) |" in (repo / "runs" / "README.md").read_text().splitlines()[-1]


@pytest.mark.parametrize(
    ("value", "unit", "verdict", "note"),
    [
        (0.2, "m", "PASS", ""),
        (0.2, "cm", "FAIL", "declared in m but reported in cm"),
        (0.2, "", "FAIL", "declared in m but the result has no unit"),
        (0.2, None, "FAIL", "declared in m but the result has no unit"),
        (0.2, "absent", "FAIL", "declared in m but the result has no unit"),
        ("deep", "m", "FAIL", "the result is not a number"),
        ("NaN", "m", "FAIL", "the result is not a number"),  # text, not the number
        ({"nonfinite": "NaN"}, "m", "FAIL", ""),
        ({"nonfinite": "Infinity"}, "m", "PASS", ""),
        ({"nonfinite": "-Infinity"}, "m", "FAIL", ""),
    ],
)
def test_a_result_that_cant_be_compared_fails(repo, value, unit, verdict, note):
    out = begin(repo, expect=FREEBOARD)
    reported = result("Float", "Freeboard", value, unit)
    if unit == "absent":
        del reported["unit"]
    write_results(out, reported)
    (outcome,) = runs.expectations(keep(out, repo))
    assert (outcome.verdict, outcome.note) == (verdict, note)


def test_an_expectation_with_no_unit_takes_the_results_unit(repo):
    out = begin(repo, expect='[[check]]\nname = "Freeboard"\nop = ">="\nlimit = 0.15\n')
    write_results(out, result("Float", "Freeboard", 0.2, "m"))
    (outcome,) = runs.expectations(keep(out, repo))
    assert (outcome.verdict, outcome.unit) == ("PASS", "m")


def test_a_group_tells_apart_results_with_one_name(repo):
    port, starboard = result("Port float", "Freeboard", 0.2, "m"), result("Starboard float", "Freeboard", 0.1, "m")
    vague = begin(repo, "vague", expect='[[check]]\nname = "Freeboard"\nop = ">="\nlimit = 0.15\n')
    write_results(vague, port, starboard)
    with pytest.raises(ValueError, match="matches results in several groups"):
        keep(vague, repo)
    exact = begin(
        repo, "exact", expect='[[check]]\ngroup = "Starboard float"\nname = "Freeboard"\nop = ">="\nlimit = 0.15\n'
    )
    write_results(exact, port, starboard)
    (outcome,) = runs.expectations(keep(exact, repo))
    assert (outcome.verdict, outcome.value) == ("FAIL", 0.1)


def test_a_run_that_declared_no_checks_says_so(repo, out):
    write_results(out, result("Float", "Freeboard", 0.21, "m", 0.15, ">="))
    kept = keep(out, repo)
    assert runs.expectations(kept) == []
    assert (kept / "KEPT.md").read_text().endswith(
        "## Expected checks\n\nThe run started with no expect.toml: no checks were declared before it.\n"
    )


@pytest.mark.parametrize(
    ("expect", "message"),
    [
        ("[[check]\n", "expect.toml is not TOML"),
        ('[[checks]]\nname = "x"\nop = "<="\nlimit = 1\n', "holds \\[\\[check\\]\\] tables only"),
        ('[[check]]\nname = "x"\nop = "<="\n', "check 1: needs a limit"),
        ('[[check]]\nname = "x"\nop = "<"\nlimit = 1\n', "check 1: op must be one of"),
        ('[[check]]\nname = "x"\nop = "<="\nlimit = "1"\n', "check 1: the limit must be a number"),
        ('[[check]]\nname = "x"\nop = "<="\nlimit = 1\nlimt = 2\n', "check 1: unknown keys \\['limt'\\]"),
        ('[[check]]\nname = "x"\nop = "<="\nlimit = 1\n' * 2, "declares these more than once: x"),
    ],
)
def test_a_malformed_expect_toml_is_refused_before_anything_is_kept(repo, unrecorded, expect, message):
    (unrecorded / "expect.toml").write_text(expect)
    with pytest.raises(ValueError, match=message):
        keep(unrecorded, repo)
    assert not (repo / "runs").exists()


def test_results_in_another_format_are_refused(repo, out):
    (out / "checks.json").write_text('[{"name": "Freeboard"}]')
    with pytest.raises(ValueError, match="checks.json is not the JSON"):
        keep(out, repo)


# --- a WARN needs a disposition before the run is kept


def test_a_warn_with_no_disposition_is_refused(repo, out):
    write_results(
        out,
        result("Bus", "Cable temperature", 68.0, "°C", 90.0, "<=", 60.0, "WARN"),
        result("Bus", "Fuse margin", 1.1, "", 1.0, ">=", 1.2, "WARN", "Sized for the 3 HP unit."),
    )
    with pytest.raises(runs.UndisposedWarningError, match="each WARN before keeping: Bus: Cable temperature$"):
        keep(out, repo)
    assert not (repo / "runs").exists()


def test_a_warn_with_a_disposition_is_kept(repo, out):
    write_results(out, result("Bus", "Cable temperature", 68.0, "°C", 90.0, "<=", 60.0, "WARN", "Peak is brief."))
    assert keep(out, repo).is_dir()


# --- runs/README.md


def test_the_readme_gets_one_row_per_kept_run(repo):
    out = begin(repo, expect=EXPECT)
    write_results(out, result("Float", "Freeboard", 0.21, "m"), result("", "Cable temperature", 55.0, "°C"))
    sha = git(repo, "rev-parse", "HEAD")[:7]
    keep(out, repo, shows="Every hull length | every payload")
    git(repo, "switch", "-q", "-c", "feat/longer-hull")
    keep(begin(repo, "longer-hull"), repo, name="longer-hull")
    readme = (repo / "runs" / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Kept runs\n")
    assert readme.splitlines()[-4:] == [
        "| Run | Commit | Branch | Expected checks | What it shows |",
        "|---|---|---|---|---|",
        f"| [{RUN}]({RUN}/) | `{sha}` | main | 2 of 3 PASS | Every hull length \\| every payload |",
        f"| [2026-11-13_longer-hull](2026-11-13_longer-hull/) | `{sha}` | feat/longer-hull | none declared |  |",
    ]


def test_an_existing_readme_is_added_to(repo, out):
    (repo / "runs").mkdir()
    header = "| Run | Commit | Branch | Expected checks | What it shows |"
    (repo / "runs" / "README.md").write_text(f"# Our runs\n\n{header}")
    keep(out, repo, shows="first")
    lines = (repo / "runs" / "README.md").read_text().splitlines()
    assert lines[0] == "# Our runs"
    assert lines[-2] == header
    assert lines[-1].endswith("| none declared | first |")


def test_the_whole_thing_holds_together_from_path_strings(repo):
    d = repo / "out" / "sweep"
    assert isinstance(runs.start(str(d), repo=str(repo), command="similitude sweep"), Path)
    write_outputs(d)
    kept = runs.keep(str(d), "sweep", repo=str(repo), date=DAY)
    assert isinstance(kept, Path)
    assert runs.read_source(str(kept))["run"] == RUN
