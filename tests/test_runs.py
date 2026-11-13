"""Kept runs, in a temporary git repo: SOURCE contents, refusal to overwrite, dirty-tree handling, expectations."""

import datetime
import json
import os
import shlex
import subprocess
import sys
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
    """A finished run's output directory, with the design it used."""
    d = repo / "out" / "sweep"
    (d / "figures").mkdir(parents=True)
    (d / "summary.md").write_text("# Sweep\n")
    (d / "figures" / "polar.svg").write_text("<svg/>\n")
    (d / "design.toml").write_text('[hull]\nlength = "4.57 m"\n')
    return d


def keep(out, repo, name="sweep", **kwargs):
    return runs.keep(out, name, repo=repo, date=DAY, **{"command": "similitude sweep", **kwargs})


def tree(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


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


def test_source_records_the_commit_branch_and_command(repo, out):
    kept = keep(out, repo, command="similitude sweep design/skiff.toml --set hull.length='5 m'")
    assert runs.read_source(kept) == {
        "run": RUN,
        "commit": git(repo, "rev-parse", "HEAD"),
        "branch": "main",
        "dirty": False,
        "command": "similitude sweep design/skiff.toml --set hull.length='5 m'",
    }
    text = (kept / "SOURCE").read_text()
    assert f'commit = "{git(repo, "rev-parse", "HEAD")}"' in text
    assert 'branch = "main"' in text


def test_source_records_another_branch_and_a_detached_head(repo, out):
    git(repo, "switch", "-q", "-c", "feat/longer-hull")
    assert runs.read_source(keep(out, repo, name="branch"))["branch"] == "feat/longer-hull"
    git(repo, "switch", "-q", "--detach")
    assert runs.read_source(keep(out, repo, name="detached"))["branch"] == "(detached)"


def test_the_command_defaults_to_this_process(repo, out):
    kept = runs.keep(out, "sweep", repo=repo, date=DAY)
    assert shlex.split(runs.read_source(kept)["command"])[1:] == sys.argv[1:]


def test_the_command_is_recorded_without_private_paths(repo, out, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home" / "somebody"))
    command = f'{tmp_path}/home/somebody/bin/tool --design {repo}/design/skiff.toml --note "a \\\\ b"'
    kept = keep(out, repo, command=command)
    assert runs.read_source(kept)["command"] == '~/bin/tool --design ./design/skiff.toml --note "a \\\\ b"'


def test_the_repo_defaults_to_the_one_around_the_current_directory(repo, out, monkeypatch):
    (repo / "out" / "elsewhere").mkdir()
    monkeypatch.chdir(repo / "out" / "elsewhere")
    assert runs.keep(out, "sweep", date=DAY) == repo / "runs" / RUN


def test_keeping_outside_a_git_repo_is_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "design.toml").write_text("")
    with pytest.raises(RuntimeError, match="git rev-parse"):
        runs.keep(out, "sweep", repo=tmp_path)


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


# --- refusal to overwrite


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
    later = runs.keep(out, "sweep", repo=repo, date=datetime.date(2026, 11, 14), command="similitude sweep")
    assert later.name == "2026-11-14_sweep"


def test_a_kept_run_isnt_kept_again(repo, out):
    kept = keep(out, repo)
    with pytest.raises(ValueError, match="is it a kept run"):
        keep(kept, repo, name="again")


def test_a_failed_keep_leaves_nothing_behind(repo, out, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(runs.shutil, "copyfile", fail)
    resolved = repo / "out" / "resolved.toml"
    (out / "design.toml").rename(resolved)
    with pytest.raises(OSError, match="disk full"):
        keep(out, repo, design=resolved)
    assert list((repo / "runs").iterdir()) == []
    monkeypatch.undo()
    assert keep(out, repo, design=resolved).is_dir()


# --- a dirty working tree


def test_a_changed_tracked_file_is_refused(repo, out):
    (repo / "model.py").write_text("thrust = 2\n")
    with pytest.raises(runs.DirtyTreeError, match="uncommitted changes.*M model.py"):
        keep(out, repo)
    assert not (repo / "runs").exists()


def test_an_untracked_file_is_refused(repo, out):
    (repo / "new_model.py").write_text("drag = 1\n")
    with pytest.raises(runs.DirtyTreeError, match=r"\?\? new_model.py"):
        keep(out, repo)


def test_a_staged_change_is_refused(repo, out):
    (repo / "model.py").write_text("thrust = 2\n")
    git(repo, "add", "model.py")
    with pytest.raises(runs.DirtyTreeError):
        keep(out, repo)


def test_allow_dirty_keeps_the_run_and_records_that_it_was_dirty(repo, out):
    (repo / "model.py").write_text("thrust = 2\n")
    (repo / "new_model.py").write_text("drag = 1\n")
    kept = keep(out, repo, allow_dirty=True)
    source = runs.read_source(kept)
    assert source["dirty"] is True
    assert source["dirty_files"] == ["M model.py", "?? new_model.py"]
    assert source["commit"] == git(repo, "rev-parse", "HEAD")
    assert "- **Working tree:** dirty when kept: `M model.py`, `?? new_model.py`" in (kept / "KEPT.md").read_text()
    assert "(dirty)" in (repo / "runs" / "README.md").read_text().splitlines()[-1]


def test_allow_dirty_on_a_clean_tree_records_clean(repo, out):
    kept = keep(out, repo, allow_dirty=True)
    assert runs.read_source(kept)["dirty"] is False
    assert "dirty_files" not in runs.read_source(kept)
    assert "- **Working tree:** clean" in (kept / "KEPT.md").read_text()


def test_kept_runs_not_yet_committed_dont_make_the_tree_dirty(repo, out):
    keep(out, repo, name="first")
    kept = keep(out, repo, name="second")
    assert runs.read_source(kept)["dirty"] is False


# --- expectations


def test_the_report_shows_each_expected_check_beside_its_result(repo, out):
    (out / "expect.toml").write_text(EXPECT, encoding="utf-8")
    write_results(
        out,
        result("Float", "Freeboard", 0.21, "m", 0.15, ">="),
        result("Bus", "Cable temperature", 68.0, "°C", 90.0, "<=", 60.0, "WARN", "Peak lasts under a minute."),
        result("Mass", "All-up mass", 412.0, "kg", status="INFO"),
    )
    kept = keep(out, repo)
    assert (kept / "expect.toml").read_text(encoding="utf-8") == EXPECT
    report = (kept / "KEPT.md").read_text(encoding="utf-8")
    assert report == runs.report(kept)
    assert report == (
        f"# Kept run {RUN}\n"
        "\n"
        f"- **Commit:** `{git(repo, 'rev-parse', 'HEAD')}` on `main`\n"
        "- **Working tree:** clean\n"
        "- **Command:** `similitude sweep`\n"
        "- **Design:** [design.toml](design.toml)\n"
        "\n"
        "## Expected checks\n"
        "\n"
        "Declared in [expect.toml](expect.toml) before the run, and judged against the limits written there. "
        "Results are from [checks.json](checks.json).\n"
        "\n"
        "| Verdict | Check | Declared | Result | Note |\n"
        "|---|---|---|---|---|\n"
        "| PASS | Float: Freeboard | >= 0.15 m | 0.21 m |  |\n"
        "| **FAIL** | Cable temperature | <= 60 °C | 68 °C | the run's own check used <= 90 |\n"
        "| **MISSING** | Range | >= 20 km |  | the run reported no check with this name |\n"
        "\n"
        "**Result:** 1 of 3 PASS.\n"
    )


def test_expectations_pair_each_declared_check_with_its_result(repo, out):
    (out / "expect.toml").write_text(EXPECT, encoding="utf-8")
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


def test_a_result_is_judged_against_the_declared_limit_not_the_runs_own(repo, out):
    (out / "expect.toml").write_text('[[check]]\nname = "Draft"\nop = "<="\nlimit = 0.30\n')
    write_results(out, result("Float", "Draft", 0.35, "m", 0.40, "<="))  # the run moved the limit, and passed itself
    (outcome,) = runs.expectations(keep(out, repo))
    assert outcome.verdict == "FAIL"
    assert outcome.note == "the run's own check used <= 0.4"


@pytest.mark.parametrize(
    ("value", "unit", "verdict", "note"),
    [
        (0.2, "cm", "FAIL", "declared in m but reported in cm"),
        ("deep", "m", "FAIL", "the result is not a number"),
        ("NaN", "m", "FAIL", ""),
        ("Infinity", "m", "PASS", ""),
        (0.2, "", "PASS", ""),
    ],
)
def test_results_that_cant_be_compared_fail(repo, out, value, unit, verdict, note):
    (out / "expect.toml").write_text('[[check]]\nname = "Freeboard"\nop = ">="\nlimit = 0.15\nunit = "m"\n')
    write_results(out, result("Float", "Freeboard", value, unit))
    (outcome,) = runs.expectations(keep(out, repo))
    assert (outcome.verdict, outcome.note) == (verdict, note)


def test_a_group_tells_apart_results_with_one_name(repo, out):
    write_results(out, result("Port float", "Freeboard", 0.2, "m"), result("Starboard float", "Freeboard", 0.1, "m"))
    (out / "expect.toml").write_text('[[check]]\nname = "Freeboard"\nop = ">="\nlimit = 0.15\n')
    with pytest.raises(ValueError, match="matches results in several groups"):
        keep(out, repo)
    (out / "expect.toml").write_text(
        '[[check]]\ngroup = "Starboard float"\nname = "Freeboard"\nop = ">="\nlimit = 0.15\n'
    )
    (outcome,) = runs.expectations(keep(out, repo))
    assert (outcome.verdict, outcome.value) == ("FAIL", 0.1)


def test_a_run_with_no_expectations_says_so(repo, out):
    write_results(out, result("Float", "Freeboard", 0.21, "m", 0.15, ">="))
    kept = keep(out, repo)
    assert runs.expectations(kept) == []
    assert (kept / "KEPT.md").read_text().endswith("## Expected checks\n\nNo checks were declared before this run.\n")


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
def test_a_malformed_expect_toml_is_refused_before_anything_is_kept(repo, out, expect, message):
    (out / "expect.toml").write_text(expect)
    with pytest.raises(ValueError, match=message):
        keep(out, repo)
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


def test_the_readme_gets_one_row_per_kept_run(repo, out):
    (out / "expect.toml").write_text(EXPECT, encoding="utf-8")
    write_results(out, result("Float", "Freeboard", 0.21, "m"), result("", "Cable temperature", 55.0, "°C"))
    sha = git(repo, "rev-parse", "HEAD")[:7]
    keep(out, repo, shows="Every hull length | every payload")
    (out / "expect.toml").unlink()
    git(repo, "switch", "-q", "-c", "feat/longer-hull")
    keep(out, repo, name="longer-hull")
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
    (repo / "runs" / "README.md").write_text("# Our runs\n\n| Run | Commit | Branch | Expected checks | What it shows |")
    keep(out, repo, shows="first")
    lines = (repo / "runs" / "README.md").read_text().splitlines()
    assert lines[0] == "# Our runs"
    assert lines[-2] == "| Run | Commit | Branch | Expected checks | What it shows |"
    assert lines[-1].endswith("| none declared | first |")


def test_the_whole_thing_holds_together_from_a_path_string(repo, out):
    kept = runs.keep(str(out), "sweep", repo=str(repo), date=DAY)
    assert isinstance(kept, Path)
    assert runs.read_source(str(kept))["run"] == RUN
