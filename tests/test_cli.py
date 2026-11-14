"""The command line, driven with a toy model that each test registers as a real entry point.

The toy project is a tin can: one module and its `.dist-info` in a temporary directory on the path, which is how
an installed project's `[project.entry-points."similitude.models"]` reaches `importlib.metadata`.
"""

import json
import os
import shutil
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

from similitude import checks, cli, design, runs

DATA = Path(__file__).parent / "data" / "trials"

TOY = '''\
"""A tin can: its wall must not dent, and the can must hold a litre."""

import math

from similitude.checks import Check
from similitude.trials import Evaluation

not_a_function = 3


def evaluate(d):
    radius, height, wall = (q.to("cm").magnitude for q in (d.body.radius, d.body.height, d.body.wall))
    area = 2 * math.pi * radius * (radius + height)
    least = d.steel.min_wall.to("mm").magnitude
    return Evaluation(
        metrics={
            "volume_cm3": math.pi * radius**2 * height,
            "mass_g": area * wall * d.steel.density.to("g/cm^3").magnitude,
        },
        checks=[
            Check("Can", "wall", d.body.wall.to("mm").magnitude, "mm", limit=least, op=">=", warn_limit=1.1 * least),
            Check("Can", "surface", area, "cm^2"),
        ],
    )


def raises(d):
    raise ZeroDivisionError("the can has no lid")


def returns_a_list(d):
    return list(evaluate(d).checks)


def no_checks(d):
    return Evaluation({"volume_cm3": 1.0})
'''

TRIAL = """\
[trial]
name = "lighter-can"
design = "can.toml"
population = 6
generations = 3
seed = 1

[[trial.gene]]
key = "body.radius"
min = 30
max = 80
step = 1
unit = "mm"

[[trial.gene]]
key = "body.height"
min = 50
max = 200
step = 1
unit = "mm"

[[trial.objective]]
metric = "mass_g"
goal = "min"

[[trial.constraint]]
metric = "volume_cm3"
min = 1000
"""


def git(repo, *args):
    env = os.environ | {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    cmd = ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com", *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout.strip()


@pytest.fixture
def register(tmp_path, monkeypatch):
    """Install the toy project with the given models: `register(can="toy_can:evaluate")`."""

    def install(**models):
        site = tmp_path / "site"
        info = site / "toy_can-0.1.dist-info"
        info.mkdir(parents=True, exist_ok=True)
        (site / "toy_can.py").write_text(TOY, encoding="utf-8")
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: toy-can\nVersion: 0.1\n", encoding="utf-8")
        lines = "".join(f"{name} = {target}\n" for name, target in models.items())
        (info / "entry_points.txt").write_text(f"[{cli.GROUP}]\n{lines}", encoding="utf-8")
        monkeypatch.syspath_prepend(str(site))
        return site

    yield install
    sys.modules.pop("toy_can", None)


@pytest.fixture
def can(register):
    """The toy project installed with its one model, `can`."""
    return register(can="toy_can:evaluate")


@pytest.fixture
def work(tmp_path, monkeypatch):
    """A working directory that is not a git checkout, holding the can's design."""
    here = tmp_path / "work"
    here.mkdir()
    shutil.copy(DATA / "can.toml", here / "can.toml")
    monkeypatch.chdir(here)
    return here


@pytest.fixture
def repo(work):
    """The working directory as a git checkout, with the design committed and `out/` ignored."""
    (work / ".gitignore").write_text("out/\n", encoding="utf-8")
    git(work, "init", "-q", "-b", "main")
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "a can")
    return work


def run(*argv):
    return cli.main([str(a) for a in argv])


# --- check: the exit code is the checks' ------------------------------------------------------------------------


def test_check_passes_with_exit_code_0(can, work, capsys):
    assert run("check", "can.toml") == 0
    out = capsys.readouterr().out
    assert "| PASS | Can | wall |" in out
    assert out.endswith("\n\n0 FAIL, 0 WARN, 1 PASS, 1 INFO\n")


def test_check_with_a_fail_exits_1(can, work, capsys):
    assert run("check", "can.toml", "--set", "body.wall=0.15 mm") == 1
    assert "| **FAIL** | Can | wall | 0.15 mm |" in capsys.readouterr().out


def test_a_warn_does_not_fail_the_check(can, work, capsys):
    assert run("check", "can.toml", "--set", "body.wall=0.21 mm") == 0
    assert "| WARN | Can | wall |" in capsys.readouterr().out


def test_the_exit_code_is_the_one_similitude_checks_gives(can, work, capsys):
    for wall in ("0.30 mm", "0.21 mm", "0.15 mm"):
        code = run("check", "can.toml", "--json", "--set", f"body.wall={wall}")
        assert code == checks.exit_code(checks.from_json(capsys.readouterr().out))


def test_check_json_prints_what_checks_to_json_writes(can, work, capsys):
    assert run("check", "can.toml", "--json") == 0
    found = checks.from_json(capsys.readouterr().out)
    assert [(c.name, c.status) for c in found] == [("wall", "PASS"), ("surface", "INFO")]


def test_a_model_with_no_checks_passes(register, work, capsys):
    register(can="toy_can:no_checks")
    assert run("check", "can.toml") == 0
    assert capsys.readouterr().out.endswith("\n0 FAIL, 0 WARN, 0 PASS, 0 INFO\n")


# --- check: --set -----------------------------------------------------------------------------------------------


def test_set_overrides_several_values(can, work, capsys):
    assert run("check", "can.toml", "--set", "body.radius=40 mm", "--set", "body.height=100 mm", "--json") == 0
    surface = checks.from_json(capsys.readouterr().out)[1]
    assert surface.value == pytest.approx(2 * 3.141592653589793 * 4 * (4 + 10))


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("3", 3),
        ("0.25", 0.25),
        ("true", True),
        ("[1.0, 2.0]", [1.0, 2.0]),
        ("{ref = 9.0, range = [7.0, 12.0]}", {"ref": 9.0, "range": [7.0, 12.0]}),
        ("0.25 mm", "0.25 mm"),
        ("pressed", "pressed"),
        ('"3"', "3"),
        ("", ""),
        ("a=b", "a=b"),
        ("1\nw = 2", "1\nw = 2"),
    ],
)
def test_a_set_value_is_toml_when_it_is_toml_and_text_otherwise(text, value):
    assert cli.parse_sets([f"some.key={text}"]) == {"some.key": value}
    assert type(cli.parse_sets([f"some.key={text}"])["some.key"]) is type(value)


@pytest.mark.parametrize("pair", ["body.wall", "=3", " =3"])
def test_set_without_a_key_and_a_value_is_an_error(can, work, capsys, pair):
    assert run("check", "can.toml", "--set", pair) == 2
    assert "--set takes key=value" in capsys.readouterr().err


def test_set_twice_for_one_key_is_an_error(can, work, capsys):
    assert run("check", "can.toml", "--set", "body.wall=0.2 mm", "--set", "body.wall=0.3 mm") == 2
    assert "--set gives body.wall twice" in capsys.readouterr().err


def test_set_follows_the_design_rules(can, work, capsys):
    assert run("check", "can.toml", "--set", "body.lid=pressed") == 2
    assert "override body.lid: no such design value" in capsys.readouterr().err
    assert run("check", "can.toml", "--set", "body.wall=0.25") == 2  # a bare number where the design has a unit
    assert "override body.wall" in capsys.readouterr().err


# --- check: what cannot run exits 2, never 1 ---------------------------------------------------------------------


def test_a_missing_design_exits_2(can, work, capsys):
    assert run("check", "no-such.toml") == 2
    captured = capsys.readouterr()
    assert "similitude: error:" in captured.err and "no-such.toml" in captured.err
    assert captured.out == ""


def test_a_design_that_is_not_toml_exits_2(can, work, capsys):
    (work / "bad.toml").write_text("body = [", encoding="utf-8")
    assert run("check", "bad.toml") == 2
    assert "bad.toml" in capsys.readouterr().err


def test_a_model_that_raises_exits_2_and_shows_where(register, work, capsys):
    register(can="toy_can:raises")
    assert run("check", "can.toml") == 2
    err = capsys.readouterr().err
    assert "the model raised on this design: ZeroDivisionError: the can has no lid" in err
    assert "Traceback" in err and "toy_can.py" in err


def test_a_model_that_returns_something_else_exits_2(register, work, capsys):
    register(can="toy_can:returns_a_list")
    assert run("check", "can.toml") == 2
    assert "must return a similitude.trials.Evaluation, not list" in capsys.readouterr().err


def test_a_command_line_argparse_refuses_exits_2(can, work, capsys):
    for argv in ([], ["check"], ["bake", "can.toml"], ["trial", "t.toml", "--workers", "many"]):
        with pytest.raises(SystemExit) as stopped:
            cli.main(argv)
        assert stopped.value.code == 2
    assert "usage: similitude" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--version"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.startswith("similitude ")


# --- models: registered through the similitude.models entry-point group -----------------------------------------


def test_the_toy_model_is_found_through_the_entry_point_group(can):
    assert [(e.name, e.value) for e in metadata.entry_points(group="similitude.models")] == [
        ("can", "toy_can:evaluate")
    ]
    assert list(cli.models()) == ["can"]
    assert cli.load_model().__module__ == "toy_can"


def test_no_registered_model_exits_2_and_says_how_to_register(work, capsys):
    assert run("check", "can.toml") == 2
    assert '[project.entry-points."similitude.models"]' in capsys.readouterr().err


def test_several_models_need_a_name(register, work, capsys):
    register(can="toy_can:evaluate", bare="toy_can:no_checks")
    assert run("check", "can.toml") == 2
    assert "2 models are registered; name one with --model: bare, can" in capsys.readouterr().err
    assert run("check", "can.toml", "--model", "can") == 0
    assert "| PASS | Can | wall |" in capsys.readouterr().out
    assert run("check", "can.toml", "--model", "bare") == 0
    assert "0 PASS" in capsys.readouterr().out


def test_an_unknown_model_name_lists_the_registered_ones(can, work, capsys):
    assert run("check", "can.toml", "--model", "tin") == 2
    assert "no model named 'tin'; the registered models are: can" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("no_such_module:evaluate", "could not be loaded: ModuleNotFoundError"),
        ("toy_can:no_such_function", "could not be loaded: AttributeError"),
        ("toy_can:not_a_function", "is a int, not a function"),
    ],
)
def test_a_model_that_cannot_be_loaded_exits_2(register, work, capsys, target, message):
    register(can=target)
    assert run("check", "can.toml") == 2
    err = capsys.readouterr().err
    assert f"model can ({target})" in err and message in err


# --- check --out: a run that can be kept ------------------------------------------------------------------------


def test_check_out_writes_the_start_record_and_the_checks(can, repo, capsys):
    assert run("check", "can.toml", "--set", "body.wall=0.25 mm", "--out", "out/can") == 0
    out = repo / "out" / "can"
    assert sorted(p.name for p in out.iterdir()) == ["START", "checks.json", "checks.md", "design-files", "design.toml"]
    start = tomllib.loads((out / "START").read_text(encoding="utf-8"))
    assert start["commit"] == git(repo, "rev-parse", "HEAD")
    assert start["command"] == "similitude check can.toml --set 'body.wall=0.25 mm' --out out/can"
    assert start["dirty"] is False
    assert [c.status for c in checks.from_json((out / "checks.json").read_text(encoding="utf-8"))] == ["PASS", "INFO"]
    assert (out / "checks.md").read_text(encoding="utf-8") == capsys.readouterr().out


def test_check_out_refuses_a_directory_that_holds_a_run(can, repo, capsys):
    assert run("check", "can.toml", "--out", "out/can") == 0
    before = (repo / "out" / "can" / "checks.json").read_bytes()
    assert run("check", "can.toml", "--set", "body.wall=0.15 mm", "--out", "out/can") == 2
    assert "start a run in a fresh output directory" in capsys.readouterr().err
    assert (repo / "out" / "can" / "checks.json").read_bytes() == before


def test_a_design_that_does_not_load_leaves_no_output_directory(can, repo):
    assert run("check", "can.toml", "--set", "body.lid=pressed", "--out", "out/can") == 2
    assert not (repo / "out").exists()


def test_outside_a_git_checkout_check_out_still_runs_and_says_nothing_was_recorded(can, work, capsys):
    assert run("check", "can.toml", "--out", "out/can") == 0
    assert "no START recorded" in capsys.readouterr().err
    assert sorted(p.name for p in (work / "out" / "can").iterdir()) == [
        "checks.json", "checks.md", "design-files", "design.toml",
    ]  # fmt: skip


# --- trial ------------------------------------------------------------------------------------------------------


def test_trial_runs_and_exits_0_when_the_best_candidate_passes(can, repo, capsys):
    (repo / "lighter-can.toml").write_text(TRIAL, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a trial")
    assert run("trial", "lighter-can.toml", "--workers", 1) == 0
    out = repo / "out" / "trials" / "lighter-can"
    assert sorted(p.name for p in out.iterdir()) == [
        "START", "best.toml", "checks.json", "checks.md", "design-files", "design.toml", "history.jsonl",
    ]  # fmt: skip
    assert tomllib.loads((out / "START").read_text(encoding="utf-8"))["command"] == (
        "similitude trial lighter-can.toml --workers 1"
    )

    records = [json.loads(line) for line in (out / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1 + 6 * 3  # the design as it stands, then three generations of six
    captured = capsys.readouterr()
    best = min(records[1:], key=lambda r: (r["fails"], r["violation"], r["score"]))
    assert best["fails"] == 0 and best["violation"] == 0
    assert best["metrics"]["mass_g"] <= records[0]["metrics"]["mass_g"]
    assert f"best: body.radius={best['genes']['body.radius']}, body.height={best['genes']['body.height']}" in (
        captured.out
    )
    assert f"mass_g = {best['metrics']['mass_g']:g}" in captured.out
    assert "0 FAIL, 0 WARN, constraint violation 0" in captured.out
    assert "generation 3/3" in captured.err  # progress goes to stderr, the result to stdout


def test_trial_exits_1_when_the_best_candidate_breaks_a_constraint(can, work, capsys):
    (work / "big-can.toml").write_text(TRIAL.replace("min = 1000", "min = 100000"), encoding="utf-8")
    assert run("trial", "big-can.toml", "--workers", 1, "--out", "out/big") == 1
    captured = capsys.readouterr()
    assert "constraint violation" in captured.out and "constraint violation 0\n" not in captured.out
    assert (work / "out" / "big" / "best.toml").is_file()


def test_trial_exits_1_when_the_best_candidate_has_a_fail(can, work, capsys):
    thin = (work / "can.toml").read_text(encoding="utf-8").replace('wall = "0.30 mm"', 'wall = "0.15 mm"')
    (work / "can.toml").write_text(thin, encoding="utf-8")
    (work / "thin-can.toml").write_text(TRIAL, encoding="utf-8")
    assert run("trial", "thin-can.toml", "--workers", 1) == 1
    assert "1 FAIL, 0 WARN" in capsys.readouterr().out


def test_trial_scores_candidates_in_worker_processes(can, work, capsys):
    (work / "lighter-can.toml").write_text(TRIAL, encoding="utf-8")
    assert run("trial", "lighter-can.toml", "--workers", 1, "--out", "out/one") == 0
    one = capsys.readouterr().out.splitlines()[:4]
    assert run("trial", "lighter-can.toml", "--workers", 2, "--out", "out/two") == 0
    assert capsys.readouterr().out.splitlines()[:4] == one  # the same answer, however many workers


def test_trial_refuses_a_directory_that_holds_a_run(can, repo, capsys):
    (repo / "lighter-can.toml").write_text(TRIAL, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a trial")
    assert run("trial", "lighter-can.toml", "--workers", 1) == 0
    assert run("trial", "lighter-can.toml", "--workers", 1) == 2
    assert "start a run in a fresh output directory" in capsys.readouterr().err


def test_a_bad_trial_file_exits_2(can, work, capsys):
    (work / "bad.toml").write_text(TRIAL.replace('key = "body.height"', 'key = "body.lid"'), encoding="utf-8")
    assert run("trial", "bad.toml") == 2
    assert "bad.toml" in capsys.readouterr().err
    assert not (work / "out").exists()
    assert run("trial", "no-such.toml") == 2


def test_a_trial_metric_the_model_does_not_return_exits_2(can, work, capsys):
    (work / "odd.toml").write_text(TRIAL.replace('metric = "mass_g"', 'metric = "cost"'), encoding="utf-8")
    assert run("trial", "odd.toml", "--workers", 1) == 2
    assert "evaluate returned no metric cost" in capsys.readouterr().err


# --- keep -------------------------------------------------------------------------------------------------------


def kept(repo, name):
    (found,) = (repo / "runs").glob(f"*_{name}")
    return found


def test_keep_keeps_a_checked_run_with_the_source_recorded_at_its_start(can, repo, capsys):
    assert run("check", "can.toml", "--out", "out/can") == 0
    capsys.readouterr()
    assert run("keep", "out/can", "first-can", "--shows", "the can passes") == 0
    run_dir = kept(repo, "first-can")
    assert capsys.readouterr().out.strip() == os.path.join("runs", run_dir.name)
    source = runs.read_source(run_dir)
    assert source["recorded"] == "at run start"
    assert source["commit"] == git(repo, "rev-parse", "HEAD")
    assert source["command"] == "similitude check can.toml --out out/can"
    assert (run_dir / "design-files" / "can.toml").read_bytes() == (repo / "can.toml").read_bytes()
    assert design.load(run_dir / "design.toml") == design.load(repo / "can.toml")
    assert "the can passes" in (repo / "runs" / "README.md").read_text(encoding="utf-8")


def test_keep_refuses_a_run_from_a_dirty_tree_unless_told(can, repo, capsys):
    (repo / "can.toml").write_text((repo / "can.toml").read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert run("check", "can.toml", "--out", "out/can") == 0
    assert run("keep", "out/can", "dirty-can") == 2
    err = capsys.readouterr().err
    assert "uncommitted changes" in err and "pass --allow-dirty to keep" in err and "allow_dirty" not in err
    assert not (repo / "runs").exists()
    assert run("keep", "out/can", "dirty-can", "--allow-dirty") == 0
    assert runs.read_source(kept(repo, "dirty-can"))["dirty"] is True


def test_keep_does_not_record_itself_as_the_command_that_made_a_run(can, repo, capsys):
    for name in ("a", "b"):
        out = repo / "out" / name
        out.mkdir(parents=True)
        (out / "checks.json").write_text(checks.to_json([]), encoding="utf-8")
    assert run("keep", "out/a", "by-hand", "--design", "can.toml") == 0
    source = runs.read_source(kept(repo, "by-hand"))
    assert source["recorded"] == "when kept" and source["command"] == "(not recorded)"
    assert run("keep", "out/b", "by-script", "--design", "can.toml", "--command", "python sweep.py") == 0
    assert runs.read_source(kept(repo, "by-script"))["command"] == "python sweep.py"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["out/can", "second", "--command", "python sweep.py"], "don't pass --command"),
        (["out/can", "second", "--design", "can.toml"], "already has a design.toml, and --design names another file"),
        (["out/bare", "second"], "or pass --design"),
        (["out/can", "a name"], "a run's name is letters, digits"),
        (["out/none", "second"], "is not an output directory"),
        (["out/can", "first"], "is already kept"),
    ],
)
def test_keep_exits_2_for_a_run_that_may_not_be_kept(can, repo, capsys, argv, message):
    assert run("check", "can.toml", "--out", "out/can") == 0
    assert run("keep", "out/can", "first") == 0
    (repo / "out" / "bare").mkdir()
    capsys.readouterr()
    assert run("keep", *argv) == 2
    captured = capsys.readouterr()
    assert message in captured.err and "Traceback" not in captured.err
    assert captured.out == ""
    assert len(list((repo / "runs").glob("*_*"))) == 1


# --- the design a run used goes with it -------------------------------------------------------------------------


def reproduces(run_dir, model="toy_can:evaluate"):
    """Whether the design a run holds, loaded and evaluated again, gives the checks the run recorded."""
    import importlib

    module, _, function = model.partition(":")
    again = getattr(importlib.import_module(module), function)(design.load(run_dir / "design.toml")).checks
    recorded = checks.from_json((run_dir / "checks.json").read_text(encoding="utf-8"))
    return [c.to_dict() | {"disposition": ""} for c in recorded if c.group == "Can"] == [c.to_dict() for c in again]


def test_a_kept_run_holds_the_design_that_was_checked_with_its_set_values(can, repo, capsys):
    assert run("check", "can.toml", "--set", "body.wall=0.25 mm", "--set", "body.height=120 mm", "--out", "out/c") == 0
    assert run("keep", "out/c", "thicker") == 0
    run_dir = kept(repo, "thicker")
    archived = design.load(run_dir / "design.toml")
    assert archived == design.load(repo / "can.toml", {"body.wall": "0.25 mm", "body.height": "120 mm"})
    assert archived != design.load(repo / "can.toml")
    assert archived.body.wall.to("mm").magnitude == pytest.approx(0.25)
    assert reproduces(run_dir)
    wall = checks.from_json((run_dir / "checks.json").read_text(encoding="utf-8"))[0]
    assert wall.value == pytest.approx(0.25)


def test_keep_refuses_another_design_for_a_run_that_holds_its_own(can, repo, capsys):
    assert run("check", "can.toml", "--set", "body.wall=0.25 mm", "--out", "out/c") == 0
    assert run("keep", "out/c", "thicker", "--design", "can.toml") == 2
    assert "already has a design.toml, and --design names another file" in capsys.readouterr().err
    assert not (repo / "runs").exists()


@pytest.mark.parametrize(
    "value",
    ["0.25 mm", '{value = 0.25, unit = "mm"}', "250 um"],
)
def test_set_values_of_every_kind_reach_the_kept_design(can, repo, value):
    assert run("check", "can.toml", "--set", f"body.wall={value}", "--out", "out/c") == 0
    assert design.load(repo / "out" / "c" / "design.toml").body.wall.to("mm").magnitude == pytest.approx(0.25)
    assert reproduces(repo / "out" / "c")


def test_a_design_with_includes_and_its_own_overrides_is_kept_whole(can, repo, capsys):
    (repo / "shared").mkdir()
    (repo / "designs" / "cans").mkdir(parents=True)
    (repo / "shared" / "steel.toml").write_text(
        '[steel]\ndensity = "7.85 g/cm^3"\nmin_wall = "0.20 mm"\n', encoding="utf-8"
    )
    (repo / "designs" / "body.toml").write_text(
        'include = "../shared/steel.toml"\n\n[body]\nradius = "50 mm"\nheight = "140 mm"\nwall = "0.30 mm"\n',
        encoding="utf-8",
    )
    (repo / "designs" / "cans" / "tall.toml").write_text(
        'include = ["../body.toml"]\nname = "tall"\n\n[overrides]\n"body.height" = "180 mm"\n', encoding="utf-8"
    )
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a tall can")
    assert run("check", "designs/cans/tall.toml", "--set", "body.wall=0.25 mm", "--out", "out/tall") == 0
    assert run("keep", "out/tall", "tall") == 0
    run_dir = kept(repo, "tall")
    assert sorted(p.relative_to(run_dir).as_posix() for p in (run_dir / "design-files").rglob("*.toml")) == [
        "design-files/designs/body.toml",
        "design-files/designs/cans/tall.toml",
        "design-files/shared/steel.toml",
    ]
    shutil.rmtree(repo / "designs")  # the kept run does not lean on the files it was made from
    shutil.rmtree(repo / "shared")
    archived = design.load(run_dir / "design.toml")
    assert archived.name == "tall"
    assert [q.to("mm").magnitude for q in (archived.body.height, archived.body.wall)] == pytest.approx([180, 0.25])
    assert reproduces(run_dir)


def test_a_design_that_includes_a_file_by_an_absolute_path_cannot_go_into_a_run(can, repo, capsys):
    (repo / "steel.toml").write_text('[steel]\ndensity = "7.85 g/cm^3"\nmin_wall = "0.20 mm"\n', encoding="utf-8")
    (repo / "abs.toml").write_text(
        f'include = {json.dumps(str(repo / "steel.toml"))}\n[body]\nradius = "50 mm"\nheight = "140 mm"\n'
        'wall = "0.30 mm"\n',
        encoding="utf-8",
    )
    assert run("check", "abs.toml") == 0  # with no output directory there is nothing to copy
    assert run("check", "abs.toml", "--out", "out/abs") == 2
    assert "by an absolute path" in capsys.readouterr().err
    assert not (repo / "out" / "abs" / "checks.json").exists()


def test_a_run_stops_if_its_design_toml_is_not_the_design_asked_for(can, work, capsys, monkeypatch):
    as_toml = cli._toml
    monkeypatch.setattr(cli, "_toml", lambda value: as_toml("0.26 mm" if value == "0.25 mm" else value))
    assert run("check", "can.toml", "--set", "body.wall=0.25 mm", "--out", "out/can") == 2
    assert "does not load as the design this run was asked for" in capsys.readouterr().err
    assert not (work / "out" / "can" / "checks.json").exists()  # the model never ran


def test_a_design_with_a_nan_cannot_be_shown_to_be_the_one_asked_for(can, work, capsys):
    (work / "can.toml").write_text((work / "can.toml").read_text(encoding="utf-8") + "\n[lid]\nfudge = nan\n", "utf-8")
    assert run("check", "can.toml") == 0
    assert run("check", "can.toml", "--out", "out/can") == 2
    assert "A design with a NaN in it never compares equal to itself" in capsys.readouterr().err


def test_a_directory_that_holds_a_design_is_not_a_fresh_one(can, work, capsys):
    (work / "out" / "can").mkdir(parents=True)
    (work / "out" / "can" / "design.toml").write_text("", encoding="utf-8")
    assert run("check", "can.toml", "--out", "out/can") == 2
    assert "start a run in a fresh output directory" in capsys.readouterr().err


# --- a trial's winner goes into the run as the checked design does ----------------------------------------------


def trial_file(folder, name="lighter-can", **changes):
    text = TRIAL
    for old, new in changes.values():
        assert old in text
        text = text.replace(old, new)
    (folder / f"{name}.toml").write_text(text, encoding="utf-8")
    return f"{name}.toml"


def test_a_trial_writes_its_winners_checks_and_design(can, repo, capsys):
    trial = trial_file(repo)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a trial")
    assert run("trial", trial, "--workers", 1) == 0
    out = repo / "out" / "trials" / "lighter-can"
    records = [json.loads(line) for line in (out / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    best = min(records[1:], key=lambda r: (r["fails"], r["violation"], r["score"]))
    written = checks.from_json((out / "checks.json").read_text(encoding="utf-8"))
    assert [c.to_dict() for c in written if c.group == "Can"] == best["checks"]
    assert [(c.group, c.name, c.status) for c in written if c.group != "Can"] == [
        ("Trial constraint", "volume_cm3 (min)", "PASS")
    ]
    winner = design.load(out / "design.toml")
    assert winner == design.load(repo / "can.toml", best["overrides"])
    assert reproduces(out)

    assert run("keep", str(out), "lighter") == 0
    run_dir = kept(repo, "lighter")
    assert design.load(run_dir / "design.toml") == winner
    assert design.load(run_dir / "best.toml") == winner  # it includes the copy in the run, not the file outside
    assert reproduces(run_dir)


def test_a_trial_whose_winner_has_a_warn_is_kept_only_once_the_warn_is_disposed(can, repo, capsys):
    thin = (repo / "can.toml").read_text(encoding="utf-8").replace('wall = "0.30 mm"', 'wall = "0.21 mm"')
    (repo / "can.toml").write_text(thin, encoding="utf-8")
    trial = trial_file(repo)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a thin can and a trial")
    assert run("trial", trial, "--workers", 1, "--out", "out/warn") == 0  # a WARN does not fail a trial
    assert "0 FAIL, 1 WARN" in capsys.readouterr().out
    results = repo / "out" / "warn" / "checks.json"
    found = checks.from_json(results.read_text(encoding="utf-8"))
    assert [c.status for c in found] == ["WARN", "INFO", "PASS"]

    assert run("keep", "out/warn", "warned") == 2
    assert "record a disposition in checks.json for each WARN before keeping: Can: wall" in capsys.readouterr().err
    assert not (repo / "runs").exists()

    found[0] = found[0].disposed("the cans ship in trays, so nothing presses on the wall")
    results.write_text(checks.to_json(found), encoding="utf-8")
    assert run("keep", "out/warn", "warned") == 0
    assert "the cans ship in trays" in (kept(repo, "warned") / "checks.json").read_text(encoding="utf-8")


def test_a_trials_exit_code_is_that_of_the_checks_it_writes(can, work, capsys):
    cases = {
        "passes": {},
        "too-big": {"volume": ("min = 1000", "min = 100000")},
        "too-small": {"volume": ("min = 1000", "max = 10")},
    }
    for name, changes in cases.items():
        code = run("trial", trial_file(work, name, **changes), "--workers", 1, "--out", f"out/{name}")
        written = checks.from_json((work / "out" / name / "checks.json").read_text(encoding="utf-8"))
        assert code == checks.exit_code(written) == (0 if name == "passes" else 1)
        if code:
            (broken,) = [c for c in written if c.status == "FAIL"]
            assert broken.group == "Trial constraint" and broken.name.startswith("volume_cm3 (")


def test_a_trial_in_which_nothing_could_be_evaluated_says_so_in_its_checks(register, work, capsys):
    register(can="toy_can:raises")
    assert run("trial", trial_file(work), "--workers", 1, "--out", "out/none") == 1
    assert "could not be evaluated: ZeroDivisionError: the can has no lid" in capsys.readouterr().out
    (only,) = checks.from_json((work / "out" / "none" / "checks.json").read_text(encoding="utf-8"))
    assert (only.group, only.name, only.status) == ("Trial", "the best candidate could be evaluated", "FAIL")
    assert only.reason == "ZeroDivisionError: the can has no lid"
    tried = tomllib.loads((work / "out" / "none" / "design.toml").read_text(encoding="utf-8"))
    assert tried["include"] == "design-files/can.toml" and set(tried["overrides"]) == {"body.radius", "body.height"}


# --- as a program -----------------------------------------------------------------------------------------------


def test_the_similitude_command_is_installed():
    (script,) = metadata.entry_points(group="console_scripts", name="similitude")
    assert script.value == "similitude.cli:main"
    assert script.load() is cli.main


def test_exit_codes_as_a_program(can, work):
    env = os.environ | {"PYTHONPATH": str(can)}

    def program(*argv):
        return subprocess.run([sys.executable, "-m", "similitude.cli", *argv], capture_output=True, text=True, env=env)

    passed = program("check", "can.toml")
    assert passed.returncode == 0 and "| PASS | Can | wall |" in passed.stdout and passed.stderr == ""
    failed = program("check", "can.toml", "--set", "body.wall=0.15 mm")
    assert failed.returncode == 1 and "| **FAIL** | Can | wall |" in failed.stdout
    broken = program("check", "can.toml", "--set", "body.lid=pressed")
    assert broken.returncode == 2 and broken.stdout == "" and "similitude: error:" in broken.stderr
    assert program().returncode == 2
