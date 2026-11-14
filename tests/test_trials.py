"""Trials: a seeded genetic algorithm finds a known optimum, repeats itself, and never lets a FAIL win.

The toy problem is a tin can that must hold a litre. For a fixed volume V, a closed cylinder has the least
surface when its height equals its diameter: r = (V / 2π)^(1/3), h = 2r, area = 6πr². The thinnest sheet on offer
would be lighter still, but it fails the can's denting check, so it must not win.
"""

import dataclasses
import json
import math
import shutil
import tomllib
from pathlib import Path

import pytest

from similitude import design, trials
from similitude.checks import Check
from similitude.trials import Constraint, Evaluation, Gene, Objective, Trial, TrialError
from similitude.trials import engine
from similitude.trials.engine import _Rng

DATA = Path(__file__).parent / "data" / "trials"

LITRE = 1000.0  # cm^3
STEEL = 7.85  # g/cm^3
MIN_WALL = 0.02  # cm: the thinnest sheet that passes the denting check
BEST_RADIUS = (LITRE / (2 * math.pi)) ** (1 / 3)  # cm
BEST_MASS = 6 * math.pi * BEST_RADIUS**2 * MIN_WALL * STEEL  # g: 86.9 g, the least any passing can could weigh


# --- the project's side: these score a can. They are at the top level so worker processes can import them. ------


def can(d):
    radius, height, wall = (q.to("cm").magnitude for q in (d.body.radius, d.body.height, d.body.wall))
    volume = math.pi * radius**2 * height
    area = 2 * math.pi * radius * (radius + height)
    return Evaluation(
        metrics={"volume_cm3": volume, "area_cm2": area, "mass_g": area * wall * d.steel.density.to("g/cm^3").magnitude},
        checks=[
            Check("Can", "wall is thick enough not to dent", d.body.wall.to("mm").magnitude, "mm",
                  limit=d.steel.min_wall.to("mm").magnitude, op=">=", reason="thinner walls dent in handling"),
            Check("Can", "surface", area, "cm^2"),
        ],
    )  # fmt: skip


def can_by_radius(d):
    return Evaluation({"radius_mm": d.body.radius.to("mm").magnitude})


def off_five(d):
    return Evaluation({"off": abs(d.x - 5)})


def off_a_fifth(d):
    return Evaluation({"off": abs(d.y - 0.2)})


def solver_that_fails(d):
    raise ValueError("solver failed\nresidual did not converge \x1b[31mat step 3\x1b[0m\r\n\nsee the log")


def can_that_cannot_be_tall(d):
    if d.body.height.to("mm").magnitude > 120:
        raise ValueError("too tall for the press")
    return can(d)


def can_without_mass(d):
    return Evaluation({"volume_cm3": 1.0})


def can_as_a_dict(d):
    return {"metrics": {}}


def can_with_a_quantity(d):
    return Evaluation({"mass_g": d.body.radius, "volume_cm3": 1.0})


def can_with_nan(d):
    return Evaluation({"mass_g": math.nan, "volume_cm3": 1.0})


# --- helpers ----------------------------------------------------------------------------------------------------


@pytest.fixture
def here(tmp_path):
    """A project folder: the trial and its design in `trials/`, with results to go in `out/`."""
    (tmp_path / "pyproject.toml").write_text("")  # what makes it a project
    shutil.copytree(DATA, tmp_path / "trials")
    return tmp_path


@pytest.fixture
def trial(here):
    return trials.load(here / "trials" / "lighter-can.toml")


@pytest.fixture
def small(trial):
    """The same search, cut short: enough to compare runs, not to find the optimum."""
    return dataclasses.replace(trial, population=8, generations=4)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def trial_file(folder, body, design_text='[body]\nradius = "50 mm"\nseats = 2\nspeed = { ref = 9.0, range = [7.0, 12.0] }\n'):
    (folder / "d.toml").write_text(design_text)
    path = folder / "t.toml"
    path.write_text(body)
    return path


GENE = '[[trial.gene]]\nkey = "body.radius"\nmin = 30\nmax = 80\nunit = "mm"\n'
HEAD = '[trial]\nname = "t"\ndesign = "d.toml"\n'


# --- the trial file ---------------------------------------------------------------------------------------------


def test_a_trial_file_loads_with_its_genes_objectives_and_constraints(trial, here):
    assert (trial.name, trial.population, trial.generations, trial.elite, trial.seed) == ("lighter-can", 24, 40, 2, 1)
    assert trial.design == here / "trials" / "can.toml"  # relative to the trial file
    assert trial.question.startswith("How light")
    assert [g.key for g in trial.genes] == ["body.radius", "body.height", "body.wall"]
    radius, _, wall = trial.genes
    assert (radius.lo, radius.hi, radius.step, radius.unit) == (30, 80, 1, "mm")
    assert radius.why.startswith("the can's proportions")
    assert wall.choices == (0.15, 0.2, 0.25, 0.3)
    assert trial.objectives == [Objective("mass_g", "min")]
    assert trial.constraints == [Constraint("volume_cm3", lo=1000)]
    assert trial.metrics() == ["mass_g", "volume_cm3"]


@pytest.mark.parametrize(
    "body, complaint",
    [
        ("[trail]\n", r"one \[trial\] table"),
        ('[trial]\nname = "t"\n' + GENE, "needs a design"),
        (HEAD, "no genes"),
        (HEAD + "populaton = 4\n" + GENE, "no setting populaton"),
        (HEAD + "population = 1\n" + GENE, "population must be at least 2"),
        (HEAD + "elite = 12\n" + GENE, "elite must be"),
        (HEAD + "seed = 1.5\n" + GENE, "seed must be a whole number"),
        (HEAD + "crossover = 1.5\n" + GENE, "crossover is a chance"),
        (HEAD + GENE + GENE, "listed twice"),
        (HEAD + '[[trial.gene]]\nkey = "body.radius"\nmin = 30\nunit = "mm"\n', "give min and max, or choices"),
        (HEAD + '[[trial.gene]]\nkey = "body.radius"\nmin = 80\nmax = 30\nunit = "mm"\n', "min must be below max"),
        (HEAD + '[[trial.gene]]\nkey = "body.radius"\nmin = 30\nmax = 80\nstep = 0\nunit = "mm"\n', "step must be"),
        (HEAD + '[[trial.gene]]\nkey = "body.radius"\nmin = 30\nmax = 80\nchoices = [40]\n', "not both"),
        (HEAD + '[[trial.gene]]\nkey = "body.radius"\nchoices = []\n', "choices is empty"),
        (HEAD + '[[trial.gene]]\nkey = "body.radius"\nchoices = [40, 40]\nunit = "mm"\n', "listed twice"),
        (HEAD + '[[trial.gene]]\nkey = "body.radius"\nchoices = ["a", "b"]\nunit = "mm"\n', "choices must be numbers"),
        (HEAD + '[[trial.gene]]\nkey = "body.radius"\nmin = 30\nmax = 80\nunit = "mn"\n', "unknown unit 'mn'"),
        (HEAD + '[[trial.gene]]\nkey = "body.radius"\nmin = 30\nmax = 80\nunit = "mm"\nwhy_not = ""\n', "no setting why_not"),
        (HEAD + GENE + '[[trial.objective]]\nmetric = "mass_g"\n', "needs a metric and a goal"),
        (HEAD + GENE + '[[trial.objective]]\nmetric = "mass_g"\ngoal = "least"\n', 'goal must be "min" or "max"'),
        (HEAD + GENE + '[[trial.objective]]\nmetric = "mass_g"\ngoal = "min"\nscale = 0\n', "scale a number above zero"),
        (HEAD + GENE + '[[trial.constraint]]\nmetric = "volume_cm3"\n', "give min, max or both"),
        (HEAD + GENE + '[[trial.constraint]]\nmetric = "volume_cm3"\nmin = 2\nmax = 1\n', "min is above max"),
        (HEAD + "= 3\n", "Invalid statement"),  # not TOML
    ],
)
def test_a_malformed_trial_file_is_an_error_naming_the_file(tmp_path, body, complaint):
    with pytest.raises(TrialError, match=f"t.toml: .*{complaint}"):
        trials.load(trial_file(tmp_path, body))


@pytest.mark.parametrize(
    "gene, complaint",
    [
        ('key = "body.radus"\nmin = 30\nmax = 80\nunit = "mm"', "gene body.radus: .*no such design value"),
        ('key = "body.radius"\nmin = 30\nmax = 80', "gene body.radius: .*needs a unit"),  # the design value has one
        ('key = "body.radius"\nmin = 30\nmax = 80\nunit = "s"', "gene body.radius: .*second does not convert"),
        ('key = "body.seats"\nmin = 1\nmax = 4\nunit = "mm"', "gene body.seats: .*has no unit"),
        ('key = "body.speed"\nmin = 8\nmax = 11', "gene body.speed: .*this is a range"),  # a range keeps its bounds
        ('key = "body.speed.ref"\nmin = 5\nmax = 11', "gene body.speed.ref: .*does not contain ref 5"),
    ],
)
def test_a_gene_that_does_not_fit_the_design_is_an_error_when_the_trial_loads(tmp_path, gene, complaint):
    with pytest.raises(TrialError, match=complaint):
        trials.load(trial_file(tmp_path, HEAD + f"[[trial.gene]]\n{gene}\n"))


def test_a_gene_on_an_uncertain_value_names_its_ref_and_stays_inside_its_bounds(tmp_path):
    t = trials.load(trial_file(tmp_path, HEAD + '[[trial.gene]]\nkey = "body.speed.ref"\nmin = 8\nmax = 11\n'))
    assert t.genes[0].key == "body.speed.ref"


def test_a_missing_design_is_an_error(tmp_path):
    path = trial_file(tmp_path, '[trial]\nname = "t"\ndesign = "nowhere.toml"\n' + GENE)
    with pytest.raises(TrialError, match="design nowhere.toml"):
        trials.load(path)


# --- genes ------------------------------------------------------------------------------------------------------


def test_genes_stay_in_range_and_on_step():
    rng = _Rng(0)
    whole = Gene("x", 400, 2000, 10)
    fine = Gene("y", 0.1, 0.9, 0.05)
    free = Gene("z", -1.0, 1.0)
    for _ in range(300):
        x = whole.mutate(whole.sample(rng), rng, 0.5)
        assert isinstance(x, int) and 400 <= x <= 2000 and (x - 400) % 10 == 0
        y = fine.mutate(fine.sample(rng), rng, 0.5)
        assert 0.1 <= y <= 0.9 and round((y - 0.1) / 0.05, 9) == round((y - 0.1) / 0.05)
        assert y == round(y, 2)  # no float noise such as 0.15000000000000002
        z = free.mutate(free.sample(rng), rng, 0.5)
        assert isinstance(z, float) and -1.0 <= z <= 1.0


def test_a_step_that_does_not_divide_the_range_never_passes_max():
    rng = _Rng(0)
    gene = Gene("x", 0, 10, 4)
    assert {gene.sample(rng) for _ in range(200)} == {0, 4, 8}


def test_a_gene_contains_only_values_in_range_and_on_a_step():
    whole = Gene("x", 0, 10, 4)
    assert [v for v in range(-1, 12) if whole.contains(v)] == [0, 4, 8]  # 10 is in range, but past the last step
    fine = Gene("y", 0.1, 0.9, 0.05)
    assert fine.contains(0.15) and fine.contains(0.1) and fine.contains(0.9)
    assert fine.contains(0.1 + 4 * 0.05)  # 0.30000000000000004: float noise is not a step off
    assert not fine.contains(0.12) and not fine.contains(0.95) and not fine.contains(0.05)
    free = Gene("z", -1.0, 1.0)
    assert free.contains(0.123456) and not free.contains(1.5)
    assert not whole.contains("4") and not whole.contains(True) and not whole.contains(None)


def test_a_choice_gene_picks_from_its_choices_and_mutates_to_another():
    rng = _Rng(0)
    gene = Gene("x", choices=["a", "b", "c"])
    assert {gene.sample(rng) for _ in range(100)} == {"a", "b", "c"}
    assert all(gene.mutate("a", rng, 0.2) in ("b", "c") for _ in range(50))
    assert Gene("x", choices=[True]).mutate(True, rng, 0.2) is True  # nothing else to pick


def test_a_gene_with_a_unit_overrides_with_that_unit():
    assert Gene("x", 30, 80, 1, unit="mm").override(54) == "54 mm"
    assert Gene("x", 0.1, 0.9, unit="kg/m^3").override(0.25) == "0.25 kg/m^3"
    assert Gene("x", choices=["a", "b"]).override("a") == "a"
    assert Gene("x", 1, 4, 1).override(3) == 3


def test_the_random_draws_are_the_same_for_the_same_seed_and_cover_what_they_should():
    assert [_Rng(7).random() for _ in range(3)] == [_Rng(7).random() for _ in range(3)]
    rng = _Rng(7)
    assert {rng.below(4) for _ in range(200)} == {0, 1, 2, 3}
    draws = [rng.normal(2.0) for _ in range(4000)]
    assert abs(sum(draws) / len(draws)) < 0.15  # mean 0
    assert math.sqrt(sum(d * d for d in draws) / len(draws)) == pytest.approx(2.0, rel=0.05)  # sigma 2
    for _ in range(100):
        picked = rng.distinct(5, 3)
        assert len(set(picked)) == 3 and set(picked) <= set(range(5))


# --- ranking ----------------------------------------------------------------------------------------------------


def test_a_candidate_with_a_fail_loses_to_every_candidate_without_one(trial):
    light_but_fails = {"metrics": {"mass_g": 1.0, "volume_cm3": 2000.0}, "fails": 1}
    heavy = {"metrics": {"mass_g": 900.0, "volume_cm3": 2000.0}, "fails": 0}
    heavy_and_too_small = {"metrics": {"mass_g": 900.0, "volume_cm3": 10.0}, "fails": 0}
    fails_twice = {"metrics": {"mass_g": 0.5, "volume_cm3": 2000.0}, "fails": 2}
    broken = {"error": "ValueError: could not be built"}
    ranked = sorted([broken, fails_twice, light_but_fails, heavy_and_too_small, heavy], key=lambda r: trials.rank_key(trial, r))
    assert ranked == [heavy, heavy_and_too_small, light_but_fails, fails_twice, broken]


def test_feasible_candidates_rank_first_and_then_the_lower_score(trial):
    ok = {"metrics": {"mass_g": 100.0, "volume_cm3": 1000.0}, "fails": 0}
    lighter = {"metrics": {"mass_g": 90.0, "volume_cm3": 1000.0}, "fails": 0}
    lighter_but_too_small = {"metrics": {"mass_g": 50.0, "volume_cm3": 990.0}, "fails": 0}
    assert trials.violation(trial, ok) == 0 and trials.violation(trial, lighter_but_too_small) == 10.0
    assert trials.score(trial, lighter) == 90.0
    ranked = sorted([lighter_but_too_small, ok, lighter], key=lambda r: trials.rank_key(trial, r))
    assert ranked == [lighter, ok, lighter_but_too_small]


def test_the_score_weighs_each_objective_by_its_scale_and_negates_a_max(tmp_path):
    t = Trial("t", tmp_path / "d.toml", [Gene("x", 0, 1)], [Objective("grip", "max", 3, 1), Objective("mass", "min", 1, 10)],
              [Constraint("width", lo=1, hi=2)])  # fmt: skip
    assert trials.score(t, {"metrics": {"grip": 12.0, "mass": 250.0, "width": 1.5}}) == -3 * 12.0 + 250.0 / 10
    assert trials.violation(t, {"metrics": {"grip": 0, "mass": 0, "width": 2.5}}) == 0.5
    assert trials.violation(t, {"metrics": {"grip": 0, "mass": 0, "width": 0.25}}) == 0.75


# --- the search -------------------------------------------------------------------------------------------------


def test_the_search_finds_the_lightest_can_that_passes(trial, here):
    result = trials.run(trial, can, workers=1, out=here / "out")
    best = result.best
    # no can that passes its check and holds a litre can weigh less than the analytic optimum, and the search
    # gets within 2 % of it on a 1 mm grid
    assert BEST_MASS <= best["metrics"]["mass_g"] <= 1.02 * BEST_MASS
    assert best["metrics"]["volume_cm3"] >= LITRE and best["violation"] == 0
    assert best["genes"]["body.wall"] == 0.2 and best["fails"] == 0
    # near the optimum the can is about as tall as it is wide; the mass is flat there, so only roughly
    assert best["genes"]["body.height"] / (2 * best["genes"]["body.radius"]) == pytest.approx(1.0, abs=0.35)
    assert best["metrics"]["mass_g"] < result.reference["metrics"]["mass_g"]


def test_a_fail_candidate_never_wins(trial, here):
    result = trials.run(trial, can, workers=1, out=here / "out")
    thin = [r for r in result.individuals if r["genes"]["body.wall"] == 0.15]
    assert thin and all(r["fails"] == 1 for r in thin)
    # the trap is real: thin cans that hold a litre and weigh less than the winner were found, and lost
    assert any(r["violation"] == 0 and r["metrics"]["mass_g"] < result.best["metrics"]["mass_g"] for r in thin)
    assert result.best["fails"] == 0
    for gen in range(trial.generations):
        ranked = [r for r in result.individuals if r["generation"] == gen]
        fails = [r["fails"] for r in ranked]
        assert fails == sorted(fails)  # every FAIL candidate comes after every candidate without one
        assert [r["rank"] for r in ranked] == list(range(trial.population))


def test_when_every_candidate_fails_the_best_says_so(small, here):
    walls = Gene("body.wall", choices=[0.1, 0.15], unit="mm")
    doomed = dataclasses.replace(small, genes=[*small.genes[:2], walls])
    result = trials.run(doomed, can, workers=1, out=here / "out")
    assert result.best["fails"] == 1
    assert "It FAILS 1 check(s)" in result.best_toml.read_text()


def test_elites_carry_over_and_the_best_never_gets_worse(trial, here):
    result = trials.run(trial, can, workers=1, out=here / "out")
    tops = [next(r for r in result.individuals if r["generation"] == g) for g in range(trial.generations)]
    keys = [trials.rank_key(trial, r) for r in tops]
    assert keys == sorted(keys, reverse=True)
    for g in range(1, trial.generations):
        now = [r["genes"] for r in result.individuals if r["generation"] == g]
        before = [r["genes"] for r in result.individuals if r["generation"] == g - 1]
        assert all(genes in now for genes in before[: trial.elite])


def test_the_design_as_it_stands_starts_the_population_when_the_genes_cover_it(small, here):
    result = trials.run(small, can, workers=1, out=here / "out")
    assert result.reference["genes"] == {"body.radius": 50, "body.height": 140, "body.wall": 0.3}
    assert result.reference["generation"] is None
    assert result.reference["metrics"]["mass_g"] == pytest.approx(140.57, abs=0.01)  # 596.9 cm² × 0.03 cm × 7.85
    assert result.reference["genes"] in [r["genes"] for r in result.individuals if r["generation"] == 0]


def test_the_design_as_it_stands_is_read_in_the_genes_unit(tmp_path):
    path = trial_file(tmp_path, HEAD + '[[trial.gene]]\nkey = "body.radius"\nmin = 30\nmax = 80\nstep = 1\nunit = "mm"\n',
                      '[body]\nradius = "0.05 m"\n')  # fmt: skip
    result = trials.run(path, can_by_radius, workers=1, out=tmp_path / "out")
    assert result.reference["genes"] == {"body.radius": 50}  # not 50.00000000000001
    assert result.reference["overrides"] == {"body.radius": "50 mm"}


def test_a_design_off_a_whole_number_step_is_compared_but_not_bred(tmp_path):
    (tmp_path / "d.toml").write_text("x = 5\n")
    t = Trial("grid", tmp_path / "d.toml", [Gene("x", 0, 10, 4)], [Objective("off", "min")], population=4, generations=3)
    result = trials.run(t, off_five, workers=1, out=tmp_path / "out")
    assert result.reference["genes"] == {"x": 5} and result.reference["overrides"] == {}
    assert result.reference["score"] == 0  # the design as it stands would have won
    assert {r["genes"]["x"] for r in result.individuals} <= {0, 4, 8}
    assert result.best["genes"] == {"x": 4}  # the nearest value the search may use


def test_a_design_off_a_fractional_step_is_compared_but_not_bred(tmp_path):
    (tmp_path / "d.toml").write_text("y = 0.21\n")
    t = Trial("grid", tmp_path / "d.toml", [Gene("y", 0.1, 0.9, 0.05)], [Objective("off", "min")], population=4,
              generations=3)  # fmt: skip
    result = trials.run(t, off_a_fifth, workers=1, out=tmp_path / "out")
    assert result.reference["genes"] == {"y": 0.21} and result.reference["overrides"] == {}
    assert all(Gene("y", 0.1, 0.9, 0.05).contains(r["genes"]["y"]) for r in result.individuals)
    assert 0.21 not in {r["genes"]["y"] for r in result.individuals}


def test_a_design_on_a_fractional_step_starts_the_population(tmp_path):
    (tmp_path / "d.toml").write_text("y = 0.30000000000000004\n")  # 0.1 + 4 × 0.05, as a float sum gives it
    t = Trial("grid", tmp_path / "d.toml", [Gene("y", 0.1, 0.9, 0.05)], [Objective("off", "min")], population=4,
              generations=2)  # fmt: skip
    result = trials.run(t, off_a_fifth, workers=1, out=tmp_path / "out")
    assert result.reference["genes"] == {"y": 0.3} and result.reference["overrides"] == {"y": 0.3}
    assert {"y": 0.3} in [r["genes"] for r in result.individuals if r["generation"] == 0]


def test_a_design_outside_the_genes_is_compared_but_not_bred(small, here):
    wide = dataclasses.replace(small, genes=[Gene("body.radius", 60, 80, 1, unit="mm"), *small.genes[1:]])
    result = trials.run(wide, can, workers=1, out=here / "out")
    assert result.reference["overrides"] == {}  # scored as written
    assert result.reference["metrics"]["mass_g"] == pytest.approx(140.57, abs=0.01)
    assert all(r["genes"]["body.radius"] >= 60 for r in result.individuals)


# --- reproducibility --------------------------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_run_and_another_seed_a_different_one(small, here):
    first = trials.run(small, can, workers=1, out=here / "out" / "a")
    again = trials.run(small, can, workers=1, out=here / "out" / "b")
    other = trials.run(dataclasses.replace(small, seed=2), can, workers=1, out=here / "out" / "c")
    assert first.history.read_bytes() == again.history.read_bytes()
    assert first.best_toml.read_bytes() == again.best_toml.read_bytes()
    assert first.individuals == again.individuals
    assert [r["genes"] for r in first.individuals] != [r["genes"] for r in other.individuals]


def test_the_result_does_not_depend_on_the_worker_count(small, here):
    runs = [trials.run(small, can, workers=n, out=here / "out" / f"w{n}") for n in (1, 2, 3)]
    assert len({r.history.read_bytes() for r in runs}) == 1
    assert len({r.best_toml.read_bytes() for r in runs}) == 1
    assert runs[0].best == runs[1].best == runs[2].best


def test_a_run_from_a_trial_file_matches_a_run_from_the_loaded_trial(here):
    path = here / "trials" / "lighter-can.toml"
    from_file = trials.run(path, can, workers=1, out=here / "out" / "a")
    from_trial = trials.run(trials.load(path), can, workers=1, out=here / "out" / "b")
    assert from_file.history.read_bytes() == from_trial.history.read_bytes()


def test_an_evaluate_that_workers_cannot_import_is_an_error_unless_it_runs_here(small, here):
    with pytest.raises(TrialError, match="top level of a module"):
        trials.run(small, lambda d: can(d), workers=2, out=here / "out")
    assert trials.run(small, lambda d: can(d), workers=1, out=here / "out").best["fails"] == 0
    with pytest.raises(TrialError, match="workers must be"):
        trials.run(small, can, workers=0, out=here / "out")


# --- what is written --------------------------------------------------------------------------------------------


def test_the_history_is_jsonl_with_every_candidates_checks(small, here):
    result = trials.run(small, can, workers=1, out=here / "out")
    assert result.history == here / "out" / "history.jsonl"
    records = lines(result.history)
    assert len(records) == 1 + small.population * small.generations  # the design as it stands, then every candidate
    assert records[0] == result.reference and records[1:] == result.individuals
    for rec in records:
        assert set(rec) == {"generation", "rank", "genes", "overrides", "metrics", "fails", "warns", "checks",
                            "violation", "score"}  # fmt: skip
        dent, surface = rec["checks"]
        assert dent["name"] == "wall is thick enough not to dent" and dent["limit"] == 0.2
        assert dent["status"] == ("FAIL" if rec["genes"]["body.wall"] < 0.2 else "PASS")
        assert rec["fails"] == (dent["status"] == "FAIL")
        assert surface["status"] == "INFO" and surface["value"] == rec["metrics"]["area_cm2"]
        assert rec["score"] == rec["metrics"]["mass_g"]


def test_the_overrides_in_the_history_rebuild_the_candidate(small, here):
    result = trials.run(small, can, workers=1, out=here / "out")
    rec = result.individuals[-1]
    assert rec["overrides"]["body.radius"] == f"{rec['genes']['body.radius']} mm"
    again = can(design.load(small.design, overrides=rec["overrides"]))
    assert again.metrics == rec["metrics"]


def test_the_best_design_is_written_as_toml_with_its_overrides(small, here):
    result = trials.run(small, can, workers=1, out=here / "out")
    assert result.best_toml == here / "out" / "best.toml"
    text = result.best_toml.read_text()
    written = tomllib.loads(text)
    assert written["include"] == "../trials/can.toml"
    assert written["overrides"] == result.best["overrides"]
    best = design.load(result.best_toml)  # it is a design file: the trial's design plus the overrides
    assert best == design.load(small.design, overrides=result.best["overrides"])
    assert can(best).metrics == result.best["metrics"]
    assert '"body.wall" = "0.2 mm"' in text and "# the design has 0.3" in text
    assert "a trial is evidence, not a decision" in text


def project(folder, marker="pyproject.toml"):
    """A project at `folder`, holding the can's design. Returns the design's path."""
    (folder / "designs").mkdir(parents=True)
    (folder / marker).mkdir() if marker == ".git" else (folder / marker).write_text("")
    shutil.copy(DATA / "can.toml", folder / "designs" / "can.toml")
    return folder / "designs" / "can.toml"


@pytest.mark.parametrize("marker", ["pyproject.toml", ".git"])
def test_a_path_to_the_design_is_given_only_inside_the_designs_own_project(tmp_path, marker):
    """Everything here is built under one temporary folder, so it does not matter where that folder, or this
    checkout, is on the machine."""
    home = tmp_path / "home"
    can_toml = project(home / "example-person" / "boats", marker)
    inside = home / "example-person" / "boats" / "out" / "trials" / "lighter-can"
    assert engine._relative(can_toml, inside) == "../../../designs/can.toml"
    assert engine._relative(can_toml, home / "example-person" / "boats") == "designs/can.toml"
    # unrelated trees that share a user's or a system folder: no path between them
    assert engine._relative(can_toml, home / "shared-results" / "trial") == ""
    assert engine._relative(can_toml, home / "example-person" / "notes") == ""
    assert engine._relative(can_toml, home) == ""
    # another project is not this one either
    project(home / "example-person" / "planes")
    assert engine._relative(can_toml, home / "example-person" / "planes" / "out") == ""


def test_a_design_in_no_project_is_never_given_a_path(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "_PROJECT_MARKERS", (".no-folder-has-this",))  # whatever lies above tmp_path
    (tmp_path / "out").mkdir()
    (tmp_path / "can.toml").write_text("")
    assert engine._project(tmp_path / "can.toml") is None
    assert engine._relative(tmp_path / "can.toml", tmp_path / "out") == ""


def test_the_best_design_written_outside_the_project_holds_overrides_alone(small, tmp_path):
    can_toml = project(tmp_path / "home" / "example-person" / "boats")
    far = dataclasses.replace(small, design=can_toml)
    result = trials.run(far, can, workers=1, out=tmp_path / "home" / "shared-results" / "lighter-can")
    text = result.best_toml.read_text()
    written = tomllib.loads(text)
    assert "include" not in written and written["overrides"] == result.best["overrides"]
    assert "can.toml" in text  # it names the design
    assert "example-person" not in text and "boats" not in text and str(tmp_path) not in text  # but not where it is
    # written inside the project, the same run includes the design, by a path that stays inside it
    near = trials.run(far, can, workers=1, out=can_toml.parent.parent / "out" / "trials" / "lighter-can")
    assert tomllib.loads(near.best_toml.read_text())["include"] == "../../../designs/can.toml"
    assert design.load(near.best_toml) == design.load(can_toml, overrides=near.best["overrides"])


def test_results_go_to_out_trials_by_default(small, here, monkeypatch):
    monkeypatch.chdir(here)
    result = trials.run(small, can, workers=1)
    assert result.history == Path("out") / "trials" / "lighter-can" / "history.jsonl"
    assert result.history.exists() and result.best_toml.exists()


def test_progress_is_reported_to_the_log_when_one_is_given(small, here):
    said = []
    trials.run(small, can, workers=1, out=here / "out", log=said.append)
    assert said[0] == "reference: the design as it stands"
    assert [s for s in said if s.startswith("generation")] == [f"generation {g}/4" for g in (1, 2, 3, 4)]


# --- candidates that go wrong -----------------------------------------------------------------------------------


def test_a_candidate_that_cannot_be_evaluated_is_recorded_and_ranks_last(small, here):
    result = trials.run(small, can_that_cannot_be_tall, workers=1, out=here / "out")
    broken = [r for r in result.individuals if "error" in r]
    assert broken and all(r["genes"]["body.height"] > 120 for r in broken)
    assert broken[0]["error"] == "ValueError: too tall for the press"
    assert broken[0]["score"] is None and broken[0]["violation"] is None
    assert "error" in result.reference  # the design as it stands is 140 mm tall
    for gen in range(small.generations):
        ranked = ["error" in r for r in result.individuals if r["generation"] == gen]
        assert ranked == sorted(ranked)
    assert "error" not in result.best and result.best["genes"]["body.height"] <= 120
    assert "it could not be evaluated" in result.best_toml.read_text()


def test_when_no_candidate_can_be_evaluated_the_best_design_overrides_nothing(small, here):
    short = dataclasses.replace(small, genes=[small.genes[0], Gene("body.height", 130, 200, 1, unit="mm"), small.genes[2]])
    result = trials.run(short, can_that_cannot_be_tall, workers=1, out=here / "out")
    assert "error" in result.best
    text = result.best_toml.read_text()
    assert "No candidate could be evaluated" in text and "overrides" not in tomllib.loads(text)
    assert design.load(result.best_toml) == design.load(short.design)


def test_an_error_of_several_lines_still_leaves_a_best_design_that_loads(small, here):
    result = trials.run(small, solver_that_fails, workers=1, out=here / "out")
    text = result.best_toml.read_text()
    assert tomllib.loads(text) == {"include": "../trials/can.toml"}  # valid TOML, with nothing overridden
    assert design.load(result.best_toml) == design.load(small.design)
    assert "# ValueError: solver failed\n# residual did not converge \\x1b[31mat step 3\\x1b[0m\n#\n# see the log\n" in text
    assert "\x1b" not in text and "\r" not in text
    assert lines(result.history)[1]["error"].startswith("ValueError: solver failed\nresidual")  # kept whole there


def test_free_text_in_the_best_design_stays_inside_its_comments(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "d.toml").write_text('finish = "plain"\n')
    finishes = Gene("finish", choices=["plain", "two\nlines # and a hash"])
    t = Trial("a name\nover two lines", tmp_path / "d.toml", [finishes], [Objective("off", "min")], population=4,
              generations=2)  # fmt: skip
    result = trials.run(t, lambda d: Evaluation({"off": float(d.finish == "plain")}), workers=1, out=tmp_path / "out")
    written = tomllib.loads(result.best_toml.read_text())
    assert written == {"include": "../d.toml", "overrides": {"finish": "two\nlines # and a hash"}}
    assert "# Trial a name over two lines:" in result.best_toml.read_text()
    assert '# the design has "plain"' in result.best_toml.read_text()


def test_a_metric_that_is_not_finite_makes_the_candidate_an_error(small, here):
    result = trials.run(small, can_with_nan, workers=1, out=here / "out")
    assert all(r["error"] == "metric 'mass_g' is nan" for r in result.individuals)


@pytest.mark.parametrize(
    "evaluate, error, complaint",
    [
        (can_without_mass, TrialError, "evaluate returned no metric mass_g .it returned: volume_cm3."),
        (can_as_a_dict, TypeError, "must return a similitude.trials.Evaluation, not dict"),
        (can_with_a_quantity, TypeError, "metric 'mass_g' must be a plain number"),
    ],
)
@pytest.mark.parametrize("workers", [1, 2])
def test_an_evaluate_that_breaks_its_contract_stops_the_run(small, here, evaluate, error, complaint, workers):
    with pytest.raises(error, match=complaint):
        trials.run(small, evaluate, workers=workers, out=here / "out")
