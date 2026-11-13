"""Checks: status ordering, the status rule, rendering, exit codes and dispositions."""

import json
import math

import pytest

from similitude import checks
from similitude.checks import Check, Status


def a_pass():
    return Check("Drive", "Tractive effort", 1200.0, "N", 1000.0, ">=", reason="pulls the loaded cart up 15°")


def a_warn():
    return Check("Bus", "Cable temperature", 68.0, "°C", 90.0, warn_limit=60.0, reason="insulation rating")


def a_fail():
    return Check("Float", "Freeboard", 0.08, "m", 0.15, ">=", reason="deck stays dry in a 0.1 m chop")


def an_info():
    return Check("Mass", "All-up mass", 412.0, "kg")


# --- status ordering


def test_statuses_order_by_severity():
    assert Status.INFO < Status.PASS < Status.WARN < Status.FAIL
    assert sorted([Status.FAIL, Status.INFO, Status.WARN, Status.PASS]) == [
        Status.INFO,
        Status.PASS,
        Status.WARN,
        Status.FAIL,
    ]
    assert max(Status) is Status.FAIL
    assert Status.WARN >= Status.WARN and Status.WARN <= Status.WARN and Status.FAIL > Status.PASS


def test_a_status_compares_equal_to_its_name():
    assert Status.FAIL == "FAIL"
    assert str(Status.WARN) == "WARN"


def test_the_worst_status_of_a_set():
    assert checks.worst([a_pass(), an_info()]) is Status.PASS
    assert checks.worst([a_pass(), a_warn(), an_info()]) is Status.WARN
    assert checks.worst([a_warn(), a_fail(), a_pass()]) is Status.FAIL
    assert checks.worst([]) is Status.INFO


def test_counts_list_the_most_severe_first():
    c = checks.counts([a_pass(), a_pass(), a_warn(), an_info()])
    assert list(c.items()) == [(Status.FAIL, 0), (Status.WARN, 1), (Status.PASS, 2), (Status.INFO, 1)]
    assert checks.summary([a_pass(), a_pass(), a_warn(), an_info()]) == "0 FAIL, 1 WARN, 2 PASS, 1 INFO"


# --- the status rule, against answers worked by hand


@pytest.mark.parametrize(
    ("value", "limit", "op", "expected"),
    [
        (9.0, 10.0, "<=", Status.PASS),
        (10.0, 10.0, "<=", Status.PASS),  # the limit itself holds
        (10.1, 10.0, "<=", Status.FAIL),
        (11.0, 10.0, ">=", Status.PASS),
        (10.0, 10.0, ">=", Status.PASS),
        (9.9, 10.0, ">=", Status.FAIL),
        (math.nan, 10.0, "<=", Status.FAIL),  # a NaN never passes
        (math.nan, 10.0, ">=", Status.FAIL),
    ],
)
def test_a_value_is_held_against_its_limit(value, limit, op, expected):
    assert Check("g", "n", value, "", limit, op).status is expected


def test_no_limit_is_info():
    assert an_info().status is Status.INFO
    assert Check("Drive", "Gearing", "12:1", "").status is Status.INFO


@pytest.mark.parametrize(
    ("value", "expected"),
    [(60.0, Status.PASS), (60.1, Status.WARN), (90.0, Status.WARN), (90.1, Status.FAIL)],
)
def test_a_warn_limit_makes_a_band_below_an_upper_limit(value, expected):
    assert Check("g", "n", value, "", 90.0, "<=", warn_limit=60.0).status is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(2.0, Status.PASS), (1.9, Status.WARN), (1.5, Status.WARN), (1.4, Status.FAIL)],
)
def test_a_warn_limit_makes_a_band_above_a_lower_limit(value, expected):
    assert Check("g", "n", value, "", 1.5, ">=", warn_limit=2.0).status is expected


@pytest.mark.parametrize(
    ("value", "op", "expected"),
    [
        (89.0, "<=", Status.PASS),  # 11 % inside a limit of 100
        (91.0, "<=", Status.WARN),  # 9 % inside
        (101.0, "<=", Status.FAIL),
        (111.0, ">=", Status.PASS),
        (109.0, ">=", Status.WARN),
        (99.0, ">=", Status.FAIL),
    ],
)
def test_warn_within_warns_close_to_the_limit(value, op, expected):
    assert Check("g", "n", value, "", 100.0, op, warn_within=0.10).status is expected


def test_warn_within_is_a_fraction_of_the_limits_size():
    assert Check("g", "n", -95.0, "", -100.0, ">=", warn_within=0.10).status is Status.WARN
    assert Check("g", "n", -80.0, "", -100.0, ">=", warn_within=0.10).status is Status.PASS
    assert Check("g", "n", -0.001, "", 0.0, "<=", warn_within=0.10).status is Status.PASS  # no fraction of zero


@pytest.mark.parametrize(
    "kwargs",
    [
        {"op": "<"},
        {"reason": "two\nlines"},
        {"warn_within": -0.1},
        {"limit": None, "warn_limit": 5.0},
        {"warn_limit": 12.0},  # looser than the limit of 10: the band could never be reached
        {"value": "12:1"},  # text can't be held against a limit
    ],
)
def test_a_malformed_check_is_refused(kwargs):
    fields = {"group": "g", "name": "n", "value": 1.0, "unit": "", "limit": 10.0} | kwargs
    with pytest.raises(ValueError):
        Check(**fields)


# --- rendering


def test_values_are_formatted_to_a_readable_precision():
    shown = {0: "0", 12345.6: "12,346", 123.4: "123", 12.34: "12.3", 1.234: "1.23", 0.012345: "0.0123", -2.5: "-2.50"}
    for value, text in shown.items():
        assert Check("g", "n", value, "").row()[3] == text


def test_a_row_carries_the_limit_and_its_warn_band():
    assert a_pass().row() == (
        "PASS",
        "Drive",
        "Tractive effort",
        "1,200 N",
        ">= 1,000",
        "pulls the loaded cart up 15°",
        "",
    )
    assert a_warn().row()[4] == "<= 60.0, fail > 90.0"
    assert Check("g", "n", 1.9, "", 1.5, ">=", warn_limit=2.0).row()[4] == ">= 2.00, fail < 1.50"
    assert an_info().row()[3:5] == ("412 kg", "")


def test_a_set_renders_as_a_markdown_table():
    disposed = a_warn().disposed("Peak lasts under a minute; cable is in free air.")
    assert checks.to_markdown([a_pass(), disposed, a_fail(), an_info()]) == (
        "| Status | Group | Check | Value | Limit | Reason | Disposition |\n"
        "|---|---|---|---|---|---|---|\n"
        "| PASS | Drive | Tractive effort | 1,200 N | >= 1,000 | pulls the loaded cart up 15° |  |\n"
        "| WARN | Bus | Cable temperature | 68.0 °C | <= 60.0, fail > 90.0 | insulation rating "
        "| Peak lasts under a minute; cable is in free air. |\n"
        "| **FAIL** | Float | Freeboard | 0.08 m | >= 0.15 | deck stays dry in a 0.1 m chop |  |\n"
        "| INFO | Mass | All-up mass | 412 kg |  |  |  |\n"
    )


def test_the_table_flags_a_warn_with_no_disposition():
    row = checks.to_markdown([a_warn()]).splitlines()[2]
    assert row.endswith("| insulation rating | *none recorded* |")


def test_table_cells_stay_on_one_line_in_one_column():
    warn = Check("a|b", "n", 68.0, "", 90.0, warn_limit=60.0).disposed("fine:\nsee the note | below")
    row = checks.to_markdown([warn]).splitlines()[2]
    assert "a\\|b" in row
    assert row.endswith("| fine: see the note \\| below |")
    assert row.replace("\\|", "").count("|") == 8


def test_a_set_renders_as_json():
    doc = json.loads(checks.to_json([a_pass(), a_warn(), a_fail(), an_info()]))
    assert doc["result"] == "FAIL"
    assert doc["exit_code"] == 1
    assert doc["counts"] == {"FAIL": 1, "WARN": 1, "PASS": 1, "INFO": 1}
    assert doc["undisposed_warnings"] == 1
    assert doc["checks"][1] == {
        "group": "Bus",
        "name": "Cable temperature",
        "status": "WARN",
        "value": 68.0,
        "unit": "°C",
        "limit": 90.0,
        "op": "<=",
        "warn_limit": 60.0,
        "warn_within": 0.0,
        "reason": "insulation rating",
        "disposition": "",
    }


def test_json_reads_back_to_the_same_checks():
    original = [a_pass(), a_warn().disposed("accepted"), a_fail(), an_info(), Check("Drive", "Gearing", "12:1", "")]
    assert checks.from_json(checks.to_json(original)) == original


def test_json_is_strict_even_for_values_json_cant_hold():
    original = [
        Check("g", "nan", math.nan, "", 1.0),
        Check("g", "inf", -math.inf, ""),
        Check("g", "open", 1.0, "", math.inf),
    ]
    text = checks.to_json(original)
    json.loads(text, parse_constant=lambda name: pytest.fail(f"bare {name} in the JSON"))
    back = checks.from_json(text)
    assert math.isnan(back[0].value) and back[0].status is Status.FAIL
    assert back[1].value == -math.inf
    assert back[2].limit == math.inf and back[2].status is Status.PASS


def test_json_whose_status_disagrees_with_its_numbers_is_refused():
    doc = json.loads(checks.to_json([a_fail()]))
    doc["checks"][0]["status"] = "PASS"
    with pytest.raises(ValueError, match="recorded as PASS but its numbers give FAIL"):
        checks.from_json(json.dumps(doc))


# --- exit codes


def test_the_exit_code_is_1_only_on_a_fail():
    assert checks.exit_code([]) == 0
    assert checks.exit_code([a_pass(), an_info()]) == 0
    assert checks.exit_code([a_pass(), a_warn()]) == 0
    assert checks.exit_code([a_pass(), a_warn(), a_fail()]) == 1
    assert checks.exit_code(iter([a_fail()])) == 1


# --- dispositions


def test_a_warn_needs_a_disposition_but_doesnt_fail():
    run = [a_pass(), a_warn(), an_info()]
    assert checks.undisposed(run) == [a_warn()]
    assert checks.exit_code(run) == 0


def test_a_recorded_disposition_clears_the_warn_for_keeping():
    disposed = a_warn().disposed("Peak lasts under a minute; cable is in free air.")
    assert disposed.status is Status.WARN  # still a WARN: the disposition explains it, it doesn't hide it
    assert disposed.disposition == "Peak lasts under a minute; cable is in free air."
    assert checks.undisposed([a_pass(), disposed]) == []
    assert a_warn().disposition == ""  # the original is unchanged


def test_a_blank_disposition_doesnt_count():
    with pytest.raises(ValueError):
        a_warn().disposed("   ")
    blank = Check("Bus", "Cable temperature", 68.0, "°C", 90.0, warn_limit=60.0, disposition="  ")
    assert checks.undisposed([blank]) == [blank]


@pytest.mark.parametrize("check", [a_pass(), a_fail(), an_info()])
def test_only_a_warn_takes_a_disposition(check):
    with pytest.raises(ValueError, match="only a WARN takes a disposition"):
        check.disposed("waived")
