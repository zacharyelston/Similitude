"""Designs load from TOML with dotted overrides, ranges, units and includes."""

import math
from pathlib import Path

import pint
import pytest

from similitude import design
from similitude.design import DesignError, Uncertain, UnitMismatchError, UnknownKeyError, ureg

DESIGNS = Path(__file__).parent / "data" / "designs"
SKIFF = DESIGNS / "skiff.toml"


def metres(quantity):
    return quantity.to("m").magnitude


def write(folder, name, text):
    path = folder / name
    path.write_text(text)
    return path


# --- the loaded object ------------------------------------------------------------------------------------------


def test_a_design_loads_as_a_nested_attribute_access_object():
    d = design.load(SKIFF)
    assert d.name == "skiff"
    assert d.hull.depth == 0.6
    assert d.layout.seats == 2
    assert d.layout.operator_cg == [1.9, 0.0, 0.45]
    assert d.tanks[1].name == "reserve"


def test_a_design_table_also_reads_by_key():
    d = design.load(SKIFF)
    assert d["hull"]["depth"] == d.hull.depth
    assert "hull" in d and "keel" not in d
    assert list(d) == ["name", "hull", "motor", "layout", "tanks"]
    assert len(d.layout) == 2


def test_a_missing_value_is_an_attribute_error_that_lists_what_is_there():
    d = design.load(SKIFF)
    with pytest.raises(AttributeError, match="no value 'keel'.*hull"):
        d.keel


def test_the_include_and_overrides_keys_are_not_part_of_the_design():
    d = design.load(DESIGNS / "skiff-long.toml")
    assert "include" not in d and "overrides" not in d


def test_a_file_that_is_not_toml_is_a_design_error_naming_the_file(tmp_path):
    bad = write(tmp_path, "bad.toml", "hull = = 3\n")
    with pytest.raises(DesignError, match="bad.toml"):
        design.load(bad)


# --- overrides --------------------------------------------------------------------------------------------------


def test_a_dotted_override_sets_a_nested_key():
    d = design.load(SKIFF, overrides={"hull.depth": 0.75, "name": "deep skiff"})
    assert d.hull.depth == 0.75
    assert d.name == "deep skiff"
    assert d.layout.seats == 2  # the rest is untouched


def test_load_takes_no_overrides_by_default_and_one_load_does_not_leak_into_the_next():
    design.load(SKIFF, overrides={"hull.depth": 0.75})
    assert design.load(SKIFF).hull.depth == 0.6
    assert design.load(SKIFF, overrides={}) == design.load(SKIFF)


def test_a_nested_table_of_overrides_means_the_same_dotted_keys():
    nested = design.load(SKIFF, overrides={"hull": {"depth": 0.75}, "layout": {"seats": 4}})
    dotted = design.load(SKIFF, overrides={"hull.depth": 0.75, "layout.seats": 4})
    assert nested == dotted
    assert metres(nested.hull.length) == pytest.approx(4.57)  # the rest of [hull] survives


def test_the_overrides_table_in_a_file_is_applied_and_the_argument_wins_over_it():
    assert design.load(DESIGNS / "skiff-long.toml").layout.seats == 3
    assert design.load(DESIGNS / "skiff-long.toml", overrides={"layout.seats": 5}).layout.seats == 5


def test_a_table_cannot_be_overridden_whole():
    with pytest.raises(DesignError, match="override hull: this is a table"):
        design.load(SKIFF, overrides={"hull": 3})


# --- list items -------------------------------------------------------------------------------------------------


def test_an_override_sets_a_list_item_by_index():
    d = design.load(SKIFF, overrides={"layout.operator_cg.0": 2.1})
    assert d.layout.operator_cg == [2.1, 0.0, 0.45]


def test_an_override_reaches_through_a_list_of_tables():
    d = design.load(SKIFF, overrides={"tanks.1.name": "day tank"})
    assert [t.name for t in d.tanks] == ["main", "day tank"]


def test_a_list_can_be_overridden_whole_by_a_list_only():
    d = design.load(SKIFF, overrides={"layout.operator_cg": [2.0, 0.1, 0.5]})
    assert d.layout.operator_cg == [2.0, 0.1, 0.5]
    with pytest.raises(DesignError, match="this is a list"):
        design.load(SKIFF, overrides={"layout.operator_cg": 2.0})


@pytest.mark.parametrize("key", ["layout.operator_cg.3", "layout.operator_cg.-1", "layout.operator_cg.x", "tanks.2.name"])
def test_a_list_index_that_is_not_there_is_an_error(key):
    with pytest.raises(UnknownKeyError, match="no such design value"):
        design.load(SKIFF, overrides={key: 1.0})


# --- unknown keys -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "hull.lenght",  # a typo
        "keel.depth",  # no such table
        "colour",  # no such top-level value
        "hull.depth.max",  # through a number
        "hull.length.value",  # through a quantity
        "hull.speed.mid",  # a range has only ref, lo and hi
        "hull..depth",
        "",
    ],
)
def test_an_unknown_key_is_an_error(key):
    with pytest.raises(UnknownKeyError, match="no such design value"):
        design.load(SKIFF, overrides={key: 1.0})


def test_an_unknown_key_in_a_nested_table_of_overrides_is_an_error_too():
    with pytest.raises(UnknownKeyError, match="override hull.lenght"):
        design.load(SKIFF, overrides={"hull": {"lenght": "5 m"}})


def test_an_unknown_key_in_a_files_overrides_table_is_an_error_naming_the_file(tmp_path):
    bad = write(tmp_path, "bad.toml", '[hull]\ndepth = 0.6\n\n[overrides]\n"hull.deep" = 0.7\n')
    with pytest.raises(UnknownKeyError, match="bad.toml.*override hull.deep"):
        design.load(bad)


def test_unquoted_dotted_keys_in_a_files_overrides_table_work_like_quoted_ones(tmp_path):
    path = write(tmp_path, "d.toml", '[hull]\ndepth = 0.6\nlength = "4 m"\n\n[overrides]\nhull.depth = 0.7\n')
    d = design.load(path)
    assert d.hull.depth == 0.7
    assert metres(d.hull.length) == 4.0


# --- ranges -----------------------------------------------------------------------------------------------------


def test_a_ref_and_range_load_as_an_uncertain():
    speed = design.load(SKIFF).hull.speed
    assert isinstance(speed, Uncertain)
    assert (speed.ref, speed.lo, speed.hi) == (9.0, 7.0, 12.0)


def test_code_that_wants_a_number_gets_ref():
    d = design.load(SKIFF)
    assert float(d.hull.speed) == 9.0
    assert math.sqrt(d.hull.speed) == 3.0
    plain = design.nominal(d)
    assert plain.hull.speed == 9.0
    assert plain.motor.power == ureg.Quantity(15.0, "kW")
    assert plain.hull.depth == d.hull.depth
    assert isinstance(d.hull.speed, Uncertain)  # nominal() copies


def test_a_range_can_carry_a_unit_either_way():
    motor = design.load(SKIFF).motor
    assert motor.power == Uncertain(ureg.Quantity(15.0, "kW"), ureg.Quantity(10.0, "kW"), ureg.Quantity(20.0, "kW"))
    # written as strings, in mixed units: the ends are converted to ref's unit
    assert motor.shaft.ref == ureg.Quantity(508.0, "mm")
    assert str(motor.shaft.hi.units) == "millimeter"
    assert motor.shaft.hi.magnitude == pytest.approx(635.0)


@pytest.mark.parametrize(
    "value, complaint",
    [
        ("{ ref = 9.0, range = [10.0, 12.0] }", "does not contain ref"),
        ("{ ref = 9.0, range = [12.0, 7.0] }", "does not contain ref"),
        ("{ ref = 9.0, range = [7.0] }", "range must be"),
        ("{ ref = 9.0, range = 7.0 }", "range must be"),
        ('{ ref = "fast", range = [7.0, 12.0] }', "must be numbers"),
        ('{ ref = 9.0, range = ["7 kn", "12 kn"] }', "all have a unit, or none"),
        ('{ ref = "9 kn", range = ["7 m", "12 m"] }', "share a unit"),
        ('{ ref = 9.0, range = [7.0, 12.0], unit = "knotz" }', "unknown unit"),
    ],
)
def test_a_malformed_range_is_an_error_naming_the_value(tmp_path, value, complaint):
    bad = write(tmp_path, "bad.toml", f"[hull]\nspeed = {value}\n")
    with pytest.raises(DesignError, match=f"bad.toml: hull.speed: .*{complaint}"):
        design.load(bad)


def test_an_override_moves_one_end_of_a_range_and_keeps_the_rest():
    d = design.load(SKIFF, overrides={"hull.speed.ref": 10, "hull.speed.hi": 13.5})
    assert d.hull.speed == Uncertain(10.0, 7.0, 13.5)


def test_an_override_cannot_move_ref_outside_its_range():
    with pytest.raises(DesignError, match="override hull.speed.ref: .*does not contain ref 20"):
        design.load(SKIFF, overrides={"hull.speed.ref": 20.0})


def test_an_override_can_replace_a_range_with_a_number_or_another_range():
    assert design.load(SKIFF, overrides={"hull.speed": 10.0}).hull.speed == 10.0
    d = design.load(SKIFF, overrides={"hull.speed": {"ref": 10.0, "range": [9.0, 11.0]}})
    assert d.hull.speed == Uncertain(10.0, 9.0, 11.0)


# --- units ------------------------------------------------------------------------------------------------------


def test_a_string_with_a_unit_and_a_value_unit_table_load_through_pint():
    hull = design.load(SKIFF).hull
    assert isinstance(hull.length, pint.Quantity) and isinstance(hull.beam, pint.Quantity)
    assert hull.length == ureg.Quantity(4.57, "m")
    assert hull.beam == ureg.Quantity(1.8, "m")
    assert hull.length.to("ft").magnitude == pytest.approx(4.57 / 0.3048)  # the foot is exactly 0.3048 m
    assert design.load(SKIFF).tanks[0].volume.to("m^3").magnitude == pytest.approx(0.040)


def test_quantities_from_a_design_work_with_the_shared_registry():
    hull = design.load(SKIFF).hull
    area = hull.length * hull.beam
    assert area.to("m^2").magnitude == pytest.approx(8.226)
    assert design.ureg is pint.get_application_registry()


@pytest.mark.parametrize("text", ["5 mm plywood", "48V", "4.57", "2 per side", "16x6.50-8", "m 4.57", ""])
def test_a_string_that_is_not_a_number_then_a_unit_stays_a_string(tmp_path, text):
    path = write(tmp_path, "d.toml", f'note = "{text}"\n')
    assert design.load(path).note == text


def test_an_unknown_unit_in_a_value_unit_table_is_an_error(tmp_path):
    bad = write(tmp_path, "bad.toml", '[hull]\nlength = { value = 4.57, unit = "metrs" }\n')
    with pytest.raises(DesignError, match="bad.toml: hull.length: unknown unit 'metrs'"):
        design.load(bad)


def test_an_override_in_a_compatible_unit_is_converted_to_the_designs_unit():
    d = design.load(SKIFF, overrides={"hull.length": "15 ft", "hull.beam": {"value": 200, "unit": "cm"}})
    assert str(d.hull.length.units) == "meter"
    assert d.hull.length.magnitude == pytest.approx(4.572)  # 15 × 0.3048
    assert d.hull.beam == ureg.Quantity(2.0, "m")


def test_an_override_can_be_a_pint_quantity():
    d = design.load(SKIFF, overrides={"hull.length": ureg.Quantity(5.0, "m")})
    assert d.hull.length == ureg.Quantity(5.0, "m")


@pytest.mark.parametrize(
    "key, value, complaint",
    [
        ("hull.length", "3 s", "second does not convert"),  # the wrong dimension
        ("hull.length", {"value": 3.0, "unit": "kg"}, "kilogram does not convert"),
        ("hull.length", 5.2, "needs a unit"),  # a bare number where the design has a unit
        ("hull.depth", "0.7 m", "has no unit"),  # a unit where the design has none
        ("hull.speed", "10 kn", "has no unit"),
        ("hull.speed.ref", "10 kn", "has no unit"),
        ("motor.power", "20 m", "meter does not convert"),
        ("motor.power", 20.0, "needs a unit"),
        ("motor.power.ref", 18.0, "needs a unit"),
        ("motor.power", {"ref": 18.0, "range": [12.0, 20.0]}, "needs a unit"),
        ("tanks.0.volume", "40 kg", "kilogram does not convert"),
    ],
)
def test_a_unit_mismatch_on_override_is_an_error(key, value, complaint):
    with pytest.raises(UnitMismatchError, match=f"override {key}: .*{complaint}"):
        design.load(SKIFF, overrides={key: value})


def test_an_override_of_a_range_with_a_unit_is_converted_too():
    d = design.load(SKIFF, overrides={"motor.power.ref": "18000 W", "motor.shaft": "20 inch"})
    assert d.motor.power == Uncertain(ureg.Quantity(18.0, "kW"), ureg.Quantity(10.0, "kW"), ureg.Quantity(20.0, "kW"))
    assert str(d.motor.shaft.units) == "millimeter"
    assert d.motor.shaft.magnitude == pytest.approx(508.0)  # the inch is exactly 25.4 mm


# --- include ----------------------------------------------------------------------------------------------------


def test_the_including_file_wins_over_the_file_it_includes():
    d = design.load(DESIGNS / "skiff-long.toml")
    assert d.name == "skiff-long"
    assert metres(d.hull.length) == pytest.approx(5.2)


def test_tables_merge_key_by_key_and_new_tables_are_added():
    d = design.load(DESIGNS / "skiff-long.toml")
    assert d.hull.beam == ureg.Quantity(1.8, "m")  # from skiff.toml, though skiff-long.toml has a [hull] too
    assert d.motor == design.load(SKIFF).motor
    assert metres(d.canopy.height) == pytest.approx(1.1)


def test_a_files_overrides_reach_into_what_it_includes():
    d = design.load(DESIGNS / "skiff-long.toml")
    assert d.layout.operator_cg == [2.2, 0.0, 0.45]


def test_later_includes_win_over_earlier_ones_and_the_including_file_over_both():
    d = design.load(DESIGNS / "skiff-fast.toml")
    assert d.motor.power.ref == ureg.Quantity(30.0, "kW")  # outboard-big.toml, listed after skiff.toml
    assert d.motor.shaft.ref == ureg.Quantity(508.0, "mm")  # skiff.toml: outboard-big.toml does not set it
    assert d.hull.speed == Uncertain(14.0, 11.0, 18.0)  # skiff-fast.toml itself
    assert d.name == "skiff-fast"


def test_the_including_file_wins_over_an_included_files_overrides(tmp_path):
    write(tmp_path, "base.toml", '[hull]\ndepth = 0.6\nbeam = "1.8 m"\n\n[overrides]\n"hull.depth" = 0.7\n"hull.beam" = "2 m"\n')
    top = write(tmp_path, "top.toml", 'include = "base.toml"\n\n[hull]\ndepth = 0.9\n')
    d = design.load(top)
    assert d.hull.depth == 0.9
    assert metres(d.hull.beam) == 2.0  # the included file's own override still holds where top.toml is silent


def test_includes_are_found_relative_to_the_file_that_names_them(tmp_path):
    (tmp_path / "parts").mkdir()
    write(tmp_path / "parts", "motor.toml", 'include = "shaft.toml"\n\n[motor]\npower = "15 kW"\n')
    write(tmp_path / "parts", "shaft.toml", '[motor]\nshaft = "508 mm"\n')
    top = write(tmp_path, "boat.toml", 'include = "parts/motor.toml"\nname = "boat"\n')
    d = design.load(top)
    assert d.motor.power == ureg.Quantity(15.0, "kW")
    assert d.motor.shaft == ureg.Quantity(508.0, "mm")


def test_an_include_cycle_is_an_error(tmp_path):
    a = write(tmp_path, "a.toml", 'include = "b.toml"\n')
    write(tmp_path, "b.toml", 'include = "a.toml"\n')
    with pytest.raises(DesignError, match="include cycle: a.toml -> b.toml -> a.toml"):
        design.load(a)


def test_a_missing_include_is_an_error_naming_both_files(tmp_path):
    top = write(tmp_path, "top.toml", 'include = "nowhere.toml"\n')
    with pytest.raises(FileNotFoundError, match="top.toml: include 'nowhere.toml'"):
        design.load(top)


def test_an_include_that_is_not_a_file_name_is_an_error(tmp_path):
    bad = write(tmp_path, "bad.toml", "include = 3\n")
    with pytest.raises(DesignError, match="include must be a file name"):
        design.load(bad)
