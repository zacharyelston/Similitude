"""Load a design: TOML in, a nested attribute-access object out.

    from similitude import design

    d = design.load("designs/skiff.toml", overrides={"hull.length": "5.2 m"})
    d.hull.length            # a pint quantity, 5.2 m
    d.hull.speed.ref         # the reference value of an uncertain one
    d.layout.operator_cg[0]  # lists stay lists

What a design file may hold, beyond plain TOML:

- **A unit.** `length = "4.57 m"` or `length = {value = 4.57, unit = "m"}` loads as a pint quantity.
  - The string form is a number, a space, and something pint reads as a unit. Any other string stays a string,
    so `"5 mm plywood"` and `"48V"` are text. The catch is text that happens to read as a unit: `"3 pm"` is
    three picometres.
  - The table form is strict: an unknown unit is an error.
- **A range.** `speed = {ref = 9.0, range = [7.0, 12.0]}` loads as an `Uncertain` with `.ref`, `.lo` and `.hi`.
  - Add `unit = "kn"`, or write the three numbers as `"9 kn"` strings, to give it a unit.
  - Code that wants one number reads `.ref`, calls `float()`, or takes `nominal(design)`, which swaps every
    range for its reference value. Each of these is a choice to evaluate the nominal design, and a result
    made that way should say so.
  - A range is how far an uncertain input may lie from `ref`, and nothing more. It is not the bounds a search
    may move a design variable between. It names no probability distribution, and says nothing about how
    inputs vary together, so nothing here treats it as one.
- **`include`.** `include = "base.toml"`, or a list of files, merges other design files in first. Paths are
  relative to the file that names them. Tables merge key by key; later includes win over earlier ones, and the
  including file wins over all of them.
- **`[overrides]`.** A table of dotted keys, applied after the file's own values and its includes:
  `"layout.operator_cg.0" = 2.1`.

Overrides, from the file's `[overrides]` or from `load(..., overrides=...)`, follow the paths you would read:

- `hull.length` is a key in a table, `layout.operator_cg.0` a list item, and `hull.speed.ref` (or `.lo`, `.hi`)
  one end of a range.
- The key must already exist. An unknown key is an error, never a silent addition.
- A value with a unit can only be overridden by a value with a compatible unit, which is converted to the
  design's unit. A bare number is a mismatch, and so is a unit on a value that has none.
- A range stays a range. Override its `.ref`, which must stay inside the bounds, or give a whole new
  `{ref, range}`. A single number in its place is an error, because it would drop the bounds. To pin a value on
  purpose, give a range of zero width: `{ref = 10.0, range = [10.0, 10.0]}`.
- A list is overridden by a list of the same length, and each item must fit the item it replaces by these same
  rules. An override cannot add or drop items, or turn a single value into a list.
- A table is never replaced whole: a nested table of overrides is read as more dotted keys.
- The `overrides` argument is applied last, so it wins over every file.

Only overrides are checked this way. A value merged in by `include` replaces the included one as written, so an
including file can change a value's unit, or turn a range into a number. That is a known limit until designs
have a schema to check against.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pint

__all__ = [
    "Design",
    "DesignError",
    "Uncertain",
    "UnitMismatchError",
    "UnknownKeyError",
    "load",
    "nominal",
    "ureg",
]

# One registry for the whole program: quantities from different registries cannot be mixed.
ureg = pint.get_application_registry()


class DesignError(ValueError):
    """A design file, or an override of it, is wrong."""


class UnknownKeyError(DesignError):
    """An override names a value the design does not have."""


class UnitMismatchError(DesignError):
    """An override's unit does not fit the design value it replaces."""


class Design(SimpleNamespace):
    """A table of a design: `d.hull.length`, or `d["hull"]["length"]` for keys that are not identifiers."""

    def __getattr__(self, name: str) -> Any:
        # Only reached when the value is missing.
        have = ", ".join(vars(self)) or "nothing"
        raise AttributeError(f"this design table has no value {name!r} (it has: {have})")

    def __getitem__(self, key: str) -> Any:
        return vars(self)[key]

    def __contains__(self, key: object) -> bool:
        return key in vars(self)

    def __iter__(self) -> Iterator[str]:
        return iter(vars(self))

    def __len__(self) -> int:
        return len(vars(self))


@dataclass(frozen=True)
class Uncertain:
    """A value known only as a range: `ref` is the working value, and `lo <= ref <= hi`.

    The three are plain numbers, or pint quantities in one unit.

    The range is the input's uncertainty. It is not a search bound for an optimizer, and it carries no
    probability distribution and no dependence on other inputs. Reading `ref` evaluates the nominal case."""

    ref: Any
    lo: Any
    hi: Any

    def __post_init__(self) -> None:
        kinds = {isinstance(v, pint.Quantity) for v in (self.ref, self.lo, self.hi)}
        if len(kinds) > 1:
            raise DesignError(f"ref and range must all have a unit, or none: {self.ref}, [{self.lo}, {self.hi}]")
        try:
            inside = self.lo <= self.ref <= self.hi
        except (TypeError, ValueError):  # pint raises these for quantities that do not compare
            raise DesignError(f"ref and range must share a unit: {self.ref}, [{self.lo}, {self.hi}]") from None
        if not inside:
            raise DesignError(f"the range [{self.lo}, {self.hi}] does not contain ref {self.ref}")

    def __float__(self) -> float:
        return float(self.ref)


def load(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Design:
    """Read a design file, with its includes and `[overrides]`, then apply `overrides` on top."""
    design = _load_file(Path(path), ())
    for dotted, value in _flatten(overrides or {}):
        _apply(design, dotted, value)
    return design


def nominal(value: Any) -> Any:
    """A copy with every `Uncertain` replaced by its `ref`, for code that wants plain numbers."""
    if isinstance(value, Uncertain):
        return value.ref
    if isinstance(value, Design):
        return Design(**{k: nominal(v) for k, v in vars(value).items()})
    if isinstance(value, list):
        return [nominal(v) for v in value]
    return value


# --- files ------------------------------------------------------------------------------------------------------


def _load_file(path: Path, including: tuple[Path, ...]) -> Design:
    here = path.resolve()
    if here in including:
        chain = " -> ".join(p.name for p in (*including, here))
        raise DesignError(f"{path}: include cycle: {chain}")
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise DesignError(f"{path}: {e}") from None

    includes = raw.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list) or not all(isinstance(name, str) for name in includes):
        raise DesignError(f"{path}: include must be a file name or a list of them")
    overrides = raw.pop("overrides", {})
    if not isinstance(overrides, dict):
        raise DesignError(f"{path}: overrides must be a table of dotted keys")

    design = Design()
    for name in includes:
        included = path.parent / name
        if not included.is_file():
            raise FileNotFoundError(f"{path}: include {name!r}: no such file")
        design = _merge(design, _load_file(included, (*including, here)))
    try:
        design = _merge(design, _convert(raw, ""))
        for dotted, value in _flatten(overrides):
            _apply(design, dotted, value)
    except DesignError as e:
        raise type(e)(f"{path}: {e}") from None
    return design


def _merge(base: Design, top: Design) -> Design:
    """`top` over `base`. Tables merge key by key; anything else, lists included, is replaced whole."""
    out = Design(**vars(base))
    for key, value in vars(top).items():
        old = vars(out).get(key)
        both_tables = isinstance(old, Design) and isinstance(value, Design)
        vars(out)[key] = _merge(old, value) if both_tables else value
    return out


# --- values -----------------------------------------------------------------------------------------------------

# A number, a space, then the unit: "4.57 m", "-3e2 kg/m^3".
_NUMBER_THEN_UNIT = re.compile(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s+(\S.*?)\s*")

_RANGE_KEYS = ({"ref", "range"}, {"ref", "range", "unit"})


def _is_value_table(raw: Mapping) -> bool:
    """A table that spells one value: `{value, unit}` or `{ref, range[, unit]}`."""
    keys = set(raw)
    return keys == {"value", "unit"} or keys in _RANGE_KEYS


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _convert(raw: Any, at: str) -> Any:
    """Parsed TOML to design values. `at` is the dotted key, for error messages."""
    if isinstance(raw, Mapping):
        if set(raw) == {"value", "unit"}:
            return _quantity(raw["value"], raw["unit"], at)
        if set(raw) in _RANGE_KEYS:
            return _uncertain(raw, at)
        return Design(**{key: _convert(value, f"{at}.{key}" if at else key) for key, value in raw.items()})
    if isinstance(raw, list):
        return [_convert(value, f"{at}.{i}") for i, value in enumerate(raw)]
    if isinstance(raw, str):
        quantity = _quantity_from_text(raw)
        return raw if quantity is None else quantity
    return raw


def _quantity_from_text(text: str) -> pint.Quantity | None:
    """`"4.57 m"` as a quantity, or None for a string that is just text."""
    match = _NUMBER_THEN_UNIT.fullmatch(text)
    if not match or not any(c.isalpha() or c in "%°" for c in match[2]):
        return None
    try:
        return ureg.Quantity(float(match[1]), match[2])
    except Exception:  # whatever pint makes of prose such as "5 mm plywood", it is not a unit
        return None


def _quantity(value: Any, unit: Any, at: str) -> pint.Quantity:
    if not _is_number(value):
        raise DesignError(f"{at}: value must be a number, not {value!r}")
    if not isinstance(unit, str):
        raise DesignError(f"{at}: unit must be a string, not {unit!r}")
    try:
        return ureg.Quantity(float(value), unit)
    except Exception:
        raise DesignError(f"{at}: unknown unit {unit!r}") from None


def _uncertain(raw: Mapping, at: str) -> Uncertain:
    span = raw["range"]
    if not isinstance(span, list) or len(span) != 2:
        raise DesignError(f"{at}: range must be [low, high], not {span!r}")
    ref, lo, hi = (_convert(v, at) for v in (raw["ref"], *span))
    if "unit" in raw:
        ref, lo, hi = (_quantity(v, raw["unit"], at) for v in (ref, lo, hi))
    elif all(isinstance(v, pint.Quantity) for v in (ref, lo, hi)):
        try:
            lo, hi = lo.to(ref.units), hi.to(ref.units)
        except pint.DimensionalityError:
            raise DesignError(f"{at}: ref and range must share a unit: {ref}, [{lo}, {hi}]") from None
    elif all(_is_number(v) for v in (ref, lo, hi)):
        ref, lo, hi = float(ref), float(lo), float(hi)
    elif not all(_is_number(v) or isinstance(v, pint.Quantity) for v in (ref, lo, hi)):
        raise DesignError(f"{at}: ref and range must be numbers, not {raw['ref']!r} and {span!r}")
    try:
        return Uncertain(ref, lo, hi)
    except DesignError as e:
        raise DesignError(f"{at}: {e}") from None


# --- overrides --------------------------------------------------------------------------------------------------


def _flatten(overrides: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Dotted keys and their values. A nested table is more dotted keys, not a table to swap in whole:
    `{"hull": {"length": "5 m"}}` is `{"hull.length": "5 m"}`, which is also what unquoted dotted keys in a
    TOML `[overrides]` table parse to."""
    for key, value in overrides.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping) and value and not _is_value_table(value):
            yield from _flatten(value, f"{dotted}.")
        else:
            yield dotted, value


def _apply(design: Design, dotted: str, value: Any) -> None:
    """Set one existing value of `design`, in place."""
    *path, key = dotted.split(".")
    holder, held_as, node = None, "", design  # `node` sits in `holder` under `held_as`
    try:
        for part in path:
            holder, held_as, node = node, part, _get(node, part)
        old = _get(node, key)
    except LookupError:
        raise UnknownKeyError(f"override {dotted}: no such design value") from None

    try:
        new = _fit(old, _convert(value, dotted))
        if isinstance(node, Uncertain):
            if not (_is_number(new) or isinstance(new, pint.Quantity)):
                raise DesignError(f"one end of a range must be a number, not {value!r}")
            _set(holder, held_as, replace(node, **{key: float(new) if _is_number(new) else new}))
        else:
            _set(node, key, new)
    except DesignError as e:
        raise type(e)(f"override {dotted}: {e}") from None


def _get(node: Any, part: str) -> Any:
    """One step down a dotted path. Raises LookupError if the step does not exist."""
    if isinstance(node, Design):
        return vars(node)[part]
    if isinstance(node, list):
        if not (part.isascii() and part.isdigit()):
            raise KeyError(part)
        return node[int(part)]
    if isinstance(node, Uncertain) and part in ("ref", "lo", "hi"):
        return getattr(node, part)
    raise KeyError(part)


def _set(node: Any, part: str, value: Any) -> None:
    if isinstance(node, list):
        node[int(part)] = value
    else:
        vars(node)[part] = value


def _fit(old: Any, new: Any) -> Any:
    """`new`, checked against the value it replaces and converted to that value's unit."""
    if isinstance(old, Design):
        raise DesignError("this is a table; override its values one by one")
    if isinstance(new, Design):
        raise DesignError("an override sets a value, not a table")
    if isinstance(old, list) or isinstance(new, list):
        return _fit_list(old, new)
    if isinstance(old, Uncertain) and not isinstance(new, Uncertain):
        # A single number here would drop the bounds, and the fact that the value is uncertain at all.
        raise DesignError("this is a range; override its .ref, or give a whole new {ref, range}")
    old_unit, new_unit = _unit(old), _unit(new)
    if old_unit is None and new_unit is None:
        return new
    if old_unit is None:
        raise UnitMismatchError(f"the design value {old!r} has no unit, but the override is in {new_unit}")
    if new_unit is None:
        example = f'"1.5 {old_unit:~}"'
        raise UnitMismatchError(f"the design value is in {old_unit}, so the override needs a unit, as in {example}")
    if not new_unit.is_compatible_with(old_unit):
        raise UnitMismatchError(f"the design value is in {old_unit}, and {new_unit} does not convert to it")
    if isinstance(new, Uncertain):
        return Uncertain(new.ref.to(old_unit), new.lo.to(old_unit), new.hi.to(old_unit))
    return new.to(old_unit)


def _fit_list(old: Any, new: Any) -> list:
    """A list for a list of the same length, each item fitted to the item it replaces."""
    if not isinstance(old, list):
        raise DesignError("this is a single value; an override cannot make it a list")
    if not isinstance(new, list):
        raise DesignError("this is a list; override it with a list, or one item by its index")
    if len(new) != len(old):
        raise DesignError(f"this list has {len(old)} items and the override has {len(new)}; it cannot add or drop items")
    fitted = []
    for i, (old_item, new_item) in enumerate(zip(old, new)):
        try:
            fitted.append(_fit(old_item, new_item))
        except DesignError as e:
            raise type(e)(f"item {i}: {e}") from None
    return fitted


def _unit(value: Any) -> pint.Unit | None:
    if isinstance(value, Uncertain):
        value = value.ref
    return value.units if isinstance(value, pint.Quantity) else None
