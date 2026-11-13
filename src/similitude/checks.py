"""PASS/WARN/FAIL bookkeeping shared by every calculation.

A calculation reports each result as a `Check`: a value held against a limit, with a one-line reason. A set of
checks renders as a Markdown table (`to_markdown`) and as JSON (`to_json`), and `exit_code` turns it into a
process exit code, so CI can gate on a FAIL.

A WARN doesn't fail a run, but it needs a disposition (free text saying why it's acceptable) before the run is
kept; `undisposed` lists the WARNs that still lack one.

`Check`, its status rule and `_fmt` are ported from eOx's `src/eox/checks.py` (MIT); see NOTICE.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum

_OPS = ("<=", ">=")
_NON_FINITE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}


class Status(StrEnum):
    """A check's outcome. Ordered by severity: INFO < PASS < WARN < FAIL."""

    INFO = "INFO"  # a value recorded with no limit
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

    @property
    def severity(self) -> int:
        return _SEVERITY[self]

    # StrEnum would otherwise order alphabetically.
    def __lt__(self, other):
        return self.severity < Status(other).severity

    def __le__(self, other):
        return self.severity <= Status(other).severity

    def __gt__(self, other):
        return self.severity > Status(other).severity

    def __ge__(self, other):
        return self.severity >= Status(other).severity


_SEVERITY = {Status.INFO: 0, Status.PASS: 1, Status.WARN: 2, Status.FAIL: 3}


@dataclass(frozen=True)
class Check:
    """One value held against a limit: `value <op> limit` must hold.

    The status follows from the numbers and is never set by hand:

    - no `limit`: INFO;
    - `value <op> limit` false (a NaN value included): FAIL;
    - past `warn_limit`, a second and stricter limit, but inside `limit`: WARN;
    - inside `limit` by less than the fraction `warn_within` of it: WARN;
    - otherwise PASS.
    """

    group: str
    name: str
    value: float | str  # text only for an INFO check
    unit: str
    limit: float | None = None
    op: str = "<="
    warn_within: float = 0.0
    reason: str = ""  # one line: why the limit is what it is, or what the value means
    warn_limit: float | None = None
    disposition: str = ""  # free text, for a WARN only: why it's acceptable

    def __post_init__(self):
        if self.op not in _OPS:
            raise ValueError(f"{self.name}: op must be one of {_OPS}, not {self.op!r}")
        if "\n" in self.reason or "\r" in self.reason:
            raise ValueError(f"{self.name}: the reason must be one line")
        if self.warn_within < 0:
            raise ValueError(f"{self.name}: warn_within is a fraction of the limit and can't be negative")
        if self.limit is None:
            if self.warn_limit is not None:
                raise ValueError(f"{self.name}: warn_limit needs a limit")
        else:
            if isinstance(self.value, str):
                raise ValueError(f"{self.name}: a text value can't be held against a limit")
            if self.warn_limit is not None and not _holds(self.warn_limit, self.op, self.limit):
                raise ValueError(f"{self.name}: warn_limit must be at least as strict as limit")
        if self.disposition.strip() and self.status is not Status.WARN:
            raise ValueError(f"{self.name}: only a WARN takes a disposition, and this check is {self.status}")

    @property
    def status(self) -> Status:
        if self.limit is None:
            return Status.INFO
        if not _holds(self.value, self.op, self.limit):
            return Status.FAIL
        if self.warn_limit is not None and not _holds(self.value, self.op, self.warn_limit):
            return Status.WARN
        if self.warn_within and self.limit:
            margin = (self.limit - self.value) if self.op == "<=" else (self.value - self.limit)
            if margin < self.warn_within * abs(self.limit):
                return Status.WARN
        return Status.PASS

    def disposed(self, disposition: str) -> Check:
        """This WARN with its disposition recorded."""
        if not disposition.strip():
            raise ValueError(f"{self.name}: a disposition can't be blank")
        return replace(self, disposition=disposition)

    def row(self) -> tuple[str, ...]:
        """Status, group, name, value, limit, reason and disposition, as text."""
        lim = "" if self.limit is None else f"{self.op} {_fmt(self.limit)}"
        if self.warn_limit is not None:
            fail_op = ">" if self.op == "<=" else "<"
            lim = f"{self.op} {_fmt(self.warn_limit)}, fail {fail_op} {_fmt(self.limit)}"
        value = f"{_fmt(self.value)} {self.unit}".strip()
        return (str(self.status), self.group, self.name, value, lim, self.reason, self.disposition)

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "name": self.name,
            "status": str(self.status),
            "value": _to_json_number(self.value),
            "unit": self.unit,
            "limit": _to_json_number(self.limit),
            "op": self.op,
            "warn_limit": _to_json_number(self.warn_limit),
            "warn_within": self.warn_within,
            "reason": self.reason,
            "disposition": self.disposition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Check:
        """The check `to_dict` wrote. Raises ValueError if the stored status isn't what the numbers give."""
        limit = _from_json_number(d.get("limit"))
        value = d["value"]
        if limit is not None or value in _NON_FINITE:  # otherwise an INFO check, whose value may be text
            value = _from_json_number(value)
        check = cls(
            group=d["group"],
            name=d["name"],
            value=value,
            unit=d.get("unit", ""),
            limit=limit,
            op=d.get("op", "<="),
            warn_within=d.get("warn_within", 0.0),
            reason=d.get("reason", ""),
            warn_limit=_from_json_number(d.get("warn_limit")),
            disposition=d.get("disposition", ""),
        )
        if "status" in d and d["status"] != check.status:
            raise ValueError(f"{check.name}: recorded as {d['status']} but its numbers give {check.status}")
        return check


def worst(checks: Iterable[Check]) -> Status:
    """The most severe status in the set; INFO for an empty one."""
    return max((c.status for c in checks), default=Status.INFO)


def counts(checks: Iterable[Check]) -> dict[Status, int]:
    """How many checks have each status, most severe first."""
    statuses = [c.status for c in checks]
    return {s: statuses.count(s) for s in sorted(Status, reverse=True)}


def exit_code(checks: Iterable[Check]) -> int:
    """1 if any check is FAIL, else 0. A WARN doesn't fail."""
    return 1 if any(c.status is Status.FAIL for c in checks) else 0


def undisposed(checks: Iterable[Check]) -> list[Check]:
    """The WARNs with no disposition recorded. A run can't be kept until this is empty."""
    return [c for c in checks if c.status is Status.WARN and not c.disposition.strip()]


def summary(checks: Iterable[Check]) -> str:
    """One line, such as `1 FAIL, 2 WARN, 14 PASS, 3 INFO`."""
    return ", ".join(f"{n} {s}" for s, n in counts(checks).items())


def to_markdown(checks: Iterable[Check]) -> str:
    """The set as a Markdown table, one row per check, in the order given."""
    lines = [
        "| Status | Group | Check | Value | Limit | Reason | Disposition |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in checks:
        status, *cells = c.row()
        if c.status is Status.FAIL:
            status = f"**{status}**"
        if c.status is Status.WARN and not c.disposition.strip():
            cells[-1] = "*none recorded*"
        lines.append("| " + " | ".join(_cell(x) for x in (status, *cells)) + " |")
    return "\n".join(lines) + "\n"


def to_json(checks: Iterable[Check]) -> str:
    """The set as JSON: the overall result, the exit code, the counts and every check.

    JSON has no NaN or infinity, so those are written as the strings "NaN", "Infinity" and "-Infinity".
    """
    checks = list(checks)
    doc = {
        "result": str(worst(checks)),
        "exit_code": exit_code(checks),
        "counts": {str(s): n for s, n in counts(checks).items()},
        "undisposed_warnings": len(undisposed(checks)),
        "checks": [c.to_dict() for c in checks],
    }
    return json.dumps(doc, indent=2, allow_nan=False) + "\n"


def from_json(text: str) -> list[Check]:
    """The checks `to_json` wrote."""
    return [Check.from_dict(d) for d in json.loads(text)["checks"]]


def _holds(value: float, op: str, limit: float) -> bool:
    return value <= limit if op == "<=" else value >= limit


def _fmt(x: float | str) -> str:
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 100:
        return f"{x:.0f}"
    if a >= 10:
        return f"{x:.1f}"
    if a >= 1:
        return f"{x:.2f}"
    return f"{x:.3g}"


def _cell(text: str) -> str:
    """Text made safe for one Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def _to_json_number(x: float | str | None) -> float | str | None:
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else "Infinity" if x > 0 else "-Infinity"
    return x


def _from_json_number(x: float | str | None) -> float | None:
    if isinstance(x, str):
        if x not in _NON_FINITE:
            raise ValueError(f"expected a number, got {x!r}")
        return _NON_FINITE[x]
    return x
