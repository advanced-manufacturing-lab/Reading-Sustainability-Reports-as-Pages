"""Typed answer -> records. The shape is already guaranteed by the schema the SDK
sends (`questions.YearAnswer`), so nothing here re-checks JSON, scope labels or
duplicates. What is left is the projection of one answer onto the four cells of
its year: a cell the model did not find keeps `value=None` and is a negative."""

from __future__ import annotations

from dataclasses import dataclass

from .questions import SCOPES, Scope, YearAnswer

MAX_UNIT = 120


@dataclass(frozen=True)
class Record:
    scope: Scope
    year: int
    value: float | None
    unit: str | None


def records_from(answer: YearAnswer, year: int) -> list[Record]:
    """The four cells of `year`, negatives included."""
    out = []
    for scope in SCOPES:
        cell = getattr(answer, f"scope_{scope}")
        out.append(Record(
            scope=scope, year=year, value=cell.value,
            unit=(cell.unit or "").strip()[:MAX_UNIT] or None,
        ))
    return out
