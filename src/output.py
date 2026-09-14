"""CSV and JSON output, appended after every report so a crash never loses paid
calls: `llm_extractions.csv` plus one `run_config_<run_id>.json` per run."""

from __future__ import annotations

import csv
import json

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# One column per `ExtractedRow` field, in declaration order.
EXTRACTION_COLUMNS = [
    "report_name", "sent_pages",
    "llm_year", "llm_scope", "llm_value", "llm_unit", "raw_json_from_vlm",
]

# One row per retrieval query, i.e. per year of the grid.
RETRIEVAL_COLUMNS = ["report_name", "year", "retrieved_pages", "scores"]


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path

    @property
    def extractions_csv(self) -> Path:
        return self.output_dir / "llm_extractions.csv"

    @property
    def dry_run_csv(self) -> Path:
        return self.output_dir / "dry_run.csv"

    def for_dry_run(self) -> "OutputPaths":
        """Its own folder, so a dry run never lands in the real extractions."""
        return OutputPaths(self.output_dir / "dry_run")


def append_rows(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> int:
    """Appends rows, writing the header when the file is new."""
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", restval="")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in columns})
    return len(rows)


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return value


def append_report(paths: OutputPaths, rows: Iterable[Any]) -> int:
    """Appends one report's cells to llm_extractions.csv. `rows` are `ExtractedRow`,
    which already carries `report_name`."""
    return append_rows(paths.extractions_csv, EXTRACTION_COLUMNS,
                       (asdict(row) for row in rows))


def append_retrievals(paths: OutputPaths, report_name: str, retrievals: Iterable[Any]) -> int:
    """Appends one report's queries to dry_run.csv. `retrievals` are `Retrieval`."""
    return append_rows(paths.dry_run_csv, RETRIEVAL_COLUMNS,
                       ({"report_name": report_name, "year": r.year, "retrieved_pages": r.pages,
                         "scores": [round(h.score, 4) for h in r.hits]}
                        for r in sorted(retrievals, key=lambda r: r.year)))


def write_run_config(paths: OutputPaths, run_id: str, config: dict[str, Any],
                     extra: dict[str, Any] | None = None) -> Path:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    path = paths.output_dir / f"run_config_{run_id}.json"
    path.write_text(json.dumps({"run_id": run_id, **(extra or {}), "config": config},
                               indent=2, default=str), encoding="utf-8")
    return path
