"""Export dual performance/certificate result tables."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResultColumns:
    performance: tuple[str, ...] = (
        "avg_queue_length",
        "avg_cost",
        "latency_proxy",
        "p95_backlog",
        "max_backlog",
        "drop_rate",
    )
    certificate: tuple[str, ...] = (
        "certificate_violation_rate",
        "min_certificate_slack",
        "avg_certificate_slack",
        "projection_activation_rate",
        "tail_fallback_activation_rate",
        "usage_activation_rate",
        "usage_mean_activation",
        "correction_magnitude",
        "instability_rate",
    )


def _flatten(prefix: str, data: dict[str, Any], out: dict[str, Any]) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            _flatten(f"{prefix}{key}.", value, out)
        else:
            out[f"{prefix}{key}"] = value


def collect_result_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in root.rglob("*.json"):
        if path.name not in {"results.json", "state_bank_audit.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            row: dict[str, Any] = {"source": str(path)}
            assert isinstance(entry, dict)
            _flatten("", entry, row)
            rows.append(row)
    return rows


def main() -> None:
    cols = ResultColumns()
    rows = collect_result_rows(Path("outputs"))
    if not rows:
        print("performance," + ",".join(cols.performance))
        print("certificate," + ",".join(cols.certificate))
        return
    out = Path("outputs") / "paper_tables" / "dual_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(str(out))


if __name__ == "__main__":
    main()
