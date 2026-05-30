"""Unified experiment logging to console, buffered JSONL, and CSV."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


class BufferedExperimentLogger:
    """Logger with buffered CSV/JSONL writes and configurable dump cycle.

    Metrics are accumulated in memory and flushed to disk in batches
    to avoid spamming I/O on every step.  Events (info) are written
    immediately since they are infrequent.
    """

    def __init__(
        self,
        log_dir: Path,
        console: Console | None = None,
        buffer_size: int = 100,
    ) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.console = console or Console()
        self.buffer_size = buffer_size

        self.events_path = self.log_dir / "events.jsonl"
        self.metrics_path = self.log_dir / "metrics.csv"
        self._csv_fields: list[str] | None = None
        self._metric_buffer: list[dict[str, Any]] = []

    def info(self, message: str, **fields: object) -> None:
        self.console.log(message)
        payload: dict[str, object] = {"event": message}
        payload.update(**fields)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def metric(self, row: dict[str, Any]) -> None:
        flat = _flatten(row)
        self._metric_buffer.append(flat)
        if len(self._metric_buffer) >= self.buffer_size:
            self._flush_metrics()

    def flush(self) -> None:
        if self._metric_buffer:
            self._flush_metrics()

    def _flush_metrics(self) -> None:
        if not self._metric_buffer:
            return
        rows = self._metric_buffer
        self._metric_buffer = []

        all_keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)

        if self._csv_fields is None:
            self._csv_fields = all_keys
            with self.metrics_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fields)
                writer.writeheader()
        elif self._csv_fields is not None:
            missing = [k for k in all_keys if k not in self._csv_fields]
            if missing:
                self._csv_fields.extend(missing)

        with self.metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields, extrasaction="ignore")
            writer.writerows(rows)

        with self.events_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"metric": row}, sort_keys=True, default=str) + "\n")

    def table(self, title: str, rows: list[dict[str, Any]]) -> None:
        table = Table(title=title)
        columns = list(rows[0].keys()) if rows else []
        for column in columns:
            table.add_column(column)
        for row in rows:
            table.add_row(*(str(row.get(column, "")) for column in columns))
        self.console.print(table)


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if is_dataclass(value) and not isinstance(value, type):
            nested = asdict(value)
            for nested_key, nested_value in nested.items():
                out[f"{key}.{nested_key}"] = nested_value
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                out[f"{key}.{nested_key}"] = nested_value
        else:
            out[key] = value
    return out
