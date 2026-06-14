"""Export dual performance/certificate result tables grouped by experiment family."""

import csv
import json
from pathlib import Path
from typing import Any


def _flatten(prefix: str, data: dict[str, Any], out: dict[str, Any]) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            _flatten(f"{prefix}{key}.", value, out)
        else:
            out[f"{prefix}{key}"] = value

def collect_grouped_results(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Collect results grouped by experiment family based on manifest.json."""
    grouped_rows: dict[str, list[dict[str, Any]]] = {
        "main_queueing": [],
        "legacy": []
    }
    
    for manifest_path in root.rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
            
        family = manifest.get("experiment_family", "legacy")
        if family not in grouped_rows:
            grouped_rows[family] = []
            
        run_root = manifest_path.parent
        # Read all json results in this run (metrics or audits)
        for result_path in run_root.rglob("*.json"):
            if result_path.name in {"manifest.json", "events.jsonl", "events.json"}:
                continue
            if result_path.name not in {"results.json", "state_bank_audit.json"}:
                continue
                
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                continue
                
            entries = payload if isinstance(payload, list) else [payload]
            for entry in entries:
                row: dict[str, Any] = {"source": str(result_path)}
                assert isinstance(entry, dict)
                _flatten("", entry, row)
                row["experiment_family"] = family
                grouped_rows[family].append(row)
                
    return grouped_rows

def main() -> None:
    """Export merged CSV tables for each experiment family."""
    grouped_rows = collect_grouped_results(Path("outputs"))
    out_dir = Path("outputs") / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for family, rows in grouped_rows.items():
        if not rows:
            continue
        out_path = out_dir / f"{family}_results.csv"
        fieldnames = sorted({key for row in rows for key in row})
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Exported {len(rows)} rows to {out_path}")

if __name__ == "__main__":
    main()
