"""Run manifest and append/resume contract."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .jsonl import append_record, load_records


def stable_run_id(*parts: object) -> str:
    payload = json.dumps([str(part) for part in parts], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class JsonlRunStore:
    """Thread-safe append-only result store with explicit resume semantics."""

    def __init__(self, root: Path, *, run_id: str, manifest: dict[str, Any] | None = None):
        self.root = root
        self.run_id = run_id
        self.results_path = root / "results.jsonl"
        self.errors_path = root / "errors.jsonl"
        self.manifest_path = root / "manifest.json"
        self._lock = threading.Lock()
        self._seen_ids = {
            str(record["record_id"])
            for record in load_records(self.results_path)
            if record.get("record_id")
        }
        if manifest is not None and not self.manifest_path.exists():
            root.mkdir(parents=True, exist_ok=True)
            payload = {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **manifest,
            }
            self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def completed_ids(self) -> set[str]:
        return {
            str(record["record_id"])
            for record in load_records(self.results_path)
            if record.get("record_id") and record.get("status", "completed") == "completed"
        }

    def append_result(self, result: dict[str, Any]) -> None:
        if not result.get("record_id"):
            raise ValueError("result requires record_id")
        with self._lock:
            record_id = str(result["record_id"])
            if record_id in self._seen_ids:
                raise ValueError(f"duplicate record_id: {record_id}")
            append_record(self.results_path, result)
            self._seen_ids.add(record_id)

    def append_error(self, error: dict[str, Any]) -> None:
        if not error.get("record_id"):
            raise ValueError("error requires record_id")
        with self._lock:
            append_record(self.errors_path, {"status": "error", **error})

    def finalize(self, summary: dict[str, Any]) -> None:
        path = self.root / "summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
