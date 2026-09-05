"""Append-only storage for v2 traces and their legacy materialized views."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from memeval.storage.jsonl import append_record, load_records
from memeval.trace import TraceEnvelope, materialize_legacy_trace


class TraceStore:
    def __init__(self, root: Path):
        self.root = root
        self.traces_path = root / "traces.jsonl"
        self.legacy_path = root / "legacy_traces.jsonl"
        self.errors_path = root / "errors.jsonl"
        self.legacy_export_path = root / "legacy_trace.json"
        self.manifest_path = root / "manifest.json"
        self.summary_path = root / "summary.json"

    def completed_ids(self) -> set[str]:
        return {str(item["record_id"]) for item in load_records(self.traces_path) if item.get("record_id")}

    def append(self, envelope: TraceEnvelope) -> None:
        data = envelope.to_dict()
        append_record(self.traces_path, {"record_id": envelope.trace_id, **data})
        for key, records in materialize_legacy_trace(envelope).items():
            append_record(self.legacy_path, {"record_id": envelope.trace_id, "conversation_id": key, "records": records})

    def append_error(self, record_id: str, error: Exception) -> None:
        append_record(self.errors_path, {"record_id": record_id, "status": "error", "error": str(error), "type": type(error).__name__})

    def export_legacy_json(self) -> Path:
        merged: dict[str, list[dict[str, Any]]] = {}
        # trace_id already encodes "sample_id:question_id", so it doubles as
        # the per-question merge key: multiple subject envelopes for the same
        # question combine into one record's subjects list instead of
        # appearing as separate list entries.
        positions: dict[str, dict[str, int]] = {}
        for item in load_records(self.legacy_path):
            conversation_id = str(item["conversation_id"])
            records = merged.setdefault(conversation_id, [])
            by_trace_id = positions.setdefault(conversation_id, {})
            for record in item.get("records", []):
                trace_id = record.get("trace_id")
                position = by_trace_id.get(trace_id) if trace_id is not None else None
                if position is not None:
                    records[position].setdefault("subjects", []).extend(record.get("subjects", []))
                else:
                    records.append(record)
                    if trace_id is not None:
                        by_trace_id[trace_id] = len(records) - 1
        self.root.mkdir(parents=True, exist_ok=True)
        self.legacy_export_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8",
        )
        return self.legacy_export_path

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8",
        )

    def write_summary(self, *, selected_samples: int, selected_questions: int, added: int, skipped: int) -> dict[str, Any]:
        traces = load_records(self.traces_path)
        errors = load_records(self.errors_path)
        summary = {
            "selected_samples": selected_samples,
            "selected_questions": selected_questions,
            "completed": len({item.get("record_id") for item in traces if item.get("record_id")}),
            "added": added,
            "failed": len(errors),
            "skipped": skipped,
            "event_count": sum(len(item.get("events", [])) for item in traces),
            "output": {
                "trace_jsonl": self.traces_path.name,
                "legacy_json": self.legacy_export_path.name,
                "errors_jsonl": self.errors_path.name,
            },
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8",
        )
        return summary
