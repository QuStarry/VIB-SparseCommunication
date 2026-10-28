"""Append-only request evidence; pending or absent usage never means zero cost."""

import copy
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid


def fingerprint(value) -> str:
    """Hash canonical JSON without retaining prompt or credential content."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class LedgerError(RuntimeError):
    """Persistence failure: stop rather than silently retry a billable request."""


def summarize_requests(records) -> dict:
    known = [record["usage"] for record in records if record.get("usage_status") == "known"]
    unknown = sum(record.get("usage_status") != "known" for record in records)
    prompt = sum(usage["prompt_tokens"] for usage in known)
    completion = sum(usage["completion_tokens"] for usage in known)
    return {
        "total_calls": sum(record.get("response_id") is not None or
                           record.get("status") in ("completed", "invalid_response") for record in records),
        "total_attempts": len(records),
        "total_prompt_tokens": prompt,
        "total_completion_tokens": completion,
        "total_tokens": prompt + completion,
        "known_usage_requests": len(known),
        "unknown_usage_requests": unknown,
        "usage_complete": unknown == 0,
        "counter_scope": "known_usage_only",
    }


class RequestLedger:
    """Thread-safe per-client evidence with optional fsynced JSONL events.

    A durable file belongs to one active client. Reopening restores evidence,
    including unresolved pending requests, but never automatically resends them.
    No token-to-currency conversion is performed without a verified price model.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self._records = {}
        if self.path and self.path.exists():
            try:
                with self.path.open(encoding="utf-8") as stream:
                    for line in stream:
                        event = json.loads(line)
                        self._records[event["request_id"]] = event
            except (OSError, ValueError, KeyError) as exc:
                raise LedgerError("Cannot restore request ledger safely") from exc

    def _append(self, event):
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except (OSError, TypeError, ValueError) as exc:
            raise LedgerError("Cannot persist request evidence") from exc

    def start(self, metadata, *, operation_id, attempt) -> str:
        """Persist a pending event before the transport is invoked."""
        request_id = uuid.uuid4().hex
        event = {
            **metadata, "request_id": request_id, "operation_id": operation_id,
            "attempt": attempt, "started_at": time.time(),
            "status": "pending", "usage_status": "unknown", "usage": None,
        }
        with self._lock:
            self._append(event)
            self._records[request_id] = event
        return request_id

    def finish(self, request_id, **fields):
        """Append the final event; never erase the pending evidence."""
        with self._lock:
            event = {**self._records[request_id], **fields, "finished_at": time.time()}
            self._append(event)
            self._records[request_id] = event

    def records(self) -> list:
        """Return a detached snapshot so callers cannot mutate evidence."""
        with self._lock:
            return copy.deepcopy(list(self._records.values()))

    def annotate_artifact(self, request_id, artifact):
        """Link sanitized code to its raw response without retaining either text."""
        with self._lock:
            event = {**self._records[request_id], "artifact_sha256": fingerprint(artifact)}
            self._append(event)
            self._records[request_id] = event

    def summary(self, start: int = 0) -> dict:
        with self._lock:
            return summarize_requests(list(self._records.values())[start:])
