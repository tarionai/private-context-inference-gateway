"""Append-only, hash-chained audit store.

Each record carries the request, the full context-assembly decision (included and
excluded items with reasons), the route taken, cost, and latency. Records are
chained by SHA-256 so the audit is tamper-evident — this is the "hash-verifiable"
property the artifact claims.

This is a declared side-effect boundary module: clock reads and file writes are
confined here. Pure logic elsewhere never reads the wall clock.

Production backend is PostgreSQL (Neon), reusing the MemLearn pattern; the
default JSONL backend keeps the evidence run credential-free and locally
verifiable. Two timestamps are recorded: `event_at` (business time, the request
instant) and `recorded_at` (wall-clock write time).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from gateway.contract import InferenceRequest, InferenceResponse

_GENESIS = "0" * 64


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash(prev_hash: str, body: str) -> str:
    return hashlib.sha256(f"{prev_hash}\n{body}".encode("utf-8")).hexdigest()


class AuditStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return _GENESIS
        last = _GENESIS
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    last = json.loads(line)["record_hash"]
        return last

    def append(
        self,
        request: InferenceRequest,
        response: InferenceResponse,
        event_at: datetime,
    ) -> str:
        """Append one tamper-evident audit record. Returns its record hash."""
        body_payload = {
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
            "event_at": event_at.astimezone(timezone.utc).isoformat(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        body = _canonical(body_payload)
        prev_hash = self._last_hash()
        record_hash = _hash(prev_hash, body)
        record = {**body_payload, "prev_hash": prev_hash, "record_hash": record_hash}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical(record) + "\n")
        return record_hash

    def verify_chain(self) -> tuple[bool, int]:
        """Re-derive every hash. Returns (intact, record_count)."""
        if not self.path.exists():
            return True, 0
        prev_hash = _GENESIS
        count = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                body = _canonical(
                    {
                        "request": record["request"],
                        "response": record["response"],
                        "event_at": record["event_at"],
                        "recorded_at": record["recorded_at"],
                    }
                )
                expected = _hash(prev_hash, body)
                if record["prev_hash"] != prev_hash or record["record_hash"] != expected:
                    return False, count
                prev_hash = expected
                count += 1
        return True, count
