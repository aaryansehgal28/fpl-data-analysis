from __future__ import annotations
import hashlib, json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

class RawStore:
    """Immutable, timestamped JSON envelopes: source truth and audit record."""
    def __init__(self, root: str | Path, season: str): self.root, self.season = Path(root), season
    def write(self, endpoint: str, payload: Any, params: dict[str, Any] | None = None) -> Path:
        at = datetime.now(UTC)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        safe = endpoint.strip("/").replace("/", "__").replace("?", "__")
        folder = self.root / f"season={self.season}" / f"endpoint={safe}" / f"date={at:%Y-%m-%d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{at:%H%M%S_%f}_{digest[:12]}.json"
        envelope = {"metadata": {"ingested_at": at.isoformat(), "season": self.season,
                    "endpoint": endpoint, "request_parameters": params or {}, "payload_sha256": digest}, "payload": payload}
        path.write_text(json.dumps(envelope, separators=(",", ":"), default=str))
        return path
    def latest(self, endpoint: str) -> Path | None:
        safe = endpoint.strip("/").replace("/", "__").replace("?", "__")
        paths = list((self.root / f"season={self.season}" / f"endpoint={safe}").glob("**/*.json"))
        return max(paths, default=None, key=lambda p: p.stat().st_mtime)
