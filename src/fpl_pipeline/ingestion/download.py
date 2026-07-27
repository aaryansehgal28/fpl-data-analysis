from __future__ import annotations
from datetime import UTC, datetime, timedelta
from pathlib import Path
from ..api.client import FPLClient
from ..api import endpoints as ep
from .raw_store import RawStore

def fetch_if_stale(client: FPLClient, store: RawStore, path: str, max_age: timedelta, force: bool = False) -> Path:
    previous = store.latest(path)
    if previous and not force and datetime.now(UTC) - datetime.fromtimestamp(previous.stat().st_mtime, UTC) < max_age:
        return previous
    return store.write(path, client.get(path))

def ingest_public_core(client: FPLClient, store: RawStore, bootstrap_age: timedelta, fixtures_age: timedelta, force: bool = False) -> list[Path]:
    return [fetch_if_stale(client, store, ep.BOOTSTRAP, bootstrap_age, force),
            fetch_if_stale(client, store, ep.FIXTURES, fixtures_age, force)]

def ingest_event_live(client: FPLClient, store: RawStore, event: int, force: bool = False) -> Path:
    return fetch_if_stale(client, store, ep.event_live(event), timedelta(minutes=15), force)
