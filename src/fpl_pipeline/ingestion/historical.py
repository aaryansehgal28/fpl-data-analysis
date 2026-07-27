"""Version-pinned historical archive ingestion with raw-file provenance."""
from __future__ import annotations
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG = logging.getLogger(__name__)

class HistoricalDownloadError(RuntimeError): pass

def _session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=.5,
        status_forcelist=(429,500,502,503,504), allowed_methods=("GET",))))
    s.headers["User-Agent"] = "fpl-data-pipeline/0.2"
    return s

def _url(base: str, commit: str, season: str, filename: str) -> str:
    return f"{base.rstrip('/')}/{commit}/data/{season}/{filename}"

def download_historical_season(root: str | Path, base: str, commit: str, season: str,
                               force: bool = False, session: requests.Session | None = None) -> dict[str, Path]:
    """Download immutable source CSVs once, alongside a verifiable provenance manifest."""
    folder = Path(root) / f"season={season}" / "source=vaastav"
    folder.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    client = session or _session()
    for filename in ("gws/merged_gw.csv", "players_raw.csv"):
        local_name = filename.replace("/", "__")
        path = folder / local_name
        metadata = folder / f"{local_name}.metadata.json"
        url = _url(base, commit, season, filename)
        if path.exists() and metadata.exists() and not force:
            out[filename] = path
            LOG.info("reusing historical %s %s", season, filename)
            continue
        try:
            response = client.get(url, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise HistoricalDownloadError(f"historical download failed for {season} {filename}: {exc}") from exc
        content = response.content
        if not content.startswith(b"name,") and filename.startswith("gws/"):
            raise HistoricalDownloadError(f"unexpected historical schema for {url}")
        path.write_bytes(content)
        metadata.write_text(json.dumps({"source_url": url, "source_commit": commit, "season": season,
          "file": filename, "ingested_at": datetime.now(UTC).isoformat(),
          "sha256": hashlib.sha256(content).hexdigest()}, indent=2))
        out[filename] = path
        LOG.info("downloaded historical %s %s", season, filename)
    return out

def download_historical_seasons(root: str | Path, base: str, commit: str, seasons: Iterable[str], force: bool = False) -> dict[str, dict[str, Path]]:
    return {season: download_historical_season(root, base, commit, season, force) for season in seasons}
