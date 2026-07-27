from __future__ import annotations

import logging
import time
from typing import Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from . import endpoints as ep

LOG = logging.getLogger(__name__)

class FPLApiError(RuntimeError): """A contextual API failure suitable for retry/alerting."""

class FPLClient:
    """Small, rate-limited client for the public FPL web API (not a stable contract)."""
    base_url = "https://fantasy.premierleague.com/api/"

    def __init__(self, timeout: float = 20, max_retries: int = 4, min_interval: float = .4,
                 session: requests.Session | None = None) -> None:
        self.timeout, self.min_interval, self._last_request = timeout, min_interval, 0.0
        self.session = session or requests.Session()
        retry = Retry(total=max_retries, backoff_factor=.5, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET",), raise_on_status=False)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({"Accept": "application/json", "User-Agent": "fpl-data-pipeline/0.1"})

    def get(self, path: str) -> dict[str, Any] | list[Any]:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0: time.sleep(wait)
        url = self.base_url + path
        try:
            response = self.session.get(url, timeout=self.timeout)
            self._last_request = time.monotonic()
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise FPLApiError(f"GET {url} failed: {exc}") from exc
        if not isinstance(payload, (dict, list)):
            raise FPLApiError(f"GET {url} returned {type(payload).__name__}, expected JSON object/list")
        LOG.info("fetched %s", path)
        return payload

    def get_bootstrap_static(self): return self.get(ep.BOOTSTRAP)
    def get_fixtures(self): return self.get(ep.FIXTURES)
    def get_gameweek_live(self, gameweek_id: int): return self.get(ep.event_live(gameweek_id))
    def get_player_summary(self, player_id: int): return self.get(ep.element_summary(player_id))
    def get_manager(self, manager_id: int): return self.get(ep.entry(manager_id))
    def get_manager_history(self, manager_id: int): return self.get(ep.entry_history(manager_id))
    def get_manager_picks(self, manager_id: int, gameweek_id: int): return self.get(ep.entry_picks(manager_id, gameweek_id))
    def get_manager_transfers(self, manager_id: int): return self.get(ep.entry_transfers(manager_id))
    def get_league_standings(self, league_id: int, page: int = 1): return self.get(ep.league_classic(league_id, page))
