from __future__ import annotations
import argparse, logging
from datetime import timedelta
from pathlib import Path
import yaml
from .api.client import FPLClient
from .ingestion.download import ingest_event_live, ingest_public_core
from .ingestion.raw_store import RawStore
from .transformation.core import bootstrap_tables, fixture_table, player_gameweek_table, write_parquet
from .transformation.analytics import fixture_run, player_analytics
from .storage import load_duckdb

def run(config_path: str, event: int | None = None, force: bool = False) -> None:
    cfg=yaml.safe_load(Path(config_path).read_text()); paths=cfg["paths"]
    store=RawStore(paths["raw"],cfg["season"]); client=FPLClient(**{"timeout":cfg["http"]["timeout_seconds"],"min_interval":cfg["http"]["min_request_interval_seconds"],"max_retries":cfg["http"]["max_retries"]})
    bootstrap, fixtures=ingest_public_core(client,store,timedelta(hours=cfg["refresh"]["bootstrap_max_age_hours"]),timedelta(hours=cfg["refresh"]["fixtures_max_age_hours"]),force)
    tables=bootstrap_tables(bootstrap); tables["fact_fixture"]=fixture_table(fixtures)
    if event:
        live=ingest_event_live(client,store,event,force); tables["fact_player_gameweek"]=player_gameweek_table(live,event,bootstrap)
        tables.update(player_analytics(tables["fact_player_gameweek"]))
    tables["fixture_run"]=fixture_run(tables["fact_fixture"])
    write_parquet(tables,Path(paths["processed"])); load_duckdb(paths["database"],tables)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/config.yaml"); p.add_argument("--event",type=int); p.add_argument("--force",action="store_true")
    args=p.parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s"); run(args.config,args.event,args.force)
