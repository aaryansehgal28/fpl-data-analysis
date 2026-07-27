from __future__ import annotations
import argparse, logging
from pathlib import Path
import yaml
from .ingestion.historical import download_historical_seasons
from .storage import load_duckdb
from .transformation.core import write_parquet
from .transformation.historical import player_season_aggregates, standardise_historical_season

def run(config_path: str = "config/config.yaml", force: bool = False) -> None:
    cfg=yaml.safe_load(Path(config_path).read_text()); hist=cfg["historical"]
    downloaded=download_historical_seasons(cfg["paths"]["raw"],hist["source_repository"],hist["commit"],hist["seasons"],force)
    facts=[standardise_historical_season(s,paths["gws/merged_gw.csv"],paths["players_raw.csv"]) for s,paths in downloaded.items()]
    import pandas as pd
    fact=pd.concat(facts,ignore_index=True); season=player_season_aggregates(fact)
    tables={"fact_player_gameweek_historical":fact,"fact_player_season":season}
    write_parquet(tables,Path(cfg["paths"]["processed"])); load_duckdb(cfg["paths"]["database"],tables)

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Backfill version-pinned FPL historical seasons")
    p.add_argument("--config",default="config/config.yaml"); p.add_argument("--force",action="store_true")
    a=p.parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s"); run(a.config,a.force)
