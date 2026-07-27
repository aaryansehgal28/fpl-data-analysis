from __future__ import annotations
import argparse, logging
from pathlib import Path
import duckdb, yaml
from .models import bayesian_player_value, build_recommendations
from .storage import load_duckdb
from .transformation.core import write_parquet

def run(config_path: str = "config/config.yaml", horizon: int = 5, as_of_gameweek: int | None = None, target_gameweek: int | None = None) -> None:
    cfg=yaml.safe_load(Path(config_path).read_text()); h=cfg["historical"]
    with duckdb.connect(cfg["paths"]["database"],read_only=True) as con:
        historical=con.execute("SELECT * FROM fact_player_season").df(); players=con.execute("SELECT * FROM dim_player").df(); teams=con.execute("SELECT * FROM dim_team").df(); fixtures=con.execute("SELECT * FROM fixture_run").df(); events=con.execute("SELECT * FROM dim_gameweek").df()
        current=con.execute("SELECT * FROM fact_player_gameweek").df() if "fact_player_gameweek" in [r[0] for r in con.execute("SHOW TABLES").fetchall()] else None
    bayes=bayesian_player_value(historical,cfg["season"],h["decay_rate"],h["prior_equivalent_90s"])
    if target_gameweek is None:
        unfinished=events[~events.finished.fillna(False)].sort_values("gameweek_id")
        target_gameweek=int(unfinished.gameweek_id.iloc[0]) if not unfinished.empty else int(events.gameweek_id.max())
    rec, projections=build_recommendations(players,teams,bayes,fixtures,current,horizon,as_of_gameweek,target_gameweek)
    tables={"player_bayesian_value":bayes,"player_recommendation":rec,"player_projection":projections}
    write_parquet(tables,Path(cfg["paths"]["processed"])); load_duckdb(cfg["paths"]["database"],tables)

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Calculate Bayesian FPL player recommendations")
    p.add_argument("--config",default="config/config.yaml"); p.add_argument("--horizon",type=int,choices=(1,3,5),default=5); p.add_argument("--as-of-gameweek",type=int); p.add_argument("--target-gameweek",type=int)
    a=p.parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s"); run(a.config,a.horizon,a.as_of_gameweek,a.target_gameweek)
