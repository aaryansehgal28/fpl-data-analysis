from __future__ import annotations
import argparse, logging
from pathlib import Path
import duckdb, yaml
from .optimiser import optimise_squad
from .storage import load_duckdb
from .transformation.core import write_parquet

def run(config_path: str = "config/config.yaml", objective: str = "balanced") -> None:
    cfg=yaml.safe_load(Path(config_path).read_text())
    with duckdb.connect(cfg["paths"]["database"],read_only=True) as con: recommendations=con.execute("SELECT * FROM player_recommendation").df()
    squad, starters, bench=optimise_squad(recommendations,objective)
    tables={"recommended_squad":squad,"recommended_starting_xi":starters,"recommended_bench":bench}
    write_parquet(tables,Path(cfg["paths"]["processed"])); load_duckdb(cfg["paths"]["database"],tables)
    print(squad[["player_name","team_name","position_id","current_price_tenths","is_starter","is_captain","is_vice_captain","bench_order"]].to_string(index=False))

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Optimise a legal 15-player FPL squad")
    p.add_argument("--config",default="config/config.yaml"); p.add_argument("--objective",choices=("points","value","balanced","risk_adjusted","differential"),default="balanced")
    a=p.parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s: %(message)s"); run(a.config,a.objective)
