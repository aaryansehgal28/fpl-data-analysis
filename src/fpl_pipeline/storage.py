from __future__ import annotations
from pathlib import Path
import duckdb, pandas as pd

def load_duckdb(database: str | Path, tables: dict[str,pd.DataFrame]) -> None:
    with duckdb.connect(str(database)) as con:
        for name, df in tables.items():
            con.register("incoming", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM incoming")
            con.unregister("incoming")
