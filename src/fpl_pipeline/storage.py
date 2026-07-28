from __future__ import annotations
from pathlib import Path
import duckdb, pandas as pd, json

def load_duckdb(database: str | Path, tables: dict[str,pd.DataFrame]) -> None:
    with duckdb.connect(str(database)) as con:
        for name, df in tables.items():
            con.register("incoming", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM incoming")
            con.unregister("incoming")

def load_bootstrap_snapshot(snapshot_path: str | Path) -> dict:
    """Load and parse FPL bootstrap-static JSON snapshot.
    
    Returns dict with 'elements', 'element_types', 'teams', 'events' keys.
    """
    with open(snapshot_path) as f:
        data = json.load(f)
    return data.get('payload', data)

def extract_players_from_bootstrap(bootstrap_data: dict) -> pd.DataFrame:
    """Extract player data from bootstrap-static payload.
    
    Returns DataFrame with columns: player_id, player_name, position_id, value_tenths
    """
    players_data = []
    
    # Create position mapping
    element_types = {et['id']: et['singular_name'] for et in bootstrap_data.get('element_types', [])}
    
    for element in bootstrap_data.get('elements', []):
        player_id = element.get('id')
        first_name = element.get('first_name', '')
        last_name = element.get('second_name', '')
        player_name = f"{first_name} {last_name}".strip()
        position_id = element.get('element_type')
        now_cost = element.get('now_cost', 0)  # Cost in tenths (e.g., 50 = £5.0m)
        
        players_data.append({
            'player_id': player_id,
            'player_name': player_name,
            'position_id': position_id,
            'position': element_types.get(position_id, f'POS_{position_id}'),
            'value_tenths': now_cost,
            'value_millions': now_cost / 10
        })
    
    return pd.DataFrame(players_data)
