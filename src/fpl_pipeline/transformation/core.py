from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from ..validation.checks import require_fk, require_unique

PLAYER_COLUMNS = ["player_id", "player_code", "first_name", "second_name", "web_name", "birth_date", "team_id", "position_id", "current_price", "status", "chance_of_playing_next_round", "news", "ownership_percent", "transfers_in_event", "transfers_out_event"]

def _envelope(path: Path) -> dict:
    return json.loads(path.read_text())["payload"]
def _frame(rows, rename: dict[str, str], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows).rename(columns=rename)
    for col in columns:
        if col not in df: df[col] = pd.NA
    return df[columns]

def bootstrap_tables(raw_path: Path) -> dict[str, pd.DataFrame]:
    p = _envelope(raw_path)
    players = _frame(p["elements"], {"id":"player_id", "code":"player_code", "team":"team_id", "element_type":"position_id", "now_cost":"current_price", "selected_by_percent":"ownership_percent"}, PLAYER_COLUMNS)
    # FPL costs are tenths of a million; preserving integer avoids floating inaccuracies.
    players["current_price"] = pd.to_numeric(players.current_price, errors="coerce").astype("Int64")
    teams = _frame(p["teams"], {"id":"team_id", "name":"team_name", "short_name":"short_name"},
        ["team_id","team_name","short_name","strength","strength_overall_home","strength_overall_away","strength_attack_home","strength_attack_away","strength_defence_home","strength_defence_away"])
    positions = _frame(p["element_types"], {"id":"position_id", "singular_name":"position_name", "squad_select":"squad_position_rules"}, ["position_id","position_name","squad_position_rules"])
    events = _frame(p["events"], {"id":"gameweek_id"}, ["gameweek_id","deadline_time","finished","data_checked","average_score","highest_score","highest_scoring_entry","most_selected","most_transferred_in","most_captained","most_vice_captained"])
    events["deadline_time"] = pd.to_datetime(events.deadline_time, utc=True)
    require_unique(players,["player_id"],"dim_player"); require_unique(teams,["team_id"],"dim_team"); require_unique(events,["gameweek_id"],"dim_gameweek")
    require_fk(players,"team_id",teams,"team_id","dim_player"); require_fk(players,"position_id",positions,"position_id","dim_player")
    return {"dim_player":players,"dim_team":teams,"dim_position":positions,"dim_gameweek":events}

def fixture_table(raw_path: Path) -> pd.DataFrame:
    df = _frame(_envelope(raw_path), {"id":"fixture_id", "event":"gameweek_id", "team_h":"home_team_id", "team_a":"away_team_id", "team_h_score":"home_score", "team_a_score":"away_score", "team_h_difficulty":"home_difficulty", "team_a_difficulty":"away_difficulty"},
      ["fixture_id","gameweek_id","kickoff_time","home_team_id","away_team_id","home_score","away_score","home_difficulty","away_difficulty","finished","finished_provisional","minutes"])
    df["kickoff_time"] = pd.to_datetime(df.kickoff_time, utc=True)
    require_unique(df,["fixture_id"],"fact_fixture")
    return df

def player_gameweek_table(raw_path: Path, gameweek_id: int, bootstrap_path: Path | None = None) -> pd.DataFrame:
    payload = _envelope(raw_path)
    rows=[]
    for element in payload["elements"]:
        row={"player_id":element["id"],"gameweek_id":gameweek_id, **element.get("stats",{})}
        rows.append(row)
    df=pd.DataFrame(rows)
    columns=["player_id","gameweek_id","minutes","goals_scored","assists","clean_sheets","goals_conceded","own_goals","penalties_saved","penalties_missed","yellow_cards","red_cards","saves","bonus","bps","influence","creativity","threat","ict_index","total_points","value","selected","transfers_in","transfers_out"]
    for col in columns:
        if col not in df: df[col]=pd.NA
    df=df[columns]
    # value/selected/transfers are time snapshots/flows from live endpoint. Bootstrap is an alternative current snapshot.
    require_unique(df,["player_id","gameweek_id"],"fact_player_gameweek")
    return df

def write_parquet(tables: dict[str,pd.DataFrame], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items(): df.to_parquet(root / f"{name}.parquet", index=False)

def manager_picks_table(raw_path: Path, manager_id: int, gameweek_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn entry picks into a manager-gameweek snapshot and its player bridge."""
    p = _envelope(raw_path); entry = p.get("entry_history", {})
    manager = pd.DataFrame([{ "manager_id":manager_id,"gameweek_id":gameweek_id,
        "points":entry.get("points"),"rank":entry.get("rank"),"overall_rank":entry.get("overall_rank"),
        "bank":entry.get("bank"),"value":entry.get("value"),"transfers":entry.get("event_transfers"),
        "transfer_cost":entry.get("event_transfers_cost"),"points_on_bench":entry.get("points_on_bench"),
        "active_chip":p.get("active_chip")}])
    bridge = _frame(p.get("picks", []), {"element":"player_id","is_captain":"is_captain","is_vice_captain":"is_vice_captain"},
        ["player_id","position","multiplier","is_captain","is_vice_captain","purchase_price","selling_price"])
    bridge.insert(0,"gameweek_id",gameweek_id); bridge.insert(0,"manager_id",manager_id)
    require_unique(bridge,["manager_id","gameweek_id","player_id"],"bridge_manager_player")
    return manager, bridge
