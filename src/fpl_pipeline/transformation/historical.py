"""Historical source normalisation and season aggregates."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..validation.checks import require_unique

POSITION_MAP = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD", "GK":"GK", "GKP":"GK", "DEF":"DEF", "MID":"MID", "FWD":"FWD"}

GW_NUMERIC = ["minutes","total_points","goals_scored","assists","clean_sheets","bonus","bps","influence","creativity","threat","ict_index","value","transfers_in","transfers_out","selected","starts"]

def standardise_historical_season(season: str, merged_path: Path, players_path: Path) -> pd.DataFrame:
    gw = pd.read_csv(merged_path, low_memory=False)
    players = pd.read_csv(players_path, low_memory=False)
    identity = players[[c for c in ("id","code","birth_date","element_type","first_name","second_name","web_name") if c in players]].rename(columns={"id":"player_id", "code":"stable_player_id", "element_type":"position_id"})
    gw = gw.rename(columns={"element":"player_id", "value":"price_tenths", "name":"player_name"})
    # Archive files currently expose both round and GW; prefer GW but retain round as fallback.
    if "GW" in gw and "round" in gw:
        gw["gameweek_id"] = pd.to_numeric(gw["GW"], errors="coerce").combine_first(pd.to_numeric(gw["round"], errors="coerce"))
    elif "GW" in gw: gw["gameweek_id"] = gw["GW"]
    elif "round" in gw: gw["gameweek_id"] = gw["round"]
    else: raise ValueError(f"{season}: merged gameweek file has no gameweek column")
    for c in GW_NUMERIC:
        source = "price_tenths" if c == "value" else c
        if source not in gw: gw[source] = pd.NA
        gw[source] = pd.to_numeric(gw[source], errors="coerce")
    gw = gw.merge(identity, on="player_id", how="left", validate="many_to_one")
    gw["season"] = season
    gw["position"] = gw.get("position").map(POSITION_MAP).fillna(gw.get("position_id").map(POSITION_MAP))
    gw["stable_player_id"] = gw["stable_player_id"].astype("Int64")
    # `code` is the reliable cross-season key. Fallback keys deliberately remain flagged uncertain.
    fallback = (gw["player_name"].astype(str).str.lower().str.replace(r"\W+", "", regex=True) + "|" + gw["position"].fillna("UNK"))
    gw["identity_key"] = gw["stable_player_id"].astype("string").where(gw["stable_player_id"].notna(), fallback)
    gw["identity_confidence"] = gw["stable_player_id"].notna().map({True:"high", False:"low"})
    gw["price_million"] = gw["price_tenths"] / 10
    gw["kickoff_time"] = pd.to_datetime(gw.get("kickoff_time"), utc=True, errors="coerce")
    columns = ["season","player_id","stable_player_id","identity_key","identity_confidence","player_name","birth_date","team","position","position_id","gameweek_id","fixture","kickoff_time","minutes","total_points","goals_scored","assists","clean_sheets","bonus","bps","influence","creativity","threat","ict_index","price_tenths","price_million","transfers_in","transfers_out","selected","starts"]
    for c in columns:
        if c not in gw: gw[c] = pd.NA
    # Source is fixture grain. Aggregate double-gameweeks to the project’s player-event grain;
    # stats/flows sum, while price/ownership are end-of-event snapshots.
    sums = [c for c in ["minutes","total_points","goals_scored","assists","clean_sheets","bonus","bps","influence","creativity","threat","ict_index","transfers_in","transfers_out","starts"] if c in gw]
    last = [c for c in columns if c not in sums + ["fixture","kickoff_time"]]
    ordered = gw.sort_values("kickoff_time")
    fact = ordered.groupby(["season","player_id","gameweek_id"], dropna=False, as_index=False).agg({**{c:"sum" for c in sums}, **{c:"last" for c in last}})
    require_unique(fact,["season","player_id","gameweek_id"],"fact_player_gameweek_historical")
    return fact

def player_season_aggregates(historical: pd.DataFrame) -> pd.DataFrame:
    x = historical.copy(); x["effective_90s"] = x["minutes"].fillna(0) / 90
    keys = ["season","identity_key","stable_player_id","identity_confidence","position"]
    result = x.groupby(keys, dropna=False, as_index=False).agg(
        player_id=("player_id","last"), player_name=("player_name","last"), team=("team","last"),
        total_points=("total_points","sum"), total_minutes=("minutes","sum"), goals=("goals_scored","sum"),
        assists=("assists","sum"), clean_sheets=("clean_sheets","sum"), bonus=("bonus","sum"),
        average_price_million=("price_million","mean"), minimum_price_million=("price_million","min"), maximum_price_million=("price_million","max"),
        total_appearances=("gameweek_id","count"), starts=("starts","sum"), effective_90s=("effective_90s","sum"))
    result["points_per_game"] = result.total_points / result.total_appearances.replace(0,pd.NA)
    # Average GW price is the denominator: it reflects the cost paid during the points-generating period.
    result["season_ppm"] = result.total_points / result.average_price_million.replace(0,pd.NA)
    result["points_per_90"] = result.total_points / result.effective_90s.replace(0,pd.NA)
    require_unique(result,["season","identity_key"],"fact_player_season")
    return result
