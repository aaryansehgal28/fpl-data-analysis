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

def calibrate_position_priors(historical_seasons: pd.DataFrame) -> dict:
    """Compute position-specific Bayesian priors from historical seasons.
    
    Returns dict of {(position, metric): (prior_mean, prior_std, prior_equivalent_90s)}
    """
    import numpy as np
    
    x = historical_seasons.dropna(subset=["position"]).copy()
    x["effective_90s"] = x["minutes"].fillna(0) / 90
    
    # Define metrics and their denominators
    metrics = {
        "goals_per_90": ("goals_scored", "effective_90s"),
        "assists_per_90": ("assists", "effective_90s"),
        "clean_sheets_per_90": ("clean_sheets", "effective_90s"),
        "bps_per_90": ("bps", "effective_90s"),
        "influence_per_90": ("influence", "effective_90s"),
    }
    
    priors = {}
    for position in ["GK", "DEF", "MID", "FWD"]:
        pos_data = x[x["position"] == position].copy()
        if len(pos_data) == 0:
            continue
            
        for metric_name, (numerator, denominator) in metrics.items():
            if numerator not in pos_data.columns or denominator not in pos_data.columns:
                continue
            
            # Compute per-90 rate with min 450 minutes filter
            pos_data_filtered = pos_data[pos_data[denominator] > 5].copy()  # >450 min ≈ >5 eff_90s
            if len(pos_data_filtered) == 0:
                continue
            
            per_90_rates = pos_data_filtered[numerator] / pos_data_filtered[denominator]
            per_90_rates = per_90_rates.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(per_90_rates) == 0:
                continue
            
            # Use median and IQR for robustness (less sensitive to outliers)
            median = per_90_rates.quantile(0.50)
            q1 = per_90_rates.quantile(0.25)
            q3 = per_90_rates.quantile(0.75)
            iqr = q3 - q1
            robust_std = iqr / 1.35  # Convert IQR to std equivalent
            robust_std = max(robust_std, 0.05)  # Floor to avoid zero
            
            # Prior equivalent 90s: position- and metric-specific shrinkage strength
            # Rare events (DEF goals): 100-150; Common events (FWD goals): 25-50
            prior_equiv_90s_map = {
                ("GK", "goals_per_90"): 150,
                ("GK", "assists_per_90"): 100,
                ("GK", "clean_sheets_per_90"): 40,
                ("GK", "bps_per_90"): 30,
                ("DEF", "goals_per_90"): 140,
                ("DEF", "assists_per_90"): 80,
                ("DEF", "clean_sheets_per_90"): 35,
                ("DEF", "bps_per_90"): 35,
                ("MID", "goals_per_90"): 60,
                ("MID", "assists_per_90"): 50,
                ("MID", "clean_sheets_per_90"): 120,
                ("MID", "bps_per_90"): 40,
                ("FWD", "goals_per_90"): 40,
                ("FWD", "assists_per_90"): 60,
                ("FWD", "clean_sheets_per_90"): 200,
                ("FWD", "bps_per_90"): 50,
                ("GK", "influence_per_90"): 50,
                ("DEF", "influence_per_90"): 50,
                ("MID", "influence_per_90"): 40,
                ("FWD", "influence_per_90"): 40,
            }
            prior_equiv = prior_equiv_90s_map.get((position, metric_name), 50)
            priors[(position, metric_name)] = (median, robust_std, prior_equiv)
    
    return priors

def compute_per_90_metrics_with_shrinkage(
    player_gw: pd.DataFrame,
    position: str,
    metric_name: str,
    priors: dict
) -> pd.DataFrame:
    """Compute per-90 rate with Bayesian shrinkage for a specific player group and metric.
    
    Args:
        player_gw: gameweek-level player data
        position: position filter (GK, DEF, MID, FWD)
        metric_name: metric key (e.g., "goals_per_90")
        priors: dict from calibrate_position_priors()
    
    Returns:
        DataFrame with posterior estimates per player-season
    """
    import numpy as np
    
    x = player_gw[player_gw["position"] == position].copy() if position else player_gw.copy()
    x = x.dropna(subset=["identity_key", "season"]).copy()
    
    # Map metric name to source columns
    metric_map = {
        "goals_per_90": "goals_scored",
        "assists_per_90": "assists",
        "clean_sheets_per_90": "clean_sheets",
        "bps_per_90": "bps",
        "influence_per_90": "influence",
    }
    
    if metric_name not in metric_map:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    numerator_col = metric_map[metric_name]
    x["effective_90s"] = x["minutes"].fillna(0) / 90
    
    # Aggregate by player-season
    agg_dict = {
        numerator_col: "sum",
        "effective_90s": "sum",
        "player_id": "last",
        "player_name": "last",
        "team": "last",
    }
    grouped = x.groupby(["season", "identity_key", "stable_player_id"], dropna=False, as_index=False).agg(agg_dict)
    
    # Filter: minimum 450 minutes (5 effective_90s)
    grouped = grouped[grouped["effective_90s"] > 5].copy()
    
    # Compute empirical rate
    grouped[f"empirical_{metric_name}"] = grouped[numerator_col] / grouped["effective_90s"]
    grouped[f"empirical_{metric_name}"] = grouped[f"empirical_{metric_name}"].replace([np.inf, -np.inf], np.nan)
    
    # Get prior from calibration
    prior_key = (position, metric_name) if position else (None, metric_name)
    if prior_key in priors:
        prior_mean, prior_std, prior_equiv_90s = priors[prior_key]
    else:
        # Fallback prior if not calibrated
        prior_mean = grouped[f"empirical_{metric_name}"].median()
        prior_std = grouped[f"empirical_{metric_name}"].std()
        prior_equiv_90s = 50
    
    prior_mean = prior_mean if pd.notna(prior_mean) else 0.0
    prior_std = prior_std if pd.notna(prior_std) else 0.1
    
    # Bayesian Normal-mean shrinkage
    # posterior_mean = (empirical_mean × n + prior_mean × prior_equiv_90s) / (n + prior_equiv_90s)
    grouped[f"posterior_{metric_name}"] = (
        (grouped[f"empirical_{metric_name}"] * grouped["effective_90s"] + prior_mean * prior_equiv_90s) /
        (grouped["effective_90s"] + prior_equiv_90s)
    ).fillna(prior_mean)
    
    # Posterior uncertainty
    grouped[f"posterior_{metric_name}_std"] = (
        (prior_std / np.sqrt(grouped["effective_90s"] + prior_equiv_90s)).clip(lower=0.01)
    )
    
    # Reliability score (0-1; higher = more reliable)
    grouped[f"reliability_score_{metric_name}"] = (
        grouped["effective_90s"] / (grouped["effective_90s"] + prior_equiv_90s)
    ).clip(0, 1)
    
    # Effective 90s for this metric (may differ if sparse)
    grouped[f"effective_90s_{metric_name}"] = grouped["effective_90s"]
    
    return grouped
