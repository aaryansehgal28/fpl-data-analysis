from __future__ import annotations
import pandas as pd
import numpy as np

def player_analytics(player_gw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    x=player_gw.sort_values(["player_id","gameweek_id"]).copy()
    for n in (3,5,10):
        x[f"rolling_{n}_gameweek_points"]=x.groupby("player_id").total_points.transform(lambda s:s.rolling(n,min_periods=1).sum())
    for metric in ("minutes","goals_scored","assists","bonus"):
        x[f"rolling_{metric}"]=x.groupby("player_id")[metric].transform(lambda s:s.rolling(5,min_periods=1).sum())
    form=x[["player_id","gameweek_id",*[c for c in x if c.startswith("rolling_")]]]
    value=x[["player_id","gameweek_id","total_points","value"]].copy()
    value["points_per_million"]=value.total_points/(value.value/10).replace(0,pd.NA)
    value["rolling_points_per_million"]=value.groupby("player_id").points_per_million.transform(lambda s:s.rolling(5,min_periods=1).mean())
    return {"player_form":form,"player_value":value.drop(columns=["total_points","value"])}

def fixture_run(fixtures: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for team in sorted(set(fixtures.home_team_id.dropna()) | set(fixtures.away_team_id.dropna())):
        f=fixtures.assign(difficulty=fixtures.apply(lambda r:r.away_difficulty if r.home_team_id==team else r.home_difficulty,axis=1))
        f=f[(f.home_team_id==team)|(f.away_team_id==team)].sort_values("gameweek_id")
        f["is_home"] = f.home_team_id == team
        for i,r in f.reset_index(drop=True).iterrows():
            future=f.iloc[i:i+5]
            rows.append({
                "team_id": team,
                "gameweek_id": r.gameweek_id,
                "next_1_fixture_difficulty": future.difficulty.iloc[0] if len(future) > 0 else 3.0,
                "next_3_fixture_average_difficulty": future.difficulty.iloc[:3].astype(float).mean() if len(future) > 0 else 3.0,
                "next_5_fixture_average_difficulty": future.difficulty.astype(float).mean() if len(future) > 0 else 3.0,
                "next_fixture_is_home": future.is_home.iloc[0] if len(future) > 0 else np.nan,
            })
    return pd.DataFrame(rows)

def compute_team_attacking_defensive_strength(player_gw: pd.DataFrame, season: str = None) -> pd.DataFrame:
    """Compute team-level attacking and defensive strength metrics.
    
    Args:
        player_gw: gameweek-level player data
        season: optional season filter
    
    Returns:
        DataFrame with team strength metrics per team-season
    """
    x = player_gw.dropna(subset=["team", "position"]).copy()
    if season:
        x = x[x["season"] == season]
    
    x["effective_90s"] = x["minutes"].fillna(0) / 90
    
    # Filter for players with sufficient minutes
    x = x[x["effective_90s"] > 2].copy()  # Minimum 180 minutes
    
    rows = []
    for team in sorted(set(x["team"].dropna())):
        team_data = x[x["team"] == team]
        team_season = team_data["season"].iloc[0] if "season" in team_data.columns else season
        
        # Attacking strength: mean goals + assists for MID/FWD
        attacking = team_data[team_data["position"].isin(["MID", "FWD"])]
        if len(attacking) > 0:
            attack_strength = (
                attacking["goals_scored"].sum() / attacking["effective_90s"].sum() +
                attacking["assists"].sum() / attacking["effective_90s"].sum()
            ) / 2
            goal_rate = attacking["goals_scored"].sum() / attacking["effective_90s"].sum()
            assists_rate = attacking["assists"].sum() / attacking["effective_90s"].sum()
        else:
            attack_strength = np.nan
            goal_rate = np.nan
            assists_rate = np.nan
        
        # Defensive strength: CS frequency and defensive stats for GK/DEF
        defensive = team_data[team_data["position"].isin(["GK", "DEF"])]
        if len(defensive) > 0:
            cs_appearances = (defensive["clean_sheets"] > 0).sum()
            total_appearances = len(defensive)
            cs_frequency = cs_appearances / total_appearances if total_appearances > 0 else 0
            defence_strength = cs_frequency
        else:
            defence_strength = np.nan
            cs_frequency = np.nan
        
        rows.append({
            "team": team,
            "season": team_season,
            "team_attack_strength": attack_strength,
            "team_goal_rate": goal_rate,
            "team_assists_rate": assists_rate,
            "team_defence_strength": defence_strength,
            "team_cs_frequency": cs_frequency,
        })
    
    return pd.DataFrame(rows)

def compute_home_away_ratios(player_gw: pd.DataFrame) -> pd.DataFrame:
    """Compute home vs. away performance ratios for each player.
    
    Args:
        player_gw: gameweek-level player data with 'is_home' flag
    
    Returns:
        DataFrame with home/away ratios per player
    """
    x = player_gw.dropna(subset=["player_id", "is_home"]).copy()
    
    rows = []
    for player_id in sorted(set(x["player_id"])):
        player_data = x[x["player_id"] == player_id]
        
        home_data = player_data[player_data["is_home"] == True]
        away_data = player_data[player_data["is_home"] == False]
        
        # Compute ratios (avoid division by zero)
        metrics = ["goals_scored", "assists", "clean_sheets", "total_points"]
        ratios = {}
        
        for metric in metrics:
            home_sum = home_data[metric].sum()
            home_90s = (home_data["minutes"].sum() or 0) / 90
            away_sum = away_data[metric].sum()
            away_90s = (away_data["minutes"].sum() or 0) / 90
            
            if away_90s > 0 and home_90s > 0:
                home_rate = home_sum / home_90s if home_90s > 0 else 0
                away_rate = away_sum / away_90s if away_90s > 0 else 0
                ratio = home_rate / away_rate if away_rate > 0 else 1.0
            else:
                ratio = 1.0
            
            ratios[f"home_away_ratio_{metric}"] = ratio
        
        rows.append({"player_id": player_id, **ratios})
    
    return pd.DataFrame(rows)

def compute_form_trends(player_gw: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """Compute recent form trends for each player at each gameweek.
    
    Args:
        player_gw: gameweek-level player data, sorted chronologically
        window_size: rolling window size for recent form (default 5 GWs)
    
    Returns:
        DataFrame with form metrics per player-gameweek
    """
    x = player_gw.sort_values(["player_id", "season", "gameweek_id"]).copy()
    x = x.dropna(subset=["player_id", "gameweek_id"]).copy()
    
    # Compute rolling means
    x["recent_points_mean"] = x.groupby("player_id")["total_points"].transform(
        lambda s: s.rolling(window_size, min_periods=1).mean()
    )
    x["recent_goals_mean"] = x.groupby("player_id")["goals_scored"].transform(
        lambda s: s.rolling(window_size, min_periods=1).mean()
    )
    x["recent_assists_mean"] = x.groupby("player_id")["assists"].transform(
        lambda s: s.rolling(window_size, min_periods=1).mean()
    )
    x["recent_minutes_mean"] = x.groupby("player_id")["minutes"].transform(
        lambda s: s.rolling(window_size, min_periods=1).mean()
    )
    
    # Compute season average (for comparison)
    x["season_points_mean"] = x.groupby(["player_id", "season"])["total_points"].transform("mean")
    
    # Form trend: direction and magnitude
    x["form_trend_magnitude"] = (
        (x["recent_points_mean"] - x["season_points_mean"]) / 
        (x["season_points_mean"] + 0.1)  # Avoid division by zero
    ).clip(-1, 1)  # Clip to [-1, 1]
    
    # Form direction: -1 (declining), 0 (stable), +1 (improving)
    x["form_trend_direction"] = 0
    x.loc[x["form_trend_magnitude"] > 0.1, "form_trend_direction"] = 1
    x.loc[x["form_trend_magnitude"] < -0.1, "form_trend_direction"] = -1
    
    # Output subset
    result = x[[
        "player_id", "season", "gameweek_id",
        "recent_points_mean", "recent_goals_mean", "recent_assists_mean", "recent_minutes_mean",
        "season_points_mean", "form_trend_direction", "form_trend_magnitude"
    ]]
    
    return result
