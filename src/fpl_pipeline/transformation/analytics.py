from __future__ import annotations
import pandas as pd

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
        for i,r in f.reset_index(drop=True).iterrows():
            future=f.iloc[i:i+5].difficulty.astype(float)
            rows.append({"team_id":team,"gameweek_id":r.gameweek_id,"next_1_fixture_difficulty":future.iloc[0],"next_3_fixture_average_difficulty":future.iloc[:3].mean(),"next_5_fixture_average_difficulty":future.mean()})
    return pd.DataFrame(rows)
