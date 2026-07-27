"""Legal FPL squad optimisation using a transparent PuLP integer model."""
from __future__ import annotations
import pandas as pd
import pulp

POSITION_COUNTS = {1:2,2:5,3:5,4:3}

def optimise_squad(recommendations: pd.DataFrame, objective: str = "balanced", budget_tenths: int = 1000) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    x = recommendations.dropna(subset=["projected_points","current_price_tenths","position_id","team_id"]).copy().reset_index(drop=True)
    if set(POSITION_COUNTS) - set(x.position_id.astype(int)): raise ValueError("recommendations must include every FPL position")
    n=len(x); m=pulp.LpProblem("fpl_squad",pulp.LpMaximize)
    squad=pulp.LpVariable.dicts("squad",range(n),cat="Binary"); start=pulp.LpVariable.dicts("start",range(n),cat="Binary")
    captain=pulp.LpVariable.dicts("captain",range(n),cat="Binary"); vice=pulp.LpVariable.dicts("vice",range(n),cat="Binary")
    for i in range(n):
        m += start[i] <= squad[i]; m += captain[i] <= start[i]; m += vice[i] <= start[i]; m += captain[i] + vice[i] <= 1
    for pos,count in POSITION_COUNTS.items(): m += pulp.lpSum(squad[i] for i in range(n) if int(x.position_id[i])==pos)==count
    m += pulp.lpSum(squad[i] for i in range(n)) == 15; m += pulp.lpSum(start[i] for i in range(n)) == 11
    m += pulp.lpSum(start[i] for i in range(n) if int(x.position_id[i])==1) == 1
    for pos,minimum in {2:3,3:2,4:1}.items(): m += pulp.lpSum(start[i] for i in range(n) if int(x.position_id[i])==pos) >= minimum
    for team in x.team_id.unique(): m += pulp.lpSum(squad[i] for i in range(n) if x.team_id[i]==team) <= 3
    m += pulp.lpSum(squad[i]*float(x.current_price_tenths[i]) for i in range(n)) <= budget_tenths
    m += pulp.lpSum(captain[i] for i in range(n))==1; m += pulp.lpSum(vice[i] for i in range(n))==1
    score=x.projected_points.astype(float); value=x.posterior_ppm.fillna(0).astype(float); risk=x.projection_uncertainty.fillna(0).astype(float)
    if objective == "points": coeff=score
    elif objective == "value": coeff=value
    elif objective == "risk_adjusted": coeff=score-risk
    elif objective == "differential": coeff=score+.000001*(100-x.ownership.fillna(100))
    elif objective == "balanced": coeff=score+.04*value-.5*risk
    else: raise ValueError("objective must be points, value, balanced, risk_adjusted, or differential")
    m += pulp.lpSum(start[i]*float(coeff[i]) + captain[i]*float(score[i]) + squad[i]*.02*float(value[i]) for i in range(n))
    if pulp.PULP_CBC_CMD(msg=False).solve(m) != pulp.LpStatusOptimal: raise RuntimeError("no legal squad found; broaden player candidates")
    result=x[["player_id","player_name","team_name","position_id","current_price_tenths","projected_points","posterior_ppm","projection_uncertainty","recommendation_score"]].copy()
    result["is_selected"]=[bool(pulp.value(squad[i])) for i in range(n)]; result["is_starter"]=[bool(pulp.value(start[i])) for i in range(n)]
    result["is_captain"]=[bool(pulp.value(captain[i])) for i in range(n)]; result["is_vice_captain"]=[bool(pulp.value(vice[i])) for i in range(n)]
    result=result[result.is_selected].copy(); bench=result[~result.is_starter].sort_values("projected_points").copy(); bench["bench_order"]=range(1,len(bench)+1); result=result.merge(bench[["player_id","bench_order"]],on="player_id",how="left")
    return result.sort_values(["is_starter","position_id","projected_points"],ascending=[False,True,False]), result[result.is_starter].copy(), bench
