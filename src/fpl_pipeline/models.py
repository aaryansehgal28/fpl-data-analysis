"""Leakage-aware Bayesian value scores and short-horizon projections."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

def _season_start(season: str) -> int: return int(str(season)[:4])

def bayesian_player_value(player_season: pd.DataFrame, current_season: str, decay_rate: float, prior_equivalent_90s: float) -> pd.DataFrame:
    """Normal-mean empirical Bayes shrinkage of season PPM, weighted by recency and minutes.

    A raw PPM average is not used: observations have precision proportional to recency-weighted
    90s and are pulled toward a position prior. Extreme PPM is winsorised within position first.
    """
    x = player_season.dropna(subset=["position","season_ppm","effective_90s","identity_key"]).copy()
    x = x[(x.effective_90s > 0) & (x.season_ppm >= 0)]
    x["years_ago"] = (_season_start(current_season) - x.season.map(_season_start)).clip(lower=0)
    x["season_weight"] = np.exp(-decay_rate * x.years_ago)
    x["weighted_90s"] = x.effective_90s * x.season_weight
    # Avoid a handful of anomalous low-minute records defining a position prior.
    x["capped_ppm"] = x.groupby("position").season_ppm.transform(lambda s: s.clip(s.quantile(.02), s.quantile(.98)))
    priors = x.groupby("position").apply(lambda g: np.average(g.capped_ppm, weights=g.weighted_90s), include_groups=False).rename("position_prior_ppm").reset_index()
    spread = x.groupby("position").capped_ppm.std().rename("position_ppm_std").reset_index()
    x = x.merge(priors,on="position").merge(spread,on="position")
    x["weighted_ppm"] = x.capped_ppm * x.weighted_90s
    result = x.groupby(["identity_key","stable_player_id","position"],dropna=False,as_index=False).agg(
        weighted_historical_ppm=("weighted_ppm","sum"), effective_weighted_90s=("weighted_90s","sum"),
        position_prior_ppm=("position_prior_ppm","first"), position_ppm_std=("position_ppm_std","first"),
        seasons_observed=("season","nunique"), most_recent_season=("season","max"))
    result["prior_strength"] = prior_equivalent_90s
    result["posterior_ppm"] = (result.weighted_historical_ppm + prior_equivalent_90s * result.position_prior_ppm) / (result.effective_weighted_90s + prior_equivalent_90s)
    result["posterior_uncertainty"] = result.position_ppm_std.fillna(0).clip(lower=.1) / np.sqrt(result.effective_weighted_90s + prior_equivalent_90s)
    result["recency_weighted_sample_size"] = result.effective_weighted_90s
    result["reliability_score"] = (result.effective_weighted_90s / (result.effective_weighted_90s + prior_equivalent_90s)).clip(0,1)
    return result

def build_recommendations(players: pd.DataFrame, teams: pd.DataFrame, bayes: pd.DataFrame, fixture_runs: pd.DataFrame, current_gw: pd.DataFrame | None = None, horizon: int = 5, as_of_gameweek: int | None = None, target_gameweek: int | None = None) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Current state + historical posterior. Only supplied current/pre-deadline snapshot fields are used."""
    x = players.merge(teams[["team_id","team_name"]],on="team_id",how="left").merge(bayes, left_on="player_code", right_on="stable_player_id", how="left")
    if target_gameweek is None: target_gameweek = int(fixture_runs.gameweek_id.min())
    run = fixture_runs[fixture_runs.gameweek_id == target_gameweek].groupby("team_id",as_index=False).first()
    x = x.merge(run[["team_id",f"next_{horizon}_fixture_average_difficulty"]],on="team_id",how="left")
    x["fixture_difficulty"] = x[f"next_{horizon}_fixture_average_difficulty"].fillna(3.0)
    x["fixture_adjustment"] = (3 / x.fixture_difficulty).clip(.7,1.3)
    chance = pd.to_numeric(x.chance_of_playing_next_round,errors="coerce")
    x["availability_factor"] = chance.div(100).where(chance.notna(), np.where(x.status.eq("a"),1.0,.55))
    x["ownership"] = pd.to_numeric(x.ownership_percent,errors="coerce")
    x["transfer_momentum"] = pd.to_numeric(x.transfers_in_event,errors="coerce").fillna(0) - pd.to_numeric(x.transfers_out_event,errors="coerce").fillna(0)
    x["recent_form"] = 0.0; x["recent_minutes"] = 0.0
    if current_gw is not None and not current_gw.empty:
        # A historical decision before GW t may only see completed events before t.
        if as_of_gameweek is not None and "gameweek_id" in current_gw:
            current_gw = current_gw[current_gw.gameweek_id < as_of_gameweek]
        recent = current_gw.groupby("player_id",as_index=False).agg(recent_form=("total_points","mean"),recent_minutes=("minutes","mean"))
        x=x.drop(columns=["recent_form","recent_minutes"]).merge(recent,on="player_id",how="left").fillna({"recent_form":0,"recent_minutes":0})
    # PPM is a season-scale measure. Divide by 38 before horizon expansion; current price converts value to points.
    x["baseline_expected_points"] = x.posterior_ppm.fillna(0) * (x.current_price / 10) / 38
    x["minutes_factor"] = np.where(x.reliability_score.fillna(0) >= .35, 1.0, .82)
    x["form_factor"] = (1 + (x.recent_form.fillna(0) - 4).clip(-2,2) * .03)
    x["projected_points"] = x.baseline_expected_points * horizon * x.minutes_factor * x.availability_factor * x.fixture_adjustment * x.form_factor
    x["projection_uncertainty"] = x.posterior_uncertainty.fillna(x.posterior_ppm.fillna(0)*.5) * (x.current_price/10) * horizon / 38
    x["recommendation_score"] = x.projected_points + .04*x.posterior_ppm.fillna(0) - .5*x.projection_uncertainty
    recommendation = x.rename(columns={"web_name":"player_name","current_price":"current_price_tenths"})[["player_id","player_code","player_name","team_id","team_name","position_id","current_price_tenths","status","ownership","transfer_momentum","posterior_ppm","reliability_score","posterior_uncertainty","recent_form","recent_minutes","fixture_difficulty","fixture_adjustment","availability_factor","baseline_expected_points","projected_points","projection_uncertainty","recommendation_score"]]
    projections = pd.concat([recommendation.assign(horizon=h, projected_points=lambda d: d.baseline_expected_points*h*d.fixture_adjustment*d.availability_factor,
        projected_points_per_million=lambda d: d.projected_points/(d.current_price_tenths/10), confidence_lower=lambda d:(d.projected_points-1.96*d.projection_uncertainty*h/horizon).clip(lower=0), confidence_upper=lambda d:d.projected_points+1.96*d.projection_uncertainty*h/horizon)
        for h in (1,3,5)], ignore_index=True)
    return recommendation, projections
