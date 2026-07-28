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

# ==================== COMPONENT-BASED SYSTEM (NEW) ====================

def build_expected_value_model(
    player_season_extended: pd.DataFrame,
    fixture_context: pd.DataFrame,
    team_stats: pd.DataFrame,
    form_trends: pd.DataFrame,
    players: pd.DataFrame,
    current_price_col: str = "current_price",
    prediction_horizon: int = 5,
    as_of_gameweek: int | None = None,
) -> pd.DataFrame:
    """Compute component-wise expected points for next GW(s).
    
    Args:
        player_season_extended: player-season data with posterior component estimates
        fixture_context: fixture difficulty and context
        team_stats: team-level attacking/defensive strength
        form_trends: recent form and trends
        players: current player status (availability, price, etc.)
        current_price_col: column name for current price
        prediction_horizon: 1, 3, or 5 gameweeks
        as_of_gameweek: for reproducibility in backtests
    
    Returns:
        DataFrame with expected points per component and total
    """
    # Start with current player snapshot
    x = players[["player_id", "team_id", "position_id", "web_name"]].copy()
    x = x.merge(player_season_extended, left_on="player_id", right_on="player_id", how="left")
    x = x.merge(form_trends, left_on="player_id", right_on="player_id", how="left")
    
    # Position mapping
    POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    x["position"] = x["position_id"].map(POSITION_MAP).fillna(x.get("position", "UNK"))
    
    # Get fixture context
    fixture_context_today = fixture_context.groupby("team_id", as_index=False).first()
    x = x.merge(fixture_context_today[["team_id", "next_1_fixture_difficulty", "next_5_fixture_average_difficulty", "next_fixture_is_home"]], 
                on="team_id", how="left")
    
    # FDR adjustment (3.0/FDR scaling)
    x["fdr"] = x["next_5_fixture_average_difficulty"].fillna(3.0)
    x["fixture_adjustment"] = (3.0 / x["fdr"]).clip(0.7, 1.3)
    
    # Availability
    x["availability_factor"] = x.get("chance_of_playing_next_round", 100.0)
    x["availability_factor"] = pd.to_numeric(x["availability_factor"], errors="coerce").fillna(100) / 100
    x["availability_factor"] = x["availability_factor"].clip(0, 1)
    
    # Form adjustment
    x["form_adjustment"] = 1.0 + (x.get("form_trend_magnitude", 0).fillna(0) * 0.1).clip(-0.3, 0.3)
    
    # Minutes estimate (based on position and recent appearance frequency)
    x["typical_minutes"] = np.where(
        x["position"].isin(["GK", "DEF"]),
        85,  # Defenders/keepers play more
        np.where(
            x["position"].isin(["MID"]),
            70,  # Midfielders play mid-range
            60   # Forwards often rotate
        )
    )
    x["expected_minutes_next_match"] = x["typical_minutes"] * x["availability_factor"]
    
    # ==== COMPONENT 1: GOALS ====
    x["posterior_goals_per_90"] = x.get("posterior_goals_per_90", 0.0).fillna(0.0)
    x["expected_goals"] = (
        x["posterior_goals_per_90"] * 
        (x["expected_minutes_next_match"] / 90) * 
        x["fixture_adjustment"] * 
        x["form_adjustment"]
    ).clip(lower=0)
    
    # ==== COMPONENT 2: ASSISTS ====
    x["posterior_assists_per_90"] = x.get("posterior_assists_per_90", 0.0).fillna(0.0)
    x["expected_assists"] = (
        x["posterior_assists_per_90"] * 
        (x["expected_minutes_next_match"] / 90) * 
        x["fixture_adjustment"] * 
        x["form_adjustment"]
    ).clip(lower=0)
    
    # ==== COMPONENT 3: CLEAN SHEETS ====
    # Get team CS probability from team_stats
    team_stats_merge = team_stats[["team", "team_cs_frequency"]].rename(columns={"team": "team_id"})
    x = x.merge(team_stats_merge, on="team_id", how="left")
    x["team_cs_prob"] = x.get("team_cs_frequency", 0.3).fillna(0.3)
    x["prob_on_pitch"] = x["availability_factor"] * (x["expected_minutes_next_match"] / 90)
    x["expected_cs_prob"] = (x["team_cs_prob"] * x["prob_on_pitch"]).clip(0, 1)
    
    # ==== COMPONENT 4: BONUS ====
    x["posterior_bps_per_90"] = x.get("posterior_bps_per_90", 10.0).fillna(10.0)
    # Logistic conversion to probability of top-3 BPS (simplified)
    x["bps_next_match"] = x["posterior_bps_per_90"] * (x["expected_minutes_next_match"] / 90)
    # Rough logistic: P(top3) ≈ sigmoid((bps - threshold) / scale)
    def logistic_bonus_prob(bps, position):
        thresholds = {"GK": 40, "DEF": 45, "MID": 50, "FWD": 55}
        threshold = thresholds.get(position, 50)
        return 1 / (1 + np.exp(-(bps - threshold) / 15))
    
    x["prob_top3_bps"] = x.apply(
        lambda r: logistic_bonus_prob(r["bps_next_match"], r["position"]), axis=1
    ).clip(0, 1)
    x["expected_bonus"] = x["prob_top3_bps"] * 1.5  # Expected value given top-3
    
    # ==== COMPONENT 5: APPEARANCE ====
    x["expected_appearance"] = 1.0 * x["availability_factor"]  # Bonus for playing
    
    # ==== COMPONENT 6: DISCIPLINE ====
    x["posterior_yellows_per_90"] = x.get("posterior_yellows_per_90", 0.15).fillna(0.15)
    x["posterior_reds_per_90"] = x.get("posterior_reds_per_90", 0.02).fillna(0.02)
    x["expected_yellows"] = x["posterior_yellows_per_90"] * (x["expected_minutes_next_match"] / 90)
    x["expected_reds"] = x["posterior_reds_per_90"] * (x["expected_minutes_next_match"] / 90)
    x["expected_discipline"] = -(x["expected_yellows"] * 0.5 + x["expected_reds"] * 2)
    
    # ==== POSITION-SPECIFIC POINT MULTIPLIERS ====
    def compute_expected_points(row):
        """Combine components via position-specific multipliers."""
        pos = row["position"]
        components = {
            "goals": row["expected_goals"],
            "assists": row["expected_assists"],
            "cs": row["expected_cs_prob"],
            "bonus": row["expected_bonus"],
            "appearance": row["expected_appearance"],
            "discipline": row["expected_discipline"],
        }
        
        # Position-specific point multipliers (from FPL rules)
        multipliers = {
            "GK": {"goals": 0, "assists": 0, "cs": 4, "bonus": 1, "appearance": 1, "discipline": 1},
            "DEF": {"goals": 5, "assists": 1, "cs": 4, "bonus": 1, "appearance": 1, "discipline": 1},
            "MID": {"goals": 5, "assists": 1, "cs": 1, "bonus": 1, "appearance": 1, "discipline": 1},
            "FWD": {"goals": 4, "assists": 1, "cs": 0, "bonus": 1, "appearance": 1, "discipline": 1},
        }
        
        mults = multipliers.get(pos, multipliers["MID"])
        expected_points = sum(components[k] * mults.get(k, 0) for k in components)
        return expected_points
    
    x["expected_points_per_match"] = x.apply(compute_expected_points, axis=1)
    
    # ==== UNCERTAINTY ====
    # Combine component-wise uncertainties
    component_stds = {
        "expected_goals": x.get("posterior_goals_per_90_std", 0.05).fillna(0.05),
        "expected_assists": x.get("posterior_assists_per_90_std", 0.05).fillna(0.05),
        "expected_cs": 0.3,  # Binomial uncertainty for CS
        "expected_bonus": 0.5,  # High uncertainty for bonus
        "expected_discipline": 0.2,
    }
    
    x["uncertainty_per_match"] = np.sqrt(
        (component_stds["expected_goals"] ** 2) +
        (component_stds["expected_assists"] ** 2) +
        (component_stds["expected_cs"] ** 2) +
        (component_stds["expected_bonus"] ** 2) +
        (component_stds["expected_discipline"] ** 2)
    )
    
    # ==== MULTI-HORIZON ====
    x[f"expected_points_{prediction_horizon}gw"] = (
        x["expected_points_per_match"] * prediction_horizon * x["availability_factor"]
    )
    x[f"uncertainty_{prediction_horizon}gw"] = (
        x["uncertainty_per_match"] * np.sqrt(prediction_horizon)
    )
    x["confidence_interval_95_lower"] = (
        (x[f"expected_points_{prediction_horizon}gw"] - 1.96 * x[f"uncertainty_{prediction_horizon}gw"]).clip(lower=0)
    )
    x["confidence_interval_95_upper"] = (
        x[f"expected_points_{prediction_horizon}gw"] + 1.96 * x[f"uncertainty_{prediction_horizon}gw"]
    )
    
    # ==== VALUE ====
    x["current_price_tenths"] = players.get(current_price_col, 50).fillna(50)
    x["points_per_million"] = (
        x[f"expected_points_{prediction_horizon}gw"] / (x["current_price_tenths"] / 10)
    ).replace([np.inf, -np.inf], np.nan)
    
    # Output
    output_cols = [
        "player_id", "web_name", "position", "team_id",
        "expected_goals", "expected_assists", "expected_cs_prob", "expected_bonus", "expected_appearance",
        "expected_points_per_match", "expected_discipline",
        f"expected_points_{prediction_horizon}gw", f"uncertainty_{prediction_horizon}gw",
        "confidence_interval_95_lower", "confidence_interval_95_upper",
        "points_per_million", "fixture_adjustment", "availability_factor", "form_adjustment",
    ]
    
    return x[[c for c in output_cols if c in x.columns]]

def compute_recommendation_score_v2(
    player_component_expectations: pd.DataFrame,
    risk_adjustment_factor: float = 0.5,
    value_weight: float = 0.04,
    prediction_horizon: int = 5,
) -> pd.DataFrame:
    """Generate final recommendation score combining expected value and risk.
    
    Args:
        player_component_expectations: from build_expected_value_model()
        risk_adjustment_factor: how much to penalize uncertainty
        value_weight: weight on points-per-million bonus
        prediction_horizon: horizon used in component model
    
    Returns:
        DataFrame with recommendation scores and metadata
    """
    x = player_component_expectations.copy()
    
    # Extract horizon column name
    exp_col = f"expected_points_{prediction_horizon}gw"
    unc_col = f"uncertainty_{prediction_horizon}gw"
    
    if exp_col not in x.columns:
        raise ValueError(f"Missing column: {exp_col}")
    
    # Recommendation score formula
    # score = expected_points + value_bonus - risk_penalty
    x["recommendation_score_v2"] = (
        x[exp_col] + 
        (x.get("points_per_million", 0).fillna(0) / 10) * value_weight - 
        risk_adjustment_factor * x[unc_col].fillna(0)
    )
    
    # Ranking tier
    def assign_tier(score):
        if pd.isna(score):
            return "UNRANKED"
        elif score >= x["recommendation_score_v2"].quantile(0.10):
            return "TOP_20"
        elif score >= x["recommendation_score_v2"].quantile(0.30):
            return "21_50"
        elif score >= x["recommendation_score_v2"].quantile(0.60):
            return "51_100"
        else:
            return "OUTSIDE_100"
    
    x["ranking_tier"] = x["recommendation_score_v2"].apply(assign_tier)
    
    output_cols = [
        "player_id", "web_name", "position", "team_id",
        "recommendation_score_v2", "ranking_tier",
        f"expected_points_{prediction_horizon}gw", f"uncertainty_{prediction_horizon}gw",
        "confidence_interval_95_lower", "confidence_interval_95_upper",
        "points_per_million", "fixture_adjustment", "availability_factor", "form_adjustment",
        "expected_goals", "expected_assists", "expected_cs_prob", "expected_bonus",
    ]
    
    return x[[c for c in output_cols if c in x.columns]]

def generate_recommendation_explanations(
    recommendations: pd.DataFrame,
    prediction_horizon: int = 5,
) -> pd.DataFrame:
    """Generate human-readable explanations for recommendations.
    
    Args:
        recommendations: from compute_recommendation_score_v2()
        prediction_horizon: horizon used
    
    Returns:
        DataFrame with explanations per player
    """
    exp_col = f"expected_points_{prediction_horizon}gw"
    
    x = recommendations.copy()
    
    explanations = []
    for _, row in x.iterrows():
        player_id = row["player_id"]
        player_name = row.get("web_name", f"Player {player_id}")
        position = row.get("position", "UNK")
        rank_tier = row.get("ranking_tier", "UNKNOWN")
        score = row.get("recommendation_score_v2", 0)
        
        # Component breakdown
        components = {
            "Goals": row.get("expected_goals", 0),
            "Assists": row.get("expected_assists", 0),
            "CS": row.get("expected_cs_prob", 0),
            "Bonus": row.get("expected_bonus", 0),
        }
        
        # Identify top drivers
        top_drivers = sorted(components.items(), key=lambda x: x[1], reverse=True)[:2]
        driver_text = ", ".join(f"{name} ({val:.1f})" for name, val in top_drivers)
        
        # Risks
        risks = []
        if row.get("availability_factor", 1.0) < 0.8:
            risks.append("low availability")
        if row.get("fixture_adjustment", 1.0) < 0.85:
            risks.append("difficult fixture")
        if row.get("form_adjustment", 1.0) < 0.95:
            risks.append("recent form dip")
        
        risk_text = "; ".join(risks) if risks else "no major risks"
        
        # Explanation text
        explanation = f"{player_name} ({position}) - Rank: {rank_tier} (score {score:.2f}). "
        explanation += f"Driven by: {driver_text}. Risks: {risk_text}."
        
        explanations.append({
            "player_id": player_id,
            "player_name": player_name,
            "position": position,
            "recommendation_score": score,
            "explanation": explanation,
            "top_drivers": driver_text,
            "risks": risk_text,
        })
    
    return pd.DataFrame(explanations)
