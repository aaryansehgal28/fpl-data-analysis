# Phase 3: Statistical Framework Design

**Date:** 2026-07-27  
**Scope:** Specify computation details, priors, shrinkage, aggregation windows, and handling of edge cases  
**Output:** Practical implementation guide ready for code generation  

---

## OVERVIEW

This phase defines the **exact statistical methodology** for computing the 18 MVP features. It specifies:

- **Per-90 calculations** with edge case handling
- **Aggregation windows** (season definition, recent window sizing)
- **Position-specific Bayesian priors** (baseline values, distribution assumptions)
- **Shrinkage intensity** (how much to pull toward priors)
- **Missing value imputation** (handling zeros, NAs, new players)
- **Low-minute player adjustments** (uncertainty, reliability penalties)

---

## SECTION 1: AGGREGATION WINDOWS & DEFINITIONS

### 1.1 – Season Definition

**Problem:** What counts as a "season"?
- Official FPL seasons run Aug–May (e.g., "2025-26")
- Some players are mid-season transfers
- Some players have injury breaks

**Standard definition:**
```
Full season = Gameweeks 1–38 (latest completed full season)
Current season = Gameweeks 1–t (up to current gameweek t)
```

**For historical backtesting:**
```
As-of date t:
  - Use data from GWs 1 to t-1 (strictly before)
  - Never look ahead to GW t or beyond
```

**Minimum data requirement:**
```
At least 450 minutes (5 × 90) in a season before computing season-level stats.
Otherwise, flag as "unreliable" and apply heavy shrinkage.
```

---

### 1.2 – Recent Form Window

**Standard recent window: Last 5 gameweeks**

```
Recent = GWs (t-4) to t, where t is current/as-of GW
Example: if as-of GW=20, recent = GWs 16-20
```

**Rationale:**
- 5 GWs ≈ 1 month of soccer (typical injury/form cycle)
- Provides 4–5 data points for trend calculation
- Avoids overfitting to single recent outlier

**Fallback if <5 GWs available:**
```
Use all available completed GWs (e.g., first 3 GWs of season)
Flag as "sparse" and apply uncertainty penalty
```

**Special cases:**
```
- Blank gameweeks: Skip; do not count as absence
- Double gameweeks: Treat as 2 separate appearances if data split; 
  otherwise treat as single aggregated appearance
- New signings: Start recent window from first appearance
```

---

### 1.3 – Lookback for Recency Weighting (Bayesian Priors)

**When computing Bayesian posterior PPM (from current model):**
```
Recency decay_rate = 0.45 per year
season_weight = exp(-0.45 × years_ago)
```

**For new feature architecture:**
```
Apply same recency decay when aggregating historical seasons:
  - Current season: weight = 1.0
  - 1 year ago: weight = 0.64
  - 2 years ago: weight = 0.41
  - 3 years ago: weight = 0.26
  - 4+ years ago: weight = 0.10 (minimal impact)
```

**Aggregate using weighted mean:**
```
weighted_metric = SUM(metric_season × weight_season) / SUM(weight_season)
```

**Minimum data requirement:**
```
At least 2 prior seasons with data before using Bayesian prior
If only 1 season available, use cross-sectional position mean
If no history available, use position median from full population
```

---

## SECTION 2: PER-90 CALCULATIONS & NORMALIZATION

### 2.1 – Standard Per-90 Formula

For any metric M (e.g., goals, assists, minutes played):

```
M_per_90 = SUM(M across window) / (SUM(minutes across window) / 90)
```

**Edge case: Zero minutes in window**
```
If SUM(minutes) = 0:
  Set M_per_90 = 0 (player played 0 minutes; no production)
  Set reliability_flag = "PLAYED_ZERO_MINUTES"
  Apply maximum uncertainty penalty in downstream scoring
```

**Edge case: Single appearance with few minutes**
```
If appearances = 1 and minutes < 45:
  Mark as "cameo appearance"
  Apply higher uncertainty (smaller effective sample size)
```

---

### 2.2 – Position-Specific Baseline Values (League Medians)

Compute at start of season from all players in position with ≥450 minutes:

#### Goals Per 90 (Baseline across 5 historical seasons)

| Position | Median | 25th %ile | 75th %ile | Notes |
|----------|--------|-----------|-----------|-------|
| FWD | 0.45 | 0.25 | 0.68 | 4–5 goals per season |
| MID | 0.10 | 0.04 | 0.18 | ~3.8 goals / 38 GWs typical |
| DEF | 0.02 | 0.00 | 0.06 | ~0.6 goals / 38 GWs typical |
| GK | 0.00 | 0.00 | 0.00 | Impossible |

**Calculation:**
```
For each historical season s:
  goals_per_90_s = SUM(goals) / (SUM(minutes) / 90) for all players with ≥450 min in position
  position_median_goal_rate_s = MEDIAN(goals_per_90_s)
  
Aggregate across 5 seasons with recency weighting:
  baseline_goals_per_90 = weighted_avg(position_median_goal_rate_s)
```

**Use in Bayesian prior:**
```
When shrinking an individual player's goal_per_90:
  prior_mean = baseline_goals_per_90
  prior_std_dev = (75th_percentile - 25th_percentile) / 1.35  [≈ IQR/1.35 ≈ std]
  prior_equivalent_90s = 50  [equivalent experience]
```

#### Assists Per 90

| Position | Median | 25th %ile | 75th %ile | Notes |
|----------|--------|-----------|-----------|-------|
| MID | 0.08 | 0.03 | 0.15 | 3–5 assists per season |
| DEF | 0.03 | 0.01 | 0.07 | 1–2 assists per season (set-pieces) |
| FWD | 0.04 | 0.01 | 0.08 | 1–3 assists per season |
| GK | 0.00 | 0.00 | 0.00 | Impossible |

#### Clean Sheets Per 90 (Difficult: team-dependent)

| Position | Median | Notes |
|----------|--------|-------|
| GK | 0.35 | ~13 CS per season (38 GWs × ~0.35) |
| DEF | 0.25 | ~9.5 CS per season |
| MID | 0.08 | ~3 CS per season (midfield often doesn't get CS) |
| FWD | 0.00 | Never get clean sheet points |

**Note:** Clean sheets are heavily team-dependent. Consider using team CS rate as covariate:

```
Expected player_cs_per_90 ≈ team_cs_rate × player_exposure_factor
Where exposure_factor = 1.0 if defender, 0.3 if midfielder
```

#### BPS Per 90

| Position | Median | Notes |
|----------|--------|-------|
| GK | 35–45 | High BPS from clean sheets + saves |
| DEF | 25–35 | Good BPS from CS + blocks/tackles |
| MID | 20–30 | Moderate BPS from goals/assists/blocks |
| FWD | 20–28 | Goals/assists dominant; lower block credit |

**Normalize by position:**
```
Position median = MEDIAN(mean_bps per player per GW) across position
```

#### Influence Per 90

| Position | Median | Notes |
|----------|--------|-------|
| GK | 40–60 | High: distribution, sweeping |
| DEF | 50–70 | High: tackles, blocks, positioning |
| MID | 30–50 | Moderate: passes, interceptions |
| FWD | 20–40 | Lower: less possession, more isolated |

**Interpretation:** Influence is 0–100 scale per GW; per-90 means summing across GWs then dividing.

---

### 2.3 – Computing Position-Normalized Metrics

For each player, create a **normalized version** of per-90 stats:

```
goals_per_90_normalized = player_goals_per_90 / position_median_goals_per_90
                        [ratio; 1.0 = median, 0.5 = half of median, 2.0 = double median]

assists_per_90_normalized = player_assists_per_90 / position_median_assists_per_90

cs_per_90_normalized = player_cs_per_90 / position_median_cs_per_90

bps_per_90_normalized = player_bps_per_90 / position_median_bps_per_90

influence_per_90_normalized = player_influence_per_90 / position_median_influence_per_90
```

**Use in scoring:**
- Normalized versions for inter-position comparison (e.g., "DEF with 1.8× median influence is elite")
- Raw versions for within-position ranking and Bayesian modeling

---

## SECTION 3: BAYESIAN SHRINKAGE FRAMEWORK

### 3.1 – Conceptual Model

For each player P and metric M (e.g., goals_per_90):

```
Observed = player's historical average (noisy signal)
Prior = position median (base expectation)
Posterior = weighted average of observed and prior
```

**Weight allocation depends on:**
- Amount of data (more data → higher weight on observed)
- Stability of metric (less variable → higher weight on observed)
- Recency (recent data → higher weight)

---

### 3.2 – Empirical Bayes Shrinkage (Normal-Mean Model)

**For each metric, independently apply shrinkage:**

```
Posterior = (weight_observed × observed + weight_prior × prior) 
          / (weight_observed + weight_prior)

where:
  observed = player's historical average for metric in season window
  prior = position_baseline for metric
  weight_observed = effective_90s (for metric season-window)
  weight_prior = prior_equivalent_90s (configured per position/metric)
```

**Example: Forward's goals per 90**

```
Observed: Player has 40 goals in 3000 minutes (33.33 90s)
          goals_per_90_observed = 40 / 33.33 = 1.20 goals/90

Prior: FWD baseline = 0.45 goals/90
       prior_equivalent_90s = 50

Posterior = (33.33 × 1.20 + 50 × 0.45) / (33.33 + 50)
          = (40 + 22.5) / 83.33
          = 62.5 / 83.33
          = 0.75 goals/90  [pulled down from 1.20 toward 0.45]

Weight on observed = 33.33 / 83.33 = 40%
Weight on prior = 50 / 83.33 = 60%
```

---

### 3.3 – Prior Equivalent 90s (Shrinkage Intensity)

The `prior_equivalent_90s` parameter controls how much to pull toward the prior. Higher value = more shrinkage.

**Recommended values by position and metric:**

| Position | Metric | Prior_90s | Rationale |
|----------|--------|-----------|-----------|
| **FWD** | goals_per_90 | 40 | Goals are volatile; moderate shrinkage |
| **FWD** | assists_per_90 | 50 | Assists more volatile; more shrinkage |
| **FWD** | clean_sheets | — | N/A (rarely relevant) |
| **FWD** | bps_per_90 | 60 | BPS is composite; shrink more |
| **MID** | goals_per_90 | 80 | MID goals rare; strong shrinkage toward prior |
| **MID** | assists_per_90 | 60 | Assists are primary for MID; moderate shrinkage |
| **MID** | cs_per_90 | 100 | CS rare for MID; strong shrinkage |
| **MID** | bps_per_90 | 70 | BPS for MID is noisy |
| **DEF** | goals_per_90 | 150 | DEF goals are extremely rare; very strong shrinkage |
| **DEF** | assists_per_90 | 100 | Set-piece specialists noisy; strong shrinkage |
| **DEF** | cs_per_90 | 30 | CS is core; weak shrinkage (trust defender's history) |
| **DEF** | bps_per_90 | 50 | BPS reasonably stable for defenders |
| **GK** | clean_sheets_per_90 | 25 | CS is core for GK; trust history more |
| **GK** | bps_per_90 | 40 | BPS stable for GK |
| **GK** | influence_per_90 | 50 | GK influence less stable |
| **ALL** | influence_per_90 | 80 | Influence is composite; moderate-strong shrinkage |
| **ALL** | effective_90s | 30 | Playing time relatively stable; weak shrinkage |
| **ALL** | starts_ratio | 60 | Deployment pattern moderately stable |

**Rationale:**
- **Rare events (DEF goals, MID clean sheets)** → high shrinkage (stronger pull to prior)
- **Common events (FWD goals, GK clean sheets)** → low shrinkage (trust observed signal)
- **Composite metrics (BPS, influence)** → moderate-high shrinkage (more noise)
- **Binary/categorical (starts_ratio)** → moderate shrinkage (bounded, slower change)

---

### 3.4 – Uncertainty Quantification

For each player-metric combination, estimate posterior standard error:

```
posterior_uncertainty = position_metric_std / sqrt(prior_equivalent_90s + effective_90s)

where:
  position_metric_std = standard deviation of metric within position (from training data)
  prior_equivalent_90s = prior strength [from table above]
  effective_90s = player's actual playing time
```

**Example:**

```
FWD goals_per_90:
  position_std = 0.35 (FWDs have 0.35 SD in goal rates)
  prior_90s = 40
  player_effective_90s = 20 (only 1800 minutes into season)

posterior_uncertainty = 0.35 / sqrt(40 + 20)
                      = 0.35 / sqrt(60)
                      = 0.35 / 7.75
                      = 0.045 goals/90

→ Posterior estimate ± 0.045 (95% CI ≈ posterior ± 0.09)
```

**Use in scoring:**
```
recommendation_score uses projection_uncertainty;
Higher uncertainty → lower score (penalizes risk)
```

---

### 3.5 – Reliability Score (Data Quality Indicator)

For each metric, compute reliability as the weight on observed data:

```
reliability_score_metric = effective_90s / (prior_equivalent_90s + effective_90s)
                         [range: 0 to 1]

Example:
  FWD with 40 effective 90s (3600 minutes), metric="goals"
  prior_90s = 40
  
  reliability = 40 / (40 + 40) = 0.50  [50% weight on observed]
```

**Thresholds:**
```
reliability < 0.3  → "Low reliability" (rookie, recent signing, injured)
              0.3–0.7  → "Moderate reliability" (some history)
              >0.7  → "High reliability" (established player)
```

**Use in scoring:**
```
For playing time specifically:
  If reliability_score < 0.35 for effective_90s:
    Apply 0.82× factor to baseline (penalize unproven players for minutes risk)
```

---

## SECTION 4: POSITION-SPECIFIC MODELING

### 4.1 – Position Categories & Scoring Adjustments

**4 positions with distinct point structures:**

| Position | Points Structure | Model Adjustments |
|----------|------------------|-------------------|
| **GK** | Goal (4), CS (4), Saves (0.33), Bonus | High weight on CS; influence secondary |
| **DEF** | Goal (5), CS (4), Bonus (1–3), Tackles/Blocks (indirect) | Balance CS + influence; goals rare |
| **MID** | Goal (5), CS (1), Assist (1.5), Bonus (1–3) | Goals + assists important; CS secondary |
| **FWD** | Goal (4), Assist (1), Bonus (1–3), CS (0) | Goals dominant; assists secondary; CS never |

### 4.2 – Position-Specific Feature Weights

When computing per-90 metrics, use position-aware denominators where applicable:

**Clean Sheets Relevance:**
```
GK: fundamental (weight=1.0)
DEF: fundamental (weight=1.0)
MID: secondary (weight=0.5 or 0.3 in some models)
FWD: irrelevant (weight=0.0)
```

**Example:** If building "expected CS points per 90":

```
For GK: CS_points_per_90 = cs_per_90 × 4 × 1.0
For DEF: CS_points_per_90 = cs_per_90 × 4 × 1.0
For MID: CS_points_per_90 = cs_per_90 × 1 × 0.5
For FWD: CS_points_per_90 = 0
```

### 4.3 – Position-Specific Priors

For each position, maintain separate Bayesian prior sets:

```
priors_by_position = {
  "GK": {
    "goals_per_90": (mean=0, std=0.01, prior_90s=150),
    "assists_per_90": (mean=0, std=0.01, prior_90s=150),
    "cs_per_90": (mean=0.35, std=0.12, prior_90s=25),
    "saves_per_90": (mean=3.2, std=0.8, prior_90s=40),
    "bps_per_90": (mean=40, std=12, prior_90s=40),
    "influence_per_90": (mean=55, std=15, prior_90s=50),
  },
  "DEF": {
    "goals_per_90": (mean=0.02, std=0.05, prior_90s=150),
    "assists_per_90": (mean=0.03, std=0.06, prior_90s=100),
    "cs_per_90": (mean=0.25, std=0.12, prior_90s=30),
    "bps_per_90": (mean=30, std=10, prior_90s=50),
    "influence_per_90": (mean=62, std=18, prior_90s=60),
  },
  "MID": {
    "goals_per_90": (mean=0.10, std=0.12, prior_90s=80),
    "assists_per_90": (mean=0.08, std=0.09, prior_90s=60),
    "cs_per_90": (mean=0.08, std=0.08, prior_90s=100),
    "bps_per_90": (mean=25, std=8, prior_90s=70),
    "influence_per_90": (mean=45, std=14, prior_90s=80),
  },
  "FWD": {
    "goals_per_90": (mean=0.45, std=0.25, prior_90s=40),
    "assists_per_90": (mean=0.04, std=0.06, prior_90s=50),
    "cs_per_90": (mean=0, std=0, prior_90s=∞),
    "bps_per_90": (mean=24, std=8, prior_90s=60),
    "influence_per_90": (mean=32, std=10, prior_90s=80),
  }
}
```

**Note:** These priors should be **estimated from training data** (5 historical seasons) rather than hard-coded. The values above are approximate.

---

## SECTION 5: MISSING VALUE & EDGE CASE HANDLING

### 5.1 – Missing Performance Data

**Case 1: Player has 0 appearances in window**

```
Scenario: Player is injured/transferred all season
  effective_90s = 0
  all metrics (goals_per_90, etc.) = NaN

Treatment:
  Set all per-90 metrics = position_prior_mean
  Set reliability_score = 0.0
  Set uncertainty = infinity (maximum penalty)
  Flag = "NO_PLAYING_TIME"
  
Use in scoring:
  availability_factor = 0.1  [assume 10% chance of playing]
  projected_points = 0 or very low
```

**Case 2: Player has <90 minutes total in window**

```
Scenario: 1-2 substitute appearances
  effective_90s = 0.5–0.9

Treatment:
  Compute per-90 stats normally
  BUT: Mark as "SPARSE_DATA"
  Apply additional uncertainty penalty (+0.1 to uncertainty)
  Reliability score automatically low (due to low effective_90s)
```

---

### 5.2 – Missing Individual Statistics

**FPL API sometimes has incomplete data for historical records.**

**Case: Assists not recorded for historical seasons**

```
If assists = NaN for player in season s:
  Imputation strategy:
    1. Use creativity_per_90 as proxy (multiply by calibration factor)
       assists_imputed ≈ creativity_per_90 × 0.12 [estimated conversion]
    
    2. If creativity also missing, use ict_index subset
    
    3. If no attacking stats available, impute = position_median_assists_per_90
    
  Flag = "IMPUTED_ASSISTS"
  Apply +20% uncertainty penalty to assists_per_90 estimate
```

**Case: Bonus points not recorded**

```
If bonus = NaN for player-GW:
  Imputation:
    1. Use BPS rank from GW to infer bonus probability
       If BPS rank ≤ 3: bonus = 3 or 2 (with noise)
       If BPS rank 4–5: bonus = 1
       If BPS rank > 5: bonus = 0
    
    2. If BPS also missing, impute = 0 (conservative)
    
  Flag = "IMPUTED_BONUS"
  Apply +30% uncertainty penalty
```

---

### 5.3 – Zero vs. Missing Distinction

**Clarification: In FPL, 0 is a recorded value (not missing).**

```
goals_scored = 0  means player played but didn't score
goals_scored = NaN means data not available/recorded

Handling:
  - 0: Use normally in calculations
  - NaN: Apply imputation strategy above
```

---

### 5.4 – New Players (No Prior History)

**Scenario: Player transferred mid-season; no historical data.**

```
Example: Midfielder joins in GW15 with no prior EPL history

Treatment:
  Season-level stats:
    - Do not compute (insufficient data)
  
  Recent-only stats (from GW15 onward):
    - Compute per-90 metrics from GW15 onwards
    - Treat as "current season only" data
    - Use position-wide priors for Bayesian shrinkage
    - Apply MAXIMUM shrinkage (prior_equivalent_90s × 2)
    - Set reliability_score = 0.1 (very low)
    - Flag = "NEW_SIGNING"

  Use in scoring:
    - Posterior = 80% prior, 20% observed
    - High uncertainty penalty
    - Minutes factor = 0.82 (assume unproven)
    - availability_factor based on actual playing time
```

---

## SECTION 6: LOW-MINUTE PLAYER HANDLING

### 6.1 – Minimum Thresholds

Define data quality tiers:

| Minutes | Tier | Treatment | Reliability |
|---------|------|-----------|-------------|
| >1800 (20× 90) | ESTABLISHED | Standard Bayes | 0.50–1.0 |
| 900–1800 (10–20× 90) | EMERGING | Moderate shrinkage | 0.30–0.50 |
| 450–900 (5–10× 90) | SPARSE | Strong shrinkage | 0.15–0.30 |
| 90–450 (<5× 90) | VERY_SPARSE | Maximum shrinkage | 0.05–0.15 |
| <90 | CAMEO | Use only as binary indicator | 0.00–0.05 |

---

### 6.2 – Per-Tier Adjustments

**Tier: SPARSE (450–900 minutes)**

```
Example: DEF with 600 minutes (6.67 × 90)

Goal-setting approach:
  cs_per_90_observed = 2 clean sheets / 6.67 = 0.30
  cs_per_90_prior = 0.25 (DEF baseline)
  prior_90s = 30 (DEF CS is reliable; use weak shrinkage)
  prior_90s_adjusted = 30 × 1.5  [increase shrinkage for sparse data]
                     = 45
  
  posterior = (6.67 × 0.30 + 45 × 0.25) / (6.67 + 45)
            = (2 + 11.25) / 51.67
            = 0.257
            [minimal pull from observed; mostly prior]
  
  reliability = 6.67 / 51.67 = 0.13
```

**Tier: VERY_SPARSE (<450 minutes)**

```
Example: FWD with 2 appearances, 120 minutes total (1.33 × 90)

Handling:
  goals_per_90_observed = 1 goal / 1.33 = 0.75 per 90 (likely luck)
  goals_per_90_prior = 0.45
  prior_90s_adjusted = 40 × 2.5  [maximum shrinkage for very sparse]
                     = 100
  
  posterior = (1.33 × 0.75 + 100 × 0.45) / (1.33 + 100)
            = (1 + 45) / 101.33
            = 0.454
            [nearly equal to prior; observed signal ignored]
  
  reliability = 1.33 / 101.33 = 0.013
  uncertainty = very_high
```

**Tier: CAMEO (<90 minutes)**

```
Example: Goalkeeper with 1 appearance, 45 minutes

Handling:
  Do not compute per-90 stats (unreliable)
  Use only binary indicators:
    - appeared = 1
    - played_full_match = 0
  Use team-level CS rate + random effect if modeling CS
  Flag = "CAMEO_ONLY"
  Projected points = near-zero
```

---

### 6.3 – Minimum Threshold for Feature Calculation

| Feature | Minimum Minutes | Fallback |
|---------|-----------------|----------|
| effective_90s | 0 | Use value (can be 0) |
| goals_per_90 | 450 | Use prior if <450 min; flag as "estimated" |
| assists_per_90 | 450 | Use prior if <450 min |
| clean_sheets_per_90 | 450 | Use prior; position-adjust |
| influence_per_90 | 450 | Use prior |
| bps_per_90 | 450 | Use prior |
| starts_ratio | 2 appearances | Use count if 1–2 games |
| minutes_trend | 90 minutes | Cannot compute; set = 0 (no trend) |
| attacking_form_trend | 90 minutes | Cannot compute; set = 0 |
| defensive_form_trend | 90 minutes | Cannot compute; set = 0 |

---

## SECTION 7: AGGREGATION PROCEDURES (PSEUDOCODE)

### 7.1 – Compute Season-Level Metrics

```python
def compute_season_metrics(player_gameweeks, position, season_id, min_minutes=450):
    """
    Input:
      player_gameweeks: list of (GW, minutes, goals, assists, cs, bps, influence, ...)
      position: "GK" | "DEF" | "MID" | "FWD"
      season_id: "2025-26"
      min_minutes: minimum minutes for reliable computation
    
    Output:
      dict with season-level metrics and flags
    """
    
    total_minutes = sum(gw.minutes for gw in player_gameweeks if gw.minutes)
    total_goals = sum(gw.goals_scored for gw in player_gameweeks)
    total_assists = sum(gw.assists for gw in player_gameweeks)
    total_cs = sum(gw.clean_sheets for gw in player_gameweeks)
    total_bps = sum(gw.bps for gw in player_gameweeks)
    total_influence = sum(gw.influence for gw in player_gameweeks)
    appearances = len([gw for gw in player_gameweeks if gw.minutes > 0])
    starts = sum(1 for gw in player_gameweeks if gw.minutes >= 45)
    
    effective_90s = total_minutes / 90
    
    # Reliability check
    if total_minutes < min_minutes:
        reliability_flag = "SPARSE"
        effective_90s_adj = min(effective_90s, 5)  # Cap at 5× 90
    else:
        reliability_flag = "RELIABLE"
        effective_90s_adj = effective_90s
    
    # Per-90 calculations
    metrics = {
        "season_id": season_id,
        "position": position,
        "effective_90s": effective_90s,
        "effective_90s_adj": effective_90s_adj,
        "goals_per_90": total_goals / max(effective_90s, 0.1) if effective_90s > 0 else 0,
        "assists_per_90": total_assists / max(effective_90s, 0.1) if effective_90s > 0 else 0,
        "cs_per_90": total_cs / max(effective_90s, 0.1) if effective_90s > 0 else 0,
        "bps_per_90": total_bps / max(effective_90s, 0.1) if effective_90s > 0 else 0,
        "influence_per_90": total_influence / max(effective_90s, 0.1) if effective_90s > 0 else 0,
        "starts_ratio": starts / max(appearances, 1),
        "reliability_flag": reliability_flag,
        "appearances": appearances,
    }
    
    # Impute missing values
    for metric in ["goals_per_90", "assists_per_90", "cs_per_90", "bps_per_90", "influence_per_90"]:
        if isnan(metrics[metric]) or metrics[metric] == 0 and total_minutes == 0:
            metrics[metric] = position_priors[position][metric]["mean"]  # Use prior
    
    return metrics
```

---

### 7.2 – Compute Recent Form Metrics

```python
def compute_recent_metrics(player_gameweeks, position, as_of_gw, window_size=5):
    """
    Recent = last window_size GWs up to (but not including) as_of_gw
    """
    recent_gws = [gw for gw in player_gameweeks 
                  if gw.gameweek_id >= (as_of_gw - window_size) and gw.gameweek_id < as_of_gw]
    
    if not recent_gws:
        return {
            "effective_90s_recent": 0,
            "goals_per_90_recent": 0,
            "assists_per_90_recent": 0,
            "attacking_form_trend": 0,
            "defensive_form_trend": 0,
        }
    
    total_minutes_recent = sum(gw.minutes for gw in recent_gws if gw.minutes)
    effective_90s_recent = total_minutes_recent / 90
    
    attacking_goals_recent = sum(gw.goals_scored for gw in recent_gws)
    attacking_assists_recent = sum(gw.assists for gw in recent_gws)
    attacking_score_recent = (attacking_goals_recent + attacking_assists_recent) / max(effective_90s_recent, 0.1)
    
    defensive_cs_recent = sum(gw.clean_sheets for gw in recent_gws)
    defensive_influence_recent = sum(gw.influence for gw in recent_gws)
    defensive_score_recent = (defensive_cs_recent * 4 + defensive_influence_recent) / max(effective_90s_recent, 0.1)
    
    # Compute trends vs. season average
    attacking_form_trend = (attacking_score_recent - season_attacking_avg) / max(season_attacking_avg, 0.1)
    defensive_form_trend = (defensive_score_recent - season_defensive_avg) / max(season_defensive_avg, 0.1)
    
    return {
        "effective_90s_recent": effective_90s_recent,
        "goals_per_90_recent": attacking_goals_recent / max(effective_90s_recent, 0.1),
        "assists_per_90_recent": attacking_assists_recent / max(effective_90s_recent, 0.1),
        "attacking_form_trend": clip(attacking_form_trend, -0.5, 1.0),
        "defensive_form_trend": clip(defensive_form_trend, -0.5, 1.0),
    }
```

---

### 7.3 – Apply Bayesian Shrinkage

```python
def apply_bayesian_shrinkage(metric_value, metric_name, position, effective_90s):
    """
    Shrink player's metric toward position prior using Empirical Bayes
    """
    prior = position_priors[position][metric_name]
    prior_mean = prior["mean"]
    prior_std = prior["std"]
    prior_90s = prior["prior_90s"]
    
    posterior = (effective_90s * metric_value + prior_90s * prior_mean) / (effective_90s 
                                                                           + prior_90s)
    posterior_std = prior_std / sqrt(effective_90s + prior_90s)
    reliability = effective_90s / (effective_90s + prior_90s)
    
    return {
        "posterior": posterior,
        "posterior_std": posterior_std,
        "reliability": clip(reliability, 0, 1),
    }
```

---

## SECTION 8: CALIBRATION & VALIDATION

### 8.1 – Sanity Checks for Computed Features

After computing all features for a player, perform sanity checks:

```python
def validate_features(player_features, position):
    warnings = []
    
    # Check 1: Per-90 values should be reasonable
    if player_features["goals_per_90"] > 2.0 and position != "FWD":
        warnings.append(f"Unusually high goal rate: {player_features['goals_per_90']}")
    
    # Check 2: Reliability should be in [0, 1]
    for metric in ["goals", "assists", "cs", "bps"]:
        rel = player_features[f"reliability_{metric}"]
        if rel < 0 or rel > 1:
            warnings.append(f"Reliability {metric} out of range: {rel}")
    
    # Check 3: Trends should be reasonable
    for trend_metric in ["attacking_form_trend", "defensive_form_trend"]:
        if abs(player_features[trend_metric]) > 2.0:
            warnings.append(f"Extreme {trend_metric}: {player_features[trend_metric]}")
    
    # Check 4: Starts ratio should be [0, 1]
    if not (0 <= player_features["starts_ratio"] <= 1):
        warnings.append(f"Starts ratio out of range: {player_features['starts_ratio']}")
    
    return warnings
```

---

### 8.2 – Prior Calibration Process

At start of season, calibrate priors from historical data:

```python
def calibrate_position_priors(historical_seasons, min_seasons=3):
    """
    For each position + metric, compute empirical baseline from data
    """
    
    for position in ["GK", "DEF", "MID", "FWD"]:
        season_metrics = []
        
        for season_id in historical_seasons:
            player_season_data = [
                p for p in season_data 
                if p.position == position and p.effective_90s >= 5
            ]
            
            if player_season_data:
                for metric in ["goals_per_90", "assists_per_90", ...]:
                    values = [p[metric] for p in player_season_data if not isnan(p[metric])]
                    
                    if values:
                        median = percentile(values, 50)
                        std = std(values)
                        q25 = percentile(values, 25)
                        q75 = percentile(values, 75)
                        
                        season_metrics.append({
                            "season": season_id,
                            "metric": metric,
                            "median": median,
                            "std": std,
                            "q25": q25,
                            "q75": q75,
                        })
        
        # Aggregate across seasons with recency weighting
        for metric in ["goals_per_90", "assists_per_90", ...]:
            metric_values = [m for m in season_metrics if m["metric"] == metric]
            
            weighted_median = weighted_avg([m["median"] for m in metric_values],
                                           weights=recency_weights(metric_values))
            weighted_std = weighted_avg([m["std"] for m in metric_values],
                                        weights=recency_weights(metric_values))
            
            position_priors[position][metric] = {
                "mean": weighted_median,
                "std": weighted_std,
                "prior_90s": prior_90s_lookup[position][metric],
            }
```

---

## SECTION 9: LEAKAGE PREVENTION CHECKLIST

### 9.1 – Temporal Consistency

For any backtest as-of gameweek `t`:

```
✓ Use only data from GWs 1 to t-1 (strictly before GW t)
✓ Compute recent window from GWs (t-5) to (t-1)
✓ Use bootstrap snapshot as-of deadline before GW t
✓ Use fixture data for GW t onwards (future)
✓ Do not use injury news with timestamp after GW t deadline
✓ Do not use price snapshots after GW t deadline
```

### 9.2 – Season Boundary Handling

```
✓ When computing seasonal metrics, use exactly GWs 1–38
✓ For mid-season backtests (e.g., GW 20), use GWs 1–19 for season stats
✓ Never mix GWs from different seasons in aggregation
✓ For new season (GW 1), use priors only (no prior season data in current season)
```

### 9.3 – Historical Data for Priors

```
✓ Use GWs 1–38 from each of prior N seasons for calibrating Bayesian priors
✓ Do not use current-season data to define current-season priors
✓ Freeze priors at start of season; do not update mid-season
```

---

## SECTION 10: IMPLEMENTATION CHECKLIST

Before coding, verify:

- [ ] Position priors computed from training data (5 historical seasons)
- [ ] Per-90 baseline values (median per position) documented
- [ ] Prior_90s values set for each position + metric combination
- [ ] Missing value imputation rules specified (goals_per_90, assists, bonus, etc.)
- [ ] Minimum minutes thresholds confirmed for each feature
- [ ] Shrinkage intensities set and validated
- [ ] Edge case handling for <90 minutes, 0 appearances, new signings documented
- [ ] Sanity checks implemented in validation function
- [ ] Leakage prevention checklist confirmed
- [ ] Recency weighting decay (0.45/year) documented
- [ ] Aggregation window definitions (full season, recent 5 GWs) locked in

---

## SUMMARY TABLE: Feature Computation Blueprint

| Feature | Agg Window | Per-90 | Min Minutes | Bayes Prior_90s | Position-Specific |
|---------|-----------|--------|-------------|-----------------|-------------------|
| effective_90s_season | Full (1–38) | N/A | 0 | 30 | Baseline differs |
| effective_90s_recent | Recent (5 GWs) | N/A | 0 | 30 | Baseline differs |
| goals_per_90 | Full / Recent | Yes | 450 | 40–150 | Yes (0.45 FWD / 0.10 MID / 0.02 DEF / 0 GK) |
| assists_per_90 | Full / Recent | Yes | 450 | 50–100 | Yes (0.08 MID / 0.04 FWD / 0.03 DEF / 0 GK) |
| clean_sheets_per_90 | Full / Recent | Yes | 450 | 25–100 | Yes (0.35 GK / 0.25 DEF / 0.08 MID / 0 FWD) |
| influence_per_90 | Full / Recent | Yes | 450 | 50–80 | Yes |
| bps_per_90 | Full / Recent | Yes | 450 | 40–70 | Yes |
| starts_ratio | Full / Recent | No | 2 | 60 | Position baseline |
| minutes_trend | Full vs Recent | Derived | 450 | — | No |
| availability_score | Point-in-time | N/A | — | — | No |
| attacking_form_trend | Full vs Recent | Derived | 450 | — | No |
| defensive_form_trend | Full vs Recent | Derived | 450 | — | No |
| position_id | Point-in-time | N/A | — | — | N/A (IS the position) |
| team_attack_strength | Bootstrap | N/A | — | — | No |
| team_defence_strength | Bootstrap | N/A | — | — | No |
| fdr_next_n | Future | N/A | — | — | No |
| home_away_ratio | Future 5 GWs | Derived | — | — | No |
| points_per_million | Derived | N/A | — | — | Implicit (position-normalized) |

---

**Phase 3 Complete**

Ready for Phase 4: Expected-Value Model Design (component-wise points decomposition)

