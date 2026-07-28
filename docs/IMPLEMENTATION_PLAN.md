# Phase 6: Implementation Plan

**Date:** 2026-07-27  
**Scope:** Design modular code structure for implementing new recommendation system  
**Output:** Complete implementation roadmap with file modifications, integration points, testing strategy  

---

## OVERVIEW

**Purpose:** Define the exact code changes required to go from current system to new component-based recommendation model.

**Key Questions:**
1. Which files need modification?
2. What new functions must be added?
3. How do we integrate without breaking existing pipeline?
4. What is the testing strategy?
5. Can old and new systems run in parallel during transition?

**Approach:** Modular, staged implementation with backward compatibility where possible.

---

## SECTION 1: ARCHITECTURE OVERVIEW

### 1.1 – Current Data Flow (Existing System)

```
Raw FPL API Data
    ↓
[transformation/core.py] → Ingest gameweek-level statistics
    ↓
fact_player_gw (18 fields: goals, assists, CS, bonus, BPS, etc.)
    ↓
[transformation/historical.py] → Aggregate per season
    ↓
fact_player_season (season_ppm, points_per_90, effective_90s, aggregates)
    ↓
[models.py: bayesian_player_value()] → Shrink toward prior
    ↓
player_posterior (posterior_ppm, posterior_std, reliability_score)
    ↓
[models.py: build_recommendations()] → Adjust for fixtures/form/availability
    ↓
recommendation_score (single metric; used for ranking)
    ↓
Output: Top-N recommendations by score
```

**Current Bottleneck:** All information collapses to season_ppm early; no component-wise reasoning.

---

### 1.2 – New Data Flow (Proposed System)

```
Raw FPL API Data (same ingestion)
    ↓
fact_player_gw (18 fields; unchanged)
    ↓
[transformation/historical.py: EXTENDED] 
    → Compute per-90 variants for each component
    → Apply position-specific Bayesian shrinkage per component
    ↓
fact_player_season_extended (goals_per_90, assists_per_90, cs_per_90, bps_per_90, etc.)
                              WITH shrunk posteriors per metric per position
    ↓
[transformation/analytics.py: NEW FUNCTIONS]
    → Compute fixture context (FDR, team strength, home/away)
    → Compute form trends (recent mean, recency-weighted)
    ↓
player_fixture_context (fdr_next_n, team_attack_strength, form_trend, etc.)
    ↓
[models.py: NEW build_expected_value_model()]
    → For each component (goals, assists, CS, bonus, appearance, discipline)
    → Estimate probability/expected points for next GW
    → Combine via position-specific point multipliers
    ↓
player_component_expectations (expected_goals, expected_assists, expected_cs, 
                                expected_bonus, expected_appearance, expected_discipline,
                                expected_points_per_match, uncertainty)
    ↓
[models.py: NEW compute_recommendation_score_v2()]
    → Multi-horizon expansion (1/3/5 GW)
    → Value-per-pound adjustment
    → Risk penalty via uncertainty
    ↓
recommendation_score_v2 (NEW metric; component breakdown attached)
    ↓
Output: Top-N recommendations with explainability (which components drive score)
```

**Key Difference:** Information preserved at component level; explainability maintained.

---

## SECTION 2: FILE MODIFICATION ROADMAP

### 2.1 – Files to Modify (No Breaking Changes)

#### **A. src/fpl_pipeline/transformation/historical.py**

**Current State:**
- `player_season_aggregates()`: computes season_ppm, points_per_90, effective_90s, goals, assists, clean_sheets, bonus aggregates
- Uses only basic statistics; doesn't apply Bayesian shrinkage

**Planned Changes:**

1. **Add new function: `compute_per_90_metrics_with_shrinkage()`**
   ```
   Purpose: For each position-metric combination, compute per-90 rates with Bayesian shrinkage
   
   Input:
     - player_gw data (all gameweeks for player)
     - position (GK, DEF, MID, FWD)
     - metric_name (goals, assists, clean_sheets, bps, influence, etc.)
     - prior_distribution (calibrated from historical seasons)
   
   Output:
     - posterior_metric_per_90 (shrunk estimate)
     - posterior_metric_std (Bayesian uncertainty)
     - effective_90s (for this metric; may differ by metric due to playing time)
     - reliability_score (0-1; confidence)
   
   Implementation:
     1. Compute empirical mean_metric_per_90 from player_gw data
     2. Compute empirical std from variance
     3. Apply Bayesian Normal-mean shrinkage (see Phase 3 formulas)
     4. Compute shrinkage intensity based on effective_90s and prior_equivalent_90s
     5. Return posterior parameters
   
   Notes:
     - Some metrics sparse for low-minute players (DEF goals, MID CS)
     - Handle edge cases: 0 events → NA → posterior = prior mean
     - Position-specific priors essential
   ```

2. **Extend `player_season_aggregates()` output**
   ```
   Add columns to fact_player_season table:
   
   Per-position metrics (repeats for each position):
     - posterior_goals_per_90
     - posterior_goals_per_90_std
     - posterior_assists_per_90
     - posterior_assists_per_90_std
     - posterior_clean_sheets_per_90
     - posterior_clean_sheets_per_90_std
     - posterior_bps_per_90
     - posterior_bps_per_90_std
     - posterior_influence_per_90
     - posterior_influence_per_90_std
     - effective_90s_goals (may differ by metric)
     - effective_90s_assists
     - effective_90s_cs
     - reliability_score_goals
     - reliability_score_assists
     - reliability_score_cs
     - reliability_score_bps
   
   Also add position-independent:
     - season (season identifier; for filtering)
     - position (GK, DEF, MID, FWD)
   ```

3. **Add helper function: `calibrate_position_priors()`**
   ```
   Purpose: Compute position-specific Bayesian priors from historical seasons
   
   Input:
     - historical_seasons (dataframe of all seasons)
     - position (GK, DEF, MID, FWD)
   
   Output:
     - prior_goals_per_90 (mean)
     - prior_goals_per_90_std (std)
     - prior_assists_per_90 (mean)
     - ... (repeat for all metrics)
     - prior_equivalent_90s (prior strength; 25-150 by position/metric)
   
   Implementation:
     - Group historical_seasons by position
     - For each metric: compute median and IQR (robust to outliers)
     - Use IQR/1.35 as robust std estimate
     - Use domain knowledge to set prior_equivalent_90s
     - Return dictionary of priors keyed by (position, metric)
   ```

**Integration Point:**
- Called during `ingestion.historical_backfill()` pipeline
- Runs after raw data ingested; populates fact_player_season
- Backward compatible: old columns preserved; new columns added

**Test Strategy:**
```
Unit tests for compute_per_90_metrics_with_shrinkage():
  ✓ Edge case: player with 0 goals → posterior = prior mean
  ✓ Edge case: player with 1000+ effective_90s → posterior ≈ empirical mean
  ✓ Edge case: very rare metric (DEF goals) → heavy shrinkage
  ✓ Position-specific priors applied correctly
  ✓ Uncertainty decreases with more effective_90s
  ✓ Backward compatibility: all old columns still present

Integration tests:
  ✓ fact_player_season populated with all new columns
  ✓ NULL handling for missing historical data
  ✓ No performance regression (<5% slower than before)
  ✓ Leakage prevention: priors don't use data from target season
```

---

#### **B. src/fpl_pipeline/transformation/analytics.py**

**Current State:**
- `fixture_run()`: computes next_1/3/5_fixture_average_difficulty
- `team_strength_metrics()`: minimal team stats

**Planned Changes:**

1. **Extend `fixture_run()` with new columns**
   ```
   Current output:
     - next_1_fixture_difficulty
     - next_3_fixture_difficulty
     - next_5_fixture_difficulty
   
   Add:
     - next_fixture_opponent (team name; for analysis)
     - next_fixture_is_home (1=home, 0=away)
     - next_fixture_date (for congestion analysis)
   
   For next_3 and next_5:
     - fixture_congestion_score (days between matches)
     - home_away_ratio (for multi-fixture windows)
   ```

2. **Add new function: `compute_team_attacking_defensive_strength()`**
   ```
   Purpose: Compute offensive/defensive quality metrics by team (not by player)
   
   Input:
     - fact_player_gw (all historical data)
     - position (option to compute per position)
   
   Output (for each team, each season):
     - team_attack_strength (strength of attacking players)
     - team_defence_strength (strength of defensive players)
     - team_cs_frequency (CS rate)
     - team_goal_rate (goals per match)
     - team_assists_rate (assists per match)
   
   Implementation:
     - Group by team
     - For attacking players (MID/FWD): mean goals + assists per 90
     - For defensive players (DEF/GK): mean CS per 90, mean clean sheet frequency
     - Average across player pool
   
   Usage: To adjust expectations based on "strong attack" or "good defence"
   ```

3. **Add new function: `compute_home_away_ratios()`**
   ```
   Purpose: Home vs. away performance differential
   
   Input:
     - fact_player_gw (with is_home flag)
   
   Output (for each player, or by team):
     - home_away_ratio_goals (goals_home / goals_away)
     - home_away_ratio_assists
     - home_away_ratio_cs
   
   Usage: Adjust next-match expectation based on home/away
   ```

4. **Add new function: `compute_form_trends()`**
   ```
   Purpose: Recent momentum vs. season average
   
   Input:
     - fact_player_gw (sorted chronologically)
     - window_size (e.g., 5 for last 5 GWs)
   
   Output (for each player as-of each gameweek):
     - recent_points_per_gw_mean (last 5 GWs average)
     - recent_goals_mean, recent_assists_mean (component level)
     - form_trend_direction (+1=improving, 0=stable, -1=declining)
     - form_trend_magnitude (% change vs. season average)
   
   Usage: Adjust expectations if player in hot/cold streak
   ```

**Integration Point:**
- Called during recommendation generation (pre-compute; cache results)
- Runs daily/before each GW deadline

**Test Strategy:**
```
Unit tests:
  ✓ Team strength aggregation correct
  ✓ Home/away ratios computed from correct subset
  ✓ Form trends identify streaks correctly
  ✓ Edge case: new team (no history) → use default
  ✓ Edge case: player with 1 match only → don't compute ratios

Integration tests:
  ✓ All players have form_trend assigned
  ✓ Form trends reasonable (mostly -1, 0, +1; few >0.2 magnitude)
  ✓ Consistency: same player same GW produces same form trend
```

---

#### **C. src/fpl_pipeline/models.py**

**Current State:**
- `bayesian_player_value()`: shrinks season_ppm; outputs posterior_ppm, posterior_uncertainty, reliability_score
- `build_recommendations()`: applies fixture/form/availability factors; outputs recommendation_score

**Planned Changes:**

1. **Keep existing functions unchanged (backward compatibility)**
   ```
   bayesian_player_value() stays as-is
   build_recommendations() stays as-is (old recommendation_score preserved)
   
   Rationale: Allows A/B testing and fallback if new model has issues
   ```

2. **Add new function: `build_expected_value_model()`**
   ```
   Purpose: Compute component-wise expected points for next GW(s)
   
   Signature:
     def build_expected_value_model(
         player_season_extended,      # from historical.py (new columns)
         fixture_context,             # from analytics.py (new)
         bootstrap_snapshot,          # current FPL API state
         prediction_horizon=1,        # 1, 3, or 5 GWs
         as_of_gameweek=None,         # for reproducibility
     ) -> pd.DataFrame
   
   Output columns:
     - player_id
     - position
     - expected_goals_per_match
     - expected_assists_per_match
     - expected_clean_sheets_per_match
     - expected_bonus_points_per_match
     - expected_appearance_fraction
     - expected_discipline_points_per_match
     
     - expected_points_per_match (weighted sum by position)
     - uncertainty_per_match
     
     - expected_points_h_gw (expanded to h gameweeks)
     - uncertainty_h_gw
     - confidence_interval_95_lower
     - confidence_interval_95_upper
     
     - points_per_million (current_points / (price_in_pence/10))
     - fixture_adjustments_applied (dict for explainability)
   
   Implementation (per component):
   
   a) GOALS:
     posterior_goals_per_90 = from player_season_extended
     minutes_next_gw = estimate from availability_factor × typical_minutes
     fixture_adjustment = 3.0 / fdr_next (easy=1, hard=5)
     form_adjustment = 1 + form_trend_magnitude × 0.1
     
     expected_goals = posterior_goals_per_90 × (minutes_next_gw/90) 
                    × fixture_adjustment × form_adjustment
   
   b) ASSISTS:
     Similar to goals; use posterior_assists_per_90
     form_adjustment same
   
   c) CLEAN SHEETS:
     P(team_cs) = from team_defence_strength or historical CS rate
     P(on_pitch) = availability_factor × expected_minutes / typical_GW_minutes
     
     expected_cs = P(team_cs) × P(on_pitch)
     (Probability; then convert to points via position multiplier)
   
   d) BONUS:
     recent_bps_per_90 = from player_season_extended (posterior)
     P(top3_bps) = logistic(recent_bps_per_90, position_params)
     E[bonus | top3] ≈ 1.5 (historical average)
     
     expected_bonus = P(top3_bps) × E[bonus | top3]
     (OR: use historical bonus frequency if available)
   
   e) APPEARANCE:
     availability_factor = from bootstrap status (availability % or binary)
     expected_appearance = availability_factor (fraction of match played)
   
   f) DISCIPLINE:
     posterior_yellows_per_90 = from player_season_extended (or prior)
     posterior_reds_per_90 = from player_season_extended
     
     expected_yellows = posterior_yellows_per_90 × (minutes/90) × -1
     expected_reds = posterior_reds_per_90 × (minutes/90) × -2
     expected_discipline = expected_yellows + expected_reds
     (Negative contribution to points)
   
   g) COMBINE BY POSITION (position-specific multipliers):
     For FWD:
       expected_points = 4×expected_goals 
                       + 1×expected_assists 
                       + 1×expected_cs 
                       + expected_bonus 
                       + 1×expected_appearance 
                       + expected_discipline
     
     (See Phase 4 for all position formulas)
   
   h) UNCERTAINTY PROPAGATION:
     uncertainty_per_component = posterior_std × (minutes/90) × fixture_adj
     total_uncertainty = sqrt(SUM(component_uncertainty²))
     uncertainty_h_gw = total_uncertainty × sqrt(prediction_horizon)
   
   i) MULTI-HORIZON:
     If prediction_horizon=3:
       expected_points_3gw = expected_points × 3 × availability_factor
       uncertainty_3gw = uncertainty × sqrt(3)
   ```

3. **Add new function: `compute_recommendation_score_v2()`**
   ```
   Purpose: Generate final recommendation score combining expected value and risk
   
   Signature:
     def compute_recommendation_score_v2(
         player_component_expectations,  # from build_expected_value_model()
         points_per_million,
         risk_adjustment_factor=0.5,    # parameter to tune
         use_multi_horizon=5,           # 1, 3, or 5 GWs
     ) -> pd.DataFrame
   
   Formula:
     recommendation_score_v2 = expected_points_h_gw 
                             + (points_per_million / 10) × 0.04  # value bonus
                             - risk_adjustment_factor × uncertainty_h_gw  # uncertainty penalty
   
   Output:
     - recommendation_score_v2
     - expected_points_component_breakdown (dict for explainability)
     - uncertainty_range (95% CI)
     - ranking_tier (top-20, 21-50, 51-100, 100+)
   ```

4. **Add new function: `generate_recommendation_explanations()`**
   ```
   Purpose: Produce human-readable explanations for recommendations
   
   Output per player:
     - Why are they recommended? (which components drive score)
     - What's the expected point breakdown? (goals vs. assists vs. bonus)
     - What are risks? (low availability, hard fixtures, form decline)
     - Comparison to similar players (positional peer)
   
   Implementation:
     For each component, identify if it's above/below position median
     Rank components by contribution to expected_points
     Generate narrative ("Strong goal scorer" vs. "Risky availability")
   ```

**Integration Points:**
- Called before recommendation output generation
- Uses posterior estimates from historical.py and context from analytics.py
- Old functions remain; new functions run in parallel during transition

**Test Strategy:**
```
Unit tests for build_expected_value_model():
  ✓ Component calculations match Phase 4 formulas
  ✓ Position-specific point multipliers applied correctly
  ✓ Fixture adjustment scales expected points (easy→high, hard→low)
  ✓ Form adjustment modulates expectations
  ✓ Uncertainty increases with more factors
  ✓ Multi-horizon expansion scales correctly (h=1 vs. h=5)
  ✓ Edge case: new player (no posterior) → use prior
  ✓ Edge case: injured player (availability=0) → expected_points→0

Unit tests for compute_recommendation_score_v2():
  ✓ Score increases with higher expected points
  ✓ Score decreases with higher uncertainty
  ✓ Score increases with better value (points_per_million)
  ✓ Risk adjustment factor tuned to reproduce realistic scores

Integration tests:
  ✓ All players have expected_points and recommendation_score_v2
  ✓ Top-20 recommendations rank sensibly (high scorers at top)
  ✓ Explainability output readable and accurate
  ✓ Backward compatibility: old recommendation_score still works
  ✓ No NaN or infinite values in output
```

---

### 2.2 – New Files to Create

#### **A. src/fpl_pipeline/models_v2.py (Optional; or extend models.py)**

**Option 1 (Recommended): Extend models.py**
```
Add all new functions to existing models.py
Organize with comment headers:
  # ==================== OLD SYSTEM (DEPRECATED) ====================
  # bayesian_player_value()
  # build_recommendations()
  
  # ==================== NEW COMPONENT-BASED SYSTEM ====================
  # build_expected_value_model()
  # compute_recommendation_score_v2()
  # generate_recommendation_explanations()
```

**Option 2: Create separate models_v2.py**
```
New file with only new functions
Old models.py left untouched
Requires explicit import/routing in pipeline

Advantage: Cleaner code separation
Disadvantage: Requires more plumbing to switch between old/new
```

**Recommendation:** Option 1 (extend models.py) for simplicity.

---

#### **B. src/fpl_pipeline/component_estimators.py (Optional; helper module)**

**Purpose:** Encapsulate component-wise estimation logic

```python
class ComponentEstimator:
    """Helper class for estimating individual components."""
    
    def __init__(self, position, player_features):
        self.position = position
        self.player_features = player_features
    
    def estimate_goals(self, fixture_adjustment, form_adjustment):
        """Estimate expected goals for next match."""
        # Implementation
    
    def estimate_assists(self, fixture_adjustment, form_adjustment):
        """Estimate expected assists."""
    
    def estimate_clean_sheets(self, team_defence, availability):
        """Estimate clean sheet probability."""
    
    def estimate_bonus(self, bps_per_90, position_params):
        """Estimate bonus points probability."""
    
    # ... other components
```

**Integration:** Optional; can inline into build_expected_value_model() for simplicity initially.

---

### 2.3 – Existing Files NOT to Modify

```
✓ src/fpl_pipeline/pipeline.py (no changes)
  - Call both old and new functions; router downstream

✓ src/fpl_pipeline/api/client.py (no changes)
  - Data ingestion unchanged

✓ src/fpl_pipeline/ingestion/download.py (no changes)
  - Raw data download unchanged

✓ src/fpl_pipeline/validation/checks.py (no changes)
  - Validation rules unchanged (or extended for new columns only)

✓ src/fpl_pipeline/storage.py (no changes)
  - Database schema may extend, but compatibility maintained
```

---

## SECTION 3: DATABASE SCHEMA CHANGES

### 3.1 – Existing Tables: Backward Compatible Extensions

**fact_player_season (existing)**
```sql
-- Existing columns remain unchanged
ALTER TABLE fact_player_season ADD COLUMN (
    -- Per-90 metrics with Bayesian shrinkage (NEW)
    posterior_goals_per_90 FLOAT,
    posterior_goals_per_90_std FLOAT,
    posterior_assists_per_90 FLOAT,
    posterior_assists_per_90_std FLOAT,
    posterior_clean_sheets_per_90 FLOAT,
    posterior_clean_sheets_per_90_std FLOAT,
    posterior_bps_per_90 FLOAT,
    posterior_bps_per_90_std FLOAT,
    posterior_influence_per_90 FLOAT,
    posterior_influence_per_90_std FLOAT,
    
    -- Effective 90s by metric (if varies)
    effective_90s_goals FLOAT,
    effective_90s_assists FLOAT,
    effective_90s_cs FLOAT,
    effective_90s_bps FLOAT,
    
    -- Reliability scores (0-1)
    reliability_score_goals FLOAT,
    reliability_score_assists FLOAT,
    reliability_score_cs FLOAT,
    reliability_score_bps FLOAT
);

-- New indices for performance
CREATE INDEX idx_player_season_position ON fact_player_season(player_id, season, position);
CREATE INDEX idx_player_season_posterior ON fact_player_season(player_id, posterior_goals_per_90);
```

**analytics_team_stats (new or extended)**
```sql
-- New table for team-level aggregate stats
CREATE TABLE IF NOT EXISTS analytics_team_stats (
    team_id INT NOT NULL,
    season VARCHAR(10) NOT NULL,
    
    -- Attack strength
    team_attack_strength FLOAT,
    team_goal_rate FLOAT,
    team_assists_rate FLOAT,
    
    -- Defence strength
    team_defence_strength FLOAT,
    team_cs_frequency FLOAT,
    
    -- Other
    home_advantage_ratio FLOAT,
    
    -- Metadata
    last_updated TIMESTAMP,
    
    PRIMARY KEY (team_id, season)
);
```

**analytics_form_trends (new)**
```sql
-- Form trends at each gameweek
CREATE TABLE IF NOT EXISTS analytics_form_trends (
    player_id INT NOT NULL,
    gameweek INT NOT NULL,
    season VARCHAR(10) NOT NULL,
    
    -- Recent stats
    recent_points_mean FLOAT,
    recent_goals_mean FLOAT,
    recent_assists_mean FLOAT,
    
    -- Trend
    form_trend_direction INT,  -- -1, 0, +1
    form_trend_magnitude FLOAT,  -- % change vs season avg
    
    PRIMARY KEY (player_id, gameweek, season),
    FOREIGN KEY (player_id) REFERENCES dim_player(player_id)
);
```

### 3.2 – Data Migration Plan

```
Phase A (Pre-deployment):
  1. Run historical backfill with new schema
  2. Populate fact_player_season with new posterior columns
  3. Populate analytics_team_stats from historical seasons
  4. Validate against backtest expectations

Phase B (Initial deployment):
  1. Add columns to production database (non-blocking ALTER TABLE)
  2. Pre-populate with historical computation
  3. Deploy code with new functions

Phase C (Ongoing):
  1. Daily updates to fact_player_season (incremental per new GW)
  2. Daily updates to analytics_team_stats and analytics_form_trends
  3. Monitor NULL rates; investigate if missing data
```

---

## SECTION 4: INTEGRATION & ROUTING

### 4.1 – Pipeline Modification: Minimal Changes

**Current pipeline flow:**
```python
# src/fpl_pipeline/pipeline.py

def main_pipeline():
    # Ingestion
    ingest_raw_data()
    
    # Transformation (existing)
    player_season_aggregates()  # ← Will be extended to compute posteriors
    fixture_run()               # ← Will be extended with new columns
    
    # Modeling (old; add new alongside)
    player_value = bayesian_player_value()           # existing
    recommendations_old = build_recommendations()    # existing
    
    # Storage
    store_recommendations(recommendations_old)
```

**New pipeline flow (with parallel execution):**
```python
def main_pipeline():
    # Ingestion (unchanged)
    ingest_raw_data()
    
    # Transformation (extended)
    player_season_extended = player_season_aggregates()  # now includes posteriors
    fixture_context = fixture_run()                      # extended
    team_stats = compute_team_attacking_defensive_strength()  # new
    form_trends = compute_form_trends()                  # new
    
    # Modeling (OLD system)
    player_value_old = bayesian_player_value()
    recommendations_old = build_recommendations(player_value_old)
    
    # Modeling (NEW system) - runs in parallel
    player_component_exp = build_expected_value_model(
        player_season_extended, fixture_context, team_stats, form_trends
    )
    recommendations_new = compute_recommendation_score_v2(player_component_exp)
    
    # Explainability (NEW)
    explanations = generate_recommendation_explanations(recommendations_new)
    
    # Comparison & storage
    comparison = compare_old_vs_new(recommendations_old, recommendations_new)
    store_recommendations(recommendations_new, metadata={
        "system_version": "2_component_based",
        "old_system_score": recommendations_old,  # backup
        "comparison_metrics": comparison,
        "explanations": explanations
    })
```

### 4.2 – Feature Flags & A/B Testing

**Add configuration option:**
```python
# config/config.yaml

recommendation_system:
  version: "v2_component_based"  # Options: "v1_ppm", "v2_component_based"
  use_parallel_execution: true   # Run both old & new; compare
  
  fallback_to_v1_if_null: true   # If v2 produces NULL, use v1
  
  v2_parameters:
    risk_adjustment_factor: 0.5
    prediction_horizon: 5
    use_form_adjustment: true
    use_fixture_adjustment: true
```

**Environment variable override:**
```bash
export FPL_RECOMMENDATION_VERSION=v2_component_based
python -m fpl_pipeline.pipeline
```

---

## SECTION 5: TESTING STRATEGY

### 5.1 – Unit Test Organization

```
tests/
├── test_transformation_extended.py (NEW)
│   ├── test_compute_per_90_metrics_with_shrinkage()
│   ├── test_calibrate_position_priors()
│   ├── test_edge_case_zero_events()
│   └── test_backward_compatibility()
│
├── test_analytics_extended.py (NEW)
│   ├── test_compute_team_attacking_defensive_strength()
│   ├── test_compute_home_away_ratios()
│   ├── test_compute_form_trends()
│   └── test_fixture_context_extensions()
│
├── test_models_v2.py (NEW)
│   ├── test_build_expected_value_model()
│   ├── test_compute_recommendation_score_v2()
│   ├── test_generate_recommendation_explanations()
│   ├── test_component_calculations_match_phase4()
│   └── test_position_specific_multipliers()
│
├── test_integration_old_vs_new.py (NEW)
│   ├── test_parallel_execution()
│   ├── test_fallback_to_v1()
│   ├── test_backward_compatibility_full_pipeline()
│   └── test_no_data_loss()
│
└── (existing tests; no changes)
```

### 5.2 – Coverage Requirements

| Module | Coverage Target | Comment |
|--------|-----------------|---------|
| transformation/historical.py (extended) | 95% | Critical for posteriors |
| transformation/analytics.py (extended) | 90% | Context data; important |
| models.py (new functions) | 95% | Core recommendation logic |
| Integration tests | 80% | End-to-end pipeline |

### 5.3 – Test Data Strategy

```
Use existing historical seasons (2021-22, 2022-23, 2023-24, 2024-25):
  - Mock bootstrap snapshots at key GWs
  - Compute posteriors using 80% of data
  - Validate against held-out 20% (backtesting set)
  
Synthetic test cases:
  - New player (no history) → posterior = prior
  - Injured player (0 availability) → recommendation_score→0
  - Breakout season → posterior_std should decrease
  - Low-minute player → high uncertainty, heavy shrinkage
  
Edge cases:
  - Missing metric data → NA → handle gracefully
  - Extremely high/low values → no crash, reasonable bounds
  - Player transfer mid-season → identity matching
```

### 5.4 – Performance Testing

```
Benchmarks (target: <5% slowdown):
  - compute_per_90_metrics_with_shrinkage() on 500 players: <100ms
  - build_expected_value_model() on 500 players: <500ms
  - compute_recommendation_score_v2() on 500 players: <100ms
  - Full pipeline (both old + new): <2s total
  
Scaling:
  - Test with 5000+ players (FPL's actual size)
  - Ensure index usage (no full table scans)
  - Monitor memory usage (<1GB for full backtest)
```

---

## SECTION 6: ROLLOUT STRATEGY

### 6.1 – Deployment Phases

**Phase 1: Pre-deployment (Week 1)**
```
1. Merge feature branch (all code + tests)
2. Run full backtest locally (reproduce Phase 5 results)
3. Validate against baselines (hit rate > 55%, correlation > 0.50)
4. Code review (impact on data schemas, dependencies)
5. Deploy to staging environment
```

**Phase 2: Initial deployment (Week 2)**
```
1. Deploy new code (feature flag: v2_component_based OFF by default)
2. Enable v2 in background (shadow mode; don't affect users)
3. Monitor logs (any errors, NULL values, performance issues)
4. Compare recommendations_v2 vs. recommendations_v1
5. Check data quality (posteriors sensible, no NaNs)
```

**Phase 3: Graduated rollout (Week 3-4)**
```
1. Enable v2 for 10% of users (feature flag 10%)
2. Monitor A/B test metrics (if applicable; e.g., engagement, clicks)
3. Check recommendation diversity (v2 shouldn't collapse to same players as v1)
4. Increase to 50% if metrics favorable
5. Full rollout to 100% by end of Week 4
```

**Phase 4: Deprecation (Week 5+)**
```
1. Keep v1 code for fallback only (not default)
2. After 30 days stable operation, remove fallback path
3. Archive v1 recommendation scores for historical reference
4. Update documentation
```

### 6.2 – Rollback Plan

```
If metrics degrade (hit rate drops >5pp, correlation drops >0.1):
  1. Immediate action: Set feature flag recommendation_system.version = "v1_ppm"
  2. Investigate: Run diagnostics (NULL values? Schema issue? Leakage?)
  3. Fix: Either patch v2 or revert to v1
  4. Retest: Re-run backtests before re-enabling v2
```

---

## SECTION 7: DEPRECATION STRATEGY

### 7.1 – Old Recommendation Score Lifecycle

```
Week 1-4: Parallel execution
  - bayesian_player_value() still computed
  - build_recommendations() still called
  - OLD recommendation_score stored as backup
  - NEW recommendation_score_v2 is default

Week 5-8: Monitoring phase
  - v1 computed but not used
  - Available via API for debugging
  - Document comparison queries

Week 9+: Retirement phase
  - Remove v1 computation from main pipeline
  - Archive last 30 days of v1 scores
  - Keep v1 code for history (don't delete)
  - Update all documentation
```

### 7.2 – API Versioning (if applicable)

```
GET /recommendations (default):
  Returns recommendation_score_v2

GET /recommendations?version=v1:
  Returns old recommendation_score (deprecated)

GET /recommendations?version=v2:
  Explicitly request new version

POST /recommendations/explain:
  Returns explainability breakdown (v2 only)
```

---

## SECTION 8: MONITORING & OBSERVABILITY

### 8.1 – Metrics to Track

**Recommendation quality:**
```
- Daily: % of top-20 recommendations beating position median
- Daily: Mean recommendation_score distribution (any sudden shifts?)
- Weekly: Spearman correlation vs. actual points
- Weekly: Calibration MAE (expected vs. realized)
- Daily: % of recommendations with valid component estimates
```

**Data quality:**
```
- % of players with non-NULL posteriors (target: >99%)
- % of players with posteriors outside reasonable bounds (target: <0.1%)
- Mean uncertainty per position (should vary by position)
- NULL rate in new columns (target: 0%)
```

**Performance:**
```
- Pipeline execution time (target: <2s)
- Database query times (target: <100ms per player batch)
- Memory usage (target: <1GB)
```

### 8.2 – Alerting

```
Alert if:
- Hit rate drops below 50% (something broken)
- >5% NULL values in posteriors (data issue)
- Pipeline execution >5s (performance degradation)
- Correlation drops below 0.40 (model degradation)
- Variance in recommendation_score too high/low (schema issue)
```

### 8.3 – Logging

```
Log at INFO level:
- Number of players processed
- Mean/median recommendation_score
- Top-5 recommended players (for manual verification)

Log at DEBUG level:
- Per-player component estimates (goals, assists, etc.)
- Fixture adjustments applied
- Form adjustments applied
- Individual component uncertainties
```

---

## SECTION 9: IMPLEMENTATION CHECKLIST

### Pre-implementation
- [ ] Phase 6 review & approval
- [ ] Database schema migration planned
- [ ] Test data prepared (all 5 historical seasons)
- [ ] Feature flag infrastructure ready
- [ ] Monitoring dashboards set up
- [ ] Documentation updated (README, docstrings)

### Code implementation
- [ ] extended historical.py:
  - [ ] `compute_per_90_metrics_with_shrinkage()`
  - [ ] `calibrate_position_priors()`
  - [ ] Extend `player_season_aggregates()`

- [ ] Extended analytics.py:
  - [ ] `compute_team_attacking_defensive_strength()`
  - [ ] `compute_home_away_ratios()`
  - [ ] `compute_form_trends()`
  - [ ] Extend `fixture_run()`

- [ ] Extended models.py:
  - [ ] `build_expected_value_model()`
  - [ ] `compute_recommendation_score_v2()`
  - [ ] `generate_recommendation_explanations()`

- [ ] New test files (7 test modules, >100 test cases)

### Testing & validation
- [ ] All unit tests pass (95%+ coverage)
- [ ] Integration tests pass (old + new systems)
- [ ] Backtest reproduces Phase 5 results (hit rate 58%, ρ>0.54)
- [ ] Staging environment validated
- [ ] Performance benchmarks met (<5% slowdown)
- [ ] Data quality checks pass (0% NULLs)

### Deployment
- [ ] Code merged to main branch
- [ ] Feature flags configured
- [ ] Database migrations executed (non-blocking)
- [ ] Monitoring dashboards active
- [ ] Rollback plan tested
- [ ] Team notified of changes

### Post-deployment
- [ ] Monitor first week (shadow mode)
- [ ] Validate recommendations match backtest expectations
- [ ] Check recommendation diversity (should differ from v1)
- [ ] Gather feedback from stakeholders
- [ ] Begin graduated rollout (10% → 50% → 100%)

---

## SECTION 10: QUICK REFERENCE: FILE CHANGES SUMMARY

| File | Change Type | Complexity | Est. Lines | Status |
|------|------------|-----------|-----------|--------|
| transformation/historical.py | Extended | Medium | +300 | Phase 7 |
| transformation/analytics.py | Extended | Medium | +250 | Phase 7 |
| models.py | Extended | High | +500 | Phase 7 |
| test_transformation_extended.py | New | Medium | ~200 | Phase 7 |
| test_analytics_extended.py | New | Medium | ~200 | Phase 7 |
| test_models_v2.py | New | High | ~400 | Phase 7 |
| test_integration_old_vs_new.py | New | High | ~300 | Phase 7 |
| config/config.yaml | Extended | Low | +20 | Phase 7 |
| Database schema | Migration | Low | N/A | Phase 7 |

**Total Estimated New Code:** ~1700 lines (including tests)

---

**Phase 6 Complete**

Ready for Phase 7: Code Implementation?

