# FPL Recommendation Score Redesign: Complete Audit & Architecture

**Date:** 2026-07-27  
**Scope:** Critical redesign of `recommendation_score` to use underlying performance metrics rather than aggregate total_points  
**Status:** Audit phase (no code changes yet)

---

## SECTION 1: CURRENT ARCHITECTURE AUDIT

### 1.1 Data Ingestion (gameweek-level)

**18 fields captured per player per gameweek** (`fact_player_gameweek`):

| Category | Fields | Purpose |
|----------|--------|---------|
| **Time** | `minutes` | Playing time |
| **Attacking** | `goals_scored`, `assists` | Primary attacking output |
| **Defending** | `clean_sheets`, `goals_conceded`, `own_goals` | Defensive outcome |
| **Penalty** | `penalties_saved`, `penalties_missed` | Specialist outcome |
| **Bonus** | `bonus`, `bps` | FPL bonus system output |
| **Underlying** | `influence`, `creativity`, `threat`, `ict_index` | Opta-derived performance proxies |
| **Saves** | `saves` | Goalkeeper-specific |
| **Discipline** | `yellow_cards`, `red_cards` | Negative outcomes |
| **Aggregate** | `total_points` | FPL's computed outcome (1 point per match + bonuses) |
| **Snapshots** | `value`, `selected`, `transfers_in`, `transfers_out` | Intra-gameweek flows/snapshots |

**Key observation:** 18 fields collected; downstream only uses `total_points`.

### 1.2 Season Aggregation Pipeline

Function: `player_season_aggregates()` in `transformation/historical.py`

**Aggregation strategy:**
```
Input: fact_player_gameweek_historical (fixture-grain, double-GW rows already aggregated)
↓
Group by: season, identity_key, position
↓
Aggregations (SUM for event stats):
  - total_points
  - total_minutes  
  - goals_scored
  - assists
  - clean_sheets
  - bonus
  - total_appearances (count of gameweeks)
  - starts (count of starting appearances)
  
Derived:
  - effective_90s = total_minutes / 90
  - points_per_game = total_points / total_appearances
  - season_ppm = total_points / average_gameweek_price_million    ← CENTRAL METRIC
  - points_per_90 = total_points / effective_90s

Output: fact_player_season
```

**Critical design choice:** The `season_ppm` metric is **value-based**, not performance-based. It conflates:
- Actual playing time
- Quality of performance (through total_points)
- FPL scoring system outcomes (bonuses, conversion rates)
- Price movements (denominator)

---

### 1.3 Bayesian Shrinkage (Historical Posterior)

Function: `bayesian_player_value()` in `models.py`

**Purpose:** Estimate a player's true underlying PPM (points-per-million) by shrinking noisy historical observations toward a position prior.

**Process:**

1. **Recency weighting:**
   ```
   season_weight = exp(-0.45 × years_ago)   # Decay older seasons
   weighted_90s = effective_90s × season_weight
   ```
   
2. **Winsorization (2–98% clipping):**
   ```
   capped_ppm = season_ppm.clip(quantile(0.02), quantile(0.98)) by position
   ```
   Rationale: Remove extreme outliers that might define a position prior due to small sample.

3. **Position prior:**
   ```
   position_prior_ppm = avg(capped_ppm, weights=weighted_90s) by position
   position_ppm_std = std(capped_ppm) by position
   ```

4. **Posterior shrinkage (Empirical Bayes Normal-mean model):**
   ```
   posterior_ppm = (prior_strength × position_prior_ppm + Σ(weighted_90s × capped_ppm))
                   / (prior_strength + Σ(weighted_90s))
   ```
   where `prior_strength = 12` (configured equivalent 90s).

   **Interpretation:** A player with 10 weighted 90s is pulled 45% toward the position prior; with 100 weighted 90s, only 11%.

5. **Uncertainty quantification:**
   ```
   posterior_uncertainty = position_ppm_std / sqrt(prior_strength + Σ(weighted_90s))
   ```
   Captures both variance in PPM and sample size.

6. **Reliability score:**
   ```
   reliability_score = Σ(weighted_90s) / (prior_strength + Σ(weighted_90s))
   ```
   Clip to [0, 1]. Ranges from ~0.5 for rookies to ~0.9+ for established players.

**Output columns:**
- `posterior_ppm` – shrunk historical value
- `posterior_uncertainty` – std error
- `reliability_score` – confidence in posterior estimate
- `effective_weighted_90s` – cumulative exposure
- `seasons_observed` – count of seasons with data

**Critical issue:** This step uses ONLY `season_ppm` and `effective_90s`. All 18 raw statistics are discarded.

---

### 1.4 Recommendation Score Calculation

Function: `build_recommendations()` in `models.py`

**Inputs merged:**
- Current bootstrap snapshot: `web_name`, `current_price`, `status`, `team_id`, `position_id`
- Availability: `chance_of_playing_next_round`
- Transfer momentum: `transfers_in_event`, `transfers_out_event`
- Ownership: `ownership_percent`
- Fixture data: `next_{1,3,5}_fixture_average_difficulty`
- Recent form (if provided): mean(`total_points`), mean(`minutes`) per player over recent GWs

**Calculation chain:**

1. **Fixture adjustment:**
   ```
   fixture_adjustment = (3.0 / fixture_difficulty).clip(0.7, 1.3)
   ```
   Official FDR ranges 1–5, so this scales: easy (1) → 3x, hard (5) → 0.6x

2. **Availability factor:**
   ```
   availability_factor = chance_of_playing_next_round / 100
                      IF chance is known
                      ELSE: 1.0 if status=="a" (active), 0.55 if injured/unavailable
   ```

3. **Minutes factor (reliability-based penalty):**
   ```
   minutes_factor = 1.0 if reliability_score >= 0.35
                  = 0.82 if reliability_score < 0.35
   ```
   Rationale: Penalize young/untested players slightly.

4. **Form factor (recent total_points based):**
   ```
   mean_recent_points = mean(total_points over last N GWs)  [default: 0.0 if no recent data]
   form_adjustment = mean_recent_points - 4                 [clip to -2, +2]
   form_factor = 1.0 + form_adjustment × 0.03               [range: 0.94 to 1.06]
   ```

5. **Baseline expected points:**
   ```
   baseline_expected_points = posterior_ppm × (current_price / 10) / 38
   ```
   Rationale: PPM is season-scale; convert to per-GW by ÷38, scale to player's price.

6. **Projected points (h-gameweek horizon, e.g., h=5):**
   ```
   projected_points = baseline_expected_points 
                    × h 
                    × minutes_factor 
                    × availability_factor 
                    × fixture_adjustment 
                    × form_factor
   ```

7. **Projection uncertainty:**
   ```
   projection_uncertainty = posterior_uncertainty 
                          × (current_price / 10) 
                          × h 
                          / 38
   ```

8. **Final recommendation score:**
   ```
   recommendation_score = projected_points 
                        + 0.04 × posterior_ppm 
                        - 0.5 × projection_uncertainty
   ```

**Interpretation:**
- Base: `projected_points` (expected value over horizon)
- Bonus: `+0.04 × posterior_ppm` (reward high career PPM)
- Penalty: `−0.5 × projection_uncertainty` (penalize risk)

---

### 1.5 Projection Variants

For each player, the system also generates multi-horizon projections (h=1, 3, 5 GWs):

```
projected_points_per_million = projected_points / (current_price_tenths / 10)
confidence_lower = max(0, projected_points - 1.96 × projection_uncertainty × h / horizon)
confidence_upper = projected_points + 1.96 × projection_uncertainty × h / horizon
```

These are populated into `player_projection` table for users to explore different risk profiles.

---

## SECTION 2: REDUNDANCY & DOUBLE-COUNTING ANALYSIS

### 2.1 Information Flow Diagram

```
Raw gameweek-level stats (18 fields)
    ↓
fact_player_gameweek (archive: goals, assists, clean_sheets, bonus, total_points)
    ↓
player_season_aggregates (SUM: total_points → season_ppm)
    ↓
bayesian_player_value (season_ppm → posterior_ppm)
    ↓
build_recommendations:
    ├─ baseline_expected_points = posterior_ppm × price / 38
    ├─ projected_points = baseline × h × factors
    ├─ recommendation_score = projected_points + 0.04×posterior_ppm - 0.5×uncertainty
    └─ Output used by optimiser
```

### 2.2 Identified Redundancies

**Issue 1: Total points appears multiple times**
- `season_ppm = total_points / price` (contains total_points)
- `posterior_ppm` = shrunk(season_ppm) (inherits total_points signal)
- `baseline_expected_points` = posterior_ppm × price (total_points signal reconstructed)
- `recommendation_score += 0.04 × posterior_ppm` (0.04 × total_points / price bonus added explicitly)

**Risk:** The 0.04×posterior_ppm term is a second dose of the total_points signal already in projected_points.

---

**Issue 2: Recent form only uses recent total_points**
- `recent_form = mean(total_points over last 5 GWs)`
- Applied as a ±6% adjustment to baseline (via `form_factor`)
- No distinction between: goals-driven, assist-driven, bonus-driven, or conversion-driven form

**Risk:** A player on a bonus streak looks equally "in form" as a player with improved underlying threat.

---

**Issue 3: Posterior uncertainty is based on historical PPM variance**
- `posterior_uncertainty = position_ppm_std / sqrt(sample_size)`
- This captures volatility in *points per pound*, not underlying performance volatility
- A player with variable goal-scoring (but consistent PPM due to price) appears less uncertain than they should

**Risk:** Systematic underestimation of uncertainty for high-variance performers.

---

**Issue 4: Reliability score is based on playing time only**
- `reliability_score = weighted_90s / (prior + weighted_90s)`
- Confounds: availability, fitness, tactical deployment, manager preferences
- No connection to actual injury status, suspension risk, or position in squad pecking order

**Risk:** A rotated player looks reliable if they have historical 90s.

---

**Issue 5: Individual stats never used as independent predictors**
- `goals_scored`, `assists`, `clean_sheets`, `influence`, `creativity`, `threat`, `ict_index`, `bps`
- All are collected, archived, but never extracted for analysis
- Everything is collapsed into `total_points`

**Risk:** Model cannot distinguish performance drivers; cannot detect substitution patterns, role changes, or conversion deviations.

---

**Issue 6: Bonus treatment is implicit**
- `bonus` is summed into `total_points` aggregation
- But `bonus` in FPL is discrete (0, 3, 2, 1 points) and driven by BPS (Bonus Points System)
- Not modeled separately, so cannot predict bonus probability going forward

**Risk:** Models learned patterns in current season's bonus distribution; these may not persist.

---

**Issue 7: Form factor is linear and small**
- Uses mean recent total_points, adjusts by ±3% per point difference from 4
- No distinction between: form reverting to mean vs. true change in role/opportunity

**Risk:** Overly conservative; cannot detect step-change in performance.

---

### 2.3 Feature Redundancy Matrix

| Feature | Depends On | Used In | Redundancy Risk |
|---------|-----------|---------|-----------------|
| `season_ppm` | `total_points` | Posterior shrinkage | HIGH – total_points is input |
| `posterior_ppm` | `season_ppm`, `effective_90s` | All downstream scoring | HIGH – contains total_points |
| `posterior_uncertainty` | `position_ppm_std` | Projection uncertainty | MEDIUM – based on historical PPM variance |
| `reliability_score` | `effective_weighted_90s` | `minutes_factor` | MEDIUM – conflates playing time with reliability |
| `recent_form` | `total_points` (recent GWs) | `form_factor` | HIGH – overlaps with posterior estimate |
| `form_factor` | `recent_form` | `projected_points` | MEDIUM – applied as small adjustment |
| `projected_points` | `baseline_expected_points`, factors | `recommendation_score` | LOW – derived correctly from inputs |
| `recommendation_score` | `projected_points`, `posterior_ppm`, `projection_uncertainty` | Output | MEDIUM – 0.04×posterior_ppm is redundant with projected_points |

### 2.4 Statistical Validity of Current Approach

**Positive aspects:**
- ✓ Empirical Bayes shrinkage is statistically sound (reduces overfitting)
- ✓ Recency weighting is reasonable (recent form more informative)
- ✓ Position-specific priors make sense (GK ≠ FWD)
- ✓ Winsorization avoids outlier influence
- ✓ Uncertainty quantification is present

**Problematic aspects:**
- ✗ Single bottleneck through `total_points` prevents granular analysis
- ✗ No separation of "sustainable" performance from "lucky" outcomes
- ✗ Recent form overlaps heavily with seasonal PPM; provides redundant signal
- ✗ Fixture adjustment uses official FDR only; no xG/xA context
- ✗ No position-specific scoring adjustments (defenders' clean sheets treated like forwards' goals)
- ✗ Bonus points never separately modeled
- ✗ Playing time factors (minutes_factor) are ad-hoc (0.82 vs 1.0), not data-driven

---

## SECTION 3: DATA AVAILABILITY AUDIT

### 3.1 Available Gameweek-Level Statistics

| Metric | FPL Field | Collected | Used | Notes |
|--------|-----------|-----------|------|-------|
| **Playing Time** | `minutes` | ✓ | ✓ (in effective_90s) | Per-match data; aggregates well |
| **Goals** | `goals_scored` | ✓ | ✗ | Available; never used |
| **Assists** | `assists` | ✓ | ✗ | Available; never used |
| **Clean Sheets** | `clean_sheets` | ✓ | ✗ | Aggregated into total_points |
| **Goals Conceded** | `goals_conceded` | ✓ | ✗ | Available; never used |
| **Bonus Points** | `bonus` | ✓ | ✗ | Summed into total_points |
| **BPS** | `bps` | ✓ | ✗ | Bonus Points System score; not used |
| **Saves** | `saves` | ✓ | ✗ | GK-specific; never used |
| **Influence** | `influence` | ✓ | ✗ | Opta-derived; never used |
| **Creativity** | `creativity` | ✓ | ✗ | Opta-derived; never used |
| **Threat** | `threat` | ✓ | ✗ | Opta-derived; never used |
| **ICT Index** | `ict_index` | ✓ | ✗ | Composite (influence, creativity, threat) |
| **Discipline** | `yellow_cards`, `red_cards` | ✓ | ✗ | Never used; FPL impact is -1, -2 points |
| **Penalties** | `penalties_saved`, `penalties_missed` | ✓ | ✗ | Position-specific; never used |
| **Outcome** | `total_points` | ✓ | ✓ (via season_ppm) | FPL's aggregated daily output |

### 3.2 Available Bootstrap-Level Snapshots

| Metric | Field | Purpose | Updated | Used |
|--------|-------|---------|---------|------|
| Current price | `now_cost` | Player cost in tenths £m | Daily/deadlines | ✓ |
| Ownership | `selected_by_percent` | % of teams | Daily | ✓ (recent form only) |
| Status | `status` | "a"=active, "d"=doubt, "u"=unavailable, "n"=not available | Realtime | ✓ (availability_factor) |
| Chance of playing | `chance_of_playing_next_round` | 0–100 forecast | Pre-deadline | ✓ (availability_factor) |
| Transfers in/out (GW) | `transfers_in_event`, `transfers_out_event` | GW flows | Post-deadline | ✓ (transfer_momentum signal only) |
| Team | `team` → team_id | Player's current club | Transfers only | ✓ |
| Position | `element_type` → position_id | GK=1, DEF=2, MID=3, FWD=4 | Transfers only | ✓ |
| News | `news` | Injury/suspension text | Realtime | ✗ |
| News added | `news_added` | Timestamp | Realtime | ✗ |

### 3.3 Available Fixture Data

| Metric | Field | Purpose | Used |
|--------|-------|---------|------|
| Fixture difficulty (FDR) | `team_h_difficulty`, `team_a_difficulty` | FPL's 1–5 rating | ✓ (fixture_adjustment) |
| Opponent strength | Not in FPL API | Would need external xG | ✗ |
| Home/away status | `home_team_id`, `away_team_id` | Context | ✗ |
| Match result | `home_score`, `away_score` | Historical context | Partially |
| Kickoff time | `kickoff_time` | Scheduling | ✗ |

### 3.4 What's NOT Available from FPL API

- ❌ Expected goals (xG) / Expected assists (xA)
- ❌ Shot count / Shot accuracy
- ❌ Pass completion rates / Key passes
- ❌ Tackle/interception/block counts (only in Opta, summarized as influence)
- ❌ Set-piece involvement flags
- ❌ Penalty taker status
- ❌ Team expected goals against (xGA) – would need external data source
- ❌ Player-team assignment history (only current season's IDs)

---

## SECTION 4: CURRENT ARCHITECTURE FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────┐
│ INGESTION LAYER (per gameweek, per player)                          │
│ ├─ minutes, goals_scored, assists, clean_sheets, bonus              │
│ ├─ bps, saves, influence, creativity, threat, ict_index             │
│ ├─ yellow_cards, red_cards, penalties_*                             │
│ ├─ total_points (FPL's aggregation)                                 │
│ └─ snapshots: value, selected, transfers_in/out                     │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ AGGREGATION LAYER (per player, per season)                          │
│ ├─ SUM(total_points) → total_points                                 │
│ ├─ SUM(goals_scored) → goals [ARCHIVED, NOT USED]                   │
│ ├─ SUM(assists) → assists [ARCHIVED, NOT USED]                      │
│ ├─ SUM(clean_sheets) → clean_sheets [ARCHIVED, NOT USED]            │
│ ├─ SUM(bonus) → bonus [ARCHIVED, NOT USED]                          │
│ ├─ SUM(minutes) / 90 → effective_90s [USED]                         │
│ ├─ AVG(price) → average_price_million                               │
│ └─ total_points / average_price → season_ppm [BOTTLENECK]           │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
        ⚠️  ALL STATS EXCEPT season_ppm DISCARDED ⚠️
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ BAYESIAN SHRINKAGE LAYER                                            │
│ ├─ Input: season_ppm (per player, per season)                       │
│ ├─ Winsorize by position: season_ppm.clip(q2, q98)                  │
│ ├─ Position prior: weighted_avg(capped_ppm)                         │
│ ├─ Posterior: shrink toward prior using effective_90s               │
│ └─ Output: posterior_ppm, posterior_uncertainty, reliability_score  │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ CURRENT SNAPSHOT MERGE                                              │
│ ├─ Bootstrap: price, status, availability, ownership                │
│ ├─ Transfers: momentum (in - out)                                   │
│ ├─ Fixtures: FDR, next_{1,3,5} average difficulty                   │
│ └─ Recent form: mean(total_points), mean(minutes) if available      │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ADJUSTMENT FACTORS                                                  │
│ ├─ fixture_adjustment = 3.0 / FDR, clipped [0.7, 1.3]              │
│ ├─ availability_factor = chance_of_playing / 100 or {1.0, 0.55}     │
│ ├─ minutes_factor = {1.0 if reliability>=0.35 else 0.82}            │
│ └─ form_factor = 1.0 + (mean_recent_points - 4) × 0.03, clipped     │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ EXPECTED VALUE CALCULATION                                          │
│ ├─ baseline = posterior_ppm × (price/10) / 38                       │
│ ├─ projected = baseline × horizon × minutes × availability          │
│ │             × fixture_adjustment × form_factor                    │
│ └─ uncertainty = posterior_uncertainty × (price/10) × horizon / 38  │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ RECOMMENDATION SCORE (FINAL OUTPUT)                                 │
│                                                                      │
│ score = projected_points                                            │
│       + 0.04 × posterior_ppm  ← BONUS (redundant with projected)   │
│       - 0.5 × projection_uncertainty                               │
│                                                                      │
│ ⚠️  Multiple appearances of total_points signal                     │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: recommendation, player_projection tables                    │
│ Consumed by: optimiser.py (squad-building MIP)                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SECTION 5: LEAKAGE & TEMPORAL INTEGRITY ANALYSIS

### 5.1 Current Leakage Prevention Mechanisms

**Good practices:**
- ✓ Raw JSON envelopes are immutable; never overwritten
- ✓ `as_of_gameweek` parameter filters current_gw to only completed GWs
- ✓ Historical aggregation uses only data before current gameweek
- ✓ Bayesian posterior uses historical data only (no current-season GW-level detail mixed in)
- ✓ Recent form is optional and must be explicitly constructed from pre-deadline data

**Gaps:**
- ⚠️ Bootstrap snapshots (ownership, transfers_in/out, price, status) are taken at prediction time, not pre-deadline
  - Should store `captured_at` timestamp to reconstruct historical state
- ⚠️ Fixture difficulty (FDR) can change before kick-off; current code uses final value
  - For backtesting, must use pre-deadline FDR snapshot
- ⚠️ Recent form calculation does not validate `as_of_gameweek` boundary
  - Should clip to `gameweek_id < as_of_gameweek`, but implementation trusts caller

### 5.2 Backtesting Leakage Risks (Forward-Looking)

For proper historical backtesting, must ensure:

1. **Only pre-deadline data available:**
   - Bootstrap snapshots as of deadline (pre-decision)
   - Fixtures with pre-deadline FDR
   - No future gameweek results

2. **Consistent historical window:**
   - Season-level prior computed from data up to target GW
   - Recent form limited to target GW −1 or earlier
   - No look-ahead to rest-of-season statistics

3. **Position consistency:**
   - Player's position must be snapshot as-of target GW (positions can change mid-season)

---

## SECTION 6: DESIGN PRINCIPLES FOR REDESIGN

Based on the audit, the redesigned recommendation score should:

### 6.1 Core Principles

**P1. Separate "Performance" from "FPL Outcomes"**
- Model underlying performance drivers (goals, assists, threat, playing time)
- Separately model FPL system outcomes (bonuses, conversion, home/away effects)
- Do not assume FPL's total_points aggregation is the only truth signal

**P2. Avoid Double-Counting Information**
- If posterior_ppm contains total_points, do not add posterior_ppm again to recommendation_score
- If recent_form is recent_total_points, do not assume it's independent of seasonal posterior
- Separate "sustainable" signals from "noisy" signals via shrinkage

**P3. Make Components Interpretable**
- Each component of recommendation_score should answer a specific question:
  - "How good is this player?" → performance quality
  - "Will they play?" → playing time reliability
  - "Are the fixtures favorable?" → fixture opportunity
  - "Is the price fair?" → value assessment
  - "How stable is this estimate?" → sustainability/risk

**P4. Use Position-Specific Modeling**
- Forwards: goals and assists are dominant; clean sheets irrelevant
- Midfielders: balance of attacking and attacking returns; clean sheets secondary
- Defenders: clean sheets and defensive contribution; attacking returns rare
- Goalkeepers: clean sheets and saves; attacking returns impossible
- Do not apply identical scoring logic to all positions

**P5. Properly Handle Uncertainty**
- Separate "model uncertainty" (Bayesian) from "performance volatility" (underlying stats)
- Distinguish "proven players" (low uncertainty) from "rookie breakouts" (high uncertainty)
- Do not conflate reliability with safety

**P6. Ensure Leakage-Free Backtesting**
- All features must be constructible from pre-deadline data
- Historical aggregations must use only past data
- Fixtures, ownership, price snapshots must be timestamped

---

## SECTION 7: NEXT PHASES (REQUIRED BEFORE CODE CHANGES)

This audit is **Phase 1 (Complete)**. Before implementation, the following must be completed:

**Phase 2: Feature Engineering Architecture** (Required next)
- Specify 30–40 candidate features across 7 categories
- For each feature: definition, calculation, position-specific variants, redundancy risk
- Create feature redundancy heatmap
- Decide: include, exclude, or transform each

**Phase 3: Statistical Framework Design** (Required next)
- Design position-specific aggregation windows
- Specify per-90 normalizations
- Define Bayesian priors for each position and statistic
- Decide on shrinkage targets and amount
- Handle missing values, low-minute players, new transfers

**Phase 4: Expected-Value Model Design** (Required next)
- Decompose expected FPL points into components:
  - Expected goals × goal points (4)
  - Expected assists × assist points (5)
  - Expected clean sheets × CS points (4 DEF/GK, 1 MID, 0 FWD)
  - Expected bonus × bonus points (1–3)
  - Appearance point (1)
- For each, use underlying stats or composite proxies
- Decide: parametric model vs. lookup table vs. ML regression

**Phase 5: Backtesting Framework** (Required next)
- Pseudocode for walk-forward historical backtest
- Definition of evaluation metrics
- Benchmark models to compare against
- Leakage-checking protocol

**Phase 6: Implementation Plan** (Required next)
- Modular design with separate feature engineering, modeling, scoring layers
- Test strategy for each component
- Integration points with existing pipeline
- Deprecation strategy for old recommendation_score

---

## SECTION 8: KEY QUESTIONS BEFORE REDESIGN

1. **Should the new model be position-agnostic or position-specific?**
   - Trade-off: Simplicity vs. accuracy
   - Recommend: Position-specific submodels with shared framework

2. **Should we eliminate total_points entirely?**
   - Risk: Losing a strong, proven signal
   - Recommend: Properly shrink historical total_points, use it as one component among many

3. **How should bonus points be treated?**
   - Option A: Ignore; treat as part of total_points luck
   - Option B: Model BPS-to-bonus conversion separately
   - Recommend: Model as component; estimate 3-point probability and bonus-point frequency

4. **How should recent form be calculated?**
   - Option A: Mean recent total_points (current approach)
   - Option B: Underlying stats only (ignoring recent bonus/luck)
   - Option C: Bayesian update of seasonal posterior using recent performance
   - Recommend: Option C with caution (requires careful priors)

5. **Should we use external data (xG, xA, team strength)?**
   - Availability: Team strength in bootstrap; xG/xA not in FPL API
   - Risk: Adds external dependency; must backfill historically
   - Recommend: Start without; add if predictive value is demonstrated

6. **How many historical seasons should we use for estimation?**
   - Current: All available (2021–2026, 5 seasons)
   - Risk: Older seasons may have different scoring rules, player base, tactics
   - Recommend: Test 2–3 recent seasons vs. full history via backtesting

---

## SUMMARY TABLE: CURRENT vs. PROPOSED DIRECTION

| Aspect | Current | Proposed |
|--------|---------|----------|
| **Data used** | Only `total_points` (via season_ppm) | Goals, assists, clean_sheets, bonus, underlying stats |
| **Primary metric** | `season_ppm` (singular) | Component-wise performance model |
| **Position treatment** | Identical priors, different only by position | Full position-specific sub-models |
| **Bonus handling** | Summed into total_points | Separately modeled as component |
| **Recent form** | Mean recent total_points | Underlying stats + Bayesian update |
| **Uncertainty** | Based on historical PPM variance | Based on underlying performance volatility |
| **Recommendation score** | Projected_points + PPM bonus − uncertainty | Position-weighted component scores |
| **Interpretability** | Single number (opaque) | Decomposed components with explanations |
| **Backtesting** | No formal framework | Walk-forward with leakage checks |

---

**AUDIT COMPLETE**

Ready to proceed to Phase 2: Feature Engineering Architecture Design.

