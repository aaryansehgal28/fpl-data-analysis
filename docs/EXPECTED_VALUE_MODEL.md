# Phase 4: Expected-Value Model Design

**Date:** 2026-07-27  
**Scope:** Design component-wise decomposition of FPL points and expected-value framework  
**Output:** Mathematical model for estimating future points from underlying statistics  

---

## OVERVIEW

Instead of predicting total_points directly, we decompose FPL scoring into independent components:

```
Expected Future FPL Points = E[Goals] + E[Assists] + E[Clean Sheets] + E[Bonus] + E[Appearances]
```

Each component is estimated from underlying statistics (not from historical total_points).

**Benefits:**
- Position-aware scoring (forwards get different points for goals than defenders)
- Interpretable: can explain "why" a player is recommended (what drives their score)
- Resilient to luck: separates "sustainable" underlying performance from bonus noise
- Forward-looking: uses current season trends, fixture context, recent form

---

## SECTION 1: FPL SCORING SYSTEM

### 1.1 – Complete Point Structure

**Appearance & Time:**
```
1 point per match if selected (on bench or starting)
2 bonus points if player is captain
```

**Goals:**
```
GK:  0 points (impossible)
DEF: 5 points per goal
MID: 5 points per goal
FWD: 4 points per goal
```

**Assists (as of 2024 rule change):**
```
ALL: 1 point per assist (unified across positions)
```

**Defensive (Clean Sheet):**
```
GK:  4 points per CS (team concedes 0 goals)
DEF: 4 points per CS
MID: 1 point per CS (rarer for midfielders)
FWD: 0 points (never awarded)
```

**Other Actions:**
```
Saves (GK only):      0.33 points per save (capped at 3 per GW)
Penalty saved:        5 points
Penalty missed:      -2 points
Yellow card:         -1 point
Red card:            -2 points
Own goal:            -2 points
Goals conceded (DEF/GK/MID): -0.5 points per goal (capped at -5 for GK/DEF, uncapped for MID)
```

**Bonus Points:**
```
1, 2, or 3 bonus points based on BPS (Bonus Points System) rank in match
Awarded to top 3 performers (usually GWs with clear winners)
Range: 0–3 per match for a player (if they finish in top 3)
```

### 1.2 – Position-Specific Scoring Summary

| Stat | GK | DEF | MID | FWD |
|------|-----|-----|-----|-----|
| Goal | 0 | 5 | 5 | 4 |
| Assist | 1 | 1 | 1 | 1 |
| Clean Sheet | 4 | 4 | 1 | 0 |
| Appearance | 1 | 1 | 1 | 1 |
| Save (0.33) | 0.33 | — | — | — |
| Bonus (avg 1.5) | 1.5 | 1.5 | 1.5 | 1.5 |

---

## SECTION 2: COMPONENT DECOMPOSITION STRATEGY

### 2.1 – Model Scope: What to Include

**Include in component model:**
1. Goals (primary driver)
2. Assists (primary driver)
3. Clean Sheets (primary for DEF/GK; secondary for MID)
4. Bonus Points (outcome of BPS, but modelable)
5. Appearance Points (trivial; always 1 if selected)
6. Savings/Penalties (tertiary; rare, but significant when occur)
7. Discipline (yellow/red cards; rare, negative impact)

**Exclude / Model Separately:**
- Own goals (extremely rare; ignore)
- Goals conceded (implicit in clean sheet; don't double-count)

### 2.2 – Component Contribution to Expected Points

For a **midfielder in a typical gameweek:**

```
Expected Points per Match ≈ 
  E[Goals] × 5
  + E[Assists] × 1
  + E[Clean Sheets] × 1
  + E[Bonus Points]
  + E[Appearances] × 1
  + E[Yellow Cards] × (-1)
  + E[Red Cards] × (-2)

Typical distribution:
  - Goals: 5% chance of scoring; 0.05 × 5 = 0.25 points
  - Assists: 8% chance of assisting; 0.08 × 1 = 0.08 points
  - Clean Sheets: 40% team CS rate × 20% player gets CS bonus = 0.08 × 1 = 0.08 points
  - Bonus: 10% chance of top 3 BPS; avg 1.5 bonus = 0.15 points
  - Appearance: 90% play; 0.9 × 1 = 0.90 points
  - Discipline: 5% yellow; 0.05 × (-1) = -0.05 points
  
Total Expected: ~1.40 points per match
```

---

## SECTION 3: ESTIMATING COMPONENT PROBABILITIES

### 3.1 – Goals: Probability & Rate Estimation

**Component expected value:**
```
E[Goals per Match] = P(scores ≥1 goal) × E[goals | scores]
                   OR
E[Goals per Match] = goals_per_90 × (minutes_played / 90)
```

**Method 1: Historical Goal Rate (Preferred for MVP)**

```
Use posterior_goals_per_90 (from Phase 3 Bayesian shrinkage):
  posterior_goals_per_90_season = shrunk toward position prior
  posterior_goals_per_90_recent = shrunk recent 5-GW average
  
Blend season and recent:
  weight_season = 0.6  (historical stability)
  weight_recent = 0.4  (current form)
  
  effective_goals_per_90 = 0.6 × posterior_goals_season 
                         + 0.4 × posterior_goals_recent
  
Convert to per-match (assume 60 minutes average play):
  expected_goals_per_match = effective_goals_per_90 × (60 / 90)
                           = effective_goals_per_90 × 0.667
```

**Position-specific adjustments:**

```
FWD: Use posterior_goals_per_90 directly (typical play 60 min/match)
MID: Use posterior_goals_per_90 × adjustment_factor (0.7; midfielders less consistent)
DEF: Use posterior_goals_per_90 × adjustment_factor (0.5; set-piece specialists)
GK: 0 (impossible)
```

**Fixture adjustment:**

```
Opponent team defence strength plays role. If available:
  relative_opponent_defence = opponent_defence_strength / league_average
  expected_goals_adjusted = expected_goals × (2.0 / relative_opponent_defence)
  [1.0 = league average; 0.5 = weak defence (2× expected); 2.0 = strong defence (0.5× expected)]
  
Otherwise: Use FDR as proxy
  fdr_adjustment = 3.0 / fdr  [FDR 1=easy, 5=hard]
  expected_goals_adjusted = expected_goals × fdr_adjustment
```

**Recent form blending (already in posterior, but explicit for clarity):**

```
If attacking_form_trend > 0 (improving):
  form_multiplier = 1.0 + 0.1 × min(attacking_form_trend, 1.0)  [+0% to +10%]
Else if attacking_form_trend < 0 (declining):
  form_multiplier = 1.0 + 0.05 × attacking_form_trend  [−5% max]
  [More conservative on downside]
  
expected_goals_adjusted = expected_goals_adjusted × form_multiplier
```

### 3.2 – Assists: Probability & Rate Estimation

**Analogous to goals:**

```
effective_assists_per_90 = 0.6 × posterior_assists_season 
                         + 0.4 × posterior_assists_recent

expected_assists_per_match = effective_assists_per_90 × (minutes_per_match / 90)
```

**Position adjustments:**

```
MID: Use posterior_assists_per_90 directly (creative midfielders primary)
DEF: Use posterior_assists_per_90 × 0.8 (set-piece specialists; fewer opportunities)
FWD: Use posterior_assists_per_90 × 0.7 (less possession; fewer passing opportunities)
GK: 0 (extremely rare)
```

**Fixture adjustment:**

```
Stronger attacking team → more assist opportunities
opposite of goals adjustment (we want to reward cutters-up):
  assist_fixture_adjustment = fdr / 3.0  [FDR 1=high assists, 5=low assists]
  expected_assists_adjusted = expected_assists × assist_fixture_adjustment
```

### 3.3 – Clean Sheets: Probability Estimation

**Different approach than goals/assists (team-level outcome):**

```
Player's clean sheet points depend on:
  1. Team's clean sheet probability (team-level)
  2. Player's presence/reliability (player-level)
  3. Position (GK/DEF get CS; MID get 1 point; FWD get 0)
```

**Calculation:**

```
Step 1: Estimate team clean sheet probability
  team_cs_rate = COUNT(matches with 0 GA) / COUNT(matches played while player was in team)
  
  Better: Use team's defensive strength rating + opponent's attacking strength
  team_cs_probability = logistic(team_defence_strength - opponent_attack_strength + intercept)
  [intercept ≈ 0 for league average; calibrate from historical data]
  
  Simplified: Use recent CS rate (last 10 matches)
  team_cs_prob_recent = COUNT(CS in last 10) / 10

Step 2: Probability player is on pitch for CS
  p_on_pitch = P(plays full 90 minutes | match is CS)
             ≈ starts_ratio × (1 - substitution_rate)
  [Assume if starting, likely plays full 90 if team keeps CS]

Step 3: Player's personal CS rate (if plays, CS occurs)
  player_cs_per_90 = COUNT(CS achieved when player played) / effective_90s
  
  Combine with team CS rate:
  p_player_gets_cs = team_cs_prob_recent × p_on_pitch
  
  Expected CS points:
  For GK/DEF: expected_cs_points = p_player_gets_cs × 4
  For MID:    expected_cs_points = p_player_gets_cs × 1
  For FWD:    expected_cs_points = 0
```

**Example: Defender's Clean Sheet Probability**

```
Team defensive strength: 70 (above average)
Opponent attacking strength: 50 (average)
Recent CS rate: 35% (7 in last 20 matches)
Defender's starts_ratio: 0.85
Defender's substitution_rate: 0.05
Recent minutes trend: +0.10 (playing more)

Step 1: team_cs_prob ≈ 0.35 (use recent directly for simplicity)
Step 2: p_on_pitch = 0.85 × (1 - 0.05) = 0.81
Step 3: p_gets_cs = 0.35 × 0.81 = 0.28

Expected CS points per match = 0.28 × 4 = 1.12 points
```

**Fixture adjustment for team CS:**

```
Easy fixture (FDR=1) → higher team CS probability
  team_cs_fixture_adj = 1.5  [50% boost]
Hard fixture (FDR=5) → lower team CS probability
  team_cs_fixture_adj = 0.5  [50% reduction]

Linear interpolation:
  team_cs_fixture_adj = 1.0 + (3.0 - fdr) / 4.0  [range 0.5 to 1.0]
  
expected_cs_prob_adjusted = base_team_cs_prob × team_cs_fixture_adj
```

### 3.4 – Bonus Points: Probability & Amount

**Bonus is semi-predictable (driven by BPS):**

```
BPS rank top 3 → 3 bonus, 2 bonus, 1 bonus (or 3-3-2 sometimes)
BPS rank 4-10 → 0 bonus (usually)
```

**Estimation:**

```
Method 1: Historical Bonus Frequency (Simplest for MVP)
  bonus_finish_frequency = COUNT(top 3 finishes) / appearances
  average_bonus_when_earned = SUM(bonus) / COUNT(non-zero bonus)
  
  expected_bonus = bonus_finish_frequency × average_bonus_when_earned
  
  Example: 
    Midfielder finishes top-3 in 15% of matches
    When in top 3, average bonus = 1.8 (mostly 1 and 2 bonuses)
    expected_bonus = 0.15 × 1.8 = 0.27 points per match

Method 2: BPS-Based Estimation (More sophisticated)
  posterior_bps_per_90 = shrunk BPS metric from Phase 3
  
  Calibrate BPS-to-bonus relationship from historical data:
    # For each GW, find p(bonus | bps_score) by position
    # Build logistic model: p(bonus) = 1 / (1 + exp(-(bps - threshold) / slope))
  
  Threshold varies by:
    - Position (GK/DEF need higher BPS for bonus; FWD lower)
    - Match-level difficulty (more competitive → higher threshold)
    - GW (early season thresholds differ from late)
  
  expected_bonus = p(top3) × E[bonus | top3]
```

**Use BPS Method for better predictive power:**

```
For each position, calibrate from historical data:
  bonus_threshold_gk = 60  (GK needs 60+ BPS to likely get bonus)
  bonus_threshold_def = 50  (DEF easier)
  bonus_threshold_mid = 45  (MID)
  bonus_threshold_fwd = 40  (FWD hardest to get bonus)

Logistic model (per position):
  p_bonus = 1 / (1 + exp(-(expected_bps_per_match - threshold) / slope))
  [slope ≈ 10 for all positions; calibrate]

Example FWD:
  posterior_bps_per_90 = 25
  expected_bps_per_match ≈ 25 × (60 / 90) = 16.7
  threshold_fwd = 40
  p_bonus = 1 / (1 + exp(-(16.7 - 40) / 10))
          = 1 / (1 + exp(2.33))
          = 1 / 11.3
          = 0.09  [9% chance]
  
  expected_bonus = 0.09 × 1.5 = 0.135 points per match
```

**Fixture adjustment for bonus:**

```
Fixture difficulty affects BPS distribution:
  Easy match (FDR=1) → more blowouts → fewer close BPS battles → fewer 3-point bonuses
  Hard match (FDR=5) → tight match → more competitive BPS → better bonus opportunities
  
fixture_bonus_adjustment = 0.8 + (fdr - 3) / 2.5  [range 0.6 to 1.2]
  
expected_bonus_adjusted = expected_bonus × fixture_bonus_adjustment
```

### 3.5 – Appearance Points (Trivial Component)

```
Appearance = 1 point if selected (on bench or starting)

E[Appearance Points] = P(selected and plays) × 1
                     ≈ (1 - P(omitted from squad)) × availability_factor
                     ≈ ownership_percent / 100 × availability_factor
                     ≈ 0.9 to 1.0 for typical starters
```

**Simple estimate:**

```
If player is fit and available:
  expected_appearance = 1.0 (certain to be selected)
Else if doubtful:
  expected_appearance = chance_of_playing_next_round / 100
Else if injured:
  expected_appearance = 0.0
```

### 3.6 – Discipline Points (Rare Negative Component)

```
Yellow cards: -1 point each
Red cards: -2 points each

Historical rates (per season):
  Avg player: ~6 yellow cards / 38 matches = 0.16 per match = −0.16 points
  Avg GK: ~1 per season = 0.03 per match = −0.03 points
  Defensive players: higher (0.20+ per match)
  Attacking players: lower (0.10 per match)

E[Discipline Points] ≈ -0.10 to -0.15 for typical player
```

**For MVP: Ignore discipline** (minor impact; add in Phase 2 if needed)

```
expected_discipline = 0  # Simplified
```

---

## SECTION 4: POSITION-SPECIFIC EXPECTED POINTS MODELS

### 4.1 – Expected Points Model: Goalkeeper

```
E[Points per Match | GK] 
  = E[Goals] × 0  [impossible]
  + E[Assists] × 1  [extremely rare, set = 0]
  + E[Clean Sheets] × 4
  + E[Saves] × 0.33  [capped at 3 per match]
  + E[Bonus]
  + E[Appearances]
  - E[Discipline]
```

**Dominant components: Clean Sheets + Saves + Bonus**

```
Typical GK expected points per match:
  CS: P(CS) × 4 = 0.35 × 4 = 1.40
  Saves: 8 saves × 0.33 = 2.64  [capped at 3, so ≈ 3.0]
  Bonus: P(top3) × 1.5 = 0.15 × 1.5 = 0.23
  Appearance: 1.0
  Discipline: -0.05
  
Total: 1.40 + 3.0 + 0.23 + 1.0 - 0.05 = 5.58 points per match
```

**Simplified MVP Model (ignore saves, discipline):**

```
E[Points | GK] ≈ E[CS Points] + E[Bonus] + 1.0
                = expected_cs_points + expected_bonus + 1.0
```

---

### 4.2 – Expected Points Model: Defender

```
E[Points per Match | DEF]
  = E[Goals] × 5
  + E[Assists] × 1
  + E[Clean Sheets] × 4
  + E[Bonus]
  + E[Appearances]
  - E[Discipline]
```

**Dominant components: Clean Sheets + Bonus; secondary Goals/Assists**

```
Typical DEF expected points per match:
  CS: 0.25 × 4 = 1.00
  Goals: 0.02 × 5 = 0.10
  Assists: 0.03 × 1 = 0.03
  Bonus: 0.12 × 1.5 = 0.18
  Appearance: 1.0
  Discipline: -0.08
  
Total: 1.00 + 0.10 + 0.03 + 0.18 + 1.0 - 0.08 = 2.23 points per match
```

**MVP Model:**

```
E[Points | DEF] ≈ expected_cs_points + 5 × expected_goals 
                 + 1 × expected_assists + expected_bonus + 1.0
```

---

### 4.3 – Expected Points Model: Midfielder

```
E[Points per Match | MID]
  = E[Goals] × 5
  + E[Assists] × 1
  + E[Clean Sheets] × 1
  + E[Bonus]
  + E[Appearances]
  - E[Discipline]
```

**Dominant components: Goals + Assists + Bonus**

```
Typical MID expected points per match:
  Goals: 0.10 × 5 = 0.50
  Assists: 0.08 × 1 = 0.08
  CS: 0.08 × 1 = 0.08
  Bonus: 0.15 × 1.5 = 0.225
  Appearance: 1.0
  Discipline: -0.10
  
Total: 0.50 + 0.08 + 0.08 + 0.225 + 1.0 - 0.10 = 1.78 points per match
```

**MVP Model:**

```
E[Points | MID] ≈ 5 × expected_goals + 1 × expected_assists 
                 + 1 × expected_cs + expected_bonus + 1.0
```

---

### 4.4 – Expected Points Model: Forward

```
E[Points per Match | FWD]
  = E[Goals] × 4
  + E[Assists] × 1
  + E[Clean Sheets] × 0  [never awarded]
  + E[Bonus]
  + E[Appearances]
  - E[Discipline]
```

**Dominant components: Goals + Bonus; secondary Assists**

```
Typical FWD expected points per match:
  Goals: 0.45 × 4 = 1.80
  Assists: 0.04 × 1 = 0.04
  CS: 0
  Bonus: 0.12 × 1.5 = 0.18
  Appearance: 1.0
  Discipline: -0.12
  
Total: 1.80 + 0.04 + 0 + 0.18 + 1.0 - 0.12 = 2.90 points per match
```

**MVP Model:**

```
E[Points | FWD] ≈ 4 × expected_goals + 1 × expected_assists 
                 + expected_bonus + 1.0
```

---

## SECTION 5: MULTI-MATCH HORIZON EXPANSION

### 5.1 – From Per-Match to N-Gameweek Expected Points

Standard horizon: 5 gameweeks (h=5)

```
E[Total Points, Next h GWs] = h × E[Points per Match] × (minutes_factor × availability_factor)
```

**Adjustments over multi-GW horizon:**

```
1. Fixture difficulty average (not individual)
   Use next_5_fixture_avg_fdr instead of next_1_fdr
   
2. Playing time reliability
   minutes_factor = 1.0 if reliability_score ≥ 0.35
                  = 0.82 if reliability_score < 0.35
   [Penalizes unproven players; risk of rotation/injury]
   
3. Availability
   availability_factor = chance_of_playing_next_round / 100
                      OR {1.0 if active, 0.55 if doubtful, 0.0 if unavailable}
   
   Apply per match or average over horizon?
   → Average over horizon: assume injury status is binary (either plays all 5 or none)
     availability_factor = 0.9 or 1.0 for "likely available"
                        = 0.5 for "uncertain"
                        = 0.1 for "probably out"
```

**Calculation for 5-GW horizon:**

```
expected_points_per_match (as computed in Section 4)
expected_points_5_gw = expected_points_per_match 
                     × 5 
                     × minutes_factor 
                     × availability_factor 
                     × fixture_difficulty_adjustment
                     × form_adjustment  [if significant trend]
```

**Example: Midfielder, 5-GW horizon**

```
Base expected points/match = 1.78
Horizon = 5
Minutes factor = 1.0 (established player, reliability 0.60)
Availability = 1.0 (fit, active)
Fixture difficulty adjustment = 0.95 (average FDR=3.2; slightly harder)
Form trend = 1.05 (attacking form improving 5%)

expected_points_5_gw = 1.78 × 5 × 1.0 × 1.0 × 0.95 × 1.05
                     = 8.85 points
```

### 5.2 – Multi-Horizon Projections (1 GW, 3 GW, 5 GW)

For each player, generate 3 projections:

```
h=1:  expected_points_1_gw = expected_per_match × 1 × factors
h=3:  expected_points_3_gw = expected_per_match × 3 × factors
h=5:  expected_points_5_gw = expected_per_match × 5 × factors
```

**Use in recommendation:**
- Primary horizon: h=5 (what most managers care about)
- Secondary horizons: h=1, h=3 (for short-term, differential, risk-averse users)

---

## SECTION 6: UNCERTAINTY & CONFIDENCE INTERVALS

### 6.1 – Sources of Uncertainty

Each component has uncertainty:

```
1. Parameter uncertainty: posterior_std from Bayesian shrinkage
2. Performance volatility: variance in underlying stats
3. Fixture uncertainty: opponent strength estimates may be wrong
4. Injury/availability: injury news can change
5. Match-level luck: variance in bonus, yellow cards, own goals
```

### 6.2 – Uncertainty Propagation

For each component, compute posterior standard error:

```
Component: E[Goals per Match]
  uncertainty_goals = posterior_goals_std × (minutes_per_match / 90)
                    = 0.08 × 0.67 = 0.054 goals
  
Component: E[CS Points per Match]
  uncertainty_cs = sqrt(p × (1-p)) × 4  [binomial variance]
                 = sqrt(0.28 × 0.72) × 4 = 1.73 points

Component: E[Bonus Points]
  uncertainty_bonus = sqrt(p_bonus × E[bonus]^2) ≈ 0.3 points
```

**Total uncertainty (assume components independent):**

```
total_uncertainty_per_match = sqrt(SUM(uncertainty_component^2))

Example (midfielder):
  uncertainty_goals = 5 × 0.054 = 0.27
  uncertainty_assists = 1 × 0.06 = 0.06
  uncertainty_cs = 0.15
  uncertainty_bonus = 0.30
  uncertainty_discipline = 0.05
  
  total_uncertainty = sqrt(0.27^2 + 0.06^2 + 0.15^2 + 0.30^2 + 0.05^2)
                    = sqrt(0.073 + 0.004 + 0.023 + 0.090 + 0.003)
                    = sqrt(0.193)
                    = 0.44 points per match
```

**Expand to h-gameweek horizon:**

```
total_uncertainty_h_gw = total_uncertainty_per_match × sqrt(h)
                       = 0.44 × sqrt(5)
                       = 0.44 × 2.24
                       = 0.98 points over 5 GWs
```

**Confidence intervals:**

```
95% CI = expected ± 1.96 × uncertainty

Example: Midfielder with expected_5gw = 8.85, uncertainty = 0.98
  lower_95 = 8.85 - 1.96 × 0.98 = 6.92
  upper_95 = 8.85 + 1.96 × 0.98 = 10.78
```

---

## SECTION 7: POSITION-SPECIFIC ADJUSTMENTS

### 7.1 – Calibration by Position

The models in Section 4 use position-specific point multipliers (goals worth 5 for DEF but 4 for FWD). 

**Verification: Aggregate expected points should roughly match historical averages.**

```
From historical data (5-season average):
  GK average points per GW: 4.0–4.5
  DEF average points per GW: 1.8–2.2
  MID average points per GW: 1.2–1.5
  FWD average points per GW: 1.8–2.3

Our models should produce:
  GK: ≈ 4.3 per match × 0.85 availability × 0.8 playing time = 2.93 per GW average
  [With selection rate ~0.7, expected points = 2.05 per GW listed]
  [vs historical 4.3 per GW for selected GK]
  
  → Model seems reasonable (selected GK get ~4.3; unselected get 0; average ~2.05)
```

### 7.2 – Adjustments for Role/Formation

If formation data is available (Tier 3 feature):

```
Example: Team switches 4-3-3 → 5-2-3
  - Defender roles increase (3 → 5)
  - Midfielder roles decrease (3 → 2)
  - Forward roles stay (3 → 3)

Implication:
  - Defenders get more minutes, more CS opportunity
  - Midfielders get fewer minutes, less playing time
  
Adjustment:
  expected_points_new_formation = expected_points × (new_minutes / old_minutes)
```

**For MVP: Skip formation tracking** (Tier 3 feature; deprioritize)

---

## SECTION 8: COMBINING INTO FINAL RECOMMENDATION SCORE

### 8.1 – From Expected Points to Recommendation Score

Once we have expected_points_5gw (h=5 horizon):

```
Base recommendation score = expected_points_5gw
```

**But we also want to account for:**
- Value (points per pound)
- Uncertainty (risk)
- Sustainability (not just luck)

**Multi-component recommendation score:**

```
recommendation_score = 
  (expected_points_5gw / baseline_points_position) × position_weighting
  + (expected_points_5gw / current_price_millions) × value_weighting
  - uncertainty_5gw × risk_penalty_factor
  + sustainability_bonus  [if high-confidence underlying stats]
```

**Simpler approach (Recommended for MVP):**

```
recommendation_score = expected_points_5gw 
                     + 0.04 × posterior_value_per_pound  [value bonus, small]
                     - 0.5 × uncertainty_5gw  [risk penalty]
```

This mirrors the current architecture but replaces `posterior_ppm` with our new component-based expected_points.

### 8.2 – Decomposable Components for Explainability

Return a breakdown showing:

```
Player: Erling Haaland (FWD)
─────────────────────────────

Component Breakdown (Next 5 GWs):
  Expected Goals: 2.1 × 4 points = 8.4 pts
  Expected Assists: 0.3 × 1 point = 0.3 pts
  Expected Bonus: 0.8 pts
  Expected Appearance: 5.0 pts
  ─────────────────────────────
  Total Expected Points: 14.5 pts
  
Uncertainty: ±1.2 pts (90% CI: 12.1–16.9)

Value Score: 14.5 / 0.9 £m = 16.1 pts/£m

Final Recommendation Score: 14.5 + 0.5 (value bonus) - 0.6 (uncertainty penalty) = 14.4

Drivers:
  ✓ High goal-scoring rate (posterior 0.42/90, +0.05 recent trend)
  ✓ Fixtures slightly easier than average (FDR 3.1 vs 3.0)
  ✓ High reliability (0.65; established player)
  
Risks:
  ⚠ Recent form trend slightly down (-0.05; minor)
  ⚠ Bonus volatility (+/-0.5 points)
```

---

## SECTION 9: VALIDATION & BACKTESTING

### 9.1 – Calibration Metrics

At end of season, measure:

```
1. Calibration: Do players with expected_points=10 actually average ~10 realized?
   Plot: E[expected] vs observed, check for bias
   
2. Ranking correlation: Do higher expected match higher realized?
   Spearman rank correlation should be 0.50–0.70
   
3. By-component accuracy:
   - Do goal-scoring predictions match realized goals? (should be yes)
   - Do CS predictions match realized CS? (harder; team effects large)
   - Do bonus predictions match realized bonus? (should be moderate)
   
4. Uncertainty calibration:
   Do 95% CIs contain actual outcomes 95% of the time?
   Plot observed within bands by predicted uncertainty
```

### 9.2 – Backtest Protocol

```
For each historical gameweek t = 1 to 38:
  
  1. Use data from GWs 1 to t-1 only (strictly before)
  2. Compute all features for all players (Phase 3)
  3. Compute expected points for each player (Phase 4)
  4. Generate recommendations for GW t
  5. Observe actual realized points in GW t
  6. Compare expected vs. realized
  7. Track hit rate: % of recommended players who outperform median
```

**Leakage prevention:**
```
✓ All feature data strictly from GWs 1 to t-1
✓ Fixture data for GW t (forward-looking, allowed)
✓ Bootstrap snapshot from deadline before GW t
✓ No knowledge of actual GW t results until after computing score
```

---

## SECTION 10: INTEGRATION WITH RECOMMENDATION SYSTEM

### 10.1 – Usage in optimiser (Squad Selection)

Current `optimise.py` uses:
```
recommendation_score as primary objective
projected_points as secondary
posterior_ppm as value measure
```

New architecture will use:
```
expected_points_component_sum as primary objective
uncertainty as penalty
posterior_points_per_pound as value measure
```

**MIP Objective (multi-objective):**

```
Maximize:
  SUM(recommendation_score × selected_player)
  - SUM(uncertainty × selected_player) × 0.5  [risk penalty]
  + SUM(points_per_pound × selected_player) × 0.1  [value bonus]
```

### 10.2 – Usage in Analytics & Dashboards

Each player recommendation output includes:

```
player_expected_value_output = {
  "player_id": 123,
  "recommendation_score": 14.2,
  "expected_points_5_gw": 14.5,
  "expected_points_1_gw": 2.9,
  "expected_points_3_gw": 8.7,
  
  # Component breakdown
  "expected_goals_points": 8.4,
  "expected_assists_points": 0.3,
  "expected_cs_points": 2.1,
  "expected_bonus_points": 1.2,
  "expected_appearance_points": 5.0,
  
  # Uncertainty
  "uncertainty_5_gw": 1.1,
  "confidence_lower": 12.4,
  "confidence_upper": 16.6,
  
  # Quality metrics
  "reliability_score": 0.65,
  "posterior_goals_per_90": 0.42,
  "posterior_goals_std": 0.08,
  "posterior_assists_per_90": 0.04,
  
  # Context
  "fixture_difficulty": 3.1,
  "availability_factor": 1.0,
  "minutes_factor": 1.0,
  "position": "FWD",
  "team": "MUN",
  "price": 9.0,
  "points_per_million": 1.61,
}
```

---

## SUMMARY: EXPECTED-VALUE MODEL FRAMEWORK

| Component | GK | DEF | MID | FWD | Calculation |
|-----------|-----|-----|-----|-----|-------------|
| Goals | 0 | 5 | 5 | 4 | posterior_goals_per_90 × (min/90) × fixtures |
| Assists | 1 | 1 | 1 | 1 | posterior_assists_per_90 × (min/90) × fixtures |
| Clean Sheets | 4 | 4 | 1 | 0 | P(team_cs) × P(plays) × multiplier |
| Bonus | 1.5 | 1.5 | 1.5 | 1.5 | P(top3_bps) × E[bonus \| top3] |
| Appearance | 1 | 1 | 1 | 1 | availability_factor × selection_prob |
| Discipline | -1.5 | -1.5 | -1.5 | -1.5 | Ignore for MVP |

**Per-match expected points (typical):**
- GK: 5.6 pts
- DEF: 2.2 pts
- MID: 1.8 pts
- FWD: 2.9 pts

**5-GW horizon (after playing time/availability adjustments):**
- GK: 14.0–18.0 pts
- DEF: 5.5–9.0 pts
- MID: 4.5–7.5 pts
- FWD: 7.0–12.0 pts

---

## IMPLEMENTATION CHECKLIST

Before coding, verify:

- [ ] FPL scoring system correctly documented (point multipliers per position)
- [ ] Component decomposition strategy finalized (which stats → which points)
- [ ] Per-component estimation methods chosen (goal rate, CS probability, etc.)
- [ ] Position-specific models specified (4 separate models for GK/DEF/MID/FWD)
- [ ] Fixture adjustments formalized (FDR to probability scaling)
- [ ] Multi-horizon calculations defined (1, 3, 5 GW)
- [ ] Uncertainty propagation method chosen (component-wise, then total)
- [ ] Backtesting protocol specified (leakage-free, metrics defined)
- [ ] Integration points with recommendation system identified
- [ ] Output schema finalized (what fields to return)
- [ ] Validation metrics chosen (calibration, ranking correlation)

---

**Phase 4 Complete**

Ready for Phase 5: Backtesting Framework Design

