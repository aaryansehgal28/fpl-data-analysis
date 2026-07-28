# Phase 2: Feature Engineering Architecture

**Date:** 2026-07-27  
**Scope:** Design 35+ candidate features across 7 categories with redundancy analysis  
**Output:** Feature inventory with position-specific variants and inclusion/exclusion recommendations  

---

## OVERVIEW: FEATURE CATEGORIES & DESIGN PHILOSOPHY

The redesigned recommendation system will evaluate player quality through **7 feature categories**:

1. **Playing-Time Reliability** (7 features)
2. **Underlying Attacking Performance** (8 features)
3. **Underlying Defensive Performance** (5 features)
4. **Bonus-Point Potential** (4 features)
5. **Team & Tactical Context** (5 features)
6. **Fixture Opportunity** (4 features)
7. **Price & Value** (3 features)

**Total candidate pool:** 36 features  
**Per-position variants:** 12 features (3-4 variants by position)  
**Total feature space:** ~50 features before selection

---

## CATEGORY 1: PLAYING-TIME RELIABILITY (7 Features)

**Purpose:** Answer: "Will this player actually be on the pitch?"

### 1.1 – Effective Playing Time (Per Season)

**Definition:** Total minutes played divided by 90, weighted by recency.

**Calculation:**
```
effective_90s = SUM(minutes) / 90  [per season]
effective_90s_recent = SUM(minutes over last N GWs) / (N × 90)  [recent form]
```

**Aggregation window:** 
- Season-level: 38 GWs (full season)
- Recent: 5 GWs (short-term trend)

**Normalization:** 
- Clip to [0, 5] for season (max 380 minutes/GW = 4.2 90s per GW, rare)
- Divide by position median to create relative index

**Position-specific:** 
- GK: rarely substituted; higher expected 90s
- DEF: occasional rotation; high expected 90s
- MID: frequent rotation; moderate expected 90s
- FWD: high rotation; lower expected 90s

**Redundancy risk:** NONE – core independent variable

**Include/Exclude:** ✓ **INCLUDE** – foundational

**Rationale:** Replaces existing `effective_90s` but computed per season and recent window separately.

---

### 1.2 – Starts per Match Ratio

**Definition:** Percentage of matches started (not brought on as substitute).

**Calculation:**
```
starts_ratio = SUM(starts) / SUM(appearances)  [per season]
starts_ratio_recent = SUM(starts over last N GWs) / N
```

**Aggregation window:** 
- Season: 38 GWs
- Recent: 5 GWs

**Normalization:** 
- Percentage [0, 1]
- Logit transform if needed for extreme values

**Position-specific:** 
- GK: almost always start; expect 0.95–1.0
- DEF: usually start; expect 0.70–1.0
- MID: mixed; expect 0.40–0.90
- FWD: often rotated; expect 0.20–0.70

**Redundancy risk:** MEDIUM – overlaps with effective_90s (high 90s typically correlates with high starts). But starts capture *deployment intention* independently of actual minutes.

**Include/Exclude:** ✓ **INCLUDE** – complements effective_90s

**Rationale:** Useful for detecting: (a) substitutes who come on in final minutes, (b) players getting rotated despite same 90s.

---

### 1.3 – Substitution Frequency

**Definition:** Average minutes per appearance (indicating when player enters match).

**Calculation:**
```
avg_minutes_per_appearance = SUM(minutes) / SUM(appearances)
substitution_rate = (SUM(appearances) - SUM(starts)) / SUM(appearances)
```

**Aggregation window:** 
- Season: 38 GWs
- Recent: 5 GWs

**Normalization:** 
- avg_minutes_per_appearance: divide by 90 (range [0, 1])
- substitution_rate: already [0, 1]

**Position-specific:** 
- GK: almost never substituted; expect avg_minutes ~90
- DEF: rarely substituted; expect avg_minutes ~75–90
- MID: often substituted; expect avg_minutes ~45–80
- FWD: frequently substituted; expect avg_minutes ~30–70

**Redundancy risk:** MEDIUM – redundant with starts_ratio and effective_90s. Is a derived feature.

**Include/Exclude:** ? **TRANSFORM** – use (1 - substitution_rate) as a binary indicator instead of continuous metric.

**Rationale:** Detects "super-sub" pattern (frequent 20-minute cameos). High substitution_rate is a negative signal (unreliable starter).

---

### 1.4 – Minutes Trend (Recent vs. Historical)

**Definition:** Change in effective 90s: recent vs. season-to-date.

**Calculation:**
```
recent_90s = SUM(minutes over last 5 GWs) / (5 × 90)
season_90s = SUM(minutes over all GWs) / (38 × 90)
minutes_trend = (recent_90s - season_90s) / season_90s  [percent change]
```

**Aggregation window:** 
- Recent: last 5 GWs
- Comparison: cumulative season-to-date

**Normalization:** 
- Percentage change, clip to [−1, 1]

**Position-specific:** 
- No major differences, but GK/DEF typically show lower volatility

**Redundancy risk:** LOW – independent trend metric

**Include/Exclude:** ✓ **INCLUDE** – detects role changes, injuries returning

**Rationale:** Captures momentum. Example: recovering from injury shows positive trend; losing position shows negative trend.

---

### 1.5 – Availability Indicator (Current Status + Forecast)

**Definition:** Composite of current status and predicted probability of playing next round.

**Calculation:**
```
availability_score = (chance_of_playing_next_round / 100) 
                   IF chance is populated
                   ELSE: 1.0 if status=="active", 0.55 if uncertain, 0.1 if unavailable
```

**Aggregation window:** 
- Point-in-time snapshot (pre-deadline)

**Normalization:** 
- [0, 1] probability

**Position-specific:** 
- No major differences

**Redundancy risk:** LOW – unique signal; depends on real-time injury/suspension news

**Include/Exclude:** ✓ **INCLUDE** – essential for current-week recommendations

**Rationale:** Current approach uses availability_factor [0.55, 1.0], which is reasonable but coarse. This extends it.

---

### 1.6 – Role Stability (Position Consistency)

**Definition:** Percentage of appearances in player's primary position (can change mid-season).

**Calculation:**
```
primary_position = MODE(position) over season
role_stability = COUNT(appearances where position == primary_position) / SUM(appearances)
```

**Aggregation window:** 
- Season: 38 GWs

**Normalization:** 
- [0, 1]

**Position-specific:** 
- Meaningful for all positions

**Redundancy risk:** MEDIUM – requires position-per-GW data (not always available in aggregates).

**Include/Exclude:** ? **CONDITIONAL** – include only if position-per-GW data is archivable.

**Rationale:** Detects players moved across positions (e.g., MID played as DEF); these typically underperform.

---

### 1.7 – Recent Injury Proximity

**Definition:** Gameweeks since last injury/suspension (if known).

**Calculation:**
```
gws_since_absence = gameweek_id - gameweek_id_when_last_absent
```

**Aggregation window:** 
- None; calculated at prediction time

**Normalization:** 
- Raw count, clip to [0, 8] (8+ GWs is "fully recovered")
- Logistic transform if needed

**Position-specific:** 
- No major differences, but impacts vary by player (e.g., full-backs vs. keepers may recover differently)

**Redundancy risk:** LOW – unique signal; requires news/absence tracking.

**Include/Exclude:** ? **OPTIONAL** – useful but requires infrastructure to track injury dates.

**Rationale:** Captures return-from-injury momentum. Recent returnees often underperform initially.

---

### Summary: Playing-Time Reliability Features

| Feature | Include | Rationale | Position-Specific |
|---------|---------|-----------|-------------------|
| effective_90s (season & recent) | ✓ | Core; foundational | Baseline differs by role |
| starts_ratio | ✓ | Deployment pattern | Yes; GK ~100%, FWD ~40% |
| substitution_frequency | ⚠ | Use as binary indicator only | Yes; moderate variance |
| minutes_trend | ✓ | Momentum/role change signal | No |
| availability_score | ✓ | Pre-deadline status | No |
| role_stability | ⚠ | Include if position-per-GW available | Yes; defensive only |
| injury_proximity | ⚠ | Useful but infrastructure-dependent | Maybe; varies by player |

**Recommended for inclusion:** 5–6 features  
**Action items:** Verify position-per-GW data availability; add injury tracking if feasible.

---

## CATEGORY 2: UNDERLYING ATTACKING PERFORMANCE (8 Features)

**Purpose:** Answer: "How good is this player at creating attacking value?"

### 2.1 – Goals per 90

**Definition:** Goals scored normalized by playing time.

**Calculation:**
```
goals_per_90 = SUM(goals_scored) / (SUM(minutes) / 90)  [per season]
goals_per_90_recent = SUM(goals over last 5 GWs) / (SUM(minutes over last 5 GWs) / 90)
```

**Aggregation window:** 
- Season: 38 GWs
- Recent: 5 GWs

**Normalization:** 
- Raw count per 90
- Divide by position median (e.g., FWD median ~0.45 goals/90, MID ~0.10)
- Clip extreme outliers [0, 1.5] per 90

**Position-specific:** 
- FWD: baseline ~0.35–0.50 goals/90
- MID: baseline ~0.08–0.15 goals/90
- DEF: baseline ~0.01–0.05 goals/90
- GK: baseline ~0 goals/90 (irrelevant)

**Redundancy risk:** MEDIUM – related to threat score, but direct measurement vs. proxy.

**Include/Exclude:** ✓ **INCLUDE** – primary attacking metric

**Rationale:** Direct, interpretable; position-normalized.

---

### 2.2 – Assists per 90

**Definition:** Assists normalized by playing time.

**Calculation:**
```
assists_per_90 = SUM(assists) / (SUM(minutes) / 90)
assists_per_90_recent = [recent 5 GWs]
```

**Aggregation window:** 
- Season: 38 GWs
- Recent: 5 GWs

**Normalization:** 
- Raw count per 90
- Divide by position median (e.g., MID median ~0.08 assists/90, FWD ~0.04)
- Clip [0, 0.5] per 90

**Position-specific:** 
- MID: baseline ~0.05–0.12 assists/90
- DEF: baseline ~0.02–0.08 assists/90 (set-piece specialists)
- FWD: baseline ~0.02–0.05 assists/90
- GK: baseline ~0 assists/90 (irrelevant)

**Redundancy risk:** MEDIUM – related to creativity, but direct measurement.

**Include/Exclude:** ✓ **INCLUDE** – primary creative metric

**Rationale:** Direct, interpretable; position-normalized.

---

### 2.3 – Threat (Opta-Derived Attacking Threat)

**Definition:** Opta's proprietary metric of attacking threat (0–100 scale per GW, aggregated as mean).

**Calculation:**
```
mean_threat = MEAN(threat) per season  [per GW; then aggregate]
threat_per_90 = SUM(threat) / (SUM(minutes) / 90)  [alternative: treat as continuous metric]
```

**Aggregation window:** 
- Season: 38 GWs (mean or sum?)
- Recent: 5 GWs

**Normalization:** 
- If mean: already 0–100 scale; divide by 100 to [0, 1] or use as-is
- If sum: divide by (SUM(minutes)/90) to get per-90 rate

**Position-specific:** 
- Different baselines by position (FWD > MID > DEF > GK)

**Redundancy risk:** HIGH – proxy for goals/assists but less direct. Correlation with goals_per_90 likely 0.7–0.85.

**Include/Exclude:** ? **OPTIONAL** – use only if goals/assists data are sparse or as validation metric.

**Rationale:** Threat is a proxy for underlying shot quality. If goals_per_90 is available, threat is redundant. But threat might be more stable (less dependent on finishing luck).

---

### 2.4 – Creativity (Opta-Derived Passing/Creation)

**Definition:** Opta's proprietary metric of chance creation (0–100 scale per GW).

**Calculation:**
```
mean_creativity = MEAN(creativity) per season
creativity_per_90 = SUM(creativity) / (SUM(minutes) / 90)
```

**Aggregation window:** 
- Season: 38 GWs
- Recent: 5 GWs

**Normalization:** 
- If mean: [0, 100] scale; divide by 100 to [0, 1]
- If per-90: normalize by position median

**Position-specific:** 
- Different baselines: MID (high creation) > DEF (set-pieces) > FWD (lower pass volume)

**Redundancy risk:** HIGH – proxy for assists; similar to assists_per_90 but less direct.

**Include/Exclude:** ? **OPTIONAL** – use as validation or if assists data unavailable.

**Rationale:** Creativity correlates with assists but is noisier (chance quality vs. chance conversion).

---

### 2.5 – ICT Index (Composite: Influence + Creativity + Threat)

**Definition:** Opta's composite metric combining influence, creativity, threat (typically 0–100).

**Calculation:**
```
mean_ict = MEAN(ict_index) per season
ict_per_90 = SUM(ict_index) / (SUM(minutes) / 90)
```

**Aggregation window:** 
- Season: 38 GWs
- Recent: 5 GWs

**Normalization:** 
- [0, 100] scale; divide by 100 or use as-is

**Position-specific:** 
- Different expected ranges by position

**Redundancy risk:** VERY HIGH – is a composite of influence, creativity, threat. If you include those three separately, ICT is redundant. If you exclude them, ICT is useful.

**Include/Exclude:** ✗ **EXCLUDE** if including influence, creativity, threat separately. Otherwise ✓ **INCLUDE** as aggregate.

**Rationale:** Choose either: (a) granular (influence, creativity, threat) OR (b) aggregate (ICT). Not both.

---

### 2.6 – Expected Goals (Proxy via Threat or Shots)

**Definition:** Expected probability of scoring given shot opportunities.

**Calculation:**
```
Available option 1: Use threat as proxy (no xG in FPL API)
xg_proxy = threat_per_90 × (goals_per_90 / mean_threat_per_90 by position)
[Calibrate proxy using historical relationship]

Available option 2: Model from historical conversion
goals_conversion_rate = goals_per_90 / threat_per_90
```

**Aggregation window:** 
- Season: 38 GWs
- Recent: 5 GWs

**Normalization:** 
- Probability [0, 1]

**Position-specific:** 
- FWD: higher expected conversion (0.08–0.15)
- MID: moderate (0.05–0.10)
- DEF: low (0.02–0.05)

**Redundancy risk:** HIGH – if threat is included, expected goals is redundant. If threat is excluded, xG (via proxy) is useful.

**Include/Exclude:** ? **CONDITIONAL** – include as proxy only if external xG data not available. Better to use threat+goals separately.

**Rationale:** Without real xG data from StatsBomb/Understat, creating a proxy is complex and may be less reliable than individual metrics.

---

### 2.7 – Recent Attacking Form Trend

**Definition:** Change in goals + assists production: recent vs. season average.

**Calculation:**
```
recent_goals_assists_per_90 = (SUM(goals + assists over last 5 GWs)) / (SUM(minutes over last 5 GWs) / 90)
season_goals_assists_per_90 = (SUM(goals + assists over season)) / (SUM(minutes) / 90)
attacking_form_trend = (recent - season) / season  [percent change]
```

**Aggregation window:** 
- Recent: 5 GWs
- Comparison: season-to-date

**Normalization:** 
- Percent change, clip [−0.5, 1.0]

**Position-specific:** 
- No major differences

**Redundancy risk:** LOW – trend metric; independent of absolute levels.

**Include/Exclude:** ✓ **INCLUDE** – captures momentum in attacking returns.

**Rationale:** Detects: (a) emerging form, (b) sudden drop-off, (c) role/deployment changes affecting output.

---

### 2.8 – Penalty-Taking Status (Binary + Recent Penalties)

**Definition:** Whether player is penalty taker; recent penalty-taking activity.

**Calculation:**
```
is_penalty_taker = 1 if penalties_scored + penalties_missed > 0 else 0
recent_penalties = COUNT(penalty_events in last 5 GWs)
penalty_conversion = penalties_scored / (penalties_scored + penalties_missed)
```

**Aggregation window:** 
- Season: 38 GWs (identify taker)
- Recent: 5 GWs (recent activity)

**Normalization:** 
- Binary [0, 1] or count

**Position-specific:** 
- Highly position-specific: only relevant for MID/FWD set-piece specialists

**Redundancy risk:** LOW – unique signal; independent of goals/assists.

**Include/Exclude:** ✓ **INCLUDE (positional)** – add 1–2 expected points if penalty taker; capture penalty variance.

**Rationale:** Penalty-takers have higher expected points. A missed penalty is binary variance; worth modeling.

---

### Summary: Attacking Performance Features

| Feature | Include | Rationale | Position-Specific |
|---------|---------|-----------|-------------------|
| goals_per_90 | ✓ | Primary metric; direct | Yes; normalized |
| assists_per_90 | ✓ | Primary metric; direct | Yes; normalized |
| threat | ⚠ | Use OR goals, not both | Yes; position baseline |
| creativity | ⚠ | Use OR assists, not both | Yes; position baseline |
| ict_index | ⚠ | Use OR influence/creativity/threat separately | Yes; position baseline |
| xg_proxy | ⚠ | Only if no real xG; otherwise redundant | Yes; conversion-rate-based |
| attacking_form_trend | ✓ | Momentum metric | No |
| penalty_taker_status | ✓ | Binary; captures 1–2 point variance | MID/FWD only |

**Recommended for inclusion:** 4–5 core features (goals_per_90, assists_per_90, attacking_form_trend, penalty_status)  
**Action items:** Decide: use granular (goals/assists/threat/creativity) or aggregate (ICT). Choose one path.

---

## CATEGORY 3: UNDERLYING DEFENSIVE PERFORMANCE (5 Features)

**Purpose:** Answer: "How good is this player at defending?" (Primarily DEF/GK; marginal for MID)

### 3.1 – Clean Sheets per 90

**Definition:** Number of clean sheets normalized by playing time.

**Calculation:**
```
clean_sheets_per_90 = SUM(clean_sheets) / (SUM(minutes) / 90)  [per season]
clean_sheets_per_game = SUM(clean_sheets) / SUM(appearances)
```

**Aggregation window:** 
- Season: 38 GWs
- Recent: 5 GWs

**Normalization:** 
- Clean sheets per 90 (range [0, 0.5] typical)
- Divide by position median

**Position-specific:** 
- GK: baseline ~0.3–0.4 clean sheets per 90 (more reliable than team CS rate)
- DEF: baseline ~0.2–0.35 clean sheets per 90 (team CS rate)
- MID: baseline ~0.05–0.15 (rarely get CS)
- FWD: baseline ~0 (never get CS)

**Redundancy risk:** MEDIUM – related to team defensive strength, but player-level metric.

**Include/Exclude:** ✓ **INCLUDE** – primary defensive metric

**Rationale:** Direct measurement; influenced by both player performance and team strength.

---

### 3.2 – Team Clean Sheet Rate (Contextual)

**Definition:** Percentage of matches where player's team conceded 0 goals (while player was on pitch).

**Calculation:**
```
team_cs_rate = COUNT(clean sheets during player's minutes) / SUM(appearances when player started)
```

**Aggregation window:** 
- Season: 38 GWs

**Normalization:** 
- Percentage [0, 1]

**Position-specific:** 
- Relevant for GK/DEF/MID (all get CS points if team CS)

**Redundancy risk:** HIGH – overlaps with clean_sheets_per_90. But separates team strength from individual performance.

**Include/Exclude:** ⚠ **OPTIONAL** – use as context/control variable, not primary predictor.

**Rationale:** Useful for: (a) detecting defensive strength of team, (b) controlling for team effects in player-level models.

---

### 3.3 – Goals Conceded per 90

**Definition:** Goals conceded by player's team, normalized by player's playing time.

**Calculation:**
```
ga_per_90 = SUM(goals_conceded during player's minutes) / (SUM(minutes) / 90)
```

**Aggregation window:** 
- Season: 38 GWs
- Recent: 5 GWs

**Normalization:** 
- Raw count per 90; clip [0, 3]

**Position-specific:** 
- Relevant for GK/DEF/MID (indirectly for FWD via pressing context)

**Redundancy risk:** HIGH – inverse of clean sheets; same information different direction.

**Include/Exclude:** ⚠ **OPTIONAL** – use either goals_against OR clean_sheets, not both.

**Rationale:** Goals against is more granular (0–4 GA vs. binary CS), so may be preferable for modeling.

---

### 3.4 – Influence (Opta-Derived Defensive Activity)

**Definition:** Opta's metric of overall player influence (0–100 scale per GW); includes defensive actions.

**Calculation:**
```
mean_influence = MEAN(influence) per season
influence_per_90 = SUM(influence) / (SUM(minutes) / 90)
```

**Aggregation window:** 
- Season: 38 GWs
- Recent: 5 GWs

**Normalization:** 
- [0, 100] scale; divide by 100 or use position-specific normalized scale

**Position-specific:** 
- GK: high influence (includes distribution)
- DEF: high influence (tackles, blocks, positioning)
- MID: moderate influence (less defensive)
- FWD: low influence (minimal defensive activity)

**Redundancy risk:** MEDIUM – proxy for defensive involvement; not independent of CS.

**Include/Exclude:** ✓ **INCLUDE** – captures defensive activity beyond CS outcomes.

**Rationale:** Influence reflects underlying defensive actions; CS is outcome. Both useful; not redundant.

---

### 3.5 – Defensive Form Trend

**Definition:** Change in clean sheets + defensive activity: recent vs. season average.

**Calculation:**
```
recent_defensive_score = (SUM(clean_sheets over last 5 GWs) × 4 + SUM(influence over last 5 GWs)) / SUM(minutes over last 5 GWs) / 90
season_defensive_score = (SUM(clean_sheets) × 4 + SUM(influence)) / SUM(minutes) / 90
defensive_form_trend = (recent - season) / season
```

**Aggregation window:** 
- Recent: 5 GWs
- Comparison: season-to-date

**Normalization:** 
- Percent change, clip [−0.5, 1.0]

**Position-specific:** 
- More relevant for GK/DEF

**Redundancy risk:** LOW – trend metric; independent of absolute levels.

**Include/Exclude:** ✓ **INCLUDE** – captures momentum in defensive performance.

**Rationale:** Detects: (a) improving form, (b) team defensive crisis, (c) role/tactical change.

---

### Summary: Defensive Performance Features

| Feature | Include | Rationale | Position-Specific |
|---------|---------|-----------|-------------------|
| clean_sheets_per_90 | ✓ | Primary metric; direct | Yes; GK~0.35, DEF~0.25, MID~0.1 |
| team_cs_rate | ⚠ | Use as context/control | Yes; relevant for GK/DEF/MID |
| goals_against_per_90 | ⚠ | Use OR clean_sheets, not both | Yes; same info, different direction |
| influence | ✓ | Activity-based; independent of CS | Yes; GK/DEF high, FWD low |
| defensive_form_trend | ✓ | Momentum metric | Yes; more relevant for DEF/GK |

**Recommended for inclusion:** 3–4 features (clean_sheets_per_90, influence, defensive_form_trend)  
**Action items:** Decide: use clean_sheets_per_90 OR goals_against_per_90, not both.

---

## CATEGORY 4: BONUS-POINT POTENTIAL (4 Features)

**Purpose:** Answer: "How likely is this player to earn bonus points?"

### 4.1 – Bonus Points per 90

**Definition:** Total bonus points earned, normalized by playing time.

**Calculation:**
```
bonus_per_90 = SUM(bonus) / (SUM(minutes) / 90)  [per season]
bonus_per_game = SUM(bonus) / SUM(appearances)
```

**Aggregation window:** 
- Season: 38 GWs
- Recent: 5 GWs

**Normalization:** 
- Raw count per 90; typical range [0, 0.2] bonus per 90
- Divide by position median to normalize

**Position-specific:** 
- GK: low bonus (0.02–0.08 per 90; mostly from clean sheets)
- DEF: low bonus (0.05–0.12 per 90; from clean sheets + activity)
- MID: moderate bonus (0.06–0.15 per 90; from goals/assists/activity)
- FWD: moderate bonus (0.08–0.15 per 90; from goals/assists)

**Redundancy risk:** MEDIUM – related to BPS and underlying performance (threat, creativity, clean sheets).

**Include/Exclude:** ✓ **INCLUDE** – captures FPL's bonus system outcomes

**Rationale:** Direct measurement; position-normalized. Useful because bonus is often driver of top players' performance.

---

### 4.2 – Bonus Points System (BPS) Score per 90

**Definition:** Average BPS score (0–100+ per GW), normalized by playing time.

**Calculation:**
```
mean_bps = MEAN(bps) per season  [average BPS per GW played]
bps_per_90 = SUM(bps) / (SUM(minutes) / 90)
```

**Aggregation window:** 
- Season: 38 GWs
- Recent: 5 GWs

**Normalization:** 
- BPS scores range 0–100+; normalize by position median

**Position-specific:** 
- Different expected baselines by position
- GK: high BPS (clean sheets worth 10 points each)
- DEF: high BPS (CS + tackles/blocks)
- MID: moderate-high BPS (goals, assists, blocks)
- FWD: moderate BPS (goals, assists, dribbles)

**Redundancy risk:** HIGH – BPS drives bonus; if you have bonus_per_90, BPS is mostly redundant. But BPS is more predictive of *future* bonus (outcome vs. input).

**Include/Exclude:** ? **OPTIONAL** – use BPS as predictor of bonus (input → outcome), not as outcome itself.

**Rationale:** BPS is "what earns bonus"; bonus is "what was earned". BPS is more forward-looking.

---

### 4.3 – Bonus Finish Frequency (Top-3 BPS Appearances)

**Definition:** Percentage of matches where player finished in top-3 BPS (eligible for bonus).

**Calculation:**
```
bonus_finishes = COUNT(GWs where BPS rank in top 3) / SUM(appearances)
```

**Aggregation window:** 
- Season: 38 GWs
- Recent: 5 GWs

**Normalization:** 
- Percentage [0, 1]

**Position-specific:** 
- Different expected frequencies by position and team strength

**Redundancy risk:** LOW – captures consistency of bonus returns, independent of total bonus.

**Include/Exclude:** ✓ **INCLUDE** – captures bonus consistency/frequency

**Rationale:** A player might get bonus in 60% of appearances (high frequency) but averaging 1.5 points per bonus (moderate amount). Both metrics useful.

---

### 4.4 – Bonus Volatility

**Definition:** Variance in bonus points across season (capturing "feast-or-famine" pattern).

**Calculation:**
```
bonus_volatility = STD(bonus points per GW over season)
bonus_coefficient_of_variation = STD / MEAN  [normalized volatility]
```

**Aggregation window:** 
- Season: 38 GWs

**Normalization:** 
- Use coefficient of variation [0, ∞) to normalize across positions

**Position-specific:** 
- GK: typically low volatility (consistency)
- DEF: low-moderate volatility
- MID: moderate volatility (depends on role)
- FWD: high volatility (rotation + inconsistent form)

**Redundancy risk:** LOW – captures risk, independent of mean bonus.

**Include/Exclude:** ? **OPTIONAL** – use in uncertainty quantification, not as primary feature.

**Rationale:** High volatility suggests unpredictability; useful for risk-adjusted recommendations but not directly predictive.

---

### Summary: Bonus-Point Features

| Feature | Include | Rationale | Position-Specific |
|---------|---------|-----------|-------------------|
| bonus_per_90 | ✓ | Primary outcome metric | Yes; normalized by position |
| bps_per_90 | ⚠ | Predictor of bonus; use instead of bonus outcome | Yes; GK high, FWD moderate |
| bonus_finish_frequency | ✓ | Consistency metric | Yes; captures top-3 frequency |
| bonus_volatility | ⚠ | Risk metric; use in uncertainty layer | Yes; GK low, FWD high |

**Recommended for inclusion:** 2–3 features (bonus_per_90 or BPS_per_90, bonus_finish_frequency)  
**Action items:** Decide: use bonus (historical outcome) OR BPS (forward-looking). Recommend: use both, BPS as primary predictor.

---

## CATEGORY 5: TEAM & TACTICAL CONTEXT (5 Features)

**Purpose:** Answer: "What is the team environment and player's role within it?"

### 5.1 – Team Attacking Strength (Bootstrap Rating)

**Definition:** FPL's team-level attacking strength rating (typically 10–90 scale).

**Calculation:**
```
team_strength_attack = [from bootstrap teams table]
team_strength_attack_home = [home attacking rating]
team_strength_attack_away = [away attacking rating]
```

**Aggregation window:** 
- Point-in-time: snapshot at prediction time

**Normalization:** 
- Divide by 100 or use 10–90 scale as-is; should be position-normalized

**Position-specific:** 
- Higher impact for FWD/MID (attacking)
- Lower impact for DEF/GK

**Redundancy risk:** LOW – team-level signal; independent of player metrics

**Include/Exclude:** ✓ **INCLUDE** – context for attacking opportunity

**Rationale:** Team attacking strength influences expected goals, assists. Useful as contextual modifier.

---

### 5.2 – Team Defensive Strength (Bootstrap Rating)

**Definition:** FPL's team-level defensive strength rating.

**Calculation:**
```
team_strength_defence = [from bootstrap teams table]
team_strength_defence_home = [home defensive rating]
team_strength_defence_away = [away defensive rating]
```

**Aggregation window:** 
- Point-in-time: snapshot at prediction time

**Normalization:** 
- Divide by 100 or use 10–90 scale as-is

**Position-specific:** 
- Higher impact for DEF/GK (defending)
- Lower impact for FWD/MID

**Redundancy risk:** LOW – team-level signal; independent of player metrics

**Include/Exclude:** ✓ **INCLUDE** – context for clean sheet opportunity

**Rationale:** Team defensive strength influences clean sheet probability. Useful as contextual modifier.

---

### 5.3 – Playing Position Role (Categorical)

**Definition:** Player's position and subtype (where available).

**Calculation:**
```
position = {GK, DEF, MID, FWD}  [from position_id mapping]
```

**Aggregation window:** 
- Point-in-time snapshot (can change mid-season)

**Normalization:** 
- One-hot encoding or ordinal [0, 3]

**Position-specific:** 
- This IS the position indicator

**Redundancy risk:** NONE – fundamental categorical variable

**Include/Exclude:** ✓ **INCLUDE** – required for position-specific modeling

**Rationale:** All position-specific sub-models require this.

---

### 5.4 – Formation/Role Changes (GW-level context)

**Definition:** Count of formation changes or tactical shifts mid-season (if available).

**Calculation:**
```
formation_changes = COUNT(shifts in team formation over season)
role_changes = COUNT(position/role changes for player over season)
```

**Aggregation window:** 
- Season: 38 GWs

**Normalization:** 
- Raw count; typically 0–3 major changes per season

**Position-specific:** 
- No major differences

**Redundancy risk:** LOW – unique signal about tactical volatility

**Include/Exclude:** ? **OPTIONAL** – useful if formation data tracked; otherwise skip.

**Rationale:** Formation changes can significantly impact player value. Example: switch from 4-3-3 to 5-2-3 increases defender value.

---

### 5.5 – Player Nationality / League Tenure

**Definition:** Familiarity with league (years in EPL for non-English players; implicit for English).

**Calculation:**
```
epl_seasons = COUNT(seasons in EPL so far)
is_foreign = 1 if nationality != England else 0
league_adjustment = 0.8 if first_season_foreign else 1.0
```

**Aggregation window:** 
- Career/season-level

**Normalization:** 
- Year count or binary

**Position-specific:** 
- No major differences

**Redundancy risk:** LOW – unique signal about adaptation

**Include/Exclude:** ? **OPTIONAL** – useful if implementing forward models; adds complexity.

**Rationale:** New foreign signings typically underperform first season; improves subsequent seasons. Hard to track precisely.

---

### Summary: Team & Tactical Context Features

| Feature | Include | Rationale | Position-Specific |
|---------|---------|-----------|-------------------|
| team_attacking_strength | ✓ | Context for attacking opportunity | MID/FWD > DEF/GK |
| team_defensive_strength | ✓ | Context for defending opportunity | DEF/GK > MID/FWD |
| position | ✓ | Required for position-specific models | N/A (IS the position) |
| formation_changes | ⚠ | Useful if tracked; otherwise skip | No |
| player_league_tenure | ⚠ | Nice-to-have; adds complexity | No |

**Recommended for inclusion:** 3 features (team_attack, team_defence, position)  
**Action items:** Formation/tenure tracking is nice-to-have; deprioritize for MVP.

---

## CATEGORY 6: FIXTURE OPPORTUNITY (4 Features)

**Purpose:** Answer: "Are the upcoming fixtures favorable?"

### 6.1 – Fixture Difficulty Rating (Next N GWs)

**Definition:** FPL's official fixture difficulty (FDR, 1–5 scale) for upcoming matches.

**Calculation:**
```
next_1_fdr = [FDR for next match]
next_3_avg_fdr = MEAN([FDR for next 3 matches])
next_5_avg_fdr = MEAN([FDR for next 5 matches])
```

**Aggregation window:** 
- Forward-looking: next 1, 3, 5 GWs (no aggregation; prospective)

**Normalization:** 
- 1–5 scale; invert to 0–1 where 1 = easiest
- fixture_adjustment = 3.0 / fdr, clip [0.7, 1.3]

**Position-specific:** 
- No major differences; same fixture for all players on a team

**Redundancy risk:** NONE – forward-looking; unique signal

**Include/Exclude:** ✓ **INCLUDE** – essential forward-looking feature

**Rationale:** Official FPL FDR is reasonable but potentially oversmoothed. May consider alternative (e.g., opponent strength rating).

---

### 6.2 – Opponent-Specific Strength (Defending Team Rating)

**Definition:** Defending team's strength rating from bootstrap.

**Calculation:**
```
opponent_attack_strength = [defending team's attacking strength, from bootstrap]
opponent_defence_strength = [defending team's defensive strength]
opponent_strength_avg = (opponent_attack + opponent_defence) / 2
```

**Aggregation window:** 
- Next opponent only (or average next 5)

**Normalization:** 
- Divide by 100; compare to league average

**Position-specific:** 
- Impact differs: attacking players care about opponent defence; defending players care about opponent attack

**Redundancy risk:** MEDIUM – relates to FDR but more granular. FDR likely already incorporates team strength.

**Include/Exclude:** ? **OPTIONAL** – use if want more granular opponent modeling; otherwise FDR is sufficient.

**Rationale:** FDR is aggregate; opponent strength is decomposed. Tradeoff: complexity vs. insight.

---

### 6.3 – Home/Away Status (Next N GWs)

**Definition:** Upcoming matches at home vs. away (if known).

**Calculation:**
```
next_home_count = COUNT(home matches in next 5 GWs)
next_away_count = COUNT(away matches in next 5 GWs)
home_ratio = next_home_count / 5
```

**Aggregation window:** 
- Next 5 GWs

**Normalization:** 
- Percentage [0, 1]

**Position-specific:** 
- No major differences (though team strength ratings split home/away)

**Redundancy risk:** LOW – independent signal; separable from opponent strength.

**Include/Exclude:** ✓ **INCLUDE** – home field advantage is real (typically 5–10% boost).

**Rationale:** Home teams typically have better expected outcomes. Worth capturing.

---

### 6.4 – Fixture Congestion (Match Frequency)

**Definition:** Number of matches per week (double gameweeks, blank gameweeks).

**Calculation:**
```
matches_per_week_next_5_gws = COUNT(matches / 5 GWs)
has_blank_gw = 1 if any of next 5 GWs is blank else 0
has_double_gw = 1 if any of next 5 GWs is double else 0
```

**Aggregation window:** 
- Next 5 GWs

**Normalization:** 
- Count; typically 1.0–2.0 matches/week average

**Position-specific:** 
- No major differences

**Redundancy risk:** LOW – unique signal about fixture scheduling

**Include/Exclude:** ✓ **INCLUDE** – double GWs increase opportunity; blanks reduce it.

**Rationale:** Double GW players have 2× points opportunity. Blank GWs eliminate them entirely.

---

### Summary: Fixture Opportunity Features

| Feature | Include | Rationale | Position-Specific |
|---------|---------|-----------|-------------------|
| fixture_difficulty_rating | ✓ | Official FDR; forward-looking | No; same for team |
| opponent_strength | ⚠ | More granular than FDR; optional | Yes; attacking vs. defending care differently |
| home_away_ratio | ✓ | Home field advantage real | No |
| fixture_congestion | ✓ | Double GWs / blanks | No |

**Recommended for inclusion:** 3 features (FDR, home_ratio, congestion)  
**Action items:** Decide on FDR alone vs. FDR + opponent_strength. Recommend: FDR + congestion (simpler).

---

## CATEGORY 7: PRICE & VALUE (3 Features)

**Purpose:** Answer: "Is this player worth the price?"

### 7.1 – Cost per Expected Points (Points per £m)

**Definition:** Expected points normalized by player cost.

**Calculation:**
```
current_price_millions = current_price / 10  [convert from tenths]
expected_points_per_million = expected_points / current_price_millions
```

**Aggregation window:** 
- Per-gameweek or horizon (e.g., next 5 GWs)

**Normalization:** 
- Points per £m; typical range 0.5–2.0 points/£m

**Position-specific:** 
- Different expected ranges:
  - FWD: 0.8–1.5 (expensive, high upside)
  - MID: 1.0–2.0 (varied; many good options)
  - DEF: 1.2–2.5 (cheaper, good value often)
  - GK: 1.5–2.5 (very cheap; good value typically)

**Redundancy risk:** LOW – unique value metric

**Include/Exclude:** ✓ **INCLUDE** – primary value metric

**Rationale:** Captures "bang for buck"; essential for squad-building under budget constraint.

---

### 7.2 – Price Trend (Moving Average Momentum)

**Definition:** Player's recent price movement; up or down.

**Calculation:**
```
price_change_gw = current_price - price_from_5_gws_ago
price_momentum_pct = (current_price - avg_price_last_5_gws) / avg_price_last_5_gws
```

**Aggregation window:** 
- Recent: 5 GWs

**Normalization:** 
- Percentage change [−0.2, 0.2] typical

**Position-specific:** 
- No major differences

**Redundancy risk:** LOW – independent signal

**Include/Exclude:** ✓ **INCLUDE** – captures in-season value shifts

**Rationale:** Rising prices indicate form/popularity; useful as signal but can be noise. Should be modest weight.

---

### 7.3 – Value Relative to Position Peers

**Definition:** Player's PPM or PPP compared to median in position and price range.

**Calculation:**
```
position_peers = all players in same position with price within ±£0.5m
median_ppps_per_million_peers = MEDIAN(points_per_million) for peers
relative_value = player_ppm / median_ppm_peers
```

**Aggregation window:** 
- Current season

**Normalization:** 
- Ratio [0.5, 2.0] typical

**Position-specific:** 
- Applied within position group

**Redundancy risk:** HIGH – derived from PPM + position context. If PPM included, this is partially redundant.

**Include/Exclude:** ? **OPTIONAL** – use for ranking within position; not essential as absolute metric.

**Rationale:** Useful for: "Is this DEF better value than alternatives?" Not needed for absolute scoring.

---

### Summary: Price & Value Features

| Feature | Include | Rationale | Position-Specific |
|---------|---------|-----------|-------------------|
| points_per_million | ✓ | Primary value metric | Yes; normalized by position |
| price_trend_momentum | ✓ | Captures recent value shifts | No |
| relative_value_vs_peers | ⚠ | Ranking within position | Yes; applied to peer group |

**Recommended for inclusion:** 2 features (points_per_million, price_momentum)  
**Action items:** Use relative value for ranking/explanations, not primary scoring.

---

## FEATURE INVENTORY SUMMARY TABLE

| Category | Feature | Include | Rationale | Pos-Spec | Redundancy Risk |
|----------|---------|---------|-----------|----------|-----------------|
| **Playing Time** | effective_90s (S/R) | ✓ | Core; foundational | Yes | None |
| | starts_ratio | ✓ | Deployment pattern | Yes | Medium |
| | substitution_freq | ⚠ | Transform to binary | Yes | Medium |
| | minutes_trend | ✓ | Momentum | No | Low |
| | availability_score | ✓ | Pre-deadline status | No | Low |
| | role_stability | ⚠ | Conditional; if data available | Yes | Medium |
| | injury_proximity | ⚠ | Infrastructure-dependent | Maybe | Low |
| **Attacking** | goals_per_90 | ✓ | Primary metric | Yes | Medium (vs threat) |
| | assists_per_90 | ✓ | Primary metric | Yes | Medium (vs creativity) |
| | threat_per_90 | ⚠ | Use OR goals | Yes | High |
| | creativity_per_90 | ⚠ | Use OR assists | Yes | High |
| | ict_index | ⚠ | Use OR granular | Yes | Very high |
| | xg_proxy | ⚠ | Use if no real xG | Yes | High |
| | attacking_form_trend | ✓ | Momentum | No | Low |
| | penalty_taker_status | ✓ | Binary/variance | MID/FWD | Low |
| **Defensive** | cs_per_90 | ✓ | Primary metric | Yes | Medium |
| | team_cs_rate | ⚠ | Context/control | Yes | High |
| | ga_per_90 | ⚠ | Use OR CS, not both | Yes | High |
| | influence_per_90 | ✓ | Activity-based | Yes | Medium |
| | defensive_form_trend | ✓ | Momentum | Yes | Low |
| **Bonus** | bonus_per_90 | ✓ | Outcome metric | Yes | Medium |
| | bps_per_90 | ⚠ | Forward predictor | Yes | High |
| | bonus_finish_freq | ✓ | Consistency | Yes | Low |
| | bonus_volatility | ⚠ | Risk; use in uncertainty | Yes | Low |
| **Team/Tactic** | team_attack_strength | ✓ | Context | Yes | Low |
| | team_defence_strength | ✓ | Context | Yes | Low |
| | position_id | ✓ | Required | N/A | None |
| | formation_changes | ⚠ | Nice-to-have; skip for MVP | No | Low |
| | league_tenure | ⚠ | Optional; adds complexity | No | Low |
| **Fixture** | fdr_next_n | ✓ | Forward-looking | No | None |
| | opponent_strength | ⚠ | Optional granularity | Yes | Medium |
| | home_away_ratio | ✓ | Field advantage | No | Low |
| | fixture_congestion | ✓ | Double/blank GWs | No | Low |
| **Price/Value** | points_per_million | ✓ | Primary value | Yes | Low |
| | price_momentum | ✓ | Trend | No | Low |
| | relative_value_vs_peers | ⚠ | Ranking; optional | Yes | High |

**Total candidate pool:** 36 features  
**Recommended for MVP:** ~24 features (marked ✓ or ⚠ that should be included)  
**High-priority for first implementation:** 18 core features (all ✓)

---

## FEATURE SELECTION DECISION TREE

### Decision 1: Granular vs. Aggregate Attacking Performance

**Option A: Granular** (recommended)
- Include: goals_per_90, assists_per_90, attacking_form_trend, penalty_status
- Exclude: threat, creativity, ict_index
- Rationale: Direct, interpretable; not masked by composite metrics

**Option B: Aggregate**
- Include: ict_index, attacking_form_trend, penalty_status
- Exclude: goals_per_90, assists_per_90, threat, creativity
- Rationale: Simpler; Opta's composite might capture interactions

**Recommendation:** Option A (Granular) – superior interpretability; Opta metrics can serve as validation.

---

### Decision 2: Defensive Output Metrics

**Option A: Clean Sheets Only**
- Include: cs_per_90, defensive_form_trend
- Exclude: ga_per_90, team_cs_rate
- Rationale: Clean sheets are outcome; simpler

**Option B: Clean Sheets + Activity**
- Include: cs_per_90, influence_per_90, defensive_form_trend
- Exclude: ga_per_90, team_cs_rate
- Rationale: CS is binary; influence captures nuance

**Recommendation:** Option B – influence and CS are complementary, not redundant.

---

### Decision 3: Bonus Modeling

**Option A: Bonus Points (Outcome)**
- Include: bonus_per_90, bonus_finish_freq
- Exclude: bps_per_90
- Rationale: Simpler; matches historical outcomes

**Option B: BPS (Predictor)**
- Include: bps_per_90, bonus_finish_freq
- Exclude: bonus_per_90
- Rationale: Forward-looking; more predictive of future bonus

**Recommendation:** Option B – BPS is input to FPL's bonus system; better for forward predictions.

---

### Decision 4: Fixture Context

**Option A: FDR Only**
- Include: fdr_next_n, fixture_congestion, home_away_ratio
- Exclude: opponent_strength
- Rationale: Simple; FDR already incorporates team strength

**Option B: FDR + Opponent Decomposed**
- Include: fdr_next_n, opponent_attack, opponent_defence, fixture_congestion, home_away_ratio
- Exclude: nothing (all included)
- Rationale: More granular; enables personalized modeling by position

**Recommendation:** Option A for MVP – complexity not justified by likely improvement.

---

### Decision 5: Price/Value Modeling

**Option A: Absolute Value**
- Include: points_per_million, price_momentum
- Exclude: relative_value_vs_peers
- Rationale: Enough for scoring; relative value used for explanations

**Option B: Absolute + Relative**
- Include: points_per_million, price_momentum, relative_value_vs_peers
- Rationale: Relative value helps with within-position selection

**Recommendation:** Option A for MVP – relative value can be computed post-hoc for explanations.

---

## RECOMMENDED MVP FEATURE SET (18 CORE FEATURES)

### Tier 1: Essential (Must include for system to work)

1. **effective_90s_season** – Playing time foundation
2. **effective_90s_recent** – Recent playing time trend
3. **goals_per_90** – Primary attacking metric
4. **assists_per_90** – Primary creative metric
5. **clean_sheets_per_90** – Primary defensive metric
6. **influence_per_90** – Defensive activity
7. **bps_per_90** – Bonus predictor
8. **attacking_form_trend** – Momentum in attack
9. **defensive_form_trend** – Momentum in defense
10. **position_id** – Required for position-specific sub-models

### Tier 2: Important (Strong signal; should include)

11. **starts_ratio** – Deployment pattern
12. **minutes_trend** – Role change detection
13. **availability_score** – Injury/suspension status
14. **team_attack_strength** – Contextual modifier
15. **team_defence_strength** – Contextual modifier
16. **fdr_next_n** – Forward-looking fixture quality
17. **home_away_ratio** – Field advantage
18. **points_per_million** – Value metric

### Tier 3: Optional (Nice-to-have; implement in Phase 2+)

- bonus_finish_frequency
- penalty_taker_status
- fixture_congestion
- price_momentum
- defensive team strength relative to home/away splits
- injury_proximity (if tracking available)

---

## REDUNDANCY HEATMAP (Feature Correlation Risks)

```
                         Per-90     Clean    Bonus   FDR  PPM   Form
                         --------   Sheets   -----   ---  ---   ----
goals_per_90            [X]        [_]      [+]     [_]  [+]   [L]
assists_per_90          [_]        [_]      [+]     [_]  [+]   [L]
threat_per_90           [H]        [_]      [_]     [_]  [_]   [_]
creativity_per_90       [H]        [_]      [_]     [_]  [_]   [_]
ict_index               [VH]       [_]      [_]     [_]  [_]   [_]
clean_sheets_per_90     [_]        [X]      [+]     [+]  [+]   [L]
influence_per_90        [_]        [L]      [_]     [_]  [_]   [L]
bps_per_90              [M]        [M]      [X]     [_]  [_]   [L]
team_attack_strength    [_]        [_]      [_]     [L]  [_]   [_]
team_defence_strength   [_]        [M]      [_]     [L]  [_]   [_]
fdr_next_n              [_]        [+]      [_]     [X]  [L]   [_]
points_per_million      [H]        [M]      [M]     [M]  [X]   [_]
attacking_form_trend    [L]        [_]      [_]     [_]  [L]   [X]

Legend: [X] = same metric; [H] = high correlation (>0.7); [M] = medium (0.4-0.7); 
        [L] = low (<0.4); [_] = independent; [+] = logical connection (not data correlation); [VH] = very high
```

**Key insights:**
- threat & goals_per_90: High redundancy → choose one
- creativity & assists_per_90: High redundancy → choose one
- ict_index: Very high redundancy with (threat, creativity, influence) → choose either ICT or granular
- clean_sheets & team_defence: Medium overlap; both useful (outcome vs. context)
- points_per_million: Some redundancy with everything (it's derived); still useful

---

## FEATURE AVAILABILITY & CALCULATION COMPLEXITY

| Feature | Data Source | Availability | Calc Complexity | Ready for MVP |
|---------|-------------|--------------|-----------------|---------------|
| effective_90s | raw gameweeks | ✓ Full history | Low | ✓ |
| starts_ratio | raw gameweeks | ✓ Full history | Low | ✓ |
| minutes_trend | raw gameweeks | ✓ Full history | Low | ✓ |
| goals_per_90 | raw gameweeks | ✓ Full history | Low | ✓ |
| assists_per_90 | raw gameweeks | ✓ Full history | Low | ✓ |
| clean_sheets_per_90 | raw gameweeks | ✓ Full history | Low | ✓ |
| influence_per_90 | raw gameweeks | ✓ Full history | Low | ✓ |
| bps_per_90 | raw gameweeks | ✓ Full history | Low | ✓ |
| team_attack_strength | bootstrap | ✓ Current | Low | ✓ |
| team_defence_strength | bootstrap | ✓ Current | Low | ✓ |
| fdr_next_n | fixtures | ✓ Current | Low | ✓ |
| home_away_ratio | fixtures | ✓ Current | Medium | ✓ |
| points_per_million | derived | ✓ Trivial | Low | ✓ |
| attacking_form_trend | raw gameweeks | ✓ Full history | Low | ✓ |
| defensive_form_trend | raw gameweeks | ✓ Full history | Low | ✓ |
| bonus_finish_frequency | raw gameweeks | ✓ Full history | Medium | ⚠ |
| penalty_taker_status | raw gameweeks | ✓ Full history | Low | ⚠ |
| price_momentum | bootstrap + history | ⚠ Requires snapshots | Medium | ⚠ |
| opponent_strength_detail | bootstrap | ✓ Current | Low | ✗ |
| injury_proximity | news/tracking | ✗ Requires tracking | High | ✗ |
| formation_changes | tactical tracking | ✗ Requires tracking | High | ✗ |

**MVP-ready features:** 15/18 without external tracking  
**Missing infrastructure:** Price snapshots per GW (for momentum), injury date tracking

---

## NEXT PHASE: STATISTICAL FRAMEWORK DESIGN

Once feature set is finalized, Phase 3 will specify:

1. **Per-90 Normalization Standards**
   - Define exactly how to compute per-90 for each stat
   - Handle edge cases (0 minutes, few appearances)

2. **Aggregation Windows**
   - Season-level: exactly 38 GWs or best available?
   - Recent form: 5 GWs, 3 GWs, or variable?
   - Winsorization/clipping thresholds

3. **Bayesian Priors**
   - Position-specific priors for each metric
   - Prior strength (equivalent 90s) for each position
   - How to update priors from training data

4. **Missing Value Handling**
   - Zero vs. missing (player didn't play vs. stat not recorded)
   - Imputation strategy for sparse players

5. **Low-Minute Handling**
   - Minimum 90s threshold before using per-90 stats
   - Shrinkage/uncertainty for players with <450 minutes

---

## SUMMARY & NEXT STEPS

**This phase has produced:**

✓ 36 candidate features across 7 categories
✓ Redundancy analysis with heatmap
✓ Position-specific variants for 12 features
✓ **18-feature MVP set recommended for immediate implementation**
✓ Decision tree resolving major trade-offs (granular vs. aggregate, etc.)
✓ Feature availability audit

**Ready to proceed to Phase 3: Statistical Framework Design**

**File saved to:** `docs/RECOMMENDATION_SCORE_REDESIGN.md` (Section already prepared; will be updated)

