# FPL data pipeline

A reproducible, local-first pipeline for the public Fantasy Premier League website API. It preserves every response as an immutable JSON envelope, creates Parquet tables, and exposes the same tables through DuckDB. The API paths are **unofficial web API endpoints**: available and publicly callable at the time of verification (27 July 2026), but not a contracted, versioned API. Treat schema drift as an operational risk.

## Stage 1 — Conceptual model

FPL has a current-season operational model. A `season` is a logical ingestion partition, not a field supplied by most endpoints. `event` means gameweek. `element` means player; `entry` means manager team.

| Entity | Key | Role and important fields | Change type / physical form |
|---|---|---|---|
| Season | `season_id` | Partition and competition context | dimension |
| Gameweek/event | `gameweek_id` | deadline, finish/data-check flags, aggregate ranks/scores | slowly changing dimension |
| Player/element | `player_id` | names, team, position, availability, current price | SCD snapshot (IDs are season-scoped in practice) |
| Team | `team_id` | name, short name, strength and home/away ratings | SCD snapshot |
| Position | `position_id` | goalkeeper/defender/midfielder/forward, squad rules | static dimension |
| Fixture | `fixture_id` | event, teams, kick-off, scores, FDR, status | fact / mutable until final |
| Player gameweek | `(player_id, gameweek_id)` | points and match aggregate performance | periodic snapshot/fact |
| Player season stats | `(player_id, season_id)` | aggregate from element summary/history or sum of facts | derived aggregate |
| Manager/entry | `manager_id` | team/player names, favourite team | dimension / restricted scope |
| Manager gameweek | `(manager_id, gameweek_id)` | points, rank, value, bank, transfer cost | snapshot fact |
| Manager picks | `(manager_id, gameweek_id, player_id)` | starting position, captaincy, multiplier, buy/sell prices | bridge: relates an entry’s 15-player roster to player and event |
| Manager transfers | transfer record (`time`, manager, in/out) | player in/out and timestamp | transaction fact |
| Classic league | `league_id` | league metadata | dimension |
| League standing | `(league_id, manager_id, page/as_of)` | rank, total | snapshot / bridge |
| Chip | manager/event/chip | wildcard, free hit, bench boost, triple captain | event attribute/snapshot |

Ownership and price are **point-in-time snapshots**: `selected` is a selected-count snapshot and `now_cost`/`value` are prices in tenths of £m. `transfers_in`/`transfers_out` are **period flows** for an event. Performance measures (minutes, goals, BPS, points) are final-event facts once gameweek data is checked. FDR is a fixture attribute that can change before kick-off.

### ERD

```text
season 1--* gameweek 1--* fact_player_gameweek *--1 dim_player *--1 dim_team
                                  |                         *--1 dim_position
                                  +--* fact_fixture --1 home_team (dim_team)
                                                   --1 away_team  (dim_team)
dim_manager 1--* fact_manager_gameweek *--1 gameweek
dim_manager 1--* bridge_manager_player *--1 dim_player; *--1 gameweek
dim_manager 1--* fact_manager_transfer --1 player_in/out; --1 gameweek
dim_classic_league 1--* fact_league_standing *--1 dim_manager
```

### Relational and star schemas

The implemented core tables are `dim_player`, `dim_team`, `dim_position`, `dim_gameweek`, `fact_fixture`, `fact_player_gameweek`, `player_form`, `player_value`, and `fixture_run`. Add `dim_manager`, `fact_manager_gameweek`, `bridge_manager_player`, `fact_manager_transfer`, `dim_classic_league`, and `fact_league_standing` when manager/league scopes are supplied.

For analytics, use a star with `fact_player_gameweek` at the centre joined to player/team/position/gameweek dimensions; a second star centres `fact_fixture` with home- and away-team role-playing dimensions; manager performance uses `fact_manager_gameweek` plus the roster bridge. Keep price and ownership snapshots keyed by `(player_id, gameweek_id, captured_at)` if intra-gameweek analysis matters.

```text
website API -> data/raw (immutable JSON envelope) -> typed pandas frames
 -> data/processed/*.parquet + DuckDB tables -> player_form/player_value/fixture_run
 -> dashboards, feature store, forecasts/optimisers
```

## Stage 2 — Endpoint map

All calls below are GET, public for the usual public records, and have no API key. Authentication/privacy behaviour and response fields can change. Avoid aggressive polling: this client spaces requests by 0.4 seconds, retries 429/5xx with exponential backoff, and defaults to 20-second timeouts.

| Endpoint pattern | Main structure / keys | Refresh and history |
|---|---|---|
| `/api/bootstrap-static/` | `elements[id]`, `teams[id]`, `element_types[id]`, `events[id]` | current snapshot; refresh 6-hourly |
| `/api/fixtures/` | fixture list, `id`, `event`, `team_h`, `team_a` | current-season fixture list; refresh 6-hourly/pre-deadline |
| `/api/event/{event}/live/` | `elements[id].stats` and explain data | poll during active event; archive final response |
| `/api/element-summary/{element}/` | `history`, `history_past`, `fixtures` | player-level current/historical career-style summary; fetch selectively |
| `/api/entry/{entry}/` | manager metadata/current event | public entry detail; changeable |
| `/api/entry/{entry}/history/` | `current`, `past`, `chips` | historical event rows for that entry |
| `/api/entry/{entry}/event/{event}/picks/` | `picks`, `entry_history`, `active_chip` | fetch after deadline/finalisation |
| `/api/entry/{entry}/transfers/` | transfer list | append-style history |
| `/api/leagues-classic/{league}/standings/?page_standings=N` | league + paginated `standings.results` | standings snapshot, page until exhausted |

Foreign-key mappings: `elements.team -> teams.id`, `elements.element_type -> element_types.id`, fixture `event -> events.id`; picks `element -> elements.id`; history/picks `event -> events.id`. The paths above are inferred from the FPL website’s public API, not official developer documentation. The official FPL rules/help material documents the game; it does not promise these endpoints. For backfilling seasons, load archived raw extracts from a versioned third-party/open dataset only after separately recording provenance and licence; current API calls do not provide a universal season parameter.

## Stage 3 — Installation and first run

```bash
cd /absolute/path/to/fpl_data_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m fpl_pipeline.pipeline --event 1
pytest
duckdb data/fpl.duckdb "SELECT player_id, avg(total_points) AS average_points FROM fact_player_gameweek GROUP BY 1 ORDER BY 2 DESC;"
```

Use `--force` to ignore freshness windows. The first run writes e.g. `data/raw/season=2026-27/endpoint=bootstrap-static/date=.../*.json`, Parquet table files under `data/processed`, and queryable DuckDB tables in `data/fpl.duckdb`.

Project layout: `api` contains isolated endpoint paths and the client; `ingestion` contains raw envelopes and idempotent freshness decisions; `transformation` flattens and validates data; `validation` contains reusable data-quality assertions; `storage.py` writes local analytical stores; `pipeline.py` orchestrates; `tests` holds unit tests. Notebooks are intentionally downstream consumers, never ingestion owners.

## Stage 4 — Transformation and quality controls

`bootstrap_tables` explicitly selects/renames schema fields, parses deadline timestamps as UTC, retains prices as integer tenths, checks primary keys, and verifies player team/position references. `fixture_table` produces the fixture fact, and `player_gameweek_table` creates one live/final aggregate row per player/event. `manager_picks_table` produces the manager snapshot and roster bridge; it is a bridge because a pick is a relationship, not an independently additive measure.

Quality gates fail fast on missing required columns, duplicate composite keys, invalid foreign keys, and negative measures. Extend `checks.py` with expectations for non-null IDs, allowed positions, price bounds, minutes >90 (permit double gameweeks only when fixture-level attribution supports it), and expected gameweek continuity. Alert rather than reject late-changing live data.

## Stage 5 — Incremental operation

Raw storage is append-only and content-hashed; it never overwrites source evidence. The raw store finds the most recent endpoint snapshot. Bootstrap and fixtures are reused inside their configured 6-hour TTL; live event calls use a 15-minute TTL; `--force` deliberately refreshes. Processed DuckDB tables are replaceable materialisations built from selected raw snapshots, so reruns are idempotent. In production, promote a gameweek only when `finished` and `data_checked` are true, append its final partition, and only overwrite provisional/current-event partitions. Persist a small `ingestion_runs` manifest (endpoint, parameters, source hash, status, watermark) to resume individual failures and distinguish retryable partial failures from successful work.

## Stage 6 — Derived tables and ML safety

`player_form` exposes rolling 3/5/10-event points and rolling minutes/goals/assists/bonus. `player_value` creates points-per-million, and `fixture_run` calculates next 1/3/5 fixture difficulty per team. `player_opportunity` should join **lagged** form, minutes, fixture run, price/ownership snapshot, and transfer momentum only as known before an event deadline. For a target at event *t*, build features from observations at or before *t-1* (or a deadline timestamp), split train/test chronologically, and version raw snapshot timestamps. Never train on final live totals, post-deadline ownership, future fixture changes, or revised injury news.

## Storage choices

Parquet is portable, compressed and ideal for immutable partitioned raw/processed data. DuckDB is the implemented local SQL engine: zero-server, fast analytical joins, and reads Parquet directly. SQLite is simpler but less suited to columnar scans. In production choose PostgreSQL for transactional scoped manager data, BigQuery/Snowflake for managed warehouse scale, or Databricks for lakehouse/Spark workflows. Retain raw JSON object storage in every option.

## Tests and next extensions

Run `pytest` for raw-store/incremental behaviour, transformation flattening, duplicate-key/FK validation, and malformed-JSON client behaviour. Add recorded fixtures for every endpoint, mocked 429/5xx retry timing, schema-contract snapshots, and integration tests against a temporary DuckDB database. Add a manager/league orchestrator that reads explicit IDs from environment/config; never crawl entries or leagues indiscriminately.

## Stage 7 — Five-season historical backfill and Bayesian recommendations

The project now backfills `2021-22` through `2025-26` from the public [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) archive. The configured source revision is the immutable commit `f2090d378ebd1b0c3d14884770dde95f38c50a0d`, never a moving branch name. Each raw CSV lives below `data/raw/season=<season>/source=vaastav/`; the adjacent `.metadata.json` includes the exact URL, commit, UTC ingestion time, and SHA-256. Reruns reuse those files unless `--force` is given.

```bash
source .venv/bin/activate
python -m fpl_pipeline.historical_backfill
python -m fpl_pipeline.pipeline --event 1 --force
python -m fpl_pipeline.recommend --horizon 5
python -m fpl_pipeline.optimise --objective balanced
```

`historical_backfill` writes `fact_player_gameweek_historical` and `fact_player_season`. The source is fixture-grain; double-gameweek fixture rows are aggregated to one player/gameweek record. Performance and transfer flows are summed; price and ownership are retained as end-of-gameweek snapshots. Historical player IDs are **not** used across seasons. The archive’s `code` becomes `stable_player_id` when available; a fallback name/position key is explicitly flagged `low` identity confidence.

`fact_player_season.season_ppm` is `total_points / average_gameweek_price_million`. Average gameweek price is preferable to a starting or closing price because it is the price representative of the period in which points were produced. It remains a value signal, not a direct forecast.

### Bayesian weighting

For each player-season observation the model calculates `weight = exp(-0.45 * years_ago)` and exposure `weight * minutes / 90`. It estimates a position-specific PPM prior using exposure-weighted, 2–98% winsorised historical PPM. The posterior is:

```text
(prior_90s × position_prior_ppm + Σ(weighted_90s × capped_season_ppm))
/ (prior_90s + Σ(weighted_90s))
```

`prior_90s` is configured as 12. Thus a player with a few high-scoring minutes is pulled strongly toward their position prior, while established players’ evidence dominates. `posterior_uncertainty` is the position PPM dispersion divided by the square root of the prior-plus-exposure; `reliability_score` is effective exposure divided by prior-plus-exposure. This is transparent empirical-Bayes shrinkage, rather than a claim of a fully specified generative Bayesian football model.

### Recommendations, leakage, and optimiser

`recommend` joins the posterior with the current bootstrap snapshot, current availability/ownership/transfers, and next-horizon fixture difficulty. It creates 1/3/5 gameweek `player_projection` rows. Its baseline is posterior season PPM × current price / 38, adjusted by availability, fixture difficulty, a conservative minutes/reliability factor, and observed current form only when supplied. It automatically selects the next unfinished gameweek for fixture runs; override it with `--target-gameweek N`. For historical simulations use `--as-of-gameweek 10`: the implementation filters observed player performance to gameweeks strictly earlier than 10. Snapshot timestamping remains required for availability, ownership, transfer, and fixture-revision features.

`optimise` solves a PuLP mixed-integer program with exactly 2 goalkeepers, 5 defenders, 5 midfielders, 3 forwards, maximum £100m (1000 tenths), and maximum three players per club. It also selects a valid XI, captain, vice-captain, and ranked bench. Objectives are `points`, `value`, `balanced`, `risk_adjusted`, and `differential`. The default balanced objective rewards projected starter/captain points and a small posterior-value premium while penalising uncertainty. PPM alone is not an adequate objective: a squad must also fit positional, budget, club, minutes, and captaincy constraints.

Detailed table keys and grains are in [docs/database_schema.md](docs/database_schema.md).

### Limitations and sensible next work

The historical archive is third-party and its schema/update cadence can change; the pinned revision protects reproducibility but does not make it official. Current pre-season form is naturally zero, and the baseline does not yet model transfers, team changes, expected goals, expected assists, exact fixture count, or dynamic starting probability. Before trusting projections competitively, add an as-of-date feature store, xG/xA and team-strength data, player-level minutes forecasts, walk-forward backtests, calibration metrics, and a Poisson/hierarchical scoring model. Treat generated squads as decision support, not automatic advice.
