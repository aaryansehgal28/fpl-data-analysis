# Database schema

The local DuckDB file `data/fpl.duckdb` is a materialised analytical store. All tables are regenerated idempotently from versioned raw source files.

| Table | Grain / key | Purpose |
|---|---|---|
| `dim_player` | current `player_id` | Current FPL player snapshot, including `player_code` stable identity candidate |
| `dim_team`, `dim_position`, `dim_gameweek` | source ID | Current reference dimensions |
| `fact_fixture` | `fixture_id` | Current season fixtures |
| `fact_player_gameweek` | player, current gameweek | Current live/final aggregate performance |
| `fact_player_gameweek_historical` | season, player, gameweek | Five-season historical player-event aggregate; double gameweeks are summed |
| `fact_player_season` | season, `identity_key` | Per-player season summary; PPM = season points / average gameweek price (£m) |
| `player_bayesian_value` | stable/identity key | Recency-weighted posterior PPM and uncertainty |
| `player_projection` | player, horizon | 1/3/5 gameweek baseline projections and intervals |
| `player_recommendation` | current `player_id` | Joined current-state and historical value score |
| `recommended_squad` | current `player_id` | Selected 15, starter/captain/vice/bench flags |
| `recommended_starting_xi`, `recommended_bench` | player | Convenience output subsets |

`stable_player_id` is the historical archive’s FPL `code` where available. `identity_key` falls back to normalised name plus position and is marked `identity_confidence='low'`; low-confidence records should not be used for identity-sensitive research without review.
