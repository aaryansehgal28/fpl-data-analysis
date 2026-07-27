import pandas as pd
from fpl_pipeline.models import bayesian_player_value, build_recommendations

def _stats():
    return pd.DataFrame({"identity_key":["a","a","b","c"],"stable_player_id":[1,1,2,3],"position":["MID"]*4,"season":["2021-22","2025-26","2025-26","2025-26"],"season_ppm":[20,10,10,100],"effective_90s":[20,20,100,0.1]})
def test_recency_and_shrinkage_protect_small_samples():
    result=bayesian_player_value(_stats(),"2026-27",.45,12).set_index("identity_key")
    assert result.loc["a","posterior_ppm"] < 16 # newer lower PPM gets higher weight
    assert result.loc["c","posterior_ppm"] < 40 # 0.1 effective 90s cannot dominate
    assert 0 < result.loc["a","reliability_score"] < 1

def test_as_of_gameweek_excludes_future_observed_form():
    players=pd.DataFrame({"player_id":[1],"player_code":[1],"web_name":["A"],"team_id":[1],"position_id":[3],"current_price":[50],"status":["a"],"chance_of_playing_next_round":[None],"ownership_percent":["1"],"transfers_in_event":[0],"transfers_out_event":[0]})
    teams=pd.DataFrame({"team_id":[1],"team_name":["X"]}); bayes=pd.DataFrame({"stable_player_id":[1],"posterior_ppm":[10.],"reliability_score":[.8],"posterior_uncertainty":[1.]})
    fixtures=pd.DataFrame({"team_id":[1],"gameweek_id":[1],"next_5_fixture_average_difficulty":[3.]})
    observed=pd.DataFrame({"player_id":[1,1],"gameweek_id":[1,10],"total_points":[2,100],"minutes":[90,90]})
    rec,_=build_recommendations(players,teams,bayes,fixtures,observed,as_of_gameweek=10)
    assert rec.loc[0,"recent_form"] == 2
