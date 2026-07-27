import pandas as pd
from fpl_pipeline.transformation.historical import player_season_aggregates, standardise_historical_season

def test_historical_standardisation_uses_code_as_stable_identity(tmp_path):
    merged=tmp_path/"merged.csv"; players=tmp_path/"players.csv"
    merged.write_text("name,position,team,element,GW,minutes,total_points,value,starts\nA,DEF,X,4,1,90,6,50,1\n")
    players.write_text("id,code,birth_date,element_type\n4,999,1990-01-01,2\n")
    actual=standardise_historical_season("2024-25",merged,players)
    assert actual.loc[0,"stable_player_id"] == 999
    assert actual.loc[0,"identity_confidence"] == "high"
    assert actual.loc[0,"price_million"] == 5

def test_player_season_ppm_uses_average_gameweek_price():
    x=pd.DataFrame({"season":["2024-25","2024-25"],"player_id":[1,1],"stable_player_id":[4,4],"identity_key":["4","4"],"identity_confidence":["high","high"],"position":["DEF","DEF"],"player_name":["A","A"],"team":["X","X"],"gameweek_id":[1,2],"minutes":[90,90],"total_points":[5,5],"goals_scored":[0,0],"assists":[0,0],"clean_sheets":[1,1],"bonus":[0,0],"price_million":[5,6],"starts":[1,1]})
    assert player_season_aggregates(x).loc[0,"season_ppm"] == 10/5.5
