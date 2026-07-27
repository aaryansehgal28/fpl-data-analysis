import json
from fpl_pipeline.transformation.core import player_gameweek_table
def test_live_rows_have_a_unique_player_event_key(tmp_path):
    raw=tmp_path/"live.json"
    raw.write_text(json.dumps({"payload":{"elements":[{"id":7,"stats":{"minutes":90,"total_points":8}}]}}))
    actual=player_gameweek_table(raw,3)
    assert actual.loc[0,["player_id","gameweek_id","minutes","total_points"]].tolist()==[7,3,90,8]
