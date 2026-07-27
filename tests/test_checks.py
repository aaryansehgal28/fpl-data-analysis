import pandas as pd
import pytest
from fpl_pipeline.validation.checks import DataQualityError, require_unique, require_fk
def test_duplicate_composite_key_fails():
    with pytest.raises(DataQualityError): require_unique(pd.DataFrame({"player_id":[1,1],"gameweek_id":[2,2]}),["player_id","gameweek_id"],"fact_player_gameweek")
def test_fk_fails_for_unknown_player():
    with pytest.raises(DataQualityError): require_fk(pd.DataFrame({"player_id":[2]}),"player_id",pd.DataFrame({"player_id":[1]}),"player_id","picks")
