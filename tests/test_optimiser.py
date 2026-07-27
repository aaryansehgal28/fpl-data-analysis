import pandas as pd
from fpl_pipeline.optimiser import optimise_squad

def test_optimiser_returns_legal_squad():
    rows=[]; i=1
    for pos,count in {1:4,2:8,3:8,4:6}.items():
        for _ in range(count):
            rows.append({"player_id":i,"player_name":str(i),"team_name":f"T{i%10}","team_id":i%10,"position_id":pos,"current_price_tenths":40 if pos==1 else 50,"projected_points":float(10+i%4),"posterior_ppm":5.,"projection_uncertainty":1.,"recommendation_score":10.,"ownership":10.}); i+=1
    squad, starters, bench=optimise_squad(pd.DataFrame(rows))
    assert len(squad)==15 and len(starters)==11 and len(bench)==4
    assert squad.current_price_tenths.sum() <= 1000
    assert squad.groupby("position_id").size().to_dict() == {1:2,2:5,3:5,4:3}
    assert squad.groupby("team_name").size().max() <= 3
    assert squad.is_captain.sum() == squad.is_vice_captain.sum() == 1
