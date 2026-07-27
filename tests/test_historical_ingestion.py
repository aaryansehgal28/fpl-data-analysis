import json
from unittest.mock import Mock
from fpl_pipeline.ingestion.historical import download_historical_season

def test_historical_download_writes_provenance_and_reuses_file(tmp_path):
    session=Mock(); response=Mock(); response.content=b"name,element,GW\nA,1,1\n"; response.raise_for_status.return_value=None; session.get.return_value=response
    players=Mock(); players.content=b"id,code\n1,99\n"; players.raise_for_status.return_value=None
    session.get.side_effect=[response,players]
    result=download_historical_season(tmp_path,"https://example.test/repo","abc123","2024-25",session=session)
    metadata=json.loads((result["gws/merged_gw.csv"].parent/"gws__merged_gw.csv.metadata.json").read_text())
    assert metadata["source_commit"] == "abc123" and metadata["season"] == "2024-25"
    download_historical_season(tmp_path,"https://example.test/repo","abc123","2024-25",session=session)
    assert session.get.call_count == 2
