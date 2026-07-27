import json
from fpl_pipeline.ingestion.raw_store import RawStore
def test_raw_store_is_append_only_and_finds_latest(tmp_path):
    store=RawStore(tmp_path,"test")
    first=store.write("fixtures/",[{"id":1}]); second=store.write("fixtures/",[{"id":2}])
    assert first != second
    assert store.latest("fixtures/") == second
    envelope=json.loads(first.read_text())
    assert envelope["metadata"]["endpoint"] == "fixtures/"
