from unittest.mock import Mock
import pytest
from fpl_pipeline.api.client import FPLApiError, FPLClient
def test_non_json_response_has_context():
    session=Mock(); response=Mock(); response.raise_for_status.return_value=None; response.json.side_effect=ValueError("bad json"); session.get.return_value=response
    with pytest.raises(FPLApiError,match="bootstrap-static"):
        FPLClient(session=session).get_bootstrap_static()
