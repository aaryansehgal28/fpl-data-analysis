"""Unofficial, public FPL website endpoint paths. Keep paths isolated here."""

BOOTSTRAP = "bootstrap-static/"
FIXTURES = "fixtures/"

def event_live(event: int) -> str: return f"event/{event}/live/"
def element_summary(element: int) -> str: return f"element-summary/{element}/"
def entry(entry: int) -> str: return f"entry/{entry}/"
def entry_history(entry: int) -> str: return f"entry/{entry}/history/"
def entry_picks(entry: int, event: int) -> str: return f"entry/{entry}/event/{event}/picks/"
def entry_transfers(entry: int) -> str: return f"entry/{entry}/transfers/"
def league_classic(league: int, page: int = 1) -> str: return f"leagues-classic/{league}/standings/?page_standings={page}"
