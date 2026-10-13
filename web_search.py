"""
web_search.py

Free, keyless web search using DuckDuckGo's search backend via the
`ddgs` package (formerly `duckduckgo_search`). No API key, no paid tier.

DuckDuckGo will occasionally rate-limit if you search too fast/too
often -- this module retries with backoff and simply returns fewer
results rather than crashing the app.
"""

import time
import re

try:
    from ddgs import DDGS
except ImportError:  # fall back to the older package name
    from duckduckgo_search import DDGS

from timing import timed


def official_office_query(claim: str):
    """Build a targeted official query for supported Indian office claims."""
    claim = (claim or "").strip()
    duration = re.search(r"\b(?:\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|"
                         r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
                         r"eighteen|nineteen|twenty)\s*(?:-\s*)?years?\b", claim, re.IGNORECASE)
    ruler = re.match(
        r"^(?P<person>.+?)\s+(?:(?:has|is)\s+(?:been\s+)?)?(?:ruling|ruled|governing|governed)\b",
        claim,
        re.IGNORECASE,
    )
    if duration and ruler:
        person = re.sub(r"\s+", " ", ruler.group("person")).strip()
        if person and ("india" in claim.lower() or "narendra modi" in person.lower()):
            return f'site:pmindia.gov.in "{person}" "head of government" 2001'

    match = re.match(
        r"^(?P<person>.+?)\s+(?:is|currently\s+is|serves\s+as|currently\s+serves\s+as)\s+"
        r"(?:the\s+)?(?P<office>.+?)[.!?]*$",
        claim,
        re.IGNORECASE,
    )
    if not match:
        return None
    office = match.group("office").lower()
    if "india" not in claim.lower() or not any(
        role in office for role in ("prime minister", "president", "pm")
    ):
        return None
    person = re.sub(r"\s+", " ", match.group("person")).strip()
    role = "Prime Minister" if "prime minister" in office or re.search(r"\bpm\b", office) else "President"
    return f'site:pmindia.gov.in "{person}" "{role}"'


def search_web(query: str, max_results: int = 5, retries: int = 2, timing: dict = None):
    """
    Search the web for `query` and return a list of dicts:
        {"title": str, "url": str, "snippet": str}
    Returns an empty list if the search fails after retries. Search outages
    are treated as missing evidence so the caller can report Unverified.

    Pass a `timing` dict to record how long this took under the
    "search" key (see timing.py) -- useful for latency reporting
    without changing the return value callers already depend on.
    """
    query = (query or "").strip()
    if not query:
        return []

    with timed(timing, "search"):
        last_error = None
        for attempt in range(retries + 1):
            try:
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        results.append({
                            "title": r.get("title", "") or r.get("Title", ""),
                            "url": r.get("href", r.get("url", "")),
                            "snippet": r.get("body", r.get("snippet", "")),
                        })
                return results
            except Exception as e:  # noqa: BLE001 - we want to swallow & retry
                last_error = e
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
        # A transient backend error should not crash the whole Streamlit run.
        # Returning no results makes the downstream verdict explicitly
        # Unverified; logs can be added by callers if diagnostics are needed.
        return []
