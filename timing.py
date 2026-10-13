"""
timing.py

A tiny, dependency-free timing utility shared across the pipeline, so
latency can be reported honestly instead of asserted ("feels real-time").

Every stage that does real work (claim extraction, web search, page
fetching, NLI scoring, an Ollama call) records its own wall-clock time
into a plain dict via `timed()`. The eval harness and the Streamlit UI
both just read that dict afterwards -- no logging framework, no global
state, nothing to configure.
"""

import time
from contextlib import contextmanager


@contextmanager
def timed(timing: dict, stage: str):
    """
    Usage:
        timing = {}
        with timed(timing, "search"):
            do_the_search()
        # timing == {"search": 0.42}

    If `stage` already has an entry -- e.g. this runs once per source
    inside a loop -- durations ACCUMULATE rather than overwrite, so a
    stage invoked N times reports its total time across all N calls.
    `timing` may be None, in which case this is a no-op passthrough
    (so callers can skip instrumentation entirely by passing nothing).
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        if timing is not None:
            elapsed = time.perf_counter() - start
            timing[stage] = timing.get(stage, 0.0) + elapsed


def total_seconds(timing: dict) -> float:
    """Sum of all recorded stages -- a quick end-to-end number."""
    return sum(timing.values()) if timing else 0.0


def merge_timings(target: dict, source: dict, prefix: str = ""):
    """Accumulate stage timings, optionally qualifying their names."""
    for name, seconds in (source or {}).items():
        key = f"{prefix}{name}"
        target[key] = target.get(key, 0.0) + seconds
