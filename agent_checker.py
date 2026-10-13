"""
agent_checker.py

A narrow, BOUNDED agentic fact-checker for a single claim.

Instead of your code always calling search -> fetch -> answer in a
fixed order, this hands a local Ollama model two tools:
    - search_web(query)
    - fetch_page(url)
and lets the model itself decide, turn by turn, whether to search,
reformulate and search again, fetch a specific page to read more than
the snippet, or stop and give a final verdict. Your code's job shrinks
to: run whatever tool the model asks for, hand back the result, repeat.

Bounded on purpose: a hard `max_steps` cap prevents runaway loops (a
real risk with agent loops), and any malformed/missing final answer is
treated as failure rather than retried indefinitely. If the model
never produces a valid final answer within the cap, or Ollama isn't
reachable at all, this returns None -- the caller (app.py) then falls
back to the fixed pipeline in fact_checker.py, so a bad agent run never
breaks the app, it just degrades to the non-agentic path.

Requires a TOOL-CALLING-capable Ollama model. Reasonably reliable:
    ollama pull llama3.1:8b     (recommended)
    ollama pull qwen2.5:7b
Smaller models like llama3.2 (3B) support tool calls too, but are
noticeably less reliable at formatting them correctly -- expect more
frequent fallback to the fixed pipeline with a small model.

No paid APIs -- Ollama and every tool here runs entirely locally.
"""

import json
import re

import requests

from web_search import search_web as _search_web
from fact_checker import fetch_page_text as _fetch_page_text
from timing import timed

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_PING_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"
MAX_STEPS = 4
REQUEST_TIMEOUT = 60
PING_TIMEOUT = 1.5

VALID_VERDICTS = {"True", "False", "Misleading", "Unverified"}

SYSTEM_PROMPT = """You are a careful, neutral fact-checking agent. You will be given a single factual claim to verify.

You have two tools:
- search_web(query): searches the live web and returns titles, URLs, and short snippets.
- fetch_page(url): fetches and returns the main text of a specific web page.

Investigate the claim using these tools. You may search more than once -- for example, if your first search returns irrelevant results, try different or more specific search terms. You may fetch a specific page to read more than its snippet. Do NOT use your own prior/trained knowledge to decide the verdict -- base it only on what you retrieve with these tools during this conversation. Stop investigating once you have enough to decide, or once you've made a reasonable number of attempts and the evidence just isn't out there.

When you are ready to give your final answer, respond with ONLY valid JSON (no markdown fences, no tool call, no other text) in exactly this shape:
{"verdict": "True|False|Misleading|Unverified", "summary": "one concise sentence under 35 words explaining the verdict, referring to what you actually found", "sources": [{"title": "...", "url": "..."}]}

Verdict meanings:
- "True": what you found clearly supports the claim.
- "False": what you found clearly contradicts the claim.
- "Misleading": what you found shows the claim is only partly accurate, exaggerated, or misattributed.
- "Unverified": you could not find enough relevant, on-topic information to decide, even after investigating.

Include only sources you actually retrieved with your tools this conversation, using their real URLs. Never invent a URL."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the live web for a query. Returns a list of results, each with a title, url, and short snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch the main text content of one specific web page, given its exact URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The exact URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
]


def _ollama_available() -> bool:
    try:
        requests.get(OLLAMA_PING_URL, timeout=PING_TIMEOUT)
        return True
    except Exception:
        return False


def _run_tool(name: str, arguments: dict, log: list, timing: dict = None) -> str:
    """Execute a tool call locally and return its result as a JSON string."""
    if name == "search_web":
        query = str(arguments.get("query", "")).strip()
        log.append(f'searched: "{query}"')
        if not query:
            return json.dumps({"error": "empty query"})
        try:
            with timed(timing, "agent_search"):
                results = _search_web(query, max_results=5)
        except Exception as e:
            return json.dumps({"error": f"search failed: {e}"})
        return json.dumps(results)

    if name == "fetch_page":
        url = str(arguments.get("url", "")).strip()
        log.append(f"fetched: {url}")
        if not url:
            return json.dumps({"error": "empty url"})
        with timed(timing, "agent_fetch_page"):
            text = _fetch_page_text(url, max_chars=2500)
        if not text:
            return json.dumps({"error": "could not fetch or parse this page"})
        return json.dumps({"url": url, "text": text})

    return json.dumps({"error": f"unknown tool: {name}"})


def _parse_final_answer(content: str):
    cleaned = re.sub(r"^```(json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None

    verdict = parsed.get("verdict")
    summary = (parsed.get("summary") or "").strip()
    sources = parsed.get("sources") or []
    if verdict not in VALID_VERDICTS or not summary or not isinstance(sources, list):
        return None

    clean_sources = [
        {"title": s.get("title") or s.get("url", "source"), "url": s.get("url", "")}
        for s in sources
        if isinstance(s, dict) and s.get("url")
    ]
    return {"verdict": verdict, "summary": summary, "sources": clean_sources}


def investigate_claim(claim: str, model: str = DEFAULT_MODEL, max_steps: int = MAX_STEPS, timing: dict = None):
    """
    Runs a bounded agent loop for one claim: the model decides whether
    to call search_web / fetch_page, or to stop with a final verdict.

    Returns a dict {"claim", "verdict", "summary", "sources", "strength", "steps"}
    on success, or None if Ollama is unreachable, a request fails, or the
    model never produces a valid final answer within `max_steps` -- in
    every None case, the caller should fall back to the fixed pipeline.

    Pass a `timing` dict to record wall-clock time under "ollama_ping",
    "agent_llm_calls" (every chat round-trip, accumulated), and
    "agent_search" / "agent_fetch_page" (from the tools themselves,
    accumulated across every call the model makes). See timing.py.
    """
    with timed(timing, "ollama_ping"):
        available = _ollama_available()
    if not available:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Claim to verify: "{claim}"'},
    ]
    tool_log = []
    tool_call_count = 0

    for _step in range(max_steps):
        try:
            with timed(timing, "agent_llm_calls"):
                resp = requests.post(
                    OLLAMA_CHAT_URL,
                    json={"model": model, "messages": messages, "tools": TOOLS, "stream": False},
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return None

        message = data.get("message", {}) or {}
        tool_calls = message.get("tool_calls") or []

        if tool_calls:
            messages.append(message)
            for call in tool_calls:
                fn = call.get("function", {}) or {}
                name = fn.get("name", "")
                args = fn.get("arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                result = _run_tool(name, args, tool_log, timing=timing)
                tool_call_count += 1
                messages.append({"role": "tool", "content": result})
            continue

        content = (message.get("content") or "").strip()
        final = _parse_final_answer(content)
        if final:
            strength = (
                "Strong evidence" if tool_call_count >= 3
                else "Some evidence" if tool_call_count >= 1
                else "Limited evidence"
            )
            return {
                "claim": claim,
                "verdict": final["verdict"],
                "summary": final["summary"],
                "sources": final["sources"],
                "strength": strength,
                "steps": tool_log,
            }

        # Got text that's neither a tool call nor valid final JSON --
        # nudge once rather than looping indefinitely on bad output.
        messages.append({
            "role": "user",
            "content": "That wasn't a tool call or valid JSON in the required shape. "
                        "Either call a tool, or respond with ONLY the JSON object.",
        })

    return None  # exhausted max_steps without a valid final answer
