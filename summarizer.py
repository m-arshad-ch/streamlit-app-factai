"""
summarizer.py

Uses a local, free LLM served by Ollama (https://ollama.com) to decide
a claim's verdict AND write a short explanation, grounded ONLY in the
retrieved web excerpts -- not the model's own trained-in knowledge.

Deliberately does verdict + explanation in ONE call: asking an LLM to
"explain why this claim is {verdict}" after the verdict is already
decided elsewhere causes it to rationalize a wrong verdict rather than
catch it. Letting it read the raw evidence and decide fresh avoids that.

If Ollama isn't installed or running, `adjudicate_claim` returns
(None, None) so the caller can fall back to the local NLI-based method
in fact_checker.py -- the app still works without Ollama, just with a
coarser, template-based explanation instead of a written one.

Setup (optional but recommended):
    1. Install Ollama: https://ollama.com/download
    2. Pull a model once:   ollama pull llama3.2
       (a larger model, e.g. `ollama pull llama3.1:8b`, reasons better)
    3. Leave Ollama running in the background (it serves a local API at
       http://localhost:11434 automatically).
"""

import json
import re

import requests

from timing import timed

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_PING_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
REQUEST_TIMEOUT = 45
PING_TIMEOUT = 1.5

VALID_VERDICTS = {"True", "False", "Misleading", "Unverified"}

ADJUDICATE_PROMPT = """You are a careful, neutral fact-checker. You will be given a claim and short excerpts pulled from live web search results.

Decide a verdict for the claim based ONLY on these excerpts. Do NOT use any outside knowledge you were trained on, even if you believe you already know the answer -- base the verdict strictly on what the excerpts actually say. If the excerpts are irrelevant, off-topic, or don't clearly settle the claim, answer "Unverified" rather than guessing.

Verdict must be exactly one of:
- "True": the excerpts clearly support the claim as stated.
- "False": the excerpts clearly contradict the claim as stated.
- "Misleading": the excerpts show the claim is only partly accurate, exaggerated, or misattributed.
- "Unverified": the excerpts don't contain enough relevant, on-topic information to decide.

Then write ONE concise sentence (under 35 words) explaining the verdict, referring to what the excerpts actually say. Do not restate the claim word-for-word. Do not mention "excerpts" or "sources" -- just state the relevant fact plainly.

Respond with ONLY valid JSON, no markdown fences, no preamble, no trailing text, in exactly this shape:
{{"verdict": "True|False|Misleading|Unverified", "summary": "..."}}

Claim: "{claim}"

Excerpts from search results:
{excerpt_block}

JSON:"""


def _ollama_available() -> bool:
    try:
        requests.get(OLLAMA_PING_URL, timeout=PING_TIMEOUT)
        return True
    except Exception:
        return False


def _fallback_summary(verdict: str, support_count: int, refute_count: int) -> str:
    """Used only when Ollama is unreachable -- a plain templated sentence,
    honest about being a count-based approximation rather than reasoning."""
    if verdict == "True":
        return "Multiple sources corroborate this claim and none of the top results contradict it."
    if verdict == "False":
        return "Multiple sources directly contradict this claim."
    if verdict == "Misleading":
        return "Sources are split -- some support this claim while others contradict it, so it appears only partly accurate."
    return "No source in the top search results could reliably confirm or deny this claim."


def adjudicate_claim(claim: str, excerpts: list, model: str = DEFAULT_MODEL, timing: dict = None):
    """
    excerpts: list of dicts with "title" and "excerpt" (short source text),
    already selected as the most claim-relevant portion of each source.

    Returns (verdict, summary), both decided fresh from the excerpts in
    a single LLM call -- or (None, None) if Ollama is unreachable or the
    call fails, signalling the caller to use the NLI-based fallback.

    Pass a `timing` dict to record wall-clock time under "ollama_ping"
    and "llm_adjudicate" -- see timing.py.
    """
    with timed(timing, "ollama_ping"):
        available = _ollama_available()
    if not available:
        return None, None

    lines = []
    for e in excerpts[:6]:
        text = (e.get("excerpt") or "").strip()
        if not text:
            continue
        lines.append(f'- ({e.get("title", "source")}): "{text}"')
    excerpt_block = "\n".join(lines) if lines else "No usable excerpts were retrieved from search results."

    prompt = ADJUDICATE_PROMPT.format(claim=claim, excerpt_block=excerpt_block)

    try:
        with timed(timing, "llm_adjudicate"):
            resp = requests.post(
                OLLAMA_URL,
                json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            raw = (resp.json().get("response") or "").strip()
        raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        verdict = parsed.get("verdict")
        summary = (parsed.get("summary") or "").strip().strip('"')
        if verdict in VALID_VERDICTS and summary:
            return verdict, summary
    except Exception:
        pass

    return None, None