"""
claim_extractor.py

Extracts distinct, verifiable factual claims from a passage of text.

Two extraction strategies:
  1. LLM-based (used automatically when Ollama is available): asks a
     local model to read the passage and pull out up to N distinct,
     ATOMIC, individually checkable claims -- e.g. splitting a compound
     sentence like "X built Y to save Z from W" into separate claims
     for the builder, the purpose, and the threat, when each part is
     independently checkable. This is what produces claim breakdowns
     like the reference app's.
  2. spaCy heuristic (automatic fallback if Ollama isn't running):
     coarser -- pulls out whole sentences rather than decomposing them,
     but keeps the app working with zero setup.

No paid APIs, no API keys, either way.
"""

import json
import re

import requests
import spacy

from timing import timed

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_PING_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
PING_TIMEOUT = 1.5
REQUEST_TIMEOUT = 30

_NLP = None

OPINION_MARKERS = [
    "i think", "i believe", "i feel", "in my opinion", "personally",
    "i guess", "i suppose", "it seems", "arguably", "i love", "i hate",
    "i like", "i dislike", "should", "shouldn't", "must be", "i hope",
    "i wish", "in my view", "to me,", "imo", "imho",
]

CLAIM_ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "NORP", "DATE", "PERCENT", "MONEY",
    "CARDINAL", "QUANTITY", "EVENT", "LAW", "FAC", "LOC", "TIME", "ORDINAL",
}

MIN_WORDS = 6
MAX_WORDS = 45

EXTRACTION_PROMPT = """You are a fact-checking assistant. Read the passage below and identify up to {max_claims} distinct, individually verifiable factual claims in it.

Break compound statements into separate ATOMIC claims. For example, "X built Y to save Z from W" should become separate claims about: who built Y, what Y was built to do, and whether W is real/relevant -- whenever each part can be checked independently of the others. Restate each claim as a plain, self-contained sentence a reader could check without needing the rest of the passage.

Skip opinions, predictions, and subjective statements.

Respond with ONLY valid JSON, no markdown fences, no preamble, no trailing text, in exactly this shape:
{{"claims": ["claim one", "claim two", "claim three"]}}

If there are no checkable factual claims, respond with {{"claims": []}}.

Passage:
\"\"\"{text}\"\"\"

JSON:"""


def _ollama_available() -> bool:
    try:
        requests.get(OLLAMA_PING_URL, timeout=PING_TIMEOUT)
        return True
    except Exception:
        return False


def _extract_claims_llm(text: str, model: str, max_claims: int):
    """Returns a list of claim strings, or None if the LLM call failed
    (signals the caller to fall back to the spaCy heuristic)."""
    prompt = EXTRACTION_PROMPT.format(max_claims=max_claims, text=text)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = (resp.json().get("response") or "").strip()
        raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("claims", []), list):
            return None
        claims = parsed.get("claims", [])

        cleaned, seen = [], set()
        for c in claims:
            c = re.sub(r"\s+", " ", str(c)).strip()
            key = c.lower()
            if c and key not in seen:
                seen.add(key)
                cleaned.append(c)
        return cleaned
    except Exception:
        return None


def _get_nlp():
    """Lazily load the spaCy model, downloading it once if missing."""
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        # Avoid downloading a large model as a side effect of a UI click.
        _NLP = spacy.blank("en")
        _NLP.add_pipe("sentencizer")
    return _NLP


def _is_opinion(sentence_lower: str) -> bool:
    return any(marker in sentence_lower for marker in OPINION_MARKERS)


def _extract_claims_spacy(text: str):
    """Whole-sentence heuristic extraction -- the no-setup fallback."""
    nlp = _get_nlp()
    doc = nlp(text)

    claims = []
    seen = set()

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        word_count = len(sent_text.split())
        if word_count < MIN_WORDS or word_count > MAX_WORDS:
            continue
        if sent_text.endswith("?"):
            continue

        lowered = sent_text.lower()
        if _is_opinion(lowered):
            continue

        has_entity = any(ent.label_ in CLAIM_ENTITY_LABELS for ent in sent.ents)
        has_verb = any(tok.pos_ in ("VERB", "AUX") for tok in sent) or bool(
            re.search(r"\b(is|are|was|were|has|have|had|does|did|will|can|contains|produces|causes|founded|published|increased|decreased|won|died|born|located|created|makes|made)\b", lowered)
        )
        has_entity = has_entity or bool(re.search(r"\b[A-Z][a-z]+\b|\b\d+(?:\.\d+)?%?\b", sent_text))

        if has_entity and has_verb:
            normalized = re.sub(r"\s+", " ", sent_text).strip()
            key = normalized.lower()
            if key not in seen:
                seen.add(key)
                claims.append(normalized)

    return claims


def extract_claims(text: str, model: str = DEFAULT_MODEL, max_claims: int = 3, timing: dict = None):
    """
    Return a list of candidate verifiable claim strings extracted from `text`.

    Uses the local Ollama LLM to decompose the passage into atomic,
    self-contained claims when Ollama is available; otherwise falls back
    to whole-sentence spaCy extraction so the app still works.

    Pass a `timing` dict to record wall-clock time under "ollama_ping",
    "claim_extraction_llm", and/or "claim_extraction_spacy" -- see timing.py.
    """
    text = (text or "").strip()
    if not text:
        return []

    with timed(timing, "ollama_ping"):
        available = _ollama_available()

    if available:
        with timed(timing, "claim_extraction_llm"):
            llm_claims = _extract_claims_llm(text, model=model, max_claims=max_claims)
        if llm_claims is not None:
            return llm_claims[:max_claims]

    with timed(timing, "claim_extraction_spacy"):
        return _extract_claims_spacy(text)[:max_claims]
