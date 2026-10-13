"""
fact_checker.py

Given a claim and a list of search results, this module:
  1. Fetches and cleans the text of each source page (falls back to
     the search snippet if the page can't be fetched/parsed).
  2. Uses a local NLI model purely to find the single most
     claim-relevant excerpt within each source's text (not to decide
     true/false -- just "which part of this page is even about the
     claim").
  3. Hands the claim + those excerpts to summarizer.adjudicate_claim(),
     which asks a local LLM to decide the verdict AND write the
     explanation together, grounded only in the excerpts. This avoids
     the failure mode of deciding a verdict mechanically first and then
     asking an LLM to rationalize it.
  4. If Ollama isn't available, falls back to a purely NLI-count-based
     verdict (the original, coarser method) with a templated summary,
     so the app still works with zero LLM setup.

The NLI model (cross-encoder/nli-deberta-v3-xsmall) runs entirely
locally via HuggingFace `transformers` -- no paid API, no API key.
It is downloaded once (a few hundred MB) the first time you run the app.
"""

import requests
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from summarizer import adjudicate_claim, _fallback_summary, DEFAULT_MODEL as DEFAULT_SUMMARY_MODEL
from timing import timed

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"
LABELS = ["contradiction", "entailment", "neutral"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

_MODEL_CACHE = {}


def load_nli_model():
    """Load (and cache) the tokenizer + model. Safe to call repeatedly."""
    if "tokenizer" not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        model.eval()
        _MODEL_CACHE["tokenizer"] = tokenizer
        _MODEL_CACHE["model"] = model
    return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"]


def fetch_page_text(url: str, timeout: int = 6, max_chars: int = 4000) -> str:
    """Best-effort fetch + extraction of the main text of a web page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs).strip()
        return text[:max_chars]
    except Exception:
        return ""


def _chunk_words(text: str, chunk_size: int = 180):
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def _classify_pair(premise: str, hypothesis: str, tokenizer, model):
    with torch.no_grad():
        features = tokenizer(
            [premise], [hypothesis],
            padding=True, truncation=True, max_length=256, return_tensors="pt",
        )
        logits = model(**features).logits
        probs = torch.softmax(logits, dim=1)[0]
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}


def score_source_against_claim(claim: str, source_text: str, tokenizer, model):
    """Find the chunk of source_text most relevant to the claim (highest
    entailment OR contradiction signal) and return its label, score, and
    the chunk itself. The label/score are only used by the NLI-only
    fallback path -- the chunk itself is what gets shown to the LLM."""
    best = {"label": "neutral", "score": 0.0, "chunk": ""}
    for chunk in _chunk_words(source_text):
        scores = _classify_pair(chunk, claim, tokenizer, model)
        top_label = max(("entailment", "contradiction"), key=lambda lbl: scores[lbl])
        if scores[top_label] > best["score"]:
            best = {"label": top_label, "score": scores[top_label], "chunk": chunk}
    return best


def _map_verdict(support_count: int, refute_count: int) -> str:
    if support_count > 0 and refute_count > 0:
        return "Misleading"
    if support_count >= 1 and refute_count == 0:
        return "True"
    if refute_count >= 1 and support_count == 0:
        return "False"
    return "Unverified"


def _evidence_strength_from_counts(support_count: int, refute_count: int) -> str:
    strongest = max(support_count, refute_count)
    if strongest >= 3:
        return "Strong evidence"
    if strongest >= 1:
        return "Some evidence"
    return "Limited evidence"


def _evidence_strength_from_excerpts(usable_excerpt_count: int) -> str:
    if usable_excerpt_count >= 3:
        return "Strong evidence"
    if usable_excerpt_count >= 1:
        return "Some evidence"
    return "Limited evidence"


def verify_claim(
    claim: str,
    search_results,
    tokenizer,
    model,
    min_confidence: float = 0.55,
    summarizer_model: str = DEFAULT_SUMMARY_MODEL,
    timing: dict = None,
):
    """
    Check `claim` against a list of search results
    (each a dict with "title", "url", "snippet").

    Returns a dict:
        {
          "claim": str,
          "verdict": "True" | "False" | "Misleading" | "Unverified",
          "strength": str,           # plain-English evidence strength
          "summary": str,            # generated explanation
          "sources": [ {"title", "url"}, ... ]
        }

    Pass a `timing` dict to record wall-clock time per stage --
    "fetch_pages" and "nli_scoring" accumulate across every source;
    "ollama_ping" / "llm_adjudicate" come from adjudicate_claim(); the
    fallback path (Ollama unavailable) adds "nli_fallback_verdict".
    See timing.py.
    """
    evidence = []
    for res in search_results:
        with timed(timing, "fetch_pages"):
            page_text = fetch_page_text(res["url"])
        text_to_check = page_text if page_text.strip() else res.get("snippet", "")
        if not text_to_check.strip():
            continue
        with timed(timing, "nli_scoring"):
            result = score_source_against_claim(claim, text_to_check, tokenizer, model)
        evidence.append({
            "url": res["url"],
            "title": res.get("title") or res["url"],
            "stance": result["label"] if result["score"] >= min_confidence else "neutral",
            "excerpt": result.get("chunk", "")[:400],
        })

    # Primary path: let the LLM read the excerpts and decide fresh.
    excerpts_for_llm = [e for e in evidence if e["excerpt"].strip()]
    verdict, summary = adjudicate_claim(claim, excerpts_for_llm, model=summarizer_model, timing=timing)

    if verdict is not None:
        strength = _evidence_strength_from_excerpts(len(excerpts_for_llm))
    else:
        # Fallback: Ollama unavailable -- use the coarser NLI-count method.
        with timed(timing, "nli_fallback_verdict"):
            support_count = sum(1 for e in evidence if e["stance"] == "entailment")
            refute_count = sum(1 for e in evidence if e["stance"] == "contradiction")
            verdict = _map_verdict(support_count, refute_count) if evidence else "Unverified"
            strength = _evidence_strength_from_counts(support_count, refute_count)
            summary = _fallback_summary(verdict, support_count, refute_count)

    sources = [{"title": e["title"], "url": e["url"]} for e in evidence]

    return {
        "claim": claim,
        "verdict": verdict,
        "strength": strength,
        "summary": summary,
        "sources": sources,
    }