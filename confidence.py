"""
confidence.py

Adds a 0-100% confidence score to each claim result WITHOUT touching how
verdicts are decided. It only reads the result dict that fact_checker.py /
agent_checker.py already return.

Two modes, chosen automatically:

1. Evidence-based (preferred): if the result carries an "evidence" list of
   {"stance": "entailment|contradiction|neutral", "score": float} items
   (per-source NLI output), the score combines how strong the NLI signal
   is, how much the sources agree, and how many sources back the verdict.

2. Fallback (works with the result dicts as they exist today): uses the
   evidence-strength label, the verdict type, and the number of sources.

IMPORTANT: this is a transparent heuristic, not a calibrated probability.
"85%" means "strong, consistent evidence was found", not "85% likely correct".
"""

STRENGTH_BASE = {
    "Strong evidence": 0.80,
    "Some evidence": 0.60,
    "Limited evidence": 0.30,
}
VERDICT_FACTOR = {
    "True": 1.0,
    "False": 1.0,
    "Misleading": 0.85,
    "Unverified": 0.60,
}


def _label(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def _clamp(x: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, x))


def _from_evidence(verdict: str, evidence: list):
    support = [float(e.get("score", 0.0)) for e in evidence if e.get("stance") == "entailment"]
    refute = [float(e.get("score", 0.0)) for e in evidence if e.get("stance") == "contradiction"]
    s, r = len(support), len(refute)

    if verdict in ("True", "False"):
        aligned = support if verdict == "True" else refute
        opposed = r if verdict == "True" else s
        if not aligned:
            return None
        mean_score = sum(aligned) / len(aligned)
        agreement = len(aligned) / (len(aligned) + opposed)
        coverage = min(1.0, len(aligned) / 3)
        score = 0.5 * mean_score + 0.3 * agreement + 0.2 * coverage
        basis = f"{len(aligned)} source(s) support this verdict, {opposed} oppose it"
        return score, basis

    if verdict == "Misleading":
        if s == 0 or r == 0:
            return None
        balance = min(s, r) / max(s, r)
        mean_score = (sum(support) + sum(refute)) / (s + r)
        coverage = min(1.0, (s + r) / 4)
        score = 0.4 * mean_score + 0.3 * balance + 0.3 * coverage
        basis = f"sources are split: {s} support, {r} contradict"
        return score, basis

    # Unverified
    score = 0.35 if (s + r) else 0.20
    return score, "no source clearly settled the claim"


def compute_confidence(result: dict) -> dict:
    """Return {"score": int 0-100, "label": High|Medium|Low, "basis": str}."""
    verdict = result.get("verdict", "Unverified")

    evidence = result.get("evidence")
    if isinstance(evidence, list) and evidence:
        computed = _from_evidence(verdict, evidence)
        if computed is not None:
            score, basis = computed
            score = _clamp(score)
            return {"score": round(score * 100), "label": _label(score), "basis": basis}

    base = STRENGTH_BASE.get(result.get("strength", "Limited evidence"), 0.30)
    score = base * VERDICT_FACTOR.get(verdict, 0.60)
    n_sources = len(result.get("sources") or [])
    if n_sources >= 3:
        score += 0.05
    score = _clamp(score)
    basis = f"{result.get('strength', 'Limited evidence').lower()} across {n_sources} source(s)"
    return {"score": round(score * 100), "label": _label(score), "basis": basis}
