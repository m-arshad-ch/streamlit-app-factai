"""
app.py

"FactAI" -- a local claim checker styled to match the
reference design. Uses free, local tools throughout:
  - Whisper for audio/video transcription
  - spaCy heuristics for claim extraction
  - DuckDuckGo search (free, keyless)
  - A local NLI model to weigh evidence
  - A local Ollama LLM (optional) to write natural-language summaries

Run with: streamlit run app.py
"""

import os
import re
import html
import hashlib
import tempfile
import time
from urllib.parse import urlparse

import streamlit as st

from transcriber import transcribe_media
from claim_extractor import extract_claims
from web_search import search_web, official_office_query
from fact_checker import verify_claim, load_nli_model
try:
    from fact_checker import is_tenure_duration_claim
except ImportError:  # older fact_checker.py without the tenure guardrail
    _DURATION_RE = re.compile(
        r"\b(?:\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
        r"\s*(?:-\s*)?years?\b", re.IGNORECASE)
    _RULER_RE = re.compile(
        r"^.+?\s+(?:(?:has|is)\s+(?:been\s+)?)?(?:ruling|ruled|governing|governed)\b",
        re.IGNORECASE)

    def is_tenure_duration_claim(claim: str) -> bool:
        claim = (claim or "").strip()
        return bool(_DURATION_RE.search(claim) and _RULER_RE.match(claim))

from confidence import compute_confidence
from summarizer import DEFAULT_MODEL as DEFAULT_SUMMARY_MODEL
from agent_checker import investigate_claim, DEFAULT_MODEL as DEFAULT_AGENT_MODEL
from timing import total_seconds

st.set_page_config(page_title="FactAI", page_icon="🔍", layout="centered")

EXAMPLE = (
    "In her speech, the senator said renewable energy now supplies more than "
    "half of all U.S. electricity generation, and that national unemployment "
    "has fallen below 2% for the first time since World War II."
)

STAGE_LABELS = [
    "Reading the text",
    "Pulling out claims",
    "Searching the web",
    "Weighing the evidence",
]

VERDICT_STYLE = {
    "True": {"color": "#2E7D5B", "rotate": "-3deg"},
    "False": {"color": "#B4423B", "rotate": "4deg"},
    "Misleading": {"color": "#C08A2E", "rotate": "-2deg"},
    "Unverified": {"color": "#767A82", "rotate": "3deg"},
}


def hostname(url: str) -> str:
    try:
        h = urlparse(url).hostname or url
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return url


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&display=swap');

:root {
  --paper: #EFEDE6;
  --paper-2: #E7E3D6;
  --ink: #20222A;
  --ink-soft: #55585F;
  --line: #D3CDBD;
  --accent: #1F5C56;
}

.stApp { background: var(--paper); }
.block-container { max-width: 660px; padding-top: 40px; padding-bottom: 64px; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }

/* Hide Streamlit's own chrome so only the paper design shows */
header[data-testid="stHeader"] { background: var(--paper); box-shadow: none; }
#MainMenu, footer, div[data-testid="stDecoration"] { visibility: hidden; height: 0; }
div[data-testid="stToolbar"] { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--paper-2);
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: var(--ink); }
section[data-testid="stSidebar"] h2 { font-family: 'Source Serif 4', serif; font-size: 20px; }

/* Radio (Text / Audio-Video toggle) */
div[data-testid="stRadio"] label {
  background: var(--paper-2);
  border: 1px solid var(--line);
  padding: 6px 14px;
  margin-right: 8px;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
  border-color: var(--ink-soft) !important;
}
div[data-testid="stRadio"] input:checked + div {
  background-color: var(--accent) !important;
  border-color: var(--accent) !important;
}

/* Sliders */
div[data-testid="stSlider"] [role="slider"] {
  background-color: var(--accent) !important;
  border-color: var(--accent) !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
  background: var(--accent) !important;
}

/* Selectbox / text input */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input {
  background: var(--paper-2) !important;
  border-color: var(--line) !important;
  border-radius: 0 !important;
  color: var(--ink) !important;
}

/* File uploader */
section[data-testid="stFileUploaderDropzone"] {
  background: var(--paper-2) !important;
  border: 1px dashed var(--line) !important;
  border-radius: 0 !important;
}
section[data-testid="stFileUploaderDropzone"] button {
  border-radius: 0 !important;
  border-color: var(--line) !important;
  color: var(--ink) !important;
  background: var(--paper) !important;
}

/* Default horizontal rule + caption, restyled to match the palette */
hr { border-color: var(--line) !important; margin: 32px 0 !important; }
div[data-testid="stCaptionContainer"] { color: var(--ink-soft) !important; }
div[data-testid="stAlert"] { border-radius: 0; }

.desk-title {
  font-family: 'Source Serif 4', serif;
  font-weight: 700;
  font-size: 34px;
  margin: 0 0 6px 0;
  letter-spacing: -0.01em;
  color: #000000;
}
.desk-tagline {
  font-size: 15px;
  color: var(--ink-soft);
  margin: 0 0 20px 0;
  max-width: 58ch;
  line-height: 1.5;
}

div[data-testid="stTextArea"] textarea {
  background: var(--paper-2);
  border: 1px solid var(--line);
  color: var(--ink);
  font-family: 'Inter', sans-serif;
  font-size: 15px;
  line-height: 1.55;
  border-radius: 0;
}
div[data-testid="stTextArea"] textarea:focus { border-color: var(--accent); box-shadow: none; }

div[data-testid="stButton"] button,
div[data-testid="stPopover"] button {
  border-radius: 0; font-family: 'Inter', sans-serif; font-size: 14px;
}
div[data-testid="stButton"] button[kind="primary"] {
  background: var(--accent); color: #F3F1E9; border: none;
  padding: 8px 16px; font-weight: 500;
}
div[data-testid="stButton"] button[kind="primary"]:hover { background: #17453F; color: #F3F1E9; }
div[data-testid="stButton"] button[kind="primary"]:disabled { background: #B9B6AA; color: #F3F1E9; }
div[data-testid="stButton"] button[kind="secondary"] {
  background: none; border: none; color: var(--ink-soft);
  text-decoration: underline; text-underline-offset: 2px; padding: 4px 0;
}
div[data-testid="stButton"] button[kind="secondary"]:hover { color: var(--ink); }

.hint-line { font-size: 13px; color: #A24A3E; margin: 6px 2px 0; }
.status-line { margin-top: 18px; font-size: 14px; color: var(--ink-soft); font-style: italic; }
.error-line { margin-top: 18px; font-size: 14px; color: #B4423B; }
.empty-note { margin-top: 16px; font-size: 13px; color: var(--ink-soft); line-height: 1.5; }

.results-header {
  display: flex; align-items: baseline; justify-content: space-between;
  margin-top: 28px; margin-bottom: 4px;
}
.results-count { font-size: 14px; color: var(--ink-soft); }

.claim-card {
  border-top: 1px solid var(--line);
  padding: 22px 0;
  animation: card-in 0.4s ease both;
}
.claim-card.last { border-bottom: 1px solid var(--line); }
.claim-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; }
.claim-text {
  font-family: 'Source Serif 4', serif;
  font-size: 19px; line-height: 1.4; margin: 0; flex: 1; color: var(--ink);
}
.stamp {
  flex-shrink: 0; border: 2px solid; padding: 4px 10px;
  font-family: 'Inter', sans-serif; font-weight: 600; font-size: 12px;
  letter-spacing: 0.08em; text-transform: uppercase;
  transform: rotate(var(--r));
  animation: stamp-in 0.5s cubic-bezier(0.2, 0.9, 0.3, 1.2) both;
}
.claim-summary { font-size: 14.5px; line-height: 1.6; color: var(--ink-soft); margin: 10px 0 0; max-width: 62ch; }
.claim-meta { display: flex; align-items: center; gap: 8px; margin: 8px 0 0; }
.claim-strength { font-size: 12.5px; color: #8A8779; margin: 0; }
.claim-timing { font-size: 12.5px; color: #B9B6AA; margin: 0; }
.claim-confidence { display: flex; align-items: center; gap: 10px; margin: 12px 0 0; font-size: 12.5px; color: var(--ink-soft); }
.conf-track { flex: 0 0 140px; height: 6px; background: var(--paper-2); border: 1px solid var(--line); }
.conf-fill { height: 100%; }
.conf-pct { font-weight: 600; color: var(--ink); }
.claim-timing::before { content: "\00b7"; margin-right: 8px; }
.claim-sources { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.source-chip {
  display: inline-flex; align-items: center; gap: 5px; font-size: 12.5px;
  color: var(--accent); text-decoration: none; border: 1px solid var(--line);
  padding: 4px 9px;
}
.source-chip:hover { border-color: var(--accent); }

@keyframes stamp-in {
  from { opacity: 0; transform: rotate(var(--r)) scale(1.8); }
  to { opacity: 1; transform: rotate(var(--r)) scale(1); }
}
@keyframes card-in {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
@media (prefers-reduced-motion: reduce) {
  .stamp, .claim-card { animation: none !important; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- state ----
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "status" not in st.session_state:
    st.session_state.status = "idle"  # idle | done | error
if "claims" not in st.session_state:
    st.session_state.claims = None
if "error_msg" not in st.session_state:
    st.session_state.error_msg = ""
if "total_elapsed" not in st.session_state:
    st.session_state.total_elapsed = 0.0
if "last_media_hash" not in st.session_state:
    st.session_state.last_media_hash = ""
if "pending_transcript" in st.session_state:
    st.session_state.input_text = st.session_state.pop("pending_transcript")


def load_example():
    st.session_state.input_text = EXAMPLE


def reset_all():
    st.session_state.input_text = ""
    st.session_state.claims = None
    st.session_state.status = "idle"
    st.session_state.error_msg = ""


# ---------------------------------------------------------------- header ---
st.markdown('<h1 class="desk-title">FactAI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="desk-tagline">Paste a paragraph, a headline, or a whole '
    "article \u2014 or upload audio/video, transcribed locally. It pulls out "
    "what's actually checkable, searches the web, and shows the sources "
    "behind each verdict.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    whisper_size = st.selectbox("Whisper model size (audio/video)", ["tiny", "base", "small"], index=1)
    max_results = st.slider("Sources to check per claim", 2, 8, 5)
    min_confidence = st.slider("Min. model confidence to count as evidence", 0.3, 0.9, 0.55, 0.05)
    summary_model = st.text_input(
        "Local Ollama model for summaries", value=DEFAULT_SUMMARY_MODEL,
        help="Requires Ollama (https://ollama.com) running locally. Falls back to a "
             "templated summary automatically if Ollama isn't available.",
    )
    st.markdown("---")
    agent_mode = st.checkbox(
        "Agentic search (experimental fallback)",
        value=False,
        help="Lets the model decide for itself whether to search, re-search with "
             "different terms, or fetch a specific page, instead of always doing one "
             "fixed search. Slower, and needs a tool-calling model. Falls back to the "
             "normal pipeline automatically per-claim if it fails.",
    )
    agent_model = st.text_input(
        "Agent model (needs tool-calling support)", value=DEFAULT_AGENT_MODEL,
        disabled=not agent_mode,
        help="e.g. llama3.1:8b or qwen2.5:7b. Small models like llama3.2 support tools "
             "but are less reliable at formatting the calls correctly.",
    )

# ----------------------------------------------------------------- input ---
st.button("Try an example", type="secondary", on_click=load_example)
with st.form("claim_check_form", clear_on_submit=False):
    st.text_area(
        "Paste text here", height=150, key="input_text",
        label_visibility="collapsed", placeholder="Paste text here\u2026",
    )
    analyze_clicked = st.form_submit_button("\U0001F50E Check this", type="primary")

attach_col, _ = st.columns([1, 6])
with attach_col:
    with st.popover("＋", help="Upload audio or video to transcribe"):
        media_kind = st.radio(
            "Upload type", ["Audio", "Video"], horizontal=True,
            key="media_kind",
        )
        media_extensions = (
            ["mp3", "wav", "m4a", "ogg", "flac", "aac"] if media_kind == "Audio"
            else ["mp4", "mov", "mkv", "webm", "avi", "mpeg"]
        )
        uploaded = st.file_uploader(
            f"Choose {media_kind.lower()} file", type=media_extensions,
            key=f"media_upload_{media_kind.lower()}",
        )
        if uploaded is not None:
            media_bytes = uploaded.getvalue()
            media_hash = hashlib.sha256(media_bytes).hexdigest()
            if media_hash != st.session_state.last_media_hash:
                suffix = os.path.splitext(uploaded.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(media_bytes)
                    tmp_path = tmp.name
                try:
                    with st.spinner("Transcribing locally with Whisper..."):
                        transcript = transcribe_media(
                            tmp_path, model_size=whisper_size
                        )
                    st.session_state.last_media_hash = media_hash
                    st.session_state.pending_transcript = transcript
                    st.session_state.claims = None
                    st.session_state.status = "idle"
                    st.rerun()
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            else:
                st.caption(f"Ready to transcribe: {uploaded.name}")

text_len = len(st.session_state.input_text.strip())

if 0 < text_len < 30:
    st.markdown(
        '<p class="hint-line">Add a bit more \u2014 at least a sentence or two \u2014 '
        "so there's something to check.</p>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------- pipeline ----
status_placeholder = st.empty()

if analyze_clicked and text_len >= 30:
    st.session_state.error_msg = ""
    st.session_state.claims = None

    try:
        status_placeholder.markdown(f'<p class="status-line">{STAGE_LABELS[0]}\u2026</p>', unsafe_allow_html=True)
        time.sleep(0.4)

        status_placeholder.markdown(f'<p class="status-line">{STAGE_LABELS[1]}\u2026</p>', unsafe_allow_html=True)
        extraction_timing = {}
        candidate_claims = extract_claims(
            st.session_state.input_text, model=summary_model, max_claims=3, timing=extraction_timing,
        )

        if not candidate_claims:
            st.session_state.claims = []
            st.session_state.status = "done"
            st.session_state.total_elapsed = total_seconds(extraction_timing)
        else:
            model_timing = {}
            from timing import timed
            with timed(model_timing, "nli_model_load"):
                tokenizer, nli_model = load_nli_model()
            results = []
            for i, claim in enumerate(candidate_claims):
                verdict = None
                claim_timing = {}

                # Tenure claims need the fixed path's date comparison, and
                # current Indian office claims need its official-source check.
                # Keep both out of the experimental agent path.
                if (
                    agent_mode
                    and not is_tenure_duration_claim(claim)
                    and not official_office_query(claim)
                ):
                    status_placeholder.markdown(
                        f'<p class="status-line">Agent investigating claim {i + 1} of '
                        f'{len(candidate_claims)}\u2026</p>',
                        unsafe_allow_html=True,
                    )
                    verdict = investigate_claim(claim, model=agent_model, timing=claim_timing)
                    # None means: Ollama unreachable, a request failed, or the model
                    # never returned a valid final answer within the step cap --
                    # fall through to the fixed pipeline below for this claim. The
                    # time already spent in the failed agent attempt still counts,
                    # so it stays in claim_timing rather than being reset.

                if verdict is None:
                    status_placeholder.markdown(f'<p class="status-line">{STAGE_LABELS[2]}\u2026</p>', unsafe_allow_html=True)
                    search_results = search_web(claim, max_results=max_results, timing=claim_timing)
                    # A targeted official query improves coverage for current
                    # Indian office-holder claims when general search misses it.
                    official_query = official_office_query(claim)
                    if official_query:
                        official_results = search_web(
                            official_query, max_results=3, timing=claim_timing
                        )
                        seen_urls = {item.get("url") for item in search_results}
                        search_results.extend(
                            item for item in official_results
                            if item.get("url") and item.get("url") not in seen_urls
                        )

                    status_placeholder.markdown(f'<p class="status-line">{STAGE_LABELS[3]}\u2026</p>', unsafe_allow_html=True)
                    verdict = verify_claim(
                        claim, search_results, tokenizer, nli_model,
                        min_confidence=min_confidence, summarizer_model=summary_model,
                        timing=claim_timing,
                    )
                for stage, seconds in model_timing.items():
                    claim_timing[stage] = claim_timing.get(stage, 0.0) + seconds
                verdict["timings"] = {k: round(v, 3) for k, v in claim_timing.items()}
                verdict["elapsed"] = round(total_seconds(claim_timing), 1)
                verdict["via"] = "agent" if agent_mode and "steps" in verdict else "fixed"
                verdict["confidence"] = compute_confidence(verdict)
                results.append(verdict)

            st.session_state.claims = results
            st.session_state.total_elapsed = total_seconds(extraction_timing) + sum(
                r["elapsed"] for r in results
            )

            st.session_state.status = "done"
    except Exception as e:
        import traceback
        traceback.print_exc()  # full traceback goes to your terminal
        st.session_state.status = "error"
        st.session_state.error_msg = f"Something broke: {e}"
    finally:
        status_placeholder.empty()

# -------------------------------------------------------------- results ---
if st.session_state.status == "error":
    st.markdown(f'<p class="error-line">{st.session_state.error_msg}</p>', unsafe_allow_html=True)

if st.session_state.status == "done" and st.session_state.claims is not None:
    claims = st.session_state.claims
    if len(claims) == 0:
        st.markdown(
            '<p class="empty-note">No independently checkable claims turned up in that '
            'text \u2014 it may be mostly opinion, or too vague to verify.</p>',
            unsafe_allow_html=True,
        )
    else:
        n = len(claims)
        st.markdown(
            f'<div class="results-header"><span class="results-count">'
            f'{n} claim{"s" if n != 1 else ""} checked · '
            f'{st.session_state.total_elapsed:.1f}s total</span></div>',
            unsafe_allow_html=True,
        )
        st.button("\u21BA Check another", type="secondary", on_click=reset_all)

        cards_html = ""
        for i, item in enumerate(claims):
            style = VERDICT_STYLE.get(item["verdict"], VERDICT_STYLE["Unverified"])
            sources_html = "".join(
                f'<a href="{html.escape(str(s["url"]), quote=True)}" target="_blank" rel="noopener noreferrer" '
                f'class="source-chip" title="{html.escape(str(s["title"]), quote=True)}">\u2197 {html.escape(hostname(s["url"]))}</a>'
                for s in item.get("sources", [])
            )
            conf = item.get("confidence") or compute_confidence(item)
            conf_color = {"High": "#2E7D5B", "Medium": "#C08A2E", "Low": "#B4423B"}[conf["label"]]
            conf_html = (
                f'<div class="claim-confidence" title="{html.escape(conf["basis"], quote=True)} '
                f'(heuristic score, not a calibrated probability)">'
                f'<span>Confidence</span>'
                f'<div class="conf-track"><div class="conf-fill" '
                f'style="width:{conf["score"]}%;background:{conf_color};"></div></div>'
                f'<span class="conf-pct">{conf["score"]}% \u00b7 {conf["label"]}</span></div>'
            )
            last_class = " last" if i == n - 1 else ""
            cards_html += f"""
            <div class="claim-card{last_class}">
              <div class="claim-top">
                <p class="claim-text">{html.escape(item['claim'])}</p>
                <div class="stamp" style="color:{style['color']};border-color:{style['color']};--r:{style['rotate']};">
                  {item['verdict']}
                </div>
              </div>
              <p class="claim-summary">{html.escape(item['summary'])}</p>
              {conf_html}
              <p class="claim-strength">{item['strength']}</p>
              <p class="claim-strength">Mode: {item.get('via', 'fixed NLI')} · stage timings: {', '.join(f'{k} {v:.2f}s' for k, v in item.get('timings', {}).items())}</p>
              {f'<div class="claim-sources">{sources_html}</div>' if sources_html else ''}
            </div>
            """
        st.markdown(cards_html, unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "Automated, heuristic estimate based on top web search results and a local "
    "language model. Not a substitute for professional fact-checking."
)
