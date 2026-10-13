# FactAI

A local claim-checker running
entirely on free, local tools — no paid APIs, no API keys, anywhere.

It:
1. Accepts **text**, or an **audio/video file**.
2. Transcribes audio/video to text with **Whisper** (runs on your machine).
3. Extracts **verifiable factual claims** from the text (spaCy heuristics).
4. Searches the web for each claim using **free, keyless DuckDuckGo search**.
5. Uses a **local NLI model** to judge whether top sources support or
   contradict each claim.
6. Writes a short, natural-language explanation for each verdict using a
   **local LLM via Ollama** (optional — falls back to a plain templated
   sentence if Ollama isn't running).

## 1. Prerequisites

- Python 3.9–3.11
- **ffmpeg** installed and on your system PATH (required by Whisper to decode
  audio/video):
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: `choco install ffmpeg` (or download from ffmpeg.org and add to PATH)
- **Ollama** (optional, but recommended for natural-sounding summaries):
  - Download from [ollama.com](https://ollama.com/download) and install it.
  - Pull a small model once: `ollama pull llama3.2`
  - Leave Ollama running in the background — it serves a local API at
    `http://localhost:11434` automatically. No account, no key, nothing
    leaves your machine.
  - If you skip this step, the app still works — verdicts get a simple
    templated sentence instead of an LLM-written one.

## 2. Setup (in VS Code's integrated terminal)

```bash
cd FactAI

# create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# download the spaCy English model (one-time)
python -m spacy download en_core_web_sm
```

## 3. Run the app

```bash
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`).
Open the `FactAI` folder in VS Code, and you can run the same command
from VS Code's built-in terminal (`` Ctrl+` `` / `` Cmd+` ``).

## 4. First run notes

- The **Whisper** model (default: `base`, ~140 MB) and the **NLI** model
  (`cross-encoder/nli-deberta-v3-xsmall`, ~140 MB) are downloaded
  automatically from their public model hubs the first time you use each
  feature. This needs internet access once — after that, both run fully
  offline/locally.
- The **Ollama** model (`llama3.2` by default, ~2 GB) is downloaded once by
  `ollama pull`, then runs fully offline.
- No API key is ever required, and nothing is sent to a paid service —
  only the free DuckDuckGo search and public web pages are fetched live.

## 5. Usage

1. Paste text in the input box. To transcribe media, click the **＋** below
   the left side of the box, choose **Audio** or **Video**, and upload a file.
2. Click **Try an example** to see it in action, or paste/upload your own.
3. Click **Check this**.
4. Each extracted claim appears as a card with:
   - A stamped verdict: **True**, **False**, **Misleading**, or **Unverified**
   - A one-sentence, plain-English explanation of *why* — written by the
     local LLM, grounded only in what was actually retrieved
   - An evidence-strength note (**Strong evidence**, **Some evidence**, or
     **Limited evidence**) instead of a raw confidence number
   - Clickable source chips for every page that was checked

## 6. Tuning

Use the sidebar to adjust:
- **Whisper model size** — bigger = more accurate transcription, slower.
- **Sources per claim** — how many search results to check per claim.
- **Minimum model confidence** — how confident the NLI model must be before a
  source counts as supporting or contradicting a claim (this stays internal —
  it only affects the verdict logic, it's never shown as a raw number).
- **Local Ollama model** — swap in any model you've pulled (e.g. `llama3.2`,
  `qwen2.5:3b`, `phi3`) to trade off summary quality vs. speed.

## 7. Limitations (please read)

- This is a **heuristic** tool, not a definitive fact-checker. Treat verdicts
  as a starting point for your own judgment, especially for high-stakes claims.
- Claim extraction is rule-based (spaCy) rather than LLM-based, so it can miss
  subtle claims or occasionally flag borderline sentences.
- The generated summary is only as good as what was retrieved — it's
  instructed to use *only* the retrieved evidence, not outside knowledge, so
  a claim with poor search results will get a vague explanation rather than
  a confident but ungrounded one.
- DuckDuckGo's free search endpoint can rate-limit if you run many searches
  in quick succession — if a search fails, wait a bit and try again.
- Some websites block scraping or are JavaScript-heavy, in which case the
  app falls back to the short search-result snippet instead of the full
  article text — this can reduce accuracy for those sources.
- "Real-time" performance depends on your machine: transcription, NLI
  inference, and (if enabled) LLM summary generation all scale with your
  CPU/GPU and the size of the models you choose.

## 8. Evaluation and selected verdict strategy

The fixed pipeline's per-source NLI classification and deterministic evidence
aggregation are the **primary verdict strategy**. The Ollama agent mode is
experimental: it can decide how to search and can produce its own verdict, but
it has not been compared on a sufficiently large, independently labeled set.
It is therefore opt-in and falls back to the fixed pipeline if it fails. The
LLM-generated summary only explains the fixed verdict; it does not choose it.

`evaluation_set.json` contains 12 hand-labeled premise/claim pairs (4 entailment,
4 contradiction, 4 neutral). Run `python evaluate.py` to score the installed
NLI model. On the project environment used for this report, it classified 10
of 12 correctly (83.3% accuracy). This is a small, synthetic smoke evaluation,
not a benchmark of real-world fact-checking; the fixtures are deliberately
simple and do not establish generalization or calibrated confidence.

### Latency sample

One local fixture run on October 10, 2026, with a warm NLI model and a local
unavailable page endpoint measured:

| Stage or mode | Observed time |
| --- | ---: |
| NLI model load (cold) | 5.089 s |
| NLI pair scoring (median, 12 fixture pairs) | 0.045 s |
| NLI pair scoring (mean, 12 fixture pairs) | 0.076 s |
| Fixed pipeline sample: page fetch attempt | 2.050 s |
| Fixed pipeline sample: NLI scoring | 0.154 s |
| Fixed pipeline sample: Ollama availability check | 3.010 s |
| Agent mode sample: Ollama availability check / total | 3.019 / 3.020 s |

These are single-run observations, not stable service-level estimates. In the
fixed pipeline sample the summary-stage total includes the Ollama availability
check, so stage values overlap and must not be added to infer wall-clock total.
The app now reports per-claim stage timings in its result cards; `evaluate.py`
reports model-load and pair-scoring timing for repeatable NLI fixture runs.

### Retrieval quality check

Retrieval is the least validated component. DuckDuckGo is a live, keyless
backend whose result set can change, omit authoritative sources, or fail under
rate limits. In the retrieval spot check for this report, the backend returned
no results for the first test query, so there were **0 usable results and no
relevance judgment**; this is a retrieval failure, not evidence that the claim
is false or that the query has no supporting sources. Search failures now
degrade to an empty result set so the checker can return **Unverified** rather
than crash. The app does not currently compute precision/recall or rank
relevance automatically. Any useful retrieval study should record query,
timestamp, returned URLs, whether each URL is on-topic, and whether it contains
evidence for or against the claim; report hit rate and precision separately
from NLI accuracy.

## 9. Tests and diagnostics

Run the parser and verdict-logic regression tests with:

```bash
python -m unittest discover -s tests -v
```

The tests cover agent JSON parsing, malformed responses, source filtering,
verdict aggregation, evidence strength, and timing helpers. They do not make
network requests or require either model to be running.

## 10. Limitations and Threats to Validity

- **Explicit tenure checks:** the fixed pipeline now checks claims that state a
  number of years in a listed office against dated appointment/start evidence
  in retrieved page text or snippets. These claims bypass experimental agent
  verdicting. This is a targeted guardrail, not a general date/number reasoner;
  it depends on retrieval finding the relevant date and does not catch every
  temporal or numeric error.
- **Model scale and coverage:** the NLI model is a small model, and its
  performance on 12 simple synthetic premise/claim pairs cannot predict its
  accuracy on nuanced, temporal, numerical, or domain-specific claims. The
  local Ollama model can also vary substantially in extraction and summary
  quality with its size and prompt-following ability.
- **Retrieval quality:** live search controls what evidence the system sees.
  Missing, stale, duplicated, low-quality, or snippet-only results can lead to
  an unsupported verdict even when the NLI classifier is correct. The small
  spot check above failed to retrieve results and is not a representative
  retrieval benchmark.
- **Prompt injection:** retrieved pages and user text are untrusted input.
  Prompts instruct local LLMs to treat page text as evidence, not instructions,
  but prompt injection is not reliably preventable by prompting alone. The
  agent's tools are bounded to search and page fetch and its loop has a step
  limit; do not treat those constraints as a security guarantee.
- **Evaluation validity:** labels are hand-assigned and the fixture set is
  tiny. There is no independent adjudication, held-out test set, calibration
  analysis, or comparison of agent verdict accuracy. The reported accuracy
  should be read as a smoke-test number only.

## Project structure

```
claim-checker/
├── app.py               # Streamlit UI (FactAI) — run this
├── transcriber.py        # Whisper-based audio/video transcription
├── claim_extractor.py    # spaCy-based verifiable claim extraction
├── web_search.py          # Free DuckDuckGo web search
├── fact_checker.py        # Scraping + local NLI-based verdicts
├── summarizer.py           # Local Ollama LLM summaries (with fallback)
├── requirements.txt
└── README.md
```
