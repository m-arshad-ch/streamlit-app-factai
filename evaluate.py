"""Run a small labeled NLI fixture evaluation and latency measurement."""
import json
import statistics
import time
from pathlib import Path

from fact_checker import load_nli_model, _classify_pair


def main():
    rows = json.loads(Path(__file__).with_name("evaluation_set.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    tokenizer, model = load_nli_model()
    load_seconds = time.perf_counter() - started
    correct = 0
    latencies = []
    for row in rows:
        started = time.perf_counter()
        scores = _classify_pair(row["premise"], row["claim"], tokenizer, model)
        latencies.append(time.perf_counter() - started)
        correct += max(scores, key=scores.get) == row["label"]
    print(json.dumps({
        "examples": len(rows), "accuracy": round(correct / len(rows), 3),
        "correct": correct, "model_load_seconds": round(load_seconds, 3),
        "median_pair_latency_seconds": round(statistics.median(latencies), 3),
        "mean_pair_latency_seconds": round(statistics.mean(latencies), 3),
        "note": "Synthetic, hand-labeled NLI fixtures; not a live retrieval or end-to-end fact-check evaluation."
    }, indent=2))


if __name__ == "__main__":
    main()
