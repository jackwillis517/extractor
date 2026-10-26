"""Run the extractor on every labeled document for a schema and score it."""

import csv
import json
from pathlib import Path
from statistics import mean

from core import get_text, load_prompt
from extract import ExtractionError, extract
from scorers import score_invoice, score_posting

ROOT = Path(__file__).resolve().parent
# Same keys as extract()'s stats, so CSV columns don't depend on whether row 1 failed
FAILED_STATS = {
    "latency_s": 0,
    "tokens_in": 0,
    "tokens_out": 0,
    "tok_per_s": 0,
    "attempts": 0,
    "retried": False,
    "errors": [],
}

EVALS = {
    "job_posting": {
        "labels": "data/job_posting_labels.jsonl",
        "inputs": "data/postings",
        "prompt": "v1",
        "score": score_posting,
    },
    "invoice": {
        "labels": "data/invoice_labels.jsonl",
        "inputs": "data/invoices",
        "prompt": "v1_invoice",
        "score": score_invoice,
    },
}


def _load_jsonl(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _save_csv(rows, path):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _save_jsonl(rows, path):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def run_eval(schema, model, prompt=None):
    cfg = EVALS[schema]
    prompt = prompt or cfg["prompt"]
    prompt_text = load_prompt(prompt)
    rows, preds = [], []
    for label in _load_jsonl(ROOT / cfg["labels"]):
        text = get_text(ROOT / cfg["inputs"] / label["file"], schema)

        try:
            result, stats = extract(text, prompt_text, model, schema)
            pred, error = result.model_dump(), ""
        except (
            ExtractionError,
            ValueError,
        ) as e:  # failed extraction scores 0 instead of stopping the run
            pred, error, stats = None, str(e)[:200], dict(FAILED_STATS)

        scores = cfg["score"](pred, label["expected"])

        rows.append(
            {
                "file": label["file"],
                **scores,
                "mean": mean(scores.values()),
                **stats,
                "error": error,
            }
        )
        preds.append(
            {"file": label["file"], "pred": pred, "expected": label["expected"]}
        )

    tag = f"{schema}_{prompt}_{model}".replace(":", "-")
    _save_csv(rows, ROOT / "results" / f"{tag}.csv")
    _save_jsonl(preds, ROOT / "results" / f"{tag}.preds.jsonl")
    return rows


def summarize(rows):
    ok = [r for r in rows if not r["error"]] or rows
    return {
        "accuracy": round(mean(r["mean"] for r in rows), 3),
        "fail_rate": round(sum(1 for r in rows if r["error"]) / len(rows), 3),
        "latency_s": round(mean(r["latency_s"] for r in ok), 2),
        "tokens/doc": round(mean(r["tokens_in"] + r["tokens_out"] for r in ok)),
        "tok_per_s": round(mean(r["tok_per_s"] for r in ok), 1),
    }
