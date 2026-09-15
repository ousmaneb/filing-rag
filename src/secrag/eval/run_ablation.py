import argparse
import json
import math
import statistics
import time

import anthropic

from secrag.config import REPORTS_DIR, VARIANTS
from secrag.eval import scoring
from secrag.eval.dataset import Question, load
from secrag.eval.metrics import judge
from secrag.generate import answer
from secrag.retrieval import Retriever

K = 8
# A full run appends each finished record here, so a run that dies partway (rate limit, empty
# credit balance) resumes without paying again for calls it already made. Delete the file by
# hand if you change the questions before resuming.
PROGRESS = REPORTS_DIR / "ablation_progress.jsonl"
# USD per million tokens, (input, output). Check against current pricing before quoting costs.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
COLUMNS = [
    ("recall@8", "recall"),
    ("MRR", "mrr"),
    ("nDCG@8", "ndcg"),
    ("correct", "correct"),
    ("faithful", "faithful"),
    ("citation valid", "citation_valid"),
    ("refusal acc", "refusal_ok"),
    ("uncited numeric", "uncited"),
    ("p50 ms", "p50_ms"),
    ("p95 ms", "p95_ms"),
    ("$/query", "cost_usd"),
]


def mean(values: list) -> float:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return statistics.fmean(values) if values else math.nan


def run(
    questions: list[Question], retriever: Retriever, client: anthropic.Anthropic | None
) -> list[dict]:
    done = {}
    if client and PROGRESS.exists():
        rows = [json.loads(line) for line in PROGRESS.read_text().splitlines()]
        done = {(r["variant"], r["id"]): r for r in rows}

    records = []
    for cfg in VARIANTS.values():
        # The first query pays for loading BM25 and warming the GPU; keep it out of latency.
        retriever.retrieve("warm up", cfg)
        for q in questions:
            if (cfg.name, q.id) in done:
                records.append(done[(cfg.name, q.id)])
                print(f"{cfg.name} {q.id} already done", flush=True)
                continue
            record = {"variant": cfg.name, "id": q.id, "type": q.type}
            if client:
                a = answer(q.question, cfg, retriever, client)
                hits, latency_ms = a.hits, a.retrieval_ms + a.generation_ms
                input_price, output_price = PRICES[a.model]
                cost = a.input_tokens * input_price + a.output_tokens * output_price
                record |= {
                    "answer": a.text,
                    "cited": a.cited,
                    "refused": a.refused,
                    "refusal_ok": scoring.refusal_matches(a.refused, q.answerable),
                    "uncited": scoring.uncited_claim_rate(a.text),
                    "cost_usd": cost / 1e6,
                }
                if q.answerable and a.refused:
                    record["correct"] = False
                elif q.answerable:
                    record |= judge(q, a, client)
            else:
                start = time.perf_counter()
                hits = retriever.retrieve(q.question, cfg)
                latency_ms = (time.perf_counter() - start) * 1000

            matches = scoring.relevance([h.chunk for h in hits], q.evidence)
            n = len(q.evidence)
            record |= {
                "latency_ms": latency_ms,
                "recall": scoring.recall_at_k(matches, n, K),
                "hit": scoring.hit_at_k(matches, n, K),
                "mrr": scoring.mrr(matches, n),
                "ndcg": scoring.ndcg_at_k(matches, n, K),
                "retrieved": [
                    {"chunk_id": h.chunk["chunk_id"], "score": h.score, "ranks": h.ranks}
                    for h in hits
                ],
            }
            records.append(record)
            print(f"{cfg.name} {q.id} recall={record['recall']:.2f} {latency_ms:.0f}ms", flush=True)
            if client:
                with open(PROGRESS, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
    return records


def summarize(records: list[dict]) -> tuple[dict, dict]:
    summary, by_type = {}, {}
    for name in VARIANTS:
        rows = [r for r in records if r["variant"] == name]
        latencies = [r["latency_ms"] for r in rows]
        summary[name] = {key: mean([r.get(key) for r in rows]) for _, key in COLUMNS}
        # Refused answerable questions count as wrong, so refusing everything can't score well.
        summary[name]["correct"] = mean(
            [r.get("correct") for r in rows if r["type"] != "unanswerable"]
        )
        summary[name]["p50_ms"] = statistics.median(latencies)
        summary[name]["p95_ms"] = statistics.quantiles(latencies, n=20, method="inclusive")[18]
        by_type[name] = {
            t: mean([r["hit"] for r in rows if r["type"] == t])
            for t in sorted({r["type"] for r in rows})
        }
    return summary, by_type


def cell(value: float, key: str) -> str:
    if math.isnan(value):
        return "-"
    if key.endswith("_ms"):
        return f"{value:.0f}"
    if key == "cost_usd":
        return f"{value:.4f}"
    return f"{value:.3f}"


def markdown(summary: dict, by_type: dict) -> str:
    lines = [
        "| variant | " + " | ".join(label for label, _ in COLUMNS) + " |",
        "|---" * (len(COLUMNS) + 1) + "|",
    ]
    for name, row in summary.items():
        lines.append(f"| {name} | " + " | ".join(cell(row[key], key) for _, key in COLUMNS) + " |")

    types = sorted({t for row in by_type.values() for t in row if t != "unanswerable"})
    lines += ["", f"hit@{K} by question type", "", "| variant | " + " | ".join(types) + " |"]
    lines.append("|---" * (len(types) + 1) + "|")
    for name, row in by_type.items():
        lines.append(f"| {name} | " + " | ".join(cell(row[t], "") for t in types) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true", help="skip every LLM call")
    args = parser.parse_args()

    questions = load()
    client = None if args.retrieval_only else anthropic.Anthropic()
    REPORTS_DIR.mkdir(exist_ok=True)
    records = run(questions, Retriever(), client)
    summary, by_type = summarize(records)

    report = {"summary": summary, "hit_by_type": by_type, "records": records}
    (REPORTS_DIR / "ablation.json").write_text(json.dumps(report, indent=2, default=str))
    (REPORTS_DIR / "ablation.md").write_text(markdown(summary, by_type))
    # A finished run clears its progress, so the next run can't mix in records made
    # against older questions or code.
    PROGRESS.unlink(missing_ok=True)
    print(markdown(summary, by_type))


if __name__ == "__main__":
    main()
