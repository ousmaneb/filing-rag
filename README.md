# secrag

Question answering over 30 SEC 10-K filings, built around an ablation harness that measures
what each retrieval component is worth. The chatbot is the least interesting part; the table
below is the point.

## Results

18 questions (15 answerable, 3 unanswerable), k = 8. The retrieval and latency columns are
from `make eval-fast`, with latency measured on an Apple-silicon Mac. The generation columns
stay blank until a full `make eval` completes. With 15 answerable questions, one question moves
recall@8 by about 0.07 and each hit@8 cell below by 0.33, so small gaps are noise.

| variant | recall@8 | MRR | nDCG@8 | correct | faithful | citation valid | refusal acc | uncited numeric | p50 ms | p95 ms | $/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_baseline | 0.374 | 0.325 | 0.245 | | | | | | 22 | 29 | |
| v1_structural_chunks | 0.353 | 0.267 | 0.241 | | | | | | 21 | 27 | |
| v2_hybrid_rrf | 0.510 | 0.366 | 0.327 | | | | | | 49 | 58 | |
| v3_rerank | 0.547 | 0.410 | 0.334 | | | | | | 4461 | 4516 | |

hit@8 by question type:

| variant | exact_lookup | comparison | multi_hop | temporal | qualitative |
|---|---|---|---|---|---|
| v0_baseline | 0.667 | 1.000 | 0.333 | 0.333 | 1.000 |
| v1_structural_chunks | 0.333 | 0.667 | 0.333 | 0.333 | 1.000 |
| v2_hybrid_rrf | 0.333 | 1.000 | 1.000 | 0.667 | 1.000 |
| v3_rerank | 0.667 | 1.000 | 1.000 | 0.667 | 1.000 |

Each row changes one thing from the row above it:

| variant | chunks | dense | BM25 + RRF | rerank |
|---|---|---|---|---|
| v0_baseline | fixed 512-token windows, 64 overlap | yes | | |
| v1_structural_chunks | structural | yes | | |
| v2_hybrid_rrf | structural | yes | yes | |
| v3_rerank | structural | yes | yes | yes |

Corpus: AMD, NVDA, INTC, AVGO, TXN, WMT, TGT, COST, HD, LOW, JPM, BAC, WFC, GS, MS. The two
most recent 10-Ks for each, pulled from EDGAR.

## Why this corpus breaks naive vector search

Every filing follows the same legally mandated outline. "Item 7. Management's Discussion and
Analysis" appears in all 30 documents, and again in each table of contents. Text that is nearly
identical across companies embeds to nearly identical vectors, so similarity search returns the
right section of the wrong company with high confidence.

The facts people ask about sit in tables. Flattened to prose, a row like `Data Center 115,186
47,525` no longer says which number belongs to which year, and a fixed-size window can cut the
column headers off entirely.

Questions turn on exact tokens: "$5,872 million", "ASC 606", "CET1". Dense embeddings blur
numbers and identifiers. BM25 matches them exactly, which is the only reason it is in the
pipeline.

Fiscal years don't line up with calendar years or with each other. Walmart and Lowe's both
closed a year on 31 January 2025; Walmart calls it fiscal 2025 and Lowe's calls it fiscal 2024.
The fiscal year here comes from each filing's own iXBRL tag.

Not every filer follows the outline. Intel and Morgan Stanley list Items only in a
cross-reference index, most of JPMorgan's MD&A and financial statements sit outside any Item
heading, and Wells Fargo files its annual report as Exhibit 13. Text outside a recognised Item
is kept as an unlabelled section rather than dropped.

## Architecture

```
EDGAR ─> ingest (6 req/s, cached) ─> parse ─> data/sections/*.json
                                     tables -> markdown
                                     split on Item headings
                                            │
                          ┌─────────────────┴─────────────────┐
                    fixed chunks                      structural chunks
                          │                                   │
                          └──── bge-base-en-v1.5 ─────────────┘
                                        │
                  Qdrant: chunks_fixed, chunks_structural
                  data/chunks/*.jsonl -> BM25, built at startup

question ─> regex filters: ticker, fiscal year
         ─> dense top 50 (Qdrant, filtered) ─┐
         ─> BM25 top 50 (same filter)  ──────┴─> RRF ─> bge-reranker-base ─> top 8
         ─> Claude with numbered excerpts ─> answer with [n] citations, or INSUFFICIENT_CONTEXT
```

## Design decisions

**Structural chunks carry a provenance header.** Chunks stay inside one Item, pack whole
paragraphs up to 512 tokens, keep tables whole, and begin with a line like
`AMD FY2024 10-K — Item 7: Management's Discussion and Analysis`. A bare table of numbers
embeds to a vector that says almost nothing about which company or section it came from.
The exception is a table longer than 512 tokens, about 8% of structural chunks before this
rule and mostly bank financial statements. The embedder only reads the first 512 tokens, so
those are split by rows, with the header row repeated in every piece so no row loses its
column labels.

**Fusion is on rank, not score.** Cosine similarity is bounded and BM25 is unbounded and
corpus-dependent, so a weighted sum of the two needs a normalisation step that is itself a
hyperparameter. Reciprocal Rank Fusion with k = 60 throws the magnitudes away.

**Filters come from regex and apply to both retrievers.** Ticker and fiscal year are pulled
from the question with regex rather than an LLM call, which would add a round trip to every
query for something regex handles in most cases. They become hard Qdrant payload filters, and
BM25 is restricted to the same subset. Without that, hybrid search leaks the wrong company back
in through the sparse side.

**Retrieval and generation are scored separately.** One end-to-end accuracy number can't tell
"the retriever missed the passage" from "the generator misread it", so it can't say what to
fix. Retrieval is scored against hand-written evidence quotes (quotes rather than chunk IDs,
because chunk IDs differ between strategies). Generation is graded by an LLM judge on three
independent binary scores: correct, faithful to the excerpts, and citations valid. An answer
that is right but not supported by what was retrieved scores correct = 1, faithful = 0.

## Refusal is a first-class metric

When the excerpts don't contain the answer, the generator must reply `INSUFFICIENT_CONTEXT`
and say what is missing. Three eval questions can't be answered from the corpus. One asks for
Apple's iPhone revenue: the model knows the figure from pretraining, retrieval has nothing on
Apple, and a system that answers anyway has quietly stopped being a retrieval system.

Refusal accuracy is reported on its own, and a refused answerable question counts as incorrect.
A system that refuses everything gets perfect refusal accuracy and zero correctness, so neither
number can be gamed alone.

## Running it

Needs Python 3.11+, Docker, an EDGAR User-Agent and an Anthropic API key.

```
cp .env.example .env    # SEC_USER_AGENT="Name email@example.com", ANTHROPIC_API_KEY
make setup              # venv and all dependencies
make qdrant             # Qdrant in docker
make ingest             # download and parse the 10-Ks
make index              # chunk with both strategies, embed, load Qdrant
make test lint
```

The eval set needs ground truth before it runs. `eval/questions.yaml` ships with `answer: null`
and no evidence on every question; write the answers and supporting quotes by hand from the
filings. The loader refuses to run until that is done. An eval set labelled by an LLM measures
agreement with that LLM, not correctness.

```
make eval-fast          # retrieval metrics only, no API calls
make eval               # generation and judge too; writes reports/ablation.json and .md
make serve              # API on :8000
```

Open http://localhost:8000 for a page that asks questions against any variant and shows the
answer with its citations linked to the excerpts behind them. The same data is available from
the API. `/search` is retrieval only and free. `/ask` returns the answer, its citations, and the
full retrieval trace: every excerpt the model was given, with its score and its rank from each
retriever. It is capped at 60 calls an hour.

```
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What was NVIDIA Data Center revenue in fiscal 2025?", "variant": "v3_rerank"}'
```

Model IDs come from `SECRAG_GENERATION_MODEL` and `SECRAG_JUDGE_MODEL`, both defaulting to
`claude-opus-5`.

## Known failures

**Short facts inside long structural chunks.** Costco's employee count (q02) sits in a
390-word Item 1 chunk that is mostly business description. Dense search ranks that chunk
46th; BM25 ranks it 4th on the exact figure. The fixed-window baseline found it by luck,
because one of its windows happened to be mostly about headcount. Packing whole paragraphs up
to the token budget trades this case away.

**The reranker scores tables of numbers low.** For NVIDIA's fiscal 2025 Data Center revenue
(q01), the chunk holding the revenue-by-market table ranked 3rd on dense search and 4th after
RRF, and bge-reranker-base moved it to 25th with a score of 0.07. The row was inside the
reranker's 512-token window, and adding the table's caption only raised the score to 0.08.

**Generic wording has nothing to match.** "How did Texas Instruments' revenue change" (q10)
matches most of Item 7. The one sentence stating the change ranks 24th on dense search with
structural chunks and never reaches the top 8 in any variant.

**Reranking costs a lot for what it adds.** Scoring 50 candidates with bge-reranker-base
takes about 4.5 seconds per query, against 20 to 50 ms for the other variants. In this run it
added 0.037 recall@8 and 0.044 MRR over hybrid search.
