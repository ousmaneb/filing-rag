| variant | recall@8 | MRR | nDCG@8 | correct | faithful | citation valid | refusal acc | uncited numeric | p50 ms | p95 ms | $/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_baseline | 0.374 | 0.325 | 0.245 | - | - | - | - | - | 22 | 29 | - |
| v1_structural_chunks | 0.353 | 0.267 | 0.241 | - | - | - | - | - | 21 | 27 | - |
| v2_hybrid_rrf | 0.510 | 0.366 | 0.327 | - | - | - | - | - | 49 | 58 | - |
| v3_rerank | 0.547 | 0.410 | 0.334 | - | - | - | - | - | 4461 | 4516 | - |

hit@8 by question type

| variant | comparison | exact_lookup | multi_hop | qualitative | temporal |
|---|---|---|---|---|---|
| v0_baseline | 1.000 | 0.667 | 0.333 | 1.000 | 0.333 |
| v1_structural_chunks | 0.667 | 0.333 | 0.333 | 1.000 | 0.333 |
| v2_hybrid_rrf | 1.000 | 0.333 | 1.000 | 1.000 | 0.667 |
| v3_rerank | 1.000 | 0.667 | 1.000 | 1.000 | 0.667 |
