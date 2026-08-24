# Selective reranking confidence plan

## Question

Can the compact retriever identify queries where an added reranking pass is
likely to help enough to justify its cost?

## Phase 1: diagnose retrieval, development split only

Run the frozen compact seed-43 checkpoint over the frozen clean development
split and full 109,976-entity KB.  Save per-query data for:

1. Fused top-1/top-2 margin and top-20 score shape (top-1 minus top-20,
   standard deviation, top-20 softmax entropy).
2. Fused, text-only, and vision-only top-20 rankings.  Report top-1 agreement
   and whether the gold entity is in each top-20 set.
3. Horizontal-flip stability: percentage of fused top-1 results unchanged
   after flipping the query image.
4. Retrieval ceiling: Recall@1/5/10/20/50/100.  A top-K reranker cannot
   repair a query when its gold entity is outside that K.

The present diagnostic script covers items 1--3 and top-20 recall.  Extend its
single `--topk` value to 100 for the remaining ceiling curve.

## Phase 2: establish whether reranking is warranted

Inspect incorrect fused top-1 results, grouped by whether gold is in top-20.

- Low gold-in-top-20: improve first-stage retrieval; reranking top-20 is not
  a solution.
- High gold-in-top-20 plus low fused margin/modal disagreement/instability:
  reranking has a credible opportunity.
- High gold-in-top-20 but high margin and stable/agreed predictions: examine
  calibration or candidate scoring before assuming uncertainty is useful.

## Phase 3: build a cheap gate, then a reranker

Train only on training-derived folds and choose thresholds on development data.
The gate features are score margin, score-shape entropy, modality agreement,
image-present flags, and flip stability.  Its target is *reranker fixes fused
top-1*, not merely *fused top-1 is wrong*.

Start with a linear/logistic gate; compare it with a shallow tree model.  Use
a top-20 cross-encoder or MIMIC-style local interaction reranker before an
LLM.  An LLM may be tested only as a separately costed fallback.

## Phase 4: evaluate the decision policy

On the untouched frozen test split, compare Never rerank, Always rerank top-20,
margin gate, learned gate, and an oracle gate.  Report H@1/MRR, Recall@20,
share of queries reranked, mean/p95 latency, GPU memory, and accuracy versus
cost curves.  Do not tune gate thresholds or reranker parameters on test.
