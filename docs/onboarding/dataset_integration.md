# Integrating a New Dataset into RAGtune

RAGtune evaluates retrieval pipelines against benchmarks. Every benchmark must be reduced to three components before it can be used:

| Component | Python type | What it represents |
|---|---|---|
| `doc_iter_fn` | `Callable → generator` | Corpus documents: `{"docno": str, "text": str}` |
| `queries` | `List[{"id": str, "text": str}]` | Test queries |
| `qrels` | `Dict[(query_id, doc_id), int]` | Relevance labels (may be empty for proxy sources) |

The `DatasetConfig` dataclass in `scripts/experiment_grid.py` captures everything about a dataset in one place.

---

## Step 1 — What does your dataset provide?

Use this decision tree to choose `RAGTaskType` and `QrelSource`:

```
Does your dataset have explicit relevance judgments (qrels)?
├── Yes, graded (0–3)   →  RAGTaskType.RETRIEVAL_GRADED  +  QrelSource.GOLD
├── Yes, binary (0/1)   →  RAGTaskType.RETRIEVAL_BINARY  +  QrelSource.GOLD
└── No → Does it have ground-truth answer strings?
    ├── Yes, short (≤ 5 tokens, factoid)
    │   └── QrelSource.PROXY_EXACT     (fast, noisiest)
    │   or  QrelSource.PROXY_TOKEN_F1  (>50% token overlap, less noisy)
    │       → RAGTaskType.OPEN_QA_FACTOID
    ├── Yes, passage (sentence or longer)
    │   └── QrelSource.PROXY_TOKEN_F1
    │       → RAGTaskType.OPEN_QA_PASSAGE
    └── No answers either
        └── QrelSource.CUSTOM — you supply a derive_qrels() function
            → RAGTaskType.RETRIEVAL_BINARY (or whichever fits)
```

If a dataset has **both** qrels and answers (e.g. MS MARCO with passage annotations), use `RAGTaskType.HYBRID`.

---

## Step 2 — Register the dataset

Add one line to the `DATASETS` list in `scripts/experiment_grid.py`:

```python
# Minimal: BEIR dataset with gold qrels
DatasetConfig("fiqa", "beir/fiqa", 0, 50, RAGTaskType.RETRIEVAL_BINARY, QrelSource.GOLD),

# With doc cap (index first 50K documents only)
DatasetConfig("trec-covid", "beir/trec-covid", 50_000, 20, RAGTaskType.RETRIEVAL_GRADED, QrelSource.GOLD),

# Open-QA dataset — answers provided, no qrels
DatasetConfig(
    name="my-nq",
    ir_id="beir/nq",
    doc_cap=0,
    n_queries=50,
    task_type=RAGTaskType.OPEN_QA_FACTOID,
    qrel_source=QrelSource.PROXY_TOKEN_F1,
    answers={"q1": ["Paris"], "q2": ["Marie Curie"]},  # loaded separately
),
```

To find the `ir_datasets` ID for a dataset:
```bash
python -c "import ir_datasets; [print(k) for k in ir_datasets.registry.keys() if 'fiqa' in k]"
```

---

## Step 3 — Metrics are selected automatically

`METRICS_FOR_TASK` in `experiment_grid.py` maps task type to metrics:

| `RAGTaskType` | Metrics computed |
|---|---|
| `RETRIEVAL_GRADED` | NDCG@5, Recall@5, MRR |
| `RETRIEVAL_BINARY` | MAP@5, Recall@5, MRR |
| `OPEN_QA_FACTOID` | Recall@5, Exact Match @1 |
| `OPEN_QA_PASSAGE` | Recall@5 |
| `HYBRID` | NDCG@5, Recall@5, MRR, Exact Match @1 |

The output CSV always includes all metric columns; unused ones are set to 0.0. The `task_type` and `qrel_source` columns in the CSV tell you which metrics are meaningful for each row.

---

## Step 4 — Custom loaders (when `ir_datasets` isn't enough)

If your dataset is not in `ir_datasets`, set `ir_id=""` and provide a `derive_qrels` function. You will also need to override `load_dataset()` in `experiment_grid.py` with an `if cfg.name == "my-bench"` branch that returns `(doc_iter_fn, queries, qrels)` using your own loading code.

```python
def my_derive_qrels(ds, queries):
    """Read qrels from a custom TSV file."""
    qrels = {}
    with open("data/my-bench/qrels.tsv") as f:
        for line in f:
            qid, did, rel = line.strip().split("\t")
            qrels[(qid, did)] = int(rel)
    return qrels

DatasetConfig(
    name="my-bench",
    ir_id="",               # not in ir_datasets
    doc_cap=0,
    n_queries=100,
    task_type=RAGTaskType.RETRIEVAL_BINARY,
    qrel_source=QrelSource.CUSTOM,
    derive_qrels=my_derive_qrels,
),
```

---

## Step 5 — Proxy qrels: what they measure and where they fail

Proxy qrels use answer string presence as a stand-in for document relevance. They are **noisy by design**:

- A document containing "Marie Curie" is not necessarily the intended source for "who discovered radium"
- Single-token answers (numbers, common names) produce the most false positives
- `PROXY_EXACT` is faster but noisiest; `PROXY_TOKEN_F1` filters out coincidental single-word matches

**`PROXY_EXACT`**: `relevance = 1` if any answer string appears verbatim in `doc.text`

**`PROXY_TOKEN_F1`**: `relevance = 1` if >50% of answer tokens appear in `doc.text`

RAGtune will print a warning whenever a proxy source is active:
```
⚠ my-nq: qrel_source=proxy_token_f1 — scores are NOT comparable to gold-qrel benchmarks
```

Scores derived from proxy qrels should be treated as directional signals, not absolute numbers. **Do not compare them directly to gold-qrel scores in papers without a caveat.**

---

## Step 6 — Smoke test

```bash
# 1. Verify ir_datasets can load the dataset
python -c "import ir_datasets; ds = ir_datasets.load('beir/fiqa'); print(next(ds.docs_iter()))"

# 2. Run the BM25 baseline only (fast, no reranker)
python scripts/experiment_grid.py --dataset fiqa --groups A

# 3. Verify task_type and qrel_source appear in the output CSV
python -c "
import pandas as pd, glob
f = sorted(glob.glob('results/experiment_grid_fiqa_*.csv'))[-1]
print(pd.read_csv(f)[['dataset', 'task_type', 'qrel_source']].drop_duplicates())
"
```

---

## Step 7 — Update the documentation manifest

Add a row to the `Dataset Integration` section of `docs/MANIFEST.md`:

```markdown
| fiqa | retrieval_binary | gold | docs/onboarding/dataset_integration.md | ✅ done |
```

See `scripts/check_docs.py` for automated coverage reporting.
