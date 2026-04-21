# RAGtune Experiment Grid Report

Detailed systematic ablation study across quality metrics (NDCG@5, Recall@5, MRR) and latency.

## 📋 Methodology & Configuration

### 1. Budget Levels
Definitions for document reranking constraints and batch sizes:

| Level | Rerank Docs | Batch Size | Description |
| :--- | :--- | :--- | :--- |
| **Tight** | 5 | 2 | Minimum cost, low latency constraint. |
| **Medium** | 15 | 5 | Balanced configuration for general use. |
| **Loose** | 30 | 10 | Aggressive reranking for maximum precision. |

> [!NOTE]
> Group E (**Ollama Medium**) uses the **Tight** budget parameters (5 docs) to keep LLM inference latency manageable.

### 2. Feedback Thresholds (ReformIR)
Thresholds for iterative convergence feedback:

| Threshold | Value | Description |
| :--- | :--- | :--- |
| **Tight** | 1% (0.01) | Stops only when score changes are highly stable. |
| **Loose** | 5% (0.05) | More aggressive stopping to save budget. |

### 3. Estimator Strategies
- **Baseline**: No prioritization (uniform).
- **Similarity**: Prioritizes documents similar to already discovered high-quality documents.
- **ReformIR**: Uses a regression model to predict the utility of remaining candidates.

### 4. Group Definitions & Rerankers
Each group targets a specific aspect of the RAG pipeline ablation:

| Group | Goal | Reranker Model | Description |
| :--- | :--- | :--- | :--- |
| **A** | **Baseline** | None (BM25) | The raw retriever performance without any reranking. |
| **B** | **Budget Ablation** | `MonoT5` | Measures how quality scales with the number of documents reranked (5/15/30). |
| **C** | **Estimator Ablation** | `MonoT5` | Compares the effectiveness of different priority estimators (Baseline/Sim/ReformIR). |
| **D** | **Feedback Ablation** | `MonoT5` | Evaluates if iterative feedback loops can save budget via early stopping. |
| **E** | **LLM Reranking** | `DeepSeek-R1 (Ollama)` | Tests a local LLM with a listwise ranking prompt vs. standard cross-encoders. |
| **F** | **Full Pipeline** | `MonoT5` | Combines query reformulation (via LLM) with iterative feedback and reranking. |

> [!IMPORTANT]
> **MonoT5**: Uses `castorini/monot5-base-msmarco`. It is a cross-encoder that outputs a relevance probability for a (query, doc) pair.
>
> **DeepSeek-R1**: A reasoning-based LLM. We use a listwise prompt where the model receives 5 documents and must output a JSON ranking.

## 📊 Dataset: NFCorpus
*~3.6K docs, 50 queries, graded qrels (BEIR)*

| Group | Config | NDCG@5 | Recall@5 | MRR | Latency (ms) | Rerank Docs |
|:---|:---|---:|---:|---:|---:|---:|
| A | bm25_only | 0.5842 | 0.104 | 0.5715 | 8 | 0 |
| B | monot5_tight | 0.6452 | 0.104 | 0.6432 | 616 | 4.8 |
| B | monot5_medium | 0.6576 | 0.1173 | 0.6485 | 1524 | 14.2 |
| B | monot5_loose | 0.6565 | 0.1133 | 0.6395 | 2960 | 27.9 |
| C | baseline_est | 0.6576 | 0.1173 | 0.6485 | 1526 | 14.2 |
| C | similarity_est | 0.6576 | 0.1173 | 0.6485 | 1540 | 14.2 |
| C | reformir_est | 0.6576 | 0.1173 | 0.6485 | 1533 | 14.2 |
| D | no_feedback | 0.6575 | 0.1133 | 0.6497 | 2989 | 27.9 |
| D | convergence_tight | 0.6609 | 0.1143 | 0.6575 | 1033 | 9.5 |
| D | convergence_loose | 0.6609 | 0.1143 | 0.6575 | 1027 | 9.5 |
| E | ollama_medium | 0.5879 | 0.104 | 0.5715 | 7228 | 4.8 |
| F | baseline_pipeline | 0.6576 | 0.1173 | 0.6485 | 1539 | 14.2 |
| F | reformir_pipeline | 0.6555 | 0.1098 | 0.6504 | 5824 | 9.5 |

---

## 📊 Dataset: SciFact
*~5K docs, 50 queries, binary qrels (BEIR)*

| Group | Config | NDCG@5 | Recall@5 | MRR | Latency (ms) | Rerank Docs |
|:---|:---|---:|---:|---:|---:|---:|
| A | bm25_only | 0.7931 | 0.854 | 0.773 | 8 | 0 |
| B | monot5_tight | 0.8536 | 0.854 | 0.8547 | 605 | 5 |
| B | monot5_medium | 0.8458 | 0.858 | 0.8485 | 1678 | 15 |
| B | monot5_loose | 0.8471 | 0.859 | 0.8484 | 3324 | 30 |
| C | baseline_est | 0.8458 | 0.858 | 0.8485 | 1695 | 15 |
| C | similarity_est | 0.8458 | 0.858 | 0.8485 | 1689 | 15 |
| C | reformir_est | 0.8458 | 0.858 | 0.8485 | 1706 | 15 |
| D | no_feedback | 0.8488 | 0.859 | 0.848 | 3360 | 30 |
| D | convergence_tight | 0.8463 | 0.858 | 0.8482 | 1114 | 10 |
| D | convergence_loose | 0.8463 | 0.858 | 0.8482 | 1119 | 10 |
| E | ollama_medium | 0.7731 | 0.834 | 0.753 | 11251 | 5 |
| F | baseline_pipeline | 0.8458 | 0.858 | 0.8485 | 1883 | 15 |
| F | reformir_pipeline | 0.8258 | 0.843 | 0.8293 | 5363 | 10 |

---

## 📊 Dataset: TREC-COVID
*50K docs, 20 queries, graded qrels (ir-datasets)*

| Group | Config | NDCG@5 | Recall@5 | MRR | Latency (ms) | Rerank Docs |
|:---|:---|---:|---:|---:|---:|---:|
| A | bm25_only | 0.5908 | 0.004 | 0.5588 | 36 | 0 |
| B | monot5_tight | 0.663 | 0.004 | 0.6421 | 502 | 5 |
| B | monot5_medium | 0.7448 | 0.0051 | 0.7583 | 1486 | 15 |
| B | monot5_loose | 0.699 | 0.0046 | 0.7081 | 3067 | 30 |
| C | baseline_est | 0.7448 | 0.0051 | 0.7583 | 1497 | 15 |
| C | similarity_est | 0.7448 | 0.0051 | 0.7583 | 1497 | 15 |
| C | reformir_est | 0.7448 | 0.0051 | 0.7583 | 1502 | 15 |
| D | no_feedback | 0.732 | 0.0048 | 0.7247 | 2964 | 30 |
| D | convergence_tight | 0.7737 | 0.0045 | 0.7729 | 992 | 10 |
| D | convergence_loose | 0.7737 | 0.0045 | 0.7729 | 1000 | 10 |
| E | ollama_medium | 0.5908 | 0.004 | 0.5588 | 7334 | 5 |
| F | baseline_pipeline | 0.7448 | 0.0051 | 0.7583 | 1517 | 15 |
| F | reformir_pipeline | 0.7638 | 0.0045 | 0.7472 | 4500 | 10 |
