# Benchmarks

Triage and integration status for retrieval, RAG, and feedback-driven retrieval benchmarks under consideration for RAGtune. Use this file to decide what's worth integrating, what's parked, and what's been rejected.

New entries land via a one-line PR using the [entry template](#entry-template) at the bottom. To **claim** a benchmark for integration, open a PR that fills in the `Picked up by` field with your GitHub handle — that signals you're working on it so others don't duplicate effort.

> **Scope note.** Issue [#3](https://github.com/avishekanand/sir/issues/3) originally proposed two companion pages — one for benchmarks and one for relevant RAG / feedback-driven retrieval papers. On reflection, only the benchmarks page lives here. RAGtune is an open-source project repo, not a research-group repo, and a curated paper list belongs in a research-group or personal wiki rather than in project documentation. (Original suggestion was mine; revisiting after thinking it through.) The benchmarks page stays because it directly informs project decisions about what to integrate.

## Status legend

| Symbol | Meaning |
|---|---|
| ✅ | Integrated — exercised by a `scripts/benchmark_*.py` script in this repo |
| 🔬 | Implementation in flight (PR linked) |
| 🎯 | Strong candidate, awaiting integration |
| ❓ | Needs investigation before triage decision |
| ❌ | Rejected — reason recorded |

## Integrated

### BRIGHT
- **Status:** ✅ `scripts/benchmark_bright.py`
- **Picked up by:** — (historical; pre-dates this triage doc)
- **Paper:** Su et al., [BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval](https://arxiv.org/abs/2407.12883), 2024
- **Source:** TBD — link to project page
- **Properties:** 12 reasoning-intensive domains; long, ambiguous queries; designed to expose limitations of dense retrieval
- **Why interesting:** Dense embeddings underperform here; reranking and iterative refinement are required to get meaningful gains. Direct fit for RAGtune's iterative loop.

## In progress

### FreshStack
- **Status:** 🔬 [PR #4](https://github.com/avishekanand/sir/pull/4) (not yet code-complete; B1–B4 blockers tracked in PR review)
- **Picked up by:** @rahulseetharaman
- **Paper:** Thakur et al., [FreshStack](https://arxiv.org/abs/2504.13128), 2024
- **Source:** `freshstack` PyPI package; HuggingFace `freshstack/queries-oct-2024` and `freshstack/corpus-oct-2024`
- **Properties:** Nugget-level IR on technical Q&A (GitHub + StackOverflow); 5 domains (langchain, yolo, angular, laravel, godot); α-NDCG@10, Coverage@20, Recall@50
- **Why interesting:** Created by the BEIR authors as a fresher, harder benchmark with finer-grained relevance (atomic facts, not docs). Lets us measure RAGtune at nugget granularity — iterative reranking should show asymmetric upside.

## Triage queue — high priority

### BEIR
- **Status:** 🎯 (meta-suite — integrating BEIR enables many sub-benchmarks at once)
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Thakur et al., [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663), NeurIPS 2021
- **Source:** https://github.com/beir-cellar/beir
- **Properties:** 18 datasets across 9 task types; zero-shot evaluation; NDCG@10 standard metric
- **Why interesting:** Industry-standard yardstick. Integrating BEIR means RAGtune can be compared head-to-head against every published retrieval system. Heavy lift (18 datasets) but high payoff.

### TREC Deep Learning Track
- **Status:** 🎯
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Craswell et al., TREC Deep Learning Track overview papers (annual, [2019](https://arxiv.org/abs/2003.07820)–2023)
- **Source:** https://trec.nist.gov/
- **Properties:** MS MARCO passage and document tasks; deep graded relevance judgments; 50–200 queries/year
- **Why interesting:** Gold-standard graded judgments. Smaller query set than BEIR but much deeper per-query annotation — well-suited to measuring iterative refinement quality, not just top-1 hits.

### CRAG (Comprehensive RAG Benchmark)
- **Status:** 🎯
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Yang et al., [CRAG -- Comprehensive RAG Benchmark](https://arxiv.org/abs/2406.04744), 2024
- **Source:** TBD — verify Meta/Facebook Research repo
- **Properties:** 4,409 QA pairs across 5 domains; dynamism + popularity axes; mock APIs for time-varying facts
- **Why interesting:** End-to-end RAG benchmark (not pure retrieval). Tests whether iterative reranking improves downstream *answer* quality, not just retrieval ranks. Closest to the RAGtune value claim.

### RGB
- **Status:** 🎯
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Chen et al., [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431), AAAI 2024
- **Source:** TBD — verify project repo
- **Properties:** Tests four RAG failure modes: noise robustness, negative rejection, information integration, counterfactual robustness
- **Why interesting:** Probes RAG failure modes directly. Useful for showing whether iterative reranking improves *robustness*, not just NDCG.

### CRUMB
- **Status:** 🎯
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Killingback & Zamani, [Benchmarking Information Retrieval Models on Complex Retrieval Tasks](https://arxiv.org/abs/2509.07253), 2025
- **Source:** https://github.com/jfkback/crumb · HuggingFace: `jfkback/crumb`
- **Properties:** 8 multi-aspect retrieval tasks (paper, code, theorem, legal QA, tip-of-the-tongue, clinical trial, StackExchange QA, SetOps); passage and full-document corpora in unified markdown; graded + binary qrels; per-task validation sets
- **Why interesting:** Queries are *complex* — multi-constraint, multi-aspect — which is precisely where one-shot dense retrieval underperforms and iterative refinement should compound gains. Eight heterogeneous domains in a single benchmark means one integration unlocks broad coverage. Authors note even SOTA retrievers struggle, leaving headroom to demonstrate RAGtune's value.

### OBLIQ-Bench
- **Status:** 🎯
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Tchuindjo, Shah & Khattab, [OBLIQ-Bench: Exposing Overlooked Bottlenecks in Modern Retrievers with Latent and Implicit Queries](https://arxiv.org/abs/2605.06235), 2026
- **Source:** TBD — recent preprint, data/code release not yet verified
- **Properties:** 5 *oblique* search problems over real long-tail corpora; queries seek docs that instantiate a latent pattern (implicit stance, failure modes, abstract scenarios) rather than match keywords
- **Why interesting:** Highlights a *retrieval-vs-verification asymmetry* — reasoning LLMs reliably recognize latent relevance when shown a document, but retrievers fail to surface those documents in the first place. RAGtune's iterative retrieve→rerank→refine loop is positioned almost exactly for this gap: a strong verifier (reranker) supplying signal back to broaden candidate proposal. Could be a marquee benchmark for the design.

## Triage queue — medium priority

### MS MARCO (passage)
- **Status:** 🎯
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Nguyen et al., [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/abs/1611.09268), 2016
- **Source:** https://microsoft.github.io/msmarco/
- **Properties:** 8.8M passages; ~500K training queries; sparse binary judgments
- **Why interesting:** Canonical retrieval training/eval set; nearly every retrieval baseline reports here. Free comparison points.

### KILT
- **Status:** 🎯
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Petroni et al., [KILT: a Benchmark for Knowledge Intensive Language Tasks](https://arxiv.org/abs/2009.02252), NAACL 2021
- **Source:** https://github.com/facebookresearch/KILT
- **Properties:** 11 tasks (fact-checking, QA, slot-filling, dialogue) over a single Wikipedia snapshot
- **Why interesting:** Same corpus across tasks isolates retrieval contribution from task variation — clean ablation surface.

### HotpotQA
- **Status:** 🎯
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Yang et al., [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600), EMNLP 2018
- **Source:** https://hotpotqa.github.io/
- **Properties:** 113K multi-hop QA pairs with sentence-level supporting facts
- **Why interesting:** Multi-hop reasoning forces iterative retrieval — natural fit for RAGtune's loop. Sentence-level supporting facts let us measure precision of intermediate retrieval steps.

### 2WikiMultihopQA
- **Status:** ❓
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Ho et al., [Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps](https://arxiv.org/abs/2011.01060), COLING 2020
- **Source:** TBD
- **Properties:** Multi-hop QA with structured reasoning chains
- **Why interesting:** Cleaner multi-hop signal than HotpotQA. Triage question: does the marginal value over HotpotQA justify dual integration cost?

### MuSiQue
- **Status:** 🎯
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Trivedi et al., [MuSiQue: Multihop Questions via Single-hop Question Composition](https://arxiv.org/abs/2108.00573), TACL 2022
- **Source:** TBD
- **Properties:** 2–4 hop questions composed from single-hop building blocks
- **Why interesting:** Compositionality enables ablations of hop depth — directly tests whether RAGtune's iterative budget scales with question complexity.

### LongBench
- **Status:** 🎯
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Bai et al., [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508), 2024
- **Source:** https://github.com/THUDM/LongBench
- **Properties:** 21 tasks across single-doc QA, multi-doc QA, summarization, few-shot, code
- **Why interesting:** Long-context regime stresses budget-aware assembly — a core RAGtune capability that other benchmarks under-exercise.

### MIRACL / NoMIRACL
- **Status:** ❓
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Zhang et al., [MIRACL: A Multilingual Retrieval Dataset Covering 18 Languages](https://arxiv.org/abs/2210.09984), TACL 2023
- **Source:** TBD
- **Properties:** 18 languages; NoMIRACL extends with unanswerable queries
- **Why interesting:** Multilingual stress test. Triage decision depends on whether multilingual is in scope for RAGtune v1.

### SRA-Bench (Skill Retrieval Augmentation)
- **Status:** ❓ — *different paradigm: skill retrieval for agents, not document retrieval*
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Su et al. (Tsinghua THUIR), [Skill Retrieval Augmentation for Agentic AI](https://arxiv.org/abs/2604.24594), 2026
- **Source:** TBD — see paper for data release
- **Properties:** 5,400 capability-intensive test instances; 636 gold skills + 25,626 distractors (26,262 total skill corpus); decomposed evaluation of retrieval → incorporation → end-task execution
- **Why interesting:** Tests whether RAGtune's retrieve-rerank-assemble design generalizes from documents to *skills* (reusable capabilities for agentic LLMs). Decomposed evaluation surfaces a bottleneck pure retrieval metrics miss: the agent's ability to decide *which* retrieved skill to actually invoke. Triage question is scope — is non-document retrieval in RAGtune's charter?

## Triage queue — feedback / interactive (active-learning aligned)

### TREC iKAT (Interactive Knowledge Assistance Track)
- **Status:** ❓
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** TREC iKAT overview papers (2023–)
- **Source:** TBD — investigate data access and license
- **Properties:** Personalized conversational search with mixed-initiative interactions
- **Why interesting:** Closest match to RAGtune's active-learning use case — explicit feedback loop in the protocol. Highest-value triage open question.

### CAsT (TREC Conversational Assistance Track)
- **Status:** 🎯
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Dalton et al., TREC CAsT overview papers (2019–2022)
- **Source:** TBD
- **Properties:** Multi-turn conversational retrieval over MARCO + Wikipedia
- **Why interesting:** Conversational context maps naturally onto RAGtune's reformulation step. Tests whether reformulation improves over raw turn-level retrieval.

### TopiOCQA
- **Status:** ❓
- **Picked up by:** _unassigned — open for pickup_
- **Paper:** Adlakha et al., [TopiOCQA: Open-domain Conversational Question Answering with Topic Switching](https://arxiv.org/abs/2110.00768), TACL 2022
- **Source:** TBD
- **Properties:** ~50K turns; explicit topic shifts within conversations
- **Why interesting:** Topic-switching breaks naive query rewriting — tests whether RAGtune's reformulator handles non-monotonic context.

## Rejected

_(empty — record entries here with a one-line rationale when triage decides against integration)_

## Entry template

```markdown
### <Benchmark Name>
- **Status:** <symbol from legend>
- **Picked up by:** <@github-handle> or `_unassigned — open for pickup_`
- **Paper:** <Author>, [Title](<arxiv-or-doi-link>), <venue> <year>
- **Source:** <project URL or repo>
- **Properties:** <one-line summary — task type, corpus size, metrics>
- **Why interesting:** <2–3 sentences on what this benchmark stresses that others don't, and how it relates to RAGtune>
```
