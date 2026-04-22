# Feedback-Driven Adaptive Retrieval for Agentic AI: From Research Breakthroughs to Production-Ready Systems

**AWS Agentic AI Call for Proposals — Spring 2026**
Submission deadline: May 6, 2026

**Principal Investigator:** Prof. Avishek Anand, Delft University of Technology

---

## 1. Abstract

The classical retrieve-and-rerank pipeline that underlies most RAG-based AI systems is fundamentally static: first-stage retrieval defines a bounded candidate pool, and any relevant document missed at this stage is permanently lost to downstream reasoning. Recent breakthroughs in information retrieval have shown that this bounded recall problem can be overcome — not by retrieving more documents upfront, but by making retrieval itself adaptive, iterative, and feedback-driven.

Our research group has established a coherent body of work demonstrating that feedback-driven retrieval consistently and substantially outperforms static approaches. Using relevance-aware affinity graphs, we achieve up to 26% recall improvement over standard adaptive retrieval [1]. Uncertainty signals from LLMs, fed back into the retrieval process, yield up to 31.84% improvement on complex multi-hop QA [3]. Online relevance estimation via bandit-based learning lets retrieval systems sample-efficiently approximate expensive ranker scores, achieving large recall gains with a fraction of the inference cost [4]. Incorporating LLM listwise feedback improves retrieval by 13.23% nDCG@10 while keeping inference budgets fixed [2]. Budget-aware reformulation optimization suppresses query drift and delivers up to 34.68% gains with 3.3–4.5× lower latency than LLM-based reranking [7].

This project proposes to unify these research advances into a complete **agentic retrieval framework** deployable on AWS, and to extend the research frontier into multi-step reasoning, multi-agent coordination, and principled cost–quality optimization. The empirical and algorithmic contributions of this project will be released as **RAGtune**, an open-source middleware library built to make feedback-driven adaptive retrieval accessible to the research community and deployable at production scale.

---

## 2. Motivation and Problem

### 2.1 The Bounded Recall Problem

Modern retrieval systems follow a telescoping architecture: a fast first-stage retriever (BM25, dense retrieval) narrows a corpus of millions to hundreds of candidates; an expensive reranker (cross-encoder, LLM) then scores these candidates in isolation. The fundamental limitation of this pipeline is that **any relevant document excluded by the first-stage retriever is permanently lost** — the reranker has no mechanism to recover it.

This bounded recall problem is well-documented for standard IR tasks [1, 4] and becomes particularly severe in complex question-answering settings, where multi-hop queries require evidence from multiple documents and failing to retrieve any one of them causes cascading failures throughout the reasoning chain [3]. The problem is further compounded in agentic systems, where retrieval errors propagate across multiple reasoning steps and accumulate over the course of an agent's action sequence.

### 2.2 The Static Pipeline Problem

Beyond bounded recall, current RAG and agentic systems treat retrieval as a one-shot preprocessing step. There is no feedback loop between the reasoning component and the retriever: the LLM's uncertainty, the consistency of retrieved evidence, or the divergence between retrieval sources are all discarded signals. A survey of the current state of the field concludes that **retrieval should be treated as a dynamic, learnable component** of end-to-end AI systems rather than a fixed preprocessing step [5] — yet existing frameworks lack the architectural support to do so.

### 2.3 The Cost Problem

Improving retrieval quality by simply retrieving more documents or calling expensive rankers more often is unsustainable. Agentic systems compound this: multiple retrieval steps per reasoning chain, multiple agents per task, and latency constraints from real-time deployment. What is needed is not more compute but **smarter allocation of retrieval budget**: using feedback from each reranked batch to decide what to retrieve next, rather than exhaustively processing everything.

---

## 3. Our Research Program: A Body of Work in Feedback-Driven Retrieval

Over the past two years, our group has published a systematic research program addressing the bounded recall, static pipeline, and cost problems from complementary directions. This section summarizes the key contributions that provide the empirical and algorithmic foundation for the proposed work.

### 3.1 Adaptive Retrieval via Affinity Graphs

The standard approach to corpus graph construction for adaptive retrieval uses heuristic lexical or semantic similarity — graphs that are agnostic to query-based relevance. Our work on **QUAM** (Query Affinity Modelling) [1] addresses this fundamental limitation by training an edge-prediction model that learns document affinities conditioned on co-relevance in existing query workloads. The resulting affinity-aware graph enables a retrieval algorithm that judiciously selects neighbors to expand based on learned affinity weights, rather than treating all neighbors equally. On TREC-DL '19 and '20, QUAM achieves up to **26% recall improvement** over static baselines and up to 12% over prior adaptive retrieval approaches — demonstrating that the quality of the graph structure is a primary lever for retrieval quality.

### 3.2 LLM-Guided Adaptive Retrieval for Listwise Rerankers

Adaptive retrieval methods assume pointwise document scoring — a score is computed for each document independently. This assumption excludes a rapidly growing class of LLM-based **listwise rerankers**, which score documents jointly and cannot be trivially integrated with existing adaptive retrieval pipelines. Our work on **SlideGAR** [2] resolves this incompatibility by designing an adaptive retrieval algorithm that operates in the listwise setting: it merges initial retrieval results with graph-based feedback documents and dynamically guides which documents are included in each LLM inference window. Across diverse LLM rerankers and first-stage retrievers, SlideGAR improves **nDCG@10 by up to 13.23%** and **recall by 28.02%** — with no increase in total LLM inference calls.

### 3.3 Uncertainty-Driven Neighborhood Retrieval for Complex QA

For complex question-answering tasks requiring multi-hop reasoning, the bounded recall problem is particularly acute: failing to retrieve evidence for one sub-question propagates failures across the entire reasoning chain. Our work on **SUNAR** [3] (Semantic Uncertainty-based Neighborhood Aware Retrieval) introduces a feedback mechanism where the uncertainty of the LLM's candidate answers is used to dynamically promote or penalize documents during iterative neighborhood expansion. When the LLM is uncertain about its current answer, the retriever is directed to expand into document neighborhoods likely to contain missing evidence. On complex QA benchmarks (MuSiQue, 2WikiMultiHop), SUNAR achieves up to **31.84% improvement** over existing retrieve-and-reason baselines, substantially closing the gap to an idealized oracle with perfect retrieval.

### 3.4 Online Relevance Estimation via Bandit Learning

A recurring cost problem in iterative retrieval is that the expensive ranker must be called for every candidate document — even those the system already has strong prior evidence are irrelevant. Our work on **online relevance estimation** (ORE) [4] introduces a bandit-based framework that learns a lightweight surrogate model to predict the expensive ranker's score for unranked candidates, using the scores of already-ranked documents as training signal. This surrogate is updated online during query processing, enabling the system to selectively call the expensive ranker only where the surrogate is uncertain. ORE is validated on TREC benchmarks in both hybrid retrieval and adaptive retrieval settings, demonstrating **sample-efficient** operation with significant recall improvements at substantially reduced ranker calls.

### 3.5 Budget-Aware Reformulation with Drift Suppression

Query reformulation is a standard technique for improving recall by generating multiple query variants, but it introduces a fundamental problem: as the number of reformulations grows, irrelevant ones introduce noise that degrades retrieval quality — the query drift problem. Our work on **ReformIR** [7] addresses this with a bi-level bandit optimization framework that treats reformulations as features in a surrogate relevance model. Under a fixed reranking budget, ReformIR learns which reformulations are useful by anchoring relevance feedback to the original query, actively suppressing drift while maximizing recall. On TREC Deep Learning benchmarks, ReformIR achieves gains of up to **34.68%** and remains **3.3–4.5× more efficient** than LLM-based reranking, even with aggressive reformulation scaling.

### 3.6 Graph-Free Corpus Feedback from Listwise Rankings

Corpus graphs — precomputed structures encoding pairwise document similarities — are central to many adaptive retrieval methods, but constructing them requires full corpus access, significant compute (e.g., bi-encoder passes over all pairs), and substantial storage. Our work on **L2G** [6] (Listwise-to-Graph) eliminates this requirement by reconstructing an explicit doc-doc graph directly from the ordering signals produced by listwise rerankers — with no additional LLM or retriever calls. L2G matches the performance of graph-based adaptive retrieval (SlideGAR) while requiring only the inference calls already needed for standard listwise reranking, making adaptive retrieval practical in dynamic, resource-constrained, or legacy retrieval settings.

### 3.7 Adaptive Retrieval for Reasoning-Intensive Tasks

Most adaptive retrieval benchmarks use topical or question-answering style queries. Our reproducibility and extension work [8] demonstrates that corpus graph-based adaptive retrieval generalizes to **reasoning-intensive retrieval** (BRIGHT benchmark), where queries require multi-step inference rather than topical matching. Adaptive retrieval boosts effectiveness across a variety of reasoning-capable rankers while contributing negligibly to latency — validating that the core adaptive retrieval framework extends to the more demanding task distributions that arise in agentic AI systems.

### 3.8 Framing: Retrieval as a Learnable Component

Across this body of work, a consistent theme emerges: performance gains come not from more retrieval, but from **closing the feedback loop** between retrieval and ranking. A survey of this research landscape [5] positions feedback-driven retrieval as the unifying paradigm for the next generation of RAG systems, categorizing feedback signals by source (reranker scores, LLM uncertainty, inter-source disagreement), role (query reformulation, candidate expansion, early stopping), and integration point (corpus graph, surrogate model, scheduler policy). This taxonomy provides the conceptual framework for the proposed work.

---

## 4. Research Objectives

Building on this foundation, this project addresses the open research problems that emerge when feedback-driven adaptive retrieval is deployed in **agentic AI settings**: multi-step reasoning chains, multiple coordinating agents, real-time cost constraints, and the need for full explainability of retrieval decisions.

**O1: Learned Scheduling Policies for Adaptive Retrieval**

The scheduling and stopping policies in iterative retrieval — when to retrieve more, which documents to expand next, when to stop — are currently hand-tuned heuristics or fixed algorithms. We will formalize these as learned policies using the bandit and surrogate modelling approaches developed in [4, 7], extending them to the multi-step agentic setting where the same agent issues multiple retrieval requests within a single reasoning chain.

**O2: Uncertainty-Driven Agent Control**

LLM uncertainty signals, demonstrated effective for single-step retrieval in [3], will be generalized into a principled agent control mechanism: a controller that observes LLM token entropy, inter-source score disagreement, and estimator confidence to decide whether to retrieve more evidence, refine the existing retrieval, or proceed to reasoning. This extends SUNAR's uncertainty feedback from single-query to multi-turn reasoning chains.

**O3: Cost-Effective Retrieval under Explicit Budgets**

We will develop retrieval policies that explicitly optimize the cost-quality Pareto frontier, building on ORE's online surrogate modelling [4] and ReformIR's budget-aware reformulation [7]. The goal is a principled framework — rather than hand-tuned thresholds — for learning optimal retrieve/reason/stop decisions under hard constraints on tokens, API calls, and latency.

**O4: Explainable and Auditable Retrieval Decisions**

The L2G approach [6] demonstrates that implicit corpus feedback from listwise rankings can be made explicit through graph reconstruction. We will extend this idea into a full explanation layer: a structured event log that attributes every agent decision to a specific retrieved document, a specific cost signal, and a specific feedback source. This makes the retrieval process inspectable and auditable.

**O5: Multi-Agent Retrieval Coordination**

When multiple agents collaborate on a complex task, they often issue redundant retrieval requests and retrieve overlapping evidence. We will design coordination protocols where agents share retrieval state via structured memory, leveraging graph-based corpus feedback [1, 6] to identify which document neighborhoods have already been explored and which remain promising.

---

## 5. Methodology

### System Architecture

| Component | Role |
|---|---|
| Controller | Decides whether to retrieve, reason, or stop |
| Retriever | Hybrid retrieval (vector + lexical + affinity graph [1]) |
| Estimator | Online surrogate model for document utility [4] |
| Reasoner | LLM producing answers and uncertainty signals [3] |
| Scheduler | Budget-aware policy for batch selection and reformulation [7] |
| Memory Layer | Shared evidence store for multi-agent coordination [5] |

### Algorithmic Contributions

**(A) Adaptive Graph Expansion with Learned Affinity**
Extend QUAM's affinity graph [1] to the agentic setting: at each retrieval step, the graph is queried based on the current set of high-scoring documents, with affinity scores used to prioritize neighborhood expansion. The graph is not static — it is conditioned on the evolving retrieval state across the agent's reasoning chain.

**(B) Uncertainty-Triggered Retrieval**
Generalize SUNAR's uncertainty mechanism [3] into a component of the agent controller. LLM token entropy and answer candidate disagreement are computed after each reasoning step; high uncertainty triggers a targeted retrieval expansion rather than proceeding to the next reasoning action.

**(C) Bandit-Based Control and Scheduling**
Extend ORE [4] and ReformIR [7] into a unified bandit-based controller that simultaneously manages: (i) which documents to score next, (ii) which reformulations to prioritize, and (iii) when to stop retrieval and commit to a reasoning step. The surrogate model is updated online within the agent's reasoning trace.

**(D) Graph-Free Feedback Integration**
Adapt L2G [6] to construct doc-doc feedback graphs dynamically during the agent's retrieval trace, without requiring a precomputed corpus graph. This enables adaptive retrieval in fully dynamic settings (e.g., web retrieval, document streams) where the corpus is not known in advance.

**(E) Multi-Agent Evidence Sharing**
Design a shared retrieval memory structure — informed by SlideGAR's listwise feedback mechanism [2] and SUNAR's neighborhood awareness [3] — that allows multiple agents to query a shared retrieval state, avoid redundant exploration, and coordinate evidence gathering across sub-tasks.

---

## 6. Innovation Beyond Prior Work

**Retrieval-centric agent control.** Existing agentic frameworks (ReAct, self-RAG) treat retrieval as one among many tools, invoked heuristically. This project, grounded in our published work [1–8], demonstrates that feedback-driven retrieval is a principled control mechanism: recall improvements of 13–34% are achievable through smarter scheduling alone, not more retrieval.

**Closed-loop pipelines with learning.** Our existing work [4, 7] shows that online bandit learning can replace hand-tuned scheduling and reformulation heuristics. This project extends that principle to multi-step reasoning chains and multi-agent systems.

**From pointwise to listwise to graph-based feedback.** Our work spans the full spectrum of feedback granularity: pointwise surrogate models [4], listwise LLM feedback [2], and graph-based co-relevance signals [1, 6]. The proposed work is the first to integrate all three within a unified agentic retrieval framework.

**Empirically grounded research program.** This proposal is not based on a hypothetical framework. Every research direction is grounded in published, peer-reviewed results from our group [1–8]. The proposed work extends a validated line of research into a new deployment setting.

**Principled cost modeling at system level.** ReformIR [7] demonstrates 3.3–4.5× efficiency gains over LLM-based reranking by learning budget-aware retrieval policies. This project scales these gains to the multi-step, multi-agent agentic setting.

---

## 7. Open-Source Contributions: RAGtune

The empirical and algorithmic contributions of this project will be released as **RAGtune** (github.com/avishekanand/sir), an open-source middleware library for budget-aware, feedback-driven retrieval. RAGtune is designed to consolidate the research outcomes from our group into a deployable, composable, and extensible framework that the broader research community can use and build upon.

RAGtune's modular architecture reflects the separation of concerns demonstrated by our published work:

- **Estimator** — pluggable document utility estimation, supporting both offline affinity-based scoring [1] and online surrogate learning [4]
- **Scheduler** — budget-aware batch selection with support for bandit policies [4, 7] and reformulation management [7]
- **Controller** — the orchestration loop with uncertainty-triggered retrieval [3] and full decision logging
- **Memory Layer** — shared evidence store for multi-agent coordination [5]

Specific open-source deliverables:

1. **RAGtune core library** (extended) — incorporating learned scheduling [O1], uncertainty-driven control [O2], bandit-based policies [O3], and multi-agent memory [O5]

2. **Benchmark Suite** — standardized evaluation tasks combining retrieval, reasoning, and cost constraints across BEIR, HotpotQA, MuSiQue, and BRIGHT [8]

3. **Evaluation Toolkit** — metrics for cost-aware retrieval: recall@K under budget, Pareto frontier visualization, explanation fidelity scoring, and latency profiling

4. **Debugging Dashboard** — an interactive web interface extending the controller trace log into a visual inspection tool for retrieval trajectories and agent decisions

---

## 8. Use of AWS ML Tools

| AWS Service | Role in project |
|---|---|
| **Amazon Bedrock** | LLM-based reasoning, uncertainty estimation, query reformulation |
| **Amazon OpenSearch Service** | Scalable hybrid retrieval (vector + lexical) for large document collections |
| **Amazon SageMaker** | Training affinity models [1], surrogate estimators [4], and bandit policies |
| **AWS Step Functions** | Orchestrating multi-step agentic workflows |
| **Amazon S3 / DynamoDB** | Memory layer — storing retrieved evidence, trace logs, learned weights, and corpus graphs |

RAGtune's registry system (`@registry.retriever`, `@registry.reranker`) will be extended with AWS-native adapters, enabling drop-in replacement of components with Bedrock and OpenSearch equivalents. This makes the research framework directly deployable as an AWS-native agentic retrieval system.

---

## 9. Evaluation Plan

**Tasks:** TREC Deep Learning (DL19–DL22), complex multi-hop QA (HotpotQA, MuSiQue, 2WikiMultiHop), reasoning-intensive retrieval (BRIGHT [8]), enterprise document workflows.

**Metrics:**
- Retrieval quality: NDCG@5, Recall@K, MRR
- Cost: reranker calls, tokens consumed, API calls, wall-clock latency
- Robustness: performance under degraded first-stage retrieval
- Explanation fidelity: faithfulness of trace to actual decision path

**Baselines:** BM25-only, static top-K reranking, standard RAG (one-shot retrieve-then-rank), ReAct-style agents, GAR [1], SlideGAR [2], SUNAR [3], ORE [4], and ReformIR [7].

---

## 10. Budget

| Item | Amount |
|---|---|
| PhD student (partial support, 12 months) | $40,000 |
| Engineering & tooling | $10,000 |
| Compute (supplement to AWS credits) | $15,000 |
| Travel & dissemination | $5,000 |
| **Total** | **$70,000** |

AWS Promotional Credits ($50,000) will be used for OpenSearch deployment, Bedrock API calls for experiments, and SageMaker training runs for affinity models and bandit policies.

---

## 11. Timeline

| Period | Milestone |
|---|---|
| M1–3 | Uncertainty-driven controller [O2]; bandit scheduling baseline [O1]; AWS adapter layer |
| M4–6 | Affinity graph expansion in agentic setting [O1]; multi-step ORE integration [O3]; BRIGHT evaluation |
| M7–9 | Cost–quality optimization [O3]; multi-agent coordination prototype [O5]; explainability layer [O4] |
| M10–12 | Full evaluation, RAGtune open-source release, 2–3 paper submissions |

---

## 12. Expected Outcomes

- **2–3 publications** at top-tier venues (SIGIR, ECIR, ACL, NeurIPS) extending the adaptive retrieval research program to agentic settings
- **RAGtune** open-source release consolidating our group's research program into a production-ready retrieval middleware with AWS-native deployment
- **Benchmarks** for cost-aware retrieval in agentic and multi-agent settings
- **AWS-integrated prototype** demonstrating end-to-end feedback-driven agentic retrieval on Bedrock + OpenSearch at production scale

---

## References

[1] Rathee, M., MacAvaney, S., and Anand, A. "QUAM: Adaptive Retrieval through Query Affinity Modelling." *Proceedings of WSDM 2025*, pp. 954–962.

[2] Rathee, M., MacAvaney, S., and Anand, A. "Guiding Retrieval using LLM-based Listwise Rankers." *Proceedings of ECIR 2025*. (arXiv:2501.09186)

[3] Venktesh, V., Rathee, M., and Anand, A. "SUNAR: Semantic Uncertainty-based Neighborhood Aware Retrieval for Complex QA." *Proceedings of NAACL 2025*.

[4] Rathee, M., Venktesh, V., MacAvaney, S., and Anand, A. "Breaking the Lens of the Telescope: Online Relevance Estimation over Large Retrieval Sets." *Proceedings of SIGIR 2025*, pp. 2287–2297.

[5] Rathee, M., Venktesh, V., MacAvaney, S., and Anand, A. "Test-time Corpus Feedback: From Retrieval to RAG." *Findings of EACL 2025*. (arXiv:2508.15437)

[6] Yoon, S., Rathee, M., Venktesh, V., and Anand, A. "On Listwise Reranking for Corpus Feedback." *Proceedings of WSDM 2026*.

[7] Venktesh, V., Rathee, M., and Anand, A. "When More Reformulations Hurt: Avoiding Drift using Ranker Feedback." *Proceedings of SIGIR 2026*.

[8] Rathee, M., Venktesh, V., MacAvaney, S., and Anand, A. "Reproducing Adaptive Reranking for Reasoning-Intensive IR." *Proceedings of SIGIR 2026*.
