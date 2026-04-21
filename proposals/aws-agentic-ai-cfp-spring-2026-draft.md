# Agentic Search Pipelines: Budget-Aware, Feedback-Driven Retrieval for Reliable AI Systems

**AWS Agentic AI Call for Proposals — Spring 2026**
Submission deadline: May 6, 2026

---

## 1. Abstract

Modern agentic AI systems increasingly rely on search and retrieval to ground their reasoning. However, today's systems treat retrieval as a static preprocessing step, resulting in high cost, limited reliability, and lack of transparency. This project proposes **agentic search pipelines**: systems where retrieval becomes the central control mechanism governing reasoning, cost, and reliability.

We build on **RAGtune**, an open-source middleware layer we have developed and validated that implements budget-aware, feedback-driven retrieval. RAGtune demonstrates that the same retrieval budget — used intelligently — produces measurably higher quality than brute-force approaches: on TREC-COVID, our convergence-based scheduling achieves NDCG@5 of 0.774 using only 10 reranked documents, outperforming exhaustive reranking of 30 documents (NDCG@5 = 0.699) at one-third the latency. We will extend this foundation into a full agentic retrieval framework, with AWS-native deployment and open-source tooling.

---

## 2. Motivation and Gap

Current agentic systems follow a simplified pipeline:

```
Retrieve → Reason → Act
```

This pipeline has three fundamental flaws:

**Retrieval is static.** First-stage retrieval defines the ceiling of performance. There is no mechanism to recover from missing or irrelevant evidence, and no adaptation based on what reasoning reveals.

**Reasoning is unbounded.** Agents overuse LLM calls without cost awareness. There is no principled stopping criterion or decision policy governing when to reason further vs. when to retrieve more.

**No feedback between components.** Errors cascade across steps. The output of reasoning does not inform what the retriever fetches next, nor does retrieval signal when reasoning has enough evidence to stop.

**Core gap.** There is no unified framework that treats retrieval as an adaptive decision process, integrates feedback from reasoning into retrieval, and optimizes cost–effectiveness trade-offs at the system level.

---

## 3. Key Idea: Agentic Search Pipelines

We propose reconceptualizing agentic AI systems as **search pipelines with feedback-driven control loops**, where:

- Retrieval is a **dynamic, iterative process** governed by an explicit budget
- Reasoning generates **signals** (uncertainty, score distributions, source disagreement) fed back to the retriever
- A **controller** optimizes the sequence of retrieve-reason-stop actions under resource constraints

**Core principle:** Retrieval is the control layer that regulates reasoning.

### RAGtune as the Foundational Artifact

RAGtune (github.com/avishekanand/sir) is our existing open-source implementation of this principle. Its architecture separates three concerns:

- **Estimator** — a cheap oracle that predicts document relevance from prior reranked batches without running the expensive model again. Implementations range from metadata clustering to constrained linear regression (ReformIR) that learns which retrieval source is predictive for a given query.
- **Scheduler** — a batch selection policy that allocates the remaining budget given estimator priorities. Separating scoring from allocation enables independent optimization of each.
- **CostTracker** — enforces hard constraints on documents reranked, tokens consumed, API calls made, and wall-clock latency. Every operation requests permission before execution.

This modular design is the software substrate into which the research contributions below will be integrated. All experimental results from the proposed work will be reproducible through RAGtune.

---

## 4. Research Objectives

**O1: Adaptive Retrieval as a Decision Process**

Develop retrieval policies that iteratively expand and refine candidate evidence using query affinity and neighborhood exploration, overcoming bounded recall in first-stage retrieval. We will formalize the scheduling problem as a sequential decision process and learn policies via bandit methods and constrained optimization.

**O2: Feedback-Driven Agent Control**

Design agent loops where LLM uncertainty and inter-source disagreement trigger retrieval, refinement, or stopping — replacing fixed pipelines with adaptive ones. We will build on RAGtune's existing convergence feedback mechanism (ReformIRConvergenceFeedback) and generalize it to multi-step reasoning chains.

**O3: Budget-Aware Search Pipelines**

Develop algorithms that optimize cost–quality trade-offs under explicit constraints (latency, API cost, token budget). Our preliminary results (Section 6) show this is achievable; the open problem is learning these policies rather than hand-configuring them.

**O4: Explainable and Auditable Agents**

Extend RAGtune's ControllerTrace — a structured event log of every retrieval and scheduling decision — into a full explanation layer. Every agent decision will be attributable to a specific retrieved document and a specific cost signal.

**O5: Multi-Agent Retrieval Coordination**

Enable multiple agents to share retrieved evidence via structured memory, coordinating through retrieval rather than direct messaging, reducing redundancy and improving robustness for complex multi-hop tasks.

---

## 5. Methodology

### System Architecture

We will build on RAGtune's modular design, adding:

| Component | Role |
|---|---|
| Controller | Decides whether to retrieve, reason, or stop |
| Retriever | Hybrid retrieval (vector + lexical + graph) with adaptive expansion |
| Estimator | Predicts document utility from accumulated feedback |
| Reasoner | LLM-based component producing answers and uncertainty signals |
| Scheduler | Budget allocation policy — which docs, which model, which iteration |
| Memory Layer | Stores retrieved evidence, intermediate results, and learned weights |

### Algorithmic Contributions

**(A) Uncertainty-Driven Retrieval Triggering**
Use LLM token entropy, inter-source score disagreement, and estimator confidence to signal when additional retrieval is warranted. This extends RAGtune's current estimator interface to propagate uncertainty signals upstream.

**(B) Query-Affinity Expansion**
Expand search space using graph-based neighborhood exploration around documents that the estimator identifies as high-utility, prioritizing documents aligned with evolving query intent.

**(C) Bandit-Based Control Policies**
Model agent decisions (retrieve / reason / stop) as a contextual bandit problem. Learn policies from interaction data rather than relying on hand-tuned thresholds.

**(D) Cost–Effectiveness Optimization**
Explicit Pareto optimization over accuracy, latency, and API cost. RAGtune already enforces hard budget constraints; we will extend this to learned soft trade-offs.

---

## 6. Preliminary Results

RAGtune has been evaluated on three BEIR benchmarks (NFCorpus, SciFact, TREC-COVID) with MonoT5 as the reranker. Key findings:

| Config | NDCG@5 | Rerank docs | Latency |
|---|---|---|---|
| BM25 only (no reranking) | 0.656 | 0 | 17ms |
| Static reranking — 5 docs | 0.720 | 5 | 574ms |
| Static reranking — 15 docs | 0.749 | 15 | 1563ms |
| Static reranking — 30 docs | 0.748 | 30 | 3117ms |
| **Feedback-driven (convergence)** | **0.760** | **10** | **1008ms** |

On TREC-COVID specifically, convergence feedback achieves NDCG@5 = 0.774 with 10 documents in under 1 second — outperforming exhaustive reranking of 30 documents (NDCG@5 = 0.699, 3.1 seconds) by 10.7% at one-third the cost. **This is the central empirical claim of the proposal: smarter scheduling within the same budget raises the performance ceiling.**

These results validate the feasibility of the approach. The open research problems are (i) learning the scheduling and stopping policies rather than hand-tuning them, (ii) extending to multi-step reasoning, and (iii) scaling to multi-agent settings.

---

## 7. Innovation Beyond Prior Work

**Retrieval-centric agents.** Prior work treats retrieval as a preprocessing step. We treat it as the primary control mechanism for the agent's reasoning budget.

**Closed-loop search pipelines.** Existing RAG systems are one-way pipelines. RAGtune introduces a feedback loop where reranker scores inform the next retrieval decision, demonstrated empirically to outperform static approaches.

**Principled cost modeling.** Most retrieval systems optimize quality alone. RAGtune enforces explicit multi-dimensional budgets (documents, tokens, API calls, latency) and our proposed work will learn optimal policies within those constraints.

**Unified framework with empirical grounding.** We do not propose a theoretical system; RAGtune is a functioning implementation with benchmark results. The proposed research extends a working artifact, not a design on paper.

---

## 8. Open-Source Contributions

1. **RAGtune** (existing, extended) — modular retrieval middleware with budget enforcement, feedback-driven scheduling, and full decision tracing. Will be extended with uncertainty signals, bandit policies, and multi-agent coordination.

2. **Benchmark Suite** — tasks combining retrieval, reasoning, and cost constraints on BEIR datasets and new multi-hop QA benchmarks.

3. **Evaluation Toolkit** — metrics for cost-aware retrieval performance, Pareto frontier visualization, and explanation fidelity scoring.

4. **Visualization & Debugging Tools** — extension of RAGtune's ControllerTrace into an interactive web dashboard for inspecting agent decisions and retrieval trajectories.

---

## 9. Use of AWS ML Tools

| AWS Service | Role in project |
|---|---|
| **Amazon Bedrock** | LLM-based reasoning, uncertainty estimation, query reformulation |
| **Amazon OpenSearch Service** | Scalable hybrid retrieval (vector + lexical) for large document collections |
| **Amazon SageMaker** | Training estimator models and bandit control policies |
| **AWS Step Functions** | Orchestrating multi-step agentic workflows |
| **Amazon S3 / DynamoDB** | Memory layer — storing retrieved evidence, trace logs, learned weights |

RAGtune's registry system (`@registry.retriever`, `@registry.reranker`) will be extended with AWS-native adapters, enabling drop-in replacement of components with Bedrock and OpenSearch equivalents.

---

## 10. Evaluation Plan

**Tasks:** Complex multi-hop QA (HotpotQA, MuSiQue), enterprise document workflows, scientific literature search (BRIGHT benchmark).

**Metrics:**
- Task accuracy / NDCG@5, Recall@5, MRR
- Cost: tokens consumed, API calls, latency
- Robustness: performance under degraded retrieval conditions
- Explanation fidelity: faithfulness of trace to actual decision path

**Baselines:** BM25-only, static top-K reranking, standard RAG (one-shot retrieve-then-rank), ReAct-style agents.

---

## 11. Budget

| Item | Amount |
|---|---|
| PhD student (partial support, 12 months) | $40,000 |
| Engineering & tooling | $10,000 |
| Compute (supplement to AWS credits) | $15,000 |
| Travel & dissemination | $5,000 |
| **Total** | **$70,000** |

AWS Promotional Credits ($50,000) will be used for OpenSearch deployment, Bedrock API calls for experiments, and SageMaker training runs.

---

## 12. Timeline

| Period | Milestone |
|---|---|
| M1–3 | Uncertainty signal integration; bandit policy baseline; AWS adapter layer |
| M4–6 | Adaptive retrieval expansion; feedback-driven agent control; BEIR evaluation |
| M7–9 | Cost–effectiveness optimization; multi-agent coordination prototype |
| M10–12 | Full evaluation, open-source release, 2–3 paper submissions |

---

## 13. Expected Outcomes

- **2–3 publications** at top-tier venues (SIGIR, ECIR, ACL, NeurIPS)
- **Extended RAGtune** with uncertainty-driven scheduling, bandit policies, and AWS-native components
- **Benchmarks** for cost-aware retrieval in agentic settings
- **AWS-integrated prototype** demonstrating production-scale deployment on Bedrock + OpenSearch
