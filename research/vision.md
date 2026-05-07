# Self‑improving cost‑aware RAG systems as a “compute allocator” problem

## Executive summary

Retrieval‑augmented generation (RAG) was introduced as a way to combine parametric language models with non‑parametric memory (retrieval from external corpora) so that knowledge‑intensive tasks can be improved and kept up to date without fully retraining the model. In practice, industry RAG systems now sit on an expanding menu of components—query reformulation, hybrid retrieval, cross‑encoder reranking, fusion/aggregation, reasoning, verification—yet most optimisation effort and spend still concentrates on the “reasoner” (bigger LLMs, more test‑time compute), not on allocating compute across the whole pipeline.

A rigorous research direction with high industrial impact is to formalise RAG as a **budgeted sequential decision problem**: given a workload and constraints (latency SLOs, $/query budget, risk tolerance), a controller chooses *how much* compute to spend on each component and when to stop. This is motivated by evidence that (i) retrieval and reranking architectures have materially different cost–effectiveness trade‑offs across domains (e.g., BEIR shows rerankers/late‑interaction models often best but “at high computational costs”), (ii) retrieval itself can harm performance when context is irrelevant/noisy, implying the controller must sometimes *avoid* retrieval or filter aggressively, and (iii) test‑time “reasoning compute” (sampling/search) improves performance but is only one knob among many.

The industrial stakes are clear: major cloud platforms now ship managed RAG primitives and explicitly frame RAG around challenges like token constraints, response‑time expectations, governance, and cost—suggesting demand for a **RAG optimiser/control plane** that can deliver near‑frontier quality under budgets.

This report proposes a concrete research programme and a prototype concept (“RAGtune”) centred on: (a) **component‑level scaling plots** (performance vs cost) measured per workload; (b) a **scheduler/controller** learned via contextual bandits / counterfactual learning to rank / off‑policy evaluation; (c) **instrumentation and replay** that enable safe self‑improvement; and (d) an evaluation suite spanning retrieval (BEIR), knowledge‑intensive tasks (KILT), fusion architectures (FiD), retrieval‑quality evaluation (eRAG), adaptive retrieval (RetrievalQA, Self‑RAG), cost‑aware routing (FrugalGPT), and factuality/verifiers (FEVER, FActScore).

Assumptions made explicitly in this report: (1) workloads, budgets, and risk policies are open‑ended; (2) “cost” is multi‑objective (tokens, GPU‑seconds, latency, and $); (3) the proposed controller operates at inference time but can update policies offline/online using logs; (4) the organisation can measure at least some downstream ground truth (task labels, human review, or proxy signals).

## Motivation and industrial relevance

RAG’s core motivation is to mitigate limitations of purely parametric knowledge (staleness, hallucination risk, lack of provenance) by conditioning generation on retrieved evidence. The “default” scaling strategy in LLM land—larger models and more training compute—has strong empirical support via scaling laws, and compute‑optimal training work further shows that training and inference efficiency matter (not only raw size). Yet retrieval‑augmented approaches show an orthogonal axis: retrieving from very large corpora can substitute for parameter increases in some regimes (e.g., retrieval‑enhanced transformers retrieving from trillions of tokens report GPT‑3‑comparable perplexity with far fewer parameters).

Industrial RAG is now explicitly shaped by constraints that are *not* well captured by “accuracy only” benchmarks:

Latency and response time expectations are treated as first‑class design constraints in managed RAG offerings. For example, Azure’s RAG guidance frames “response time expectations” and “token constraints” as core challenges and highlights features like parallel subquery execution and adjustable “reasoning effort” in its agentic retrieval pipeline. Cost is equally concrete: vendors publish per‑token pricing for foundation models and embeddings, making “how many calls/tokens do I spend?” an explicit engineering decision rather than an abstract compute budget.

Managed platform trends strongly suggest demand for *automatic* pipeline optimisation. **Amazon Web Services** (cloud provider) markets “Knowledge Bases” in Bedrock as a managed RAG capability with citations and configurable retrieval; Google’s Vertex AI RAG Engine positions itself as a managed framework for context‑augmented LLM applications; Microsoft’s Azure AI Search documentation describes both “classic RAG” and “agentic retrieval” with query planning and structured outputs. Across these, the implicit message is that enterprises want (i) production‑grade ingestion/chunking/vectorisation, (ii) controllable retrieval strategies, (iii) governance and access control, and (iv) predictable cost/latency.

The research gap: academia has strong results for individual components (retrievers, rerankers, fusion, reasoning, evaluation) but far less mature methodology for **workload‑conditioned compute allocation** across components under real SLOs and budgets. BEIR already observes that the best zero‑shot retrieval effectiveness often comes from expensive architectures, while cheaper methods can underperform and generalise differently across tasks. Separately, cost‑aware model cascading/routing shows that a learned policy can match a best model’s quality with extreme cost reductions in some regimes, motivating analogous policies over *pipeline knobs* rather than just “which model?”.

## Literature survey: key papers, benchmarks, and industrial products

### Canonical RAG and retrieval foundations

The original RAG formulation combines a seq2seq generator with a dense retriever over an external corpus and trains/fine‑tunes end‑to‑end for knowledge‑intensive tasks. Dense Passage Retrieval (DPR) established a widely used dual‑encoder dense retrieval baseline for open‑domain QA, demonstrating large gains in top‑k retrieval accuracy over BM25‑style baselines when trained with QA supervision.

Fusion‑in‑Decoder (FiD) is a major architectural line for multi‑passage conditioning: it encodes passages separately and fuses at decoding time; crucially its compute grows roughly linearly with the number of retrieved passages, making “#passages” a clean scaling knob. More recent work explicitly studies sensitivity to context quality/quantity in FiD training, reinforcing that “more context” is not automatically better without quality control.

### Reranking and late interaction

Cross‑encoder rerankers remain one of the most effective ways to increase precision at the top of the ranked list, but their cost scales with the number of query–document pairs scored. MonoT5 demonstrates ranking with a pretrained encoder–decoder (T5) and is emblematic of “better ranking via more compute per candidate”. ColBERT motivates late interaction to recover much of cross‑encoder effectiveness while being orders of magnitude more efficient in FLOPs per query by precomputing document representations. BEIR’s benchmark‑level analysis explicitly highlights this cost–effectiveness spread among lexical, dense, late‑interaction, and reranking models.

### Reformulation, multi‑query retrieval, and fusion

LLM‑enabled reformulation is a practical bridge between classic query expansion/relevance feedback ideas and modern dense retrieval. HyDE (Hypothetical Document Embeddings) is a primary example: generate a hypothetical document from the query, embed it, and retrieve nearest neighbours—achieving strong zero‑shot dense retrieval improvements without relevance labels. Multi‑query fusion methods often combine rankings; Reciprocal Rank Fusion (RRF) is a classic, strong, simple rank fusion method with robust empirical performance in IR experiments. RAG‑Fusion operationalises multi‑query generation plus rank fusion in a RAG context, representing an applied instantiation of the “more reformulations → potential performance gains” knob.

### Adaptive retrieval, robustness, and evaluation

Retrieval is not uniformly beneficial: work on robustness to irrelevant context provides analysis showing that retrieval augmentation can hurt performance and proposes robustness methods. “When do we retrieve?” is now explicit research: “When Do LLMs Need Retrieval Augmentation?” studies uncertainty/overconfidence and the correlation between model certainty and dependence on retrieved information. RetrievalQA is a benchmark specifically targeting evaluation of adaptive RAG approaches for short‑form open‑domain QA and is useful for stress‑testing retrieval‑gating policies. Self‑RAG trains a model to retrieve on demand and critique/reflect, explicitly arguing that indiscriminate fixed‑k retrieval reduces versatility and can cause unhelpful generation.

On evaluation, eRAG provides a key methodological building block for this research direction: it proposes document‑level evaluation by running the generator with each retrieved document individually and using downstream task performance as a proxy “relevance” label, reporting higher correlation with downstream RAG performance and large compute/memory savings versus end‑to‑end evaluation. This directly supports the need to estimate component scaling curves cheaply and per workload.

### Benchmarks for “workload‑conditioned” RAG research

KILT provides a suite of knowledge‑intensive tasks grounded in a shared Wikipedia snapshot and evaluates provenance—useful for end‑to‑end grounded QA and citation‑style evaluation. BEIR provides heterogeneous retrieval tasks spanning domains and task types—useful for profiling retriever/reranker generalisation and cost–effectiveness. For factuality and verification, FEVER provides a large‑scale fact extraction and verification dataset (Supported/Refuted/NEI) that can serve as a testbed for “verifier” components. FActScore introduces a fine‑grained factuality metric for long‑form generation using atomic facts and retrieval‑based support checking, offering a practical way to quantify “verifier value” and risk reduction.

### Cost‑aware inference and routing as a nearby reference point

FrugalGPT formalises cost‑aware strategies for using paid LLM APIs and proposes LLM cascades that learn which queries need which models, reporting large cost reductions for matched quality. This is conceptually aligned with the proposed scheduler, but your direction generalises the action space from “choose model” to “choose retrieval/reformulation/reranking/reasoning budgets”.

### Industrial products and ecosystem

Managed RAG stacks on major clouds are now well documented. Bedrock Knowledge Bases are described as a managed RAG workflow integrating proprietary data and supporting citations; AWS also documents inference pricing and token accounting (useful for $/query modelling). Vertex AI RAG Engine is documented as a component for building context‑augmented LLM applications, with published generative AI pricing. Azure AI Search describes RAG patterns, semantic ranking, and “agentic retrieval” with query planning and logs, and Microsoft documents vector search and semantic ranking concepts relevant to retrieval cost/latency.

A large open ecosystem supports “bring your own pipeline”: **LangChain** (llm app framework company) positions retrieval building blocks and integrations for RAG; **LlamaIndex** (rag framework company) maintains RAG tutorials and indexing/query engines; vector database vendors such as **Pinecone** (vector database company) and **Weaviate** (vector database company) provide RAG guides and APIs. These tools make it feasible to prototype the proposed scheduler as a control plane spanning heterogeneous backends.

## Component‑level scaling plots and how to estimate them per workload

### Formal definitions

Let a workload be $w \in \mathcal{W}$ (distribution over queries, context, answer types, and risk policies). Let a set of knobs (budgets) be $\mathbf{k}$ with per‑component allocations:


$$
\mathbf{k} = (k_{\text{ref}}, k_{\text{ret}}, k_{\text{rer}}, k_{\text{fus}}, k_{\text{rea}}, k_{\text{ver}})
$$

Define end‑to‑end performance and cost as:

$$
\text{Perf} = f(\mathbf{k}; w), \qquad \text{Cost} = g(\mathbf{k}; w)
$$

Performance is multi‑metric (task success + groundedness + citation correctness). Cost is multi‑metric (latency, tokens, GPU‑seconds, $). The optimisation target is typically Pareto‑optimality, or a constrained objective, e.g.:

$$
\min_{\mathbf{k}} \ \mathbb{E}_{q \sim w}[g(\mathbf{k};q)] \ \text{s.t.}\ \mathbb{E}_{q \sim w}[f(\mathbf{k};q)] \ge \tau,\ \text{p95\_lat} \le L, \ \text{risk} \le \rho
$$

This framing is consistent with the reality that vendors expose explicit token budgets and pricing and that RAG systems face token constraints and response‑time expectations.

### Axes: practical performance and cost metrics

Performance metrics (recommendation by component and system type):

- Retriever / candidate generation: Recall@k, MRR, nDCG, KILT retrieval/provenance metrics where applicable.  
- Reranker: nDCG@k / MRR@k uplift over base retriever; downstream answer accuracy delta.  
- Generator/reasoner: EM/F1 for QA (KILT tasks), BLEU/ROUGE for summarisation, plus groundedness/citation accuracy.  
- Verifier: FEVER accuracy (Supported/Refuted/NEI), FActScore, or claim‑level precision/recall.  
- System‑level: task success + attribution metrics + abstention calibration (selective answering), and robustness under distractors/irrelevant context.  

Cost metrics:

- Tokens: input/output tokens per call; model‑specific pricing; prompt caching effects where available.  
- Compute: GPU‑seconds for embedding/retrieval/reranking; number of cross‑encoder calls (pairs scored); number of retrieval passes; fusion passes.  
- Latency: end‑to‑end and per stage (p50/p95), including external store latency and parallelisation effects.  
- $/query: derived from pricing APIs plus infrastructure costs; vendors publish baseline per‑token pricing for managed LLM calls.  

### Typical scaling behaviours by component

The key hypothesis for this research direction is that each component exhibits its own “diminishing returns” curve, and that the dominant curve depends on workload.

A compact comparison (representative methods cited as primaries):

| Component | Main knob(s) $k$ | Typical scaling shape | Dominant failure mode if under‑budget | Represents compute as | Representative primary methods | 
|---|---|---|---|---|---|
| Reformulator | # rewrites; decomposition depth; HyDE generations | Often steep early gains for recall, then plateaus | Low recall / vocabulary mismatch | extra LLM calls + retrieval passes | HyDE; RRF fusion |
| Retriever | k (candidates); hybrid weights; ANN params | Improves recall with k; may add noise | Missing evidence; long‑tail failure | ANN queries + embedding calls | DPR; BEIR profiling |
| Reranker | #pairs scored; cascade depth | Often monotonic in precision@k, expensive | Noisy context; wrong top docs | cross‑encoder calls | MonoT5; ColBERT efficiency |
| Fusion | #passages fused; aggregation strategy | Near‑linear cost in #passages for FiD | Missed multi‑hop evidence | encoder passes + decoding | FiD |
| Reasoner | sampling N; search depth; long‑context | Deliberation improves hard queries but costly | Hallucination; shallow reasoning | extra decoding / self‑eval calls | Self‑consistency; Tree‑of‑Thoughts |
| Verifier | #claims checked; entailment checks; doc triangulation | Often large marginal value in regulated domains | Unchecked hallucinations; low trust | claim extraction + retrieval + NLI | FEVER; FActScore; SelfCheckGPT |
| Controller | policy complexity; features; exploration | Better allocation as more data/logs accrue | Wrong spend allocation; instability | inference policy eval | FrugalGPT cascades analogy; counterfactual LTR |

### Example conceptual scaling curves (illustrative)

The intent is to empirically estimate these curves per workload; the shapes below are “typical” sketches used for planning experiments, not claimed as universal laws. Findings like BEIR’s architecture trade‑offs and FiD’s linear passage cost motivate the expectation of different curve slopes by component.

```
Performance ↑
1.0 |             reasoner (test-time compute)
    |            /
0.8 |   reranker/
    |     /    reformulator
0.6 |    /   _.-'
    |   /_.-'        retriever k
0.4 | _.-'
    +--------------------------------→ Cost
      low           medium            high
```

### Methods to empirically estimate curves per workload

A credible “scaling plot” programme needs *repeatable measurement* and *cost accounting*.

Workload stratification. Curves should be conditioned on workload slices: query type (factoid vs compositional), novelty/freshness, domain shift, ambiguity, and risk tier. BEIR explicitly shows that models generalise unevenly OOD, implying curves cannot be treated as globally stable across domains.

DoE for knobs. Use structured perturbations (factorial or sequential) to sweep knobs: e.g., vary {#reformulations, retrieval k, rerank depth, #passages fused, reasoning samples, verifier checks} under a fixed dataset slice and record performance/cost. FiD’s linear scaling in #passages provides a clean axis; MonoT5/ColBERT provide reranking cost axes in #pairs scored; self‑consistency/ToT provide reasoning axes in #samples/#search steps.

Efficient curve estimation via proxy evaluation. Full end‑to‑end RAG evaluation is expensive; eRAG proposes a document‑level method that correlates more strongly with downstream performance than traditional relevance labels and is far cheaper than full end‑to‑end evaluation. This makes it practical to (i) estimate “value of an extra reranker call” and (ii) build per‑document utility models for scheduling.

Accounting for “retrieval can hurt”. Robustness studies show irrelevant context can distract LLMs and reduce task performance; therefore curves must include a noise dimension (context quality) and not assume monotonic gains with k. Methods to handle this include modelling the distribution of retrieval scores, adding learned filters, and explicitly measuring robustness under distractors.

Infrastructure cost models. Retrieval cost depends on ANN algorithms and index parameters; primary references like FAISS (GPU similarity search) and HNSW (graph‑based ANN) justify modelling retrieval latency/compute as a function of index and search parameters, not only “k”. Vendor documentation further supports practical token/cost accounting via published pricing and token policies.

## Scheduler/controller formalisation and candidate learning algorithms

### Why scheduling is the central scientific object

Once multiple components have meaningful cost–performance curves, the main problem becomes: **given partial information mid‑pipeline, decide whether to spend more compute and where**. This is essentially the same structural problem as cost‑aware cascades over LLMs (FrugalGPT), but the action space is richer (reformulate, retrieve more, rerank deeper, fuse differently, reason more, verify claims).

Azure’s “agentic retrieval” description (LLM‑assisted query planning, subqueries, parallel execution, adjustable reasoning effort, query activity logs) is an existence proof that vendors are productising scheduling‑like behaviours in retrieval layers. Your research direction is to make this *learned, general, measurable,* and *cost‑optimal* rather than heuristic and vendor‑specific.

### Formal model

Let each query episode be a trajectory $(s_t, a_t, r_t)$.

State $s_t$ (observable context at step $t$) typically includes:

- Query features: length, ambiguity signals, domain classifier, risk tier.
- Retrieval signals: score distributions, entropy, redundancy among top docs, disagreement across retrievers, evidence coverage.
- Generator signals: uncertainty proxies, self‑consistency variance, citation alignment scores.
- Budget remaining: token budget, latency remaining, $ cap.

Actions $a_t$ are discrete/continuous decisions such as:

- Generate $m$ reformulations; choose a fusion operator (e.g., RRF); decompose query.
- Retrieve with $k'$ (increase $k$), switch retriever/hybrid weight, apply filters.
- Rerank top‑$n$ with cross‑encoder; use late‑interaction stage; stop early.
- Choose fusion depth (#passages), summarise context, compress.
- Choose reasoner model size/routing; set sampling $N$ or ToT search depth.
- Trigger verifier: extract claims, retrieve supporting evidence, entailment checks.

Reward/objective. A practical constrained objective:

$$
\max_{\pi} \ \mathbb{E}\big[\text{Quality}(\tau) - \lambda C - \mu \cdot \text{RiskPenalty} - \nu \cdot \text{LatencyPenalty}\big]
$$

where $C$ aggregates tokens, GPU‑sec, and external calls; risk penalties can include unsupported claims (FActScore‑style), FEVER misclassification, or citation mismatches.

Constraints can be “hard”: $\text{p95 latency} \le L$, $/query ≤ B, and compliance constraints (must produce citations, must log evidence). Cloud docs explicitly foreground token and latency constraints, making these constraints realistic rather than academic.

### Candidate learning algorithms

Contextual bandits for budget allocation. If the controller’s main decision is “which next action to take” given a context and immediate reward proxy, contextual bandits are a natural fit. Online learning‑to‑rank via contextual bandits is a mature line of work in IR and provides algorithms and modelling ideas for choosing actions under uncertainty.

Counterfactual learning and off‑policy evaluation (OPE) from logs. For safety and practicality, much learning should occur offline from logged trajectories. Unbiased learning‑to‑rank with biased feedback provides principles for correcting logging bias, and doubly robust policy evaluation provides variance‑reduced estimators for OPE in contextual bandits. These are directly applicable if you log propensities for exploration actions (e.g., occasional deeper reranking) and then evaluate candidate policies offline before deployment.

RL‑lite / sequential decision processes. When actions have long‑term effects (e.g., early reformulation changes retrieval quality which changes whether verification is needed), modelling as a short‑horizon MDP can be justified. In practice, an “RL‑lite” approach often means: (i) design a small action space, (ii) learn value estimates from logs (OPE), and (iii) deploy conservatively with guardrails. The maturity of doubly robust OPE and counterfactual LTR makes this credible for production‑adjacent research.

Heuristic baselines and ablations are essential. Self‑RAG and RetrievalQA both emphasise that adaptive retrieval is non‑trivial to evaluate; therefore, learned controllers must be compared against strong and transparent heuristics (fixed‑k RAG; retrieve‑only‑when‑uncertain; rerank‑only‑if‑score‑entropy‑high; etc.).

## Instrumentation and the self‑improvement loop

### Why instrumentation is non‑optional

Two empirical facts make instrumentation central:

1) Retrieval and reranking metrics are imperfect proxies for end‑to‑end RAG utility; eRAG explicitly motivates document‑level evaluation because traditional relevance labels correlate weakly with downstream RAG performance and full end‑to‑end evaluation is expensive.
2) Retrieval augmentation can hurt under irrelevant context, so the system must detect and react to low‑quality evidence rather than blindly “spend more compute”.  

### Suggested logging schema (minimum viable)

A practical schema should make offline replay and OPE possible (store enough to reconstruct decisions and costs).

- **Query envelope**: query_id, timestamp, user/tenant, risk tier, language, topic classifier, PII flags.
- **Policy info**: policy_version, exploration flag, action propensities (for OPE), budget limits.
- **Reformulation**: reformulations[], model_id, tokens_in/out, latency.
- **Retrieval**: retriever_id(s), k, ANN params, index_version, retrieved_doc_ids with scores, filters applied.
- **Reranking**: reranker_id, n_pairs_scored, scores, latency/GPU‑sec (if available).
- **Fusion/context**: selected chunks, ordering, compression/summaries, context token count.
- **Generation**: model_id, decoding params, tokens_in/out, latency; produced citations and mapping to doc_ids.
- **Verification**: claims extracted, evidence retrieved, entailment scores, abstention decisions.
- **Outcome**: ground truth label (if available), human review result, user feedback, downstream tool acceptance.

This mirrors what industrial RAG docs emphasise as operational necessities: citations/provenance, token constraints, query logs and governance.

### Proxy signals for learning under limited labels

High‑scale self‑improvement depends on proxy signals that correlate with ground truth:

- Document utility labels: eRAG’s method (run generation per doc, score vs ground truth) yields a proxy “doc usefulness” signal and is cheaper than full evaluation.  
- Consistency/variance: self‑consistency uses multiple reasoning paths and can produce an uncertainty proxy (agreement rate).  
- Robustness indicators: detect high distractor sensitivity via perturbations; robustness studies motivate measuring distractor effects explicitly.  
- Factuality proxies: FActScore’s atomic fact checking and SelfCheckGPT’s sampling‑based hallucination signals can serve as verifier‑side risk proxies.  

### Replay and safe deployment

A safe loop is: collect logs → offline replay evaluation → conservative online A/B or canary with guardrails → update policy.

Counterfactual LTR and doubly robust OPE provide the theoretical basis for learning from logged interaction data while correcting bias/variance, which is critical if you cannot constantly perturb production traffic. Azure’s documentation explicitly mentions query activity logs and structured outputs with citations in its agentic retrieval flow, which aligns with the need to observe and audit controller behaviour.

## Evaluation suite, experimental matrix, and proposed new metrics

### Baseline suite to cover the required benchmark set

A credible evaluation suite for your research direction should cover:

- Retrieval generalisation: BEIR (heterogeneous retrieval tasks).  
- Knowledge‑intensive end‑to‑end tasks with provenance: KILT.  
- Fusion models and multi‑passage conditioning: FiD (as a controllable fusion baseline).  
- Retrieval evaluation aligned to downstream RAG: eRAG.  
- Reformulation: HyDE as a representative “reformulate then retrieve” technique.  
- Adaptive retrieval / retrieve‑when‑needed: RetrievalQA benchmark; Self‑RAG as a trained adaptive method.  
- Cost‑aware routing baseline: FrugalGPT.  

For verifier‑centric evaluation: FEVER and FActScore provide complementary regimes (short claims vs long‑form factuality) and are useful to quantify “risk reduction per unit cost.”

### Proposed new cost‑aware metrics

To make results meaningful for industry, augment traditional metrics with cost‑aware ones:

Cost‑conditioned Pareto frontier. For each workload $w$, estimate the Pareto frontier $ \mathcal{P}_w = \{(C, P)\}$ from knob sweeps. Report:
- **Area under the Pareto frontier** over a cost range (AUPF) as a single number.
- **Budgeted performance**: $P(C \le B)$ at several budgets $B$.
- **Cost to target**: minimal $C$ achieving $P \ge \tau$.  
BEIR’s explicit cost commentary motivates reporting effectiveness jointly with compute cost rather than only accuracy.

Risk‑adjusted utility. For regulated settings, define:

$$
U = P - \alpha \cdot \Pr(\text{unsupported claim}) - \beta \cdot \Pr(\text{missing citation})
$$

Use FActScore‑style atomic support measures and citation correctness from RAG outputs to estimate risk.

Robustness under distractors. Use robustness benchmarks or “needle in a haystack” style tests (including multilingual distractor variants) to measure how performance degrades as distractors increase.

### Proposed experimental matrix

This matrix is designed to estimate scaling curves and policy gains without assuming a specific domain upfront.

| Workload family (open‑ended) | Knobs swept | Metrics | Budgets/constraints |
|---|---|---|---|
| Knowledge‑intensive QA (KILT) | reformulations m; retrieval k; rerank n; FiD passages K; verifier claim checks | EM/F1; provenance; citation accuracy; eRAG doc utility correlation | token cap; latency cap; $/query |
| Zero‑shot retrieval generalisation (BEIR) | retriever type; k; ANN params; rerank depth | nDCG@10; recall@k; latency | compute/sec; p95 |
| Adaptive retrieval QA (RetrievalQA) | retrieval trigger threshold; fallback policies; confidence estimation | accuracy; abstention calibration; compute saved | fixed latency SLO |
| Long‑form grounded generation | fusion passages; summarise/compress; verifier intensity | FActScore; citation coverage; hallucination proxies | token & cost cap |
| Regulated fact verification (FEVER) | retrieval depth; reranking; verifier model choice | FEVER label accuracy; evidence correctness | risk threshold dominates |

This design is consistent with the benchmarks’ intended uses (KILT provenance, BEIR retrieval heterogeneity, RetrievalQA for adaptive retrieval).

### Key ablations (to make papers publishable)

- Static pipeline vs adaptive controller under equal budgets (main claim).
- Remove each knob class (no reformulation, no rerank, no verification) to measure marginal value.
- Replace learned policy with heuristics (retrieve‑always, retrieve‑if‑uncertain) to isolate learning benefit; “When Do LLMs Need Retrieval Augmentation?” gives a strong uncertainty‑retrieval baseline.  
- Evaluate under irrelevant context stress tests to show robustness gains (robust RAG literature).  

## RAGtune prototype design, industrial use‑cases, and roadmap

### Conceptual architecture: RAGtune as a RAG “control plane”

RAGtune should be positioned as a **policy engine + measurement layer** that sits above heterogeneous RAG components (retrievers/rerankers/LLMs/vector DBs) and produces: (i) an execution plan per request and (ii) a continuously updated Pareto model per workload.

A minimal architecture:

- **Connectors**: ingestion/chunking/vectorisation connectors to data sources and indexers (leveraging managed options where possible).
- **Component registry**: declarative catalogue of available reformulators, retrievers, rerankers, fusers, reasoners, verifiers with their cost models.
- **Policy runtime**: executes a per‑query plan with budget tracking and early stopping.
- **Observability store**: logging + replay.
- **Offline optimiser**: fits scaling curves \(f(\mathbf{k};w)\), \(g(\mathbf{k};w)\), learns policies via OPE/bandits, and compiles Pareto‑optimal configurations.
- **Dashboards**: Pareto frontier, cost drivers, failure analysis (retrieval noise vs reasoning limits vs verification gaps).

This aligns with vendor framing: Azure highlights query planning, parallel subqueries, and “query activity logs”; AWS/Bedrock highlights citations and managed retrieval; Vertex highlights managed RAG corpora and engine APIs.

### Figures

Pipeline diagram (Mermaid). This is the conceptual object you will measure and optimise.

```mermaid
flowchart LR
  Q[User query] --> S[Scheduler / Controller]
  S -->|0..m| Rf[Reformulator]
  Rf -->|queries| Rt[Retriever]
  Rt -->|top-k candidates| Rk[Reranker / Cascade]
  Rk -->|selected evidence| Fu[Fusion / Context builder]
  Fu --> Re[Reasoner (SLM/LLM)]
  Re -->|draft answer + citations| Ve[Verifier / Critic]
  Ve -->|final answer| Out[Response]
  Ve -->|signals| S
  Rt -->|signals| S
  Rk -->|signals| S
```

Scheduler decision flow (Mermaid). This is a deployable policy skeleton independent of model choice.

```mermaid
flowchart TD
  A[Start: new query] --> B{Risk tier / compliance?}
  B -->|High| C[Enable citations + verifier budget]
  B -->|Normal| D[Default budgets]
  C --> E[Initial retrieval (small k)]
  D --> E
  E --> F{Evidence quality OK?}
  F -->|No| G[Reformulate / decompose]
  G --> H[Retrieve again or increase k]
  H --> I[Optional rerank deeper]
  F -->|Yes| I[Optional rerank]
  I --> J{Need multi-hop / synthesis?}
  J -->|Yes| K[Increase fusion passages or reasoning compute]
  J -->|No| L[Use SLM / low compute]
  K --> M[Generate answer]
  L --> M
  M --> N{Verifier passes?}
  N -->|Fail| O[Escalate: more evidence or bigger reasoner]
  N -->|Pass| P[Return final + citations]
  O --> E
```

Example scaling curves figure placeholders (URLs to be supplied/created during implementation). Vendors and papers provide the “knobs”; you provide the empirical curves.

### APIs (suggested)

A practical API surface is small and “policy first”:

- `POST /rag/answer` with: query, workload_id, constraints (`max_latency_ms`, `max_cost_usd`, `risk_tier`), and policy hints.
- `POST /rag/profile` to ingest labelled evaluation sets and define workload slices.
- `GET /pareto/{workload_id}` returns Pareto frontier estimates and recommended configs.
- `POST /policy/evaluate_offline` runs OPE on logs and outputs risk bounds.
- `POST /policy/deploy_canary` deploys a new policy with guardrails.

This aligns with the need for OPE and safe deployment emphasised by counterfactual LTR and doubly robust evaluation work.

### Dashboards (what executives and engineers will actually use)

- Cost breakdown: tokens vs retrieval vs reranking vs verification.
- Pareto frontier: quality vs $/query and quality vs p95 latency.
- “Why we spent” explanations: evidence quality triggers, uncertainty triggers (auditability).
- Failure clustering: “retrieval miss”, “retrieval noise”, “reasoning error”, “verification fail”.

Azure’s emphasis on “query activity log” and provenance/citations shows that auditability is product‑relevant, not only academic.

### Industrial use‑cases and impact analysis

Customer support / internal knowledge. This domain has high volume and strong cost sensitivity; many queries are repetitive and should be answerable with small models if retrieval quality is good and evidence is clean. Managed RAG documentation explicitly targets “grounding in proprietary content” and highlights token constraints and response‑time expectations typical of support chat. The controller’s value is to (i) keep most queries in a “cheap lane” (small retrieval + light rerank + SLM), (ii) escalate only on low evidence quality or low confidence.

Regulated domains (legal/health/finance). Here, the risk term dominates the objective: you pay for verification, citation correctness, and abstention. Verifier components can be benchmarked with FEVER‑style claim verification and FActScore‑style atomic factuality to quantify risk reduction. A key industrial deliverable is a policy that guarantees “no unsupported claims above threshold” under a budget, even if that means refusing more often.

Developer assistants. Developer Q&A is retrieval‑heavy (docs, codebases) and benefits from reranking and context selection; long context windows do not remove the need for retrieval planning, as retrieval vs long context trade‑offs are actively studied and are non‑trivial. Here the controller can trade reranking cost against context length and use verification (tests, compilation, doc citation) selectively.

Enterprise search and report synthesis. Multi‑document synthesis stresses fusion and verification. FiD provides a controllable fusion baseline, and factuality metrics like FActScore provide a way to score groundedness of long outputs. Controller value is especially high because the naive approach (stuffing many chunks into a large LLM) is both expensive and can degrade quality under irrelevant context.

### Risks, limitations, and open research questions

The hard truth: the controller can easily learn the wrong thing if you cannot measure downstream quality reliably. eRAG explicitly identifies weak correlation between traditional relevance labels and downstream RAG performance; without better proxy signals, scaling curves can be misestimated. Robustness work shows retrieval can hurt; therefore policies that blindly “spend more on retrieval” can degrade quality.

Key open questions that remain publishable and industrially decisive:

- **Identifiability**: When can you infer per‑component marginal value from logs without full ground truth? (eRAG helps but doesn’t solve all settings.)  
- **Safe exploration**: How to explore new allocations without harming users? Counterfactual LTR and doubly robust OPE are foundations, but exploration policy design in RAG is still open.  
- **Robustness vs cost**: How to detect “retrieval noise” early enough to stop or filter cheaply? Robust RAG work motivates this but does not yet yield universal detectors.  
- **Controller generalisation**: Will a policy learned on one query mix generalise (BEIR suggests OOD shifts are substantial)?  
- **Evaluation leakage**: LLM‑as‑judge and proxy evaluators can be biased; better calibration against human labels is needed, especially in regulated settings.  

### Timeline and milestone plan (prototype + papers)

A practical plan that yields both a prototype and publishable results:

| Phase | Milestones | Expected outputs |
|---|---|---|
| Foundation | Implement logging + replay; baseline static pipelines (fixed‑k RAG, rerank depth sweep, FiD passage sweep) | Reproducible measurement harness; initial scaling curves on 2–3 workloads |
| Curve estimation | Implement eRAG‑style efficient evaluation; cost accounting (tokens, latency, rerank calls) | “Component scaling plots” paper draft; public curve dataset |
| Controller v1 | Heuristic controller + contextual‑bandit selector; offline OPE with propensities | Workshop paper: budgeted RAG scheduling; internal canary |
| Controller v2 | RL‑lite sequential policy; robustness constraints; verifier integration | Main conference submission (ACL/SIGIR/NeurIPS track fit depends on framing) |
| Productisation | Pareto compiler; dashboards; integration with at least one managed backend (Bedrock/Vertex/Azure) | RAGtune prototype demo; case study in one industrial vertical |

This plan is anchored in the maturity of OPE/CLTR methods and the existence of benchmark suites to measure retrieval and RAG outcomes.

```mermaid
flowchart TD
  A[Start: new query] --> B{Risk tier / compliance?}
  B -->|High| C[Enable citations + verifier budget]
  B -->|Normal| D[Default budgets]
  C --> E[Initial retrieval (small k)]
  D --> E
  E --> F{Evidence quality OK?}
  F -->|No| G[Reformulate / decompose]
  G --> H[Retrieve again or increase k]
  H --> I[Optional rerank deeper]
  F -->|Yes| I[Optional rerank]
  I --> J{Need multi-hop / synthesis?}
  J -->|Yes| K[Increase fusion passages or reasoning compute]
  J -->|No| L[Use SLM / low compute]
  K --> M[Generate answer]
  L --> M
  M --> N{Verifier passes?}
  N -->|Fail| O[Escalate: more evidence or bigger reasoner]
  N -->|Pass| P[Return final + citations]
  O --> E
```