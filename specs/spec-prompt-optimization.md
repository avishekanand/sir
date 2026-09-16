This is the executable specification for **RAGtune Prompt Optimization (v0.6)** — reflective evolutionary search over the *prompt text* used by RAGtune's LLM-backed components, as opposed to the structured configuration search already covered by `spec-mobo-tuning.md` and the GEPA config agent in `ragtune/tuning/llm_optimizer.py`.

---

## 1. Motivation & Goals

RAGtune has two optimizers today, and both search the same thing: a dictionary of
structured hyperparameters (`reranker_type`, `original_query_depth`,
`budget_rerank_docs`, …). Bayesian TPE proposes those values with a surrogate
model; the GEPA agent proposes them with an LLM. Neither ever changes a **prompt**.

Meanwhile two components are driven entirely by prompt text loaded from
`config/prompts.yaml`:

| Component | Registry name | Prompt key | Placeholders |
|-----------|---------------|------------|--------------|
| `LLMReranker` | `llm` | `reranking.pointwise_scoring` | `{query}`, `{document}` |
| `LLMReformulator` | `llm_rewrite` | `reformulation.llm_rewrite` | `{query}`, `{m}` |

Those strings were written once by hand and have never been tuned, even though
pointwise relevance scoring is exactly the task where prompt phrasing moves
NDCG by a large margin. This spec closes that gap.

**Primary objective:** maximize NDCG@10 by evolving prompt text, holding the
pipeline configuration and retriever fixed.

**Secondary objectives:**
- Keep the cost axis visible — each pointwise rerank is one API call per
  document, so a prompt that needs a larger rerank depth is more expensive.
- Emit prompts as a YAML fragment that can be dropped into `config/prompts.yaml`
  or loaded at runtime.

**Non-goals:** optimizing the *optimizer's* own reflection prompt; multi-turn or
agentic prompts; prompts for components that do not exist yet.

---

## 2. Relationship to Existing Optimizers

| | search space | proposer | artifact |
|---|---|---|---|
| `tuning/optimizer.py` | structured params | Optuna TPE | pipeline YAML |
| `tuning/llm_optimizer.py` | structured params | LLM reflection (GEPA) | pipeline YAML |
| **this spec** | **prompt strings** | **LLM reflection (GEPA)** | **prompts YAML** |

The search strategy is deliberately the same GEPA shape already validated in
`llm_optimizer.py` — Pareto pool over per-query scores, reflective mutation of
one component at a time, minibatch screening before full evaluation. Only the
genome changes: text instead of numbers. This is also what GEPA is in the
literature (Agrawal et al., 2025), where the evolved artifact *is* the prompt;
RAGtune's existing use of the machinery on a config space is the adaptation.

---

## 3. Injection Contract

Prompts must be overridable at runtime without editing `config/prompts.yaml`,
because the optimizer evaluates dozens of candidates per run.

`ConfigLoader` holds two independent dictionaries — `_config` (from
`defaults.yaml`) and `_prompts` (from `prompts.yaml`) — with `get`/`set` reading
and writing the former and `get_prompt` reading the latter. There is **no
setter for `_prompts`**, so this spec adds one:

```python
config.set_prompt("reformulation.llm_rewrite", {"system": ..., "user": ...})
```

`set_prompt` mirrors `set` exactly (dot-path traversal, creating intermediate
dicts) but targets `_prompts`. This is the only `src/` change the spec requires.

### 3.1 Known defect this spec must work around

`LLMReranker.rerank()` reads its prompt with:

```python
prompts = config.get("prompts.reranking.pointwise")   # _config, key 'pointwise'
```

Two things are wrong: `get` reads `_config`, not `_prompts`; and the key in
`prompts.yaml` is `pointwise_scoring`, not `pointwise`. The call therefore
returns `None` and the next line (`prompts.get("system")`) raises
`AttributeError`. **The `llm` reranker cannot currently run.**

Fixing `rerankers.py` is out of scope here — it is an unrelated defect and
belongs in its own `fix/` branch (Rule 1). The optimizer instead seeds the
baseline prompt into the location the component actually reads:

```python
config.set("prompts.reranking.pointwise", {...})   # lands in _config; get() finds it
```

which both works around the defect and provides the override hook. When the
defect is fixed, the optimizer switches to `set_prompt` for this component too
and nothing else changes.

---

## 4. Prompt Genome & Validation

A candidate is `{"system": str, "user": str}`. A candidate is **invalid** and is
rejected without evaluation when:

1. A required placeholder is missing. The component calls
   `template.format(query=..., document=...)`, so a missing placeholder produces
   a silently query-free prompt, and an *extra* one raises `KeyError`.
2. `str.format` raises on the candidate (unescaped braces from JSON examples in
   the prompt body — the most common LLM failure mode here).
3. The user template exceeds `--max-prompt-chars` (default 4000), which would
   otherwise let the optimizer buy score with context length.

Rejected candidates are recorded in history with `error` set so the reflection
step can see and avoid the failure mode; they do not consume evaluation budget.

---

## 5. Evaluation Protocol

Identical to `tuning/evaluator.py` so numbers are comparable across optimizers:

- Metric: NDCG@10 via `ragtune.tuning.evaluator.ndcg_at_k`, over the same
  `EvalDataset.from_pyterrier_irds` sample.
- Cost: mean `rerank_docs` from `ControllerOutput.final_budget_state`, which for
  the pointwise LLM reranker equals the number of API calls per query.
- Pipeline: a fixed configuration (`--rerank-depth`, `--n-queries`), so the only
  independent variable is the prompt.
- Stage 1 minibatch screen on `--minibatch` queries; a candidate that does not
  beat its parent there is dropped before the full evaluation.

---

## 6. Loop

```text
seed     baseline prompt (from prompts.yaml, or --seed-prompt file)
repeat n_iterations:
    parent   <- sample from Pareto pool, weighted by per-query wins
    failures <- k lowest-NDCG queries for parent, with the documents it
                mis-scored and the scores it assigned
    candidate<- LLM(reflection prompt | parent, its score, failures)
    validate placeholders  -> on failure, record and continue
    screen on minibatch    -> if not better than parent, record and continue
    evaluate fully, update pool
write best + Pareto prompts to <out>/prompts_*.yaml, history to history.json
```

---

## 7. Deliverables

```text
specs/spec-prompt-optimization.md      # this file
src/ragtune/utils/config.py            # + set_prompt()
tests/unit/utils/test_config.py        # get/set/get_prompt/set_prompt isolation
examples/prompt_optimizer.py           # the optimizer + CLI
```

The optimizer lives in `examples/` rather than `src/ragtune/tuning/` for this
version: it shares no interface with `BaseOptimizer` (there isn't one), and
keeping it out of the package avoids committing to an API before the approach
has produced evidence it beats the hand-written prompt. Promotion to
`src/ragtune/tuning/prompt_optimizer.py` is the natural follow-up once it has.

---

## 8. Cost Model

Pointwise reranking is one API call per document, so a full evaluation costs
`n_queries * rerank_depth` calls, plus one reflection call per iteration:

```text
calls ≈ iterations * (minibatch * depth)            # screening
      + accepted   * (n_queries * depth)            # full evaluations
      + iterations                                  # reflection
```

At the defaults (20 iterations, 20 queries, depth 10, minibatch 5) that is
~1,000–5,000 calls depending on the screen pass rate. The CLI prints this
estimate and requires `--yes` to proceed past it when the projection exceeds
`--max-calls`.

---

## 9. Success Criteria

1. The evolved reranker prompt beats the hand-written `pointwise_scoring` prompt
   on held-out queries by a margin larger than the seed-to-seed spread, measured
   over ≥3 seeds.
2. Evolved prompts are valid `prompts.yaml` fragments and load unmodified.
3. A run is resumable and reports honestly when the LLM budget ran out.
