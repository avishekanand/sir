"""
The AutoIndex loop: index → analyze → synthesize → select (arXiv:2607.18603).

Typical use::

    ai = AutoIndex(AutoIndexConfig(), analysis_llm=LiteLLMClient(model),
                   code_llm=LiteLLMClient(model))
    result = ai.fit(corpus, val_queries, val_qrels, n_eval_queries=len(eval_queries))
    held_out = ai.evaluate_code(result.best_code, corpus, eval_queries, eval_qrels)

Only validation queries go into ``fit``. Adoption and the returned checkpoint
are chosen on validation Recall@100.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ragtune.indexing.autoindex import prompts
from ragtune.indexing.autoindex.agents import AnalysisAgent, Candidate, CodeAgent, PromptContext
from ragtune.indexing.autoindex.bm25 import ChunkBM25Index, Tokenizer
from ragtune.indexing.autoindex.history import CandidateRecord, IterationRecord, SearchHistory
from ragtune.indexing.autoindex.llm import LLMClient, UsageTracker
from ragtune.indexing.autoindex.metrics import EvalSummary, diff_outcomes, evaluate_index
from ragtune.indexing.autoindex.program import (
    IDENTITY_PROGRAM_CODE,
    Document,
    available_packages,
    ProgramError,
    corpus_to_documents,
    load_program,
    run_program,
    validate_program_code,
)
from ragtune.indexing.autoindex.tools import AnalysisToolbox

logger = logging.getLogger(__name__)


@dataclass
class AutoIndexConfig:
    """Defaults follow the reference implementation's config.yaml."""

    n_iterations: int = 5
    n_candidates: int = 4
    code_concurrency: Optional[int] = None  # parallel code-writing calls; None = n_candidates (as in the paper).
                                            # Use 1 for single-slot servers such as a default Ollama.
    improvement_threshold: float = 1e-5     # "proven" ⇔ ΔR@100 ≥ threshold
    analysis_max_turns: int = 5
    min_tool_turns: int = 3
    preprocess_timeout_s: float = 900.0
    use_analysis: bool = True               # ablation: off ⇒ Code Agent sees aggregate metrics only
    use_history: bool = True                # ablation: off ⇒ no search history in prompts
    use_synthesis: bool = True              # merge ≥2 proven candidates
    k1: float = 1.5
    b: float = 0.75
    candidate_k: int = 1000
    bm25_backend: str = "auto"
    validation_sample_docs: int = 20
    corpus_description: str = ""
    log_dir: Optional[str] = None


@dataclass
class CandidateResult:
    candidate: Candidate
    summary: Optional[EvalSummary] = None
    delta_recall_100: float = 0.0
    delta_recall_10: float = 0.0
    delta_ndcg_10: float = 0.0
    improved: List[str] = field(default_factory=list)
    regressed: List[str] = field(default_factory=list)
    error: Optional[str] = None
    proven: bool = False


@dataclass
class AutoIndexResult:
    best_code: str
    best_val: EvalSummary
    initial_val: EvalSummary
    history: SearchHistory
    usage: UsageTracker
    analyses: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_code": self.best_code,
            "initial_val": self.initial_val.as_dict(),
            "best_val": self.best_val.as_dict(),
            "history": self.history.to_dict(),
            "usage": self.usage.stats,
            "analyses": self.analyses,
        }


class AutoIndex:
    def __init__(
        self,
        config: Optional[AutoIndexConfig] = None,
        analysis_llm: Optional[LLMClient] = None,
        code_llm: Optional[LLMClient] = None,
        tokenizer: Optional[Tokenizer] = None,
        usage: Optional[UsageTracker] = None,
    ):
        self.config = config or AutoIndexConfig()
        if code_llm is None:
            raise ValueError("code_llm is required")
        if self.config.use_analysis and analysis_llm is None:
            raise ValueError("analysis_llm is required when use_analysis=True")
        self.analysis_llm = analysis_llm
        self.code_llm = code_llm
        self.tokenizer = tokenizer or Tokenizer()
        self.usage = usage or UsageTracker()

    # -- building blocks ------------------------------------------------------

    def build_index(self, code: str, docs: List[Document]) -> ChunkBM25Index:
        chunks = run_program(load_program(code), docs, timeout_s=self.config.preprocess_timeout_s)
        c = self.config
        return ChunkBM25Index(chunks, k1=c.k1, b=c.b, tokenizer=self.tokenizer,
                              candidate_k=c.candidate_k, backend=c.bm25_backend)

    def evaluate_code(self, code: str, corpus: Dict[str, Dict], queries: Dict[str, str],
                      qrels: Dict[str, Dict[str, int]]) -> EvalSummary:
        """Score a program on any query set (e.g. held-out queries, after ``fit``)."""
        return evaluate_index(self.build_index(code, corpus_to_documents(corpus)), queries, qrels)

    def _score_candidate(self, cand: Candidate, docs, sample, val_queries, val_qrels,
                         incumbent: EvalSummary) -> CandidateResult:
        res = CandidateResult(candidate=cand)
        if not cand.code:
            res.error = "no code produced"
            return res
        err = validate_program_code(cand.code, sample)
        if err:
            res.error = f"validation: {err}"
            return res
        try:
            res.summary = evaluate_index(self.build_index(cand.code, docs), val_queries, val_qrels)
        except (ProgramError, ValueError) as e:
            res.error = str(e)
            return res
        res.delta_recall_100 = res.summary.recall_at_100 - incumbent.recall_at_100
        res.delta_recall_10 = res.summary.recall_at_10 - incumbent.recall_at_10
        res.delta_ndcg_10 = res.summary.ndcg_at_10 - incumbent.ndcg_at_10
        diff = diff_outcomes(res.summary, incumbent)
        res.improved, res.regressed = diff["improved"], diff["regressed"]
        res.proven = res.delta_recall_100 >= self.config.improvement_threshold
        return res

    # -- main loop ------------------------------------------------------------

    def fit(
        self,
        corpus: Dict[str, Dict],
        val_queries: Dict[str, str],
        val_qrels: Dict[str, Dict[str, int]],
        n_eval_queries: int = 0,
        initial_code: str = IDENTITY_PROGRAM_CODE,
    ) -> AutoIndexResult:
        cfg = self.config
        val_queries = {q: t for q, t in val_queries.items() if q in val_qrels}
        if not val_queries:
            raise ValueError("fit() needs at least one validation query with qrels")
        docs = corpus_to_documents(corpus)
        sample = docs[: cfg.validation_sample_docs]
        log_dir = Path(cfg.log_dir) if cfg.log_dir else None
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

        ctx = PromptContext(n_val=len(val_queries), n_eval=n_eval_queries, tokenizer=self.tokenizer.describe(),
                            corpus_description=cfg.corpus_description or "(no corpus description provided)",
                            min_tool_turns=cfg.min_tool_turns,
                            packages=", ".join(available_packages()) or "none")
        analysis_agent = (AnalysisAgent(self.analysis_llm, ctx, max_turns=cfg.analysis_max_turns,
                                        min_tool_turns=cfg.min_tool_turns) if cfg.use_analysis else None)
        code_agent = CodeAgent(self.code_llm, ctx, max_workers=cfg.code_concurrency or cfg.n_candidates)
        history = SearchHistory()
        analyses: List[str] = []

        incumbent_code = initial_code
        incumbent_index = self.build_index(incumbent_code, docs)
        initial = incumbent = evaluate_index(incumbent_index, val_queries, val_qrels)
        logger.info("AutoIndex θ₀: val R@100=%.4f nDCG@10=%.4f", initial.recall_at_100, initial.ndcg_at_10)

        for t in range(cfg.n_iterations):
            history.record_iteration(IterationRecord(t, incumbent.recall_at_100, incumbent.ndcg_at_10,
                                                     incumbent.recall_at_10))
            history_text = history.render() if cfg.use_history else ""

            # 2. ANALYZE + 3. SYNTHESIZE. An LLM failure (after the client's own retry)
            # skips this iteration instead of aborting the run, as in the reference loop.
            try:
                if analysis_agent is not None:
                    toolbox = AnalysisToolbox(incumbent_index, corpus, val_queries, val_qrels)
                    analysis = analysis_agent.analyze(incumbent, initial, toolbox, incumbent_code,
                                                      history_text or None).summary
                else:
                    analysis = prompts.render(prompts.NO_ANALYSIS_SECTION, **incumbent.as_dict())
                candidates = code_agent.propose(analysis, incumbent_code, cfg.n_candidates, history_text)
            except Exception as e:
                logger.warning("iter %d: LLM stage failed (%s: %s); skipping iteration", t, type(e).__name__, e)
                analyses.append(f"[iteration skipped: {type(e).__name__}: {e}]")
                continue
            analyses.append(analysis)
            for i, c in enumerate(candidates):
                c.id = f"it{t}-{c.id or f'H{i + 1}'}"

            # 4. SELECT
            results = [self._score_candidate(c, docs, sample, val_queries, val_qrels, incumbent) for c in candidates]
            for r in results:
                history.record_candidate(CandidateRecord(
                    iteration=t, candidate_id=r.candidate.id, description=r.candidate.description,
                    rationale=r.candidate.rationale, delta_recall_100=r.delta_recall_100,
                    delta_recall_10=r.delta_recall_10, delta_ndcg_10=r.delta_ndcg_10, proven=r.proven,
                    improved_query_ids=r.improved, regressed_query_ids=r.regressed, error=r.error))
                logger.info("iter %d %s: %s", t, r.candidate.id,
                            r.error or f"ΔR@100={r.delta_recall_100:+.4f} proven={r.proven}")

            adopted = self._select(t, results, analysis, incumbent_code, code_agent, docs, sample,
                                   val_queries, val_qrels, incumbent)
            if adopted is not None:
                adopted_id, adopted_code, adopted_summary, synthesized = adopted
                history.set_adoption(t, adopted_id, synthesized)
                incumbent_code, incumbent = adopted_code, adopted_summary
                incumbent_index = self.build_index(incumbent_code, docs)
                logger.info("iter %d: adopted %s → val R@100=%.4f", t, adopted_id, incumbent.recall_at_100)
            else:
                logger.info("iter %d: no candidate improved validation R@100", t)

            if log_dir:
                (log_dir / f"iteration_{t}.json").write_text(json.dumps({
                    "analysis": analysis,
                    "candidates": [{**asdict(r.candidate), "error": r.error, "proven": r.proven,
                                    "delta_recall_100": r.delta_recall_100} for r in results],
                    "adopted": adopted[0] if adopted else None,
                }, indent=2), encoding="utf-8")

        # Adoption requires a validation gain, so the incumbent is the best validation checkpoint.
        result = AutoIndexResult(best_code=incumbent_code, best_val=incumbent, initial_val=initial,
                                 history=history, usage=self.usage, analyses=analyses)
        if log_dir:
            (log_dir / "best_program.py").write_text(incumbent_code + "\n", encoding="utf-8")
            (log_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result

    def _select(self, t, results, analysis, incumbent_code, code_agent, docs, sample,
                val_queries, val_qrels, incumbent) -> Optional[Tuple[str, str, EvalSummary, bool]]:
        proven = [r for r in results if r.proven and r.error is None]
        if not proven:
            return None
        best = max(proven, key=lambda r: (r.summary.recall_at_100, r.summary.ndcg_at_10))
        choice = (best.candidate.id, best.candidate.code, best.summary, False)

        if self.config.use_synthesis and len(proven) > 1:
            try:
                merged = code_agent.synthesize(
                    analysis, [(r.candidate, r.delta_recall_100, r.delta_ndcg_10) for r in proven], incumbent_code)
            except Exception as e:
                logger.warning("iter %d: synthesis call failed (%s); keeping %s", t, e, best.candidate.id)
                merged = None
            if merged:
                synth = self._score_candidate(Candidate(f"it{t}-synth", "synthesis of " +
                                                        ", ".join(r.candidate.id for r in proven), code=merged),
                                              docs, sample, val_queries, val_qrels, incumbent)
                if synth.error is None and synth.summary.recall_at_100 > best.summary.recall_at_100:
                    choice = (synth.candidate.id, merged, synth.summary, True)
                else:
                    logger.info("iter %d: synthesis did not beat %s (%s)", t, best.candidate.id,
                                synth.error or f"R@100={synth.summary.recall_at_100:.4f}")
        return choice
