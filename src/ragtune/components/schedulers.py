from typing import List, Optional, Union
from ragtune.core.interfaces import BaseScheduler
from ragtune.core.types import BatchProposal, RemainingBudgetView, CostObject
from ragtune.core.pool import CandidatePool, PoolItem, ItemState
from ragtune.registry import registry


# ── Token estimation helper ───────────────────────────────────────────────
# Shuvam (2026-07-27): Initial estimate for per-batch token cost.
# This is a conservative estimate — users may want to configure it
# differently based on their reranker's actual token consumption:
#   - CrossEncoder: ~query_tokens + ~doc_tokens per pair
#   - LLM reranker: ~query_tokens + ~doc_tokens + ~prompt_overhead
#   - NoOp reranker: 0 tokens (no computation)
# The default (512 per doc) is a reasonable upper bound for most use cases.

_DEFAULT_TOKENS_PER_DOC = 512


def _estimate_batch_tokens(
    items: List[PoolItem], default_tokens: int = _DEFAULT_TOKENS_PER_DOC
) -> int:
    """Estimate total tokens for a batch of documents.

    Uses token_count from document metadata if available, otherwise
    falls back to default_tokens per document.

    Args:
        items: PoolItems selected for reranking.
        default_tokens: Fallback token count per document when metadata
            doesn't contain token_count. Default 512.

    Returns:
        Estimated total tokens for the batch.
    """
    total = 0
    for item in items:
        total += item.metadata.get("token_count", default_tokens)
    return total


@registry.scheduler("active-learning")
class ActiveLearningScheduler(BaseScheduler):
    def __init__(
        self,
        batch_size: int = 5,
        strategy: str = "cross_encoder",
    ):
        self.batch_size = batch_size
        self.strategy = strategy

    def select_batch(
        self, pool: CandidatePool, budget: RemainingBudgetView
    ) -> Optional[BatchProposal]:
        eligible = pool.get_eligible()
        if not eligible or budget.remaining_rerank_docs <= 0:
            return None

        # Sort by priority_value (set by Estimator in controller loop)
        # Tie-break by initial_rank then doc_id
        eligible.sort(key=lambda x: (-x.priority_value, x.initial_rank, x.doc_id))

        batch_size = min(self.batch_size, budget.remaining_rerank_docs, len(eligible))
        if batch_size <= 0:
            return None

        selected = eligible[:batch_size]
        doc_ids = [it.doc_id for it in selected]

        # Strategy escalation: when the top two candidates are nearly tied
        # (priority gap < 5%), upgrade from cross_encoder to llm to break the
        # tie with a more expensive but higher-fidelity signal.
        current_strategy = self.strategy
        if len(selected) >= 2:
            gap = selected[0].priority_value - selected[1].priority_value
            if gap < 0.05 and current_strategy == "cross_encoder":
                current_strategy = "llm"

        # Estimate per-batch token cost for budget-aware loop control.
        # Shuvam (2026-07-27): Uses metadata token_count when available,
        # falls back to _DEFAULT_TOKENS_PER_DOC. Users with custom rerankers
        # should override this via strategy-specific token estimation.
        batch_tokens = _estimate_batch_tokens(selected)

        # Skip batch if token budget is insufficient — prevents wasted reranking
        # when no document can fit in the remaining token budget.
        if budget.remaining_tokens > 0 and batch_tokens > budget.remaining_tokens:
            return None

        return BatchProposal(
            doc_ids=doc_ids,
            strategy=current_strategy,
            expected_cost=CostObject(docs=len(doc_ids), calls=1, tokens=batch_tokens),
        )


@registry.scheduler("graceful-degradation")
class GracefulDegradationScheduler(BaseScheduler):
    def __init__(
        self, llm_limit: int = 3, cross_encoder_limit: int = 10, batch_size: int = 5
    ):
        self.llm_limit = llm_limit
        self.cross_encoder_limit = cross_encoder_limit
        self.batch_size = batch_size

    def select_batch(
        self, pool: CandidatePool, budget: RemainingBudgetView
    ) -> Optional[BatchProposal]:
        eligible = pool.get_eligible()
        if not eligible or budget.remaining_rerank_docs <= 0:
            return None

        # Count how many have been reranked by each strategy
        active = pool.get_active_items()
        reranked = [it for it in active if it.state == ItemState.RERANKED]
        num_llm = len([it for it in reranked if it.reranker_strategy == "llm"])
        num_ce = len([it for it in reranked if it.reranker_strategy == "cross_encoder"])

        if num_llm < self.llm_limit:
            strategy = "llm"
            rem_strategy = self.llm_limit - num_llm
        elif num_ce < self.cross_encoder_limit:
            strategy = "cross_encoder"
            rem_strategy = self.cross_encoder_limit - num_ce
        else:
            return None

        # Sort eligible by priority_value (from retrieval score usually)
        eligible.sort(key=lambda x: (-x.priority_value, x.initial_rank, x.doc_id))

        batch_size = min(
            self.batch_size, budget.remaining_rerank_docs, rem_strategy, len(eligible)
        )
        if batch_size <= 0:
            return None

        selected = eligible[:batch_size]

        # Estimate per-batch token cost for budget-aware loop control.
        # Shuvam (2026-07-27): Same estimation as ActiveLearningScheduler.
        batch_tokens = _estimate_batch_tokens(selected)

        # Skip batch if token budget is insufficient
        if budget.remaining_tokens > 0 and batch_tokens > budget.remaining_tokens:
            return None

        return BatchProposal(
            doc_ids=[it.doc_id for it in selected],
            strategy=strategy,
            expected_cost=CostObject(docs=len(selected), calls=1, tokens=batch_tokens),
        )
