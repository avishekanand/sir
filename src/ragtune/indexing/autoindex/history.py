"""
Search history for AutoIndex: what was tried, and what happened on validation.

Rendered into the Analysis and Code Agent prompts when ``use_history`` is on
(the paper's "search history" ablation).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class IterationRecord:
    iteration: int
    recall_at_100: float
    ndcg_at_10: float
    recall_at_10: float
    adopted_id: Optional[str] = None
    synthesized: bool = False


@dataclass
class CandidateRecord:
    iteration: int
    candidate_id: str
    description: str
    rationale: str = ""
    delta_recall_100: float = 0.0
    delta_recall_10: float = 0.0
    delta_ndcg_10: float = 0.0
    proven: bool = False
    adopted: bool = False
    improved_query_ids: List[str] = field(default_factory=list)
    regressed_query_ids: List[str] = field(default_factory=list)
    error: Optional[str] = None


class SearchHistory:
    def __init__(self) -> None:
        self.iterations: List[IterationRecord] = []
        self.candidates: List[CandidateRecord] = []

    def record_iteration(self, rec: IterationRecord) -> None:
        self.iterations.append(rec)

    def record_candidate(self, rec: CandidateRecord) -> None:
        self.candidates.append(rec)

    def set_adoption(self, iteration: int, adopted_id: str, synthesized: bool = False) -> None:
        for rec in self.iterations:
            if rec.iteration == iteration:
                rec.adopted_id, rec.synthesized = adopted_id, synthesized
        for c in self.candidates:
            if c.iteration == iteration and c.candidate_id == adopted_id:
                c.adopted = True

    def render(self, max_candidates: int = 12) -> str:
        """Concise markdown summary for prompts."""
        if not self.iterations and not self.candidates:
            return "## Search history\n(none yet)"
        lines = ["## Search history", "", "### Validation score per iteration (incumbent program)"]
        prev = None
        for r in self.iterations:
            delta = f" ({r.recall_at_100 - prev:+.4f})" if prev is not None else ""
            adopted = f" → adopted {r.adopted_id}{' (synthesized)' if r.synthesized else ''}" if r.adopted_id else ""
            lines.append(f"- iter {r.iteration}: R@100={r.recall_at_100:.4f}{delta} nDCG@10={r.ndcg_at_10:.4f}{adopted}")
            prev = r.recall_at_100

        if self.candidates:
            n_proven = sum(c.proven for c in self.candidates)
            lines += ["", f"### Previously evaluated candidates ({n_proven}/{len(self.candidates)} improved validation)"]
            for c in self.candidates[-max_candidates:]:
                if c.error:
                    status = f"error: {c.error[:80]}"
                else:
                    tag = "adopted" if c.adopted else ("improved" if c.proven else "no gain")
                    status = (f"{tag}; ΔR@100={c.delta_recall_100:+.4f} ΔnDCG@10={c.delta_ndcg_10:+.4f}; "
                              f"fixed {len(c.improved_query_ids)}, broke {len(c.regressed_query_ids)}")
                lines.append(f"- iter {c.iteration} {c.candidate_id}: {c.description[:100]} — {status}")
            overfit = [c for c in self.candidates if c.adopted and len(c.regressed_query_ids) > len(c.improved_query_ids)]
            if overfit:
                lines.append("")
                lines.append("Warning: these adopted candidates broke more validation queries than they fixed: "
                             + ", ".join(f"{c.candidate_id} (iter {c.iteration})" for c in overfit))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"iterations": [asdict(r) for r in self.iterations],
                "candidates": [asdict(c) for c in self.candidates]}

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
