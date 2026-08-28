"""The one definition of a candidate's total score.

Previously three different totals existed (CSV export, results table, CLI
printer) and they disagreed, so the number a reviewer saw on screen was not the
number in the downloadable file. Everything now routes through here.

The score is the four rubric categories, each capped at its maximum: a flat
0-100 with no bonus or deduction adjustments.
"""

from typing import Any, Dict, Optional

from models import CATEGORY_MAX

CATEGORY_ORDER = (
    "motivation_fit",
    "collaboration_perspective",
    "values_judgment",
    "commitments_experience",
)

CATEGORY_LABELS = {
    "motivation_fit": "Motivation & Fit",
    "collaboration_perspective": "Collaboration & Perspective-Taking",
    "values_judgment": "Values & Judgment",
    "commitments_experience": "Commitments, Leadership & Experience",
}

TOTAL_MAX = CATEGORY_MAX * len(CATEGORY_ORDER)


def _as_scores_dict(evaluation: Any) -> Dict[str, Dict[str, Any]]:
    """Accept either an EvaluationData model or a plain dict."""
    if evaluation is None:
        return {}
    scores = getattr(evaluation, "scores", None)
    if scores is None and isinstance(evaluation, dict):
        scores = evaluation.get("scores")
    if scores is None:
        return {}
    if hasattr(scores, "model_dump"):
        return scores.model_dump()
    return scores if isinstance(scores, dict) else {}


def category_score(evaluation: Any, category: str) -> float:
    """A single category's score, clamped to its maximum."""
    entry = _as_scores_dict(evaluation).get(category) or {}
    try:
        value = float(entry.get("score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(value, float(CATEGORY_MAX)))


def compute_total(evaluation: Any) -> float:
    """Total score out of 100: the four categories, each clamped."""
    return sum(category_score(evaluation, c) for c in CATEGORY_ORDER)


def sort_key(candidate: Dict[str, Any]):
    """Rank by score descending, then name -- never by submission order.

    Ties were previously broken by CSV row order, which quietly made "applied
    earlier" a ranking criterion.
    """
    return (-float(candidate.get("total_score") or 0.0),
            str(candidate.get("candidate_name") or "").lower())
