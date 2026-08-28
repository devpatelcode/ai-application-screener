import os
import sys
import json
import logging
import csv

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from extraction import extract_resume_text
from models import EvaluationData
from typing import List, Optional, Dict
from evaluator import ResumeEvaluator
from pathlib import Path
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from config import DEVELOPMENT_MODE
from models import CATEGORY_MAX
from scoring import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    TOTAL_MAX,
    category_score,
    compute_total,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)5s - %(lineno)5d - %(funcName)33s - %(levelname)5s - %(message)s",
)


def print_evaluation_results(
    evaluation: EvaluationData, candidate_name: str = "Candidate"
):
    """Print evaluation results, using the shared score definition.

    This printer used to compute its own total (categories + bonus - deductions,
    capped at 120) which disagreed with both the UI and the CSV export.
    """
    print("\n" + "=" * 80)
    print(f"APPLICATION EVALUATION: {candidate_name}")
    print("=" * 80)

    if not evaluation:
        print("No evaluation data available")
        return

    print(f"\nOVERALL SCORE: {compute_total(evaluation):.1f}/{TOTAL_MAX}\n")
    print("DETAILED SCORES:")
    print("-" * 60)

    for key in CATEGORY_ORDER:
        entry = getattr(evaluation.scores, key, None)
        if not entry:
            continue
        print(f"{CATEGORY_LABELS[key]}: {category_score(evaluation, key):.0f}/{CATEGORY_MAX}")
        print(f"   Evidence: {entry.evidence}\n")

    if getattr(evaluation, "key_strengths", None):
        print("KEY STRENGTHS:")
        print("-" * 30)
        for i, s in enumerate(evaluation.key_strengths, 1):
            print(f"  {i}. {s}")

    if getattr(evaluation, "areas_for_improvement", None):
        print("\nAREAS FOR IMPROVEMENT:")
        print("-" * 30)
        for i, a in enumerate(evaluation.areas_for_improvement, 1):
            print(f"  {i}. {a}")

    print("\n" + "=" * 80)


def main(pdf_path):
    """Score a single resume file from the CLI.

    Uses the same one-call extraction path as the web app: read the file to
    text, then score it. The old flow re-parsed each resume into JSON Resume
    over 6-12 LLM calls and cached the result under a filename-derived key --
    which collided whenever two applicants both submitted "resume.pdf",
    silently scoring one candidate on another's data.

    Note this path has no essay answers, so the three essay categories are
    scored from resume content alone. The web CSV+ZIP flow is the intended
    entry point for real application review.
    """
    result = extract_resume_text(pdf_path)
    if not result.ok:
        print(f"Could not read {pdf_path}: {result.error}")
        return None

    model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
    evaluator = ResumeEvaluator(model_name=DEFAULT_MODEL, model_params=model_params)

    text = (
        "Why interested essay:\nNot provided\n"
        "\nCollaboration essay:\nNot provided\n"
        "\nValues essay:\nNot provided\n"
        "\nCurrent commitments:\nNot provided\n"
        f"\nResume:\n{result.text}"
    )

    score = evaluator.evaluate_resume(text)
    print_evaluation_results(score, Path(pdf_path).stem)
    return score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python score.py <pdf_path_or_folder> [output_csv]")
        print("Examples:")
        print("  python score.py ./resume/sample.pdf")
        print("  python score.py ./resumes/")
        print("  python score.py ./resumes/ ./output/results.csv")
        exit(1)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print(f"Error: Path '{input_path}' does not exist.")
        exit(1)

    if os.path.isdir(input_path):
        from batch import process_batch, write_batch_csv, print_batch_summary
        output_csv = sys.argv[2] if len(sys.argv) > 2 else "batch_evaluations.csv"
        results = process_batch(input_path)
        if results:
            write_batch_csv(results, output_csv)
            print_batch_summary(results)
    else:
        main(input_path)
