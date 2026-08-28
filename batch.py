import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from score import main as evaluate_single, print_evaluation_results
from models import EvaluationData
from transform import transform_evaluation_response
from config import DEVELOPMENT_MODE

logger = logging.getLogger(__name__)


def process_single_resume(pdf_path: str) -> Optional[dict]:
    try:
        result = evaluate_single(pdf_path)
        if result is None:
            return None

        candidate_name = os.path.basename(pdf_path).replace(".pdf", "")
        return {
            "file_name": os.path.basename(pdf_path),
            "candidate_name": candidate_name,
            "evaluation": result,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return {
            "file_name": os.path.basename(pdf_path),
            "candidate_name": os.path.basename(pdf_path).replace(".pdf", ""),
            "evaluation": None,
            "success": False,
            "error": str(e),
        }


def process_batch(folder_path: str) -> List[dict]:
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        return []

    pdf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'.")
        return []

    print(f"Found {len(pdf_files)} PDF files to process.")
    results = []

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {os.path.basename(pdf_path)}")
        result = process_single_resume(pdf_path)
        if result:
            results.append(result)
            if result["success"]:
                print(f"  ✓ Scored successfully")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")

    return results


def write_batch_csv(results: List[dict], output_path: str):
    import csv

    if not results:
        print("No results to write.")
        return

    successful_results = [r for r in results if r["success"] and r["evaluation"]]
    if not successful_results:
        print("No successful evaluations to write to CSV.")
        return

    csv_rows = []
    for result in successful_results:
        row = transform_evaluation_response(
            file_name=result["file_name"],
            evaluation=result["evaluation"],
        )
        csv_rows.append(row)

    if not csv_rows:
        return

    fieldnames = list(csv_rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nCSV written to: {output_path}")
    print(f"Total candidates: {len(csv_rows)}")


def print_batch_summary(results: List[dict]):
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\nTOP CANDIDATES BY SCORE:")
        print("-" * 60)

        scored = []
        for r in successful:
            if r["evaluation"] and hasattr(r["evaluation"], "scores"):
                scores = r["evaluation"].scores
                total = (
                    scores.motivation_fit.score
                    + scores.collaboration_perspective.score
                    + scores.values_judgment.score
                    + scores.commitments_experience.score
                )
                scored.append((r["candidate_name"], total))

        scored.sort(key=lambda x: x[1], reverse=True)

        for i, (name, score) in enumerate(scored[:10], 1):
            print(f"  {i}. {name}: {score:.1f}/100")

    if failed:
        print("\nFAILED FILES:")
        for r in failed:
            print(f"  - {r['file_name']}: {r.get('error', 'Unknown error')}")

    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch.py <folder_path> [output_csv]")
        print("Example: python batch.py ./resumes ./output/results.csv")
        exit(1)

    folder_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "batch_evaluations.csv"

    results = process_batch(folder_path)

    if results:
        write_batch_csv(results, output_csv)
        print_batch_summary(results)
