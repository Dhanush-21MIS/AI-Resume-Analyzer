"""
Evaluate app.job_matcher against a small manually labeled dataset.

Run from the backend directory:
    python -m python evaluation/evaluate_matcher.py

The dataset supplies requirements explicitly, so this script evaluates matching
behavior independently from requirement extraction. It temporarily replaces
the extractor during each example and restores it afterward.
"""

import json
from pathlib import Path

from app import job_matcher


DATASET_PATH = Path(__file__).with_name("evaluation_dataset.json")
VALID_STATUSES = {"matched", "partially_matched", "missing"}


def normalize_requirements(items):
    normalized = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"requirement": item, "importance": 1.0})
        else:
            normalized.append({
                "requirement": item["requirement"],
                "importance": item.get("importance", 1.0),
            })
    return normalized


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    original_extractor = job_matcher.extract_requirements

    total_requirements = 0
    exact_status_matches = 0
    false_positive = 0
    false_negative = 0
    per_status = {
        status: {"tp": 0, "fp": 0, "fn": 0}
        for status in VALID_STATUSES
    }

    print(f"Dataset: {DATASET_PATH}")
    print(f"Examples: {len(dataset['examples'])}\n")

    try:
        for example in dataset["examples"]:
            requirements = normalize_requirements(example["requirements"])
            job_matcher.extract_requirements = (
                lambda _job_description, top_n=30, reqs=requirements: reqs
            )

            result = job_matcher.match_resume_to_job(
                example["resume_text"],
                example["job_description"],
            )

            predicted = {}
            for bucket in (
                "matched_requirements",
                "partially_matched_requirements",
                "missing_requirements",
            ):
                for record in result[bucket]:
                    predicted[record["requirement"]] = record["status"]

            expected = example["expected"]
            example_correct = 0

            for requirement_name, expected_status in expected.items():
                actual_status = predicted.get(requirement_name, "missing")
                total_requirements += 1

                if actual_status == expected_status:
                    exact_status_matches += 1
                    example_correct += 1

                # Binary supported/not-supported view:
                expected_supported = expected_status != "missing"
                actual_supported = actual_status != "missing"

                if actual_supported and not expected_supported:
                    false_positive += 1
                elif expected_supported and not actual_supported:
                    false_negative += 1

                # Per-status one-vs-rest counts for transparent diagnostics.
                for status in VALID_STATUSES:
                    expected_is_status = expected_status == status
                    actual_is_status = actual_status == status
                    if expected_is_status and actual_is_status:
                        per_status[status]["tp"] += 1
                    elif actual_is_status and not expected_is_status:
                        per_status[status]["fp"] += 1
                    elif expected_is_status and not actual_is_status:
                        per_status[status]["fn"] += 1

            print(
                f"{example['id']}: "
                f"{example_correct}/{len(expected)} statuses correct | "
                f"score={result['match_percentage']:.2f}% | "
                f"coverage={result['coverage_percentage']:.2f}%"
            )

            for requirement_name, expected_status in expected.items():
                actual_status = predicted.get(requirement_name, "missing")
                mark = "OK" if actual_status == expected_status else "MISMATCH"
                print(
                    f"  [{mark}] {requirement_name}: "
                    f"expected={expected_status}, actual={actual_status}"
                )

    finally:
        job_matcher.extract_requirements = original_extractor

    status_accuracy = (
        exact_status_matches / total_requirements * 100
        if total_requirements else 0.0
    )

    print("\n=== Summary ===")
    print(f"Requirements evaluated: {total_requirements}")
    print(f"Exact status agreement: {exact_status_matches}/{total_requirements} "
          f"({status_accuracy:.2f}%)")
    print(f"False positives (unsupported marked supported): {false_positive}")
    print(f"False negatives (supported marked missing): {false_negative}")

    print("\nOne-vs-rest metrics by status:")
    for status, counts in per_status.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        print(
            f"  {status}: precision={precision:.3f}, recall={recall:.3f} "
            f"(TP={tp}, FP={fp}, FN={fn})"
        )

    print(
        "\nNote: This is a small starter dataset, not a general accuracy claim. "
        "Review mismatches manually and expand the labeled examples before "
        "using these metrics to compare model versions."
    )


if __name__ == "__main__":
    main()
