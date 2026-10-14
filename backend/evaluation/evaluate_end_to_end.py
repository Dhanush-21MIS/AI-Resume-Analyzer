"""
End-to-end evaluation for the real requirement extractor + resume matcher.

Run from backend:
    python -m evaluation.evaluate_end_to_end

The script calls the real match_resume_to_job() implementation. It compares
actual extracted requirements and match statuses with manually labeled data.
"""
import json
from collections import Counter
from pathlib import Path

from app.job_matcher import match_resume_to_job
from app.requirement_extractor import extract_requirements

DATASET_PATH = Path(__file__).with_name("end_to_end_dataset.json")
STATUSES = ("matched", "partially_matched", "missing")


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    extraction_tp = extraction_fp = extraction_fn = 0
    status_confusion = {s: Counter() for s in STATUSES}
    status_correct = status_total = 0
    case_passes = 0
    detailed_failures = []

    print("=" * 84)
    print("END-TO-END RESUME/JOB MATCHER EVALUATION")
    print("=" * 84)
    print(f"Dataset cases: {len(dataset['cases'])}")

    for case in dataset["cases"]:
        expected_requirements = set(case["expected_requirements"])
        extracted = extract_requirements(case["job_description"])
        actual_requirements = {item["requirement"] for item in extracted}

        extraction_tp += len(expected_requirements & actual_requirements)
        extraction_fp += len(actual_requirements - expected_requirements)
        extraction_fn += len(expected_requirements - actual_requirements)

        result = match_resume_to_job(case["resume_text"], case["job_description"])
        actual_statuses = {}
        for key in ("matched_requirements", "partially_matched_requirements", "missing_requirements"):
            status = {
                "matched_requirements": "matched",
                "partially_matched_requirements": "partially_matched",
                "missing_requirements": "missing",
            }[key]
            for record in result.get(key, []):
                actual_statuses[record["requirement"]] = status

        expected_statuses = case["expected_statuses"]
        case_issues = []

        for requirement in sorted(expected_requirements):
            expected_status = expected_statuses[requirement]
            actual_status = actual_statuses.get(requirement, "not_extracted")
            status_total += 1
            if actual_status == expected_status:
                status_correct += 1
            else:
                case_issues.append(
                    f"{requirement}: expected={expected_status}, actual={actual_status}"
                )

            if actual_status in STATUSES:
                status_confusion[expected_status][actual_status] += 1

        unexpected_requirements = sorted(actual_requirements - expected_requirements)
        if unexpected_requirements:
            case_issues.append(f"unexpected extracted requirements={unexpected_requirements}")

        passed = not case_issues
        case_passes += int(passed)
        print(
            f"[{'PASS' if passed else 'FAIL'}] {case['id']} | "
            f"score={result.get('match_percentage', 0):.2f}% | "
            f"coverage={result.get('coverage_percentage', 0):.2f}% | "
            f"requirements={result.get('requirements_analyzed', 0)}"
        )
        if case_issues:
            detailed_failures.append((case["id"], case_issues))
            for issue in case_issues:
                print(f"    - {issue}")

    ep, er, ef1 = prf(extraction_tp, extraction_fp, extraction_fn)
    print("\n" + "=" * 84)
    print("REQUIREMENT EXTRACTION")
    print("=" * 84)
    print(f"TP={extraction_tp} | FP={extraction_fp} | FN={extraction_fn}")
    print(f"Precision: {ep:.3f} ({ep * 100:.1f}%)")
    print(f"Recall:    {er:.3f} ({er * 100:.1f}%)")
    print(f"F1-score:  {ef1:.3f} ({ef1 * 100:.1f}%)")

    print("\n" + "=" * 84)
    print("MATCH STATUS AGREEMENT")
    print("=" * 84)
    print(f"Correct requirement statuses: {status_correct}/{status_total}")
    print(f"Status agreement: {status_correct / status_total * 100 if status_total else 0:.1f}%")
    print("\nExpected-status rows / actual-status columns:")
    print(f"{'Expected':<22}" + "".join(f"{s:>22}" for s in STATUSES))
    for expected in STATUSES:
        print(f"{expected:<22}" + "".join(f"{status_confusion[expected][actual]:>22}" for actual in STATUSES))

    print("\n" + "=" * 84)
    print("SUMMARY")
    print("=" * 84)
    print(f"Cases passing all expected checks: {case_passes}/{len(dataset['cases'])}")
    print(f"Cases with mismatches: {len(detailed_failures)}")
    if detailed_failures:
        print("\nReview these mismatches manually before changing matcher behavior.")
        for case_id, issues in detailed_failures:
            print(f"- {case_id}: {'; '.join(issues)}")

    print("\nLimitations: this is a synthetic, manually labeled regression dataset.")
    print("Its scores are diagnostic only and are not production accuracy estimates.")


if __name__ == "__main__":
    main()
