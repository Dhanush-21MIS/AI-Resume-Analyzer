"""
Evaluate app.requirement_extractor.extract_requirements against manually labeled cases.

Run from the backend directory:
    python -m evaluation.evaluate_extractor

This is a small diagnostic evaluation. Do not treat its scores as general model accuracy.
"""
import json
import sys
from pathlib import Path

# The project root should be the current working directory when run as a module.
from app.requirement_extractor import extract_requirements

DATASET_PATH = Path(__file__).with_name("extractor_dataset.json")


def load_dataset():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1


def main():
    data = load_dataset()
    total_tp = total_fp = total_fn = 0
    importance_checks = []

    print("=" * 76)
    print("REQUIREMENT EXTRACTOR EVALUATION")
    print("=" * 76)

    for case in data["cases"]:
        expected = set(case["expected_requirements"])
        results = extract_requirements(case["job_description"])
        actual = {item["requirement"] for item in results}

        tp = len(expected & actual)
        fp = len(actual - expected)
        fn = len(expected - actual)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        passed = not missing and not unexpected

        print(f"\n[{ 'PASS' if passed else 'FAIL' }] {case['id']}")
        print(f"  Expected ({len(expected)}): {sorted(expected)}")
        print(f"  Extracted ({len(actual)}): {sorted(actual)}")
        if missing:
            print(f"  Missed requirements: {missing}")
        if unexpected:
            print(f"  Unexpected requirements: {unexpected}")

        expected_importance = case.get("expected_importance", {})
        if expected_importance:
            by_name = {item["requirement"]: item["importance"] for item in results}
            for requirement, expected_weight in expected_importance.items():
                if requirement in by_name:
                    correct = abs(by_name[requirement] - expected_weight) < 0.001
                    importance_checks.append(correct)
                    if not correct:
                        print(
                            f"  Importance mismatch for {requirement}: "
                            f"expected {expected_weight}, got {by_name[requirement]}"
                        )
                else:
                    importance_checks.append(False)
                    print(f"  Importance could not be checked: {requirement} was not extracted")

    precision, recall, f1 = safe_prf(total_tp, total_fp, total_fn)
    print("\n" + "=" * 76)
    print("AGGREGATE REQUIREMENT METRICS (micro-averaged)")
    print("=" * 76)
    print(f"True positives : {total_tp}")
    print(f"False positives: {total_fp}")
    print(f"False negatives: {total_fn}")
    print(f"Precision      : {precision:.3f} ({precision * 100:.1f}%)")
    print(f"Recall         : {recall:.3f} ({recall * 100:.1f}%)")
    print(f"F1-score       : {f1:.3f} ({f1 * 100:.1f}%)")

    if importance_checks:
        accuracy = sum(importance_checks) / len(importance_checks)
        print(f"Importance checks: {sum(importance_checks)}/{len(importance_checks)} ({accuracy * 100:.1f}%)")

    top_n = data.get("top_n_check")
    if top_n:
        result = extract_requirements(top_n["job_description"], top_n=top_n["top_n"])
        passed = len(result) == top_n["expected_count"]
        print("\n" + "=" * 76)
        print("TOP_N LIMIT CHECK")
        print("=" * 76)
        print(f"Requested top_n={top_n['top_n']}; extracted {len(result)} requirements")
        print("Result:", "PASS" if passed else "FAIL")
        print("Returned:", [item["requirement"] for item in result])

    print("\nNote: this dataset is small and manually labeled. Review every mismatch;")
    print("do not interpret the percentages as production-level accuracy.")

if __name__ == "__main__":
    main()
