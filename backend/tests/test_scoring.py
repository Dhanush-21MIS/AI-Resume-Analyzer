"""Unit tests for deterministic matching score and coverage calculations.

Copy this file to backend/tests/test_scoring.py and run:
    python -m pytest tests/test_scoring.py -v
"""

import pytest

from app.job_matcher import calculate_coverage, calculate_weighted_score


def record(requirement, importance, score, status):
    return {
        "requirement": requirement,
        "importance": importance,
        "score": score,
        "status": status,
    }


def test_weighted_score_uses_importance_weights():
    records = [
        record("python", 1.0, 1.0, "matched"),
        record("aws", 3.0, 0.0, "missing"),
    ]
    # (1*1 + 3*0) / (1+3) * 100 = 25%
    assert calculate_weighted_score(records) == 25.0


def test_partial_match_contributes_partial_credit():
    records = [
        record("nosql", 1.0, 0.65, "partially_matched"),
        record("python", 1.0, 1.0, "matched"),
    ]
    assert calculate_weighted_score(records) == 82.5


def test_overlapping_api_requirements_do_not_double_count():
    records = [
        record("api development", 1.0, 1.0, "matched"),
        record("rest apis", 0.7, 0.0, "missing"),
        record("python", 1.0, 0.0, "missing"),
    ]
    # The API group contributes once at its strongest score: 1 point.
    # Python contributes 0; total = 1 / 2 = 50%.
    assert calculate_weighted_score(records) == 50.0


def test_score_is_clamped_to_zero_and_one():
    records = [
        record("python", 1.0, 1.5, "matched"),
        record("aws", 1.0, -0.5, "missing"),
    ]
    assert calculate_weighted_score(records) == 50.0


def test_negative_importance_does_not_reduce_score():
    records = [
        record("python", 1.0, 1.0, "matched"),
        record("aws", -2.0, 0.0, "missing"),
    ]
    assert calculate_weighted_score(records) == 100.0


def test_empty_records_return_zero_score_and_coverage():
    assert calculate_weighted_score([]) == 0.0
    assert calculate_coverage([]) == 0.0


def test_coverage_counts_full_and_partial_matches():
    records = [
        record("python", 1.0, 1.0, "matched"),
        record("nosql", 1.0, 0.65, "partially_matched"),
        record("aws", 1.0, 0.0, "missing"),
        record("redis", 1.0, 0.0, "missing"),
    ]
    assert calculate_coverage(records) == 50.0


def test_coverage_is_based_on_requirement_count_not_importance():
    records = [
        record("python", 10.0, 1.0, "matched"),
        record("aws", 1.0, 0.0, "missing"),
    ]
    assert calculate_coverage(records) == 50.0
