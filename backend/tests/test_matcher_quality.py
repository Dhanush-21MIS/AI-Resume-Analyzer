"""
Quality-focused tests for app.job_matcher.

Run from the backend directory:
    python -m pytest tests/test_matcher_quality.py -v

These tests mock requirement extraction so they evaluate the matcher independently
from the job-description requirement extractor. Semantic matching is mocked for
deterministic results.
"""

import pytest

from app import job_matcher


def requirement(name: str, importance: float = 1.0) -> dict:
    return {"requirement": name, "importance": importance}


@pytest.fixture
def controlled_extractor(monkeypatch):
    """Return a helper that supplies known requirements to the matcher."""

    def set_requirements(requirements):
        monkeypatch.setattr(
            job_matcher,
            "extract_requirements",
            lambda _job_description, top_n=30: requirements,
        )

    return set_requirements


def test_exact_technical_skills_are_matched(controlled_extractor):
    controlled_extractor([
        requirement("Python"),
        requirement("FastAPI"),
    ])

    result = job_matcher.match_resume_to_job(
        "Skills: Python, FastAPI.",
        "A job requiring Python and FastAPI.",
    )

    assert result["matched_count"] == 2
    assert result["partially_matched_count"] == 0
    assert result["missing_count"] == 0
    assert result["match_percentage"] == 100.0
    assert result["coverage_percentage"] == 100.0


def test_mongodb_is_partial_evidence_for_nosql(controlled_extractor):
    controlled_extractor([requirement("NoSQL")])

    result = job_matcher.match_resume_to_job(
        "Database: MongoDB.",
        "Experience with NoSQL databases.",
    )

    assert result["matched_count"] == 0
    assert result["partially_matched_count"] == 1
    assert result["missing_count"] == 0
    assert result["partially_matched_requirements"][0]["score"] == 0.65
    assert result["coverage_percentage"] == 100.0
    assert result["match_percentage"] == 65.0


def test_react_native_does_not_match_react_js(controlled_extractor):
    controlled_extractor([requirement("React.js")])

    result = job_matcher.match_resume_to_job(
        "Built a mobile app with React Native.",
        "The role requires React.js.",
    )

    assert result["matched_count"] == 0
    assert result["partially_matched_count"] == 0
    assert result["missing_count"] == 1
    assert result["coverage_percentage"] == 0.0
    assert result["match_percentage"] == 0.0


def test_api_development_requires_implementation_context(controlled_extractor):
    controlled_extractor([requirement("API development")])

    result = job_matcher.match_resume_to_job(
        "Technical skills: REST APIs.",
        "The role requires API development.",
    )

    assert result["matched_count"] == 0
    assert result["partially_matched_count"] == 0
    assert result["missing_count"] == 1


def test_api_development_matches_when_action_and_api_context_exist(
    controlled_extractor,
):
    controlled_extractor([requirement("API development")])

    result = job_matcher.match_resume_to_job(
        "Designed and implemented REST APIs for a web application.",
        "The role requires API development.",
    )

    assert result["matched_count"] == 1
    assert result["missing_count"] == 0
    assert result["match_percentage"] == 100.0


def test_unrelated_resume_has_no_technical_skill_matches(controlled_extractor):
    controlled_extractor([
        requirement("Python"),
        requirement("FastAPI"),
        requirement("PostgreSQL"),
    ])

    result = job_matcher.match_resume_to_job(
        "Graphic designer experienced in typography, branding, and layouts.",
        "A backend role requiring Python, FastAPI, and PostgreSQL.",
    )

    assert result["matched_count"] == 0
    assert result["partially_matched_count"] == 0
    assert result["missing_count"] == 3
    assert result["coverage_percentage"] == 0.0
    assert result["match_percentage"] == 0.0


def test_weighted_score_respects_importance(controlled_extractor):
    controlled_extractor([
        requirement("Python", importance=3.0),
        requirement("PostgreSQL", importance=1.0),
    ])

    result = job_matcher.match_resume_to_job(
        "Skills: Python.",
        "A role requiring Python and PostgreSQL.",
    )

    assert result["matched_count"] == 1
    assert result["missing_count"] == 1
    assert result["match_percentage"] == 75.0
    assert result["coverage_percentage"] == 50.0


def test_coverage_counts_partial_matches_as_covered():
    records = [
        {"requirement": "A", "status": "matched"},
        {"requirement": "B", "status": "partially_matched"},
        {"requirement": "C", "status": "missing"},
    ]

    assert job_matcher.calculate_coverage(records) == pytest.approx(66.67)


def test_empty_resume_raises_value_error():
    with pytest.raises(ValueError, match="Resume text cannot be empty"):
        job_matcher.match_resume_to_job(
            "",
            "A job requiring Python.",
        )


def test_empty_job_description_raises_value_error():
    with pytest.raises(ValueError, match="Job description cannot be empty"):
        job_matcher.match_resume_to_job(
            "Python developer.",
            "",
        )


def test_no_extracted_requirements_returns_zero_metrics(controlled_extractor):
    controlled_extractor([])

    result = job_matcher.match_resume_to_job(
        "Python developer.",
        "A job description without extractable requirements.",
    )

    assert result["requirements_analyzed"] == 0
    assert result["match_percentage"] == 0.0
    assert result["coverage_percentage"] == 0.0
    assert result["recommendations"][0]["type"] == "review_job_description"
