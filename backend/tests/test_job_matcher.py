"""Unit tests for the AI Resume Analyzer's deterministic matcher.

Copy this file to backend/tests/test_job_matcher.py and run pytest from
backend/. These tests isolate exact/equivalent matching from semantic matching
so that each assertion checks the deterministic rules.
"""

import pytest

from app import job_matcher


@pytest.fixture(autouse=True)
def disable_semantic_fallback(monkeypatch):
    """Keep tests focused on deterministic evidence rules."""
    monkeypatch.setattr(job_matcher, "find_semantic_evidence", lambda *_args, **_kwargs: None)


def match(resume_text: str, requirement: str) -> dict:
    return job_matcher.match_single_requirement(
        resume_text,
        {"requirement": requirement, "importance": 1.0},
    )


def test_api_development_requires_implementation_context():
    bare_skill_list = match("Skills: REST APIs, React.js, Node.js", "api development")
    assert bare_skill_list["status"] == "missing"

    project_evidence = match(
        "Designed and integrated RESTful APIs for authentication and review management.",
        "api development",
    )
    assert project_evidence["status"] == "matched"
    assert project_evidence["evidence"]


def test_mongodb_is_partial_evidence_for_nosql():
    result = match("Built a web application using MongoDB.", "nosql")
    assert result["status"] == "partially_matched"
    assert result["match_type"] == "related_technology"
    assert result["score"] == pytest.approx(0.65)


def test_react_native_alone_does_not_match_react_js():
    result = match("Developed a mobile application using React Native.", "react.js")
    assert result["status"] == "missing"


def test_explicit_react_js_matches():
    result = match("Built the frontend using React.js.", "react.js")
    assert result["status"] == "matched"


def test_ordinary_word_express_does_not_match_express_js():
    result = match("I express my ideas clearly and communicate well.", "express.js")
    assert result["status"] == "missing"


def test_express_js_explicitly_matches():
    result = match("Created backend routes with Express.js.", "express.js")
    assert result["status"] == "matched"


def test_deployment_requires_action_and_context():
    vague = match("Deployment is an important software engineering topic.", "deployment")
    assert vague["status"] == "missing"

    specific = match("Deployed the web application using Vercel.", "deployment")
    assert specific["status"] == "matched"


def test_missing_skill_has_zero_score_and_no_evidence():
    result = match("Frontend developer with HTML and CSS experience.", "kubernetes")
    assert result["status"] == "missing"
    assert result["score"] == 0.0
    assert result["evidence"] is None


def test_github_profile_url_alone_is_not_usage_evidence():
    result = match("Contact: https://github.com/example-user", "github")
    assert result["status"] == "missing"


def test_github_tool_list_is_recognized_as_a_skill_mention():
    result = match("Tools & Platforms: Git, GitHub, Docker", "github")
    assert result["status"] == "matched"


def test_empty_inputs_raise_value_error_for_full_matcher():
    with pytest.raises(ValueError, match="Resume text cannot be empty"):
        job_matcher.match_resume_to_job("   ", "Python developer")

    with pytest.raises(ValueError, match="Job description cannot be empty"):
        job_matcher.match_resume_to_job("Python developer", "   ")
