import io

import pytest
from fastapi.testclient import TestClient

from app import main


client = TestClient(main.app)


def upload_file(
    filename: str = "resume.pdf",
    content: bytes = b"sample resume content",
):
    return {
        "resume": (
            filename,
            io.BytesIO(content),
            "application/pdf" if filename.lower().endswith(".pdf")
            else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    }


@pytest.fixture
def mock_extraction(monkeypatch):
    def fake_extract_resume_text(file_bytes, filename):
        return "Python developer with FastAPI, SQL, and project experience."

    monkeypatch.setattr(main, "extract_resume_text", fake_extract_resume_text)


def test_home_returns_running_status():
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["health"] == "/api/health"
    assert "/docs" == data["docs"]


def test_health_check():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "message": "API is running.",
    }


def test_extract_resume_success(monkeypatch):
    monkeypatch.setattr(
        main,
        "extract_resume_text",
        lambda file_bytes, filename: "Python developer with FastAPI.",
    )

    response = client.post(
        "/api/resume/extract",
        files=upload_file(),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "resume.pdf"
    assert data["resume_text"] == "Python developer with FastAPI."
    assert data["characters_extracted"] == len(data["resume_text"])
    assert data["word_count"] == 4


def test_rejects_unsupported_file_extension():
    response = client.post(
        "/api/resume/extract",
        files=upload_file("resume.txt", b"plain text"),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF and DOCX files are supported."


def test_rejects_empty_uploaded_file():
    response = client.post(
        "/api/resume/extract",
        files=upload_file("resume.pdf", b""),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty."


def test_rejects_file_larger_than_5_mb():
    response = client.post(
        "/api/resume/extract",
        files=upload_file("resume.pdf", b"x" * (main.MAX_FILE_SIZE + 1)),
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "File size must not exceed 5 MB."


def test_extract_returns_422_when_no_readable_text(monkeypatch):
    monkeypatch.setattr(main, "extract_resume_text", lambda file_bytes, filename: " ")

    response = client.post(
        "/api/resume/extract",
        files=upload_file(),
    )

    assert response.status_code == 422
    assert "No readable text" in response.json()["detail"]


def test_analyze_resume_success(monkeypatch):
    monkeypatch.setattr(
        main,
        "extract_resume_text",
        lambda file_bytes, filename: "Python developer with FastAPI.",
    )
    monkeypatch.setattr(
        main,
        "analyze_resume",
        lambda resume_text: {"completeness_score": 80, "skills": ["Python", "FastAPI"]},
    )

    response = client.post(
        "/api/resume/analyze",
        files=upload_file(),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "resume.pdf"
    assert data["analysis"]["completeness_score"] == 80
    assert data["analysis"]["skills"] == ["Python", "FastAPI"]


def test_match_resume_success(monkeypatch, mock_extraction):
    expected_result = {
        "match_percentage": 75.0,
        "matched_requirements": ["Python"],
        "missing_requirements": ["Docker"],
    }
    monkeypatch.setattr(main, "match_resume_to_job", lambda resume, jd: expected_result)

    response = client.post(
        "/api/resume/match",
        files=upload_file(),
        data={"job_description": "Looking for Python and Docker."},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "resume.pdf"
    assert data["job_description_length"] == len("Looking for Python and Docker.")
    assert data["matching_result"] == expected_result


def test_match_rejects_empty_job_description():
    response = client.post(
        "/api/resume/match",
        files=upload_file(),
        data={"job_description": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Job description cannot be empty."


def test_match_rejects_job_description_over_limit():
    response = client.post(
        "/api/resume/match",
        files=upload_file(),
        data={"job_description": "x" * (main.MAX_JOB_DESCRIPTION_LENGTH + 1)},
    )

    assert response.status_code == 413
    assert "50,000 characters" in response.json()["detail"]


def test_llm_analysis_success(monkeypatch, mock_extraction):
    matching_result = {"match_percentage": 70.0}
    llm_result = {"explanation": "Good Python match.", "recommendations": ["Add Docker"]}

    monkeypatch.setattr(
        main,
        "match_resume_to_job",
        lambda resume, jd: matching_result,
    )

    def fake_llm_analysis(resume_text, job_description, matching_result):
        return llm_result

    monkeypatch.setattr(main, "analyze_resume_with_llm", fake_llm_analysis)

    response = client.post(
        "/api/resume/analyze-llm",
        files=upload_file(),
        data={"job_description": "Python developer"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "resume.pdf"
    assert data["matching_result"] == matching_result
    assert data["llm_analysis"] == llm_result


def test_llm_runtime_error_returns_503(monkeypatch, mock_extraction):
    monkeypatch.setattr(
        main,
        "match_resume_to_job",
        lambda resume, jd: {"match_percentage": 70.0},
    )

    def fail_llm(**kwargs):
        raise RuntimeError("Simulated provider outage")

    monkeypatch.setattr(main, "analyze_resume_with_llm", fail_llm)

    response = client.post(
        "/api/resume/analyze-llm",
        files=upload_file(),
        data={"job_description": "Python developer"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "AI analysis is temporarily unavailable."


def test_matcher_value_error_returns_422(monkeypatch, mock_extraction):
    def fail_matcher(resume, jd):
        raise ValueError("Invalid resume/job input")

    monkeypatch.setattr(main, "match_resume_to_job", fail_matcher)

    response = client.post(
        "/api/resume/match",
        files=upload_file(),
        data={"job_description": "Python developer"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid resume/job input"
