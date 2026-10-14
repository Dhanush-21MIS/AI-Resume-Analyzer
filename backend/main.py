import logging

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware

from app.document_parser import extract_resume_text
from app.resume_analyzer import analyze_resume
from app.job_matcher import match_resume_to_job
from app.llm_service import analyze_resume_with_llm


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_JOB_DESCRIPTION_LENGTH = 50_000

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
}


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="AI Resume Analyzer and Job Matcher",
    description=(
        "Extract resume text, analyze resume completeness, "
        "identify keywords, compare resumes against job "
        "descriptions, and generate LLM-powered explanations "
        "and recommendations."
    ),
    version="0.6.0",
)


# ---------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------
# Keep only the frontend origins you actually use.
# If Vite starts on a different port, add that exact origin.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# File validation
# ---------------------------------------------------------

def validate_filename(filename: str) -> None:
    filename_lower = filename.lower()

    if not any(
        filename_lower.endswith(extension)
        for extension in SUPPORTED_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported.",
        )


async def read_uploaded_file(resume: UploadFile) -> tuple[str, bytes]:
    """
    Validate and read an uploaded resume.

    Reads at most MAX_FILE_SIZE + 1 bytes so oversized files
    can be rejected without reading the entire upload.
    """
    filename = resume.filename or ""

    try:
        validate_filename(filename)

        file_bytes = await resume.read(MAX_FILE_SIZE + 1)

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size must not exceed 5 MB.",
            )

        return filename, file_bytes

    finally:
        await resume.close()


# ---------------------------------------------------------
# Resume text extraction
# ---------------------------------------------------------

def extract_text_or_raise(file_bytes: bytes, filename: str) -> str:
    try:
        text = extract_resume_text(file_bytes, filename)

        if not text or not text.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "No readable text was found in the document. "
                    "If this is a scanned PDF, OCR may be required."
                ),
            )

        return text

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Document extraction failed.")
        raise HTTPException(
            status_code=422,
            detail="Unable to read the uploaded document.",
        ) from exc


# ---------------------------------------------------------
# Shared job-description validation
# ---------------------------------------------------------

def validate_job_description(job_description: str) -> None:
    if not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty.",
        )

    if len(job_description) > MAX_JOB_DESCRIPTION_LENGTH:
        raise HTTPException(
            status_code=413,
            detail="Job description must not exceed 50,000 characters.",
        )


# ---------------------------------------------------------
# Home
# ---------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to AI Resume Analyzer and Job Matcher",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
        "features": [
            "Resume text extraction",
            "Resume completeness analysis",
            "Resume-job matching",
            "LLM-based resume analysis and recommendations",
        ],
    }


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "message": "API is running.",
    }


# ---------------------------------------------------------
# Extract resume text
# ---------------------------------------------------------

@app.post("/api/resume/extract")
async def extract_resume(resume: UploadFile = File(...)):
    filename, file_bytes = await read_uploaded_file(resume)
    text = extract_text_or_raise(file_bytes, filename)

    return {
        "filename": filename,
        "characters_extracted": len(text),
        "word_count": len(text.split()),
        "resume_text": text,
    }


# ---------------------------------------------------------
# Analyze resume completeness
# ---------------------------------------------------------

@app.post("/api/resume/analyze")
async def analyze_uploaded_resume(resume: UploadFile = File(...)):
    filename, file_bytes = await read_uploaded_file(resume)
    text = extract_text_or_raise(file_bytes, filename)

    try:
        analysis = analyze_resume(text)

        # Ensure the frontend receives a numeric, explainable score.
        # This is a heuristic resume-completeness estimate, not an official ATS score.
        if not isinstance(analysis, dict):
            analysis = {"analysis_result": analysis}

        normalized_text = text.lower()
        section_patterns = {
            "contact_information": ["@", "phone", "mobile", "linkedin", "github"],
            "summary": ["professional summary", "career objective", "profile summary", "objective"],
            "education": ["education", "university", "college", "degree", "bachelor", "master"],
            "experience": ["experience", "employment", "work history", "internship"],
            "skills": ["skills", "technical skills", "technologies", "tech stack"],
            "projects": ["projects", "project experience", "academic projects"],
            "certifications": ["certifications", "certificates", "certification"],
        }
        weights = {
            "contact_information": 15,
            "summary": 10,
            "education": 15,
            "experience": 20,
            "skills": 15,
            "projects": 15,
            "certifications": 10,
        }
        section_presence = {
            name: any(pattern in normalized_text for pattern in patterns)
            for name, patterns in section_patterns.items()
        }
        score = round(sum(weights[name] for name, present in section_presence.items() if present))
        word_count = len(text.split())
        if word_count < 80:
            score = min(score, 45)
        elif word_count < 150:
            score = min(score, 65)

        # Prefer a valid numeric score already returned by the analyzer;
        # otherwise use the computed section-completeness estimate.
        existing_score = None
        for key in (
            "ats_score",
            "ats_compatibility_score",
            "completeness_score",
            "resume_completeness_score",
            "resume_score",
        ):
            candidate = analysis.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                if 0 <= candidate <= 100:
                    existing_score = round(float(candidate), 1)
                    break

        final_score = existing_score if existing_score is not None else score

        # Publish consistent aliases so different frontend versions can read the score.
        analysis["ats_score"] = final_score
        analysis["ats_compatibility_score"] = final_score
        analysis["completeness_score"] = final_score
        analysis["resume_completeness_score"] = final_score
        analysis["resume_score"] = final_score
        analysis.setdefault("score_breakdown", {
            "section_presence": section_presence,
            "weights": weights,
            "word_count": word_count,
            "method": "heuristic resume completeness estimate",
            "score_used": final_score,
        })
        analysis.setdefault("missing_sections", [
            name.replace("_", " ").title()
            for name, present in section_presence.items()
            if not present
        ])
        analysis.setdefault("score_disclaimer", "Estimated resume completeness/ATS compatibility indicator; not an official score from any ATS provider.")
        analysis.setdefault("score_label", "Estimated ATS Compatibility")

        return {
            "filename": filename,
            "analysis": analysis,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Resume analysis failed.")
        raise HTTPException(
            status_code=500,
            detail="Unable to analyze the resume.",
        ) from exc


# ---------------------------------------------------------
# Match resume against job description
# ---------------------------------------------------------

@app.post("/api/resume/match")
async def match_uploaded_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    """
    Compare an uploaded resume against a job description.

    This endpoint preserves the existing deterministic matcher.
    """
    validate_job_description(job_description)

    filename, file_bytes = await read_uploaded_file(resume)
    resume_text = extract_text_or_raise(file_bytes, filename)

    try:
        result = match_resume_to_job(resume_text, job_description)

        return {
            "filename": filename,
            "job_description_length": len(job_description),
            "matching_result": result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Job matching failed.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to match the resume against the job description."
            ),
        ) from exc


# ---------------------------------------------------------
# LLM-based resume analysis
# ---------------------------------------------------------

@app.post("/api/resume/analyze-llm")
async def analyze_resume_llm(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    """
    Calculate the existing match score and generate an
    LLM-based explanation and recommendations.
    """
    validate_job_description(job_description)

    filename, file_bytes = await read_uploaded_file(resume)
    resume_text = extract_text_or_raise(file_bytes, filename)

    try:
        matching_result = match_resume_to_job(resume_text, job_description)

        llm_analysis = analyze_resume_with_llm(
            resume_text=resume_text,
            job_description=job_description,
            matching_result=matching_result,
        )

        return {
            "filename": filename,
            "matching_result": matching_result,
            "llm_analysis": llm_analysis,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.exception("LLM analysis failed.")
        raise HTTPException(
            status_code=503,
            detail="AI analysis is temporarily unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception("Resume LLM analysis failed.")
        raise HTTPException(
            status_code=500,
            detail="Unable to complete AI resume analysis.",
        ) from exc
