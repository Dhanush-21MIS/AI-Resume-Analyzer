import re

from app.keyword_extractor import (
    clean_text,
    extract_keywords,
    extract_sections,
    extract_section_text,
)


CONTACT_PATTERNS = [
    r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
    r"\+?\d[\d\s().-]{7,}\d",
    r"https?://\S+",
    r"www\.\S+",
]


def remove_contact_information(text: str) -> str:
    """
    Remove email addresses, phone numbers, and URLs from text.
    """

    cleaned = text

    for pattern in CONTACT_PATTERNS:
        cleaned = re.sub(
            pattern,
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    cleaned = re.sub(r"\s*[|•]\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def _limit_summary(
    summary: str,
    max_characters: int = 650,
) -> str:
    """
    Limit summary length without cutting off a word.
    """

    summary = clean_text(summary)

    if len(summary) <= max_characters:
        return summary

    shortened = summary[:max_characters]
    last_space = shortened.rfind(" ")

    if last_space > 0:
        shortened = shortened[:last_space]

    return shortened.rstrip(" ,;:") + "..."


def generate_summary(
    text: str,
    sections: dict,
) -> str:
    """
    Generate a concise summary from the resume text.

    Uses an existing summary section when available.
    Otherwise, uses meaningful lines from the resume.
    """

    summary_section = extract_section_text(
        text,
        "summary",
    )

    if summary_section:
        summary = remove_contact_information(
            clean_text(summary_section)
        )

        if summary:
            return _limit_summary(summary)

    excluded_headings = {
        "education",
        "experience",
        "work experience",
        "professional experience",
        "skills",
        "technical skills",
        "projects",
        "certification",
        "certifications",
        "achievements",
    }

    candidates = []

    for line in text.splitlines():
        cleaned_line = remove_contact_information(
            clean_text(line)
        )

        if not cleaned_line:
            continue

        if cleaned_line.lower().strip(" :|-") in excluded_headings:
            continue

        if len(cleaned_line.split()) < 5:
            continue

        candidates.append(cleaned_line)

        if len(" ".join(candidates)) >= 700:
            break

    if not candidates:
        return (
            "A summary could not be generated from the extracted "
            "resume text. Please check the document formatting."
        )

    return _limit_summary(" ".join(candidates))


def calculate_resume_score(
    text: str,
    sections: dict,
) -> dict:
    """
    Calculate a preliminary resume completeness score.

    This is a heuristic score, not an official ATS score.
    """

    criteria = {}

    # 1. Readable content: maximum 20 points.
    text_length = len(text.strip())

    if text_length >= 1500:
        criteria["readable_content"] = 20
    elif text_length >= 800:
        criteria["readable_content"] = 15
    elif text_length >= 300:
        criteria["readable_content"] = 10
    elif text_length >= 100:
        criteria["readable_content"] = 5
    else:
        criteria["readable_content"] = 0

    # 2. Important resume sections: maximum 60 points.
    section_weights = {
        "summary": 10,
        "education": 15,
        "experience": 15,
        "skills": 10,
        "projects": 5,
        "certifications": 5,
    }

    for section_name, weight in section_weights.items():
        criteria[section_name] = (
            weight
            if sections.get(section_name, False)
            else 0
        )

    # 3. Document structure: maximum 20 points.
    nonempty_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(nonempty_lines) >= 15:
        criteria["document_structure"] = 20
    elif len(nonempty_lines) >= 10:
        criteria["document_structure"] = 15
    elif len(nonempty_lines) >= 5:
        criteria["document_structure"] = 10
    elif nonempty_lines:
        criteria["document_structure"] = 5
    else:
        criteria["document_structure"] = 0

    score = min(100, sum(criteria.values()))

    missing_sections = [
        section_name
        for section_name in section_weights
        if not sections.get(section_name, False)
    ]

    return {
        "score": score,
        "maximum_score": 100,
        "criteria": criteria,
        "missing_sections": missing_sections,
        "note": (
            "This preliminary score estimates resume completeness "
            "using simple heuristics. It is not an official ATS "
            "score and does not predict hiring success."
        ),
    }


def analyze_resume(text: str) -> dict:
    """
    Run the complete resume analysis pipeline.
    """

    text = text.strip()

    if not text:
        raise ValueError(
            "Resume text is empty. Upload a readable PDF or DOCX file."
        )

    # Preserve original line breaks for section detection.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    sections = extract_sections(text)

    summary = generate_summary(
        text,
        sections,
    )

    resume_score = calculate_resume_score(
        text,
        sections,
    )

    keywords = extract_keywords(
        text,
        top_n=20,
    )

    normalized_text = clean_text(text)

    return {
        "summary": summary,
        "resume_score": resume_score,
        "sections": sections,
        "keywords": keywords,
        "text_length": len(normalized_text),
        "word_count": len(normalized_text.split()),
    }