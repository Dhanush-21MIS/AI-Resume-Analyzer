import re
from functools import lru_cache
from typing import Optional

from app.requirement_extractor import (
    extract_requirements,
    normalize_requirement_name,
    TECHNICAL_TERMS,
)

from app.semantic_matcher import calculate_semantic_similarity


# ============================================================
# CONFIGURATION
# ============================================================

SEMANTIC_THRESHOLD = 0.55
MAX_SEMANTIC_CREDIT = 0.70

NON_TECHNICAL_REQUIREMENTS = {
    "problem solving",
    "communication",
    "teamwork",
    "leadership",
    "analytical thinking",
    "critical thinking",
    "time management",
    "collaboration",
    "attention to detail",
    "adaptability",
    "project management",
    "written communication",
    "verbal communication",
}


# ============================================================
# SKILL EQUIVALENTS
# ============================================================

SKILL_EQUIVALENTS = {
    "object-oriented programming": [
        "object-oriented programming",
        "object oriented programming",
        "oop",
    ],
    "powerbi": [
        "powerbi",
        "power bi",
        "microsoft power bi",
    ],
    # Do not use bare "react": it can incorrectly match React Native.
    "react.js": [
        "react.js",
        "react js",
    ],
    # Do not use bare "express": ordinary English uses this word.
    "express.js": [
        "express.js",
        "express js",
    ],
    "postgresql": [
        "postgres",
        "postgresql",
    ],
    # REST API experience is relevant evidence for API development.
    "api development": [
        "api development",
        "developed apis",
        "developing apis",
        "building apis",
        "built apis",
        "rest api",
        "rest apis",
        "restful api",
        "restful apis",
    ],
    # Verb forms are checked together with context in
    # is_valid_requirement_evidence().
    "deployment": [
        "deployment",
        "deploy",
        "deployed",
        "deploying",
        "hosted",
        "published",
    ],
    # MongoDB is a NoSQL database; this only proves database exposure,
    # not broad experience with every NoSQL technology.
    "nosql": [
        "nosql",
        "non-relational database",
        "non relational database",
        "mongodb",
    ],
    "scikit-learn": [
        "scikit-learn",
        "scikit learn",
        "sklearn",
    ],
    "aws": [
        "aws",
        "amazon web services",
    ],
    "azure": [
        "azure",
        "microsoft azure",
    ],
    "gcp": [
        "gcp",
        "google cloud",
        "google cloud platform",
    ],
    "vscode": [
        "vscode",
        "visual studio code",
    ],
    "ci/cd": [
        "ci/cd",
        "continuous integration and continuous delivery",
        "continuous integration and continuous deployment",
    ],
    "rest apis": [
        "rest api",
        "rest apis",
        "restful api",
        "restful apis",
    ],
    "llm": [
        "llm",
        "llms",
        "large language model",
        "large language models",
    ],
    "rag": [
        "rag",
        "retrieval augmented generation",
        "retrieval-augmented generation",
    ],
}


# ============================================================
# OVERLAPPING REQUIREMENT GROUPS
# ============================================================

# Requirements in a group contribute only once to the score.
# They remain separate in the detailed API response.

OVERLAPPING_REQUIREMENT_GROUPS = [
    {
        "api development",
        "rest api",
        "rest apis",
        "restful api",
        "restful apis",
    },
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize text for consistent comparisons."""

    if not text:
        return ""

    text = str(text).lower().strip()

    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2019", "'")

    text = re.sub(r"\s+", " ", text)

    return text


def is_negated_skill_mention(
    chunk: str,
    matched_term: str,
) -> bool:
    """Detect common statements explicitly denying experience with a skill."""

    text = normalize_text(chunk)
    term = normalize_text(matched_term)

    if not text or not term:
        return False

    match = make_phrase_pattern(term).search(text)
    if not match:
        return False

    # Inspect a short window before the skill mention. This catches examples
    # such as "no Docker project experience is described" and
    # "no experience with Docker".
    preceding = text[max(0, match.start() - 80):match.start()].strip()

    negation_patterns = (
        r"\bno\s*$",
        r"\bno\s+(?:hands[- ]on\s+)?experience\s+with\s*$",
        r"\bwithout\s+(?:any\s+)?(?:prior\s+)?(?:hands[- ]on\s+)?experience\s+with\s*$",
        r"\bnever\s+(?:used|worked\s+with|learned|handled)\s*$",
        r"\bnot\s+(?:used|experienced\s+with|familiar\s+with|proficient\s+in)\s*$",
        r"\black(?:s|ed)?\s+of\s*$",
        r"\blacking\s+(?:any\s+)?experience\s+with\s*$",
    )

    if any(re.search(pattern, preceding) for pattern in negation_patterns):
        return True

    # Also catch constructions where the denial follows the skill.
    following = text[match.end():match.end() + 80]
    after_skill_patterns = (
        r"^\s+(?:is|was|has been)?\s*(?:not\s+)?(?:mentioned|described|required|used)\b",
    )
    if re.search(r"\bno\b", preceding) and any(
        re.search(pattern, following) for pattern in after_skill_patterns
    ):
        return True

    return False


def canonical_requirement(requirement: str) -> str:
    """Return the canonical name of a requirement."""

    normalized = normalize_text(requirement)

    if not normalized:
        return ""

    try:
        canonical = normalize_requirement_name(normalized)
    except (TypeError, ValueError, AttributeError):
        canonical = normalized

    return normalize_text(canonical)


def make_phrase_pattern(phrase: str) -> re.Pattern:
    """
    Create a phrase pattern with word boundaries.

    Prevents Java from matching JavaScript.
    Supports phrases containing spaces and hyphens.
    """

    phrase = normalize_text(phrase)

    if not phrase:
        return re.compile(r"(?!)")

    parts = re.split(r"[\s-]+", phrase)

    escaped_parts = [
        re.escape(part)
        for part in parts
        if part
    ]

    if not escaped_parts:
        return re.compile(r"(?!)")

    pattern = r"[\s-]+".join(escaped_parts)

    return re.compile(
        rf"(?<!\w){pattern}(?!\w)",
        re.IGNORECASE,
    )


# ============================================================
# RESUME CHUNKING
# ============================================================

def split_resume_into_chunks(resume_text: str) -> list[str]:
    """
    Split resume text into readable lines and sentences.
    """

    if not resume_text:
        return []

    text = str(resume_text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\u2022", "\n")
    text = text.replace("\u25aa", "\n")

    chunks = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            line,
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence:
                chunks.append(sentence)

    return chunks


# ============================================================
# REQUIREMENT ALIASES
# ============================================================

def get_requirement_aliases(
    requirement: str,
) -> list[str]:
    """Return alternative names for the same requirement."""

    canonical = canonical_requirement(requirement)

    candidates = [
        requirement,
        canonical,
    ]

    for key, aliases in SKILL_EQUIVALENTS.items():
        if canonical_requirement(key) == canonical:
            candidates.extend(aliases)

    unique_aliases = []
    seen = set()

    for alias in candidates:
        alias = normalize_text(alias)

        if not alias or alias in seen:
            continue

        seen.add(alias)
        unique_aliases.append(alias)

    return unique_aliases


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================

def make_evidence_snippet(
    text: str,
    start: int,
    end: int,
    context: int = 85,
) -> str:
    """
    Return a readable excerpt surrounding matched evidence.
    """

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= 280:
        return text

    left = max(0, start - context)
    right = min(len(text), end + context)

    snippet = text[left:right].strip()

    if left > 0:
        snippet = "..." + snippet

    if right < len(text):
        snippet += "..."

    return snippet


def is_valid_requirement_evidence(
    requirement: str,
    chunk: str,
    matched_term: str,
) -> bool:
    """Reject known false positives while keeping matching evidence-based."""

    canonical = canonical_requirement(requirement)
    normalized_chunk = normalize_text(chunk)
    normalized_term = normalize_text(matched_term)

    if is_negated_skill_mention(chunk, matched_term):
        return False

    # React Native is a separate framework; it alone does not prove React.js.
    if canonical == "react.js":
        if "react native" in normalized_chunk and not re.search(
            r"(?<!native\s)(?<!react\s)react\s*\.\s*js|\breact\s+js\b",
            normalized_chunk,
            re.IGNORECASE,
        ):
            return False
        if normalized_term in {"react", "react native"}:
            return False

    # "express" as an ordinary verb must not count as Express.js.
    if canonical == "express.js" and normalized_term == "express":
        return False

    # A GitHub URL/profile in the contact header is not proof of GitHub usage.
    if canonical == "github":
        contact_only = re.search(
            r"(?:https?://)?(?:www\.)?github\.com/|\blinked\s*in\b|\bemail\b|\bphone\b|\+?\d[\d ()-]{7,}",
            normalized_chunk,
            re.IGNORECASE,
        )
        usage_evidence = re.search(
            r"\b(?:used|using|managed|maintained|pushed|committed|created|contributed|collaborated|version control|repository|repositories|pull request|github actions|workflow)\b",
            normalized_chunk,
            re.IGNORECASE,
        )
        if contact_only and not usage_evidence:
            return False

    # API development requires an action or implementation context. A bare
    # technology-list entry such as "REST APIs" is related evidence, not proof
    # that the candidate implemented APIs.
    if canonical == "api development":
        implementation_action = re.search(
            r"\b(?:built|build|building|developed|develop|developing|implemented|implement|implementing|created|create|creating|designed|design|designing|integrated|integrate|integrating|maintained|maintain|maintaining)\b",
            normalized_chunk,
            re.IGNORECASE,
        )
        api_context = re.search(
            r"\b(?:api|apis|restful|endpoint|endpoints)\b",
            normalized_chunk,
            re.IGNORECASE,
        )
        if not (implementation_action and api_context):
            return False

    # MongoDB is evidence of NoSQL database exposure, but by itself should not
    # receive full credit for the broader NoSQL requirement. The caller assigns
    # partial credit to this specific equivalence.

    # Deployment needs an object/platform/context, not just a generic word form.
    if canonical == "deployment":
        deployment_context = re.search(
            r"\b(?:app|application|website|web app|api|service|project|model|backend|frontend|container|production|vercel|render|aws|azure|gcp|cloud|docker|server)\b",
            normalized_chunk,
            re.IGNORECASE,
        )
        deployment_verb = re.search(
            r"\b(?:deploy|deployed|deploying|deployment|hosted|published|released)\b",
            normalized_chunk,
            re.IGNORECASE,
        )
        if not (deployment_context and deployment_verb):
            return False

    return True


def find_exact_evidence(
    resume_text: str,
    requirement: str,
) -> Optional[dict]:
    """
    Find exact or explicitly configured equivalent evidence.

    Returns a dictionary containing evidence, matched_term and match_type.
    Known ambiguous terms are validated against their sentence context.
    """

    if not resume_text or not requirement:
        return None

    canonical = canonical_requirement(requirement)
    aliases = get_requirement_aliases(requirement)
    chunks = split_resume_into_chunks(resume_text)
    aliases = sorted(aliases, key=len, reverse=True)

    for chunk in chunks:
        for alias in aliases:
            match = make_phrase_pattern(alias).search(chunk)
            if not match:
                continue

            matched_term = match.group(0)
            if not is_valid_requirement_evidence(
                requirement, chunk, matched_term
            ):
                continue

            normalized_alias = normalize_text(alias)
            match_type = "exact" if normalized_alias == canonical else "equivalent"

            return {
                "evidence": make_evidence_snippet(
                    chunk, match.start(), match.end()
                ),
                "matched_term": matched_term,
                "match_type": match_type,
            }

    return None


# ============================================================
# TECHNICAL REQUIREMENT DETECTION
# ============================================================

@lru_cache(maxsize=1)
def get_normalized_technical_terms() -> set[str]:
    """Return normalized technical terms from the extractor."""

    terms = {
        canonical_requirement(term)
        for term in TECHNICAL_TERMS
    }

    terms.update(
        canonical_requirement(term)
        for term in SKILL_EQUIVALENTS
    )

    return terms


def is_technical_requirement(
    requirement: str,
) -> bool:
    """Determine whether a requirement is technical."""

    canonical = canonical_requirement(requirement)

    return canonical in get_normalized_technical_terms()


def is_non_technical_requirement(
    requirement: str,
) -> bool:
    """Check whether semantic matching is permitted."""

    canonical = canonical_requirement(requirement)

    return canonical in {
        canonical_requirement(term)
        for term in NON_TECHNICAL_REQUIREMENTS
    }


# ============================================================
# MATCH RECORD
# ============================================================

def build_match_record(
    requirement: str,
    importance: float,
    status: str,
    match_type: str,
    score: float,
    evidence: Optional[str] = None,
    semantic_similarity: Optional[float] = None,
) -> dict:
    """Create a consistent requirement result."""

    safe_score = max(
        0.0,
        min(float(score), 1.0),
    )

    return {
        "requirement": requirement,
        "importance": round(
            max(0.0, float(importance)),
            3,
        ),
        "status": status,
        "match_type": match_type,
        "score": round(safe_score, 4),
        "evidence": evidence,
        "semantic_similarity": (
            round(float(semantic_similarity), 4)
            if semantic_similarity is not None
            else None
        ),
    }


# ============================================================
# OVERLAPPING REQUIREMENTS
# ============================================================

def get_overlap_group(
    requirement: str,
) -> Optional[str]:
    """Return the scoring group for an overlapping requirement."""

    canonical = canonical_requirement(requirement)

    for index, group in enumerate(
        OVERLAPPING_REQUIREMENT_GROUPS
    ):
        normalized_group = {
            canonical_requirement(item)
            for item in group
        }

        if canonical in normalized_group:
            return f"group_{index}"

    return None


# ============================================================
# SEMANTIC MATCHING
# ============================================================

@lru_cache(maxsize=512)
def cached_semantic_similarity(
    resume_text: str,
    requirement: str,
) -> tuple[float, Optional[str]]:
    """Cache semantic results to avoid repeated model inference."""

    result = calculate_semantic_similarity(
        resume_text,
        requirement,
    )

    similarity = float(
        result.get("similarity", 0.0)
    )

    evidence = result.get("evidence")

    return similarity, evidence


def find_semantic_evidence(
    resume_text: str,
    requirement: str,
) -> Optional[dict]:
    """
    Perform semantic matching for selected general requirements.

    Semantic similarity is not proof of a technical skill.
    """

    if not is_non_technical_requirement(requirement):
        return None

    try:
        similarity, evidence = cached_semantic_similarity(
            resume_text,
            canonical_requirement(requirement),
        )
    except Exception:
        return None

    if similarity < SEMANTIC_THRESHOLD:
        return None

    score = min(
        max(similarity, 0.0),
        MAX_SEMANTIC_CREDIT,
    )

    return {
        "score": score,
        "similarity": similarity,
        "evidence": evidence,
        "match_type": "semantic_relatedness",
    }


# ============================================================
# MATCH ONE REQUIREMENT
# ============================================================

def match_single_requirement(
    resume_text: str,
    requirement_data: dict,
) -> dict:
    """Match one extracted requirement against the resume."""

    requirement = str(
        requirement_data.get("requirement", "")
    ).strip()

    try:
        importance = float(
            requirement_data.get("importance", 1.0)
        )
    except (TypeError, ValueError):
        importance = 1.0

    importance = max(0.0, importance)

    # --------------------------------------------------------
    # 1. Exact or equivalent matching
    # --------------------------------------------------------

    evidence_result = find_exact_evidence(
        resume_text,
        requirement,
    )

    if evidence_result:
        # MongoDB is a valid example of a NoSQL database, but a mention of
        # MongoDB alone is narrower than demonstrating broad NoSQL experience.
        if (
            canonical_requirement(requirement) == "nosql"
            and normalize_text(evidence_result.get("matched_term", "")) == "mongodb"
        ):
            return build_match_record(
                requirement=requirement,
                importance=importance,
                status="partially_matched",
                match_type="related_technology",
                score=0.65,
                evidence=evidence_result["evidence"],
            )

        return build_match_record(
            requirement=requirement,
            importance=importance,
            status="matched",
            match_type="exact_or_equivalent",
            score=1.0,
            evidence=evidence_result["evidence"],
        )

    # --------------------------------------------------------
    # 2. Semantic matching for general requirements
    # --------------------------------------------------------

    semantic_result = find_semantic_evidence(
        resume_text,
        requirement,
    )

    if semantic_result and semantic_result["evidence"]:
        return build_match_record(
            requirement=requirement,
            importance=importance,
            status="partially_matched",
            match_type=semantic_result["match_type"],
            score=semantic_result["score"],
            evidence=str(semantic_result["evidence"]),
            semantic_similarity=semantic_result["similarity"],
        )

    # --------------------------------------------------------
    # 3. No supporting evidence
    # --------------------------------------------------------

    return build_match_record(
        requirement=requirement,
        importance=importance,
        status="missing",
        match_type="no_evidence",
        score=0.0,
        evidence=None,
    )


# ============================================================
# WEIGHTED MATCH SCORE
# ============================================================

def calculate_weighted_score(
    records: list[dict],
) -> float:
    """
    Calculate the weighted matching percentage.

    Ordinary requirements are scored independently.
    Configured overlapping requirements contribute only once.

    Formula:
        sum(weight * score) / sum(weight) * 100
    """

    if not records:
        return 0.0

    ordinary_records = []
    grouped_records = {}

    for record in records:
        group = get_overlap_group(
            record["requirement"]
        )

        if group is None:
            ordinary_records.append(record)
        else:
            grouped_records.setdefault(
                group,
                [],
            ).append(record)

    numerator = 0.0
    denominator = 0.0

    # Score ordinary requirements.
    for record in ordinary_records:
        weight = max(
            float(record.get("importance", 1.0)),
            0.0,
        )

        score = max(
            0.0,
            min(float(record.get("score", 0.0)), 1.0),
        )

        numerator += weight * score
        denominator += weight

    # Score each overlapping group once.
    for group_records in grouped_records.values():
        group_weight = max(
            max(
                float(record.get("importance", 1.0)),
                0.0,
            )
            for record in group_records
        )

        group_score = max(
            max(
                0.0,
                min(float(record.get("score", 0.0)), 1.0),
            )
            for record in group_records
        )

        numerator += group_weight * group_score
        denominator += group_weight

    if denominator <= 0:
        return 0.0

    return round(
        (numerator / denominator) * 100,
        2,
    )


# ============================================================
# COVERAGE
# ============================================================

def calculate_coverage(
    records: list[dict],
) -> float:
    """
    Calculate the percentage of requirements with supporting
    evidence.

    Both full and partial matches count as covered.
    """

    if not records:
        return 0.0

    supported_count = sum(
        1
        for record in records
        if record.get("status") in {
            "matched",
            "partially_matched",
        }
    )

    return round(
        (supported_count / len(records)) * 100,
        2,
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

def build_recommendations(
    missing: list[dict],
    partially_matched: list[dict],
) -> list[dict]:
    """Generate evidence-aware recommendations."""

    recommendations = []

    if missing:
        recommendations.append({
            "type": "missing_requirements",
            "message": (
                "These job requirements were not found in your "
                "resume. If you have relevant experience, add "
                "specific supporting details. Otherwise, consider "
                "developing the skills where appropriate."
            ),
            "requirements": [
                record["requirement"]
                for record in missing
            ],
        })

    if partially_matched:
        recommendations.append({
            "type": "strengthen_evidence",
            "message": (
                "Related evidence was found for these general "
                "requirements, but the match is not conclusive. "
                "Add specific project or work examples when "
                "they accurately reflect your experience."
            ),
            "requirements": [
                record["requirement"]
                for record in partially_matched
            ],
        })

    if not recommendations:
        recommendations.append({
            "type": "good_alignment",
            "message": (
                "All extracted requirements have supporting "
                "evidence. Review the evidence for accuracy and "
                "make sure your achievements are clearly described."
            ),
            "requirements": [],
        })

    return recommendations


# ============================================================
# MAIN MATCHING FUNCTION
# ============================================================

def match_resume_to_job(
    resume_text: str,
    job_description: str,
) -> dict:
    """
    Compare resume text with a job description.

    Matching sequence:
        1. Extract job requirements.
        2. Search for exact or equivalent evidence.
        3. Apply semantic matching to selected general skills.
        4. Mark unsupported requirements as missing.
        5. Calculate the weighted score and coverage.
        6. Generate recommendations.
    """

    if (
        not isinstance(resume_text, str)
        or not resume_text.strip()
    ):
        raise ValueError("Resume text cannot be empty.")

    if (
        not isinstance(job_description, str)
        or not job_description.strip()
    ):
        raise ValueError("Job description cannot be empty.")

    # --------------------------------------------------------
    # 1. Extract requirements
    # --------------------------------------------------------

    requirements = extract_requirements(
        job_description,
        top_n=30,
    )

    matched = []
    partially_matched = []
    missing = []
    all_records = []

    # --------------------------------------------------------
    # 2. Match each requirement
    # --------------------------------------------------------

    for item in requirements:
        if not isinstance(item, dict):
            continue

        requirement = str(
            item.get("requirement", "")
        ).strip()

        if not requirement:
            continue

        record = match_single_requirement(
            resume_text=resume_text,
            requirement_data=item,
        )

        all_records.append(record)

        if record["status"] == "matched":
            matched.append(record)

        elif record["status"] == "partially_matched":
            partially_matched.append(record)

        else:
            missing.append(record)

    # --------------------------------------------------------
    # 3. Calculate metrics
    # --------------------------------------------------------

    total_requirements = len(all_records)

    matched_count = len(matched)
    partially_matched_count = len(partially_matched)
    missing_count = len(missing)

    match_percentage = calculate_weighted_score(
        all_records
    )

    coverage_percentage = calculate_coverage(
        all_records
    )

    # --------------------------------------------------------
    # 4. Generate recommendations
    # --------------------------------------------------------

    if total_requirements == 0:
        recommendations = [
            {
                "type": "review_job_description",
                "message": (
                    "No recognizable requirements were extracted "
                    "from the job description. Review the extraction "
                    "rules or verify the supplied job description."
                ),
                "requirements": [],
            }
        ]
    else:
        recommendations = build_recommendations(
            missing,
            partially_matched,
        )

    # --------------------------------------------------------
    # 5. Return the API response
    # --------------------------------------------------------

    return {
        "match_percentage": match_percentage,
        "coverage_percentage": coverage_percentage,
        "requirements_analyzed": total_requirements,
        "matched_count": matched_count,
        "partially_matched_count": partially_matched_count,
        "missing_count": missing_count,
        "matched_requirements": matched,
        "partially_matched_requirements": partially_matched,
        "missing_requirements": missing,
        "recommendations": recommendations,
        "note": (
            "The match percentage is a heuristic score based on "
            "extracted requirements, importance weights, and resume "
            "evidence. Configured overlapping requirements are "
            "grouped during scoring. Coverage represents the "
            "percentage of extracted requirements with supporting "
            "evidence. Neither metric is an official ATS score or "
            "a hiring probability."
        ),
    }