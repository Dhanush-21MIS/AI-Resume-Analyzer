import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ============================================================
# 1. STOP WORDS
# ============================================================

CUSTOM_STOP_WORDS = {
    "using",
    "used",
    "use",
    "work",
    "worked",
    "working",
    "developed",
    "development",
    "project",
    "projects",
    "including",
    "various",
    "good",
    "strong",
    "ability",
    "responsible",
    "responsibilities",
    "knowledge",
    "experience",
    "experienced",
    "expertise",
    "skills",
    "skill",
    "role",
    "team",
    "teams",
    "provided",
    "provide",
    "across",
    "within",
    "etc",
    "secondary",
    "education",
    "percentage",
    "vellore",
    "trinity",
    "matriculation",
}

STOP_WORDS = set(ENGLISH_STOP_WORDS).union(
    CUSTOM_STOP_WORDS
)

IMPORTANT_TECHNICAL_TERMS = {
    "c++",
    "c#",
    "r",
}


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving line breaks.
    """

    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# 3. TOKENIZATION
# ============================================================

TOKEN_PATTERN = re.compile(
    r"(?u)"
    r"[a-z0-9]+"
    r"(?:[.+#/-][a-z0-9+#.-]+)*"
)


def _tokenize(text: str) -> list[str]:
    """
    Extract useful terms while preserving common technical names.
    """

    raw_tokens = TOKEN_PATTERN.findall(text.lower())

    tokens = []

    for token in raw_tokens:
        token = token.strip(".,:;!?()[]{}\"'`")

        if not token or token.isdigit():
            continue

        if token in IMPORTANT_TECHNICAL_TERMS:
            tokens.append(token)
            continue

        if token in STOP_WORDS:
            continue

        if len(token) < 2:
            continue

        tokens.append(token)

    return tokens


# ============================================================
# 4. KEYWORD EXTRACTION
# ============================================================

def extract_keywords(
    text: str,
    top_n: int = 20,
) -> list[dict]:
    """
    Extract frequent keywords and adjacent phrases.

    Scores are heuristic ranking values, not probabilities
    or measures of job suitability.
    """

    if not text or not text.strip() or top_n <= 0:
        return []

    tokens = _tokenize(text)

    if not tokens:
        return []

    frequencies = Counter(tokens)

    # Count adjacent two-word phrases.
    for first, second in zip(tokens, tokens[1:]):
        if first.isdigit() or second.isdigit():
            continue

        phrase = f"{first} {second}"
        frequencies[phrase] += 1

    results = []

    for keyword, frequency in frequencies.items():
        word_count = len(keyword.split())

        score = (
            frequency * 1.2
            if word_count == 2
            else float(frequency)
        )

        results.append(
            {
                "keyword": keyword,
                "frequency": frequency,
                "score": round(score, 3),
            }
        )

    results.sort(
        key=lambda item: (
            item["score"],
            item["frequency"],
            item["keyword"],
        ),
        reverse=True,
    )

    selected = results[:top_n]

    if selected:
        maximum_score = max(
            item["score"]
            for item in selected
        )

        for item in selected:
            item["relative_score"] = round(
                item["score"] / maximum_score,
                3,
            )

    return selected


# ============================================================
# 5. SECTION PATTERNS
# ============================================================

SECTION_PATTERNS = {
    "summary": [
        r"professional\s+summary",
        r"executive\s+summary",
        r"career\s+summary",
        r"profile\s+summary",
        r"about\s+me",
        r"career\s+objective",
        r"objective",
        r"profile",
    ],

    "education": [
        r"education",
        r"academic\s+background",
        r"educational\s+background",
        r"educational\s+qualifications",
        r"academic\s+qualifications",
        r"qualifications",
    ],

    "experience": [
        r"professional\s+experience",
        r"work\s+experience",
        r"relevant\s+experience",
        r"employment\s+history",
        r"work\s+history",
        r"professional\s+background",
        r"internship\s+experience",
        r"internships",
        r"experience",
        r"employment",
    ],

    "skills": [
        r"technical\s+skills",
        r"core\s+skills",
        r"key\s+skills",
        r"programming\s+skills",
        r"technical\s+expertise",
        r"technologies",
        r"tech\s+stack",
        r"skills",
    ],

    "projects": [
        r"academic\s+projects",
        r"personal\s+projects",
        r"key\s+projects",
        r"project\s+experience",
        r"projects",
    ],

    "certifications": [
        r"professional\s+certifications",
        r"licenses\s+and\s+certifications",
        r"certification",
        r"certifications",
        r"certificates",
    ],

    "achievements": [
        r"achievements",
        r"awards",
        r"honors",
        r"accomplishments",
    ],
}


COMPILED_SECTION_PATTERNS = {
    section_name: [
        re.compile(pattern, re.IGNORECASE)
        for pattern in patterns
    ]
    for section_name, patterns in SECTION_PATTERNS.items()
}


# ============================================================
# 6. SECTION HEADING DETECTION
# ============================================================

def _normalize_heading(line: str) -> str:
    """
    Normalize a line before checking for a section heading.
    """

    line = line.strip()

    line = re.sub(
        r"^[\s\-•*|:#\d.)]+",
        "",
        line,
    )

    line = re.sub(
        r"[\s:#|]+$",
        "",
        line,
    )

    return re.sub(
        r"\s+",
        " ",
        line,
    ).lower()


def _match_section_heading(line: str) -> str | None:
    """
    Identify a line that contains only a recognized heading.
    """

    normalized_line = _normalize_heading(line)

    for section_name, patterns in COMPILED_SECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern.fullmatch(normalized_line):
                return section_name

    return None


def _match_inline_heading(
    line: str,
) -> tuple[str, int] | None:
    """
    Identify a recognized heading at the beginning of a line.

    Example:
        Executive Summary Software Engineer with experience...
    """

    normalized_line = _normalize_heading(line)

    for section_name, patterns in COMPILED_SECTION_PATTERNS.items():
        for pattern in patterns:
            match = pattern.match(normalized_line)

            if not match:
                continue

            if match.end() == len(normalized_line):
                return section_name, match.end()

            next_character = normalized_line[match.end()]

            if next_character.isspace() or next_character in ":|-":
                return section_name, match.end()

    return None


def extract_sections(text: str) -> dict:
    """
    Detect resume sections using recognized headings.
    """

    sections = {
        section_name: False
        for section_name in SECTION_PATTERNS
    }

    for line in clean_text(text).splitlines():
        matched_section = _match_section_heading(line)

        if matched_section:
            sections[matched_section] = True
            continue

        inline_match = _match_inline_heading(line)

        if inline_match:
            section_name, _ = inline_match
            sections[section_name] = True

    return sections


# ============================================================
# 7. SECTION TEXT EXTRACTION
# ============================================================

def extract_section_text(
    text: str,
    target_section: str,
) -> str:
    """
    Extract content from a specific resume section.

    Stops when another recognized section heading begins.
    """

    if target_section not in SECTION_PATTERNS:
        return ""

    lines = clean_text(text).splitlines()

    collected_lines = []
    collecting = False

    for line in lines:
        matched_section = _match_section_heading(line)

        if matched_section:
            if collecting:
                break

            if matched_section == target_section:
                collecting = True

            continue

        inline_match = _match_inline_heading(line)

        if inline_match:
            section_name, heading_end = inline_match

            if collecting:
                break

            if section_name == target_section:
                normalized_line = _normalize_heading(line)

                remaining_text = normalized_line[
                    heading_end:
                ].strip(" \t:|-")

                if remaining_text:
                    collected_lines.append(remaining_text)

                collecting = True

            continue

        if collecting:
            collected_lines.append(line.strip())

    return "\n".join(
        line
        for line in collected_lines
        if line
    ).strip()