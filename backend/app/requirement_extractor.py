import re
from collections import defaultdict


# =========================================================
# TECHNICAL REQUIREMENTS
# =========================================================

TECHNICAL_TERMS = [
    # Programming languages
    "python",
    "java",
    "javascript",
    "typescript",
    "c",
    "c++",
    "c#",
    "php",
    "ruby",
    "go",
    "golang",
    "rust",
    "kotlin",
    "swift",
    "r",
    "sql",

    # Frontend development
    "html",
    "css",
    "react",
    "react.js",
    "angular",
    "vue",
    "vue.js",
    "next.js",
    "nuxt.js",
    "redux",
    "tailwind css",
    "bootstrap",
    "sass",
    "responsive design",

    # Backend development
    "node.js",
    "express",
    "express.js",
    "django",
    "flask",
    "fastapi",
    "spring boot",
    "spring framework",
    "asp.net",
    ".net",
    "laravel",
    "nestjs",
    "rest api",
    "rest apis",
    "restful api",
    "restful apis",
    "api development",
    "graphql",
    "websocket",
    "microservices",

    # Databases
    "mysql",
    "postgresql",
    "postgres",
    "mongodb",
    "nosql",
    "redis",
    "sqlite",
    "oracle",
    "sql server",
    "mariadb",
    "dynamodb",
    "cassandra",
    "elasticsearch",
    "firebase",

    # AI and machine learning
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "generative ai",
    "natural language processing",
    "computer vision",
    "neural networks",
    "transformer models",
    "large language models",
    "llm",
    "llms",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "keras",
    "opencv",
    "pandas",
    "numpy",
    "hugging face",
    "langchain",
    "retrieval augmented generation",
    "rag",
    "prompt engineering",
    "data science",
    "data analysis",
    "data analytics",
    "statistical analysis",

    # Cloud platforms
    "aws",
    "amazon web services",
    "azure",
    "microsoft azure",
    "google cloud",
    "gcp",
    "cloud computing",
    "cloud architecture",
    "serverless",

    # DevOps and infrastructure
    "docker",
    "kubernetes",
    "jenkins",
    "terraform",
    "ansible",
    "github actions",
    "gitlab ci",
    "ci/cd",
    "continuous integration",
    "continuous delivery",
    "continuous deployment",
    "linux",
    "nginx",
    "apache",
    "devops",
    "deployment",
    "render",
    "vercel",

    # Version control and development tools
    "git",
    "github",
    "gitlab",
    "bitbucket",
    "postman",
    "jira",
    "power bi",
    "powerbi",
    "vscode",
    "visual studio code",

    # Software engineering
    "object-oriented programming",
    "oop",
    "data structures",
    "algorithms",
    "system design",
    "design patterns",
    "software development",
    "software engineering",
    "debugging",
    "problem solving",
    "unit testing",
    "integration testing",
    "automated testing",
    "test automation",
    "pytest",
    "junit",
    "selenium",
    "playwright",
    "cypress",
    "agile",
    "scrum",
    "software testing",
    "code review",
    "clean code",
    "security",
    "cybersecurity",
    "jwt",
    "oauth",
    "authentication",
    "authorization",
]


# =========================================================
# GENERIC TERMS
# =========================================================

GENERIC_TERMS = {
    "experience",
    "experienced",
    "knowledge",
    "understanding",
    "ability",
    "skills",
    "skill",
    "strong",
    "good",
    "working",
    "familiar",
    "familiarity",
    "proficiency",
    "proficient",
    "development",
    "application",
    "applications",
    "software",
    "system",
    "systems",
    "technology",
    "technologies",
    "using",
    "including",
    "related",
    "required",
    "preferred",
    "candidate",
    "responsibilities",
    "requirements",
}


# =========================================================
# ALIASES
# =========================================================

ALIASES = {
    "react": "react.js",
    "express": "express.js",
    "vue": "vue.js",
    "postgres": "postgresql",
    "golang": "go",
    "power bi": "powerbi",
    "visual studio code": "vscode",
    "amazon web services": "aws",
    "microsoft azure": "azure",
    "google cloud platform": "gcp",
    "object oriented programming": "object-oriented programming",
    "oop": "object-oriented programming",
    "rest api": "rest apis",
    "restful api": "rest apis",
    "restful apis": "rest apis",
    "rest api development": "api development",
    "continuous integration and continuous delivery": "ci/cd",
    "continuous integration and continuous deployment": "ci/cd",
    "continuous delivery": "ci/cd",
    "continuous deployment": "ci/cd",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "opencv-python": "opencv",
    "natural language processing (nlp)": "natural language processing",
    "large language model": "llm",
    "large language models": "llm",
    "retrieval-augmented generation": "rag",
    "retrieval augmented generation": "rag",
    "google cloud": "gcp",
    "continuous integration": "ci/cd",
}


# =========================================================
# NORMALIZATION
# =========================================================

def clean_job_description(job_description: str) -> str:
    """Clean whitespace and common formatting in a job description."""

    if not job_description:
        return ""

    text = str(job_description)

    text = text.replace("\u2022", " ")
    text = text.replace("\u25aa", " ")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2019", "'")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_requirement(requirement: str) -> str:
    """Normalize a requirement without changing its meaning."""

    if not requirement:
        return ""

    requirement = str(requirement).lower().strip()

    requirement = requirement.replace("\u2013", "-")
    requirement = requirement.replace("\u2014", "-")

    requirement = re.sub(r"\s+", " ", requirement)

    return requirement


def normalize_requirement_name(requirement: str) -> str:
    """Convert known aliases to a canonical requirement name."""

    normalized = normalize_requirement(requirement)

    return ALIASES.get(normalized, normalized)


def make_term_pattern(term: str) -> re.Pattern:
    """
    Build a phrase pattern with boundaries.

    Prevents a term such as 'java' from matching
    unrelated words such as 'javascript'.
    """

    term = normalize_requirement(term)

    if not term:
        return re.compile(r"(?!)")

    # Allow whitespace and hyphens between phrase components.
    parts = re.split(r"[\s-]+", term)

    escaped = [
        re.escape(part)
        for part in parts
        if part
    ]

    if not escaped:
        return re.compile(r"(?!)")

    pattern = r"[\s-]+".join(escaped)

    return re.compile(
        rf"(?<!\w){pattern}(?!\w)",
        re.IGNORECASE,
    )


def split_into_segments(text: str) -> list[str]:
    """Split job descriptions into sentences and bullet-like segments."""

    if not text:
        return []

    text = str(text)

    segments = re.split(
        r"[\n\r]+|(?<=[.!?])\s+|[;•]+",
        text,
    )

    return [
        segment.strip()
        for segment in segments
        if segment.strip()
    ]


# =========================================================
# IMPORTANCE DETECTION
# =========================================================

HIGH_IMPORTANCE_PATTERNS = [
    r"\bmust[- ]have\b",
    r"\brequired\b",
    r"\brequirements?\b",
    r"\bmandatory\b",
    r"\bessential\b",
    r"\bminimum qualifications?\b",
    r"\bstrong experience\b",
    r"\bproven experience\b",
]

MEDIUM_IMPORTANCE_PATTERNS = [
    r"\bpreferred\b",
    r"\bpreferably\b",
    r"\bdesired\b",
    r"\bnice to have\b",
    r"\bbonus\b",
    r"\bfamiliarity with\b",
    r"\bfamiliar with\b",
]


def calculate_importance(segment: str) -> float:
    """
    Estimate importance from explicit job-description wording.

    High importance: 1.5
    Normal importance: 1.0
    Preferred skills: 0.7
    """

    text = normalize_requirement(segment)

    for pattern in MEDIUM_IMPORTANCE_PATTERNS:
        if re.search(pattern, text):
            return 0.7

    for pattern in HIGH_IMPORTANCE_PATTERNS:
        if re.search(pattern, text):
            return 1.5

    return 1.0


# =========================================================
# GENERIC PHRASE FILTERING
# =========================================================

def is_generic_phrase(phrase: str) -> bool:
    """Identify phrases that are too generic to be skills."""

    normalized = normalize_requirement(phrase)

    if not normalized:
        return True

    if normalized in GENERIC_TERMS:
        return True

    words = set(re.findall(r"[a-z]+", normalized))

    if words and words.issubset(GENERIC_TERMS):
        return True

    return False


# =========================================================
# JOB-TITLE FILTERING
# =========================================================

def remove_title_only_devops_lines(job_description: str) -> str:
    """
    Remove DevOps Engineer when it appears only as the role title.
    Preserve DevOps when it is mentioned elsewhere as a required skill.
    """
    if not job_description:
        return ""

    text = str(job_description)

    # Remove DevOps Engineer from a Role: field, including inline descriptions.
    text = re.sub(
        r"\bRole\s*:\s*(?:(?:Senior|Junior|Lead|Principal|Associate|Staff|"
        r"Entry[- ]Level|Graduate|Trainee)\s+)?DevOps\s+Engineer\b\s*\.?",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Also remove a standalone job-title line.
    title_pattern = re.compile(
        r"^\s*(?:(?:Senior|Junior|Lead|Principal|Associate|Staff|"
        r"Entry[- ]Level|Graduate|Trainee)\s+)?"
        r"DevOps\s+Engineer"
        r"(?:\s*[-|,:].*)?\s*$",
        re.IGNORECASE,
    )

    lines = [
        line for line in text.splitlines()
        if not title_pattern.fullmatch(line)
    ]

    return "\n".join(lines)


# =========================================================
# REQUIREMENT EXTRACTION
# =========================================================

def extract_requirements(
    job_description: str,
    top_n: int = 30,
) -> list[dict]:
    """
    Extract recognized technical requirements that explicitly
    occur in the job description.

    Returns:
        [
            {
                "requirement": "python",
                "importance": 1.5
            }
        ]

    Exact duplicate aliases are consolidated.
    Related but distinct skills are not automatically treated
    as identical.
    """

    if not job_description:
        return []

    # Ignore a standalone DevOps Engineer heading, but preserve DevOps
    # when it appears elsewhere in the description as a stated requirement.
    description_without_title = remove_title_only_devops_lines(
        job_description
    )

    cleaned_text = clean_job_description(
        description_without_title
    )

    if not cleaned_text:
        return []

    segments = split_into_segments(
        cleaned_text
    )

    # Maps canonical skill to its highest observed importance.
    requirement_importance = defaultdict(float)

    # Keep track of first appearance for stable result ordering.
    first_seen = {}

    position_counter = 0

    # Sort longer phrases first to prioritize specific terms.
    sorted_terms = sorted(
        set(TECHNICAL_TERMS),
        key=len,
        reverse=True,
    )

    for segment in segments:

        importance = calculate_importance(
            segment
        )

        found_in_segment = set()

        for term in sorted_terms:

            if is_generic_phrase(term):
                continue

            pattern = make_term_pattern(term)

            if not pattern.search(segment):
                continue

            canonical = normalize_requirement_name(
                term
            )

            if not canonical:
                continue

            # Prevent the same canonical requirement from being
            # added repeatedly within a single segment.
            if canonical in found_in_segment:
                continue

            found_in_segment.add(canonical)

            if canonical not in first_seen:
                first_seen[canonical] = position_counter
                position_counter += 1

            requirement_importance[canonical] = max(
                requirement_importance[canonical],
                importance,
            )

    # Sort by importance, then original appearance.
    sorted_requirements = sorted(
        requirement_importance.items(),
        key=lambda item: (
            -item[1],
            first_seen[item[0]],
        ),
    )

    if top_n is not None:
        sorted_requirements = sorted_requirements[
            :max(0, top_n)
        ]

    return [
        {
            "requirement": requirement,
            "importance": round(importance, 2),
        }
        for requirement, importance in sorted_requirements
    ]