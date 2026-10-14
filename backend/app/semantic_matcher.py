import re
from functools import lru_cache


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model only when semantic matching is needed."""
    # Lazy import allows exact/equivalent matching to work even when the
    # optional semantic-matching dependency has not yet been installed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def split_resume_into_chunks(resume_text: str) -> list[str]:
    """Split resume text into readable lines/sentences for evidence retrieval."""
    if not resume_text or not resume_text.strip():
        return []

    chunks = []
    for line in resume_text.replace("\r", "\n").splitlines():
        line = re.sub(r"^\s*[-*•▪]+\s*", "", line.strip())
        if not line:
            continue

        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if len(sentence.split()) >= 2:
                chunks.append(sentence)

    # Preserve original order while removing duplicate sentences.
    return list(dict.fromkeys(chunks))


def calculate_semantic_similarity(resume_text: str, requirement: str) -> dict:
    """Return the closest resume sentence and its cosine similarity to a requirement."""
    chunks = split_resume_into_chunks(resume_text)
    if not chunks or not requirement or not requirement.strip():
        return {"similarity": 0.0, "evidence": None}

    model = get_model()
    requirement_embedding = model.encode(
        [requirement.strip()], normalize_embeddings=True, show_progress_bar=False
    )
    resume_embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )

    # Embeddings are normalized, so their dot product is cosine similarity.
    similarities = (resume_embeddings @ requirement_embedding[0]).tolist()
    best_index = max(range(len(similarities)), key=similarities.__getitem__)
    best_similarity = max(0.0, min(1.0, float(similarities[best_index])))

    return {
        "similarity": round(best_similarity, 4),
        "evidence": chunks[best_index],
    }
