from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.rag.qdrant_sources import (
    QDRANT_URL,
    MODEL_NAME,
    search_sources,
)


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)

    model = SentenceTransformer(MODEL_NAME)

    tests = [
        "I need code implementations of RoPE",
        "Give me a definition and background of RoPE",
        "Find papers about rotary positional embeddings",
        "Practical discussion threads about RoPE bugs",
        "use github",
        "arxiv papers about RoPE",
    ]
    for q in tests:
        search_sources(client, model, q, limit=4)


if __name__ == "__main__":
    main()
