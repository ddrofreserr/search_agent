from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.rag.qdrant_sources import (
    QDRANT_URL,
    COLLECTION,
    MODEL_NAME,
    ensure_collection,
    upsert_sources,
)


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Узнаём размер вектора (у MiniLM это 384)
    vec_size = len(model.encode(["test"])[0])

    ensure_collection(client, vector_size=vec_size)

    print(f"Upserting sources into Qdrant коллекцию '{COLLECTION}' @ {QDRANT_URL}")
    upsert_sources(client, model)

    print("Done. Qdrant 'sources' collection is ready.")


if __name__ == "__main__":
    main()
