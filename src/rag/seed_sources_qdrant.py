"""
Мы создаём коллекцию 'sources' и кладём туда 4 "источника":
- у каждого источника есть:
  - vector: dense-эмбеддинг текста (MiniLM)
  - payload: простые поля (source_id/title/domain/desc)

Потом делаем поиск:
- векторизуем query
- отправляем query vector в Qdrant
- получаем top-k ближайших points по cosine similarity

Важно:
- Здесь НЕТ "настоящего hybrid внутри Qdrant".
  Это только dense поиск в Qdrant.
  Hybrid (BM25 + dense) можно добавить позже (или сделать локально).
"""

import os
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_SOURCES_COLLECTION", "sources")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SOURCES: List[Dict[str, Any]] = [
    {
        "source_id": "wikipedia",
        "title": "Wikipedia",
        "domain": "wikipedia.org",
        "desc": "General-purpose encyclopedia: definitions, overviews, background context.",
    },
    {
        "source_id": "github",
        "title": "GitHub",
        "domain": "github.com",
        "desc": "Code repositories: implementations, libraries, issues, examples.",
    },
    {
        "source_id": "reddit",
        "title": "Reddit",
        "domain": "reddit.com",
        "desc": "Community discussions: practical tips, opinions, debugging threads (verify claims).",
    },
    {
        "source_id": "arxiv",
        "title": "arXiv",
        "domain": "arxiv.org",
        "desc": "Scientific preprints: papers, methods, research context.",
    },
]


def text_for_embedding(src):
    return src["desc"]


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    """
    Коллекция в Qdrant = таблица/контейнер для points.
    В нашем случае у points будет 1 dense-вектор фиксированного размера.
    """
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,  # cosine similarity
        ),
    )


def upsert_sources(client: QdrantClient, model: SentenceTransformer) -> None:
    """
    upsert = вставить или обновить points.
    Point = { id, vector, payload }.
    """
    texts = [text_for_embedding(s) for s in SOURCES]

    # normalize_embeddings=True делает cosine-скоры более "предсказуемыми"
    vectors = model.encode(texts, normalize_embeddings=True).tolist()

    points: List[PointStruct] = []
    for idx, src in enumerate(SOURCES, start=1):
        payload = {
            "source_id": src["source_id"],
            "title": src["title"],
            "domain": src["domain"],
            "desc": src["desc"],
        }
        points.append(PointStruct(id=idx, vector=vectors[idx - 1], payload=payload))

    client.upsert(collection_name=COLLECTION, points=points)


def search_sources(client: QdrantClient, model: SentenceTransformer, query: str, limit: int = 4) -> None:
    """
    Векторный поиск:
    1) query -> эмбеддинг
    2) отправляем query vector в Qdrant
    3) получаем ближайшие points (top-k) + их payload
    """
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()

    # В новых версиях клиента рекомендуется Query API:
    # client.query_points(..., query=[...])
    hits = client.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=limit,
        with_payload=True,
    ).points

    print(f"\nQuery: {query}")
    for i, h in enumerate(hits, start=1):
        p = h.payload or {}
        print(f"{i}) score={h.score:.4f}  -> {p.get('source_id')}  ({p.get('domain')})  | {p.get('title')}")


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)

    print(f"Loading model: {MODEL_NAME}")
    # При первом запуске sentence-transformers скачает модель автоматически (в кеш HuggingFace)
    model = SentenceTransformer(MODEL_NAME)

    # Узнаём размер вектора (у MiniLM это 384)
    vec_size = len(model.encode(["test"])[0])

    ensure_collection(client, vector_size=vec_size)

    print(f"Upserting {len(SOURCES)} sources into Qdrant коллекцию '{COLLECTION}' @ {QDRANT_URL}")
    upsert_sources(client, model)

    print("\n--- TEST DENSE SEARCH (Qdrant) ---")
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
