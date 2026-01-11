from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

from config import settings


SEED_SOURCES = {
    "wikipedia": {
        "title": "Wikipedia",
        "domain": "wikipedia.org",
        "desc": "General-purpose encyclopedia: definitions, overviews, background context.",
    },
    "github": {
        "title": "GitHub",
        "domain": "github.com",
        "desc": "Code repositories: implementations, libraries, issues, examples.",
    },
    "reddit": {
        "title": "Reddit",
        "domain": "reddit.com",
        "desc": "Community discussions: practical tips, opinions, debugging threads (verify claims).",
    },
    "arxiv": {
        "title": "arXiv",
        "domain": "arxiv.org",
        "desc": "Scientific preprints: papers, methods, research context.",
    },
}


def main() -> None:
    client = QdrantClient(url=settings.QDRANT_URL)
    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    dim = model.get_sentence_embedding_dimension()
    collection = settings.QDRANT_SOURCES_COLLECTION

    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = []
    for i, (source_id, meta) in enumerate(SEED_SOURCES.items(), start=1):
        text = f"{source_id} {meta['title']} {meta['domain']} {meta['desc']}"
        vec = model.encode(text, normalize_embeddings=True).tolist()

        payload = {
            "source_id": source_id,
            "title": meta["title"],
            "domain": meta["domain"],
            "desc": meta["desc"],
            "text": text, 
        }

        points.append(PointStruct(id=i, vector=vec, payload=payload))

    client.upsert(collection_name=collection, points=points)
    print(f"OK: initialized {collection} with {len(points)} sources")


if __name__ == "__main__":
    main()
