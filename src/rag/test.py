from qdrant_client import QdrantClient
from config import settings

client = QdrantClient(url=settings.QDRANT_URL)
pts, _ = client.scroll(
    collection_name=settings.QDRANT_SOURCES_COLLECTION,
    limit=50,
    with_payload=True,
    with_vectors=False,
)

for p in pts:
    print(p.payload)
