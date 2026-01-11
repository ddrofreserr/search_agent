# src/rag/qdrant_sources.py

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import settings


_client: Optional[QdrantClient] = None
_model: Optional[SentenceTransformer] = None

# кеш источников из Qdrant (payload)
_SOURCES: Optional[Dict[str, Dict[str, str]]] = None
_BM25: Optional[BM25Okapi] = None
_BM25_DOC_IDS: List[str] = []
_BM25_DOCS: List[List[str]] = []


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.QDRANT_URL)
    return _client


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def _tokenize(s: str) -> List[str]:
    s = "".join(ch.lower() if ch.isalnum() else " " for ch in (s or ""))
    return [t for t in s.split() if t]


def _load_sources_from_qdrant() -> Dict[str, Dict[str, str]]:
    client = _get_client()
    collection = settings.QDRANT_SOURCES_COLLECTION

    points, _ = client.scroll(
        collection_name=collection,
        limit=256,
        with_payload=True,
        with_vectors=False,
    )

    out: Dict[str, Dict[str, str]] = {}
    for p in points:
        payload = p.payload or {}
        sid = payload.get("source_id")
        if not sid:
            continue
        out[sid] = {
            "title": payload.get("title") or sid,
            "domain": payload.get("domain") or "",
            "desc": payload.get("desc") or "",
            "text": payload.get("text") or "",
        }

    return out


def get_sources() -> Dict[str, Dict[str, str]]:
    global _SOURCES
    if _SOURCES is None:
        _SOURCES = _load_sources_from_qdrant()
    return _SOURCES


def _ensure_bm25() -> None:
    global _BM25, _BM25_DOC_IDS, _BM25_DOCS

    if _BM25 is not None:
        return

    sources = get_sources()
    _BM25_DOC_IDS = list(sources.keys())
    _BM25_DOCS = [_tokenize(sources[sid]["text"]) for sid in _BM25_DOC_IDS]
    _BM25 = BM25Okapi(_BM25_DOCS)


def _dense_search_scores(query: str) -> Dict[str, float]:
    client = _get_client()
    model = _get_model()

    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
    hits = client.query_points(
        collection_name=settings.QDRANT_SOURCES_COLLECTION,
        query=qvec,
        limit=64,
        with_payload=True,
    ).points

    out: Dict[str, float] = {}
    for h in hits:
        p = h.payload or {}
        sid = p.get("source_id")
        if sid:
            out[sid] = float(h.score)
    return out


def _bm25_scores(query: str) -> Dict[str, float]:
    _ensure_bm25()
    assert _BM25 is not None

    q_tokens = _tokenize(query)
    scores = _BM25.get_scores(q_tokens)

    out: Dict[str, float] = {}
    for sid, sc in zip(_BM25_DOC_IDS, scores):
        out[sid] = float(sc)
    return out


def pick_source(
    query: str,
    alpha: float = 0.65,
    exclude: Optional[List[str]] = None,
) -> Tuple[str, str]:
    sources = get_sources()
    ql = query.lower()

    # правило: если явно упомянули source_id в запросе
    for sid in sources.keys():
        if sid in ql and not (exclude and sid in exclude):
            return sid, f"rule: query mentions '{sid}'"

    dense = _dense_search_scores(query)
    bm25 = _bm25_scores(query)

    for sid in sources.keys():
        dense.setdefault(sid, 0.0)
        bm25.setdefault(sid, 0.0)

    bm_vals = list(bm25.values())
    bm_min, bm_max = min(bm_vals), max(bm_vals)

    def bm_norm(x: float) -> float:
        if bm_max - bm_min < 1e-9:
            return 0.0
        return (x - bm_min) / (bm_max - bm_min)

    fused: Dict[str, float] = {}
    debug: Dict[str, Dict[str, float]] = {}

    for sid in sources.keys():
        d = dense[sid]
        d_norm = (d + 1.0) / 2.0  # как у тебя было

        b = bm25[sid]
        b_norm = bm_norm(b)

        f = alpha * d_norm + (1.0 - alpha) * b_norm
        fused[sid] = f
        debug[sid] = {"dense": d, "bm25": b, "fused": f}

    candidates = fused.copy()
    if exclude:
        for sid in exclude:
            candidates.pop(sid, None)
    if not candidates:
        candidates = fused

    best = max(candidates.items(), key=lambda kv: kv[1])[0]
    info = debug[best]
    reason = (
        f"hybrid: {best} fused={info['fused']:.4f} "
        f"(dense={info['dense']:.4f}, bm25={info['bm25']:.4f}, alpha={alpha})"
    )
    return best, reason
