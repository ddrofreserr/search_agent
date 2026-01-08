import os
from typing import Dict, Any, List, Tuple, Optional

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_SOURCES_COLLECTION", "sources")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SOURCES: Dict[str, Dict[str, str]] = {
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


def _tokenize(s: str) -> List[str]:
    s = "".join(ch.lower() if ch.isalnum() else " " for ch in s)
    return [t for t in s.split() if t]


def _bm25_text(source_id: str, meta: Dict[str, str]) -> str:
    return f"{source_id} {meta['title']} {meta['domain']} {meta['desc']}"


_BM25_DOC_IDS = list(SOURCES.keys())
_BM25_DOCS = [_tokenize(_bm25_text(sid, SOURCES[sid])) for sid in _BM25_DOC_IDS]
_BM25 = BM25Okapi(_BM25_DOCS)

_client: Optional[QdrantClient] = None
_model: Optional[SentenceTransformer] = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _dense_search_scores(query: str) -> Dict[str, float]:
    client = _get_client()
    model = _get_model()

    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
    hits = client.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=len(SOURCES),
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
    ql = query.lower()

    # hard rule: explicit source mentioned (unless excluded)
    for sid in SOURCES.keys():
        if sid in ql:
            if exclude and sid in exclude:
                break
            return sid, f"rule: query mentions '{sid}'"

    dense = _dense_search_scores(query)
    bm25 = _bm25_scores(query)

    for sid in SOURCES.keys():
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

    for sid in SOURCES.keys():
        d = dense[sid]
        d_norm = (d + 1.0) / 2.0

        b = bm25[sid]
        b_norm = bm_norm(b)

        f = alpha * d_norm + (1.0 - alpha) * b_norm
        fused[sid] = f

        debug[sid] = {
            "dense": d,
            "bm25": b,
            "fused": f,
        }

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
