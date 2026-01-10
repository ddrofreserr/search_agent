import os
from pydantic_settings import BaseSettings
from pydantic import Field


# -----------------------------
# Prompts (templates)
# -----------------------------

INTENT_GUARD_PROMPT = """
You are a gatekeeper for an information-retrieval agent.

The agent is ONLY allowed to:
- choose a source from an allowlist,
- (after user approval) search that source on the web,
- summarize findings,
- generate a report (Markdown/HTML) from collected evidence.

If the user request is clearly outside this scope (e.g., requests for illegal wrongdoing,
harm, explicit hacking instructions, or unrelated tasks), refuse.

Output STRICTLY:
ALLOW: <yes|no>
REASON: <short reason>

User query:
{user_query}
""".strip()


REPORT_ANSWER_PROMPT = """
You are an information research assistant.

Rules:
- Write: "What I found:" and give 2–4 bullet points with key findings.
- Include 1–2 short direct quotes (already provided) and cite the URL right after each quote.
- End with a concise answer to the user's question ("Therefore...").

User query:
{user_query}

Chosen source:
{source_id} ({source_domain})

Evidence (search results + extracted quotes):
{evidence}
""".strip()


FORMAT_QUESTION = "Could you clarify the format? Examples: papers / code / discussion / short summary"


# -----------------------------
# Settings
# -----------------------------

class Settings(BaseSettings):
    # LLM (Ollama)
    OLLAMA_MODEL: str = Field(default="qwen2.5:3b", description="Default Ollama model name")

    # Qdrant (RAG for source selection)
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant URL")
    QDRANT_SOURCES_COLLECTION: str = Field(default="sources", description="Qdrant collection for sources")
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="SentenceTransformer model")

    # Web search / enrichment
    WEB_MAX_RESULTS: int = Field(default=5, description="DDG max search results")
    ENRICH_TOP_K: int = Field(default=2, description="How many top results to fetch and quote")
    MAX_PAGE_CHARS: int = Field(default=6000, description="Max chars to keep from fetched page")
    MAX_QUOTES: int = Field(default=2, description="Quotes per fetched page")
    MAX_QUOTE_LEN: int = Field(default=220, description="Max length of each quote")

    # Reports
    REPORTS_DIR: str = Field(default="reports/reports", description="Directory for generated reports")

    # Misc
    EXPECT_ENGLISH: bool = Field(default=True, description="Project is designed for English queries")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
