
from __future__ import annotations

from typing import TypedDict, Optional

# -----------------------------
# STATE (память графа)
# -----------------------------
class AgentState(TypedDict):

    user_query: str

    guard_blocked: Optional[bool]

    # То, что агент "предложил" как источник.
    candidate_source_id: Optional[str]        # например "wikipedia"
    candidate_source_reason: Optional[str]    # короткое объяснение "почему"
    approval_question: Optional[str]

    # Human-in-the-loop поля:
    user_approval_raw: Optional[str]          # что ввёл человек (y / github / etc.)
    approved: Optional[bool]                  # True/False (после обработки)
    source_id: Optional[str]                  # подтверждённый источник (итог)

    web_results: Optional[list]

    # Текст ответа
    final_answer: Optional[str]
