from __future__ import annotations
from typing import TypedDict, Optional


class AgentState(TypedDict):
    user_query: str
    guard_blocked: Optional[bool]

    candidate_source_id: Optional[str]
    candidate_source_reason: Optional[str]
    approval_question: Optional[str]

    user_approval_raw: Optional[str]
    approved: Optional[bool]
    source_id: Optional[str]

    user_format_pref: Optional[str]
    source_query: Optional[str]

    rejected_source_ids: Optional[list[str]]
    need_format: Optional[bool]

    web_results: Optional[list]
    final_answer: Optional[str]
