from typing import Dict, Any
from langgraph.types import interrupt

from src.graph.state import AgentState
from src.rag.allowlist import ALLOWED_SOURCES
from src.graph.ollama import call_ollama

from src.web.tools import web_search_allowed, fetch_page_text, pick_short_quotes


# -----------------------------------------------------------------------------
# NODE 1: проверка на долбоеба (intent/scope guard)
# -----------------------------------------------------------------------------
def node_intent_guard(state: AgentState) -> Dict[str, Any]:
    """
    Узел-предохранитель: проверяет, что запрос соответствует кейсу системы:
    "information research / summarization from allowed sources".
    """
    prompt = f"""
You are a gatekeeper for an information-retrieval agent.

The agent is ONLY allowed to:
- choose a source from an allowlist,
- (after user approval) search that source on the web,
- summarize findings.

If the user request is clearly outside this scope (e.g., requests for illegal wrongdoing,
harm, explicit hacking instructions, or unrelated tasks), refuse.

Output STRICTLY:
ALLOW: <yes|no>
REASON: <short reason>

User query:
{state["user_query"]}
""".strip()

    text = call_ollama(prompt, model="qwen2.5:3b")

    allow = "yes"
    reason = "OK"

    for line in text.splitlines():
        if line.startswith("ALLOW:"):
            allow = line.replace("ALLOW:", "").strip().lower()
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    if allow != "yes":
        return {
            "guard_blocked": True,
            "final_answer": f"Sorry — I can’t help with that request. {reason}",
        }

    return {"guard_blocked": False}


# -----------------------------------------------------------------------------
# NODE 2: выбор источника (упрощённая эвристика)
# -----------------------------------------------------------------------------
def node_select_source(state: AgentState) -> Dict[str, Any]:
    """
    LLM-based source selection.

    Роль узла:
    - semantic decision (анализ смысла запроса)
    - выбор источника ИЗ allowlist
    - генерация объяснения + текста для подтверждения

    ВАЖНО:
    - модель НЕ управляет графом
    - модель только возвращает данные
    """

    # красиво упаковываем allowlist для модели
    sources_description = "\n".join(
        f"- {sid}: {meta['desc']}"
        for sid, meta in ALLOWED_SOURCES.items()
    )

    prompt = f"""
You are an information retrieval agent.

Your task:
- Analyze the user's query.
- Choose exactly ONE source from the allowed list.
- Base your decision on the meaning of the query and the source descriptions.
- Explain your choice clearly and concisely.

CONFIRMATION_TEXT must:
- be ONE short line (max 120 characters)
- ask ONLY to confirm the chosen source (not the topic)
- contain the source id and domain
- end with: "(y/n or type another source_id)"
- do NOT ask any other questions

Allowed sources:
{sources_description}

User query:
{state['user_query']}

Respond STRICTLY in the following format:

SOURCE: <source_id>
REASON: <max 12 words>
CONFIRMATION_TEXT: <one line>
""".strip()

    text = call_ollama(prompt, model="qwen2.5:3b")

    # очень простой парсинг (без магии и regex)
    source_id = None
    reason = None
    confirmation = None

    for line in text.splitlines():
        if line.startswith("SOURCE:"):
            source_id = line.replace("SOURCE:", "").strip()
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
        if line.startswith("CONFIRMATION_TEXT:"):
            confirmation = line.replace("CONFIRMATION_TEXT:", "").strip()

    # минимальная защита от мусора
    if source_id not in ALLOWED_SOURCES:
        source_id = "wikipedia"
        reason = "Fallback to a general-purpose source."
        confirmation = "Shall I use Wikipedia for this query?"

    return {
        "candidate_source_id": source_id,
        "candidate_source_reason": reason,
        "approval_question": confirmation,
        "approved": None,
        "source_id": None,
        "final_answer": None,
    }


# -----------------------------------------------------------------------------
# NODE 3: human-in-the-loop interrupt (спросить подтверждение)
# -----------------------------------------------------------------------------
def node_approval_interrupt(state: AgentState) -> Any:
    # Если пользователь уже ответил (raw заполнен), не спрашиваем снова.
    if (state.get("user_approval_raw") or "").strip():
        return {}

    # approval_question лучше формировать заранее (в select или handle replan)
    q = (state.get("approval_question") or "").strip()
    if not q:
        sid = state.get("candidate_source_id") or "wikipedia"
        domain = ALLOWED_SOURCES[sid]["domain"]
        q = f"Use {sid} ({domain})? (y/n or type another source_id)"

    return interrupt({"question": q})


# -----------------------------------------------------------------------------
# NODE 4: обработать ответ человека
# -----------------------------------------------------------------------------
def node_handle_approval(state: AgentState) -> Dict[str, Any]:
    raw = (state.get("user_approval_raw") or "").strip()
    candidate = state.get("candidate_source_id") or "wikipedia"

    # 1) быстрый путь: пользователь подтвердил -> идём к финалу
    if raw.lower() in {"y", "yes", "да", "ok", "ага"}:
        return {
            "approved": True,
            "source_id": candidate,
            "user_approval_raw": None,   # <-- чтобы state был чистым после финала
        }

    # 2) иначе: LLM перепланирует источник на основе feedback
    sources_description = "\n".join(
        f"- {sid}: {meta['desc']} (domain: {meta['domain']})"
        for sid, meta in ALLOWED_SOURCES.items()
    )

    prompt = f"""
You are an information retrieval agent.

We proposed this source: {candidate}

User did NOT approve and said:
{raw}

Allowed sources:
{sources_description}

Task:
- Pick exactly ONE better source from the allowed list, based on the user's feedback.
- Explain briefly why.
- Write a short confirmation question for the new source.

CONFIRMATION_TEXT must:
- be ONE short line (max 120 characters)
- ask ONLY to confirm the chosen source (not the topic)
- contain the source id and domain
- end with: "(y/n or type another source_id)"
- do NOT ask any other questions

Output STRICTLY:
SOURCE: <source_id>
REASON: <max 12 words>
CONFIRMATION_TEXT: <one line>
""".strip()

    text = call_ollama(prompt, model="qwen2.5:3b")

    new_source = candidate
    new_reason = "Replanned after user feedback."
    new_confirm = "Use wikipedia (wikipedia.org)? (y/n or type another source_id)"

    for line in text.splitlines():
        if line.startswith("SOURCE:"):
            new_source = line.replace("SOURCE:", "").strip()
        if line.startswith("REASON:"):
            new_reason = line.replace("REASON:", "").strip()
        if line.startswith("CONFIRMATION_TEXT:"):
            new_confirm = line.replace("CONFIRMATION_TEXT:", "").strip()

    if new_source not in ALLOWED_SOURCES:
        new_source = "wikipedia"

    print("[handle_approval] replanned -> asking again")
    return {
        "approved": False,
        "source_id": None,
        "candidate_source_id": new_source,
        "candidate_source_reason": new_reason,
        "approval_question": new_confirm,

        "user_approval_raw": None, 
    }


# -----------------------------------------------------------------------------
# NODE 5: approved acknowledgement 
# -----------------------------------------------------------------------------
# def node_approved_ack(state: AgentState) -> Dict[str, Any]:
#     sid = state.get("source_id") or state.get("candidate_source_id") or "wikipedia"
#     domain = ALLOWED_SOURCES[sid]["domain"]
#     q = state["user_query"]

#     prompt = f"""
# You are an information research agent.

# Write ONE short paragraph that:
# - acknowledges the confirmation,
# - says you will now search within the chosen source,
# - does NOT claim the search already happened.

# Chosen source: {sid} ({domain})
# User query: {q}
# """.strip()

#     return {"approved_answer": call_ollama(prompt, model="qwen2.5:3b")}


# -----------------------------------------------------------------------------
# NODE 6: web search tool execution (domain-restricted)
# -----------------------------------------------------------------------------
def node_web_search(state: AgentState) -> Dict[str, Any]:
    sid = state.get("source_id") or state.get("candidate_source_id") or "wikipedia"
    domain = ALLOWED_SOURCES[sid]["domain"]
    query = state["user_query"]

    results = web_search_allowed(query, domain, max_results=5)

    enriched = []
    for r in results[:2]:
        text = fetch_page_text(r["url"])
        quotes = pick_short_quotes(text, max_quotes=2)
        enriched.append({
            "title": r["title"],
            "url": r["url"],
            "snippet": r["snippet"],
            "quotes": quotes,
        })

    return {"web_results": enriched}


# -----------------------------------------------------------------------------
# NODE 7: сгенерировать финальный ответ (LLM)
# -----------------------------------------------------------------------------
def node_compose_answer(state: AgentState) -> Dict[str, Any]:
    sid = state.get("source_id") or state.get("candidate_source_id") or "wikipedia"
    domain = ALLOWED_SOURCES[sid]["domain"]

    ack = (state.get("approved_answer") or "").strip()
    web_results = state.get("web_results") or []

    evidence = ""
    for i, r in enumerate(web_results, 1):
        evidence += f"\nResult {i}:\nTitle: {r.get('title')}\nURL: {r.get('url')}\nSnippet: {r.get('snippet')}\nQuotes:\n"
        for q in (r.get("quotes") or []):
            evidence += f"- \"{q}\"\n"

    prompt = f"""
You are an information research assistant.

Rules:
- Write: "What I found:" and give 2–4 bullet points with key findings.
- Include 1–2 short direct quotes (already provided) and cite the URL right after each quote.
- End with a concise answer to the user's question ("Therefore...").

User query:
{state["user_query"]}

Chosen source:
{sid} ({domain})

Evidence (search results + extracted quotes):
{evidence}
""".strip()

    final = call_ollama(prompt, model="qwen2.5:3b")
    if ack:
        final = ack + "\n\n" + final

    return {"final_answer": final}
