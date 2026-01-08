from typing import Dict, Any
from langgraph.types import interrupt

from src.graph.state import AgentState
from src.rag.qdrant_sources import pick_source, SOURCES
from src.graph.ollama import call_ollama

from src.web.tools import web_search_allowed, fetch_page_text, pick_short_quotes


def node_intent_guard(state: AgentState) -> Dict[str, Any]:
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


def node_select_source(state: AgentState) -> Dict[str, Any]:
    q = (state.get("source_query") or state["user_query"]).strip()
    excluded = state.get("rejected_source_ids") or []

    source_id, reason = pick_source(q, alpha=0.65, exclude=excluded)

    domain = SOURCES[source_id]["domain"]
    confirmation = f"Use {source_id} ({domain})? (y/n or type another source_id)"

    return {
        "candidate_source_id": source_id,
        "candidate_source_reason": reason,
        "approval_question": confirmation,
        "approved": None,
        "source_id": None,
        "final_answer": None,
        "need_format": None,
    }


def node_approval_interrupt(state: AgentState) -> Any:
    if (state.get("user_approval_raw") or "").strip():
        return {}

    q = (state.get("approval_question") or "").strip()
    if not q:
        sid = state.get("candidate_source_id") or "wikipedia"
        domain = SOURCES[sid]["domain"]
        q = f"Use {sid} ({domain})? (y/n or type another source_id)"

    return interrupt({"question": q})


def node_format_interrupt(state: AgentState) -> Any:
    if (state.get("user_format_pref") or "").strip():
        return {}

    q = "Could you clarify the format? Examples: papers / code / discussion / short summary"
    return interrupt({"question": q})


def node_handle_format(state: AgentState) -> Dict[str, Any]:
    raw = (state.get("user_approval_raw") or "").strip()
    fmt = raw
    base = state["user_query"]
    source_query = base if not fmt else f"{base}\nPreferred format: {fmt}"

    return {
        "user_format_pref": fmt,
        "source_query": source_query,
        "user_approval_raw": None,
        "need_format": None,
    }


def node_handle_approval(state: AgentState) -> Dict[str, Any]:
    raw = (state.get("user_approval_raw") or "").strip()
    if not raw:
        return {
            "user_approval_raw": None,
            "approved": None,
            "need_format": None,
        }
    candidate = state.get("candidate_source_id") or "wikipedia"
    low = raw.lower()

    if low in {"y", "yes", "да", "ok", "ага"}:
        return {
            "approved": True,
            "source_id": candidate,
            "user_approval_raw": None,
            "need_format": None,
        }

    if low in SOURCES:
        return {
            "approved": True,
            "source_id": low,
            "user_approval_raw": None,
            "need_format": None,
        }

    if low in {"n", "no", "нет", "неа"}:
        rejected = list(state.get("rejected_source_ids") or [])
        if candidate not in rejected:
            rejected.append(candidate)

        return {
            "approved": False,
            "source_id": None,
            "user_approval_raw": None,
            "need_format": True,
            "rejected_source_ids": rejected,
            "approval_question": None,
            "candidate_source_id": None,
            "candidate_source_reason": None,
        }

    rejected = list(state.get("rejected_source_ids") or [])
    if candidate not in rejected:
        rejected.append(candidate)

    return {
        "approved": False,
        "source_id": None,
        "user_approval_raw": None,
        "need_format": True,
        "rejected_source_ids": rejected,
        "user_format_pref": raw,
        "approval_question": None,
        "candidate_source_id": None,
        "candidate_source_reason": None,
    }


def node_web_search(state: AgentState) -> Dict[str, Any]:
    sid = state.get("source_id") or state.get("candidate_source_id") or "wikipedia"
    domain = SOURCES[sid]["domain"]
    query = (state.get("source_query") or state["user_query"])

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


def node_compose_answer(state: AgentState) -> Dict[str, Any]:
    sid = state.get("source_id") or state.get("candidate_source_id") or "wikipedia"
    domain = SOURCES[sid]["domain"]

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
