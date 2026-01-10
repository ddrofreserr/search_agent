from typing import Dict, Any
from langgraph.types import interrupt

from src.graph.state import AgentState
from src.rag.qdrant_sources import pick_source, SOURCES
from src.graph.ollama import call_ollama

from src.reports.generate_report import save_reports

from src.web.tools import web_search_allowed, fetch_page_text, pick_short_quotes

from config import settings, INTENT_GUARD_PROMPT, REPORT_ANSWER_PROMPT, FORMAT_QUESTION


def node_intent_guard(state: AgentState) -> Dict[str, Any]:
    prompt = INTENT_GUARD_PROMPT.format(user_query=state["user_query"])

    text = call_ollama(prompt, model=settings.OLLAMA_MODEL)

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
        "source_domain": domain,
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

    q = FORMAT_QUESTION
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

    results = web_search_allowed(query, domain, max_results=settings.WEB_MAX_RESULTS)

    enriched = []
    for r in results[:settings.ENRICH_TOP_K]:
        text = fetch_page_text(r["url"], max_chars=settings.MAX_PAGE_CHARS)
        quotes = pick_short_quotes(text, max_quotes=2)
        enriched.append({
            "title": r["title"],
            "url": r["url"],
            "snippet": r["snippet"],
            "quotes": quotes,
        })

    return {"web_results": enriched}


def node_generate_report_answer(state: AgentState) -> Dict[str, Any]:
    sid = state.get("source_id") or state.get("candidate_source_id") or "wikipedia"
    domain = SOURCES[sid]["domain"]

    web_results = state.get("web_results") or []

    evidence = ""
    for i, r in enumerate(web_results, 1):
        evidence += (
            f"\nResult {i}:\n"
            f"Title: {r.get('title')}\n"
            f"URL: {r.get('url')}\n"
            f"Snippet: {r.get('snippet')}\n"
            f"Quotes:\n"
        )
        for q in (r.get("quotes") or []):
            evidence += f"- \"{q}\"\n"

    prompt = REPORT_ANSWER_PROMPT.format(
        user_query=state["user_query"],
        source_id=sid,
        source_domain=domain,
        evidence=evidence,
    )
    text = call_ollama(prompt, model=settings.OLLAMA_MODEL)

    text = call_ollama(prompt, model=settings.OLLAMA_MODEL)
    return {"report_answer": text}


def node_save_report(state: AgentState) -> Dict[str, Any]:
    paths = save_reports(state, out_dir=settings.REPORTS_DIR)
    return {
        "report_paths": {"md": paths["md"], "html": paths["html"]},
        "report_basename": paths["base"],
    }


def node_compose_answer(state: AgentState) -> Dict[str, Any]:
    paths = state.get("report_paths") or {}
    md_path = paths.get("md")
    html_path = paths.get("html")

    msg = "I wrote a report for your question."
    if html_path:
        msg += f"\nHTML: {html_path}"
    if md_path:
        msg += f"\nMD: {md_path}"

    return {"final_answer": msg}
