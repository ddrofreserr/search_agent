from __future__ import annotations

import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from src.agent import SearchAgent


app = FastAPI(title="Search Agent Web CLI", version="0.1")

agent = SearchAgent()
graph = agent.build_graph()

# In-memory sessions: {session_id: {"state": AgentState, "created_at": float, "log": [str]}}
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SECONDS = 30 * 60  # 30 minutes


def _esc(x: str) -> str:
    return (
        (x or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _cleanup_sessions() -> None:
    now = time.time()
    dead = []
    for sid, data in SESSIONS.items():
        if now - float(data.get("created_at", now)) > SESSION_TTL_SECONDS:
            dead.append(sid)
    for sid in dead:
        SESSIONS.pop(sid, None)


def _get_interrupt_question(out: Dict[str, Any]) -> str | None:
    intr = out.get("__interrupt__")
    if not intr:
        return None

    payload = intr[0].value if hasattr(intr[0], "value") else intr[0]
    if isinstance(payload, dict):
        return payload.get("question")
    return "Input required."


def _render_page(
    *,
    title: str,
    session_id: str | None = None,
    query: str = "",
    log_lines: list[str] | None = None,
    question: str | None = None,
    final_answer: str | None = None,
    report_paths: dict | None = None,
) -> HTMLResponse:
    log_lines = log_lines or []

    log_html = ""
    if log_lines:
        items = "\n".join(f"<li>{_esc(x)}</li>" for x in log_lines)
        log_html = f"""
        <div class="card">
          <h3>Conversation</h3>
          <ul>{items}</ul>
        </div>
        """

    question_html = ""
    if question and session_id:
        question_html = f"""
        <div class="card">
          <h3>Agent question</h3>
          <div class="muted">{_esc(question)}</div>
          <form method="post" action="/continue" style="margin-top:12px">
            <input type="hidden" name="session_id" value="{_esc(session_id)}"/>
            <input name="answer" placeholder="Type y/n or a source_id (github/arxiv/...)" />
            <div style="height:10px"></div>
            <button type="submit">Send</button>
          </form>
        </div>
        """

    result_html = ""
    if final_answer is not None:
        rp = report_paths or {}
        md_path = rp.get("md")
        html_path = rp.get("html")

        result_html = f"""
        <div class="card">
          <h3>Final answer</h3>
          <pre>{_esc(final_answer)}</pre>
        </div>
        <div class="card">
          <h3>Reports</h3>
          <div class="muted">HTML: {_esc(str(html_path) if html_path else "")}</div>
          <div class="muted">MD: {_esc(str(md_path) if md_path else "")}</div>
        </div>
        """

    base_form = """
    <div class="card">
      <div class="muted">English queries only.</div>
      <form method="post" action="/run" style="margin-top:12px">
        <label><b>Query</b></label><br/>
        <textarea name="query" placeholder="Find papers about rotary positional embeddings..."></textarea>
        <div style="height:10px"></div>
        <button type="submit">Start</button>
      </form>
    </div>
    """

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_esc(title)}</title>
  <style>
    body{{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;max-width:920px;margin:32px auto;padding:0 16px;line-height:1.5;color:#111}}
    .card{{border:1px solid #e6e6e6;border-radius:14px;padding:16px;margin:14px 0}}
    textarea,input{{width:100%;box-sizing:border-box;border:1px solid #ddd;border-radius:10px;padding:10px;font-size:14px}}
    textarea{{min-height:120px}}
    button{{border:0;border-radius:12px;padding:10px 14px;font-size:14px;cursor:pointer}}
    .muted{{color:#666}}
    pre{{white-space:pre-wrap;background:#fafafa;border:1px solid #eee;border-radius:12px;padding:12px}}
    ul{{margin:8px 0 0 18px}}
  </style>
</head>
<body>
  <h1>Search Agent (Web CLI)</h1>
  {base_form}
  {log_html}
  {question_html}
  {result_html}
</body>
</html>
"""
    return HTMLResponse(html)


@app.get("/", response_class=HTMLResponse)
def index():
    _cleanup_sessions()
    return _render_page(title="Search Agent")


@app.get("/favicon.ico")
def favicon():
    return HTMLResponse("", status_code=204)


@app.post("/run", response_class=HTMLResponse)
def run(query: str = Form(...)):
    _cleanup_sessions()

    q = (query or "").strip()
    if not q:
        return _render_page(title="Search Agent", final_answer="Empty query.")

    session_id = uuid.uuid4().hex
    state = agent._initial_state(q)  # uses your existing initializer in src/agent.py

    SESSIONS[session_id] = {
        "state": state,
        "created_at": time.time(),
        "log": [f"User: {q}"],
    }

    # Run graph until interrupt or finish
    out = graph.invoke(state)
    state.update(out)

    question = _get_interrupt_question(out)
    if question:
        SESSIONS[session_id]["log"].append(f"Agent: {question}")
        return _render_page(
            title="Search Agent",
            session_id=session_id,
            query=q,
            log_lines=SESSIONS[session_id]["log"],
            question=question,
        )

    # Finished immediately
    final_answer = state.get("final_answer") or ""
    SESSIONS[session_id]["log"].append("Agent: (finished)")
    return _render_page(
        title="Search Agent",
        session_id=session_id,
        query=q,
        log_lines=SESSIONS[session_id]["log"],
        final_answer=final_answer,
        report_paths=state.get("report_paths"),
    )


@app.post("/continue", response_class=HTMLResponse)
def cont(session_id: str = Form(...), answer: str = Form(...)):
    _cleanup_sessions()

    sid = (session_id or "").strip()
    ans = (answer or "").strip()

    data = SESSIONS.get(sid)
    if not data:
        return _render_page(
            title="Search Agent",
            final_answer="Session expired or not found. Start again.",
        )

    state = data["state"]
    log_lines = data["log"]

    if not ans:
        return _render_page(
            title="Search Agent",
            session_id=sid,
            query=state.get("user_query") or "",
            log_lines=log_lines,
            question="Please type an answer.",
        )

    # Put user answer into the same field that CLI uses
    state["user_approval_raw"] = ans
    log_lines.append(f"User: {ans}")

    out = graph.invoke(state)
    state.update(out)

    question = _get_interrupt_question(out)
    if question:
        log_lines.append(f"Agent: {question}")
        return _render_page(
            title="Search Agent",
            session_id=sid,
            query=state.get("user_query") or "",
            log_lines=log_lines,
            question=question,
        )

    final_answer = state.get("final_answer") or ""
    log_lines.append("Agent: (finished)")
    return _render_page(
        title="Search Agent",
        session_id=sid,
        query=state.get("user_query") or "",
        log_lines=log_lines,
        final_answer=final_answer,
        report_paths=state.get("report_paths"),
    )
