from __future__ import annotations

import os
import re
from typing import Dict, Any, List


DEFAULT_REPORTS_DIR = os.getenv("REPORTS_DIR", "src/reports/reports")


def _slugify(s: str, max_len: int = 60) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if not s:
        s = "report"
    return s[:max_len].rstrip("-")


def _next_report_id(out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)

    best = 0
    for name in os.listdir(out_dir):
        m = re.match(r"^(\d{4})__", name)
        if not m:
            continue
        n = int(m.group(1))
        if n > best:
            best = n
    return best + 1


def render_markdown(state: Dict[str, Any]) -> str:
    user_query = state.get("user_query") or ""
    sid = state.get("source_id") or state.get("candidate_source_id") or ""
    domain = state.get("source_domain") or ""
    reason = state.get("candidate_source_reason") or ""
    report_answer = state.get("report_answer") or ""
    web_results: List[Dict[str, Any]] = state.get("web_results") or []

    lines: List[str] = []
    lines.append("# Search Agent Report\n")
    lines.append(f"- **Query:** {user_query}")
    lines.append(f"- **Source:** {sid} ({domain})")
    if reason:
        lines.append(f"- **Why this source:** {reason}")
    lines.append("")

    lines.append("## Evidence\n")
    if not web_results:
        lines.append("_No web results._\n")
    else:
        for i, r in enumerate(web_results, 1):
            title = r.get("title") or ""
            url = r.get("url") or ""
            snippet = r.get("snippet") or ""
            quotes = r.get("quotes") or []

            lines.append(f"### Result {i}: {title}")
            if url:
                lines.append(f"- URL: {url}")
            if snippet:
                lines.append(f"- Snippet: {snippet}")
            if quotes:
                lines.append("- Quotes:")
                for q in quotes:
                    lines.append(f'  - "{q}"')
            lines.append("")

    lines.append("## Answer\n")
    lines.append(report_answer.strip() + "\n")

    return "\n".join(lines).strip() + "\n"


def render_html(state: Dict[str, Any]) -> str:
    user_query = (state.get("user_query") or "").strip()
    sid = (state.get("source_id") or state.get("candidate_source_id") or "").strip()
    domain = (state.get("source_domain") or "").strip()
    reason = (state.get("candidate_source_reason") or "").strip()
    report_answer = (state.get("report_answer") or "").strip()
    web_results: List[Dict[str, Any]] = state.get("web_results") or []

    def esc(x: str) -> str:
        return (
            x.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
        )

    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html lang='en'>")
    parts.append("<head>")
    parts.append("<meta charset='utf-8'/>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'/>")
    parts.append("<title>Search Agent Report</title>")
    parts.append(
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;"
        "max-width:920px;margin:32px auto;padding:0 16px;line-height:1.5;color:#111}"
        ".muted{color:#666}"
        ".card{border:1px solid #e6e6e6;border-radius:14px;padding:16px;margin:14px 0}"
        "h1{font-size:28px;margin:0 0 8px}"
        "h2{font-size:18px;margin:24px 0 10px}"
        "h3{font-size:15px;margin:0 0 8px}"
        "pre{white-space:pre-wrap;background:#fafafa;border:1px solid #eee;border-radius:12px;padding:12px}"
        "a{color:#0b57d0;text-decoration:none}"
        "a:hover{text-decoration:underline}"
        "ul{margin:8px 0 0 18px}"
        "</style>"
    )
    parts.append("</head>")
    parts.append("<body>")

    parts.append("<h1>Search Agent Report</h1>")
    parts.append("<div class='card'>")
    parts.append(f"<div><b>Query:</b> {esc(user_query)}</div>")
    parts.append(f"<div><b>Source:</b> {esc(sid)} ({esc(domain)})</div>")
    if reason:
        parts.append(f"<div class='muted'><b>Why:</b> {esc(reason)}</div>")
    parts.append("</div>")

    parts.append("<h2>Evidence</h2>")
    if not web_results:
        parts.append("<div class='card'><div class='muted'>No web results.</div></div>")
    else:
        for i, r in enumerate(web_results, 1):
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            quotes = r.get("quotes") or []

            parts.append("<div class='card'>")
            parts.append(f"<h3>Result {i}: {esc(title)}</h3>")
            if url:
                parts.append(
                    f"<div><b>URL:</b> <a href='{esc(url)}' target='_blank' rel='noreferrer'>{esc(url)}</a></div>"
                )
            if snippet:
                parts.append(f"<div class='muted' style='margin-top:6px'>{esc(snippet)}</div>")
            if quotes:
                parts.append("<div style='margin-top:10px'><b>Quotes:</b></div>")
                parts.append("<ul>")
                for q in quotes:
                    parts.append(f"<li>{esc(str(q))}</li>")
                parts.append("</ul>")
            parts.append("</div>")

    parts.append("<h2>Answer</h2>")
    parts.append("<div class='card'>")
    parts.append(f"<pre>{esc(report_answer)}</pre>")
    parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def save_reports(state: Dict[str, Any], out_dir: str | None = None) -> Dict[str, str]:
    out_dir = out_dir or DEFAULT_REPORTS_DIR

    user_query = state.get("user_query") or ""
    rid = _next_report_id(out_dir)
    slug = _slugify(user_query)
    base = f"{rid:04d}__{slug}"

    md_path = os.path.join(out_dir, base + ".md")
    html_path = os.path.join(out_dir, base + ".html")

    md = render_markdown(state)
    html = render_html(state)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return {"md": md_path, "html": html_path, "base": base}
