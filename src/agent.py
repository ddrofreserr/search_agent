from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, START, END

from config import settings  # central config (model, paths, limits, etc.)
from src.graph.state import AgentState
from src.graph.router import route_after_handle_approval, route_after_guard
from src.rag.qdrant_sources import SOURCES

from src.graph.nodes import (
    node_intent_guard,
    node_select_source,
    node_approval_interrupt,
    node_handle_approval,
    node_format_interrupt,
    node_handle_format,
    node_web_search,
    node_generate_report_answer,
    node_save_report,           
    node_compose_answer,
)


class SearchAgent:
    """
    Thin wrapper around existing functional nodes.
    Responsibilities:
    - build LangGraph
    - provide run_cli() (interactive human-in-the-loop)
    - provide run_once() (non-interactive, useful for API/tests)
    """

    def build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("intent_guard", node_intent_guard)
        g.add_node("select_source", node_select_source)
        g.add_node("approval", node_approval_interrupt)
        g.add_node("handle_approval", node_handle_approval)

        g.add_node("format", node_format_interrupt)
        g.add_node("handle_format", node_handle_format)

        g.add_node("web_search", node_web_search)
        g.add_node("generate_report_answer", node_generate_report_answer)
        g.add_node("save_report", node_save_report)
        g.add_node("compose_answer", node_compose_answer)

        g.add_edge(START, "intent_guard")

        g.add_conditional_edges(
            "intent_guard",
            route_after_guard,
            {"blocked": END, "ok": "select_source"},
        )

        g.add_edge("select_source", "approval")
        g.add_edge("approval", "handle_approval")

        g.add_conditional_edges(
            "handle_approval",
            route_after_handle_approval,
            {"approved": "web_search", "need_format": "format", "revise": "approval"},
        )

        g.add_edge("format", "handle_format")
        g.add_edge("handle_format", "select_source")

        g.add_edge("web_search", "generate_report_answer")
        g.add_edge("generate_report_answer", "save_report")
        g.add_edge("save_report", "compose_answer")
        g.add_edge("compose_answer", END)

        return g.compile()

    def _initial_state(self, user_query: str) -> AgentState:
        # Keep this aligned with src/graph/state.py fields
        return {
            "user_query": user_query,

            "candidate_source_id": None,
            "candidate_source_reason": None,
            "approval_question": None,

            "user_approval_raw": None,
            "approved": None,
            "source_id": None,

            "user_format_pref": None,
            "source_query": None,
            "need_format": None,
            "rejected_source_ids": [],

            "source_domain": None,

            "web_results": None,
            "guard_blocked": None,

            "report_answer": None,
            "report_paths": None,
            "report_basename": None,

            "final_answer": None,
        }

    def run_cli(self, user_query: str) -> AgentState:
        """
        Interactive mode: respects interrupts and asks user for input.
        """
        app = self.build_graph()
        state = self._initial_state(user_query)

        while True:
            out = app.invoke(state)
            state.update(out)

            if state.get("approved") and state.get("source_id") and not state.get("_approval_announced"):
                sid = state["source_id"]
                meta = SOURCES[sid]
                print(f"\n✔ Source confirmed: {sid} ({meta['domain']})")
                print(f"→ Searching now and will generate a report in: {settings.REPORTS_DIR}\n")
                state["_approval_announced"] = True

            if "__interrupt__" in out:
                intr = out["__interrupt__"]
                for payload in intr:
                    data = payload.value if hasattr(payload, "value") else payload
                    question = data.get("question", "Confirm? (y/n or type another source_id)")
                    print("\n" + question)

                    ans = ""
                    while not ans:
                        ans = input("> ").strip()
                    state["user_approval_raw"] = ans
                continue

            print(state.get("final_answer"))
            return state

    def run_once(
        self,
        user_query: str,
        approval: Optional[str] = "y",
        format_pref: Optional[str] = None,
    ) -> AgentState:
        """
        Non-interactive mode: provides inputs up-front so the graph doesn't interrupt.
        Useful for FastAPI or tests.

        - approval: "y" / "n" / "github" / "arxiv" ...
        - format_pref: optional string that will be appended to source_query (via handle_format)
        """
        app = self.build_graph()
        state = self._initial_state(user_query)

        if format_pref:
            state["user_format_pref"] = format_pref
            state["source_query"] = f"{user_query}\nPreferred format: {format_pref}"

        if approval:
            state["user_approval_raw"] = approval

        out = app.invoke(state)
        state.update(out)

        return state
