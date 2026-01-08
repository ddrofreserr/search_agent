from langgraph.graph import StateGraph, START, END

from src.graph.state import AgentState
from src.graph.nodes import (
    node_intent_guard,
    node_select_source,
    node_approval_interrupt,
    node_handle_approval,
    node_web_search,
    node_compose_answer,
    node_format_interrupt,
    node_handle_format,
    node_generate_report_answer,
    node_save_report,
)
from src.graph.router import route_after_handle_approval, route_after_guard
from src.rag.qdrant_sources import SOURCES


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("intent_guard", node_intent_guard)
    g.add_node("select_source", node_select_source)
    g.add_node("approval", node_approval_interrupt)
    g.add_node("handle_approval", node_handle_approval)

    g.add_node("format", node_format_interrupt)
    g.add_node("handle_format", node_handle_format)

    g.add_node("web_search", node_web_search)
    g.add_node("compose_answer", node_compose_answer)

    g.add_node("generate_report_answer", node_generate_report_answer)
    g.add_node("save_report", node_save_report)

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


def run_agent(user_query: str):
    app = build_graph()

    state: AgentState = {
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

        "web_results": None,
        "guard_blocked": None,
        "final_answer": None,

        "source_domain": None,

        "report_answer": None,
        "report_paths": None,
        "report_basename": None,
    }

    while True:
        out = app.invoke(state)
        state.update(out)

        if state.get("approved") and state.get("source_id") and not state.get("_approval_announced"):
            sid = state["source_id"]
            state["source_domain"] = SOURCES[sid]["domain"]
            meta = SOURCES[sid]
            print(f"\n✔ Source confirmed: {sid} ({meta['domain']})")
            print("→ Searching now and will summarize.\n")
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


if __name__ == "__main__":
    q = input("User query> ").strip()
    run_agent(q)
