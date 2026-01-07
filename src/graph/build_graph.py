from langgraph.graph import StateGraph, START, END

from src.graph.state import AgentState
from src.graph.nodes import (
    node_intent_guard,
    node_select_source,
    node_approval_interrupt,
    node_handle_approval,
    node_web_search,
    node_compose_answer,
)
from src.graph.router import route_after_handle_approval, route_after_guard
from src.rag.allowlist import ALLOWED_SOURCES


# -----------------------------------------------------------------------------
# BUILD GRAPH: собираем "машину состояний"
# -----------------------------------------------------------------------------
def build_graph():
    g = StateGraph(AgentState)

    # --- nodes
    g.add_node("intent_guard", node_intent_guard)
    g.add_node("select_source", node_select_source)
    g.add_node("approval", node_approval_interrupt)
    g.add_node("handle_approval", node_handle_approval)
    g.add_node("web_search", node_web_search)
    g.add_node("compose_answer", node_compose_answer)

    # --- edges
    g.add_edge(START, "intent_guard")

    # intent guard: либо стоп, либо продолжаем
    g.add_conditional_edges(
        "intent_guard",
        route_after_guard,
        {"blocked": END, "ok": "select_source"},
    )

    # выбор источника -> подтверждение -> обработка ответа
    g.add_edge("select_source", "approval")
    g.add_edge("approval", "handle_approval")

    # после handle_approval:
    # - если confirmed -> идём искать в интернете
    # - если нет -> снова спрашиваем (loop)
    g.add_conditional_edges(
        "handle_approval",
        route_after_handle_approval,
        {"approved": "web_search", "revise": "approval"},
    )

    # поиск -> финальный ответ -> конец
    g.add_edge("web_search", "compose_answer")
    g.add_edge("compose_answer", END)

    return g.compile()



# -----------------------------------------------------------------------------
# RUNTIME LOOP: обработка interrupt (invoke -> ask user -> resume)
# -----------------------------------------------------------------------------
def run_agent_once(user_query: str):
    """
    Внешний runtime loop нужен, потому что interrupt требует "человека снаружи".

    Что здесь происходит:
    1) build_graph() -> получаем app
    2) создаём initial state (минимальный)
    3) запускаем app.invoke(state)
    4) если граф вернул "__interrupt__", то:
        - берём payload с вопросом
        - печатаем вопрос
        - читаем ввод пользователя (input)
        - кладём его в state["user_approval_raw"]
        - продолжаем цикл (снова invoke)
    5) если interrupt нет — значит дошли до END, печатаем final_answer

    Это "шлюз" между автоматикой графа и человеком.
    """
    app = build_graph()

    # Начальное состояние: все поля, которые ожидают узлы, должны существовать.
    state: AgentState = {
    "user_query": user_query,

    "candidate_source_id": None,
    "candidate_source_reason": None,
    "approval_question": None,

    "user_approval_raw": None,
    "approved": None,
    "source_id": None,

    "web_results": None,

    "guard_blocked": None,
    "final_answer": None,
    }


    while True:
        out = app.invoke(state)

        # сохраняем прогресс графа в state (важно и для interrupt, и для финала)
        state.update(out)

        # Красивое сообщение после подтверждения (без LLM), печатаем ровно 1 раз
        if state.get("approved") and state.get("source_id") and not state.get("_approval_announced"):
            sid = state["source_id"]
            meta = ALLOWED_SOURCES[sid]
            print(f"\n✔ Source confirmed: {sid} ({meta['domain']})")
            print("→ Searching now and will summarize.\n")
            state["_approval_announced"] = True

        # Если граф остановился на interrupt — он вернёт "__interrupt__"
        if "__interrupt__" in out:
            payload = out["__interrupt__"][0]
            data = payload.value if hasattr(payload, "value") else payload

            print("\n" + data.get("question", "Confirm? (y/n or type another source_id)"))
            state["user_approval_raw"] = input("> ").strip()
            continue

        # Если interrupt нет, то граф дошёл до END
        print(state.get("final_answer"))
        return state


# -----------------------------------------------------------------------------
# CLI entrypoint: запуск из терминала
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Это просто удобный способ проверить, что агент:
    # - читает user_query
    # - предлагает источник
    # - спрашивает подтверждение
    # - после подтверждения отвечает через LLM
    q = input("User query> ").strip()
    run_agent_once(q)
