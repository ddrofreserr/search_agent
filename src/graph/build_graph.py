
from __future__ import annotations

import subprocess
from typing import TypedDict, Optional, Dict, Any, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt


# -----------------------------
# 1) POLICY / ALLOWLIST
# -----------------------------
ALLOWED_SOURCES: Dict[str, Dict[str, str]] = {
    "wikipedia": {
        "domain": "wikipedia.org",
        "desc": "General-purpose encyclopedia: good for definitions, overviews, background context.",
    },
    "github": {
        "domain": "github.com",
        "desc": "Code repositories: good for implementations, libraries, issues, and examples.",
    },
    "reddit": {
        "domain": "reddit.com",
        "desc": "Community discussions: good for practical tips, opinions, debugging threads (verify claims).",
    },
    "arxiv": {
        "domain": "arxiv.org",
        "desc": "Scientific preprints: good for papers, methods, and research context.",
    },
}


# -----------------------------
# STATE (память графа)
# -----------------------------
class AgentState(TypedDict):
    """
    AgentState = память агента внутри LangGraph.

    LangGraph запускает узлы последовательно.
    Каждый узел:
      - читает state (словарь)
      - возвращает dict с обновлениями
    LangGraph сам "мерджит" эти обновления обратно в общий state.

    Важно:
    - state — это не обязательно "память диалога" как messages.
      Это просто общая структура данных, которую ты выбрал хранить между шагами.
    """

    # Вход пользователя (в этом MVP — один запрос).
    user_query: str

    # То, что агент "предложил" как источник.
    candidate_source_id: Optional[str]        # например "wikipedia"
    candidate_source_reason: Optional[str]    # короткое объяснение "почему"
    approval_question: Optional[str]

    # Human-in-the-loop поля:
    user_approval_raw: Optional[str]          # что ввёл человек (y / github / etc.)
    approved: Optional[bool]                  # True/False (после обработки)
    source_id: Optional[str]                  # подтверждённый источник (итог)

    # Итоговый текст, который мы покажем в терминале.
    final_answer: Optional[str]


# -----------------------------------------------------------------------------
# TOOL: LLM CALL (Ollama
# -----------------------------------------------------------------------------
def call_ollama(prompt: str, model: str = "qwen2.5:3b") -> str:
    """
    Это внешний "инструмент" (tool) — то есть действие вне графа:
    мы обращаемся к системе Ollama, чтобы получить ответ от LLM.

    Важная идея:
    - В agent-системах tools обычно выносят наружу:
      web, базы, файлы, shell-команды, модели и т.п.
    - Здесь наш tool — просто "сгенерировать текст LLM".

    Как работает конкретно:
    - subprocess.run запускает команду:
        ollama run <model> <prompt>
    - stdout содержит ответ модели (мы его возвращаем)

    Почему так просто:
    - ты уже запускал Ollama из терминала, значит этот способ самый прямой и понятный.
    """
    r = subprocess.run(["ollama", "run", model, prompt], capture_output=True, text=True)
    return (r.stdout or "").strip()


# -----------------------------------------------------------------------------
# NODE 1: выбор источника (упрощённая эвристика)
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
# NODE 2: human-in-the-loop interrupt (спросить подтверждение)
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
# NODE 3: обработать ответ человека
# -----------------------------------------------------------------------------
def node_handle_approval(state: AgentState) -> Dict[str, Any]:
    raw = (state.get("user_approval_raw") or "").strip()
    candidate = state.get("candidate_source_id") or "wikipedia"

    # 1) быстрый путь: пользователь подтвердил -> идём к финалу
    if raw.lower() in {"y", "yes", "да", "ok", "ага"}:
        print("[handle_approval] approved")
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

        "user_approval_raw": None,  # <-- КРИТИЧНО: очистили, чтобы approval снова спросил
    }


# -----------------------------------------------------------------------------
# NODE 4: сгенерировать финальный ответ (LLM)
# -----------------------------------------------------------------------------
# def node_compose_answer(state: AgentState) -> Dict[str, Any]:
#     """
#     Generate a user-facing response after source confirmation.

#     Сейчас:
#     - интернет ещё не подключён
#     - ответ объясняет, ЧТО будет сделано и ПОЧЕМУ выбран источник
#     """

#     sid = state.get("source_id") or state.get("candidate_source_id")
#     meta = ALLOWED_SOURCES[sid]

#     prompt = f"""
# You are an information research assistant.

# Context:
# - The user asked a question.
# - You selected an information source based on the query.
# - The user confirmed this source.
# - Live web search is NOT implemented yet.

# Your task:
# - Explain what kind of information would be searched.
# - Explain why the chosen source is appropriate.
# - Be clear, concise, and professional.

# User query:
# {state['user_query']}

# Chosen source:
# - ID: {sid}
# - Description: {meta['desc']}
# - Domain: {meta['domain']}

# Write a helpful response to the user.
# """.strip()

#     return {
#         "final_answer": call_ollama(prompt, model="qwen2.5:3b")
#     }


def node_compose_answer(state: AgentState) -> Dict[str, Any]:
    sid = state.get("source_id") or state.get("candidate_source_id") or "wikipedia"
    domain = ALLOWED_SOURCES[sid]["domain"]
    q = state["user_query"]

    prompt = f"""
You are an information research agent.

The user has confirmed the source.
Live web search is not implemented yet, so do not claim you actually searched.

Write a short message that:
1) acknowledges confirmation,
2) says you will search in the chosen source,
3) repeats the query in your own words,
4) states the next step ("I'll search and summarize").

Chosen source: {sid} ({domain})
User query: {q}
""".strip()

    return {"final_answer": call_ollama(prompt, model="qwen2.5:3b")}


# -----------------------------------------------------------------------------
# ROUTER: решает ветвление "куда идти дальше"
# -----------------------------------------------------------------------------
def route_after_handle_approval(state: AgentState) -> Literal["approved", "revise"]:
    """
    Router = часть планировщика/контроллера.

    Чем является:
    - Это НЕ узел, а функция, которая возвращает "ключ маршрута".
    - LangGraph использует этот ключ, чтобы выбрать нужное ребро (edge).

    Как работает:
    - если approved=True -> идём по ветке "approved" (к compose_answer)
    - иначе -> "revise" (возвращаемся к approval и спрашиваем заново)
    """
    if state.get("approved"):
        return "approved"
    return "revise"


# -----------------------------------------------------------------------------
# BUILD GRAPH: собираем "машину состояний"
# -----------------------------------------------------------------------------
def build_graph():
    """
    Здесь мы описываем граф (план выполнения).

    Важно понимать:
    - Узлы (nodes) — это функции, которые возвращают обновления state
    - Рёбра (edges) — порядок выполнения
    - Conditional edges — ветвление

    Поток в этом MVP:

      START
        -> select_source
        -> approval (interrupt)
        -> handle_approval
            -> (approved?) -> compose_answer -> END
            -> (not approved) -> approval (loop)

    То есть:
    - мы выбираем источник
    - просим подтверждение
    - если не подтвердили, переспрашиваем (возможен смена источника)
    - если подтвердили, генерируем ответ и завершаем
    """
    g = StateGraph(AgentState)

    # add_node(name, function)
    # name — это метка узла в графе
    g.add_node("select_source", node_select_source)
    g.add_node("approval", node_approval_interrupt)
    g.add_node("handle_approval", node_handle_approval)
    g.add_node("compose_answer", node_compose_answer)

    # линейные рёбра
    g.add_edge(START, "select_source")
    g.add_edge("select_source", "approval")
    g.add_edge("approval", "handle_approval")

    # ветвление после handle_approval
    g.add_conditional_edges(
        "handle_approval",
        route_after_handle_approval,
        {"approved": "compose_answer", "revise": "approval"},
    )

    # финал
    g.add_edge("compose_answer", END)

    # compile() превращает описание графа в runnable "app"
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
        "user_approval_raw": None,
        "approved": None,
        "source_id": None,
        "final_answer": None,
    }

    while True:
        out = app.invoke(state)

        # Если graph остановился на interrupt — он вернёт "__interrupt__"
        if "__interrupt__" in out:
            payload = out["__interrupt__"][0]
            # В некоторых версиях langgraph полезные данные лежат в payload.value
            data = payload.value if hasattr(payload, "value") else payload

            print("\n" + data["question"])

            # сохраняем промежуточные апдейты state из out
            state.update(out)

            # пишем ответ человека в state (это прочитает node_handle_approval)
            state["user_approval_raw"] = input("> ").strip()
            continue

        # Если interrupt нет, то граф дошёл до END
        print("\n=== FINAL ANSWER ===\n")
        print(out.get("final_answer"))
        return out


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
