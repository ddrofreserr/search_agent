# Search Agent

---

# Search Agent (EN)

**Search Agent** is a learning-oriented agentic system built with **LangGraph**, designed to explore how modern LLM-based agents can:

* reason about user intent,
* select appropriate information sources,
* involve a human-in-the-loop for confirmation,
* search the web in a controlled way,
* use RAG for source selection,
* and generate grounded, structured reports based on real data.

The project is intentionally modular and explicit: it focuses on **agent architecture, state management, and control flow**, rather than opaque end-to-end pipelines.

---

## Project goals

The original task behind this project was to build an agent that:

1. Communicates with a user in natural language
2. Selects a suitable information source from an allowlist
3. Requests human confirmation before using that source
4. Searches the internet **only within approved sources**
5. Uses RAG to reason about source relevance
6. Synthesizes findings in a grounded, explainable way
7. Generates structured reports (Markdown / HTML)

---

## Current status

### ✅ Implemented

* Agent orchestration using **LangGraph**
* Explicit **human-in-the-loop** approval
* LLM-based source selection with replanning
* **RAG-based hybrid retrieval** (dense + BM25) for source selection
* Intent / scope guard (refusal for out-of-scope queries)
* Controlled web search via allowlisted domains
* Structured evidence collection with quotes and links
* **Report generation (Markdown and HTML files)**
* Deterministic agent flow with explicit state
* Clean modular project structure

### ❌ Not implemented (by design)

* Persistent memory / long-term user context
* Multilingual support
  *(the system is designed to work correctly with English queries only)*

---

## Supported sources (allowlist)

The agent can reason about and search within the following explicitly allowed sources:

* **Wikipedia** — background, definitions, overviews
* **GitHub** — code repositories and implementations
* **Reddit** — discussions and community insights (experimental)
* **arXiv** — research papers and preprints

All searches are strictly restricted to these domains.

---

## Project structure

```
src/
├── graph/
│   ├── build_graph.py        # LangGraph assembly + runtime loop
│   ├── nodes.py              # Agent nodes (LLM logic, tools, decisions)
│   ├── router.py             # Conditional routing logic
│   ├── state.py              # AgentState definition
│   └── ollama.py             # Local LLM (Ollama) wrapper
│
├── rag/
│   ├── qdrant_sources.py     # RAG-based source selection (dense + BM25)
│   └── seed_sources_qdrant.py
│
├── web/
│   └── tools.py              # Web search & page parsing utilities
│
├── reports/
│   ├── generate_report.py    # Report rendering & file generation
│   └── reports/              # Generated Markdown / HTML reports
│
requirements.txt
```

### Responsibilities by module

* **graph/state.py**
  Defines the explicit `AgentState` passed through the graph.

* **graph/nodes.py**
  Contains all agent steps:

  * intent guard
  * source selection
  * approval handling
  * web search
  * report content generation
  * final user-facing response

* **graph/router.py**
  Small, explicit routing functions for conditional branches.

* **graph/build_graph.py**
  Assembles the LangGraph and provides a CLI runtime loop.

* **rag/qdrant_sources.py**
  Implements RAG-based hybrid retrieval for source selection.

* **web/tools.py**
  Low-level web tools (search, fetch, text extraction).

* **reports/generate_report.py**
  Deterministic generation of Markdown and HTML reports.

---

## Agent pipeline (high-level)

```
START
  │
  ▼
Intent Guard
  │
  ├── blocked ──► END (refusal)
  │
  ▼
Select Source (RAG + LLM)
  │
  ▼
Ask User for Approval (interrupt)
  │
  ▼
Handle Approval
  │
  ├── revise ──► Ask again (loop)
  │
  ▼
Web Search (allowlisted tools)
  │
  ▼
Generate Report Content (LLM)
  │
  ▼
Save Report (Markdown / HTML)
  │
  ▼
Compose Final Message
  │
  ▼
END
```

---

## Node roles (brief)

* **Intent guard**
  Ensures the query fits the agent’s scope.

* **Select source**
  Uses RAG-based retrieval to select the most relevant source.

* **Approval interrupt**
  Pauses execution and asks the user to confirm or revise.

* **Handle approval**
  Parses user input and either approves or replans.

* **Web search**
  Executes a domain-restricted internet search.

* **Generate report content**
  Uses the LLM to synthesize findings with citations.

* **Save report**
  Writes deterministic Markdown and HTML files.

* **Compose final message**
  Returns a short message pointing to the generated report.

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama

```bash
ollama pull qwen2.5:3b
```

### 3. Run the agent

```bash
python -m src.graph.build_graph
```

You will be prompted for a query and asked to approve the suggested source.

---

## Usage notes

* The agent expects **English-language queries**.
* All searches are restricted to approved domains.
* Human confirmation is required before any web search.
* The main output of the system is a **generated report**, not a chat-style answer.

---

## 📄 Report output (EN)

The primary output of the system is a **generated report**, rather than a long chat-style answer.

Each report is saved in both **Markdown** and **HTML** formats and contains:

* the original user query,
* the selected source and the rationale for choosing it,
* a structured **Evidence** section with:

  * search results,
  * source URLs,
  * extracted short quotes,
* a synthesized **Answer** section generated by the LLM, grounded in the collected evidence.

Reports are written to the `reports/reports/` directory and use **deterministic, sequential filenames**
(e.g. `0001__find-papers-about-rotary-positional-embeddings.html`),
making them easy to browse, diff, and version-control.

An example of a generated report is included in the repository to demonstrate the expected structure and level of detail.

---

## Disclaimer

This project is educational and experimental.
It is not intended for production use without additional safeguards.

---

---

# Search Agent (RU)

**Search Agent** — это учебный агентный проект на базе **LangGraph**, предназначенный для практического изучения того,
как проектируются и реализуются **детерминированные LLM-агенты с контролируемым доступом к данным**.

Проект демонстрирует полный агентный пайплайн:

* анализ пользовательского запроса,
* выбор источника информации,
* использование human-in-the-loop,
* ограниченный веб-поиск,
* применение RAG для выбора источника,
* генерацию структурированных отчётов (Markdown и HTML).

Основной фокус проекта — **архитектура агента, управление состоянием и контроль потока**, а не end-to-end “магия”.

---

## Цели проекта

Исходная задача проекта:

1. Реализовать диалог с пользователем
2. Выбирать источник информации из списка разрешённых
3. Запрашивать подтверждение пользователя перед поиском
4. Искать информацию в интернете **строго в рамках выбранного источника**
5. Использовать RAG для выбора источника
6. Формировать обоснованный, объяснимый результат
7. Генерировать структурированные отчёты (Markdown / HTML)

---

## Текущий статус

### ✅ Реализовано

* Оркестрация агента с помощью **LangGraph**
* Явный **human-in-the-loop**
* Выбор источника и перепланирование с помощью LLM
* **RAG (dense + BM25) для выбора источника**
* Проверка интента и ограничение области задач
* Контролируемый веб-поиск через allowlist
* Сбор доказательств (цитаты, ссылки)
* **Генерация отчётов в Markdown и HTML**
* Явное управление состоянием агента
* Модульная структура проекта

### ❌ Не реализовано (осознанно)

* Долгосрочная память агента
* Многоязычная поддержка
  *(система корректно работает только с англоязычными запросами)*

---

## Поддерживаемые источники

Агент умеет работать со следующими источниками:

* **Wikipedia** — определения и базовый контекст
* **GitHub** — репозитории с кодом и реализациями
* **Reddit** — обсуждения и практические мнения (экспериментально)
* **arXiv** — научные статьи и препринты

Все источники явно перечислены и проверяются.

---

## Структура проекта

```
src/
├── graph/
│   ├── build_graph.py        # Сборка LangGraph и runtime
│   ├── nodes.py              # Узлы агента
│   ├── router.py             # Условные переходы
│   ├── state.py              # AgentState
│   └── ollama.py             # Обёртка над Ollama
│
├── rag/
│   ├── qdrant_sources.py     # RAG для выбора источников
│   └── seed_sources_qdrant.py
│
├── web/
│   └── tools.py              # Поиск и парсинг страниц
│
├── reports/
│   ├── generate_report.py    # Генерация отчётов
│   └── reports/              # Готовые HTML / Markdown
│
requirements.txt
```

---

## Архитектура агента (пайплайн)

```
START
  │
  ▼
Проверка интента
  │
  ├── отклонено ──► END
  │
  ▼
Выбор источника (RAG + LLM)
  │
  ▼
Подтверждение пользователя
  │
  ▼
Обработка подтверждения
  │
  ├── пересмотр ──► повтор
  │
  ▼
Поиск в интернете
  │
  ▼
Формирование текста отчёта (LLM)
  │
  ▼
Сохранение отчёта (MD / HTML)
  │
  ▼
Короткий ответ пользователю
  │
  ▼
END
```

---

## Примечания по использованию

* Агент рассчитан **только на англоязычные запросы**
* Все поисковые действия требуют подтверждения
* Поиск всегда ограничен allowlist-источниками
* Основной результат работы — **файлы отчёта**

Отличная мысль — это как раз закрывает “осязаемость” проекта. Ниже даю **готовые абзацы**, **отдельно для EN и RU**, в стиле уже существующего README.
Ты можешь вставить их **в раздел Usage notes / Примечания по использованию** или сразу после описания отчётов — они автономные.

---

## 📄 Формат отчётов (RU)

Каждый отчёт сохраняется сразу в двух форматах — **Markdown** и **HTML** — и включает:

* исходный пользовательский запрос,
* выбранный источник и обоснование выбора,
* структурированный раздел **Evidence**, содержащий:

  * результаты поиска,
  * ссылки на источники,
  * короткие извлечённые цитаты,
* итоговый раздел **Answer**, сформированный LLM и основанный на собранных данных.

Отчёты сохраняются в директорию `reports/reports/` и имеют **детерминированные порядковые имена файлов**
(например, `0001__find-papers-about-rotary-positional-embeddings.html`),
что упрощает навигацию, сравнение и хранение результатов.

В репозитории приведён пример сгенерированного отчёта, демонстрирующий ожидаемую структуру и формат вывода.

---

## Дисклеймер

Проект носит учебный и исследовательский характер
и не предназначен для промышленного использования.
