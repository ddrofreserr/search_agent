# Search Agent

# Search Agent (EN)

**Search Agent** is a learning-oriented agentic system built with **LangGraph**, designed to explore how modern LLM-based agents can:

* reason about user intent,
* select appropriate information sources,
* involve a human-in-the-loop for confirmation,
* search the web in a controlled way,
* and synthesize grounded answers based on real data.

The project is intentionally modular and incremental: it focuses first on **agent architecture and control flow**, with RAG and report generation planned for later stages.

---

## Project goals

The original task behind this project was to build an agent that:

1. Communicates with a user in natural language
2. Selects a suitable information source from an allowlist
3. Requests human confirmation before using that source
4. Searches the internet **only within approved sources**
5. Summarizes findings in a grounded, explainable way
6. (Planned) Generates structured reports (Markdown / HTML)
7. (Planned) Uses RAG for retrieval and grounding

---

## Current status

### ✅ Implemented

* Agent orchestration using **LangGraph**
* Explicit **human-in-the-loop** approval
* LLM-based source selection and replanning
* Intent / scope guard (refusal for out-of-scope queries)
* Controlled web search via allowlisted domains
* Result synthesis with quotes and links
* Clean modular project structure

### 🚧 Not yet implemented

* RAG (vector store, embeddings, chunking)
* Report generation (Markdown / HTML files)
* Persistent memory / long-term context
* Multilingual support (currently English-only)

---

## Supported sources (allowlist)

At the moment, the agent can reason about and search within:

* **Wikipedia** — background, definitions, overviews
* **GitHub** — code repositories and implementations
* **Reddit** — discussions and community insights (experimental)
* **arXiv** — research papers and preprints

The allowlist is explicitly defined and enforced.

---

## Project structure

```
src/
├── graph/
│   ├── build_graph.py   # LangGraph assembly + runtime loop
│   ├── nodes.py         # Agent nodes (LLM logic, tools, decisions)
│   ├── router.py        # Conditional routing logic
│   ├── state.py         # AgentState definition
│   └── ollama.py        # Local LLM (Ollama) wrapper
│
├── rag/
│   └── allowlist.py     # Allowed sources configuration
│
├── web/
│   └── tools.py         # Web search & page parsing utilities
│
├── report/              # (planned) report generation
│
requirements.txt
```

### Responsibilities by module

* **graph/state.py**
  Defines the shared `AgentState` flowing through the graph.

* **graph/nodes.py**
  Contains all agent steps:

  * intent guard
  * source selection
  * approval handling
  * web search
  * answer synthesis

* **graph/router.py**
  Small, explicit routing functions for conditional branches.

* **graph/build_graph.py**
  Assembles the LangGraph and provides a CLI runtime loop.

* **rag/allowlist.py**
  Central policy file describing allowed sources.

* **web/tools.py**
  Low-level web tools (search, fetch, text extraction).

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
Select Source (LLM)
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
Web Search (tools, allowlisted)
  │
  ▼
Compose Final Answer (LLM)
  │
  ▼
END
```

### Node roles (brief)

* **Intent guard**
  Ensures the query fits the agent’s purpose.

* **Select source**
  LLM chooses the most appropriate source from the allowlist.

* **Approval interrupt**
  Pauses execution and asks the user to confirm or reject.

* **Handle approval**
  Parses user input, approves or replans using LLM reasoning.

* **Web search**
  Executes a domain-restricted internet search.

* **Compose answer**
  Synthesizes findings into a grounded response with citations.

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama

Make sure Ollama is installed and a model is available, for example:

```bash
ollama pull qwen2.5:3b
```

### 3. Run the agent

From the project root:

```bash
python -m src.graph.build_graph
```

You will be prompted for a query and can interactively approve or reject suggested sources.

---

## Usage notes

* The agent currently expects **English input**.
* All web searches are restricted to approved domains.
* Human confirmation is required before any search is performed.
* Output quality depends on both search results and the local LLM.

---

## Future improvements

Planned next steps include:

* Integrating **RAG** with a vector database (e.g. Qdrant)
* Generating structured **Markdown / HTML reports**
* Better result ranking and deduplication
* Multilingual support (including Russian)
* Automated tests for agent flows

---

## Disclaimer

This project is primarily educational and experimental.
It is not intended for unrestricted web scraping or production deployment without further safeguards.


---


# Search Agent (RU)

**Search Agent** — это учебный агентный проект на базе **LangGraph**, цель которого — на практике разобраться, как устроены современные LLM-агенты:

* как они анализируют пользовательский запрос,
* выбирают источник информации,
* используют human-in-the-loop,
* выполняют ограниченный поиск в интернете,
* и формируют обоснованный ответ на основе реальных данных.

Проект спроектирован как **пошаговый и расширяемый**: основной фокус сейчас — архитектура агента и контроль потока, а RAG и генерация отчетов будут добавлены позже.

---

## Цели проекта

Исходная задача проекта:

1. Реализовать диалог с пользователем
2. Выбирать источник информации из списка разрешённых
3. Запрашивать подтверждение пользователя перед поиском
4. Искать информацию в интернете **строго в рамках выбранного источника**
5. Формировать обобщённый, объяснимый ответ
6. (В планах) Генерировать отчёты (Markdown / HTML)
7. (В планах) Использовать RAG для извлечения релевантного контекста

---

## Текущий статус

### ✅ Реализовано

* Оркестрация агента с помощью **LangGraph**
* Явный **human-in-the-loop** (подтверждение источника)
* Выбор источника и перепланирование с помощью LLM
* Проверка запроса на соответствие назначению агента (intent guard)
* Контролируемый поиск в интернете через allowlist
* Синтез ответа с цитатами и ссылками
* Модульная и расширяемая структура проекта

### 🚧 Пока не реализовано

* RAG (векторное хранилище, эмбеддинги, чанкинг)
* Генерация отчётов (Markdown / HTML)
* Долгосрочная память агента
* Поддержка нескольких языков (пока только английский)

---

## Поддерживаемые источники

В текущей версии агент умеет работать со следующими источниками:

* **Wikipedia** — определения, базовый контекст
* **GitHub** — репозитории с кодом и реализациями
* **Reddit** — обсуждения и практические мнения (экспериментально)
* **arXiv** — научные статьи и препринты

Все источники явно перечислены и проверяются.

---

## Структура проекта

```
src/
├── graph/
│   ├── build_graph.py   # Сборка графа и runtime
│   ├── nodes.py         # Узлы агента (логика, LLM, инструменты)
│   ├── router.py        # Условные переходы (routing)
│   ├── state.py         # Описание AgentState
│   └── ollama.py        # Обертка над локальной LLM (Ollama)
│
├── rag/
│   └── allowlist.py     # Список разрешённых источников
│
├── web/
│   └── tools.py         # Поиск в интернете и парсинг страниц
│
├── report/              # (планируется) генерация отчётов
│
requirements.txt
```

---

## Архитектура агента (пайплайн)

```
START
  │
  ▼
Проверка интента (intent guard)
  │
  ├── отклонено ──► END (отказ)
  │
  ▼
Выбор источника (LLM)
  │
  ▼
Запрос подтверждения у пользователя
  │
  ▼
Обработка ответа пользователя
  │
  ├── пересмотр ──► повторный запрос подтверждения
  │
  ▼
Поиск в интернете (строго по источнику)
  │
  ▼
Формирование финального ответа (LLM)
  │
  ▼
END
```

---

## Роли узлов

* **Проверка интента**
  Определяет, соответствует ли запрос назначению агента.

* **Выбор источника**
  LLM выбирает наиболее подходящий источник из allowlist.

* **Подтверждение пользователя**
  Останавливает выполнение и ждёт решения человека.

* **Обработка подтверждения**
  Принимает решение или перепланирует источник.

* **Поиск в интернете**
  Выполняет реальный поиск с ограничением по домену.

* **Финальный ответ**
  Синтезирует найденную информацию в обоснованный ответ.

---

## Запуск проекта

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Установка модели для Ollama

```bash
ollama pull qwen2.5:3b
```

### Запуск агента

```bash
python -m src.graph.build_graph
```

После запуска агент запросит вопрос и предложит выбрать источник.

---

## Примечания по использованию

* Агент сейчас рассчитан на **англоязычные запросы**
* Все поисковые действия требуют подтверждения пользователя
* Поиск всегда ограничен разрешёнными источниками
* Проект ориентирован на обучение и эксперименты

---

## Планы по развитию

* Интеграция RAG (Qdrant / FAISS)
* Генерация отчётов (Markdown / HTML)
* Улучшение ранжирования результатов
* Поддержка русского языка
* Автоматические тесты для агентных сценариев

---

## Дисклеймер

Проект носит учебный и исследовательский характер
и не предназначен для использования без дополнительных ограничений и проверок.

