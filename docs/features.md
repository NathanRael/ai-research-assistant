# airi — Features

## What It Is

**airi** is a multi-agent CLI personal assistant built with LangChain and LangGraph. It runs interactively in the terminal, dynamically routes user requests to specialist agents, and uses custom tools to search the live web, manage document knowledge, persist long-term memories and user facts, and execute real-world automations (like email).

All user data is strictly separated from application code and persisted in `~/.airi/` (or via `AI_ASSISTANT_HOME`), surviving restarts across terminal sessions.

---

## Core Features

### 1. Dynamic Multi-Agent Architecture & Routing

- **LangGraph Workflow**: Built on a `StateGraph` state machine (`START -> supervisor -> specialist | general -> END`).
- **Supervisor Routing**:
  - Uses function calling (`with_structured_output`) to classify user intent and route to the best specialist agent or general fallback.
  - Features a **Plain-Text Fallback Router** for LLMs that do not support tool/function calling, ensuring basic routing still functions.
  - Automatically routes direct greetings, small talk, identity questions, and general inquiries to a dedicated `General` node to avoid unnecessary tool invocations.
- **Context & Message Management**:
  - Automatically truncates message history (capped at 20 messages).
  - Truncates oversized tool outputs (capped at 4,000 characters) and chat messages (capped at 8,000 characters) to avoid overflowing context limits.
  - Automatically strips reasoning/thinking tags (e.g., `<think>...</think>`) from LLM outputs.

---

### 2. Specialist Agents & Tooling

#### Web Search Agent (`WebSearchAgent`)
- **Live Search**: Uses `LangSearch API` to retrieve up-to-date web results.
- **Context Integration**: Checks stored memory/user profile before searching to tailor queries to user preferences and tech stack.
- **Quality & Spam Filter**: Filters out non-Latin / garbled web results using character threshold analysis.
- **Resilience**: Employs exponential backoff retry mechanisms (`tenacity`) for rate limits (`429`) and network errors.

#### User Context & Document Agent (`UserContextAgent`)
- **Structured Profile (`profile.json`)**: Persists structured user attributes (`name`, `email`, `preferences` dict, `facts` list, `goals` list). Merges new entries accumulatively.
- **Free-Form Vector Memory (`user_memory` collection)**: Long-term memory backed by ChromaDB for storing learned user habits, communication styles, and workflows (`save_memory`, `retrieve_memory`).
- **Document RAG Engine**:
  - Supports loading and indexing **PDF**, **TXT**, and **Markdown (`.md`)** files (`/add <path>`).
  - Uses `RecursiveCharacterTextSplitter` for chunking and `OllamaEmbeddings` (`nomic-embed-text-v2-moe`) for vector embeddings.
  - Document RAG search (`search_documents`) with optional filtering by specific document name.
  - Listing indexed documents with chunk counts (`/list`).

#### Automation Agent (`AutomationAgent`)
- **SMTP Email Sending**: Preview drafts and send emails via SMTP (`send_email`).
- **IMAP Email Reading**: Preview and fetch recent emails from inbox via IMAP SSL (`read_emails`).
- **Safety Authorization Flow**: Mandatory two-step confirmation (`confirmed=True` / `authorized=True`) required before sending emails or reading the inbox.
- **Dry-Run Fallback**: Safe dry-run behavior when SMTP/IMAP credentials are not configured.

---

### 3. OpenCode Model Adapter (`ChatOpenCode`)

- **Custom BaseChatModel**: Purpose-built adapter for the OpenCode API gateway (`https://opencode.ai/zen/go/v1`).
- **Dual-Provider Auto-Detection**: Automatically tests and switches between OpenAI-compatible and Anthropic-compatible API formats based on model responsiveness.
- **Built-in Retries**: Retries transient server errors (`5xx`) and rate limits (`429`) with exponential backoff (`_retry_with_backoff`, up to 3 retries).
- **Runtime Model Switching (`/model`)**:
  - Dynamically fetches available models from the OpenCode API.
  - Validates model health with a test prompt (`Say OK`) before switching.
  - Highlights high-token models in red and requests explicit user confirmation.
  - Persists selected model to `~/.airi/.env` across terminal sessions.

---

### 4. Interactive CLI & Modern Terminal UI

- **Rich Terminal Styling**:
  - Gradient ASCII logo banner.
  - Formatted tables for help menus and model selection.
  - Status spinners for agent processing states.
  - Markdown-rendered assistant responses.
- **Advanced Interactive Prompt (`prompt_toolkit`)**:
  - Auto-completion for slash commands (`/add`, `/list`, `/debug`, `/setup`, `/status`, `/model`, `/clear`, `/help`).
  - Interactive `/debug` option autocompletion (`enable` / `disable`).
  - Persistent command history (`~/.airi/.history`).
  - Dynamic bottom toolbar showing the currently active LLM model and shortcut keys (`Escape`, `Ctrl+C`, `Ctrl+D`).
- **Interactive Setup Wizard (`/setup`)**:
  - Step-by-step wizard for configuring API keys, default model, Ollama URL, and SMTP/IMAP settings.
  - Validates OpenCode API key and Ollama server connection before saving settings.
- **CLI Flags**:
  - `airi --version`: Displays the current version.
  - `airi --config-dir`: Outputs the user data directory path.
  - `airi --help`: Displays CLI flag documentation.
- **Debug & Tracing (`/debug enable|disable`)**:
  - Live tracing of agent routing decisions and tool invocations with input/output snippets.
- **System Diagnostics (`/status`)**:
  - Shows active paths, model choice, API key status, Search API status, Ollama connection, SMTP configuration, document counts, and memory health.

---

### 5. Persistent User Data Directory

All user data lives outside the codebase in `~/.airi/` (configurable via `AI_ASSISTANT_HOME`):

```text
~/.airi/
├── .env           # Configuration and API key settings
├── profile.json   # Structured user facts, goals, and preferences
├── .history       # CLI command history
├── storage/       # Chroma vector store (user_memory and document chunks)
├── documents/     # Cached/stored document files
└── assistant.log  # Application debug logs
```

- **Data Migration**: Automatically migrates legacy configuration and profile files from `~/.ai-research-assistant/` if upgrading.

---

## How It Works

```
                        +----------------------+
                        |      User Input      |
                        +----------+-----------+
                                   |
                                   v
                        +----------------------+
                        |   SupervisorAgent    |
                        | (Structured/Fallback)|
                        +----------+-----------+
                                   |
         +-------------------------+-------------------------+-------------------------+
         |                         |                         |                         |
         v                         v                         v                         v
+------------------+      +------------------+      +------------------+      +------------------+
|  WebSearchAgent  |      | UserContextAgent |      | AutomationAgent  |      |  General Node    |
+--------+---------+      +--------+---------+      +--------+---------+      +--------+---------+
         |                         |                         |                         |
         v                         v                         v                         v
  [ search_web ]          [ save_memory ]           [ preview_email ]         (Direct LLM
                          [ retrieve_memory ]       [ send_email ]             Response)
                          [ search_documents ]      [ preview_read_emails ]
                          [ list_documents ]        [ read_emails ]
                          [ get_user_profile ]
                          [ update_user_profile ]
```

Each user request is evaluated by the **SupervisorAgent**, which dispatches execution to the appropriate specialist agent or returns a direct conversational response.
