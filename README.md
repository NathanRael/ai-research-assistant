# airi

A multi-agent CLI personal assistant powered by LangChain and LangGraph.
Search the web, manage documents, store memories, and automate tasks — all from your terminal.

## Features

- **Web Search** — Ask questions that need up-to-date information from the internet.
- **Document Memory** — Add PDF, TXT, or MD files and ask questions about their content.
- **User Context** — Store personal facts and preferences; the assistant remembers them across sessions.
- **Automation** — Send emails and read inbox with confirmation gates.
- **Model Switching** — Fetch available models, test them, and switch at runtime with `/model`.
- **Debug Mode** — See exactly which agent and tools are working in real time.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-research-assistant

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Build and Install as a Package

The project ships a console script (`airi`) via `pyproject.toml`, so it can be built into a distributable wheel and installed system-wide.

```bash
# Build the distribution (wheel + sdist)
uv build

# Or with pip/build
python -m build
```

This produces artifacts in `dist/` (e.g. `dist/airi-0.2.0-py3-none-any.whl`). Install them into any environment:

```bash
# Install the wheel
pip install dist/airi-0.2.0-py3-none-any.whl

# Or install directly from the source tree (editable, for development)
pip install -e .
```

To install globally and manage it like a standalone tool, use `uv`:

```bash
uv tool install .
```

Once installed via any of the above, run `airi` from anywhere. Update a `uv tool` install later with:

```bash
uv tool upgrade airi
```

## Quick Start

```bash
# Run the assistant
uv run python -m app.cli.main

# Or after pip install -e .
airi
```

CLI flags:

```bash
airi --version     # Display version
airi --config-dir  # Print user data directory path
```

For detailed installation and usage instructions, see the [Getting Started guide](docs/getting-started.md).

## Configuration

Run the interactive setup wizard from inside the assistant:

```
> /setup
```

Or create a `.env` file in your user data directory (`~/.airi/.env`):

```env
OPENCODE_API_KEY=your-api-key
OPENCODE_MODEL=kimi-k2.6
LANG_SEARCH_API=your-search-api-key
OLLAMA_URL=http://localhost:11434
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENCODE_API_KEY` | Yes | API key for the OpenCode provider |
| `OPENCODE_MODEL` | No | Model to use (default: `kimi-k2.6`) |
| `LANG_SEARCH_API` | No | API key for web search |
| `OLLAMA_URL` | No | Local Ollama URL for embeddings |
| `SMTP_HOST` | No | SMTP server for email automation |
| `SMTP_PORT` | No | SMTP port (default: 587) |
| `SMTP_USERNAME` | No | SMTP login username |
| `SMTP_PASSWORD` | No | SMTP login password |
| `SMTP_SENDER` | No | Email address to send from |
| `IMAP_HOST` | No | IMAP server for reading emails |
| `IMAP_PORT` | No | IMAP port (default: 993) |
| `IMAP_USERNAME` | No | IMAP login username |
| `IMAP_PASSWORD` | No | IMAP login password |
| `IMAP_FOLDER` | No | IMAP folder to read (default: `INBOX`) |

## CLI Commands

| Command | Description |
|---|---|
| `/add <path>` | Add and index a document (PDF, TXT, MD) |
| `/list` | List indexed documents |
| `/model` | View, test, and switch LLM models at runtime |
| `/debug enable\|disable` | Toggle tracing of routing and tool calls |
| `/setup` | Interactive configuration wizard |
| `/status` | Show current configuration and services |
| `/clear` | Clear screen (chat history preserved) |
| `/help` | Show available commands |
| `exit` | Quit the assistant |

## User Data

All user-specific data is stored separately from the application:

- **Config**: `~/.airi/.env`
- **Profile**: `~/.airi/profile.json` (structured user facts and preferences)
- **Storage**: `~/.airi/storage/` (Chroma vector store)
- **History**: `~/.airi/.history` (command history)
- **Documents**: `~/.airi/documents/` (optional)

Override the data directory with `AI_ASSISTANT_HOME`:

```bash
export AI_ASSISTANT_HOME=/path/to/custom/dir
```

## Architecture

```
You
  ↓
SupervisorAgent (routes to the right specialist, or responds directly for greetings/small talk)
  ├── WebSearchAgent      → search_web, save_memory, retrieve_memory, get_user_profile, update_user_profile
  ├── UserContextAgent    → save_memory, retrieve_memory, search_documents, list_documents, get_user_profile, update_user_profile
  └── AutomationAgent     → preview_email, send_email, preview_read_emails, read_emails, save_memory, retrieve_memory, get_user_profile, update_user_profile
```

Each specialist agent has access to memory and profile tools for context-aware responses. The supervisor uses structured output (function calling) to route requests, with a plain-text fallback for models that don't support it.

## Tech Stack

- **LLM**: OpenCode provider (via ChatOpenCode adapter)
- **Framework**: LangChain + LangGraph
- **Vector Store**: ChromaDB (persistent, local)
- **Embeddings**: Ollama (nomic-embed-text-v2-moe)
- **Web Search**: LangSearch API

## Project Structure

```
app/
├── agents/
│   ├── base_agent.py           # Base class wrapping LLM + tools + prompt
│   ├── supervisor_agent.py     # Routes requests to specialist agents
│   ├── web_search_agent.py     # Live internet search agent
│   ├── user_context_agent.py   # Memory and document agent
│   ├── automation_agent.py     # Email automation agent
│   └── prompts.py              # Shared prompt fragments
├── cli/
│   ├── main.py                 # CLI entry point and REPL
│   ├── setup.py                # Interactive configuration wizard
│   ├── ui.py                   # Rich console output helpers
│   ├── prompt.py               # Prompt toolkit session
│   └── input.py                # Input handling
├── graph/
│   └── workflow.py             # LangGraph supervisor workflow
├── services/
│   ├── document_service.py     # Document indexing and search
│   ├── document_loader.py      # PDF/TXT/MD loading and chunking
│   ├── memory_service.py       # Persistent user memory (Chroma)
│   ├── user_profile_service.py # Structured user profile (JSON)
│   ├── email_service.py        # SMTP send and IMAP read
│   ├── vector_store.py         # ChromaDB wrapper
│   └── embedding.py            # Ollama embedding function
├── tools/
│   ├── search_tools.py         # search_web tool
│   ├── memory_tools.py         # save_memory, retrieve_memory tools
│   ├── document_tools.py       # search_documents, list_documents tools
│   ├── user_profile_tools.py   # get_user_profile, update_user_profile tools
│   ├── automation_tools.py     # Email tools with safety gates
│   └── web_search_client.py    # LangSearch API client with retry logic
├── chat_opencode.py            # LLM adapter for OpenCode provider
├── config.py                   # Pydantic settings management
└── user_data.py                # User data directory (~/.airi) management
```

## Development

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .
```
