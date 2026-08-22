# airi

A multi-agent CLI personal assistant powered by LangChain and LangGraph.
Search the web, manage documents, store memories, and automate tasks — all from your terminal.

## Features

- **Web Search** — Ask questions that need up-to-date information from the internet.
- **Document Memory** — Add PDF, TXT, or MD files and ask questions about their content.
- **User Context** — Store personal facts and preferences; the assistant remembers them across sessions.
- **Automation** — Send emails and trigger external actions.
- **Debug Mode** — See exactly which agent and tools are working in real time.
- **Rich Markdown Output** — Formatted responses with code blocks, tables, and styling.

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

## Quick Start

```bash
# Run the assistant
uv run python -m app.cli.main

# Or after pip install -e .
airi
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
| `/debug enable\|disable` | Toggle tracing of routing and tool calls |
| `/setup` | Interactive configuration wizard |
| `/status` | Show current configuration and services |
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
SupervisorAgent (routes to the right specialist)
  ├── WebSearchAgent      → search_web tool
  ├── UserContextAgent    → save_memory, retrieve_memory, search_documents, list_documents, get_user_profile, update_user_profile
  └── AutomationAgent     → preview_email, send_email, preview_read_emails, read_emails
```

Each agent uses its own tools and system prompt. The supervisor decides which agent handles each request.

## Tech Stack

- **LLM**: OpenCode provider (via ChatOpenCode adapter)
- **Framework**: LangChain + LangGraph
- **Vector Store**: ChromaDB (persistent, local)
- **Embeddings**: Ollama (nomic-embed-text-v2-moe)
- **Web Search**: LangSearch API

## Project Structure

```
app/
├── agents/           # Specialist agents and supervisor
├── cli/              # CLI entry point and setup wizard
├── graph/            # LangGraph workflow
├── services/         # Business logic (memory, documents, email, embeddings, vector store)
├── tools/            # LangChain tool factories
├── chat_opencode.py  # LLM adapter
├── config.py         # Settings
└── user_data.py      # User data directory management
```

## Development

```bash
# Run tests
uv run python -m pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .
```
