# Getting Started

This guide explains how to run the AI Research Assistant during development, install it as a proper CLI command, configure it, and use it day-to-day.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Run during development](#run-during-development)
3. [Install as a CLI command](#install-as-a-cli-command)
4. [First-time configuration](#first-time-configuration)
5. [Daily usage](#daily-usage)
6. [CLI flags](#cli-flags)
7. [Interactive commands](#interactive-commands)
8. [Where your data lives](#where-your-data-lives)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.14+**
- **Ollama** running locally with an embedding model, e.g.:
  ```bash
  ollama pull nomic-embed-text-v2-moe
  ```
- An **OpenCode API key** from [opencode.ai](https://opencode.ai)
- (Optional) A **Lang Search API key** for web search

---

## Run during development

From the project root:

```bash
# With uv (recommended)
uv run python -m app.cli.main

# Or with pip in an active virtual environment
python -m app.cli.main
```

This starts the interactive REPL with the `You >` prompt.

---

## Install as a CLI command

The project defines a console script named `airesearch` in `pyproject.toml`:

```toml
[project.scripts]
airesearch = "app.cli.main:main"
```

Choose one of the installation methods below.

### Option A: Editable install (recommended for development)

```bash
# With uv
uv pip install -e .

# Or with pip
pip install -e .
```

Then run from anywhere:

```bash
airesearch
```

### Option B: Build and install the wheel

```bash
# Build
uv build

# Or with pip/build
python -m build

# Install the resulting wheel
pip install dist/ai_research_assistant-0.2.0-py3-none-any.whl
```

Then run:

```bash
airesearch
```

### Option C: Install with uv tools (cleanest for end users)

```bash
uv tool install .
```

Then run globally:

```bash
airesearch
```

Update later with:

```bash
uv tool upgrade ai-research-assistant
```

---

## First-time configuration

When you run `airesearch` for the first time without a config file, it warns you and suggests running `/setup`.

Inside the assistant:

```text
> /setup
```

The wizard will prompt you for:

| Setting | Required | Purpose |
|---|---|---|
| `OPENCODE_API_KEY` | Yes | Your OpenCode provider API key |
| `OPENCODE_MODEL` | No | Model to use, e.g. `kimi-k2.6` |
| `LANG_SEARCH_API` | No | Web search API key |
| `OLLAMA_URL` | No | Local Ollama endpoint, default `http://localhost:11434` |
| `SMTP_*` settings | No | Optional email automation |

The wizard tests the API key and Ollama connection before saving.

You can also create the `.env` file manually in your user data directory.

---

## Daily usage

Start the assistant:

```bash
airesearch
```

Ask questions naturally:

```text
You > What is the latest news about AI?
You > What do you remember about me?
You > Send a summary to bob@example.com
```

Add documents:

```text
You > /add ./my-notes.pdf
You > /list
You > What does my-notes.pdf say about architecture?
```

Toggle debug mode to see routing and tool calls:

```text
You > /debug enable
```

---

## CLI flags

Because `airesearch` is a proper console script, it supports flags before entering the REPL:

```bash
# Show help
airesearch --help

# Show version
airesearch --version

# Print the user data directory
airesearch --config-dir
```

---

## Interactive commands

| Command | Description |
|---|---|
| `/add <path>` | Add and index a document (PDF, TXT, MD) |
| `/list` | List indexed documents |
| `/setup` | Interactive configuration wizard |
| `/status` | Show current configuration and service state |
| `/debug enable` | Show routing and tool activity |
| `/debug disable` | Hide routing and tool activity |
| `/help` | Show available commands |
| `exit` | Quit the assistant |

---

## Where your data lives

All user-specific data is stored separately from the application code:

| Data | Default location |
|---|---|
| Configuration | `~/.ai-research-assistant/.env` |
| Vector store | `~/.ai-research-assistant/storage/` |
| Command history | `~/.ai-research-assistant/.history` |
| Documents cache | `~/.ai-research-assistant/documents/` |

Override the base directory:

```bash
export AI_ASSISTANT_HOME=/path/to/custom/dir
airesearch
```

On Windows, the default is `%USERPROFILE%\.ai-research-assistant\`.

---

## Troubleshooting

### `No API key configured`

Run `/setup` inside the assistant or create `~/.ai-research-assistant/.env` with at least:

```env
OPENCODE_API_KEY=your-api-key
```

### `Cannot connect to Ollama`

Make sure Ollama is running:

```bash
ollama serve
```

And the required embedding model is pulled:

```bash
ollama pull nomic-embed-text-v2-moe
```

### Web search returns no results

Web search requires `LANG_SEARCH_API` to be set in your config. Run `/setup` to add it.

### `airesearch` command not found

The package is not installed in your active environment. Reinstall with:

```bash
uv pip install -e .
# or
pip install -e .
```

Then reload your shell or open a new terminal.
