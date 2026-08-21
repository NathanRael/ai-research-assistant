# airi — Features

## What It Is

airi is a multi-agent CLI personal assistant built with LangChain and LangGraph. It runs in the terminal, routes user requests to specialist agents, and uses tools to search the web, manage documents, store memories, and automate tasks.

The assistant keeps user data separate from the application code and persists vector-based document memory and user profile facts across sessions.

## Core Features

### Web Search

- Ask questions that require current information from the internet.
- Uses the LangSearch API to retrieve live results.
- The supervisor routes web questions to the dedicated `WebSearchAgent`.

### Document Memory

- Add PDF, TXT, or Markdown files with `/add <path>`.
- Documents are chunked, embedded, and stored in a local ChromaDB vector store.
- Ask questions about document contents; the assistant retrieves relevant chunks and answers.
- List indexed documents with `/list`.

### User Context

- Save personal facts and preferences to a persistent profile.
- The assistant remembers context across sessions.
- Profile is stored as structured JSON in the user data directory.

### Automation

- Send emails via SMTP using the `send_email` tool.
- Preview emails before sending.
- Read emails via IMAP.
- Trigger external actions through the automation agent.

### Debug Mode

- Toggle `/debug enable|disable` to trace routing decisions and tool calls.
- See which agent handles each request and which tools are invoked.

### Plain Text Output

- Responses are returned as clean, readable plain text without Markdown formatting by default.

### Persistent User Data

All user-specific data lives in `~/.airi/`:

- `profile.json` — structured user facts and preferences
- `storage/` — Chroma vector store for documents
- `.env` — configuration and API keys
- `.history` — command history
- `documents/` — optional local document copies

Override the data directory with the `AI_ASSISTANT_HOME` environment variable.

## How It Works

```
You
  ↓
SupervisorAgent
  ├── WebSearchAgent    → search_web
  ├── UserContextAgent  → save_memory, retrieve_memory, search_documents,
  │                       list_documents, get_user_profile, update_user_profile
  └── AutomationAgent   → preview_email, send_email, preview_read_emails, read_emails
```

The supervisor classifies each request and dispatches it to the specialist agent with the right tools and system prompt.
