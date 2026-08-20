"""User data directory management.

All user-specific data (config, storage, documents, history) lives under
a dedicated directory so the application code stays separate from user data.

Default:  ~/.ai-research-assistant/
Override: set the AI_ASSISTANT_HOME environment variable.
"""

import os
from pathlib import Path

_APP_NAME = "ai-research-assistant"


def _home_dir() -> Path:
    """Return the root user-data directory, creating it if needed."""
    override = os.environ.get("AI_ASSISTANT_HOME")
    if override:
        base = Path(override)
    else:
        base = Path.home() / f".{_APP_NAME}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def storage_dir() -> Path:
    """Directory for Chroma vector-store persistence."""
    d = _home_dir() / "storage"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_file() -> Path:
    """Path to the user's .env configuration file."""
    return _home_dir() / ".env"


def history_file() -> Path:
    """Path to the CLI command-history file."""
    return _home_dir() / ".history"


def documents_dir() -> Path:
    """Default directory where added documents are copied (optional)."""
    d = _home_dir() / "documents"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_file() -> Path:
    """Path to the debug log file."""
    return _home_dir() / "assistant.log"
