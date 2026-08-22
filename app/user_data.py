"""User data directory management.

All user-specific data (config, storage, documents, history) lives under
a dedicated directory so the application code stays separate from user data.

Default:  ~/.airi/
Override: set the AI_ASSISTANT_HOME environment variable.
"""

import os
import shutil
from pathlib import Path

_APP_NAME = "airi"
_OLD_APP_NAME = "ai-research-assistant"


def _home_dir() -> Path:
    """Return the root user-data directory, creating it if needed."""
    override = os.environ.get("AI_ASSISTANT_HOME")
    if override:
        base = Path(override)
    else:
        base = Path.home() / f".{_APP_NAME}"
    base.mkdir(parents=True, exist_ok=True)
    _migrate_old_data(base)
    return base


def _old_home_dir() -> Path:
    return Path.home() / f".{_OLD_APP_NAME}"


def _migrate_old_data(new_base: Path) -> None:
    """Copy profile.json and .env from old directory if they exist and new copies don't."""
    old_base = _old_home_dir()
    if not old_base.is_dir():
        return
    for name in ("profile.json", ".env"):
        old_file = old_base / name
        new_file = new_base / name
        if old_file.is_file() and not new_file.is_file():
            shutil.copy2(old_file, new_file)


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


def profile_file() -> Path:
    """Path to the structured user profile file (JSON)."""
    return _home_dir() / "profile.json"
