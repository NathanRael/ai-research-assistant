from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.user_data import config_file

# Build a list of .env search paths: user data dir first, then project root.
_ENV_FILES: list[str] = [str(config_file())]
_project_env = Path(__file__).resolve().parents[1] / ".env"
if _project_env.is_file():
    _ENV_FILES.append(str(_project_env))


class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    opencode_api_key: str = ""
    opencode_anthropic_url: str = ""
    opencode_open_ai_url: str = ""
    lang_search_api: str = ""

    opencode_model: str = "kimi-k2.6"

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_sender: str = ""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
