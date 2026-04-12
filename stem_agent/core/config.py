"""
Stem Agent Configuration

Global configuration constants and defaults. These are the "knobs"
that control the stem agent's behavior without changing code.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class StemConfig(BaseSettings):
    """
    Configuration loaded from environment variables and/or .env file.

    Env vars are prefixed with STEM_AGENT_ and uppercased.
    Example: STEM_AGENT_MODEL=gpt-4o
    """

    model: str = Field(default="gpt-4o", alias="STEM_AGENT_MODEL")
    fallback_model: str = Field(
        default="gpt-4o-mini", alias="STEM_AGENT_FALLBACK_MODEL"
    )
    temperature: float = 0.2
    max_retries: int = 3

    serpapi_api_key: str = Field(default="", alias="SERPAPI_API_KEY")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    branch_threshold: float = 0.75
    stop_threshold: float = 0.85
    plateau_window: int = 5
    plateau_min_delta: float = 0.02
    max_iterations: int = 20

    max_checkpoints: int = 5
    task_history_limit: int = 20

    log_level: str = Field(default="INFO", alias="STEM_AGENT_LOG_LEVEL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


config = StemConfig()
