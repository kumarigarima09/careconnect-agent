"""Configuration loader.

Reads OPENAI_API_KEY and other env vars. Fails with a clear, catchable error when
the LLM key is missing so the caller can surface it to the user.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing or invalid."""


def _load_env_file() -> None:
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            return


@dataclass(frozen=True)
class Settings:
    openai_api_key: Optional[str]
    openai_model: str
    salesforce_domain: Optional[str]
    salesforce_username: Optional[str]
    salesforce_password: Optional[str]
    salesforce_security_token: Optional[str]

    @property
    def llm_available(self) -> bool:
        return bool(self.openai_api_key)

    def require_llm(self) -> None:
        if not self.llm_available:
            raise ConfigError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export OPENAI_API_KEY in your shell."
            )


def get_settings() -> Settings:
    _load_env_file()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        salesforce_domain=os.getenv("SALESFORCE_DOMAIN") or None,
        salesforce_username=os.getenv("SALESFORCE_USERNAME") or None,
        salesforce_password=os.getenv("SALESFORCE_PASSWORD") or None,
        salesforce_security_token=os.getenv("SALESFORCE_SECURITY_TOKEN") or None,
    )
