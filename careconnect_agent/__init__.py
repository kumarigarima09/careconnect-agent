"""CareConnect Clinics AI Agent package.

Run:
    pip install -r requirements.txt
    cp .env.example .env    # set OPENAI_API_KEY
    python run.py           # interactive CLI

Entrypoints (available once all submodules are implemented):
    from careconnect_agent import CareConnectAgent, ConversationState
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config import get_settings, ConfigError, Settings

if TYPE_CHECKING:
    from .agent import CareConnectAgent
    from .state import ConversationState, PatientType, Intent, LeadTemperature
    from .knowledge_base import KnowledgeBase

__all__ = [
    "get_settings",
    "ConfigError",
    "Settings",
    "CareConnectAgent",
    "ConversationState",
    "PatientType",
    "Intent",
    "LeadTemperature",
    "KnowledgeBase",
]


def __getattr__(name):
    if name == "CareConnectAgent":
        from .agent import CareConnectAgent
        return CareConnectAgent
    if name in ("ConversationState", "PatientType", "Intent", "LeadTemperature"):
        from . import state
        return getattr(state, name)
    if name == "KnowledgeBase":
        from .knowledge_base import KnowledgeBase
        return KnowledgeBase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
