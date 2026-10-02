"""Shared fixtures for all scenario tests."""
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402  (after sys.path fix)

from careconnect_agent.agent import CareConnectAgent  # noqa: E402
from careconnect_agent.llm import MockLLMClient  # noqa: E402
from careconnect_agent.salesforce import MockSalesforceAdapter  # noqa: E402


@pytest.fixture
def agent(tmp_path) -> CareConnectAgent:
    sf = MockSalesforceAdapter(out_dir=tmp_path / "sf")
    llm = MockLLMClient(fallback="(mock LLM)")
    a = CareConnectAgent(llm=llm, salesforce=sf)
    return a


def say(agent: CareConnectAgent, msg: str) -> str:
    return agent.handle(msg)
