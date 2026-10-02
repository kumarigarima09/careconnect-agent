"""CareConnect Clinics AI Agent — local run entrypoint.

Usage:
    # Interactive CLI chat loop. Ctrl-D to exit.
    python run.py

    # Non-interactive: feed a script file (one user message per non-empty line).
    python run.py --script test_scripts/new_patient_dermatology.txt

    # HTTP API (FastAPI): POST /chat  {"message": "...", "conversation_id": "abc123"}
    python run.py --http --port 8000

Requires:
    pip install -r requirements.txt
    cp .env.example .env     # set OPENAI_API_KEY (or omit for offline rule-based fallback)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover — pydantic required only for --http
    class BaseModel:  # type: ignore[no-redef]
        pass

from careconnect_agent.agent import CareConnectAgent
from careconnect_agent.llm import MockLLMClient, build_default_client
from careconnect_agent.salesforce import MockSalesforceAdapter
from careconnect_agent.config import get_settings, ConfigError


class ChatIn(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatOut(BaseModel):
    reply: str
    conversation_id: str


def make_agent(args: argparse.Namespace) -> CareConnectAgent:
    settings = get_settings()
    if args.mock_llm or not settings.llm_available:
        llm = MockLLMClient(fallback="(LLM disabled — using rule-based fallback answers.)")
    else:
        try:
            settings.require_llm()
            llm = build_default_client()
        except ConfigError as e:
            print(f"[warning] {e} — falling back to rule-based answers.")
            llm = MockLLMClient(fallback="(LLM unavailable.)")
    sf = MockSalesforceAdapter()
    agent = CareConnectAgent(llm=llm, salesforce=sf)
    return agent


def run_cli(agent: CareConnectAgent, script_path: str | None = None) -> None:
    print(agent.start())
    print()
    if script_path:
        lines = [ln.strip() for ln in Path(script_path).read_text().splitlines() if ln.strip()]
        for msg in lines:
            print(f">> {msg}")
            reply = agent.handle(msg)
            print()
            print(reply)
            print()
    else:
        try:
            while True:
                try:
                    msg = input("You: ").strip()
                except EOFError:
                    print("\nBye for now!")
                    break
                if not msg:
                    continue
                if msg.lower() in {"quit", "exit", "bye"}:
                    print("Thank you. Take care!")
                    break
                reply = agent.handle(msg)
                print(f"\nAgent: {reply}\n")
        except KeyboardInterrupt:
            print("\nInterrupted. Bye!")
    # Session summary
    if agent.state.submitted:
        print(
            "A Salesforce Lead and Task were created by the mock adapter. "
            f"See: .out/salesforce/ (conversation id {agent.state.conversation_id})"
        )


def run_http(agent: CareConnectAgent, port: int) -> None:
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException, Body
        from fastapi.responses import HTMLResponse
    except ImportError as e:  # pragma: no cover
        print(f"Install fastapi + uvicorn to use --http mode: pip install fastapi uvicorn  ({e})")
        sys.exit(1)

    use_mock_llm: bool = not getattr(agent.llm, "llm_available", False) or isinstance(agent.llm, MockLLMClient)
    app = FastAPI(title="CareConnect Clinics AI Agent")
    sessions: dict[str, CareConnectAgent] = {}

    INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>CareConnect AI Assistant</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f4f6fb; display: flex; justify-content: center; align-items: center;
    min-height: 100vh; padding: 24px 16px;
  }
  .chat {
    width: 100%; max-width: 720px; height: 86vh; max-height: 820px;
    background: #fff; border-radius: 16px; box-shadow: 0 10px 40px rgba(20,30,60,0.08);
    display: flex; flex-direction: column; overflow: hidden; border: 1px solid #e6ebf3;
  }
  .header {
    padding: 18px 22px; background: linear-gradient(135deg, #2563eb, #7c3aed); color: #fff;
    display: flex; align-items: center; gap: 12px;
  }
  .header .logo {
    width: 38px; height: 38px; border-radius: 10px; background: rgba(255,255,255,0.18);
    display: flex; align-items: center; justify-content: center; font-weight: 700;
  }
  .header h1 { margin: 0; font-size: 16px; font-weight: 600; }
  .header .sub { font-size: 12px; opacity: 0.85; margin-top: 2px; }
  .messages { flex: 1; padding: 20px 22px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 82%; padding: 10px 14px; border-radius: 14px; line-height: 1.45; white-space: pre-wrap; word-wrap: break-word; font-size: 14px; }
  .msg.agent { background: #eef2ff; color: #1f2a44; align-self: flex-start; border-bottom-left-radius: 4px; }
  .msg.user { background: #2563eb; color: #fff; align-self: flex-end; border-bottom-right-radius: 4px; }
  .msg.typing { opacity: 0.7; font-style: italic; }
  .inputbar { display: flex; gap: 8px; padding: 14px 16px; border-top: 1px solid #eef1f7; background: #fafbfe; }
  .inputbar input {
    flex: 1; border: 1px solid #dde2ec; border-radius: 10px; padding: 10px 14px; font-size: 14px;
    outline: none; transition: border 0.15s;
  }
  .inputbar input:focus { border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,0.12); }
  .inputbar button {
    border: 0; background: #2563eb; color: #fff; padding: 10px 18px; border-radius: 10px;
    font-weight: 600; cursor: pointer; transition: background 0.15s;
  }
  .inputbar button:hover { background: #1d4ed8; }
  .inputbar button:disabled { background: #94a3b8; cursor: not-allowed; }
  .footer { font-size: 11px; color: #64748b; padding: 6px 16px 10px; text-align: center; }
</style>
</head>
<body>
  <div class="chat">
    <div class="header">
      <div class="logo">C</div>
      <div>
        <h1>CareConnect AI Assistant</h1>
        <div class="sub">Mock LLM mode &middot; Fully offline</div>
      </div>
    </div>
    <div id="messages" class="messages"></div>
    <form id="form" class="inputbar" autocomplete="off">
      <input id="msg" type="text" placeholder="Type your message… (e.g. I need a pediatrician in Delhi)" required />
      <button id="btn" type="submit">Send</button>
    </form>
    <div class="footer">Delhi &middot; Noida &middot; Meerut &mdash; AC-15/16 disclaimers apply &mdash; emergencies go to 108</div>
  </div>
<script>
  const $ = (s) => document.querySelector(s);
  const messages = $("#messages");
  const input = $("#msg");
  const btn = $("#btn");
  let cid = null;

  function addMsg(role, text) {
    const el = document.createElement("div");
    el.className = "msg " + role;
    el.textContent = text;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
    return el;
  }

  async function send(text) {
    addMsg("user", text);
    const typing = addMsg("agent typing", "Agent is typing…");
    btn.disabled = true;
    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message: text, conversation_id: cid})
      });
      const data = await res.json();
      cid = data.conversation_id;
      typing.remove();
      addMsg("agent", data.reply);
    } catch (e) {
      typing.remove();
      addMsg("agent", "Network error: " + e.message);
    } finally {
      btn.disabled = false;
      input.value = "";
      input.focus();
    }
  }

  $("#form").addEventListener("submit", (e) => {
    e.preventDefault();
    const v = input.value.trim();
    if (v) send(v);
  });

  // Kick off: first empty message triggers agent.start() via backend session init
  send("");
</script>
</body>
</html>
"""

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(INDEX_HTML)

    @app.get("/@vite/client", status_code=200)
    def vite_client_stub():
        return ""

    @app.post("/chat", response_model=ChatOut)
    def chat(body: ChatIn = Body(...)):
        session: CareConnectAgent
        if body.conversation_id and body.conversation_id in sessions:
            session = sessions[body.conversation_id]
        else:
            session = make_agent(argparse.Namespace(mock_llm=use_mock_llm))
            session.start()
            sessions[session.state.conversation_id] = session
        try:
            reply = session.handle(body.message) if body.message else session.start()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Agent error: {e}")
        return ChatOut(reply=reply, conversation_id=session.state.conversation_id)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CareConnect AI Agent demo")
    p.add_argument("--mock-llm", action="store_true",
                   help="Use rule-based fallback only (no LLM calls).")
    p.add_argument("--http", action="store_true",
                   help="Run FastAPI HTTP server instead of CLI.")
    p.add_argument("--port", type=int, default=8000, help="HTTP port")
    p.add_argument("--script", type=str, default=None,
                   help="Path to a text file of user messages (one per line) for CLI non-interactive mode.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    agent = make_agent(args)
    if args.http:
        run_http(agent, args.port)
    else:
        run_cli(agent, args.script)


if __name__ == "__main__":
    main()
