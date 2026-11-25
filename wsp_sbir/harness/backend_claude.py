"""One editor turn = one call of Claude Code in print mode, with every tool switched off.

  new conversation:  claude -p --session-id <uuid> ...   (prompt on stdin)
  later turns:       claude -p --resume <uuid> ...

Tools off: --tools "" (no built-in tools), --strict-mcp-config with an empty --mcp-config (no MCP servers),
--safe-mode (no CLAUDE.md, skills, hooks, plugins' commands, custom agents), --disable-slash-commands, --no-chrome,
and --system-prompt replaces Claude Code's system prompt with the neutral instructions file.

Every turn is audited from its stream-json events: it is tool-free when the init event lists no tools, MCP servers
or skills, and no assistant message holds a tool_use / server_tool_use block.
"""
import json
import os
import subprocess
import time
import uuid

CLAUDE = os.path.expanduser("~/.local/bin/claude")


def new_session_id():
    return str(uuid.uuid4())


def build_cmd(session_id, resume, model, effort, instructions, autocompact=None):
    cmd = [CLAUDE, "-p", "--output-format", "stream-json", "--verbose",
           "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           "--safe-mode", "--disable-slash-commands", "--no-chrome", "--permission-mode", "dontAsk",
           "--system-prompt", instructions, "--model", model,
           # ask for thinking summaries; in tests (Opus 5.5, Claude Code 2.1.311) the thinking blocks still came back
           # empty (signature only), so events.jsonl holds the thinking-token counts, like Exp 8's "no reasoning
           # summaries were emitted"; if summaries are returned, they are kept in events.jsonl
           "--settings", '{"showThinkingSummaries": true}']
    if effort:
        cmd += ["--effort", effort]
    if autocompact:  # Exp 8 set model_auto_compact_token_limit=1000000: conversations are never summarized
        cmd += ["--autocompact", str(autocompact)]
    cmd += ["--resume", session_id] if resume else ["--session-id", session_id]
    return cmd


def audit(events):
    """Summarize a turn's events; tool_free is the condition for executing the reply."""
    a = {"event_types": {}, "content_types": {}, "tool_blocks": [], "init": None, "result": None,
         "stop_reasons": []}
    for e in events:
        key = e.get("type", "?") + ("/" + e["subtype"] if e.get("subtype") else "")
        a["event_types"][key] = a["event_types"].get(key, 0) + 1
        if e.get("type") == "system" and e.get("subtype") == "init":
            a["init"] = {"model": e.get("model"), "tools": e.get("tools"), "mcp_servers": e.get("mcp_servers"),
                         "skills": e.get("skills"), "slash_commands": len(e.get("slash_commands") or []),
                         "session_id": e.get("session_id"), "claude_code_version": e.get("claude_code_version"),
                         "permissionMode": e.get("permissionMode")}
        elif e.get("type") == "assistant":
            msg = e.get("message", {})
            if msg.get("stop_reason"):
                a["stop_reasons"].append(msg["stop_reason"])
            for c in msg.get("content", []):
                t = c.get("type", "?")
                a["content_types"][t] = a["content_types"].get(t, 0) + 1
                if t in ("tool_use", "server_tool_use", "mcp_tool_use"):
                    a["tool_blocks"].append(c.get("name"))
        elif e.get("type") == "result":
            a["result"] = {k: e.get(k) for k in ("subtype", "is_error", "num_turns", "session_id",
                                                  "duration_ms", "stop_reason", "usage")}
    init = a["init"] or {}
    a["compacted"] = any("compact" in k for k in a["event_types"])  # e.g. system/compact_boundary
    a["tool_free"] = bool(a["init"]) and not init.get("tools") and not init.get("mcp_servers") \
        and not init.get("skills") and not a["tool_blocks"]
    return a


def call(prompt, session_id, resume, turn_dir, model, effort, instructions, timeout=7200, retries=4,
         log=print, autocompact=None):
    """Run one turn. Writes events.jsonl, stderr.log, answer.md, run.json into turn_dir. Returns the run record."""
    os.makedirs(turn_dir, exist_ok=True)
    cwd = os.path.join(os.path.dirname(os.path.dirname(turn_dir)), "agent_cwd")  # empty folder, nothing to read
    os.makedirs(cwd, exist_ok=True)
    waits = []
    record = None
    for attempt in range(1, retries + 1):
        if attempt > 1 and not resume:
            session_id = new_session_id()  # a failed first call may already have claimed the old id
        cmd = build_cmd(session_id, resume, model, effort, instructions, autocompact)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=cwd)
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as exc:
            rc, out, err = -9, (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""), \
                f"[timed out after {timeout} s]"
        wall = round(time.time() - t0, 1)
        events = []
        for line in out.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        with open(os.path.join(turn_dir, "events.jsonl"), "w") as f:
            f.write(out)
        with open(os.path.join(turn_dir, "stderr.log"), "w") as f:
            f.write(err)
        a = audit(events)
        result = next((e for e in reversed(events) if e.get("type") == "result"), None)
        answer = (result or {}).get("result") if result and not result.get("is_error") else None
        refused = "refusal" in a["stop_reasons"]
        ok = rc == 0 and answer is not None and not refused
        record = {"ok": ok, "rc": rc, "wall_s": wall, "attempt": attempt, "waits": waits, "refused": refused,
                  "model": model, "effort": effort, "session": session_id, "resume": resume,
                  "audit": a, "stderr_tail": err[-2000:], "cmd": cmd[:-1] + ["<session>"],
                  "end": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if ok or refused:
            break
        wait = [60, 300, 900, 1800][min(attempt - 1, 3)]
        log(f"{os.path.basename(turn_dir)} attempt {attempt}: claude failure (rc {rc}); waiting {wait} s; "
            f"result {json.dumps((result or {}).get('result'))[:300]} stderr {err[-300:]!r}")
        waits.append(wait)
        if attempt < retries:
            time.sleep(wait)
    if record["ok"]:
        with open(os.path.join(turn_dir, "answer.md"), "w") as f:
            f.write(answer)
    record["answer"] = answer
    with open(os.path.join(turn_dir, "run.json"), "w") as f:
        json.dump({k: v for k, v in record.items() if k != "answer"}, f, indent=1)
    return record
