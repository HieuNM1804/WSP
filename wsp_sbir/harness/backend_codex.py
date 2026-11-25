"""Run a tool-free Codex turn and record its JSONL audit."""

import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path


DISABLED_FEATURES = (
    "code_mode", "code_mode_host", "shell_tool", "unified_exec", "unified_exec_tty", "shell_snapshot",
    "view_image", "image_generation", "multi_agent", "multi_agent_v2", "apps", "plugins", "remote_plugin",
    "browser_use", "browser_use_external", "browser_use_full_cdp_access", "in_app_browser", "computer_use",
    "skill_search", "skill_mcp_dependency_install", "sleep_tool", "tool_suggest", "goals", "hooks",
    "workspace_dependencies",
)


def new_session_id():
    return None  # Codex assigns the thread id in thread.started.


def build_cmd(session_id, resume, model, effort, instructions_path, autocompact):
    binary = os.environ.get("WSP_CODEX_BIN") or shutil.which("codex.exe") or "codex"
    if binary.lower().endswith(".exe") and shutil.which("wslpath"):
        instructions_path = subprocess.check_output(["wslpath", "-m", str(instructions_path)], text=True).strip()
    cmd = [binary, "exec"]
    if resume:
        cmd.append("resume")
    cmd += ["--json", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules", "-m", model]
    options = {
        "model_reasoning_effort": json.dumps(effort),
        "model_instructions_file": json.dumps(str(instructions_path)),
        "approval_policy": '"never"',
        "sandbox_mode": '"read-only"',
        "web_search": '"disabled"',
        "mcp_servers": "{}",
        "agents.enabled": "false",
    }
    if autocompact:
        options["model_auto_compact_token_limit"] = str(autocompact)
    for key, value in options.items():
        cmd += ["-c", f"{key}={value}"]
    for feature in DISABLED_FEATURES:
        cmd += ["--disable", feature]
    if resume:
        cmd.append(session_id)
    return cmd + ["-"]


def audit_output(output):
    events = Counter()
    items = Counter()
    unexpected = []
    thread_id = usage = answer = None
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            unexpected.append("invalid JSON event")
            continue
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            unexpected.append("invalid JSON event")
            continue
        kind = event.get("type")
        events[kind] += 1
        if kind == "thread.started":
            thread_id = event.get("thread_id")
        elif kind == "turn.completed":
            usage = event.get("usage")
        elif kind and kind.startswith("item."):
            item = event.get("item", {})
            if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                unexpected.append("invalid item")
                continue
            item_type = item.get("type")
            items[item_type] += 1
            if item_type == "agent_message" and kind == "item.completed":
                answer = item.get("text")
            elif item_type in {"agent_message", "reasoning"}:
                continue
            elif item_type == "error" and item.get("message", "").startswith(
                "Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed;"
            ):
                continue
            else:
                unexpected.append(item_type or "unknown item")
        elif kind not in {"turn.started", "turn.failed", "error"}:
            unexpected.append(kind or "unknown event")
    completed = events["turn.started"] == events["turn.completed"] == events["thread.started"] == 1
    tool_free = completed and thread_id and not unexpected and not events["error"] and not events["turn.failed"]
    audit = {
        "init": {"thread_id": thread_id} if thread_id else None,
        "event_types": dict(events),
        "item_types": dict(items),
        "tool_blocks": unexpected,
        "tool_free": bool(tool_free),
        "usage": usage,
    }
    return audit, answer


def call(prompt, session_id, resume, turn_dir, model, effort, instructions, timeout=7200, retries=4,
         log=print, autocompact=None):
    turn_dir = Path(turn_dir)
    turn_dir.mkdir(parents=True, exist_ok=True)
    run_dir = turn_dir.parent.parent
    cwd = run_dir / "agent_cwd"
    cwd.mkdir(exist_ok=True)
    cmd = build_cmd(session_id, resume, model, effort, run_dir / "instructions.md", autocompact)

    for attempt in range(1, retries + 1):
        start = time.monotonic()
        try:
            result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=timeout, cwd=cwd)
            rc, out, err = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            rc, out, err = -9, (exc.stdout or b"").decode("utf-8", "replace"), f"Timed out after {timeout} s"
        except OSError as exc:
            rc, out, err = -1, "", str(exc)
        (turn_dir / "events.jsonl").write_text(out, encoding="utf-8")
        (turn_dir / "stderr.log").write_text(err, encoding="utf-8")
        audit, answer = audit_output(out)
        thread = (audit["init"] or {}).get("thread_id")
        if resume and thread and thread != session_id:
            audit["tool_free"] = False
            audit["tool_blocks"].append("unexpected resume thread")
        ok = rc == 0 and isinstance(answer, str) and bool(answer.strip())
        record = {
            "ok": ok, "refused": False, "rc": rc, "session": thread, "resume": resume, "model": model,
            "effort": effort, "attempt": attempt, "wall_s": round(time.monotonic() - start, 1),
            "audit": audit, "cmd": cmd, "stderr_tail": err[-2000:],
        }
        (turn_dir / "run.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        if ok:
            (turn_dir / "answer.md").write_text(answer, encoding="utf-8")
        if ok or audit["tool_blocks"] or rc == -1 or attempt == retries:
            return dict(record, answer=answer)
        delay = (60, 300, 900)[min(attempt - 1, 2)]
        log(f"{turn_dir.name} attempt {attempt}: Codex exit {rc}, retrying in {delay} s")
        time.sleep(delay)
