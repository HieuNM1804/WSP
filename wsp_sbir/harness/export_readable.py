"""Readable export of a run, in the format of lm_understand-logs/exp8_math/runs/p50c/readable/.

  python3 harness/export_readable.py runs/b25_r1      # -> runs/b25_r1/readable/

  README.md        the run in one paragraph and its final answer
  prompt.md        the first message
  trajectory.md    every turn: the message the editor received (cut at 4,000 characters) and its reply in full
  passes.md        every forward request: type, model, size, result
  explanations.md  the editor's per-step explanations (copied)
  notes.md         the editor's notebook /work/notes.md (copied)
  final_answer.md  (copied)
"""
import json
import os
import shutil
import sys

CUT = 4000


def read(path, default=""):
    return open(path, errors="replace").read() if os.path.exists(path) else default


def pass_row(k, req, done):
    typ = req.get("type", "?") if isinstance(req, dict) else "?"
    model = req.get("model", "?") if isinstance(req, dict) else "?"
    if typ == "retrieve":
        size = f"{done.get('sketches', '?')} sketches x {done.get('photos', '?')} photos" if done.get("ok") else \
            (f"{len(req.get('categories', []))} categories" if "categories" in req else
             f"{len(req.get('sketches', []))} x {len(req.get('photos', []))}")
    else:
        size = f"{len(req.get('images', []))} images, {len(req.get('texts', []))} texts"
    if not done.get("ok"):
        result = f"failed (not counted): {str(done.get('error', ''))[:200]}"
    elif typ == "retrieve" and "paired" in done:
        metric = next(iter(done["paired"]))
        p = done["paired"][metric]
        result = (f"{metric} edited - original {p['difference']:+.4f} +- {p['se_over_categories']} (over categories), "
                  f"+- {p['se_over_queries']} (over queries); original {done['original'][metric]:.4f}, "
                  f"edited {done['edited'][metric]:.4f}")
    elif typ == "retrieve":
        name = done.get("model")
        result = ", ".join(f"{k2} {v:.4f}" for k2, v in done.get(name, {}).items())
    else:
        result = f"ok, {done.get('images', 0)} images, {done.get('texts', 0)} texts"
    return f"| {k} | {typ} | {model} | {size} | {result.replace('|', '/')} |"


def export(run_dir):
    run_dir = os.path.abspath(run_dir)
    name = os.path.basename(run_dir)
    out = os.path.join(run_dir, "readable")
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    state = json.load(open(os.path.join(run_dir, "state.json")))
    turns = sorted(os.listdir(os.path.join(run_dir, "turns"))) if os.path.isdir(os.path.join(run_dir, "turns")) else []

    final = read(os.path.join(run_dir, "final_answer.md"), "(no final answer yet)")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"# Run {name}\n\nTask {state['task']}, prompt {state['prompt']}, editor {cfg['backend']} "
                f"({cfg['model']}, effort {cfg['effort']}), budget {cfg['budget']} forward passes "
                f"({state['passes']} used), {state['turn']} turns, {state['runs']} Python runs, "
                f"{state['handoffs']} conversation restarts"
                f"{'' if state['done'] else ' (not finished)'}.\n\n"
                "Files: prompt.md (first message), trajectory.md (every turn), passes.md (every forward pass), "
                "explanations.md, notes.md (the editor's notebook), final_answer.md.\n\n## Final answer\n\n"
                f"{final}\n")
    with open(os.path.join(out, "prompt.md"), "w") as f:
        f.write(read(os.path.join(run_dir, "turns", turns[0], "prompt.txt")) if turns else "")

    with open(os.path.join(out, "trajectory.md"), "w") as f:
        f.write(f"# {name}: trajectory ({len(turns)} turns)\n\nEach turn: the message the editor received (tool "
                f"output; cut at {CUT:,} characters) and its reply in full. The first message is prompt.md.\n")
        for i, t in enumerate(turns):
            tdir = os.path.join(run_dir, "turns", t)
            received = read(os.path.join(tdir, "prompt.txt"))
            if i == 0:
                received = "(the prompt: see prompt.md)"
            elif received.startswith(read(os.path.join(run_dir, "message.txt"))[:500]) and len(received) > 1000:
                received = "(the task message again: a new conversation)\n" + received[len(read(os.path.join(
                    run_dir, "message.txt"))):]
            if len(received) > CUT:
                received = received[:CUT] + f"\n[cut at {CUT:,} characters]"
            answer = read(os.path.join(tdir, "answer.md"), "(no reply recorded)")
            f.write(f"\n## {t}\n\n### received\n\n````text\n{received}\n````\n\n### editor\n\n{answer}\n\n")

    rows = []
    req_dir = os.path.join(run_dir, "requests")
    ks = sorted({int(f.split("_")[1].split(".")[0]) for f in os.listdir(req_dir)}) if os.path.isdir(req_dir) else []
    for k in ks:
        try:
            req = json.load(open(os.path.join(req_dir, f"fwd_{k}.json.claimed")))
        except (OSError, json.JSONDecodeError):
            req = {}
        try:
            done = json.load(open(os.path.join(req_dir, f"fwd_{k}.done.json")))
        except (OSError, json.JSONDecodeError):
            done = {"ok": False, "error": "no result recorded"}
        rows.append(pass_row(k, req, done))
    with open(os.path.join(out, "passes.md"), "w") as f:
        f.write(f"# {name}: forward passes\n\n| request | type | model | size | result |\n|---|---|---|---|---|\n"
                + "\n".join(rows) + "\n")

    for src, dst in (("explanations.md", "explanations.md"), ("work/notes.md", "notes.md"),
                     ("final_answer.md", "final_answer.md")):
        source = os.path.join(run_dir, src)
        if os.path.isfile(source) and not os.path.islink(source):
            shutil.copy2(source, os.path.join(out, dst))
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(export(d))
