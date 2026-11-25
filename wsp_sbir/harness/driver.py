"""WSP-SBIR driver: one editor conversation that edits CLIP ViT-B/32's image encoder for zero-shot SBIR.

Mirrors the harness described in lm_understand-logs (Experiment 8): every turn is one tool-free call of the editor
(Codex CLI by default); the editor acts only by replying with exactly one ```python block (run in
the persistent sandbox kernel) or one ```forward block (run by forward.py on the GPU, counted against the budget).

  python3 harness/driver.py --task tasks/sketchy_2 --run runs/b25_r1 --budget 25
  python3 harness/driver.py --run runs/b25_r1 --resume          # continue after a stop or crash

Files of a run (as in the released logs):
  message.txt, instructions.md, state.json, loop.log, tool_calls.jsonl, explanations.md, final_answer.md
  turns/tNNN/{prompt.txt, answer.md, events.jsonl, run.json, stderr.log}
  requests/fwd_<k>.json.claimed, requests/fwd_<k>.done.json
  work/   the editor's /work (edited/, fwd_<k>/, notes.md, ...)
  edit/   the submitted edit: manifest.json and the edited tensors (float32 .npy, as submitted)
  result.zip   the completed run's portable archive for GitHub
  notices/  drop a .txt here to deliver a note from the experimenters with the next message
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from string import Template

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import backend_claude  # noqa: E402
import backend_codex  # noqa: E402
import resources  # noqa: E402
from kernel_client import Kernel  # noqa: E402
from publish_results import archive_run, source_revision  # noqa: E402

GPU_PYTHON = os.environ.get("WSP_GPU_PYTHON", os.path.join(ROOT, "..", "clip_zs_sbir", ".venv", "bin", "python"))
BACKENDS = {"codex": backend_codex, "claude-code": backend_claude}
BLOCK_RE = re.compile(r"```(python|forward)[ \t]*\n(.*?)```", re.S)
ANY_FENCE_RE = re.compile(r"```[^\n]*\n.*?```", re.S)
EXPLAIN = ("EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English (about 3 "
           "to 8 sentences): what the last result showed, what you conclude from it, what you will do now and why, "
           "and what outcome would confirm or refute your current hypothesis. In your final answer, also summarize "
           "the path that led to your edits. These explanations are read by the experimenters.")
MIN_EXPLANATION_WORDS = 20


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Run:
    def __init__(self, run_dir):
        self.dir = os.path.abspath(run_dir)
        self.state_path = os.path.join(self.dir, "state.json")
        self.cfg = json.load(open(os.path.join(self.dir, "config.json")))
        self.backend = BACKENDS[self.cfg.get("backend", "claude-code")]
        self.state = json.load(open(self.state_path))
        self.task_dir = self.cfg["task"]
        self.task = json.load(open(os.path.join(self.task_dir, "task.json")))
        self.work = os.path.join(self.dir, "work")
        self.message = open(os.path.join(self.dir, "message.txt")).read()
        self.instructions = open(os.path.join(self.dir, "instructions.md")).read().strip()
        categories = json.load(open(os.path.join(self.task_dir, "practice", "categories.json")))
        self.kernel = Kernel(self.work, os.path.join(self.task_dir, "practice"), self.task["root"], categories,
                             timeout=self.cfg["timeout"], mem_gb=self.cfg["mem_gb"],
                             min_free_ram_gb=self.cfg["min_free_ram_gb"], log=self.log)
        self.checkpoints = sorted(int(x) for x in str(self.cfg.get("checkpoints", "")).split(",") if x.strip())
        self.eval_dir = os.path.join(ROOT, "eval", os.path.basename(self.dir))

    # ------------------------------------------------------------------ bookkeeping
    def log(self, text):
        line = f"[{now()}] {text}"
        print(line, flush=True)
        for name in ("loop.log", "driver.log"):
            with open(os.path.join(self.dir, name), "a") as f:
                f.write(line + "\n")

    def save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1)
        os.replace(tmp, self.state_path)

    def tool_log(self, entry):
        with open(os.path.join(self.dir, "tool_calls.jsonl"), "a") as f:
            f.write(json.dumps(dict(t=time.time(), **entry)) + "\n")

    def edited(self):
        d = os.path.join(self.work, "edited")
        return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npy")) if os.path.isdir(d) else []

    def edits_digest(self):
        h = hashlib.sha256()
        for name in self.edited():
            h.update(name.encode())
            with open(os.path.join(self.work, "edited", name + ".npy"), "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
        return h.hexdigest()

    # ------------------------------------------------------------------ messages
    def budget_line(self):
        s, c = self.state, self.cfg
        return f"Forward passes left: {c['budget'] - s['passes']}."

    def handoff_message(self):
        s, c = self.state, self.cfg
        kernel = ("Your kernel is still running with its variables." if self.kernel.alive()
                  else "Your kernel was restarted: its variables are lost (/work is kept).")
        notes_path = os.path.join(self.work, "notes.md")
        notes = (open(notes_path, errors="replace").read()[-8000:]
                 if os.path.isfile(notes_path) and not os.path.islink(notes_path) else "(no notes.md)")
        files = sorted(os.listdir(self.work))
        return (f"{self.message.rstrip()}\n\nTHIS IS A RESTARTED CONVERSATION (to keep it short): {s['turn']} replies "
                f"happened before. Forward passes used: {s['passes']} of {c['budget']}; Python runs used: {s['runs']} "
                f"of {c['max_runs']}. {kernel} Tensors currently edited: {self.edited()}. Files in /work: {files}.\n"
                f"YOUR NOTEBOOK (/work/notes.md, last 8,000 characters):\n```\n{notes}\n```\nContinue from here.\n\n"
                f"Last result:\n{s['next']}")

    def pending_notices(self):
        d = os.path.join(self.dir, "notices")
        if not os.path.isdir(d):
            return []
        return [f for f in sorted(os.listdir(d)) if f.endswith(".txt") and f not in self.state["notices"]]

    # ------------------------------------------------------------------ actions
    def run_python(self, code):
        s, c = self.state, self.cfg
        if s["runs"] >= c["max_runs"]:
            return f"You have used all {c['max_runs']} Python runs: the block was not run. {self.budget_line()}"
        resources.wait_for(c["min_free_ram_gb"] + 2, 0, log=self.log, what="a Python run", max_wait=1800)
        out, restarted, seconds = self.kernel.run(code)
        s["runs"] += 1
        work_gb = resources.dir_size_gb(self.work)
        if work_gb > c["work_quota_gb"]:
            out += (f"\n[/work now holds {work_gb:.1f} GB, over its quota of {c['work_quota_gb']} GB: forward passes "
                    "are refused (not counted) until you delete files you no longer need]")
        if restarted:
            s["kernel_restarts"] += 1
        self.tool_log({"kind": "python", "turn": s["turn"] + 1, "run": s["runs"], "s": seconds, "code": code,
                       "out": out})
        return f"OUTPUT of Python run {s['runs']} of at most {c['max_runs']}:\n```\n{out}\n```"

    def run_forward(self, text):
        s, c = self.state, self.cfg
        if s["passes"] >= c["budget"]:
            return ("All forward passes are used: the forward block was not run. Make your final choice (the tensors "
                    "in /work/edited are submitted) and reply without a code block.")
        try:
            request = json.loads(text)
        except json.JSONDecodeError as exc:
            return f"The forward pass failed and was not counted: the block is not valid JSON ({exc})."
        s["requests"] += 1
        k = s["requests"]
        req_dir = os.path.join(self.dir, "requests")
        os.makedirs(req_dir, exist_ok=True)
        claimed = os.path.join(req_dir, f"fwd_{k}.json.claimed")
        done = os.path.join(req_dir, f"fwd_{k}.done.json")
        with open(claimed, "w") as f:
            json.dump(request, f, indent=1)
        uses_edit = isinstance(request, dict) and request.get("model") in ("edited", "both")
        key = hashlib.sha1((json.dumps(request, sort_keys=True) + (self.edits_digest() if uses_edit else ""))
                           .encode()).hexdigest()
        if key in s["req_keys"]:
            summary = {"ok": False, "error": f"this request is identical to forward pass {s['req_keys'][key]} and "
                       "the edits it uses are unchanged; retrieval is deterministic, so its results are already in "
                       "/work. Not run, not counted."}
        elif resources.dir_size_gb(self.work) > c["work_quota_gb"]:
            summary = {"ok": False, "error": f"/work holds {resources.dir_size_gb(self.work):.1f} GB, over its quota "
                       f"of {c['work_quota_gb']} GB; delete files you no longer need (for example old fwd_<k> "
                       "activations). Not run, not counted."}
        elif resources.disk_free_gb(self.work) < c["min_disk_gb"]:
            summary = {"ok": False, "error": "the machine's disk is nearly full; delete files you no longer need in "
                       "/work. Not run, not counted."}
            self.log(f"disk free {resources.disk_free_gb(self.work):.1f} GB < {c['min_disk_gb']} GB: pass refused")
        else:
            t0 = time.time()
            for attempt in range(1, 7):
                waited = resources.wait_for(c["forward_ram_gb"], c["forward_gpu_gb"], gpu=c["gpu"], log=self.log,
                                            what=f"forward request {k}")
                if waited:
                    self.log(f"forward request {k}: waited {waited:.0f} s for free RAM/GPU memory")
                room = min(c["pass_gb"], c["work_quota_gb"] - resources.dir_size_gb(self.work),
                           resources.disk_free_gb(self.work) - c["min_disk_gb"]) * 2**30
                summary = self.run_gpu_job([os.path.join(HERE, "forward.py"), "--task", self.task_dir, "--work",
                                            self.work, "--k", str(k), "--request", claimed, "--out", done,
                                            "--max-bytes", str(max(room, 0))], done, c["forward_timeout"])
                if summary.get("ok") or "out of memory" not in str(summary.get("error", "")).lower():
                    break
                self.log(f"forward request {k} attempt {attempt}: out of memory ({summary.get('error')[:160]}); "
                         "retrying in 5 minutes (not counted)")
                time.sleep(300)
            summary.setdefault("s", round(time.time() - t0, 1))
        self.tool_log({"kind": "forward", "turn": s["turn"] + 1, "request": k, "ok": summary.get("ok"),
                       "s": summary.get("s"), "code": text, "out": json.dumps(summary)})
        if not summary.get("ok"):
            return f"The forward pass failed and was not counted: {summary.get('error')}"
        s["passes"] += 1
        s["req_keys"][key] = s["passes"]
        if s["passes"] in self.checkpoints:
            self.checkpoint(f"pass_{s['passes']:03d}")
        shown = {k2: v for k2, v in summary.items() if k2 not in ("trace",)}
        msg = (f"Forward pass {s['passes']} of {c['budget']} done ({summary['type']}); results in {summary['dir']}. "
               f"Summary:\n```\n{json.dumps(shown, indent=1)}\n```\n{self.budget_line()}")
        if s["passes"] >= c["budget"]:
            msg += ("\nAll forward passes are used. You can still run Python to finalize /work/edited; then make "
                    "your final choice (the tensors in /work/edited when you finish are submitted) and reply without "
                    "a code block, saying what you changed, the evidence for it and why you expect it to help.")
        return msg

    def run_gpu_job(self, args, done, timeout):
        """Run a GPU script (forward pass or evaluation) niced and OOM-first; its JSON result is in `done`."""
        threads = str(self.cfg.get("gpu_job_threads", 6))  # CPU threads of a GPU job: the machine is shared
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.cfg["gpu"]), OMP_NUM_THREADS=threads,
                   MKL_NUM_THREADS=threads, WSP_TORCH_THREADS=threads,
                   WSP_FORWARD_WORKERS=str(self.cfg.get("forward_workers", 4)))
        if os.path.exists(done):
            os.remove(done)
        try:
            p = subprocess.run([GPU_PYTHON] + args, capture_output=True, text=True, timeout=timeout, env=env,
                               preexec_fn=resources.protect_others)
            if os.path.exists(done):
                summary = json.load(open(done))
            else:
                err = p.stderr[-1500:]
                if p.returncode < 0 or p.returncode == 137:
                    err = f"killed by signal {-p.returncode if p.returncode < 0 else 9} (out of memory?) " + err
                summary = {"ok": False, "error": f"the GPU job crashed (rc {p.returncode}): {err}"}
        except subprocess.TimeoutExpired:
            summary = {"ok": False, "error": f"timed out after {timeout} s"}
        if not os.path.exists(done):
            with open(done, "w") as f:
                json.dump(summary, f, indent=1)
        return summary

    def write_edit(self, out):
        """Copy /work/edited to out/tensors with a manifest. Returns the number of tensors."""
        import numpy as np
        if os.path.isdir(os.path.join(out, "tensors")):
            shutil.rmtree(os.path.join(out, "tensors"))  # no stale tensor from an earlier export may remain
        os.makedirs(os.path.join(out, "tensors"), exist_ok=True)
        manifest = {"tensors": [], "time": now(), "passes": self.state["passes"], "turn": self.state["turn"] + 1}
        for name in self.edited():
            src = os.path.join(self.work, "edited", name + ".npy")
            shutil.copy2(src, os.path.join(out, "tensors", name + ".npy"))
            e = np.load(src).astype(np.float32)
            o = np.load(os.path.join(ROOT, "model", "tensors", name + ".npy")).astype(np.float32)
            with open(src, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            manifest["tensors"].append({
                "name": name, "shape": list(e.shape), "changed_entries": int((e != o).sum()),
                "relative_change": float(np.linalg.norm(e - o) / max(np.linalg.norm(o), 1e-30)),
                "submitted_sha256": sha})
        with open(os.path.join(out, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        return len(manifest["tensors"])

    def checkpoint(self, label):
        """Snapshot of the edit when a pass count is reached (the edit that pass measured), evaluated on the unseen
        split. Never shown to the editor."""
        d = os.path.join(self.dir, "checkpoints", label)
        n = self.write_edit(os.path.join(d, "edit"))
        notes = os.path.join(self.work, "notes.md")
        if os.path.isfile(notes) and not os.path.islink(notes):
            shutil.copy2(notes, os.path.join(d, "notes.md"))
        best = sorted(x for x in os.listdir(self.work) if x.startswith("best"))
        with open(os.path.join(d, "info.json"), "w") as f:
            json.dump({"label": label, "passes": self.state["passes"], "turn": self.state["turn"] + 1,
                       "python_runs": self.state["runs"], "edited_tensors": self.edited(), "best_folders": best,
                       "time": now()}, f, indent=1)
        self.log(f"checkpoint {label}: {n} edited tensors")
        if self.cfg.get("eval_checkpoints", True):
            self.evaluate(os.path.join(d, "edit", "tensors"), label)

    def evaluate(self, edit_dir, label):
        os.makedirs(self.eval_dir, exist_ok=True)
        out = os.path.join(self.eval_dir, f"{label}.json")
        if not [f for f in os.listdir(edit_dir) if f.endswith(".npy")]:
            base = os.path.join(ROOT, "eval", f"unedited_{self.task['dataset']}", "result.json")
            result = dict(json.load(open(base)) if os.path.exists(base) else {}, run=os.path.basename(self.dir),
                          label=label, note="no edit at this point: identical to the unedited model")
            with open(out, "w") as f:
                json.dump(result, f, indent=1)
            self.log(f"eval {label}: no edit (unedited model)")
            return result
        t0, waited = time.time(), 0.0
        for attempt in range(1, 7):  # the GPU is shared: an out-of-memory failure is retried
            waited += resources.wait_for(self.cfg["forward_ram_gb"], self.cfg["forward_gpu_gb"], gpu=self.cfg["gpu"],
                                         log=self.log, what=f"the evaluation of {label}")
            result = self.run_gpu_job([os.path.join(ROOT, "eval", "evaluate_edit.py"), "--task", self.task_dir,
                                       "--edit-dir", edit_dir, "--name", f"{os.path.basename(self.dir)}/{label}",
                                       "--workers", str(self.cfg.get("eval_workers", 2)),
                                       "--test_batch_size", str(self.cfg.get("eval_batch_size", 1024)),
                                       "--out", out], out, 3 * 3600)
            if result.get("ok", True):  # any crash of the eval job (OOM, DataLoader worker IPC) is retried
                break
            self.log(f"eval {label} attempt {attempt}: failed ({str(result.get('error'))[:120]}); retrying in 5 minutes")
            time.sleep(300)
        metrics = {k: round(v, 4) for k, v in result.items() if k.startswith(("mAP", "P@")) and isinstance(v, float)}
        self.log(f"eval {label} (unseen split, not shown to the editor): {metrics or result.get('error')}; "
                 f"{time.time() - t0:.0f} s, waited {waited:.0f} s")
        return result

    def handle(self, answer, turn):
        s, c = self.state, self.cfg
        blocks = BLOCK_RE.findall(answer)
        explanation = ANY_FENCE_RE.sub("", answer).strip()
        if not blocks:
            if s["passes"] >= c["budget"]:
                with open(os.path.join(self.dir, "final_answer.md"), "w") as f:
                    f.write(answer)
                self.append_explanation(turn, answer, final=True)
                return "final", None
            s["nudges"] += 1
            return "no block", (f"Your reply had no ```python or ```forward block, so nothing was run. The session "
                                f"continues until all {c['budget']} forward passes are used. {self.budget_line()}")
        if len(blocks) > 1:
            return "several blocks", (f"Your reply had {len(blocks)} blocks; reply with exactly one ```python or "
                                      "```forward block. Nothing was run (no Python run or forward pass was used).")
        if c["explain"] and len(explanation.split()) < MIN_EXPLANATION_WORDS:
            if s["unexplained_streak"] < 2:
                s["unexplained_streak"] += 1
                s["unexplained"] += 1
                return "unexplained", ("Your reply had no explanation, so its block was NOT run (no Python run or "
                                       "forward pass was used). " + EXPLAIN)
        s["unexplained_streak"] = 0
        self.append_explanation(turn, explanation)
        kind, body = blocks[0]
        if kind == "python":
            return "python", self.run_python(body)
        return "forward", self.run_forward(body)

    def append_explanation(self, turn, text, final=False):
        with open(os.path.join(self.dir, "explanations.md"), "a") as f:
            f.write(f"\n## Turn {turn}{' (final answer)' if final else ''}\n\n{text.strip()}\n")

    def export_edit(self):
        return self.write_edit(os.path.join(self.dir, "edit"))

    # ------------------------------------------------------------------ main loop
    def loop(self):
        s, c = self.state, self.cfg
        if s["turn"] > 0 and s["runs"] > 0 and not s["done"]:  # a resumed run starts a fresh kernel
            s["next"] = ("[The harness was restarted: your kernel was restarted and its variables are lost; /work is "
                         "kept.]\n\n" + s["next"])
        while not s["done"]:
            if s["turn"] >= c["max_turns"]:
                self.log(f"stopping: max_turns {c['max_turns']} reached")
                break
            turn = s["turn"] + 1
            tdir = os.path.join(self.dir, "turns", f"t{turn:03d}")
            new_thread = s["thread"] is None or s["since_handoff"] >= c["handoff_every"]
            if s["thread"] is None:
                prompt = self.message if turn == 1 else self.handoff_message()
            elif new_thread:
                prompt = self.handoff_message()
            else:
                prompt = s["next"]
            for name in self.pending_notices():
                note = open(os.path.join(self.dir, "notices", name)).read().strip()
                prompt = f"[note from the experimenters] {note}\n\n{prompt}"
                s["notices"].append(name)
            os.makedirs(tdir, exist_ok=True)
            with open(os.path.join(tdir, "prompt.txt"), "w") as f:
                f.write(prompt)
            session = self.backend.new_session_id() if new_thread else s["thread"]
            snap = resources.snapshot(self.work, self.cfg["gpu"])
            snap.update(turn=turn, kernel_rss_gb=round(resources.tree_rss_gb(self.kernel.proc.pid), 2)
                        if self.kernel.alive() else 0.0, kernel_peak_rss_gb=round(self.kernel.peak_rss_gb, 2))
            with open(os.path.join(self.dir, "resources.jsonl"), "a") as f:
                f.write(json.dumps(snap) + "\n")
            self.log(f"turn {turn}: {'new thread' if new_thread else 'resume'} ({len(prompt):,} chars)")
            rec = self.backend.call(prompt, session, not new_thread, tdir, c["model"], c["effort"], self.instructions,
                                    timeout=c["turn_timeout"], log=self.log, autocompact=c.get("autocompact"))
            if rec["audit"].get("compacted"):
                self.log(f"turn {turn}: WARNING: editor compacted the conversation "
                         f"({[k for k in rec['audit']['event_types'] if 'compact' in k]})")
            if not rec["audit"]["tool_free"] and (rec["ok"] or rec["audit"]["tool_blocks"]):
                s["flags"] += 1
                self.save()
                self.log(f"turn {turn}: NOT tool-free ({rec['audit']['tool_blocks']}, init {rec['audit']['init']}); "
                         "stopping the run")
                raise SystemExit(2)
            if rec["refused"]:
                s["refusals"] += 1
                s["thread"] = rec["session"]
                s["since_handoff"] += 1
                s["turn"] = turn
                s["next"] = ("[The output of your last step could not be delivered: the provider refused the reply. "
                             "Continue the work; nothing was run.]\n\n" + s["next"])
                self.log(f"turn {turn}: refused by the provider")
                self.save()
                continue
            if not rec["ok"]:
                self.save()
                self.log(f"turn {turn}: the editor could not be reached after {rec['attempt']} attempts; stopping "
                         "(continue later with --resume)")
                raise SystemExit(1)
            if new_thread:
                s["handoffs"] += 1 if turn > 1 else 0
                s["since_handoff"] = 0
            s["thread"] = rec["session"]
            s["since_handoff"] += 1
            outcome, nxt = self.handle(rec["answer"], turn)
            s["turn"] = turn
            if outcome == "final":
                n = self.export_edit()
                s["done"] = True
                self.save()
                if self.cfg.get("eval_checkpoints", True):
                    self.evaluate(os.path.join(self.dir, "edit", "tensors"), "final")
                self.log(f"final answer: {s['runs']} python runs, {s['passes']} forward passes, {s['nudges']} nudges, "
                         f"{s['handoffs']} handoffs, {n} edited tensors")
                self.log(f"turn {turn}: no block; runs {s['runs']}, passes {s['passes']}/{c['budget']}, tool-free True")
                break
            s["next"] = nxt
            self.save()
            self.log(f"turn {turn}: {outcome}; runs {s['runs']}, passes {s['passes']}/{c['budget']}, tool-free True")
        self.kernel.stop()


def create(args):
    run_dir = os.path.abspath(args.run)
    if os.path.exists(os.path.join(run_dir, "state.json")):
        raise SystemExit(f"{run_dir} already has a run; use --resume to continue it")
    task_dir = os.path.abspath(args.task)
    os.makedirs(os.path.join(run_dir, "work"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "notices"), exist_ok=True)
    stats = json.load(open(os.path.join(task_dir, "practice", "stats.json")))
    sys.path.insert(0, os.path.join(ROOT, "..", "clip_zs_sbir"))
    from src.data_config import UNSEEN_CLASSES
    task = json.load(open(os.path.join(task_dir, "task.json")))
    values = {"budget": args.budget, "max_runs": args.max_runs, "timeout": args.timeout, "mem_gb": args.mem_gb,
              "n_cat": stats["categories"], "n_sk": f"{stats['sketches']:,}", "n_ph": f"{stats['photos']:,}",
              "n_unseen": len(UNSEEN_CLASSES[task["dataset"]]),
              "work_quota_gb": f"{args.work_quota_gb:g}", "pass_gb": f"{args.pass_gb:g}"}
    message = Template(open(os.path.join(task_dir, "message.txt")).read()).substitute(values)
    if not args.explain:
        message = message.replace(EXPLAIN + "\n\n", "")
    with open(os.path.join(run_dir, "message.txt"), "w") as f:
        f.write(message)
    shutil.copy2(os.path.join(HERE, "instructions.md"), os.path.join(run_dir, "instructions.md"))
    cfg = {"task": task_dir, "budget": args.budget, "max_runs": args.max_runs, "timeout": args.timeout,
           "mem_gb": args.mem_gb, "model": args.model, "effort": args.effort, "handoff_every": args.handoff_every,
           "max_turns": args.max_turns, "turn_timeout": args.turn_timeout, "forward_timeout": args.forward_timeout,
           "gpu": args.gpu, "explain": args.explain, "backend": args.backend, "prompt": "sbir-v3",
           "pass_gb": args.pass_gb, "autocompact": args.autocompact,
           "checkpoints": args.checkpoints, "eval_checkpoints": args.eval_checkpoints,
           "min_free_ram_gb": args.min_free_ram_gb, "forward_ram_gb": args.forward_ram_gb,
           "forward_gpu_gb": args.forward_gpu_gb, "work_quota_gb": args.work_quota_gb, "min_disk_gb": args.min_disk_gb,
           "gpu_job_threads": args.gpu_job_threads, "forward_workers": args.forward_workers,
           "eval_workers": args.eval_workers, "eval_batch_size": args.eval_batch_size,
           "created": now(), "source_commit": source_revision()}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    state = {"budget": args.budget, "task": task["dataset"], "prompt": "sbir-v3", "turn": 0, "runs": 0, "passes": 0,
             "requests": 0, "thread": None, "done": False, "nudges": 0, "handoffs": 0, "since_handoff": 0,
             "flags": 0, "refusals": 0, "explain": args.explain, "unexplained_streak": 0, "unexplained": 0,
             "kernel_restarts": 0, "notices": [], "req_keys": {}, "next": ""}
    with open(os.path.join(run_dir, "state.json"), "w") as f:
        json.dump(state, f, indent=1)
    return run_dir


def extend(run_dir, extra, add_checkpoints, extra_runs=None, extra_turns=None):
    """Give a run more forward passes (also after its final answer), as a note from the experimenters.

    The pass-count state before the extension is kept: the submitted edit, final answer and evaluation become the
    checkpoint pass_<old budget>; the run's message.txt is updated to the new budget (it is re-sent at handoffs) and
    the old one kept as message_budget<old>.txt."""
    cfg_path, state_path = os.path.join(run_dir, "config.json"), os.path.join(run_dir, "state.json")
    cfg, s = json.load(open(cfg_path)), json.load(open(state_path))
    old, new = cfg["budget"], cfg["budget"] + extra
    old_runs = cfg["max_runs"]
    new_runs = old_runs + (extra_runs if extra_runs is not None else 6 * extra)
    tag = f"pass_{old:03d}"
    eval_dir = os.path.join(ROOT, "eval", os.path.basename(run_dir))
    if s["done"]:
        ck = os.path.join(run_dir, "checkpoints", tag)
        os.makedirs(ck, exist_ok=True)
        if os.path.isdir(os.path.join(run_dir, "edit")) and not os.path.exists(os.path.join(ck, "edit")):
            shutil.copytree(os.path.join(run_dir, "edit"), os.path.join(ck, "edit"))
        for src, dst in (("final_answer.md", f"final_answer_{tag}.md"), ("work/notes.md", None)):
            source = os.path.join(run_dir, src)
            if os.path.isfile(source) and not os.path.islink(source):
                shutil.copy2(source, os.path.join(ck, dst or "notes.md"))
        if os.path.exists(os.path.join(run_dir, "final_answer.md")):
            shutil.copy2(os.path.join(run_dir, "final_answer.md"), os.path.join(run_dir, f"final_answer_{tag}.md"))
        if os.path.exists(os.path.join(eval_dir, "final.json")) and not os.path.exists(os.path.join(eval_dir, f"{tag}.json")):
            shutil.copy2(os.path.join(eval_dir, "final.json"), os.path.join(eval_dir, f"{tag}.json"))
        with open(os.path.join(ck, "info.json"), "w") as f:
            json.dump({"label": tag, "passes": s["passes"], "turn": s["turn"], "python_runs": s["runs"],
                       "note": "the edit submitted at the end of the session before the extension", "time": now()},
                      f, indent=1)
    msg_path = os.path.join(run_dir, "message.txt")
    msg = open(msg_path).read()
    shutil.copy2(msg_path, os.path.join(run_dir, f"message_budget{old}.txt"))
    for a, b in ((f"a budget of {old} forward passes", f"a budget of {new} forward passes"),
                 (f"until all {old} forward passes are used", f"until all {new} forward passes are used"),
                 (f"up to {old_runs} times", f"up to {new_runs} times")):
        assert msg.count(a) == 1, a
        msg = msg.replace(a, b)
    with open(msg_path, "w") as f:
        f.write(msg)
    note_dir = os.path.join(run_dir, "notices")
    os.makedirs(note_dir, exist_ok=True)
    n = len([f for f in os.listdir(note_dir) if f.endswith(".txt")]) + 1
    note = (f"The session is extended. Your forward-pass budget is raised from {old} to {new} passes: you have "
            f"{new - s['passes']} more forward passes. Python runs are raised from {old_runs} to {new_runs} "
            f"({new_runs - s['runs']} left). Everything else is unchanged: the same rules, tools, limits and "
            "evaluation. Your current edit in /work/edited stays as it is unless you change it. Continue the work "
            f"with the new passes; when all {new} are used, make your final choice (the tensors in /work/edited "
            "when you finish are submitted) and reply without a code block, as before, saying what you changed, "
            "the evidence for it and why you expect it to help.")
    with open(os.path.join(note_dir, f"{n:03d}_budget_extension_{new}.txt"), "w") as f:
        f.write(note)
    cks = sorted({int(x) for x in str(cfg.get("checkpoints", "")).split(",") if x.strip()} | set(add_checkpoints))
    cfg.update(budget=new, max_runs=new_runs, checkpoints=",".join(map(str, cks)),
               max_turns=cfg["max_turns"] + (extra_turns if extra_turns is not None else 8 * extra))
    cfg.setdefault("extensions", []).append({"time": now(), "from": old, "to": new, "max_runs": new_runs,
                                             "checkpoints": cks})
    s.update(budget=new, done=False, next=f"Your final answer for {old} passes was received." if s["done"] else s["next"])
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    with open(state_path, "w") as f:
        json.dump(s, f, indent=1)
    return old, new


def _stop(*_):
    raise KeyboardInterrupt  # SIGINT/SIGTERM: leave through the finally block, which saves the state


def main():
    import signal
    signal.signal(signal.SIGINT, _stop)  # also when started from a background shell, where SIGINT is ignored
    signal.signal(signal.SIGTERM, _stop)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run folder, e.g. runs/b25_r1")
    p.add_argument("--resume", action="store_true", help="continue an existing run")
    p.add_argument("--extend-budget", type=int, default=0,
                   help="with --resume: give the run this many more forward passes (also after its final answer)")
    p.add_argument("--add-checkpoints", default="", help="with --extend-budget: more pass counts to evaluate at")
    p.add_argument("--task", default=os.path.join(ROOT, "tasks", "sketchy_2"))
    p.add_argument("--budget", type=int, default=25, help="forward passes (Exp 8 used 10, 25, 50)")
    p.add_argument("--max-runs", type=int, default=300, help="Python runs")
    p.add_argument("--timeout", type=int, default=1800, help="seconds per Python run")
    p.add_argument("--mem-gb", type=int, default=16, help="kernel address-space limit")
    p.add_argument("--backend", choices=BACKENDS, default="codex")
    p.add_argument("--model", help="editor model id (defaults to gpt-6.2-astra for Codex)")
    p.add_argument("--effort", help="low, medium, high, xhigh, max (defaults to max for Codex)")
    p.add_argument("--autocompact", type=int, default=None,
                   help="Claude Code auto-compact window in tokens (e.g. 1000000, as Exp 8's limit); default: CLI default")
    p.add_argument("--handoff-every", type=int, default=40, help="start a new conversation every N turns")
    p.add_argument("--max-turns", type=int, default=600)
    p.add_argument("--turn-timeout", type=int, default=7200, help="seconds per editor call")
    p.add_argument("--forward-timeout", type=int, default=7200, help="seconds per forward pass")
    p.add_argument("--gpu", default="0")
    p.add_argument("--no-explain", dest="explain", action="store_false",
                   help="leave out the EXPLAIN EVERY STEP paragraph and its enforcement")
    p.add_argument("--checkpoints", default="10,25", help="pass counts at which the edit is snapshotted and evaluated")
    p.add_argument("--no-eval-checkpoints", dest="eval_checkpoints", action="store_false",
                   help="snapshot checkpoints without evaluating them (evaluate later with eval/evaluate_edit.py)")
    p.add_argument("--min-free-ram-gb", type=float, default=8.0,
                   help="stop a Python cell if the machine's free RAM falls below this")
    p.add_argument("--forward-ram-gb", type=float, default=10.0, help="free RAM needed before a GPU job starts")
    p.add_argument("--forward-gpu-gb", type=float, default=6.0, help="free GPU memory needed before a GPU job starts")
    p.add_argument("--work-quota-gb", type=float, default=80.0, help="size quota of /work")
    p.add_argument("--pass-gb", type=float, default=40.0, help="most a single forward pass may write")
    p.add_argument("--min-disk-gb", type=float, default=50.0, help="free disk space below which passes are refused")
    p.add_argument("--gpu-job-threads", type=int, default=6)
    p.add_argument("--forward-workers", type=int, default=4)
    p.add_argument("--eval-workers", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    args = p.parse_args()
    defaults = ("gpt-6.2-astra", "max") if args.backend == "codex" else ("opus", "xhigh")
    args.model = args.model or defaults[0]
    args.effort = args.effort or defaults[1]
    if min(args.forward_workers, args.eval_workers) < 0 or min(args.gpu_job_threads, args.eval_batch_size) < 1:
        p.error("worker counts must be nonnegative, and thread count and batch size must be positive")
    run_dir = os.path.abspath(args.run) if args.resume else create(args)
    if args.resume and args.extend_budget:
        old, new = extend(run_dir, args.extend_budget,
                          [int(x) for x in args.add_checkpoints.split(",") if x.strip()])
    run = Run(run_dir)
    if args.resume and args.extend_budget:
        run.log(f"budget extended from {old} to {new} passes (note delivered with the next message)")
    run.log(f"start: {run.cfg['model']} effort {run.cfg['effort']}, budget {run.cfg['budget']}, "
            f"task {run.task['dataset']}, turn {run.state['turn']}")
    try:
        run.loop()
    finally:
        run.kernel.stop()
        run.save()
        import export_readable
        export_readable.export(run_dir)
    if run.state["done"]:
        completed = Path(run_dir).resolve()
        bundle = completed / "result.zip"
        archive_run(completed, bundle)
        print(f"Result bundle: {bundle}", flush=True)


if __name__ == "__main__":
    main()
