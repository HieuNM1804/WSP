"""Harness side of the persistent sandbox kernel (bubblewrap: no network, no GPU, clean environment).

  /model     read-only  model/tensors/*.npy, model/arch.json
  /practice  read-only  index.jsonl, categories.json, stats.json, sketch/<category>/, photo/<category>/ (seen only)
  /work      read-write the run's work folder, kept between runs and across kernel restarts
  /opt/kernel read-only kernel.py, prelude.py, edit_rules.py
"""
import json
import os
import resource
import select
import signal
import subprocess
import time

from resources import mem_available_gb, protect_others, tree_rss_gb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SANDBOX_VENV = os.environ.get("WSP_SANDBOX_VENV", os.path.join(ROOT, ".venv_sandbox"))
UV_PYTHONS = os.path.expanduser("~/.local/share/uv/python")


def ascii_only(text):
    return text.encode("ascii", "replace").decode("ascii")


class Kernel:
    def __init__(self, work_dir, practice_dir, dataset_root, categories, timeout=900, mem_gb=16, max_out=10_000,
                 threads=8, min_free_ram_gb=8.0, log=None):
        self.work_dir = os.path.abspath(work_dir)
        self.practice_dir = os.path.abspath(practice_dir)
        self.dataset_root = os.path.abspath(dataset_root)
        self.categories = categories
        self.timeout, self.mem, self.max_out, self.threads = timeout, int(mem_gb) << 30, max_out, threads
        self.min_free_ram_gb = min_free_ram_gb
        self.log = log or (lambda text: None)
        self.proc = None
        self.starts = 0
        self.peak_rss_gb = 0.0

    def _cmd(self):
        cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
               "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
               "--symlink", "usr/bin", "/bin", "--dev", "/dev", "--tmpfs", "/tmp",
               "--ro-bind", SANDBOX_VENV, "/opt/venv",
               "--ro-bind", os.path.join(HERE, "sandbox_lib"), "/opt/kernel",
               "--ro-bind", os.path.join(ROOT, "model"), "/model",
               "--dir", "/practice"]
        if os.environ.get("WSP_SANDBOX_PROC", "1") != "0":
            cmd += ["--proc", "/proc"]
        if os.path.isdir(UV_PYTHONS):
            cmd += ["--ro-bind", UV_PYTHONS, UV_PYTHONS]
        for name in ("index.jsonl", "categories.json", "stats.json"):
            cmd += ["--ro-bind", os.path.join(self.practice_dir, name), f"/practice/{name}"]
        for mode in ("sketch", "photo"):
            for category in self.categories:
                cmd += ["--ro-bind", os.path.join(self.dataset_root, mode, category), f"/practice/{mode}/{category}"]
        cmd += ["--bind", self.work_dir, "/work", "--chdir", "/work",
                "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin",
                "--setenv", "OMP_NUM_THREADS", str(self.threads), "--setenv", "MKL_NUM_THREADS", str(self.threads),
                "--setenv", "CUDA_VISIBLE_DEVICES", "",
                "/opt/venv/bin/python", "-I", "/opt/kernel/kernel.py"]
        return cmd

    def _limits(self):
        resource.setrlimit(resource.RLIMIT_AS, (self.mem, self.mem))
        protect_others()

    def start(self):
        self.stop()
        self.proc = subprocess.Popen(self._cmd(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1, preexec_fn=self._limits)
        self.starts += 1

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self.proc.kill()
            self.proc.wait()
        self.proc = None

    def run(self, code):
        """Run one cell. Returns (output text, restarted: bool, seconds)."""
        restarted_before = False
        if not self.alive():
            self.start()
            restarted_before = self.starts > 1
        t0 = time.time()
        note = ""
        try:
            self.proc.stdin.write(json.dumps({"code": code}) + "\n")
            self.proc.stdin.flush()
            ready, low_memory = [], None
            while not ready and time.time() - t0 < self.timeout:
                ready, _, _ = select.select([self.proc.stdout], [], [], 2.0)
                if ready:
                    break
                rss = tree_rss_gb(self.proc.pid)
                self.peak_rss_gb = max(self.peak_rss_gb, rss)
                ram, _ = mem_available_gb()
                if ram < self.min_free_ram_gb and rss >= 2.0:
                    low_memory = (ram, rss)
                    break
            if low_memory:
                note = (f"[stopped: the machine's free memory fell to {low_memory[0]:.1f} GB while this cell ran "
                        f"(the kernel held {low_memory[1]:.1f} GB; the machine is shared). The kernel was restarted "
                        "and its variables are lost. Use less memory: load fewer arrays at once, use float16 or "
                        "memory-mapped arrays (np.load(..., mmap_mode='r')), and delete what you no longer need.]")
                self.log(f"kernel watchdog: free RAM {low_memory[0]:.1f} GB < {self.min_free_ram_gb} GB with the "
                         f"kernel at {low_memory[1]:.1f} GB; cell stopped")
                out = ""
            elif not ready:
                note = f"[timed out after {self.timeout} s: the kernel was restarted and its variables are lost]"
                out = ""
            else:
                line = self.proc.stdout.readline()
                if line:
                    out = json.loads(line)["out"]
                else:
                    self.proc.wait()
                    err = self.proc.stderr.read()[-2000:]
                    note = (f"[the kernel died (exit code {self.proc.returncode}; the memory limit is "
                            f"{self.mem >> 30} GB): it was restarted and its variables are lost]")
                    if err.strip():
                        note += "\n[stderr]\n" + err
                    out = ""
        except (BrokenPipeError, OSError) as exc:
            out, note = "", f"[the kernel could not be reached ({exc}): it was restarted and its variables are lost]"
        seconds = round(time.time() - t0, 2)
        if note:
            self.stop()
            self.start()
            out = (out + "\n" + note).strip()
        out = ascii_only(out)
        if len(out) > self.max_out:
            out = out[:self.max_out] + f"\n[output cut at {self.max_out:,} characters]"
        return (out or "(no output)"), bool(note) or restarted_before, seconds
