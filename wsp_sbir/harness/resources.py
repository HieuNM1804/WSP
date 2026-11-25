"""System resource checks, so that a long run cannot take the shared machine down.

The machine is shared (vLLM servers, databases, other users' GPU jobs) and its swap is often full, so:
  - every process the harness starts (kernel, forward pass, evaluation) is niced and marked as the OOM killer's
    first choice (oom_score_adj 1000): if memory runs out, our process dies, not someone else's service;
  - the kernel watchdog stops a cell when the machine's free memory falls below a floor (kernel_client.py);
  - forward passes and evaluations wait until enough GPU memory and RAM are free (driver.py);
  - /work has a size quota, and the disk must keep a minimum of free space.
"""
import os
import shutil
import subprocess
import time


def mem_available_gb():
    with open("/proc/meminfo") as f:
        info = {line.split(":")[0]: int(line.split()[1]) for line in f}
    return info["MemAvailable"] / 2**20, info["SwapFree"] / 2**20


def gpu_free_gb(index=0):
    try:
        out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=memory.free,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        free, used, util = (float(x) for x in out.strip().split(","))
        return free / 1024, used / 1024, util
    except Exception:  # noqa: BLE001 - treat an unreadable GPU as unknown, not as full
        return None, None, None


def disk_free_gb(path):
    return shutil.disk_usage(path).free / 2**30


def dir_size_gb(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total / 2**30


def tree_rss_gb(pid):
    """Resident memory of a process and all its descendants."""
    total, todo, seen = 0, [pid], set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            with open(f"/proc/{p}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
            with open(f"/proc/{p}/task/{p}/children") as f:
                todo += [int(c) for c in f.read().split()]
        except OSError:
            pass
    return total / 2**20


def protect_others(nice=10):
    """preexec_fn for our child processes: lower priority and volunteer to the OOM killer."""
    try:
        os.nice(nice)
    except OSError:
        pass
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


def snapshot(work=None, gpu=0):
    ram, swap = mem_available_gb()
    gfree, gused, gutil = gpu_free_gb(gpu)
    snap = {"t": time.time(), "ram_available_gb": round(ram, 2), "swap_free_gb": round(swap, 2),
            "gpu_free_gb": None if gfree is None else round(gfree, 2),
            "gpu_used_gb": None if gused is None else round(gused, 2), "gpu_util": gutil,
            "load1": round(os.getloadavg()[0], 2)}
    if work:
        snap["disk_free_gb"] = round(disk_free_gb(work), 1)
        snap["work_gb"] = round(dir_size_gb(work), 2)
    return snap


def wait_for(need_ram_gb, need_gpu_gb, gpu=0, log=print, what="the forward pass", poll=60, max_wait=None):
    """Block until the RAM and GPU floors are met. Returns seconds waited."""
    t0 = time.time()
    last_log = 0.0
    while True:
        ram, _ = mem_available_gb()
        gfree = gpu_free_gb(gpu)[0] if need_gpu_gb else None
        ok = ram >= need_ram_gb and (not need_gpu_gb or gfree is None or gfree >= need_gpu_gb)
        if ok:
            return round(time.time() - t0, 1)
        waited = time.time() - t0
        if max_wait is not None and waited >= max_wait:
            return round(waited, 1)
        if time.time() - last_log >= 600:
            log(f"waiting for resources before {what}: RAM available {ram:.1f} GB (need {need_ram_gb}), "
                f"GPU free {gfree if gfree is None else round(gfree, 1)} GB (need {need_gpu_gb})")
            last_log = time.time()
        time.sleep(poll)
