# Defined in advance in the kernel. Runs inside the sandbox: /model and /practice read-only, /work read-write.
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, "/opt/kernel")
from edit_rules import check_edit as _check_edit, affects_retrieval as _affects_retrieval

torch.set_grad_enabled(False)
np.set_printoptions(precision=4, suppress=True, linewidth=160)

MODEL, PRACTICE, WORK = "/model", "/practice", "/work"
EDIT_DIR = WORK + "/edited"
ARCH = json.load(open(MODEL + "/arch.json"))
NAMES = [t["name"] for t in ARCH["tensors"]]
SHAPES = {t["name"]: tuple(t["shape"]) for t in ARCH["tensors"]}
DTYPES = {t["name"]: t["model_dtype"] for t in ARCH["tensors"]}
EDITABLE = list(NAMES)  # every tensor can be edited
VISUAL = [n for n in NAMES if _affects_retrieval(n)]  # the image encoder: the only tensors the retrieval uses
CATEGORIES = json.load(open(PRACTICE + "/categories.json"))


def load_original(name):
    """The original values of a tensor, as a float32 torch tensor."""
    if name not in SHAPES:
        raise KeyError(f"unknown tensor {name!r}; see NAMES")
    return torch.from_numpy(np.load(f"{MODEL}/tensors/{name}.npy").astype(np.float32))


def load(name):
    """Your current edit of a tensor if there is one, else the original (float32 torch tensor)."""
    path = f"{EDIT_DIR}/{name}.npy"
    if os.path.exists(path):
        return torch.from_numpy(np.load(path).astype(np.float32))
    return load_original(name)


def save_edit(name, tensor):
    """Replace a tensor by your values (saved as float32 in /work/edited)."""
    if name not in SHAPES:
        raise KeyError(f"unknown tensor {name!r}; see NAMES")
    array = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    array = array.astype(np.float32)
    original = np.load(f"{MODEL}/tensors/{name}.npy").astype(np.float32)
    reason = _check_edit(name, array, original)
    if reason:
        raise ValueError("save_edit refused: " + reason)
    os.makedirs(EDIT_DIR, exist_ok=True)
    if np.array_equal(array, original):
        drop_edit(name)
        return f"{name}: identical to the original, no edit kept"
    np.save(f"{EDIT_DIR}/{name}.npy", array)
    changed = int((array != original).sum())
    rel = float(np.linalg.norm(array - original) / max(np.linalg.norm(original), 1e-30))
    return f"{name}: saved ({changed:,} entries changed, relative change {rel:.4g}; evaluated as {DTYPES[name]})"


def drop_edit(name):
    """Remove your edit of a tensor (it returns to its original values)."""
    path = f"{EDIT_DIR}/{name}.npy"
    if os.path.exists(path):
        os.remove(path)


def edited():
    """Names of the tensors you have edited."""
    if not os.path.isdir(EDIT_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(EDIT_DIR) if f.endswith(".npy"))


def open_npy(path):
    """A large .npy file, memory-mapped (read-only): slicing it reads only the slice from disk."""
    return np.load(path, mmap_mode="r")


def chunks(array, rows=8192):
    """Iterate over (start, float32 block) of an array or memory-mapped file, `rows` rows at a time."""
    for start in range(0, len(array), rows):
        yield start, np.asarray(array[start:start + rows], dtype=np.float32)
