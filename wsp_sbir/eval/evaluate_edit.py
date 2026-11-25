"""Evaluate a run's submitted edit on the unseen split, with KD-SBIR's protocol (never shown to the editor).

  ../clip_zs_sbir/.venv/bin/python eval/evaluate_edit.py --run runs/b25_r1
  ../clip_zs_sbir/.venv/bin/python eval/evaluate_edit.py --original        # the unedited baseline

The pipeline is clip_zs_sbir/eval_zeroshot.py (KD-SBIR's ValidDataset, loaders, seeds, feature extraction and
_retrieval_metrics, copied unchanged); the only addition is that the edited tensors are cast to each parameter's
dtype and copied into the model built by build_model. The dataset root is the task's eval_root (the dataset with
non-file entries such as sketch/cow/_contact_sheets left out, see prepare.py).
Writes eval/<run name>/result.json (or eval/unedited/result.json).
"""
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "..", "clip_zs_sbir"))
sys.path.insert(0, os.path.join(ROOT, "harness", "sandbox_lib"))

import eval_zeroshot as ez  # noqa: E402
from src.data_config import UNSEEN_CLASSES  # noqa: E402
from edit_rules import check_edit  # noqa: E402


def apply_edit(model, edit_dir):
    params = dict(model.state_dict(keep_vars=True))
    applied = []
    for f in sorted(os.listdir(edit_dir)):
        if not f.endswith(".npy"):
            continue
        name = f[:-4]
        array = np.load(os.path.join(edit_dir, f)).astype(np.float32)
        original = np.load(os.path.join(ROOT, "model", "tensors", name + ".npy")).astype(np.float32)
        reason = check_edit(name, array, original)
        if reason:
            raise ValueError(f"{f}: {reason}")
        with torch.no_grad():
            params[name].data.copy_(torch.from_numpy(array).to(params[name].dtype))
        applied.append(name)
    return applied


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="run folder; its edit/tensors (or work/edited) is evaluated")
    p.add_argument("--edit-dir", help="evaluate this folder of <tensor>.npy files instead")
    p.add_argument("--original", action="store_true", help="evaluate the unedited model")
    p.add_argument("--task", default=os.path.join(ROOT, "tasks", "sketchy_2"))
    p.add_argument("--workers", type=int, default=2,
                   help="DataLoader workers (results do not depend on it; 8 workers x prefetch 4 x batch 1024 keep about "
                        "20 GB of images in memory, too much for the shared machine)")
    p.add_argument("--name", default=None, help="label stored in the result")
    p.add_argument("--test_batch_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    task = json.load(open(os.path.join(a.task, "task.json")))
    if a.original:
        edit_dir, name = None, f"unedited_{task['dataset']}"
    elif a.edit_dir:
        edit_dir, name = os.path.abspath(a.edit_dir), os.path.basename(os.path.abspath(a.edit_dir))
    elif a.run:
        run = os.path.abspath(a.run)
        edit_dir = os.path.join(run, "edit", "tensors")
        if not os.path.isdir(edit_dir):
            edit_dir = os.path.join(run, "work", "edited")
            print(f"[warning] {run} has no submitted edit yet; evaluating its current work/edited")
        name = os.path.basename(run)
    else:
        raise SystemExit("give --run, --edit-dir or --original")

    args = SimpleNamespace(root=task["eval_root"], dataset=task["dataset"], max_size=224,
                           test_batch_size=a.test_batch_size, workers=a.workers, seed=a.seed)
    ez.check_split(args)
    sketch_loader, photo_loader = ez.get_val_loaders(args)
    device = torch.device("cuda")
    model = ez.load_clip_model(task["weights"])
    applied = apply_edit(model, edit_dir) if edit_dir else []
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = model.to(device).eval()
    sf, sl = ez.extract_features(model, sketch_loader, device, "sketch", False)
    pf, pl = ez.extract_features(model, photo_loader, device, "photo", False)
    mAP, prec, map_k, p_k = ez._retrieval_metrics(sf, pf, sl, pl, args.dataset)
    map_name = f"mAP@{map_k}" if map_k else "mAP@all"
    result = {"run": a.name or name, "dataset": args.dataset, "unseen_classes": len(UNSEEN_CLASSES[args.dataset]),
              "edited_tensors": applied, "num_sketches": len(sl), "num_photos": len(pl),
              map_name: mAP.item(), f"P@{p_k}": prec.item()}
    print(json.dumps(result, indent=1))
    out = a.out or os.path.join(HERE, name, "result.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
