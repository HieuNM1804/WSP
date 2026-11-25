"""Prepare the read-only material a WSP-SBIR run mounts in its sandbox.

  model/tensors/<name>.npy   every CLIP ViT-B/32 tensor, values exactly as stored in ViT-B-32.pt
  model/arch.json            names, shapes, stored dtype, dtype in the evaluated model, layout notes
  tasks/<dataset>/practice/  index.jsonl, categories.json, stats.json: the SEEN categories of the split
  data/<name>_clean/         (only if needed) a symlink view of the dataset without stray entries

The seen categories follow KD-SBIR's TrainDataset exactly:
  sorted(set(os.listdir(root/sketch)) - set(UNSEEN_CLASSES[dataset]) - {".ipynb_checkpoints"})
and the files of a category are sorted(glob(root/<mode>/<category>/*)) (regular files only).

Run with the CUDA venv of clip_zs_sbir:
  ../clip_zs_sbir/.venv/bin/python prepare.py --root ../Sketchy --dataset sketchy_2
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "clip_zs_sbir"))

from src.data_config import UNSEEN_CLASSES  # noqa: E402

META_KEYS = {"input_resolution", "context_length", "vocab_size"}

LAYOUT = {
    "residual_stream": "the vision transformer works in [tokens, batch, 768] (LND); token 0 is the class token, tokens 1..49 are the 7x7 patches in row-major order",
    "visual.conv1.weight": "[768, 3, 32, 32] patch embedding (no bias), applied to the normalized 224x224 RGB image with stride 32",
    "visual.class_embedding": "[768] class token, added before positional embeddings",
    "visual.positional_embedding": "[50, 768]",
    "visual.ln_pre": "LayerNorm on all tokens before block 0",
    "visual.transformer.resblocks.<i>": "pre-LN block: x = x + out_proj(MHA(ln_1(x))); x = x + c_proj(QuickGELU(c_fc(ln_2(x)))), QuickGELU(z) = z * sigmoid(1.702 z)",
    "attn.in_proj_weight": "[2304, 768] = rows [q; k; v], 768 rows each, each split into 12 heads of 64 rows (head h = rows 64h..64h+63); in_proj_bias [2304] likewise",
    "attn.out_proj.weight": "[768, 768]; input columns 64h..64h+63 read head h's output",
    "mlp.c_fc.weight": "[3072, 768] (neuron j = row j), c_fc.bias [3072]",
    "mlp.c_proj.weight": "[768, 3072] (neuron j writes column j), c_proj.bias [768]",
    "visual.ln_post": "LayerNorm on the class token after block 11",
    "visual.proj": "[768, 512]: embedding = ln_post(x[0]) @ proj; retrieval uses the L2-normalized embedding",
    "text": "the text tower (transformer.*, token_embedding, positional_embedding, ln_final, text_projection) and logit_scale are not used by the image retrieval",
}


def export_model(weights, out_dir):
    import numpy as np
    import torch

    from clip.model import build_model

    tensor_dir = os.path.join(out_dir, "tensors")
    os.makedirs(tensor_dir, exist_ok=True)
    jit = torch.jit.load(weights, map_location="cpu").eval()
    state_dict = {k: v for k, v in jit.state_dict().items()}
    model = build_model(dict(state_dict))
    model_dtypes = {k: str(v.dtype).replace("torch.", "") for k, v in model.state_dict().items()}

    tensors = []
    for name, value in state_dict.items():
        if name in META_KEYS:
            continue
        array = value.cpu().numpy()
        np.save(os.path.join(tensor_dir, name + ".npy"), array)
        tensors.append({
            "name": name,
            "shape": list(array.shape),
            "stored_dtype": str(array.dtype),
            "model_dtype": model_dtypes[name],
            "editable": name.startswith("visual."),
        })
    arch = {
        "model": "CLIP ViT-B/32 (OpenAI release, ViT-B-32.pt)",
        "image_size": 224, "patch_size": 32, "width": 768, "layers": 12, "heads": 12,
        "head_dim": 64, "mlp_width": 3072, "embed_dim": 512,
        "layout": LAYOUT,
        "tensors": tensors,
    }
    with open(os.path.join(out_dir, "arch.json"), "w") as f:
        json.dump(arch, f, indent=1)
    print(f"model: {len(tensors)} tensors -> {out_dir}")


def list_files(root, mode, category):
    directory = os.path.join(root, mode, category)
    if not os.path.isdir(directory):
        return [], []
    files, stray = [], []
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.name.startswith("."):
                (files if entry.is_file() else stray).append(entry.path)
    return sorted(files), sorted(stray)


def build_practice(root, dataset, out_dir):
    sketch_root = os.path.join(root, "sketch")
    excluded = set(UNSEEN_CLASSES[dataset]) | {".ipynb_checkpoints"}
    categories = sorted(set(os.listdir(sketch_root)) - excluded)
    os.makedirs(out_dir, exist_ok=True)
    stats = {"dataset": dataset, "categories": len(categories), "sketches": 0, "photos": 0, "skipped_non_files": []}
    with open(os.path.join(out_dir, "index.jsonl"), "w") as f:
        for category in categories:
            for mode in ("sketch", "photo"):
                files, stray = list_files(root, mode, category)
                stats["skipped_non_files"] += [os.path.relpath(p, root) for p in stray]
                if not files:
                    raise RuntimeError(f"no {mode} files for practice category {category}")
                for path in files:
                    rel = os.path.relpath(path, root).replace(os.sep, "/")
                    f.write(json.dumps({"id": rel, "modality": mode, "category": category}) + "\n")
                    stats["sketches" if mode == "sketch" else "photos"] += 1
    with open(os.path.join(out_dir, "categories.json"), "w") as f:
        json.dump(categories, f, indent=0)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"practice[{dataset}]: {stats['categories']} categories, {stats['sketches']} sketches, {stats['photos']} photos")


def build_clean_view(root, out_root):
    """Symlink view of root/{sketch,photo}/<category>/<file> keeping regular files only."""
    stray = []
    for mode in ("sketch", "photo"):
        for category in sorted(os.listdir(os.path.join(root, mode))):
            src = os.path.join(root, mode, category)
            if not os.path.isdir(src):
                continue
            with os.scandir(src) as entries:
                files, bad = [], []
                for entry in entries:
                    (files if entry.is_file() else bad).append(entry.name)
            stray += [os.path.join(mode, category, b) for b in sorted(bad)]
            dst = os.path.join(out_root, mode, category)
            if os.path.lexists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if bad:
                os.makedirs(dst)
                for name in files:
                    os.symlink(os.path.join(src, name), os.path.join(dst, name))
            else:
                os.symlink(src, dst)
    return stray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="dataset root with sketch/ and photo/")
    parser.add_argument("--dataset", required=True, choices=sorted(UNSEEN_CLASSES))
    parser.add_argument("--weights", default=os.path.join(HERE, "..", "ViT-B-32.pt"))
    args = parser.parse_args()
    root = os.path.abspath(args.root)

    model_dir = os.path.join(HERE, "model")
    if not os.path.exists(os.path.join(model_dir, "arch.json")):
        export_model(args.weights, model_dir)
    build_practice(root, args.dataset, os.path.join(HERE, "tasks", args.dataset, "practice"))

    clean = os.path.join(HERE, "data", os.path.basename(root.rstrip("/")) + "_clean")
    stray = build_clean_view(root, clean)
    task_path = os.path.join(HERE, "tasks", args.dataset, "task.json")
    task = {
        "dataset": args.dataset,
        "root": root,
        "eval_root": clean,
        "weights": os.path.abspath(args.weights),
        "stray_entries_skipped_in_eval_root": stray,
    }
    with open(task_path, "w") as f:
        json.dump(task, f, indent=1)
    print(f"task -> {task_path}; eval root {clean} (stray entries skipped: {stray})")


if __name__ == "__main__":
    main()
