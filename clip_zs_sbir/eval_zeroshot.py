"""Zero-shot SBIR evaluation of the original (frozen, un-prompted) CLIP ViT-B/32.

No training. Data loading, unseen splits, preprocessing, feature normalisation
and metrics follow KD-SBIR (https://github.com/HieuNM1804/KD-SBIR,
commit 741ea272b258c9e7b92a46173618e2fe1d401eaa):

- src/data_config.py, src/dataset.py, clip/ : copied unchanged.
- src/metrics.py : _retrieval_metrics copied unchanged from src/model.py.
- Model loading mirrors src/model.py::_load_clip_model (jit -> state_dict ->
  build_model, fp16 weights), but reads a local checkpoint path.
- Feature extraction mirrors encode_student_image with no visual prompts:
  clip_model.visual(image.type(dtype)) followed by L2 normalisation.
- Loaders mirror src/train.py::get_loaders (validation part) and
  seed_everything (deterministic settings).

Usage:
    python eval_zeroshot.py --root /path/Sketchy --dataset sketchy_1
    python eval_zeroshot.py --root /path/Sketchy --dataset sketchy_2
    python eval_zeroshot.py --root /path/tuberlin --dataset tuberlin
"""
import os

# Same as KD-SBIR src/train.py: must be set before importing torch.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from clip.model import build_model
from src.data_config import UNSEEN_CLASSES
from src.dataset import ValidDataset
from src.metrics import _retrieval_metrics

DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "ViT-B-32.pt"
)


def seed_everything(seed):
    """Copied from KD-SBIR src/train.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(_worker_id):
    """Copied from KD-SBIR src/train.py."""
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_clip_model(model_path):
    """KD-SBIR src/model.py::_load_clip_model, with a local checkpoint path."""
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    return build_model(state_dict)


def get_val_loaders(args):
    """Validation part of KD-SBIR src/train.py::get_loaders."""
    seed_everything(args.seed)

    val_sketch = ValidDataset(args, mode="sketch")
    val_photo = ValidDataset(args)

    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        worker_init_fn=seed_worker,
    )
    val_sketch_loader = DataLoader(
        dataset=val_sketch,
        batch_size=args.test_batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 1),
        **loader_kwargs,
    )
    val_photo_loader = DataLoader(
        dataset=val_photo,
        batch_size=args.test_batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 2),
        **loader_kwargs,
    )
    return val_sketch_loader, val_photo_loader


def check_split(args):
    """Fail loudly if an unseen class has no folder (ValidDataset would skip it)."""
    for mode in ("sketch", "photo"):
        missing = [
            c for c in UNSEEN_CLASSES[args.dataset]
            if not os.path.isdir(os.path.join(args.root, mode, c))
        ]
        if missing:
            raise FileNotFoundError(
                f"[{args.dataset}] unseen classes missing under "
                f"{os.path.join(args.root, mode)}: {missing}"
            )


@torch.no_grad()
def extract_features(clip_model, loader, device, desc, show_progress):
    """encode_student_image without prompts: visual(x.type(dtype)) then L2-normalise."""
    dtype = clip_model.dtype
    features, labels = [], []
    for images, current_labels in tqdm(loader, desc=desc, disable=not show_progress):
        images = images.to(device, non_blocking=True)
        feat = clip_model.visual(images.type(dtype))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        features.append(feat)
        labels.append(current_labels)
    return torch.cat(features), torch.cat(labels).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Dataset root containing sketch/ and photo/.")
    parser.add_argument("--dataset", type=str, default="sketchy_1",
                        choices=sorted(UNSEEN_CLASSES), help="Zero-shot split.")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help="Path to the OpenAI CLIP ViT-B-32.pt checkpoint.")
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=str, default="",
                        help="Optional JSON file to write the metrics to.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required: KD-SBIR evaluates on GPU with fp16 CLIP weights."
        )
    device = torch.device("cuda")

    check_split(args)
    val_sketch_loader, val_photo_loader = get_val_loaders(args)
    print(
        f"[{args.dataset}] unseen classes={len(UNSEEN_CLASSES[args.dataset])}, "
        f"sketches={len(val_sketch_loader.dataset)}, "
        f"photos={len(val_photo_loader.dataset)}"
    )

    clip_model = load_clip_model(args.weights)
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)
    clip_model = clip_model.to(device).eval()

    sketch_features, sketch_labels = extract_features(
        clip_model, val_sketch_loader, device, "sketch", args.progress
    )
    photo_features, photo_labels = extract_features(
        clip_model, val_photo_loader, device, "photo", args.progress
    )

    mAP, precision, map_k, p_k = _retrieval_metrics(
        sketch_features,
        photo_features,
        sketch_labels,
        photo_labels,
        args.dataset,
    )
    map_name = f"mAP@{map_k}" if map_k else "mAP@all"
    print(f"[{args.dataset}] {map_name}: {mAP.item():.4f}, P@{p_k}: {precision.item():.4f}")

    if args.output:
        result = {
            "dataset": args.dataset,
            "weights": os.path.abspath(args.weights),
            "num_sketches": len(sketch_labels),
            "num_photos": len(photo_labels),
            map_name: mAP.item(),
            f"P@{p_k}": precision.item(),
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
