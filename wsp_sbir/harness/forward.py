"""Forward-pass tool: runs CLIP ViT-B/32 (original and/or edited) on the GPU, outside the sandbox.

Called by the driver once per request:
  <clip_zs_sbir venv>/bin/python harness/forward.py --task tasks/sketchy_2 --work runs/<run>/work \
      --k <request number> --request <requests/fwd_k.json.claimed> --out <requests/fwd_k.done.json> --max-bytes N

Model loading, preprocessing, feature extraction and the retrieval metric are KD-SBIR's (via clip_zs_sbir):
build_model from the jit state dict (fp16), Resize((224,224)) + CLIP mean/std, visual(x.type(dtype)) then
L2 normalisation, and the per-query loop of _retrieval_metrics with the split's mAP@k / P@k rule.

Memory: images are streamed through the GPU in batches; everything kept per image (activations, features) is
written batch by batch into memory-mapped .npy files, so the process's RAM does not grow with the number of images.
The size of a pass's output is bounded by --max-bytes (the driver passes min(pass limit, room left in the quota)).
Three pass types: retrieve, read, stats (per-group means and within-group covariances accumulated on the GPU).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.nn import functional as F
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "..", "clip_zs_sbir"))
sys.path.insert(0, os.path.join(HERE, "sandbox_lib"))

from clip import clip as clip_lib  # noqa: E402
from clip.model import build_model  # noqa: E402
from src.dataset import load_image, normal_transform  # noqa: E402
from edit_rules import check_edit  # noqa: E402

LIMITS = {"read_images": 16384, "read_texts": 4096, "attn_images": 256}
N_RESIDUAL, N_BLOCKS = 13, 12
WIDTH, MLP_WIDTH, EMBED, TOKENS = 768, 3072, 512, 50
BATCH = 128
WORKERS = int(os.environ.get("WSP_FORWARD_WORKERS", "4"))
OOM_WAIT_S, OOM_TRIES = 30, 40  # the GPU is shared with jobs whose memory use swings by 25 GB within seconds


def is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


def gpu_retry(fn, what="a GPU step"):
    """Run fn(); on CUDA out-of-memory, free the cache, wait and retry the same step (results are unchanged)."""
    for attempt in range(1, OOM_TRIES + 1):
        try:
            return fn()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not is_oom(exc) or attempt == OOM_TRIES:
                raise
            torch.cuda.empty_cache()
            print(f"[gpu_retry] out of memory in {what} (attempt {attempt}); waiting {OOM_WAIT_S} s", flush=True)
            time.sleep(OOM_WAIT_S)


class RequestError(Exception):
    pass


def deterministic():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# --------------------------------------------------------------------------- metrics
def split_rule(dataset):
    """Copied from KD-SBIR _retrieval_metrics."""
    if dataset == "sketchy_2":
        return 200, 200
    return 0, (200 if dataset == "quickdraw" else 100)


def per_query_metrics(query_features, gallery_features, query_labels, gallery_labels, dataset):
    """The loop body of KD-SBIR _retrieval_metrics, returning the per-query values instead of their means."""
    query_labels = query_labels.cpu()
    gallery_labels = gallery_labels.cpu()
    ap = torch.zeros(len(query_features))
    precision_at_k = torch.zeros(len(query_features))
    map_k, p_k = split_rule(dataset)
    for index, query_feature in enumerate(query_features):
        cosine = gpu_retry(lambda: F.cosine_similarity(query_feature.unsqueeze(0), gallery_features).cpu(),
                           "the metric loop")
        score = ((cosine + 1.0) * 0.5).clamp(min=torch.finfo(cosine.dtype).eps, max=1.0)
        target = gallery_labels.eq(query_labels[index])
        if map_k:
            ap[index] = retrieval_average_precision(score, target, top_k=min(map_k, len(gallery_features)))
        else:
            ap[index] = retrieval_average_precision(score, target)
        precision_at_k[index] = retrieval_precision(score, target, top_k=p_k)
    return ap.numpy().astype(np.float64), precision_at_k.numpy().astype(np.float64), map_k, p_k


def metric_names(dataset):
    map_k, p_k = split_rule(dataset)
    return (f"mAP@{map_k}" if map_k else "mAP@all"), f"P@{p_k}"


# --------------------------------------------------------------------------- models
def edit_files(work):
    d = os.path.join(work, "edited")
    if not os.path.isdir(d):
        return {}
    return {f[:-4]: os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".npy")}


def build(weights, edits=None, original_dir=None):
    """KD-SBIR's build_model; edits {name: path to float32 .npy} are cast to each parameter's dtype."""
    jit = torch.jit.load(weights, map_location="cpu").eval()
    model = build_model(dict(jit.state_dict()))
    del jit
    applied = []
    if edits:
        params = dict(model.state_dict(keep_vars=True))
        for name, path in edits.items():
            if name not in params:
                raise RequestError(f"/work/edited/{name}.npy: unknown tensor name")
            array = np.load(path)
            original = np.load(os.path.join(original_dir, name + ".npy")).astype(np.float32)
            reason = check_edit(name, array, original)
            if reason:
                raise RequestError(f"/work/edited/{name}.npy refused: {reason}")
            with torch.no_grad():
                params[name].data.copy_(torch.from_numpy(array.astype(np.float32)).to(params[name].dtype))
            applied.append(name)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval(), applied


# --------------------------------------------------------------------------- capture
class Capture:
    """Forward hooks on one model's image encoder. Every captured tensor goes to self.sink(key, x) with
    key = (kind, index, part): ("residual", layer 0..12, part), ("mlp", block, part), ("attn", block, "probs"),
    ("pre", 0, "vec") (ln_post output) and ("emb", 0, "vec") (normalized embedding, sent by the encode loop).
    part: "cls" (class token), "patch" (mean over the 49 patch tokens) or "all" (all 50 tokens)."""

    def __init__(self, model, residual=(), mlp_blocks=(), attn_blocks=(), parts=("cls", "patch")):
        self.visual = model.visual
        self.parts = tuple(parts)
        self.sink = None
        self.handles = []
        blocks = self.visual.transformer.resblocks
        if 0 in residual:
            self.handles.append(self.visual.ln_pre.register_forward_hook(self._tokens("residual", 0, nld=True)))
        for layer in residual:
            if layer > 0:
                self.handles.append(blocks[layer - 1].register_forward_hook(self._tokens("residual", layer)))
        for b in mlp_blocks:
            self.handles.append(blocks[b].mlp.gelu.register_forward_hook(self._tokens("mlp", b)))
        for b in attn_blocks:
            self.handles.append(blocks[b].attn.register_forward_hook(self._attn(b)))
        self.handles.append(self.visual.ln_post.register_forward_hook(
            lambda _m, _i, o: self.sink(("pre", 0, "vec"), o)))

    def _tokens(self, kind, index, nld=False):
        def hook(_module, _inputs, output):
            x = output if nld else output.permute(1, 0, 2)  # -> [N, 50, D]
            for part in self.parts:
                if part == "all":
                    self.sink((kind, index, "all"), x)
                elif part == "cls":
                    self.sink((kind, index, "cls"), x[:, 0])
                elif part == "patch":
                    self.sink((kind, index, "patch"), x[:, 1:].float().mean(dim=1))
        return hook

    def _attn(self, index):
        state = {"inside": False}

        def hook(module, inputs, _output):
            if state["inside"]:
                return
            state["inside"] = True
            try:  # a second call only to read the probabilities; the block's output is left unchanged
                q, k, v = inputs[:3]
                _, weights = module(q, k, v, need_weights=True, average_attn_weights=False)
            finally:
                state["inside"] = False
            self.sink(("attn", index, "probs"), weights)  # [N, heads, 50 queries, 50 keys]
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def file_name(key):
    kind, _index, part = key
    return kind if part == "vec" else f"{kind}_{part}"


class MemmapSink:
    """Writes captured tensors into [n, layers, ...] float16 memory-mapped .npy files, batch by batch."""

    def __init__(self, out_dir, prefix, n, layers):
        self.out_dir, self.prefix, self.n, self.layers = out_dir, prefix, n, layers  # layers: {kind: [indices]}
        self.files, self.offset, self.batch = {}, 0, 0

    def __call__(self, key, x):
        kind, index, _part = key
        arr = x.detach().float().cpu().numpy().astype(np.float16)
        name = file_name(key)
        if name not in self.files:
            per = tuple(arr.shape[1:])
            shape = (self.n,) + per if kind in ("pre", "emb") else (self.n, len(self.layers[kind])) + per
            self.files[name] = open_memmap(os.path.join(self.out_dir, f"{self.prefix}{name}.npy"), mode="w+",
                                           dtype=np.float16, shape=shape)
        target = self.files[name]
        rows = slice(self.offset, self.offset + len(arr))
        if kind in ("pre", "emb"):
            target[rows] = arr
        else:
            target[rows, self.layers[kind].index(index)] = arr
        self.batch = len(arr)

    def end_batch(self):
        self.offset += self.batch

    def close(self):
        meta = {}
        for name, arr in self.files.items():
            arr.flush()
            meta[f"{self.prefix}{name}.npy"] = list(arr.shape)
        self.files = {}
        return meta


class StatsSink:
    """Accumulates, on the GPU in float64, per-group sums (group = category x modality) and per-modality sums of
    outer products, for the within-group covariance. Deterministic (one-hot matmul, no atomic adds)."""

    def __init__(self, n_groups, group_modality, covariance, device):
        self.n_groups = n_groups
        self.group_modality = torch.tensor(group_modality, device=device)  # 0 sketch, 1 photo
        self.covariance = covariance
        self.device = device
        self.sums, self.sxx = {}, {}
        self.counts = torch.zeros(n_groups, dtype=torch.float64, device=device)
        self.onehot = None
        self.modality = None
        self.pending = {}  # this batch's contributions, added to the totals only when the whole batch succeeded

    def set_batch(self, groups):
        g = torch.as_tensor(groups, device=self.device)
        self.onehot = F.one_hot(g, self.n_groups).double()  # [b, G]
        self.modality = self.group_modality[g]
        self.pending = {}

    def __call__(self, key, x):
        if key[2] == "all":
            return
        x = x.detach().double().reshape(x.shape[0], -1)
        s = self.onehot.T @ x  # [G, D]
        sxx = None
        if self.covariance and x.shape[1] <= 1024:
            sxx = torch.zeros(2, x.shape[1], x.shape[1], dtype=torch.float64, device=self.device)
            for m in (0, 1):
                xm = x[self.modality == m]
                if len(xm):
                    sxx[m] = xm.T @ xm
        self.pending[key] = (s, sxx)  # a retried batch overwrites, never double-counts

    def commit(self):
        for key, (s, sxx) in self.pending.items():
            self.sums[key] = s if key not in self.sums else self.sums[key] + s
            if sxx is not None:
                self.sxx[key] = sxx if key not in self.sxx else self.sxx[key] + sxx
        self.counts += self.onehot.sum(0)
        self.pending = {}

    def results(self):
        counts = self.counts.clamp(min=1)
        means = {k: (v / counts[:, None]) for k, v in self.sums.items()}
        covs = {}
        for key, sxx in self.sxx.items():
            mu = means[key]
            out = torch.zeros_like(sxx)
            for m in (0, 1):
                sel = self.group_modality == m
                n_m = float(self.counts[sel].sum())
                g_m = int((self.counts[sel] > 0).sum())
                between = (mu[sel].T * self.counts[sel]) @ mu[sel]
                out[m] = (sxx[m] - between) / max(n_m - g_m, 1.0)
            covs[key] = out
        return means, covs


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items, self.transform = items, normal_transform(224)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        kind, value = self.items[i]
        if kind == "path":
            image = load_image(value, 224)
        else:
            from PIL import Image
            image = Image.fromarray(np.load(value)).convert("RGB")
        return self.transform(image), i


@torch.no_grad()
def run_images(models, captures, items, device, sinks, before_batch=None, keep_emb=True, batch_size=BATCH):
    """Stream items through every model. sinks: {model name: sink}. Returns {name: embeddings on device} if
    keep_emb (float16, [n, 512]; 1 KB per image)."""
    loader = torch.utils.data.DataLoader(ImageSet(items), batch_size=batch_size, shuffle=False,
                                         num_workers=WORKERS, pin_memory=True)
    for name, cap in captures.items():
        cap.sink = sinks[name]
    embs = {name: [] for name in models}
    for images, idx in loader:
        if before_batch:
            before_batch(idx.numpy())
        images = gpu_retry(lambda: images.to(device, non_blocking=True), "moving a batch")
        for name, model in models.items():
            def step():  # the sinks write in place by batch offset, so a retried batch overwrites itself
                f = model.visual(images.type(model.dtype))
                return f / f.norm(dim=-1, keepdim=True)
            feat = gpu_retry(step, "a batch")
            sinks[name](("emb", 0, "vec"), feat)
            if hasattr(sinks[name], "end_batch"):
                sinks[name].end_batch()
            if keep_emb:
                embs[name].append(feat)
    return {name: torch.cat(e) for name, e in embs.items()} if keep_emb else None


@torch.no_grad()
def encode_texts(model, texts, device, layers=()):
    tokens = clip_lib.tokenize(texts, truncate=True).to(device)
    store = {layer: [] for layer in layers}
    current = {}
    handles = []
    for layer in layers:
        def hook(_m, _i, o, layer=layer):
            x = o.permute(1, 0, 2)  # [N, 77, 512]
            store[layer].append(x[torch.arange(x.shape[0], device=x.device), current["eot"]].float().cpu().numpy()
                                .astype(np.float16))
        handles.append(model.transformer.resblocks[layer - 1].register_forward_hook(hook))
    feats = []
    for i in range(0, len(texts), 256):
        batch = tokens[i:i + 256]
        current["eot"] = batch.argmax(dim=-1)
        f = model.encode_text(batch)
        feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy().astype(np.float16))
    for h in handles:
        h.remove()
    hidden = {layer: np.concatenate(chunks) for layer, chunks in store.items() if chunks}
    return np.concatenate(feats), hidden, tokens.cpu().numpy().astype(np.int32)


# --------------------------------------------------------------------------- request handling
class Context:
    def __init__(self, task_dir, work, max_bytes):
        self.task_dir = os.path.abspath(task_dir)
        self.task = json.load(open(os.path.join(self.task_dir, "task.json")))
        self.dataset = self.task["dataset"]
        self.root = self.task["root"]
        self.work = os.path.abspath(work)
        self.max_bytes = max_bytes
        self.original_dir = os.path.join(ROOT, "model", "tensors")
        self.index, self.by_category = {}, {}
        with open(os.path.join(self.task_dir, "practice", "index.jsonl")) as f:
            for line in f:
                row = json.loads(line)
                self.index[row["id"]] = row
                self.by_category.setdefault((row["modality"], row["category"]), []).append(row["id"])
        self.categories = json.load(open(os.path.join(self.task_dir, "practice", "categories.json")))

    def practice_path(self, item_id, modality=None):
        row = self.index.get(item_id)
        if row is None:
            raise RequestError(f"unknown practice id {item_id!r} (ids are the 'id' field of /practice/index.jsonl)")
        if modality and row["modality"] != modality:
            raise RequestError(f"{item_id!r} is a {row['modality']}, not a {modality}")
        return os.path.join(self.root, item_id)

    def work_path(self, path):
        if not path.startswith("/work/"):
            raise RequestError(f"{path!r}: images must be practice ids or files under /work/")
        real = os.path.realpath(os.path.join(self.work, path[len("/work/"):]))
        if not real.startswith(self.work + os.sep) or not os.path.isfile(real):
            raise RequestError(f"{path!r}: not a file inside /work")
        return ("array" if real.endswith(".npy") else "path"), real

    def subset(self, modality, category, count):
        ids = self.by_category[(modality, category)]
        if count is None or count >= len(ids):
            return list(ids)
        seed = int(hashlib.sha256(f"{modality}/{category}".encode()).hexdigest()[:8], 16)
        order = np.random.default_rng(seed).permutation(len(ids))
        return [ids[i] for i in sorted(order[:count])]

    def select(self, request):
        """(sketch ids, photo ids) from "categories" (a list or "all") with per-category counts, or explicit lists."""
        if "categories" in request:
            if "sketches" in request or "photos" in request:
                raise RequestError("give either 'categories' (with per-category counts) or explicit 'sketches' and "
                                   "'photos'")
            cats = self.categories if request["categories"] == "all" else request["categories"]
            if not isinstance(cats, list) or not cats:
                raise RequestError("'categories' must be \"all\" or a non-empty list of practice categories")
            bad = [c for c in cats if c not in self.categories]
            if bad:
                raise RequestError(f"not practice categories: {bad[:10]}")
            if len(set(cats)) != len(cats):
                raise RequestError("duplicate categories")
            ks, kp = request.get("sketches_per_category"), request.get("photos_per_category")
            for key, v in (("sketches_per_category", ks), ("photos_per_category", kp)):
                if v is not None and not (isinstance(v, int) and v >= 0):
                    raise RequestError(f"'{key}' must be a non-negative integer (leave it out to take all)")
            sketches = [i for c in cats for i in (self.subset("sketch", c, ks) if ks != 0 else [])]
            photos = [i for c in cats for i in (self.subset("photo", c, kp) if kp != 0 else [])]
        else:
            sketches, photos = request.get("sketches", []), request.get("photos", [])
            if not isinstance(sketches, list) or not isinstance(photos, list):
                raise RequestError("'sketches' and 'photos' must be lists of practice ids")
        if len(set(sketches)) != len(sketches) or len(set(photos)) != len(photos):
            raise RequestError("duplicate ids")
        for i in sketches:
            self.practice_path(i, "sketch")
        for i in photos:
            self.practice_path(i, "photo")
        return sketches, photos


def int_list(value, name, low, high):
    if not isinstance(value, list) or not all(isinstance(v, int) and low <= v <= high for v in value):
        raise RequestError(f"{name} must be a list of integers in {low}..{high}")
    return sorted(set(value))


def which_models(request, allowed):
    model = request.get("model")
    if model not in allowed:
        raise RequestError(f"'model' must be one of {allowed}")
    return ["original", "edited"] if model == "both" else [model]


def make_models(ctx, names, device):
    models, applied = {}, []
    for name in names:
        edits = edit_files(ctx.work) if name == "edited" else None
        if name == "edited" and not edits:
            raise RequestError("'edited' requested but /work/edited holds no edits")
        model, applied_now = build(ctx.task["weights"], edits, ctx.original_dir)
        models[name] = gpu_retry(lambda: model.to(device), "loading the model")
        applied = applied_now or applied
    return models, applied


def per_image_bytes(residual, mlp_blocks, attn_blocks, n_parts, tokens_all=False):
    per_tok = TOKENS if tokens_all else n_parts
    return 2 * (EMBED + WIDTH + per_tok * (len(residual) * WIDTH + len(mlp_blocks) * MLP_WIDTH)
                + len(attn_blocks) * 12 * TOKENS * TOKENS)


def check_size(ctx, est):
    if est > ctx.max_bytes:
        raise RequestError(f"this pass would write {est / 2**30:.1f} GB, more than the {ctx.max_bytes / 2**30:.1f} GB "
                           "allowed now (the per-pass limit, or the room left in the /work quota): request fewer "
                           "images, layers or blocks, or delete old fwd_<k> folders")


def parse_acts(spec, allowed_tokens):
    spec = spec or {}
    unknown = set(spec) - {"residual", "mlp_blocks", "tokens"}
    if unknown:
        raise RequestError(f"unknown activation keys {sorted(unknown)} (use 'residual', 'mlp_blocks', 'tokens')")
    residual = int_list(spec.get("residual", []), "residual", 0, N_RESIDUAL - 1)
    mlp_blocks = int_list(spec.get("mlp_blocks", []), "mlp_blocks", 0, N_BLOCKS - 1)
    tokens = spec.get("tokens", allowed_tokens[0])
    if tokens not in allowed_tokens:
        raise RequestError(f"'tokens' must be one of {allowed_tokens}")
    return residual, mlp_blocks, tokens


PARTS = {"summary": ("cls", "patch"), "cls": ("cls",), "patch": ("patch",), "all": ("all",)}


def retrieve(ctx, request, out_dir, device):
    names = which_models(request, ["original", "edited", "both"])
    sketches, photos = ctx.select(request)
    if not sketches or not photos:
        raise RequestError("a retrieval needs at least one sketch and one photo")
    residual, mlp_blocks, tokens = parse_acts(request.get("activations"), ["summary", "cls", "patch"])
    parts = PARTS[tokens]
    n_img = len(sketches) + len(photos)
    check_size(ctx, len(names) * n_img * per_image_bytes(residual, mlp_blocks, [], len(parts)))

    cats_all = sorted({ctx.index[i]["category"] for i in sketches + photos})
    label = {c: j for j, c in enumerate(cats_all)}
    sk_labels = torch.tensor([label[ctx.index[i]["category"]] for i in sketches])
    ph_labels = torch.tensor([label[ctx.index[i]["category"]] for i in photos])
    missing = sorted({ctx.index[i]["category"] for i in sketches} - {ctx.index[i]["category"] for i in photos})

    models, applied = make_models(ctx, names, device)
    os.makedirs(out_dir, exist_ok=True)
    subdirs = {n: (out_dir if len(names) == 1 else os.path.join(out_dir, n)) for n in names}
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)
    layers = {"residual": residual, "mlp": mlp_blocks}
    captures = {n: Capture(m, residual, mlp_blocks, (), parts) for n, m in models.items()}
    files, embs = {n: {} for n in names}, {}
    for prefix, ids in (("sketch_", sketches), ("photo_", photos)):
        sinks = {n: MemmapSink(subdirs[n], prefix, len(ids), layers) for n in names}
        e = run_images(models, captures, [("path", ctx.practice_path(i)) for i in ids], device, sinks)
        for n in names:
            files[n].update(sinks[n].close())
            embs.setdefault(n, {})[prefix] = e[n]
    for cap in captures.values():
        cap.close()
    with open(os.path.join(out_dir, "ids.json"), "w") as f:
        json.dump({"sketches": sketches, "photos": photos, "categories": cats_all}, f)

    map_name, p_name = metric_names(ctx.dataset)
    results = {}
    for n in names:
        ap, pk, _, _ = per_query_metrics(embs[n]["sketch_"], embs[n]["photo_"], sk_labels, ph_labels, ctx.dataset)
        results[n] = (ap, pk)
        with open(os.path.join(subdirs[n], "files.json"), "w") as f:
            json.dump(files[n], f, indent=1)

    q_cat = [ctx.index[i]["category"] for i in sketches]
    with open(os.path.join(out_dir, "per_query.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sketch", "category"] + [f"{n}_{m}" for n in names for m in ("ap", "p")])
        for j, sid in enumerate(sketches):
            w.writerow([sid, q_cat[j]] + [f"{results[n][k][j]:.6f}" for n in names for k in (0, 1)])
    per_cat = {}
    for j, c in enumerate(q_cat):
        per_cat.setdefault(c, []).append(j)
    n_ph = {}
    for i in photos:
        c = ctx.index[i]["category"]
        n_ph[c] = n_ph.get(c, 0) + 1
    with open(os.path.join(out_dir, "per_category.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "sketches", "photos"] + [f"{n}_{m}" for n in names for m in ("map", "p")])
        for c, js in sorted(per_cat.items()):
            w.writerow([c, len(js), n_ph.get(c, 0)] + [f"{results[n][k][js].mean():.6f}" for n in names
                                                         for k in (0, 1)])

    summary = {"ok": True, "type": "retrieve", "model": request["model"], "dir": f"/work/{os.path.basename(out_dir)}/",
               "sketches": len(sketches), "photos": len(photos), "categories": len(cats_all),
               "metric": f"{map_name} and {p_name} (as in the evaluation: relevant = same category)",
               "files_written_gb": round(sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(out_dir)
                                             for fn in fns) / 2**30, 2)}
    if applied:
        summary["edited_tensors"] = applied
    if missing:
        summary["warning"] = f"{len(missing)} query categories have no photo in the gallery (their AP is 0): {missing[:5]}"
    for n in names:
        ap, pk = results[n]
        summary[n] = {map_name: round(float(ap.mean()), 5), p_name: round(float(pk.mean()), 5)}
    if len(names) == 2:
        cats = sorted(per_cat)
        summary["paired"] = {}
        for k, metric in ((0, map_name), (1, p_name)):
            d = results["edited"][k] - results["original"][k]
            dc = np.array([d[per_cat[c]].mean() for c in cats])
            summary["paired"][metric] = {
                "difference": round(float(d.mean()), 5),
                "se_over_queries": round(float(d.std(ddof=1) / np.sqrt(len(d))), 5) if len(d) > 1 else None,
                "se_over_categories": round(float(dc.std(ddof=1) / np.sqrt(len(dc))), 5) if len(dc) > 1 else None,
                "queries_better": int((d > 1e-9).sum()), "queries_worse": int((d < -1e-9).sum()),
                "categories_better": int((dc > 1e-9).sum()), "categories_worse": int((dc < -1e-9).sum()),
            }
    return summary


def read(ctx, request, out_dir, device):
    names = which_models(request, ["original", "edited"])
    images = request.get("images", [])
    texts = request.get("texts", [])
    if not isinstance(images, list) or not isinstance(texts, list) or not (images or texts):
        raise RequestError("give 'images' (practice ids or /work files) and/or 'texts' (strings)")
    if len(images) > LIMITS["read_images"] or len(texts) > LIMITS["read_texts"]:
        raise RequestError(f"at most {LIMITS['read_images']} images and {LIMITS['read_texts']} texts per read pass")
    if not all(isinstance(t, str) for t in texts):
        raise RequestError("'texts' must be strings")
    ret = request.get("return", {}) or {}
    unknown = set(ret) - {"residual", "tokens", "mlp_blocks", "attn_blocks", "text_residual"}
    if unknown:
        raise RequestError(f"unknown 'return' keys {sorted(unknown)}")
    residual, mlp_blocks, tokens = parse_acts({k: ret[k] for k in ("residual", "mlp_blocks", "tokens") if k in ret},
                                              ["cls", "summary", "patch", "all"])
    attn_blocks = int_list(ret.get("attn_blocks", []), "return.attn_blocks", 0, N_BLOCKS - 1)
    text_residual = int_list(ret.get("text_residual", []), "return.text_residual", 1, 12)
    if attn_blocks and len(images) > LIMITS["attn_images"]:
        raise RequestError(f"attention probabilities for at most {LIMITS['attn_images']} images")
    items = []
    for im in images:
        if not isinstance(im, str):
            raise RequestError("'images' entries must be strings")
        items.append(ctx.work_path(im) if im.startswith("/") else ("path", ctx.practice_path(im)))
    parts = PARTS[tokens]
    check_size(ctx, len(items) * per_image_bytes(residual, mlp_blocks, attn_blocks, len(parts), tokens == "all"))

    models, applied = make_models(ctx, names, device)
    model = models[names[0]]
    os.makedirs(out_dir, exist_ok=True)
    meta = {"model": names[0], "images": images, "texts_count": len(texts), "files": {}}
    if items:
        cap = Capture(model, residual, mlp_blocks, attn_blocks, parts)
        sink = MemmapSink(out_dir, "image_", len(items), {"residual": residual, "mlp": mlp_blocks, "attn": attn_blocks})
        run_images(models, {names[0]: cap}, items, device, {names[0]: sink}, keep_emb=False,
                   batch_size=64 if attn_blocks else BATCH)
        cap.close()
        meta["files"].update(sink.close())
        meta["layers"] = {"residual": residual, "mlp_blocks": mlp_blocks, "attn_blocks": attn_blocks,
                          "tokens": tokens}
    if texts:
        temb, thidden, toks = encode_texts(model, texts, device, text_residual)
        np.save(os.path.join(out_dir, "text_emb.npy"), temb)
        np.save(os.path.join(out_dir, "text_tokens.npy"), toks)
        meta["files"]["text_emb.npy"] = list(temb.shape)
        meta["files"]["text_tokens.npy"] = list(toks.shape)
        if thidden:
            stacked = np.stack([thidden[l] for l in sorted(thidden)], axis=1)
            np.save(os.path.join(out_dir, "text_residual_eot.npy"), stacked)
            meta["files"]["text_residual_eot.npy"] = {"layers": sorted(thidden), "shape": list(stacked.shape)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    summary = {"ok": True, "type": "read", "model": names[0], "dir": f"/work/{os.path.basename(out_dir)}/",
               "images": len(items), "texts": len(texts), "files": meta["files"]}
    if applied:
        summary["edited_tensors"] = applied
    return summary


def stats(ctx, request, out_dir, device):
    names = which_models(request, ["original", "edited"])
    sketches, photos = ctx.select(request)
    if not sketches and not photos:
        raise RequestError("no images selected")
    unknown = set(request) - {"type", "model", "categories", "sketches_per_category", "photos_per_category",
                              "sketches", "photos", "residual", "mlp_blocks", "tokens", "covariance"}
    if unknown:
        raise RequestError(f"unknown keys {sorted(unknown)}")
    residual, mlp_blocks, tokens = parse_acts({k: request[k] for k in ("residual", "mlp_blocks", "tokens")
                                               if k in request}, ["summary", "cls", "patch"])
    covariance = bool(request.get("covariance", False))
    parts = PARTS[tokens]
    ids = sketches + photos
    groups = sorted({(ctx.index[i]["category"], ctx.index[i]["modality"]) for i in ids})
    gidx = {g: j for j, g in enumerate(groups)}
    item_group = np.array([gidx[(ctx.index[i]["category"], ctx.index[i]["modality"])] for i in ids])
    check_size(ctx, len(ids) * 2 * (EMBED + WIDTH)
               + len(groups) * 4 * (len(parts) * (len(residual) * WIDTH + len(mlp_blocks) * MLP_WIDTH) + EMBED + WIDTH)
               + (8 * len(parts) * len(residual) * WIDTH ** 2 + 8 * (EMBED ** 2 + WIDTH ** 2) if covariance else 0))

    models, applied = make_models(ctx, names, device)
    model = models[names[0]]
    os.makedirs(out_dir, exist_ok=True)
    acc = StatsSink(len(groups), [0 if m == "sketch" else 1 for _, m in groups], covariance, device)
    per_image = MemmapSink(out_dir, "image_", len(ids), {})

    def sink(key, x):
        acc(key, x)
        if key[0] in ("pre", "emb"):
            per_image(key, x)

    def end_batch():
        gpu_retry(acc.commit, "committing statistics")
        per_image.end_batch()

    sink.end_batch = end_batch
    cap = Capture(model, residual, mlp_blocks, (), parts)
    run_images(models, {names[0]: cap}, [("path", ctx.practice_path(i)) for i in ids], device, {names[0]: sink},
               before_batch=lambda idx: acc.set_batch(item_group[idx]), keep_emb=False)
    cap.close()
    files = per_image.close()
    means, covs = acc.results()
    layer_lists = {"residual": residual, "mlp": mlp_blocks}
    for kind in ("residual", "mlp"):
        for part in parts:
            ls = layer_lists[kind]
            if ls:
                arr = torch.stack([means[(kind, l, part)] for l in ls], dim=1).float().cpu().numpy()  # [G, L, D]
                np.save(os.path.join(out_dir, f"mean_{kind}_{part}.npy"), arr)
                files[f"mean_{kind}_{part}.npy"] = list(arr.shape)
                if covariance and kind == "residual":
                    c = torch.stack([covs[(kind, l, part)] for l in ls], dim=1).float().cpu().numpy()  # [2, L, D, D]
                    np.save(os.path.join(out_dir, f"cov_{kind}_{part}.npy"), c)
                    files[f"cov_{kind}_{part}.npy"] = list(c.shape)
    for kind in ("pre", "emb"):
        arr = means[(kind, 0, "vec")].float().cpu().numpy()
        np.save(os.path.join(out_dir, f"mean_{kind}.npy"), arr)
        files[f"mean_{kind}.npy"] = list(arr.shape)
        if covariance:
            c = covs[(kind, 0, "vec")].float().cpu().numpy()
            np.save(os.path.join(out_dir, f"cov_{kind}.npy"), c)
            files[f"cov_{kind}.npy"] = list(c.shape)
    counts = acc.counts.cpu().numpy().astype(int)
    with open(os.path.join(out_dir, "groups.json"), "w") as f:
        json.dump({"groups": [{"category": c, "modality": m, "count": int(n)} for (c, m), n in zip(groups, counts)],
                   "layers": {"residual": residual, "mlp_blocks": mlp_blocks}, "tokens": list(parts),
                   "covariance": covariance, "ids": ids}, f)
    summary = {"ok": True, "type": "stats", "model": names[0], "dir": f"/work/{os.path.basename(out_dir)}/",
               "images": len(ids), "sketches": len(sketches), "photos": len(photos), "groups": len(groups),
               "files": files}
    if applied:
        summary["edited_tensors"] = applied
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-bytes", type=float, default=40 * 2**30)
    args = parser.parse_args()
    t0 = time.time()
    try:
        with open(args.request) as f:
            request = json.load(f)
        if not isinstance(request, dict):
            raise RequestError("the request must be a JSON object")
        deterministic()
        torch.set_num_threads(int(os.environ.get("WSP_TORCH_THREADS", "6")))  # the machine is shared
        device = torch.device("cuda")
        torch.zeros(1, device=device)  # create the CUDA context first: if the GPU is full this fails with
        # "out of memory" before any work is done, and the driver retries the request later (not counted)
        ctx = Context(args.task, args.work, args.max_bytes)
        out_dir = os.path.join(ctx.work, f"fwd_{args.k}")
        handler = {"retrieve": retrieve, "read": read, "stats": stats}.get(request.get("type"))
        if handler is None:
            raise RequestError("'type' must be 'retrieve', 'read' or 'stats'")
        summary = handler(ctx, request, out_dir, device)
    except RequestError as exc:
        summary = {"ok": False, "error": str(exc)}
    except json.JSONDecodeError as exc:
        summary = {"ok": False, "error": f"the request is not valid JSON: {exc}"}
    except Exception as exc:  # noqa: BLE001
        summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]}
    summary["s"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
