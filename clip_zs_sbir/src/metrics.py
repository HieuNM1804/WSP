"""Retrieval metrics copied verbatim from KD-SBIR src/model.py::_retrieval_metrics
(commit 741ea272b258c9e7b92a46173618e2fe1d401eaa).

Kept in its own module so evaluation does not import open_clip / pytorch_lightning.
"""
import torch
from torch.nn import functional as F
from torchmetrics.functional.retrieval import (
    retrieval_average_precision,
    retrieval_precision,
)


def _retrieval_metrics(
    query_features,
    gallery_features,
    query_labels,
    gallery_labels,
    dataset,
):
    query_labels = query_labels.cpu()
    gallery_labels = gallery_labels.cpu()
    ap = torch.zeros(len(query_features))
    precision_at_k = torch.zeros(len(query_features))
    if dataset == "sketchy_2":
        map_k = 200
        p_k = 200
    else:
        map_k = 0
        p_k = 200 if dataset == "quickdraw" else 100

    for index, query_feature in enumerate(query_features):
        cosine = F.cosine_similarity(
            query_feature.unsqueeze(0), gallery_features
        ).cpu()
        score = ((cosine + 1.0) * 0.5).clamp(
            min=torch.finfo(cosine.dtype).eps,
            max=1.0,
        )
        target = gallery_labels.eq(query_labels[index])
        if map_k:
            ap[index] = retrieval_average_precision(
                score,
                target,
                top_k=min(map_k, len(gallery_features)),
            )
        else:
            ap[index] = retrieval_average_precision(score, target)
        precision_at_k[index] = retrieval_precision(
            score,
            target,
            top_k=p_k,
        )

    return ap.mean(), precision_at_k.mean(), map_k, p_k
