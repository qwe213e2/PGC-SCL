from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def supervised_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
    subject_ids: Tensor,
    personality: Tensor,
    temperature: float = 0.07,
    personality_temperature: float = 0.5,
    variant: str = "pa_scl",
) -> Tensor:
    """Valid variants: pa_scl, cross_subject_supcon, supcon, none.

    For pa_scl: positives = same label AND different subject; denominator
    includes ONLY different-subject samples. Soft pair weights are based
    on standardized OCEAN cosine similarity. No-valid-positive => zero.
    """
    if variant not in {"pa_scl", "cross_subject_supcon", "supcon", "none"}:
        raise ValueError(f"Unknown contrastive variant: {variant}")
    if temperature <= 0 or personality_temperature <= 0:
        raise ValueError("Temperatures must be positive")
    if variant == "none" or embeddings.shape[0] <= 1:
        return embeddings.sum() * 0.0

    batch = embeddings.shape[0]
    labels = labels.reshape(-1)
    subject_ids = subject_ids.reshape(-1)
    if labels.numel() != batch or subject_ids.numel() != batch or personality.shape != (batch, 5):
        raise ValueError("Incompatible batch dimensions")
    eye = torch.eye(batch, dtype=torch.bool, device=embeddings.device)
    if variant == "supcon":
        comparisons = ~eye
    else:
        comparisons = (subject_ids[:, None] != subject_ids[None, :]) & ~eye
    positives = comparisons & (labels[:, None] == labels[None, :])
    valid = positives.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0

    z = F.normalize(embeddings, p=2, dim=-1)
    logits = (z @ z.T) / temperature
    logits = logits.masked_fill(~comparisons, -torch.inf)
    # Compute only on valid anchors to avoid -inf rows (0 * NaN).
    valid_logits = logits[valid]
    valid_positives = positives[valid]
    log_prob = valid_logits - torch.logsumexp(valid_logits, dim=1, keepdim=True)

    if variant == "pa_scl":
        pp = F.normalize(personality, p=2, dim=-1, eps=1e-8)
        sim = (pp @ pp.T).clamp(-1, 1)[valid]
        weights = torch.exp(-(1.0 - sim) / (2.0 * personality_temperature))
        weights = weights * valid_positives.to(weights.dtype)
    else:
        weights = valid_positives.to(log_prob.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    # Replace invalid entries before multiplication, since 0 * -inf is NaN.
    log_prob = log_prob.masked_fill(~valid_positives, 0.0)
    return -(weights * log_prob).sum(dim=1).mean()
