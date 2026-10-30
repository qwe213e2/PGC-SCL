from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SpectralEncoder(nn.Module):
    """Shared 1D convolutions along the 5-band axis for EACH EEG channel."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, c, f = x.shape
        if f != 5:
            raise ValueError(f"Expected 5 spectral bands, got {f}")
        y = self.net(x.reshape(b * c, 1, f)).squeeze(-1)
        return self.dropout(y.reshape(b, c, -1))


class SparseGraphEncoder(nn.Module):
    """Learned CxCx2 logits -> Gumbel/softmax -> soft top-k -> normalized GCN."""

    def __init__(
        self, channels: int, bands: int = 5, hidden_dim: int = 64,
        topk_ratio: float = 0.15, tau_s: float = 0.1, eta: float = 1.0,
        dropout: float = 0.3,
    ):
        super().__init__()
        if channels < 2 or not 0 < topk_ratio <= 1:
            raise ValueError("channels must be >= 2 and 0 < topk_ratio <= 1")
        if tau_s <= 0 or eta <= 0:
            raise ValueError("tau_s and eta must be > 0")
        self.channels = channels
        self.topk_ratio = topk_ratio
        self.tau_s = tau_s
        self.eta = eta
        self.edge_logits = nn.Parameter(0.02 * torch.randn(channels, channels, 2))
        self.projection = nn.Linear(bands, hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("eye", torch.eye(channels), persistent=False)

    def forward(self, x: Tensor, gumbel_tau: float = 1.0) -> tuple[Tensor, Tensor, Tensor]:
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        if self.training:
            edge_p = F.gumbel_softmax(self.edge_logits, tau=gumbel_tau, hard=False, dim=-1)[..., 1]
        else:
            # Deterministic expected edge probabilities at inference.
            edge_p = torch.softmax(self.edge_logits / gumbel_tau, dim=-1)[..., 1]

        # No learned self-edges: identity is added only for stable message passing.
        candidate = edge_p.masked_fill(self.eye.bool(), -1.0)
        k = max(1, min(self.channels - 1, math.ceil(self.topk_ratio * (self.channels - 1))))
        # Row-wise k-th largest off-diagonal edge. Detaching the threshold
        # keeps the soft gate differentiable w.r.t. individual edge scores.
        threshold = torch.topk(candidate, k=k, dim=-1).values[..., -1:].detach()
        sparse = edge_p * torch.sigmoid(self.eta * (edge_p - threshold) / self.tau_s)
        sparse = sparse * (1.0 - self.eye)

        # Eq. (9) uses an unnormalized L1 norm of the sparse learned graph.
        sparse_loss = sparse.abs().sum()
        a = sparse + self.eye
        degrees = a.sum(dim=1).clamp_min(1e-8)
        a_norm = degrees.rsqrt().unsqueeze(1) * a * degrees.rsqrt().unsqueeze(0)
        h = self.projection(x)
        h = torch.einsum("ij,bjd->bid", a_norm, h)
        h = self.dropout(self.norm(self.activation(h)))
        return h, sparse, sparse_loss


class PersonalityGuidedFusion(nn.Module):
    """Eqs. (10)–(14), scalar softmax gating over the two EEG views."""

    def __init__(self, hidden_dim: int, dropout: float, personality_fusion: bool = True):
        super().__init__()
        self.personality_fusion = personality_fusion
        self.personality_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, spectral: Tensor, spatial: Tensor, personality: Tensor):
        u_spe = spectral.mean(dim=1)
        u_spa = spatial.mean(dim=1)
        p = self.personality_mlp(personality)
        if not self.personality_fusion:
            p = torch.zeros_like(p)
        alpha = torch.softmax(self.gate_mlp(torch.cat((u_spe, u_spa, p), dim=-1)), dim=-1)
        z = alpha[:, :1] * u_spe + alpha[:, 1:] * u_spa
        return z, alpha


class PGCSCL(nn.Module):
    """End-to-end classification network + training-only projection head."""

    def __init__(
        self, channels: int, classes: int, bands: int = 5,
        hidden_dim: int = 64, projection_dim: int = 64,
        dropout: float = 0.3, graph_topk_ratio: float = 0.15,
        graph_tau_s: float = 0.1, graph_eta: float = 1.0,
        personality_fusion: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.classes = classes
        self.spectral = SpectralEncoder(hidden_dim, dropout)
        self.spatial = SparseGraphEncoder(
            channels, bands, hidden_dim, graph_topk_ratio, graph_tau_s, graph_eta, dropout
        )
        self.fusion = PersonalityGuidedFusion(hidden_dim, dropout, personality_fusion)
        self.classifier = nn.Linear(hidden_dim, classes)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim)
        )

    def forward(
        self, x: Tensor, personality: Tensor, gumbel_tau: float = 1.0,
        with_projection: bool | None = None,
    ) -> dict[str, Tensor | None]:
        if x.ndim != 3 or x.shape[1] != self.channels or x.shape[2] != 5:
            raise ValueError(f"x must be [B, {self.channels}, 5], got {tuple(x.shape)}")
        if personality.shape != (x.shape[0], 5):
            raise ValueError("personality must be [B, 5]")
        h_spe = self.spectral(x)
        h_spa, adjacency, sparse_loss = self.spatial(x, gumbel_tau)
        z, gate = self.fusion(h_spe, h_spa, personality)
        logits = self.classifier(z)
        use_projection = self.training if with_projection is None else with_projection
        embedding = F.normalize(self.projection_head(z), dim=-1) if use_projection else None
        return {
            "logits": logits, "embedding": embedding, "gate": gate,
            "adjacency": adjacency, "sparse_loss": sparse_loss, "fused": z,
        }
