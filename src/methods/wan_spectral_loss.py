"""Spectral-WAN (DFR-WAN) finite-dimensional dual norm loss."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


def _pair(value, dim: int, name: str):
    values = (value,) * dim if isinstance(value, (int, float, str)) else tuple(value)
    if len(values) != dim:
        raise ValueError(f"{name} must have {dim} entries")
    return values


class SpectralWANLoss(nn.Module):
    """Deterministic Parseval-reduced WAN loss on a Fourier test space.

    ``forward(moments)`` expects modal weak residuals and returns
    ``sum moments**2/lambda``.  ``project(values)`` additionally computes
    those moments from uniformly sampled values using tensorized orthogonal
    DST/DCT matrices.  ``boundary`` may be a scalar or one value per axis.
    """

    def __init__(self, modes: int | Sequence[int], spatial_dim: int = 1,
                 boundary: str | Sequence[str] = "dirichlet",
                 domain: float | Sequence[float] = math.pi,
                 transform: str = "auto", shift: float = 1.0) -> None:
        super().__init__()
        if spatial_dim not in (1, 2):
            raise ValueError("spatial_dim must be 1 or 2")
        self.spatial_dim = spatial_dim
        self.modes = tuple(int(v) for v in _pair(modes, spatial_dim, "modes"))
        if any(v < 1 for v in self.modes):
            raise ValueError("modes must be positive")
        self.boundary = tuple(str(v).lower() for v in _pair(boundary, spatial_dim, "boundary"))
        if any(v not in {"dirichlet", "neumann"} for v in self.boundary):
            raise ValueError("boundary must be Dirichlet or Neumann")
        self.domain = tuple(float(v) for v in _pair(domain, spatial_dim, "domain"))
        self.transform = transform.lower()
        self.shift = float(shift)

    @staticmethod
    def _matrix(size: int, modes: int, kind: str, dtype, device) -> torch.Tensor:
        j = torch.arange(size, dtype=dtype, device=device)[:, None]
        if kind.startswith("dst"):
            k = torch.arange(1, modes + 1, dtype=dtype, device=device)[None, :]
            if kind in {"dst-iv", "dstiv"}:
                phase = (j + 0.5) * (k - 0.5)
            else:
                phase = (j + 0.5) * k
            matrix = math.sqrt(2.0 / size) * torch.sin(math.pi * phase / size)
        else:
            k = torch.arange(modes, dtype=dtype, device=device)[None, :]
            phase = (j + 0.5) * (k + 0.5) if kind in {"dct-iv", "dctiv"} else (j + 0.5) * k
            matrix = torch.cos(math.pi * phase / size)
            if kind not in {"dct-iv", "dctiv"}:
                matrix[:, 0] *= 1.0 / math.sqrt(size)
                if modes > 1:
                    matrix[:, 1:] *= math.sqrt(2.0 / size)
            else:
                matrix *= math.sqrt(2.0 / size)
        return matrix

    def _kind(self, axis: int) -> str:
        if self.transform != "auto":
            return self.transform
        return "dst-ii" if self.boundary[axis] == "dirichlet" else "dct-ii"

    def eigenvalues(self, dtype, device) -> torch.Tensor:
        values = []
        for axis, count in enumerate(self.modes):
            if self.boundary[axis] == "dirichlet":
                index = torch.arange(1, count + 1, dtype=dtype, device=device)
            else:
                index = torch.arange(count, dtype=dtype, device=device)
            values.append((math.pi * index / self.domain[axis]).square())
        if self.spatial_dim == 1:
            return values[0]
        return values[0][:, None] + values[1][None, :]

    def weights(self, dtype, device) -> torch.Tensor:
        return 1.0 / (self.shift + self.eigenvalues(dtype, device))

    def forward(self, moments: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        weights = self.weights(moments.dtype, moments.device)
        shape = (1,) * (moments.ndim - self.spatial_dim) + tuple(weights.shape)
        values = moments.square() * weights.reshape(shape)
        values = values.sum(dim=tuple(range(moments.ndim - self.spatial_dim, moments.ndim)))
        if reduction == "sum":
            return values.sum()
        if reduction == "mean":
            return values.mean()
        if reduction == "none":
            return values
        raise ValueError("reduction must be sum, mean, or none")

    def project(self, values: torch.Tensor) -> torch.Tensor:
        """Project uniformly sampled values onto the truncated orthogonal basis."""
        result = values
        for axis in range(self.spatial_dim):
            spatial_axis = values.ndim - self.spatial_dim + axis
            size = result.shape[spatial_axis]
            matrix = self._matrix(size, self.modes[axis], self._kind(axis), result.dtype, result.device)
            result = torch.tensordot(result, matrix, dims=([spatial_axis], [0])).movedim(-1, spatial_axis)
            result = result * math.sqrt(self.domain[axis] / size)
        return result

    def from_samples(self, values: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        return self(self.project(values), reduction=reduction)


__all__ = ["SpectralWANLoss"]
