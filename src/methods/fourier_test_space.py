"""Differentiable Fourier/DCT/DST dual-norm losses for 1D and 2D grids.

The transforms use orthonormal discrete sine (DST-I) bases for Dirichlet
axes and cosine (DCT-II) bases for Neumann axes.  Thus coefficients are
compatible with an L2 inner product (including the supplied domain length),
and truncation is performed by keeping the first ``modes`` on each axis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


Boundary = str | Sequence[str]


def dst(values: torch.Tensor, modes: int | None = None, dim: int = -1) -> torch.Tensor:
    """Orthonormal DST-I of ``values`` along ``dim`` (Dirichlet basis)."""
    dim = dim % values.ndim
    size = values.shape[dim]
    count = size if modes is None else int(modes)
    matrix = _sine_matrix(size, count, dtype=values.dtype, device=values.device)
    result = torch.tensordot(values, matrix, dims=([dim], [0])).movedim(-1, dim)
    return result


def dct(values: torch.Tensor, modes: int | None = None, dim: int = -1) -> torch.Tensor:
    """Orthonormal DCT-II of ``values`` along ``dim`` (Neumann basis)."""
    dim = dim % values.ndim
    size = values.shape[dim]
    count = size if modes is None else int(modes)
    matrix = _cosine_matrix(size, count, dtype=values.dtype, device=values.device)
    return torch.tensordot(values, matrix, dims=([dim], [0])).movedim(-1, dim)


def _boundaries(value: Boundary, dimensions: int) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,) * dimensions
    else:
        values = tuple(value)
    if len(values) != dimensions:
        raise ValueError(f"boundary must have {dimensions} entries")
    normalized = tuple(item.lower() for item in values)
    if any(item not in {"dirichlet", "neumann"} for item in normalized):
        raise ValueError("boundary entries must be 'dirichlet' or 'neumann'")
    return normalized


def _tuple_int(value: int | Sequence[int], dimensions: int, name: str) -> tuple[int, ...]:
    values = (value,) * dimensions if isinstance(value, int) else tuple(value)
    if len(values) != dimensions or any(int(item) != item or int(item) < 1 for item in values):
        raise ValueError(f"{name} must contain {dimensions} positive integers")
    return tuple(int(item) for item in values)


def _tuple_float(value: float | Sequence[float], dimensions: int, name: str) -> tuple[float, ...]:
    values = (value,) * dimensions if isinstance(value, (int, float)) else tuple(value)
    if len(values) != dimensions or any(float(item) <= 0 for item in values):
        raise ValueError(f"{name} must contain {dimensions} positive values")
    return tuple(float(item) for item in values)


def _sine_matrix(size: int, modes: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # DST-I on interior points x_j=j/(size+1), k=1,...,modes.
    j = torch.arange(1, size + 1, dtype=dtype, device=device)[:, None]
    k = torch.arange(1, modes + 1, dtype=dtype, device=device)[None, :]
    return math.sqrt(2.0 / (size + 1.0)) * torch.sin(math.pi * j * k / (size + 1.0))


def _cosine_matrix(size: int, modes: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # Orthonormal DCT-II on cell-centre points; k=0,...,modes-1.
    j = torch.arange(size, dtype=dtype, device=device)[:, None]
    k = torch.arange(modes, dtype=dtype, device=device)[None, :]
    matrix = torch.cos(math.pi * (j + 0.5) * k / size)
    matrix[:, 0] *= 1.0 / math.sqrt(size)
    if modes > 1:
        matrix[:, 1:] *= math.sqrt(2.0 / size)
    return matrix


class FourierDualLoss(nn.Module):
    """Truncated Fourier dual-norm loss on a uniformly sampled grid.

    Parameters
    ----------
    modes:
        Number of modes per axis (an integer or a tuple for 2D).
    spatial_dim:
        ``1`` or ``2``; spatial axes are the last dimensions of ``residual``.
    boundary:
        ``"dirichlet"``, ``"neumann"``, or one boundary type per axis.
    domain:
        Physical lengths per axis.  The default is ``pi`` on every axis.
    norm:
        ``"hminus1"`` (default), ``"l2"``, or ``"h1"`` spectral weighting.
    """

    def __init__(self, modes: int | Sequence[int], spatial_dim: int = 1,
                 boundary: Boundary = "dirichlet", domain: float | Sequence[float] = math.pi,
                 norm: str = "hminus1") -> None:
        super().__init__()
        if spatial_dim not in (1, 2):
            raise ValueError("spatial_dim must be 1 or 2")
        self.spatial_dim = spatial_dim
        self.modes = _tuple_int(modes, spatial_dim, "modes")
        self.boundary = _boundaries(boundary, spatial_dim)
        self.domain = _tuple_float(domain, spatial_dim, "domain")
        self.norm = norm.lower().replace("^", "")
        if self.norm not in {"hminus1", "h-1", "l2", "h1"}:
            raise ValueError("norm must be 'hminus1', 'l2', or 'h1'")

    def _matrix(self, size: int, modes: int, boundary: str, dtype: torch.dtype,
                device: torch.device) -> torch.Tensor:
        if modes > size:
            raise ValueError(f"modes={modes} exceeds grid size={size}")
        if boundary == "dirichlet":
            return _sine_matrix(size, modes, dtype=dtype, device=device)
        return _cosine_matrix(size, modes, dtype=dtype, device=device)

    def _wave_numbers(self, axis: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.boundary[axis] == "dirichlet":
            index = torch.arange(1, self.modes[axis] + 1, dtype=dtype, device=device)
        else:
            index = torch.arange(self.modes[axis], dtype=dtype, device=device)
        return math.pi * index / self.domain[axis]

    def transform(self, residual: torch.Tensor) -> torch.Tensor:
        """Return truncated orthonormal coefficients, preserving batch axes."""
        if residual.ndim < self.spatial_dim:
            raise ValueError("residual has fewer dimensions than spatial_dim")
        result = residual
        for offset, (modes, boundary) in enumerate(zip(self.modes, self.boundary)):
            axis = residual.ndim - self.spatial_dim + offset
            grid_size = result.shape[axis]
            matrix = self._matrix(grid_size, modes, boundary, result.dtype, result.device)
            result = torch.tensordot(result, matrix, dims=([axis], [0]))
            result = result.movedim(-1, axis)
            result = result * math.sqrt(self.domain[offset] / grid_size)
        return result

    def mode_numbers(self, *, dtype: torch.dtype = torch.float32,
                     device: torch.device | None = None) -> tuple[torch.Tensor, ...]:
        return tuple(self._wave_numbers(axis, dtype, device or torch.device("cpu")) for axis in range(self.spatial_dim))

    def eigenvalues(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        waves = self.mode_numbers(dtype=dtype, device=device)
        if self.spatial_dim == 1:
            return waves[0].square()
        return waves[0][:, None].square() + waves[1][None, :].square()

    def weights(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        eigenvalues = self.eigenvalues(dtype=dtype, device=device)
        if self.norm in {"hminus1", "h-1"}:
            return 1.0 / (1.0 + eigenvalues)
        if self.norm == "h1":
            return 1.0 + eigenvalues
        return torch.ones_like(eigenvalues)

    def energy(self, residual: torch.Tensor) -> torch.Tensor:
        coefficients = self.transform(residual)
        weights = self.weights(dtype=residual.dtype, device=residual.device)
        view_shape = (1,) * (coefficients.ndim - self.spatial_dim) + tuple(weights.shape)
        return coefficients.square() * weights.reshape(view_shape)

    def forward(self, residual: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        values = self.energy(residual).sum(dim=tuple(range(residual.ndim - self.spatial_dim, residual.ndim)))
        if reduction == "sum":
            return values.sum()
        if reduction == "mean":
            return values.mean()
        if reduction == "none":
            return values
        raise ValueError("reduction must be 'sum', 'mean', or 'none'")


__all__ = ["FourierDualLoss", "dct", "dst"]
