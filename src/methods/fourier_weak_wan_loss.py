"""Explicit weak Fourier-test WAN moments and DFR dual-norm evaluation.

Only first spatial derivatives of the trial function are evaluated here.
Fourier sine functions are fixed test functions; DFR is the weighted norm
evaluation over their finite-dimensional span.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


PI = math.pi


@dataclass
class WeakFourierOutput:
    loss: torch.Tensor
    moments: torch.Tensor
    weighted_contributions: torch.Tensor
    volume_moments: torch.Tensor
    forcing_moments: torch.Tensor
    maximum_absolute_moment: torch.Tensor
    largest_residual_mode: tuple[int, ...]
    finite: bool


def _trial_gradient(trial: Callable[[torch.Tensor], torch.Tensor],
                    points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = points.detach().clone().requires_grad_(True)
    values = trial(coordinates)
    gradient = torch.autograd.grad(values.sum(), coordinates, create_graph=True)[0]
    return coordinates, gradient


def _summarize(volume: torch.Tensor, source: torch.Tensor,
               weights: torch.Tensor) -> WeakFourierOutput:
    moments = volume - source
    contributions = moments.square() * weights
    flat = moments.reshape(-1)
    index = int(torch.argmax(flat.detach().abs()).item())
    if moments.ndim == 1:
        largest = (index + 1,)
    else:
        largest = (index // moments.shape[1] + 1, index % moments.shape[1] + 1)
    tensors = (volume, source, moments, contributions)
    return WeakFourierOutput(
        loss=contributions.sum(), moments=moments,
        weighted_contributions=contributions,
        volume_moments=volume, forcing_moments=source,
        maximum_absolute_moment=flat.abs().max(),
        largest_residual_mode=largest,
        finite=all(bool(torch.isfinite(item).all()) for item in tensors),
    )


def weak_fourier_moments_1d(
    trial: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    quadrature_weights: torch.Tensor,
    forcing_values: torch.Tensor,
    modes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(R, A, B)`` for normalized 1D sine test functions."""
    if points.ndim != 2 or points.shape[1] != 1:
        raise ValueError("1D points must have shape (Q, 1)")
    if modes < 1:
        raise ValueError("modes must be positive")
    x, gradient = _trial_gradient(trial, points)
    qweights = quadrature_weights.reshape(-1)
    source = forcing_values.reshape(-1)
    k = torch.arange(1, modes + 1, dtype=x.dtype, device=x.device)
    phi = math.sqrt(2.0 / PI) * torch.sin(x[:, 0, None] * k[None, :])
    dphi = math.sqrt(2.0 / PI) * torch.cos(x[:, 0, None] * k[None, :]) * k[None, :]
    volume = (qweights[:, None] * gradient[:, :1] * dphi).sum(dim=0)
    forcing_moments = (qweights[:, None] * source[:, None] * phi).sum(dim=0)
    return volume - forcing_moments, volume, forcing_moments


def weak_fourier_loss_1d(
    trial: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    quadrature_weights: torch.Tensor,
    forcing_values: torch.Tensor,
    modes: int,
) -> WeakFourierOutput:
    _, volume, source = weak_fourier_moments_1d(
        trial, points, quadrature_weights, forcing_values, modes
    )
    k = torch.arange(1, modes + 1, dtype=points.dtype, device=points.device)
    return _summarize(volume, source, 1.0 / (1.0 + k.square()))


def weak_fourier_moments_2d(
    trial: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    quadrature_weights: torch.Tensor,
    forcing_values: torch.Tensor,
    modes: int,
    grid_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tensorized 2D ``(R, A, B)`` without an ``N^2 x Q^2`` tensor."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("2D points must have shape (Qx*Qy, 2)")
    if modes < 1:
        raise ValueError("modes must be positive")
    if grid_shape is None:
        order = math.isqrt(points.shape[0])
        grid_shape = (order, order)
    qx, qy = grid_shape
    if qx * qy != points.shape[0]:
        raise ValueError("grid_shape does not match the number of points")
    coordinates, gradient = _trial_gradient(trial, points)
    x = coordinates[:, 0].reshape(qx, qy)[:, 0]
    y = coordinates[:, 1].reshape(qx, qy)[0, :]
    weight_matrix = quadrature_weights.reshape(qx, qy)
    # Tensor-product rules used by the repository have positive equal-axis weights.
    if qx != qy:
        raise ValueError("current implementation requires equal tensor-product orders")
    axis_weights = torch.sqrt(torch.diagonal(weight_matrix))
    source = forcing_values.reshape(qx, qy)
    gx = gradient[:, 0].reshape(qx, qy)
    gy = gradient[:, 1].reshape(qx, qy)
    k = torch.arange(1, modes + 1, dtype=points.dtype, device=points.device)
    sine_x = torch.sin(x[:, None] * k[None, :])
    sine_y = torch.sin(y[:, None] * k[None, :])
    cosine_x = torch.cos(x[:, None] * k[None, :]) * k[None, :]
    cosine_y = torch.cos(y[:, None] * k[None, :]) * k[None, :]
    sx, sy = axis_weights[:, None] * sine_x, axis_weights[:, None] * sine_y
    cx, cy = axis_weights[:, None] * cosine_x, axis_weights[:, None] * cosine_y
    normalization = 2.0 / PI
    volume = normalization * (cx.T @ gx @ sy + sx.T @ gy @ cy)
    forcing_moments = normalization * (sx.T @ source @ sy)
    return volume - forcing_moments, volume, forcing_moments


def weak_fourier_loss_2d(
    trial: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    quadrature_weights: torch.Tensor,
    forcing_values: torch.Tensor,
    modes: int,
    grid_shape: tuple[int, int] | None = None,
) -> WeakFourierOutput:
    _, volume, source = weak_fourier_moments_2d(
        trial, points, quadrature_weights, forcing_values, modes, grid_shape
    )
    k = torch.arange(1, modes + 1, dtype=points.dtype, device=points.device)
    weights = 1.0 / (1.0 + k[:, None].square() + k[None, :].square())
    return _summarize(volume, source, weights)


__all__ = [
    "WeakFourierOutput", "weak_fourier_moments_1d", "weak_fourier_loss_1d",
    "weak_fourier_moments_2d", "weak_fourier_loss_2d",
]
