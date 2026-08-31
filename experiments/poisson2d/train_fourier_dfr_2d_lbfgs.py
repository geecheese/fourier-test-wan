#!/usr/bin/env python3
"""2D Fourier dual-residual solver for ``-Delta u = f`` on ``[0,pi]^2``.

The trial network is multiplied by ``x(pi-x)y(pi-y)``.  The residual is
projected with an orthonormal 2D DST-II and minimized with the spectral
weights ``1/(1+lambda_mn)``.  The default exact solution is sin(x)sin(y),
whose forcing is ``2 sin(x)sin(y)``.

Objective classification: strong residual DST-II projection, not explicit weak moments.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch import nn

from src.methods.fourier_test_space import FourierDualLoss


PI = math.pi


class TrialNet2D(nn.Module):
    def __init__(self, width: int = 48, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, :1], xy[:, 1:2]
        return x * (PI - x) * y * (PI - y) * self.net(xy)


def dst2_matrix(size: int, modes: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Orthonormal DST-II matrix, with rows as grid points and columns modes."""
    if modes > size:
        raise ValueError("modes cannot exceed the grid size")
    j = torch.arange(size, dtype=dtype, device=device)[:, None]
    k = torch.arange(1, modes + 1, dtype=dtype, device=device)[None, :]
    matrix = math.sqrt(2.0 / size) * torch.sin(math.pi * (j + 0.5) * k / size)
    # The terminal DST-II mode has a different normalization.
    if modes == size:
        matrix[:, -1] *= 1.0 / math.sqrt(2.0)
    return matrix


class DST2DualLoss(FourierDualLoss):
    """FourierDualLoss using DST-II matrices on both spatial axes."""

    def __init__(self, modes: int, grid_size: int, domain: float = PI) -> None:
        super().__init__(modes=(modes, modes), spatial_dim=2,
                         boundary=("dirichlet", "dirichlet"), domain=(domain, domain),
                         norm="hminus1")
        self.grid_size = grid_size

    def transform(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 2 or residual.shape != (self.grid_size, self.grid_size):
            raise ValueError("residual must have shape (grid_size, grid_size)")
        matrix = dst2_matrix(self.grid_size, self.modes[0], residual.dtype, residual.device)
        return matrix.T @ residual @ matrix * (self.domain[0] * self.domain[1] / self.grid_size**2) ** 0.5


def exact_solution(xy: torch.Tensor, steepness: float = 0.0) -> torch.Tensor:
    x, y = xy[:, :1], xy[:, 1:2]
    if steepness <= 0:
        return torch.sin(x) * torch.sin(y)
    return torch.sin(x) * torch.sin(y) * (1.0 + 0.25 * torch.tanh(steepness * (x - PI / 2)))


def laplacian(value: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(value, xy, torch.ones_like(value), create_graph=True)[0]
    dxx = torch.autograd.grad(gradient[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
    dyy = torch.autograd.grad(gradient[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    return dxx + dyy


def make_grid(size: int, device: str) -> torch.Tensor:
    axis = torch.linspace(0.0, PI, size, device=device)[1:-1]
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def error_metrics(model: TrialNet2D, size: int, device: str, steepness: float) -> dict[str, float]:
    xy = make_grid(size, device).requires_grad_(True)
    prediction = model(xy)
    exact = exact_solution(xy, steepness)
    grad = torch.autograd.grad(prediction.sum(), xy)[0]
    exact_grad = torch.autograd.grad(exact.sum(), xy)[0]
    area_weight = (PI / (size - 1)) ** 2
    l2_sq = area_weight * (prediction - exact).square().sum()
    exact_l2_sq = area_weight * exact.square().sum()
    h1_sq = l2_sq + area_weight * (grad - exact_grad).square().sum()
    return {"relative_l2": float(torch.sqrt(l2_sq / exact_l2_sq)),
            "h1": float(torch.sqrt(h1_sq)), "absolute_l2": float(torch.sqrt(l2_sq))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, default=32, help="modes per axis (N x N)")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--audit-size", type=int, default=257)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lbfgs-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steepness", type=float, default=0.0,
                        help="optional steep x-transition; 0 uses sin(x)sin(y)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
    if args.modes < 1 or args.modes > args.grid_size - 2:
        raise ValueError("modes must be smaller than grid-size-1")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    model = TrialNet2D(args.width, args.depth).to(args.device)
    xy = make_grid(args.grid_size, args.device)
    source_xy = xy.detach().clone().requires_grad_(True)
    exact = exact_solution(source_xy, args.steepness)
    f = -laplacian(exact, source_xy).detach().reshape(-1)
    loss_fn = DST2DualLoss(args.modes, args.grid_size - 2)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0,
                                  max_iter=args.lbfgs_iterations, history_size=50,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-10,
                                  tolerance_change=1e-14)
    started = time.perf_counter()
    calls = 0

    def closure() -> torch.Tensor:
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        coordinates = xy.detach().clone().requires_grad_(True)
        residual = -laplacian(model(coordinates), coordinates).reshape(-1) - f
        residual = residual.reshape(args.grid_size - 2, args.grid_size - 2)
        value = loss_fn(residual)
        value.backward()
        return value

    optimizer.step(closure)
    elapsed = time.perf_counter() - started
    final = error_metrics(model, args.audit_size, args.device, args.steepness)
    print("=" * 88)
    print("2D FOURIER-DFR (DST-II) L-BFGS")
    print("=" * 88)
    print(f"modes                 : {args.modes} x {args.modes}")
    print(f"grid / audit          : {args.grid_size}^2 / {args.audit_size}^2")
    print(f"LBFGS iterations/calls: {args.lbfgs_iterations} / {calls}")
    print(f"Relative L2           : {final['relative_l2']:.10e}")
    print(f"H1                    : {final['h1']:.10e}")
    print(f"training time (s)     : {elapsed:.6f}")


if __name__ == "__main__":
    main()
