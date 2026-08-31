#!/usr/bin/env python3
"""Unified Spectral-WAN trainer and independent error/spectral audit.

Objective classification: non-Dirac problems use a strong residual DST-II
projection; the Dirac branch uses explicit weak moments with the point load.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from src.methods.wan_spectral_loss import SpectralWANLoss

PI = math.pi


class TrialNet(nn.Module):
    def __init__(self, dimensions: int, width: int = 48, depth: int = 3, boundary_distance: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dimensions, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, 1))
        self.net, self.dimensions, self.boundary_distance = nn.Sequential(*layers), dimensions, boundary_distance

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        value = self.net(coordinates)
        if not self.boundary_distance:
            return value
        distance = torch.ones_like(value)
        for axis in range(self.dimensions):
            coordinate = coordinates[:, axis:axis + 1]
            distance = distance * coordinate * (PI - coordinate)
        return distance * value


def exact(x: torch.Tensor, problem: str) -> torch.Tensor:
    if problem == "poisson2d":
        return torch.sin(x[:, :1]) * torch.sin(x[:, 1:2])
    if problem == "large_gradient1d":
        return x * (PI - x) * torch.tanh(20.0 * (x - PI / 2))
    return torch.where(x < PI / 2, 0.5 * x, 0.5 * (PI - x))


def laplacian(value, coordinates):
    gradient = torch.autograd.grad(value.sum(), coordinates, create_graph=True)[0]
    result = torch.zeros_like(value)
    for axis in range(coordinates.shape[1]):
        second = torch.autograd.grad(gradient[:, axis].sum(), coordinates, create_graph=True)[0][:, axis:axis + 1]
        result = result + second
    return result


def grid(size: int, dimensions: int, device: str) -> torch.Tensor:
    axis = torch.linspace(0, PI, size, device=device)[1:-1]
    if dimensions == 1:
        return axis[:, None]
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), dim=1)


def dirac_weak_loss(model, points: torch.Tensor, modes: int, source: float = PI / 2):
    """Weak sine moments for -u''=delta_source, including exact point value."""
    x = points.detach().clone().requires_grad_(True)
    u = model(x)
    du = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 0]
    n = x.shape[0]
    k = torch.arange(1, modes + 1, dtype=x.dtype, device=x.device)[:, None]
    phi_prime = math.sqrt(2.0 / PI) * k * torch.cos(k * x[:, 0][None, :])
    moments = (PI / (n + 1)) * (phi_prime * du[None, :]).sum(dim=1)
    source_values = math.sqrt(2.0 / PI) * torch.sin(k[:, 0] * source)
    return moments - source_values


def audit(model, size: int, dimensions: int, problem: str, device: str):
    coordinates = grid(size, dimensions, device).requires_grad_(True)
    prediction = model(coordinates)
    target = exact(coordinates, problem)
    gradient = torch.autograd.grad(prediction.sum(), coordinates)[0]
    target_gradient = torch.autograd.grad(target.sum(), coordinates)[0]
    weight = (PI / (size - 1)) ** dimensions
    l2 = weight * (prediction - target).square().sum()
    ref = weight * target.square().sum()
    h1 = l2 + weight * (gradient - target_gradient).square().sum()
    return {"relative_l2": float(torch.sqrt(l2 / ref)), "h1": float(torch.sqrt(h1)), "absolute_l2": float(torch.sqrt(l2))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("dirac1d", "large_gradient1d", "poisson2d"), default="poisson2d")
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--audit-size", type=int, default=257)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--optimizer", choices=("adam", "lbfgs"), default="lbfgs")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=Path("spectral_wan_result.json"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    dimensions = 2 if args.problem == "poisson2d" else 1
    model = TrialNet(dimensions, args.width, args.depth).to(args.device)
    coordinates = grid(args.grid_size, dimensions, args.device)
    source = coordinates.detach().clone().requires_grad_(True)
    target = exact(source, args.problem)
    if args.problem != "dirac1d":
        forcing = -laplacian(target, source).detach()
    loss_fn = SpectralWANLoss(args.modes if dimensions == 1 else (args.modes, args.modes), dimensions,
                              boundary="dirichlet", domain=PI)
    optimizer = (torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=args.inner,
                                   history_size=50, line_search_fn="strong_wolfe",
                                   tolerance_grad=1e-10, tolerance_change=1e-14)
                 if args.optimizer == "lbfgs" else torch.optim.Adam(model.parameters(), lr=args.lr))
    start = time.perf_counter(); calls = 0
    def closure():
        nonlocal calls
        calls += 1; optimizer.zero_grad(set_to_none=True)
        points = coordinates.detach().clone().requires_grad_(True)
        if args.problem == "dirac1d":
            loss = loss_fn(dirac_weak_loss(model, points, args.modes))
        else:
            residual = -laplacian(model(points), points) - forcing
            shape = (args.grid_size - 2,) * dimensions
            loss = loss_fn.from_samples(residual.reshape(shape))
        loss.backward(); return loss
    for _ in range(args.steps if args.optimizer == "lbfgs" else args.steps):
        if args.optimizer == "lbfgs": optimizer.step(closure)
        else: closure(); optimizer.step()
    elapsed = time.perf_counter() - start
    audited = audit(model, args.audit_size, dimensions, args.problem, args.device)
    result = {"problem": args.problem, "modes": args.modes, "relative_l2": audited["relative_l2"],
              "h1": audited["h1"], "training_seconds": elapsed, "calls": calls}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
