#!/usr/bin/env python3
"""Explicit WAN weak moments with a fixed Fourier test space.
Only first spatial derivatives of the trial network are used.
The inner supremum over the truncated orthogonal test space is evaluated by the DFR weighted dual norm.
No neural test network is trained.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.methods.fourier_weak_wan_loss import weak_fourier_loss_2d
from experiments.poisson2d.train_spectral_wan import TrialNet


PI = math.pi


def exact_solution(xy: torch.Tensor) -> torch.Tensor:
    return torch.sin(xy[:, :1]) * torch.sin(xy[:, 1:2])


def forcing(xy: torch.Tensor) -> torch.Tensor:
    return 2.0 * exact_solution(xy)


def tensor_gauss_rule(order: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    axis = 0.5 * PI * (nodes + 1.0)
    axis_weights = 0.5 * PI * weights
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    ww = np.multiply.outer(axis_weights, axis_weights)
    points = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    return (torch.as_tensor(points, dtype=torch.float64, device=device),
            torch.as_tensor(ww.reshape(-1, 1), dtype=torch.float64, device=device))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--inner", type=int, default=100)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("fourier_weak_wan_poisson2d_run"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to fall back to CPU")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    model = TrialNet(2, args.width, args.depth, boundary_distance=True).to(args.device)
    points, weights = tensor_gauss_rule(args.quadrature_order, args.device)
    source = forcing(points)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=args.inner,
                                  history_size=50, line_search_fn="strong_wolfe",
                                  tolerance_grad=1.0e-10, tolerance_change=1.0e-14)
    calls = 0
    started = time.perf_counter()
    def closure() -> torch.Tensor:
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        output = weak_fourier_loss_2d(model, points, weights, source, args.modes,
                                      (args.quadrature_order, args.quadrature_order))
        output.loss.backward()
        return output.loss
    for _ in range(args.steps):
        optimizer.step(closure)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"method": "fourier_weak_wan", "problem": "poisson2d",
              "seed": args.seed, "closure_calls": calls,
              "wall_clock_seconds": time.perf_counter() - started}
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    (args.output_dir / "history.json").write_text(json.dumps([result], indent=2))
    torch.save({"trial": model.state_dict(), "result": result}, args.output_dir / "final_checkpoint.pt")


if __name__ == "__main__":
    main()
