#!/usr/bin/env python3
"""Classical parameterized adversarial WAN for the 2D Poisson problem.

Objective classification: learned-neural-test weak moment.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from experiments.poisson2d.train_fourier_dfr_2d_lbfgs import PI, TrialNet2D, error_metrics, exact_solution, laplacian, make_grid


EPS = 1.0e-30


def set_trainable(model: torch.nn.Module, value: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(value)


def gradients(model: TrialNet2D, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    points = xy.detach().clone().requires_grad_(True)
    value = model(points)
    gradient = torch.autograd.grad(value.sum(), points, create_graph=True)[0]
    return value, gradient


def weak_score(trial: TrialNet2D, test: TrialNet2D, xy: torch.Tensor,
               forcing: torch.Tensor, grid_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, du = gradients(trial, xy)
    v, dv = gradients(test, xy)
    area_weight = (PI / (grid_size - 1)) ** 2
    moment = area_weight * (du * dv - forcing * v).sum()
    norm2 = area_weight * (v.square() + dv.square().sum(dim=1, keepdim=True)).sum()
    return moment.square() / (norm2 + EPS), moment, norm2


def lbfgs(parameters, inner: int) -> torch.optim.LBFGS:
    return torch.optim.LBFGS(parameters, lr=1.0, max_iter=inner, history_size=50,
                             line_search_fn="strong_wolfe", tolerance_grad=1e-10,
                             tolerance_change=1e-14)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--inner-u", type=int, default=5)
    parser.add_argument("--inner-v", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--audit-size", type=int, default=257)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-seed", type=int, default=3102)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
    if args.rounds < 1 or args.inner_u < 1 or args.inner_v < 1 or args.grid_size < 4:
        raise ValueError("rounds, inner steps, and grid-size must be positive")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    trial = TrialNet2D(args.width, args.depth).to(args.device)
    torch.manual_seed(args.test_seed)
    test = TrialNet2D(args.width, args.depth).to(args.device)
    xy = make_grid(args.grid_size, args.device)
    source_xy = xy.detach().clone().requires_grad_(True)
    forcing = -laplacian(exact_solution(source_xy), source_xy).detach()

    start = time.perf_counter()
    calls_u = calls_v = 0
    set_trainable(trial, False)
    set_trainable(test, True)
    for round_number in range(1, args.rounds + 1):
        optimizer_v = lbfgs(test.parameters(), args.inner_v)

        def closure_v() -> torch.Tensor:
            nonlocal calls_v
            calls_v += 1
            optimizer_v.zero_grad(set_to_none=True)
            score, _, _ = weak_score(trial, test, xy, forcing, args.grid_size)
            objective = -torch.log(score + EPS)
            objective.backward()
            return objective

        optimizer_v.step(closure_v)
        set_trainable(test, False)
        set_trainable(trial, True)
        optimizer_u = lbfgs(trial.parameters(), args.inner_u)

        def closure_u() -> torch.Tensor:
            nonlocal calls_u
            calls_u += 1
            optimizer_u.zero_grad(set_to_none=True)
            score, _, _ = weak_score(trial, test, xy, forcing, args.grid_size)
            score.backward()
            return score

        optimizer_u.step(closure_u)
        set_trainable(trial, False)
        set_trainable(test, True)
        if round_number == 1 or round_number % 10 == 0 or round_number == args.rounds:
            score = float(weak_score(trial, test, xy, forcing, args.grid_size)[0].detach())
            print(f"round={round_number:4d} score={score:.6e} calls_u/v={calls_u}/{calls_v}", flush=True)

    elapsed = time.perf_counter() - start
    set_trainable(trial, True)
    final = error_metrics(trial, args.audit_size, args.device, 0.0)
    print("=" * 88)
    print("ORIGINAL PARAMETERIZED WAN 2D POISSON + ALTERNATING L-BFGS")
    print("=" * 88)
    print("Problem: -Delta u = 2 sin(x)sin(y), u|boundary=0")
    print(f"grid / audit          : {args.grid_size}^2 / {args.audit_size}^2")
    print(f"rounds, inner u/v     : {args.rounds}, {args.inner_u}/{args.inner_v}")
    print(f"Relative L2           : {final['relative_l2']:.10e}")
    print(f"H1                    : {final['h1']:.10e}")
    print(f"training time (s)     : {elapsed:.6f}")


if __name__ == "__main__":
    main()
