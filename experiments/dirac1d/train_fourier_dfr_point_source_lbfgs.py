#!/usr/bin/env python3
"""Fourier-DFR + L-BFGS for a 1D Dirac point-source Poisson problem.

Objective classification: explicit weak moments, including the exact point load.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.common.train_fourier_dfr_1d_lbfgs import PI, TrialNet, derivative


SOURCE = 0.5 * PI


@dataclass(frozen=True)
class Config:
    seed: int
    modes: int
    quadrature_order_per_half: int
    audit_order_per_half: int
    steps: int
    inner: int
    width: int
    depth: int
    device: str


def interval_rule(order: int, left: float, right: float, device: str):
    points, weights = np.polynomial.legendre.leggauss(order)
    x = 0.5 * (right - left) * points + 0.5 * (right + left)
    w = 0.5 * (right - left) * weights
    return (
        torch.as_tensor(x[:, None], device=device),
        torch.as_tensor(w[:, None], device=device),
    )


def split_rule(order: int, device: str):
    left_x, left_w = interval_rule(order, 0.0, SOURCE, device)
    right_x, right_w = interval_rule(order, SOURCE, PI, device)
    return torch.cat((left_x, right_x)), torch.cat((left_w, right_w))


def basis_data(x: torch.Tensor, modes: int):
    k = torch.arange(1, modes + 1, device=x.device, dtype=x.dtype)[:, None]
    normalizer = math.sqrt(2.0 / PI)
    phi = normalizer * torch.sin(k * x.T)
    dphi = normalizer * k * torch.cos(k * x.T)
    spectral_weights = 1.0 / (1.0 + k[:, 0].square())
    source_values = normalizer * torch.sin(k[:, 0] * SOURCE)
    return phi, dphi, spectral_weights, source_values


def objective(model: TrialNet, rule, basis):
    coordinates, weights = rule
    x = coordinates.detach().clone().requires_grad_(True)
    _, dphi, spectral_weights, source_values = basis
    u = model(x)
    du = derivative(u, x)
    moments = (weights.T * du.T * dphi).sum(dim=1) - source_values
    loss = (spectral_weights * moments.square()).sum()
    return loss, moments


def exact_solution(x: torch.Tensor):
    value = torch.where(
        x < SOURCE,
        x * (PI - SOURCE) / PI,
        SOURCE * (PI - x) / PI,
    )
    gradient = torch.where(
        x < SOURCE,
        torch.full_like(x, (PI - SOURCE) / PI),
        torch.full_like(x, -SOURCE / PI),
    )
    return value, gradient


def diagnostics(model: TrialNet, rule, basis):
    coordinates, weights = rule
    x = coordinates.detach().clone().requires_grad_(True)
    u = model(x)
    du = derivative(u, x)
    exact, exact_gradient = exact_solution(x)
    l2_squared = (weights * (u - exact).square()).sum()
    h1_semi_squared = (weights * (du - exact_gradient).square()).sum()
    exact_l2_squared = (weights * exact.square()).sum()
    loss, moments = objective(model, rule, basis)
    return {
        "fourier_loss": float(loss.detach()),
        "max_abs_moment": float(moments.detach().abs().max()),
        "absolute_l2": float(torch.sqrt(l2_squared).detach()),
        "relative_l2": float(torch.sqrt(l2_squared / exact_l2_squared).detach()),
        "h1_seminorm": float(torch.sqrt(h1_semi_squared).detach()),
        "h1": float(torch.sqrt(l2_squared + h1_semi_squared).detach()),
    }


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--quadrature-order-per-half", type=int, default=256)
    parser.add_argument("--audit-order-per-half", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.quadrature_order_per_half <= 2 * args.modes:
        raise ValueError("Use quadrature order per half > 2*modes")
    if args.audit_order_per_half <= args.quadrature_order_per_half:
        raise ValueError("audit order must exceed training order")
    return Config(**vars(args))


def main() -> None:
    config = parse_args()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)

    model = TrialNet(config.width, config.depth).to(config.device)
    train_rule = split_rule(config.quadrature_order_per_half, config.device)
    audit_rule = split_rule(config.audit_order_per_half, config.device)
    train_basis = basis_data(train_rule[0], config.modes)
    audit_basis = basis_data(audit_rule[0], config.modes)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path("fourier_noninterface") / "runs" / f"dfr_point_source_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    torch.save(model.state_dict(), output / "initial_model.pt")

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=config.inner,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-14,
    )
    closure_calls = 0
    history = []
    start = time.perf_counter()

    print("=" * 100)
    print("FOURIER DFR 1D DIRAC POINT SOURCE + L-BFGS")
    print("=" * 100)
    print("Problem: -u''=delta_{pi/2}; exact solution is an H1 tent profile, not H2")
    print("Loss: sum_k |integral u' phi_k' - phi_k(pi/2)|^2/(1+k^2)")
    print("No adversarial test network and no interface decomposition")
    print(
        f"modes={config.modes} split_Q={config.quadrature_order_per_half}/"
        f"{config.audit_order_per_half} steps={config.steps} inner={config.inner}"
    )
    print(f"Output: {output.resolve()}\n", flush=True)

    def report(step: int):
        train = diagnostics(model, train_rule, train_basis)
        refined = diagnostics(model, audit_rule, audit_basis)
        absolute_gap = abs(train["fourier_loss"] - refined["fourier_loss"])
        relative_gap = absolute_gap / max(
            train["fourier_loss"], refined["fourier_loss"], 1.0e-30
        )
        record = {
            "step": step,
            "closure_calls": closure_calls,
            "seconds": time.perf_counter() - start,
            "quadrature_absolute_gap": absolute_gap,
            "quadrature_relative_gap": relative_gap,
            **{f"train_{key}": value for key, value in train.items()},
            **{f"audit_{key}": value for key, value in refined.items()},
        }
        history.append(record)
        print(
            f"step={step:4d} calls={closure_calls:5d} loss_Q={train['fourier_loss']:.6e} "
            f"refined={refined['fourier_loss']:.6e} gap={relative_gap:.3e}"
        )
        print(
            f"  absolute_L2={refined['absolute_l2']:.6e} "
            f"relative_L2={refined['relative_l2']:.6e} "
            f"H1_seminorm={refined['h1_seminorm']:.6e} H1={refined['h1']:.6e}",
            flush=True,
        )
        return record

    report(0)
    for step in range(1, config.steps + 1):

        def closure():
            nonlocal closure_calls
            closure_calls += 1
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective(model, train_rule, train_basis)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite point-source Fourier objective")
            loss.backward()
            return loss

        optimizer.step(closure)
        if step == 1 or step % 10 == 0 or step == config.steps:
            record = report(step)
            if (
                record["quadrature_relative_gap"] > 1.0e-6
                and record["quadrature_absolute_gap"] > 1.0e-12
            ):
                raise RuntimeError("Split Q/refined Fourier discrepancy is too large")

    final_metrics = diagnostics(model, audit_rule, audit_basis)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "step": config.steps,
            "closure_calls": closure_calls,
            "metrics": final_metrics,
            "status": "bounded_pilot_finished",
        },
        output / "final_checkpoint.pt",
    )
    (output / "history.json").write_text(json.dumps(history, indent=2))

    print("\n" + "=" * 100)
    print("POINT-SOURCE DFR FINISHED")
    print("=" * 100)
    print(f"closure_calls    : {closure_calls}")
    print(f"training_seconds : {time.perf_counter() - start:.6f}")
    for key, value in final_metrics.items():
        print(f"{key:16s}: {value:.10e}")
    print("This is a single-seed low-regularity pilot, not a superiority claim.")


if __name__ == "__main__":
    main()
