#!/usr/bin/env python3
"""Train a 1D non-interface Poisson problem with a Fourier dual residual.

This is a bounded correctness pilot for the advisor-requested Fourier test
space.  It uses a neural trial function, fixed sine test functions, the
H^{-1} spectral weights 1/(1+k^2), and L-BFGS.  There is no adversarial test
network and no interface term.

Objective classification: explicit weak moments ``int(u' phi'_k)-int(f phi_k)``.
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
from torch import nn


PI = math.pi


@dataclass(frozen=True)
class Config:
    seed: int
    modes: int
    quadrature_order: int
    audit_order: int
    steps: int
    inner: int
    width: int
    depth: int
    device: str


class TrialNet(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hard enforcement of u(0)=u(pi)=0.
        return x * (PI - x) * self.net(x)


def gauss_rule(order: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    points, weights = np.polynomial.legendre.leggauss(order)
    x = 0.5 * PI * (points + 1.0)
    w = 0.5 * PI * weights
    return (
        torch.as_tensor(x[:, None], device=device),
        torch.as_tensor(w[:, None], device=device),
    )


def derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]


def fourier_data(
    x: torch.Tensor, modes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = torch.arange(1, modes + 1, device=x.device, dtype=x.dtype)[:, None]
    normalizer = math.sqrt(2.0 / PI)
    phi = normalizer * torch.sin(k * x.T)
    dphi = normalizer * k * torch.cos(k * x.T)
    spectral_weights = 1.0 / (1.0 + k[:, 0].square())
    return phi, dphi, spectral_weights


def objective(
    model: TrialNet,
    rule: tuple[torch.Tensor, torch.Tensor],
    basis: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates, weights = rule
    x = coordinates.detach().clone().requires_grad_(True)
    phi, dphi, spectral_weights = basis
    u = model(x)
    du = derivative(u, x)
    forcing = 4.0 * torch.sin(2.0 * x)
    moments = (
        weights.T * (du.T * dphi - forcing.T * phi)
    ).sum(dim=1)
    loss = (spectral_weights * moments.square()).sum()
    return loss, moments


def diagnostics(
    model: TrialNet,
    rule: tuple[torch.Tensor, torch.Tensor],
    basis: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    coordinates, weights = rule
    x = coordinates.detach().clone().requires_grad_(True)
    u = model(x)
    du = derivative(u, x)
    exact = torch.sin(2.0 * x)
    exact_derivative = 2.0 * torch.cos(2.0 * x)
    l2_squared = (weights * (u - exact).square()).sum()
    h1_semi_squared = (weights * (du - exact_derivative).square()).sum()
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
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--quadrature-order", type=int, default=128)
    parser.add_argument("--audit-order", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.modes < 1 or args.quadrature_order <= 2 * args.modes:
        raise ValueError("Use quadrature_order > 2*modes to avoid the Q32 aliasing seen in the self-check")
    if args.audit_order <= args.quadrature_order:
        raise ValueError("audit_order must exceed quadrature_order")
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
    train_rule = gauss_rule(config.quadrature_order, config.device)
    audit_rule = gauss_rule(config.audit_order, config.device)
    train_basis = fourier_data(train_rule[0], config.modes)
    audit_basis = fourier_data(audit_rule[0], config.modes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path("fourier_noninterface") / "runs" / f"dfr_1d_{timestamp}"
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
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()

    print("=" * 96)
    print("FOURIER DFR 1D POISSON + L-BFGS PILOT")
    print("=" * 96)
    print("Problem: -u''=4 sin(2x) on (0,pi), u=0 on the boundary")
    print("Trial: hard-boundary neural network; test space: fixed normalized sine basis")
    print("Loss: sum_k |R_k|^2/(1+k^2); no test network and no interface")
    print(
        f"modes={config.modes} Q={config.quadrature_order}/{config.audit_order} "
        f"steps={config.steps} inner={config.inner} device={config.device}"
    )
    print(f"Output: {output.resolve()}\n", flush=True)

    def report(step: int) -> dict[str, float | int]:
        train = diagnostics(model, train_rule, train_basis)
        audit = diagnostics(model, audit_rule, audit_basis)
        gap = abs(train["fourier_loss"] - audit["fourier_loss"]) / max(
            train["fourier_loss"], audit["fourier_loss"], 1.0e-30
        )
        record: dict[str, float | int] = {
            "step": step,
            "closure_calls": closure_calls,
            "seconds": time.perf_counter() - start,
            **{f"train_{key}": value for key, value in train.items()},
            **{f"audit_{key}": value for key, value in audit.items()},
            "quadrature_gap": gap,
        }
        history.append(record)
        print(
            f"step={step:4d} calls={closure_calls:5d} "
            f"loss_Q={train['fourier_loss']:.6e} "
            f"loss_refined={audit['fourier_loss']:.6e} gap={gap:.3e}"
        )
        print(
            f"  absolute_L2={audit['absolute_l2']:.6e} "
            f"relative_L2={audit['relative_l2']:.6e} "
            f"H1_seminorm={audit['h1_seminorm']:.6e} H1={audit['h1']:.6e}",
            flush=True,
        )
        return record

    report(0)
    for step in range(1, config.steps + 1):

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            closure_calls += 1
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective(model, train_rule, train_basis)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Fourier objective")
            loss.backward()
            return loss

        optimizer.step(closure)
        if step == 1 or step % 10 == 0 or step == config.steps:
            record = report(step)
            if record["quadrature_gap"] > 1.0e-8:
                raise RuntimeError("Q/refined Fourier loss discrepancy exceeds 1e-8")

    final = diagnostics(model, audit_rule, audit_basis)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "step": config.steps,
            "closure_calls": closure_calls,
            "metrics": final,
            "status": "bounded_pilot_finished",
        },
        output / "final_checkpoint.pt",
    )
    (output / "history.json").write_text(json.dumps(history, indent=2))

    print("\n" + "=" * 96)
    print("PILOT FINISHED")
    print("=" * 96)
    print(f"closure_calls : {closure_calls}")
    print(f"training_seconds: {time.perf_counter() - start:.6f}")
    for key, value in final.items():
        print(f"{key:16s}: {value:.10e}")
    print("This is a smooth-problem correctness pilot, not an improvement or publication claim.")


if __name__ == "__main__":
    main()
