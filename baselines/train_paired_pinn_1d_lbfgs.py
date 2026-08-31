#!/usr/bin/env python3
"""Paired strong-PINN baseline for the 1D Fourier-DFR experiment.

The trial network is loaded from the exact initial checkpoint used by DFR.
It is trained with the strong Poisson residual and the same L-BFGS outer/inner
budget.  Fourier residuals are reported only as endpoint diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch

from src.common.train_fourier_dfr_1d_lbfgs import (
    TrialNet,
    derivative,
    diagnostics,
    fourier_data,
    gauss_rule,
)


def strong_objective(
    model: TrialNet,
    rule: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    coordinates, weights = rule
    x = coordinates.detach().clone().requires_grad_(True)
    u = model(x)
    du = derivative(u, x)
    ddu = derivative(du, x)
    residual = -ddu - 4.0 * torch.sin(2.0 * x)
    return (weights * residual.square()).sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_model", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--quadrature-order", type=int, default=256)
    parser.add_argument("--audit-order", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
    if not args.initial_model.is_file():
        raise FileNotFoundError(args.initial_model)
    if args.audit_order <= args.quadrature_order:
        raise ValueError("audit_order must exceed quadrature_order")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    model = TrialNet(args.width, args.depth).to(args.device)
    initial_state = torch.load(
        args.initial_model, map_location="cpu", weights_only=True
    )
    model.load_state_dict(initial_state)

    train_rule = gauss_rule(args.quadrature_order, args.device)
    audit_rule = gauss_rule(args.audit_order, args.device)
    train_basis = fourier_data(train_rule[0], args.modes)
    audit_basis = fourier_data(audit_rule[0], args.modes)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path("fourier_noninterface") / "runs" / f"pinn_1d_paired_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "method": "strong_pinn",
        "seed": args.seed,
        "modes_for_diagnostics": args.modes,
        "quadrature_order": args.quadrature_order,
        "audit_order": args.audit_order,
        "steps": args.steps,
        "inner": args.inner,
        "width": args.width,
        "depth": args.depth,
        "device": args.device,
        "initial_model": str(args.initial_model.resolve()),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(model.state_dict(), output / "initial_model.pt")

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=args.inner,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-14,
    )
    closure_calls = 0
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()

    print("=" * 100)
    print("PAIRED STRONG PINN 1D POISSON + L-BFGS")
    print("=" * 100)
    print("Problem and hard-boundary trial network are identical to the Fourier-DFR pilot.")
    print("Training objective: integral |-u''-f|^2; Fourier loss is diagnostic only.")
    print(f"Initial model: {args.initial_model.resolve()}")
    print(
        f"Q={args.quadrature_order}/{args.audit_order} steps={args.steps} "
        f"inner={args.inner} device={args.device}"
    )
    print(f"Output: {output.resolve()}\n", flush=True)

    def report(step: int) -> dict[str, float | int]:
        strong_q = float(strong_objective(model, train_rule).detach())
        strong_refined = float(strong_objective(model, audit_rule).detach())
        gap = abs(strong_q - strong_refined) / max(
            strong_q, strong_refined, 1.0e-30
        )
        metrics = diagnostics(model, audit_rule, audit_basis)
        record: dict[str, float | int] = {
            "step": step,
            "closure_calls": closure_calls,
            "seconds": time.perf_counter() - start,
            "strong_q": strong_q,
            "strong_refined": strong_refined,
            "quadrature_gap": gap,
            **metrics,
        }
        history.append(record)
        print(
            f"step={step:4d} calls={closure_calls:5d} "
            f"strong_Q={strong_q:.6e} strong_refined={strong_refined:.6e} "
            f"gap={gap:.3e}"
        )
        print(
            f"  absolute_L2={metrics['absolute_l2']:.6e} "
            f"relative_L2={metrics['relative_l2']:.6e} "
            f"H1_seminorm={metrics['h1_seminorm']:.6e} H1={metrics['h1']:.6e}"
        )
        print(
            f"  Fourier_N{args.modes}_diagnostic={metrics['fourier_loss']:.6e}",
            flush=True,
        )
        return record

    report(0)
    for step in range(1, args.steps + 1):

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            closure_calls += 1
            optimizer.zero_grad(set_to_none=True)
            loss = strong_objective(model, train_rule)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite strong PINN objective")
            loss.backward()
            return loss

        optimizer.step(closure)
        if step == 1 or step % 10 == 0 or step == args.steps:
            record = report(step)
            absolute_gap = abs(
                float(record["strong_q"]) - float(record["strong_refined"])
            )
            if record["quadrature_gap"] > 1.0e-6 and absolute_gap > 1.0e-12:
                raise RuntimeError("Q/refined strong-residual discrepancy exceeds 1e-6")

    final = report(args.steps) if history[-1]["step"] != args.steps else history[-1]
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "step": args.steps,
            "closure_calls": closure_calls,
            "metrics": final,
            "status": "paired_pilot_finished",
        },
        output / "final_checkpoint.pt",
    )
    (output / "history.json").write_text(json.dumps(history, indent=2))

    print("\n" + "=" * 100)
    print("PAIRED PINN FINISHED")
    print("=" * 100)
    print(f"closure_calls    : {closure_calls}")
    print(f"training_seconds : {time.perf_counter() - start:.6f}")
    print(f"absolute_L2      : {float(final['absolute_l2']):.10e}")
    print(f"relative_L2      : {float(final['relative_l2']):.10e}")
    print(f"H1_seminorm      : {float(final['h1_seminorm']):.10e}")
    print(f"H1               : {float(final['h1']):.10e}")
    print("This is a paired smooth-problem baseline, not a superiority claim.")


if __name__ == "__main__":
    main()
