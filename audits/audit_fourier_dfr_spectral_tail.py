#!/usr/bin/env python3
"""Read-only spectral-tail audit for the 1D Fourier-DFR pilot."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from src.common.train_fourier_dfr_1d_lbfgs import (
    PI,
    TrialNet,
    derivative,
    diagnostics,
    fourier_data,
    gauss_rule,
)


BANDS = ((1, 8), (9, 16), (17, 32), (33, 64), (65, 128))


def moments(
    model: TrialNet,
    order: int,
    mode_count: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinates, weights = gauss_rule(order, device)
    x = coordinates.detach().clone().requires_grad_(True)
    phi, dphi, spectral_weights = fourier_data(x, mode_count)
    u = model(x)
    du = derivative(u, x)
    forcing = 4.0 * torch.sin(2.0 * x)
    residual_moments = (
        weights.T * (du.T * dphi - forcing.T * phi)
    ).sum(dim=1)

    exact_derivative = 2.0 * torch.cos(2.0 * x)
    exact_moments = (
        weights.T * (exact_derivative.T * dphi - forcing.T * phi)
    ).sum(dim=1)
    return residual_moments.detach(), exact_moments.detach(), spectral_weights


def summarize(
    values: torch.Tensor,
    spectral_weights: torch.Tensor,
    trained_modes: int,
) -> dict[str, float]:
    weighted = spectral_weights * values.square()
    result: dict[str, float] = {
        "total": float(weighted.sum()),
        "max_abs_moment": float(values.abs().max()),
    }
    for first, last in BANDS:
        result[f"band_{first}_{last}"] = float(weighted[first - 1:last].sum())
    result["trained"] = float(weighted[:trained_modes].sum())
    result["tail"] = float(weighted[trained_modes:].sum())
    result["tail_fraction"] = result["tail"] / max(result["total"], 1.0e-300)
    return result


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--order", type=int, default=512)
    parser.add_argument("--refined-order", type=int, default=1024)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
    if args.order <= 256 or args.refined_order <= args.order:
        raise ValueError("Use order > 256 and refined_order > order for 128 modes")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    torch.set_default_dtype(torch.float64)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    trained_modes = int(config["modes"])
    if not 1 <= trained_modes < 128:
        raise ValueError("This audit requires 1 <= trained modes < 128")
    model = TrialNet(int(config["width"]), int(config["depth"])).to(args.device)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)

    print("=" * 96)
    print("READ-ONLY FOURIER-DFR SPECTRAL-TAIL AUDIT")
    print("=" * 96)
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Saved step: {state.get('step')}; trained modes: {config.get('modes')}")
    print("Audit modes: 1,...,128; no training, parameter updates, or file writes.\n")

    summaries: dict[int, dict[str, float]] = {}
    exact_scores: dict[int, float] = {}
    for order in (args.order, args.refined_order):
        residual, exact, weights = moments(model, order, 128, args.device)
        summary = summarize(residual, weights, trained_modes)
        summaries[order] = summary
        exact_scores[order] = float((weights * exact.square()).sum())

        print(f"QUADRATURE Q{order}")
        print(f"  total N128 score  : {summary['total']:.12e}")
        print(f"  trained N{trained_modes} score : {summary['trained']:.12e}")
        for first, last in BANDS:
            print(
                f"  band {first:3d}-{last:3d}       : "
                f"{summary[f'band_{first}_{last}']:.12e}"
            )
        print(
            f"  tail {trained_modes + 1}-128      : "
            f"{summary['tail']:.12e}"
        )
        print(f"  tail fraction    : {summary['tail_fraction']:.6e}")
        print(f"  exact N128 score : {exact_scores[order]:.3e}")
        print(f"  max_abs_moment   : {summary['max_abs_moment']:.6e}\n")

    coarse = summaries[args.order]
    refined = summaries[args.refined_order]
    total_gap = relative_difference(coarse["total"], refined["total"])
    tail_gap = relative_difference(coarse["tail"], refined["tail"])

    audit_rule = gauss_rule(args.refined_order, args.device)
    audit_basis = fourier_data(audit_rule[0], 128)
    error_metrics = diagnostics(model, audit_rule, audit_basis)

    print("=" * 96)
    print("DECISION DATA")
    print("=" * 96)
    print(f"Q{args.order}/Q{args.refined_order} total-score discrepancy: {total_gap:.3e}")
    print(f"Q{args.order}/Q{args.refined_order} tail-score discrepancy : {tail_gap:.3e}")
    print(f"Refined tail fraction                         : {refined['tail_fraction']:.6e}")
    print(f"absolute_L2                                  : {error_metrics['absolute_l2']:.10e}")
    print(f"H1_seminorm                                  : {error_metrics['h1_seminorm']:.10e}")
    print(f"H1                                            : {error_metrics['h1']:.10e}")

    finite = all(math.isfinite(value) for value in refined.values())
    passed = (
        finite
        and total_gap < 1.0e-8
        and tail_gap < 1.0e-8
        and exact_scores[args.refined_order] < 1.0e-18
    )
    print(f"\nDECISION: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)
    print("The reported tail is numerically resolved. Its size determines whether N=32 was sufficient.")
    print("SPECTRAL-TAIL AUDIT FINISHED.")


if __name__ == "__main__":
    main()
