#!/usr/bin/env python3
"""Read-only spectral-tail audit for point-source Fourier-DFR.

For each requested cutoff ``N``, this reports
``sum_{k>N} |R_hat(k)|^2/lambda_k`` relative to the resolved maximum-mode
energy.  The default cutoffs make it easy to compare N=16,...,512.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from src.common.train_fourier_dfr_1d_lbfgs import TrialNet, derivative
from experiments.dirac1d.train_fourier_dfr_point_source_lbfgs import (
    SOURCE,
    basis_data,
    diagnostics,
    exact_solution,
    split_rule,
)


BANDS = ((1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 512), (513, 1024))


def residual_moments(model: TrialNet, order: int, modes: int, device: str):
    coordinates, weights = split_rule(order, device)
    x = coordinates.detach().clone().requires_grad_(True)
    _, dphi, spectral_weights, source_values = basis_data(x, modes)
    u = model(x)
    du = derivative(u, x)
    residual = (weights.T * du.T * dphi).sum(dim=1) - source_values

    _, exact_gradient = exact_solution(x)
    exact = (weights.T * exact_gradient.T * dphi).sum(dim=1) - source_values
    return residual.detach(), exact.detach(), spectral_weights, (coordinates, weights)


def summarize(values, weights, cutoffs):
    weighted = weights * values.square()
    max_modes = values.numel()
    total = float(weighted.sum())
    result = {
        "total": total,
        "max_abs_moment": float(values.abs().max()),
    }
    for first, last in BANDS:
        if first <= max_modes:
            result[f"band_{first}_{last}"] = float(weighted[first - 1:min(last, max_modes)].sum())
    for cutoff in cutoffs:
        retained = float(weighted[:cutoff].sum())
        tail = float(weighted[cutoff:].sum())
        result[f"N{cutoff}_retained"] = retained
        result[f"N{cutoff}_tail"] = tail
        result[f"N{cutoff}_tail_fraction"] = tail / max(total, 1.0e-300)
    return result


def relative_difference(a: float, b: float):
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--order-per-half", type=int, default=2048)
    parser.add_argument("--refined-order-per-half", type=int, default=4096)
    parser.add_argument("--max-modes", type=int, default=512,
                        help="largest resolved Fourier mode (default: 512)")
    parser.add_argument("--cutoffs", default="16,32,64,128,256",
                        help="comma-separated truncation orders N")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.order_per_half <= args.max_modes * 2 or args.refined_order_per_half <= args.order_per_half:
        raise ValueError("Use quadrature order > 2*max_modes and refined order > order")
    try:
        cutoffs = sorted({int(value) for value in args.cutoffs.split(",") if value.strip()})
    except ValueError as exc:
        raise ValueError("--cutoffs must be comma-separated integers") from exc
    if not cutoffs or cutoffs[0] < 1 or cutoffs[-1] >= args.max_modes:
        raise ValueError("Cutoffs must satisfy 1 <= N < max-modes")

    torch.set_default_dtype(torch.float64)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    trained_modes = int(config["modes"])
    model = TrialNet(int(config["width"]), int(config["depth"])).to(args.device)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)

    print("=" * 100)
    print("READ-ONLY POINT-SOURCE FOURIER-DFR SPECTRAL-TAIL AUDIT")
    print("=" * 100)
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Saved step: {state.get('step')}; trained modes: {trained_modes}")
    print(f"Audit modes: 1,...,{args.max_modes}; cutoffs: {cutoffs}; split quadrature at pi/2; no writes.\n")

    summaries = {}
    exact_scores = {}
    final_rule = None
    for order in (args.order_per_half, args.refined_order_per_half):
        residual, exact, spectral_weights, rule = residual_moments(
            model, order, args.max_modes, args.device
        )
        summary = summarize(residual, spectral_weights, cutoffs)
        summaries[order] = summary
        exact_scores[order] = float((spectral_weights * exact.square()).sum())
        final_rule = rule

        print(f"SPLIT QUADRATURE Q{order}+Q{order}")
        print(f"  total N{args.max_modes} score       : {summary['total']:.12e}")
        for cutoff in cutoffs:
            print(f"  N{cutoff:4d} retained energy  : {summary[f'N{cutoff}_retained']:.12e}")
            print(f"  N{cutoff:4d} tail energy      : {summary[f'N{cutoff}_tail']:.12e}")
            print(f"  N{cutoff:4d} tail fraction   : {summary[f'N{cutoff}_tail_fraction']:.6e}")
        for first, last in BANDS:
            key = f"band_{first}_{last}"
            if key in summary:
                print(f"  band {first:4d}-{last:4d}          : {summary[key]:.12e}")
        print(f"  exact N{args.max_modes} score : {exact_scores[order]:.3e}")
        print(f"  max_abs_moment         : {summary['max_abs_moment']:.6e}\n")

    coarse = summaries[args.order_per_half]
    refined = summaries[args.refined_order_per_half]
    total_gap = relative_difference(coarse["total"], refined["total"])
    tail_gap = max(
        relative_difference(coarse[f"N{cutoff}_tail"], refined[f"N{cutoff}_tail"])
        for cutoff in cutoffs
    )

    assert final_rule is not None
    final_basis = basis_data(final_rule[0], args.max_modes)
    errors = diagnostics(model, final_rule, final_basis)

    print("=" * 100)
    print("DECISION DATA")
    print("=" * 100)
    print(f"Total-score discrepancy : {total_gap:.3e}")
    print(f"Tail-score discrepancy  : {tail_gap:.3e}")
    for cutoff in cutoffs:
        print(f"N{cutoff:4d} refined tail fraction : {refined[f'N{cutoff}_tail_fraction']:.6e}")
    print(f"absolute_L2             : {errors['absolute_l2']:.10e}")
    print(f"H1                      : {errors['h1']:.10e}")

    finite = all(math.isfinite(v) for v in refined.values())
    passed = (
        finite
        and total_gap < 1.0e-6
        and tail_gap < 1.0e-6
        and exact_scores[args.refined_order_per_half] < 1.0e-16
    )
    print(f"\nDECISION: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)
    print("The reported tails are numerically resolved; their fractions quantify truncation error.")
    print("POINT-SOURCE SPECTRAL-TAIL AUDIT FINISHED.")


if __name__ == "__main__":
    main()
