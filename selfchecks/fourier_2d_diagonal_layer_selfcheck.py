#!/usr/bin/env python3
"""Read-only Fourier weak-form check for a smooth 2D diagonal layer."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch

from experiments.poisson2d.train_spectral_wan import laplacian
from src.methods.wan_spectral_loss import SpectralWANLoss


PI = math.pi
DTYPE = torch.float64
WEAK_ATOL = 1.0e-10
WEAK_RTOL = 5.0e-3
NORM_ATOL = 1.0e-12
NORM_RTOL = 1.0e-10


def exact_solution(xy: torch.Tensor, beta: float) -> torch.Tensor:
    """u*=x(pi-x)y(pi-y)tanh(beta(x+y-pi))."""
    x, y = xy[:, :1], xy[:, 1:2]
    return x * (PI - x) * y * (PI - y) * torch.tanh(beta * (x + y - PI))


def exact_gradient(xy: torch.Tensor, beta: float) -> torch.Tensor:
    x, y = xy[:, :1], xy[:, 1:2]
    ax, ay = x * (PI - x), y * (PI - y)
    dax, day = PI - 2.0 * x, PI - 2.0 * y
    t = torch.tanh(beta * (x + y - PI))
    sech2 = 1.0 - t.square()
    ux = dax * ay * t + beta * ax * ay * sech2
    uy = ax * day * t + beta * ax * ay * sech2
    return torch.cat((ux, uy), dim=1)


def forcing(xy: torch.Tensor, beta: float) -> torch.Tensor:
    """Analytic f=-Delta u*."""
    x, y = xy[:, :1], xy[:, 1:2]
    ax, ay = x * (PI - x), y * (PI - y)
    dax, day = PI - 2.0 * x, PI - 2.0 * y
    t = torch.tanh(beta * (x + y - PI))
    sech2 = 1.0 - t.square()
    return (2.0 * (ax + ay) * t
            - 2.0 * beta * (dax * ay + ax * day) * sech2
            + 4.0 * beta**2 * ax * ay * t * sech2)


def tensor_gauss_rule(order: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    axis = 0.5 * PI * (nodes + 1.0)
    axis_weights = 0.5 * PI * weights
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    ww = np.multiply.outer(axis_weights, axis_weights)
    points = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    return (torch.as_tensor(points, dtype=DTYPE, device=device),
            torch.as_tensor(ww.reshape(-1), dtype=DTYPE, device=device))


@dataclass
class VersionResult:
    maximum_absolute_moment: float
    raw_score: float
    dfr_score: float
    largest_mode: tuple[int, int]


def summarize(moments: torch.Tensor, modes: int, loss: SpectralWANLoss) -> VersionResult:
    flat = moments.reshape(-1)
    largest = int(torch.argmax(flat.abs()).item())
    weights = loss.weights(flat.dtype, flat.device).reshape(-1)
    return VersionResult(
        maximum_absolute_moment=float(flat.abs().max()),
        raw_score=float(flat.square().sum()),
        dfr_score=float((flat.square() * weights).sum()),
        largest_mode=(largest // modes + 1, largest % modes + 1),
    )


def weak_moments(order: int, modes: int, chunk_size: int, beta: float,
                 device: torch.device) -> tuple[dict[str, VersionResult], dict[str, float]]:
    points, quadrature_weights = tensor_gauss_rule(order, device)
    x, y = points[:, 0], points[:, 1]
    grad_u = exact_gradient(points, beta)
    source = forcing(points, beta)[:, 0]
    pairs = torch.cartesian_prod(
        torch.arange(1, modes + 1, dtype=DTYPE, device=device),
        torch.arange(1, modes + 1, dtype=DTYPE, device=device),
    )
    gradient_parts, source_parts = [], []
    normalizer = 2.0 / PI
    for start in range(0, pairs.shape[0], chunk_size):
        pair = pairs[start:start + chunk_size]
        k, ell = pair[:, :1], pair[:, 1:2]
        sin_x, sin_y = torch.sin(k * x), torch.sin(ell * y)
        phi = normalizer * sin_x * sin_y
        dphi_x = normalizer * k * torch.cos(k * x) * sin_y
        dphi_y = normalizer * ell * sin_x * torch.cos(ell * y)
        gradient_parts.append(((dphi_x * grad_u[:, 0] + dphi_y * grad_u[:, 1])
                               * quadrature_weights).sum(dim=1))
        source_parts.append((phi * source * quadrature_weights).sum(dim=1))
    gradient_term = torch.cat(gradient_parts)
    source_term = torch.cat(source_parts)
    moment_sets = {
        "correct": gradient_term - source_term,
        "source_omitted": gradient_term,
        "source_sign_reversed": gradient_term + source_term,
    }
    loss = SpectralWANLoss((modes, modes), spatial_dim=2,
                           boundary="dirichlet", domain=PI, shift=1.0).to(device)
    results = {name: summarize(value, modes, loss) for name, value in moment_sets.items()}
    diagnostics = {
        "points": float(points.shape[0]),
        "finite": float(all(torch.isfinite(value).all() for value in moment_sets.values())),
    }
    return results, diagnostics


def exact_norms(order: int, beta: float, device: torch.device) -> dict[str, float]:
    points, weights = tensor_gauss_rule(order, device)
    value = exact_solution(points, beta)[:, 0]
    gradient = exact_gradient(points, beta)
    l2_sq = (weights * value.square()).sum()
    seminorm_sq = (weights * gradient.square().sum(dim=1)).sum()
    return {
        "l2": float(torch.sqrt(l2_sq)),
        "h1_seminorm": float(torch.sqrt(seminorm_sq)),
        "h1": float(torch.sqrt(l2_sq + seminorm_sq)),
    }


def forcing_check(beta: float, device: torch.device) -> dict[str, float | bool]:
    # Fixed, non-symmetric points include samples close to and away from the layer.
    points = torch.tensor([
        [0.31, 0.47], [0.73, PI - 0.73], [1.11, PI - 1.11 + 0.013],
        [1.57, 1.57], [2.21, 0.88], [2.71, 2.43], [0.19, 2.94],
    ], dtype=DTYPE, device=device, requires_grad=True)
    analytic = forcing(points, beta)
    automatic = -laplacian(exact_solution(points, beta), points)
    maximum = float((analytic - automatic).abs().max())
    reference = float(automatic.abs().max())
    tolerance = 1.0e-10 + 1.0e-12 * reference
    return {"maximum_absolute_difference": maximum, "reference_maximum": reference,
            "tolerance": tolerance, "passed": maximum <= tolerance,
            "finite": bool(torch.isfinite(analytic).all() and torch.isfinite(automatic).all())}


def comparison(primary: float, refined: float, atol: float, rtol: float) -> dict[str, float | bool]:
    absolute = abs(primary - refined)
    relative = absolute / max(abs(refined), 1.0e-300)
    tolerance = atol + rtol * abs(refined)
    return {"absolute_difference": absolute, "relative_discrepancy": relative,
            "mixed_tolerance": tolerance, "passed": absolute <= tolerance}


def print_version(name: str, result: VersionResult) -> None:
    print(f"  {name}")
    print(f"    maximum_absolute_moment={result.maximum_absolute_moment:.17e}")
    print(f"    unweighted_raw_score={result.raw_score:.17e}")
    print(f"    weighted_DFR_score={result.dfr_score:.17e}")
    print(f"    largest_residual_mode={result.largest_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument("--refined-order", type=int, default=192)
    parser.add_argument("--mode-chunk-size", type=int, default=128)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if min(args.modes, args.quadrature_order, args.refined_order, args.mode_chunk_size) < 1:
        raise ValueError("modes, quadrature orders, and chunk size must be positive")
    torch.set_default_dtype(DTYPE)
    device = torch.device(args.device)

    print("=" * 108)
    print("2D NON-SEPARABLE DIAGONAL-LAYER FOURIER WEAK-FORM SELF-CHECK")
    print("=" * 108)
    print("Domain: Omega=(0,pi)^2; beta={}; modes=(k,l) in {{1,...,{}}}^2".format(args.beta, args.modes))
    print("Exact solution: x(pi-x)y(pi-y)tanh(beta(x+y-pi))")
    print("Normalized basis: phi_kl=(2/pi) sin(kx) sin(ly), ||phi_kl||_L2=1")
    print("Weak moment: R_kl=int_Omega grad(u*) dot grad(phi_kl) - int_Omega f phi_kl")
    print("Test norm: full H1; ||phi_kl||_H1^2=1+k^2+l^2")
    print("DFR score (squared dual norm): sum_kl |R_kl|^2/(1+k^2+l^2)")
    print("Moment integration: tensor-product Gauss-Legendre; explicit sine moments in mode chunks")
    print("Existing training projection note: uniform interior grid with tensorized orthogonal DST-II matrices")
    print(f"dtype={DTYPE}; device={device}; mode_chunk_size={args.mode_chunk_size}")
    print(f"weak refinement tolerance: atol={WEAK_ATOL:.1e}, rtol={WEAK_RTOL:.1e}")
    print(f"norm refinement tolerance: atol={NORM_ATOL:.1e}, rtol={NORM_RTOL:.1e}")
    print("No training, optimizer creation, parameter updates, checkpoints, or result-file writes.\n")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA_CHECK: FAIL (CUDA is unavailable; refusing to fall back to CPU)")
        print("\nDECISION: FAIL")
        print("training_tasks_started=0")
        print("optimizers_created=0")
        print("parameter_updates=0")
        print("checkpoints_written=0")
        raise RuntimeError("CUDA is unavailable; refusing to fall back to CPU")

    force = forcing_check(args.beta, device)
    print("FORCING INDEPENDENT CHECK (analytic formula versus autograd -Laplacian at fixed points)")
    for key, value in force.items():
        print(f"  {key}={value}")
    print()

    records: dict[int, dict[str, VersionResult]] = {}
    norm_records: dict[int, dict[str, float]] = {}
    finite = bool(force["finite"])
    for order in (args.quadrature_order, args.refined_order):
        results, diagnostics = weak_moments(order, args.modes, args.mode_chunk_size,
                                            args.beta, device)
        norms = exact_norms(order, args.beta, device)
        records[order], norm_records[order] = results, norms
        finite = finite and bool(diagnostics["finite"]) and all(math.isfinite(v) for v in norms.values())
        print("-" * 108)
        print(f"TENSOR GAUSS-LEGENDRE Q{order} x Q{order} ({int(diagnostics['points'])} points)")
        for name in ("correct", "source_omitted", "source_sign_reversed"):
            print_version(name, results[name])
        print("  exact_solution_norms")
        print(f"    L2={norms['l2']:.17e}")
        print(f"    H1_seminorm={norms['h1_seminorm']:.17e}")
        print(f"    H1={norms['h1']:.17e}")

    primary, refined = records[args.quadrature_order], records[args.refined_order]
    print("\n" + "=" * 108)
    print("PRIMARY/REFINED COMPARISONS")
    weak_checks = {}
    for name in ("correct", "source_omitted", "source_sign_reversed"):
        check = comparison(primary[name].dfr_score, refined[name].dfr_score,
                           WEAK_ATOL, WEAK_RTOL)
        weak_checks[name] = check
        print(f"  {name}_DFR_score: primary={primary[name].dfr_score:.17e} refined={refined[name].dfr_score:.17e}")
        print(f"    absolute_difference={check['absolute_difference']:.17e}")
        print(f"    relative_discrepancy={check['relative_discrepancy']:.17e}")
        print(f"    mixed_tolerance={check['mixed_tolerance']:.17e} passed={check['passed']}")
    max_check = comparison(primary["correct"].maximum_absolute_moment,
                           refined["correct"].maximum_absolute_moment,
                           WEAK_ATOL, WEAK_RTOL)
    print(f"  correct_maximum_absolute_moment: primary={primary['correct'].maximum_absolute_moment:.17e} "
          f"refined={refined['correct'].maximum_absolute_moment:.17e}")
    print(f"    absolute_difference={max_check['absolute_difference']:.17e}")
    print(f"    relative_discrepancy={max_check['relative_discrepancy']:.17e}")
    print(f"    mixed_tolerance={max_check['mixed_tolerance']:.17e} passed={max_check['passed']}")

    norm_checks = {}
    print("  exact norm refinement")
    for name in ("l2", "h1_seminorm", "h1"):
        check = comparison(norm_records[args.quadrature_order][name],
                           norm_records[args.refined_order][name], NORM_ATOL, NORM_RTOL)
        norm_checks[name] = check
        print(f"    {name}: primary={norm_records[args.quadrature_order][name]:.17e} "
              f"refined={norm_records[args.refined_order][name]:.17e}")
        print(f"      absolute_difference={check['absolute_difference']:.17e}")
        print(f"      relative_discrepancy={check['relative_discrepancy']:.17e}")
        print(f"      mixed_tolerance={check['mixed_tolerance']:.17e} passed={check['passed']}")

    correct = refined["correct"].dfr_score
    omitted = refined["source_omitted"].dfr_score
    reversed_score = refined["source_sign_reversed"].dfr_score
    conditions = {
        "all_outputs_finite": finite,
        "forcing_independent_check": bool(force["passed"]),
        "refined_correct_near_zero": correct <= 1.0e-20,
        "correct_refinement_mixed_check": bool(weak_checks["correct"]["passed"]),
        "correct_max_moment_refinement_mixed_check": bool(max_check["passed"]),
        "source_omission_detected": omitted >= 1.0e-3,
        "source_sign_reversal_detected": reversed_score >= 1.0e-3,
        "exact_norms_stable": all(bool(check["passed"]) for check in norm_checks.values()),
        "correct_significantly_below_omitted": correct <= 1.0e-10 * omitted,
        "correct_significantly_below_reversed": correct <= 1.0e-10 * reversed_score,
    }
    passed = all(conditions.values())
    print("\nDECISION CONDITIONS")
    for name, value in conditions.items():
        print(f"  {name}={value}")
    print(f"\nDECISION: {'PASS' if passed else 'FAIL'}")
    print("training_tasks_started=0")
    print("optimizers_created=0")
    print("parameter_updates=0")
    print("checkpoints_written=0")
    if not passed:
        print("The requested orders do not pass the fixed criteria; increase quadrature order before training.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
